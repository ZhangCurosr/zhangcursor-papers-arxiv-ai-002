# REWARD-WEIGHTED PREFERENCE OPTIMIZATION FORSELECTIVE COMBINATORIAL OPTIMIZATION PROBLEMS

Haohui Zhang   
University of Twente   
Enschede   
Netherlands   
h.zhang-5@utwente.nl

Wouter van Heeswijk University of Twente Enschede Netherlands w.j.a.vanheeswijk@utwente.nl

Neil Yorke-Smith   
Delft University of Technology   
Delft   
Netherlands   
n.yorke-smith@tudelft.nl Xinyu Hu   
University of Bath Bath United Kingdom   
xh922@bath.ac.uk   
Martijn Mes   
University of Twente   
Enschede   
Netherlands   
m.r.k.mes@utwente.nl

## ABSTRACT

Reinforcement learning has emerged as an effective approach for combinatorial optimization, but most existing methods focus on problems such as the Travelling Salesperson Problem (TSP) in which all elements must be served. Selective combinatorial optimization problems (SCOPs), including the Orienteering Problem (OP), Prize-Collecting TSP (PCTSP), Stochastic PCTSP and Knapsack Problem (KP), additionally require selecting which elements to include. This selection creates sampled solutions with substantially different objective values, yet standard preference optimization reduces such comparisons to binary rankings and discards the magnitude of these differences.

We introduce Reward-Weighted Preference Optimization (RWPO), a pairwise training objective that incorporates reward-gap information into preference learning. For candidate trajectories sampled from the same instance, RWPO weights each preference loss according to the normalized difference in their rewards, assigning greater training emphasis to comparisons that represent larger improvements in solution quality. We further combine RWPO with scale-specific fine-tuning and reward- and likelihood-weighted tilted sampling to adapt trained policies to larger and shifted instance distributions without relying on local-search post-processing.

Experiments on the OP, PCTSP, Stochastic PCTSP, and KP show that RWPO consistently improves over unweighted preference optimization and policy-gradient baselines across a wide range of instance sizes. On the OP, the resulting solver exceeds the mean reward of the Compass metaheuristic at scales from 100 to 2,000 nodes. The results demonstrate that preserving reward-gap information can substantially improve preference-based learning for selective combinatorial optimization and support generalization to large-scale instances.

Keywords Combinatorial Optimization · Reinforcement Learning · Preference Optimization · Large-Scale Combinatorial Optimization · Vehicle Routing Problems · Machine Learning

## 1 Introduction

Many routing and resource allocation problems involve deciding which customers to serve or which items to select under resource constraints. These decisions are naturally formulated as selective combinatorial optimization problems (SCOPs), including the Orienteering Problem (OP), Prize-Collecting Travelling Salesperson Problem (PCTSP), Stochas tic PCTSP (SPCTSP), and the Knapsack Problem (KP) [Gunawan et al., 2016, Balas, 1989, Martello and Toth, 1990]. Unlike the canonical Travelling Salesperson Problem (TSP) and Capacitated Vehicle Routing Problem (CVRP), SCOP require choosing which elements to include in a feasible solution. In selective routing, this choice also affects route construction: visiting a customer changes both the objective value and the resources required to maintain feasibility. Sequential construction therefore produces candidate solutions with potentially large differences in objective value. Preference-based training can exploit these comparisons, but binary preference labels discard the magnitude of these quality differences.

![](images/f7751a11b93325671cd7cc67e8af722add66dab1d40c146c07e0d318f6485ae1.jpg)  
Figure 1: Overview of the RWPO training and adaptation pipeline. The distributions below are conceptual visualizations of probability mass over solutions (with empirical validation provided in Appendix E.2.1). Starting from a random initial policy, RWPO learns reward-aware preferences, scale-specific fine-tuning adapts the policy to different scales (N), and tilted sampling further refines the policy toward high-quality solutions. Yellow stars denote solution quality, and τ<sup>⋆</sup> marks the optimal solution.

Neural constructive solvers have emerged as effective approaches to combinatorial optimization, learning policies that construct solutions sequentially and enabling fast inference on new instances Bengio et al. [2021], Kool et al. [2019]. To tackle large-scale combinatorial optimization problems (COPs), recent efforts have introduced sophisticated scaling techniques such as divide-and-conquer strategies and advanced search guidance Cheng et al. [2023], Hou et al. [2023], Zheng et al. [2024], Ye et al. [2024], Pan et al. [2025b], Chen et al. [2026]. Although neural-based methods have been applied to selective problems such as OP, PCTSP, and KP Kool et al. [2019], Xin et al. [2021], Kim et al. [2022], Drakulic et al. [2023], Zheng et al. [2024], Ye et al. [2024], their performance at large and shifted instance scales remains challenging.

Policy-based Reinforcement Learning (RL) methods train such policies by directly optimizing expected rewards, but their stochastic gradient estimates can have high variance [Nazari et al., 2018, Hottung and Tierney, 2020, Kool et al., 2019, Kwon et al., 2020, Wu et al., 2021]. Preference Optimization (PO) offers an alternative by learning from comparisons between sampled solutions [Rafailov et al., 2023] and has been applied to TSP and CVRP [Pan et al., 2025a]. However, standard PO reduces each pair of solutions to a strictly binary preference: one solution is preferred to the other, regardless of whether their rewards differ marginally or substantially. This discards information that is directly available in combinatorial optimization, where exact objective values can be evaluated for every sampled solution. This limitation is particularly relevant to SCOPs, where different subset selections can produce widely varying solution quality. We therefore ask whether preference learning can retain its pairwise stability while exploiting the magnitude of observed reward differences.

To address these challenges, we introduce Reward-Weighted Preference Optimization (RWPO), for SCOP-solving RL agents. Rather than relying solely on binary preferences, RWPO directly embeds batch-normalized reward gaps as external weights on the pairwise log-likelihood differences. Unlike BOPO [Liao et al., 2025], which uses best-anchored comparisons and reward-dependent scaling inside the sigmoid, RWPO retains all sampled rollout pairs and applies reward-dependent weights to the loss while maintaining a shared inverse temperature parameter.

Second, we develop a confidence-regularized all-pair formulation that combines weighted-reward-gap preference fitting with a symmetric pairwise penalty over sampled rollout pairs. We use all sampled rollout pairs and regularize pairwise confidence, so that small reward differences do not induce unnecessarily strong separation while larger improvements receive stronger supervision.

Third, beyond the training objective, we develop a scale adaptation pipeline that combines RWPO with scale-specific fine-tuning and tilted sampling. These components enable transfer to larger and shifted instance distributions without retraining from scratch or relying on local-search (LS) post-processing during inference, providing a scalable approach for improving generalizability of neural solver across SCOPs.

Summarizing, our contributions are as follows: 1) We introduce RWPO for SCOPs, a preference learning objective that explicitly incorporates solution-quality differences by weighting pairwise comparisons using normalized reward gaps. 2) We develop a scalable training and adaptation pipeline combining RWPO, scale-specific fine-tuning, and tilted sampling. Across SCOP benchmarks, the resulting neural solver achieves state-of-the-art (SOTA) solution quality and generalizes robustly to larger instance sizes. 3) We evaluate solution quality and cross-scale generalization, separating the zero-shot performance of the trained policy from the gains obtained through adaptation. Finally, 4) we benchmark against standard preference and policy-gradient baselines, characterizing the learned policies through information-theoretic analyses.

Due to space limitations, we provide a more comprehensive discussion of related work in Appendix A.

## 2 Preliminaries

## 2.1 Constructive Neural Solvers

Constructive neural solvers generate a feasible solution sequentially by learning a stochastic policy over the next decision, typically trained via RL. Following the attention-based architecture of Kool et al. [2019], a problem instance $x \sim \mathcal { D }$ is first encoded into node embeddings using a transformer-style encoder with multi-head self-attention, which allows each node to aggregate information from all other nodes and capture the global structure of the instance. A decoder then constructs a solution autoregressively: at each step t, the policy $\pi _ { \theta } ( a _ { t } \mid s _ { t } , x )$ assigns probabilities to feasible actions, where the state $s _ { t }$ represents the partial solution constructed until step t and the action $a _ { t }$ corresponds to selecting the next node, item, or operation. Feasibility constraints are enforced through masking, so invalid actions receive zero probability during decoding. This sequential construction process can be naturally formulated as a Markov Decision Process (MDP), where the initial state is an empty solution, the transition appends the selected action to the partial solution, and the episode terminates when a complete solution is obtained or when no further feasible action can be selected. The final reward $r ( x , \tau )$ is computed from the completed trajectory $\tau = ( a _ { 1 } , \dots , a _ { T } )$ , such as the negative tour length or the collected prize. We employ this ‘standard’ constructive policy as our base architecture, focusing specifically on how to effectively train and scale neural solvers for SCOPs.

## 2.2 Reinforcement Learning for COPs

For solving COPs, a standard reinforcement learning (RL) objective is to maximize the expected reward

$$
\operatorname* { m a x } _ { \theta } \ \mathbb { E } _ { x \sim \mathcal { D } } \mathbb { E } _ { \tau \sim \pi _ { \theta } ( . | x ) } [ r ( x , \tau ) ] ,\tag{1}
$$

whose policy-gradient estimator takes the REINFORCE form

$$
\nabla _ { \theta } J ( \theta ) = \mathbb { E } _ { x \sim \mathcal { D } , \tau \sim \pi _ { \theta } ( \cdot \vert x ) } \left[ r ( x , \tau ) \nabla _ { \theta } \log \pi _ { \theta } ( \tau \mid x ) \right] .\tag{2}
$$

Although this objective is directly aligned with reward maximization, it often suffers from high gradient variance, poor trajectory-level credit assignment, leading to unstable policy updates and inefficient learning in large combinatorial decision spaces [Sutton et al., 2000, Greensmith et al., 2004, Schulman et al., 2015, Rafailov et al., 2023]. These issues become more pronounced in SCOPs, where feasible solutions may differ substantially in reward and informative high-quality trajectories are difficult to sample.

POMO for SCOPs. The RL-based framework POMO [Kwon et al., 2020] relies on the symmetry-equivalence of rollouts initialized from different starting nodes, a property inherent to sequence-invariant routing problems such as the TSP. This symmetry assumption is less appropriate for SCOPs, as the first selected node can affect the remaining feasible region, achievable reward, and final solution quality. Nevertheless, POMO-style multi-start decoding remains an effective strategy when reinterpreted as learning a top-k branch-selection policy [Zhang et al., 2025]. Specifically, for instance x, the model selects a set $\kappa _ { \theta } ( x )$ of promising first-step actions greedily according to the policy distribution and performs parallel rollouts conditioned on these starts. The resulting objective can be interpreted as maximizing the expected reward, conditioned on the trajectory’s initial action belonging to the selected top-k branches:

$$
\operatorname* { m a x } _ { \theta } \mathbb { E } _ { x \sim \mathcal { D } } \left[ \mathbb { E } _ { \tau \sim \pi _ { \theta } ( \cdot \vert x ) } \left[ r ( x , \tau ) \vert a _ { 1 } ( \tau ) \in \mathcal { K } _ { \theta } ( x ) \right] \right] ,\tag{3}
$$

where $a _ { 1 } ( \tau )$ denotes the first action of trajectory τ. Top-k branch decoding encourages exploration across promising initial actions, which is critical for SCOPs where early decisions strongly shape the remaining feasible region. However, the resulting branch heterogeneity makes reward-based policy gradients highly noisy. As detailed next, preference-based training mitigates this instability by comparing candidate branches within the same instance, yielding a much more stable learning signal.

We use POMO-style multi-start decoding to obtain multiple candidate trajectories for each instance. In SCOPs, these trajectories need not be equivalent because different initial selections can induce different feasible regions and rewards. This provides a natural within-instance comparison set for preference optimization.

## 2.3 Preference Optimization for COPs

Preference Optimization (PO) provides an alternative training paradigm for RL by replacing direct reward maximization with preference-based MLE [Rafailov et al., 2023, Pan et al., 2025a]. Its connection to RL can be motivated from the entropy-regularized objective. For a fixed instance $x ,$ consider

$$
\operatorname* { m a x } _ { \pi } \mathbb { E } _ { \tau \sim \pi ( \cdot \vert x ) } [ r ( x , \tau ) ] + \alpha \mathcal { H } ( \pi ( \cdot \mid x ) ) ,\tag{4}
$$

where $\alpha > 0$ controls the strength of entropy regularization. The latent reward can be reparameterized in relation to the corresponding policy $\pi ^ { \star } ( \tau | x )$ , which is written as

$$
\hat { r } ( x , \tau ) = \alpha \log \pi ^ { \star } ( \tau \mid x ) + \alpha \log Z ( x ) ,\tag{5}
$$

where $Z ( x )$ is the partition function over all feasible trajectories. For any pair $( \tau _ { i } , \tau _ { j } )$

$$
\hat { r } ( x , \tau _ { i } ) - \hat { r } ( x , \tau _ { j } ) = \alpha \left[ \log \pi ^ { \star } ( \tau _ { i } \mid x ) - \log \pi ^ { \star } ( \tau _ { j } \mid x ) \right] .\tag{6}
$$

This allows reward comparison to be transformed into preference classification over trajectory pairs. Given a pair of sampled trajectories $( \tau _ { i } , \tau _ { j } )$ , we define the binary preference label as

$$
y _ { i j } = \mathbb { I } \big ( r ( x , \tau _ { i } ) > r ( x , \tau _ { j } ) \big ) .\tag{7}
$$

Since the true optimal policy $\pi ^ { \star }$ is unknown, PO uses the parameterized policy $\pi _ { \theta }$ to model the preference probability. The resulting objective maximizes the expected log-likelihood of the observed preference pairs:

$$
\operatorname* { m a x } _ { \theta } \mathbb { E } _ { x \sim \mathcal { D } , ( \tau _ { i } , \tau _ { j } ) } \left[ y _ { i j } \log p _ { \theta } ( \tau _ { i } \succ \tau _ { j } \mid x ) + y _ { j i } \log p _ { \theta } ( \tau _ { j } \succ \tau _ { i } \mid x ) \right] .\tag{8}
$$

Here, we use $y _ { j i }$ rather than $1 - y _ { i j }$ in standard binary cross-entropy maximization because trajectory pairs with equal rewards satisfy $y _ { i j } = y _ { j i } = 0$ and should be excluded from preference learning.

Under a Bradley–Terry (BT) model, which represents pairwise preference probabilities as a sigmoid of the difference between two latent utilities, the probability that $\tau _ { i }$ is preferred to $\tau _ { j }$ is defined as

$$
p _ { \theta } ( \tau _ { i } \succ \tau _ { j } \mid x ) = \sigma \left( \alpha \left[ \log \pi _ { \theta } ( \tau _ { i } \mid x ) - \log \pi _ { \theta } ( \tau _ { j } \mid x ) \right] \right) ,\tag{9}
$$

where $\sigma ( \cdot )$ denotes the sigmoid function and the inverse-temperature parameter $\alpha > 0$ controls the preference sharpness.

In this way, PO converts reward-based policy learning into a pairwise MLE problem. Instead of estimating high-variance policy gradients from scalar trajectory rewards, the model is trained to assign higher likelihood to preferred trajectories than to inferior ones. Despite this stability, standard PO treats all correctly ordered pairs equally, stripping away the rich, exact reward differences that are readily available in combinatorial optimization.

For combinatorial optimization, however, the binary preference label is unnecessarily restrictive: the exact reward difference between two sampled trajectories is available at negligible additional cost. For instance, a pair whose rewards differ by 0.01 and a pair whose rewards differ by 10 receive the same preference signal under standard PO. RWPO addresses this mismatch by weighting the preference loss according to the normalized reward gap.

## 3 Methodology

## 3.1 Reward-Weighted Preference Optimization

To overcome the limitations of standard preference learning and explicitly leverage the exact reward signals available in SCOPs, we introduce the RWPO framework. The core idea is to dynamically scale the training signal based on the magnitude of the reward improvement, rather than treating all correctly ordered trajectory pairs equally. This explicitly accounts for the massive variance in subset-selection quality while retaining a shared inverse-temperature parameter across pairs.

Given an instance x and a pair of sampled trajectories $( \tau _ { i } , \tau _ { j } )$ , RWPO weights the preference log-likelihood by a reward-dependent coefficient. The RWPO objective is defined as:

$$
\operatorname* { m a x } _ { \theta } \mathbb { E } _ { x \sim \mathcal { D } , ( \tau _ { i } , \tau _ { j } ) } \left[ \omega ( x , \tau _ { i } , \tau _ { j } ) y _ { i j } \log p _ { \theta } ( \tau _ { i } \succ \tau _ { j } \mid x ) + \omega ( x , \tau _ { j } , \tau _ { i } ) y _ { j i } \log p _ { \theta } ( \tau _ { j } \succ \tau _ { i } \mid x ) \right] .\tag{10}
$$

where $\omega ( x , \tau _ { i } , \tau _ { j } )$ determines the importance of the preference pair.

We instantiate the weight using the normalized reward gap:

$$
\omega ( x , \tau _ { i } , \tau _ { j } ) : = 1 + \lambda \frac { \Delta ( x , \tau _ { i } , \tau _ { j } ) } { \bar { \Delta } ^ { ( b ) } + \epsilon } , \Delta ( x , \tau _ { i } , \tau _ { j } ) = r ( x , \tau _ { i } ) - r ( x , \tau _ { j } ) ,\tag{11}
$$

where $\bar { \Delta } ^ { ( b ) }$ denotes the average reward gap within the batch, $\lambda \geq 0$ controls the reward reweighting strength, and ϵ is a small constant for numerical stability. This normalization makes the reward weight comparable across instances and reduces sensitivity to the scale of the objective values. In contrast to standard PO, which assigns equal weight to all ordered preference pairs, RWPO amplifies comparisons with large reward gaps while relatively suppressing near ties.

Theoretical connection to policy gradients. At the level of pairwise update coefficients, this weighting mechanism provides a connection between standard PO and reward-weighted policy gradients. For a fixed instance x, both REINFORCE and BT preference updates can be decomposed into pairwise score directions of the form:

$$
\nabla _ { \theta } \log \pi _ { \theta } ( \tau _ { i } | x ) - \nabla _ { \theta } \log \pi _ { \theta } ( \tau _ { j } | x ) , \quad \mathrm { w h e r e } \ r ( x , \tau _ { i } ) - r ( x , \tau _ { j } ) > 0 .
$$

The fundamental difference lies in the scalar coefficient attached to each direction: policy gradients scale naturally with the reward gap $r ( x , \tau _ { i } ) - r ( x , \tau _ { j } )$ , whereas unweighted PO assigns the same explicit constant to every preferred pair. Under a bounded-gradient assumption, and when the pair-sampling distribution matches the current policy, the discrepancy between the RWPO preferred-pair update and the corresponding policy-gradient update is bounded by the mismatch between these effective scalar coefficients (see Appendix B.4 for the full proof). Consequently, incorporating reward-gap information provides a local motivation for making the preference-learning signal more closely resemble reward-weighted policy-gradient ascent, but it does not establish strict superiority, improved expected return, or convergence.

Confidence-regularized all-pair RWPO loss. To further stabilize training, we introduce a confidence-regularized all-pair formulation. For each unordered rollout pair $\{ \tau _ { i } , \tau _ { j } \}$ , both preference directions are evaluated, but the direction corresponding to the better trajectory receives an additional reward-gap-dependent weight. Specifically, we define the bidirectional weight as:

$$
\tilde { \omega } _ { i j } = 1 + \lambda \frac { \Delta ( x , \tau _ { i } , \tau _ { j } ) } { \bar { \Delta } ^ { ( b ) } + \epsilon } y _ { i j } .\tag{12}
$$

The implemented objective is

$$
\operatorname* { m a x } _ { \theta } \mathbb { E } _ { x \sim \mathcal { D } , ( \tau _ { i } , \tau _ { j } ) } \Big [ \tilde { \omega } _ { i j } \log p _ { \theta } ( \tau _ { i } \succ \tau _ { j } \mid x ) + \tilde { \omega } _ { j i } \log p _ { \theta } ( \tau _ { j } \succ \tau _ { i } \mid x ) \Big ] .\tag{13}
$$

This formulation differs from the strictly ordered RWPO objective (Eq. 11) by retaining the log-likelihoods for all rollout directions rather than discarding the non-preferred pairs. The constant unit baseline acts as a symmetric regularizer, providing a smoothing effect over the POMO rollout set. Simultaneously, the reward-gap-dependent term asymmetrically weights this all-pair objective toward the valid preference directions, proportionally scaling with the reward improvement. This design is particularly advantageous for POMO-style training, where multiple rollouts from the same instance provide a rich local comparison set: the symmetric all-pair component stabilizes preference learning, while the reward-weighted component prevents the loss function from treating near-ties and substantial reward improvements as equally important.

Corollary 1 (Reward-aware preference separation). Under the confidence-regularized all-pair RWPO objective, the induced optimal preference margin log $\pi _ { \theta } ( \tau _ { i } \mid x ) - \log \pi _ { \theta } ( \tau _ { j } \mid x )$ is monotonically increasing with the reward gap. Therefore, RWPO encourages stronger separation between trajectories with larger reward improvements while avoiding excessive separation for near-tie solutions.

The gradient of the two objectives are provided in Appendix B.3.

## 3.2 Training and Adaptation Pipeline for SCOPs

Building on the RWPO objective, we design a complete training and adaptation pipeline for SCOPs to enhance both in-distribution performance and zero-shot performance on much larger problem sizes. While a trained RWPO policy generalizes well, applying it to significantly larger target scales requires distribution-specific adaptation to account for expanding decision spaces. To achieve this efficiently, our pipeline (Algorithm 1) introduces two lightweight post-training stages: low-learning-rate scale-specific fine-tuning and tilted sampling adaptation. Crucially, this design explicitly bypasses computationally expensive local-search (LS) post-processing or LS-guided fine-tuning [Pan et al., 2025a]. Instead, our pipeline shifts the computational burden entirely to a one-time, offline adaptation phase, ensuring inference on new instances remains fast and search-free.

Scale-specific fine-tuning. A straightforward yet effective adaptation strategy is continued training with a reduced learning rate [Mosbach et al., 2020, Lin et al., 2026]. Empirically, it acts as conservative policy refinement: once RWPO has established a strong base policy, low-learning-rate adaptation further concentrates probability mass on high-quality actions without substantially altering the learned solution structure. While full backpropagation through the Transformer scales with instance size, this is strictly a one-time, offline sunk cost for a target distribution; a direct contrast to LS heuristics, which impose a substantial recurring computational penalty on every single instance during inference. To keep this offline adaptation efficient on massive instances, we purposefully employ small batch sizes, reduced multi-start sizes, and aggressive early stopping (e.g., 30 epochs).

Tilted sampling adaptation. To further concentrate probability mass on high-quality actions without destabilizing the underlying solution structure, we introduce a tilted sampling strategy for target-scale adaptation. Rather than relying on computationally expensive LS oracles to generate supervisory targets [Pan et al., 2025a], this method efficiently updates the policy using self-generated rollouts. Building on the KL-regularized policy optimization framework [Mittal et al., 2026], we define a reward-anchored sharpening objective that re-weights each rollout according to both its task reward and its likelihood under a frozen RWPO reference policy. Maximizing the weighted log-likelihood of these rollouts effectively performs weighted behavioral cloning, allowing us to exploit target-scale rewards while the frozen reference acts as an implicit KL-regularizer to preserve structural priors.

For each trajectory τ , we compute

$$
\varsigma ( x , \tau ) = \lambda _ { r } z ( r ( x , \tau ) ) + \lambda _ { p } z ( \log \pi _ { \mathrm { R W P O } } ( \tau \mid x ) ) ,\tag{14}
$$

where $z ( \cdot )$ denotes within-instance normalization, $\lambda _ { r }$ controls the contribution of task reward, and $\lambda _ { p }$ controls the strength of the base-policy likelihood anchor. Given a candidate set $\mathcal { C } ( x ) = \{ \tau _ { 1 } , . . . , \tau _ { K } \}$ , the scores are converted into normalized weights using a softmax temperature $T _ { w }$ :

$$
w _ { k } = \frac { \exp ( \varsigma ( x , \tau _ { k } ) / T _ { w } ) } { \sum _ { j = 1 } ^ { K } \exp ( \varsigma ( x , \tau _ { j } ) / T _ { w } ) } .\tag{15}
$$

The policy is then adapted by weighted behavioural cloning:

$$
\mathcal { L } _ { \mathrm { t i l t } } ( \theta ) = - \mathbb { E } _ { x } \left[ \sum _ { k = 1 } ^ { K } w _ { k } \log \pi _ { \theta } ( \tau _ { k } \mid x ) \right] .\tag{16}
$$

This objective can be interpreted as an approximation of a reward-tilted distribution specific for COPs. High-reward trajectories receive larger weights, while the likelihood anchor prevents the adapted policy from moving too far away from the original RWPO policy. Therefore, tilted sampling performs reward-guided but conservative probability redistribution: it is more reward-aware than pure distribution sharpening, yet more stable than unconstrained reward-only fine-tuning. More details are provided in Appendix B.5.

## 4 Experiments

We present the main results of our experiments, in which we observe superior performance of the proposed RWPO for SCOPs compared to approaches in the literature. We answer the following questions: 1. How does RWPO compare to existing algorithms on standard benchmarks ofthe OP, PCTSP, SPCTSP, and $K P ? ~ 2$ . How does the out-of-distribution (OOD) generalization ofthe RWPO compare to existing neural combinatorial optimization (NCO) algorithms? and 3. What structural information is encoded within in the learned representation, and does that explain generalization?

Evaluation Protocol. To ensure a fair comparison with prior studies, we follow standard benchmark protocols for both mid- and large-scale instances. For mid-scale instances $( n = 1 0 0 )$ , we use test sets sampled from the same uniform distributions as in Kool et al. [2019] and Kwon et al. [2020]. For large-scale instances $( n \in \mathsf { \bar { \{ s o o , 1 0 0 0 , 2 0 0 0 \} } } )$ ), we adopt the test sets from Zheng et al. [2024]. To further evaluate generalization, we conduct zero-shot testing on instances generated with different random seeds and on the standard OPLib benchmark dataset Joshi et al. [2019]. We evaluate performance using the average accumulated reward or cost; for minimization problems, the optimality gap is computed as $\mathrm { G a p ( \% ) { \stackrel { \sim } { = } } ( O b j _ { a l g } - \bar { O b j } _ { r e f } ) / O b j _ { r e f } \times 1 0 0 }$ . Finally, to reflect practical usage, solver runtimes are measured out-of-the-box—evaluating non-neural methods on CPU and neural methods on GPU—using an Intel i5-14600KF CPU and an NVIDIA RTX 5060 Ti (16GB) GPU.

Algorithm 1 RWPO Training and Scale Adaptation   
1: Input: Training distribution $\mathcal { D } _ { \mathrm { t r a i n } } ;$ target-scale distribution $\mathcal { D } _ { \mathrm { a d a p t } } ;$ policy π ; RWPO parameters $( \alpha , \beta , \lambda )$ ; tilting   
parameters $\left( \lambda _ { r } , \lambda _ { p } , T _ { w } \right)$   
2: Phase I: Base RWPO training   
3: for each batch $x _ { b } \sim \mathcal { D } _ { \mathrm { t r a i n } }$ do   
4: Sample M multi-start trajectories $\{ \tau _ { b , m } \} _ { m = 1 } ^ { M } \sim \pi _ { \theta } ( \cdot \mid x _ { b } )$   
5: Evaluate rewards ${ r _ { b , m } }$ and compute normalized trajectory log-likelihood   
$\ell _ { b , m } = T _ { b , m } ^ { - \beta } \sum _ { t = 1 } ^ { T _ { b , m } } \log \pi _ { \theta } ( a _ { t } \mid s _ { t } , x _ { b } ) .$   
6: for each ordered pair $( i , j )$ do   
7:   
$\Delta _ { i j }  r _ { b , i } - r _ { b , j } , \quad \tilde { \omega } _ { i j }  1 + \lambda \frac { [ \Delta _ { i j } ] _ { + } } { \bar { \Delta } ^ { ( b ) } + \epsilon } ,$   
where $\bar { \Delta } ^ { ( b ) }$ is the mean absolute reward gap in the batch.   
8: end for   
9: Compute the confidence-regularized all-pair RWPO loss   
$\mathcal { L } _ { \mathrm { R W P O } } = - \frac { 1 } { | \mathcal { P } _ { b } | } \sum _ { ( i , j ) \in \mathcal { P } _ { b } } \left[ \tilde { \omega } _ { i j } \log \sigma \big ( \alpha ( \ell _ { b , i } - \ell _ { b , j } ) \big ) \right] .$   
10: Update $\theta  \theta - \eta \nabla _ { \theta } \mathcal { L } _ { \mathrm { { R W P O } } } .$   
11: end for   
12: Phase II: Scale-specific fine-tuning   
13: Continue Phase-I RWPO training on $\mathcal { D } _ { \mathrm { a d a p t } }$ using $\begin{array} { r } { \eta _ { \mathrm { F T } } \ll \eta . } \end{array}$   
14: Set the frozen reference policy $\pi _ { \mathrm { r e f } }  \pi _ { \theta } .$   
15: Phase III: Tilted-sampling adaptation   
16: Freeze the pre-tilt policy: $\pi _ { \mathrm { r e f } }  \pi _ { \theta } .$   
17: for each batch $\{ x _ { b } \} _ { b = 1 } ^ { B } \sim \mathcal { D } _ { \mathrm { a d a p t } }$ do   
18: Sample K trajectories $\mathcal { C } ( x _ { b } ) \overset { \cdot } { = } \{ \tau _ { b , k } \} _ { k = 1 } ^ { K } \sim \pi _ { \theta } ( \cdot \mid x _ { b } )$   
19: Standardize rewards and reference log-likelihood:   
$\hat { r } _ { b , k } = z _ { b } ( r ( x _ { b } , \tau _ { b , k } ) ) , \qquad \hat { q } _ { b , k } = z _ { b } ( \log \pi _ { \mathrm { r e f } } ( \tau _ { b , k } \mid x _ { b } ) ) .$   
20: Compute tilted scores and weights:   
$\varsigma _ { b , k } = \lambda _ { r } \hat { r } _ { b , k } + \lambda _ { p } \hat { q } _ { b , k } , \qquad w _ { b , k } = \frac { \exp \left( \varsigma _ { b , k } / T _ { w } \right) } { \sum _ { j = 1 } ^ { K } \exp \left( \varsigma _ { b , j } / T _ { w } \right) } .$   
21: Compute the tilted-sampling loss   
$\mathcal { L } _ { \mathrm { t i l t } } = - \frac { 1 } { B } \sum _ { b = 1 } ^ { B } \sum _ { k = 1 } ^ { K } \operatorname { s g } ( w _ { b , k } ) \log \pi _ { \theta } ( \tau _ { b , k } \mid x _ { b } ) ,$   
where sg(·) stops gradients through the sampling weights.   
22: Update $\right. \theta \left. \theta - \eta _ { \mathrm { T S } } \nabla _ { \theta } \mathcal { L } _ { \mathrm { t i l t } }$   
23: end for

Implementation. For same-distribution evaluation, we apply continued low-learning-rate training followed by tilted sampling (TS). For large-scale or distribution-shifted instances, we employ scale-specific fine-tuning (40 epochs for $N = 5 0 0 ; 3 0$ epochs for $N \in \{ 1 0 0 0 , 2 0 0 0 \} )$ , paired with TS for 100 epochs. Regarding hyperparameters, we follow Pan et al. [2025a] by setting the inverse temperature to $\alpha = 0 . 0 5$ for both RWPO and PO. The reward-weighting coefficient is set to $\bar { \lambda } = 0 . 5$ for the OP and KP, and $\lambda = 1 . 0$ for the remaining problems. For the TS stage, we use $\lambda _ { r } = 1 , \lambda _ { s } = 0 . 2 5$ , and $T _ { w } = 1$ . During training, our multi-start rollout size is $\mathrm { \Delta } M = \operatorname* { m i n } \{ N , 1 0 0 \}$ for the KP and $M = \operatorname* { m i n } \{ N / 2 , 1 0 0 \}$ for all other problems, whereas for inference, we report multi-start decoding performance using

$M = N$ across all SCOPs. Lastly, we apply a length-normalized log-probability adjustment (see Appendix B.1 for details), setting the normalization coefficient to $\beta = 0 . 3$ for the OP and KP, and $\dot { \beta } = \bar { 0 } . 1$ for the remaining SCOPs.

Baselines. This paper compares the proposed RWPO (confidence-regularized all pair version) to advanced (meta-)heuristic algorithms and existing neural constructive solvers for COPs, namely non-neural solvers (LKH [Helsgaun, 2017], Compass [Joshi et al., 2019], OR-Tools [Perron and Furnon, 2019]), neural constructive solvers (AM [Kool et al., 2019], POMO [Kwon et al., 2020], MDAM [Xin et al., 2021], Sym-NCO [Kim et al., 2022], PO [Pan et al., 2025a]) and large-scale neural solvers (BQ [Drakulic et al., 2023], GLOP [Ye et al., 2024], UDC [Zheng et al., 2024]). The implementation details and settings of all methods compared are described in Appendix C.

## 4.1 Comparison with Existing Algorithms on Standard Benchmarks

Tab. 1 demonstrates the effectiveness of our framework across both zero-shot scale generalization (OOD) and targetscale adaptation (ID) (for CVRP results, see Appendix E.1). Without any scale-specific fine-tuning, the base RWPO model exhibits robust OOD generalization, significantly mitigating the severe performance degradation seen in standard PO and prior constructive methods at $N = 1 0 0 0$ and $N = 2 0 0 0$ . Although this zero-shot performance trails BQ by a narrow margin on OP and KP, our approach successfully surpasses BQ in the ID setting. When combined with tilted sampling for target-scale adaptation (RWPO+TS), our approach establishes a new SOTA. It consistently outperforms the highly optimized UDC baseline across all problems, regularly matching or exceeding the performance of the rigorous OR-Tools and Compass baselines while requiring only seconds to minutes of computational time, including a 4.92% improvement at OP2000 on Compass. Furthermore, RWPO+TS is highly computationally efficient: on the OP, it achieves performance comparable to the RWPO+LS proposed in [Pan et al., 2025a], but substantially reduces the required training time.

## 4.2 Analysis of Generalization

To evaluate our framework on standardized routing benchmarks, Tab. 2 presents zero-shot generalization results evaluated exclusively on the Euclidean 2D instances from OPLib Generation 3. While the absolute optimality gaps reported here are larger than those on our synthetic test sets, this is expected: many OPLib instances feature severe spatial distribution shifts—such as heavily clustered or highly irregular node coordinates—that deviate drastically from our training distribution. Despite these challenging shifts, RWPO demonstrates remarkable robustness. For instance, $\mathrm { R W P O _ { 1 0 0 } }$ significantly outperforms standard $\mathrm { P O _ { 1 0 0 } }$ across all regimes, cutting the optimality gap by more than half on the massive (2000, 5934) instances (36.24% vs. 75.99%). The results also reveal a natural cross-scale trade-off: policies trained on smaller graphs (e.g., $\mathrm { R W P O _ { 5 0 } ) }$ excel on smaller instances, whereas models adapted on massive graphs $\left( \mathrm { R W P O _ { 2 0 0 0 } } \right)$ generalize much better to the largest regimes. To bridge this trade-off, we introduce $\mathrm { R W P O _ { c r o s s } }$ , a decoding strategy that processes each instance using all scale-specific RWPO models and simply selects the final valid trajectory with the highest reward. By combining the complementary structural priors learned across different scales, the cross-scale decoding strategy strictly dominates any individual model, achieving the lowest gaps across all problem sizes (e.g., 4.23% on the smallest scale and 22.82% on the largest) with only a modest increase in overall inference time.

Table 2: Average gap to the Compass results.
<table><tr><td>Method</td><td>(50, 200]</td><td>(200, 400]</td><td>(400, 2000]</td><td>(2000, 5934]</td></tr><tr><td> $\mathrm { P O M O _ { 1 0 0 } }$ </td><td>6.03%</td><td>10.98%</td><td>41.04%</td><td>61.54%</td></tr><tr><td> $\mathrm { P O _ { 1 0 0 } }$ </td><td>13.31%</td><td>41.47%</td><td>62.79%</td><td>75.99%</td></tr><tr><td> $\mathrm { R W P O _ { 5 0 } }$ </td><td>5.71%</td><td>15.89%</td><td>34.48%</td><td>46.74%</td></tr><tr><td> $\mathrm { R W P O _ { 1 0 0 } }$ </td><td>6.35%</td><td>10.77%</td><td>26.71%</td><td>35.72%</td></tr><tr><td> ${ \mathrm { R W P O } } _ { 5 0 0 }$ </td><td>14.68%</td><td>16.72%</td><td>25.96%</td><td>28.47%</td></tr><tr><td> $\mathrm { R W P O _ { 1 0 0 0 } }$ </td><td>13.82%</td><td>17.02%</td><td>22.92%</td><td>26.72%</td></tr><tr><td> $\mathrm { R W P O _ { 2 0 0 0 } }$ </td><td>15.61%</td><td>17.76%</td><td>23.81%</td><td>27.29%</td></tr><tr><td> $\mathrm { R W P O _ { c r o s s } }$ </td><td>3.20%</td><td>6.07%</td><td>19.11%</td><td>22.78%</td></tr></table>

Fig. 2 demonstrates that RWPO improves both average solution quality and instance-level robustness. Across all evaluated problem sizes and travel budgets, RWPO consistently achieves smaller objective gaps against Compass than both standard PO and POMO, with its advantage becoming particularly pronounced on larger instances. PO degrades severely at scale, suggesting that binary preference supervision struggles when the decision space expands and the selected subset becomes sparse. Conversely, while POMO outperforms PO in several large-scale settings—indicating the necessity of explicit reward signals—it remains limited by high-variance training and inefficient exploration. As the travel budget increases and the OP shifts from a sparse selection problem to a dense routing problem (similar to the TSP), RWPO maintains its strong performance, proving it captures fundamental routing structures rather than just exploiting sparse reward signals. These results confirm that preserving reward-gap information makes preference learning vastly more robust to weak comparisons and distribution shifts.

Table 1: Comparison on OP, PCTSP, SPCTSP, and KP instances across different scales.
<table><tr><td rowspan="2">Methods</td><td colspan="3">N = 100</td><td rowspan="2"></td><td colspan="3">N = 500</td><td colspan="3">N = 1,000</td><td colspan="3">N = 2,000</td></tr><tr><td>Obj.</td><td>Gap</td><td>Time |</td><td>Obj.</td><td>Gap</td><td>Time</td><td>Obj.</td><td>Gap</td><td>Time</td><td>Obj.</td><td>Gap</td><td>Time</td></tr><tr><td rowspan="9">Compass</td><td>33.133</td><td>0.00%</td><td></td><td>12m</td><td>81.123</td><td>0.00%</td><td>56s</td><td>115.690</td><td>0.00%</td><td>4.1m</td><td>166.749</td><td>0.00%</td><td>1.3m</td></tr><tr><td>AM-bs1024</td><td>32.686</td><td>1.35%</td><td>20m</td><td>68.592</td><td>15.45%</td><td>2.5m</td><td>79.655</td><td>31.15%</td><td>7m</td><td>95.484</td><td>42.74%</td><td>3.4m</td></tr><tr><td>MDAM-bs50</td><td>32.975</td><td>0.48%</td><td>22m 6m</td><td>65.842</td><td>18.84%</td><td>3.5m 52s</td><td>81.114 95.358</td><td>29.89%</td><td>33m</td><td>102.983 122.832</td><td>38.24%</td><td>12m</td></tr><tr><td>Sym-NCO</td><td>33.039</td><td>0.28%</td><td>1.5m</td><td>73.225 0OD 78.140</td><td>9.74% 3.68%</td><td>48s</td><td>107.446</td><td>17.57% 7.13%</td><td>2.7m 4.3m</td><td>144.007</td><td>26.34% 13.64%</td><td>40s 3.3m</td></tr><tr><td>BQ</td><td>33.074</td><td>0.18%</td><td>24s</td><td>65.089</td><td>19.77%</td><td>7s</td><td>79.329</td><td>31.43%</td><td>37s</td><td>99.403</td><td>40.39%</td><td></td></tr><tr><td>POMO</td><td>33.150</td><td>-0.05%</td><td>24s</td><td>35.661</td><td>56.04%</td><td>6s</td><td>41.233</td><td></td><td>33s</td><td>49.028</td><td></td><td>18s</td></tr><tr><td>PO PO+LS</td><td>33.259</td><td>-0.38%</td><td>25s</td><td>33.701</td><td>58.46%</td><td></td><td></td><td>64.36%</td><td>33s</td><td>44.452</td><td>70.60%</td><td>16s</td></tr><tr><td>RWPO</td><td>33.281 33.285</td><td>-0.45%</td><td>25s</td><td>74.879</td><td>7.70%</td><td>6s 7s</td><td>38.010 99.577</td><td>67.14% 13.93%</td><td>46s</td><td>134.607</td><td>73.34% 19.28%</td><td>15s 23s</td></tr><tr><td>RWPO+LS</td><td>33.302</td><td>-0.46% -0.51%</td><td>25s</td><td>75.763</td><td>6.61%</td><td></td><td>99.907</td><td>13.64%</td><td>45s</td><td>134.408</td><td>19.40%</td><td>24s</td></tr><tr><td>UDC</td><td></td><td></td><td></td><td></td><td>79.248</td><td>2.31%</td><td>8s 57s</td><td>113.173</td><td>2.18%</td><td>19m</td><td>145.001</td><td></td><td>25s</td></tr><tr><td rowspan="2">RWPO+TS</td><td>33.302 -0.51%</td><td></td><td>25s</td><td rowspan="2">e</td><td>83.493</td><td>-2.92%</td><td>8s</td><td>121.273</td><td>-4.83%</td><td>56s</td><td></td><td>13.04% 175.372-5.17%</td><td>29s</td></tr><tr><td>OR-Tools*</td><td>1.61%</td><td>5h</td><td></td><td>14.000</td><td>7.63%</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="9">PCTSP Sym-NCO PO RWPO GLOP</td><td rowspan="9">AM-bs1024 MDAM-bs50</td><td>6.070</td><td>2.01%</td><td rowspan="9"></td><td rowspan="9">0OD</td><td rowspan="9">19.127</td><td></td><td>16h 8m</td><td>20.600 30.812</td><td>14.53% 71.30%</td><td>16h 35m 1.4h</td><td>一 47.420</td><td>94.31%</td><td>23m</td></tr><tr><td>6.094</td><td>22m</td><td></td><td>47.04%</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>6.063</td><td>28m</td><td>14.829</td><td>14.00%</td><td>9m</td><td>22.510 25.15%</td><td></td><td>33.015 45.329</td><td>35.29%</td><td>1.2h</td></tr><tr><td>6.054</td><td>1.34% 6m</td><td>19.433</td><td>49.39%</td><td>1.8m</td><td>29.443</td><td>63.69%</td><td>8m</td><td>85.74%</td><td>3m</td></tr><tr><td>5.982</td><td>0.13% 22s 19s</td><td>16.132</td><td>24.02%</td><td>22s</td><td>26.926 49.70%</td><td>4m</td><td>41.755</td><td>71.10%</td><td>4m</td></tr><tr><td>5.977</td><td>0.05%</td><td>15.758</td><td>21.14%</td><td>22s 24.487</td><td>36.14%</td><td>4m</td><td>37.934 55.44%</td><td></td><td>3m</td></tr><tr><td></td><td></td><td>14.432</td><td>10.95%</td><td>1m</td><td>19.869</td><td>10.46%</td><td>1.8m 28.418</td><td>16.45%</td><td>36s</td></tr><tr><td></td><td></td><td>D 13.158</td><td>1.15%</td><td>1.4m</td><td>18.283</td><td>1.65%</td><td>3.6m</td><td>25.045 2.63%</td><td>56s</td></tr><tr><td>RWPO+TS</td><td>5.974 0.00%</td><td>21s</td><td>13.008</td><td>0.00%</td><td>14s</td><td>17.987</td><td>0.00% 2m</td><td>24.404</td><td>0.00%</td></tr><tr><td rowspan="6">SPCSP PO RWPO</td><td>AM-bs1024</td><td></td><td>23m</td><td rowspan="5"></td><td rowspan="5"></td><td>32.96%</td><td>6.5m</td><td rowspan="5">26.646</td><td rowspan="5">48.35%</td><td rowspan="5">29m</td><td rowspan="5">37.421 32.102</td><td rowspan="5"></td><td rowspan="5">1.5m 53.59% 20m 31.75%</td></tr><tr><td>MDAM-bs50</td><td>6.115 6.071</td><td>2.36% 30m</td><td>17.285</td></tr><tr><td></td><td>1.62% 5.983</td><td>0OD 14.826 16.562</td><td>14.05% 8.3m 27.40% 26s</td><td>23.24% 1.3h 59.97% 3m</td></tr><tr><td>5.977</td><td>0.15% 0.05%</td><td>16.601</td><td>27.70% 26s</td><td>42.35%</td><td>46.397 90.42% 37.538 54.07%</td></tr><tr><td></td><td></td><td>19s D</td><td>13.233</td><td>1.79%</td><td>3m 3.13% 3.7m</td></tr><tr><td>UDC RWPO+TS</td><td>5.974 0.00%</td><td>20s</td><td>13.000</td><td>0.00%</td><td>1.5m 16s</td><td>18.523 17.961</td><td>0.00%</td><td>1.7m</td><td>25.551 4.87% 24.365 0.00%</td></tr></table>

<table><tr><td>BQ</td><td>OR-Tools</td><td>57.640</td><td>0.00%</td><td>6.5m</td><td>128.369</td><td>0.00%</td><td>25s</td><td>258.254</td><td></td><td>0.00%</td><td>1.8m</td><td>518.397</td><td>0.00%</td><td>52s</td></tr><tr><td rowspan="5">R</td><td></td><td>57.578</td><td>0.11%</td><td>7m</td><td></td><td>128.279</td><td>0.07%</td><td>1.1m</td><td>256.409</td><td>0.71%</td><td>7.3m</td><td>517.124</td><td>0.25%</td><td>6.1m</td></tr><tr><td>POMO</td><td>57.623</td><td>0.03%</td><td>49s</td><td>0OD</td><td>128.316</td><td>0.04%</td><td>8s</td><td>254.537</td><td>1.44%</td><td>53s</td><td>471.631</td><td>9.02%</td><td>46s</td></tr><tr><td>PO</td><td>57.629</td><td>0.02%</td><td>1.3m</td><td></td><td>108.750</td><td>15.28%</td><td>10s</td><td>147.525</td><td>42.88%</td><td>41s</td><td>230.181</td><td>55.60%</td><td>34s</td></tr><tr><td>RWPO</td><td>57.629</td><td>0.02%</td><td>1.3m</td><td></td><td>126.541</td><td>1.42%</td><td>12s</td><td>253.548</td><td>1.82%</td><td>1.5m</td><td>503.235</td><td>2.92%</td><td>1.4m</td></tr><tr><td>UDC</td><td></td><td></td><td></td><td></td><td>128.356</td><td>0.01%</td><td>47s</td><td>258.224</td><td>0.01%</td><td>1.9m</td><td>518.226</td><td>0.03%</td><td>30s</td></tr><tr><td>RWPO</td><td></td><td>57.629</td><td>0.02%</td><td>1.3m</td><td>D</td><td>128.364</td><td>0.00%</td><td>12s</td><td>258.247</td><td>0.00%</td><td>1.3m</td><td>518.284</td><td>0.02%</td><td>1.2m</td></tr></table>

For each problem, the upper block reports out-of-distribution (OOD) results, evaluating the zero-shot scale generalization of the base models, while the lower block reports in-distribution (ID) results after scale-specific adaptation. For AM, MDAM, Sym-NCO, BQ, UDC, and POMO on KP, we use the official pre-trained models from the original papers.

## 4.3 Entropy Analysis

To better understand why RWPO generalizes on SCOPs, we analyze policy-level uncertainty and decoder-query representation diversity using two complementary measures. Policy entropy is the Shannon entropy of the decoder’s probability distribution over feasible next actions, measuring how concentrated its action probabilities are at each decoding step. Decoder-query soft entropy applies the soft-entropy procedure used by Conklin et al. [2026] to decoder queries collected across instances and rollouts at each step. Specifically, queries are $L _ { 2 }$ normalized and softly assigned to randomly sampled reference directions on the unit sphere; we then compute the Shannon entropy of the aggregated assignment distribution. This metric serves as a entropy-based proxy for angular representation diversity. Since query– key alignment determines relative decoder candidate scores, this measure helps identify when the query component of the decoder becomes more concentrated across different partial solutions (see Appendix D.1 for more details). We target decoder queries because they act as the information bottleneck for next-action scoring, uniquely aggregating critical state context: the global graph, node embeddings, and feasibility constraints. Together, the two measures characterize how action uncertainty and query-distribution diversity evolve during solution construction as feasibility constraints change. We use OP as the primary testbed because it combines reward-based subset selection with budget-constrained routing, allowing us to examine these dynamics as candidate-set size increases.

![](images/727a38c3859f68b4194a38dc7e094377cf7e2276401e8e91a3acdc813c2fe07a.jpg)  
(a) N = 100

![](images/b79813acfc8ae4520620c5e4c26cfa88613f2cd8e823e0ef828e529d826f0606.jpg)  
(b) N = 500

![](images/66058f33c486dab4af50df181da239dc36711f15d1769620947a6caef209fba6.jpg)  
(c) N = 1,000

Figure 2: Objective gap against Compass on OP instances under different travel budgets T and problem sizes.  
![](images/a9203e1f048087f8a70d12e5ed368b51cfef6030723d60ae64055cb496163b6d.jpg)  
(a) N = 100

![](images/7ac8cb8bbf0577aa711793bab88e834fba031fb1b38e2c483486600d6508d60c.jpg)  
(b) N = 200

![](images/dd168c3cebf7ba217411414fea6c5d595377ef7b0f196bbf215d932a7272014d.jpg)  
(c) N = 500

![](images/3e63c015640045afe8b7dda3ea04f3ba415ddd0fbd0163f7bbef5d01b13283c6.jpg)  
(d) N = 1,000  
Figure 3: Step-wise policy entropy during decoding on OP instances of different sizes.

We contrast next-action uncertainty within individual rollouts (Fig.3) against query representation diversity across the batch (Fig.4). On OP500 and OP1000, PO exhibits prolonged high policy entropy alongside comparatively lower decoder-query soft entropy, showing that uncertain action distributions need not be accompanied by diverse query representations. POMO attains high intermediate query entropy, but subsequently enters a sustained lower-entropy regime earlier than the RWPO variants. In contrast, RWPO exhibits lower early-stage policy entropy than both baselines while maintaining or recovering query diversity over substantial shared late-decoding intervals. Fine-tuned RWPO variants successfully decouple these metrics, maintaining high query diversity despite concentrated action distributions—a behavior most evident in RWPO-FT500 (OP500) and RWPO-FT1k (OP1000). On OP100, RWPO-TS similarly exhibits higher query entropy but lower policy entropy than RWPO over most of decoding, although RWPO itself shows a pronounced late decline in both measures. Overall, the results show that RWPO variants can combine diverse query representations across partial solutions with concentrated next-action distributions within individual branches, rather than achieving representation diversity simply by maintaining greater action randomness.

## 4.4 Ablation Study

Fig. 5a shows that PO and both RWPO variants achieve similar best-reward values on OP100; however, this parity does not hold for the average performance across the k rollouts. As shown in Fig.5b, the mean-reward curve of standard PO is highly oscillatory and fails to stabilize, whereas RWPO—particularly the all-pair variant—maintains a smooth, stable trajectory. A similar pattern emerges in the OP500 OOD evaluation (Fig.5c), as well as the PCTSP (Fig. 5d) and SPCTSP/KP results (Appendix E.5). Because top-k POMO aggregates the learning signal across all k rollouts, the learned policy should improve the entire rollout set collectively rather than merely optimizing its single best member. The observation that PO retains a competitive best reward while exhibiting unstable mean performance supports our hypothesis: without explicit reward magnitudes, purely ordinal updates fail to properly differentiate the relative quality of the full rollout batch. In contrast, RWPO all-pair remains stable across all tested problem settings, preserving the stabilizing properties of symmetric pairwise learning while explicitly scaling its gradient updates by the reward gaps.

![](images/d71f405e1043a75f26b27d818565d9371c838542b9eeba48680c68171f78fcd4.jpg)  
(a) N = 100

![](images/a5afc85cef1e7fb03e6e7c747ce0f577bc0c9f9e6f93020657d3c6d33c6c47e1.jpg)  
(b) N = 200

![](images/b82c83793240c7bbfdb7d157df13d636cfa7badc663c6a4889e5f9cbbe82338f.jpg)  
(c) N = 500

![](images/e66fd9196b9da04d48c4de2004196e88b581d8e95a804c4f6edd2674797b7f28.jpg)  
(d) N = 1,000

Figure 4: Step-wise decoder-query soft entropy during decoding on OP instances of different sizes.  
![](images/a5b60126f3f5bf8fc804209deec668ba2a45a95f20b2ec7c9e1f4fd99705d01b.jpg)  
(a) Best reward on OP100

![](images/2aa84aaebe54c43e6753a3deae150ce8b2fc0eee7f67c62bf537dbba2f257409.jpg)  
(b) Mean reward on OP100

![](images/31e255a68f495f22f09879459038a381bbbf26c97b599c0b3864dda6bdae8686.jpg)  
(c) Best reward on OP500

![](images/2f5e81d95fbfd70b5ffc4e673b34e0088e0d9f3bc108d58f6d43a9eccec43d9e.jpg)  
(d) Mean reward on PCTSP100  
Figure 5: Training and evaluation curves on the OP and PCTSP problem.

## 5 Discussion and Conclusion

This work presented an effective learning-based framework for selective combinatorial optimization problems (SCOPs). Central to our approach is Reward-Weighted Preference Optimization (RWPO), which improves upon standard preference learning by rigorously preserving reward-gap information. Unlike standard PO, which reduces trajectory comparisons to purely ordinal binary preferences, RWPO scales the loss of each pair by their exact reward difference, directly aligning the agent’s learning signal with reward maximization. This mechanism is essential for SCOPs, where subset-selection decisions cause feasible solutions to vary drastically in actual quality even when correctly ordered. By naturally down-weighting weak comparisons, RWPO learns a highly constraint-aware policy, delivering stronger out-of-distribution (OOD) generalization than PO while maintaining much greater training stability than standard POMO. When integrated with our complete adaptation pipeline—comprising top-k selective decoding, scale-specific fine-tuning, and tilted sampling—RWPO establishes state-of-the-art performance across four SCOP benchmarks.

Although RWPO shows strong performance on SCOPs, it relies on exact reward evaluations, which could be difficult to obtain in partially observable or multi-objective environments. Second, its reward-weighting strength and lengthnormalization coefficient are currently fixed, suggesting future works on adaptive weighting based on reward dispersion, instance difficulty, or route-length statistics. Finally, while our entropy analyses reveal a strong correlation between representation diversity and generalization, they are purely observational. Future works could employ representation interventions to definitively prove whether these diverse decoder queries are the direct causal factors of the model’s OOD robustness.

## Acknowledgments

This publication is part of the project FMaaS - it’s a cargo match (with project number NWA.1518.22.023) of the research programme Dutch Research Agenda - research along NWA routes by consortia which is (partly) financed by the Dutch Research Council (NWO).

## References

Riad Akrour, Marc Schoenauer, and Michèle Sebag. Preference-based reinforcement learning. In European Conference on Machine Learning and Knowledge Discovery in Databases (ECML PKDD), pages 12–27. Springer, 2011.

Mohammad Gheshlaghi Azar, Mark Rowland, Bilal Piot, Daniel Guo, Rémi Munos, Branislav Kveton, et al. A general theoretical paradigm to understand learning from human preferences. In International Conference on Learning Representations, 2024.

Egon Balas. The prize collecting traveling salesman problem. Networks, 19(6):621–636, 1989.

Irwan Bello, Hieu Pham, Quoc V Le, Mohammad Norouzi, and Samy Bengio. Neural combinatorial optimization with reinforcement learning. arXiv preprint arXiv:1611.09940, 2016.

Yoshua Bengio, Andrea Lodi, and Antoine Prouvost. Machine learning for combinatorial optimization: A methodologi cal tour d’horizon. European Journal of Operational Research, 290(2):405–421, 2021.

Jieyi Bi, Yining Ma, Jiahai Wang, Zhiguang Cao, Jinbiao Chen, Yuan Sun, and Yeow Meng Chee. Learning generalizable models for vehicle routing problems via knowledge distillation. Advances in Neural Information Processing Systems, 35:31226–31238, 2022.

Christopher J.C. Burges, Robert Ragno, and Quoc V. Le. Learning to rank using gradient descent. In Proceedings of ICML, 2006.

Quentin Cappart, Thierry Moisan, Louis-Martin Rousseau, Isabeau Prémont-Schwarz, and Andre A Cire. Combining reinforcement learning and constraint programming for combinatorial optimization. In Proceedings of the AAAI conference on artificial intelligence, volume 35, pages 3677–3687, 2021.

Shengkai Chen, Zhiguang Cao, Jianan Zhou, Yaoxin Wu, Senthilnath Jayavelu, Zhuoyi Lin, Xiaoli Li, and Shili Xiang. Dragon: Llm-driven decomposition and reconstruction agents for large-scale combinatorial optimization. In Proceedings ofthe 25th International Conference on Autonomous Agents and Multiagent Systems, pages 2809–2817. International Foundation for Autonomous Agents and Multiagent Systems, 2026. doi: 10.65109/SBAY6258.

Xinyun Chen and Yuandong Tian. Learning to perform local rewriting for combinatorial optimization. Advances in neural information processing systems, 32, 2019.

Hui Cheng, Hao Zheng, Yuru Cong, Wei Jiang, and Shiliang Pu. Select and optimize: Learning to solve large-scale tsp instances. In Proceedings ofthe 26th International Conference on Artificial Intelligence and Statistics, volume 206 of Proceedings ofMachine Learning Research, pages 1219–1231, 2023.

Paul F Christiano, Jan Leike, Tom Brown, Miljan Martic, Shane Legg, and Dario Amodei. Deep reinforcement learning from human preferences. Advances in Neural Information Processing Systems, 30, 2017.

Henry C Conklin, Tom Hosking, Tan Yi-Chern, Julian Gold, Jonathan D Cohen, Thomas L Griffiths, Max Bartolo, and Seraphina Goldfarb-Tarrant. Learning is forgetting: LLM training as lossy compression. arXiv preprint arXiv:2604.07569, 2026.

Darko Drakulic, Sofia Michel, Florian Mai, Arnaud Sors, and Jean-Marc Andreoli. BQ-NCO: bisimulation quotienting for efficient neural combinatorial optimization. In Alice Oh, Tristan Naumann, Amir Globerson, Kate Saenko, Moritz Hardt, and Sergey Levine, editors, Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023, 2023. URL http://papers.nips.cc/paper\_files/paper/2023/hash/ f445ba15f0f05c26e1d24f908ea78d60-Abstract-Conference.html.

Johannes Fürnkranz, Eyke Hüllermeier, Weiwei Cheng, and Sang-Hyeun Park. Preference-based reinforcement learning: a formal framework and a policy iteration algorithm. Machine learning, 89:123–156, 2012.

Evan Greensmith, Peter L. Bartlett, and Jonathan Baxter. Variance reduction techniques for gradient estimates in reinforcement learning. Journal ofMachine Learning Research, 5:1471–1530, 2004.

Aldy Gunawan, Hoong Chuin Lau, and Pieter Vansteenwegen. Orienteering problem: A survey of recent variants, solution approaches and applications. European Journal of Operational Research, 255(2):315–332, 2016. doi: 10.1016/j.ejor.2016.04.059.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

Keld Helsgaun. An extension of the Lin-Kernighan-Helsgaun TSP solver for constrained traveling salesman and vehicle routing problems. Roskilde: Roskilde University, 12:966–980, 2017.

André Hottung and Kevin Tierney. Neural large neighborhood search for the capacitated vehicle routing problem. In Proceedings ofECAI 2020, pages 443–450, 2020.

André Hottung, Paula Wong-Chung, and Kevin Tierney. Neural deconstruction search for vehicle routing problems. arXiv preprint arXiv:2501.03715, 2025.

Qi Hou, Jianan Yang, Yining Su, Xin Wang, and Yu Deng. Generalize learned heuristics to solve large-scale vehicle routing problems in real-time. In International Conference on Learning Representations, 2023.

Edward Hu, Moksh Jain, Eric Elmoznino, Younesse Kaddar, Guillaume Lajoie, Yoshua Bengio, and Nikolay Malkin. Amortizing intractable inference in large language models. In International Conference on Learning Representations, volume 2024, pages 43053–43083, 2024.

Julian Ibarz, Jan Leike, Tobias Pohlen, Geoffrey Irving, Shane Legg, Dario Amodei, and Paul Christiano. Reward learning from human preferences and demonstrations in atari. In Advances in Neural Information Processing Systems, volume 31, 2018.

Yuan Jiang, Zhiguang Cao, Yaoxin Wu, Wen Song, and Jie Zhang. Ensemble-based deep reinforcement learning for vehicle routing problems under distribution shift. Advances in Neural Information Processing Systems, 36: 53112–53125, 2023.

Chaitanya K. Joshi, Thomas Laurent, and Xavier Bresson. An efficient graph convolutional network technique for the travelling salesman problem. arXiv preprint arXiv:1906.01227, 2019.

Elias Khalil, Hanjun Dai, Yuyu Zhang, Bistra Dilkina, and Le Song. Learning combinatorial optimization algorithms over graphs. Advances in neural information processing systems, 30, 2017.

Minsu Kim, Junyoung Park, and Jinkyoo Park. Sym-nco: Leveraging symmetricity for neural combinatorial optimization. Advances in Neural Information Processing Systems, 35:1936–1949, 2022.

Gorka Kobeaga, María Merino, and Jose A Lozano. An efficient evolutionary algorithm for the orienteering problem. Computers & Operations Research, 90:42–59, 2018.

Wouter Kool, Herke van Hoof, and Max Welling. Attention, learn to solve routing problems! In International Conference on Learning Representations, 2019.

Yeong-Dae Kwon, Jinho Choo, Byoungjip Kim, Iljoo Yoon, Youngjune Gwon, and Seungjai Min. POMO: Policy optimization with multiple optima for reinforcement learning. Advances in Neural Information Processing Systems, 33:21188–21198, 2020.

Kimin Lee, Laura Smith, Pieter Abbeel, and Chelsea Finn. Pebble: Feedback-efficient interactive reinforcement learning via relabeling experience and unsupervised pre-training. In International Conference on Machine Learning, pages 6152–6163. PMLR, 2021.

Sirui Li, Zhongxia Yan, and Cathy Wu. Learning to delegate for large-scale vehicle routing. In Advances in Neural Information Processing Systems, volume 34, 2021.

Zijun Liao, Jinbiao Chen, Debing Wang, Zizhen Zhang, and Jiahai Wang. Bopo: Neural combinatorial optimization via best-anchored and objective-guided preference optimization. arXiv preprint arXiv:2503.07580, 2025.

Jiacheng Lin, Zhongruo Wang, Kun Qian, Tian Wang, Arvind Srinivasan, Hansi Zeng, Ruochen Jiao, Xie Zhou, Jiri Gesi, Dakuo Wang, et al. Sft doesn’t always hurt general capabilities: Revisiting domain-specific fine-tuning in llms. In International Conference on Learning Representations, volume 2026, pages 46954–46992, 2026.

Manuj Malik, Jianan Zhou, Shashank Reddy Chirra, and Zhiguang Cao. Pyvrp+: Llm-driven metacognitive heuristic evolution for hybrid genetic search in vehicle routing problems. In Proceedings of the 25th International Conference on Autonomous Agents and Multiagent Systems, pages 2800–2808. International Foundation for Autonomous Agents and Multiagent Systems, 2026. doi: 10.65109/ISFV2587.

Silvano Martello and Paolo Toth. Knapsack Problems: Algorithms and Computer Implementations. Wiley, 1990.

Sarthak Mittal, Leo Gagnon, and Guillaume Lajoie. Beyond distribution sharpening: The importance of task rewards. arXiv preprint arXiv:2604.16259, 2026.

Marius Mosbach, Maksym Andriushchenko, and Dietrich Klakow. On the stability of fine-tuning bert: Misconceptions, explanations, and strong baselines. arXiv preprint arXiv:2006.04884, 2020.

Mohammadreza Nazari, Afshin Oroojlooy, Lawrence Snyder, and Martin Takác. Reinforcement learning for solving the vehicle routing problem. Advances in neural information processing systems, 31, 2018.

Alex Nowak, David Folqué, and Joan Bruna. Divide and conquer networks. In International Conference on Learning Representations, 2018.

Mingjun Pan, Guanquan Lin, You-Wei Luo, Bin Zhu, Zhien Dai, Lijun Sun, and Chun Yuan. Preference optimization for combinatorial optimization problems. arXiv preprint arXiv:2505.08735, 2025a.

Xuanhao Pan, Yan Jin, Yuandong Ding, Mingxiao Feng, Li Zhao, Lei Song, and Jiang Bian. H-tsp: Hierarchically solving the large-scale traveling salesman problem. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 37, pages 9345–9353, 2023. doi: 10.1609/aaai.v37i8.26120.

Yuxin Pan, Ruohong Liu, Yize Chen, Zhiguang Cao, and Fangzhen Lin. Hierarchical learning-based graph partition for large-scale vehicle routing problems. In Proceedings of the 24th International Conference on Autonomous Agents and Multi-Agent Systems, 2025b.

Xue Bin Peng, Aviral Kumar, Grace Zhang, and Sergey Levine. Advantage-weighted regression: Simple and scalable off-policy reinforcement learning. In arXiv preprint arXiv:1910.00177, 2019.

Laurent Perron and Vincent Furnon. Or-tools. Google.[Online]. Available: https://developers. google. com/optimization, pages 33–46, 2019.

Jan Peters and Stefan Schaal. Reinforcement learning by reward-weighted regression. In Proceedings of the 24th International Conference on Machine Learning, pages 745–750, 2007.

Fredrik Präntare, Herman Appelgren, Mattias Tiger, David Bergström, and Fredrik Heintz. Learning heuristics for combinatorial assignment by optimally solving subproblems. In Proceedings ofthe 21st International Conference on Autonomous Agents and Multiagent Systems, pages 1074–1082, 2022.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. Advances in neural information processing systems, 36:53728–53741, 2023.

John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, and Pieter Abbeel. High-dimensional continuous control using generalized advantage estimation. arXiv preprint arXiv:1506.02438, 2015.

Richard S. Sutton, David McAllester, Satinder Singh, and Yishay Mansour. Policy gradient methods for reinforcement learning with function approximation. In Advances in Neural Information Processing Systems, 2000.

Chenguang Wang, Zhouliang Yu, Stephen McAleer, Tianshu Yu, and Yaodong Yang. Asp: Learn a universal neural solver! IEEE Transactions on Pattern Analysis and Machine Intelligence, 46(6):4102–4114, 2024.

Christian Wirth, Riad Akrour, Gerhard Neumann, and Jan Peters Günther. Model-free preference-based reinforcement learning. Proceedings ofthe AAAI Conference on Artificial Intelligence, 30, 2016.

Christian Wirth, Riad Akrour, Gerhard Neumann, and Jan Peters. A survey of preference-based reinforcement learning methods. Journal ofMachine Learning Research, 18(136):1–46, 2017.

Yaoxin Wu, Wen Song, Zhiguang Cao, Jie Zhang, and Andrew Lim. Learning improvement heuristics for solving routing problems. IEEE transactions on neural networks and learning systems, 33(9):5057–5069, 2021.

Yue Wu, Zhaojian Sun, Shao Huang, Xiaoyu Wang, et al. Pairwise proximal policy optimization: Harnessing feedback from comparative data for reinforcement learning. In Advances in Neural Information Processing Systems, volume 36, 2023.

Liang Xin, Wen Song, Zhiguang Cao, and Jie Zhang. Multi-decoder attention model with embedding glimpse for solving vehicle routing problems. In Proceedings of the AAAI conference on artificial intelligence, volume 35, pages 12042–12049, 2021.

Haoran Ye, Jiarui Wang, Helan Liang, Zhiguang Cao, Yong Li, and Fanzhang Li. Glop: Learning global partition and local construction for solving large-scale routing problems in real-time. In Proceedings of the AAAI conference on artificial intelligence, volume 38, pages 20284–20292, 2024.

Haohui Zhang, Wouter van Heeswijk, Xinyu Hu, Neil Yorke-Smith, and Martijn Mes. Deep learning–accelerated multi-start large neighborhood search for real-time freight bundling. arXiv preprint arXiv:2512.11187, 2025.

Zeyu Zheng, Yining Zhou, Yuxuan Xie, Xinyun Chen, Yinyu Ye, and Hanjun Dai. Udc: A unified neural divideand-conquer framework for large-scale combinatorial optimization problems. In Advances in Neural Information Processing Systems, 2024.

Changliang Zhou, Xi Lin, Zhenkun Wang, Xialiang Tong, Mingxuan Yuan, and Qingfu Zhang. Instance-conditioned adaptation for large-scale generalization of neural routing solver. arXiv preprint arXiv:2405.01906, 2024a.

Changliang Zhou, Xi Lin, Zhenkun Wang, Xialiang Tong, Mingxuan Yuan, and Qingfu Zhang. Instance-conditioned adaptation for large-scale generalization of neural routing solver. IEEE Transactions on Intelligent Transportation Systems, 2026.

Chunting Zhou, Pengfei Liu, Jing Xu, et al. Wpo: Learning from human preferences with weighted preference optimization. arXiv preprint arXiv:2404.XXX, 2024b.

Jianan Zhou, Yaoxin Wu, Wen Song, Zhiguang Cao, and Jie Zhang. Towards omni-generalizable neural methods for vehicle routing problems. In International conference on machine learning, pages 42769–42789. PMLR, 2023.

Zefang Zong, Hansen Wang, Jingwei Wang, Meng Zheng, and Yong Li. Rbg: Hierarchically solving large-scale routing problems in logistic systems via reinforcement learning. In Proceedings ofthe 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pages 4648–4658, 2022. doi: 10.1145/3534678.3539037.

## A Related Work

## A.1 RL-based Neural Network Solvers

Reinforcement-learning-based neural combinatorial optimization (NCO) formulates solution construction as a finitehorizon decision process. A neural policy incrementally selects feasible nodes, edges, or assignments, and the resulting solution quality provides the training signal. Early end-to-end methods used pointer-style recurrent networks trained with REINFORCE for the traveling salesperson problem and knapsack problems [Bello et al., 2016]. Later work incorporated dynamic demands and capacities for vehicle routing [Nazari et al., 2018], graph-embedding-based Qlearning [Khalil et al., 2017], and attention-based encoder–decoder architectures [Kool et al., 2019]. POMO improves exploration by constructing multiple solutions from different starting points and optimizing their returns jointly [Kwon et al., 2020], while Sym-NCO leverages problem symmetries to boost performance [Kim et al., 2022]. MDAM trains multiple decoders and applies a pairwise KL-divergence regularizer at the initial step to diversify their action distributions [Xin et al., 2021]. Recent neural vehicle-routing studies address generalization through adaptive multi-distribution knowledge distillation [Bi et al., 2022], policy-space exploration with curriculum-based scale adaptation [Wang et al., 2024], meta-learning for rapid adaptation to new tasks [Zhou et al., 2023], ensemble RL with mixed-distribution training and diversity regularization [Jiang et al., 2023], and geometry- and density-conditioned policy adaptation for large-scale generalization [Zhou et al., 2026]. Collectively, these methods modify the policy architecture, action-space representation, search procedure, conditioning mechanism, or training distribution. Our work is complementary in that it modifies the optimization objective through reward-weighted preference optimization (RWPO), using pairwise preferences together with reward gaps to guide policy updates.

Hybrid neural solvers combine learned policies with classical search or optimization. NeuRewriter learns local rewriting operations for improving partial solutions [Chen and Tian, 2019], while Neural Large Neighborhood Search learns repair operators within a destroy–repair framework [Hottung and Tierney, 2020]. Neural Deconstruction Search follows a related iterative improvement paradigm by learning an RL policy to remove customers from incumbent routes and combining it with greedy insertion and augmented simulated annealing to reconstruct and improve the solutions [Hottung et al., 2025]. Expanding this learned-heuristic direction, PyVRP<sup>+</sup> employs LLM-driven heuristic evolution within a hybrid genetic search framework [Malik et al., 2026]. Other approaches instead pair deep reinforcement learning with exact subproblem solvers or constraint programming [Cappart et al., 2021, Präntare et al., 2022]. In summary, these hybrid methods improve the reliability of neural construction by retaining classical search, but they introduce complex algorithmic design choices concerning solver calls, neighborhood selection, and the allocation of credit between learned and hand-crafted components. Furthermore, extending these iterative local-search operators to selective combinatorial optimization problems (SCOPs) is notoriously difficult, as the agent must simultaneously optimize routing efficiency and search the massive combinatorial space of unvisited nodes without violating strict global budget constraints.

## A.2 Neural Divide-and-Conquer Methods

Neural divide-and-conquer methods improve the scalability of data-driven approaches by partitioning a large instance, solving smaller subproblems, and combining the resulting partial solutions. Early networks in this direction learned recursive split-and-merge operations for generic combinatorial tasks [Nowak et al., 2018]. To tackle large-scale routing specifically, frameworks like Learning to Delegate, RBG, and H-TSP employ hierarchical policies to assign instance components to localized solvers [Li et al., 2021, Zong et al., 2022, Pan et al., 2023]. Building on this, several recent approaches explicitly separate global partitioning from local construction. For example, TAM learns a constraint-aware VRP partition that allows local TSP heuristics to generalize to larger instances [Hou et al., 2023], while GLOP and Select-and-Optimize alternate between selecting local subproblems and applying learned optimization procedures [Ye et al., 2024, Cheng et al., 2023]. To address the compounded errors that arise from these splits, Hierarchical Learningbased Graph Partitioning (HLGP) uses dual global and local policies [Pan et al., 2025b], and UDC trains a graph-neural divider jointly with a subpath solver via a unified Divide–Conquer–Reunion objective [Zheng et al., 2024]. Pushing this paradigm further, DRAGON deploys LLM-driven agents to identify promising regions and reconstruct global solutions using feedback-guided experience memory [Chen et al., 2026]. These approaches highlight a fundamental challenge in scalable neural solvers: partition decisions must be rigorously evaluated against the completed solution, as an unfavorable early split irreversibly restricts cross-partition connections and bounds the improvements available to local solvers. This vulnerability is especially pronounced in SCOPs, where localized partitions prematurely fragment the global budget and prevent the solver from discovering highly rewarding, globally dispersed subsets.

## A.3 Preference-based Reinforcement Learning Methods

Preference-based reinforcement learning (PbRL) learns from comparisons between trajectories or candidate solutions rather than requiring a fully specified scalar reward. Early approaches inferred latent utilities from pairwise preferences and used preference-based policy iteration or Bayesian model-free policy updates [Akrour et al., 2011, Fürnkranz et al., 2012, Wirth et al., 2016, 2017]. Deep RL from Human Preferences learned a neural reward model from pairwise trajectory-segment labels and optimized the learned reward with RL [Christiano et al., 2017, Ibarz et al., 2018]. PEBBLE improved feedback efficiency through active query selection, unsupervised exploration, and replay relabeling [Lee et al., 2021]. These methods provide a general framework for learning objectives that are difficult to write explicitly, but their performance depends on the coverage and accuracy of the preference or reward model.

Direct preference-optimization methods avoid a separate reward-model training stage. DPO derives a pairwise objective from the KL-regularized RLHF solution [Rafailov et al., 2023], while IPO modifies the preference objective to reduce overfitting under deterministic labels [Azar et al., 2024]. P3O performs policy updates directly from comparative trajectory feedback [Wu et al., 2023]. WPO reweights preference examples according to their likelihood under the current policy to reduce off-policy distribution shift [Zhou et al., 2024b]; these weights reflect policy likelihood rather than the reward gap between two solutions.

Recent neural-based methods have successfully adapted preference signals to avoid scalar reward modeling. For example, Pan et al. [2025a] use statistical preference models to derive a preference-learning objective from entropyregularized RL, with local search generating additional preference pairs for fine-tuning. BOPO similarly establishes preference optimization as a viable data-driven paradigm by constructing best-anchored pairs and scaling its pairwise loss using reward differences [Liao et al., 2025]. Our method fundamentally differs in the scope and application of this reward signal. Rather than relying on anchor-based comparisons, the reward-dependent weight in RWPO dictates the contribution of every preference pair. Thus, the RWPO update strictly preserves both the ordinal direction of the preference and the cardinal magnitude of the reward improvement. This dual signal is mathematically critical for SCOPs, where feasible subsets are massively numerous, frequently near-tied in routing cost, but differ drastically in their accumulated global reward.

## B Additional Methodology Details

We provide further details to Section 3 of the main paper.

## B.1 Decoding-Length Bias of Summed Log-Probabilities in SCOPs

In Preference Optimization (PO) and Reward-Weighted Preference Optimization (RWPO), the standard preference objective increases the trajectory log-probability of preferred solutions. For an autoregressive constructive policy, the trajectory log-probability is

$$
\log \pi _ { \boldsymbol { \theta } } ( \tau \mid \boldsymbol { x } ) = \sum _ { t = 1 } ^ { T ( \tau ) } \log \pi _ { \boldsymbol { \theta } } ( a _ { t } \mid s _ { t } , \boldsymbol { x } ) ,\tag{17}
$$

where $T ( \tau )$ denotes the number of decoding steps in trajectory $\tau .$

This summed log-probability can introduce a length bias in SCOPs. Since each per-step log-probability is non-positive, longer trajectories typically have more negative total log-probability even when their average per-step confidence is comparable to that of shorter trajectories. This can cause PO to favor shorter routes beyond the intended preference for higher-reward solutions.

To mitigate this effect, we use the following length-adjusted trajectory score:

$$
\ell _ { \theta } ( \tau \mid x ) = T ( \tau ) ^ { - \beta } \sum _ { t = 1 } ^ { T ( \tau ) } \log \pi _ { \theta } ( a _ { t } \mid s _ { t } , x ) , \qquad \beta \in [ 0 , 1 ] .\tag{18}
$$

This quantity is a scoring function and, for $\beta \neq 0$ , is not the log-probability of a normalized trajectory distribution. In particular, $\bar { \ell _ { \theta } } ( \tau \mid x ) = \bar { \log } \pi _ { \theta } ( \tau \mid x )$ when $\beta = 0$ , whereas $\beta = 1$ gives the average per-step log-probability.

In our experiments, we use $\beta = 0 . 3$ as a partial length correction in OP and KP, and use $\beta = 0 . 1$ in PCTSP and SPCTSP. This hyperparameter distinction reflects the underlying structural biases of the problem classes: the OP and KP inherently encourage visiting more nodes to maximize accumulated rewards within a budget, whereas the PCTSP and SPCTSP penalize routing costs and therefore favor shorter decision sequences. The resulting preference term is

$$
\log \sigma \left( \alpha \left[ \ell _ { \boldsymbol { \theta } } ( \tau _ { i } \mid \boldsymbol { x } ) - \ell _ { \boldsymbol { \theta } } ( \tau _ { j } \mid \boldsymbol { x } ) \right] \right) .\tag{19}
$$

## B.2 Motivation for Reward-Weighted Preference Optimization

Illustrative Example. Consider an extreme example with two preference pairs under the Bradley–Terry (BT) model. First, suppose $\tau _ { i } \succ \tau _ { j }$ with

$$
\log \pi _ { \theta } ( \tau _ { i } | x ) = \log \pi _ { \theta } ( \tau _ { j } | x ) + 0 . 0 1 , \quad r ( x , \tau _ { i } ) = r ( x , \tau _ { j } ) + 1 0 .
$$

Second, suppose $\tau _ { 3 } \succ \tau _ { 4 }$ with

$$
\log \pi _ { \theta } ( \tau _ { 3 } | x ) = \log \pi _ { \theta } ( \tau _ { 4 } | x ) - 5 0 , \quad r ( x , \tau _ { 3 } ) = r ( x , \tau _ { 4 } ) + 0 . 0 1 .
$$

Under the pairwise logistic model

$$
p _ { \theta } \big ( \tau _ { i } \succ \tau _ { j } \ | \ x \big ) = \sigma \big ( \alpha \big ( \log \pi _ { \theta } \big ( \tau _ { i } | x \big ) - \log \pi _ { \theta } \big ( \tau _ { j } | x \big ) \big ) \big ) ,
$$

with $\alpha = 0 . 0 5$ , we obtain

$$
\begin{array} { c } { p _ { \theta } ( \tau _ { i } \succ \tau _ { j } ) = \sigma ( 0 . 0 5 \times 0 . 0 1 ) \approx 0 . 5 0 , } \\ { p _ { \theta } ( \tau _ { 3 } \succ \tau _ { 4 } ) = \sigma ( 0 . 0 5 \times ( - 5 0 ) ) \approx 0 . 0 8 . } \end{array}
$$

Thus, the model assigns approximately 50% probability to the correct preference in the first pair and only $8 \%$ in the second pair. The corresponding log-likelihood terms are

$$
\begin{array} { r } { \log p _ { \theta } ( \tau _ { i } \succ \tau _ { j } ) \approx - 0 . 6 9 , \qquad \log p _ { \theta } ( \tau _ { 3 } \succ \tau _ { 4 } ) \approx - 2 . 5 8 . } \end{array}
$$

Since the gradient magnitude of the binary cross-entropy loss scales with $\sigma ( z ) - 1$ (for positive labels), the second pair produces a substantially larger gradient than the first. Consequently, the optimization strongly pushes the model to correct the misranking of $( \tau _ { 3 } , \tau _ { 4 } )$ , even though their reward gap is negligible, while only weakly encouraging preference between $( \tau _ { i } , \tau _ { j } )$ despite their large reward difference.

Fundamental Issue. This example reveals a structural limitation of the PO: the update magnitude depends solely on the model’s probability error, rather than on the reward gap between trajectories. As a result, pairs with small reward differences but large probability errors can dominate the gradient, whereas pairs with substantial reward gaps may induce only minor updates.

To tackle this issue, we employ a weighted pairwise logistic likelihood, scaling each comparison by a function of the reward gap to emphasize more informative preferences. This is conceptually aligned with reward-weighted likelihood/regression in policy learning [Peters and Schaal, 2007, Peng et al., 2019] and with importance-weighted pairwise losses in learning-to-rank (e.g., LambdaRank-style reweighting) [Burges et al., 2006].

## B.3 Gradient Analysis of RWPO Objectives

We analyze the gradient behavior of the ordered RWPO objective and the implemented confidence-regularized all-pair RWPO objective. For a sampled trajectory pair, define the normalized trajectory log-likelihood margin as

$$
m _ { \theta } ( x , \tau _ { i } , \tau _ { j } ) = \ell _ { \theta } ( \tau _ { i } | x ) - \ell _ { \theta } ( \tau _ { j } | x ) .\tag{20}
$$

For simplicity, let

$$
p _ { i j } = \sigma ( \alpha m _ { \theta } ( x , \tau _ { i } , \tau _ { j } ) ) .
$$

Gradient of ordered RWPO For an ordered preference pair $( \tau ^ { + } , \tau ^ { - } )$ with $r ( x , \tau ^ { + } ) > r ( x , \tau ^ { - } )$ , the RWPO loss is

$$
\mathcal { L } _ { \mathrm { R W P O } } = - \omega _ { + - } \log \sigma ( \alpha m _ { \theta } ) ,\tag{21}
$$

where

$$
\omega _ { + - } = 1 + \lambda \frac { r ( x , \tau ^ { + } ) - r ( x , \tau ^ { - } ) } { \bar { \Delta } ^ { ( b ) } + \epsilon } .\tag{22}
$$

Taking the gradient with respect to $\theta$ gives

$$
\begin{array} { r l } { \nabla _ { \theta } \mathcal { L } _ { \mathrm { R W P O } } = - \omega _ { + - } \nabla _ { \theta } \log \sigma ( \alpha m _ { \theta } ) } & { } \\ { = - \alpha \omega _ { + - } \sigma ( - \alpha m _ { \theta } ) \nabla _ { \theta } m _ { \theta } . } \end{array}\tag{23}
$$

Because $\omega _ { + - }$ increases monotonically with the reward gap, larger improvements between the preferred and rejected trajectories produce larger gradient magnitudes when the current preference margin is identical. Therefore, RWPO assigns stronger optimization pressure to comparisons with larger reward improvements while preserving the same preference probability mapping $\sigma ( \alpha m _ { \theta } )$ for all pairs as PO.

Gradient of confidence-regularized all-pair RWPO Unlike the ordered RWPO objective, which increases preference-separation pressure through reward-dependent gradients, the confidence-regularized all-pair RWPO objective induces a finite reward-gap-dependent optimal preference margin.

For an unordered rollout pair $\{ \tau _ { i } , \tau _ { j } \}$ , define

$$
\begin{array} { r } { m _ { i j } = \ell _ { \theta } ( \tau _ { i } | x ) - \ell _ { \theta } ( \tau _ { j } | x ) . } \end{array}
$$

The implemented all-pair objective for this pair is

$$
\mathcal { L } _ { i j } = - \tilde { \omega } _ { i j } \log \sigma ( \alpha m _ { i j } ) - \tilde { \omega } _ { j i } \log \sigma ( - \alpha m _ { i j } ) ,\tag{24}
$$

where

$$
\tilde { \omega } _ { i j } = 1 + \lambda \frac { [ r ( x , \tau _ { i } ) - r ( x , \tau _ { j } ) ] _ { + } } { \bar { \Delta } ^ { ( b ) } + \epsilon } ,\tag{25}
$$

and

$$
\tilde { \omega } _ { j i } = 1 + \lambda \frac { [ r ( x , \tau _ { j } ) - r ( x , \tau _ { i } ) ] _ { + } } { \bar { \Delta } ^ { ( b ) } + \epsilon } .\tag{26}
$$

The derivative with respect to the pairwise margin is

$$
\begin{array} { c } { { \frac { \partial \mathcal { L } _ { i j } } { \partial m _ { i j } } = - \alpha \tilde { \omega } _ { i j } \sigma ( - \alpha m _ { i j } ) + \alpha \tilde { \omega } _ { j i } \sigma ( \alpha m _ { i j } ) } } \\ { { { } } } \\ { { = \alpha \left[ ( \tilde { \omega } _ { i j } + \tilde { \omega } _ { j i } ) \sigma ( \alpha m _ { i j } ) - \tilde { \omega } _ { i j } \right] . } } \end{array}\tag{27}
$$

For a pair where $\tau _ { i }$ has higher reward than $\tau _ { j }$ ,

$$
r ( x , \tau _ { i } ) > r ( x , \tau _ { j } ) ,
$$

we have

$$
\tilde { \omega } _ { i j } = 1 + \lambda \frac { \Delta _ { i j } } { \bar { \Delta } ^ { ( b ) } + \epsilon } , \qquad \tilde { \omega } _ { j i } = 1 .
$$

Therefore,

$$
\frac { \partial \mathcal { L } _ { i j } } { \partial m _ { i j } } = \alpha \left[ \left( 2 + \lambda \frac { \Delta _ { i j } } { \bar { \Delta } ^ { ( b ) } + \epsilon } \right) \sigma ( \alpha m _ { i j } ) - \left( 1 + \lambda \frac { \Delta _ { i j } } { \bar { \Delta } ^ { ( b ) } + \epsilon } \right) \right] .\tag{28}
$$

The gradient consists of two components. The unit terms produce the symmetric preference regularization:

$$
\alpha ( 2 \sigma ( \alpha m _ { i j } ) - 1 ) ,
$$

which prevents unconstrained growth of the pairwise margin. The unit baseline alone does not learn preference ordering. It acts as a margin regularizer that prevents the model from pushing all pairwise differences to +∞. The reward-gap-dependent component is

$$
\lambda \frac { \Delta _ { i j } } { \bar { \Delta } ^ { ( b ) } + \epsilon } \alpha ( \sigma ( \alpha m _ { i j } ) - 1 ) ,
$$

which increases the preference pressure toward the higher-reward trajectory as the reward gap grows.

Proposition 1 (Reward-gap-dependent preference margin). For a fixed trajectory pair $( \tau _ { i } , \tau _ { j } )$ with $r ( x , \tau _ { i } ) > r ( x , \tau _ { j } )$ the confidence-regularized all-pair RWPO loss induces an optimal log-likelihood margin

$$
m _ { i j } ^ { * } = \frac { 1 } { \alpha } \log \left( 1 + \lambda \frac { r ( x , \tau _ { i } ) - r ( x , \tau _ { j } ) } { \bar { \Delta } ^ { ( b ) } + \epsilon } \right) ,\tag{29}
$$

where

$$
\begin{array} { r } { m _ { i j } = \ell _ { \theta } ( \tau _ { i } | x ) - \ell _ { \theta } ( \tau _ { j } | x ) . } \end{array}\tag{30}
$$

Therefore, RWPO assigns larger optimal preference margins to trajectory pairs with larger normalized reward gaps, while allowing near-tie pairs to remain weakly separated.

Proof. For a pair $( \tau _ { i } , \tau _ { j } )$ with $r ( x , \tau _ { i } ) > r ( x , \tau _ { j } )$ , as the derivative of $\mathcal { L } _ { i j }$ with respect to $m _ { i j }$ gives

$$
\frac { \partial \mathcal { L } _ { i j } } { \partial m _ { i j } } = \alpha \left[ ( \tilde { \omega } _ { i j } + 1 ) \sigma ( \alpha m _ { i j } ) - \tilde { \omega } _ { i j } \right] .\tag{31}
$$

Reward-Weighted Preference Optimization for Selective Combinatorial Optimization Problems

At the stationary point,

$$
\sigma ( \alpha m _ { i j } ^ { * } ) = \frac { \tilde { \omega } _ { i j } } { \tilde { \omega } _ { i j } + 1 } = \frac { 1 + \lambda \frac { \Delta _ { i j } } { \bar { \Delta } ^ { ( b ) } + \epsilon } } { 2 + \lambda \frac { \Delta _ { i j } } { \bar { \Delta } ^ { ( b ) } + \epsilon } } .\tag{32}
$$

Applying the inverse sigmoid function gives

$$
m _ { i j } ^ { * } = \frac { 1 } { \alpha } \log \left( 1 + \lambda \frac { \Delta _ { i j } } { \bar { \Delta } ^ { ( b ) } + \epsilon } \right) ,\tag{33}
$$

which proves the result.

Corollary 2 (Reward-aware preference separation). Under the confidence-regularized all-pair RWPO objective, the induced optimal preference margin is monotonically increasing with the reward gap:

$$
\frac { \partial m _ { i j } ^ { * } } { \partial \Delta _ { i j } } = \frac { \lambda } { \alpha ( \bar { \Delta } ^ { ( b ) } + \epsilon ) \left( 1 + \lambda _ { \bar { \Delta } ^ { ( b ) } + \epsilon } \right) } > 0 .
$$

Therefore, RWPO encourages stronger separation between trajectories with larger reward improvements while avoiding excessive separationfor near-equivalent solutions.

Consequently, the all-pair formulation preserves the stabilizing effect of symmetric pairwise learning while selectively increasing the margin between trajectories with larger objective improvements.

## B.4 Gradient-Matching Interpretation of Reward-Weighted Preference Optimization

The analysis below focuses on the reward-gap-weighted preferred-direction component of RWPO, which provides the direct connection to reward-weighted policy-gradient updates. The implemented confidence-regularized all-pair RWPO loss additionally includes the symmetric all-pair baseline, which serves as a confidence regularizer by controlling pairwise separation. Since this term does not correspond to a reward-weighted update direction, it is excluded from the coefficient-matching comparison with the policy-gradient formulation.

Fix an instance x and a policy $\pi _ { \boldsymbol { \theta } } ( \cdot | \boldsymbol { x } )$ . At a given training iteration, let trajectory pairs $( \tau _ { i } , \tau _ { j } )$ be sampled from the current policy (or another fixed sampling distribution). When differentiating the preference objective, we hold the sampled trajectories and their reward-dependent weights fixed, as in standard PO. For analytical tractability, we condition on a fixed realization of the discrete sampling support during differentiation. Equivalently, the sampling policy is treated as locally constant with respect to its support mask, so gradients are computed within a region of parameter space where the selected support does not change. Notably, this condition holds for top-k selective decoding provided the selected support mask remains fixed during the backward pass.

Write

$$
r _ { i } : = r ( x , \tau _ { i } ) , \qquad \ell _ { i } : = \log \pi _ { \theta } ( \tau _ { i } | x ) ,
$$

and define the reward and log-probability gaps as

$$
\Delta r : = r _ { i } - r _ { j } , \qquad \Delta \ell : = \ell _ { i } - \ell _ { j } .
$$

Here, $\ell _ { i }$ and $\ell _ { j }$ denote the standard trajectory log-probability. The length-normalized trajectory score introduced in Appendix B.1 is not considered in the following theoretical analysis; instead, we analyze the standard PO/RWPO scoring function based on trajectory log-probabilities.

The BT preference model induced by the implicit reward

$$
\hat { r } _ { \theta } ( x , \tau ) = \alpha \log \pi _ { \theta } ( \tau | x ) + \alpha \log Z ( x )
$$

defines the pairwise preference probability

$$
p _ { \theta } ( \tau _ { i } \succ \tau _ { j } | x ) = \sigma ( \hat { r } _ { \theta } ( x , \tau _ { i } ) - \hat { r } _ { \theta } ( x , \tau _ { j } ) ) = \sigma ( \alpha \Delta \ell ) ,
$$

where $\sigma ( u ) = 1 / ( 1 + e ^ { - u } )$

Let $\omega ( \tau _ { i } , \tau _ { j } , x ) \ge 0$ be a pairwise weight that is independent of θ. For a fixed pair-sampling distribution $q ,$ the weighted preference objective is

$$
\mathcal { L } _ { \omega } ^ { q } ( \theta ) = \mathbb { E } _ { x ; ( \tau _ { i } , \tau _ { j } ) \sim q } \left[ \mathbb { I } ( \Delta r > 0 ) \omega ( \tau _ { i } , \tau _ { j } , x ) \log \sigma ( \alpha \Delta \ell ) \right] .\tag{34}
$$

During gradient computation, q is treated as fixed and corresponds to the distribution induced by the sampled rollout pairs.

For comparison, consider the standard expected-reward objective:

$$
J ( \theta ) = \mathbb { E } _ { x , \tau \sim \pi _ { \theta } ( \cdot | x ) } [ r ( x , \tau ) ] .
$$

## B.4.1 Pairwise Representation of the Policy Gradient

We assume that the reward function is independent of $\theta ,$ as in standard policy-gradient optimization.

Lemma 1 (Pairwise representation of the policy gradient). For any fixed instance $x ,$

$$
\nabla _ { \boldsymbol { \theta } } \mathbb { E } _ { \tau \sim \pi _ { \boldsymbol { \theta } } ( . | x ) } [ r ( x , \tau ) ] = \frac { 1 } { 2 } \mathbb { E } _ { \tau _ { i } , \tau _ { j } \sim \pi _ { \boldsymbol { \theta } } } \left[ \Delta r ( \nabla _ { \boldsymbol { \theta } } \ell _ { i } - \nabla _ { \boldsymbol { \theta } } \ell _ { j } ) \right] .\tag{35}
$$

Equivalently, when ties $\Delta r = 0$ have zero contribution,

$$
\nabla _ { \theta } \mathbb { E } _ { \tau \sim \pi _ { \theta } ( \cdot | x ) } [ r ( x , \tau ) ] = \mathbb { E } _ { \tau _ { i } , \tau _ { j } \sim \pi _ { \theta } } \left[ \mathbb { I } ( \Delta r > 0 ) \Delta r ( \nabla _ { \theta } \ell _ { i } - \nabla _ { \theta } \ell _ { j } ) \right] .\tag{36}
$$

Proof. The likelihood-ratio identity gives

$$
\nabla _ { \theta } \mathbb { E } _ { \tau \sim \pi _ { \theta } ( \cdot | x ) } [ r ( x , \tau ) ] = \mathbb { E } _ { \tau \sim \pi _ { \theta } ( \cdot | x ) } [ r ( x , \tau ) \nabla _ { \theta } \ell ( \tau ) ] .
$$

For two independent trajectories,

$$
\begin{array} { r l } & { \mathbb { E } [ ( r _ { i } - r _ { j } ) ( \nabla _ { \theta } \ell _ { i } - \nabla _ { \theta } \ell _ { j } ) ] } \\ & { \ = \mathbb { E } [ r _ { i } \nabla _ { \theta } \ell _ { i } ] - \mathbb { E } [ r _ { i } \nabla _ { \theta } \ell _ { j } ] - \mathbb { E } [ r _ { j } \nabla _ { \theta } \ell _ { i } ] + \mathbb { E } [ r _ { j } \nabla _ { \theta } \ell _ { j } ] . } \end{array}
$$

Because the two trajectories $\tau _ { i }$ and $\tau _ { j }$ are independent and

$$
\mathbb { E } _ { \tau \sim \pi _ { \theta } } [ \nabla _ { \theta } \ell ( \tau ) ] = \nabla _ { \theta } \sum _ { \tau } \pi _ { \theta } ( \tau | x ) = \nabla _ { \theta } 1 = 0 ,
$$

based on the assumption that the support of $\pi _ { \theta }$ does not change when differentiating, the cross terms vanish:

$$
\begin{array} { r } { \mathbb { E } [ r _ { i } \nabla _ { \theta } \ell _ { j } ] = \mathbb { E } [ r _ { i } ] \mathbb { E } [ \nabla _ { \theta } \ell _ { j } ] = 0 , } \end{array}
$$

and similarly for $\mathbb { E } [ r _ { j } \nabla _ { \theta } \ell _ { i } ]$ . Therefore,

$$
\begin{array} { r } { \mathbb { E } [ ( r _ { i } - r _ { j } ) ( \nabla _ { \theta } \ell _ { i } - \nabla _ { \theta } \ell _ { j } ) ] = 2 \mathbb { E } [ r ( x , \tau ) \nabla _ { \theta } \ell ( \tau ) ] , } \end{array}
$$

which proves Eq. (35).

For Eq. (36), define

$$
g ( \tau _ { i } , \tau _ { j } ) = \Delta r ( \nabla _ { \theta } \ell _ { i } - \nabla _ { \theta } \ell _ { j } ) .
$$

Since both $\Delta r$ and the gradient difference change sign when exchanging $\tau _ { i }$ and $\tau _ { j }$ , we have

$$
g ( \tau _ { i } , \tau _ { j } ) = g ( \tau _ { j } , \tau _ { i } ) .
$$

Therefore, positive and negative reward-gap pairs contribute equally, and restricting the expectation to $\Delta r > 0$ gives the equivalent preferred-direction representation. □

## B.4.2 Policy-Score Update Induced by Weighted Preference Optimization

Lemma 2 (Gradient of the weighted preference loss). For the fixed pair-sampling distribution $q ,$ the policy-score gradient of (34) is

$$
\begin{array} { r l } & { \nabla _ { \theta } \mathcal { L } _ { \omega } ^ { q } ( \theta ) = } \\ & { \alpha \mathbb { E } _ { x , ( \tau _ { i } , \tau _ { j } ) \sim q } \left[ \mathbb { I } ( \Delta r > 0 ) \omega ( \tau _ { i } , \tau _ { j } , x ) \sigma ( - \alpha \Delta \ell ) \left( \nabla _ { \theta } \ell _ { i } - \nabla _ { \theta } \ell _ { j } \right) \right] . } \end{array}\tag{37}
$$

Proof. Let $u = \alpha \Delta \ell .$ . Since

$$
\frac { d } { d u } \log \sigma ( u ) = \sigma ( - u ) ,
$$

we have

$$
\nabla _ { \boldsymbol { \theta } } \log \sigma ( \alpha \Delta \ell ) = \alpha \sigma ( - \alpha \Delta \ell ) \left( \nabla _ { \boldsymbol { \theta } } \ell _ { i } - \nabla _ { \boldsymbol { \theta } } \ell _ { j } \right) .
$$

## B.4.3 Coefficient-Matching Bound

The pairwise RL ascent direction in (36) and the preference-learning update in (37) share the same feature direction

$$
\begin{array} { r } { \boldsymbol { g } : = \nabla _ { \boldsymbol { \theta } } \ell _ { 1 } - \nabla _ { \boldsymbol { \theta } } \ell _ { 2 } . } \end{array}
$$

They differ in the scalar coefficient multiplying this direction.

When the pair-sampling distribution is the current policy, $q = \pi _ { \theta } \times \pi _ { \theta }$ , define

$$
c _ { \mathrm { R L } } : = \mathbb { I } ( \Delta r > 0 ) \Delta r , \qquad c _ { \omega } : = \alpha \mathbb { I } ( \Delta r > 0 ) \omega \sigma ( - \alpha \Delta \ell ) .
$$

Then

$$
\begin{array} { r } { \nabla _ { \theta } J ( \theta ) = \mathbb { E } [ c _ { \mathrm { { R L } } } g ] , \qquad \nabla _ { \theta } \mathcal { L } _ { \omega } ^ { q } ( \theta ) = \mathbb { E } [ c _ { \omega } g ] . } \end{array}
$$

Theorem 1 (Coefficient-matching bound). Assume there exists $G < \infty$ such that

$$
\| g \| \leq G
$$

almost surely. Then

$$
\begin{array} { r } { \| \nabla _ { \theta } J ( \theta ) - \nabla _ { \theta } \mathcal { L } _ { \omega } ^ { q } ( \theta ) \| \leq G \operatorname { \mathbb { E } } \left[ | c _ { \mathrm { { R L } } } - c _ { \omega } | \right] . } \end{array}\tag{38}
$$

Moreover,

$$
\begin{array} { r } { \| \nabla _ { \theta } J ( \theta ) - \nabla _ { \theta } \mathcal { L } _ { \omega } ^ { q } ( \theta ) \| \leq G \sqrt { { \mathbb E } \left[ ( c _ { \mathrm { { R L } } } - c _ { \omega } ) ^ { 2 } \right] } . } \end{array}\tag{39}
$$

Proof. By the pairwise representations above,

$$
\begin{array} { r } { \nabla _ { \theta } J - \nabla _ { \theta } \mathcal { L } _ { \omega } ^ { q } = \mathbb { E } [ ( c _ { \mathrm { { R L } } } - c _ { \omega } ) g ] . } \end{array}
$$

$$
\begin{array} { r l r } & { } & { \| \nabla _ { \theta } J - \nabla _ { \theta } \mathcal { L } _ { \omega } ^ { q } \| \leq \mathbb { E } [ | c _ { \mathrm { R L } } - c _ { \omega } | \| g \| ] } \\ & { } & { \leq G \mathbb { E } [ | c _ { \mathrm { R L } } - c _ { \omega } | ] , } \end{array}
$$

which proves (38). Applying Cauchy–Schwarz,

$$
\mathbb { E } [ | X | ] \leq { \sqrt { \mathbb { E } [ X ^ { 2 } ] } } ,
$$

gives (39).

The bound shows that reward weighting can be interpreted as selecting the scalar coefficient multiplying each pairwise policy-score direction. The RL update gives coefficient $\Delta r$ for preferred pairs, whereas the BT preference update gives αω $\overset { \cdot } { \sigma } ( - \alpha \Delta \ell )$

Thus, reward-gap weighting directly bridges preference learning and policy gradients. By incorporating ∆r-dependent information into ω, the preference coefficient αω $\sigma ( - \alpha \Delta \ell )$ adapts to the standard RL reward scaling. Consequently, reducing the error term

$$
| \Delta r - \alpha \omega \sigma ( - \alpha \Delta \ell ) |
$$

steers the resulting update toward the policy-gradient direction, serving as a gradient-matching motivation rather than a strict convergence guarantee."

## B.5 Tilted Sampling

Let $\pi _ { \theta _ { \mathrm { b a s e } } } ( \tau \mid x )$ denote a pretrained base policy, where x is an instance and τ is a generated trajectory. Following the distinction in Mittal et al. [2026], we first separate task-reward optimization from distribution sharpening at the level of their ideal objectives. Pure task-reward optimization directly seeks trajectories with high task reward:

$$
\pi _ { \theta } ^ { * } = \arg \operatorname* { m a x } _ { \tau } r ( x , \tau ) ,
$$

where $r ( x , \tau )$ denotes the task-specific reward. This objective is driven only by solution quality and does not explicitly account for whether the selected trajectory is likely under the pretrained policy.

In contrast, pure distribution-sharpening optimization seeks trajectories that are most likely under the base policy:

$$
\pi _ { \theta } ^ { * } = \arg \operatorname* { m a x } _ { \tau } \pi _ { \theta _ { \mathrm { b a s e } } } ( \tau \mid x ) ,
$$

or equivalently,

$$
\pi _ { \theta } ^ { * } = \arg \operatorname* { m a x } _ { \tau } \log \pi _ { \theta _ { \mathrm { b a s e } } } ( \tau \mid x ) .
$$

This objective does not use task reward directly. Instead, it assumes that the base policy already assigns higher likelihood to better trajectories, and therefore attempts to improve performance by concentrating probability mass on the base model’s existing high-likelihood modes.

A softer distributional version of this idea is tempered sampling [Hu et al., 2024]. Rather than collapsing onto the single most likely trajectory, tempered sampling defines a sharpened target distribution

$$
q _ { \mathrm { t e m p } } ( \tau \mid x ) \propto \pi _ { \theta _ { \mathrm { b a s e } } } ( \tau \mid x ) ^ { \alpha } , \qquad \alpha > 1 .
$$

Equivalently, this is a low-temperature version of the base policy with temperature $T = 1 / \alpha < 1$

$$
q _ { \mathrm { t e m p } } ( \tau \mid x ) \propto \pi _ { \theta _ { \mathrm { b a s e } } } ( \tau \mid x ) ^ { 1 / T } .
$$

Thus, distribution-sharpening optimization and tempered sampling share the same underlying principle: both favor trajectories that are already likely under the base policy. The former corresponds to a hard maximization view, while the latter gives a softened distributional version. However, both approaches improve task performance only when base-policy likelihood is well aligned with solution quality. When this alignment is weak, sharpening may reinforce trajectories that are plausible under the base model but not necessarily high-reward.

Tilted sampling combines the two signals. It keeps the base-policy likelihood as a conservative anchor while also incorporating task reward. The ideal tilted target is

$$
q _ { \mathrm { t i l t } } ( \tau \mid x ) \propto \pi _ { \theta _ { \mathrm { b a s e } } } ( \tau \mid x ) ^ { \alpha } \exp \left( \beta ^ { - 1 } r ( x , \tau ) \right) .
$$

Note that, $\alpha = 1$ corresponds to normal KL-regularized RL fine-tuning [Guo et al., 2025].

Taking logarithms, this target assigns each trajectory a combined score:

$$
\begin{array} { r } { \log q _ { \mathrm { t i l t } } ( \tau \mid x ) = \alpha \log \pi _ { \theta _ { \mathrm { b a s e } } } ( \tau \mid x ) + \beta ^ { - 1 } r ( x , \tau ) - \log z ( x ) , } \end{array}
$$

where $z ( x )$ is the instance-level normalizing constant. Therefore, tilted sampling can be interpreted as a reward-guided form of distribution sharpening: the reward term favors high-quality trajectories, while the base-likelihood term keeps the adapted policy close to trajectories that remain plausible under the pretrained model. In neural combinatorial optimization, the full trajectory space is combinatorially large, making the exact normalizer $z ( x )$ intractable. We therefore approximate the tilted target over a finite candidate set

$$
\begin{array} { r } { \boldsymbol { \mathcal { C } } ( \boldsymbol { x } ) = \{ \tau _ { 1 } , \dots , \tau _ { K } \} , \qquad \tau _ { k } \sim \pi _ { \boldsymbol { \theta } } ( \cdot \mid \boldsymbol { x } ) , } \end{array}
$$

sampled from the current policy. To make the reward and likelihood terms comparable across candidate groups, we normalize them within each group:

$$
\widetilde { r } ( x , \tau _ { k } ) = \frac { r ( x , \tau _ { k } ) - \mu _ { r } } { \sigma _ { r } + \epsilon } ,
$$

and

$$
\widetilde { \ell } _ { \mathrm { b a s e } } ( \tau _ { k } ) = \frac { \log \pi _ { \theta _ { \mathrm { b a s e } } } ( \tau _ { k } \mid x ) - \mu _ { \ell } } { \sigma _ { \ell } + \epsilon } .
$$

The finite-candidate tilted score is then defined as

$$
\begin{array} { r } { \varsigma _ { k } = \lambda _ { r } \widetilde { r } ( \boldsymbol { x } , \tau _ { k } ) + \lambda _ { p } \widetilde { \ell } _ { \mathrm { b a s e } } ( \tau _ { k } ) , } \end{array}
$$

where $\lambda _ { r }$ controls the contribution of the task reward and $\lambda _ { p }$ controls the strength of the base-policy likelihood anchor. We convert these scores into normalized candidate weights using a softmax temperature $T _ { \omega }$

$$
w _ { k } = \frac { \exp ( \varsigma _ { k } / T _ { \omega } ) } { \sum _ { j = 1 } ^ { K } \exp ( \varsigma _ { j } / T _ { \omega } ) } .
$$

The adapted policy is then trained by weighted behavioural cloning:

$$
\mathcal { L } _ { \mathrm { t i l t } } ( \theta ) = - \mathbb { E } _ { x } \left[ \sum _ { k = 1 } ^ { K } w _ { k } \log \pi _ { \theta } ( \tau _ { k } \mid x ) \right] .
$$

This objective can be viewed as a sampled approximation to the ideal tilted distribution. When $\lambda _ { r } = 0 ,$ , it reduces to base-policy sharpening or tempered imitation, concentrating probability mass on trajectories that the base policy already considers likely. When $\lambda _ { s } = 0 ,$ it becomes reward-only adaptation over the sampled candidate set. When both terms are active, the objective encourages task-reward improvement while retaining a conservative connection to the pretrained policy. In this sense, tilted sampling provides a controlled interpolation between distribution sharpening and reward-driven adaptation.

## C Baseline Method Details and Hyperparameters

To aid reproducibility, we provide further details of the methods compared in Section 4 of the main paper, and the datasets used.

## C.1 Baselines

Classical solvers. For LKH3, we adopt the commonly used configuration in the machine learning literature [Kool et al., 2019] and implement the TSP and CVRP baselines based on the code from https://github.com/wouterkool/ attention-learn-to-route. For KP, we use OR-Tools and follow the implementation provided in the official Google OR-Tools documentation: https://developers.google.com/optimization/pack/knapsack. For OP, we use Compass [Kobeaga et al., 2018], implemented from https://github.com/gkobeaga/op-solver, as the main classical baseline.

RL-based constructive solvers. We compare against representative RL-based constructive solvers, including AM [Kool et al., 2019], POMO [Kwon et al., 2020], ICAM [Zhou et al., 2024a], MDAM [Xin et al., 2021], Sym-NCO [Kim et al., 2022] and PO Pan et al. [2025a]. For POMO, PO, and ICAM, we use multi-start decoding with M = N and 8× augmentation on non-SCOP instances with size N ≤ 1,000, and multi-start decoding without augmentation for instances with N > 1,000. For SCOPs, we use multi-start decoding with M = N and no augmentation. For Sym-NCO, we adopt 200 augmentations for COP instances with N ≤ 1,000 and 100 augmentations for 2,000-node instances. Unless otherwise specified, we use the pretrained checkpoints released by the original papers for all neural baselines except POMO and PO, which we train under our unified experimental setup.

Neural divide-and-conquer methods. We also include neural divide-and-conquer baselines, including GLOP [Ye et al., 2024] and UDC [Zheng et al., 2024], to compare against methods specifically designed for large-scale combinatorial optimization.

## C.2 Dataset

OPLib benchmark. To evaluate out-of-distribution generalization on real benchmark instances, we use OPLib, a public benchmark library collected from classical OP instance families and released by BCAM Data Science https://github.com/bcamath-ds/OPLib. The repository provides instance files organized by generation setting, where different generations correspond to different ways of assigning node rewards. In our experiments, we focus on Generation 3 instances because their reward-generation rule is closest to the reward distribution used to train our neural policy. This choice allows us to evaluate scale and structural generalization on standard benchmark instances while avoiding an additional reward-distribution shift that would confound the comparison. Therefore, performance on Generation 3 primarily reflects whether the learned policy can transfer from synthetic training instances to OPLib geometry and budget structures under a compatible reward distribution.

## C.3 Hyperparameters Setting

Tab. 3 presents the parameter settings for the three training frameworks: POMO, PO, and RWPO. The attention layer of RWPO is the same as for POMO and PO.

## D Further Experimental Details

We provide further details of the experiments of Section 4 of the main paper.

## D.1 Soft Entropy Estimation

In addition to policy entropy, which is computed directly from the decoder action distribution, we use a representation level entropy estimator to quantify how information is distributed in the hidden embedding space. Directly estimating the entropy of high-dimensional continuous neural representations is challenging: differential entropy is sensitive to scaling and coordinate transformations, while hard discretization or clustering-based estimators can be computationally expensive and difficult to apply at scale. We therefore adopt the soft entropy estimator of Conklin et al. [2026], which can be interpreted as a differentiable relaxation of binning-based Shannon entropy. The method softly quantizes hidden representations into a fixed set of reference directions on the unit hypersphere and then computes the Shannon entropy of the resulting categorical distribution.

Table 3: Hyperparameter setting for POMO, PO, and RWPO.
<table><tr><td></td><td>CVRP100</td><td>OP100</td><td>PCTSP100/SPCTSP100</td><td>KP200</td></tr><tr><td>Alpha</td><td>0.03</td><td>0.05</td><td>0.05</td><td>0.05</td></tr><tr><td>Preference Function</td><td>Exponential</td><td>BT</td><td>BT</td><td>BT</td></tr><tr><td>Length normalization</td><td>0</td><td>0.3</td><td>0.1</td><td>0.3</td></tr><tr><td>RWPO Lambda</td><td>0.5</td><td>0.5</td><td>1</td><td>0.5</td></tr><tr><td>Epochs</td><td>4000</td><td>1000</td><td>500</td><td>500</td></tr><tr><td>Epochs (FT)</td><td>200</td><td>200</td><td>100</td><td>100</td></tr><tr><td>Epochs (TS)</td><td>100</td><td>100</td><td>100</td><td>100</td></tr><tr><td>Epoch (FT) Size</td><td>50000</td><td>1280000</td><td>1280000</td><td>1280000</td></tr><tr><td>Epoch (TS) Size</td><td>51200</td><td>51200</td><td>51200</td><td>51200</td></tr><tr><td>Encoder Layer Number</td><td>6</td><td>6</td><td>6</td><td>6</td></tr><tr><td>Batch Size</td><td>64</td><td>64</td><td>64</td><td>64</td></tr><tr><td>Embedding Dimension</td><td>128</td><td>128</td><td>128</td><td>128</td></tr><tr><td>Attention Head Number</td><td>8</td><td>8</td><td>8</td><td>8</td></tr><tr><td>Feed Forward Dimension</td><td>512</td><td>512</td><td>512</td><td>512</td></tr><tr><td>Tanh Clip</td><td>50</td><td>50</td><td>50</td><td>50</td></tr><tr><td>Learning Rate</td><td>3e-4</td><td>3e-4</td><td>3e-4</td><td>3e-4</td></tr><tr><td>Small Learning Rate</td><td>3e-5</td><td>3e-5</td><td>3e-5</td><td>3e-5</td></tr><tr><td>Norm Type</td><td>InstanceNorm</td><td>LayerNorm</td><td>LayerNorm</td><td>LayerNorm</td></tr></table>

Let $Z \in \mathbb { R } ^ { M \times d }$ denote a collection of hidden representations, where M is the number of representation vectors and d is the hidden dimension. In our experiments, $Z$ can correspond to encoder node embeddings, decoder query embeddings, or other intermediate representations collected over instances and decoding steps. The first step is to normalize each representation vector onto the unit sphere:

$$
\bar { z } _ { m } = \frac { z _ { m } } { \| z _ { m } \| _ { 2 } + \epsilon _ { \mathrm { n o r m } } } , \qquad m = 1 , \ldots , M ,\tag{40}
$$

where $\epsilon _ { \mathrm { n o r m } }$ is a small constant for numerical stability. We then sample K reference anchors $\{ w _ { k } \} _ { k = 1 } ^ { K }$ uniformly at random from the unit sphere $\mathbb { S } ^ { d }$ . Equivalently, this can be implemented by sampling $\tilde { w } _ { k } \sim \mathcal N ( 0 , I _ { d } )$ and normalizing:

$$
w _ { k } = \frac { \tilde { w } _ { k } } { \lVert \tilde { w } _ { k } \rVert _ { 2 } } .\tag{41}
$$

Let $W = [ w _ { 1 } , \dots , w _ { K } ] \in \mathbb { R } ^ { d \times K }$ denote the matrix of anchor directions.

For each normalized representation $\bar { z } _ { m }$ , we compute its cosine similarity to all anchors and apply a temperature-scaled softmax:

$$
\check { z } _ { m , k } = \frac { \exp \left( \bar { z } _ { m } ^ { \top } w _ { k } / T \right) } { \sum _ { j = 1 } ^ { K } \exp \left( \bar { z } _ { m } ^ { \top } w _ { j } / T \right) } , \qquad k = 1 , \ldots , K ,\tag{42}
$$

where temperature parameter $T > 0$ controls the sharpness of the soft assignment. A smaller T makes the assignment closer to hard nearest-anchor quantization, while a larger $T$ produces smoother assignments. Thus, each representation vector is mapped to a probability vector $\check { z } _ { m } \in \Delta ^ { K - 1 }$ over the anchor set.

The empirical soft-quantized distribution over anchors is obtained by averaging the assignments over all representations:

$$
\hat { p } _ { k } = \frac { 1 } { M } \sum _ { m = 1 } ^ { M } \check { z } _ { m , k } , \qquad k = 1 , \ldots , K .\tag{43}
$$

Since $\hat { p }$ is a categorical distribution over K anchors, its Shannon entropy is

$$
\mathcal { H } _ { \mathrm { s o f t } } ( Z ) = - \sum _ { k = 1 } ^ { K } \hat { p } _ { k } \log \hat { p } _ { k } .\tag{44}
$$

For easier comparison across different choices of K, we report the normalized soft entropy

$$
\bar { \mathcal { H } } _ { \mathrm { s o f t } } ( Z ) = \frac { \mathcal { H } _ { \mathrm { s o f t } } ( Z ) } { \log K } ,\tag{45}
$$

which lies in [0, 1]. A value close to 1 indicates that the representations are spread relatively evenly across anchor directions, while a smaller value indicates that the representation mass is concentrated around fewer directions.

When representations are collected layer-wise, the same computation can be applied independently to each layer. Let $Z ^ { ( \ell ) }$ denote the representations from layer $\ell = 1 , \ldots , L$ . We compute the layer-wise soft entropy $\bar { \mathcal { H } } _ { \mathrm { s o f t } } ( Z ^ { ( \ell ) } )$ and report the average:

$$
\bar { \mathcal { H } } _ { \mathrm { s o f t } } ^ { \mathrm { m o d e l } } = \frac { 1 } { L } \sum _ { \ell = 1 } ^ { L } \bar { \mathcal { H } } _ { \mathrm { s o f t } } \mathopen { } \mathclose \bgroup \left( Z ^ { ( \ell ) } \aftergroup \egroup \right) .\tag{46}
$$

In our analysis, we apply this estimator separately to encoder and decoder representations. Encoder soft entropy measures how broadly instance-level information is distributed across representation directions, while decoder soft entropy characterizes the diversity of the decision representation used to construct the solution.

Algorithm. The full procedure is summarized below.

```latex
Algorithm 2 Soft entropy estimation for hidden representations
Require: Hidden representations $Z = \{ z _ { m } \} _ { m = 1 } ^ { M }$ , number of anchors $K ,$ , temperature $\tau$
1: Normalize each representation: $\bar { z } _ { m } \gets z _ { m } / ( \lVert z _ { m } \rVert _ { 2 } + \epsilon _ { \mathrm { n o r m } } )$
2: Sample anchors $\tilde { w } _ { k } \sim \mathcal { N } ( 0 , I _ { d } )$ for $k = 1 , \ldots , K$
3: Normalize anchors: $w _ { k }  \tilde { w } _ { k } / \| \tilde { w } _ { k } \| _ { 2 }$
4: for $m = 1 , \ldots , M$ do
5: Compute cosine similarities $s _ { m , k } \gets \bar { z } _ { m } ^ { \top } w _ { k }$
6: Compute soft assignment:
$\breve { z } _ { m , k } \gets \frac { \exp ( s _ { m , k } / \tau ) } { \sum _ { j = 1 } ^ { K } \exp ( s _ { m , j } / \tau ) }$
7: end for
8: Aggregate assignments: $\begin{array} { r } { \hat { p } _ { k } \gets \frac { 1 } { M } \sum _ { m = 1 } ^ { M } \check { z } _ { m , k } } \end{array}$
9: Compute entropy: $\begin{array} { r } { \mathcal { H } _ { \mathrm { s o f t } } ( Z )  - \sum _ { k = 1 } ^ { K } \hat { p } _ { k } } \end{array}$ log ˆp<sub>k</sub>
10: Normalize: $\bar { H } _ { \mathrm { s o f t } } ( Z )  \mathcal { H } _ { \mathrm { s o f t } } ( Z ) / \log K$
11: return $\bar { H } _ { \mathrm { s o f t } } ( Z )$
```

Soft entropy estimator. We employ soft entropy as a representation-level estimator for continuous neural embeddings. Unlike dimension-wise discretization, soft entropy yields a smooth Shannon-entropy estimate without relying on hard binning, thereby circumventing the instability and memory overhead inherent to high-dimensional spaces. Furthermore, compared to clustering-based quantization, it eliminates the need for iterative centroid fitting. Anchor directions are sampled once and reused, reducing the entire estimation process to highly efficient cosine similarities followed by a softmax operation. Crucially, because vectors are $L _ { 2 }$ -normalized prior to estimation, this metric specifically captures the angular diversity of the latent space. This is particularly suitable for Transformer-based models, where high-dimensional embedding geometry and attention mechanisms rely heavily on relationships among continuous vector representations.

## D.2 Interpretation of Representation and Policy Entropy

In this work, soft entropy complements policy entropy by characterizing different aspects of the learned solver. Policy entropy measures behavioural uncertainty in the decoder action distribution,

$$
\mathcal { H } _ { \mathrm { p o l i c y } } ( \pi _ { t } ) = - \sum _ { a \in \mathcal { A } _ { t } } \pi ( a | s _ { t } ) \log \pi ( a | s _ { t } ) ,
$$

whereas soft entropy measures the diversity of continuous hidden representations. For encoder representations, soft entropy reflects whether the model preserves diverse instance-level information before decoding. For decoder queries, it characterizes the diversity of decision-state representations used to construct the next action. In particular, the decoder query

$$
q _ { t } = f ( h _ { \mathrm { m e a n } } , h _ { \mathrm { c u r r e n t } } , h _ { \mathrm { e n d } } , B _ { t } )
$$

summarizes the current partial solution, global context, depot/end information, and remaining budget. Therefore, decoder-query soft entropy measures how diverse or compressed the internal decision state is before interacting with candidate actions.

This distinction separates two effects: whether the model retains rich information about the input instance and whether the decoder maintains an expressive representation of the evolving solution state. High decoder-query soft entropy should not be interpreted as behavioural uncertainty; rather, it indicates diversity in the latent decision-state representation. Our analyses in Section 4.3 and Appendix E.3 shows that RWPO preserves richer encoder representations while maintaining structured decoder-query representations, suggesting that reward-weighted preference learning improves generalization through more informative representations.

Decoder hidden-state entropy, as studied in Conklin et al. [2026], provides a complementary measure of representation compression. However, these hidden states are computed after the decoder query has interacted with encoded candidate information through attention layers. They therefore combine decision-state information with candidate-dependent features and are more dependent on decoder architecture. We use decoder hidden-state entropy as a secondary representation metric rather than the primary measure of route-level decision-state diversity.

Policy entropy remains the most direct measure of behavioural uncertainty, but it is affected by the size of the feasible action set because its maximum value scales with log $| \mathcal { A } _ { t } |$ |. Therefore, when comparing action-level uncertainty across states, we additionally consider normalized or residual entropy to reduce the effect of varying support sizes.

## D.3 Mutual-Information Analysis

While soft entropy measures representation diversity, it does not reveal whether this diversity corresponds to taskrelevant information. We therefore complement our entropy analysis with a mutual-information analysis between learned representations and OP-specific labels (Tab. 4). For encoder representations, we consider static instance properties, including reward, depot distance, reward–distance structure, spatial region, and top-rollout membership, to evaluate whether the model captures reward and geometric information before decoding. For decoder-query representations, we consider dynamic decision-state variables, including accumulated reward, current distance, remaining budget, feasible action count, budget–feasibility interactions, reward quantiles, and top-versus-bottom trajectory membership.

This analysis distinguishes meaningful task-aware representations from merely diffuse features. In particular, higher mutual information between decoder queries and feasibility or preference-related labels indicates that representation diversity corresponds to structured decision information: the policy can maintain flexibility over candidate actions while still encoding constraints, remaining resources, and solution quality signals. Therefore, the entropy and mutualinformation analyses together characterize whether RWPO induces informative exploration rather than unstructured randomness.

## E Additional Experimental Results

We provide further experiments, not reported in Section 4 of the main paper.

## E.1 Comparison with Existing Algorithms on CVRP

Tab. 5 evaluates zero-shot scale generalization (OOD) and target-scale adaptation (ID) on CVRP instances. In the OOD setting, traditional neural solvers suffer severe performance degradation as instance size increases; specifically, the optimality gaps of PO and POMO compound rapidly at $N \geq 5 0 0$ . By anchoring preference updates to the true reward gap, RWPO successfully mitigates this degradation. This advantage becomes more pronounced at larger scales: compared to standard PO, RWPO reduces the optimality gap from 17.53% to 14.10% at $\mathrm { \bar { \mathit { N } } } = 5 0 0$ , from 45.78% to 38.35% at $N = 1 { , } 0 0 0$ , and from 108.09% to 93.26% at $\bar { N } = 2 \mathrm { , 0 0 0 }$

Following scale-specific adaptation with tilted sampling, RWPO+TS achieves performance comparable to dedicated large-scale solvers. By restricting optimality gaps to 2.45%, 4.40%, and 7.21% on $N \in \{ 5 0 0 , 1 \bar { 0 0 0 } , 2 0 0 0 \}$ instances, respectively, RWPO+TS drastically improves upon the directly transferred base policies. These results indicate that reward-weighted preference learning provides a highly robust initialization for scale adaptation, which TS subsequently refines. Finally, in the ID setting, RWPO+TS remains strictly competitive with specialized methods such as ICAM and UDC, yielding near-optimal performance without relying on the extensive data augmentation pipelines required by those baselines.

## E.2 Comparison Among Fine-tuning Methods

## E.2.1 Comparison Between Local Search Fine-tuning with Scale-specific Finetuning and Tilted Sampling

The likelihood–quality frontiers in Fig. 6 reveal a scale-dependent adaptation pattern. Each point represents a sampled trajectory, where the horizontal axis is the length-normalized log-likelihood assigned by the base RWPO policy and the vertical axis is the objective value. A higher log-likelihood, i.e., a value closer to zero, indicates that the trajectory is more probable and more consistent with the base RWPO policy’s existing preferences. A lower log-likelihood, i.e., a more negative value, indicates that the trajectory is less likely under the base RWPO policy and therefore lies farther from the policy’s original high-probability region. Note that in this section, the base RWPO represents the model trained without the small learning-rate continued (scale-specific) post-training; RWPO-FT represents the model with continued post-training.

Table 4: Task-relevant labels used for mutual-information analysis on OP. The labels cover static instance properties, dynamic decoding states, and preference-level solution-quality signals.
<table><tr><td>Category</td><td>Label</td><td>Definition and motivation</td></tr><tr><td rowspan="6">Encoder-side static instance labels</td><td>Depot distance</td><td> $d _ { \mathrm { d e p o t } } ( i ) = d ( i , 0 )$  Measures the geometric accessibility of node i from the depot. Nodes far from the depot is costly to visit even if they have high reward.</td></tr><tr><td>Depot feasible</td><td> $\mathbb { I } ( D _ { t } + d ( v _ { t } , i ) + d ( i , 0 ) \le B )$  Indicates whether node i can be visited while still allowing return to the depot. This tests whether representations encode feasibility in addition to reward attractiveness.</td></tr><tr><td>Node type</td><td>A categorical label distinguishing depot, selected nodes, unvisited candidates, and infeasible nodes. This tests whether the model separates structurally different roles.</td></tr><tr><td>Reward</td><td> $r _ { i \cdot }$  The static prize of node ¿. Since OP is a selective maximization problem, this is the core objective-relevant node attribute.</td></tr><tr><td>Reward-distance ratio</td><td> $\begin{array} { r } { \rho _ { i } = \frac { r _ { i } } { d ( v _ { t } , i ) + \epsilon } } \end{array}$  . Measures reward efficiency relative to travel cost, which is critical when high reward and high travel distance conflict.</td></tr><tr><td>Spatial quadrant</td><td> $q _ { i } = \mathrm { Q u a d r a n t } ( x _ { i } , y _ { i } )$  . A coarse spatial-region label used to test whether encoder embeddings capture geometric clustering.</td></tr><tr><td rowspan="6">Decoder-side dynamic state and constraint labels</td><td>Accumulated reward</td><td> $\begin{array} { r } { R _ { t } = \sum _ { s = 1 } ^ { t } r _ { v _ { s } } } \end{array}$  . Measures the objective value collected by the partial route. The decoder should adapt future choices based on how much reward has already been obtained.</td></tr><tr><td>Budget-feasibility interac- tion</td><td> $\begin{array} { r } { ( B - D _ { t } ) | \mathcal { A } _ { t } | \ \mathrm { o r } \ \frac { B - D _ { t } } { | \mathcal { A } _ { t } | + \epsilon } } \end{array}$  . Combines remaining budget with the number of feasible actions, capturing how constrained the current decision state is.</td></tr><tr><td>Current distance</td><td> $\begin{array} { r } { D _ { t } = \sum _ { s = 1 } ^ { t } d ( v _ { s - 1 } , v _ { s } ) } \end{array}$  . Measures consumed travel budget and determines how much flexibility remains.  $\begin{array} { r } { \phi _ { t } ~ = ~ \frac { t } { T _ { \mathrm { e n d } } } } \end{array}$  . Indicates whether the policy is in the early, middle, or late</td></tr><tr><td></td><td>construction stage. Good OP policies should behave differently across these phases.</td></tr><tr><td>Feasible actions Remaining budget</td><td> $\left| { \mathcal { A } } _ { t } \right| .$  Measures the size of the valid action set at step t. High mutual informa- tion with this label indicates awareness of the current feasible decision space.  $B _ { t } ^ { \mathrm { r e m } } = B - D _ { t }$  . Captures the remaining travel resource and determines</td></tr><tr><td></td><td>whether the policy should continue exploring, select nearby nodes, or termi- nate.</td></tr><tr><td>Preference-level solution-quality</td><td>Selected count Top-rollout membership</td><td> ${ \cal C } _ { t } \ = \ t \ \mathrm { o r } \ C _ { t } \ = \ | \{ v _ { 1 } , . . . , v _ { t } \} | ,$  Tracks route length and construction progress, which affects both accumulated reward and feasibility. For encoder analysis,  $y _ { i } ^ { \mathrm { t o p } } ~ = ~ \mathbb { I } ( i ~ \in ~ \tau ^ { \mathrm { t o p } } )$  , where  $\tau ^ { \mathrm { t o p } }$  denotes a top-</td></tr><tr><td rowspan="2">labels</td><td></td><td>performing rollout or the union of nodes visited by top rollouts. For decoder analysis,  $y _ { m } ^ { \mathrm { t o p / b o t t o m } } = \mathbb I ( R ( \tau _ { m } ) \geq Q _ { 1 - q } ( R ) )$  . This tests whether represen- tations encode information associated with globally high-quality solutions.</td></tr><tr><td>Reward quantile</td><td> $q _ { m } = \mathrm { Q u a n t i l e B i n } ( R ( \tau _ { m } ) )$  . This assigns each sampled rollout to a dis- crete final-reward quantile. Unlike binary top-versus-bottom membership, it provides a graded preference signal and tests whether decoder states encode relative solution quality.</td></tr></table>

On the N = 100 setting (Fig. 6a), all adapted policies remain concentrated in regions with relatively high likelihood under the base RWPO policy, suggesting that post-training improves trajectories largely within regions already covered by RWPO. It is interesting that RWPO-FT has a lower log-likelihood under the base RWPO policy, while achieving substantially higher objective values. This suggests that the performance gain is not simply due to sharpening the original RWPO distribution or concentrating probability mass on trajectories that the base policy already assigns high likelihood. Instead, RWPO-FT appears to increase probability on trajectories that are less preferred by the original RWPO policy but achieve higher objective values. In contrast, on the N = 500 setting (Fig. 6b), RWPO-FT substantially shifts the sampled solutions to a much higher objective region, while the tilted variant further improves the binned quality curve slightly. This indicates that, under larger-scale distribution shift, high-quality trajectories are not necessarily those with the highest likelihood under the original RWPO policy.

Table 5: Comparison on instances across different scales. Best results are highlighted in bold.
<table><tr><td rowspan="2"></td><td rowspan="2">Methods</td><td colspan="3">N = 100</td><td rowspan="2"></td><td colspan="3">N = 500</td><td colspan="3"> $N = 1 { , } 0 0 0$ </td><td colspan="3"> $N = 2 { , } 0 0 0$ </td></tr><tr><td>Obj. ↓</td><td>Gap</td><td>Time</td><td>Obj. ↓</td><td>Gap</td><td>Time</td><td>Obj. ↓</td><td>Gap</td><td>Time</td><td>Obj. ↓</td><td>Gap</td><td>Time</td></tr><tr><td rowspan="9">CVRP</td><td>LKH3*</td><td>15.65</td><td>0.00%</td><td></td><td></td><td>37.23</td><td>0.00%</td><td>1.2m</td><td>46.40</td><td>7.91%</td><td>4m</td><td>64.90</td><td>8.15%</td><td>14m</td></tr><tr><td>BQ</td><td>16.09</td><td>2.79%</td><td>2.2m</td><td></td><td>38.44</td><td>3.25%</td><td>1.5m</td><td>44.17</td><td>2.72%</td><td>13m</td><td>62.59</td><td>4.30%</td><td>30m</td></tr><tr><td>POMO</td><td>15.75</td><td>0.66%</td><td>2.3m</td><td></td><td>44.64</td><td>19.90%</td><td>2.5m</td><td>52.97</td><td>23.18%</td><td>19m</td><td>92.34</td><td>53.87%</td><td>2.2h</td></tr><tr><td>PO</td><td>15.80</td><td>0.94%</td><td>4.4m</td><td></td><td>43.76</td><td>17.53%</td><td>2.9m</td><td>62.69</td><td>45.78%</td><td>20m</td><td>124.87</td><td>108.09%</td><td>2.2h</td></tr><tr><td>PO+LS</td><td>15.82</td><td>1.11%</td><td>4.5m</td><td>0OD</td><td>44.20</td><td>18.73%</td><td>3.0m</td><td>63.95</td><td>48.71%</td><td>21m</td><td>116.63</td><td>94.35%</td><td>2.2h</td></tr><tr><td>RWPO</td><td>15.79</td><td>0.90%</td><td>4.4m</td><td></td><td>42.48</td><td>14.10%</td><td>2.9m</td><td>59.49</td><td>38.35%</td><td>20m</td><td>115.98</td><td>93.26%</td><td>2.2h</td></tr><tr><td>RWPO+LS</td><td>15.81</td><td>1.04%</td><td>4.5m</td><td></td><td>42.95</td><td>15.37%</td><td>3.0m</td><td>62.51</td><td>45.36%</td><td>22m</td><td>134.67</td><td>124.42%</td><td>2.2h</td></tr><tr><td>RWPO+TS</td><td>15.75</td><td>0.61%</td><td>4.4m</td><td></td><td>38.14</td><td>2.45%</td><td>3m</td><td>44.89</td><td>4.40%</td><td>20m</td><td>64.34</td><td>7.21%</td><td>2.2h</td></tr><tr><td>ICAM</td><td>15.87</td><td>1.41%</td><td>1.9m</td><td>n</td><td>37.49</td><td>0.70%</td><td>1.3m</td><td>43.07</td><td>0.16%</td><td>27m</td><td>61.34</td><td>2.22%</td><td>4h</td></tr><tr><td></td><td>UDC</td><td></td><td></td><td></td><td></td><td>37.99</td><td>2.04%</td><td>2.3h</td><td>43.00</td><td>0.00%</td><td>10h</td><td>60.01</td><td>0.00%</td><td>18h</td></tr></table>

For each problem, the upper block reports out-of-distribution (OOD) results, evaluating the zero-shot scale generalization of the base models, while the lower block reports in-distribution (ID) results after scale-specific adaptation. For BQ (larger model), POMO (training 30500 epochs), ICAM, and UDC (augmentation x200), we use the official pre-trained models from the original papers. For POMO, PO, ICAM, and RWPO, we apply the augmentation (x8) proposed in [Kwon et al., 2020].

Table 6: Effect of fine-tuning on larger-scale generalization on OP. Relative differences are computed using mean tour revenue, where higher is better.
<table><tr><td>Fine-tuned model</td><td>Reference model</td><td>OOD scales</td><td>Mean ∆ (%)</td><td>Mean |∆| (%)</td><td>Max loss (%)</td><td>Better/equal (%)</td></tr><tr><td>RWPO-FT</td><td>RWPO</td><td>200, 500, 1000</td><td>-0.57</td><td>0.66</td><td>2.06</td><td>22.86</td></tr><tr><td>RWPO-FT-LS</td><td>RWPO-FT</td><td>200, 500, 1000</td><td>-0.28</td><td>1.40</td><td>2.68</td><td>34.29</td></tr><tr><td>RWPO-FT-TS</td><td>RWPO-FT</td><td>200, 500, 1000</td><td>+0.63</td><td>0.63</td><td>0.00</td><td>100.00</td></tr><tr><td>RWPO-FT-TS</td><td>RWPO-FT-LS</td><td>200, 500, 1000</td><td>+0.94</td><td>1.57</td><td>2.14</td><td>74.29</td></tr><tr><td> $\mathrm { R W P O - F T _ { 2 0 0 } }$ </td><td>RWPO-FT</td><td>500, 1000</td><td>+6.71</td><td>6.72</td><td>0.15</td><td>96.15</td></tr><tr><td> $\mathrm { R W P O - F T _ { 5 0 0 } }$ </td><td>RWPO-FT</td><td>1000</td><td>+9.42</td><td>12.83</td><td>6.35</td><td>66.67</td></tr><tr><td> $\mathrm { R W P O  – F T _ { 5 0 0 } – T S }$ </td><td> $\mathrm { R W P O - F T _ { 5 0 0 } }$ </td><td>1000</td><td>+1.54</td><td>1.54</td><td>0.00</td><td>100.00</td></tr></table>

∆ denotes the relative change in mean tour revenue between the fine-tuned model and its reference model. Mean ∆ captures the average signed change, while Mean $| \Delta |$ measures the average magnitude of change regardless of direction. Max loss reports the largest performance degradation, and Better/equal reports the percentage of cases where the fine-tuned model matches or outperforms the reference model.

Comparing the two fine-tuning strategies, RWPO-FT-LS samples extend farther toward lower base RWPO likelihood values, while RWPO-FT-TS samples are more concentrated near the high-likelihood region. In the likelihood–quality frontier, RWPO-FT-LS generates many trajectories with lower likelihood under the base RWPO policy, suggesting that LS guidance encourages exploration of solutions less represented in the original RWPO sampling distribution. This suggests that LS-guided fientuning can improve trajectories that were assigned relatively low probability by the base RWPO policy. However, this larger deviation from the base likelihood distribution may indicate a larger policy shift, which could require additional evaluation for robustness. In contrast, RWPO-FT-TS remains closer to the high-likelihood region of the base RWPO policy. Its reward term encourages improvement, while the base-likelihood term anchors the adapted policy to trajectories that remain plausible under the original model. As a result, RWPO-FT-TS behaves as a more conservative reward-guided redistribution mechanism. Therefore, RWPO-FT-TS appears to induce a smaller shift relative to the base policy, whereas LS-guidedfine-tuning explores a broader region ofthe trajectory space. These observations further support the illustration in Fig. 1 in the main text, which depicts the evolution of the policy probability distribution during adaptation.

## E.2.2 Fine-tuning does not degrade generalization ability.

Tab. 6 evaluates whether post-training adaptation improves or harms larger-scale generalization. Continued post-training on the original scale (RWPO-FT) only slightly changes OOD performance relative to the base RWPO model, with a small average drop of 0.57%, suggesting that low-learning-rate refinement is largely conservative but does not by itself substantially improve transfer. LS-guided fine-tuning also shows limited OOD benefit and can introduce larger variation, indicating possible sensitivity to the fine-tuning distribution. In contrast, RWPO-TS consistently improves over RWPO-FT, achieving a positive mean gain of 0.63% with no observed loss and a 100% better/equal rate, showing that tilted sampling provides a safer adaptation mechanism. Scale-specific fine-tuning is more effective for large-scale transfer: fine-tuning on $N = 2 0 0$ improves performance on $N = 5 0 0$ and $N = 1 0 0 0$ by $6 . 7 1 \%$ on average, while fine-tuning on $N = 5 0 0$ improves transfer to $\bar { N } = 1 0 0 0$ by 9.42%. Applying tilted sampling after $N = 5 0 0$ fine-tuning further improves performance by 1.54% with no degradation. Overall, the table shows that RWPO supports stable post-training adaptation, and that the most reliable pipeline combines scale-specificfine-tuning with tilted sampling.

## E.3 Representation Diversity under Scale Generalization

Route-level visualization of decoder-query representation diversity. To complement the aggregate entropy statistics, we visualize representative decoded OP routes with visited nodes colored by decoder-query soft entropy. Decoder-query soft entropy measures the diversity of the decoder’s latent decision-state representation rather than the uncertainty of the next-action distribution. A high value indicates that the decoder maintains a richer representation of the current partial solution, potentially preserving diverse information about accumulated reward, remaining budget, feasibility constraints, and future trade-offs; a low value indicates a more compressed decision-state representation. Therefore, decoder-query soft entropy provides a representation-level view that complements policy entropy, which directly measures behavioral uncertainty in the next-action distribution. These route-level visualizations connect the numerical representation statistics to the actual spatial decision process: in OP, a policy must decide not only which high-reward regions to visit but also when to move between regions under a limited travel budget. Visualizing decoder-query soft entropy along the constructed route allows us to inspect whether the decision-state representation changes in a spatially structured manner, for example, whether it becomes more concentrated within locally coherent route segments and more diverse near transitions between candidate regions.

Figures 7 and 8 show that the methods differ substantially in how decoder-query representation diversity is distributed along the route. The route-level entropy patterns are consistent with the inductive biases of the corresponding training objectives. PO, trained only to preserve pairwise preference order, lacks reward-gap information and therefore learns a weaker reward–geometry representation under scale shift; this is reflected by collapsed decoder-query soft entropy, confinement to low-reward regions, inefficient first moves, and crossed routes. POMO, trained by policy-gradient reward maximization, can move toward high-reward regions, but its representation diversity increases mainly after reaching those regions, suggesting that it is better at local exploitation than early global region selection. In contrast, RWPO weights preference updates by reward differences, so large objective gaps receive stronger learning signals. This encourages the decoder to preserve richer decision-state representations during early route-shaping decisions, when the policy must choose which high-reward region to approach. Scale-specific fine-tuning further adapts this representation to larger candidate sets: $\mathrm { R W P O - F T _ { 5 0 0 } }$ maintains high decoder-query diversity through the transition into high-reward regions and then becomes more concentrated near route completion, leading to fewer crossings and better geometric route quality. Overall, the visualization supports the view that reward-weighted preference learning improves $O P$ generalization by aligning representation diversity with reward magnitude, feasibility, and route structure.

## E.4 Mutual Information Between Hidden Representations and Task-Relevant Labels on OP

Scale-dependent division of labor in encoder and decoder representations. To examine whether RWPO’s improved generalization is accompanied by a different allocation of information between the encoder and decoder, we estimate the soft mutual information between intermediate representations and labels describing the OP instance and routeconstruction state. The encoder labels include static instance-level attributes, such as depot distance, depot feasibility, node type, reward, reward–distance, spatial quadrant, and top-rollout membership. The decoder labels describe the evolving partial route, including accumulated reward, budget–feasibility interaction, current distance, decoding phase, feasible actions, remaining budget, and selected-node count.

Figures 9 and 10 reveal a relative division of labor between encoder and decoder representations across problem scales. In both settings, the encoder exhibits its strongest mutual information with static instance-level attributes, particularly depot distance and node reward, whereas the decoder-query representation contains substantially more information about dynamic route-construction variables, including budget–feasibility, feasible actions, remaining budget, and current distance. This pattern is consistent with an encoder that represents the candidate set and its reward–geometry attributes, and a decoder that tracks the evolving feasibility state of the partial route. The method comparison shows a clear scale-dependent difference: at $N = 5 0 0$ , PO exhibits a marked attenuation of decoder information for accumulated reward and constraint-related variables, while RWPO retains strong information about budget–feasibility, feasible actions, remaining budget, and current distance. POMO also preserves strong decoder information for several of these labels and is comparable to RWPO in some cases, so the heatmaps do not establish a uniform superiority of RWPO over POMO. The RWPO variants, including RWPO-TS, consistently maintain strong constraint-related decoder information, which is consistent with the hypothesis that reward-weighted post-training encourages more constraint-aware representations. Together with the independently observed performance degradation of PO at $N = 5 0 0$ , these results provide representational evidence consistent with improved large-scale generalization.

Scale-dependent organization of task-relevant decoder information To further examine whether the improved generalization of RWPO is reflected in the internal decision process, we measure how much task-relevant information is encoded in the decoder-query representation at each decoding step. Specifically, we consider (i) constraint-related signals, including feasible actions, current distance, remaining budget, and budget–feasibility interaction; (ii) objective related signals, including accumulated reward; and (iii) preference-related labels, including reward quantile and top-versus-bottom trajectory membership. Together, these labels characterize whether the decoder representation encodes the key quantities needed to balance reward collection, feasibility preservation, and route completion.

Fig. 11 shows that decoder-query representations encode different decision-relevant factors at different stages of route construction. The difference becomes much clearer at $N = 5 0 0$ . For PO, the information profile becomes concentrated primarily on current distance, top-versus-bottom trajectory membership, and reward quantile, whereas the information associated with accumulated reward, feasible actions, budget–feasibility interaction, and remaining budget is substantially weaker than at $N = 1 0 0$ . In contrast, RWPO retains strong encoding of feasible actions and budget-related signals during the middle stages, including a pronounced peak in the remaining-budget signal, before shifting toward current distance, top-versus-bottom trajectory membership, and reward quantile during later decoding. Thus, as the instance size grows, RWPO maintains a staged decision process that first emphasizes feasibility and budget management and subsequently emphasizes distance and trajectory quality, while PO becomes more narrowly focused on distance and outcome-related signals. The $\mathrm { R W P O  – F T _ { 5 0 0 } – T S }$ variants further amplify this qualitative pattern through larger and more sustained information peaks. Combined with the observed degradation of PO performance at $N = 5 0 0$ these results are consistent with the hypothesis that RWPO improves large-scale generalization by preserving constraintand objective-relevant information in the decoder representation.

## E.5 Training Curves

The stability advantage of RWPO extends beyond the OP. On SPCTSP100, the RWPO all-pair variant rapidly improves the mean objective and subsequently follows a smooth, low-variance trajectory, whereas PO remains highly oscillatory throughout training (Fig.12a). On KP200, although PO and RWPO converge to similar final mean values, RWPO exhibits substantially lower variance, indicating more consistent optimization across the k parallel starts (Fig.12b). On CVRP100, RWPO maintains a lower objective cost than PO for nearly the entire training process, finishing with a small but consistent advantage in the best-objective metric (Fig.12c). These cross-domain findings corroborate the OP and PCTSP results (Fig.5 in Section 4.4): while purely ordinal PO remains competitive when evaluating only the single best rollout (Fig.5a in Section 4.4), its batch-wide mean performance is considerably noisier and less stable (Fig.5b in Section 4.4). In contrast, RWPO consistently improves the entire rollout distribution. This behavior directly reflects the RWPO all-pair mechanism, which extracts dense learning signals across all sampled trajectories while explicitly scaling updates by the true reward differences.

## F Ablation Study of Hyperparameters

We conduct a reduced-budget sensitivity analysis on OP50 and OP100 to evaluate the effects of the reward-weighting strength (λ) and the length-normalization factor (β). Because these instances are computationally lightweight, they permit efficient hyperparameter screening. Rather than exhaustively tuning hyperparameters, our objective is to identify robust configurations that transfer reliably across problem scales.

As shown in Tab. 7 and 8, introducing reward-gap weighting consistently improves the final validation objective compared to the purely ordinal baseline $( \lambda = 0 )$ . Moderate weighting strengths yield the most stable improvements across both OP50 and OP100; conversely, excessively high values provide no additional gains and risk destabilizing the preference updates. We therefore adopt $\lambda = 0 . 5$ as a conservative default for the main OP experiments.

Furthermore, we observe that the optimal length-normalization strength scales directly with instance size. While weaker normalization suffices for OP50, OP100 strictly benefits from a stronger $\beta$ adjustment. This occurs because larger instances generate longer trajectories, inducing greater variance in cumulative log-likelihood magnitudes. To proactively handle this variance at larger scales, we fix $\beta = 0 . 3$ across all OP experiments. We apply this same validation-based procedure to determine stable configurations for the remaining SCOP benchmarks prior to final evaluation.

Table 7: Ablation results for different loss functions and hyperparameter settings on OP50. Lower (more negative) objective values indicate better performance.
<table><tr><td>Loss type</td><td>β</td><td>λ</td><td>Best Reward @ 20 ↓</td><td>Mean Reward @ 20 ↓</td><td>Best Reward @ 50 ↓</td><td>Mean Reward @ 50 ↓</td><td>Best Reward ↓ (epoch)</td></tr><tr><td>RWPO base</td><td>0.0</td><td>0.5</td><td>-16.022</td><td>-15.155</td><td>-16.117</td><td>-15.296</td><td>-16.117 (50)</td></tr><tr><td>RWPO base</td><td>0.2</td><td>0.5</td><td>-15.990</td><td>-15.071</td><td>-16.080</td><td>-15.214</td><td>-16.080 (50)</td></tr><tr><td>RWPO all pair</td><td>0.3</td><td>0.0</td><td>-15.913</td><td>-14.950</td><td>-12.256</td><td>-10.205</td><td>-15.953 (15)</td></tr><tr><td>RWPO all pair</td><td>0.0</td><td>0.5</td><td>-16.122</td><td>-15.336</td><td>-16.172</td><td>-15.434</td><td>-16.176 (45)</td></tr><tr><td>RWPO all pair</td><td>0.2</td><td>0.5</td><td>-16.096</td><td>-15.292</td><td>-16.143</td><td>-15.387</td><td>-16.160 (45)</td></tr><tr><td>RWPO all pair</td><td>0.3</td><td>0.5</td><td>-16.031</td><td>-15.194</td><td>-16.107</td><td>-15.332</td><td>-16.111 (49)</td></tr><tr><td>RWPO all pair</td><td>0.3</td><td>1.0</td><td>-15.967</td><td>-15.037</td><td>-16.065</td><td>-15.254</td><td>-16.080 (50)</td></tr><tr><td>RWPO all pair</td><td>0.3</td><td>2.0</td><td>-15.942</td><td>-14.996</td><td>-16.070</td><td>-15.223</td><td>-16.070 (50)</td></tr></table>

Table 8: Ablation results for different loss functions and hyperparameter settings. Lower (more negative) objective values indicate better performance.
<table><tr><td>Loss type</td><td>β</td><td>λ</td><td>Best Reward @ 10 ↓</td><td>Mean Reward @ 10 ↓</td><td>Best Reward @ 20 ↓</td><td>Mean Reward @ 20 ↓</td><td>Best Reward ↓ (epoch)</td></tr><tr><td>RWPO base</td><td>0.0</td><td>0.5</td><td>-31.817</td><td>-30.394</td><td>-32.230</td><td>-30.872</td><td>-32.230 (19)</td></tr><tr><td>RWPO base</td><td>0.3</td><td>0.5</td><td>-31.710</td><td>-30.190</td><td>-32.221</td><td>-30.827</td><td>-32.221 (19)</td></tr><tr><td>RWPO all pair</td><td>0.3</td><td>0.0</td><td>-32.049</td><td>-30.556</td><td>-32.294</td><td>-30.824</td><td>-32.294 (19)</td></tr><tr><td>RWPO all pair</td><td>0.0</td><td>0.5</td><td>-32.404</td><td>-30.989</td><td>-32.644</td><td>-31.300</td><td>-32.644 (19)</td></tr><tr><td>RWPO all pair</td><td>0.2</td><td>0.5</td><td>-32.678</td><td>-31.353</td><td>-32.897</td><td>-31.684</td><td>-32.897 (19)</td></tr><tr><td>RWPO all pair</td><td>0.3</td><td>0.5</td><td>-32.569</td><td>-31.210</td><td>-32.815</td><td>-31.585</td><td>-32.815 (19)</td></tr><tr><td>RWPO all pair</td><td>0.3</td><td>1.0</td><td>-32.086</td><td>-30.655</td><td>-32.116</td><td>-30.754</td><td>-32.116 (18)</td></tr><tr><td>RWPO all pair</td><td>0.3</td><td>2.0</td><td>-32.363</td><td>-30.925</td><td>-32.538</td><td>-31.093</td><td>-32.538 (19)</td></tr></table>

![](images/3d3a9104aa6be475238a83ff6eef48766370ea1d72e58a3a49076cbb5b83f589.jpg)

(a) N = 100: same-scale adaptation  
![](images/0a5592a0756765d557764bbb4dd16d7c495419d3c16afe85f99108a7d52378c8.jpg)  
(b) N = 500: large-scale adaptation

Figure 6: Length-normalized likelihood–quality frontier under the base RWPO policy (1k epochs). Each point represents a sampled trajectory, plotted by its length-normalized log-likelihood under the base policy (x-axis) versus its objective value (y-axis). Solid curves indicate binned mean objectives. On $N = 1 0 0$ , scale-specific fine-tuning (FT) and tilted sampling (TS) shift the distribution toward substantially higher-quality solutions while retaining high likelihood under the base policy, whereas local search (LS) fine-tuning induces a significantly larger distribution shift. On $N = 5 0 0$ the frontier flattens and trajectories tightly cluster in the high-likelihood region; nevertheless, RWPO-based adaptation successfully elevates the upper quality envelope. These dynamics confirm that FT performs a conservative probability redistribution among already-plausible trajectories—avoiding drastic shifts into low-likelihood regions—while TS effectively concentrates the probability mass of the base policy.

top route | global instance=0 | reward=72.930  
top route | global instance=0 | reward=81.450  
![](images/e536156301ee0bc343dc82470fe379daf25401a48cc6f1129a0686a8b11d7113.jpg)  
(a) PO

![](images/d7aec0b9dac9b7546943134f8c8106f69df726e24e055e82756bbaf95ae80c4f.jpg)  
(b) POMO

![](images/a1d79149030940f4f56d7af3348978d42c816c18f75933e2768ec648c8948aa3.jpg)  
(c) RWPO

![](images/d1cc1fa4e63f67170e17fba28bbfaf080cd88c041c21a8b62c3cd01c6ce4e400.jpg)  
(d) RWPO-FT<sub>500</sub>  
Figure 7: Route-level decoder-query soft entropy visualization on a representative OP instance with $N \ : = \ : 5 0 0$ Candidate nodes are shown in light blue, selected routes are shown as connected paths, and visited nodes are coloured by decoder-query soft entropy $\bar { \mathcal { H } } ( Z _ { t } )$ . These plots provide qualitative evidence of how different methods modulate uncertainty along the constructed route.

![](images/31984a09e7fbd98fa4eab9d8eb07f9fd5b4bc041845ea47b197578851b46d86b.jpg)  
(a) PO

![](images/2df65e41a59bb492b5397992ea9e54291c70875eaaf12eb479f09019bb9250b0.jpg)  
(b) POMO

![](images/83a3d7be53c99ce563d1437145d091c85228fd5633c5bbfd7f0693e149557ba6.jpg)  
(c) RWPO

![](images/aaa9fc9a01820162743f67a3381a33b8ab8fcd9b5c87b9b5c52bb6e12fa71d3d.jpg)  
(d) RWPO-FT<sub>500</sub>  
Figure 8: Route-level decoder-query soft entropy visualization on a second representative OP instance with $N = 5 0 0$ The comparison illustrates how entropy evolves spatially along the selected route and whether uncertainty is concentrated near local route segments or transitions between regions.

![](images/778b85f6e0fcfbe66fc14a03c64945db9bb43dcc1748ca001267267645cce20e.jpg)  
(a) PO

![](images/934b8fa99f101d1c3015409ae294893ec793795697033ec370b2f7dfed5bd041.jpg)  
(b) POMO

![](images/ca5bb52a2542789f0498b4b90e8aaf46c5a1b1792a66769f75b3057aff1f9fd2.jpg)  
(c) RWPO

![](images/12ae7facd844b769f4021929793cafea1224cfc4b8ae3062ba76c4e9718bbccd.jpg)  
(d) RWPO-TS  
Figure 9: Mutual information between encoder/decoder representations and task-relevant labels on $\mathrm { O P }$ instances with $N = 1 0 0$ . The encoder labels include static instance-level information such as depot distance, node reward, spatial quadrant, and top-rollout membership. The decoder labels include dynamic route-construction signals such as accumulated reward, budget feasibility, current distance, decoding phase, feasible actions, remaining budget, and selected count. Brighter colours indicate that the corresponding representation contains more information about the label.

![](images/b356c639e82daee0fda056b19348e5f99f4b2cb7f355306f18232046b4a4a8ab.jpg)

![](images/e627351867c28ebe92f19c8547601e8edbe7de5bde1fdfe4b5fb162abbf3ae84.jpg)  
(b) POMO

![](images/78815d1eb8d979b2adbfa7fa23da54d3fe0f51a324d8dc3ce2b49bcfa43d9c94.jpg)  
(c) RWPO

![](images/4bdc138cc297d5f7917144e9863fe29324c927969eecc2a4ce59576d4d9b81d8.jpg)  
(d) RWPO-FT<sub>500</sub>  
Figure 10: Mutual information between encoder/decoder representations and task-relevant labels on OP instances with $N = 5 0 0$ . Encoder-side labels describe static instance-level information, including depot distance, node reward, spatial quadrant, and top-rollout membership. Decoder-side labels describe dynamic route-construction states, including accumulated reward, budget feasibility, current distance, decoding phase, feasible actions, remaining budget, and selected count. Brighter colours indicate that the corresponding representation contains more information about the label.

![](images/4764ba11a1f11e341b8fd929dd0922c5ce064420d8b7dcef78f7906a2fcd075d.jpg)  
(a) PO, N = 100

![](images/e0bef91deed289aa23d3e53f99b764166b4a2d78d2a0fd7db0de2b8edb977bcc.jpg)  
(b) RWPO, N = 100

![](images/f9b9a9687206437c6b4da6803ea4af9352b9cfd172f69902c5d1e3e9bd54cc41.jpg)  
(c) RWPO-TS, N = 100

![](images/31f2e738b8ee8f511b1d5dc4b07405ab7baa5e07866a868599a027681ca85114.jpg)  
(d) PO, N = 500

![](images/4c1c0fa6cf71c66e813c2c365faf9b05e99f8cef7fa57876aa1ee2b2bc1323a1.jpg)  
(e) RWPO, N = 500

![](images/9eec1e32e0f574678d062645e85f434a1a514a34d7bac139a88921d00c6b102c.jpg)  
(f) RWPO-FT<sub>500</sub>-TS, N = 500  
Figure 11: Step-wise mutual information between decoder-query representations and trajectory-level features during OP decoding.

![](images/2d24865ddd18174e23bbc8db30e95c28fe5b0d22bd11b888fc5fd8b6888f8e1b.jpg)  
(a) Mean reward on SPCTSP100

![](images/43665dad15312086d6e271639cffe3e62629587acfde254b370b7029bad98f17.jpg)  
(b) Mean reward on KP200

![](images/817d3c9fc33376dd7a05f2acbd8f2235b3de5c3126eba93ae5ebe7635a561dd3.jpg)  
(c) Best reward on CVRP100

Figure 12: Training dynamics of PO and RWPO on SPCTSP100, KP200, and CVRP100. All three panels report the mean validation objective over all top-k starts. For SPCTSP and CVRP, lower objective values indicate better performance and for KP, vice versa. RWPO shows smoother and more stable optimization than PO, with the clearest final improvement on SPCTSP100 and CVRP100. On KP200, both methods reach similar final values, but RWPO exhibits substantially lower variance during training. These results complement the training curves in Fig. 5.