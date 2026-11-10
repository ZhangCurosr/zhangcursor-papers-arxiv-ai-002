# RAIL: Representation-Aligned Imitation Learning for Student-Compatible Teacher Policies

Meraj Mammadov Örebro University

Pedro Zuidberg Dos Martires Örebro University

Johannes A. Stork Örebro University

## Abstract

Reinforcement learning (RL) from raw sensory inputs such as images or onboard sensors can be challenging due to high-dimensional observations and sparse rewards. A common strategy is to train a teacher policy with access to privileged state information and then distill it into a student policy that acts from raw inputs alone. However, when the teacher relies on information unavailable to the student, exact imitation may be impossible, creating an irreducible imitation gap. Existing approaches typically mitigate this mismatch through careful reward shaping or additional reinforcement learning on the student policy after the teacher has been trained. Instead, we introduce Representation-Aligned Imitation Learning (RAIL), which addresses the mismatch during teacher training itself. RAIL learns a latent representation shared across teacher and student observations using contrastive learning, and trains the teacher policy directly in this space. Because the teacher policy is constrained to operate on this shared space, it learns behaviors that are reproducible from student observations. This substantially reduces the imitation gap while preserving task performance. Across multiple environments, RAIL outperforms strong baselines without reward modification or post-hoc student fine-tuning, and surpasses direct reinforcement learning from raw observations. The learned representations further enable zero-shot transfer to new tasks through teacher-only training. Project site: https://merajmammadov.com/rail/.

## 1 Introduction

Training reinforcement learning policies directly on the target observation space can be impractical due to high dimensional observation space, sparse rewards, or costly exploration. A common strategy is therefore to first train a teacher policy with RL using privileged state information, and then to distill its behavior into a student policy that operates with the actual deployment observations [2]. For example, in navigation tasks the teacher may observe a full map, obstacle locations, or target coordinates, while the student must act on onboard camera images or limited local sensing.

While this teacher–student paradigm can greatly simplify training, it also introduces a fundamental mismatch: when the teacher exploits privileged variables that are not recoverable from the student’s observations, the resulting policy may be impossible for the student to reproduce. In that case, no policy conditioned only on student observations can exactly match the teacher’s decisions, leading to an irreducible imitation gap [31].

To illustrate this, consider the navigation task to the right (Figure 1), where an agent is dropped on the black tile and must reach the goal (green tile), while avoiding obstacles (red tiles). A privileged teacher observes all neighboring cells (dashed blue) and can move toward the goal without the need to change orientation. In contrast, the student observes only three cells in front of it (dashed orange) and needs to actively change its orientation to complete the mission successfully. This means that the teacher can distinguish between states that are indistinguishable to the student, which in turn means that the teacher is capable of taking more informed actions than the student.

![](images/8929f4942c04418d46ded410468530b63f0879de1cb0339feff5928bf9ac6984.jpg)  
Figure 1: Observation mismatch in TunnelVision: Teacher (blue) vs. student (orange).

Existing approaches typically address this mismatch after teacher training, either by incorporating teacher signals into the student’s RL objective [13, 14], or by additional RL fine-tuning of the student [33]. While effective in certain settings, these methods reintroduce the cost of RL optimization and do not remove the underlying information asymmetry.

In this work, we address the teacher–student imitation gap at its source: instead of training an unconstrained teacher and repairing the student afterward, we constrain the teacher to act only on information accessible to the student. We propose Representation-Aligned Imitation Learning (RAIL), which learns a common representation of teacher and student observations that captures only the information available to both. The teacher policy is trained with RL in this space, and the same policy is applied by the student at deployment time.

The contributions of this paper are: (1) We propose RAIL, a teacher–student imitation framework that enforces imitability during teacher training by restricting the teacher policy to a shared latent space that is also recoverable from student observations. (2) We introduce a contrastive representation-learning objective with gradient blocking, alignment, and stability regularization to prevent the leakage of private teacher information and enable direct policy transfer. (3) Across multiple environments, we show that RAIL reduces the imitation gap, outperforms strong baselines, learns faster than pixel-based RL, and supports zero-shot transfer to new tasks.

## 2 Preliminaries

We consider a sequential decision-making problem in which a teacher $T$ and a student $S$ act in the same environment but observe it through different observation mappings. Concretely, the environment is defined by the state space $s ,$ action space ${ \mathcal { A } } ,$ transition kernel $\bar { P ( } \cdot \mid s , a )$ , and reward function $r : \mathcal { S } \times \mathcal { A }  \mathbb { R }$ . At time-step t, both agents observe the same latent state $s _ { t } \in S$ through their corresponding observation mappings, $o _ { t } ^ { T } \equiv f _ { T } ( s _ { t } )$ and $o _ { t } ^ { S } = f _ { S } ( s _ { t } )$ , where $o _ { t } ^ { T } \in \mathcal { O } _ { T }$ and $o _ { t } ^ { \overline { { S } } } \in { \mathcal { O } } _ { S }$ denote teacher and student observations, respectively. The teacher and student follow policies $a _ { t } ^ { T } \sim \pi _ { T } ( \cdot \ | \ o _ { t } ^ { T } )$ and $a _ { t } ^ { S } \sim \pi _ { S } ( \cdot \ | \ o _ { t } ^ { S } )$ that map their respective observations to a probability distribution over their shared action space.

In classical teacher–student settings [2, 14], the teacher policy $\pi _ { T }$ is trained with reinforcement learning to maximize expected discounted return:

$$
\operatorname* { m a x } _ { \pi _ { T } } \mathbb { E } _ { \tau \sim \pi _ { T } } \left[ \sum _ { t = 0 } ^ { \infty } \gamma ^ { t } r ( s _ { t } , a _ { t } ^ { T } ) \right] ,\tag{1}
$$

where ${ \boldsymbol \tau } = ( s _ { 0 } , a _ { 0 } , \dots )$ is a trajectory generated by $\pi _ { T }$ under the transition kernel $P$ and $\gamma \in ( 0 , 1 ]$ is the discount factor. The student policy $\pi _ { S }$ is later trained to approximate the teacher’s policy under its own observation space via imitation learning.

Given paired observations $( o _ { t } ^ { T } , o _ { t } ^ { S } )$ from the same state $s _ { t } .$ , standard behavior cloning [18] fits a student policy $\pi _ { S }$ to match a fixed teacher policy $\pi _ { T }$ by minimizing a divergence $D ( \cdot , \cdot )$ between the action distribution of the teacher and student policies, conditioned on their respective aligned observations:

$$
\operatorname* { m i n } _ { \pi _ { S } } ~ \mathbb { E } _ { ( o _ { t } ^ { T } , o _ { t } ^ { S } ) \sim \mathcal { D } } \big [ D \big ( \pi _ { T } ( \cdot \mid o _ { t } ^ { T } ) , \pi _ { S } ( \cdot \mid o _ { t } ^ { S } ) \big ) \big ] ,\tag{2}
$$

where $\mathcal { D } = \{ ( o _ { t } ^ { T } , o _ { t } ^ { S } ) \}$ <sub>t</sub> is a dataset of aligned observations collected from trajectories $\tau \sim \pi _ { T }$

## 2.1 Irreducible imitation gap

Behavior cloning implicitly assumes that the teacher policy is realizable under the student’s observation space. When this assumption holds, small imitation error translates to small performance gap between the agents. In particular, if the policy mismatch in Eq. (2) is upper-bounded by ε, and rewards are bounded as $| r ( s , a ) | \leq R _ { \operatorname* { m a x } }$ , the return gap satisfies the bound of Xu, Li, and Yu [34]:

$$
V _ { \pi _ { T } } - V _ { \pi _ { S } } \leq \frac { 2 \sqrt { 2 } R _ { \operatorname* { m a x } } } { ( 1 - \gamma ) ^ { 2 } } \sqrt { \varepsilon } ,\tag{3}
$$

![](images/0b744d3070e34dd1359229147f2c9b4039cb60bf2a9c704a135c214c47e4c14e.jpg)  
Figure 2: Left: teacher and student observations are generated from the same underlying state via different observation mappings. Middle: teacher and student encoders $h _ { T }$ and $h _ { S }$ , followed by a shared encoder $^ { g , }$ map both observations into a common latent space; $e _ { T }$ and $e _ { S }$ are learned agent identity embeddings. Right: a shared policy π<sub>θ</sub> operates on these representations to produce action distributions $\pi _ { \boldsymbol { \theta } } ( \cdot \mid ^ { - } z _ { t } ^ { T } )$ and $\pi _ { \boldsymbol { \theta } } \big ( \cdot \mid z _ { t } ^ { S } \big ) \dot { . } z _ { \mathrm { o l d } , t } ^ { T }$ and $\dot { z } _ { \mathrm { o l d } , t } ^ { S }$ denote embeddings computed using a frozen copy of the encoders and are used to evaluate the stability objective $L _ { \mathrm { s t a b i l i t y } }$ (cf. Subsection 3.3). Blue arrows indicate the losses used to train the encoders (cf. Section 3).

where $V _ { \pi }$ denotes expected discounted return. Thus, even small one-step imitation error can induce substantial long-horizon performance loss.

A sufficient condition for zero imitation gap is that the teacher action be conditionally independent of teacher observation given the student observation,

$$
a ^ { T } \perp o ^ { T } \mid o ^ { S } .\tag{4}
$$

That is, once $o ^ { S }$ is known, any additional variation in the teacher’s observation does not affect its action distribution. This implies that the teacher policy is a function of $o ^ { S }$ alone and can therefore be matched exactly by a student policy conditioned on $\bar { o ^ { S } }$

In practice, however, the condition in Eq. (4) is often violated because the teacher bases its decisions on information unavailable to the student. Then there may exist two distinct states $s _ { 1 } , s _ { 2 } \in S$ that are identical to the student:

$$
f _ { S } ( s _ { 1 } ) = f _ { S } ( s _ { 2 } ) ,\tag{5}
$$

while the teacher distinguishes them and acts differently:

$$
\pi _ { T } ( \cdot \mid f _ { T } ( s _ { 1 } ) ) \neq \pi _ { T } ( \cdot \mid f _ { T } ( s _ { 2 } ) ) .\tag{6}
$$

Because the student conditions only on $o ^ { S } = f _ { S } ( s _ { 1 } ) = f _ { S } ( s _ { 2 } )$ , it assigns the same action distribution to both states and therefore cannot match the teacher when distinguishing the states is necessary. The minimum achievable imitation loss is therefore strictly positive whenever such indistinguishable states occur with nonzero probability.

As shown by Ross, Gordon, and Bagnell [21], these local mismatches can compound over time, and we also show this in our experimental section (cf. Section 5). In particular, we show that such errors drive the student off-distribution, prevent recovery, and ultimately lead to large teacher– student performance gaps in state-of-the-art imitation learning approaches [18, 21], while RAIL successfully mitigates the presence of such scenarios and achieves a low imitation gap and high student performance.

## 3 Representation-Aligned Imitation Learning

RAIL enforces imitability at the level of the teacher policy by restricting it to depend only on information recoverable from the student observations. Instead of training an unconstrained teacher and compensating for the resulting mismatch afterward, we constrain the policy input during training so that all decisions are realizable under the student observation space.

RAIL builds on the view that teacher and student observations contain overlapping information about the same underlying state, together with private information available only to one side [4]. Exact transfer requires the teacher policy to depend only on the overlapping information. We therefore learn encoders $\phi _ { T }$ and $\phi _ { S }$ that map teacher and student observations into a common latent space:

$$
z _ { t } ^ { T } = \phi _ { T } ( o _ { t } ^ { T } ) , \qquad z _ { t } ^ { S } = \phi _ { S } ( o _ { t } ^ { S } ) .\tag{7}
$$

For aligned observations from the same state $s _ { t }$ , we encourage $z _ { t } ^ { T }$ and $z _ { t } ^ { S }$ to agree while keeping representations of different states distinguishable. This naturally leads to a contrastive objective: paired teacher–student observations are treated as positives, while observations from different states act as negatives. Optimizing this objective encourages the latent space to retain information shared across both observation spaces while suppressing factors specific to each observation.

We train a single policy $\pi _ { \theta }$ on top of this learned representation. During training, the teacher acts through teacher embeddings,

$$
a \sim \pi _ { \theta } ( \cdot \mid z ^ { T } ) ,\tag{8}
$$

and at deployment the student applies the same policy to student embeddings,

$$
a \sim \pi _ { \theta } ( \cdot \mid z ^ { S } ) .\tag{9}
$$

Because the same policy π<sub>θ</sub> operates on both teacher and student embeddings, transfer requires only replacing $\phi _ { T }$ with $\phi _ { S }$ once the embeddings are aligned.

An overview of the framework is shown in Figure 2. RAIL iteratively updates the encoder networks $\phi _ { T }$ and $\phi _ { S }$ with contrastive learning and the policy network $\pi _ { \theta }$ with reinforcement learning on the learned embeddings. We describe this in more detail in Subsection 3.4.

To learn imitable teacher policies that allow direct transfer to the student, RAIL jointly optimizes three objectives: a contrastive loss to construct the shared latent space (cf. Subsection 3.1), an alignment loss to enable direct policy transfer (cf. Subsection 3.2), and a stability loss to regularize representation updates during policy learning (cf. Subsection 3.3).

## 3.1 Learning a shared latent space

We learn the shared latent space from aligned teacher–student observation pairs collected during teacher rollouts. For each pair $( o _ { t } ^ { T } , o _ { t } ^ { S } )$ corresponding to the same state $s _ { t }$ , the embeddings $z _ { t } ^ { T }$ and $z _ { t } ^ { S }$ should agree, while embeddings from different states should remain distinguishable. We use a contrastive objective [15] as its positive-negative structure directly matches this goal.

We parameterize the encoders as $\phi _ { T } = g \circ h _ { T }$ and $\phi _ { S } = g \circ h _ { S }$ , where $h _ { T }$ and $h _ { S }$ map teacher and student observations into feature spaces, and $g$ maps both into a shared latent space. Given an observation $o _ { i } ^ { A }$ from agent $A \in \{ T , S \}$ , we compute

$$
z _ { i } ^ { A } = \phi _ { A } ( o _ { i } ^ { A } ) = \frac { g ( h _ { A } ( o _ { i } ^ { A } ) , e _ { A } ) } { \| g ( h _ { A } ( o _ { i } ^ { A } ) , e _ { A } ) \| _ { 2 } } ,\tag{10}
$$

where $e _ { A } \in \mathbb { R } ^ { 4 }$ is a learned indicator of the input source.

The contrastive loss over the embeddings is given by the symmetric InfoNCE objective [15]:

$$
\mathcal { L } _ { \mathrm { c o n t r a s t i v e } } = - \frac { 1 } { 2 N } \sum _ { i = 1 } ^ { N } \left[ \log \frac { \exp ( ( z _ { i } ^ { T } ) ^ { \top } z _ { i } ^ { S } / \tau ) } { \sum _ { j = 1 } ^ { N } \exp ( ( z _ { i } ^ { T } ) ^ { \top } z _ { j } ^ { S } / \tau ) } + \log \frac { \exp ( ( z _ { i } ^ { S } ) ^ { \top } z _ { i } ^ { T } / \tau ) } { \sum _ { j = 1 } ^ { N } \exp ( ( z _ { i } ^ { S } ) ^ { \top } z _ { j } ^ { T } / \tau ) } \right] ,\tag{11}
$$

where $z _ { i } ^ { T }$ and $z _ { i } ^ { S }$ are unit-normalized embeddings of aligned observations, N is the batch size, and $\tau$ is a learnable temperature parameter.

For each embedding, the aligned embedding from the other agent is treated as positive, while all other embeddings from that agent in the batch serve as negatives. This encourages agreement only on information shared between the teacher and student observations and suppresses factors specific to each [8].

## 3.2 Aligning learned embeddings

The shared policy $\pi _ { \theta }$ is trained on teacher embeddings but deployed on student embeddings. To enable this direct policy transfer, paired teacher–student observations must therefore map to nearly identical latent embeddings. The contrastive objective (Eq. (11)) enforces relative similarity by distinguishing aligned pairs from negatives, but it does not minimize the distance between positive pairs directly. Therefore, small residual differences can remain and translate into an imitation gap, especially as reinforcement learning policies can be highly sensitive to input perturbations [5]. To address this, we introduce an absolute pairwise alignment objective that directly maximizes the cosine similarity between embeddings of paired observations:

$$
\mathcal { L } _ { \mathrm { a l i g n m e n t } } = - \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \frac { ( \boldsymbol { z } _ { i } ^ { T } ) ^ { \top } \boldsymbol { z } _ { i } ^ { S } } { \| \boldsymbol { z } _ { i } ^ { T } \| _ { 2 } \| \boldsymbol { z } _ { i } ^ { S } \| _ { 2 } } .\tag{12}
$$

Empirically, we find that this term is critical for reducing the imitation gap and improving transfer performance (cf. Section 5.3).

## 3.3 Stabilizing policy learning

In RAIL, the encoder parameters are updated throughout training, which induces a non-stationary input distribution for the policy. As the embedding space evolves, the same observation may map to different latent representations, resulting in unstable policy outputs. Moreover, contrastive representations may continue to drift even after alignment appears saturated [1] and further increase this non-stationarity.

We therefore regularize encoder updates to preserve the current policy behavior. Specifically, we penalize changes in the policy outputs induced by representation updates:

$$
\mathcal { L } _ { \mathrm { s t a b i l i t y } } = - \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \left( \frac { ( { \boldsymbol { u } } _ { \mathrm { o l d } , i } ^ { T } ) ^ { \top } { \boldsymbol { u } } _ { i } ^ { T } } { \| { \boldsymbol { u } } _ { \mathrm { o l d } , i } ^ { T } \| _ { 2 } \| { \boldsymbol { u } } _ { i } ^ { T } \| _ { 2 } } + \frac { ( { \boldsymbol { u } } _ { \mathrm { o l d } , i } ^ { S } ) ^ { \top } { \boldsymbol { u } } _ { i } ^ { S } } { \| { \boldsymbol { u } } _ { \mathrm { o l d } , i } ^ { S } \| _ { 2 } \| { \boldsymbol { u } } _ { i } ^ { S } \| _ { 2 } } \right) ,\tag{13}
$$

where $u _ { i } ^ { T } , u _ { i } ^ { S }$ denote the current outputs of the policy network (action logits for discrete policies and distribution parameters for continuous policies), and $u _ { \mathrm { o l d } , i } ^ { T } , u _ { \mathrm { o l d } , i } ^ { S }$ are the corresponding outputs for the same observations computed using a copy of the encoder frozen at the start of the encoder updates.

## 3.4 Training procedure

RAIL can be implemented as a standard reinforcement learning loop with an additional representationlearning phase inside each iteration. At iteration t, the agent first collects rollouts using the current teacher-side policy, $a \sim \pi _ { \theta } ( \cdot \mid \phi _ { T } ( o ^ { T } ) )$ ), exactly as in ordinary policy optimization [22]. The resulting rollout buffer contains rewards, actions, and paired teacher–student observations $( \bar { o } ^ { T } , o ^ { S } )$ from the same underlying states.

Before updating the policy, RAIL performs K encoder updates on minibatches sampled from this buffer. During this inner loop, the policy is held fixed and the encoders are trained to map paired teacher and student observations to a shared latent space. The encoder loss combines the contrastive, alignment, and stability objectives:

$$
\mathcal { L } _ { \mathrm { e m b e d } } = \lambda _ { \mathrm { c o n } } \mathcal { L } _ { \mathrm { c o n t r a s t i v e } } + \lambda _ { \mathrm { a l i g n } } \mathcal { L } _ { \mathrm { a l i g n m e n t } } + \lambda _ { \mathrm { s t a b } } \mathcal { L } _ { \mathrm { s t a b i l i t y } } .\tag{14}
$$

The stability term is computed relative to a frozen copy of the encoders taken before the inner loop, so that representation updates preserve the current policy behavior. Note that in our experimental evaluation (cf. Section 5) we used $\lambda _ { \mathrm { c o n } } = \lambda _ { \mathrm { a l i g n } } = \bar { \lambda } _ { \mathrm { s t a b } } \dot { = } 1$ throughout.

After the encoder updates, the encoders are frozen and RAIL performs the usual RL policy update using teacher embeddings $\phi _ { T } ( o ^ { T } )$ from the rollout buffer. Crucially, gradients from the RL objective are not propagated into the encoders. Without this gradient blocking, policy optimization drives the representation to encode teacher-private information that improves teacher performance but is unrecoverable from student observations, thereby breaking direct transfer. Thus, RAIL keeps the outer loop identical to standard teacher-side RL while inserting an inner contrastive loop that maintains student-compatible representations. Algorithm 1 summarizes the overall training procedure.

One implementation detail not explicitly shown in the algorithm is the computation of the stability objective. At the beginning of each representation-learning phase, we store frozen copies of the current encoders, denoted $\bar { \phi } _ { T }$ and $\bar { \phi } _ { S } .$ . These reference encoders remain fixed throughout the subsequent K representation updates and define the policy behavior prior to representation refinement. The stability loss penalizes changes in the policy outputs induced by updates to the embeddings relative to these frozen reference encoders.

Algorithm 1 Representation-Aligned Imitation Learning (RAIL)   
Require: Teacher encoder $\phi _ { T } ,$ student encoder $\phi _ { S } ,$ , policy $\pi _ { \theta }$   
1: for each training iteration do   
2: Collect teacher rollouts using $\pi _ { \theta } \big ( \phi _ { T } ( o ^ { T } ) \big )$   
3: Store aligned teacher–student observations   
4: for K iterations with $\pi _ { \theta }$ fixed do   
5: Update $\phi _ { T } , \phi _ { S }$ with $\mathcal { L } _ { \mathrm { e m b e d } }$   
6: end for   
7: Update policy $\pi _ { \theta }$ with RL while keeping $\phi _ { T } , \phi _ { S }$ fixed   
8: end for

## 4 Related Work

Teacher–student setups in imitation learning commonly train a privileged RL teacher with state information and use behavior cloning (BC) [18] or dataset aggregation (DAgger) [21] to distill it into a deployable student policy operating on restricted or high-dimensional observations [2, 35]. This improves teacher learning, but creates a fundamental mismatch: when the teacher uses information unavailable to the student, the student cannot, in general, reproduce its actions, which yields an irreducible imitation gap [25, 31, 32]. In this case, imitation marginalizes over information private to the teacher, leading to suboptimal student policies.

Most prior work treats this mismatch as a student-side optimization problem after the teacher has already been trained. Several methods follow imitation with additional reinforcement learning, either by bootstrapping the policy with imitation learning [33], incorporating teacher guidance into the student’s RL objective [13, 31, 32], or adaptively querying the teacher to improve learning under partial observability [7]. COSIL [14] augments the student’s RL objective with a soft imitation term that encourages agreement with a privileged teacher. These methods can recover performance when the student can compensate through exploration, but, as we show in Section 5, can severely hinder learning when the optimal behavior differs substantially across teacher and student observations.

A second line of work enforces imitability during teacher training. Monaci et al. [12] explicitly restrict the teacher observation space to state variables that are known to be recoverable by the student, which requires task-specific prior knowledge. Reward-shaping methods encourage student-compatible behavior through auxiliary objectives [11, 26], but involve careful reward design. SITT [10] jointly trains teacher and student and penalizes teacher behavior that the student cannot match. However, it modifies the original task objective and introduces a sensitive environment-dependent trade-off hyperparameter. RAIL instead preserves the original reward function and enforces imitability by training the teacher on a shared latent space that enables direct transfer of the learned policy.

More broadly, our work relates to representation learning methods for transfer under domain shifts. Shon et al. [24] construct a common latent space for robotic imitation of human motions. Subsequent work learns representations that are invariant to domain changes using multi-view contrastive objectives [4, 23] or adversarial domain confusion [27]. These approaches combine representation learning with policy training to remove nuisance variation (e.g., viewpoint or background), while the same task-related information is present across views. Under information asymmetry, however, transfer requires excluding information that is not recoverable from the student observations, which we enforce by stopping policy gradients from updating the encoders. Giammarino et al. [6] train adversarial imitation learning in a contrastive latent space and report failure under this constraint, which highlights the importance of our stability and alignment objectives.

Table 1: Environments and imitation gaps used in experiments. Further details and the reward functions are provided in Appendix A.
<table><tr><td></td><td><img src="images/2f3a335610b80598371e54182d5fb268fa96c7e0b3efddea1cc9f3f5d44897fd.jpg"/></td><td>TunnelVision</td><td>CollectHealth</td><td><img src="images/71dbafeb47064bc1e1eb67f2f25df1c0d75e7e745ec41dd301f73e3bd64258f2.jpg"/></td></tr><tr><td>Task</td><td>Navigate from center to the hidden target corner (green).</td><td>Reach the goal (green) while avoiding obstacles (red).</td><td>Collect all health boxes in the room while avoid- ing walls.</td><td>Run forward as fast as possible.</td></tr><tr><td>Teacher obs.</td><td>Position, previous action, and the target corner identity.</td><td>Position, previous action, and occupancy of all neighboring cells (blue).</td><td>Position, previous action, and relative coordinates of all remaining boxes.</td><td>Full proprioceptive state (17D joint positions and velocities).</td></tr><tr><td>Student obs.</td><td>Position and previous ac- tion only; target identity is hidden.</td><td>Position, previous action, and three neighboring cells facing its current orientation (orange).</td><td>Stack of 5 limited field- of-view first-person RGB observations.</td><td>Third-person RGB obser- vation with partial oc- clusion (left half is re- moved).</td></tr><tr><td>Action space</td><td>Move to any neighboring cells,  $| { \mathcal { A } } | = 8 .$ </td><td>Move or rotate to com- pass directions, |A| = 8.</td><td>Move in 8 directions or rotate ±30°, |A| = 10.</td><td>Continuous torques, |.A| = 6.</td></tr><tr><td>gap</td><td>rectly to the target, stu- dent must visit all cor- ners.</td><td>Teacher can move di- rectly to goal, student must turn to discover ob- stacles.</td><td>Teacher can move with- out turning, student must turn to discover next tar- get.</td><td>Teacher has full state; stu- dent must infer dynamics and state from partial vi- sual observations.</td></tr></table>

## 5 Experiments

We evaluate RAIL in simulation environments with asymmetric observation spaces that induce an irreducible imitation gap. The teacher has access to privileged information, whereas the student operates under restricted observations, which makes direct imitation of an unconstrained teacher insufficient. Table 1 summarizes the environments; full details are provided in Appendix A. Our experiments address four questions: (i) whether RAIL reduces the imitation gap, (ii) how it compares to strong baselines, (iii) the contribution of its components, and (iv) whether the learned representation can be used to zero-shot transfer new behaviors through teacher-only training.

## 5.1 Experimental setup

We compare RAIL against four imitation learning baselines, along with direct RL from student observations (Student RL) as a reference and which requires reward signals in the student domain.

(1) Behavior Cloning (BC) [18] We train the teacher with RL, and the student is trained offline to imitate teacher actions via supervised learning on collected trajectories.

## (2) DAgger [21]

We use the same state-based teacher as BC and iteratively collect transitions under the student policy, label them with teacher actions at the visited states, aggregate the dataset, and train the student via supervised imitation.

(3) COSIL [14] We train the student with RL augmented by a soft imitation term that encourages agreement with a privileged teacher. Following Messikommer et al. [10], we use the PPO variant with a fixed target divergence of 0.5.

(4) SITT [10] We jointly train the teacher with RL and student with BC and penalize the teacher when its actions are difficult to imitate. The trade-off parameter is selected via grid search.

We use Proximal Policy Optimization (PPO) [22] for all RL-based methods. All methods use comparable network capacity, matched training budgets, and identical PPO hyperparameters where applicable. Each method is trained with five random seeds and evaluated on 100 environment seeds

Table 2: Teacher performance (T), student performance after transfer (S), and imitation gap $( \Delta =$ $T - S )$ . Performance is measured by success rate for Corners, TunnelVision, and CollectHealth, and episodic return for QuarterCheetah. Success rates show the mean with upper and lower semideviations; QuarterCheetah returns show the mean ± standard deviation. Higher $\ddot { S }$ and lower ∆ are better.
<table><tr><td></td><td colspan="3">Corners</td><td colspan="3">TunnelVision</td><td colspan="3">CollectHealth</td><td colspan="3">QuarterCheetah</td></tr><tr><td>Method</td><td>T</td><td>S↑</td><td>∆↓</td><td>T</td><td>S↑</td><td>∆↓</td><td>T</td><td>S↑</td><td>∆↓</td><td>T</td><td>S↑</td><td>∆↓</td></tr><tr><td>BC</td><td> $1 . 0 0 _ { - 0 . 0 0 } ^ { + 0 . 0 0 }$ </td><td> $0 . 2 4 _ { - 0 . 0 2 } ^ { + 0 . 0 3 }$ </td><td>0.76</td><td> $1 . 0 0 _ { - 0 . 0 0 } ^ { + 0 . 0 0 }$ </td><td> $0 . 1 2 _ { - 0 . 0 3 } ^ { + 0 . 0 3 }$ </td><td>0.88</td><td> $1 . 0 0 _ { - 0 . 0 0 } ^ { + 0 . 0 0 }$ </td><td> $0 . 2 8 _ { - 0 . 0 9 } ^ { + 0 . 1 1 }$ </td><td>0.72</td><td> $2 1 8 6 . 8 3 \pm 8 2 . 4 7$ </td><td> $- 2 6 8 . 1 7 \pm 2 9 . 0 2$ </td><td>2455.00</td></tr><tr><td>DAgger</td><td> $1 . 0 0 _ { - 0 . 0 0 } ^ { + 0 . 0 0 }$ </td><td> $0 . 3 8 _ { - 0 . 1 1 } ^ { + 0 . 1 5 }$ </td><td>0.62</td><td> $1 . 0 0 _ { - 0 . 0 0 } ^ { + 0 . 0 0 }$ </td><td> $0 . 1 5 _ { - 0 . 0 2 } ^ { + 0 . 0 2 }$ </td><td>0.85</td><td> $1 . 0 0 _ { - 0 . 0 0 } ^ { + 0 . 0 0 }$ </td><td> $0 . 8 4 _ { - 0 . 0 1 } ^ { + 0 . 0 2 }$ </td><td>0.16</td><td> $2 1 8 6 . 8 3 \pm 8 2 . 4 7$ </td><td> $- 4 8 4 . 2 5 \pm 2 3 . 1 4$ </td><td>2671.08</td></tr><tr><td>COSIL</td><td> $1 . 0 0 _ { - 0 . 0 0 } ^ { + 0 . 0 0 }$ </td><td> $0 . 2 8 _ { - 0 . 0 6 } ^ { + 0 . 0 9 }$ </td><td>0.72</td><td> $1 . 0 0 _ { - 0 . 0 0 } ^ { + 0 . 0 0 }$ </td><td> $0 . 0 0 _ { - 0 . 0 0 } ^ { + 0 . 0 0 }$ </td><td>1.00</td><td> $1 . 0 0 _ { - 0 . 0 0 } ^ { + 0 . 0 0 }$ </td><td> $0 . 0 8 _ { - 0 . 0 1 } ^ { + 0 . 0 1 }$ </td><td>0.92</td><td> $2 1 8 6 . 8 3 \pm 8 2 . 4 7$ </td><td> $- 3 0 8 . 0 4 \pm 1 4 . 4 6$ </td><td>2494.87</td></tr><tr><td>SITT</td><td> $0 . 9 3 _ { - 0 . 1 3 } ^ { + 0 . 0 6 }$ </td><td> $0 . 3 7 _ { - 0 . 1 1 } ^ { + 0 . 1 0 }$ </td><td>0.56</td><td> $0 . 9 5 _ { - 0 . 0 3 } ^ { + 0 . 0 3 }$ </td><td> $0 . 7 4 _ { - 0 . 1 6 } ^ { + 0 . 1 0 }$ </td><td>0.21</td><td> $0 . 9 3 _ { - 0 . 0 4 } ^ { + 0 . 0 4 }$ </td><td> $0 . 4 9 _ { - 0 . 0 2 } ^ { + 0 . 0 3 }$ </td><td>0.44</td><td> $9 6 2 . 6 7 \pm 2 5 6 . 1 7$ </td><td> $7 2 1 . 1 2 \pm 2 3 9 . 6 3$ </td><td>241.55</td></tr><tr><td>RAIL (ours)</td><td> $1 . 0 0 _ { - 0 . 0 0 } ^ { + 0 . 0 0 }$ </td><td> $\mathbf { 1 . 0 0 } _ { - 0 . 0 0 } ^ { + 0 . 0 0 }$ </td><td></td><td> $\begin{array} { l l } { 0 . 0 0 } & { 1 . 0 0 _ { - 0 . 0 0 } ^ { + 0 . 0 0 } } \end{array}$ </td><td> $\mathbf { 1 . 0 0 } _ { - 0 . 0 0 } ^ { + 0 . 0 0 }$ </td><td>0.00</td><td> $0 . 9 9 _ { - 0 . 0 1 } ^ { + 0 . 0 0 }$ </td><td> $\mathbf { 0 . 9 8 } _ { - 0 . 0 2 } ^ { + 0 . 0 1 }$ </td><td>0.01</td><td> $2 6 2 7 . 6 7 \pm 3 3 . 8 3$ </td><td> $2 5 1 0 . 2 4 \pm 7 2 . 0 5$ </td><td>117.43</td></tr><tr><td>RAIL w/o stability</td><td> $1 . 0 0 _ { - 0 . 0 0 } ^ { + 0 . 0 0 }$ </td><td> $1 . 0 0 _ { - 0 . 0 0 } ^ { + 0 . 0 0 }$ </td><td>0.00</td><td> $0 . 5 5 _ { - 0 . 3 1 } ^ { + 0 . 2 9 }$ </td><td> $0 . 5 3 _ { - 0 . 1 7 } ^ { + 0 . 1 6 }$ </td><td>0.02</td><td> $0 . 9 7 _ { - 0 . 0 3 } ^ { + 0 . 0 2 }$ </td><td> $0 . 9 5 _ { - 0 . 0 6 } ^ { + 0 . 0 4 }$ </td><td>0.02</td><td> $2 3 5 2 . 7 9 \pm 1 8 0 . 6 4$ </td><td>2343.32 ± 247.17</td><td>9.47</td></tr><tr><td>RAIL w/o alignment</td><td> $0 . 9 8 _ { - 0 . 0 4 } ^ { + 0 . 0 2 }$ </td><td> $0 . 9 8 _ { - 0 . 0 4 } ^ { + 0 . 0 2 }$ </td><td></td><td> $0 . 0 0 ~ 0 . 9 6 _ { - 0 . 0 7 } ^ { + 0 . 0 4 }$ </td><td> $0 . 6 6 _ { - 0 . 2 2 } ^ { + 0 . 1 4 }$ </td><td></td><td> $0 . 3 0 \ : \ : 0 . 9 2 _ { - 0 . 0 6 } ^ { + 0 . 0 4 }$ </td><td> $0 . 7 8 _ { - 0 . 1 2 } ^ { + 0 . 1 0 }$ </td><td>0.14</td><td> $2 6 3 4 . 8 0 \pm 7 5 . 3 8$ </td><td> $2 4 7 6 . 8 6 \pm 1 1 6 . 8 2$ </td><td>157.94</td></tr><tr><td>RAIL end-to-end</td><td> $1 . 0 0 _ { - 0 . 0 0 } ^ { + 0 . 0 0 }$ </td><td> $0 . 2 8 _ { - 0 . 0 2 } ^ { + 0 . 0 2 }$ </td><td></td><td> $0 . 7 2 ~ 0 . 9 9 _ { - 0 . 0 0 } ^ { + 0 . 0 0 }$ </td><td> $0 . 1 8 _ { - 0 . 1 1 } ^ { + 0 . 2 1 }$ </td><td> $0 . 8 1 ~ 0 . 9 3 _ { - 0 . 0 1 } ^ { + 0 . 0 1 }$ </td><td></td><td> $0 . 7 1 _ { - 0 . 1 1 } ^ { + 0 . 1 4 }$ </td><td>0.22</td><td> $2 2 8 3 . 1 9 \pm 9 1 . 3 5$ </td><td> $1 8 9 8 . 9 6 \pm 5 6 9 . 9 2$ </td><td>384.23</td></tr><tr><td>RAIL sep. encoders</td><td> $0 . 9 1 _ { - 0 . 0 9 } ^ { + 0 . 0 7 }$ </td><td> $0 . 8 9 _ { - 0 . 1 0 } ^ { + 0 . 0 8 }$ </td><td></td><td> $0 . 0 2 ~ 0 . 9 5 _ { - 0 . 0 4 } ^ { + 0 . 0 4 }$ </td><td> $0 . 6 0 _ { - 0 . 1 1 } ^ { + 0 . 0 9 }$ </td><td>0.35</td><td> $0 . 9 8 _ { - 0 . 0 1 } ^ { + 0 . 0 1 }$ </td><td> $0 . 9 8 _ { - 0 . 0 2 } ^ { + 0 . 0 1 }$ </td><td>0.00</td><td> $1 8 1 3 . 8 0 \pm 7 3 3 . 1 5$ </td><td>1690.38 ± 917.19</td><td>123.42</td></tr><tr><td>Student RL</td><td>一</td><td> $1 . 0 0 _ { - 0 . 0 0 } ^ { + 0 . 0 0 }$ </td><td>一</td><td>一</td><td> $1 . 0 0 _ { - 0 . 0 0 } ^ { + 0 . 0 0 }$ </td><td>一</td><td>一</td><td> $0 . 7 8 _ { - 0 . 0 8 } ^ { + 0 . 0 9 }$ </td><td>一</td><td>一</td><td>1824.27 ± 30.35</td><td>一</td></tr></table>

per run; we report averages across seeds. Full implementation details are provided in Appendix C. The implementation is available at https://github.com/meraccos/RAIL.

## 5.2 Experimental results

We report teacher performance (T), student performance after transfer (S), and the imitation gap $( \Delta = T - S )$ . For Corners, TunnelVision, and CollectHealth, performance is measured by success rate. For QuarterCheetah, performance is measured by episodic return. Higher S indicates stronger transfer, while lower ∆ indicates more reproducible teacher behavior. Results are summarized in Table 2, and evaluation curves for the higher-dimensional environments are shown in Figure 3.

Baseline results. BC performs poorly throughout. When the teacher relies on privileged information, identical student observations can correspond to contradictory teacher actions and BC cannot match these from student observations alone. DAgger improves over BC by mitigating covariate shift through student rollouts, but querying the same privileged teacher leaves this information mismatch unresolved. COSIL also performs poorly: its auxiliary imitation objective can conflict with reward maximization under student observations by encouraging actions that are suboptimal under student observations. SITT is the strongest prior baseline, indicating the benefit of changing the teacher’s behavior to make it more imitable. However, reward-level regularization remains insufficient when the teacher’s private information induces qualitatively different strategies. The student’s gains also come with degraded teacher performance, which reflects an explicit trade-off between task optimality and imitability. Student RL can solve Corners and TunnelVision better than all other baselines, showing that these approaches can actually impede student performance.

![](images/f8ce4eddfcf77f826f929f463e5a005812e548c8cc236f1741981806cbdf213c.jpg)

![](images/866ea2fc89c04ce81e1f0014ab861ebece958f162db26110d26fa5378f113151.jpg)  
Figure 3: Student evaluation curves for CollectHealth and QuarterCheetah. CollectHealth reports success rate, while QuarterCheetah reports episodic return. BC and DAgger are shown as horizontal lines corresponding to their final student performance.

To test whether the failure of the baselines arises from information mismatch, as we argue above, we repeat the training protocol but with matched information content: students receive teacher observations, or unmasked images in QuarterCheetah. In this experiment, all baselines match or approach teacher performance across environments, which supports our explanation (Appendix D.3).

RAIL results. RAIL consistently achieves the highest student performance and the smallest imitation gap across all environments. In Corners and TunnelVision, it attains perfect transfer (S=1.00) with zero gap, fully matching teacher performance. In CollectHealth, it reaches S=0.98 with a negligible gap of 0.01 and substantially outperforms all baselines. Compared to Student RL, RAIL yields higher performance in CollectHealth (0.98 vs. 0.78) and QuarterCheetah (2510.24 vs. 1824.27), showing that improved representations and decoupled representation learning can translate into substantially better performance in high-dimensional observation spaces, which is consistent with prior findings in RL [9, 28]. Notably, RAIL maintains strong teacher performance while achieving strong transfer, unlike SITT, which reduces the gap at the cost of degraded teacher returns. This likely reflects the difficulty of balancing task reward and imitation constraints, together with sensitivity to the reward trade-off. Figure 3 further shows that RAIL reaches strong student performance faster than the online baselines in both high-dimensional environments.

Representation analysis. To assess what information can be recovered from the learned teacher embeddings, we train two-layer prediction networks on frozen embeddings in the Corners environment. Across ten seeds, prediction of the privileged target corner from the initial state achieves 25.4 ± 1.4% accuracy, close to the 25% chance level, whereas prediction of the agent’s coordinates, which are available to both agents, achieves 100.0 ± 0.0%. This provides evidence that the teacher embeddings retain shared state information while suppressing information private to the teacher.

## 5.3 Ablation study

The ablations in Table 2 (lower half) isolate the contribution of each component. Removing the alignment objective (RAIL w/o alignment) significantly reduces transfer performance, particularly in TunnelVision and CollectHealth. Although the contrastive objective encourages agreement, it does not enforce sufficient embedding similarity for direct policy transfer, and small mismatches in the embeddings translate into large imitation gaps.

Removing the stability objective (RAIL w/o stability) has little effect in Corners and CollectHealth, but reduces both teacher and student performance in TunnelVision and QuarterCheetah. The imitation gap nevertheless remains small across all environments, suggesting that stability primarily improves the teacher’s training, while alignment helps keep student behavior close to the teacher’s.

Allowing gradients from the policy to update the encoder (RAIL end-to-end) leads to large imitation gaps across all tasks despite strong teacher performance. This confirms that RL updates without gradient blocking drive the representation to encode teacher-specific information, resulting in policies that are no longer compatible with student embeddings.

Replacing the shared encoder with separate encoders (RAIL sep. encoders) reduces student performance and increases variability in Corners, TunnelVision, and QuarterCheetah, with little effect in CollectHealth. Consistent with prior work [17], this suggests that parameter sharing stabilizes the learning of common embeddings.

Finally, to assess RAIL’s sensitivity to the contrastive batch size, we evaluate batch sizes from 128 to 1024 while holding the remaining training setup fixed. Performance remains high across this range, with larger batches typically improving success rates and reducing variability (Appendix D.2).

## 5.4 Zero-shot transfer to new tasks

We evaluate whether the learned representations support transferring new behaviors from the teacher agent without collecting additional student data. We modify Corners, TunnelVision and CollectHealth to support two tasks that require different control strategies while sharing the same observation manifold. The task changes involve switching from corner to side targets in Corners, adding diagonal actions in TunnelVision, and switching the rewarded object class in CollectHealth.

We first train RAIL on an initial task, freeze the encoder networks, and then train a new policy from scratch using only teacher observations on the second task. The resulting policy is transferred zero-shot by pairing it with the frozen student encoder. For comparison, we also train complete RAIL, including its encoders and policy, and Student RL from scratch on the second task.

Across all environments, policies trained on the frozen RAIL representation match or surpass the performance of Student RL, despite using no student observations during adaptation. This demonstrates that the learned latent space captures rich environment structure shared across observations rather than features specific to a single objective. The transferred policies also perform on par with RAIL trained on the second task directly, indicating that the representation is sufficiently expressive to support new policies without modification. Detailed descriptions of the modified environments and complete experimental results are provided in Appendix B.

## 6 Conclusions

We introduced RAIL, a representation-level approach to teacher–student imitation under asymmetric observations. Rather than training a privileged teacher first and correcting the resulting mismatch afterward, RAIL constrains the teacher to act on representations that are also shared with the student. Empirically, RAIL achieves a near-zero imitation gap and outperforms strong baselines across environments. We further show that the learned representation can be reused for new tasks in the same environment through teacher-only policy training.

## Limitations

RAIL assumes access to aligned teacher–student observations generated from the same underlying environment states, which is standard in teacher–student imitation learning but may not hold when observations are collected independently or only partially overlap temporally. Moreover, while our experiments show that gradient blocking and stability regularization substantially reduce the imitation gap, we do not provide formal guarantees that the learned policy is fully realizable under the student observation space. The zero-shot study evaluates task transfer within the same environments and observation spaces; the conclusions are therefore specific to this setting.

## Acknowledgments and Disclosure of Funding

This work was partially supported by the Wallenberg AI, Autonomous Systems and Software Program (WASP) funded by the Knut and Alice Wallenberg Foundation. The authors declare no competing interests.

## References

[1] Roy Betser, Eyal Gofer, Meir Yossef Levi, and Guy Gilboa. “InfoNCE Induces Gaussian Distribution”. In: International Conference on Learning Representations. 2026.

[2] Dian Chen, Brady Zhou, Vladlen Koltun, and Philipp Krähenbühl. “Learning by Cheating”. In: Conference on Robot Learning. 2019.

[3] Maxime Chevalier-Boisvert, Bolun Dai, Mark Towers, Rodrigo de Lazcano, Lucas Willems, Salem Lahlou, Suman Pal, Pablo Samuel Castro, and Jordan Terry. Minigrid & Miniworld: Modular & Customizable Reinforcement Learning Environments for Goal-Oriented Tasks. 2023. arXiv: 2306.13831.

[4] André Correia and Luís A. Alexandre. “Multi-View Contrastive Learning from Demonstrations”. In: IEEE International Conference on Robotic Computing. 2022.

[5] Jesse Farebrother, Marlos C. Machado, and Michael Bowling. Generalization and Regularization in DQN. 2018. arXiv: 1810.00123.

[6] Vittorio Giammarino, James Queeney, and Ioannis Ch. Paschalidis. “Visually Robust Adversarial Imitation Learning from Videos with Contrastive Learning”. In: International Conference on Robotics and Automation. 2025.

[7] Yujin Kim, Nathaniel Chin, Arnav Vasudev, and Sanjiban Choudhury. Distilling Realizable Studentsfrom Unrealizable Teachers. 2025. arXiv: 2505.09546.

[8] Julius von Kügelgen, Yash Sharma, Luigi Gresele, Wieland Brendel, Bernhard Schölkopf, Michel Besserve, and Francesco Locatello. “Self-Supervised Learning with Data Augmentations Provably Isolates Content from Style”. In: Advances in Neural Information Processing Systems. 2021.

[9] Michael Laskin, Aravind Srinivas, and Pieter Abbeel. “CURL: Contrastive Unsupervised Representations for Reinforcement Learning”. In: International Conference on Machine Learning. 2020.

[10] Nico Messikommer, Jiaxu Xing, Elie Aljalbout, and Davide Scaramuzza. “Student-Informed Teacher Training”. In: International Conference on Learning Representations. 2025.

[11] Nico Messikommer, Jiaxu Xing, Leonard Bauersfeld, Marco Cannici, Elie Aljalbout, and Davide Scaramuzza. Approximate Imitation Learningfor Event-based Quadrotor Flight in Cluttered Environments. 2026. arXiv: 2603.07578.

[12] Gianluca Monaci, Michel Aractingi, and Tomi Silander. “DiPCAN: Distilling Privileged Information for Crowd-Aware Navigation”. In: Robotics: Science and Systems. 2022.

[13] Malte Mosbach and Sven Behnke. “Grasp Anything: Combining Teacher-Augmented Policy Gradient Learning with Instance Segmentation to Grasp Arbitrary Objects”. In: IEEE International Conference on Robotics and Automation. 2024.

[14] Hai Nguyen, Andrea Baisero, Dian Wang, Christopher Amato, and Robert Platt. “Leveraging Fully Observable Policies for Learning under Partial Observability”. In: Conference on Robot Learning. 2022.

[15] Aaron van den Oord, Yazhe Li, and Oriol Vinyals. Representation Learning with Contrastive Predictive Coding. 2018. arXiv: 1807.03748.

[16] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. “PyTorch: An Imperative Style, High-Performance Deep Learning Library”. In: Advances in Neural Information Processing Systems. 2019.

[17] Petra Poklukar, Miguel Vasco, Hang Yin, Francisco S. Melo, Ana Paiva, and Danica Kragic. “Geometric Multimodal Contrastive Representation Learning”. In: International Conference on Machine Learning. 2022.

[18] Dean A. Pomerleau. “ALVINN: An autonomous land vehicle in a neural network”. In: Advances in Neural Information Processing Systems. 1989.

[19] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. “Learning Transferable Visual Models From Natural Language Supervision”. In: International Conference on Machine Learning. 2021.

[20] Antonin Raffin, Ashley Hill, Adam Gleave, Anssi Kanervisto, Maximilian Ernestus, and Noah Dormann. “Stable-Baselines3: Reliable Reinforcement Learning Implementations”. In: Journal ofMachine Learning Research 22.268 (2021).

[21] Stéphane Ross, Geoffrey J. Gordon, and J. Andrew Bagnell. “A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning”. In: International Conference on Artificial Intelligence and Statistics. 2011.

[22] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal Policy Optimization Algorithms. 2017. arXiv: 1707.06347.

[23] Pierre Sermanet, Corey Lynch, Yevgen Chebotar, Jasmine Hsu, Eric Jang, Stefan Schaal, and Sergey Levine. “Time-Contrastive Networks: Self-Supervised Learning from Video”. In: IEEE International Conference on Robotics and Automation. 2018.

[24] Aaron P. Shon, Keith Grochow, Aaron Hertzmann, and Rajesh P. N. Rao. “Learning Shared Latent Structure for Image Synthesis and Robotic Imitation”. In: Advances in Neural Information Processing Systems. 2005.

[25] Yuda Song, Dhruv Rohatgi, Aarti Singh, and J. Bagnell. “To Distill or Decide? Understanding the Algorithmic Trade-off in Partially Observable RL”. In: Advances in Neural Information Processing Systems. Vol. 38, Main Conference. 2025. DOI: 10.52202/085713-2996.

[26] Yunlong Song, Kexin Shi, Robert Penicka, and Davide Scaramuzza. “Learning Perception-Aware Agile Flight in Cluttered Environments”. In: IEEE International Conference on Robotics and Automation. 2023.

[27] Bradly C. Stadie, Pieter Abbeel, and Ilya Sutskever. “Third-Person Imitation Learning”. In: International Conference on Learning Representations. 2017.

[28] Adam Stooke, Kimin Lee, Pieter Abbeel, and Michael Laskin. “Decoupling Representation Learning from Reinforcement Learning”. In: International Conference on Machine Learning. 2021.

[29] Emanuel Todorov, Tom Erez, and Yuval Tassa. “MuJoCo: A physics engine for model-based control”. In: International Conference on Intelligent Robots and Systems. 2012.

[30] Mark Towers, Ariel Kwiatkowski, Jordan K. Terry, John U. Balis, Gianluca De Cola, Tristan Deleu, Manuel Goulão, Andreas Kallinteris, Markus Krimmel, Arjun KG, Rodrigo Perez-Vicente, Andrea Pierré, Sander Schulhoff, Jun Jet Tai, Hannah Tan, and Omar G. Younis. “Gymnasium: A Standard Interface for Reinforcement Learning Environments”. In: Advances in Neural Information Processing Systems. Vol. 38. 2025. DOI: 10.52202/085713-4916.

[31] Aaron Walsman, Muru Zhang, Sanjiban Choudhury, Dieter Fox, and Ali Farhadi. “Impossibly Good Experts and How to Follow Them”. In: International Conference on Learning Representations. 2023.

[32] Luca Weihs, Unnat Jain, Iou-Jen Liu, Jordi Salvador, Svetlana Lazebnik, Aniruddha Kembhavi, and Alex Schwing. “Bridging the Imitation Gap by Adaptive Insubordination”. In: Advances in Neural Information Processing Systems. 2021.

[33] Jiaxu Xing, Angel Romero, Leonard Bauersfeld, and Davide Scaramuzza. “Bootstrapping Reinforcement Learning with Imitation for Vision-Based Agile Flight”. In: Conference on Robot Learning. 2024.

[34] Tian Xu, Ziniu Li, and Yang Yu. “Error Bounds of Imitating Policies and Environments”. In: Advances in Neural Information Processing Systems. 2020.

[35] Zhejun Zhang, Alexander Liniger, Dengxin Dai, Fisher Yu, and Luc Van Gool. “End-to-End Urban Driving by Imitating a Reinforcement Learning Coach”. In: IEEE/CVF International Conference on Computer Vision. 2021.

## A Environments

Corners is a grid-world navigation task on an $n \times n$ grid $( n { = } 9 )$ , where the agent starts at the center and must reach a target corner sampled uniformly at the beginning of each episode. The action space consists of 8 discrete movements (cardinal and diagonal), with positions clipped to remain within the grid. Observations encode the agent position $( x , y )$ as one-hot vectors of size n each, and the previous action as an 8-dimensional one-hot vector over $\{ 0 , \ldots , 7 \}$ . The teacher additionally observes the target corner as a 4-dimensional one-hot vector, while the student does not, yielding dim $( o ^ { T } ) = 2 n { \bar { + } } 1 2$ and dim $\begin{array} { r } { \ i ( o ^ { S } ) = 2 n + 8 . } \end{array}$ . This private goal information induces an imitation gap, as the teacher can move directly to the target while the student must infer it through exploration. Episodes terminate upon reaching the target and are truncated after 60 steps. The reward function is $r _ { t } = 1 \{ \mathrm { g o a l r e a c h e d } \} - 0 . 0 1$

TunnelVision is a grid-world navigation task on an $n \times n$ grid $( n { = } 8 )$ surrounded by a one-cell obstacle border, where the agent starts at (1, 1) and must reach the fixed goal at $( n - 2 , n - 2 )$ . Obstacles are sampled randomly at reset with probability 0.5, subject to constraints that remove diagonal obstacle patterns and preserve a valid path from start to goal. The agent can move or rotate in the four compass directions, yielding 8 discrete actions. Observations contain the local $3 \times 3$ occupancy grid, 4-dimensional one-hot-coded orientation, normalized previous action, and normalized position. The teacher observes all 9 cells in the local neighborhood, while the student observes only the 3 cells in front of its current orientation, with unseen cells masked by a constant; both observations have dimension $\dim ( o ^ { T } ) = \dim ( o ^ { S } ) = 1 6$ . This asymmetry in information induces an imitation gap, as the teacher can solve the task without turning while the student must reorient to reveal hidden obstacles. Episodes terminate when the agent reaches the goal or collides with an obstacle/border and are truncated after 80 steps. The reward function is $r _ { t } = \bar { 1 } \times 1 \{ \mathrm { g o a l r e a c h e d } \} - 1 \times 1 \{ \mathrm { c o l l i s i o n } \} -$ $0 . 0 1 \times \mathbf { 1 } \{ d _ { t + 1 } \geq d _ { t } \}$ , where $d _ { t }$ is the Euclidean distance to the goal before the action.

CollectHealth is an episodic 3D navigation task based on MiniWorld [3], where the agent operates in a 15 m × 15 m room and must collect 5 randomly placed health boxes. The action space has 10 discrete actions: movement in 8 uniformly spaced relative directions with step size 0.5 m, and two camera rotations of $\pm 3 0 ^ { \circ }$ . The teacher observes the relative positions of all remaining boxes in the agent frame, sorted by distance and padded with −1, together with the normalized agent position and previous action, giving dim $\left( o ^ { T } \right) = 1 3$ The student observes a stack of 5 RGB first-person images, each resized to $8 4 \times 8 4$ and normalized to [−1, 1], giving dim $( o ^ { S } ) = 5 \times 3 \times 8 \dot { 4 } \times 8 4$ This extra target information induces an imitation gap, as the teacher can move directly toward unseen boxes while the student must actively explore to locate them. A box is collected when its distance to the agent is below 1.0m. Episodes terminate when all boxes are collected or when the agent collides with a wall, and are truncated after 200 steps. The reward function is $r _ { t } = $ 10 × 1{all boxes collected} $+ 1 \times 1$ {one box collected} $\mathrm { ~ - ~ } 5 \times { \bf 1 }$ {wall collision} − 0.05. Success is defined as the fraction of collected boxes, with terminated episodes assigned zero success.

QuarterCheetah is a continuous-control task based on HalfCheetah [30], simulated in MuJoCo [29]. The agent must maximize forward velocity using a 6-dimensional continuous action space, $\mathcal { A } \subset \mathbb { R } ^ { 6 }$ The teacher observes the full 17-dimensional proprioceptive state (joint positions and velocities), while the student observes only the right half of a stack of 4 RGB renders, each resized to $8 4 \times 8 4$ and normalized $\mathrm { t o } [ - 1 , 1 ]$ , yielding dim $\left( o ^ { T } \right) = 1 7$ and dim $( o ^ { S } ) = 4 \times 3 \times 8 4 \times 8 4$ . This asymmetry in the observations induces an imitation gap, as the teacher can act directly on full state information while the student must find a working policy while seeing only the right half of the visual input. Episodes follow the default HalfCheetah horizon (1000 steps). The reward function is the standard MuJoCo HalfCheetah reward: $r _ { t } = v _ { t } - 0 . 1 \| a _ { t } \| ^ { 2 }$ , where $v _ { t }$ is forward velocity and $a _ { t }$ is the action.

## B Zero-shot transfer to new tasks

This appendix provides additional details for the zero-shot transfer experiment described in Section 5.4. The goal of this experiment is to evaluate whether the representation learned by RAIL can be reused to learn new behaviors through teacher-only training and transferred directly to the student agent.

## B.1 Modified environments

To evaluate the transfer capabilities of RAIL, we construct modified versions of Corners, TunnelVision, and CollectHealth environments (cf. Appendix A) that support two distinct tasks within the same observation space. The tasks require different behaviors while preserving the same underlying observation manifolds.

CornersPlus. We modify Corners by introducing an additional set of target locations at the midpoints of each wall. At the start of each episode, both a corner target and a side target are sampled uniformly and revealed only to the teacher. In Task 1, the agent must reach the corner target while avoiding side targets; in Task 2, it must reach the side target while avoiding corner targets. To ensure that the required behaviors differ substantially, visiting a side target in Task 1 or a corner target in Task 2 immediately terminates the episode as failure. The modified reward function is $r _ { t } = 1 \times 1$ {goal reached} $- 0 . 1 \times \mathbf { 1 }$ {termination} − 0.01.

TunnelVisionPlus. We modify TunnelVision by changing the action space while keeping observations unchanged. In Task 1, as in the original TunnelVision, the agent operates with 8 actions (four movements and four rotations). In Task 2, four diagonal movement actions are added, increasing the action space to 12. The reward and termination conditions remain identical to TunnelVision.

CollectHealthPlus. We modify CollectHealth by adding an equal number of secondary objects into the environment (rubber ducks) and changing the rewarded object class. In Task 1, medkits are rewarded and ducks penalized; in Task 2, ducks are rewarded and medkits penalized. Both medkits and ducks are randomly placed with a minimum separation of 3.0m, and the agent is initialized at least 3.0m away from all objects. Unlike the original CollectHealth, we place only two objects of each type to reduce scene density and avoid trivial collection strategies. The modified reward function is $r _ { t } = 1 0 \times { \bf 1 }$ {all target objects collected}+1×1{target collected}−1×1{incorrect object collected∨ wall collision} − 0.01.

## B.2 Transfer results

Table 3 shows that the representations learned by RAIL are not tied to a single policy or reward function, but instead capture rich environment details that support new behaviors without additional student-side training. In all three environments, training a new policy on top of a frozen representation learned from Task 1 yields nearly identical student performance to training RAIL from scratch on Task 2. This is notable because, after freezing the encoders, training the new policy is performed entirely using teacher observations. Nevertheless, the resulting policies remain directly executable from student observations, which indicate that the learned latent space preserves the shared environment structure required for learning new behaviors while maintaining teacher–student alignment.

Table 3: Student success rates on the new task. Task 2 denotes direct training on the second task from scratch. Task $1  2$ denotes transfer after first learning the representation on Task 1, freezing both encoders, training a new Task 2 policy using only teacher observations, and deploying zero-shot to the student. Matching Task 1 → 2 to Task 2 indicates successful representation reuse. Rates show the mean with upper and lower semideviations across five runs.
<table><tr><td></td><td colspan="2">CornersPlus</td><td colspan="2">TunnelVisionPlus</td><td colspan="2">CollectHealthPlus</td></tr><tr><td>Method</td><td>Task 2</td><td>Task 1→2</td><td>Task 2</td><td>Task 1→2</td><td>Task 2</td><td> $\mathrm { T a s k } 1  2$ </td></tr><tr><td>RAIL (student)</td><td> $1 . 0 _ { - 0 . 0 0 } ^ { + 0 . 0 0 }$ </td><td> $1 . 0 _ { - 0 . 0 0 } ^ { + 0 . 0 0 }$ </td><td> $1 . 0 _ { - 0 . 0 0 } ^ { + 0 . 0 0 }$ </td><td> $1 . 0 _ { - 0 . 0 0 } ^ { + 0 . 0 0 }$ </td><td> $0 . 8 4 _ { - 0 . 0 7 } ^ { + 0 . 0 7 }$ </td><td> $0 . 8 3 _ { - 0 . 0 6 } ^ { + 0 . 0 6 }$ </td></tr><tr><td>Student RL</td><td> $1 . 0 _ { - 0 . 0 0 } ^ { + 0 . 0 0 }$ </td><td></td><td> $1 . 0 _ { - 0 . 0 0 } ^ { + 0 . 0 0 }$ </td><td></td><td> $0 . 8 1 _ { - 0 . 1 1 } ^ { + 0 . 1 1 }$ </td><td></td></tr></table>

## C Implementation details

## C.1 InfoNCE implementation

We use a symmetric InfoNCE objective following the bidirectional formulation of CLIP [19]. Given minibatch embeddings $\{ z _ { T } ^ { i } , z _ { S } ^ { i } \} _ { i = 1 } ^ { \tilde { N } }$ , we first $\ell _ { 2 }$ -normalize both embedding sets and compute similarity logits

$$
M _ { i j } = s \left( z _ { T } ^ { i } \right) ^ { \top } z _ { S } ^ { j } ,\tag{15}
$$

where $s = \exp ( \alpha ) = 1 / \tau$ is a learnable logit scale with unconstrained parameter α. We initialize $s = 1 / 0 . 0 5$ , optimize α jointly with all parameters, and clamp $s \leq 5 0$ for numerical stability.

The contrastive objective averages teacher-to-student and student-to-teacher cross-entropies:

$$
\mathcal { L } _ { \mathrm { I n f o N C E } } = \frac { 1 } { 2 } \Big [ \mathrm { C E } ( M , y ) + \mathrm { C E } ( M ^ { \top } , y ) \Big ] ,\tag{16}
$$

where CE is the cross entropy loss and $y = ( 1 , \ldots , N )$ specifies that sample i is matched to its paired sample i.

Multi-positive variant. In environments with small or discrete observation spaces, multiple samples in a minibatch may correspond to identical teacher or student observations. Treating these duplicates as negatives creates false-negative pairs. We therefore replace the one-hot target with a multi-positive mask $\mathbf { \bar { \mathit { P } } } \in \{ 0 , 1 \} ^ { N \times N }$

$$
P _ { i j } = { \bf 1 } \left[ o _ { T } ^ { i } = o _ { T } ^ { j } \ \vee \ o _ { S } ^ { i } = o _ { S } ^ { j } \right] .\tag{17}
$$

The teacher-to-student term becomes

$$
\mathcal { L } _ { T  S } = - \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \log \sum _ { j = 1 } ^ { N } P _ { i j } \frac { \exp ( M _ { i j } ) } { \sum _ { k = 1 } ^ { N } \exp ( M _ { i k } ) } ,\tag{18}
$$

and the reverse-direction term is

$$
\mathcal { L } _ { S \to T } = - \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \log \sum _ { j = 1 } ^ { N } P _ { j i } \frac { \exp ( M _ { j i } ) } { \sum _ { k = 1 } ^ { N } \exp ( M _ { k i } ) } .\tag{19}
$$

The final objective is

$$
\mathcal { L } _ { \mathrm { M P - I n f o N C E } } = \frac { 1 } { 2 } ( \mathcal { L } _ { T  S } + \mathcal { L } _ { S  T } ) .\tag{20}
$$

We use this variant only in Corners and TunnelVision, where repeated observations are common.

## C.2 Network architectures

All methods use comparable model capacity. RAIL consists of a teacher and a student encoder, a shared encoder, and a policy network with actor–critic heads.

Vector observations. For state-based agents, each observation encoder is a three-layer MLP with hidden width 512 and LeakyReLU activations: $d _ { \mathrm { o b s } }  5 1 2  5 1 2  5 1 2$

Pixel observations. For pixel observations, we use a four-layer CNN with 3×3 kernels, 32 channels per layer, and strides (2, 2, 1, 1), followed by a linear layer producing a 512-dimensional feature vector.

Shared encoder. Teacher and student features are concatenated with a learned agent-identity embedding of dimension 4, then mapped by a shared MLP: $5 1 6  5 1 2  5 1 2  d _ { z }$ , followed by $\ell _ { 2 }$ -normalization for contrastive learning.

Policy network. We use the standard PPO policy architecture: two shared hidden layers of size 64 with ReLU activations, followed by separate linear actor and critic output heads.

## C.3 Training hyperparameters

We use PPO for all RL-based methods and Adam for both policy and encoder optimization. Table 4 lists the environment-specific training budgets and hyperparameters. Unless stated otherwise, all remaining hyperparameters use the default values from PyTorch [16] and Stable-Baselines3 [20].

For QuarterCheetah, we found that direct Student RL and the privileged teacher training used for BC became unstable with the standard PPO learning rate. We therefore reduced the PPO learning rate to $2 \times 1 0 ^ { - 5 }$ for these settings to prevent training collapse.

All experiments were conducted on a machine with an Intel Core i7-13700KF CPU (16 cores, 24 threads, up to 5.4 GHz), an NVIDIA RTX 4070 GPU with 12 GB VRAM, and 32 GB RAM. Depending on the environment and method, individual runs required approximately 1–7 hours. In QuarterCheetah, RAIL required approximately four hours, compared with seven hours for standard PPO from pixels under the same environment-interaction budget.

The BC datasets are collected from rollouts of the trained state-based teacher and contain 4,096 transitions for Corners and TunnelVision, 20,000 for CollectHealth, and 80,000 for QuarterCheetah. DAgger uses replay buffers of the same respective sizes, populated from student rollouts.

Table 4: Training hyperparameters for each environment.
<table><tr><td></td><td>Corners</td><td>TunnelVision</td><td>CollectHealth</td><td>QuarterCheetah</td></tr><tr><td>training steps</td><td>1.0M</td><td>1.5M</td><td>0.4M</td><td>2.5M</td></tr><tr><td>parallel environments</td><td>12</td><td>12</td><td>12</td><td>4</td></tr><tr><td>PPO</td><td></td><td></td><td></td><td></td></tr><tr><td>optimizer</td><td>Adam 3 × 10−4</td><td>Adam</td><td>Adam  $3 \times 1 0 ^ { - 4 }$ </td><td>Adam</td></tr><tr><td>learning rate batch size</td><td>256</td><td> $3 \times 1 0 ^ { - 4 }$  512</td><td></td><td> $3 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>rollout steps (nsteps)</td><td>256</td><td>512</td><td>256 256</td><td>64 512</td></tr><tr><td>epochs</td><td>10</td><td>10</td><td>10</td><td>20</td></tr><tr><td>entropy coefficient</td><td>0.002</td><td>0.001</td><td>0.000</td><td>0.0004</td></tr><tr><td>clip range</td><td>0.1</td><td>0.1</td><td>0.1</td><td>0.1</td></tr><tr><td>discount factor (γ)</td><td>0.99</td><td>0.99</td><td>0.99</td><td>0.98</td></tr><tr><td>RAIL</td><td></td><td></td><td></td><td></td></tr><tr><td>encoder optimizer</td><td>Adam</td><td>Adam</td><td>Adam</td><td></td></tr><tr><td>encoder learning rate</td><td>10−3</td><td> $1 0 ^ { - 3 }$ </td><td>10⁻3</td><td>Adam</td></tr><tr><td>embedding dimension (dz)</td><td>64</td><td>128</td><td>256</td><td>10⁻3</td></tr><tr><td>contrastive batch size</td><td>512</td><td>512</td><td></td><td>256</td></tr><tr><td>contrastive iterations (K)</td><td>1</td><td>2</td><td>256</td><td>512</td></tr><tr><td>λcon</td><td>1.0</td><td></td><td>12</td><td>8</td></tr><tr><td></td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td></tr><tr><td> $\lambda _ { \mathrm { a l i g n } }$ </td><td></td><td>1.0</td><td>1.0</td><td>1.0</td></tr><tr><td>λstab</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td></tr><tr><td>initial InfoNCE scale</td><td>20</td><td>20</td><td>20</td><td>20</td></tr><tr><td>maximum InfoNCE scale</td><td>50</td><td>50</td><td>50</td><td>50</td></tr></table>

## D Additional experiments

## D.1 Privileged-information probes

We use Corners to test whether teacher embeddings retain shared information while suppressing information private to the teacher. The target corner is sampled uniformly from four possibilities and observed only by the teacher, whereas the agent’s coordinates are available to both agents. We train RAIL from scratch with ten random seeds and freeze the learned encoders. For each model, we collect probe data from 100 independently initialized episodes with independently sampled target corners.

We train separate two-layer MLP probes on the frozen teacher embeddings to predict the target corner and the agent’s coordinates. We evaluate target-corner prediction only at the initial state, where the agent always starts at the grid center. This prevents the probe from exploiting correlations between the agent’s position and target corner that arise during a rollout. The coordinate probe uses embedding from the full rollout buffer to assess whether the representation preserves shared spatial information.

Across ten seeds, target-corner prediction achieves $2 5 . 4 \pm 1$ .4% accuracy, close to the 25% chance level, while coordinate prediction achieves $1 0 0 . 0 \pm 0 . 0 \%$ . These results provide evidence that RAIL preserves shared information while suppressing information private to the teacher.

## D.2 Contrastive batch size

The number of negatives in our contrastive objective (Eq. (11)) depends on the batch size used for representation learning. We experiment with batch sizes of 128, 256, 512, and 1024, keeping the remaining training setup fixed and using five random seeds per setting. Table 5 reports student performance.

RAIL achieves high student performance throughout this range. In Corners, batch sizes of 512 and 1024 yield perfect success in all five runs, whereas smaller batches have lower mean success and greater variability. TunnelVision and CollectHealth approach saturation at moderate batch sizes, and QuarterCheetah returns remain similar across settings. Thus, larger batches can improve reliability, while RAIL maintains strong performance even with smaller batches.

Table 5: Contrastive batch-size ablation. Student success rates show the mean with upper and lower semideviations over five seeds; QuarterCheetah episodic returns show mean ± standard deviation.
<table><tr><td>Batch size</td><td>Corners</td><td>TunnelVision</td><td>CollectHealth</td><td>QuarterCheetah</td></tr><tr><td>128</td><td> $0 . 9 1 _ { - 0 . 1 6 } ^ { + 0 . 0 8 }$ </td><td> $0 . 9 9 _ { - 0 . 0 1 } ^ { + 0 . 0 1 }$ </td><td> $0 . 9 4 _ { - 0 . 0 4 } ^ { + 0 . 0 3 }$ </td><td> $2 4 9 1 . 3 3 \pm 1 5 4 . 0 6$ </td></tr><tr><td>256</td><td> $0 . 8 9 _ { - 0 . 1 0 } ^ { + 0 . 0 8 }$ </td><td> $1 . 0 0 _ { - 0 . 0 0 } ^ { + 0 . 0 0 }$ </td><td> $0 . 9 8 _ { - 0 . 0 1 } ^ { + 0 . 0 0 }$ </td><td> $2 5 2 3 . 3 3 \pm 8 4 . 6 3$ </td></tr><tr><td>512</td><td> $1 . 0 0 _ { - 0 . 0 0 } ^ { + 0 . 0 0 }$ </td><td> $1 . 0 0 _ { - 0 . 0 0 } ^ { + 0 . 0 0 }$ </td><td> $0 . 9 9 _ { - 0 . 0 1 } ^ { + 0 . 0 0 }$ </td><td> $2 5 1 0 . 2 4 \pm 7 2 . 0 5$ </td></tr><tr><td>1024</td><td> $1 . 0 0 _ { - 0 . 0 0 } ^ { + 0 . 0 0 }$ </td><td> $1 . 0 0 _ { - 0 . 0 0 } ^ { + 0 . 0 0 }$ </td><td> $0 . 9 9 _ { - 0 . 0 1 } ^ { + 0 . 0 1 }$ </td><td> $2 5 4 3 . 3 3 \pm 4 5 . 7 6$ </td></tr></table>

## D.3 Baseline experiments under matched observations

To test whether the failure of the baselines arises from information mismatch, we repeat BC, DAgger, COSIL, and SITT with the same training protocols but matched information content. Students receive the teacher’s observations in Corners, TunnelVision, and CollectHealth. In QuarterCheetah, students receive stacked, unmasked RGB images while teachers retain state observations.

All four methods match or approach teacher performance across all environments (Table 6). This contrast with Table 2 supports that the information mismatch is a principal contributor to the failure of baselines in our environments.

Table 6: Baseline performance under matched observations. Asterisks denote environments with modified student observations. Teacher (T) and student (S) success rates show the mean with upper and lower semideviations over five seeds; QuarterCheetah returns show mean ± standard deviation. The imitation gap is $\Delta = T - S .$ Higher S and lower ∆ are better.
<table><tr><td></td><td colspan="3">Corners*</td><td colspan="3">TunnelVision*</td><td colspan="3">CollectHealth*</td><td colspan="3">QuarterCheetah*</td></tr><tr><td>Method</td><td>T</td><td>S↑</td><td>∆↓</td><td>T</td><td>S↑</td><td>∆↓</td><td>T</td><td>S↑</td><td>∆↓</td><td>T</td><td>S↑</td><td>∆↓</td></tr><tr><td>BC</td><td> $1 . 0 0 _ { - 0 . 0 0 } ^ { + 0 . 0 0 }$ </td><td>1.00+0.00 -0.00</td><td></td><td>0.00 1.00+0.00 -0.00</td><td> $1 . 0 0 _ { - 0 . 0 0 } ^ { + 0 . 0 0 }$ </td><td></td><td>0.00 1.00+0.00 -0.00</td><td>1.00+0.00 -0.00</td><td>0.00</td><td> $2 1 8 6 . 8 3 \pm 8 2 . 4 7$ </td><td>2113.43 ± 57.39</td><td>73.40</td></tr><tr><td></td><td>DAgger 1.00+0.00 -0.00</td><td>1.00+0.00 -0.00</td><td>0.00</td><td> $1 . 0 0 _ { - 0 . 0 0 } ^ { + 0 . 0 0 }$ </td><td> $1 . 0 0 _ { - 0 . 0 0 } ^ { + 0 . 0 0 }$ </td><td>0.00</td><td> $1 . 0 0 _ { - 0 . 0 0 } ^ { + 0 . 0 0 }$ </td><td>1.00+0.00 -0.00</td><td>0.00</td><td> $2 1 8 6 . 8 3 \pm 8 2 . 4 7$ </td><td> $2 0 9 9 . 6 7 \pm 6 2 . 2 7$ </td><td>87.16</td></tr><tr><td>COSIL 1.00</td><td>+0.00 -0.00</td><td>1.00+0.00 -0.00</td><td></td><td>0.00 1.00+0.00 -0.00</td><td>1.00+0.00 -0.00</td><td></td><td>0.00 1.00+0.00 -0.00</td><td>1.00 +0.00 -0.00</td><td>0.00</td><td> $2 1 8 6 . 8 3 \pm 8 2 . 4 7$ </td><td> $2 0 9 0 . 5 2 \pm 6 8 . 5 0$ </td><td>96.31</td></tr><tr><td>SITT</td><td> $1 . 0 0 _ { - 0 . 0 0 } ^ { + 0 . 0 0 }$ </td><td> $1 . 0 0 _ { - 0 . 0 0 } ^ { + 0 . 0 0 }$ </td><td>0.00</td><td> $1 . 0 0 _ { - 0 . 0 0 } ^ { + 0 . 0 0 }$ </td><td> $1 . 0 0 _ { - 0 . 0 0 } ^ { + 0 . 0 0 }$ </td><td>0.00</td><td> $1 . 0 0 _ { - 0 . 0 0 } ^ { + 0 . 0 0 }$ </td><td> $1 . 0 0 _ { - 0 . 0 0 } ^ { + 0 . 0 0 }$ </td><td>0.00</td><td> $1 8 6 5 . 4 2 \pm 8 6 . 7 2$ </td><td> $1 7 2 0 . 4 3 \pm 7 4 . 4 5$ </td><td>144.99</td></tr></table>