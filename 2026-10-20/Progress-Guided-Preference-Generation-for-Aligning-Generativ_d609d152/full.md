# Progress-Guided Preference Generation for Aligning Generative Control Policies

Tung M. Luu<sup>1</sup> Tri Ton<sup>1</sup> Duc M. Nguyen<sup>2,3</sup> Xuan Thanh Nguyen<sup>1</sup> Nguyen Gia Binh<sup>2,3</sup> Quang Tan Nguyen<sup>3</sup> Vu Duong<sup>2</sup> Daniel Sonntag<sup>6,7</sup> Duy M. H. Nguyen<sup>4,5,6</sup> Vien Anh Ngo<sup>2,3</sup> Chang D. Yoo<sup>1</sup>

<sup>1</sup>KAIST <sup>2</sup>Center for AI Research & CECS, VinUniversity <sup>3</sup>VinRobotics <sup>4</sup>University of Stuttgart <sup>5</sup>Max Planck Research School for Intelligent Systems <sup>6</sup>DFKI <sup>7</sup>University of Oldenburg

Abstract: Generative policies have achieved strong performance in imitation learning tasks, but their success is often constrained by the limited availability of expert demonstrations. Although preference-based fine-tuning can leverage suboptimal rollouts to improve performance, existing approaches heavily rely on expensive human annotations. In this work, we introduce Progress-Guided Preference Generation (PGPG), an unsupervised preference alignment framework that eliminates the need for human-in-the-loop supervision. Our method leverages ordered coverage alignment to measure sequential progress relative to a small set of expert demonstrations, enabling the automatic extraction of pseudo-preference labels from autonomously collected rollouts. Furthermore, we introduce key enhancements to improve the reliability of these progress-guided preferences, enabling effective fine-tuning of pretrained generative policies without requiring manually designed reward functions. Extensive experiments across a suite of challenging long-horizon manipulation tasks demonstrate that PGPG outperforms standard supervised fine-tuning baselines. Moreover, our conceptually simple framework is broadly applicable across multiple classes of generative policies, demonstrating strong versatility when integrated with both diffusion and flowmatching policies. Project site is available at: https://pgpg2026.github.io.

Keywords: Preference-based Reinforcement Learning, Generative Policy Finetuning, Robotic Manipulation

## 1 Introduction

Learning robotic policies that generalize beyond curated demonstrations remains a central challenge in reinforcement learning and imitation learning [1, 2, 3]. Behavior cloning can produce strong initial policies from expert data, but it is often brittle under distribution shift and highly sensitive to data quality. This issue is particularly pronounced in real-world settings, where expert trajectories are scarce, training coverage is limited, and the available demonstrations may themselves be suboptimal [4, 5]. Reinforcement learning, by contrast, can improve robustness through trial-and-error interaction. However, it typically requires carefully engineered reward functions, extensive environment interactions, and challenging exploration processes, all of which may be inefficient or unsafe in real-world control scenarios [6, 1, 7]. Recently, preference-based learning [8] has emerged as a promising middle ground between imitation learning and reinforcement learning [9, 2]. Instead of relying solely on expert demonstrations or manually engineered reward functions, preference-based methods replace explicit reward design with relative comparisons between trajectory segments. This allows policies to learn from simple preference queries rather than handcrafted scalar rewards [1, 10, 11, 2]. This paradigm is particularly attractive for robotics, where designing reliable reward functions is difficult, and where improving the performance and robustness of pretrained policies while minimizing costly environment interaction is highly desirable [9, 12, 13, 14, 15].

A major obstacle to scaling preference-based learning is its heavy reliance on human feedback, which is inherently costly and labor-intensive [16, 2]. This bottleneck is particularly restrictive for modern generative control policies, such as diffusion [5] and flow-matching models [17]. While these architectures are highly expressive and excel at capturing multimodal action distributions, they currently lack scalable alignment mechanisms when labeled preferences are scarce [14]. Because existing alignment methods typically assume access to abundant, high-quality human annotations, their real-world applicability remains severely limited [9, 18, 14]. These limitations highlight an open question: How can we automatically extract meaningful preference signalsfrom autonomously collected rollouts to effectivelyfine-tune generative policies without human-in-the-loop supervision?

To this end, we propose Progress-Guided Preference Generation (PGPG), an unsupervised framework for fine-tuning generative control policies. The key idea is to use ordered coverage alignment to measure the continuous progress of collected rollouts relative to a small set of expert demonstrations, and then convert this progress into preference labels for segment pairs. Importantly, the resulting signal does not merely evaluate whether a rollout reaches the final goal, but also whether it exhibits ordered progress that is structurally consistent with expert behavior. To make this signal suitable for preference ranking, we transform cumulative progress into marginal rewards, avoiding temporal double-counting and enabling meaningful comparisons between trajectory segments. We further aggregate progress across multiple demonstrations, which is particularly important in robotic manipulation tasks where multiple valid strategies may exist and a single demonstration should not overly constrain learning. In this way, PGPG transforms limited demonstration coverage into a scalable source of supervision for preference alignment. Extensive experiments across a suite of challenging manipulation tasks from Robomimic [4] demonstrate that PGPG can effectively fine-tune generative policies using only a small set of expert demonstrations together with a large collection of autonomously collected rollouts. We further demonstrate the versatility of our framework by seamlessly integrating it with multiple classes of generative policies, including Diffusion Policy [5] and Rectified Flow Policy [19].

## 2 Related Work

Preference-based Reinforcement Learning. Extensive research has explored aligning agent behavior through pairwise trajectory ranking [1, 20, 21, 10, 11, 13, 12]. The traditional paradigm in preference-based reinforcement learning (PbRL) typically operates in two stages: first fitting a proxy reward model from preference data, and subsequently using the learned reward to guide downstream policy optimization [1, 11, 22, 12, 23]. While this paradigm has demonstrated strong performance across domains ranging from Atari [1, 20] to large language models [24], the intermediate rewardmodeling stage introduces additional optimization complexity. More recently, methods such as Direct Preference Optimization (DPO) [25] and Contrastive Preference Learning (CPL) [2] have eliminated the need for an explicit reward model. Instead, these approaches directly fine-tune policies to satisfy preference constraints through a simplified classification objective. Owing to their simplicity and efficiency, these methods have been adopted for fine-tuning generative control policies [9, 14, 26].

However, applying PbRL to robotic control remains heavily bottlenecked by the substantial human effort required to label large amounts of interaction data [16, 27, 28, 29, 30, 31]. To eliminate this human-in-the-loop burden, our framework automatically extracts preference labels from a limited set of offline expert demonstrations. Furthermore, while recent automated labeling pipelines rely on large language models to generate Python reward functions [9], PGPG derives preference labels directly from expert demonstrations, naturally grounding the learning signal in the visual and kinematic structure of the task.

Generative Control and Preference Alignment. Recent advances in generative modeling, particularly diffusion and flow-matching frameworks [32, 33, 34, 35], have demonstrated a remarkable capability to capture complex, multi-modal distributions. This high expressiveness makes them uniquely suited for decision-making and robotic control, triggering broad adoption across offline RL [36, 37], online RL [38, 39, 40], and manipulation tasks [5, 41, 42]. By framing policies as generative processes, these approaches model multi-modal action trajectories more robustly than conventional Gaussian or VAE-based baselines [7, 43]. Early foundational works like DQL [7] integrated diffusion processes directly into Q-learning frameworks. Subsequent iterations focused on stabilizing optimization and improving evaluation. For instance, SfBC [3] decoupled behavior cloning from explicit action selection, while CEP [44] introduced generalized energy-guided sampling. To address the computational overhead of iterative inference denoising, recent lightweight formulations such as FQL [45] and OFQL [46] drastically minimize sampling complexity while preserving competitive policy performance.

Concurrently, generative control models have emerged as competitive backbones for preferencebased alignment. While early methods like AlignDiff [47] and FTB [48] utilize complex, two-stage diffusion planning loops, recent frameworks like FKPD [26] and UAPA [14] pivot toward highly efficient, one-step direct policy optimization. While PGPG operates within this efficient one-step alignment paradigm, it bypasses the core bottleneck of existing literature. Rather than relying on costly, large-scale human annotations, PGPG synthesizes reliable preference labels directly from autonomously collected rollouts by anchoring progress to a small expert dataset.

## 3 Preliminaries

Preference-based Reinforcement Learning. Reinforcement learning (RL) algorithms [49] typically consider a Markov decision process (MDP), formally defined as a tuple $\mathcal { M } = ( \mathcal { S } , \mathcal { A } , \mathcal { T } , r , \gamma , p _ { 0 } )$ where $s$ and A denote the state and action spaces, $\mathcal { T } ( \mathbf { s } ^ { \prime } | \mathbf { s } , \mathbf { a } )$ and $r ( \mathbf { s } , \mathbf { a } )$ represent the transition dynamics and reward function, $\gamma \in [ 0 , 1 )$ is the discount factor, and $p _ { 0 }$ denotes the initial state distribution. In our setting, the agent additionally receives a visual observation $\mathbf { o } \in \mathcal { O }$ , represented as an image rendered from the underlying state s. In many real-world applications, the reward function is not directly available, resulting in a reward-free MDP M \ r. Preference-based RL [1] addresses this challenge by learning a reward function $r _ { \phi }$ from preference data, typically provided by human annotators. Given an unlabeled dataset of trajectory segments, annotators compare sampled pairs and indicate which segment they prefer. These pairwise comparisons form a preference dataset $\mathcal { D } _ { \mathrm { p r e f } } = \{ ( \sigma ^ { w } , \sigma ^ { l } ) \}$ , where each trajectory segment of length L is defined as $\sigma = ( \mathbf { s } _ { 1 } , \mathbf { a } _ { 1 } , \dots , \mathbf { s } _ { L } , \mathbf { a } _ { L } )$ and $\sigma ^ { w }$ is preferred over $\sigma ^ { l }$ . Recent methods [1, 20] typically employ the Bradley-Terry model [50] to learn the reward function $r _ { \phi }$ from these preference comparisons.

Improving Generative Policy via Preference Alignment. In this work, we consider a family of generative control policies, including diffusion policies [5] and flow-matching policies [17]. Let $\mathcal { D } _ { \mathrm { r o l l o u t } } = \{ \tau \}$ denote the dataset of additional rollouts collected by a pretrained policy interacting with the environment, where each trajectory is defined as $\tau = ( \mathbf { s } _ { 1 } , \mathbf { a } _ { 1 } , \dots , \mathbf { s } _ { H } , \mathbf { a } _ { H } )$ . Given a small set of expert demonstrations $\mathcal { D } _ { \mathrm { d e m o } }$ and a large collection of autonomously collected rollouts $\mathcal { D } _ { \mathrm { r o l l o u t } }$ our goal is to effectively fine-tune the pretrained policy from both data sources.

Because explicit environmental rewards are unavailable, directly applying RL algorithms such as PPO [51] is challenging. We therefore formulate the fine-tuning stage as a preference alignment problem, leveraging the limited set of expert demonstrations as a weak supervisory signal to extract preferences from the uncurated rollouts. We follow recent work [9, 26, 14] that adopts contrastive preference learning (CPL) [2] to align generative policies with human preferences. Concretely, for a pair of preferred and rejected trajectory segments $( \sigma ^ { w } , \sigma ^ { l } )$ , the CPL objective is:

$$
\mathcal { L } _ { \mathrm { C P L } } ^ { ( \sigma ^ { w } , \sigma ^ { l } ) } ( \theta ) = - \log \sigma \left( \alpha \cdot \lbrack \sum _ { t = 1 } ^ { L } \gamma ^ { t } \log \frac { \pi _ { \theta } ( \mathbf { a } _ { t } ^ { w } | \mathbf { s } _ { t } ^ { w } ) } { \pi _ { \mathrm { r e f } } ( \mathbf { a } _ { t } ^ { w } | \mathbf { s } _ { t } ^ { w } ) } - \sum _ { t = 1 } ^ { L } \gamma ^ { t } \log \frac { \pi _ { \theta } ( \mathbf { a } _ { t } ^ { l } | \mathbf { s } _ { t } ^ { l } ) } { \pi _ { \mathrm { r e f } } ( \mathbf { a } _ { t } ^ { l } | \mathbf { s } _ { t } ^ { l } ) } \rbrack \right) ,\tag{1}
$$

where $\sigma ( \cdot )$ is the Sigmoid function, $\sigma ^ { w } = ( \mathbf { s } _ { t } ^ { w } , \mathbf { a } _ { t } ^ { w } ) _ { t = 1 } ^ { L }$ and $\boldsymbol { \sigma } ^ { l } = ( \mathbf { s } _ { t } ^ { l } , \mathbf { a } _ { t } ^ { l } ) _ { t = 1 } ^ { L }$ denote the preferred and rejected trajectory segments, respectively, $\pi _ { \mathrm { r e f } }$ denotes the reference policy, typically instantiated as the pretrained policy, and $\alpha > 0$ controls the deviation from the reference policy. Detailed learning objectives for diffusion and flow-matching policies are provided in the Appendix.

![](images/4850e2549fce2953e75e64c64c50eab0fb38a803acc5768c5adf57e59c401f15.jpg)  
Figure 1: Overview of PGPG. Autonomously collected rollouts $( \mathcal { D } _ { \mathrm { r o l l o u t } } )$ are scored against expert references $\left( \mathcal { D } _ { \mathrm { d e m o } } \right)$ using an alignment pipeline consisting of Ordered Coverage Alignment, Marginal Progress Scoring, and Multi-Demonstration Aggregation. The aggregated segment scores are then used to derive the preference labels. The resulting dataset $\mathcal { D } _ { \mathrm { p r e f } }$ provides structured, dense pseudopreferences to stably fine-tune generative policies $\pi _ { \boldsymbol { \theta } } ( a | \boldsymbol { s } )$ without manual reward engineering.

Sequence Matching. Sequence-matching problems [52] have been widely explored in inverse RL to align a learner’s trajectories with expert demonstrations. While many approaches rely on optimal transport [53, 54, 55] to match expert trajectory distributions, these methods often struggle to preserve temporal consistency and are prone to goal abandonment in long-horizon robotic tasks [56]. This limitation motivates the need for a structurally ordered metric. In this work, we adopt ordered coverage alignment (ORCA) [56] to quantify the continuous progress of collected rollouts relative to expert demonstrations. This measure is particularly well-suited for our framework because it inherently handles temporal misalignments, such as variations in execution speed between the learner and the expert. Concretely, given a learner trajectory τ of length H and an expert demonstration τ˜ of length H<sup>˜</sup> , the ORCA step-wise progress at timestep t in the learner’s trajectory is computed as:

$$
\mathcal { R } ( t , \tau , \tilde { \tau } ) = C _ { t , \tilde { H } - 1 } P _ { t , \tilde { H } } ,\tag{2}
$$

where $P _ { t , j } = \exp ( - \lambda \| \mathbf { o } _ { t } - \tilde { \mathbf { o } } _ { j } \| )$ denotes the probability that the learner occupies the j-th demonstration subgoal at timestep t, and λ is a temperature hyperparameter. The term $C _ { t , j }$ represents the recur sive historical coverage, computed via dynamic programming as $C _ { t , j } = \operatorname* { m a x } \{ C _ { t - 1 , j } , C _ { t , j - 1 } P _ { t , j } \}$ This formulation tracks the probability that the learner has successfully completed all preceding subgoals up to j in the correct order. Intuitively, Eq. (2) assigns a high score only when the agent reaches and remains at the final target without skipping intermediate subgoals. Additionally, the ordered coverage is non-decreasing over time: once the learner has covered a subgoal at a given timestep, it remains at least equally successful at future timesteps. In this work, we leverage this continuous progress measure to automatically derive preference labels for segment pairs, enabling preference learning without a human-in-the-loop.

## 4 Method

While the preference learning objective in Eq. (1) provides a powerful mechanism for aligning generative policies, existing approaches typically rely on human-provided preference labels. This manual annotation process is expensive, difficult to scale, and introduces a major bottleneck for robotic control tasks. To eliminate this dependence on human annotation, we propose Progress-Guided Preference Generation (PGPG), a framework that automatically generates progress-guided preferences to fine-tune policies toward expert behavior, as shown in Figure 1.

## 4.1 Progress-Guided Preference Generation

Marginal Progress Scoring. Let $\mathcal { D } = \mathcal { D } _ { \mathrm { d e m o } } \cup \mathcal { D } _ { \mathrm { r o l l o u t } }$ denote the combined dataset used to finetune the pretrained policy. For each trajectory $\tau \sim \mathcal { D }$ , we first compute the progress score at every timestep t using the ORCA metric anchored against a reference demonstration $\tilde { \tau } \sim \mathcal { D } _ { \mathrm { d e m o } }$ . This yields a progress-labeled trajectory $\hat { \tau } = ( \mathbf { s } _ { 1 } , \mathbf { a } _ { 1 } , \hat { r } _ { 1 } , \hdots , \mathbf { s } _ { H } , \mathbf { a } _ { H } , \hat { r } _ { H } )$ , where $\hat { r } _ { t } = \mathcal { R } ( t , \tau , \tilde { \tau } )$ , resulting in a scored dataset $\hat { \mathcal { D } } = \left\{ \hat { \tau } \right\}$ . To construct the preference dataset, a naive approach would sample two trajectory segments $\sigma _ { 1 }$ and $\sigma _ { 2 }$ from $\begin{array} { r } { \hat { \mathcal { D } } , } \end{array}$ calculate the sum of their step-wise scores $\textstyle \sum _ { t \in \sigma _ { 1 } } { \hat { r } } _ { t }$ and $\textstyle \sum _ { t \in \sigma _ { 2 } } { \hat { r } } _ { t }$ , and label the segment with the higher sum as preferred. However, applying this naive summation directly to the raw ORCA outputs fundamentally distorts the preference ranking. Because the dynamic programming formulation of ORCA guarantees that the progress score is monotonically non-decreasing over time, $\hat { r } _ { t }$ functions as a measure of cumulative progress rather than an instantaneous, per-step reward. Directly summing these cumulative scores across a trajectory segment causes temporal double-counting. This obscures the true advancement achieved within the segment and can lead to inaccurate preference rankings between segments.

To address this issue, we propose converting the progress score into an instantaneous reward signal. Concretely, we define the per-step reward $r _ { t }$ as the marginal difference of the ORCA score:

$$
r _ { t } = \hat { r } _ { t } - \hat { r } _ { t - 1 } , \quad \forall t \in \{ 2 , \ldots , H \} ,\tag{3}
$$

with the boundary condition $r _ { 1 } = \hat { r } _ { 1 }$ . This formulation captures the net sequential progress achieved at timestep t. Consequently, to evaluate any arbitrary trajectory segment $\sigma = ( \mathbf { s } _ { t + 1 } , \ldots , \mathbf { s } _ { t + L } )$ , we compute the segment score $R ( \sigma )$ by summing these marginal rewards. Crucially, because this forms a telescoping sum, the segment score reduces to the difference between the ORCA score at the final timestep of the segment and the score immediately before the segment begins:

$$
R ( \sigma ) = \sum _ { k = t + 1 } ^ { t + L } r _ { k } = \hat { r } _ { t + L } - \hat { r } _ { t } .\tag{4}
$$

This simplification enables efficient segment evaluation. Subsequently, to derive a preference label for a pair of segments $( \sigma _ { 1 } , \sigma _ { 2 } )$ , we compare their segment scores and assign the preferred segment $\sigma ^ { w }$ and rejected segment $\sigma ^ { l }$ as follows:

$$
( \sigma ^ { w } , \sigma ^ { l } ) = \left\{ \begin{array} { l l } { ( \sigma _ { 1 } , \sigma _ { 2 } ) } & { \mathrm { i f } R ( \sigma _ { 1 } ) > R ( \sigma _ { 2 } ) } \\ { ( \sigma _ { 2 } , \sigma _ { 1 } ) } & { \mathrm { i f } R ( \sigma _ { 2 } ) > R ( \sigma _ { 1 } ) . } \end{array} \right.\tag{5}
$$

To ensure a clear learning signal and avoid updating the policy on ambiguous preferences, we filter out pairs where the score difference $| R ( \sigma _ { 1 } ) - R ( \sigma _ { 2 } ) |$ falls below a discrimination threshold $\epsilon _ { \sigma }$

While step-wise progress can also be estimated using recent progress reward models [57, 58, 59, 60], these approaches typically rely on large-scale vision-language architectures $( e . g .$ , the 8B-parameter RoboReward [59]), resulting in substantial computational overhead. Furthermore, their zero-shot evaluations may suffer from hallucinations or poor generalization when the downstream task distribution differs from the pretraining data [58]. In contrast, our approach employs a lightweight dynamic programming-based metric that grounds the progress signal directly in domain-specific demonstrations rather than pretrained VLM priors. As a result, PGPG can generate preference labels significantly more efficiently while ensuring that the resulting supervision remains tightly aligned with the underlying task dynamics.

Multi-Demonstration Aggregation. In robotic control tasks, there are often multiple valid strategies for achieving the same goal. Relying on a single demonstration to generate preference labels may inadvertently penalize a rollout that successfully completes the task using an alternative yet equally valid behavioral mode. To capture this multimodality, we evaluate the segment score using a subset of M demonstrations. Specifically, we compute the segment score $R ( \sigma , \tilde { \tau } _ { m } )$ independently for each demonstration $\tilde { \tau } _ { m } .$ , and then extract the final score $R ( \sigma )$ via max-pooling across this subset:

$$
R ( \sigma ) = \operatorname* { m a x } _ { m \in \{ 1 \ldots M \} } R ( \sigma , \tilde { \tau } _ { m } ) .\tag{6}
$$

Algorithm 1 Pseudo-code for Progress-Guided Preference Generation (PGPG)   
1: Input: Dataset of demonstrations $\mathcal { D } _ { \mathrm { d e m o } } ,$ dataset of rollouts $\mathcal { D } _ { \mathrm { r o l l o u t } } ,$ number of expert references $M ,$ number   
of queries N, discrimination threshold $\epsilon _ { \sigma }$ , pretrained policy $\pi _ { \theta } .$   
2: Initialize: Select reference set $\{ \tilde { \tau } _ { m } \} _ { m = 1 } ^ { M } \stackrel {  } { \subset } \mathcal { D } _ { \mathrm { d e m o } } , \hat { \mathcal { D } } = \dot { \mathcal { D } } _ { \mathrm { d e m o } } \cup \mathcal { D } _ { \mathrm { r o l l o u t } } ,$ , preference dataset $\begin{array} { r } { \mathcal { D } _ { \mathrm { p r e f } }  \emptyset . } \end{array}$   
3: Construct progress-labeled dataset $\hat { \mathcal { D } }$ by calculating ORCA score for trajectories in $\mathcal { D }$ using Eq. (2).   
4: for $i = 1$ to N do   
5: Sample trajectory segments $( \sigma _ { 1 } , \sigma _ { 2 } ) \sim \hat { \mathcal { D } } .$   
6: Compute segment scores $R ( \dot { \sigma } _ { 1 } )$ and $R ( \sigma _ { 2 } )$ using Eq. (4) and Eq. (6).   
7: $\mathbf { i f } \ | \hat { R ( \sigma _ { 1 } ) } - \mathbf { \bar { \mu } } R ( \sigma _ { 2 } ) | > \epsilon _ { \sigma }$ then   
8: Assign preferred segment $\sigma ^ { w }$ and rejected segment $\sigma ^ { l }$ based on Eq. (5).   
9: $\mathcal { D } _ { \mathrm { p r e f } }  \dot { \mathcal { D } } _ { \mathrm { p r e f } } \cup \{ ( \smile ^ { w } , \sigma ^ { l } ) \}$   
10: end if   
11: end for   
12: Fine-tune policy $\pi \theta$ with dataset $\mathcal { D } _ { \mathrm { p r e f } }$ utilizing the CPL objective in Eq. (1).

Once the aggregated scores for two segments $\sigma _ { 1 }$ and $\sigma _ { 2 }$ are computed, we apply the discrimination threshold $\epsilon _ { \sigma }$ and the comparison in Eq. (5) to construct the final preference pair $( \sigma ^ { w } , \sigma ^ { l } )$ . Crucially, our approach applies this max-pooling operation at the segment level, rather than over the sum of the entire trajectory as is commonly done in prior sequence-matching methods [55, 56]. This segment-wise aggregation provides a key behavioral advantage by enabling the framework to properly reward composite trajectories. For instance, a learner may follow one expert’s strategy during an initial segment and transition seamlessly to an alternative strategy in a later segment, receiving reliable progress signals without being forced to merely imitate a single global demonstration.

## 4.2 Implementation Details

Since ORCA evaluates high-dimensional visual observations within trajectories, we first project raw image frames into a semantic latent space using a pretrained visual encoder before computing pairwise distances. Specifically, we employ R3M [61] as the visual backbone, which in our preliminary experiments provided more robust semantic alignment than the ResNet50 [62] used in the original ORCA implementation [56]. By applying PGPG to score these trajectory segments, we construct a preference dataset $\mathcal { D } _ { \mathrm { p r e f } }$ . This dataset is subsequently utilized to fine-tune the pre-trained policy via the CPL objective defined in Eq. (1). The overall procedure is summarized in Algorithm 1.

## 5 Experiments

In this section, we conduct comprehensive experiments across a suite of complex manipulation tasks to investigate the following questions: (1) Can PGPG synthesize reliable preference labels to guide policy learning? (2) How effectively does our framework improve the performance of pretrained generative policies compared to supervised fine-tuning (SFT) baselines? (3) How do critical hyperparameters influence the overall performance of PGPG?

## 5.1 Experimental Setups

To evaluate the effectiveness of our method, we adopt the Robomimic benchmark [4], which consists of five tabletop manipulation tasks: Lift, Can, Square, Transport, and Tool Hang. These tasks cover diverse manipulation skills and long-horizon behaviors, ranging from simple pick-and-place tasks to precise insertion tasks. To evaluate the generality of our framework, we conduct experiments using both Diffusion Policy [5] and Rectified Flow policies [19]. Following prior work [14], for each task, the reference policy $\pi _ { \mathrm { r e f } }$ is trained to achieve approximately 40% success rate, after which we collect 500 rollout trajectories to construct the rollout dataset. We further retain only 40% of the original demonstration dataset, resulting in approximately 120 demonstrations per task (80 for Tool Hang). We use the mh datasets from Robomimic for all tasks except Tool Hang, which only provides a ph dataset. All methods are trained using low-dimensional state observations. Visual observations are used solely by PGPG for preference generation. We evaluate the performance of the fine-tuned policy after 1000 training epochs. Results are averaged over 50 episodes across three random seeds, as in [63]. We compare PGPG against the following baselines:

![](images/c95a45b5389481e4638824ec422172328ebcd6870e39bd3efdc88f5beefa3994.jpg)  
(a) Diffusion policy

![](images/4514fc8045f2e3527e8eb97d2d440b239013f94583726e53e171ef30d76645bf.jpg)  
(b) Flow-matching policy  
Figure 2: Success rates of methods on five Robomimic manipulation tasks. Error bars denote standard deviation across seeds. DPPO is applied only to the diffusion policy.

• Base: The pretrained generative policy $\pi _ { \mathrm { r e f } }$ (diffusion or flow-matching) prior to any fine-tuning.

• Mixed BC: Standard behavioral cloning that fine-tunes the base policy using the combined dataset of expert demonstrations and all uncurated rollouts.

• Filtered BC [64, 65]: Behavioral cloning fine-tuned using expert demonstrations and only successful rollout trajectories. Crucially, these successful rollouts are filtered using the benchmark’s oracle success criteria, which represents privileged environmental information inaccessible to our method.

$\mathbf { C P L } _ { K L }$ [2]: A contrastive preference learning baseline that follows our exact training setup, but derives preference labels directly from the ground-truth benchmark reward functions [11] rather than our progress metric. For environments with dense rewards (Lift, Can, and Square), we use 1000 preference labels; for the remaining sparse-reward tasks, we use 5000 labels.

• DPPO [39]: A reinforcement learning baseline that fine-tunes the base diffusion policy using a PPO-based variant [39]. For a fair comparison, the pretrained model in this baseline achieves comparable performance to our reference policy, and we restrict it to approximately the same number of additional rollout trajectories used by PGPG.

## 5.2 Main Results

As shown in Figure 2, PGPG achieves the highest average success rate across all tasks, significantly outperforming non-privileged baselines such as Mixed BC and DPPO. In fact, Mixed BC yields only marginal improvements and occasionally degrades base policy performance. Among the baselines that use privileged information, Filtered BC and $\mathrm { C P L } _ { K L }$ , the preference-based fine-tuning of $\mathrm { C P L } _ { K L }$ achieves stronger results. This indicates that selectively ranking trajectory quality provides a more effective learning signal than directly mixing successful rollouts with expert demonstrations.

Notably, on tasks with dense ground-truth rewards, PGPG matches the performance of the rewardguided $\mathrm { C P L } _ { K L }$ baseline (with the sole exception of the Flow Policy on Square). This confirms that our automatically generated preferences effectively replace manually designed reward functions. Furthermore, PGPG consistently outperforms $\mathrm { C P L } _ { K L }$ on sparse-reward tasks (Transport and Tool-Hang). We attribute this advantage to the limitations of sparse rewards, which assign nearly identical supervision to distinct segment pairs, failing to capture subtle differences in progress. In contrast, our progress-guided preferences deliver dense, temporally structured supervision, leading to superior policy fine-tuning.

Table 1: Comparison of fine-tuning performance using preference labels generated by different sequence-matching metrics within our framework. Best results and those within one standard deviation are bolded.
<table><tr><td>Method</td><td>Lift</td><td>Can</td><td>Square</td><td>Transp.</td><td>ToolHang</td></tr><tr><td>Temporal OT</td><td> $2 0 . 7 \pm 1 . 2$ </td><td> $5 8 . 7 \pm 5 . 8 $ </td><td> $6 2 . 7 \pm 3 . 1$ </td><td> $3 6 . 7 \pm 6 . 1$ </td><td> $3 2 . 0 { \scriptstyle \pm 5 . 3 }$ </td></tr><tr><td>Temporal  ${ \mathrm { O T } } + { \mathrm { M P S } }$ </td><td> $8 2 . 0 { \scriptstyle \pm 2 . 0 }$ </td><td> $7 2 . 0 { \scriptstyle \pm 5 . 3 }$ </td><td> $5 1 . 3 { \pm } 1 0 . 1 $ </td><td> $4 3 . 3 { \pm } 5 . 0$ </td><td> $4 1 . 3 { \pm } 8 . 3 $ </td></tr><tr><td>DTW</td><td> $9 0 . 7 \pm 3 . 1$ </td><td> $7 0 . 0 { \scriptstyle \pm 5 . 3 }$ </td><td> $7 2 . 7 { \scriptstyle \pm 1 . 2 }$ </td><td> $4 5 . 3 { \pm } 1 1 . 0 $ </td><td> $3 0 . 0 { \scriptstyle \pm 8 . 7 }$ </td></tr><tr><td> $\mathrm { D T W + M P S }$ </td><td> $8 0 . 0 { \scriptstyle \pm 5 . 3 }$ </td><td> $7 8 . 0 { \pm } 7 . 2$ </td><td> $6 2 . 7 \pm 7 . 0$ </td><td> $3 8 . 7 \pm 2 . 3$ </td><td> $3 1 . 3 { \pm } 7 . 0$ </td></tr><tr><td> $\mathrm { O R C A }$ </td><td> $3 2 . 0 { \scriptstyle \pm 2 . 0 }$ </td><td> $7 0 . 7 { \scriptstyle \pm 5 . 0 }$ </td><td> $4 2 . 0 { \pm } 3 . 5 $ </td><td> $4 7 . 3 { \pm } 4 . 6 $ </td><td> $4 1 . 3 { \pm } 3 . 1 $ </td></tr><tr><td> $\mathrm { O R C A } + \mathrm { M P S } \mathrm { ( O u r s ) }$ </td><td> $\mathbf { 9 6 . 7 \bot } 1 . 2$ </td><td> ${ \bf 8 9 . 3 } \pm 4 . 6$ </td><td> $7 2 . 0 { \scriptstyle \pm 7 . 2 }$ </td><td> ${ \pm 8 . 7 \pm 6 . 1 }$ </td><td> ${ \pm } 6 . 0 { \pm } 8 . 7$ </td></tr></table>

![](images/c26779049fc3eec02dba62c00ecc273ae3bf7ecf8b313927fbdee6ee15df32a7.jpg)  
(a) Varying Demonstrations

![](images/2781c2acdc26f6273d53cff450decab7d7a3a6ac54070165820be1df865616ae.jpg)  
(b) Effect of $\epsilon _ { \sigma }$

![](images/d17cc6a49e59ecc6fb3d402e462619592412aca95c0511fc88548c740a0c73db.jpg)  
(c) Accuracy and Retention Rate  
Figure 3: Ablation study across different design choices.

## 5.3 Ablation Studies

Unless otherwise stated, all ablation studies are conducted with the diffusion policy.

Choice of Alignment Metric. We investigate the effectiveness of different sequence alignment metrics for preference generation by applying our Marginal Progress Scoring (MPS) to Temporal OT [55], DTW [66], and the ORCA baseline [56]. As shown in Table 1, MPS yields large, consistent gains only when combined with ORCA. When applied to other metrics, MPS produces inconsistent results: it generally improves Temporal OT (except on Square) but degrades DTW on three of five tasks. This suggests that the gains arise from the interaction between marginal scoring and ORCA’s ordered-coverage structure, rather than from MPS or ORCA alone. This finding is consistent with ORCA’s analysis [56]: DTW and Temporal OT do not reliably capture full subgoal coverage, resulting in an unreliable ranking signal. In contrast, ORCA’s score is provably monotone in the number of subgoals covered.

Impact of the Number of Demonstrations. We evaluate the data efficiency of PGPG by varying the number of expert demonstrations used for label synthesis (Figure 3a). The results indicate rapid performance improvements as the number of demonstrations increases from 1 to 10 across both the Lift and Square tasks. Notably, the success rate saturates after 10–20 demonstrations, highlighting that our framework can achieve near-optimal policy alignment without the burden of large-scale expert data collection.

Quality of Pseudo-Preference Labels. We evaluate the accuracy and retention rate of our synthe sized preference labels against ground-truth preferences under varying discrimination thresholds $( \epsilon _ { \sigma } )$ As illustrated in Figures 3b and 3c, adjusting this threshold reveals a critical trade-off between label reliability and dataset scale. While larger thresholds improve label accuracy by filtering out ambigu ous segment pairs, they simultaneously reduce the amount of usable training data. Consequently, downstream policy performance peaks at a moderate threshold $( \epsilon _ { \sigma } = 0 . 6 )$ , which effectively balances label quality and data retention. In contrast, excessively large thresholds $( \epsilon _ { \sigma } = 0 . 9 5 )$ severely reduce the available training data, leading to a sharp performance degradation despite near-perfect label accuracy. Further ablations are provided in the Appendix.

## 6 Conclusion

We presented Progress-Guided Preference Generation (PGPG), an unsupervised framework that aligns generative control policies without relying on human annotations or explicit reward engineering. By leveraging ordered coverage alignment, our method autonomously evaluates sequential progress against a small set of expert demonstrations, translating suboptimal rollouts into robust pseudopreference labels. Empowered by key enhancements to ensure label reliability, PGPG significantly improves the performance of pretrained generative policies on challenging, long-horizon manipulation tasks, outperforming standard SFT baselines.

Limitations. First, PGPG relies heavily on the robustness of the pre-trained visual encoder; future work will explore online fine-tuning to support cross-embodiment alignment where temporal mismatches are severe. Second, visual sequence matching remains susceptible to task misspecification under heavy occlusion. We plan to mitigate this by integrating richer modalities, such as depth or tactile sensing. Finally, our Marginal Progress Scoring assumes a largely monotonic progression toward the goal, which may struggle to accurately score long-horizon tasks that require strategic backtracking or non-quasistatic behavior.

## Acknowledgements

This work was supported by Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT) (No.RS-2021-II211381, Development of Causal AI through Video Understanding and Reinforcement Learning, and Its Applications to Real Environments) and the National Research Foundation of Korea (NRF) grant funded by the Korea government(MSIT) (RS-2025-24742969, Intelligent Robotic System using Continual Learning and Multimodal Language Model based Multi Attribute Feedback). Duy M. H. Nguyen and Daniel Sonntag also acknowledge the support of the Federal Ministry of Research, Technology, and Space (BMFTR) under grant number 16IW26002 (AMENABLE), the Lower Saxony Ministry of Science and Culture (MWK) in the Zukunft.niedersachsen program, and the Endowed Chair of AAI at the University of Oldenburg.

## References

[1] P. F. Christiano, J. Leike, T. Brown, M. Martic, S. Legg, and D. Amodei. Deep reinforcement learning from human preferences. NeurIPS, 2017.

[2] J. Hejna, R. Rafailov, H. Sikchi, C. Finn, S. Niekum, W. B. Knox, and D. Sadigh. Contrastive preference learning: learning from human feedback without rl. ICLR, 2024.

[3] H. Chen, C. Lu, C. Ying, H. Su, and J. Zhu. Offline reinforcement learning via high-fidelity generative behavior modeling. arXiv:2209.14548, 2022.

[4] A. Mandlekar, D. Xu, J. Wong, S. Nasiriany, C. Wang, R. Kulkarni, L. Fei-Fei, S. Savarese, Y. Zhu, and R. Martín-Martín. What matters in learning from offline human demonstrations for robot manipulation. arXiv:2108.03298, 2021.

[5] C. Chi, Z. Xu, S. Feng, E. Cousineau, Y. Du, B. Burchfiel, R. Tedrake, and S. Song. Diffusion policy: Visuomotor policy learning via action diffusion. IJRR, 2023.

[6] J. Clark and D. Amodei. Faulty reward functions in the wild. https://openai.com/index/ faulty-reward-functions/, 2016.

[7] Z. Wang, J. J. Hunt, and M. Zhou. Diffusion policies as an expressive policy class for offline reinforcement learning. ICLR, 2023.

[8] S. Casper, X. Davies, C. Shi, T. K. Gilbert, J. Scheurer, J. Rando, R. Freedman, T. Korbak, D. Lindner, P. Freire, et al. Open problems and fundamental limitations of reinforcement learning from human feedback. TMLR, 2023.

[9] Z. Zhang, K. Zheng, Z. Chen, J. Jang, Y. Li, S. Han, C. Wang, M. Ding, D. Fox, and H. Yao. Grape: Generalizing robot policy via preference alignment. arXiv:2411.19309, 2024.

[10] K. Lee, L. Smith, and P. Abbeel. Pebble: Feedback-efficient interactive reinforcement learning via relabeling experience and unsupervised pre-training. In ICML, 2021.

[11] K. Lee, L. Smith, A. Dragan, and P. Abbeel. B-pref: Benchmarking preference-based reinforcement learning. In NeurIPS, 2021.

[12] H. Choi, S. Jung, H. Ahn, and T. Moon. Listwise reward estimation for offline preference-based reinforcement learning. In ICML, 2024.

[13] N. Mu, H. Hu, X. Hu, Y. Yang, B. Xu, and Q.-S. Jia. Clarify: Contrastive preference reinforcement learning for untangling ambiguous queries. In ICML, 2025.

[14] R. Miao, S. Xu, R. Zhao, W. K. V. Chan, and G. Liu. Uncertainty-aware preference alignment for diffusion policies. NeurIPS, 2025.

[15] H. Cai, Z. M. Peng, and B. Zhou. Predictive preference learning from human interventions. NeurIPS, 2025.

[16] J. Hejna and D. Sadigh. Inverse preference learning: Preference-based rl without a reward function. In NeurIPS, 2023.

[17] Q. Zhang, Z. Liu, H. Fan, G. Liu, B. Zeng, and S. Liu. Flowpolicy: Enabling fast and robust 3d flow-based policy via consistency flow matching for robot manipulation. In AAAI, 2025.

[18] W. Xia, Y. Yang, H. Wu, X. Ma, T. Kong, and D. Hu. Human-assisted robotic policy refinement via action preference optimization. NeurIPS, 2025.

[19] T. Zhang, C. Yu, S. Su, and Y. Wang. Reinflow: Fine-tuning flow matching policy with online reinforcement learning. NeurIPS, 2025.

[20] B. Ibarz, J. Leike, T. Pohlen, G. Irving, S. Legg, and D. Amodei. Reward learning from human preferences and demonstrations in atari. In NeurIPS, 2018.

[21] E. Bıyık, N. Huynh, M. J. Kochenderfer, and D. Sadigh. Active preference-based gaussian process regression for reward learning. In RSS, 2020.

[22] C. Kim, J. Park, J. Shin, H. Lee, P. Abbeel, and K. Lee. Preference transformer: Modeling human preferences using transformers for rl. In ICLR, 2023.

[23] T. M. Luu, H. Kim, Y. Lee, and C. D. Yoo. Video-based optimal transport for feedback-efficient offline preference-based reinforcement learning. In ICML, 2026.

[24] L. Ouyang, J. Wu, X. Jiang, D. Almeida, C. Wainwright, P. Mishkin, C. Zhang, S. Agarwal, K. Slama, A. Ray, et al. Training language models to follow instructions with human feedback. NeurIPS, 2022.

[25] R. Rafailov, A. Sharma, E. Mitchell, C. D. Manning, S. Ermon, and C. Finn. Direct preference optimization: Your language model is secretly a reward model. NeurIPS, 2023.

[26] Z. Shan, C. Fan, S. Qiu, J. Shi, and C. Bai. Forward kl regularized preference optimization for aligning diffusion policies. In AAAI, 2025.

[27] G. An, J. Lee, X. Zuo, N. Kosaka, K.-M. Kim, and H. O. Song. Direct preference-based policy optimization without reward modeling. In NeurIPS, 2023.

[28] Y. Yuan, J. Hao, Y. Ma, Z. Dong, H. Liang, J. Liu, Z. Feng, K. Zhao, and Y. Zheng. Unirlhf: Universal platform and benchmark suite for reinforcement learning with diverse human feedback. In ICLR, 2024.

[29] Y. Wang, Z. Sun, J. Zhang, Z. Xian, E. Biyik, D. Held, and Z. Erickson. Rl-vlm-f: Reinforcement learning from vision language foundation model feedback. ICML, 2024.

[30] T. M. Luu, Y. Lee, D. Lee, S. Kim, M. J. Kim, and C. D. Yoo. Enhancing rating-based reinforcement learning to effectively leverage feedback from large vision-language models. In ICML, 2025.

[31] T. M. Luu, D. Lee, Y. Lee, and C. D. Yoo. Policy learning from large vision-language model feedback without reward modeling. In IROS, 2025.

[32] J. Ho, A. Jain, and P. Abbeel. Denoising diffusion probabilistic models. NeurIPS, 2020.

[33] Y. Lipman, R. T. Chen, H. Ben-Hamu, M. Nickel, and M. Le. Flow matching for generative modeling. arXiv:2210.02747, 2022.

[34] T. X. Pham, T. Ton, and C. Yoo. Mdsgen: Fast and efficient masked diffusion temporal-aware transformers for open-domain sound generation. In ICLR, volume 2025, 2025.

[35] T. Ton, J. W. Hong, and C. D. Yoo. Taro: Timestep-adaptive representation alignment with onset-aware conditioning for synchronized video-to-audio synthesis. In ICCV, 2025.

[36] M. Janner, Y. Du, J. B. Tenenbaum, and S. Levine. Planning with diffusion for flexible behavior synthesis. ICML, 2022.

[37] A. Ajay, Y. Du, A. Gupta, J. Tenenbaum, T. Jaakkola, and P. Agrawal. Is conditional generative modeling all you need for decision-making? ICLR, 2023.

[38] L. Yang, Z. Huang, F. Lei, Y. Zhong, Y. Yang, C. Fang, S. Wen, B. Zhou, and Z. Lin. Policy representation via diffusion probability model for reinforcement learning. arXiv:2305.13122, 2023.

[39] A. Ren, J. Lidard, L. Ankile, A. Simeonov, P. Agrawal, A. Majumdar, B. Burchfiel, H. Dai, and M. Simchowitz. Diffusion policy policy optimization. In ICLR, 2025.

[40] Y. Zhang, S. Yu, T. Zhang, M. Guang, H. Hui, K. Long, Y. Wang, C. Yu, and W. Ding. Sac flow: Sample-efficient reinforcement learning of flow-based policies via velocity-reparameterized sequential modeling. arXiv:2509.25756, 2025.

[41] A. Sridhar, D. Shah, C. Glossop, and S. Levine. Nomad: Goal masked diffusion policies for navigation and exploration. In ICRA, 2024.

[42] Y. Ze, G. Zhang, K. Zhang, C. Hu, M. Wang, and H. Xu. 3d diffusion policy: Generalizable visuomotor policy learning via simple 3d representations. arXiv:2403.03954, 2024.

[43] T. Nguyen, T. Ton, H. Choe, T. M. Luu, and C. D. Yoo. Fast and highly expressive policy learning for offline reinforcement learning via bootstrapped flow q-learning. In ICML, 2026.

[44] C. Lu, H. Chen, J. Chen, H. Su, C. Li, and J. Zhu. Contrastive energy prediction for exact energy-guided diffusion sampling in offline reinforcement learning. In ICML, 2023.

[45] S. Park, Q. Li, and S. Levine. Flow q-learning. In ICML, 2025.

[46] T. Nguyen and C. D. Yoo. One-step flow q-learning: Addressing the diffusion policy bottleneck in offline reinforcement learning. In ICLR, 2026.

[47] Z. Dong, Y. Yuan, J. Hao, F. Ni, Y. Mu, Y. Zheng, Y. Hu, T. Lv, C. Fan, and Z. Hu. Aligndiff: Aligning diverse human preferences via behavior-customisable diffusion model. In ICLR, volume 2024, pages 8929–8956, 2024.

[48] Z. Zhang, Y. Sun, J. Ye, T.-S. Liu, J. Zhang, and Y. Yu. Flow to better: Offline preference-based reinforcement learning via preferred trajectory generation. In ICLR, 2024.

[49] R. S. Sutton and A. G. Barto. Reinforcement Learning: An Introduction. MIT Press, 2018.

[50] R. A. Bradley and M. E. Terry. Rank analysis of incomplete block designs: I. the method of paired comparisons. Biometrika, 1952.

[51] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov. Proximal policy optimization algorithms. arXiv:1707.06347, 2017.

[52] D. Garreau, R. Lajugie, S. Arlot, and F. Bach. Metric learning for temporal sequence alignment. NeurIPS, 27, 2014.

[53] S. Haldar, V. Mathur, D. Yarats, and L. Pinto. Watch and match: Supercharging imitation with regularized optimal transport. In CoRL, 2022.

[54] Y. Luo, Z. Jiang, S. Cohen, E. Grefenstette, and M. P. Deisenroth. Optimal transport for offline imitation learning. In ICLR, 2023.

[55] Y. Fu, H. Zhang, D. Wu, W. Xu, and B. Boulet. Robot policy learning with temporal optimal transport reward. In NeurIPS, 2024.

[56] W. Huey, H. Wang, A. Wu, Y. Artzi, and S. Choudhury. Imitation learning from a single temporally misaligned video. ICML, 2025.

[57] J. Zhang, Y. Luo, A. Anwar, S. A. Sontakke, J. J. Lim, J. Thomason, E. Biyik, and J. Zhang. Rewind: Language-guided rewards teach robot policies without new demonstrations. CoRL, 2025.

[58] H. Tan, S. Chen, Y. Xu, Z. Wang, Y. Ji, C. Chi, Y. Lyu, Z. Zhao, X. Chen, P. Co, et al. Robo-dopamine: General process reward modeling for high-precision robotic manipulation. arXiv:2512.23703, 2025.

[59] T. Lee, A. Wagenmaker, K. Pertsch, P. Liang, S. Levine, and C. Finn. Roboreward: Generalpurpose vision-language reward models for robotics. arXiv:2601.00675, 2026.

[60] A. Liang, Y. Korkmaz, J. Zhang, M. Hwang, A. Anwar, S. Kaushik, A. Shah, A. S. Huang, L. Zettlemoyer, D. Fox, et al. Robometer: Scaling general-purpose robotic reward models via trajectory comparisons. arXiv:2603.02115, 2026.

[61] S. Nair, A. Rajeswaran, V. Kumar, C. Finn, and A. Gupta. R3m: A universal visual representation for robot manipulation. In CoRL, 2022.

[62] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In Proceedings ofthe IEEE conference on computer vision and pattern recognition, pages 770– 778, 2016.

[63] Y. Liu, J. Hamid, A. Xie, Y. Lee, M. Du, and C. Finn. Bidirectional decoding: Improving action chunking via guided test-time sampling. In ICLR, 2025.

[64] S. Mirchandani, S. Belkhale, J. Hejna, E. Choi, M. S. Islam, and D. Sadigh. So you think you can scale up autonomous robot data collection? arXiv:2411.01813, 2024.

[65] A. S. Chen, A. M. Lessing, Y. Liu, and C. Finn. Curating demonstrations using online experience. arXiv:2503.03707, 2025.

[66] H. Sakoe and S. Chiba. Dynamic programming algorithm optimization for spoken word recognition. IEEE transactions on acoustics, speech, and signal processing, 26(1):43–49, 1978.

[67] A. Majumdar, K. Yadav, S. Arnaud, J. Ma, C. Chen, S. Silwal, A. Jain, V.-P. Berges, T. Wu, J. Vakil, et al. Where are we in the search for an artificial visual cortex for embodied intelligence? NeurIPS, 36:655–677, 2023.

[68] M. Tschannen, A. Gritsenko, X. Wang, M. F. Naeem, I. Alabdulmohsin, N. Parthasarathy, T. Evans, L. Beyer, Y. Xia, B. Mustafa, et al. Siglip 2: Multilingual vision-language encoders with improved semantic understanding, localization, and dense features. arXiv:2502.14786, 2025.

[69] J. Rocamonde, V. Montesinos, E. Nava, E. Perez, and D. Lindner. Vision-language models are zero-shot reward models for reinforcement learning. In ICLR, volume 2024, pages 28446–28463, 2024.

[70] S. Sontakke, J. Zhang, S. Arnold, K. Pertsch, E. Bıyık, D. Sadigh, C. Finn, and L. Itti. Roboclip: One demonstration is enough to learn robot policies. NeurIPS, 2023.

[71] J. Bjorck, F. Castañeda, N. Cherniadev, X. Da, R. Ding, L. Fan, Y. Fang, D. Fox, F. Hu, S. Huang, et al. Gr00t n1: An open foundation model for generalist humanoid robots. arXiv:2503.14734, 2025.

[72] B. Liu, Y. Zhu, C. Gao, Y. Feng, Q. Liu, Y. Zhu, and P. Stone. Libero: Benchmarking knowledge transfer for lifelong robot learning. NeurIPS, 2023.

[73] X. Liu, C. Gong, and Q. Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv:2209.03003, 2022.

## A Appendix

A.1 Additional Experiments and Results 14   
A.1.1 Learning from Different Numbers of Rollouts 14   
A.1.2 Learning from Different Numbers of Demonstrations . 14   
A.1.3 Impact of Visual Encoder . 15   
A.1.4 Impact of Segment Length 15   
A.1.5 Multi-Demonstration Aggregation Operator . 15   
A.1.6 Comparison with Vision-Language and Progress Reward Baselines 15   
A.1.7 Computational Efficiency 16   
A.1.8 Applying PGPG with GR00T N1.7 16   
A.2 Experimental Details 17   
A.2.1 Base Policy 17   
A.2.2 Contrastive Preference Learning for Generative Policies 18   
A.2.3 Environmental Details 20   
A.2.4 Details of Baselines . 20   
A.2.5 Experimental Settings 20

## A.1 Additional Experiments and Results

## A.1.1 Learning from Different Numbers of Rollouts

We investigate how the size of the autonomous rollout dataset impacts the fine-tuning performance of PGPG. As shown in Table 2, decreasing the number of uncurated rollouts leads to a slight drop in policy performance across both tasks. For the Square task, the success rate declines steadily as the dataset shrinks, dropping from a peak of $7 2 . 0 \%$ at 500 rollouts down to 65.7% at 125 rollouts. Conversely, the Lift task demonstrates higher resilience to reduced data; it maintains near-optimal alignment (96.0%) with just 250 rollouts, and only experiences a slight decrease to 92.3% when the dataset is further reduced to 125 rollouts. This trend demonstrates that PGPG effectively extracts dense preference gradients from suboptimal data, achieving robust behavioral alignment without the need for large-scale data collection.

Table 2: PGPG’s performance using different numbers of rollouts $( i . e . , | \mathcal { D } _ { \mathrm { r o l l o u t } } | )$
<table><tr><td>Number of rollouts</td><td>Lift</td><td>Square</td></tr><tr><td>500</td><td> $9 6 . 7 \pm 1 . 2$ </td><td> $7 2 . 0 \pm 7 . 2$ </td></tr><tr><td>375</td><td> $9 4 . 7 \pm 3 . 2 $ </td><td> $7 1 . 7 \pm 2 . 5$ </td></tr><tr><td>250</td><td> $9 6 . 0 \pm 3 . 0$ </td><td> $6 9 . 7 \pm 6 . 1$ </td></tr><tr><td>125</td><td> $9 2 . 3 \pm 4 . 5 $ </td><td> $6 5 . 7 \pm 3 . 1$ </td></tr></table>

## A.1.2 Learning from Different Numbers of Demonstrations

In the main experiments, we found that 120 demonstrations were sufficient to achieve strong perfor mance for both the baselines and our method. We further evaluate smaller demonstration budgets in Table 3, while keeping the preference-generation procedure unchanged (i.e., using a fixed number of reference demonstrations, as described in Section A.2.5). With only 40 demonstrations, PGPG’s performance remains nearly unchanged on $L i \dot { \mathcal { \hat { H } } }$ and Can, with a moderate drop on Square. Even with no demonstrations (pure rollouts), PGPG still improves over the base policy on Can and Square.

Table 3: PGPG’s performance using different numbers of demonstrations $( i . e . , | \mathcal { D } _ { \mathrm { d e m o } } | )$
<table><tr><td>Number of demos</td><td>Lift</td><td>Can</td><td>Square</td></tr><tr><td>Base policy</td><td> $4 4 . 0 { \pm } 7 . 2$ </td><td> $4 2 . 7 { \pm } 1 . 2 $ </td><td> $3 9 . 3 { \scriptstyle \pm 3 . 1 }$ </td></tr><tr><td>0 (pure rollouts)</td><td> $3 2 . 7 \pm 4 . 6$ </td><td> $6 4 . 7 { \scriptstyle \pm 6 . 4 }$ </td><td> $5 1 . 3 { \scriptstyle \pm 5 . 0 }$ </td></tr><tr><td>40</td><td> $9 5 . 3 { \pm } 1 . 2 $ </td><td> $8 8 . 0 { \scriptstyle \pm 4 . 0 }$ </td><td> $5 6 . 7 \pm 6 . 1$ </td></tr><tr><td>120</td><td> $9 6 . 7 \pm 1 . 2$ </td><td> $8 9 . 3 { \scriptstyle \pm 4 . 6 }$ </td><td> $7 2 . 0 { \scriptstyle \pm 7 . 2 }$ </td></tr></table>

## A.1.3 Impact of Visual Encoder

We vary the visual encoder within the PGPG framework. The results are reported in Table 4. The R3M encoder used in our main experiments performs best, while VC-1 [67] is competitive and the general vision-language SigLIP2 [68] is weaker. Consistent with ORCA [56], the choice of encoder is a practical, task-dependent consideration.

Table 4: PGPG performance with different visual encoders.
<table><tr><td>Encoder</td><td>Lift</td><td>Can</td><td>Square</td></tr><tr><td>Base policy</td><td> $4 4 . 0 { \pm } 7 . 2 $ </td><td> $4 2 . 7 { \pm } 1 . 2 $ </td><td> $3 9 . 3 { \scriptstyle \pm 3 . 1 }$ </td></tr><tr><td>PGPG w/ VC-1</td><td> $9 6 . 0 { \scriptstyle \pm 4 . 0 }$ </td><td> $8 3 . 3 { \scriptstyle \pm 4 . 6 }$ </td><td> $5 2 . 7 \pm 3 . 1$ </td></tr><tr><td>PGPG w/ SigLIP2</td><td> $6 8 . 7 \pm 7 . 0$ </td><td> $5 8 . 0 { \scriptstyle \pm 8 . 7 }$ </td><td> $4 2 . 7 { \pm } 7 . 0$ </td></tr><tr><td>PGPG w/ R3M (Ours)</td><td> $9 6 . 7 { \scriptstyle \pm 1 . 2 }$ </td><td> $8 9 . 3 { \scriptstyle \pm 4 . 6 }$ </td><td> $7 2 . 0 { \scriptstyle \pm 7 . 2 }$ </td></tr></table>

## A.1.4 Impact of Segment Length

Table 5 illustrates the impact of segment length on the Square task. Performance is notably lower at $L = 1 0 0$ , suggesting that segments must be sufficiently long to capture meaningful physical transitions rather than trivial movements. However, at $L = 1 5 0 ,$ , performance increases substantially and plateaus through $L = 2 0 0$ . This indicates that once the segment length is adequate to encapsulate meaningful sub-behaviors, preference learning with PGPG remains highly stable.

Table 5: Performance of PGPG with different segment lengths L on Square task.
<table><tr><td>L</td><td>100</td><td>150</td><td>200</td></tr><tr><td>Success Rate</td><td> $5 4 . 3 \pm 7 . 6$ </td><td> $7 1 . 3 \pm 7 . 1$ </td><td> $7 2 . 0 \pm 7 . 2$ </td></tr></table>

## A.1.5 Multi-Demonstration Aggregation Operator

As shown in Table 6, the max operator consistently outperforms mean pooling across both tasks. This validates our hypothesis: averaging progress scores across diverse demonstrations heavily penalizes valid, multimodal behaviors by forcing segments to match a potentially non-existent “blurry” trajectory. Conversely, the max operator explicitly preserves multimodality by rewarding a segment if it aligns with any single valid expert mode, effectively mitigating spurious matches and inconsistent preference rankings.

## A.1.6 Comparison with Vision-Language and Progress Reward Baselines

To compare PGPG with other preference-generation methods that do not require querying large VLMs or LLMs, we evaluate three approaches within our exact pipeline, replacing only the label source: frame-based CLIP [69], video-based S3D [70], and the Robometer progress reward model [60]. As shown in Table 7, PGPG performs on par with or better than all baselines on all considered tasks. Robometer performs competitively only on Can and struggles on the precision and long-horizon tasks, while being approximately 31× slower than PGPG (see Section A.1.7). CLIP and S3D measure similarity between visual observations and the task description but do not explicitly account for ordered subgoal completion, a key property captured by ORCA.

Table 6: Performance of PGPG with different aggregation operators across demonstrations.
<table><tr><td>Aggregation</td><td>Lift</td><td>Square</td></tr><tr><td>Mean</td><td> $9 0 . 7 \pm 2 . 3 $ </td><td> $5 2 . 7 \pm 2 . 1$ </td></tr><tr><td>Max</td><td> $9 6 . 7 \pm 1 . 2$ </td><td> $7 2 . 0 \pm 7 . 2$ </td></tr></table>

Table 7: Comparison with Vision-Language and Progress Reward Baselines. Best results and those within one standard deviation are bolded.
<table><tr><td>Method</td><td>Lift</td><td>Can</td><td>Square</td><td></td><td>Transp. ToolHang</td></tr><tr><td>CLIP</td><td> $3 6 . 0 { \scriptstyle \pm 7 . 2 }$ </td><td> $6 2 . 0 { \scriptstyle \pm 6 . 0 }$ </td><td> $5 8 . 0 { \scriptstyle \pm 8 . 7 }$ </td><td> $2 1 . 3 { \pm } 4 . 2$ </td><td> $3 0 . 7 { \scriptstyle \pm 9 . 0 }$ </td></tr><tr><td>S3D</td><td> $5 3 . 3 { \scriptstyle \pm 3 . 1 }$ </td><td> $6 0 . 0 { \scriptstyle \pm 3 . 5 }$ </td><td> $6 2 . 8 { \scriptstyle \pm 4 . 6 }$ </td><td> $2 6 . 7 \pm 4 . 2$ </td><td> ${ \pm } 4 . 0 { \pm } 8 . 7$ </td></tr><tr><td>Robometer</td><td> $3 8 . 7 \pm 8 . 1 $ </td><td> $\mathbf { 8 5 . 3 } \pm 3 . 1 $ </td><td> $4 0 . 0 { \scriptstyle \pm 3 . 5 }$ </td><td> $2 8 . 0 { \scriptstyle \pm 6 . 9 }$ </td><td> $3 1 . 3 { \pm } 6 . 1$ </td></tr><tr><td>PGPG (Ours)</td><td> ${ \bf 9 6 . 7 } \pm 1 . 2$ </td><td> ${ \bf 8 9 . 3 } \pm 4 . 6$ </td><td> $7 2 . 0 { \scriptstyle \pm 7 . 2 }$ </td><td> ${ \pm 8 . 7 \pm 6 . 1 }$ </td><td> ${ \pm } 6 . 0 { \pm } 8 . 7$ </td></tr></table>

## A.1.7 Computational Efficiency

Figure 4 compares the inference latency of PGPG against Robometer [60]. While PRMs and VLMs introduce significant wall-clock latency per rollout due to heavy network inference, leveraging the ORCA metric [56] allows PGPG to evaluate trajectory progress significantly faster.

![](images/3a315ff54b327577633cac43baa074fe03fc2364d73b4c6d78baeeb467c34d5d.jpg)  
Figure 4: Average wall-clock time required to compute progress rewards per trajectory, evaluated over 50 rollouts. By leveraging the ORCA progress metric, PGPG is nearly 31× faster than the Robometer [60] baseline, successfully eliminating the high inference latency typically associated with progress reward models.

## A.1.8 Applying PGPG with GR00T N1.7

The experiments in the main paper fine-tune task-specific diffusion and flow-based policies, each trained on a single Robomimic task. To test whether PGPG generalizes beyond this setting, we apply it to GR00T N1.7 [71], a large pretrained vision-language-action (VLA) model, on the multi-task LIBERO-10 benchmark [72]. This evaluation further demonstrates the effectiveness of our method in three aspects:

(i) Multi-task learning. A single language-conditioned policy is fine-tuned on all ten tasks at once. Preferences are generated for each task against that task’s own demonstrations and then pooled into one fine-tuning run, without task-specific models or hyperparameters.

(ii) Pretrained VLA. The policy is a 3B-parameter model in which a vision-language backbone conditions a flow-matching action head. PGPG needs only the flow-matching loss of the action head, which takes the place of the log-likelihood in $\mathrm { C P L } _ { K L }$ , so it applies to the VLA without architectural changes.

(iii) Few-shot data. Only 5 demonstrations and 20 policy rollouts are available per task, compared with about 120 demonstrations and 500 rollouts per task in the main experiments.

Benchmark and Dataset. LIBERO-10 (also known as LIBERO-Long) consists of ten long-horizon, language-conditioned manipulation tasks for a simulated Franka Panda arm, each chaining two sub goals (illustrated in Figure 5). The policy takes as input a third-person and a wrist-camera image $( 2 5 6 \times 2 5 6 )$ , the end-effector state, and the instruction. We study a few-shot learning setting in which only five teleoperated demonstrations are available per task (50 in total, from the LIBERO-10 dataset with no-op actions removed). The pretrained policy collects 20 rollouts per task (200 in total), of which 75 are successful.

Pretrained Policy. GR00T N1.7 combines a vision-language model (VLM) backbone (Cosmos-Reason2-2B) with a Diffusion Transformer action head trained with flow matching, which predicts action chunks of length $T _ { p } = 1 6$ . We obtain the pretrained policy $\pi _ { \mathrm { r e f } }$ by fine-tuning the released GR00T N1.7 checkpoint on the 50 demonstrations for 5000 iterations, using a batch size of 16 and a learning rate of $1 0 ^ { - 4 }$ , while keeping the VLM backbone frozen.

Implementation of PGPG. Pseudo-preference labels are generated as in the main paper: ORCA progress [56] is computed on R3M features [61] of the third-person image by comparing it with the five demonstrations of the same task, and each segment is scored by its marginal progress, maximized over the demonstrations. Segments span 160 environment steps (ten action chunks). For each task we sample 5000 demonstration-vs-rollout queries and 5000 rollout-vs-rollout queries, require both segments in each query to start at the same progress level $\mathrm { ( \pm 0 . 0 5 ) }$ , and discard queries whose score gap is below 0.1, leaving 47878 labeled pairs. The policy is fine-tuned with $\mathrm { C P L } _ { K L }$ [2] using $\pi _ { \mathrm { r e f } }$ as the reference. Because the action head is a flow-matching model, the per-chunk log-likelihood ratio is replaced by the difference between the velocity-prediction errors of $\pi _ { \theta }$ and $\pi _ { \mathrm { r e f } } .$ , computed with shared noise and flow time $( \alpha = 0 . 0 1 5$ , discount 0.999, weight 0.25 on the losing segment, with per-step terms clipped at 0.2). Only the action head is updated using cached backbone features. We train for 8000 steps with AdamW (learning rate $1 0 ^ { - 4 }$ , cosine schedule with 5% warm-up) and 48 preference pairs per batch on four NVIDIA A100 80 GB GPUs.

Baseline and Evaluation. Filtered BC fine-tunes $\pi _ { \mathrm { r e f } }$ using the flow-matching objective on the demonstrations and 75 successful rollouts, with the same learning rate, batch size, and number of training steps. Unlike PGPG, it relies on the simulator’s success signal to select rollouts. All policies are evaluated at their final checkpoints over 50 episodes per task for each of three seeds. For a given seed, every method starts from the same initial states. The policies execute the first eight actions of each predicted chunk, and an episode ends upon success or after 720 steps.

Results. The results are shown in Table 8. PGPG improves the average success rate from 50.5% for the pretrained policy to 72.1%, compared with 64.8% for Filtered BC. This corresponds to gains of $2 1 . 5 \pm 1 . 8$ percentage points over the pretrained policy and $7 . 3 \pm 3 . 6$ percentage points over Filtered BC. PGPG achieves the best performance on seven of the ten tasks, with the largest gains on T1 $( 1 8 . 0 \%  6 4 . 7 \% )$ , T8 $( 4 0 . 0 \%  8 4 . 0 \% )$ and $\Gamma 2 ( 5 8 . 0 \% \to 9 4 . 7 \% )$ . Filtered BC performs better on T5 and T6. Neither method solves T9 (success rates of at most 3.3%), where none of the 20 rollouts collected by the pretrained policy succeeds.

## A.2 Experimental Details

## A.2.1 Base Policy

We adopt Diffusion Policy [5] and Rectified Flow Policy [73, 19] as the base policies in PGPG due to their strong ability to model complex robot action distributions. Both policies employ Transformerbased architectures, serving as the noise predictor and velocity predictor for diffusion and flow matching, respectively. Additionally, following [5], we use two historical states as policy inputs and predict $T _ { p }$ future actions, where $T _ { p } = 1 6$ for Diffusion Policy and $T _ { p } = 1 0$ for Rectified Flow Policy. We denote the predicted action chunk by $A _ { t } = \left( \mathbf { a } _ { t : t + T _ { p } - 1 } \right)$

Diffusion Policy. During training, we follow the original Diffusion Policy implementation of [5]. At inference time, given a state ${ \mathbf { s } } ,$ an initial action chunk $A ^ { K }$ is sampled from a standard Gaussian distribution. A learned denoising network $\epsilon _ { \theta } ( A ^ { k } , { \bf s } , k )$ then iteratively denoises $A ^ { k }$ over $K$ diffusion steps, yielding a sequence of intermediate action chunks and ultimately the final noise-free action

T2: Put both the cream cheese box and the butter in the basket

T1: Put both the alphabet soup and the tomato sauce in the basket  
![](images/b7ff711b919e73a211beb5f2f1f99b5780529c35cdb463b46ed3d58bb60fab5e.jpg)  
T3: Turn on the stove and put the moka pot on it

![](images/01475fb4ce85f5a76f4927368a84e6b07f512f900a49d98aa1c46c1b25e4dc51.jpg)  
T4: Put the black bowl in the bottom drawer of the cabinet and close it

![](images/4afa1136755bdce5c3d2078b9e8c01022faf5b7fd78fe2e0d1e8a1d75507b681.jpg)

![](images/534472133ce0323bd76a7cf9dedde58dd80886d63d64eff03883a59cdd3c9dc1.jpg)  
T5: Put the white mug on the left plate and put the yellow and white mug on the right plate  
T6: Pick up the book and place it in the back compartment of the caddy

![](images/f73cbe067f297a10078ede3e1180811d04df77ecac5c582e2a02e3c0d449e13e.jpg)  
T7: Put the white mug on the plate and put the chocolate pudding to the right of the plate

![](images/85bc64eccae38d0a24d1f59b93c72aef71ccd48a82b1461a628732102a142b69.jpg)

![](images/cded28f0f68d05a30cfab4f45a0f678d98bd811549f189e0a9169f606191a1a2.jpg)  
T8: Put both the alphabet soup and the cream cheese box in the basket  
T9: Put both moka pots on the stove

![](images/c25b78b565953a91236fdd438a99a511ff45e2f900d5d6411328377013d553d7.jpg)

![](images/5ed8f8be4dd190b9b66d7cce79a5b6cc32abbb69645e46f261e09dd80309f512.jpg)  
T10: Put the yellow and white mug in the microwave and close it

![](images/a63fd87a3a67c26ff9fc50fa4c8820eba6afbd8628fb825046c2ad6f789ad4a2.jpg)  
Figure 5: LIBERO-10 tasks: Each row of frames shows one of the five available expert demonstrations for a task, progressing from the initial state (left) to task completion (right).

chunk $A ^ { 0 }$ . This process can be written as:

$$
A ^ { k - 1 } = \alpha _ { k } ( A ^ { k } - \beta _ { k } \epsilon _ { \theta } ( A ^ { k } , { \bf s } , k ) + \mathcal { N } ( 0 , \sigma _ { k } ^ { 2 } I ) ) ,\tag{7}
$$

where $\alpha _ { k } , \beta _ { k } , \sigma _ { k }$ are functions of the iteration step k that define the noise schedule. Here, the state s serves as the conditioning variable in the denoising process.

Flow Policy. Flow matching [73] offers a streamlined and generalized alternative to traditional diffusion models by learning a continuous vector field that maps a simple, tractable prior to the target data distribution. We adopt the Rectified Flow policy used in [19] in our work. During training, the learnable velocity field $v _ { \theta } ( \cdot )$ learns to transport Gaussian noise $A ^ { 0 } \sim \mathcal { N } ( 0 , I )$ to an expert action $A ^ { 1 }$ via the probability path $A ^ { \tau } = ( 1 - \tau ) A ^ { 0 } + \tau A ^ { 1 }$ , where $\tau \in [ 0 , 1 ]$ is the flow matching step. The policy is trained by regressing $v _ { \theta } ( \cdot )$ to the target velocity $A ^ { 1 } - A ^ { 0 }$ as follows:

$$
\mathcal { L } _ { \mathrm { F M } } ( \theta ) = \mathbb { E } _ { \tau , A ^ { 0 } \sim \mathcal { N } ( 0 , I ) , ( A ^ { 1 } , \mathbf { s } ) } \left[ \| A ^ { 1 } - A ^ { 0 } - v _ { \theta } ( A ^ { \tau } , \tau , \mathbf { s } ) \| _ { 2 } ^ { 2 } \right] .\tag{8}
$$

We define $\{ \tau _ { k } \} _ { k = 0 } ^ { K }$ as a discretization of the time interval [0, 1], with $\Delta \tau = \tau _ { k + 1 } - \tau _ { k }$ . At inference, actions are generated by solving the ODE $\begin{array} { r } { \frac { d A ^ { \tau } } { d \tau } = v _ { \theta } ( A ^ { \tau } , \bar { \tau } , \mathbf { s } ) } \end{array}$ via the Euler update:

$$
A ^ { \tau _ { k + 1 } } = A ^ { \tau _ { k } } + \Delta \tau v _ { \theta } ( A ^ { \tau _ { k } } , \tau _ { k } , \mathbf { s } ) .\tag{9}
$$

## A.2.2 Contrastive Preference Learning for Generative Policies

As our backbone preference learning algorithm, we adopt a variant of contrastive preference learning (CPL) [2], following [14] to align the generative policies. Specifically, the training objective for the

Table 8: Success rate (%) of methods on LIBERO-10. Mean ± standard deviation over three seeds with 50 episodes per task each. The best result for each task is shown in bold.
<table><tr><td>Task</td><td>Instruction</td><td></td><td>Pretrained Filtered BC</td><td>PGPG</td></tr><tr><td>T1</td><td>put both the alphabet soup and the tomato sauce in the basket</td><td> $1 8 . 0 \pm 5 . 3$ </td><td> $3 7 . 3 \pm 3 . 1$ </td><td> ${ \bf 6 4 . 7 \pm 7 . 6 }$ </td></tr><tr><td>T2</td><td>put both the cream cheese box and the butter in the basket</td><td> $5 8 . 0 \pm 2 . 0$ </td><td> $7 6 . 7 \pm 1 . 2$ </td><td> $\mathbf { 9 4 . 7 \pm 3 . 1 }$ </td></tr><tr><td>T3</td><td>turn on the stove and put the moka pot on it</td><td> $8 0 . 0 \pm 3 . 5$ </td><td> $7 6 . 7 \pm 1 . 2$ </td><td> ${ \bf 8 7 . 3 \pm 3 . 1 }$ </td></tr><tr><td>T4</td><td>put the black bowl in the bottom drawer of the cabinet and close</td><td> $8 6 . 7 \pm 4 . 2$ </td><td> $9 6 . 0 \pm 2 . 0$ </td><td> ${ \bf 1 0 0 . 0 \pm 0 . 0 }$ </td></tr><tr><td>T5</td><td>it put the white mug on the left plate and put the yellow and white mug on the right plate</td><td> $6 4 . 0 \pm 5 . 3$ </td><td> ${ \bf 8 1 . 3 \pm 1 . 2 }$ </td><td> $7 8 . 7 \pm 1 . 2$ </td></tr><tr><td>T6</td><td>pick up the book and place it in the back compartment of the caddy</td><td> $5 3 . 3 \pm 8 . 1$ </td><td> $7 2 . 7 \pm 4 . 2$ </td><td> $5 8 . 7 \pm 1 . 2$ </td></tr><tr><td>T7</td><td>put the white mug on the plate and put the chocolate pudding to</td><td> $6 3 . 3 \pm 1 . 2$ </td><td> $7 8 . 7 \pm 8 . 1$ </td><td> ${ \bf 8 4 . 7 \pm 9 . 9 }$ </td></tr><tr><td>T8</td><td>the right of the plate put both the alphabet soup and the cream cheese box in the basket</td><td> $4 0 . 0 \pm 1 0 . 4$ </td><td> $6 2 . 0 \pm 6 . 0$ </td><td> ${ \bf 8 4 . 0 \pm } 2 . 0$ </td></tr><tr><td>T9</td><td>put both moka pots on the stove</td><td> $3 . 3 \pm 3 . 1$ </td><td> $3 . 3 \pm 3 . 1$ </td><td> $1 . 3 \pm 1 . 2$ </td></tr><tr><td>T10</td><td>put the yellow and white mug in the microwave and close it</td><td> $3 8 . 7 \pm 7 . 0$ </td><td> $6 3 . 3 \pm 1 3 . 3$ </td><td> ${ \bf 6 6 . 7 \pm 6 . 4 }$ </td></tr><tr><td>Average</td><td></td><td> $5 0 . 5 \pm 2 . 0$ </td><td> $6 4 . 8 \pm 2 . 8$ </td><td> $7 2 . 1 \pm 1 . 1$ </td></tr></table>

diffusion policy is defined as follows:

$$
\begin{array} { l } { \displaystyle \mathcal { L } _ { \mathrm { D P } } ^ { ( \sigma ^ { w } , \sigma ^ { l } ) } ( \theta ) = - \mathbb { E } _ { \mathbf { a } _ { t } ^ { k , w } \sim q ( \mathbf { a } _ { t } ^ { k , w } | \mathbf { a } _ { t } ^ { 0 , w } ) , \mathbf { a } _ { t } ^ { k , l } \sim q ( \mathbf { a } _ { t } ^ { k , l } | \mathbf { a } _ { t } ^ { 0 , l } ) , k \sim \mathcal { U } ( 1 , K ) } \Bigg [ \log \sigma \Bigg ( - \alpha K \cdot } \\ { \displaystyle ( \sum _ { t = 1 } ^ { L } \gamma ^ { t } ( \| { \epsilon } _ { t } ^ { w } - { \epsilon } _ { \theta } ( \mathbf { a } _ { t } ^ { k , w } , \mathbf { s } _ { t } ^ { w } , k ) \| _ { 2 } ^ { 2 } - \| { \epsilon } _ { t } ^ { w } - { \epsilon } _ { \mathrm { r e f } } ( \mathbf { a } _ { t } ^ { k , w } , \mathbf { s } _ { t } ^ { w } , k ) \| _ { 2 } ^ { 2 } )  \Bigg . \Bigg . } \\ { \displaystyle   - \sum _ { t = 1 } ^ { L } \gamma ^ { t } ( \| { \epsilon } _ { t } ^ { l } - { \epsilon } _ { \theta } ( \mathbf { a } _ { t } ^ { k , l } , \mathbf { s } _ { t } ^ { l } , k ) \| _ { 2 } ^ { 2 } - \| { \epsilon } _ { t } ^ { l } - { \epsilon } _ { \mathrm { r e f } } ( \mathbf { a } _ { t } ^ { k , l } , \mathbf { s } _ { t } ^ { l } , k ) \| _ { 2 } ^ { 2 } ) ) \Bigg ) \Bigg ] , } \end{array}\tag{10}
$$

where $\sigma ^ { w }$ and $\sigma ^ { l }$ denote the preferred and rejected trajectory segments, respectively; k is the diffusion timestep, K is the number of diffusion denoising steps, $\mathbf { \bar { a } } ^ { k , w } \sim q ( \mathbf { a } _ { t } ^ { k , w } | \mathbf { \bar { a } } _ { t } ^ { 0 , w } )$ denotes the groundtruth action $\mathbf { a } ^ { 0 , w }$ corrupted with noise $\epsilon _ { t } ^ { w }$ after k forward diffusion steps. The noisy rejected action $\mathbf { a } ^ { k , l }$ is defined similarly.

Similarly, for the flow policy, we adapt the CPL objective by replacing the diffusion denoising error with the velocity-prediction error. In Rectified Flow, the network regresses the target velocity between the ground-truth action and the sampled noise. Thus, the preference learning objective for the flow policy is defined as:

$$
\begin{array} { l } { \displaystyle \mathcal { L } _ { \mathrm { F P } } ^ { ( \sigma ^ { w } , \sigma ^ { l } ) } ( \theta ) = - \mathbb { E } _ { \mathbf { a } _ { t } ^ { 0 , w } , \mathbf { a } _ { t } ^ { 0 , l } \sim \mathcal { N } ( 0 , \mathbf { I } ) , \tau \sim \mathcal { U } ( 0 , 1 ) } \Bigg [ \log \sigma \Bigg ( - \alpha \cdot } \\ { \displaystyle ( \sum _ { t = 1 } ^ { L } \gamma ^ { t } ( \| ( \mathbf { a } _ { t } ^ { 1 , w } - \mathbf { a } _ { t } ^ { 0 , w } ) - v _ { \theta } ( \mathbf { a } _ { t } ^ { \tau , w } , \tau , \mathbf { s } _ { t } ^ { w } ) \| _ { 2 } ^ { 2 } - \| ( \mathbf { a } _ { t } ^ { 1 , w } - \mathbf { a } _ { t } ^ { 0 , w } ) - v _ { \mathrm { r e f } } ( \mathbf { a } _ { t } ^ { \tau , w } , \tau , \mathbf { s } _ { t } ^ { w } ) \| _ { 2 } ^ { 2 } )  \Bigg . } \\ { \displaystyle   - \sum _ { t = 1 } ^ { L } \gamma ^ { t } ( \| ( \mathbf { a } _ { t } ^ { 1 , l } - \mathbf { a } _ { t } ^ { 0 , l } ) - v _ { \theta } ( \mathbf { a } _ { t } ^ { \tau , l } , \tau , \mathbf { s } _ { t } ^ { l } ) \| _ { 2 } ^ { 2 } - \| ( \mathbf { a } _ { t } ^ { 1 , l } - \mathbf { a } _ { t } ^ { 0 , l } ) - v _ { \mathrm { r e f } } ( \mathbf { a } _ { t } ^ { \tau , l } , \tau , \mathbf { s } _ { t } ^ { l } ) \| _ { 2 } ^ { 2 } ) ) \Bigg ) \Bigg ] , } \end{array}\tag{11}
$$

where $\tau \in [ 0 , 1 ]$ is the continuous flow-matching step, and $\mathbf { a } _ { t } ^ { 1 , w }$ denotes the ground-truth preferred action. The term $\mathbf { a } _ { t } ^ { 0 , w } \sim \mathcal { N } ( 0 , \mathbf { I } )$ is the initial sampled noise, and $\mathbf { a } _ { t } ^ { \tau , w } = \overline { { ( 1 - \tau ) } } \mathbf { a } _ { t } ^ { 0 , w ^ { \bullet } } + \tau \mathbf { a } _ { t } ^ { 1 , w }$ represents the interpolated action along the probability path. The actions for the rejected segment $\mathbf { a } _ { t } ^ { \mathrm { i } , l } , \mathbf { a } _ { t } ^ { 0 , l }$ , and $\mathbf { a } _ { t } ^ { \tau , l }$ are defined similarly.

In practice, because our generative policies predict action chunks of length $T _ { p } ,$ evaluating the loss at every single timestep within a segment of length L incurs prohibitive memory costs due to gradient accumulation. To alleviate this overhead and accelerate training, we stride the timestep summation in the learning objective by $T _ { p }$ . As a result, the loss is evaluated only at non-overlapping action chunks, substantially reducing memory consumption without compromising training stability. This strategy is applied to both diffusion and flow policies.

## A.2.3 Environmental Details

We use Robomimic [4], a large-scale robotic manipulation benchmark designed for imitation learning and offline reinforcement learning (RL). It consists of five manipulation tasks, each accompanied by a proficient human (ph) teleoperated demonstration dataset. Four tasks (excluding Tool Hang) additionally provide mixed proficient/non-proficient human (mh) demonstration datasets. In our experiments, we use the mh datasets when applicable, as they are generally considered more challenging than the ph datasets due to the inclusion of demonstrations from multiple non-proficient operators. The five tasks considered in this paper are:

$L i f t { \mathrm { : } }$ The robot arm must lift a small cube.

• Can: The robot must place a coke can from a large bin into a smaller target bin.

• Square: The robot must pick a square nut and place it on a rod.

• Transport: Two robot arms must transfer a hammer from a closed container on a shelf to a target bin on another shelf. One robot arm must retrieve the hammer from the container, while the other arm must clear the target bin by moving a piece of trash to the nearby receptacle. Finally, one arm must hand the hammer over to the other, which must place the hammer in the target bin.

• Tool Hang: A robot arm must assemble a frame consisting of a base piece and hook piece by inserting the hook into the base, and hang a wrench on the hook. This is the most difficult task due to the multiple stages that each require precise, dexterous, rotation-heavy movements.

## A.2.4 Details of Baselines

For both Mixed BC and Filtered BC, we fine-tune the pretrained policies using a small set of expert demonstrations combined with the exact same rollout trajectories used by PGPG. For Filtered BC, any trajectories that fail to satisfy the task-specific success criteria are discarded. To ensure a fair comparison, the reinforcement learning baseline, DPPO [39], is restricted to approximately the same number of environment interactions (rollouts) as our method. Furthermore, both $\mathbf { C P L } _ { K L }$ and DPPO utilize the same low-dimensional state inputs as all other methods. $\mathbf { C P L } _ { K L }$ relies on the ground-truth environment rewards provided by the benchmark (dense when available), whereas DPPO uses the sparse task rewards for policy optimization. All methods use low-dimensional state observations. For PGPG, we additionally leverage the corresponding image observations (at a resolution of 224 × 224) solely for pseudo-preference label generation.

## A.2.5 Experimental Settings

For pseudo-preference label generation, we set the number of expert reference demonstrations to $M = 1 0$ for Lift, $M = 2 0$ for Can, and $M = 4 0$ for the remaining tasks. In practice, we select the M shortest demonstrations to form the reference set. To construct the initial preference dataset, we uniformly sample N candidate segment pairs from the merged dataset—specifically, $N = 1 0 0 0 0$ for Lift, $N = 2 0 0 0 0$ for Can, and $N = 3 0 0 0 0$ for the remaining tasks—and perform a search over the discrimination threshold $\epsilon _ { \sigma } \in \{ 0 . 3 , 0 . 4 , 0 . 6 \}$ . We select the highest threshold that successfully retains at least 1,000 valid preference pairs for Lift and Can, and 5,000 pairs for the remaining tasks.

From a computational perspective, computing the ORCA metric in Eq. (2) between a rollout and an expert demonstration incurs a dynamic programming cost of $\mathcal { O } ( H \cdot \tilde { H } )$ [56]. To improve efficiency, the entire labeling process is performed offline prior to policy training. Specifically, instead of anchoring each rollout to a single demonstration, we compute and cache M independent progress sequences per rollout trajectory, one for each expert demonstration. With these step-wise progress traces precomputed, evaluating the aggregated score for an arbitrary segment reduces to an O(M) operation. This requires only computing the marginal progress difference for each of the M expert demonstrations and applying the max-pooling operation described previously. As a result, the proposed caching strategy eliminates redundant sequence-matching computations and substantially accelerates preference dataset construction. We follow the implementation of [56] using cosine distance for the spatial cost matrix, while utilizing an adaptive temperature that scales dynamically with the minimum spatial cost to ensure robust temporal alignment.

Our implementation is built upon the codebases of [5] and [14]. Additional hyperparameter details for PGPG are provided in Table 9.

Table 9: Comprehensive list of hyperparameters for PGPG.
<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td></td><td>Preference Generation (PGPG)</td></tr><tr><td>Reference Demonstrations (M) 10-40</td><td></td></tr><tr><td>Discrimination Threshold (€σ)</td><td>{0.3, 0.4, 0.6}</td></tr><tr><td>Visual Feature Encoder</td><td>R3M [61]</td></tr><tr><td>Distance Metric</td><td>Cosine Distance</td></tr><tr><td>Segment Length (L)</td><td>100 (Lift, Can), 200 (Square), 300 (others)</td></tr><tr><td></td><td>Policy Optimization</td></tr><tr><td>Optimizer</td><td>AdamW</td></tr><tr><td>Learning Rate</td><td> $1 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Weight Decay</td><td> $1 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>Batch Size</td><td>128</td></tr><tr><td>Training Epochs</td><td>1000</td></tr><tr><td>Denoise Steps</td><td>100 (Diffusion), 8 (Flow)</td></tr></table>