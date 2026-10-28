# LEARNING QUALITY-CONTROLLABLE ROBOT POLI-CIES WITH ITERATIVE HINDSIGHT RELABELING

Sinjae Kang<sup>1∗</sup> Dong Kyu Shin<sup>1</sup> Jaehong Kim<sup>2</sup> Kamin Lee<sup>2</sup> Hyoeun Kim<sup>2</sup> Kaixin Wang<sup>3</sup> Li Zhao<sup>3</sup> Seonghyeon Ye<sup>1†</sup> Kimin Lee<sup>1,4†</sup>

<sup>1</sup>KAIST <sup>2</sup>LGE <sup>3</sup>Microsoft Research <sup>4</sup>Config

## ABSTRACT

Robot policy evaluation often relies on a binary success signal for task completion. In real-world deployment, however, desired execution behaviors are also defined by quality attributes such as execution speed and motion smoothness, which need to be dynamically controlled according to task requirements. To address this, we introduce Quality-Controllable Policy (QCP), a robot policy that enables inference-time control over both individual quality attributes and their combinations. To extend the controllable quality range beyond initial demonstrations, we further propose an iterative hindsight relabeling loop: QCP collects online rollouts under target quality labels, adds successful trajectories to the dataset, relabels the accumulated trajectories in hindsight according to their measured quality, and fine-tunes the policy. Experiments on Robomimic and RoboEval demonstrate that QCP consistently improves both task execution quality and success rates across iterations, producing successful rollouts that exceed the quality of the initial demonstrations. Finally, we validate QCP on real-world manipulation tasks, where controlling speed increases throughput in conveyor operations and controlling transport stability improves success when carrying wine glasses, demonstrating that quality-controllable policies yield practical gains in physical deployment.

Project page: https://sinnnj.github.io/qcp

## 1 INTRODUCTION

Robot foundation models have shown strong generalization across tasks, environments, and objects (Black et al., 2025; Bjorck et al., 2025; Ye et al., 2026). These policies are usually trained with behavior cloning (BC) on successful demonstrations, which treats every successful trajectory as equally desirable regardless of how the task was executed. As a result, the learned policy reproduces the mixture of execution behaviors in the data, whether fast or slow, smooth or jerky, and offers no way to steer toward the behavior a deployment actually requires. In practice, however, the desired behavior of a robot is not defined by task success alone (Wang et al., 2025; Liu et al., 2026; Mao et al., 2026) and different tasks call for different quality attributes. For example, a robot deployed on a conveyor line should prioritize execution speed, whereas a robot carrying wine glasses on a tray must prioritize transport stability.

To address this, prior work has proposed quality-controllable policies whose execution behavior can be adjusted along a chosen quality axis at inference time (Zhan et al., 2020; Yang et al., 2025; Petitbois et al., 2026). However, such policies are trained on static demonstration datasets, which typically cover only a bounded quality range and lack the high-performance regimes needed in deployment (e.g., maximum speed or ideal smoothness). Our goal is therefore not only to learn a quality-controllable policy, but also to continuously expand its controllable range beyond the initial demonstrations through online data collection and policy updates.

In this paper, we introduce Quality-Controllable Policy (QCP), which addresses this limitation of BC by making execution quality an explicit conditioning input (Figure 1). For each demonstration, we measure the target quality attributes with rule-based score functions and quantize the scores into discrete quality labels. The policy is then trained to generate actions conditioned on both the observation history and these labels. Rather than averaging over execution behaviors, QCP learns task completion from all successful trajectories while distinguishing how each one was executed. At inference time, the user specifies a target label to steer execution behavior, for example, requesting the highest speed label for conveyor operations.

![](images/556279f286438a27ae6a6508c4f4011735b3d8fbd9bdb95ea5eb2d9275014f66.jpg)  
Figure 1: Overview of QCP and iterative hindsight relabeling. Left: QCP controls execution quality at inference time through conditioning on quantized quality tokens. Right: Iterative hindsight relabeling shifts the quality distribution of successful rollouts toward higher quality, expanding the controllable range beyond the initial demonstrations.

To extend the controllable quality range beyond the demonstrations, we pair QCP with iterative hindsight relabeling. At each iteration, we collect rollouts conditioned on the highest quality label, keep the successful ones, relabel them by the quality they actually achieved, and add them to the dataset. We then recompute the quality labels of all trajectories and retrain the policy. As higher-quality rollouts accumulate, the highest label comes to represent progressively higher quality, allowing the policy to reach operating regimes absent from the initial demonstrations. Because quality is measured by rule-based score functions, this loop requires no manual quality annotation.

Experiments on Robomimic (Mandlekar et al., 2022) and RoboEval (Wang et al., 2025) show that QCP with iterative hindsight relabeling improves both target quality and task success across iterations, outperforming quality-filtered BC and RL baselines (Mao et al., 2024; Petitbois et al., 2026; Wagenmaker et al., 2025; Dong et al., 2026). Iterative relabeling pushes the quality distribution well beyond the range of the initial demonstrations, reaching an IQR-normalized Wasserstein distance above 1.5. Beyond single-attribute control, QCP controls multiple attributes jointly, improving six quality attributes simultaneously in RoboEval. Finally, in real-world experiments, controlling execution speed substantially increases throughput in conveyor operations, and controlling transport stability improves success when carrying wine glasses on a tray.

## 2 RELATED WORK

Policy learning with quality control. Prior work studies how to learn different ways of executing a task, representing behavioral styles through learned trajectory representations or explicit labels (Petitbois et al., 2025; Zhan et al., 2020; Yang et al., 2025). Learning a requested style, however, can conflict with task performance. SORL addresses this tension by combining trajectory clustering with task advantage weighting (Mao et al., 2024), while SCIQL combines task and style advantages to account for both task performance and style alignment (Petitbois et al., 2026). QCP combines quantized quality conditioning with iterative hindsight relabeling. Our experiments show simultaneous gains in execution quality and task success, demonstrating that this trade-off can be overcome.

Online improvement of behavioral quality. Freeform Preference Learning (FPL) learns a reward model for the desired quality axes from human comparisons of robot trajectories and trains policies conditioned on the predicted rewards (Torne et al., 2026). This supports online improvement through repeated rollout collection and policy training, but requires additional human comparisons at each round. In text generation, Quark repeatedly samples under the highest reward token, relabels accu mulated samples by reward quantiles, and retrains the language model conditioned on the assigned labels (Lu et al., 2022). QCP applies a similar loop to robot execution quality, using automatically computed quality labels and accumulated successful trajectories to extend the controllable range beyond the initial demonstrations without additional human annotation.

## 3 METHOD

In real-world deployment, users often care about quality metrics beyond binary task success, such as execution speed or motion smoothness. To account for these, we propose Quality-Controllable Policy (QCP), which adjusts its execution behavior along user-specified quality axes by conditioning on quantized quality levels (Section 3.1). Because demonstration data typically covers only a limited range of quality, we extend $\mathrm { Q C P ^ { \bullet } s }$ controllable quality range through iterative hindsight relabeling, alternating between collecting online rollouts, relabeling trajectories by their achieved quality, and updating the policy (Section 3.2). Figure 1 illustrates the overall framework.

## 3.1 QUALITY-CONTROLLABLE POLICY

Quality scores and quantization. Let D be a dataset of successful demonstrations, where each trajectory $\tau = ( o _ { 0 } , a _ { 0 } , \ldots , o _ { T - 1 } , a _ { T - 1 } , o _ { T } )$ consists of observations $o _ { t }$ and executed actions $a _ { t } .$ with $T$ the number of executed actions. We assume access to a rule-based score function $m ( \tau ) \in \mathbb { R }$ that measures a quality axis of interest. The simplest example is $\begin{array} { r } { m ( \tau { } ) = \frac { 1 } { T } } \end{array}$ as a proxy for execution speed, and our experiments additionally consider more complex scores such as normalized action differences for smoothness and total joint path length for efficiency (see Appendix A for detailed definitions).

We then quantize the scores into B equal-frequency bins and use the bin index as a discrete quality label:

$$
q ( \tau ) = \big [ \boldsymbol { B } \cdot \hat { \boldsymbol { F } } ( \boldsymbol { m } ( \tau ) ) \big ] - 1 \in \{ 0 , \dots , B - 1 \} , \qquad \hat { \boldsymbol { F } } ( s ) = \frac { 1 } { | \mathcal { D } | } \sum _ { \tau ^ { \prime } \in \mathcal { D } } \mathbf { 1 } \{ \boldsymbol { m } ( \tau ^ { \prime } ) \leq s \} ,\tag{1}
$$

where $\hat { F }$ is the empirical CDF of scores in $\mathcal { D } ,$ , so that $q ( \tau ) = j$ if and only if $m ( \tau )$ lies between the $j / B$ and $( j + 1 ) / \bar { B }$ empirical quantiles. Every trajectory in D is thus paired with its label $q ( \tau )$ , and we train a quality-conditioned policy on this labeled dataset.<sup>1</sup>

Quality-conditioned policy training. We train a policy $\pi _ { \theta }$ that, given a history of $L$ observations $\begin{array} { r c l } { h _ { t } } & { = } & { \left( o _ { t - L + 1 } , \dots , o _ { t } \right) } \end{array}$ and a quality label $q ,$ predicts a chunk of H future actions $A _ { t } = \left( a _ { t } , \ldots , a _ { t + H - 1 } \right)$ . We construct training samples $( h _ { t } , A _ { t } , q )$ by slicing each trajectory $\tau \in \mathcal { D }$ at every time step t and attaching its label $q = q ( \tau )$ . Denoting the resulting set of training samples by ${ \mathcal { D } } _ { \mathrm { t r } }$ , the policy is trained with the following quality-conditioned behavior cloning objective:

$$
\mathcal { L } ( \theta ) = - \mathbb { E } _ { ( h _ { t } , A _ { t } , q ) \sim \mathcal { D } _ { \mathrm { t r } } } \left[ \log \pi _ { \theta } ( A _ { t } \mid h _ { t } , q ) \right] .\tag{2}
$$

We model $\pi _ { \theta }$ as a flow matching policy (Lipman et al., 2023), widely used in recent robot policies (Black et al., 2025; Bjorck et al., 2025; Kang et al., 2026), which optimizes equation $2$ in practice by training a velocity field with a regression objective. Given noise $\epsilon \sim \mathcal { N } ( 0 , I )$ and flow time $u \sim \mathcal { U } [ 0 , 1 ]$ , we form the interpolated action chunk $A _ { t } ^ { u } = ( 1 - u ) \epsilon + u A _ { t }$ and train a velocity field $v _ { \theta }$ to predict the target $A _ { t } - \epsilon \colon$

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { F M } } ( \theta ) = \mathbb { E } _ { ( h _ { t } , A _ { t } , q ) \sim \mathcal { D } _ { \mathrm { t r } } , \ u , \epsilon } \left[ \left| \left| v _ { \theta } \left( A _ { t } ^ { u } , u , h _ { t } , q \right) - \left( A _ { t } - \epsilon \right) \right| \right| ^ { 2 } \right] , } \end{array}\tag{3}
$$

where the norm is taken over all action dimensions and time steps in the chunk. We consider two cases for conditioning on quality $q .$ For simple policies trained from scratch, an embedding layer maps q to a learned vector $\mathbf { e } _ { q } \in \mathbb { R } ^ { d }$ , which is concatenated with the encoded observation history to condition the velocity network.<sup>2</sup> For vision-language-action (VLA) models such as $\pi _ { 0 . 5 }$ (Black et al., 2025), we instead insert the label into the language prompt as text $( \mathrm { e . g . }$ , “quality level: $3 ^ { \mathfrak { s } } )$ similar to $\pi _ { 0 . 7 } ( \mathrm { A i }$ et al., 2026), and fine-tune the model with its original training recipe. This enables the model to interpret quality levels through language conditioning without architectural changes.

Inference during deployment. At inference time, the quality label becomes a control knob: the user specifies a target bin $q ^ { \star } \in \{ 0 , \ldots , B - 1 \}$ , and actions are generated by integrating the learned velocity field $v _ { \theta } ( \cdot , u , h _ { t } , q ^ { \star } )$ from $u = 0 \mathrm { t o } u = 1$ starting from Gaussian noise. We find that QCP responds reliably to this input, shifting its behavior toward the requested bin and attaining a quality score within the corresponding range (Figure 3). Unless noted, we condition on the highest bin $q ^ { \star } = B - 1$ throughout our evaluation and in the iterative training procedure of Section 3.2.

## 3.2 ITERATIVE HINDSIGHT RELABELING

A policy trained only on demonstrations is largely bounded by the quality present in D. To push beyond this range, we introduce a simple iterative pipeline that alternates between rollout collection, hindsight relabeling, and policy updates. The key observation is that rollouts conditioned on the highest bin can exceed the best demonstrations (Figure 4). We attribute this to two factors. First, the label is attached to every action chunk of a trajectory, so the policy learns how each stage of the task is executed in the high-quality demonstrations and can recombine these segments into a trajectory whose quality exceeds that of any single demonstration. Second, action generation is stochastic, and conditioning on the highest bin of the target quality axis raises the probability that a sampled rollout attains higher quality along that axis than the demonstrations. Folding these rollouts into the dataset and recomputing the labels moves the highest bin upward, and retraining on the relabeled data lets the policy produce behavior that the previous policy did not; repeating this expands the controllable quality range over iterations (Section 5.1).

Algorithm 1 QCP with Iterative Hind  
sight Relabeling   
Require: Demonstrations $\mathcal { D } ,$ score m, $B$   
bins, K collection rounds.   
1: Set $\mathcal { D } _ { 0 }  \mathcal { D } , q ^ { \star }  B - 1 .$   
2: Label $\mathcal { D } _ { 0 }$ and train $\pi _ { \theta _ { 0 } }$ as in Section 3.1.   
3: for $k = 0 , \ldots , K - 1$ do   
4: Collect rollouts with $\pi _ { \boldsymbol { \theta } _ { k } } ( \cdot \mid h _ { t } , \boldsymbol { q } ^ { \star } )$   
5: Compute m $( \tau )$ for each rollout.   
6: Append successful rollouts to $\mathcal { D } _ { k }$ to ob  
tain $\mathcal { D } _ { k + 1 }$   
7: Compute $\hat { F }$ from scores in $\mathcal { D } _ { k + 1 }$   
8: Recompute $q ( \tau )$ for all $\tau \in \mathcal { D } _ { k + 1 }$ us  
ing Equation 1.   
9: Train $\pi _ { \boldsymbol { \theta } _ { k + 1 } }$ from scratch on $\mathcal { D } _ { k + 1 }$ us  
ing Equation 3.   
10: end for

We initialize the dataset as $\mathcal { D } _ { 0 } = \mathcal { D }$ and train an initial policy $\pi _ { \theta _ { 0 } }$ on it as in Section 3.1. At each iteration k, we execute $\pi _ { \theta _ { k } }$ conditioned on the highest bin $q ^ { \star } = B - 1$ to collect a batch of rollouts, and keep only those that succeed at the task.<sup>3</sup> Although every rollout is requested at the highest bin, the realized scores vary. We therefore relabel each successful rollout in hindsight by its realized score $m ( \tau )$ rather than the requested label, and append it to $\mathcal { D } _ { k }$ to obtain $\mathcal { D } _ { k + 1 }$ . We then recompute the quality labels $q ( \tau )$ for all trajectories in $\mathcal { D } _ { k + 1 }$ , including the original demonstrations, by applying equation 1 with the empirical CDF taken over $\mathcal { D } _ { k + 1 }$ . As higher-scoring rollouts accumulate, the bin boundaries shift upward, progressively raising the quality represented by the highest bin while the original demonstrations preserve the low-quality range and thus widen the controllable range.

Finally, we train $\pi _ { \boldsymbol { \theta } _ { k + 1 } }$ on the relabeled $\mathcal { D } _ { k + 1 }$ as in Section 3.1. Rather than fine-tuning $\pi _ { \boldsymbol { \theta } _ { k } }$ , we reinitialize the policy at every iteration, from random weights for policies trained from scratch and from the pretrained checkpoint for VLAs. This is because relabeling changes which quality range each label refers to, so the label-to-behavior mapping learned in the previous iteration no longer matches the new targets and would need to be unlearned.<sup>4</sup> We repeat rollout collection, relabeling, and retraining for a fixed number of iterations, as summarized in Algorithm 1.

## 4 EXPERIMENTS

We investigate the performance of QCP with iterative hindsight relabeling by focusing on the following four research questions:

(1) Does iterative hindsight relabeling consistently improve the target quality of QCP while maintaining task success? (Section 4.2)

(2) Does iterative hindsight relabeling expand the controllable quality range of QCP? (Section 4.3)

(3) Can QCP jointly control and improve multiple quality attributes? (Section 4.4)

(4) Does our approach remain effective on real-world robot tasks? (Section 4.5)

## 4.1 EXPERIMENTAL SETUP

Tasks and datasets. Our main simulation experiments use Square and Transport tasks from Robomimic (Mandlekar et al., 2022) and Pack Box and Pick Book tasks from RoboEval (Wang et al., 2025). For Robomimic, we use the Multi-Human (MH) datasets, whose demonstrations are collected by operators of varying proficiency and thus span a wide range of execution quality. For RoboEval, we use the demonstration datasets with randomized object positions and orientations. For the real-world experiments in Section 4.5, we use a Franka Research 3 arm with the DROID hardware setup (Khazatsky et al., 2024), and collect 400 teleoperated demonstrations for the conveyor task and 50 for the wine-glass transport task to form each task’s initial dataset $\mathcal { D } _ { 0 }$

Baselines. We compare QCP against three groups of baselines. To test whether data accumulation or quality filtering alone can achieve similar gains, we consider BC trained on accumulated successful rollouts $\mathrm { ( B C _ { a c c } ) }$ , and quality-filtered BC using the best 10% of successful trajectories along the target axis $( \mathrm { B C _ { f i l t e r } } ) . ^ { 5 }$ To compare against existing methods for learning desired behavioral styles, we evaluate SORL (Mao et al., 2024) and SCIQL (Petitbois et al., 2026), which learn style-conditioned policies through offline RL. Finally, to compare quality conditioning against direct optimization of quality rewards, we evaluate DSRL (Wagenmaker et al., 2025) and EXPO (Dong et al., 2026), which improve flow matching policies with online RL. For both, we add a weighted quality reward to the task reward and train them for 720k environment steps, the budget of QCP’s 1,800 rollouts (300 per iteration over six iterations) at a 400-step limit. Reward settings and interaction budgets are detailed in Appendix C.

Quality metrics. On Robomimic, we target execution speed, measured by steps to success, and action smoothness, measured by action jerk. On RoboEval, we target execution speed, measured by completion time, and motion efficiency, measured by total joint path length. We negate or invert qualities for which lower values indicate higher quality to obtain the score $m ( \tau )$ used in equation 1. In real-world experiments, we target execution speed, measured by end-effector speed, and trans port stability, measured by the end-effector tilt from vertical. Detailed definitions are provided in Appendix A.

Training and evaluation. In simulation, we use a simple flow matching policy with an MLP backbone, trained for 100k steps on Robomimic and 30k steps on RoboEval with a batch size of 1024. In real-world experiments, we fine-tune the $\pi _ { 0 . 5 }$ vision-language-action (VLA) model (Black et al., 2025) for 10 epochs with a batch size of 32. These settings are fixed across iterations. We use $B = 4$ quality bins and always condition QCP on the highest bin $q ^ { \star } = 3$ during rollout collection.<sup>6</sup> At each iteration, we collect 300 rollouts per task and seed in simulation, and in the real world, as many rollouts as there are demonstrations in the initial dataset $\mathcal { D } _ { 0 }$ . These rollouts serve both for evaluation and as training data for the next iteration. Throughout, iteration k refers to the policy $\pi _ { \boldsymbol { \theta } _ { k } }$ trained on $\mathcal { D } _ { k }$ and to the rollouts it collects. Iteration 0 thus refers to the policy trained on the demonstrations alone, and results at iteration k reflect k rounds of rollout collection. We report mean quality over successful rollouts and success rate over all rollouts, both averaged across three seeded runs of the full iterative procedure.

![](images/a99c8c9b98f355e6613078d4039cc68b642606fccdb4cb53e986e89cada65520.jpg)  
Figure 2: Comparison of QCP with BC baselines $( \mathbf { B C _ { \mathrm { f i l t e r } } } , \ \mathbf { B C _ { \mathrm { a c c } } } )$ and behavioral quality learning methods (SORL, SCIQL) across iterations on (a) Robomimic and (b) RoboEval. Each plot shows task success rate against a quality metric, with the y-axis oriented so that higher is better. Darker and larger markers correspond to later iterations. QCP improves quality and task success together more effectively than the baselines. Results are averaged over three random seeds.

## 4.2 ITERATIVE IMPROVEMENT OVER QUALITY AND SUCCESS RATE

Figure 2 shows execution quality and task success across iterations. Across tasks and quality attributes, QCP improves both target quality and task success, whereas the baselines tend to improve one at the expense of the other, each for a different reason. $\mathrm { \Delta B C _ { \mathrm { a c c } } }$ has no quality signal and improves task success alone. $\mathbf { B C } _ { \mathrm { f i l t e r } }$ and SORL, which train only on trajectories whose label matches the request, discard most of the successful data, at a cost in task success. SCIQL keeps all trajectories but weights them by the sum of a success advantage and a quality advantage, which creates a tradeoff between the two (Appendix D). QCP avoids these problems: all successful trajectories are used by conditioning on their quality labels, so even low-quality trajectories contribute to learning task completion, while the requested label selects high-quality behavior at inference.

QCP also achieves higher final quality than the flow matching RL baselines DSRL and EXPO (Table 1), even when we assign high weights to their quality rewards.<sup>7</sup> We attribute this to the fact that these RL methods for flow matching policies are tuned to consume a success signal, and a reward along a new quality axis is not consumed as effectively. QCP instead provides the quality signal in a supervised manner by conditioning the policy on achieved quality labels, achieving high execution quality together with strong task success.

Table 1: QCP versus RL baselines. Results at the final iteration (iteration 5, i.e., after five rounds of rollout collection) are averaged across three random seeds. Higher success rates and lower quality metric values are better.
<table><tr><td rowspan="2">Method</td><td colspan="2">Sq+Speed</td><td colspan="2">Sq+Smoothness</td><td colspan="2">Tr+Speed</td><td colspan="2">Tr+Smoothness</td><td colspan="2">Pack+Speed</td><td colspan="2">Pack+JPL</td><td colspan="2">Book+Speed</td><td colspan="2">Book+JPL</td></tr><tr><td>SR (%)</td><td>Speed (steps)</td><td>SR (%)</td><td>Smoothness (×10−3)</td><td>SR (%)</td><td>Speed (steps)</td><td>SR (%)</td><td>Smoothness (×10−3)</td><td>SR (%)</td><td>Speed (s)</td><td>SR (%)</td><td>JPL (rad)</td><td>SR (%)</td><td>Speed (s)</td><td>SR (%)</td><td>JPL (rad)</td></tr><tr><td>QCP (ours)</td><td>82.2</td><td>120.3</td><td>79.7</td><td>8.5</td><td>22.2</td><td>432.8</td><td>12.3</td><td>22.2</td><td>45.8</td><td>4.82</td><td>35.3</td><td>6.42</td><td>47.0</td><td>2.43</td><td>37.9</td><td>4.63</td></tr><tr><td>SORL</td><td>53.2</td><td>156.0</td><td>48.4</td><td>11.5</td><td>4.6</td><td>466.8</td><td>5.7</td><td>34.9</td><td>39.7</td><td>5.64</td><td>41.8</td><td>7.11</td><td>48.9</td><td>3.37</td><td>41.4</td><td>5.15</td></tr><tr><td>SCIQL</td><td>68.7</td><td>206.5</td><td>83.0</td><td>12.8</td><td>10.6</td><td>540.8</td><td>7.9</td><td>47.2</td><td>39.9</td><td>9.24</td><td>37.4</td><td>9.06</td><td>48.4</td><td>4.75</td><td>47.0</td><td>6.40</td></tr><tr><td>DSRL</td><td>37.0</td><td>208.8</td><td>43.0</td><td>46.9</td><td>4.0</td><td>549.5</td><td>7.0</td><td>79.4</td><td>60.7</td><td>6.35</td><td>50.7</td><td>66.05</td><td>0.0</td><td></td><td>0.0</td><td></td></tr><tr><td>EXPO</td><td>96.0</td><td>162.5</td><td>79.0</td><td>157.4</td><td>11.0</td><td>567.7</td><td>3.0</td><td>176.5</td><td>39.0</td><td>8.72</td><td>43.7 24.43</td><td></td><td>3.0</td><td>5.73</td><td>4.3</td><td>29.20</td></tr></table>

## 4.3 QUALITY CONTROL AND RANGE EXPANSION

Quality control across iterations. We first test whether QCP can produce different levels of execution quality on request by evaluating it at each bin in {0, 1, 2, 3} at every iteration (Figure 3). Even the initial policy trained only on demonstrations (iteration 0) follows the requested bins: on Square, for example, requesting the highest bin yields faster execution than the lowest, with mean steps to success of $1 9 3 . { \bar { 7 } }$ at $q = 3$ versus 308.1 at $q = 0 ,$ , and the intermediate bins fall in between. We further find that this ordering holds for every attribute at every iteration, even though new data comes only from the highest bin. Iterative relabeling thus raises the quality of the highest bin over iterations while preserving the ability to select different execution behaviors from a single policy.

![](images/3f6d656b92fb274632fe2e856f50a3c160d634d4f9b9a2d4ab44adf5e1112f39.jpg)

![](images/f27ce82e733254d27f3fae1f33345e0e37d493a0451e2b217ec3f1c3a112f054.jpg)

![](images/831fb99597c9d1c3c88db6dedc6c5357703589cdd47c89840353ad97e8e4e638.jpg)

![](images/4f19149d1d2a7b602dd7995d895463ad71058046de95c6e03732fc21dfe00fa0.jpg)

![](images/e378c34f254d4e997d27d83309f38cab848e4a7d065c9639af36e93f1fde5806.jpg)

![](images/8b55932835b39723eeda43117de2151daf8ad38a16f1a0916fed5f9da35b79de.jpg)

![](images/afb58c2f0aa585001d75b22be85fda9e4fda7fb5c4cb39c044a9826003ea5131.jpg)  
Figure 3: Quality achieved at each requested bin across iterations on Robomimic Square. The ordering of the bins is preserved while the whole range moves toward higher quality (arrows). Mean ± one standard deviation over three seeds.

Quality beyond the initial demonstrations. Figure 4 further shows that successful rollout dis tributions shift beyond the initial demonstration range for Square smoothness and RoboEval joint path length. We quantify the distribution shift using the Wasserstein-1 distance between the initial demonstrations and final successful rollouts, normalized by the interquartile range (IQR) of the initial demonstration scores. This distance exceeds 1.5 for smoothness and joint path length, indicating large shifts from the initial demonstration distribution. For speed, the rollout distribution shifts toward shorter completion times, extending beyond the range observed in the initial demonstrations.<sup>8</sup> These results show that online rollout collection and iterative learning enable QCP to expand its capable quality range beyond the initial demonstrations while successfully completing the task.

![](images/12017ab37012fb3924025f6e7d47d05a546efe9edb68fda0a7fe50553634557b.jpg)  
Initial Demos It0 It1 It2 It3 It4 It5  
Figure 4: Quality distributions across iterations. Starting from the initial demonstration distribution, QCP shifts the quality distribution of successful rollouts toward higher quality, extending beyond the initial demonstration range. Arrows point toward higher quality.

## 4.4 MULTI-ATTRIBUTE QUALITY CONTROL AND IMPROVEMENT

We next evaluate whether QCP can control multiple quality attributes in the requested directions and improve them through iterative learning.

Bidirectional control of two quality attributes. On Robomimic Square, we train a separate QCP policy for each of the four combinations of faster or slower execution and smaller or larger lift amplitude.<sup>9</sup> Figure 5 shows that both attributes change in the requested directions over iterations. Across all four combinations, the final policies achieve quality levels close to those of policies trained for each attribute alone, showing that QCP can pursue both objectives together with comparable quality.

![](images/6b49b2d8f9d13bb821d06f61bcaea4f26da4f1bc6c91de49cc4b435e5b78de32.jpg)  
Fast + Low Fast + High Slow + Low Slow + High  
Figure 5: QCP jointly adjusts execution speed and lift amplitude in the requested directions over iterations.

Joint improvement across six attributes. We further evaluate whether QCP can jointly improve up to six quality attributes across four RoboEval tasks (Figure 6). We condition each policy on six quality labels and request the best bin for every attribute during iterative learning (see Appendix E for experimental details). The final policies achieve better quality than initial BC and $\mathrm { \Delta B C _ { \mathrm { a c c } } }$ for most quality attributes across the evaluated tasks. These results suggest that QCP can jointly improve multiple aspects of execution even when a single policy is conditioned on up to six quality attributes.

![](images/ca968ea093ac955cd19dedf6e9592790a603cffe5f5ee64f7a135284912d25b8.jpg)  
Figure 6: Joint improvement of six quality attributes on RoboEval. Final QCP and $\mathrm { \Delta B C _ { \mathrm { a c c } } }$ policies are compared with initial BC on four tasks. Each axis shows the initial BC score divided by the method’s score, so the initial BC polygon lies on the 1× ring and outward indicates better quality. Success rates of initial BC and final QCP are shown above each task.

## 4.5 QUALITY CONTROL FOR REAL ROBOT TASKS

We evaluate whether controlling execution quality leads to practical gains in real-world robot tasks. We consider two tasks with distinct quality requirements: execution speed for the conveyor task, measured by end-effector speed, and transport stability for the wine-glass task, measured by the tray’s tilt angle from horizontal (Appendix A).

Serving wine. The robot must carry a tray holding wine glasses from one table to another without dropping any glass. This task is challenging even for human operators: only 50 of 382 teleoperated attempts (13.1%) succeeded during data collection, as keeping the tray level while moving proved difficult. Conditioning QCP on the highest stability bin reduces the mean tilt angle from 0.0660 rad for BC to 0.0482 rad, a 27.0% reduction, while the success rate improves from 30.7% to 42.0% over iterations (Figure 7a). In contrast, $\mathrm { \Delta B C _ { f i l t e r } }$ improves tilt angle and success rate through iteration 1 but degrades at iteration 2, suggesting that retaining lower quality data provides broader coverage needed to sustain task success. Notably, even the initial QCP policy, trained only on demonstrations, already achieves a higher success rate than BC. We attribute this to the quality label, which separates stable from unstable executions in the demonstrations so that the policy does not average over them. The final policy succeeds far more often than teleoperation did during data collection.

Conveyor belt work. Twenty cans are placed on a conveyor belt moving at a fixed speed, one every three seconds, and throughput is defined as the number of cans successfully grasped per minute. Achieving high throughput under this fixed schedule requires both fast execution and a high success rate. Relative to BC, QCP increases mean end-effector speed from 188.6 to 257.6 mm/s and throughput from 8.2 to 13.0 cans per minute, improvements of 36.6% and 59.3% (Figure 7b). In contrast, $\mathbf { B C } _ { \mathrm { f i l t e r } }$ falls behind $\mathsf { Q C P }$ in end-effector speed after iteration 0 (217.7 mm/s at iteration 2), and its throughput drops from 8.3 to 6.3 cans per minute, below BC, consistent with the loss of coverage observed in wine serving. The initial QCP policy again already exceeds the throughput of BC, while the final QCP policy surpasses the 173.5 mm/s of human teleoperation, where high-speed control is challenging.

(a) Serving wine  
![](images/f6d057c67432a6c88723dba1ee4b87d49bbdcef05416ebba2108979798776cbb.jpg)

![](images/734f26a662d9a3b8464eba05eafb16c07127e0dd064eac6754be24eb51128693.jpg)

(b) Conveyor belt work  
![](images/56cfe5b03b514e2095e5f6b9d4295d62fc8f9d19658d604fc267c58301797b06.jpg)  
Figure 7: Quality control on real robots. QCP achieves higher task success than BC baselines while (a) reducing the tray tilt angle in wine serving and (b) increasing end-effector speed in conveyor work. Values are means over three seeds; error bars and the shaded band around BC denote ±1 standard deviation.

## 5 ANALYSIS

## 5.1 EFFECT OF QUALITY CONDITIONING ON ROLLOUT DISTRIBUTIONS

We examine whether the distribution shift in Figure 4 comes from the iterative hindsight relabeling loop or from quality conditioning alone, since a conditioned policy might reach rare behaviors simply by sampling more. To separate the two, we spend the same rollout budget in two ways under the best quality bin $q ^ { \star } = 3 \colon$ 1,800 rollouts from the QCP policy frozen at iteration 0, or 300 rollouts at each of iterations 0 to 5 with retraining in between (Figure 8). The fixed policy reproduces its iteration-0 distribution regardless of sample count, and none of its successful rollouts reaches the median smoothness of the final iteration. The improvement thus comes from policy updates on the relabeled data, which move the best bin to a quality level the previous policy could not produce.

![](images/4c163e3856315c9ac8652e1d7cf1cab7106ec46169f4f492f27a7a59c2d0c909.jpg)  
Figure 8: Iterative versus fixed policy rollouts at equal sample counts.

![](images/103552e95e2e3115750575da7113252501059ce2e89d8c21724f5847def963ae.jpg)  
Figure 9: Quantized labels with 4 to 32 bins versus continuous score conditioning on Square speed.

## 5.2 CONTINUOUS QUALITY SCORES VERSUS QUANTIZED QUALITY LABELS

We examine why QCP uses quantized quality labels rather than continuous scores. With four quantile bins, roughly the best quarter of the training trajectories share the best label, giving the policy multiple examples of the requested behavior. Learning from multiple trajectories with the same la bel can provide a more stable training signal than conditioning on a score represented by only a few examples. The importance of training support is also evident in return conditioning, where Decision Transformer (Chen et al., 2021) and RvS (Emmons et al., 2022) can be unreliable for requests outside the training distribution (Brandfonbrener et al., 2022). On Square, quantized quality labels execute faster than a fixed continuous target at every iteration, and four bins also reach higher task success from iteration 1 onward; increasing the number of bins beyond four buys a small gain in speed at a growing cost in task success (Figure 9). These results support simple quantization as an effective representation for quality control. Experimental details are provided in Appendix F.

## 6 CONCLUSION

We presented QCP, a robot policy that uses quantized quality labels to control individual execution attributes and their combinations at inference time. Iterative hindsight relabeling extends the controllable quality range beyond the initial demonstrations through online rollout collection and policy updates, without additional human annotation. Experiments on Robomimic and RoboEval show that QCP can improve execution quality and task success while retaining quality control across iterations. Real robot experiments further demonstrate that QCP can be directed to pursue the execution quality desired for a given task.

## REFERENCES

Bo Ai, Ali Amin, Raichelle Aniceto, Ashwin Balakrishna, Greg Balke, Kevin Black, George Bokinsky, Shihao Cao, Thomas Charbonnier, Vedant Choudhary, Foster Collins, Ken Conley, Grace Connors, James Darpinian, Karan Dhabalia, Maitrayee Dhaka, Jared DiCarlo, Danny Driess, Michael Equi, Adnan Esmail, Yunhao Fang, Chelsea Finn, Catherine Glossop, Thomas Godden, Ivan Goryachev, Lachlan Groom, Haroun Habeeb, Hunter Hancock, Karol Hausman, Gashon Hussein, Victor Hwang, Brian Ichter, Connor Jacobsen, Szymon Jakubczak, Rowan Jen, Tim Jones, Gregg Kammerer, Ben Katz, Liyiming Ke, Mairbek Khadikov, Chandra Kuchi, Marinda Lamb, Devin LeBlanc, Brendon LeCount, Sergey Levine, Xinyu Li, Adrian Li-Bell, Vladislav Lialin, Zhonglin Liang, Wallace Lim, Yao Lu, Enyu Luo, Vishnu Mano, Nandan Marwaha, Aikys Mongush, Liam Murphy, Suraj Nair, Tyler Patterson, Karl Pertsch, Allen Z. Ren, Gavin Schelske, Charvi Sharma, Baifeng Shi, Lucy Xiaoyang Shi, Laura Smith, Jost Tobias Springenberg, Kyle Stachowicz, Will Stoeckle, Jiaming Tang, Jimmy Tanner, Shalom Tekeste, Marcel Torne, Kyle

Vedder, Quan Vuong, Anna Walling, Haohuan Wang, Jason Wang, XuDong Wang, Chris Whalen, Samuel Whitmore, Blake Williams, Charles Xu, Sukwon Yoo, Lili Yu, Wuming Zhang, Zhuoyang Zhang, and Ury Zhilinsky. π : A steerable generalist robotic foundation model with emergent capabilities. arXiv preprint arXiv:2604.15483, 2026.

Johan Bjorck, Valts Blukis, Fernando Castaneda, Nikita Cherniadev, Xingye Da, Runyu Ding, Linxi ˜ Fan, Yu Fang, Dieter Fox, Fengyuan Hu, Spencer Huang, Joel Jang, Xiaowei Jiang, Kaushil Kundalia, Jan Kautz, Zhiqi Li, Kevin Lin, Zongyu Lin, Loic Magne, Yunze Man, Ajay Mandlekar, Avnish Narayan, Soroush Nasiriany, Scott Reed, You Liang Tan, Guanzhi Wang, Jing Wang, Qi Wang, Shihao Wang, Jiannan Xiang, Yuqi Xie, Yinzhen Xu, Seonghyeon Ye, Zhiding Yu, Yizhou Zhao, Zhe Zhang, Ruijie Zheng, and Yuke Zhu. GR00T N1.5: An improved open foundation model for generalist humanoid robots. NVIDIA Research Blog, 2025. URL https://research.nvidia.com/labs/gear/gr00t-n1\_5/.

Kevin Black, Noah Brown, James Darpinian, Karan Dhabalia, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Manuel Y. Galliker, Dibya Ghosh, Lachy Groom, Karol Hausman, Brian Ichter, Szymon Jakubczak, Tim Jones, Liyiming Ke, Devin LeBlanc, Sergey Levine, Adrian Li-Bell, Mohith Mothukuri, Suraj Nair, Karl Pertsch, Allen Z. Ren, Lucy Xiaoyang Shi, Laura Smith, Jost Tobias Springenberg, Kyle Stachowicz, James Tanner, Quan Vuong, Homer Walke, Anna Walling, Haohuan Wang, Lili Yu, and Ury Zhilinsky. π<sub>0.5</sub>: A vision-language-action model with open-world generalization. In Conference on Robot Learning, 2025.

David Brandfonbrener, Alberto Bietti, Jacob Buckman, Romain Laroche, and Joan Bruna. When does return-conditioned supervised learning work for offline reinforcement learning? In Advances in Neural Information Processing Systems, 2022.

Lili Chen, Kevin Lu, Aravind Rajeswaran, Kimin Lee, Aditya Grover, Michael Laskin, Pieter Abbeel, Aravind Srinivas, and Igor Mordatch. Decision transformer: Reinforcement learning via sequence modeling. In Advances in Neural Information Processing Systems, 2021.

Cheng Chi, Siyuan Feng, Yilun Du, Zhenjia Xu, Eric Cousineau, Benjamin Burchfiel, and Shuran Song. Diffusion policy: Visuomotor policy learning via action diffusion. In Robotics: Science and Systems, 2023.

Perry Dong, Qiyang Li, Dorsa Sadigh, and Chelsea Finn. EXPO: Stable reinforcement learning with expressive policies. In International Conference on Learning Representations, 2026.

Scott Emmons, Benjamin Eysenbach, Ilya Kostrikov, and Sergey Levine. RvS: What is essential for offline RL via supervised learning? In International Conference on Learning Representations, 2022.

Sinjae Kang, Chanyoung Kim, Kaixin Wang, Li Zhao, and Kimin Lee. WarmPrior: Straightening flow-matching policies with temporal priors. arXiv preprint arXiv:2605.13959, 2026.

Alexander Khazatsky, Karl Pertsch, Suraj Nair, Ashwin Balakrishna, Sudeep Dasari, Siddharth Karamcheti, Soroush Nasiriany, Mohan Kumar Srirama, Lawrence Yunliang Chen, Kirsty Ellis, Peter David Fagan, Joey Hejna, Masha Itkina, Marion Lepert, Yecheng Jason Ma, Patrick Tree Miller, Jimmy Wu, Suneel Belkhale, Shivin Dass, Huy Ha, Arhan Jain, Abraham Lee, Youngwoon Lee, Marius Memmel, Sungjae Park, Ilija Radosavovic, Kaiyuan Wang, Albert Zhan, Kevin Black, Cheng Chi, Kyle Beltran Hatch, Shan Lin, Jingpei Lu, Jean Mercat, Abdul Rehman, Pannag R Sanketi, Archit Sharma, Cody Simpson, Quan Vuong, Homer Rich Walke, Blake Wulfe, Ted Xiao, Jonathan Heewon Yang, Arefeh Yavary, Tony Z. Zhao, Christopher Agia, Rohan Baijal, Mateo Guaman Castro, Daphne Chen, Qiuyu Chen, Trinity Chung, Jaimyn Drake, Ethan Paul Foster, Jensen Gao, Vitor Guizilini, David Antonio Herrera, Minho Heo, Kyle Hsu, Jiaheng Hu, Muhammad Zubair Irshad, Donovon Jackson, Charlotte Le, Yunshuang Li, Kevin Lin, Roy Lin, Zehan Ma, Abhiram Maddukuri, Suvir Mirchandani, Daniel Morton, Tony Nguyen, Abigail O’Neill, Rosario Scalise, Derick Seale, Victor Son, Stephen Tian, Emi Tran, Andrew E. Wang, Yilin Wu, Annie Xie, Jingyun Yang, Patrick Yin, Yunchu Zhang, Osbert Bastani, Glen Berseth, Jeannette Bohg, Ken Goldberg, Abhinav Gupta, Abhishek Gupta, Dinesh Jayaraman, Joseph J Lim, Jitendra Malik, Roberto Mart´ın-Mart´ın, Subramanian Ramamoorthy, Dorsa Sadigh, Shuran Song, Jiajun Wu, Michael C. Yip, Yuke Zhu, Thomas Kollar, Sergey Levine, and Chelsea Finn.

DROID: A large-scale in-the-wild robot manipulation dataset. arXiv preprint arXiv:2403.12945, 2024.

Ilya Kostrikov, Ashvin Nair, and Sergey Levine. Offline reinforcement learning with implicit qlearning. In International Conference on Learning Representations, 2022.

Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. In International Conference on Learning Representations, 2023.

Mengyuan Liu, Juyi Sheng, Peiming Li, Ziyi Wang, Tianming Xu, Tiantian Xu, and Hong Liu. Eval-Actions: Fine-grained execution quality evaluation for robotic manipulation. arXiv preprint arXiv:2601.18723, 2026.

Ximing Lu, Sean Welleck, Jack Hessel, Liwei Jiang, Lianhui Qin, Peter West, Prithviraj Ammanabrolu, and Yejin Choi. Quark: Controllable text generation with reinforced unlearning. In Advances in Neural Information Processing Systems, 2022.

Ajay Mandlekar, Danfei Xu, Josiah Wong, Soroush Nasiriany, Chen Wang, Rohun Kulkarni, Li Fei-Fei, Silvio Savarese, Yuke Zhu, and Roberto Mart´ın-Mart´ın. What matters in learning from offline human demonstrations for robot manipulation. In Conference on Robot Learning, 2022.

Yanbo Mao, Jianlong Fu, Ruoxuan Zhang, Hongxia Xie, and Meibao Yao. Beyond success: Refining elegant robot manipulation from mixed-quality data via just-in-time intervention. In Conference on Computer Vision and Pattern Recognition, 2026.

Yihuan Mao, Chengjie Wu, Xi Chen, Hao Hu, Ji Jiang, Tianze Zhou, Tangjie Lv, Changjie Fan, Zhipeng Hu, Yi Wu, Yujing Hu, and Chongjie Zhang. Stylized offline reinforcement learning: Extracting diverse high-quality behaviors from heterogeneous datasets. In International Conference on Learning Representations, 2024.

Evgenii Nikishin, Max Schwarzer, Pierluca D’Oro, Pierre-Luc Bacon, and Aaron Courville. The primacy bias in deep reinforcement learning. In International Conference on Machine Learning, 2022.

Mathieu Petitbois, Remy Portelas, Sylvain Lamprier, and Ludovic Denoyer. Offline learning of´ controllable diverse behaviors. arXiv preprint arXiv:2504.18160, 2025.

Mathieu Petitbois, Remy Portelas, and Sylvain Lamprier. Offline reinforcement learning of high- ´ quality behaviors under robust style alignment. In International Conference on Machine Learning, 2026.

Marcel Torne, Anubha Mahajan, Abhijnya Bhat, and Chelsea Finn. Freeform preference learning for robotic manipulation. arXiv preprint arXiv:2606.32027, 2026.

Andrew Wagenmaker, Mitsuhiko Nakamoto, Yunchu Zhang, Seohong Park, Waleed Yagoub, Anusha Nagabandi, Abhishek Gupta, and Sergey Levine. Steering your diffusion policy with latent space reinforcement learning. In Conference on Robot Learning, 2025.

Yi Ru Wang, Carter Ung, Christopher Tan, Grant Tannert, Jiafei Duan, Josephine Li, Anh Le, Rishabh Oswal, Markus Grotz, Wilbert Pumacay, Yuquan Deng, Ranjay Krishna, Dieter Fox, and Siddhartha Srinivasa. RoboEval: Where robotic manipulation meets structured and scalable evaluation. arXiv preprint arXiv:2507.00435, 2025.

Hanlin Yang, Jian Yao, Weiming Liu, Qing Wang, Hanmin Qin, Hansheng Kong, Kirk Tang, Jiechao Xiong, Chao Yu, Kai Li, Junliang Xing, Hongwu Chen, Juchao Zhuo, Qiang Fu, Wei Yang, and Haobo Fu. Diverse policies recovering via pointwise mutual information weighted imitation learning. In International Conference on Learning Representations, 2025.

Seonghyeon Ye, Yunhao Ge, Kaiyuan Zheng, Shenyuan Gao, Sihyun Yu, George Kurian, Suneel Indupuru, You Liang Tan, Chuning Zhu, Jiannan Xiang, Ayaan Malik, Kyungmin Lee, William Liang, Nadun Ranawaka, Jiasheng Gu, Yinzhen Xu, Guanzhi Wang, Fengyuan Hu, Avnish Narayan, Johan Bjorck, Jing Wang, Gwanghyun Kim, Dantong Niu, Ruijie Zheng, Yuqi Xie, Jimmy Wu, Qi Wang, Ryan Julian, Danfei Xu, Yilun Du, Yevgen Chebotar, Scott Reed, Jan Kautz, Yuke Zhu, Linxi Fan, and Joel Jang. World action models are zero-shot policies. arXiv preprint arXiv:2602.15922, 2026.

Eric Zhan, Albert Tseng, Yisong Yue, Adith Swaminathan, and Matthew Hausknecht. Learning calibratable policies using programmatic style-consistency. In International Conference on Machine Learning, 2020.

## A QUALITY SCORE FUNCTIONS

Every quality attribute is measured on one complete trajectory, from the initial observation to the step at which the task is completed. Both simulators run at a control rate of 20 Hz, so one step corresponds to 0.05 s. Either direction of an attribute can be the preferred one depending on the task; for each attribute we state below the direction preferred in our experiments. The score $m ( \tau )$ of Section 3.1 is the measurement itself when a higher value is preferred and its reciprocal when a lower value is preferred, as in $m ( \tau ) = 1 / T$ for speed, so that the highest bin $q ^ { \star } = B - 1$ always denotes the preferred end of the attribute; the figures and tables report the raw measurements.

Quantization. For every score, the bin edges are the $1 / B , \ldots , ( B - 1 ) / B$ quantiles of the scores of the successful trajectories in the current dataset, recomputed at every iteration. A trajectory whose score coincides with an edge is assigned to the better bin. For the joint conditioning experiments, each attribute is quantized independently.

Robomimic. Let $T$ be the number of executed actions, $a _ { t } \in \mathbb { R } ^ { 6 | \mathcal { A } | }$ the arm part of the action at step t (the position and rotation deltas of every arm in ${ \mathcal { A } } ,$ gripper dimensions excluded), $p _ { t } ^ { r } \in \mathbb { R } ^ { 3 }$ the end-effector position of arm $r ,$ and $q _ { t } ^ { r }$ its orientation quaternion. Let $\begin{array} { r } { s _ { t } = \frac { 1 } { | \boldsymbol { A } | } \sum _ { r } \| p _ { t + 1 } ^ { r } - p _ { t } ^ { r } \| _ { 2 } } \end{array}$ denote the arm-averaged displacement per step. Square uses one arm and Transport two.

• Speed (steps to success): $T ,$ , in steps. Lower is preferred in the main experiments; Section 4.4 also requests slower execution.

• Action smoothness (normalized action jerk): $\begin{array} { r } { \frac { 1 } { T - 1 } \sum _ { t } \lVert a _ { t + 1 } - a _ { t } \rVert _ { 2 } ^ { 2 } / \frac { 1 } { T } \sum _ { t } \lVert a _ { t } \rVert _ { 2 } ^ { 2 } . } \end{array}$ , dimensionless. Lower is better.

• End-effector smoothness (normalized EEF jerk): $\begin{array} { r } { \frac { 1 } { \left. A \right. \left( T - 2 \right) } \sum _ { r , t } \Vert p _ { t + 2 } ^ { r } - 2 p _ { t + 1 } ^ { r } + } \end{array}$ $\begin{array} { r } { p _ { t } ^ { r } \| _ { 2 } / \frac { 1 } { T - 1 } \sum _ { t } s _ { t } } \end{array}$ , dimensionless. Lower is better.

• End-effector speed: $\begin{array} { r l r } {  { \frac { 1 } { T - 1 } \sum _ { t } s _ { t } . } } \end{array}$ , in m per step. Higher is better.

• Translation path length: $\begin{array} { r } { \frac { 1 } { | \boldsymbol { \mathcal { A } } | } \sum _ { r } \sum _ { t } \| p _ { t + 1 } ^ { r } - p _ { t } ^ { r } \| _ { 2 } } \end{array}$ , in m. Lower is better.

• Rotation path length: $\textstyle \sum _ { t } { \frac { 1 } { | A | } } \sum _ { r } 2$ arccos $| \langle q _ { t + 1 } ^ { r } , q _ { t } ^ { r } \rangle |$ , in rad. Lower is better.

• Grasp timing: the index of the first step with a closing gripper command divided by $T ,$ averaged over arms. Lower is better.

• Lift amplitude: $\begin{array} { r } { \frac { 1 } { \left| \mathcal { A } \right| } \sum _ { r } \left( \operatorname* { m a x } _ { t } z _ { t } ^ { r } - \operatorname* { m i n } _ { t } z _ { t } ^ { r } \right) } \end{array}$ with $\boldsymbol { z } _ { t } ^ { r }$ the height of the end-effector, in m. Section 4.4 requests both a smaller and a larger lift amplitude.

RoboEval. The following scores use the metric implementation of RoboEval (Wang et al., 2025). Let $p _ { t } ^ { r }$ be the wrist position of arm r, $q _ { t } ^ { r }$ its orientation quaternion, and $\theta _ { t } ^ { r } \in \mathbb { R } ^ { 7 }$ its joint configuration.

• Completion time $( \mathbf { C T } , \mathbf { \overbrace { \mathbf { \Lambda } } } ^ { 6 6 } \mathbf { S p e e d } ^ { 3 9 } )$ : the number of control steps until success divided by 20 Hz, in simulated seconds. Lower is better. RoboEval reports wall-clock time; we use simulated time so that the score does not depend on the machine.

• Joint path length (JPL): $\begin{array} { r l } { } & { { } \sum _ { r } \sum _ { t } \| \theta _ { t + 1 } ^ { r } - \theta _ { t } ^ { r } \| _ { 2 } } \end{array}$ , in rad, summed over both arms. Lower is better.

• Cartesian path length (CPL): $\begin{array} { r l } { } & { { } \sum _ { r } \sum _ { t } \| p _ { t + 1 } ^ { r } - p _ { t } ^ { r } \| _ { 2 } } \end{array}$ , in m, summed over both arms. Lower is better.

• Orientation path length $\begin{array} { r } { ( \mathbf { O P L } ) \colon \sum _ { r } \sum _ { t } 2 } \end{array}$ arccos $\langle q _ { t + 1 } ^ { r } , q _ { t } ^ { r } \rangle |$ , in rad, summed over both arms. Lower is better.

• Mean joint jerk (MJJ): the mean over arms and steps of $\| \stackrel { \cdot \because } { \theta } _ { t } ^ { r } \| _ { 2 }$ , with velocity, acceleration, and jerk obtained by successive finite differences with $\Delta t .$ , in $\mathrm { r a d } / \mathrm { s } ^ { 3 }$ . Lower is better.

• Mean Cartesian jerk (MCJ): the same construction on $p _ { t } ^ { r }$ , in $\mathrm { { m } / \mathrm { { s } ^ { 3 } } }$ . Lower is better.

• Bimanual arm velocity difference (BAVD): $\begin{array} { r } { \frac { 1 } { T } \sum _ { t } \lvert \rvert \dot { \theta } _ { t } ^ { L } \rvert \rvert _ { 2 } - \lvert \rvert \dot { \theta } _ { t } ^ { R } \rvert \rvert _ { 2 } } \end{array}$ using the simulator joint velocities, in rad/s. Lower is better.

• Bimanual gripper vertical difference (BGVD): $\begin{array} { r } { \frac { 1 } { T } \sum _ { t } | z _ { t } ^ { L } - z _ { t } ^ { R } | } \end{array}$ with z the wrist height, in m. Lower is better.

Real robot. Let $\mathbf { p } _ { t }$ be the end-effector position at timestep $t , \mathbf { z } _ { t } ^ { \mathrm { e e f } }$ the unit z-axis of the end-effector frame, $\mathbf { z } ^ { \mathrm { w o r l d } }$ the unit z-axis of the world frame, and T the episode length.

• End-effector speed (conveyor work):

$$
{ \bar { v } } = { \frac { 1 } { T } } \sum _ { t = 1 } ^ { T } \| { \dot { \mathbf { p } } } _ { t } \| _ { 2 } ,\tag{4}
$$

in mm/s, computed over successful grasp segments. Higher is better.

• Tilt angle (wine serving):

$$
\bar { \theta } = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \left| \operatorname { a r c c o s } \left( \mathbf { z } _ { t } ^ { \mathrm { e e f } } \cdot \mathbf { z } ^ { \mathrm { w o r l d } } \right) \right| ,\tag{5}
$$

in rad, computed over successful episodes. Since the tray is horizontal when it is rigidly held by an end-effector aligned with $\mathbf { z } ^ { \mathrm { w o r l d } }$ (Figure 7a), <sup>¯</sup>θ directly reflects the uprightness of the wine glasses. Lower is better.

## B TRAINING AND EVALUATION DETAILS

Policy and optimization. All simulation policies share one architecture and optimizer. The velocity network is an MLP with three layers of width 512 (expansion factor 1, no dropout) and a 128-dimensional flow-time embedding; a two-layer MLP encodes the observation history of $L = 2$ steps. The quality label is mapped to a learned embedding and concatenated with the encoded observation before the velocity network. Policies are trained from scratch at every iteration with AdamW (learning rate $1 0 ^ { - 4 }$ , weight decay $1 0 ^ { - 5 }$ , gradient-norm clipping at 10), a cosine learning-rate schedule over the training steps, batch size 1024, and an exponential moving average of the weights with rate 0.995. Actions are sampled with nine Euler steps of the learned velocity field. Conditioned on an observation history of $L = 2$ steps, the policy predicts a chunk of H actions that starts at the first observation of the history; the $L - 1$ actions aligned with past observations are discarded, the next $H - L$ actions are executed, and the last one is dropped before replanning. Both choices follow the original implementation of Diffusion Policy (Chi et al., 2023). The quality bins are recomputed as adaptive quartiles of the successful trajectories in the current dataset at every iteration, and rollouts are collected with the best bin requested. Table 2 lists the settings that differ between experiments.

Table 2: Per-experiment settings. Steps are gradient steps per iteration. “Number of iterations” counts iterations including iteration 0, which is trained on the demonstrations alone. The action chunk is written as chunk length / executed steps.
<table><tr><td rowspan="2"></td><td colspan="3">Main results (Figs. 2, 4, Tab. 1)</td><td rowspan="2">Fig. 3 Square</td><td rowspan="2">Fig. 5 Square</td><td rowspan="2">Fig. 6</td><td rowspan="2">Fig. 8 RoboEval Square</td><td rowspan="2">Fig. 9 Square</td></tr><tr><td>Square</td><td>Transport</td><td>RoboEval</td></tr><tr><td>Number of iterations</td><td>6</td><td>6</td><td>6</td><td>10</td><td>10</td><td>7</td><td>6</td><td>7</td></tr><tr><td>Steps per iteration</td><td>100k</td><td>100k</td><td>30k</td><td>30k</td><td>30k</td><td>30k</td><td>30k</td><td>30k</td></tr><tr><td>Episode limit (steps)</td><td>400</td><td>700</td><td>400</td><td>400</td><td>800</td><td>400</td><td>800</td><td>400</td></tr><tr><td>Action chunk</td><td>6/4</td><td>10/8</td><td>10/8</td><td>10/8</td><td>10/8</td><td>10/8</td><td>10/8</td><td>10/8</td></tr></table>

Reported statistics. Success rates are computed over all rollouts of an evaluation batch. Quality is computed as the mean over the successful rollouts of the batch, first within each seed and then averaged across seeds with equal weight.

Real robot experiments. We use a Franka Research 3 arm with the DROID hardware and camera setup (Khazatsky et al., 2024). The policy is $\pi _ { 0 . 5 }$ (Black et al., 2025) fine-tuned with its original recipe for 10 epochs with batch size 32 at every iteration, starting from the pretrained checkpoint each time; the quality label is inserted into the language prompt as text. For wine serving, the initial dataset contains the 50 successful episodes among 382 teleoperated attempts; each iteration collects 50 rollouts, and success requires that the tray reaches the target table with no glass fallen. For conveyor work, the initial dataset contains 400 successful grasp segments from 936 teleoperated segments; each iteration collects 20 rollouts of 20 arriving cans, that is 400 segments, with cans arriving at one can every three seconds on a belt of fixed speed. Throughput is the success fraction multiplied by the arrival rate of 20 cans per minute. Speed is measured on successful segments and tilt on successful episodes. The quality label requested at collection is the best bin, $\mathbf { B C } _ { \mathrm { f i l t e r } }$ retrains on the best 20% of successful trajectories along the target axis at every iteration, and BC denotes $\pi _ { 0 . 5 }$ fine-tuned on the initial dataset without a quality label.

## C BASELINE DETAILS

The BC baselines, SORL, and SCIQL use the network, optimizer, training steps, rollout budget, retraining from scratch, success filter, and seeds of QCP (Appendix B); they differ only in how the training set is formed and in the learning objective. DSRL and EXPO are described separately below.

$\mathbf { B C _ { a c c } } . \quad \mathbf { B C _ { \mathrm { { a c c } } } }$ runs the loop of Algorithm 1 without a quality label: rollouts are collected by the unconditioned policy, successful ones are appended to the dataset, and a plain behavior-cloning policy is retrained on the accumulated set.

$\mathbf { B C _ { f i l t e r } . B C _ { f i l t e r } }$ additionally ranks the successful trajectories of the accumulated set (demonstrations included) by the target score at every iteration and keeps the best 10%. At every iteration the filter is applied again to the whole accumulated set, not to the previously filtered set, so the filtered training set grows with the accumulated data rather than shrinking over iterations; the setting does not force the dataset to become smaller. On Robomimic at least 20 trajectories are kept, and on RoboEval the filter is implemented as a sampling weight of 0 or 1 without a floor. In the real robot experiments the filter keeps the best 20%, because the initial datasets are small (50 episodes for wine serving) and a 10% filter would leave too few trajectories to train on.

SORL and SCIQL. Both methods proceed in two stages at every iteration. In the first stage, a value function is learned with implicit Q-learning (Kostrikov et al., 2022) on a chunk-level MDP whose state is the two-step observation history, whose action is the executed action chunk, and whose discount is 0.99 per decision; the task reward is 1 on the chunk in which the task succeeds and 0 otherwise. The value and twin Q networks are two-layer MLPs of width 256 with layer normalization, trained for 60k steps with Adam (learning rate $3 \times 1 0 ^ { - 4 }$ , cosine schedule, batch 1024), expectile 0.7, and Polyak averaging with rate 0.005. Advantages are divided by the 99th percentile of their magnitude, and the resulting weights exp(βA) are clipped at 100.

In the second stage, the actor is trained. It is our quality-conditioned flow policy with the same network and the same number of gradient steps as QCP (100k on Robomimic, 30k on RoboEval), and the weights from the first stage are used as sampling probabilities over the training samples. The two methods differ in how the weights are formed. SORL (Mao et al., 2024) trains on the samples whose label equals the requested one, weighted by exp(βA<sup>r</sup>) with $A ^ { r }$ the task advantage. SCIQL (Petitbois et al., 2026) relabels every trajectory under each of the four labels, learns a style advantage $A ^ { \lambda }$ from the reward $\mathbf { 1 } \{ z = q ( \tau ) \}$ , and weights samples by $\exp ( \beta \xi )$ with $\xi = A ^ { \dot { \lambda } } +$ $\sigma ( A ^ { \lambda } ) \bar { A ^ { r } }$ , as in the original paper: the style advantage is the primary term, and the task advantage counts only for actions that already match the requested style. Both stages are retrained from scratch at every iteration on the accumulated success-filtered set, and rollouts are collected with the best label requested. The main results use $\beta = 3 ;$ Appendix D reports the other values of $\beta$ and explains this choice.

DSRL and EXPO. Both methods start from a frozen flow matching BC policy trained on the demonstrations for 100k steps with the network of Appendix B and sampled with nine Euler steps. DSRL (Wagenmaker et al., 2025) learns a soft actor-critic policy over the initial noise of that policy (update-to-data ratio 20, discount 0.999, batch 256, learning rate $3 \times 1 0 ^ { - 4 }$ , twin critics of three layers of width 2048 with layer normalization). EXPO (Dong et al., 2026) learns an edit policy and a critic ensemble on top of the frozen policy (eight candidate actions per decision, edit scale 0.1, update-to-data ratio 5, discount 0.99, batch 256, an ensemble of ten critics with the minimum over two), with half of every batch drawn from the frozen policy’s own 300 initial rollouts. Both act on chunks of 4 actions on Square and 8 elsewhere.

The reward has two parts, a task reward and a quality reward:

$$
r _ { t } = \mathbf { 1 } \{ { \mathrm { t a s k ~ s u c c e e d s ~ a t ~ s t e p ~ } } t \} + \lambda r _ { t } ^ { \mathrm { q } } .\tag{6}
$$

The task reward is 1 at the step where the task succeeds, and the episode ends there. The quality reward $r _ { t } ^ { \mathrm { q } }$ is a small penalty given at every step, and λ sets how much it counts relative to the task reward.

For speed, the penalty is the same at every step, $r _ { t } ^ { \mathrm { q } } = - 1 / L _ { \mathrm { r e f } } ,$ where $L _ { \mathrm { r e f } }$ is the average length of the demonstrations. Summed over an episode, the quality reward equals minus the episode length divided by $L _ { \mathrm { r e f } } .$ , so a shorter episode earns a higher return and an episode as long as a typical demonstration loses about 1.

For smoothness and joint path length, the penalty at step t depends on how much the motion at that step exceeds what is typical in the demonstrations:

$$
r _ { t } ^ { \mathrm { q } } = - \frac { 1 } { L _ { \mathrm { r e f } } } \Big ( \mathrm { c l i p } \big ( \frac { j _ { t } - J _ { \mathrm { r e f } } } { J _ { \mathrm { r e f } } } , - 1 , 5 \big ) - \bar { c } \Big ) ,\tag{7}
$$

where $j _ { t }$ is the per-step motion measure $( \lVert a _ { t } - a _ { t - 1 } \rVert ^ { 2 }$ over the arm dimensions on Robomimic, the joint-space increment on RoboEval), $J _ { \mathrm { r e f } }$ is its average over the demonstrations, the clipping keeps a single very jerky step from dominating the return, and c¯ is the average of the clipped term over the demonstrations, so that a demonstration-like step receives zero penalty on average. Dividing by $L _ { \mathrm { r e f } }$ puts the return on the same scale as the speed reward.

All constants are computed once from the successful demonstrations of each task and then kept fixed: $L _ { \mathrm { r e f } } = 2 4 0 , J _ { \mathrm { r e f } } = 0 . 0 3 5 4 , \bar { c } = - 0 . 2 5 3$ on Square; $L _ { \mathrm { r e f } } = 6 3 0 , J _ { \mathrm { r e f } } = 0 . 0 6 6 9 , \bar { c } = - 0$ .145 on Transport; $L _ { \mathrm { r e f } } = 1 3 6 , J _ { \mathrm { r e f } } = 0 . 1 4 1 , { \bar { c } } = 0$ on Pack Box; $L _ { \mathrm { r e f } } = 7 1 , J _ { \mathrm { r e f } } = 0 . 1 3 7 , \bar { c } = 0$ on Pick Book. DSRL multiplies the whole reward by $L _ { \mathrm { r e f } } / \operatorname* { m a x } ( 1 , \lambda )$ so that the size of its returns stays similar across values of λ. Table 1 uses $\lambda = 4 ; \mathtt { A }$ ppendix D reports the other values.

The policy is evaluated on 100 episodes every 120k environment steps, which is the number of steps of 300 rollouts at the 400-step limit of Square and RoboEval, and the paper reports the sixth evaluation, that is 720k environment steps after the prefill. Transport uses the same interval.

## D EFFECT OF THE QUALITY COEFFICIENT

The RL baselines have one coefficient that sets how strongly the quality signal enters training: the advantage temperature $\beta$ of SORL and SCIQL and the quality reward weight λ of DSRL and EXPO. To choose the values used in the main results, we swept these coefficients on Robomimic Square for both attributes with the settings of Appendix C, three seeds per setting, at the same endpoint as Table 1 (Table 3). For each method we selected the setting that gives a good balance of success rate and quality, $\beta = 3$ and λ = 4, and used it for the main results on all tasks.

Table 3: Coefficient sweep on Robomimic Square. Each cell is success rate (%) / quality of successful episodes (steps to success for Speed, action jerk for Smoothness; lower is better), mean ± standard deviation over three seeds. Left: SORL and SCIQL over the advantage temperature β. Right: DSRL and EXPO over the quality reward weight λ. Bold rows are the settings used in the main results $( \beta = 3 , \lambda = 4 )$ and reproduce Table 1.
<table><tr><td rowspan="2">β</td><td colspan="2">SORL</td><td colspan="2">SCIQL</td></tr><tr><td>Speed</td><td>Smoothness</td><td>Speed</td><td>Smoothness</td></tr><tr><td>0</td><td>34.0±3.5/ 213.8±17.0</td><td>36.7±5.3 / 0.0118±0.0010</td><td>77.1±9.0/ 251.2±25.6</td><td>72.3±6.0/ 0.0205±0.0017</td></tr><tr><td>1</td><td>39.0±6.0/ 206.5±18.8</td><td>40.3±8.3 / 0.0114±0.0012</td><td>78.0±8.71 231.0±26.7</td><td>79.3±6.2/ 0.0187±0.0018</td></tr><tr><td>2</td><td>45.0±9.5/ 189.4±9.8</td><td>51.3±3.7/ 0.0126±0.0007</td><td>74.7±4.3/ 218.9±18.6</td><td>79.7±9.8/ 0.0176±0.0018</td></tr><tr><td>3</td><td>53.2±7.8/ 156.0±18.1</td><td>48.4±6.5 / 0.0115±0.0010</td><td>68.7±5.2/ 206.5±23.2</td><td>83.0±6.0 / 0.0128±0.0007</td></tr><tr><td>4</td><td>41.7±5.7/ 188.1±18.0</td><td>43.3±4.9/ 0.0142±0.0008</td><td>70.7±8.4/ 212.5±14.6</td><td>85.0±4.7/ 0.0185±0.0022</td></tr><tr><td>10</td><td>48.7±5.5/ 194.2±23.3</td><td>44.7±3.71 0.0212±0.0017</td><td>63.7±7.71 208.2±19.0</td><td>59.2±6.0/ 0.0163±0.0012</td></tr><tr><td>16</td><td>47.3±7.4/ 159.5±18.0</td><td>44.7±4.6/ 0.0201±0.0015</td><td>52.3±5.1/ 195.3±13.8</td><td>61.0±9.4/ 0.0177±0.0014</td></tr></table>

<table><tr><td rowspan="2">λ</td><td colspan="2">DSRL</td><td colspan="2">EXPO</td></tr><tr><td>Speed</td><td>Smoothness</td><td>Speed</td><td>Smoothness</td></tr><tr><td rowspan="2">0</td><td>30.0±8.2/</td><td>42.0±5.0/</td><td>92.0±8.6/</td><td>96.0±7.6/</td></tr><tr><td>196.2±15.6</td><td>0.0463±0.0055</td><td>161.8±16.4</td><td>0.1183±0.0116</td></tr><tr><td rowspan="2">1</td><td>25.3±8.0/</td><td>34.0±4.8 /</td><td>90.0±6.6/</td><td>94.0±5.9/</td></tr><tr><td>192.2±17.4</td><td>0.0455±0.0039</td><td>164.2±13.8</td><td>0.1428±0.0159</td></tr><tr><td rowspan="2">4</td><td>37.0±4.2 /</td><td>43.0±8.1 /</td><td>96.0±4.1 /</td><td>79.0±10.0 /</td></tr><tr><td>208.8±22.4</td><td>0.0469±0.0044</td><td>162.5±10.7</td><td>0.1574±0.0147</td></tr><tr><td rowspan="2">16</td><td>52.0±6.9 /</td><td>49.0±3.4/</td><td>97.0±5.4/</td><td>68.0±7.4/</td></tr><tr><td>224.4±20.1</td><td>0.0510±0.0034</td><td>163.8±14.6</td><td>0.1631±0.0176</td></tr></table>

## E MULTI-QUALITY CONDITIONING EXPERIMENT DETAILS

Two-attribute control on Square. Figure 5 uses Square with an 800-step episode limit so that slow executions can complete, ten policies, and three seeds. Each trajectory receives two labels, a speed quartile (steps to success) and a lift-amplitude quartile (Appendix A), both recomputed on the current dataset at every iteration. Each label is mapped through its own embedding and the two embeddings are concatenated with the encoded observation. A separate lineage is trained for each of the four corner commands, namely the lowest or highest speed quartile combined with the lowest or highest lift-amplitude quartile, and every lineage collects its rollouts under its own command. The dashed reference lines in Figure 5 are the extreme iteration means of single-label lineages run with the identical protocol: fastest and slowest speed commands, and lowest and highest lift-amplitude commands. The joint commands reach the two single-label references on both coordinates within a few percent, at a lower success rate for the slow commands.

Six-attribute improvement on RoboEval. Figure 6 uses the state-based policy of Table 2 on the Lift Pot, Stack Two Blocks, Handover, and Valve tasks of RoboEval with the demonstrations of the fixed-position variant, seven policies, 300 rollouts per iteration, and three seeds. Each trajectory receives six quartile labels, one per attribute, each mapped through its own embedding; all six labels are requested at their best bin during collection. The six attributes are chosen per task from the eight RoboEval metrics of Appendix A by dropping the two that a preliminary eight-attribute run showed to be unaffected by conditioning or to be in conflict with the task (Table 4). The reference policies are the unconditioned BC policy trained on the demonstrations (initial BC) and the unconditioned loop that accumulates its successful rollouts for six iterations $\mathrm { ( B C _ { a c c } ) }$ , both with the same trainer, demonstrations, and seeds. Per-seed values are means over successful rollouts of the evaluation batch and are averaged across seeds. Table 5 lists the values behind the radar plot.

## F ANALYSIS EXPERIMENT DETAILS

Rollout collection comparison (Section 5.1). The comparison uses the Square smoothness attribute with 30k training steps per iteration, an 800-step episode limit, and three seeds. The frozen variant draws 1,800 rollouts per seed from the checkpoint trained at iteration 0, conditioned on the best bin $q ^ { \star } = 3 ,$ , with the dataset normalization that checkpoint was trained under, and without any further training. The iterative variant is the loop itself, which draws 300 rollouts per seed at each of iterations 0 to 5 under the same best-bin request, so both variants spend 1,800 rollouts per seed. The purpose of this comparison is to separate the effect of the quality-conditioned architecture from the effect of the iterative hindsight relabeling loop. The frozen variant has the architecture but no further loop iterations, so any additional shift of the iterative variant comes from the loop rather than from conditioning or sampling alone. Statistics are computed over successful rollouts pooled across seeds; per-seed medians agree with the pooled values to within 0.002. The frozen policy has a success rate of 54.5% and a median action jerk of 0.0335 (interquartile range 0.028 to 0.043) over 2,943 successful rollouts, against 57.0% and 0.0324 at iteration 0 of the loop. By iteration 5 the loop reaches 73.9% and a median of 0.0098 (interquartile range 0.0088 to 0.0109). The smoothest frozen rollout has a jerk of 0.0171, above the 75th percentile of the iteration-5 distribution. The later iterations therefore produce behavior that the iteration-0 policy does not generate at all, not merely behavior that it generates rarely: the quality-conditioned architecture alone does not reach this range, and the additional shift comes from the iterative hindsight relabeling loop.

Table 4: Conditioned attributes per task. The two dropped attributes and the reason.
<table><tr><td>Task</td><td>Conditioned</td><td>Dropped (reason)</td></tr><tr><td>Lift Pot</td><td>JPL, OPL, CT, MJJ, BAVD, BGVD</td><td>CPL (a fixed vertical lift leaves little Cartesian path to vary), MCJ (redundant with MJJ)</td></tr><tr><td>Stack</td><td>CPL, JPL, CT, OPL, MJJ, MCJ</td><td>BAVD (one-arm task, equal arm velocities are not a quality), BGVD (large by design of the task)</td></tr><tr><td>Handover</td><td>CPL, OPL, MJJ, MCJ, BAVD, BGVD</td><td>CT (in conflict with the smoothness attributes), JPL (redundant with CPL and OPL)</td></tr><tr><td>Valve</td><td>JPL, OPL, CT, MJJ, BAVD, BGVD</td><td>MCJ (nearly constant for a wrist rotation), CPL (path fixed by the valve geometry)</td></tr></table>

Table 5: Six-attribute quality on RoboEval. Numerical values of Figure 6: success rate and the six conditioned attributes at iteration 6 for QCP and $\mathrm { { B C } _ { \mathrm { { a c c } } } , }$ and for the initial BC. Mean ± standard deviation over seeds; lower is better for every attribute; units as in Appendix A.
<table><tr><td>Lift Pot</td><td>SR</td><td>JPL</td><td>OPL</td><td>CT</td><td>MJJ</td><td>BAVD</td><td>BGVD</td></tr><tr><td>Initial BC</td><td>0.92±0.01</td><td>4.37±0.06</td><td> $3 . 6 5 { \pm } 0 . 0 4$ </td><td> $2 . 0 4 \pm 0 . 0 2$ </td><td>228.8±2.6</td><td>0.418±0.004</td><td>0.0052±0.0003</td></tr><tr><td> $\mathrm { { B C } _ { \mathrm { { a c c } } } }$ </td><td> $0 . 9 9 { \pm } 0 . 0 0 \ $ </td><td> $3 . 5 9 { \pm } 0 . 0 7$ </td><td> $3 . 0 8 \pm 0 . 0 6$ </td><td> $1 . 9 5 { \pm } 0 . 0 3$ </td><td> $1 7 6 . 6 { \pm } 2 . 6 $ </td><td>0.278±0.015</td><td> $0 . 0 0 2 7 { \scriptstyle \pm 0 . 0 0 0 3 }$ </td></tr><tr><td>QCP</td><td>1.00±0.00</td><td>2.82±0.09</td><td> $1 . 9 8 { \pm } 0 . 0 9$ </td><td> $1 . 7 2 { \pm } 0 . 0 0$ </td><td> $1 3 0 . 5 { \pm } 6 . 4 $ </td><td>0.173±0.019</td><td>0.0022±0.0002</td></tr><tr><td>Stack</td><td>SR</td><td>CPL</td><td>JPL</td><td>CT</td><td>OPL</td><td>MJJ</td><td>MCJ</td></tr><tr><td>Initial BC</td><td>0.46±0.02</td><td>1.98±0.03</td><td>8.16±0.24</td><td>6.64±0.05</td><td>5.72±0.11</td><td> $1 5 3 . 2 { \pm } 3 . 0 $ </td><td>39.2±0.6</td></tr><tr><td> $\mathrm { { B C } _ { \mathrm { { a c c } } } }$ </td><td>0.46±0.02</td><td>1.68±0.05</td><td>7.59±0.22</td><td>8.62±0.86</td><td>5.05±0.09</td><td> $8 0 . 4 \pm 6 . 7$ </td><td> $1 9 . 2 \pm 1 . 2$ </td></tr><tr><td>QCP</td><td>0.79±0.21</td><td>1.33±0.02</td><td>3.63±0.06</td><td>5.85±0.03</td><td>2.75±0.12</td><td> $5 8 . 5 { \pm } 4 . 1 $ </td><td> $1 5 . 4 \pm 0 . 9$ </td></tr><tr><td>Handover</td><td>SR</td><td>CPL</td><td>OPL</td><td>MJJ</td><td>MCJ</td><td>BAVD</td><td>BGVD</td></tr><tr><td>Initial BC</td><td>0.96±0.00</td><td> $1 . 1 0 { \pm } 0 . 0 9$ </td><td> $2 . 2 4 \pm 0 . 2 9$ </td><td> $1 6 9 . 6 { \pm } 1 0 . 6 $ </td><td> $4 8 . 1 \pm 3 . 2 $ </td><td>0.293±0.004</td><td>0.0253±0.0008</td></tr><tr><td> $\mathrm { { B C } _ { \mathrm { { a c c } } } }$ </td><td> $0 . 9 5 { \pm } 0 . 0 4$ </td><td> $0 . 7 7 { \pm } 0 . 0 5$ </td><td> $1 . 1 3 { \pm } 0 . 1 5$ </td><td> $6 7 . 1 { \pm } 0 . 8 $ </td><td>19.4±0.8</td><td>0.215±0.004</td><td>0.0236±0.0010</td></tr><tr><td> $\mathrm { Q C P }$ </td><td>1.00±0.00</td><td> $0 . 7 3 { \pm } 0 . 0 1$ </td><td> $0 . 8 2 { \pm } 0 . 0 5$ </td><td> $3 6 . 2 \pm 1 . 7$ </td><td> $1 1 . 5 { \pm } 0 . 5 $ </td><td>0.127±0.013</td><td>0.0172±0.0032</td></tr><tr><td>Valve</td><td>SR</td><td>JPL</td><td>OPL</td><td>CT</td><td>MJJ</td><td>BAVD</td><td>BGVD</td></tr><tr><td>Initial BC</td><td>1.00±0.00</td><td>1.98±0.11</td><td> $1 . 6 8 \pm 0 . 0 8$ </td><td> $2 . 0 3 { \pm } 0 . 0 8$ </td><td> $6 2 . 8 { \pm } 0 . 4 $ </td><td>0.486±0.023</td><td>0.0094±0.0004</td></tr><tr><td> $\mathrm { { B C } _ { \mathrm { { a c c } } } }$ </td><td>1.00±0.00</td><td> $1 . 6 0 { \pm } 0 . 0 2 \ $ </td><td> $1 . 3 7 { \pm } 0 . 0 2 $ </td><td> $1 . 7 1 { \pm } 0 . 0 2 $ </td><td> $5 6 . 0 { \pm } 3 . 4 $ </td><td>0.413±0.030</td><td>0.0096±0.0003</td></tr><tr><td>QCP</td><td> $1 . 0 0 { \pm } 0 . 0 0 \ \^$ </td><td> $1 . 4 6 { \pm } 0 . 0 3$ </td><td> $1 . 2 8 { \pm } 0 . 0 2$ </td><td> $1 . 4 4 { \pm } 0 . 1 3$ </td><td> $4 4 . 1 { \pm } 1 . 8 $ </td><td>0.225±0.017</td><td>0.0048±0.0022</td></tr></table>

![](images/d080d44953a87057760e0a376ebafc28bb31b390b28967c75cc81f8f3698fd04.jpg)

![](images/9c4b5224aa0f0da413a616e765f93f0ef4bcee0ec4caad452549d136111ee727.jpg)  
Trajectory (QCP) Segment Hybrid 10% filtered BC  
Figure 10: Quality labels for shorter segments. Success rate versus mean end-effector (EEF) displacement per step over successful rollouts on Square and Transport. We compare complete trajectory labels (QCP), segment labels, hybrid labels, and 10% filtered BC over iterations 0–9; darker points indicate later iterations. Transport 10% filtered BC has no successful rollouts, so its quality values are undefined and omitted.

Quality representation (Section 5.2). This experiment examines how the quality condition should be represented: as a quantized label, and if so with how many bins, or as a continuous score. Figure 9 compares six variants on the Square speed attribute, all run with the same loop (30k training steps and 300 rollouts per iteration, seven policies, three seeds). Four variants use quantized labels with B = 4, 8, 16, or 32 equal-frequency bins, recomputed at every iteration, and always request the best bin; with more bins the requested bin covers a smaller share (1/B) of the training data. Two variants use a continuous score instead of a label: the score is normalized so that the best and worst initial demonstrations map to 0 and 1, this normalization is fixed for the whole loop, and the requested value is set once to the 25th or the 12.5th percentile of the initial demonstrations. Curves are means over seeds and bands are one standard deviation. At iteration 6, the four-, eight-, sixteen-, and thirty-two-bin variants reach 70.3%, 63.2%, 57.7%, and 28.1% success with 139, 120, 116, and 112 steps; the two continuous variants reach 55.0% and 67.0% success with 246 and 209 steps.

More bins make executions slightly faster but cost success, sharply so beyond sixteen bins, and neither continuous target matches the bins in speed at any iteration; in success, four bins exceed both continuous targets from iteration 1 onward. Because the continuous variants also use a different way of selecting the requested value, this experiment supports quartile labels with per-iteration recomputation as a reliable default rather than ruling out continuous conditioning in general.

## G QUALITY LABELS FOR SHORTER SEGMENTS

This experiment tests whether a quality label assigned to shorter segments of a trajectory is a better training signal than one label for the complete trajectory. We compare three labeling options. Trajectory is QCP as used in the main experiments: one label per trajectory. Segment assigns a label to every decision step from the score measured in a window of ±8 steps around it. Hybrid conditions on both the trajectory label and the segment label through two embeddings. A segment score depends strongly on the stage of the task (for example, approaching versus inserting), so for the segment labels the quartile boundaries are computed separately within each of ten intervals of normalized trajectory time, and the best quartile is requested at every step. All options run the same loop otherwise.

Figure 10 shows the comparison on the end-effector speed attribute for Square and Transport, with 10% filtered BC as a reference, over ten iterations with 100k training steps and 300 rollouts per iteration and one seed. At iteration 9 on Square, the trajectory label reaches 84.3% success at 6.08 mm per step, the segment label 48.0% at 4.80, and the hybrid 71.0% at 5.07; on Transport, the trajectory label reaches 13.3% success against 3.7% and 4.0% for the segment and hybrid labels, at similar speeds (5.2 to 5.7 mm per step). The trajectory label therefore gives the largest gains in both success and speed on Square and the highest success on Transport. A segment label provides a more local signal, but the signal reflects the stage of the task as much as the quality of execution even after the stage-wise normalization, and the policy can improve a local score without making progress toward completion. These results support labeling complete trajectories; using finer quality signals may require methods that separate quality from task progress, which we leave to future work.