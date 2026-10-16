# Progress Reward Model for Reinforcement Learning via Large Language Models

Xiuhui Zhang, Ning Gao, Xingyu Jiang, Yihui Chen, Yuheng Pan, Mohan Zhang, Yue Deng

Beihang University 37 Xueyuan Road, Haidian District, Beijing   
{zhangxiuhui, gaoning\_ai, jxy33zrhd}@buaa.edu.cn   
{21375399, fzxl2003, zmh666, ydeng}@buaa.edu.cn

## Abstract

Traditional reinforcement learning (RL) algorithms face significant limitations in handling long-term tasks with sparse rewards. Recent advancements have leveraged large language models (LLMs) to enhance RL by utilizing their world knowledge for task planning and reward generation. However, planning-based approaches often depend on pre-defined skill libraries and fail to optimize low-level control policies, while reward-based methods require extensive human feedback or exhaustive searching due to the complexity of tasks. In this paper, we propose the Progress Reward Model for RL (PRM4RL), a novel framework that integrates task planning and dense reward to enhance RL. For high-level planning, a complex task is decomposed into a series of simple manageable subtasks, with a subtask-oriented, fine-grained progress function designed to monitor task execution progress. For low-level reward generation, inspired by potential-based reward shaping, we use the progress function to construct a Progress Reward Model (PRM), providing theoretically grounded optimality and convergence guarantees, thereby enabling effective policy optimization. Experimental results on robotics control tasks demonstrate that our approach outperforms both LLM-based planning and reward methods, achieving state-of-the-art performance. Code is available at https://github.com/deng-ai-lab/PRM4RL.

## 1 Introduction

Reinforcement learning (RL) has shown exceptional potential in solving sequential decision-making problems. However, in real-world scenarios involving complex, long-term tasks with sparse rewards, agents face significant challenges in planning and decision-making due to insufficient feedback signals[1, 2]. Traditional RL frameworks address these challenges primarily by focusing on two strategies: task decomposition and reward shaping. Hierarchical RL[3, 4] simplifies decision-making by autonomously decomposing complex tasks into hierarchical subtasks, while reward shaping techniques provide dense training signals through the reconstruction of reward functions, such as inverse RL[5, 6, 7] and intrinsic reward mechanisms[8]. Despite their advantages, both approaches heavily rely on expert data for priors[9] or supervision[10], and the large-scale acquisition of such data is often challenging, limiting the scalability and applicability of these methods.

Recent advancements in large language model (LLM) research have made significant progress in areas such as dialogue[11, 12], sequential decision-making[13, 14], and coding[15, 16, 17]. By leveraging the extensive and diverse human knowledge acquired during the pre-training phase, LLMs possess a broad range of world knowledge and emergent reasoning abilities[18, 19]. Consequently, LLM-Augmented RL has emerged as a promising approach to overcome the challenges faced by traditional RL methods[20]. To tackle the issues of long-term planning and sparse rewards, previous LLM-based approaches have evolved into two main paradigms: LLM as a planner[21, 22, 23] and LLM as a rewarder[24, 25, 26], as illustrated in Figure 1. As a planner, LLM decomposes complex, long-term tasks into a series of short-term, goal-oriented subtasks, incorporating subtask priors into the decision-making process. As a rewarder, LLM constructs a dense reward function for sparsereward tasks, providing guidance to the underlying policy based on the meaningful priors encoded within the LLM.

![](images/3c66031dfc70fbb67bee20026adc388b56e171fb6e0fa0eeda8a00e621edf144.jpg)  
Figure 1: Quick view. (a), (b) and (c) shows the structual similarities and differences between LLM-Augmented methods. (d) is the average evaluation results on Metaworld[27] and Maniskill[28].

However, LLM as a planner primarily focuses on high-level planning. In the absence of LLMaugmented dense reward guidance, task decomposition alone—relying solely on natural-language task priors—is insufficient to support the effective learning of subtasks[29, 22]. Previous approaches have alleviated this limitation by constructing pre-trained policy libraries[21] or incorporating traditional motion planning to link subtasks[22], yet these methods lack flexibility and efficiency due to their non-end-to-end nature. On the other hand, LLM as a rewarder provides low-level guidance. However, without subtask decomposition, constructing a reward function for an entire long-term, complex task remains a significant challenge for LLMs[30, 31]. Previous methods have enhanced the quality of LLM-generated reward functions by incorporating human feedback[24] or evolutionary search algorithms[25, 26], but these approaches still require extensive search and significant human expert assistance. As shown in Figures 1a and 1b, both of these approaches focus on a single aspect of RL, neglecting the importance of integrating both high-level planning and low-level reward shaping.

To address the challenges of long-term planning and sparse rewards in reinforcement learning more efficiently, we propose a novel framework, Progress Reward Model for Reinforcement Learning (PRM4RL), which integrates both high-level long-term planning and low-level dense rewards to provide a comprehensive solution, as illustrated in Figure 1c. In high-level planning, the task is decomposed into subtasks, each with termination conditions and fine-grained progress evaluations. For low-level reward generation, we utilize the progress to construct an efficient dense reward signal with theoretically grounded optimality and convergence guarantees. In practice, our pipeline decomposes a complex task into a series of manageable, goal-oriented subtasks, each associated with a natural language subtask prior. For each subtask, we build a fine-grained progress function that dynamically tracks the execution of the task. Inspired by the potential field theorem, we regard the progress function as a potential function and design a Progress Reward Model with optimality and convergence guarantees. The evaluation results in Metaworld[27] and Maniskill[28] as shown in Figure 1d demonstrate the effectiveness of our proposed framework, with significant improvements in success rate and reduced LLM calls, outperforming existing methods in both performance and efficiency.

## 2 Related Works

## 2.1 LLM for Planning

Large language models (LLMs), enriched with diverse world knowledge and human priors during their pretraining phase, possess the ability to understanding and reasoning[18, 19, 17, 16, 11]. When provided with relevant task information, LLMs are able to decompose complex, long-term tasks into simpler, short-term tasks, each focused on specific objectives[32, 33]. Numerous approaches have leveraged Task and Motion Planning (TAMP) to realize such task decomposition in robotics scenes[34, 35]. However, when using RL for low-level control strategies, a significant gap arises between high-level planning and low-level control—how to efficiently leverage the natural language plans generated by LLMs to guide policy training? SayCan[21] addresses this challenge by utilizing a set of pretrained policies, where the LLM simply selects the appropriate policy based on the plans. Plan-Seq-Learn[22] uses traditional motion planning to link subtasks, alleviating the burden on RL policies. In text-based environments, ELLM[23] trains RL policies by describing state transition in natural language and computing the similarity between the executed actions and the plan, using this similarity as a reward signal. However, this approach is computationally expensive and not suitable for robotics tasks. In contrast, our method constructs an efficient and dense reward signal through the Progress Reward Model (PRM), effectively bridging the gap between high-level planning and low-level policy training.

## 2.2 LLM for Reward Generation

Reward functions are the primary mechanism to guide RL training[36, 37], leading to numerous approaches that utilize LLMs to generate executable reward functions that provide dense rewards. Text2Reward[24] uses LLMs to generate executable python functions based on a goal described in natural language, with human involvement in the loop. Eureka[25] extends this paradigm by introducing evolutionary search and automatic reward refinement through LLMs. REvolve[26] translates implicit human knowledge into explicit reward functions for training RL with human feedback. However, these methods treat the task as a whole and do not account for the potential stage-wise features of complex tasks. As a result, the generation process of reward function tend to be highly complex and often require human feedback[24, 26] or extensive search[25]. In contrast, our approach simplifies the reward generation process by decomposing the task into short-term, simple subtasks, providing more specific and clear guidance for each stage of the task.

## 3 Preliminary

In standard reinforcement learning (RL)[1] tasks, the problem is typically modeled as a Markov Decision Process (MDP), denoted by $\mathcal { M } = \langle \boldsymbol { S } , \mathcal { A } , \mathcal { T } , \bar { R } , \boldsymbol { \gamma } , \rho _ { 0 } \rangle$ . Here, S and A represent the state and action spaces, respectively, γ is the discount factor, and $\rho _ { 0 }$ is the distribution of initial states. The dynamics of environment are captured by the transition function $\mathcal { T } : \mathcal { S } \times \mathcal { A } \times \mathcal { S }  [ 0 , 1 ]$ , which defines the probability of transitioning from one state to another given a specific action. The reward function $R : S \times \mathcal { A } \stackrel { \cdot } {  }$ R quantifies the immediate reward received for taking an action in a particular state. A trajectory $\tau = ( s _ { 0 } , a _ { 0 } , r _ { 0 } , \ldots , s _ { T } , a _ { T } , r _ { T } )$ describes the sequence of states, actions, and rewards encountered by an agent following a decision-making policy π. At each time step t, the agent selects an action $a _ { t }$ according to $a _ { t } \sim \pi ( \cdot | s _ { t } )$ , and the environment responds by transitioning to the next state $s _ { t + 1 } \sim \mathcal { T } ( \cdot | s _ { t } , a _ { t } )$ . The state-action value function (Q-function) $Q ^ { \bar { \pi } } ( s , a )$ represents the expected cumulative return when taking action a in state s and subsequently following policy π:

$$
Q ^ { \pi } ( s , a ) = \mathbb { E } _ { \tau \sim p ( \tau \mid \pi ) } \left[ \sum _ { k = 0 } ^ { T } \gamma ^ { k } R ( s _ { t + k } , a _ { t + k } ) \biggm | s _ { t } = s , a _ { t } = a \right]\tag{1}
$$

The objective in RL is for the agent to learn a policy $\pi : { \mathcal { S } }  A$ that maximizes the expected cumulative return defined as:

$$
J ( \pi ) = \mathbb { E } _ { \tau \sim p ( \tau \mid \pi ) } \left[ \sum _ { t = 0 } ^ { T } \gamma ^ { t } R ( s _ { t } , a _ { t } ) \right]\tag{2}
$$

## 4 Method

Our framework, as illustrated in Figure 2, bridges the gap between high-level planning and low-level training. We begin by decomposing a complex long-term task into a series of short-term subtasks, with a determination function Ψ to classify the current subtask (Section 4.1). Next, we construct a potential-based Progress Reward Model (PRM) derived from the progress function $\Phi _ { i }$ , which generates effective reward signals that are proved to have convergence and optimality guarantees (Section 4.2). Finally, the policy is augmented using both the PRM reward and the subtask priors (Section 4.3).

![](images/4e40be47d3737bf6d3ba27e0cf453046417857828def34c083bc5890cc10aac8.jpg)  
Figure 2: An overview of our framework. (a) We prompt the LLM using a Chain-of-Thought approach in Python code format. The reasoning process is provided as comments (e.g., using #), while the intermediate steps and final response are represented as executable Python statements. (b) Our framework augments RL in two ways: subtask prior and the PRM rewards.

LLM usage pattern. In our framework, we prompt an LLM to decompose the tasks and generate the determination function Ψ and progress function Φ. Considering that the intermediate inference steps and the required final response are primarily represented as Python code (as shown later in Section 4.1, 4.2), inspired from PAL[38], we require the LLM to generate programs as intermediate steps, as shown in Figure 2a. Specifically, we instruct the LLM to generate natural language reasoning steps using comment syntax (e.g., ‘#’ in Python), while the code itself is generated normally. Through this program-aided Chain-of-Thought reasoning, the LLM can generate both the plan and progress function in a single interaction.

## 4.1 Subtask Decomposition

Decomposing a long-term task into a series of short-term, manageable tasks provides the policy with subtask priors and alleviates the complexity of constructing the progress function. Existing methods[21, 22] have demonstrated that, given sufficient environment-related information, LLMs are capable of task decomposition. In our framework, we leverage an LLM to break down the task into a series of subtasks, each contributing to the success of the overall task. We construct a Pythonic representation prompt that includes semantically rich contextual information such as state space, environmental descriptions, and task descriptions (detailed in Appendix E), providing a high-dimensional abstraction of environment-specific information.

While decomposing the subtasks, the LLM is also required to generate a natural language form subtask prior, which is a description of the corresponding subtask. To ensure consistency and generalization across the robotics control domain, we designed a fixed pattern for the subtask priors. Specifically, we instruct the LLM to output each subtask prior in a "verb + noun" format (e.g., ‘Move to object’ or ‘Grasp object’), significantly enhancing the generalization ability of the trained policy.

Algorithm 1 PRM4RL   
Require: pythonic LLM prompt prompt, sentence encoder E, policy $\pi _ { \theta } .$ , MDP $\begin{array} { r l } { \mathcal { M } } & { { } = } \end{array}$   
$\langle S , \mathsf { \bar { \mathcal { A } } } , \tau , R , \mathsf { \bar { \gamma } } , \rho _ { 0 } \rangle$ , RL optimization algorithm Optimize\_Algo   
// Request LLM   
plan $\bar { \iota } i s t , \Psi ( s ) , \Phi ( s ) = L L M ( p r o m p t )$   
// Augment State Space   
$\mathcal { P } : \Breve { p r i o r } = \mathbf { E } \bigl ( p l a n \_ l i s t [ \Psi ( s ) ] \bigr )$   
// Augment Reward Function   
$R ^ { P R \breve { M } } = \gamma \cdot \Phi ( s _ { t + 1 } ) - \Phi ( s _ { t } ) + \mathbb { I } ( s _ { t + 1 } ) \cdot r _ { b o n u s }$   
// Optimizing policy under augmented MDP   
$\mathcal { M } ^ { \widehat { \prime } } = \langle \bar { S } + \mathcal { \bar { P } } , \mathcal { A } , \mathcal { T } , R ^ { P R M } , \mathcal { \bar { \gamma } } , \rho _ { 0 } \rangle$   
$\theta ^ { * } = O p t i m i z e \_ A l g o ( M ^ { \prime } )$

The expected output from the LLM is a Python list, called plan\_list, as shown in Figure 2a(II), where each element corresponds to a subtask prior.

However, guiding the policy with subtask priors presents another challenge: real-time tracking of which subtask the policy is currently executing. Previous approaches require calling the LLM at each timestep, either to regenerate a real-time plan[23] or to check the completion of ongoing subtasks[22], leading to significant time and computational overhead.

Our key insight is that, with a thorough understanding of the state space, we can determine which subtask the policy is in based solely on current state. For the previously generated plan\_list, we instruct the LLM to identify the completion condition for each subtask and locate the relevant features in the state space. We then construct a subtask determination function $\Psi ( s )$ , where given state s, it will return the index of current subtask in plan\_list. As shown in Figure 2a(II), the LLM ultimately generates an efficient, low-cost logical chain that ensures rigorous subtask determination.

## 4.2 Potential-Based Progress Reward Model

Our framework aims to construct a dense and efficient reward signal to guide the training of low-level policies. While previous approaches[24, 25, 26] directly design a reward function, we choose to model the progress of task execution instead. This approach is more task-relevant and provides stronger theoretical guarantees on optimality and convergence as proved later.

With the decomposition of subtasks, the focus of our progress function shifts from a long-term, complex task to short-term, simpler subtasks. This transformation simplifies task comprehension and the design of usable metrics. Based on the established subtask determination logic Ψ, we instruct the LLM to further define a progress function for each subtask, and then integrating these fine-grained sub-progress indicators into an overall evaluation of the task progress $\Phi ( s )$ , as shown in Figure 2a(III). The progress function offers a precise and efficient method for monitoring overall task execution.

Inspired by potential-based framework, we treat the progress function $\Phi ( s )$ as a form of potential and construct the Progress Reward Model (PRM) using the temporal difference form:

$$
R _ { t } ^ { P R M } = \gamma \cdot \Phi ( s _ { t + 1 } ) - \Phi ( s _ { t } ) + \mathbb { I } ( s _ { t + 1 } ) \cdot r _ { b o n u s }\tag{3}
$$

where $R _ { t } ^ { P R M }$ is the dense reward given by PRM at timestep $t , \gamma$ is the discount factor, and the indicator function I equals 1 when $s _ { t + 1 }$ meets the success conditions and 0 otherwise. $\Phi ( s _ { t + 1 } )$ and $\Phi ( s _ { t } )$ represent the progress at timesteps $t + 1$ and t, respectively. The PRM reward can be regarded as consisting of two parts: a sparse success reward and a potential-based reward shaping term. Following [39], we present the following theorem:

Theorem 4.1 (Policy Invariance). Define an MDP $M = \langle S , \mathcal { A } , \mathcal { T } , R , \gamma , \rho _ { 0 } \rangle$ , where $R _ { t } = \mathbb { I } ( s _ { t + 1 } )$ r represents the sparse reward when the task is considered successful. Giving a potential-based reward shapingfunction asfollows:

$$
F = \gamma \cdot \Phi ( s _ { t + 1 } ) - \Phi ( s _ { t } )
$$

Let another MDP $M ^ { \prime } = \langle S , A , \mathcal { T } , R + F , \gamma , \rho _ { 0 } \rangle$ , then every optimal policy in M<sup>′</sup> will also be an optimal policy in M and vice versa.

The proof of Theorem 4.1 is provided in Appendix A.1. Theorem 4.1 indicates that $R ^ { P R M }$ does not distort the learning of the policy by introducing "deceptive" or "short-sighted" behaviors due to reward shaping. The optimal policy in ${ \dot { M } } ^ { \prime }$ remains focused on maximizing the original task objective, i.e. maximize success rate. Based on Theorem 4.1, we have the following corollary:

Corollary 4.2 (Convergence Guarantee). The introduction ofpotential-based reward shaping will not affect the convergence ofreinforcement learning algorithms.

Since the PRM reward has a theoretical convergence guarantee, we proceed to demonstrate its efficiency in convergence, following the work of [40] (Detailed proof provided in Appendix A.2).

Theorem 4.3 (Convergence Efficiency). Consider two MDPs: $M = \langle S , A , T , R + F , \gamma , \rho _ { 0 } \rangle$ with Q-function $Q$ and $M ^ { \prime } \bar { = } \langle S , \mathcal { A } , \mathcal { T } , R , \bar { \gamma } , \rho _ { 0 } \rangle$ . If we initialize the Q-function in $M ^ { \prime }$ as follows:

$$
Q _ { 0 } ^ { \prime } ( s , a ) = Q _ { 0 } ( s , a ) + \Phi ( s )
$$

Then for all timesteps $t \geq 0$ and $( s , a ) \in S \times A .$

$$
Q _ { t } ^ { \prime } ( s , a ) = Q _ { t } ( s , a ) + \Phi ( s )
$$

where $Q _ { t }$ and $Q _ { t } ^ { \prime }$ are the Q-functions of MDP M and M<sup>′</sup> respectively.

The mathematical equivalence between PRM and Q-value initialization, as established in Theorem 4.3, provides two fundamental insights into why PRM accelerates convergence. First, it demonstrates that PRM effectively provides prior knowledge-based initialization for value functions, directly reducing the initial approximation error $| Q _ { 0 } - Q ^ { * } |$ . Second, the proof formally justifies that any well-designed potential function $\Phi ( s )$ serves as an actionable curriculum, where the shaping term $\gamma \Phi ( s ^ { \prime } ) - \Phi ( s )$ progressively guides exploration toward high-value regions without altering optimal policies.

These theoretical results establish PRM not only as a heuristic reward design but also as a mathematically grounded approach to guide the policy with optimality and convergence.

## 4.3 Reinforcement Learning Augmentation

Giving an MDP $M = \langle S , \mathcal { A } , \mathcal { T } , R , \gamma , \rho _ { 0 } \rangle$ , our framework augment it through both high-level planning and low-level reward shaping.

At the high level, the policy cannot directly process the natural language form of the plan\_list. To address this, we employ SimCSE[41], a sentence encoder E, to embed the subtask prior. Specifically, the subtask prior is expressed as:

$$
\mathcal { P } : p r i o r = \mathbf { E } ( p l a n \_ l i s t [ \Psi ( s ) ] )
$$

where subtask prior is the embedding of current subtask corresponding to state s. The state space is then augmented to $\mathcal { S } + \mathcal { P }$ . At the low level, we use the reward $\dot { R } ^ { P R M }$ from Section 4.2 to efficiently guide the learning of policy. Finally, the augmented MDP is expressed as:

$$
\mathcal { M } ^ { \prime } = \langle \mathcal { S } + \mathcal { P } , \mathcal { A } , \mathcal { T } , R ^ { P R M } , \gamma , \rho _ { 0 } \rangle
$$

This formulation integrates both high-level planning through the subtask prior and low-level control via the PRM reward to guide policy learning. Consequently, we can utilize existing RL optimization algorithms to optimize the policy under the augmented MDP $M ^ { \prime }$ . In this paper, we use Soft Actor-Critic (SAC)[42] and Proximal Policy Optimization (PPO)[43].

## 5 Experiment

In this section, we evaluate PRM4RL on a diverse suite of long-term, sparse reward RL tasks, testing its ability to provide high-level plan and guide low-level training.

## 5.1 Experiment Settings

The framework is implemented based on the Stable-Baselines3 framework[44], ensuring consistency across all methods. We employ the same RL algorithm with identical training parameters for all methods. For all LLM-related requirements, we utilize OpenAI’s GPT-4o[45]. Details of the hyperparameters are provided in Appendix C. To ensure robustness and reliability, all experiments are conducted with five different random seeds.

![](images/cd94e9ec95b6b57d11c389597b7a2e80d0d75bef0b3eeb8765651823928f7d29.jpg)  
Figure 3: The learning curves of comparison methods on Metaworld and Maniskill measured by success rate. All experiments are conducted with 5 random seeds.

Baselines. ELLM(ICML23) [23] utilizes an LLM as a high-level planner, embedding subtask priors into the policy. It serves as an example of an LLM-planner approach. T2R(ICLR24) [24] combines LLMs and human feedback to generate dense reward functions for policy training. We use it as an example of an LLM-rewarder approach. Oracle is the expert-written dense reward function provided by the environment. The details of the baseline settings are provided in Appendix B.2.

Environments MetaWorld[27] is an open-source simulated benchmark features a Sawyer robot interacting with a tabletop setup that includes drawer, window, ball, faucet, door and many objects. Maniskill[28] is an advanced robotics simulation platform designed for high-fidelity manipulation tasks. Further details about the environment and the tasks are provided in Appendix D.

## 5.2 Evaluation Analysis

## 5.2.1 Outperform previous LLM approaches

As shown in Figures 3 and 4, our method consistently outperforms the comparison methods across nearly all environments, demonstrating faster convergence and superior performance. We observe the following advantages: (1) In complex long-term tasks such as MW-pick-out-of-hole and MS-LiftPegUpright, the baseline methods achieve less than 15% success, while our method achieves nearly 100% success, highlighting our ability to effectively solve challenging tasks. (2) When compared to ELLM, our approach demonstrates a significantly improved convergence speed, which underscores the efficiency of the Progress Reward Model in providing dense, theoretically guaranteed rewards for the low-level control policy. This results in more stable and faster learning compared to traditional LLM-based planning approaches. (3) In comparison with T2R, the subtask decomposition in our framework plays a critical role in simplifying the coding process. By breaking down complex tasks into more manageable subtasks, our method reduces the difficulty of understanding environment dynamics and designing useful metrics, enabling us to solve complex tasks that T2R fails to address. This advantage is particularly evident in tasks where the environment is more intricate and requires a clear separation of the planning and reward stages.

![](images/38a93385d55990685a6fc6f584df81d533b74e79a64c2499e533c01e0e0d35a2.jpg)  
Figure 4: The evaluation results of comparison methods on Metaworld(With prefix ‘MW’) and Maniskill(with prefix ‘MS’) measured by success rate. All experiments are conducted with 5 random seeds.

![](images/1636606bad3562decaa0698381e54412e7aad2ef7dc01cb6b7dbc3fa9995f6ff.jpg)

![](images/1c98baff0c6142086cf1d3e2fd17f885310bf3e320404e4b88d7572c435858d4.jpg)

![](images/a0dad3c0ade344d24db24135067ea6b8295d0089539333ec530ea548aac22ac8.jpg)

![](images/9292e5509f9875926f6ec172e3aacc3805d59d2b12983095b02ca56a5a179e89.jpg)

![](images/c95675f788ecdf929e8d638cb24ba95061607298f2deba5451d25edbd0fb0781.jpg)  
Figure 5: The evaluation results of ablation study on Metaworld measured by success rate. All experiment is conducted with 5 random seeds.

## 5.2.2 Ablation Study

To further validate the effectiveness of the two augmentations in our framework, we conduct an ablation study by constructing three additional baselines:

w.o. PRM reward (denoted as ‘our w.o. prm’): The PRM reward is replaced with the oracle reward, while retaining the subtask prior.

w.o. subtask prior (denoted as ‘our w.o. prior’): No subtask prior is provided, while retaining the PRM reward.

w.o. potential-based reward shaping (denoted as ‘our w.o. potential reward’): The progress function is used directly as the reward, while retaining the subtask prior.

![](images/9b9a651302beb020f11aeb63445f25484483fc26700d313e1cf23a085f6710a8.jpg)

The results are shown in Figure 5. We observe the following findings:

(1) Compared with ‘our w.o. prm’, replacing the PRM reward with the oracle reward leads to a significant drop in performance. This demonstrates that, without an efficient and effective reward signal, directly feeding the subtask prior to the policy does not improve its performance. In most cases, the training curves and evaluation results of ‘our w.o. prm’ are similar to those of the oracle, with some instances where adding the subtask prior even causes a slight performance decrease.

![](images/18407c779f9e3ae052ccaf5c1356a140224f541fdc27faf3b5e4f4d273d894d0.jpg)  
Figure 6: Training details of our and our w.o. potential reward.

(2) The performance of ‘our w.o. prior’ shows that abandoning the subtask prior results in slower convergence in the training curves. This highlights the role of the subtask prior in enabling efficient convergence for complex, long-term tasks. Moreover, ‘our w.o. prior’ still outperforms T2R, further demonstrating the effectiveness of the PRM reward in guiding the policy.

(3)The performance of ‘our w.o. potential reward’ further demonstrate the effectiveness of our potential-based reward model, where directly using progress as reward signal lead to a significant drop in performance. We also analyzed the anomalous behavior of ‘our w.o. potential reward’, where the success rate initially rises but then declines as training progresses. As shown in Figure 6, the episode reward increases, while the success rate paradoxically decreases. This suggests that the using progress function directly as reward function suffers from inefficiencies, as there are instances where the task fails despite receiving a high reward, leading the policy to learn in an undesirable direction. In contrast, our method ensures stable convergence, with the PRM providing well-structured and task-relevant rewards with optimality and convergence guarantees.

In summary, the two enhancements in our framework—the subtask prior and the PRM reward—work synergistically to produce a "1+1 > 2" effect. Without the subtask prior, convergence speed is hindered, and without the reward signal, providing effective and efficient guidance becomes challenging.

![](images/e02cb18959a6fb98001e27da6c97474c19021a6aae089a083a182512ae8b73b9.jpg)  
Figure 7: The evaluation results of generalization experiment. The results are averaged on 5 random seeds and 100 evaluations per seed.

## 5.2.3 Generalization Results

To evaluate the generalization capabilities of the trained policy, we test the trained policy’s performance on unseen tasks in MetaWorld. Additional results and detailed information regarding the tasks are provided in Appendix B.1. As shown in Figure 7, while most methods perform well on the original task, we observe a performance decline on the unseen tasks. In comparison to other methods, our framework experiences the least performance loss, demonstrating relatively strong generalization abilities. Notably, when comparing to our ‘w.o. prior’ (our framework without the subtask prior), we observe a significant drop in performance on unseen tasks. This highlights the critical role of the subtask prior in enhancing generalization. The "verb + noun" pattern we designed effectively captures commonalities across different tasks, acting as a powerful prior that guides the policy to perform well on unseen tasks. Thus, the subtask prior serves as a robust tool for improving the generalization of reinforcement learning policies across a wide range of tasks.

## 6 Conclusion

In this paper, we introduced the Progress Reward Model for Reinforcement Learning (PRM4RL), a novel framework that integrates high-level planning and low-level reward shaping to address the challenges of long-term planning and sparse rewards in reinforcement learning. At the high level, we decompose a complex task into manageable subtasks with subtask prior. At the low level, we design a Progress Reward Model with optimality and convergence guarantee. The experimental results demonstrate the effectiveness and efficiency of our method, significantly improving both convergence speed and the ability to solve complex, multi-stage tasks.

Limitations. Similar to T2R[24] and other methods[23, 21, 22] that depend on LLMs for planning or coding, our approach could be affected by LLM hallucinations during reasoning[46]. Besides, while our method demonstrates strong performance across various tasks, its applicability to domains outside of robotics remains unexplored. We anticipate that further optimization and adaptation of the framework will be necessary to address diverse environments and tasks.

## References

[1] Richard S Sutton, Andrew G Barto, et al. Reinforcement learning: An introduction, volume 1. MIT press Cambridge, 1998.

[2] Gabriel Dulac-Arnold, Nir Levine, Daniel J Mankowitz, Jerry Li, Cosmin Paduraru, Sven Gowal, and Todd Hester. Challenges of real-world reinforcement learning: definitions, benchmarks and analysis. Machine Learning, 110(9):2419–2468, 2021.

[3] Ofir Nachum, Shixiang Shane Gu, Honglak Lee, and Sergey Levine. Data-efficient hierarchical reinforcement learning. Advances in neural information processing systems, 31, 2018.

[4] Shubham Pateria, Budhitama Subagdja, Ah-hwee Tan, and Chai Quek. Hierarchical reinforcement learning: A comprehensive survey. ACM Computing Surveys (CSUR), 54(5):1–35, 2021.

[5] Saurabh Arora and Prashant Doshi. A survey of inverse reinforcement learning: Challenges, methods and progress. Artificial Intelligence, 297:103500, 2021.

[6] Brian D Ziebart, Andrew L Maas, J Andrew Bagnell, Anind K Dey, et al. Maximum entropy inverse reinforcement learning. In Aaai, volume 8, pages 1433–1438. Chicago, IL, USA, 2008.

[7] Chelsea Finn, Sergey Levine, and Pieter Abbeel. Guided cost learning: Deep inverse optimal control via policy optimization. In International conference on machine learning, pages 49–58. PMLR, 2016.

[8] Shanchuan Wan, Yujin Tang, Yingtao Tian, and Tomoyuki Kaneko. Deir: efficient and robust exploration through discriminative-model-based episodic intrinsic rewards. arXiv preprint arXiv:2304.10770, 2023.

[9] Anna Penzkofer, Simon Schaefer, Florian Strohm, Mihai Bâce, Stefan Leutenegger, and Andreas Bulling. Int-hrl: towards intention-based hierarchical reinforcement learning. Neural Computing and Applications, pages 1–12, 2024.

[10] Xinlei Pan, Eshed Ohn-Bar, Nicholas Rhinehart, Yan Xu, Yilin Shen, and Kris M Kitani. Human-interactive subgoal supervision for efficient inverse reinforcement learning. arXiv preprint arXiv:1806.08479, 2018.

[11] Zihao Yi, Jiarui Ouyang, Yuwen Liu, Tianhao Liao, Zhe Xu, and Ying Shen. A survey on recent advances in llm-based multi-turn dialogue systems. arXiv preprint arXiv:2402.18013, 2024.

[12] Ethan Brooks, Logan Walls, Richard L Lewis, and Satinder Singh. Large language models can implement policy iteration. Advances in Neural Information Processing Systems, 36:30349– 30366, 2023.

[13] Andrew Zhao, Daniel Huang, Quentin Xu, Matthieu Lin, Yong-Jin Liu, and Gao Huang. Expel: Llm agents are experiential learners. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pages 19632–19642, 2024.

[14] Yuanzhao Zhai, Tingkai Yang, Kele Xu, Dawei Feng, Cheng Yang, Bo Ding, and Huaimin Wang. Enhancing decision-making for llm agents via step-level q-value models. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 27161–27169, 2025.

[15] Baptiste Roziere, Jonas Gehring, Fabian Gloeckle, Sten Sootla, Itai Gat, Xiaoqing Ellen Tan, Yossi Adi, Jingyu Liu, Romain Sauvestre, Tal Remez, et al. Code llama: Open foundation models for code. arXiv preprint arXiv:2308.12950, 2023.

[16] Weixiang Yan, Haitian Liu, Yunkun Wang, Yunzhe Li, Qian Chen, Wen Wang, Tingyu Lin, Weishan Zhao, Li Zhu, Hari Sundaram, et al. Codescope: An execution-based multilingual multitask multidimensional benchmark for evaluating llms on code understanding and generation. arXiv preprint arXiv:2311.08588, 2023.

[17] An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, et al. Qwen2. 5 technical report. arXiv preprint arXiv:2412.15115, 2024.

[18] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. In International Conference on Learning Representations (ICLR), 2023.

[19] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

[20] Yuji Cao, Huan Zhao, Yuheng Cheng, Ting Shu, Yue Chen, Guolong Liu, Gaoqi Liang, Junhua Zhao, Jinyue Yan, and Yun Li. Survey on large language model-enhanced reinforcement learning: Concept, taxonomy, and methods, 2024.

[21] Anthony Brohan, Yevgen Chebotar, Chelsea Finn, Karol Hausman, Alexander Herzog, Daniel Ho, Julian Ibarz, Alex Irpan, Eric Jang, Ryan Julian, et al. Do as i can, not as i say: Grounding language in robotic affordances. In Conference on robot learning, pages 287–318. PMLR, 2023.

[22] Murtaza Dalal, Tarun Chiruvolu, Devendra Chaplot, and Ruslan Salakhutdinov. Plan-seqlearn: Language model guided rl for solving long horizon robotics tasks. arXiv preprint arXiv:2405.01534, 2024.

[23] Yuqing Du, Olivia Watkins, Zihan Wang, Cédric Colas, Trevor Darrell, Pieter Abbeel, Abhishek Gupta, and Jacob Andreas. Guiding pretraining in reinforcement learning with large language models. In International Conference on Machine Learning, pages 8657–8677. PMLR, 2023.

[24] Tianbao Xie, Siheng Zhao, Chen Henry Wu, Yitao Liu, Qian Luo, Victor Zhong, Yanchao Yang, and Tao Yu. Text2reward: Automated dense reward function generation for reinforcement learning. In International Conference on Learning Representations (ICLR), 2024 (07/05/2024- 11/05/2024, Vienna, Austria), 2024.

[25] Yecheng Jason Ma, William Liang, Guanzhi Wang, De-An Huang, Osbert Bastani, Dinesh Jayaraman, Yuke Zhu, Linxi Fan, and Anima Anandkumar. Eureka: Human-level reward design via coding large language models. arXiv preprint arXiv:2310.12931, 2023.

[26] Rishi Hazra, Alkis Sygkounas, Andreas Persson, Amy Loutfi, and Pedro Zuidberg Dos Martires. Revolve: Reward evolution with large language models using human feedback. arXiv preprint arXiv:2406.01309, 2024.

[27] Tianhe Yu, Deirdre Quillen, Zhanpeng He, Ryan Julian, Karol Hausman, Chelsea Finn, and Sergey Levine. Meta-world: A benchmark and evaluation for multi-task and meta reinforcement learning. In Conference on Robot Learning (CoRL), 2019.

[28] Stone Tao, Fanbo Xiang, Arth Shukla, Yuzhe Qin, Xander Hinrichsen, Xiaodi Yuan, Chen Bao, Xinsong Lin, Yulin Liu, Tse kai Chan, Yuan Gao, Xuanlin Li, Tongzhou Mu, Nan Xiao, Arnav Gurha, Viswesh Nagaswamy Rajesh, Yong Woo Choi, Yen-Ru Chen, Zhiao Huang, Roberto Calandra, Rui Chen, Shan Luo, and Hao Su. Maniskill3: Gpu parallelized robotics simulation and rendering for generalizable embodied ai. Robotics: Science and Systems, 2025.

[29] Archiki Prasad, Alexander Koller, Mareike Hartmann, Peter Clark, Ashish Sabharwal, Mohit Bansal, and Tushar Khot. Adapt: As-needed decomposition and planning with language models. arXiv preprint arXiv:2311.05772, 2023.

[30] Vishnu Sarukkai, Brennan Shacklett, Zander Majercik, Kush Bhatia, Christopher Ré, and Kayvon Fatahalian. Automated rewards via llm-generated progress functions, 2024.

[31] Meng Cao, Lei Shu, Lei Yu, Yun Zhu, Nevan Wichers, Yinxiao Liu, and Lei Meng. Drlc: Reinforcement learning with dense rewards from llm critic. arXiv e-prints, pages arXiv–2401, 2024.

[32] Tushar Khot, Harsh Trivedi, Matthew Finlayson, Yao Fu, Kyle Richardson, Peter Clark, and Ashish Sabharwal. Decomposed prompting: A modular approach for solving complex tasks, 2023.

[33] Quan Yuan, Mehran Kazemi, Xin Xu, Isaac Noble, Vaiva Imbrasaite, and Deepak Ramachandran. Tasklama: Probing the complex task understanding of language models, 2023.

[34] Yongchao Chen, Rujul Gandhi, Yang Zhang, and Chuchu Fan. Nl2tl: Transforming natural languages to temporal logics using large language models. arXiv preprint arXiv:2305.07766, 2023.

[35] Yongchao Chen, Jacob Arkin, Charles Dawson, Yang Zhang, Nicholas Roy, and Chuchu Fan. Autotamp: Autoregressive task and motion planning with llms as translators and checkers. In 2024 IEEE International conference on robotics and automation (ICRA), pages 6695–6702. IEEE, 2024.

[36] Rati Devidze, Parameswaran Kamalaruban, and Adish Singla. Informativeness of reward functions in reinforcement learning, 2024.

[37] Jonas Eschmann. Reward function design in reinforcement learning. Reinforcement learning algorithms: Analysis and Applications, pages 25–33, 2021.

[38] Luyu Gao, Aman Madaan, Shuyan Zhou, Uri Alon, Pengfei Liu, Yiming Yang, Jamie Callan, and Graham Neubig. Pal: Program-aided language models. In International Conference on Machine Learning, pages 10764–10799. PMLR, 2023.

[39] Andrew Y Ng, Daishi Harada, and Stuart Russell. Policy invariance under reward transformations: Theory and application to reward shaping. In Icml, volume 99, pages 278–287. Citeseer, 1999.

[40] E. Wiewiora. Potential-based shaping and q-value initialization are equivalent. Journal of Artificial Intelligence Research, 19:205–208, September 2003.

[41] Tianyu Gao, Xingcheng Yao, and Danqi Chen. Simcse: Simple contrastive learning of sentence embeddings, 2022.

[42] Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine. Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor, 2018.

[43] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

[44] Antonin Raffin, Ashley Hill, Adam Gleave, Anssi Kanervisto, Maximilian Ernestus, and Noah Dormann. Stable-baselines3: Reliable reinforcement learning implementations. Journal of Machine Learning Research, 22(268):1–8, 2021.

[45] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024.

[46] Fang Liu, Yang Liu, Lin Shi, Houkun Huang, Ruifeng Wang, Zhen Yang, Li Zhang, Zhongqi Li, and Yuchi Ma. Exploring and evaluating hallucinations in llm-powered code generation. arXiv preprint arXiv:2404.00971, 2024.

## A Theoretical Proof

Theorem A.1 (Policy Invariance). Define an MDP $M = \langle S , \mathcal { A } , \mathcal { T } , R , \gamma , \rho _ { 0 } \rangle$ , where $R _ { t } = \mathbb { I } ( s _ { t + 1 } )$ $r _ { b o n u s }$ represents the sparse reward when the task is considered successful. Giving a potential-based reward shapingfunction asfollows:

$$
F = \gamma \cdot \Phi ( s _ { t + 1 } ) - \Phi ( s _ { t } )
$$

Let another MDP $M ^ { \prime } = \langle S , A , \mathcal { T } , R + F , \gamma , \rho _ { 0 } \rangle$ , then every optimal policy in $M ^ { \prime }$ will also be an optimal policy in M and vice versa.

Proof. For the original MDP M, we know that its optimal Q-function $Q _ { M } ^ { * }$ satisfies the Bellman Equations

$$
Q _ { M } ^ { * } ( s , a ) = \mathbb { E } _ { s ^ { \prime } \sim P _ { s , a } ( \cdot ) } \left[ R ( s , a , s ^ { \prime } ) + \gamma \operatorname* { m a x } _ { a ^ { \prime } \in A } Q _ { M } ^ { * } ( s ^ { \prime } , a ^ { \prime } ) \right]
$$

Some simple algebraic manipulation then gives us

$$
Q _ { M } ^ { * } ( s , a ) - \Phi ( s ) = \mathbb { E } _ { s ^ { \prime } } \left[ R ( s , a , s ^ { \prime } ) + \gamma \Phi ( s ^ { \prime } ) - \Phi ( s ) + \gamma \operatorname* { m a x } _ { a ^ { \prime } \in A } ( Q _ { M } ^ { * } ( s ^ { \prime } , a ^ { \prime } ) - \Phi ( s ^ { \prime } ) ) \right]
$$

If we now define $\hat { Q } _ { M ^ { \prime } } ( s , a ) \triangleq Q _ { M } ^ { * } ( s , a ) - \Phi ( s )$ and substitute that and $F ( s , a , s ^ { \prime } ) = \gamma \Phi ( s ^ { \prime } ) - \Phi ( s )$ back into the previous equation, we get

$$
\begin{array} { r l } & { \hat { Q } _ { M ^ { \prime } } ( s , a ) = \mathbb { E } _ { s ^ { \prime } } \left[ R ( s , a , s ^ { \prime } ) + F ( s , a , s ^ { \prime } ) + \gamma \operatorname* { m a x } _ { a ^ { \prime } \in A } \hat { Q } _ { M ^ { \prime } } ( s ^ { \prime } , a ^ { \prime } ) \right] } \\ & { \quad \quad \quad = \mathbb { E } _ { s ^ { \prime } } \left[ R ^ { \prime } ( s , a , s ^ { \prime } ) + \gamma \operatorname* { m a x } _ { a ^ { \prime } \in A } \hat { Q } _ { M ^ { \prime } } ( s ^ { \prime } , a ^ { \prime } ) \right] } \end{array}
$$

But this is exactly the Bellman equation for $M ^ { \prime }$ . For the undiscounted case, we moreover have $\hat { Q } _ { M ^ { \prime } } ( s _ { 0 } , a ) = Q _ { M } ^ { * } ( s _ { 0 } , a ) - \Phi ( s _ { 0 } ) = 0 - 0 = 0 .$ . So, $\hat { Q } _ { M ^ { \prime } } ( s , a )$ satisfies the Bellman equations for $M ^ { \prime }$ , and must in fact be the unique optimal Q-function. Thus, $Q _ { M ^ { \prime } } ^ { * } ( s , a ) = \hat { Q } _ { M ^ { \prime } } ( s , a ) =$ $Q _ { M } ^ { * } ( s , a ) - \Phi ( s )$ and the optimal policy for $M ^ { \prime }$ therefore satisfies.

$$
\begin{array} { r l } & { \pi _ { M ^ { \prime } } ^ { * } ( s ) \in \arg \underset { a \in A } { \operatorname* { m a x } } Q _ { M ^ { \prime } } ^ { * } ( s , a ) } \\ & { \quad \quad \quad = \arg \underset { a \in A } { \operatorname* { m a x } } Q _ { M } ^ { * } ( s , a ) - \Phi ( s ) } \\ & { \quad \quad \quad = \arg \underset { a \in A } { \operatorname* { m a x } } Q _ { M } ^ { * } ( s , a ) } \end{array}
$$

and is therefore also optimal in M. To show every optimal policy in M is also optimal in $M ^ { \prime }$ , simply apply the same proof with the roles of M and $M ^ { \dagger }$ interchanged (and using the shaping function $- F )$ This completes the proof. 口

Theorem A.2 (Convergence Efficiency). Consider two MDPs: $M = \langle \mathcal { S } , \mathcal { A } , \mathcal { T } , R + F , \gamma , \rho _ { 0 } \rangle$ with Q-function $Q$ and $M ^ { \prime } \bar { = } \langle S , \mathcal { A } , \mathcal { T } , R , \bar { \gamma } , \rho _ { 0 } \rangle$ . If we initialize the Q-function in $M ^ { \prime }$ as follows:

$$
Q _ { 0 } ^ { \prime } ( s , a ) = Q _ { 0 } ( s , a ) + \Phi ( s )
$$

Then for all timesteps $t \geq 0$ and $( s , a ) \in S \times A .$

$$
Q _ { t } ^ { \prime } ( s , a ) = Q _ { t } ( s , a ) + \Phi ( s )\tag{4}
$$

where $Q _ { t }$ and $Q _ { t } ^ { \prime }$ are the Q-functions of MDP M and M<sup>′</sup> respectively.

Proof. We prove the equivalence by induction on the update steps.

Base case (t = 0): By initialization definition:

$$
Q _ { 0 } ^ { \prime } ( s , a ) = Q _ { 0 } ( s , a ) + \Phi ( s ) \quad \forall ( s , a )\tag{5}
$$

Inductive step: Assume $Q _ { k } ^ { \prime } ( s , a ) = Q _ { k } ( s , a ) + \Phi ( s )$ holds for all $( s , a )$ at step k. Consider the Q-learning update for both agents when transitioning from $( s , a ) \mathrm { t o } ( s ^ { \prime } , a ^ { \prime } )$ with reward r: In MDP M:

$$
\begin{array} { r l } & { Q _ { k + 1 } ( s , a ) = Q _ { k } ( s , a ) + \alpha _ { k } \Big [ r + F ( s , s ^ { \prime } ) + \gamma \operatorname* { m a x } _ { a ^ { \prime } } Q _ { k } ( s ^ { \prime } , a ^ { \prime } ) - Q _ { k } ( s , a ) \Big ] } \\ & { \qquad = Q _ { k } ( s , a ) + \alpha _ { k } \Big [ r + ( \gamma \Phi ( s ^ { \prime } ) - \Phi ( s ) ) + \gamma \operatorname* { m a x } _ { a ^ { \prime } } Q _ { k } ( s ^ { \prime } , a ^ { \prime } ) - Q _ { k } ( s , a ) \Big ] } \end{array}
$$

In MDP $M ^ { \prime } { : }$

$$
\begin{array} { r l } & { Q _ { k + 1 } ^ { \prime } ( s , a ) = Q _ { k } ^ { \prime } ( s , a ) + \alpha _ { k } \Big [ r + \gamma \operatorname* { m a x } _ { a ^ { \prime } } Q _ { k } ^ { \prime } ( s ^ { \prime } , a ^ { \prime } ) - Q _ { k } ^ { \prime } ( s , a ) \Big ] } \\ & { \qquad = [ Q _ { k } ( s , a ) + \Phi ( s ) ] + \alpha _ { k } \Big [ r + \gamma \operatorname* { m a x } _ { a ^ { \prime } } [ Q _ { k } ( s ^ { \prime } , a ^ { \prime } ) + \Phi ( s ^ { \prime } ) ] - [ Q _ { k } ( s , a ) + \Phi ( s ) ] \Big ] } \\ & { \qquad = Q _ { k } ( s , a ) + \Phi ( s ) + \alpha _ { k } \Big [ r + \gamma \operatorname* { m a x } _ { a ^ { \prime } } Q _ { k } ( s ^ { \prime } , a ^ { \prime } ) + \gamma \Phi ( s ^ { \prime } ) - Q _ { k } ( s , a ) - \Phi ( s ) \Big ] } \end{array}
$$

Simplify the difference:

$$
\begin{array} { r l } & { Q _ { k + 1 } ^ { \prime } ( s , a ) - Q _ { k + 1 } ( s , a ) = \Phi ( s ) + \alpha _ { k } \Big [ \gamma \Phi ( s ^ { \prime } ) - \Phi ( s ) \Big ] - \alpha _ { k } \Big [ \gamma \Phi ( s ^ { \prime } ) - \Phi ( s ) \Big ] } \\ & { \qquad = \Phi ( s ) } \end{array}
$$

Thus $Q _ { k + 1 } ^ { \prime } ( s , a ) = Q _ { k + 1 } ( s , a ) + \Phi ( s )$ maintains the invariant. By induction, the equivalence holds for all $t \geq 0$ □

## B Experiment Details

## B.1 Additional Generalization Results

![](images/107f6e399da46d170a3e95c2d6f46b884e18c4d21c01118304566c9b4b04e93c.jpg)  
Figure 8: The evaluation results of generalization experiment.

We conduct additional generalization evaluations. Specifically, for the evaluation results shown in Section 5.2.3, the original task is ‘Metaworld-faucet-open’, the unseen tasks are ‘Metaworlddrawer-close’, ‘Metaworld-button-press-wall’, ‘Metaworld-button-press’, ‘Metaworld-coffee-button’, respectively. For the results shown in Figure 8, the original task is ‘Metaworld-pick-place’, the unseen tasks are ‘Metaworld-drawer-close’, ‘Metaworld-pick-place-wall’, ‘Metaworld-sweep-into’, ‘Metaworld-handle-press’, respectively. We test the trained policy in evaluation tasks, and the results are averaged over 5 seeds and 100 test per seed.

## B.2 Baseline Settings

ELLM(ICML23)[23] utilizes an LLM as a high-level planner, embedding subtask priors into the policy. Since ELLM has not been previously tested on Metaworld and Maniskill, we re-implement it the experiments. For goal embedding, we gave the same subtask prior as in our framework to the policy; For reward, we then gave it the subtask success reward, which is also goal-related as in its original application. We also use our Ψ for determining current subtask to avoid enormous LLM calls.

Text2Reward(ICLR24)[24] utilizes an LLM to write dense reward function. As T2R has evaluate its performance on Metaworld and Maniskill, we directly reuse its generated reward functions. If there are few-shot and zero-shot versions of reward functions, we always choose the few-shot version for better performance. Specifically, for ‘Metaworld-drawer-close’, ‘Metaworld-window-open’, ‘Metaworld-button-press’, ‘Metaworld-sweep-into’, ‘Metaworld-door-close’, ‘Maniskill-LiftCube’, ‘Maniskill-PickCube’, we reuse its generated reward functions. For other tasks that are not orginally evaluated in T2R, we generate a reward function following its code.

## C Hyper-Parameter Details

In this section, we provide the hyper-parameter details used for our reward function generation and reinforcement learning backbones. For progress function generation, we base on GPT-4o. In the experiments of the main body, the temperature of sampling is set to 0.7 for each experiment.

For reinforcement learning training, we employed the open-source PPO[43] and SAC implementation[42] from Stable-Baselines3[44] <sup>1</sup> , and list the hyper-parameters in Table 1 and 2.

Table 1: Hyper-parameter of SAC algorithm applied to each task.
<table><tr><td>Hyper-parameter</td><td>Value</td></tr><tr><td>Discount factor γ Target update frequency</td><td>0.99 2</td></tr><tr><td>Learning rate</td><td> $3 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Train frequency</td><td>1</td></tr><tr><td>Soft update τ Gradient steps</td><td> $5 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>Learning starts</td><td>1</td></tr><tr><td>Batch Size # of layers</td><td>4000 512</td></tr><tr><td></td><td>3</td></tr><tr><td>Hidden units per layer</td><td>256</td></tr><tr><td>Initial temperature Rollout steps per episode</td><td>0.1 500</td></tr></table>

Table 2: Hyper-parameter of PPO algorithm applied to each task.
<table><tr><td>Hyper-parameter</td><td>Value</td></tr><tr><td>Discount factor γ</td><td>0.8</td></tr><tr><td># of epochs per update</td><td>15</td></tr><tr><td>Learning rate</td><td> $1 \times 1 0 ^ { - 4 }$ </td></tr><tr><td># of environments Batch size</td><td>8 400</td></tr><tr><td>Target KL divergence</td><td>0.1</td></tr><tr><td># of layers</td><td>3</td></tr><tr><td>Hidden units per layer</td><td>256</td></tr><tr><td># of steps per update</td><td>3200</td></tr></table>

We utilize a 4 NVidia Geforce RTX-3090 graphic cards, 128 core CPUs, and 256 GiB memory server for RL training. The time required for training a policy is approximately 1 hours per task (5 seeds running simultaneouly) for MetaWorld, and between 1 to 10 hours for ManiSkill, varying with the task’s difficulty.

## D Tasks

In this section, we provide a full list of tasks within each simulation environment, accompanied by their corresponding language instructions.

## D.1 Metaworld

In METAWORLD[27]<sup>2</sup> environment, we use a 7 DoF Sawyer robot arm with a fixed base to complete tabletop tasks. For all tasks, the observation space is a combination of the 3D position of the robot end-effector, a normalized measurement of gripper openness, the 3D position of the manipulated object, the quaternion of the object, all of the previous measurements, and the goal position. The environment adopts end-effector delta position control, which means the action space consists of the change of the end effector’s 3D position, as well as the normalized torque the gripper should apply. For all tasks, the initial and target positions of the manipulated object and the initial joint positions of the robot arm are variable.

![](images/ca2427bc39039550cfd2b1851be29493a7cd8d18cebf1ac38f9cb5e0d340a047.jpg)  
Figure 9: Tasks in Metaworld.

The task descriptions are provided in the following Table 3.

Table 3: Task list of Metaworld
<table><tr><td>Task</td><td>Task Description</td></tr><tr><td>button-press</td><td>Press a button in y coordination.</td></tr><tr><td>door-close</td><td>Close a door with a revolving joint by pushing the door&#x27;s handle.</td></tr><tr><td>drawer-close</td><td>Close a drawer by its handle.</td></tr><tr><td>faucet-open</td><td>Rotate the handle counter-clockwise.</td></tr><tr><td>window-open</td><td>Push and open a sliding window by its handle.</td></tr><tr><td>sweep-into</td><td>Sweep a puck from the initial position into a hole.</td></tr><tr><td>pick-place</td><td>Pick up an object and move it to the goal location.</td></tr><tr><td>pick-out-of-hole</td><td>Pick an object out of a hole and move it to the goal location.</td></tr><tr><td>sweep</td><td>Sweep a puck off the table.</td></tr><tr><td>push</td><td>Push an object to the goal location.</td></tr></table>

## D.2 Maniskill

MANISKILL[28]<sup>3</sup> environment uses a 7 DoF Franka Panda as the default robot arm. For all tasks, the observation space consists of robot proprioception information (e.g. current joint positions, current joint velocities, robot base position and quaternion in the world frame) and task-specific information (e.g. goal position, endeffector position). We use end-effector delta pose control mode for this environment, which controls the change of 3D position and rotation (represented as an axis-angle in the end-effector frame). For all tasks, the initial and target positions of the manipulated object, the initial joint positions of the robot arm and physical parameters (e.g. friction and damping coefficient) are variable.

![](images/5758e527e5caeeceb5a1472c5ff5e031d137b1835f6617ce56949f49b0149d41.jpg)  
LiftCube

![](images/255800631a2dff262a9d7402d202f192fdf7ddc052c263235519156f4ca0c547.jpg)  
PullCube

![](images/1cc43419b39dc2d1532296d75fde912a618a82a72544738d3c91cdf8c5abaf0d.jpg)  
PushCube

![](images/d0d78ecbdcdff0e54d0e4dc77296ca7fb3abf708f5b4467df31385afe2f8e9c6.jpg)  
PickCube

![](images/5529f4c957eca603e9f27b926bde3c2721daf9bccca46379d339c530ea01c6d4.jpg)  
LiftPegUpright  
Figure 10: Tasks in Maniskill.

The task descriptions are provided in the following Table 4.

Table 4: Task list of Maniskill
<table><tr><td>Task</td><td>Task Description</td></tr><tr><td>LiftCube</td><td>Pick up cube A and lift it up by 0.2 meters.</td></tr><tr><td>PullCube</td><td>Pull a cube to the goal position.</td></tr><tr><td>PushCube</td><td>Push a cube to the goal position.</td></tr><tr><td>PickCube</td><td>Pick up cube A and move it to the 3D goal position.</td></tr><tr><td>LiftPegUpright</td><td>Move a peg laying on the table to any upright position on the table.</td></tr></table>

## E Prompt Details

To ground LLMs into robotics simulation environments, following Text2Reward[24] we propose a Pythonic prompt, which can be abstracted by the experts who developed the environment. This class-like prompt is a more compact representation than simply listing all environment attributes linearly, which can delete redundant information and save more tokens, and this Pythonic prompt can also better bootstrap Python code generation. More specifically, our prompt leverages python class, class attribute, python typing and comments to recursively define the environment. For simplicity, here we prompt the progress function to return the current subtask index together with the main progress.

Here we provide an example of the zero-shot prompt for Metaworld manipulation tasks:

Basic Prompt provides the foundational context for the task. The prompt establishes the user’s role as an expert in robotics, reinforcement learning, and code generation, focusing on the development of a Python function called ‘progress function’ for RL. The objective is to help generate code that tracks progress during task execution, guiding the learning of a low-level policy.

## Basic Propmt

You are an expert in robotics, reinforcement learning and code generation.We are going to use a robot arm to complete given tasks. Now I want you to help me write a python function named ‘progress function’ of reinforcement learning.

Environment Information defines the environment in which the robot operates. The BaseEnv class describes the robot and its interaction with objects in the environment, including their positions and states. The Robot and RigidObject classes further define the robot’s end effector, object positions, and other relevant parameters. This context provides essential details for the task, enabling the LLM to make informed decisions about task progress and control.

Environment Information   
class BaseEnv (gym. env):   
self . robot : Robot # the robot in the environment   
self .obj : RigidObject # the first object in the environment   
self . goal\_position : np. ndarray [(3 ,)] # indicate the 3D   
position of the goal   
near\_object = float ( tcp\_to\_obj <= 0.3)   
success = float ( obj\_to\_target <= 0.07)   
class Robot :   
self . ee\_position : np. ndarray [(3 ,)] # indicate the 3D   
position of effector   
self . hand\_init\_pos : np. ndarray [(3 ,)] # indicate the initial   
3D position of effector   
self .tcp = self . ee\_position - [0, 0, 0.045] # indicate the   
3D position of effector ’s tool center point   
class RigidObject :   
self . position : np . ndarray [(3 ,) ] # indicate the 3D position   
of the rigid object   
self . quaternion : np . ndarray [(4 ,) ] # indicate the quaternion   
of the rigid object   
self . obj\_init\_pos : np . ndarray [(3 ,) ] # indicate the initial   
3D position of the rigid object

Task Information provides a description of the task to be completed by the robot. The task description, which needs to be filled in by the user, is crucial for decomposing the task into manageable subtasks.

## Task Information

{ Fill in the task description here. } (e.g. In pick-place, the robot need to pick up the object and move it to the goal position.)

Basic Instructions provide guidelines for writing the code necessary for the task. The instructions emphasize the importance of using existing Python packages only when necessary, avoiding the invention of new variables or attributes, and incorporating step-by-step reasoning with clear comments. These instructions ensure that the code remains efficient, understandable, and aligned with the task’s requirements.

## Basic Instruction

1. You are allowed to use any existing python package if applicable. But only use these packages when it’s really necessary.

2. Do not invent any variable or attribute that is not given.

3. Think step by step, add comment for reasoning and thought when you write code.

Reasoning Instructions outline a step-by-step procedure for task decomposition and the generation of key functions.

## Reasoning Instructions

Follow the steps below:   
a. Decompose the task and generate plan\_list   
a.1 Decompose the given task into a series of subtasks. Each subtask should be described in a ‘verb + noun’ form. tips: ‘move to object’, ‘grasp object’, ‘move to goal’...   
a.2 Generate a python list named plan\_list where each subtask is an element of the plan\_list.

b. Write a ‘determination\_function’ for determine which subtask we are in based on Environment Information above.

b.1 For each subtask in the plan\_list, consider when each subtask is considered completed, express the condition in python code form. Typical completion conditions: -‘Move’ completion: distance under threshold -‘Grasp’ completion: object’s z-coordinate higher than its initialize z-coordinate.

b.2 Write a ‘determination\_function’ using ‘if’ and ‘elif’ to categorize the current subtask based on Environment Information - consider the logic of determining subtasks completion, using ‘if not’ ‘elif not’ if needed.

b.3 The ‘determination\_function’ return the current subtask’s index in plan\_list.

c.1 For each subtask, construct a progress metric named ‘subprogress’ indicating the progress of the corresponding subtask. The ‘subprogress’ should be range from 0 (no progress) to 1 (completed) Typical subprogress metric: the distance between the robot’s gripper and target object normalized by its initial distance;

c.2 Based on the ‘determination\_function’, add the ‘subprogress’ computation codes in the corresponding ‘if’ or ‘elif’ block.

c.3 Add the ‘main\_progress’ computation codes in each subtask’s ‘if’ or ‘elif’ block. - ‘main\_progress’ is the progress of the entire task. - typically, main\_progress can be calculated by adding the current subtask idx and current subprogress, regarding all completed subprogress as ‘1’. The completed progress can be set to ‘1’ without computing again.

c.4 ‘progress function’ return the main\_progress and the current subtask’s index in plan\_list.

## F Example of Generated Progress Function

In this section, to help better understand the generation ability and results, we provide zero-shot generated reward function on Metaworld. The response of the prompt defined in Appendix E is as follows:

LLM response in Metaworld-pick-place

```python
import numpy as np
# Plan the task into subtasks
def generate_plan_list ():
# Task decomposition : [ verb + noun ]
plan_list = [
’move ␣to␣ object ’,
’grasp ␣ object ’,
’move ␣to␣ goal ’,
’release ␣ object ’
]
return plan_list
# Determining the current subtask based on environment
information
def determination_function (env , plan_list ):
# Decompose the task completion into conditions for each
subtask
# Subtask 1: ’move to object ’
if np. linalg . norm( env. robot .tcp - env. obj. position ) > 0.3:
return 0 # Still moving towards the object
# Subtask 2: ’grasp object ’
elif env. obj . position [2] <= env .obj. obj_init_pos [2]:
return 1 # Object not yet grasped
```

```python
# Subtask 3: ’move to goal ’
elif np. linalg . norm( env. obj. position - env. goal_position ) >
0.07:
return 2 # Still moving towards the goal
# Subtask 4: ’release object ’
else :
return 3 # Task is completed
# Progress computation for each subtask
def progress_function (env):
plan_list = generate_plan_list ()
subtask_idx = determination_function (env , plan_list )
# Initialize variables
main_progress = 0.0 # Overall task progress
subprogress = 0.0 # Progress of the current subtask
# Subtask 1: ’move to object ’
if subtask_idx == 0:
# Progress of moving towards the object
initial_distance = np. linalg . norm(env. robot . tcp - env.
obj . obj_init_pos )
current_distance = np. linalg . norm(env. robot . tcp - env.
obj . position )
subprogress = 1 - ( current_distance / initial_distance )
main_progress = subtask_idx + subprogress # Task
progress is the sum of subtasks
# Subtask 2: ’grasp object ’
elif subtask_idx == 1:
# Progress of grasping the object ( object lifted above
initial position )
subprogress = ( env. robot . ee_position [2] - env .obj .
obj_init_pos [2]) / 0.5 # Normalize based on height
main_progress = subtask_idx + subprogress
# Subtask 3: ’move to goal ’
elif subtask_idx == 2:
# Progress of moving towards the goal position
initial_distance_to_goal = np . linalg . norm ( env . obj .
position - env. obj_init_pos )
current_distance_to_goal = np. linalg . norm(env. obj .
position - env. goal_position )
subprogress = 1 - ( current_distance_to_goal /
initial_distance_to_goal )
main_progress = subtask_idx + subprogress
# Subtask 4: ’release object ’
elif subtask_idx == 3:
# Task is complete , no further progress is needed
subprogress = 1.0
main_progress = 1.0
return main_progress , subtask_idx
```