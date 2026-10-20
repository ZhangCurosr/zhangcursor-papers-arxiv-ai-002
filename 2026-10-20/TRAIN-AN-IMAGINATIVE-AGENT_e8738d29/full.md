# TRAIN AN IMAGINATIVE AGENT

Jiaxin Ge<sup>1,∗</sup> Junyi Zhang<sup>1,∗</sup> Tianning Chai<sup>1,∗</sup> Hanjun Yoo<sup>1</sup> Chenyang Wang<sup>2</sup> Xiuyu Li<sup>2</sup> Weijia Shi<sup>3</sup> XuDong Wang<sup>1,4</sup> Sewon Min<sup>1</sup> Luke Zettlemoyer<sup>3</sup> Trevor Darrell<sup>1,†</sup> Haiwen Feng<sup>1,2,†</sup>

<sup>1</sup>University of California, Berkeley <sup>2</sup>Impossible Research

<sup>3</sup>University of Washington <sup>4</sup>Duke University

## ABSTRACT

A seemingly reasonable push in Sokoban can make the puzzle unsolvable. To avoid such mistakes, an agent has to anticipate the consequences of its actions before execution, especially when feedback from the real environment is sparse or actions are irreversible. We introduce Agentic Imagination (Agima), a training framework that trains an agent to imagine before acting, i.e. to be imaginative. Agima trains a single omni model to propose actions, generate images of their outcomes, and then reason over these imagined futures to decide what to execute in the real environment. Rather than committing to a single plan, the agent can simulate a sequence of actions, explore alternatives, and compare possible futures. We first use supervised fine-tuning (SFT) to unify policy and world modeling within a single omni model. This enables the agent to predict future outcomes and perform basic forms of imagination, such as forward thinking and counterfactual branching. We then use agentic reinforcement learning (RL) to teach the agent to reason with its imagined futures and use them effectively for planning. Across four environments spanning Maze2D, Sokoban, Pacman, and PushT, Agima generalizes to harder settings unseen during training. Trained with SFT alone on two-box Sokoban, Agima solves 40% of four-box puzzles while seeing the real board only once every five moves, compared with an action-only planner that only solves 2%. After RL, new behaviors emerge in how Agima uses its imagination: it learns a new unified format that was unseen during the SFT stage, learns to conduct forward thinking only when necessary, and branches more frequently at critical decision points. An agent that can picture where its actions lead can learn when and where to look ahead, that is, to plan. Project page: https://agima.net/.

## 1 INTRODUCTION

Agents take actions and receive observations from the environment (Sutton & Barto, 2018; Yao et al., 2023b), but making good decisions often requires anticipating the consequences of actions several steps ahead. When environment interactions are expensive or actions are irreversible, directly testing alternative actions through execution may not be feasible. For example, in Sokoban, a seemingly reasonable push can block another box and make the puzzle unsolvable. To avoid such failures, agents can predict and evaluate the consequences of possible actions before executing them (Racaniere et al. \` , 2017).

We introduce Agentic Imagination (Agima), a training framework that teaches an agent to imagine future observations of its actions and reason over them to plan before acting in the real environment. We build Agima on an omni model that unifies visual generation and language reasoning within the same multimodal context (Zhou et al., 2025; Deng et al., 2025). The same model serves as both the policy model that proposes actions and as the world model that imagines their visual states. These predictions become observations that the agent can use for further planning: it can extend a rollout, explore counterfactual actions, or compare longer candidate plans before execution (Figure 1). Importantly, imagined and real observations share the same interface and can be used interchangeably as inputs to the model. This allows the agent to use imagined observations in place of rea environment feedback to continue planning, while real observations can replace imagined ones after execution to ground the next round of planning.

![](images/0e51e591453eff47a8aaf561970419037e06bf3a355718bcc6a7a1beb5474ba4.jpg)

![](images/799cbbfb1bbbf30f27a628a0d8aa99a8db1469169a38393beb178fed33df34f5.jpg)

![](images/e4a15d604d12522ac3b914143f4f7ae151e90f80b90fffc6771719cfbccc54d7.jpg)  
Figure 1: Planning with agentic imagination. (a) A conventional agent receives feedback only after acting in the environment. (b) Agima uses a single model to propose actions and predict the consequences of them. Each prediction serves as an internal observation for the next decision. (c) Planning with Agima in Sokoban: counterfactual branching rejects a deadlocking action, and forward thinking compares candidate plans before execution. The executed actions reach $O _ { t + K } .$ which grounds the next round of imagination. In (c), $\hat { o } _ { j }$ is the predicted outcome of $\hat { a } _ { j }$

To teach the agent to imagine futures and plan with them, we construct training trajectories that interleave action proposals, visual predictions, and outcome evaluation. This allows the agent to unify policy, world modeling, and reward modeling within a single omni model. We also teach the agent basic ways to reason over its imagined futures for planning. A linear rollout follows a single action sequence, predicting the state after each step. Counterfactual branching explores alternative actions at individual steps and rejects those that lead to failures. Forward thinking imagines and compares longer candidate plans before deciding which one to execute. Together, these trajectories teach the agent to generate imagined futures and use them to evaluate different actions before execution. We use these trajectories for supervised fine-tuning (SFT).

Once the agent can imagine futures, we want it to learn how to reason over them more effectively, rather than simply following predefined reasoning formats. Recent progress in reinforcement learning (RL) for reasoning and tool use in language models (Kumar et al., 2025; Jin et al., 2025) shows that models can learn new reasoning behaviors from outcome rewards. We adapt this idea to Agima using GRPO (Shao et al., 2024). During RL, the agent imagines and evaluates candidate plans, executes its selected actions in the real environment, and receives a binary reward based on task success. We optimize the agent’s reasoning in language space and keeps its generated visual observations as context for planning. Unlike SFT, we do not provide additional expert demonstrations or supervise how the agent should organize its imagination. This allows the agent to internalize adaptive reasoning strategies. It learns from task success when to explore another plan, which alternatives to consider, and when to proceed with its current plan.

We evaluate Agima in Maze2D (Wang et al., 2026), Sokoban and Pacman (Racaniere et al.\` , 2017), and PushT (Florence et al., 2022; Chi et al., 2024), training on simpler instances and testing on harder settings excluded from training. With SFT alone, Agima directly generalizes from two-box to fourbox Sokoban without further training. With real observations every five actions, it achieves 40% success, compared with 2% for an action-only planner fine-tuned from the same backbone. Forward thinking further improves long-horizon planning in Pacman using the same training instances as linear rollouts. Agentic RL further improves task success and leads to new behaviors in how Agima uses its imagination. For forward thinking, the agent learns a new reasoning format unseen during SFT: it first imagines a single plan and only explores another candidate when the first is predicted to fail. For counterfactual branching, the agent explores more alternatives and increasingly focuses its branches on critical decisions where a wrong action could make the puzzle unsolvable. These behaviors emerge from task-level rewards without additional supervision on how to organize its imagination.

![](images/69e8da3e4ce0e28404e9f8d13d9312a45e36e872866a7d8a37b07ca3d3bb5620.jpg)

![](images/c1e463db317bd51baa998437963056c547fc212377f089891501cdfdb3b614c4.jpg)  
Figure 2: Evaluation environments and imagined visual rollouts. (a) Training and test states in four environments. (b) Agima interleaves actions and predicted states, then executes the actions.

Our contributions are threefold. First, we introduce Agima, a training framework for agentic imagination that teaches agents to explore, evaluate, and revise plans using predicted visual observations before acting. Second, we show that agentic imagination generalizes to out-of-domain settings across four environments, with further gains from learning to compare imagined futures using the same underlying training instances. Third, we show that agentic RL teaches Agima to organize its own imagination through task-success rewards, eliciting adaptive reasoning strategies that determine when to explore another plan and where to examine alternative actions.

## 2 LEARNING AGENTIC IMAGINATION

Agentic imagination (Agima) trains an agent to imagine the future observations of its actions and reason over these imagined futures before acting in the environment. Given the current observation and a goal, the agent proposes an action, generates an image of its predicted outcome, and uses this imagined observation to continue planning. We build Agima on a single omni model that unifies language reasoning and visual generation. This allows the same model to both propose actions and predict their consequences, serving as both the agent policy and the world model. Our training framework has two stages. First, supervised fine-tuning (SFT) teaches the agent to unify policy and world modeling within a single model, allowing the agent to imagine future observations and perform basic reasoning over its imagined futures through different planning traces. Second, reinforcement learning (RL) teaches the agent to use its imagination more effectively for planning, learning when to explore additional futures and which alternatives to consider using only task-level rewards.

## 2.1 PLANNING WITH IMAGINED OBSERVATIONS

Consider an agent at time t with visual observation $o _ { t }$ and goal g. Under action chunking (Zhao et al., 2023), an action-only agent proposes a sequence of K actions:

$$
A _ { t } ^ { ( K ) } = \left( \widehat { a } _ { t } , \widehat { a } _ { t + 1 } , \ldots , \widehat { a } _ { t + K - 1 } \right) .
$$

Agentic imagination interleaves proposed actions with predicted future observations:

$$
\tau _ { t } ^ { ( K ) } = \left( \widehat { a } _ { t } , \widehat { o } _ { t + 1 } , \widehat { a } _ { t + 1 } , \widehat { o } _ { t + 2 } , \ldots , \widehat { a } _ { t + K - 1 } , \widehat { o } _ { t + K } \right) .
$$

Each predicted state becomes an internal observation for subsequent planning. The agent can therefore propose its next action from imagined consequences without waiting for real feedback. Once a plan is selected, its proposed actions are executed as $a _ { t + j } = \widehat { a } _ { t + j }$ . The resulting real observation then serves as the starting point for the next round of imagination. Beyond a single rollout, the agent can also explore alternative actions, return to previous plans, and compare different imagined futures before deciding what to execute. We next describe how we train the agent to perform these different forms of imagination through supervised fine-tuning.

Table 1: Performance comparison across Pacman, Maze, Sokoban, and PushT environments. K indicates the number of actions generated before feedback. Reactive (K = 1) receives groundtruth feedback after every action. Pacman, Maze, and Sokoban use the hard splits. For PushT, we report success accuracy and final coverage on the far split. A7B denotes 7B active parameters (14B total); No Imagine is trained to predict actions without intermediate visual predictions. Bold indicates the best result at each K (including ties); gains are relative to Agima-A7B (No Imagine).
<table><tr><td rowspan="2">K Group Method</td><td rowspan="2"></td><td rowspan="2">Pacman</td><td rowspan="2">Maze</td><td rowspan="2">Sokoban</td><td colspan="2">PushT</td></tr><tr><td>Acc.</td><td>Final Cov.</td></tr><tr><td rowspan="3">1 SFT</td><td>Gemini 3.1 Pro Closed GPT-5.6 Sol</td><td>0%</td><td>38%</td><td>3%</td><td>0%</td><td>0.2%</td></tr><tr><td></td><td>0%</td><td>48%</td><td>32%</td><td>0%</td><td>1.8%</td></tr><tr><td>Agima-A7B</td><td>84%</td><td>62%</td><td>47%</td><td>42%</td><td>45.9%</td></tr><tr><td rowspan="5">3 SFT</td><td>Closed GPT-5.6 Sol</td><td>Gemini 3.1 Pro</td><td>0%</td><td>21% 4%</td><td>0%</td><td>2.2%</td></tr><tr><td>Qwen2.5-VL-7B</td><td>0%</td><td>39%</td><td>26%</td><td>0%</td><td>3.0%</td></tr><tr><td></td><td>0%</td><td>40%</td><td>4%</td><td>4%</td><td>8.48%</td></tr><tr><td>Qwen3-VL-8B</td><td>0%</td><td>85%</td><td>10%</td><td>6%</td><td>11.77%</td></tr><tr><td>Agima-A7B (No Imagine) Agima-A7B</td><td>25% 38% (+13%)</td><td>54% 59% (+5%)</td><td>10% 53% (+43%)</td><td>12% 24% (+12%)</td><td>18.6% 31.7% (+13.1%)</td></tr><tr><td rowspan="5">5 SFT</td><td rowspan="5">Closed</td><td>Gemini 3.1 Pro</td><td>0%</td><td>20%</td><td>6%</td><td>0%</td><td>1.8%</td></tr><tr><td>GPT-5.6 Sol</td><td>1%</td><td>34%</td><td>20%</td><td>1%</td><td>8.3%</td></tr><tr><td>Qwen2.5-VL-7B</td><td>0%</td><td>28%</td><td>0%</td><td>2%</td><td>4.07%</td></tr><tr><td>Qwen3-VL-8B</td><td>0%</td><td>30%</td><td>3%</td><td>1%</td><td>3.41%</td></tr><tr><td>Agima-A7B (No Imagine)</td><td>1%</td><td>23%</td><td>2%</td><td>1%</td><td>6.7%</td></tr><tr><td rowspan="5"></td><td></td><td>Agima-A7B</td><td>37% (+36%)</td><td>40% (+17%)</td><td>40% (+38%)</td><td>18% (+17%)</td><td>28.5% (+21.8%)</td></tr><tr><td rowspan="5">Closed 10</td><td>Gemini 3.1 Pro</td><td>0%</td><td>15%</td><td>4%</td><td>0%</td><td>2.0%</td></tr><tr><td>GPT-5.6 Sol</td><td>1%</td><td>34%</td><td>23%</td><td>0%</td><td>8.7%</td></tr><tr><td>Qwen2.5-VL-7B</td><td>0%</td><td>0%</td><td>0%</td><td>2%</td><td>5.36%</td></tr><tr><td>Qwen3-VL-8B</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>5.55%</td></tr><tr><td rowspan="3">SFT</td><td>Agima-A7B (No Imagine)</td><td>0%</td><td>2%</td><td>0%</td><td>1%</td><td>5.3%</td></tr><tr><td>Agima-A7B</td><td>7% (+7%)</td><td>21% (+19%)</td><td>15% (+15%)</td><td>9% (+8%)</td><td>22.2% (+16.9%)</td></tr><tr><td>Gemini 3.1 Pro</td><td>0%</td><td>9%</td><td>13%</td><td>0%</td><td>0.8%</td></tr><tr><td rowspan="5">∞ SFT</td><td rowspan="5">Closed</td><td>GPT-5.6 Sol</td><td>2%</td><td>36%</td><td>21%</td><td>0%</td><td>5.5%</td></tr><tr><td>Qwen2.5-VL-7B</td><td>0%</td><td>0%</td><td>0%</td><td>2%</td><td>5.60%</td></tr><tr><td>Qwen3-VL-8B</td><td>0%</td><td>0%</td><td>0%</td><td>3%</td><td>5.87%</td></tr><tr><td>Àgima-A7B (No Imagine)</td><td>0%</td><td>0%</td><td>0%</td><td>3%</td><td>6.2%</td></tr><tr><td>Agima-A7B</td><td>6% (+6%)</td><td>3% (+3%)</td><td>19% (+19%)</td><td>3%</td><td>8.3% (+2.1%)</td></tr></table>

## 2.2 SUPERVISED LEARNING FROM IMAGINATION TRACES

We build Agima on the BAGEL backbone (Deng et al., 2025), an omni model that can understand and generate both images and text. The backbone has 14B total parameters and 7B active parameters, denoted by A7B in our model names. During SFT, we train on interleaved sequences of text and images such as the imagined rollout in Section 2.1. The current observation and goal are provided as input, and the remaining text and images are generated sequentially under teacher forcing. We apply next-token cross-entropy to text outputs and the image-generation model’s MSE objective to image outputs, combining them in the SFT loss:

$$
\mathcal { L } _ { \mathrm { C E } } = - \sum _ { i \in \mathrm { t e x t } } \log p \theta ( x _ { i } \mid x _ { < i } ) , \quad \mathcal { L } _ { \mathrm { M S E } } = \sum _ { j \in \mathrm { i m a g e } } \Vert \hat { v } _ { j } - v _ { j } \Vert _ { 2 } ^ { 2 } , \quad \mathcal { L } _ { \mathrm { S F T } } = \mathcal { L } _ { \mathrm { C E } } + \lambda _ { \mathrm { i m g } } \mathcal { L } _ { \mathrm { M S E } } .
$$

Training on these interleaved sequences teaches the model to both predict future observations and reason over them to decide subsequent actions. We design three types of imagination traces that differ in how the agent explores alternative futures.

Linear Rollout. A linear rollout alternates proposed actions and predicted observations:

$$
[ o _ { t } , g ]  \hat { a } _ { t }  \hat { o } _ { t + 1 }  \hat { a } _ { t + 1 }  \hat { o } _ { t + 2 }  \cdots .
$$

Table 2: Replacing imagined observations at inference. No Imagine is a separately trained action-only baseline; the four Agima-A7B conditions share checkpoint, episodes, and decoding within each environment. Blank replaces predictions with gray images, Stale with the latest real observation, and Oracle GT with ground-truth frames for the proposed actions.
<table><tr><td>Model</td><td>Condition</td><td>Sokoban</td><td>Pacman</td><td>Maze2D</td><td>PushT</td></tr><tr><td>No Imagine</td><td></td><td>2%</td><td>1%</td><td>23%</td><td>1%</td></tr><tr><td rowspan="4">Agima-A7B</td><td>Own predictions</td><td>40%</td><td>37%</td><td>40%</td><td>18%</td></tr><tr><td>Blank</td><td>1%</td><td>0%</td><td>8%</td><td>1%</td></tr><tr><td>Stale</td><td>0%</td><td>0%</td><td>8%</td><td>0%</td></tr><tr><td>Oracle GT</td><td>43%</td><td>81%</td><td>69%</td><td>47%</td></tr></table>

Each imagined observation is fed back into the sequence before the next action is generated, allowing the model to continue planning without a new observation from the environment.

Counterfactual Branching. A linear rollout commits to each action as it develops a plan. Counterfactual branching instead checks local alternatives while maintaining a primary rollout. At selected imagined states, the model tries a different action, predicts its immediate consequence, rejects the alternative if it is undesirable, and then continues along the main trajectory:

$$
[ o _ { t } , g ]  \hat { a } _ { t }  \hat { o } _ { t + 1 }  \{ \begin{array} { l l } { { \hat { a } _ { t + 1 } ^ { \prime }  \hat { o } _ { t + 2 } ^ { \prime }  \mathrm { r e j e c t } , } } \\ { { \hat { a } _ { t + 1 }  \hat { o } _ { t + 2 }  \dots  \hat { o } _ { j }  \{ \hat { a } _ { j } ^ { \prime }  \hat { o } _ { j + 1 } ^ { \prime }  \mathrm { r e j e c t } ,  } }  \end{array} 
$$

A single rollout may contain multiple such checks. The main trajectory follows the current plan, while the counterfactual branches explore alternative actions that may lead to failure.

Forward Thinking. While counterfactual branching explores local alternative actions at individual steps along a plan, forward thinking allows the agent to conduct long-horizontal imagination and compare different complete plans before deciding what to execute. Each candidate is rolled forward over a longer horizon:

$$
[ o _ { t } , g ]  \{ \begin{array} { l l } { \hat { a } _ { t } ^ { ( 1 ) }  \hat { o } _ { t + 1 } ^ { ( 1 ) }  \hat { a } _ { t + 1 } ^ { ( 1 ) }  \hat { o } _ { t + 2 } ^ { ( 1 ) }  \cdots , } \\ { \hat { a } _ { t } ^ { ( 2 ) }  \hat { o } _ { t + 1 } ^ { ( 2 ) }  \hat { a } _ { t + 1 } ^ { ( 2 ) }  \hat { o } _ { t + 2 } ^ { ( 2 ) }  \cdots . } \end{array} 
$$

The model can therefore compare how different plans unfold before choosing which one to execute.

## 2.3 REINFORCEMENT LEARNING FOR ADAPTIVE IMAGINATION

SFT teaches the model how to generate and reason with imagination traces, but the demonstrations also specify which alternatives to explore and compare. We use RL to optimize these choices directly from task success. Starting from the SFT model, we apply GRPO (Shao et al., 2024). For each problem, the model generates a complete planning trace and final executable actions without receiving additional observations from the real environment. The selected actions are then executed, producing a binary task-success reward. We obtain group-relative advantages by standardizing rewards within each group of sampled traces. GRPO updates the language tokens in the planning trace; generated images remain in the multimodal context but receive no direct RL loss. We use a group size of 8 and batch size of 16. Additional optimization details are provided in Appendix B.

## 3 GENERALIZATION AND PLANNING WITH IMAGINATION

We evaluate two capabilities enabled by agentic imagination. First, we ask whether imagined observations can support multi-step planning without frequent feedback from the environment, and whether this ability generalizes beyond the configurations seen during training. Second, we ask whether the agent can use imagined futures not only to extend a rollout, but also to evaluate alternatives before acting. We study these questions across four environments with qualitatively dif ferent dynamics: Maze2D (Wang et al., 2026) for spatial navigation, Sokoban (Racaniere et al. \` , 2017) for irreversible object interactions, Pacman (Racaniere et al.\` , 2017) for reactive dynamics, and PushT (Florence et al., 2022; Chi et al., 2024) for continuous contact dynamics. Figure 2 illustrates the environments and an example imagined rollout. Full environment details are provided in Appendix A.

![](images/19a159c153a0d1f3a98777a77c247813c26fbec0dd33578a98b1f4d32174749e.jpg)  
Figure 3: Forward thinking compares alternative futures before execution. A Pacman example (Table 3(b)): (a) Agima-A7B is caught after executing a single rollout; (b) Agima-A7B (Forward) compares two imagined futures and selects Candidate 1, which succeeds.

## 3.1 AGIMA ENABLES EASY-TO-HARD GENERALIZATION

Setup. We first evaluate linear imagination rollouts. For each environment, we train on simpler task configurations and evaluate on harder or unseen configurations excluded from training (Figure 2(a)). Let K denote the number of actions planned and executed before the next ground-truth observation is provided. The fully reactive setting K = 1 provides a reference with feedback after every action. At K = ∞, no intermediate environment observation is provided during the rollout. We denote the model trained with linear imagination rollouts as Agima-A7B. Our primary controlled baseline is Agima-A7B (No Imagine), which is separately fine-tuned from the same BAGEL backbone on the same task data, but predicts action chunks without intermediate visual states. We also supervised fine-tune Qwen2.5-VL-7B (Bai et al., 2025b) and Qwen3-VL-8B (Bai et al., 2025a) on the same task data as additional action-only baselines. Finally, we evaluate Gemini 3.1 Pro (Google DeepMind, 2026) and GPT-5.6 Sol (OpenAI, 2026) as proprietary baselines.

Imagination Transfers to Harder Settings and Improves Planning. Table 1 shows that Agima transfers from simpler training instances to harder and unseen settings across all four environments, especially when planning multiple actions without real feedback. At $K = 5 ,$ , Agima improves over the action-only model by 36 points in Pacman, 17 points in Maze2D, 38 points in Sokoban, and 17 points in PushT. For example, after training on two-box Sokoban, Agima reaches 40% success on four-box puzzles at K = 5, compared with 2% for the action-only model. Performance eventually degrades for very long open-loop planning: at $K = \infty$ , both absolute performance and the advantage over action-only planning shrink in several environments.

Agima Relies on the Content of Its Imagined Observations. Replacing imagined observations with blank or stale frames sharply reduces success across all four environments (Table 2). Replacing predicted frames with ground-truth frames improves success by 29-44 percentage points in Pacman, Maze2D, and PushT, and by 3 points in Sokoban. These results show that Agima actually uses the content of its imagined observations for planning and benefits from more accurate predictions.

## 3.2 REASONING OVER ALTERNATIVE IMAGINED FUTURES

Linear rollouts use imagined observations to extend a single future. We next test whether the same learned world model can support richer planning operations: probing local counterfactual actions with branching and comparing long-horizon candidate futures with forward thinking.

Setup. All experiments use $K = \infty$ , without intermediate observations from the real environment. Agima-A7B (Branching) and Agima-A7B (Forward) use the same training instances as the linearrollout model, differing only in the structure of their imagination traces. The underlying training distribution remains unchanged. In Sokoban, models trained on standard two-box puzzles are evalu ated on two-, three-, and four-box Deadlock and Structural Fork sets, testing delayed consequences and critical action choices, respectively. In Pacman, we evaluate route-order traps requiring longhorizon plan comparison on out-of-domain grid sizes. Training traces and evaluation details are provided in Appendices B and A.1.

Table 3: Success rates on Sokoban and Pacman reasoning tasks. (a) Sokoban Deadlock and Structural Fork configurations with 2, 3, and 4 boxes. (b) Pacman route-order traps, where G denotes grid size, and F denotes the number of food items. Bold indicates the best result in each column; changes in parentheses are absolute percentage-point differences relative to Agima-A7B.
<table><tr><td rowspan="2">(a) Sokoban Method</td><td colspan="3">Deadlock</td><td colspan="3">Structural Fork</td></tr><tr><td>2-box</td><td>3-box</td><td>4-box</td><td>2-box</td><td>3-box</td><td>4-box</td></tr><tr><td>Gemini 3.1 Pro</td><td>74%</td><td>42%</td><td>19%</td><td>84%</td><td>34%</td><td>7%</td></tr><tr><td>GPT-5.6 Sol</td><td>44%</td><td>23%</td><td>17%</td><td>60%</td><td>20%</td><td>12%</td></tr><tr><td>Qwen2.5-VL-7B</td><td>1%</td><td>2%</td><td>0%</td><td>12%</td><td>2%</td><td>0%</td></tr><tr><td>Qwen3-VL-8B</td><td>6%</td><td>1%</td><td>1%</td><td>19%</td><td>2%</td><td>0%</td></tr><tr><td>Agima-A7B (No Imagine)</td><td>12%</td><td>5%</td><td>1%</td><td>15%</td><td>2%</td><td>1%</td></tr><tr><td>Agima-A7B</td><td>22%</td><td>39%</td><td>27%</td><td>87%</td><td>34%</td><td>22%</td></tr><tr><td>Agima-A7B (Branching)</td><td>30% (+8%)</td><td>39% (+0%)</td><td>32% (+5%)</td><td>84% (-3%)</td><td>39% (+5%)</td><td>32% (+10%)</td></tr><tr><td>Agima-A7B (Forward)</td><td>60% (+38%)</td><td>48% (+9%)</td><td>24% (-3%)</td><td>93% (+6%)</td><td>46% (+12%)</td><td>22% (+0%)</td></tr><tr><td>(b) Pacman</td><td></td><td></td><td>G8</td><td></td><td></td><td>G10</td></tr><tr><td>Method</td><td>F2</td><td>F4</td><td></td><td>F6</td><td>F8</td><td>F8</td></tr><tr><td>Gemini 3.1 Pro</td><td>1%</td><td>1%</td><td></td><td>0%</td><td>1%</td><td>0%</td></tr><tr><td>GPT-5.6 Sol</td><td>16%</td><td>19%</td><td></td><td>6%</td><td>8%</td><td>0%</td></tr><tr><td>Qwen2.5-VL-7B</td><td>0%</td><td>0%</td><td></td><td>0%</td><td>0%</td><td>0%</td></tr><tr><td>Qwen3-VL-8B</td><td>0%</td><td>0%</td><td></td><td>0%</td><td>1%</td><td>0%</td></tr><tr><td>Agima-A7B (No Imagine)</td><td>3%</td><td>0%</td><td></td><td>2%</td><td>0%</td><td>0%</td></tr><tr><td>Agima-A7B</td><td>7%</td><td>22%</td><td></td><td>16%</td><td>16%</td><td>5%</td></tr><tr><td>Agima-A7B (Forward)</td><td>40% (+33%)</td><td>54% (+32%)</td><td></td><td>41% (+25%)</td><td>37% (+21%)</td><td>12% (+7%)</td></tr></table>

Counterfactual Branching Helps at Critical Decision Forks. Counterfactual branching allows the agent to explore alternative actions at each step along the main trajectory. It is particularly helpful in the harder Structural Fork settings, where the agent encounters locally states that a seemingly reasonable action can lead to failure. Compared with the linear-rollout model, branching improves three-box success from 34% to 39% and four-box success from 22% to 32% (Table 3(a)). These results show that Agima can use imagined states to probe local alternatives that may lead to failure before actually committing to the next step.

Forward Thinking Improves Long-Horizon Plan Selection. Forward thinking allows the agent to imagine and compare different futures before execution, helping it avoid failures that only become apparent several steps later. On Sokoban Deadlock, forward thinking improves success from 22% to 60% on two-box puzzles and from 39% to 48% on three-box puzzles. The effect is particularly consistent on Pacman route-order traps: forward-thinking supervision improves success across every evaluated configuration, including from 22% to 54% on G8/F4 and from 5% to 12% on G10/F8 (Table 3(b)). Figure 3 illustrates why long-horizon comparison matters. With linear rollout, the model directly chooses the more tempting option: collecting the nearer food first, which appears safe because the ghost is still far away. However, when Pacman turns toward the remaining food, it can no longer escape the ghost and eventually dies. With forward thinking, Agima imagines alternative futures that collect the food items in different orders and discovers that collecting the farther food first leads to a successful outcome.

## 4 ADAPTIVE REASONING THROUGH AGENTIC RL

SFT teaches Agima to construct and reason over imagined futures, but the training traces follow predefined heuristics. Since Agima combines language-space reasoning with visual prediction, we can apply agentic RL to improve how the agent uses its imagination for planning. Specifically, we ask whether task-level rewards can teach the agent when to think ahead, when to reconsider a plan, and where to explore counterfactual alternatives.

Setup. We initialize RL from four SFT checkpoints trained on two-box Sokoban: Agima-A7B (No Imagine), Agima-A7B, Agima-A7B (Forward), and Agima-A7B (Branching). Agima-A7B (No Imagine) provides a no-imagination baseline, generating actions without intermediate visua predictions. For the latter two, we use mixed-format checkpoints trained on random mixtures of ∼50% linear rollouts and ∼50% of the corresponding reasoning traces. This keeps their pre-RL performance close to the linear-rollout checkpoint while letting the model being able to generate multiple reasoning formats, leaving RL to learn when and how to use them. Because ordinary two box Sokoban is nearly saturated after SFT, we perform RL on ordinary three-box puzzles. The agent receives only a binary task-success reward, with no additional expert demonstrations or trace-level supervision. We train for 200 RL steps using GRPO, and evaluate on held-out three- and four-box Deadlock and Structural Fork sets. Additional details are provided in Appendix B.4.

![](images/50c4be4082056aef976ff3b38a9d0f420b380db552593073b593865398104cb4.jpg)

![](images/d5446f57b544cff1a0d31bbfdf13c43801160e3162263d08a681b2303f5c9035.jpg)

![](images/ca9c08dc1ce53587fae06f6d6e2568428e3b5961998098e1ab8f9dca5205466f.jpg)  
Agima-A7B (No Imagine) Agima-A7B Agima-A7B (Forward) Agima-A7B (Branching) Size (b, c): 3 boxes 4 boxes

Figure 4: RL improves training reward and held-out planning performance. (a) Training reward during RL. (b–c) Accuracy on the Deadlock and Structural Fork evaluation sets. Solid and dashed lines denote three-box and four-box tasks, respectively.  
![](images/9aac99cb19a06ecbd557d3b41a0991f4816be9debf4290c9c1d970b89a91204f.jpg)  
Figure 5: RL induces a unified, adaptive planning format. (a) SFT produces a linear rollout or a fixed two-candidate comparison. (b) After RL, the model evaluates each imagined plan and generates alternatives as needed, here rejecting two failed plans before finding a successful third.

RL Improves Planning Beyond SFT. Figure 4 shows that RL improves training reward for all four initializations. Even Agima-A7B (No Imagine) rises from near-zero reward to roughly 15% by the end of training, showing that task reward can improve the action policy. However, this gain transfers little to the held-out Deadlock and Structural Fork sets, where its accuracy remains near zero. In contrast, all three imagination-based initializations improve substantially on both training reward and held-out planning performance, including the four-box settings never seen during RL. The gain is particularly strong for Agima-A7B (Branching): despite starting near the linear-rollout Agima-A7B checkpoint, RL eventually pushes it beyond Agima-A7B on Structural Fork.

Forward Thinking Becomes Adaptive. RL teaches Agima when to explore alternative plans in forward thinking. During SFT, the training data only contains two predefined reasoning formats: (1) a linear rollout, or (2) a forward thinking trace that decides from the beginning to compare two candidates. After 100 steps of RL training, a new format emerges that combines these two behaviors: the agent first rolls out a plan and evaluates its predicted outcome. If the plan is predicted to succeed, it ends the reasoning trace and proceeds with the plan; otherwise, it generates another candidate (Figure 5, Appendix C.1). Meanwhile, the model also learns when additional exploration is necessary. When a second candidate is generated, the first plan fails in 100% of cases, compared to 32% before RL. This suggests that the agent learns to reason over its predicted outcomes and only uses forward thinking when needed, rather than simply following the predefined formats seen during SFT. The agent also learns to explore beyond the two-candidate format seen during SFT which is trained on two-box puzzles. As shown in Figure 5, in the three-box setting, it learns to explore a third candidate after the first two plans fail.

![](images/a5f631d616fa53618ba7700be18b320cf3b9a0a585054c9270a3dee84dcff19c.jpg)

![](images/3e9524e8e88c91b406fde70faa6614d4c9cd86f3938d4a96ac099a3943479a61.jpg)

![](images/101c73f6a2d9ae2302ab728aec49ec0fea50e74a1b5aee16540b712f44a51e6d.jpg)

![](images/e6bc017bb54bd1af018e2371853863416b43d12aa68f7785843d0390b1348d7d.jpg)  
Agima-A7B (Forward): Deadlock Agima-A7B (Branching): Fork 3 boxes 4 boxes  
Figure 6: RL induces adaptive use of imagination. Forward thinking becomes increasingly failuretriggered, while branching explores more alternatives and shifts toward critical decision points.

Branching Becomes More Exploratory and Targeted. RL also changes the branching behavior of Agima. The model explores more alternatives once branching begins, with the average number of branches per trace increasing from 1.70 before RL to 1.95 after 150 steps of RL training. We define Critical Branch Rate as the fraction of branches opened at critical states, where at least one alternative action would make the puzzle unsolvable. As shown in Figure 6, this rate increases after RL, indicating that counterfactual reasoning becomes more concentrated on decisions where the choice of action matters. Meanwhile, the total number of imagined frames does not increase (Appendix C.3), suggesting that RL teaches the agent to use branching more effectively to avoid potential failures without increasing its overall imagination budget.

## 5 RELATED WORK

Model-based planning. Model-based reinforcement learning learns a model of the environment dynamics and uses it to plan or to train a policy (Sutton, 1991; Ha & Schmidhuber, 2018; Hafner et al., 2019b;a; Schrittwieser et al., 2020). I2A (Racaniere et al.\` , 2017) is particularly relevant here: it uses learned environment rollouts to inform a model-free policy on Sokoban and MiniPacman. These tasks share the irreversible decisions and long planning horizons of our setting. Language models have also been combined with tree search, where they act as the policy, the world model, or the evaluator (Hao et al., 2023; Yao et al., 2023a), and recent video world models generate high fidelity future frames conditioned on actions (Bruce et al., 2024; Parker-Holder & Fruchter, 2025; Valevski et al., 2025; Alonso et al., 2024; Yang et al., 2024; NVIDIA et al., 2025; Bar et al., 2025). I2A leaves the choice of imagined actions to a rollout policy, while the tree-search methods use an external search procedure. In Agima, action proposals and visual predictions come from the same model. Each prediction stays in the context for subsequent reasoning, so choosing whether to extend a rollout, try an alternative, or reject a plan becomes part of the generated trace.

Reasoning with generated images. Several recent methods include images in the reasoning process of multimodal models. Visual Sketchpad (Hu et al., 2024) calls tools to draw auxiliary lines and boxes, MVoT (Li et al., 2025) generates images of intermediate states in spatial reasoning, and ThinkMorph (Gu et al., 2026) alternates between text and image thoughts. Other work keeps the visual thought in a latent space (Zhang et al., 2025), and Su et al. (2025) provide a survey. In these methods, the image helps the model reach a final answer, and there is usually no single correct image to generate. UniWM (Dong et al., 2026) and WorldVLA (Cen et al., 2025) jointly predict actions and future images for navigation and manipulation, but they follow a single predicted trajectory and do not use the predicted images to compare alternative actions. In our work, each generated image is the predicted result of a specific action, and the model learns to carry out the planning steps itself, including counterfactual branching and forward thinking (Section 2).

Reinforcement learning for VLM agents. Reinforcement learning with outcome rewards has improved the reasoning of language models (Shao et al., 2024; Guo et al., 2025) and has been applied to multi-turn LLM agents (Wang et al., 2025b; Feng et al., 2025) and to VLM agents that act from visual observations (Zhai et al., 2024). Two works are closest to ours. VAGEN (Wang et al., 2025a) trains a VLM agent to describe the current state and predict the next state in its textual reasoning, and rewards the accuracy of these predictions; Sokoban is among its tasks. Visual Planning (Xu et al., 2026) represents a plan as a sequence of generated images and rewards each generated state. Neither method trains an agent to reason in language about its own visual predictions. We apply RL with a binary task-success reward to an agent of this kind. Only the text tokens are optimized. These tokens carry the agent’s planning decisions, such as whether to imagine another plan and where to branch, while the generated images remain in the context as observations.

## 6 CONCLUSION

We present Agima, a framework that teaches agents to plan with predicted visual observations. Across four environments, Agima generalizes beyond training configurations and supports multistep planning without intermediate feedback, counterfactual branching, and comparison of candidate futures. With agentic RL, Agima learns new behaviors, including a new reasoning format, conditioning forward thinking only on failure, and focusing branching on critical decisions. Our results show that a single omni model can be trained as an imaginaive agent that can imagine possible futures and reason over them to guide its own actions.

## ACKNOWLEDGMENT

We thank Angjoo Kanazawa, Ranjay Krishna, Joey Gonzalez, Rulin Shao, Zixian Ma, Xiaochuang Han, and Baifeng Shi for feedback on our manuscript.

## REFERENCES

Eloi Alonso, Adam Jelley, Vincent Micheli, Anssi Kanervisto, Amos Storkey, Tim Pearce, and Franc¸ois Fleuret. Diffusion for World Modeling: Visual Details Matter in Atari. In A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang (eds.), Advances in Neural Information Processing Systems, volume 37, pp. 58757–58791. Curran Associates, Inc., 2024. doi: 10.52202/ 079017-1873. URL https://proceedings.neurips.cc/paper\_files/paper/ 2024/file/6bdde0373d53d4a501249547084bed43-Paper-Conference.pdf.

Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, Wenbin Ge, Zhifang Guo, Qidong Huang, Jie Huang, Fei Huang, Binyuan Hui, Shutong Jiang, Zhaohai Li, Mingsheng Li, Mei Li, Kaixin Li, Zicheng Lin, Junyang Lin, Xuejing Liu, Jiawei Liu, Chenglong Liu, Yang Liu, Dayiheng Liu, Shixuan Liu, Dunjie Lu, Ruilin Luo, Chenxu Lv, Rui Men, Lingchen Meng, Xuancheng Ren, Xingzhang Ren, Sibo Song, Yuchong Sun, Jun Tang, Jianhong Tu, Jianqiang Wan, Peng Wang, Pengfei Wang, Qiuyue Wang, Yuxuan Wang, Tianbao Xie, Yiheng Xu, Haiyang Xu, Jin Xu, Zhibo Yang, Mingkun Yang, Jianxin Yang, An Yang, Bowen Yu, Fei Zhang, Hang Zhang, Xi Zhang, Bo Zheng, Humen Zhong, Jingren Zhou, Fan Zhou, Jing Zhou, Yuanzhi Zhu, and Ke Zhu. Qwen3-VL Technical Report, 2025a. URL https://arxiv.org/abs/2511.21631.

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-VL Technical Report, 2025b. URL https://arxiv.org/abs/2502.13923.

Amir Bar, Gaoyue Zhou, Danny Tran, Trevor Darrell, and Yann LeCun. Navigation World Models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 15791–15801, June 2025.

Jake Bruce, Michael D Dennis, Ashley Edwards, Jack Parker-Holder, Yuge Shi, Edward Hughes, Matthew Lai, Aditi Mavalankar, Richie Steigerwald, Chris Apps, Yusuf Aytar, Sarah Maria Elisabeth Bechtle, Feryal Behbahani, Stephanie C.Y. Chan, Nicolas Heess, Lucy Gonzalez, Simon Osindero, Sherjil Ozair, Scott Reed, Jingwei Zhang, Konrad Zolna, Jeff Clune, Nando De Freitas, Satinder Singh, and Tim Rocktaschel. Genie: Generative Interactive Environments. In Ruslan¨ Salakhutdinov, Zico Kolter, Katherine Heller, Adrian Weller, Nuria Oliver, Jonathan Scarlett, and Felix Berkenkamp (eds.), Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pp. 4603–4623. PMLR, 21–27 Jul 2024. URL https://proceedings.mlr.press/v235/bruce24a.html.

Jun Cen, Chaohui Yu, Hangjie Yuan, Yuming Jiang, Siteng Huang, Jiayan Guo, Xin Li, Yibing Song, Hao Luo, Fan Wang, Deli Zhao, and Hao Chen. WorldVLA: Towards Autoregressive Action World Model, 2025. URL https://arxiv.org/abs/2506.21539.

Cheng Chi, Zhenjia Xu, Siyuan Feng, Eric Cousineau, Yilun Du, Benjamin Burchfiel, Russ Tedrake, and Shuran Song. Diffusion policy: Visuomotor policy learning via action diffusion. The International Journal of Robotics Research, 44(10-11):1684–1704, October 2024. ISSN 1741-3176. doi: 10.1177/02783649241273668. URL http://dx.doi.org/10.1177/ 02783649241273668.

Chaorui Deng, Deyao Zhu, Kunchang Li, Chenhui Gou, Feng Li, Zeyu Wang, Shu Zhong, Weihao Yu, Xiaonan Nie, Ziang Song, Guang Shi, and Haoqi Fan. Emerging Properties in Unified Multimodal Pretraining, 2025. URL https://arxiv.org/abs/2505.14683.

Yifei Dong, Fengyi Wu, Guangyu Chen, Lingdong Kong, Xu Zhu, Qiyu Hu, Yuxuan Zhou, Jingdong Sun, Jun-Yan He, Qi Dai, Alexander G. Hauptmann, and Zhi-Qi Cheng. Towards Unified World Models for Visual Navigation via Memory-Augmented Planning and Foresight, 2026. URL https://arxiv.org/abs/2510.08713.

Lang Feng, Zhenghai Xue, Tingcong Liu, and Bo An. Group-in-Group Policy Optimization for LLM Agent Training. In D. Belgrave, C. Zhang, H. Lin, R. Pascanu, P. Koniusz, M. Ghassemi, and N. Chen (eds.), Advances in Neural Information Processing Systems, volume 38, Main Conference, pp. 46375–46408. Curran Associates, Inc., 2025. doi: 10.52202/ 085713-1544. URL https://proceedings.neurips.cc/paper\_files/paper/ 2025/file/420c9f777c0b4f78d515e53cf74d58b2-Paper-Conference.pdf.

Pete Florence, Corey Lynch, Andy Zeng, Oscar A Ramirez, Ayzaan Wahid, Laura Downs, Adrian Wong, Johnny Lee, Igor Mordatch, and Jonathan Tompson. Implicit Behavioral Cloning. In Aleksandra Faust, David Hsu, and Gerhard Neumann (eds.), Proceedings ofthe 5th Conference on Robot Learning, volume 164 of Proceedings ofMachine Learning Research, pp. 158–168. PMLR, 08–11 Nov 2022. URL https://proceedings.mlr.press/v164/florence22a. html.

Google DeepMind. Gemini 3.1 Pro Model Card. Google DeepMind Model Cards, 2026. URL https://deepmind.google/models/model-cards/gemini-3-1-pro/.

Jiawei Gu, Yunzhuo Hao, Huichen Wang, Linjie Li, Michael Qizhe Shieh, Yejin Choi, Ranjay Krishna, and Yu Cheng. ThinkMorph: Emergent Properties in Multimodal Interleaved Chainof-Thought Reasoning. In C. Vondrick, B. Hariharan, C. Raffel, L. Pinto, D. Yang, and A. Faust (eds.), International Conference on Learning Representations, volume 2026, pp. 141405–141447, 2026. URL https://proceedings.iclr.cc/paper\_files/paper/2026/file/ e5095602ad1c6a835e2b643ec4ed97d0-Paper-Conference.pdf.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, Xiaokang Zhang, Xingkai Yu, Yu Wu, Z. F. Wu, Zhibin Gou, Zhihong Shao, Zhuoshu Li, Ziyi Gao, Aixin Liu, Bing Xue, Bingxuan Wang, Bochao Wu, Bei Feng, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chong Ruan, Damai Dai, Deli Chen, Dongjie Ji, Erhang Li, Fangyun Lin, Fucong Dai, Fuli Luo, Guangbo Hao, Guanting Chen, Guowei Li, H. Zhang, Hanwei Xu, Honghui Ding, Huazuo Gao, Hui Qu, Hui Li, Jianzhong Guo, Jiashi Li, Jingchang Chen, Jingyang Yuan, Jinhao Tu, Junjie Qiu, Junlong Li, J. L. Cai, Jiaqi Ni, Jian Liang, Jin Chen, Kai Dong, Kai Hu, Kaichao You, Kaige Gao, Kang Guan, Kexin Huang, Kuai Yu, Lean Wang, Lecong Zhang, Liang Zhao, Litong Wang, Liyue Zhang, Lei Xu,

Leyi Xia, Mingchuan Zhang, Minghua Zhang, Minghui Tang, Mingxu Zhou, Meng Li, Miaojun Wang, Mingming Li, Ning Tian, Panpan Huang, Peng Zhang, Qiancheng Wang, Qinyu Chen, Qiushi Du, Ruiqi Ge, Ruisong Zhang, Ruizhe Pan, Runji Wang, R. J. Chen, R. L. Jin, Ruyi Chen, Shanghao Lu, Shangyan Zhou, Shanhuang Chen, Shengfeng Ye, Shiyu Wang, Shuiping Yu, Shunfeng Zhou, Shuting Pan, S. S. Li, Shuang Zhou, Shaoqing Wu, Tao Yun, Tian Pei, Tianyu Sun, T. Wang, Wangding Zeng, Wen Liu, Wenfeng Liang, Wenjun Gao, Wenqin Yu, Wentao Zhang, W. L. Xiao, Wei An, Xiaodong Liu, Xiaohan Wang, Xiaokang Chen, Xiaotao Nie, Xin Cheng, Xin Liu, Xin Xie, Xingchao Liu, Xinyu Yang, Xinyuan Li, Xuecheng Su, Xuheng Lin, X. Q. Li, Xiangyue Jin, Xiaojin Shen, Xiaosha Chen, Xiaowen Sun, Xiaoxiang Wang, Xinnan Song, Xinyi Zhou, Xianzu Wang, Xinxia Shan, Y. K. Li, Y. Q. Wang, Y. X. Wei, Yang Zhang, Yanhong Xu, Yao Li, Yao Zhao, Yaofeng Sun, Yaohui Wang, Yi Yu, Yichao Zhang, Yifan Shi, Yiliang Xiong, Ying He, Yishi Piao, Yisong Wang, Yixuan Tan, Yiyang Ma, Yiyuan Liu, Yongqiang Guo, Yuan Ou, Yuduan Wang, Yue Gong, Yuheng Zou, Yujia He, Yunfan Xiong, Yuxiang Luo, Yuxiang You, Yuxuan Liu, Yuyang Zhou, Y. X. Zhu, Yanping Huang, Yaohui Li, Yi Zheng, Yuchen Zhu, Yunxian Ma, Ying Tang, Yukun Zha, Yuting Yan, Z. Z. Ren, Zehui Ren, Zhangli Sha, Zhe Fu, Zhean Xu, Zhenda Xie, Zhengyan Zhang, Zhewen Hao, Zhicheng Ma, Zhigang Yan, Zhiyu Wu, Zihui Gu, Zijia Zhu, Zijun Liu, Zilin Li, Ziwei Xie, Ziyang Song, Zizheng Pan, Zhen Huang, Zhipeng Xu, Zhongyu Zhang, and Zhen Zhang. DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning. Nature, 645 (8081):633–638, September 2025. ISSN 1476-4687. doi: 10.1038/s41586-025-09422-z. URL http://dx.doi.org/10.1038/s41586-025-09422-z.

David Ha and Jurgen Schmidhuber. World Models, 2018. URL¨ https://arxiv.org/abs/ 1803.10122.

Danijar Hafner, Timothy Lillicrap, Jimmy Ba, and Mohammad Norouzi. Dream to Control: Learning Behaviors by Latent Imagination. arXiv preprint arXiv:1912.01603, 2019a.

Danijar Hafner, Timothy Lillicrap, Ian Fischer, Ruben Villegas, David Ha, Honglak Lee, and James Davidson. Learning Latent Dynamics for Planning from Pixels. In Kamalika Chaudhuri and Ruslan Salakhutdinov (eds.), Proceedings of the 36th International Conference on Machine Learning, volume 97 of Proceedings of Machine Learning Research, pp. 2555–2565. PMLR, 09–15 Jun 2019b. URL https://proceedings.mlr.press/v97/hafner19a.html.

Shibo Hao, Yi Gu, Haodi Ma, Joshua Hong, Zhen Wang, Daisy Wang, and Zhiting Hu. Reasoning with Language Model is Planning with World Model. In Houda Bouamor, Juan Pino, and Kalika Bali (eds.), Proceedings ofthe 2023 Conference on Empirical Methods in Natural Language Processing, pp. 8154–8173, Singapore, December 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.emnlp-main.507. URL https://aclanthology.org/2023. emnlp-main.507/.

Yushi Hu, Weijia Shi, Xingyu Fu, Dan Roth, Mari Ostendorf, Luke Zettlemoyer, Noah Smith, and Ranjay Krishna. Visual Sketchpad: Sketching as a Visual Chain of Thought for Mul timodal Language Models. In A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang (eds.), Advances in Neural Information Processing Systems, volume 37, pp. 139348–139379. Curran Associates, Inc., 2024. doi: 10.52202/ 079017-4423. URL https://proceedings.neurips.cc/paper\_files/paper/ 2024/file/fb82011040977c7712409fbdb5456647-Paper-Conference.pdf.

Bowen Jin, Hansi Zeng, Zhenrui Yue, Jinsung Yoon, Sercan Arik, Dong Wang, Hamed Zamani, and Jiawei Han. Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning, 2025. URL https://arxiv.org/abs/2503.09516.

Aviral Kumar, Vincent Zhuang, Rishabh Agarwal, Yi Su, JD Co-Reyes, Avi Singh, Kate Baumli, Shariq Iqbal, Colton Bishop, Rebecca Roelofs, Lei Zhang, Kay McKinney, Disha Shrivastava, Cosmin Paduraru, George Tucker, Doina Precup, Feryal Behbahani, and Aleksandra Faust. Training Language Models to Self-Correct via Reinforcement Learning. In Y. Yue, A. Garg, N. Peng, F. Sha, and R. Yu (eds.), International Conference on Learning Representations, volume 2025, pp. 54523–54549, 2025. URL https://proceedings.iclr.cc/paper\_files/paper/ 2025/file/871ac99fdc5282d0301934d23945ebaa-Paper-Conference.pdf.

Chengzu Li, Wenshan Wu, Huanyu Zhang, Yan Xia, Shaoguang Mao, Li Dong, Ivan Vulic, and Furu´ Wei. Imagine While Reasoning in Space: Multimodal Visualization-of-Thought. In Aarti Singh, Maryam Fazel, Daniel Hsu, Simon Lacoste-Julien, Felix Berkenkamp, Tegan Maharaj, Kiri Wagstaff, and Jerry Zhu (eds.), Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings ofMachine Learning Research, pp. 36340–36364. PMLR, 13–19 Jul 2025. URL https://proceedings.mlr.press/v267/li25cz.html.

NVIDIA et al. Cosmos World Foundation Model Platform for Physical AI. arXiv preprint arXiv:2501.03575, 2025.

OpenAI. GPT-5.6 Sol. OpenAI API documentation, 2026. URL https://developers. openai.com/api/docs/models/gpt-5.6-sol.

Jack Parker-Holder and Shlomi Fruchter. Genie 3: A new frontier for world models. Google DeepMind, 2025. URL https://deepmind.google/blog/ genie-3-a-new-frontier-for-world-models/.

Sebastien Racani´ ere, Theophane Weber, David Reichert, Lars Buesing, Arthur Guez, Danilo\` Jimenez Rezende, Adria Puigdom\` enech Badia, Oriol Vinyals, Nicolas Heess, Yujia Li,\` Razvan Pascanu, Peter Battaglia, Demis Hassabis, David Silver, and Daan Wierstra. Imagination-Augmented Agents for Deep Reinforcement Learning. In I. Guyon, U. Von Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett (eds.), Advances in Neural Information Processing Systems, volume 30. Curran Associates, Inc., 2017. URL https://proceedings.neurips.cc/paper\_files/paper/2017/ file/9e82757e9a1c12cb710ad680db11f6f1-Paper.pdf.

Julian Schrittwieser, Ioannis Antonoglou, Thomas Hubert, Karen Simonyan, Laurent Sifre, Simon Schmitt, Arthur Guez, Edward Lockhart, Demis Hassabis, Thore Graepel, Timothy Lillicrap, and David Silver. Mastering Atari, Go, chess and shogi by planning with a learned model. Nature, 588 (7839):604–609, December 2020. ISSN 1476-4687. doi: 10.1038/s41586-020-03051-4. URL http://dx.doi.org/10.1038/s41586-020-03051-4.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal Policy Optimization Algorithms, 2017. URL https://arxiv.org/abs/1707.06347.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models, 2024. URL https://arxiv.org/abs 2402.03300.

Zhaochen Su, Peng Xia, Hangyu Guo, Zhenhua Liu, Yan Ma, Xiaoye Qu, Jiaqi Liu, Yanshu Li, Kaide Zeng, Zhengyuan Yang, Linjie Li, Yu Cheng, Heng Ji, Junxian He, and Yi R. Fung. Thinking with Images for Multimodal Reasoning: Foundations, Methods, and Future Frontiers, 2025. URL https://arxiv.org/abs/2506.23918.

Richard S. Sutton. Dyna, an integrated architecture for learning, planning, and reacting. ACM SIGART Bulletin, 2(4):160–163, July 1991. ISSN 0163-5719. doi: 10.1145/122344.122377. URL http://dx.doi.org/10.1145/122344.122377.

Richard S. Sutton and Andrew G. Barto. Reinforcement Learning: An Introduction. MIT Press, 2 edition, 2018. ISBN 9780262039246. URL https://mitpress.mit.edu/ 9780262039246/reinforcement-learning/.

Dani Valevski, Yaniv Leviathan, Moab Arar, and Shlomi Fruchter. Diffusion Models Are Real-Time Game Engines. In Y. Yue, A. Garg, N. Peng, F. Sha, and R. Yu (eds.), International Conference on Learning Representations, volume 2025, pp. 73754–73776, 2025. URL https://proceedings.iclr.cc/paper\_files/paper/2025/file/ b71ecea210f7159f31e46631fe5c838f-Paper-Conference.pdf.

Kangrui Wang, Pingyue Zhang, Zihan Wang, Yaning Gao, Linjie Li, Qineng Wang, Hanyang Chen, Chi Wan, Yiping Lu, Zhengyuan Yang, Lijuan Wang, Ranjay Krishna, Jiajun Wu, Li Fei-Fei, Yejin Choi, and Manling Li. VAGEN: Reinforcing World Model Reasoning for Multi-Turn VLM Agents. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025a. URL https://arxiv.org/abs/2510.16907.

Zihan Wang, Kangrui Wang, Qineng Wang, Pingyue Zhang, Linjie Li, Zhengyuan Yang, Xing Jin, Kefan Yu, Minh Nhat Nguyen, Licheng Liu, Eli Gottlieb, Yiping Lu, Kyunghyun Cho, Jiajun Wu, Li Fei-Fei, Lijuan Wang, Yejin Choi, and Manling Li. RAGEN: Understanding Self-Evolution in LLM Agents via Multi-Turn Reinforcement Learning, 2025b. URL https://arxiv.org/ abs/2504.20073.

Zirui Wang, Junyi Zhang, Jiaxin Ge, Long Lian, Letian Fu, Lisa Dunlap, Ken Goldberg, XuDong Wang, Ion Stoica, David M. Chan, Sewon Min, and Joseph E. Gonzalez. VisGym: Diverse, Customizable, Scalable Environments for Multimodal Agents, 2026. URL https://arxiv. org/abs/2601.16973.

Yi Xu, Chengzu Li, Han Zhou, Xingchen Wan, Caiqi Zhang, Anna Korhonen, and Ivan Vulic. Visual Planning: Let's Think Only with Images. In C. Von-´ drick, B. Hariharan, C. Raffel, L. Pinto, D. Yang, and A. Faust (eds.), International Conference on Learning Representations, volume 2026, pp. 36706–36739, 2026. URL https://proceedings.iclr.cc/paper\_files/paper/2026/file/ 3de60dc2ff7ac8866e805bbf457e2291-Paper-Conference.pdf.

Sherry Yang, Yilun Du, Seyed Ghasemipour, Jonathan Tompson, Leslie Kaelbling, Dale Schuurmans, and Pieter Abbeel. Learning Interactive Real-World Simulators. In B. Kim, Y. Yue, S. Chaudhuri, K. Fragkiadaki, M. Khan, and Y. Sun (eds.), International Conference on Learning Representations, volume 2024, pp. 45210–45234, 2024. URL https://proceedings.iclr.cc/paper\_files/paper/2024/file/ c4d66eae503694424123b93ac0fbaf17-Paper-Conference.pdf.

Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Tom Griffiths, Yuan Cao, and Karthik Narasimhan. Tree of Thoughts: Deliberate Problem Solving with Large Language Models. In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine (eds.), Advances in Neural Information Processing Systems, volume 36, pp. 11809–11822. Curran Associates, Inc., 2023a. doi: 10.52202/075280-0517. URL https://proceedings.neurips.cc/paper\_files/paper/2023/file/ 271db9922b8d1f4dd7aaef84ed5ac703-Paper-Conference.pdf.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. ReAct: Synergizing Reasoning and Acting in Language Models. In International Conference on Learning Representations (ICLR), 2023b.

Yuexiang Zhai, Hao Bai, Zipeng Lin, Jiayi Pan, Shengbang Tong, Yifei Zhou, Alane Suhr, Saining Xie, Yann LeCun, Yi Ma, and Sergey Levine. Fine-Tuning Large Vision-Language Models as Decision-Making Agents via Reinforcement Learning. In A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang (eds.), Advances in Neural Information Processing Systems, volume 37, pp. 110935–110971. Curran Associates, Inc., 2024. doi: 10.52202/ 079017-3522. URL https://proceedings.neurips.cc/paper\_files/paper/ 2024/file/c848b7d3adc08fcd0bf1df3101ba6728-Paper-Conference.pdf.

Huanyu Zhang, Wenshan Wu, Chengzu Li, Ning Shang, Yan Xia, Yangyu Huang, Yifan Zhang, Li Dong, Zhang Zhang, Liang Wang, Tieniu Tan, and Furu Wei. Latent Sketchpad: Sketching Visual Thoughts to Elicit Multimodal Reasoning in MLLMs, 2025. URL https://arxiv. org/abs/2510.24514.

Tony Z. Zhao, Vikash Kumar, Sergey Levine, and Chelsea Finn. Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware. In Proceedings of Robotics: Science and Systems, Daegu, Republic of Korea, July 2023. doi: 10.15607/RSS.2023.XIX.016.

Chunting Zhou, Lili Yu, Arun Babu, Kushal Tirumala, Michihiro Yasunaga, Leonid Shamis, Jacob Kahn, Xuezhe Ma, Luke Zettlemoyer, and Omer Levy. Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Model. In Y. Yue, A. Garg, N. Peng, F. Sha, and R. Yu (eds.), International Conference on Learning Representations, volume 2025, pp. 6446– 6469, 2025. URL https://proceedings.iclr.cc/paper\_files/paper/2025/ file/12678c3948153f4bc391f51e2082bd6e-Paper-Conference.pdf.

## APPENDIX CONTENTS

A Environments and Evaluation Sets 16   
A.1 Planning Evaluation Sets 16   
B Training Data and Optimization 17   
B.1 Action Chunking and Linear Rollout Data 17   
B.2 Constructing Alternative-Future Traces 17   
B.3 Training Data Examples and Statistics 17   
B.4 Model and Training Setup 21   
C Analysis of Adaptive Imagination 22   
C.1 Combining Linear Rollouts and Forward Thinking 22   
C.2 Counterfactual Branching . 22   
C.3 Task Success and Rollout Length . 23   
D Qualitative Dynamics Predictions and Rollouts 24   
D.1 Maze2D Rollouts 24   
D.2 Sokoban Rollouts 30   
D.3 Pacman Rollouts 35   
D.4 PushT Rollouts 44   
D.5 Rollouts for different reasoning formats 51   
D.6 RL Trajectories across Checkpoints 52   
D.7 Failure Cases 63

## A ENVIRONMENTS AND EVALUATION SETS

The training and test configurations below are used for the easy-to-hard evaluation in Table 1. All grid sizes include the outer boundary walls. Maze2D, Sokoban, and Pacman use four movement actions: up, down, left, and right. PushT uses continuous $( x , y )$ pusher targets. Training traces end with STOP. We report episode success rates for the three discrete tasks, and both success rate and mean final coverage for PushT.

Maze2D: Constrained Navigation. Maze2D is a grid navigation task from VisGym (Wang et al., 2026). The input image shows the agent, walls, and goal. The model must navigate to the goal while predicting its position after each move; an episode succeeds when the agent reaches the goal. We train on $1 1 \times 1 1$ mazes with $9 \times 9$ interiors and test on larger mazes.

Sokoban: Irreversible Object Dynamics. Sokoban is a box-pushing puzzle (Racaniere et al. \` , 2017). The input image shows the agent, boxes, walls, and targets. Movement into a box pushes it when the next cell is free. A wrong push can permanently trap a box or create a deadlock, so the model must predict how actions change the box configuration. Success requires placing all boxes on targets. For Table 1, we train on $7 \times 7$ two-box puzzles and test on four-box puzzles.

Pacman: Reactive Dynamics. Pacman adds a moving opponent, as in prior work on MiniPacman (Racaniere et al.\` , 2017). The input image shows Pacman, the red ghost, walls, and food pellets. After each Pacman move, the ghost moves by greedily chasing Pacman. Success requires collecting all food without being caught. We train on $9 \times 9$ grids with 8 food pellets. This training configuration is shared by the linear-rollout and forward-thinking data. We test on $1 2 \times 1 2$ grids with 12 food pellets and a closer initial ghost position.

PushT: Continuous Contact Dynamics. PushT is a continuous manipulation task (Florence et al., 2022; Chi et al., 2024). The input image shows the pusher, a T-shaped object, and its target pose. The model predicts pusher targets and the resulting object motion. Training uses initial object positions nearer the center; evaluation uses unseen positions farther from the center. Final coverage measures the fraction of the target covered by the object at the end of execution. We report its mean across episodes and the success rate, defined as the fraction of episodes with final coverage greater than 0.80.

## A.1 PLANNING EVALUATION SETS

Table 3 evaluates counterfactual branching and forward thinking on the Sokoban and Pacman sets below. These sets test decisions whose consequences become apparent only after several actions, separately from the easy-to-hard evaluation in Table 1. We report task success rates with $K = \infty$ without intermediate observations from the real environment.

Sokoban Deadlock and Structural Fork. Each family contains 2-box, 3-box, and 4-box evaluation sets, with 100 episodes per set. Deadlock instances contain locally legal pushes that can make the puzzle unsolvable, with harmful consequences becoming apparent later in the trajectory. Structural Fork instances offer multiple legal choices, some of which lead to dead ends. The supervised checkpoints are trained on standard 2-box puzzles; the 3-box and 4-box sets test transfer to larger object configurations.

Pacman route-order traps. We evaluate on out-of-domain grids of size $8 \times 8$ with 2, 4, 6, or 8 food items and $1 0 \times 1 0$ with 8 food items. Both grid sizes differ from the $9 \times 9$ training grids. In these instances, pursuing the nearer food first leaves no successful continuation, while a farther-first route has a replay-verified solution that clears the food, avoids the ghost, and ends with STOP.

We denote a configuration as $\mathrm { G } x / \mathrm { F } y .$ , where x is the grid size and y is the number of food items; for example, G8/F4 denotes an $8 \times 8$ grid with four food items.

## B TRAINING DATA AND OPTIMIZATION

We first describe the supervised training traces and their statistics, then give the model and optimization settings separately.

## B.1 ACTION CHUNKING AND LINEAR ROLLOUT DATA

We segment expert trajectories into action chunks with $K \in \{ 1 , 3 , 5 , 1 0 , \infty \}$ . Each example starts from an observation $o _ { t }$ and goal g. The action-only target contains the next K expert actions; the linear-imagination target interleaves each action with its successor image. The images used as training targets are observations from the expert trajectory. At inference time, the model generates these images as predicted observations and uses them to plan subsequent actions.

For example, a $K = 3$ training segment has the following schematic form:

$$
\begin{array} { r l } { \mathrm { I n p u t : } } & { { } [ o _ { t } , g ] , } \\ { \mathrm { A c t i o n - o n l y ~ t a r g e t : } } & { { } [ a _ { t } , a _ { t + 1 } , a _ { t + 2 } ] , } \\ { \mathrm { L i n e a r - i m a g i n a t i o n ~ t a r g e t : } } & { { } [ a _ { t } , o _ { t + 1 } , a _ { t + 1 } , o _ { t + 2 } , a _ { t + 2 } , o _ { t + 3 } ] . } \end{array}
$$

After these three actions are executed, the next chunk starts from the new real observation. $K = 1$ gives feedback after every action, while $K = \infty$ uses the full remaining plan without intermediate real observations. The final segment ends with STOP when the expert task is complete. Figure 7 shows a qualitative example of different action chunk sizes.

## B.2 CONSTRUCTING ALTERNATIVE-FUTURE TRACES

Counterfactual-branching and forward-thinking traces use the same underlying expert episodes as linear rollouts. We do not select additional training instances specifically for failures or difficult forks; only the imagination trace structure changes. The additional supervision teaches the model how to explore and compare imagined futures. Figures 8, 9, and 10 show examples of the three constructions below.

Counterfactual branching. For Sokoban, expert trajectories are scanned for critical states where the expert push keeps the level solvable and another legal push is certified unsolvable by moving a box against a wall. At up to three earliest such states, a trace inserts an alternative push, its successor image, a rejection, and a return to the expert continuation. The executable sequence after </think> remains the original expert solution. Episodes without a certified harmful alternative remain single-plan rollouts.

Forward thinking in Sokoban. Each trace compares two candidate box-pushing plans from the same initial state. One follows the expert solution; the other starts with a different first push and uses the best continuation found by search. Breadth-first search (BFS) labels each candidate as solved, unfinished at the action limit, or deadlocked. The target demonstrates both futures and selects a successful plan, preferring the shorter plan when both succeed. Candidate order is randomized.

Forward thinking in Pacman. Each trace begins with a shared prefix of the expert trajectory, reaching a decision state 2 to 10 moves into the episode. It then compares two continuations from that state: the expert continuation and an alternative that takes a different first move toward a nearer food item. The trace returns to the same decision state between candidates, compares their outcomes, and commits a complete plan. Candidate order is randomized. Figure 10 shows an example in which the alternative is caught by the ghost. The outcome proportions across all training traces are given in Table 5.

## B.3 TRAINING DATA EXAMPLES AND STATISTICS

Table 4 summarizes the expert rollouts used to construct action chunks. Table 5 reports the alternative-future trace statistics. The Sokoban forward-thinking traces used for training contain two candidates; their outcome proportions are reported within that population. PushT expert trajectories are generated with PPO (Schulman et al., 2017). Training budgets are given separately below.

Complete supervised example | source row 5 | saved training targets

Table 4: Linear-rollout training data per task. Plan length counts the expert’s actions excluding the final STOP and is reported as median (mean) [range]. Grid sizes include the outer wall.
<table><tr><td></td><td>Maze2D</td><td>Sokoban</td><td>Pacman</td><td>PushT</td></tr><tr><td>Grid size</td><td>11 × 11</td><td> $7 \times 7$ </td><td>9 × 9</td><td>N/A</td></tr><tr><td>Training setting</td><td>maze navigation</td><td>2 boxes</td><td>8 food</td><td>diverse T start</td></tr><tr><td>Expert</td><td>shortest-path oracle</td><td>Sokoban solver</td><td>oracle planner</td><td>PPO policy</td></tr><tr><td>Action space Action budget</td><td>4 moves 40</td><td>4 moves 20</td><td>4 moves 80</td><td>continuous (x, y) 30</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Plan length</td><td>9 (9.9) [5–33]</td><td>10 (9.9) [2–19]</td><td>23 (22.5) [9–27]</td><td>20 (20.0) [4–29]</td></tr></table>

Table 5: Forward-Thinking and Counterfactual-Branching trace statistics. Forward-thinking outcome percentages are over two-candidate traces.
<table><tr><td>Sokoban Forward Thinking</td></tr><tr><td>Both solve 79%</td></tr><tr><td>At least one candidate deadlocks 21%</td></tr><tr><td>Sokoban Counterfactual Branching</td></tr><tr><td>Branches per episode: 0 / 1 / 2 / 3 27.1 / 35.3 / 20.6 / 17.1%</td></tr><tr><td>Pacman Forward Thinking</td></tr><tr><td>Both candidates succeed 96.4%</td></tr><tr><td>A candidate dies 3.6%</td></tr></table>

## Training data | Action chunking

Action chunking, Sokoban | Expert trajectory | steps 0-6<sub>row</sub> <sub>5</sub> <sub>|</sub> <sub>saved</sub> <sub>training</sub> <sub>targets</sub>  
![](images/e898c8d7e0b4ed0c55b874643d0c577fc5d228477cae9bbad3aff63ce1e3f0f8.jpg)

Action chunking, Sokoban | Expert trajectory | steps 6-12  
![](images/f19cfc7ae2ca6b17b811e8d4d25ea7da20c29fc47570cef704a242cd13128239.jpg)

<sup>o</sup>14 <sup>o</sup>15 <sup>o</sup>16Action chunking, Sokoban | Expert trajectory | steps 12-17  
![](images/ad8ef4de826fbf62edba9cec1cb5c83848324a3ca1e4db201177b434c8b66623.jpg)

Counterfactual branching, Sokoban | Main plan | steps 0-6<sub>e</sub> <sub>row</sub> <sub>2</sub> <sub>|</sub> <sub>saved</sub> <sub>training</sub> <sub>targets</sub>  
Counterfactual branching, Sokoban | Main plan | steps 6-12  
Chunk boundaries for this same expert trajectory
<table><tr><td>K=1</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td><td>8</td><td>9</td><td>10</td><td>11</td><td>12</td><td>13</td><td>14</td><td>15</td><td>16</td><td>17</td></tr><tr><td>17 chunks K=3</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td><td></td><td></td><td>10</td><td>11</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>6 chunks</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>8</td><td>9</td><td></td><td></td><td>12</td><td>13</td><td>14</td><td>15</td><td>16</td><td>17</td></tr><tr><td>K= 5 4 chunks</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td><td>8</td><td>9</td><td>10</td><td>11</td><td>12</td><td>13</td><td>14</td><td>15</td><td>16</td><td>17</td></tr><tr><td>K = 10 2 chunks</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td><td>8</td><td>9</td><td>10</td><td>11</td><td>12</td><td>13</td><td>14</td><td>15</td><td>16</td><td>17</td></tr><tr><td>K=∞</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td><td>8</td><td>9</td><td>10</td><td>11</td><td>12</td><td>13</td><td></td><td>15</td><td></td><td>17</td></tr></table>

Figure 7: A complete expert trajectory and its action chunks. The same 17-action training trajectory, including STOP, is segmented with $K \in \{ 1 , 3 , 5 , 1 0 , \infty \}$ . Numbers in the lower panel refer to the actions above. Images are saved environment observations used as supervision.

## Training data | Counterfactual branching

Complete supervised example | source row 2 | saved training targets

![](images/3a36a379a5514a4611f17dd2e6c3dc4bb49c5d8f4f322cea9d61585b551ffbf0.jpg)

![](images/1443296d307e669baefe5effb9e0be20dcb653df9cdb36a9ba06f35efed74f8d.jpg)

![](images/4265fffcf9cc1b1ff805c02f24250a3bb6332173d8b69a67aaf0b34cc8b3971f.jpg)  
Both counterfactual pushes are rejected. The final executable sequence follows the complete expert main line shown above.

Figure 8: Complete counterfactual-branching training example. The main line contains 12 actions including STOP. The two alternative pushes, inserted after main-line actions 3 and 5, lead to certified deadlocks and are rejected.

## Training data | Forward thinking

Complete supervised example | source row 1 | saved training targets

![](images/b12bbb19e7a19abe146042f85eed6efa16b5f9e979f493fbc06f30e33a361d0b.jpg)  
Result: this plan reaches a certified deadlock; the puzzle cannot be solved from this state.

<sup>o</sup>11 <sup>o</sup>12 <sup>o</sup>13 <sup>o</sup>14 <sup>o</sup>15Forward thinking, Pacman | Candidate 2 | selected; reset to the saved decision state | steps 10-15  
5 <sup>o</sup>16 <sup>o</sup>17 <sup>o</sup>18 <sup>o</sup>19 <sup>o</sup>20Forward thinking, Pacman | Candidate 2 | selected; reset to the saved decision state | steps 15-20  
Forward thinking, Sokoban | Candidate 2 | selected | steps 0-5  
![](images/4da3c17f8cc7de55f3a9561a00c0f8a24cc3395fdce0b537b008d4bd03bb1f51.jpg)  
Comparison: Candidate 1 deadlocks, while Candidate 2 solves the puzzle. I will commit Candidate 2.  
Figure 9: Complete forward-thinking training example in Sokoban. Candidate 1 deadlocks after two moves. Candidate 2 solves the puzzle in four moves and ends with STOP.

## Training data | Forward thinking

Complete supervised example | source row 51 | saved training targets

![](images/01c5f0d20b964e662f492e0624a578a33ae6e75fb59453cdbb5f4a8981251a48.jpg)

![](images/1bf16f5970016387aa139efbc04012497fa4de8a37d59b6f61ad5ef4b47f7c35.jpg)

![](images/05312ef217f118177f2f1882f77d83c38da34f43994cafdb0b895768a7cbf70a.jpg)  
Forward thinking, Pacman | Candidate 2 | selected; reset to the saved decision state | stepCandidate 1 reaches nearby food, then is caught by the ghost. Both candidates start from the same decision state.Down Left Left

<sup>o</sup>6 <sup>o</sup>7 <sup>o</sup>8 <sup>o</sup>9 <sup>o</sup>10Forward thinking, Pacman | Candidate 2 | selected; reset to the saved decision state | steps 5-10es nearby food, then is caught by the ghost. Both candidates start from the same decision state.  
![](images/5bc9e95a94b7ec127f63afb111adb314392a7c713520579c8e03d102b5795008.jpg)

![](images/f8ea48886fc43633569072089f9176d8b98f0e936f0ece002836d985e67705fa.jpg)

![](images/4c7c1954b1ca177af098233e5f60d1d58cee82b322b260632bba8dee2646e80c.jpg)

Forward thinking, Pacman | Candidate 2 | selected; reset to the saved decision state | steps 20-24  
![](images/c51c5a372cadc4b7ad8fe294e2f3fd0caffe32264099823920bc3848b2448fa3.jpg)  
Candidate 2 collects all food in 24 moves including the shared prefix. The executable target follows this route and ends with STOP.  
Figure 10: Complete forward-thinking training example in Pacman. After a shared five-move prefix, Candidate 1 is caught by the ghost after seven further moves. Candidate 2 collects all food after 19 further moves; its executable target ends with STOP. All saved training images are shown.

## B.4 MODEL AND TRAINING SETUP

Backbone and SFT. We use BAGEL (Deng et al., 2025), with 14B total parameters and 7B active parameters. SFT uses teacher forcing, with next-token cross-entropy for text outputs and the imagegeneration MSE objective described in Section 2.2.

For PushT world-model training, we use a keep-last-VAE attention mask that restricts VAE conditioning to the most recent block while retaining all preceding text and ViT tokens. At inference, we enforce the same policy by evicting the previous conditioning VAE block from the KV cache whenever a new image is added.

Table 6: SFT settings.
<table><tr><td>Setting</td><td>Value</td></tr><tr><td>Standard SFT steps</td><td>5,000</td></tr><tr><td>SFT batch size</td><td>8</td></tr><tr><td>Training-example presentations</td><td>40,000</td></tr><tr><td>Released Sokoban checkpoints</td><td>BAGEL-7B-MoT, exponential-moving-average (EMA) weights</td></tr><tr><td>Optimizer</td><td>AdamW  $( \beta _ { 1 } = 0 . 9 , \beta _ { 2 } = 0 . 9 5 )$ </td></tr><tr><td>Learning rate</td><td> $2 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>Learning-rate schedule</td><td>Cosine</td></tr><tr><td>Warmup steps</td><td>100</td></tr><tr><td>EMA decay</td><td>0.993</td></tr></table>

Reinforcement learning. During RL, the model samples a reasoning trace containing textual decisions, imagined visual states, and a final executable action sequence. It receives no real environment observations during this deliberation. Only the final sequence is executed, and the reward is binary task success. We use GRPO (Shao et al., 2024) with group size $N = 8 ,$ batch size 16, and 200 training steps on ordinary $8 \times 8$ three-box Sokoban puzzles, without additional expert demonstrations. The learning rate is $\mathrm { \dot { 2 } \times 1 0 ^ { - 6 } }$ . The Sokoban RL recipe uses sampling temperature 1.0. Evaluation uses held-out three- and four-box Deadlock and Structural Fork sets. The policy-gradient objective uses the language-token log probabilities. Generated images are not directly supervised with an image reconstruction loss during this stage.

## C ANALYSIS OF ADAPTIVE IMAGINATION

We examine how RL changes the plans generated by the model. All results use the four 3/4-box Deadlock and Structural Fork sets, with 100 episodes per set. Pooled results therefore contain 400 episodes at each checkpoint. Pre-RL denotes the SFT checkpoint; RL50, for example, denotes 50 RL training steps.

## C.1 COMBINING LINEAR ROLLOUTS AND FORWARD THINKING

SFT teaches two separate formats: a linear rollout that produces one plan, and a fixed comparison that generates two candidate plans before choosing one. RL combines them into a hybridformat: the model begins with a linear rollout, evaluates its predicted outcome, and adds another candidate when it judges the first plan unsuccessful. Figure 5 shows this process continuing to a third candidate. The model can therefore decide to compare plans after inspecting its first imagined outcome. Related embodied agents use execution feedback to diagnose failures and revise plans. Here, imagined outcomes trigger further planning before execution. Table 7 separates the format change from the decision to generate another candidate. From RL100 onward, almost every comparison uses the hybrid format, and the model always judges the first candidate unsuccessful before adding a second. We check that judgment by executing the first candidate’s actions from the original state. At RL200, these actions fail to solve the puzzle in 95.5–100% of the comparisons across the four sets.

Table 7: Forward-Thinking formats and first-plan failures. C1 and C2 denote the first and second candidate plans. Generates C2, Fixed comparison, and Hybrid use all 100 episodes per set as their denominator. The last two rows use comparisons with readable candidate actions and verdicts: Predicts C1 failure records the model’s judgment; Actual C1 failure records whether C1 fails when executed independently. All values are percentages.
<table><tr><td>Set</td><td>Metric</td><td>Pre-RL</td><td>RL50</td><td>RL100</td><td>RL150</td><td>RL200</td></tr><tr><td rowspan="5">3-box Deadlock</td><td>Generates C2</td><td>83%</td><td>57%</td><td>18%</td><td>20%</td><td>22%</td></tr><tr><td>Fixed comparison</td><td>79%</td><td>51%</td><td>0%</td><td>0%</td><td>0%</td></tr><tr><td>Hybrid</td><td>4%</td><td>6%</td><td>18%</td><td>20%</td><td>22%</td></tr><tr><td>Predicts C1 failure</td><td>31.7%</td><td>33.9%</td><td>100%</td><td>100%</td><td>100%</td></tr><tr><td>Actual C1 failure</td><td>61.0%</td><td>64.3%</td><td>94.4%</td><td>100%</td><td>95.5%</td></tr><tr><td rowspan="5">4-box Deadlock</td><td>Generates C2</td><td>76%</td><td>56%</td><td>34%</td><td>36%</td><td>37%</td></tr><tr><td>Fixed comparison</td><td>72%</td><td>42%</td><td>0%</td><td>0%</td><td>0%</td></tr><tr><td>Hybrid</td><td>4%</td><td>14%</td><td>34%</td><td>36%</td><td>37%</td></tr><tr><td>Predicts C1 failure</td><td>55.4%</td><td>63.0%</td><td>100%</td><td>100%</td><td>100%</td></tr><tr><td>Actual C1 failure</td><td>81.1%</td><td>83.3%</td><td>97.1%</td><td>97.2%</td><td>97.3%</td></tr><tr><td rowspan="5">3-box Structural Fork</td><td>Generates C2</td><td>80%</td><td>59%</td><td>20%</td><td>29%</td><td>30%</td></tr><tr><td>Fixed comparison</td><td>75%</td><td>47%</td><td>0%</td><td>0%</td><td>0%</td></tr><tr><td>Hybrid</td><td>5%</td><td>12%</td><td>20%</td><td>29%</td><td>30%</td></tr><tr><td>Predicts C1 failure</td><td>53.5%</td><td>61.1%</td><td>100%</td><td>100%</td><td>100%</td></tr><tr><td>Actual C1 failure</td><td>73.2%</td><td>72.2%</td><td>100%</td><td>96.6%</td><td>96.7%</td></tr><tr><td rowspan="5">4-box Structural Fork</td><td>Generates C2</td><td>77%</td><td>60%</td><td>31%</td><td>34%</td><td>34%</td></tr><tr><td>Fixed comparison</td><td>70%</td><td>42%</td><td>1%</td><td>0%</td><td>0%</td></tr><tr><td>Hybrid</td><td>7%</td><td>18%</td><td>30%</td><td>34%</td><td>34%</td></tr><tr><td>Predicts C1 failure</td><td>57.6%</td><td>60.0%</td><td>100%</td><td>100%</td><td>100%</td></tr><tr><td>Actual C1 failure</td><td>84.8%</td><td>83.6%</td><td>100%</td><td>100%</td><td>100%</td></tr></table>

## C.2 COUNTERFACTUAL BRANCHING

RL mainly changes how many alternatives the model examines within a plan. Across the four sets, the percentage of episodes with any branch changes from 81.8% before RL to 83.5% at RL200. The mean number of branches per episode increases from 1.70 to 1.89, reaching 1.95 at RL150.

Table 8 shows the results for each set. On 3-box Structural Fork, the percentage of episodes with at least three branches rises from 41% before RL to 56% at RL200; on 4-box Structural Fork, it

rises from 35% to 54%. Each alternative branch still contains one imagined action. The critical-step analysis in Figure 6 measures whether these branches occur at decisions where a wrong action could make the puzzle unsolvable.

Table 8: Number of counterfactual branches per episode. Mean counts branches across all 100 episodes in each row, including episodes with no branch. Columns 0, 1, 2, and ≥ 3 give the percentage of episodes with that many complete branches. Each branch contains one imagined action.
<table><tr><td>Set</td><td>Checkpoint</td><td>Mean</td><td>0</td><td>1</td><td>2</td><td>≥3</td></tr><tr><td rowspan="5">3-box Deadlock</td><td>Pre-RL</td><td>1.18</td><td>34%</td><td>31%</td><td>18%</td><td>17%</td></tr><tr><td>RL50</td><td>1.13</td><td>33%</td><td>36%</td><td>16%</td><td>15%</td></tr><tr><td>RL100</td><td>1.20</td><td>34%</td><td>30%</td><td>18%</td><td>18%</td></tr><tr><td>RL150</td><td>1.25</td><td>35%</td><td>25%</td><td>20%</td><td>20%</td></tr><tr><td>RL200</td><td>1.25</td><td>35%</td><td>25%</td><td>20%</td><td>20%</td></tr><tr><td rowspan="5">4-box Deadlock</td><td>Pre-RL</td><td>1.67</td><td>22%</td><td>23%</td><td>21%</td><td>34%</td></tr><tr><td>RL50</td><td>1.67</td><td>23%</td><td>20%</td><td>24%</td><td>33%</td></tr><tr><td>RL100</td><td>1.73</td><td>17%</td><td>29%</td><td>18%</td><td>36%</td></tr><tr><td>RL150</td><td>1.88</td><td>13%</td><td>24%</td><td>25%</td><td>38%</td></tr><tr><td>RL200</td><td>1.76</td><td>19%</td><td>22%</td><td>23%</td><td>36%</td></tr><tr><td rowspan="5">3-box Structural Fork</td><td>Pre-RL</td><td>2.08</td><td>7%</td><td>19%</td><td>33%</td><td>41%</td></tr><tr><td>RL50</td><td>2.14</td><td>5%</td><td>23%</td><td>25%</td><td>47%</td></tr><tr><td>RL100</td><td>2.23</td><td>4%</td><td>18%</td><td>29%</td><td>49%</td></tr><tr><td>RL150</td><td>2.43</td><td>5%</td><td>11%</td><td>20%</td><td>64%</td></tr><tr><td>RL200</td><td>2.30</td><td>6%</td><td>14%</td><td>24%</td><td>56%</td></tr><tr><td rowspan="5">4-box Structural Fork</td><td>Pre-RL</td><td>1.88</td><td>10%</td><td>27%</td><td>28%</td><td>35%</td></tr><tr><td>RL50</td><td>2.00</td><td>6%</td><td>26%</td><td>30%</td><td>38%</td></tr><tr><td>RL100</td><td>2.04</td><td>7%</td><td>22%</td><td>31%</td><td>40%</td></tr><tr><td>RL150</td><td>2.25</td><td>5%</td><td>19%</td><td>22%</td><td>54%</td></tr><tr><td>RL200</td><td>2.24</td><td>6%</td><td>18%</td><td>22%</td><td>54%</td></tr></table>

## C.3 TASK SUCCESS AND ROLLOUT LENGTH

Table 9 reports task success and the mean numbers of imagined and executed actions. From Pre-RL to RL200, forward thinking reduces imagined actions from 27.82 to 22.18 per episode, while task success rises from 30.3% to 41.0%. Branching reduces imagined actions from 23.62 to 21.59, while task success rises from 30.0% to 50.3%. Thus, both models improve without generating longer imagined trajectories. These are action counts; we do not measure inference time or FLOPs here.

Table 9: Mean rollout lengths and pooled task success on the four 3/4-box sets. Each row pools 400 episodes. Task success is a percentage; Imagined and Executed are mean action counts per episode.
<table><tr><td>Format</td><td>Checkpoint</td><td>Success</td><td>Imagined</td><td>Executed</td></tr><tr><td rowspan="5">Forward</td><td>Pre-RL</td><td>30.3%</td><td>27.82</td><td>17.88</td></tr><tr><td>RL50</td><td>35.0%</td><td>24.95</td><td>17.36</td></tr><tr><td>RL100</td><td>37.5%</td><td>20.87</td><td>16.96</td></tr><tr><td>RL150</td><td>38.5%</td><td>21.40</td><td>16.97</td></tr><tr><td>RL200</td><td>41.0%</td><td>22.18</td><td>17.50</td></tr><tr><td rowspan="5">Branching</td><td>Pre-RL</td><td>30.0%</td><td>23.62</td><td>18.63</td></tr><tr><td>RL50</td><td>32.0%</td><td>22.60</td><td>18.15</td></tr><tr><td>RL100</td><td>40.5%</td><td>21.88</td><td>17.95</td></tr><tr><td>RL150</td><td>45.8%</td><td>23.16</td><td>18.87</td></tr><tr><td>RL200</td><td>50.3%</td><td>21.59</td><td>18.24</td></tr></table>

## D QUALITATIVE DYNAMICS PREDICTIONS AND ROLLOUTS

The following examples compare predicted visual states with ground-truth states along action sequences in each environment. These selected trajectories provide qualitative evidence; they do not report aggregate prediction accuracy. The examples show complete recorded trajectories, from the initial observation to episode termination. Every action and saved state is displayed; adjacent rows or pages repeat their shared endpoint. Examples at different chunk sizes may start from different initial states. Additionally, we show example traces of counterfactual branching and forward thinking.

## D.1 MAZE2D ROLLOUTS

Task configurations are described in Appendix A.

![](images/4ca534e4c383064e08f5cd875b69f074891c0acc4193096c39299f91398500f3.jpg)  
Figure 11: Complete Maze2D trajectory at K = 1. Selected successful Ordinary Hard episode; all 17 actions and their states are shown. The reactive reference receives a real observation after every action.

![](images/af61feb0feaa5a0e1f62f880b91407592f0db32eb2f7fa11a62723ee2cb4294c.jpg)  
Figure 12: Complete Maze2D trajectory at K = 3. Selected successful Ordinary Hard episode; this page shows steps 0–15 of 27. The trajectory continues on the next page. Blue rows show predictions; teal inputs and feedback arrows show the real observations supplied between chunks.

![](images/4c58179a21a4efdcddc3694a8ee3803addc42e93dac4e503c77fd77790e6a8ba.jpg)  
Figure 12 (continued). Maze2D, K = 3. Steps 15–27, ending in success. The shared endpoint connects the two pages.

![](images/62cf0199bf20a3f7564c07195a4efe4ff07974b14f670a9982a5bb0a7498aa21.jpg)  
Figure 13: Complete Maze2D trajectory at K = 5. Selected successful Ordinary Hard episode; all 23 actions and their states are shown. Blue rows show predictions; teal inputs and feedback arrows show the real observations supplied between chunks.

![](images/70fb5e041bcafec064bd79c54e7a5a0f7a54c8a6cbe1c2283d70c33ed1af49ea.jpg)  
Figure 14: Complete Maze2D trajectory at K = 10. Selected successful Ordinary Hard episode; all 21 actions and their states are shown. Blue rows show predictions; teal inputs and feedback arrows show the real observations supplied between chunks.

![](images/ed314d8d9f09e18133f148e1147f7ddfb060144760e0ec9e60a37f5adfcb3701.jpg)  
Figure 15: Complete Maze2D trajectory at $K = \infty .$ . Selected successful Ordinary Hard episode; all 13 actions and their states are shown. The model receives only the initial observation, with no intermediate feedback. Ground-truth states are shown for comparison.

## D.2 SOKOBAN ROLLOUTS

Task configurations are described in Appendix A.

![](images/28ef4f703abd10a9b45cb114b18d96b316d7711af3ae550f70bbf4ffc8afb6d9.jpg)  
Figure 16: Complete Sokoban trajectory at K = 1. Selected successful Ordinary Hard episode; all 21 actions and their states are shown. The reactive reference receives a real observation after every action.

![](images/cb843ae492f6d492b42e4898d60291e852f11064ad8bbbe0047324beebc8a267.jpg)  
Figure 17: Complete Sokoban trajectory at K = 3. Selected successful Ordinary Hard episode; all 21 actions and their states are shown. Blue rows show predictions; teal inputs and feedback arrows show the real observations supplied between chunks.

![](images/113acafce53ddfd21ff641ab0471df1585fe8cd5c8599fdd303f58caef0aaa04.jpg)  
Figure 18: Complete Sokoban trajectory at K = 5. Selected successful Ordinary Hard episode; all 19 actions and their states are shown. Blue rows show predictions; teal inputs and feedback arrows show the real observations supplied between chunks.

![](images/68cda8c4e4d8764f73671f08e334465c299fc215b68c9fb426a6bc9c0178be92.jpg)  
Figure 19: Complete Sokoban trajectory at K = 10. Selected successful Ordinary Hard episode; all 19 actions and their states are shown. Blue rows show predictions; teal inputs and feedback arrows show the real observations supplied between chunks.

![](images/6b7a3031cbacd13475a4e963511fe71e1d261da21cb04e4f753570105a3e01ef.jpg)  
Figure 20: Complete Sokoban trajectory at K = ∞. Selected successful Ordinary Hard episode; all 17 actions and their states are shown. The model receives only the initial observation, with no intermediate feedback. Ground-truth states are shown for comparison.

## D.3 PACMAN ROLLOUTS

Task configurations are described in Appendix A.

![](images/c0681523a2bf88fbd7e373079a8d947edd9d34ea3d43d26d4986d7c4c8d622d4.jpg)  
Figure 21: Complete Pacman trajectory at K = 1. Selected successful Ordinary Hard episode; all 33 actions and their states are shown. The reactive reference receives a real observation after every action.

![](images/024a6d0cc36d95318333cffa941e8f54f50cb2e0d97e36e2d077c5f83a1ddaf0.jpg)  
Figure 22: Complete Pacman trajectory at K = 3. Selected successful Ordinary Hard episode; this page shows steps 0–20 of 33. The trajectory continues on the next page. Blue rows show predictions; teal inputs and feedback arrows show the real observations supplied between chunks.

![](images/b922e814c1bcfd9041e33945ff59c1cf3a669124539f7a10ce4b9d7eca279e15.jpg)  
Figure 22 (continued). Pacman, K = 3. Steps 20–33, ending in success. The shared endpoint connects the two pages.

![](images/1184273a4095ef7155e2f594dc2fe61432cd25c6c4b51fb1edf701a4a1c62249.jpg)  
Figure 23: Complete Pacman trajectory at K = 5. Selected successful Ordinary Hard episode; this page shows steps 0–20 of 33. The trajectory continues on the next page. Blue rows show predictions; teal inputs and feedback arrows show the real observations supplied between chunks.

![](images/4fa98bf5752d96846061de0fd9f009c34ecf8c9e18fa42187c072831f2cf553c.jpg)  
Figure 23 (continued). Pacman, K = 5. Steps 20–33, ending in success. The shared endpoint connects the two pages.

![](images/df88f7db442e0540ec67a6b995bd8f6406ab1165529bc1cee23094869826c122.jpg)  
Figure 24: Complete Pacman trajectory at K = 10. Selected successful Ordinary Hard episode; this page shows steps 0–20 of 35. The trajectory continues on the next page. Blue rows show predictions; teal inputs and feedback arrows show the real observations supplied between chunks.

![](images/f2608b2f8dfef848fc4e54446d51c33ebbeec7bcd2dd5caf83ec5c78e58b554f.jpg)  
Figure 24 (continued). Pacman, K = 10. Steps 20–35, ending in success. The shared endpoint connects the two pages.

![](images/067ca3073786ed1e36e674ff4fff4ea46d676db66cee8c85b634ab2202f12d3c.jpg)  
Figure 25: Complete Pacman trajectory at K = ∞. Selected successful Ordinary Hard episode; this page shows steps 0–15 of 27. The trajectory continues on the next page. The model receives only the initial observation, with no intermediate feedback. Ground-truth states are shown for comparison.

![](images/ebb536209dbaccf87a206d27499a0d7ea8b4a6bbad5cc8e678530bd3b1390bc5.jpg)  
Figure 25 (continued). Pacman, $K = \infty ,$ . Steps 15–27, ending in success. The shared endpoint connects the two pages; no new observation is supplied.

## D.4 PUSHT ROLLOUTS

The starting-position shift is described in Appendix A.

![](images/57d54e73894149ad5007221bf7b21a706a32f0ed4172521839f26270fa41858d.jpg)  
Figure 26: Complete PushT trajectory at K = 1. Selected successful Far episode; all 19 actions and their states are shown. The reactive reference receives a real observation after every action. Actions give normalized target coordinates (x, y).

![](images/76b59ad7f7d00b5e4e167b96916ba7612c054fcf50032c4b16ad41e081ff9a51.jpg)  
Figure 27: Complete PushT trajectory at K = 3. Selected successful Far episode; all 18 actions and their states are shown. Blue rows show predictions; teal inputs and feedback arrows show the real observations supplied between chunks. Actions give normalized target coordinates (x, y).

![](images/455fc9d505758962e433d35b216d89f3280076ac605f150c38934a028565ad8c.jpg)  
Figure 28: Complete PushT trajectory at K = 5. Selected successful Far episode; this page shows steps 0–15 of 27. The trajectory continues on the next page. Blue rows show predictions; teal inputs and feedback arrows show the real observations supplied between chunks. Actions give normalized target coordinates (x, y).

![](images/131886ed6fb537a89a9b5499bab5190eed1b989684f33153edd61a43ee7c9430.jpg)  
Figure 28 (continued). PushT, K = 5. Steps 15–27, ending in success. The shared endpoint connects the two pages.

![](images/068b918a1c3344e6be198b9976cf89a0feaab8b308c62a47a0c073c42e44d2ed.jpg)  
Figure 29: Complete PushT trajectory at K = 10. Selected successful Far episode; all 20 actions and their states are shown. Blue rows show predictions; teal inputs and feedback arrows show the real observations supplied between chunks. Actions give normalized target coordinates (x, y).

![](images/c18e3a9e77413ae9c3a5b6314091d15315cf49834a342bb2db0f55ed39a140ca.jpg)  
Figure 30: Complete PushT trajectory at K = ∞. Selected successful Far episode; this page shows steps 0–15 of 28. The trajectory continues on the next page. The model receives only the initial observation, with no intermediate feedback. Ground-truth states are shown for comparison. Actions give normalized target coordinates (x, y).

![](images/2ce208256a269f92e077a4436e6728f9a9cac8378f6c7b3338da0abacf360c0e.jpg)  
Figure 30 (continued). PushT, K = ∞. Steps 15–28, ending in success. The shared endpoint connects the two pages; no new observation is supplied.

## D.5 ROLLOUTS FOR DIFFERENT REASONING FORMATS

![](images/fef2dca92ed969fe4a15b872ebf315e33150abeaaa45a5ad56d0f374fbcc4fac.jpg)  
Figure 31: Counterfactual Branching. Example trace of counterfactual branching in Sokoban.

![](images/c61c46cb68fdb81cd66bd9cb11ad217c25f030da6dca7b4461b0e1280fad20a2.jpg)  
Figure 32: Forward Thinking. Example trace of Forward thinking in Sokoban.

## D.6 RL TRAJECTORIES ACROSS CHECKPOINTS

We show complete trajectories at RL50, RL100, RL150, and RL200 on the three- and four-box Deadlock and Structural Fork sets. Each group uses one fixed initial state across all four checkpoints. The Deadlock examples use Agima (Forward), and the Structural Fork examples use Agima (Branching).

Blue rows show model-generated states. Ground-truth rows show execution of the committed plan; unexecuted candidates have no ground-truth row. Orange panels show the alternatives rejected by the model and identify where each branches from the main plan. Quoted assessments are the model’s own text. The recorded execution result is reported separately. All saved generated frames, executed states, and actions are shown.

Table 10: Outcomes of the illustrated RL episodes. Each row follows one selected initial state across four checkpoints. Aggregate performance is reported in Figure 4.
<table><tr><td>Test set</td><td>Model</td><td>RL50</td><td>RL100</td><td>RL150</td><td>RL200</td></tr><tr><td>3-box Deadlock</td><td>Forward</td><td>Fail</td><td>Success</td><td>Success</td><td>Success</td></tr><tr><td>4-box Deadlock</td><td>Forward</td><td>Fail</td><td>Fail</td><td>Success</td><td>Success</td></tr><tr><td>3-box Structural Fork</td><td>Branching</td><td>Fail</td><td>Success</td><td>Success</td><td>Success</td></tr><tr><td>4-box Structural Fork</td><td>Branching</td><td>Fail</td><td>Fail</td><td>Success</td><td>Success</td></tr></table>

## RL50 | Forward

3-box Deadlock

Full trace | 21 generated frames | 10 executed actions | episode 84  
RL50, 3-box Deadlock | Candidate 1 | committed | steps 0-5<sub>xecuted</sub> <sub>actions</sub> <sub>|</sub> <sub>episode</sub> <sub>84</sub>  
![](images/1d9ad4b3a0ef35e65fcc9d73fea984965c4ba2db130efb56c2766a3c09e9598a.jpg)

RL50, 3-box Deadlock | Candidate 1 | committed | steps 5-10  
![](images/a2c26a6f33342b3c03893cfdd43af87a2737ab77837487c84fad8f43df4971db.jpg)  
Model assessment: "this plan succeeds; all boxes reach their targets in 9 moves."

1 <sup>o</sup>2 <sup>o</sup>3 <sup>o</sup>4RL50, 3-box Deadlock | Candidate 2 | not executed | steps 0-6  
![](images/6ed274951c9865ca969b5249c38442e9a37d62c71c18c32b1ba47b997da46486.jpg)

Full trace | 11 generated frames | 11 executed actions | episode 84

RL100, 3-box Deadlock | Main plan | committed | steps 0-6  
RL100, 3-box Deadlock | Main plan | committed | steps 6-11  
RL50, 3-box Deadlock | Candidate 2 | not executed | steps 6-11  
![](images/b81ccce12237f4c83f51241be562b3dc19bf7912bacda4f2b34bd06f7bc289bb.jpg)  
Model assessment: "this plan succeeds; all boxes reach their targets in 10 moves."

Model comparison: "both candidates solve the puzzle, but Candidate 1 is shorter, requiring 9 moves instead of 10. I will commit Candidate 1."

Figure 33: Forward at RL50 on three-box Deadlock. Complete trace for episode 84; recorded execution ends in failure.

## RL100 | Forward

3-box Deadlock

![](images/393932a58023a70b5914dd855b052007c7c5f5d39774a3fd789d1fa4fb64508e.jpg)

![](images/ac370e5045a43a3cbd8c06110a41378350f7b59fa7991ef512bdb9b5376038b5.jpg)  
Figure 34: Forward at RL100 on three-box Deadlock. Complete trace for episode 84; recorded execution ends in success.

## RL150 | Forward

3-box Deadlock

Full trace | 11 generated frames | 11 executed actions | episode 84

![](images/89be53440d29c8abdd7da53435bced8b0dbbaef52ed7b488e38d4ec730278948.jpg)

RL150, 3-box Deadlock | Main plan | committed | steps 6-11  
![](images/e3780959a5d848da3655223d8627d2eba77c68607914d0ef4db02a8eab597d0c.jpg)  
Figure 35: Forward at RL150 on three-box Deadlock. Complete trace for episode 84; recorded execution ends in success.

## RL200 | Forward

3-box Deadlock

Full trace | 11 generated frames | 11 executed actions | episode 84  
RL200, 3-box Deadlock | Main plan | committed | steps 0-6  
![](images/59faf74a81bafdd666eea1824dab60742160d1c22863a8d0c04fe99c523e7049.jpg)

RL200, 3-box Deadlock | Main plan | committed | steps 6-11  
![](images/51800a3ded028d1f118e4a4ba320c82bc8b855685a7803e56e26a2cd078924ac.jpg)  
Figure 36: Forward at RL200 on three-box Deadlock. Complete trace for episode 84; recorded execution ends in success.

## RL50 | Forward

4-box Deadlock

Full trace | 34 generated frames | 14 executed actions | episode 21

RL50, 4-box Deadlock | Candidate 1 | committed | steps 0-5  
![](images/f0158318f81e02a1f8755a083c254ab89a27a03595c1eeb36cbb032d77de9fe6.jpg)

RL50, 4-box Deadlock | Candidate 1 | committed | steps 5-10Left Down Down  
![](images/b1a1695338351674532ed15932fdec7f5bb01a6b73a62bcee35b08fa46081ea1.jpg)  
Figure 37: Forward at RL50 on four-box Deadlock. Complete trace for episode 21; recorded execution ends in failure.

## RL100 | Forward

Full trace | 34 generated frames | 14 executed actions | episode 21  
RL100, 4-box Deadlock | Main plan | not executed | steps 0-5<sub>executed</sub> <sub>actions</sub> <sub>|</sub> <sub>episode</sub> <sub>21</sub>  
![](images/f51fd104aa794dfc1ede5553f08b06e2252df6ce693f998ece3f0d0ce0656f49.jpg)

RL100, 4-box Deadlock | Main plan | not executed | steps 5-10Left Down Left  
![](images/333362431e7d69f9d182e318a5e6ff7d17bc7ab4da92f5c9820832e591dbc2be.jpg)

11 <sup>o</sup>12 <sup>o</sup>13 <sup>o</sup>14RL100, 4-box Deadlock | Main plan | not executed | steps 10-15Down Right Right  
![](images/6b96c2f1bfc974b1646519fe84b3b64929fa67decc3fa6f1b1514ba11a95e514.jpg)

16 <sup>o</sup>17 <sup>o</sup>18 <sup>o</sup>19RL100, 4-box Deadlock | Main plan | not executed | steps 15-20  
![](images/69770f2ecaa6e5505a43dced954dac0458e563af316b610e95e62d7884aaeb34.jpg)  
RL100, 4-box Deadlock | Candidate 2 | committed | steps 0-5Model assessment: "this plan has not solved the puzzle after 20 moves. I stop evaluating this candidate at the task horizon."

1 <sup>o</sup>2 <sup>o</sup>3 <sup>o</sup>4RL100, 4-box Deadlock | Candidate 2 | committed | steps 0-5  
![](images/dfc1ccad2e96a576e010612b62a7540bab345a8719190bc6d8feac822fdea236.jpg)

6 <sup>o</sup>7 <sup>o</sup>8 <sup>o</sup>9RL100, 4-box Deadlock | Candidate 2 | committed | steps 5-10  
![](images/aa97f745da9b68da2a474aad4038b34564e270a7423568f24aa6639beed82b4e.jpg)

RL150, 4-box Deadlock | Main plan | committed | steps 5-10  
RL200, 4-box Deadlock | Main plan | committed | steps 0-5  
RL100, 4-box Deadlock | Candidate 2 | committed | steps 10-14  
![](images/84829512e236d4c01d69993e2312eb626a3f220755b134f93ffc44ea0fc92e54.jpg)  
Model assessment: "this plan succeeds; all boxes reach their targets in 13 moves."

Figure 38: Forward at RL100 on four-box Deadlock. Complete trace for episode 21; recorded execution ends in failure.

4-box Deadlock

Full trace | 10 generated frames | 10 executed actions | episode 21

![](images/8be949ff6897767673f8f47ddeca93be3781ced213f859044c4ac19557c7a3ef.jpg)

![](images/080494c17d74bb6c45f0685947d33f6707e9cb816646ee85617b80663774092d.jpg)  
Figure 39: Forward at RL150 on four-box Deadlock. Complete trace for episode 21; recorded execution ends in success.

RL200 | Forward

4-box Deadlock

Full trace | 10 generated frames | 10 executed actions | episode 21

![](images/1fed0638a56ec44a2f4575eb76f34ea6541fec4201ee28db95a3ddff823b80e9.jpg)

![](images/ca999a99536d89008c4110ce75340bc760f89949fc3a76e4a2fe29a3e7d8f019.jpg)

RL200, 4-box Deadlock | Main plan | committed | steps 5-10  
![](images/54b788cf7aef3e47f68ef092e4388a42c5f1e8c549cf5a1775f02ec93fdf5801.jpg)  
Figure 40: Forward at RL200 on four-box Deadlock. Complete trace for episode 21; recorded execution ends in success.

## RL50 | Branching

3-box Structural Fork

Full trace | 9 generated frames | 8 executed actions | episode 17

RL50, 3-box Structural Fork | Main plan | committed | steps 0-4<sub>xecuted</sub> <sub>actions</sub> <sub>|</sub> <sub>episode</sub> <sub>17</sub>  
![](images/54fac37aa5bfe723de19b903e198b500bcafdccf939d89b33507902e6d81d516.jpg)  
RL50, 3-box Structural Fork | Main plan | committed | steps 4-8

<sub>Probe</sub> <sub>1</sub> <sub>at</sub> <sub>main</sub> <sub>step</sub> <sub>2</sub>RL50, 3-box Structural Fork | Counterfactual probes | rejected by the model  
![](images/f9f65755bf86e616344dc7f6e85d12316cee044adf287e81e00d5885d7be0e81.jpg)  
Figure 41: Branching at RL50 on three-box Structural Fork. Complete trace for episode 17; recorded execution ends in failure.

## RL100 | Branching

3-box Structural Fork

Full trace | 19 generated frames | 16 executed actions | episode 17

RL100, 3-box Structural Fork | Main plan | committed | steps 0-6  
![](images/5541142f06c17c8e41caa8ec05e1b231f80e65315cdc68c5eb1256cfcd27033a.jpg)

RL100, 3-box Structural Fork | Main plan | committed | steps 6-11  
![](images/ac32ba5eb8b2086ae43826636be2332dc9db62abf901c8a9723e799cee51893f.jpg)

RL100, 3-box Structural Fork | Main plan | committed | steps 11-16  
![](images/9ed64936776e9f881d8887b125c9409e884ba9154e3a94b1f7058903f3ee193a.jpg)

<sub>Probe</sub> <sub>1</sub> <sub>at</sub> <sub>main</sub> <sub>step</sub> <sub>2 Probe</sub> <sub>2</sub> <sub>at</sub> <sub>main</sub> <sub>step</sub> <sub>9</sub>RL100, 3-box Structural Fork | Counterfactual probes | rejected by the model  
![](images/1282ad137774401a9685030ced69902707a1d677170c7397d024f5aaefe2db78.jpg)  
Figure 42: Branching at RL100 on three-box Structural Fork. Complete trace for episode 17; recorded execution ends in success.

## RL150 | Branching

3-box Structural Fork

Full trace | 19 generated frames | 16 executed actions | episode 17

RL150, 3-box Structural Fork | Main plan | committed | steps 0-6  
![](images/3be2f9c3200ce2d97f38b09d5e2346f8525554305a23b2090a902f45f5745637.jpg)

RL150, 3-box Structural Fork | Main plan | committed | steps 6-11  
![](images/afdd94f0874b25875ddd320118b820e0515cf4f5a8a914c4bbf6db84656e7f48.jpg)

RL150, 3-box Structural Fork | Main plan | committed | steps 11-16  
![](images/267deb1c74713b38db9f7d23320c7253435620a67a05c97b3468f90e214e2e57.jpg)

RL150, 3-box Structural Fork | Counterfactual probes | rejected by the model  
![](images/c0222b9b533cdd22c297612f82dce5abad29315b39f162da0e2ae3d7c4f90ac8.jpg)  
Figure 43: Branching at RL150 on three-box Structural Fork. Complete trace for episode 17; recorded execution ends in success.

## RL200 | Branching

3-box Structural Fork

Full trace | 19 generated frames | 16 executed actions | episode 17

RL200, 3-box Structural Fork | Main plan | committed | steps 0-6<sub>16</sub> <sub>executed</sub> <sub>actions</sub> <sub>|</sub> <sub>episode</sub> <sub>17</sub>  
![](images/856833d97b9be521b08715e28169d200c3b22de3848e77acf33a236ccfa0663d.jpg)

RL200, 3-box Structural Fork | Main plan | committed | steps 6-11  
![](images/252c6efa94ffbfa1d9928e63c7c06a863cd0d50d29bc2adfb26335cdba824e22.jpg)

<sup>o</sup>12 <sup>o</sup>13 <sup>o</sup>14 <sup>o</sup>15RL200, 3-box Structural Fork | Main plan | committed | steps 11-16  
![](images/e8ec26b35cb92eb2a0842e244634ce7ad14103faa15659f5651137b6f751d5bd.jpg)

![](images/6b8d38587dda393a0ec125167e762157d45ec10fbd0fc45ecfffc94a18d5f324.jpg)

RL200, 3-box Structural Fork | Counterfactual probes | rejected by the model  
![](images/19277169783cb79d5230b04caef59c9a3fb6a2e6902a5a529e4548cf43fd876e.jpg)  
Figure 44: Branching at RL200 on three-box Structural Fork. Complete trace for episode 17; recorded execution ends in success.

## RL50 | Branching

4-box Structural Fork

Full trace | 13 generated frames | 11 executed actions | episode 56

RL50, 4-box Structural Fork | Main plan | committed | steps 0-6<sub>1</sub> <sub>executed</sub> <sub>actions</sub> <sub>|</sub> <sub>episode</sub> <sub>56</sub>  
![](images/e8b4390cf81ef1473e29c932786e36f6d802c4012f9f47f6a4d3589eaf0c6646.jpg)

RL50, 4-box Structural Fork | Main plan | committed | steps 6-11  
![](images/8816af2e9210fe1cd6e9c2a4d953c20a7c14d5c4a549b98debf7b5c4cb71b23c.jpg)  
<sub>Probe</sub> <sub>1</sub> <sub>at</sub> <sub>main</sub> <sub>step</sub> <sub>3</sub>RL50, 4-box Structural Fork | Counterfactual probes | rejected by the model  
Figure 45: Branching at RL50 on four-box Structural Fork. Complete trace for episode 56; recorded execution ends in failure.

## RL100 | Branching

4-box Structural Fork

Full trace | 13 generated frames | 11 executed actions | episode 56

RL100, 4-box Structural Fork | Main plan | committed | steps 0-6  
![](images/c1e85aeab82126c018458f0ee0ddf91bc46ec42e22ca363c596e3047d76b9b25.jpg)

RL100, 4-box Structural Fork | Main plan | committed | steps 6-11  
![](images/7901b9767e2c6949fdb00aad0d10dfad4961786300c51e5855255ee008d01189.jpg)

RL100, 4-box Structural Fork | Counterfactual probes | rejected by the model  
![](images/d90167e0738094d47e8060b075f52e32c7c7408ef145f1f7a15a6225f1005a4d.jpg)  
Figure 46: Branching at RL100 on four-box Structural Fork. Complete trace for episode 56; recorded execution ends in failure.

## RL150 | Branching

4-box Structural Fork

Full trace | 12 generated frames | 10 executed actions | episode 56  
RL150, 4-box Structural Fork | Main plan | committed | steps 0-5  
![](images/cb54638dafbb9a17b4a308fb615f4b10e4749abafa29ba7bd52751eb08e18ae1.jpg)

RL150, 4-box Structural Fork | Main plan | committed | steps 5-10  
![](images/9b0c0ee18840558615e568442ef997d0c667523e139003c4bc58178cb00515fa.jpg)

RL150, 4-box Structural Fork | Counterfactual probes | rejected by the model  
![](images/67ba57f3aa3bb0ce28389ff1f8ef7efbc7c3966e1602c70a926cef27e303514e.jpg)

![](images/78232c3ae603d648945af2324955edb6f01a6836bd5876d535239ac983b0132b.jpg)  
Figure 47: Branching at RL150 on four-box Structural Fork. Complete trace for episode 56; recorded execution ends in success.

RL200 | Branching Full trace | 11 generated frames | 10 executed actions | episode 56  
![](images/88a95d9db7f31a4ede1d781ef348ca143839b33795a7260fb4d31ea34081a8b5.jpg)  
Figure 48: Branching at RL200 on four-box Structural Fork. Complete trace for episode 56; recorded execution ends in success.

## D.7 FAILURE CASES

The following recorded K = ∞ episodes show failures in all four environments: stopping before reaching the goal in Maze2D, leaving boxes off their goals in Sokoban, being caught by the ghost in Pacman, and stopping with insufficient target coverage in PushT. These examples use the same task settings and checkpoints as the K = ∞ evaluations in Section 3.

The real execution can also disagree with the model’s outcome assessment. For example, the RL50 Forward model describes its chosen plan as successful in Figure 33, but the executed plan fails. In the Pacman example below, execution ends at the collision. Later predicted frames remain visible, with the corresponding real states marked as not executed.

## Maze2D | Failure

Full trace | 11 generated frames | 11 executed actions | episode 1  
Maze2D | Failure, K = ∞ | Main plan | committed | steps 0-6  
![](images/f0f0853b531d7f42f518453ffac3c7d3c563e48dca5ca8c2895c5cb2f5090b8a.jpg)

Maze2D | Failure, K = ∞ | Main plan | committed | steps 6-11  
![](images/a606428f0e08c06b278ad91dfb3dda5a7aa249f8079d07a41e7fa6a455c4f88b.jpg)  
Figure 49: Maze2D failure at $K = \infty .$ . The agent issues STOP before reaching the goal.

## Sokoban | Failure

Full trace | 14 generated frames | 14 executed actions | episode 0 Sokoban | Failure  
![](images/32cfb4192bdea0e27f4ce3e80584d2e9f14747cfd3ab3fd049a8d59a46029e06.jpg)  
Figure 50: Sokoban failure at $K = \infty .$ The agent stops with 2 of 4 boxes on goals.

## Pacman | Failure

Full trace | 24 generated frames | 17 executed actions | episode 27

![](images/d28aeada7ad0888bdc6754e33916fc51ab6a947d86d1d15e9302cc1027c91639.jpg)

Pacman | Failure, K = ∞ | Main plan | committed | steps 6-12  
![](images/801c8867b7f069dc57cc81869410e1df967aeee2aca846596c1528a093ed3546.jpg)  
Figure 51: Pacman failure at K = ∞. The ghost catches Pacman at action 17. The remaining seven predicted states are never executed.

PushT | Failure

Full trace | 15 generated frames | 15 executed actions | episode 9

![](images/1c0f1a6005ba2ab8ab0ee737c0bf54a4ba8edd1ba85b77bb40f5f4ae02c00fbe.jpg)

![](images/e4d0438faad2a43daa9157f45a1c701969cf66edae6273eedb4895550447e3f7.jpg)  
Figure 52: PushT failure at $K = \infty .$ . The agent stops with 32.4% final target coverage, below the paper’s 80% success threshold.