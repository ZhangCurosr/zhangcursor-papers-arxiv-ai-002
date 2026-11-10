# Learning to Act with World Models by Valuing What Comes Next

Chenghao Gao<sup>1∗</sup> Zhi Zhang<sup>2∗</sup> Dian Jin<sup>1</sup> Yihe Pan<sup>3</sup> Doudou Zhou<sup>1†</sup>

<sup>1</sup>Department of Statistics and Data Science, National University of Singapore

<sup>2</sup>Department of Statistics and Data Science, University of California, Los Angeles

<sup>3</sup>Department of Data Science, Fudan University

<sup>∗</sup>Equal contribution. <sup>†</sup>Corresponding author: doudouzhou@nus.edu.sg

## Abstract

A sparse terminal reward tells an agent little about its steps. Much of the missing information is implicit in the environment’s dynamics, because each state transition can change the conditions for success long before the reward arrives. A world model trained on real transitions can learn to predict what each action changes, but a predicted observation does not by itself show whether success has become more likely. With a binary terminal reward, PPO’s critic makes this judgment by estimating the current policy’s probability of success, but only for states the agent reaches. Our method, WM-PPO, elegantly brings the two together. For each action the policy samples, executed or not, the world model predicts the next observation and the critic values the predicted successor. In the policy channel, comparisons among these values give the policy auxiliary advantages; in the value channel, their mean gives the critic an auxiliary target. A shared gate weights both channels according to how reliably comparisons with similar margins predicted real outcomes earlier in training. The policy, critic and world model all learn from the agent’s own interaction, with no teacher or privileged information, and only the policy is used at inference. With a comparable number of real environment steps, WM-PPO raises success from PPO’s 63.5% to 87.5% on ALFWorld and from 39.3% to 52.9% on ScienceWorld. It matches the strongest PPO refinements we tested on ALFWorld and exceeds the best of them by 6.5 points on ScienceWorld. The trained policy more often completes prerequisite steps before placing an object or committing to an answer.

## 1 Introduction

Agents trained with reinforcement learning on multi-step tasks often receive only sparse terminal rewards, which provide limited feedback for learning. Yet each state transition can change the conditions for success long before the reward arrives. A needed object is found, an operation’s precondition is met, or a command is rejected, and the next observations record all these changes. When tasked with cleaning a plate and putting it in a cabinet, for example, an agent may find the plate and pick it up, but put it away without first cleaning it at the sink. The final reward marks the whole attempt as a failure. It does not say which steps helped, or which other action would have left the agent better placed to succeed.

World models learn environment dynamics by predicting how an agent’s actions change the environment. What a predicted change means for success depends on the task and on what the current policy can accomplish from the resulting state. In text environments, an action’s contribution to success is not necessarily reflected in how much the observation text changes. A transition that produces a very different textual observation may leave the chances of success unchanged: opening a cabinet, for example, may reveal a long list of objects unrelated to the goal. Conversely, a small change in the environment may establish a critical prerequisite or make the goal unreachable. Heating a mug, for instance, may produce only a brief confirmation, yet satisfy a necessary condition for a task that requires a hot mug. We therefore evaluate predicted transitions by their effect on eventual success.

Fortunately, PPO (Schulman et al., 2017) already learns such a valuation through its critic. The critic estimates the expected cumulative reward from a state under the current policy, and its evaluation evolves as the policy learns. When agents must learn from sparse terminal success rewards, the critic offers feedback before the task outcome is known. Its values estimate the probability of eventual success, and changes across a transition reveal how an action affects that probability. While PPO uses these estimates only through the advantages of executed actions, we apply the critic to the next states predicted by a world model for the policy’s candidate actions (Figure 2). These estimates, which we call transition values, let us compare the policy’s candidates through their predicted consequences. Diagnostics on ALFWorld (Shridhar et al., 2021) support this idea: selecting among the policy’s candidates using these values achieves higher success than direct policy sampling (Section 2).

We therefore introduce WM-PPO, which incorporates this feedback through policy and value channels. At each state along the main training trajectory, the policy samples several candidate actions and executes the first as usual. The policy channel turns candidate comparisons into auxiliary advantages in PPO’s clipped objective, while the value channel uses the mean transition value as an additional one-step target for the critic at the current state. These critic updates also affect the advantages used in subsequent policy updates. Since value estimation in LLM reasoning can be biased and yield unreliable comparisons between candidate steps (Yuan et al., 2025; Kazemnejad et al., 2025), WM-PPO calibrates both channels against real outcomes. For each training task, it collects one main episode and three branches, each trying a different action at a randomly chosen step. A shared gate uses earlier branch outcomes to weight both channels according to the margin between transition values (Section 3). The design stays elegant: the world model is the only learned component added to PPO and reuses the real transitions collected during training. Real returns remain the critic’s main supervision. Training requires no teacher model or privileged information, and the policy acts alone at inference.

Prior work also draws on signals beyond terminal rewards to improve language agents (Table 3 in Section 6). Some methods add observation-prediction losses during RL (Shrivastava et al., 2026; Lu et al., 2026; Ran et al., 2026), while others learn environment dynamics before downstream policy training (Zhang et al., 2026a; Chen et al., 2026a; Yu et al., 2026b). Distillation methods obtain finer feedback from teacher distributions or from models given hindsight observations or reference solutions (Agarwal et al., 2024; Yang et al., 2026; Wu et al., 2026). World models also provide imagined outcomes for action selection and refinement in WMA, ITP, and CoMAP, where they remain part of the deployed agent (Chae et al., 2025; Liu et al., 2026a,b). Internalizing the Future (Zhang et al., 2026b) learns from teacher-generated foresight traces. Its policy expresses confidence in an imagined plan before generating an action, and this confidence is calibrated against task outcomes to construct step-level advantages. ProAct’s MC-GRPO (Yu et al., 2026c) compares actions sampled at the same state using values estimated from simulator rollouts. WM-PPO instead evaluates candidates through their predicted consequences using a world model and PPO’s existing critic. Both networks learn from the agent’s own interactions, turning predicted state changes into feedback for policy and value learning.

We evaluate WM-PPO on the simpler ALFWorld and on the harder ScienceWorld (Wang et al., 2022), whose tasks often require establishing intermediate conditions before the agent commits to an answer. The comparisons test whether learning from predicted consequences improves success for a comparable amount of real interaction (Section 4). They cover PPO, its multi-turn refinements and group-based methods; structural controls that change one component at a time; and auxiliary supervision from observation prediction, agreement and teachers. The main table compares 17 methods and variants, and the interaction-efficiency analysis places WM-PPO among 23 other methods and settings on ALFWorld and 20 on ScienceWorld. We then analyze how the two channels affect learning, and how the trained policy advances a task (Section 5).

Our contributions are threefold:

• Learning from dynamics by valuing predicted consequences. We show that valuing worldmodel predictions with PPO’s critic turns information implicit in environment dynamics into feedback that supplements sparse terminal rewards. Diagnostics demonstrate that these transition values support both action selection and policy improvement.

• Calibrating value-based feedback with real outcomes. To account for noisy critic estimates, we use outcomes of real branches to calibrate a shared gate that weights both channels by how often comparisons with similar margins agreed with real outcomes earlier in training. The critic’s values thus reach both the policy and the critic itself as a stable signal grounded in real outcomes.

• WM-PPO: transition values for policy and critic learning. We introduce two complementary learning channels: comparisons among transition values provide auxiliary policy advantages, while their mean supplies an auxiliary critic target. Only the policy is needed at inference. On ALFWorld and ScienceWorld, WM-PPO reaches PPO’s success levels with fewer real interaction steps and achieves higher success at comparable interaction budgets.

## 2 From Predicted Consequences to Learning Signals

We first relate state transitions to changes in the probability of success, then test whether predicted consequences can guide a policy update.

Setting. A task is a natural-language goal in a text environment. At step t the agent is in state $s _ { t } ,$ by which we mean the content of its prompt: the goal, its five most recent observations and actions, the current observation, and the actions or objects currently available. The policy $\pi _ { \theta }$ samples a short response $y _ { t }$ , from which one action $a _ { t }$ is parsed. The environment returns the next observation $o _ { t + 1 }$ and the next state $s _ { t + }$ adds $a _ { t }$ and $o _ { t + 1 }$ to the history and refreshes the list of available actions or objects. The value of a state under the policy π is the expected return that follows it,

$$
V ^ { \pi } ( s _ { t } ) = \mathbb { E } _ { \pi } { \Big [ } \sum _ { k \geq 0 } \gamma ^ { k } r _ { t + k + 1 } ~ { \Big | } ~ s _ { t } { \Big ] } ,\tag{1}
$$

where $\gamma$ is the discount factor, r denotes reward, and we treat $s _ { t }$ as sufficient for predicting returns. In the environments we study, the only reward is the final reward $R \in \{ 0 , 1 \}$ , given at the last step: 1 if the task was completed within 40 steps and 0 otherwise. With $\gamma = 1$ , the value is then the probability of eventually succeeding,

$$
V ^ { \pi } ( s _ { t } ) = \mathbb { E } _ { \pi } { \big [ } R \mid s _ { t } { \big ] } = \operatorname* { P r } _ { \pi } { \big ( } { \mathrm { s u c c e s s ~ } } | \ s _ { t } { \big ) } ,\tag{2}
$$

and we write $V ^ { \pi } ( s ) = R$ for a state that ends the episode. The superscript marks that these values belong to the current policy π, so they can change as the policy learns. An intermediate transition from $s _ { t } \ \mathrm { t o } \ s _ { t + 1 }$ receives no reward, yet it changes this probability by $V ^ { \pi } ( s _ { t + 1 } ) - V ^ { \pi } ( s _ { t } )$ . The observation records what changed in the environment; the difference in state values measures how that change affects the current policy’s probability of success. The probability of success after an action is $Q ^ { \pi } ( s _ { t } , a ) = \mathbb { E } [ V ^ { \pi } ( s _ { t + 1 } ) \mid s _ { t } , a _ { t } = a ]$ , and the action’s advantage is the expected change in state value caused by the action,

$$
\begin{array} { r } { A ^ { \pi } ( s _ { t } , a ) = Q ^ { \pi } ( s _ { t } , a ) - V ^ { \pi } ( s _ { t } ) = \mathbb { E } \big [ V ^ { \pi } ( s _ { t + 1 } ) - V ^ { \pi } ( s _ { t } ) \big \vert ~ s _ { t } , a _ { t } = a \big ] . } \end{array}\tag{3}
$$

At a fixed state, the difference between two actions’ advantages is therefore the difference between their expected success probabilities. PPO estimates this advantage only for the executed action, through the critic’s values along the executed path (Appendix B.2). Estimating it for the alternatives requires their transitions, which a world model can predict.

The critic $V _ { \psi } ( s _ { t } )$ estimates $V ^ { \pi } ( s _ { t } )$ , and a world model $p _ { \phi } ( o _ { t + 1 } \mid x _ { t } , a _ { t } )$ predicts the next observation from its own view $x _ { t }$ of the history (Section 3.1). All models are LoRA adapters (Hu et al., 2022)

![](images/61ec4ff2e49c225e4dfa0618208fcce1b38a9e8d21608591ac90324b0c8636e3.jpg)

![](images/bbed7317c43d3e3ced77e5296a889d92c1b735ddba47631cf7a419e273ffaf0b.jpg)

![](images/a8946e6cf483b73caff6cf41a12bf2bbcbe4d770186dddc0363b14e280d5308d.jpg)  
Figure 1: Valuing successors and learning from their comparisons on ALFWorld (PPO policy and critic after 24 iterations; 274 evaluation games). (a) Success when the agent executes one of four sampled candidates, chosen by majority vote or by the critic’s value of the candidate’s next state; the next state is predicted by an untrained or a trained world model, or obtained from the environment. (b) Agreement of the critic’s ordering with estimated success probabilities for candidate pairs at the same state (195 pairs), split by the gap between the critic’s two values. The last bar contains the pairs in which one candidate succeeded in all three continuations and the other in none; dotted line: chance. (c) Policy-only success after one offline PPO-clip epoch whose labels come from the world model and the critic or from majority vote (temperature 1, two episodes per game); dashed line: test-time selection with the trained world model from (a). Horizontal axes start at 40%.

on Qwen3-4B-Instruct-2507 (Yang et al., 2025). The diagnostics use ALFWorld (Shridhar et al., 2021) and the policy and critic of a PPO run continued from the common warm-up of Section 4.1 to iteration 24 (53.6% success on the 274 evaluation games); all evaluations are paired by game.

Equation (3) suggests three tests. The critic must order the successors of different actions by their probability of success; predicted successors must preserve these differences for actions that were not taken; and updates built on the comparisons must improve the policy when it acts alone. We test these conditions before introducing the online method.

## 2.1 Does the critic see the value of a state transition?

We first test whether the critic can compare actions at the same state by valuing their real successors. At every step, the policy samples four candidate actions. Each distinct candidate is executed for one step in a copy of the environment, and the critic values each resulting next state. The agent then executes the candidate with the highest-valued successor, which is expected to leave the greatest chance of success (Equation (3)). The policy is held fixed.

Selecting by the critic’s value of the real successor raises success from 56.9% (majority vote over the four candidates) to 70.4%, a gain of 13.5 points (SE 2.8, $p < 1 0 ^ { - 4 }$ ; Figure 1a); executing a single sampled candidate gives 54.4%. Most of this gain comes from valuing intermediate successors: a control that selects a candidate only when it completes the task immediately, and otherwise uses majority vote, reaches 59.5%. The critic therefore carries information about intermediate transitions that the agent’s own choices do not use.

The reliability of this ranking depends on the gap between the critic’s values. Prior work reports noisy and initially biased value estimates that rank candidate next steps near chance (Kazemnejad et al., 2025; Yuan et al., 2025; Yue et al., 2025). At randomly chosen states, we executed each distinct candidate and estimated its success probability from three continuations. Of the 195 candidate pairs whose estimates differ, the critic orders 57% correctly, and the median gap between its two values is 0.008. Ranking accuracy increases from 53% in the lower half of the critic’s value gaps to 71% in the top quarter. Among the 12 pairs for which one candidate succeeded in all three continuations and the other in none, the critic ranked 11 correctly (Figure 1b). Large value gaps identify the comparisons the critic orders most reliably. The reliability gate uses real outcomes to determine how much weight each margin should receive (Section 3.5).

## 2.2 Do predicted state transitions preserve this value?

We next test whether learned dynamics make these values available for candidates that are not executed. A world model replaces the one-step execution: a separate LoRA adapter on the same base model, trained with a next-observation loss on transitions that the same policy generated on 1,000 training games. For this diagnostic only, its data also include one-step executions of candidates the policy did not choose (48k transitions, 44% of them for such candidates); the world model of WM-PPO is trained on executed transitions alone. At each step, the world model greedily predicts the next observation of every distinct candidate, and the critic values each predicted next state. The agent selects the candidate with the highest value and executes it in the environment; the other candidates are never executed.

Choosing by predicted consequences reaches 66.4%, 9.5 points above majority vote $( \mathrm { S E } 2 . 5 , p <$ 0.001; Figure 1a) and close to the 70.4% obtained with the true consequences. With the untrained base model as the world model, selection reaches 61.7%; the trained model is 4.7 points higher. Two controls examine why selection with true consequences performs better. When the true next observation is combined with the action list from before the step, as in the world model’s successor, selection reaches 65.3%. When newly revealed objects are removed from the true observation, it reaches 59.1% (Appendix C). Both the refreshed action list and the newly revealed content contribute to the gain. The predictions are exact for 99.7% of the held-out observations that follow from what the agent has already seen, but for only 26% of those that reveal objects it has not seen; for the latter, they still give a prior over what will be found (Appendix C). In this diagnostic, predicted transitions preserve most of the selection gain obtained with real ones.

## 2.3 Can the policy learn from these values?

The final test asks whether the comparisons produced by the world model and critic can improve the policy itself. Starting from the same checkpoint, the policy played 500 new training games, executing the first of four samples at each step. At the 8,042 decision points with at least two distinct candidates, the candidate $\hat { a } _ { t }$ preferred by the world model and the critic served as a label, and each of the four samples $a _ { t } ^ { ( 1 ) } , \ldots , a _ { t } ^ { ( K ) } \left( K = 4 \right)$ received the advantage

$$
\tilde { A } _ { t , k } = z _ { t , k } - \bar { z } _ { t } , \qquad z _ { t , k } = { \bf 1 } \big [ a _ { t } ^ { ( k ) } = \hat { a } _ { t } \big ] , \qquad \bar { z } _ { t } = \frac { 1 } { K } \sum _ { k ^ { \prime } = 1 } ^ { K } z _ { t , k ^ { \prime } } .\tag{4}
$$

These advantages sum to zero across the samples at each decision point. One epoch of PPO-clip on these samples used no task reward. The updated policy was then evaluated alone, at temperature 1, twice on the 274 games.

The update raises policy-only success from 57.1% to 65.3% (+8.2 points, SE 2.0, $p = 0 . 0 0 0 3 ;$ Figure 1c). Using majority-vote labels on the same data for the same number of updates gives 56.9%, no change from 57.1%, although the two sets of labels agree at 44% of the decision points. With the data and update count held fixed, the consequence-based labels supply the information that improves the policy. The gain persists when the updated policy acts alone, without the world model or the critic.

From diagnostics to an online method. Starting from a fixed policy, the diagnostics show that the critic can compare actions by valuing their successors, that the world model extends this evaluation to unexecuted candidates, and that the policy learns from the resulting comparisons. Reliability depends on the value margin, so the comparisons must be calibrated against real outcomes. As the policy improves, it visits new states and proposes new actions, requiring the world model, critic and reliability estimates to be updated. WM-PPO brings these steps into online training and uses the transition values for both candidate comparisons and critic targets.

## 3 WM-PPO

WM-PPO adds one learned component to PPO: a world model $p _ { \phi }$ trained on real state transitions to predict the next observation after an action. The policy $\pi _ { \theta }$ keeps PPO’s clipped objective, and the critic $V _ { \psi }$ learns primarily from real returns. At each training state, the world model predicts the consequences of the policy’s candidate actions, and the critic values the predicted successors. These transition values enter learning through two channels. Comparisons among the candidates supply auxiliary advantages for the policy (the policy channel), and the mean transition value supplies a one-step target for the critic (the value channel). Small rollout trees provide real branch outcomes, and a gate uses earlier branches to weight both channels by the reliability of their value margins. Section 3.1 defines the evaluation, Sections 3.2 and 3.3 the two channels, Section 3.4 the training loop, and Section 3.5 the gate. Figure 2 gives an overview, and Algorithm 1 summarizes one iteration (Algorithm 2 in Appendix B.1 gives the full procedure).

## 3.1 Evaluating predicted state transitions

The state $s _ { t }$ is what the agent sees in its prompt at step t: the task, the step counter, the five most recent observations and actions, the current observation, and the admissible actions (ALFWorld) or the objects that can be referred to (ScienceWorld). It is not the simulator’s internal state, which no model sees. At a state $s _ { t }$ reached during training, the policy samples K responses $y _ { t } ^ { ( 1 ) } , \ldots , y _ { t } ^ { ( K ) }$ at temperature 1, and the actions $a _ { t } ^ { ( 1 ) } , \ldots , a _ { t } ^ { ( K ) }$ are parsed from them. The first action is executed, $a _ { t } = a _ { t } ^ { ( 1 ) }$ with response $y _ { t } = y _ { t } ^ { ( 1 ) }$ , so collecting the candidates leaves the behavior policy unchanged. The success probability after an action and the value of a state (Equation (2)) are related by

$$
Q ^ { \pi } ( \mathfrak { s } _ { t } , a ) = \mathbb { E } \bigl [ V ^ { \pi } ( \mathfrak { s } _ { t + 1 } ) \ \big | \ \mathfrak { s } _ { t } , a _ { t } = a \bigr ] , \qquad V ^ { \pi } ( \mathfrak { s } _ { t } ) = \mathbb { E } _ { a \sim \pi ( \cdot | \mathfrak { s } _ { t } ) } \bigl [ Q ^ { \pi } ( \mathfrak { s } _ { t } , a ) \bigr ] ,\tag{5}
$$

where $V ^ { \pi } ( s _ { t + 1 } ) = R$ if the step ends the episode. When at least two candidates differ, the world model predicts the next observation of each distinct candidate a (Figure 2a), a successor state is assembled from the prediction, and the candidate is valued:

$$
\begin{array} { r l } & { \qquad \hat { o } _ { t + 1 } ( a ) = \operatorname* { g r e e d y } p _ { \phi } \big ( \cdot | \ x _ { t } , a \big ) , \qquad \hat { s } _ { t + 1 } ( a ) = C \big ( s _ { t } , a , \hat { o } _ { t + 1 } ( a ) \big ) , } \\ & { \qquad \mathrm { i f } \ \hat { o } _ { t + 1 } ( a ) \ \mathrm { r e p o r t s ~ t h a t ~ t h e ~ t a s k ~ i s ~ c o m p l e t e d } } \\ & { \qquad \hat { q } _ { t } ( a ) = \left\{ \begin{array} { l l } { 0 } & { \mathrm { i f } \ \hat { o } _ { t + 1 } ( a ) \ \mathrm { r e p o r t s ~ t h a t ~ t h e ~ e p i s o d e ~ h a s ~ e n d e d ~ w i t h o u t ~ c o m p l e t i o n } } \\ { V _ { \psi } \big ( \hat { s } _ { t + 1 } ( a ) \big ) } & { \mathrm { o t h e r w i s e } . } \end{array} \right. } \end{array}\tag{6}
$$

We call ${ \hat { q } } _ { t } ( a )$ the transition value of a: the estimated value of the successor predicted for that action. It approximates $Q ^ { \pi } ( { \boldsymbol { s } } _ { t } , { \boldsymbol { a } } )$ in Equation (5) using one predicted next observation. The change in state value across a transition also subtracts the current-state value. A predicted ending gives its outcome directly, without a value estimate; otherwise the critic values the predicted successor. Only ScienceWorld’s observations report the end of an episode, so in ALFWorld every transition value is a critic value.

The world model’s context $x _ { t }$ uses the same interaction history as $s _ { t }$ with a different window: all previous observations and actions in ALFWorld, and the last eight actions without their observations in ScienceWorld. It also contains the task, the current observation and the admissible actions or

![](images/3da6e6fac700836df900eb845751c0bd02183253762860cca806195b900de124.jpg)

## b Real branches calibrate a shared weight

<sup>U</sup>n<sup>if</sup>orm<sup>l</sup>y samp<sup>l</sup>e a c<sup>h</sup>a<sup>i</sup>n s<sup>t</sup>ep, rep<sup>l</sup>ay <sup>it</sup>s pre<sup>fi</sup>x, an<sup>d</sup> <sup>t</sup>ry a <sup>diff</sup>eren<sup>t</sup> ac<sup>ti</sup>on.

![](images/a09296fb6dc1eb04e66845f8f653feedac49ee233bdbb5fb15392635a53d4045.jpg)

![](images/585be5cac57655af6bc24de5b08d8477c401c1daac804b6d1eaaf53fe1b03a13.jpg)  
Figure 2: Transition values for policy and value learning. a, Real state transitions train the world model; real returns are the main supervision for the critic. The policy samples K responses and executes the first. The world model predicts each candidate’s next observation. The same critic $V _ { \psi }$ values real states for PPO’s advantages $\hat { A } _ { t }$ and predicted successors for transition values ${ \hat { q } } _ { t } ( a )$ (Equation (6); 1 or 0 at predicted endings). Comparisons supply auxiliary advantages $A _ { t , k } ^ { \mathrm { a u x } }$ in PPOclip (policy channel). The mean transition value ${ \bar { q } } _ { t }$ is held fixed as a target for $V _ { \psi } ( s _ { t } )$ during the critic update (value channel). Both use the weight $w _ { \ell }$ of the state’s margin bin $\ell ( m _ { t } )$ . Solid arrows carry information, and dashed arrows update models. At evaluation, the policy acts alone. b, Replaying a prefix lets chain and branch actions be tested from the same state; their outcomes calibrate each margin bin’s weight for both channels (Figure 3). $\mathbf { c } ,$ The preferred action $\hat { a } _ { t }$ sets the indicators $z _ { t , k } ,$ and the margin $m _ { t }$ selects the bin whose weight scales both $A _ { t , k } ^ { \mathrm { a u x } }$ and the value-channel loss. Trees and values are schematic.

objects. The prediction is one greedy decode of the next observation as the environment would print it. In ScienceWorld, this observation includes the room, the inventory, the game’s progress score and, when the simulator ends the episode, a completion or failure marker.

The deterministic prompt constructor C appends the candidate and the predicted observation to the history, advances the step counter, and updates the window of recent steps. The list of admissible actions or objects is carried over from before the step; only the observation is predicted. A real next state instead contains a refreshed list. Valuing an action through the state it leads to is the afterstate evaluation used when the effect of a move is known, as in backgammon (Tesauro, 1995; Sutton & Barto, 2018); here the effect is predicted. Appendix G gives complete model inputs and outputs for one evaluated state (Figure 19).

Equation (5) underlies both channels of WM-PPO. By the first relation, differences between the candidates’ transition values estimate differences between their action values; the policy channel uses them to rank the candidates (Section 3.2). By the second, the mean transition value estimates the value of the state and serves as a critic target (Section 3.3). For two candidates at the same state, the difference between their advantages in Equation (3) equals the difference between the success probabilities after them,

$$
A ^ { \pi } \big ( s _ { t } , a _ { t } ^ { ( k ) } \big ) - A ^ { \pi } \big ( s _ { t } , a _ { t } ^ { ( k ^ { \prime } ) } \big ) = Q ^ { \pi } \big ( s _ { t } , a _ { t } ^ { ( k ) } \big ) - Q ^ { \pi } \big ( s _ { t } , a _ { t } ^ { ( k ^ { \prime } ) } \big ) \approx \hat { q } _ { t } \big ( a _ { t } ^ { ( k ) } \big ) - \hat { q } _ { t } \big ( a _ { t } ^ { ( k ^ { \prime } ) } \big ) .\tag{7}
$$

The common term $V ^ { \pi } ( s _ { t } )$ cancels, so the comparison does not depend on how well the critic values the current state. The approximation uses a single predicted observation in place of an expectation, a learned critic in place of $V ^ { \pi }$ , and truncated prompts with the previous action list in place of full states. Section 3.5 checks the resulting comparisons against real outcomes. A preferred candidate need not increase the value in absolute terms; it only has to do better than the other candidates the policy proposed. The critic outputs an unconstrained scalar and is trained toward the probability of success; its estimates may fall outside [0, 1].

## 3.2 From value comparisons to policy updates

Among the distinct sampled actions, the preferred candidate is $\hat { a } _ { t } = \arg \operatorname* { m a x } _ { a } \hat { q } _ { t } ( a )$ , whose predicted successor has the highest estimated probability of success. The margin $m _ { t } = \hat { q } _ { t } ( \hat { a } _ { t } ) - \operatorname* { m a x } _ { a \neq \hat { a } _ { t } } \hat { q } _ { t } ( a )$ measures how clearly it is preferred; a tie at the highest transition value $( m _ { t } = 0 )$ gives no preference. Each of the K sampled responses receives an auxiliary advantage, which is the advantage $\ddot { A } _ { t , k }$ of the diagnostic in Section 2.3 (Equation (4)) scaled by a reliability weight:

$$
A _ { t , k } ^ { \mathrm { a u x } } = \nu w _ { \ell ( m _ { t } ) } \big ( z _ { t , k } - \bar { z } _ { t } \big ) , \qquad z _ { t , k } = { \bf 1 } \big [ a _ { t } ^ { ( k ) } = \hat { a } _ { t } \big ] , \qquad \bar { z } _ { t } = \frac { 1 } { K } \sum _ { k ^ { \prime } = 1 } ^ { K } z _ { t , k ^ { \prime } } ,\tag{8}
$$

where $\ell ( m _ { t } )$ is the margin bin into which $m _ { t }$ falls, $w _ { \ell } ~ \in ~ [ 0 , 1 ]$ is that bin’s reliability weight (Section 3.5), and ν sets the strength of the auxiliary term relative to the PPO term.

Duplicate actions share one prediction, but advantages are assigned to all K responses, duplicates included. The samples of the preferred candidate receive the positive advantage $\nu w _ { \ell ( m _ { t } ) } ( 1 - \bar { z } _ { t } )$ and the other samples the negative advantage $- \nu w _ { \ell ( m _ { t } ) }$ z¯<sub>t</sub>; the weight scales both. Because the advantages sum to zero within the state (Figure 2c), the update moves probability from the other sampled responses toward those of the preferred candidate. At the start of an update, when every probability ratio is 1 and no clipping applies, the auxiliary advantages give the gradient direction

$$
\sum _ { k = 1 } ^ { K } \left( z _ { t , k } - \bar { z } _ { t } \right) \nabla _ { \theta } \log \pi _ { \theta } \left( y _ { t } ^ { ( k ) } \mid s _ { t } \right) = \frac { 1 } { K } \sum _ { k : z _ { t , k } = 1 } \sum _ { k ^ { \prime } : z _ { t , k ^ { \prime } } = 0 } \nabla _ { \theta } \log \frac { \pi _ { \theta } \left( y _ { t } ^ { ( k ) } \mid s _ { t } \right) } { \pi _ { \theta } \left( y _ { t } ^ { ( k ^ { \prime } ) } \mid s _ { t } \right) } ,\tag{9}
$$

scaled by $\nu w _ { \ell ( m _ { t } ) }$ and the token normalization of Equation (10). Each term raises the log-probability of a response with the preferred action relative to another sampled response at the same state, a pairwise preference. The size of the margin enters only through the bin’s weight: the policy learns which candidate is preferred, not by how much. Equation (8) is the group-relative advantage of GRPO (Shao et al., 2024) without division by the standard deviation, computed among the samples of one state with a label of 1 for the preferred candidate. Free-text actions cannot be enumerated, and PPO learns a state-value critic. We therefore evaluate the K samples through their predicted successors and retain only which candidate is preferred (related methods in Section 6).

The policy is trained on two kinds of examples, each a (state, response, advantage) triple. Ordinary PPO examples $( s _ { t } , y _ { t } , \hat { A } _ { t } )$ come from the executed steps. We compute their advantages by tree GAE on real returns (Section 3.4), which gives raw estimates $A _ { t } ^ { \mathrm { G A E } }$ , and whiten these over the executed steps to obtain $\hat { A } _ { t }$ . Auxiliary examples $( s _ { t } , y _ { t } ^ { ( k ) } , A _ { t , k } ^ { \mathrm { a u x } } )$ come from the evaluated candidates; their advantages are given by Equation (8) and are not whitened. A minibatch B containing both kinds of example enters one clipped objective:

$$
\mathcal { L } _ { \mathrm { P P O } } ( \theta ) = - \frac { 1 } { N } \sum _ { ( s , y , A ) \in { \mathcal B } } \sum _ { i = 1 } ^ { | y | } \operatorname* { m i n } \Bigl ( \rho _ { i } A , ~ \mathrm { c l i p } ( \rho _ { i } , 1 - \epsilon _ { \mathrm { l o } } , 1 + \epsilon _ { \mathrm { h i } } ) A \Bigr ) ,\tag{10}
$$

where $\rho _ { i } = \pi _ { \theta } ( y _ { i } \mid s , y _ { < i } ) / \pi _ { \mathrm { o l d } } ( y _ { i } \mid s , y _ { < i } )$ is the probability ratio of the i-th response token, $\pi _ { \mathrm { o l d } }$ is the sampling policy, $\begin{array} { r } { N = \sum _ { ( s , y , A ) \in B } | y | } \end{array}$ is the number of response tokens in the minibatch, and $\epsilon _ { \mathrm { l o } } , \epsilon _ { \mathrm { h i } }$ are the clipping bounds (Yu et al., 2025). For a positive advantage, the clipping removes the incentive to raise a token’s probability ratio above $1 + \epsilon _ { \mathrm { h i } } ;$ for a negative advantage, it removes the incentive to lower the ratio below $1 - \epsilon _ { \mathrm { l o } }$ . An executed response can appear both as an ordinary example and as an auxiliary example. States whose bin has weight $0 ,$ and tied states, contribute no auxiliary examples. We use $K = 4 , \nu = 1$ and $( \epsilon _ { \mathrm { l o } } , \epsilon _ { \mathrm { h i } } ) = ( 0 . 2 , 0 . 2 8 )$ . The world model’s text predictions and the critic’s values are fixed before the update. Gradients reach the policy through its log-probabilities of the sampled responses.

## 3.3 From predicted transitions to value targets

The value channel uses predicted consequences to estimate how likely the current state is to lead to success. By the second part of Equation (5), the value of a state is the policy’s average action value. The K samples are draws from $\pi _ { \boldsymbol { \theta } } ( \cdot \mid s _ { t } )$ and ${ \hat { q } } _ { t } ( a )$ estimates $Q ^ { \pi } ( { \boldsymbol { s } } _ { t } , { \boldsymbol { a } } )$ , so their mean

$$
\bar { q } _ { t } = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \hat { q } _ { t } \big ( \boldsymbol { a } _ { t } ^ { ( k ) } \big )\tag{11}
$$

is a one-step model-based estimate of $V ^ { \pi } ( s _ { t } )$ : the critic’s own values one step ahead, averaged over the actions the policy proposes, with predicted endings valued at 1 or 0. This estimate supplies an auxiliary critic target, weighted by reliability,

$$
\mathcal { L } _ { \mathrm { c o n s } } ( \psi ) = \beta \sum _ { s _ { t } \in \mathcal { D } } w _ { \ell ( m _ { t } ) } \big ( V _ { \psi } ( s _ { t } ) - \mathrm { s g } [ \bar { q } _ { t } ] \big ) ^ { 2 } ,\tag{12}
$$

where $\mathcal { D }$ is the set of evaluated chain states with $m _ { t } > 0 .$ , sg stops the gradient, and $w _ { \ell ( m _ { t } ) }$ is the gate weight that also scales the state’s auxiliary advantages; we use $\beta = 0 . 5$ . Like the transition values, the target is computed with the critic before the update. The term is added to the critic’s regression on returns (Equation (13)), which remains the critic’s main supervision. The two targets average over different things. The return target $G ( s _ { t } )$ averages the final rewards of the few episodes that pass through $s _ { t }$ , all of which follow the executed actions, whereas ${ \bar { q } } _ { t }$ averages over the K actions the policy proposed, each valued at its predicted successor. Where the world model predicts an ending, the target includes the predicted outcome directly, also for candidates that were not executed.

Both channels reuse the same predictions, critic and gate. Write the transition value of each sample as $\hat { q } _ { t } ( a _ { t } ^ { ( k ) } ) = \bar { q } _ { t } + \left( \hat { q } _ { t } ( a _ { t } ^ { ( k ) } ) - \bar { q } _ { t } \right)$ . The value channel trains the critic on the mean, an estimate of how good the state is under the current policy. The policy channel uses the ordering of the deviations to form the gated preferences in Equation (8); both these deviations and the auxiliary advantages sum to zero over the K samples. The value channel also affects the policy through the critic. PPO’s advantages are built from critic-value differences along the executed path (Section 3.4), so a critic that values states more accurately can give $\mathrm { P P O ^ { \circ } s }$ own update better advantages. This effect starts from the next iteration, because each iteration’s advantages are computed before its critic update. We call the method with both channels WM-PPO, and the variant without the term of Equation (12) WM-PPO (policy channel only).

## 3.4 Learning through real interaction

Each iteration uses one batch of branched real rollouts. Actions and returns train the policy and critic, observed transitions train the world model, and paired outcomes calibrate the gate (Figure 2; Table 4 in Appendix B).

Episodes and branches. Each iteration draws 64 training tasks. For each task, the policy plays one main episode, the chain, in which every step also records the K candidates. The policy then plays three branches from the chain. For each branch, we draw a step of the chain uniformly at random. Because the environments are deterministic, replaying the chain’s actions up to that step restores its state. We resample the policy’s first response at this state up to eight times, stopping when its action differs from the chain’s, and then continue the branch to the end. If all eight samples repeat the chain’s action, the branch continues with that action but yields no comparison. The policy proposed a different action within eight samples in 62–67% of the branches on ScienceWorld and in 40–61% on ALFWorld. Each iteration therefore plays 256 episodes. Branching from intermediate states to obtain real outcomes for different actions is common in tree-based RL for language models (Ji et al., 2026; Hou et al., 2025). Here the branches add real interaction to the PPO update and test the candidate comparisons against real outcomes (Figure 2b; Section 3.5). The resampled action enters PPO as an ordinary sample with its unconditioned log-probabilities (Section 7).

Critic. The episodes of a task form a tree whose nodes are states and whose edges are executed steps; an edge shared by several episodes, such as the chain’s prefix before a branch point, appears once. Let $s$ be the set of nodes in the iteration’s trees and $G ( s )$ the mean final reward of the episodes that pass through node s. The critic minimizes

$$
\mathcal { L } _ { V } ( \psi ) = \sum _ { s \in \mathcal { S } } \bigl ( V _ { \psi } ( s ) - G ( s ) \bigr ) ^ { 2 } + \mathcal { L } _ { \mathrm { c o n s } } ( \psi ) ,\tag{13}
$$

that is, a regression on returns plus the value channel of Equation (12), which the policy-channel-only variant omits (Appendix B.2 gives implementation details of the regression). PPO’s advantages are computed by GAE (Schulman et al., 2016) on the tree, with $\gamma = 1$ and parameter λ. An edge e from $s _ { t }$ receives

$$
\begin{array} { r } { A ^ { \mathrm { G A E } } ( e ) = \left\{ \begin{array} { l l } { R - V _ { \psi } ( s _ { t } ) } & { \mathrm { i f ~ } e \mathrm { ~ e n d s ~ t h e ~ e p i s o d e , } } \\ { V _ { \psi } ( s _ { t + 1 } ) - V _ { \psi } ( s _ { t } ) + \lambda \overline { { A } } ^ { \mathrm { G A E } } ( s _ { t + 1 } ) } & { \mathrm { i f ~ } e \mathrm { ~ l e a d s ~ t o ~ } s _ { t + 1 } , } \end{array} \right. } \end{array}\tag{14}
$$

where $\begin{array} { r } { \overline { { A } } ^ { \mathrm { G A E } } ( s ) = | \mathcal { E } ( s ) | ^ { - 1 } \sum _ { e ^ { \prime } \in \mathcal { E } ( s ) } A ^ { \mathrm { G A E } } ( e ^ { \prime } ) } \end{array}$ is the mean raw advantage of the edges ${ \mathcal { E } } ( s )$ leaving s. On a path without branches, this is the usual GAE recursion. Its term $V _ { \psi } ( s _ { t + 1 } ) - V _ { \psi } ( s _ { t } )$ is the critic’s estimate of the change in state value across the executed transition (Equation (3)). The executed step t of an episode receives the raw advantage $A _ { t } ^ { \mathrm { G A E } }$ of its edge. An episode cut at the step limit ends with $R = 0$ , without bootstrapping. We use $\lambda = 0 . 9 5$

a Test two actions at one state  
![](images/2dd99041e27754cfbbd7cda76dd089a04a9aa9c7bd38a4e9eb2c7ab08370e612.jpg)

<sup>b L</sup>oca<sup>t</sup>e <sup>th</sup>e compar<sup>i</sup>son on <sup>th</sup>e marg<sup>i</sup>n sca<sup>l</sup>e  
![](images/21ba760dda55b1d0b06ba30387544246189b9cb010355d65dfaa5d0671fdc46f.jpg)

c <sup>E</sup>ar<sup>li</sup>er pa<sup>i</sup>rs se<sup>t</sup> <sup>th</sup>e we<sup>i</sup>g<sup>ht</sup> <sup>f</sup>or <sup>th</sup>e curren<sup>t</sup> up<sup>d</sup>a<sup>t</sup>e  
![](images/3a23a777540aea7957c273abc256667d3f0f1765ce19f36f4a19928fef55c751.jpg)  
Figure 3: Calibrating transition-value comparisons with real outcomes. $\mathbf { a } , \mathbf { A }$ chain action and a different branch action are played from the same state; the pair’s gain is $\Delta G$ sign $( \Delta \hat { q } )$ (Equation (16)). b, The nonzero margins $m _ { t }$ (best minus runner-up transition value) at the current iteration’s evaluated states define two quantile cutoffs and thus three bins. Each new pair is assigned to a bin by $| \Delta \hat { q } |$ c, Pairs stored in the preceding M iterations, with their recorded bins, determine $g _ { \ell }$ and $w _ { \ell }$ for the current iteration. Each current state’s margin selects one weight for both its auxiliary advantages and its value-channel loss. New pairs are kept for later iterations. Dots and trajectories are schematic.

World model. The world model is trained on the state transitions that the policy executes in training tasks, with the cross-entropy of the next observation given the world model’s context and the action,

$$
\mathcal { L } _ { \mathrm { W M } } ( \phi ) = - \frac { 1 } { N _ { o } } \sum _ { \substack { ( x _ { t } , a _ { t } , o _ { t + 1 } ) \in \mathcal { B } _ { \mathrm { W M } } } } \sum _ { i = 1 } ^ { | o _ { t + 1 } | } \log p _ { \phi } \big ( o _ { t + 1 , i } | x _ { t } , a _ { t } , o _ { t + 1 , < i } \big ) ,\tag{15}
$$

where $B _ { \mathrm { W M } }$ is a minibatch of executed transitions, $o _ { t + 1 , i }$ the i-th token of the observation the environment printed, and $N _ { o }$ the number of observation tokens in the minibatch. The initial world model is trained at the end of iteration 12 on the executed transitions of the common PPO warm-up (Section 4.1). Afterward, each iteration’s executed transitions from chains and branches update it for one epoch, in parallel with the policy update; replayed prefix steps are not added again. Every recorded observation supervises what an action changed, whether or not the episode succeeded. This training reuses the interaction collected for RL and follows the states and actions of the current policy.

Algorithm 1 One iteration of WM-PPO (simplified; Algorithm 2 in Appendix B.1 gives the full   
procedure). Predictions and transition values are computed without gradients.   
Require: policy $\pi _ { \boldsymbol { \theta } } ,$ critic $V _ { \psi } ,$ world model $p _ { \phi } ;$ branch pairs $\mathcal { P }$ of the last M iterations   
1: play 64 chains with $\pi _ { \theta }$ (at every state $s _ { t } ,$ sample $K$ responses and execute the first) and 3 real branches from   
each chain   
2: for each chain state $s _ { t }$ with at least two distinct candidate actions do   
3: value each distinct a at its successor predicted by $p _ { \phi } \colon \hat { q } _ { t } ( a )  V _ { \psi } \big ( \hat { s } _ { t + 1 } ( a ) \big )$ , or $1 / 0$ at a predicted   
ending Eq. (6)   
4: compute the preferred candidate $\hat { a } _ { t } ,$ margin $m _ { t }$ and mean transition value $\bar { q } _ { t }$ Eq. (11)   
5: end for   
6: compute bin weights $w _ { \ell }$ from the real gains $\Delta G$ sign $\boldsymbol { \mathsf { 1 } } ( \Delta \hat { q } )$ of the pairs in $\mathcal { P }$ Eqs. (16), (17)   
7: policy: one PPO-clip objective, with $\hat { A } _ { t }$ (whitened tree GAE on real returns) on executed responses and   
$A _ { t , k } ^ { \mathrm { a u x } } = \nu w _ { \ell ( m _ { t } ) } ( z _ { t , k } - \bar { z } _ { t } )$ on all $K$ samples Eqs. (8), (10)   
8: critic: regression on real returns plus $\beta w _ { \ell ( m _ { t } ) } \big ( V _ { \psi } ( s _ { t } ) - \mathrm { s g } [ \bar { q } _ { t } ] \big ) ^ { 2 }$ Eqs. (12), (13)   
9: world model: next-observation loss on this iteration’s executed transitions Eq. (15)   
10: store the new branch pairs with their pre-update values in $\mathcal { P } ;$ use the updated models next iteration

## 3.5 Reliability calibration and the complete training loop

Section 2 shows that value margins carry information about comparison reliability. Both the policydependent values $V ^ { \pi }$ and the models estimating them change during training, so the gate recalibrates the margins in every iteration using recent real outcomes. The first step of every branch that obtained a different action provides a test (Figure 3a). At the same state $s _ { t } ,$ the chain took action $a _ { t } ^ { \mathrm { c h } }$ and the branch took $a _ { t } ^ { \mathrm { b r } } \neq a _ { t } ^ { \mathrm { c h } }$ , and both continuations were played to the end, with returns $G ^ { \mathrm { c h } }$ and $G ^ { \mathrm { b r } }$ Before the iteration’s update, the world model and the critic value both actions as in Equation (6). We write $\Delta \hat { q } = \hat { q } _ { t } ( a _ { t } ^ { \mathrm { c h } } ) - \hat { q } _ { t } ( a _ { t } ^ { \mathrm { b r } } )$ and $\Delta G = G ^ { \mathrm { c h } } - G ^ { \mathrm { b r } }$ . The pair’s margin $| \Delta \hat { q } |$ assigns it to one of three bins $\ell = 1 , 2 , 3$ , ordered from the smallest to the largest margins, whose boundaries are the 0.5 and 0.8 quantiles of that iteration’s nonzero candidate margins $m _ { t }$ (Figure 3b).

For bin $\ell ,$ the realized gain of following the preference is

$$
g _ { \ell } = \frac { 1 } { n _ { \ell } } \sum _ { \mathrm { p a i r s i n } \ : \mathrm { b i n } \ : \ell } \Delta G \ : \mathrm { s i g n } ( \Delta \hat { q } ) ,\tag{16}
$$

computed over the $n _ { \ell }$ pairs that the previous M iterations stored in the bin, each with the value difference and bin recorded in its own iteration. The gain is positive when the preferred actions led to better real outcomes. Because final rewards lie in {0, 1}, $\Delta G \in \{ - 1 , 0 , 1 \}$ , and a pair contributes only when its two outcomes differ. A bin thus earns weight when its comparisons often decide the task and its preferences are then usually right (Section 5.1).

The gate shrinks and rescales this gain to a weight capped at 1 (Figure 3c),

$$
w _ { \ell } = \left\{ \begin{array} { l l } { \mathrm { c l i p } \left( \displaystyle \frac { n _ { \ell } } { n _ { \ell } + n _ { 0 } } \cdot \frac { g _ { \ell } } { g _ { \mathrm { r e f } } } , 0 , 1 \right) , } & { n _ { \ell } \geq n _ { \mathrm { m i n } } , } \\ { 0 , } & { n _ { \ell } < n _ { \mathrm { m i n } } . } \end{array} \right.\tag{17}
$$

Here $n _ { 0 }$ sets how strongly a bin with few pairs is shrunk toward zero, $g _ { \mathrm { r e f } }$ is the shrunken gain at which a bin receives full weight, and $n _ { \mathrm { m i n } }$ is the number of pairs below which a bin stays closed. A bin whose comparisons do not pay off in real outcomes thus contributes nothing. We use $M = 4 ,$ $n _ { 0 } = 4 0 , g _ { \mathrm { r e f } } = 0 . 0 5$ and $n _ { \mathrm { m i n } } = 2 0$ ; these values were set once from the first branch pairs of a preliminary ALFWorld run and kept unchanged in both environments. Pairs valued by the current pre-update models are stored for later iterations. The current weights depend on earlier branch outcomes.

All three models are separate LoRA adapters on one frozen base model and share no parameters beyond it. Only the policy is used at evaluation.

Table 1: Comparison protocol. Columns show feedback beyond PPO’s executed-step advantage, episodes per iteration (64 × (1 + 3): 64 tasks, each with one chain and three branches), and trainingtask coverage. Coverage is the share of distinct tasks seen in iterations 13–36 on ALFWorld (of 3,553 tasks) and 13–60 on ScienceWorld (of 1,228 variations). With 40 tasks × 4, only GRPO was run on ALFWorld. All methods start from the same warm-up checkpoint with the same task order.
<table><tr><td colspan="3"></td><td colspan="2">Tasks seen</td></tr><tr><td>Method</td><td>Episodes</td><td>Feedback beyond PPO&#x27;s advantage</td><td></td><td>ALFWorld ScienceWorld</td></tr><tr><td>WM-PPO</td><td>64× (1 + 3)</td><td>candidates valued by the critic at predicted successors; comparisons as auxiliary advantages, mean as critic target;</td><td>43%</td><td>100%</td></tr><tr><td>PPO-160, PPO-256</td><td>160 / 256</td><td>both weighted by real branches</td><td>100%</td><td>100%</td></tr><tr><td>GRPO, GiGPO</td><td>20 × 8</td><td>group-relative advantages</td><td>14%</td><td>51%</td></tr><tr><td>GRPO, GiGPO</td><td>40 × 4</td><td>instead of a critic as above, more tasks, fewer</td><td>27%</td><td>91%</td></tr><tr><td>CAPO, SO-PPO, Turn-ratio PPO 160</td><td></td><td>episodes each action- or turn-level ratios and</td><td>100%</td><td>100%</td></tr><tr><td>WM-PPO (policy channel only)</td><td>64×(1 + 3)</td><td>clipping as WM-PPO, without the critic</td><td>43%</td><td>100%</td></tr><tr><td>Q-PPO</td><td></td><td>target 64 × (1 + 3) as policy channel only;</td><td>43%</td><td>100%</td></tr><tr><td>Wide-Tree-PPO</td><td></td><td>candidates valued by a learned Qω(st, y)</td><td></td><td></td></tr><tr><td>Fixed WM</td><td></td><td>64 × (1 + 3) none; same trees and critic as policy channel only</td><td>43%</td><td>100%</td></tr><tr><td></td><td>64 × (1 + 3)</td><td>as policy channel only; world model never updated</td><td>43%</td><td>100%</td></tr><tr><td>PaW</td><td>160</td><td>auxiliary next-observation loss on the policy</td><td>100%</td><td>100%</td></tr><tr><td>Majority-vote labels</td><td>160</td><td>plurality label among 4 samples per step</td><td>100%</td><td>100%</td></tr><tr><td>OPD (32B)</td><td>160</td><td>teacher&#x27;s token log-probabilities in the student&#x27;s context</td><td>100%</td><td>100%</td></tr><tr><td>OCSD (self, 32B)</td><td>160</td><td>advantage scaled by the teacher&#x27;s response to the</td><td>100%</td><td>100%</td></tr></table>

## 4 Experimental Evaluation

We test whether learning to value predicted consequences raises policy-only success for a comparable amount of real interaction. All methods share a 12-iteration PPO warm-up checkpoint and task order, and are evaluated as policies alone at fixed iterations. WM-PPO and PPO-160 use approximately the same number of real steps. WM-PPO, its policy-channel-only variant, PPO-160, PPO-256, Q-PPO and Wide-Tree-PPO have three training runs each, as does self-teacher OCSD on ALFWorld. WM-PPO’s hyperparameters are set once on ALFWorld; baselines are re-tuned by rules fixed in advance when their settings fail (Section 4.1).

## 4.1 Experimental Setup

Environments. ALFWorld (Shridhar et al., 2021) is the simpler environment. Its six household task types mostly follow the same routine: find an object, heat, cool or clean it when required, and place it at the requested location. The environment lists admissible actions at every step, leaving search for hidden objects as the main difficulty. We train on 3,553 games and pool evaluation over 140 seenroom and 134 unseen-room games. ScienceWorld (Wang et al., 2022) poses harder elementary science tasks. The 18 task types, fixed before training, all have oracle solutions within 40 steps. Eleven are procedural: the agent must first create or find an intermediate state, such as a mixture, a heated substance, a connected circuit or an organism in another room. The other seven require choosing the right object among candidates. The agent must identify the relevant conditions in long room descriptions and express its actions as free text using 25 templates. Unparseable commands consume a step, and focusing on the wrong object ends the task in failure. These dependencies make the value of a state transition less apparent from the next observation alone: the agent must judge whether a change establishes a condition needed for later success. Appendix B describes ScienceWorld’s fixed training and evaluation sets. In both environments, the reward is 1 for completion within 40 actions and 0 otherwise, with no intermediate rewards. ScienceWorld’s reported progress is part of the observation and is also predicted by the world model.

Model and common start. All methods train LoRA adapters (rank 64) on a frozen Qwen3-4B-Instruct-2507 (Yang et al., 2025) in non-thinking mode, with responses of at most 256 tokens. The shared PPO warm-up reaches 27.0% success on ALFWorld and 20.8% on ScienceWorld. Its real transitions also train the initial world model. Training then runs for 24 iterations on ALFWorld (13–36) and 48 on ScienceWorld (13–60).

Comparisons. Table 1 lists the comparisons in three groups. The first group contains established ways of training the same policy: PPO at the two budgets, GRPO and GiGPO (Shao et al., 2024; Feng et al., 2025), and three refinements of PPO for multi-turn agents, CAPO (Wang et al., 2026), SO-PPO (Li et al., 2026) and Turn-ratio PPO, a turn-level form of GSPO (Zheng et al., 2025).

The second group contains structural controls. WM-PPO (policy channel only) removes the critic target of Section 3.3; each other control changes one element of that variant. Wide-Tree-PPO keeps the trees and critic training but removes the auxiliary update. Q-PPO values candidates directly with $Q _ { \omega } ( s _ { t } , y )$ , learned from real returns. Fixed WM keeps the initial world model throughout training (Appendix D).

The third group adds other auxiliary supervision to PPO-160. PaW (Lu et al., 2026) uses the published auxiliary next-observation loss to train the policy on dynamics, without evaluating predicted consequences for task success. Majority-vote labels use ungated self-training on the most common of four sampled actions (Zuo et al., 2025; Shafayat et al., 2025). Standard on-policy distillation (OPD) (Agarwal et al., 2024) uses a frozen Qwen3-32B teacher (Appendix D.2). OCSD (Yang et al., 2026) uses the policy itself or the 32B model as a teacher with access to the observations that followed the action.

Hyperparameters across environments. WM-PPO uses its ALFWorld hyperparameters unchanged on ScienceWorld, adapting the prompts and the world model’s context and output lengths (Appendix B). Baselines also start from their ALFWorld settings and follow pre-set re-tuning rules if those settings fail. CAPO, SO-PPO and PaW required re-tuning (Appendix D).

Interaction budgets. The main comparison approximately matches the number of real steps. A real step is an environment step whose action the policy generated and executed, in chains and branches alike, counted from iteration 13; the shared warm-up and the prefix replays are not counted. Because branch points are drawn uniformly along the chain, the 64 chains and 192 branches of a WM-PPO iteration generate about as many real steps as $6 4 \times ( 1 + 3 \times { \textstyle { \frac { 1 } { 2 } } } ) = 1 6 0$ episodes, the budget of

Table 2: Policy-only success (%) of the main comparisons. Bold marks the highest mean in each column. Results are means over training runs; ± gives the standard deviation for methods with at least two runs. Runs: ALFWorld / ScienceWorld. Real steps from iteration 13 (ALFWorld / ScienceWorld): WM-PPO 102k / 128k, WM-PPO (policy channel only) 97k / 130k, PPO-160 105k / 113k, PPO-256 150k / 156k; Figure 4 and Appendix E give all methods. <sup>†</sup>: WM-PPO is significantly higher $( p < 0 . 0 5$ , per-game paired test over the evaluated checkpoints, uncorrected). <sup>a</sup>ScienceWorld: the best of three settings, clipping ranges for SO-PPO and loss coefficients for PaW (Appendix D). <sup>b</sup>ScienceWorld: all three clipping settings of CAPO diverged; the entry uses clip 0.1 at iteration 18, its highest evaluation before divergence, without a test.
<table><tr><td colspan="2"></td><td colspan="2">ALFWorld, iteration 36</td><td colspan="2">ScienceWorld, iteration 60</td></tr><tr><td>Method</td><td>Runs</td><td>Greedy</td><td>pass@1</td><td>Greedy</td><td>pass@1</td></tr><tr><td>WM-PPO (ours)</td><td>3/3</td><td> $8 7 . 5 \pm 0 . 9$ </td><td> $8 7 . 8 \pm 1 . 7$ </td><td> $\pm 2 . 9 \pm 2 . 3$ </td><td> ${ \bf 5 1 . 6 \pm 4 . 8 }$ </td></tr><tr><td>WM-PPO (policy channel only)</td><td>3/3</td><td> ${ \bf 8 8 . 4 \pm 2 . 3 }$ </td><td> ${ \bf 8 9 . 4 \pm 1 . 6 }$ </td><td> $4 7 . 2 ^ { \dagger } \pm 2 . 8$ </td><td> $4 7 . 2 ^ { \dagger } \pm 1 . 7$ </td></tr><tr><td colspan="6">Standard training and refinements</td></tr><tr><td>PPO-160</td><td>3/3</td><td> $6 3 . 5 ^ { \dagger } \pm 9 . 8$ </td><td> $7 1 . 0 ^ { \dagger } \pm 6 . 0$ </td><td> $3 9 . 3 ^ { \dagger } \pm 1 . 5$ </td><td> $4 2 . 6 ^ { \dagger } \pm 2 . 5$ </td></tr><tr><td>PPO-256</td><td>3/3</td><td> $8 3 . 1 ^ { \dagger } \pm 3 . 1$ </td><td> $8 4 . 8 ^ { \dagger } \pm 1 . 2$ </td><td> $4 3 . 8 ^ { \dagger } \pm 8 . 7$ </td><td> $4 3 . 5 ^ { \dagger } \pm 6 . 6$ </td></tr><tr><td>GRPO</td><td>1/1</td><td> $5 9 . 5 ^ { \dagger }$ </td><td> $6 7 . 8 ^ { \dagger }$ </td><td> $3 5 . 2 ^ { \dagger }$ </td><td>35.7†</td></tr><tr><td>GiGPO</td><td>1/1</td><td> $7 8 . 1 ^ { \dag }$ </td><td> $8 1 . 0 ^ { \dagger }$ </td><td> $3 8 . 4 ^ { \dagger }$ </td><td>36.3†</td></tr><tr><td>CAPO</td><td>1/1</td><td>87.2</td><td>86.7</td><td> $2 1 . 1 ^ { b }$ </td><td> $2 0 . 0 ^ { b }$ </td></tr><tr><td>SO-PPOª</td><td>1/1</td><td>86.5</td><td> $8 3 . 1 ^ { \dagger }$ </td><td> $4 6 . 5 ^ { \dagger }$ </td><td>49.4</td></tr><tr><td>Turn-ratio PPO</td><td>1/1</td><td>85.8</td><td>85.0</td><td> $2 5 . 0 ^ { \dagger }$ </td><td> $2 6 . 5 ^ { \dagger }$ </td></tr><tr><td colspan="6">Structural controls (Section 5)</td></tr><tr><td>Q-PPO (learned Q values)</td><td>3/3</td><td> $8 4 . 4 ^ { \dagger } \pm 9 . 0 $ </td><td> $8 8 . 6 \pm 3 . 3$ </td><td> $4 6 . 1 ^ { \dagger } \pm 5 . 2$ </td><td> $4 6 . 2 ^ { \dagger } \pm 2 . 5$ </td></tr><tr><td>Wide-Tree-PPO (no aux. update)</td><td>3/3</td><td> $7 8 . 5 ^ { \dagger } \pm 9 . 4$ </td><td> $8 2 . 3 ^ { \dagger } \pm 5 . 2$ </td><td> $4 0 . 8 ^ { \dagger } \pm 3 . 9$ </td><td> $4 0 . 0 ^ { \dagger } \pm 3 . 3$ </td></tr><tr><td>Fixed WM</td><td>1/1</td><td> $7 6 . 3 ^ { \dagger }$ </td><td> $7 9 . 4 ^ { \dagger }$ </td><td> $3 8 . 7 ^ { \dagger }$ </td><td>39.0†</td></tr><tr><td colspan="6">Auxiliary feedback added to PPO-160</td></tr><tr><td> $\mathrm { P a W } ^ { a }$ </td><td>1/1</td><td>85.4</td><td>87.3</td><td> $3 5 . 6 ^ { \dagger }$ </td><td>38.8†</td></tr><tr><td>Majority-vote labels</td><td>1/1</td><td> $1 7 . 2 ^ { \dagger }$ </td><td> $1 7 . 6 ^ { \dagger }$ </td><td> $2 5 . 0 ^ { \dagger }$ </td><td>25.0†</td></tr><tr><td>OCSD (self teacher)</td><td>3/1</td><td> $8 3 . 0 ^ { \dagger } \pm 1 . 8$ </td><td> $8 2 . 7 ^ { \dagger } \pm 3 . 1$ </td><td> $3 4 . 2 ^ { \dagger }$ </td><td> $4 3 . 7 ^ { \dagger }$ </td></tr><tr><td>OCSD (32B teacher)</td><td>1/1</td><td>83.2</td><td> $7 0 . 1 ^ { \dagger }$ </td><td>43.0†</td><td> $4 1 . 5 ^ { \dagger }$ </td></tr><tr><td>OPD (32B teacher)</td><td>1/1</td><td>41.2†</td><td> $5 1 . 0 ^ { \dagger }$ </td><td> $3 6 . 3 ^ { \dagger }$ </td><td>33.9†</td></tr></table>

PPO-160 (Table 1). In practice, WM-PPO executed 3% fewer real steps than PPO-160 on ALFWorld and 13% more on ScienceWorld, where its 128k steps lie between those of PPO-160 (113k) and PPO-256 (156k). We report gains relative to PPO-160. Restoring a branch point by replaying the chain’s prefix brings the total environment calls to about those of the $6 4 \times 4 = 2 5 6$ episodes of PPO-256, which we include as a supplementary reference. Within this budget, WM-PPO covers 43% of the ALFWorld training tasks, against 100% for PPO.

Evaluation protocol. Checkpoints are evaluated with the policy alone. Greedy success is the success rate of one greedy episode per evaluation game; pass@1 is the success probability of a single temperature-1 episode, estimated from three episodes per game (pass@3 in Appendix E). Unless stated otherwise, success refers to greedy success. The main comparison reports iteration 36 on ALFWorld and iteration 60 on ScienceWorld, with the checkpoints and hyperparameters of WM-PPO and PPO fixed before evaluation. On ScienceWorld, SO-PPO and PaW each use the best of three settings in Table 2. All three CAPO settings diverged, so its entry uses the checkpoint with the highest greedy success during training, without a significance test (Appendix D). Each seed is represented by its first training run. Differences are tested per evaluation game with two-sided z-tests on the mean paired difference, uncorrected for multiple comparisons; Appendix E.3 reports each run.

## 4.2 Policy-only Task Performance

Higher success at comparable real interaction. On ALFWorld at iteration 36, WM-PPO succeeds on 87.5% of the evaluation games under greedy decoding and on 87.8% under pass@1 evaluation (Table 2); WM-PPO (policy channel only) reaches 88.4% and 89.4%. PPO-160, whose budget of real steps is matched by design, reaches 63.5% and 71.0%, and every run of either variant is above every PPO-160 run. The supplementary PPO-256, with about 50% more real steps, reaches 83.1% and 84.8%, also significantly lower in the per-game comparison.

On ScienceWorld at iteration 60, using the hyperparameters from ALFWorld, WM-PPO reaches 52.9% greedy and 51.6% pass@1 success, against 39.3% and 42.6% for PPO-160 $( p < 0 . 0 0 1 )$ Every WM-PPO run (50.4–54.9% greedy) is above every PPO-160 run (37.7–40.5%). PPO-256, with 22% more real steps, is 9.2 points lower under greedy decoding and 8.0 points lower for pass@1 $( p < 0 . 0 0 1 )$ . GRPO and GiGPO remain well below WM-PPO in both environments. They stay below when drawing 40 tasks of 4 episodes each, which covers 91% of the ScienceWorld variations (GiGPO on ScienceWorld only; Appendix E).

ScienceWorld separates the strongest methods. On the simpler ALFWorld, CAPO, SO-PPO, Turn-ratio PPO, PaW and the two WM-PPO variants cluster at 85.4–88.4% greedy success, approaching the environment’s ceiling. This leaves limited room for success rates to distinguish the methods. The two WM-PPO variants have the highest greedy success, and adding the value channel gives no further gain at this level (87.5% against 88.4%, not significant). Under pass@1, WM-PPO (policy channel only) is ahead of CAPO by 2.7 points, SO-PPO by $6 . 3 \ : ( p < 0 . 0 0 1 )$ and Turn-ratio PPO by 4.4 $( p = 0 . 0 0 2 )$ , and the full method is 4.7 points ahead of SO-PPO $( p = 0 . 0 1 )$ .

ScienceWorld, with its greater demands on intermediate conditions, gives a clearer separation. WM-PPO reaches 52.9% greedy success against 46.5% for the best of SO-PPO’s three settings, the strongest baseline: a gain of 6.5 points $( p = 0 . 0 0 4 )$ . The lead under pass@1 is 2.2 points. Here, the value channel adds 5.8 greedy and 4.4 pass@1 points to the policy channel alone $( p < 0 . 0 0 1$ Section 5.2). The PPO refinements change how executed actions are credited; WM-PPO add comparisons with unexecuted actions and a critic target built from them.

PaW reaches 85.4% success on ALFWorld, 2.1 points below WM-PPO. On ScienceWorld, the single run with its best of three loss weights reaches 35.6%, broadly at PPO-160’s level and slightly below its 39.3%. WM-PPO is 17.4 points higher $( p < 0 . 0 0 1 )$ . WM-PPO uses the predicted observations to evaluate which consequences matter for task success.

Higher success than the tested teacher-based methods. OPD and OCSD use teachers to supply feedback finer than a final 0/1 reward. Self-teacher OCSD improves PPO-160 to 83.0% greedy and 82.7% pass@1 success on ALFWorld over three runs; WM-PPO remains 4.5 and 5.1 points higher $( p = 0 . 0 0 3$ and $p < 0 . 0 0 1 )$ . On ScienceWorld, self-teacher OCSD is 5.2 points below PPO-160 under greedy decoding $( p = 0 . 0 2 )$ , level with it for pass@1, and 18.8 and 7.9 points below WM-PPO. The teacher’s judgment separates valid from invalid actions on ALFWorld’s short observations (AUC 0.80), but is inverted on ScienceWorld’s long descriptions (AUC 0.25 and 0.21; Appendix D.2). With the 32B teacher, OCSD reaches 70.1% pass@1 on ALFWorld and is level with PPO-160 on ScienceWorld. Standard OPD ends below PPO-160 in both environments (Appendix D.2). Ungated majority-vote labels reinforce the policy’s repeated choices without information about consequences; success falls to 17.2% on ALFWorld and 25.0% on ScienceWorld. WM-PPO outperforms these methods using a world model and critic trained on the agent’s own interaction.

Performance across runs. The runs of WM-PPO vary little (86.5–88.3% success on ALFWorld, 50.4–54.9% on ScienceWorld), whereas several baselines vary much more, for example PPO-160 over 52.2–70.1% on ALFWorld and PPO-256 over 33.8–49.3% on ScienceWorld (Appendix E.3).

![](images/a6cd1d7d9c30f49ffd85e31b85458a191dfd94cb9ffb9e81189ca5efcffc9518.jpg)  
Figure 4: Success against real interaction, for every method. Each gray or red point shows one method at the end of the common protocol (ALFWorld iteration 36, ScienceWorld iteration 60): mean greedy success in the official evaluation against mean real steps from iteration 13, averaged over training runs. Green: the evaluated checkpoints of WM-PPO (IT: iteration); dashed: WM-PPO (policy channel only). The shaded region lies below and to the right of at least one evaluated checkpoint (of either variant on ALFWorld, of WM-PPO on ScienceWorld). A method inside it used more real steps and achieved lower success; darker shading marks regions dominated by more checkpoints. Red squares: methods outside the region. Bracketed numbers identify the other methods, named in Figure 12 in Appendix E. Same-seed reruns and the diverged ScienceWorld CAPO runs are omitted.

## 4.3 Interaction Efficiency

No other method achieves higher success with fewer real steps. Figure 4 compares each method’s success and real interaction. On ALFWorld, 18 of the 23 other methods lie in the region below and to the right of the evaluated checkpoints of the two WM-PPO variants; on ScienceWorld, 18 of 20 lie below and to the right of WM-PPO’s checkpoints. The methods outside the region used fewer real steps but reached lower success: on ALFWorld, CAPO (with two clipping ranges), Turn-ratio PPO, OCSD and PaW; on ScienceWorld, the tuned SO-PPO and OCSD with the 32B teacher. PPO-256 uses 150k real steps against WM-PPO’s 102k on ALFWorld and 156k against 128k on ScienceWorld, and it ends below WM-PPO in both environments.

The same success takes fewer real steps than with PPO. Figure 13 in Appendix E plots success against real interaction; Table 14 reports each run’s first evaluated checkpoint above a reference success. By iteration 36 on ALFWorld, the three WM-PPO runs have passed 80% success after 69k–97k real steps and the three runs with the policy channel only after 67k–84k, against 129k–141k for PPO-256. No PPO-160 run has reached 80% by this iteration. The PPO refinements rise fastest early: CAPO and Turn-ratio PPO pass 80% after 63k and 64k real steps. WM-PPO reaches their level by iterations 30–36, and the two variants end with the highest success at iteration 36. On ScienceWorld, WM-PPO separates from PPO-160 after about 30k real steps, and the gap widens to about 6 points by 100k. Its runs pass 45% after 94k–120k real steps and the tuned SO-PPO after 97k, whereas no PPO-160 run passes 45% within its 108k–118k steps. The gap grows over training, in step with the reliability of the comparisons (Section 5.1).

![](images/dc64cba343eadae86980b2c9ad43d6688484360d6d9791593b0179af7242f00a.jpg)

![](images/897a03e3d662c34d9d9d46473c449f44d8b1d01f4ea2edb2d2e5d34811936459.jpg)  
Figure 5: Agreement of preferences with real outcomes (ScienceWorld). (a) Training branch pairs from three WM-PPO runs with different real outcomes and untied transition values, evaluated before each update, grouped by phase and margin bin (0.5 and 0.8 quantiles of nonzero margins). (b) Final evaluators score a shared pool of 775 candidate pairs from evaluation states whose outcome values, estimated from five real continuations per candidate, differ. The smaller half and largest quarter are selected by each evaluator’s own margins. Dotted line: 50%; 95% intervals: bands in (a), error bars in (b), from resampling (run, iteration) clusters and decision points, respectively. Training protocol and full results: Appendix F.1.

Computation. Serving and training the world model takes additional GPUs during training. At comparable real steps, an iteration therefore takes 1.3–1.6 times as long as for PPO-160 in wall-clock time and uses 2.7–3.8 times as many GPU-hours (Appendix E.5). This cost stays in training, where the world model and the critic extract the information implicit in environment dynamics; at inference the policy acts alone, without the extra models and GPUs of agents that keep a world model (Chae et al., 2025; Liu et al., 2026a,b). The additional computation lets WM-PPO learn more from each real state transition, a tradeoff that is useful when interaction is slow or costly.

## 5 Understanding the Gains

A transition value estimates the current policy’s chance of success after a candidate action (Equation (6)). This section examines whether these values identify choices that determine task success, how the policy and critic learn from them, and how the trained policy behaves as a result.

## 5.1 Valuing Predicted State Transitions

Section 2.1 showed that large value gaps identify comparisons the critic orders more reliably. This pattern persists during online training with predicted successors. Late in ScienceWorld training, the preferred action has the better outcome in 84% of branch pairs with different real outcomes in the largest margin bin, while the smaller bins are close to chance (Figure 5a; Table 20).

What a large margin means. A large margin indicates a large estimated difference in the current policy’s chances of success after the candidate actions. Such a difference can arise when an action commits the agent to a choice that prevents later completion. For example, a ScienceWorld task requires focusing on a non-living thing and moving it into a box. Focusing on air leaves the episode running, but air cannot be moved into the box. Across the three runs, the critics value the predicted successor of focusing on air at 0.03–0.34, compared with 0.54–0.80 for going to the hallway. None of the 24 real continuations after focusing on air succeeds, whereas 19 of 24 succeed after going to the hallway.

![](images/473867476f94638eeadbb0c4f9f7a52dfa55d42c98203c5baa2eacdc37db58c3.jpg)  
Figure 6: Online updates maintain prediction quality as the policy learns (ScienceWorld). WM-PPO (policy channel only) and Fixed WM start from the same world model. (a) Exact nextobservation prediction on executed steps. (b) Recall of episode endings. (c) Greedy success during training on dev and test games. Each model is evaluated on its policy’s visited states, in windows of three iterations plotted at the last iteration. Band: range of three online runs; Fixed WM: one run; dotted gray line: mean of three PPO-160 runs.

Such decisive choices make up a growing part of the largest bin. Its share of pairs with different real outcomes grows from 14% early in training to 33% later, and its realized gain per pair from +0.08 to +0.22 (Table 19), which keeps its gate weight high (Section 5.2).

Predicting successors improves candidate comparisons. Since a transition value approximates an action value, directly learning a Q network is a natural alternative. The relation from Equation (5),

$$
\begin{array} { r } { \hat { q } _ { t } ( a ) \approx Q ^ { \pi } ( s _ { t } , a ) = \mathbb { E } \big [ V ^ { \pi } ( s _ { t + 1 } ) \mid s _ { t } , a _ { t } = a \big ] , } \end{array}
$$

allows either a direct estimate of $Q ^ { \pi }$ or an estimate through the action’s successor. Q-PPO tests the first route by adding a network $Q _ { \omega } ( s _ { t } , y )$ trained on executed actions’ returns. It replaces the world model and critic’s candidate evaluation while retaining the candidates, margins, gate and auxiliary update of WM-PPO (policy channel only). The world model can also learn from the next observation at every executed step, including those in failed episodes. Predicting successors therefore lets the evaluation use supervision about what actions change as well as the returns used to learn value. In ScienceWorld’s largest margin bin, WM-PPO (policy channel only), which differs from Q-PPO only in how candidates are valued, favors the successful action in 85% of training branch pairs with different real outcomes, against 76% for Q-PPO (Table 20). On a shared pool of candidates, the comparisons using the world model are also more accurate at large margins (Figure 5b).

Online updates preserve prediction quality on ScienceWorld. As the policy learns, it visits new states and proposes different actions, so the world model must keep learning from the resulting transitions. With a fixed world model, prediction accuracy declines as the policy changes (Figure 6a). This decline also affects candidate evaluation: late in training, the Fixed WM run’s largest-margin agreement is 53–64%, against 88–93% for the online run with the same seed (Appendix F.1).

## 5.2 Learning from Transition Values

The comparisons supply information that additional branches alone do not. Wide-Tree-PPO keeps the trees and critic of WM-PPO (policy channel only) but drops the auxiliary update; across three runs per method, the policy-channel-only variant exceeds it in success by 10.0 percentage points on ALFWorld and 6.3 on ScienceWorld (Table 2). A single offline update on these comparisons alone also improves the policy (Section 2.3).

The value channel adds feedback for the critic. Training the critic toward the mean transition value ${ \bar { q } } _ { t }$ changes how it values the next round’s predicted successors, and hence the preferences used in later policy updates. In ScienceWorld iterations 13–36, the largest-margin comparisons favor the successful action in 78–82% of branch pairs with different outcomes, against 71–74% with the policy channel only (Figure 7b).

![](images/916f05e8e81121aba7703ef9c3503de697522129744fbb24cc026ceeebe8bbc8.jpg)

Figure 7: What the value channel adds (ScienceWorld, three runs per method). (a) Explained variance of real returns at tree nodes, using the critic for that iteration’s advantages; lines: means over runs, markers: individual runs. (b) Agreement in the largest margin bin among branch pairs whose outcomes and transition values differ; bars: 95% intervals from resampling (run, iteration) clusters. (c) Episode endings in greedy re-runs of the final checkpoints on 188 procedural games per run. Official procedural success is reported in Section 5.3.  
![](images/80b52882a1f6a4828324194936c4c549dc464488efd7094cb56e12d6e0aa616d.jpg)  
Figure 8: Gate weights during training: WM-PPO (a, green) and the policy-channel-only variant (b, blue). Top: ScienceWorld, iterations 13–60; bottom: ALFWorld, iterations 13–36. Dotted, dashed and solid lines: means over three runs for the smallest, middle and largest margin bins; band: range over runs for the largest bin. Weights use branch pairs from the preceding four iterations (Section 3.5).

A single gate decides when, and how much, these comparisons shape both networks. The same weight $w _ { \ell ( m _ { t } ) }$ scales the policy’s auxiliary advantages and the critic’s auxiliary loss (Equations (8) and (12)); while the gate is closed, the update is ordinary PPO on the trees. Each margin bin earns its weight from the realized gain of its earlier comparisons (Section 3.5).

On ScienceWorld, the gate starts closed and opens as the critic and world model become reliable. The largest bin reaches full weight early in training and keeps it as its comparisons agree more often with real outcomes (Figure 8a; Table 19); its weight still dips when its recent comparisons are less reliable, as around iteration 27. The smaller bins, whose comparisons are close to chance but carry a slightly positive gain, stay at low and fluctuating weights, so their weak signal enters both channels without dominating them.

The value channel feeds this calibration. With more reliable comparisons, the largest bin of WM-PPO opens earlier and stays open; with the policy channel only, the weights are less stable and the largest bin dips in mid-training, so less feedback reaches the two networks (Figure 8a,b). With the fuller feedback, WM-PPO pulls ahead of the policy-channel-only variant from iteration 36 onward (Tables 12 and 13), its critic fits real returns better late in training (explained variance 0.49 against 0.30; Figure 7a), and fewer procedural episodes end in failure on a wrong focus (Figure 7c).

On ALFWorld, the same calibration withdraws feedback as success approaches the ceiling. Few branch pairs then have different outcomes, and equal outcomes contribute no realized gain. The weights fluctuate, and WM-PPO’s largest-margin bin closes at iteration 36 (Figure 8a, lower). With less branch evidence and little room to raise success, the value channel gives no further gain on ALFWorld (Section 4.2).

![](images/f46c0e1265d6169205fdd05c9cc3a2aa1a7b092a4bd0bf07aeb517eccc5b8159.jpg)  
Figure 9: Gate ablation on ScienceWorld (one seed: the first run of each method). Success during training on the dev and test games. The two copies of WM-PPO fix the bin weights from iteration 13 at (1, 1, 1) (no gate) or (0, 0, 1) (top bin, the largest-margin bin only) in both channels; all other settings are unchanged.

Fixing the weights shows what this calibration contributes (Figure 9). Full weight rises fastest at first, because it admits the comparisons that the gate still holds back. Later, the near-chance preferences of the smaller bins keep entering at the same weight as the reliable ones, and this copy’s success grows no faster than PPO-160’s; on its own branch records, the gate would have admitted the small bin in only about a quarter of the iterations. Keeping only the largest bin is stable but stays below WM-PPO: it never receives the weak signal that the gate would have drawn from the smaller bins in more than half of its iterations, and it trusts the largest bin from the first iteration, before its comparisons have proven reliable (Appendix F.1). Past the early phase, WM-PPO pulls clearly ahead, and both copies end significantly below it, ordered by the reliability of the margin bins (Section 5.1).

a ALFWorld success

b ALFWorld behavior

![](images/18f89786f16034839c8387273ec3c4b7e116c7f2421d0e83e5a5ef5bf323ec91.jpg)  
c ScienceWorld success

![](images/0f3a31cd23d3d78d696ee54840eeecec3d0bd92cff9ccb03b964d32f61f8c4f3.jpg)

![](images/84d190145e9d24e5bfa496bdd981488e04ff21a36cfbc761a482de717e846824.jpg)  
Figure 10: How the trained policies advance the task: WM-PPO, PPO-256 and PPO-160, three runs each. Bars: pooled percentages; markers: individual runs. (a, c) Official greedy success by task type; shaded ALFWorld tasks require processing before placement. (b) ALFWorld behavior in greedy re-runs. Processing is measured among episodes in which the object was picked up; other rates cover all episodes. Back-and-forth walking: at least two A–B–A–B patterns, possibly overlapping, in the go to destinations; within the step limit: ending before the 40th action. (d) Procedural ScienceWorld episodes without premature focus in greedy re-runs: all procedural games and the find subset. A premature focus commits to an answer before its prerequisite state exists. Further behavior analyses: Appendix F.5.

## 5.3 How the Policy Advances the Task

The trained WM-PPO policy more often completes prerequisites before taking the action that depends on them.

ALFWorld. WM-PPO more often processes an object before placing it. Its gain over PPO-160 is largest on tasks that require heating, cooling or cleaning before placement: 31.5 points, against 21.5 on plain placing (Figure 10a). The action order helps explain this difference. On processing tasks, WM-PPO places the object before processing it in 17% of episodes in which the object was picked up, against 41% for PPO-160 (Appendix F.5). In the mug-heating game, both policies find the mug; WM-PPO takes it to the microwave, whereas PPO-160 puts it into the cabinet unheated and fails (Figure 11a). PPO-256 also learns the processing order with more interaction; WM-PPO learns it within PPO-160’s budget (Section 4.3).

ScienceWorld. WM-PPO is more successful on tasks that require intermediate steps and less often commits to an answer before meeting its prerequisites. On the procedural games, it reaches 46.3% success, against 29.6% for PPO-160 (Figure 10c). An accepted focus command commits the agent to an answer, so its timing reveals whether the necessary conditions have been met. On these games, premature focus occurs in 22.0% of WM-PPO’s episodes, against 28.7% for PPO-160 (Figure 10d). The living-thing task illustrates the difference: WM-PPO goes outside and brings a baby beaver to the box, while PPO-160 focuses on a fruit in the starting room and fails (Figure 11b).

![](images/e82a7756a432b1bb7a24b14241cb157e42afc6df90c8dcd255d5c6a79f6b66e5.jpg)

![](images/17dfc8f3e1f9dea7dadca5b7a3e35897fea5e52a707dce2282318cda4e3a4571.jpg)  
Figure 11: Two evaluation games played greedily by the final policies; highlighted: decisive steps. Check: post hoc evaluation by the world model and critic of the same WM-PPO run; displayed critic values are raw outputs. (a) ALFWorld, the second PPO-160 run and first WM-PPO run. (b) ScienceWorld, the first run of each method.

## 6 Related Work

Language agents can draw on information beyond the terminal reward in several ways. Table 3 compares the main approaches by what is predicted, how it is evaluated, where the resulting signal enters training, and what the deployed agent needs.

Observation prediction and dynamics learning. ECHO, PaW and DGS-WM use real observations as auxiliary prediction targets during RL, so the policy’s own parameters are also trained to predict what its actions produce (Shrivastava et al., 2026; Lu et al., 2026; Ran et al., 2026). DGS-WM keeps the observation fragments that best support the policy’s real next response, which favors decision-relevant content, but this selection score does not estimate eventual success. Other methods learn the dynamics before task-level policy training. Early Experience offers next-state prediction as a warm-up before imitation, and alternatively mixes self-reflection rationales into imitation (Zhang et al., 2026a); SPA writes true states into the policy’s reasoning before RL (Chen et al., 2026a); and RWML rewards predictions that are close to the true next state (Yu et al., 2026b). In all of these methods, a prediction is judged by how well it matches the real observation. WM-PPO keeps the world model separate and values each predicted change by its effect on task success. The two uses of a world model could be combined.

Table 3: How information beyond the terminal reward enters agent training. “–”: not part of the method. References are given in the text.
<table><tr><td>Method</td><td>What is predicted</td><td>How it is evaluated</td><td>Where the signal enters</td><td>Deployed agent</td></tr><tr><td>ECHO, PaW, DGS-WM</td><td>next observation, by the policy</td><td></td><td>auxiliary loss on the policy</td><td>policy</td></tr><tr><td>Early Experience, SPA</td><td>future or true states</td><td></td><td>supervised stage before RL</td><td>policy</td></tr><tr><td>RWML</td><td>next state</td><td>similarity to the true state</td><td>RL stage before task RL</td><td>policy</td></tr><tr><td>WMA, WebDreamer</td><td>change caused by each candidate</td><td>value model or prompted judge</td><td>action selection at test time</td><td>policy, model, evaluator</td></tr><tr><td>ITP</td><td>multi-step imagined rollout</td><td>the policy&#x27;s reflection (training-free variant)</td><td>context of the policy; SFT, then actor-critic RL</td><td>policy and model</td></tr><tr><td>CoMAP</td><td>imagined next state</td><td>the policy&#x27;s own reflection</td><td>supervised refinement</td><td>policy and model</td></tr><tr><td>Internalizing the Future</td><td>imagined plan (prospective rollout)</td><td>verbalized confidence, calibrated against outcomes</td><td>auxiliary step-level advantage in GRPO</td><td>policy with explicit foresight</td></tr><tr><td>MC-PPO)</td><td>ProAct (MC-GRPO / – (simulator transitions)</td><td>immediate reward and random-policy Monte-Carlo state values</td><td>group advantages (MC-GRPO); TD(0) advantages (MC-PPO)</td><td>policy</td></tr><tr><td>OPD, OCSD</td><td></td><td>token likelihood under a teacher or a hindsight self-teacher</td><td>distillation loss or rescaled token advantage</td><td>policy</td></tr><tr><td>Dreamer, MBPO, DreamGym</td><td>imagined rollouts</td><td>critic or returns on imagined data</td><td>training on imagined rollouts</td><td>policy</td></tr><tr><td>WM-PPO</td><td>next observation of each candidate, by a separate model</td><td>PPO&#x27;s critic on the predicted successor, weighted by real branches for the critic</td><td>auxiliary advantage in PPO-clip; one-step target</td><td>policy</td></tr></table>

Predicted outcomes at decision time. WMA already pairs candidate actions with predicted consequences and a value judgment. With the policy frozen, a world model predicts the next observation of each candidate, a value model scores it, and the best candidate is executed at test time (Chae et al., 2025). WMA also reports that scoring the predicted next state works better than scoring the action alone; Q-PPO makes the corresponding comparison inside training (Section 5.1). WebDreamer plans in the same way with an LLM simulator and a prompted judge (Gu et al., 2025), and QWM selects robot actions by world-model search with a learned Q-function (Dong et al., 2026). ITP lets the agent reflect on imagined multi-step rollouts before acting, either without training or after an expert warm-up and actor-critic RL that learns the imagination length together with the action; in both variants the world model stays in the deployed agent (Liu et al., 2026a).

CoMAP, which is also evaluated on ALFWorld and ScienceWorld, shares several design choices with WM-PPO (Liu et al., 2026b). Its world model is a separately parameterized copy of the same backbone, retrained on the agent’s own transitions; its labels come from real two-branch rollouts; and a gate decides when to refine. The policy drafts an action, reads the predicted next state and revises the draft after reflection. Training is supervised by expert or corrected actions, and the world model also learns from an EMA teacher that sees the real next state. The prediction thus enters the decision itself, and the world model remains part of the deployed agent. WM-PPO instead turns predicted consequences into training signals through PPO’s critic, and only the policy acts at deployment.

Learned foresight. Internalizing the Future combines future simulation and success estimation within a single policy (Zhang et al., 2026b). Its mid-training and supervised fine-tuning use foresight traces that DeepSeek V3.1 writes from recorded trajectories, including their future continuations. During RL, the policy generates an imagined plan and a confidence score before reasoning about the next action. A calibration reward trains this score against the terminal outcome, and the scores of corresponding steps, normalized within the group, form an auxiliary step-level advantage in GRPO. The feedback therefore concerns the generated plan rather than a comparison of alternative actions from a shared state. WM-PPO makes that comparison explicitly: the world model predicts the consequences of sampled actions, and PPO’s critic evaluates the resulting states. Both learn from the agent’s own interactions, and their estimates provide feedback for policy and critic updates. At inference, the policy acts without an explicit foresight stage.

Teacher and hindsight feedback. Distillation supplies token-level feedback from a stronger or better-informed model. In standard on-policy distillation, the student generates its own sequences and a teacher supervises its token distribution on these prefixes (Agarwal et al., 2024). OCSD, in its main configuration, uses the agent’s own earlier policy instead: it scores already generated tokens in matched replay contexts with and without the real later observations, and the score difference rescales the token update while keeping the sign of the original advantage (Yang et al., 2026). Its hindsight also comes from the agent’s own interaction; the difference is that OCSD re-scores executed responses, whereas WM-PPO predicts the consequences of candidates that were not executed. VASPO gives a return-trained critic and a self-teacher access to reference solutions during training and weights tokens by their judgments, on mathematical and logical reasoning (Wu et al., 2026). Like WM-PPO, it deploys the policy alone, but it predicts no state transitions. Section 4.2 compares WM-PPO with OPD and OCSD.

Model-based reinforcement learning. WM-PPO learns dynamics from the agent’s own interactions and values predicted consequences to guide policy and critic learning. Dyna adds simulated transitions to learning (Sutton, 1991); World Models and Dreamer learn behavior inside a latent model (Ha & Schmidhuber, 2018; Hafner et al., 2020, 2025); MBPO branches short model rollouts from real states (Janner et al., 2019); MVE and STEVE build the critic’s targets from model rollouts (Feinberg et al., 2018; Buckman et al., 2018); and DreamGym replaces the environment of text agents with an LLM simulator (Chen et al., 2026b). Value Prediction Networks and MuZero learn models through values (Oh et al., 2017; Schrittwieser et al., 2020), and value equivalence judges a model by the values it induces (Grimm et al., 2020). In WM-PPO, real rollouts provide every trajectory, the returns that are the critic’s main supervision, and PPO’s advantages. The world model predicts one step for the policy’s candidates, and the critic’s values of these predictions enter learning as candidate comparisons and an auxiliary critic target. This target, the mean transition value, is related to value expansion (Feinberg et al., 2018; Buckman et al., 2018) and supplements regression on real returns. The gate checks the comparisons on real outcomes, in the spirit of value equivalence; the world model itself is trained by likelihood.

Policy improvement over sampled actions. Valuing actions that were not executed, through a learned critic, is the idea of all-action policy gradients (Allen et al., 2017; Ciosek & Whiteson, 2020). MPO reweights actions sampled from the policy by their estimated values, and AWR reweights the actions in its replay buffer by their estimated advantages (Abdolmaleki et al., 2018; Peng et al., 2019); Muesli regularizes policy-gradient learning toward sampled actions valued with a learned model (Hessel et al., 2021); Sampled and Gumbel MuZero improve the policy over a sampled subset of actions with search (Hubert et al., 2021; Danihelka et al., 2022); and the MC-GRPO variant of ProAct executes candidates sampled at one state in the simulator and values each by its immediate reward plus Monte-Carlo continuations from a random policy (Yu et al., 2026c). The auxiliary term of WM-PPO belongs to this family: it moves probability toward the best of the policy’s own K samples. Its choices concern how to value these samples and how much to trust their comparisons. WM-PPO values candidates through their predicted successor states and PPO’s critic, then uses outcomes from real branches to calibrate the reliability weights. Expert Iteration, AlphaZero, Dyna-Mind and the first stage of ProAct instead train the policy toward the output of a search (Anthony et al., 2017; Silver et al., 2018; Yu et al., 2026a,c), whereas WM-PPO supplies scalar advantages for the policy’s own samples rather than a target distribution.

ProAct’s MC-PPO variant estimates state values through random-policy rollouts in the simulator, with 1,000 rollouts per state in our Sokoban comparison (Yu et al., 2026c). With comparable numbers of real environment steps, WM-PPO reaches 87.0% success versus 77.3% for MC-PPO, a significant gain of 9.7 percentage points (Appendix E.6).

Value and credit signals for language agents. QLASS guides action selection with a Q-network trained on exploration trees (Lin et al., 2025); SWEET-RL turns a privileged turn-level critic’s ranking of sampled actions into preference pairs (Zhou et al., 2025); SPA-RL redistributes the final reward into per-step progress rewards (Wang et al., 2025); and process reward models score steps from step-level labels (Lightman et al., 2024). Agent Q trains on sibling pairs whose value gap exceeds a fixed threshold (Putta et al., 2024), whereas the gate of WM-PPO measures which margins are reliable. VinePPO finds that PPO’s value network ranks candidate next steps near chance and replaces it by Monte-Carlo estimates (Kazemnejad et al., 2025), consistent with the small-margin comparisons of Sections 2.1 and 5.1. Group-based methods compare episodes (Shao et al., 2024; Ji et al., 2026) or actions taken at the same state or history (Feng et al., 2025; He et al., 2026); PPO refinements change how advantages and ratios apply within a turn (Wang et al., 2026; Li et al., 2026; Yue et al., 2025; Yu et al., 2025); and majority-vote self-training rewards agreement with the policy’s most frequent answer (Zuo et al., 2025; Shafayat et al., 2025). WM-PPO complements these forms of credit assignment with learned comparisons of unexecuted alternatives.

## 7 Discussion and Limitations

The value of a predicted consequence depends on what the current policy can still accomplish from the resulting state. As the policy improves, the same successor can acquire a different value. The world model learns which changes an action produces, and the critic relates those changes to task success using supervision from real returns. Evaluating alternatives in this way lets the policy learn preferences that a few realized outcomes leave unresolved, including between candidates whose observed continuations end with the same reward (Appendix B.2).

Large margins increasingly identify choices that change what remains achievable (Section 5.1). A premature commitment can close off success, while a step that establishes a prerequisite can make it attainable. These are the decisions on which the comparisons are most reliable. The behavioral analyses show the corresponding change in the policy: it more often completes prerequisites before placing an object or committing to an answer. Gains are largest on procedural ScienceWorld tasks and ALFWorld tasks that require heating, cooling or cleaning before placement. They are smaller on forced-choice tasks, where correctness becomes apparent only at the end.

What the world model predicts determines which part of a comparison the critic must supply. Predicted endings supply outcomes directly, and comparisons between continuing successors rely on the critic. For the latter comparisons on ScienceWorld, the value channel raises agreement with real outcomes from 55% to 62% (Appendix F.1). On ALFWorld, no predicted ending settles a comparison, so every comparison rests on the critic. The gate assigns only partial weights there (Figure 8). The value channel improves early candidate comparisons and the critic’s fit to real returns later in training (Figure 7; Section 5.2).

WM-PPO trades additional prediction and evaluation during training for more learning from each real interaction. The benefit is higher success at comparable interaction budgets and fewer steps to reach PPO’s success levels. The extra computation is confined to training; the deployed policy acts alone.

Limitations. Our main evaluation covers one 4B base model and two text environments, with episodes limited to 40 steps. Branching and gate calibration require a reproducible prefix, as in our deterministic environments, or a saved state. A branch’s first action is sampled conditional on differing from the chain’s action. The update does not correct for this conditioning.

A single greedy prediction of the next observation does not represent uncertainty over possible outcomes. Longer-range effects enter only through the critic. Prediction errors can bias the auxiliary policy advantages and critic targets. Comparisons cover only the policy’s candidates, and many small-margin comparisons receive little or no weight. Extending the benefit to these decision points remains an open problem.

## 8 Conclusion

WM-PPO uses learned dynamics to evaluate candidate actions through their predicted consequences. A world model trained on real transitions predicts each candidate’s next observation, and PPO’s critic, trained on real returns, values the predicted successor according to the current policy’s chance of success. In the policy channel, comparisons between transition values become auxiliary advantages for the policy’s own samples. In the value channel, the mean transition value becomes a one-step target for the critic at the current state. Real branch outcomes calibrate the weights of both channels. On ALFWorld and ScienceWorld, the resulting policy achieves higher success than PPO with comparable real interaction. It matches the strongest PPO refinements we tested on ALFWorld and exceeds the best of them by 6.5 points on ScienceWorld. Unlike distillation, WM-PPO needs neither a teacher model nor privileged information, yet it also surpasses the teacher-based methods we tested. When deployed alone, the policy more often completes prerequisite steps before committing to an answer or placing an object.

## Acknowledgements

We thank the developers of ALFWorld and ScienceWorld for making their environments available, the Qwen team for releasing the base model, and the authors of the baseline methods for releasing their code.

## Code and Data Availability

The code, training configurations and evaluation scripts will be released upon publication. Both environments and the base model are publicly available, and Appendix B describes the training and evaluation task sets.

## References

Abdolmaleki, A., Springenberg, J. T., Tassa, Y., Munos, R., Heess, N., and Riedmiller, M. Maximum a posteriori policy optimisation. In International Conference on Learning Representations, 2018. URL https://openreview.net/forum?id=S1ANxQW0b.

Agarwal, R., Vieillard, N., Zhou, Y., Stanczyk, P., Ramos Garea, S., Geist, M., and Bachem, O. On-policy distillation of language models: Learning from self-generated mistakes. In Kim, B., Yue, Y., Chaudhuri, S., Fragkiadaki, K., Khan, M., and Sun, Y. (eds.), International Conference on Learning Representations, volume 2024, pp. 21246–21263, 2024. URL https://proceedi ngs.iclr.cc/paper\_files/paper/2024/file/5be69a584901a26c521c2b51 e40a4c20-Paper-Conference.pdf.

Allen, C., Asadi, K., Roderick, M., Mohamed, A., Konidaris, G., and Littman, M. Mean actor critic, 2017. URL https://arxiv.org/abs/1709.00503.

Anthony, T., Tian, Z., and Barber, D. Thinking fast and slow with deep learning and tree search. In Guyon, I., Luxburg, U. V., Bengio, S., Wallach, H., Fergus, R., Vishwanathan, S., and Garnett, R. (eds.), Advances in Neural Information Processing Systems, volume 30. Curran Associates, Inc., 2017. URL https://proceedings.neurips.cc/paper\_files/paper/2017/fi le/d8e1344e27a5b08cdfd5d027d9b8d6de-Paper.pdf.

Buckman, J., Hafner, D., Tucker, G., Brevdo, E., and Lee, H. Sample-efficient reinforcement learning with stochastic ensemble value expansion. In Bengio, S., Wallach, H., Larochelle, H., Grauman, K., Cesa-Bianchi, N., and Garnett, R. (eds.), Advances in Neural Information Processing Systems, volume 31. Curran Associates, Inc., 2018. URL https://proceedings.neurips.cc/p aper\_files/paper/2018/file/f02208a057804ee16ac72ff4d3cec53b-Pap er.pdf.

Chae, H., Kim, N., Ong, K., Gwak, M., Song, G., Kim, J., Kim, S., Lee, D., and Yeo, J. Web agents with world models: Learning and leveraging environment dynamics in web navigation. In Yue, Y., Garg, A., Peng, N., Sha, F., and Yu, R. (eds.), International Conference on Learning Representations, volume 2025, pp. 63707–63738, 2025. URL https://proceedings.ic lr.cc/paper\_files/paper/2025/file/a00548031e4647b13042c97c922fad f1-Paper-Conference.pdf.

Chen, S., Zhu, T., Wang, Z., Zhang, J., Wang, K., Zhou, R., Gao, S., Xiao, T., Teh, Y. W., He, J., and Li, M. Why do LLM agents fail in exploring new environments? A world-modeling perspective. In Lee, H.-Y., Dev, S., Li, J., and Elliott, D. (eds.), Findings ofthe Associationfor Computational Linguistics: EMNLP 2026, pp. 21612–21626, Budapest, Hungary, October 2026a. Association for Computational Linguistics. URL https://aclanthology.org/2026.findings-emn lp.1023/.

Chen, Z., Zhao, Z., Zhang, K., Liu, B., Qi, Q., Wu, Y., Kalluri, T., Cao, X., Xiong, Y., Tong, H., Yao, H., Li, H., Zhu, J., Li, X., Song, D., Li, B., Weston, J. E., and Huynh, D. Scaling agent learning via experience synthesis. In Vondrick, C., Hariharan, B., Raffel, C., Pinto, L., Yang, D., and Faust, A. (eds.), International Conference on Learning Representations, volume 2026, pp. 121394–121420, 2026b. URL https://proceedings.iclr.cc/paper\_files/paper/2026/file/ c53793534833d30a40f3d18352735519-Paper-Conference.pdf.

Ciosek, K. and Whiteson, S. Expected policy gradients for reinforcement learning. Journal of Machine Learning Research, 21(52):1–51, 2020.

Danihelka, I., Guez, A., Schrittwieser, J., and Silver, D. Policy improvement by planning with Gumbel. In International Conference on Learning Representations, 2022.

Dong, P., Jia, Y., Finn, C., and Sadigh, D. Q-learning with world models, 2026. URL https: //arxiv.org/abs/2608.17163.

Feinberg, V., Wan, A., Stoica, I., Jordan, M. I., Gonzalez, J. E., and Levine, S. Model-based value estimation for efficient model-free reinforcement learning, 2018. URL https://arxiv.org/ abs/1803.00101.

Feng, L., Xue, Z., Liu, T., and An, B. Group-in-group policy optimization for LLM agent training. In Belgrave, D., Zhang, C., Lin, H., Pascanu, R., Koniusz, P., Ghassemi, M., and Chen, N. (eds.), Advances in Neural Information Processing Systems, volume 38, Main Conference, pp. 46375–46408. Curran Associates, Inc., 2025. doi: 10.52202/085713-1544. URL https: //proceedings.neurips.cc/paper\_files/paper/2025/file/420c9f777c0 b4f78d515e53cf74d58b2-Paper-Conference.pdf.

Grimm, C., Barreto, A., Singh, S., and Silver, D. The value equivalence principle for model-based reinforcement learning. In Larochelle, H., Ranzato, M., Hadsell, R., Balcan, M., and Lin, H. (eds.), Advances in Neural Information Processing Systems, volume 33, pp. 5541–5552. Curran Associates, Inc., 2020. URL https://proceedings.neurips.cc/paper\_files/p aper/2020/file/3bb585ea00014b0e3ebe4c6dd165a358-Paper.pdf.

Gu, Y., Zhang, K., Ning, Y., Zheng, B., Gou, B., Xue, T., Chang, C., Srivastava, S., Xie, Y., Qi, P., Sun, H., and Su, Y. Is your LLM secretly a world model of the internet? Model-based planning for web agents. Transactions on Machine Learning Research, 2025. URL https: //openreview.net/forum?id=c6l7yA0HSq.

Ha, D. and Schmidhuber, J. World models, 2018. URL https://arxiv.org/abs/1803.1 0122.

Hafner, D., Lillicrap, T. P., Ba, J., and Norouzi, M. Dream to control: Learning behaviors by latent

imagination. In 8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020. OpenReview.net, 2020. URL https://openreview.n et/forum?id=S1lOTC4tDS.

Hafner, D., Pasukonis, J., Ba, J., and Lillicrap, T. Mastering diverse control tasks through world models. Nature, 640(8059):647–653, April 2025. ISSN 1476-4687. doi: 10.1038/s41586-025-087 44-2. URL http://dx.doi.org/10.1038/s41586-025-08744-2.

He, S., Feng, L., Wei, Q., Cheng, X., Feng, L., and An, B. Hierarchy-of-groups policy optimization for long-horizon agentic tasks. In Vondrick, C., Hariharan, B., Raffel, C., Pinto, L., Yang, D., and Faust, A. (eds.), International Conference on Learning Representations, volume 2026, pp. 27572–27593, 2026. URL https://proceedings.iclr.cc/paper\_files/paper/ 2026/file/2e7eb932f93d5db3b61c7a001e72fd0f-Paper-Conference.pdf.

Hessel, M., Danihelka, I., Viola, F., Guez, A., Schmitt, S., Sifre, L., Weber, T., Silver, D., and van Hasselt, H. Muesli: Combining improvements in policy optimization. In Proceedings ofthe 38th International Conference on Machine Learning, volume 139 of Proceedings ofMachine Learning Research, pp. 4214–4226. PMLR, 2021. URL https://proceedings.mlr.press/v139 /hessel21a.html.

Hou, Z., Hu, Z., Li, Y., Lu, R., Tang, J., and Dong, Y. TreeRL: LLM reinforcement learning with onpolicy tree search. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 12355–12369, 2025. doi: 10.18653/v1/2025.acl-long.604. URL https://aclanthology.org/2025.acl-long.604/.

Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., and Chen, W. LoRA: Low-rank adaptation of large language models. In The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022. OpenReview.net, 2022. URL https://openreview.net/forum?id=nZeVKeeFYf9.

Hubert, T., Schrittwieser, J., Antonoglou, I., Barekatain, M., Schmitt, S., and Silver, D. Learning and planning in complex action spaces. In Proceedings ofthe 38th International Conference on Machine Learning, volume 139 of Proceedings of Machine Learning Research, pp. 4476–4486, 2021.

Janner, M., Fu, J., Zhang, M., and Levine, S. When to trust your model: Model-based policy optimization. In Wallach, H., Larochelle, H., Beygelzimer, A., d'Alché-Buc, F., Fox, E., and Garnett, R. (eds.), Advances in Neural Information Processing Systems, volume 32. Curran Associates, Inc., 2019. URL https://proceedings.neurips.cc/paper\_files/p aper/2019/file/5faf461eff3099671ad63c6f3f094f7f-Paper.pdf.

Ji, Y., Ma, Z., Wang, Y., Chen, G., Chu, X., and Wu, L. Tree search for LLM agent reinforcement learning. In Vondrick, C., Hariharan, B., Raffel, C., Pinto, L., Yang, D., and Faust, A. (eds.), International Conference on Learning Representations, volume 2026, pp. 87362–87388, 2026. URL https://proceedings.iclr.cc/paper\_files/paper/2026/file/8c73 04e77c832ddc70075dfee081ca6c-Paper-Conference.pdf.

Kazemnejad, A., Aghajohari, M., Portelance, E., Sordoni, A., Reddy, S., Courville, A., and Le Roux, N. VinePPO: Refining credit assignment in RL training of LLMs. In Singh, A., Fazel, M., Hsu, D., Lacoste-Julien, S., Berkenkamp, F., Maharaj, T., Wagstaff, K., and Zhu, J. (eds.), Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings ofMachine Learning Research, pp. 29557–29590. PMLR, 13–19 Jul 2025. URL https://proceeding s.mlr.press/v267/kazemnejad25a.html.

Li, C., Elmahdy, A., Boyd, A., Wang, Z., Zeng, S., Garcia, A., Bhatia, P., Kass-Hout, T., Xiao, C., and Hong, M. Stabilizing off-policy training for long-horizon LLM agent via turn-level importance sampling and clipping-triggered normalization, 2026. URL https://arxiv.org/abs/25 11.20718.

Lightman, H., Kosaraju, V., Burda, Y., Edwards, H., Baker, B., Lee, T., Leike, J., Schulman, J., Sutskever, I., and Cobbe, K. Let’s verify step by step. In Kim, B., Yue, Y., Chaudhuri, S., Fragkiadaki, K., Khan, M., and Sun, Y. (eds.), International Conference on Learning Representations,

volume 2024, pp. 39578–39601, 2024. URL https://proceedings.iclr.cc/paper\_ files/paper/2024/file/aca97732e30bcf1303bc22ac3924fd16-Paper-Con ference.pdf.

Lin, Z., Tang, Y., Yao, X., Yin, D., Hu, Z., Sun, Y., and Chang, K.-W. QLASS: Boosting language agent inference via Q-guided stepwise search. In Singh, A., Fazel, M., Hsu, D., Lacoste-Julien, S., Berkenkamp, F., Maharaj, T., Wagstaff, K., and Zhu, J. (eds.), Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings ofMachine Learning Research, pp. 37942–37958. PMLR, 13–19 Jul 2025. URL https://proceedings.mlr. press/v267/lin25l.html.

Liu, Y., Wang, J., Wang, H., Guo, B., and Li, W. Imagine-then-Plan: Agent learning from adaptive lookahead with world models. In Lee, H.-Y., Dev, S., Li, J., and Elliott, D. (eds.), Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing, pp. 27427–27445, Budapest, Hungary, October 2026a. Association for Computational Linguistics. ISBN 979-8- 89176-459-0. URL https://aclanthology.org/2026.emnlp-main.1198/.

Liu, Y., Wang, J., Wang, H., and Li, W. CoMAP: Co-evolving world models and agent policies for LLM agents. In Lee, H.-Y., Dev, S., Li, J., and Elliott, D. (eds.), Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing, pp. 27461–27480, Budapest, Hungary, October 2026b. Association for Computational Linguistics. ISBN 979-8-89176-459-0. URL https://aclanthology.org/2026.emnlp-main.1200/.

Lu, N., Lin, B., Liu, S., Wu, J., Lv, H., Wei, Y., Zhu, L., Qian, S., Wang, X., Chen, Y.-C., Wang, Q., and Tang, K. Policy and world modeling co-training for language agents. In Lee, H.-Y., Dev, S., Li, J., and Elliott, D. (eds.), Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing, pp. 42962–42979, Budapest, Hungary, October 2026. Association for Computational Linguistics. ISBN 979-8-89176-459-0. URL https://aclanthology.o rg/2026.emnlp-main.1885/.

Oh, J., Singh, S., and Lee, H. Value prediction network. In Guyon, I., Luxburg, U. V., Bengio, S., Wallach, H., Fergus, R., Vishwanathan, S., and Garnett, R. (eds.), Advances in Neural Information Processing Systems, volume 30. Curran Associates, Inc., 2017. URL https://proceedings. neurips.cc/paper\_files/paper/2017/file/ffbd6cbb019a1413183c8d08f 2929307-Paper.pdf.

Peng, X. B., Kumar, A., Zhang, G., and Levine, S. Advantage-weighted regression: Simple and scalable off-policy reinforcement learning. arXiv preprint arXiv:1910.00177, 2019.

Putta, P., Mills, E., Garg, N., Motwani, S., Finn, C., Garg, D., and Rafailov, R. Agent Q: Advanced reasoning and learning for autonomous AI agents, 2024. URL https://arxiv.org/abs/ 2408.07199.

Ran, M., Chen, M., Huang, D., Xiong, W., Huang, F., and Liu, W. Model what matters: Decisionguided selective world modeling for language agents, 2026. URL https://arxiv.org/ab s/2610.19642.

Schrittwieser, J., Antonoglou, I., Hubert, T., Simonyan, K., Sifre, L., Schmitt, S., Guez, A., Lockhart, E., Hassabis, D., Graepel, T., Lillicrap, T., and Silver, D. Mastering Atari, Go, chess and shogi by planning with a learned model. Nature, 588(7839):604–609, December 2020. doi: 10.1038/s41586-020-03051-4.

Schulman, J., Moritz, P., Levine, S., Jordan, M. I., and Abbeel, P. High-dimensional continuous control using generalized advantage estimation. In 4th International Conference on Learning Representations, ICLR 2016, San Juan, Puerto Rico, May 2-4, 2016, Conference Track Proceedings, 2016. URL https://doi.org/10.48550/ARXIV.1506.02438.

Schulman, J., Wolski, F., Dhariwal, P., Radford, A., and Klimov, O. Proximal policy optimization algorithms, 2017. URL https://arxiv.org/abs/1707.06347.

Shafayat, S., Tajwar, F., Salakhutdinov, R., Schneider, J., and Zanette, A. Can large reasoning models self-train?, 2025. URL https://arxiv.org/abs/2505.21444.

Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Bi, X., Zhang, H., Zhang, M., Li, Y. K., Wu, Y., and

Guo, D. DeepSeekMath: Pushing the limits of mathematical reasoning in open language models, 2024. URL https://arxiv.org/abs/2402.03300.

Shridhar, M., Yuan, X., Côté, M.-A., Bisk, Y., Trischler, A., and Hausknecht, M. J. ALFWorld: Aligning text and embodied environments for interactive learning. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021. OpenReview.net, 2021. URL https://openreview.net/forum?id=0IOX0YcCdTn.

Shrivastava, V., Kauffmann, P., Awadallah, A., and Papailiopoulos, D. ECHO: Terminal agents learn world models for free, 2026. URL https://arxiv.org/abs/2605.24517.

Silver, D., Hubert, T., Schrittwieser, J., Antonoglou, I., Lai, M., Guez, A., Lanctot, M., Sifre, L., Kumaran, D., Graepel, T., Lillicrap, T., Simonyan, K., and Hassabis, D. A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play. Science, 362(6419): 1140–1144, December 2018. doi: 10.1126/science.aar6404.

Sutton, R. S. Dyna, an integrated architecture for learning, planning, and reacting. ACM SIGART Bulletin, 2(4):160–163, July 1991. ISSN 0163-5719. doi: 10.1145/122344.122377. URL http://dx.doi.org/10.1145/122344.122377.

Sutton, R. S. and Barto, A. G. Reinforcement Learning: An Introduction. MIT Press, second edition, 2018.

Tesauro, G. Temporal difference learning and TD-Gammon. Communications ofthe ACM, 38(3): 58–68, 1995.

Wang, D., Li, Q., Cheng, M., Ouyang, J., Yu, S., Liu, C., Wang, S., Liu, Q., and Chen, E. CAPO: Critic-guided action-aligned policy optimization for advancing LLM agent capabilities, 2026. URL https://arxiv.org/abs/2604.18401.

Wang, H., Leong, C. T., Wang, J., Wang, J., and Li, W. SPA-RL: Reinforcing LLM agents via stepwise progress attribution, 2025. URL https://arxiv.org/abs/2505.20732.

Wang, R., Jansen, P., Côté, M.-A., and Ammanabrolu, P. ScienceWorld: Is your agent smarter than a 5th grader? In Goldberg, Y., Kozareva, Z., and Zhang, Y. (eds.), Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pp. 11279–11298, Abu Dhabi, United Arab Emirates, December 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.emnlp-main.775. URL https://aclanthology.org/2022.emnl p-main.775/.

Wu, F., Zhang, Z., Zhang, J., and Du, J. VASPO: Self-distillation with privileged value estimation, 2026. URL https://arxiv.org/abs/2610.20516.

Yang, A., Li, A., Yang, B., Zhang, B., Hui, B., Zheng, B., Yu, B., Gao, C., Huang, C., Lv, C., Zheng, C., Liu, D., Zhou, F., Huang, F., Hu, F., Ge, H., Wei, H., Lin, H., Tang, J., Yang, J., Tu, J., Zhang, J., Yang, J., Yang, J., Zhou, J., Zhou, J., Lin, J., Dang, K., Bao, K., Yang, K., Yu, L., Deng, L., Li, M., Xue, M., Li, M., Zhang, P., Wang, P., Zhu, Q., Men, R., Gao, R., Liu, S., Luo, S., Li, T., Tang, T., Yin, W., Ren, X., Wang, X., Zhang, X., Ren, X., Fan, Y., Su, Y., Zhang, Y., Zhang, Y., Wan, Y., Liu, Y., Wang, Z., Cui, Z., Zhang, Z., Zhou, Z., and Qiu, Z. Qwen3 technical report, 2025. URL https://arxiv.org/abs/2505.09388.

Yang, Y., Qin, C., Liu, X., Chen, C., Dong, Q., Zhang, Y., Liu, C., Yang, Z., Pan, L., Lin, J., and Feng, Y. Agentic reinforcement learning with observation-calibrated self-distillation, 2026. URL https://arxiv.org/abs/2608.04788.

Yu, Q., Zhang, Z., Zhu, R., Yuan, Y., Zuo, X., Yue, Y., Dai, W., Fan, T., Liu, G., Liu, J., Liu, L., Liu, X., Lin, H., Lin, Z., Ma, B., Sheng, G., Tong, Y., Zhang, C., Zhang, M., Zhang, R., Zhang, W., Zhu, H., Zhu, J., Chen, J., Chen, J., Wang, C., Yu, H., Song, Y., Wei, X., Zhou, H., Liu, J., Ma, W.-Y., Zhang, Y.-Q., Yan, L., Wu, Y., and Wang, M. DAPO: An open-source LLM reinforcement learning system at scale. In Belgrave, D., Zhang, C., Lin, H., Pascanu, R., Koniusz, P., Ghassemi, M., and Chen, N. (eds.), Advances in Neural Information Processing Systems, volume 38, pp. 113222–113244. Curran Associates, Inc., 2025. doi: 10.52202/085713-3775. URL https://proceedings.neurips.cc/paper\_files/paper/2025/file/a4277 440d50f1f15d2cb4c14f7e0c0d2-Paper-Conference.pdf.

Yu, X., Peng, B., Galley, M., Cheng, H., Wu, Q., Kulkarni, J., Nath, S., Yu, Z., and Gao, J. Dyna-Mind: Learning to simulate from experience for better AI agents. In Vondrick, C., Hariharan, B., Raffel, C., Pinto, L., Yang, D., and Faust, A. (eds.), International Conference on Learning Representations, volume 2026, pp. 32435–32459, 2026a. URL https://proceedings.ic lr.cc/paper\_files/paper/2026/file/372bd0e47f2d5bceca7e300e144684 9c-Paper-Conference.pdf.

Yu, X., Peng, B., Xu, R., Shen, Y., He, P., Nath, S., Singh, N., Gao, J., and Yu, Z. Reinforcement world model learning for LLM-based agents, 2026b. URL https://arxiv.org/abs/2602 .05842.

Yu, Y., Yang, M., Li, J., Gao, Y., Liu, F., Yang, Y., Lin, Z., Lyu, J., Liu, Y., Lu, Z., Ye, D., and Jiang, J. ProAct: Agentic lookahead in interactive environments, 2026c. URL https: //arxiv.org/abs/2602.05327.

Yuan, Y., Yue, Y., Zhu, R., Fan, T., and Yan, L. What’s behind PPO’s collapse in long-CoT? Value optimization holds the secret. arXiv preprint arXiv:2503.01491, 2025.

Yue, Y., Yuan, Y., Yu, Q., Zuo, X., Zhu, R., Xu, W., Chen, J., Wang, C., Fan, T., Du, Z., Wei, X., Yu, X., Liu, G., Liu, J., Liu, L., Lin, H., Lin, Z., Ma, B., Zhang, C., Zhang, M., Zhang, W., Zhu, H., Zhang, R., Liu, X., Wang, M., Wu, Y., and Yan, L. VAPO: Efficient and reliable reinforcement learning for advanced reasoning tasks, 2025. URL https://arxiv.org/abs/2504.05118.

Zhang, K., Chen, X., Liu, B., Xue, T., Liao, Z., Liu, Z., Wang, X., Ning, Y., Chen, Z., Fu, X., Xie, J., Sun, Y., Gou, B., Qi, Q., Meng, Z., Yang, J., Zhang, N., Li, X., Shah, A., Huynh, D., Li, H., Yang, Z., Cao, S., Jang, L., Zhou, S., Zhu, J., Sun, H., Weston, J., Su, Y., and Wu, Y. Agent learning via early experience. In Proceedings of the 43rd International Conference on Machine Learning, volume 306 of Proceedings of Machine Learning Research, pp. 154443–154475. PMLR, 2026a. URL https://proceedings.mlr.press/v306/zhang26m.html.

Zhang, X., Zhou, Z., Qiao, L., Qin, Y., Li, K., Sun, X., Tan, X., Qu, C., and Qi, Y. Internalizing the future: A unified agentic training paradigm for world model planning. In Lee, H.-Y., Dev, S., Li, J., and Elliott, D. (eds.), Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing, pp. 16068–16118, Budapest, Hungary, October 2026b. Association for Computational Linguistics. URL https://aclanthology.org/2026.emnlp-mai n.698/.

Zheng, C., Liu, S., Li, M., Chen, X.-H., Yu, B., Gao, C., Dang, K., Liu, Y., Men, R., Yang, A., Zhou, J., and Lin, J. Group sequence policy optimization, 2025. URL https://arxiv.org/abs/ 2507.18071.

Zhou, Y., Jiang, S., Tian, Y., Weston, J., Levine, S., Sukhbaatar, S., and Li, X. SWEET-RL: Training multi-turn LLM agents on collaborative reasoning tasks, 2025. URL https://arxiv.org/ abs/2503.15478.

Zuo, Y., Zhang, K., Sheng, L., Qu, S., Cui, G., Zhu, X., Li, H., Zhang, Y., Long, X., Hua, E., Qi, B., Sun, Y., Ma, Z., Yuan, L., Ding, N., and Zhou, B. TTRL: Test-time reinforcement learning. In Belgrave, D., Zhang, C., Lin, H., Pascanu, R., Koniusz, P., Ghassemi, M., and Chen, N. (eds.), Advances in Neural Information Processing Systems, volume 38, Main Conference, pp. 131459–131483. Curran Associates, Inc., 2025. doi: 10.52202/085713-4376. URL https: //proceedings.neurips.cc/paper\_files/paper/2025/file/be690ea16f0 05c174f6c4102a5970e67-Paper-Conference.pdf.

## A Prompts

The policy and world model use chat prompts, each consisting of a system message followed by one user message. Fields in braces are filled at every step. The policy sees the five most recent observation–action pairs; the action is the last <action> tag of its response (ALFWorld: one of the admissible actions; ScienceWorld: a free-text command, judged by the simulator).

Policy · ALFWorld   
System: You are an expert agent operating in the ALFRED embodied household environment.   
User:   
Your task is to: {task}   
Prior to this step, you have already taken {t} step(s). Below are the most recent {k}   
observations and the corresponding actions you took: [Observation {i}: ’{o\_i}’,   
Action {i}: ’{a\_i}’] ...   
You are now at step {t+1} and your current observation is: {obs}   
Your admissible actions of the current situation are: [{admissible actions}].   
Now it’s your turn to take an action. First reason briefly (at most two sentences) about   
the current situation inside <think> </think> tags. Then choose one admissible action   
for the current step and present it inside <action> </action> tags.

At the first step the history line is omitted and the observation is the initial room description.

## Policy · ScienceWorld

System: You are an expert agent operating in the ScienceWorld environment, a text-based   
simulator of elementary science experiments.   
User:   
Your task is to: {task}   
Rules: ’focus on OBJ’ declares which object the task is about and cannot be undone;   
focusing on a wrong object ends the task as a failure, so explore first and focus   
only on the object the task asks about. OBJ must be an object you can refer to now (   
listed below); LOC is a room name (teleport moves you there directly). Answering   
questions such as ’which one ...’ is done by focusing on the answer object.   
Prior to this step, you have already taken {t} step(s). Below are the most recent {k}   
observations and the corresponding actions you took:   
[Observation {i}: ’{o\_i}’, Action {i}: ’{a\_i}’] ...   
You are now at step {t+1} and your current observation is:   
{obs}   
Action templates: [look around, look at OBJ, look in OBJ, inventory, task, teleport to LOC,   
go to LOC, open OBJ, close OBJ, activate OBJ, deactivate OBJ, pick up OBJ, put down   
OBJ, move OBJ to OBJ, pour OBJ in OBJ, dunk OBJ in OBJ, mix OBJ, use OBJ on OBJ,   
connect OBJ to OBJ, disconnect OBJ, eat OBJ, flush OBJ, read OBJ, focus on OBJ, wait,   
wait1].   
Objects you can refer to: [{objects}].   
Now it’s your turn to take an action. First reason step by step (at most two sentences)   
inside <think> </think>, then give exactly one action inside <action> </action>.

## World model · ALFWorld

System: You are a world model of the ALFRED text household environment.   
User:   
Task: {task}   
Initial observation: {room}   
History of steps so far: [Step {i}: observation ’{o\_i}’, action ’{a\_i}’] ...   
Current observation: {obs}   
Admissible actions: [{admissible actions}]   
Action taken now: {action}   
Predict the environment’s next observation exactly as the environment would print it,   
inside <obs> </obs>.

World model · ScienceWorld   
System: You are a world model of the ScienceWorld text environment.   
User:   
Task: {task}   
Recent actions: [Step {i}: action ’{a\_i}’] ... (the last 8 actions)   
Current observation:   
{obs}   
Objects the agent can refer to: [{objects}]   
Action taken now: {action}   
Predict the environment’s next observation exactly as the environment would print it (   
action feedback, [Current room], [Inventory] and [score] lines), inside <obs> </obs>.

The world model’s target is the true next observation inside <obs> tags. In ScienceWorld, our environment interface appends [task completed] or [task failed] at an episode ending, which the world model can then predict. The critic receives the predicted successor in the policy’s prompt format, with the prediction as its current observation.

## B Implementation and method details

## B.1 One training iteration

Algorithm 2 follows one iteration from real interaction through candidate evaluation to the updates of all three models.

Algorithm 2 Full iteration of WM-PPO (summarized in Algorithm 1). Predictions and transition   
values are computed without gradients; each model is updated only by its own loss. Equation number   
refer to Section 3.   
Require: policy π<sub>θ</sub>, critic $\overline { { V _ { \psi } } }$ and world model $p _ { \phi }$ from the previous iteration; branch pairs $\overline { { \mathcal { P } } }$ stored in the last   
M iterations   
▷ Real interaction (Section 3.4)   
1: for each of 64 training tasks do   
2: play a chain with $\pi _ { \boldsymbol { \theta } } ;$ at each $s _ { t } ,$ sample $K$ responses $y _ { t } ^ { ( k ) }$ with actions $a _ { t } ^ { ( k ) }$ and execute $a _ { t } ^ { ( 1 ) }$   
3: play 3 branches: replay the chain to a uniformly drawn step, resample until the action differs from the   
chain’s (at most eight times; otherwise no calibration pair), play to the end   
4: end for   
▷ Evaluating predicted transitions (Section 3.1)   
5: for each chain state $s _ { t }$ with at least two distinct candidates do   
6: for each distinct a: $\hat { o } _ { t + 1 } ( a ) \gets \mathrm { g r e e d y } p _ { \phi } ( \cdot \mid x _ { t } , a ) , ~ \hat { q } _ { t } ( a ) \gets V _ { \psi } \big ( C ( s _ { t } , a , \hat { o } _ { t + 1 } ( a ) ) \big )$ Eq. (6)   
7: ScienceWorld: if $\hat { o } _ { t + 1 } ( a )$ ends the episode, $\hat { q } _ { t } ( a ) \gets 1$ (completed) or 0 (failed)   
8: aˆ<sub>t</sub> ← arg max<sub>a</sub> $\hat { q } _ { t } ( a ) .$ margin $m _ { t } \gets \hat { q } _ { t } ( \hat { a } _ { t } ) - \operatorname* { m a x } _ { a \neq \hat { a } _ { t } } \hat { q } _ { t } ( a ) ;$ skip $s _ { t }$ if $m _ { t } = 0$   
9: compute q¯<sub>t</sub> over all K samples, including duplicates Eq. (11)   
10: end for   
▷ Reliability gate (Section 3.5)   
11: margin bins $\ell = 1 , 2 , 3 \colon$ split at the 0.5 and 0.8 quantiles of this iteration’s nonzero margins $m _ { t }$   
12: value this iteration’s branch pairs $( a _ { t } ^ { \mathrm { c h } } , a _ { t } ^ { \mathrm { b r } } )$ in the same way and bin them; they only enter later weights   
13: for each bin with $n _ { \ell } \ge n _ { \mathrm { m i n } }$ stored pairs in $\mathcal { P } \mathrm { : }$ g<sub>ℓ</sub> ← mean-∆G sign(∆ˆq) Eq. (16)   
14: w<sub>ℓ</sub> $\begin{array} { r } { \gets \mathrm { c l i p } \big ( \frac { n _ { \ell } } { n _ { \ell } + n _ { 0 } } \cdot \frac { g _ { \ell } } { g _ { \mathrm { r e f } } } , 0 , 1 \big ) } \end{array}$ ; other bins: w<sub>ℓ</sub> ← 0; iterations 13–14: start-up rule of Table 5 Eq. (17)   
▷ Updates (Sections 3.2 and 3.4)   
15: each sample $y _ { t } ^ { ( k ) }$ at an evaluated $s _ { t }$ with $w _ { \ell ( m _ { t } ) } > 0 ; A _ { t , k } ^ { \mathrm { a u x } }  \nu w _ { \ell ( m _ { t } ) } ( z _ { t , k } - \bar { z } _ { t } )$ Eq. (8)   
16: each executed response: $A _ { t } ^ { \mathrm { G A E } }$ by tree GAE on real returns, whitened to $\hat { A } _ { t }$ Eq. (14)   
17: policy π : one PPO-clip loss over real rows $( s _ { t } , y _ { t } , \hat { A } _ { t } )$ and auxiliary rows $( s _ { t } , y _ { t } ^ { ( k ) } , A _ { t , k } ^ { \mathrm { a u x } } )$ Eq. (10)   
18: critic $V _ { \psi } \colon$ regression on returns plus $\beta w _ { \ell ( m _ { t } ) } \big ( V _ { \psi } ( s _ { t } ) - \mathrm { s g } [ \bar { q } _ { t } ] \big ) ^ { 2 }$ at evaluated states Eqs. (12), (13)   
19: world model $p _ { \phi } \colon$ next-observation loss on this iteration’s executed transitions (run in parallel) Eq. (15)   
20: add the new branch pairs to $\mathcal { P } ;$ the updated models are used from the next iteration

Order within an iteration. All data collection and candidate evaluation in an iteration use the models from the preceding iteration. The world-model update on the iteration’s executed transitions runs in parallel with the policy and critic updates and replaces the served model when it finishes. Within the trainer, the critic first values every real node and every predicted successor, without gradients. The gate weights, auxiliary rows and auxiliary critic targets $\bar { q } _ { t }$ are then computed, the policy is updated on ordinary and auxiliary rows, and the critic is updated last. The branch pairs of an iteration are valued by the same pre-update critic, binned with that iteration’s thresholds, and used from the next iteration on.

## B.2 Advantage estimation on the trees

The critic’s regression. Besides the regression of Equation (13), the implementation adds a siblingcentered term at the nodes s with two or more executed children $s ^ { ( c ) } , c = 1 , \dots , C _ { s }$ :

$$
\sum _ { c = 1 } ^ { C _ { s } } \Big ( \big ( V _ { \psi } ( s ^ { ( c ) } ) - \overline { { V } } ( s ) \big ) - \big ( G ( s ^ { ( c ) } ) - \overline { { G } } ( s ) \big ) \Big ) ^ { 2 } ,
$$

where $\overline { { V } } ( s )$ and $\overline { { G } } ( s )$ average over the children of $s ,$ and a child that ends the episode enters with its final reward in place of a value. The term, with weight 1, asks that the centered values of the children match their centered returns, which is the within-state comparison that Equation (7) relies on. Wide-Tree-PPO and Q-PPO use the same term.

The tree estimator. The raw estimates $A ^ { \mathrm { G A E } }$ follow Equation (14) on the tree of a task’s episodes, whose edges are executed (state, response) pairs; the ordinary PPO rows use their whitened values ${ \hat { A } } _ { t }$ R is the episode’s final reward, 0 also for an episode cut at the 40-step limit, without bootstrapping. If $s _ { t + 1 }$ has a single outgoing edge, $\overline { { A } } ^ { \mathrm { G A E } } ( s _ { t + 1 } ) \overline { { \mathbf { \Omega } } } = A _ { t + 1 } ^ { \mathrm { G A E } }$ and Equation (14) is the usual GAE recursion $\begin{array} { r } { \dot { A } _ { t } ^ { \mathrm { G } \mathrm { \bar { A } E } } = \delta _ { t } + \lambda A _ { t + 1 } ^ { \mathrm { G } \mathrm { \check { A } E } } } \end{array}$ with $\delta _ { t } = V _ { \psi } ( s _ { t + 1 } ) - V _ { \psi } ( s _ { t } )$

Two executed actions at the same state. For two executed actions with non-terminal successors $s _ { t + 1 }$ and $s _ { t + 1 } ^ { \prime }$ , Equation (14) gives the difference between their raw advantages,

$$
\left[ V _ { \psi } ( s _ { t + 1 } ) - V _ { \psi } ( s _ { t + 1 } ^ { \prime } ) \right] + \lambda \bigl [ \overline { { { A } } } ^ { \mathrm { G A E } } ( s _ { t + 1 } ) - \overline { { { A } } } ^ { \mathrm { G A E } } ( s _ { t + 1 } ^ { \prime } ) \bigr ] .\tag{18}
$$

The first term compares the successor values, and the second carries feedback from their continuations.

A terminated path. Let the actions at $s _ { t } , \ldots , s _ { t + L - 1 }$ lead to termination after the last of them, with reward R at that step and zero before, and write $V _ { j } = V _ { \psi } ( s _ { t + j } )$ . Then

$$
\begin{array} { l } { { \displaystyle { \cal A } _ { t } ^ { \mathrm { G A E } } = ( V _ { 1 } - V _ { 0 } ) + \lambda ( V _ { 2 } - V _ { 1 } ) + \dots + \lambda ^ { L - 2 } ( V _ { L - 1 } - V _ { L - 2 } ) + \lambda ^ { L - 1 } ( R - V _ { L - 1 } ) } } \\ { { \displaystyle ~ = - V _ { 0 } + ( 1 - \lambda ) \sum _ { j = 1 } ^ { L - 1 } \lambda ^ { j - 1 } V _ { j } + \lambda ^ { L - 1 } R . } } \end{array}\tag{19}
$$

The executed continuation, summarized by a λ-weighted average of the critic’s values along it and the final reward, is compared with the value of the state in which the action was taken. For $L = 1$ the sum is empty and $A _ { t } ^ { \mathrm { G A E } } = R - V _ { 0 } ; { \mathrm { f o r } } \lambda = 1$ the intermediate values cancel and $A _ { t } ^ { \mathrm { G A E } } = R - V _ { 0 }$ , the Monte-Carlo return minus the baseline. The coefficients of the intermediate values sum to $1 - \lambda ^ { L - 1 }$ so with $\lambda = 0 . 9 5$ and $L = 3 6$ the final reward has weight 0.17 and the values along the path 0.83 together, although each single value has a small weight. This follows directly from the definition of GAE (Schulman et al., 2016).

GAE and candidate comparisons. With the true policy value function and an on-policy continuation, the conditional expectation of $A _ { t } ^ { \mathrm { G A E } }$ is the executed action’s true advantage. How informative a sampled estimate is depends on the variances and covariances of the critic’s errors and of the outcome, rather than on the coefficients alone. PPO already provides step-level feedback through its critic and GAE, using state values and returns along the executed paths. The single-path expression in Equation (19) compares the executed continuation with $V _ { 0 } = V _ { \psi } ( s _ { t } )$ . WM-PPO retains these advantages and extends evaluation to other candidates at the same state through their predicted successors. Their transition values supply direct preferences for the auxiliary policy update. In the policy-channel-only variant, these preferences are the additional policy feedback; critic training and ordinary PPO advantages remain standard.

Equal terminal returns. In Wide-Tree-PPO on ScienceWorld, all executed candidates have the same terminal return at 82% of its branching states. Those observed outcomes provide no preference among the candidates, although their immediate consequences can differ in whether they advance the task. Appendix G gives a training example in which both continuations succeed but the transition values favor completing the task’s prerequisite first. WM-PPO (policy channel only) outperforms this branch-only control in both environments (Section 5.2).

Value and advantage scales. On candidate pairs whose estimated true values differ by at least 0.4, the critics’ value differences average 0.02–0.10 in absolute value (Appendix F.4), and the raw advantages of PPO-160 have a median standard deviation of 0.22–0.23 per iteration on ScienceWorld and 0.27–0.30 on ALFWorld before whitening. These numbers indicate the scales involved; they are sampled and normalized differently.

## B.3 Settings and components

Table 5 lists the hyperparameters. Table 4 identifies each component’s data, what it learns or computes, and how its output is used.

Table 4: The components of WM-PPO and their data. The three models are LoRA adapters on one frozen base model; the gate has no learned parameters. Only real transitions train the world model and supply returns; predictions enter learning only through transition values, and only the policy’s own loss sends gradients to the policy.
<table><tr><td>Component</td><td>Learns or computes</td><td>From</td><td>Used for</td></tr><tr><td>Policy πθ</td><td>action distribution given the state</td><td>ordinary PPO rows of real executed steps (tree GAE) and auxiliary rows of</td><td>acting; alone at test time</td></tr><tr><td>World model  $p _ { \phi }$ </td><td>next observation after an action</td><td>evaluated candidates cross-entropy on real executed transitions of chains and branches, every</td><td>predicting the consequences of the K candidates</td></tr><tr><td>Critic  $V _ { \psi }$ </td><td>expected task return from a state</td><td>iteration Monte-Carlo returns of the trees, the sibling-centered term, and, at each evaluated state st, the mean transition value āt of its candidates</td><td>GAE of real steps; transition values  ${ \hat { q } } _ { t } ( a )$  of predicted successors</td></tr><tr><td>Gate</td><td>weight we per margin bin</td><td>(value channel) real branch pairs of the previous M iterations</td><td>scaling the auxiliary advantages and the value-channel loss</td></tr></table>

Task sets. ScienceWorld training draws up to 120 official training variations per task type (1,228 of 2,106) in a fixed shuffled order with equal weight per task type. The evaluation set is fixed. Within each of the development and test splits, it contains all variations of the seven task types that have few variations and nine evenly spaced variations of every other type: 140 of the 1,053 development and 144 of the 1,060 test variations. Three settings of WM-PPO are environment-specific (Table 5): the prompts, the world model’s context and output length, and the rule for predicted endings, which applies only where observations report the end of an episode. Among the baselines’ settings, only those of CAPO, SO-PPO and PaW differ between the environments, because their ALFWorld settings diverged or lost the action format on ScienceWorld (Appendix D).

Table 5: Hyperparameters of WM-PPO (identical in both environments unless noted).
<table><tr><td>Base model Adapters Critic head</td><td>Qwen3-4B-Instruct-2507, non-thinking mode, frozen policy, critic, world model: separate LoRA, rank 64,  $\alpha = 1 2 8 ,$  all linear layers separate LoRA with a linear value head on the last prompt token, zero- initialized; inputs left-truncated to 8,192 tokens</td></tr><tr><td>Rollouts Tasks per iteration Candidates</td><td>temperature 1, at most 256 response tokens, at most 40 actions per episode 64; one chain and three branches each  $K = 4$  per chain step, first one executed</td></tr><tr><td>Branches Policy optimizer</td><td>branch step uniform over the chain; up to 8 samples to obtain a different action  $\mathrm { A d a m W } , \mathrm { l r } 1 0 ^ { - 5 }$  , betas (0.9, 0.99), gradient clipping 1.0, 64 rows per mini-</td></tr><tr><td>PPO clip</td><td>batch, one epoch  $\epsilon _ { \mathrm { l o } } = 0 . 2 , \epsilon _ { \mathrm { h i } } = 0 . 2 8 ;$  token-level normalization per minibatch</td></tr><tr><td>Advantages</td><td>tree GAE (Appendix B.2),  $\gamma = 1 , \lambda = 0 . 9 5$  , no bootstrap at the step limit, whitened over executed steps</td></tr><tr><td>Critic optimizer</td><td> $\mathrm { A d a m w , l r \thinspace 5 \times 1 0 ^ { - 5 } ; }$  target: mean final reward below the node; sibling term</td></tr><tr><td>Auxiliary rows</td><td>weight 1; value-channel weight  $\beta = 0 . 5$  (Equation (12)) Eq. (8), ν = 1; decision points with  $\geq 2$  distinct candidates; exact ties carry</td></tr><tr><td></td><td>no preference (checked in ScienceWorld; none occurred in ALFWorld)</td></tr><tr><td>Margin bins Reliability gate</td><td>0.5 and 0.8 quantiles of the iteration&#x27;s nonzero margins mt pairs of the previous M = 4 iterations;  $n _ { \mathrm { m i n } } = 2 0$  pairs per bin;  $n _ { 0 } = 4 0 ,$ </td></tr><tr><td></td><td> $g _ { \mathrm { r e f } } = 0 . 0 5$  (a shrinkage estimate of the bin&#x27;s gain, Equation (17), scaled so that a shrunken gain of 0.05 per decision earns full weight; set once from the</td></tr><tr><td></td><td>first branch pairs of a preliminary ALFWorld run and used unchanged in both environments)</td></tr><tr><td>Gate, iterations 13–14</td><td>iteration 13: no stored pairs, all  $w _ { \ell } = 0 ;$  iteration 14:  $w _ { \ell } = \operatorname* { m a x } ( g _ { \ell } -$ </td></tr><tr><td></td><td> $1 . 6 4 \mathrm { s e } _ { \ell } , 0 ) /$  maxl/ max  $( g _ { \ell ^ { \prime } } - 1 . 6 4 \mathrm { s e } _ { \ell ^ { \prime } } , 0 )$  ; bins with fewer than  $n _ { \mathrm { m i n } }$  count as 0, and if no bound is positive all weights are 0 (in 9 of the 12 runs of</td></tr><tr><td></td><td>the two WM-PPO variants all iteration-14 weights were 0; in the other three</td></tr><tr><td>World-model input</td><td>one or two bins opened), the rule of the preliminary run, kept for comparability goal, history (ALFWorld: all previous observations and actions; ScienceWorld:</td></tr><tr><td></td><td>the last 8 actions, without observations),</td></tr><tr><td></td><td>current observation, admissible actions or referable objects, the action the next observation as the environment prints it; greedy decoding</td></tr><tr><td>World-model output</td><td>(at most 300 tokens in ALFWorld, 1,024 in ScienceWorld)</td></tr><tr><td>Initial world model</td><td>executed transitions of warm-up iterations 1–12 (ALFWorld 29,030, Science- World 10,291), trained at the end of iteration  $1 2 ; \mathrm { ~ l r ~ } 1 0 ^ { - 4 }$  , one epoch, 32</td></tr><tr><td></td><td>transitions per step.</td></tr><tr><td>Online world model</td><td>this iteration&#x27;s executed transitions; lr  $5 \times 1 0 ^ { - 5 }$  , one epoch, continues from</td></tr><tr><td></td><td>the current model</td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td>ScienceWorld: predicted completion or failure replaces the critic&#x27;s value by 1</td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td>Predicted endings</td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td>or 0</td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td>Evaluation</td><td></td></tr></table>

## C Diagnostic experiments

The diagnostics in Section 2 test whether the critic can compare actions by valuing their real successors, whether predictions preserve these comparisons, and whether the policy can learn from them. All use the policy and critic of the PPO run continued from the common warm-up to iteration 24 on ALFWorld (Section 2), the 274 evaluation games and per-game paired tests.

Selection with true next states. At every step, the policy samples four candidates at temperature 1. Each distinct candidate is executed for one step in a copy of the game. The critic values the history extended by the resulting observation and admissible-action list, and the candidate with the highest value is executed in the real game. In the peek control, a candidate that completes the task is taken if one exists; otherwise the majority vote decides.

Pairwise test. In each evaluation game, one state of the policy’s own episode is drawn at random; the draw is repeated with three random seeds. Every distinct candidate is executed and followed by three independent continuations of the same policy, and the fraction of successes estimates its value. For the 195 candidate pairs with different estimates, this order is compared with the order of the critic’s values of their true next states.

World model for the diagnostic. The policy played 1,000 training games at temperature 1, executing the first of four samples; every distinct candidate was also executed for one step in a copy of the game, which gives 48k transitions, 44% of them for unexecuted candidates. The world model is a LoRA adapter on the base model, trained for one epoch with the next-observation loss. Accuracy is measured on 1,500 random transitions of the evaluation games, by category of the true next observa tion: observations determined by information already seen (71% of the transitions; 99.7% predicted exactly), observations revealing objects not seen before (20%; 26% predicted exactly, object F1 0.41), and “Nothing happens” (9%). Choosing with the true observation from which newly revealed objects are removed gives 59.1%, so hidden content affects selection. Where it matters, the world model still provides a prior: it estimates whether the target object is present with an AUC of 0.81 (untrained model: 0.62).

Offline update. The policy played 500 training games that are not in the world model’s data. At the 8,042 decision points with at least two distinct candidates, the label is the candidate whose predicted successor the critic values highest. One epoch of PPO-clip (lr 10<sup>−5</sup>, about 32k samples) uses the advantages of Equation (4) and no task reward. The KL divergence of the updated policy from the original is 0.017 with world-model labels and 0.033 with majority-vote labels. The evaluation samples at temperature 1, twice per game, without the world model.

Inference-time lookahead on trained policies. Table 6 applies the selection test of Section 2.1 to trained ALFWorld policies on the 274 evaluation games. At every step, the policy samples four responses at temperature 1. pass@1 executes the first, which is the standard evaluation (three runs). Majority of4 executes the most frequent action. True next state executes each distinct candidate for one step in a copy of the game, values each extended history with the policy’s own critic of the same iteration, and executes the candidate with the highest value. Predicted next state does the same with the next observation predicted greedily by the world model of the diagnostic in Section 2.2, which is not updated during RL. Each lookahead rule has two runs, and differences are paired per game against pass@1.

Table 6: Inference-time lookahead on trained ALFWorld policies (success in $\% ;$ pass@1: the policy’s own first sample; in parentheses, the change against pass@1 in percentage points; <sup>†</sup>: $p < 0 . 0 5 )$ . One training run per policy (the first run of each method).
<table><tr><td>It.</td><td>Policy</td><td>pass@1</td><td></td><td>Majority of 4 True next state</td><td>Predicted next state</td></tr><tr><td></td><td>24 PPO-160</td><td>68.9</td><td>68.8 (-0.1)</td><td> $7 4 . 6 ( + 5 . 8 ^ { \dagger } )$ </td><td>73.9 (+5.0†)</td></tr><tr><td></td><td>PPO-256</td><td>72.6</td><td>69.7 (-2.9)</td><td> $8 1 . 0 ( + 8 . 4 ^ { \dagger } )$ </td><td> $7 7 . 2 ( + 4 . 6 ^ { \dagger } )$ </td></tr><tr><td></td><td>Wide-Tree-PPO</td><td>64.0</td><td>59.5 (-4.5†)</td><td>79.0 (+15.0†)</td><td>76.1 (+12.1†)</td></tr><tr><td>36</td><td>PPO-160</td><td>77.4</td><td>76.6 (-0.7)</td><td>84.9 (+7.5†)</td><td>82.3 (+4.9†)</td></tr><tr><td></td><td>PPO-256</td><td>86.1</td><td>85.4 (-0.7)</td><td>90.1 (+4.0†)</td><td>89.2 (+3.1†)</td></tr><tr><td></td><td>Wide-Tree-PPO</td><td>83.8</td><td>83.0 (-0.8)</td><td>91.1 (+7.2†)</td><td>86.9 (+3.0†)</td></tr><tr><td>55</td><td>PPO-160</td><td>89.8</td><td>88.7 (-1.1)</td><td>94.0 (+4.2†)</td><td>92.5 (+2.7†)</td></tr><tr><td></td><td>PPO-256</td><td>91.0</td><td>90.7 (-0.3)</td><td>94.9 (+3.9†)</td><td>93.2 (+2.3†)</td></tr><tr><td></td><td>Wide-Tree-PPO</td><td>89.2</td><td>89.2 (+0.1)</td><td>94.2 (+5.0†)</td><td>93.6 (+4.4†)</td></tr></table>

## D Baselines

## D.1 Budgets, methods and tuning

Budgets. PPO-160: 160 tasks × 1 episode per iteration; PPO-256: 256 tasks × 1; Wide-Tree-PPO, Q-PPO and Fixed WM: WM-PPO’s trees (64 tasks × (1 chain + 3 branches)); group methods: 20 tasks × 8 episodes, also 40 tasks × 4 for GRPO and, on ScienceWorld, GiGPO (Tree-GRPO: WM-PPO’s trees or 32 trees of 8 leaves, Table 7); PPO refinements and the auxiliary-feedback methods: PPO-160’s sampling. All start from the warm-up checkpoint with the same task order.

Methods. Wide-Tree-PPO: WM-PPO (policy channel only) without the auxiliary term, that is, the same trees, tree GAE, critic targets and sibling term, and the same ordinary PPO rows of the executed steps. Q-PPO: a third LoRA adapter with a linear head reads the history followed by a candidate response and is regressed on the return below each executed step, plus the sibling term; it is initialized from the warm-up critic and replaces $\hat { q }$ in Equation (6); candidates, margins, gate and auxiliary rows are unchanged. Fixed WM: WM-PPO (policy channel only) with a world model that is never updated after its initialization; the policy and the critic are trained, and the gate and the auxiliary rows computed, as in that variant. In both environments it starts from the same initial world model as WM-PPO, trained on the warm-up’s executed transitions (on ScienceWorld, predictions at iteration 13 are token-identical for 78 of 81 shared state–candidate pairs). CAPO (Wang et al., 2026): action-level critic and importance ratio. SO-PPO (Li et al., 2026): one geometric-mean importance ratio per turn with clipping-triggered normalization. Turn-ratio PPO: the GSPO-style turn-level ratio (Zheng et al., 2025) with a narrow clip, as implemented in the CAPO code base (Wang et al., 2026). VAPO-lite (Yue et al., 2025): PPO-160 plus a language-model loss (weight 0.1) on responses of successful episodes. GiGPO (Feng et al., 2025) and HGPO (He et al., 2026): step-level grouping by identical environment states and by history, ported from their released code. Tree-GRPO (Ji et al., 2026): tree-based group advantages on WM-PPO’s trees. Majority-vote labels: PPO-160 plus four samples per step and auxiliary rows as in Equation (8) with the unique plurality action as the preferred one, weight 1, no gate; computed on PPO-160’s episodes, their total auxiliary weight is about two to three times that of WM-PPO. Standard OPD (Agarwal et al., 2024): a frozen Qwen3-32B in non-thinking mode, which shares the student’s vocabulary, computes the log-probabilities of the executed response given exactly the student’s prompt; every token’s advantage is the whitened GAE advantage plus clip(log π<sub>T</sub> − log π<sub>old</sub>, ±5), the reverse-KL reward of on-policy distillation, with coefficient 1. OCSD (Yang et al., 2026): the teacher, the policy of the previous iteration (self) or the same Qwen3-32B, computes the log-probability of the executed response with and without the next one or two observations of the episode, but not the current action; the PPO advantage is multiplied by 1 + 0.5 sign(A) tanh $( \Delta / 2 )$ , where $\Delta$ is the difference of the two log-probabilities, on the 20% least likely steps per episode, following the released implementation. All OPD and OCSD variants are built on PPO-160 and use its interaction. PaW (Lu et al., 2026): PPO-160 plus PaW’s auxiliary next-observation loss on the policy with the published settings: the clipped loss with $\rho = 0 . 2$ on the tokens of the true next observation, applied to the 75% of transitions with the highest action uncertainty, the reward-adaptive weight following the paper’s formula (with the batch’s mean episode return under PPO), and the paper’s 20 prediction prompts. Unlike the original full fine-tuning at learning rate $1 0 ^ { - 6 }$ , it trains PPO-160’s LoRA at 10<sup>−5</sup>.

Published results. Several of these methods report higher ALFWorld success in their own papers. Those results come from different base models, longer training and other budgets. Here, every method shares one base model, one warm-up checkpoint, one task order and a comparable interaction budget, and is reported at fixed iterations.

Tuning rules fixed in advance. With the published clipping ranges, CAPO (0.001) clipped about 97% of tokens and Turn-ratio PPO $( 3 \times 1 0 ^ { - 4 } )$ about 85% on ALFWorld, so both barely learned (Table 7). For each of the two, three larger clipping values were tried for one update on the iteration-13 data. The rule chose the smallest value whose clipped-token fraction was at most 20%, or the largest value if none qualified: 0.3 for CAPO and 0.1 for Turn-ratio PPO. On ScienceWorld every method first used its ALFWorld setting unchanged, and only a setting that diverged or lost the action format was re-tuned (CAPO, SO-PPO and PaW); WM-PPO keeps its ALFWorld hyperparameters unchanged. Table 8 lists the ScienceWorld settings.

SO-PPO on ScienceWorld. On ScienceWorld, SO-PPO with its published clip of 0.2 had a periteration KL of 0.07–0.14 from the first update (PPO-160: 0.01–0.02) and collapsed after iteration 36 (15.8% at iteration 60). The rule fixed beforehand was to take the largest of 0.1, 0.05 and 0.02 whose KL after one update on the iteration-13 data was at most 0.02, or the smallest if none qualified. All three exceeded the threshold slightly (0.054, 0.035 and 0.022), so 0.02 was used. For SO-PPO we also ran the clip of 0.1 chosen by the clipped-token rule of CAPO and Turn-ratio PPO. This run did not diverge (per-update KL at most 0.08), but its success peaked at 22.9% among the evaluated checkpoints (iteration 36) and fell to 4.2% at iteration 60, when 92% of its evaluation episodes ended at the step limit. The main table reports the best of the three settings, clip 0.02. The same SO-PPO with clip 0.2 did not collapse on ALFWorld and is reported with the published value there.

CAPO on ScienceWorld. CAPO with its ALFWorld clip of 0.3 diverged on ScienceWorld at iteration 14, and with 0.1 at iteration 28 (per-update KL spikes of 0.15–0.33, zero success); the last value of the pre-set grid, 0.03, diverged at iteration 18. No setting of the grid remains, so the main table reports, for CAPO on ScienceWorld, the official evaluation of clip 0.1 at iteration 18, the checkpoint with the highest greedy success in the evaluations during training (21.5%; 20.1% at iteration 15, 16.9% at 21 and 9.5% at 24), without a test against WM-PPO. In the official evaluation it reaches 21.1% greedy and 20.0% pass@1 success, after 12.8k real steps; the neighboring checkpoints reach 19.0% (iteration 15) and 19.4% (iteration 21) greedy, so the choice does not depend on which evaluation is used.

Turn-ratio PPO and PaW on ScienceWorld. Turn-ratio PPO runs on ScienceWorld with its ALFWorld clip of 0.1. It did not diverge (per-update KL of 0.02–0.06, three to five times that of PPO-160), but its success stayed near 25% from iteration 21 on, below PPO-160; because only diverged settings were re-tuned, it keeps this setting. PaW with its published coefficient of 1.0 kept a per-update KL close to that of PPO-160 on ScienceWorld but gradually stopped writing the action tag: the share of training responses without it rose from 0.8% at iteration 16 to 78% at iteration 20, and 67% of its greedy evaluation episodes at iteration 60 end at the step limit; with the coefficients 0.5 and 0.25, the share stays at about 1%, as for PPO-160. Before any result of the two runs, we fixed the coefficients 0.5 and 0.25 and the rule that the main table reports the best of the three settings by greedy success at iteration 60, which is 0.5. On ALFWorld, PaW uses the published coefficient.

Table 7: Baseline settings tried on ALFWorld (iteration 36, policy-only, greedy and sampling success in %). The main table uses the setting chosen by the rule.
<table><tr><td>Baseline</td><td>Setting</td><td>Greedy</td><td>pass@1</td></tr><tr><td rowspan="2">CAPO</td><td>action-ratio clip 0.001 (original)</td><td>46.0</td><td>63.9</td></tr><tr><td>clip 0.1</td><td>82.8</td><td>84.1</td></tr><tr><td rowspan="3">Turn-ratio PPO</td><td>clip 0.3 (selected by the pre-set rule; main table)</td><td>87.2</td><td>86.7</td></tr><tr><td>clip  $3 \times 1 0 ^ { - 4 }$  (released code)</td><td>56.6</td><td>74.6</td></tr><tr><td>clip 0.1 (selected)</td><td>85.8</td><td>85.0</td></tr><tr><td rowspan="2">GRPO</td><td>40 tasks × 4</td><td>54.4</td><td>60.0</td></tr><tr><td>20 tasks × 8 (main table)</td><td>59.5</td><td>67.8</td></tr><tr><td rowspan="2">Tree-GRPO</td><td>32 trees × 8 leaves</td><td>47.4</td><td>60.2</td></tr><tr><td>64 trees × 4 leaves</td><td>48.9</td><td>59.6</td></tr></table>

Table 8: Baseline settings on ScienceWorld (policy-only success in %; iteration 60 unless stated). Every method first used its ALFWorld setting; only settings that diverged or lost the action format were re-tuned by the rules fixed in advance.
<table><tr><td>Baseline</td><td>Setting</td><td>Outcome</td><td>Greedy</td><td>pass@1</td></tr><tr><td rowspan="2">CAPO</td><td>clip 0.3 (ALFWorld setting)</td><td>diverged, it. 14</td><td></td><td></td></tr><tr><td>clip 0.1</td><td>diverged, it. 28; it. 18 in Table 2</td><td>21.1</td><td>20.0</td></tr><tr><td rowspan="4">SO-PPO</td><td>clip 0.03</td><td>diverged, it. 18</td><td></td><td></td></tr><tr><td>clip 0.2 (published)</td><td>collapsed after it. 36</td><td>15.8</td><td>16.2</td></tr><tr><td>clip 0.02 (KL rule; Table 2)</td><td>it. 60</td><td>46.5</td><td>49.4</td></tr><tr><td>clip 0.1 (clipped-token rule)</td><td>no divergence; it. 60 (best 22.9%, it. 36)</td><td>4.2</td><td>10.8</td></tr><tr><td rowspan="4">PaW</td><td>Turn-ratio PPO clip 0.1 (ALFWorld setting)</td><td>it. 60</td><td>25.0</td><td>26.5</td></tr><tr><td>coefficient 1.0 (published)</td><td>action tags lost from it. 20</td><td>6.0</td><td>30.8</td></tr><tr><td>coefficient 0.5 (Table 2)</td><td>it. 60</td><td>35.6</td><td>38.8</td></tr><tr><td>coefficient 0.25</td><td>it. 60</td><td>32.4</td><td>32.9</td></tr></table>

## D.2 Teacher-based feedback

How strong is the teacher? Table 9 evaluates candidate teachers as agents in the student’s prompt and action format on the evaluation games. Under greedy decoding and strict parsing of the <action> tag, Qwen3-32B solves 13.5% of the ALFWorld games and 26.4% of the Science-World games. On ALFWorld it often writes its action in its own tool-call format; counting those as actions raises its success to 48.5%. On ScienceWorld, in non-thinking mode, it tends to act without reasoning and to focus early on a wrong object. Qwen3-30B-A3B-Instruct, a second candidate, performs at the same level. The teacher was fixed before these evaluations, which serve only to report its level; no available model of this family was clearly stronger than the trained student, so we did not add a second external teacher.

What OCSD’s judgment tracks. For every step that OCSD selected during the first self-teacher run, we averaged tanh $( \Delta / 2 )$ over the response tokens, which is OCSD’s judgment of the step, and compared it with labels taken from the true next observation. On ALFWorld (10,019 steps), this judgment separates valid actions from those answered by Nothing happens with an AUC of 0.80, and the adjustment concentrates on the action tokens. On ScienceWorld (8,055 steps), the same judgment is inverted: its AUC is 0.25 for valid against rejected commands and 0.21 for steps that raised the game’s score against steps that did not, and the strongest negative adjustments fall on the reasoning tokens of steps that made progress, which are followed by observations with much new content. The adjustment is bounded and keeps the sign of the advantage; on ScienceWorld, the resulting policy is level with PPO-160 for pass@1.

Table 9: Candidate teachers as agents (success in $\% ,$ evaluation games, student’s prompt format).
<table><tr><td>Model</td><td>ALFWorld</td><td>ScienceWorld</td></tr><tr><td>Student at the common start (iteration 12)</td><td>27.0</td><td>20.8</td></tr><tr><td>WM-PPO after training (iteration 36 / 60, greedy)</td><td>87.5</td><td>52.9</td></tr><tr><td>Qwen3-32B, non-thinking, greedy (the teacher used)</td><td>13.5</td><td>26.4</td></tr><tr><td>sampled at temperature 1, three runs</td><td>35.0 / 34.0 / 33.2</td><td></td></tr><tr><td>tool-call format also accepted as an action</td><td>48.5</td><td></td></tr><tr><td>Qwen3-30B-A3B-Instruct-2507, greedy</td><td>28.8</td><td>25.4</td></tr></table>

Standard OPD in training. On ALFWorld, the student’s responses changed abruptly at the start. The KL divergence from the previous policy was 1.33 in iteration 13; in iteration 14, only 0.5% of the responses contained a parsable action; and the format recovered in iteration 15. The teacher prefers its own tool-call format and suppressed the student’s action tags. On ScienceWorld, the teacher penalized the student’s reasoning heavily (about 63 nats per response early on), and the median response length fell from 77 tokens to 12 by iteration 16. The student led PPO-160 at iteration 36 and then stayed at 31–34% pass@1; the teacher itself reaches 26.4% as a greedy agent (Table 9).

## E Complete results

This appendix reports results for every method at every evaluated iteration (Appendix E.1), the learning curves and the real steps to reference success levels (Appendix E.2), the individual runs, the ScienceWorld task types and the costs per iteration. Marks are as in Table 2; a number in parentheses after a method is its number of runs.

## E.1 Results at every evaluated iteration

Tables 10 and 12 list every method at every evaluated iteration, with the real steps up to that iteration and pass@3. On ALFWorld, the methods trained to iteration 55 approach the ceiling of the benchmark. There, WM-PPO succeeds on 88.7% (greedy) and 90.8% (pass@1) and WM-PPO (policy channel only) on 91.4% and 92.9%; PPO-160 reaches 85.0% and 87.3%, PPO-256, with about 1.5 times the real steps, 90.8% and 91.1%, Wide-Tree-PPO (three runs) 87.3% and 90.6%, Fixed WM 93.4% and 92.2%, and PaW 93.1% and 91.8%. With this much training, the differences on ALFWorld lie in how quickly each method gets there (Figure 13, Table 14) rather than in the final level. GRPO with 40 tasks of 4 episodes per iteration, which sees about twice as many tasks as with 20 tasks of 8 episodes, also stays below WM-PPO in both environments, and so does GiGPO on ScienceWorld. On ScienceWorld at iteration 60, SO-PPO has a slightly higher pass@3 than WM-PPO (60.9% against 59.6%).

Table 10: ALFWorld, all methods, iterations 24 and 36 (policy-only success in %; iteration 55 in Table 11). <sup>†</sup>/<sup>‡</sup>: significantly lower/higher than WM-PPO at that iteration (per-game paired test, $p < 0 . 0 5$ , uncorrected). <sup>s</sup>Second run of an existing seed, launched later with the same configuration; every mean and test in the paper uses the first run of each seed.
<table><tr><td rowspan="2">Method</td><td colspan="4">iteration 24</td><td colspan="4">iteration 36</td></tr><tr><td></td><td>Steps Greedy pass@1</td><td></td><td>pass@3</td><td>Steps Greedy pass@1 pass@3</td><td></td><td></td><td></td></tr><tr><td>WM-PPO (ours) (3)</td><td>60k</td><td>55.7</td><td>60.8</td><td>73.5</td><td>102k</td><td>87.5</td><td>87.8</td><td>93.4</td></tr><tr><td>WM-PPO (policy channel only) (3)</td><td>58k</td><td>68.4</td><td>70.9</td><td>82.5</td><td>97k</td><td>88.4</td><td>89.4</td><td>94.6</td></tr><tr><td>PPO-160 (3)</td><td>57k</td><td>47.1†</td><td>54.3†</td><td>67.9</td><td>105k</td><td>63.5†</td><td>71.0†</td><td>81.1</td></tr><tr><td>PPO-256 (3)</td><td>88k</td><td>63.6</td><td>69.3</td><td>81.9</td><td>150k</td><td>83.1†</td><td>84.8†</td><td>92.9</td></tr><tr><td>GRPO</td><td>58k</td><td>53.3</td><td>57.2†</td><td>73.0</td><td>113k</td><td>59.5†</td><td>67.8†</td><td>83.6</td></tr><tr><td>GRPO, 40 tasks × 4</td><td>60k</td><td>36.5†</td><td>42.6†</td><td>56.6</td><td>113k</td><td>54.4†</td><td>60.0†</td><td>71.2</td></tr><tr><td>GiGPO</td><td>60k</td><td>44.9†</td><td>54.4†</td><td>65.7</td><td>109k</td><td>78.1†</td><td>81.0†</td><td>90.9</td></tr><tr><td>HGPO</td><td>56k</td><td>52.6</td><td>59.4</td><td>68.2</td><td>103k</td><td>65.0†</td><td>70.8†</td><td>81.0</td></tr><tr><td>Tree-GRPO</td><td>66k</td><td>43.4†</td><td>47.8†</td><td>69.0</td><td>124k</td><td>48.9†</td><td>59.6†</td><td>74.8</td></tr><tr><td>CAPO</td><td>54k</td><td>78.5</td><td>80.5</td><td>87.6</td><td>88k</td><td>87.2</td><td>86.7</td><td>92.0</td></tr><tr><td>SO-PPO</td><td>55k</td><td>74.8</td><td>74.0</td><td>89.8</td><td>100k</td><td>86.5</td><td>83.1†</td><td>89.8</td></tr><tr><td>Turn-ratio PPO</td><td>54k</td><td>71.5</td><td>74.6</td><td>88.3</td><td>91k</td><td>85.8</td><td>85.0</td><td>92.7</td></tr><tr><td>VAPO-lite</td><td>56k</td><td>65.0</td><td>66.8</td><td>80.7</td><td>100k</td><td>68.2†</td><td>80.2†</td><td>90.9</td></tr><tr><td>Q-PPO (3)</td><td>60k</td><td>63.7</td><td>68.3</td><td>81.1</td><td>100k</td><td>84.4†</td><td>88.6</td><td>94.3</td></tr><tr><td>Wide-Tree-PPO (3)</td><td>61k</td><td>54.0</td><td>63.3</td><td>76.5</td><td>106k</td><td>78.5†</td><td>82.3†</td><td>90.6</td></tr><tr><td>Fixed WM</td><td>59k</td><td>67.2</td><td>67.4</td><td>77.0</td><td>99k</td><td>76.3†</td><td>79.4†</td><td>87.2</td></tr><tr><td>PaW (coefficient 1.0)</td><td>52k</td><td>68.6</td><td>70.6</td><td>81.8</td><td>89k</td><td>85.4</td><td>87.3</td><td>92.7</td></tr><tr><td>Majority-vote labels</td><td>65k</td><td>22.6†</td><td>24.1†</td><td>28.8</td><td>132k</td><td>17.2†</td><td>17.6†</td><td>20.8</td></tr><tr><td>OCSD (self teacher) (3)</td><td>54k</td><td>60.6</td><td>64.0</td><td>81.0</td><td>96k</td><td>83.0†</td><td>82.7†</td><td>93.6</td></tr><tr><td>OCSD (32B teacher)</td><td>56k</td><td>66.1</td><td>67.2</td><td>82.1</td><td>99k</td><td>83.2</td><td>70.1†</td><td>88.7</td></tr><tr><td>OPD (32B teacher)</td><td>58k</td><td>43.4†</td><td>51.6†</td><td>64.6</td><td>112k</td><td>41.2†</td><td>51.0†</td><td>64.2</td></tr><tr><td>WM-PPO, second run of seed 2⁸</td><td>59k</td><td>55.8</td><td>61.1</td><td>74.8</td><td>102k</td><td>85.8</td><td>86.7</td><td>92.0</td></tr></table>

Table 11: ALFWorld, the methods trained to iteration 55 (continued from Table 10; same marks).
<table><tr><td></td><td colspan="4">iteration 55</td></tr><tr><td>Method</td><td>Steps</td><td></td><td>Greedy pass@1</td><td>pass@3</td></tr><tr><td>WM-PPO (ours) (3)</td><td>153k</td><td>88.7</td><td>90.8</td><td>95.4</td></tr><tr><td>WM-PPO (policy channel only) (3) 147k</td><td></td><td>91.4‡</td><td>92.9</td><td>96.5</td></tr><tr><td>PPO-160 (3)</td><td>161k</td><td>85.0†</td><td>87.3†</td><td>93.4</td></tr><tr><td>PPO-256 (3)</td><td>226k</td><td>90.8</td><td>91.1</td><td>95.4</td></tr><tr><td>Wide-Tree-PPO (3)</td><td>165k</td><td>87.3</td><td>90.6</td><td>96.1</td></tr><tr><td>Fixed WM</td><td>153k</td><td>93.4</td><td>92.2</td><td>96.0</td></tr><tr><td>PaW (coefficient 1.0)</td><td>136k</td><td>93.1‡</td><td>91.8</td><td>95.6</td></tr><tr><td>WM-PPO, second run of seed 2⁸</td><td>152k</td><td>90.1</td><td>91.2</td><td>93.8</td></tr></table>

Table 12: ScienceWorld, all methods, iterations 24 and 36 (policy-only success in %; iterations 48 and 60 in Table 13). <sup>s</sup>Second runs of an existing seed, launched later with the same configuration; every mean and test in the paper uses the first run of each seed, and these rows are listed for completeness, without tests. <sup>†</sup>/<sup>‡</sup>: significantly lower/higher than WM-PPO at that iteration (per-game paired test, p < 0.05, uncorrected).
<table><tr><td rowspan="2">Method</td><td colspan="4">iteration 24</td><td colspan="4">iteration 36</td></tr><tr><td></td><td>Steps Greedy pass@1</td><td></td><td>pass@3</td><td>Steps</td><td>Greedy pass@1</td><td></td><td>pass@3</td></tr><tr><td>WM-PPO (ours) (3)</td><td>34k</td><td>28.5</td><td>27.7</td><td>35.9</td><td>67k</td><td>38.0</td><td>38.0</td><td>45.8</td></tr><tr><td>WM-PPO (policy channel only) (3)</td><td>32k</td><td>26.6</td><td>27.2</td><td>35.2</td><td>67k</td><td>33.0†</td><td>32.9†</td><td>42.3</td></tr><tr><td>WM-PPO, terminal-only value target</td><td>31k</td><td>29.9</td><td>31.0‡</td><td>37.7</td><td>64k</td><td>34.5†</td><td>38.0</td><td>45.8</td></tr><tr><td>PPO-160 (3)</td><td>25k</td><td>26.5</td><td>26.2†</td><td>33.2</td><td>54k</td><td>32.0†</td><td>31.5†</td><td>39.1</td></tr><tr><td>PPO-256 (3)</td><td>38k</td><td>26.8</td><td>26.5</td><td>34.7</td><td>75k</td><td>33.7†</td><td>32.3†</td><td>40.1</td></tr><tr><td>GRPO</td><td>23k</td><td>20.8†</td><td>22.1†</td><td>29.6</td><td>50k</td><td>24.3†</td><td>24.4†</td><td>33.8</td></tr><tr><td>GRPO, 40 tasks × 4</td><td>23k</td><td>24.3†</td><td>24.6†</td><td>32.0</td><td>46k</td><td>28.5†</td><td>25.4†</td><td>33.5</td></tr><tr><td>GiGPO</td><td>26k</td><td>25.7</td><td>23.9†</td><td>31.0</td><td>53k</td><td>27.8†</td><td>25.9†</td><td>33.5</td></tr><tr><td>GiGPO, 40 tasks × 4</td><td>25k</td><td>22.2†</td><td>23.0†</td><td>29.9</td><td>51k</td><td>27.8†</td><td>27.5†</td><td>35.6</td></tr><tr><td>SO-PPO (tuned clip)</td><td>26k</td><td>26.1</td><td>23.0†</td><td>28.9</td><td>53k</td><td>39.8</td><td>33.5†</td><td>47.2</td></tr><tr><td>SO-PPO (original clip)</td><td>34k</td><td>19.4†</td><td>27.9</td><td>38.0</td><td>75k</td><td>33.5</td><td>31.8†</td><td>41.9</td></tr><tr><td>Turn-ratio PPO</td><td>26k</td><td>19.7†</td><td>19.8†</td><td>25.0</td><td>68k</td><td>25.0†</td><td>23.5†</td><td>33.5</td></tr><tr><td>Q-PPO (3)</td><td>31k</td><td>23.8†</td><td>24.4†</td><td>31.2</td><td>65k</td><td>33.9†</td><td>33.8†</td><td>41.4</td></tr><tr><td>Wide-Tree-PPO (3)</td><td>32k</td><td>27.1</td><td>26.4</td><td>33.8</td><td>65k</td><td>32.7†</td><td>31.2†</td><td>38.4</td></tr><tr><td>Fixed WM</td><td>30k</td><td>22.9†</td><td>21.4†</td><td>25.4</td><td>57k</td><td>31.7†</td><td>29.8†</td><td>34.9</td></tr><tr><td>PaW (coefficient 0.5)</td><td>27k</td><td>21.5†</td><td>22.7†</td><td>28.5</td><td>53k</td><td>26.8†</td><td>25.4†</td><td>35.6</td></tr><tr><td>PaW (coefficient 1.0)</td><td>30k</td><td>18.7†</td><td>18.2†</td><td>26.1</td><td>67k</td><td>10.9†</td><td>23.8†</td><td>34.2</td></tr><tr><td>PaW (coefficient 0.25)</td><td>24k</td><td>20.8†</td><td>21.0†</td><td>25.7</td><td>44k</td><td>26.1†</td><td>26.4†</td><td>35.2</td></tr><tr><td>Majority-vote labels</td><td>25k</td><td>23.2†</td><td>23.9†</td><td>29.9</td><td>57k</td><td>26.8†</td><td>26.3†</td><td>30.3</td></tr><tr><td>OCSD (self teacher)</td><td>26k</td><td>29.2</td><td>25.7</td><td>32.7</td><td>53k</td><td>29.2†</td><td>26.9†</td><td>34.9</td></tr><tr><td>OCSD (32B teacher)</td><td>25k</td><td>25.7</td><td>25.4†</td><td>32.0</td><td>48k</td><td>27.1†</td><td>27.0†</td><td>34.9</td></tr><tr><td>OPD (32B teacher)</td><td>27k</td><td>27.5</td><td>27.6</td><td>39.4</td><td>54k</td><td>32.4†</td><td>34.0†</td><td>47.2</td></tr><tr><td>PPO-256, second run of seed 1º</td><td>40k</td><td>25.4</td><td>24.1</td><td>33.1</td><td>77k</td><td>35.6</td><td>34.6</td><td>40.5</td></tr><tr><td>WM-PPO (policy channel only), second run of seed 3ª</td><td>32k</td><td>29.2</td><td>26.9</td><td>34.5</td><td>65k</td><td>39.4</td><td>38.1</td><td>46.1</td></tr></table>

Table 13: ScienceWorld, all methods, iterations 48 and 60 (continued from Table 12; same marks).
<table><tr><td rowspan="2">Method</td><td colspan="4">iteration 48</td><td colspan="4">iteration 60</td></tr><tr><td></td><td>Steps Greedy pass@1</td><td></td><td>pass@3</td><td>Steps</td><td></td><td>Greedy pass@1</td><td>pass@3</td></tr><tr><td>WM-PPO (ours) (3)</td><td>99k</td><td>44.4</td><td>43.2</td><td>51.5</td><td>128k</td><td>52.9</td><td>51.6</td><td>59.6</td></tr><tr><td>WM-PPO (policy channel only) (3)</td><td>99k</td><td>41.1†</td><td>41.2†</td><td>50.1</td><td>130k</td><td>47.2†</td><td>47.2†</td><td>56.0</td></tr><tr><td>WM-PPO, terminal-only value target</td><td>93k</td><td>43.3</td><td>42.4</td><td>50.0</td><td>125k</td><td>48.2†</td><td>49.3</td><td>59.9</td></tr><tr><td>PPO-160 (3)</td><td>84k</td><td>36.3†</td><td>37.1†</td><td>45.9</td><td>113k</td><td>39.3†</td><td>42.6†</td><td>52.2</td></tr><tr><td>PPO-256 (3)</td><td>114k</td><td>38.7†</td><td>38.3†</td><td>47.4</td><td>156k</td><td>43.8†</td><td>43.5†</td><td>51.9</td></tr><tr><td>GRPO</td><td>79k</td><td>31.0†</td><td>28.1†</td><td>38.4</td><td>103k</td><td>35.2†</td><td>35.7†</td><td>43.3</td></tr><tr><td>GRPO, 40 tasks × 4</td><td>69k</td><td>29.9†</td><td>29.1†</td><td>35.9</td><td>88k</td><td>31.3†</td><td>31.5†</td><td>39.8</td></tr><tr><td>GiGPO</td><td>79k</td><td>28.9†</td><td>28.3†</td><td>35.9</td><td>105k</td><td>38.4†</td><td>36.3†</td><td>43.7</td></tr><tr><td>GiGPO, 40 tasks × 4</td><td>78k</td><td>29.6†</td><td>31.9†</td><td>39.1</td><td>100k</td><td>34.2†</td><td>33.0†</td><td>41.2</td></tr><tr><td>SO-PPO (tuned clip)</td><td>83k</td><td>40.5</td><td>35.7†</td><td>48.6</td><td>111k</td><td>46.5†</td><td>49.4</td><td>60.9</td></tr><tr><td>SO-PPO (original clip)</td><td>128k</td><td>16.9†</td><td>16.4†</td><td>17.6</td><td>164k</td><td>15.8†</td><td>16.2†</td><td>16.9</td></tr><tr><td>Turn-ratio PPO</td><td>104k</td><td>25.4†</td><td>23.8†</td><td>29.9</td><td>138k</td><td>25.0†</td><td>26.5†</td><td>38.4</td></tr><tr><td>Q-PPO (3)</td><td>99k</td><td>40.6†</td><td>38.9†</td><td>46.7</td><td>133k</td><td>46.1†</td><td>46.2†</td><td>54.8</td></tr><tr><td>Wide-Tree-PPO (3)</td><td>95k</td><td>36.0†</td><td>35.8†</td><td>43.2</td><td>125k</td><td>40.8†</td><td>40.0†</td><td>47.8</td></tr><tr><td>Fixed WM</td><td>82k</td><td>35.2†</td><td>34.9†</td><td>40.1</td><td>108k</td><td>38.7†</td><td>39.0†</td><td>47.5</td></tr><tr><td>PaW (coefficient 0.5)</td><td>80k</td><td>29.2†</td><td>28.9†</td><td>38.7</td><td>109k</td><td>35.6†</td><td>38.8†</td><td>48.9</td></tr><tr><td>PaW (coefficient 1.0)</td><td>97k</td><td>6.0†</td><td>29.3†</td><td>39.1</td><td>131k</td><td>6.0†</td><td>30.8†</td><td>37.0</td></tr><tr><td>PaW (coefficient 0.25)</td><td>65k</td><td>29.2†</td><td>29.5†</td><td>37.7</td><td>87k</td><td>32.4†</td><td>32.9†</td><td>39.4</td></tr><tr><td>Majority-vote labels</td><td>94k</td><td>24.6†</td><td>24.6†</td><td>29.6</td><td>135k</td><td>25.0†</td><td>25.0†</td><td>27.8</td></tr><tr><td>OCSD (self teacher)</td><td>81k</td><td>33.5†</td><td>36.6†</td><td>43.0</td><td>112k</td><td>34.2†</td><td>43.7†</td><td>53.2</td></tr><tr><td>OCSD (32B teacher)</td><td>70k</td><td>33.8†</td><td>32.7†</td><td>41.9</td><td>90k</td><td>43.0†</td><td>41.5†</td><td>49.6</td></tr><tr><td>OPD (32B teacher)</td><td>80k</td><td>28.2†</td><td>31.3†</td><td>44.4</td><td>106k</td><td>36.3†</td><td>33.9†</td><td>46.8</td></tr><tr><td>PPO-256, second run of seed 1ª</td><td>116k</td><td>39.4</td><td>39.4</td><td>45.4</td><td>160k</td><td>40.8</td><td>44.6</td><td>55.3</td></tr><tr><td>WM-PPO (policy channel only), second run of seed 38</td><td>95k</td><td>43.7</td><td>43.5</td><td>50.7</td><td>122k</td><td>44.4</td><td>45.7</td><td>53.2</td></tr></table>

## E.2 Learning curves and steps to a reference success

Figure 12 names every method of Figure 4. Figure 13 gives the learning curves against real interaction, and Table 14 gives the real steps at which each run first passed reference success levels.

![](images/967364a397c007d06c423cc318a5dec53cb3188e20ab7dec877028a75fc0122c.jpg)

![](images/07e7b6921a529fe8593297e2dcbbb58397113c8014ae65ca6ccda387ed452dc3.jpg)  
Figure 12: Success against real interaction, with every method named. The data and conventions of Figure 4; the bracketed numbers there refer to the names given here.

![](images/27472e15c37fb6d423a7c476fa82532b44b1e15577acb7c5293b6245f608ef41.jpg)

![](images/9927f07b18dc9ef00c6d2f47a52563ccff8b1078f277b52331443ca407e24172.jpg)  
WM-PPO (3) WM-PPO (policy channel only) (3) PPO-160 (3) PPO-256 (3) Q-PPO (3) Wide-Tree-PPO (3) CAPO (1) SO-PPO (1)  
Figure 13: Learning curves against real interaction. Greedy success on all evaluation games during training (every third iteration; one greedy episode per game) against the real steps executed since the common warm-up checkpoint (warm-up and prefix replays for branches are not counted). Thin gray lines: individual training runs; colored lines: their mean, with the number of runs in parentheses.

Table 14: Real steps to reach a reference success. Real steps (thousands, from iteration 13) at which each training run first reached the given greedy success in the evaluations during training (every third iteration, one greedy episode per game, so the resolution is coarse); one entry per run, methods as in Table 2; >: not reached by the run’s last evaluation, after the given number of real steps. On ALFWorld both WM-PPO variants, PPO-160, PPO-256, Wide-Tree-PPO, Fixed WM and PaW were trained to iteration 55 and the other methods to iteration 36, so their last evaluations come at different step counts.
<table><tr><td></td><td colspan="2">ALFWorld</td><td colspan="2">ScienceWorld</td></tr><tr><td>Method</td><td>80%</td><td>85%</td><td>40%</td><td>45%</td></tr><tr><td>WM-PPO (ours)</td><td>69, 94, 97</td><td>78, 103, 106</td><td>69, 86, 93</td><td>94, 109, 120</td></tr><tr><td>WM-PPO (policy channel only)</td><td>84,67,78</td><td>93,86,96</td><td>89,86,90</td><td>96, 110, 130</td></tr><tr><td colspan="5">Standard training and refinements</td></tr><tr><td>PPO-160</td><td>106, 157, 158</td><td>121, &gt;165, 167</td><td>85, &gt;118, 93</td><td>&gt;114,&gt;118,&gt;108</td></tr><tr><td>PPO-256</td><td>129, 141, 133</td><td>185, 180, 133</td><td>&gt;172, 82, 111</td><td>&gt;172, 142, 119</td></tr><tr><td>GRPO</td><td>&gt;113</td><td>&gt;113</td><td>&gt;103</td><td>&gt;103</td></tr><tr><td>GiGPO</td><td>&gt;109</td><td>&gt;109</td><td>&gt;105</td><td>&gt;105</td></tr><tr><td>CAPO</td><td>63</td><td>88</td><td></td><td></td></tr><tr><td>SO-PPO</td><td>100</td><td>100</td><td>67</td><td>97</td></tr><tr><td>Turn-ratio PPO</td><td>64</td><td>73</td><td></td><td></td></tr><tr><td colspan="5">Structural controls</td></tr><tr><td>Q-PPO</td><td>81,81, &gt;102</td><td>100, 89, &gt;102</td><td>98, 92, 105</td><td>&gt;135, 118, &gt;129</td></tr><tr><td>Wide-Tree-PPO Fixed WM</td><td>96, 93, 139 80</td><td>143, 103, &gt;168 108</td><td>&gt;120, 110, 125 &gt;108</td><td>&gt;120, &gt;123, &gt;133 &gt;108</td></tr><tr><td>Auxiliary feedback added to PPO-160</td><td></td><td></td><td></td><td></td></tr><tr><td colspan="5"></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Majority-vote labels</td><td>&gt;132</td><td>&gt;132</td><td>&gt;135</td><td>&gt;135</td></tr><tr><td>OCSD (self teacher)</td><td>87,81, 101</td><td>&gt;97,&gt;90,&gt;101</td><td>&gt;112</td><td>&gt;112</td></tr><tr><td>OCSD (32B teacher)</td><td>89</td><td>&gt;99</td><td>90</td><td>&gt;90</td></tr><tr><td>OPD (32B teacher)</td><td>&gt;112</td><td>&gt;112</td><td>&gt;106</td><td>&gt;106</td></tr><tr><td>PaW</td><td>89</td><td>97</td><td>&gt;109</td><td>&gt;109</td></tr></table>

## E.3 Individual training runs

Table 15 lists every run of the methods with more than one run.

Table 15: Results of individual training runs for the methods with more than one completed run (policy-only success in %, greedy / pass@1). Each seed is represented by its first run. Same-seed reruns, in no mean or test: ALFWorld WM-PPO seed 2, 85.8 / 86.7 at iteration 36 (first run 87.6 / 86.9); ScienceWorld PPO-256 seed 1, 40.8 / 44.6, and WM-PPO (policy channel only) seed 3, 44.4 / 45.7 (first run 44.0 / 45.5).
<table><tr><td>Environment</td><td>It.</td><td>Method</td><td>Run 1</td><td>Run 2</td><td>Run 3</td></tr><tr><td rowspan="6">ALFWorld</td><td>36</td><td>WM-PPO</td><td>88.3 / 89.7</td><td>87.6/ 86.9</td><td>86.5 / 86.7</td></tr><tr><td></td><td>WM-PPO (policy channel only)</td><td>89.4 / 90.8</td><td>85.8 / 87.7</td><td>90.1 / 89.8</td></tr><tr><td>PPO-160</td><td></td><td>70.1 / 77.4</td><td>52.2 / 65.5</td><td>68.2 / 70.3</td></tr><tr><td>PPO-256</td><td></td><td>83.2 / 86.1</td><td>86.1 / 83.8</td><td>79.9 / 84.4</td></tr><tr><td>Q-PPO</td><td></td><td>85.8 / 86.1</td><td>92.7 / 92.3</td><td>74.8 / 87.2</td></tr><tr><td>OCSD (self teacher)</td><td>Wide-Tree-PPO</td><td>76.6 / 83.8 84.7 / 83.7</td><td>88.7 / 86.6</td><td>70.1 / 76.5 83.2 / 79.2</td></tr><tr><td rowspan="5">ALFWorld</td><td>55</td><td>WM-PPO</td><td>91.2 / 92.3</td><td>81.0/ 85.2 88.3 / 89.7</td><td></td></tr><tr><td></td><td>WM-PPO (policy channel only)</td><td>91.2 / 92.5</td><td>90.5 / 93.8</td><td>86.5 / 90.5 92.3 / 92.6</td></tr><tr><td>PPO-160</td><td></td><td>89.8 / 89.8</td><td>80.3 / 86.6</td><td></td></tr><tr><td>PPO-256</td><td></td><td></td><td></td><td>85.0 / 85.4</td></tr><tr><td>Wide-Tree-PPO</td><td></td><td>86.1 / 91.0 86.1 / 89.2</td><td>93.8 / 90.3 92.0 / 93.6</td><td>92.3 / 92.1 83.9 / 89.2</td></tr><tr><td>ScienceWorld</td><td>60</td><td>WM-PPO</td><td>54.9 / 56.5</td><td>53.5 / 51.3</td><td>50.4 / 46.9</td></tr><tr><td rowspan="4"></td><td></td><td>WM-PPO (policy channel only)</td><td>48.6 / 48.9</td><td>48.9 / 47.1</td><td>44.0 / 45.5</td></tr><tr><td></td><td>PPO-160</td><td>40.5 / 43.4</td><td>37.7 / 39.8</td><td>39.8 / 44.5</td></tr><tr><td>PPO-256</td><td></td><td>33.8 / 36.0</td><td>49.3 / 46.4</td><td></td></tr><tr><td>Q-PPO</td><td></td><td></td><td></td><td>48.2 / 48.2</td></tr><tr><td></td><td></td><td>Wide-Tree-PPO</td><td>45.1 / 44.6 37.3 / 36.3</td><td>51.8 / 49.2 45.1 / 42.4</td><td>41.5 / 45.0 40.1 / 41.4</td></tr></table>

Table 16: ScienceWorld at iteration 60 by task type (policy-only success in %, greedy / pass@1; methods averaged over their completed runs as in Table 2). Shortened task names: melting-point (known) = measure-melting-point-known-substance; paint-secondary, paint-tertiary = chemistrymix-paint-secondary-color, -tertiary-color; power-(non)renewable = power-component-renewable-vsnonrenewable-energy; conductivity (unknown) = test-conductivity-of-unknown-substances; lifespan longest-then-shortest = lifespan-longest-lived-then-shortest-lived.
<table><tr><td>Task type</td><td>Games</td><td>WM-PPO</td><td>Policy ch. only</td><td>PPO-160</td><td>Q-PPO</td><td>SO-PPO (tuned)</td></tr><tr><td>Procedural (all)</td><td>188</td><td>46.3/45.1</td><td>39.5/38.6</td><td>29.6/34.4</td><td>39.0/39.4</td><td>39.4/42.2</td></tr><tr><td>use-thermometer melting-point (known)</td><td>18</td><td>0.0/0.0</td><td>0.0/0.0</td><td>0.0/0.0</td><td>0.0/0.0</td><td>0.0/0.0</td></tr><tr><td>power-component</td><td>18</td><td>66.7/58.0</td><td>53.7/48.1</td><td>37.0/46.9</td><td>38.9/47.5</td><td>50.0/44.4</td></tr><tr><td>test-conductivity</td><td>10</td><td>73.3/76.7</td><td>70.0/70.0</td><td>43.3/48.9</td><td>43.3/36.7</td><td>60.0/76.7</td></tr><tr><td>find-living-thing</td><td>18</td><td>25.9/25.3</td><td>35.2/34.6</td><td>22.2/21.0</td><td>20.4/24.7</td><td>27.8/31.5</td></tr><tr><td>find-non-living-thing</td><td>18</td><td>70.4/77.8 83.3/75.9</td><td>40.7/34.6</td><td>31.5/32.1</td><td>53.7/45.7</td><td>100.0/88.9</td></tr><tr><td>find-plant</td><td>18 18</td><td>90.7/97.5</td><td>66.7/71.0</td><td>63.0/66.7</td><td>85.2/83.3</td><td>33.3/44.4</td></tr><tr><td>find-animal</td><td>18</td><td>75.9/67.3</td><td>96.3/98.8</td><td>88.9/95.7</td><td>92.6/99.4</td><td>77.8/88.9</td></tr><tr><td>chemistry-mix</td><td></td><td></td><td>51.9/48.8</td><td>25.9/51.2</td><td>64.8/67.3</td><td>77.8/88.9</td></tr><tr><td>paint-secondary</td><td>16</td><td>0.0/0.0 29.6/26.5</td><td>0.0/0.7</td><td>0.0/0.0</td><td>0.0/1.4</td><td>6.2/0.0</td></tr><tr><td>paint-tertiary</td><td>18 18</td><td>0.0/0.0</td><td>29.6/27.8 0.0/0.0</td><td>16.7/18.5 0.0/0.0</td><td>27.8/22.2</td><td>5.6/11.1</td></tr><tr><td>Forced-choice (all)</td><td></td><td></td><td></td><td></td><td>0.0/0.0</td><td>0.0/0.0</td></tr><tr><td>power-(non)renewable</td><td>96 10</td><td>66.0/64.2 30.0/24.4</td><td>62.2/63.9 16.7/25.6</td><td>58.3/58.6</td><td>60.1/59.6</td><td>60.4/63.5</td></tr><tr><td>conductivity (unknown)</td><td>18</td><td>55.6/51.2</td><td>31.5/45.1</td><td>13.3/11.1</td><td>16.7/6.7</td><td>40.0/36.7</td></tr><tr><td>lifespan-longest-lived</td><td>18</td><td>100.0/100.0</td><td>100.0/100.0</td><td>35.2/40.7</td><td>27.8/35.2</td><td>55.6/51.9</td></tr><tr><td>lifespan-shortest-lived</td><td>18</td><td>94.4/90.1</td><td>94.4/93.5</td><td>100.0/100.0</td><td>100.0/100.0</td><td>94.4/96.3</td></tr><tr><td>lifespan-longest-then-shortest</td><td>18</td><td>79.6/80.9</td><td>90.7/84.6</td><td>92.6/92.6</td><td>96.3/93.2</td><td>83.3/88.9</td></tr><tr><td>identify-life-stages-1</td><td>8</td><td>0.0/0.0</td><td></td><td>72.2/70.4</td><td>83.3/79.6</td><td>66.7/70.4</td></tr><tr><td></td><td></td><td></td><td>0.0/0.0</td><td>0.0/0.0</td><td>0.0/0.0</td><td>0.0/0.0</td></tr><tr><td>identify-life-stages-2</td><td>6</td><td>16.7/20.4</td><td>16.7/11.1</td><td>11.1/7.4</td><td>11.1/18.5</td><td>0.0/33.3</td></tr></table>

## E.4 ScienceWorld by task type

Table 16 breaks down the iteration-60 results of Table 2 by task type. With 6–18 games per type, differences on single types are noisy, so Section 5.3 uses the procedural and forced-choice totals. Three types are solved by none of these methods, and the chemistry-mix tasks are almost never solved.

## E.5 Interaction and computation

Table 17 separates the costs per iteration. WM-PPO executes 3% fewer real steps per iteration than PPO-160 on ALFWorld and 13% more on ScienceWorld, where its branches run longer than the budget assumes. It generates about twice as many policy responses (the four candidates per chain step) and adds 1,500–1,950 world-model generations. An iteration takes 40 minutes on ALFWorld against 32 for PPO-160, and 55 minutes against 34 on ScienceWorld. Because WM-PPO also occupies the world-model servers and trainers, its GPU-hours per iteration are about 2.7 and 3.8 times those of PPO-160; these are upper bounds, since shared servers are counted in full. The value channel adds no generation, only a term of the critic’s loss, and its GPU-hours per iteration are similar to those of the policy-channel-only variant. Branching, which Wide-Tree-PPO shares, adds little (1.5 against 1.7 GPU-hours on ALFWorld, 1.9 against 1.7 on ScienceWorld); most of the difference comes from sampling the candidates and from the world model’s predictions and training.

Table 17: Interaction and computation per iteration (iterations 13–36 for ALFWorld and 13–60 for ScienceWorld). Real steps: steps the policy generated and executed, averaged over all runs as in Table 2. Policy samples: generated responses (four per chain step for WM-PPO and Q-PPO; resampling at branch points is not logged, so these are lower bounds for branched methods). WM generations: one per distinct candidate at an evaluated decision point and two per branch pair (Q-PPO values the same candidates with its action-value model and generates no predictions). Wall: measured wall-clock time per iteration, including the in-training evaluation; GPU·h: wall-clock time times all GPUs allocated to the run (7 for WM-PPO, 3–5 for the other methods; shared servers counted in full, an upper bound on use). Both are means over the iterations of each method’s three runs on A100-40GB nodes; two early segments on H100 nodes are excluded.
<table><tr><td>Environment</td><td>Method</td><td>Real steps</td><td>Policy samples</td><td>WM generations</td><td>Wall (min)</td><td>GPU·h</td></tr><tr><td>ALFWorld</td><td>WM-PPO</td><td>4,232</td><td>8,914</td><td>1,930</td><td>40</td><td>4.6</td></tr><tr><td></td><td>PPO-160</td><td>4,381</td><td>4,381</td><td>0</td><td>32</td><td>1.7</td></tr><tr><td></td><td>PPO-256</td><td>6,252</td><td>6,252</td><td>0</td><td>28</td><td>2.3</td></tr><tr><td></td><td>Wide-Tree-PPO</td><td>4,399</td><td>4,399</td><td>0</td><td>31</td><td>1.5</td></tr><tr><td></td><td>Q-PPO</td><td>4,152</td><td>8,795</td><td>一</td><td>46</td><td>2.7</td></tr><tr><td>ScienceWorld</td><td>WM-PPO</td><td>2,674</td><td>5,205</td><td>1,483</td><td>55</td><td>6.4</td></tr><tr><td></td><td>PPO-160</td><td>2,361</td><td>2,361</td><td>0</td><td>34</td><td>1.7</td></tr><tr><td></td><td>PPO-256</td><td>3,258</td><td>3,258</td><td>0</td><td>32</td><td>2.4</td></tr><tr><td></td><td>Wide-Tree-PPO</td><td>2,607</td><td>2,607</td><td>0</td><td>37</td><td>1.9</td></tr><tr><td></td><td>Q-PPO</td><td>2,767</td><td>5,443</td><td>一</td><td>70</td><td>3.5</td></tr></table>

## E.6 Sokoban comparison with ProAct’s MC-PPO

We compare WM-PPO with the MC-PPO configuration that ProAct releases for Sokoban (Yu et al., 2026c). The environment is text Sokoban on a 6 × 6 grid with one box and at most 15 steps; solving a level gives a terminal reward of 1 and any other ending 0. Training draws from 18,908 levels, and evaluation uses 300 held-out levels with greedy decoding and three temperature-1 samples per level. All three methods start from a common 12-iteration PPO warm-up (52.0% greedy success) and train through iteration 36 with the budgets and hyperparameters of the main experiments: PPO-160 plays 160 levels per iteration, and WM-PPO plays 64 levels with one chain and three branches, both channels and the gate as in Section 3; its initial world model is trained on the 7,962 executed transitions of the warm-up. MC-PPO uses PPO-160’s budget and level schedule and estimates the value of each visited state from 1,000 uniformly random rollouts of at most 15 steps in a copy of the simulator, with TD(0) advantages, discount 0.95 and a Monte-Carlo value weight of 1. ProAct’s GLAD stage, which fine-tunes on reasoning derived from environment search, is not part of this comparison. Over iterations 13–36, MC-PPO performs about 33 million random rollouts, which are not counted as real steps. ProAct’s own Sokoban experiments use one to five boxes and shaped rewards. Each method has one run.

Table 18: Sokoban (300 held-out levels; success in $\% ;$ one run per method). Steps: real environment steps from iteration 13. Marks as in Table 2, against WM-PPO.
<table><tr><td rowspan="2">Method</td><td colspan="3">iteration 24</td><td colspan="3">iteration 36</td></tr><tr><td>Steps</td><td>Greedy</td><td>pass@1</td><td>Steps</td><td>Greedy</td><td>pass@1</td></tr><tr><td>WM-PPO (ours)</td><td>20k</td><td>80.3</td><td>69.8</td><td>36k</td><td>87.0</td><td>84.4</td></tr><tr><td>ProAct (MC-PPO)</td><td>18k</td><td>77.3</td><td>70.9</td><td>33k</td><td>77.3†</td><td>68.1†</td></tr><tr><td>PPO-160</td><td>19k</td><td>62.0†</td><td>53.4†</td><td>36k</td><td>67.7†</td><td>63.2†</td></tr></table>

## F Analyses behind Section 5

The following analyses support Section 5. Real branch outcomes test the reliability of transition values (Section 5.1); measurements of prediction quality, advantages and critic values examine how transition values enter learning (Section 5.2); and task trajectories show how the trained policy acts (Section 5.3).

## F.1 Branch pairs and the reliability gate

Protocol. In each iteration, every branch pair (the chain action and the branch action at the branch point, both played to the end) was valued by the iteration’s pre-update models and assigned to a margin bin with that iteration’s thresholds. A pair therefore never contributes to the weights that were applied to it. Pairs whose two returns differ and whose value difference is non-zero are decisive, and a decisive pair is correct when the preferred action has the higher return. WM-PPO’s transition value includes the ending rule: a predicted completion or failure is assigned 1 or 0, and any other successor is valued by the critic. Q-PPO’s value is $Q _ { \omega } ( s _ { t } , y )$ . Each method’s pairs come from its own sampling, so the two methods are not evaluated on the same pairs. In ScienceWorld, a pair involves focus if either action starts with focus on; this is a property of the action text, not a label of whether the outcome is decided at that step. Intervals: 2,000 bootstrap resamples of (run, iteration) clusters. Data: three runs of each method, iterations 13–60 on ScienceWorld and 13–36 (both WM-PPO variants: to 55) on ALFWorld. Table 19 gives the agreement and the gain by phase and margin bin, and Table 20 splits the late ScienceWorld pairs. For the single Fixed WM run on ScienceWorld, agreement in the largest bin is 0.67, 0.75, 0.53 and 0.64 in the four phases, against 0.72, 0.79, 0.88 and 0.93 for the run of WM-PPO (policy channel only) with the same seed and the same initial world model.

Gain and agreement. The gain in Equation (16) averages over all branch pairs in a margin bin. Among untied pairs in bin ℓ, let $d _ { \ell }$ be the share whose outcomes differ and $c _ { \ell }$ the share of these in which the preferred action has the better outcome. Then

$$
g _ { \ell } = \operatorname* { P r } \left( \Delta \hat { q } \neq 0 \ | \ \ell \right) d _ { \ell } \left( 2 c _ { \ell } - 1 \right) .\tag{20}
$$

When the outcomes differ, a correct preference contributes +1 and an incorrect preference contributes −1; equal outcomes or tied transition values contribute zero. The gate converts the gain and the number of supporting pairs into a training weight through Equation (17).

Early comparisons with the value channel. Among ScienceWorld branch pairs with different real outcomes, the largest-margin comparisons favor the successful action in 78–82% of cases with the value channel early in training, against 71–74% with the policy channel only (Figure 7b). Later, the two variants reach similar ranking accuracy.

Table 19: Agreement of training-time preferences with real outcomes, by phase and margin bin (decisive pairs only). Gain per pair: Equation (16) averaged over all pairs of the bin, decisive or not (small+middle: mean of the two bins).
<table><tr><td colspan="5"></td><td colspan="3">Agreement by margin bin</td><td colspan="2">Gain per pair</td></tr><tr><td>Environment</td><td>Valued by</td><td>Iterations</td><td></td><td>Pairs Agreement small</td><td></td><td>1 middle</td><td>large</td><td>small+middle</td><td>large</td></tr><tr><td rowspan="10">ScienceWorld</td><td rowspan="5">WM-PPO</td><td>13-24</td><td>415</td><td>0.71</td><td>0.61</td><td>0.55</td><td>0.78</td><td>+0.008</td><td>+0.077</td></tr><tr><td>25-36</td><td>549</td><td>0.73</td><td>0.48</td><td>0.58</td><td>0.82</td><td>+0.007</td><td>+0.145</td></tr><tr><td>37-48</td><td>626</td><td>0.77</td><td>0.49</td><td>0.58</td><td>0.88</td><td>+0.008</td><td>+0.199</td></tr><tr><td>49-60</td><td>654</td><td>0.76</td><td>0.55</td><td>0.58</td><td>0.84</td><td>+0.010</td><td>+0.222</td></tr><tr><td>Policy channel only 13-24</td><td>397</td><td>0.67 0.65</td><td>0.49 0.49</td><td>0.65</td><td>0.71</td><td>+0.009</td><td>+0.060</td></tr><tr><td></td><td>25-36</td><td>543</td><td></td><td>0.55</td><td>0.74</td><td>+0.005</td><td></td><td>+0.092</td></tr><tr><td></td><td>37-48 49-60</td><td>527</td><td>0.73</td><td>0.41</td><td>0.57</td><td>0.84</td><td>+0.003</td><td>+0.164</td></tr><tr><td rowspan="4">Q-PPO (learned Q)</td><td></td><td>579</td><td>0.73</td><td>0.54</td><td>0.45</td><td>0.87</td><td>-0.003</td><td>+0.200</td></tr><tr><td>13-24</td><td>514</td><td>0.59</td><td>0.53</td><td>0.52</td><td>0.65</td><td>+0.003</td><td>+0.053</td></tr><tr><td>25-36</td><td>579</td><td>0.70</td><td>0.62</td><td>0.58</td><td>0.75</td><td>+0.012</td><td>+0.117</td></tr><tr><td>37-48 49-60</td><td>679 775</td><td>0.67 0.70</td><td>0.67 0.63</td><td>0.54 0.52</td><td>0.71 0.78</td><td>+0.012</td><td>+0.114</td></tr><tr><td>ALFWorld</td><td>WM-PPO</td><td>13-24</td><td>424</td><td></td><td></td><td></td><td></td><td>+0.010</td><td>+0.171</td></tr><tr><td rowspan="6"></td><td></td><td>25-36</td><td>424</td><td>0.60 0.55</td><td>0.51 0.49</td><td>0.61 0.52</td><td>0.66 0.64</td><td>+0.015 +0.002</td><td>+0.051 +0.047</td></tr><tr><td></td><td>37-55</td><td>398</td><td>0.53</td><td>0.56</td><td>0.44</td><td>0.60</td><td>-0.002</td><td>+0.026</td></tr><tr><td>Policy channel only</td><td>13-24</td><td>525</td><td>0.55</td><td>0.52</td><td>0.48</td><td>0.64</td><td>-0.000</td><td>+0.050</td></tr><tr><td></td><td>25-36</td><td>334</td><td>0.58</td><td>0.50</td><td>0.59</td><td>0.68</td><td>+0.009</td><td>+0.051</td></tr><tr><td>37-55</td><td></td><td>361 0.58</td><td></td><td>0.58</td><td>0.50</td><td>0.65</td><td>+0.004</td><td>+0.037</td></tr><tr><td rowspan="3">Q-PPO (learned Q)</td><td>13-24</td><td>501</td><td>0.54</td><td>0.53</td><td>0.52</td><td>0.56</td><td>+0.007</td><td></td><td>+0.020</td></tr><tr><td></td><td></td><td>390 0.56</td><td></td><td>0.54</td><td>0.56</td><td>0.57</td><td>+0.013</td><td></td></tr><tr><td></td><td>25-36</td><td></td><td></td><td></td><td></td><td></td><td></td><td>+0.016</td></tr></table>

Table 20: ScienceWorld, iterations 45–60: which comparisons the agreement rests on (agreement in %, with the number of decisive pairs, i.e. pairs whose real outcomes differ; three runs of each method). Last column: WM-PPO minus Q-PPO in percentage points, with cluster-bootstrap 95% intervals.
<table><tr><td>Decisive pairs</td><td>WM-PPO</td><td>Policy ch. only</td><td>Q-PPO</td><td>WM-PPO – Q-PPO</td></tr><tr><td>All</td><td>76.2 (867)</td><td>72.3 (755)</td><td>69.9 (1017)</td><td>+6.3 [+1, +11]</td></tr><tr><td>Largest-margin bin</td><td>84.4 (604)</td><td>84.7 (483)</td><td>76.3 (683)</td><td>+8.1 [+3, +14]</td></tr><tr><td>A focus on command involved</td><td>85.5 (543)</td><td>84.0 (451)</td><td>77.7 (623)</td><td>+7.8 [+2, +13]</td></tr><tr><td>No focus on command</td><td>60.8 (324)</td><td>54.9 (304)</td><td>57.6 (394)</td><td>+3.2 [−5, +10]</td></tr><tr><td>One successor predicted to end (1 / 0)</td><td>94.7 (378)</td><td>94.5 (328)</td><td></td><td></td></tr><tr><td>Both continue, valued by the critic</td><td>62.0 (489)</td><td>55.3 (427)</td><td></td><td>一</td></tr><tr><td>of these, in the largest-margin bin</td><td>67.0 (230)</td><td>62.2 (172)</td><td></td><td>一</td></tr></table>

Table 20 splits the late ScienceWorld pairs by the source of the values and by the kind of action. Where one successor is predicted to end the episode, its transition value is the predicted outcome, 1 or 0, rather than a critic value (Equation (6)). These comparisons agree with the real outcome in 95% of the pairs, with or without the value channel. Where both successors continue, the critic orders them, and the value channel raises agreement from 55% to 62% (from 62% to 67% in the largest-margin bin). The value channel thus strengthens the part of the valuation that rests on the critic. The run with a terminal-only value target, which values every candidate not predicted to end the episode at the critic’s value of the current state, reaches 48.2% greedy success at iteration 60, close to the policy-channel-only run with the same seed (48.6%) and below the WM-PPO run with that seed (54.9%; Tables 13 and 15). Against Q-PPO, the difference lies in comparisons that involve a focus command, which commits the agent to an answer and whose consequence the next observation often reveals; without such a command, the two methods are close. Q-PPO’s gate gives more weight in total, a summed weight of about 263 per iteration against 183 for WM-PPO, yet WM-PPO is the more successful of the two (Section 5.1).

Table 22 summarizes the gate per phase of training, averaged over iterations and over the three runs in each environment. Decision pts: chain steps with at least two distinct candidates. Non-zero, Full: the number of these whose margin bin had a positive weight or a weight of 1. Mean w : the weight applied to each margin bin in the iteration’s update; over ScienceWorld iterations 45–60, the window of Table 20, it is 0.15, 0.37 and 1.00. The 13–24 rows include iterations 13 and 14, which use the start-up rule of Table 5. Agreement: fraction of the iteration’s branch pairs with different outcomes in which the action with the higher value had the better outcome. Pairs: number of such pairs per iteration. Gain: the realized gain of Equation (16) on the iteration’s pairs, per margin bin.

Gate ablation. Two copies of the first WM-PPO run on ScienceWorld fix the three bin weights from iteration 13 onward; both channels, the world-model updates and all other settings are unchanged. Weights (1, 1, 1) admit every comparison at full weight, and weights (0, 0, 1) (top bin) keep only the largest-margin bin. Relative to the calibrated first run, the summed gate weight per iteration is 1.9–2.8 times as large without the gate and 0.4–0.7 times as large with the top bin (ranges over the four training phases). Applying the gate rule to each copy’s own branch records shows what calibration would have admitted: in the top-bin copy, positive weight for the small and middle bins in 28 and 27 of 48 iterations; in the full-weight copy, positive weight for the small bin in only 11 of 48 iterations, where that bin’s mean realized gain is −0.005. Table 21 gives the official evaluations.

Table 21: Gate ablation on ScienceWorld (284 evaluation games; success in %; first run of each method). <sup>†</sup>/<sup>‡</sup>: WM-PPO / the other method is significantly higher in a per-game paired test (p < 0.05).
<table><tr><td colspan="2"></td><td colspan="2">iteration 36</td><td colspan="2">iteration 48</td><td colspan="2">iteration 60</td></tr><tr><td>Method</td><td> $( w _ { 1 } , w _ { 2 } , w _ { 3 } )$ </td><td>Greedy</td><td>pass@1</td><td>Greedy</td><td>pass@1</td><td>Greedy</td><td>pass@1</td></tr><tr><td>WM-PPO (ours) calibrated</td><td></td><td>40.8</td><td>41.7</td><td>45.8</td><td>46.5</td><td>54.9</td><td>56.5</td></tr><tr><td>No gate</td><td>(1, 1, 1)</td><td>35.6†</td><td>35.9†</td><td>39.1†</td><td>41.4†</td><td>44.0†</td><td>43.1†</td></tr><tr><td>Top bin</td><td>(0,0, 1)</td><td>34.9†</td><td>34.0†</td><td>43.0</td><td>45.3</td><td>48.6†</td><td>48.6†</td></tr><tr><td>PPO-160</td><td></td><td>32.4†</td><td>29.8†</td><td>40.5†</td><td>40.4†</td><td>40.5†</td><td>43.4†</td></tr></table>

Gate coverage with the value channel. On ScienceWorld, the value channel increases the number of evaluated states with nonzero weight early in training. In iterations 13–24 and 25–36, WM-PPO admits 327 states per iteration in each phase, against 281 and 275 with the policy channel only, averaged over iterations and the three runs. The corresponding full-weight counts are 56 against 41 in the first phase and 98 against 75 in the second. The largest-margin bin of WM-PPO remains at full weight from iteration 29 onward; the policy-channel-only variant’s mean weight falls during iterations 28–39, reaching about 0.67 (Figure 8a,b).

Table 22: Reliability gate of WM-PPO during training (per iteration, averaged over each phase and over the runs).
<table><tr><td>Environment</td><td>Iterations</td><td>Decision pts</td><td>Non-zero</td><td></td><td>Full Mean we: small / middle / large Agreement</td><td></td><td>Pairs</td><td>Gain: small / middle / large</td></tr><tr><td>ALFWorld</td><td>13-24</td><td>846</td><td>470</td><td>45</td><td>0.17 / 0.36 / 0.42</td><td>0.61</td><td>12</td><td>+0.002 / +0.033 / +0.051</td></tr><tr><td></td><td>25-36</td><td>593</td><td>339</td><td>55</td><td>0.11 / 0.33 / 0.70</td><td>0.57</td><td>12</td><td>-0.003 / +0.005 / +0.046</td></tr><tr><td></td><td>37-48</td><td>448</td><td>201</td><td>5</td><td>0.18 / 0.03 / 0.38</td><td>0.56</td><td>7</td><td>+0.008 / -0.013 / +0.026</td></tr><tr><td></td><td>49-55</td><td>470</td><td>204</td><td>6</td><td>0.17 / 0.01 / 0.26</td><td>0.54</td><td>7</td><td>+0.001 / -0.004 / +0.029</td></tr><tr><td>ScienceWorld</td><td>13-24</td><td>540</td><td>327</td><td>56</td><td>0.15 / 0.27 / 0.64</td><td>0.71</td><td>12</td><td>+0.009 / +0.010 / +0.078</td></tr><tr><td></td><td>25-36</td><td>542</td><td>327</td><td>98</td><td>0.11 / 0.25 / 0.98</td><td>0.73</td><td>15</td><td>-0.002 / +0.016 / +0.149</td></tr><tr><td></td><td>37-48</td><td>473</td><td>253</td><td>117</td><td>0.11 / 0.35 / 1.00</td><td>0.77</td><td>17</td><td>-0.001 /+0.016 /+0.199</td></tr><tr><td></td><td>49-60</td><td>471</td><td>302</td><td>97</td><td>0.15 / 0.32 / 1.00</td><td>0.75</td><td>18</td><td>+0.003 / +0.015 / +0.225</td></tr></table>

## F.2 World-model predictions along training

Table 23 gives the statistics behind Figure 6a,b in four phases of training. At each evaluated chain state, we compare that iteration’s world model’s prediction for the executed action with the observation the environment returned. Exact: the two match after whitespace normalization. Rejections: recall and precision of the prediction that the simulator rejects the command. Endingsforeseen: fraction of the steps that ended the episode for which the world model predicted an ending. On ScienceWorld, the online world model (the three policy-channel-only runs) is compared with the run whose world model is fixed, starting from the same initial model; each is measured on the states visited by its own policy. By phase, the online model predicts 84–86% of the executed steps’ observations exactly throughout training, whereas the fixed model falls from 81% to 68%. The online model also foresees 73–75% of the steps that end an episode, against about half for the fixed model. Figure 6 shows the same quantities in windows of three iterations.

Table 23: World-model predictions of executed steps along training.
<table><tr><td>Environment</td><td>World model</td><td>Iterations</td><td>Steps</td><td>Exact</td><td>Rejections: recall / precision</td><td>Endings foreseen</td></tr><tr><td rowspan="6">ScienceWorld</td><td>online</td><td>13-24</td><td>16,602</td><td>0.84</td><td>0.99 / 0.99</td><td>0.74</td></tr><tr><td></td><td>25-36</td><td>18,066</td><td>0.86</td><td>0.99 / 0.99</td><td>0.73</td></tr><tr><td></td><td>37-48</td><td>17,036</td><td>0.86</td><td>0.99 / 0.99</td><td>0.74</td></tr><tr><td>fixed</td><td>49-60</td><td>16,452</td><td>0.85</td><td>0.99 / 0.99</td><td>0.75</td></tr><tr><td></td><td>13-24</td><td>5,114</td><td>0.81</td><td>0.97 / 0.98</td><td>0.51</td></tr><tr><td></td><td>25-36</td><td>4,543</td><td>0.80</td><td>0.96 / 0.98</td><td>0.52</td></tr><tr><td></td><td></td><td>37-48</td><td>3,959</td><td>0.74 0.68</td><td>0.96 / 0.97 0.92 / 0.89</td><td>0.50</td></tr><tr><td></td><td></td><td>49-60</td><td>4,308</td><td></td><td></td><td>0.49</td></tr></table>

## F.3 Recomputed advantages

Simulator feedback makes accepted and rejected commands a direct probe of how learning signals distinguish candidates at the same state. A rejected command usually leaves the task recoverable and rarely determines its final outcome. The following measurements compare advantages and candidate preferences; Section F.4 reports the critics’ comparisons during training and at evaluation, and Section F.5 reports the policy’s command choices during training.

To measure how the advantages separate useful from useless steps, we recomputed the raw tree-GAE advantages of ScienceWorld iterations 15, 30, 45 and 60 for the three runs each of PPO-160, WM-PPO, WM-PPO (policy channel only) and Wide-Tree-PPO. The recomputation uses the critic of training (saved at the end of the preceding iteration) and the training code’s estimator (γ = 1, $\lambda = 0 . 9 5 )$ . The means and standard deviations of the recomputed advantages match the training logs to four decimals in all 48 (run, iteration) cases, and so do the counts of auxiliary rows of both WM-PPO variants. Executed non-terminal edges are classified by the true next observation as rejected (the simulator’s No known action matches that input) or accepted. Table 24 summarizes the results, and Table 25 gives the counts.

The first term of the advantage, $V _ { \psi } ( s _ { t + 1 } ) - V _ { \psi } ( s _ { t } )$ , averages −0.003 for rejected and +0.002 for accepted commands under PPO-160’s critics (−0.005 and +0.003 for Wide-Tree-PPO, −0.005 and +0.003 for WM-PPO, −0.006 and +0.003 for the policy-only variant). At states where both kinds were executed, the accepted command has the higher advantage in 58% of 476 pairs for Wide-Tree-PPO, in 61% of 431 pairs for WM-PPO and in 56% of 476 pairs for the policy-only variant; in 77%, 84% and 81% of these pairs, the two advantages differ by less than 0.25 standard deviations. The auxiliary comparisons were examined at the states with nonzero weight at which candidates were predicted to be rejected and accepted (prediction precision 0.99): 1,789 states for WM-PPO and 1,560 for the policy-only variant. At these states, the comparison prefers an accepted candidate in 64% for WM-PPO (78% at full weight and 60% at partial weight; mean weight 0.49) and in 67% for the policy-only variant (80% and 64%; mean weight 0.44). In the remaining third, mostly in partially weighted bins, the critic values the predicted successor of the rejected command, which leaves the task where it was, above those of the alternatives. These are comparisons in which the critic values both successors (Table 20).

a Accepted command: higher GAE  
![](images/157ddb992ad5d7f2b238449cc30f0d52f4c4e3119b4e3f30be8cee62df6086ea.jpg)  
Share of pairs (%)  
Different state sets in (a) and (b)

b Accepted command: preferred  
![](images/1518c639c6390f6200f683c1028fa8b8178e47dac7f1d6b9b95c2a0d2826fe6e.jpg)  
Share of states (%)  
Full-weight states (b)  
Figure 14: Which signal prefers the accepted command at the same state? ScienceWorld, iterations 15, 30, 45 and 60 of all runs of each method. (a) Advantages: at states where an accepted and a rejected command were both executed, the share of pairs in which the tree-GAE advantage computed with the method’s own critic is higher for the accepted command (number of pairs). (b) Comparisons: at evaluated states with nonzero weight and both kinds of candidate, the share of states at which the comparison prefers an accepted candidate (number of states; open marker: states at full weight). Both axes are compressed below 50%. The two rates are computed on different sets of states. Q-PPO compares candidates by its learned action value. Acceptance of an unexecuted candidate is read from one fixed world model for all methods; each run’s own world model gives nearly the same rates (Table 24). PPO-160 plays one episode per task and has no such states. Advantages on all steps: Table 24 in Appendix F.3.

Table 24: Do the advantages tell useful from useless steps? Advantages of executed non-terminal steps, recomputed with the critics used in training (iterations 15, 30, 45 and 60 on ScienceWorld; 15, 24, 30 and 36 on ALFWorld; all runs of each method, except Wide-Tree-PPO on ALFWorld, first run only). Useful / useless: accepted / rejected commands on ScienceWorld; steps that take, process or place the task object / wasted steps (“Nothing happens”, or returning to a searched location before the object is found) on ALFWorld. AUC: over all executed steps, the probability that a useful step has the higher advantage; reinforced: share of useless steps with a positive advantage. Same state: at states with one useful and one useless executed step, the share at which the useful one has the higher advantage (number of such pairs; PPO-160 plays one episode per task and has none). For the comparisons of the two WM-PPO variants: among the states with nonzero weight and both kinds of candidate, the share at which the comparison prefers a useful candidate (at full weight on ScienceWorld, 80% for the policy-channel-only variant and 78% for WM-PPO; on ALFWorld, the variant with the policy channel only prefers a useless candidate at 21% of the states and a third, neutral candidate at 14%, the full method a useless one at 26%). Acceptance of an unexecuted candidate is read from each run’s own world model here, and from one fixed world model for all methods in Figure 14, which gives 66% and 64% on ScienceWorld. Counts in Table 25.
<table><tr><td rowspan="2"></td><td colspan="3">ScienceWorld</td><td colspan="3">ALFWorld</td></tr><tr><td>AUC</td><td>Reinforced</td><td>Same state</td><td>AUC</td><td>Reinforced</td><td>Same state</td></tr><tr><td>PPO-160, tree GAE</td><td>0.51</td><td>39%</td><td></td><td>0.57</td><td>34%</td><td></td></tr><tr><td>Wide-Tree-PPO, tree GAE</td><td>0.51</td><td>45%</td><td>58% (476)</td><td>0.62</td><td>31%</td><td>86% (21)</td></tr><tr><td>WM-PPO (policy channel only)</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>tree GAE</td><td>0.48</td><td>40%</td><td>56% (476)</td><td>0.65</td><td>29%</td><td>74% (43)</td></tr><tr><td>comparisons</td><td></td><td></td><td>67% (1,560)</td><td></td><td></td><td>66% (116)</td></tr><tr><td>WM-PPO</td><td></td><td></td><td></td><td>0.61</td><td></td><td></td></tr><tr><td>tree GAE</td><td>0.51</td><td>43%</td><td>61% (431)</td><td></td><td>20%</td><td>75% (52)</td></tr><tr><td>comparisons</td><td>一</td><td></td><td>64% (1,789)</td><td>一</td><td>一</td><td>59% (219)</td></tr></table>

The same measurement on ALFWorld. For ALFWorld, we recomputed iterations 15, 24, 30 and 36 of the same methods in the same way, and the logs again match to four decimals. Executed non-terminal steps are classified from the true next observation. Advancing steps take the task object, heat, cool or clean it, place it in the target receptacle, or use the desk lamp. Wasted steps receive Nothing happens or go to a location already visited in the episode before the task object has been taken (returning to the target receptacle is not counted). Under PPO-160’s critics, the first term of the advantage averages +0.014 for advancing steps and −0.011 and −0.009 for the two kinds of wasted step. The comparisons were examined at the states with nonzero weight at which both an advancing and a wasted candidate were sampled (219 for WM-PPO, 116 for the policy-only variant); the preferred candidate is advancing at 59% and 66% of these states and wasted at 26% and 21% (mean weights 0.64 and 0.60). Few states have both kinds of executed step (21–52 pairs per method; Table 24), and these states differ from those of the comparisons. Unlike on ScienceWorld, the advantages already separate the two kinds of step: the gap between their mean advantages is 0.41 standard deviations under PPO-160’s critics, against 0.10 on ScienceWorld (Table 25).

Table 25: Counts and mean raw advantages behind Table 24. ScienceWorld: rejected and accepted commands; ALFWorld: wasted (Nothing happens / revisits) and advancing steps. Three runs each, except Wide-Tree-PPO on ALFWorld (one run).
<table><tr><td colspan="8">ScienceWorld</td></tr><tr><td>Method</td><td>Useless</td><td>Useful</td><td>Mean: useless / useful</td><td>Gap</td><td>AUC</td><td>Useless with</td><td> $A ^ { \mathrm { G A E } } > 0$ </td></tr><tr><td rowspan="4">PPO-160 Wide-Tree-PPO WM-PPO</td><td>11,200</td><td>14,624</td><td>-0.018/+0.003</td><td>0.10</td><td>0.51</td><td></td><td>39%</td></tr><tr><td>13,167</td><td>13,739</td><td>-0.013/+0.004</td><td>0.08</td><td>0.51</td><td></td><td>45%</td></tr><tr><td>11,387</td><td>14,665</td><td>-0.008/+0.010</td><td>0.11</td><td>0.51</td><td></td><td>43%</td></tr><tr><td>11,410</td><td>17,150</td><td>+0.004/+0.006</td><td></td><td>0.02</td><td>0.48</td><td>40%</td></tr><tr><td colspan="8">ALFWorld (wasted = Nothing happens + revisits)</td></tr><tr><td>PPO-160</td><td>2,909 + 11,288</td><td>5,007</td><td>-0.051, -0.074/+0.032</td><td>0.41</td><td>0.57</td><td></td><td>34%</td></tr><tr><td>Wide-Tree-PPO</td><td>1,154 + 3,826</td><td>1,769</td><td>-0.041, -0.048/+0.017</td><td></td><td>0.29</td><td>0.62</td><td>31%</td></tr><tr><td>WM-PPO</td><td>2,331 + 12,398</td><td>4,275</td><td>-0.053, -0.069 / -0.009</td><td></td><td>0.35 0.61</td><td></td><td>20%</td></tr><tr><td>policy channel only</td><td>1,884 + 10,786</td><td>5,358</td><td>-0.025, -0.070 / +0.019</td><td></td><td>0.33</td><td>0.65</td><td>29%</td></tr></table>

## F.4 Candidates at evaluation states

At the decision points of the final (iteration-60) PPO-160 and WM-PPO policies on 60 procedural ScienceWorld evaluation games, every distinct candidate among four samples was executed in a copy of the game, and the same policy then played the episode to the end five times. The success rate estimates the candidate’s true value. With four candidates and five continuations each, no candidate led to a success in any continuation at 73.5% of PPO-160’s decision points and at 55.9% of WM-PPO’s (Figure 15a). At such points, these continuations cannot tell the candidates apart, although the candidates’ true values need not be exactly zero. Figure 16 shows how three critics order the true next states of the candidate pairs whose estimates differ. On both sets of decision points, all three critics are close to chance (47–54% of all such pairs and 48–54% of the 20% with the largest value gap). With five continuations, each estimate has a standard error of up to about 0.2.

Rejected commands in ScienceWorld: detection. During training, on the executed steps of the evaluated decision points, the online world model of WM-PPO (policy channel only) predicts 98–99% of the rejections with a precision of 0.99 (Table 23). On evaluation states, the recall is lower. On 80 procedural evaluation games, the final WM-PPO and PPO-160 policies played as in training. At every decision point, each distinct candidate among four samples was executed in a copy of the game and also predicted by a final world model. Of 1,717 candidates on WM-PPO’s decision points, 670 were rejected by the simulator; the world model of the first WM-PPO run predicted a rejection for 582 candidates, 580 of them correctly (recall 0.87, precision 1.00). On PPO-160’s decision points, the world model of the first policy-channel-only run finds the rejections among 664 of 1,760 candidates with recall 0.84 and precision 0.99.

![](images/13b272fb7b146663cbaebef795636f4fa6a307d5a39f958ccd6a7d3cdd6325c1.jpg)

![](images/778e6e16269ed124d4768d33334946c8334ee336b310eac4f6cd90964d6bd973.jpg)

Figure 15: Candidates at evaluation states (ScienceWorld, 60 procedural evaluation games, final policies). Every distinct candidate among four samples was executed and then played to the end five times by the same policy; its success rate estimates the value of its next state. (a) Decision points at which at least one candidate succeeded in one of its five continuations, each policy on its own decision points (not a comparison of the methods). (b) Pairs of non-terminal candidates at the same state of which the simulator accepted one and rejected the other: share in which a critic values the accepted command’s true next state higher (ties count half; 481 pairs on PPO-160’s states, 430 on $\mathbf { W M - P P O ^ { \prime } s } ) ;$ groups give the policy whose states were used, rows the critic. All critics and states are those of the first run of each method; WM-PPO is the full method. Horizontal bars: 95% bootstrap intervals clustered by decision point. The critics’ ordering of all candidate pairs is in Figure 16 (Appendix F.4).  
![](images/743d549711a37474ae81ccda1009d1859c95988fcb8dcb2dcc74dca0d6902abd.jpg)  
Figure 16: Critics’ ordering of the candidates’ true next states (ScienceWorld, 60 procedural evaluation games, final policies). Agreement with the success rates estimated from five continuations, over candidate pairs whose estimates differ (filled circles; 313 pairs on WM-PPO’s states, 171 on PPO-160’s) and over the 20% of pairs with the largest gap in each critic’s own values (open diamonds; 63 and 35 pairs, selected separately for each critic, so the three diamonds do not refer to the same pairs); (a) states from WM-PPO, (b) states from PPO-160; rows give the critic. Horizontal bars: 95% bootstrap intervals clustered by decision point.

Rejected commands: comparisons during training. Of the 17,385 real branch pairs of iterations 13–60 in the three runs of WM-PPO (policy channel only), 3,835 have exactly one action predicted to be rejected and no predicted ending. The critic of that iteration valued the accepted action higher in 67% of them (59% in the smallest-margin bin, 73% in the middle bin and 74% in the largest). By phase, the rate is 64%, 70%, 69% and 67% (iterations 13–24, 25–36, 37–48 and 49–60), and in the largest bin of iterations 49–60 it is 78%. In the three runs of WM-PPO, 3,971 pairs qualify; the rate is 69% (60%, 72% and 81% by margin bin; 63%, 72%, 74% and 68% by phase), and in the largest bin of iterations 49–60 it is 90% (171 pairs).

Rejected commands: final critics. On the evaluation states of the true-outcome test (Appendix F.4; Figure 15b), we consider pairs of non-terminal candidates of which the simulator accepted one and rejected the other. On PPO-160’s decision points (481 pairs), the iteration-60 critics value the accepted command’s true next state higher in 65% (PPO-160), 80% (Wide-Tree-PPO) and 60% (WM-PPO) of the pairs; on WM-PPO’s decision points (430 pairs), the rates are 76%, 80% and 60% (53% for the critic of a second WM-PPO run). A second run of the test on the 80 procedural evaluation games described above gives the same ordering (PPO-160 72–76%, Wide-Tree-PPO 80–83%, WM-PPO 58–59%). The critics of PPO-160 and Wide-Tree-PPO separate rejected commands more sharply, while their policies keep issuing them (37.5% and 40.2% of procedural commands late in training, means over runs); the policy of WM-PPO issues fewer (29.5%). The final critic of WM-PPO (policy channel only) orders 43–48% of these pairs correctly (45–47% in the separate test).

## F.5 Behavior of the trained policies

Section 5.3 takes success rates from the official evaluation and behavior rates from greedy re-runs of the final checkpoints that log every step. Greedy decoding on the inference server is not bit-exact, and the re-runs reproduce the official per-game outcome for 94–99% of the ALFWorld and 91.5–96.5% of the ScienceWorld games. The pooled behavior differences of Figure 10b do not hold in every run: one PPO-160 run completes the processing in all 91 of its picked-up episodes and places the object before processing it in 10.

Tables 26 and 27 are computed from the training chains (the episode’s own path, without branches), decoded into actions and observations. Table 26 covers procedural ScienceWorld tasks in the first and last windows of training for nine methods, and Table 27 covers ALFWorld in iterations 34–36, pooling the runs of each method. One pattern is not discussed method by method in Section 5.3: the tuned SO-PPO ends slightly fewer ScienceWorld evaluation episodes with a failure than WM-PPO (27.1% against 30.3%) but more at the step limit (26.4% against 16.8%).

Premature focus. Figure 10d counts, in the greedy re-runs of the 188 procedural ScienceWorld games, the episodes whose first accepted focus on command comes before the state that the answer depends on exists; the definition was fixed per task type before counting. In the chemistry tasks (52 games), the first focus precedes any mix whose feedback reports a product; in the melting-point task (18), the focus on an answer box precedes any thermometer reading; in the find tasks (72), the first focus is on an object that does not qualify, and the simulator ends the episode as failed. The remaining 46 procedural games ask for a focus first or are answered by placing an object, so they cannot contain a premature focus. The re-runs reproduce the official per-game outcome for 89–97% of the games. Over all procedural games, the premature share is 22.0% for WM-PPO, 26.8% for PPO-256, 28.7% for PPO-160 and 28.7% for the policy-channel-only variant (per run 13.3–35.1%,

![](images/7365b4a66207de3dfe28eb5519915259e85a006ef6b31730ef63cffd80793293.jpg)  
Figure 17: Rejected commands during training (ScienceWorld, procedural tasks of the training episodes, chains only, windows of three iterations). Thin gray lines: individual runs; colored lines: mean over the runs of each method. Other methods: Table 26.

Table 26: ScienceWorld, procedural tasks of the training episodes (% of commands, or of episodes for wrong-focus endings).
<table><tr><td rowspan="2">Method</td><td colspan="2">Rejected commands</td><td colspan="2">Repeated commands</td><td colspan="2">Ended by a wrong focus</td></tr><tr><td>it. 13–15</td><td>it. 58–60</td><td>it. 13–15</td><td>it. 58–60</td><td>it. 13–15</td><td>it. 58–60</td></tr><tr><td>WM-PPO (3 runs)</td><td>49.9</td><td>29.5</td><td>65.1</td><td>54.0</td><td>64.0</td><td>28.6</td></tr><tr><td>WM-PPO (policy channel only, 3 runs)</td><td>48.5</td><td>30.5</td><td>64.6</td><td>55.3</td><td>65.9</td><td>32.5</td></tr><tr><td>PPO-160 (3 runs)</td><td>48.1</td><td>37.5</td><td>65.3</td><td>57.7</td><td>62.1</td><td>36.2</td></tr><tr><td>PPO-256 (3 runs)</td><td>47.2</td><td>49.9</td><td>65.3</td><td>65.0</td><td>62.4</td><td>33.6</td></tr><tr><td>OCSD (self teacher)</td><td>51.5</td><td>43.8</td><td>65.7</td><td>66.7</td><td>60.1</td><td>28.3</td></tr><tr><td>Q-PPO (3 runs)</td><td>50.0</td><td>40.7</td><td>66.2</td><td>52.9</td><td>62.1</td><td>32.4</td></tr><tr><td>SO-PPO (tuned clip)</td><td>49.6</td><td>34.4</td><td>67.3</td><td>55.2</td><td>60.5</td><td>28.6</td></tr><tr><td>Wide-Tree-PPO (3 runs)</td><td>48.2</td><td>40.2</td><td>64.4</td><td>57.2</td><td>63.7</td><td>44.9</td></tr><tr><td>Fixed WM</td><td>50.0</td><td>17.8</td><td>65.2</td><td>52.5</td><td>63.0</td><td>62.6</td></tr></table>

Table 27: ALFWorld training episodes of iterations 34–36. Repeated actions: % of steps whose action was already taken earlier in the episode; the last column counts go to actions before the first take in successful episodes.
<table><tr><td>Method</td><td>Success</td><td>Steps (successful)</td><td>Repeated actions</td><td>go to before first take (successful)</td></tr><tr><td>WM-PPO (ours)</td><td>83.9</td><td>16.5 (12.0)</td><td>42.4</td><td>3.4</td></tr><tr><td>WM-PPO (policy channel only)</td><td>86.8</td><td>16.0 (12.4)</td><td>38.7</td><td>3.6</td></tr><tr><td>PPO-160</td><td>64.7</td><td>23.3 (14.1)</td><td>53.2</td><td>4.5</td></tr><tr><td>PPO-256</td><td>83.2</td><td>18.1 (13.7)</td><td>42.9</td><td>3.9</td></tr><tr><td>Q-PPO</td><td>86.5</td><td>16.4 (12.7)</td><td>40.0</td><td>3.8</td></tr><tr><td>Wide-Tree-PPO</td><td>77.3</td><td>19.5 (13.4)</td><td>47.1</td><td>4.0</td></tr></table>

23.9–30.9%, 25.5–33.0% and 25.5–34.0%); per game, the share of WM-PPO is lower than that of PPO-256 in 37 games and higher in 25. Split by task type after this count, it is 6.9%, 19.0%, 19.4% and 21.3% on the find tasks (per game, WM-PPO is lower than PPO-256 in 14 games and higher in 1), 64.8%, 79.6%, 74.1% and 83.3% on the melting-point task, and 47.4%, 42.9%, 51.3% and 45.5% on the chemistry tasks, where one WM-PPO run focuses prematurely in all of its chemistry episodes. A premature focus ends in failure in the chemistry and find tasks; in the melting-point task, 20 of 35 such WM-PPO episodes still name the right box.

![](images/91e8eb048a82c7cf989bcf05f31b0b3720ce841f1d5ff753be6a6ed3ed157976.jpg)  
Figure 18: How ScienceWorld evaluation episodes end (final checkpoints, iteration 60, greedy, 284 games, mean over three runs): success, failure declared by the simulator, or the 40-step limit. WM-PPO ends fewer episodes than PPO-160 with a failure and fewer at the step limit.

Table 28: Further evaluation games (greedy re-runs of the final checkpoints; successes over the three runs of each method).
<table><tr><td>Game</td><td>WM-PPO</td><td>PPO-160</td><td>What differs</td></tr><tr><td>ALFWorld: heat an apple, put it on the dining table</td><td>3/3</td><td>0/3</td><td>WM-PPO takes the apple and heats it in the microwave (7 steps); PPO-160 never takes the apple and circles the microwave, fridge and stove burners</td></tr><tr><td>ALFWorld: clean a cloth, put it in a cabinet</td><td>3/3</td><td>0/3</td><td>WM-PPO finds the cloth in the first cabinet it opens and cleans it at the sink (8 steps); PPO-160 alternates between sink and cabinets without taking it</td></tr><tr><td>ALFWorld: put a pepper shaker in a drawer</td><td>3/3</td><td>0/3</td><td>WM-PPO checks the countertops, where the pepper shaker is (7 steps); PPO-160 opens drawers and cabinets in turn and never finds it</td></tr></table>

Further cases. Table 28 lists further evaluation games from the greedy re-runs in which the decisive difference is an intermediate operation or the direction of search. At iteration 31 of a run with the policy channel only, the policy held a dirty pot and sampled go to sinkbasin 1 twice and move pot 1 to stoveburner 2 twice. The world model predicted You arrive at sinkbasin 1. . . and You move the pot 1 to the stoveburner 2, and the critic valued the two predicted successors at 0.952 and 0.950. The bin of this small margin had weight 0.85 in that iteration, so the four samples received auxiliary advantages of +0.42, +0.42, −0.42 and −0.42. PPO-160 made the second move in the same game one iteration later and failed. In one further ALFWorld game (heating an apple), the final world model and critic of one run rank the fridge slightly above the microwave (1.051 against 1.044; another run ranks the microwave first), although WM-PPO succeeds in all three runs. We therefore report these additional cases without assigning decisive-state values to them.

## G Inputs and outputs of the models

In the training example of Figure 19, the task requires making orange paint and then focusing on it. The bowl currently contains red and yellow paint, but the orange paint does not yet exist. The world model predicts that ‘mix bowl’ will produce orange paint, whereas ‘focus on orange paint’ will be rejected and leave the bowl unchanged; both predictions match the environment. The critic assigns a higher transition value to the successor in which the paint has been made. Although both real continuations ultimately succeed, these immediate consequences supply a preference for mixing before focusing. Of the four sampled responses, the three ‘mix bowl’ samples each receive $+ { \frac { 1 } { 4 } }$ and the ‘focus on orange paint’ sample receives $- \frac { 3 } { 4 }$ , times the bin’s weight. These auxiliary advantages raise the log-probability of the mixing responses relative to the focus response (Equation (9)).

Candidate mix bowl Candidate focus on orange paint   
Current observation You pour the contents of the wood cup into the bowl. [Current room] . . . a   
(both) table. On the table is: a bowl (containing red paint, yellow paint). . . .   
[score: 40]   
World model predicts Yellow and red paint mix to produce No known action matches that input.   
oˆ<sub>t+1</sub>(a) orange paint. [Current room] . . . a bowl [Current room] . . . a bowl (containing   
(containing orange paint). . . . [score: red paint, yellow paint). . . . [score:   
50] 40]   
Critic reads $\hat { s } _ { t + 1 } ( a )$ task and rules (copied); step counter and last five observation–action pairs, now ending with the real current   
observation and the candidate (updated by the program); oˆ<sub>t+1</sub>(a) as the current observation (generated);   
referable objects from before the step, still without orange paint (copied)   
Comparison qˆ<sub>t</sub>(mix bowl) − qˆ<sub>t</sub>(focus on orange paint) = +0.153, in the large-margin bin  
Figure 19: From predicted changes to transition values at a training decision point (ScienceWorld, iteration 59 of a run with the policy channel only). Predictions are reproduced verbatim (“. . . ” marks identical room lines); each matches the observation returned when that action was executed in the chain or a branch. The world model predicts the whole next observation, including the progress score. Neither prediction ends the episode, so both transition values come from the critic’s evaluation of the predicted successors. The complete inputs and outputs of two decision points follow below.

Table 29: What each model reads and writes. Generated: produced by the world model; copied: taken unchanged from $s _ { t } ;$ program: updated deterministically by the prompt constructor C.
<table><tr><td></td><td>ALFWorld</td><td>ScienceWorld</td><td>Source</td></tr><tr><td>Policy and critic input st</td><td>task; step counter; last 5 observation-action pairs; current observation; admissible actions</td><td>task and rules; step counter; last 5 observation-action pairs; current observation (action feedback, room, inventory, score); action templates;</td><td>real interaction</td></tr><tr><td>World-model input (xt , a)</td><td>task; initial observation; all previous observations and actions; current observation; admissible actions; candidate</td><td>referable objects task; last 8 actions, without observations; current observation; referable objects; candidate</td><td>real interaction</td></tr><tr><td>World-model output ôt+1</td><td>next observation as the environment prints it</td><td>action feedback, [Current room], [Inventory],[score: S],and [task completed] or [task failed] when the simulator ends the</td><td>generated</td></tr><tr><td>Successor state st+1 =</td><td>policy format: window of recent pairs ends with (real current observation, candidate);</td><td>task the same; referable objects from before the step</td><td>program (window, step counter); copied (task,</td></tr><tr><td> $\tilde { C } ( { s } _ { t } , a , \hat { o } _ { t + 1 } )$ </td><td> $\hat { o } _ { t + 1 }$  as current observation; admissible actions from before the step none (observations report no endings)</td><td>an end marker in  $\hat { o } _ { t + 1 }$  replaces the critic&#x27;s</td><td>lists); generated  $( \hat { o } _ { t + 1 } )$  parsed from  $\hat { o } _ { t + 1 }$ </td></tr><tr><td>Predicted ending Progress score</td><td>none</td><td>value by 1 (completed) or 0 (failed) printed in every observation; read by the policy and the critic; part of the world</td><td>environment text</td></tr><tr><td></td><td></td><td>model&#x27;s input and of its next-observation target; not used as a reward or as a separate</td><td></td></tr><tr><td>Reward</td><td>1 if completed within 40 actions, else 0</td><td>training target 1 if the simulator reports completion within 40 actions, else 0</td><td>environment</td></tr></table>

Table 29 distinguishes the simulator’s internal state, which no model sees; the prompt state $s _ { t }$ read by the policy and critic; and the next observation $\hat { o } _ { t + 1 }$ generated by the world model. Two evaluated decision points from the training logs follow verbatim. The world-model input is not logged as text; we rebuilt it from the logged transition record with the training prompt function, which reproduces the logged policy prompt exactly. The prediction and the critic input are decoded from the logged token ids, and re-assembling the critic input from the decoded prediction reproduces those ids. The

raw world-model generation and the critic’s individual values of predicted successors are not logged;   
the trainer logs only the value difference of each branch pair.

## ALFWorld: training iteration 48, step 14

Run 1 of WM-PPO (policy channel only); models of iteration 47; training game train/pick\_ cool\_then\_place\_in\_recep-Pot-None-CounterTop-24/trial\_T20190907\_ 002332\_305243. The policy sampled move pan 1 to fridge 1, cool pan 1 with fridge 1, cool pan 1 with fridge 1, cool pan 1 with fridge 1; move pan 1 to fridge 1 was executed in the chain and cool pan 1 with fridge 1 in a branch from the same state. Logged difference of transition values qˆ(chain) − qˆ(branch) = +0.137 (margin bin 3 of 3); real returns 1 (chain) and 0 (branch). Both predictions equal the observations the environment printed; this is not representative of predictions that involve content the agent has not seen.

(i) World-model input (x<sub>t</sub>, a) for candidate move pan 1 to fridge 1 (rebuilt token-exactly from the logged record; the input for the other candidate differs only in its last line)

```csv
<|im_start|>system
You are a world model of the ALFRED text household environment.<|im_end|>
<|im_start|>user
Task: put a cool pot in countertop.
Initial observation: You are in the middle of a room. Looking quickly around you, you see a cabinet
23, a cabinet 22, a cabinet 21, a cabinet 20, a cabinet 19, a cabinet 18, a cabinet 17, a
cabinet 16, a cabinet 15, a cabinet 14, a cabinet 13, a cabinet 12, a cabinet 11, a cabinet 10,
a cabinet 9, a cabinet 8, a cabinet 7, a cabinet 6, a cabinet 5, a cabinet 4, a cabinet 3, a
cabinet 2, a cabinet 1, a coffeemachine 1, a countertop 3, a countertop 2, a countertop 1, a
diningtable 1, a drawer 4, a drawer 3, a drawer 2, a drawer 1, a fridge 1, a garbagecan 1, a
microwave 1, a sinkbasin 1, a stoveburner 4, a stoveburner 3, a stoveburner 2, a stoveburner 1,
and a toaster 1.
History of steps so far: [Step 1: observation ’You are in the middle of a room. Looking quickly
around you, you see a cabinet 23, a cabinet 22, a cabinet 21, a cabinet 20, a cabinet 19, a
cabinet 18, a cabinet 17, a cabinet 16, a cabinet 15, a cabinet 14, a cabinet 13, a cabinet 12,
a cabinet 11, a cabinet 10, a cabinet 9, a cabinet 8, a cabinet 7, a cabinet 6, a cabinet 5, a
cabinet 4, a cabinet 3, a cabinet 2, a cabinet 1, a coffeemachine 1, a countertop 3, a
countertop 2, a countertop 1, a diningtable 1, a drawer 4, a drawer 3, a drawer 2, a drawer 1, a
fridge 1, a garbagecan 1, a microwave 1, a sinkbasin 1, a stoveburner 4, a stoveburner 3, a
stoveburner 2, a stoveburner 1, and a toaster 1.’, action ’go to countertop 1’] [Step 2:
observation ’You arrive at countertop 1. On the countertop 1, you see a bread 1, and a spatula
1.’, action ’go to countertop 2’] [Step 3: observation ’You arrive at countertop 2. On the
countertop 2, you see a pan 1.’, action ’go to countertop 3’] [Step 4: observation ’You arrive
at countertop 3. On the countertop 3, you see a fork 3, a knife 1, and a mug 1.’, action ’go to
diningtable 1’] [Step 5: observation ’You arrive at diningtable 1. On the diningtable 1, you see
a apple 1, a butterknife 1, a cup 1, a egg 2, a knife 2, a lettuce 1, a mug 2, a papertowelroll
1, a potato 3, a potato 2, a potato 1, a soapbottle 2, a spatula 2, a spoon 3, a spoon 1, and a
tomato 1.’, action ’go to fridge 1’] [Step 6: observation ’You arrive at fridge 1. The fridge 1
is closed.’, action ’open fridge 1’] [Step 7: observation ’You open the fridge 1. The fridge 1
is open. In it, you see nothing.’, action ’go to microwave 1’] [Step 8: observation ’You arrive
at microwave 1. The microwave 1 is closed.’, action ’open microwave 1’] [Step 9: observation ’
You open the microwave 1. The microwave 1 is open. In it, you see a egg 1.’, action ’go to
countertop 1’] [Step 10: observation ’You arrive at countertop 1. On the countertop 1, you see a
bread 1, and a spatula 1.’, action ’go to countertop 2’] [Step 11: observation ’You arrive at
countertop 2. On the countertop 2, you see a pan 1.’, action ’take pan 1 from countertop 2’] [
Step 12: observation ’You pick up the pan 1 from the countertop 2.’, action ’go to countertop
3’] [Step 13: observation ’You arrive at countertop 3. On the countertop 3, you see a fork 3, a
knife 1, and a mug 1.’, action ’go to fridge 1’]
Current observation: You arrive at fridge 1. The fridge 1 is open. In it, you see nothing.
Admissible actions: [’close fridge 1’, ’cool pan 1 with fridge 1’, ’examine fridge 1’, ’examine pan
1’, ’go to cabinet 1’, ’go to cabinet 10’, ’go to cabinet 11’, ’go to cabinet 12’, ’go to
cabinet 13’, ’go to cabinet 14’, ’go to cabinet 15’, ’go to cabinet 16’, ’go to cabinet 17’, ’go
to cabinet 18’, ’go to cabinet 19’, ’go to cabinet 2’, ’go to cabinet 20’, ’go to cabinet 21’,
go to cabinet 22’, ’go to cabinet 23’, ’go to cabinet 3’, ’go to cabinet 4’, ’go to cabinet 5’,
go to cabinet 6’, ’go to cabinet 7’, ’go to cabinet 8’, ’go to cabinet 9’, ’go to coffeemachine
1’, ’go to countertop 1’, ’go to countertop 2’, ’go to countertop 3’, ’go to diningtable 1’, ’go
to drawer 1’, ’go to drawer 2’, ’go to drawer 3’, ’go to drawer 4’, ’go to garbagecan 1’, ’go
to microwave 1’, ’go to sinkbasin 1’, ’go to stoveburner 1’, ’go to stoveburner 2’, ’go to
stoveburner 3’, ’go to stoveburner 4’, ’go to toaster 1’, ’inventory’, ’look’, ’move pan 1 to
fridge 1’]
Action taken now: move pan 1 to fridge 1
Predict the environment’s next observation exactly as the environment would print it, inside <obs> </
obs>.<|im_end|>
<|im_start|>assistant
```

(ii) World-model output $\hat { o } _ { t + 1 } ( a )$ for move pan 1 to fridge 1 (generated; identical to the true next observation)

You move the pan 1 to the fridge 1.   
(iii) World-model output $\hat { o } _ { t + 1 } ( a ) f o r$ cool pan 1 with fridge 1 (generated; identical to the   
true next observation)

(iv) Critic input $\hat { s } _ { t + 1 } ( a ) = C ( s _ { t } , a , \hat { o } _ { t + 1 } ( a ) )$ for move pan 1 to fridge 1 (decoded from the logged ids; the value head reads the last token)

<|im\_start|>system   
You are an expert agent operating in the ALFRED embodied household environment.<|im\_end|>   
<|im\_start|>user   
Your task is to: put a cool pot in countertop.   
Prior to this step, you have already taken 14 step(s). Below are the most recent 5 observations and   
the corresponding actions you took: [Observation 10: ’You arrive at countertop 1. On the   
countertop 1, you see a bread 1, and a spatula 1.’, Action 10: ’go to countertop 2’] [   
Observation 11: ’You arrive at countertop 2. On the countertop 2, you see a pan 1.’, Action 11:   
take pan 1 from countertop $2 ^ { \prime } ]$ [Observation 12: ’You pick up the pan 1 from the countertop 2.’,   
Action 12: ’go to countertop 3’] [Observation 13: ’You arrive at countertop 3. On the countertop   
3, you see a fork 3, a knife 1, and a mug 1.’, Action 13: ’go to fridge 1’] [Observation $\ u _ { 1 4 } : \mathbf { \Omega } ^ { \prime }$   
You arrive at fridge 1. The fridge 1 is open. In it, you see nothing.’, Action 14: ’move pan 1   
to fridge 1’]   
You are now at step 15 and your current observation is: You move the pan 1 to the fridge 1.   
Your admissible actions of the current situation are: [’close fridge 1’, ’cool pan 1 with fridge 1’,   
examine fridge 1’, ’examine pan $\mathrm { 1 ^ { \prime } ~ , ~ \gamma ~ ^ { \prime } ~ g  o ~ ~ t o }$ cabinet $\mathrm { ~ \mathbb ~ 1 ~ ' ~ } , \mathrm { ~ ~ ' ~ } { \sf g o }$ to cabinet 10’, ’go to cabinet 11’,   
go to cabinet 12’, ’go to cabinet $_ { 1 3 ^ { \prime } , ~ ^ { \prime } { \sf g o } }$ to cabinet 14’, ’go to cabinet 15’, ’go to cabinet   
16’, ’go to cabinet $\mathrm { 1 7 ^ { \prime } ~ , ~ \tau ~ ' ~ g o ~ t } \mathrm { o }$ cabinet ${ 1 8 ^ { \prime } , ~ ^ { \prime } 9 0 }$ to cabinet ${ 1 9 ^ { \prime } , ~ ^ { \prime } { 9 } } \mathrm { o }$ to cabinet $2 ^ { \prime } , \prime { \tt g o t o }$   
cabinet 20’, ’go to cabinet 21’, ’go to cabinet $2 2 ^ { \prime } , \mathrm { ~  ~ { ~ \prime ~ } g o ~ }$ to cabinet 23’, ’go to cabinet $3 ^ { \prime } , \mathrm { ~  ~ { ~ ' ~ } ~ } 9 0$   
to cabinet $4 ^ { \prime } , \prime$ go to cabinet $5 ^ { \prime } , \mathrm { ~  ~ { ~ ' ~ } ~ } { \mathrm { g o } }$ to cabinet $6 ^ { \prime } , \prime$ go to cabinet 7’, ’go to cabinet $8 ^ { \prime } , \prime _ { 9 0 }$   
to cabinet ${ \mathfrak { s } } ^ { \prime } , \prime { \mathfrak { g } } { \mathfrak { o } }$ to coffeemachine $\mathrm { ~ \mathbb ~ 1 ~ } ^ { \prime } , \mathrm { ~ \tiny ~ \cdot ~ } { \mathsf { g o } }$ to countertop $1 ^ { \prime } , \mathrm { ~  ~ { ~ ' ~ } ~ } 9 0 \mathrm { ~  ~ { ~ t ~ o ~ } ~ }$ countertop $2 ^ { \prime } , \prime { \tt g o t o }$   
countertop $3 ^ { \prime } , \mathrm { ~  ~ { ~ ' ~ } ~ } 9 0$ to diningtable $\mathrm { ~ \mathbb ~ { ~ 1 ~ } ~ } ^ { \prime } ~ , ~ ^ { \prime } \mathrm { ~ \mathbb { g } \circ ~ \Delta ~ t \circ ~ }$ drawer $1 ^ { \prime } , ~ \dot { ^ { \prime } } { _ { \mathtt { g o } } }$ to drawer 2’, ’go to drawer 3’, ’go   
to drawer 4’, ’go to garbagecan $\mathrm { ~ \mathbb ~ 1 ~ ' ~ } , \mathrm { ~ ~ ' ~ } \mathrm { g o }$ to microwave $1 ^ { \prime } , \bar { \prime } _ { \mathfrak { g } ( }$ o to sinkbasin $\mathrm { ~ \mathbb ~ 1 ~ } ^ { \prime } , \mathrm { ~ \tiny ~ \cdot ~ } { \mathsf { g o } }$ to stoveburner   
1’, ’go to stoveburner $2 ^ { \prime } , ~ ^ { \prime } { \mathfrak { g } } \circ$ to stoveburner $3 ^ { \prime } , \mathrm { ~  ~ { ~ ' ~ } ~ } { \mathrm { g o } }$ to stoveburner $^ { 4 ^ { \prime } } , \quad ^ { \prime } { \mathsf { g o } }$ to toaster 1’, ’   
inventory’, ’look’, ’move pan 1 to fridge 1’].   
Now it’s your turn to take an action. First reason briefly (at most two sentences) about the current   
situation inside <think> </think> tags. Then choose one admissible action for the current step   
and present it inside <action> </action> tags.<|im\_end|>   
<|im\_start|>assistant

<table><tr><td>Field of the critic input</td><td>Source</td></tr><tr><td>system message</td><td>fixed (prompts.SYSTEM)</td></tr><tr><td>&#x27;Your task is to: &lt;task&gt;&#x27; &#x27;Prior to this step, you have already</td><td>copied from h (same task string)</td></tr><tr><td>taken &lt;t+1&gt; step(s)′</td><td>updated by the program (len(history) after appending (obs, a))</td></tr><tr><td>&#x27;Below are the most recent &lt;k&gt; observations ...: [Observation i:..</td><td>updated by the program: last 5 pairs of history + (real current observation, candidate action); all texts copied, none generated</td></tr><tr><td>Action i: ...]&#x27;</td><td></td></tr><tr><td>&#x27;You are now at step &lt;t+2&gt; and your current observation is: &lt;o_hat&gt;&#x27;</td><td>step number: program; &lt;o_hat&gt;: GENERATED by the WM</td></tr><tr><td>&#x27;Your admissible actions of the current situation are:[...]&#x27;</td><td>COPIED from h (the list before the action; not predicted, not updated)</td></tr><tr><td>instruction sentence</td><td>fixed template</td></tr></table>

ScienceWorld: training iteration 59, step 24

Run 1 of WM-PPO (policy channel only); models of iteration 58; training game chemistrymix-paint-secondary-color|16. The policy sampled mix bowl, mix bowl, mix bowl, focus on orange paint; mix bowl was executed in the chain and focus on orange paint in a branch from the same state. Logged difference of transition values $\hat { q } ( \mathrm { c h a i n } ) - \hat { q } ( \mathrm { b r a n c h } ) = + 0 . 1 5 3$ (margin bin 3 of 3); real returns 1 (chain) and 1 (branch). Both predictions equal the observations the environment printed; this is not representative of predictions that involve content the agent has not seen.

(i) World-model input $( x _ { t } , a )$ for candidate mix bowl (rebuilt token-exactly from the logged record; the inputfor the other candidate differs only in its last line)

You are a world model of the ScienceWorld text environment.<|im\_end|>   
<|im\_start|>user   
Task: use chemistry to create orange paint. When you are done, focus on the orange paint.   
Recent actions: [Step 1: action ’teleport to hallway’] [Step 2: action ’teleport to art studio’] [   
Step 3: action ’mix cup containing red paint and cup containing yellow paint’] [Step 4: action   
mix cup containing red paint and cup containing yellow paint’] [Step 5: action ’mix cup   
containing red paint and cup containing yellow paint’] [Step 6: action ’mix cup containing red   
paint and cup containing yellow paint’] [Step 7: action ’pour cup containing red paint into bowl   
’] [Step 8: action ’pour cup containing yellow paint into bowl’]   
Current observation:   
You pour the contents of the wood cup into the bowl.   
[Current room]   
This room is called the art studio. In it, you see:   
a large cupboard. The large cupboard door is closed.   
a substance called air   
a table. On the table is: a bowl (containing red paint, yellow paint).   
a wood cup (containing blue paint)   
a wood cup (containing nothing)   
a wood cup (containing nothing)   
the agent   
You also see:   
A door to the hallway (that is open)   
[Inventory]   
In your inventory, you see:   
an orange   
[score: 40]   
Objects the agent can refer to: [agent, air, art studio, blue paint, bowl, cup containing blue paint,   
cup containing nothing in art studio, cup on room, cupboard, door, hallway, red paint, table,   
yellow paint]   
Action taken now: mix bowl   
Predict the environment’s next observation exactly as the environment would print it (action feedback,   
[Current room], [Inventory] and [score] lines), inside <obs> </obs>.<|im\_end|>   
<|im\_start|>assistant

(ii) World-model output $\hat { o } _ { t + 1 } ( a )$ for mix bowl (generated; identical to the true next observation)

Yellow and red paint mix to produce orange paint.   
[Current room]   
This room is called the art studio. In it, you see:   
a large cupboard. The large cupboard door is closed.   
a substance called air   
a table. On the table is: a bowl (containing orange paint).   
a wood cup (containing blue paint)   
a wood cup (containing nothing)   
a wood cup (containing nothing)   
the agent   
You also see:   
A door to the hallway (that is open)   
[Inventory]   
In your inventory, you see:   
an orange   
[score: 50]

(iii) World-model output $\hat { o } _ { t + 1 } ( a )$ for focus on orange paint (generated; identical to the true next observation)

No known action matches that input.   
[Current room]   
This room is called the art studio. In it, you see:   
a large cupboard. The large cupboard door is closed.   
a substance called air   
a table. On the table is: a bowl (containing red paint, yellow paint).   
a wood cup (containing blue paint)   
a wood cup (containing nothing)   
a wood cup (containing nothing)   
the agent   
You also see:   
A door to the hallway (that is open)   
[Inventory]   
In your inventory, you see:   
an orange   
[score: 40]

(iv) Critic input $\hat { s } _ { t + 1 } ( a ) = C ( s _ { t } , a , \hat { o } _ { t + 1 } ( a ) )$ for mix bowl (decoded from the logged ids; the value head reads the last token)

<|im\_start|>system   
You are an expert agent operating in the ScienceWorld environment, a text-based simulator of   
elementary science experiments.<|im\_end|>   
<|im\_start|>user   
Your task is to: use chemistry to create orange paint. When you are done, focus on the orange paint.

Rules: ’focus on OBJ’ declares which object the task is about and cannot be undone; focusing on a   
wrong object ends the task as a failure, so explore first and focus only on the object the task   
asks about. OBJ must be an object you can refer to now (listed below); LOC is a room name (   
teleport moves you there directly). Answering questions such as ’which one ...’ is done by   
focusing on the answer object.   
Prior to this step, you have already taken 24 step(s). Below are the most recent 5 observations and   
the corresponding actions you took:   
[Observation 20: ’No known action matches that input.   
[Current room]   
This room is called the art studio. In it, you see:   
a large cupboard. The large cupboard door is closed.   
a substance called air   
a table. On the table is: a bowl (containing nothing).   
a wood cup (containing blue paint)   
a wood cup (containing red paint)   
a wood cup (containing yellow paint)   
the agent   
You also see:   
A door to the hallway (that is open)   
[Inventory]   
In your inventory, you see:   
an orange   
[score: 30]’, Action 20: ’mix cup containing red paint and cup containing yellow paint’]   
[Observation 21: ’No known action matches that input.   
[Current room]   
This room is called the art studio. In it, you see:   
a large cupboard. The large cupboard door is closed.   
a table. On the table is: a bowl (containing nothing).   
a wood cup (containing blue paint)   
a wood cup (containing red paint)   
a wood cup (containing yellow paint)   
the agent   
You also see:   
A door to the hallway (that is open)   
[Inventory]   
In your inventory, you see:   
an orange   
[score: 30]’, Action 21: ’mix cup containing red paint and cup containing yellow paint’]   
[Observation 22: ’No known action matches that input.   
[Current room]   
This room is called the art studio. In it, you see:   
a large cupboard. The large cupboard door is closed.   
a substance called air   
a table. On the table is: a bowl (containing nothing).   
a wood cup (containing blue paint)   
a wood cup (containing red paint)   
a wood cup (containing yellow paint)   
the agent   
You also see:   
A door to the hallway (that is open)   
[Inventory]   
In your inventory, you see:   
an orange   
[score: 30]’, Action 22: ’pour cup containing red paint into bowl’]   
[Observation 23: ’You pour the contents of the wood cup into the bowl.   
[Current room]   
This room is called the art studio. In it, you see:   
a large cupboard. The large cupboard door is closed.   
a substance called air   
a table. On the table is: a bowl (containing red paint).   
a wood cup (containing blue paint)   
a wood cup (containing nothing)   
a wood cup (containing yellow paint)   
the agent   
You also see:   
A door to the hallway (that is open)   
[Inventory]   
In your inventory, you see:   
an orange   
[score: 30]’, Action 23: ’pour cup containing yellow paint into bowl’]   
[Observation 24: ’You pour the contents of the wood cup into the bowl.   
[Current room]   
This room is called the art studio. In it, you see:   
a large cupboard. The large cupboard door is closed.   
a substance called air   
a table. On the table is: a bowl (containing red paint, yellow paint).   
a wood cup (containing blue paint)   
a wood cup (containing nothing)   
a wood cup (containing nothing)   
the agent   
You also see:   
A door to the hallway (that is open)   
[Inventory]   
In your inventory, you see:

an orange   
[score: 40]’, Action 24: ’mix bowl’]   
You are now at step 25 and your current observation is:   
Yellow and red paint mix to produce orange paint.   
[Current room]   
This room is called the art studio. In it, you see:   
a large cupboard. The large cupboard door is closed.   
a substance called air   
a table. On the table is: a bowl (containing orange paint).   
a wood cup (containing blue paint)   
a wood cup (containing nothing)   
a wood cup (containing nothing)   
the agent   
You also see:   
A door to the hallway (that is open)   
[Inventory]   
In your inventory, you see:   
an orange   
[score: 50]   
Action templates: [look around, look at OBJ, look in OBJ, inventory, task, teleport to LOC, go to LOC,   
open OBJ, close OBJ, activate OBJ, deactivate OBJ, pick up OBJ, put down OBJ, move OBJ to OBJ,   
pour OBJ in OBJ, dunk OBJ in OBJ, mix OBJ, use OBJ on OBJ, connect OBJ to OBJ, disconnect OBJ,   
eat OBJ, flush OBJ, read OBJ, focus on OBJ, wait, wait1].   
Objects you can refer to: [agent, air, art studio, blue paint, bowl, cup containing blue paint, cup   
containing nothing in art studio, cup on room, cupboard, door, hallway, red paint, table, yellow   
paint].   
Now it’s your turn to take an action. First reason step by step (at most two sentences) inside <think>   
</think>, then give exactly one action inside <action> </action>.<|im\_end|>   
<|im\_start|>assistant

<table><tr><td>Field of the critic input</td><td>Source</td></tr><tr><td>system message</td><td>fixed (prompts.SYSTEM)</td></tr><tr><td>&#x27;Your task is to: &lt;task&gt;&#x27;</td><td>copied from h</td></tr><tr><td>&#x27;Rules:...&#x27;</td><td>fixed (prompts.RULES)</td></tr><tr><td>&#x27;Prior to this step, you have already taken &lt;t+1&gt; step(s) ...&#x27;</td><td>updated by the program</td></tr><tr><td>&#x27;[Observation i: ..., Action i: ...]&#x27;</td><td>updated by the program: window shifted by one; newest pair = (real current</td></tr><tr><td>(last 5) &#x27;You are now at step &lt;t+2&gt; and your</td><td>observation, candidate action); copied texts step number: program; &lt;o_hat&gt; (action feedback, [Current room], [Inventory],</td></tr><tr><td>current observation is:\n&lt;o_hat&gt;&#x27;</td><td>[score: S], optional [task completed]/[task failed]): GENERATED by the WM</td></tr><tr><td>&#x27;Action templates:[...]&#x27;</td><td>fixed (prompts.TEMPLATES)</td></tr><tr><td>&#x27;objects you can refer to:[...]&#x27;</td><td>COPIED from h (referable objects before the action; not predicted, not</td></tr><tr><td></td><td></td></tr><tr><td>instruction sentence</td><td>updated) fixed template</td></tr></table>