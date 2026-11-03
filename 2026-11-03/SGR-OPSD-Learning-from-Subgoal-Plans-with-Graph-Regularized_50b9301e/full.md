# SGR-OPSD: Learning from Subgoal Plans with Graph-Regularized Credit Assignment for Agentic Reinforcement Learning

Shu Liu<sup>1,2\*</sup>, Yang Chong<sup>2</sup>, WenShi Chen<sup>2</sup>, Man Lan<sup>1†</sup>

<sup>1</sup>School of Computer Science and Technology, East China Normal University <sup>2</sup>MeiTuan

shuliu@stu.ecnu.edu.cn, mlan@cs.ecnu.edu.cn

## Abstract

OPSD provides dense supervision for agentic reinforcement learning through privileged information feedback. However, existing privileged information often lacks explicit taskprogression planning, while teacher–student log-probability gaps reflect shifts in model preference rather than true task progress. To address these limitations, we propose Subgoal-Guided Graph-Regularized OPSD (SGR-OPSD), a framework that combines subgoal supervision with interaction-derived evidence for turn-level credit assignment. Specifically, SGR-OPSD uses LLM-generated subgoal plans to provide dense action-level guidance, converting them into advantage weights via a constrained, KL-regularized objective. To ground credit in observed outcomes, we construct a Subgoal Progress Graph that estimates empirical returns from sampled transition frequencies. The graph compares empirical action returns against the state average to derive preferences that calibrate the teacher’s credit weights. Evaluated on ALF-World, WebShop, and Search-QA across two model scales (Qwen2.5-3B and 7B), SGR-OPSD consistently outperforms GRPO and strong self-distillation baselines—improving success rates over GRPO by up to 7.4 percentage points and reaching 85.8% on ALFWorld with Qwen2.5-7B. Ablation studies confirm that subgoal guidance and graph-based weight calibration are key to these performance gains.

## 1 Introduction

Reinforcement learning (RL) enables LLM-based agents to solve complex tasks through multi-turn interactions, ranging from household manipulation to online shopping (Shridhar et al., 2020; Yao et al., 2022). Effective training requires fine-grained supervision to evaluate intermediate decisions within a trajectory. On-Policy Self-Distillation (OPSD) (Zhao et al., 2026a) provides dense feedback by conditioning a teacher model on privileged training-time information to score student-sampled actions (Lv et al., 2026; Lan et al., 2026). This feedback complements the sparse outcome rewards used in methods like Group Relative Policy Optimization (GRPO) (Shao et al., 2024), enabling the student to internalize inferenceunavailable guidance. OPSD thus offers a promising framework for dense supervision in agentic RL.

![](images/90251a1619324b5c2b83e3cbc513d06436403ff1552208b3a197d8fe3c47b06c.jpg)  
Figure 1: Overview of SGR-OPSD. (a) Subgoal-guided OPSD produces credit weights, which are jointly calibrated using subgoal progress graph evidence. (b) The figure shows a real example in which calibration shifts favorable credit from the sink to the dining table.

Existing methods supply privileged information through trajectory-derived skills or reference answers, integrating teacher feedback via advantage reweighting or auxiliary distillation objectives (Wang et al., 2026a; Yang et al., 2026a; Lu et al., 2026). AgentOPSD further accumulates likelihood evidence across turns, while TASPO maps action-level support into bounded advantage weights (Wang et al., 2026b; Yang et al., 2026b). However, two key gaps remain between such supervision and true task progress. First, privileged guidance often lacks an explicit, ordered plan for the task at hand. For “cool bread and place it on the countertop”, a structured plan—locate the bread, cool it, then place it—makes the required progression explicit. Second, teacher–student logprobability gaps measure shifts in model preference rather than progress made through real interactions; even plan-conditioned feedback can incorrectly favor actions that stray from successful continuations. Thus, effective credit assignment requires both taskspecific planning information and a mechanism that grounds its influence in execution evidence. To bridge these gaps, our approach uses subgoal plans to structure teacher guidance and uses evidence from the Subgoal Progress Graph to calibrate the resulting credit.

We propose SGR-OPSD, illustrated in Figure 1. An LLM generates a fixed subgoal plan from the task description alone, providing the scoring teacher with explicit task-progression guidance. We derive a constrained KL-regularized rule that converts action-level teacher–student likelihood evidence into bounded exponential weights on GRPO advantages. The rule accounts for advantage signs and enforces a fixed token-weighted budget within each trajectory, establishing a teacher reference for turn-level credit. To calibrate these weights, we construct a Subgoal Progress Graph from student rollouts and derive action preferences from empirical Q − V: each action’s discounted return relative to the visit-weighted state average. A joint objective incorporates these preferences while regularizing the allocation toward the teacher reference, connecting plan-guided supervision with evidence of task progress. The trained student requires neither the plan nor the graph at inference. We evaluate SGR-OPSD on ALFWorld, WebShop, and Search-QA with Qwen2.5-3B and Qwen2.5-7B. On the 7B model, it reaches 85.8% success on ALFWorld and 78.2% on WebShop, improving over GRPO by 7.4 and 7.6 percentage points, respectively.

Our contributions are as follows:

• We propose leveraging subgoal plans as privileged supervision and derive a constrained KLregularized formulation to allocate turn-level credit via bounded exponential weights.

• We develop a Subgoal Progress Graph that derives action preferences from empirical return estimates to calibrate teacher-reference weights, while preserving advantage signs and a fixed token-weighted budget per trajectory.

• Extensive experiments across three environments and two model scales demonstrate the superiority of SGR-OPSD, outperforming GRPO by up to

7.6 percentage points and achieving an 85.8% success rate on ALFWorld.

## 2 Related Works

## 2.1 On-Policy Self-Distillation for LLMs

Self-distillation derives the teacher from the student and conditions it on privileged training-time information (Wang et al., 2026a,b). Existing works integrate diverse privileged signals via distinct mechanisms: Skill-SD leverages trajectory-derived skills (Wang et al., 2026a), RLSD scales token updates using log-probability gaps between an answer-conditioned teacher and the student (Yang et al., 2026a), and SDAR gates SkillBank guidance alongside RL (Lu et al., 2026). However, these privileged signals often contain irrelevant noise and lack explicit task-progress planning. To address this, SGR-OPSD uses a fixed, LLM-generated subgoal plan derived solely from the task description as privileged information and applies a constrained KL-regularized objective to transform action-level evidence into bounded turn weights.

## 2.2 Credit Assignment for Multi-Turn Agents

While GRPO provides trajectory-level advantages under outcome rewards, it fails to distinguish the contributions of individual turns (Shao et al., 2024). Recent lines of work refine this coarse outcome signal: AgentOPSD accumulates privileged likelihood evidence via recursive belief updates (Wang et al., 2026b), TASPO converts action-level support into bounded advantage weights (Yang et al., 2026b), and AlignOPSD aligns sibling contexts to allocate credit across spans and turns (Chen et al., 2026). Although such teacher likelihood preferences offer dense feedback, they lack explicit measurements of task progress. To bridge this gap, SGR-OPSD constructs a Subgoal Progress Graph from student rollouts, leveraging graph-derived action comparisons to calibrate turn-level credit from a subgoalconditioned teacher.

## 3 Method

SGR-OPSD combines plan-conditioned teacher preferences with rollout-derived outcome evidence for turn-level credit assignment. Constrained KL weighting converts teacher feedback into reference weights, which the Subgoal Progress Graph calibrates through action comparisons (Figure 2).

![](images/370d02073524066546e4bd01143d7c6dde8190b567bebe6dfc397a800c595528.jpg)  
Figure 2: Overview of SGR-OPSD. Subgoal plans guide teacher feedback, which constrained KL weighting converts into turn-level credit weights. The Subgoal Progress Graph uses student trajectories to derive action preferences and calibrate these weights. The calibrated weights $w _ { i } ^ { * }$ redistribute GRPO credit while preserving advantage signs and a fixed token-weighted budget.

## 3.1 Background

We consider multi-turn tasks where an agent interacts with an environment to complete an instruction x. Given a task x, the rollout policy $\pi _ { \theta _ { \mathrm { o l d } } }$ samples K trajectories $\{ \tau _ { i } \} _ { i = 1 } ^ { K }$ . Each trajectory τ<sub>i</sub> consists of $N _ { i }$ turns indexed by t, with k indexing individual tokens within a response. At turn t, given context $h _ { i , t } ^ { S }$ , the agent generates a response $y _ { i , t }$ containing action ${ a } _ { i , t }$ and receives the next observation.

Outcome-based policy optimization assigns each turn a base advantage $A _ { i , t }$ , where positive values reinforce the sampled turn and negative values discourage it. Self-distillation supplements this sparse outcome feedback by re-scoring student-sampled trajectories using privileged information. Both scoring branches share the pre-update parameters $\theta _ { \mathrm { o l d } }$ , so their likelihood difference reflects the effect of privileged information.

## 3.2 Subgoal-Guided Credit Weights

Subgoal-conditioned supervision. Subgoal plans make the intermediate requirements of task completion explicit. We employ a fixed, ordered plan z(x), generated by DeepSeek-V3.1 (DeepSeek-AI, 2024) solely from the task instruction x, as privileged information. $z ( x )$ combines task-specific intermediate goals and ordering constraints without requiring prior task interactions or successful trajectories (Appendix A). Figure 2(b) illustrates how this plan conditions teacher scoring to yield subgoal-guided teacher–student log-likelihood gaps.

Specifically, the teacher branch receives the privileged context $h _ { i , t } ^ { T }$ augmented with $z ( x )$ , whereas the student operates on the standard context $h _ { i , t } ^ { S }$ Token probabilities for branch $B \in \{ T , S \}$ are defined as $\pi _ { i , t , k } ^ { B } = \pi _ { \mathrm { o l d } } ( y _ { i , t , k } \mid h _ { i , t } ^ { B } , y _ { i , t , < k } )$ , with both branches retaining identical response prefixes $y _ { i , t , < k }$ . Summing the token-level log-probability differences over the token positions $\mathcal { A } _ { i , t }$ of the executed action yields the action-level gap:

$$
d _ { i , t } = \sum _ { k \in \mathcal { A } _ { i , t } } \left( \log \pi _ { i , t , k } ^ { T } - \log \pi _ { i , t , k } ^ { S } \right) .\tag{1}
$$

The gap $d _ { i , t }$ measures how the plan changes the action’s log-likelihood, providing dense action-level evidence for credit assignment.

Constrained KL weighting. Subgoal support should strengthen positive updates and soften negative ones. We derive turn weights by balancing signed teacher evidence against KL regularization toward unit weights:

$$
w ^ { \mathrm { r e f } } = \arg \operatorname* { m a x } _ { w \in \mathcal { W } } \{ \mathcal { U } ( w ) - T D _ { p } ( w \| \mathbf { 1 } ) \} ,\tag{2}
$$

where $w = ( w _ { i , t } ) _ { i , t }$ collects the turn weights and $w ^ { \mathrm { r e f } }$ is their optimized teacher reference. The feasible set W bounds each weight by $1 - \epsilon \leq w _ { i , t } \leq$ $1 + \epsilon ( 0 \leq \epsilon < 1 )$ , fixes $w _ { i , t } = 1$ when $A _ { i , t } = 0$ and imposes a token-weighted budget for each trajectory:

$$
\sum _ { t = 1 } ^ { N _ { i } } p _ { i , t } w _ { i , t } = 1 , \qquad p _ { i , t } = \frac { m _ { i , t } } { \sum _ { r = 1 } ^ { N _ { i } } m _ { i , r } } ,\tag{3}
$$

where $m _ { i , t }$ counts effective response tokens in turn t. Thus, $p _ { i , t }$ weights turns by their token share, and the budget redistributes emphasis while keeping the weighted mean at one. Within this budget, the signed teacher-support term is

$$
\mathcal { U } ( w ) = \sum _ { i , t } p _ { i , t } s _ { i , t } d _ { i , t } w _ { i , t } ,\tag{4}
$$

where $d _ { i , t }$ is the action-level gap in Equation 1 and $s _ { i , t } = \mathrm { s i g n } ( A _ { i , t } )$ . The sign factor makes stronger teacher support favor larger weights for positive advantages and smaller weights for negative ones.

The KL term limits departures from unit weights, with $T > 0$ controlling its strength. We use the token-weighted generalized KL divergence

$$
D _ { p } ( u \| v ) = \sum _ { i , t } p _ { i , t } \left( u _ { i , t } \log \frac { u _ { i , t } } { v _ { i , t } } - u _ { i , t } + v _ { i , t } \right) ,\tag{5}
$$

where u is a positive weight vector and v its positive reference, with entries $u _ { i , t }$ and $v _ { i , t }$ . In Equation 2, these are w and the all-ones vector 1, respectively. For $A _ { i , t } \neq 0$ , this objective yields

$$
w _ { i , t } ^ { \mathrm { r e f } } = \mathrm { c l i p } \left( \mathrm { e x p } \left( \frac { s _ { i , t } d _ { i , t } - \nu _ { i } } { T } \right) , 1 - \epsilon , 1 + \epsilon \right) ,\tag{6}
$$

where $\nu _ { i }$ is the trajectory-specific multiplier enforcing the budget (Appendix B). These bounded weights provide the reference for calibration by the Subgoal Progress Graph.

## 3.3 Subgoal Progress Graph

To complement teacher preferences with outcome evidence, we merge matched states across sampled rollouts into a Subgoal Progress Graph. Transition counts define empirical values:

$$
\begin{array} { r l } & { \widehat { Q } ( v , a ) = \widehat { \mathbb { E } } [ r + \gamma I \widehat { V } ( v ^ { \prime } ) \mid v , a ] , } \\ & { \quad \widehat { V } ( v ) = \displaystyle \sum _ { a } \widehat { \pi } ( a \mid v ) \widehat { Q } ( v , a ) . } \end{array}\tag{7}
$$

Here $v , v ^ { \prime }$ are current and successor nodes, $r \in$ {0, 1} is the success reward, and $\gamma \in \mathsf { ( 0 , 1 ) }$ discounts continuation. The expectation averages recorded transitions; ${ \widehat { \pi } } ( a \mid v )$ is the fraction of visits to v taking action a. The indicator I permits continuation only for nonterminal transitions with remaining budget and a successor recorded as a source node.

The relative graph signal is

$$
\begin{array} { r l } & { \delta _ { i , t } = \widehat { Q } ( v _ { i , t } , a _ { i , t } ) - \widehat { V } ( v _ { i , t } ) , } \\ & { g _ { i , t } = \mathrm { c l i p } ( \delta _ { i , t } , - b _ { G } , b _ { G } ) . } \end{array}\tag{8}
$$

Here $v _ { i , t }$ is the pre-action node, $b _ { G } > 0$ bounds the signal, and positive $\delta _ { i , t }$ indicates above-average empirical return.

Graph-derived action comparisons. The Subgoal Progress Graph determines action ordering; calibration translates these preferences into bounded weight adjustments. Beyond node matching for value estimation, comparisons require matching plans, student contexts, and remaining budgets, with all associated visits scoreable and having nonzero advantages:

$$
\bar { g } ( a ) = H _ { a } ^ { \top } g , \qquad a \succ b \iff \bar { g } ( a ) > \bar { g } ( b ) .\tag{9}
$$

Here $g$ collects turn signals; $H _ { a }$ averages unique visits equally within trajectories, then across trajectories. Preferred pairs $e = ( a , b )$ form E. The same $H _ { a }$ aggregates credit corrections below; graphscore magnitudes do not set the correction margin.

Joint credit calibration. Graph-preferred actions should receive stronger reinforcement for positive advantages and softer penalties for negative ones. We parameterize the turn weights as

$$
w _ { i , t } ( c ) = 1 + s _ { i , t } c _ { i , t } .\tag{10}
$$

Here c collects the credit corrections $c _ { i , t }$ and $w ( c )$ their resulting weights. With $s _ { i , t }$ defined above, increasing $c _ { i , t }$ favors the action under either advantage sign. We balance teacher retention against graph preferences:

$$
\begin{array} { r } { c ^ { * } = \arg \operatorname* { m i n } _ { c } \big \{ { T _ { G } D _ { p } ( w ( c ) \| w ^ { \mathrm { r e f } } ) + \beta \mathcal { R } _ { G } ( c ) } \big \} , } \end{array}\tag{11}
$$

where $D _ { p }$ is defined in Equation $5 , T _ { G } > 0$ controls anchoring to the teacher reference $w ^ { \mathrm { r e f } }$ , and $\beta \geq 0$ scales the graph preference penalty

$$
\mathcal { R } _ { G } ( c ) = \frac { 1 } { 2 } \sum _ { e \in \mathcal { E } } \rho _ { e } [ \mu - b _ { e } ^ { \top } c ] _ { + } ^ { 2 } .\tag{12}
$$

Here $b _ { e } ~ = ~ H _ { a } - H _ { b }$ for each preferred pair $e = ( a , b ) \in \mathcal { E }$ , so $b _ { e } ^ { \top } c$ is the difference in mean corrections. With $[ u ] _ { + } = \operatorname* { m a x } ( u , 0 )$ , the penalty encourages this difference to reach margin $\mu \geq 0$ The weight $\rho _ { e }$ normalizes comparison count and coefficient scale (Appendix B). The optimization is subject to

$$
\begin{array} { r l r } {  { \sum _ { t = 1 } ^ { N _ { i } } p _ { i , t } s _ { i , t } c _ { i , t } = 0 \quad \forall i , } } \\ & { \vert c _ { i , t } \vert \leq \epsilon , \quad } & { c _ { i , t } = 0 \mathrm { i f } A _ { i , t } = 0 . } \end{array}\tag{13}
$$

These constraints preserve the token-weighted budget and weight bounds, with zero-advantage turns unchanged. Graph comparisons therefore redistribute emphasis within each trajectory; trajectories without eligible comparisons retain their teacher reference exactly.

<table><tr><td></td><td colspan="7">ALFWorld</td><td colspan="7">SearchQA</td><td colspan="2">WebShop</td></tr><tr><td>Method</td><td>Pick Look</td><td></td><td>Clean</td><td>Heat</td><td>Cool</td><td>Pick2</td><td>Avg</td><td>NQ</td><td>Triv</td><td>Pop Hotp</td><td>2Wk</td><td>MuS</td><td>Bam</td><td>Avg</td><td>Score</td><td>Succ.</td></tr><tr><td>Closed-Source Models</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Qwen3.5-Plus</td><td>82.9</td><td>30.8</td><td>66.7</td><td>6.3</td><td>44.0</td><td>66.7</td><td>49.5</td><td>28.5 55.1</td><td>35.0</td><td>28.6</td><td>37.2</td><td>23.5</td><td>42.8</td><td>35.8</td><td>9.8</td><td>9.5</td></tr><tr><td>DeepSeek-V3.1</td><td>91.4</td><td>69.2</td><td>48.1</td><td>50.0</td><td>32.0</td><td>54.2</td><td>57.5</td><td>20.6 64.0</td><td>35.6</td><td>35.4</td><td>31.0</td><td>18.6</td><td>54.4</td><td>37.1</td><td>10.2</td><td>8.4</td></tr><tr><td colspan="10">Qwen2.5-3B-Instruct</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Vanilla</td><td>44.4</td><td>11.1</td><td>6.2</td><td>15.4</td><td>28.6</td><td>12.5</td><td>19.7</td><td>24.6 48.1</td><td>31.0</td><td>26.3</td><td>25.3</td><td>7.2</td><td>21.5</td><td>26.3</td><td>6.7</td><td>0.8</td></tr><tr><td>GRPO</td><td>91.4</td><td>66.6</td><td>75.3</td><td>58.3</td><td>61.3</td><td>62.5</td><td>69.3 45.3</td><td>61.2</td><td>44.9</td><td>40.1</td><td>38.1</td><td>15.1</td><td>31.2</td><td>39.4</td><td>78.5</td><td>63.1</td></tr><tr><td>Skill-Prompt</td><td>51.7</td><td>66.7</td><td>48.4</td><td>0.0</td><td>4.3</td><td>10.0</td><td>30.2</td><td>23.7 46.2</td><td>30.6</td><td>24.4</td><td>22.1</td><td>7.5</td><td>12.5</td><td>23.9</td><td>0.2</td><td>0.8</td></tr><tr><td>OPSD</td><td>48.8</td><td>41.7</td><td>16.7</td><td>0.0</td><td>15.8</td><td>16.7</td><td>23.3</td><td>0.1 0.1</td><td>0.1</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>11.3</td><td>3.1</td></tr><tr><td>GRPO+OPSD</td><td>95.2</td><td>59.2</td><td>84.0</td><td>60.4</td><td>66.7</td><td>61.1</td><td>71.1</td><td>45.0 61.2</td><td>44.7</td><td>39.4</td><td>40.0</td><td>14.8</td><td>30.1</td><td>39.3</td><td>80.6</td><td>63.3</td></tr><tr><td>Skill-SD</td><td>81.8</td><td>43.3</td><td>56.9</td><td>39.8</td><td>42.0</td><td>45.3</td><td>51.5</td><td>44.1 60.7</td><td>43.8</td><td>39.3</td><td>40.2</td><td>16.3</td><td>34.4</td><td>39.8</td><td>75.6</td><td>58.8</td></tr><tr><td>RLSD</td><td>88.6</td><td>56.4</td><td>76.5</td><td>75.0</td><td>60.0</td><td>58.3</td><td>69.1</td><td>44.0 59.6</td><td>43.0</td><td>40.3</td><td>43.0</td><td>18.2</td><td>34.4</td><td>40.4</td><td>79.7</td><td>62.4</td></tr><tr><td>SDAR</td><td>96.2</td><td>61.5</td><td>86.4</td><td>70.8</td><td>69.3</td><td>65.3</td><td>74.9</td><td>44.5 60.0</td><td>43.6</td><td>39.5</td><td>37.5</td><td>15.7</td><td>31.2</td><td>38.9</td><td>72.3</td><td>57.5</td></tr><tr><td>AgentOPSD</td><td>91.4</td><td>53.8</td><td>82.7</td><td>58.3</td><td>65.3</td><td>59.7</td><td>68.6</td><td>43.0 59.4</td><td>45.1</td><td>36.7</td><td>38.3</td><td>12.9</td><td>24.8</td><td>37.2</td><td>71.3</td><td>54.7</td></tr><tr><td>AlignOPSD</td><td>82.9</td><td>53.8</td><td>70.4</td><td>50.0</td><td>80.0</td><td>58.3</td><td>65.9</td><td>44.7 60.3</td><td>44.7</td><td>39.8</td><td>40.3</td><td>14.5</td><td>28.8</td><td>39.0</td><td>75.9</td><td>58.6</td></tr><tr><td>SGR-OPSD</td><td>97.1</td><td>69.2</td><td>81.5</td><td>75.0</td><td>68.0</td><td>70.8</td><td>76.9</td><td>45.4 62.0</td><td>45.4</td><td>40.3</td><td>41.5</td><td></td><td>15.2 34.6</td><td>40.6</td><td>85.5</td><td>68.8</td></tr><tr><td colspan="10">Qwen2.5-7B-Instruct</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Vanilla</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GRPO</td><td>36.1</td><td>22.2</td><td>3.1</td><td>0.0</td><td>0.0</td><td>0.0</td><td>10.2 25.2</td><td>50.8</td><td>29.5</td><td>29.0</td><td>29.0</td><td>10.4</td><td>23.7</td><td>28.2</td><td>5.9</td><td>1.6 70.6</td></tr><tr><td>Skill-Prompt</td><td>90.5</td><td>69.2</td><td>88.9</td><td>79.2</td><td>73.3</td><td>68.1</td><td>78.2</td><td>48.2 64.0</td><td>45.8</td><td>43.9</td><td>45.0</td><td>20.2</td><td>43.2</td><td>44.3</td><td>81.7</td><td></td></tr><tr><td>OPSD</td><td>51.7</td><td>50.0</td><td>32.3</td><td>5.3</td><td>4.3</td><td>0.0</td><td>23.9</td><td>30.9 52.1</td><td>32.7</td><td>32.7</td><td>27.9</td><td>12.7</td><td>66.1</td><td>36.4</td><td>1.7</td><td>0.8</td></tr><tr><td>GRPO+OPSD</td><td>50.0</td><td>60.0</td><td>22.7</td><td>21.4</td><td>17.6</td><td>9.5</td><td>30.2</td><td>8.8 8.6</td><td>17.5</td><td>2.5</td><td>4.2</td><td>0.5</td><td>1.2</td><td>6.2</td><td>4.5</td><td>2.3</td></tr><tr><td>Skill-SD</td><td>97.1 93.9</td><td>69.2</td><td>88.9</td><td>62.5</td><td>76.0</td><td>75.0</td><td>78.1</td><td>47.2 63.8</td><td>45.6</td><td>42.7</td><td>41.8</td><td>19.5</td><td>40.8</td><td>43.1</td><td>85.2</td><td>72.8</td></tr><tr><td></td><td>91.4</td><td>67.3</td><td>79.4</td><td>54.7</td><td>61.0</td><td>82.3</td><td>73.1</td><td>48.0 64.4</td><td>45.8</td><td>43.9</td><td>40.7</td><td>19.4</td><td>44.8</td><td>43.9</td><td>82.2</td><td>69.4</td></tr><tr><td>RLSD</td><td></td><td>69.2</td><td>76.5</td><td>77.1</td><td>56.0</td><td>58.3</td><td>71.4</td><td>47.2 62.8</td><td>46.1</td><td>44.4</td><td>45.8</td><td>20.6</td><td>39.2</td><td>43.7</td><td>85.6</td><td>74.1</td></tr><tr><td>SDAR AgentOPSD</td><td>94.3 96.2</td><td>71.8 69.2</td><td>84.0 91.4</td><td>93.8 83.3</td><td>72.0 72.0</td><td>58.3 69.4</td><td>79.0 80.3</td><td>47.9 64.4 45.5 60.4</td><td>47.6 42.5</td><td>44.4 41.7</td><td>46.2 33.3</td><td>20.8 18.6</td><td>44.8 40.8</td><td>45.2 40.4</td><td>84.5 83.0</td><td>72.4 74.0</td></tr></table>

Table 1: Results on ALFWorld, SearchQA, and WebShop (%); Avg denotes the macro-average. All baseline results are obtained from our own reruns. Bold and underline mark the best and second-best results within each backbone.

## 3.4 Policy Optimization

The calibrated weights redistribute outcome-based credit through

$$
\widetilde { A } _ { i , t } = A _ { i , t } w _ { i , t } ^ { * } .\tag{14}
$$

Here $w _ { i , t } ^ { * } = w _ { i , t } ( c ^ { * } )$ is the calibrated weight from Equation 11. The resulting advantage $\widetilde { A } _ { i , t }$ is shared by all effective response tokens in the turn, including reasoning tokens. The student minimizes

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { S G R } } ( \theta ) = \mathcal { L } _ { \mathrm { p o l i c y } } \Big ( \theta ; \sec ( \widetilde { A } ) \Big ) + \lambda _ { \mathrm { K L } } \mathcal { L } _ { \mathrm { K L } } ( \theta ) , } \end{array}\tag{15}
$$

Here θ denotes the student parameters, ${ \mathcal { L } } _ { \mathrm { p o l i c y } }$ is the base clipped policy loss, and sg stops gradients through the advantages. The term ${ \mathcal { L } } _ { \mathrm { K I } }$ regularizes

the student toward the reference policy with coefficient $\lambda _ { \mathrm { K L } }$ ; Teacher and graph guidance thus enter the policy update through $\widetilde { A }$

## 4 Experiments

We evaluate SGR-OPSD across three interactive environments and two model scales to examine whether subgoal-guided, graph-calibrated credit improves policy learning.

## 4.1 Experimental Setup

Benchmarks and metrics. For ALFWorld (Shridhar et al., 2020), we evaluate success on the 140- task seen split across six household task categories: Pick, Look, Clean, Heat, Cool, and Pick2. For WebShop (Yao et al., 2022), we evaluate product search and selection on 500 goals, reporting task score and full-task success rate. SearchQA follows Search-R1 (Jin et al., 2025); we report exactmatch accuracy on 51,713 questions across NQ, TriviaQA, PopQA, HotpotQA, 2WikiMultiHopQA,

MuSiQue, and Bamboogle. Together, these environments test sequential state changes, constraint satisfaction, and evidence gathering.

Baselines. We compare with Vanilla and Skill-Prompt as training-free references, GRPO (Shao et al., 2024) as the outcome-based learner, and OPSD (Zhao et al., 2026b), GRPO+OPSD, Skill-SD (Wang et al., 2026a), RLSD (Yang et al., 2026a), SDAR (Lu et al., 2026), and AgentOPSD (Wang et al., 2026b) as self-distillation baselines. Qwen3.5-Plus and DeepSeek-V3.1 provide additional closed-source model references.

Training configuration. We use the Qwen2.5- 3B-Instruct and Qwen2.5-7B-Instruct backbones and carry out three independent training runs with seeds {0, 42, 2026}. We train SGR-OPSD on 8 NVIDIA A100 GPUs for 150 policy updates, with eight rollouts per task and a learning rate of $1 0 ^ { - 6 }$ The maximum interaction lengths are 50, 15, and 4 turns for ALFWorld, WebShop, and SearchQA, respectively. We set $\epsilon = 0 . 1 , T = T _ { G } = 1$ β = 2, $\mu \ = \ 0 . 0 2$ , and $\gamma = 0 . 1$ . We provide subgoal-generation prompts in Appendix A and environment-specific training and evaluation settings in Appendix C.

## 4.2 Main Results

Stronger multi-turn task completion. SGR-OPSD leverages task-specific subgoals and Subgoal Progress Graph to calibrate turn-level credit redistribution. As shown in Table 1, this method consistently improves task completion across both model scales, achieving the highest ALFWorld macro-average and top performance on both Web-Shop metrics among all baselines. At 7B scale, it achieves 85.6% success on ALFWorld and 78.2% on WebShop. The performance gain over GRPO holds across both scales, with improvements of 7.4– 7.6 pp on ALFWorld and 5.7–7.6 pp on WebShop success, underscoring the benefit of augmenting outcome feedback with structured credit allocation.

Gains beyond existing Self-distillation methods. The advantage persists against GRPO+OPSD and the Skill-guided methods Skill-SD, SDAR, and AgentOPSD on both evaluated environments at each model scale. Relative to the strongest baseline for each backbone, SGR-OPSD improves the ALFWorld macro-average by 2.0–5.3 pp and Web-Shop success by 4.1–5.5 pp Its leading 7B results on Cool and Pick2 sub-tasks provide concrete examples: these tasks require completing an objectstate change or collecting two distinct objects before final placement. The results support the full method’s effectiveness on tasks whose success depends on satisfying intermediate requirements.

<table><tr><td>Method</td><td>ALFWorld SearchQA WebShop</td><td></td><td></td></tr><tr><td>SGR-OPSD</td><td>76.9</td><td>40.6</td><td>68.8</td></tr><tr><td>w/o Graph</td><td>77.4</td><td>39.5</td><td>64.8</td></tr><tr><td>w/o Teacher Signal</td><td>74.1</td><td>40.8</td><td>62.2</td></tr><tr><td>Skill instead of Subgoal</td><td>74.6</td><td>37.7</td><td>60.4</td></tr></table>

Table 2: Component ablations with Qwen2.5-3B-Instruct (%): ALFWorld macro-average seen success, SearchQA macro-average exact-match accuracy, and WebShop success rate.
<table><tr><td>Method</td><td>ALFWorld SearchQA WebShop</td><td></td><td></td></tr><tr><td>SGR-OPSD</td><td>85.6</td><td>46.4</td><td>78.2</td></tr><tr><td>w/o Graph</td><td>66.7</td><td>43.8</td><td>74.2</td></tr><tr><td>w/o Teacher Signal</td><td>79.9</td><td>44.4</td><td>72.6</td></tr><tr><td>Skill instead of Subgoal</td><td>70.9</td><td>44.2</td><td>68.4</td></tr></table>

Table 3: Component ablations with Qwen2.5-7B-Instruct (%), using the same metrics as Table 2.

Generalizability to retrieval-based reasoning. SearchQA tests the method in a different action space, where progress depends on retrieving evidence and resolving information dependencies. SGR-OPSD improves macro-average accuracy over GRPO by 1.2 and 2.1 points at 3B and 7B, respectively. It achieves the highest macro-average at both scales, exceeding the strongest baseline by 1.2 points at 7B, with the best 7B results on TriviaQA and MuSiQue. These gains show that the approach also benefits retrieval-based reasoning beyond environment manipulation.

## 4.3 Component Ablations

The ablations examine three components: the Subgoal Progress Graph, teacher evidence, and taskspecific subgoal plans (Tables 2 and 3). The metrics are ALFWorld macro-average seen success, SearchQA macro-average exact-match accuracy, and WebShop success rate.

Role of graph calibration. Relative to w/o Graph, which removes the Subgoal Progress Graph and directly uses $w ^ { \mathrm { r e f } }$ , the full method scores 4.0 pp higher on WebShop at both model scales, and 1.1/2.6 pp higher on SearchQA at 3B/7B. These consistent improvements support calibrating teacher evidence with rollout-derived progress preferences. ALFWorld exhibits a scale-dependent pattern: performance at 3B is comparable (76.9% vs. 77.4%), whereas the 7B variant yields a substantial 18.9 pp gain.

Role of teacher evidence. Teacher guidance complements preferences from the Subgoal Progress Graph on ALFWorld and WebShop. Compared to w/o Teacher Signal, which initializes graph calibration from unit weights, the full method achieves 2.8/5.7 pp higher success rates on ALF-World and 6.6/5.6 pp higher on WebShop at 3B/7B. This pattern underscores the necessity of pairing explicit task requirements with observed progress evidence. SearchQA shows a 2.0 pp gain at 7B and comparable scores at 3B (40.6% vs. 40.8%).

Choice of privileged guidance. Subgoal plans consistently outperform Skill guidance across all available comparisons, achieving gains of 8.4/9.8 pp on WebShop, 2.9/2.2 pp on SearchQA at 3B/7B, and 2.3 pp on ALFWorld at 3B, under an identical allocation framework. While both representations encode ordered workflows, these results demonstrate that instantiating guidance with task-specific objects, destinations, and intermediate state requirements is strictly superior (Section 4.4).

## 4.4 Subgoal Guidance Analysis

Task-specific requirements. Subgoal plans instantiate workflow rules with the task’s objects, destination, and required states. Table 4 illustrates this distinction for cleaning a soapbar and placing it on a countertop. The Skill rule specifies a general cleantask workflow; the Subgoal plan names the soapbar and countertop and makes placement conditional on the soapbar being clean and held. The contrast is concrete: the Skill rule refers to a “target object” and “final location,” while the plan specifies which object must reach which destination and in what state. These explicit completion conditions give teacher scoring a task-specific reference for assessing sampled actions. Through the teacher–student likelihood gap, the plan supplies credit evidence tied to the instruction’s intermediate requirements.

Compact privileged context. Across 601 ALF-World plans, Subgoal guidance averages 87.79 tokens, whereas the six task-type Skill inputs range from 861 to 1,168 tokens. Counts use the Qwen2.5- 3B-Instruct tokenizer without special tokens and exclude interaction history. The compact plan retains task-specific goals and preconditions, and is reused unchanged across turns. Appendix A provides the generation prompts and full comparison.

<table><tr><td>Skill: reusable workflow Always execute clean tasks in the fixed sequence:</td></tr><tr><td>(2) bring it to an available water source or sink to clean, (3) navigate to final location, (4) place object.</td></tr><tr><td>Subgoal: task-specific requirement</td></tr><tr><td>5. Put the clean soapbar on the countertop Preconditions: Countertop is found and reachable, soapbar is</td></tr></table>

Table 4: Skill and Subgoal excerpts for the same task, reformatted for readability. Full context is in Appendix A.  
![](images/1337c59095420c87a9112e1ed7d5c065a3393651d56b07810f235bf9ee37d14b.jpg)  
Figure 3: Validation success of GRPO, AgentOPSD, and SGR-OPSD with Qwen2.5-3B-Instruct on ALF-World and WebShop.

## 4.5 Training Dynamics

We track validation performance every five training updates, comparing methods at 3B and model scales for SGR-OPSD.

Continued gains in later training. On ALF-World, SGR-OPSD sustains its performance lead across the final four validation checkpoints (updates 135–150), finishing 11.7 and 17.2 pp ahead of GRPO and AgentOPSD, respectively (Figure 3). While the competing methods overlap during early and intermediate training, SGR-OPSD establishes a sustained late-stage advantage, ultimately reaching 82.0% success. This trajectory confirms that the final improvement reflects a consistent separation maintained throughout the closing updates. On WebShop, all three methods progress from nearzero success to over 50%; their final evaluations under the standard protocol are detailed in Table 1.

Learning across scales and environments. Figure 4 illustrates performance gains across both model scales in all three environments. The 7B model accelerates earlier on ALFWorld and Web-Shop, whereas the 3B variant exhibits steady latestage gains. On SearchQA, macro-average Exact Match (EM) accuracy rises from 26.1% to 39.5% at 3B scale and from 28.8% to 43.2% at 7B scale between updates 5 and 150. The 7B trajectory maintains a clear margin over the 3B curve throughout training on this benchmark, with both scales reaching stability in the closing updates.

![](images/a667226864f0b9bb714ae211fa10cd460d09376af2abe3adaa780e4a8556d22d.jpg)

![](images/18333c2c8985036c6181d7365ede52795471d639bb91daaf5f1c103880f5abd7.jpg)

![](images/6d2b1863b99623a31ac85904695b8f2c19b7efb52218fa31d83dac68f3e8aa5b.jpg)  
Figure 4: SGR-OPSD with Qwen2.5-3B/7B-Instruct: validation success on ALFWorld/WebShop and macroaverage EM on SearchQA.

![](images/b5eb793621e962d0efd0277d885b325fa884a225a09b7ec1507ebe6513f09b1a.jpg)

![](images/f2ff8bda51ef67f26281aa1be3c5708dd67cefc0cb80e8ec6187319ac16c804a.jpg)  
(a) 60 matched pairs  
(b) 21 eligible pairs  
Figure 5: Agreement with independent return rankings on ALFWorld (3B): (a) teacher evidence versus Subgoal Progress Graph preferences; (b) credit before and after calibration. Error bars show 95% Wilson intervals.

## 4.6 Credit Assignment Analysis

Independent progress verification. Preferences from the Subgoal Progress Graph agree with independent return rankings on 49/60 action pairs (81.7%), versus 30/60 (50.0%) for teacher evidence (Figure 5a). Using a fixed Qwen2.5-3B-Instruct checkpoint, we test one pair per ALFWorld task with 32 continuations per action from matched states, contexts, and budgets. Mean discounted returns determine the reference ranking; the graph and pairs are frozen before sampling these independent continuations. Graph-preferred actions also achieve 2.8 pp higher continuation success.

Credit calibration analysis. Calibration translates this progress evidence into better credit allocation (Figure 5b). Applying Equation 11 offline to saved rollouts raises ranking agreement from 8/21 (38.1%) to 16/21 (76.2%) on the same 21 eligible pairs, selected independently of continuation outcomes. Sign-aware credit measures support through stronger reinforcement or reduced penalties, depending on the advantage sign. These findings confirm that calibration effectively aligns credit with independently observed returns. Appendix D provides eligibility criteria, confidence intervals, coverage, and state-merging diagnostics.

<table><tr><td>Trajectory Action</td><td></td><td> $A _ { i , t }$ </td><td>Base</td><td> $w ^ { \mathrm { r e f } }$ </td><td> $w ^ { \ast }$ </td></tr><tr><td>T1</td><td>go to garbagecan 1 +2.7351.0000 0.95960.9738</td><td></td><td></td><td></td><td></td></tr><tr><td>T2</td><td>go to cabinet 1</td><td>-0.3651.00001.03971.0540</td><td></td><td></td><td></td></tr><tr><td>T3</td><td>go to cabinet 1</td><td>-0.3651.00001.01921.0366</td><td></td><td></td><td></td></tr></table>

Table 5: Credit calibration at a matched state in the tomato-cooling task. Each row is a distinct trajectory; Base denotes the unit multiplier on GRPO advantages.

<table><tr><td rowspan="2">Method</td><td colspan="2">Qwen2.5-3B</td><td colspan="2">Qwen2.5-7B</td></tr><tr><td>Seen</td><td>Unseen</td><td>Seen</td><td>Unseen</td></tr><tr><td>GRPO</td><td>69.3</td><td>67.5</td><td>78.2</td><td>79.6</td></tr><tr><td>AgentOPSD</td><td>68.6</td><td>68.2</td><td>80.3</td><td>78.7</td></tr><tr><td>AlignOPSD</td><td>65.9</td><td>72.1</td><td>75.0</td><td>65.8</td></tr><tr><td>SGR-OPSD</td><td>76.9</td><td>73.1</td><td>85.6</td><td>82.3</td></tr></table>

Table 6: ALFWorld seen and unseen macro-average success (%; means across three training seeds).

Action-level case study. For the tomato-cooling task (Table 5), visiting garbagecan 1 reveals a tomato, whereas cabinet 1 is closed. Independent continuations from matched states favor the former in discounted return and success (32/32 vs 13/32). Relative to the teacher reference, offline calibration strengthens reinforcement for the positive-advantage garbagecan visit and penalties for the negative-advantage cabinet visits, shifting the credit gap from −0.0109 to +0.0191.

## 4.7 Generalization to Unseen Environments

SGR-OPSD successfully extends its performance gains to unseen ALFWorld environments, outperforming the strongest baseline by 1.0 pp at 3B scale and 2.7 pp at 7B scale (Table 6). It achieves 73.1% and 82.3% success on unseen environments at 3B and 7B, respectively, demonstrating robust transfer capabilities beyond seen evaluation settings.

These gains extend the benefit of structured credit assignment to unseen environments. Subgoal plans express task requirements, while the Subgoal Progress Graph calibrates action credit during training; evaluation uses the student’s ordinary interaction context. The unseen results support the transfer of the resulting policy improvements beyond the environments encountered during training.

## 5 Conclusion

We introduce SGR-OPSD, which combines subgoal-conditioned teacher evidence with calibration by the Subgoal Progress Graph to allocate credit across agent turns. It improves over GRPO at both 3B and 7B, reaching 85.6% ALFWorld macro-average success and 78.2% WebShop success at 7B, supporting structured credit assignment.

## Limitations

Evaluation scope. We evaluate SGR-OPSD on three benchmarks using Qwen2.5-3B/7B-Instruct. Its applicability to other model families and openended environments requires further evaluation, particularly when observations and action spaces differ from those studied here.

Fixed subgoal guidance. Subgoal plans are generated from task instructions before interaction and remain fixed across turns. They may omit requirements or fail to accommodate new information revealed by the environment. Updating plans from observed feedback is a natural extension of the current framework.

Rollout-dependent graph evidence. The Subgoal Progress Graph estimates empirical action values from recorded transitions and outcomes; comparison coverage depends on exploration and the availability of matched action visits. When eligible comparisons are absent, the method retains the teacher reference. Graph scores summarize progress within sampled rollouts rather than establish true action values; broader exploration and more reliable state matching are directions for extending this evidence.

## References

Mingju Chen, Can Lv, Jinrong Liu, Huan Zhang, Heng Chang, and Shiji Zhou. 2026. Beyond Timestamps: Decision-Aligned On-Policy Distillation for Long-Horizon Agents. arXiv:2609.33391.

DeepSeek-AI. 2024. Deepseek-v3 technical report. Preprint, arXiv:2412.19437.

Zenghuang Fu, Zhaoyang Li, Qiuyuan Ai, Xiaofeng Han, Zelong Zheng, Haoyu Wu, Tianyu Fu, Chenxu Zhao, Minghui Wu, Guannan He, and Changwei Wang. 2026. UniOPSD: Unifying Outcome and Hindsight Feedback for Agentic Reinforcement Learning. arXiv:2609.34810.

Bowen Jin, Hansi Zeng, Zhenrui Yue, Jinsung Yoon, Sercan Arik, Dong Wang, Hamed Zamani, and Jiawei Han. 2025. Search-R1: Training LLMs to Reason

and Leverage Search Engines with Reinforcement Learning. arXiv:2503.09516.

Qizhen Lan, Xi Xiao, Xiangchen Guan, Mengchen Fan, Moule Lin, Jung Im Choi, and Lijing Zhu. 2026. Trust is not enough: Influence calibration for onpolicy self-distillation in agentic rl. arXiv preprint arXiv:2608.14945.

Zhengxi Lu, Zhiyuan Yao, Zhuowen Han, Zi-Han Wang, Jinyang Wu, Qi Gu, Xunliang Cai, Weiming Lu, Jun Xiao, Yueting Zhuang, and Yongliang Shen. 2026. Self-Distilled Agentic Reinforcement Learning. arXiv:2605.15155.

Chunji Lv, Yangguang Wei, Junlin Liu, Yang Gao, Ming Liu, Xinming Wang, Jinyang Wu, Guoren Wang, and Changsheng Li. 2026. Pcsd: Persistent consistency for self-distillation in agentic reinforcement learning. arXiv preprint arXiv:2608.01837.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. 2024. DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. arXiv:2402.03300.

Mohit Shridhar, Xingdi Yuan, Marc-Alexandre Côté, Yonatan Bisk, Adam Trischler, and Matthew Hausknecht. 2020. ALFWorld: Aligning Text and Embodied Environments for Interactive Learning. arXiv:2010.03768.

Hao Wang, Guozhi Wang, Han Xiao, Yufeng Zhou, Yue Pan, Jichao Wang, Ke Xu, Yafei Wen, Xiaohu Ruan, Xiaoxin Chen, and Honggang Qi. 2026a. Skill-SD: Skill-Conditioned Self-Distillation for Multiturn LLM Agents. arXiv:2604.10674.

Zi-Han Wang, Zhengxi Lu, Zhiyuan Yao, Jinyang Wu, Jie Wu, Zhengzhou Cai, Yueqing Sun, Ziang Ye, Linji Hao, Qi Gu, Xunliang Cai, Yongliang Shen, and Yujiu Yang. 2026b. AgentOPSD: Recursive Self-Distillation for Agentic Reinforcement Learning. arXiv:2608.05987.

Chenxu Yang, Chuanyu Qin, Qingyi Si, Minghui Chen, Naibin Gu, Dingyu Yao, Zheng Lin, Weiping Wang, Jiaqi Wang, and Nan Duan. 2026a. Self-Distilled RLVR. arXiv:2604.03128.

Jingxiao Yang, Wangjie Gan, Yingxuan Zhuang, Wenqi Zhang, Jintao Chen, and Xuhong Zhang. 2026b. Reconciling Process Supervision with Outcome-Based Credit in Agentic Policy Optimization. arXiv:2608.31077.

Shunyu Yao, Howard Chen, John Yang, and Karthik Narasimhan. 2022. WebShop: Towards Scalable Real-World Web Interaction with Grounded Language Agents. arXiv:2207.01206.

Siyan Zhao, Zhihui Xie, Mengchen Liu, Jing Huang, Guan Pang, Feiyu Chen, and Aditya Grover.

2026a. Self-distilled reasoner: On-policy selfdistillation for large language models. arXiv preprint arXiv:2601.18734.

Siyan Zhao, Zhihui Xie, Mengchen Liu, Jing Huang, Guan Pang, Feiyu Chen, and Aditya Grover. 2026b. Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models. Preprint, arXiv:2601.18734.

## A Privileged Information and Subgoal Generation

## A.1 Comparison of Privileged Information

Table 7 compares task specificity, subgoal planning, and interaction requirements of privileged guidance in related methods. Subgoal plans condition guidance on the current instruction and explicitly connect intermediate goals through ordering preconditions. They are generated before student interaction and reused across turns, making the task’s object, destination, count, and required state explicit without requiring a successful rollout or a reference answer.

Concrete Skill–Subgoal comparison. Table 8 pairs a recorded Subgoal plan with the Skill text selected for the same ALFWorld instruction. In this Skill configuration, task-type matching concatenates general rules with the corresponding taskspecific file. Both forms encode ordering: Skill expresses reusable rules with generic objects and destinations, while Subgoal instantiates these requirements for the current task. The table reproduces two Skill excerpts and the complete stored plan.

Guidance length. We compare the six task-type Skill inputs with an ALFWorld corpus of 601 unique cached Subgoal plans, using the Qwen2.5- 3B-Instruct tokenizer without special tokens. Skill inputs ranged from 861 to 1168 tokens; Subgoal plans averaged 87.79 tokens (median 85; range 28– 162). These counts cover the privileged text supplied to the teacher, excluding interaction history. They quantify the compactness of the task-specific representation in this corpus.

## A.2 DeepSeek-V3.1 Generation Prompts

We query DeepSeek-V3.1 (DeepSeek-AI, 2024) through a chat-completion interface with temperature 0 and a maximum output of 1024 tokens. Each initial request contains the environment-specific system prompt reproduced below and a user message containing only the task instruction x (the question text for Search-QA). Outputs that fail validation are retried with the previous response and validation feedback. The deployment model identifier is deepseek-v31-meituan. Plans are cached by task and generation configuration and reused unchanged; ALFWorld and WebShop generate them before the first rollout for a new instruction, while Search-QA precomputes the scheduled training questions. The external model generates guidance, and the rollout policy supplies both likelihood scores.

The response is a JSON object containing 2–8 subgoals, each with nonempty goal and precondition fields. The teacher receives the numbered rendering j. <goal> Preconditions: <precondition>. The complete plan is supplied at each turn, without stage-dependent filtering, and is absent from the student’s acting context and evaluation prompts.

## ALFWorld system prompt.

Plan an ALFWorld household task from its task description only. Return JSON only: {"subgoals": [{"goal": "...", "precondition": "..."}]}. Give 2 to 8 concise ordered subgoals, including any necessary exploration, acquisition, cleaning/heating/cooling, and final placement/use. State ordering preconditions. Describe desired states rather than a guessed action transcript. You do not know the initial scene or object locations. Never invent locations, object instance numbers, observations, or completed steps. Use conditional search where needed. Preserve the task’s exact object type, destination type, count, and required state. Never substitute another receptacle or add an alternative destination (a table is not interchangeable with a counter). A two-object task requires two distinct objects. Keep the whole plan below 350 English words. Do not include an explanation outside JSON.

## WebShop system prompt.

Plan a WebShop shopping task using only the customer’s instruction. Return JSON only: {"subgoals": [{"goal": "...", "precondition": "..."}]}. Give 2 to 8 concise ordered subgoals and their ordering preconditions. Include searching for candidates, inspecting product details, checking ALL requested attributes, selecting the exact required options, verifying the price constraint, and buying only after the

<table><tr><td>Method</td><td>Task-instance specific</td><td>Task-specific subgoal plan</td><td>No successful rollout required</td><td>No task-specific interaction required</td></tr><tr><td>OPSD (Skill) (Zhao et al., 2026b)</td><td>X</td><td>X</td><td></td><td>√</td></tr><tr><td>RLSD (Yang et al., 2026a)</td><td></td><td>X</td><td></td><td>√</td></tr><tr><td>Skill-SD (Wang et al., 2026a)</td><td>√</td><td>√</td><td></td><td>X</td></tr><tr><td>SDAR (Lu et al., 2026)</td><td>X</td><td>X</td><td></td><td>√</td></tr><tr><td>AgentOPSD (Wang et al., 2026b)</td><td>X</td><td>X</td><td></td><td>√</td></tr><tr><td>UniOPSD (Fu et al., 2026)</td><td>7</td><td>X</td><td>X</td><td>X</td></tr><tr><td>SGR-OPSD</td><td>L</td><td>L</td><td></td><td>V</td></tr></table>

Table 7: Comparison of privileged guidance (✓: present; ×: absent): task specificity refers to the current instance, and subgoal plans specify intermediate goals with explicit ordering or dependencies. Interaction requirements concern constructing privileged guidance from current-task executions, not ordinary student rollouts.
<table><tr><td colspan="2">Task: clean some soapbar and put it in countertop.</td></tr><tr><td>Skill guidance: verbatim excerpts</td><td>Subgoal guidance: complete recorded plan</td></tr><tr><td>10. Progressive Goal Decomposition: Break tasks into or- dered sub-goals: (1) locate object, (2) obtain or transform it, (3) locate destination or reference object, (4) satisfy spatial relation or placement. Complete each sub-goal before mov- ing to the next. Apply this at the start of every episode. 1. Phase-Ordered Plan: Always execute clean tasks in the fixed sequence: (1) locate and acquire target object, (2) bring it to an available water source or sink to clean, (3) navigate to final location, (4) place object. Apply this as soon as the goal specifies the object must be clean before placement.</td><td>1. Find a soapbar in the environment Preconditions: None 2. Pick up the soapbar Preconditions: Soapbar is found and reachable 3. Clean the soapbar (e.g., rinse or wash it) so it is clean Preconditions: Soapbar is held 4. Find a countertop receptacle Preconditions: Soapbar is clean and held 5. Put the clean soapbar on the countertop Preconditions: Coun- tertop is found and reachable, soapbar is clean and held</td></tr></table>

Table 8: Actual ALFWorld guidance for the same task. Skill uses type-level rules; the recorded Subgoal plan names the soapbar and countertop and ties placement to the object’s required state.

visible evidence supports the requirements. Describe desired states, not a guessed action transcript. Preserve every stated product type, attribute, option (such as size/color), quantity, and price bound. Do not silently relax or substitute a requirement, even if it appears unusual. You have no catalog, search results, product IDs, prices, observations, past actions, or reward information. Never invent these, assert that a suitable product exists, or claim a step is complete. Use conditional search/refinement where needed. Do not add unstated purchase requirements. Keep the plan under 350 English words.

## Search-QA system prompt.

Plan how to answer a retrieval question using only the question. Return JSON only: {"subgoals": [{"goal": "...", "precondition": "..."}]}. Give 2 to 8 concise ordered evidence goals and their dependencies, under 350 English words. Identify missing facts, bridge entities and constraints that must be established by retrieved evidence. For a single-hop question use a simple evidence lookup and answer verification. For a multi-hop question order lookups according to their

dependencies. End with checking the evidence against the question and producing a concise answer. You have no search results, gold answer, supporting-fact labels, student history or reward. Do not supply or guess the answer, invent bridge entities, assert an unknown fact, or claim that a goal is complete. Use placeholders or conditional goals for unknown entities. Preserve the exact entities, relation and time constraints in the question. Do not prescribe more searches than the four-turn interaction budget.

## B Implementation Details of Credit Allocation

This appendix specifies the scoring, weighting, and graph conventions used by the allocation in Section 3.

## B.1 Scoring Context and Action Tokens

The teacher prompt prepends the cached task plan to the student prompt and is left-truncated to the environment-specific context limit. Consequently, $h _ { i , t } ^ { T }$ and $h _ { i , t } ^ { S }$ can retain different parts of the interaction context. Both branches use the same pre-update actor and score the original sampled response IDs. Action-token probabilities remain conditioned on earlier reasoning tokens; Equation 1 is not a probability marginalized over reasoning.

The action mask selects original response tokens that overlap the trimmed executable-action span, including tokens crossing its byte boundaries. The span is not retokenized. A turn without a scoreable action receives $d _ { i , t } = 0$ and is excluded from graph comparisons. Its outcome penalty and fullresponse policy update remain active. Such a turn still participates in its trajectory’s allocation budget, so its multiplier need not remain one.

## B.2 Training Mass and Unique Visits

Training batches can contain padding copies of an environment turn. The allocator represents each unique turn once, after checking agreement of its copies’ evidence and base advantages. Its mass $m _ { i , t }$ is the number of effective response tokens summed across all training copies. These copies contribute to the policy loss and hence to $p _ { i , t }$ , but do not create additional graph visits. The allocated advantage is broadcast back to all copies.

Base advantages retain the reward normalization and invalid-action penalties of the training recipe in Section 4.1. The allocator does not replace them by a common trajectory advantage. Accordingly, positive, negative, and zero advantages can occur within one trajectory. The multiplier budget $\begin{array} { r } { \sum _ { t = 1 } ^ { N _ { i } } p _ { i , t } w _ { i , t } = 1 } \end{array}$ does not imply conservation of the sum of advantages or the optimizer’s gradient norm.

## B.3 Graph Construction and Pair Eligibility

State and action identities. Each task’s group of eight student rollouts forms a separate Subgoal Progress Graph at each update. Recorded visits contribute transition counts once, excluding training padding copies. ALFWorld node keys combine observation-memory anchors with physicalstate signatures hashing simulator facts and entity metadata. WebShop keys combine formatted observations with product identity, page, page name, and selected options. These descriptors support graph estimation without entering teacher or student prompts. Within a node, actions are grouped by their exact executed strings.

SearchQA states and actions. The graph ID hashes the update, rollout-group ID, and question ID. A node represents the exact formatted student context: the question, interaction count, and ordered executed queries with their returned evidence.

The four-turn history window covers preceding interactions within the four-turn budget. Nodes merge only when their context strings match within the graph. Different query histories remain separate even if they retrieve the same passage. The pre-action state signature hashes the question ID and context; the student-prompt hash checks the scoring input after prompt processing.

Actions use the projected executable string: <search>query</search> or <answer>answer</answer>. Projection removes surrounding reasoning, standardizes the wrapper, and trims payload-boundary whitespace without semantically normalizing queries or answers. Gold answers and reward labels are excluded from node keys and state signatures.

Empirical graph values. Let $N ( v , a )$ count recorded visits taking action a at node $v ,$ and $\begin{array} { r } { N ( v ) = \sum _ { a } N ( v , a ) } \end{array}$ . Equation 7 uses ${ \widehat { \pi } } ( a \mid v ) =$ $N ( v , a ) / N ( v ) \colon \widehat { Q }$ averages immediate rewards and discounted successor values over recorded transitions, and $\widehat { V }$ averages action values by visit frequency. Their difference measures an action’s empirical return relative to the state average.

For ALFWorld and WebShop, the recorded binary success reward (0 or 10) is divided by 10 before value estimation; SearchQA uses binary terminal exact-match reward directly. Thus all three environments use $r \in \{ 0 , 1 \}$ and $\gamma = 0 . 1$ , with success assigned to the terminating transition. Terminal transitions, exhausted budgets, and successors absent from the recorded source-node set have zero continuation value; immediate rewards are retained. Failed or budget-truncated episodes remain in the transition counts, with zero terminal success reward.

Values are initialized to zero and iterated until the maximum absolute change is below 10<sup>−12</sup>, with at most 200 iterations; failure to converge raises an error. Cycles are handled by these discounted fixed-point updates. Clipping is applied after subtracting $\widehat { V }$ from ${ \widehat { Q } } ,$ as in Equation $\mathbf { \boldsymbol { 8 } ; }$ there is no subsequent reward rescaling. All visits to a state– action pair share its empirical signal. The values summarize continuation behavior pooled by node; the additional context and budget checks below restrict comparisons, not the value-estimation pool.

Pair eligibility and averaging. Within each node–plan bucket, every visit associated with the two candidate actions must be scoreable, have nonzero base advantage, and agree on the preaction state signature, student-prompt hash, and remaining step budget. ALFWorld signatures use simulator facts and entity metadata; WebShop signatures use recorded session state and available actions. Success reachability is not an additional eligibility condition. We retain $( a , b )$ when $\bar { g } ( a ) > \bar { g } ( b )$ , excluding ties. The vector $H _ { a }$ averages unique visits equally within each trajectory, then equally across trajectories. Although identical state–action visits share a graph score, this averaging also defines their mean credit correction. Graph scores supply pair ordering, while $\mu$ sets the correction margin. For each eligible pair $e ,$ the comparison weight is

$$
\rho _ { e } = \frac { 1 } { \left. \mathcal { E } _ { v } \right. \left. \left. b _ { e } \right. \right. _ { 2 } ^ { 2 } } ,\tag{16}
$$

where $\mathcal { E } _ { v }$ contains the eligible pairs in the same node–plan bucket. This weight normalizes comparison count and coefficient scale; it is not a bootstrap reliability estimate.

## B.4 Optimization and Fallback

For a nonzero advantage, differentiating the teacher objective before clipping gives $s _ { i , t } d _ { i , t } -$ T log $w _ { i , t } - \nu _ { i } = 0$ . The resulting exponential tilt is clipped to the prescribed interval, and the trajectory multiplier $\nu _ { i }$ is found by bisection to satisfy the budget. Zero-advantage turns retain $w _ { i , t } = 1$ . This solves the bounded allocation directly, rather than clipping weights and subsequently renormalizing them outside their bounds.

Joint solver. We use float64 Nesterovaccelerated projected gradient descent, initialized at the teacher correction $c ^ { \mathrm { r e f } }$ , with monotone restart. Let $F ( c )$ denote the joint objective, $D = \deg ( p )$ the token-mass metric, and $\Pi _ { D }$ its projection onto the bounded, per-trajectory budget constraints. The step size is $1 / L$ , with

$$
L = \frac { T _ { G } } { 1 - \epsilon } + \beta \bigl \| \boldsymbol { M } ^ { \intercal } \mathrm { d i a g } ( \rho ) \boldsymbol { M } \bigr \| _ { \infty } ,\tag{17}
$$

where B has rows $b _ { e } ^ { \top } , M = | B | D ^ { - 1 / 2 }$ , and the matrix infinity norm is the maximum row sum. The momentum coefficient is $( \sqrt { L } - \sqrt { m } ) / ( \sqrt { L } + \sqrt { m } )$ with $m = T _ { G } / ( 1 + \epsilon )$ . An extrapolated step that increases the objective beyond $1 0 ^ { - 1 3 } ( 1 + | F ( c ) | )$ is replaced by the ordinary projected-gradient step and momentum is reset. Outside the feasible box, extrapolation uses the KL term’s quadratic extension matching its value, gradient, and curvature at the clipped boundary.

We stop when

$$
L \big | \big | c - \Pi _ { D } \big ( c - D ^ { - 1 } \nabla F ( c ) / L \big ) \big | \big | _ { \infty } \leq 1 0 ^ { - 8 } ,\tag{18}
$$

with at most 5,000 iterations. Teacher-budget bisection uses 100 iterations; each metric projection uses 55. A solution must be finite, have maximum trajectory-budget error at most $1 0 ^ { - 1 0 }$ and bound violation at most $1 0 ^ { - 1 2 }$ , and increase the objective over the teacher initialization by at most 10<sup>−9</sup>. Failure of convergence or these checks raises an error and stops the update; it does not silently substitute teacher weights.

No-comparison case. A trajectory with no eligible graph term retains its teacher reference exactly. Within a trajectory containing graph comparisons, the budget can also adjust turns that do not participate in a pair. The soft hinge term encourages pairwise margins but does not guarantee that every graph comparison is satisfied. The experimental allocation hyperparameters are reported in Section 4.1.

## C Experimental Details

## C.1 Training and Evaluation Protocols

The following settings describe the recorded SGR-OPSD training and full-evaluation configurations. The ALFWorld and WebShop training configurations use 16 tasks per batch, eight trajectories per task, and 150 policy updates. The learning rate is $1 0 ^ { - 6 }$ , the reference-policy KL coefficient is 0.01, and training and validation temperatures are 1.0 and 0.4, respectively. ALFWorld and WebShop retain a two-turn history window. SearchQA uses a four-turn history window, 128 tasks per batch, a reference-policy KL coefficient of 0.001, and a four-turn interaction budget. Its training data comprise NQ and HotpotQA; evaluation covers all seven datasets listed in Section 4.1.

Base advantages. For ALFWorld and WebShop, row-level rewards are centered and scaled by the mean and sample standard deviation over training rows sharing a task UID, including padding copies, with stabilizer 10<sup>−6</sup>. The invalid-action penalty coefficient is 0.1, so advantages can differ across turns of one trajectory. The policy loss averages over effective response tokens. Allocation deduplicates turns while retaining their training token mass, as described in Appendix B.

Evaluation sets and decoding. Full ALFWorld evaluation uses 140 seen and 134 unseen tasks, with one rollout per task, temperature 0.4, topp = 1, and top-k = −1. Table 1 uses the seen split and reports the unweighted mean of its six category success rates. WebShop evaluation uses the 500 goals indexed 0–499 with temperature 0.4. Success is the fraction of fully completed goals; Score is the mean native partial-completion reward multiplied by 100. SearchQA uses greedy decoding and question-level exact match across 51,713 questions in seven datasets. ALFWorld seen and Web-Shop full evaluation sets overlap their respective checkpoint-selection validation sets; ALFWorld unseen evaluation is separate from selection.

<table><tr><td>Setting</td><td>ALFWorld</td><td>WebShop</td></tr><tr><td>Maximum environment steps</td><td>50</td><td>15</td></tr><tr><td>Prompt / response tokens</td><td>2048 / 512</td><td>4096 / 512</td></tr><tr><td>PPO minibatch</td><td>256</td><td>64</td></tr><tr><td>Actor microbatch per GPU</td><td>32</td><td>8</td></tr><tr><td>Rollout tensor parallelism</td><td>2</td><td>2</td></tr><tr><td>Validation tasks per check</td><td>128</td><td>128</td></tr><tr><td>Validation interval (updates)</td><td>5</td><td>5</td></tr><tr><td>Periodic save interval</td><td>10</td><td>10</td></tr></table>

Table 9: Training and evaluation settings for the full method. ALFWorld settings apply to both model sizes. WebShop uses two GPUs for 3B and eight for 7B; the other listed settings are shared.

Checkpoint selection. ALFWorld and WebShop select saved checkpoints using training-time validation success. For the original SearchQA mainresult runs and w/o Graph controls, scheduled post-update validation occurs at update 150 on all 51,713 questions; these runs use the update-150 checkpoint. The separate SearchQA curve runs validate every five updates on a fixed 3,125-question subset. Their selection criterion is the highest subset micro EM, with the earliest checkpoint retained on a tie; the displayed curve metric is macro EM. Full-set evaluation follows selection. The newly completed 7B w/o Teacher Signal and Skill controls use selected updates 150 and 140, respectively, for full-set evaluation.

Evaluation randomness. The frozen ALFWorld full-evaluation protocol uses environment seed 0, task seed 1000 + index, and engine seed 0. Training and evaluation seeds are recorded separately; a shared engine seed does not imply paired perrequest sampling.

Aggregation. Table 1 reports three-seed aggregates for trained methods, using seeds {0, 42, 2026}. ALFWorld and SearchQA Avg first average equally over the six task categories and seven datasets, respectively, within each run, then across runs. WebShop Score and success use the same seed aggregation. In Tables 2 and 3, fullmethod rows reproduce the main-table means. Figures 3 and 4 show individual seed-0 runs.

## C.2 Training-Curve Sources

SearchQA validation runs. Figure 4(c) uses dedicated 3B and 7B runs with every-five-update validation, separate from the original update-150 main-result runs. Each point at updates 5, 10, . . ., 150 comes from greedy student-only evaluation of the same fixed 3,125 questions: 500 each from NQ, TriviaQA, PopQA, HotpotQA, 2WikiMulti-HopQA, and MuSiQue, and all 125 Bamboogle questions. The subset is drawn from the 51,713- question evaluation pool and reused unchanged across checkpoints and model sizes. Macro EM is $\textstyle 1 0 0 \times { \frac { 1 } { 7 } } \sum _ { d = 1 } ^ { 7 } C _ { d } / N _ { d } ,$ where $C _ { d }$ and $N _ { d }$ are each dataset’s correct-answer and question counts. The 60 plotted points are computed from these perdataset validation counts. The original and curverun packages use byte-identical graph construction, pair eligibility, and allocation solver files.

The full evaluation pool contains 3,610 NQ, 11,313 TriviaQA, 14,267 PopQA, 7,405 HotpotQA, 12,576 2WikiMultiHopQA, 2,417 MuSiQue, and 125 Bamboogle questions. Main-table and ablation SearchQA scores use this full pool. Both evaluations report macro EM across seven datasets: curves use the fixed subset at each checkpoint, while tables use full-pool checkpoint evaluations.

ALFWorld and WebShop curves. The other panels use recorded training-validation points from the seed-0 method-comparison and full-method runs, including their documented checkpointresume chains. Validation uses 128 episodes per check, and the plotted success is the fraction of successful episodes. ALFWorld curves therefore differ from the six-category macro-average in the result tables. In the 3B WebShop method comparison, GRPO and AgentOPSD use fixed validation tasks while SGR-OPSD samples tasks; final Web-Shop table entries use all 500 goals. All curves use recorded checkpoints and retain their validation protocols.

## D Progress Verification and Credit Calibration

## D.1 Independent Progress Verification

Agreement with independent outcomes. Preferences from the Subgoal Progress Graph agree with independent continuation returns on 49/60 matched ALFWorld action pairs (81.7%), compared with 30/60 (50.0%) for teacher evidence. Graph-preferred actions also achieve higher mean discounted return and continuation success (Table 10). They succeed in 1,756/1,920 continuations, compared with 1,702/1,920 for their alternatives. At the task level, success is higher for 22 pairs, lower for nine, and tied for 29. On the 31 pairs with unequal success rates, graph preferences agree with the success ordering on 22 pairs (71.0%), compared with 16 (51.6%) for teacher evidence. Return agreement evaluates all 60 pairs and captures both completion and its timing.

Independent continuation protocol. The evaluation fixes Qwen2.5-3B-Instruct at update 150, selected by training-validation success (82.03%), and constructs graphs from eight rollouts for each of 140 ALFWorld tasks. Candidate actions must be legal and share the exact simulator-state signature, student-prompt hash, and remaining budget. The sampling protocol requires an absolute graphscore difference above $1 0 ^ { - 8 }$ and positive recorded reliability for at least one source visit. The latter requires that its state–action pair occur in at least two trajectories, have nonzero empirical advantage, and appear in at least 16 of 64 trajectory-bootstrap replicates. Each replicate resamples the eight trajectories with replacement. These criteria select the evaluation pool; allocation uses the objective and pair weights defined in Appendix B.

The pool contains 311 action pairs across 94 tasks. Using selection seed 20261027, we choose one pair uniformly within each task, reserve eight disjoint tasks for replay checks, and select 60 formal tasks proportionally across the remaining task categories. The state and action manifest is frozen before continuation sampling. For each pair, we restore the recorded state and history, force either action, and sample 32 continuations per action with the remaining interaction budget. Sampling uses temperature 1, top-p 1, no top-k restriction, and at most 512 response tokens per turn. Request seeds differ between graph construction and continuation evaluation; the inference-engine initialization seed is 0. The resulting 3,840 continuations are excluded from graph estimation and pair selection.

<table><tr><td>Measure</td><td>Estimate</td><td>95% interval</td></tr><tr><td>Teacher return agreement</td><td>50.0%</td><td>[37.7, 62.3]</td></tr><tr><td>Graph return agreement</td><td>81.7%</td><td>[70.1, 89.4]</td></tr><tr><td>Mean return difference</td><td>+0.03461</td><td>[0.00054, 0.08394]</td></tr><tr><td>Success difference (pp)</td><td>+2.81</td><td>[0.05, 5.89]</td></tr></table>

Table 10: Independent outcomes for 60 matched action pairs. Differences subtract the alternative action from the graph-preferred action.
<table><tr><td>Covered unit</td><td colspan="2">Count / total</td><td>Fraction</td></tr><tr><td>Tasks</td><td colspan="2">94 / 140</td><td>67.1%</td></tr><tr><td>Trajectories</td><td colspan="2">658 / 1,120</td><td>58.8%</td></tr><tr><td>Turns</td><td colspan="2">1,055 / 26,717</td><td>3.95%</td></tr><tr><td>Additional node key</td><td>Same</td><td>Reversed</td><td>Below threshold</td></tr><tr><td colspan="2">Formal pairs  $( n = 6 0 )$ </td><td></td><td></td></tr><tr><td>Remaining budget</td><td>52</td><td>2</td><td>6</td></tr><tr><td>Budget and prompt</td><td>52</td><td>2</td><td>6</td></tr><tr><td colspan="2">All candidates (n = 311)</td><td></td><td></td></tr><tr><td>Remaining budget</td><td>278</td><td>8</td><td>25</td></tr><tr><td>Budget and prompt</td><td>278</td><td>8</td><td>25</td></tr></table>

Table 11: Candidate coverage and sensitivity to node splitting. Directions are relative to the original graph; the gap threshold is $1 0 ^ { - 8 }$

Outcome metrics. For each action, we average the independent discounted returns $\textstyle \sum _ { j \geq 0 } \gamma ^ { j } r _ { j }$ with $\gamma = 0 . 1$ , binary terminal success reward, and $j = 0$ denoting the forced action. Ranking agreement compares a predicted gap’s sign with the difference between these two sample means. All 60 empirical return differences exceed the outcome sign tolerance $1 0 ^ { - 1 2 }$ . Graph gaps use tolerance $1 0 ^ { - 8 } ;$ teacher and credit gaps use 10<sup>−6</sup>. Success is also measured without discounting. Table 10 reports both endpoints. Proportion intervals are 95% Wilson intervals; paired differences use 10,000 task-level bootstrap resamples with seed 20261027, preserving the two actions within each task.

Coverage and state merging. Candidate comparisons cover 67.1% of tasks and 58.8% of trajectories in the graph-building pool, with direct participation concentrated in 3.95% of turns (Table 11). Coverage counts each task, trajectory, or turn once before nonzero-advantage allocation filtering. This measures where action comparisons are available; the trajectory budget determines how their calibration also affects other turns.

Splitting nodes by remaining budget preserves 52 of the 60 evaluated preference directions, reverses two, and leaves six below threshold. Adding prompt identity produces no further direction changes. Both refinements agree with independent returns on 44/54 remaining predictions (81.5%), with six abstentions. The diagnostic holds candidate actions and independent outcomes fixed while recomputing graph values under each node key. The original nodes include physical-state signatures; among 17,991 visited source nodes, 4,375 have repeated visits, 3,513 mix remaining budgets, and 3,540 mix student-prompt hashes. These mixed nodes account for 9,440 and 9,502 of the 26,717 turns, respectively. The results quantify how successor pooling affects preferences even when the compared actions share a matched starting state.

For example, the original graph favors go to countertop 1 over go to cabinet 1 by 0.000333; budget splitting reverses the gap to −0.000045. Both actions succeed in all 32 continuations, while the countertop action has 0.00028884 higher mean discounted return. This pair illustrates why success and discounted return are reported separately, and why finer merging keys are assessed against independent outcomes.

## D.2 Credit Calibration Analysis

Improved outcome alignment. Calibration using Subgoal Progress Graph preferences increases agreement with independent returns from 8/21 (38.1%) to 16/21 (76.2%) on the same directly eligible pairs, with Wilson intervals [20.8%, 59.1%] and [54.9%, 89.4%] (Table 12). Across all 60 pairs, calibration yields eight additional correct rankings: nine corrections and one change from correct to incorrect. The paired gain is 13.3 percentage points over all 60 pairs (95% task-bootstrap interval [3.3, 23.3]). Thirty-five pairs involve zero-advantage visits and remain tied under both reference and calibrated credit; four further pairs fall outside the exact direct-comparison conditions. The table reports correctness and ties for both the complete set and the directly eligible subset.

Retrospective scoring and allocation. At a fixed checkpoint, we recompute credit weights from the saved graph-building rollouts of the 60 formal tasks using Equations 2 and 11. This offline comparison isolates the allocation mechanism while holding likelihood scores and continuations fixed. Teacher and student score the same saved response under their respective contexts. Scoring uses saved-text retokenization, checked by text round trips and student-prompt hashes, because original generation token IDs are unavailable. Of the 60 task plans, 53 reuse cached plans and seven follow the original instruction-only generation protocol. Continuation outcomes are held out from plan generation, scoring, and calibration.

<table><tr><td>Signal</td><td>Correct</td><td>Incorrect</td><td>Tied</td></tr><tr><td>All 60 pairs</td><td></td><td></td><td></td></tr><tr><td>Teacher evidence</td><td>30</td><td>30</td><td>0</td></tr><tr><td>Graph preference</td><td>49</td><td>11</td><td>0</td></tr><tr><td>Teacher-reference credit</td><td>11</td><td>14</td><td>35</td></tr><tr><td>Calibrated credit</td><td>19</td><td>6</td><td>35</td></tr><tr><td>Same 21 directly eligible pairs</td><td></td><td></td><td></td></tr><tr><td>Teacher evidence</td><td>9</td><td>12</td><td>0</td></tr><tr><td>Graph preference</td><td>16</td><td>5</td><td>0</td></tr><tr><td>Teacher-reference credit</td><td>8</td><td>13</td><td>0</td></tr><tr><td>Calibrated credit</td><td>16</td><td>5</td><td>0</td></tr></table>

Table 12: Credit-ranking agreement with independent mean continuation returns. Correct, incorrect, and tied predictions form disjoint categories.

The reconstruction includes all eight graphbuilding trajectories per task: 8,202 unique turns, of which 8,201 have scoreable actions. Both actions in every selected pair are scoreable. Base advantages use the frozen GRPO reward normalization and invalid-action penalty; token masses count the canonical response tokens once per unique visit. We use $\epsilon = 0 . 1 , T = T _ { G } = 1 , \beta = 2 , \mu = 0 . 0 2$ and graph-score bound $b _ { G } = 0 . 3$ , with the solver checks in Appendix B.

For pair e, raw teacher evidence is $b _ { e } ^ { \top } d ,$ while teacher-reference and calibrated credit gaps are $b _ { e } ^ { \top } c ^ { \mathrm { r e f } }$ and $b _ { e } ^ { \top } c ^ { * }$ , respectively, with $c ^ { \mathrm { r e f } } ~ = ~ s \odot$ $( w ^ { \mathrm { r e f } } - \mathbf { 1 } )$ . These sign-aware gaps compare favorable credit across advantage signs. Eligibility selects 21 pairs using scoreability, nonzero advantages, and matching state, context, and budget across their source visits, independently of continuation outcomes.