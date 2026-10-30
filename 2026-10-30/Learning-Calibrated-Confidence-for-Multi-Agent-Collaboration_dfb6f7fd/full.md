# Learning Calibrated Confidence for Multi-Agent Collaboration

Chenxiao Fan<sup>1,2</sup>, Chongming Gao<sup>1</sup>, Gangyi Zhang<sup>2</sup>, Yaxin Gong<sup>1</sup>, Leyang Shen<sup>3</sup>, Jiakai Wang<sup>2</sup>, Dong Wang<sup>2</sup>, Yang Liu<sup>2</sup>, Haoyan Liu<sup>1</sup>, Xinting Hu<sup>1</sup>, Fuli Feng<sup>1</sup>, Xiangnan He<sup>1</sup>

<sup>1</sup>University of Science and Technology of China <sup>2</sup>Qwen Business Unit of Alibaba <sup>3</sup>National University of Singapore simonfan@mail.ustc.edu.cn

## Abstract

Complex information seeking on the Web often requires collecting evidence from multiple documents and reasoning across them, as in multi-hop question answering. Multi-agent systems address such tasks by dividing work among specialized agents, where upstream agents produce intermediate answers and supporting evidence that downstream agents build upon. However, upstream answers vary in correctness, and most systems provide no quantitative signal about it, leaving the downstream agent to judge whether to trust or verify what it receives based solely on the textual evidence.

In this paper, we propose MASCAL. Each agent learns to produce a correctness probability for its answer by jointly optimizing task performance and confidence calibration in reinforcement learning; the upstream agent passes this probability at handof, giving the downstream agent a quantitative basis for its decisions, and the final answer also carries a calibrated probability, quantifying the reliability of the system output. To guide the downstream agent in using this signal during training, we introduce confidence-aware weighting, which increases the learning weight at positions responsive to the upstream probability, so that the agent learns more from decisions where it actively uses the signal. Across three collaboration architectures spanning Web retrieval and reasoning tasks, MASCAL improves task accuracy and calibration quality for both agents, and generalizes to deeper collaboration chains and diferent model scales. Intervention analysis confirms that upstream confi dence afects downstream decision behavior, and the learned final confidence supports selective answering and targeted rechecking.

## Keywords

Multi-Agent Systems, Agentic Information Retrieval, Confidence Calibration, Large Language Models, Reinforcement Learning

## 1 Introduction

Complex information seeking on the Web often requires collecting evidence from multiple sources and reasoning across them, as in multi-hop question answering [39, 48] or cross-document fact verification [21]. Large language model (LLM) agents have made substantial progress on such tasks by combining Web retrieval with step-by-step reasoning [23, 35]. However, a single agent remains limited in both coverage and reliability when the task spans many evidence sources or requires extended reasoning chains [41, 45]. Multi-agent systems (MAS) address this through role specialization: diferent agents retrieve separate pieces of evidence, answer diferent sub-questions, or produce candidate answers for another role to review [6, 29, 32, 44]. In a typical pipeline, an upstream agent produces an intermediate answer with supporting evidence and hands of a report to a downstream agent for further processing [43]. The quality of the final output therefore depends both on each role’s subtask performance and on how the downstream agent handles the upstream conclusions it receives [15].

![](images/c387c3eb04086bc9901fa20860e4540a74c133e68134b52165ad746432fe3f3a.jpg)  
Figure 1: Confidence at the agent handof. Without a confidence signal, the downstream agent cannot distinguish a reliable report from an unreliable one. MASCAL attaches calibrated confidence to each report and to the final answer, guiding downstream decisions and enabling reliability assessment of the system output.

Upstream agents vary in quality across inputs, and the downstream agent must assess each report it receives. It must decide whether to trust the upstream conclusion and move forward, or to launch additional searches to verify [54]. Yet as Figure 1 illustrates, most current MAS convey only the answer and textual evidence, giving the downstream agent no quantitative signal about answer correctness to inform this choice. If the upstream agent could convey, alongside its answer, a probability reflecting how likely that answer is to be correct, the downstream agent would gain a concrete basis for its decision. This raises the question: can each agent learn to express calibrated answer confidence while learning its task, and can the downstream agent learn to act on this signal?

Training with calibration objectives has shown that a single LLM can learn to produce well-calibrated correctness probabilities for its answers, providing a basis for introducing reliable confidence signals into multi-agent settings [8, 11, 31]. In multi-agent settings, existing work has explored confidence expression and utilization in debates, discussions, and pairwise interactions [36, 40, 58], while other studies have investigated cross-role uncertainty propagation and system-level probability estimation [20, 30]. These eforts each address part of the problem, but no existing framework connects confidence learning, transmission, and utilization across the full collaboration pipeline: the upstream agent needs to learn calibrated confidence for its own answer, and the downstream agent needs to learn to incorporate this signal into its decisions.

To bridge this gap, we propose MASCAL, in which each agent outputs a correctness probability for its answer alongside complet ing its subtask; the upstream agent passes this probability to the downstream agent together with its intermediate conclusions. This probability is learned by incorporating a calibration objective into reinforcement learning: task performance and confidence calibration are jointly optimized within a single training loop, so that the model produces probabilities matching its actual correctness rates. To further guide the downstream agent in leveraging this signal, we introduce confidence-aware weighting, which measures how much each generation position responds to the upstream probability and increases the learning weight of the most responsive positions, so that the agent learns more from decisions where it actively uses the upstream signal. The downstream agent also outputs a calibrated probability for its final answer, allowing the system to withhold uncertain outputs or direct rechecking where it is most needed.

We evaluate MASCAL under three collaboration architectures (sequential handof, report fusion, and answer review) spanning Web-based multi-hop question answering, logical reasoning, and mathematical reasoning. MASCAL improves task accuracy over single-agent baselines, MAS without confidence transmission, and existing confidence-based approaches, while the confidence signals of both upstream and downstream agents show improved calibration quality. Intervention analysis further shows that upstream confidence has a measurable efect on downstream decisions, and the learned final confidence supports coverage-accuracy control through selective answering and targeted rechecking. In generalization experiments, models trained on a two-hop architecture transfer directly to deeper Web-retrieval chains, and the same recipe remains efective at smaller model scales.

Our contributions are as follows:

• To enable calibrated confidence communication in multi-agent collaboration, we propose MASCAL, which integrates calibration objectives into reinforcement learning so that each agent learns to produce a calibrated correctness probability for its answer, and further introduces confidence-aware weighting to guide the downstream agent’s use of upstream confidence.

• Across three collaboration architectures spanning Web retrieval and reasoning tasks, we demonstrate that calibrated confidence transmission improves task accuracy and calibration quality in MAS, and that upstream confidence measurably afects downstream decisions. MASCAL further generalizes to deeper collaboration chains and diferent model scales.

• We show that the learned final confidence supports selective answering and targeted rechecking, reducing the error rate of delivered outputs substantially, or directing additional computation to where it is most needed.

## 2 Related Work

In this section, we review related work along two lines: how agents in MAS learn to collaborate and how confidence and uncertainty signals have been introduced into multi-agent settings.

## 2.1 Multi-Agent Collaboration and Learning

LLM-driven MAS have been widely applied to Web tasks requiring multi-step reasoning and evidence gathering [6, 29, 50]. At the orchestration level, one line of work learns to select collaboration modes and role configurations per query [25, 37, 52, 53]. Given a fixed role assignment, another line focuses on training each role’s policy, curating data from successful and failed trajectories [32, 56] or designing rewards that combine correctness with interaction behavior [33]. Scaling up to the system level, other methods jointly improve multi-role performance [4, 5, 28] through terminal reward propagation [1], preference objectives [3, 42], and inter-role evaluation signals [46]. In Web retrieval settings, confidence has been used to control the expansion of query decomposition trees, balancing answer quality against search cost [22].

These methods improve MAS task performance through various mechanisms, yet their training objectives center on final-answer correctness or system eficiency. The reliability of handof content, specifically whether the upstream agent’s answer is likely correct, remains outside the scope of learning. MASCAL incorporates calibrated answer confidence into role-level training, making reliability judgments part of the information agents exchange.

## 2.2 Confidence and Uncertainty in MAS

Calibrated reinforcement learning has recently shown that a single LLM can learn to produce well-calibrated correctness probabilities for its answers, providing a foundation for introducing reliable confidence signals into MAS [8, 10, 11, 31]. In multi-agent settings, early attempts used self-reported confidence for weighted voting [2] or injected uncertainty into peer prompts to modulate answer updates in debate [49]. These untrained signals, however, have been found unreliable [17], sometimes causing unstable interaction behavior. Subsequent work has therefore begun to train confidence expression and utilization: incorporating the listener’s behavior into the speaker’s confidence learning [36], using correctness labels to train verbalized confidence and update behavior in debate [38, 58], or learning confidence scorers for speaker selection and listener filtering [40]. These methods focus primarily on debate and discussion, where all participants address the same question.

When collaboration goes beyond same-question debate and involves multi-step reasoning or role-level handofs, the scope of uncertainty broadens. Along this direction, work has studied how to propagate uncertainty through reasoning steps [55], let downstream agents inherit uncertainty from predecessor decisions [30], or use uncertainty to trigger task re-delegation in hierarchical systems [24]. At the output end, counterfactual calibration over communication graphs [20] and multi-source confidence composition [9] have been used to improve final-answer probability quality.

The methods above introduce confidence into MAS from complementary angles: expression training, signal utilization, crossrole propagation, and final-answer probability estimation. Each, however, addresses only part of the pipeline, and no existing framework connects confidence production, transmission, and utilization within a unified learning process. MASCAL fills this gap: each role learns to produce calibrated confidence through calibration RL, the downstream agent trains on reports actually generated by the upstream to incorporate this signal into its decisions, and confidence-aware weighting further strengthens this utilization.

![](images/f09c7e415ec74fb6d72e0b160e2b5e2e42a6a2b2eeffb66a3fe73106bfac787a.jpg)  
Figure 2: The MASCAL training pipeline. (a) Each rollout is split into a task segment and a confidence segment, trained with separate policy gradients and a BCE loss on the readout �. (b) Masking the upstream confidence reveals which downstream tokens depend on it; those positions are up-weighted during task learning.

## 3 Preliminary

In this section, we formalize the multi-agent collaboration setting and the notion of calibrated answer confidence, and introduce the policy optimization framework used throughout our method.

## 3.1 Task Formulation

We consider collaborative systems where a complex task � is decomposed into subtasks handled by multiple LLM agents in sequence. Each agent � receives its subtask $t _ { i } ,$ , initial context $o _ { i }$ (e.g., retrieved documents), and a set of upstream reports $M _ { i } ,$ produces a trajectory $\tau _ { i }$ containing reasoning actions, an answer $a _ { i } ,$ and supporting rationale $e _ { i } ,$ and passes a report to the downstream agent:

$$
m _ { i } = ( t _ { i } , \ a _ { i } , \ e _ { i } ) .\tag{1}
$$

The final agent in the chain produces the system’s answer to the original task �, based on its own subtask and all upstream reports.

To enable each agent to communicate a judgment of answer correctness at handof, we augment the report with answer confidence $c _ { i } \in [ 0 , 1 ]$ , the agent’s estimate of the probability that $a _ { i }$ is correct:

$$
m _ { i } = ( t _ { i } , \ a _ { i } , \ e _ { i } , \ c _ { i } ) .\tag{2}
$$

We aim for $c _ { i }$ to be calibrated with respect to actual correctness [16]:

$$
\begin{array} { r } { P \big ( V ( t _ { i } , a _ { i } ) = 1 \big | c _ { i } = p \big ) \approx p , \quad \forall p \in [ 0 , 1 ] , } \end{array}\tag{3}
$$

where $V ( t _ { i } , a _ { i } ) \in \{ 0 , 1 \}$ is a verification function that determines the correctness of $a _ { i }$ . Calibration is especially important in this multi-agent setting: unlike a single agent reporting to a human user, $c _ { i }$ is consumed automatically by another agent and directly afects its decisions. A miscalibrated probability can systematically bias downstream behavior, making calibration a prerequisite for efective confidence communication.

## 3.2 Group Relative Policy Optimization

We adopt Group Relative Policy Optimization (GRPO) [34] as our training framework. GRPO does not require a separate critic network; instead, it estimates advantages directly from rewards within each sampling group. For each input, the policy $\pi _ { \theta }$ samples � trajectories, and the advantage of each is the standardized deviation of its reward from the group mean:

$$
\hat { A } _ { g i } = \frac { r _ { g i } - \bar { r } _ { g } } { \mathrm { s t d } ( r _ { g } ) + \epsilon } ,\tag{4}
$$

where $r _ { g i }$ is the reward for the �-th trajectory in group ${ \mathit { g } } ,$ and $\bar { r } _ { g } ,$ $\operatorname { s t d } ( r _ { g } )$ are the group mean and standard deviation. The policy is

updated via an importance-weighted objective:

$$
\mathcal { L } _ { \mathrm { G R P O } } = - \frac { 1 } { G } \sum _ { i = 1 } ^ { G } \frac { 1 } { | \tau _ { i } | } \sum _ { t = 1 } ^ { | \tau _ { i } | } \frac { \pi _ { \theta } ( z _ { t } \mid z _ { < t } , x ) } { \pi _ { \theta _ { \mathrm { o l d } } } ( z _ { t } \mid z _ { < t } , x ) } \hat { A } _ { g i } ,\tag{5}
$$

where $\pi _ { \boldsymbol { \theta } _ { \mathrm { o l d } } }$ is the policy used to sample the trajectories. Because the advantage computation depends only on the reward function’s output, diferent reward signals can each produce their own advantages from the same group of rollouts, which provides the basis for the multi-signal training described next.

## 4 Method: MASCAL

In this section, we present MASCAL (Figure 2). We first describe how each agent learns to output calibrated confidence alongside its task answer, and then introduce confidence-aware downstream training, which trains the downstream agent to leverage the upstream confidence signal in its decisions.

## 4.1 Joint Task and Confidence Learning

In a standard MAS, each agent generates a trajectory containing reasoning actions and an answer. MASCAL additionally requires each agent to produce a natural-language confidence analysis assessing the reliability of its own answer. The full trajectory of agent � is split into a task segment $\tau _ { i } ^ { \mathrm { t a s k } }$ (reasoning, tool calls, and the answer) and a confidence segment $\tau _ { i } ^ { \mathrm { c o n f } }$ (reliability analysis; see Figure 2a):

$$
\tau _ { i } = [ \tau _ { i } ^ { \mathrm { t a s k } } ; \tau _ { i } ^ { \mathrm { c o n f } } ] .\tag{6}
$$

This segmentation carries over to training: task and confidence policy gradients are computed over their respective segments, keeping calibration learning separate from task behavior [11, 31].

At the end of the confidence segment, the confidence value is read out from the logits of two designated tokens H and L [7, 11]:

$$
c _ { i } = \sigma ( z _ { \mathrm { H } } - z _ { \mathrm { L } } ) ,\tag{7}
$$

where $\sigma$ is the sigmoid function. Unlike verbalized confidence, which requires sampling the probability as discrete tokens, this diferentiable readout allows gradients from a probability loss to propagate directly, providing finer-grained calibration supervision than reward signals alone.

Each agent is trained with three learning signals: the task policy gradient drives reasoning behavior, the confidence policy gradient shapes the quality of the analysis text, and a direct probability loss calibrates the numerical readout. We describe each below.

Task policy gradient. The task reward guides reasoning and tool use. Let $Y _ { i } = V \big ( t _ { i } , a _ { i } \big )$ denote correctness and � format compliance:

$$
r _ { i } ^ { \mathrm { t a s k } } = F \cdot Y _ { i } .\tag{8}
$$

The corresponding GRPO advantage (Eq. 4) is applied only to the task segment $\tau _ { i } ^ { \mathrm { t a s k } }$ of each trajectory.

Confidence policy gradient. The confidence reward measures how close the readout is to correctness when the output format is valid:

$$
r _ { i } ^ { \mathrm { c o n f } } = F \cdot \big ( 1 - ( Y _ { i } - c _ { i } ) ^ { 2 } \big ) .\tag{9}
$$

Its policy gradient is applied only to $\tau _ { i } ^ { \mathrm { c o n f } }$ , encouraging the model to produce analyses that better distinguish correct from incorrect answers, which in turn makes the logit readout more reliable.

Direct probability supervision. A BCE loss aligns the confidence readout directly with the correctness label:

$$
\mathcal { L } _ { i } ^ { \mathrm { B C E } } = - \frac { 1 } { | \mathcal { V } _ { i } | } \sum _ { j \in \mathcal { V } _ { i } } \big [ Y _ { i j } \log c _ { i j } + ( 1 - Y _ { i j } ) \log ( 1 - c _ { i j } ) \big ] ,\tag{10}
$$

where $\mathcal { N } _ { i }$ is the set of trajectories with valid confidence readouts. Gradients propagate directly through the readout, calibrating the probability value independently of the generated text.

The confidence policy gradient and the BCE loss complement each other: the former trains the model to write analyses that better separate correct from incorrect answers, and the latter directly adjusts the probability value through the readout. The joint objective for agent � combines all three:

$$
\mathcal { L } _ { i } = \mathcal { L } _ { i , \mathrm { t a s k } } ^ { \mathrm { P G } } + \mathcal { L } _ { i , \mathrm { c o n f } } ^ { \mathrm { P G } } + \alpha \mathcal { L } _ { i } ^ { \mathrm { B C E } } ,\tag{11}
$$

where � controls the weight of direct probability supervision.

## 4.2 Confidence-Aware Downstream Training

With the upstream agent trained to express calibrated confidence, the downstream agent needs to learn in an environment where it actually consumes these signals. The downstream agent is trained with the same joint objective (Eq. 11):

$$
\operatorname* { m i n } _ { \theta _ { B } } \ \mathbb { E } _ { { x } \sim { D _ { B } } , \mathcal { M } \sim \Pi _ { A } ( \cdot \vert { x } ) } \left[ \mathcal { L } _ { B } ( \theta _ { B } ; { x } , \mathcal { M } ) \right] ,\tag{12}
$$

where $D _ { B }$ is the downstream training set and $\Pi _ { A }$ the report distribution of the frozen upstream.

Standard policy gradients apply the same task advantage to every position and cannot distinguish which decisions are influenced by the upstream confidence. If a token’s generation probability changes substantially when the upstream confidence is present versus absent, the model is relying on the confidence signal at that position. Strengthening the learning signal at these positions rewards the pattern ofincorporating confidence into correct decisions, turning an implicit dependency into an explicit training objective. We operationalize this through confidence-aware weighting.

Given a downstream trajectory, we replace the upstream confidence in the report with an unavailability marker and re-score the trajectory with the same model (Figure 2b). The resulting perposition log-probability diference is:

$$
\Delta _ { t } = \log P _ { \theta } ( z _ { t } \mid x , z _ { < t } ) - \log P _ { \theta } ( z _ { t } \mid x _ { \backslash c } , z _ { < t } ) ,\tag{13}
$$

![](images/ed6669a119e5ac867f85350d30f9a4fae20c1b0b81a5c484a19e4c64857b6930.jpg)  
Figure 3: Collaboration architectures in our experiments.

Table 1: Dataset splits and question counts.
<table><tr><td>Task</td><td>Main dataset</td><td>Train / Val. / Test</td><td colspan="2">Additional tests (n)</td></tr><tr><td>Multi-hop QA</td><td>MuSiQue</td><td>12,938 / 719 / 1,252</td><td>HotpotQA 2Wiki</td><td>1,000 1,000</td></tr><tr><td>Logic</td><td>LogiQA2</td><td>9,000 / 500 / 1,572</td><td>ReClor AR-LSAT</td><td>500 230</td></tr><tr><td>Math</td><td>DeepScaleR</td><td>9,000 / 500 / 500</td><td>MATH500 AMC &amp; AIME</td><td>500 181</td></tr></table>

where $x _ { \backslash c }$ is the input with confidence masked and $z _ { < t }$ the generated tokens before position �. The absolute diference is normalized and centered around the trajectory mean to produce weights:

$$
s _ { t } = \mathrm { t a n h } ( | \Delta _ { t } | ) , \qquad w _ { t } = 1 + \lambda ( s _ { t } - \bar { s } ) ,\tag{14}
$$

with �¯ the trajectory-level mean and � controlling the adjustment magnitude. The weighted task advantage is:

$$
\widetilde { A } _ { g i t } ^ { \mathrm { t a s k } } = \mathrm { s t o p g r a d } ( w _ { g i t } ) \cdot \hat { A } _ { g i } ^ { \mathrm { t a s k } } ,\tag{15}
$$

$\widetilde { A } _ { q i t } ^ { \mathrm { t a s k } }$ replaces the original task advantage in $\mathcal { L } _ { B , \mathrm { t a s k } } ^ { \mathrm { P G } }$ (Eq. 11), so that positions more responsive to upstream confidence receive correspondingly stronger task learning signals. The weighting applies only to the task policy gradient; the confidence policy gradient and BCE loss are both left unchanged.

For the downstream agent to encounter realistic confidence signals during training, including cases where the upstream is uncertain or incorrect, it must train on reports produced by the actual upstream policy. The upstream agent is therefore first trained with the joint objective (Eq. 11) and then frozen to generate confidencebearing reports. The downstream agent trains on these reports with the same objective, with its task policy gradient additionally modulated by confidence-aware weighting, ensuring that the report distribution seen during training matches deployment.

## 5 Experiments

In this section, we first describe the experimental setup, then organize our analysis around five research questions:

• RQ1. Does calibrated confidence transmission improve task performance in multi-agent collaboration?

Table 2: Task accuracy (%) comparing MASCAL with single-agent , MAS without confidence , and confidence-based MAS baselines. Best results are bolded.
<table><tr><td></td><td colspan="4">Sequential handoff</td><td colspan="4">Report fusion</td><td colspan="4">Answer review (logic)</td><td colspan="4">Answer review (math)</td></tr><tr><td>Method</td><td>MuSiQue HotpotQA</td><td></td><td>2Wiki</td><td>Avg.</td><td>MuSiQue</td><td>HotpotQA</td><td>2Wiki</td><td>Avg.</td><td>LogiQA2</td><td>ReClor</td><td>AR-LSAT</td><td>Avg.</td><td></td><td>MATH500 AMC &amp; AIME</td><td>DeepScaleR</td><td>Avg.</td></tr><tr><td>Base Single-Agent</td><td>24.65</td><td>42.67</td><td>40.90</td><td>36.07</td><td>24.65</td><td>42.67</td><td>40.90</td><td>36.07</td><td>69.94</td><td>80.30</td><td>35.65</td><td>61.96</td><td>83.20</td><td>37.43</td><td>60.60</td><td>60.41</td></tr><tr><td>Task-RL Single-Agent</td><td>47.26</td><td>52.27</td><td>51.87</td><td>50.47</td><td>47.26</td><td>52.27</td><td>51.87</td><td>50.47</td><td>76.91</td><td>85.65</td><td>52.07</td><td>71.54</td><td>91.30</td><td>51.31</td><td>75.70</td><td>72.77</td></tr><tr><td>Base MAS</td><td>30.35</td><td>43.93</td><td>50.27</td><td>41.52</td><td>27.69</td><td>46.77</td><td>51.37</td><td>41.94</td><td>68.73</td><td>78.10</td><td>34.84</td><td>60.56</td><td>82.85</td><td>37.50</td><td>60.35</td><td>60.23</td></tr><tr><td>Task-RL MAS</td><td>43.32</td><td>51.90</td><td>47.27</td><td>47.49</td><td>45.07</td><td>58.27</td><td>61.00</td><td>54.78</td><td>77.53</td><td>86.25</td><td>57.77</td><td>73.85</td><td>91.30</td><td>51.38</td><td>76.05</td><td>72.91</td></tr><tr><td>MAGRPO</td><td>44.12</td><td>53.43</td><td>49.07</td><td>48.87</td><td>43.93</td><td>57.77</td><td>67.20</td><td>56.30</td><td>73.58</td><td>82.95</td><td>49.73</td><td>68.75</td><td>88.80</td><td>48.96</td><td>73.40</td><td>70.39</td></tr><tr><td>MHGPO</td><td>36.61</td><td>50.83</td><td>60.43</td><td>49.29</td><td>36.29</td><td>52.27</td><td>63.13</td><td>50.56</td><td>71.36</td><td>80.75</td><td>42.77</td><td>64.96</td><td>90.35</td><td>50.55</td><td>74.65</td><td>71.85</td></tr><tr><td>ConfMAD</td><td>34.58</td><td>47.93</td><td>57.83</td><td>46.78</td><td>34.58</td><td>51.13</td><td>59.33</td><td>48.35</td><td>71.33</td><td>81.30</td><td>37.50</td><td>63.38</td><td>77.45</td><td>27.76</td><td>51.90</td><td>52.37</td></tr><tr><td>DMAD</td><td>45.50</td><td>55.87</td><td>48.00</td><td>49.79</td><td>40.36</td><td>51.87</td><td>58.03</td><td>50.09</td><td>74.25</td><td>83.10</td><td>51.14</td><td>69.50</td><td>86.40</td><td>42.06</td><td>67.45</td><td>65.30</td></tr><tr><td>MASCAL</td><td>49.31</td><td>60.57</td><td>67.13</td><td>59.00</td><td>47.98</td><td>62.23</td><td></td><td>74.13 61.45</td><td>78.05</td><td>88.25</td><td>63.48</td><td>76.59</td><td>92.90</td><td>62.36</td><td>80.80</td><td>78.69</td></tr></table>

• RQ2. Is upstream confidence well calibrated, and how does it influence downstream decisions?

• RQ3. Is final-answer confidence well calibrated, and can it support selective answering and targeted rechecking?

• RQ4. What does each design component contribute?

• RQ5. Does MASCAL generalize to deeper collaboration chains and diferent model scales?

## 5.1 Experimental Setup

5.1.1 Collaboration architectures. We evaluate MASCAL and the multi-agent baselines under the three architectures in Figure 3:

• Sequential handof. A planner decomposes the original question. The upstream agent retrieves evidence, answers the sub-question, and hands of its report. The downstream agent continues retrieving and answers the original question.

• Report fusion. After the planner decomposes the question, two upstream agents independently answer the same sub-question and each delivers a report; the downstream agent synthesizes both and proceeds.

• Answer review. The upstream agent answers the full question and delivers a candidate answer; the downstream agent reviews and either retains or revises it.

5.1.2 Datasets and metrics. Our primary evaluation uses multihop QA [19, 39, 48], where agents retrieve evidence from Web documents and reason across them to answer complex questions. These questions naturally decompose into sub-questions, making them suitable for sequential handof and report fusion, where each agent handles a separate retrieval step. We further evaluate on logical reasoning with reading comprehension questions [27, 51, 57], and mathematical reasoning with competition-level problems [18]. These tasks involve a single reasoning process that is dificult to decompose, so we evaluate them under answer review, where one agent produces a candidate solution and the other examines and revises it. For each task type, we train on one dataset and evaluate on both its in-domain test set and additional out-of-domain datasets to assess generalization. Table 1 summarizes the splits.

We report task accuracy (Acc.) and assess confidence quality using the Brier score [14] and AUROC [12]:

$$
\begin{array} { l } { { \displaystyle \mathrm { B r i e r } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } ( c _ { i } - y _ { i } ) ^ { 2 } , } \ ~ } \\ { { \displaystyle \mathrm { A U R O C } = \mathrm { P r } ( c ^ { + } > c ^ { - } ) + \frac { 1 } { 2 } \mathrm { P r } ( c ^ { + } = c ^ { - } ) } , } \end{array}\tag{16}
$$

where $c _ { i }$ is the agent’s confidence for the �-th answer with binary correctness label �<sub>�</sub>, and $c ^ { + } , c ^ { - }$ denote the confidence on correct and incorrect answers, respectively.

5.1.3 Models and baselines. All agents are initialized from Qwen3- 8B [47]. We compare MASCAL against three groups of baselines: (1) Single-agent: the base model and its RL-trained variant, each solving the task without collaboration. (2) MAS without confidence: agents collaborate under the same architectures but transmit no confidence signal, including an untrained and an RL-trained MAS, as well as MAGRPO [28] and MHGPO [1], which optimize multirole performance through cooperative MARL and terminal reward propagation, respectively. (3) Confidence-based MAS: ConfMAD derives confidence from sequence-level probabilities with post-hoc calibrators [26]; DMAD trains verbalized confidence expression and its use through correctness rewards [58].

5.1.4 Implementation. We split the training data equally into disjoint subsets: one trains the upstream agent, which is then frozen to generate reports for downstream training on the other. For downstream training, we use a learning rate of $2 \times 1 0 ^ { - 6 } , G = 8$ rollouts per input, three epochs, and confidence-aware weighting strength � = 0.5. The calibration loss weight is � = 0.03 on MuSiQue and 0.01 on LogiQA2 and DeepScaleR. Retrieval experiments use a total search budget of three calls. Baselines were tuned from published settings, with validation performance for hyperparameter selection.

Full experimental settings are provided in Appendix A.

## 5.2 Main Results (RQ1)

Table 2 compares MASCAL with single-agent, multi-agent, and confidence-based baselines across four evaluation scenarios. MAS-CAL achieves the highest accuracy in all cases. The gains over Task-RL MAS, which shares the same architecture but transmits no confidence, are most pronounced in search-based collaboration, where an explicit reliability signal gives the downstream agent richer information for guiding retrieval and reasoning.

In sequential handof, Task-RL MAS falls below the single agent: the downstream agent inherits one upstream report with no indication ofits reliability, and uninformed collaboration hurts rather than helps. By attaching calibrated confidence at the handof, MASCAL raises accuracy well above the single-agent level.

Other multi-agent approaches do not match MASCAL’s improvements. Cooperative RL methods (MAGRPO, MHGPO) jointly optimize multiple roles but establish no reliability communication between them. MAGRPO outperforms Task-RL MAS on average under both retrieval architectures, but neither method improves consistently across the four scenarios. Confidence-based methods (ConfMAD, DMAD) do transmit a reliability signal, but post-hoc calibration and verbalized expressions yield limited accuracy gains over Task-RL MAS; on answer review, both fall below it. These results indicate that the confidence signal should be learned through calibration training alongside the task, and the downstream agent should also learn to act on it when making decisions.

## 5.3 Confidence Quality and Its Efect on Downstream Agents (RQ2)

This section analyzes the calibration quality of the upstream confidence signal and how it afects downstream behavior.

Table 3: Upstream confidence quality. Scores are averaged over test sets for logic and math (Brier↓, AUROC↑).
<table><tr><td></td><td colspan="2">MuSiQue</td><td colspan="2">Logic</td><td colspan="2">Math</td></tr><tr><td>Method</td><td>Brier</td><td>AUROC</td><td>Brier</td><td>AUROC</td><td>Brier</td><td>AUROC</td></tr><tr><td>Base MAS</td><td>0.1496</td><td>0.7666</td><td>0.3472</td><td>0.6316</td><td>0.3726</td><td>0.7973</td></tr><tr><td>Task-RL MAS</td><td>0.1576</td><td>0.7887</td><td>0.1952</td><td>0.7081</td><td>0.2513</td><td>0.8361</td></tr><tr><td>MASCAL-Verbal1</td><td>0.1440</td><td>0.6527</td><td>0.2403</td><td>0.5013</td><td>0.1092</td><td>0.7482</td></tr><tr><td>MASCAL</td><td>0.0972</td><td>0.8663</td><td>0.1516</td><td>0.7500</td><td>0.0827</td><td>0.8951</td></tr></table>

Upstream confidence quality. Table 3 evaluates the upstream agent’s confidence on its own subtask answers. The base model produces poorly calibrated confidence, with Brier scores exceeding 0.34 on logic and math. Task-only RL training improves AUROC, suggesting that task optimization can implicitly produce some discriminative signal, but Brier scores remain well above MASCAL’s, so the probability values are not reliable. MASCAL-Verbal shows unstable calibration across tasks, performing near MASCAL on some but dropping to chance-level AUROC on logic. Verbalized confidence goes through discrete token sampling, which introduces variance at inference and blocks direct gradient-based calibration during training. MASCAL achieves the best Brier and AUROC across all settings, indicating that combining the logit-based readout with calibration training is essential for reliable upstream confidence.

Confidence interventions. To isolate the causal efect of upstream confidence, we replace it with an unavailability marker (Hidden) or uniform random values (Random) at test time while keeping all other inputs unchanged (Table 4). In search-based collaboration, both interventions cause substantial accuracy drops and Brier degradation, confirming that the downstream agent relies on the confidence signal when making retrieval and reasoning decisions. Answer review shows a diferent pattern: hiding confidence leads to only minor accuracy drops, but replacing it with random values causes a clear degradation (math accuracy falls from 78.69% to 73.07%). When the signal is marked unavailable, the downstream agent falls back to content-based judgment and maintains accuracy. Random values, however, actively mislead its decisions, confirming that the model has learned to incorporate upstream confidence into its reasoning. Both interventions also degrade the downstream agent’s own Brier scores across all scenarios, showing that upstream confidence shapes not only task accuracy but also the calibration of the downstream agent’s output.

Table 4: Accuracy (%) and Brier score under upstream confidence interventions on a trained MASCAL system.
<table><tr><td></td><td colspan="2">Multi-hop QA</td><td colspan="2">Answer review</td></tr><tr><td>Confidence</td><td>Handoff</td><td>Fusion</td><td>Logic</td><td>Math</td></tr><tr><td>Original</td><td>59.00 / 0.1947</td><td>61.45 / 0.1940</td><td>76.59 / 0.1489</td><td>78.69 / 0.0816</td></tr><tr><td>Hidden</td><td>50.00 / 0.3041</td><td>54.98 / 0.3251</td><td>76.28 / 0.2237</td><td>77.42 / 0.1646</td></tr><tr><td>Random</td><td>49.32 / 0.3055</td><td>54.80 / 0.3580</td><td>73.86 / 0.2470</td><td>73.07 / 0.2621</td></tr></table>

![](images/781d34ea4da06eb687861d195185e64d3f2c61419f329fd351b932d025e2efae.jpg)

![](images/f358184ab33c4910161ed3580bf408c2e4bcbd89e2f7a54c6d4a1c05b26198a4.jpg)

![](images/c47c09fb064bfd479935f337d3d20413128b51e23437ead63f87d6b483bee5f4.jpg)  
Upstream confidence

![](images/3566ac61e6f871c0db5f96b6dd2820a5c66ead1440078bee85d4d4f1b3bc12bc.jpg)  
Figure 4: MASCAL’s behavior across ten confidence bins: additional search calls and answer revision rates. Fusion uses mean upstream confidence.

Downstream behavior. Figure 4 bins MASCAL’s downstream actions by upstream confidence. In search scenarios, lower confidence leads to more additional retrieval calls; in answer review, lower confidence is associated with higher revision rates. Together with the intervention results, these patterns confirm that calibrated confidence systematically shapes downstream decisions.

## 5.4 Final Confidence Quality and Utility (RQ3)

This section evaluates the confidence quality ofthe final answer and its practical utility for selective answering and targeted rechecking.

Final-answer confidence. Table 5 reports Brier scores and AU-ROC for the final answer’s confidence across all methods. MASCAL achieves the best or near-best scores in every scenario. Among MAS without confidence methods, Task-RL MAS produces higher Brier scores on search tasks than the single agent (0.4498 vs. 0.4275 on handof). Collaboration introduces additional uncertainty, and without a calibration mechanism, the downstream agent’s confidence quality degrades rather than improves. Both confidencebased baselines retain moderate discrimination on search tasks, but their AUROC drops to 0.52–0.58 on review, where neither post-hoc calibration (ConfMAD) nor debate-trained verbalized confidence (DMAD) produces signals that generalize to the handof collaboration setting. MASCAL-Verbal similarly shows unstable calibration quality across tasks, consistent with the upstream analysis in RQ2. MASCAL-Unweighted also achieves strong calibration, but MAS-CAL further improves Brier and AUROC in most scenarios through its confidence-aware weighting.

![](images/004f584d2bc60ef397ff0c22bf998d4f3c61681027ae12e1fe85433dda3febe6.jpg)

Table 5: Final confidence quality for single-agent , MAS without confidence , confidence-based MAS , and MASCAL variants , averaged over test sets (Brier↓, AUROC↑).
<table><tr><td></td><td colspan="2">Multi-hop QA</td><td colspan="2">Answer review</td></tr><tr><td>Method</td><td>Handoff</td><td>Fusion</td><td>Logic</td><td>Math</td></tr><tr><td>Base Single-Agent</td><td>0.3689 / 0.7262</td><td></td><td>0.3472 / 0.6316</td><td>0.3726 / 0.7973</td></tr><tr><td>Task-RL Single-Agent</td><td>0.4275 / 0.6873</td><td></td><td>0.1952 / 0.7081</td><td>0.2513 / 0.8361</td></tr><tr><td>Base MAS</td><td>0.4111 / 0.6528</td><td>0.3999 / 0.6671</td><td>0.3008 / 0.7183</td><td>0.3560 / 0.7655</td></tr><tr><td>Task-RL MAS</td><td>0.4498 / 0.6272</td><td>0.3965 / 0.5920</td><td>0.2266 / 0.6929</td><td>0.2608 / 0.8106</td></tr><tr><td>ConfMAD</td><td>0.3933 / 0.6957</td><td>0.3936 / 0.6937</td><td>0.3339 / 0.5616</td><td>0.4645 / 0.5778</td></tr><tr><td>DMAD</td><td>0.2919 / 0.7839</td><td>0.2859 / 0.7351</td><td>0.3023 / 0.5196</td><td>0.3440 / 0.5255</td></tr><tr><td>MASCAL-Verbal</td><td>0.3022 / 0.6965</td><td>0.3200 / 0.6857</td><td>0.2496 / 0.5192</td><td>0.1170 / 0.7600</td></tr><tr><td>MASCAL-Unweighted²</td><td>0.2212 / 0.7297</td><td>0.1970 / 0.7408</td><td>0.1517 / 0.7816</td><td>0.0865 / 0.8937</td></tr><tr><td>MASCAL</td><td>0.1947 / 0.8183</td><td>0.1940 / 0.7701</td><td>0.1489 / 0.7870</td><td>0.0816 / 0.8930</td></tr></table>

Selective answering. If final confidence is well calibrated, a system can improve output quality by answering only the questions it is most confident about [13]. Figure 5 ranks answers by confidence and plots accuracy at each coverage level. MASCAL’s curve lies above all other methods at every coverage level across all four scenarios. At 20% coverage, accuracy reaches 86–97% depending on the task, well above full-coverage levels, confirming that the confidence signal efectively concentrates correct answers in the high-confidence range. The other methods show substantially smaller accuracy spreads between low and full coverage, and MASCAL-Verbal is nearly flat on logic, indicating that its verbalized confidence carries almost no useful answer-ranking information for selective answering. These results demonstrate that calibrated final confidence gives the system the ability to actively control output quality: in scenarios requiring high reliability, reducing coverage substantially lowers the overall error rate.

Targeted rechecking. Selective answering improves reliability by withholding uncertain answers, but real deployments usually need to return an answer for every query. Beyond deciding which answers to drop, confidence can also indicate where to invest additional computation. Our search-based collaboration already operates under a fixed per-question search budget, and a natural question is whether this budget can be routed to the answers most likely to be wrong. We study this on the handof and fusion architectures by sorting answers by final confidence, sending the least-confident fraction back through the same MAS pipeline for one more round of retrieval, and comparing the accuracy gain against a random fraction of equal size (Table 6). Low-confidence rechecking improves accuracy monotonically with the budget on both architectures. Random rechecking stays near zero and even turns negative on handof, since additional retrieval on already-confident answers can overwrite correct ones. The gap widens with the budget, indicating that final confidence directs the extra search to the cases that most need it. Together with selective answering, this shows that calibrated final confidence is actionable: it decides not only which answers to return, but also where additional search should be spent.

Base MAS Task-RL MAS MASCAL-Verbal MASCAL  
Figure 5: Selective answering by final confidence. Accuracy is measured on the most confident answers at each coverage level, then averaged over test sets.  
Table 6: Targeted rechecking in MASCAL under a fixed budget: ΔAcc. (pp) when the budget is spent on the least-confident �% of answers versus a random �%.
<table><tr><td></td><td>Initial</td><td colspan="3">Rechecking budget</td></tr><tr><td>Scenario</td><td>Acc. (%)</td><td>5%</td><td>10%</td><td>20%</td></tr><tr><td>Handoff</td><td>59.00</td><td> $+ 0 . 1 5 \ / \ - 0 . 0 4$ </td><td> $+ 0 . 3 7 \ / \ - 0 . 0 9$ </td><td> $+ 0 . 7 4 / - 0 . 1 7$ </td></tr><tr><td>Fusion</td><td>61.45</td><td> $+ 0 . 2 4 \ / \ + 0 . 0 2$ </td><td> $+ 0 . 3 9 \ / \ + 0 . 0 5$ </td><td> $+ 0 . 6 6 \ / \ + 0 . 0 9$ </td></tr></table>

## 5.5 Ablation Study (RQ4)

We isolate the contribution of each training component by removing it individually from the downstream agent’s training (Table 7).

Table 7: Training ablations across four scenarios. Each pair reports Acc. (%) $/ \mathrm { \bf B r i e r } \downarrow ,$ averaged over test sets.
<table><tr><td></td><td colspan="2">Multi-hop QA</td><td colspan="2">Answer review</td></tr><tr><td>Variant</td><td>Handoff</td><td>Fusion</td><td>Logic</td><td>Math</td></tr><tr><td>MASCAL</td><td>59.00 / 0.1947</td><td>61.45 / 0.1940</td><td>76.59 / 0.1489</td><td>78.69 / 0.0816</td></tr><tr><td>w/o weighting</td><td>55.45 / 0.2212</td><td>57.26 / 0.1970</td><td>73.92 / 0.1517</td><td>78.32 / 0.0865</td></tr><tr><td>w/o calibration loss</td><td>52.37 / 0.3092</td><td>51.81 / 0.4419</td><td>74.47 / 0.2413</td><td>78.05 / 0.0929</td></tr><tr><td>w/o confidence reward</td><td>57.55 / 0.2102</td><td>54.46 / 0.2306</td><td>74.01 / 0.1538</td><td>78.08 / 0.0888</td></tr><tr><td>MASCAL-Verbal</td><td>49.97 / 0.3022</td><td>56.30 / 0.3200</td><td>73.86 / 0.2496</td><td>77.42 / 0.1170</td></tr><tr><td>Task-RL MAS</td><td>47.49 / 0.4498</td><td>54.78 / 0.3965</td><td>73.85 / 0.2266</td><td>72.91 / 0.2608</td></tr></table>

The calibration loss has the largest efect on confidence quality, with its removal causing the most severe Brier degradation across scenarios. On fusion, removing calibration loss drops accuracy below Task-RL MAS, which transmits no confidence at all. An uncalibrated signal can mislead the downstream agent more than the absence of one. The confidence reward plays a complementary role: its removal leads to moderate accuracy drops but only slight Brier increases, indicating that it mainly improves the quality of the generated reliability analysis while the probability value is shaped more by the calibration loss. Replacing the readout with verbalized confidence degrades calibration across all scenarios, consistent with the RQ2 finding that discrete token sampling hinders stable and reliable probability calibration.

Confidence-aware weighting primarily afects search scenarios, where its removal reduces accuracy by 3–4pp; review scenarios are less afected, consistent with the scenario-level diferences observed in the intervention analysis. All ablations have the smallest efect on math, where the clear structure of mathematical solutions allows the reviewing agent to judge correctness largely from the solution process. Each component’s removal causes measurable degradation in at least one scenario or metric, confirming that calibration loss, confidence gradient, weighting mechanism, and readout design each serve a distinct and necessary function in MASCAL.

## 5.6 Generalization (RQ5)

We now examine whether MASCAL generalizes beyond the archi tecture and model scale used in the preceding experiments.

Deeper collaboration chains. The sequential handof in our main experiments involves a single upstream-downstream handof. To test whether the learned confidence mechanism transfers to deeper architectures, we deploy the same checkpoints to 3-hop and 4-hop MuSiQue questions, forming deeper multi-hop retrieval and-reasoning chains with two and three successive handofs, each carrying a calibrated upstream confidence signal.

Table 8: Accuracy (%) and calibration on 3-hop and 4-hop MuSiQue questions, using checkpoints trained only on 2- hop data. Best results are bolded.
<table><tr><td></td><td colspan="3">3-hop</td><td colspan="3">4-hop</td></tr><tr><td>Method</td><td>Acc</td><td>Brier↓</td><td>AUROC↑</td><td>Acc</td><td>Brier↓</td><td>AUROC↑</td></tr><tr><td>Base MAS</td><td>15.61</td><td>0.3771</td><td>0.6767</td><td>6.10</td><td>0.3149</td><td>0.6279</td></tr><tr><td>Task-RL MAS</td><td>32.39</td><td>0.5209</td><td>0.6653</td><td>17.21</td><td>0.6737</td><td>0.5791</td></tr><tr><td>MASCAL-Unweighted</td><td>34.51</td><td>0.1914</td><td>0.7764</td><td>19.65</td><td>0.1759</td><td>0.7439</td></tr><tr><td>MASCAL</td><td>37.85</td><td>0.1903</td><td>0.7701</td><td>20.19</td><td>0.1741</td><td>0.7555</td></tr></table>

MASCAL achieves the highest accuracy under both architectures (Table 8). The learned confidence communication, where the upstream conveys a calibrated reliability judgment and the downstream adjusts its retrieval and reasoning accordingly, remains ef fective in deeper chains unseen during training: each downstream node can leverage the upstream confidence to inform its own decisions. The calibration gap further highlights the value of confidence training. The Brier score of Task-RL MAS deteriorates sharply with handof count, far exceeding Base MAS at 4-hop. Without a reliability signal between agents, the downstream has no basis to discount unreliable upstream answers; these errors accumulate along the chain, and task-only training leaves the resulting predictions overconfident at every hop. MASCAL’s Brier remains stable near the single-handof level, and AUROC stays in the 0.74–0.78 range. The calibration ability acquired through training is not overfit to the training architecture but serves as a transferable capability that continues to function in longer retrieval chains.

Model scale. To verify that the confidence learning mechanism does not depend on a specific model capacity, we replicate MASCAL with the same training recipe on Qwen3-1.7B and 4B, and compare alongside the 8B results from the main experiments.

MASCAL outperforms all baselines in both accuracy and cali bration quality at every scale (Table 9), confirming that calibration training and confidence-aware weighting are efective regardless of model capacity. The accuracy gain from confidence training grows with model scale, and calibration quality improves alongside it, suggesting that larger models benefit from the calibration objective on both dimensions. The calibration contrast is especially pronounced at smaller scales: both Base MAS and Task-RL MAS at 1.7B achieve near-chance AUROC, indicating that without calibration training, smaller models produce almost no discriminative signal regardless of task optimization. MASCAL achieves substantially better Brier and AUROC even at 1.7B, showing that the calibration objective equips models with reliable probability estimation independent of model scale. Confidence-aware collaboration thus remains practical and viable even at smaller model scales.

Table 9: Accuracy (%) and calibration on sequential handof across model scales. Best results per scale are bolded.
<table><tr><td></td><td colspan="3">1.7B</td><td colspan="3">4B</td><td colspan="3">8B</td></tr><tr><td>Method</td><td>Acc</td><td>Brier↓</td><td>AUROC↑</td><td>Acc</td><td>Brier↓</td><td>AUROC↑</td><td>Acc</td><td>Brier↓</td><td>AUROC↑</td></tr><tr><td>Base MAS</td><td>19.75</td><td>0.5964</td><td>0.4985</td><td>17.60</td><td>0.5794</td><td>0.6245</td><td>41.52</td><td>0.4111</td><td>0.6528</td></tr><tr><td>Task-RL MAS</td><td>40.01</td><td>0.5537</td><td>0.5036</td><td>41.08</td><td>0.4367</td><td>0.6892</td><td>47.49</td><td>0.4498</td><td>0.6272</td></tr><tr><td>MASCAL-Unweighted</td><td>44.50</td><td>0.2721</td><td>0.7000</td><td>45.73</td><td>0.2225</td><td>0.7460</td><td>55.45</td><td>0.2212</td><td>0.7297</td></tr><tr><td>MASCAL</td><td>46.82</td><td>0.2205</td><td>0.7115</td><td>51.24</td><td>0.2190</td><td>0.7481</td><td>59.00</td><td>0.1947</td><td>0.8183</td></tr></table>

## 6 Conclusion and Future Work

When multiple agents collaborate to retrieve and reason across Web content, the reliability of each agent’s intermediate conclusions directly shapes downstream decisions. We proposed MASCAL, which trains each agent to express calibrated answer confidence alongside its task and introduces confidence-aware weighting to train the downstream agent to act on this signal. Across three collaboration architectures, MASCAL improves task accuracy and calibration quality, and the learned confidence communication transfers to deeper Web-retrieval chains without additional training, while the same training recipe remains efective at smaller model scales. Intervention analysis and ablation experiments together show that neither producing a confidence signal nor exposing the downstream agent to it is suficient on its own: the signal must be made reliable through dedicated calibration training, and the downstream agent must also learn to incorporate it into its decisions.

As agentic collaboration on the Web scales to more sources and longer reasoning chains, the reliability challenges facing collaborative systems grow as well. The current study uses a fixed role assignment with all agents initialized from the same base model, and calibration relies on binary correctness feedback. Extending to heterogeneous MAS composed of diferent model families, and to open-ended generation tasks where correctness is not binary, calls for more flexible calibration objectives and end-to-end training across roles. Extending confidence learning to multi-round interaction and dynamic role assignment is a natural next step. In such settings, confidence signals could inform system-level decisions such as role selection and computational allocation. We plan to pursue these directions toward agentic systems in which calibrated confidence serves the full collaboration lifecycle, from each role’s reliability judgment to system-level resource decisions, providing a more principled and reliable foundation for increasingly complex multi-agent information processing on the Web.

## References

[1] Guanzhong Chen, Shaoxiong Yang, Chao Li, Wei Liu, Jian Luan, and Zenglin Xu. 2026. End-to-End Optimization of LLM-Driven Multi-Agent Search Systems via Heterogeneous-Group-Based Reinforcement Learning. In Proceedings of the 64th Annual Meeting ofthe Association for Computational Linguistics (Volume 1: Long Papers). 30319–30338.

[2] Justin Chen, Swarnadeep Saha, and Mohit Bansal. 2024. Reconcile: Round-table conference improves reasoning via consensus among diverse llms. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 7066–7085.

[3] Weize Chen, Jiarui Yuan, Chen Qian, Cheng Yang, Zhiyuan Liu, and Maosong Sun. 2025. Optima: Optimizing efectiveness and eficiency for llm-based multi agent system. In Findings ofthe Association for Computational Linguistics: ACL 2025. 11534–11557.

[4] Yiqun Chen, Jiaxin Mao, Yi Zhang, Dehong Ma, Long Xia, Jun Fan, Daiting Shi, Zhicong Cheng, Simiu Gu, and Dawei Yin. 2025. Ma4div: Multi-agent reinforcement learning for search result diversification. In Proceedings ofthe ACM on Web Conference 2025. 1703–1715.

[5] Yiqun Chen, Lingyong Yan, Weiwei Sun, Xinyu Ma, Yi Zhang, Shuaiqiang Wang, Dawei Yin, Yiming Yang, and Jiaxin Mao. 2026. Improving retrieval-augmented generation through multi-agent reinforcement learning. Advances in Neural Information Processing Systems 38 (2026), 121336–121367.

[6] Zehui Chen, Kuikun Liu, Qiuchen Wang, Jiangning Liu, Wenwei Zhang, Kai Chen, and Feng Zhao. 2025. Mindsearch: Mimicking human minds elicits deep ai searcher. In International Conference On Learning Representations, Vol. 2025. 90007–90029.

[7] Yu-Neng Chuang, Prathusha Kameswara Sarma, Parikshit Gopalan, John Boccio, Sara Bolouki, Xia Hu, and Helen Zhou. 2025. Learning to Route LLMs with Confidence Tokens. In Proceedings of the 42nd International Conference on Machine Learning. 10859–10878.

[8] Mehul Damani, Isha Puri, Stewart Slocum, Idan Shenfeld, Leshem Choshen, Yoon Kim, and Jacob Andreas. 2026. Beyond binary rewards: Training lms to reason about their uncertainty. In International conference on learning representations, Vol. 2026. 59456–59493.

[9] Ali Elahi and Barbara Di Eugenio. 2026. Multiagent Protocols with Aggregated Confidence Signals. arXiv preprint arXiv:2606.13591 (2026).

[10] Chenxiao Fan, Chongming Gao, Yaxin Gong, Haoyan Liu, Fuli Feng, and Xiangnan He. 2026. Uncertainty-aware generative recommendation. In Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 2. 1015–1026.

[11] Chenxiao Fan, Chongming Gao, Gangyi Zhang, Leyang Shen, Yaxin Gong, Ji amin Wang, Jiakai Wang, Dong Wang, Yang Liu, Fuli Feng, and Xiangnan He. 2026. Beyond Verbalized Confidence: Calibrating Reasoners with Diferentiable Readouts. arXiv preprint arXiv:2609.34857 (2026).

[12] Tom Fawcett. 2006. An introduction to ROC analysis. Pattern recognition letters 27, 8 (2006), 861–874.

[13] Yonatan Geifman and Ran El-Yaniv. 2019. Selectivenet: A deep neural network with an integrated reject option. In International conference on machine learning. PMLR, 2151–2159.

[14] Tilmann Gneiting and Adrian E Raftery. 2007. Strictly proper scoring rules, prediction, and estimation. Journal ofthe American statistical Association 102, 477 (2007), 359–378.

[15] Yaxin Gong, Gangyi Zhang, Chongming Gao, Leyang Shen, Chenxiao Fan, Jiakai Wang, Dong Wang, Yang Liu, Wenjie Wang, and Xiangnan He. 2026. When Upstream Messages Override Correct Answers: A Controlled Study of Multi Agent LLM Collaboration. arXiv preprint arXiv:2609.36855 (2026).

[16] Chuan Guo, Geof Pleiss, Yu Sun, and Kilian Q Weinberger. 2017. On calibration of modern neural networks. In International conference on machine learning. PMLR, 1321–1330.

[17] Jiuzhou Han, Wray Buntine, and Ehsan Shareghi. 2024. Towards uncertaintyaware language agent. In Findings of the Association for Computational Linguistics: ACL 2024. 6662–6685.

[18] Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. 2021. Measuring mathematical problem solving with the math dataset. arXiv preprint arXiv:2103.03874 (2021).

[19] Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara, and Akiko Aizawa. 2020. Constructing a multi-hop qa dataset for comprehensive evaluation of reason ing steps. In Proceedings of the 28th International Conference on Computational Linguistics. 6609–6625.

[20] Jiatan Huang, Mingchen Li, Ziming Li, Sunjae Kwon, Hong Yu, and Chuxu Zhang. 2026. Counterfactual Graph for Multi-Agent LLM Calibration. arXiv preprint arXiv:2605.30653 (2026).

[21] Yichen Jiang, Shikha Bordia, Zheng Zhong, Charles Dognin, Maneesh Singh, and Mohit Bansal. 2020. HoVer: A dataset for many-hop fact extraction and claim verification. In Findings ofthe Association for Computational Linguistics: EMNLP 2020. 3441–3460.

[22] Shuguang Jiao, Xinyu Xiao, Yunfan Wei, Shuhan Qi, Chengkai Huang, Quan Z Sheng, and Lina Yao. 2026. Prunerag: Confidence-guided query decomposition trees for eficient retrieval-augmented generation. In Proceedings ofthe ACM Web Conference 2026. 1923–1934.

[23] Bowen Jin, Hansi Zeng, Zhenrui Yue, Jinsung Yoon, Sercan Arik, Dong Wang, Hamed Zamani, and Jiawei Han. 2025. Search-r1: Training llms to reason and leverage search engines with reinforcement learning. arXiv preprint arXiv:2503.09516 (2025).

[24] John Knowlton, Aritra Guha, and Risto Miikkulainen. 2026. Semantic Uncertainty-Guided Orchestration in Hierarchical Multi-Agent Systems. arXiv preprint arXiv:2608.14707 (2026).

[25] Shiyuan Li, Yixin Liu, Yu Zheng, Mei Li, Quoc Viet Hung Nguyen, and Shirui Pan. 2026. OFA-MAS: One-for-all multi-agent system topology design based on mixture-of-experts graph generative models. In Proceedings of the ACM Web Conference 2026. 1333–1344.

[26] Zijie Lin and Bryan Hooi. 2025. Enhancing Multi-Agent Debate System Perfor mance via Confidence Expression. In Findings ofthe AssociationforComputational Linguistics: EMNLP 2025. Association for Computational Linguistics, 6453–6471.

[27] Hanmeng Liu, Jian Liu, Leyang Cui, Zhiyang Teng, Nan Duan, Ming Zhou, and Yue Zhang. 2023. Logiqa 2.0—an improved dataset for logical reasoning in natural language understanding. IEEE/ACM Transactions on Audio, Speech, and Language Processing 31 (2023), 2947–2962.

[28] Shuo Liu, Zeyu Liang, Xueguang Lyu, and Christopher Amato. 2026. Llm collaboration with multi-agent reinforcement learning. In Proceedings ofthe AAAI Conference on Artificial Intelligence, Vol. 40. 32150–32158.

[29] Yu Liu, Yanbing Liu, Fangfang Yuan, Cong Cao, Youbang Sun, Kun Peng, WeiZhuo Chen, Jianjun Li, and Zhiyuan Ma. 2026. OPERA: A Reinforcement Learning– Enhanced Orchestrated Planner-Executor Architecture for Reasoning-Oriented Multi-Hop Retrieval. In Proceedings ofthe AAAI Conference on Artificial Intelligence, Vol. 40. 32258–32266.

[30] Yaokun Liu, Yifan Liu, Daniel Yue Zhang, Ruichen Yao, Zelin Li, and Dong Wang. 2026. PropUQ-MAS: Propagation-Aware Uncertainty Quantification for LLM Multi-Agent Systems. arXiv preprint arXiv:2608.22130 (2026).

[31] Zhengzhao Ma, Xueru Wen, Boxi Cao, Yaojie Lu, Hongyu Lin, Jinglin Yang, Min He, Xianpei Han, and Le Sun. 2026. Decoupling Reasoning and Confidence: Resurrecting Calibration in Reinforcement Learning from Verifiable Rewards. In Proceedings ofthe 43rd International Conference on Machine Learning. 85080– 85098.

[32] Sumeet Ramesh Motwani, Chandler Smith, Rocktim Jyoti Das, Rafael Rafailov, Ivan Laptev, Philip HS Torr, Fabio Pizzati, Ronald Clark, and Christian Schroeder de Witt. 2024. Malt: Improving reasoning with multi-agent llm training. arXiv preprint arXiv:2412.01928 (2024).

[33] Chanwoo Park, Seungju Han, Xingzhi Guo, Asuman E Ozdaglar, Kaiqing Zhang, and Joo-Kyung Kim. 2025. Maporl: Multi-agent post-co-training for collaborative large language models with reinforcement learning. In Proceedings ofthe 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 30215–30248.

[34] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. 2024. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300 (2024).

[35] Zhengliang Shi, Shen Gao, Lingyong Yan, Yue Feng, Xiuyi Chen, Zhumin Chen, Dawei Yin, Suzan Verberne, and Zhaochun Ren. 2025. Tool learning in the wild: Empowering language models as automatic tool agents. In Proceedings of the ACM on Web Conference 2025. 2222–2237.

[36] Elias Stengel-Eskin, Peter Hase, and Mohit Bansal. 2024. LACIE: Listener-aware finetuning for calibration in large language models. Advances in Neural Information Processing Systems 37 (2024), 43080–43106.

[37] Jinwei Su, Qizhen Lan, Yinghui Xia, Lifan Sun, Weiyou Tian, Tianyu Shi, and Lewei He. 2026. Dificulty-aware agentic orchestration for query-specific multiagent workflows. In Proceedings ofthe ACM Web Conference 2026. 2060–2070.

[38] Luoxi Tang, Yuqiao Meng, Joseph Costa, Yingxue Zhang, Muchao Ye, and Zhaohan Xi. 2026. The Value of Variance: Mitigating Debate Collapse in Multi-Agent Systems via Uncertainty-Driven Policy Optimization. In Proceedings ofthe 43rd International Conference on Machine Learning. 118988–119005.

[39] Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. 2022. Musique: Multihop questions via single-hop question composition. Transactions ofthe Association for Computational Linguistics 10 (2022), 539–554.

[40] Mehmet Kerem Turkcan. 2026. Reinforcement Learning of Communication in a Mesh of Small Language Models. arXiv preprint arXiv:2609.30578 (2026)

[41] Rongzheng Wang, Shuang Liang, Qizhi Chen, Yihong Huang, Muquan Li, Yizhuo Ma, Dongyang Zhang, Ke Qin, and Man-Fai Leung. 2026. GraphCogent: Miti gating LLMs’ Working Memory Constraints via Multi-Agent Collaboration in Complex Graph Understanding. In Proceedings ofthe ACM Web Conference 2026. 3811–3822.

[42] Xiangwen Wang, Yibo Jacky Zhang, Zhoujie Ding, Katherine Tsai, Haolun Wu, and Sanmi Koyejo. 2026. Aligning compound ai systems via system-level dpo. Advances in Neural Information Processing Systems 38 (2026), 113728–113762.

[43] Zongwei Wang, Min Gao, Junliang Yu, Tong Chen, and Chenghua Lin. 2026. PAMAS: Self-Adaptive Multi-Agent System with Perspective Aggregation for Misinformation Detection. In Proceedings ofthe ACM Web Conference 2026. 6977– 6988.

[44] Zixiang Wang, Yinghao Zhu, Huiya Zhao, Xiaochen Zheng, Dehao Sui, Tian long Wang, Wen Tang, Yasha Wang, Ewen Harrison, Chengwei Pan, et al. 2025. Colacare: Enhancing electronic health record modeling through large language model-driven multi-agent collaboration. In Proceedings ofthe ACM on Web Conference 2025. 2250–2261.

[45] Jiasheng Xu, Mingda Li, Yongqiang Tang, Peijie Wang, and Wensheng Zhang. 2026. Towards open-world retrieval-augmented generation on knowledge graph: A multi-agent collaboration framework. In Proceedings of the ACM Web Conference 2026. 2171–2182.

[46] Xiangyuan Xue, Yifan Zhou, Guibin Zhang, Zaibin Zhang, Yijiang Li, Chen Zhang, Zhenfei Yin, Philip Torr, Wanli Ouyang, and Lei Bai. 2026. Comas: Coevolving multi-agent systems via interaction rewards. In International Conference on Learning Representations, Vol. 2026. 32373–32394.

[47] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. 2025. Qwen3 technical report. arXiv preprint arXiv:2505.09388 (2025).

[48] Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William Cohen, Ruslan Salakhutdinov, and Christopher D Manning. 2018. HotpotQA: A dataset for diverse, explainable multi-hop question answering. In Proceedings ofthe 2018 conference on empirical methods in natural language processing. 2369–2380.

[49] Luke Yofe, Alfonso Amayuelas, and William Yang Wang. 2025. DebUnc: Improving Large Language Model Agent Communication With Uncertainty Metrics.. In EMNLP (Findings). 23299–23315.

[50] Chuanyue Yu, Kuo Zhao, Yuhan Li, Heng Chang, Mingjian Feng, Xiangzhe Jiang, Yufei Sun, Jia Li, Yuzhi Zhang, Qingyun Sun, et al. 2026. Graphrag-r1: Graph retrieval-augmented generation with process-constrained reinforcement learning. In Proceedings ofthe ACM Web Conference 2026. 1398–1409.

[51] Weihao Yu, Zihang Jiang, Yanfei Dong, and Jiashi Feng. 2020. ReClor: A Reading Comprehension Dataset Requiring Logical Reasoning. In International Conference on Learning Representations (ICLR).

[52] Yanwei Yue, Guibin Zhang, Boyang Liu, Guancheng Wan, Kun Wang, Dawei Cheng, and Yiyan Qi. 2025. Masrouter: Learning to route llms for multi-agent systems. In Proceedings ofthe 63rd Annual Meeting ofthe Association for Computational Linguistics (Volume 1: Long Papers). 15549–15572.

[53] Guibin Zhang, Luyang Niu, Junfeng Fang, Kun Wang, Lei Bai, and Xiang Wang. 2025. Multi-agent Architecture Search via Agentic Supernet. In Proceedings of the 42nd International Conference on Machine Learning. 75834–75852.

[54] Wenlin Zhang, Kuicai Dong, Junyi Li, Yingyi Zhang, Xiaopeng Li, Pengyue Jia, Yi Wen, Derong Xu, Maolin Wang, Yichao Wang, et al. 2026. To search or not to search: Aligning the decision boundary of deep search agents via causal intervention. In Proceedings ofthe ACM Web Conference 2026. 2049–2059.

[55] Qiwei Zhao, Dong Li, Yanchi Liu, Wei Cheng, Yiyou Sun, Mika Oishi, Takao Osaki, Katsushi Matsuda, Huaxiu Yao, Chen Zhao, et al. 2025. Uncertainty propagation on llm agent. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 6064–6073.

[56] Wanjia Zhao, Mert Yuksekgonul, Shirley Wu, and James Zou. 2026. Sirius: Selfimproving multi-agent systems via bootstrapped reasoning. Advances in Neural Information Processing Systems 38 (2026), 124475–124504.

[57] Wanjun Zhong, Siyuan Wang, Duyu Tang, Zenan Xu, Daya Guo, Yining Chen, Jiahai Wang, Jian Yin, Ming Zhou, and Nan Duan. 2022. Analytical reasoning of text. In Findings ofthe Association for Computational Linguistics: NAACL 2022. 2306–2319.

[58] Xiaochen Zhu, Caiqi Zhang, Yizhou Chi, Tom Staford, Nigel Collier, and Andreas Vlachos. 2026. Demystifying multi-agent debate: The role of confidence and diversity. In Findings ofthe Association for Computational Linguistics: ACL 2026. 33909–33930.

## A Experimental Details

In this appendix, we describe the retrieval setup, the training and evaluation protocols, and the prompts used in our experiments.

## A.1 Retrieval Infrastructure

We use the Wiki-18 English Wikipedia dump released with Search-R1 [23] as the retrieval corpus. To keep facts needed during training and evaluation reachable, we additionally index the candidate paragraphs supplied by MuSiQue, HotpotQA, and 2Wiki. Retrieval is performed by E5-base-v2 over a flat inner-product index across the combined corpus. Each call returns the top three passages, each truncated to the first 200 words.

Each trajectory is allocated at most three retrieval calls, split across roles diferently in each architecture. In sequential handof, the upstream agent uses one call to answer the sub-question, and the downstream agent uses up to two more calls for the rest of the question. In report fusion, each of the two upstream agents uses one call on the shared sub-question, and the downstream agent uses up to two calls after reading both reports. The sub-question itself is produced by a Qwen3-8B planner used zero-shot, whose outputs are frozen for both training and evaluation.

## A.2 Training and Evaluation

All four training scenarios share the hyperparameters in Table 10, with the scenario-specific choices described below. Following prior work [11], we add H and L as reserved tokens whose embeddings are initialized from the existing high and low rows in the base vocabulary, giving the readout a semantically meaningful starting point. The training set is split into two equal, disjoint halves: the upstream agent is trained on one half with the joint objective, then frozen, and generates confidence-bearing reports on the other half; the downstream agent is subsequently trained on these reports.

Table 10: Training hyperparameters shared across scenarios.
<table><tr><td>Setting</td><td>Value</td><td>Setting</td><td>Value</td></tr><tr><td>Base model</td><td>Qwen3-8B</td><td>Rollouts G</td><td>8</td></tr><tr><td>Optimizer</td><td>AdamW (ZeRO-2)</td><td>PPO clipping €</td><td>0.2</td></tr><tr><td>Learning rate</td><td> $2 \times 1 0 ^ { - 6 }$ </td><td>Calibration warmup</td><td>30 steps</td></tr><tr><td>Epochs</td><td>3</td><td>Weighting strength λ</td><td>0.5</td></tr><tr><td>Inputs per step</td><td>128</td><td>Train sampling</td><td>T=1, top-p=1</td></tr></table>

The calibration loss weight is �=0.03 on MuSiQue and �=0.01 on LogiQA2 and DeepScaleR. For report fusion, we train two upstream agents with random seeds 43 and 44 so that they produce independent reports on the shared sub-question.

Evaluation. Retrieval scenarios sample with �=0.7, top-�=0.8, top-�=20 and average over three seeds. Review scenarios sample with �=0.7, top-�=1.0, top-� of, drawing multiple independent samples per question. Task correctness is measured by sub-EM [23] for retrieval, math-verify for math, and exact choice matching for logic; confidence quality is measured by the Brier score and AUROC against the per-answer correctness label.

## A.3 Prompt Templates

The system prompts for all roles are listed below.

Sequential handof and report fusion. Both retrieval architectures use the same pair of prompts: the upstream agent answers its subquestion from retrieved documents, and the downstream agent continues with its remaining budget to answer the full question. In report fusion, the downstream input simply contains two upstream reports instead of one.

## Retrieval, upstream agent

You are the first agent in a two-agent research team. You answer one sub-question of a larger question using the retrieved documents only; a teammate will use your report to finish the larger question. Reason briefly, then give your answer within <answer> </answer> tags. The answer must be as short as possible: only the entity, date, number or short phrase asked for, with no explanation and no extra qualifiers. Then cite at most two supporting sentences copied verbatim from the documents, each as <evidence doc="i"> sentence </evidence> where i is the document number. After the answer, analyze the uncertainty of your answer within <analysis> </analysis> tags: point out what could be wrong or ambiguous (missing or conflicting evidence, other possible readings); do not change your answer; if you find no uncertainty, say so explicitly. Then, within <confidence> </confidence> tags, output exactly one token: <CONF\_HIGH> if your answer is more likely correct than not, otherwise <CONF\_LOW>. Final format: brief reasoning, <answer> sub-answer here </answer> <evidence doc="i"> copied sentence here </evidence> <analysis> uncertainty analysis here </analysis> <confidence><CONF\_HIGH></confidence>.

## Retrieval, downstream agent

You are the second agent in a two-agent research team. A teammate has already answered the first sub-question of the question and sent you a report. The report covers only that sub-question; your job is to resolve the rest and answer the full question. Reason step by step. If you lack knowledge, you can call a search engine by <search> query </search>; it returns the top documents between <information> and </information>. Prefer facts found in retrieved documents over your own memory. Each search costs budget; you will be told how many searches remain. When you are ready, give the final answer to the full question within <answer> </answer> tags; it must answer the full question, never just the teammate’s sub-question. The answer must be as short as possible: only the entity, date, number or short phrase asked for, with no explanation and no extra qualifiers. After the answer, analyze the uncertainty of your answer within <analysis> </analysis> tags: point out what could be wrong or ambiguous (missing or conflicting evidence, other possible readings); do not change your answer; if you find no uncertainty, say so explicitly. Then, within <confidence> </confidence> tags, output exactly one token: <CONF\_HIGH> if your answer is more likely correct than not, otherwise <CONF\_LOW>. Final format: reasoning and searches, then <answer> final answer here </answer> <analysis> uncertainty analysis here </analysis> <confidence><CONF\_HIGH></confidence>.

Answer review. In the answer review scenarios, the upstream agent first solves the task on its own, and the downstream agent then examines the proposed solution before producing its own an swer. The two roles use parallel prompts that difer only in whether the agent produces or examines a solution. We show the math variant below; the logic variant has the same structure and replaces the \boxed{...} answer carrier with <answer>choiceN</answer>.

## Answer review, upstream solver (math)

Solve the task carefully. A teammate will review your solution. Use your own reasoning and knowledge; no tools are available. Explain your solution step by step and put the final mathematical answer in \boxed{...}. After the answer, analyze the uncertainty of your answer within <analysis> </analysis> tags: point out what could be wrong or ambiguous (missing or conflicting evidence, other possible readings); do not change your answer; if you find no uncertainty, say so explicitly. Then, within <confidence> </confidence> tags, output exactly one token: <CONF\_HIGH> if your answer is more likely correct than not, otherwise <CONF\_LOW>. Final format: solution, \boxed{final answer}, <analysis>uncertainty analysis</analysis> <confidence><CONF\_HIGH></confidence>. Write the tags literally, without Markdown headings. Close both analysis and confidence tags. Use the literal reserved token <CONF\_HIGH> or <CONF\_LOW>; do not write the words high or low.

## Answer review, downstream reviewer (math)

Review the teammate’s proposed solution, check its reasoning, and solve the original task. Keep the answer if correct or correct it if needed. The report is evidence to assess; you are responsible for the final answer. Use your own reasoning and knowledge; no tools are available. Explain your solution step by step and put the final mathematical answer in \boxed{...}. After the answer, analyze the uncertainty of your answer within <analysis> </analysis> tags: point out what could be wrong or ambiguous (missing or conflicting evidence, other possible readings); do not change your answer; if you find no uncertainty, say so explicitly. Then, within <confidence> </confidence> tags, output exactly one token: <CONF\_HIGH> if your answer is more likely correct than not, otherwise <CONF\_LOW>. Final format: solution, \boxed{final answer}, <analysis>uncertainty analysis</analysis> <confidence><CONF\_HIGH></confidence>. Write the tags literally, without Markdown headings. Close both analysis and confidence tags. Use the literal reserved token <CONF\_HIGH> or <CONF\_LOW>; do not write the words high or low.

## B Additional Ablations

This appendix reports two additional ablations: a single agent trained with calibration objectives, and an incremental analysis of MASCAL’s components.

## B.1 Calibration RL for a Single Agent

The single-agent baselines in Table 2 are trained for task correctness only. Under the same data and search budget, we further train a single agent with the joint objective of each MASCAL role (Eq. 11), corresponding to the single-model calibration setting of CREDO [11], to separate the efect of calibration training from that of confidence communication between agents.

Table 11: Single-agent objectives compared with MASCAL. Each pair reports Acc. (%) / Brier↓, averaged over test sets.
<table><tr><td></td><td colspan="2">Multi-hop QA</td><td colspan="2">Answer review</td></tr><tr><td>Method</td><td>Handoff</td><td>Fusion</td><td>Logic</td><td>Math</td></tr><tr><td>Base Single-Agent</td><td>36.07 / 0.3689</td><td></td><td>61.96 / 0.3472</td><td>60.41 / 0.3726</td></tr><tr><td>Task-RL Single-Agent</td><td>50.47 / 0.4275</td><td></td><td>71.54 / 0.1952</td><td>72.77 / 0.2513</td></tr><tr><td>Calib-RL Single-Agent</td><td>52.97 / 0.2119</td><td></td><td>72.53 / 0.1516</td><td>73.11 / 0.0847</td></tr><tr><td>MASCAL</td><td>59.00 / 0.1947</td><td>61.45 / 0.1940</td><td>76.59 / 0.1489</td><td>78.69 / 0.0816</td></tr></table>

Calibration training brings the single agent’s Brier scores close to MASCAL’s without reducing accuracy, but its accuracy gains over Task-RL Single-Agent are small and remain well below MASCAL in every scenario. A single agent’s confidence has no consumer within the task; it turns into accuracy only when a downstream agent relies on it to search further or revise an answer. Calibration training alone is therefore insuficient: MASCAL’s accuracy gains rely on a collaboration mechanism that passes calibrated confidence to a downstream agent that acts on it.

## B.2 Component Analysis of MASCAL

The training ablation in Table 7 removes components from the full MASCAL, but each MASCAL variant there still passes confidence between agents, so it cannot tell how much of the gain comes from calibration training itself and how much from transmitting the confidence. We therefore take the opposite direction: starting from Task-RL MAS, we add MASCAL’s three components one at a time to trace how each contributes to accuracy and calibration (Table 12):

• Cal.: confidence policy gradient and BCE loss.

• Trans.: upstream confidence included in the report.

• Wt.: confidence-aware weighting in downstream training.

Table 12: Component analysis of MASCAL. Each pair reports Acc. (%) / Brier↓, averaged over test sets.
<table><tr><td colspan="2">Components</td><td colspan="2">Multi-hop QA</td><td colspan="2">Answer review</td></tr><tr><td></td><td>Cal. Trans. Wt.</td><td>Handoff</td><td>Fusion</td><td>Logic</td><td>Math</td></tr><tr><td></td><td></td><td>47.49 / 0.4498</td><td>54.78 / 0.3965</td><td>73.85 / 0.2266</td><td>72.91 / 0.2608</td></tr><tr><td>√</td><td></td><td>53.28 / 0.2919</td><td>56.23 / 0.3016</td><td>73.89 / 0.1650</td><td>76.01 / 0.0918</td></tr><tr><td>√</td><td>√</td><td>55.45 / 0.2212</td><td>57.26 / 0.1970</td><td>73.92 / 0.1517</td><td>78.32 / 0.0865</td></tr><tr><td>√</td><td>√</td><td>59.00 / 0.1947</td><td>61.45 / 0.1940</td><td>76.59 / 0.1489</td><td>78.69 / 0.0816</td></tr></table>

With calibration training alone, the Brier score drops in every scenario, and accuracy rises clearly in sequential handof and on math but changes little in report fusion and on logic. The downstream agent is still less calibrated than the calibrated single agent in Table 11, especially in search-based collaboration: it builds its final answer on an upstream answer whose reliability it cannot assess, and this uncertainty carries into its own confidence. Answer review is less afected because the downstream agent can check the full proposed solution directly. Passing the upstream confidence in the report closes most of this calibration gap and raises accuracy further, especially in sequential handof and on math. In report fusion and on logic, most of the accuracy gain appears only once confidence-aware weighting trains the downstream agent to act on the signal, and weighting also adds a clear gain in sequential handof. Calibration training thus improves confidence quality throughout and already lifts accuracy in some scenarios, whereas the remaining gains depend on passing this confidence downstream and training the downstream agent to use it.

## C Training Overhead

Confidence-aware weighting replaces the upstream confidence with an unavailability marker and scores each generated downstream trajectory once more, introducing additional computation during training. We compare MASCAL with MASCAL-Unweighted across the four scenarios and measure the complete upstream and downstream training-iteration time with and without confidence-aware weighting. We run both variants on eight 80-GB GPUs using the training configuration in Appendix A.

improving accuracy and calibration in every scenario (Table 7), with the largest accuracy gains in search-based collaboration. Since the scoring pass only computes training weights, the downstream agent at deployment simply reads the confidence from the upstream report and answers without an extra scoring step.

Table 13: Training-iteration time with and without confidence-aware weighting.
<table><tr><td></td><td colspan="2">Time (h)</td><td></td></tr><tr><td>Scenario</td><td>w/o weighting</td><td>MASCAL</td><td>Extra time</td></tr><tr><td>Sequential handoff</td><td>8.13</td><td>9.28</td><td>+14%</td></tr><tr><td>Report fusion</td><td>9.95</td><td>10.29</td><td>+3%</td></tr><tr><td>Answer review (logic)</td><td>9.61</td><td>10.45</td><td>+9%</td></tr><tr><td>Answer review (math)</td><td>28.69</td><td>31.16</td><td>+9%</td></tr></table>

Across the four scenarios, confidence-aware weighting increases the total training-iteration time by only 3–14%, since the extra pass scores existing trajectories rather than generating new ones. In return, the downstream agent learns to act on upstream confidence,