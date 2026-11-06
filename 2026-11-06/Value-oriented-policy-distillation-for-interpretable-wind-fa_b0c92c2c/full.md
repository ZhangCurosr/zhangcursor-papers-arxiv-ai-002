# Value-oriented policy distillation for interpretable wind farm power control

Max Bruninx, Timothy Verstraeten, Jan Decuyper, Ann Nowé, Jan Helsen

Vrije Universiteit Brussel, Brussels, Belgium

## Abstract

Wind farm power control can help the system maintain balance and avoid blackouts. In Europe, this can be achieved through implicit balancing, where a wind farm adjusts its power setpoint in anticipation of the imbalance price using information published by the Transmission System Operator. Thereby, the wind farm helps restore system balance while also reducing its own imbalance costs resulting from forecast errors. However, implicit balancing is a challenging problem since the settlement price is unknown in advance and can be impacted by the pursued control strategy. Reinforcement learning methods could help to solve this problem, but lack the interpretability needed to be deployed on real wind farms. Therefore, we propose a policy distillation framework using explainable boosting machines (EBMs), a variant of generalized additive models, to obtain a transparent control policy. To increase the value captured by the EBM model, we introduce a valueoriented perspective to policy distillation inspired by research in the field of energy forecasting. Furthermore, we suggest to include expert knowledge during model design, such that the control strategy is also justifiable to stakeholders. Using real data of the Belgian grid and an ofshore wind farm, we find that our value-oriented and expert-based controller can reduce imbalance costs by 82%, while outperforming a price threshold controller as well as a policy distillation baseline using decision trees.

Keywords: ofshore wind farm, implicit balancing, reinforcement learning, interpretability

## 1 Introduction

Electricity supply and demand must be equal at all times to avoid blackouts. This has, however, become more and more challenging as renewable generation capacity continues to grow (IEA 2025). Therefore, the Transmission System Operators (TSOs), who are responsible for grid stability, rely on flexible assets that can help counteract system imbalances. The growing need for flexibility has driven a significant increase in battery storage projects, but controllable renewable assets can also contribute to this (Elia Transmission Belgium SA 2025a). In Europe, one way to provide flexibility as a renewable asset is by allowing the power setpoint to deviate from its trading schedule in response to real-time information published by the TSO, a practice known as implicit balancing. The producer is then rewarded or penalized for this deviation during imbalance settlement, depending on whether it contributed to restoring system balance. Renewable assets can participate in implicit balancing by curtailing their power output in case the system has a surplus. This practice could reduce imbalance costs resulting from forecast errors and improve their overall revenue. Finding these new revenue streams is particularly important for ofshore wind farms, since subsidies have recently been decreasing (see e.g., TotalEnergies SE (2024); Rijkdienst voor Ondernemend Nederland (2024)).

Implicit balancing is, however, a challenging control problem, since the settlement price is only known after the settlement period and a setpoint that negatively afects system imbalance can result in serious financial losses. Furthermore, this study focuses on ofshore wind farms, whose large installed capacity relative to typical system imbalances means that their actions can influence the imbalance price. Reinforcement learning (RL), and actor-critic methods in particular, have been proposed to solve this problem and were found to achieve good performance (Madahi, Claessens, and Develder 2024; Ally et al. 2025; Madahi et al. 2025; Rasic et al. 2025; Bruninx et al. 2026b). Unfortunately, these RL-based controllers are rarely deployed in practice due to their black-box nature. To facilitate deployment on an actual wind farm, a controller must demonstrate interpretability, which comprises two aspects: First, the model should be transparent, meaning that the efect of the input variables on the model or control policy should be clear. Second, the input variables should have a clear definition and the actions should be justifiable within the domain. The latter implies that a market expert must be able to explain each decision to stakeholders such as the TSO and the wind farm operator. To achieve interpretability, the model or control policy should be developed in interaction with experts from the field.

The goal of our work is to explore whether we can build an interpretable controller based on the control policy learned with soft actor-critic (SAC). To this end, we propose a policy distillation framework using explainable boosting machines (EBMs) (Lou et al. 2013; Nori et al. 2019), a variant of generalized additive models (GAMs), which were shown to be transparent and attain high predictive power. To the best of our knowledge, we are the first paper to apply EBMs in this context. Moreover, two improvements are made to the EBM model. First, to achieve interpretability, we develop an expert-based model in which the univariate and interaction efects are selected based on the explanations from the complete model combined with expert input on the drivers of implicit balancing. Second, to emphasize the high-value actions during policy distillation, we introduce value-oriented policy distillation, where the distillation loss is weighted based on the return associated with each state-action pair.

In our experiments, the proposed method is applied to Belgian data for the years 2022 and 2023. The results indicate that our value-oriented and expert-based control policy can reduce imbalance costs by 82%. While the controller was not deployed on a real wind farm, we took a lot of care to make the experiments close to reality. More specifically, we used real measurements from the supervisory control and data acquisition (SCADA) system of Norther, an ofshore wind farm in the Belgian ofshore zone. To account for the price impact of the control strategy, we have implemented the calculation method used by the Belgian TSO, which makes it possible to derive the price for diferent scenarios using historical data.

## 2 Related work

In our work, we follow the definition of interpretability by Glanois et al. (2024), which makes the important distinction against explainability in the field of artificial intelligence. Explainability relies on an external method or proxy to explain the predictions made by a given model. Examples are posthoc explanations using techniques such as SHAP (Lundberg and Lee 2017) or permutation feature importance (Altmann et al. 2010). In contrast, interpretability requires transparency to be an intrinsic property of the model, by using techniques such as decision trees or linear models and avoiding complex input variables. However, in addition to Glanois et al. (2024), we argue that interpretability also requires the model to be justifiable within the domain. For example, we can have a transparent model where the efect of an input variable cannot be explained by a domain expert to other stakeholders, then this efect is not justifiable and should be removed.

In the context of RL, interpretability can be achieved by immediately employing transparent models when learning the RL policy, using, e.g., tree-based methods (Likmeta et al. 2020; Silva and Gombolay 2021) or genetic programming (Hein, Udluft, and Runkler 2018). Alternatively, one can first learn an RL policy using a non-interpretable technique and then apply policy distillation (Rusu et al. 2015) with a transparent model, which is the approach we follow in this study. To this end, deterministic decision trees were employed by Liu et al. (2018) and soft decision trees by Coppens et al. (2019). Although soft decision trees are generally considered transparent, they only produce filters for each node which are dificult to explain to end users. Therefore, in Gokhale et al. (2024), the authors map the soft decision trees to deterministic rules at inference. On the contrary, this work adopts EBMs, which are transparent by design and require no postprocessing.

Within policy distillation for interpretable RL, another distinction can be made between value-based and policy-based distillation. Examples of the former are Liu et al. (2018); Gokhale et al. (2024), where the aim is to replicate the Qfunction learned by a deep Q-network. In contrast, the objective of the latter is to directly distill the explicit policy learned by the RL algorithm (see, e.g., Coppens et al. (2019)). In our work, we pursue the second approach. Nonetheless, drawing inspiration from the field of value-oriented forecasting (Zhang, Wang, and Hug 2022; Wang et al. 2023; Smets et al. 2025), we propose to include a value perspective while distilling the explicit policy as explained in Section 5.

## 3 Soft actor-critic

## 3.1 Formulation as a Markov Decision Process

To enable applying RL methods to this problem setting, we first formulate the implicit balancing problem as a Markov Decision Process (MDP) using a tuple $( S , { \mathcal { A } } , { \mathcal { P } } , { \mathcal { R } } , \gamma )$ . The state and action spaces are denoted by $s$ and ${ \mathcal { A } } ,$ , respectively. The reward function is defined as $\mathcal { R } : \mathcal { S } \times \mathcal { A } $ R and the transition function as $\mathcal { P } : \mathcal { S } \times \mathcal { A } \times \mathcal { S }  [ 0 , 1 ]$ . Last, $\gamma \in ( 0 , 1 ]$ denotes the discount factor for future rewards.

To transition towards the next state, we have built a simulator that models the dynamics of the wind farm and the reserve market. First, we apply the power setpoint $x _ { t }$ to the wind farm and assess the actual power production $P _ { t }$ . During this step, we account for the possible power $\overline { { P } } _ { t } { } ^ { 1 }$ as well as other constraints on the power setpoints imposed by the wind farm operator. We then compute the residual system imbalance, which includes the diference between the power production $P _ { t }$ and the trading schedule $v _ { t }$ of the wind farm, and use it to determine which reserves should be activated to resolve the imbalance. To this end, we employ a similar set-up to Bruninx et al. (2026a) but apply the actual price formula used in Belgium (Elia Transmission Belgium SA 2024). Furthermore, we also include the efect of the balancing strategy pursued by the TSO (Håberg and Doorman 2016) using historical data on proactive reserve actions.

Given that there is a publication delay for the information published by the $\mathrm { T S O } ^ { 2 }$ , we are dealing with a partially observable MDP. Therefore, for each timestep $t ,$ the state space $s$ is defined as:

• The latest published imbalance price estimate: $\widehat { \lambda } _ { t } ^ { \mathrm { B } }$

• A forecast for the system imbalance during the imbalance settlement period (ISP) based on the minute-level system imbalances: $\hat { S I } _ { t } ^ { \mathrm { B P } }$

• The accumulated position during the settlement period up to time t: $\begin{array} { r } { \Delta E _ { t } \dot { = } \sum _ { i \in \mathrm { I S P } , i < t } \mathsf { \bar { ( } } P _ { i } - v _ { i } ) \Delta t } \end{array}$

• The estimated power forecast error: $\hat { \varepsilon } _ { t } ~ = ~ \overline { { P } } _ { t - 1 } - v _ { t }$ which approximates the power forecast error using the possible power at the previous timestep.

• The minute of the settlement period: $t ^ { \mathrm { I S P } } \in [ 0 , 1 4 ]$

The actions are defined as the deviation of the setpoint $x _ { t }$ from the trading schedule $v _ { t } \colon$

$$
a _ { t } = x _ { t } - v _ { t } \in { \mathcal { A } } ,\tag{1}
$$

with a continuous action space ${ \mathcal { A } } \subset \mathbb { R }$

The reward is obtained at the end of every settlement period and is given by:

$$
r _ { t } = \lambda ^ { \mathrm { B } } \sum _ { i \in \mathrm { I S P } } ( P _ { i } - v _ { i } ) \Delta t ,\tag{2}
$$

where $\lambda ^ { \mathrm { B } }$ denotes the settlement price.

In contrast to batteries participating in implicit balancing (see, e.g., Madahi, Claessens, and Develder (2024); Madahi et al. (2025); Rasic et al. (2025)), wind farm power production cannot be stored for later use. We therefore set the episode length to match the settlement period (15 minutes). The discount factor γ is set to 1, since every minute within the settlement period contributes equally to the reward received at its end.

## 3.2 Learning the optimal policy

The optimal policy is obtained using soft actor-critic (Haarnoja et al. 2018a), an of-policy deep RL method that employs an entropy term to encourage exploration. The policy is parameterized by an actor network $\pi _ { \phi }$ , which maps states to a distribution over actions, and a critic network $Q _ { \theta } ,$ which estimates the expected return of state-action pairs and guides the actor’s updates. To stabilize training, SAC employs twin critic networks and updates their target networks $Q _ { \theta ^ { \prime } }$ via Polyak averaging.

At each training step, a minibatch of transitions $( s , a , r , s ^ { \prime } )$ is sampled from the replay bufer D, with $a ^ { \prime }$ denoting the action related to the next state $s ^ { \prime } .$ . The actor is trained to maximize expected return while maintaining high entropy, giving the loss

$$
J _ { \pi } ( \phi ) = \mathbb { E } _ { s \sim \mathcal { D } , a \sim \pi _ { \phi } } [ \alpha \ln \pi _ { \phi } ( a | s ) - Q _ { \theta } ( s , a ) ] ,\tag{3}
$$

where $\alpha$ denotes the temperature parameter, which weighs the entropy term against the Q-value. The critic is trained by minimizing the soft Bellman residual

$$
J _ { Q } ( \theta ) = \mathbb { E } _ { ( s , a ) \sim \mathcal { D } } [ ( y - Q _ { \theta } ( s , a ) ) ^ { 2 } ] ,\tag{4}
$$

with target

$$
\begin{array} { r } { y = r + \gamma \mathbb { E } _ { s ^ { \prime } \sim \mathcal { D } , a ^ { \prime } \sim \pi _ { \phi } } \left[ Q _ { \theta ^ { \prime } } \left( s ^ { \prime } , a ^ { \prime } \right) - \alpha \ln \pi _ { \phi } \left( a ^ { \prime } | s ^ { \prime } \right) \right] } \end{array}\tag{5}
$$

Finally, following Haarnoja et al. (2018b), the temperature parameter α is not fixed but learned via gradient descent on

$$
J ( \alpha ) = \mathbb { E } _ { s \sim \mathcal { D } , a \sim \pi _ { \phi } } [ - \alpha ( \ln \pi _ { \phi } ( a | s ) + \mathcal { H } ) ] ,\tag{6}
$$

where H is the target entropy.

## 4 Explainable boosting machines

Explainable boosting machines (Nori et al. 2019) are generalized additive models which also include pairwise interaction terms $( \mathrm { G A } ^ { 2 } \mathrm { M } )$ (Lou et al. 2013). For regression, the models have the form

$$
\mathbb { E } \left[ y \right] = \beta _ { 0 } + \sum f _ { i } \left( z _ { i } \right) + \sum f _ { i , j } \left( z _ { i } , z _ { j } \right) ,\tag{7}
$$

where $\beta _ { 0 }$ denotes the intercept and $z _ { i }$ the input features. Unlike traditional GAMs, which fit the shape functions $f _ { i }$ using splines, EBMs learn each function by combining bagging and cyclic gradient boosting of shallow decision trees.

The models with pairwise interactions have proven more powerful than traditional GAMs (Hastie and Tibshirani 1986), which only model univariate efects, while remaining transparent. Specifically, the relationship between the response variable and the input variables can be visualized through a 2D-plot (univariate efects) or a heatmap (interaction efects). Moreover, for each prediction, the final outcome is obtained as the sum of all the terms, allowing for local explanations.

In a $\mathrm { G A ^ { 2 } M }$ model, all univariate efects and the most important interaction efects, selected using the FAST algorithm (Lou et al. 2013), are included. While this results in an accurate and transparent model, the model explanations are not necessarily justifiable based on the domain knowledge of an expert. Therefore, we propose to first train a full model to achieve the best possible accuracy and then prune or extend the model based on expert input. By doing so, the final model can have a reduced accuracy but will be interpretable.

## 5 Value-oriented policy distillation

The goal of policy distillation is to transfer the knowledge from a teacher agent $\pi ^ { T }$ to a student agent $\pi ^ { S }$ (Rusu et al. 2015). To this end, the teacher agent is first used to generate a dataset $\mathcal { D } ^ { T } = \langle ( s _ { i } , a _ { i } ) \rangle _ { i = 1 } ^ { M }$ . Afterwards, the student agent can be trained to replicate the policy of the teacher. In a continuous action setting, this can be done by minimizing the loss function

$$
L \left( \pi ^ { T } , \pi ^ { S } \right) = \frac { 1 } { M } \sum _ { i = 1 } ^ { M } \left( a _ { i } - \hat { a } _ { i } \right) ^ { 2 } ,\tag{8}
$$

where $a _ { i }$ and $\hat { a } _ { i }$ are the actions given state $s _ { i }$ for, respectively, the teacher and the student agent.

However, directly distilling the policy of the teacher agent does not make a distinction between high and low value actions within the policy. Nonetheless, we can extend the dataset previously generated to include the return associated with each state-action pair $g = \langle g _ { i } \rangle _ { i = } ^ { M }$ and extend the loss function to

$$
L \left( \pi ^ { T } , \pi ^ { S } \right) = \frac { 1 } { M } \sum _ { i = 1 } ^ { M } \omega \left( g _ { i } \right) \left( a _ { i } - \hat { a } _ { i } \right) ^ { 2 } ,\tag{9}
$$

where $\omega : \mathbb { R }  \mathbb { R } ^ { + }$ is a function which maps the returns to a desired weight depending on the problem setting. By doing $\mathbf { S O } ,$ the efective training distribution is shifted toward higher-value actions. Since, in our setting, $\gamma = 1$ and the episode length is set equal to the settlement period, $g _ { i }$ equals the imbalance settlement payment during the period of that timestep. Note that, alternatively, the value weights could be obtained based on the Q-value predictions of the critic network instead of the observed returns.

## 6 Numerical experiments

## 6.1 Set-up

Experiments are conducted using Belgian data of the years 2022 and 2023. The data comes from the Open Data Platform by Elia (Elia Transmission Belgium SA 2025b) and the SCADA system of Norther, a Belgian ofshore wind farm with an installed capacity of 370 MW. In our set-up, we assume the wind farm ofers its day-ahead power forecast to the market instead of bidding strategically (see e.g., Heiser, Pourahmadi, and Kazempour (2025); Bruninx et al. (2025); Heiser, Kazempour, and Pourahmadi (2026)). The power forecasts are obtained using the approach described in Bruninx et al. (2027), while the system imbalance forecasts are produced with Chronos-2 (Ansari et al. 2025). All other variables can be derived from the data directly.

The SAC and interpretable controllers were trained using the first year and a half, whereas the last 6 months were used for evaluation. As baselines, we include decision trees (DTs), with a tree depth of 2 and 3, and a price threshold controller, which is often used in industry. The details of the price threshold controller are described in Appendix A.1. We also evaluated soft decision trees (Frosst and Hinton 2017) as an alternative baseline. However, they did not perform competitively in our setting (cf., Appendix A.2) and were therefore excluded from the main experiments. The SAC agents were trained for 200k episodes using a learning rate of $3 \times 1 0 ^ { - 4 }$ for the actor network and a learning rate of $1 \times 1 0 ^ { - 3 }$ for the critic network and the optimizer of the temperature parameter α. The actor and the critic network are multi-layer perceptrons with two hidden layers containing 256 neurons each. Furthermore, the soft update factor τ is set to 0.01, the replay bufer has a size of $\mathrm { { 1 0 \times 1 0 ^ { 6 } } }$ and the batch size equals 1024.

For the expert-based EBM model, the univariate and interaction efects were decided based on the justifiable efects of the full EBM model. For instance, in the full EBM model, a positive accumulated position $\Delta E _ { t }$ increased the power setpoint and vice versa. When the controller is actively curtailing, this behavior is justifiable, as it corresponds to a continuation of the existing strategy. However, the accumulated position is driven not only by the control actions but also by the power forecast error. This formulation would therefore imply that, in the absence of curtailment, the strategy is to persist in the direction of the power forecast error irrespective of the system’s needs, which is undesired. The term was therefore replaced with an interaction efect between the accumulated position and the estimated power forecast error $\hat { \varepsilon } _ { t } ,$ , ensuring that the setpoint only responds to the accumulated position when the wind farm is in fact curtailed. All efects included in the expert-based model can be found in Appendix A.3.

For the expert-based, value-oriented model, we employ the additional profit of the policy relative to the price threshold controller instead of using the observed returns directly. The reason for this being that there are certain states in which any controller will achieve a negative reward. This is, for example, the case when the day-ahead forecast overestimated the possible power production and the settlement price is positive. Here the controller cannot do much to ofset the imbalance penalty apart from producing at possible power. To stabilize training, outliers are first capped and the resulting values are normalized to the range [−1, 1]. Afterwards, we apply exponential weighting to ensure that each observation receives a strictly positive weight.

Table 1: Performance evaluation metrics across the interpretable models and a LightGBM baseline. Mean absolute error (MAE) and root mean squared error (RMSE) are normalized by the capacity of the wind farm. Lower scores indicate a better fit for MAE and RMSE, while higher scores are better for $R ^ { 2 }$
<table><tr><td></td><td>MAE</td><td>RMSE</td><td> $R ^ { 2 }$ </td></tr><tr><td>DT (depth=2)</td><td>0.21</td><td>0.29</td><td>0.38</td></tr><tr><td>DT (depth=3)</td><td>0.21</td><td>0.27</td><td>0.45</td></tr><tr><td>EBM (full)</td><td>0.18</td><td>0.23</td><td>0.61</td></tr><tr><td>EBM (expert)</td><td>0.19</td><td>0.24</td><td>0.57</td></tr><tr><td>EBM (expert + value)</td><td>0.19</td><td>0.24</td><td>0.57</td></tr><tr><td>LightGBM</td><td>0.15</td><td>0.19</td><td>0.72</td></tr></table>

Table 2: Observed imbalance costs (+) or revenues (-) for the diferent policies over the evaluation period. The results are presented in <sup>=</sup>C per MW installed per settlement period.
<table><tr><td></td><td>Mean</td><td>Min</td><td>Max</td><td>P05</td><td>P95</td></tr><tr><td>No control</td><td>3.22</td><td>-73.01</td><td>211.77</td><td>-6.55</td><td>22.13</td></tr><tr><td>Price threshold</td><td>1.47</td><td>-129.73</td><td>197.53</td><td>-8.39</td><td>16.30</td></tr><tr><td>DT (depth=2)</td><td>1.21</td><td>-105.74</td><td>193.24</td><td>-9.72</td><td>16.01</td></tr><tr><td>DT (depth=3)</td><td>0.75</td><td>-116.04</td><td>193.24</td><td>-13.17</td><td>15.74</td></tr><tr><td>EBM (full)</td><td>0.36</td><td>-99.00</td><td>193.24</td><td>-14.13</td><td>15.56</td></tr><tr><td>EBM (expert)</td><td>0.63</td><td>-73.01</td><td>193.24</td><td>-13.89</td><td>16.30</td></tr><tr><td>EBM (expert + value)</td><td>0.58</td><td>-73.01</td><td>193.24</td><td>-13.92</td><td>16.24</td></tr><tr><td>SAC</td><td>0.21</td><td>-114.27</td><td>195.85</td><td>-15.30</td><td>15.90</td></tr></table>

## 6.2 Performance of the interpretable models

In Table 1, we show how accurately each method can distill the SAC policy. We also include the performance of a LightGBM model (Ke et al. 2017) to allow for comparison against a black-box technique that can capture complex interaction efects. Overall, we note that all EBM models outperform the DT baselines, but they still attain $R ^ { 2 }$ values that are rather low. Nonetheless, if we consider the performance of the LightGBM model, we notice that even for a blackbox model it is complicated to completely distill the policy learned by the SAC actor and, therefore, the performance is found to be acceptable.

In Table 2, the performance of the diferent models is shown in terms of imbalance costs per settlement period. The imbalance costs are determined by the forecast error and the pursued control strategy. All controllers observe a positive cost over the evaluation period, meaning that they were not able to completely ofset the efect of the forecast errors. The SAC model achieves a cost reduction of 93% relative to the scenario without control, whereas the interpretable controllers attain a lower reduction. Nonetheless, they still ofer a substantial improvement over the price threshold baseline: the expert-based, value-oriented policy achieves an imbalance cost reduction of 82% compared to 54% for the price threshold controller. When comparing the DT and EBM policies, the latter were found to achieve lower imbalance costs. Even the worst-performing EBM attains a 16% improvement over the best-performing DT. Furthermore, we observe that the full EBM model outperforms the expert-based models.

![](images/6b56030128479e77eaf443b430b4718ebaeea943da5f9015f4a42a7a0719544b.jpg)  
Figure 1: Visualization of the learned shape function for the latest published imbalance price estimate $\widehat { \lambda } _ { t } ^ { \mathrm { B } }$ . The score illustrates the efect the variable has on the final prediction. A negative score implies that this variable reduces the power setpoint and vice versa.

However, the full EBM model includes five univariate and ten interaction efects, which makes predictions more difficult to interpret. Therefore, this decrease in performance is the price paid for an increase in interpretability. Finally, we note that the improvement in performance using valueoriented policy distillation is relatively small. Nevertheless, a decrease in imbalance costs of almost 8% does imply a significant cost reduction in absolute terms, especially for a model with the same level of interpretability.

## 6.3 Model explanations

In this section we will show how global and local explanations can be obtained from the EBM models. Global explanations describe how the model works in general, whereas local explanations are used to explain an individual prediction. To obtain global explanations, we can visualize the shape functions learned for each individual and interaction efect. To explain an individual prediction, we show how the output of each shape function contributes to the final control action.

In Figure 1, the global efect of the latest published imbalance price estimate $\widehat { \lambda } _ { t } ^ { \mathrm { B } }$ on the control policy is shown. We observe that the policy is in line with what we would expect, since a negative price results in a curtailment action, whereas a positive price results in a power setpoint above the trading schedule. However, interestingly, we also note that for slightly positive prices the controller still decides to curtail. This might be explained by the fact that sometimes flexibility providers ofer decremental energy to the TSO at a positive price, meaning that they pay the TSO to be activated, so the imbalance price can be positive when the system has a power surplus. They do so because they also account for the marginal cost of producing the electricity sold beforehand, for example, the fuel consumption cost for a gas turbine. Nonetheless, the capacity ofered at this positive price is limited and once it is consumed the imbalance price will become negative, which the controller could be anticipating. Surprisingly, we observe a sharp decrease of the setpoint within the range of −150 <sup>=</sup>C/MWh and −200 <sup>=</sup>C/MWh compared to the surrounding intervals, which is also true for the SAC policy. Because we expected the control policy to respond more strongly to more negative prices, we investigated this further and found that, within this interval, the final settlement price is more likely to end up negative. A possible explanation may be that other market participants are also reacting to this price using a price threshold controller with a threshold set below this range. Since these participants have not reacted yet, there is a lower chance of the combined reaction of all participants overshooting.

![](images/a1b76e9a20c4b774b8f8505474596388dcda79cb4adfc736200bd683ed73e232.jpg)  
Figure 2: Visualization of the learned shape function for the interaction efect between the system imbalance forecast $\hat { S I } _ { t } ^ { \mathrm { B P } }$ and the minute of the settlement period $t ^ { \mathrm { I S P } }$ . A negative score implies that this interaction efect reduces the power setpoint and vice versa.

In Figure 2, we show an example of one of the interaction efects of the model, namely the interaction between the system imbalance forecast and the minute of the settlement period. A first remarkable observation is that the efect for the second minute during the settlement period is clearly diferent from the efects for the other minutes. Interestingly, this minute is also the last timestep where the imbalance price estimate is still based on the previous settlement period due to the publication delay. This implies that the controller could potentially have learned to rely on the system imbalance forecast, which is forward-looking, because the imbalance price estimate is not reflective of the situation during the current settlement period. Secondly, we observe how the controller is accounting for the price impact of its control strategy. Specifically, we notice that during the first half of the settlement period, a system imbalance forecast around zero results in a small reaction in the same direction as the system imbalance. During the second half of the settlement period, this reaction becomes more extreme as an attempt to not switch the sign of the system imbalance as this would result in a penalty for previously taken control actions in the correct direction.

In Figure 3, we show an example of a local explanation of a

![](images/c1c0b11706260ab308aca35f0993c44d6950e4033d373b05c277df9be4088b07.jpg)  
Figure 3: Example of local explanation of the interpretable policy visualized as a waterfall chart. The control action in this example is to reduce the power setpoint by 35.2 MW compared to the trading schedule.

$$
( 1 ) : \hat { \lambda } _ { t } ^ { \mathsf { B } } = - 1 6 8 \notin / \mathsf { M w n }
$$

$$
( 2 ) \colon \hat { \lambda } _ { t } ^ { \mathsf { B } } = - 1 6 8 \in / \mathsf { M W h } , \hat { S I } _ { t } ^ { I S P } = 2 1 \mathsf { M W }
$$

$$
( 3 ) \colon \hat { \lambda } _ { t } ^ { \mathsf { B } } = - 1 6 8 \mathsf { \in } / \mathsf { M W h } , \Delta E _ { t } = - 5 \mathsf { M W h }
$$

$$
( 4 ) \colon \hat { \varepsilon } _ { t } = 1 4 2 \mathsf { M W } , \hat { S I } _ { t } ^ { I S P } = 2 1 \mathsf { M W }
$$

$$
( 5 ) \colon \hat { S I } _ { t } ^ { I S P } = 2 1 ~ \mathsf { M W } , t ^ { I S P } = 1 2
$$

$$
( 6 ) \colon \Delta E _ { t } = - 5 \ \mathsf { M W h } , \hat { \varepsilon } _ { t } = 1 4 2 \ \mathsf { M W }
$$

$$
( 7 ) \colon \hat { \lambda } _ { t } ^ { \mathsf { B } } = - 1 6 8 \notin / \mathsf { M W h } , t ^ { I S P } = 1 2
$$

control action that reduces the setpoint compared to the trading schedule. The action could be explained to a stakeholder as follows: (1) The main driver of this action is the fact that we observe a negative price in the range of −150 <sup>=</sup>C/MWh and −200 <sup>=</sup>C/MWh. (2) However, since the system imbalance only forecasts a moderate surplus for this negative price, the setpoint is increased again. (3) Furthermore, the controller is already expecting a reward for this settlement period and therefore it increases its setpoint to reduce the risk of overreacting. (4) Nonetheless, we observe that both the system and the wind farm have a power surplus and hence the wind farm can curtail to help the system, resulting in reducing the setpoint again. (5) Since the end of the settlement period is approaching and the system imbalance is moderate, the setpoint is slightly increased. (6) Finally, the controller continues the strategy pursued throughout the settlement period and (7) slightly lowers the setpoint in response to a negative imbalance price estimate toward the end of the settlement period.

## 7 Conclusion

In this work, we have developed a policy distillation framework to obtain an interpretable policy from one that has been obtained through reinforcement learning. Since RL-based controllers can achieve high performance, but are too blackbox to run on an actual wind farm, our method bridges the gap between theoretical research and real-world deployment. To evaluate our method, we use SCADA data from Norther, a Belgian ofshore wind farm, together with an imbalance settlement simulator that combines historical data with the controller’s actions. The distilled policy has been shown to achieve a significant reduction in imbalance costs, with a reduction of 82% compared to 93% using SAC. However, in contrast to SAC, this policy is interpretable since it is based on a transparent model and its control actions can be justified to the stakeholders. Furthermore, it outperforms other interpretable control strategies such as a price threshold controller, often used in industry, or a distilled policy using a decision tree.

## Acknowledgments

The authors gratefully acknowledge the financial support of the Flemish Government through the Flanders AI Research Program, the INSPIRE project (project no. SBEP2023-440) funded by the Sustainable Blue Economy Partnership and the Blue Cluster project called CORE funded by Flanders Innovation & Entrepreneurship (VLAIO).

## References

Ally, S.; Verstraeten, T.; Nowé, A.; and Helsen, J. 2025. Dayahead trading and power control for hybrid wind-hydrogen plants with multi-agent reinforcement learning. Applied Energy, 401: 126588.

Altmann, A.; Toloşi, L.; Sander, O.; and Lengauer, T. 2010. Permutation importance: a corrected feature importance measure. Bioinformatics, 26(10): 1340–1347.

Ansari, A. F.; Shchur, O.; Küken, J.; Auer, A.; Han, B.; Mercado, P.; Rangapuram, S. S.; Shen, H.; Stella, L.; Zhang, X.; et al. 2025. Chronos-2: From univariate to universal forecasting. arXiv preprint arXiv:2510.15821.

Bruninx, M.; Madahi, S. S. K.; Verstraeten, T.; Decuyper, J.; Develder, C.; and Helsen, J. 2026a. Herd behavior in decentralized balancing models: A case study in Belgium. In 2026 22nd International Conference on the European Energy Market (EEM), 1–9. IEEE.

Bruninx, M.; Madahi, S. S. K.; Verstraeten, T.; Decuyper, J.; Develder, C.; Nowé, A.; and Helsen, J. 2026b. Every minute counts: Wind farm power control during imbalance settlement using distributional soft actor-critic. arXiv preprint arXiv:2610.26055.

Bruninx, M.; van Binsbergen, D.; Verstraeten, T.; Nowé, A.; and Helsen, J. 2027. Probabilistic wind power forecasting

with tree-based machine learning and weather ensembles. Electric Power Systems Research, 264: 113847.

Bruninx, M.; Verstraeten, T.; Kazempour, J.; and Helsen, J. 2025. Day-ahead bidding strategies for wind farm operators under a one-price balancing scheme. In Proceedings of the 16th ACM International Conference on Future and Sustainable Energy Systems, 719–726.

Coppens, Y.; Efthymiadis, K.; Lenaerts, T.; Nowé, A.; Miller, T.; Weber, R.; and Magazzeni, D. 2019. Distilling deep reinforcement learning policies in soft decision trees. In Proceedings of the IJCAI 2019 Workshop on Explainable Artificial Intelligence, 1–6.

Elia Transmission Belgium SA. 2024. Terms and Conditions for Balance Responsible Parties (BRPs). Accessed: 2026- 08-18.

Elia Transmission Belgium SA. 2025a. Adequacy and Flexibility Study for Belgium 2026–2036.

Elia Transmission Belgium SA. 2025b. Elia Open Data Platform. Accessed: 2025-07-01.

Frosst, N.; and Hinton, G. 2017. Distilling a neural network into a soft decision tree. arXiv preprint arXiv:1711.09784.

Glanois, C.; Weng, P.; Zimmer, M.; Li, D.; Yang, T.; Hao, J.; and Liu, W. 2024. A survey on interpretable reinforcement learning. Machine Learning, 113(8): 5847–5890.

Gokhale, G.; Karimi Madahi, S. S.; Claessens, B.; and Develder, C. 2024. Distill2Explain: Diferentiable decision trees for explainable reinforcement learning in energy application controllers. In Proceedings of the 15th ACM International Conference on Future and Sustainable Energy Systems, 55– 64.

Haarnoja, T.; Zhou, A.; Abbeel, P.; and Levine, S. 2018a. Soft actor-critic: Of-policy maximum entropy deep reinforcement learning with a stochastic actor. In International Conference on Machine Learning, 1861–1870. PMLR.

Haarnoja, T.; Zhou, A.; Hartikainen, K.; Tucker, G.; Ha, S.; Tan, J.; Kumar, V.; Zhu, H.; Gupta, A.; Abbeel, P.; et al. 2018b. Soft actor-critic algorithms and applications. arXiv preprint arXiv:1812.05905.

Håberg, M.; and Doorman, G. 2016. Classification of balancing markets based on diferent activation philosophies: Proactive and reactive designs. In Proceedings of the 13th International Conference on the European Energy Market (EEM), 1–5. IEEE.

Hastie, T.; and Tibshirani, R. 1986. Generalized additive models. Statistical Science, 1(3): 297–310.

Hein, D.; Udluft, S.; and Runkler, T. A. 2018. Interpretable policies for reinforcement learning by genetic programming. Engineering Applications of Artificial Intelligence, 76: 158– 169.

Heiser, Y.; Kazempour, J.; and Pourahmadi, F. 2026. When and how should a power trader engage in arbitrage? Predict, then contextually optimize. arXiv preprint arXiv:2607.07351.

Heiser, Y.; Pourahmadi, F.; and Kazempour, J. 2025. Betting vs. trading: Learning a linear decision policy for selling

wind power and hydrogen. Sustainable Energy, Grids and Networks, 101848.

IEA. 2025. Global Energy Review 2025. Licence: CC BY 4.0.

Ke, G.; Meng, Q.; Finley, T.; Wang, T.; Chen, W.; Ma, W.; Ye, Q.; and Liu, T.-Y. 2017. LightGBM: A highly eficient gradient boosting decision tree. Advances in Neural Information Processing Systems, 30.

Likmeta, A.; Metelli, A. M.; Tirinzoni, A.; Giol, R.; Restelli, M.; and Romano, D. 2020. Combining reinforcement learning with rule-based controllers for transparent and general decision-making in autonomous driving. Robotics and Autonomous Systems, 131: 103568.

Liu, G.; Schulte, O.; Zhu, W.; and Li, Q. 2018. Toward interpretable deep reinforcement learning with linear model U-trees. In Joint European Conference on Machine Learning and Knowledge Discovery in Databases, 414–429. Springer.

Lou, Y.; Caruana, R.; Gehrke, J.; and Hooker, G. 2013. Accurate intelligible models with pairwise interactions. In Proceedings ofthe 19thACMSIGKDD International Conference on Knowledge Discovery and Data Mining, 623–631.

Lundberg, S. M.; and Lee, S.-I. 2017. A unified approach to interpreting model predictions. Advances in Neural Information Processing Systems, 30.

Madahi, S. S. K.; Bruninx, K.; Claessens, B.; and Develder, C. 2025. Model predictive control-guided reinforcement learning for implicit balancing. arXiv preprint arXiv:2510.04868.

Madahi, S. S. K.; Claessens, B.; and Develder, C. 2024. Distributional reinforcement learning-based energy arbitrage strategies in imbalance settlement mechanism. Journal of Energy Storage, 104: 114377.

Nori, H.; Jenkins, S.; Koch, P.; and Caruana, R. 2019. InterpretML: A unified framework for machine learning interpretability. arXiv preprint arXiv:1909.09223.

Rasic, C.; Favaro, P.; Wang, Y.; and Toubeau, J.-F. 2025. Safe reinforcement learning for battery energy storage participation in the imbalance settlement. IEEE Transactions on Energy Markets, Policy and Regulation.

Rijkdienst voor Ondernemend Nederland. 2024. Noordzeker and Zeevonk II Receive Permits for the Largest Dutch Wind Farms in the North Sea. Accessed: 2024-10-31.

Rusu, A. A.; Colmenarejo, S. G.; Gulcehre, C.; Desjardins, G.; Kirkpatrick, J.; Pascanu, R.; Mnih, V.; Kavukcuoglu, K.; and Hadsell, R. 2015. Policy distillation. arXiv preprint arXiv:1511.06295.

Silva, A.; and Gombolay, M. 2021. Encoding human domain knowledge to warm start reinforcement learning. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 35, 5042–5050.

Smets, R.; Toubeau, J.-F.; Dolanyi, M.; Bruninx, K.; and Delarue, E. 2025. Value-oriented price forecasting for arbitrage strategies of energy storage systems through loss function tuning. Energy, 333: 137112.

TotalEnergies SE. 2024. Germany: TotalEnergies Wins Further Maritime Lease in the North Sea to Develop 1.5 GW of Ofshore Wind. Accessed: 2024-10-31.

Wang, J.; Zhou, Y.; Zhang, Y.; Lin, F.; and Wang, J. 2023. Risk-averse optimal combining forecasts for renewable energy trading under CVaR assessment of forecast errors. IEEE Transactions on Power Systems, 39(1): 2296–2309.

Zhang, J.; Wang, Y.; and Hug, G. 2022. Cost-oriented load forecasting. Electric Power Systems Research, 205: 107723.

## A Appendix

## A.1 Price threshold controller

The price threshold controller decides the power setpoint $x _ { t }$ based on the latest published imbalance price estimate $\widehat { \lambda } _ { t } ^ { \mathrm { B } }$

$$
x _ { t } = \left\{ \begin{array} { l l } { \overline { { P } } _ { t } } & { \mathrm { ~ i f ~ } \hat { \lambda } _ { t } ^ { \mathrm { B } } > \overline { { \lambda } } ^ { \mathrm { B } } } \\ { \underline { { P } } } & { \mathrm { ~ i f ~ } \hat { \lambda } _ { t } ^ { \mathrm { B } } < \underline { { \lambda } } ^ { \mathrm { B } } } \\ { v _ { t } } & { \mathrm { ~ e l s e , } } \end{array} \right.\tag{10}
$$

where $\underline { { P } }$ denotes the minimum power setpoint and the price thresholds $\overline { { \lambda } } ^ { \mathrm { B } }$ and $\underline { { \lambda } } ^ { \mathrm { B } }$ are calibrated using historical price information following the procedure of (Bruninx et al. 2026a). In this work, they are set to $\underline { { \lambda } } ^ { \mathrm { B } } \ = \ - 1 8 6 \ \mathrm { { \epsilon / M W h } }$ and $\overline { { \lambda } } ^ { \mathrm { B } } = 2 3 0 \ : \mathrm { { \epsilon m w h } }$

## A.2 Soft decision trees

We have implemented soft decision trees (SDT; Frosst and Hinton 2017) as an additional method for policy distillation following (Coppens et al. 2019; Gokhale et al. 2024). However, since we are performing regression between the policy of the student and the teacher, we had to use a mean squared error loss function rather than a cross entropy loss or a KL divergence loss.

After training the model, predictions can be obtained in two ways:

1. Take the prediction of the leaf with the maximum path probability (Method = Max)

2. Averaging all leaf predictions by their corresponding path probabilities (Method = Mean)

In the original SDT paper by Frosst and Hinton (2017), the authors argue that the former makes the model explainable, whereas the latter only achieves “marginally better test accuracy” at “an exponential increase in complexity of the explanation”. We agree with the authors that the second approach does not correspond to the definition of a transparent model employed in this paper. However, we find that, in our regression setting, the explainable predictions result in a significant drop in accuracy, as shown in Table 3, attaining even worse performance than a simple decision tree.

## A.3 Expert-based EBM

The expert-based model only includes one single univariate efect of the imbalance price estimate $\widehat { \lambda } _ { t } ^ { \mathrm { B } }$ such that it closely resembles the price threshold controller. Furthermore, it includes six interaction efects:

Table 3: Performance evaluation metrics for the SDT models. The inference method is specified in the ‘Method’ column (cf., below). Mean absolute error (MAE) and root mean squared error (RMSE) are normalized by the capacity of the wind farm. Lower scores indicate a better fit for MAE and RMSE, while higher scores are better for $R ^ { 2 }$
<table><tr><td></td><td>Method</td><td>MAE</td><td>RMSE</td><td> $R ^ { 2 }$ </td></tr><tr><td>SDT (depth=2)</td><td>Mean</td><td>0.20</td><td>0.26</td><td>0.50</td></tr><tr><td>SDT (depth=3)</td><td>Mean</td><td>0.18</td><td>0.24</td><td>0.56</td></tr><tr><td>SDT (depth=2)</td><td>Max</td><td>0.26</td><td>0.33</td><td>0.18</td></tr><tr><td>SDT (depth=3)</td><td>Max</td><td>0.24</td><td>0.31</td><td>0.27</td></tr></table>

1. The interaction between the imbalance price estimate $\widehat { \lambda } _ { t } ^ { \mathrm { B } }$ and the system imbalance forecast $\hat { S I } _ { t } ^ { \mathrm { I S P } }$

2. The interaction between the imbalance price estimate $\widehat { \lambda } _ { t } ^ { \mathrm { B } }$ and the accumulated position during the settlement period $\Delta E _ { t }$

3. The interaction between the estimated power forecast error $\hat { \varepsilon } _ { t }$ and the system imbalance forecast $\hat { S I } _ { t } ^ { \mathrm { B P } }$

4. The interaction between the system imbalance forecast $\hat { S I } _ { t } ^ { \mathrm { B P } }$ and the minute of the settlement period $t ^ { \mathrm { I S P } }$

5. The interaction between the imbalance price estimate $\widehat { \lambda } _ { t } ^ { \mathrm { B } }$ and the minute of the settlement period $t ^ { \mathrm { I S P } }$

6. The interaction between the accumulated position during the settlement period $\Delta E _ { t }$ and the estimated power forecast error $\hat { \varepsilon } _ { t }$