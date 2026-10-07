# LSC-DPO: LEARNING-SIGNAL-CONTROLLED DIRECT PREFERENCE OPTIMIZATION

Yang Qu<sup>1</sup> Yusheng Han<sup>1</sup> Chengjia Feng<sup>1</sup> Handan Liu<sup>∗</sup>

<sup>1</sup>Northeastern University

{qu.yang1, han.yus, feng.chen, h.liu}@northeastern.edu <sup>∗</sup>Corresponding author

## ABSTRACT

Direct Preference Optimization (DPO) has become a standard reward-model-free approach for aligning language models with preference data. However, as the scaled preference margin grows during training, the logistic DPO loss becomes progressively less sensitive to further changes. We study DPO from a loss-level geometric perspective and identify the sigmoid factor as a learning signal that characterizes the local sensitivity of the objective. Based on this view, we propose Learning-Signal-Controlled Direct Preference Optimization (LSC-DPO), which dynamically regulates the learning signal near a target regime. A log-space analysis establishes conditions for stable tracking of the target learning-signal regime. Experiments on AlpacaEval 2, MT-Bench, and Anthropic-HH show that LSC-DPO consistently improves over DPO and strong preference-optimization baselines. We further find that different coefficient initializations induce distinct transient learning-signal trajectories even when their later signal levels become similar. Based on this observation, we derive a signal-budget compensation rule that adjusts the target learning signal to compensate for these transient differences. The resulting compensation substantially reduces performance variation across coefficient initializations.<sup>1</sup>

## 1 INTRODUCTION

Large language models (LLMs) have achieved remarkable performance across a broad range of tasks. However, their outputs are not always aligned with human preferences. Reinforcement learning from human feedback (RLHF) (Christiano et al., 2017; Ouyang et al., 2022; Stiennon et al., 2020) has become a widely used paradigm for improving alignment. In the standard RLHF pipeline, approaches such as Proximal Policy Optimization (PPO) (Schulman et al., 2017) optimize a policy based on a learned reward model that captures human preferences, while a KL penalty constrains the policy from deviating excessively from a reference model.

Recent work has explored reward-model-free approaches to reduce the computational cost of RLHF. Direct Preference Optimization (DPO) (Rafailov et al., 2023) directly optimizes a policy from offline preference data without explicitly training a reward model. In DPO, preference learning is formulated through the policy-reference log-ratio difference between chosen and rejected responses. Under the Bradley–Terry model (Bradley & Terry, 1952), the scaled log-ratio difference can be interpreted as an implicit reward difference. The coefficient β scales this quantity and thereby affects the resulting optimization dynamics.

A central question in DPO is how to characterize and regulate its evolving optimization state during training. The DPO objective contains a logistic loss term whose sensitivity decreases as optimization progresses (Rafailov et al., 2023; 2024; Gorbatovski et al., 2024). As this sensitivity decreases, the same preference pairs receive progressively smaller gradient weight, weakening their contribution to subsequent policy updates. Importantly, the initial choice of the coefficient β also affects the sensitivity of the logistic loss term and can therefore lead to different optimization trajectories and downstream performance (Rafailov et al., 2023). These observations motivate a principled mechanism for characterizing and regulating variations in the training state.

In this paper, we analyze the DPO objective from a loss-level geometric perspective and identify a sigmoid factor that characterizes the local sensitivity of the objective during training. This factor reflects both the evolving optimization state and the effect of the coefficient $\beta ,$ making it a useful quantity for diagnosing and regulating DPO training. Based on this observation, we propose Learning-Signal-Controlled Direct Preference Optimization (LSC-DPO), which dynamically regulates the learning signal to keep it close to a target level during preference optimization. We further derive a local stability condition for tracking the desired learning-signal regime. Empirically, LSC-DPO achieves strong performance compared with existing preference-optimization methods (Rafailov et al., 2023; Wu et al., 2024b; Lee et al., 2025; Park et al., 2024; Azar et al., 2024; Ethayarajh et al., 2024; Xu et al., 2024; Hong et al., 2024; Zhao et al., 2023; Meng et al., 2024) across standard alignment benchmarks and preference-optimization experiments. We further use the learning-signal view to analyze sensitivity to the initial choice of $\beta$ and derive a signal-budget compensation rule that reduces initialization-induced performance degradation and improves robustnes across different initial coefficients.

Our contributions are summarized in three aspects. (1) We provide a loss-level geometric characterization of DPO, showing that the sigmoid factor serves as a learning signal for tracking the local sensitivity of the objective and the evolving training state. (2) We propose LSC-DPO, a learningsignal-controlled preference optimization method that dynamically adjusts the DPO coefficient according to the evolving training state. We further derive a local stability condition that characterizes stable tracking around the target learning-signal regime. Empirically, LSC-DPO achieves strong performance compared with existing preference-optimization baselines across standard alignment benchmarks. (3) We investigate the relationship between the initial coefficient and learning-signal dynamics, and derive a signal-budget compensation rule that reduces initialization-induced performance degradation and improves robustness across different initial coefficients.

## 2 PRELIMINARIES

In RLHF, a policy is optimized against a reward function while being regularized toward a reference policy (Christiano et al., 2017; Jaques et al., 2017; 2020):

$$
\operatorname* { m a x } _ { \pi _ { \theta } } \mathbb { E } _ { x \sim \mathcal { D } , y \sim \pi _ { \theta } ( \cdot | x ) } \left[ r ( x , y ) \right] - \beta \mathbb { E } _ { x \sim \mathcal { D } } \left[ \mathbb { D } _ { \mathrm { K L } } \left( \pi _ { \theta } ( \cdot | x ) \| \pi _ { \mathrm { r e f } } ( \cdot | x ) \right) \right] ,\tag{1}
$$

where $\pi _ { \theta }$ denotes the policy and $\pi _ { \mathrm { r e f } }$ denotes the reference model. DPO (Rafailov et al., 2023) uses the closed-form solution (Ziegler et al., 2019; Peng et al., 2019; Peters & Schaal, 2007; Go et al., 2023; Rafailov et al., 2023) to reparameterize the reward function as:

$$
r ( x , y ) = \beta \log \frac { \pi _ { \theta } ( y \mid x ) } { \pi _ { \mathrm { r e f } } ( y \mid x ) } + \beta \log Z ( x ) ,\tag{2}
$$

where $Z ( x )$ is the partition function. Substituting the reparameterized reward into the Bradley–Terry model (Bradley $\&$ Terry, 1952) yields the DPO objective:

$$
\mathcal { L } _ { \mathrm { D P O } } ( \pi _ { \theta } ; \pi _ { \mathrm { r e f } } ) = - \mathbb { E } _ { ( x , y _ { + } , y _ { - } ) } \left[ \log \sigma \left( \beta \log \frac { \pi _ { \theta } ( y _ { + } \mid x ) } { \pi _ { \mathrm { r e f } } ( y _ { + } \mid x ) } - \beta \log \frac { \pi _ { \theta } ( y _ { - } \mid x ) } { \pi _ { \mathrm { r e f } } ( y _ { - } \mid x ) } \right) \right] ,\tag{3}
$$

where $y _ { + }$ and $y _ { - }$ denote the chosen and rejected responses, respectively. The DPO objective depends on the difference between the policy log-ratio of the chosen and rejected responses and the corresponding reference model log-ratio. We define this difference as the preference margin:

$$
m = \log { \frac { \pi _ { \theta } ( y _ { + } \mid x ) } { \pi _ { \theta } ( y _ { - } \mid x ) } } - \log { \frac { \pi _ { \mathrm { r e f } } ( y _ { + } \mid x ) } { \pi _ { \mathrm { r e f } } ( y _ { - } \mid x ) } } .\tag{4}
$$

This margin measures how much more the policy favors the chosen response over the rejected response, relative to the reference model. We further define the scaled preference margin as

$$
z = \beta m .\tag{5}
$$

Using the margin formulation above, the DPO objective for a single preference pair can be written compactly as

$$
\ell _ { \mathrm { D P O } } = - \log \sigma ( \beta m ) = - \log \sigma ( z ) .\tag{6}
$$

Thus, the optimization behavior of DPO depends on the scaled margin $z = \beta m$ , which couples the evolving preference margin with the coefficient $\beta$

## 3 LEARNING-SIGNAL-CONTROLLED DIRECT PREFERENCE OPTIMIZATION

In this section, we introduce Learning-Signal-Controlled Direct Preference Optimization (LSC-DPO). We first identify the sigmoid factor $\sigma ( - z )$ in the DPO objective as a learning signal that reflects the local sensitivity of the DPO loss during training and illustrate how different initial choices of β affect the evolution of this signal. We then use this signal to characterize the training state of DPO and derive an update rule that regulates the signal during preference optimization. We analyze the stability of the resulting dynamics in log-space.

## 3.1 LEARNING SIGNAL IN DPO

To understand the training behavior of DPO, we analyze the local structure of the DPO objective in Eq. 6. The gradient can be written as

$$
\nabla _ { \theta } \ell = - \sigma ( - z _ { \theta } ) \nabla _ { \theta } z _ { \theta } .\tag{7}
$$

Taking another derivative gives the Hessian of the DPO objective:

$$
\begin{array} { r } { \nabla _ { \theta } ^ { 2 } \ell = \sigma ( z _ { \theta } ) \sigma ( - z _ { \theta } ) \nabla _ { \theta } z _ { \theta } \nabla _ { \theta } z _ { \theta } ^ { \top } - \sigma ( - z _ { \theta } ) \nabla _ { \theta } ^ { 2 } z _ { \theta } . } \end{array}\tag{8}
$$

Equations 7 and 8 show that $\sigma ( - z _ { \theta } )$ plays a central role in the training dynamics of DPO. From the gradient perspective, $\sigma ( - z _ { \theta } )$ directly measures the first-order sensitivity of the DPO loss. From the curvature perspective, the Hessian contains two scaled terms: the outer-product term is weighted by $\sigma ( z _ { \theta } ) \sigma ( - z _ { \theta } )$ , while the secondorder term is weighted by $\sigma ( - z _ { \theta } )$ . During training, z<sub>θ</sub> typically increases for preference pairs that become better aligned with the preference ordering. As $z _ { \theta }$ grows, the sigmoid factor $\sigma ( - z _ { \theta } )$ decreases. Since these factors scale the Hessian terms that determine local curvature, their decrease indicates reduced local sensitivity of the objective and a weaker preference-learning tendency. Figure 1 illustrates this loss geometry.

![](images/b160e03e20900062f20f6718fefa5081a34263fa55e9d69f94243ed1ac490ce9.jpg)  
Figure 1: Illustration of the DPO loss geometry as a function of the scaled margin $z = \beta m$ The learning signal $\sigma ( - z )$ and curvature factor $c ( z ) = \sigma ( { z } ) \sigma ( - { z } )$ decrease as z grows.

We therefore use $\sigma ( - z _ { \theta } )$ as a learning signal for

DPO training. When this signal is relatively large, the DPO loss remains more sensitive to changes in the scaled margin, corresponding to a high-signal regime. As $z _ { \theta }$ increases, $\sigma ( - z _ { \theta } )$ decreases and the loss becomes progressively less sensitive to further changes in the scaled margin. We refer to this as a low-signal regime, which progressively approaches a lower-curvature region as z<sub>θ</sub> grows further. Figures 2a and 2b further show that $\sigma ( - z _ { \theta } )$ evolves systematically over training. It remains relatively high during the early stage and gradually decreases before stabilizing at a lower level. This evolution is consistent with our interpretation of $\sigma ( - z _ { \theta } )$ as reflecting the local responsiveness of the DPO loss during training. The figures further show that different choices of the DPO coefficient lead to markedly different learning-signal trajectories $\sigma ( - z )$ This observation suggests that the learning-signal dynamics are related to the coefficient setting and motivates our later analysis of coefficient initialization in Section 4.4.

## 3.2 LEARNING-SIGNAL-CONTROLLED UPDATE

Since $\sigma ( - z )$ reflects the learning signal in DPO, we aim to keep it near a target level rather than allowing it to decay rapidly during training. Because $\sigma ( - z )$ decreases monotonically with the scaled margin z, regulating z provides a direct way to control the signal. Since $z = \beta m$ , we use $\beta$ as the control variable.

At training step $k ,$ we compute the per-example preference margins on the current batch and use their mean as the batch-level margin $m _ { k }$ . We then obtain the corresponding scaled margin:

$$
z _ { k } = \beta _ { k } m _ { k } .\tag{9}
$$

![](images/49901fa2a815bf074adc8039a6a6a2d47f92a2e5e5023689d5eca2c8b88a64ee.jpg)  
(a) HH-harmless-base

![](images/f73df345d74a0ad0ffff9ff1288921fbc66c8f8585c4cd6c089210e86305d06d.jpg)  
(b) HH-helpful-base  
Figure 2: $\sigma ( - z )$ under fixed $\beta$ values during DPO training with Llama3-8B on two Anthropic-HH subsets: (a) harmless-base and (b) helpful-base. EMA-smoothed trajectories are shown for readability; raw trajectories are provided in Appendix A.

Thus, the controller computes the signal from the scaled batch-mean margin $z _ { k }$ . We then measure its deviation from a target scaled margin $z ^ { * }$

$$
u _ { k } = z _ { k } - z ^ { * } .\tag{10}
$$

The target $z ^ { * }$ determines the learning-signal regime that LSC-DPO tracks, equivalently corresponding to the target signal level $\sigma ( - z ^ { * } )$ ). LSC-DPO updates $\beta$ using the following multiplicative rule:

$$
\beta _ { k + 1 } = \beta _ { k } \exp ( - \eta u _ { k } ) = \beta _ { k } \exp \left[ - \eta ( \beta _ { k } m _ { k } - z ^ { * } ) \right] ,\tag{11}
$$

where $\eta > 0$ controls the adjustment rate. The multiplicative form preserves the positivity of $\beta$ and implements feedback control over the scaled margin. In the positive-margin regime, when $z _ { k } > z ^ { * }$ $\sigma ( - z _ { k } )$ is below the target level and the update decreases $\beta _ { k }$ , moving the scaled margin downward. Conversely, when $z _ { k } < z ^ { * }$ , the update increases $\beta _ { k }$ , moving the scaled margin upward toward the target. The complete procedure is provided in $\mathbf { A } ]$ lgorithm 1 in Appendix B.

## 3.3 STABILITY ANALYSIS OF THE UPDATE PROCESS

We analyze the LSC-DPO update in log-space to characterize controller-level tracking of the target $z ^ { * }$ , rather than convergence of the full policy optimization problem. Starting from Eq. 11, let

$$
x _ { k } : = \log \beta _ { k } .\tag{12}
$$

Substituting $\beta _ { k } = e ^ { x _ { k } }$ into Eq. 11 gives

$$
x _ { k + 1 } = x _ { k } - \eta \left( e ^ { x _ { k } } m _ { k } - z ^ { * } \right) .\tag{13}
$$

Since the margin statistic $m _ { k }$ evolves during training, the induced system is non-autonomous. Here, $m _ { k }$ denotes the realized batch-level margin at step k and may depend on the current policy, minibatch sampling, and previous coefficient updates. Accordingly, our analysis is conditional on the realized margin sequence and characterizes controller-level tracking rather than joint convergence of the full policy-training dynamics. Conditioned on $m _ { k }$ , the step-dependent update map is

$$
F _ { k } ( x ) = x - \eta \left( e ^ { x } m _ { k } - z ^ { * } \right) .\tag{14}
$$

Instantaneous equilibrium. For a realized $m _ { k }$ , the map $F _ { k }$ admits an instantaneous fixed point

$$
x _ { k } ^ { f } = \log \frac { z ^ { * } } { m _ { k } } , \qquad \beta _ { k } ^ { f } = \frac { z ^ { * } } { m _ { k } } ,\tag{15}
$$

provided that $m _ { k } \ > \ 0$ and $z ^ { * } > 0$ . Because $m _ { k }$ evolves during training, this equilibrium moves over time. We next characterize its local stability.

Local stability. Linearizing the step-dependent map $F _ { k }$ around the instantaneous equilibrium $x _ { k } ^ { f }$ gives

$$
F _ { k } ^ { \prime } ( x _ { k } ^ { f } ) = 1 - \eta z ^ { \ast } .\tag{16}
$$

Hence, the instantaneous equilibrium is locally asymptotically stable whenever

$$
0 < \eta z ^ { * } < 2 .\tag{17}
$$

Notably, this condition is independent of the realized margin m<sub>k</sub>. The formal proposition and proof are provided in Appendix C.1.

Bounded tracking. Because $m _ { k }$ evolves during training, the corresponding equilibrium $x _ { k } ^ { f }$ also moves over time. Under local contraction and bounded equilibrium drift, the resulting tracking error satisfies

$$
\big | x _ { k } - x _ { k } ^ { f } \big | \leq \rho ^ { k - T } \big | x _ { T } - x _ { T } ^ { f } \big | + \frac { \bar { \delta } } { 1 - \rho } , \qquad k \geq T ,\tag{18}
$$

where $\rho \in \left( 0 , 1 \right)$ denotes a local contraction factor and $\bar { \delta }$ bounds the step-to-step motion of the instantaneous equilibrium. The first term decays geometrically, while the second captures the persistent error induced by the moving target. Thus, when the realized margin evolves sufficiently gradually, the controller remains close to the target scaled-margin regime and keeps the learning signal near $\sigma ( - z ^ { * } )$ ). The complete theorem, assumptions, and proof are provided in Appendix C.2.

## 4 EXPERIMENTS

In this section, we evaluate LSC-DPO on standard alignment benchmarks and analyze its training dynamics. We first compare LSC-DPO with representative preference-optimization baselines on AlpacaEval 2 (Dubois et al., 2024) and MT-Bench (Zheng et al., 2023b), and then study its performance and learning-signal dynamics on Anthropic-HH (Bai et al., 2022). We further validate the proposed learning-signal control through optimization-dynamics analyses and comparisons with predetermined $\beta$ schedules. Finally, we investigate robustness to the initial DPO coefficient and the proposed signal-budget compensation rule. Additional analyses of target-signal selection and hyperparameters are provided in Appendix J, control stability in Appendix ${ \bf G } ,$ and downstream capability evaluations in Appendix K.

## 4.1 EXPERIMENTAL SETUP

Models and training datasets. We evaluate on two open-source models, meta-llama/Meta-Llama-3-8B (Llama3-8B) (Grattafiori et al., 2024) and Qwen/Qwen3-8B-Base (Qwen3-8B) (Yang et al., 2025). We first perform supervised fine-tuning on UltraChat-200k (Ding et al., 2023), and then perform preference optimization on UltraFeedback (Cui et al., 2023). We additionally use the harmlessbase and helpful-base subsets of Anthropic-HH (Bai et al., 2022) for controlled analyses. Full train ing settings are provided in Appendix E.1.

Evaluation benchmarks and baselines. For the UltraFeedback experiments, we evaluate on AlpacaEval 2 (Dubois et al., 2024) using win rate and length-controlled (LC) win rate, and on MT-Bench (Zheng et al., 2023b). On Anthropic-HH, we report pairwise win rates on both harmless-base and helpful-base. We compare our method with representative preference-optimization baselines, including DPO (Rafailov et al., 2023), β-DPO (Wu et al., 2024b), ε-DPO (Lee et al., 2025), R-DPO (Park et al., 2024), IPO (Azar et al., 2024), KTO (Ethayarajh et al., 2024), CPO (Xu et al., 2024), ORPO (Hong et al., 2024), SLiC-HF (Zhao et al., 2023), and SimPO (Meng et al., 2024). For the controlled Anthropic-HH analyses, we focus on DPO, β-DPO, and ε-DPO as the most relevant coefficient-control baselines. Detailed evaluation protocols and baseline configurations are provided in Appendix E.2.

## 4.2 MAIN RESULTS ON STANDARD ALIGNMENT BENCHMARKS

LSC-DPO achieves the strongest overall performance. Table 1 reports that LSC-DPO achieves the best AlpacaEval 2 performance on both Llama3-8B and Qwen3-8B, while attaining top or tiedtop MT-Bench scores. Compared with DPO, LSC-DPO improves the length-controlled win rate from 19.2 to 22.7 on Llama3-8B and from 17.2 to 22.1 on Qwen3-8B, with similar gains in standard win rate. It also consistently outperforms β-DPO and ε-DPO and remains competitive with broader preference-optimization baselines, including R-DPO, KTO, CPO, ORPO, SLiC-HF, IPO, and SimPO. We next use Anthropic-HH for controlled comparisons of coefficient-regulation meth ods and their learning-signal behavior.

Table 1: Evaluation results on AlpacaEval 2 and MT-Bench. For AlpacaEval 2, we report both win rate (WR) and length-controlled win rate (LC). For MT-Bench, we report the average score on a 1–10 scale.
<table><tr><td rowspan="2">Method</td><td colspan="3">Llama3-8B</td><td colspan="3">Qwen3-8B</td></tr><tr><td>AlpacaEval 2</td><td></td><td>MT-Bench</td><td>AlpacaEval 2</td><td></td><td>MT-Bench</td></tr><tr><td></td><td>LC (%)</td><td>WR (%)</td><td>Score</td><td>LC (%)</td><td>WR (%)</td><td>Score</td></tr><tr><td>SFT</td><td>6.8</td><td>3.7</td><td>6.3</td><td>8.5</td><td>4.8</td><td>6.9</td></tr><tr><td>DPO</td><td>19.2</td><td>17.7</td><td>6.6</td><td>17.2</td><td>13.1</td><td>7.6</td></tr><tr><td>ε-DPO</td><td>19.6</td><td>17.6</td><td>6.9</td><td>19.4</td><td>14.0</td><td>7.6</td></tr><tr><td>β-DPO</td><td>16.6</td><td>16.5</td><td>6.8</td><td>16.4</td><td>10.3</td><td>7.2</td></tr><tr><td>CPO</td><td>11.2</td><td>10.3</td><td>6.4</td><td>16.5</td><td>14.5</td><td>7.6</td></tr><tr><td>IPO</td><td>8.6</td><td>8.5</td><td>6.6</td><td>14.9</td><td>9.2</td><td>7.4</td></tr><tr><td>KTO</td><td>16.9</td><td>15.1</td><td>6.8</td><td>17.2</td><td>11.3</td><td>7.4</td></tr><tr><td>ORPO</td><td>6.4</td><td>5.3</td><td>5.5</td><td>9.4</td><td>6.8</td><td>6.9</td></tr><tr><td>R-DPO</td><td>19.5</td><td>18.2</td><td>6.8</td><td>19.3</td><td>13.2</td><td>7.6</td></tr><tr><td>SLiC-HF</td><td>10.1</td><td>8.7</td><td>5.9</td><td>12.5</td><td>10.5</td><td>7.1</td></tr><tr><td>SimPO</td><td>19.7</td><td>19.2</td><td>6.9</td><td>16.8</td><td>10.9</td><td>7.4</td></tr><tr><td>LSC-DPO</td><td>22.7</td><td>21.1</td><td>6.9</td><td>22.1</td><td>16.1</td><td>7.8</td></tr></table>

![](images/9c72cda26234f2bbd850dc850b6c5d9afd35ee899da32b8879f6ebbf998df761.jpg)  
(a) HH-harmless-base

![](images/7605b03dff12ee0c8c98113c46ded476df88f6a7aa268a1e0612ff3d8a0cafd4.jpg)  
(b) HH-helpful-base  
Figure 3: Performance comparison on Anthropic-HH. We report win rates of DPO, β-DPO, ε-DPO, and LSC-DPO using Llama3-8B and Qwen3-8B. Error bars denote approximate 95% confidence intervals computed from prompt-level GPT-4 judgment results.

![](images/1f65455efde41e644fc664ac06bb55c69dada3ade3a581b9c674f870fd6bdb3d.jpg)  
(a) harmless-base

![](images/b19a2213c9e7cdcd8a8f3df0d7a60e6a5a535d8dbf987edac29c13371b40d39a.jpg)  
(b) helpful-base  
Figure 4: Learning signal $\sigma ( - z )$ on Anthropic-HH using Llama3-8B. The corresponding raw curves and Qwen results are provided in Appendix F.

Table 2: Optimization diagnostics before and after LSC-DPO control activation. We report the mean curvature factor and mean pre-clipping gradient norm.
<table><tr><td></td><td colspan="2">Before control (steps 1–83)</td><td colspan="2">After control (steps 84–477)</td></tr><tr><td>Method</td><td>Curvature</td><td>Grad. norm</td><td>Curvature</td><td>Grad. norm</td></tr><tr><td>DPO</td><td>0.24993</td><td>29.861</td><td>0.23724</td><td>90.330</td></tr><tr><td>LSC-DPO</td><td>0.24993</td><td>29.896</td><td>0.24371</td><td>73.373</td></tr></table>

Table 3: Comparison with predetermined β schedules.
<table><tr><td>Method</td><td>LC (%)</td><td>WR (%)</td></tr><tr><td>DPO</td><td>19.2</td><td>17.7</td></tr><tr><td>Linear β schedule</td><td>19.7</td><td>18.4</td></tr><tr><td>Exponential β schedule</td><td>20.3</td><td>18.5</td></tr><tr><td>Cosine β schedule</td><td>19.1</td><td>18.1</td></tr><tr><td>LSC-DPO</td><td>22.7</td><td>21.1</td></tr></table>

Overall performance on Anthropic-HH. Figures 3a and 3b show that LSC-DPO achieves the best pairwise win rates across both model families and Anthropic-HH subsets, reaching approximately 79% on harmless-base and 82–87% on helpful-base. It consistently outperforms DPO, β- DPO, and ε-DPO, further supporting the effectiveness of learning-signal control under controlled preference-optimization settings.

Learning-signal dynamics. Figures 4a and 4b show that DPO exhibits a sustained signal decay and eventually moves toward a low-signal regime, β-DPO shows larger fluctuations, and ε-DPO partially recovers after an initial decline. In contrast, LSC-DPO maintains a higher and more stable signal near its target regime across most of training. These relative dynamics are consistent with the performance trends in Figure 3; on helpful-base, for instance, ε-DPO maintains a lower signal than β-DPO for much of training before recovering later, aligning with its lower win rate. Overall, these results support the intended learning-signal regulation behavior of LSC-DPO.

## 4.3 MECHANISTIC VALIDATION OF LEARNING-SIGNAL CONTROL

Learning signal and realized optimization dynamics. Table 2 shows that DPO and LSC-DPO are nearly indistinguishable before control activation: both have the same mean curvature factor and almost identical pre-clipping gradient norms. After control activation, LSC-DPO maintains a larger curvature factor than DPO. The slower reduction of this quantity indicates that LSC-DPO keeps the objective farther from the flatter, low-curvature regime. At the same time, its pre-clipping gradient norm is lower, showing that this increased loss-level responsiveness is not obtained by simply amplifying the overall parameter-gradient magnitude. Together, these results support our interpretation of σ(−z) as a meaningful loss-level training signal. Full training trajectories are provided in Appendix H.1.

Feedback versus predetermined β schedules. Table 3 compares LSC-DPO with several prede termined schedules. Linear, exponential, and cosine schedules yield only modest changes relative to DPO, whereas LSC-DPO achieves substantially higher LC and WR. This suggests that the improvement cannot be explained solely by using a time-varying $\beta$ schedule and supports the role of state-dependent learning-signal feedback in LSC-DPO.

## 4.4 INITIALIZATION ROBUSTNESS AND SIGNAL COMPENSATION

Initialization sensitivity. Table 4 shows substantial sensitivity to $\beta _ { 0 }$ across all methods. LSC-DPO achieves the highest performance at every tested initialization and degrades less severely, but still exhibits a 7.5-point LC spread. Figure 5 further shows how different $\beta _ { 0 }$ values affect the

![](images/feacbfbd07dee7cb49310302dce1d770d4a125104b6cdf96e75d6aeea2339eac.jpg)  
Figure 5: Learning-signal trajectories under different initial $\beta _ { 0 } .$ . Curves are EMA-smoothed.

Table 4: Initialization sensitivity on AlpacaEval 2. We report length-controlled (LC) win rate (%). † denotes the reference setting. Full LC/WR results are provided in Appendix I.1.
<table><tr><td></td><td colspan="5">Initial coefficient  $\beta _ { 0 }$ </td></tr><tr><td>Method</td><td> $0 . 0 1 ^ { \dagger }$ </td><td>0.05</td><td>0.10</td><td>0.30</td><td>0.80</td></tr><tr><td>DPO</td><td>19.2</td><td>10.4</td><td>8.7</td><td>8.3</td><td>6.6</td></tr><tr><td>β-DPO</td><td>16.6</td><td>10.5</td><td>10.8</td><td>7.3</td><td>6.9</td></tr><tr><td>ε-DPO</td><td>19.6</td><td>15.6</td><td>12.9</td><td>8.8</td><td>6.9</td></tr><tr><td>LSC-DPO</td><td>22.7</td><td>19.6</td><td>19.1</td><td>15.3</td><td>15.2</td></tr></table>

learning-signal trajectory $\sigma ( - z )$ . Although LSC-DPO eventually regulates the trajectories toward a similar controlled level, larger $\beta _ { 0 }$ causes a faster decay of $\sigma ( - z )$ during the early stage of training.

Signal-budget compensation. We hypothesize that the residual performance variation across different initial coefficients is related to differences in the cumulative learning signal during the transient stage. For an arbitrary anchor run $^ { a , }$ we define the cumulative signal deficit as

$$
D _ { H } ( \beta _ { 0 } ) = \sum _ { t = 1 } ^ { H } \left[ \sigma ( - z _ { t } ^ { a } ) - \sigma ( - z _ { t } ^ { ( \beta _ { 0 } ) } ) \right] ,\tag{19}
$$

where H denotes the end of the transient phase. A positive $D _ { H } ( \beta _ { 0 } )$ indicates that the run accumulates a smaller learning signal than the anchor during this stage. We compensate for this deficit over the remaining $T - H$ steps by setting

$$
\sigma ( - z _ { \mathrm { c o m p } } ^ { * } ) = \sigma ( - z _ { a } ^ { * } ) + \frac { D _ { H } ( \beta _ { 0 } ) } { T - H } , \qquad z _ { \mathrm { c o m p } } ^ { * } = \log \frac { 1 - \sigma ( - z _ { \mathrm { c o m p } } ^ { * } ) } { \sigma ( - z _ { \mathrm { c o m p } } ^ { * } ) } ,\tag{20}
$$

where $z _ { \mathrm { c o m p } } ^ { * }$ denotes the compensated target scaled margin corresponding to the compensated learning-signal target. Thus, a larger early-stage deficit leads to a higher compensated learning signal target and, equivalently, a smaller $z ^ { * }$ . Under ideal post-transient tracking, this rule matches the cumulative learning signal of the anchor run. We evaluate below whether this compensation also reduces the corresponding performance gap.

Compensation results. Table 5 shows that signal-budget compensation improves the LC win rate for all tested non-anchor initializations, with gains of up to 5.1 points, while reducing the LC spread from 4.4 to 1.8 points. These results indicate that adjusting the learning signal can partially compensate for the performance differences induced by coefficient initialization. Overall, the compensation substantially reduces, but does not eliminate, initialization sensitivity. Full results and additional analyses are provided in Appendix I.2.

## 5 RELATED WORK

Reinforcement learning from human feedback. Reinforcement learning from human feedback (RLHF) has become a central approach for aligning large language models with human preferences and values (Bai et al., 2022; Christiano et al., 2017; Ouyang et al., 2022; Stiennon et al., 2020;

Table 5: Effect of full signal-budget compensation on AlpacaEval 2. We report length-controlled (LC) win rate (%).
<table><tr><td></td><td colspan="2">Baseline</td><td colspan="3">Full compensation</td></tr><tr><td> $\beta _ { 0 }$ </td><td> $z ^ { * }$ </td><td>LC</td><td> $z _ { \mathrm { c o m p } } ^ { * }$ </td><td>LC</td><td>(∆)</td></tr><tr><td>0.05</td><td>0.400</td><td>19.6</td><td>0.359</td><td>21.5</td><td>(+1.9)</td></tr><tr><td>0.10</td><td>0.400</td><td>19.1</td><td>0.348</td><td>21.1</td><td>(+2.0)</td></tr><tr><td>0.30</td><td>0.400</td><td>15.3</td><td>0.333</td><td>20.4</td><td>(+5.1)</td></tr><tr><td>0.80</td><td>0.400</td><td>15.2</td><td>0.324</td><td>19.7</td><td>(+4.5)</td></tr></table>

Ziegler et al., 2019). A typical RLHF pipeline first collects preference comparisons, trains a reward model from human feedback (Chen et al., 2024; Gao et al., 2023; Havrilla et al., 2024; Luo et al., 2023), and then optimizes the policy with a reinforcement learning algorithm (Anthony et al., 2017; Schulman et al., 2017). Proximal policy optimization (PPO) is commonly used for this policy optimization stage (Schulman et al., 2017). Preference-based alignment has also been applied to improve helpfulness and safety, reduce toxic outputs, improve factuality, and control multiple objectives (Dai et al., 2023; Korbak et al., 2023; Zheng et al., 2023a; Tian et al., 2024; Wang et al., 2024).

Direct preference optimization and variants. DPO (Rafailov et al., 2023) has motivated a broad family of direct alignment methods that optimize language models from offline preference data without training an explicit reward model. Representative extensions include IPO (Azar et al., 2024), KTO (Ethayarajh et al., 2024), RRHF (Yuan et al., 2023), R-DPO (Park et al., 2024), ODPO (Amini et al., 2024), CPO (Xu et al., 2024), ORPO (Hong et al., 2024), SLiC-HF (Zhao et al., 2023), and SimPO (Meng et al., 2024). These methods extend preference optimization through different objective designs, reference-model formulations, and robustness mechanisms.

DPO training dynamics and coefficient control. Recent work has studied DPO optimization from training-dynamics and coefficient-control perspectives. Direct alignment methods can also exhibit overoptimization behavior as the policy moves away from the reference model (Rafailov et al., 2024). A related line of work studies this training behavior and mitigates it by updating the reference policy during training (Gorbatovski et al., 2024). In the coefficient-control direction, β- DPO (Wu et al., 2024b) updates β using empirical reward-margin statistics, while ε-DPO (Lee et al., 2025) perturbs the coefficient to reduce sensitivity to a fixed β. Related margin-adaptive approaches include AlphaDPO (Wu et al., 2024a) and γ-PO (Sun et al., 2025).

## 6 CONCLUSION

In this paper, we introduced Learning-Signal-Controlled Direct Preference Optimization (LSC-DPO), a feedback-control approach that regulates the learning signal of DPO during preference optimization. By analyzing the DPO objective from a loss-level geometric perspective, we identified the sigmoid factor as a learning signal that characterizes the local sensitivity of the objective and used it to design a control rule that regulates DPO training dynamics around a target learning-signal level. Our log-space analysis shows that, under a simple local stability condition, the feedback update can stably track the target learning-signal regime. Empirical results further show that LSC-DPO maintains more sustained learning-signal dynamics and achieves stronger alignment performance. We further showed that different coefficient initializations are related to learning-signal dynamics and derived a signal-budget compensation rule that substantially reduces the resulting performance variation.

Limitations and Future Work. Although the best-performing learning-signal levels are similar across the model families studied in this work, we do not yet provide a complete theory explaining why these target ranges emerge or when they transfer across models and datasets. In addition, although signal-budget compensation substantially reduces initialization-induced performance variation, it does not completely eliminate the remaining performance gap. Future work may develop a more complete theory of target learning-signal selection and signal-budget dynamics, and extend learning-signal control beyond DPO to other preference-optimization objectives.

## AI USE STATEMENT

Generative AI tools were used in this work to edit the manuscript for grammar and readability, to check the formatting and the bibliography against the original sources, and to assist with writing and debugging the training and evaluation scripts. Generative AI tools were not used to generate synthetic data sets, to develop the theoretical framework, or to formulate or prove the mathematical claims; the remaining tasks with required disclosure are not applicable to this work. All AI-assisted text was reviewed and revised by the authors, AI-assisted code was verified and tested by the authors, and every citation was checked against the original publication. We take responsibility for the final content of this work, including text, claims, and artifacts produced with the aid of generative AI.

## ETHICS STATEMENT

This work studies preference optimization for language-model alignment. All experiments use publicly available datasets (UltraChat-200k, UltraFeedback, and Anthropic-HH) and open-weight base models under their respective licenses, and no human-subject data were collected. The Anthropic-HH harmless-base subset contains prompts designed to elicit harmful responses; we use it only to evaluate harmlessness. As with any preference-optimization method, LSC-DPO can in principle be used to align a model toward undesirable preferences; our method does not change this risk relative to DPO. We do not release any new dataset.

## REPRODUCIBILITY STATEMENT

The complete LSC-DPO procedure is given in Algorithm 1 (Appendix B). Models, datasets, hyperparameters, and compute are listed in Appendix E.1, and the evaluation protocols, judge models, and decoding settings in Appendix E.2. The assumptions and full proofs of the stability results are in Appendices C and D. The predetermined β schedules and the signal-budget compensation targets are specified in Appendices H.2 and I.2.

## REFERENCES

Afra Amini, Tim Vieira, and Ryan Cotterell. Direct preference optimization with an offset. In Findings ofthe Associationfor Computational Linguistics: ACL 2024, pp. 9954–9972, 2024.

Thomas Anthony, Zheng Tian, and David Barber. Thinking fast and slow with deep learning and tree search. Advances in Neural Information Processing Systems, 30, 2017.

Mohammad Gheshlaghi Azar, Zhaohan Daniel Guo, Bilal Piot, Remi Munos, Mark Rowland, Michal Valko, and Daniele Calandriello. A general theoretical paradigm to understand learning from human preferences. In International Conference on Artificial Intelligence and Statistics, pp. 4447–4455. PMLR, 2024.

Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, et al. Training a helpful and harmless assistant with reinforcement learning from human feedback. arXiv preprint arXiv:2204.05862, 2022.

Ralph Allan Bradley and Milton E Terry. Rank analysis of incomplete block designs: I. The method of paired comparisons. Biometrika, 39(3/4):324–345, 1952.

Lichang Chen, Chen Zhu, Jiuhai Chen, Davit Soselia, Tianyi Zhou, Tom Goldstein, Heng Huang, Mohammad Shoeybi, and Bryan Catanzaro. ODIN: Disentangled reward mitigates hacking in RLHF. In International Conference on Machine Learning, volume 235, pp. 7935–7952. PMLR, 2024.

Paul F Christiano, Jan Leike, Tom Brown, Miljan Martic, Shane Legg, and Dario Amodei. Deep reinforcement learning from human preferences. Advances in Neural Information Processing Systems, 30, 2017.

Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? Try ARC, the AI2 reasoning challenge. arXiv preprint arXiv:1803.05457, 2018.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.

Ganqu Cui, Lifan Yuan, Ning Ding, Guanming Yao, Bingxiang He, Wei Zhu, Yuan Ni, Guotong Xie, Ruobing Xie, Yankai Lin, et al. UltraFeedback: Boosting language models with scaled AI feedback. arXiv preprint arXiv:2310.01377, 2023.

Josef Dai, Xuehai Pan, Ruiyang Sun, Jiaming Ji, Xinbo Xu, Mickel Liu, Yizhou Wang, and Yaodong Yang. Safe RLHF: Safe reinforcement learning from human feedback. arXiv preprint arXiv:2310.12773, 2023.

Ning Ding, Yulin Chen, Bokai Xu, Yujia Qin, Shengding Hu, Zhiyuan Liu, Maosong Sun, and Bowen Zhou. Enhancing chat language models by scaling high-quality instructional conversations. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pp. 3029–3051, 2023.

Yann Dubois, Balazs Galambosi, Percy Liang, and Tatsunori B Hashimoto. Length-controlled Al-´ pacaEval: A simple way to debias automatic evaluators. arXiv preprint arXiv:2404.04475, 2024.

Kawin Ethayarajh, Winnie Xu, Niklas Muennighoff, Dan Jurafsky, and Douwe Kiela. Model alignment as prospect theoretic optimization. In International Conference on Machine Learning, volume 235, pp. 12634–12651. PMLR, 2024.

Leo Gao, John Schulman, and Jacob Hilton. Scaling laws for reward model overoptimization. In International Conference on Machine Learning, pp. 10835–10866. PMLR, 2023.

Dongyoung Go, Tomasz Korbak, German Kruszewski, Jos Rozen, Nahyeon Ryu, and Marc Dymet-´ man. Aligning language models with preferences through f-divergence minimization. In International Conference on Machine Learning, volume 202, pp. 11546–11583. PMLR, 2023.

Alexey Gorbatovski, Boris Shaposhnikov, Alexey Malakhov, Nikita Surnachev, Yaroslav Aksenov, Ian Maksimov, Nikita Balagansky, and Daniil Gavrilov. Learn your reference model for real good alignment. arXiv preprint arXiv:2404.09656, 2024.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The Llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.

Alex Havrilla, Sharath Raparthy, Christoforus Nalmpantis, Jane Dwivedi-Yu, Maksym Zhuravinskyi, Eric Hambro, and Roberta Raileanu. GLoRe: When, where, and how to improve LLM reasoning via global and local refinements. arXiv preprint arXiv:2402.10963, 2024.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. arXiv preprint arXiv:2009.03300, 2020.

Jiwoo Hong, Noah Lee, and James Thorne. ORPO: Monolithic preference optimization without reference model. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pp. 11170–11189, 2024.

Natasha Jaques, Shixiang Gu, Dzmitry Bahdanau, Jose Miguel Hern´ andez-Lobato, Richard E´ Turner, and Douglas Eck. Sequence tutor: Conservative fine-tuning of sequence generation models with KL-control. In International Conference on Machine Learning, pp. 1645–1654. PMLR, 2017.

Natasha Jaques, Judy Hanwen Shen, Asma Ghandeharioun, Craig Ferguson, Agata Lapedriza, Noah Jones, Shixiang Gu, and Rosalind Picard. Human-centric dialog training via offline reinforcement learning. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 3985–4003, 2020.

Tomasz Korbak, Kejian Shi, Angelica Chen, Rasika Vinayak Bhalerao, Christopher Buckley, Jason Phang, Samuel R Bowman, and Ethan Perez. Pretraining language models with human preferences. In International Conference on Machine Learning, pp. 17506–17533. PMLR, 2023.

Sangkyu Lee, Janghoon Han, Hosung Song, Stanley Jungkyu Choi, Honglak Lee, and Youngjae Yu. KL penalty control via perturbation for direct preference optimization. Advances in Neural Information Processing Systems, 38:168095–168121, 2025.

Tianle Li, Wei-Lin Chiang, Evan Frick, Lisa Dunlap, Tianhao Wu, Banghua Zhu, Joseph E Gonzalez, and Ion Stoica. From crowdsourced data to high-quality benchmarks: Arena-Hard and BenchBuilder pipeline. arXiv preprint arXiv:2406.11939, 2024.

Stephanie Lin, Jacob Hilton, and Owain Evans. TruthfulQA: Measuring how models mimic human falsehoods. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 3214–3252, 2022.

Haipeng Luo, Qingfeng Sun, Can Xu, Pu Zhao, Jianguang Lou, Chongyang Tao, Xiubo Geng, Qingwei Lin, Shifeng Chen, and Dongmei Zhang. WizardMath: Empowering mathematical reasoning for large language models via reinforced Evol-Instruct. arXiv preprint arXiv:2308.09583, 2023.

Yu Meng, Mengzhou Xia, and Danqi Chen. SimPO: Simple preference optimization with a reference-free reward. Advances in Neural Information Processing Systems, 37:124198–124235, 2024.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in Neural Information Processing Systems, 35: 27730–27744, 2022.

Ryan Park, Rafael Rafailov, Stefano Ermon, and Chelsea Finn. Disentangling length from quality in direct preference optimization. In Findings of the Association for Computational Linguistics: ACL 2024, pp. 4998–5017, 2024.

Xue Bin Peng, Aviral Kumar, Grace Zhang, and Sergey Levine. Advantage-weighted regression: Simple and scalable off-policy reinforcement learning. arXiv preprint arXiv:1910.00177, 2019.

Jan Peters and Stefan Schaal. Reinforcement learning by reward-weighted regression for operational space control. In Proceedings of the 24th International Conference on Machine Learning, pp. 745–750, 2007.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. Advances in Neural Information Processing Systems, 36:53728–53741, 2023.

Rafael Rafailov, Yaswanth Chittepu, Ryan Park, Harshit Sikchi, Joey Hejna, W Bradley Knox, Chelsea Finn, and Scott Niekum. Scaling laws for reward model overoptimization in direct alignment algorithms. Advances in Neural Information Processing Systems, 37:126207–126242, 2024.

Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. WinoGrande: An adversarial Winograd schema challenge at scale. Communications of the ACM, 64(9):99–106, 2021. doi: 10.1145/3474381.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

Nisan Stiennon, Long Ouyang, Jeffrey Wu, Daniel Ziegler, Ryan Lowe, Chelsea Voss, Alec Radford, Dario Amodei, and Paul F Christiano. Learning to summarize with human feedback. Advances in Neural Information Processing Systems, 33:3008–3021, 2020.

Jie Sun, Junkang Wu, Jiancan Wu, Zhibo Zhu, Xingyu Lu, Jun Zhou, Lintao Ma, and Xiang Wang. Robust preference optimization via dynamic target margins. In Findings of the Association for Computational Linguistics: ACL 2025, pp. 5399–5416, 2025.

Katherine Tian, Eric Mitchell, Huaxiu Yao, Christopher D Manning, and Chelsea Finn. Fine-tuning language models for factuality. In The Twelfth International Conference on Learning Representations, 2024.

Haoxiang Wang, Yong Lin, Wei Xiong, Rui Yang, Shizhe Diao, Shuang Qiu, Han Zhao, and Tong Zhang. Arithmetic control of LLMs for diverse user preferences: Directional preference alignment with multi-objective rewards. In Proceedings ofthe 62nd Annual Meeting ofthe Association for Computational Linguistics (Volume 1: Long Papers), pp. 8642–8655, 2024.

Junkang Wu, Xue Wang, Zhengyi Yang, Jiancan Wu, Jinyang Gao, Bolin Ding, Xiang Wang, and Xiangnan He. AlphaDPO: Adaptive reward margin for direct preference optimization. arXiv preprint arXiv:2410.10148, 2024a.

Junkang Wu, Yuexiang Xie, Zhengyi Yang, Jiancan Wu, Jinyang Gao, Bolin Ding, Xiang Wang, and Xiangnan He. β-DPO: Direct preference optimization with dynamic β. Advances in Neural Information Processing Systems, 37:129944–129966, 2024b.

Haoran Xu, Amr Sharaf, Yunmo Chen, Weiting Tan, Lingfeng Shen, Benjamin Van Durme, Kenton Murray, and Young Jin Kim. Contrastive preference optimization: Pushing the boundaries of LLM performance in machine translation. In International Conference on Machine Learning, pp. 55204–55224. PMLR, 2024.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

Zheng Yuan, Hongyi Yuan, Chuanqi Tan, Wei Wang, Songfang Huang, and Fei Huang. RRHF: Rank responses to align language models with human feedback without tears. arXiv preprint arXiv:2304.05302, 2023.

Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. HellaSwag: Can a machine really finish your sentence? In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pp. 4791–4800, 2019.

Yao Zhao, Rishabh Joshi, Tianqi Liu, Misha Khalman, Mohammad Saleh, and Peter J Liu. SLiC-HF: Sequence likelihood calibration with human feedback. arXiv preprint arXiv:2305.10425, 2023.

Chujie Zheng, Pei Ke, Zheng Zhang, and Minlie Huang. Click: Controllable text generation with sequence likelihood contrastive learning. In Findings of the Association for Computational Linguistics: ACL 2023, pp. 1022–1040, 2023a.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, et al. Judging LLM-as-a-judge with MT-Bench and Chatbot Arena. Advances in Neural Information Processing Systems, 36:46595–46623, 2023b.

Daniel M Ziegler, Nisan Stiennon, Jeffrey Wu, Tom B Brown, Alec Radford, Dario Amodei, Paul Christiano, and Geoffrey Irving. Fine-tuning language models from human preferences. arXiv preprint arXiv:1909.08593, 2019.

## A LEARNING SIGNAL σ(−z) UNDER FIXED $\beta$

We report the raw per-step trajectories of the learning signal under fixed-β DPO with different β values. Figures 6a and 6b show that σ(−z) generally decreases during training on both Anthropic-HH subsets.

![](images/2b1f7e20761e5c68dc412303b93f51b2970d252b7c461981a30ec99297de7e1e.jpg)  
(a) harmless-base

![](images/4515b09a45692b02c5f222345ef136df654513c9fcbeeed55b4e00ab2d33351f.jpg)  
(b) helpful-base  
Figure 6: Learning signal under DPO. Raw per-step trajectories of $\sigma ( - z )$ are shown for different fixed $\beta$ values on Anthropic-HH: (a) harmless-base and (b) helpful-base.

## B LSC-DPO ALGORITHM

Algorithm 1 summarizes the complete training procedure of LSC-DPO, including computation of the learning signal, controller activation, and the subsequent coefficient update described in Section 3.2.

Algorithm 1 Learning-Signal-Controlled Direct Preference Optimization   
Require: policy π , reference $\pi _ { \mathrm { r e f } } ,$ , initial $\beta _ { 0 } , z ^ { * }$ , η, τ<sub>start</sub>   
1: $\beta \gets \beta _ { 0 } ;$ control is inactive   
2: while not converged do   
3: Sample a batch of preference triples $( x , y _ { + } , y _ { - } ) \sim \mathcal { D }$   
4: Compute margin m, scaled margin $z = \beta m ,$ and signal $\sigma ( - z )$   
5: Activate control if $\dot { } \sigma ( - z ) \le \tau _ { \mathrm { s t } }$ art   
6: If control is active, update $\overline { { \beta } } \gets \beta \exp [ - \eta ( z - z ^ { * } ) ]$   
7: Update $\pi _ { \theta }$ using the DPO loss with the current $\beta$   
8: end while   
9: return aligned policy π<sub>θ</sub>

## C PROOFS FOR THE STABILITY ANALYSIS

In this appendix, we provide the formal statements and detailed proofs for the local stability and bounded tracking results summarized in Section 3.3. Recall that the log-space update is

$$
x _ { k + 1 } = F _ { k } ( x _ { k } ) , \qquad F _ { k } ( x ) : = x - \eta ( e ^ { x } m _ { k } - z ^ { * } ) ,\tag{21}
$$

and that the instantaneous equilibrium is

$$
x _ { k } ^ { f } = \log { \frac { z ^ { * } } { m _ { k } } } .\tag{22}
$$

## C.1 PROOF OF LOCAL STABILITY

The formal local-stability result is stated as follows.

Proposition 1 (Local stability). Assume $m _ { k } \ > \ 0$ and $z ^ { * } > 0 .$ . The instantaneous equilibrium $x _ { k } ^ { f } = \log ( z ^ { * } / m _ { k } )$ is locally asymptotically stable whenever

$$
0 < \eta z ^ { * } < 2 .\tag{23}
$$

The corresponding equilibrium linearizationfactor is

$$
\rho _ { 0 } : = | 1 - \eta z ^ { * } | < 1 .\tag{24}
$$

This result follows by linearizing $F _ { k }$ around $x _ { k } ^ { f } ,$ , where

$$
F _ { k } ^ { \prime } ( x _ { k } ^ { f } ) = 1 - \eta z ^ { \ast } .\tag{25}
$$

Proof. Since $m _ { k } > 0$ and $z ^ { * } > 0$ , the instantaneous equilibrium is well defined and satisfies

$$
e ^ { x _ { k } ^ { f } } m _ { k } = z ^ { \ast } .\tag{26}
$$

The derivative of the step-dependent map is

$$
F _ { k } ^ { \prime } ( x ) = 1 - \eta e ^ { x } m _ { k } .\tag{27}
$$

Evaluating this derivative at the equilibrium gives

$$
\begin{array} { r } { F _ { k } ^ { \prime } ( x _ { k } ^ { f } ) = 1 - \eta e ^ { x _ { k } ^ { f } } m _ { k } = 1 - \eta z ^ { \ast } . } \end{array}\tag{28}
$$

For a one-dimensional discrete-time dynamical system, a fixed point is locally asymptotically stable if the derivative at the fixed point has magnitude smaller than one. Hence,

$$
| F _ { k } ^ { \prime } ( x _ { k } ^ { f } ) | < 1 \iff | 1 - \eta z ^ { * } | < 1 .\tag{29}
$$

Solving this inequality yields

$$
0 < \eta z ^ { * } < 2 .\tag{30}
$$

Under this condition, the equilibrium linearization factor is

$$
\rho _ { 0 } : = | F _ { k } ^ { \prime } ( x _ { k } ^ { f } ) | = | 1 - \eta z ^ { \ast } | < 1 ,\tag{31}
$$

which proves the proposition.

## C.2 BOUNDED TRACKING THEOREM AND PROOF

Theorem 1 (Bounded tracking). Assume the local stability condition in $E q .$ . 17 holds, and define

$$
\rho _ { 0 } : = | 1 - \eta z ^ { * } | < 1 .
$$

For any ρ satisfying

$$
\rho _ { 0 } < \rho < 1 ,
$$

there exists a radius $r > 0$ such that, for every $k ,$

$$
| F _ { k } ^ { \prime } ( x ) | \leq \rho \qquad w h e n e v e r | x - x _ { k } ^ { f } | \leq r .\tag{32}
$$

Suppose that there exists a step $T$ such that

$$
| x _ { T } - x _ { T } ^ { f } | \leq r ,\tag{33}
$$

and that,for some constant $\bar { \delta } \geq 0$ and all $k \geq T$

$$
| x _ { k + 1 } ^ { f } - x _ { k } ^ { f } | \leq \bar { \delta } , \qquad \bar { \delta } \leq ( 1 - \rho ) r .\tag{34}
$$

Then the iterates remain within the local contraction neighborhood,

$$
| x _ { k } - x _ { k } ^ { f } | \leq r , \qquad \forall k \geq T ,
$$

and satisfy

$$
\left| x _ { k } - x _ { k } ^ { f } \right| \leq \rho ^ { k - T } | x _ { T } - x _ { T } ^ { f } | + \frac { \bar { \delta } } { 1 - \rho } , \qquad \forall k \geq T .\tag{35}
$$

The complete proof is given below. An interpretation of the resulting tracking bound is provided in Appendix C.3.

We define the tracking error by

$$
e _ { k } : = x _ { k } - x _ { k } ^ { f } .\tag{36}
$$

Lemma 1 (One-step tracking recursion). For each step $k ,$ there exists some $\xi _ { k }$ between $x _ { k }$ and $x _ { k } ^ { f }$ such that

$$
e _ { k + 1 } = F _ { k } ^ { \prime } ( \xi _ { k } ) e _ { k } + \left( x _ { k } ^ { f } - x _ { k + 1 } ^ { f } \right) .\tag{37}
$$

Consequently,

$$
| e _ { k + 1 } | \leq | F _ { k } ^ { \prime } ( \xi _ { k } ) | | e _ { k } | + | x _ { k + 1 } ^ { f } - x _ { k } ^ { f } | .\tag{38}
$$

Proof. By definition,

$$
e _ { k + 1 } = x _ { k + 1 } - x _ { k + 1 } ^ { f } .\tag{39}
$$

Using the update $x _ { k + 1 } = F _ { k } ( x _ { k } )$ and adding and subtracting $x _ { k } ^ { f } .$ , we obtain

$$
\begin{array} { r l } & { e _ { k + 1 } = F _ { k } ( x _ { k } ) - x _ { k + 1 } ^ { f } } \\ & { \qquad = \Big ( F _ { k } ( x _ { k } ) - F _ { k } ( x _ { k } ^ { f } ) \Big ) + \Big ( x _ { k } ^ { f } - x _ { k + 1 } ^ { f } \Big ) , } \end{array}\tag{40}
$$

where we used $F _ { k } ( x _ { k } ^ { f } ) = x _ { k } ^ { f }$ . By the mean value theorem, there exists some $\xi _ { k }$ between $x _ { k }$ and $x _ { k } ^ { f }$ such that

$$
F _ { k } ( x _ { k } ) - F _ { k } ( x _ { k } ^ { f } ) = F _ { k } ^ { \prime } ( \xi _ { k } ) ( x _ { k } - x _ { k } ^ { f } ) = F _ { k } ^ { \prime } ( \xi _ { k } ) e _ { k } .\tag{41}
$$

Substituting this identity gives Eq. 37. Taking absolute values gives Eq. 38.

□

ProofofTheorem 1. From Proposition 1, the derivative at the instantaneous equilibrium satisfies

$$
\rho _ { 0 } : = | F _ { k } ^ { \prime } ( x _ { k } ^ { f } ) | = | 1 - \eta z ^ { \ast } | < 1 .\tag{42}
$$

Choose any $\rho$ such that

$$
\rho _ { 0 } < \rho < 1 .\tag{43}
$$

For a displacement e from the instantaneous equilibrium,

$$
F _ { k } ^ { \prime } ( x _ { k } ^ { f } + e ) = 1 - \eta e ^ { x _ { k } ^ { f } + e } m _ { k } = 1 - \eta z ^ { \ast } e ^ { e } .\tag{44}
$$

The right-hand side is independent of $m _ { k }$ . Since it is continuous in e and equals $1 - \eta z ^ { * }$ at $e = 0$ there exists a radius $r > 0$ such that

$$
| F _ { k } ^ { \prime } ( x _ { k } ^ { f } + e ) | \leq \rho , \qquad { \mathrm { f o r ~ a l l ~ } } | e | \leq r\tag{45}
$$

and for all $k \geq T$

We now show that this neighborhood is invariant. By assumption,

$$
| e _ { T } | \le r .\tag{46}
$$

Suppose inductively that $| e _ { k } | \le r$ . Since $\xi _ { k }$ lies between $x _ { k }$ and $x _ { k } ^ { f } .$ , it also satisfies

$$
| \xi _ { k } - x _ { k } ^ { f } | \leq | x _ { k } - x _ { k } ^ { f } | = | e _ { k } | \leq r .\tag{47}
$$

Therefore,

$$
| F _ { k } ^ { \prime } ( \xi _ { k } ) | \le \rho .\tag{48}
$$

Using Lemma 1 and the bounded-drift assumption gives

$$
| e _ { k + 1 } | \leq \rho | e _ { k } | + \bar { \delta } .\tag{49}
$$

Since $\bar { \delta } \leq ( 1 - \rho ) r$

$$
| e _ { k + 1 } | \le \rho r + ( 1 - \rho ) r = r .\tag{50}
$$

Hence, by induction,

$$
| e _ { k } | \leq r , \qquad \forall k \geq T .\tag{51}
$$

Thus, the local contraction bound used in Eq. 49 remains valid along the entire tracked trajectory.

Unrolling Eq. 49 gives

$$
| e _ { k } | \leq \rho ^ { k - T } | e _ { T } | + \bar { \delta } \sum _ { j = 0 } ^ { k - T - 1 } \rho ^ { j } .\tag{52}
$$

Since $0 \leq \rho < 1$

$$
\sum _ { j = 0 } ^ { k - T - 1 } \rho ^ { j } = \frac { 1 - \rho ^ { k - T } } { 1 - \rho } \leq \frac { 1 } { 1 - \rho } .\tag{53}
$$

Substituting this bound into Eq. 52 yields

$$
| e _ { k } | \le \rho ^ { k - T } | e _ { T } | + \frac { \bar { \delta } } { 1 - \rho } , \qquad \forall k \ge T .\tag{54}
$$

Since $e _ { k } = x _ { k } - x _ { k } ^ { f }$ , this proves the theorem.

## C.3 INTERPRETATION OF THE BOUND

The bound

$$
\vert e _ { k } \vert \leq \rho ^ { k - T } \vert e _ { T } \vert + \frac { \bar { \delta } } { 1 - \rho }\tag{55}
$$

contains two terms. The first term, $\rho ^ { k - T } | e _ { T } |$ , is a transient error that decays geometrically. The second term, $\bar { \delta } / ( 1 - \rho )$ , is a persistent tracking-error bound controlled by the maximum equilibrium drift. Therefore, the log-space $\beta$ dynamics remains close to the instantaneous equilibrium trajectory as long as the equilibrium drift is bounded and the trajectory stays within the local contraction neighborhood.

## D A SUFFICIENT CONDITION FOR BOUNDED EQUILIBRIUM DRIFT

The bounded tracking theorem assumes that the moving equilibrium satisfies

$$
| x _ { k + 1 } ^ { f } - x _ { k } ^ { f } | \leq \bar { \delta } .
$$

We show that this condition follows from a simple regularity condition on the margin statistic.

Lemma 2 (Sufficient condition for bounded equilibrium drift). Assume that there exist constants $T \in \mathbb { N } , m _ { \operatorname* { m i n } } > 0$ , and $\varepsilon \geq 0$ such thatfor all $\bar { k } \geq T$

$$
m _ { k } \geq m _ { \operatorname* { m i n } } , \qquad | m _ { k + 1 } - m _ { k } | \leq \varepsilon .
$$

Then

$$
| x _ { k + 1 } ^ { f } - x _ { k } ^ { f } | \leq \frac { \varepsilon } { m _ { \mathrm { m i n } } } , \qquad \forall k \geq T .
$$

Proof. Using $x _ { k } ^ { f } = \log ( z ^ { * } / m _ { k } ) = \log z ^ { * } - \log m _ { k }$ , we have

$$
| x _ { k + 1 } ^ { f } - x _ { k } ^ { f } | = | \log m _ { k + 1 } - \log m _ { k } | .
$$

By the mean value theorem, for some $\zeta _ { k }$ between $m _ { k }$ and $m _ { k + 1 }$

$$
| \log m _ { k + 1 } - \log m _ { k } | = \frac { 1 } { \zeta _ { k } } | m _ { k + 1 } - m _ { k } | .
$$

Since $m _ { k } , m _ { k + 1 } \geq m _ { \operatorname* { m i n } } > 0$ , we have $\zeta _ { k } \ge m _ { \operatorname* { m i n } }$ . Therefore,

$$
| x _ { k + 1 } ^ { f } - x _ { k } ^ { f } | \leq \frac { 1 } { m _ { \operatorname* { m i n } } } | m _ { k + 1 } - m _ { k } | \leq \frac { \varepsilon } { m _ { \operatorname* { m i n } } } .
$$

This lemma provides an explicit bound on the equilibrium drift whenever the margin statistic stays bounded away from zero and changes gradually across training steps. In particular, the drift condition in Theorem 1 is satisfied whenever $\varepsilon / m _ { \mathrm { m i n } } \leq ( 1 - \rho ) r$

## E IMPLEMENTATION DETAILS

## E.1 MODELS AND TRAINING DATASETS

UltraFeedback experiments. For the UltraFeedback experiments, we follow the SimPO (Meng et al., 2024) training setup. We first perform supervised fine-tuning of the Meta-Llama-3-8B<sup>2</sup> and Qwen3-8B-Base<sup>3</sup> models on UltraChat-200k<sup>4</sup> (Ding et al., 2023); the resulting SFT checkpoints are used as the policy initializations and reference models for preference optimization on the UltraFeedback binarized<sup>5</sup> (Cui et al., 2023) dataset. We also follow the SimPO protocol for common training settings, including the learning rate, scheduler, and optimizer, and keep them fixed across experiments. Table 6 reports these fixed training settings and the LSC-DPO control parameters, with bold entries denoting the main settings used in Table 1.

Table 6: Training configuration of LSC-DPO on UltraFeedback.
<table><tr><td>Configuration</td><td>Llama3-8B</td><td>Qwen3-8B</td></tr><tr><td>Model</td><td>meta-llama/Meta-Llama-3-8B</td><td>Qwen/Qwen3-8B-Base</td></tr><tr><td>Optimizer</td><td>AdamW</td><td>AdamW</td></tr><tr><td>Epoch</td><td>1</td><td>1</td></tr><tr><td>Batch size</td><td>128</td><td>128</td></tr><tr><td>Learning rate</td><td>5e-7</td><td>5e-7</td></tr><tr><td>Scheduler</td><td>cosine</td><td>cosine</td></tr><tr><td>Warmup ratio</td><td>0.1</td><td>0.1</td></tr><tr><td>Weight decay</td><td>0</td><td>0</td></tr><tr><td>Precision</td><td>BF16</td><td>BF16</td></tr><tr><td> $\beta _ { 0 }$ </td><td>0.01</td><td>0.01</td></tr><tr><td>η</td><td>0.1</td><td>0.1</td></tr><tr><td rowspan="2"> $z ^ { * }$ </td><td>{0.20, 0.30, 0.35, 0.40,</td><td>{0.20, 0.30, 0.35, 0.40,</td></tr><tr><td>0.45, 0.50, 0.60, 0.70}</td><td>0.45, 0.50, 0.60, 0.70}</td></tr><tr><td>Tstart</td><td>{0.40, 0.43, 0.45}</td><td>{0.40, 0.43, 0.45}</td></tr><tr><td>Device</td><td>4×H200 GPUs</td><td>4×H200 GPUs</td></tr></table>

Anthropic-HH experiments. For the Anthropic-HH (Bai et al., 2022)<sup>6</sup> experiments, we use both the harmless-base and helpful-base subsets. We follow DPO (Rafailov et al., 2023) and β-DPO (Wu et al., 2024b) for the batch size (64) and single epoch, ε-DPO (Lee et al., 2025) for the optimizer, scheduler and warmup, and keep the learning rate of $5 \times 1 0 ^ { - 7 }$ shared with our UltraFeedback runs. For method-specific hyperparameters of β-DPO and ε-DPO, we use the recommended values reported in the respective papers. Tables 7 and 8 summarize the SFT-stage and preferenceoptimization-stage configurations, respectively.

Table 7: SFT training configuration for Anthropic-HH.
<table><tr><td>Configuration</td><td>Llama3-8B</td><td>Qwen3-8B</td></tr><tr><td>Model</td><td>meta-llama/Meta-Llama-3-8B</td><td>Qwen/Qwen3-8B-Base</td></tr><tr><td>Epochs</td><td>1</td><td>1</td></tr><tr><td>Batch size</td><td>64</td><td>64</td></tr><tr><td>Scheduler</td><td>cosine</td><td>cosine</td></tr><tr><td>Warmup ratio</td><td>0.1</td><td>0.1</td></tr><tr><td>Weight decay</td><td>0</td><td>0</td></tr><tr><td>Learning rate</td><td>2e-5</td><td>2e-5</td></tr><tr><td>Precision</td><td>BF16</td><td>BF16</td></tr><tr><td>Device</td><td>4×H200 GPUs</td><td>4×H200 GPUs</td></tr></table>

<sup>2</sup>https://huggingface.co/meta-llama/Meta-Llama-3-8B; Llama 3 Community License.  
<sup>3</sup>https://huggingface.co/Qwen/Qwen3-8B-Base; Apache 2.0 License.  
<sup>4</sup>https://huggingface.co/datasets/HuggingFaceH4/ultrachat\_200k; MIT License.  
<sup>5</sup>https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback\_binarized; MIT License.  
<sup>6</sup>https://huggingface.co/datasets/Anthropic/hh-rlhf; MIT License.

Table 8: Training configurations of LSC-DPO on Anthropic-HH.
<table><tr><td>Configuration</td><td>Llama3-8B</td><td>Qwen3-8B</td></tr><tr><td>Model</td><td>meta-llama/Meta-Llama-3-8B</td><td>Qwen/Qwen3-8B-Base</td></tr><tr><td>Epochs</td><td>1</td><td>1</td></tr><tr><td>Batch size</td><td> $^ { 6 4 }$ </td><td>64</td></tr><tr><td>Scheduler</td><td>cosine</td><td>cosine</td></tr><tr><td>Warmup ratio</td><td>0.1</td><td>0.1</td></tr><tr><td>Weight decay</td><td>0</td><td>0</td></tr><tr><td>Learning rate</td><td>5e-7</td><td>5e-7</td></tr><tr><td>Precision</td><td>BF16</td><td>BF16</td></tr><tr><td>β₀</td><td>0.1</td><td>0.1</td></tr><tr><td>Tstart</td><td>{0.40, 0.43, 0.45, 0.48, 0.50}</td><td>{0.40, 0.43, 0.45, 0.48, 0.50}</td></tr><tr><td> $z ^ { * }$ </td><td>{0.35, 0.40, 0.45, 0.60, 0.80, 0.85}</td><td>{0.35, 0.40, 0.45, 0.60, 0.80, 0.85}</td></tr><tr><td>η</td><td>{0.01, 0.05, 0.1, 0.3, 0.5, 1, 5, 8}</td><td>{0.01, 0.05, 0.1, 0.3, 0.5, 1, 5, 8}</td></tr><tr><td>Device</td><td>4×H200 GPUs</td><td>4×H200 GPUs</td></tr></table>

Computation environment. All training experiments in this paper are based on the Hugging Face alignment-handbook<sup>7</sup> and TRL<sup>8</sup>. In our setup with 4 H200 GPUs, each LSC-DPO training run on UltraFeedback completes within two hours, while each run on the Anthropic-HH helpful-base or harmless-base subset completes within one hour.

## E.2 EVALUATION BENCHMARKS AND BASELINES

UltraFeedback evaluation. For the UltraFeedback experiments, we follow the SimPO openalignment evaluation protocol (Meng et al., 2024) and the official benchmark settings of AlpacaEval 2 (Dubois et al., 2024) and MT-Bench (Zheng et al., 2023b). We compare LSC-DPO with SFT, DPO (Rafailov et al., 2023), β-DPO (Wu et al., 2024b), ε-DPO (Lee et al., 2025), R-DPO (Park et al., 2024), IPO (Azar et al., 2024), KTO (Ethayarajh et al., 2024), CPO (Xu et al., 2024), ORPO (Hong et al., 2024), SLiC-HF (Zhao et al., 2023), and SimPO (Meng et al., 2024). All methods are evaluated with the same generation settings and evaluation pipeline to ensure a fair comparison. On AlpacaEval 2, we report both the raw win rate and the length-controlled win rate, with the length controlled metric used as the primary metric because preference optimization can increase response length. On MT-Bench, we report the average score over both turns.

Anthropic-HH evaluation. For the Anthropic-HH experiments, we follow the evaluation protocol used in DPO (Rafailov et al., 2023) and β-DPO (Wu et al., 2024b). We evaluate harmless-base and helpful-base separately by comparing each model response with the HH chosen response, using gpt-4-0613 as the judge with randomized response order. We report the win rate of the model response. All methods use the same decoding configuration, with temperature 0.7 and top-p sampling (p = 0.9), to ensure a fair comparison.

Control-parameter selection. We selected the LSC-DPO control parameters using a separate Anthropic-HH development subset before the main AlpacaEval 2 evaluation. This preliminary study identified a coarse candidate region around $z ^ { * } \in [ 0 . 3 5 , 0 . 4 5 ]$ and $\tau _ { \mathrm { s t a r t } } \in [ 0 . 4 0 , 0 . 4 5 ]$ . The AlpacaEval 2 results in Appendix J.1 are therefore reported as a target-sensitivity analysis rather than as an unrestricted benchmark-specific hyperparameter search. After the target and training settings were fixed, the same trained checkpoints were evaluated on MT-Bench without additional tuning.

Arena-Hard reproducibility considerations. We also considered Arena-Hard (Li et al., 2024) as an additional open-ended benchmark, but do not include it in the main evaluation because prior Arena-Hard v0.1 results for SimPO (Meng et al., 2024) and ε-DPO (Lee et al., 2025) used gpt-4-1106-preview as the judge. OpenAI has since retired this model, so the original protocol cannot be reproduced in our evaluation environment. To examine how the released checkpoints behave under currently available judging setups, we re-evaluated SimPO’s released Llama3- 8B-base preference-optimization checkpoints under two settings: Arena-Hard v0.1 judged with

gpt-4.1, and Arena-Hard v2.0 hard-prompt judged with gpt-4.1. Table 9 shows that all evaluated preference-optimization methods score below the SFT baseline under these updated settings, in contrast to their behavior on AlpacaEval 2 and MT-Bench. The relative ordering also differs substantially from prior Arena-Hard reports, suggesting that results obtained under the updated judging setups may not be directly comparable to those reported under the original protocol. We therefore use AlpacaEval 2 and MT-Bench as the main evaluations.  
Table 9: Re-evaluation of the released Llama3-8B-base preference-optimization checkpoints from the SimPO study on Arena-Hard. Panel (a) reports Arena-Hard v0.1 results using $\mathtt { g p t - 4 . 1 }$ Panel (b) reports Arena-Hard v2.0 hard-prompt results using $\mathtt { g p t - 4 . 1 }$ . The original v0.1 judge, gpt-4-1106-preview, is no longer available in our evaluation environment.
<table><tr><td rowspan=1 colspan=4>Model               Score (%)  95% CI (%)</td></tr><tr><td rowspan=2 colspan=4> $9 \mathrm { { p t } - 4 - 0 3 1 4 \ ( \mathrm { { r e f . } ) } }$     50.0     $( - 0 . 0 / + 0 . 0 )$ SFT                    26.4     ${ \left( - 1 . 9 \right) } / \left. + 1 . 7 \right)$  $( - 1 . 3 \dot { / } + 1 . 6 \dot { ) }$ </td></tr><tr><td rowspan=1 colspan=2>SFT + CPO 22.4</td></tr><tr><td rowspan=1 colspan=2> $\mathrm { S F T } + \mathrm { S i m P O }$           20.1    <e</td><td rowspan=2 colspan=2>q>\left( - 1 . 7 / + 2 . 1 \right)</eq>>( - 1 . 4 \dot { / } + 1 . 6 \dot { ) }</eq></td></tr><tr><td rowspan=1 colspan=2> $\mathrm { S F T + D P O }$             16.3    <eq</td></tr><tr><td rowspan=1 colspan=2> $\mathrm { S F T + R \mathrm { - } D P O }$           14.5    <eq</td><td rowspan=1 colspan=2>>( - 1 . 3 \dot { / } + 1 . 3 \dot { ) }</eq></td></tr><tr><td rowspan=1 colspan=2> $\mathrm { S F T } + \mathrm { I P O }$              13.4    <eq</td><td rowspan=1 colspan=2>>( - 1 . 3 \dot { / } + 1 . 5 \dot { ) }</eq></td></tr><tr><td rowspan=1 colspan=2> $\mathrm { S F T + O R P O }$            12.4    <e</td><td rowspan=1 colspan=1>q>\dot { ( - 1</td><td rowspan=1 colspan=1>. 2 / + 1 . 2 ) }</eq></td></tr><tr><td rowspan=1 colspan=1> $\mathrm { S F</td><td rowspan=1 colspan=1>T } + \mathrm { K T O }$              10.3    <e</td><td rowspan=1 colspan=1>q>( - 1 . 1 )</td><td rowspan=1 colspan=1>+ 1 . 1 )</eq></td></tr><tr><td rowspan=1 colspan=1> $\mathrm { S F T } + \</td><td rowspan=1 colspan=1>mathrm { S L i C } { \mathrm { - } } \mathrm { H F }$          7.7     <e</td><td rowspan=1 colspan=1>q>( - 0 . 9</td><td rowspan=1 colspan=1>/ + 0 . 8 )</eq></td></tr></table>

(a) Arena-Hard v0.1, judge: gpt-4.1.

<table><tr><td></td><td rowspan=1 colspan=5>Model          Score (%)  95% CI (%)</td></tr><tr><td></td><td rowspan=1 colspan=5> $\circ 3 \mathrm { - m i n i \ ( r e f . ) }$     50.0     $( - 0 . 0 / + 0 . 0 )$ SFT                9.2      $( - 0 . 8 ) + 0 . 8 )$ </td></tr><tr><td></td><td rowspan=1 colspan=2> $\mathrm { S F T } + \mathrm { C P O }$          7.4     <e</td><td rowspan=2 colspan=3>q>\dot { ( - 0 . 7 / + 0 . 9 ) }</eq>q>( - 0 . 4 \dot { / } + 0 . 4 \dot { ) }</eq></td></tr><tr><td></td><td rowspan=1 colspan=2> $\mathrm { S F T + D P O }$         2.3     <e</td></tr><tr><td></td><td rowspan=1 colspan=2> $\mathrm { S F T } + \mathrm { S i m P O }$        1.8     <e</td><td rowspan=1 colspan=1>q>( - 0 . 4 \dot { /</td><td rowspan=1 colspan=2>} + 0 . 4 \dot { ) }</eq></td></tr><tr><td></td><td rowspan=1 colspan=1> $\mathrm {</td><td rowspan=1 colspan=1>S F T + I P O }$           1.4     <e</td><td rowspan=1 colspan=1>q>( - 0 . 3 \dot { /</td><td rowspan=1 colspan=1>} + 0 . 4 \dot { ) }</eq></td><td rowspan=1 colspan=1>)</td></tr><tr><td rowspan=2 colspan=2> $\mathr</td><td rowspan=1 colspan=1>SFT</td><td rowspan=1 colspan=3><eq>\mathrm { S F T + R \mathrm { - } D P O }$       1.4     <e</td></tr><tr><td rowspan=1 colspan=1>m { S F T + O R P O }</eq>       1.1     <e</td><td rowspan=1 colspan=1>q>\dot { ( - 0</td><td rowspan=1 colspan=1>. 4 / + 0 . 4 ) }</eq></td><td rowspan=1 colspan=1>)</td></tr><tr><td rowspan=1 colspan=2> $\mathrm { S F</td><td rowspan=1 colspan=1>T } + \mathrm { K T O }$          0.7     <e</td><td rowspan=1 colspan=1>q>( - 0 . 2</td><td rowspan=1 colspan=1>/ + 0 . 3 )</eq></td><td rowspan=1 colspan=1>)</td></tr><tr><td rowspan=1 colspan=2> $\mathrm { S F T } +</td><td rowspan=1 colspan=1>\mathrm { S L i C } \mathrm { - } \mathrm { H F }$     0.5     <e</td><td rowspan=1 colspan=1>q>( - 0 . 2</td><td rowspan=1 colspan=1>/ + 0 . 2 )</eq></td><td rowspan=1 colspan=1></td></tr></table>

(b) Arena-Hard v2.0 hard-prompt, judge: gpt-4.1.

## F ADDITIONAL LEARNING-SIGNAL TRAJECTORIES ON ANTHROPIC-HH

This appendix provides additional learning-signal trajectories for the Anthropic-HH analysis in Section 4.2. We first report the raw per-step trajectories corresponding to the EMA-smoothed Llama3- 8B results, and then provide the corresponding Qwen3-8B results as a cross-model comparison.

## F.1 RAW LEARNING-SIGNAL TRAJECTORIES ON LLAMA3-8B

Figures 7a and 7b show the raw per-step trajectories together with their EMA-smoothed trends for Llama3-8B. Compared with the smoothed curves reported in Section 4.2, the raw trajectories make the short-term fluctuations in the learning signal more visible, especially for β-DPO. This behavior suggests that dynamically adjusting $\beta$ based on margin statistics can introduce large shortterm fluctuations in the learning signal. In contrast, LSC-DPO maintains a more stable smoothed trajectory while avoiding the persistent low-signal regime observed under DPO, particularly on the helpful-base subset.

![](images/bc152b92a1e1432e89245bac19253c12c731341c2fc8a0ab4ce2aaa62bf68c64.jpg)  
(a) harmless-base

![](images/a64b49e40ba3a5c3636c8d4a43e392e456c2dce37af6d6f1bf453a5bab4785c8.jpg)  
(b) helpful-base  
Figure 7: Raw and EMA-smoothed trajectories of the learning signal $\sigma ( - z )$ on Anthropic-HH with Llama3-8B. Faint curves show raw per-step values, while bold curves show EMA-smoothed trends.

## F.2 LEARNING-SIGNAL TRAJECTORIES ON QWEN3-8B

Figures 8 and 9 report the corresponding learning-signal trajectories for Qwen3-8B. The same qualitative pattern observed with Llama3-8B also appears for Qwen3-8B. DPO drives the learning signal toward a lower regime as training proceeds, while β-DPO exhibits larger fluctuations. ε-DPO partially recovers the signal in later stages. Across both harmless-base and helpful-base, LSC-DPO maintains a higher and more stable learning signal, providing additional evidence that the proposed control rule is not specific to a single model family.

![](images/4f88327106821ff94af92f93909f2fdae1e820010f79c0e1b24266f5e6dfe7fc.jpg)  
(a) harmless-base

![](images/65484bdec360af61675119ffe6769de12b5fdde1fca2feec007f4a670b576bb7.jpg)  
(b) helpful-base

Figure 8: Raw and EMA-smoothed trajectories of the learning signal $\sigma ( - z )$ on Anthropic-HH with Qwen3-8B. Faint curves denote raw per-step values, while bold curves denote EMA-smoothed trends.  
![](images/98f06e7be60a024db8195d74dd174bfc58696c5e74166a7bc13da9716cc8af8d.jpg)  
(a) harmless-base

![](images/794c79817d7beffc2eb763452d09aa3bfa739384995f0456285389e08377aa68.jpg)  
(b) helpful-base  
Figure 9: EMA-smoothed trajectories of the learning signal $\sigma ( - z )$ on Anthropic-HH with Qwen3- 8B.

## G STABILITY UNDER DIFFERENT UPDATE RATES

We further examine the stability behavior of LSC-DPO by varying the control rate η while fixing the target scaled margin to $z ^ { * } = 0 . 4$ . Under this setting, the target learning signal is $\sigma ( - z ^ { * } ) \approx 0 . 4 0 1 3 .$ and changing η directly changes the stability-relevant product $\eta z ^ { * }$ in Proposition 1. Figures 10 and 11 show the EMA-smoothed trajectories of the learning signal $\sigma ( - z )$ on Anthropic-HH subsets. Very small values of η react slowly to deviations from the target signal. For example, when $\eta = 0 . 0 1$ the signal often drops below the target level and recovers only gradually. In contrast, moderate values such as $\eta = 0 . 3 , 0 . 5$ , and 1.0 provide more responsive correction and keep the signal close to the target after the initial adjustment phase across both model families and subsets.

With $z ^ { * } = 0 . 4 ,$ , the local stability condition $0 < \eta z ^ { * } < 2$ reduces to $\eta < 5$ . At $\eta = 5$ , which lies exactly on the boundary with $| 1 - \eta z ^ { * } | = 1$ , the local contraction is lost and the learningsignal trajectory exhibits substantially larger fluctuations. In contrast, $\eta = 8$ violates the stability condition and fails to reliably track the target signal. These results support the feedback-control interpretation of LSC-DPO: appropriately bounded control rates maintain the learning signal near the target regime, whereas overly large rates can destabilize the signal dynamics.

![](images/78bfe26a648353f689e970f85ec9f0e6d5a5574413b5f0a9404a016d1c5dd44b.jpg)  
(a) harmless-base

![](images/681513be8e2581507c78206c4e8b76044ebf569c35f5fcfa0aa4e30e041e36b5.jpg)  
(b) helpful-base

Figure 10: Learning signal $\sigma ( - z )$ trajectories of LSC-DPO under different control rates η with Llama3-8B on Anthropic-HH subsets. The dashed line marks the target signal $\sigma ( - z ^ { * } ) \approx 0 . 4 0 1 3$  
![](images/ec1f264eb3335f2e4498a8d2027163d3b36ec9a3a24c7d77735bbee74212bac0.jpg)  
(a) harmless-base

![](images/e183a8fb199d4b5e5d96c95b1f8af3980e5aa036ac9d46a145159365007c1ef7.jpg)  
(b) helpful-base  
Figure 11: Learning signal $\sigma ( - z )$ trajectories of LSC-DPO under different control rates η with Qwen3-8B on Anthropic-HH subsets. The dashed line marks the target signal $\sigma ( - z ^ { * } ) \approx 0 . 4 0 1 3$

## H ADDITIONAL MECHANISTIC VALIDATION OF LEARNING-SIGNALCONTROL

The mechanistic experiments in Section 4.3 use the same Llama3-8B SFT checkpoint and Ultra-Feedback training configuration as the main experiments. Unless otherwise specified, all training and evaluation settings are kept fixed across the compared methods.

## H.1 LEARNING SIGNAL AND REALIZED OPTIMIZATION DYNAMICS

We compute the curvature factor as $c _ { t } = \sigma ( - z _ { t } ) ( 1 { - } \sigma ( - z _ { t } ) )$ and report the gradient norm measured before gradient clipping. LSC-DPO control is activated at step 84. For visualization, solid curves show EMA-smoothed trajectories, whereas all phase-averaged statistics in Table 2 are computed from the original unsmoothed measurements.

![](images/bc25b82f8556ed2b6160a5db363eac24e08f7f34b099e861f29b1005dbb5829d.jpg)

![](images/d8602183404669e1550afed3692ec3c6c13fb4b6ce9295c48e5ce128e35a7a5f.jpg)  
Figure 12: Optimization diagnostics for DPO and LSC-DPO throughout training. Left: curvature factor. Right: pre-clipping gradient norm. The vertical dashed line marks LSC-DPO control activation at step 84.

Before control activation, DPO and LSC-DPO follow nearly identical trajectories. After activation, LSC-DPO consistently maintains a larger curvature factor, whereas DPO continues moving toward a lower-curvature regime. The pre-clipping gradient trajectories also separate after control activa tion. These full trajectories confirm that the phase-averaged differences reported in Table 2 persist throughout the controlled stage rather than arising from a small number of training steps.

## H.2 FEEDBACK VERSUS PREDETERMINED $\beta$ SCHEDULES

For the predetermined-schedule baselines in Table 3, we vary $\beta$ according to linear, cosine, and exponential schedules fixed before training. Let $t = 0 , \ldots , T - 1$ index the $\beta$ updates and $f _ { t } =$ $t / { \bar { ( T - 1 ) } }$ . All schedules start from $\beta _ { 0 }$ and end at $\beta _ { \mathrm { e n d } }$ , and are defined as

$$
\beta _ { t } ^ { \mathrm { l i n } } = \beta _ { 0 } + f _ { t } ( \beta _ { \mathrm { e n d } } - \beta _ { 0 } ) ,\tag{56}
$$

$$
\beta _ { t } ^ { \mathrm { c o s } } = \beta _ { 0 } + \frac { 1 } { 2 } \left( 1 - \cos ( \pi f _ { t } ) \right) ( \beta _ { \mathrm { e n d } } - \beta _ { 0 } ) ,\tag{57}
$$

$$
\begin{array} { r } { \beta _ { t } ^ { \mathrm { e x p } } = \exp \left( \log \beta _ { 0 } + f _ { t } ( \log \beta _ { \mathrm { e n d } } - \log \beta _ { 0 } ) \right) . } \end{array}\tag{58}
$$

All predetermined schedules use the same initial and final coefficient values as LSC-DPO, differ only in how $\beta$ evolves between these endpoints, and update the coefficient once per gathered microbatch.

## I ADDITIONAL RESULTS ON INITIALIZATION ROBUSTNESS AND SIGNAL COMPENSATION

The experiments in Section 4.4 use the same Llama3-8B, UltraFeedback, training, and evaluation settings as those in Section 4.3; only $\beta _ { 0 }$ and, where applicable, the compensation target are varied.

## I.1 INITIALIZATION SENSITIVITY: FULL LC/WR RESULTS

Table 10 reports the complete AlpacaEval 2 results for the initialization-sensitivity study, including both length-controlled (LC) win rate and standard win rate (WR). The $\beta _ { 0 } ~ = ~ 0 . 0 1$ configuration corresponds to the reference setting used in the main experiments.

Table 10: Full initialization-sensitivity results on AlpacaEval 2. We report length-controlled win rate (LC) and standard win rate (WR), both in percentage points. † denotes the reference setting.
<table><tr><td>Method</td><td colspan="2"> $\beta _ { 0 } = 0 . 0 1 ^ { \dag }$ </td><td colspan="2"> $\beta _ { 0 } = 0 . 0 5$ </td><td colspan="2"> $\beta _ { 0 } = 0 . 1 0$ </td><td colspan="2"> $\beta _ { 0 } = 0 . 3 0$ </td><td colspan="2"> $\beta _ { 0 } = 0 . 8 0$ </td></tr><tr><td></td><td>LC</td><td>WR</td><td>LC</td><td>WR</td><td>LC</td><td>WR</td><td>LC</td><td>WR</td><td>LC</td><td>WR</td></tr><tr><td>DPO</td><td>19.2</td><td>17.7</td><td>10.4</td><td>7.7</td><td>8.7</td><td>5.8</td><td>8.3</td><td>4.8</td><td>6.6</td><td>3.9</td></tr><tr><td>β-DPO</td><td>16.6</td><td>16.5</td><td>10.5</td><td>7.8</td><td>10.8</td><td>7.8</td><td>7.3</td><td>4.7</td><td>6.9</td><td>3.8</td></tr><tr><td>ε-DPO</td><td>19.6</td><td>17.6</td><td>15.6</td><td>13.2</td><td>12.9</td><td>9.8</td><td>8.8</td><td>5.9</td><td>6.9</td><td>4.5</td></tr><tr><td>LSC-DPO</td><td>22.7</td><td>21.1</td><td>19.6</td><td>18.3</td><td>19.1</td><td>17.2</td><td>15.3</td><td>14.1</td><td>15.2</td><td>13.4</td></tr></table>

## I.2 ADDITIONAL SIGNAL-BUDGET COMPENSATION RESULTS

We provide additional results for the signal-budget compensation experiments described in Section 4.4. The $\beta _ { 0 } = 0 . 0 1$ run is used as the anchor configuration, and the baseline LSC-DPO target is $z ^ { * } = 0 . 4 0 0$ . For each non-anchor initialization, the full-compensation target is determined by the signal-budget rule in Eq. 20. For $\beta _ { 0 } = 0 . 1 0$ and $\beta _ { 0 } = 0 . 8 0$ , we additionally evaluate a halfcompensation setting, obtained by applying half of the target shift from the baseline $z ^ { * } = 0$ .400 toward the corresponding full-compensation target. All other training and evaluation settings are kept unchanged.

Table 11: Detailed signal-budget compensation results on AlpacaEval 2. LC and WR denote lengthcontrolled and standard win rates, respectively.
<table><tr><td>Setting</td><td> $z ^ { * }$ </td><td>LC (%)</td><td>WR (%)</td></tr><tr><td> $\beta _ { 0 } = 0 . 0 5$ </td><td>0.359</td><td>21.5</td><td>20.1</td></tr><tr><td> $\beta _ { 0 } = 0 . 1 0 ( \mathrm { h a l f } )$ </td><td>0.374</td><td>20.7</td><td>18.9</td></tr><tr><td> $\beta _ { 0 } = 0 . 1 0 ( \mathrm { f u l l } )$ </td><td>0.348</td><td>21.1</td><td>19.5</td></tr><tr><td> $\beta _ { 0 } = 0 . 3 0$ </td><td>0.333</td><td>20.4</td><td>19.4</td></tr><tr><td> $\beta _ { 0 } = 0 . 8 0 ( \mathrm { h a l f } )$ </td><td>0.362</td><td>17.4</td><td>15.8</td></tr><tr><td> $\beta _ { 0 } = 0 . 8 0 ( \mathrm { f u l l } )$ </td><td>0.324</td><td>19.7</td><td>18.3</td></tr></table>

Compared with the corresponding uncompensated runs reported in Table 10, full compensation improves both LC and WR across all tested non-anchor initializations. For the two initializations where both half and full compensation are evaluated, half compensation yields intermediate performance, while full compensation produces larger improvements. This pattern is consistent with the intended effect of progressively correcting the initialization-induced learning-signal deficit.

To directly examine whether the target adjustment produces the intended signal-level correction, we compare the cumulative learning-signal recovery under half and full compensation. The required recovery is defined relative to the anchor run and represents the cumulative signal deficit that must be recovered to match its signal budget.

![](images/44c2908763e8d15d1a1ef7597d047bd4d09c4f7a47404b519c6600f8eb04fb7b.jpg)  
Figure 13: Cumulative learning-signal recovery under half and full signal-budget compensation for $\beta _ { 0 } = 0 . 1 0$ and $\beta _ { 0 } = 0 . 8 0$ . The dashed curves denote the recovery required to match the anchor run $( \beta _ { 0 } = 0 . 0 1 )$ , while the solid curves show the recovery achieved by half and full compensation. Full compensation closely matches or slightly exceeds the required recovery, whereas half compensation recovers approximately half of the deficit.

Figure 13 shows that the realized cumulative recovery follows the intended compensation strength. Full compensation approaches the required cumulative recovery by the end of training, whereas half compensation produces a substantially smaller correction. These results verify that the target adjustment acts on the cumulative learning signal in the intended direction. Matching the cumulative signal budget does not imply identical optimization trajectories or policies, but provides a direct mechanism-level check of the proposed compensation rule.

## J ABLATION ANALYSIS OF PARAMETERS

## J.1 TARGET LEARNING SIGNAL

Table 12 analyzes the target learning signal in LSC-DPO on AlpacaEval 2. In our update rule, the target scaled margin $\bar { z ^ { * } }$ specifies the target signal level $\sigma ( - \bar { z } ^ { * } )$ and therefore determines the learning-signal regime that the method attempts to track. We also include DPO as a reference point; its reported signal is the final stabilized value observed during training rather than a controlled target.

The results indicate that stronger performance is achieved in an intermediate target-signal region rather than at either extreme. When $z ^ { * }$ is too small, the target signal remains high and the loss stays in a relatively high-sensitivity, high-curvature region. The empirical results show that maintaining an overly high target signal does not yield the best performance. Conversely, when $z ^ { * }$ is too large, the target signal becomes small and the sigmoid loss moves toward a flatter, lower-sensitivity region. For Llama3-8B, increasing $z ^ { * }$ from 0.40 to 0.70 reduces the LC win rate from 22.7 to 19.5; for Qwen3-8B, increasing $z ^ { * }$ from 0.35 to 0.70 reduces the LC win rate from 22.1 to 13.7.

The best results are obtained at moderate target signal levels. Llama3-8B performs best at $z ^ { * } = 0 . 4 0 ,$ corresponding to $\sigma ( - z ^ { * } ) = 0 . 4 0 1 3$ , while Qwen3-8B performs best at $z ^ { * } = 0 . 3 5$ , corresponding to $\sigma ( - z ^ { * } ) = 0 . 4 1 3 4$ . Compared with DPO, the best LSC-DPO settings improve the LC win rate from 19.2 to 22.7 on $\mathrm { L l a m a } \bar { 3 } – 8 \mathrm { B }$ and from 17.2 to 22.1 on Qwen3-8B. Overall, these results suggest that LSC-DPO benefits from an intermediate target learning signal: in our experiments, the strongest settings concentrate around $\sigma ( - z ^ { * } ) \approx 0 . 4 0 \mathrm { - } \bar { 0 } . 4 1$ , where the loss retains meaningful local responsiveness without staying too close to the overly high-signal region or drifting into saturation.

Table 12: Target learning signal on AlpacaEval 2. The DPO row reports the final stabilized signal under fixed-β training. Scores are official AlpacaEval 2 point estimates; standard errors for WR/LC are typically 1.0–1.2 percentage points in our runs.
<table><tr><td>Model</td><td>2* / Method</td><td>Target signal σ(−z*)</td><td>WR (%)</td><td>LC (%)</td></tr><tr><td rowspan="8">Llama3-8B</td><td>0.20</td><td>0.4502</td><td>3.8</td><td>4.3</td></tr><tr><td>0.30</td><td>0.4256</td><td>12.8</td><td>14.7</td></tr><tr><td>0.35</td><td>0.4134</td><td>20.1</td><td>20.9</td></tr><tr><td>0.40</td><td>0.4013</td><td>21.1</td><td>22.7</td></tr><tr><td>0.45</td><td>0.3894</td><td>19.7</td><td>21.1</td></tr><tr><td>0.50</td><td>0.3775</td><td>18.6</td><td>20.7</td></tr><tr><td>0.60</td><td>0.3543</td><td>18.7</td><td>20.2</td></tr><tr><td>0.70</td><td>0.3318</td><td>17.5</td><td>19.5</td></tr><tr><td rowspan="10">Qwen3-8B</td><td>DPO</td><td>0.3429</td><td>17.7</td><td>19.2</td></tr><tr><td>0.20</td><td>0.4502</td><td>12.9</td><td>18.6</td></tr><tr><td>0.30</td><td>0.4256</td><td>15.2</td><td>21.1</td></tr><tr><td>0.35</td><td>0.4134</td><td>16.1</td><td>22.1</td></tr><tr><td>0.40</td><td>0.4013</td><td>13.5</td><td>18.9</td></tr><tr><td>0.45</td><td>0.3894</td><td>12.9</td><td>18.6</td></tr><tr><td>0.50</td><td>0.3775</td><td>12.3</td><td>17.8</td></tr><tr><td>0.60</td><td>0.3543</td><td>10.9</td><td>18.2</td></tr><tr><td>0.70</td><td>0.3318</td><td>8.8</td><td>13.7</td></tr><tr><td>DPO</td><td>0.3787</td><td>13.1</td><td>17.2</td></tr></table>

## J.2 ABLATION OF CONTROL PARAMETERS

We further study the parameters of LSC-DPO on Anthropic-HH using Llama3-8B and Qwen3-8B.   
Figures 14 and 15 vary the target scaled margin $z ^ { * }$ , the control rate η, and the start threshold τ<sub>start</sub>.

The target scaled margin $z ^ { * }$ specifies the desired learning-signal regime through $\sigma ( - z ^ { * } )$ . Moderate values of $z ^ { * }$ generally achieve stronger win rates, whereas overly large values reduce performance, consistent with the view that a small target signal moves the loss toward a lower-sensitivity regime. The control rate $\eta$ determines the responsiveness of the feedback update. Moderate values perform well, while overly large rates can degrade performance, consistent with the stability dependence on $\eta z ^ { * }$ in Proposition 1. The start threshold $\tau _ { \mathrm { s t a r t } }$ determines when the dynamic update is activated. Compared with $z ^ { * }$ and $\eta ,$ performance is less sensitive to $\tau _ { \mathrm { s t a r t } }$ within the tested range, although moderate activation thresholds tend to perform best.

Overall, these ablations suggest that LSC-DPO tolerates moderate variations in its control parameters, while benefiting from target margins and control rates that maintain a responsive learning-signal regime.

![](images/1e25a3b8fcff6c737243847e7df6c89f092e90c9a509c228723d6823ff9954f5.jpg)  
(a) Target scaled margin $z ^ { * }$

![](images/fdcb4b8d04d402510fcb4a35a8ef27b6700c0d1fd9ebc3766245afca373fb3dd.jpg)  
(b) Control rate η

![](images/f1cc4d4f3ef3efb63566a899bd1a591c8e42594f204cca58e84163d4f8f79c5f.jpg)  
(c) Start threshold $\tau _ { \mathrm { s t a r t } }$  
Figure 14: Ablation analysis of key control parameters with Llama3-8B on Anthropic-HH subsets: (a) target scaled margin $z ^ { * }$ , (b) control rate $\eta ,$ and (c) start threshold $\tau _ { \mathrm { s t a r t } }$ . The star marks the default setting used in the main experiments.

![](images/8b92789ebfe86b47634d880dee72b0ebcf5e6508d735d6a2811ad6cd223d98b5.jpg)  
(a) Target scaled margin $z ^ { * }$

![](images/b7c51ef3954e955c17997b7c8831b97e9dfbe919e42d14bfcbf8d1970d9e18b3.jpg)  
(b) Control rate η

![](images/89da31aa929ba1e1cf249732b7ccf1bc6821abd679d560dfa1b5ca3abcf1fb8e.jpg)  
(c) Start threshold $\tau _ { \mathrm { s t a r t } }$  
Figure 15: Ablation analysis of key control parameters with Qwen3-8B on Anthropic-HH subsets: (a) target scaled margin $z ^ { * }$ , (b) control rate $\eta ,$ and (c) start threshold $\tau _ { \mathrm { s t a r t } }$ . The star marks the default setting used in the main experiments.

## K DOWNSTREAM EVALUATION

Following the downstream evaluation tasks used in SimPO (Meng et al., 2024) and ε-DPO (Lee et al., 2025), we further evaluate whether preference optimization preserves general task capabilities beyond open-ended chatbot benchmarks. We consider MMLU (Hendrycks et al., 2020), ARC-Challenge (Clark et al., 2018), HellaSwag (Zellers et al., 2019), TruthfulQA (Lin et al., 2022), WinoGrande (Sakaguchi et al., 2021), and GSM8K (Cobbe et al., 2021), covering knowledge, reading comprehension, commonsense reasoning, truthfulness, and mathematical reasoning. Tables 13 and 14 report the results for Llama3-8B and Qwen3-8B, respectively.

Overall, LSC-DPO maintains downstream task performance at a level comparable to strong preference-optimization baselines and does not exhibit a consistent degradation relative to SFT. The effect of preference optimization varies across tasks and model families, consistent with prior observations that downstream task changes can depend on the pretrained model, preference data, and optimization objective. Notably, on GSM8K, LSC-DPO does not show a score drop relative to the SFT initialization on either model family, suggesting that the alignment improvements do not come from a substantial loss of mathematical reasoning capability. As an additional reproducibility check, we also evaluate the released SimPO checkpoints on the same set of downstream tasks under the same evaluation protocol and obtain highly similar results. This consistency strengthens the reliability of our capability-preservation conclusion. We therefore interpret these downstream results mainly as a capability-preservation check rather than as the primary evidence for alignment improvement.

Table 13: Downstream capability evaluation on Llama3-8B. Scores are reported as percentages. Avg. denotes the mean over the displayed task scores.
<table><tr><td>Method</td><td>MMLU</td><td>ARC-C</td><td>HellaSwag</td><td>TruthfulQA</td><td>WinoGrande</td><td>GSM8K</td><td>Avg.</td></tr><tr><td>SFT</td><td>62.54</td><td>60.24</td><td>81.30</td><td>45.50</td><td>74.43</td><td>49.20</td><td>62.20</td></tr><tr><td>DPO</td><td>63.15</td><td>64.51</td><td>83.14</td><td>53.12</td><td>75.14</td><td>54.97</td><td>65.67</td></tr><tr><td>β-DPO</td><td>62.92</td><td>63.99</td><td>83.04</td><td>51.37</td><td>75.14</td><td>53.45</td><td>64.99</td></tr><tr><td>ε-DPO</td><td>62.98</td><td>64.33</td><td>83.29</td><td>52.94</td><td>75.14</td><td>53.15</td><td>65.30</td></tr><tr><td>IPO</td><td>62.96</td><td>62.63</td><td>81.68</td><td>49.02</td><td>73.88</td><td>54.36</td><td>64.09</td></tr><tr><td>CPO</td><td>62.73</td><td>60.32</td><td>81.44</td><td>52.97</td><td>75.06</td><td>48.22</td><td>63.46</td></tr><tr><td>KTO</td><td>62.88</td><td>64.51</td><td>83.39</td><td>53.93</td><td>75.85</td><td>53.37</td><td>65.65</td></tr><tr><td>ORPO</td><td>62.80</td><td>55.29</td><td>80.70</td><td>47.72</td><td>74.03</td><td>44.35</td><td>60.82</td></tr><tr><td>SLiC-HF</td><td>62.69</td><td>57.85</td><td>80.57</td><td>50.82</td><td>74.35</td><td>46.47</td><td>62.13</td></tr><tr><td>R-DPO</td><td>62.41</td><td>61.95</td><td>80.91</td><td>43.74</td><td>74.35</td><td>39.58</td><td>60.49</td></tr><tr><td>SimPO</td><td>62.85</td><td>65.78</td><td>83.12</td><td>58.48</td><td>74.25</td><td>52.54</td><td>66.17</td></tr><tr><td>LSC-DPO</td><td>63.02</td><td>65.25</td><td>83.34</td><td>52.57</td><td>74.82</td><td>53.98</td><td>65.50</td></tr></table>

Table 14: Downstream capability evaluation on Qwen3-8B. Scores are reported as percentages. Avg. denotes the mean over the displayed task scores.
<table><tr><td>Method</td><td>MMLU</td><td>ARC-C</td><td>HellaSwag</td><td>TruthfulQA</td><td>WinoGrande</td><td>GSM8K</td><td>Avg.</td></tr><tr><td>SFT</td><td>75.99</td><td>67.49</td><td>80.17</td><td>53.87</td><td>75.85</td><td>82.18</td><td>72.59</td></tr><tr><td>DPO</td><td>76.54</td><td>68.69</td><td>81.30</td><td>57.39</td><td>76.01</td><td>84.46</td><td>74.06</td></tr><tr><td>β-DPO</td><td>76.48</td><td>68.77</td><td>81.13</td><td>56.40</td><td>76.72</td><td>83.40</td><td>73.82</td></tr><tr><td>ε-DPO</td><td>76.63</td><td>68.34</td><td>81.44</td><td>57.57</td><td>76.01</td><td>84.53</td><td>74.09</td></tr><tr><td>IPO</td><td>76.48</td><td>67.83</td><td>80.92</td><td>56.30</td><td>76.09</td><td>84.15</td><td>73.63</td></tr><tr><td>CPO</td><td>75.63</td><td>68.09</td><td>80.03</td><td>56.72</td><td>75.61</td><td>83.55</td><td>73.27</td></tr><tr><td>KTO</td><td>76.41</td><td>68.77</td><td>81.12</td><td>57.37</td><td>76.09</td><td>84.99</td><td>74.12</td></tr><tr><td>ORPO</td><td>75.84</td><td>66.38</td><td>79.66</td><td>53.83</td><td>75.22</td><td>67.48</td><td>69.73</td></tr><tr><td>SLiC-HF</td><td>75.01</td><td>67.06</td><td>79.33</td><td>55.03</td><td>74.51</td><td>79.98</td><td>71.82</td></tr><tr><td>R-DPO</td><td>75.92</td><td>68.00</td><td>80.60</td><td>54.07</td><td>75.93</td><td>84.23</td><td>73.13</td></tr><tr><td>SimPO</td><td>76.37</td><td>68.69</td><td>81.24</td><td>56.73</td><td>76.80</td><td>84.84</td><td>74.11</td></tr><tr><td>LSC-DPO</td><td>76.50</td><td>68.52</td><td>81.38</td><td>57.50</td><td>76.01</td><td>84.53</td><td>74.07</td></tr></table>