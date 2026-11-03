# When Prediction Shapes the Future: A BlueSky Agenda for Reflexive Time Series Analysis

Tao Xie<sup>1</sup>, Yiqun Zhang<sup>2,\*</sup>, Yiu-ming Cheung<sup>3</sup>

<sup>1</sup>School of Automation, Guangdong University of Technology

<sup>2</sup>School of Future Technology, South China University of Technology

<sup>3</sup>Department of Computer Science, Hong Kong Baptist University

3123001085@mail2.gdut.edu.cn, yqzhangzyq@gmail.com, ymc@comp.hkbu.edu.hk

Abstract—Time-series models are increasingly deployed in operational systems (e.g., inventory management and clinical care) where predictions trigger interventions that alter future observations, producing records we term AI-shaped trajectories. This sequence-prediction coupling deepens as such systems evolve into agents that autonomously predict, act, and remember across long horizons. Such AI-shaped trajectories support immediate performance monitoring, but turn ambiguous once reused as passive data. An algorithm may then misread a predictionprevented stockout as naturally stable demand, or an averted medical crisis as a false alarm. This ambiguity then propagates into everything built on the record, from retraining and agent memory to downstream predictive tasks. As a result, systems may internalize these intervention effects as natural phenomena, drifting further from the truth at every closedloop cycle. In this Blue Sky paper, we advocate an agenda for reflexive time-series analysis that shifts the analytical object from the intervention-agnostic sequence to the prediction–action– observation loop. The agenda requires closed-loop provenance to track the operational context (e.g., predictions, policies, and actions), sequence identification to estimate how those actions affect future values under explicit causal assumptions, and closedloop evaluation to separate intrinsic process prediction from policy-driven effects. This foundation is essential for integrating autonomous agents into such loops at scale, enabling the joint learning of intervention effects and intrinsic dynamics for robust temporal representations.

Index Terms—Time series mining, performative prediction, closed-loop evaluation, representation learning, forecasting

## I. INTRODUCTION

Time series are predominantly conceptualized as passive records of an exogenous process. This assumption holds primarily for non-interventional settings, where model outputs do not feed back into the underlying data-generating mechanism. However, this premise becomes untenable when deployed predictive systems drive an operational policy [1], [2], as actions based on predictions can alter subsequent observations. Prior work on performative prediction [3], [4] shows that deployed predictions can influence the future target distributions. Moving beyond static distribution shifts, we focus on the sequential artifacts continuously inscribed by these interventions, a phenomenon we term the AI-shaped trajectory. Once reused for model retraining, benchmarking, or agent memory, these AI-shaped trajectories carry both the underlying process dynamics and the effects of the policy under which they were generated. As systems become increasingly autonomous [5], this reuse renders the pipeline self-referential, thereby conflating action effects with the intrinsic dynamics of the underlying process.

![](images/1b5ad6f5a569c1db8f1c66df170d3ed5d80cd04e7c2c01797bc689c34d13a1a5.jpg)  
Fig. 1. Reflexive generation of an AI-shaped trajectory. A forecast-driven action alters subsequent observations, which can re-enter later training, memory, or evaluation. Repeated feedback can progressively separate the observed trajectory from the underlying process dynamics.

In inventory control, a demand-forecasting model drives replenishment [6], [7]. When a predicted stockout triggers early restocking, the shortage is prevented, and recorded sales remain flat. This observation is causally confounded: it might reflect intrinsically stable demand, or volatile demand successfully mitigated by the intervention. Clinical monitoring exhibits a similar pathology [8]. A model flags a patient for deterioration, prompting preventive care that stabilizes the patient. Consequently, subsequent evaluations may misinterpret the initial alert as a false alarm. A similar closed loop arises in smart-grid demand response, where short-term load forecasts can determine dynamic electricity prices or loadshifting decisions, which can in turn influence subsequent consumption and make the realized load partly dependent on the forecasting-and-control policy [9]. Across these settings, the observational data records an outcome shaped by the intervention, thereby masking the underlying process dynamics. This confounding becomes particularly critical in autonomous agents that forecast, act, and retrain on the records their own actions produce. Without external calibration mechanisms, the causal ambiguity is easily overlooked, causing the system to continuously internalize its own policy effects as inherent natural phenomena.

Standard action logs document interventions but exclude counterfactual outcomes [10], [11]. Consequently, the AIshaped trajectory entangles the intervention effect with the underlying process dynamics, rendering the two components difficult to disentangle from observed values alone [12]. This mixture spreads through routine downstream applications, effectively corrupting three fundamental phases: 1) Model retraining on these trajectories injects past policy effects into the newly learned dynamics, exacerbating the structural bias with each successive cycle [13]–[15]; 2) When repurposed for performance evaluation, benchmarking heavily rewards agreement with policy-constrained futures, mistaking selffulfilling prophecies for genuine predictive capability [16]; 3) Utilizing these trajectories as agent memory forces the system to internalize its own interventions as objective environmental truths [17]. All three phases operate on the flawed assumption of an undisturbed data-generating process. However, because the AI-shaped trajectory preserves only the final outcome and omits the causal link between action and effect, this assumption catastrophically fails, and standard predictive approaches inherently struggle to overcome this missing link. Therefore, extracting genuine process dynamics from such contaminated records necessitates a dedicated analytical framework.

In this Blue Sky paper, we propose an agenda for reflexive time-series analysis that explicitly acknowledges the system’s active role in shaping its own future data. This agenda shifts the primary unit of analysis from the intervention-agnostic sequence to the dynamic, closed loop of prediction, action, and observation. The agenda unfolds across three essential stages: 1) Closed-loop provenance traces and records the operational context, encompassing the deployed prediction, the governing policy, and the executed action; 2) Building upon this foundation, sequence identification integrates explicit causal assumptions to estimate how these interventions perturb future sequence values; 3) Closed-loop evaluation leverages these counterfactual estimates to disentangle intrinsic process predictability from policy-driven artifacts. As autonomous agents increasingly govern these operational loops at scale, this threestage foundation provides the external calibration required to break the self-referential cycle. By jointly learning intervention effects and intrinsic dynamics, this agenda produces robust temporal representations that maintain their scientific validity even as systems repeatedly reuse them for retraining, benchmarking, and agentic memory.

The remainder of this paper is organized as follows: Section II formalizes the analytical object and examines its urgency under current deployment trends. Section III develops the three-stage agenda of loop provenance, sequential identification, and closed-loop evaluation. Section IV establishes criteria for methodological progress, and Section V extends this framework to broader data mining applications.

## II. THE REFLEXIVE PARADIGM: FROM PASSIVE RECORDS TO ACTIVE LOOPS

Time-series mining has long prioritized predicting future sequences over analyzing the temporal records left behind by deployed models. An AI-shaped trajectory should therefore be treated not as an intervention-agnostic sequence, but as empirical evidence of the dynamic loop that produced it. Reflexive time series analysis therefore shifts the analytical unit from the intervention-agnostic sequence $X _ { 1 : T }$ to the underlying prediction-action-observation mechanism. Standard forecasting frameworks inherently lack the capacity to recover this mechanism, demanding new infrastructure built on three capabilities: closed-loop provenance, sequential identification, and rigorous evaluation.

The phenomenon of predictions reshaping their target environments is well documented across disciplines, spanning self-fulfilling prophecies, Goodhart’s law, and performative prediction [3], [18]. The gap lies in how time-series mining formalizes the resulting sequence. Current practice discards this causal context: benchmarks expose recorded values while omitting the prediction, policy, action, operational context, and counterfactual uncertainty needed for deployment interpretation. Without this information, the record is difficult to interpret because standard evaluation relies on an assumption that deployment can violate. This assumption is causal rather than statistical: although a valid forecast should correlate with its target, evaluation typically assumes that the target itself is not altered by the forecast [3], [19]. Under operational policies, this assumption may fail, rendering outcomes policydependent and motivating a dedicated analytical formulation and evaluation infrastructure.

Two structural shifts necessitate this paradigm. First, the prediction-to-action pathway is increasingly designed rather than incidental. Decision-focused and control-oriented methods now train forecasts to drive actions rather than serve as terminal outputs [1], [20], [21], foundation models lower barriers to deploying such forecasters across domains [22], [23], and agentic systems make the loop persistent by executing actions and storing outcomes in memory [5]. Consequently, the passive-record assumption fails at scale. Second, the empirical evidence of this feedback is broad, spanning modelinduced distribution shifts [3], [24], bias amplification in data pipelines [15], feedback in deployed recommenders and clinical risk scores [25], [26], and formal performativity in timeseries forecasting [4], [27]. Although existing performativityaware methods anticipate induced shifts to preserve accuracy, predictive accuracy alone cannot establish how the underlying loop should be recorded, identified, and evaluated. Without this causal context, downstream analyses may conflate intervention effects with underlying process dynamics. Maintaining explicit links among forecasts, policies, and actions preserves the evidence needed to examine this ambiguity under causal assumptions. Once these links are lost, the intervention context becomes unrecoverable from the sequence alone, which complicates subsequent reuse and motivates the proposed agenda.

Formalizing this paradigm requires explicit definition of the loop’s causal mechanics. In a minimal forecasting setting:

$$
\widehat Y _ { t } = f _ { \theta } ( X _ { \le t } ) , \qquad A _ { t } = \pi ( \widehat Y _ { t } ) , \qquad Y _ { t + h } ^ { o b s } = Y _ { t + h } ( A _ { t } ) .\tag{1}
$$

Here, the model $f _ { \theta }$ predicts $\widehat { Y } _ { t }$ from history, policy π triggers action $A _ { t } ,$ and the outcome $Y _ { t + h } ^ { o b s }$ materializes strictly under this intervention. This continuous propagation distinguishes reflexive analysis from standard performative prediction [3]. Rather than merely shifting a global data distribution, the action $A _ { t }$ alters the underlying state space, dynamically distributing intervention effects across the entire subsequent trajectory. This creates a severe sequential identification problem, where naturally stable processes and action-suppressed volatile processes yield mathematically identical empirical observations. Since standard action logs cannot reveal counterfactual outcomes [10], [11], resolving this ambiguity requires explicit structural assumptions, such as transition models or policy variation. Our agenda aims to formalize this causal disentanglement, ensuring models are evaluated against objective realities rather than artificial futures they helped construct.

## III. A THREE-TIER AGENDA FOR REFLEXIVE ANALYSIS

Analyzing AI-shaped trajectories requires a strict dependency chain: recording the loop, identifying its causal effects, and evaluating predictive claims under closed-loop conditions. The objective is to build infrastructure for prediction–action– observation loops, rather than appending new metrics to passive benchmarks.

Upstream: recording the loop. To make forecast-mediated action part of the temporal record itself. A minimal provenance schema should store the value sequence along with the deployed model or model version, the forecast, the policy mapping, the action taken, the operational context, and the subsequent observed outcome [28], [29]:

$$
( t , X _ { \leq t } , f _ { \theta } , \widehat { Y } _ { t } , \pi , A _ { t } , H _ { t } , Y _ { t + h } ^ { o b s } ) .\tag{2}
$$

In this schema, $H _ { t }$ includes human overrides, constraints, or operational information available at decision time. In the inventory example, a complete record stores the sales series together with the forecast or stockout-risk estimate, the replenishment rule that used it, and the order placed under that rule. Such a record cannot identify action effects by itself; it can only make the intervention visible to downstream analysis. The challenge remains that forecasting, decision-making, and logging are often separated across software systems and organizational boundaries [30]. Because policies change and model versions update between cycles [31], analysts who do not record the links among model version, forecast, policy, action, and outcome observe only policy-dependent outcomes without the mechanisms that produced them. This raises the first open problem (O1): to define a minimal loop provenance standard that is expressive enough to support subsequent identification yet simple enough to be attached to deployed temporal data.

Midstream: identifying action effects under temporal dependence. To identify action effects by modeling actions as components of the temporal state transition rather than as external metadata [12], [32]. This requires estimating how a sequence would evolve under alternative actions or no action [10], [11], [33]. In the inventory example, this asks how the sales series would have evolved under a different order or no order. In a smart-grid demand-response system, the analogous question is how the load trajectory would have evolved under a different price signal or without forecastdriven load shifting [9]. Estimating such counterfactual evolution requires assumptions about the transition mechanism, deliberate policy variation, experimental interventions, simulators, or other sources of counterfactual information. Existing counterfactual and sequential decision methods provide foundations for action-dependent outcomes [34], [35], but closedloop temporal records add a propagation problem. The timeseries difficulty is that an intervention propagates over time rather than affecting only an immediate instance. An actionconditioned model must represent both the direct effect of the action and its delayed effect on future states; otherwise, evaluation may reward models that fit immediate outcomes while missing feedback-driven state changes. The midstream open problem (O2) is to connect counterfactual estimation, state-transition modeling, and data conditions that make action effects estimable rather than merely logged.

![](images/c7f4b573e07c5ec1791800e533c04adce699544b285d0d55de679443b6b4b4a9.jpg)  
Fig. 2. A three-tier agenda for analyzing AI-shaped trajectories. Provenance records the loop, identification estimates action effects, and evaluation separates predictive skill from self-fulfilling forecasts. Each tier builds on the one before it, and each ends in an open problem, numbered O1 to O3, marking a direction for future work.

Downstream: evaluating under a closed loop. To design protocols that distinguish predictive skill from agreement with outcomes shaped by the forecast-policy system. In a closed loop, the recorded future may no longer be a neutral target, because it can be a policy-dependent outcome shaped by the forecast-policy system. Low error can have multiple interpretations. It may indicate that the model captured the intrinsic process dynamics, or that the system steered the process toward a state closer to the prediction. Conversely, an accurate risk forecast can appear inaccurate when the action it triggers prevents the event against which the forecast is later evaluated (Section II). Evaluation should therefore report the model, policy, action, and context that produced the observed target, not only the prediction error measured against that target. Performative prediction provides a starting point for analyzing model-induced distribution changes [3], [19], and feedback detection identifies predictors that shape their own inputs [36], but closed-loop time-series evaluation must additionally account for delayed temporal propagation and targets produced under policy intervention. The downstream open problem (O3) is to extend benchmark design beyond agreement with the recorded future [16], [37], defining metrics and protocols that separate skill at forecasting intrinsic process dynamics, policy effects, and self-fulfilling forecasts.

Across the tiers. The three tiers are interdependent: provenance identifies which interventions shaped the record, sequential assumptions support counterfactual estimation, and closedloop evaluation distinguishes predictive skill from policyshaped outcomes. This dependency becomes more important in agentic systems, where forecasts, actions, and stored outcomes are coupled and repeatedly reused for training, evaluation, or memory.

## IV. OBJECTIVES OF REFLEXIVE TIME SERIES ANALYSIS

The previous section set out the tasks and open problems of the agenda. This section describes the end state those tasks aim at, that is, what a successful paradigm looks like in practice. Success here extends beyond improving passive forecasting accuracy to formalizing deployed records as artifacts of prediction–action–observation loops. This end state rests on four objectives. The first three carry the agenda over to the deployed setting, and the fourth is new.

Routine loop provenance. Datasets and benchmarks must encode the generative loop alongside the time series. Recording the model, forecast, policy, action, and context establishes the structural prerequisite for identifying intervention effects.

Estimable action effects. Counterfactual claims about how a trajectory evolves should rest on explicit assumptions, such as transition models or policy variation. Evaluations can then attribute performance gains to specific causal mechanisms rather than conflating them with intrinsic process dynamics.

Disentangled evaluation. Protocols should isolate genuine predictive skill from policy-dependent outcomes. Metrics should then distinguish whether reduced error reflects accurate intrinsic modeling or a self-fulfilling forecast. An averted risk should be read as a successful intervention rather than an apparent prediction failure.

Auditable agentic memory. As systems archive outcomes as reusable experience, their memory should remain auditable. An auditable memory preserves loop provenance and classifies each observation as an exogenous process, an intervention response, or an unresolved mixture.

Underlying these four objectives is a single structural shift, one that treats deployment as an integral part of the datagenerating mechanism rather than an external disturbance.

## V. GENERALIZING REFLEXIVITY: BEYOND FORECASTING TO AGENTIC SYSTEMS

The mechanism that produces AI-shaped trajectories also links this agenda to broader data mining frontiers, extending concerns from interpreting the temporal evidence behind individual predictions [38] to tracing how deployed models shape the records they subsequently consume [39]. Recommender systems shape exposure and then learn from the resulting clicks or purchases [25], [40]. Models trained on synthetic or model-filtered data can amplify biases already present in the selected records [14], [15]. Similarly, clinical systems updated after deployment create feedback once predictions influence care and subsequent outcomes [13], [41]. In each case, the record is produced after a model or policy has participated in the process it represents. Time series is distinct because interventions enter an evolving state where, unlike recommendation feedback localized to items or interactions, their effects propagate along the subsequent trajectory.

This framing clarifies the role of future agentic temporal reasoning systems [42], whose lasting outputs are themselves stored as temporal records. An agent does not merely read such a record once; it stores it in memory, reusing it across many later decisions. Without loop provenance, each reuse feeds a policy-dependent outcome back into the next forecast as an independent exogenous observation. Consequently, the contamination described in the introduction grows cumulatively with each cycle, rather than remaining a fixed error. Conversely, when recorded and evaluated under the proposed agenda, these same records support reliable audit, adaptation, and reuse. The frontier is thus a specific mandate that timeseries mining should learn to analyze records generated under systems that autonomously predict, act, and remember.

## VI. CONCLUSION

This paper establishes that the traditional passive-record assumption could be fundamentally invalidated when deployed forecasts dictate operational actions. Such coupling continuously generates policy-dependent outcomes that we conceptualize as AI-shaped trajectory. Properly interpreting these contaminated records necessitates treating the dynamic predictionaction-observation loop as a primary analytical object, thereby enabling the separation of intervention effects from intrinsic process dynamics. Crucially, as temporal systems increasingly predict, act, and remember, standard accuracy metrics may fail to meaningfully evaluate a forecast against a future it actively helped to create. To overcome this methodological blind spot, we urge the time-series mining community to develop new data provenance standards, causal identification techniques, and closed-loop benchmarking protocols. Although deployed forecasting serves as the minimal instance of this phenomenon, the rapid transition toward autonomous temporal agents makes this reflexive agenda necessary. Future investigations should therefore move beyond isolated prediction–action effects toward understanding how such effects accumulate through repeated retraining, memory reuse, and interactions among adaptive agents. Ultimately, the illusion of passive observation is no longer tenable. The future of data mining lies in modeling how AI systems recursively shape their own empirical reality.

## REFERENCES

[1] J. Mandi, J. Kotary, S. Berden, M. Mulamba, V. Bucarey, T. Guns, and F. Fioretto, “Decision-focused learning: Foundations, state of the art,

benchmark and future opportunities,” Journal of Artificial Intelligence Research, vol. 80, pp. 1623–1701, 2024.

[2] B. Wilder, B. Dilkina, and M. Tambe, “Melding the data-decisions pipeline: Decision-focused learning for combinatorial optimization,” in Proceedings of the AAAI Conference on Artificial Intelligence, 2019, pp. 1658–1665.

[3] J. C. Perdomo, T. Zrnic, C. Mendler-Dünner, and M. Hardt, “Performative prediction,” in Proceedings of the 37th International Conference on Machine Learning, 2020, pp. 7599–7609.

[4] Z. Zhao, H. Liu, A. Rodríguez, and B. A. Prakash, “Performative timeseries forecasting,” in Proceedings ofthe 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 2025, pp. 3968–3979.

[5] M. Cheng, X. Tao, Q. Liu, Z. Guo, and E. Chen, “Position: Beyond model-centric prediction—agentic time series forecasting,” arXiv:2602.01776, 2026.

[6] S. Makridakis, E. Spiliotis, and V. Assimakopoulos, “The M5 accuracy competition: Results, findings and conclusions,” International Journal of Forecasting, vol. 38, pp. 1346–1364, 2022.

[7] A. N. Elmachtoub and P. Grigas, “Smart “predict, then optimize”,” Management Science, vol. 68, pp. 9–26, 2022.

[8] G. Y. E. Kim, C. K. Corbin, F. Grolleau, M. T. M. Baiocchi, and J. H. Chen, “Monitoring strategies for continuous evaluation of deployed clinical prediction models,” Journal of Biomedical Informatics, vol. 168, p. 104854, 2025.

[9] B. Palaniyappan and S. K. Ramu, “Optimized lstm-based electric power consumption forecasting for dynamic electricity pricing in demand response scheme of smart grid,” Results in Engineering, vol. 25, p. 104356, 2025.

[10] I. Bica, A. M. Alaa, J. Jordon, and M. van der Schaar, “Estimating counterfactual treatment outcomes over time through adversarially balanced representations,” in International Conference on Learning Representations, 2020.

[11] V. Melnychuk, D. Frauen, and S. Feuerriegel, “Causal transformer for estimating counterfactual outcomes,” in Proceedings of the 39th International Conference on Machine Learning, 2022, pp. 15 293– 15 329.

[12] J. Runge, A. Gerhardus, G. Varando, V. Eyring, and G. Camps-Valls, “Causal inference for time series,” Nature Reviews Earth & Environment, vol. 4, pp. 487–505, 2023.

[13] G. Adam, C.-H. K. Chang, B. Haibe-Kains, and A. Goldenberg, “Error amplification when updating deployed machine learning models,” in Proceedings of the 7th Machine Learning for Healthcare Conference, 2022, pp. 715–740.

[14] R. Taori and T. Hashimoto, “Data feedback loops: Model-driven amplification of dataset biases,” in Proceedings of the 40th International Conference on Machine Learning, 2023, pp. 33 883–33 920.

[15] S. Wyllie, I. Shumailov, and N. Papernot, “Fairness feedback loops: Training on synthetic data amplifies bias,” arXiv:2403.07857, 2024.

[16] Z. Qiao, S. Pan, A. Wang, V. Zhukova, Y. Liu, X. Jiang, Q. Wen, M. Long, M. Jin, and C. Liu, “It’s TIME: Towards the next generation of time series forecasting benchmarks,” arXiv:2602.12147, 2026.

[17] Z. Xiong, Y. Lin, W. Xie, P. He, J. Tang, H. Lakkaraju, and Z. Xiang, “How memory management impacts LLM agents: An empirical study of experience-following behavior,” arXiv:2505.16067, 2025.

[18] O. C. King and M. Mertens, “Self-fulfilling prophecy in practical and automated prediction,” Ethical Theory and Moral Practice, vol. 26, pp. 127–152, 2023.

[19] M. Hardt and C. Mendler-Dünner, “Performative prediction: Past and future,” arXiv:2310.16608, 2023.

[20] M. Beichter, N. Friederich, J. Pinter, D. Werling, K. Phipps, S. Beichter, O. Neumann, R. Mikut, V. Hagenmeyer, and B. Heidrich, “Decisionfocused fine-tuning of time series foundation models for dispatchable feeder optimization,” Energy and AI, vol. 21, p. 100533, 2025.

[21] J. B. Rawlings, D. Q. Mayne, and M. M. Diehl, Model Predictive Control: Theory, Computation, and Design, 2nd ed. Nob Hill Publishing, 2017.

[22] A. F. Ansari, L. Stella, A. C. Turkmen, X. Zhang, P. Mercado, H. Shen, O. Shchur, S. S. Rangapuram, S. Pineda Arango, S. Kapoor, J. Zschiegner, D. C. Maddix, H. Wang, M. W. Mahoney, K. Torkkola, A. G. Wilson, M. Bohlke-Schneider, and B. Wang, “Chronos: Learning the language of time series,” Transactions on Machine Learning Research, 2024.

[23] M. Goswami, K. Szafer, A. Choudhry, Y. Cai, S. Li, and A. Dubrawski, “MOMENT: A family of open time-series foundation models,” in

Proceedings of the 41st International Conference on Machine Learning, 2024, pp. 16 115–16 152.

[24] T. Kehrenberg, J. Sanguino, J. A. Lozano, and N. Quadrianto, “Dissecting performative prediction: A comprehensive survey,” arXiv:2602.10176, 2026.

[25] M. Mansoury, H. Abdollahpouri, M. Pechenizkiy, B. Mobasher, and R. Burke, “Feedback loop and bias amplification in recommender systems,” in Proceedings of the 29th ACM International Conference on Information & Knowledge Management, 2020, pp. 2145–2148.

[26] J. T. Rosenthal, A. Beecy, and M. R. Sabuncu, “Rethinking clinical trials for medical AI with dynamic deployments of adaptive systems,” npj Digital Medicine, vol. 8, p. 252, 2025.

[27] R. Bhati, J. Jones, D. Langelier, A. Reiman, J. Greenland, K. Campbell, and A. Durand, “Performative prediction in time series: A case study,” in NeurIPS 2022 Workshop on Learning from Time Series for Health, 2022.

[28] T. Gebru, J. Morgenstern, B. Vecchione, J. W. Vaughan, H. Wallach, H. Daumé III, and K. Crawford, “Datasheets for datasets,” Communications of the ACM, vol. 64, pp. 86–92, 2021.

[29] M. Pushkarna, A. Zaldivar, and O. Kjartansson, “Data cards: Purposeful and transparent dataset documentation for responsible AI,” in Proceedings of the 2022 ACM Conference on Fairness, Accountability, and Transparency, 2022, pp. 1776–1826.

[30] N. Sambasivan, S. Kapania, H. Highfill, D. Akrong, P. Paritosh, and L. M. Aroyo, ““everyone wants to do the model work, not the data work”: Data cascades in high-stakes AI,” in Proceedings of the 2021 CHI Conference on Human Factors in Computing Systems, 2021, pp. 1–15.

[31] M. Mitchell, S. Wu, A. Zaldivar, P. Barnes, L. Vasserman, B. Hutchinson, E. Spitzer, I. D. Raji, and T. Gebru, “Model cards for model reporting,” in Proceedings ofthe Conference on Fairness, Accountability, and Transparency, 2019, pp. 220–229.

[32] L. Ljung, System Identification: Theory for the User, 2nd ed. Upper Saddle River, NJ: Prentice Hall PTR, 1999.

[33] M. A. Hernán and J. M. Robins, Causal Inference: What If. Boca Raton: Chapman & Hall/CRC, 2020.

[34] A. Swaminathan and T. Joachims, “Counterfactual risk minimization: Learning from logged bandit feedback,” in Proceedings of the 32nd International Conference on Machine Learning, 2015, pp. 814–823.

[35] S. Levine, A. Kumar, G. Tucker, and J. Fu, “Offline reinforcement learning: Tutorial, review, and perspectives on open problems,” arXiv:2005.01643, 2020.

[36] S. Wager, N. Chamandy, O. Muralidharan, and A. Najmi, “Feedback detection for live predictors,” in Advances in Neural Information Processing Systems, 2014, pp. 3428–3436.

[37] W. Mu, F. Yan, A. Mumssen, M. Eisenberg, and A. Rodríguez, “Benchmarking counterfactual prediction in epidemic time series with time-varying interventions,” in Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 2026.

[38] T. Xie, Z. Tan, H. Xiao, M. Li, Y. Zhang, Y. Lu, C. Yang, and Y.-m. Cheung, “AnchorMoE: Interpretable time series classification via anchor-routed moe,” in Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2, 2026, pp. 5720–5731.

[39] N. Pagan, J. Baumann, E. Elokda, G. De Pasquale, S. Bolognani, and A. Hannák, “A classification of feedback loops and their relation to biases in automated decision-making systems,” in Proceedings of the 3rd ACM Conference on Equity and Access in Algorithms, Mechanisms, and Optimization, 2023, pp. 1–14.

[40] A. Klimashevskaia, D. Jannach, M. Elahi, and C. Trattner, “A survey on popularity bias in recommender systems,” User Modeling and User-Adapted Interaction, vol. 34, pp. 1777–1834, 2024.

[41] D. R. Balcarcel, S. D. Mehta, C. G. Dixon, C. Z. Woods-Hill, E. C. Goligher, W. A. C. van Amsterdam, and N. Yehya, “Feedback loops in intensive care unit prognostic models: An under-recognised threat to clinical validity,” The Lancet Digital Health, vol. 7, p. 100880, 2025.

[42] C. Chang, Y. Shi, D. Cao, W. Yang, J. Hwang, H. Wang, J. Pang, W. Wang, Y. Liu, W.-C. Peng, and T.-F. Chen, “A survey of reasoning and agentic systems in time series with large language models,” arXiv:2509.11575, 2025.