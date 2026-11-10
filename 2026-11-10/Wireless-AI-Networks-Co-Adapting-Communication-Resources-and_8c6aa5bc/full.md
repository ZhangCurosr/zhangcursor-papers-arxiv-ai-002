# Wireless AI Networks: Co-Adapting Communication Resources and Distributed Intelligence

Ziqiang Ye, Student Member, IEEE, Yue Xiao, Member, IEEE, Qiyan Zhan,

Yulan Gao, Member, IEEE, Xianfu Lei, Member, IEEE, and Vladimir Poulkov, Senior Member, IEEE

Abstract—Distributed artificial intelligence (AI) increasingly relies on wireless networks, where communication and computing resources shape how AI tasks are configured and executed. However, resource availability alone does not determine how effectively these tasks can be supported, since different information selections and transmission strategies may yield different AI performance under the same resource constraints. To address this challenge, this article presents Wireless AI Networks as a unified design perspective for bidirectional adaptation between wireless communication and distributed intelligence. Specifically, an AI-executable resource budget characterizes candidate AI configurations, while task-level information value guides their selection and wireless resource allocation. Building on these principles, a wireless-AI co-adaptation framework coordinates resource abstraction, joint configuration and allocation, and execution feedback. Two complementary experiments illustrate this perspective: update selection and statistical correction improve federated learning under identical model-update payload budgets, while sensitivity-aware power allocation reduces model accuracy degradation under the same average transmit-power constraint. Together, these results highlight the importance of coordinating resource feasibility and task-level information value in distributed AI execution.

Index Terms—Wireless AI networks, distributed AI, wireless-AI co-adaptation, resource-aware AI, federated learning.

## I. INTRODUCTION

Artificial intelligence (AI) is increasingly deployed across mobile devices, edge servers, connected vehicles, robots, and other networked agents [1]. As a result, AI training and inference increasingly depend on information exchange among distributed nodes. Federated learning, collaborative inference, foundation-model services, and multi-agent systems rely on wireless connectivity to exchange model updates, intermediate features, and task-related information [2]. In these settings, variations in wireless connectivity and computing resources influence both information delivery and AI execution. For example, limited uplink capacity may restrict the model updates available for federated aggregation, while communication latency and edge computing capacity may determine the feasible partition points of an inference model. Consequently, supporting distributed intelligence requires coordinated adaptation of wireless communication and AI execution to changing resource conditions.

To accommodate these resource variations, communication and computing constraints have been incorporated into distributed AI design. In particular, communication-aware federated learning adapts client participation and model-update transmission to available network resources [3], [4], while collaborative inference adjusts model partitioning and intermediate representations to communication and computing conditions [5]. These approaches demonstrate the benefits of adapting AI execution to resource availability. However, their resource representations, adaptation decisions, and performance criteria are typically tailored to individual applications. A unified design perspective can therefore help organize these task-specific approaches by connecting network resource constraints with AI execution requirements. Such a perspective should account for both which AI configurations are supportable and how the information exchanged within those configurations contributes to task performance.

Motivated by these considerations, this article presents Wireless AI Networks as a unified design perspective built on resource feasibility and task-level information value. The former characterizes which AI operations can be supported by the available communication and computing resources. To express these constraints in terms meaningful to AI execution, an AI-executable resource budget is introduced, specifying task-relevant operating limits such as admissible update sizes, model partitions, and device participation. However, multiple configurations may satisfy these limits while yielding different AI performance. Task-level information value therefore provides a complementary basis for selecting among candidate configurations and allocating transmission resources. Depending on the task, this value may be estimated from modelupdate characteristics, intermediate-feature relevance, or the sensitivity of model accuracy to transmission errors.

These two considerations give rise to complementary directions of wireless-AI adaptation. In the wireless-to-AI direction, AI configurations, including update selection, model partitioning, and device participation, are adjusted to available network resources. Conversely, in the AI-to-wireless direction, the importance and error sensitivity of AI information guide transmission decisions, such as power allocation, scheduling, and reliability protection. These decisions are interdependent, since the selected AI configuration determines transmission requirements, while wireless allocation affects configuration feasibility and execution performance. Their coordination therefore requires joint consideration of resource constraints and task performance. Furthermore, feedback from communication and AI execution informs subsequent decisions as network conditions and task requirements evolve.

The main contributions of this article are summarized as follows:

• Wireless AI Networks is presented as a unified design perspective connecting resource feasibility with tasklevel information value. Specifically, an AI-executable resource budget maps communication and computing constraints to candidate AI configurations, while taskdependent value estimates guide information selection and transmission.

• Building on this perspective, a wireless-AI co-adaptation framework is developed to coordinate AI configuration and wireless resource allocation under common resource and performance constraints. The framework integrates resource abstraction, joint decision making, and execution feedback, thereby supporting adaptation to changing network conditions and task requirements.

• Finally, two complementary experiments illustrate the adaptation principles through partial model updates in wireless federated learning and sensitivity-aware model transmission. The results demonstrate that update selection and statistical correction improve learning performance under identical model-update payload budgets, while sensitivity-aware power allocation reduces accuracy degradation under the same average transmit-power constraint.

The remainder of this article is organized as follows. Section II introduces the Wireless AI Networks design perspective and its resource abstraction. Building on these concepts, Section III presents the wireless-AI co-adaptation framework, while Section IV illustrates its design principles through two complementary experiments Finally, Section V concludes the article and outlines future research directions.

## II. FROM AI OVER WIRELESS TO WIRELESS AI NETWORKS

Wireless connectivity enables distributed AI nodes to exchange information and coordinate computation across devices, edge servers, and cloud platforms [6]. However, changes in communication and computing resources affect which AI operations can be supported and how effectively they can be executed. This section examines these relationships and introduces resource feasibility and task-level information value as complementary considerations in wireless-AI design.

## A. From Communication Efficiency to Task-Oriented Adaptation

Communication-efficient AI over wireless seeks to reduce the cost of exchanging information through techniques such as model compression, sparsification, and quantization [7]– [9]. Beyond reducing payload size, communication-aware federated learning adapts client participation and model-update transmission to network conditions [10], [11], while collaborative inference adjusts model partitioning and intermediate representations to available communication and computing resources [12], [13]. Although these approaches address different tasks, they share a common requirement: translating network resource constraints into AI execution decisions.

This translation connects two different descriptions of the same system. Wireless networks are characterized by channel quality, available bandwidth, achievable rate, and connectivity, together with latency, reliability, and energy constraints. AI execution, in turn, is configured through model components, feature representations, participating devices, and computation placement. Consequently, a resource description becomes useful to an AI task when it identifies the operations and information exchanges that can be supported. For example, an uplink transmission budget may be translated into an admissible model-update size, while communication and computing constraints jointly restrict the candidate partition points of an inference model.

Resource feasibility alone, however, does not determine which configuration should be selected. Different update subsets or feature representations may satisfy the same resource limits yet contribute differently to learning or inference performance. Moreover, their sensitivity to transmission errors may differ, leading to different requirements for wireless protection. Task-oriented adaptation therefore requires both identifying supportable AI configurations and evaluating the task-level consequences of selecting and transmitting their information. These complementary considerations motivate the resource abstraction and information-value perspective introduced next.

B. Wireless AI Networks and Task-Aware Resource Abstraction

Wireless AI Networks describes a design perspective in which communication and computing conditions shape AI execution, while AI task requirements and information characteristics guide wireless resource allocation. As illustrated in Fig. 1, this perspective applies to distributed learning, collaborative inference, foundation-model services, and multi-agent systems. Across these applications, the common objective is to coordinate the information required by an AI task with the resources available for its processing and delivery. To establish this connection, an AI-executable resource budget is introduced to express network resource constraints as taskrelevant operating limits. The abstraction accounts for payload and signaling overhead, transmission deadlines, reliability requirements, and processing demands. Accordingly, available resources can be mapped to candidate AI configurations, including update sizes and client participation in federated learning, or model partitions and feature dimensions in collaborative inference. Since transmission requirements depend on the selected configuration, these candidates remain subject to the wireless allocation and transmission mode determined during joint decision making. Consider, for example, a federated-learning client whose effective uplink budget permits the transmission of only 2% of its model coordinates. After accounting for representation and signaling overhead, multiple coordinate subsets may satisfy this limit. Nevertheless, random selection and update-dependent sampling can produce different aggregation outcomes despite transmitting the same number of coordinates. The budget therefore characterizes the admissible update size, while the selection and aggregation strategy determines how effectively that budget supports learning.

![](images/42e6d666b78da5e971120a4402cc934f6b92778da05ef17e4663415a65c7000a.jpg)  
Figure 1. Overview of Wireless AI Networks as a design perspective for distributed intelligence. Wireless conditions shape feasible AI execution, while task-level information value guides wireless resource allocation across federated learning, collaborative inference, foundation-model services, and multi-agent systems.

Task-level information value provides a basis for distinguishing among these alternatives [14]. Its estimation depends on both the AI task and the decision under consideration. Model-update characteristics may guide coordinate selection, whereas validation-based sensitivity estimates may indicate how corruption of different model bits affects accuracy. Such estimates represent task-dependent consequences rather than a universal information-value metric. They can therefore guide both the selection of AI information and the allocation of transmission resources, including power and reliability protection [15]. Together, resource feasibility and task-level information value form the basis of the wireless-AI co-adaptation framework developed in the following section.

## III. WIRELESS-AI CO-ADAPTATION FRAMEWORK

Building on the resource abstraction and information-value perspective introduced in Section II, Fig. 2 presents a wireless-AI co-adaptation framework comprising three stages: resource abstraction and candidate AI configuration identification, joint AI configuration and wireless resource allocation, and distributed execution with feedback. Together, these stages connect current resource conditions with task-dependent decisions and observed execution outcomes. The interaction between Stages A and B refines candidate configurations before execution, while Stage C provides feedback for subsequent adaptation.

## A. Resource Abstraction and Candidate AI Configurations

The first stage establishes a resource envelope from current wireless and computing conditions, together with task requirements. Channel estimates, available bandwidth and power, energy budgets, and computing capacity characterize the available resources, while transmission deadlines and reliability requirements constrain their use. Candidate transmission modes are then considered to translate this envelope into an AI-executable resource budget, accounting for payload representation, signaling overhead, and processing demands.

![](images/61a9498d739914bd010a3ebe042eaf00e07666a55129f670e2e214ce6c3a6bd5.jpg)  
Figure 2. Wireless-AI co-adaptation framework. Stage A derives candidate AI configurations from the resource envelope. Stage B jointly refines AI configuratio and wireless allocation, revisiting Stage A when feasibility constraints require an updated candidate set. Stage C uses observed communication and AI outcomes to support subsequent adaptation.

The resulting budget is expressed through task-specific configuration variables. In federated learning, for example, it restricts admissible combinations of client participation and update size. In collaborative inference, it identifies candidate model partitions and intermediate-feature dimensions that can be supported by communication and computing resources. The output is therefore an initial set of candidate AI configurations whose final feasibility depends on the wireless allocation and transmission protection determined in Stage B. If joint evaluation reveals that a candidate cannot meet the resource or performance constraints, the operating limits are revised and the candidate set is updated accordingly.

## B. Task-Value-Guided AI Configuration and Wireless Allocation

Given the candidate configurations identified in Stage A, the second stage jointly refines AI configuration and wireless resource allocation according to task-level information value. Specifically, the AI configuration determines which information is processed or exchanged, while wireless allocation determines how that information is delivered. Since these decisions jointly affect resource consumption and task performance, they are evaluated under common communication, computing, and performance constraints.

Information-value estimates are supplied by the AI task, whereas channel and resource estimates are provided by wireless resource management. In federated learning, for example, client-side update statistics can guide coordinate selection, together with the statistical correction required for aggregation. For quantized model transmission, validation-based sensitivity estimates can guide power allocation or reliability protection across bit positions. An edge-side task controller can coordinate these inputs with the wireless scheduler, with the resulting decisions implemented at the relevant devices and radio nodes.

Each candidate configuration and its associated resource assignment are evaluated in terms of communication cost, processing demand, latency, reliability, energy consumption, and estimated AI performance. Communication cost includes the transmitted values and any required indexing, quantization metadata, and control signaling. Consequently, a configuration that satisfies a nominal payload limit may still be infeasible under a particular transmission or protection strategy.

If the requirements cannot be met within the current candidate space, Stage B returns updated operating restrictions to Stage A so that the candidate set can be revised. Otherwise, the selected AI configuration and wireless allocation proceed to execution. This interaction refines feasibility within the current decision cycle, while the execution feedback introduced next informs subsequent adaptation.

## C. Distributed Execution and Feedback-Driven Adaptation

The third stage executes the selected AI configurations and transmission strategies across devices, edge nodes, and cloud platforms. Observed transmission latency, packet delivery, and energy consumption are collected alongside training progress, inference accuracy, or task-completion quality. These observations update both resource estimates and subsequent decisions.

Feedback operates across different timescales. Channel and delivery measurements update the resource envelope and transmission estimates, supporting short-term adjustments to scheduling, power, and protection. AI-side outcomes refine task-dependent value estimates and configuration policies, including update-selection probabilities, model partitioning, and bit-sensitivity profiles. Execution feedback therefore revises the inputs to Stages A and B for subsequent decisions, while the B-to-A feasibility revision operates within the current decision cycle.

## IV. CASE STUDY

Two complementary experiments illustrate wireless-AI coadaptation through model-information exchange over wireless links. The first examines how time-varying uplink resources constrain partial model updates in federated learning (FL), whereas the second investigates how the error sensitivity of quantized model information guides transmission power allocation. Together, they illustrate resource-driven AI configuration and task-guided wireless allocation.

## A. Wireless AI Scenario and Experimental Setup

For wireless-to-AI adaptation, ten clients collaboratively train a lightweight convolutional neural network on CIFAR-10. Local datasets follow a non-IID Dirichlet partition with $\alpha = 0 . 3 .$ . Training proceeds for 100 communication rounds, with one local epoch per admitted client. In each round, seven clients are selected uniformly without replacement, giving each client an admission probability of 0.7. The remaining clients do not upload updates in that round.

To represent time-varying wireless opportunities, each client’s uplink SNR is independently sampled in each round from a Gaussian distribution in decibels with mean 12 dB and standard deviation 4 dB, truncated to [3, 21] dB. Each admitted client is allocated orthogonal uplink resources with a bandwidth of 180 kHz and a 200 ms transmission window. The model-update payload budget is obtained from the instantaneous Shannon rate using an effective-rate factor of 0.8. This budget counts FP32 update values and excludes coordinate indices and control signaling. At the nominal operating point, the average uploaded coordinate ratio is 2.336%; the budget sweep varies this average while retaining the same SNR model. Thus, client admission and channel-dependent payload budgets jointly constrain the information available for aggregation.

Channel-adaptive unbiased partial-update federated learning (CAUP-FL) is adopted to accommodate these constraints. For each admitted client, the payload budget determines the number of coordinates to upload. Coordinate inclusion probabilities are then constructed from update magnitudes subject to a minimum coverage constraint, and a fixed-cardinality subset is sampled accordingly. Before transmission, the selected values are corrected by inverse-probability weighting at the client, so the inclusion probabilities need not be communicated. The server additionally accounts for client admission probabilities during aggregation. In this way, channel-adaptive update selection is combined with statistical correction.

TABLE I  
CIFAR-10 RESULTS OVER 12 PAIRED SEEDS, DIRICHLET α = 0.3, AND 30% ADMISSION OUTAGE. ACCURACY IS REPORTED AS MEAN ± STANDARD DEVIATION.
<table><tr><td>Method</td><td>Accuracy (%)</td><td>Payload (Mbit)</td><td>Coordinates (%)</td></tr><tr><td>FedAvg (dense oracle)</td><td> $6 5 . 5 4 \pm 2 . 4 2$ </td><td>3025.344</td><td>100.000</td></tr><tr><td>Fixed Top-k</td><td> $5 0 . 7 9 \pm 3 . 2 2$ </td><td>70.664</td><td>2.336</td></tr><tr><td>Fixed Random</td><td> $5 8 . 3 6 \pm 4 . 7 7$ </td><td>70.664</td><td>2.336</td></tr><tr><td>Dynamic Random</td><td> $5 7 . 3 2 \pm 5 . 7 0$ </td><td>70.664</td><td>2.336</td></tr><tr><td>CAUP-FL</td><td> ${ \bf 6 0 . 7 6 \pm 4 . 5 8 }$ </td><td>70.664</td><td>2.336</td></tr></table>

Payload counts FP32 model-update values and excludes coordinate indices and control signaling. CAUP-FL and Dynamic Random have identical payloads within each paired seed. Their paired accuracy difference is +3.43 percentage points (95% interval [+1.61, +5.26]).

CAUP-FL is compared with five partial-update baselines. Dynamic Random selects coordinates uniformly, with the upload size determined by the instantaneous payload budget. Fixed Random and Fixed Top-k impose predefined upper limits on update size and reduce the number of uploaded coordinates when the instantaneous budget is insufficient. By contrast, Static-rate Random and Static-rate Top-k use upload sizes determined by long-term channel statistics and skip transmission whenever the instantaneous budget cannot accommodate the prescribed update. Random-based schemes select coordinates uniformly, whereas Top-k schemes retain the largest-magnitude update coordinates. The two static-rate schemes are included in the learning-curve comparison. Dense FedAvg transmits complete model updates and serves as a communication-intensive reference.

For the complementary AI-to-wireless direction, 8-bit uniformly quantized CNN parameters are transmitted over independent Rayleigh flat-fading channel uses with coherent binary phase-shift keying (BPSK) and perfect receiver channel knowledge. Quantized weights use a signed two’s-complement representation. The power-allocation strategies are evaluated at SNRs from 6 to 12 dB under a common average transmitpower constraint, using 200 independent channel realizations per SNR point.

## B. Wireless-to-AI Adaptation: Resource-Aware Model Updates

Fig. 3(a) compares the learning trajectories over 100 communication rounds, while Table I summarizes final test accuracy across 12 paired seeds. CAUP-FL achieves 60.76% mean accuracy, compared with 57.32% for Dynamic Random. The paired improvement is 3.43 percentage points, with a 95% interval of [+1.61, +5.26] percentage points.

At the nominal operating point, CAUP-FL and Dynamic Random have the same reported model-update payload of

![](images/a8d8b566d9d371b2978b91f537806696de7ddd23c1f08a23a67f64f78fbb3656.jpg)

![](images/c218326fcae8e5102091d01ac7669c133337e20e873e2d9a2ac03ae594062c1d.jpg)  
Figure 3. Wireless-to-AI adaptation through resource-aware model updates: (a) test accuracy over communication rounds and (b) final test accuracy under different uplink budgets.

70.664 Mbit. Their performance difference reflects the combined effects of update-dependent coordinate selection and statistical correction rather than additional uploaded model values. Dense FedAvg achieves 65.54% accuracy with a substantially larger payload of 3025.344 Mbit.

Fixed Random achieves 58.36% accuracy, compared with 57.32% for Dynamic Random, although both employ uniform coordinate selection and adjust upload sizes to instantaneous channel conditions. CAUP-FL achieves higher accuracy by combining channel-adaptive upload sizes with update-dependent coordinate selection and statistical correction. Fig. 3(b) further shows that its advantage persists across the evaluated upload ratios, with larger gains under tighter budgets. These results illustrate how the selection and delivery of model information affect learning performance under constrained wireless resources.

## C. AI-to-Wireless Adaptation: Sensitivity-Aware Transmission

For quantized model parameters, bit errors at different positions have unequal effects on the reconstructed model and its task performance. We estimate the sensitivity of each of the eight bit planes by perturbing the corresponding bits in a copy of the quantized model and measuring the resulting validation-accuracy loss.

For each bit plane, we flip that bit in 1% of randomly selected quantized weights and average the validation-accuracy loss over ten random masks. Transmit power is allocated in proportion to the measured sensitivity, with a positive offset of $1 0 ^ { - 4 }$ , and normalized to the same average bit power across all schemes. For comparison, Equal assigns uniform power to all bit planes, while UEP-3dB and UEP-6dB provide bit planes 4–7 with fixed 3 and 6 dB power advantages over planes 0– 3, respectively. All schemes satisfy the same average power constraint.

The accuracy drop is defined as the nonnegative difference between the test accuracy of the error-free quantized model and that of the received model, expressed as a fraction. Fig. 4 reports the mean accuracy drop over 200 independent channel realizations at each SNR. Equal power allocation yields the largest accuracy loss, while both fixed UEP schemes reduce the degradation. UEP-3dB outperforms UEP-6dB, indicating that a larger fixed power imbalance does not necessarily improve model protection. Sensitivity-aware allocation achieves the lowest mean accuracy drop across the evaluated SNR range by assigning greater protection to bit planes with higher measured impact on model performance.

![](images/97f558a6e2ecc9e5bb2faaf61cd7ce1735cdedc115edb8e43e928b0f8826de9a.jpg)  
Figure 4. AI-to-wireless adaptation through sensitivity-aware bit-level power allocation. Mean accuracy drop versus signal-to-noise ratio (SNR) under equal power, fixed unequal error protection (UEP), and sensitivity-aware allocation, with the same average transmit-power budget.

The two experiments illustrate complementary adaptation decisions: wireless resource availability determines the model information that can be uploaded for learning, while taskdependent error sensitivity guides the protection of transmitted AI information.

## V. CONCLUSION AND FUTURE DIRECTIONS

This article has presented Wireless AI Networks as a unified design perspective connecting wireless resource availability with distributed AI execution. An AI-executable resource budget identifies candidate AI configurations, while task-level information value guides their joint refinement with wireless resource allocation. Two complementary experiments have illustrated these principles. Specifically, update selection and statistical correction improve federated learning under identical model-update payload budgets, while sensitivity-aware power allocation reduces model accuracy degradation under a common average transmit-power constraint. These findings motivate the following research directions.

1) Task-Level Information Value Estimation and Resource Abstraction: The contribution of exchanged information depends on both the AI task and its current execution state. Efficient estimation methods are therefore needed to relate update characteristics, feature relevance, and error sensitivity to task performance. Moreover, the computation and signaling costs of obtaining these estimates should be incorporated into the resource abstraction, so that adaptation gains can be assessed against their implementation overhead.

2) Multi-Timescale Adaptation with Reconfiguration Overhead: Wireless conditions may vary faster than AI configurations and information-value estimates. Consequently, rapid adjustments to scheduling, power, and protection must be coordinated with slower changes in model partitioning, participation policies, and configuration strategies. Future designs should determine when reconfiguration is beneficial while accounting for signaling cost, switching overhead, and decision stability.

3) Cross-Task Value Comparison and Network-Wide Coordination: Concurrent AI services may quantify information value through different measures, including learning progress, inference accuracy, and task success. Allocating shared bandwidth, power, and computing resources therefore requires comparable task-level utility signals that also reflect service priorities and performance requirements. Developing such mappings would extend the proposed perspective from individual tasks to the coordination of heterogeneous AI workloads.

## REFERENCES

[1] J. Liu, Y. Du, K. Yang, J. Wu, Y. Wang, X. Hu, Z. Wang, Y. Liu, P. Sun, A. Boukerche, et al., “Edge-cloud collaborative computing on distributed intelligence and model optimization: A survey,” IEEE Commun. Surv. Tutorials, 2026.

[2] F. Jiang, C. Pan, K. Wang, P. Michiardi, O. A. Dobre, and M. Debbah, “From large AI models to agentic AI: A tutorial on future intelligent communications,” IEEE J. Sel. Areas Commun., 2026.

[3] S. Wang, T. Tuor, T. Salonidis, K. K. Leung, C. Makaya, T. He, and K. Chan, “Adaptive federated learning in resource constrained edge computing systems,” IEEE J. Sel. Areas Commun., vol. 37, no. 6, pp. 1205–1221, 2019.

[4] H. H. Yang, Z. Liu, T. Q. Quek, and H. V. Poor, “Scheduling policies for federated learning in wireless networks,” IEEE Trans. Commun., vol. 68, no. 1, pp. 317–333, 2019.

[5] Y. Kang, J. Hauswald, C. Gao, A. Rovinski, T. Mudge, J. Mars, and L. Tang, “Neurosurgeon: Collaborative intelligence between the cloud and mobile edge,” ACM Comput. Archit. News, vol. 45, no. 1, pp. 615– 629, 2017.

[6] H. Luo, Y. Yan, Y. Bian, W. Feng, R. Zhang, Y. Liu, J. Wang, G. Sun, D. Niyato, H. Yu, et al., “Ai reasoning for wireless communications and networking: A survey and perspectives,” ACM Comput. Surv., vol. 58, no. 13, pp. 1–38, 2026.

[7] X. Wang, Z. Tang, J. Guo, T. Meng, C. Wang, T. Wang, and W. Jia, “Empowering edge intelligence: A comprehensive survey on on-device AI models,” ACM Comput. Surv., vol. 57, no. 9, pp. 1–39, 2025.

[8] Y. Lin, S. Han, H. Mao, Y. Wang, and W. J. Dally, “Deep gradient compression: Reducing the communication bandwidth for distributed training,” arXiv preprint arXiv:1712.01887, 2017.

[9] D. Alistarh, D. Grubic, J. Li, R. Tomioka, and M. Vojnovic, “QSGD: Communication-efficient SGD via gradient quantization and encoding,” Adv. Neural. Inf. Process Syst., vol. 30, 2017.

[10] E. Paolini, A. Pinto, L. Valcarenghi, and F. Esposito, “Programmable in-network aggregation for communication-aware federated learning in 5g rans,” IEEE Trans. Netw. Serv., 2026.

[11] M. M. Amiri and D. Gund¨ uz, “Federated learning over wireless fading¨ channels,” IEEE Trans. Wireless Commun., vol. 19, no. 5, pp. 3546– 3557, 2020.

[12] Y. Hao, N. Ding, W. Xia, H. Ge, and L. Xu, “Dnn partitioning for cooperative inference in edge intelligence: Modeling, solutions, toolchains,” ACM Comput. Surv., vol. 58, no. 8, pp. 1–34, 2026.

[13] J. Shao and J. Zhang, “Bottlenet++: An end-to-end approach for feature compression in device-edge co-inference systems,” in 2020 IEEE Int Conf. Commun. Workshops (ICC Workshops), pp. 1–6, IEEE, 2020.

[14] J. Shao, Y. Mao, and J. Zhang, “Learning task-oriented communication for edge inference: An information bottleneck approach,” IEEE J. Sel Areas Commun., vol. 40, no. 1, pp. 197–211, 2021.

[15] M. Jankowski, D. Gund ¨ uz, and K. Mikolajczyk, “Airnet: Neural network¨ transmission over the air,” IEEE Trans. Wireless Commun., vol. 23, no. 9, pp. 12126–12139, 2024.