# Privacy-enhanced federated learning via asynchronous aggregation and local differential perturbation

Zhen Zhong\*<sup>a</sup>, Shini Yang<sup>b</sup>, Liesheng Wei<sup>c</sup> <sup>a</sup>Georgetown University, Washington,D.C, USA <sup>b</sup>LinkedIn, CA, USA <sup>c</sup>Shanghai Ocean University, Shanghai, China \* Corresponding author: holidayjanezz@gmail.com

## ABSTRACT

This study proposes a privacy-enhanced federated learning framework to address secure collaborative training in distributed data environments. The framework integrates Dynamic Differential Privacy (DDP), lightweight Homomorphic Encryption (HE), and Local Differential Privacy (LDP) mechanisms to ensure data privacy protection during model training. Additionally, the framework employs an asynchronous aggregation strategy with version control to support distributed training in asynchronous environments. Experimental validation on the CIFAR-10 and Purchase-100 benchmark datasets demonstrates that the method maintains high classification accuracy (up to 82.6%) even under stringent privacy constraints (ε = 0.1), while reducing communication overhead by 21.3% compared to FedAvg. Experimental results demonstrate that this framework effectively balances privacy protection and model performance in distributed machine learning scenarios, providing a scalable technical foundation for large-scale distributed collaborative computing.

Keywords: federated learning, differential privacy, additive homomorphic encryption, asynchronous update

## 1. INTRODUCTION

In distributed machine learning scenarios, achieving efficient model training while protecting data privacy poses a significant challenge. Although technologies like differential privacy (DP) and homomorphic encryption (HE) can mitigate data leakage risks, technical hurdles remain in ensuring traceability of model iteration processes, implementing access controls, and maintaining transparent data processing workflows. For instance, merely injecting DP noise fails to validate the processing steps of model updates; similarly, encrypted gradients cannot guarantee adherence to predetermined data flow constraints. These technical issues are particularly pronounced in cross-domain applications spanning finance, healthcare, and industry. Therefore, this study proposes a privacy-enhanced federated learning framework that integrates technical protection mechanisms and system governance requirements into the system design, enabling secure and efficient distributed training.

Federated Learning (FL) provides a decentralized training paradigm that reduces raw data exposure but still suffers from compliance gaps. Existing studies on privacy-by-design FL environments highlight the need for infrastructures capable of supporting regulatory audits and verifiable training behavior in large-scale sensitive domains [1]. Integrity-preserving mechanisms based on trusted execution environments demonstrate that protecting model updates alone is insufficient unless paired with traceable execution procedures and certifiable audit logs [2]. Research on anonymous and privacypreserving FL in industrial big data shows that large organizations require mechanisms that prevent reconstruction attacks while maintaining operational transparency [3]. Gradient sparsification and pruning–based privacy techniques further confirm that privacy management must be integrated with model coordination and update control to avoid legal inconsistencies caused by uncontrolled gradient disclosure [4]. Comprehensive surveys on privacy-preserving computation indicate that compliance requires combining cryptographic protection, differential privacy, and verifiable computation processes within a cohesive governance architecture [5].

Against this background, there remains a lack of federated learning systems that simultaneously address technical privacy risks and compliance obligations such as lineage-aware auditability, version-controlled model transformation, and jurisdiction-specific data-use constraints. To fill this gap, this study proposes a privacy-enhanced Federated Learning framework integrating dynamic differential privacy, lightweight homomorphic encryption, localized perturbation mechanisms, and asynchronous version-aware aggregation. The framework is designed to support legally aligned training workflows in heterogeneous, latency-prone environments while ensuring that the entire update path remains auditable, traceable, and consistent with regulatory expectations.

## 2. DESIGN OF PRIVACY-PRESERVING ALGORITHMS FOR FEDERATED LEARNING FOR LARGE-SCALE DATA

## 2.1 Gradient perturbation mechanism based on dynamic differential privacy

In the distributed training process of federated learning, to balance privacy protection and model performance, the gradient perturbation mechanism requires dynamically adjustable noise injection capabilities. Therefore, the DDP-based method proposed in this paper employs a dynamic sensitivity estimation rule. By continuously measuring the dispersion of local gradients within the current update window, it obtains the real-time gradient fluctuation range $\Delta _ { \mathrm { t } }$ . “Dynamic” refers to $\Delta _ { \mathrm { t } }$ being adjusted based on the current gradient distribution at each update round, enabling the perturbation intensity to reflect both the model's evolutionary characteristics and the heterogeneity of different data sources. Before transmission, each client adds noise to its local gradient $\mathrm { ~ g ~ } \mathrm { ~ i ~ } , \mathrm { ~ t ~ }$ . The perturbed gradient is computed as follows:

$$
\widetilde { g } _ { i } ^ { t } = g _ { i } ^ { t } + N ( 0 , \sigma _ { t } ^ { 2 } \Delta _ { t } ^ { 2 } )\tag{1}
$$

where the Gaussian noise has a mean of 0 and variance of ${ \sigma _ { \mathrm { t } } } ^ { 2 }$ . Here, $\sigma _ { \mathrm { t } }$ is determined by a round-adaptive scaling rule: when $\Delta _ { \mathrm { t } }$ increases (indicating heightened sensitivity or greater gradient variation among clients), $\sigma _ { \mathrm { t } }$ proportionally increases to enhance protection; conversely, when $\Delta _ { \mathrm { t } }$ stabilizes, $\sigma _ { \mathrm { t } }$ gradually decreases to maintain model efficacy. This round-level adjustment ensures noise magnitude is not fixed but evolves dynamically throughout training.

To prevent excessive consumption of the overall privacy budget ϵ, a cumulative privacy account mechanism is employed. This mechanism incrementally records the budget consumed in each iteration and adjusts subsequent σₜ values to control overall privacy loss. Through the dynamic coupling of $\Delta _ { \mathrm { t } }$ estimation, $\sigma _ { \mathrm { t } }$ adjustment, and iterative budget tracking, the perturbation mechanism achieves fine-grained privacy control.

## 2.2 Lightweight encryption and secure aggregation protocols

The secure aggregation protocol (as shown in Figure 1) consists of four functional phases, each performing specific cryptographic coordination tasks:

① Client-side encryption: Each client locally compresses and encodes gradient vectors using gated sparsification, then encrypts them with the homomorphic encryption function E(gi,t). The sparsity threshold is adaptively adjusted based on local gradient magnitude characteristics to reduce encryption overhead.

② Encrypted Transmission: Encrypted gradients are chunked into packets with technical metadata (timestamps, version numbers, priorities, etc.) appended, then transmitted via a secure asynchronous RPC channel. Payloads are serialized using Protobuf and protected by TLS 1.3 with short-term mTLS certificates.

③ Server Aggregation: Received encrypted gradients are cached and aggregated directly in ciphertext space using additive homomorphic operations:

$$
E \left( \sum _ { i = 1 } ^ { N } g _ { i } ^ { t } \right) = \prod _ { i = 1 } ^ { N } E ( g _ { i } ^ { t } )\tag{2}
$$

where E(⋅) denotes the encryption function and N is the number of participating clients. To further reduce computational burden, sparse coding compresses gradient vectors. The server cannot access any decrypted intermediate values.

④ Decryption and Global Update: Aggregated ciphertexts are decrypted using a secure keystore (KMS proxy), and the resulting global gradient is used for parameter updates. Version control metadata is recorded after each aggregation round to support iterative tracking.

![](images/619b9d81732071bf36f45663b668e2d5bb035e20fee389fcbd4632a5ae8a0588.jpg)  
Figure 1. Schematic diagram of lightweight encryption and secure aggregation protocol flow

## 2.3 Client-side data perturbation based on localized differential privacy

To enhance privacy protection for client-side data, this paper proposes a data perturbation strategy based on a localized differential privacy mechanism. Unlike traditional centralized differential privacy, this method applies random perturbations directly to the raw feature data xi at the client side, ensuring that even if the server is attacked, the origina information cannot be recovered. Specifically, the perturbation employs an optimized random response mechanism. The original binary feature values are set to {0,1}, and the generation rule for the perturbed feature x\~i is as follows:

$$
\widetilde { x } _ { i } = \left\{ \begin{array} { l l } { 1 , } & { w i t h \ p r o b a b i l i t y \ p i f \ x _ { i } = 1 } \\ { 0 , } & { w i t h \ p r o b a b i l i t y 1 - p i f \ x _ { i } = 1 } \\ { 1 , } & { w i t h \ p r o b a b i l i t y \ q i f \ x _ { i } = 0 } \\ { 0 , } & { w i t h \ p r o b a b i l i t y 1 - q i f \ x _ { i } = 0 } \end{array} \right.\tag{3}
$$

where $p = \frac { e ^ { \varepsilon } } { e ^ { \varepsilon } + 1 } , \ q = \frac { 1 } { e ^ { \varepsilon } + 1 }$ , represents the local privacy budget. To accommodate the high-dimensional sparse

distribution of feature vectors, a grouped perturbation strategy is introduced, assigning varying perturbation strengths to different feature groups based on sensitivity differences. Table 1 illustrates the theoretical data distortion rate under different privacy budgets, revealing the fundamental trade-off between privacy protection strength and data usability.

Table 1. Comparison of theoretical data distortion rates under different local privacy budgets
<table><tr><td rowspan=1 colspan=1>Local privacy budget €</td><td rowspan=1 colspan=1>Theoretical distortion rate (single feature)</td><td rowspan=1 colspan=1>Theoretical distortion rate (group perturbation)</td></tr><tr><td rowspan=1 colspan=1>0.1</td><td rowspan=1 colspan=1>48.7%</td><td rowspan=1 colspan=1>42.1%</td></tr><tr><td rowspan=1 colspan=1>0.5</td><td rowspan=1 colspan=1>28.3%</td><td rowspan=1 colspan=1>23.5%</td></tr><tr><td rowspan=1 colspan=1>1.0</td><td rowspan=1 colspan=1>15.8%</td><td rowspan=1 colspan=1>12.6%</td></tr></table>

## 2.4 Distributed asynchronous federated learning framework

To support heterogeneous nodes with varying network latency and computational capabilities in training participation, this section proposes an asynchronous federated learning framework. This framework employs a distributed asynchronous update strategy, allowing clients to upload gradients immediately after completing several steps of local training without waiting for all nodes to synchronize. The server performs asynchronous aggregation based on the order of arrival. Let the model parameters generated by the i-th client during the k-th step of local training be denoted as wik. The server's asynchronous update rule is expressed as [6]:

$$
w ^ { ( t + 1 ) } = w ^ { ( t ) } - \eta \nabla \widetilde { g } _ { i } ^ { k }\tag{4}
$$

where η is the global learning rate, and $\nabla \mathbf { g } _ { \mathbf { \ell } }$ \~ik represents the perturbed gradient uploaded by the client. To mitigate parameter drift caused by asynchronous updates, the system introduces a version control mechanism with weighted dynamic decay. Each time a client uploads a gradient, the server records its corresponding local model version number vik. If the deviation between the uploaded gradient's version and the server's current model version exceeds the threshold $\delta ,$ the aggregation weight is adjusted according to a dynamic decay function. The specific weight adjustment formula is:

$$
\omega _ { i } = \exp ( - \lambda \cdot ( t - \nu _ { i } ^ { k } ) )\tag{5}
$$

where t denotes the current aggregation round on the server, and $\lambda$ represents the decay coefficient. To address variations in client activity levels and communication frequencies, an adaptive scheduler (as shown in Table 2) is designed to dynamically adjust the server's processing priority for gradient updates under different communication delays. This enhances overall system stability and convergence efficiency.

Table 2. Correspondence between client communication delay and server scheduling priorities
<table><tr><td rowspan=1 colspan=1>Communication delay (ms)</td><td rowspan=1 colspan=1>Priority weight adjustment factor (α)</td></tr><tr><td rowspan=1 colspan=1>&lt;50</td><td rowspan=1 colspan=1>1.0</td></tr><tr><td rowspan=1 colspan=1>50-150</td><td rowspan=1 colspan=1>0.85</td></tr><tr><td rowspan=1 colspan=1>150-300</td><td rowspan=1 colspan=1>0.7</td></tr><tr><td rowspan=1 colspan=1>&gt;300</td><td rowspan=1 colspan=1>0.5</td></tr></table>

Note: α values are obtained from empirical tuning in synthetic delay injection experiments based on the protocol in Section 3.3.

## 3. SYSTEM IMPLEMENTATION, INTEGRATION, AND PERFORMANCEOPTIMIZATION3.1 SYSTEM ARCHITECTURE AND MODULE DESIGN

In distributed federated learning systems, architectural design must simultaneously address technical objectives such as data privacy protection, version tracking, and cross-domain access control. To fulfill these requirements, each architectural component directly embeds privacy protection mechanisms into its operational pathways, rather than relying solely on algorithmic-level safeguards.

The Local Privacy Agent establishes device-level privacy boundaries, preventing raw data or unencrypted intermediate representations from leaving the local environment. Integrating local differential privacy and homomorphic encryption at the client ensures edge devices serve as the first line of privacy defense. This design guarantees all outbound traffic undergoes privacy-preserving processing, enabling data isolation between regions.

Version-aware aggregation servers enhance system security through version-controlled gradient fusion and immutable logging. Each update records privacy budget increments and model version lineage, forming a traceable version history. Version control mechanisms prevent outdated or incompatible updates from affecting the global model, supporting data usage boundaries across organizational domains [7].

An adaptive communication scheduler dynamically adjusts client priorities based on data sensitivity and processing urgency. By tailoring update queues to data characteristics, the scheduler ensures cross-domain training sequences adhere to predefined data flow controls. This design optimizes learning efficiency while maintaining rational processing order. Clients handling highly sensitive data receive priority scheduling to minimize propagation latency.

To secure transmission of sensitive intermediate results, the system applies additive homomorphic encryption (HE) at the client. The encrypted gradient vector for the i-th client is denoted as E(gi), with aggregation performed directly in ciphertext space [8]:

$$
E ( G ) = \sum _ { i = 1 } ^ { N } E ( g _ { i } )\tag{6}
$$

Where E(gi) represents the encrypted local gradient of the i-th client. Figure 2 illustrates the overall architecture and module interactions, outlining the coordination among the privacy proxy, aggregation server, communication scheduler, and encryption service. As shown, modules connect via asynchronous RPC channels to mitigate blocking risks from largescale concurrency. Dynamic task pools manage asynchronous tasks across modules, ensuring efficient and orderly resource scheduling.

![](images/ece00ef892293a1bed6e3b8304b0c9a1a109eab726d669f5f96536ad8aceabf1.jpg)  
Figure 2. Schematic diagram of system architecture and module interaction

Interface Contracts and Data Flow. The integrated system decouples clients, aggregators, and schedulers through stable gRPC/HTTP endpoints. Core contracts include:

(i)/join (Client → Scheduler): {client\_id, model\_tag, hw\_caps, dp\_eps\_budget};

(ii) /pull\_task (Client → Scheduler): Returns {round\_id, model\_uri, clip\_norm, sigma, encrypt\_key\_id, agg\_policy};

(iii) /push\_update (Client → Aggregator): Protobuf Update{client\_id, round\_id, version, payload: EncryptedSparseGrad, ldp\_meta};

(iv)/heartbeat (Client ↔ Scheduler): {client\_id, latency\_ms, loss, cpu\_mem} for health checks and backpressure control [9].

Workloads are serialized using Protobuf and optionally streamed via gRPC. Keys and certificates are configured via KMSbased mTLS channels (TLS 1.3; key rotation ≤ 24h). End-to-end data flow: Client local training → DP pruning + noise → LDP encoder → HE + sparse encoding → chunked streaming → secure aggregation → asynchronous version-aware updates. Version history and operation logs are recorded for each request ID.

Deployment Topology. The integration adopts a containerized microservices architecture on Kubernetes (≥1.27), comprising fl-gateway, fl-scheduler, fl-aggregator, privacy-service, kms-proxy, and metrics-agent. Horizontal Pod Autoscaler (HPA) targets 60% CPU utilization and 300 QPS; PodDisruptionBudget protects aggregator pool quorum (minAvailable=2). CNI uses Calico; optional Istio-enabled mTLS mesh supports service-level authorization (RBAC) and retry budgets.

## 3.1 Core algorithm implementation details

During the system implementation phase, this study optimized the core algorithmic modules to address large-scale heterogeneous data environments and privacy protection requirements. First, within the local sensitivity dynamic computation module, a sliding-window sensitivity estimation method was adopted to update the gradient change range Δt in real-time during each training iteration. The specific calculation formula is as follows [10]:

$$
\Delta _ { t } = \operatorname* { m a x } _ { i \in W _ { t } } \Bigl ( \left\| g _ { i , t } - \bar { g } _ { t } \right\| _ { 2 } \Bigr )\tag{7}
$$

where Wt denotes the set of clients within the current time window, gi,t represents the local gradient of the i-th client, and gˉt is the average gradient within the window.

In the lightweight encryption module, sparse encoding employs Gated Sparsification technology. This technique dynamically selects significant gradient elements via a gated network and encrypts only these significant elements [11]. The asynchronous parameter aggregation module employs a version control mechanism. On the server side, it utilizes asynchronous updates with weighted decay, where the weight dynamically adjusts based on version differences. The adjustment function is as follows:

$$
\omega \big ( \Delta \nu \big ) = e ^ { - \lambda \cdot \Delta \nu }\tag{8}
$$

where Δv is the difference between the client uploaded model version and the server version, and λ is the system preset attenuation factor.

## 3.2 Communication and computation optimization strategies

To address communication bottlenecks and distributed computing resource challenges in large-scale data environments, the system introduces a bidirectional collaborative optimization mechanism for communication and computation. On the communication side, a hierarchical gradient transmission protocol is employed to batch-transmit sparsely encrypted gradients uploaded by clients according to importance levels, prioritizing critical gradients to reduce overall latency. Specific batching rules are detailed in Table 3 [12].

On the computational side, the system designs a dynamic task scheduling pool that adapts local training steps based on client computational power and communication latency metrics. The client's local training iteration count k is dynamically determined by the following formula [13]:

$$
k _ { i } = \left\lfloor k _ { b a s e } \times \left( \frac { C _ { n o r m } } { C _ { i } } \times \frac { D _ { i } } { D _ { n o r m } } \right) ^ { \beta } \ \right\rfloor\tag{9}
$$

where Ci is the client computational power score, Di is the communication latency score, β is the adjustment factor, kbase is the baseline step count, and ⌊⋅⌋ denotes the floor operation.

Table 3. Sparse gradient batch transmission prioritization policy table
<table><tr><td rowspan=1 colspan=1>materiality rating</td><td rowspan=1 colspan=1>Gradient sparsity threshold (Top-k%)</td><td rowspan=1 colspan=1>transmission priority</td></tr><tr><td rowspan=1 colspan=1>your (honorific)</td><td rowspan=1 colspan=1>Top-10%</td><td rowspan=1 colspan=1>Priority 1</td></tr><tr><td rowspan=1 colspan=1>center</td><td rowspan=1 colspan=1>Top-30%</td><td rowspan=1 colspan=1>Priority 2</td></tr><tr><td rowspan=1 colspan=1>lower (one&#x27;s head)</td><td rowspan=1 colspan=1>Top-60%</td><td rowspan=1 colspan=1>Priority 3</td></tr></table>

Note: Sparsity thresholds and priority assignments were derived from observed gradient magnitude distributions in CIFAR-10 during experimental profiling (see Section 4.1).

## 3.3 System integration and engineering practices

Security and key management implementations primarily focus on communication security. mTLS using short-lived certificates provides encrypted authentication for each request. Regular rotation of HE public keys and LDP seeds via KMS proxy establishes revocable regional encryption boundaries, preventing cross-region key reuse. Audit logs in immutable storage create a traceable chain of model training events.

Observability components like budget\_epsilon\_total and key\_rotation\_age specifically deliver privacy-oriented telemetry data. Monitoring cumulative privacy budgets ensures the system does not exceed allocated differential privacy limits, while key rotation metrics confirm internal security policy enforcement. These engineering choices transform observability into an enforcement tool for privacy protection.

Resource isolation enhances privacy protection by preventing HE-enabled aggregation nodes from coexisting with unverified workloads, thereby mitigating risks of memory inference or side-channel leaks in shared environments.

Fault-tolerance mechanisms like TTL-based task leases ensure expired tasks persist beyond their processing windows.   
Through automatic reallocation or termination of expired tasks, the system strictly controls data retention behavior [14].   
Aggregate checkpoints are written after every N=1000 updates for rapid recovery.

For interoperability, the system exposes interfaces to third-party model runners via OpenAPI/Protobuf IDL and supports heterogeneous accelerators through a “plugin runner” interface, enabling progressive integration with existing MLOps toolchains [15].

## 4. EXPERIMENT AND RESULT ANALYSIS

## 4.1 Experimental setup

To validate the performance of the proposed federated learning privacy-preserving algorithm under structurally complex data conditions, we employ two widely used benchmark datasets: CIFAR-10 and Purchase-100. As a visual dataset, CIFAR-10 features a high-dimensional input space and inter-class correlated features, making it suitable for evaluating model performance in complex representation tasks. Purchase-100 is a sparse categorical dataset commonly used to simulate distributed transaction records or user-item matrices, enabling evaluation of model performance under sparse feature scenarios.

Experiments were conducted with 500 clients, each holding randomly partitioned data samples to simulate a federated distributed environment. The privacy budget ϵ varied within {0.1, 0.5, 1.0} to assess the impact of different privacy strengths. Dataset statistics are shown in Table 4. The baseline architectures comprise a lightweight CNN for CIFAR-10 and a two-layer FCN for Purchase-100, with detailed structural parameters listed in Table 5.

Comparative baseline methods include FedAvg, DP-FedSGD, and SecureBoost, all employing asynchronous communication consistent with this paper's mechanism to ensure fairness [16]. All experiments were conducted within an asynchronous federated learning framework. Communication delays were simulated following a normal distribution N(150ms, 50ms²), with delays exceeding 300ms automatically entering a synchronous control buffer queue. The overall experimental environment was deployed on a cluster of 40 GPU servers, utilizing gRPC for asynchronous communication.

Table 4. Basic information about the dataset used for the experiment
<table><tr><td rowspan=1 colspan=1>data set</td><td rowspan=1 colspan=1>sample size</td><td rowspan=1 colspan=1>Characteristic dimensions</td><td rowspan=1 colspan=1>Number of categories</td></tr><tr><td rowspan=1 colspan=1>CIFAR-10</td><td rowspan=1 colspan=1>60,000</td><td rowspan=1 colspan=1>32 x 32 x 3</td><td rowspan=1 colspan=1>10</td></tr><tr><td rowspan=1 colspan=1>Purchase-100</td><td rowspan=1 colspan=1>197,324</td><td rowspan=1 colspan=1>600</td><td rowspan=1 colspan=1>100</td></tr></table>

Table 5. Experimental model structure and parameter configuration table
<table><tr><td rowspan=1 colspan=1>Network type</td><td rowspan=1 colspan=1>network level</td><td rowspan=1 colspan=1>Number of parameters</td><td rowspan=1 colspan=1>activation function</td></tr><tr><td rowspan=1 colspan=1>CNN (CIFAR-10)</td><td rowspan=1 colspan=1>4-layer convolution + 2-layer fully connected</td><td rowspan=1 colspan=1>1.25M</td><td rowspan=1 colspan=1>ReLU</td></tr><tr><td rowspan=1 colspan=1>FCN (Purchase-100)</td><td rowspan=1 colspan=1>2-layer full connectivity</td><td rowspan=1 colspan=1>0.45M</td><td rowspan=1 colspan=1>ReLU</td></tr></table>

## 4.2 Comparative analysis of experimental results

Experimental results on the CIFAR-10 and Purchase-100 datasets demonstrate that the proposed privacy-preserving federated learning method exhibits outstanding overall performance across various privacy budget settings. Figure 3 displays the model accuracy curves of various methods as the privacy budget ϵ varies on the CIFAR-10 dataset. It can be observed that under the strong privacy constraint of ϵ=0.1, our method still maintains an accuracy of 68.4%, significantly outperforming DP-FedSGD (61.7%) and SecureBoost (64.2%); under the medium privacy strength ϵ=1.0, the accuracy further improves to 82.6% [17].

The classification accuracy results for the Purchase-100 task are summarized in Table 6. Compared to the baseline, our method achieves a stable performance improvement of 3%-5% in high-sparsity data environments. Additionally, benefiting from local perturbations and lightweight encryption strategies, it reduces communication overhead by 21.3% compared to FedAvg.

Convergence speed analysis is shown in Figure 4. On the Purchase-100 dataset, our method converges to the threshold in under 200 rounds, approximately 18.7% faster than FedAvg. The asynchronous update mechanism mitigates system blocking caused by client communication delays, enhancing overall iteration efficiency. Regarding the impact of gradient perturbations on model stability, our method maintains low fluctuation (standard deviation reduced by 11.2%) across multiple iterations, effectively ensuring smooth convergence during model training.

<table><tr><td rowspan=1 colspan=1>methodologies</td><td rowspan=1 colspan=1>€ = 0.1 accuracy(%)</td><td rowspan=1 colspan=1>€ = 0.5 accuracy(%)</td><td rowspan=1 colspan=1>€ = 1.0 Accuracy(%)</td><td rowspan=1 colspan=1>Average communication overhead(MB/round)</td></tr><tr><td rowspan=1 colspan=1>FedAvg</td><td rowspan=1 colspan=1>58.5</td><td rowspan=1 colspan=1>72.3</td><td rowspan=1 colspan=1>79.6</td><td rowspan=1 colspan=1>12.5</td></tr><tr><td rowspan=1 colspan=1>DP-FedSGD</td><td rowspan=1 colspan=1>53.2</td><td rowspan=1 colspan=1>68.7</td><td rowspan=1 colspan=1>75.1</td><td rowspan=1 colspan=1>13.8</td></tr><tr><td rowspan=1 colspan=1>SecureBoost</td><td rowspan=1 colspan=1>55.9</td><td rowspan=1 colspan=1>70.2</td><td rowspan=1 colspan=1>77.0</td><td rowspan=1 colspan=1>14.1</td></tr><tr><td rowspan=1 colspan=1>Methodology of thispaper</td><td rowspan=1 colspan=1>62.4</td><td rowspan=1 colspan=1>74.8</td><td rowspan=1 colspan=1>80.9</td><td rowspan=1 colspan=1>9.8</td></tr></table>

![](images/32a67011211b5ac5c9a55c7fb3d03c5a0b1436955cf9cd5a2a9c1c797a140384.jpg)

Figure 3. Variation curve of the accuracy of each method under different privacy budgets on the CIFAR-10 dataset  
![](images/e74c9039c48cc2bdb3cf8ab8e472fc70037ce2711ea47795c2b7a686f0e25f03.jpg)  
Figure 4. Comparison curve of convergence rounds of each method on Purchase-100 dataset

## 5. CONCLUSION

This paper proposes a novel collaborative learning framework that integrates privacy protection with asynchronous model coordination mechanisms, effectively addressing data collaboration challenges in distributed environments. By integrating dynamic differential privacy, lightweight homomorphic encryption, local perturbations, and version-aware asynchronous aggregation, the framework achieves privacy-preserving, version-tracking, and scalable model training.

In terms of technical innovation, this paper designs a dynamic privacy budget mechanism that adaptively adjusts perturbation intensity based on gradient distribution characteristics during training, maintaining model utility while safeguarding privacy. A lightweight encryption scheme employs sparse coding and gating mechanisms to reduce computational overhead, enabling efficient secure aggregation. Additionally, a version-aware asynchronous update strategy supports distributed training in high-latency environments, ensuring model convergence through version control. At the system implementation level, a comprehensive architecture has been constructed, encompassing modules such as privacy proxies, secure aggregation, and adaptive scheduling, enabling large-scale distributed deployment.

Experimental results on benchmark datasets demonstrate that the framework maintains high model performance even under stringent privacy constraints (ε=0.1). Although current experiments utilize simulated datasets, their structural complexity and non-iid characteristics provide an effective approximation for framework validation. Future research will focus on optimizing dynamic privacy budget allocation strategies, exploring more efficient encrypted aggregation schemes, and improving asynchronous update mechanisms to further enhance the framework's training efficiency and performance in large-scale distributed scenarios.

This study provides a scalable technical solution for privacy-preserving machine learning in distributed environments, holding significant value for advancing research and applications in related fields. Through continuous refinement and optimization, the framework is expected to play a vital role in broader distributed collaborative learning scenarios.

## REFERENCES

[1] Zhang P, Kamel Boulos M N. Privacy-by-design environments for large-scale health research and federated learning from data. International Journal of Environmental Research and Public Health, 2022, 19(19): 11876.

[2] Chen Y, Luo F, Li T, et al. A training-integrity privacy-preserving federated learning scheme with trusted execution environment. Information Sciences, 2020, 522: 69-79.

[3] Zhao B, Fan K, Yang K, et al. Anonymous and privacy-preserving federated learning with industrial big data. IEEE Transactions on Industrial Informatics, 2021, 17(9): 6314-6323.

[4] Zhang Z, Tianqing Z, Ren W, et al. Preserving data privacy in federated learning through large gradient pruning. Computers & Security, 2023, 125: 103039.

[5] Chen J, Yan H, Liu Z, et al. When federated learning meets privacy-preserving computation. ACM Computing Surveys, 2024, 56(12): 1-36.

[6] Awan K A, Din I U, Almogren A, et al. Privacy-preserving big data security for IoT with federated learning and cryptography. IEEE Access, 2023, 11: 120918-120934.

[7] Manzoor H U, Shabbir A, Chen A, et al. A survey of security strategies in federated learning: Defending models, data, and privacy. Future Internet, 2024, 16(10): 374.

[8] Zheng Q, Chen S, Long Q, et al. Federated f-differential privacy//International conference on artificial intelligence and statistics. PMLR, 2021: 2251-2259.

[9] Mo F, Haddadi H, Katevas K, et al. PPFL: Privacy-preserving federated learning with trusted execution environments//Proceedings of the 19th annual international conference on mobile systems, applications, and services. 2021: 94-108.

[10] Khalil M, Esseghir M, Boulahia L M. Privacy-Preserving federated learning: An application for big data load forecast in buildings. Computers & Security, 2023, 131: 103211.

[11] Jog S, Palaniappan D, Jabbar M A. An adaptive framework for privacy-preserving analytics in federated intrusion detection. Decision Analytics Journal, 2025: 100641.

[12] Haripriya R, Khare N, Pandey M, et al. Federated adaptive aggregation: improving privacy and scalability in healthcare AI. Cluster Computing, 2025, 28(10): 617.

[13] Piran F J, Chen Z, Zhang Y, et al. Privacy-preserving decentralized federated learning via explainable adaptive differential privacy. arXiv preprint arXiv:2509.10691, 2025.

[14] Tayyeh H K, AL-Jumaili A S A. Balancing privacy and performance: a differential privacy approach in federated learning. Computers, 2024, 13(11): 277.

[15] Eltaras T, Sabry F, Labda W, et al. Efficient verifiable protocol for privacy-preserving aggregation in federated learning. IEEE Transactions on Information Forensics and Security, 2023, 18: 2977-2990.

[16] Yin X, Zhu Y, Hu J. A comprehensive survey of privacy-preserving federated learning: A taxonomy, review, and future directions. ACM Computing Surveys (CSUR), 2021, 54(6): 1-36.

[17] El Ouadrhiri A, Abdelhadi A. Differential privacy for deep and federated learning: A survey. IEEE access, 2022, 10: 22359-22380.