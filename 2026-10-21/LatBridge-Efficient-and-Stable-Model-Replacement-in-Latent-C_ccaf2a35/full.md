# LatBridge: Efficient and Stable Model Replacement in Latent-Communication Multi-Agent Systems

Jiajun Zhu<sup>1</sup>\*, Haobo Li<sup>1</sup>, Zixin Chen<sup>1</sup>, Huamin Qu<sup>1</sup>, Wenshuo Zhang<sup>1†</sup> <sup>1</sup>The Hong Kong University of Science and Technology

## Abstract

Multi-agent systems rely on communication to complete tasks, and latent communication passes continuous representations to reduce text generation and re-encoding. However, multi-agent systems must reconfigure agent models, whereas existing latentcommunication methods are built for fixed model combinations, so their interfaces may fail after replacement. Our pilot study confirms this interface failure and identifies the requirement for interface reuse after model replacement. We present LatBridge, which connects agent models through a shared Bridge and separate interfaces under a shared latentcommunication protocol. After agent model replacement, LatBridge reuses unchanged interfaces, selects matching low-rank interfaces, and combines them through the Bridge, training only missing ones with all models frozen. The Bridge processes source representations in the basic mode and constructs target input without autoregressive generation in the accelerated mode. Experiments support effective communication after model replacement, with mean basic-mode accuracies of 14.8% on MATH and 27.3% on ALFWorld next-action prediction across eight small-model configurations. Analyses examine the factors supporting task accuracy and communication efficiency.

## 1 Introduction

Multi-agent systems organize language models as agents with different roles and use communication to exchange information, divide tasks, and make collaborative decisions (Li et al., 2023; Hong et al., 2024; Wu et al., 2024). Most existing systems communicate through text, but generating and reencoding intermediate text increases communication time and converts continuous internal representations into discrete text. To reduce these limitations, previous studies have explored latent communication, which exchanges continuous representations such as embeddings (Pham et al., 2024), activations (Ramesh and Li, 2025), hidden states (Du et al., 2026), and key-value caches (Shi et al., 2026). At the same time, multi-agent systems need to select models according to task and resource requirements (Yue et al., 2025) and adjust communication topology for different forms of collaboration (Li et al., 2024; Shen et al., 2025a). These changes may require replacing the model assigned to an agent. However, existing latent-communication methods mainly build communication interfaces for fixed sending and receiving models (Du et al., 2026; Zou et al., 2025; Fu et al., 2025). It remains unclear whether communication interfaces trained by these methods can continue to pass useful information after the model assigned to an agent is replaced.

![](images/fa6f84d859eadd8bb33a94f14b88c2e20dcf2d952c8bd81395988e92f5aa909b.jpg)  
Figure 1: Agent model replacement in latent communication. Interfaces in latent-communication methods can fail after replacement. LatBridge keeps the Bridge model fixed, reuses the unchanged interface, and selects or trains a matching interface for the new model.

To test whether the original communication interface remains usable after model replacement, we conduct a pilot study based on Interlat (Du et al., 2026). The results show that replacing the model assigned to an agent without retraining the interface substantially reduces the original communication gain, indicating that an interface trained for the original model combination cannot be directly applied to a new endpoint model. This result motivates a shared communication protocol for all agents, implemented through a shared communication component that divides the model-pair-dependent representation mapping across its two sides. With this protocol, replacing a model changes only its connection to this component, allowing the interface on the unchanged side to be reused and a new adjacent interface to be trained only when no matching interface exists. However, if this component uses a complete language model to generate and process communicated content token by token, this additional step increases communication latency and makes communication time depend on the generated sequence length. We therefore need a latentcommunication method that reduces both the interface retraining required after model replacement and the processing latency introduced by the shared communication component.

To this end, we present LatBridge for efficient and stable model replacement in multi-agent systems. Figure 1 contrasts interface mismatch after model replacement with interface reuse through the Bridge. Structurally, LatBridge implements the shared communication component as the Bridge and connects the sending source agent and the receiving target agent through an input interface, a fixed Bridge model, and an output interface. When a model is replaced, the system reuses the unchanged interface and looks for a stored interface matching the new model. The two interfaces are connected through the fixed Bridge model. Only missing interfaces are trained, with all models frozen. The input and output interfaces learn representation mappings between adjacent models in separate training stages. Trained interfaces are stored by their adjacent models for use in the current and future matching configurations. Once the required interfaces are available, the system communicates through the input interface, complete frozen Bridge model, and output interface, preserving source-dependent communication. This procedure is called the basic mode. To reduce the latency caused by token-by-token Bridge generation, LatBridge prepares a vocabulary mapping and a corresponding output interface in advance, then uses the Bridge tokenizer and embedding lookup to construct target-side continuous input directly during communication. This procedure is called the accelerated mode. In the accelerated mode, the system bypasses autoregressive generation with the complete Bridge model, reducing communication processing time.

To evaluate LatBridge, we conduct modelreplacement, communication-efficiency, ablation, and representation analyses at two model scales and across different endpoint-model configurations. As existing methods do not provide a suitable direct baseline for agent model replacement, we use controlled experiments, with Interlat as an accuracy reference. With matched interfaces, mean basic-mode accuracy across eight small-model configurations is 14.8% on MATH and 27.3% on ALFWorld nextaction prediction. The scaling experiments show effective communication across the tested configurations. Ablation and representation analyses examine how interfaces, training, and communication content support task accuracy after replacement. The accelerated mode reduces Bridge processing latency while maintaining or improving average task accuracy. Complete experimental settings, results, and analyses are provided in Section 5.

Our contributions are threefold.

• We identify interface failure after model replacement as an unresolved problem in latentcommunication methods, and use a pilot study to show that the original interface cannot be directly reused after model replacement.

• We present LatBridge, which implements a shared communication protocol through a communication component shared by all agents to support interface reuse after model replacement and avoids token-by-token Bridge generation in the accelerated mode.

• We evaluate LatBridge in terms of task accuracy, communication efficiency, component roles, and representation space. The results show that matched interfaces support task accuracy after agent model replacement, while the accelerated mode improves communication efficiency.

## 2 Related Work

Multi-agent systems coordinate role-specialized agents and have developed from dialogue-based collaboration to structured workflows (Li et al., 2023; Hong et al., 2024; Wu et al., 2024). Because text-based communication adds token generation and re-encoding overhead and may discard internal information, latent-communication methods pass embeddings (Pham et al., 2024), activations (Ramesh and Li, 2025), hidden states (Du et al., 2026), or key-value caches (Shi et al., 2026), with recent work further studying joint latent collaboration (Zou et al., 2025), cache transfer (Fu et al., 2025), learned messages (Zheng et al., 2025), and cross-model conversion (Liu et al., 2026a). Meanwhile, model assignment (Yue et al., 2025), communication topology (Li et al., 2024), model diversity (Chen et al., 2024), and interaction design (Liang et al., 2024) affect both performance and cost, creating a practical need to replace agent models while preserving existing task divisions and communication relations.

However, existing latent-communication methods mainly build communication interfaces for fixed sending and receiving models, leaving interface reuse after model replacement unresolved. Drawing on model stitching (Bansal et al., 2021) and low-rank adaptation (Hu et al., 2022), Lat-Bridge establishes a shared communication protocol. This allows LatBridge to extend latent communication to changing agent-model configurations, with unaffected and matching interfaces reused after model replacement.

## 3 Background and Motivation

This section examines the difficulty of reusing interfaces from existing latent-communication methods after model replacement and motivates our design. Section 3.1 states this interface reuse problem, and Section 3.2 examines it through a pilot study.

## 3.1 Problem Statement

As reviewed in Section 2, latent communication enables agents to exchange continuous representations, while multi-agent systems may need to replace agent models. However, interfaces in latentcommunication methods are trained for fixed sending and receiving models. Model replacement can change the representations these interfaces receive or produce, but prior work has not tested whether the original interfaces remain usable.

This leaves an unexamined problem: After the model assigned to an agent in a multi-agent system is replaced, can a latent-communication interface trainedfor the original model combination be reused directly and continue to pass task-relevant information to the target agent?

To answer this question, we model a directed communication relation in a multi-agent system as one-way communication between two models. A multi-agent system can be represented as a graph whose nodes are agents and whose directed edges are communication relations. Applying this model to each directed communication relation extends the analysis to multi-agent systems with different sizes and communication topologies. The sending model corresponds to the source agent, and the receiving model corresponds to the target agent, while both agents retain their task roles in the system. The next subsection examines this problem through a pilot study of model replacement.

<table><tr><td>Task</td><td>Target</td><td>Original</td><td>No communication</td><td>Replacement</td></tr><tr><td rowspan="2">MATH</td><td>Qwen</td><td> $1 2 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $0 . 5 0 { \pm } 0 . 0 0$ </td><td> $0 . 6 7 { \pm } 0 . 2 4$ </td></tr><tr><td>Llama</td><td> $5 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $0 . 5 0 { \pm } 0 . 4 1$ </td></tr><tr><td rowspan="2">ALFWorld</td><td>Qwen</td><td> $2 3 . 6 3 \pm 2 . 5 3$ </td><td> $0 . 2 5 { \pm } 0 . 3 6 $ </td><td> $0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td></tr><tr><td>Llama</td><td> $1 9 . 4 1 \pm 4 . 0 9$ </td><td> $0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td></tr></table>

Table 1: Pilot study of source-agent model replacement based on Interlat (Du et al., 2026). Accuracies are percentages reported as mean ± standard deviation.

## 3.2 Pilot Study

Experimental setting. We use Interlat as a representative latent-communication method (Du et al., 2026). We use Qwen2.5-0.5B-Instruct (Yang et al., 2024) and Llama-3.2-1B-Instruct (Grattafiori et al., 2024) as the source- and target-agent models and evaluate their communication on MATH (Hendrycks et al., 2021) and ALF-World (Shridhar et al., 2021). We compare the original model combination, source-agent model replacement with the target-agent model and original interface fixed, and no communication; complete condition definitions and evaluation details are provided in Appendix F.

Results. Under the original model combinations, latent communication provides clear performance gains on both tasks. As shown in Table 1, sourceagent model replacement retains only 1.5–10% of the original communication gain on MATH and none of the original gain on ALFWorld.

Analysis. Because the target-agent model, task setting, and original interface remain fixed, the decline shows that the replacement model’s representations are incompatible with the original interface. The original interface therefore cannot be directly reused after model replacement. Supporting model replacement requires separating communication from a fixed endpoint-model combination.

Motivation. Existing latent-communication methods rely on interfaces trained for fixed model pairs, motivating separate representation mappings so that replacing one endpoint leaves the mapping on the other side unchanged. A natural solution is to establish a shared communication protocol implemented through a shared communication component used by all agents in a multi-agent system. For each directed communication relation, separate source-side and target-side interfaces divide the model-pair-dependent conversion across the two sides of this component. When an agent’s model is replaced, only its connection to this component changes, so the interface on the unchanged side can be reused and a new interface is trained only when no matching interface exists. This shared protocol forms the basis of LatBridge in Section 4.

## 4 LatBridge

Building on the motivation in Section 3, LatBridge supports agent model replacement through a shared communication component and reusable interfaces, while reducing Bridge processing time. Section 4.1 gives an overview of the method and its workflow after model replacement. Section 4.2 describes the interfaces and their use after model replacement. Section 4.3 explains how to speed up communication under the new model configuration.

## 4.1 Overview

LatBridge supports model replacement through a shared communication protocol implemented by the Bridge. Figure 2 shows the workflow after agent model replacement: interface retrieval and composition, training and storage of missing interfaces, and latent communication around the Bridge. The following paragraphs explain these steps and how communication is accelerated. Section 4.2.1 details the interface structure, and Section 4.2.2 explains interface use after model replacement. Section 4.3 describes communication acceleration.

Interface retrieval and composition finds matching interfaces for the new configuration $( s , b , t )$ . Here, s, b, and t denote the source-agent, Bridge, and target-agent models, with model functions $f _ { s } , f _ { b } ,$ and $f _ { t }$ . The Bridge contains an input interface $\phi _ { s , b } ,$ a frozen Bridge model, and an output interface $\psi _ { b , t }$ . The input and output interfaces connect the source and target models to the Bridge model, separating the representation conversion into two adjacent mappings. Replacement changes only one adjacent connection, so the system reuses the unchanged interface and retrieves a matching interface for the new model. Matching interfaces from other communication relations in the system can be reused. Once both interfaces are available, they are composed through the fixed Bridge model.

Training missing interfaces supplies a mapping when no stored interface matches the adjacent model pair, with all models frozen and only the missing interface trained. The input and output interfaces are trained in separate stages. The input interface is trained to let the Bridge model read source-agent representations, while the output interface is trained to let the target-agent model read the Bridge output. The interface is stored by its adjacent models for reuse and combined with the unchanged interface.

Latent communication around the Bridge transfers source-agent information to the target agent after model replacement. Let $x _ { s } .$ , p<sub>b</sub>, p<sub>t</sub>, and $\hat { y }$ denote the source input, Bridge-side task prompt, fixed target instruction, and final output. The operation $\mathcal { P } _ { L }$ controls the length of source states $H _ { s } = f _ { s } ( x _ { s } )$ under communication budget $L$ The input interface $\phi _ { s , b }$ maps these states into Bridge-model inputs. Conditioned on these inputs and $p _ { b } .$ $f _ { b }$ generates a sequence. The output interface $\psi _ { b , t }$ maps the hidden states at generated positions into target-side continuous input. The target model $f _ { t }$ generates $\hat { y }$ using this input and $p _ { t }$

Efficient communication reduces the latency added by token-by-token Bridge generation after model replacement. LatBridge prepares a vocabulary mapping and output interface, then uses embedding lookup to construct target-side continuous input directly from the Bridge-side task prompt without this generation step. Section 4.3 explains this procedure and its relation to the complete communication path.

For example, consider replacing an agent’s model in a multi-agent system using LatBridge. For its incoming and outgoing communication, the system selects matching interfaces, reuses the unchanged sides, and trains only missing mappings. In the accelerated mode, the target-matched mapping and output interface construct continuous input without autoregressive Bridge generation. This limits retraining, supports communication with the new model, and reduces communication time.

## 4.2 Interface Design

Interface design addresses representation differences after model replacement and allows unaffected interfaces to be reused. Section 4.2.1 describes the Bridge and interface structure, and Section 4.2.2 explains interface retrieval and composition, training, storage, and communication.

![](images/3cda61843b410e5ee8499a759c80ef5f57413f2a5bd47730c2fde2c78765cf07.jpg)  
Figure 2: Workflow of LatBridge. (1) Matching interfaces are retrieved and composed after agent model replacement. (2) Only missing interfaces are trained and stored. (3) Latent communication around the Bridge uses either the basic or accelerated mode.

## 4.2.1 Interface Structure

The Bridge connects endpoint models through a fixed Bridge model, providing a shared connection point for interface reuse after model replacement. The input interface maps length-controlled source states $\bar { H } _ { s }$ into representations the Bridge model can consume. The output interface maps Bridge hidden states $H _ { b }$ into target-side representations $U _ { t }$ . Each interface connects one adjacent model pair, so a single interface does not need to learn a mapping that depends on the complete endpointmodel combination. The following describes how these interfaces handle representation differences between adjacent models.

Models differ in hidden dimension, value distribution, and internal representation, so their hidden states do not correspond directly. Following model stitching and cross-model representation research (Bansal et al., 2021; Huh et al., 2024), Lat-Bridge uses separate interfaces to learn mappings between the adjacent model pairs. For an input representation h, a low-rank interface is written as

$$
g ( h ) = W _ { 0 } h + \frac { \alpha } { r } B A \mathrm { D r o p o u t } ( h ) ,\tag{1}
$$

where $W _ { 0 }$ is the fixed base mapping, A and $B$ are trainable low-rank matrices, r is the rank, and α is the scaling coefficient. This parameterization follows the basic form of low-rank adaptation (Hu et al., 2022) and limits trainable parameters to the interfaces between frozen models.

## 4.2.2 Interface Use After Model Replacement

After agent model replacement, LatBridge connects the new configuration while retaining the unchanged interface. The following describes interface retrieval and composition, training of missing interfaces, storage, and communication.

Interface retrieval and composition identifies reusable and missing interfaces for the new configuration. The system retrieves matching input and output interfaces from its stored collection. For configuration $( s , b , t )$ , the Bridge maps source-agent hidden states to target-side input representations as

$$
G _ { s , b , t } : = \psi _ { b , t } \circ f _ { b } \circ \phi _ { s , b } .\tag{2}
$$

With the Bridge model fixed, Equation 2 limits the change to the interface adjacent to the replaced model. Replacing the source-agent model requires a matching $\phi _ { s , b }$ while $\psi _ { b , t }$ is reused. Replacing the target-agent model instead requires a matching $\psi _ { b , t }$ while $\phi _ { s , b }$ is reused. Stored interfaces, including those from other communication relations, can be reused without retraining when their adjacent models and required processing match.

Training missing interfaces provides mappings unavailable for the new model configuration. Only the missing interface is trained, with all three models frozen. For a missing input interface, sourceagent hidden states are mapped through $\phi _ { s , b }$ and supplied with the Bridge-side task prompt, training the input interface to support Bridge-model generation. For a missing output interface, the frozen Bridge model produces teacher-forced task states $H _ { b } ^ { \mathrm { t f } }$ , which $\psi _ { b , t }$ maps into continuous input for the target-agent model. Let $y$ denote the training target sequence, with the two interfaces trained using the following objectives

$$
\mathcal { L } _ { \mathrm { i n } } : = - \log p _ { f _ { b } } \left( y ~ \vert ~ p _ { b } , \phi _ { s , b } ( \bar { H } _ { s } ) \right) ,\tag{3}
$$

$$
\mathcal { L } _ { \mathrm { o u t } } : = - \log p _ { f _ { t } } \Big ( y \mid p _ { t } , \psi _ { b , t } ( H _ { b } ^ { \mathrm { t f } } ) \Big ) .\tag{4}
$$

Each training stage updates only its corresponding interface, so model replacement does not require retraining the complete communication path.

Interface storage makes trained mappings available for reuse across communication relations and later model replacements. Interfaces are added to the shared collection and indexed by their adjacent models and, when needed, processing mode. Complete record fields are provided in Section E.2.

Latent communication around the Bridge resumes communication once matching interfaces are available after replacement. The system follows Equation 2, with all models and interfaces frozen during inference.

## 4.3 Efficient Communication

Efficient communication reduces Bridge processing time after model replacement. Once interfaces are available, the full Bridge path, called the basic mode, preserves source-dependent communication but adds token-by-token generation latency. To avoid this generation, LatBridge directly constructs target-side continuous input using a prepared vocabulary mapping and output interface. This procedure is called the accelerated mode and retains the same target-side input convention. The following describes the training-time preparation and communication-time processing.

Training-time preparation establishes the vocabulary correspondence and trains the output interface $\psi _ { b , t } ^ { \mathrm { t a b } }$ for each Bridge-model and target-agentmodel pair. Let $c _ { b }$ denote the length-controlled Bridge-side token sequence. When the two models share a vocabulary, $c _ { t } = c _ { b }$ . When they use different vocabularies, the system precomputes a vocabulary mapping $V _ { b , t }$ that converts each Bridge token into one or more target-side tokens, producing $c _ { t } .$ The corresponding output interface $\psi _ { b , t } ^ { \mathrm { t a b } }$ is trained with the target-model cross-entropy objective in Equation 4, using $E _ { t } ( c _ { t } )$ instead of teacher-forced Bridge hidden states. Here, $E _ { t }$ is the target-agent model’s embedding table. The trained $\psi _ { b , t } ^ { \mathrm { t a b } }$ follows the storage and reuse procedure in Section 4.2.2, indexed by the Bridge model, target-agent model, and processing mode.

Communication-time processing constructs target-side continuous input using the prepared mapping and output interface. The system tokenizes $p _ { b }$ and applies budget L to obtain $c _ { b } .$ It obtains $c _ { t }$ using the vocabulary correspondence above and queries $E _ { t }$ . The output interface transforms these embeddings as

$$
U _ { t } ^ { \mathrm { t a b } } = \psi _ { b , t } ^ { \mathrm { t a b } } \left( E _ { t } ( c _ { t } ) \right) .\tag{5}
$$

The target agent generates its output from $p _ { t }$ and $U _ { t } ^ { \mathrm { t a b } }$ . The budget limits the communication sequence length. Token selection, padding masks,

vocabulary mapping, and caching details are provided in Section E.3.

## 5 Experiments

This section evaluates how LatBridge supports efficient and stable agent model replacement through task accuracy after replacement, communication efficiency, and the factors affecting its performance. We study the following three research questions.

• RQ1: Task Accuracy. After agent model replacement, can LatBridge maintain task accuracy using matched interfaces?

• RQ2: Communication Efficiency. After agent model replacement, can LatBridge reduce processing time while maintaining task accuracy?

• RQ3: Component and Factor Analysis. How do the interface components, training procedure, communication content, communication budget, and representation space affect the performance of LatBridge after model replacement?

Section 5.1, Section 5.2, Section 5.3, and Section 5.4 describe the shared setup and answer the three research questions, respectively.

## 5.1 Experimental Setup

This section describes the settings shared by all experiments. Each subsequent section introduces only the additional controls and interventions required for its research question.

Models and configurations. Our main experiments use Qwen2.5-0.5B-Instruct and Llama-3.2-1B-Instruct, and our scaling experiments use Qwen2.5-7B and Llama-3.1-8B-Instruct (Yang et al., 2024; Grattafiori et al., 2024). At each scale, the source-agent model, Bridge model, and targetagent model are selected from the two model families, resulting in eight configurations. Agent model replacement changes only the source-agent model or target-agent model, while the Bridge model remains fixed within each comparison.

Data and tasks. We evaluate MATH using normalized exact match of the final answer (Hendrycks et al., 2021), and ALFWorld using normalized exact match of the next action (Shridhar et al., 2021). For both tasks, the target agent receives only a fixed task instruction and the continuous input produced by the Bridge, without direct access to the original task input. Dataset splits and scoring details are provided in Appendix G.

<table><tr><td rowspan="3">Bridge</td><td rowspan="3">Source</td><td rowspan="3">Target</td><td colspan="4">Qwen2.5-0.5B / Llama-3.2-1B</td><td colspan="4">Qwen2.5-7B / Llama-3.1-8B</td></tr><tr><td colspan="2">Basic mode</td><td colspan="2">Accelerated mode</td><td colspan="2">Basic mode</td><td colspan="2">Accelerated mode</td></tr><tr><td>MATH</td><td>ALFWorld</td><td>MATH</td><td>ALFWorld</td><td>MATH</td><td>ALFWorld</td><td>MATH</td><td>ALFWorld</td></tr><tr><td rowspan="5">Qwen</td><td>Qwen</td><td>Qwen</td><td>16.0</td><td>33.6</td><td>16.5</td><td>24.6</td><td>30.0</td><td>26.9</td><td>57.0</td><td>45.5</td></tr><tr><td>Qwen</td><td>Llama</td><td>11.0</td><td>32.8</td><td>16.0</td><td>42.5</td><td>39.0</td><td>24.6</td><td>44.0</td><td>44.8</td></tr><tr><td>Llama</td><td>Qwen</td><td>14.0</td><td>31.3</td><td>16.5</td><td>24.6</td><td>28.5</td><td>30.6</td><td>57.0</td><td>45.5</td></tr><tr><td>Llama</td><td>Llama</td><td>18.0</td><td>21.6</td><td>16.0</td><td>42.5</td><td>38.5</td><td>27.6</td><td>44.0</td><td>44.8</td></tr><tr><td>Qwen</td><td>Qwen</td><td>16.5</td><td>22.4</td><td>19.5</td><td>29.9</td><td>27.5</td><td>40.3</td><td>59.0</td><td>44.8</td></tr><tr><td rowspan="4">Llama</td><td>Qwen</td><td>Llama</td><td>14.5</td><td>23.9</td><td>20.0</td><td>49.3</td><td>17.0</td><td>29.9</td><td>42.0</td><td>44.0</td></tr><tr><td>Llama</td><td>Qwen</td><td>12.5</td><td>26.1</td><td>19.5</td><td>29.9</td><td>29.5</td><td>23.9</td><td>59.0</td><td>44.8</td></tr><tr><td>Llama</td><td>Llama</td><td>16.0</td><td>26.9</td><td>20.0</td><td>49.3</td><td>19.5</td><td>20.1</td><td>42.0</td><td>44.0</td></tr><tr><td>Mean over eight configurations</td><td></td><td>14.8</td><td>27.3</td><td>18.0</td><td>36.6</td><td>28.7</td><td>28.0</td><td>50.5</td><td>44.8</td></tr></table>

Table 2: Task accuracy (%) across eight model configurations at two model scales, grouped by Bridge model.

Interface training. All language models remain frozen, and the input and output interfaces are trained in two stages following Section 4.2.2. The two stages use the losses defined in Equation 3 and Equation 4, respectively, with a communication budget of $L = 1 2 8$ . The complete training settings are provided in Appendix H.

Evaluation. We use controlled experiments because existing methods do not provide a suitable direct baseline for agent model replacement. The Interlat pilot study provides an accuracy reference, while no-communication and ablation conditions assess the contribution of communication and individual components. The no-communication condition removes the Bridge output and retains only the fixed task instruction of the target agent. Communication efficiency is measured using synchronized GPU timing, with the timing protocol provided in Appendix B. The message replacement, communication budget, and representation analyses use trained interfaces without further model training.

## 5.2 RQ1: Task Accuracy

This section evaluates whether LatBridge maintains task accuracy after agent model replacement through interface reuse and missing-interface training. The results show that it supports all tested configurations at both model scales and in both the basic and accelerated modes.

Settings and controls. In each comparison, we replace one agent’s model while keeping the Bridge model and the other agent’s model fixed. The system reuses the interface on the unchanged side and selects or trains only the interface adjacent to the new model. The complete replacement operations are provided in Appendix A.1.

Results. Table 2 reports the main small-model experiments and the large-model scaling experiments. Across the four model-scale–task aggregates, average accuracy ranges from 14.8% to 28.7% in the basic mode and from 18.0% to 50.5% in the accelerated mode. All large-model configurations in the basic mode achieve nonzero task accuracy, and the accelerated mode performs better in all 16 configuration–task comparisons.

Analysis. Compared with the substantial degradation caused by retaining the original interface in Table 1, matched interfaces restore effective communication under the new agent-model configurations. Changing the source-agent model changes the basic-mode results, showing that this mode preserves source-dependent information, whereas the accelerated mode provides a more stable target-side continuous input. These results support interface selection, interface reuse, and missing-interface training as ways for LatBridge to maintain task accuracy after model replacement while retaining the interface on the unchanged side.

## 5.3 RQ2: Communication Efficiency

This section evaluates whether LatBridge reduces Bridge processing time in the accelerated mode while maintaining task accuracy, once interfaces for the new models are available.

Settings. We compare the basic and accelerated modes under the shared models, data, interfaces, and communication budget in Section 5.1. Timing covers Bridge processing and output adaptation; the complete boundary and hardware settings are provided in Appendix B.

Results. The aggregate timing results are reported in Table 5, and the task accuracies are shown in Table 2. Across both tasks and model scales, average communication time ranges from 559.84 ms to 2,256.96 ms in the basic mode and from 0.334 ms to 0.367 ms in the accelerated mode. The accelerated mode reduces average communication time by approximately 1,669× to 6,149×, while maintaining or improving average task accuracy in all four model-scale–task aggregates.

Analysis. Token-by-token generation with the complete Bridge model accounts for most of the Bridge processing latency. By avoiding these autoregressive forward passes, LatBridge reduces the processing overhead of the shared communication component in the accelerated mode while maintaining or improving average task accuracy. Thus, matched interfaces support communication after model replacement, and acceleration reduces Bridge processing time during communication.

## 5.4 RQ3: Component and Factor Analysis

This section examines which designs in LatBridge support effective communication after model replacement and how communication content, budget, and representation differences affect task performance under the new configurations.

Module ablations. We ablate the interface components and training procedure in the small-model basic-mode setting while keeping the remaining experimental conditions fixed. As reported in Table 7, removing either interface or using coupled output training reduces accuracy on both tasks, while joint optimization has task-dependent effects and updating the Bridge model provides no consistent improvement. These results show that separate interfaces and decoupled two-stage training support effective communication and provide a basis for adapting the adjacent interface while reusing the unchanged interface after model replacement. The complete variants, settings, and learned-parameter counts are provided in Appendix C.1.

Communication content and budget. We replace increasing proportions of the Bridge-output states before the output interface while keeping the models, interfaces, and decoding settings fixed. As shown in Figure 3, accuracy decreases consistently across both tasks and target-agent families as the replacement ratio increases, showing that the Bridge output carries sample-specific information used by the target agent.

We also vary the communication budget by retaining different proportions of the Bridge-output states. Accuracy decreases for most configurations as fewer states are retained, with different model pairs showing different rates of degradation. The communication budget therefore controls both processing cost and the amount of sample information available to the target agent. The new model configuration must therefore balance communication length and task accuracy. The budget curves and intervention details are provided in Appendix C.3.

![](images/0271c7a9e71100fde1aad3280a51266898896e90c4600ced6e277d6738a02f70.jpg)  
Figure 3: Task accuracy under Bridge-output replacement in the basic mode. Thin curves show model configurations, labeled by source and Bridge models. Thick curves show target-family means with bootstrap 95% confidence bands.

Representation analysis. Nearest-neighbor retrieval and linear probing both perform above their respective baselines, showing that the representations produced by both processing modes retain sample information. Although representations of the same sample have low raw similarity across the two modes, linear alignment recovers part of their correspondence. These results indicate that performance differences arise mainly from differences between representation spaces and from how well the target agent reads the transformed representations, rather than from a complete loss of task information. This also shows why model replacement requires matching the new model’s representation space so that the target agent can use the sample information carried by communication. Complete retrieval, probing, alignment, and receiver-space results are provided in Appendix C.

## 6 Conclusion

This work studies interface reuse in existing latentcommunication methods after agent model replacement. We presented LatBridge, which establishes a shared communication protocol through the Bridge and trains the interfaces on its two sides in two stages. After model replacement, it reuses unchanged and matching interfaces and trains only missing ones. In the accelerated mode, LatBridge avoids autoregressive Bridge generation to reduce communication processing time. Experiments at two model scales show that LatBridge maintains effective communication after agent model replacement and improves communication efficiency. Ablation and representation analyses confirm the roles of the interfaces, training procedure, and communicated representations in task performance.

## Limitations

Interpretability of continuous representations. The analyses in Section 5.4 and Appendix C examine whether sample information is retained and used, but do not explain the meaning of representation positions. Task outputs alone may not reveal whether errors arise from source-agent generation, interface mapping, or target-agent interpretation. More direct tools are needed to explain continuous messages and locate communication errors.

Dependence on a fixed Bridge model. Interface composition and reuse in Section 4.2.2 assume that the Bridge model remains fixed, allowing the interface on the unchanged side to be reused after agent model replacement. If the Bridge model changes, the representation mappings on both sides may need adaptation, and the original interfaces may no longer be directly reusable. This work therefore addresses agent model replacement with a fixed shared communication component, rather than interface transfer after a Bridge-model update.

Model coverage. As described in Section 5.1, our experiments cover two model families and two model scales to evaluate interface reuse, task accuracy, and communication efficiency across model configurations. These configurations include differences in hidden dimension and representation space, but do not cover a wider range of architectures, training approaches, or model-size differences. Other models may require different representation mappings, so the applicability of the current interface structure to broader model combinations remains to be evaluated.

Multi-agent communication scale. Section 3.1 explains how individual communication relations form systems with different sizes and topologies, providing a basis for interface reuse across relations. Our experiments primarily test replacement within a single relation. Successive agent model replacements and multi-hop communication may introduce interactions and error accumulation, whose effects on overall task performance require systemlevel evaluation.

Tasks and evaluation. We evaluate task accuracy after model replacement through mathematical reasoning and offline action prediction, with data and scoring details in Appendix G. These evaluations test whether communicated representations support final-answer or next-action generation, but do not cover open-ended tasks or complete interactions with continuing environment feedback. In particular, offline next-action accuracy on ALF-World does not directly measure full-task success. Multi-step decision-making and error recovery after model replacement remain to be evaluated through interactive tasks.

## Ethical Considerations

Models and data. Our evaluation uses publicly available pretrained models and the MATH and ALFWorld benchmarks. MATH contains competition problems, and ALFWorld is a simulated household environment. This study involves no human participants, new human annotation, or new data collection. Model weights and datasets remain subject to their original licenses and terms of use.

Safety and privacy. Latent inter-agent messages may be difficult to inspect with text-only moderation tools. Latent communication should not be treated as privacy protection because states may encode task or user information. After model replacement, deployed systems should reassess safety and retain human oversight for high-impact actions. Our offline next-action evaluation on ALFWorld does not establish safety for physical agents.

AI use statement. We used AI tools to assist with drafting, translation, language editing, code editing, and figure preparation. The authors take responsibility for the accuracy and integrity of the paper, including its claims, results, citations, and final content.

## References

Yamini Bansal, Preetum Nakkiran, and Boaz Barak. 2021. Revisiting model stitching to compare neural representations. In Advances in Neural Information Processing Systems, volume 34.

Luís Brito and Carlos Baquero. 2026. When latent agents lie: KV-cache integrity in multi-agent LLM collaboration. arXiv preprint arXiv:2606.28958.

Shuochen Chang, Tong Bai, Xiaofeng Zhang, Qianli Ma, Qingyang Liu, Zhaohe Liao, Yibo Miao, and Li Niu. 2026. Unlocking the black box of latent reasoning: An interpretability-guided approach to intervention. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 34019–34032. Association for Computational Linguistics.

Jingdi Chen, Hanquing Yang, Zongjun Liu, and Carlee Joe-Wong. 2026. The five ws of multi-agent communication: Who talks to whom, when, what, and why. Transactions on Machine Learning Research.

Justin Chen, Swarnadeep Saha, and Mohit Bansal. 2024. ReConcile: Round-table conference improves reasoning via consensus among diverse LLMs. In Proceedings ofthe 62nd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 7066–7085. Association for Computational Linguistics.

Jeffrey Cheng and Benjamin Van Durme. 2024. Compressed chain of thought: Efficient reasoning through dense representations. arXiv preprint arXiv:2412.13171.

Jiaming Cheng, Subhransu Das, and Rajiv Ramnath. 2026. When does latent communication pay? a causal audit of relayed KV caches in multi-agent LLMs. arXiv preprint arXiv:2608.04893.

Zhuoyun Du, Runze Wang, Huiyu Bai, Zouying Cao, Xiaoyong Zhu, Yu Cheng, Bo Zheng, Wei Chen, and Haochao Ying. 2026. Enabling agents to communicate entirely in latent space. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 27106–27129. Association for Computational Linguistics.

Jacob Fein-Ashley, Dhruv Parikh, Rajgopal Kannan, and Viktor Prasanna. 2025. Mixture of thoughts: Learning to aggregate what experts think, not just what they say. arXiv preprint arXiv:2509.21164.

Tianyu Fu, Zihan Min, Hanling Zhang, Jichao Yan, Guohao Dai, Wanli Ouyang, and Yu Wang. 2025. Cache-to-cache: Direct semantic communication between large language models. arXiv preprint arXiv:2510.03215.

Sachin Goyal, Ziwei Ji, Ankit Singh Rawat, Aditya Krishna Menon, Sanjiv Kumar, and Vaishnavh Nagarajan. 2024. Think before you speak: Training language models with pause tokens. In The Twelfth International Conference on Learning Representations.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. 2024. The Llama 3 herd of models. arXiv preprint arXiv:2407.21783.

Shibo Hao, Sainbayar Sukhbaatar, DiJia Su, Xian Li, Zhiting Hu, Jason Weston, and Yuandong Tian. 2025. Training large language models to reason in a continuous latent space. In Second Conference on Language Modeling.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. 2021. Measuring mathematical problem solving with the MATH dataset. In Advances in

Neural Information Processing Systems Datasets and Benchmarks Track.

Sirui Hong, Mingchen Zhuge, Jonathan Chen, Xiawu Zheng, Yuheng Cheng, Jinlin Wang, Ceyao Zhang, Zili Wang, Steven Ka Shing Yau, Zijuan Lin, Liyang Zhou, Chenyu Ran, Lingfeng Xiao, Chenglin Wu, and Jürgen Schmidhuber. 2024. MetaGPT: Meta programming for a multi-agent collaborative framework. In The Twelfth International Conference on Learning Representations.

Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2022. LoRA: Low-rank adaptation of large language models. In The Tenth International Conference on Learning Representations.

Minyoung Huh, Brian Cheung, Tongzhou Wang, and Phillip Isola. 2024. Position: The platonic representation hypothesis. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 20617–20642. PMLR.

Huiqiang Jiang, Qianhui Wu, Chin-Yew Lin, Yuqing Yang, and Lili Qiu. 2023. LLMLingua: Compressing prompts for accelerated inference of large language models. In Proceedings ofthe 2023 Conference on Empirical Methods in Natural Language Processing, pages 13358–13376. Association for Computational Linguistics.

Simon Kornblith, Mohammad Norouzi, Honglak Lee, and Geoffrey Hinton. 2019. Similarity of neural network representations revisited. In Proceedings of the 36th International Conference on Machine Learning, volume 97 of Proceedings ofMachine Learning Research, pages 3519–3529. PMLR.

Boris Kriuk and Logic Ng. 2025. Q-KVComm: Efficient multi-agent communication via adaptive KV cache compression. arXiv preprint arXiv:2512.17914.

Chong Li, Jiajun Zhang, and Chengqing Zong. 2025. TokAlign: Efficient vocabulary adaptation via token alignment. In Proceedings ofthe 63rd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 4109–4126. Association for Computational Linguistics.

Guohao Li, Hasan Abed Al Kader Hammoud, Hani Itani, Dmitrii Khizbullin, and Bernard Ghanem. 2023. CAMEL: Communicative agents for “mind” exploration of large language model society. In Advances in Neural Information Processing Systems, volume 36.

Yunxuan Li, Yibing Du, Jiageng Zhang, Le Hou, Peter Grabowski, Yeqing Li, and Eugene Ie. 2024. Improving multi-agent debate with sparse communication topology. In Findings of the Association for Computational Linguistics: EMNLP 2024, pages 7281– 7294. Association for Computational Linguistics.

Tian Liang, Zhiwei He, Wenxiang Jiao, Xing Wang, Yan Wang, Rui Wang, Yujiu Yang, Shuming Shi, and Zhaopeng Tu. 2024. Encouraging divergent thinking in large language models through multi-agent debate. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 17889–17904. Association for Computational Linguistics.

Jiyao Liu, Qi Zhang, Yaoyi Jia, Ziwen Kan, and Song Wang. 2026a. Dual-cache latent space communication between heterogeneous language models. arXiv preprint arXiv:2608.20617.

Luyang Liu, Jonas Pfeiffer, Jiaxing Wu, Jun Xie, and Arthur Szlam. 2024. Deliberation in latent space via differentiable cache augmentation. arXiv preprint arXiv:2412.17747.

Xiaoze Liu, Ruowang Zhang, Weichen Yu, Siheng Xiong, Liu He, Feijie Wu, Hoin Jung, Matt Fredrik son, Xiaoqian Wang, and Jing Gao. 2026b. The vision wormhole: Latent-space communication in heterogeneous multi-agent systems. arXiv preprint arXiv:2602.15382.

Yingzhuo Liu. 2026. Beyond tokens: A unified framework for latent communication in LLM-based multiagent systems. arXiv preprint arXiv:2606.05711.

Jacob Pfau, William Merrill, and Samuel R. Bowman. 2024. Let’s think dot by dot: Hidden computation in transformer language models. arXiv preprint arXiv:2404.15758.

Chau Pham, Boyi Liu, Yingxiang Yang, Zhengyu Chen, Tianyi Liu, Jianbo Yuan, Bryan A. Plummer, Zhaoran Wang, and Hongxia Yang. 2024. Let models speak ciphers: Multiagent debate through embeddings. In The Twelfth International Conference on Learning Representations.

Vignav Ramesh and Kenneth Li. 2025. Communicating activations between language model agents. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings ofMachine Learning Research, pages 51094–51116. PMLR.

Siyu Ren, Qi Jia, and Kenny Zhu. 2023. Context compression for auto-regressive transformers with sentinel tokens. In Proceedings ofthe 2023 Conference on Empirical Methods in Natural Language Processing, pages 12860–12867. Association for Computational Linguistics.

Xu Shen, Yixin Liu, Yiwei Dai, Yili Wang, Rui Miao, Yue Tan, Shirui Pan, and Xin Wang. 2025a. Understanding the information propagation effects of communication topologies in LLM-based multi-agent systems. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 12347–12361. Association for Computational Linguistics.

Zhenyi Shen, Hanqi Yan, Linhai Zhang, Zhanghao Hu, Yali Du, and Yulan He. 2025b. CODI: Compressing chain-of-thought into continuous space via selfdistillation. In Proceedings ofthe 2025 Conference on Empirical Methods in Natural Language Processing, pages 677–693. Association for Computational Linguistics.

Xiangyu Shi, Marco Chiesa, Gerald Q. Maguire, Jr., and Dejan Kostic. 2026. KVComm: Enabling efficient LLM communication through selective KV sharing. In The Fourteenth International Conference on Learning Representations.

Mohit Shridhar, Xingdi Yuan, Marc-Alexandre Côté, Yonatan Bisk, Adam Trischler, and Matthew Hausknecht. 2021. ALFWorld: Aligning text and embodied environments for interactive learning. In The Ninth International Conference on Learning Representations.

Avyav Kumar Singh, Yen-Chen Wu, Alexandru Cioba, Alberto Bernacchia, and Davide Buffelli. 2026. Cross-tokenizer LLM distillation through a byte-level interface. In Proceedings of the 2nd Workshop on Customizable NLP: Progress and Challenges in Customizing NLP for a Domain, Application, Group, or Individual, pages 84–96. Association for Computational Linguistics.

Andries Petrus Smit, Nathan Grinsztajn, Paul Duckworth, Thomas D. Barrett, and Arnu Pretorius. 2024. Should we be going MAD? a look at multi-agent debate strategies for LLMs. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 45883–45905. PMLR.

Yichen Tang, Weihang Su, Yujia Zhou, Yiqun Liu, Min Zhang, Shaoping Ma, and Qingyao Ai. 2025. Augmenting multi-agent communication with state delta trajectory. In Proceedings ofthe 2025 Conference on Empirical Methods in Natural Language Processing, pages 10219–10240. Association for Computational Linguistics.

Minh-Phuc Truong, Hai An Vu, Tu Vu, Nguyen Thi Ngoc Diep, Linh Ngo Van, Thien Huu Nguyen, and Trung Le. 2025. EMO: Embedding model distillation via intra-model relation and optimal transport alignments. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 7594–7606. Association for Computational Linguistics.

Qineng Wang, Zihao Wang, Ying Su, Hanghang Tong, and Yangqiu Song. 2024. Rethinking the bounds of LLM reasoning: Are multi-agent discussions the key? In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 6106–6131. Association for Computational Linguistics.

Qingyun Wu, Gagan Bansal, Jieyu Zhang, Yiran Wu, Beibin Li, Erkang Zhu, Li Jiang, Xiaoyun Zhang,

Shaokun Zhang, Jiale Liu, Ahmed Hassan Awadallah, Ryen W. White, Doug Burger, and Chi Wang. 2024. AutoGen: Enabling next-gen LLM applications via multi-agent conversation. In First Conference on Language Modeling.

Heming Xia, Chak Tou Leong, Wenjie Wang, Yongqi Li, and Wenjie Li. 2025. TokenSkip: Controllable chain-of-thought compression in LLMs. In Proceedings ofthe 2025 Conference on Empirical Methods in Natural Language Processing, pages 3351–3363. Association for Computational Linguistics.

An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, et al. 2024. Qwen2.5 technical report. arXiv preprint arXiv:2412.15115.

John Seon Keun Yi, Aaron Mueller, and Dokyun Lee. 2026. Latent agents: A post-training procedure for internalized multi-agent debate. In Proceedings ofthe 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 15569–15602. Association for Computational Linguistics.

Yanwei Yue, Guibin Zhang, Boyang Liu, Guancheng Wan, Kun Wang, Dawei Cheng, and Yiyan Qi. 2025. MasRouter: Learning to route LLMs for multi-agent systems. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 15549–15572. Association for Computational Linguistics.

Yujia Zheng, Zhuokai Zhao, Zijian Li, Yaqi Xie, Mingze Gao, Lizhu Zhang, and Kun Zhang. 2025. Thought communication in multiagent collaboration. arXiv preprint arXiv:2510.20733.

Kunlun Zhu, Hongyi Du, Zhaochen Hong, Xiaocheng Yang, Shuyi Guo, Zhe Wang, Zhenhailong Wang, Cheng Qian, Xiangru Tang, Heng Ji, and Jiaxuan You. 2025. MultiAgentBench: Evaluating the collaboration and competition of LLM agents. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 8580–8622. Association for Computational Linguistics.

Xiaochen Zhu, Caiqi Zhang, Yizhou Chi, Tom Stafford, Nigel Collier, and Andreas Vlachos. 2026. Demystifying multi-agent debate: The role of confidence and diversity. In Findings of the Association for Computational Linguistics: ACL 2026, pages 33909–33930. Association for Computational Linguistics.

Jiaru Zou, Xiyuan Yang, Ruizhong Qiu, Gaotang Li, Katherine Tieu, Pan Lu, Ke Shen, Hanghang Tong, Yejin Choi, Jingrui He, James Zou, Mengdi Wang, and Ling Yang. 2025. Latent collaboration in multiagent systems. arXiv preprint arXiv:2511.20639.

## Appendix

## A RQ1: Task Accuracy

This section supplements Section 5.2 with replacement operations, result settings, and teacher-forced target loss. The task-accuracy results are in Table 2. Table 3 reports teacher-forced target loss, and Table 4 summarizes replacement operations.

## A.1 Replacement Settings

Replacement comparisons hold the Bridge model and the other agent’s model fixed. When the source-agent model is replaced, the output interface is reused. The system selects an input interface matched to the new source-agent model and current Bridge model, or trains this interface if it is unavailable. When the target-agent model is replaced, the input interface is reused. The system selects an output interface matched to the current Bridge model and new target-agent model, or trains this interface if it is unavailable. Matching interfaces can also be retrieved from other communication relations in the system.

Table 4 summarizes these operations and their trainable-parameter requirements. The replacement-without-adaptation condition retains an input interface matched to the original source model. The stored-interface selection condition uses an interface matched to the replacement model. The table reports the number of interface parameters trained for each operation.

## A.2 Task-Accuracy Results

Table 2 groups the eight configurations by Bridge model and reports the basic and accelerated modes at both model scales. The mean row gives an unweighted average across the eight configurations for each task and mode. Across tasks, small-model means range from 14.8% to 27.3% in the basic mode and from 18.0% to 36.6% in the accelerated mode. The corresponding large-model means range from 28.0% to 28.7% and from 44.8% to 50.5%. The accelerated mode improves the average accuracy in all four model-scale–task comparisons, while individual small-model configurations can favor the basic mode.

The large-model MATH results for a Qwen Bridge model and Llama target model use targettoken re-embedding of the sequence generated by the Bridge. This setting retains autoregressive Bridge generation but changes how its output is supplied to the target model. The corresponding ALFWorld results use the hidden-state output mapping. The message-replacement and budget experiments use the hidden-state mapping throughout, as specified in Section C.2.

<table><tr><td>Bridge</td><td>Source</td><td>Target</td><td>MATH</td><td>ALFWorld</td></tr><tr><td>Qwen</td><td>Qwen</td><td>Qwen</td><td>1.39</td><td>1.81</td></tr><tr><td>Qwen</td><td>Qwen</td><td>Llama</td><td>1.07</td><td>1.59</td></tr><tr><td>Qwen</td><td>Llama</td><td>Qwen</td><td>1.37</td><td>1.44</td></tr><tr><td>Qwen</td><td>Llama</td><td>Llama</td><td>1.08</td><td>1.52</td></tr><tr><td>Llama</td><td>Qwen</td><td>Qwen</td><td>1.21</td><td>1.33</td></tr><tr><td>Llama</td><td>Qwen</td><td>Llama</td><td>1.78</td><td>1.59</td></tr><tr><td>Llama</td><td>Llama</td><td>Qwen</td><td>1.21</td><td>1.22</td></tr><tr><td>Llama</td><td>Llama</td><td>Llama</td><td>1.80</td><td>1.52</td></tr><tr><td>Mean</td><td></td><td></td><td>1.36</td><td>1.50</td></tr></table>

Table 3: Mean teacher-forced target loss for the largemodel basic mode. Each value uses the same examples, interfaces, Bridge outputs, and target-side inputs as the corresponding large-model accuracy result in Table 2.

## A.3 Teacher-Forced Target Loss

Teacher-forced target loss supplements generatedanswer accuracy by measuring the target model’s cross-entropy on the reference output. The model receives the same fixed instruction and continuous representation as in the corresponding accuracy evaluation. The evaluation code first computes one mean answer-token loss for each example and then averages these example-level losses. The loss is computed only at answer-token positions.

Table 3 covers all eight large-model basic-mode configurations on both tasks using their corresponding accuracy-evaluation inputs. The mean losses are 1.36 on MATH and 1.50 on ALFWorld. Loss and exact-match accuracy describe different aspects of the output distribution, and the generated answer can remain unchanged as loss decreases. Appendix I illustrates this distinction with individual examples.

## B RQ2: Communication Efficiency

This section supplements Section 5.3 with the timing boundary, hardware settings, and complete latency results. The comparison measures communication once the interfaces required by the model configuration are available.

Table 5 reports the mean across eight configurations for each task and model scale. Table 6 gives the individual configuration means.

<table><tr><td>Replacement case</td><td>Input interface</td><td>Output interface</td><td>New interfaces</td><td>Small (K)</td><td>Large (K)</td></tr><tr><td>Source: no adaptation</td><td>Reuse mismatched</td><td>Reuse matched</td><td>0</td><td>0</td><td>0</td></tr><tr><td>Source: interface available</td><td>Select matched</td><td>Reuse matched</td><td>0</td><td>0</td><td>0</td></tr><tr><td>Target: interface available</td><td>Reuse matched</td><td>Select matched</td><td>0</td><td>0</td><td>0</td></tr><tr><td>Source: interface missing</td><td>Train matched</td><td>Reuse matched</td><td>1</td><td>57-131</td><td>229-262</td></tr><tr><td>Target: interface missing</td><td>Reuse matched</td><td>Train matched</td><td>1</td><td>57-131</td><td>229-262</td></tr></table>

Table 4: Interface operations after agent model replacement. Parameter counts are in thousands for rank-32 interfaces and exclude the frozen agent and Bridge-model backbones. Stored matching interfaces are reused, and only missing interfaces are trained.

![](images/76580c42f3e4c428cadeff4aa046aab53ef9dd8dee22695ae4f7b52102eb0108.jpg)  
Figure 4: Target-side representation-space analysis on held-out ALFWorld tasks. Columns separate target-agent families, while rows show target-native prompt references, Qwen Bridge outputs, and Llama Bridge outputs. Colors denote task type and marker shapes denote the source-agent model. PCA is fit separately within each target family. Each row uses an independent local scale with coordinate values retained on the axes.

## B.1 Timing Setup

The local timer measures input-interface processing, Bridge processing, and output adaptation, ending when the target-side continuous input is ready. Device synchronization is applied before and after the measured interval. Source-state collection, source generation, target prompt construction, target decoding, sequential model loading, and hostdevice transfer take place outside the measured interval. In the accelerated mode, prompt tokenization and preloading also take place outside this interval. All measurements use an NVIDIA GeForce RTX 3090, bfloat16 precision, batch size 1, and 20 warm-up runs. After warm-up, each configuration is evaluated over the fixed task test set.

## B.2 Timing Results

Across the four task-scale comparisons, the accelerated mode reduces mean local communication time from 559.84–2,256.96 ms to 0.334–0.367 ms. The reduction ranges from approximately 1,669× to 6,149×. Once the required mappings are prepared, directly constructing target-side continuous input saves the time spent on autoregressive Bridge generation.

Interlat’s reported timing is included as an external reference. Interlat reports end-to-end time, while LatBridge reports local Bridge processing time. Each reduction is computed within its own system and timing boundary. Task-accuracy comparisons for the two modes are reported separately in Table 2.

<table><tr><td>Scale Task</td><td></td><td>Before</td><td>After</td><td>Reduction</td></tr><tr><td>Interlat (end-to-end) 一</td><td>ALFWorld</td><td>9,190</td><td>390</td><td>23.6×</td></tr><tr><td>LatBridge (local Bridge)</td><td></td><td></td><td></td><td></td></tr><tr><td>Small</td><td>MATH</td><td>1,262.53</td><td>0.334</td><td>3,777×</td></tr><tr><td>Small</td><td>ALFWorld</td><td>559.84</td><td>0.335</td><td>1,669×</td></tr><tr><td>Large</td><td>MATH</td><td>2,256.96</td><td>0.367</td><td>6,149×</td></tr><tr><td>Large</td><td>ALFWorld</td><td>1,181.89</td><td>0.354</td><td>3,335×</td></tr></table>

Table 5: Mean communication latency (ms). LatBridge compares basic and accelerated modes, averaged over eight configurations. Interlat compares full and 8-step latent generation (Du et al., 2026). Reductions use each system’s own timing boundary.

## C RQ3: Component and Factor Analysis

This section expands the component and representation analyses in Section 5.4. We first examine the interface and training components, then the effects of message content and length, and finally the information retained in different representation spaces. Each experiment uses its stated model configuration and data without changing the remaining conditions.

Table 7 and Table 8 summarize the ablation and representation-analysis results. Figure 5 and Figure 4 show the communication-budget results and target-side representation spaces.

## C.1 Component and Training Ablations

The ablations test which parts of interface construction support task accuracy in the small-model basic mode. For MATH, the source and Bridge models are Qwen and the target model is Llama. For ALF-World, the source and target models are Qwen and the Bridge model is Llama. All variants use seed 0 and keep the training data, evaluation examples, epochs, learning rates, communication budget, and decoding settings fixed. The complete method uses the corresponding small-model basic-mode results in Table 2.

The five variants change one component or training choice at a time. Without two-stage training, the interfaces are jointly optimized. Without decoupled output training, the output interface’s training states change from separately collected teacherforced states to Bridge states produced by the current communication process. The input-interface and output-interface variants remove the respective trainable mapping. Without Bridge-model freezing, rank-32 low-rank parameters in the Bridge model’s query and value projections are also updated.

Table 7 reports task accuracy and the total number of trainable parameters for each variant. Removing either interface or using coupled output training reduces accuracy on both tasks. Joint optimization has task-dependent effects, while updating the Bridge model adds more than two million trainable parameters without a consistent accuracy improvement. The separate interfaces and training stages therefore support communication while keeping the Bridge model frozen.

## C.2 Communication Content

Message replacement tests whether the target model uses information specific to the current sample. This experiment uses the large models, a communication budget of 128, and the fixed evaluation sets described in Section G.1. Bridge hidden states are passed through the trained output interface in every configuration, including the Qwen-Bridge-to-Llama-target configurations. This keeps the intervention on the same hidden-state path.

Before the output interface, we replace randomly selected rows of a sample’s Bridge output with rows from another sample. The replacement sample is selected from the same MATH subject or ALFWorld task type and matched by sequence length. Its states are adjusted using row-wise rootmean-square matching. Replacement proportions are 0, 0.10, 0.25, 0.50, 0.75, 0.90, and 1.00. Models, interfaces, and greedy decoding remain fixed.

Figure 3 shows the results for perturbation seed 0. Target-family means average the four source– Bridge combinations. The plotted 95% confidence bands use 10,000 bootstrap resamples of task identifiers, keeping the configurations for each example together. For Qwen targets, macro-average accuracy decreases from 22.7% to 0.4% on MATH and from 30.4% to 7.5% on ALFWorld. For Llama targets, it decreases from 10.0% to 0.8% and from 21.8% to 6.5%, respectively. These changes show that the target output depends on the sample information carried by the message. A fully mismatched message and the no-communication condition can produce different results.

## C.3 Communication Budget

<table><tr><td rowspan="2">Scale</td><td rowspan="2">Bridge</td><td rowspan="2">Source</td><td rowspan="2">Target</td><td colspan="2">MATH</td><td colspan="2">ALFWorld</td></tr><tr><td>Basic mode</td><td>Accelerated mode</td><td>Basic mode</td><td>Accelerated mode</td></tr><tr><td>Small</td><td>Qwen</td><td>Qwen</td><td>Qwen</td><td>1545.78</td><td>0.257</td><td>651.82</td><td>0.257</td></tr><tr><td>Small</td><td>Qwen</td><td>Qwen</td><td>Llama</td><td>1534.54</td><td>0.404</td><td>649.37</td><td>0.384</td></tr><tr><td>Small</td><td>Qwen</td><td>Llama</td><td>Qwen</td><td>1508.14</td><td>0.268</td><td>655.67</td><td>0.269</td></tr><tr><td>Small</td><td>Qwen</td><td>Llama</td><td>Llama</td><td>1499.46</td><td>0.401</td><td>572.73</td><td>0.398</td></tr><tr><td>Small</td><td>Llama</td><td>Qwen</td><td>Qwen</td><td>987.26</td><td>0.404</td><td>573.44</td><td>0.402</td></tr><tr><td>Small</td><td>Llama</td><td>Qwen</td><td>Llama</td><td>989.53</td><td>0.263</td><td>464.80</td><td>0.306</td></tr><tr><td>Small</td><td>Llama</td><td>Llama</td><td>Qwen</td><td>1010.12</td><td>0.405</td><td>456.70</td><td>0.413</td></tr><tr><td>Small</td><td>Llama</td><td>Llama</td><td>Llama</td><td>1025.37</td><td>0.270</td><td>454.21</td><td>0.254</td></tr><tr><td>Large</td><td>Qwen</td><td>Qwen</td><td>Qwen</td><td>2211.68</td><td>0.290</td><td>1192.28</td><td>0.372</td></tr><tr><td>Large</td><td>Qwen</td><td>Qwen</td><td>Llama</td><td>2262.02</td><td>0.592</td><td>1143.66</td><td>0.372</td></tr><tr><td>Large</td><td>Qwen</td><td>Llama</td><td>Qwen</td><td>2208.19</td><td>0.282</td><td>1173.31</td><td>0.284</td></tr><tr><td>Large</td><td>Qwen</td><td>Llama</td><td>Llama</td><td>2242.59</td><td>0.432</td><td>1137.61</td><td>0.382</td></tr><tr><td>Large</td><td>Llama</td><td>Qwen</td><td>Qwen</td><td>2260.25</td><td>0.396</td><td>1170.86</td><td>0.397</td></tr><tr><td>Large</td><td>Llama</td><td>Qwen</td><td>Llama</td><td>2313.33</td><td>0.261</td><td>1220.85</td><td>0.258</td></tr><tr><td>Large</td><td>Llama</td><td>Llama</td><td>Qwen</td><td>2300.87</td><td>0.419</td><td>1190.55</td><td>0.402</td></tr><tr><td>Large</td><td>Llama</td><td>Llama</td><td>Llama</td><td>2256.73</td><td>0.265</td><td>1225.97</td><td>0.367</td></tr></table>

Table 6: Per-configuration Bridge processing latency in milliseconds. Each value is the mean over the fixed task test set under the synchronized timing boundary in Appendix B.
<table><tr><td rowspan="2">Variant</td><td colspan="2">Qwen-Qwen-Llama, MATH</td><td colspan="2">Llama-Qwen-Qwen, ALFWorld</td></tr><tr><td>Accuracy (%)</td><td>Parameters (K)</td><td>Accuracy (%)</td><td>Parameters (K)</td></tr><tr><td>LatBridge (basic mode)</td><td>11.0</td><td>151.6</td><td>22.4</td><td>188.4</td></tr><tr><td>Without two-stage training</td><td>2.5</td><td>151.6</td><td>33.6</td><td>188.4</td></tr><tr><td>Without decoupled output training</td><td>3.0</td><td>94.2</td><td>0.0</td><td>94.2</td></tr><tr><td>Without input interface</td><td>2.5</td><td>94.2</td><td>0.0</td><td>94.2</td></tr><tr><td>Without output interface</td><td>0.5</td><td>57.3</td><td>1.5</td><td>94.2</td></tr><tr><td>Without Bridge-model freezing</td><td>2.5</td><td>2,314.2</td><td>22.4</td><td>3,596.3</td></tr></table>

Table 7: Small-model component and training ablations on two representative configurations. Configurations are listed as Bridge, source, and target models. Learned-parameter counts include all parameters updated by each variant; the final row additionally updates low-rank parameters in the Bridge model. Bold marks the best accuracy for each task.

![](images/693667fcdc0e6503a8e98f3bdf1e3ca800d0cce207b87a95be0d57e02ddeffb2.jpg)  
Figure 5: Task accuracy under different communication budgets in the basic mode. We retain different prefixes of the Bridge output before the output interface. Thin curves show source–Bridge pairs, with circles and squares denoting Qwen and Llama targets. Thick curves show target-family means with bootstrap 95% confidence bands.

The budget experiment tests how much of the Bridge-output sequence is needed by the target model. It uses the same hidden-state paths, evaluation examples, and trained interfaces as the message-replacement experiment. We retain the first 5%, 10%, 20%, 40%, 60%, 80%, or 100% of the valid Bridge-output states before the output interface. The models and decoding settings remain unchanged.

Figure 5 reports the configuration-level curves and target-family means. The confidence bands use the same task-level bootstrap procedure as the message-replacement plot. Accuracy decreases for most configurations as fewer states are retained, with different Bridge and target model pairs showing different rates of degradation. This experiment relates available message length to task accuracy. Truncation is applied after Bridge generation, so all retention proportions share the preceding complete autoregressive generation process.

<table><tr><td>Analysis</td><td>Qwen</td><td>Llama</td></tr><tr><td>Cross-source retrieval</td><td>76.0</td><td>88.5</td></tr><tr><td>Basic-mode probe</td><td> $5 0 . 5 \pm 3 . 7$ </td><td> $5 6 . 5 \pm 5 . 6$ </td></tr><tr><td>Accelerated-mode probe</td><td> $7 3 . 0 \pm 7 . 3$ </td><td> $7 1 . 5 \pm 4 . 6$ </td></tr><tr><td>Cross-mode retrieval</td><td>38.5</td><td>44.5</td></tr></table>

Table 8: Representation-analysis accuracy (%) by Bridge model. Probe values are five-fold means ± standard deviations.

## C.4 Representation Retrieval, Probing, and Alignment

These analyses test whether sample information remains recoverable across source models and processing modes. They use the large models and a separate set of 200 MATH algebra examples. Each representation sequence is mean-pooled into one vector. Cross-source retrieval compares the two source models while holding the Bridge model fixed. The correct neighbor is the representation of the same task among 200 candidates, giving a random baseline of 0.5%.

The linear probe predicts four clusters constructed from task-text TF-IDF features. The reported basic-mode probes use Qwen source states with each Bridge model, while accelerated-mode probes use matching Bridge and target model families. The clusters contain 33, 49, 79, and 39 examples, giving a majority-class baseline of 39.5%. The probe standardizes the pooled vectors, reduces them to 64 principal components, and fits logistic regression within five-fold stratified crossvalidation. For cross-mode alignment, separate 64-dimensional PCA transforms and a ridge mapping are fitted on four folds and evaluated on the held-out fold. Retrieval among the 40 held-out candidates has a random baseline of 2.5%.

Table 8 reports the retrieval and probing results. Both modes predict the text-cluster labels above the majority-class baseline, with higher probe accuracy for the accelerated-mode representations. Crosssource retrieval also remains well above chance. Although raw cosine similarity across modes is close to zero, linear alignment recovers part of the sample correspondence. The results show that differences between representation spaces coexist with recoverable sample information.

## C.5 Target-Side Representation Space

This analysis examines how the output interfaces place communicated representations in the target model’s input space. It uses 134 held-out ALF-World examples for each of the eight large-model configurations. We mean-pool the valid target-side message states and normalize each vector. Targetnative references are mean-pooled task-token embeddings used only for representation analysis.

For each target family, PCA is fitted once to the four sets of composed messages and one native reference per task. This gives 536 message vectors and 134 reference vectors. Figure 4 shows the native, Qwen-Bridge, and Llama-Bridge regions under this shared transform, with independently labeled local scales.

The representations separate mainly by source– Bridge configuration. Nearest-neighbor accuracy for identifying the same configuration is 100%, while same-task accuracy is 0% for both target families. Task-type silhouette scores are −0.008 for Qwen targets and −0.006 for Llama targets. The visible geometry therefore mainly reflects representation source and interface differences, with weak separation by task category.

## D Additional Related Work

This section expands the discussion in the main Related Work section. We first review agent organization and model selection, then latent communication, and finally efficient processing and representation mappings related to LatBridge.

## D.1 Agent Organization and Model Selection

Research on agent organization explains why a multi-agent system may need to change its models while retaining its task roles and communication relations. Role-playing, software-development organizations, reusable conversation frameworks, and evaluation suites study how agents acquire responsibilities and coordinate (Li et al., 2023; Hong et al., 2024; Wu et al., 2024; Zhu et al., 2025; Chen et al., 2026). MasRouter jointly chooses a collaboration mode, role assignment, and language model (Yue et al., 2025), while sparse and learned topologies change information propagation and computation (Li et al., 2024; Shen et al., 2025a). ReConcile aggregates diverse answers through confidenceweighted discussion (Chen et al., 2024). Controlled debate studies further show that gains depend on diversity, prompting, topology, and protocol design (Liang et al., 2024; Smit et al., 2024; Wang et al., 2024; Zhu et al., 2026). These studies motivate flexible agent configurations, whereas our work examines how latent communication can continue after an agent’s model is replaced.

## D.2 Latent Communication

Latent-communication methods exchange model representations instead of relying entirely on text. CIPHER sends expectations over token embeddings (Pham et al., 2024). Other methods communicate activations, token-level state-change trajectories, hidden states, or selected key-value layers (Ramesh and Li, 2025; Tang et al., 2025; Du et al., 2026; Shi et al., 2026). Recent work also studies joint latent collaboration, cache-to-cache transfer, mixtures of hidden thoughts, learned thought messages, query-conditioned cache exchange, cross-model key-value conversion, and vision-language feature transfer (Zou et al., 2025; Fu et al., 2025; Fein-Ashley et al., 2025; Zheng et al., 2025; Kriuk and Ng, 2025; Liu et al., 2026a,b). These approaches demonstrate communication within their evaluated configurations, but do not establish whether the same interfaces remain usable after agent model replacement. Surveys and audits also examine semantic fidelity, causal dependence, and message integrity alongside efficiency (Liu, 2026; Cheng et al., 2026; Brito and Baquero, 2026). Unlike work focused on communication within fixed model configurations, LatBridge supports agent model replacement by reusing unaffected and existing matching interfaces and training only missing ones.

## D.3 Efficient Processing

Related efficiency methods change either the reasoning process or the information exchanged between models. Transcript internalization distills a multi-agent interaction into one model (Yi et al., 2026). Continuous and compressed reasoning studies computation within a single model (Hao et al., 2025; Shen et al., 2025b; Cheng and Van Durme, 2024; Liu et al., 2024; Goyal et al., 2024; Pfau et al., 2024; Chang et al., 2026), while prompt and rationale compression shortens text inputs (Jiang et al., 2023; Ren et al., 2023; Xia et al., 2025). In contrast, LatBridge retains distinct source and target agents and reduces the processing time of the shared communication component while supporting model replacement. It does not merge collaboration into a single model or merely compress text.

## D.4 Representation Mappings

Research on representation mappings provides methods for converting representations between models. Model stitching studies connections between model representations, while low-rank adaptation reduces the number of trainable parameters (Bansal et al., 2021; Hu et al., 2022). Vocabulary adaptation and embedding alignment address related differences between tokenizers (Li et al., 2025; Truong et al., 2025; Singh et al., 2026). Research on representational convergence and representation similarity also informs our analysis of sample correspondence across models (Huh et al., 2024; Kornblith et al., 2019). LatBridge draws on these representation-mapping methods to implement a shared communication protocol. Its focus is on independently matching and reusing interfaces after agent model replacement, rather than only learning mappings between fixed models.

## E Implementation Details

This section supplements the workflow in Section 4 with details of representation handling, interface records, and accelerated-mode processing. Training hyperparameters are reported separately in Appendix H.

## E.1 Representation and Input Construction

Representation handling keeps the source states, Bridge states, and target inputs associated with the same task example. All states are matched to examples by task identifier. The input interface receives the source hidden-state sequence after length control by $\mathcal { P } _ { L }$ . The communication budget is $L = 1 2 8$ Valid positions are distinguished from padding by an attention mask.

For basic-mode output-interface training, the state collector uses the supervised answer interval under teacher forcing. During basic-mode evaluation, the Bridge model instead generates a sequence, and the output interface receives the hidden states at its generated positions. The target agent receives the resulting continuous input together with its fixed task instruction $p _ { t }$ . The source input and

Bridge-side task prompt are not separately supplied as text to the target agent. The output-mapping variant used in two large-model MATH configurations is described in Section A.1.

## E.2 Interface Storage and Retrieval

Interface records allow a trained mapping to be found and reused for a matching model pair. Each stored interface record contains the connected models, interface direction, input and output dimensions, trained parameters, and the processing mode when it affects the output representation. The source-agent and Bridge models identify an input interface, while the Bridge and target-agent models identify an output interface. A stored interface from another communication relation can be reused when these fields match. Basic-mode and accelerated-mode output interfaces are distinguished because they are trained on different input representations.

## E.3 Accelerated-Mode Processing

The accelerated mode constructs target-side continuous input from the Bridge-side task prompt using the prepared vocabulary mapping and output interface. The Bridge tokenizer first tokenizes $p _ { b } .$ , and the anchor-token suffix used to mark the input position is removed. The implementation retains the last L tokens if the remaining sequence is longer than the communication budget. When padding is used, a validity mask M identifies the positions supplied to the target model with $p _ { t }$ and $U _ { t } ^ { \mathrm { t a b } }$

When the vocabularies match, Bridge-side token identifiers can be used directly for target embedding lookup. Otherwise, the precomputed vocabulary mapping expands each Bridge token into its corresponding target-token sequence. The expanded sequence is again limited to the last L tokens before querying the target embedding table. The output interface then transforms the embeddings as specified in Equation 5.

The vocabulary mapping is constructed and cached before evaluation. For the timing experiment, tokenization and input preloading are also completed before the measured interval, and embedding lookup is performed on the GPU. The complete timing boundary is given in Appendix B.

## F Pilot Study

This section provides the settings and reporting details for the pilot study in Section 3.2. The experiment tests whether an interface trained by an

existing latent-communication method remains usable after an agent model is replaced.

## F.1 Experimental Setup

We use Interlat (Du et al., 2026) with Qwen2.5- 0.5B-Instruct and Llama-3.2-1B-Instruct as the source- and target-agent models. For each target model, we compare three conditions. The original model combination uses the source model for which the interface was trained. The modelreplacement condition changes the source model while keeping the target model and original interface fixed. The no-communication condition removes the communicated representation and retains only the target agent’s fixed instruction.

The source agent reads the task and its context and produces a continuous representation. The target agent receives only this representation and a fixed task instruction, without directly observing the task input. We evaluate 200 MATH examples using final-answer exact match and 134 held-out ALFWorld examples using normalized next-action exact match. These task scores measure whether the interface passes useful information.

When the source models have different hidden dimensions, an input projection maps the replacement model’s states to the dimension required by the original interface. This projection changes the tensor dimension without retraining the original communication interface. The three conditions otherwise retain the same target-side task setting.

## F.2 Results and Reporting

Table 1 reports the mean and standard deviation over three recorded random seeds. For a fixed target model, communication gain is the accuracy of a condition minus the no-communication accuracy. The retained gain is the replacement condition’s gain divided by the original model combination’s gain.

The original combinations provide clear gains on both tasks. After replacement, only 1.5% to 10% of the original gain remains on MATH, and neither ALFWorld replacement condition retains a positive gain. Since the target model and original interface are unchanged, the comparison shows that the original interface is not directly reusable for the new source model. This result motivates the separate, reusable interfaces developed in Section 4.

## G Data and Evaluation

This section describes the datasets, evaluation inputs, scoring rules, and resource terms. The pilot study and each research-question section specify their own experimental conditions. Training settings are given in Appendix H.

## G.1 Datasets and Evaluation Splits

We use English mathematical problems from MATH (Hendrycks et al., 2021) and English task descriptions and actions from ALFWorld (Shridhar et al., 2021). The large-model interface-training data contain 2,614 MATH algebra and geometry examples and 2,985 ALFWorld examples. ALFWorld evaluation uses 134 held-out examples. MATH evaluations use 200 examples, with fixed task identifiers specifying each evaluation set.

The message-replacement and communicationbudget experiments use a seed-42 MATH sample with 100 algebra and 100 geometry examples. The retrieval, probing, alignment, and qualitative analyses use a separate set of 200 algebra examples selected by task-identifier order. Their respective settings are described in Section C.2 and Section C.4. These sets serve different analyses and are kept distinct when reporting results.

## G.2 Task Inputs and Scoring

Evaluation preserves the target-side input condition across the compared configurations. The source agent receives the task and context, and the Bridge receives the Bridge-side task prompt. The target agent receives only its fixed instruction and the continuous input constructed by the Bridge. In the no-communication condition, this continuous input is removed.

MATH scoring extracts a boxed answer when present and applies LaTeX normalization before exact comparison. The reported small-model evaluation also uses fallback extraction for responses without a boxed answer, checking mathematical expressions, explicit answer phrases, and a final standalone number. ALFWorld scoring normalizes the predicted and reference actions before exact comparison. Its score measures offline next-action prediction rather than full interactive task success.

Teacher-forced target loss provides a separate measure of the target model’s predictions on the reference output. Its computation and results are given in Section A.3, rather than combined with the task-accuracy metric.

## G.3 Resource Sources and Licenses

The model and dataset citations identify the original resources, while this subsection records their access sources and license names. The MATH loader uses the EleutherAI/hendrycks\_math distribution. ALFWorld examples are read from the supervised task-action data used by the evaluation pipeline.

Qwen2.5-0.5B-Instruct and Qwen2.5-7B are distributed under Apache License 2.0.<sup>1</sup> Llama-3.2- 1B-Instruct and Llama-3.1-8B-Instruct are subject to their respective Llama 3.2 and Llama 3.1 Community License Agreements and acceptable-use policies.<sup>2</sup> The recorded configurations load the Llama weights from Unsloth distributions of these models. The MATH and ALFWorld project repositories publish MIT licenses for their repository contents.<sup>3</sup> These repository licenses should not be taken as a separate grant of rights over all underlying competition problems or third-party resources. Use and redistribution remain subject to the terms of the particular model and data distributions.

## H Training Settings

This section reports the model dimensions and interface-training settings used by the experiments. It complements the training objectives in Section 4.2.2 and the implementation details in Appendix E. Changes made for component ablations are described separately in Section C.1.

## H.1 Models and Interface Parameters

The small-model experiments use Qwen2.5-0.5B-Instruct with a hidden dimension of 896 and Llama-3.2-1B-Instruct with a hidden dimension of 2,048. The large-model experiments use Qwen2.5-7B with a hidden dimension of 3,584 and Llama-3.1- 8B-Instruct with a hidden dimension of 4,096. At each scale, selecting source, Bridge, and target models from the two families gives eight configurations. The Bridge model remains fixed within each replacement comparison.

The standard low-rank interface uses rank r = 32, scaling coefficient α = 64, and dropout 0.1.

Its trainable matrices follow Equation 1. Interface parameters are trained while the language-model parameters remain frozen. The small-model result records also include output-interface variants and ensembles, so the rank-32 setting describes the standard interface rather than every small-model variant. The parameter counts for the evaluated ablations are reported in Table 7.

## H.2 Two-Stage Interface Training

The two stages train the adjacent mappings separately, allowing an existing interface to be reused when the other side changes. The input interface is trained for three epochs with a learning rate of $2 \times 1 0 ^ { - 4 }$ . The output interface is trained for ten epochs with a learning rate of $5 \times 1 0 ^ { - 4 }$ . The standard configurations use AdamW and training seed 0. For these configurations, the small-model microbatch size is 2 with four accumulation steps, while the large-model micro-batch size is 1 with eight accumulation steps. Both therefore have an effective batch size of 8.

Input-interface training supplies mapped source states to the frozen Bridge model and minimizes Equation 3. Basic-mode output-interface training uses teacher-forced Bridge states from the supervised answer interval and minimizes Equation 4. States and reference outputs are matched by task identifier. After training, each interface is stored using the fields in Section E.2.

## H.3 Accelerated-Mode Output-Interface Training

The accelerated mode requires an output interface trained for the embedding-based input that it will receive during communication. The training procedure constructs target embeddings from the Bridge-side task prompt using the vocabulary mapping described in Section E.3. These embeddings replace teacher-forced Bridge states in the target-model cross-entropy objective. The largemodel experiments use the output-interface settings given above. The small-model accelerated-mode report uses rank-32 output interfaces trained for five epochs with AdamW, a learning rate of $5 \times 1 0 ^ { - 4 }$ and dropout 0.1. The resulting interfaces are stored by their Bridge model, target model, and processing mode.

## I Qualitative Cases

This section illustrates how target outputs can differ even when the communicated representations concern the same task. The three MATH examples use Qwen2.5-7B as the source, Bridge, and target models and come from the representation-analysis set in Section C.4. We retain the sample identifiers to distinguish the cases.

## I.1 Correct Bridge Answer but Incorrect Target Answer

In MATH\_100, the task gives $f ( x ) = ( x + 5 ) / 3$ and $g ( x ) = 1 / ( f ^ { - 1 } ( x ) + 1 )$ , and asks for $g ( 3 )$ . The reference answer is $1 / 5$ . The Bridge model generates the correct answer, but the target agent outputs 1 after receiving the basic-mode representation. With the accelerated-mode input constructed from the task prompt, the target instead outputs $1 / 5 .$ . The difference illustrates that a correct answer in the Bridge’s generated sequence does not ensure that the target can read it from the transformed hidden states.

## I.2 An Answer Corrected by the Accelerated Mode

In MATH\_1001, the task asks for the positive value of $\sqrt { x ^ { 3 } - 2 ^ { y } }$ when $x = 5$ and $y = 2$ . The reference answer is 11. The Bridge’s basic-mode generation reaches ${ \sqrt { 1 2 1 } } = 1 1$ , but the target does not produce the correct answer. With the accelerated-mode representation, the target outputs 11. As in the first case, the outcome depends on how the information is presented to the target model.

## I.3 Lower Loss without a Correct Answer

In MATH\_10, the task asks for the smallest x satisfying $| 5 x - 1 | = | 3 x + 2 |$ . The reference answer $\mathrm { i s } - 1 / 8$ . Both modes produce $1 / 8$ and miss the required negative solution. The accelerated mode nevertheless lowers the teacher-forced target loss. This case explains why we report generated-answer accuracy and target loss separately in Section A.3.