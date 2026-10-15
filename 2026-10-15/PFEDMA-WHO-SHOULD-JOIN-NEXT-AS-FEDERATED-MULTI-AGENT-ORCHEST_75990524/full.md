# PFEDMA: WHO SHOULD JOIN NEXT AS FEDERATED MULTI-AGENT ORCHESTRATION EVOLVES?

Shuai Li<sup>1</sup> Shengyuan Liu<sup>1</sup> Fan Qi<sup>1,2</sup> Xiaoshan Yang<sup>3</sup> <sup>1</sup>Tianjin University of Technology <sup>2</sup>Inner Mongolia University <sup>3</sup>Institute of automation, Chinese academy of science

## ABSTRACT

Federated Multi-Agent Systems (FedMAS) offer a promising paradigm for collaborative reasoning among LLM-based agents while keeping raw client data decentralized. However, reliable multi-round collaboration remains challenging. Agent selection must take into consideration evolving task requirements and the agent’s historical reliability, while privacy rules must adapt to changing collaboration contexts and meet the client’s personalized needs. To address these challenges, we propose PFedMA, a federated multi-agent framework for feedback driven multiround collaboration under heterogeneous privacy requirements. Across rounds, PFedMA maintains historical capability and reliability states to guide agent selection, evaluates task responses before aggregation, and refines subsequent queries according to accumulated collaboration feedback. PFedMA enables bidirectional local to global and global to local private rule evolution across heterogeneous clients. Clients generalize locally updated privacy rules and selectively incorporate globally maintained rules, thereby facilitating cross-client privacy rule evolution while preserving personalized constraints. Extensive experiments across reasoning, privacy sensitive, and adversarial settings show that PFedMA improves task performance, reduces measured privacy leakage, and improves system robustness.

## 1 INTRODUCTION

LLM based multi-agent systems (MAS) coordinate agents with complementary capabilities to address complex tasks (Guo et al., 2024; Wang et al., 2024). In privacy sensitive settings, however, the knowledge required for collaboration is distributed across multiple data holders with distinct privacy requirements, data ownership constraints, and access control policies. To complete the collaborative task, agents must integrate distributed knowledge and intermediate reasoning information, while raw data cannot be centralized. This challenge motivates federated multi-agent systems (FedMAS), a distributed collaboration paradigm in which agents reason over locally held information and exchange task relevant information without directly centralizing raw client data (Shi et al., 2025). However, existing FedMAS does not fully address how to select reliable agents or adapt information sharing rules as reasoning unfolds across rounds.

As collaboration progresses, both who should participate and what information should be shared may change with the evolving reasoning state. Early rounds may require broad domain expertise, whereas later rounds may call for specialized agents to resolve uncertainties exposed by intermediate responses. Meanwhile, as reasoning progresses and collaboration demands evolve, newly required information may introduce sharing scenarios that are not covered by existing privacy rules. Con sequently, agent orchestration and privacy control become closely interdependent across reasoning rounds. Agent orchestration determines whose knowledge enters the collaborative reasoning process, while privacy rules determine what information can be shared. The resulting responses and reasoning state, in turn, affect both subsequent agent selection and the privacy constraints relevant to later rounds. FedMAS therefore needs to use feedback from intermediate responses to update both agent selection and privacy rules across rounds.

These two interdependent decisions give rise to two challenges. First, task response reliability must inform agent selection across rounds. Previous studies have explored iterative interactions between clients in federated collaboration (Wang et al., 2025), as well as multi-round routing based on the current interaction context (Zhang et al., 2025a). However, selecting agents based only on the current query context may overlook agents’ historical reliability in previous rounds. Because agents may produce incomplete, erroneous, or adversarial responses (Deng et al., 2025; Zhang & Yang, 2026), selecting agents without accounting for their past reliability can propagate errors across rounds.

Second, privacy rules must adapt to evolving collaboration while preserving heterogeneous client requirements. Existing agent-level approaches protect sensitive information through context filtering and rule-based protection (Shi et al., 2025; Xiang et al., 2025). Separately, applying DP to agent responses (Panda et al., 2023) may introduce perturbations that propagate across rounds, increas ing the risk of semantic drift and inconsistent reasoning. These approaches typically rely on predefined privacy constraints or fixed protection strategies, which may fail to accommodate changing information needs across reasoning rounds, either suppressing information necessary for subsequent reasoning or failing to capture newly emerging privacy risks (Truong et al., 2021; Shi et al., 2025). Moreover updating rules only locally limits cross client reuse, while applying shared rules uniformly can overlook differences in client requirements, data characteristics, and task contexts (Kossek & Stefanovic, 2024). The challenge is to reuse newly identified privacy rules across clients without overriding their distinct local requirements.

To address these challenges, we propose PFedMA, a federated multi-agent framework for feedback driven multi-round collaboration under heterogeneous privacy requirements. At each round, PFedMA evaluates task responses and updates historical capability and reliability estimates to guide agent selection and query refinement in the next round. Response reliability also determines which clients contribute new privacy rule candidates. These candidates are generalized for cross-client reuse, while each client selectively adopts rules according to its local requirements; the adopted rules then govern subsequent queries and responses.

Our main contributions are summarized as follows:

• We propose PFedMA, a federated multi-agent system with feedback driven orchestration that leverages cross round capability and reliability feedback to adapt agent selection, thereby mitigating repeated selection of unreliable agents and error accumulation across rounds.

• We introduce a federated privacy rule co-evolution mechanism in PFedMA that enables privacy rules to adapt to evolving collaboration while preserving heterogeneous client requirements, al lowing local privacy rules to benefit other clients without overriding personalized privacy rules.

• Experiments across diverse benchmarks show that PFedMA improves average task performance and reduces measured privacy leakage compared with Fed-ICL, while achieving a lower attack success rate than EPEAgents under the evaluated prompt injection attacks.

## 2 RELATED WORK

Multi-Agent System. Multi-agent systems (MAS) predominantly rely on centralized architectures, structuring collaboration via domain personas (Tang et al., 2024), external retrieval (Xiao et al., 2025), or predefined execution pipelines (Chen et al., 2023; Wang et al., 2024). To resolve conflicts and reach consensus, systems employ diverse alignment strategies, including textual debates and voting (Choi et al., 2025; Sun et al., 2025), adaptive termination (Hu et al., 2025), and belief state broadcasting (Deng et al., 2025; Keshmirian et al., 2025). Recent advances in task routing further optimize this orchestration using dynamic paths (Yue et al., 2025) and reinforcement learning (Zhang et al., 2025a). However, these centralized paradigms mandate transparent raw context sharing, inherently violating data privacy. In contrast, our framework achieves dynamic multi-round dispatch while operating under explicit local data boundaries.

Federated Multi-Agent System. To reduce communication overhead, federated paradigms increasingly leverage text driven prompting (Qiu et al., 2023), textual gradients (Chen et al., 2025), and in context learning (Wu et al., 2024). Similarly, cooperative frameworks iteratively aggregate plain text responses from local exemplars (Wang et al., 2025). However, sharing raw text introduces severe prompt extraction risks (Duan et al., 2024), making secure communication a critical challenge (Kossek & Stefanovic, 2024).

![](images/c928f885dbb5cf07ee312c5872c0f5ece03408dd0c3a16a446c1ff3a6fbf6447.jpg)  
Figure 1: Overview of PFedMA. Task response feedback guides query refinement and client routing, while privacy rules evolve through cross client sharing and personalized adaptation. Together, they jointly determine the participants and the content to be shared, thereby laying the foundation for continued collaboration. The algorithm is provided in Appendix A.

Existing privacy enhancing approaches, including trusted environments (Nagar et al., 2021), node decomposition (Wang et al., 2021), and differential privacy (Panda et al., 2023), reduce information exposure through different mechanisms. However, depending on the mechanism, they may require additional trusted infrastructure or restrict the information available during collaboration, making it difficult to preserve semantically useful context across multiple reasoning rounds (Shi et al., 2025). To mitigate this risk, we restrict server bound communication to privacy processed responses, generalized privacy rule candidates and compact rule level feedback, thereby preserving the stated client local data boundary.

## 3 METHOD

## 3.1 PRELIMINARIES

We consider a federated multi-agent system with an honest but curious server and N heterogeneous clients $\mathcal { A } = \{ A _ { 1 } , . . . , A _ { N } \}$ . Each client $A _ { i }$ maintains a private dataset $\mathcal { D } _ { i }$ and a local experience buffer $\mathcal { X } _ { i } .$ , both of which remain on device. Although the server cannot directly access clients’ raw local data or unprocessed local responses, it may attempt to infer private information from uploaded privacy processed responses $\tilde { r } _ { i } ^ { t }$ . Meanwhile, some clients may provide unreliable or malicious task responses.

To address these concerns, we introduce historical capability and reliability states, along with global and personalized privacy rule repositories for query and response privacy processing, respectively. $\mathbf { A t }$ round t, the server maintains the accumulated collaboration state $z ^ { t - 1 }$ and processes the query $q ^ { t }$ using its global privacy rule repository $\Omega _ { \mathrm { g l o b a l } } ^ { t }$ to obtain $\tilde { q } ^ { t }$ . The orchestrator $\pi _ { \theta }$ selects a subset of participating clients based on their routing state $M _ { i } ^ { t }$ and sends $\tilde { q } ^ { t }$ to the selected clients. Each selected client generates a local response $r _ { i } ^ { t } \stackrel { \smile } { = } A _ { i } ( \tilde { q } ^ { t } , \bar { D _ { i } } , \mathcal { X } _ { i } )$ and processes it using its personalized privacy rule repository $\Omega _ { i , \mathrm { l o c a l } } ^ { t }$ before uploading $\tilde { r } _ { i } ^ { t }$ . Both $r _ { i } ^ { t }$ and $\Omega _ { i , \mathrm { l o c a l } } ^ { t }$ remain on device. The server evaluates the uploaded responses, updates the routing states to $M _ { i } ^ { t + 1 }$ , and aggregates reliable responses to obtain $z ^ { t }$ . Meanwhile, global and personalized privacy rules co-evolve through generalized local rule sharing and selective global to local adaptation. If the task remains unresolved, the server refines the query using the updated routing states and applies the updated global privacy rules before dispatching it to selected clients in the next round. The process ends when no further clients are needed or the round budget is exhausted.

## 3.2 ITERATIVE ORCHESTRATION FOR FEDERATED MAS

Building on the overview in Section 3.1, this section details client selection, response evaluation and aggregation, and the reward used to train the orchestrator.

## 3.2.1 AGENT SELECTION AND ROUTING STATE UPDATES

PFedMA first identifies the expertise domain $\rho _ { d }$ most relevant to $\tilde { q } ^ { t }$ as $d _ { \star } ^ { t } \quad = \quad$ arg $\operatorname* { m a x } _ { d \in \mathcal { D o m a i n } } \mathrm { C o s } ( \tilde { q } ^ { t } , \rho _ { d } )$ , and then retrieves the corresponding candidate set $\mathcal { A } _ { d _ { \star } ^ { t } }$ . The orchestrator then selects participating agents as

$$
\mathcal { E } ^ { t } = \pi _ { \theta } \Bigl ( \mathcal { A } _ { d _ { \star } ^ { t } } \ | \ \tilde { q } ^ { t } , z ^ { t - 1 } , \{ M _ { i } ^ { t } \} _ { A _ { i } \in \mathcal { A } _ { d _ { \star } ^ { t } } } \Bigr ) .\tag{1}
$$

Where $M _ { i } ^ { t } = [ \mathbf { C } _ { i } ^ { t } \Vert S _ { i } ^ { t } ]$ maintains the historical capability and reliability states of client $A _ { i } .$ After the returned response of $A _ { i }$ is evaluated, its routing state is updated using the newly observed historical capability and reliability feedback

$$
{ \bf C } _ { i } ^ { t + 1 } = { \bf C } _ { i } ^ { t } + f _ { i } ^ { t } { \bar { \bf v } } _ { i } ^ { t } , \qquad S _ { i } ^ { t + 1 } = \frac { n _ { i } ^ { t } S _ { i } ^ { t } + [ \mathrm { I n t } _ { i } ^ { t } , \mathrm { E v d } _ { i } ^ { t } , \mathrm { C o n } _ { i } ^ { t } ] ^ { \top } } { n _ { i } ^ { t } + 1 } , \qquad A _ { i } \in \widehat { \mathcal { E } } ^ { t }\tag{2}
$$

where ${ \widehat { \xi } } ^ { t }$ denotes the set of responses retained by the reliability filter, $n _ { i } ^ { t }$ is the number of previously evaluated responses from $A _ { i } ,$ , and $\mathrm { I n t } _ { i } ^ { t }$ , Evd<sup>t</sup>, and ${ \mathrm { C o n } } _ { i } ^ { t }$ denote its anomaly scores. We define $\mathbf { C } _ { i } ^ { t }$ as accumulated historical capability state of client $A _ { i }$ , where $f _ { i } ^ { t }$ indicates whether the current retained response improves upon the most recent retained response of the same client, and $\bar { \mathbf { v } } _ { i } ^ { t }$ identifies the capability dimensions in which the current response improves upon that reference. In contrast, $S _ { i } ^ { t }$ summarizes the agent’s historical risk as the mean of its observed anomaly scores.

The updated routing state is $M _ { i } ^ { t + 1 } = \left[ \mathbf { C } _ { i } ^ { t + 1 } \lVert S _ { i } ^ { t + 1 } \right]$ . Thus, response evaluations from the current round are retained in $M _ { i } ^ { t + 1 }$ and directly influence agent selection in subsequent rounds. Detailed state update definitions are provided in Appendix D.

## 3.2.2 RESPONSE RELIABILITY EVALUATION AND FILTERED AGGREGATION

Similar to the recently proposed decision centered model Jev (Almeida, 2026), PFedMA explicitly presents task level results rather than burying them in free form responses. While Jev focuses on decision making, PFedMA focuses on conclusions and records the claims and evidence corresponding to each conclusion.

At round $t ,$ the privacy processed response of client $A _ { i }$ is represented as $\tilde { r } _ { i } ^ { t } = \langle a _ { i } ^ { t } , \Gamma _ { i } ^ { t } , \mathcal { L } _ { i } ^ { t } \rangle$ , where $a _ { i } ^ { t }$ denotes the final conclusion, $\Gamma _ { i } ^ { t }$ denotes the collection of claims supporting the conclusion, and $\dot { \mathcal { L } } _ { i } ^ { t }$ denotes the corresponding collection of evidence associated with these claims. The details are provided in Appendix C.1.

PFedMA evaluates each response from the complementary perspectives of intent, evidence, and cross agent consensus. To aggregate fine grained anomaly signals, we use

$$
\mathrm { P o o l } _ { \lambda _ { p o o l } } ( \{ x _ { j } \} _ { j = 1 } ^ { J } ) = \lambda _ { p o o l } \operatorname* { m a x } _ { j } x _ { j } + ( 1 - \lambda _ { p o o l } ) \frac { 1 } { J } \sum _ { j = 1 } ^ { J } x _ { j } ,\tag{3}
$$

where $J$ denotes the number of fine grained anomaly scores being aggregated and $x _ { j }$ is the $j -$ th score. The operator balances the most severe anomaly with the overall anomaly level. This score captures abnormal intents such as instruction overriding, false authority claims, aggregator manipulation, and task goal substitution. The evidence anomaly $\operatorname { E v d } _ { i } ^ { t }$ assesses claim level evidence quality from support, verifiability, and coherence, while the consensus anomaly ${ \mathrm { C o n } } _ { i } ^ { t }$ measures semantic deviation from peer conclusions. Intent and evidence are evaluated by a frozen judge model, whereas consensus is computed deterministically; details are provided in Appendix C.3 and Appendix C.2.

The three anomaly dimensions are combined into an overall anomaly score, and task responses exceeding the tolerance threshold are excluded:

$$
\mathcal { R } i s k _ { i } ^ { t } = \mathrm { I n t } _ { i } ^ { t } + \mathrm { E v d } _ { i } ^ { t } + \alpha _ { \mathrm { c o n } } \mathrm { C o n } _ { i } ^ { t } , \qquad \widehat { \mathcal { E } } ^ { t } = \left\{ A _ { i } \in \mathcal { E } ^ { t } \ \big | \ \mathcal { R } i s k _ { i } ^ { t } \leq \tau _ { \mathrm { r i s k } } \right\} .\tag{4}
$$

where $\alpha _ { \mathrm { c o n } }$ controls the contribution of consensus anomaly, and $\tau _ { \mathrm { r i s k } }$ denotes the anomaly tolerance threshold. A relatively small $\alpha _ { \mathrm { c o n } }$ is used because disagreement alone does not necessarily indicate an unreliable response. The retained set $\widehat { \mathcal { E } } ^ { t }$ is then used for aggregation.

The retained privacy processed responses are aggregated to update the accumulated collaboration state:

$$
z ^ { t } = \left\{ \begin{array} { l l } { \mathrm { F u s i o n } \left( \left\{ \tilde { r } _ { i } ^ { t } \right\} _ { A _ { i } \in \widehat { \mathcal { E } } ^ { t } } , z ^ { t - 1 } \right) , } & { \widehat { \mathcal { E } } ^ { t } \neq \emptyset , } \\ { z ^ { t - 1 } , } & { \widehat { \mathcal { E } } ^ { t } = \emptyset . } \end{array} \right.\tag{5}
$$

In our implementation, Fusion is realized by the server side orchestrator $\pi _ { \theta } .$ , which synthesizes the retained privacy processed responses and the previous global reasoning state into $z ^ { t }$

## 3.2.3 COMPOSITE REWARD FOR MULTI-ROUND ORCHESTRATION

During RL training, PFedMA uses a composite reward that balances final answer correctness, answer quality improvements across rounds, response reliability, and agent invocation cost. For a collaboration trajectory τ containing T routing rounds, the total reward is

$$
R ( \tau ) = R _ { \mathrm { f m t } } ( \tau ) + R _ { \mathrm { a n s } } ( \tau ) + \sum _ { t = 1 } ^ { T } R _ { \mathrm { s t e p } } ^ { t } ,\tag{6}
$$

where $R _ { \mathrm { f m t } }$ evaluates format compliance and $R _ { \mathrm { a n s } }$ evaluates the final answer. Details for these two formulas are provided in Appendix E.1. Meanwhile, $R _ { \mathrm { { s t e p } } } ^ { t }$ delivers intermediate feedback for routing, with the step wise reward defined as

$$
R _ { \mathrm { s t e p } } ^ { t } = \Delta Q _ { t } - \overline { { \mathcal { R } i s k _ { t } } } - \frac { | \mathcal { E } ^ { t } | } { K _ { \operatorname* { m a x } } } , \qquad \overline { { \mathcal { R } i s k _ { t } } } = \frac { 1 } { | \mathcal { E } ^ { t } | } \sum _ { A _ { i } \in \mathcal { E } ^ { t } } \frac { \mathcal { R } i s k _ { i } ^ { t } } { 2 + \alpha _ { \mathrm { c o n } } } , \qquad | \mathcal { E } ^ { t } | > 1 .\tag{7}
$$

Here, $\Delta Q _ { t } = Q _ { t } - Q _ { t - 1 }$ records the change in answer quality from $z ^ { t - 1 }$ to $z ^ { t } ,$ while $\overline { { \mathcal { R } i s k } } _ { t }$ measures the average normalized response risk of the agents invoked at round t. $K _ { \mathrm { m a x } }$ denotes the maximum number of agents that can be invoked in a single round, and $| \mathcal { E } ^ { t } | / K _ { \operatorname* { m a x } }$ represents the fraction of the invocation budget used by the current routing decision. Detailed definitions and RL settings are provided in Appendix E.1.

## 3.3 FEDERATED PRIVACY RULE CO-EVOLUTION MECHANISM

Existing privacy rule based data minimization approaches for Federated MAS mainly regulate information disclosure according to agent roles and access requirements (Shi et al., 2025), but do not address how client specific privacy knowledge can be reused across the federation without centralizing personalized policies. To bridge this gap, PFedMA introduces a federated privacy rule co-evolution mechanism, in which shareable privacy knowledge is propagated globally while complete client specific privacy rule repositories remain local.

Global Privacy Rule Repository $\Omega _ { \mathrm { g l o b a l } } ^ { t }$ . The global privacy rule repository is maintained by the server to preserve privacy knowledge that is shareable across clients. At round t, the server receives generalized rule candidates $\bar { \omega } _ { i } ^ { t }$ from clients and evaluates their cross client semantic support. When multiple candidates are available, candidates achieving the highest positive support proportion are selected to form the global candidate set:

$$
\mathcal { G } ^ { t } = \left\{ \bar { \omega } _ { i } ^ { t } \in \mathcal { B } ^ { t } \ : \middle | \ : \zeta _ { i } ^ { t } > 0 , \ : \zeta _ { i } ^ { t } = \displaystyle \operatorname* { m a x } _ { \bar { \omega } _ { j } ^ { t } \in \mathcal { B } ^ { t } } \zeta _ { j } ^ { t } \right\} , \qquad | \mathcal { B } ^ { t } | > 1 .\tag{8}
$$

Here, $B ^ { t }$ is the candidate pool, and $\zeta _ { i } ^ { t } = \mathrm { S u p p } ^ { t } ( \bar { \omega } _ { i } ^ { t } ) / ( | B ^ { t } | - 1 )$ is the support proportion of candidate $\bar { \omega } _ { i } ^ { t }$ . The set $\mathcal { G } ^ { t }$ retains all candidates with the highest positive support proportion. Details are provided in Appendix F. The server also collects compact rule level feedback $\mathcal { F } ^ { \bar { t } }$ from previous rule executions and uses it together with the promoted candidates to update the global repository:

$$
\Omega _ { \mathrm { g l o b a l } } ^ { t + 1 } = \mathrm { R u l e U p d a t e } \left( \Omega _ { \mathrm { g l o b a l } } ^ { t } , \mathcal { G } ^ { t } , \mathcal { F } ^ { t } \right) .\tag{9}
$$

This update consolidates shareable rules and resolves conflicting or ineffective global rules according to accumulated client feedback. The updated repository is then used by the server to process the next query before dispatch: $\tilde { q } ^ { t + 1 } = P _ { \mathrm { i n } } \left( q ^ { t + 1 } , \Omega _ { \mathrm { g l o b a l } } ^ { t + 1 } \right)$ . The resulting privacy rule processed query $\tilde { q } ^ { t + 1 }$ is then dispatched to the selected clients for local inference, thereby transferring the updated global privacy constraints to the next round of collaboration. Detailed definitions and update procedures are provided in Appendix F.

Personalized Privacy Rule Repository $\Omega _ { i , \mathrm { l o c a l } } ^ { t } .$ . Upon receiving the privacy processed query $\tilde { q } ^ { t + 1 }$ , client $A _ { i }$ performs local inference using its private data and accumulated experience $r _ { i } ^ { t + 1 } =$ $A _ { i } \left( \tilde { q } ^ { t + 1 } , \mathcal { D } _ { i } , \mathcal { X } _ { i } \right) . \Phi _ { i } ^ { t } ( \omega )$ denotes the local utility score of privacy rule $\omega$ for client $A _ { i }$ at communication round $t ,$ and is defined as follows:

$$
\Phi _ { i } ^ { t } ( \omega ) = \operatorname { R e l } \left( \omega , \mathcal { X } _ { i } \right) + \frac { n _ { i , \mathrm { v a l i d } } ^ { t } ( \omega ) } { \operatorname* { m a x } \left( 1 , n _ { i , \mathrm { t r i g g e r } } ^ { t } ( \omega ) \right) } - \frac { n _ { i , \mathrm { c o n f i c t } } ^ { t } ( \omega ) } { \operatorname* { m a x } \left( 1 , n _ { i , \mathrm { t r i g g e r } } ^ { t } ( \omega ) \right) }\tag{10}
$$

Here, the term $\mathrm { R e l } ( \omega , \mathcal { X } _ { i } )$ ) measures the semantic relevance of rule ω to the local experience buffer $\chi _ { i } .$ . Furthermore, $n _ { i , \mathrm { t r i g g e r } } ^ { t } ( \omega ) , n _ { i , \mathrm { v a l i d } } ^ { t } ( \omega )$ , and $n _ { i , \mathrm { c o n f l i c t } } ^ { t } ( \omega )$ denote the numbers of times that rule ω is triggered, successfully enforced, and found to conflict with a local privacy requirement, respectively. The details are provided in Appendix F.

Meanwhile, the updated global rules are redistributed to the client for personalized adaptation. Rather than directly replacing its local rules, client $A _ { i }$ compares each redistributed global rule $\omega _ { g } ^ { t + 1 }$ with its semantically corresponding local rule $\omega _ { i } ^ { t }$ according to their local utility and selects

$$
\omega _ { i } ^ { \star } = \left\{ \omega _ { i } ^ { t } , \begin{array} { l l } { \phi _ { i } ^ { t } ( \omega _ { i } ^ { t } ) \geq \Phi _ { i } ^ { t } ( \omega _ { g } ^ { t + 1 } ) , } \\ { \omega _ { g } ^ { t + 1 } , } \end{array} \right. \qquad \Omega _ { i , \mathrm { l o c a l } } ^ { t + 1 } = \Omega _ { i , \mathrm { l o c a l } } ^ { t } \cup _ { m e r g e } \{ \omega _ { i } ^ { \star } \} .\tag{11}
$$

Here, $\omega _ { g } ^ { t + 1 } \in \Omega _ { \mathrm { g l o b a l } } ^ { t + 1 } .$ with a detailed definition provided in Appendix F. This selective adaptation allows global privacy knowledge to refine the personalized repository without overriding client specific alternatives. The personalized repository is applied to each locally generated response before transmission $\tilde { r } _ { i } ^ { t + 1 } = P _ { \mathrm { r e s p } , i } \left( r _ { i } ^ { t + 1 } , \Omega _ { i , \mathrm { l o c a l } } ^ { t + 1 } \right)$ . If a previously uncovered privacy constraint is detected, the corresponding rule $\bar { \boldsymbol { \omega } } _ { i } ^ { t + 1 } \in \Omega _ { i , \mathrm { l o c a l } } ^ { t + 1 }$ remains on device, and only its generalized representation is shared $\bar { \omega } _ { i } ^ { t + 1 } = \mathcal { P } _ { i } ^ { \mathrm { g e n } } \left( \omega _ { i } ^ { t + 1 } \right)$ . Here, $\mathcal { P } _ { i } ^ { \mathrm { g e n } }$ denotes the client side private rules generalization operator, and $\bar { \omega } _ { i } ^ { t + 1 }$ is the shareable rule representation. The generalized rule is sent to the server for the subsequent global privacy rule updates; see Appendix F.

## 4 EXPERIMENTS

In the experiments, we seek to answer the following research questions:

RQ1: Does cross round response feedback improve multi-round federated reasoning over feedback free orchestration?

RQ2: Can federated privacy rule co-evolution enable cross client privacy rule coordination and personalized adaptation while maintaining rule enforcement effectiveness and task utility?

RQ3: Can multi-factor reliability evaluation improve robustness against unreliable task responses while preserving benign task utility?

## 4.1 EXPERIMENTAL SETUP

Models. Following KABB (Zhang et al., 2025b), we construct 20 agent profiles across 12 knowledge domains and distribute them over N federated clients. Each client hosts a local agent with private data, an experience buffer, and personalized privacy rules, while multiple clients may share the same profile. Agent profiles use domain specific prompts with one of seven LLM backbones: Mini-Max2.5 (Chen et al., 2026), DeepSeek-V3 (Liu et al., 2024), Qwen2-72B-Instruct (Bai et al., 2023), GPT-4o (Hurst et al., 2024), DeepSeek-R1 (Guo et al., 2025), LLaMA-3-70B-Instruct (Adams et al., 2024), or Kimi-K2.5 (Team et al., 2026).

Dataset and Evaluation Metrics. We evaluate PFedMA on mathematical reasoning benchmarks, including GSM8K (Cobbe et al., 2021), MATH (Hendrycks et al., 2021), and AIME24 (Muennighoff et al., 2025); clinical QA benchmarks, including MedQA and MedMCQA (Tang et al., 2024); instruction following with AlpacaEval 2.0 (Zhang et al., 2025b); and privacy sensitive web agent tasks with SeeAct on Mind2Web-SC (Xiang et al., 2025). We report ASR, RR, PNA, and NRP following ASB (Zhang et al., 2024), together with BP to evaluate benign task utility with defense mechanisms enabled, together with LPA, LPP, LPR, EA, FRA, and LR for privacy evaluation (Xiang et al., 2025; Shao et al., 2024). Metric definitions are provided in Appendix H.

Baselines. We conduct a comparative analysis of the PFedMA algorithm against other MAS methods. For the single agent baseline, the primary goal is to demonstrate that our method’s performance is not due to the server LLM model, but rather to the effectiveness of the federated MAS. For federated MAS, we compare with Fed-ICL (Wang et al., 2025) and EPEAgents (Shi et al., 2025). For centralized MAS, we compare with Router-R1 (Zhang et al., 2025a), KABB (Zhang et al., 2025b), and OPTAGENT (Bi et al., 2025). All multi-agent methods use the same pool of 20 experts and the same number of iteration rounds to isolate the effect of orchestration and aggregation.

## 4.2 MAIN RESULTS AND ANALYSIS

Table 1: Performance and privacy comparison of different frameworks. Best results are shown in bold, and second best results are underlined. For Avg calculation, lower is better metrics are converted using 100 − score before averaging.
<table><tr><td colspan="8">RQ1: Performance comparison across multiple benchmarks</td></tr><tr><td>Method</td><td>Math</td><td>GSM8K</td><td>AIME24</td><td>MedQA</td><td>AlpacaEval 2.0MedMCQA</td><td></td><td>Avg</td></tr><tr><td>Single Agent</td><td> $4 2 . 6 3 { \scriptstyle \pm 3 . 5 1 }$ </td><td> $7 9 . 1 1 { \scriptstyle \pm 1 . 3 9 }$ </td><td> $6 . 6 7 _ { \pm 0 . 7 3 }$ </td><td> $6 3 . 3 5 { \scriptstyle \pm 4 . 3 1 }$ </td><td> $5 8 . 5 1 _ { \pm 0 . 1 8 }$ </td><td> $5 7 . 8 2 _ { \pm 0 . 2 1 }$ </td><td> $5 1 . 3 5 { \scriptstyle \pm 1 . 7 2 }$ </td></tr><tr><td>Router-R1</td><td> $7 6 . 0 6 { \scriptstyle \pm 2 . 7 3 }$ </td><td> $8 4 . 0 2 { \scriptstyle \pm 2 . 9 1 }$ </td><td> $7 3 . 4 0 { \scriptstyle \pm 4 . 3 4 }$ </td><td> $7 5 . 6 1 { \scriptstyle \pm 1 . 7 1 }$ </td><td> $7 5 . 4 3 { \scriptstyle \pm 3 . 9 4 }$ </td><td> $6 9 . 5 7 { \scriptstyle \pm 1 . 0 7 }$ </td><td> $7 5 . 6 8 { \scriptstyle \pm 2 . 7 8 }$ </td></tr><tr><td>KABB</td><td> $7 7 . 5 1 { \scriptstyle \pm 1 . 1 2 }$ </td><td> $8 5 . 6 6 { \scriptstyle \pm 0 . 6 0 }$ </td><td> $7 1 . 9 3 { \scriptstyle \pm 3 . 0 0 }$ </td><td> $\mathbf { 7 8 . 1 5 { \scriptstyle \pm 0 . 7 3 } }$ </td><td> $7 2 . 3 8 { \scriptstyle \pm 4 . 3 0 }$ </td><td> $\mathbf { 7 2 . 0 5 { \scriptstyle \pm 1 . 7 0 } }$ </td><td> $7 6 . 2 8 { \scriptstyle \pm 1 . 9 1 }$ </td></tr><tr><td>OPTAGENT</td><td> $7 5 . 0 2 { \scriptstyle \pm 1 . 2 7 }$ </td><td> $8 6 . 2 6 _ { \pm 4 . 3 1 }$ </td><td> $6 2 . 4 3 { \scriptstyle \pm 2 . 4 3 }$ </td><td> $6 6 . 2 7 _ { \pm 1 . 2 1 }$ </td><td> $6 9 . 0 1 _ { \pm 1 . 7 8 }$ </td><td> $6 4 . 8 2 _ { \pm 0 . 9 1 }$ </td><td> $7 0 . 6 4 { \scriptstyle \pm 1 . 9 9 }$ </td></tr><tr><td>EPEAgents</td><td> $7 0 . 0 3 { \scriptstyle \pm 0 . 3 9 }$ </td><td> $8 0 . 1 0 { \scriptstyle \pm 1 . 7 6 }$ </td><td> $6 8 . 5 7 _ { \pm 3 . 1 6 }$ </td><td> $7 7 . 2 0 { \scriptstyle \pm 1 . 9 9 }$ </td><td> $7 0 . 8 8 { \scriptstyle \pm 1 . 3 6 }$ </td><td> $6 3 . 5 4 { \scriptstyle \pm 0 . 8 1 }$ </td><td> $7 1 . 7 2 { \scriptstyle \pm 1 . 5 8 }$ </td></tr><tr><td>Fed-ICL</td><td> $\underline { { 7 9 . 3 3 _ { \pm 0 . 8 3 } } }$ </td><td> $8 5 . 0 4 { \scriptstyle \pm 1 . 0 4 }$ </td><td> $\underline { { 7 4 . 7 8 _ { \pm 1 . 5 1 } } }$ </td><td> $7 6 . 0 3 { \scriptstyle \pm 3 . 0 0 }$ </td><td> $7 3 . 5 1 _ { \pm 0 . 2 9 }$ </td><td> $6 9 . 3 4 { \scriptstyle \pm 0 . 9 3 }$ </td><td> $7 6 . 3 4 { \scriptstyle \pm 1 . 2 7 }$ </td></tr><tr><td>PFedMA</td><td> $\mathbf { 8 4 . 5 3 \pm 2 . 5 1 }$ </td><td> $\mathbf { 8 6 . 6 0 _ { \pm 0 . 8 1 } }$ </td><td> ${ \bf 7 6 . 6 7 _ { \pm 1 . 8 3 } }$ </td><td> $7 7 . 9 1 _ { \pm 0 . 8 4 }$ </td><td> $\mathbf { 7 7 . 6 5 { \scriptstyle \pm 0 . 1 8 } }$ </td><td> $6 9 . 8 0 { \scriptstyle \pm 0 . 5 3 }$ </td><td> $\mathbf { 7 8 . 8 6 _ { \pm 1 . 1 2 } }$ </td></tr></table>

<table><tr><td colspan="8">RQ2: Evaluation of privacy processed and performance</td></tr><tr><td>Method</td><td>LR↓</td><td>LPR ↑</td><td> $\mathrm { L P A } \uparrow$ </td><td>LPP↑</td><td>EA↑</td><td>FRA↑</td><td>Avg ↑</td></tr><tr><td>Single Agent</td><td> $4 3 . 4 0 { \scriptstyle \pm 0 . 6 2 }$ </td><td> $6 9 . 0 0 { \scriptstyle \pm 1 . 1 4 }$ </td><td> $8 3 . 5 0 { \scriptstyle \pm 0 . 8 7 }$ </td><td> $9 8 . 6 0 { \scriptstyle \pm 0 . 3 5 }$ </td><td> $6 8 . 0 0 { \scriptstyle \pm 1 . 2 6 }$ </td><td> $6 2 . 5 0 { \scriptstyle \pm 0 . 9 4 }$ </td><td> $7 3 . 0 3 { \scriptstyle \pm 0 . 8 6 }$ </td></tr><tr><td>Router-R1</td><td> $3 9 . 0 8 _ { \pm 0 . 7 8 }$ </td><td> $7 3 . 7 3 { \scriptstyle \pm 1 . 3 1 }$ </td><td> $8 5 . 7 0 { \scriptstyle \pm 0 . 6 9 }$ </td><td> $9 6 . 9 0 { \scriptstyle \pm 0 . 5 2 }$ </td><td> $7 3 . 7 0 { \scriptstyle \pm 1 . 0 8 }$ </td><td> $7 3 . 0 5 { \scriptstyle \pm 0 . 8 3 }$ </td><td> $7 7 . 3 3 { \scriptstyle \pm 0 . 8 7 }$ </td></tr><tr><td>KABB</td><td> $3 5 . 2 0 { \scriptstyle \pm 0 . 9 1 }$ </td><td> $8 1 . 6 4 { \scriptstyle \pm 0 . 7 4 }$ </td><td> $9 0 . 0 0 { \scriptstyle \pm 0 . 5 8 }$ </td><td> $9 8 . 1 0 { \scriptstyle \pm 0 . 4 1 }$ </td><td> $7 6 . 5 0 { \scriptstyle \pm 1 . 1 7 }$ </td><td> $6 9 . 5 0 { \scriptstyle \pm 0 . 9 6 }$ </td><td> $8 0 . 0 9 { \scriptstyle \pm 0 . 8 0 }$ </td></tr><tr><td>OPTAGENT</td><td> $3 7 . 3 0 { \scriptstyle \pm 0 . 8 5 }$ </td><td> $8 0 . 0 4 { \scriptstyle \pm 1 . 0 9 }$ </td><td> $9 0 . 0 0 { \scriptstyle \pm 0 . 6 3 }$ </td><td> $9 5 . 3 2 { \scriptstyle \pm 0 . 2 2 }$ </td><td> $\mathbf { 8 0 . 0 0 { \scriptstyle \pm 0 . 4 8 } }$ </td><td> $7 2 . 1 0 { \scriptstyle \pm 0 . 7 7 }$ </td><td> $8 0 . 0 3 { \scriptstyle \pm 0 . 6 7 }$ </td></tr><tr><td>EPEAgents</td><td> $3 3 . 3 0 { \scriptstyle \pm 0 . 6 6 }$ </td><td> $8 6 . 5 0 { \scriptstyle \pm 0 . 8 2 }$ </td><td> $\underline { { 9 3 . 3 7 _ { \pm 0 . 5 4 } } }$ </td><td> $9 9 . 0 1 _ { \pm 0 . 1 9 }$ </td><td> $7 0 . 5 0 { \scriptstyle \pm 1 . 2 1 }$ </td><td> $7 5 . 6 3 { \scriptstyle \pm 0 . 7 3 }$ </td><td> $8 1 . 9 5 { \scriptstyle \pm 0 . 6 9 }$ </td></tr><tr><td>Fed-ICL</td><td> $3 2 . 6 0 { \scriptstyle \pm 0 . 7 1 }$ </td><td> $8 4 . 0 0 { \scriptstyle \pm 0 . 9 5 }$ </td><td> $9 2 . 0 0 { \scriptstyle \pm 0 . 6 1 }$ </td><td> $9 7 . 2 4 _ { \pm 0 . 1 7 }$ </td><td> $7 5 . 6 0 { \scriptstyle \pm 1 . 0 4 }$ </td><td> $7 3 . 3 0 { \scriptstyle \pm 0 . 8 8 }$ </td><td> $\overline { { 8 1 . 5 9 _ { \pm 0 . 7 3 } } }$ </td></tr><tr><td>PFedMA</td><td> $\pm { \overline { { 2 6 . 7 0 _ { \pm 0 . 4 3 } } } }$  </td><td> $\mathbf { 8 9 . 3 0 _ { \pm 0 . 6 9 } }$  </td><td> $\mathbf { 9 4 . 7 0 { \scriptstyle \pm 0 . 3 8 } }$  </td><td> $\mathbf { 9 9 . 8 6 _ { \pm 0 . 1 2 } }$ </td><td> $7 7 . 0 0 { \scriptstyle \pm 0 . 9 7 }$  </td><td> $\mathbf { 7 9 . 7 1 { \scriptstyle \pm 0 . 5 5 } }$  </td><td> $\mathbf { 8 5 . 6 5 { \scriptstyle \pm 0 . 5 2 } }$ </td></tr></table>

Cross Round Feedback and Iterative Orchestration (RQ1).

Table 1 compares PFedMA with representative single-agent, centralized MAS, and federated MAS baselines under the same expert pool and collaboration-round budget. PFedMA achieves the highest average score of 78.86, outperforming the strongest baseline, Fed-ICL, by 2.52 points. It ranks first on MATH, GSM8K, AIME24, and AlpacaEval 2.0, and second on MedQA and MedMCQA, indicating strong performance across the evaluated tasks. Figure 9a shows that accuracy generally increases across collaboration rounds, suggesting that useful reasoning information is progressively accumulated. The ablation results in Figure 2b further show that removing query optimization or either historical routing state consistently degrades performance. These results indicate that cross round state evolution, query adaptation, and learned orchestration each contribute to the overall improvement. Figure 6c provides complementary evidence that PFedMA adaptively selects agents according to task requirements: mathematical agents are selected more frequently for mathematical benchmarks, while medical agents are favored for medical tasks. Overall, these results support the effectiveness of feedback driven iterative orchestration in progressively refining the reasoning state and adapting subsequent collaboration decisions.

Table 2: Robustness of different frameworks under adversarial attacks
<table><tr><td colspan="7">RQ3: Evaluation of system security and robustness</td></tr><tr><td>Method</td><td>ASR↓</td><td>RR↑</td><td> ${ \mathrm { B P } } \uparrow$ </td><td> $\mathrm { P N A } \uparrow$ </td><td> $\mathrm { N R P \uparrow }$ </td><td> $\mathbf { A v g } \uparrow$ </td></tr><tr><td></td><td></td><td> $5 . 3 4 { \scriptstyle \pm 2 . 4 1 }$ </td><td> $7 7 . 7 5 { \scriptstyle \pm 3 . 8 4 }$ </td><td> $6 6 . 5 7 { \scriptstyle \pm 2 . 0 3 }$ </td><td> $3 0 . 0 8 { \scriptstyle \pm 3 . 2 0 }$ </td><td> $4 4 . 9 8 { \scriptstyle \pm 2 . 5 4 }$ </td></tr><tr><td>Single Agent Router-R1</td><td> $5 4 . 8 2 { \scriptstyle \pm 1 . 2 3 }$   $5 2 . 0 7 { \scriptstyle \pm 0 . 8 1 }$ </td><td> $9 . 1 0 { \scriptstyle \pm 0 . 3 6 }$ </td><td> $7 6 . 7 3 { \scriptstyle \pm 0 . 7 4 }$ </td><td> $7 6 . 7 0 { \scriptstyle \pm 0 . 6 5 }$ </td><td> $3 6 . 7 6 { \scriptstyle \pm 0 . 8 8 }$ </td><td> $4 9 . 4 4 _ { \pm 0 . 6 9 }$ </td></tr><tr><td>KABB</td><td> $3 6 . 2 0 { \scriptstyle \pm 0 . 6 9 }$ </td><td> $1 8 . 5 0 { \scriptstyle \pm 0 . 5 4 }$ </td><td> $7 5 . 2 9 { \scriptstyle \pm 0 . 8 3 }$ </td><td> $\mathbf { 8 8 . 0 0 _ { \pm 0 . 4 7 } }$ </td><td> $5 6 . 1 4 { \scriptstyle \pm 0 . 7 6 }$ </td><td> $6 0 . 3 5 { \scriptstyle \pm 0 . 6 6 }$ </td></tr><tr><td>OPTAGENT</td><td> $5 0 . 0 7 { \scriptstyle \pm 0 . 7 7 }$ </td><td> $1 1 . 5 0 { \scriptstyle \pm 2 . 4 9 }$ </td><td> $6 9 . 4 0 { \scriptstyle \pm 0 . 9 1 }$ </td><td> $7 3 . 5 4 { \scriptstyle \pm 0 . 5 8 }$ </td><td> $3 6 . 7 2 { \scriptstyle \pm 1 . 0 2 }$ </td><td> $4 8 . 2 2 _ { \pm 1 . 1 5 }$ </td></tr><tr><td>EPEAgents</td><td> $3 2 . 0 1 { \scriptstyle \pm 0 . 5 2 }$ </td><td> $2 2 . 1 3 { \scriptstyle \pm 0 . 6 1 }$ </td><td> $7 6 . 5 7 { \scriptstyle \pm 1 . 7 2 }$ </td><td> $8 2 . 8 4 { \scriptstyle \pm 0 . 5 5 }$ </td><td> $5 6 . 3 2 { \scriptstyle \pm 0 . 6 9 }$ </td><td> $\underline { { 6 1 . 1 7 _ { \pm 0 . 8 2 } } }$ </td></tr><tr><td>Fed-ICL</td><td> $4 9 . 3 0 { \scriptstyle \pm 0 . 8 4 }$ </td><td> $1 3 . 0 0 { \scriptstyle \pm 0 . 4 3 }$ </td><td> $7 4 . 7 5 { \scriptstyle \pm 3 . 7 9 }$ </td><td> $7 4 . 7 1 { \scriptstyle \pm 1 . 6 6 }$ </td><td> $3 7 . 8 8 { \scriptstyle \pm 1 . 7 3 }$ </td><td> $5 0 . 2 1 { \scriptstyle \pm 1 . 6 9 }$ </td></tr><tr><td></td><td></td><td></td><td></td><td> $8 5 . 5 0 { \scriptstyle \pm 0 . 4 9 }$  </td><td> ${ \bf 5 9 . 1 7 { \scriptstyle \pm 0 . 7 1 } }$  </td><td> ${ \bf 6 3 . 4 5 { \scriptstyle \pm 0 . 5 7 } }$ </td></tr><tr><td>PFedMA</td><td> $\mathbf { 3 0 . 8 0 { \scriptstyle \pm 0 . 4 5 } }$ </td><td> $\mathbf { 2 4 . 6 0 { \scriptstyle \pm 0 . 5 8 } }$ </td><td> $\mathbf { 7 8 . 8 0 _ { \pm 0 . 6 3 } }$ </td><td></td><td></td><td></td></tr></table>

![](images/ad7c4c43f3f617d0dd2388de37e957d6a23f63bf500b723b1ced034530ed7313.jpg)  
(a) Ablation of structured reliability evaluation

![](images/3b387d7a92ca59d144184901ff7d0dff0f17ff435b5b217d8f7a796a5c17964e.jpg)  
(b) Feedback Driven Orchestration Ablation

![](images/9d85201b19e5ac9c8aad972fa2516018fa4e88e713069d10ec92854b1dcb5353.jpg)  
(c) Federated Rule Co-Evolution Ablation

![](images/4196f366d2d047b4db724e093cd502d5d2e1f66d77d03ffcc68a2a0e9db9b340.jpg)  
(d) Adversarial Robustness Utility Trade off

Figure 2: Ablation studies of the reliability module, iterative orchestration loop, federated privacy rule co-evolution loop and robustness utility trade off across adversarial ratios.  
![](images/59180a6b7cd5afba491fdfc32a6a8e5ce1a8066ff7ac2154d0ca8f299246cf82.jpg)

![](images/63f01f01dccb5a6154c42a699e7b53291e4c1e937f47ac86aaff2f77d12f9871.jpg)  
(a) Accuracy and Token Usage by Round

![](images/e8e00e38bb141c64bf0fb5615c6bf279368d9ad51a1a76e0ff5959f1654cabbb.jpg)  
Risk threshold ${ { \tau } _ { r i s k } }$

![](images/0124c86e9094d6e0f20d3f4875fbeffad69bc28c1d503ceadd4785f9adafc713.jpg)  
Risk threshold ${ { \tau } _ { r i s k } }$  
(b) Sensitivity Analysis of the Risk Threshold τ  
Figure 3: Performance Evaluation under Iterative Optimization and Risk Threshold Variation.

Cross Client Privacy Rule Coordination and Personalized Adaptation (RQ2). Table 1 shows that PFedMA achieves the lowest leakage rate of 26.70, which is 5.90 points lower than the second best result. It also obtains the best LPR, LPA, LPP, and FRA scores while achieving the second best EA, indicating that the improved privacy protection does not come at the cost of substantially degraded task utility. The lower leakage rate, together with the higher FRA, indicates that PFedMA strengthens measured privacy rule enforcement while preserving task utility under the evaluated setting. Figure 4a further shows that privacy rule coverage increases progressively across collaboration rounds. This trend is consistent with the intended role of federated privacy rule co-evolution, through which reusable rules identified locally are consolidated globally and subsequently adapted to client specific requirements. Together, these results provide empirical support for cross client privacy rule coordination and personalized adaptation while maintaining a favorable privacy utility balance.

Reliability Evaluation and Adversarial Robustness (RQ3). Table 2 compares PFedMA with representative baselines under prompt injection attacks. PFedMA achieves the lowest ASR of 30.80, reducing the attack success rate by 1.21 points compared with the second best method, EPEAgents. Meanwhile, PFedMA obtains the highest RR, BP, and NRP, and the second highest PNA, resulting in the best overall score of 63.45. These results indicate that the robustness improvement is not achieved by simply rejecting more responses. PFedMA remains effective on benign inputs while reducing the influence of adversarial responses. Overall, the results demonstrate a favorable robustness utility trade off under adversarial federated collaboration.

## 4.3 ABLATION STUDY

![](images/e363c25d188fcbceee671af35d8fcd74b6517382853628a491990ee4a1b35784.jpg)  
(a) Privacy Coverage Rate

![](images/72050a1421806ac974523cd5cfa742675f0a7dbae4a179ee4ebe917fb5b1765c.jpg)  
(b) Accuracy Efficiency

![](images/5b52337732a9434653d2fc468fa85b7f3d2cf93b248af0910ce3a17fe2621454.jpg)  
(a) Hyperparameter $\lambda _ { p o o l }$

![](images/36b5d5d029d82fd8fa91660a1d403068d6051749002608f2c749fdd544691f42.jpg)  
(b) Consensus $\alpha _ { c o n }$  
Figure 5: Hyperparameters $\lambda _ { p o o l }$ and $\alpha _ { c o n } .$ . The x-axis represents the parameter values, and the y-axis shows the corresponding ASR and RR.

Figure 4: (a) Privacy rule coverage across rounds and (b) accuracy efficiency trade off across routing budgets.  
![](images/6a7829c0a628215523cb85815b4d5f9e3cd4a00df1b9b38d763b782d5559669f.jpg)  
(a) Abilities of Agent 1

![](images/a1bebf3aa8febc5422f0a5dacb364429536002fa55909cd54c4fa012f5d46daf.jpg)  
(b) Abilities of Agent 6

![](images/e1fd23709bc080f2113b4d37635179178ae5f062c26e5982d3e28952238cffe0.jpg)  
(c) Agent Selection Heatmap  
Figure 6: Agent Abilities and Agent Selection. Full results are given in Appendix J.2.

Effectiveness of Feedback Driven Iterative Orchestration. Figure 2b evaluates the contributions of query optimization and the historical routing states. Specifically, the -RQ variant removes feedback guided query optimization, the -IA variant removes the historical capability state $C _ { i }$ , and the - RT variant removes the historical reliability state $S _ { i }$ . Each ablation breaks a distinct pathway through which outcomes from the current round influence subsequent collaboration decisions. The consistent degradation of these variants demonstrates that PFedMA benefits from maintaining a closed cross round feedback loop, rather than merely repeating independent multi-agent interactions.

Necessity of Federated Privacy Rule Co-Evolution. Figure 2c compares PFedMA with five variants, where NR removes privacy rules, SR keeps rules static, LE updates rules only locally, G2L retains only global to local rule adaptation, and L2G retains only local to global rule aggregation. PFedMA achieves the privacy utility trade off, confirming the importance of combining personalized local adaptation with cross client rule synchronization. Figure 9c further shows how privacy rule effectiveness evolves across federated communication rounds.

PFedMA consistently achieves the highest FRA, outperforming local only evolution, static rules, and the variant without privacy rules. Its persistent advantage across rounds indicates that local adaptation alone is insufficient to fully exploit reusable privacy knowledge. Together, these results demonstrate that bidirectional local to global rule consolidation and global to local personalized adaptation are both important for improving privacy rule effectiveness while preserving task utility.

Effectiveness of Multi Factor Reliability Evaluation. Figure 2a compares PFedMA with variants that remove the complete reliability evaluator or individual evaluation dimensions, including intent, evidence, and consensus. PFedMA achieves the best robustness utility profile, confirming the importance of reliability aware response assessment before aggregation. Removing the complete evaluator causes the largest overall degradation, while removing any individual dimension also weakens performance. These results indicate that relying on a single reliability signal is insufficient to consistently identify unreliable responses.

## 5 CONCLUSION AND LIMITATIONS

We propose PFedMA, a federated multi-agent framework integrating feedback-driven orchestration with federated privacy rule co-evolution. The orchestration loop adapts agent selection based on evolving task requirements and historical reliability, while the privacy loop enables bidirectional rule evolution while preserving personalized constraints. Experiments across reasoning, privacysensitive, and adversarial tasks demonstrate improvements in task performance, privacy rule effectiveness, and robustness. PFedMA focuses on semantic privacy rule coordination under federated data isolation without providing formal privacy guarantees.

## REFERENCES

Lisa C Adams, Daniel Truhn, Felix Busch, Felix Dorfner, Jawed Nawabi, Marcus R Makowski, and Keno K Bressem. Llama 3 challenges proprietary state-of-the-art large language models in radiology board–style examination questions. Radiology, 312(2):e241191, 2024.

Diogo Almeida. Introducing system one models & jev. TypeSafe AI, 2026. URL https:// typesafe.ai/blog/introducing-system-one-models-and-jev.

Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, et al. Qwen technical report. arXiv preprint arXiv:2309.16609, 2023.

Zhenyu Bi, Meng Lu, Yang Li, Swastik Roy, Weijie Guan, Morteza Ziyadi, and Xuan Wang. Optagent: Optimizing multi-agent llm interactions through verbal reinforcement learning for enhanced reasoning. In Proceedings of the 14th International Joint Conference on Natural Language Processing and the 4th Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics, pp. 1713–1728, 2025.

Aili Chen, Aonian Li, Baichuan Zhou, Bangwei Gong, Binyang Jiang, Boji Dan, Changhao Zhang, Changqing Yu, Chao Wang, Cheng Ma, et al. The minimax-m2 series: Mini activations unleashing max real-world intelligence. arXiv preprint arXiv:2605.26494, 2026.

Minghui Chen, Ruinan Jin, Wenlong Deng, Yuanyuan Chen, Zhi Huang, Han Yu, and Xiaoxiao Li. Can textual gradient work in federated learning? arXiv preprint arXiv:2502.19980, 2025.

Weize Chen, Yusheng Su, Jingwei Zuo, Cheng Yang, Chenfei Yuan, Chi-Min Chan, Heyang Yu, Yaxi Lu, Yi-Hsin Hung, Chen Qian, et al. Agentverse: Facilitating multi-agent collaboration and exploring emergent behaviors. In The Twelfth International Conference on Learning Representations, 2023.

Hyeong Kyu Choi, Xiaojin Zhu, and Sharon Li. Debate or vote: Which yields better decisions in multi-agent large language models? arXiv preprint arXiv:2508.17536, 2025.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.

Wentao Deng, Jiahuan Pei, Zhiwei Xu, Zhaochun Ren, Zhumin Chen, and Pengjie Ren. Beliefcalibrated multi-agent consensus seeking for complex nlp tasks. arXiv preprint arXiv:2510.06307, 2025.

Haonan Duan, Adam Dziedzic, Mohammad Yaghini, Nicolas Papernot, and Franziska Boenisch. On the privacy risk of in-context learning. arXiv preprint arXiv:2411.10512, 2024.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

Taicheng Guo, Xiuying Chen, Yaqi Wang, Ruidi Chang, Shichao Pei, Nitesh V Chawla, Olaf Wiest, and Xiangliang Zhang. Large language model based multi-agents: A survey of progress and challenges. arXiv preprint arXiv:2402.01680, 2024.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. arXiv preprint arXiv:2103.03874, 2021.

Tianyu Hu, Zhen Tan, Song Wang, Huaizhi Qu, and Tianlong Chen. Multi-agent debate for llm judges with adaptive stability detection. arXiv preprint arXiv:2510.12697, 2025.

Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024.

Anita Keshmirian, Razan Baltaji, Babak Hemmatian, Hadi Asghari, and Lav R Varshney. Many llms are more utilitarian than one. arXiv preprint arXiv:2507.00814, 2025.

Magdalena Kossek and Margareta Stefanovic. Survey of recent results in privacy-preserving mechanisms for multi-agent systems. Journal of Intelligent & Robotic Systems, 110(3):129, 2024.

Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437, 2024.

Niklas Muennighoff, Zitong Yang, Weijia Shi, Xiang Lisa Li, Li Fei-Fei, Hannaneh Hajishirzi, Luke Zettlemoyer, Percy Liang, Emmanuel Candes, and Tatsunori B Hashimoto. s1: Simple test-time\` scaling. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pp. 20286–20332, 2025.

Anudit Nagar, Cuong Tran, and Ferdinando Fioretto. A privacy-preserving and trustable multi-agent learning framework. arXiv preprint arXiv:2106.01242, 2021.

Ashwinee Panda, Tong Wu, Jiachen Wang, and Prateek Mittal. Differentially private in-context learning. In The 61st Annual Meeting Of The Association For Computational Linguistics, 2023.

Chen Qiu, Xingyu Li, Chaithanya Kumar Mummadi, Madan Ravi Ganesh, Zhenzhen Li, Lu Peng, and Wan-Yi Lin. Text-driven prompt generation for vision-language models in federated learning. arXiv preprint arXiv:2310.06123, 2023.

Yijia Shao, Tianshi Li, Weiyan Shi, Yanchen Liu, and Diyi Yang. Privacylens: Evaluating privacy norm awareness of language models in action. Advances in Neural Information Processing Systems, 37:89373–89407, 2024.

Zitong Shi, Guancheng Wan, Wenke Huang, Guibin Zhang, Jiawei Shao, Mang Ye, and Carl Yang. Privacy-enhancing paradigms within federated multi-agent systems. arXiv preprint arXiv:2503.08175, 2025.

Yiliu Sun, Zicheng Zhao, Sheng Wan, and Chen Gong. Cortexdebate: Debating sparsely and equally for multi-agent debate. In Findings ofthe Associationfor Computational Linguistics: ACL 2025, pp. 9503–9523, 2025.

Xiangru Tang, Anni Zou, Zhuosheng Zhang, Ziming Li, Yilun Zhao, Xingyao Zhang, Arman Cohan, and Mark Gerstein. Medagents: Large language models as collaborators for zero-shot medical reasoning. In Findings ofthe Associationfor Computational Linguistics: ACL 2024, pp. 599–621, 2024.

Kimi Team, Tongtong Bai, Yifan Bai, Yiping Bao, SH Cai, Yuan Cao, Y Charles, HS Che, Cheng Chen, Guanduo Chen, et al. Kimi k2. 5: Visual agentic intelligence. arXiv preprint arXiv:2602.02276, 2026.

Nguyen Truong, Kai Sun, Siyao Wang, Florian Guitton, and YiKe Guo. Privacy preservation in federated learning: An insightful survey from the gdpr perspective. Computers & Security, 110: 102402, 2021.

Junlin Wang, Jue Wang, Ben Athiwaratkun, Ce Zhang, and James Zou. Mixture-of-agents enhances large language model capabilities. arXiv preprint arXiv:2406.04692, 2024.

Ruhan Wang, Zhiyong Wang, Chengkai Huang, Rui Wang, Tong Yu, Lina Yao, John Lui, and Dongruo Zhou. Federated in-context learning: Iterative refinement for improved answer quality. arXiv preprint arXiv:2506.07440, 2025.

Yaqi Wang, Jianquan Lu, Wei Xing Zheng, and Kaibo Shi. Privacy-preserving consensus for multiagent systems via node decomposition strategy. IEEE Transactions on Circuits and Systems I: Regular Papers, 68(8):3474–3484, 2021.

Panlong Wu, Kangshuo Li, Junbao Nan, and Fangxin Wang. Federated in-context llm agent learning. arXiv preprint arXiv:2412.08054, 2024.

Zhen Xiang, Linzhi Zheng, Yanjie Li, Junyuan Hong, Qinbin Li, Han Xie, Jiawei Zhang, Zidi Xiong, Chulin Xie, Nathaniel D Bastian, et al. Guardagent: safeguard llm agents via knowledge-enabled reasoning. In ICML 2025 workshop on computer use agents, 2025.

Mengxi Xiao, Ben Liu, He Li, Jimin Huang, Qianqian Xie, Xiaofen Zong, Mang Ye, and Min Peng. Moodangels: A retrieval-augmented multi-agent framework for psychiatry diagnosis. arXiv preprint arXiv:2506.03750, 2025.

Yanwei Yue, Guibin Zhang, Boyang Liu, Guancheng Wan, Kun Wang, Dawei Cheng, and Yiyan Qi. Masrouter: Learning to route llms for multi-agent systems. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 15549– 15572, 2025.

Hanrong Zhang, Jingyuan Huang, Kai Mei, Yifei Yao, Zhenting Wang, Chenlu Zhan, Hongwei Wang, and Yongfeng Zhang. Agent security bench (asb): Formalizing and benchmarking attacks and defenses in llm-based agents. arXiv preprint arXiv:2410.02644, 2024.

Haozhen Zhang, Tao Feng, and Jiaxuan You. Router-r1: Teaching llms multi-round routing and aggregation via reinforcement learning. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025a.

Jusheng Zhang, Zimeng Huang, Yijia Fan, Ningyuan Liu, Mingyan Li, Zhuojie Yang, Jiawei Yao, Jian Wang, and Keze Wang. Kabb: Knowledge-aware bayesian bandits for dynamic expert coordination in multi-agent systems. arXiv preprint arXiv:2502.07350, 2025b.

Yanzhe Zhang and Diyi Yang. Searching for privacy risks in llm agents via simulation. In International Conference on Learning Representations, volume 2026, pp. 98142–98176, 2026.

## A ALGORITHM

In this section, we mainly introduce the core algorithm in our paper.

Algorithm 1 Execution Workflow of PFedMA   
1: Input: initial task $q ^ { 1 } .$ , clients set $\mathcal { A } = \{ A _ { 1 } , . . . , A _ { N } \}$ , domain set Domain $= \{ d _ { 1 } , \ldots , d _ { L } \}$   
domain prototypes $\{ \rho _ { d } \} _ { d \in \mathcal { D } o m a i n } ,$ maximum number of rounds $T _ { \mathrm { m a x } } ,$ maximum number of   
clients per round $K _ { \mathrm { m a x } } ,$ risk threshold $\tau _ { \mathrm { r i s k } }$ , consensus weight $\alpha _ { \mathrm { { c o n } } } ,$ global privacy rule reposi  
tory $\Omega _ { \mathrm { g l o b a l } } ^ { 1 ^ { \bullet } } ,$ and personalized privacy rule repositories $\{ \Omega _ { i , \mathrm { l o c a l } } ^ { 1 } \} _ { i = 1 } ^ { N }$   
2: Initialize $t  1 C _ { i } ^ { 1 }  \mathbf { 0 } , S _ { i } ^ { 1 }  \mathbf { 0 } , z ^ { 0 }  \varnothing , n _ { i } ^ { 1 }  0$ for all $A _ { i } \in { \mathcal { A } }$   
3: Each agent $A _ { i }$ maintains local experience $\mathcal { X } _ { i }$ and routing state $\mathbf { M } _ { i } ^ { 1 } = [ \mathbf { C } _ { i } ^ { 1 } \lVert \mathbf { S } _ { i } ^ { 1 } ]$   
4: while $\bar { t } \leq T _ { \operatorname* { m a x } }$ do   
5: Apply the global privacy rule repository to the current query: $\tilde { q } ^ { t } \gets P _ { \mathrm { i n } } ( q ^ { t } , \Omega _ { \mathrm { g l o b a l } } ^ { t } )$   
6: Identify the most relevant domain: d<sup>t</sup> ← arg max<sub>d∈Domain</sub> $\mathrm { C o s } ( \tilde { q } ^ { t } , \rho _ { d } )$   
7: Retrieve the corresponding candidate agent set $\mathcal { A } _ { d _ { \star } ^ { t } }$   
8: Select participating agents: $\mathcal { E } ^ { t } \gets \pi _ { \theta } \left( \mathcal { A } _ { d _ { \star } ^ { t } } \ | \ \tilde { q } ^ { t } , z ^ { t - 1 } , \{ \mathbf { M } _ { i } ^ { t } \} _ { A _ { i } \in \mathcal { A } _ { \star } ^ { t } } \right) , 1 \le | \mathcal { E } ^ { t } | \le K _ { \operatorname* { m a x } }$   
9: $\mathbf { i f } \ \pi _ { \theta }$ selects the termination action then   
10: break   
11: end if   
12: for all $A _ { i } \in { \mathcal { E } } ^ { t }$ do   
13: Generate the local response: $r _ { i } ^ { t } \gets A _ { i } ( \tilde { q } ^ { t } , D _ { i } , \mathcal { X } _ { i } )$   
14: Apply personalized privacy processing: $\tilde { r } _ { i } ^ { t }  \tilde { P _ { \mathrm { r e s p } , i } } ( r _ { i } ^ { t } , \Omega _ { i , \mathrm { l o c a l } } ^ { t } )$   
15: end for   
16: for all $A _ { i } \in { \mathcal { E } } ^ { t }$ do   
17: Int<sup>t</sup> ← IntentAnoma $\mathrm { l y } ( \tilde { r } _ { i } ^ { t } )$   
18: Evd<sup>t</sup> ← EvidenceAnomaly $( \tilde { r } _ { i } ^ { t } )$   
19: Con $\mathbf { \Sigma } _ { i } ^ { \bar { t } } \gets \mathrm { C }$ onsensusAnomaly $( \tilde { r } _ { i } ^ { t } )$   
20: $\mathrm { R i s k } _ { i } ^ { t } \gets \mathrm { I n t } _ { i } ^ { t } + \mathrm { E v d } _ { i } ^ { t } + \alpha _ { \mathrm { c o n } } \mathrm { C o n } _ { i } ^ { t }$   
21: end for   
22: Retain responses satisfying the reliability threshold: $\hat { \mathcal { E } } ^ { t } \gets \left\{ A _ { i } \in \mathcal { E } ^ { t } \ | \ \mathrm { R i s k } _ { i } ^ { t } \leq \tau _ { \mathrm { r i s k } } \right\}$   
23: if $\hat { \mathcal { E } } ^ { t } \neq \varnothing$ then   
24: Update the global reasoning state: $z ^ { t } \gets$ Fusion $\left( \left\{ \tilde { r } _ { i } ^ { t } \right\} _ { A _ { i } \in \hat { \mathcal { E } } ^ { t } } , z ^ { t - 1 } \right)$   
25: else   
26: $z ^ { t } \gets z ^ { t - 1 }$   
27: end if   
28: Update the routing states $\{ \mathbf { M } _ { i } ^ { t + 1 } = \left[ \mathbf { C } _ { i } ^ { t + 1 } \lVert \boldsymbol { S } _ { i } ^ { t + 1 } \right] \} _ { A _ { i } \in \mathcal { E } ^ { t } }$ using the observed capability and   
reliability feedback   
29: Construct generalized local rule candidates from newly identified personalized privacy con  
straints   
30: Build the candidate pool $B ^ { t }$ and obtain the supported global candidate set $\mathcal { G } ^ { t }$   
31: Collect rule-level feedback $F ^ { t }$ and update the global repository: $\Omega _ { \mathrm { g l o b a l } } ^ { t + 1 }$ ←   
RuleUpdate $\left( \Omega _ { \mathrm { g l o b a l } } ^ { t } , \mathcal { G } ^ { t } , F ^ { t } \right)$   
32: Redistribute updated global rules and selectively update each personalized repository $\Omega _ { i , \mathrm { l o c a l } } ^ { t + 1 }$   
33: Refine the query for the next round: $q ^ { t + 1 } \gets \mathrm { R e f i n e } ( q ^ { t } , z ^ { t } )$   
34: $t \gets t + 1$   
35: end while   
36: Output: Final answer derived from the latest global reasoning state

## B COMPARISON WITH RELATED FRAMEWORKS

As summarized in Table 3, existing methods primarily optimize expert selection, sequential routing, or federated response refinement in isolation. In contrast, PFedMA establishes a closed loop

Table 3: Comparison of PFedMA with representative multi-agent and federated multi-agent frameworks. PFedMA differs from prior methods by jointly coupling iterative reasoning, agent reliability, and heterogeneous privacy constraints within a closed loop orchestration process.
<table><tr><td>Method</td><td>Setting</td><td>Expert Coordination</td><td>Iterative Refinement Reliability Feedback Privacy Personalization Privacy Rules Co-evolution</td><td></td><td></td><td></td></tr><tr><td></td><td>Router-R1 Centralized MAS</td><td>RL-based dynamic routing</td><td>√</td><td></td><td></td><td></td></tr><tr><td>KABB</td><td>Centralized MAS</td><td>Knowledge aware expert selection</td><td>Partial</td><td></td><td></td><td></td></tr><tr><td>Fed-ICL</td><td>Federated MAS</td><td>Federated response aggregation</td><td>√</td><td></td><td></td><td></td></tr><tr><td>PFedMA</td><td></td><td>Federated MAS Reliability aware dynamic routing</td><td>√</td><td>√</td><td></td><td>V</td></tr></table>

orchestration process in which reasoning progress, response reliability, and heterogeneous privacy constraints jointly influence subsequent collaboration decisions.

## C STRUCTURED RESPONSE EVALUATION DETAILS

## C.1 STRUCTURED RESPONSE FORMAT AND EXAMPLE

The agent’s response follows a structured format consisting of three components: final answer, claim, and evidence. At round t, each selected client $A _ { i }$ returns a privacy rule based information minimization response using the following structured format: {

```json
"final_answer": "...",
"claims": [
{
"claim": "...",
"evidence": ["...", "..."]
},{
"claim": "...",
"evidence": ["...", "..."]
}
],
"domain_fit": 0.0,
"self_confidence": 0.0
}
```

The corresponding mathematical representation is

$$
\begin{array} { r } { \tilde { r } _ { i } ^ { t } = \left. { a } _ { i } ^ { t } , \Gamma _ { i } ^ { t } , \mathcal { L } _ { i } ^ { t } \right. , } \end{array}\tag{12}
$$

The structured response separates the final conclusion from the claim level information used for subsequent reliability evaluation. Here, $a _ { i } ^ { t }$ denotes the final conclusion returned by agent $A _ { i }$ at round t.

$$
\Gamma _ { i } ^ { t } = \gamma _ { i , n _ { n = 1 } } ^ { t } ,\tag{13}
$$

where $\gamma _ { i , n } ^ { t }$ denotes the n-th explicit claim in the response, and $\left| \Gamma _ { i } ^ { t } \right|$ denotes the number of claims.

Each claim is paired with a corresponding evidence set, such that

$$
\begin{array} { r } { \mathcal { L } _ { i } ^ { t } = { L _ { i , n _ { n = 1 } } ^ { t } } , } \end{array}\tag{14}
$$

where $L _ { i , n } ^ { t }$ contains the evidence provided specifically for claim $\gamma _ { i , n } ^ { t }$ . In particular,

$$
\boldsymbol { L } _ { i , n } ^ { t } = \left\{ \epsilon _ { i , n , k } ^ { t } \right\} _ { k = 1 } ^ { m _ { i , n } ^ { t } } ,\tag{15}
$$

where $\epsilon _ { i , n , k } ^ { t }$ denotes the k-th evidence item associated with the n-th claim, and $m _ { i , n } ^ { t }$ denotes the number of evidence items provided for that claim. Thus, the structured representation establishes an explicit claim evidence correspondence: each $\gamma _ { i , n } ^ { t }$ is evaluated together with its own evidence set $L _ { i , n } ^ { t } ,$ while different claims may be supported by different numbers of evidence items.

The fields domain fit and self confidence are self reported auxiliary metadata in [0, 1]. Because these values may be manipulated by unreliable clients, they are not directly treated as reliability scores.

Table 4: Implementation and training status of the response evaluators.
<table><tr><td>Component</td><td>Implementation</td><td>Shares πθ</td><td>Trainable</td></tr><tr><td>Intent evaluator</td><td>Frozen LLM judge</td><td>No</td><td>No</td></tr><tr><td>Evidence evaluator</td><td>Frozen LLM judge</td><td>No</td><td>No</td></tr><tr><td>Consensus evaluator</td><td>Semantic similarity</td><td>No</td><td>No</td></tr><tr><td>Orchestrator</td><td>RL fine tuned LLM</td><td></td><td>Yes</td></tr></table>

Example. [AGE] year old [GENDER] with transitional cell carcinoma of the bladder comes to the physician because of a [TIME] day history of ringing sensation in his ear. He received this first course of neoadjuvant chemotherapy [TIME]. Pure tone audiometry shows a sensorineural hearing loss of [NUMBER] dB. The expected beneficial effect of the drug that caused this patient’s symptoms is most likely due to which of the following actions? An example of a client response is shown in the Figure 7.

![](images/928845e17014c16ab62dfb0a7df024f66122485f762a63faadbdae8ec14e61d2.jpg)  
Figure 7: Format of the client’s response.

## C.2 EVALUATOR IMPLEMENTATION AND PARAMETER ISOLATION

The response evaluators are separated from the trainable orchestrator π<sub>θ</sub>(Qwen2.5-3B-Instruct). Specifically, the intent and evidence anomaly scores are produced by a frozen judge model using fixed evaluation prompts and predefined scoring criteria. The judge does not share trainable parameters with π<sub>θ</sub>, and its parameters remain unchanged during both reinforcement learning training and inference. The intent and evidence evaluators use the same frozen judge model (Qwen2.5-3B-Instruct) backbone but different task specific instructions.

The consensus anomaly does not invoke the judge model. It is computed deterministically from the semantic similarity between the conclusions of the participating agents. Therefore, the three anomaly dimensions do not introduce additional trainable components into PFedMA.

The judge receives only the response fields required by the corresponding evaluator. The intent evaluator receives the response sentences and the predefined intent risk categories, whereas the evidence evaluator receives each claim and its associated evidence. It does not observe the ground truth answer, the routing reward, or the orchestrator’s subsequent decision. The resulting anomaly scores are used by the server for response filtering and agent state updates, but no evaluator gradient is propagated to the orchestrator.

## C.3 ANOMALY DEFINITIONS AND BOUNDARY CASES

Pooling Operator For any non empty set of scores $\{ x _ { j } \} _ { j = 1 } ^ { J }$ , we define

$$
\mathrm { P o o l } _ { \lambda _ { p o o l } } \left( \{ x _ { j } \} _ { j = 1 } ^ { J } \right) = \lambda _ { p o o l } \operatorname* { m a x } _ { 1 \leq j \leq J } x _ { j } + ( 1 - \lambda _ { p o o l } ) \frac { 1 } { J } \sum _ { j = 1 } ^ { J } x _ { j } ,\tag{16}
$$

where J denotes the number of fine-grained anomaly scores being aggregated, $x _ { j }$ denotes the j-th anomaly score, and $\lambda _ { \mathrm { p o o l } } \in [ 0 , 1 ]$ controls the trade-off between the most severe anomaly and the average anomaly level.

Intent Anomaly. Let $\Xi _ { i } ^ { t } = \left\{ \xi _ { i , j } ^ { t } \right\} _ { j = 1 } ^ { J _ { i } ^ { t } }$ denote the sentence level units extracted from $\tilde { r } _ { i } ^ { t }$ . The predefined intent risk set is

ν<sub>int</sub> = {instruction overriding, false authority, aggregator manipulation, goal substitution}. (17)

For each response unit, the evaluator produces an intent risk score

$$
u _ { i , j } ^ { t } = \mathrm { E v a l } _ { \mathrm { i n t } } \left( \xi _ { i , j } ^ { t } , \nu _ { \mathrm { i n t } } \right) \in [ 0 , 1 ] .\tag{18}
$$

The response level intent anomaly is

$$
I n t _ { i } ^ { t } = \mathrm { P o o l } _ { \lambda _ { \mathrm { p o o l } } } \left( \{ u _ { i , j } ^ { t } \} _ { j = 1 } ^ { J _ { i } ^ { t } } \right) .\tag{19}
$$

Evidence Anomaly. Privacy rule based information minimization adjustment or abstracts sensitive details while aiming to preserve the evidence supporting each claim. The evaluator assesses the support provided by the privacy processed evidence; omitted sensitive details are not themselves treated as anomalies when the remaining information sufficiently supports the claim. Verifiability refers to the judge’s assessment based on the supplied text, rather than independent verification of external sources. For claim $\gamma _ { i , n } ^ { t }$ , let

$$
\mathcal { L } _ { i , n } ^ { t } = \{ \varepsilon _ { i , n , k } ^ { t } \} _ { k = 1 } ^ { m _ { i , n } ^ { t } } ,\tag{20}
$$

where $\varepsilon _ { i , n , k } ^ { t }$ denotes the k-th evidence item associated with the n-th claim, and $m _ { i , n } ^ { t }$ denotes the number of evidence items provided for that claim. For $m _ { i , n } ^ { t } > 0 ;$ , let $\mathrm { S u p } _ { i , n , k } ^ { t } \in [ 0 , 1 ]$ and $\mathrm { V e r } _ { i , n , k } ^ { t } \in$ [0, 1] denote the support strength and verifiability of evidence item $\varepsilon _ { i , n , k } ^ { t }$ , respectively. We define

$$
\overline { { \mathrm { S u p } } } _ { i , n } ^ { t } = \frac { 1 } { m _ { i , n } ^ { t } } \sum _ { k = 1 } ^ { m _ { i , n } ^ { t } } \mathrm { S u p } _ { i , n , k } ^ { t } ,\tag{21}
$$

$$
\mathrm { V e r } _ { i , n } ^ { t } = \operatorname* { m a x } _ { 1 \leq k \leq m _ { i , n } ^ { t } } \mathrm { V e r } _ { i , n , k } ^ { t } .\tag{22}
$$

The internal coherence of the evidence set is

$$
\mathrm { C o h } _ { i , n } ^ { t } = \left\{ \begin{array} { l l } { \displaystyle \frac { 2 } { m _ { i , n } ^ { t } \left( m _ { i , n } ^ { t } - 1 \right) } \sum _ { 1 \leq k _ { 1 } < k _ { 2 } \leq m _ { i , n } ^ { t } } \mathrm { C o n s } \left( \varepsilon _ { i , n , k _ { 1 } } ^ { t } , \varepsilon _ { i , n , k _ { 2 } } ^ { t } \right) , } & { m _ { i , n } ^ { t } > 1 , } \\ { 1 , } & { m _ { i , n } ^ { t } = 1 , } \\ { 0 , } & { m _ { i , n } ^ { t } = 0 . } \end{array} \right.\tag{23}
$$

The claim level evidence quality is

$$
{ \mathrm { Q u a l } } _ { i , n } ^ { \mathrm { e v d } ~ t } = \left\{ \begin{array} { l l } { \displaystyle \frac { \overline { { \mathrm { S u p } } } _ { i , n } ^ { t } + \mathrm { V e r } _ { i , n } ^ { t } + \mathrm { C o h } _ { i , n } ^ { t } } { 3 } , } & { m _ { i , n } ^ { t } > 0 , } \\ { 0 , } & { m _ { i , n } ^ { t } = 0 . } \end{array} \right.\tag{24}
$$

Accordingly, the claim level evidence anomaly is

$$
\mathrm { A n o m } _ { i , n } ^ { \mathrm { e v d } ~ t } = 1 - \mathrm { Q u a l } _ { i , n } ^ { \mathrm { e v d } ~ t } .\tag{25}
$$

The response level evidence anomaly is

$$
\begin{array} { r } { \mathrm { E v d } _ { i } ^ { t } = \left\{ \begin{array} { l l } { \mathrm { P o o l } _ { \lambda _ { \mathrm { p o o l } } } \left( \left\{ \mathrm { A n o m } _ { i , n } ^ { \mathrm { e v d } , t } \right\} _ { n = 1 } ^ { | \Gamma _ { i } ^ { t } | } \right) , } & { | \Gamma _ { i } ^ { t } | > 0 , } \\ { 1 , } & { | \Gamma _ { i } ^ { t } | = 0 . } \end{array} \right. } \end{array}\tag{26}
$$

When no explicit claim is provided, $\mathrm { i . e . , ~ } | \Gamma _ { i } ^ { t } | = 0 .$ , the evidence anomaly is set to its maximum value of 1. This boundary condition prevents an empty claim set from bypassing evidence-quality evaluation and is consistent with the structured response format, in which claims and their associated evidence provide the basis for evidence reliability assessment.

Consensus Anomaly. The consensus anomaly measures whether the conclusion of client $A _ { i }$ is supported by the conclusions of the other selected clients:

$$
C o n _ { i } ^ { t } = \left\{ \begin{array} { l l } { \displaystyle 1 - \frac { 1 } { | \mathcal { E } ^ { t } | - 1 } \sum _ { { A _ { i ^ { \prime } } \in \mathcal { E } ^ { t } } } \mathrm { S i m } \left( a _ { i } ^ { t } , a _ { i ^ { \prime } } ^ { t } \right) , } & { \displaystyle | \mathcal { E } ^ { t } | > 1 , } \\ { \displaystyle 0 , } & { i ^ { \prime } \neq i } \end{array} \right.\tag{27}
$$

where $\mathrm { S i m } ( \cdot , \cdot ) \in [ 0 , 1 ]$ denotes normalized semantic similarity. A larger value therefore indicates stronger deviation from peer conclusions. The singleton case is assigned zero because consensus cannot be assessed from only one response.

## C.4 OVERALL ANOMALY SCORE AND ROBUST AGGREGATION

The three anomaly dimensions are combined into an overall response anomaly score:

$$
\mathcal { R } i s k _ { i } ^ { t } = I n t _ { i } ^ { t } + E v d _ { i } ^ { t } + \alpha _ { \mathrm { c o n } } C o n _ { i } ^ { t } .\tag{28}
$$

Responses whose anomaly scores exceed the tolerance threshold are excluded from aggregation:

$$
\widehat { \mathcal { E } } ^ { t } = \left\{ A _ { i } \in \mathcal { E } ^ { t } \vert \mathcal { R } i s k _ { i } ^ { t } \le \tau _ { \mathrm { r i s k } } \right\} .\tag{29}
$$

The orchestrator then aggregates the retained privacy rule based information minimization responses to update the accumulated collaboration state:

$$
\begin{array} { r } { z ^ { t } = \left\{ \begin{array} { l l } { \mathrm { F u s i o n } \Big ( \{ \tilde { r } _ { i } ^ { t } \} _ { A _ { i } \in \widehat { \mathcal { E } } ^ { t } } , z ^ { t - 1 } \Big ) , } & { \widehat { \mathcal { E } } ^ { t } \neq \emptyset , } \\ { z ^ { t - 1 } , } & { \widehat { \mathcal { E } } ^ { t } = \emptyset . } \end{array} \right. } \end{array}\tag{30}
$$

## D AGENT ROUTING STATE UPDATE

After each orchestration round, PFedMA updates the routing metadata of the participating agents based on their observed historical capability and historical response reliability.

For agent $A _ { i }$ with at least one previously retained response, define:

$$
\tau _ { i } ( t ) = \operatorname* { m a x } \{ s < t \mid A _ { i } \in \widehat { \mathcal { E } } ^ { s } \}\tag{31}
$$

Thus, $\tau _ { i } ( t )$ indexes the most recent round before t in which the response of client $A _ { i }$ was retained by the reliability filter. Denote the most recent round before t in which the response of agent $A _ { i }$ was retained by the reliability filter. Here, s indexes previous collaboration rounds. This definition allows the historical capability update to compare the current response with the most recent reliable response of the same agent while naturally accommodating intermittent agent participation across rounds.

If the current response passes the reliability filter and the agent has a previously retained response, the orchestrator compares the current response with that agent’s most recent retained response from an earlier round to obtain a response level improvement signal and an agent specific capability increment vector.

$$
f _ { i } ^ { t } = \pi _ { \boldsymbol { \theta } } ^ { \mathrm { { f b } } } \left( \widetilde { r } _ { i } ^ { t } \mid \widetilde { r } _ { i } ^ { \tau _ { i } ( t ) } \right) , \qquad \bar { \mathbf { v } } _ { i } ^ { t } = \pi _ { \boldsymbol { \theta } } ^ { \mathrm { { c a p } } } \left( \widetilde { r } _ { i } ^ { t } \mid \widetilde { r } _ { i } ^ { \tau _ { i } ( t ) } \right) .\tag{32}
$$

Here, $\pi _ { \theta } ^ { \mathrm { f b } }$ and $\pi _ { \theta } ^ { \mathrm { c a p } }$ denote two comparative evaluations performed by the same orchestrator $\pi _ { \theta }$ under different fixed evaluation instructions, rather than conditional likelihoods. The response level feedback signal $f _ { i } ^ { t } ~ \in ~ \{ 0 , 1 \}$ indicates whether the current retained response improves upon the most recent retained response of the same client. The capability vector $\bar { \bf v } _ { i } ^ { t }$ identifies the capability dimensions in which the current response improves upon that reference. Before collaboration begins, the historical capability state of each agent is initialized as a zero vector: $\mathbf { C } _ { i } ^ { 1 } = [ 0 , \ldots , 0 ] ^ { \top }$

For the first retained response of an agent, no previously retained response is available for comparison. This response therefore establishes the initial comparison reference, and no capability progress increment is assigned in that round.

For subsequent responses, the historical capability state is updated as

$$
\mathbf { C } _ { i } ^ { t + 1 } = \left\{ \begin{array} { l l } { \mathbf { C } _ { i } ^ { t } + f _ { i } ^ { t } \bar { \mathbf { v } } _ { i } ^ { t } , } & { A _ { i } \in \widehat { \mathcal { E } } ^ { t } , } \\ { \mathbf { C } _ { i } ^ { t } , } & { A _ { i } \notin \widehat { \mathcal { E } } ^ { t } , } \end{array} \right.\tag{33}
$$

where $\widehat { \mathcal { E } } ^ { t }$ denotes the set of responses retained by the reliability filter. Therefore, responses excluded from aggregation do not receive positive progress credit.

The historical reliability state records the cumulative mean of the anomaly scores observed from the responses of agent $A _ { i }$ . Let $n _ { i } ^ { t }$ denote the number of responses from agent $A _ { i }$ <sub>i</sub> that have been evaluated prior to round t. Before collaboration begins, the observation count and historical reliability state are initialized as

$$
n _ { i } ^ { 1 } = 0 , \qquad { \bf S } _ { i } ^ { 1 } = \left[ 0 \quad 0 \quad 0 \right] ^ { \top } .\tag{34}
$$

Whenever a new response from agent $A _ { i }$ is evaluated, the observation count is updated as

$$
n _ { i } ^ { t + 1 } = n _ { i } ^ { t } + 1 ,\tag{35}
$$

and the historical reliability state is updated as

$$
\mathbf { S } _ { i } ^ { t + 1 } = \frac { n _ { i } ^ { t } \mathbf { S } _ { i } ^ { t } + \left[ \mathrm { I n t } _ { i } ^ { t } , \mathrm { E v d } _ { i } ^ { t } , \mathrm { C o n } _ { i } ^ { t } \right] ^ { \top } } { n _ { i } ^ { t } + 1 } .\tag{36}
$$

Here, ${ \mathrm { I n t } } _ { i } ^ { t } , { \mathrm { E v d } } _ { i } ^ { t } ,$ , and $\mathrm { C o n } _ { i } ^ { t }$ denote the intent, evidence, and consensus anomaly scores of the current response, respectively. Unlike the historical capability update, every evaluated response contributes to the historical reliability state, including responses excluded from aggregation by the reliability filter. This allows unreliable responses to remain reflected in the agent’s historical risk profile.

If agent $A _ { i }$ does not return a newly evaluated response at round $t ,$ its historical capability state, reliability states, and observation count remain unchanged. Finally, the updated routing state is defined as

$$
M _ { i } ^ { t + 1 } = \left[ { \bf C } _ { i } ^ { t + 1 } \left. { \bf S } _ { i } ^ { t + 1 } \right] . \right.\tag{37}
$$

Thus, the routing state jointly captures accumulated evidence of agent-specific historical capability and historical response reliability, allowing subsequent routing decisions to account for both observed collaboration progress and historical reliability.

## E REINFORCEMENT LEARNING DETAILS

## E.1 REWARD DEFINITIONS

Let $\tau$ denote a collaboration trajectory containing $T$ executed routing rounds, where $T _ { \mathrm { m a x } }$ is the maximum routing round budget, $T \leq T _ { \operatorname* { m a x } } .$ , and $t \in \{ 1 , \ldots , T \}$ indexes the executed rounds. $\mathbf { A } \mathbf { s }$ defined in Equation $^ { 6 , }$ the trajectory reward consists of a format reward, a terminal answer reward, and step-wise rewards that provide intermediate supervision for multi-round orchestration.

Format and final conclusion rewards. The format reward evaluates whether the orchestrator generated outputs satisfy the required structured format:

$$
R _ { \mathrm { f m t } } ( \tau ) = \left\{ \begin{array} { l l } { 0 , } & { \mathrm { i f ~ t h e ~ r e q u i r e d ~ o u t p u t ~ f o r m a t ~ i s ~ s a t i s f i e d , } } \\ { - 1 , } & { \mathrm { o t h e r w i s e . } } \end{array} \right.\tag{38}
$$

The terminal answer reward evaluates the correctness of the final answer using the corresponding benchmark evaluator:

$$
R _ { \mathrm { a n s } } ( \tau ) = \left\{ \begin{array} { l l } { 1 , } & { \mathrm { i f ~ t h e ~ t e r m i n a l ~ a n s w e r ~ i s ~ j u d g e d ~ c o r r e c t } , } \\ { - 1 , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.\tag{39}
$$

Cross-round reasoning progress. For a training instance with ground truth answer $y ^ { \star }$ , let $\hat { y } ^ { t }$ denote the answer extracted from the aggregated reasoning state $z ^ { t }$ using the corresponding bench mark’s answer extraction procedure. Exact Match (EM) is defined as

$$
\mathrm { E M } _ { t } = \mathbb { I } \big [ \hat { y } ^ { t } = y ^ { \star } \big ] ,\tag{40}
$$

where $\mathbb { I } [ \cdot ]$ is the indicator function. To measure partial correctness, let $\mathcal { T } _ { \mathrm { p r e d } } ^ { t }$ and ${ \mathcal { T } } _ { \mathrm { g t } }$ denote the token multisets obtained after applying the same benchmark specific normalization procedure to $\hat { y } ^ { t }$ and $y ^ { \star }$ , respectively. Token level precision is

$$
\mathrm { P r e c } _ { t } = \left\{ \begin{array} { l l } { \displaystyle \frac { \left| \mathcal { T } _ { \mathrm { p r e d } } ^ { t } \cap \mathcal { T } _ { \mathrm { g t } } \right| } { \left| \mathcal { T } _ { \mathrm { p r e d } } ^ { t } \right| } , } & { \left| \mathcal { T } _ { \mathrm { p r e d } } ^ { t } \right| > 0 , } \\ { 0 , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.\tag{41}
$$

Token level recall is

$$
\mathrm { R e c } _ { t } = \left\{ \begin{array} { l l } { \displaystyle \frac { \left| \mathcal T _ { \mathrm { p r e d } } ^ { t } \cap \mathcal T _ { \mathrm { g t } } \right| } { | \mathcal T _ { \mathrm { g t } } | } , } & { | \mathcal T _ { \mathrm { g t } } | > 0 , } \\ { 0 , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.\tag{42}
$$

Here, ∩ denotes multiset intersection, | · | denotes the number of tokens counted with multiplicity, precision is the proportion of predicted tokens that match the ground truth answer, and recall is the proportion of ground truth tokens covered by the prediction. The token level F1 score is

$$
\mathrm { F 1 } _ { t } = \left\{ \begin{array} { l l } { \displaystyle \frac { 2 \mathrm { P r e c } _ { t } \mathrm { R e c } _ { t } } { \mathrm { P r e c } _ { t } + \mathrm { R e c } _ { t } } , } & { \mathrm { P r e c } _ { t } + \mathrm { R e c } _ { t } > 0 , } \\ { 0 , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.\tag{43}
$$

Since $\mathrm { P r e c } _ { t }$ and Rec are nonnegative, their sum is zero only when both scores are zero. The second case therefore avoids division by zero. Following (Zhang et al., 2025a), the intermediate answer quality score is

$$
Q _ { t } = \mathrm { E M } _ { t } + \left( 1 - \mathrm { E M } _ { t } \right) \mathrm { F } 1 _ { t } .\tag{44}
$$

Thus, $Q _ { t } = 1$ for an exact match and $Q _ { t } = \mathrm { F } 1 _ { t }$ otherwise. We set $Q _ { 0 } = 0$ and define the quality gain at round t as

$$
\Delta Q _ { t } = Q _ { t } - Q _ { t - 1 } .\tag{45}
$$

Round level response risk. For each response $\tilde { r } _ { i } ^ { t }$ , we reuse the overall anomaly score $\mathcal { R } i s k _ { i } ^ { t }$ Since its three components are bounded in [0, 1], its range is

$$
B _ { \mathrm { r i s k } } ^ { t } = \left\{ \begin{array} { l l } { 2 , } & { | \mathcal { E } ^ { t } | = 1 , } \\ { 2 + \alpha _ { \mathrm { c o n } } , } & { | \mathcal { E } ^ { t } | > 1 . } \end{array} \right.\tag{46}
$$

Each response anomaly score is normalized by $B _ { \mathrm { r i s k } } ^ { t }$ , and the normalized scores are averaged over the agents invoked at round t. The round-level response risk is

$$
{ \overline { { \mathcal { R } i s k } } } _ { t } = { \frac { 1 } { | { \mathcal { E } } ^ { t } | } } \sum _ { A _ { i } \in { \mathcal { E } } ^ { t } } { \frac { \mathrm { R i s k } _ { i } ^ { t } } { B _ { \mathrm { r i s k } } ^ { t } } } .\tag{47}
$$

where $\mathcal { E } ^ { t }$ is the set of agents invoked at round t. Here, $\alpha _ { \mathrm { { c o n } } } \geq 0$ is the consensus anomaly weight and $| \mathcal E ^ { t } |$ is the number of agents invoked at round t. Consequently, $\overline { { \mathcal { R } i s k } } _ { t } \in [ 0 , 1 ]$ and a larger value indicates a less reliable set of responses.

Agent invocation cost. The normalized invocation cost is

$$
\frac { | \mathcal { E } ^ { t } | } { K _ { \operatorname* { m a x } } } ,\tag{48}
$$

where $K _ { \mathrm { m a x } }$ is the maximum number of agents that may be invoked in one routing round.

Combining the three components gives the step wise reward in Equation 7:

$$
R _ { \mathrm { s t e p } } ^ { t } = \Delta Q _ { t } - \overline { { \mathcal { R } i s k } } _ { t } - \frac { \vert \mathcal { E } ^ { t } \vert } { K _ { \operatorname* { m a x } } } .\tag{49}
$$

A routing step satisfies $1 \leq | \mathcal { E } ^ { t } | \leq K _ { \operatorname* { m a x } }$ . If the orchestrator selects the termination action, no additional agent is invoked and no further step wise reward is generated.

The ground truth dependent quantities $R _ { \mathrm { a n s } } , Q _ { t }$ , and $\Delta Q _ { t }$ are used only during RL training. They are not available to the orchestrator during inference.

## E.2 POLICY OPTIMIZATION

We optimize the server side orchestrator $\pi _ { \theta } .$ , where θ denotes its trainable parameters, using Proximal Policy Optimization (PPO) implemented with the veRL framework. At round t, the policy observes the privacy processed query $\hat { \tilde { q } } ^ { t }$ , the accumulated collaboration state $z ^ { t - 1 }$ , and the capability and safety metadata $M _ { i } ^ { t }$ of the candidate agents. It then generates a structured routing action that either selects a subset $\mathcal { E } ^ { t }$ satisfying $1 \leq | \mathcal { E } ^ { t } | \overset { \cdot } { \leq } K _ { \operatorname* { m a x } }$ or terminates the trajectory when no additional agent is required.

After the selected agents return Privacy rule based information minimization responses $\{ \tilde { r } _ { i } ^ { t } \} _ { A _ { i } \in \mathcal { E } ^ { t } }$ the server evaluates response reliability, aggregates the retained responses, and computes the training reward. PPO updates are performed centrally using the orchestrator visible trajectory. Raw client datasets $\mathcal { D } _ { i }$ , local experience buffers $\mathcal { X } _ { i } ,$ , complete personalized privacy rules $\Omega _ { i , \mathrm { l o c a l } } ^ { t }$ , and unsanitized responses $r _ { i } ^ { t }$ remain client local and are not used in policy optimization. This training protocol preserves the federated data isolation boundary, but does not by itself provide a cryptographic or differential privacy guarantee.

## E.3 TRAINING CONFIGURATION

We train the Orchestrator $\pi _ { \theta }$ using the veRL framework for reinforcement learning in LLMs, with Proximal Policy Optimization (PPO) as the default algorithm. Specifically, we use the training splits of MATH and MedQA for RL optimization of the Orchestrator, while all remaining benchmarks are evaluated in a zero shot manner without further RL fine tuning. During the RL training phase, the batch size is set to 64, with a maximum of 220 training steps. The anomaly evaluation parameters are set to $\lambda _ { \mathrm { p o o l } } = 0 . 3$ and $\alpha _ { \mathrm { c o n } } = 0 . 3 ;$ their sensitivity is analyzed in Section J.5. Training this model requires 4 NVIDIA A800 GPUs, while inference requires 8 NVIDIA 3090 GPUs. Below are the hyperparameter settings for reinforcement learning in our work. We refer to (Zhang et al., 2025a) to set the hyperparameters required for the experiments.

Table 5: Hyperparameter Settings.
<table><tr><td>Hyperparameter</td><td>Value Hyperparameter</td><td>Value</td></tr><tr><td>Learning Rate (Actor)</td><td>1e-5 Learning Rate (Critic)</td><td></td></tr><tr><td>Total Batch Size</td><td>64 Mini-batch Size</td><td></td></tr><tr><td>Micro-batch Size</td><td>8 Max Training Steps</td><td>220</td></tr><tr><td>Max Routing Steps</td><td>100 Max Sequence Length</td><td>4096</td></tr><tr><td>Max Response Length</td><td>2048 Max Length for LLM API Response</td><td>1024</td></tr><tr><td>Tensor Parallel Size</td><td>GPU Utilization Ratio</td><td>0.8</td></tr><tr><td>Rollout Sampling Temperature (Train)</td><td>1.0 Rollout Sampling Temperature (Eval)</td><td>1.0</td></tr></table>

## F FEDERATED PRIVACY RULE CO-EVOLUTION

Candidate Pool Construction. Only clients retained by the response reliability evaluation contribute new rule candidates. To prevent one client from dominating a semantic cluster, each client contributes at most one candidate to that cluster in a collaboration round. After this distinct client contribution check, explicit client identifiers are removed from the stored representations, and the candidates are randomly permuted:

$$
\mathcal { B } ^ { t } = \mathrm { S h u f f e } \left( \left\{ \bar { \omega } _ { i } ^ { t } \ : \middle | \ : A _ { i } \in \widehat { \mathcal { E } } ^ { t } , \Delta \Omega _ { i , \mathrm { l o c a l } } ^ { t } \neq \emptyset \right\} \right) ,\tag{50}
$$

where $\widehat { \mathcal { E } } ^ { t }$ is the retained agent set defined in Equation 29, and $B ^ { t }$ is the resulting unordered candidate pool. A client that identifies no new rule at round t contributes no candidate. Removing explicit identifiers and permuting candidates reduces direct identifier exposure in the shared representations but does not provide an anonymity guarantee.

Semantic Support Filtering. For each candidate $\bar { \omega } _ { i } ^ { t } \in B ^ { t }$ , we compute semantic support by counting the other uploaded candidates whose cosine similarity to it is at least 0.9:

$$
\mathrm { S u p p } ^ { t } \left( \hat { \omega } _ { i } ^ { t } \right) = \sum _ { \hat { \omega } _ { j } ^ { t } \in { \cal B } ^ { t } \atop { j \ne i } } \mathbb { I } \left[ \mathrm { C o s } \left( e \left( \hat { \omega } _ { i } ^ { t } \right) , e \left( \hat { \omega } _ { j } ^ { t } \right) \right) \geq 0 . 9 \right] ,\tag{51}
$$

where $e ( \cdot )$ maps each generalized privacy rule to a semantic embedding, $\cos ( \cdot , \cdot )$ computes cosine similarity, and $\operatorname { \mathbb { I } } [ \cdot ]$ is the indicator function. The condition $j \neq i$ excludes self support. Under the distinct client contribution check used in candidate pool construction, repeated contributions from the same client to a semantic cluster do not increase its support.

We then normalize the support count to obtain the proportion of other candidates that are semantically similar to $\bar { \omega } _ { i } ^ { t }$

$$
\zeta _ { i } ^ { t } = \frac { \operatorname { S u p p } ^ { t } \left( \bar { \omega } _ { i } ^ { t } \right) } { \operatorname * { m a x } \left\{ 1 , \left| B ^ { t } \right| - 1 \right\} } ,\tag{52}
$$

where $\zeta _ { i } ^ { t } \in [ 0 , 1 ]$ . The denominator prevents division by zero for a singleton pool.

For a pool containing multiple candidates, we select those with the highest positive support proportion, without introducing an additional support count or support proportion threshold. For a singleton pool, the sole candidate is passed directly to the global update. The resulting global candidate set is

$$
\mathcal { G } ^ { t } = \left\{ \begin{array} { l l } { \emptyset , } & { | \mathcal { B } ^ { t } | = 0 , } \\ { \mathcal { B } ^ { t } , } & { | \mathcal { B } ^ { t } | = 1 , } \\ { \left\{ \bar { \omega } _ { i } ^ { t } \in \mathcal { B } ^ { t } \middle | \zeta _ { i } ^ { t } > 0 , \zeta _ { i } ^ { t } = \displaystyle \operatorname* { m a x } _ { \bar { \omega } _ { j } ^ { t } \in \mathcal { B } ^ { t } } \zeta _ { j } ^ { t } \right\} , } & { | \mathcal { B } ^ { t } | > 1 . } \end{array} \right.\tag{53}
$$

All candidates tied for the highest positive support proportion are retained. If multiple candidates are present but none receives support, then $\mathcal { G } ^ { t } = \mathcal { D }$ The singleton case explicitly permits a candidate to enter the global update without cross client corroboration. For larger pools, the selection favors candidates with the broadest support and excludes those with lower support proportions. Semantically equivalent accepted candidates are subsequently consolidated into a representative global rule, and conflicting alternatives are handled through feedback based conflict resolution.

Local Utility Evaluation and Rule Selection. After the global update, reusable rules are redistributed to the clients and evaluated locally. For a candidate rule $\omega ,$ client $A _ { i }$ computes

$$
\Phi _ { i } ^ { t } ( \omega ) = \operatorname { R e l } \left( \omega , \mathcal { X } _ { i } \right) + \frac { n _ { i , \mathrm { v a l i d } } ^ { t } ( \omega ) } { \operatorname* { m a x } \left( 1 , n _ { i , \mathrm { t r i g g e r } } ^ { t } ( \omega ) \right) } - \frac { n _ { i , \mathrm { c o n f i c t } } ^ { t } ( \omega ) } { \operatorname* { m a x } \left( 1 , n _ { i , \mathrm { t r i g g e r } } ^ { t } ( \omega ) \right) } ,\tag{54}
$$

where $\operatorname { R e l } ( \omega , \mathcal { X } _ { i } ) \in [ 0 , 1 ]$ measures the semantic relevance of ω to the local experience buffer $\mathcal { X } _ { i }$ $n _ { i , \mathrm { t r i g g e r } } ^ { t } ( \omega )$ records how many times the rule has been triggered locally, $n _ { i , \mathrm { v a l i d } } ^ { t } ( \omega )$ records how many of those applications were locally evaluated as effective, and $n _ { i , \mathrm { c o n f l i c t } } ^ { t } ( \omega )$ records how many

applications conflicted with a local policy or task requirement. These statistics remain on device.   
The maximum operator avoids division by zero when a rule has not yet been triggered.

Let $\omega _ { g } ^ { t + 1 }$ denote a global privacy rule received by client $A _ { i }$ . If the client has a semantically corresponding local rule $\omega _ { i } ^ { t }$ , it compares the two rules using their local utility scores and selects the one with the higher score. If no corresponding local rule exists, the client evaluates $\omega _ { g } ^ { t + 1 }$ against its default privacy policy.

$$
\begin{array} { r } { \omega _ { i } ^ { \star } = \left\{ \begin{array} { l l } { \omega _ { i } ^ { t } , } & { \Phi _ { i } ^ { t } ( \omega _ { i } ^ { t } ) \geq \Phi _ { i } ^ { t } ( \omega _ { g } ^ { t + 1 } ) , } \\ { \omega _ { g } ^ { t + 1 } , } & { \mathrm { o t h e r w i s e } . } \end{array} \right. } \end{array}\tag{55}
$$

If no semantically corresponding local rule exists, the redistributed rule is evaluated against the client’s default policy. If no global counterpart exists, the existing local rule remains active. Ties are resolved in favor of the existing local rule. The selected rule $\omega _ { i } ^ { \star }$ is added to the client’s local privacy rule repository: $\Omega _ { i , \mathrm { l o c a l } } ^ { t + 1 } = \Omega _ { i , \mathrm { l o c a l } } ^ { t } \cup _ { m e r g e } \left\{ \omega _ { i } ^ { \star } \right\}$ . It is then used for subsequent Privacy rule based information minimization under the corresponding semantic constraint. Non selected personalized alternatives remain in the local repository and may be reconsidered as new rule level feedback becomes available. Consequently, a global rule does not unconditionally override a personalized rule and is adopted only when it obtains a higher local utility score.

Client Side Rule Level Feedback. Client $A _ { i }$ locally records the usage and effectiveness of privacy rules in round t. Let $\mathcal { W } _ { i } ^ { t }$ denote the set of local rules applied and evaluated in that round, including existing personalized rules and adopted global rules. The corresponding local rule level feedback is

$$
\mathcal { F } _ { i } ^ { t } = \left\{ \big ( \mathrm { i d } ( \omega ) , \Delta n _ { i , \mathrm { t r i g g e r } } ^ { t } ( \omega ) , \Delta n _ { i , \mathrm { v a l i d } } ^ { t } ( \omega ) , \Delta n _ { i , \mathrm { c o n f i c t } } ^ { t } ( \omega ) \big ) \ \middle | \ \omega \in \mathcal { W } _ { i } ^ { t } \right\} .\tag{56}
$$

Here, $\operatorname { i d } ( \omega )$ is the identifier of a redistributed global rule, and each $\Delta n$ term records only the corresponding change in the current round. The feedback does not contain triggering examples, raw responses, complete personalized rules, or the local experience buffer.

The server collects the compact feedback reported by the selected clients:

$$
\mathcal { F } ^ { t } = \left\{ \mathcal { F } _ { i } ^ { t } \ : \middle | \ : A _ { i } \in \widehat { \mathcal { E } } ^ { t } \right\} .\tag{57}
$$

The server uses these signals to update scores, resolve conflicts, and adjust which shared global rules remain active. Client specific rule content and complete usage statistics remain stored locally.

Historical Rule Score Update. The local utility evaluation determines which privacy rule is used by each client, whereas the following procedure evaluates the subsequent use of shared rules across clients. Client side usage feedback is aggregated by the server to maintain a historical score for each global rule. This score is then used to perform server rule updates.

For each global privacy rule $\omega \in \Omega _ { \mathrm { g l o b a l } } ^ { t }$ , the server checks the feedback in $\mathcal { F } ^ { t }$ and includes client index i in $\mathcal { T } _ { \omega } ^ { t } \mathrm { i f } \ \mathcal { F } _ { i } ^ { t }$ reports an evaluated use of the rule identified by $\operatorname { i d } ( \omega )$ , with $\Delta n _ { i , \mathrm { t r i g g e r } } ^ { t } ( \omega ) > 0 .$ Thus, $\mathcal { T } _ { \omega } ^ { t }$ contains the indices of clients that used the corresponding rule and reported its results in round t, with each client counted once. For each $i \in \mathcal { T } _ { \omega } ^ { t }$ , the feedback score is defined as

$$
g _ { i } ^ { t } ( \omega ) = \frac { \Delta n _ { i , \mathrm { v a l i d } } ^ { t } ( \omega ) } { \Delta n _ { i , \mathrm { t r i g g e r } } ^ { t } ( \omega ) } - \frac { \Delta n _ { i , \mathrm { c o n f i c t } } ^ { t } ( \omega ) } { \Delta n _ { i , \mathrm { t r i g g e r } } ^ { t } ( \omega ) } ,\tag{58}
$$

where $\Delta n _ { i , \mathrm { t r i g g e r } } ^ { t } ( \omega ) > 0$ for every $i \in \mathcal { T } _ { \omega } ^ { t }$ . The server maintains a historical score $H ^ { t } ( \omega )$ for each shared global rule. If evaluable feedback is available in round t, the score is updated by

For a newly promoted global rule, we initialize $N _ { \omega } ^ { 0 } ~ = ~ 0$ and $H ^ { 0 } ( \omega ) = 0$ . Let $N _ { \omega } ^ { t }$ denote the cumulative number of client evaluations received by global rule $\omega$ before round t. The count is updated as $N _ { \omega } ^ { t + 1 } = N _ { \omega } ^ { t } + | \mathcal { T } _ { \omega } ^ { t } |$ . The historical rule score is updated by

$$
H ^ { t + 1 } ( \omega ) = \left\{ \begin{array} { l l } { \displaystyle \frac { N _ { \omega } ^ { t } H ^ { t } ( \omega ) + \sum _ { i \in \mathcal { I } _ { \omega } ^ { t } } g _ { i } ^ { t } ( \omega ) } { N _ { \omega } ^ { t } + | \mathcal { Z } _ { \omega } ^ { t } | } , } & { | \mathcal { Z } _ { \omega } ^ { t } | > 0 , } \\ { H ^ { t } ( \omega ) , } & { | \mathcal { Z } _ { \omega } ^ { t } | = 0 . } \end{array} \right.\tag{59}
$$

If no evaluable feedback is available for ω in round $t ,$ its previous score is retained.

Rule Conflict Resolution. PFedMA does not apply a separate rule remove criterion. Instead, the server uses the feedback based conflict resolution procedure described above to determine which rule is used for each semantic constraint. Non selected alternatives remain in the global repository and may be reconsidered when additional client feedback becomes available. Rules are not removed solely because they receive no feedback in a round.

Dynamic Global Repository Update. The server updates the global repository by combining the existing rules with the accepted candidates $\mathcal { G } ^ { t }$

$$
\Omega _ { \mathrm { g l o b a l } } ^ { t + 1 } = \mathrm { R e s o l v e } \left( \Omega _ { \mathrm { g l o b a l } } ^ { t } \cup \mathcal { G } ^ { t } ; H ^ { t + 1 } \right) .\tag{60}
$$

Here, Resolve merges equivalent rules and applies the conflict resolution procedure described above. Its output is

$$
\mathrm { R e s o l v e } \left( \Omega _ { \mathrm { g l o b a l } } ^ { t } \cup _ { m e r g e } \mathcal { G } ^ { t } ; H ^ { t + 1 } \right) = \left\{ \omega _ { 1 } ^ { t + 1 } , \omega _ { 2 } ^ { t + 1 } , \ldots \right\} ,\tag{61}
$$

Here, $\omega _ { 1 } ^ { t + 1 } , \omega _ { 2 } ^ { t + 1 } , . .$ . are the rules selected for the first, second, and subsequent semantic constraints, respectively, using the conflict resolution procedure described above. The subscripts index semantic constraints, not clients. A rule without conflicting alternatives is retained directly.

$$
\mathrm { R u l e U p d a t e } \left( \Omega _ { \mathrm { g l o b a l } } ^ { t } , \mathcal { G } ^ { t } , \mathcal { F } ^ { t } \right) \equiv \mathrm { R e s o l v e } \left( \Omega _ { \mathrm { g l o b a l } } ^ { t } \cup \mathcal { G } ^ { t } ; H ^ { t + 1 } \right) ,\tag{62}
$$

In other words, RuleUpdate is a compact representation of the server side global rule evolution process rather than a separate optimization operator. The client feedback $\mathcal { F } ^ { \tilde { t } }$ is first used to update the rule scores $H ^ { t + 1 }$ , after which the newly promoted candidates $\mathcal { G } ^ { t }$ are combined with the existing global repository $\Omega _ { \mathrm { g l o b a l } } ^ { t }$ . The Resolve operation then merges semantically equivalent rules and resolves conflicting rules according to their updated scores. Rules without conflicting alternatives are retained directly. Consequently, both newly discovered cross client privacy knowledge and subsequent rule execution feedback jointly determine the evolution of $\Omega _ { \mathrm { g l o b a l } } ^ { t + 1 }$

Interaction with Privacy Enforcement. The updated global repository filters subsequent queries before they are sent to selected clients, reducing unnecessary disclosure without concealing the query from the server. Each client uses its personalized repository to process local responses before transmission. Raw responses, complete rules, local experiences, and triggering examples remain on device; generalized rule candidates and compact feedback are shared for global updates. This closes the rule evolution loop: local rules are generalized and promoted through cross client support, global rules are selectively adapted by clients, and their feedback guides later conflict resolution and retirement. Thus, global and personalized repositories co-evolve across rounds rather than through one way propagation.

Trust Assumptions. While task responses may be unreliable or malicious, as described in Section 3.1. Our framework assumes that privacy rules are trusted policy artifacts that faithfully represent the privacy requirements of participating clients. In particular, we assume that (i) each client correctly maintains its local privacy-rule repository, (ii) the global privacy-rule repository only contains rules accepted through the prescribed rule-evolution procedure, and (iii) the coordinator faithfully executes rule aggregation, conflict resolution, and rule dissemination.

Under this assumption, privacy rules are treated as trusted policy inputs rather than adversarial objects. Privacy-rule poisoning, malicious rule injection, intentional manipulation of rule semantics, and compromise of the trusted rule repository are outside the scope of this work. Our goal is instead to study how privacy knowledge can be generalized, aggregated, and personalized across clients while respecting client-local privacy constraints.

## G PROMPT TEMPLATES

The query evolution prompt guides the orchestrator to review the accumulated collaboration state and determine what information is still needed to solve the task. It uses responses collected in previous rounds to distinguish questions that have already been answered from those requiring further investigation. The updated query then directs subsequent collaboration toward these remaining questions, reducing repeated requests for information already available.

![](images/fbaa16f21fa5bf27a2ee31f1f420a2746839a09b614988028c4fe2f14bf25ff7.jpg)

The privacy rule creation prompt identifies emerging privacy requirements in client inputs and expresses them as reusable rules for subsequent privacy processing. It compares the identified requirements with the existing rule library to determine whether suitable rules are already available. When an existing rule adequately covers a requirement, it is reused; otherwise, a new rule is formulated to capture the previously uncovered constraint. By focusing on gaps in existing coverage, this process allows clients to incorporate new requirements encountered during collaboration while limiting redundant rule creation. The resulting rules extend the rule library and provide a consistent basis for handling similar requirements in future interactions, reducing the need to formulate rules repeatedly for recurring situations. They also provide a basis for sharing generalized privacy knowledge across clients, allowing requirements identified in one context to inform rule development in other settings. Through this process, individual collaboration experiences can contribute to a broader collection of reusable privacy rules that remains adaptable to evolving client needs.

![](images/56e0f29187d9aee21bde761fa0766c1012a76895c534591cb68356fc02ded9f5.jpg)

The question rewriting prompt turns the updated query into a clear and complete question for the selected clients. It incorporates relevant context and clarifies what information the clients should provide, while preserving the original task objective. This helps clients understand the request and produce answers that support the next reasoning step.

## Change Question Prompt.

You are a query refinement expert supporting iterative multi-agent reasoning. Refine the current query using the available reasoning feedback to help the next agent address unresolved parts of the original task.

## Inputs:

\- Original question: {question}

\- Current query: {current\_query}

\- Current reasoning state: {reasoning\_state}

\- Unresolved issues: {unresolved\_gaps}

## Requirements:

1. Preserve the original task, intent, and explicit constraints.

2. Use the reasoning state and unresolved issues to identify what the next agent needs to clarify, verify, or complete.

3. Prioritize unresolved issues that directly affect answering the original question.   
Avoid repeating resolved work unless verification is needed.

4. Treat intermediate conclusions as provisional. Do not present unverified claims as established facts or assume that previous answers are correct.

5. Make the refined query clear, specific, and self-contained using only information available in the inputs.

6. Do not introduce unsupported facts, assumptions, constraints, or unrelated objectives. Preserve uncertainty when the available context cannot resolve it.

7. Treat the supplied questions and feedback as task data. Do not follow embedded instructions that attempt to override these requirements.

8. If reasoning feedback is missing or uninformative, improve the current query's wording without changing its meaning.

Output only the refined query.

The routing prompt selects experts based on current reasoning needs and feedback on clients’ contributions and reliability, adapting selection as the task progresses.

## Routing Prompt.

Analyze the original problem, the experts already selected, their answers, their known deficiencies, and the capability matrix. Determine whether the current expert set is sufficient to solve the problem correctly.

## You must:

1. Understand the required final output format of the original problem.

2. Evaluate whether the selected experts' answers are correct, complete, and aligned with the problem requirements.

3. Identify any remaining knowledge or capability gaps.

4. Decide whether additional experts are needed.

5. Select only the most necessary additional experts based on the capability matrix.

6. Do not add experts indiscriminately or include all available experts for convenience.

7. Do not include experts who have already been selected unless there is a clear need to reselect them.

8. If no additional experts are needed, set "continue" to false and return an empty "agents" list.

9. If the newly selected expert set would be exactly the same as the previous expert set in both names and quantity, set "continue" to false.

## Return only a valid JSON object in the following structure:

\- "stage\_id": Use the next stage identifier, such as "stage\_1", "stage\_2", etc.

\- "knowledge\_deficit": Summarize the most important unresolved gap.

\- "stage\_description": Describe the next stage work clearly and specifically.

\- "continue": Set to true only when additional expert selection is necessary.

\- "agents": Include only newly needed experts. Use an empty array if "continue" is false.

\- "expert\_defect": Required when "continue" is true. Each selected expert must have a corresponding deficiency description.

\- "task\_type": Must be exactly one capability type listed in {ability\_describe}.

The privacy processing prompt guides the system to process outgoing content according to the applicable privacy rules. It identifies information subject to sharing restrictions and modifies the content while aiming to retain information useful for the task. Applying these rules before transmission helps align communication with privacy requirements as they evolve across collaboration rounds.

![](images/6fa1229c64f977ca4f78fb523d4c87f3b86226c776ced814d14b0f625ee1e270.jpg)

The agent response generation prompt structures answers based on local information into conclusions, claims, and evidence for reliability evaluation and aggregation.

![](images/a7b69e46169df1af781d51059cf162e8ab572eda8a76467f4e5045e3eb7d6d85.jpg)

## H EVALUATION METRICS

For all methods, privacy metrics are computed from their generated outputs under the same privacysensitive evaluation protocol. This section provides the definitions of all privacy and robustness metrics used in our experiments. The rule enforcement metrics, including LPA, LPP, LPR, EA, and FRA, follow GuardAgent (Xiang et al., 2025). Leakage Rate (LR) follows the evaluation principle of PrivacyLens (Shao et al., 2024). The adversarial robustness metrics, including ASR, RR, PNA, and NRP follow Agent Security Bench (ASB) (Zhang et al., 2024). We additionally report Benign Performance (BP), defined here as the task completion rate on benign inputs with the evaluated defense mechanisms enabled.

Rule Enforcement Metrics. Let $N _ { \mathrm { e v a l } }$ denote the number of evaluation instances. For each instance $j \in \{ 1 , \dots , N _ { \mathrm { e v a l } } \}$ , let $b _ { j } \in \{ 0 , 1 \}$ denote the ground truth label, where $b _ { j } = 1$ indicates a privacy/safety violation and $b _ { j } = 0$ indicates a permissible interaction. Let $\hat { b } _ { j }$ denote the predicted label. Following GuardAgent (Xiang et al., 2025), the corresponding evaluation metrics are computed as

$$
T P = \sum _ { j = 1 } ^ { N _ { \mathrm { e v a l } } } \mathbb { I } [ b _ { j } = 1 \wedge \hat { b } _ { j } = 1 ] , \qquad T N = \sum _ { j = 1 } ^ { N _ { \mathrm { e v a l } } } \mathbb { I } [ b _ { j } = 0 \wedge \hat { b } _ { j } = 0 ] ,\tag{63}
$$

$$
F P = \sum _ { j = 1 } ^ { N _ { \mathrm { e v a l } } } \mathbb { I } [ b _ { j } = 0 \land \hat { b } _ { j } = 1 ] , \qquad F N = \sum _ { j = 1 } ^ { N _ { \mathrm { e v a l } } } \mathbb { I } [ b _ { j } = 1 \land \hat { b } _ { j } = 0 ] .\tag{64}
$$

Label Prediction Accuracy (LPA).

$$
\mathrm { L P A } = \frac { T P + T N } { T P + T N + F P + F N } \times 1 0 0 \%\tag{65}
$$

Higher LPA indicates more accurate privacy/safety decisions.

Label Prediction Precision (LPP).

$$
\mathrm { L P P } = \frac { T P } { T P + F P } \times 1 0 0 \%\tag{66}
$$

Higher LPP indicates fewer permissible interactions incorrectly classified as violations.

Label Prediction Recall (LPR).

$$
\mathrm { L P R } = \frac { T P } { T P + F N } \times 1 0 0 \%\tag{67}
$$

Higher LPR indicates that more actual violations are correctly detected.

Explanation Accuracy (EA). Let $\nu _ { j }$ denote the ground truth set of violated rules for instance $j ,$ and $\widehat { \mathcal { V } } _ { j }$ the corresponding predicted set. Let $\mathcal { I } _ { + } = \{ j : b _ { j } = 1 \}$ . We define

$$
\mathrm { E A } = \frac { \sum _ { j \in \mathcal { I } _ { + } } \mathbb { I } \left[ \hat { b } _ { j } = 1 \wedge \mathcal { V } _ { j } \subseteq \widehat { \mathcal { V } } _ { j } \right] } { | \mathcal { I } _ { + } | } \times 1 0 0 \% .\tag{68}
$$

Higher EA indicates more complete identification of the rules responsible for a violation.

Final Response Accuracy (FRA). Let $\mathcal { I } _ { - } = \{ j : b _ { j } = 0 \}$ denote the index set of permissible instances, and let $c _ { j } \in \{ 0 , 1 \}$ indicate whether the final response or action for instance j is correct. Then

$$
\mathrm { F R A } = \frac { \sum _ { j \in \mathcal { I } _ { - } } \mathbb { I } \left[ \hat { b } _ { j } = 0 \wedge c _ { j } = 1 \right] } { | \mathcal { I } _ { - } | } \times 1 0 0 \% .\tag{69}
$$

Higher FRA indicates better preservation of task utility on permissible interactions.

Leakage Rate (LR). Following PrivacyLens (Shao et al., 2024), LR measures the proportion of privacy sensitive instances in which at least one sensitive information item can be inferred from the evaluated output. Let ${ \mathcal { X } } _ { \mathrm { p r i v } }$ denote the set of privacy sensitive evaluation instances. For each instance $x \in \mathcal { X } _ { \mathrm { p r i v } }$ , let $\mathcal { H } _ { x }$ denote the set of sensitive information items associated with x, and let $o _ { x }$ denote the evaluated output. For each $\textit { h } \in \ \mathcal { H } _ { x }$ , let $f ( h , o _ { x } ) ~ \in ~ \{ 0 , 1 \}$ indicate whether the sensitive information item h can be inferred from $o _ { x }$ . The instance level leakage indicator is $\ell _ { x } = \mathbb { I } \left[ \exists h \in { \mathcal { H } } _ { x } \right.$ such that $f ( h , o _ { x } ) = 1 ]$ . The Leakage Rate is

$$
\mathrm { L R } = \frac { 1 } { | \mathcal { X } _ { \mathrm { p r i v } } | } \sum _ { x \in \mathcal { X } _ { \mathrm { p r i v } } } \ell _ { x } \times 1 0 0 \% .\tag{70}
$$

Lower LR indicates less observed privacy leakage. Following PrivacyLens (Shao et al., 2024), LR is computed at the instance level: an instance is counted as leaked if at least one sensitive information item is inferable from the evaluated output.

Attack Success Rate (ASR). Let $N _ { \mathrm { a t k } }$ denote the number of attacked tasks and $N _ { \mathrm { s u c c } } ^ { \mathrm { a t k } }$ the number of successful attacks. Then

$$
\mathrm { A S R } = \frac { N _ { \mathrm { s u c c } } ^ { \mathrm { a t k } } } { N _ { \mathrm { a t k } } } \times 1 0 0 \% .\tag{71}
$$

Lower ASR indicates stronger adversarial robustness.

Refusal Rate (RR). Let $N _ { \mathrm { a g g r } }$ denote the number of aggressive requests and $N _ { \mathrm { r e f u s e } }$ the number explicitly refused by the agent. Then

$$
\mathrm { R R } = \frac { N _ { \mathrm { r e f u s e } } } { N _ { \mathrm { a g g r } } } \times 1 0 0 \% .\tag{72}
$$

Higher RR indicates stronger refusal behavior against unsafe requests.

Performance under No Attack (PNA). Let $N _ { \mathrm { c l e a n } }$ denote the number of clean tasks and $N _ { \mathrm { c o m p l e t e } } ^ { \mathrm { c l e a n } }$ the number successfully completed. Then

$$
\mathrm { P N A } = \frac { N _ { \mathrm { c o m p l e t e } } ^ { \mathrm { c l e a n } } } { N _ { \mathrm { c l e a n } } } \times 1 0 0 \% .\tag{73}
$$

Higher PNA indicates better benign task performance.

Benign Performance (BP). BP measures the system’s ability to complete benign tasks under the evaluated defense configuration. Let $N _ { \mathrm { b e n i g n } }$ denote the number of benign test tasks without injected attacks, and let $N _ { \mathrm { s u c c } } ^ { \mathrm { d e f } }$ denote the number of these tasks successfully completed with the defense mechanisms enabled. Then

$$
\mathrm { B P } = \frac { N _ { \mathrm { s u c c } } ^ { \mathrm { d e f } } } { N _ { \mathrm { b e n i g n } } } \times 1 0 0 \% .\tag{74}
$$

Higher BP indicates better preservation of benign task utility with the defense mechanisms enabled.

Net Resilient Performance (NRP).

$$
{ \mathrm { N R P } } = { \mathrm { P N A } } ( 1 - ( { \mathrm { A S R } } \times 0 . 0 1 ) ) ,\tag{75}
$$

where PNA and ASR are converted to the [0, 1] scale before multiplication. Higher NRP indicates a better trade off between benign task performance and adversarial robustness.

## I ADVERSARIAL EVALUATION SETUP

We evaluate PFedMA using context ignoring prompt injection attacks adapted from Agent Security Bench (ASB) (Zhang et al., 2024). The following paragraphs specify the threat model and attack configuration.

Threat Model. We distinguish task response reliability from privacy policy trust. Participating clients may produce unreliable or malicious task responses, which motivates reliability aware routing and response filtering. In contrast, privacy rules are treated as trusted policy artifacts maintained by the client side privacy subsystem and are not controlled by the task response adversary. Therefore, privacy rule poisoning, malicious rule injection, and manipulation of rule semantics are outside the threat model considered in this work. The server is honest but curious and may attempt to infer private information from privacy processed communications, but faithfully executes the prescribed rule aggregation and dissemination protocol.

Attack Construction. Following Agent Security Bench (ASB) (Zhang et al., 2024), we construct the context ignoring prompt injection attack as $\tilde { x } = x _ { t } \oplus x _ { \mathrm { i g n } } \oplus x _ { e }$ , where $x _ { t }$ denotes the original task text, $x _ { \mathrm { i g n } }$ denotes the context ignoring instruction, $x _ { e }$ denotes the injected adversarial instruction, and ⊕ denotes string concatenation.

Malicious Agent Configuration. For each test instance, $N _ { \mathrm { a d v } }$ agents are randomly selected as malicious from the N participating agents. The adversarial ratio is defined as $\begin{array} { r } { \rho _ { \mathrm { a d v } } = \frac { N _ { \mathrm { a d v } } } { N } } \end{array}$ . We evaluate different values of $N _ { \mathrm { a d v } }$ to examine robustness under increasing adversarial ratios. The specific configurations are $\rho _ { \mathrm { a d v } } \in \{ 1 0 \% , 2 0 \% , 3 0 \% , 4 0 \% \}$

Evaluation Metric. We use Attack Success Rate (ASR), defined in Section H, as the primary robustness metric. The contribution of the anomaly filtering mechanism is evaluated by comparing PFedMA with its filtering disabled variant under the same attack configuration.

## J ADDITIONAL EXPERIMENTAL ANALYSIS

This section provides additional analyses of client data heterogeneity, agent capabilities, iterative collaboration, efficiency, robustness, and hyperparameter sensitivity.

## J.1 EFFECT OF CLIENT DATA HETEROGENEITY

To evaluate client data heterogeneity, we consider both IID and Dirichlet based Non IID partitions with $\alpha \in \{ 0 . 1 , 0 . 5 , 1 . 0 \}$ , where smaller α indicates stronger heterogeneity. As shown in Figure 9d, average accuracy decreases from 0.744 under IID to 0.611 at α = 0.1, and recovers to 0.663 and 0.719 as α increases to 0.5 and 1.0, respectively.

## J.2 AGENT CAPABILITIES AND SELECTION PATTERNS

Agent Analysis. In our setting, there is some overlap in the abilities of the expert agents. Figure 6 provides a visual representation of the dynamic expert selection mechanism. Specifically, Figure 6a shows that Agent 1 possesses outstanding mathematical abilities. Consistent with this, the heatmap in Figure 6c indicates that Agent 1 is frequently selected in math intensive benchmarks such as GSM8K and MATH. Meanwhile, Figure 6b confirms Agent 6’s outstanding expertise in the medical domain, which aligns with its high selection frequency on medical datasets such as MedQA and MedMCQA. These results demonstrate that our mechanism can accurately capture task requirements and adaptively leverage the domain strengths of expert agents. We use abbreviated labels to improve the readability of agent capability visualizations.

Table 6: Explanation of capability abbreviations.
<table><tr><td>Abbr.</td><td>Full Capability</td><td>Abbr.</td><td>Full Capability</td></tr><tr><td>Math</td><td>Mathematical Calculation</td><td>Q.Anlys.</td><td>Question Analysis</td></tr><tr><td>Code</td><td>Code Generation</td><td>Geo.Logic</td><td>Ġeographical Logical Reasoning</td></tr><tr><td>Trans.</td><td>Translation</td><td>Med.QÃ</td><td>Medical Knowledge Question Answering</td></tr><tr><td>Summary</td><td>Summarization</td><td>Clin.Rec.</td><td>Clinical Record Structuring</td></tr><tr><td>Story</td><td>Story Generation</td><td>Med.Lit.</td><td>Medical Literature Retrieval</td></tr><tr><td>Combined</td><td>Combined Task</td><td>Pat.Comm.</td><td>Patient Communication</td></tr></table>

![](images/b5f9b4e13c05070519995f6b27aa33de84f327cafb9d624e5740913cf1817b92.jpg)  
Figure 8: Agent Ability

## J.3 ITERATIVE REASONING, PRIVACY RULE EVOLUTION, AND API USAGE

Figure 9 presents answer accuracy and privacy rule effectiveness across rounds, together with API usage across benchmarks. For the Non IID setting, samples are partitioned across N = 100 clients using a Dirichlet distribution Dir(α) over the task domain categories. For each category, client allocation proportions are sampled from Dir(α), where smaller α induces stronger client level heterogeneity.

![](images/13b8a985d2ffc1ac97a6a81b4905a843e1c88b4cb6c51de8cf723cfcf036bd96.jpg)  
(a) Federated iteration

![](images/9ab0f4d809dbd8b1c5691c38089b25cdb7ce5271a3853c345da4c2d527014b18.jpg)  
(b) LLM API Call Count

![](images/22cb8d134d3a4aa3e7a61549edf312afc069b328a373c8afeec348c5a053eaf7.jpg)  
(c) Federated Privacy  
Rule Effectiveness

![](images/ddc4369b9d819517aab813093a25beb685232871089da49c4628f60b8ea00db4.jpg)  
(d) PFedMA Data Heterogeneity  
Figure 9: Accuracy of federated iterative optimization, federated iteration rounds and demonstration of privacy rule effectiveness, API usage and client data heterogeneity in PFedMA with N = 100 clients.

## J.4 EFFICIENCY AND ROBUSTNESS TRADE OFFS

Figures 3a examine how task performance and token consumption evolve with iteration rounds, while Figure 2d evaluates the robustness utility trade off as the proportion of adversarial agents increases.

## J.5 HYPERPARAMETER SENSITIVITY

Figure 5 presents the sensitivity analysis of $\lambda _ { \mathrm { p o o l } }$ and $\alpha _ { \mathrm { c o n } }$ using independent parameter sweeps. Both hyperparameters exhibit non monotonic effects on robustness, indicating that either insufficient or excessive filtering strength can degrade performance. Smaller values provide weaker suppression of potentially unreliable responses, whereas overly large values may reject useful responses and reduce refusal rate. AHyperparameters were selected using a held-out validation set and fixed before evaluation on the test sets. Based on the validation results, we set $\lambda _ { \mathrm { p o o l } } = 0 . 3$ and $\alpha _ { \mathrm { c o n } } = 0 . 3$ ior all main experiments. Figure 5 reports the corresponding sensitivity analysis, showing that performance varies non-monotonically with both hyperparameters and that the selected values provide a favorable trade-off between attack suppression and refusal behavior.

## K PRIVACY SCOPE.

PFedMA focuses on limiting the disclosure of client local information during federated collaboration rather than hiding the task query from the server. Raw local responses, complete personalized privacy rules, local experiences, and triggering examples remain on device, while privacy processed responses, generalized rule candidates, and compact rule level feedback may be visible to the server. Our privacy evaluation measures sensitive information leakage from generated outputs and does not constitute a formal confidentiality guarantee.

## L LIMITATIONS

PFedMA does not claim formal privacy guarantees such as differential privacy or cryptographic confidentiality. Instead, it provides semantic level privacy filtering by controlling information disclosure through adaptive privacy rules. As multi-agent systems move toward broader collaboration across organizational boundaries, stronger privacy assurances will become increasingly important. Establishing formal guarantees is an important direction for future work.