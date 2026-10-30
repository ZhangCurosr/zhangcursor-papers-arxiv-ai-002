# JUDGING BEFORE PROPAGATION: TASK-GRAPH CON-TEXTUAL INTEGRITY FOR MULTI-AGENT PRIVACY

Hengyu Wu, Blaise Delattre, Yang Cao

Department of Computer Science, School of Computing

Institute of Science Tokyo

wu.h.bc16@m.isct.ac.jp,delattre.b.aa@m.titech.ac.jp,cao@c.titech.ac.jp

## ABSTRACT

Multi-agent LLM systems increasingly rely on persistent memory and inter-agent communication. These capabilities create privacy risks that extend beyond individual interactions: a release appropriate in isolation may become inappropriate when combined with information previously disclosed to a recipient or propagated to downstream agents along the planned workflow. Existing Contextual Integrity (CI) mechanisms for agentic systems typically rely on local judgments that assess each disclosure in isolation, with or without regard to the recipient's history, but do not account for how information propagates downstream. However, compliance at the current boundary does not guarantee appropriateness across the workflow. We introduce Task-Graph Contextual Integrity (TG-CI), to our knowledge the first task-graph-native formalization of Contextual Integrity for multi-agent systems. TG-CI represents a workflow as a Multi-Agent Task Graph (MATG) and extends CI from atomic transfers to stateful, graph-level information flows by jointly considering an item's intended propagation scope and recipients' evolving disclosure states. We operationalize TG-CI through propagation-scoped authorization under two complementary policies: Full-Scope, which requires admissibility across the complete downstream scope, and Auto-Region, which confines each item to its maximal admissible region using lightweight frontier gates. We further introduce MAPLE-Bench, a benchmark of 144 cases across four scenarios that jointly vary MATG topology, disclosure history, compromised-agent placement, and item-level privacy policies. In our primary experimental configuration, TG-CI reduces the mean unauthorized disclosure count by 85.8–94.1% relative to the Local-History CI baseline, while retaining 93.6–96.1% of the baseline's task accuracy, with the two enforcement policies providing complementary privacycost-utility trade-offs.

## 1 INTRODUCTION

Multi-agent systems increasingly combine persistent memory with task delegation and inter-agent communication to solve complex problems (Guo et al., 2024; Tran et al., 2025). While these capabilities improve coordination and continuity, they also enlarge the privacy attack surface: sensitive information retrieved by one agent can be retained (Lahjouji & Colaco, 2026), combined with prior knowledge (Patil et al., 2025), and propagated to other agents throughout the workflow (Lahjouji & Colaco, 2026; Bhosale et al., 2026). Figure 1 illustrates three primary privacy violation modes in such systems. This risk is especially concerning when some agents are compromised. For instance, while disclosing medical appointment metadata to a scheduling service may be permissible in isolation, aggregating this with preexisting recipient data might expose sensitive clinical details, which may then propagate to a compromised downstream notification agent. In such settings, protecting an individual memory access in isolation is insufficient, because the privacy consequences of a disclosure depend also on prior disclosure and the downstream multi-agent workflow.

Contextual Integrity (CI) provides a natural foundation for privacy in multi-agent workflows (Chen et al., 2025), yet how downstream task structure should inform history-aware disclosure judgments remains underexplored. CI evaluates whether an information flow is appropriate within its surrounding context (Nissenbaum, 2004; Barth et al., 2006), rather than relying solely on fixed sensitivity labels. However, existing privacy mechanisms for agentic systems largely operationalize privacy at individual disclosure or interaction boundaries, determining whether a current transfer is acceptable based on its immediate context (Bagdasarian et al., 2024; Yu et al., 2026). The AgentSCOPE (Ngong et al., 2026) benchmark further expands it to multi-tool agentic workflow by labeling internal channels but still considers each information flow independently and provides no defense. Recent work has begun to address cumulative disclosure risks through trajectory-level privacy budgeting (Xie & Li, 2026). Meanwhile, empirical studies identify inter-agent communication as a major channel of privacy leakage in multi-agent systems (Yagoubi et al., 2026), yet comparatively little work considers how the downstream task structure itself should influence an earlier disclosure decision. This leaves a central question underexplored: how can we extend contextual integrity to evaluate an information flow by jointly accounting for direct, cumulative, and transitive disclosure risks within a multi-agent task structure?

![](images/f69c961ef80f0bc037887285cf42824c4ff063ac5f449a7676c9a1f22368f237.jpg)

Figure 1: Three failure patterns in multi-agent systems. Direct disclosure violations occur when the current transfer itself is contextually inappropriate. Sequential Cumulative disclosure violations occur when a current disclosure combined with previous disclosures could reveal inappropriate sensitive information. Transitive disclosure violation occurs when an initially appropriate disclosure would propagate to an inappropriate downstream recipient along the foreseeable task workflow.
<table><tr><td>Paper</td><td>CI Grounded</td><td>Direct Disclosure</td><td>Cumulative Disclosure</td><td>Transitive Disclosure</td></tr><tr><td>AirGapAgent (Bagdasarian et al., 2024)</td><td>Y</td><td>Y</td><td>N</td><td>N</td></tr><tr><td>The Sum Leaks (Patil et al., 2025)</td><td>N</td><td>N</td><td>Y</td><td>N</td></tr><tr><td>OCELOT (Xie &amp; Li, 2026)</td><td>Y</td><td>Y</td><td>Y</td><td>N</td></tr><tr><td>Minim (Yu et al., 2026)</td><td>Y</td><td>Y</td><td>N</td><td>N</td></tr><tr><td>CAMP (Panjwani, 2026)</td><td>N</td><td>Y</td><td>Y</td><td>N</td></tr><tr><td>TG-CI (Ours)</td><td>Y</td><td>Y</td><td>Y</td><td>Y</td></tr></table>

Table 1: Comparison of related work on multi-agent privacy evaluation and protection.

To address this gap, we introduce Task-Graph Contextual Integrity (TG-CI), operationalize it through propagation-scoped authorization with dual-mode enforcement, and develop MAPLE-Bench for controlled evaluation of multi-agent privacy. TG-CI uses a Multi-Agent Task Graph (MATG) to evaluate a current information flow in light of recipients’disclosure histories and intended downstream propagation, jointly accounting for direct, cumulative, and transitive disclosure risks. To make these judgments effective beyond the release boundary, the runtime binds each released item to an authorized propagation scope, allowing either conservative admission or selective downstream containment. MAPLE-Bench jointly varies task structure, recipient disclosure history, and compromised-agent placement to assess whether these controls limit all three disclosure risks while preserving task utility.

## In summary, our contributions are:

(1) Task-graph-native contextual integrity. We formalize TG-CI, to our knowledge the first taskgraph-native CI formalism that judges a current disclosure jointly with its intended downstream propagation scope and recipients’ evolving disclosure states. TG-CI preserves classical CI semantics at each atomic transfer while lifting the judgment to a graph-level information flow.

(2) Propagation-scoped authorization and dual-mode enforcement. We operationalize each positive TG-CI decision as a propagation-scoped authorization and introduce two complementary enforcement policies. Full-Scope Mode blocks an item if any transfer in its complete downstream closure is inadmissible, whereas Auto-Region Authorization computes the maximal rooted admissible region and removes the item at its propagation frontier. This design avoids expensive and repeated contextual re-evaluation along downstream routes while offering complementary trade-offs among privacy protection, computational cost, and task utility.

(3) A benchmark for multi-agent privacy. We release MAPLE-Bench (Multi-Agent Privacy Leakage Evaluation Benchmark), a reusable benchmark of 144 cases spanning four application scenarios, diverse MATG topologies, and compromised-agent configurations. Each case provides item-level memory, privacy policies and ground-truth annotations for disclosure violations, enabling controlled evaluation of information exposure and propagation containment.

## 2 PROBLEM SETTING

System model. We consider a multi-agent system consisting of a set of agents V that collaboratively execute a user task using persistent memory and inter-agent communication. At execution, we represent the current workflow as a directed, labeled Multi-Agent Task Graph (MATG)

$$
G = ( V , E , \tau , \pi ) ,\tag{1}
$$

where $E \subseteq V \times V$ contains planned task-dependent delegation or information-transfer relations, $\tau ( v )$ denotes the current subtask assigned to agent v, and $\pi ( e )$ denotes the edge transmission rule associated with edge e. The MATG differs from a communication-connectivity graph: $( u , v ) \in E$ indicates that the current workflow anticipates a task-relevant interaction from u to v, not merely that the two agents are technically able to communicate.

Memory model. Agents access a persistent memory system M that stores information accumulated from interactions. We model a memory record as a set of information items $m _ { i } = \langle d _ { i } , a _ { i } , x _ { i } \rangle$ where $d _ { i }$ is the data subject, $a _ { i }$ the information attribute or type, and $x _ { i }$ its value or content. This granularity matches the subject and attribute parameters of CI (Barth et al., 2006) and the field-level user data in prior CI evaluations of agents (Bagdasarian et al., 2024; Shao et al., 2024). During task execution, an agent $v \in V$ may issue a memory query $q _ { v }$ and receive a retrieved subset $\mathcal { R } ( q _ { v } ) \subseteq \mathcal { M }$ The formulation does not depend on the memory architecture: $\mathcal { M }$ may be shared, agent-specific, or hierarchical, as in current agent memory systems (Packer et al., 2024; Chhikara et al., 2025; Zhang et al., 2025). We only require that every memory access goes through a system-mediated interface. Access-controlled shared memory makes the same requirement (Rezazadeh et al., 2025).

Adversary model. An unknown subset $C \subseteq V$ with $| C | \leq f$ may be compromised before execution, as in prior studies of malicious agents in LLM multi-agent systems (tse Huang et al., 2025; Liu et al., 2026). A compromised agent may deviate arbitrarily from its intended behavior: it can issue overbroad or adversarial memory requests, misrepresent task necessity, and transform sensitive content in inter-agent messages. The adversary observes all information delivered to agents in $C$ and seeks to maximize the protected information learned by the coalition and the inappropriate flows in the system. We assume that adversarial communication goes through the system-mediated memory and messaging interfaces, as in information-flow control for agents (Debenedetti et al., 2025; Costa et al., 2025). Out-of-band channels and covert encodings in permitted messages, such as steganographic collusion (Motwani et al., 2024), are outside our threat model.

Privacy objective. Let $D _ { t } ( v )$ summarize the privacy-relevant information delivered to recipient v through monitored channels before execution step t (formalized in Section 3). Given the MATG $G$ and disclosure states $D _ { t }$ , we seek a CI-based judgment of whether an item $m _ { i }$ may flow from a source $s \in V \cup \{ \mathcal { M } \}$ to a recipient $r _ { 0 } \in V$ under an intended propagation scope $P \colon$

$$
J ( m _ { i } , s , r _ { 0 } , P \mid G , D _ { t } ) \in \{ 0 , 1 \} ,
$$

where $J = 1$ denotes admissibility. An admissible flow must not cause direct, cumulative, or transitive disclosure violations within its intended propagation scope $P .$ The objective is to prevent inappropriate exposure and propagation while supporting task-relevant information use, not to ensure secrecy from compromised agents when access is contextually justified.

## 3 TASK-GRAPH CONTEXTUAL INTEGRITY

TG-CI lifts an atomic CI judgment to a task-graph information flow: a protected information item is judged together with its intended propagation scope and the evolving disclosure state of the recipients along that scope.

Recipient disclosure state. To capture cumulative disclosure risks, TG-CI maintains the recipientspecific disclosure state introduced in Section 2 as a trusted mapping

$$
D _ { t } : V \to \mathcal { H } ,\tag{2}
$$

where $\mathcal { H }$ denotes the space of privacy-relevant disclosure summaries and $D _ { t } ( v )$ records the information delivered to recipient v through monitored channels before execution step $t .$ Thus, $D _ { t } ( v )$ represents the system-observable disclosure footprint of v.

$\operatorname { I f } \Delta _ { t } ( v )$ denotes the privacy-relevant information actually delivered to v at step t, after any generalization or communication filtering, the state evolves according to

$$
D _ { t + 1 } ( v ) = D _ { t } ( v ) \oplus \Delta _ { t } ( v ) ,\tag{3}
$$

where $\oplus$ incorporates the new disclosure into the recipient's existing state. Authorization alone does not update $D _ { t } ( v ) ;$ the state changes only when information is actually delivered.

Given the MATG $G ,$ the contextual state under which TG-CI evaluates a proposed information flow is

$$
S _ { t } = \langle G , D _ { t } \rangle .\tag{4}
$$

Task-Graph Information Flow. Let s denote the information source. The source is an agent or the memory system M. TG-CI distinguishes the MATG $G = ( V , E , \tau , \pi )$ from the propagation scope of a disclosure, denoted P. The propagation scope is a rooted, labeled graph

$$
P = ( V ^ { P } , E ^ { P } , \tau ^ { P } , \pi ^ { P } ) , \qquad \mathrm { r o o t } ( P ) = s .\tag{5}
$$

Here, $V ^ { P } \subseteq V \cup \{ s \}$ and $P$ includes the initial disclosure edge $( s , r _ { 0 } )$ to the immediate recipient $r _ { 0 } \in V$ . If $s \in V$ , this edge must belong to the $\mathbf { M A T G } ; \mathrm { i f } \ s = \mathcal { M } \notin V .$ it is added as a memoryrelease edge. All subsequent edges satisfy $E ^ { P } \setminus \{ ( s , i _ { 0 } ) \} \subseteq E$ , and every node in $V ^ { P } \setminus \{ s \}$ is reachable from s through a directed path in $P .$

The node-task annotations in P are inherited from the MATG, such that $\tau ^ { P } ( v ) = \tau ( v )$ for every $v \in V ^ { P } \cap V$ . Similarly, the edge transmission rule of each propagation edge is defined as

$$
\pi ^ { P } ( e ) = { \binom { \pi ( e ) , } { \pi _ { 0 } } } \quad e \in E ^ { P } \cap E ,\tag{6}
$$

where $\pi _ { 0 }$ is the edge transmission rule governing the release from memory to $r _ { 0 }$

A task-graph information flow is then defined as

$$
\Phi = \langle d , a , P \rangle .\tag{7}
$$

Atomic CI flow. TG-CI preserves the standard five-parameter CI representation at each individual information-transfer boundary. For every propagation edge $e ~ = ~ ( \bar { u } , v ) ~ \in ~ E ^ { P }$ , we instantiate a task-conditioned transmission principle

$$
\rho ^ { P } ( e ) = \mathrm { I n s t } \left( \pi ^ { P } ( e ) , \tau ^ { P } ( v ) \right) ,\tag{8}
$$

The corresponding atomic CI flow is

$$
\phi _ { e } ( a ) = \left. d , u , v , a , \rho ^ { P } ( e ) \right. .\tag{9}
$$

TG-CI evaluates this flow against the recipient's evolving disclosure state. We define the contextual admissibility of edge e as

$$
\mathsf { A d m } _ { t } ( e , a ) = \mathsf { C l } \left( \phi _ { e } ( a ) \mid D _ { t } ( v ) \right) .\tag{10}
$$

Task-Graph Contextual Integrity. TG-CI treats $\Phi = \langle d , a , P \rangle$ as a single task-graph-level information flow and judges it under the contextual state $S _ { t } = \langle G , D _ { t } \rangle$ . The proposed flow is contextually appropriate if every atomic transfer contained in its propagation scope P is contextually admissible:

$$
\begin{array} { r l } & { \mathsf { T G C l } \left( \Phi , S _ { t } \right) = \displaystyle \bigwedge _ { e = ( u , v ) \in E ^ { P } } \mathsf { A d m } _ { t } ( e , a ) } \\ & { \qquad = \displaystyle \bigwedge _ { e = ( u , v ) \in E ^ { P } } \mathsf { C l } \left( \left. d , u , v , a , \rho ^ { P } ( e ) \right. \big | D _ { t } ( v ) \right) . } \end{array}\tag{11}
$$

Thus, $\mathsf { T G C l } ( \Phi ,  { \mathcal { S } } _ { t } ) = 1$ only if the entire rooted propagation scope is admissible. Although the underlying CI predicate is evaluated at each atomic boundary, the TG-CI judgment is graph-level because the propagation scope $P$ is part of the proposed flow $\mathbf { \bar { \Phi } }$ and the flow is accepted only when all transfers in $P$ are appropriate.

![](images/f27b34904d4c30ca1ba81a3474a475da1a2702ea4703be0f2a29d41130ae4899.jpg)  
Figure 2: Overview of the operationalization of TG-CI and two enforcement policies.

## 4 OPERATIONALIZATION

Architecture overview. Our implementation treats TG-CI as a binary judge and separates it from the mechanisms used to construct and enforce a propagation region. For each memory item $m _ { i }$ , the disclosure controller returns either (RELEASE, $\bar { P _ { i } } )$ , where $P _ { i }$ is the authorized propagation scope, or (BLOCK, ∅). We instantiate this controller in two modes, as shown in Figure 2. Full-Scope Mode evaluates the item over the complete downstream MATG, whereas Auto-Region Mode constructs the maximal rooted region through which the item remains admissible. Both modes use the same TG-CI formalism and differ only in how $P _ { i }$ is selected. We assume that the MATG remains fixed throughout each task execution and leave supporting dynamic replanning to future work.

Item-level authorization and exposure-state tracking. A memory response is first decomposed into semantic items $\mathcal { R } ( q _ { r _ { 0 } } ) = \{ m _ { 1 } , . . . , m _ { n } \}$ , following the memory representation in Section $2 .$ For the disclosure state $D _ { t }$ , the implementation maintains a conservative authorization state ${ \overline { { D } } } _ { t } \left( \mathbf { A p } \cdot \mathbf { \partial } \right.$ pendix $\mathbf { A } )$ , where $\overline { { \boldsymbol { D } } } _ { t } ( \boldsymbol { v } )$ includes information actually delivered to v and items currently authorized to reach v. Each memory response is evaluated using a transaction-local state $\widehat { D }$ initialized from $\overline { { D } } _ { t }$ . Whenever $m _ { i }$ is tentatively authorized over $P _ { i }$ , it is added to $\widehat { \cal D } ( v )$ for every $v \in V ( P _ { i } )$ before the next item is evaluated. This captures cumulative disclosure both within the current response and across previously authorized flows. If the response is authorized, the tentative updates are committed to $\overline { { D } } _ { t } .$ otherwise, they are discarded.

Full-scope mode. For an item $m _ { i } .$ let $P _ { i } ^ { \mathrm { f u l l } }$ denote the complete downstream closure of the requesting agent $r _ { 0 } .$ , augmented with the initial memory-release edge $( s , r _ { 0 } )$ . The controller returns $\left( \mathbf { \bar { R } E L E A S E } , P _ { i } ^ { \mathrm { f u l l } } \right)$ only if

$$
\mathsf { T G C l } \left( \langle d _ { i } , a _ { i } , P _ { i } ^ { \mathrm { f u l l } } \rangle , \widehat { \mathcal { S } } _ { t } \right) = 1 ;\tag{12}
$$

otherwise, it returns (BLOCK, ∅). Consequently, any inadmissible downstream transfer blocks the item before it enters the workflow. This provides the strongest anticipatory containment among our two modes, but is deliberately conservative: a distant recipient can prevent release even when the immediate recipient could appropriately use the item.

Auto-region mode. Auto-Region Mode avoids allowing a distant incompatibility to block an otherwise appropriate current disclosure. For each item, it constructs the maximal rooted propagation region $\bar { P _ { i } ^ { \star } } \subseteq \bar { P _ { i } ^ { \mathrm { f u l l } } }$ reachable from the memory source through contextually admissible edges. If the initial memory-to-agent edge is inadmissible, the item is blocked; otherwise, the controller returns (RELEASE, $P _ { i } ^ { \star } ~$ ). Expansion stops at the first inadmissible edge on each downstream path, producing the item-specific propagation frontier

$$
\begin{array} { r } { \partial P _ { i } ^ { \star } = \left\{ ( u , v ) \in E ( P _ { i } ^ { \mathrm { f u l l } } ) ~ \middle | ~ u \in V ( P _ { i } ^ { \star } ) , ~ ( u , v ) \notin E ( P _ { i } ^ { \star } ) \right\} . } \end{array}\tag{13}
$$

By construction, $P _ { i } ^ { \star }$ satisfies TG-CI, while workflow stages outside the region may continue after the protected item has been removed.

Frontier-aware communication gate. Each active authorization installs an item-specific frontier rule in a lightweight communication gate. For sender u, the gate consults its outgoing neighborhood in the current MATG, $\mathcal { N } _ { G } ^ { + } ( u ) = \{ v \in V \mid ( u , v ) \in E \}$ , and maintains a filter map $\mathcal { F } _ { u } ( v )$ containing protected items prohibited on edge $u  v$ . Messages to recipients v $\notin \mathcal { N } _ { G } ^ { + } ( u )$ are blocked. An on-graph message passes without semantic inspection when $\mathcal { F } _ { u } ( v ) = \emptyset ;$ otherwise, the gate applies semantic claim detection only for the listed items and removes matching content, blocking the message if safe separation is not possible. Thus, expensive content inspection is concentrated at TG-CI frontiers rather than repeated at every communication boundary.

## 5 EXPERIMENTS

## 5.1 MAPLE-BENCH

Motivation for a complementary benchmark. Recent benchmarks have expanded privacy evaluation beyond final outputs, but emphasize different dimensions of multi-agent leakage. AgentSCOPE localizes contextual violations across intermediate agent-tool flows (Ngong et al., 2026), while AgentLeak audits leakage through internal channels, including inter-agent messages and shared memory (Yagoubi et al., 2026). PiSAs annotates both task appropriateness and recipient visibility across agent topologies and memory regimes (Gupta et al., 2026); MAMA isolates the effect of network topology on adversarial memory extraction (Liu et al., 2026); and The Sum Leaks More Than Its Parts studies sensitive inferences formed by composing individually benign disclosures (Patil et al., 2025). These benchmarks establish important individual risk dimensions, but do not provide a single controlled setting that jointly varies MATG topology, recipient disclosure history, compromised-agent placement, and item-level propagation permissions while retaining an objective measure of task completion. MAPLE-Bench complements this literature by combining these factors within the same cases, enabling controlled evaluation of whether a defense can preserve task utility while preventing direct, cumulative, and transitive disclosure.

Benchmark composition. MAPLE-Bench contains 144 cases across four application scenarios, with 36 cases per scenario. Each case instantiates a four-agent MATG and a terminal node representing the user or final-output recipient. Within each scenario, cases are evenly divided among chain, tree, and branching topologies. 108 cases contain one compromised agent and 36 contain two, with compromised-agent locations varied across the MATG. Each case contains 30 synthetic memory items with fictitious values: 12 agent-specific private items, with three associated with each task agent, and 18 shared items. Only seven items contribute to the answers of five multiple-choice questions, either individually or in combination; the remaining 23 items serve as distractors to model memory stores in which most recorded information is irrelevant to the current task, simulating the real-life scenario. The detail of the benchmark is provided in Appendix B.

Privacy annotations and evaluation. Each case provides a task-conditioned item-recipient policy matrix Y, where $Y _ { i , v } = 1$ iff recipient v may receive memory item $m _ { i }$ for its assigned task. Policies cover all four agents and the terminal output recipient, providing ground truth for direct disclosures and for transitive violations. Cases targeting cumulative leakage additionally identify item combinations whose joint disclosure reveals a protected fact. Each benchmark instance therefore specifies the MATG, agent subtasks, compromised-agent configuration, memory items, privacy policies, multiple-choice questions, and gold answers. Because all sensitive values are synthetic, MAPLE-Bench enables measurement of task accuracy, unauthorized and cumulative exposure, and propagation containment without using real personal data.

## 5.2 BASELINES AND EXPERIMENTAL CONDITIONS

Baselines. We compare the two TG-CI enforcement modes against three baselines. No Protection returns all retrieved memory items to the requesting agent and delivers inter-agent messages without privacy judgment or filtering. Local-History CI retains item-level judgment and recipient disclosure-state tracking, and evaluates each item over the initial memory-release edge. Public implementations of the closest cumulation-aware systems, OCELOT and CAMP, were unavailable at the time of evaluation (Xie & Li, 2026; Panjwani, 2026); we therefore use Local-History CI as a controlled in-framework comparison rather than claiming an exact reproduction. We also evaluate Online Atomic CI, which applies the same edge-level atomic CI used by TG-CI independently at every realized memory and inter-agent transfer. The proposed systems are evaluated as TG-CI Full-Scope and TG-CI Auto-Region.

Experimental conditions. We implement the multi-agent workflows using LangChain and use Mem0 as the persistent memory backend. Task agents perform memory retrieval, inter-agent communication, and final-answer generation, while a separate privacy judge evaluates item- and edgelevel contextual admissibility. We evaluate three agent-judge model assignments: LUNA-LUNA, HAIKU-HAIKU, and HAIKU-LUNA, where the two names denote the task-agent and privacy-judge models, respectively. GPT-5.6 Luna and Claude Haiku 4.5 are used as the underlying model families. Unless otherwise specified, all main-text results are reported for the HAIKU-LUNA assignment, with results for remaining agent-judge configurations provided in Appendix C. Frontier filtering is performed locally using MiniLM for semantic candidate retrieval and DeBERTa for bidirectional entailment verification. Details of this procedure are provided in Appendix D.

Metrics. We report four run-level metrics. Unauthorized Disclosure Count is the number of delivery events in which a memory item reaches a recipient for whom it is inappropriate according to the benchmark policy; repeated disclosures are counted separately. For cumulative cases, the delivery that first completes an annotated prohibited item combination is counted as an unauthorized disclosure. Compromised-Agent Exposure Count is the number of distinct inappropriate item observed by compromised agents, counting repeated delivery of the same item to the same agent only once. Judge Execution Token Count is the total number of additional input and output tokens used by privacy-judge LLM calls; local MiniLM and DeBERTa inference is excluded. Utility Score is the fraction of the five multiple-choice questions answered correctly.

## 5.3 PRIVACY-COST TRADE-OFF

![](images/07072ce4b6830f69afc7a73e4413a0495ea29bf9f5a122b0915d312e1858a560.jpg)  
(a) Healthcare

![](images/75e35bd065ccaabd9a2b5aa9860b0a2963bfd0372aace1808d423c4b0af2fd33.jpg)  
(b) Finance

![](images/37b19397b5a85d2810fe749f5eea090d2d7438e4aaab1c3cdaa0eee439480ce8.jpg)  
(c) HR Affairs

![](images/f919c436d17e9016bfe7a1d739921d1746ee1adad178fdb4cd8744bea13e707f.jpg)  
(d) Lifestyle  
Figure 3: Privacy-cost trade-off across four application scenarios. Stacked bars decompose the mean Unauthorized Disclosure Count into mutually exclusive direct, cumulative, and transitive violations (left axis). The dashed line reports Judge Execution Token Count (right axis).

Figure 3 shows a consistent privacy-cost trade-off across all four scenarios: both TG-CI variants incur higher computation than the baselines, but achieve markedly stronger privacy protection. Averaged across scenarios, the unauthorized disclosure count decreases from 15.84 under No Protection to 10.47 under Local-History CI and 6.11 under Online Atomic CI. TG-CI further reduces the count to 1.49 under Auto-Region and 0.62 under Full-Scope. This improvement comes with increased LLM usage: Mean CI-judge token consumption rises to 133.1 K for Auto-Region Mode and 144.5 K for Full-Scope Mode.

The disclosure-type decomposition clarifies where these gains arise. All three violation types occur under No Protection, with transitive disclosures forming the largest component. Local-History CI reduces direct and cumulative violations but leaves substantial transitive leakage. Online Atomic CI further reduces each category, yet transitive disclosure remains its dominant residual source. Both TG-CI variants substantially reduce direct, cumulative, and transitive violations, with their largest advantage over the atomic baselines appearing in the transitive component. These results suggest that atomically checking realized transfers is insufficient in practice and prospectively reasoning over the downstream MATG with either whole-scope admission or maximal admissible-region authorization enforcement provides additional protection against transitive leakage. Categories are mutually exclusive with cumulative classification takes precedence whenever a delivery completes an annotated prohibited combination.

Auto-Region and Full-Scope occupy different points on this frontier. Auto-Region retains a small residual disclosure count while requiring approximately 7.8% fewer judge tokens than Full-Scope. Full-Scope achieves the lowest disclosure count by requiring the complete downstream closure to be admissible. The reported Full-Scope slightly higher cost reflects stricter source-level blocking, which induces additional memory requests by downstream task agents in workflows.

## 5.4 COMPROMISED-AGENT EXPOSURE

![](images/eed875ac30aae778eb5a7e7f8a1a4e17ffcae474b6c2087fef84015c21eb6f2a.jpg)  
Figure 4: Mean number of distinct inappropriate memory items exposed to compromised agents across four application scenarios.

Figure 4 evaluates whether the reduction in disclosure events also limits the distinct protected information acquired by compromised agents. Averaged across the four scenarios, the mean exposure count decreases from 2.90 under No Protection to 1.58 under Local-History CI and 0.99 under Online Atomic CI. TG-CI further reduces exposure to 0.35 under Auto-Region and 0.16 under Full-Scope. The metric counts distinct exposed items rather than repeated disclosure events. Therefore, this result shows that TG-CI does not merely reduce repeated violation events: it effectively limits the distinct protected information available to the compromised coalition. Full-Scope provides the strongest exposure reduction, while Auto-Region retains a higher exposure in exchange for its lower computational cost.

## 5.5 UTILITY EVALUATION

Table 2 shows that TG-CI's privacy gains do not require indiscriminately removing task-relevant information. Across the three agent-judge assignments, Auto-Region retains 84.4–98.8% of No Protection utility and 87.6–96.1% of Local-History CI utility. Full-Scope retains 76.8–96.2% and 79.7–93.6%, respectively. Together with the substantial leakage reductions reported above, these results indicate that TG-CI primarily suppresses contextually inappropriate disclosure while retaining information needed for task completion.

<table><tr><td></td><td colspan="2">HAIKU-LUNA</td><td colspan="2">HAIKU-HAIKU</td><td colspan="2">LUNA-LUNA</td></tr><tr><td>Condition</td><td></td><td>Utility Score vs. NP / LH-CI</td><td>Utility Score</td><td>vs. NP / LH-CI</td><td>Utility Score</td><td>vs. NP / LH-CI</td></tr><tr><td>No Protection</td><td>80.97%</td><td>(—,97.25%)</td><td>81.04%</td><td>(—,103.82%)</td><td>94.72%</td><td>(—,98.55%)</td></tr><tr><td>Local-History CI</td><td>83.26%</td><td>(102.83%,—)</td><td>78.06%</td><td>(96.32%,—)</td><td>96.11%</td><td>(101.47%,—)</td></tr><tr><td>Online Atomic CI</td><td>80.69%</td><td>(99.65%,96.91%)</td><td>74.04%</td><td>(91.36%,94.85%)</td><td>88.47%</td><td>(93.40%,92.05%)</td></tr><tr><td>TG-CI Auto-Region</td><td>80.00%</td><td>(98.80%,96.08%)</td><td>68.40%</td><td>(84.40%,87.62%)</td><td>92.22%</td><td>(97.36%,95.95%)</td></tr><tr><td>TG-CI Full-Scope</td><td>77.92%</td><td>(96.23%,93.58%)</td><td>62.22%</td><td>(76.78%,79.72%)</td><td>88.06%</td><td>(92.96%,91.62%)</td></tr></table>

Table 2: Task utility across agent-judge model assignments. Utility Score denotes mean multiplechoice accuracy in percentage points. Values in parentheses report score relative to No Protection and Local-History CI, respectively.

The two TG-CI modes also provide a clear privacy-utility trade-off. Auto-Region consistently outperforms Full-Scope by 2.1–6.2 percentage. Full-Scope therefore provides stronger containment at the cost of potentially more aggressive removal of potentially useful information. Online Atomic CI generally achieves a similar utility performance as TG-CI Auto-Region and fluctuates depending on specific assignments.

## 5.6 MATG TOPOLOGY

![](images/11d722bcbcb48db6872b5c0e8b98a43053eceb63790de4186b4d864dfdd3f121.jpg)  
(a) Healthcare

![](images/d00256f2e3ae6d32c7fe03216cba541aa1c6aefad2cb5525e034d7d92d08120a.jpg)  
(b) Finance

![](images/b92e6147e64e6701eb243bf6be7e16364fb98560aa2ecc80544123645ef467c3.jpg)  
(c) HR Affairs

![](images/7e3a5274d8940172117cc8d4f6fcfdb57ea3c0b57c46381a49017a9e682eee62.jpg)  
(d) Lifestyle  
Figure 5: Mean unauthorized disclosure count (left axis) across chain, tree, and branching MATG topologies for four application scenarios.

Figure 5 examines unauthorized disclosure across chain, tree, and branching MATGs. The relative ordering of the methods is consistent across all four application scenarios: No Protection produces the most violations, Local-History CI and Online Atomic CI provides a moderate reduction, and both TG-CI variants maintain substantially lower disclosure counts. In particular, TG-CI Auto-Region and Full-Scope show only limited variation across the three topologies, with Full-Scope remaining the strongest protection in every setting. Although branching structures introduce additional potential propagation paths, we observe no systematic degradation in TG-CI performance. These results suggest that the privacy improvements of TG-CI are robust across the evaluated MATG structures rather than being specific to a particular topology.

## 6 CONCLUSION

We introduced Task-Graph Contextual Integrity (TG-CI), a task-graph-native extension of Contextual Integrity that evaluates protected disclosures as stateful information flows over a Multi-Agent Task Graph, jointly accounting for recipient disclosure histories and intended downstream propagation. We operationalized TG-CI through propagation-scoped authorization with two complementary policies: Full-Scope Mode for conservative end-to-end protection and Auto-Region Mode for lowercost frontier-based containment. We further introduced MAPLE-Bench, a controlled benchmark for direct, cumulative, and transitive privacy violations in multi-agent workflows. Our experiments show TG-CI effectively reduced unauthorized disclosure and compromised-agent exposure while retaining substantial task utility. These results demonstrate that moving beyond local disclosure checks toward graph-aware, stateful privacy reasoning can substantially contain information leakage while preserving useful multi-agent collaboration.

## AI USE STATEMENT

In this work, we used generative AI tools to generate synthetic data sets. We have not used generative AI tools to help develop theoretical models or conceptual frameworks, formulate mathematical claims, provide critical ingredients for proving mathematical claims, assist in the writing of proofs, propose or refine hypotheses, design or provide feedback on research methodology or experiments, implement methods, assist with translation, clean and reformat datasets, support qualitative and thematic data analysis, or interpret results. We have reviewed all AI-assisted work. All the synthetic dataset and notation is human-reviewed by the researcher. We take responsibility for the final content of this work, including text, claims, or artifacts produced with the aid of generative AI.

## ETHICS STATEMENT

This work studies privacy risks and defenses in multi-agent systems but does not involve human participants or real personal records. All identities, memory items, sensitive attributes, and task scenarios in MAPLE-Bench are synthetically generated and use fictitious values that are not intended to correspond to actual individuals. The benchmark and data also include task-specific privacy policies, which may encode incomplete or culturally contingent assumptions about appropriate information flow. We consequently do not claim that these policies represent universal privacy norms, and deployment in real settings would require domain expertise, stakeholder input, and compliance with applicable legal and organizational requirements.

## REPRODUCIBILITY STATEMENT

To support reproducibility, we provide our source code and MAPLE-Bench as supplementary materials accompanying this submission. Sections 2–4 specify the problem setting, TG-CI formalization, and the Full-Scope and Auto-Region enforcement procedures. Section 5.1 describes the benchmark composition and privacy annotations, while Section 5.2 presents the baselines, experimental conditions, and evaluation metrics.

## REFERENCES

Eugene Bagdasarian, Ren Yi, Sahra Ghalebikesabi, Peter Kairouz, Marco Gruteser, Sewoong Oh, Borja Balle, and Daniel Ramage. Airgapagent: Protecting privacy-conscious conversational agents. In Proceedings of the 2024 on ACM SIGSAC Conference on Computer and Communications Security, CCS '24, pp. 3868–3882, New York, NY, USA, 2024. Association for Computing Machinery. ISBN 9798400706363. doi: 10.1145/3658644.3690350. URL https://doi.org/10.1145/3658644.3690350.

Adam Barth, Anupam Datta, John C. Mitchell, and Helen Nissenbaum. Privacy and contextual integrity: Framework and applications. In Proceedings of the 2006 IEEE Symposium on Security and Privacy, SP '06, pp. 184–198, USA, 2006. IEEE Computer Society. ISBN 0769525741. doi: 10.1109/SP.2006.32.URL https://doi.org/10.1109/SP.2006.32.

Rohini Bhosale, Pankaj Chandre, Sushma Mehetre, Swati Powar, Shubhra Mathur, and Arun Ghandat. The dark side of autonomous intelligence: a survey on data leakage and privacy failures in agentic ai. Frontiers in Computer Science, Volume 8 - 2026, 2026. ISSN 2624- 9898. doi: 10.3389/fcomp.2026.1802727. URL https://www.frontiersin.org/ journals/computer-science/articles/10.3389/fcomp.2026.1802727.

Kang Chen, Xiuze Zhou, Yuanguo Lin, Shibo Feng, Li Shen, and Pengcheng Wu. A survey on privacy risks and protection in large language models. Journal of King Saud University Computer and Information Sciences, 37(7):163, 2025.

Prateek Chhikara, Dev Khant, Saket Aryan, Taranjeet Singh, and Deshraj Yadav. Mem0: Building production-ready ai agents with scalable long-term memory. arXiv preprint arXiv:2504.19413, 2025.

Manuel Costa, Boris Köpf, Aashish Kolluri, Andrew Paverd, Mark Russinovich, Ahmed Salem, Shruti Tople, Lukas Wutschitz, and Santiago Zanella-Béguelin. Securing AI agents with information-flow control, 2025.

Edoardo Debenedetti, Ilia Shumailov, Tianqi Fan, Jamie Hayes, Nicholas Carlini, Daniel Fabian, Christoph Kern, Chongyang Shi, Andreas Terzis, and Florian Tramèr. Defeating prompt injections by design, 2025.

Taicheng Guo, Xiuying Chen, Yaqi Wang, Ruidi Chang, Shichao Pei, Nitesh V. Chawla, Olaf Wiest, and Xiangliang Zhang. Large language model based multi-agents: A survey of progress and challenges. In Kate Larson (ed.), Proceedings of the Thirty-Third International Joint Conference on Artificial Intelligence, IJCAI-24, pp. 8048–8057. International Joint Conferences on Artificial Intelligence Organization, 8 2024. doi: 10.24963/ijcai.2024/890. URL https: //doi.org/ 10.24963/ijcai.2024/890.SurveyTrack.

Shubham Gupta, Nazanin Mohammadi Sepahvand, Abhinav Kumar, Cem Subakan, Spandana Gella, Pierre-André Noël, Perouz Taslakian, Eugene Bagdasarian, and Valentina Zantedeschi. Pisas: Benchmarking contextual integrity in multi-user agentic systems. arXiv preprint arXiv:2607.05318, 2026.

Nada Lahjouji and Ashwin Gerard Colaco. Agents that know too much: A data-centric survey of privacy in llm agents. arXiv preprint arXiv:2606.26627, 2026.

Jinbo Liu, Defu Cao, Yifei Wei, Tianyao Su, Yuan Liang, Yushun Dong, Yan Liu, Yue Zhao, and Xiyang Hu. Topology matters: Measuring memory leakage in multi-agent LLMs. In Maria Liakata, Viviane P. Moreira, Jiajun Zhang, and David Jurgens (eds.), Findings of the Association for Computational Linguistics: ACL 2026, pp. 39728–39746, San Diego, California, United States, July 2026. Association for Computational Linguistics. ISBN 979-8-89176-395- 1. doi: 10.18653/v1/2026.findings-acl.1980. URL https://aclanthology.org/2026. findings-acl.1980/.

Sumeet Ramesh Motwani, Mikhail Baranchuk, Martin Strohmeier, Vijay Bolina, Philip Torr, Lewis Hammond, and Christian Schroeder de Witt. Secret collusion among AI agents: Multi-agent deception via steganography. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024.

Ivoline C Ngong, Keerthiram Murugesan, Swanand Kadhe, Justin D Weisz, Amit Dhurandhar, and Karthikeyan Natesan Ramamurthy. Agentscope: Evaluating contextual privacy across agentic workflows. arXiv preprint arXiv:2603.04902, 2026.

Helen Nissenbaum. Privacy as contextual integrity. Wash. L. Rev., 79:119, 2004.

Charles Packer, Sarah Wooders, Kevin Lin, Vivian Fang, Shishir G. Patil, Ion Stoica, and Joseph E. Gonzalez. Memgpt: Towards llms as operating systems, 2024.

Aman Panjwani. Camp: Cumulative agentic masking and pruning for privacy protection in multiturn llm conversations. arXiv preprint arXiv:2604.16521, 2026.

Vaidehi Patil, Elias Stengel-Eskin, and Mohit Bansal. The sum leaks more than its parts: Compositional privacy risks and mitigations in multi-agent collaboration. arXiv preprint arXiv:2509.14284, 2025.

Alireza Rezazadeh, Zichao Li, Ange Lou, Yuying Zhao, Wei Wei, and Yujia Bao. Collaborative memory: Multi-user memory sharing in llm agents with dynamic access control, 2025.

Yijia Shao, Tianshi Li, Weiyan Shi, Yanchen Liu, and Diyi Yang. PrivacyLens: Evaluating privacy norm awareness of language models in action. In Advances in Neural Information Processing Systems, 2024.

Khanh-Tung Tran, Dung Dao, Minh-Duong Nguyen, Quoc-Viet Pham, Barry O'Sullivan, and Hoang D Nguyen. Multi-agent collaboration mechanisms: A survey of llms. arXiv preprint arXiv:2501.06322, 2025.

Jen tse Huang, Jiaxu Zhou, Tailin Jin, Xuhui Zhou, Zixi Chen, Wenxuan Wang, Youliang Yuan, Michael Lyu, and Maarten Sap. On the resilience of LLM-based multi-agent collaboration with faulty agents. In Forty-second International Conference on Machine Learning, 2025.

Jin Xie and Songze Li. Ocelot: Inference-leakage budgets for privacy-preserving llm agents. arXiv preprint arXiv:2606.12341, 2026.

Faouzi El Yagoubi, Godwin Badu-Marfo, and Ranwa Al Mallah. Agentleak: A benchmark for internal-channel privacy leakage in multi-agent llm systems. IEEE Access, 14:94960–94978, 2026. doi: 10.1109/ACCESS.2026.3704541.

Hexuan Yu, Chaoyu Zhang, Heng Jin, Shanghao Shi, Ning Zhang, Thomas Hou, and Wenjing Lou. MINIM: Privacy-aware minimal view for agents via trusted local sanitization. In Forty-third International Conference on Machine Learning, 2026. URL https: //openreview.net/ forum?id=rW2xHmYrZA.

Guibin Zhang, Muxin Fu, Kun Wang, Guancheng Wan, Miao Yu, and Shuicheng YAN. G-memory: Tracing hierarchical memory for multi-agent systems. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025.

## A CONSERVATIVE DISCLOSURE STATES

Let $H _ { 1 } ~ \preceq ~ H _ { 2 }$ mean that $H _ { 1 }$ contains no more privacy-relevant information than $H _ { 2 }$ . Since the conservative authorization state includes both delivered information and items authorized to reach a recipient, we have $D _ { t } ( v ) \preceq \overline { { D } } _ { t } ( v )$

Assumption (downward-closed admissibility). For a fixed atomic flow $\varphi _ { e } ( a )$ , acceptance under a larger disclosure footprint implies acceptance under every smaller footprint:

$$
H _ { 1 } \preceq H _ { 2 } \wedge \operatorname { C I } ( \varphi _ { e } ( a ) \mid H _ { 2 } ) = 1 \implies \operatorname { C I } ( \varphi _ { e } ( a ) \mid H _ { 1 } ) = 1 .\tag{14}
$$

Under this assumption, evaluating against $\overline { { \boldsymbol { D } } } _ { t } ( \boldsymbol { v } )$ , or its transaction-local extension $\widehat { \cal D } ( v )$ , is conservative: acceptance also holds under any actual disclosure state bounded by the evaluated footprint. Undelivered authorizations may therefore cause additional rejections, but not false acceptances due solely to state overapproximation. This justification requires the footprint used for authorization to remain an upper bound on the recipient's actual state at delivery.

## B BENCHMARK CONSTRUCTION

<table><tr><td colspan="6">(a) Domain distribution</td></tr><tr><td>Domain</td><td>Workflows</td><td>Cases</td><td>One attacker</td><td></td><td>Two attackers</td></tr><tr><td>Healthcare</td><td>12</td><td>36</td><td></td><td>27</td><td>9</td></tr><tr><td>Finance</td><td>12</td><td>36 36</td><td></td><td>27</td><td>9</td></tr><tr><td>Corporate HR</td><td>12</td><td></td><td></td><td>27</td><td>9</td></tr><tr><td>Lifestyle</td><td>12</td><td>36</td><td></td><td>27</td><td>9</td></tr><tr><td>Total</td><td>48</td><td></td><td>144</td><td>108</td><td>36</td></tr></table>

<table><tr><td colspan="2">(b) Structure, annotations, and reuse</td></tr><tr><td>Graph topologies</td><td>Chain, tree, and branching; 48 cases each.</td></tr><tr><td>Agents per case</td><td>4 executing agents and 1 passive answer receiver.</td></tr><tr><td>Question format</td><td>5 four-option questions per case: 3 single-fact and 2 two-fact questions.</td></tr><tr><td>Question instances</td><td>720 total; 240 workflow-level questions before topology replication.</td></tr><tr><td>Memory per case</td><td>30 items: 12 private (3 per executing agent) and 18 shared.</td></tr><tr><td>Memory relevance</td><td>7 necessary, 7 relevant-only, and 16 unrelated items per case.</td></tr><tr><td>Cumulative-disclosure annotations</td><td>All 144 cases (100%); 2 sensitive fact-pair rules per case; 288 rule instances.</td></tr><tr><td>Single-attacker positions</td><td>P1: 36 cases; P2, P3, and P4: 24 cases each.</td></tr><tr><td>Two-attacker position pairs</td><td>(P1, P2), (P1, P3), and (P2, P4) : 12 cases each.</td></tr><tr><td>Task-template reuse</td><td>Each workflow&#x27;s task, memory, and complete question-answer set are</td></tr><tr><td>Exact question-prompt reuse</td><td>reused across all three topologies. 237 distinct prompts among the 240 workflow-level questions; 3 prompts each appear in two workflows.</td></tr></table>

Table 3: MAPLE-Bench composition and statistics. Each of the 48 workflows is instantiated under three MATG topologies, yielding 144 cases. Attacker counts and position distributions are reported at the case level. $P _ { i }$ represent agent in i-th position.

## B.1 EXAMPLE TASK AND QUESTION

Table 4 presents a task from MAPLE-Bench, one associated two-fact question, and its supporting memory items. Only the two memory items needed for this question are shown.

<table><tr><td>Component</td><td>Content</td></tr><tr><td>Task</td><td>Please confirm my agreed return date, the first-week hours, and whether my fit note is signed. Find a rehabilitation review-call slot that fits my availability and the right place and day to collect my own paperwork copies.</td></tr><tr><td>Two-fact question</td><td>Which offered rehabilitation review call slot meets both the patient&#x27;s availability and the service timetable?</td></tr><tr><td>Answer options</td><td>A. Thursday at 10:30 B. Thursday at 15:30 C. Tuesday at 15:30 D. Tuesday at 10:30</td></tr><tr><td>Memory item 1</td><td>appointment_availability Storage: owner-private; owner: records_agent. &quot;For the rehabilitation review call, the patient can attend on Thursday only. Either 10:30 or 15:30 is acceptable on that day.&quot;</td></tr><tr><td>Memory item 2</td><td>appointment_slot_catalog Storage: shared. “The rehabilitation review call service currently offers 15:30 on both Tuesday and</td></tr><tr><td>Correct answer</td><td>Thursday. It offers no 10:30 slot on either day.&quot; B: Thursday at 15:30.</td></tr><tr><td>Two-fact reasoning</td><td>The availability item leaves options A and B possible, whereas the service timetable leaves options B and C possible. Their intersection uniquely selects B. Neither</td></tr></table>

Table 4: Example of two-fact question answering in MAPLE-Bench. The correct answer requires combining the patient's availability with the service timetable. Memory storage types and ownership are shown alongside the original item contents.

## C ADDITIONAL RESULTS

![](images/61d895fb0e779d0473e08d214db108a18a460f11629b756b98fd03f140efe776.jpg)  
(a) Healthcare

![](images/815aaeca7d3a721b09b04a095107fa1418862fa5b590f434c4ce016400d1e52e.jpg)  
(b) Finance

![](images/76bc2dc3d1eba811db409af6c2b9b8978eae9bda62690e60accbd50c5ebed2b3.jpg)  
(c) HR Affairs

![](images/f417ddcabc6abc54039011e2f2b968e462e21c0646037f4bfa3fd410a2776440.jpg)  
(d) Lifestyle  
Figure 6: Privacy-cost trade-off result under HAIKU-HAIKU agent-judge configuration. Stacked bars decompose the mean Unauthorized Disclosure Count into mutually exclusive direct, cumulative, and transitive violations (left axis). The dashed line reports Extra Execution Token Count relative to No Protection (right axis).

![](images/cca4d4a575514d3152878d100b46607fb7e15c54d055654f72a70caec0f74abc.jpg)  
(a) Healthcare

![](images/8e024251b467933672dfe77b4e2bf7518f736018d31f54241909c38b74f30cfd.jpg)  
(b) Finance

![](images/d7aa42a8094bbd120fe63720fb9460b831822265b525d1e230935b444880e066.jpg)  
(c) HR Affairs

![](images/aff1797c340dc905d9bbcb9d512ec3b89dbc0756ab74008af3de43c91286e7b5.jpg)  
(d) Lifestyle

Figure 7: Privacy-cost trade-off result under LUNA-LUNA agent-judge configuration. Stacked bars decompose the mean Unauthorized Disclosure Count into mutually exclusive direct, cumulative, and transitive violations (left axis). The dashed line reports Extra Execution Token Count relative to No Protection (right axis).  
![](images/e808d832348ffa21e5038b50b01aa5652f8a7f8fb1a7b7bbadd2da66a632a7bd.jpg)  
Figure 8: Mean number of distinct inappropriate memory items exposed to compromised agents under HAIKU-HAIKU agent-judge configuration.

![](images/14f21119005bb9c77470d75f27e6c6056fa62021721c1b3d070e26637f88557d.jpg)  
(a) Healthcare

![](images/ba814dc50e88adeb799e3e31c2a76b34d267538a070dcc4b1cdcdc95798eba08.jpg)  
(b) Finance

![](images/5f452253022083de7d61b5eb3887030ce5e59204ff900b43e63af33077c6189d.jpg)  
(c) HR Affairs

![](images/e3fe4df0c9209f4372e2d5760b58bd1e71c43ffc0ea1b2b6b081eeb55e4b76c2.jpg)  
(d) Lifestyle  
Figure 9: Mean unauthorized disclosure count (left axis) across chain, tree, and branching MATG topologies under HAIKU-HAIKU agent-judge configuration.

![](images/b76c08ecd6976ffbbb79e3d5f72cde882535a6203eed8788907a1e2bd1e62f4d.jpg)  
Figure 10: Mean number of distinct inappropriate memory items exposed to compromised agents under LUNA-LUNA agent-judge configuration.

![](images/1b6364afe5f374b855fe8b680e4e6acb3f814e61d8360ad2e63ece454b1f869c.jpg)  
(a) Healthcare

![](images/4627ebfac51798942b509b6d430f7cf5b5b97111a4aaaf32d0e48429d4c57096.jpg)  
(b) Finance

![](images/f499df8e7ae752ef73dc87d6b8f27af3706256ee268f6011d3a4ea87c59a6f47.jpg)  
(c) HR Affairs

![](images/b5e6b820ccb42a5c65130e6a50a2e99617cbb8c826483a6b3847dc4c87459153.jpg)  
(d) Lifestyle  
Figure 11: Mean unauthorized disclosure count (left axis) across chain, tree, and branching MATG topologies under LUNA-LUNA agent-judge configuration.

## D SEMANTIC FRONTIER FILTERING

The communication gate performs semantic filtering only when an on-graph message $q $ v crosses an edge with a non-empty filter list $\mathcal { F } _ { u } ( v )$ . The message is divided into sentence-like claims

$$
\Delta ( q ) = c _ { 1 } , \ldots , c _ { m } .\tag{15}
$$

For each protected memory item $m _ { i } = \langle d _ { i } , a _ { i } , x _ { i } \rangle$ in $\mathcal { F } _ { u } ( v )$ , the released representation is rendered as a proposition $p _ { i } = \cdot$ 'Information about $d _ { i } \mathrm { ~ s ~ } a _ { i } \colon x _ { i } . ^ { \mathrm { ~ } \prime \mathrm { ~ } }$ Only the representation previously released by the memory controller is used; the filter does not access a more detailed, unreleased value.

Lexical and semantic screening. The gate first applies a normalized lexical check to determine whether the protected value $x _ { i }$ occurs directly in claim $c _ { j }$ Matching pairs are immediately flagged. For the remaining pairs, we embed $c _ { j }$ and $p _ { i }$ using sentence-transformers/all-MiniLM-L6-v2 andcompute

$$
s _ { j i } = \cos ( E ( c _ { j } ) , E ( p _ { i } ) ) .\tag{16}
$$

Pairs with $s _ { i i } \mathrm { ~  ~ { ~ \mu ~ \leq ~ \mu ~ } ~ } 0 . 3 5$ are discarded, while the remaining pairs are sent to $\mathtt { c r o s s - e n c o d e r / n l i - d e b e r t a - v 3 - b a s e }$ Similarities above 0.75 are additionally recorded as high-risk events for audit purposes but do not bypass entailment verification.

Bidirectional entailment and redaction. For each retained pair, the NLI model is evaluated in both directions. Let

$$
e _ { j i } ^ { \right. } = \mathrm { P r } ( \mathrm { E N T A I L } \mid c _ { j } , p _ { i } ) , \qquad e _ { j i } ^ { \left. } = \mathrm { P r } ( \mathrm { E N T A I L } \mid p _ { i } , c _ { j } ) .\tag{17}
$$

Forward entailment captures claims that establish the protected fact, whereas reverse entailment captures broader claims implied by that fact. The disclosure score is

$$
L ( c _ { j } , m _ { i } ) = \operatorname* { m a x } \left( e _ { j i } ^ { \right. } , e _ { j i } ^ { \left. \right)}  ,\tag{18}
$$

and claim $c _ { j }$ is flagged when $L ( c _ { j } , m _ { i } ) \ge 0 . 7 0$ Flagged content is removed and the remaining message is rechecked; if a safe residual message cannot be produced, the gate blocks the complete message. All embedding and NLI comparisons are executed as batched local inference rather than individual API calls.

## E PROVENANCE-PRESERVING ABSTRACTIONS.

The core TG-CI formalism does not explicitly model arbitrary semantic transformations performed by intermediate agents. Let

$$
a ^ { \prime } \preceq a\tag{19}
$$

denote a provenance-preserving abstraction of $a ,$ meaning that $a ^ { \prime }$ reveals no more protected information than a.

We assume contextual admissibility is monotone under such abstractions:

$$
a ^ { \prime } \preceq a \land \mathsf { A d m } _ { t } ( e , a ) = 1 \implies \mathsf { A d m } _ { t } ( e , a ^ { \prime } ) = 1 .\tag{20}
$$

Thus, TG-CI conservatively evaluates the propagation of a itself. A provenance-preserving abstraction $a ^ { \prime } \preceq a$ may traverse any edge already authorized for a, but abstraction does not automatically authorize propagation outside $P .$

Transformations that combine a with independent information, reveal a new protected attribute, or otherwise increase privacy-relevant information are not treated as abstractions under Eq. 20; they require a separate privacy-state update or TG-CI judgment.

## F MODEL IDENTIFIERS AND CI JUDGE SPECIFICATION

## F.1 MODEL IDENTIFIERS

Table 5 lists the model identifiers used for the task agents and the contextual integrity judge. In an agent-judge configuration, the first model denotes the task-agent model and the second denotes the privacy-judge model. Accordingly, Claude/GPT, Claude/Claude, and GPT/GPT correspond to HAIKU–LUNA, HAIKU–HAIKU, and LUNA-LUNA, respectively.

<table><tr><td>Model label</td><td>Manuscript alias</td><td>Model identifier</td></tr><tr><td>GPT</td><td>LUNA</td><td>gpt-5.6-luna</td></tr><tr><td>Claude</td><td>HAIKU</td><td>claude-haiku-4-5-20251001</td></tr></table>

Table 5: Model identifiers and their corresponding manuscript aliases.

## F.2 ATOMIC CI JUDGE RUBRIC

The CI judge evaluates a proposed atomic information flow using the data subject, information attribute, sender role, recipient role, and transmission principle. The transmission principle contains the trusted edge transmission principle, the recipient's assigned task, and the applicable norms. Candidate information is supplied explicitly as text, and the judge is instructed to evaluate its meaning rather than rely on its disclosure-level label

The rubric distinguishes disclosure permission from task completion: an appropriate input need not independently answer the recipient's task. Recipient history is used to assess whether the candidate would complete an inappropriate cumulative disclosure, but history does not itself authorize the current flow. A nul1 history excludes history from the judgment, whereas [] indicates that no prior exposure was recorded. The complete rubric is provided in Listing 1.

![](images/72494061a5da4986c3ed6510f119fe4b957493a3d762476092a3f503bec5a719.jpg)  
Listing 1: Instruction rubric for the atomic CI judge.

## F.3 USER-MESSAGE STRUCTURE AND OUTPUT SCHEMA

The user message contains two top-level fields: context and output\_schema. The context specifies the candidate representation, recipient disclosure history, and atomic flow. Each applicable norm is represented by its identifier, source, text, and citation field.

The requested response contains exactly three fields: maximum\_admissible\_level, reason\_codes, and applied\_norm\_ids. The latter two fields are nonempty arrays. The rubric requests structured reason codes and norm identifiers, but no free-form reasoning field.

Listing 2 shows an example involving the disclosure of a service window to a service coordinator. It illustrated call decides whether the supplied representation is admissible.

{   
"disclosure\_history": [],   
"flow": {   
"attribute": "service\_window",   
"recipient": {   
"role": "Service coordinator"   
},   
"sender": {   
"role": "Protected persistent-memory source"   
},   
"subject": "jordan",   
"transmission\_principle": {   
"applicable\_norms":[

```csv
"citation": null,
"norm_id": "service_purpose",
"source": "workflow_policy",
"text": "Disclose only information appropriate for arranging
the requested service, and no more revealing than needed."
}
],
"edge_transmission_principle": "Use operational details only to
arrange the requested service.",
"recipient_task": "Determine the service arrangements."
}
},
"candidates": [
3
"disclosure_level":"exact",
"value": "The service window is Tuesday morning."
}
1,
"output_schema": {
"$defs": {
"CIReasonCode": {
"description": "Generic auditable factors returned by a CI judge
. ",
"enum":[
"task_necessary",
"minimum_necessary",
"task_mismatch",
"transmission_principle_mismatch",
"norm_prohibits",
"cumulative_disclosure_risk",
"downstream_recipient_risk",
"insufficient_context",
"other"
],
"title": "CIReasonCode",
"type": "string"
},
"DisclosureLevel": {
"description": "Representations ordered from no disclosure to
exact disclosure.",
"enum": [
"blocked",
"exact"
],
"title": "DisclosureLevel",
"type": "string"
一
},
"additionalProperties": false,
"description": "Most-specific admissible representation from one per
edge joint call.",
"properties": {
"applied_norm_ids": {
"items": {
"minLength": 1,
"pattern": "^[a-z][a-z0-9_-]*$",
"type": "string"
},
"minItems": 1,
"title": "Applied Norm Ids",
"type": "array"
},
"maximum_admissible_level": {
"$ref": "#/$defs/DisclosureLevel"
},
```

```json
"reason_codes": {
"items": {
"$ref": "#/$defs/CIReasonCode"
},
"minItems": 1,
"title": "Reason Codes",
"type": "array"
}
},
"required": [
"maximum_admissible_level",
"reason_codes",
"applied_norm_ids"
1,
"title": "CIJointLevelJudgeOutput",
"type": "object"
}
```  
Listing 2: Example user-message payload and requested output schema.