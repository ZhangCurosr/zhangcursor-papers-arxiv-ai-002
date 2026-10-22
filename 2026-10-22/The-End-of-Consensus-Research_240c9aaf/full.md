# The End of Consensus Research?

Clément Burgelin Rachid Guerraoui Zicong Hong Yinan Hu

Anne-Marie Kermarrec Rafael Pires Rasmus Moorits Veski Martijn de Vos

Authors listed in alphabetical order

École Polytechnique Fédérale de Lausanne (EPFL) Lausanne, Switzerland

## Abstract

Consensus is a fundamental problem in distributed computing and at the core of state machine replication (SMR). De spite decades of active human research, discovering practical low-latency solutions to the problem is still open. This en deavor is very challenging since it requires combining proto col knowledge, correctness reasoning, implementation, and experimental evaluation. We report on how AI helped us discover A-GREEMENT, a new consensus protocol for georeplicated state machines that significantly outperforms state of-the-art solutions. Averaged across six deployments with 5 or 9 replicas, A-GREEMENT reduces the mean latency of a geo-distributed key-value store over the fastest baseline by 12.2% and 33.3% on write-heavy and read-heavy workloads, respectively, and matches the fastest baseline on read-only workloads. A- combines lease-authorized local reads, direct client learning for leader-ordered writes, and a fast fixed-quorum per-key write path. It selects replica roles and write paths according to network topology and workload, accounting for both write completion and read waiting. At the heart of our consensus discovery lies AI-GREEMENT, an agent harness for the “autonomous” discovery, implementation, experimentation, and verification of candidate protocols. AI-GREEMENT integrates (1) a concept-oriented protocol knowledge base that explores consensus mechanisms through published protocols, (2) a modular implementation skeleton that structures protocol development, and (3) a LEAN-based verification framework with reusable safety definitions, refer ence specifications, and proofs. Our harness supports iterative design exploration, using performance measurements and fault-injection feedback to refine protocols. AI-GREEMENT is of independent interest, beyond A-GREEMENT. Its LEAN formalization uncovers several previously unreported vulnerabilities in existing consensus protocols. AI-GREEMENT provides guidelines on how domain-specific harnesses can turn general-purpose coding agents into effective tools for (distributed) computing research. Yet we argue that human consensus experts remain essential before, during, and after the search: agents reshape the researcher’s role rather than

end consensus research.

## 1 Introduction

In state machine replication (SMR) [37, 65], a universal paradigm to build highly-available computing systems, consensus enables replicas to agree on a common sequence of client commands, maintaining consistent state despite concurrent requests and failures. Consensus is a fundamental building block of modern distributed systems, underpinning replicated databases [17,30,71], storage systems [6], cloud coordination services [10] and blockchain infrastructures [8,81]. Consensus has been an active research area for over four decades. However, designing and implementing consensus protocols that are both correct and practically efficient remains an exceptionally challenging task [12, 55]. Protocol decisions involving ordering, quorums, recovery, and execution are tightly coupled: an optimization that shortens the common-case path may complicate recovery or introduce subtle correctness violations [12, 64, 70], as we also find for the optimizations of EPaxos Revisited [75] (§ 8). Translating a protocol design into a working system further requires preserving its safety properties throughout implementation optimizations [25,68,79], as well as demonstrating performance improvements across diverse network topologies and workloads. Advancing the state of the art in consensus therefore requires tightly integrating protocol knowledge, correctness reasoning, implementation and experimental evaluation.

The overall question we address in this paper is whether, just like in other fields of computing, AI could help “crack” the consensus problem. More specifically, we aim to come up with a new practically effective solution that would, on the one hand, tolerate, besides failures of the processes, failures of the network (i.e., asynchronous periods) and, on the other hand, expedite decisions in common failure-free executions. In particular, we ask whether we can develop a consensus protocol that would yield an SMR scheme in which, in the common case, (a) read-only commands complete at a nearby replica and (b) commands with updates take one round trip to a majority of replicas. This would essentially be optimal. Importantly, and unlike many recent AI-assisted results in mathematics and theoretical computer science, we seek a protocol that is practically useful and hence, besides algorithmic design and proofs, consider the implementation, deployment and evaluation as part of the AI-assisted task.

In short, we show in this paper that the answer is positive. Yet, we highlight the crucial role of the human researcher in the development of an adequate AI agent, the prompting of the agent as well as the certification of the agent’s output. We furthermore argue that this human researcher needs to be a consensus expert to effectively steer and leverage the capabilities of agents.

Recent advances in the capabilities of large language models (LLMs) have indeed enabled coding agents to autonomously perform increasingly complex software engineering tasks [35, 80]. For example, CODEX by OpenAI [57], CLAUDE CODE by Anthropic [4], and CURSOR AGENT by Anysphere [5] can navigate repositories, modify code, execute tests, and iteratively refine implementations based on execution feedback. However, these general-purpose capabilities do not necessarily translate into the autonomous discovery of consensus protocols. Existing AI-driven systems research frameworks optimize self-contained policies such as schedulers and load balancers in simulators [14, 54], but do not seem an adequate fit for consensus, whose logic is spread across many communicating, failure-prone nodes [14]. Our preliminary experiment in § 2.3 also suggests that a coding agent combined with a frontier LLM, starting from a single existing consensus codebase, finds only marginal improvements. This gap motivates the need for specific support for consensus research beyond general-purpose coding capabilities.

We present AI-GREEMENT, an agent harness that equips general-purpose coding agents with domain-specific support for autonomous consensus research. AI-GREEMENT addresses three key challenges in consensus discovery. First, synthesizing new protocols requires understanding and combining mechanisms scattered across the consensus literature. To support this, a concept-oriented protocol knowledge base organizes consensus mechanisms and grounds them in concrete examples from published protocols. Second, translating novel designs into executable implementations is challenging because consensus mechanisms often have complex protocol states and message flows. A modular implementation skeleton structures this scheme through well-defined components and interfaces, allowing the agent to focus on protocol-specific logic. Third, subtle correctness violations can escape conventional testing, which covers only a limited set of message orderings and failure scenarios. Formal verification closes this gap by proving safety for every execution of a protocol model, but writing such models and proofs from scratch for each candidate is too costly. A LEAN-based verification framework therefore provides reusable safety definitions, reference specifications, and proof libraries, so the agent can obtain machinechecked evidence that its design, as modeled, is safe. These components work together in an iterative discovery process, where performance measurements and fault-injection results guide the agent toward more efficient and robust designs.

We demonstrate the capabilities of AI-GREEMENT through A-GREEMENT, a new consensus protocol that significantly outperforms state-of-the-art solutions. To the best of our knowledge, A-GREEMENT is thefirst consensus protocol that is designed and implemented by an LLM agent. When replicat ing a key-value store, A-GREEMENT jointly optimizes read and write latency. A-GREEMENT builds on the quorum-lease local reads of Paxos Quorum Leases (PQL) [50], generalizes BODEGA’s leader-side gating of writes [29] to any replica, and adds a leaderless write path in the style of SWIFTPAXOS [63]. Its design couples write completion with read availability: for frequently accessed keys, a designated completion owner can continue serving reads while a write is pending; for ordinary keys, clients learn leader-ordered writes directly from acceptor votes; and for infrequently accessed keys, writes can bypass leader ordering through a fixed-majority per-key path. These paths share rules ensuring that local reads do not miss completed writes. A planner uses network delays and workload statistics to select replica roles and write paths, accounting for both write latency and the time reads spend waiting behind pending writes.

Averaged across six deployments with 5 or 9 replicas, on write-intensive workloads (50% writes) A-GREEMENT reduces mean latency by 12.2% relative to SWIFTPAXOS, the best alternative, at the cost of a 13.8% higher p99 latency. On read-dominated workloads (5% writes), A-GREEMENT reduces mean and p99 latency by 33.3% and 30.1%, respectively, relative to BODEGA, the alternative with the lowest mean latency, with a p99 latency within 3.4% of the best alternative on that metric, SWIFTPAXOS. In the common case, A-GREEMENT’s reads complete at a single nearby replica, and writes to rarely accessed keys complete in one round trip. AI-GREEMENT completed the discovery process (including the verification) of A-GREEMENT in around 24 hours, consuming 778.45 million tokens at an estimated API-equivalent cost of \$198.69.

To summarize, we make the following contributions:

• We design AI-GREEMENT, an agent harness designed for the automated discovery of consensus protocols (§ 3). AI-GREEMENT integrates a consensus knowledge base, a modular consensus skeleton, and LEAN-based formalization and verification.

• Using AI-GREEMENT, we synthesize A-GREEMENT, a new consensus protocol “autonomously” designed and implemented by AI-GREEMENT. A-GREEMENT combines localized reads, complementary write paths, as well as topology- and workload-aware planning (§ 4,5,6).

• We perform an extensive experimental evaluation of A-GREEMENT across replica counts, workloads, and network topologies. Averaged across six deployments with 5 or 9 replicas, A-GREEMENT reduces mean latency relative to the best alternative by 12.2% and 33.3% on write-heavy and read-heavy workloads, respectively, while matching the best-performing baseline on the readonly workload (§ 7).

• Using AI-GREEMENT, we describe four previously unreported safety vulnerabilities in existing consensus protocols, found through agent-driven LEAN formalization and verification, demonstrating AI-GREEMENT’s ability to uncover subtle protocol flaws (§ 8).

• We reflect on our results and ask whether this marks the end of human-driven consensus research, based on where our agent succeeded, where it needed human steering, and what remains for researchers to do (§ 10). Our findings imply that a human consensus expert is (still) badly needed. Many of these findings should apply to other (distributed) computing problems, especially when the goal is to discover, formally prove and practically evaluate a solution.

## 2 Background & Motivation

We first provide the background needed to motivate AI-GREEMENT by explaining consensus and state machine replication (§ 2.1). We then describe how LLM agents and their harnesses can solve complex problems (§ 2.2). Finally, we show that a frontier general-purpose coding agent, given an existing consensus codebase and repeated experimental feedback, rarely improves on existing designs, and we identify the shortcomings that directly motivate the design of our AI-GREEMENT (§ 2.3) agent.

## 2.1 Consensus & State Machine Replication

State machine replication (SMR) implements a fault-tolerant service by maintaining copies of its state at multiple replicas [65]. The service is modeled as a deterministic state machine: a command and the current state uniquely determine the next state and the response. Clients submit commands and receive their results through the SMR protocol, which coordinates execution across replicas to present the behavior of a single state machine.

This behavior is formalized by linearizability: the responses observed by all clients must be consistent with a single sequential execution of their commands by the state machine, preserving real-time order across clients [26]. Our verification framework uses linearizability, with the corrected definition of Sela et al. [66], as a uniform proof target for all candidate SMR protocols. Command ordering can exploit commutativity: replicas may choose different execution orders for commuting commands while agreeing on the order of conflicting commands [39, 60]. Two commands commute if exchanging their order from any state leaves both the resulting state and each command’s response unchanged; otherwise, they conflict.

![](images/f9961bd60dc37c5b608cb2323f3f88526f60eb8f0d2a860b3b9af4d1a8743268.jpg)  
Figure 1: The workflow of an LLM agent.

To coordinate command execution, SMR protocols use consensus, in which processes propose values and agree on one of them [38]. For example, consensus can be used to commit commands in an order proposed by a leader, as in MULTI-PAXOS [38]. It can also establish agreement on command dependencies from which replicas determine an execution order, as in EPAXOS [49, 64, 75].

An SMR protocol must also provide liveness: requests from non-faulty clients eventually complete under the protocol’s stated assumptions. Our verification currently targets safety only; we discuss the extension to liveness in § 10.

The performance of SMR protocols is commonly measured using four metrics. (i) Mean latency measures the average time between a client submitting a request and receiving its response, while (ii) tail latency (P99) measures the 99th percentile of request completion times. (iii) Throughput measures the number of requests completed per unit of time. (iv) Recovery time measures how long it takes for request latency or throughput to stabilize after the failure of a leader or replica.

## 2.2 LLM Agents

Large language models (LLMs) are neural networks trained on large text corpora to predict the next token given the preceding context. The underlying transformer architecture was orig inally developed for natural language tasks such as machine translation [77], and early LLMs were primarily used for tasks such as translation, summarization, and question answering [7]. Instruction tuning and reinforcement learning from human feedback turned these models into general-purpose conversational assistants, which have been popularized by CHATGPT [56]. As their capabilities improved, LLMs also became capable at reasoning and more complex tasks such as writing code, extending their use beyond language tasks [78].

On its own, an LLM can only produce text in response to its input prompt. Agentic AI removes this limitation by placing the model in a loop with tools and an environment, as shown in Figure 1 [1]. At each step, the model decides on an action, such as reading a file, running a command, or querying a search engine; the action is executed in the environment, and its result is returned to the model as an observation that informs the next decision. This loop lets an LLM agent pursue multi-step goals, gather information it was not given up front, and correct its own mistakes based on feedback. Coding agents, such as CODEX [57], CLAUDE CODE [4], and CURSOR AGENT [5], apply this paradigm to software development and enable agents to solve complex coding tasks. Given a task, a coding agent iteratively inspects relevant files, edits code, executes programs, and runs tests, and it revises its approach based on the results until the task is complete. LLM agents are also increasingly used for scientific discovery [23]. Our objective is to use agentic AI and LLM agents to synthesize and discover new consensus protocols.

![](images/a95211867b69eb2a57297b37c983d36bfc3aa210fcbbb9dba8c97e9755636d7f.jpg)  
Figure 2: Mean and tail (P99) latency of discovered consensus protocols across 20 search iterations, using the original CODEX harness and the GPT-6.1 Sol model (with high reasoning effort).

An agent harness is the software infrastructure that organizes this interaction between the LLM, its tools, and the environment [53]. At each step, it supplies the model with instructions and relevant context, dispatches the model’s tool calls, and feeds the resulting observations back into subsequent model inputs. It also manages execution state and stopping conditions across steps. The model proposes actions and interprets feedback, while the harness makes those actions executable and maintains the interaction loop. Beyond these general functions, a harness can provide domain-specific knowledge, implementation interfaces, and evaluation tools for a particular class of tasks. The harness has a large effect on agent performance: changing the harness around a fixed LLM can produce a 6 performance gap on the same benchmark [74]. In this work, we build on this observation by providing a harness specialized for consensus research.

## 2.3 Limitations of General-Purpose Agents

We first investigate whether a general-purpose coding agent, given existing consensus implementations and repeated experimental feedback, can discover protocol changes that improve performance. We choose the SWIFTPAXOS [63] codebase as our starting point because it provides a high-performance consensus implementation alongside implementations of sev eral baseline protocols. Using this codebase, we give the CODEX harness powered by the GPT-6.1 Sol model with High reasoning effort the prompt in Appendix B to conduct an iterative 20-round search for consensus protocols with lower mean and tail (P99) latency. § 7 provides additional information about our experimental setup. Unlike AI-GREEMENT, this agent receives no curated protocol knowledge: it may search for the state of the art itself, but the codebase includes neither BODEGA nor PQL, on which A-GREEMENT builds. This experiment is therefore not a controlled comparison with AI-GREEMENT, but an illustration of the difficulties a generalpurpose agent faces.

Figure 2 shows the mean (left) and tail (P99) latency (right) across the 20 search iterations, with the latency of SWIFT-PAXOS indicated by a blue dotted line. For reference, the red line reports the latency of A-GREEMENT, discovered with the domain-specific support of our AI-GREEMENT (§ 4).

We observe an initial latency reduction in the first two rounds, followed by a plateau. The agent begins with a basic leader-based replication protocol in round 0, in which both reads and writes undergo PAXOS-style consensus and execution before clients receive replies. Our manual inspection of the agent trajectories reveal that the early gains mainly come from applying established optimizations: quorum-validated reads, early write completion, and improved leader placement shorten read and write completion paths. Later rounds mainly explore secondary engineering optimizations, including batching, message encoding, and caching, which only result in marginal reductions in latency. Mean latency spikes in round 12 which stems from processing and reply-traffic overhead introduced by periodic snapshot refreshes. Moreover, we noticed that candidate implementations between rounds 3–14 violate availability despite a surviving majority, due to defects in client registration, recovery transfers, or peer connection handling. At an offered load of 20000 commands per second, the best candidate achieves 179.87 ms mean latency and 289.77 ms average P99, averaged across deployments with 5, 9, and 13 replicas. These results are only marginally better than the corresponding mean and P99 latencies of 196.16 ms and 297.62 ms, respectively, for SWIFTPAXOS.

After careful inspection of the traces, we identify three potential problems that limit the ability by the general-purpose agent to discover consensus algorithm with more substantial gains. The first issue is that the agent relies mainly on conventional optimizations and secondary engineering refinements and appears not to fully understand the principles behind techniques in existing consensus work. This motivates explicit consensus-design knowledge that explains these techniques rationale and tradeoffs. Second, we observed that the agent re-implements the entire protocol each search iteration. Implementing consensus, however, requires substantial infrastructure for communication, client handling, and recovery, making implementation from scratch complex and error-prone. The observed defects, including loss of availability despite a surviving majority, motivate a reusable and modular implementation foundation that reduces this burden and lets the agent focus on protocol-specific logic. Third, experimental feedback measures performance and can expose some correctness violations, but cannot by itself establish protocol safety beyond the tested executions. This motivates explicit invariants and formal proofs that guarantee safety for all executions permitted by the fault model.

We believe these limitations stem in part from a gap between the general-purpose support provided by the agent’s harness and the domain-specific requirements of consensus discovery. In the following section, we introduce an additional harness layer that builds on the existing software engineering capabilities of state-of-the-art coding agents and augments them with three domain-specific modules for consensus re search.

We also conduct a trial with SkyDiscover-Synthesize (SKYSYNTH [43]) and use its test-driven agentic workflow to search for SMR protocols with lower mean and tail latency. Despite a requested budget of 20 search rounds, the run remained focused on implementing and repairing its initial candidate. We terminated the experiment after approximately 11.5 hours, and 486 million input and output tokens across the main agent, workers, and automatic reviews. Although the workflow produced correctness tests and diagnostic traces, unresolved client-completion failures prevented a complete latency evaluation, and no performance improvement was demonstrated. This trial further illustrates the difficulty of reaching reliable end-to-end evaluation in the synthesis of consensus protocols, even with a structured agent workflow.

## 3 Design of the AI-GREEMENT Harness

We design AI-GREEMENT, an agentic harness to support the iterative discovery process of consensus algorithms (Figure 3). AI-GREEMENT equips general-purpose coding agents with the knowledge, implementation structure, and verification support needed for consensus research. Given a failure model, performance objective, and search budget, the agent takes researcher, developer, and verifier roles in an iterative process. We next explain these components and roles in detail.

## 3.1 Protocol Knowledge Base

General-purpose agents can search the web and retrieve relevant papers. However, access to the literature alone does not provide a systematic way to design new consensus protocols. Relevant design ideas are scattered across many papers that differ in their assumptions, terminology, and protocol structures. Understanding how a mechanism used by one consensus protocol can be applied to another requires reasoning about its underlying principles, safety conditions, and interac tions with other mechanisms. To address this, we construct a protocol knowledge base (the left column in Figure 3) that organizes existing consensus research around reusable designs in Markdown format, instead of supplying individual papers. This gives the LLM agent a structured way to understand and reason about the concepts relevant to consensus research. The knowledge base is constructed through human-guided, AI-assisted organization and expert review.

Knowledge base construction. As a first step, we identified 22 representative papers on foundational and state-of-the-art consensus protocols [3,9,13,15,20,27–29,36,39,40,48,49,51, 52,58,59,63,64,72,75,76]. We then define a shared extraction structure covering system assumptions, roles, messages, local state, normal operation, completion conditions, recovery rules, and safety arguments. Afterwards, we let AI use this structure to produce notes from the published papers. We manually verify the produced notes against the source papers, correct errors, and add missing explanations.

Concept-oriented organization. The produced notes streamline cross-protocol comparison for the agent. The concept index connects 24 consensus concepts, such as quorums, conflicts, and dependencies, to concrete protocol examples. The associated mechanism cards summarize each example’s message flow, assumptions, recovery behavior, and safety conditions. The shared structure also enables comparisons across 11 design dimensions, including quorum systems, completion rules, recovery mechanisms, and latency. By organizing mechanisms under common concepts, the knowledge base allows the agent to more easily compare how different protocols ad dress similar design challenges without being constrained by the organization and presentation of individual papers.

Knowledge-guided discovery. In the researcher role, AI-GREEMENT uses the concept index during protocol discovery to design a candidate protocol and to identify mechanisms that could cause a performance bottleneck. It consults the mechanism cards and cross-protocol comparisons to assess their expected benefits, costs, and compatibility conditions. For example, changing a completion quorum requires examining what evidence remains available during recovery. These checks guide the compare and compose activity, where the agent selects and combines candidate mechanisms. The resulting transition-level design records the candidate’s states, transitions, quorum rules, and recovery behavior.

![](images/5862317c87863bdd5b56645f3f1d0e6cb3d750c1b97d9de62c5164fe045ea159.jpg)  
Figure 3: AI-GREEMENT harness architecture. Blue components provide reusable knowledge, interfaces, execution semantics, and proof support; orange components contain protocol-specific logic supplied by the agent. In the verification column, protocol specifications and proofs build on a shared execution model and linearizability definition.

## 3.2 Modular Consensus Skeleton

Many consensus protocols have open-source implementations. However, their engineering choices differ in code organization, communication primitives, and runtime infrastructure. Exploring mechanisms across these codebases therefore requires the agent to understand and adapt the surrounding implementation as well as the protocol itself. We address this with CONSENSUSARENA, a framework written in the Go language in which seven baseline protocols share a common runtime, client framework, and benchmark harness. We let an AI agent integrate baseline protocols, including classical PAXOS [38], FASTPAXOS [40], EPAXOS [49, 64, 75], SWIFT-PAXOS [63], BODEGA [29], CURP [59], and KCENSUS [9].

On top of this framework, we provide a modular consensus skeleton for implementing new candidates (Figure 3, middle). The AI-GREEMENT (in the developer role) supplies the orange components (deployment planning, the client policy, and the protocol core) while the blue components are shared by all protocols and created by us. For each candidate, a generator instantiates an independent package with one file per component. Every agent-supplied component returns an explicit error until the agent replaces it, so a partially implemented candidate cannot produce measurements by silently inheriting a default leader, quorum, commit rule, or routing policy. Baselines are read-only to the agent, and a finished candidate is registered through a single entry in a log, after which it runs on the same runtime and benchmark as the baselines. Next, we further describe the components in Figure 3 (middle).

Deployment planning. Before startup, the planner maps the network topology and workload to an initial leader, quorums, read responders, and one ingress replica per client. All such policy lives in the candidate’s planner rather than in launch scripts, which lets AI-GREEMENT’s workload-aware planner (§ 4.3) run without changes to the framework.

Client. Protocols differ in how clients interact with replicas: a MULTI-PAXOS client sends each request to the leader, whereas a Fast Paxos client sends it to all replicas. The client component specifies this behavior: which replicas receive a request, which replies complete it, and whether pending requests are retried after a failure.

Serialized event loop and adapters. Each replica processes all events on a single loop, so the protocol core needs no locks. The loop drives the core through three adapters: Propose for client requests, Handle for peer messages and local events, and Tick for timers.

Protocol core. AI-GREEMENT defines the core’s state, messages, and handlers, including the quorum checks that determine acceptance and commitment. Recovery and execution mechanisms are also part of the core: recovery collects evidence and selects safe values or orders, and execution orders, deduplicates, and applies commands to a shared key-value store. The skeleton defines accepted, committed, executed, and client-completed as distinct states for a command.

Output dispatch and network I/O. Each transition returns outgoing messages and client replies, where a reply signals that the protocol has authorized completion. The dispatcher freezes each message and hands it to a per-connection sender with a bounded, non-blocking queue; received messages return through the event loop.

Benchmark and fault tests. In the developer role, candidate protocols are evaluated with the CONSENSUSARENA benchmark, which emulates geo-replication on a SLURM cluster using TOXIPROXY to emulate network delays and inter-region delays from 35 AWS cloud regions. The development suite covers 5, 9, and 13 replicas in three topologies under the YCSB workloads. Candidates that meet the performance target are then crash-tested: replicas are killed until $f = ( n - 1 ) / 2$ have failed, and throughput must recover to 90% of its pre-crash level and remain there. Note that our final evaluation in § 7 uses more advanced topologies and workloads disjoint from this suite to show the generalization.

## 3.3 Lean Formalization & Verification

Testing can expose safety violations, but even extensive testing cannot establish safety under all possible event orderings. Informal correctness arguments must account for interactions among concurrent actions, delayed messages, and failures, making subtle omissions difficult to detect. For example, AI-GREEMENT found that the dependency-pruning rules of EPaxos Revisited [75] allow replicas to execute conflicting commands in opposite orders, also see § 8. Proof assistants such as LEAN [18] and ROCQ [73] close this gap by mechanically checking that specified safety properties hold for every execution admitted by a formal model. However, developing such models and proofs requires substantial human effort, often spanning several person-months [61].

We therefore aim to enable AI-GREEMENT to automate protocol formalization and proof construction with LEAN within the discovery loop, using safety properties predefined by experts (Figure 3, right). The process proceeds in two stages. First, the agent encodes the candidate protocol as a formal specification in LEAN. Second, it proves that this specification satisfies the predefined linearizability property § 2, using LEAN’s feedback on remaining proof obligations and errors to refine its proof iteratively. We integrate the reference LEAN specifications and proofs or disproofs of linearizability for MULTI-PAXOS [38], FASTPAXOS [40], EPAXOS\* [64] (the corrected version of EPAXOS [49, 75]), SWIFTPAXOS [63], and BODEGA [29].

The framework lets the agent specify a protocol much as it would in pseudocode, by defining a node’s local state and its handlers for messages and timeouts. A shared transition model uses these handlers to describe how node and network state change as messages are delivered and timeouts occur. Client invocations and responses form the observable history of each execution and provide the basis for the framework’s linearizability specification. We then establish protocol safety by proving that every execution starting from the initial state and following these transition rules produces a linearizable history. Although researchers must still manually check that the specification faithfully captures the intended protocol and its assumptions, the shared foundation reduces this review burden: once the common definitions have been checked, researchers can focus on the protocol-specific state, initialization, and transition rules.

For proving, the agent develops protocol-specific invariants and uses reference proofs and reusable lemmas to establish the predefined safety properties. LEAN provides feedback on unsuccessful proof attempts, and its kernel checks the completed proof against the stated theorem. An unresolved proof obligation may also prompt AI-GREEMENT to investigate whether the required property can be violated, although proof failure alone does not always result in a bug. If this investigation yields a counterexample, the agent checks whether it arises from an inaccurate encoding or a flaw in the intended protocol, with human review where needed. A modeling error requires correcting the specification; a protocol flaw calls for revising the protocol and updating its specification before attempting the proof again. When AI-GREEMENT establishes a protocol vulnerability, it records the counterexample, the violated property, and any resulting repair, preserving the finding as feedback for subsequent discovery iterations.

We found that reference specifications and proofs are valuable beyond the protocols they verify: they provide modeling patterns and proof strategies that the agent can reuse. Expand ing this library helped the agent build more faithful models and complete proofs faster. More implementation detail, however, is not always helpful. The agent sometimes carries incidental engineering choices into the formal model, adding proof obligations without improving its fidelity to the protocol. We therefore ground formalization in the paper’s pseudocode and use open-source implementations only to resolve details that the paper leaves unspecified.

## 3.4 AI-GREEMENT for Consensus Discovery

We implement AI-GREEMENT on top of the CODEX coding harness. The search begins with user-defined objectives and a search budget specified as the number of discovery rounds. The user may target a single metric, such as latency or maximum achievable throughput, or specify tradeoffs among multiple metrics, for example, requiring improvements in both mean and tail latency.

In each discovery round, the agent first acts as a researcher, selecting a parent protocol and forming a hypothesis about the bottleneck limiting the target metrics. It consults the consensus-specific knowledge base (§ 3.1) to compare candi date mechanisms, assess their safety risks, and estimate their expected benefits and additional overhead. The resulting design document specifies messages, local state, quorum and completion rules, execution, fallback, and recovery.

To test each candidate design, the agent then acts as a developer, implementing the candidate design within the skeleton (§ 3.2) and comparing it with other candidates under the same benchmark. Instrumentation helps relate the results to the original hypothesis, for example by separating latency spent collecting votes, waiting for execution, and resolving conflicts. Candidates that meet the user-defined performance criteria proceed to fault-injection testing. Fault injection separately tests recovery after replica failures. Candidates that meet the performance criteria and pass the experimental checks proceed to formal verification. The agent then acts as a verifier, constructing a LEAN model and attempting to prove the safety target (§ 3.3).

The staged process in AI-GREEMENT rejects unpromising candidates before committing effort to their implementation, experimental evaluation, and formal verification. Feedback from each stage guides subsequent revision or rejection, concentrating verification effort on promising candidates. To conserve resources, our current implementation defers formal verification until the end of the search and proves only the final version.

After every discovery round, the agent records the candi date’s parent, mechanism changes, expected and observed bottlenecks, measurements, recovery results, and proof status. It summarizes both successful and unsuccessful attempts into reusable lessons about which mechanisms improved perfor mance, which failed to deliver the expected gains, and which correctness questions remain unresolved. These accumulated lessons guide the agent’s choice of subsequent search strategy, including whether to refine a candidate, revise a bottleneck hypothesis, revisit an earlier parent, or explore a different mechanism family. For example, when successive changes leave the same bottleneck, the agent reassesses the remaining improvement potential and decides whether to change direction. By carrying these lessons forward, the agent builds on successful mechanisms and avoids repeating unsuccessful approaches without new evidence.

## 4 A-GREEMENT Protocol Design

Our agent synthesized A-GREEMENT, a consensus algorithm that jointly reduces read and write latency. The design starts from a simple observation: the best way to complete a write depends on how that choice affects subsequent reads. A frequently accessed key benefits from a nearby replica that can serve reads and control write completion. A less frequently accessed key may benefit more from sending acceptance evidence directly to the client. For sufficiently rarely accessed keys, writes can also avoid leader ordering altogether. A-GREEMENT therefore combines three write paths with local reads, which allow a replica holding a read lease to answer reads without contacting other replicas.

Because AI-GREEMENT draws on prior consensus designs through its knowledge base (§ 3.1), many of A-GREEMENT’s mechanisms are borrowed from existing algorithms; § 4.1 points out where each one comes from. At its core, A-GREEMENT is closest to Paxos Quorum Leases (PQL) [50] and BODEGA [29]: a MULTI-PAXOS log in which a stable leader orders writes, combined with lease-based local reads at selected replicas. A write completes only once a majority and every read responder for its key have accepted it, so a responder holding a lease can answer reads from its local state. A-GREEMENT makes two changes to this core. First, writes need not complete at the leader: depending on the key, they can complete at a responder near the readers or directly at the client. Second, writes to rarely accessed keys bypass the log entirely and follow a fast path in the style of SWIFTPAXOS.

Before deployment, A-GREEMENT’s planner takes as input a workload profile: a sample of key accesses, obtained from historical traces or a profiling phase before the protocol starts serving requests, together with measured network delays between clients and replicas and the expected write fraction. From this profile, the planner selects the write path for each key and places the replica roles. The resulting plan is fixed during execution. A profile that does not match the actual workload can reduce performance but never affects safety: keys absent from the profile are ordered by the shared log, and every path enforces the same completion and read-visibility conditions. This section explains the high-level design of A-GREEMENT; Appendix A contains the detailed description, and Table 1 compares its latency and quorum requirements with prior protocols.

![](images/016ca2764c7eca5872072f3d7d69d94b3e96707cb0631c6da6245c83138e6569.jpg)

![](images/2b71af00583d5ac175417c13e19c2fed765a7832bb0d27ca1cd776a59514ea6b.jpg)

![](images/3c5366658a5a6033b6353f07c3cba08e6436783df0cf5c4947d1a72a4b4cb6f5.jpg)  
C client leader read responder completion owner cold quorum member  
Figure 4: Message propagation in A-GREEMENT on the hot (left), ordinary (middle) and cold write (right) paths.

## 4.1 A-GREEMENT in Simple Terms

A-GREEMENT divides the key space into three: frequently accessed (hot) keys, regularly accessed (ordinary) keys and infrequent (cold) keys. Before execution, the planner chooses this partition from the workload profile and assigns the associated replica roles (leader, responder, quorum, etc.) using measured network delays. All three classes share one read mechanism; they differ in how writes complete and how the replicas serving reads learn about them. Figure 4 shows the write path of each class. Before describing these paths, we explain how replicas serve local reads, since each write path must ensure that these reads do not miss completed writes. For each mechanism, we give the main idea, how it works, and where it comes from.

Common ground: a leader-ordered log and local reads. Hot and ordinary keys retain a MULTI-PAXOS-style shared log ordered by a stable leader [38]. For all three classes, the planner chooses read responders: replicas that answer reads without contacting other replicas. These local reads follow PQL [50]: replicas grant each other read leases covering all keys at once, a responder may serve local reads while it holds leases from a majority, and read responders are chosen from access statistics. As in BODEGA’s roster leases, A-GREEMENT binds the lease assignment, that is, the leader and the read responders, to the ballot [29]. A lease establishes who may answer a read, not which value is safe to return: a responder must also know that its local value includes every write that could have completed before the read began. Following PQL, a write therefore completes only once a majority and every read responder for its key have accepted it, and a responder that has accepted a write holds reads of that key until it can safely answer. A local read thus costs a single round trip to one replica, but every responder must learn about each write to the keys it serves. The three write paths differ in how they achieve this. To our knowledge, prior protocols that vary the write path by key access pattern [21] do not support local reads; A-GREEMENT instead chooses each class’s write path for how it keeps read responders up to date.

Hot keys: control completion near the reader. Hot keys receive the most reads, so their write path aims to keep responders serving reads while writes are pending. As in BODEGA, the leader orders writes in the shared log and broadcasts ACCEPT to the acceptors. In BODEGA, acceptors send AC-CEPTED only to the leader, so no other replica can learn that a write committed before the leader does. The leader therefore implicitly gates every write and can keep serving the committed value while writes are pending [12], whereas the other responders must hold reads of a key once they have accepted a write to it. A-GREEMENT generalizes this gate to a completion owner: a responder the planner selects for a range of hot keys, typically close to their readers. A client sends each write for these keys directly to the leader and names the completion owner as the replica that will reply. Acceptors send ACCEPTED to the leader, the other responders and the completion owner. Since the completion owner is itself a responder, no write to its keys can complete without its ACCEPTED. It therefore withholds that message and keeps serving the old value until it has ACCEPTED from a majority and every other responder. Only then does it switch to the new value and send its own ACCEPTED; it replies to the client once it has executed the write. Because votes reach several learners rather than only the leader, the gate is no longer implied by the message pattern; withholding the vote enforces it explicitly. The planner assigns completion owners only where it estimates a latency gain. Other hot keys keep BODEGA’s arrangement: the leader is their implicit completion owner, and gates the writes it receives itself; writes that clients send through another replica are not gated, and that replica replies once it has executed them.

Ordinary keys: deliver the evidence directly. Ordinary keys receive fewer reads, so holding back a write to keep one responder serving the old value gains little; their write path instead minimizes write latency. The leader orders writes in the same log, but acceptors send their ACCEPTED votes directly to the client, which completes the write once it holds matching votes from a majority that includes every read responder. This moves PQL’s learner into the client: in $\mathrm { P Q L }$ , acceptors also send their votes to the replica near the client, which commits and replies [11, 50]. There is no completion-owner shortcut on this path: once a responder accepts a write, it holds later reads of the key until it has itself gathered acceptances from a majority and every responder.

Cold keys: avoid the shared log. Writes to infrequently accessed keys rarely conflict, so they can skip leader ordering altogether. The client sends such a write directly to a fixed majority of replicas, the cold quorum. Instead of a shared log, each cold key has its own version sequence. Each member votes for the next version of the key and notifies the client directly; without conflicts, the write completes once the client holds votes from every member. This uncontended case follows the fast path of SWIFTPAXOS, but omits dependency ordering and the second-vote fallback [63]. Because every member of the cold quorum votes on every write, all of them learn about it, so they also serve as the key’s read responders without adding a participant to the fast path. A member holds reads of the key from its vote until it learns which value was chosen. Upon a conflict, the leader checks whether either write could already have completed and, if so, keeps it first. It then proposes the next write in a new per-key round, which every cold quorum member must accept before completion. If higher-ballot recovery is needed while the cold path is active, the protocol recovers its potentially completed writes and transfers their state into the shared log. This transfer is permanent; subsequent operations use the log.

## 4.2 Model & Notation

A-GREEMENT replicates a key–value store across a fixed set R of $n = 2 f + 1$ replicas; $m = f + 1$ denotes a majority. Replicas may crash but do not behave arbitrarily or restart. Surviving replicas retain accepted commands and duplicatedetection state in memory. The protocol tolerates delayed or repeated messages. Progress requires partial synchrony: eventually, message delays among available replicas stay below the failure-detection and lease timeouts, and a majority remains available; a fast path may additionally depend on particular replicas.

Clients issue read and write operations with unique request identifiers. A write returns an acknowledgment. Operations on distinct keys commute. The target consistency guarantee is linearizability.

The ordinary log operates in ballots. Ballot b identifies a leader $L _ { b }$ and the read responders $S _ { b } ( k )$ for each key $k ;$ we call this assignment the roster. The planner chooses the roster of the initial ballot. A higher ballot may change the leader and removes failed role holders (§ 4.5), but otherwise keeps the planned roles. A responder needs a valid lease and sufficiently up-to-date state before answering locally. Leases assume bounded clock-rate drift, but not synchronized clocks. When the cold path is enabled, $Q _ { b }$ denotes its fixed majority quorum and its set of read responders.

We distinguish acceptance, when a replica records a proposal and votes for it, from execution, when it applies the chosen command to its state. A certificate is a set of matching votes sufficient to establish that a write has been chosen.

## 4.3 Preparation and Role Placement

The planner runs once, before deployment, on the workload profile introduced above; it does not run during execution. It first divides keys into hot, ordinary, and cold classes. A key is hot if the profile records more accesses to it than to the average observed key. Among the remaining observed keys, the least frequently accessed ones may be assigned to the cold path. The planner considers up to seven candidate cold cutoffs: the hot boundary itself, and the boundaries that split the remaining keys’ cumulative accesses into seven equal parts. All other keys use the shared log. By default, observed keys among them take the ordinary path, while unobserved keys, and hot keys outside every completion-owner range, have the leader as their implicit completion owner; the planner may instead enable direct client certificates for all log-ordered keys, which removes completion owners altogether. Using the sampled read and write counts, measured network delays, and configured workload rate and write fraction, the planner chooses this cold cutoff together with the leader, read respon ders, completion owners, direct client completion policy, and cold quorum. These choices remain fixed during execution; § A.7 gives the exact rules.

The planner balances read and write latency: placing a responder closer to readers can make it more costly for writes to reach that responder. The cold path avoids leader ordering but requires every member of its quorum and incurs recovery work when writes conflict. The planner can therefore leave some non-hot keys on the ordinary log or disable the cold path entirely. Its cost model and search procedure are specified in $\ S \operatorname { A . 7 } ;$ they guide placement rather than guarantee a latency improvement for every workload.

## 4.4 Normal Operation

This subsection states the precise conditions behind the paths of $\ S 4 . 1$ . Before serving requests, replicas install the planned roles under a common ballot: the leader, the read responders of each key range, the completion owners, and, when enabled, the cold quorum. Each client also receives from the plan its read responder and its write ingress, the replica that replies to its writes; for a key in a completion-owner range, the ingress is that owner. The client sends each write both to its ingress and directly to the leader, which proposes whichever copy arrives first. Read responders obtain leases from a majority and catch up before answering locally, and the cold quorum installs the initial per-key state before accepting proposals.

Leader-ordered writes. The leader assigns each proposed entry a log slot, and replicas retain the complete entry before voting. For an entry E, let $V ( E )$ be the set of matching voters and $S _ { b } ( E )$ the union of read responders for the keys modified by E. Every completion of a leader-ordered write, whether at a completion owner, at the client, or at the ingress, requires

$$
| V ( E ) | \geq m \quad \land \quad S _ { b } ( E ) \subseteq V ( E ) .\tag{1}
$$

The majority fixes the log value; the coverage ensures that every authorized reader has recorded the write before it completes. Replicas execute chosen entries in log order and use request identifiers to avoid applying retries twice.

Completion gates. A completion owner gates exactly the writes for which it is the ingress: it withholds its vote until it has observed (1) and advanced its read target to the write, and serves the preceding value meanwhile. The leader plays the same role for the writes it receives itself on log-ordered keys that use neither a completion owner nor direct certificates. Batches are split so that each log entry has at most one gate. If a write that the completion owner does not gate is pending on the key, it holds reads like any other responder. Without a gate, a responder holds reads of a key from the moment it accepts a write to it until it learns that the write is chosen.

Reading ahead of execution. A responder can sometimes return a chosen value while execution is still waiting on earlier log entries. It needs matching votes from a majority and every required read responder for the write, together with evidence that a majority has accepted a contiguous log sequence through its entry. The responder can then check whether the write is a new operation or a retry of an earlier request, and return the key’s latest chosen value without waiting for unrelated earlier commands to execute. When the shortcut is unavailable, the read waits. The full eligibility and visibility conditions appear in $\ S \ A . 2 .$

Cold writes. A member of $Q _ { b }$ votes for version v of a key only if it has applied every version up to v  1 and has not voted for another command at v in the current key round. Matching votes from every member complete the write; the client then distributes the certificate, and replicas apply certified writes in per-key version order. Consequently, a write that reaches a member before the certificate of the key’s preceding write collides there, even if the two writes did not overlap from the clients’ point of view. Such collisions, like concurrent proposals, are resolved by the leader in a higher key round that preserves any write that might already have completed; the fast path resumes once every member of $Q _ { b }$ has installed the resolved state.

## 4.5 Failure Recovery

Failure of the leader or a required read responder triggers a higher ballot. The leader for that ballot revokes conflicting read leases or waits for them to expire. It collects accepted log entries from a majority, which promise to reject proposals from earlier ballots, and preserves any entry that could already have been chosen. Replacement read responders obtain leases and catch up before answering reads. Other replica failures need no role change if a majority and all required role holders remain available.

If the cold path is active, higher-ballot recovery also freezes its voting and collects acceptance records from a majority. A write might have completed before its client distributed the certificate, so a missing certificate does not establish that the write was unchosen. Recovery preserves every write that might have completed and binds the recovered cold state to a bridge entry in the ordinary log. Executing this entry installs that state and imports its request identifiers for duplicate detection. The handoff is permanent: subsequent operations use the ordinary log, and write completion and read visibility require execution of the relevant log prefix. A same-key collision instead uses a leader-coordinated round to resolve the next per-key version, after which the cold quorum can resume the fast path.

Because replicas store the full command before voting, recovery can recover the write even if its client has failed. Retried requests retain the same identifier so that they are not applied twice. The complete recovery and bridge-installation procedures appear in § A.3.

## 5 Engineering Optimization in A-GREEMENT

Turning a consensus design into an efficient implementation requires engineering optimization beyond pseudocode [12]. When acting as a developer to implement each finalized candi date protocol, we allow AI-GREEMENT to perform up to five rounds of performance tuning. These rounds use experimental feedback to refine implementation choices while preserving the protocol idea. The following engineering optimizations reduce communication and waiting in A-GREEMENT and limit repeated execution and recovery work.

## 5.1 Completion-Aware Batching

The completion owner’s read shortcut imposes a constraint on batching: each log entry may have at most one owner withholding its vote. If an entry depended on several such owners, each could wait for a vote that another owner was withholding. The leader therefore splits a pending batch when it encounters an eligible write with a different completion owner, preserving command order across the resulting entries.

Only writes eligible for the owner shortcut determine these split points. Writes eligible for direct client certificates never acquire a gate of their own, even when an owner is assigned to their key. They can, however, be batched into the same log entry as a gated write, and then wait for its gate (§ 10). Replicas also check received entries and disable the shortcut if completion owners are incompatible. This permits batching among compatible writes while retaining the owner’s ability to serve the preceding value during a pending write. Changes in eligible completion owners can reduce batch size.

## 5.2 Selective Vote Dissemination

The leader broadcasts ordinary proposals, while acceptors direct their votes to replicas that collect completion or readvisibility evidence. For fresh current-ballot entries, the recipient set includes the leader, planned learners, all read responders in the installed roster, and the ingresses of batched requests. Planned learners include write ingresses and completion owners; previously observed request ingresses are also retained as recipients. These sites collect evidence directly while votes to other replicas can be omitted.

For eligible ordinary writes, acceptors also send votes directly to the client. The majority and responder-coverage requirements remain unchanged. Vote traffic decreases when the recipient set excludes some replicas; if the selected roles cover the entire membership, no votes are omitted. Recovered entries and configurations without a selective learner set retain dissemination to all replicas.

## 5.3 Read Visibility Indexes

A read’s target depends on both the state of a write and who controls its completion. A chosen current-ballot write may be visible before prefix execution, whereas an accepted write that could complete elsewhere must already prevent stale reads. Each replica therefore maintains per-key indexes for the latest accepted write, the latest committed write, and the latest write whose completion is outside the owner’s control. The third index identifies writes that prevent the owner from continuing to serve the preceding value. Together, these indexes support the owner’s read shortcut while fencing writes that may have completed elsewhere.

The indexes locate a read’s target log position without searching the log. Blocked reads are grouped by that position in two ordered queues: one for prefix execution and another for earlier visibility once the acceptance-prefix evidence permits it. As each prefix advances, the replica revisits the corresponding groups instead of scanning all waiting readers. This limits repeated work as blocked reads accumulate while preserving both ways of making a read ready. Every release still checks the lease, responder assignment, and chosen-value visibility conditions in § 4.4.

## 5.4 Planned Read Racing

A nearby responder may return later than a more distant one if a concurrent write holds its read. For selected hot keys, the planner therefore assigns a second responder using network delays and modeled read waiting. The client sends the same request to both responders without waiting for a timeout, after checking that the second target is available and authorized for the key. The assignment applies only while the initial ballot and leader remain unchanged.

Each responder independently checks its lease and readvisibility conditions. The client returns the first valid result and sends a best-effort cancellation to release waiting state on the other path. This trades an additional read request for an opportunity to avoid waiting behind a write at one responder. The planner selects this extra path only for ranges where its model predicts a sufficient gain.

## 5.5 Direct Submission to the Leader

Placing write completion near a reader may select an ingress different from the leader. Sending the command only through that ingress would make ordering wait for an additional forwarding hop. When the selected ingress differs from an available leader, the client sends an ordinary write directly to the leader as well as to the ingress. The direct copy identifies the selected ingress, allowing the leader to start ordering while execution replies retain their planned return path.

Both copies retain the original request identifier, so a later forwarded copy is recognized as the same operation. The leader can thus begin ordering without waiting for ingress forwarding, while the ingress keeps its completion role. The additional client-to-leader transmission trades bandwidth for earlier ordering; its net benefit depends on the network routes.

## 5.6 Recovering Per-Key Write Path

The per-key write path can complete a cold-key write before its chosen proof reaches every replica. Recovery must reconstruct these writes from retained records, but full snapshots would repeatedly transfer command contents held by several participants. The implementation therefore separates the evidence needed to select the recovered state from the command data needed to install it.

Each replica freezes a metadata report containing per-key versions, request identifiers, command digests, and acceptance evidence, while retaining full commands locally. The coordinator collects complete reports from a majority and selects the state to preserve according to § 4.5. The reports also identify potential command holders, allowing selection without attaching full command contents to every report.

The coordinator distributes a manifest listing recovered versions and request identities, with digest references to the required commands. Recipients check these references against their local command stores and fetch only missing contents from replicas that retain them. Each recipient tracks missing commands and votes only after retaining and validating all referenced data. This reduces repeated payload transfer when reports refer to the same command or recipients already cache it; cache misses still require data fetching before installation.

Bridge installation reuses the command store and missingcommand tracking, pausing until the referenced contents are complete. It also imports recovered request identifiers into ordinary duplicate detection, preserving operation identities as pending requests move to the log. Command transfer continues over ordinary peer streams after the per-key write path is disabled, so installed replicas can help others finish recovery without reactivating that path.

## 6 Human-Enforced Changes in A-GREEMENT

AI-GREEMENT did not create the final version of A-GREEMENT fully autonomously, however. Before accepting A-GREEMENT’s great performance in our measured benchmarks, we reviewed whether its improvements reflected generalizable design choices or hard-coded benchmarkhacking. It is known that agents may use unexpected methods or cheat to reach their goals [31, 69]. As our discovery process uses a quantitative benchmark to assess performance, the agent might use questionable methods to optimize for latency, resulting in a non-generalizable algorithm with shortcuts. We observed multiple cases where AI-GREEMENT employed such shortcuts, especially when the agent was aware only of the evaluation it could invoke itself.

We mitigated this by introducing a separate test set and telling the agent about it, while keeping its details hidden. While this discouraged hard-coded execution paths, it did not eliminate questionable optimizations. A separate test set can expose overfitting to particular workloads, but it cannot rule out unsupported claims of improvement or shortcuts that exploit assumptions shared by the development and test environments.

Automated checks can enforce explicit requirements, but those requirements may still fail to capture the intended behavior, even in formally verified systems [22]. Deciding which requirements capture a meaningful consensus experiment therefore still requires domain knowledge. We reviewed the generated implementation and intervened where a latency improvement depended on assumptions we did not consider acceptable.

For example, our benchmark sampled network jitter independently for each message. The agent exploited this by sending duplicate requests simultaneously over two TCP connections to the same destination and using whichever arrived first. This reduced measured latency by giving each logical request two chances to receive a smaller delay, an advantage that is unlikely to materialize on a real network, where messages sent together to the same destination are rarely reordered. Moreover, due to borrowing ideas from BODEGA, A-GREEMENT inherited one of its flaws, detailed in § 8. We removed both the inherited flaw and benchmark-hacking before the final evaluation.

## 7 Evaluation

We implement and evaluate the efficiency of A-GREEMENT. Our experiments answer the following questions:

• How does failure-free latency of A-GREEMENT compare with that of state-of-the-art protocols across workloads, deployment sizes and topologies (§ 7.3)?

• Can A-GREEMENT keep making progress and recover when replicas fail, despite the role assignments that reduce its normal-case latency (§ 7.4)?

• What does it cost, in terms of tokens and time, for AI-GREEMENT to discover A-GREEMENT (§ 7.5)?

## 7.1 Implementation

We implement AI-GREEMENT as a repository-based harness and run it using the goal mode of CODEX powered by GPT-6.1 Sol High, with the prompt in Appendix B. A toplevel AGENTS.md encodes the researcher–developer–verifier workflow, consistency requirements across designs, implementations, and proofs, and the performance and recovery criteria for accepting candidates. Four project-specific skills provide operational guidance for knowledge retrieval, prototype development and benchmarking, LEAN formalization, and LEAN proof construction. The skills connect these roles to the three components in § 3.

Our implementation is organized into three repositories: CONSENSUSWIKI, CONSENSUSARENA, and CONSEN-SUSVERIFIER. CONSENSUSWIKI provides the knowledge base described in § 3.1, comprising 97 Markdown documents and approximately 8,600 lines. CONSENSUSARENA provides the implementation framework described in § 3.2. It contains approximately 6,800 lines of shared Go runtime code, 19,100 lines across the baseline implementations, and 8,300 lines for A-GREEMENT. CONSENSUSVERIFIER provides the verification framework described in § 3.3. It comprises approximately 135K lines of LEAN code, covering shared definitions and formal specifications and proofs for MULTI-PAXOS, FASTPAXOS, SWIFTPAXOS, EPAXOS\*, BODEGA and A-GREEMENT. Of these, approximately 98K lines support the final model and proofs of A-GREEMENT. For BODEGA, we prove its linearizability when the earlyaccept optimization is disabled; when it is enabled, we exhibit a counterexample to linearizability (§ 8).

To compare the baselines under common deployment conditions without hardcoded assumptions about topology or key popularity, we adapt their planners to use supplied topology inputs and, where needed, independent pre-measurement workload samples, following the core configuration principles of the respective papers and source code. We also address implementation issues to support the broader range of workloads and deployment configurations exercised by our benchmarks.

Specifically, for BODEGA, whose original evaluation scripts manually select the leader, we reuse our MULTI-PAXOS planner to choose the initial leader that minimizes the mean client–leader round-trip latency plus the estimated time to collect a majority of acknowledgments. When an independent pre-measurement sample contains only reads, we configure every replica as a responder for all keys, enabling local reads throughout the deployment. In addition, we also disable its early-accept notification optimization.

## 7.2 Experimental Setup

We evaluate A-GREEMENT against SWIFTPAXOS [63], BODEGA [29], MULTI-PAXOS [38], and KCENSUS [9]. We consider four deployment sizes with 5, 9, 13, and 17 repli cas, each containing an equal number of clients. At each size, we generate one uniform, one clustered, and one colocated network topology, for a total of twelve configurations. Uniform topologies place all participants independently, clustered topologies group them around three centers, and colocated topologies place each client at the same location as one replica. Network delay, asymmetry, jitter, bandwidth, client placement, leader placement, workload arrival rate, value size, key-space size, and key-access skew vary across these configurations.

Workload. We generate key–value workloads with read-/write mixes corresponding to YCSB workloads A, B, and C [16]. Reads and writes correspond to GET and PUT operations, respectively. Workload A contains 50% writes, workload B contains 5% writes, and workload C is read-only. The per-client arrival rate is sampled independently for each topology between 500 and 1,800 requests per second. Commands carry values of 256, 1,000, or 2,048 bytes and operate over either 250,000 or one million keys. For each topology, we sample a Zipf skew parameter θ from [0.65, 1.15] and hold it fixed throughout the experiment. Each request independently selects a popularity rank r with probability $p _ { r } \propto ( r + 1 ) ^ { - \theta } ,$ ranks are mapped to key identifiers through a fixed random permutation shared by all clients. Across the twelve generated topologies, the sampled values of θ range from 0.651 to 1.120. Clients submit requests according to an open-loop Poisson process, meaning that request arrivals are scheduled independently of response completion. After a five-second warm-up, clients generate requests at the configured arrival rates during a ten-second measurement window. After the measurement window ends, we wait for all requests submitted during that window to complete and include their end-to-end latencies.

![](images/b520e760eb831ba0ed54ac09c78ccd9b4d01414cbc0dcaa1b70f72e97761db9a.jpg)  
Figure 5: Failure-free latency averaged with equal weight over the six deployments with 5 or 9 replicas. Bars report mean latency, while upper whiskers report p99 latency. Lower bound is the idealized communication baseline.

Hardware. We run all experiments on EPFL’s SCITAS Jed CPU cluster using SLURM [34]<sup>2</sup>. Experiments use 2–5 compute nodes, depending on the deployment size. Each node contains two 36-core Intel Xeon Platinum 8360Y processors running at 2.40 GHz, for a total of 72 physical cores, and 492 GB of memory. Replicas, clients, and the coordinating master execute as separate SLURM tasks, each requesting eight CPUs. Communication uses TCP over the cluster’s Ethernet interconnect. CONSENSUSARENA uses Toxiproxy to emulate the generated network conditions. For each topology, we select a bandwidth limit of 200 Mb/s, 1 Gb/s, or 4 Gb/s for each TCP connection and direction. We also select a jitter parameter $J \in \{ 0 , 3 , 1 0 , 2 0 \}$ ms for each topology. Toxiproxy introduces random variations of approximately J around each nonzero configured directional delay.

Metrics. Our primary metrics are mean and p99 end-toend latency. For each measured read or write, latency is measured from client submission until the client receives sufficient protocol-specific evidence to declare the operation complete. Mean latency averages these measurements, while p99 latency is the value below which 99% of measured operations complete and therefore captures tail performance.

## 7.3 Normal Case Performance

Figure 5 reports mean and p99 latency for each workload, averaged across the six deployments with 5 or 9 replicas. We exclude larger deployments from this average because, at 13 and 17 replicas, saturation of individual baselines dominates it: for example, SWIFTPAXOS’s mean latency reaches 516.8 ms with 17 replicas in the uniform topology. Fine-grained results for every algorithm, topology, and workload, including all twelve configurations, are reported in Appendix D, together with the corresponding operation-level read and write latency breakdowns.

On workload A (50% writes), A-GREEMENT reduces mean latency by 12.2% relative to SWIFTPAXOS, the alternative with the lowest mean and p99 latency, while its p99 latency is 13.8% higher than that of SWIFTPAXOS. Notably, the mean read and write latencies of A-GREEMENT are 56.5 ms and 145.4 ms, respectively, compared with 115.0 ms and 115.2 ms for SWIFTPAXOS. Thus, A-GREEMENT’s overall advantage over SWIFTPAXOS comes from its local-read path, at the cost of slower writes. BODEGA also provides fast reads, with a mean read latency of 77.2 ms, but incurs a substantially higher mean write latency of 212.6 ms. These results show that A-GREEMENT combines fast local reads with a write path that avoids BODEGA’s substantial write-latency penalty.

On workload B (5% writes), it reduces mean and p99 latency by 33.3% and 30.1%, respectively, relative to BODEGA, the alternative with the lowest mean latency, while its p99 latency is 3.4% higher than that of SWIFTPAXOS, the alternative with the lowest p99 latency. As in workload A, the main weakness of BODEGA is its write path, for which it spent on average 207.3 ms compared to A-GREEMENT’s 150.5 ms.

On the read-only workload C, A-GREEMENT achieves mean and p99 latencies of 34.0 ms and 56.3 ms, respectively, which is comparable to BODEGA, because both protocols use lease-based local reads.

The lower bound in Figure 5 is an analytical communication reference computed from the same network topologies. For each client, it assigns reads the minimum direct round-trip latency to any replica and writes the time to collect replies from the fastest majority, allowing shortest paths through replicas. It assumes ideal read freshness and write ordering, and excludes processing, queueing, serialization, bandwidth limits, and jitter.

## 7.4 Recovery under Failure

We evaluate recovery with five replicas under workloads A, B, and C across three network topologies. The offered load is 20,000 requests per second. Each run lasts 40 seconds, includ ing an initial five-second warm-up. At 5 seconds, we crash the current leader for protocols with a leader, or a replica for protocols without one. At 20 seconds, we crash one additional replica, prioritizing the latest reported live leader when available and otherwise selecting the surviving replica. The two crashes leave a majority of three replicas. Failed replicas remain stopped for the rest of the run. Figure 6 reports completed requests per second throughout this sequence.

Across all nine runs, A-GREEMENT experiences a temporary throughput drop after each failure and recovers to at least 90% of its initial failure-free throughput within approximately three seconds. It maintains this level until the next crash or the end of the run. After the second crash, its throughput settles near 20,000 requests per second through the end of the observation window, as do the baselines. The temporary peaks above this rate reflect the completion of requests accumulated during the interruption; they do not indicate a higher sustainable throughput. The curves show that A-GREEMENT retains progress with two failed replicas, although it does not consistently recover faster than the other protocols.

![](images/17298ccd68ae4bb641bfd41f8003fd32f52a004676311da79e13bcb09da0aee9.jpg)  
Figure 6: Recovery throughput with five replicas under workloads A, B, and C (left to right). Lines show mean completed requests per second across three topologies; bands show their minimum–maximum range.

These failures test whether the role assignments used to reduce normal-case latency prevent progress when replicas become unavailable. As described in § 4.5, recovery replaces failed role holders, preserves potentially chosen writes, and reestablishes read authority. If the per-key write path is active, recovery transfers its retained state into the shared log. The recovered system can therefore continue processing requests using the surviving majority without requiring the original fast-path participants to return.

## 7.5 Discovery Cost

The complete discovery process, including 20 search rounds and formalization and verification of the final protocol, took approximately 24 hours in total. The search rounds consumed 68.14 million tokens across 333 model responses, including 67.83 million input tokens and 0.308 million output tokens. Thus, cumulative token usage largely reflects repeated processing of the accumulated context, rather than newly generated content. Across the complete process, model reasoning accounted for approximately 23 hours. The remaining elapsed time includes output generation, tool execution, explicit waiting, context compaction, and unattributed runtime intervals. Formalization and verification of the final protocol consumed an additional 710.31 million tokens, bringing total usage to approximately 778.45 million. For the stage breakdown in Figure 7, we attribute this verification usage to iteration 19.

Figure 7 attributes token usage to execution stages. Research and implementation includes literature and code inspection, candidate selection, feasibility analysis, parent-protocol diagnostic experiments, protocol design, coding, compilation, and initial candidate benchmark submission. Performance evaluation covers subsequent monitoring, troubleshooting, and metric analysis; recovery evaluation covers crashrecovery monitoring and analysis. Finalization includes evidence preservation and reporting.

![](images/4060b0ca1f37209617695adf2baef93c700a1cccaa097a033ecca355c60545d4.jpg)

![](images/8e226978419b2570daf08a6d148342a9f7830aa02aeae8f73be9763a5b08b7bb.jpg)  
Figure 7: Stage-attributed token usage, including input and output tokens. (a) Per-iteration usage in millions of tokens, with a broken axis omitting the range from 10 to 700 million. Formalization and verification adds 710.31 million tokens at iteration 19. (b) Phase shares within the search, excluding formalization and verification.

Formalization and verification accounts for 91.25% of the combined token usage. Within the original search, excluding this additional stage as in Figure 7(b), research and implementation accounts for 46.67%. Performance and recovery evaluation account for 20.78% and 15.60%, respectively. Finalization accounts for 16.28%, and context compaction for 0.67%. Ten iterations produced benchmarked candidate prototypes, while ten were rejected before candidate implementation. Early rejection avoided candidate evaluation, but was not necessarily inexpensive: feasibility assessment could itself require instrumenting and benchmarking a parent protocol.

Individual iterations illustrate these costs. Iteration 3 used a parent-protocol diagnostic to establish that the proposed execution-lane change had no observed prefix-blocking delay to remove. Iteration 5 similarly measured an optimistic latency-saving bound below the promotion threshold and rejected the candidate before implementation. Together, these two iterations consumed 7.36 million tokens. Iteration 9 investigated acknowledgment queueing and implemented cumulative prefix certificates, but reduced acknowledgment traffic did not translate into improved client latency. By contrast, iteration 14 combined the rejected iteration 13’s read racing mechanism with per-key completion rules, passed the performance gate and all 27 recovery cases, and became the final promoted version. These results show that discovery cost includes feasibility diagnostics, unsuccessful experiments, and reusable findings from rejected candidates, in addition to implementation of the final design.

## 8 Finding Vulnerabilities in Existing Consensus Protocols

While constructing the reference LEAN specifications and proofs in § 3.3, AI-GREEMENT uncovered four vulnerabilities in existing consensus protocols that, to the best of our knowledge, were previously unreported. Two affect the optimizations introduced by EPaxos Revisited [75]: they allow replicas to execute conflicting commands in opposite orders. Two concern Bodega [29]: its optional early-accept optimization can expose inconsistent values across read paths, while its client-side timeout retries can execute one client write twice. Appendix E details the counterexamples and analysis.

Pruning with non-final metadata. Dependency pruning uses sequence numbers and dependencies to skip commands that appear to execute after the current command. However, preaccepted metadata can change before commitment. A delayed proposal can therefore make a dependency appear safe to prune, allowing one replica to execute commands in an order opposite to another replica’s. The counterexample in § E.1 shows why pruning requires committed or otherwise immutable metadata.

Pruning that hides unresolved dependencies. Committed metadata alone is insufficient: skipping a committed node can also hide commands reachable through its dependency branch. One replica can consequently execute a command before an unresolved conflicting dependency, while another preserves the opposite order. § E.2 presents the counterexample.

Inconsistent visibility across read paths. BODEGA [29]’s early-accept notification allows a responder to return a write after collecting majority accept notifications, while a leader reads only committed state. As mentioned in § E.3, consider WRITE(k, 7) for an initially absent key. All replicas accept the write, but their acceptance replies to the leader are delayed, so the leader has not committed it or replied to the writing client. A responder nevertheless receives enough accept notifications to return 7 to a read. After that read completes, a second read reaches the leader, which returns absence from its unchanged committed state. Returning 7 and then absence in two successive reads violates linearizability.

Duplicate writes after timeout retries. The client-side timeout retry path in BODEGA [29]’s implementation assigns a new request identifier to each attempt, so the server can execute the original request and its retry as separate writes. As mentioned in § E.4, consider a write that sets x = 1 but whose reply is delayed. Another client reads 1 and then completes a write of 2. After a timeout, the original client retries with a new identifier; the retry sets x = 1 again, and a later read returns 1. If both attempts represent one client write, the first read requires that write to precede the write of 2, so the final read must return 2. This violates linearizability even though replicas agree on the order of the individual requests.

## 9 Related Work

State-machine replication. PAXOS supplies the ballot-andquorum safety baseline for replicated agreement [38]. FAST-PAXOS then lets proposers send values directly to acceptors in prepared fast rounds, enabling two-message-delay learning without collisions at the cost of stronger quorum intersection and recovery rules [40]. Later systems remove different parts of the consensus critical path. MENCIUS spreads log slots across rotating coordinators [48]. EPAXOS and ATLAS use per-command coordination and dependency ordering, and ATLAS additionally sizes its fast quorum around a chosen concurrent site-failure budget [20, 49]. RABIA uses randomized weak multi-valued consensus to avoid a distinguished leader and separate fail-over protocol [58]. Later analysis exposed EPAXOS’s tail-latency and recovery subtleties [75], while SWIFTPAXOS reduces the cost of dependency disagreement through leader-including quorums and double voting [63].

Other designs change the system boundary or optimize a particular workload. NOPAXOS and HYDRA move commoncase ordering into the network [15,41]. CURP uses witnesses for unordered durability when operations commute [59], and JETPACK retrofits a parallel one-RTT path onto compatible consensus protocols [72]. COPILOT tolerates one slow replica by redundantly processing commands at two pilots [52]. BODEGA combines roster leases with responder-covering write quorums for local linearizable reads during interfering

writes [29].

AI-driven algorithm and systems discovery. ALPHAE-VOLVE combines LLM-driven evolutionary search with automated evaluators to discover improved algorithms [54]. ADRS studies this approach in systems research, using OPENEVOLVE [67], an open-source framework inspired by ALPHAEVOLVE, for tasks such as spot-instance scheduling, mixture-of-experts load balancing, and transaction scheduling [14]. GLIA instead uses a multi-agent LLM workflow, in which a researcher agent proposes, implements, and evaluates designs in a simulator under a supervisor agent, to design request-routing, scheduling, and autoscaling policies for LLM-serving clusters [24]. These frameworks optimize isolated policies in which every candidate is valid by construction or checked exactly, and a simulator scores it in seconds. ADRS identifies consensus protocols, whose logic spans state management, communication, and failure handling, as a poor fit for this approach. AI-GREEMENT targets this gap, where three challenges arise together: (C1) faster candidates can break safety, (C2) tests cannot establish safety, and (C3) each candidate takes minutes to hours to evaluate. Table 4 and Table 6 in Appendix F compare these approaches and the challenges they face.

Agent-synthesized verified systems. The SKYDISCOVER framework [44], from the authors of ADRS, extends this line from evolving isolated programs to synthesizing complete systems. Its proof-driven mode, IDS [2], is the work closest to AI-GREEMENT: agents jointly synthesize an implementation and its ROCQ proof, use incomplete proofs as a search oracle, and use benchmarks to steer toward efficient designs, producing verified distributed key-value stores for session and causal consistency specifications. IDS, however, starts from a fixed, human-written specification, so performance only ranks candidates that already meet it (sidestepping C1), and its design freedom lies in data layout and proof structure rather than in new protocol mechanisms. Its targets involve neither fault-tolerant agreement nor recovery. AI-GREEMENT instead designs new consensus protocols under a performance objective. Conversely, IDS proves the extracted implementation, whereas our LEAN proofs cover a protocol model whose correspondence to the prototype is additionally checked by a human consensus expert.

AI for consensus. Prior AI work on consensus finds bugs in existing protocols, generates small protocols, or assists human protocol designers. AGORA’s agents inspect RAFT, EPAXOS, HOTSTUFF, and BULLSHARK codebases, synthesize attacks, and generate executable tests. AGORA reports 15 new defects classified as protocol-level logic bugs because the observed executions violate protocol properties [45]. GGMS learns protocols without an LLM, combining Monte Carlo tree search, a transformer action encoder, and a model checker [32]. It targets synchronous consensus and atomic commit with crash failures, and verifies candidates by exhaustive model checking for at most four processes. For consensus, it rediscovers FLOODSET [46]. By dropping the performance objective and restricting the search to four processes, it sidesteps C1 and keeps evaluation cheap (C3), but its guarantees do not extend beyond the checked instance sizes. An LLM also substantially assisted the design and proofs of a communication-efficient Byzantine broadcast protocol [19]. There, however, humans drive an interactive collaboration toward an asymptotic bound established by pen-and-paper proof; no candidate is implemented or measured, so C3 does not arise and safety checking (C2) rests on the authors. AI-GREEMENT instead faces all three challenges: it designs protocols under a latency objective and evaluates them on deployments of up to 17 replicas.

## 10 Conclusion: The End of Consensus Research?

Our answer is no; this is just a new beginning. We believe this work marks a turning point in how consensus research is conducted. The low latency achieved might still be decreased in specific situations we did not focus on. More generally, our consensus solution could be extended to other failure models and settings. Our results show that AI agents can take on substantial parts of the research process, including retrieving protocol knowledge, designing and implementing candidates, as well as constructing formal arguments and counterexam ples. Human expertise, however, remains essential for defining performance objectives and safety requirements, designing evaluation environments, and reviewing agent behavior and research outcomes. A crucial question therefore is: how do our results reshape the role of consensus researchers?

## 10.1 Where Is Human Judgment Needed?

From our experiences, we found that human judgment was needed before, during, and after the agentic search.

Before the search. Human expertise is required before the quest starts. Researchers need to develop the agent harness by checking the protocol knowledge base, defining the implemen tation interfaces, and choosing the fault model, correctness properties, performance objectives, and evaluation conditions (§ 3). Although an agent drafted the knowledge-base notes, we verified them against the source papers and corrected errors. This preparation determines which designs the agent can express and what its feedback measures.

During the search. Inspecting an agentic run while it is in progress allows researchers to steer the agent early and avoids restarting a long run that wastes many tokens. Giving the agent an explicit budget, such as a fixed amount of time to search for a new candidate, can reduce the need for such manual checks.

After the search. Human review remains necessary when interpreting improvements. As described in ${ \ S \ 6 , }$ the agent duplicated requests over two TCP connections to the same destination to exploit independently sampled network jitter. The resulting latency reduction reflected an opportunity in the benchmark’s delay model, and we removed the mechanism before the final evaluation. A separate test set helps expose overfitting, but shared assumptions in the development and test environments can still reward such shortcuts. Researchers must therefore examine both candidate protocols and their evaluation environments to reduce the risk of reward hacking.

Human review can also expose inefficiencies that the agent’s own tuning left in place. In A-GREEMENT’s batching (§ 5.1), the leader cuts a batch into log entries in arrival order. This couples some writes to the slower completion of a hot key and fragments batches when completion owners alternate. Grouping each batch by completion owner would remove both inefficiencies and is safe, but the agent chose not to do it. Moreover, upon a ballot change A-GREEMENT does not restore the cold quorum (§ 4.5), moving all write-related actions to the common log, essentially sacrificing a part of its optimization for no reason. We did not fix either of these inefficiencies, so that the evaluated protocol stays as close as possible to the agent’s design, with only correctness-critical fixes.

Human review is also necessary for formal verification. LEAN checks proofs against the encoded specification; researchers must check that this specification faithfully captures the protocol and its implementation. In our experience, formalizing a protocol from pseudocode presents substantially different challenges from formalizing it from implementation code. The discovery loop could generate an implementation first and then formalize its behavior, or develop both the implementation and the formal specification from shared pseudocode. Which approach produces more faithful specifications with less human intervention remains an open question.

## 10.2 The Changing Role of Researchers

As agents take on more research tasks, we believe that human researchers retain three central responsibilities. First, they must define meaningful problems and translate practical requirements into explicit research objectives. This includes deciding which workloads and deployment conditions matter, what safety and availability guarantees are required, and how to prioritize latency, throughput, and resource costs. These decisions give the agent a concrete target and determine which tradeoffs are acceptable. Second, they must test and review agent outputs, which may still contain hallucinations, implementation errors, or unsupported conclusions.

Generated code, explanations, and experimental reports require independent scrutiny before they can support a research claim. Researchers must assess whether the evidence justifies the claimed improvement, investigate failures and unexpected behavior, and determine how broadly the conclusions apply. Third, they remain important for proposing new ideas and opening directions beyond the current search. The search reported here primarily combined and adapted known mechanisms, showing how agents can explore combinations that are costly to investigate manually. Human researchers can con tribute new abstractions, question assumptions inherited from existing protocols, and formulate hypotheses that expand the design space. Agents can then help develop these ideas into protocols, construct implementations and proofs, and evaluate their consequences, allowing researchers to pursue more alternatives and refine the ideas using concrete evidence.

## References

[1] Deepak Bhaskar Acharya, Karthigeyan Kuppan, and B Divya. Agentic ai: Autonomous intelligence for complex goals—a comprehensive survey. IEEE Access, 13:18912–18936, 2025. doi:10.1109/ACCESS.2025.3532853.

[2] Shubham Agarwal, Alexander Krentsel, Shu Liu, Mert Cemri, Audrey Cheng, Rui Meng, Tomas Pfister, Chun-Liang Li, Sylvia Ratnasamy, Aditya Parameswaran, Matei Zaharia, Ion Stoica, and Mohsen Lesani. Inductive deductive synthesis: Enabling AI to generate formally verified systems, 2026. arXiv:2605.23109.

[3] Ailidani Ailijiang, Aleksey Charapko, Murat Demirbas, and Tevfik Kosar. WPaxos: Wide area network flexible consensus. IEEE Transactions on Parallel and Distributed Systems, 31(1):211–223, 2020. doi:10.1109/TPDS.2019.2929793.

[4] Anthropic. Claude Code: Overview. https://code.claude. com/docs/en/overview, 2026. Official documentation, accessed 10 October 2026.

[5] Anysphere. Cursor Agent Documentation. https://cursor. com/docs/agent/overview, 2026. Official documentation, accessed 11 October 2026.

[6] William J. Bolosky, Dexter Bradshaw, Randolph B. Haagens, Norbert P. Kusters, and Peng Li. Paxos replicated state machines as the basis of a high-performance data store. In 8th USENIX Symposium on Networked Systems Design and Implementation (NSDI 11). USENIX Association, 2011. [link].

[7] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020. [link].

[8] Ethan Buchman, Jae Kwon, and Zarko Milosevic. The latest gossip on BFT consensus, 2018. [link]. arXiv:1807.04938.

[9] Clément Burgelin, Antoine Murat, Gal Sela, Marcos K. Aguilera, and Rachid Guerraoui. KCensus: Synthesizing latencyoptimal consensus fast paths (extended version), 2026. arXiv: 2609.31302.

[10] Mike Burrows. The Chubby lock service for loosely-coupled distributed systems. In 7th USENIX Symposium on Operating Systems Design and Implementation (OSDI 06), pages 335– 350. USENIX Association, 2006. [link].

[11] Miguel Castro and Barbara Liskov. Practical Byzantine fault tolerance and proactive recovery. ACM Transactions on Computer Systems, 20(4):398–461, 2002. doi:10.1145/571637. 571640.

[12] Tushar Deepak Chandra, Robert Griesemer, and Joshua Redstone. Paxos made live - an engineering perspective (2006 invited talk). In Proceedings of the 26th Annual ACM Symposium on Principles of Distributed Computing, 2007. doi: 10.1145/1281100.1281103.

[13] Aleksey Charapko, Ailidani Ailijiang, and Murat Demirbas. PigPaxos: Devouring the communication bottlenecks in distributed consensus. In Proceedings ofthe 2021 International Conference on Management of Data (SIGMOD). ACM, 2021. doi:10.1145/3448016.3452834.

[14] Audrey Cheng, Shu Liu, Melissa Pan, Zhifei Li, Bowen Wang, Alex Krentsel, Tian Xia, Mert Cemri, Jongseok Park, Shuo Yang, Jeff Chen, Lakshya Agrawal, Aditya Desai, Jiarong Xing, Koushik Sen, Matei Zaharia, and Ion Stoica. Barbarians at the gate: How AI is upending systems research, 2025. arXiv: 2510.06189.

[15] Inho Choi, Ellis Michael, Yunfan Li, Dan R. K. Ports, and Jialin Li. Hydra: Serialization-free network ordering for strongly consistent distributed applications. In 20th USENIX Symposium on Networked Systems Design and Implementation (NSDI 23), pages 293–320. USENIX Association, 2023. [link].

[16] Brian F. Cooper, Adam Silberstein, Erwin Tam, Raghu Ramakrishnan, and Russell Sears. Benchmarking cloud serving systems with YCSB. In Proceedings of the 1st ACM Symposium on Cloud Computing (SoCC), pages 143–154, New York, NY, 2010. ACM. doi:10.1145/1807128.1807152.

[17] James C. Corbett, Jeffrey Dean, Michael Epstein, Andrew Fikes, Christopher Frost, JJ Furman, Sanjay Ghemawat, Andrey Gubarev, Christopher Heiser, Peter Hochschild, Wilson Hsieh, Sebastian Kanthak, Eugene Kogan, Hongyi Li, Alexander Lloyd, Sergey Melnik, David Mwaura, David Nagle, Sean Quinlan, Rajesh Rao, Lindsay Rolig, Yasushi Saito, Michal Szymaniak, Christopher Taylor, Ruth Wang, and Dale Woodford. Spanner: Google’s globally-distributed database. In 10th USENIX Symposium on Operating Systems Design and Imple mentation (OSDI 12). USENIX Association, 2012. [link].

[18] Leonardo de Moura and Sebastian Ullrich. The Lean 4 theorem prover and programming language. In Automated Deduction – CADE 28, volume 12699 of Lecture Notes in Computer Science, pages 625–635. Springer, 2021. doi: 10.1007/978-3-030-79876-5\_37.

[19] Yuval Efron, Jovan Komatovic, and Ling Ren. Byzantine broadcast in ${ \tilde { O } } ( n ^ { 2 } )$ communication. Cryptology ePrint Archive, Paper 2026/2504, 2026.

[20] Vitor Enes, Carlos Baquero, Tuanir França Rezende, Alexey Gotsman, Matthieu Perrin, and Pierre Sutra. State-machine replication for planet-scale systems. In Proceedings ofthe 15th European Conference on Computer Systems (EuroSys), pages 24:1–24:15. ACM, 2020. doi:10.1145/3342195.3387543.

[21] Tanisha Fonseca and Gengrui Zhang. WOC: Dual-path weighted object consensus made efficient, 2025. URL: https: //arxiv.org/abs/2512.20485, arXiv:2512.20485.

[22] Eli Goldweber, Weixin Yu, Seyed Armin Vakil Ghahani, and Manos Kapritsos. IronSpec: Increasing the reliability of formal specifications. In 18th USENIX Symposium on Operating Systems Design and Implementation (OSDI 24), pages 875– 891. USENIX Association, 2024. [link].

[23] Mourad Gridach, Jay Nanavati, Khaldoun Zine El Abidine, Lenon Mendes, and Christina Mack. Agentic ai for scientific discovery: A survey of progress, challenges, and future directions, 2025. arXiv:2503.08979.

[24] Pouya Hamadanian, Pantea Karimi, Arash Nasr-Esfahany, Kimia Noorbakhsh, Joseph Chandler, Ali ParandehGheibi, Mohammad Alizadeh, and Hari Balakrishnan. Glia: A humaninspired AI for automated systems design and optimization, 2025. arXiv:2510.27176.

[25] Chris Hawblitzel, Jon Howell, Manos Kapritsos, Jacob R. Lorch, Bryan Parno, Michael L. Roberts, Srinath T. V. Setty, and Brian Zill. Ironfleet: Proving practical distributed systems correct. In Proceedings of the 25th ACM Symposium on Operating Systems Principles (SOSP), pages 1–17, 2015. doi:10.1145/2815400.2815428.

[26] Maurice P. Herlihy and Jeannette M. Wing. Linearizability: A correctness condition for concurrent objects. ACM Transactions on Programming Languages and Systems, 12(3):463–492, 1990. doi:10.1145/78969.78972.

[27] Christopher Hodsdon, Zijian Qin, Khiem Ngo, Siddhartha Sen, Ethan Katz-Bassett, and Wyatt Lloyd. Avicenna: Masking slowdowns in replicated state machines with counterfactual evaluation. In Proceedings ofthe 21st European Conference on Computer Systems (EuroSys). ACM, 2026. doi:10.1145/ 3767295.3803615.

[28] Heidi Howard, Dahlia Malkhi, and Alexander Spiegelman. Flexible Paxos: Quorum intersection revisited, 2016. arXiv: 1608.06696.

[29] Guanzhou Hu, Andrea C. Arpaci-Dusseau, and Remzi H. Arpaci-Dusseau. Bodega: Localized linearizable reads at anywhere anytime via roster leases. In 20th USENIX Symposium on Operating Systems Design and Implementation (OSDI 26), pages 1243–1281. USENIX Association, 2026. [link].

[30] Dongxu Huang, Qi Liu, Qiu Cui, Zhuhe Fang, Xiaoyu Ma, Fei Xu, Li Shen, Liu Tang, Yuxing Zhou, Menglong Huang, Wan Wei, Cong Liu, Jian Zhang, Jianjun Li, Xuelian Wu, Lingyu Song, Ruoxi Sun, Shuaipeng Yu, Lei Zhao, Nicholas Cameron, Liquan Pei, and Xin Tang. TiDB: A Raft-based HTAP database. Proceedings of the VLDB Endowment, 13(12):3072–3084, 2020. doi:10.14778/3415478.3415535.

[31] Hugging Face. Anatomy of a frontier lab agent intrusion: A technical timeline of the july 2026 incident. https://huggingface.co/blog/ agent-intrusion-technical-timeline, July 2026. Published 27 July 2026.

[32] Yujie Hui, Xiaoyi Lu, Andrew Perrault, and Yang Wang. Learning provably correct distributed protocols without human knowledge. arXiv preprint arXiv:2601.22369, 2026. doi:10.48550/arXiv.2601.22369.

[33] Yujie Hui, Drew Ripberger, Xiaoyi Lu, and Yang Wang. Learning distributed protocols with zero knowledge. In NeurIPS 2023 Workshop on Machine Learningfor Systems, 2023. [link].

[34] Morris A. Jette and Tim Wickberg. Architecture of the slurm workload manager. In Job Scheduling Strategiesfor Parallel Processing, volume 14283 of Lecture Notes in Computer Science. Springer, 2023. doi:10.1007/978-3-031-43943-8\_ 1.

[35] Carlos E Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, and Karthik Narasimhan. Swe-bench: Can language models resolve real-world github issues? In International Conference on Learning Representations, volume 2024, pages 54107–54157, 2024. [link].

[36] Antonios Katsarakis, Vasilis Gavrielatos, M. R. Siavash Katebzadeh, Arpit Joshi, Aleksandar Dragojevic, Boris Grot, and

Vijay Nagarajan. Hermes: A fast, fault-tolerant and linearizable replication protocol. In Proceedings ofthe 25th International Conference on Architectural Support for Programming Languages and Operating Systems (ASPLOS), pages 201–217. ACM, 2020. doi:10.1145/3373376.3378496.

[37] Leslie Lamport. Time, clocks, and the ordering of events in a distributed system. Communications ofthe ACM, 21(7):558– 565, 1978. doi:10.1145/359545.359563.

[38] Leslie Lamport. Paxos made simple. ACM SIGACT News, 32(4):51–58, 2001. [link].

[39] Leslie Lamport. Generalized consensus and Paxos. Technical Report MSR-TR-2005-33, Microsoft Research, 2005. [link].

[40] Leslie Lamport. Fast paxos. Distributed Computing, 19(2):79–103, 2006. doi:10.1007/s00446-006-0005-x.

[41] Jialin Li, Ellis Michael, Naveen Kr. Sharma, Adriana Szekeres, and Dan R. K. Ports. Just say NO to paxos overhead: Replacing consensus with network ordering. In 12th USENIX Symposium on Operating Systems Design and Implementation (OSDI 16), pages 467–483. USENIX Association, 2016. [link].

[42] Shu Liu, Shubham Agarwal, et al. EvoX: Meta-evolution for automated discovery, 2026. arXiv:2602.23413.

[43] Shu Liu, Shubham Agarwal, Alexander Krentsel, Mert Cemri, Sidharth Sankhe, Ziming Mao, Alexandros G. Dimakis, Matei Zaharia, and Ion Stoica. Building specialized systems we can trust with agents. SkyDiscover Blog, September 2026. URL: https://skydiscover-ai.github.io/ blog-skysynth.html.

[44] Shu Liu, Mert Cemri, Shubham Agarwal, Alexander Krentsel, Ashwin Naren, Qiuyang Mang, Zhifei Li, Akshat Gupta, Monishwaran Maheswaran, Audrey Cheng, Melissa Pan, Ethan Boneh, Kannan Ramchandran, Koushik Sen, Matei Zaharia, Alexandros G. Dimakis, and Ion Stoica. SkyDiscover: A flexible, adaptive framework for AI-driven scientific and algorithmic discovery. In Proceedings of the ACM Conference on AI and Agentic Systems (CAIS ’26), pages 1223– 1227. Association for Computing Machinery, 2026. doi: 10.1145/3786335.3813221.

[45] Xiang Liu, Sa Song, Zhaowei Zhang, Huiying Lan, Jason Zeng, Ming Wu, Michael Heinrich, Yong Sun, and Ceyao Zhang. Agora: Toward autonomous bug detection in production-level consensus protocols with LLM agents. In Forty-third International Conference on Machine Learning (ICML), 2026. [link].

[46] Nancy A. Lynch. Distributed Algorithms. Morgan Kaufmann, 1996.

[47] Daniel J. Mankowitz, Andrea Michi, Anton Zhernov, Marco Gelmi, Marco Selvi, Cosmin Paduraru, Edouard Leurent, Shariq Iqbal, Jean-Baptiste Lespiau, Alex Ahern, Thomas Köppe, Kevin Millikin, Stephen Gaffney, Sophie Elster, Jackson Broshear, Chris Gamble, Kieran Milan, Robert Tung, Minjae Hwang, Taylan Cemgil, Mohammadamin Barekatain, Yujia Li, Amol Mandhane, Thomas Hubert, Julian Schrittwieser, Demis Hassabis, Pushmeet Kohli, Martin Riedmiller, Oriol Vinyals, and David Silver. Faster sorting algorithms discovered using deep reinforcement learning. Nature, 618:257–263, 2023. doi:10.1038/s41586-023-06004-9.

[48] Yanhua Mao et al. Mencius: Building efficient replicated state machines for WANs. In 8th USENIX Symposium on Operating Systems Design and Implementation (OSDI 08), pages 369– 384. USENIX Association, 2008. [link].

[49] Iulian Moraru, David G. Andersen, and Michael Kaminsky. There is more consensus in egalitarian parliaments. In Proceedings of the 24th ACM Symposium on Operating Systems Principles (SOSP), pages 358–372, 2013. doi:10.1145/2517349. 2517350.

[50] Iulian Moraru, David G. Andersen, and Michael Kaminsky. Paxos quorum leases: Fast reads without sacrificing writes. In Proceedings of the ACM Symposium on Cloud Computing, SOCC ’14, page 1–13, New York, NY, USA, 2014. Association for Computing Machinery. doi:10.1145/2670979. 2671001.

[51] Harald Ng, Seif Haridi, and Paris Carbone. Omni-Paxos: Breaking the barriers of partial connectivity. In Proceedings ofthe 18th European Conference on Computer Systems (EuroSys). ACM, 2023. doi:10.1145/3552326.3587441.

[52] Khiem Ngo et al. Tolerating slowdowns in replicated state machines using copilots. In 14th USENIX Symposium on Operating Systems Design and Implementation (OSDI 20), pages 583–598. USENIX Association, 2020. [link].

[53] Xuying Ning, Katherine Tieu, Dongqi Fu, Tianxin Wei, Zihao Li, Yuanchen Bei, Jiaru Zou, Mengting Ai, Zhining Liu, Ting Wei Li, Lingjie Chen, Yanjun Zhao, Ke Yang, Bingxuan Li, Cheng Qian, Gaotang Li, Xiao Lin, Zhichen Zeng, Ruizhong Qiu, Sirui Chen, Yifan Sun, Xiyuan Yang, Ruida Wang, Rui Pan, Chenyuan Yang, Dylan Zhang, Liri Fang, Zikun Cui, Yang Cao, Pan Chen, Dorothy Sun, Ren Chen, Mahesh Srinivasan, Nipun Mathur, Yinglong Xia, Hong Li, Hong Yan, Pan Lu, Lingming Zhang, Tong Zhang, Hanghang Tong, and Jingrui He. Code as agent harness: Toward executable, verifiable, and stateful agent systems, 2026. arXiv:2605.18747.

[54] Alexander Novikov, Ngân Vu, Marvin Eisenberger, Emilien˜ Dupont, Po-Sen Huang, Adam Zsolt Wagner, Sergey Shirobokov, Borislav Kozlovskii, Francisco J. R. Ruiz, Abbas Mehrabian, M. Pawan Kumar, Abigail See, Swarat Chaudhuri, George Holland, Alex Davies, Sebastian Nowozin, Pushmeet Kohli, and Matej Balog. AlphaEvolve: A coding agent for scientific and algorithmic discovery, 2025. arXiv:2506.13131.

[55] Diego Ongaro and John Ousterhout. In search of an understandable consensus algorithm. In 2014 USENIX annual technical conference (USENIX ATC 14), pages 305–319, 2014. [link].

[56] OpenAI. Introducing ChatGPT. https://openai.com/ index/chatgpt/, November 2022. Blog post, 30 November 2022. Accessed 13 October 2026.

[57] OpenAI. Codex Documentation. https://developers. openai.com/codex/, 2026. Official documentation, accessed 11 October 2026.

[58] Haochen Pan et al. Rabia: Simplifying state-machine replication through randomization. In Proceedings ofthe 28th ACM Symposium on Operating Systems Principles (SOSP), pages 472–487. ACM, 2021. doi:10.1145/3477132.3483582.

[59] Seo Jin Park and John Ousterhout. Exploiting commutativity for practical fast replication. In 16th USENIX Symposium on

Networked Systems Design and Implementation (NSDI 19), pages 47–64. USENIX Association, 2019. [link].

[60] Fernando Pedone and André Schiper. Generic broadcast. In Prasad Jayanti, editor, Distributed Computing (DISC 1999), volume 1693 of Lecture Notes in Computer Science, pages 94–106, Berlin, Heidelberg, 1999. Springer. doi:10.1007/ 3-540-48169-9\_7.

[61] Longfei Qiu, Jingqi Xiao, Ji-Yong Shin, and Zhong Shao. Suredistrib: Verifying almost-sure termination of composite asynchronous byzantine protocols. Proc. ACM Program. Lang., 10(PLDI), June 2026. doi:10.1145/3808293.

[62] Bernardino Romera-Paredes, Mohammadamin Barekatain, Alexander Novikov, Matej Balog, M. P. Brenner, Emilien Dupont, Francisco J. R. Ruiz, Jordan S. Ellenberg, Pengming Wang, Omar Fawzi, Pushmeet Kohli, and Alhussein Fawzi. Mathematical discoveries from program search with large language models. Nature, 625:468–475, 2024. doi: 10.1038/s41586-023-06924-6.

[63] Fedor Ryabinin, Alexey Gotsman, and Pierre Sutra. Swift-Paxos: Fast geo-replicated state machines. In 21st USENIX Symposium on Networked Systems Design and Implementation (NSDI 24), pages 345–369, 2024. [link].

[64] Fedor Ryabinin, Alexey Gotsman, and Pierre Sutra. Making democracy work: Fixing and simplifying egalitarian paxos. In 29th International Conference on Principles of Distributed Systems (OPODIS 2025), volume 361 of Leibniz International Proceedings in Informatics, pages 22:1–22:19. Schloss Dagstuhl – Leibniz-Zentrum für Informatik, 2026. doi:10. 4230/LIPIcs.OPODIS.2025.22.

[65] Fred B. Schneider. Implementing fault-tolerant services using the state machine approach: A tutorial. ACM Computing Surveys, 22(4):299–319, 1990. doi:10.1145/98163.98167.

[66] Gal Sela, Maurice Herlihy, and Erez Petrank. Brief announcement: Linearizability: A typo. In Proceedings ofthe 2021 ACM Symposium on Principles of Distributed Computing, PODC ’21, pages 561–564. Association for Computing Machinery, 2021. doi:10.1145/3465084.3467944.

[67] Asankhaya Sharma. OpenEvolve: An open-source evolutionary coding agent. GitHub repository, 2025. Accessed 17 October 2026. URL: https://github.com/ algorithmicsuperintelligence/openevolve.

[68] Upamanyu Sharma, Ralf Jung, Joseph Tassarotti, M. Frans Kaashoek, and Nickolai Zeldovich. Grove: A separation-logic library for verifying distributed systems. In Proceedings of the 29th ACM Symposium on Operating Systems Principles (SOSP), pages 113–129. ACM, 2023. doi:10.1145/3600006. 3613172.

[69] Joar Skalse, Nikolaus Howe, Dmitrii Krasheninnikov, and David Krueger. Defining and characterizing reward gaming. In Advances in Neural Information Processing Systems, volume 35, pages 9460–9471, 2022. doi:10.52202/ 068431-0687.

[70] Pierre Sutra. On the correctness of Egalitarian Paxos. Information Processing Letters, 156:105901, 2020. doi:10.1016/ j.ipl.2019.105901.

[71] Rebecca Taft, Irfan Sharif, Andrei Matei, Nathan VanBenschoten, Jordan Lewis, Tobias Grieger, Kai Niemi, Andy Woods, Anne Birzin, Raphael Poss, Paul Bardea, Amruta Ranade, Ben Darnell, Bram Gruneir, Justin Jaffray, Lucy Zhang, and Peter Mattis. CockroachDB: The resilient geodistributed SQL database. In Proceedings ofthe 2020 ACM SIGMOD International Conference on Management ofData, pages 1493–1509. Association for Computing Machinery, 2020. doi:10.1145/3318464.3386134.

[72] Ze Tang et al. Jetpack: Consensus made generally fast. In 20th USENIX Symposium on Operating Systems Design and Implementation (OSDI 26), pages 1299–1323. USENIX Association, 2026. [link].

[73] The Rocq Development Team. The rocq prover. https:// rocq-prover.org/, 2026. Accessed October 2026.

[74] Muxin Tian, Zhe Wang, Zhenwei Tang, Blair Yang, Kunlun Zhu, Honghua Dong, Hanchen Li, Xinni Xie, Guangjing Wang, and Jiaxuan You. Swe-bench mobile: Can large language model agents develop industry-level mobile applications? In Proceedings ofthe 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2, KDD ’26, page 8077–8087, New York, NY, USA, 2026. Association for Com puting Machinery. doi:10.1145/3770855.3818488.

[75] Sarah Tollman, Seo Jin Park, and John Ousterhout. EPaxos revisited. In 18th USENIX Symposium on Networked Systems Design and Implementation (NSDI 21), pages 613–632. USENIX Association, 2021. [link].

[76] Muhammed Uluyol, Anthony Huang, Ayush Goel, Mosharaf Chowdhury, and Harsha V. Madhyastha. Near-optimal latency versus cost tradeoffs in geo-distributed storage. In 17th USENIX Symposium on Networked Systems Design and Implementation (NSDI 20), pages 157–180. USENIX Association, 2020. [link].

[77] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017. [link].

[78] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-ofthought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824– 24837, 2022. doi:10.52202/068431-1800.

[79] James R. Wilcox, Doug Woos, Pavel Panchekha, Zachary Tatlock, Xi Wang, Michael D. Ernst, and Thomas Ander son. Verdi: A framework for implementing and formally verifying distributed systems. In Proceedings of the 36th ACM SIGPLAN Conference on Programming Language Design and Implementation (PLDI), pages 357–368, 2015. doi: 10.1145/2737924.2737958.

[80] John Yang et al. SWE-agent: Agent-computer interfaces enable automated software engineering. arXiv preprint arXiv:2405.15793, May 2024. doi:10.48550/arXiv.2405. 15793.

[81] Maofan Yin, Dahlia Malkhi, Michael K. Reiter, Guy Golan Gueta, and Ittai Abraham. HotStuff: BFT consensus with linearity and responsiveness. In Proceedings of the 2019

ACM Symposium on Principles of Distributed Computing (PODC), pages 347–356. ACM, 2019. doi:10.1145/ 3293611.3331591.

## A Detailed A-GREEMENT Specification

This appendix specifies the three write paths introduced in $\ S 4 { : }$ completion at a designated replica, direct client learning from the ordinary log, and direct client learning from the cold perkey history. The first two paths share log state and recovery. We specify their different completion and read-visibility rules explicitly. Message names below describe logical actions; transport framing, compression, and retransmission do not change the evidence required by a handler.

## A.1 State, Identity, and Authority

Requests and retained state. A request c contains an identifier $\operatorname { i d } ( c )$ , its operation, key, and original payload. Retries preserve this identity and command body. An ordinary request also retains its designated reply replica. This identity is needed to determine whether the completion owner controls the request’s reply. The cold fast path completes at the client and has no completion-owner role. Participants are non-Byzantine. Vote sets count distinct replica identities; hashes bind content and are not signatures. We assume that different retained bodies do not collide under the content hash H.

Ordinary state. Ballot b installs leader $L _ { b }$ and responder sets $S _ { b } ( k )$ . Slot s contains an ordered batch $E = ( b , s , C , f r e s h ) ;$ C includes the original requests and their logical reply destinations. A fresh entry was newly proposed in the current ballot; a recovered entry is not fresh. Replicas retain accepted entries, votes, completion identifiers, and the highest contiguous executed slot executedPrefix. The completion-owner assignment, when applicable, is denoted $O _ { b } ( k )$ ; a delegated owner must remain an authorized responder. Otherwise, the leader supplies the default owner used by the read shortcut. The ordinary executed-reply route still follows the request’s retained reply destination.

For a batch, define its required reader coverage as

$$
S _ { b } ( E ) = \bigcup _ { c \in C : \mathrm { o p } ( c ) = \mathrm { W R I T E } } S _ { b } ( \mathrm { k e y } ( c ) ) .
$$

An ordinary completion certificate contains matching acceptance evidence $V ( E )$ satisfying $| V ( E ) | \geq m$ and $S _ { b } ( E ) \subseteq$ $V ( E )$ . A Paxos majority fixes the log value; responder coverage is the additional condition used here before completion or exposure of a not-yet-executed value. Below, an ordinary entry marked covered-chosen satisfies both conditions. A trusted current-ballot commit notification can convey this outcome, but it does not supply a missing batch body or authorize reading a different ballot’s entry.

For a fresh current-ballot entry E, a gate candidate is a non-certificate write whose designated reply replica is its completion owner. If the candidates in E all name the same owner $p ,$ define the completion gate $g ( E ) = p ;$ if there is no candidate or candidates name different owners, let $g ( E ) = \bot$

Normal batching splits entries when candidate ownership changes, and an unsplit conflicting entry conservatively has no gate. A local acceptance counts in $V ( E )$ immediately, but $g ( E )$ does not publish its ACCEPTED or any client vote for the entry until it has itself marked E covered-chosen. Thus the gate advances its read target before its vote can allow another learner to complete the entry.

Cold state. Cold control state has generation $G = \left( b , e \right)$ where e is an epoch within ballot b. Epochs distinguish freeze and resume barriers, including those used by scans. Each key also has a key-round number a. Rounds $( b , e , a )$ are ordered lexicographically; version v identifies the position in the key’s history, independently of its round. A cold vote binds

$$
( b , e , a , k , \nu , \mathrm { i d } ( c ) , H ( c ) , \mathrm { r e p l y T o } ( c ) , \mathrm { t i m e s t a m p } ( c ) , Q _ { b } ) .
$$

Here $H ( c )$ hashes the original operation, key, and payload; the remaining fields bind its request and routing identity. The fast quorum $Q _ { b }$ has size m and remains fixed across key rounds and scan epochs within an installed ballot. A new ballot may select a different quorum for recovery. For each key, a replica retains its promise, chosen prefix, accepted tail, chosen evidence, pending original requests, duplicate identifiers, and readiness state. A bare payload or routing hint supplies none of this authority.

Lease and installation preconditions. Local reads require an installed roster and a majority of unexpired lease grants whose catch-up thresholds the responder has executed. Self-grants count only when the local threshold is met. A leader must additionally have completed ordinary preparation. Grantor and holder deadlines include a conservative clockdrift margin; delayed renewal replies cannot restore a revoked ballot. Before replacing conflicting read authority, grantors revoke the old grants and receive acknowledgments, or wait until the grants expire. Failure suspicion starts this process but cannot replace it. Cold readers also require the installed generation to equal the promised generation and the key to be ready.

Completion policy. Let η be the measured hot boundary and $\tau \geq \eta$ the selected cold boundary, defined in $\ S \ A . 7 .$ . Under the reference policy, fresh ordinary writes to observed ranks below η complete through their reply destination after execution; ranks at or above η are eligible for direct certificates unless routed to the cold path. The alternative policy enables direct certificates for all ordinary keys, including hot and unobserved keys. Unobserved keys never enter the cold path. Recovered entries are ineligible for early ordinary completion, and bridge acceptance disables new early votes. Thus heat classification, completion-owner placement, and certificate eligibility are related planning decisions, not interchangeable definitions.

## A.2 Normal-Operation Details

Ordinary read visibility. Each ACCEPTED vote reports acceptedPrefix, the highest slot through which the sender has accepted a contiguous current-ballot sequence. The mth largest report, counting missing reports as zero, is majorityFixedPrefix. It fixes the preceding entry identities and the first occurrences of duplicate request identifiers; it does not establish responder coverage by itself.

The target of a read depends on whether its responder gates completion. At replica $p ,$ let $A _ { p } ( k )$ be the greatest accepted slot touching $k , C _ { p } ( k )$ the greatest covered-chosen slot touching $k ,$ and $D _ { p } ( k )$ the greatest accepted slot touching k for which the write is not a gate candidate for $p$ or $g ( E ) \neq p$ Missing positions are zero. Direct-certificate writes, writes with another reply destination, recovered entries, differentballot entries, and entries without a unique gate therefore contribute to $D _ { p } ( k )$ . The read target is

$$
t _ { p } ( k ) = \left\{ \begin{array} { l l } { \operatorname* { m a x } \{ C _ { p } ( k ) , D _ { p } ( k ) \} , } & { p = O _ { b } ( k ) , } \\ { A _ { p } ( k ) , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.
$$

The gate can therefore bypass its gated pending entries while its vote remains unpublished: no other learner can yet establish responder coverage. It cannot bypass an accepted entry without this gate. After collecting majority and coverage evi dence, it marks the entry covered-chosen and advances $C _ { p } ( k )$ before releasing its vote; subsequent reads target the chosen write.

After checking read authority, a responder returns $\mathrm { a p - }$ plied state if $t _ { p } ( k ) \leq$ executedPrefix. Otherwise, it can return the last write to k in that slot only when the entry is covered-chosen, fresh, in the current ballot, and within majorityFixedPrefix. This reads a chosen value without executing an arbitrary log suffix. If any required condition is missing, the read waits. strictPrefix disables this shortcut after handoff. No read may bypass an unapplied bridge. Authority and visibility are rechecked when a waiting read is released.

Cold fast-path details. Each cold key advances through versions v in order. The round $( b , e , a )$ distinguishes voting authority from version order; the leader coordinates preparation and conflict recovery. The client sends its original write in COLDPROPOSE to all of $Q _ { b }$ and sends INTENT to notify the coordinator of the pending write. Acceptors require the installed, prepared generation and a fast-enabled key round. Each proposes its locally next version $\nu = \mathrm { k e y P r e f i x } ( k ) + 1 ;$ differing prefixes cannot produce matching votes. They retain the full command, block local reads of the key, and send FASTVOTE directly to the client. Within $\displaystyle ( b , e , a , k , \nu )$ , they vote for only one command; retries repeat that vote. Each vote binds the ballot, epoch, key round, key, version, request identifier, command digest, original reply destination, client timestamp, and fixed quorum. Matching FASTVOTEs from all of $Q _ { b }$ form the fast certificate, completing the write. The client asynchronously disseminates this certificate as a CHO-SENPROOF. Identical retries of chosen requests may receive COLDDONE, binding the original identifier, key, and digest. Replicas execute in per-key order, fetching missing proofs or bodies while retaining pending commands.

Each round permits exactly one fast quorum, reducing Fast Paxos’s recovery-intersection requirement [40] to

$$
P \cap Q _ { b } \neq \emptyset \qquad { \mathrm { f o r ~ e v e r y ~ m a j o r i t y ~ } } P .
$$

Only the designated quorum can vote in a fast round. A mem ber’s failure blocks this path and triggers roster recovery.

Collision fallback and cold reads. Competing writes trigger a higher key round. The coordinator collects complete snapshots from a majority and preserves chosen prefixes and tails that might already have been chosen. It then enters a classic round, proposing one safe value for replica acceptance. The value requires matching votes from every member of $Q _ { b }$ Losing inputs remain pending for later versions. Fast resumption requires a higher prepare, a safe empty remaining tail, and catch-up and READY replies from all of $Q _ { b }$ confirming installation of the prepared key round.

Cold reads require caught-up members of the installed $Q _ { b }$ and leases from the parent roster. The predicate keyReady(k) records installation and catch-up of the chosen key prefix in the prepared round. An unresolved accepted tail blocks reads until a valid chosen prefix is installed. Completed fast writes fence all cold readers, even before proof delivery; neither routing hints nor payloads restore reads.

## A.3 Leader and Cold-Plane Recovery

Leadership change and ordinary recovery. Failure suspicion triggers a higher-ballot leader or replacement of an unavailable required read responder; suspicion alone cannot select values or authorize reads. Replicas suspend old read au thority, revoke grants or await expiration, and retain accepted evidence. The new leader runs Paxos phase one: it sends PRE-PARE, collects complete promises from m replicas, and adopts the highest-ballot accepted entry at each remaining log slot. Only unreported gaps receive internal no-ops. It re-accepts recovered entries under the new roster withfresh cleared. New read responders require majority grants and catch-up through reported thresholds.

Recovering possible cold choices. The coordinator sends COLDPREPARE for a frozen generation $G ^ { \prime } = ( b ^ { \prime } , e ^ { \prime } )$ with $b ^ { \prime } > b$ to freeze the cold plane: replicas stop old cold-path votes and promise $G ^ { \prime }$ over all admitted cold keys and their future versions. A majority P returns PROMISECATALOG replies containing complete immutable catalogs of chosen prefixes, duplicate identifiers, accepted tails, historical ballots, key rounds, quorums, and references to command bodies.

1 function submit(c):   
2 retain pending[id(c)] = exact original c   
3 if c.op = READ:   
4 send c to selected read responder   
5 else if coldEnabled and coldKey(c.key)   
6 and c.op = WRITE and not migrated:   
7 send ColdPropose(c) to every q in Q\_b   
8 send Intent(c) to current coordinator   
9 else:   
10 submit Request(c) for leader ordering   
11 with its designated reply replica   
12   
13 when ClientVote(b,s,id,hE,hc,S) arrives from p:   
14 pre: p in R, pending[id] is eligible WRITE   
15 and hc = H(pending[id]), b>0, s>0,   
16 nonempty S subset R   
17 cert := (b,s,id,hE,hc,S)   
18 voters[cert] := voters[cert] union {p}   
19 if |voters[cert]| >= m   
20 and S subset voters[cert]:   
21 completeOnce(id, acknowledgement)   
22   
23 when FastVote $( \mathrm { b } , \mathsf { e } , \mathsf { a } , \mathrm { k } , \mathsf { v } , \mathrm { i d } , \mathrm { h } , \mathsf { o r i g i n } ,$ ,timestamp,Q\_b)   
24 arrives from p:   
25 pre: p in Q\_b, exact pending[id] is WRITE,   
26 k = pending[id].key, h = H(pending[id]),   
27 origin matches pending[id].replyReplica,   
28 b>0, v>0, |Q\_b|=m, Q\_b subset R   
29 $\begin{array} { r } { \begin{array} { r c l } { \mathrm { x } } & { : = } & { ( \mathrm { b } , \tt e , a , k , v , i d , h , o r i g i n , } \end{array} , } \end{array}$ timestamp,Q\_b)   
30 fastVoters[x] := fastVoters[x] union {p}   
31 if Q\_b subset fastVoters[x]:   
32 disseminate ChosenProof(x,Q\_b,pending[id]) to R   
33 asynchronously   
34 completeOnce(id, acknowledgement)   
35   
36 when a valid read/executed reply or ColdDone   
37 with (id,result) arrives:   
38 pre: reply satisfies its identity/output checks   
39 completeOnce(id, result)  
Figure 8: A-GREEMENT client. Vote sets count each replica once; $h _ { E }$ binds the ordered batch contents and reply destinations, $h _ { c } = H ( c )$ , and $S = S _ { b } ( E )$ . The function completeOnce completes each original request at most once.

For an open version, recovery first selects the highest reported accepted round $( b _ { 0 } , e _ { 0 } , a )$ , ordered lexicographically. Classic rounds retain their unique accepted value. Fast rounds using $Q _ { b _ { 0 } }$ retain x if

$$
\forall p \in P \cap Q _ { b _ { 0 } } : \quad p { \mathrm { ~ r e p o r t s ~ a c c e p t a n c e ~ o f ~ } } x { \mathrm { ~ a t ~ } } ( b _ { 0 } , e _ { 0 } , a ) .
$$

This condition identifies a possible old fast choice, not a new certificate. Because $| P | = | Q _ { b _ { 0 } } | = m$ and $| R | = 2 f + 1$ , the intersection is nonempty. Any value previously accepted by all of $Q _ { b _ { 0 } }$ must appear at every member of this intersection, provided each promise is complete. A missing acceptance in a complete snapshot rules out that candidate; a missing snapshot or body does not. Recovery uses the identities of intersecting replicas, not just a count of matching reports. If no value could have been fast-chosen, the coordinator deterministically selects a retained original input and leaves others pending. It records recovered prefixes, selected tails, and body references in a manifest that remains immutable for genera

tion $G ^ { \prime } .$ . Missing bodies must be fetched from original holders;   
their absence cannot justify discarding a possible choice.

Recovered state and its certificate. The frozen manifest $M = ( G ^ { \prime } , Q _ { b ^ { \prime } } , \ldots )$ records recovered prefixes, chosen identifiers, and selected classic tails. Together, these form the foundation state to be transferred into the ordinary log. The coordinator broadcasts M and $H ( M )$ in BULKACCEPT. Each BULKVOTE requires a matching promise, local retention of every referenced body, installation of valid chosen-prefix evidence, and acceptance of every selected tail. Matching voters $V _ { M }$ must satisfy $| V _ { M } | \ge m$ and $Q _ { b ^ { \prime } } \subseteq V _ { M } ; Q _ { b ^ { \prime } }$ is selected from complete promise respondents. BULKCOMMIT certifies the tails for per-key application. Replicas acknowledge application with BULKAPPLIED. All of $\boldsymbol { Q } _ { b ^ { \prime } }$ and the coordinator must finish application; READY replies from all of $\boldsymbol { Q } _ { b ^ { \prime } }$ confirm installation before the log transition is proposed. Bulk recovery shares evidence transfer across keys.

One-way handoff. Handoff permanently moves cold keys to the ordinary log. To bind this transition to the recovered foundation, the coordinator proposes a bridge entry

$$
F = { \tt B R I D G E } ( M , H ( M ) , V _ { M } ) .
$$

The bridge has an internal identifier and no client response. Ordinary Paxos phase one and choice preserve its exact metadata. Accepting it votes for the foundation entry without reaccepting referenced user writes. Bulk voters retain the bound bodies, with a survivor after at most f crashes.

A replica cannot execute past F until it validates the foundation and obtains all hash-checked bodies. Executing F installs recovered cold state, imports chosen identifiers into ordinary deduplication, and atomically marks the replica migrated. Installation occurs once, preventing replay from overwriting later writes with an old foundation. Local bridge acceptance, mode hints, and body receipt cannot authorize migration.

After migration, all keys use the ordinary log in strict-prefix mode: reads wait for their relevant slot to execute, and write replies require actual execution. Direct acceptance certificates and cold admission stay disabled. Stale cold requests retain their identities when rerouted to the log. Stale control and proof messages cannot alter migrated state. Historical bodies support reconstruction but grant no authority.

Further failures. Later leaders run ordinary phase one, preserve selected bridges, and resume prefix replay and body fetching without starting a separate cold history. A locally accepted but unselected bridge awaits phase-one resolution; acceptance alone cannot establish handoff. After migration, failures use ordinary-log and roster recovery.

## A.4 Client and Replica Failure Details

Client failures. Both ordering mechanisms retain full commands at replicas before voting. If a client fails before disseminating a proof, phase one preserves possibly chosen values from acceptance records; proof absence cannot rule out a choice.

Replica failures. Ordinary writes can continue after nonmandatory replica failures if a majority and all required re sponders survive. Losing an ordinary read responder or a member of $Q _ { b }$ requires a higher roster ballot, even with a live leader. Old leases are fenced and replacement responders catch up before activation. The cold plane uses the recovery and handoff in § A.3; its fast quorum stays fixed throughout ballot b, including changes of key round.

Payload and response recovery. Replicas fetch hashbound bodies and predecessor proofs from original holders. Transfers are idempotent and confer no vote or read authority. COLDDONE requires the original command’s chosen evidence and, after migration, ordinary prefix execution. Retries retain original identifiers and complete at most once. Eligible writes may complete through certificates; executed replies require prefix execution at the planned completion replica.

## A.5 Completion and Read-Execution Details

Early completion. Ordinary certificates shorten evidence delivery to clients; the leader still orders commands. The cold path also removes this ordering hop for enabled, uncontended per-key instances. Both certificate paths require recoverable acceptance certificates before write completion. Write acknowledgements need no speculative execution of unchosen commands. Read results require valid read visibility.

Read execution. Normal-mode reads beyond the applied prefix use chosen key values under the freshness, fixed-prefix, and lease conditions in § A.2, without speculative suffix execution. Broader certificate eligibility can hold more reads behind accepted writes that may already have completed elsewhere. Read races preserve visibility checks, so placement accounts for write-evidence delivery and read waiting.

The handoff boundary. Bridge acceptance disables early client votes; ordered application installs the foundation and permanently enables strict-prefix execution. Reads cannot pass an unapplied bridge, and later writes cannot combine cold votes with ordinary acceptances. Later operations follow one recovered foundation and one ordinary executed prefix.

## A.6 Routing and Delivery Optimizations

Ingress and direct admission. The designated reply replica is implemented as a logical ingress. The planner chooses this ordinary write ingress using the modeled end-to-end write cost, rather than simply choosing the nearest replica. A desig nated completion owner, when usable, becomes the ingress for its key. The client can also submit the same write directly to the leader while retaining that ingress identity. Duplicate submissions are coalesced. A leader’s executed-result notification does not let the ingress acknowledge a write it has not itself executed. These routes change message arrival times, not certificate conditions.

Selective acceptance delivery. Accepting replicas send ordinary votes to the leader and the learners selected to complete writes; eligible writes additionally generate client votes. Restricting recipients reduces traffic. It does not reduce the number or identity requirements of the votes a recipient must collect. Commit notices and repair messages let other replicas learn and execute the chosen history. Recipient selection is an optimization; delaying publication of a completion gate’s own vote until it has locally established coverage is a protocol rule.

Read races. A client may send the same read to more than one authorized responder and use the first valid reply. Each responder independently checks its lease, catch-up state, and visibility predicate. Later replies do not create another logical completion. Racing requests cannot make an unsafe value visible.

Streams and body transfers. Protocol messages, including recovery traffic, share one protocol stream per peer pair. Shared control/RTT RPCs are separate from these protocol streams. Fragments, retransmissions, and fetched bodies remain bound to their identifiers and hashes; receiving them alone neither chooses a value nor restores read authority.

## A.7 Detailed Workload-Aware Planning

Inputs and decisions. Let K be the key domain. From an independent profiling sample of N offered requests, the system records

$$
N ( k ) = N ^ { R } ( k ) + N ^ { W } ( k )
$$

accesses to each observed key k, where $N ^ { R } ( k )$ and $N ^ { W } ( k )$ are its sampled read and write counts. Observed keys are ordered by decreasing $N ( k )$ , with key identifier used only to break ties. If $k _ { j }$ is the key at popularity rank $j ,$ its empirical probability is

$$
p _ { j } = \frac { N ( k _ { j } ) } { N } .
$$

The runtime retains the inverse mapping $\rho ( k _ { j } ) = j$ . An unobserved key has no rank and always uses the ordinary log. Its direct-certificate eligibility depends on the selected reference or all-key policy.

Measured boundaries. Let $U = | \{ k : N ( k ) > 0 \} |$ be the number of observed keys and ${ \overline { { N } } } = N / U$ their mean sampled count. The hot boundary is

$$
\mathfrak { \eta } = | \{ k : N ( k ) > \overline { { N } } \} | .
$$

Thus ranks $0 , \ldots , \boldsymbol { \mathfrak { n - 1 } }$ are hot. Integer implementation compares a key’s count with $\lfloor N / U \rfloor$ , which is equivalent to the strict comparison above. Equal-frequency keys remain on the same side of the boundary. A uniform sample has $\eta = 0$ . This is a fixed empirical classification rule, not an optimality claim or an online estimate that changes during execution.

The cold-path search starts with candidate cutoff η. It computes cumulative request mass over the remaining observed ranks and adds boundaries at the first complete frequency group crossing each of the fractions $1 / 7 , \ldots , 6 / 7$ of that mass. $\mathbf { A }$ boundary at the end of all observed keys is omitted, and coincident boundaries are merged. There are at most seven candidates. The selected cutoff $\tau \geq \mathsf { \mathfrak { n } }$ may leave a non-hot ordinary region $\boldsymbol { \mathfrak { \eta } } \leq \mathfrak { p } ( { \boldsymbol { k } } ) < \boldsymbol { \tau } .$ If the cold path is disabled, every key uses the ordinary log. Neither boundary depends on a numeric key identifier. The heat profile is hash-pinned, loaded before planning, and immutable across recovery; original keys, rather than their ranks, remain in commands.

For $N _ { c }$ clients, placement additionally uses directed delays $d _ { x y } ,$ , request rates, and write fraction w. A plan π selects leader ℓ, read responders S containing ℓ, client read responders $r _ { i } \in S ,$ write ingresses $h _ { i } ,$ completion owners, certificate eligibility, and an optional cold quorum Q. A rank cutoff τ assigns observed keys with $\rho ( k ) \geq \tau$ to the cold path.

The model groups popularity ranks into cohorts g, with probability mass $P _ { g }$ and effective write rate $\lambda _ { g } .$

$$
P _ { g } = \sum _ { j \in g } p _ { j } , \qquad \lambda _ { g } = \lambda \frac { \sum _ { j \in g } p _ { j } ^ { 2 } } { P _ { g } } ,
$$

where λ is the aggregate configured write rate per millisecond. This grouping approximates the per-key distribution. Ordinary candidate search uses configured client weights; the final combined screen averages over clients uniformly.

Ordinary write costs. Let $z _ { \nu } ( x ) = d _ { \ell \nu } + d _ { \nu x }$ be the time for replica $\nu \mathbf { \bar { s } }$ acceptance to reach recipient x after leader propagation. If or $\mathrm { i } _ { m }$ denotes the mth smallest arrival time, the required evidence reaches x at

$$
\begin{array} { c l c r } { { } } & { { \displaystyle E _ { \pi } ( x ) = \operatorname* { m a x } \big \{ \operatorname { o r d } _ { m } \{ z _ { \nu } ( x ) : \nu \in R \} , \ : } }  \\ { { } } & { { \displaystyle \operatorname* { m a x } _ { s \in S } z _ { s } ( x ) \big \} . } } \end{array}\tag{2}
$$

The two terms capture majority acceptance and coverage of every authorized read responder. A nearby responder can lower read delay while raising write-evidence time.

For client i, the model accounts for forwarding and direct submission using

$$
a _ { i } = \operatorname* { m i n } \{ d _ { i h _ { i } } + d _ { h _ { i } \ell } , \ : d _ { i \ell } + 1 \ : \mathrm { m s } \} .
$$

Certificate and ingress-return costs are

$$
T _ { i } ^ { C } = a _ { i } + E _ { \pi } ( i ) + 2 \mathrm { m s } ,\tag{3}
$$

$$
T _ { i } ^ { H } = \operatorname* { m a x } \{ d _ { i h _ { i } } , a _ { i } + E _ { \pi } ( h _ { i } ) \} + d _ { h _ { i } i } .\tag{4}
$$

An eligible command has modeled cost min $\{ T _ { i } ^ { C } , T _ { i } ^ { H } \}$ ; other commands use $T _ { i } ^ { H }$ . A cohort may designate a completion owner, the replica returning its executed replies, in place of the default ingress. Ordinary candidate search also adds 1 ms to read-network costs and selected write costs. These are fixed heuristic offsets, not measured processing delays or required runtime timers. In particular, the direct-admission implementation sends immediately; the model’s 1 ms term is not an enforced delay.

Read waiting. The interval between acceptance and evidence arrival at responder r is approximated by

$$
H _ { r } = [ E _ { \pi } ( r ) - d _ { \ell r } ] _ { + } , \qquad [ x ] _ { + } = \operatorname* { m a x } \{ x , 0 \} .
$$

Let $N _ { i r } = d _ { i r } + d _ { r i }$ and $\lambda _ { g r } ^ { f } = \lambda _ { g } s _ { g r }$ , where $s _ { g r }$ scales the rate of writes imposing read fences. It is one for certificate-eligible cohorts. Otherwise, it is zero at a designated completion owner, reflecting entries for which that owner gates publication of its vote; it is the estimated remote-write fraction at a leader without such an owner, and one at other responders. With 1 denoting an indicator, the exponential approximation gives the read cumulative distribution function (CDF):

$$
\begin{array} { r } { F _ { i g r } ^ { R } ( t ) = 1 _ { t \geq N _ { i r } } \exp \bigl ( - \lambda _ { g r } ^ { f } [ H _ { r } + N _ { i r } - t ] _ { + } \bigr ) . } \end{array}\tag{5}
$$

The extra waiting time $W _ { g r }$ has expectation

$$
\mathbb { E } [ W _ { g r } ] = H _ { r } - \frac { 1 - e ^ { - \lambda _ { g r } ^ { f } H _ { r } } } { \lambda _ { g r } ^ { f } } ,
$$

with value zero when $\lambda _ { g r } ^ { f } = 0 .$ . This residual-wait expression is used in preliminary mean estimates. Broader certificate eligibility can lower write latency while increasing read waiting. For a planned race between responders r and $s ,$ the combined screen uses $F _ { r } ( t ) + F _ { s } ( t ) - F _ { r s } ( t )$ , where $F _ { r s }$ accounts for overlapping fencing intervals caused by the same writes.

Cold-path costs. A cold write requires every member of $Q .$ so its uncontended cost for client i is

$$
T _ { i } ^ { F } = \operatorname* { m a x } _ { q \in \mathcal { Q } } ( d _ { i q } + d _ { q i } ) + 2 \operatorname* { m s } .
$$

The conflict approximation uses V<sub>Q</sub> = max<sub>i</sub> $T _ { i } ^ { F } + 2 0 0$ ms and $\rho _ { g } = \operatorname* { m i n } \{ 1 , \lambda _ { g } V _ { Q } \}$ . Let A be the majority leader–replica RTT, B the largest leader–Q RTT, and D the largest client– replica RTT. The additional recovery-cost proxy is $\Delta _ { Q } =$ $2 A + 2 \operatorname* { m a x } \{ A , B \} + B + D + 2 0$ ms. The cold-write CDF is

$$
F _ { i g } ^ { W , \mathrm { c o l d } } ( t ) = ( 1 - \mathfrak { p } _ { g } ) 1 _ { t \geq T _ { i } ^ { F } } + \mathfrak { p } _ { g } 1 _ { t \geq T _ { i } ^ { F } + \Delta _ { Q } } .\tag{6}
$$

Cold reads select a nearby authorized member of $Q .$ Their waiting model replaces the fixed interval $H _ { r }$ in Equation 5 with a write-duration distribution $B _ { g } \colon$ each of the $N _ { c }$ clients contributes durations $T _ { j } ^ { F }$ and $T _ { i } ^ { F } + \Delta _ { Q }$ with weights $( 1 -$ ${ \rho } _ { g } ) / N _ { c }$ and ${ \rho } _ { g } / N _ { c }$ . The exponent becomes $- \lambda _ { g } \mathbb { E } [ ( B _ { g } + N _ { i r } -$ $t ) _ { + } ]$ , capturing both uncontended and recovering writes.

Joint objective and search. Let $F _ { i g , \pi } ^ { R }$ and $F _ { i g , \pi } ^ { W }$ be the modeled CDFs for cohort $g .$ The final screen combines cohort masses and uniform client weights:

$$
F _ { \pi } ( t ) = \sum _ { g } \frac { P _ { g } } { N _ { c } } \sum _ { i } \left[ ( 1 - w ) F _ { i g , \pi } ^ { R } ( t ) + w F _ { i g , \pi } ^ { W } ( t ) \right] .\tag{7}
$$

The implemented midpoint sum with $h = 0 . 2 5$ ms estimates the truncated mean

$$
\mathbb { E } [ \operatorname* { m i n } \{ T , 1 0 0 0 \mathrm { m s } \} ] .
$$

Quantiles are defined by the CDF and estimated with 35 bisection iterations on [0, 1000] ms:

$$
{ \widehat { \mu } } _ { \pi } = h \sum _ { j = 0 } ^ { 3 9 9 9 } [ 1 - F _ { \pi } ( ( j + { \textstyle \frac { 1 } { 2 } } ) h ) ] ,\tag{8}
$$

$$
q _ { \pi } ( u ) = \operatorname* { i n f } \{ t : F _ { \pi } ( t ) \geq u \} .\tag{9}
$$

This mixture evaluates overlapping read/write effects together; individual path savings are not added separately.

For memberships up to 13 replicas, the ordinary search considers leader/responder combinations, chooses a nearby responder for each client, and compares ingress and completion assignments. An alternative stage selects a best-mean candidate allowing certificates for all ordinary keys. Above 13 replicas, this all-key exploration is skipped; the inherited planner uses a bounded topology-derived base search and subsequent refinement. These are implementation search restrictions, not limits on the quorum rules. The cold stage ranks size-m cold quorums by a workload-weighted combination of cold-read and uncontended-write cost, retains up to 64 candidates, and searches the measured cutoff candidates de fined above. This stage screens a coarse mean-gain proxy and nearby tail quantiles. Its tail search uses a 10,000 ms upper bound, distinct from the 1000 ms final comparison below. It prefers modeled 0.99-quantile improvement, breaking ties by mean gain; it can also disable the cold plane.

Candidate $\pi _ { 1 }$ is compared with reference plan $\pi _ { 0 }$ . The reference plan permits direct ordinary certificates only for observed keys with $\rho ( k ) \geq \eta$ , whereas $\pi _ { 1 }$ permits them for all ordinary-path writes. Keys absent from the empirical profile remain on the ordinary path and use the conservative reference behavior unless the all-key policy is selected. Define $\Delta \mu =$ $\widehat { \mu } _ { \pi _ { 0 } } - \widehat { \mu } _ { \pi _ { 1 } }$ and $\Delta q ( u ) = q _ { \pi _ { 0 } } ( u ) - q _ { \pi _ { 1 } } ( u )$ . When the alternative enables all-key certificates, select it when

$$
\Delta \mu > \varepsilon \quad \wedge \quad \operatorname* { m i n } _ { u \in \mathcal { U } } \Delta q ( u ) \geq - \gamma \Delta \mu ,\tag{10}
$$

where the current heuristic uses $\pmb { \mathrm { \varepsilon } } = 0 . 0 5$ ms, $\gamma = 5 ,$ and $\mathcal { U } =$ 0.9898,0.9900,0.9902 . The selected plan is fixed before normal operation, and its roles become active through the lease, promise, and catch-up transitions in $\ S 4 . 4$

Scope of the model. The numerical grid, truncation bounds, and fixed offsets above are performance heuristics. They do not grant votes, leases, or read authority, and the selection test does not guarantee an improvement in measured mean or tail latency. The rank aggregation and moment-expansion boundaries in the implementation control numerical approximation independently of hot membership and candidate cold cutoffs. The profile counts both reads and writes; the model uses a configured write fraction to estimate per-key write pressure rather than inferring a separate write fraction for every key. A workload change can therefore make a fixed plan inefficient without changing its safety conditions.

1 when Request(c) arrives at prepared leader L\_b:   
2 pre: active installed roster and ordinary route   
3 append exact c to next ordered batch E   
4 perform local acceptance handler for E   
5 send Accept(E) to the other replicas   
6   
when Accept(E) arrives from installed L\_b:   
pre: matching ballot and immutable slot value   
9 retain full E before acknowledging   
10 update pending-key indexes and read fences   
11 record self in V[b,s]   
12 update local contiguous acceptedPrefix   
13 if g(E) != self:   
14 publishVotesOnce(E)   
15 tryChoose(E)   
16   
17 when Accepted(b,s,prefix) arrives from replica p:   
18 pre: p in R, matching installed ballot/slot   
19 V[b,s] := V[b,s] union {p}   
20 retain p’s highest reported accepted prefix   
21 majorityFixedPrefix := mth largest repor   
22 tryChoose(E)   
23   
24 function tryChoose(E):   
25 if full local E exists and |V[b,s]| >= m   
26 and S\_b(E) subset V[b,s]   
27 and not coveredChosen(E):   
28 mark E covered-chosen; advance read targets   
29 execute contiguous covered-chosen prefix once   
30 deliver executed replies from completion replicas   
31 if g(E) = self:   
32 publishVotesOnce(E)   
33 if self = L\_b:   
34 send Commit(b,s) to the other replicas   
35 reconsider held reads   
36   
37 when Commit(b,s) arrives from installed L\_b:   
38 if local E at s has ballot b:   
39 mark E covered-chosen; advance read targets   
40 execute contiguous covered-chosen prefix once   
41 reconsider held reads   
42   
43 function publishVotesOnce(E):   
44 send Accepted(b,s,acceptedPrefix)   
45 to replicas learning the entry   
46 if fresh(E) and not migrated   
47 and no Bridge entry is present:   
48 for each certificate-eligible WRITE c in E:   
49 send ClientVote(b,s,id(c),batchDigest(E),   
50 H(c),S\_b(E)) to client(c)   
51   
52 when READ(k) arrives at ordinary responder p:   
53 pre: p in S\_b(k), valid majority leases,   
54 caught up through grantor thresholds   
55 t := target t\_p(k) defined above   
56 if t <= executedPrefix:   
57 reply from applied state   
58 else if coveredChosen(t) and fresh(t)   
59 and ballot(t)=b   
60 and t <= majorityFixedPrefix   
61 and not strictPrefix   
62 and no unapplied Bridge at or before t:   
63 reply from chosen key value   
64 else:   
65 hold request until safe or forward it

1 when ColdPropose(c) arrives at q in Q\_b:   
2 pre: installed generation (b,e), fast key round a,   
3 matching generation hint, known predecessor,   
4 no freeze or handoff   
5 if exact id/body is already chosen:   
6 send ColdDone to client(c)   
7 else if next version v is unvoted in (b,e,a):   
8 retain full original c   
9 invalidate key k; accept (b,e,a,k,v,c,Q\_b)   
10 send FastVote(b,e,a,k,v,id(c),H(c),   
11 origin(c),timestamp(c),Q\_b)   
12 to client(c)   
13 else if the same c was accepted in (b,e,a,v):   
14 resend its unchanged FastVote   
15 else if a different c competes:   
16 retain c; notify coordinator of collision   
17   
18 when a valid ChosenProof(x,Q\_b,c) arrives:   
19 pre: all Q\_b votes match x   
20 retain hash-checked body   
21 if migrated or a Bridge is present: return   
22 retain proof and chosen identifier   
23 if chosen predecessor of x is installed:   
24 apply next per-key version once   
25 recompute key readiness; release safe reads   
26 else:   
27 buffer proof and fetch missing predecessor   
28   
29 when READ(k) arrives at cold member q:   
30 pre: q in installed Q\_b, matching installed/promised   
generation,   
31 stable lease, fully caught-up key,   
32 no freeze or handoff   
33 if keyReady(k) and no unresolved accepted tail:   
34 reply from chosen per-key state   
35 else:   
36 hold or forward to authorized Q\_b member   
37   
38 when collision recovery prepares key round a’:   
39 promise a’ across remaining versions   
40 collect m complete key snapshots   
41 preserve chosen prefix; select safe tail   
42 by the highest (b,e,a) and intersection rule   
43 propose one classic value at each next version   
44 choose only with matching votes from all Q\_b   
45 apply in per-key order; replay losing inputs   
46 before fast resumption: higher prepare,   
47 safe empty tail, catch-up and all-Q\_b Ready  
Figure 9: Ordinary (left) and cold (right) handlers. Votes require full bodies; reads need leases, catch-up, and visibility checks.

```csv
function recover(b’) at new leader:
2 choose b’ above observed ballots
3 fence old leases and install new roster
keep reads disabled until grants/catch-up
5 and required preparation complete
6 send Prepare(b’) to R
7
8 when complete promises arrive from majority P:
9 for each ordinary suffix slot s:
10 adopt highest-ballot accepted entry in P
11 use internal no-op only for unreported gap
12 mark adopted entry non-fresh
13 re-accept suffix with ordinary quorum rules
14 if migrated or suffix has a selected Bridge:
15 continue its prefix replay and body fetch
16 else if cold plane is enabled:
17 start frozen recovery at generation G’=(b’,e’)
18
19 when ColdPrepare(G’) arrives:
20 promise G’ over every key and future version
21 disable old cold read and fast authority
22 retain old bodies, votes, choices, and IDs
23 send complete immutable PromiseCatalog
24
25 when complete PromiseCatalogs arrive from majority P:
26 Q_b’ := majority selected from actual responders
27 keep highest valid chosen prefixes and IDs
28 for each open tail:
29 (b0,e0,a) := highest reported accepted round
30 if that round is classic: retain its value
31 else if every p in P intersect Q_b0 reports
32 the same x accepted at (b0,e0,a): retain x
33 else: deterministically select retained input
34 fetch and verify every referenced body
35 freeze one immutable manifest M=(G’,Q_b’,...)
36 send BulkAccept(M,H(M)) to R
37
38 when complete BulkAccept(M,h) arrives:
39 pre: current promise/leader, same M, h=H(M),
40 not migrated and no Bridge present
41 fetch every missing referenced full body
42 only after all bodies are locally retained:
43 install valid old chosen-prefix/ID evidence
44 invalidate and accept each selected tail
45 send BulkVote(G’,Q_b’,h)
```

![](images/c670b58e0742d9f324712045f324d90489dd39eda85a78342928a7de851179d5.jpg)  
Figure 10: One-way handoff. Votes require full bodies and acceptance; bridge acceptance does not authorize reads.

## B Prompt Used for Consensus Discovery by the General-purpose Agent

## Prompt

Run a 20-round iterative protocol search to discover, implement, and verify a new state-machine replication protocol with lower mean and p99 latency.

Assume a non-Byzantine crash-failure model with $\mathrm { n } = 2 \mathrm { f } + 1$ replicas, tolerating any f replica failures.

Promote a protocol version only if one latency metric improves by at least 1% while the other degrades by no more than 0.2% compared with the current best promoted version.

## C Qualitative Analysis on Latency

<table><tr><td>Protocol</td><td>Leader</td><td>Path</td><td>W</td><td>Write quorum</td><td>R</td><td>Read quorum</td></tr><tr><td>PAXOS</td><td>Yes</td><td>Normal</td><td>4δ</td><td> $f + 1$ </td><td>4δ</td><td>f+1</td></tr><tr><td>FASTPAXOS</td><td>Fallback only</td><td>Fast</td><td> $\ge 3 8$ </td><td> $> 3 n / 4$ </td><td> $\ge 3 8$ </td><td> $> 3 n / 4$ </td></tr><tr><td></td><td></td><td>Slow</td><td>≥4δ</td><td>f+1</td><td>≥4δ</td><td>f +1</td></tr><tr><td>EPAXOS</td><td>Per command</td><td>Fast</td><td>≥4δ</td><td> $f + \lfloor ( f + 1 ) / 2 \rfloor$ </td><td>≥4δ</td><td> $f + \lfloor ( f + 1 ) / 2 \rfloor$ </td></tr><tr><td></td><td></td><td>Slow</td><td> $\geq 6 8$ </td><td>f+1</td><td>≥6δ</td><td>f+1</td></tr><tr><td>CURP</td><td>Yes</td><td>Fast</td><td> $\geq 2 8$ </td><td> $f + \lceil f / 2 \rceil + 1$ </td><td>≥2δ</td><td> $f + \lceil f / 2 \rceil + 1$ </td></tr><tr><td></td><td></td><td>Slow</td><td> $\geq 4 8$ </td><td>f +1</td><td>≥4δ</td><td>f+1</td></tr><tr><td>SWIFTPAXOS</td><td>Yes</td><td>Fast</td><td>2δ</td><td>Fixed  $f + 1 \mathrm { o r a n y } > 3 n / 4$ </td><td>2δ</td><td>Fixed f + 1 or any &gt; 3n/4</td></tr><tr><td></td><td></td><td>Slow</td><td>3δ</td><td>f + 1 or corrected fast quorum</td><td>3δ</td><td>f + 1 or corrected fast quorum</td></tr><tr><td>KCENSUS</td><td>Fallback only</td><td>Fast</td><td>≥3δ</td><td> $f + 1 \sim 2 f + 1$ </td><td>≥4δ</td><td>f+1</td></tr><tr><td></td><td></td><td>Slow</td><td>≥5δ</td><td>f+1</td><td></td><td></td></tr><tr><td>BODEGA</td><td>Yes</td><td>Normal</td><td> $\geq 4 8$ </td><td> $\operatorname { A n y } f + 1 \cup \operatorname { a l l } S$ </td><td>≥2δ</td><td>1</td></tr><tr><td></td><td></td><td>Cold fast</td><td>2δ</td><td>Fixed  $f + 1$ </td><td></td><td></td></tr><tr><td>A-GREEMENT (ours)</td><td>Yes</td><td>Cold slow</td><td>4δ</td><td></td><td>≥2δ</td><td>1</td></tr><tr><td></td><td></td><td>Ordinary cert</td><td> $\ge 3 8$ </td><td> $\operatorname { A n y } f + 1 \cup \operatorname { a l l } S$ </td><td></td><td></td></tr><tr><td></td><td></td><td>Ordinary exec</td><td> $\geq 4 8$ </td><td></td><td></td><td></td></tr></table>

Table 1: Qualitative comparison of latency and quorum requirements. W and R denote write and read latency, respectively; δ is a one-way message delay, $n = 2 f + 1$ , and S is the set of authorized read responders. “Cert” and “exec” denote certificate-based completion and completion after execution, respectively.

## D Quantitative Results Extended

<table><tr><td rowspan="2">Replicas</td><td rowspan="2">Clients</td><td rowspan="2">Topology</td><td rowspan="2">Workload</td><td colspan="2">SWIFTPAXOS</td><td colspan="2">BODEGA</td><td colspan="2">MULTI-PAXOS</td><td colspan="2">KCENSUS</td><td colspan="2">A-GREEMENT (ours)</td><td colspan="2">Lower bound</td></tr><tr><td>mean</td><td>p99</td><td>mean</td><td>p99</td><td>mean</td><td>p99</td><td>mean</td><td>p99</td><td>mean</td><td>p99</td><td>mean</td><td>p99</td></tr><tr><td>5</td><td>5</td><td>clustered</td><td>A</td><td>80.955</td><td>116.199</td><td>121.120</td><td>223.832</td><td>113.834</td><td>151.313</td><td>94.267</td><td>180.063</td><td>66.412</td><td>139.075</td><td>44.250</td><td>101.500</td></tr><tr><td>5</td><td>5</td><td>clustered</td><td>B</td><td>80.960</td><td>115.654</td><td>54.306</td><td>217.560</td><td>113.693</td><td>151.195</td><td>97.384</td><td>119.678</td><td>24.687</td><td>134.241</td><td>19.905</td><td>98.000</td></tr><tr><td>5</td><td>5</td><td>clustered</td><td>C</td><td>80.908</td><td>115.247</td><td>19.602</td><td>28.499</td><td>113.637</td><td>151.255</td><td>97.763</td><td>119.604</td><td>19.598</td><td>28.490</td><td>17.200</td><td>25.000</td></tr><tr><td>5</td><td>5</td><td>uniform</td><td>A</td><td>168.977</td><td>244.626</td><td>233.824</td><td>352.695</td><td>214.035</td><td>264.545</td><td>230.193</td><td>403.652</td><td>166.515</td><td>263.005</td><td>118.700</td><td>176.500</td></tr><tr><td>5</td><td>5</td><td>uniform</td><td>B</td><td>168.956</td><td>244.639</td><td>144.595</td><td>346.727</td><td>214.058</td><td>264.543</td><td>225.345</td><td>384.629</td><td>122.017</td><td>238.409</td><td>99.710</td><td>170.500</td></tr><tr><td>5</td><td>5</td><td>uniform</td><td>C</td><td>168.789</td><td>244.591</td><td>99.664</td><td>173.462</td><td>213.860</td><td>264.618</td><td>220.368</td><td>314.027</td><td>99.672</td><td>173.441</td><td>97.600</td><td>170.500</td></tr><tr><td>5</td><td>5</td><td>colocated</td><td>A</td><td>153.695</td><td>208.565</td><td>161.040</td><td>332.649</td><td>229.564</td><td>305.973</td><td>150.883</td><td>371.423</td><td>117.837</td><td>263.490</td><td>52.750</td><td>136.000</td></tr><tr><td>5</td><td>5</td><td>colocated</td><td>B</td><td>153.724</td><td>208.115</td><td>55.159</td><td>298.163</td><td>228.324</td><td>306.047</td><td>142.547</td><td>196.971</td><td>13.598</td><td>236.153</td><td>5.275</td><td>120.000</td></tr><tr><td>5</td><td>5</td><td>colocated</td><td>C</td><td>153.808</td><td>208.484</td><td>1.172</td><td>1.941</td><td>228.488</td><td>304.936</td><td>142.128</td><td>172.418</td><td>1.175</td><td>1.944</td><td>0.000</td><td>0.000</td></tr><tr><td>9</td><td>9</td><td>clustered</td><td>A</td><td>79.813</td><td>103.833</td><td>87.442</td><td>161.186</td><td>117.727</td><td>154.410</td><td>124.762</td><td>229.154</td><td>70.493</td><td>124.253</td><td>34.694</td><td>62.500</td></tr><tr><td>9</td><td>9</td><td>clustered</td><td>B</td><td>79.883</td><td>104.306</td><td>46.822</td><td>132.945</td><td>117.225</td><td>153.806</td><td>121.158</td><td>155.653</td><td>43.330</td><td>101.677</td><td>21.219</td><td>54.000</td></tr><tr><td>9</td><td>9</td><td>clustered</td><td>C</td><td>79.775</td><td>103.899</td><td>39.072</td><td>63.176</td><td>116.788</td><td>153.267</td><td>117.940</td><td>153.957</td><td>38.934</td><td>63.322</td><td>19.722</td><td>30.500</td></tr><tr><td>9</td><td>9</td><td>uniform</td><td>A</td><td>122.627</td><td>185.610</td><td>174.834</td><td>324.223</td><td>162.447</td><td>233.553</td><td>155.954</td><td>290.973</td><td>105.987</td><td>174.604</td><td>63.000</td><td>139.000</td></tr><tr><td>9</td><td>9</td><td>uniform</td><td>B</td><td>122.701</td><td>185.536</td><td>84.364</td><td>299.766</td><td>162.025</td><td>233.145</td><td>151.243</td><td>209.582</td><td>55.114</td><td>178.455</td><td>36.250</td><td>118.500</td></tr><tr><td>9</td><td>9</td><td>uniform</td><td>C</td><td>122.856</td><td>185.277</td><td>43.658</td><td>68.721</td><td>162.114</td><td>233.574</td><td>148.907</td><td>200.658</td><td>43.601</td><td>68.897</td><td>33.278</td><td>52.500</td></tr><tr><td>9</td><td>9</td><td>colocated</td><td>A</td><td>84.417</td><td>123.636</td><td>91.433</td><td>182.198</td><td>112.918</td><td>158.605</td><td>97.413</td><td>217.760</td><td>78.660</td><td>153.258</td><td>29.861</td><td>92.500</td></tr><tr><td>9</td><td>9</td><td>colocated</td><td>B</td><td>84.388</td><td>123.739</td><td>30.395</td><td>157.426</td><td>112.048</td><td>158.044</td><td>79.084</td><td>172.399</td><td>18.657</td><td>126.553</td><td>2.986</td><td>78.500</td></tr><tr><td>9</td><td>9</td><td>colocated</td><td>C</td><td>84.265</td><td>123.641</td><td>0.993</td><td>1.760</td><td>111.907</td><td>157.370</td><td>74.374</td><td>108.523</td><td>0.995</td><td>1.769</td><td>0.000</td><td>0.000</td></tr><tr><td>13</td><td>13</td><td>clustered</td><td>A</td><td>150.335</td><td>181.058</td><td>235.239</td><td>431.051</td><td>205.402</td><td>290.612</td><td>194.503</td><td>405.152</td><td>133.820</td><td>257.339</td><td>73.788</td><td>163.500</td></tr><tr><td>13</td><td>13</td><td>clustered</td><td>B</td><td>150.315</td><td>181.092</td><td>102.102</td><td>417.248</td><td>205.353</td><td>290.481</td><td>166.134</td><td>318.501</td><td>50.181</td><td>241.444</td><td>22.575</td><td>151.000</td></tr><tr><td>13</td><td>13</td><td>clustered</td><td>C</td><td>150.335</td><td>181.062</td><td>19.146</td><td>38.409</td><td>205.528</td><td>290.659</td><td>155.950</td><td>186.375</td><td>19.158</td><td>38.444</td><td>16.885</td><td>35.000</td></tr><tr><td>13</td><td>13</td><td>uniform</td><td>A</td><td>151.033</td><td>200.002</td><td>189.139</td><td>345.329</td><td>204.076</td><td>273.261</td><td>188.260</td><td>350.724</td><td>121.836</td><td>224.449</td><td>86.865</td><td>177.500</td></tr><tr><td>13</td><td>13</td><td>uniform</td><td>B</td><td>151.037</td><td>199.996</td><td>92.750</td><td>307.940</td><td>201.136</td><td>259.295</td><td>165.960</td><td>281.061</td><td>60.988</td><td>201.785</td><td>48.563</td><td>163.500</td></tr><tr><td>13</td><td>13</td><td>uniform</td><td>C</td><td>151.091</td><td>199.981</td><td>46.742</td><td>83.545</td><td>201.257</td><td>259.212</td><td>159.612</td><td>203.161</td><td>46.717</td><td>83.492</td><td>44.308</td><td>81.000</td></tr><tr><td>13 13</td><td>13</td><td>colocated colocated</td><td>A</td><td>119.665</td><td>181.827</td><td>151.955</td><td>301.688</td><td>165.229</td></table>

Table 2: Normal-case latency results in milliseconds (mean and p99). The Lower bound is analytical and excluded from ranking. Within each row, bold marks the best result among protocols, separately for mean and p99 latency.

<table><tr><td>Workload</td><td>Command</td><td colspan="2">SWIFTPAXOS</td><td colspan="2">BODEGA</td><td colspan="2">MULTI-PAXOS</td><td colspan="2">KCENSUS</td><td colspan="2">A-GREEMENT (ours)</td><td colspan="2">Lower bound</td></tr><tr><td></td><td></td><td>mean</td><td>p99</td><td>mean</td><td>p99</td><td>mean</td><td>p99</td><td>mean</td><td>p99</td><td>mean</td><td>p99</td><td>mean</td><td>p99</td></tr><tr><td rowspan="3">A (50% writes)</td><td>Read</td><td>114.975</td><td>164.010</td><td>77.220</td><td>176.345</td><td>158.315</td><td>211.431</td><td>138.746</td><td>210.334</td><td>56.544</td><td>129.864</td><td>27.967</td><td>46.417</td></tr><tr><td>Write</td><td>115.185</td><td>163.568</td><td>212.630</td><td>265.170</td><td>158.526</td><td>211.380</td><td>145.781</td><td>297.498</td><td>145.359</td><td>188.035</td><td>86.452</td><td>118.000</td></tr><tr><td>All</td><td>115.081</td><td>163.745</td><td>144.949</td><td>262.797</td><td>158.421</td><td>211.400</td><td>142.245</td><td>282.171</td><td>100.984</td><td>186.281</td><td>57.209</td><td>118.000</td></tr><tr><td rowspan="3">B (5% writes)</td><td>Read</td><td>115.081</td><td>163.654</td><td>61.927</td><td>164.521</td><td>157.869</td><td>211.112</td><td>136.461</td><td>200.008</td><td>40.669</td><td>96.328</td><td>27.967</td><td>46.417</td></tr><tr><td>Write</td><td>115.481</td><td>163.833</td><td>207.254</td><td>262.435</td><td>158.382</td><td>211.527</td><td>129.774</td><td>268.959</td><td>150.453</td><td>187.524</td><td>86.452</td><td>118.000</td></tr><tr><td>All</td><td>115.102</td><td>163.665</td><td>69.273</td><td>242.098</td><td>157.895</td><td>211.130</td><td>136.127</td><td>206.485</td><td>46.234</td><td>169.248</td><td>30.891</td><td>106.583</td></tr><tr><td>C (read-only)</td><td>Read (= All)</td><td>115.067</td><td>163.523</td><td>34.027</td><td>56.260</td><td>157.799</td><td>210.837</td><td>133.580</td><td>178.198</td><td>33.996</td><td>56.310</td><td>27.967</td><td>46.417</td></tr></table>

Table 3: Operation-level normal-case latency in milliseconds, averaged with equal weight over the six deployments with 5 or 9 replicas. The Lower bound is analytical and excluded from ranking. Within each row, bold marks the best result among protocols, separately for mean and p99 latency.

## E Protocol Counterexamples

## E.1 Tentative Metadata Authorizes an Unsafe Pruning Decision

Audit question. Is the metadata used to justify pruning already final? EPaxos Revisited [75] uses sequence numbers and dependencies to prune commands that appear to execute after the current instance. A tentative PREACCEPTED tuple can differ from the tuple fixed by COMMIT, so using it to authorize a cutoff can expose an incorrect order.

![](images/0c84bee6193e57e5353c82653c13cf1fc222db617db1af6c58b986f482498034.jpg)  
Figure 11: Pruning with tentative metadata reverses conflicting command order: $r _ { 0 }$ executes X before Y, while $r _ { 2 }$ prunes X using its tentative tuple and executes Y before X.

Setup. Three replicas $r _ { 0 } , r _ { 1 } , r _ { 2 }$ run EPaxos with dependency pruning enabled. Conflicting commands X and Y originate at $r _ { 0 }$ and $r _ { 2 } ,$ respectively. No replica fails; selected PREACCEPT and COMMIT messages are delayed.

Counterexample. In Figure 11, command X first commits at $r _ { 0 }$ with no dependencies. Command Y subsequently commits with a dependency on $X ,$ so $r _ { 0 }$ executes X before Y. At $r _ { 2 } ,$ however, a delayed PREACCEPT for X arrives only after Y has committed. The tentative local tuple gives X a higher sequence number and a dependency on Y. If the pruning rule treats that tentative tuple as final evidence, it removes X from Y’s dependency chain and permits $r _ { 2 }$ to execute Y first. The later COMMIT for X overwrites the tentative tuple, but it arrives too late to undo the order already exposed at $r _ { 2 } .$ . Replica $r _ { 2 }$ subsequently executes X, reversing $r _ { 0 }$ ’s order and violating execution consistency.

## E.2 Pruning a Committed Node Can Hide an Earlier Dependency

Audit question. Does finality of the pruned node justify discarding its entire dependency branch in EPaxos Revisited [75]? Unlike § E.1, this counterexample already requires the pruned node to be committed.

Setup. Three replicas originate conflicting commands C, A, and B at owners r<sub>0</sub>, r<sub>1</sub>, and $r _ { 2 } .$ , respectively. Dependency pruning is enabled; no replica fails and no recovery occurs.

Counterexample. Normal-path processing and fast decisions produce the tuples shown in Figure 12. The full committed graph is one SCC. Commands C and A tie at sequence 1, and the replica-ID tie-breaker places C before A; B has sequence 2. Replica $r _ { 0 }$ selects B as its execution root and executes C,A,B. Replica $r _ { 1 }$ , however, learns the committed B tuple while executing A. The reciprocal dependency and strict sequence inequality allow it to prune B. The cutoff hides C, whose final tuple is not yet known locally. It executes A, later learns C, and executes C, reversing the order of the conflicting pair A,C and violating execution consistency despite the pruned node’s committed metadata.

![](images/02388614add7db6f1dab44c2e437080220b1983798118c99d7920744310a2b7e.jpg)  
Figure 12: Pruning committed B hides C, allowing $r _ { 1 }$ to execute A before C while $r _ { 0 }$ executes C before A.

## E.3 Early Reads Expose a Value Missing from the Leader’s Read View

One of the authors had identified this issue before the agent but the agent received no hint about it and found it independently.

Audit question. Does every read path respect a value already exposed by an early read, even before the leader commits that write? Bodega’s main algorithm [29, §3.2.2 and Fig. 7] permits a non-leader responder to return an accepted value after collecting majority accept notifications (lines 38–39), but directs a stable leader to return the last committed value (line 36).

![](images/8f32b5c1a615d1937ba55427174df6603426e3ec301fd495dbfd12a27660ae29.jpg)  
Figure 13: Counterexample to Bodega’s early-accept read optimization. All replicas accept $w ,$ but delayed acceptance replies prevent L from committing it. Accept notifications let A return 7, while a subsequent read at L returns absence.

Setup. Three replicas L,A, B tolerate one failure; the trace itself has no failures. The fixed ballot and roster designate L as leader and A as the responder for key k, initially absent. The roster is installed, the leader is prepared, and read-serving replicas hold valid majority-backed leases with catch-up thresholds zero. Early accept is enabled. No lease expiration or roster change occurs.

No failures; x initially 0; both attempts belong to one client write $W _ { 1 }$

![](images/e651e0dffe8dcd2e45b88c035bcdf3f42a85705603e41ae0e894320dd45404cc.jpg)  
C<sub>2</sub> reads 1 before writing 2; retrying W<sub>1</sub> must not overwrite that 2.  
Figure 14: A Bodega client retry can repeat a write that has already taken effect. The first attempt sets $x = 1$ , but its reply is delayed. After $C _ { 2 }$ reads 1 and completes a write of 2, a retry with a new identifier overwrites x with 1. If both attempts represent one client write, the final read must return 2.

Counterexample. As shown in Figure 13, Writer W invokes $w = \mathrm { W R I T E } ( k , 7 )$ , which L proposes at log slot zero after preparation. All three replicas accept w. The accept replies from A and B to L remain delayed, so the leader cannot commit the write. A collects accept notifications from the majority $\{ L , B \}$ . Reader R then invokes $r _ { 1 } = \mathbf { R E A D } ( k )$ . At A, the notification authorize an early reply with 7 while the write remains uncommitted locally. After receiving that result, R invokes $r _ { 2 } = \mathrm { R E A D } ( k )$ When this read reaches L, its stable-leader read path uses the committed state, which is still empty, and returns absence. The second read can reach the leader after the client unhold timeout while its response from A is delayed. The write remains pending throughout.

## E.4 Duplicate Writes from Client Retries

Audit question. Can retrying a timed-out write cause the same client operation to take effect twice? In Bodega [29], the client’s timeout retry path assigns a new request identifier to each write attempt. The server treats the original request and its retry as separate writes, so both can execute.

Setup. Two clients access a key x, initially 0, through a leader in a three-replica deployment. Client $C _ { 1 }$ invokes $W _ { 1 } = \mathbf { W } \mathbf { R I T E } ( x , 1 )$ with timeout retries enabled. The original request and its retries represent this single write and should take effect only once. The reply to the first attempt is delayed beyond the client timeout; all reads return committed values.

Counterexample. As shown in Figure 14, the first attempt of $W _ { 1 }$ , with request identifier u, sets $x = 1$ , but its reply is delayed. Client $C _ { 2 }$ reads x and obtains 1, then completes $W _ { 2 } = \mathrm { W R I T E } ( x , 2 )$ ). After timing out, $C _ { 1 }$ resubmits $\mathrm { W R I T E } ( x , 1 )$ with a new identifier $u ^ { \prime } \ne u .$ . The server executes the retry, overwriting 2 with 1, and its reply completes $C _ { 1 } \mathrm { { ' } s }$ write. A subsequent read by $C _ { 2 }$ returns 1. The first read requires $W _ { 1 }$ to take effect before $W _ { 2 } .$ . If both attempts belong to the same write $W _ { 1 }$ , there is no additional write after $W _ { 2 } .$ , so the final read must return 2. Returning 1 therefore violates linearizability for these client operations.

## F Comparison with AI-Driven Discovery Approaches

Table 4 compares AI-GREEMENT with prior AI-based work that generates algorithms or protocols automatically. AI-driven discovery has already produced new algorithms, but mostly for single-node code or for policies that a simulator can score. For distributed systems, prior AI work either synthesizes verified implementations of a fixed specification, finds bugs in existing protocols, learns small protocols checked only for a bounded number of processes, without a performance objective, or assists humans in designing protocols proved by hand.

<table><tr><td>Work</td><td>Search</td><td>Target</td><td>Correctness check</td><td>Objective</td><td>New algorithm or protocol?</td></tr><tr><td colspan="6">Algorithm and policy discovery</td></tr><tr><td>AlphaDev [47]</td><td>Deep RL LLM + evolution</td><td>Sorting, hashing (assembly)</td><td>Tests Evaluator</td><td>Latency Score</td><td>Yes: sorting routines merged into LLVM libc++ Yes: new cap-set constructions and packing</td></tr><tr><td>FunSearch [62]</td><td></td><td>Math, bin packing</td><td></td><td></td><td>heuristics</td></tr><tr><td>AlphaEvolve [54]</td><td>LLM + evolution</td><td>Math, kernels, scheduling</td><td>Evaluator</td><td>Score</td><td>Yes: new algorithms (e.g., matrix multiplication) and data-center heuristics</td></tr><tr><td>ADRS [14]</td><td>LLM + evolution</td><td>Systems policies</td><td>Simulator, tests</td><td>Cost, latency</td><td>Yes: single-node policies (scheduling, load balancing), consensus deemed a poor fit</td></tr><tr><td>Glia [24] SkyDiscover [42, 44]</td><td>Multi-agent LLM</td><td>LLM-serving routing, scaling 200+ tasks (math, systems)</td><td>Simulator Evaluator</td><td>Completion time Score</td><td>Yes: request-routing algorithm Yes: improved policies (e.g., cross-cloud</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>transfer, MoE placement), no distributed protocol</td></tr><tr><td colspan="6">Protocol synthesis, verification, and testing</td></tr><tr><td>IDS [2]</td><td>LLM agents</td><td>Verified key-value stores</td><td>Rocq proof of code</td><td>Throughput</td><td>No new mechanism: data layout and proofs for a fixed specification</td></tr><tr><td>GGMS [32]</td><td></td><td>MCTS + model checker (no LLM) Sync. consensus, atomic commit (n ≤ 4, crash) Bounded model checking None</td><td></td><td></td><td>Consensus: rediscovers FloodSet [46], one atomic-commit variant claimed new</td></tr><tr><td>Agora [45]</td><td>LLM agents</td><td>Raft, EPaxos, HotStuff, BullShark</td><td>Tests</td><td></td><td>No: finds 15 bugs in existing implementations</td></tr><tr><td>Efron et al. [19]</td><td>Human-led, LLM-assisted</td><td>Byzantine broadcast (sync., t &lt; n)</td><td>Pen-and-paper proof</td><td>Communication</td><td>Yes: communication-efficient Byzantine broadcast protocol</td></tr><tr><td>AI-GREEMENT (ours)</td><td>LLM agents</td><td>WAN consensus (crash)</td><td>Fault injection, Lean</td><td>Read/write latency</td><td>Yes: A-GREEMENT</td></tr></table>

Table 4: AI-driven discovery of algorithms and protocols. Before AI-GREEMENT, no automated system had produced a new fault-tolerant consensus protocol that improves measured performance.

To compare discovery procedures, we adopt the nomenclature of SkyDiscover [44], which decomposes AI-driven discovery into four components, and use it to describe all systems in Table 5, including AI-GREEMENT. A context builder assembles the prompt from the problem and from earlier solutions, a solution generator produces candidates, an evaluator scores them (by simulation, benchmarking, testing, or proof checking), and a solution selector decides which earlier solutions guide the next iteration. AI-GREEMENT follows the same loop. It differs in three ways: its context is a curated consensus knowledge base rather than prior programs alone, its evaluator stages benchmarks, fault injection, and Lean proofs so that cheaper checks reject candidates first, and its selector is the agent itself, which chooses a parent from recorded lessons rather than from a population by a fixed rule.

The selectors of FunSearch, AlphaEvolve, ADRS, and SkyDiscover come from evolutionary computation and operate on a population of candidates. In the island model, the population is split into subpopulations (islands) that evolve independently: each new candidate is built from parents of a single island, and islands occasionally exchange their best candidates or are reseeded from stronger ones, which keeps the search from converging prematurely on one line of solutions. MAP-Elites preserves diversity differently, by keeping the best candidate for each cell of a grid of solution features, such as program length or behavior. None of the other systems maintains such a population: AlphaDev and GGMS select by tree search, IDS by a coordinator, and Glia and AI-GREEMENT by the agent’s own judgment.

Three challenges set consensus apart from the problems that AI-driven discovery has addressed so far. (C1) Performance conflicts with correctness: the cheapest way to reduce latency is to shrink a quorum or skip a round, which breaks agreement, so a performance score alone steers the search toward incorrect protocols. (C2) Tests cannot establish correctness: safety violations arise in rare interleavings of messages, failures, and recovery that benchmarks seldom reach, so safety requires a proof, and a proof is only as good as its specification. (C3) Evaluation is expensive: each candidate requires an implementation, a geo-distributed benchmark, fault injection, and a proof attempt, which takes minutes to hours rather than seconds, so thousands of candidates cannot be evaluated. Table 6 shows which of these challenges prior work faces. In classical algorithms and in distributed-systems policies, every candidate is either valid by construction or checked exactly and cheaply, so none of the three arises. The few AI-driven approaches to distributed protocols sidestep them by dropping the performance objective, fixing the specification, or restricting the search to three or four processes.

<table><tr><td>Work</td><td>Context builder</td><td>Solution generator</td><td>Evaluator</td><td>Solution selector</td></tr><tr><td>AlphaDev [47]</td><td>Current assembly program and its CPU state</td><td>Policy network adding one instruction at a time</td><td>Correctness tests and measured latency</td><td>Monte Carlo tree search</td></tr><tr><td>FunSearch [62]</td><td>Prompt with the best programs sampled from one island</td><td>Pretrained code LLM</td><td>Runs the program on problem instances and scores it</td><td>Island-based evolutionary database</td></tr><tr><td>AlphaEvolve [54]</td><td>Prompt with prior programs, scores, and human-written context, with prompts that can</td><td>Ensemble of Gemini models emitting code diffs</td><td>User-supplied evaluation function, applied as a cascade of tests</td><td>Program database combining MAP-Elites and islands</td></tr><tr><td>ADRS [14] (OpenEvolve)</td><td>themselves evolve Fixed prompt template with problem statement and prior solutions</td><td>Ensemble of LLMs editing code</td><td>Simulator replaying workload traces</td><td>MAP-Elites and islands</td></tr><tr><td>Glia [24]</td><td>Task description and shell access to the codebase</td><td>Researcher LLM agent, guided by a Supervisor agent</td><td>Simulator (request completion time)</td><td>Researcher&#x27;s own analysis, then best of independent runs</td></tr><tr><td>SkyDiscover [42, 44]</td><td>Problem context with feedback and reflections from prior solutions</td><td>LLM calls with optional tool use</td><td>User-supplied scoring function</td><td>Pluggable (Top-K, MAP-Elites, Pareto), adaptive (AdaEvolve), or evolved (EvoX)</td></tr><tr><td>IDS [2]</td><td>Fixed specification, current strategy, proof state, and failed strategies</td><td>Deductive agents writing code and Rocq proofs, and an inductive agent proposing new strategies</td><td>Rocq checker on partial proofs, then benchmarks</td><td>Coordinator reverts on stagnation and keeps the fastest verified candidate</td></tr><tr><td>GGMS [32]</td><td>Partial protocol as a game state</td><td>Tree search with a transformer action encoder (no LLM)</td><td>Bounded model checker</td><td>Tree-search values and global depth-first backtracking</td></tr><tr><td>AI-GREEMENT (ours)</td><td>Protocol knowledge base (concept index, mechanism cards), parent protocol, and lessons from earlier attempts</td><td>Coding agent writing a design document, then a Go implementation in the consensus skeleton</td><td>Staged: emulated geo-replicated benchmark, crash fault injection, then Lean proof of safety</td><td>Agent chooses the parent and direction from recorded lessons and user-defined latency targets</td></tr></table>

Table 5: Discovery procedures, described in the nomenclature of SkyDiscover [44].

<table><tr><td>Work</td><td>Problem</td><td>C1</td><td>C2</td><td>C3 sidestepped</td><td>Why the challenges do not arise, or how they are</td></tr><tr><td colspan="6">Classical algorithms</td></tr><tr><td>AlphaDev [47]</td><td>Sorting, hashing</td><td>O</td><td>O</td><td>O</td><td>Output is checked by tests on small deterministic inputs. Latency is measured in microseconds</td></tr><tr><td>FunSearch [62]</td><td>Cap sets, bin packing</td><td>O</td><td>O</td><td>O</td><td>Every packing is valid and every construction is checked exactly</td></tr><tr><td>AlphaEvolve [54]</td><td>Matrix multiplication, scheduling</td><td>O</td><td>O</td><td>O</td><td>Decompositions are checked exactly, and every schedule is valid</td></tr><tr><td colspan="6">Distributed-systems policies</td></tr><tr><td>ADRS [14]</td><td>Spot scheduling, load balancing</td><td>O</td><td>O</td><td>O</td><td>Optimizations rarely alter system semantics. Candidates are scored in a simulator</td></tr><tr><td>Glia [24]</td><td>Request routing</td><td>O</td><td>O</td><td>O</td><td>Every route is valid, and candidates are scored in a simulator</td></tr><tr><td>SkyDiscover [42, 44]</td><td>Cloud transfer, GPU placement</td><td>O</td><td>O</td><td>O</td><td>Every placement is valid, and candidates are scored by an evaluator</td></tr><tr><td colspan="6">Distributed protocols</td></tr><tr><td>Hui et al. [33]</td><td>Atomic commit (n = 3)</td><td>0</td><td>0</td><td>0</td><td>No performance objective. Exhaustive model checking of three processes</td></tr><tr><td>GGMS [32]</td><td>Consensus, atomic commit (n ≤ 4)</td><td>0</td><td>0</td><td>C</td><td>No performance objective. Bounded model checking of at most four processes</td></tr><tr><td>IDS [2]</td><td>Verified key-value stores</td><td>0</td><td></td><td></td><td>Fixed human-written specification. Performance only ranks verified candidates</td></tr><tr><td></td><td>AI-GREEMENT (ours) Fault-tolerant WAN consensus</td><td></td><td></td><td></td><td>Staged evaluation (benchmark, fault injection, Lean proof) rejects candidates early. Humans define the safety properties</td></tr></table>

Table 6: Challenges of consensus discovery (C1: performance conflicts with correctness, C2: tests cannot establish correctness, C3: evaluation is expensive) faced by AI-driven discovery approaches. faced, sidestepped by restricting the problem, does not arise.