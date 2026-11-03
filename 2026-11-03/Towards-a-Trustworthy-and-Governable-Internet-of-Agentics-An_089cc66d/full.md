# Towards a Trustworthy and Governable Internet of Agentics: An Architectural Perspective on Protocols, Systems, and Services

Haoyi Xiong, Senior Member, IEEE, Yi Liu, Jiang Bian, Member, IEEE, Peng Diao, Bin Zhu, Tingting Hu, Yi Wang, and Sumi Helal, Fellow, IEEE

Abstract—Large language model applications are increasingly being organized as tool-using and multi-agent workflows rather than as isolated chat interfaces. This perspective paper examines the architectural problem that follows from that shift: agents may reason, invoke tools, and delegate work, but dependable deployment still requires interoperability across frameworks, runtimes, data boundaries, and organizational control domains. We study this problem through the lens of the Internet of Agentics (IoA), an emerging service-computing view in which heterogeneous agents are treated as governable service participants rather than only as stand-alone model endpoints. The paper organizes the landscape as a three-layer progression: from today’s fragmented agent ecosystem, to interoperable agents, and ultimately to trustworthy and governable autonomy. First, it examines the protocol stack centered on the Agentic AI Foundation, including MCP for agent–tool interaction; A2A for agent–agent coordination; AGENTS.md and Skills-style procedural manifests; and discovery, transport, and trust mechanisms such as AGNTCY and SLIM. Second, it reviews the systems abstractions that make agents deployable as service components, including logical agent architectures, orchestration patterns, gateways, registries, runtime control loops, and hardware-aware inference infrastructure. Third, it situates these developments within services computing, arguing that agentic systems reinterpret classical concerns from SOA and microservices—discovery, composition, governance, pricing, and trust—at the level of cognitive capabilities rather than deterministic functions. The paper concludes that the central IoA research challenge is trustworthy autonomy: ensuring that interoperable agent services are observable, auditable, privilege-bounded, economically controllable, and sufficiently dependable for real-world production use.

Index Terms—Internet of Agentics, Large Language Models, Multi-Agent Systems, Model Context Protocol, Agent-to-Agent Protocol, Agentic AI Foundation, Distributed Computing, Services Computing, Trustworthy Autonomy, Delegation Governance

## 1 INTRODUCTION

Recent agentic systems have moved beyond single-turn assistants toward networks of specialized agents that plan, delegate, invoke tools, and interact with enterprise services [1– 6]. This shift changes the principal systems challenge. Model capability remains necessary, but production deployment increasingly depends on whether heterogeneous agents can interoperate across frameworks, tools, data boundaries, and organizational control domains [1, 2, 7–10]. Existing deployments remain constrained by fragmented interfaces, duplicated integration logic, weak governance mechanisms, and brittle coordination loops [1, 2, 7–10]. The central claim of this paper is that these constraints define a service-computing problem: interoperability must mature into trustworthy autonomy before agentic systems can operate as dependable production services.

We use the term Internet of Agentics (IoA) to denote an emerging distributed service environment in which heterogeneous agents communicate, coordinate, and expose capabilities through open protocols rather than isolated proprietary integrations [11–14]. From a services-computing perspective, the IoA does not displace established service abstractions; instead, it extends them to systems whose components exhibit planning, tool use, memory, and adaptive delegation.

The unit of composition shifts from deterministic software services to cognitive agent services, while enduring concerns such as interface description, discovery, orchestration, state management, trust, and accountability remain central [15–18].

Figure 1 illustrates this framing: it depicts the IoA as a transition from fragmented agent ecosystems, in which individual runtimes depend on local tools and ad hoc integration code, toward an explicit interoperability layer composed of protocols, registries, and gateways. The figure further highlights the services-computing implication of this transition: agent systems become viable production services only when interoperability is accompanied by governance mechanisms for discovery, orchestration, auditability, bounded delegation, and trust.

The first wave of LLM deployment favored vertically integrated, provider-specific stacks, whereas the subsequent spread of open-weight models and orchestration frameworks reproduced a familiar pre-standardization pattern: strong local capability coupled with limited cross-system interoperability [12–14, 19]. This pattern motivates the IoA as a servicescomputing problem rather than merely an agent-engineering problem. The relevant research question is not only whether individual agents can achieve high task performance, but whether independently developed agent systems can cooperate through explicit contracts, observable execution boundaries, and reliability expectations comparable to those already imposed on APIs, workflows, and service platforms.

Reframing agentic interoperability as a challenge of institutionalized service composition, this perspective asserts that basic connectivity is inadequate for autonomous agents that initiate actions, negotiate tasks, and operate across administrative boundaries. Because of this complexity, productiongrade ecosystems require robust mechanisms to specify capabilities, mediate coordination, constrain authority, and ensure comprehensive accountability across heterogeneous participants. Consequently, the Internet of Agentics (IoA) aims to transform fragmented agent implementations into a governed, discoverable, and composable cognitive service infrastructure [14, 16, 17, 20] or also namely (ASC) framework proposed by Deng et al. [15]. While the earlier work formalized the agent lifecycle as first-class services encompassing perception, decision-making, and organizational collaboration, we extend this theoretical paradigm by mapping these dimensions onto a concrete protocol-system-service stack to operationalize ASC principles for real-world deployment.

![](images/e8d16c29637fc4b6604639858ec727e4b9486b2ede5ae7ec893d1d99cc2a8009.jpg)  
Fig. 1. The transitions from fragmented agent ecosystems to the Internet of Agentics (IoA) with governance capabilities for discovery, orchestration, audit, delegation, and trust.

The recent convergence of industry actors around the Agentic AI Foundation (AAIF) gives this architectural transition an institutional basis. MCP standardizes agent access to tools and data sources, A2A specifies peer coordination among opaque agents, and complementary mechanisms such as AGENTS.md, Claude Skills, and secure messaging frameworks begin to provide governance, reusable procedural knowledge, and operational control [12–14, 21–25]. Detailed protocol definitions are centralized in Section 3 to avoid repeating them throughout the paper.

The intellectual roots of this development extend beyond contemporary protocol work. Classical multi-agent systems contributed formal agent models, communication languages, and delegation mechanisms, while distributed systems and services computing contributed the interface, workflow, and governance abstractions that modern agentic platforms are now reinterpreting for probabilistic and adaptive components [16, 26–33].

This paper presents an integrative architectural perspective on the emerging Internet of Agentics (IoA), synthesizing protocols, systems, and services-computing developments into a unified framework for trustworthy and governable agentic ecosystems. The paper develops a three-layer architecture of the IoA, as shown in Fig. 2, moving from protocols, to system design, to services-computing implications. It makes the following contributions:

• It synthesizes the emerging protocol layer for interoperable agent ecosystems, with particular attention to MCP, A2A, Agent Cards, AGENTS.md, Claude Skillsstyle procedural manifests, and AAIF-led standardization and protocols.

![](images/0842730a972e72e2e1a17fca195f5b0c318a35caa47a49c1606c28a8ad56fc0f.jpg)  
Fig. 2. Three-layer architecture of IoA, spanning the protocol, systems, and services layers. The framework captures agent interoperability, deployment infrastructure, and services-computing implications.

It analyzes the systems layer that turns agents into deployable service components, connecting orchestration, gateways, registries, tracing, and execution control to services-computing concerns.

It maps the services layer of the IoA onto the established lineage of discovery, composition, governance, and trust, showing in which ways agentic cognitive services extend rather than replace earlier SOA and microservice principles.

It identifies trustworthy autonomy as the integrative research agenda for the IoA: agent services must become observable, auditable, privilege-bounded, and economically controllable before they can support production workloads [15, 20].

Fig. 2 shows the proposed three-layer view of the Internet of Agentics (IoA), comprising protocol, systems, and services layers, with trustworthy autonomy spanning all layers.

The remainder of the paper follows this progression. Section 2 explains the perspective methodology, source boundaries, and positioning against closely related works. Section 3 analyzes the emerging protocol stack. Section 4 examines logical agent design, orchestration, runtime abstractions, and hardware-aware implementation concerns relevant to deployable agent systems. Section 5 maps these developments to services computing and representative agentic services. Section 6 synthesizes the resulting research agenda around trustworthy autonomy, and Section 7 discusses limitations and concludes.

## 2 METHODOLOGY AND POSITIONING

Because much recent Internet of Agentics activity appears first in protocol repositories, foundation documents, industry technical reports, as well as materials from the Linux Foundation, rather than in archival journals [8, 11, 12, 14, 34], we use a structured narrative review rather than a strict PRISMAstyle systematic review. The goal is explanatory synthesis: connecting protocol specifications, deployable system abstractions, and service-computing implications in a field whose terminology and governance structures are still stabilizing.

## 2.1 Sources and Scope

The corpus combines four source types: protocol and standards documents for MCP, A2A, ACP, AGENTS.md,

TABLE 1  
Comparison with five recent and closely related works.
<table><tr><td>Paper</td><td>Primary Focus</td><td>Protocol Coverage</td><td>Systems Coverage</td><td>Services Coverage</td></tr><tr><td>Masterman et al. [18]</td><td>Survey of emerging AI-agent architectures</td><td>None</td><td>Reasoning, planning, tool calling, and agent communication</td><td>Limited</td></tr><tr><td>Guo et al. [2]</td><td>Survey of LLM-based multi-agent collaboration</td><td>Minimal</td><td>Collaboration patterns, benchmarks, and challenges</td><td>Limited</td></tr><tr><td>Yan et al. [35]</td><td>Survey of communication-centric LLM-MAS review</td><td>Discusses communication protocols conceptually</td><td>Communication mechanisms, negotiation, scalability, and security</td><td>Limited</td></tr><tr><td>Mohammadi et al. [36]</td><td>Survey of evaluation and benchmarking of LLM agents</td><td>None</td><td>Evaluation objectives, benchmarks, metrics, and tooling</td><td>Enterprise evaluation concerns</td></tr><tr><td>ASC survey [15]</td><td>Survey of agentic services computing</td><td>Brief MCP/A2A discussion</td><td>Focus on software rather than software-hardware synergy Runtime control,</td><td>Lifecycle framework and service view</td></tr><tr><td>This paper</td><td>Integrative architectural perspective on IoA across protocols, systems, and services</td><td>Detailed MCP, A2A, ACP, Agent Cards, AGENTS.md, and AGNTCY/SLIM</td><td>orchestration, gateways, registries, and hardware-aware implementation</td><td>SOA mapping, domains, governance, and trustworthy autonomy</td></tr></table>

Claude Skills and the Agent Skills specification, and AG-NTCY/SLIM [13, 14, 22–25, 34, 37–40]; recent surveys and peer-reviewed work on LLM agents, multi-agent systems, service computing, and agent architectures [1, 2, 4–6, 15, 18, 26, 35]; industry reports and failure analyses describing production constraints [7, 8, 11]; and foundation governance materials that document AAIF-led standardization and project convergence [12, 21]. Sources were collected through citation tracing, targeted searches in Google Scholar, Semantic Scholar, and arXiv, monitoring of relevant protocol repositories and foundation announcements, and review of implementationoriented technical reports.

Table 1 compares the scope and contributions of this article with those of the five closest recent reviews. Existing work emphasizes agent internals and architectures [1, 18], multi-agent collaboration and communication [2, 35], evaluation [36], or agentic services computing [15]. This work uses that literature as context but focuses on the combined protocol–system–service stack of the IoA. Because the field is changing rapidly through protocol releases, foundation coordination, and production-oriented technical reports [8, 11, 12, 14, 34], the result should be read as an integrative architectural map rather than an exhaustive census of all agentic systems.

## 2.2 Methodological Limitations

This methodology has several limitations. First, a structured narrative review cannot provide the reproducibility guarantees of a full PRISMA-style systematic review. It is appropriate for an emerging architectural field in which specifications, repositories, and foundation announcements often precede archival publication, but it necessarily involves judgment in selecting which systems and standards are most representative [11, 12, 14, 34]. Second, public documentation is uneven across vendors and communities. Protocol specifications tend to expose interface semantics, whereas production systems often expose only partial details about deployment architecture, performance, governance, or adoption. We therefore treat industry sources as evidence of design intent and ecosystem direction rather than as independently verified performance claims [7, 8, 11].

Third, the corpus is biased toward English-language and publicly documented materials. Internal enterprise deployments, non-English technical communities, and proprietary agent platforms may therefore be underrepresented. Finally, this paper synthesizes architectural patterns rather than benchmarking implementations under controlled workloads. Claims about protocol maturity, runtime design, service governance, and hardware-aware inference infrastructure should be read as a map of emerging design pressures, not as a definitive ranking of products or platforms [11, 15, 18].

Accordingly, this work should be read not as a purely exhaustive survey, but as an integrative architectural perspective that synthesizes protocols, systems, and services-computing developments into a coherent IoA framework.

## 3 PROTOCOLS AND STANDARDS OF THE IOA

The first layer in the IoA progression is the protocol layer. It addresses a familiar service-computing problem in a new setting: heterogeneous components must interoperate without pairwise integration work. In the agentic setting, those components include not only services and data sources but also autonomous agents with different internal models, tool sets, procedural skills, and governance boundaries. The resulting standards landscape is best understood as a layered stack rather than as an exclusive standardization contest [13, 14, 23– 25].

## 3.1 AAIF and Protocol Governance

AAIF was created to prevent the agent ecosystem from fragmenting into incompatible vendor silos [12, 21]. Its significance is less the number of member organizations than the fact that protocol governance is being pulled into a neutral institutional setting early in the lifecycle of the ecosystem. For services computing, this institutionalization is consequential because interoperability depends not only on message formats but also on durable processes for versioning, conformance, and change control.

Within this stack, MCP occupies the agent–tool layer, A2A occupies the agent–agent layer, AGENTS.md supplies declarative local governance, Claude Skills package reusable procedural knowledge, and complementary transport and identity work such as AGNTCY and SLIM target discovery, routing, and secure messaging [13, 14, 23–25]. The protocols are therefore complementary: they define different boundaries in the same distributed system.

Figure 3 summarizes this layered interpretation. It highlights the main organizing principle of the current ecosystem:

![](images/3f2850e696d8d1caa740784b6cbb0041f03d87512348b26ffb7eb7f85f5e8bd4.jpg)  
Fig. 3. Layered IoA protocol stack: agent interoperability from infrastructure and service substrates through MCP-based capability access, A2A/ACP delegation, AGENTS.md governance, Claude Skills-style procedural knowledge, AGNTCY/SLIM discovery and secure transport, and AAIF-led ecosystem governance.

inter-agent communication, tool access, local governance, procedural skills, and secure transport are being standardized separately because they solve different service-integration problems.

## 3.2 Capability Access Layer

The lowest protocol layer connects agents to tools, data, prompts, and deterministic services. Its design problem is vertical integration: how a reasoning system discovers and invokes external capabilities without a custom adapter for every model–tool pair. MCP is a representative of this layer [13, 34, 41].

## 3.2.1 Model Context Protocol (MCP)

MCP standardizes how an agent discovers and invokes external capabilities [13, 34]. Its practical value lies in replacing bespoke adapters for each model–tool pair with a common protocol for exposing tools, resources, and prompts through a stable interface. For services computing, MCP is the main bridge between probabilistic reasoning and deterministic back-end functions, and more concretely a response to the N×M integration problem that earlier API- and plugin-centric ecosystems handled poorly [13, 41].

Architecturally, MCP uses JSON-RPC and supports both local and remote transports. The important abstraction is not the wire format itself but the separation of concerns: servers expose capabilities, clients mediate access, and the model decides when to call a tool or retrieve a resource. This reduces integration cost while preserving the possibility of policy control at the host boundary. The client-mediated sampling model is especially important because it prevents servers from directly controlling model outputs and keeps authority with the host application [13, 34].

MCP also highlights the main deployment challenge of agentic services: capability exposure is easy to standardize, but trust is not. Community servers vary in quality, permission models remain inconsistent, and enterprise use still depends heavily on local isolation, auditing, and approval flows.

MCP should therefore be read as a necessary interoperability layer, not as a complete governance solution.

This distinction is important for service design. A capability contract can tell an agent what a tool does and how to invoke it, but it does not by itself guarantee that the tool invocation is authorized, safe in context, or aligned with organizational policy. The operational burden therefore shifts toward runtime mediation, which makes MCP most defensible for production use when paired with explicit hostside controls [13, 34].

## 3.3 Delegation and Coordination Layer

Above capability access, the coordination layer treats agents themselves as addressable service actors. Its design problem is horizontal delegation: how one autonomous agent discovers another, negotiates a task, receives progress, and obtains artifacts without requiring shared internals [14, 22].

## 3.3.1 Agent-to-Agent (A2A) Protocol

A2A specifies delegation among autonomous agents [14]. If MCP answers “how does an agent call a capability?”, A2A answers “how does one agent ask another agent to do work?” The protocol introduces a task model, asynchronous execution semantics, streaming updates, and a discovery mechanism based on Agent Cards. That design is directly relevant to service ecosystems because it makes agents addressable participants in distributed workflows and consolidates prior interagent designs such as ACP into a broader industry-facing coordination model [14, 22].

The key architectural choice in A2A is opaque execution. Agents publish their capabilities and interaction requirements, but they do not have to expose their internal plans, tools, or prompts. This is attractive for enterprise and crossorganizational coordination because it preserves implementation autonomy while still permitting capability-based delegation. At the same time, the same opacity makes debugging, verification, and liability assignment more difficult. In practice, production use will require richer telemetry and audit trails than the protocol alone currently guarantees [9, 14].

The service-computing implication is that A2A effectively turns agents into discoverable service endpoints with longrunning, stateful task semantics. That is materially different from REST-style request/response integration. Task state, streaming updates, human escalation, and artifact return all become part of the service contract. This richer interaction model is useful, but it also raises the bar for observability and conformance [14, 17].

AAIF’s incorporation of ACP into the broader A2A direction reduces fragmentation, but important questions remain open: how Agent Cards should be validated at scale, how multiple bindings should converge operationally, and how coordination failures should be diagnosed when task state alone is insufficient. These are not merely ergonomic issues; they determine whether A2A can support reliable service composition rather than isolated demonstrations [12, 14, 22].

## 3.4 Governance and Procedure Layer

Connectivity and delegation are insufficient unless agent behavior is bounded by local policies and repeatable methods. This layer supplies context about what an agent is allowed to do, which conventions it should follow, and how recurring tasks should be executed. Declarative agent manifests such as AGENTS.md and SOUL.md, alongside Claude Skills, are therefore not peripheral prompt files; they are emerging specification-like mechanisms for making agent behavior portable across projects, hosts, and service environments [23, 24, 39, 42].

## 3.4.1 AGENTS.md Context Provisioning

AGENTS.md addresses a different problem than MCP or A2A: local behavioral governance [23]. It gives agents projectspecific instructions about constraints, approval policies, repository conventions, and role boundaries. From a services perspective, it functions as a lightweight governance overlay that constrains how an agent may act once it already has access to tools and collaborators [23].

Its importance can be underestimated because it is not a transport protocol. In practice, however, interoperable agent services need local policy hooks that live closer to repositories, workflows, and organizational control points than to network specifications. AGENTS.md is one concrete example of this broader design need [23].

## 3.4.2 Claude Skills and Agent Skills

Claude Skills are reusable procedural packages that describe when an agent should invoke a specialized workflow, what instructions it should follow, and which supporting files, scripts, or assets should be available during execution [24, 37, 38]. They push standardization one layer higher than capability exposure: instead of only exposing tools, they package operational knowledge for how a task should be performed. In service terms, a skill is a portable playbook for agent behavior, analogous to a service recipe or runbook that can be installed, versioned, audited, and reused across task contexts. This distinction matters because dependable agent services require both access and method. Knowing which tool exists is not the same as knowing how to use it safely and effectively, and analogous abstractions are already appearing across multiple vendor ecosystems [14, 24, 40].

The broader Agent Skills specification further positions this pattern as an open, cross-platform format rather than only a Claude-specific customization mechanism [39, 43]. A skill is conventionally organized as a directory with a SKILL.md entry point, YAML front matter, natural-language instructions, and optional supporting files such as scripts, examples, templates, or reference documents [24, 38, 44]. This structure supports progressive disclosure: the agent can see lightweight metadata when selecting a skill, then load the full instructions and supporting material only when needed. From an IoA perspective, that makes skills a procedural standard for packaging local expertise, evaluation lessons, and organizational workflows into reusable agent capabilities.

Table 2 clarifies why this layer is not redundant with MCP. Connectivity protocols and Claude Skills-style procedural manifests solve adjacent but different problems, and mature agent services generally need both.

## 3.5 Discovery, Transport, and Trust Layer

The outer protocol layer concerns ecosystem-scale operation. Once agents and capabilities span vendors, tenants, and organizations, the stack needs discovery, routing, identity, secure transport, and gateways that can translate among partially overlapping standards [25].

TABLE 2  
Connectivity protocols and Claude Skills-style procedural packages as complementary layers in agentic service execution, based on MCP and Skills documentation [13, 24, 34, 39].
<table><tr><td>Dimension</td><td>MCP [13, 34]</td><td>Claude Skills / Agent Skills [24, 39]</td></tr><tr><td>Core goal</td><td>Capability exposure</td><td>Reusable procedural knowledge</td></tr><tr><td>Mechanism</td><td>Protocol interface</td><td>SKILL.md packages, Markdown, scripts,</td></tr><tr><td>Primary value </td><td>Reusable tool access</td><td>and assets Safer, more consistent method</td></tr><tr><td>Main limit</td><td>Weak built-in governance</td><td>execution Weak cross-vendor standardization and validation</td></tr></table>

## 3.5.1 AGNTCYand SLIM Messaging

AGNTCY and SLIM extend the stack toward discovery, routing, identity, and secure messaging [25]. They matter most when agent ecosystems become multi-tenant and crossorganizational, because at that point capability invocation is only part of the problem; systems also need verifiable identity, secure transport, auditability, and scalable message routing [25]. Their ecosystem visibility and implementation maturity appear less established than MCP or A2A, but they signal where the stack may evolve as production requirements harden.

## 3.5.2 Complementary and Emerging Protocols

Other efforts, including Agent Protocol, ANP, and IoToriented standardization work, illustrate two continuing pressures on the ecosystem [14, 45]. One is simplification: lightweight protocols remain attractive for local execution and experimentation. The other is decentralization: some use cases demand identity and discovery models that do not rely on centralized registries. These efforts suggest that long-term interoperability may depend on gatewaying and translation rather than on perfect protocol monoculture.

## 3.6 Protocol Comparison and Synthesis

Taken together, the protocol stack suggests a stable conceptual separation. MCP handles vertical integration with tools and data. A2A handles horizontal interaction among autonomous peers. AGENTS.md and procedural manifests constrain local behavior. AGNTCY- and SLIM-like efforts target discovery, trust, and secure transport at larger scale.

However, with respect to the interoperability goal, this protocol stack appears to provide only the scaffolding for interoperability, while falling short of guaranteeing semantic, behavioral, economic, and governance interoperability required for trustworthy autonomous agent ecosystems. For example, consider an orchestrator delegating a financial risk assessment. When a sub-agent uses MCP to log a risk entity, the request might succeed syntactically but fail semantically if prompt drift outputs natural language instead of a required enumeration. This issue highlights a critical lack of formal interface models to mathematically align the semantic intent of an agent with a tool’s execution boundary. Furthermore, the ecosystem exhibits immature failure semantics, i.e., when an agentic workflow halts mid-task, current protocols offer no native state-reconciliation mechanisms–such as automated rollbacks – to clean up partial database updates. Additionally, the opaque delegation driven by A2A protocol obscures correctness guarantees; an orchestrator receiving a “Low Risk” summary cannot currently demand a verifiable provenance graph proving which data was actually analyzed, natively by the protocols. Ultimately, protocols today standardize connectivity, but the future Internet of Agentics must standardize trust. Until formal models, failure semantics, and correctness guarantees are integrated, trustworthy autonomy will remain heavily reliant on custom runtime infrastructure.

Table 3 summarizes the current division of labor across the major protocol families.

## 4 IOA SYSTEM DESIGN

Protocols make interoperability possible, but systems determine whether interoperable agents can be deployed as reliable services. The second layer in the IoA progression is therefore the systems layer: logical agent architectures, orchestration patterns, runtime control planes, and hardwareaware inference infrastructure. From a services-computing perspective, the central systems question is how probabilistic reasoning can be bounded, observed, coordinated, accelerated, and recovered under production conditions. This section therefore treats system design at two connected levels: the logical design of agents and multi-agent runtimes, and the implementation substrate that makes long-running, stateful agent services feasible.

## 4.1 Architectural Patterns

The system patterns of the IoA appear at three levels: the internal structure of a single agent, the topology used to coordinate multiple agents, and the control-plane infrastructure used to expose those agents as services [9, 10].

## 4.1.1 Agent Architectures

Individual agents range from simple chat interfaces to retrieval-augmented agents, tool-using planners, reflective agents, computer-use agents, and long-horizon coding agents [4, 5]. The main design axes are planning horizon, tool depth, statefulness, and environmental coupling. Systems near the low end of the spectrum remain close to traditional software clients; systems near the high end begin to look like bounded autonomous services with persistent context, approval policies, and execution loops.

From a deployment perspective, the crucial distinction is whether an agent is exposed as an isolated assistant or as a reusable service component. Once exposed as a service, the agent needs explicit contracts around capabilities, memory, execution boundaries, identity, and failure handling. That is the point at which protocols such as MCP and A2A become operational necessities rather than developer conveniences: they make the agent’s reasoning loop addressable, delegable, and governable by surrounding software infrastructure [13, 14].

## 4.1.2 Multi-Agent Orchestration

Four orchestration patterns recur across current systems. Hierarchical supervision is a common and enterprise-compatible pattern because it simplifies delegation, retries, and result aggregation. Peer-to-peer collaboration appears when teams of agents debate or negotiate without a single controller. Map-reduce styles suit parallelizable analysis, while pipeline styles remain useful when work naturally decomposes into ordered stages [6, 9, 10, 14, 22]. These patterns are not novel in distributed systems, but agentic protocols give them a standardized control surface.

![](images/b9a0f92354305efbdc7733313325975316cb0cbfc5aa26da773382d0683b89da.jpg)  
Fig. 4. Common orchestration patterns in current agent systems. These patterns recur because they balance delegation flexibility against different observability, retry, and policy-enforcement requirements.

Among these patterns, hierarchical supervision appears most compatible with existing service-governance practice because it localizes responsibility and makes oversight tractable. Peer-to-peer collaboration is conceptually appealing but operationally harder to audit. Pipeline and map-reduce compositions are often easier to evaluate because they preserve clearer stage boundaries and output contracts [6, 9, 14].

Figure 4 summarizes these recurring coordination topologies and highlights their different observability burdens. A supervised hierarchy concentrates responsibility and simplifies rollback, whereas peer collaboration complicates provenance. Map-reduce and pipeline forms remain operationally attractive because they preserve stage boundaries and aggregation points.

## 4.1.3 Distributed Agentics

At scale, the defining infrastructure components are agent gateways, registries, and runtime managers [25]. Gateways translate protocols, enforce policy, and expose observability. Registries index capabilities and health. Runtimes separate declarative configuration from execution, making agents look more like deployable service units than like ad hoc prompt scripts. This is the systems shift that most clearly links the IoA to mainstream services computing.

The result is a recognizable service architecture. Agent gateways resemble API gateways with added policy and task semantics. Registries resemble service discovery systems with richer capability descriptions. Agent runtimes resemble managed execution environments that bind policy, context, and tool access together. The novelty lies in the probabilistic reasoning core, but the operational scaffolding is still unmistakably that of distributed services [16, 17, 25].

## 4.2 System Implementation and Hardware-Aware Runtime Design

The implementation layer connects the logical agent patterns above to the physical resources needed to run them at service scale. This subsection separates two implementation questions: how runtime work is divided between deterministic orchestration and model inference, and how inference-serving mechanisms manage long-lived context, latency, and cost.

TABLE 3  
Comparison of major IoA protocol families and the distinct service-integration problem each addresses [13, 14, 23–25, 34, 39].
<table><tr><td>Protocol Family</td><td>Primary Role</td><td>Strength</td><td>Main Open Issue</td></tr><tr><td>MCP [13, 34]</td><td>Agent-tool and agent-data integration</td><td>Reduces bespoke adapters through standardized capability</td><td>Fine-grained authorization and trust remain runtime-dependent</td></tr><tr><td>A2A [14, 22]</td><td>Agent-agent delegation and coordination</td><td>exposure Supports stateful, asynchronous multi-agent workflows</td><td>Opaque execution complicates diagnosis, auditing, and liability</td></tr><tr><td>AGENTS.md / procedural manifests [23, 24, 39]</td><td>Local governance and method guidance</td><td>Binds agent behavior to project and organizational policy</td><td>Lacks a universal cross-vendor operational model</td></tr><tr><td>AGNTCY / SLIM [25]</td><td>Discovery, routing, identity, secure transport</td><td>Provides identity, transport, and policy primitives for inter-organization agent meshes</td><td>Specifications and conformance suites are still evolving, with limited independent deployment evidence</td></tr></table>

## 4.2.1 CPU–GPU Runtime Bifurcation

Agent runtimes increasingly look like control planes that assemble context, call tools, validate outputs, and manage long-lived task state [46, 47]. This control plane is software architectural, but it is not hardware-neutral. Agent workloads are dominated by repeated context assembly, LLM inference, tool waiting, validation, and state handoff. As a result, production systems must coordinate two different substrates: CPU-side orchestration for deterministic control and GPUside inference for probabilistic reasoning.

The CPU side performs event parsing, routing, retrieval, prompt construction, tool execution, policy checks, and validation. The GPU side performs model inference, which is itself split into compute-bound prefill and memory-bound decode phases. This split matters for system design because agent workflows repeatedly alternate between reasoning and external action. While the GPU generates or revises a plan, the CPU can execute tools, collect observations, and prepare the next context. Efficient runtimes therefore resemble staged event-driven systems in which inference, retrieval, tool execution, and validation are pipelined rather than serialized.

## 4.2.2 Inference Serving and Stateful Context

Hardware-aware inference serving further changes the design space. Prefill–decode disaggregation separates the computeintensive processing of long prompts from the memorybandwidth-limited generation of output tokens, allowing different GPU pools or scheduling policies to optimize each phase. Dynamic migration systems such as Llumnix [48] and disaggregated serving systems such as DistServe [49] are important not merely as accelerator optimizations, but because they make bursty, multi-step agent workloads easier to schedule under latency and cost constraints. For services computing, the practical implication is that the agent runtime must expose budget, latency, and quality controls that reflect the physical cost of reasoning.

The most agent-specific hardware pressure is state management. Long-running workflows repeatedly reuse large contexts, branch into subtasks, and return to earlier evidence. KV cache memory therefore becomes a first-class systems resource rather than an invisible model-inference detail. Techniques such as PagedAttention [50], prefill reuse, and cacheaware scheduling are relevant to agentic services because they affect how quickly delegated tasks can resume and how expensive long-horizon reasoning becomes. We treat these mechanisms as implementation constraints rather than as a separate architectural layer, because the service-facing question is whether the runtime can expose useful latency, cost, and isolation controls to the agent system above it.

![](images/11e8187918c854081e4b64914ef7545ea283d35acedbf925c23648e226e253ed.jpg)  
Fig. 5. CPU–GPU bifurcation in agentic runtime design. CPU-side orchestration handles deterministic routing, context assembly, tool execution, and validation, while GPU-side inference handles model reasoning through prefill and decode phases.

These implementation choices do not replace logical agent design; they constrain it. A planner with long context, multiple delegated subtasks, and frequent validation loops imposes different hardware demands from a short-lived retrieval agent. Conversely, hardware-aware runtimes can make richer logical designs practical by reducing redundant prefill, reusing shared context, and hiding latency behind tool execution. The systems layer of the IoA therefore sits between software architecture and physical infrastructure: gateways, registries, and policies make agents governable, while inference serving and memory hierarchy make them operationally feasible. It is important to note, however, that current requestlevel abstractions do not by themselves capture the complex structure of multi-agent delegation graphs. We detail the physical isolation and context-provenance risks of this gap in

Section 4.3.2, and examine how SLA design and cost modeling must evolve to address it in Section 6.4.

## 4.3 Limitations of Current System Architectures

The same architectural features that make agent systems useful also create unresolved systems risks. This subsection groups those risks into two categories: coordination and observability failures in the logical runtime, and security or isolation failures that emerge when agents receive persistent authority over tools, context, and infrastructure [7, 8, 14, 20].

## 4.3.1 Coordination, Context, and Observability

Current systems remain constrained by four recurring weaknesses. Multi-agent execution still suffers from context degradation and error amplification: long delegation chains either pass forward too much conversational residue or compress intermediate state too aggressively, allowing a small hallucination to propagate downstream [7]. Coordination overhead also grows faster than many demonstrations suggest; gains are strongest on naturally parallel workloads, whereas sequential workflows often accumulate latency, token cost, and verification overhead faster than they gain quality [7, 8].

Observability and accountability are likewise immature relative to conventional service platforms. Opaque execution is often desirable at the protocol boundary, but today’s runtimes still expose limited diagnostics about why a delegated task failed, which context was used, how tools were selected, or where policy violations emerged [14]. Existing codingagent frameworks make progress through validation loops and approval checkpoints, yet the broader ecosystem still lacks stable norms for execution tracing, conformance testing, and post hoc audit.

## 4.3.2 Security and Hardware Isolation

Security and privilege management are also not commensurate with the authority granted to modern agents. Application-layer prompts and allowlists are insufficient once agents can act persistently on external systems or delegate work to other agents. More mature systems will need bounded credentials, stronger sandboxing, revocable delegation, and clearer separation between reasoning authority and execution authority [14, 20, 51].

From a service-governance perspective, hardware-aware implementation still exposes unresolved maturity problems. Disaggregated serving, cache sharing, and memory pooling improve feasibility, but they introduce new scheduling and isolation problems. Shared context caches can reduce cost, yet they also raise questions about tenant separation, cache provenance, and side-channel leakage. Similarly, moving state across GPU, CPU, CXL, NVMe, and DPU-managed network tiers can make reasoning cheaper, but it makes reproducibility and fault diagnosis harder unless runtimes preserve clear execution traces. Mature IoA systems therefore need diagnostics that connect logical events—prompts, retrieved context, tool calls, approvals, and side effects—to physical events such as cache reuse, migration, inference latency, and resource consumption.

## 5 AGENTIC SERVICES COMPUTING

With the protocol and system layers in place, the third layer in the IoA progression is the services layer. The main question for services computing is what agents contribute beyond conventional service composition. Our argument is that agentic systems do not replace service-oriented thinking; they relocate it. The system boundary shifts from deterministic functions toward protocol-described agent and tool capabilities, but the core service concerns remain discovery, orchestration, policy, and trust [14, 16, 33].

## 5.1 ASC Framework

The agentic services computing framework (the term was firstly coincided in [15]) explains how conventional serviceengineering concepts reappear when the service actor is no longer only a deterministic endpoint but a bounded reasoning system. Here we extends the ASC framework through three lenses: the historical recurrence from SOA to agentic services, the changing role of data and semantics, and the lifecycle implications of deploying agents as governed services [16, 17].

## 5.1.1 Cyclical Thesis

The clearest services-computing interpretation of the IoA is one of architectural recurrence rather than wholesale replacement [16]. RPC becomes model-mediated function invocation. BPEL-style orchestration becomes agent planning and delegation. SOAP/WSDL/UDDI-style interface and discovery concerns reappear in MCP and Agent Cards. Microservice boundaries become capability boundaries, where an agent may serve as an intelligent facade over conventional services while deterministic microservices continue to handle precise operations. The ESB reappears in more decentralized form through agent coordination protocols such as A2A [14, 33, 52].

This analogy is useful but incomplete. Agentic systems add failure modes that traditional SOA did not face, including hallucination cascades, context degradation, and nondeterministic coordination. The value of the cyclical thesis is therefore diagnostic, not predictive [7, 8].

As formalized in Table 4, the transition from SOA to the IOA represents an evolutionary relocation of core services computing principles. While classical services computing systems relied on deterministic, structural coupling via RPC and microservice interfaces, the agentics introduces a paradigm shift toward semantic, intent-based late binding and cognitive capability boundaries. Similarly, the static finite state machines that once dictated BPEL/BPMN workflow orchestration are supplanted by emergent, probabilistic cognitive loops. The point is not that old paradigms can simply be renamed; rather, the same fundamental design pressures–such as remote invocation, integration middleware, and state management–recur at a higher layer of abstraction. Ultimately, while services computing once standardized deterministic software interfaces, the IoA reinterprets these boundaries to govern the interaction among bounded reasoning systems.

## 5.1.2 Data Standards and Semantics

Agentic services also alter data handling. XML and JSON remain essential interchange formats, but they are increasingly complemented by vector representations that support semantic retrieval and cross-modal matching [53]. This does not eliminate structured data management; rather, it adds a semantic access layer above conventional service contracts. For service composition, the practical implication is that retrieval quality and grounding become part of the reliability story, especially as hybrid relational-vector systems become part of mainstream service infrastructure [53–55].

TABLE 4  
Contrasting SOA with Agentic AI: A paradigm shift from deterministic, programmatic composition to autonomous, LLM-based orchestration.
<table><tr><td>Dimension</td><td>Historical Paradigm</td><td>Agentic Equivalent</td><td>Key Architectural Shift</td></tr><tr><td>Remote Invocation</td><td>RPC / gRPC / CORBA</td><td>LLM Function Calling</td><td>Deterministic coupling → semantic intent-based late binding</td></tr><tr><td>Workflow Orchestration BPEL / BPMN</td><td></td><td>Agentic Workflows (ReAct, CoT)</td><td>Static finite state machines → emergent cognitive loops</td></tr><tr><td>Interface / Discovery</td><td>SOAP / WSDL / UDDI</td><td>MCP (JSON-RPC manifests)</td><td>Manual WSDL integration → LLM-driven autonomous discovery</td></tr><tr><td>Service Decomposition</td><td>REST Microservices</td><td>Agentic Microservices</td><td>Business logic boundaries → cognitive capability boundaries</td></tr><tr><td>Integration MiddlewareEnterprise Service Bus</td><td></td><td>A2A Protocol</td><td>Centralized hub-and-spoke → decentralized peer-to-peer</td></tr><tr><td>Data Representation</td><td>XML, JSON (syntactic)</td><td>Vector Embeddings (semantic)</td><td>Keyword matching → geometric similarity in latent</td></tr><tr><td>State Management</td><td>Stateless REST</td><td>Deeply Stateful (KV Cache, Memory)</td><td>space Ephemeral requests → persistent context across multi-step workflows</td></tr></table>

Note: The mappings shown are conceptual analogies intended to highlight shifts in abstraction and system design. They do not imply strict architectural equivalence.

This observation is central because service reliability now depends partly on information selection before execution. If an agent service retrieves the wrong context, downstream reasoning can fail even when the surrounding infrastructure is correct. Grounding, retrieval quality, and evidence ranking therefore become service-quality concerns rather than only model-quality concerns [53, 55].

## 5.1.3 ASC Lifecycle

Agentic services also expand the service lifecycle. Design now includes capability scoping and policy definition. Deployment includes runtime configuration and model binding. Operation includes iterative reasoning, tool use, and cross-agent coordination. Evolution includes agent-capability updates, prompt, policy, workflow, and skill refinement over time [4, 5, 24]. Claude Skills make this evolution concrete: recurring failures, new tools, domain conventions, or improved procedures can be folded back into reusable skill packages, creating a controlled self-evolution loop in which the agent service improves its own procedural substrate while remaining subject to review, versioning, and regression tests. Trustworthiness cuts across all four phases, which is why services governance becomes central rather than peripheral. The lifecycle lens is useful because it shows that agentic services add managerial work at every stage rather than only at runtime.

Table 5 makes this lifecycle shift explicit by mapping each phase to its service purpose and corresponding protocol– system design concerns. The table highlights that security, evaluation, value alignment, and policy control cannot be postponed until deployment; they shape design, runtime operation, and post-deployment evolution as continuous service-management responsibilities.

## 5.2 Service Domains and Web 4.0

The clearest early service domains are those already rich in orchestration and policy. Telecommunications, enterprise workflow automation, software engineering, scientific pipelines,

TABLE 5

Agentic services computing lifecycle across service purpose and design responsibilities, synthesized from agentic services computing, agent runtime, protocol, and procedural-skill sources [4, 5, 13, 14, 23, 24].

<table><tr><td>Phase</td><td>Purpose</td><td>Design</td></tr><tr><td>Design</td><td>Define the agentic service contract: task intent, QoS/SLA targets, risk class, approval gates, and value metric.</td><td>Translate requirements into systems architecture and services interfaces: MCP tools, A2A boundaries, Agent Cards, AGENTS.md constraints, memory, orchestration, validation, and</td></tr><tr><td>Deployment</td><td>Instantiate the agentic service for a tenant: permissions, compliance policy, data boundaries, SLA tier, and billing</td><td>supervision [13, 14, 23]. Bind systems endpoints, schemas, credentials, identity, discovery, and conformance assumptions; configure services routing, retrieval, sandboxing, observability, budgets, and</td></tr><tr><td>Operation</td><td>model [20]. Run the agentic service under QoS/SLA commitments: bounded delegation, escalation, audit,</td><td>isolation [14, 25]. Coordinate systems state, tool results, artifacts, handoffs, and policy signals across MCP/AŽA interfaces; manage services traces, validation, scheduling, latency, cost, and</td></tr><tr><td>Evolution</td><td>accountability, and value delivery [20]. Evolve the agentic service contract: update agent capabilities, pricing, compliance evidence, provenance, and trust controls.</td><td>recovery [13, 14]. Version systems protocols, capability descriptions, Agent Cards, governance files, and Claude Skills; refine services prompts, workflows, caches, evaluation harnesses, and regression tests from incidents [24, 38, 39].</td></tr></table>

and financial services all fit this pattern because they require coordination across heterogeneous capabilities under explicit constraints [14]. In these domains, agents are valuable not because they replace deterministic services, but because they sit above them as adaptive coordinators. A credible near-term deployment pattern is therefore hybrid: agents orchestrate reliable service components rather than attempting to subsume them. The following discussion moves from concrete representative services to broader domain patterns, so the examples first show service boundaries in practice and the summary table then abstracts the recurring roles, backbones, and governance pressures.

![](images/92afab2680195a43d5b1765e9678c6729d40b3a3cef266a7c94ad4ae24c90bd9.jpg)  
Fig. 6. Enterprise agentic mesh workflow illustrating how an orchestrator decomposes a high-level request into specialized services connected through protocol-governed delegation and tool access.

## 5.2.1 Representative Agentic Services

Representative deployments clarify this services view. Claude Code emphasizes composable developer workflow integration, while Goose emphasizes local-first execution and explicit control over context and tools [56, 57]. Enterprise copilots and AI coworkers such as Microsoft 365 Copilot, Claude CoWork, and Gemini for Google Workspace illustrate a different service pattern: agentic assistance embedded inside existing productivity and collaboration suites, grounded in organizationspecific content, and constrained by identity, permissions, compliance controls, and administrative governance [58–62]. These examples belong in the services layer because their defining contribution is not a new protocol or inference algorithm, but a service boundary that connects reasoning to organizational work.

Table 6 summarizes the service-runtime implication that each representative deployment foregrounds. The table is not intended as a product comparison; it identifies recurring design pressures, from supervised validation in coding workflows to identity-grounded governance in enterprise productivity environments.

Figure 6 illustrates this hybrid deployment pattern. A high-level request is decomposed by an orchestrating agent into delegated subtasks, while deterministic back-end services remain responsible for data access, validation, execution, and record updates. This separation is important because it preserves the reliability of conventional service components while allowing agents to coordinate across them.

## 5.2.2 Enterprise and Finance

Enterprise service domains are attractive for agentic deployment because they already contain explicit process boundaries, policy requirements, and heterogeneous back-end systems. In supply-chain analysis, compliance review, or financial reporting, the main difficulty is often not computation itself but coordination across multiple data sources and service interfaces. An agentic layer can reduce the amount of custom integration logic by translating a high-level request into a set of structured subtasks, invoking the required services, and reconciling intermediate results before producing a final artifact [14].

Microsoft 365 Copilot illustrates this enterprise-service pattern in a production productivity environment. Rather than presenting an agent as a free-standing reasoning endpoint, it embeds agentic assistance into Microsoft 365 applications, grounds responses through Microsoft Graph, and constrains access through tenant permissions and organizational controls [58–60]. Its extension model also shows how agentic services can connect to external enterprise knowledge: Copilot connectors bring line-of-business content into Graph-based retrieval or federated access patterns while preserving admin consent, user permissions, and connector governance [63]. Claude CoWork and Gemini for Google Workspace follow the same broad pattern in different collaboration suites: the agent is valuable because it operates over governed organizational context rather than as an isolated model endpoint [61, 62]. These systems make enterprise agentic services computing concrete: the agent layer coordinates across documents, communications, meetings, and external systems, while deterministic identity, compliance, search, and data-governance services bound its behavior.

Financial services provide an even stricter version of this requirement. Market interpretation, strategy synthesis, and anomaly triage are all plausible agentic tasks, but deterministic services remain essential for trade execution, portfolio accounting, regulatory checks, and record updates. This makes finance a useful model for the broader IoA: the future is most likely to be hybrid, with agentic reasoning layered above reliable transactional services rather than replacing them outright.

## 5.2.3 Scientific, Engineering, and Local Automation

Scientific workflows and software-engineering pipelines also fit the IoA pattern because they require iterative reasoning over diverse tools rather than a single monolithic API call. In research review, literature triage, or engineering validation, an agentic service can gather evidence, call specialist tools, compare outputs, and assemble a structured conclusion while leaving consequential decisions subject to human verification [14]. Coding agents and research-oriented agents are useful examples for the same reason: their value comes from orchestrating already useful service components under a higherlevel reasoning loop. Claude Code and Goose make this pattern visible in developer services. Both expose the full loop— repository retrieval, plan formation, tool invocation, validation, and iterative repair—but place the boundary differently: Claude Code emphasizes composable workflow integration and supervised validation, whereas Goose emphasizes localfirst execution for privacy-sensitive settings [56, 57].

Table 7 summarizes the cross-domain pattern after these examples. It shows where deterministic substrates remain indispensable and where governance pressure is strongest across enterprise, finance, scientific, engineering, and local automation settings.

## 5.2.4 The Agentic Web

The idea of a “Web 4.0” or agentic web is useful as a directional metaphor, but it should be treated cautiously. The strongest near-term interpretation is not fully autonomous machine commerce; it is a web in which agents become first-class service consumers and providers under explicit governance. If that vision matures, the protocols surveyed here may play a role for agent interoperability analogous to what earlier web standards did for documents and APIs [64]. Whether that happens depends less on visionary framing than on conformance, security, and deployment economics.

TABLE 6  
Representative agentic services and the service-runtime tradeoffs they foreground [56–61].
<table><tr><td>Service</td><td>Primary Emphasis</td><td>Key Service Implication</td></tr><tr><td>Claude Code [56]</td><td>Composable developer workflow integration</td><td>Strong fit for supervised orchestration and iterative validation</td></tr><tr><td>Goose [57]</td><td>Local-first execution and explicit control</td><td>Strong fit for privacy-sensitive deployments with bounded tool access</td></tr><tr><td>Microsoft 365 Copilot [58–60, 63]</td><td>Enterprise productivity assistance grounded in Microsoft Graph</td><td>Highlights identity-scoped grounding, tenant data boundaries, Purview compliance,</td></tr><tr><td>Claude CoWork [61]</td><td>Enterprise AI coworker for organizational knowledge and shared projects</td><td>auditability, and admin-governed extensibility Highlights collaborative reasoning over governed work context, artifacts, projects, and workspace administration</td></tr></table>

TABLE 7

Representative service domains for the IoA and the hybrid architectures they favor, synthesized from agentic-services, protocol, enterprise-copilot, and developer-agent sources [14, 56–58].
<table><tr><td>Domain</td><td>Agentic Role</td><td>Deterministic Service Backbone</td><td>Primary Governance Pressure</td></tr><tr><td>Enterprise workflows [58–60, 63]</td><td>Intent translation, knowledge grounding, and exception handling</td><td>ERP, Microsoft Graph, document, identity, reporting services</td><td>Auditability, policy conformance, and data-governance controls</td></tr><tr><td>Financial services</td><td>Interpretation, triage, analysis synthesis</td><td>Execution, accounting, compliance services</td><td>Risk controls and regulatory traceability</td></tr><tr><td>Scientific and engineering workflows [14]</td><td>Evidence gathering and tool chaining</td><td>Databases, simulation, build-validation services</td><td>Reproducibility and provenance</td></tr><tr><td>Developer and local automation [56, 57]</td><td>Repository assistance and local orchestration</td><td>File systems, build tools, terminals, OS sandboxes, and schedulers</td><td>Containment, privilege protection, and reproducible validation</td></tr></table>

## 5.3 Agentics or Services: an Open Debate

The emergence of agentic systems raises a fundamental question for services computing: should agents be treated as applications consumed by users, or as first-class service actors that compose, invoke, and deliver functionality within larger service ecosystems and workflows? The question is not mereley semantics. It reflects a deeper shift in the unit of computation, the unit of value, and the nature of service abstraction [65–67].

Unlike traditional software services, which expose deterministic functionality with relatively stable cost and performance characteristics, agentic systems exhibit dynamic execution behavior, uncertain outcomes, and variable resource consumption. As a result, treating agents as conventional services risks obscuring critical properties such as cost, latency, and confidence, which are essential for reliable operation at scale. This section examines the implications of this shift, focusing on how agentic execution redefines service composition, economic models, and system-level guarantees.

## 5.3.1 Service-Economic Unit Shift

Classical services computing grew with subscription economics, where the unit of value was access by an authorized human or organization. SOA, service registries, orchestration engines, access-control systems, and SLA management therefore assumed relatively stable subscribers, interfaces, and entitlement boundaries [16, 33]. Prior services-computing research has nevertheless long treated services as economic objects, including cost-optimized cloud provisioning, gametheoretic service pricing, dynamic pricing for heterogeneous sensor-cloud services, security-service market investment, and profit-aware resource oversubscription [65–69].

Agentics extends this line of work because an agentic service is not a single endpoint call. It is a bundle of model inference, tool execution, validation, and sometimes human review. To make this economic shift precise on the major costs–especially token consumption and latency (both caused by the use of LLMs)–we model execution as a probabilistic directed graph, or namely a random graph, $G = ( V , E )$ . Here, nodes represent heterogeneous operations including LLM inference, tool invocation, validation, and human interaction.

Unlike traditional static graphs, this graph is not known a priori. Because LLMs sample from probability distributions, the graph is constructed incrementally, evolving randomly through emergent branching, delegation, and retries. Each node $v \in V$ incurs an expected cost $\mathbb { E } [ C ( v ) ] \ ( \mathrm { e . g . }$ , token pricing and API fees) and an expected latency E[L(v)], and produces an output with a success probability $Q ( v ) \in [ 0 , 1 ] .$ This model enables defining the expected Economic Quality of Service (E-QoS) for an execution graph as the tuple:

$$
E ( G ) = \langle \mathbb { E } [ C ( G ) ] , \mathbb { E } [ L ( G ) ] , Q ( G ) \rangle ,\tag{1}
$$

where $\begin{array} { r } { \mathbb { E } [ C ( G ) ] = \sum _ { v \in V } \mathbb { E } [ C ( v ) ] } \end{array}$ denotes the expected total cost and $\mathbb { E } [ L ( G ) ]$ denotes the expected critical-path latency.

Modeling the workflow as a probabilistic graph exposes fundamental economic risks that deterministic SOA models ignore. Most notably, ensuring an agent completes a complex task without exhausting its budget becomes a stochastic reachability problem [70]. This problem introduces significant operational and economic complexity, as probabilistic execution risks massive sunk token costs from non-terminating or prematurely halted reasoning loops, alongside severe path suboptimality where even successful tasks take unnecessarily expensive, heuristic-driven routes.

Consequently, agent orchestration cannot simply minimize a static graph topology. It must be formulated as a policy optimization problem under uncertainty—minimizing expected cost while providing formal guarantees that latency and quality constraints are met with high probability [71]:

$$
\begin{array} { r l r } { \underset { \pi } { \operatorname* { m i n } } } & { \mathbb { E } [ C ( G _ { \pi } ) ] } & { \mathrm { s . t . } \quad P ( L ( G _ { \pi } ) \leq L _ { \mathrm { S L A } } ) \geq 1 - \delta , } \\ & { } & { \mathbb { E } [ Q ( G _ { \pi } ) ] \geq Q _ { \mathrm { S L A } } } \end{array}\tag{2}
$$

where $\pi$ represents the orchestration policy governing dynamic graph expansion, and δ bounds the maximum allowable risk of SLA violation, while $Q _ { \mathrm { S L A } }$ and $L _ { \mathrm { S L A } }$ refers to the minimal success rate and maximal latency, expected by SLA, respectively. This perspective shifts agentic systems from heuristic orchestration toward mathematically verifiable, riskaware service composition.

This shift produces a profound economic dilemma: if the value unit is a completed workflow, how should platforms charge for agentic services? Traditional billing mechanisms are fundamentally misaligned with probabilistic execution. While raw LLM access is easily metered by tokens, charging complex agentic workflows per-token forces the consumer to bear the financial risk of non-completion or path suboptimality, as heavily consumed tokens do not guarantee a desired outcome. Conversely, seat-based subscription models expose service providers to unbounded inference costs from heavy utilization, while unfairly penalizing infrequent users. An outcome-driven pricing model appears to be the logical alternative, yet it remains practically intractable because LLMdriven agents currently lack the strict correctness and success guarantees of classical deterministic services computing. Industry and investment analyses increasingly frame this tension as a question of whether agentic AI could “eat software”<sup>1</sup>, earn durable infrastructure returns<sup>2</sup>, and generate enough end-user revenue to support the build-out<sup>3</sup>.

This shift also exposes a deeper tension in service abstraction. A useful perspective emerges when comparing agentic services to earlier efforts such as Service-Oriented Device Architectures (SODA) [72, 73]. In SODA, inherently fragile and resource-constrained devices were abstracted as stable services, often over-promising reliability and safety while masking underlying limitations. This abstraction mismatch led to well-documented challenges in smart environments and cyber-physical systems, including unreliable sensing, intermittent connectivity, and safety risks arising from exposing physical devices as easily invokable services [74]. In contrast, agentic systems invert this tension: powerful, computationally expensive, and dynamically evolving agents are exposed through simplified service interfaces that underrepresent their cost, non-determinism, and operational complexity. This observation reveals a broader duality in service abstraction: when critical properties of the underlying entity are suppressed, whether fragility in devices or expansiveness in agents, system-level risks emerge. In the former case, reliability and safety degrade; in the latter, cost, predictability, and control become the dominant challenges. This duality suggests that future agentic service models must explicitly surface execution bounds, economic cost, and uncertainty as first-class service properties, rather than treating them as hidden implementation details.

## 5.3.2 Transaction Costs in Human-Agent Service Work

Agentic service execution reduces certain forms of coordination overhead while introducing new sources of system complexity. In traditional workflows, task decomposition, information retrieval, and synthesis often require explicit human coordination across multiple tools and services. Agentic systems can partially automate these steps by constructing and executing an Agentic Service Graph (ASG), thereby reducing the need for manual orchestration [75, 76].

However, this reduction in human coordination shifts complexity into the execution graph itself. The dynamic expansion of the ASG introduces variability in cost, latency, and outcome quality, making it difficult to predict resource consumption or guarantee performance. Similar challenges have been observed in multi-agent systems and workflow orchestration, where coordination overhead and uncertainty increase with system scale and autonomy [77, 78].

Furthermore, as agents take on greater responsibility for task decomposition and execution, new forms of oversight become necessary, including validation checkpoints, audit trails, and human-in-the-loop escalation mechanisms [79]. From a services perspective, the key challenge is not simply reducing transaction cost, but redistributing it: human coordination costs are replaced by system-level costs associated with monitoring, validation, and control of agentic execution. This shift reinforces the need for explicit modeling of execution graphs and their associated economic and operational properties.

## 5.3.3 Metering, Contracting, and Market Design

The shift from user-centric applications to agentic service execution necessitates a corresponding shift in how services are metered, priced, and contracted. In traditional service models, billing is typically tied to user access $( \mathrm { e . g . }$ , subscription or seat-based pricing) or to individual API calls. In contrast, agentic systems execute composite workflows represented as Agentic Service Graphs (ASGs), where a single user request may trigger a sequence of heterogeneous operations including model inference, retrieval, tool invocation, and validation [80].

This compositional nature of execution implies that the economically relevant unit is no longer a single service invocation, but the entire execution graph. Consequently, pricing and metering mechanisms must account for the aggregate cost, latency, and quality characteristics of the ASG. This suggests a transition toward outcome-oriented or graphbased pricing models, where charges reflect the resources consumed and the guarantees provided across the full execution path [81, 82].

From a systems perspective, this raises several challenges. First, the dynamic structure of ASGs makes cost estimation non-trivial prior to execution. Second, variability in execution paths complicates the enforcement of service-level agreements (SLAs), particularly when quality and confidence must be considered alongside latency and cost. Third, transparency and auditability become essential, as users and service providers require visibility into how resources were consumed and how outcomes were produced.

These challenges point toward the need for billing-grade execution tracing, cost-aware orchestration policies, and service contracts that explicitly incorporate Economic QoS (E-

QoS) metrics [83]. In this model, agentic services are evaluated not only by functionality, but by their ability to deliver predictable, auditable, and cost-efficient execution under uncertainty. Recent industry analyses also point to emerging pricing diversity and uncertainty in monetizing agentic workflows, reinforcing the need for principled system-level approaches [84, 85].

## 6 FUTURE DIRECTIONS

The preceding sections describe a progression from protocols, to deployable systems, to service-computing interpretation. The next phase of the IoA is not simply broader adoption; it is the transition from interoperable demonstrations to governable services. Connectivity is increasingly available, but trustworthy autonomy remains unresolved. The key research agenda is therefore to make agent services observable, auditable, and economically controllable enough for sustained production use [20]. Future directions should therefore be read not as speculative add-ons but as the conditions under which the IoA can become a durable services-computing paradigm.

## 6.1 Protocol Maturity

Protocol maturity is primarily an operational problem: fewer competing cores, stronger interoperability evidence, and explicit safety guarantees. MCP and A2A are plausible anchors, but long-term value depends on whether they produce dependable behavior across heterogeneous implementations rather than only specification-level agreement [12–14, 86].

## 6.1.1 Conformance and Interoperability

The immediate requirement is a conformance ecosystem: test suites, interoperability events, and certification processes so that “MCP-compatible” or “A2A-compatible” implies behavioral interoperability instead of syntax-level similarity [12, 52].

## 6.1.2 Safety and Verifiable Identity

Safety and identity must mature together. Formal analysis is needed for liveness, delegation bounds, and policy soundness in long-running workflows, while cross-organizational deployment will likely require stronger, verifiable agent identity and zero-trust interaction patterns beyond today’s baseline OAuth and certificate assumptions [14, 25, 86].

## 6.2 Systems Maturity

Systems maturity will come from control-plane quality, not from thicker agent wrappers. Managed runtimes must make traces, policy checkpoints, grounding quality, and cost envelopes first-class operational controls so that long-running multi-agent workloads are governable under production constraints [20, 54, 55].

## 6.2.1 Agent Operating-System Layers

A plausible trajectory is an agent operating-system layer in which compute budgets, memory residency, task scheduling, and inter-agent coordination become native runtime responsibilities. Under this view, lifecycle management and declarative deployment are not convenience features; they are core service infrastructure.

![](images/f6d2d89f7c377f141889dcf9c3c050eff863fd67f19a41dbb9bdb961a554c802.jpg)  
Fig. 7. Delegation Capability Token architecture for multi-agent trust enforcement, showing privilege attenuation and cascade revocation across delegated workflows.

## 6.2.2 Grounded Inference and Infrastructure Co-Design

Grounded inference and hardware co-design must also advance together. “Filter first, reason last” pipelines reduce failure propagation by improving context quality before expensive reasoning begins, while serving stacks need to adapt from static batch assumptions toward irregular, stateful, tool-using workloads with dynamic branching and context reuse [53–55].

## 6.3 Services Maturity

Services maturity depends on governance discipline. The operational shift is from querying models to commissioning bounded agents, which requires delegation policies, supervision points, auditability, and cost controls as default service semantics [20].

## 6.3.1 Enterprise Governance and Commissioning

Enterprise copilots already demonstrate this transition by embedding grounding, identity, retention, audit, and admin governance directly into service environments [58, 63, 87, 88]. A key remaining challenge is portability: governance controls that work in one tenant need to generalize to cross-organizational delegation and multi-provider workflows. This is where the five requirements articulated by Tomašev et al.–dynamic assessment, adaptive execution, structural transparency, scalable market coordination, and systemic resilience–become useful as a practical control model for task fit, supervisory depth, and revocation design [20].

## 6.3.2 Bounded Delegation and Accountability

Cryptographically bounded delegation is a concrete path forward: orchestrators grant narrowly scoped authority to sub-agents instead of broad static credentials, and delegation rights follow task decomposition boundaries [20, 51].

Figure 7 visualizes this idea as a Delegation Capability Token architecture. The figure shows how authority can be attenuated as work moves through a multi-agent task graph, and why revocation must cascade along the same delegation chain if downstream agents inherit limited execution rights from upstream orchestrators.

If agent services are to cross organizational boundaries, authority must be explicit, attenuable, and revocable at taskgraph granularity. In parallel, service maturity will require clearer liability chains and auditable accountability in highstakes domains where no single provider controls full workflow behavior [14].

## 6.4 Open Research Challenges

The open research problems are best framed as servicecomputing questions rather than as generic LLM limitations. Hallucination, latency, and cost are already familiar concerns in agent deployments, but they become research challenges when they must be translated into quality-of-service (QoS), service-level agreement (SLA), and service-governance terms. Prior work on QoS prediction, recommendation, and probabilistic service selection shows that service quality has long been treated as a measurable, context-dependent contract rather than a single accuracy score [89–91]. A conventional service contract can specify availability, latency percentiles, throughput, error rates, and recovery objectives. An agentic service contract must additionally specify how much uncertainty is tolerable, how validation coverage is measured, how latency budgets are allocated across reasoning and tool calls, and how marginal cost is bounded when the agent can branch, retry, and delegate work.

## 6.4.1 Hallucination, Harness Coverage, and Semantic QoS

Hallucination is unlikely to be eliminated completely, especially in long-running, multi-agent workflows where retrieved context, intermediate summaries, and delegated outputs can all introduce uncertainty. The services-computing question is therefore not only how to reduce hallucination, but how to deliver acceptable QoS in its presence. Harness engineering provides one practical direction: agents can be surrounded by retrieval checks, schema validators, policy gates, tool-output verifiers, sandbox execution, approval checkpoints, and post hoc audits [7, 20, 55]. Yet this raises a deeper measurement problem. Unlike code coverage, harness coverage is not simply the fraction of statements or branches exercised; it must describe which semantic claims, tool side effects, delegation paths, policy constraints, and high-risk business states are covered by validation.

This creates a new research agenda for semantic QoS: an extension of classical QoS in which contractual quality attributes range over the meaning of outputs (grounding fidelity, factual accuracy, policy conformance) rather than only over observable service attributes such as latency or availability [55, 91]. Future agentic services need metrics that quantify the coverage of grounding evidence, the sensitivity of outputs to missing or corrupted context, the fraction of delegated actions protected by deterministic checks, and the expected harm of an undetected false claim. These measures must then be mapped to SLA language, extending earlier QoS models from observable service attributes toward evidence-backed semantic behavior [55, 91]. For example, an enterprise agent may not be able to promise zero hallucinations, but it might promise that claims affecting financial reporting, compliance decisions, or external side effects are backed by auditable evidence and deterministic validation before execution. The unresolved problem is how to make such guarantees portable across heterogeneous agents and protocols rather than bespoke to one application.

## 6.4.2 Inference Latency and Agentic SLA Design

Latency is also different for agentic services than for ordinary API calls. A user-facing response may involve multiple LLM invocations, retrieval calls, tool executions, validation passes, and cross-agent delegations. At the infrastructure layer, inference latency depends on GPU availability, prefill and decode behavior, KV-cache residency, cache eviction, GPU–CPU bandwidth, and migration across memory tiers. These details are often hidden from the service consumer, but they directly shape whether an agent can meet an SLA for an interactive task, a batch workflow, or a time-sensitive business process.

The research challenge is to design SLAs that reflect the compositional nature of agent latency. Traditional percentiles such as p95 response time are insufficient by themselves when a task graph may expand dynamically after intermediate reasoning, just as classical service selection had to reason over end-to-end QoS rather than isolated endpoint quality [91]. Agentic SLAs need latency budgets that can be decomposed across planning, model inference, tool waiting, validation, and human escalation. They also need runtime policies that can trade quality against time: using a smaller model, reusing cached context, truncating low-value evidence, parallelizing subtasks, skipping optional validation, or escalating to a human when the remaining latency budget is too small. This connects hardware-aware scheduling to services computing: GPU and KV-cache management become QoS controls only when the runtime exposes their effects to the orchestration layer in terms that service contracts can use.

## 6.4.3 Cost, Elasticity, and Economic QoS

Cost is the third agentic challenge that must be elevated into service-level semantics. Multi-agent execution makes cost difficult to predict because a single request can trigger variable numbers of model calls, tool invocations, retrieval operations, validation passes, and retries [8, 14]. Under a cost-per-use model, this variability is not merely an accounting issue; it changes the architecture of the service. A reliable agentic service must estimate the expected cost of a task before accepting it, monitor cost during execution, and degrade, defer, or reject work when the remaining budget is inconsistent with the requested quality level.

This suggests an economic form of QoS for agentic services. Future systems should expose service contracts that combine functional quality, confidence, latency, and marginal cost rather than optimizing them separately. This direction generalizes QoS-aware service recommendation and selection from choosing among service endpoints to choosing among reasoning, retrieval, tool, and validation paths [89–91]. For example, an agent marketplace may need to compare services by expected dollars per verified outcome, not only by accuracy or uptime. Orchestrators may need cost-aware planning algorithms that decide when extra verification is worth its price, when cheaper models are sufficient, when cached evidence can be reused safely, and when additional delegation creates more cost than value. The unresolved research problem is how to make these choices auditable and contractible: a user should be able to understand why an agent spent a given amount, what quality improvement that spending purchased, and whether the final output satisfied the agreed service objective.

Taken together, hallucination, latency, and cost show why agentic services cannot be evaluated only with single-agent benchmarks [55]. The field needs benchmarks and contracts that measure end-to-end task graphs under uncertainty: coverage of validation harnesses, latency under realistic GPU and memory constraints, cost under branching delegation, and recovery behavior when the agent violates a policy or exhausts a budget. Cross-protocol interoperability will remain necessary even if the ecosystem partially converges [22, 45], but interoperability alone is not enough. The durable research challenge is to turn probabilistic reasoning into service behavior whose quality, delay, and cost can be measured, governed, and promised.

## 7 DISCUSSION AND CONCLUSION

This paper presented an integrative architectural perspective on the emerging Internet of Agentics (IoA), synthesizing protocols, systems, and services-computing developments into a unified framework for trustworthy and governable agentic ecosystems. Rather than treating agentic systems as isolated AI applications, the paper argued that the IoA represents a broader evolution of services computing toward interoperable cognitive service infrastructures.

The paper emphasizes architectural synthesis and servicelevel interpretation over empirical benchmarking, a positioning that aligns with the still-emerging standards landscape while necessarily limiting the scope of empirical claims. Protocol details and production claims often appear first in specifications, public repositories, engineering blogs, and foundation announcements, rather than in peer-reviewed venues. Accordingly, the paper treats such materials as signals of ecosystem direction and architectural intent rather than as conclusive evidence of adoption, performance, or long-term stability. Given the rapid evolution of the field, protocols, governance structures, and enterprise deployments—including MCP, A2A, Agent Cards, and AAIF initiatives—may continue to evolve between manuscript writing and publication [8, 12, 14, 34]. Some recent technical concepts, such as using Command Line Interface (CLI) to replace MCP <sup>4</sup> and You Only Live Once (YOLO) for coding agent permission control <sup>5</sup>, which are not yet standardized but widely adopted are not discussed here.

In addition, the service-computing analogies used throughout the paper are analytical aids rather than exact historical equivalences: they clarify recurring problems of discovery, composition, governance, and trust, while the uncertainty introduced by probabilistic reasoning and emergent multi-agent behavior remains distinctive. Legal liability, labor impacts, domain-specific compliance, and controlled benchmarking of protocol implementations require separate treatment as agent services move into high-stakes settings [20].

Within these boundaries, the Internet of Agentics should be understood as an effort to transform interoperable agents into trustworthy cognitive services. Its protocols are beginning to standardize how agents access tools and coordinate with peers. Its systems abstractions are beginning to make agents deployable as managed runtime components. Its service implications are best interpreted through the long history of service discovery, orchestration, and governance. The main unresolved issue is not whether agents can act, but whether they can act under constraints strong enough for production environments [13, 14, 16].

Seen in that light, this perspective paper argues that the central question is shifting from whether distributed autonomy is technically possible to whether it can be made trustworthy enough for production services. The unresolved question is whether distributed autonomy can be transformed into trustworthy autonomy through protocol verification, grounded inference architectures, bounded delegation, stronger runtime governance, and service models that treat agents as accountable actors rather than as stochastic search boxes. That transformation is the condition under which the IoA can become a durable services-computing paradigm rather than a transient integration wave [20, 55, 86].

4. Summary of the MCP vs. CLI debate on X this week. (Mar 14, 2026), https://x.com/randal\_olson/status/2032513098838597714

5. [FEATURE] Add YOLO mode toggle support to Claude Code CLI, https://github.com/anthropics/claude-code/issues/33046

## REFERENCES

[1] L. Wang, C. Ma, X. Feng, Z. Zhang, H. Yang, J. Zhang, Z. Chen, J. Tang, X. Chen, Y. Lin, W. X. Zhao, Z. Wei, and J. Wen, “A survey on large language model based autonomous agents,” Frontiers of Computer Science, vol. 18, no. 6, 2024. [Online]. Available: https://doi.org/10.1007/s11704-024-40231-1

[2] T. Guo, X. Chen, Y. Wang, R. Chang, S. Pei, N. V. Chawla, O. Wiest, and X. Zhang, “Large language model based multi-agents: A survey of progress and challenges,” arXiv preprint arXiv:2402.01680, 2024. [Online]. Available: https://arxiv.org/abs/2402.01680

[3] C. B˘adic˘a, A. B˘adic˘a, M. Ganzha, M. Ivanovi´c, M. Paprzycki, D. Seli¸steanu, and Z. Wrona, “Contemporary agent technology: LLM-driven advancements vs classic multiagent systems,” arXiv preprint arXiv:2509.02515, 2025. [Online]. Available: https://arxiv.org/abs/2509.02515

[4] S. Yao, J. Zhao, D. Yu, N. Du, I. Shafran, K. Narasimhan, and Y. Cao, “ReAct: Synergizing reasoning and acting in language models,” in Proceedings of the International Conference on Learning Representations (ICLR), 2023. [Online]. Available: https://arxiv.org/abs/2210.03629

[5] N. Shinn, F. Cassano, E. Berman, A. Gopinath, K. Narasimhan, and S. Yao, “Reflexion: Language agents with verbal reinforcement learning,” in Advances in Neural Information Processing Systems (NeurIPS), 2023. [Online]. Available: https://arxiv.org/abs/2303.11366

[6] S. Hong, M. Zhuge, J. Chen, X. Zheng, Y. Cheng, C. Zhang, J. Wang, Z. Wang, S. K. S. Yau, Z. Lin, L. Zhou, C. Ran, L. Xiao, C. Wu, and J. Schmidhuber, “MetaGPT: Meta programming for A multi-agent collaborative framework,” in Proceedings of the International Conference on Learning Representations (ICLR), 2024. [Online]. Available: https://arxiv.org/abs/2308.00352

[7] M. Cemri, M. Z. Pan, S. Yang, L. A. Agrawal, B. Chopra, R. Tiwari, K. Keutzer, A. Parameswaran, D. Klein, K. Ramchandran, M. Zaharia, J. E. Gonzalez, and I. Stoica, “Why do multi-agent LLM systems fail?” arXiv preprint arXiv:2503.13657, 2025. [Online]. Available: https://arxiv.org/abs/2503.13657

[8] Y. Kim, K. Gu, C. Park, C. Park, S. Schmidgall, A. A. Heydari, Y. Yan, Z. Zhang, Y. Zhuang, Y. Liu, et al., “Towards a science of scaling agent systems,” arXiv preprint arXiv:2512.08296, 2025.

[9] Q. Wu, G. Bansal, J. Zhang, Y. Wu, B. Li, E. Zhu, L. Jiang, X. Zhang, S. Zhang, J. Liu, A. H. Awadallah, R. W. White, D. Burger, and C. Wang, “AutoGen: Enabling next-gen LLM applications via multi-agent conversation,” arXiv preprint arXiv:2308.08155, 2023. [Online]. Available: https://arxiv.org/abs/2308.08155

[10] H. Chase et al., “LangGraph: Multi-actor applications with LLMs,” LangChain, Inc., 2024. [Online]. Available: https://github.com/langchain-ai/langgraph

[11] Y. Wang, S. Guo, Y. Pan, Z. Su, F. Chen, T. H. Luan, P. Li, J. Kang, and D. Niyato, “Internet of agents: Fundamentals, applications, and challenges,”

arXiv preprint arXiv:2505.07176, 2025. [Online]. Available: https://arxiv.org/abs/2505.07176

[12] Agentic AI Foundation, “Advancing agentic AI together,” Project homepage, 2026. [Online]. Available: https://aaif.io/

[13] Anthropic, “Introducing the model context protocol,” Anthropic News, Nov. 2024. [Online]. Available: https: //www.anthropic.com/news/model-context-protocol

[14] A2A Protocol Working Group, “Agent2agent (A2A) protocol specification, version 1.0.0,” The Linux Foundation, Tech. Rep., 2026. [Online]. Available: https://a2a-protocol.org/latest/specification/

[15] S. Deng, H. Zhao, Z. Wang, G. Cheng, P. Chen, W. Qian, Z. Ling, J. Yin, A. Y. Zomaya, and S. Dustdar, “Agentic services computing,” arXiv preprint arXiv:2509.24380, 2025. [Online]. Available: https://arxiv.org/abs/2509.24380

[16] M. P. Papazoglou, “Service-oriented computing: Concepts, characteristics and directions,” in Proceedings of the International Conference on Web Information Systems Engineering (WISE), 2003. [Online]. Available: https://doi.org/10.1109/WISE.2003.1254461

[17] R. T. Fielding, “Architectural styles and the design of network-based software architectures,” Ph.D. dissertation, University of California, Irvine, 2000. [Online]. Available: https://www.ics.uci.edu/\~fielding/ pubs/dissertation/top.htm

[18] T. Masterman, S. Besen, M. Sawtell, and A. Chao, “The landscape of emerging AI agent architectures for reasoning, planning, and tool calling: A survey,” arXiv preprint arXiv:2404.11584, 2024. [Online]. Available: https://arxiv.org/abs/2404.11584

[19] Z. Xi, W. Chen, X. Guo, W. He, Y. Ding, B. Hong, M. Zhang, J. Wang, S. Jin, E. Zhou, R. Zheng, X. Fan, X. Wang, L. Xiong, Y. Zhou, W. Wang, C. Jiang, Y. Zou, X. Liu, Z. Yin, S. Dou, R. Weng, W. Cheng, Q. Zhang, W. Qin, Y. Zheng, X. Qiu, X. Huang, and T. Gui, “The rise and potential of large language model based agents: A survey,” arXiv preprint arXiv:2309.07864, 2023. [Online]. Available: https://arxiv.org/abs/2309.07864

[20] N. Tomašev, M. Franklin, and S. Osindero, “Intelligent AI delegation,” arXiv preprint arXiv:2602.11865, 2026. [Online]. Available: https://arxiv.org/abs/2602.11865

[21] Agentic AI Foundation, “AAIF welcomes 97 new members as demand for open, collaborative agent standardization increases,” Press Release, Feb. 2026. [Online]. Available: https://www.linuxfoundation.org/ press/agentic-ai-foundation-welcomes-97-new-membe rs

[22] IBM Research / BeeAI, “Agent communication protocol (ACP) specification,” IBM Research / BeeAI, Tech. Rep., 2025. [Online]. Available: https://github.com/i-am-bee /acp

[23] OpenAI and AGENTS.md Contributors, “AGENTS.md: A simple, open format for guiding coding agents,” Open specification, 2025. [Online]. Available: https: //agents.md/

[24] Anthropic, “Extend Claude with skills,” https://code.c laude.com/docs/en/skills, 2025, accessed: 2026-03-11.

[25] AGNTCY / Cisco Outshift, “AGNTCY: Open-source agent interoperability framework,” 2025. [Online]. Available: https://docs.agntcy.org/

[26] A. S. Rao and M. P. Georgeff, “Bdi agents: From theory

to practice,” ICMAS, vol. 95, pp. 312–319, 1995. [Online]. Available: https://www.researchgate.net/publication/2 15544174

[27] M. Wooldridge and N. R. Jennings, “Intelligent agents: Theory and practice,” The Knowledge Engineering Review, vol. 10, no. 2, pp. 115–152, 1995. [Online]. Available: https://doi.org/10.1017/S0269888900008122

[28] K. P. Sycara, “Multiagent systems,” AI magazine, vol. 19, no. 2, pp. 79–92, 1998. [Online]. Available: https://doi.org/10.1609/aimag.v19i2.1370

[29] T. Finin, Y. Labrou, and J. Mayfield, “KQML as an agent communication language,” in Software Agents. MIT Press, 1997. [Online]. Available: https://dl.acm.org/doi /10.5555/267625.267637

[30] FIPA, “FIPA agent communication language specifications,” Foundation for Intelligent Physical Agents, Tech. Rep., 2002, fIPA00061. [Online]. Available: http://www.fipa.org/specs/fipa00037/SC00037J.html

[31] Foundation for Intelligent Physical Agents, “FIPA ACL message structure specification,” FIPA, Tech. Rep. SC00061G, 2002. [Online]. Available: http://www.fipa. org/specs/fipa00061/SC00061G.html

[32] R. G. Smith, “The contract net protocol: High-level communication and control in a distributed problem solver,” IEEE Transactions on Computers, vol. C-29, no. 12, pp. 1104–1113, 1980. [Online]. Available: https://doi.org/10.1109/TC.1980.1675516

[33] M. B. Juric, “Business process execution language for web services,” Packt Publishing, 2006, BPEL specification and orchestration patterns for web services. [Online]. Available: https://www.packtpub.com/product/busin ess-process-execution-language-for-web-services/9781 904811817

[34] Model Context Protocol Working Group, “Model context protocol specification,” LF Projects, LLC, Tech. Rep., 2025. [Online]. Available: https://modelcontextprotocol .io/

[35] B. Yan, Z. Zhou, L. Zhang, L. Zhang, Z. Zhou, D. Miao, Z. Li, C. Li, and X. Zhang, “Beyond self-talk: A communication-centric survey of LLM-based multiagent systems,” arXiv preprint arXiv:2502.14321, 2025. [Online]. Available: https://arxiv.org/abs/2502.14321

[36] M. Mohammadi, Y. Li, J. Lo, and W. Yip, “Evaluation and benchmarking of LLM agents: A survey,” arXiv preprint arXiv:2507.21504, 2025. [Online]. Available: https://arxiv.org/abs/2507.21504

[37] Anthropic, “Skills,” Claude Blog, 2025, accessed: 2026-04- 28. [Online]. Available: https://claude.com/blog/skills

[38] B. Zhang, K. Lazuka, and M. Murag, “Equipping agents for the real world with Agent Skills,” Anthropic Engineering, 2025, accessed: 2026-04-28. [Online]. Available: https://www.anthropic.com/engineering/eq uipping-agents-for-the-real-world-with-agent-skills

[39] Anthropic, “Agent Skills specification,” Agent Skills, 2025, accessed: 2026-04-28. [Online]. Available: https: //agentskills.io/specification

[40] N. Subramanya, “Claude Skills vs. MCP: A tale of two AI customization philosophies,” IntuitionLabs, 2025, accessed: 2026-03-11.

[41] J. MSV, “Model context protocol: A primer for the developers,” The New Stack, 2025, mCP architecture and comparison with REST-oriented integration approaches. [Online]. Available: https://thenewstack.io/model-con

text-protocol-a-primer-for-the-developers/

[42] R. Kosenko and soul.md Contributors, “soul.md: A portable, provider-agnostic specification for AI agent identity and behavior,” GitHub, 2025. [Online]. Available: https://github.com/rokoss21/soul.md

[43] Anthropic, “What are skills?” Claude Help Center, 2026, accessed: 2026-04-28. [Online]. Available: https://supp ort.claude.com/en/articles/12512176-what-are-skills

[44] ——, “Anthropic Skills repository,” GitHub repository, 2026, accessed: 2026-04-28. [Online]. Available: https: //github.com/anthropics/skills

[45] AGI, Inc., “Agent protocol: A universal standard for AI agent communication,” 2024. [Online]. Available: https://agentprotocol.ai/

[46] M. Welsh, D. Culler, and E. Brewer, “SEDA: An architecture for well-conditioned, scalable internet services,” in Proceedings of the 18th ACM Symposium on Operating Systems Principles (SOSP), 2001. [Online]. Available: https://doi.org/10.1145/502034.502057

[47] D. C. Schmidt, M. Stal, H. Rohnert, and F. Buschmann, Pattern-Oriented Software Architecture, Volume 2: Patterns for Concurrent and Networked Objects. Wiley, 2000. [Online]. Available: https://www.wiley.com/en-us/Patt ern-Oriented+Software+Architecture%2C+Volume+2-p -9780471606956

[48] B. Sun, Z. Huang, H. Zhao, W. Xiao, X. Zhang, Y. Li, and W. Lin, “Llumnix: Dynamic scheduling for large language model serving,” in Proceedings of the 18th USENIX Symposium on Operating Systems Design and Implementation (OSDI). USENIX Association, 2024. [Online]. Available: https://www.usenix.org/conferenc e/osdi24/presentation/sun-biao

[49] Y. Zhong, S. Liu, J. Chen, J. Hu, Y. Zhu, X. Liu, X. Jin, and H. Zhang, “DistServe: Disaggregating prefill and decoding for goodput-optimized large language model serving,” in Proceedings of the 18th USENIX Symposium on Operating Systems Design and Implementation (OSDI). USENIX Association, 2024. [Online]. Available: https://www.usenix.org/conferenc e/osdi24/presentation/zhong-yinmin

[50] W. Kwon, Z. Li, S. Zhuang, Y. Sheng, L. Zheng, C. H. Yu, J. E. Gonzalez, H. Zhang, and I. Stoica, “Efficient memory management for large language model serving with PagedAttention,” in Proceedings of the 29th Symposium on Operating Systems Principles (SOSP). ACM, 2023, pp. 611–626.

[51] A. Birgisson, J. G. Politz, U. Erlingsson, A. Taly, M. Vrable, and M. Lentczner, “Macaroons: Cookies with contextual caveats for decentralized authorization in the cloud,” in Proceedings of the 21st Network and Distributed System Security Symposium (NDSS). Internet Society, 2014.

[52] P. Birkholm-Buch, “Old wine, new bottles: Why MCP might succeed where UDDI and OData failed,” 2025. [Online]. Available: https://birkholm-buch.dk/2025/06 /05/old-wine-new-bottles-why-mcp-might-succeed-w here-uddi-and-odata-failed/

[53] P. Lewis, E. Perez, A. Piktus, F. Petroni, V. Karpukhin, N. Goyal, H. Küttler, M. Lewis, W. tau Yih, T. Rocktäschel, S. Riedel, and D. Kiela, “Retrieval-augmented generation for knowledgeintensive NLP tasks,” in Advances in Neural Information Processing Systems (NeurIPS), 2020. [Online]. Available:

https://arxiv.org/abs/2005.11401

[54] J. Dean, “Exciting directions for machine learning,” Sequoia AI Ascent, 2025, keynote presentation, June 2025.

[55] A. Jacovi, A. Wang, C. Alberti, C. Tao, J. Lipovetz, K. Olszewska, L. Haas, M. Liu, N. Keating, A. Bloniarz, C. Saroufim, C. Fry, D. Marcus, D. Kukliansky, G. S. Tomar, J. Swirhun, J. Xing, L. Wang, M. Gurumurthy, M. Aaron, M. Ambar, R. Fellinger, R. Wang, Z. Zhang, S. Goldshtein, and D. Das, “The FACTS grounding leaderboard: Benchmarking LLMs’ ability to ground responses to long-form input,” arXiv preprint arXiv:2501.03200, 2025. [Online]. Available: https://arxiv.org/abs/2501.03200

[56] Anthropic, “How Claude Code works,” https://code .claude.com/docs/en/how-claude-code-works, 2025, accessed: 2026-03-11.

[57] Block, Inc., “Goose architecture,” https://block.github .io/goose/docs/goose-architecture/, 2025, accessed: 2026-03-11.

[58] Microsoft, “Microsoft 365 copilot architecture and how it works,” Microsoft Learn, 2026, accessed: 2026-04-26. [Online]. Available: https://learn.microsoft.com/en-us/ microsoft-365/copilot/microsoft-365-copilot-architectur e

[59] ——, “Data, privacy, and security for microsoft 365 copilot,” Microsoft Learn, 2026, accessed: 2026-04-26. [Online]. Available: https://learn.microsoft.com/en-us/ microsoft-365/copilot/microsoft-365-copilot-privacy

[60] ——, “Overview of microsoft graph,” Microsoft Learn, 2026, accessed: 2026-04-26. [Online]. Available: https: //learn.microsoft.com/en-us/graph/overview

[61] Anthropic, “Claude for work,” Anthropic, 2026, accessed: 2026-04-27. [Online]. Available: https://www.anthropic. com/claude/work

[62] Google Workspace, “Gemini for google workspace,” Google Workspace, 2026, accessed: 2026-04-27. [Online]. Available: https://workspace.google.com/solutions/ai/

[63] Microsoft, “Microsoft 365 copilot connectors overview,” Microsoft Learn, 2026, accessed: 2026-04-26. [Online]. Available: https://learn.microsoft.com/en-us/graph/co nnecting-external-content-connectors-overview

[64] M. Lanham, “MCP + A2A: The TCP/IP moment for AI agents,” Medium, Feb. 2026. [Online]. Available: https://medium.com/@Micheal-Lanham/mcp-a2a-the -tcp-ip-moment-for-ai-agents-bf1927112b07

[65] S. Chaisiri, B.-S. Lee, and D. Niyato, “Optimization of resource provisioning cost in cloud computing,” IEEE Transactions on Services Computing, vol. 5, no. 2, pp. 164–177, 2012. [Online]. Available: https://doi.org/10.1 109/TSC.2011.7

[66] S. Chatterjee, R. Ladia, and S. Misra, “Dynamic optimal pricing for heterogeneous service-oriented architecture of sensor-cloud infrastructure,” IEEE Transactions on Services Computing, vol. 10, no. 2, pp. 203–216, 2017. [Online]. Available: https://doi.org/10.1109/TSC.2015.2 453958

[67] D. Saxena and A. K. Singh, “An oversubscription and service pricing exploitation-based profit maximization framework for industry cloud resource management,” IEEE Transactions on Services Computing, vol. 17, no. 5, pp. 2041–2053, 2024. [Online]. Available: https://doi.org/10.1109/TSC.2024.3445379

[68] X. Liu, M. Dong, K. Ota, P. Hung, and A. Liu, “Service

pricing decision in cyber-physical systems: Insights from game theory,” IEEE Transactions on Services Computing, vol. 9, no. 2, pp. 186–198, 2016. [Online]. Available: https://doi.org/10.1109/TSC.2015.2449314

[69] S. Feng, Z. Xiong, D. Niyato, P. Wang, S. S. Wang, and S. X. Shen, “Joint pricing and security investment in cloud security service market with user interdependency,” IEEE Transactions on Services Computing, vol. 15, no. 3, pp. 1461–1472, 2022. [Online]. Available: https://doi.org/10.1109/TSC.2020.2996382

[70] M. L. Puterman, Markov decision processes: discrete stochastic dynamic programming. John Wiley & Sons, 2014.

[71] S.-Y. Hwang, C.-C. Hsu, and C.-H. Lee, “Service selection for web services with probabilistic qos,” IEEE transactions on services computing, vol. 8, no. 3, pp. 467–480, 2014.

[72] C. Chen and A. Helal, “Device integration in soda using the device description language,” in Proceedings of the IEEE/IPSJ Symposium on Applications and the Internet, Seattle, Washington, USA, July 2009.

[73] S. de Deugd, R. Carroll, K. Kelly, B. Millett, and J. Ricker, “Soda: Service oriented device architecture,” IEEE Pervasive Computing, vol. 5, no. 3, pp. 94–96, 2006.

[74] S. Helal, “The monkey, the ant, and the elephant: Addressing safety in smart spaces,” Computer, vol. 53, no. 5, pp. 73–76, 2020.

[75] T. W. Malone and K. Crowston, “What is coordination theory and how can it help design cooperative work systems?” ACM Computing Surveys, vol. 26, no. 1, pp. 87–119, 1994.

[76] M. S. Bernstein, J. Brandt, R. C. Miller, and D. R. Karger, “Crowds in two seconds: Enabling realtime crowdpowered interfaces,” UIST, 2011.

[77] M. Wooldridge, An Introduction to MultiAgent Systems. Wiley, 2009.

[78] W. M. P. van der Aalst, “Workflow management: Models, methods, and systems,” MIT Press, 2003.

[79] S. Amershi et al., “Guidelines for human-ai interaction,” in CHI, 2019.

[80] V. Cardellini, E. Casalicchio, V. Grassi, and F. Lo Presti, “Flow-based service selection for web service composition supporting multiple qos classes,” IEEE Transactions on Services Computing, 2011.

[81] R. Buyya et al., “Market-oriented cloud computing: Vision, hype, and reality,” Future Generation Computer Systems, 2009.

[82] R. N. Calheiros et al., “Workload prediction using arima model and its impact on cloud applications’ qos,” IEEE Transactions on Cloud Computing, 2015.

[83] Z. Zheng, Y. Zhang, and M. R. Lyu, “Investigating qos of real-world web services,” IEEE transactions on services computing, vol. 7, no. 1, pp. 32–39, 2012.

[84] Sequoia Capital, “Ai’s \$600b question,” 2024.

[85] Goldman Sachs, “Will ai eat software?” 2024.

[86] W. Ren, R. W. Beard, and E. M. Atkins, “Information consensus in multivehicle cooperative control,” IEEE Control Systems Magazine, vol. 27, no. 2, pp. 71–82, 2007. [Online]. Available: https://doi.org/10.1109/MCS.2007 .338264

[87] Microsoft, “How data is protected and audited in microsoft 365 and microsoft 365 copilot,” Microsoft Learn, 2026, accessed: 2026-04-26. [Online]. Available: https://learn.microsoft.com/en-us/microsoft-365/copil ot/microsoft-365-copilot-architecture-data-protection-a

uditing

[88] ——, “Microsoft purview data security and compliance protections for generative ai apps,” Microsoft Learn, 2025, accessed: 2026-04-26. [Online]. Available: https: //learn.microsoft.com/en-us/purview/ai-microsoft-p urview

[89] Z. Zheng, H. Ma, M. R. Lyu, and I. King, “QoSaware web service recommendation by collaborative filtering,” IEEE Transactions on Services Computing, vol. 4, no. 2, pp. 140–152, 2011. [Online]. Available: https://doi.org/10.1109/TSC.2010.52

[90] M. Tang, Y. Jiang, J. Liu, and X. Liu, “Locationaware collaborative filtering for QoS-based service recommendation,” in Proceedings of the IEEE International Conference on Web Services, 2012, pp. 202–209. [Online]. Available: https://doi.org/10.1109/ICWS.2012.61

[91] S.-Y. Hwang, C.-C. Hsu, and C.-H. Lee, “Service selection for web services with probabilistic QoS,” IEEE Transactions on Services Computing, vol. 8, no. 3, pp. 467–480, 2015. [Online]. Available: https://doi.org/10.1 109/TSC.2014.2338851