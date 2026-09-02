# SoK: When Safe Agents Fail Together: The Security of Multi Agent LLM Systems

Rui Yang¹, Junjie Xu², Zhengyu Liu¹, Neil Fendley¹, Yang Hong¹, Ziyang Li¹, Yinzhi Cao¹

¹ Johns Hopkins University ² Nanyang Technological University

ryang54@jh.edu, andrexu736269@gmail.com, zliu192@jhu.edu,

neil.fendley@jhuapl.edu, yhong51@jh.edu, ziyang@cs.jhu.edu, yinzhi.cao@jhu.edu

## Abstract

Safe agents can fail together. Multi-agent LLM systems (MAS) move information, state, decisions, and authority across principal boundaries, creating failures that local checks may miss. Without an execution-level view, a multi-agent setting can easily be mistaken for evidence of a genuinely multi-agent security effect. We thus systematize MAS security through an execution-centered analysis of 197 works, covering six interaction interfaces, four adversary positions, seven system-level risks, and eight recurring attack paths. We introduce an A-I-R framework that organizes attacks by adversary position, interaction interface, and resulting system-level risk, unifying otherwise fragmented attack mechanisms across MAS. We organize defenses through a five-part contract covering path target, observation, intervention, trust boundary, and recovery, and identify path closure and recovery as key challenges. We audit 44 evaluation and benchmark works and identify open challenges in isolating interaction effects, designing comparable and diagnostic metrics, supporting reuse across MAS designs, and evaluating open-system operation. Together, these findings motivate an interaction-aware view of MAS security: trace attacks end to end, test whether defenses close those paths, and evaluate system-level effects with appropriate counterfactuals.

## 1 Introduction

Agents that are safe when used individually can fail when used together in a MAS. Benign prompt fragments can become harmful when combined [11]; truthful reports can steer a group toward a false belief [55]; and attacker-controlled content can pass through honest specialists to a privileged tool [62]. These failures arise through interactions among the separately addressable participants of the MAS, which we call principals. Messages propagate influence between principals, a shared state preserves propagated influence, aggregation combines local outputs from different principals, and delegation transfers authority across boundaries [62, 79, 93, 95].

We therefore study the end-to-end execution of MAS, where interactions may leave a failure largely unchanged, amplify it, create a failure through composition, or require defining a security property that exists only when principals interact.

Existing surveys cover many important threats, approaching from different perspectives. Agent-security surveys organize risks around models, prompts, memory, tools, workflows, interfaces, and environments [34, 50, 71, 181], while MASfocused surveys emphasize communication, trust, topology, governance, and collective behavior [49, 72, 132, 139]. What remains unclear is how an attack moves through a multiagent system: where it enters, which interaction interfaces it uses, and what system-level security failure it finally causes. Without this execution view, simply observing a failure in a multi-agent setting can be mistaken for evidence that the failure depends on interaction. This distinction is increasingly important as the field grows, because closely related studies still use different threat models, units of analysis, metrics, and evidence standards. A common way to describe and evaluate these security effects is therefore needed.

We address this gap by treating the complete multi-agent execution as the unit of analysis. Our core representation traces adversarial influence from its starting position (A), through one or more interaction interfaces (I), to a systemlevel security failure (R), yielding the A → I → R framework. Eight configuration dimensions (C1–C8) describe the conditions under which such an execution occurs; they condition a path. Across the corpus, we group recurring A → I → R executions into eight non-exclusive attack paths (P1–P8). These paths are synthesis shorthands for recurring execution patterns, not an additional taxonomy of security outcomes. The same execution view is then used to analyze where defenses can intervene and how security claims should be evaluated. Figure 1 summarizes this organization. Our contributions are therefore fourfold:

• We define multi-agent security around end-to-end execution, providing a common basis for deciding when interaction matters to a security claim. This executioncentered view provides a common model for connecting system configuration, interaction paths, security effects, defenses, and evaluation.

![](images/fea3bdcc7466560d80339db56f655eccc4eaf54948718db42572cee16547287e.jpg)  
Figure 1: Unified execution view of MAS security. C1–C8 condition the execution. A1–A4 identify where adversarial influence begins, I1–I6 identify the transitions it crosses, and R1–R7 identify the resulting system-level failure. D1–D4 denote defensive control primitives applied along the execution.

• We introduce the A → I → R framework and synthesize seven system-level risk families and eight recurring attack paths. This structure follows adversarial influence from entry to consequence while showing how multiple risks and paths can compose within one execution.

• We organize defenses with a five-part contract and map where each control primitive can interrupt an attack path. This exposes gaps in path closure, observer adequacy, intervention timing, mediation, and recovery.

• We audit 44 evaluation and benchmark works and identify open problems in interaction-specific comparisons, designing comparable and diagnostic metrics, benchmark portability, and open-system evaluation.

## 2 Overview

This section defines the scope of the SoK, positions our analysis relative to prior surveys, and summarizes the evidence base used for the synthesis.

## 2.1 Scope

System boundary. We define an MAS by separately addressable principals: at least two participants whose outputs can affect one another through messages, shared state, collective decisions, or delegated actions [56, 178]. We call each such participant a principal. Principals may share the same LLM backend or use multiple models internally, so the system boundary is defined by interaction rather than model count [71, 178]. Debate systems, orchestrated teams, swarms, hierarchies, and A2A networks are therefore in scope when their participants act as separate principals [46, 86, 111]. Routing, ensembling, repeated sampling, and planner–critic–verifier pipelines are out of scope when all steps remain within one principal [71]. Security boundary. For the synthesis, we treat a result as evidence of a MAS-specific security effect when interaction changes a security-relevant outcome or when the protected property itself depends on interaction. Prompt injection, for example, may remain local, propagate across principals, persist in shared state, influence a collective decision, or gain authority through delegation [62,79,93,95]. Authentication and authorization are likewise interaction-level concerns when trust or authority crosses principal boundaries [62, 140, 173]. Changes in accuracy, agreement, or behavior, including behavioral drift, count as security failures only when they violate a stated protected property or boundary. Results that reproduce an agent vulnerability in an MAS without a material interaction effect may be retained as comparative context, but they do not by themselves establish a MAS-specific causal claim.

Interaction can play four roles in a security result. An inherited effect remains largely unchanged when the relevant relation is removed; we use this as a comparison label, not evidence of interaction dependence. An interaction-amplified effect remains possible, but interaction changes likelihood, reach, persistence, privilege, or impact [48,95,151,182]. A composition-induced effect appears only when principals or steps are combined [11, 75, 124]. A structurally multi-agent effect concerns a property defined by relations among principals, such as collusion, coalition formation, or quorum manipulation [105, 111, 178]. For inherited, amplified, and composition-induced effects, relationremoval counterfactuals can distinguish whether the effect persists, changes in magnitude, or disappears when the protected property remains meaningful. Structurally multi-agent effects require comparisons that preserve the relation. These labels describe how interaction contributes to a security effect and what evidence is needed to support that claim.

<table><tr><td></td><td colspan="5">End-to-end Threat-path-</td></tr><tr><td>Work</td><td>Interaction execution counterfactual</td><td>trace</td><td>protected effect</td><td>Defense Outcome-stage assumptions evidence</td><td></td></tr><tr><td>Hammond et al. [49]</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Schroeder de Witt et al. [139]</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Sun et al. [144]</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Ko et al. [72]</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Raza et al. [132]</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Rafe et al. [129]</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Vangalapat et al. [153]</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Kale et al. [68]</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Tanveer [148]</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Kong et al. [73]</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Reid et al. [133]</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Shi et al. [140]</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Nguyen et al. [115]</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Ours</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Systematic</td><td colspan="2">Partial</td><td colspan="3">Not systematically treated</td></tr></table>

Table 1: Coverage of prior multi-agent security syntheses across our analytical dimensions, using the coding criteria stated in Section 2.2.

## 2.2 Related Work

Prior surveys, reviews, and synthesis-style studies cover complementary parts of agent and multi-agent security. Broad agent-security surveys organize threats and defenses around models, prompts, memory, tools, workflows, interfaces, and environments [34,50,71,181]. MAS-focused syntheses study communication, trust, topology, coordination, authorization, governance, and system-level risks [49, 72, 73, 132, 139, 140, 144]. More focused or adjacent studies examine orchestration and verification, trustworthy coordination, privacy-preserving retrieval, governed risk analysis, and the coverage of existing security frameworks [68, 115, 129, 133, 148, 153].

These works have different goals and scopes, and we do not treat the comparison as a ranking of their quality. Our distinction is the unit of analysis. Rather than organizing security primarily by component, attack name, design pattern, or governance concern, we organize it around cross-principal execution. The same execution model connects where adversarial influence enters, which interaction interfaces it crosses, which protected system-level effect it reaches, where a defense can intervene and under what assumptions, and what evidence supports the claim. Table 1 compares prior syntheses along these analytical dimensions. We code a dimension as systematic only when it is used as an explicit organizing or evaluation axis, as partial when it is substantively discussed or illustrated but not applied systematically, and as not systematically treated otherwise. These markers describe analytical coverage, not the quality of the cited work.

Literature Selection. We assembled the corpus through searches across major security, AI, and NLP venues and scholarly indexes, supplemented by backward and forward citation chasing. After deduplication and review against Section 2.1, the final corpus contains 197 works. We divide the corpus into two evidence sets with different roles in the synthesis. Set 1 contains 115 works that were peer reviewed or had received at least ten citations by the frozen review cutoff. We use these works to characterize the current MAS-security landscape and to support the main synthesis in the paper. Set 2 contains 82 additional in-scope works that point to emerging mechanisms, system settings, and security questions that may shape the next stage of the field. We use these works more selectively, primarily to highlight extensions of established attack paths, defenses, and evaluation settings. The literature corpus and associated artifacts will be released publicly with the final version of the paper.

## 3 System Model and Interaction Surface

Building on the scope above, we separate the conditions around an execution from the transitions within it. Configuration dimensions (C1–C8) describe how the system is arranged and therefore condition the reach, persistence, and impact of an attack. Interaction interfaces (I1–I6) identify the security-relevant transitions through which information, state, decisions, authority, or security controls cross boundaries. The interfaces later form the middle of the A → I → R execution path; the configuration dimensions provide context for interpreting that path.

## 3.1 Configuration Dimensions

The eight configuration dimensions describe the conditions that shape an MAS execution. The same attack may therefore have different reach, persistence, or impact across configurations, and several dimensions may matter at once.

C1: Communication topology. Communication topology describes which principals can communicate directly and which paths connect them. MAS studies commonly consider complete, chain or line, ring or circle, tree, and star topologies, as well as other system-specific communication graphs [48, 166, 182]. Topology changes path length, connectivity, and centrality, shaping how far adversarial influence can travel and which principals become important intermediaries.

C2: Protocol semantics. Protocol semantics describe how principals exchange information and what security context is preserved across each exchange. MAS communication ranges from free-form natural-language messages to messages with fixed roles or formats and agent-to-agent protocols [52,62,86]. Security depends on whether identity, origin and provenance, task context, and authorization information remain available across principal boundaries. End-to-end enforcement may require the security context needed by downstream checks to remain available across the protocol chain. Each exchange may be valid on its own, yet the full execution can still become unsafe if translation, summarization, state updates, or delegation lose the origin, purpose, provenance, or authority needed by a later decision [1, 62].

C3: Principal composition. Principal composition describes which principals participate, what roles they play, and how independent they are. Systems may use homogeneous or heterogeneous models and generalist or specialist roles. Shared models, prompts, or evidence sources can create correlated behavior even when several principals participate [48, 105].

C4: Coordination mechanism. Coordination mechanisms describe how principals combine their contributions into a system-level decision or action. Common mechanisms include debate, voting, judge or aggregator models, consensus, and orchestrator-mediated coordination [55, 93, 111]. Their security depends on how malicious or correlated contributions affect the collective outcome.

C5: State architecture. State architecture describes where state is stored, who can access it, and how long it persists. Distinctions include principal-local versus shared state, ephemeral versus persistent state, and whether state is copied or inherited across principals [21, 95, 187]. These choices determine which principals can read or modify state and whether derived state can later be traced, revoked, or repaired.

C6: Membership and trust. Membership and trust describe which principals may participate and which relationships the system treats as trusted. Membership may be closed or open, static or dynamic, with persistent or ephemeral identities [21, 86, 89]. Dynamic systems must update admission, trust, and revocation as principals join, leave, or change roles. Membership is part of the system’s security state. In open or self-spawning systems, the population can change during execution, so the fraction of malicious principals is not necessarily fixed. Robustness claims based on that fraction therefore depend on how admission, identity, spawning, and revocation define who counts as a participant [21, 89].

C7: Authority placement. Authority placement describes which principals hold or can delegate privileged capabilities. Relevant authority includes tool permissions, credentials, resource budgets, delegation rights, and routing or control-plane privileges [1, 62, 173]. Security impact depends not only on which principals are reachable, but also on what authority is reachable through them.

C8: Oversight architecture. Oversight architecture describes who can observe an execution and how much of it they can see. Monitoring may be local, distributed across principals, or centralized with graph or trace visibility. Broader visibility can improve detection and attribution, but it may also concentrate sensitive state and control in the oversight plane. A defense can only make claims about interactions that fall within its view. A local observer may miss behavior that spans edges, principals, shared state, or a long execution trace, while a broader observer may recover more of the path at the cost of stronger trust and privacy assumptions [40, 162, 193].

Limitations and Open Challenges. Configuration dimensions often change together. An architectural change may simultaneously alter topology, protocol behavior, state sharing, authority, and oversight [48, 95]. Security results should therefore state which C1–C8 dimensions vary across configurations. Controlled comparisons should isolate one dimension when possible and report coupled changes otherwise. More broadly, interaction does not always make a system less secure. The same architecture may increase or reduce risk under different models, protocols, roles, or protected properties.

## 3.2 Interaction Interfaces

The six interaction interfaces mark security-relevant transitions during an MAS execution. I1–I5 describe how inputs, messages, state, authority, and collective decisions move through the system. I6 captures how the system observes and intervenes in those transitions.

I1: Admission. Admission covers what is allowed to enter the MAS boundary, including users, new principals, external data, messages, and artifacts. Relevant context includes identity, provenance, role, initial trust, and participation rules [62, 79, 86]. An input that is safe at entry may still become harmful after later interactions give it more reach, persistence, or authority.

I2: Message transfer. Message transfer begins once information moves between admitted principals. Security depends on who receives the message, in what order and context, and whether sender identity, provenance, and task purpose remain clear across the transfer [52, 79, 176]. Routing can therefore change which principals a message can influence.

I3: State propagation. State propagation occurs when information from one interaction is written into state and later affects another principal or execution. This includes memories, summaries, plans, histories, and derived artifacts [21,95,187]. Unlike direct message transfer at I2, I3 can preserve and reintroduce influence after the original interaction has ended.

I4: Authority transfer and action. Authority transfer and action cover delegation and use of privileged capabilities. A message, state update, or decision may lead a principal to use a tool, credential, resource, or delegated capability [1, 36, 62]. Security depends on whether the action stays within the authority, purpose, resource, and scope that were originally allowed. I4 therefore marks the transition from information or delegated authority to a privileged action; attacks exploit this interface when adversarial influence reaches that action. I5: Collective commitment. Collective commitment occurs when local outputs become a group-level decision, such as a vote, consensus, final answer, termination decision, or action trigger. Security depends on whether that decision is based on valid and sufficiently independent evidence [55, 111, 178]. Shared models, context, sources, or attacker influence can make agreement overstate the evidence.

I6: Detection and intervention. Detection and intervention cover security checks and responses applied during an execution. These include monitoring, blocking, isolation, revocation, rollback, and recovery across I1–I5 [40, 141, 162]. A defense can act only on the information it can observe and the parts of the execution it can control. Intervention must also occur before the protected effect becomes irreversible when prevention is the goal.

In short, C1–C8 are conditioning descriptors, whereas I1– I6 are transitions within an execution. Section 4 completes the execution record by adding the adversary position A and resulting risk R, forming A → I → R. We reuse this single representation for attacks, defenses, and evaluation.

## 4 Attack Landscape

Attacks become system-level failures when adversarial influence crosses interaction boundaries under a particular system configuration. We first define adversary positions and seven system-level risks, then synthesize eight recurring attack paths. We close with cross-cutting findings and the evidence gaps that remain for interaction-dependent harm.

## 4.1 Threat Model

An MAS threat model must state the adversary’s position, capabilities, persistence, and the protected party and property. Figure 1 distinguishes four starting positions. An A1 external adversary controls content or services outside the MAS. An A2 user-level adversary acts through an authorized user interface. An A3 member adversary controls one or more participating principals. An A4 infrastructure adversary controls shared components such as the coordinator, state store, router, telemetry service, or defense plane. Adversary position alone does not determine capability. Attackers at the same position may differ in whether they can alter messages, routing, shared state, aggregation, membership, tools, or delegated authority [46, 52, 62, 93, 176]. Coalition size and persistence further shape how far that influence can propagate or coordinate [105, 111, 143]. A complete threat model must therefore specify not only where the adversary is, but also what it can control and which security property is protected. Limitations and Open Challenges. Many evaluated settings keep these variables fixed throughout a run, while deployed MAS may spawn principals, reroute communication, update trust, and delegate or revoke authority during execution. Threat models for such systems should track changes in membership, communication, state lineage, and reachable authority, rather than describe only the initial attacker state. In open or self-spawning systems, membership is itself part of the security state. Fraction-based robustness claims are incomplete unless admission and identity mechanisms define who counts toward the population and how spawned or ephemeral principals inherit trust [21, 86, 89, 173].

![](images/a8adf80fb672ae2af0d61f3a2b0b9339e195ce498df3714011d4158381ef8d68.jpg)  
Figure 2: Example: covert coordination. Two compromised principals use a protocol-valid message to coordinate covertly, so their apparent agreement cannot be treated as independent evidence.

## 4.2 System-Level Risk Families

Risk families describe the system-level security failure reached by an attack. We group consequences into seven nonexclusive families. The same risk may arise through different interfaces, and one execution may trigger several risks.

R1. Containment failure. Containment fails when adversarial influence escapes the principal or context in which it entered and remains effective elsewhere. Agent Smith, Prompt Infection, and the Wolf Within show spread across principals, while Flooding shows manipulated knowledge spreading and persisting through interaction and memory, and CORBA shows that blocking behavior can itself become contagious [46, 65, 79, 147, 195]. Topology and shared state determine how far that influence travels and whether it remains after the original source disappears [95, 182].

R2. Loss of independence. Loss of independence occurs when principals expected to provide independent evidence become strategically or statistically dependent. Secret Collusion and CoMet establish covert coordination channels, while hidden-role and steganographic studies show that such dependence may be difficult to observe [105, 111, 170, 172]. As illustrated in Fig. 2, protocol-valid messages can hide coordination between compromised principals, making their agreement unreliable as independent evidence. Dependence can also arise from a shared model, prompt, retriever, or poisoned source without explicit collusion. Agreement alone therefore does not establish independent support.

R3. Collective-decision integrity failure. Collectivedecision integrity fails when adversarial influence changes a protected vote, consensus, commitment, quorum, or other group decision. Among Us studies malicious contributions, manipulation and persuasion attacks show how one member can steer otherwise benign participants, and Lying with Truths shows that individually truthful reports can still induce a misleading collective belief [55, 76, 93, 178]. The protected object is the group decision, not merely the correctness of each message.

R4. Trajectory and goal integrity failure. Trajectory or goal integrity fails when interaction redirects a protected objective, policy, or execution trajectory beyond an authorized boundary. CRDA measures content-risk drift under adversarial multiagent interaction, while persuasion studies provide more direct evidence that peer interaction can redirect later behavior [76, 98]. FlowSteer studies workflow steering, while the LLM Drift software artifact provides an evaluation framework for behavioral drift and latent-state studies examine longer or stateful changes in multi-agent execution [19,82,138]. These forms of evidence do not all establish the same trajectorysecurity claim. Unlike ordinary accuracy degradation, an R4 claim needs an explicit protected objective and a boundary after which the deviation becomes unauthorized. Direct evidence for this risk remains thinner than for propagation or collective manipulation.

R5. Cross-principal confidentiality breach. A crossprincipal confidentiality breach occurs when protected information reaches an unauthorized principal, coalition, service, or observer. MASLeak studies extraction of multi-agent intellectual property, including prompts, tools, and architecture, while AgentLeak measures privacy leakage across internal coordination channels; CIA shows that communication topol ogy itself can be inferred [160, 166, 174]. Compositional privacy further shows that several locally permissible views may reveal protected information when combined [124]. The protected object therefore extends beyond user secrets to internal context, state, workflow information, and system structure.

R6. Transitive authorization failure. A transitive authorization failure occurs when state, instructions, or delegated authority cross principal boundaries and produce effects outside their original purpose or scope. Control-flow hijacking follows attacker-controlled influence through trusted handoffs into privileged tools, while cascading instruction and software-team attacks show similar composition across roles and artifacts [1, 62, 152, 163]. The central failure is loss of origin, purpose, or scope along the delegation chain [173].

R7. Availability and resource-isolation failure. Availability or resource isolation fails when shared communication, execution, or control resources allow one adversarial path to degrade the service available to other principals. CORBA provides direct evidence through contagious blocking [195]. Other studies expose resource and control-plane coupling, but systematic evidence for adversarial resource isolation remains limited. The defining property is cross-principal loss of availability or resource isolation, not ordinary latency, lower accuracy, or efficiency loss.

Risk interactions. The risks can form chains rather than isolated outcomes. Propagation or persistent influence (R1) may redirect later behavior (R4), become accepted as collective evidence (R3), acquire delegated authority (R6), and eventually cross a confidentiality boundary (R5). Loss of independence (R2) can accelerate this chain, while resource interference (R7) can weaken observation or intervention at I6. This is a cross-paper synthesis rather than a claim that one study

<table><tr><td>Path</td><td>Typical A → I → R</td><td>Key C</td><td>Primary evidence</td><td>Emerging evidence</td></tr><tr><td>P1. Admission / composition</td><td> $\mathrm { A } 1 / \mathrm { A } 2  \mathrm { I } 1 / \mathrm { I } 2 / \mathrm { I } 5  \mathrm { R } 3 / \mathrm { R } 6$ </td><td>C2/C4/C7</td><td>[7,9,11,75,79,124,145,147,152,163]</td><td>[74,82,92,127,197]</td></tr><tr><td>P2. Propagation / persistence</td><td> $\mathrm { A } 1 / \mathrm { A } 3  \mathrm { I } 1 / \mathrm { I } 2 / \mathrm { I } 3  \mathrm { R } 1$ </td><td>C1/C5/C6</td><td>[1,46,65,79,95,107,128,145,147,182,195]</td><td>[21,37,60,82,91,156,187,197]</td></tr><tr><td>P3. Routing / role manipulation</td><td> $\mathrm { A } 3 / \mathrm { A } 4  \mathrm { I } 1 / \mathrm { I } 2 / \mathrm { I } 5  \mathrm { R } 1 / \mathrm { R } 3 / \mathrm { R } 6$ </td><td></td><td>C1/C2/C3/C6[36,52,70,86,166,176,182]</td><td>[10,19,32,37,82,83,143]</td></tr><tr><td>P4. Coordination / manipulation</td><td> $\mathbf { A } 3  \mathbf { I } 2 / \mathbf { I } 5  \mathbf { R } 2 / \mathbf { R } 3$ </td><td>C3/C4/C6</td><td>[6,55,63,75,76,93,105,111,117,150,170,172,178][5,15,32,37,78,94,113,127,137,143]</td><td></td></tr><tr><td>P5. Peer / state steering</td><td> $\mathrm { A } 3  \mathrm { I } 2 / \mathrm { I } 3 / \mathrm { I } 5  \mathrm { R } 4$ </td><td>C3/C4/C5</td><td>Sparse; adjacent evidence [55,76,98, 106]</td><td>[19,82,92]</td></tr><tr><td>P6. Cross-principal leakage</td><td> $\mathrm { A } 2 / \mathrm { A } 3 / \mathrm { A } 4  \mathrm { I } 2 / \mathrm { I } 3 / \mathrm { I } 4  \mathrm { R } 5$ </td><td>C1/C5/C7</td><td>[17,62,66,95,111,124,145,160,166,172,174]</td><td>[14,112,171,198]</td></tr><tr><td>P7. Delegation / authority</td><td> $\mathrm { A } 1 / \mathrm { A } 3  \mathrm { I } 2 / \mathrm { I } 3 / \mathrm { I } 4  \mathrm { R } 6$ </td><td>C2/C6/C7</td><td>[1,36,62,86,100,120,128,152,163,173]</td><td>[10,21,60,91,112,156,187,190,197]</td></tr><tr><td>P8. Resource / control plane</td><td> $\mathrm { A } 3 / \mathrm { A } 4  \mathrm { I } 2 / \mathrm { I } 4 / \mathrm { I } 6  \mathrm { R } 7$ </td><td>C1/C7/C8</td><td>Sparse; direct evidence [195]</td><td>[60,96]</td></tr></table>

Table 2: Representative evidence across recurring attack paths; C1–C8 denote conditioning configuration dimensions.

demonstrates the full chain.

Limitations and Open Challenges. Risk labels identify consequences, not causes, and they should not be inferred from attack names alone. The same risk can arise through several interaction paths, while one execution can violate several protected properties at once. Evaluations should therefore state the protected property and verify the downstream consequence before assigning a system-level risk.

## 4.3 Attack Paths from Adversary to Risk

Attack paths connect an adversary’s starting position to a security consequence through one or more interaction interfaces. Across the corpus, we identify eight recurring $A  I  R$ paths, summarized in Table 2. These paths describe recurring mechanisms, so one execution may contain several paths.

Running example. Control-flow hijacking shows why the full execution matters. In our coding of CFH-Hard, the core path is A1 → I1 → I2 → I4: external content enters the system, is relayed through trusted principals, and reaches a privileged coder or emailer, causing an R6 authorization failure [62]. Replanning may also involve I3 when an intermediate plan or execution state persists into downstream execution. Each step may appear valid on its own: the input may be allowed, the intermediate task may look reasonable, and the delegated action may be valid for its immediate sender. The violation becomes clear only when provenance and authority are traced across the full execution.

P1. Admission and compositional injection. Locally acceptable inputs can become unsafe when combined. Prompt Infection and the Wolf Within show malicious influence crossing principal boundaries, while conjunctive and semantic-intent fragmentation split unsafe intent across pieces that appear less suspicious on their own [7,11,79,147]. Similar effects appear in software teams, RAG-based debate, and ACIARena, where prompts, retrieved content, or generated artifacts are combined later in the execution [9, 75, 152, 163]. Recent work extends this pattern to debate jailbreaks, distributed backdoors, multimodal injection, poisoned retrieval, and workflow steering [74, 82, 92, 118, 127, 197]. The common problem is that harmful meaning or authority may emerge only after several locally acceptable steps. Conjunctive Prompt Attacks directly isolates this compositional effect with single-fragment and combined-fragment controls [11].

P2. Propagation and persistence. Interaction can turn a local compromise into a system-wide or persistent failure. Agent Smith, Troublemaker, Flooding, Prompt Infection, the Wolf Within, and CORBA spread malicious influence or disruption through communication, retrieval, generated content, or agent state [46, 65, 79, 107, 125, 147, 183, 195]. NetSafe and memory studies further show that topology affects reach, while shared state affects persistence [95, 182, 185]. AgentWorm extends propagation across communication, state, and downstream actions, while When Child Inherits considers spawned principals and inherited state or authority [21, 104, 184, 187]. Propagation is therefore not only about message delivery. It also depends on what remains in the system after the original source is removed. Recovery is much less studied than spread, especially when memories, artifacts, delegations, or spawned principals remain after the initial compromise [21, 95, 187].

P3. Routing and role manipulation. The communication path itself can become an attack surface. Communication red teaming and MAST intercept or rewrite messages between principals without compromising endpoint models, while CIA treats topology itself as protected information [52, 166, 176, 188]. NetSafe shows how topology changes attack reach, A2ASecBench exposes protocol-stage attack surfaces, and PEAR highlights security effects of role separation [36, 86, 182]. Router hijacking, FlowSteer, and insider studies make routing and role placement explicit attack variables, while MESA identifies security-critical communication edges [10,82,83,143,168,192]. Together, these studies show that routing and role placement can directly change decision influence and reachable authority. Many studies evaluate fixed membership, topology, and roles, leaving runtime admission, rerouting, and role reassignment less studied [10, 21, 82, 86].

P4. Adversarial coordination and collective manipulation. Collective security depends on independence, not only agreement. Secret Collusion and CoMet show covert coordination channels; Among Us measures malicious contributions; Lying with Truths shows how individually truthful statements can produce a misleading group conclusion; and persuasion and hidden role studies show how strategic members can steer a group [2,55,57,76,93,105,110,111,172,178]. Other work studies adversarial majorities, coordinated insiders, debate jailbreaks, Byzantine communication, and consensus traps [15, 37, 94, 127, 134, 143]. Byzantine fault-tolerance studies further show that collective outcomes depend on assumptions about faulty membership and aggregation [191]; Fig. 3 illustrates the classical fault-bound intuition. A group may therefore agree strongly even when its evidence is compromised or not independent. It can also be hard to separate strategic coordination from common causes such as shared models, prompts, retrievers, or evidence sources [30,55,105,111,135]. This also limits content-only defenses: individually harmless messages can still create dependence through the communication channel, so message content alone cannot establish contributor independence [105, 111, 172].

![](images/cd82eb41b663bf8526c32fb9a41040af85d3f3d50a4e38c2b7805d0867657906.jpg)  
Figure 3: Example: collective manipulation. With three malicious agents among seven, the classical Byzantine faulttolerance condition $f < n / 3$ is not satisfied, so the classical guarantee no longer applies.

P5. Peer- and state-mediated steering. P5 has limited direct evidence in our current map. CRDA measures content-risk drift under adversarial multi-agent interaction, while persuasion studies provide more direct evidence that peer interaction can redirect later behavior [76, 98]. FlowSteer and latentstate studies extend this analysis over longer or stateful executions [19, 82]. The LLM Drift software artifact provides an additional framework for measuring behavioral drift in adversarial multi-agent simulations [138]. Lower accuracy, changed reasoning, or generic drift does not by itself constitute an R4 failure. A study must identify the protected objective or allowed trajectory and show that interaction causes an unauthorized deviation. Direct evidence remains limited on when peer or state influence causes persistent, securityrelevant goal redirection rather than temporary behavioral change [82, 98].

P6. Cross-principal information leakage. Confidentiality can fail during execution even when the final answer appears safe. MASLeak studies extraction of multi-agent intellectual property, including prompts, tools, and architecture, while AgentLeak measures privacy leakage across internal coordina tion channels; CIA shows that communication topology itself can be inferred [160,166,174]. Compositional privacy shows that several acceptable local views can reveal more when combined, and collusion studies show that inter-principal channels can hide strategic information transfer [111, 124].

Recent work studies distinct extensions: OMNI-LEAK examines orchestrator-mediated leakage, WebWeaver targets topology confidentiality, CalBench measures coordination–privacy trade-offs, and SNEAK evaluates strategic communication and information leakage [14, 112, 171, 198]. The protected object is therefore the information available to a principal or coalition across the execution, not only secret strings in the final response. Existing work gives limited attention to how ownership, provenance, receiver constraints, and intended use survive summarization, replication, joint inference, and tool use [95, 124, 174]. We found limited controlled evidence on cross-principal tool-prompt theft or timing, I/O, and network side channels.

P7. Delegation and authority escalation. A locally valid handoff does not guarantee valid authority across the full execution. Control-flow hijacking traces attacker-controlled influence into privileged tools, while cascading instruction shows how upstream content can gain downstream authority. Multi-agent code-generation attacks show similar failures across specialized roles and generated artifacts [1, 62, 152, 163]. PEAR and A2ASecBench provide role- and protocolaware views of these transitions [36,86]. AgentWorm extends the path across communication, state, and downstream actions, while When Child Inherits considers spawned principals and inherited state or authority [21, 187]. The recurring failure is the loss of origin, purpose, or scope along the delegation chain. Defense work on Agentic JWT and SentinelAgent begins to address multi-hop delegation, but preserving intent, scope, and provenance across semantic transformation, and revoking authority after credentials, capabilities, or state have already propagated, remain open challenges [21, 45, 123].

P8. Shared-resource and control-plane interference. Shared resources create availability risks across principals. CORBA provides direct evidence of communicationlevel availability loss through contagious blocking [195]. Architectural-resilience and defection studies show that malicious or uncooperative participation can degrade shared execution, although such degradation does not by itself establish an R7 resource-isolation failure [77,130]. Capability-paradox and sabotage evaluations show that defensive components can themselves change system-level security outcomes [96, 106]. This suggests that the defense plane can become part of the interaction surface, but systematic evidence on adversarial resource exhaustion and control-plane failure remains limited. Centralized control may improve visibility while also increasing the impact of a control-plane failure. Our current map contains relatively little direct security evidence for P8 compared with propagation and collective manipulation [96, 106, 195]. Path composition. A single execution may combine several paths. For example, malicious influence may enter through P1, persist through P2, alter a collective decision through P4, and finally reach a privileged action through P7 [1, 62, 79]. This composition shows why attacks should be traced across the full execution.

## 4.4 Cross-Cutting Findings

Across P1–P8, four findings cut across individual attack mechanisms. First, relations are themselves attack surfaces: routing, role placement, aggregation, and delegation can change security even when message content is unchanged [10, 52, 62, 176]. Second, security effects can outlive the interaction that created them: shared state, spawned principals, and inherited authority can preserve a compromise after the original source is removed [21, 95, 187]. Third, the protected object extends beyond thefinal output: independence, topol ogy, private context, and delegated authority can fail even when the final answer appears correct [62, 111, 166, 174]. Finally, observing an attack path is not the same as attributing its cause. Execution traces can show where harm travels, but additional comparisons are needed to establish which interaction caused or amplified it [9, 86, 182].

Together, these findings shift the security unit from an isolated malicious message or agent to the execution path that carries influence across principals. This distinction also deter mines what a defense must interrupt and what an evaluation must observe or vary to support a system-level security claim.

## 5 Defense Landscape

The attack paths in Section 4.3 show how adversarial influence moves through an MAS. A defense may interrupt such a path by constraining the system before execution, intervening during execution, or containing and repairing effects after they occur. We therefore ask which path the defense targets, what it can observe, where and when it can intervene, what it must trust, and what recovery remains possible. Figure 1 shows the same execution from the defensive side.

## 5.1 Defense Contract

We characterize each defense along five dimensions: path target, observation, intervention, trust boundary, and recovery. Because security effects may span several principals, a defense must account for how influence, state, and authority move through the execution.

Path target. Which system-level risk or attack path must the defense interrupt? A claim should identify the risk R1– R7 and the interaction or path segment that must be broken. Protecting one agent or filtering one message is insufficient if the same effect remains reachable through another route.

Observation. What part of the cross-principal execution can the defender observe? A defense may see a message, principal, communication edge, shared state, subgraph, latent state, or full execution trace. Observation depends mainly on oversight architecture C8 and which interfaces I1–I6 are visible. A defense cannot reliably detect a failure whose relevant interaction lies outside this view. Observation also varies in scope and depth: a local monitor may have detailed neighbor history, while a global monitor may cover more of the system through coarser logs or metadata [40, 119].

Intervention. What can the defender change, and when? A defense may reshape topology, policy, or authority before execution; block or rewrite messages, reroute communication, or quarantine principals during execution; or revoke authority and roll back shared state afterward. Timing matters: blocking a message before I2 may prevent a state change at I3 that is difficult to undo later.

Trust boundary. Which principals, components, and system conditions must remain trustworthy? Examples include authenticated identities, trusted coordinators, preserved provenance, bounded membership, honest-neighbor or honestmajority conditions, stable topology, synchrony, and complete logs. Together, these assumptions define the defense’s effective trust boundary. Changes in membership, routing, delega tion, or shared state can invalidate it during execution.

Recovery. What remains when prevention or containment fails? Harmful influence may already have reached another principal, shared state, delegated authority, or a collective decision. A complete defense claim should state what can still be contained, revoked, rolled back, or repaired, and how the system verifies that the protected property has been restored.

We use these five dimensions as a common comparison basis below. The control primitives describe what a defense acts on, while the contract describes how its protection claim is obtained and bounded.

## 5.2 Defense Control Primitives

We organize existing defenses into four control primitives according to the main object they control in a multi-agent execution. These primitives are not mutually exclusive: one system may combine several of them. For each primitive, we compare its target paths, observation scope, intervention points, trust assumptions, and recovery limits. A smaller emerging literature instead asks where limited defensive capacity should be allocated across these control surfaces; we discuss this cross-cutting strategy separately below. Table 3 summarizes the comparison.

D1. Identity, policy, and authority enforcement. D1 primarily interrupts admission and delegation paths (P1/P7) by controlling who may participate, what actions they may perform, and what authority may cross principal boundaries. Provenance controls can also help limit leakage along P6. Set 1 evidence includes identity and local-authorization mechanisms such as BlockA2A, decentralized identity and credential systems, SPIFFE-based authentication, SAFEFLOW, and ConvPayMAS [29, 85, 121, 136, 199]. They reduce reliance on message content alone by making identity, policy, and authorization explicit.

Several Set 2 proposals extend these controls to multihop delegation, identity-aware routing, and context scoping. Agentic JWT binds authorization to user intent and workflow context, SentinelAgent targets intent-verified delegation chains, and LDP and Nexus add identity- or protocol-aware controls across principals [45, 47, 122, 123, 126, 159]. Set 1 control-flow work separately shows how downstream invocations can be restricted at execution time [62]. Recent work also broadens D1 with governance and context-level controls [27, 33, 165]. Across the defense contract, D1 observes identity, provenance, intent, and scope at admission and authorization boundaries, then constrains participation, actions, or delegated authority. Its guarantees depend on authenticated identities, mediated handoffs, and preservation of the security context needed by downstream checks. Revocation alone may not recover authority, credentials, or derived state that have already propagated.

<table><tr><td>ID</td><td>Control primitive</td><td>Configuration</td><td>Interfaces</td><td>Risks</td><td>Primary evidence</td><td>Emerging evidence</td></tr><tr><td>D1</td><td>Identity / policy / authority enforcement C2/C6/C7/C8</td><td></td><td>I1/I2/I4/I6</td><td>R1/R5/R6</td><td>[29,62,85,121,136,199]</td><td>[45,47,123,126,165]</td></tr><tr><td>D2</td><td>Trust / interaction-structure control</td><td>C1/C3/C4/C6/C7 I2/I5/I6</td><td></td><td>R1/R2/R3/R6</td><td>[24–26,35,51,80,84,97,102,191,[18,22,64,89,94,157,161] 196]</td><td></td></tr><tr><td>D3</td><td>Detection / attribution / containment</td><td>C1/C3/C5/C6/C8  I1–I6</td><td></td><td></td><td>162,164,193,194]</td><td>Mechanism-specific [39,40,53,90, 109,119,130,141, [3,20,23,41,58,61,101,154,169,175]</td></tr><tr><td>D4</td><td>State / provenance / information-flow gov- C2/C3/C5/C7/C8 I1/I2/I3/I4/I6 R1/R5/R6 ernance</td><td></td><td></td><td></td><td>[4,87,145,146,155,180]</td><td>[13,31,44,103,108,142,149,167]</td></tr><tr><td></td><td>Adaptive hardening / defense allocation C1/C3/C4/C8</td><td></td><td>I2/I4/I5/I6</td><td>R1/R3/R6</td><td>[54,120,186]</td><td>[83]</td></tr></table>

Table 3: Representative evidence for four defense control primitives and the cross-cutting adaptive allocation strategy. Risk mappings indicate the main protected effects addressed by the primitive; D3 coverage is mechanism-specific.

D2. Trust and interaction-structure control. D2 mainly targets routing and collective-manipulation paths (P3/P4) by changing who can influence whom and how collective evidence is combined. Topology control can also reduce propagation along P2. ResMAS and topology-linearization methods modify influence paths, while RoMa and trust-aware communication use credibility or trust signals to limit unreliable influence [35, 84, 97, 196]. AgentChain provides a separate coordination mechanism, while MIN-Trust restricts information disclosure across collaborating principals [25, 26]. Other Set 1 work studies faulty or Byzantine members and changes how collective evidence is aggregated [24, 88, 191].

D2 also includes dynamic-trust mechanisms such as DynaTrust, A-Trust, and Ev-Trust, which update influence from interaction history or message signals [51, 89, 157]. A separate line changes the collective-decision rule itself through local coordination, Byzantine protocols, conformal social choice, majority-resistant aggregation, or institutional con trols [16, 18, 22, 64, 80, 94, 102, 161]. D2 therefore intervenes by changing communication, contributor influence, or aggregation. Its guarantees depend on assumptions about identity, membership, fault bounds, observability, and sometimes synchrony. Reconfiguring trust or topology can limit future influence, but does not by itself remove manipulated evidence already committed to persistent state or collective decisions. D3. Detection, attribution, and containment. D3 observes harmful influence and intervenes once suspicious behavior or propagation is identified. It applies across several attack paths, with much of the current evidence centered on propagation, routing, and collective manipulation (P2– P4). G-Safeguard, BlindGuard, XG-Guard, SentinelAgent, PeerGuard, GUARDIAN, and SentinelNet use communication structure, message or response signals, peer history, or anomaly scores to identify harmful participants [39, 40, 53, 109, 119, 162, 193].

Several recent works consider propagation rather than only suspicious endpoints. Cross-Layer Semantic Flow Reconstruction, CASPIAN, From Spark to Fire, INFA-Guard, PropGuard, contribution-based methods, and related cascade defenses study how harmful influence moves across messages, principals, or communication paths [20, 41, 154, 164, 169, 175, 194]. Other mechanisms target narrower intervention points, including misinformation rectification, sentencelevel repair, linguistic routing controls, stubborn-neighbor detection, and game-theoretic or machine-checkable controls [3,8,58,59,61,67,90,99,177]. SAIGuard moves intervention earlier by predicting the state change caused by a candidate message and intercepting it before delivery [141]. More broadly, D3 ranges from local message or peer history to graph- and trace-level observation, with guarantees bounded by the coverage and integrity of that telemetry [40, 119]. Detection and containment do not imply recovery: once P2 has changed downstream state, stopping further spread does not undo earlier effects.

D4. State, provenance, and information-flow governance. D4 controls what information and derived state may cross principal boundaries, where they may persist, and what security context remains attached to them. It therefore most directly addresses propagation, leakage, and delegation (P2/P6/P7). Set 1 evidence includes differential-privacy and privacy-preserving systems that limit disclosure across collaborating principals, together with provenance-aware work on cross-agent information flows [4, 43, 146, 180]. Set 1 also includes memory-oriented defenses such as Self-Healing Memory Architectures [155].

Set 2 work targets additional forms of shared and latent state, including Governed Shared Memory and EquiMem [103, 108]. PRISM models leakage that accumulates across multi-stage generation and intervenes before a complete secret is emitted, while LCGuard targets information carried through shared latent or KV state [13, 149]. Maris applies reference monitors and checked policies to declared crossprincipal message and tool flows, while semantic-taint mech anisms attach tracking information to derived data [31, 142]. Other work combines information governance with promptinjection controls, verifiable coordination, or contextual privacy mechanisms [44, 87, 145, 167]. D4’s guarantees are strongest when the relevant flow is mediated and its security context remains trustworthy. Maris, for example, assumes relevant message and tool flows pass through its reference monitors [31], while PRISM and latent-state defenses cover different leakage channels [13, 149]. Semantic rewriting, covert channels, and unmediated side effects remain harder to govern, and recovery requires tracing state that has already been copied or transformed.

Adaptive defense allocation. The four primitives above de scribe what a defense controls. A smaller emerging literature asks where limited defensive capacity should be concentrated. SafeSieve progressively prunes communication, while MESA prioritizes communication edges according to estimated security relevance [83, 186]. Adjacent adaptive-hardening work uses randomized smoothing or adversarial co-evolution to improve robustness [54,120]. These approaches can concentrate stronger protection on interactions that dominate propagation (P2), collective influence (P4), or downstream authority (P7), rather than applying the same control everywhere. We treat adaptive allocation as a cross-cutting strategy rather than a fifth control primitive because it selects where existing controls should be strengthened rather than defining a distinct object of enforcement. Its smaller evidence base also leaves open how well allocation signals transfer across tasks, models, topologies, and attack strategies.

## 5.3 Cross-Cutting Findings

Comparing the defense literature through the contract above reveals three cross-cutting limitations that are not specific to any one control primitive.

First, local protection does not imply path closure. Attack paths can compose: an attack may enter through P1, persist through P2, and gain authority through P7. A defense that blocks one transition therefore does not establish end-to-end protection when another route remains reachable. The mappings in Table 3 identify where a control may interrupt a path, not whether it closes every route to the protected effect.

Second, defense effectiveness depends on where and when control is applied. Some defenses constrain the system before execution, while runtime defenses must observe enough of the relevant execution to support their intervention. A global observer may cover more of the execution but rely on coarse or trusted telemetry, while a local observer may see detailed behavior along only a small part of the path [40, 119]. Timing also determines what can still be prevented or reversed: ResMAS changes interaction structure before execution, SAI-Guard can block candidate messages before delivery, while other mechanisms intervene after suspicious influence has been observed [119, 141, 196]. Defense guarantees therefore depend on the intervention point and, for runtime mechanisms, on whether their observation and control are sufficient for the claimed protection, together with the trust assumptions on which those mechanisms rely.

Finally, containment is not recovery. Removing the source or stopping further propagation does not restore the system once its effects have been copied into memory, artifacts, credentials, delegated authority, or spawned principals [21,95,187]. Recovery requires identifying these descendants, revoking inherited authority, repairing state, and verifying that the protected property has been restored. In our map, recovery is evaluated less directly than prevention and containment.

Together, these findings suggest that end-to-end MAS defense requires more than a strong local control: defenses must close the relevant attack paths, place interventions at points where they can affect those paths, provide adequate observation when runtime decisions depend on it, and account for residual effects that survive the original compromise.

## 6 Benchmarks and Evaluation

Beyond mapping the attack and defense landscape, we examine how the security of multi-agent systems is evaluated. We first systematize existing benchmarks and their accompanying artifacts. We then identify four gaps in current evaluation practice and discuss the corresponding open challenges.

## 6.1 Existing Benchmarks and Artifacts

Within the 197-work corpus, we identify 44 works whose main contribution is the security evaluation of multi-agent systems. These works test attacks, defenses, and security properties under different system designs and threat models. We reviewed the paper PDFs for 43 works and the software archive for one Zenodo record without an accompanying paper. For each work, we record its adversary position A1–A4, configuration dimensions C1–C8, interaction interfaces I1–I6, attack paths P1–P8, and system-level risks R1–R7. We also record their baselines, metrics, observation depth, and artifact availability. All counts below cover the full set of 44 works; the detailed per-work mapping is provided in Table 4.

The works provide broad coverage of both attacks and defenses. Forty-two evaluate an attack or failure, including 16 that also evaluate a defense. CalBench and MAGPIE evaluate privacy without an explicit attacker [66,198]. The benchmarks also cover many MAS designs. All 44 report or vary principal composition (C3), 39 cover coordination (C4), and 38 cover communication topology (C1). Membership is fixed in 42 works, while two allow limited runtime spawning. Member adversaries (A3), message transfer (I2), detection (I6), peer or state steering (P5), and collective-decision failure (R3) are the most common settings [9, 69, 95, 182].

The evaluations provide different depths of execution evidence. Classified by their deepest observation, 20 use full traces, 20 use principal-level messages or actions, two use episode outcomes, and two use final outputs. Thirty-one works identify code, data, logs, a test tool, or a project page, including 26 with an exact URL. The works also report communication topology for 38 works, model information for 32, and an exact principal count for 18.

Summary. The 44 works evaluate diverse MAS designs and security questions, with varied observation depth and artifact support.

## 6.2 Gaps and Open Challenges

Our analysis identifies four gaps in current evaluation practice. First, evaluations often show that a failure occurs without isolating how multi-agent interaction affects it. Second, metric names may hide what is measured, while execution traces rarely support causal diagnosis. Third, comparisons within one benchmark do not by themselves provide configurable or extensible components for other MAS designs. Finally, current evaluations largely focus on fixed, closed systems rather than practical open settings.

## 6.2.1 Isolating Multi-Agent Effects

Determining how interaction changes a security outcome remains an open evaluation challenge. As defined in Section 2.1, a failure may remain unchanged without interaction, become worse because of interaction, appear only because of interaction, or involve a security property that exists only among multiple principals. Distinguishing these four roles requires comparisons designed for the specific role being tested.

When an attack has a single-principal counterpart, a matched single-principal comparison can distinguish an inherited effect from an interaction-amplified effect. Our audit identifies 21 works that evaluate at least one of these two effects. Six compare an MAS with a single-principal execution [12, 60, 92, 128, 174, 189]. The remaining 15 evaluate the attack in an MAS or compare different MAS configurations. These evaluations show that the attack succeeds under the tested conditions, but they do not determine whether the same failure would occur with one principal or become worse through interaction.

When no meaningful single-principal counterpart exists, the evaluation should preserve the MAS and vary the relation under study. Among 36 such works, 12 include a relationfocused comparison. The remaining 24 use another control or provide no clear relation-focused comparison. For example, MESA removes or masks communication edges, while

CalBench compares models and coordination protocols under the same private-information constraints [83, 198].

Summary. Interaction-specific comparison remains an open evaluation challenge. Future benchmarks should match their comparison design to the interaction effect being tested.

## 6.2.2 Comparable and Diagnostic Metrics

Current metrics are inadequate in two respects: their outcomelevel measures are not comparable across benchmarks, and their trace-level measures do not explain how a compromise propagates across agents.

Underspecified ASR. Attack Success Rate (ASR) is the most common security metric in our corpus. Thirteen of the 44 works compute ASR in their own evaluation, while ATAG additionally records ASR values reported by other sources [42]. Across these studies, ASR refers to different evaluation units and success conditions because the benchmarks study different attacks. The name “ASR” alone therefore does not identify what event is counted or over which eligible population. An ASR of 60% in one benchmark may therefore measure a different event from an ASR of 60% in another. The values cannot be directly compared, ranked, or aggregated unless these definitions match. When propagation is part of the claim, an end-to-end ASR also cannot distinguish an attack stopped at entry from one that spreads across several agents before being blocked.

Future benchmarks should qualify ASR by its evaluation unit, using names such as per-principal ASR, per-message ASR, or per-run ASR, and define the counted success event. Each result should provide its taxonomy mapping, evaluation unit, execution protocol, and verification method. When propagation is part of the claim, stage-level measures should follow the relevant I1–I6 path to the verified R1–R7 consequence. These stages should be selected based on the attack rather than imposed on every risk. Existing work already measures individual dimensions such as harm amplification, topology-dependent leakage, and internal-channel exposure [95,131,174]. Benign task utility should be reported separately from security outcomes.

Missing Causal Provenance. Full execution traces provide more information than final-output evaluation, but they do not necessarily explain why a failure occurred. Twenty records use the full execution trace as their deepest observation. These traces may show that one principal received malicious information, another updated memory, and a third invoked a tool, without showing how these events depend on one another. Across the corpus, ATAG models attack dependencies and Agent-BOM reconstructs dependencies from executions [42, 81]. Agent-BOM is the only one of these 20 full-trace records coded as reporting an explicit dependency or provenance graph. This evidence is unnecessary for benchmarks that report only final outcomes, but it is needed for claims about propagation, responsibility, or defense placement.

Benchmarks that make such claims should represent the relevant execution path as a provenance graph. Its nodes should identify the principals and events at the I1–I6 interfaces, including attacker-controlled inputs, messages, memory operations, tool results, delegation decisions, and security-sensitive actions. Its edges should record data, control, and delegation dependencies. The path should begin with the first adversarially influenced event or violated trust boundary and end with a verified R1–R7 consequence. Because a dependency graph does not prove causation, the claimed path should also be tested through controlled replay or ablation.

Summary. Thirteen works compute ASR in their own evaluation, while ATAG additionally records ASR values reported by other sources. Twenty records use the full execution trace as their deepest observation, and Agent-BOM is the only one coded as reporting an explicit dependency or provenance graph.

## 6.2.3 Configurable and Extensible Benchmarks

Thirty-nine of the 44 works compare two or more evaluation conditions. These comparisons include component or attack controls, defense controls, single-principal baselines, and relation-focused MAS comparisons. They support controlled variation within each study, but do not show whether benchmark factors can be changed independently or reused elsewhere. We call a benchmark configurable when its supplied factors can be changed independently and extensible when new components can be added through a documented interface.

ACIARena defines interfaces for MAS implementations, attacks, defenses, and evaluation modules [9]. RiskLab provides configurable experiment components and a registry for adding risk detectors [63]. A2ASecBench uses an LLM-based adapter to apply attack descriptions to different A2A scenarios [86]. These benchmarks expose useful extension points, but their task, trace, attack, and verifier formats differ. Trans ferring a component between them would therefore require an additional adapter. Similarly, the 31 works with identified artifacts may provide access to code or data without making their components reusable. Because our audit does not systematically record interface reuse, we do not quantify how often such reuse occurs.

Benchmarks intended for cross-system comparison should expose the factors named in their claims through documented interfaces. A paired test should change one factor while preserving the protected property and outcome definition. Differences between system interfaces should be handled through validated adapters. Each run should record four groups of information: the model specification, execution settings, environment state, and verifier version.

Summary. Thirty-nine works compare conditions within their own benchmark, but these comparisons do not by themselves establish configurability or extensibility. Future cross-system benchmarks should use documented interfaces while preserving the same security goal and success condition.

## 6.2.4 Open-System Evaluation

Current benchmarks primarily evaluate closed systems whose members and trust relationships are fixed before execution. Forty-two of the 44 works use fixed membership, while the other two allow limited runtime spawning under one administrator. Ten works use deception games or negotiation environments, and many others rely on synthetic tasks with predefined roles. These settings support controlled experiments but do not capture how practical MAS operate across changing trust boundaries.

Identity, trust, authority, and policy also matter in closed systems. However, they become more dynamic and consequential when previously unknown principals interact across administrative boundaries [72, 139]. Current benchmarks largely begin after these relationships have been established. They therefore provide little evidence about how principals enter the system, how trust changes during execution, or how access is removed.

Future benchmarks should evaluate deployed coordination protocols and frameworks with independently managed principals. Testbeds should include heterogeneous implementations, persistent state, external services, and changing membership. They should cover the lifecycle from discovery and admission through operation and removal. Security outcomes should reflect consequences across real trust boundaries rather than only success or failure in a synthetic task.

Summary. Forty-two works use fixed membership, while two allow limited spawning under one administrator. Evaluating changing membership and cross-domain trust remains an open benchmark problem.

## 7 Conclusion

This SoK frames multi-agent LLM security around end-toend execution. Across 197 works, we organize MAS security through system configuration, interaction interfaces, attack paths, defenses, and evaluation. Our analysis shows that security depends on how adversarial influence crosses principal boundaries, what state and authority remain reachable, whether defenses can close the path, and whether evaluations verify the resulting system-level effect. This executioncentered view provides a common basis for comparing security claims across MAS designs.

## Ethical Considerations

This SoK synthesizes publicly available research on the security of multi-agent systems. It does not involve human-subject studies, private user data, or experiments against live or thirdparty systems. Our analysis is based on published papers, public artifacts, advisories, and bibliographic metadata. The topic is inherently dual use: organizing attack mechanisms and system weaknesses can help both defenders and attackers understand where multi-agent systems fail. We therefore emphasize security properties, interaction paths, threat assumptions, defenses, and evaluation evidence rather than operational payloads or step-by-step exploitation procedures. We do not disclose non-public vulnerabilities or unnecessary identifying details. Finally, we distinguish experimental evidence from deployment prevalence. A vulnerability demonstrated in a benchmark or controlled environment does not by itself imply that the same failure is common in deployed systems. We report claims within the scope and assumptions supported by the underlying studies.

## Open Science

The literature corpus and associated artifacts will be released publicly with the final version of the paper.

## Acknowledgement

Research reported in this publication was supported by an Amazon Research Award, Fall 2025, and by the 2026 Amazon Nova AI Challenge: Trusted Software Agents. The views and conclusions contained herein are those of the authors and should not be interpreted as necessarily representing the official policies of the supporting sponsors.

## References

[1] Mohamed Refaat Mohamed Abdellah and Alber S. Aziz. Cascading instruction influence: A simulation-based analysis of indirect prompt injection in hierarchical multi-agent systems. Statistics, Optimization & Information Computing, 15(6):5445–5458, 2026.

[2] Sahar Abdelnabi, Amr Gomaa, Sarath Sivaprasad, Lea Schönherr, and Mario Fritz. Cooperation, competition, and maliciousness: Llmstakeholders interactive negotiation. In NeurIPS 2024, volume 37, 2024.

[3] Samira Abedini, Sina Mavali, Lea Schönherr, Martin Pawelczyk, and Rebekka Burkholz. Don’t trust stubborn neighbors: A security framework for agentic networks. In arXiv.org, 2026.

[4] Chirag Agarwal, Naresh Erukulla, and Rachit Gupta. Privacypreserving LLM infrastructure with multi-agent orchestration and RAG-driven retrieval. Journal ofInternational Crisis and Risk Communication Research, pages 133–140, 2026.

[5] Mahak Agarwal and Divyam Khanna. When persuasion overrides truth in multi-agent llm debates: Introducing a confidence-weighted persuasion override rate (cw-por). arXiv preprint arXiv:2504.00374, 2025.

[6] Mrinal Agarwal, Saad Rana, Theo Sundoro, Hermela Berhe, Spencer Kim, Vasu Sharma, Sean O’Brien, and Kevin Zhu. WOLF: Werewolf based observations for LLM deception and falsehoods. In NeurIPS 2025 Workshop on Multi-Turn Interactions in Large Language Models, 2025.

[7] Tanzim Ahad, Ismail Hossain, Md Jahangir Alam, Sai Puppala, Yoonpyo Lee, Syed Bahauddin Alam, and Sajedul Talukder. Semantic intent fragmentation: A single-shot compositional attack on multiagent AI pipelines. AAAI Symposium Series, 9(1):229–237, 2026.

[8] Dvir Alsheich, Adar Peleg, Ben Hagag, Rom Himelstein, Amit Levi, and Avi Mendelson. Linguistic firewall: Geometry as defense in multi-agent systems routing. arXiv, 2026.

[9] Hengyu An, Minxi Li, Jinghuai Zhang, Naen Xu, Chunyi Zhou, Changjiang Li, Xiaogang Xu, Tianyu Du, and Shouling Ji. ACIA-Rena: Toward unified evaluation for agent cascading injection. In ACL, pages 10049–10066, 2026.

[10] Sandro Andric. LLM hijacking: When models manipulate their routers. Zenodo, 2025.

[11] Nokimul Hasan Arif, Qian Lou, and Mengxin Zheng. Conjunctive prompt attacks in multi-agent llm systems. In ACL, pages 34175– 34191, 2026.

[12] Nirmit Arora, Sathvik Joel, Ishan Kavathekar, Palak, Rohan Gandhi, Yash Pandya, Tanuja Ganu, Aditya Kanade, and Akshay Nambi. Exposing weak links in multi-agent systems under adversarial prompting. In AAMAS 2026 Strategic Engineering Workshop, 2026.

[13] Sadia Asif, Mohammad Mohammadi Amiri, Momin Abbas, Prasanna Sattigeri, and Karthikeyan Natesan Ramamurthy. Lcguard: Latent communication guard for safe KV sharing in multi-agent systems. arXiv preprint arXiv:2605.22786, 2026.

[14] Adar Avsian and Larry Heck. Sneak: Evaluating strategic communication and information leakage in large language models. arXiv preprint arXiv:2603.29846, 2026.

[15] Adeela Bashir, The Anh Han, and Zia Ush-Shamszaman. Many-toone adversarial consensus: Exposing multi-agent collusion risks in ai-based healthcare. arXiv preprint arXiv:2512.03097, 2025.

[16] Saurav Bhattacharya. Consensus-driven metacognition in multi-agent systems: A logic-based byzantine fault-tolerant protocol. International Journal ofGlobal Innovations and Solutions, 3(4), 2026.

[17] Milon Biswas, Wei Yu, Qianlong Wang, and Weixian Liao. Quantifying privacy leakage in multi-agent llms: A unified theoretical and empirical analysis. In BigData Congress [Services Society], pages 1685–1694, 2025.

[18] Marcantonio Bracale Syrnikov, Federico Pierucci, Marcello Galisai, Matteo Prandi, Piercosma Bisconti, Francesco Giarrusso, Olga Sorokoletova, Vincenzo Suriani, and Daniele Nardi. Institutional AI: Governing LLM collusion in multi-agent cournot markets via public governance graphs. arXiv preprint arXiv:2601.11369, 2026.

[19] Luís Brito and Carlos Baquero. When latent agents lie: Kvcache integrity in multi-agent llm collaboration. arXiv preprint arXiv:2606.28958, 2026.

[20] Qizhi Cai, Yangyang Wei, Zhipeng Chen, Shouling Ji, and Zhenyuan Li. Cross-layer semantic flow reconstruction for attack detection in agentic systems. arXiv preprint arXiv:2603.04469, 2026.

[21] Ziwen Cai, Yihe Zhang, and Xiali Hei. When child inherits: Modeling and exploiting subagent spawn in multi-agent networks. arXiv preprint arXiv:2605.08460, 2026.

[22] Nihir Chadderwala. Byzantine fault-tolerant multi-agent system for healthcare: A gossip protocol approach to secure medical message propagation. arXiv preprint arXiv:2512.17913, 2025.

[23] Chi-Min Chan, Jianxuan Yu, Weize Chen, Chunyang Jiang, Xinyu Liu, Weijie Shi, Zhiyuan Liu, Wei Xue, and Yike Guo. Agentmonitor: A plug-and-play framework for predictive and secure multi-agent systems. arXiv preprint arXiv:2408.14972, 2024.

[24] Bei Chen, Gaolei Li, Xi Lin, Zheng Wang, and Jianhua Li. Blockagents: Towards byzantine-robust LLM-based multi-agent coordination via blockchain. In ACM Turing Celebration Conference, pages 187– 192. ACM, 2024.

[25] Bei Chen, Gaolei Li, Jun Wu, Jianhua Li, Mingzhe Chen, and Jiacheng Wang. Agentchain: Blockchain-empowered multi-agent coordination for trustworthy LLM question-answering systems. IEEE Transactions on Dependable and Secure Computing, 23(4):8501–8518, 2026.

[26] Jinyu Chen, Feiyang Wang, Tian Guan, Yumeng Ma, Linghao Yang, and Yutong Wang. MIN-Trust: A minimum necessary information trust orchestration framework for multi-agent collaboration. In GAIE 2026, 2026.

[27] Jinyu Chen, Jixiao Yang, Ziyang Zeng, Zixiao Huang, Jinming Li, and Yutong Wang. Securegov-agent: A governance-centric multi-agent framework for privacy-preserving and attack-resilient LLM agents. In Proceedings ofthe 2025 6th International Conference on Computer Science and Management Technology, pages 875–881. ACM, 2025.

[28] Kai Chen, Taihang Zhen, Hewei Wang, Kailai Liu, Xinfeng Li, Jing Huo, Tianpei Yang, Jinfeng Xu, Wei Dong, and Yang Gao. MedSentry: Understanding and mitigating safety risks in medical LLM multiagent systems. arXiv, 2025.

[29] Joon Kiat Chua, Chen Hao Tse, Donghao Huang, and Zhaoxia Wang. Convpaymas: Conversational payment multi-agent system with agent to-agent protocol and three-mandate verification. In Proceedings of the 25th International Conference on Autonomous Agents and Multiagent Systems. International Foundation for Autonomous Agents and Multiagent Systems, 2026.

[30] Romain Cosentino, Sarath Shekkizhar, and Adam Earle. Convergence dynamics of agent-to-agent interactions with misaligned objectives. arXiv preprint arXiv:2511.08710, 2025.

[31] Jian Cui, Zichuan Li, Chi Wang, Luyi Xing, and Xiaojing Liao. Maris: A formally verifiable privacy policy enforcement paradigm for multiagent collaboration systems. arXiv preprint arXiv:2505.04799, 2025.

[32] Yu Cui and Hongyang Du. MAD-Spear: A conformity-driven prompt injection attack on multi-agent debate systems. arXiv preprint arXiv:2507.13038, 2025.

[33] J. de Curtò, I. de Zarzà, and Carlos T. Calafate. Ethical coordination of LLM multi-agent systems. Electronics, 15(11):2278, 2026.

[34] Zehang Deng, Yongjian Guo, Changzhou Han, Wanlun Ma, Junwu Xiong, Sheng Wen, and Yang Xiang. Ai agents under threat: A survey of key security challenges and future pathways. ACM Computing Surveys, 2025.

[35] Yepeng Ding, Ahmed Twabi, Junwei Yu, Lingfeng Zhang, Tohru Kondo, and Hiroyuki Sato. Decentralized multi-agent system with trust-aware communication. In IEEE International Symposium on Parallel and Distributed Processing with Applications (ISPA), pages 1439–1445, 2025.

[36] Shen Dong, Mingxuan Zhang, Pengfei He, Li Ma, Bhavani Thuraisingham, Hui Liu, and Yue Xing. PEAR: Planner-executor agent robustness benchmark. In Findings ofthe Associationfor Computational Linguistics: EACL, 2026.

[37] Aya El Mir, Martin Takác, and Salem Lahlou. Byzantine cheap talk: Adversarial resilience and topology effects in LLM coordination games. arXiv preprint arXiv:2606.07790, 2026.

[38] Casey Fan, Diyana Tial, Vladislav Dubrovenski, Mengtao Zhang, Yugyung Lee, and Dianxiang Xu. Security of llm agents: A case study approach. In International Conference on Trust, Security and Privacy in Computing and Communications, pages 1952–1957, 2025.

[39] Falong Fan and Xi Li. Peerguard: Defending multi-agent systems against backdoor attacks through mutual reasoning. In IEEE International Conference on Information Reuse and Integration and Data Science (IRI), 2025.

[40] Yang Feng and Xudong Pan. SentinelNet: Safeguarding multi-agent collaboration through credit-based dynamic threat detection. In the ACM Web Conference, 2026.

[41] Nickolas Gamb. Memetic cascade detection and symbolic immunity in multi-agent LLM systems. Zenodo, 2026.

[42] Parth Atulbhai Gandhi, David Tayouri, Akansha Shukla, Beni Ifland, Yuval Elovici, Rami Puzis, and Asaf Shabtai. ATAG: Ai-agent application threat assessment with attack graphs. In ACM Asia Conference on Computer and Communications Security, pages 805–819, 2026.

[43] Pawan Kumar Goel, Satya Prakash Yadav, and Monika Singh. Security and privacy in multi-agent llm networks: Addressing vulnerabilities. In Advances in computational intelligence and robotics book series, pages 225–244. IGI Global Scientific Publishing, 2026.

[44] Diego Gosmar and Deborah A. Dahl. Prompt injection mitigation with agentic AI, nested learning, and AI sustainability via semantic caching. arXiv preprint arXiv:2601.13186, 2026.

[45] Abhishek Goswami. Agentic JWT: A secure delegation protocol for autonomous AI agents. arXiv preprint arXiv:2509.13597, 2025.

[46] Xiangming Gu, Xiaosen Zheng, Tianyu Pang, Chao Du, Qian Liu, Ye Wang, Jing Jiang, and Min Lin. Agent smith: A single image can jailbreak one million multimodal LLM agents exponentially fast. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 16647–16672. PMLR, 2024.

[47] Saurav Gupta. Nexus protocol: A cryptographically secure, zerolatency semantic routing engine for multi-agent systems. SSRN Electronic Journal, 2026.

[48] Ben Hagag, William L. Anderson, Christian Schroeder de Witt, and Sarah Scheffler. Architecture matters for multi-agent security. arXiv preprint arXiv:2604.23459, 2026.

[49] Lewis Hammond, Alan Chan, Jesse Clifton, Jason Hoelscher-Obermaier, Akbir Khan, Euan McLean, Chandler Smith, Wolfram Barfuss, Jakob Foerster, Tomáš Gavenciak, The Anh Han, Edwardˇ Hughes, Vojtech Kovaˇ ˇrík, Jan Kulveit, Joel Z. Leibo, Caspar Oesterheld, Christian Schroeder de Witt, Nisarg Shah, Michael Wellman, Paolo Bova, Theodor Cimpeanu, Carson Ezell, Quentin Feuillade-Montixi, Matija Franklin, Esben Kran, Igor Krawczuk, Max Lamparth, Niklas Lauffer, Alexander Meinke, Sumeet Motwani, Anka Reuel, Vin cent Conitzer, Michael Dennis, Iason Gabriel, Adam Gleave, Gillian Hadfield, Nika Haghtalab, Atoosa Kasirzadeh, Sébastien Krier, Kate Larson, Joel Lehman, David C. Parkes, Georgios Piliouras, and Iyad Rahwan. Multi-agent risks from advanced AI. Cooperative AI Foun dation Technical Report #1, 2025.

[50] Feng He, Tianqing Zhu, Dayong Ye, Bo Liu, Wanlei Zhou, and Philip S. Yu. The emerged security and privacy of llm agent: A survey with case studies. ACM Computing Surveys, 2025.

[51] Pengfei He, Zhenwei Dai, Xianfeng Tang, Yue Xing, Hui Liu, Jingying Zeng, Qiankun Peng, Shrivats Agrawal, Samarth Varshney, Suhang Wang, Jiliang Tang, and Qi He. To trust or not to trust: Attention-based trust management for LLM multi-agent systems. ACL, 2026.

[52] Pengfei He, Yuping Lin, Shen Dong, Han Xu, Yue Xing, and Hui Liu. Red-teaming llm multi-agent systems via communication attacks. In Findings ofACL, pages 6726–6747, 2025.

[53] Xu He, Di Wu, Yan Zhai, and Kun Sun. SentinelAgent: Graph-based anomaly detection in multi-agent systems. arXiv, 2025.

[54] Jinwei Hu, Yi Dong, Zhengtao Ding, and Xiaowei Huang. Enhancing robustness of LLM-driven multi-agent systems through randomized smoothing. Chinese Journal ofAeronautics, 2026.

[55] Jinwei Hu, Xinmiao Huang, Youcheng Sun, Yi Dong, and Xiaowei Huang. Lying with truths: Open-channel multi-agent collusion for belief manipulation via generative montage. In ACL, pages 5979– 5996, 2026.

[56] Jen-Tse Huang, Jiaxu Zhou, Tailin Jin, Xuhui Zhou, Zixi Chen, Wenx uan Wang, Youliang Yuan, Michael R. Lyu, and Maarten Sap. On the resilience of LLM-based multi-agent collaboration with faulty agents. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings ofMachine Learning Research, pages 26202–26226. PMLR, 2025.

[57] Kaibo Huang, Yukun Wei, Wansheng Wu, Tianhua Zhang, Zhongliang Yang, and Linna Zhou. Whispering agents: An event-driven covert communication protocol for the internet of agents. In AAAI, volume 40, pages 31185–31192, 2026.

[58] Saeid Jamshidi. Collective hallucination in multi-agent LLMs: Modeling and defense. arXiv preprint arXiv:2606.07941, 2026.

[59] Saeid Jamshidi. SGTO-MAS: Secure gorilla troops optimization for multi-agent LLM systems. arXiv preprint arXiv:2606.07940, 2026.

[60] Saeid Jamshidi, Arghavan Moradi Dakhel, Kawser Wazed Nafi, and Foutse Khomh. Security engineering of OpenClaw: Analyzing at tack surface expansion and trust-boundary violations. arXiv preprint arXiv:2606.15008, 2026.

[61] Saeid Jamshidi, Amin Nikanjam, Arghavan Moradi Dakhel, Kawser Wazed Nafi, and Foutse Khomh. Game-theoretic multi-agent control for robust contextual reasoning in LLMs. arXiv preprin arXiv:2606.10322, 2026.

[62] Rishi Jha, Harold Triedman, Justin Wagle, and Vitaly Shmatikov. Breaking and fixing defenses against control-flow hijacking in multiagent systems. In International Conference on Learning Representa tions, 2026.

[63] Yu Jiang, Wenjie Wang, Yue Huang, Yanbo Wang, Zhenhong Zhou, Xiuying Chen, Yang Liu, Pin-Yu Chen, Wei Wang, and Xiangliang Zhang. RiskLab: A controlled toolkit for probing emergent risks in LLM-based multi-agent systems. In ACL System Demonstrations, 2026.

[64] Yongrae Jo and Chanik Park. Byzantine-robust decentralized coordination of LLM agents. arXiv preprint arXiv:2507.14928, 2025.

[65] Tianjie Ju, Yiting Wang, Xinbei Ma, Pengzhou Cheng, Haodong Zhao, Yulong Wang, Lifeng Liu, Jian Xie, Zhuosheng Zhang, and Gongshen Liu. Flooding spread of manipulated knowledge in llm-based multi agent communities. Science China Information Sciences, 2026.

[66] Gurusha Juneja, Jayanth Naga Sai Pasupulati, Alon Albalak, Wenyue Hua, and William Yang Wang. Magpie: A benchmark for multi-agent contextual privacy evaluation. In NeurIPS Workshop on Socially Responsible and Trustworthy Foundation Models, 2025.

[67] Ravi Kiran Kadaboina. Pratyahara: A neural tissue defense model for detecting compromised agents in multi-agent networks. Zenodo, 2026.

[68] Jaidatt Raosaheb Kale, Harshita Sanjay Jain, Atharva Rahul Korwar, and Kanchan Ashish Jojare. A literature survey on privacy-preserving multi-agent RAG systems with an intelligent tag-inference routing sys tem. International Research Journal of Modernization in Engineering Technology and Science, 8(5), 2026.

[69] Ishan Kavathekar, Hemang Jain, Ameya Rathod, Ponnurangam Kumaraguru, and Tanuja Ganu. TAMAS: Benchmarking adversarial risks in multi-agent LLM systems. In Proceedings of the 64th An nual Meeting ofthe Associationfor Computational Linguistics, pages 31238–31268, 2026.

[70] Rana Muhammad Shahroz Khan, Zhen Tan, Sukwon Yun, Charles Fleming, and Tianlong Chen. Agents under siege: Breaking pragmatic multi-agent llm systems with optimized prompt attacks. In Proceed ings ofthe 63rd Annual Meeting ofthe Associationfor Computational Linguistics, 2025.

[71] Juhee Kim, Wenbo Guo, and Dawn Song. SoK: Attack and defense landscape of agentic AI systems. In 35th USENIX Security Symposium (USENIX Security 26). USENIX Association, 2026.

[72] Ronny Ko, Jiseong Chung, Shuyuan Zheng, Chuan Xiao, Tae-Wan Kim, Makoto Onizuka, and Won-Yong Shin. Seven security challenges in cross-domain multi-agent llm systems. npj Artificial Intelli gence, 2026.

[73] Dezhang Kong, Shi Lin, Zhenhua Xu, Zhebo Wang, Minghao Li, Yufeng Li, Yilun Zhang, Hujin Peng, Xiang Chen, Zeyang Sha, Yuyuan Li, Changting Lin, Xun Wang, Xuan Liu, Ningyu Zhang, Chaochao Chen, Chunming Wu, Muhammad Khurram Khan, and Meng Han. A survey of LLM-driven AI agent communication: Protocols, security risks, and defense countermeasures. arXiv preprint arXiv:2506.19676, 2025.

[74] Samuel Korn. Architecture matters: Comparing RAG systems under knowledge base poisoning. arXiv preprint arXiv:2605.05632, 2026.

[75] Insaf Kraidia, Iyas Qaddara, Nida Al-Shafi, Mohammad O. Hiari, Rania Alzubaidi, and Amira Benlecheb. Rag-induced failures in multiagent large language model debate. In 2026 1st International Conference on Emerging Technologies and Engineering Systems (ICETES), pages 326–331, 2026.

[76] Insaf Kraidia, Iyas Qaddara, Alhanof Almutairi, Nada Alzaben, and Samir Brahim Belhouari. When collaboration fails: Persuasion-driven adversarial influence in multi-agent llm debate. Scientific Reports, 16:11640, 2026.

[77] Devang Kulshreshtha, Wanyu Du, Raghav Jain, Srikanth Doss, Hang Su, Sandesh Swamy, and Yanjun Qi. The subtle art of defection: Understanding uncooperative behaviors in llm based multi-agent systems. In EACL 2026 Industry Track, pages 571–585, 2026.

[78] Alexandre Le Mercier, Chris Develder, and Thomas Demeester. GAM-BIT: A three-mode benchmark for adversarial robustness in multiagent LLM collectives. arXiv preprint arXiv:2605.09027, 2026.

[79] Donghyun Lee, Mo Tiwari, and Brando Miranda. Prompt infection: LLM-to-LLM prompt injection within multi-agent systems. In ES-ORICS 2025 International Workshops, Lecture Notes in Computer Science, pages 511–520. Springer Nature Switzerland, 2026.

[80] Haejoon Lee, Vincent-Daniel Yun, Hyeonho Oh, Dimitra Panagou, and Sai Praneeth Karimireddy. Robust multi-agent LLMs under byzantine faults. In Proceedings ofthe 2026 Conference on Empirical Methods in Natural Language Processing (EMNLP), 2026.

[81] Chaofan Li, Lyuye Zhang, Jintao Zhai, Siyue Feng, Xichun Yang, Huahao Wang, Shihan Dou, Yu Ji, Yutao Hu, Yueming Wu, Yang Liu, and Deqing Zou. Towards security-auditable LLM agents: A unified graph representation. arXiv preprint arXiv:2605.06812, 2026.

[82] Fanxiao Li, Jiaying Wu, Tingchao Fu, Natasha Jaques, Wei Zhou, and Min-Yen Kan. FlowSteer: Prompt-only workflow steering exposes planning-time vulnerabilities in multi-agent LLM systems. arXiv preprint arXiv:2605.11514, 2026.

[83] Kunyang Li, Kyle Domico, Jonathan Gregory, and Patrick McDaniel. MESA: Prioritizing vulnerable communication channels for securing multi-agent systems. arXiv preprint arXiv:2606.30602, 2026.

[84] Kunyu Li and FengHe Wu. Roma: A credibility-aware fault-tolerance framework for LLM multi-agent systems. In International Conference on Computer Supported Cooperative Work in Design, pages 1526– 1531. IEEE, 2026.

[85] Peiran Li, Xinkai Zou, Zhuohang Wu, Ruifeng Li, Shuo Xing, Hanwen Zheng, Zhikai Hu, Yuping Wang, Haoxi Li, Qin Yuan, Yingmo Zhang, and Zhengzhong Tu. SAFEFLOW: A principled protocol for trustworthy and transactional autonomous agent systems. arXiv preprint arXiv:2506.07564, 2025.

[86] Tianhao Li, Chuangxin Chu, Yujia Zheng, Bohan Zhang, Neil Zhenqiang Gong, and Chaowei Xiao. A2ASecBench: A protocol-aware security benchmark for agent-to-agent multi-agent systems. In The Fourteenth International Conference on Learning Representations (ICLR), 2026.

[87] Wenkai Li, Liwen Sun, Zhenxiang Guan, Xuhui Zhou, and Maarten Sap. 1-2-3 check: Enhancing contextual privacy in LLM via multiagent reasoning. In LLMSEC, 2025.

[88] Xujia Li, Beirong Cui, Junyu Huang, and Lei Chen. Robust LLMbased multi-agent system with action negotiation and sharing redundancy enhancement. In Knowledge Discovery and Data Mining, pages 736–747. ACM, 2026.

[89] Yu Li, Qiang Hu, Yao Zhang, Lili Quan, Jiongchi Yu, and Junjie Wang. Dynatrust: Defending multi-agent systems against sleeper agents via dynamic trust graphs. arXiv, 2026.

[90] Zherui Li, Yan Mi, Zhenhong Zhou, Houcheng Jiang, Guibin Zhang, Kun Wang, and Junfeng Fang. Goal-aware identification and rectification of misinformation in multi-agent systems. In The Fourteenth International Conference on Learning Representations (ICLR), 2026.

[91] Ruichao Liang, Le Yin, Jing Chen, Yebo Feng, Cong Wu, Xiaoyu Zhang, Huangpeng Gu, Zijian Zhang, and Yang Liu. Don’t trust your upstream: Exploiting LLM multi-agent system via topology-guided adversarial propagation. arXiv preprint arXiv:2512.04129, 2025.

[92] Chengzhi Liu, Yichen Guo, Yepeng Liu, Yuzhe Yang, Qianqi Yan, Xuandong Zhao, Wenyue Hua, Sheng Liu, Sharon Li, Yuheng Bu, and Xin Eric Wang. Auditing agent harness safety. arXiv preprint arXiv:2605.14271, 2026.

[93] Fengyuan Liu, Rui Zhao, Shuo Chen, Guohao Li, Philip Torr, Lei Han, and Jindong Gu. Can an individual manipulate the collective decisions of multi-agents? In Proceedings ofthe 2025 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 12158–12182. Association for Computational Linguistics, 2025.

[94] Jiayuan Liu, Shiyi Du, Weihua Du, Mingyu Guo, and Vincent Conitzer. The consensus trap: Rescuing multi-agent LLMs from adversarial majorities via token-level collaboration. arXiv preprint arXiv:2604.17139, 2026.

[95] Jinbo Liu, Defu Cao, Yifei Wei, Tianyao Su, Yuan Liang, Yushun Dong, Yan Liu, Yue Zhao, and Xiyang Hu. Topology matters: Measuring memory leakage in multi-agent LLMs. In Findings of the Associationfor Computational Linguistics: ACL 2026, pages 39728– 39746. Association for Computational Linguistics, 2026.

[96] Qiqi Liu, Thorsten Holz, Shilin Ye, and Runhan Song. The capability paradox: How smarter auditors make multi-agent systems less secure. arXiv preprint arXiv:2605.17480, 2026.

[97] Yan Liu, Sujie Shao, Shaoyong Guo, Chenyu Wang, Chao Yang, and Zhibin Zang. Topology linearization for multi-agent systems security: Mitigating malicious propagation via path decomposition. IEEE Transactions on Network Science and Engineering, 13:8481–8498, 2026.

[98] Zongzhen Liu, Guoyi Li, Bingkang Shi, Xiaodan Zhang, Jingguo Ge, Yulei Wu, and Honglei Lyu. CRDA: Content risk drift assessment of large language models through adversarial multi-agent interaction. In 2024 International Joint Conference on Neural Networks (IJCNN), pages 1–8. IEEE, 2024.

[99] Yaoyang Luo, Zhi Zheng, Ziwei Zhao, Tong Xu, Jielun Zhao, Wenjun Xue, Yong Chen, and Enhong Chen. Defending LLM-based multi-agent systems against cooperative attacks with sentence-level rectification. arXiv preprint arXiv:2605.28104, 2026.

[100] Matteo Lupinacci, Francesco Aurelio Pironti, Francesco Blefari, Francesco Romeo, Luigi Arena, and Angelo Furfaro. The dark side of LLMs: Agent-based attack vectors for system-level compromise. ARES Workshop (AI4TCI), 2026.

[101] Junyuan Mao, Fanci Meng, Yifan Duan, Miao Yu, Xiaojun Jia, Junfeng Fang, Yuxuan Liang, Kun Wang, and Qingsong Wen. AgentSafe: Safeguarding large language model-based multi-agent systems via hierarchical data management. arXiv preprint arXiv:2503.04392, 2025.

[102] Yihuan Mao, Yipeng Kang, Peilun Li, Ning Zhang, Wei Xu, and Chongjie Zhang. IBGP: Imperfect byzantine generals problem for zero-shot robustness in communicative multi-agent systems. In Proceedings ofthe 24th International Conference on Autonomous Agents and Multiagent Systems (AAMAS), pages 2654–2656. International Foundation for Autonomous Agents and Multiagent Systems (IFAA MAS), 2025.

[103] Yanki Margalit, Nurit Cohen-Inger, Erni Avram, Ran Taig, and Oded Margalit. Governed shared memory for multi-agent LLM systems. arXiv preprint arXiv:2606.24535, 2026.

[104] Pablo Mateo-Torrejón and Alfonso Sánchez-Macián. GAMMAF: A common framework for graph-based anomaly monitoring benchmarking in LLM multi-agent systems. arXiv preprint arXiv:2604.24477, 2026.

[105] Yohan Mathew, Ollie Matthews, Robert McCarthy, Joan Velja, Chris tian Schroeder de Witt, Dylan Cope, and Nandi Schoots. Hidden in plain text: Emergence and mitigation of steganographic collusion in LLMs. In Proceedings of the 14th International Joint Conference on Natural Language Processing and the 4th Conference ofthe Asia-Pacific Chapter of the Association for Computational Linguistics (IJCNLP-AACL), pages 585–624. Association for Computational Linguistics, 2025.

[106] Timothy McAllister, Sina Abdidizaji, Ivan Garibay, and Ozlem Ozmen Garibay. Smarter saboteurs, better fixers: Scaling & security in linear multi-agent workflows. In Agents in the Wild (AIWILD) Workshop at ICML, 2026.

[107] Tianyi Men, Pengfei Cao, Zhuoran Jin, Yubo Chen, Kang Liu, and Jun Zhao. A troublemaker with contagious jailbreak makes chaos in honest towns. In Proceedings ofthe 63rd Annual Meeting ofthe Asso ciationfor Computational Linguistics (Volume 1: Long Papers), pages 17561–17587. Association for Computational Linguistics, 2025.

[108] Yuqiao Meng, Sakshi Sunil Narvekar, Luoxi Tang, Rupali Rajendra Vaje, Yingxue Zhang, Muchao Ye, and Zhaohan Xi. EquiMem: Calibrating shared memory in multi-agent debate via game-theoretic equilibrium. arXiv preprint arXiv:2605.09278, 2026.

[109] Rui Miao, Yixin Liu, Yili Wang, Xu Shen, Yue Tan, Yiwei Dai, Shirui Pan, and Xin Wang. BlindGuard: Safeguarding LLM-based multiagent systems under unknown attacks. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 39215–39234. Association for Computational Linguistics, 2026.

[110] Maria Milkowski and Tim Weninger. Deception and communication in autonomous multi-agent systems: An experimental study with among us. In Proceedings ofthe 25th International Conference on Autonomous Agents and Multiagent Systems (AAMAS). International Foundation for Autonomous Agents and Multiagent Systems (IFAA MAS), 2026.

[111] Sumeet Ramesh Motwani, Mikhail Baranchuk, Martin Strohmeier, Vijay Bolina, Philip H. S. Torr, Lewis Hammond, and Christian Schroeder de Witt. Secret collusion among ai agents: Multi-agent deception via steganography. In NeurIPS, volume 37, 2024.

[112] Akshat Naik, Jay Culligan, Yarin Gal, Philip Torr, Rahaf Aljundi, Alasdair Paren, and Adel Bibi. OMNI-LEAK: Orchestrator multiagent network induced data leakage. arXiv preprint arXiv:2602.13477, 2026.

[113] Mason Nakamura, Abhinav Kumar, Saswat Das, Sahar Abdelnabi, Saaduddin Mahmud, Ferdinando Fioretto, Shlomo Zilberstein, and Eugene Bagdasarian. Colosseum: Auditing collusion in cooperative multi-agent systems. arXiv preprint arXiv:2602.15198, 2026.

[114] Mason Nakamura, Abhinav Kumar, Saaduddin Mahmud, Sahar Abdelnabi, Shlomo Zilberstein, and Eugene Bagdasarian. Terrarium: Revisiting the blackboard for multi-agent safety, privacy, and security studies. arXiv preprint arXiv:2510.14312, 2025.

[115] Tam Nguyen, Moses Ndebugre, and Dheeraj Arremsetty. Security con siderations for multi-agent systems. arXiv preprint arXiv:2603.09002, 2026.

[116] Jonathan Nöther, Adish Singla, and Goran Radanovic. Benchmarking the robustness of agentic systems to adversarially-induced harmful behaviors. In Proceedings ofthe 3rd Conference on Language Modeling (COLM), 2026.

[117] Matthew Lyle Olson, Neale Ratzlaff, Musashi Hinck, Tri Nguyen, Vasudev Lal, Joseph Campbell, Simon Stepputtis, and Shao-Yen Tseng. LieCraft: A multi-agent framework for evaluating deceptive capabilities in language models. In Proceedings of the Fortieth AAAI Conference on Artificial Intelligence (AAAI), volume 40, pages 37802– 37809. AAAI Press, 2026.

[118] Aaditya Pai. Blind spots in the guard: How domain-camouflaged injection attacks evade detection in multi-agent LLM systems. arXiv preprint arXiv:2605.22001, 2026.

[119] Junjun Pan, Yixin Liu, Rui Miao, Kaize Ding, Yu Zheng, Quoc Viet Hung Nguyen, Alan Wee-Chung Liew, and Shirui Pan. Explain able and fine-grained safeguarding of LLM multi-agent systems via bi-level graph anomaly detection. In Proceedings of the 64th Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 30491–30506. Association for Computational Linguistics, 2026.

[120] Zhenyu Pan, Yiting Zhang, Zhuo Liu, Yolo Yunlong Tang, Zeliang Zhang, Haozheng Luo, Yuwei Han, Jianshu Zhang, Dennis Wu, Hong Yu Chen, Haoran Lu, Haoyang Fang, Manling Li, Chenliang Xu, Philip S. Yu, and Han Liu. AdvEvo-MARL: Shaping internalized safety through adversarial co-evolution in multi-agent reinforcement learning. In Proceedings of the 43rd International Conference on Machine Learning (ICML), 2026.

[121] Karthik Pappu, Badal Bhushan, and Akshay Mittal. SPIFFE-based zero-trust authentication for AI agent ecosystems. In 2025 International Conference on Computer and Applications (ICCA). IEEE, 2025.

[122] Nickson Patel. Dynamic attentional context scoping: Agent-triggered focus sessions for isolated per-agent steering in multi-agent LLM orchestration. arXiv preprint arXiv:2604.07911, 2026.

[123] KrishnaSaiReddy Patil. SentinelAgent: Intent-verified delegation chains for securing federal multi-agent AI systems. arXiv preprint arXiv:2604.02767, 2026.

[124] Vaidehi Patil, Elias Stengel-Eskin, and Mohit Bansal. The sum leaks more than its parts: Compositional privacy risks and mitigations in multi-agent collaboration. arXiv preprint arXiv:2509.14284, 2025.

[125] Pierre Peigné, Mikolaj Kniejski, Filip Sondej, Matthieu David, Ja son Hoelscher-Obermaier, Christian Schroeder de Witt, and Esben Kran. Multi-agent security tax: Trading off security and collaboration capabilities in multi-agent systems. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 27573–27581, 2025.

[126] Sunil Prakash. LDP: An identity-aware protocol for multi-agent LLM systems. arXiv preprint arXiv:2603.08852, 2026.

[127] Senmao Qi, Yifei Zou, Peng Li, Ziyi Lin, Xiuzhen Cheng, and Dongxiao Yu. Amplified vulnerabilities: Structured jailbreak attacks on LLM-based multi-agent debate. arXiv preprint arXiv:2504.16489, 2025.

[128] Zhaoyang Qu, Mingyang Geng, Yunxin Mao, Shanzhi Gu, Chuanfu Xu, and Haotian Wang. Collaborative-adversarial jailbreaking: A propagation-aware attack framework for multi-agent code generation systems. Neural Networks, 205:109280, 2027. Available online in 2026; assigned to the 2027 journal volume.

[129] Isteak Ahmed Rafe, Md Rasel Dewan, and Md Reajul Islam. Orchestration and verification of agentic ai systems: A survey of multi-agent collaboration and safety. European Journal of Applied Science, Engineering and Technology, 4(2):238–256, 2026.

[130] Viswapriyan Ragupathy. Architectural resilience in ai-driven decision systems under adversarial conditions. In International Conference on Applied Informatics and Communication, pages 1–6, 2026.

[131] Md Hafizur Rahman, Zafaryab Haider, Tanzim Mahfuz, and Prabuddha Chakraborty. HARP: Measuring harm amplification in multiagent LLM systems. arXiv preprint arXiv:2605.27489, 2026.

[132] Shaina Raza, Ranjan Sapkota, Manoj Karkee, and Christos Emmanouilidis. TRiSM for agentic AI: A review of trust, risk, and security management in LLM-based agentic multi-agent systems. AI Open, 7:71–95, 2026.

[133] Alistair Reid, Simon O’Callaghan, Liam Carroll, and Tiberio Caetano. Risk analysis techniques for governed LLM-based multi-agent systems. Technical report, Gradient Institute, 2025.

[134] Qibing Ren, Zhijie Zheng, Jiaxuan Guo, Junchi Yan, Lizhuang Ma, and Jing Shao. When AI agents collude online: Financial fraud risks by collaborative LLM agents on social platforms. In International Conference on Learning Representations, 2026. arXiv:2511.06448.

[135] Jimmy Laurence Rippin, Simon C. Marshall, David Demitri Africa, and Christian Schroeder de Witt. Tool use enables undetectable steganography in multi-agent LLM systems. arXiv preprint arXiv:2606.28425, 2026.

[136] Sandro Rodriguez Garzon, Awid Vaziry, Enis Mert Kuzu, Dennis Enrique Gehrmann, Buse Varkan, Alexander Gaballa, and Axel Küpper. AI agents with decentralized identifiers and verifiable credentials. In ICAART 2026, pages 252–259, 2026.

[137] Aaron Rose, Carissa Cullen, Sahar Abdelnabi, Philip Torr, Brandon Gary Kaplowitz, and Christian Schroeder de Witt. Detecting multi-agent collusion through multi-agent interpretability. arXiv preprint arXiv:2604.01151, 2026.

[138] Rishav Saigal. LLM drift experiment: A framework for quantifying behavioral decay in adversarial multi-agent simulations. Zenodo software, 2026. Software artifact.

[139] Christian Schroeder de Witt, Klaudia Krawiecka, Igor Krawczuk, Ben Hagag, William L. Anderson, Peter Belcak, Ben Bucknall, Xiaohong Cai, Ayush Chopra, Doron Cohen, Ron F. Del Rosario, Andis Draguns, Annie Gray, Keren Katz, Vasilios Mavroudis, Jaron Mink, Sumeet Ramesh Motwani, Jonathan Petit, Leif-Sebastian Rem beck, Chandler Smith, John Sotiropoulos, Steven Young, Sarah Scheffler, and Mary Llewellyn. Open challenges in multi-agent security: Towards secure systems of interacting AI agents. arXiv preprint arXiv:2505.02077, 2025.

[140] Guanquan Shi, Haohua Du, Zhiqiang Wang, Xiaoyu Liang, Weiwenpei Liu, Song Bian, and Zhenyu Guan. SoK: Trust-authorization mismatch in LLM agent interactions. arXiv preprint arXiv:2512.06914, 2025.

[141] Ruxue Shi, Yili Wang, Mengnan Du, Qinggang Zhang, Rui Miao, Yixin Liu, and Xin Wang. SAIGuard: Communication-state simulation for proactive defense of LLM multi-agent systems. In Findings of the Associationfor Computational Linguistics: EMNLP 2026, 2026.

[142] Munish Singh and Namita Mittal. Semantic taint propagation: Embedding-based semantic flow monitoring for multi-agent large language model systems. Zenodo, 2026.

[143] Xiaolin Sun, Zixuan Liu, Yibin Hu, and Zizhan Zheng. Insider attacks in multi-agent LLM consensus systems. arXiv preprint arXiv:2605.08268, 2026.

[144] Yinggang Sun, Haining Yu, Wei Jiang, Xiangzhan Yu, Dongyang Zhan, Lixu Wang, Siyue Ren, Yue Sun, and Tianqing Zhu. A survey on the unique security of autonomous and collaborative LLM agents: Threats, defenses, and futures. Preprints.org, 2026.

[145] Toqeer Ali Syed, Mishal Ateeq Almutairi, and Mahmoud Abdel Moaty. Toward trustworthy agentic AI: A multimodal framework for preventing prompt injection attacks. In 2025 International Conference on Computer and Applications (ICCA), 2025.

[146] Toqeer Ali Syed, Mishal Ateeq Almutairi, and Mahmoud Abdel Moaty. Cross-agent multimodal provenance-aware framework for robust prompt injection defense in large language and vision-language models. In 2025 International Conference on Computer and Applica tions (ICCA). IEEE, 2025.

[147] Zhen Tan, Chengshuai Zhao, Raha Moraffah, Yifan Li, Yu Kong, Tianlong Chen, and Huan Liu. The wolf within: Covert injection of malice into MLLM societies via an MLLM operative. In CVPR Workshop, 2024.

[148] Rizwan Tanveer. Multi-agent orchestration: Coordination, trust, and cascading failures. SSRN, 2026.

[149] Riya Tapwal, Abhishek Kumar, and Carsten Maple. PRISM: Generation-time detection and mitigation of secret leakage in multi agent LLM pipelines. arXiv preprint arXiv:2605.10614, 2026.

[150] Yingtao Tian. Prompt optimization enables stable algorithmic collusion in LLM agents. In ICLR 2026 Workshop on AIfor Mechanism Design and Strategic Decision Making, 2026.

[151] Yu Tian, Xiao Yang, Jingyuan Zhang, Yinpeng Dong, and Hang Su. Evil geniuses: Delving into the safety of LLM-based agents, 2023.

[152] Harold Triedman, Rishi Dev Jha, and Vitaly Shmatikov. Multi-agent systems execute arbitrary malicious code. In COLM, 2025.

[153] Tharakesvulu Vangalapat and Samreen Iftekhar Shaikh. Trustworthy agentic ai: A survey and taxonomy of secure coordination and hallucination mitigation in multi-agent large language model systems. International Journal of Innovative Science and Research Technology, 11(2):1660–1669, 2026.

[154] Kavana Venkatesh, Jafar Isbarov, Saad Amin, Murat Kantarcioglu, and Jiaming Cui. CASPIAN: Online detection and attribution of cascade attacks in LLM multi-agent systems via cross-channel causa monitoring. arXiv preprint arXiv:2605.19240, 2026.

[155] Toby Walsh, Anton van den Hengel, and Stephen J. Roberts. Selfhealing memory architectures for large language model-based multi agent collaboration. American journal ofartificial intelligence and neural networks, 7(1):149–159, 2026.

[156] Haochuan Kevin Wang and Zechen Zhang. Kill-chain canaries: Stagelevel tracking of prompt injection across attack surfaces and model safety tiers. arXiv preprint arXiv:2603.28013, 2026.

[157] Jiye Wang, Shiduo Yang, Ting Qiao, Jiayu Qin, Jianbin Li, Yu Wang, and Yuanhe Zhao. Ev-Trust: An evolutionarily stable trust mechanism for decentralized LLM-based multi-agent service economies. arXiv preprint arXiv:2512.16167, 2025.

[158] Kai Wang, Biaojie Zeng, Zeming Wei, Chang Jin, Hefeng Zhou, Xi angtian Li, Chao Yang, Jingjing Qu, Xingcheng Xu, and Xia Hu. TrinityGuard: A unified framework for safeguarding multi-agent systems. arXiv preprint arXiv:2603.15408, 2026.

[159] Kaixiang Wang, Zhaojiacheng Zhou, Bunyod Suvonov, Jiong Lou, and Jie Li. MAS-shield: A defense framework for secure and efficien LLM MAS. arXiv preprint arXiv:2511.22924, 2025.

[160] Liwen Wang, Wenxuan Wang, Shuai Wang, Zongjie Li, Zhenlan Ji, Zongyi Lyu, Daoyuan Wu, and Shing-Chi Cheung. MASLeak: Inves tigating and exposing intellectual property leakage vulnerabilities in multi-agent systems. In USENIX Security 2026, 2026.

[161] Mengdie Flora Wang, Haochen Xie, Guanghui Wang, Aijing Gao, Guang Yang, Ziyuan Li, Qucy Wei Qiu, Fangwei Han, Hengzhi Qiu, Yajing Huang, Bing Zhu, and Jae Oh Woo. From debate to decision: Conformal social choice for safe multi-agent deliberation. arXiv preprint arXiv:2604.07667, 2026.

[162] Shilong Wang, Guibin Zhang, Miao Yu, Guancheng Wan, Fanci Meng, Chongye Guo, Kun Wang, and Yang Wang. G-safeguard: A topologyguided security lens and treatment on LLM-based multi-agent systems. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 7261– 7276. Association for Computational Linguistics, 2025.

[163] Xiaoqing Wang, Keman Huang, Bin Liang, Hongyu Li, and Xiaoyong Du. Shadows in the code: Exploring the risks and defenses of llmbased multi-agent software development systems. In AAAI, volume 40, pages 37970–37978, 2026.

[164] Chengcan Wu, Zhixin Zhang, Mingqian Xu, Zeming Wei, and Meng Sun. Securing multi-agent systems against corruptions via node contribution backpropagation. In Proceedings ofthe 43rd International Conference on Machine Learning (ICML), 2026.

[165] Jie Wu and Ming Gong. Beyond single-agent alignment: Preventing context-fragmented violations in multi-agent systems. arXiv preprint arXiv:2604.22879, 2026.

[166] Yongxuan Wu, Xixun Lin, He Zhang, Nan Sun, Kun Wang, Chuan Zhou, Shirui Pan, and Yanan Cao. CIA: Inferring the communication topology from LLM-based multi-agent systems. In Proceedings of the 64th Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 17917–17933. Association for Computational Linguistics, 2026.

[167] Yihan Xia, Taotao Wang, Wenxin Xu, and Shengli Zhang. DAO-Agent: Zero knowledge-verified incentives for decentralized multiagent coordination. arXiv preprint arXiv:2512.20973, 2025.

[168] Tianzhe Xiao, Haozhao Wang, Yichen Li, Tianlong Luo, Xiong Yongfu, Daiming Kuang, Yi Wang, Yi Liu, and Ruixuan Li. You don’t have to be the one doing evil! locate a scapegoat within multiagent systems for executing covert attacks. In KDD 2026 Workshop on Federated Learning and Multi-Agent Systems (FedKDD/FedMAS), 2026.

[169] Yizhe Xie, Congcong Zhu, Xinyue Zhang, Tianqing Zhu, Dayong Ye, Minfeng Qi, Huajie Chen, and Wanlei Zhou. From spark to fire: Modeling and mitigating error cascades in LLM-based multi-agent collaboration. arXiv preprint arXiv:2603.04474, 2026.

[170] Yizhe Xie, Congcong Zhu, Xinyue Zhang, Tianqing Zhu, Dayong Ye, Minghao Wang, and Chi Liu. Who’s the mole? modeling and detecting intention-hiding malicious agents in LLM-based multi-agent systems. arXiv preprint arXiv:2507.04724, 2025.

[171] Zixun Xiong, Gaoyi Wu, Lingfeng Yao, Miao Pan, Xiaojiang Du, and Hao Wang. WebWeaver: Breaking topology confidentiality in LLM multi-agent systems with stealthy context-based inference. arXiv preprint arXiv:2603.11132, 2026.

[172] Shuhang Xu and Fangwei Zhong. Comet: Metaphor-driven covert communication for multi-agent language games. In ACL 2025, pages 7892–7917, 2025.

[173] Zijie Xu, Minfeng Qi, Shiqing Wu, Lefeng Zhang, Qiwen Wei, Han He, and Ningran Li. The trust paradox in LLM-based multi-agent systems: When collaboration becomes a security vulnerability. IEEE Transactions on Computational Social Systems, pages 1–12, 2026.

[174] Faouzi El Yagoubi, Godwin Badu-Marfo, and Ranwa Al Mallah. AgentLeak: A benchmark for internal-channel privacy leakage in multi-agent LLM systems. IEEE Access, pages 94960–94978, 2026.

[175] Bingyu Yan, Xiaoming Zhang, Jinyu Hou, Chaozhuo Li, Ziyi Zhou, Xiaozhe Zhang, and Litian Zhang. PropGuard: Safeguarding LLM-MAS via propagation-aware exploration and remediation. arXiv preprint arXiv:2605.16346, 2026.

[176] Bingyu Yan, Xiaoming Zhang, Ziyi Zhou, Chaozhuo Li, Ruilin Zeng, Yirui Qi, Tianbo Wang, and Litian Zhang. Attack the messages, not the agents: A multi-round adaptive stealthy tampering framework for llm-mas. In AAAI, volume 40, pages 29784–29792, 2026.

[177] Yiliu Yang, Yilei Jiang, Qunzhong Wang, Yingshui Tan, Xiaoyong Zhu, Sherman S. M. Chow, Bo Zheng, and Xiangyu Yue. QuadSentinel: Sequent safety for machine-checkable control in multi-agent systems. arXiv preprint arXiv:2512.16279, 2025.

[178] Ziyuan Yang, Wenxuan Ding, Shangbin Feng, and Yulia Tsvetkov. Among us: Measuring and mitigating malicious contributions in

model collaboration systems. In Proceedings of the 64th Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 15969–15988. Association for Computational Linguistics, 2026.

[179] Israel Yankeloviz. Algorithmic cowardice: Cognitive dissonance and moral conformity in multi-agent LLM interactions. Zenodo, 2026.

[180] M Sai Yaswanth, Srinivasarao Chilakala, C. Madana Kumar Reddy, K. Nageswararao, G. Jeevana Manikanta, and Padma Vijetha. Capri-DP: A differentially private extension to context-aware multi-agent privacy frameworks. In 2026 5th International Conference on Communication, Computing and Electronics Systems (ICCCES), pages 1780–1784. IEEE, 2026.

[181] Miao Yu, Fanci Meng, Xinyun Zhou, Shilong Wang, Junyuan Mao, Linsey Pang, Tianlong Chen, Kun Wang, Xinfeng Li, Yongfeng Zhang, Bo An, and Qingsong Wen. A survey on trustworthy llm agents: Threats and countermeasures. In KDD, 2025.

[182] Miao Yu, Shilong Wang, Guibin Zhang, Junyuan Mao, Chenlong Yin, Qijiong Liu, Kun Wang, Qingsong Wen, and Yang Wang. Netsafe: Exploring the topological safety of multi-agent system. In Findings ofACL, 2025.

[183] Weichen Yu, Kai Hu, Tianyu Pang, Chao Du, Min Lin, and Matt Fredrikson. LLM-based multi-agents system attack via continuous optimization with discrete efficient search. In Proceedings of the Second Conference on Language Modeling (COLM), 2025.

[184] Mingming Zha and Xiaofeng Wang. Autonomous LLM agent worms: Cross-platform propagation, automated discovery and temporal reentry defense. arXiv preprint arXiv:2605.02812, 2026.

[185] Lingxi Zhang, Guangtao Zheng, and Hanjie Chen. When embeddingbased defenses fail: Rethinking safety in LLM-based multi-agent systems. In ICML, volume 306 of Proceedings of Machine Learning Research, 2026.

[186] Ruijia Zhang, Xinyan Zhao, Ruixiang Wang, Sigen Chen, Guibin Zhang, An Zhang, Kun Wang, and Qingsong Wen. Safesieve: From heuristics to experience in progressive pruning for llm-based multiagent communication. In AAAI 2026, volume 40, pages 29892–29900, 2026.

[187] Yihao Zhang, Zeming Wei, Xiaokun Luan, Chengcan Wu, Zhixin Zhang, Jiangrong Wu, Haolin Wu, Huanran Chen, Jun Sun, and Meng Sun. AgentWorm: Self-propagating attacks across LLM agent ecosystems. arXiv preprint arXiv:2603.15727, 2026.

[188] Yiting Zhang, Yijiang Li, Tianwei Zhao, Kaijie Zhu, Haohan Wang, and Nuno Vasconcelos. Achilles heel of distributed multi-agent systems. arXiv preprint arXiv:2504.07461, 2025.

[189] Yiyang Zhao, Zhuo Zhang, Qingxuan Le, Lizhen Qu, and Zenglin Xu. Beyond goodhart’s law: A dynamic benchmark for evaluating compliance in multi-agent systems. arXiv preprint arXiv:2606.07805, 2026.

[190] Can Zheng, Yuhan Cao, Xiaoning Dong, and Tianxing He. Demonstrations of integrity attacks in multi-agent systems. arXiv preprint arXiv:2506.04572, 2025.

[191] Lifan Zheng, Jiawei Chen, Qinghong Yin, Jingyuan Zhang, Xinyi Zeng, and Yu Tian. Rethinking the reliability of multi-agent system: A perspective from byzantine fault tolerance. In AAAI, volume 40, pages 35012–35020, 2026.

[192] Hao Zhou, Tiru Wu, Yan Jiang, Wanqi Zhou, Junxing Hu, and Ai Han. Hierarchical attacks for multi-modal multi-agent reasoning. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pat tern Recognition, pages 42331–42340, 2026.

[193] Jialong Zhou, Lichao Wang, and Xiao Yang. GUARDIAN: Safeguarding LLM multi-agent collaborations with temporal graph modeling. In Advances in Neural Information Processing Systems, volume 38, pages 8881–8909. Neural Information Processing Systems Foundation, 2025.

[194] Yijin Zhou, Xiaoya Lu, Dongrui Liu, Junchi Yan, and Jing Shao. INFA-Guard: Mitigating malicious propagation via infection-aware safeguarding in LLM-based multi-agent systems. In EMNLP, 2026.

[195] Zhenhong Zhou, Zherui Li, Jie Zhang, Yuanhe Zhang, Kun Wang, Yang Liu, and Qing Guo. CORBA: Contagious recursive blocking attacks on multi-agent systems based on large language models. In Findings ofACL, pages 6899–6908, 2026.

[196] Zhilun Zhou, Zihan Liu, Jiahe Liu, Qingyu Shao, Yihan Wang, Kun Shao, Depeng Jin, and Fengli Xu. ResMAS: Resilience optimization in LLM-based multi-agent systems. In AAAI, pages 35176–35184, 2026.

[197] Pengyu Zhu, Lijun Li, Yaxing Lyu, Li Sun, Sen Su, and Jing Shao. Collaborative shadows: Distributed backdoor attacks in LLM-based multi-agent systems. arXiv preprint arXiv:2510.11246, 2025.

[198] Chelsea Zou, Yiheng Yao, Selena She, Noah D. Goodman, and Robert D. Hawkins. CalBench: Evaluating coordination-privacy tradeoffs in multi-agent LLMs. arXiv preprint arXiv:2605.09823, 2026.

[199] Zhenhua Zou, Zhuotao Liu, Lepeng Zhao, and Qiuyang Zhan. BlockA2A: Towards secure and verifiable agent-to-agent interoperability. arXiv preprint arXiv:2508.01332, 2025.

## A Evaluation Audit Details

Table 4 provides the detailed mapping for the 44 works in our evaluation audit. It records their A, C, I, P, and R assignments together with observation depth, comparison design, membership setting, artifact availability, and reported system information. The taxonomy codes follow Sections 2–4.

<table><tr><td>Work</td><td>Role</td><td>Adv.</td><td>Config.</td><td>Interface</td><td>Path</td><td>Risk</td><td>Obs.</td><td>Comp.</td><td>Member</td><td>Artifact</td><td>Report</td></tr><tr><td>A2ASecBench [86]</td><td>A+D</td><td>A1-A4</td><td>C1-C8</td><td>I1-I6</td><td>P1-P8</td><td>R1,R3-R7</td><td>Trace</td><td>AC</td><td>Fixed</td><td>URL</td><td>MT</td></tr><tr><td>Achilles Heel [188]</td><td>A</td><td>A3</td><td>C1,C3-C7</td><td>12,14,15</td><td>P2,P4,P5,P7,P8</td><td>R1-R7</td><td>Episode</td><td>Rel</td><td>Fixed</td><td></td><td>MT</td></tr><tr><td>ACIARena [9]</td><td>A+D</td><td>A2-A4</td><td>C1-C3,C5,C6,C8</td><td>11–13,15,16</td><td>P1-P3,P5,P6,P8</td><td>R1,R3–R5,R7</td><td>Episode</td><td>Rel</td><td>Fixed</td><td>URL</td><td>MT</td></tr><tr><td>Agent-BOM [81]</td><td>A</td><td>A1,A3</td><td>C2,C3,C5-C8</td><td>I1-I4,16</td><td>P1-P3,P5-P8</td><td>R1,R4–R7</td><td>Trace</td><td>None</td><td>Spawn</td><td></td><td>M</td></tr><tr><td>AgentLeak [174]</td><td>A</td><td>A1-A4</td><td>C1-C8</td><td>I1–I4,I6</td><td>P1-P3,P5-P8</td><td>R1,R5</td><td>Trace</td><td>SP</td><td>Fixed</td><td>URL</td><td>NMT</td></tr><tr><td>AgentXposed [170]</td><td>A+D</td><td>A3</td><td>C1,C3,C4,C6,C8</td><td>I2,15,16</td><td>P2,P4,P5,P8</td><td>R1-R4,R7</td><td>Events</td><td>AC</td><td>Fixed</td><td>URL</td><td>NMT</td></tr><tr><td>Algorithmic Cowardice [179]</td><td>A</td><td>A3</td><td>C1-C4,C6-C8</td><td>12,16</td><td>P4,P5</td><td>R1,R4</td><td>Events</td><td>AC</td><td>Fixed</td><td>URL</td><td>NMT</td></tr><tr><td>Among Us [110]</td><td>A+D</td><td>A3</td><td>C1,C3,C4,C6,C8</td><td>12,15,16</td><td>P4,P5</td><td>R2,R3</td><td>Events</td><td>AC</td><td>Fixed</td><td>URL</td><td>MT</td></tr><tr><td>ATAG [42]</td><td>A</td><td>A1,A2</td><td>C1-C3,C5,C7,C8</td><td>I1-I5</td><td>P1-P3,P5,P7,P8</td><td>R1,R4–R6</td><td>Events</td><td>None</td><td>Fixed</td><td>URL</td><td>MT</td></tr><tr><td>BAD-ACTS [116]</td><td>A+D</td><td>A1-A4</td><td>C1-C8</td><td>I1-I6</td><td>P1-P8</td><td>R1-R7</td><td>Trace</td><td>AC</td><td>Fixed</td><td>URL</td><td>MT</td></tr><tr><td>CalBench [198]</td><td>S</td><td></td><td>C1-C8</td><td>I2-I6</td><td>P3-P8</td><td>R2,R3,R5,R7</td><td>Trace</td><td>Rel</td><td>Fixed</td><td>Claim</td><td>NMT</td></tr><tr><td>Capability Paradox [96]</td><td>A</td><td>A2</td><td>C1–C4,C7,C8</td><td>I1,12,14,16</td><td>P1,P2,P5,P7</td><td>R1,R4,R6</td><td>Trace</td><td>Rel</td><td>Fixed</td><td></td><td>MT</td></tr><tr><td>Child Inherits [21]</td><td>A</td><td>A2,A3</td><td>C1,C3,C5-C8</td><td>11,13,14,16</td><td>P1-P3,P5-P8</td><td>R1,R4-R7</td><td>Final</td><td>Rel</td><td>Spawn</td><td></td><td>T</td></tr><tr><td>Collab. Jailbreaking [128]</td><td>A</td><td>A2</td><td>C1-C8</td><td>I1-I6</td><td>P1-P5,P7</td><td>R1,R3,R4,R6</td><td>Events</td><td>SP</td><td>Fixed</td><td></td><td>MT</td></tr><tr><td>Collusion Interpretability [137]</td><td>A</td><td>A3</td><td>C1-C5,C8</td><td>I2-I6</td><td>P2,P4,P5,P7</td><td>R2-R4,R6</td><td>Events</td><td>AC</td><td>Fixed</td><td>URL</td><td>NT</td></tr><tr><td>Colosseum [113]</td><td>A+D</td><td>A3</td><td>C1–C5,C7,C8</td><td>I2,14-I6</td><td>P2-P5</td><td>R1-R3</td><td>Trace</td><td>Rel</td><td>Fixed</td><td>URL</td><td>MT</td></tr><tr><td>CRDA [98]</td><td>A</td><td>A3</td><td>C2-C6</td><td>12,13,15,16</td><td>P2-P5</td><td>R1,R3,R4</td><td>Events</td><td>Rel</td><td>Fixed</td><td></td><td>N</td></tr><tr><td>Defection [77]</td><td>A</td><td>A3</td><td>C2-C8</td><td>I2-I6</td><td>P3-P5,P7,P8</td><td>R2-R4,R7</td><td>Events</td><td>AC</td><td>Fixed</td><td></td><td>N</td></tr><tr><td>GAMBIT [78]</td><td>A+D</td><td>A3</td><td>C1-C4,C6,C8</td><td>I2,15,16</td><td>P4,P5</td><td>R2,R3</td><td>Events</td><td>SP</td><td>Fixed</td><td>URL</td><td>MT</td></tr><tr><td>GAMMAF [104]</td><td>A+D</td><td>A3</td><td>C1-C4,C6-C8</td><td>12,15,16</td><td>P2-P5,P8</td><td>R1-R3,R7</td><td>Events</td><td>AC</td><td>Fixed</td><td>URL</td><td>NMT</td></tr><tr><td>Harness Audit [92]</td><td>A</td><td>A1</td><td>C1-C5,C7,C8</td><td>I1-I6</td><td>P1,P2,P6-P8</td><td>R1,R4-R6</td><td>Trace</td><td>SP</td><td>Fixed</td><td>URL</td><td>MT</td></tr><tr><td>HARP [131]</td><td>A+D</td><td>A3,A4</td><td>C1-C5,C7,C8</td><td>I2-I6</td><td>P2,P4,P5,P7</td><td>R1,R3,R4,R6</td><td>Trace</td><td>AC</td><td>Fixed</td><td></td><td>NMT</td></tr><tr><td>Hidden in Plain Text [105]</td><td>A+D</td><td>A3</td><td>C2-C4,C6,C8</td><td>12,15,16</td><td>P4,P6</td><td>R2,R3,R5</td><td>Events</td><td>DC</td><td>Fixed</td><td></td><td></td></tr><tr><td>LieCraft [117]</td><td>A+D</td><td>A3</td><td>C1,C3,C4,C6,C8</td><td>12,15,16</td><td>P4,P5</td><td>R2,R3</td><td>Events</td><td>AC</td><td>Fixed</td><td>URL</td><td>NMT</td></tr><tr><td></td><td>A</td><td>A3</td><td>C1-C8</td><td>12,13,15,16</td><td>P2-P5</td><td>R2-R4</td><td>Trace</td><td>None</td><td>Fixed</td><td>URL</td><td>MT</td></tr><tr><td>LLM Drift [138]</td><td>A</td><td>A3</td><td>C1-C7</td><td>12,13,15,16</td><td>P3-P6</td><td>R2-R5</td><td>Events</td><td>Rel</td><td>Fixed</td><td>Claim</td><td>MT</td></tr><tr><td>LLM-Deliberation [2]</td><td>A+D</td><td>A2,A3</td><td>C1–C5,C7,C8</td><td>I1-I6</td><td>P1,P3-P8</td><td>R3-R7</td><td>Trace</td><td>SP</td><td>Fixed</td><td>Claim</td><td>MT</td></tr><tr><td>MAC-Bench [189]</td><td></td><td></td><td>C1-C8</td><td>I2,13,15,16</td><td>P3-P6</td><td>R2-R5</td><td>Trace</td><td>DC</td><td>Fixed</td><td>URL</td><td>NMT</td></tr><tr><td>MAGPIE [66]</td><td>SA</td><td>A3</td><td>C1–C3,C5,C6,C8</td><td>12,13,16</td><td>P2,P3,P5,P6</td><td>R1,R5</td><td>Events</td><td>Rel</td><td>Fixed</td><td>URL</td><td>NMT</td></tr><tr><td>MAMA [95]</td><td>A+D</td><td>A2,A3</td><td>C1-C6,C8</td><td>11,12,15,16</td><td>P1,P2,P4-P6</td><td>R1,R3,R4</td><td>Events</td><td>Rel</td><td>Fixed</td><td>Claim</td><td>NMT</td></tr><tr><td>MedSentry [28]</td><td>A</td><td>A4</td><td>C1–C4,C7,C8</td><td>12,15,16</td><td>P2,P3,P5</td><td>R1,R3</td><td>Events</td><td>Rel</td><td>Fixed</td><td>URL</td><td>T</td></tr><tr><td>MESA [83]]</td><td>A</td><td>A3</td><td>C1–C4,C6,C8</td><td>I2,15,16</td><td>P2-P5</td><td>R1,R3</td><td>Events</td><td>Rel</td><td>Fixed</td><td>URL</td><td>NT</td></tr><tr><td>NetSafe [182]</td><td>A</td><td>A1,A4</td><td>C1–C5,C7,C8</td><td>11,12,14–I6</td><td>P1,P2,P7,P8</td><td>R1,R4,R6</td><td>Trace</td><td>SP</td><td>Fixed</td><td></td><td>NMT</td></tr><tr><td>OpenClaw Security [60]</td><td>A</td><td>A2,A4</td><td>C2–C5,C7,C8</td><td>I1-I4,I6</td><td>P1-P3,P5-P8</td><td>R4-R7</td><td>Trace Events</td><td>Rel SP</td><td>Fixed Fixed</td><td>Claim</td><td>NM NMT</td></tr><tr><td>PEAR [36] Privacy Leakage [17]</td><td>A A+D</td><td>A3 A3</td><td>C1-C6,C8 C1-C5,C8</td><td>I1-I4,16 I2-I6</td><td>P1,P2,P4–P7 P4,P5</td><td>R1,R2,R5,R6</td></table>

<sub>security-evaluation</sub> <sub>works.</sub> <sub>Role</sub> <sub>gives</sub> <sub>attack</sub> <sub>or</sub> <sup>failure</sup> <sup>evaluation</sup> <sup>(A),</sup> <sup>defense</sup> <sup>evaluation</sup> <sup>(D),</sup> <sup>and</sup> <sup>security-p</sup> <sub>.</sub> <sub>is</sub> <sub>the</sub> <sub>deepest</sub> <sub>data</sub> <sub>consumed</sub> <sub>by</sub> <sub>the</sub> <sub>evaluation:</sub> <sub>final</sub> <sub>output,</sub> <sub>episode</sub> <sub>outco</sub>m<sup>e,</sup> <sup>principal</sup> <sup>or</sup> <sup>message</sup> <sup>events,</sup> <sub>son:</sub> <sub>attack</sub> <sub>or</sub> <sub>component</sub> <sub>control</sub> <sub>(AC),</sub> <sub>defense</sub> <sub>or</sub> <sub>co</sub>m<sup>ponent</sup> <sup>control</sup> <sup>(DC),</sup> <sup>matched</sup> <sup>single-principal</sup> <sup>compa</sup> <sub>none.</sub> M<sup>ember</sup> <sup>reports</sup> <sup>fixed</sup> <sup>membership</sup> <sup>or</sup> <sup>limited</sup> <sup>runtime</sup> <sup>spawning.</sup> <sup>Artifact</sup> <sup>distinguishes</sup> <sup>an</sup> <sup>exact</sup> <sup>evalu</sup> <sub>ecoverable</sub> <sub>URL</sub> <sub>(Claim),</sub> <sub>an</sub> <sub>upstream-o</sub>n<sup>ly</sup> <sup>URL</sup> <sup>(Up),</sup> <sup>and</sup> <sup>no</sup> <sup>identified</sup> <sup>artifact</sup> <sup>(–).</sup> <sup>Report</sup> <sup>marks</sup> <sup>an</sup> <sup>exact</sup> <sub>),</sub> <sub>and</sub> <sub>com</sub>m<sup>unication</sup> <sup>topology</sup> <sup>(T).</sup> <sup>Taxonomy</sup> <sup>codes</sup> <sup>follow</sup>