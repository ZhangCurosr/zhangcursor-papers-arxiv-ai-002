# Scientific Liquidity Agents: An Agent-Based Model of Knowledge Arbitrage in Scientific Communities<sup>∗</sup>

Koutian Wu<sup>1,2</sup>, You Liu<sup>3</sup>, Yue Liu<sup>3</sup>, Zhiping Li<sup>4</sup>, Ergan Shang<sup>5</sup>

<sup>1</sup>The University of Texas at Austin <sup>2</sup>TaciteAI <sup>3</sup>Independent Researcher <sup>4</sup>Princeton University <sup>5</sup>Carnegie Mellon University

k@tacite.ai

## Abstract

Scientific knowledge gains new uses when ideas, methods, and results move between research contexts. Understanding this process requires a model of both specialized knowledge produc tion and the intermediary activities that connect it. We introduce Scientific Liquidity Agents (SLA), an agent-based model of knowledge arbitrage in scientific communities. SLA represents domain specialists and epistemic arbitrageurs within a shared environment of research problems, researcher interactions, and scientific artifacts. Individual decisions connect to collective state through a belief–desire–intention architecture, a prerequisite-based problem graph, a dynamic interaction network, and a persistent publication registry. The framework makes researcher roles and knowledge-transfer frictions explicit, providing a process-oriented account of scientific production and exchange. Two exploratory simulations with 10 and 18 agents produce 68 and 274 registered papers. In the smaller run, arbitrageurs take part in 16 of the 26 contact pairs that join this community into one connected component, while topical similarity alone splits it into seven. The same run yields three similarity edges, 26 contact pairs, and two coauthorship pairs, so measured community structure depends on what counts as a link. Paper production is abundant, but no recorded replication succeeds, which places the bottleneck in verification rather than production. Solved-problem sets correspond to the prerequisite reachability of the disciplines represented in each community. SLA makes knowledge liquidity, the movement and reuse of artifacts across domains, a quantity that agentic science can simulate and measure.

Keywords: knowledge arbitrage; agent-based modeling; science of science; knowledge liquidity;   
scientific communities; epistemic intermediaries.

## 1 Introduction

Scientific progress depends on the circulation of existing knowledge as well as the production of new results. A method developed in one field may become an input to another, while a useful result may remain separated from potential users by disciplinary language, unfamiliar tools, or the efort required to adapt it. These movements connect individual research activities to the collective development of science [2]. They also create a role for researchers who work across disciplinary boundaries and help bring otherwise separated knowledge into contact.

We use knowledge arbitrage to describe the pursuit of value from diferences in where knowledge is available, understood, and potentially useful. An epistemic arbitrageur occupies an intermediary position: its research profile emphasizes cross-domain activity, cooperation, and tools that can support the handling of unfamiliar work. Knowledge liquidity describes the ease with which scientific artifacts can be discovered, understood, reproduced, and incorporated into subsequent research. Together, these concepts bring the scientific division of labor into view: specialists develop knowledge within domains, while intermediaries are positioned to connect knowledge across them.

Studying this process requires following multiple forms of change at once. Researchers choose activities and partners; interactions alter their contact histories; publication adds artifacts to a shared pool; and progress on one problem changes the prerequisites available for later work. A computational model can place these processes within a common environment and expose the relationships between local activity and collective state. The central question motivating this work is: How can knowledge arbitrage be represented as a process within an evolving scientific community, and what patterns of production, interaction, and cumulative progress arise in that environment?

We introduce Scientific Liquidity Agents (SLA), an agent-based model organized around this question. SLA combines domain specialists and epistemic arbitrageurs with a shared problem graph, interaction network, and publication registry. Researcher decisions use either an LLMbased structured-memory policy or an equation-based policy. Actions update the environment, whose evolving state informs subsequent decisions. This structure places specialized production and intermediary activity within the same continuing research process.

Our contributions are threefold:

1. A process-oriented model of knowledge arbitrage. We connect knowledge production, cross-domain interaction, and attempted reuse through explicit representations of heterogeneous researchers, transfer frictions, and intermediary roles. This formulation makes knowledge arbitrage a concrete object of computational investigation.

2. An integrated multi-agent framework. SLA links researcher decisions to a prerequisitebased problem graph, a dynamic interaction network, and a persistent publication registry, connecting individual activities with collective scientific state.

3. An exploratory characterization of scientific-community dynamics. Two simulations describe publication, cross-role contact, coauthorship, and problem progress. They exhibit extensive cross-role contact alongside sparse joint production, and correspondences between solved-problem sets, disciplinary coverage, and prerequisite structure.

The paper develops the process model and its implementation, then uses the two simulation cases to examine how its diferent components evolve together. The resulting contribution is a computational account of knowledge arbitrage as a form of scientific division of labor.

## 2 Related Work

Scientific communities and interdisciplinary knowledge. Science-of-science research studies how individual choices, collaboration, and institutions shape collective scientific activity [2]. Agent-based models have examined paper generation and the formation of specialties [3], communication and scientific learning [11], and the relationship between individual research choices and aggregate interdisciplinarity [9]. SLA develops this tradition around a particular process: the coexistence of specialized knowledge production and cross-domain intermediary activity. It represents that process through persistent problems, contacts, and artifacts.

Language-model agents and social simulation. BDI architectures organize agent state into beliefs, desires, and intentions [10]. Generative Agents connects memory, reflection, and planning in a social environment [8], while EconAgent studies LLM-driven activity in a macroeconomic setting [5]. MatrAIx develops heterogeneous persona agents and interaction environments for simulated-user evaluation [6]. SLA uses agent-based simulation to examine a domain-specific social process: how researcher roles and activities relate to the production and circulation of scientific knowledge.

Scientific ideation and research ecosystems. IdeaScientist retrieves mechanisms from analogous problems across fields to support grounded scientific ideation [7]. Academic Simulacra studies research-idea forecasting through multi-agent simulations [1]. SciUtopia models continuing academic ecosystems and investigates institutional mechanisms through counterfactual simulation worlds [4]. These studies address complementary scales of scientific activity, from individual proposals to institutional dynamics. SLA centers the intermediary role and the relationship among knowledge production, cross-domain contact, and prospective reuse. Its distinctive emphasis is the process through which a community’s division of labor interacts with the structure of its accumulating knowledge.

## 3 Modeling Knowledge Arbitrage

## 3.1 Knowledge Production, Intermediation, and Use

SLA models a scientific community as researchers acting on a shared collection of problems and artifacts. The process begins with specialized research: an agent works on a problem, contributes to its solution, and may publish a structured artifact. Publication makes that artifact available to later decisions. Collaboration creates contact between researchers, and replication provides an action through which an agent can attempt to use published work. These activities place knowledge creation and prospective circulation within one evolving system.

The knowledge-market analogy organizes three elements of the model: reusable assets, frictions in their movement, and intermediaries. Scientific knowledge is non-exclusive: one researcher’s use can coexist with use by others. The model therefore focuses on access, efort, and use rather than ownership transfer or a transaction price. In the simulations reported here, an asset is a structured paper record with an identifier, authors, a publication step, and synthetic attributes. Problem completion, publication, and replication are state transitions in this computational environment.

Table 1 maps the conceptual process to the simulator. Discoverability concerns which opportunities enter an agent’s decisions. Tooling and replication costs concern the efort needed to handle a candidate artifact. Intermediary profiles combine cross-domain positioning with tool proficiency and cooperative tendencies. Knowledge liquidity provides the organizing question linking these elements: how readily can produced knowledge enter subsequent research?

## 3.2 Researcher Roles and Scientific Division of Labor

Specialists and arbitrageurs instantiate the same agent class and share the action vocabulary. Specialists have named disciplinary assignments and topic profiles. Arbitrageurs have a generic crossdomain assignment, stronger initialized tool proficiency, and additional cooperation and helping parameters. This configuration represents the intermediary as a recognizable research role within the community. Table 2 specifies the role profiles used by the project runner.

Table 1: Knowledge-arbitrage concepts and their representation in SLA.
<table><tr><td>Concept</td><td>Process interpretation</td><td>Model representation</td></tr><tr><td>Knowledge</td><td>Developing and communicating a</td><td>Research actions, solved-problem records, and registered papers.</td></tr><tr><td>production Intermediation</td><td>result Connecting research across domains</td><td>Epistemic-arbitrageur profiles and</td></tr><tr><td></td><td>Encountering relevant people and</td><td>cross-role collaboration opportunities Ranked partner candidates,</td></tr><tr><td>Discovery</td><td>artifacts</td><td>recent-paper candidates, and profile visibility.</td></tr><tr><td>Adaptation effort</td><td>Handling unfamiliar knowledge or tools</td><td>Tool proficiency, action costs, and domain-mismatch terms.</td></tr><tr><td>Attempted reuse</td><td>Trying to reproduce or use an artifact</td><td>Replication selection and downstream sampled outcomes.</td></tr><tr><td></td><td>Cumulative progress Earlier results enabling further work</td><td>Problem prerequisites and persistent solver state.</td></tr></table>

Table 2: Role initialization. Beta(a, b) denotes a beta draw; additive boosts are capped at one. The two roles use the same agent architecture.
<table><tr><td>Attribute</td><td>Specialist</td><td>Arbitrageur</td></tr><tr><td>Domain and topic</td><td>Named discipline and topic</td><td>cross_domain; interdisciplinary</td></tr><tr><td>Skills</td><td>Programming/writing: Beta(2, 2); mathematics:</td><td>Same draws with +0.2 per skill</td></tr><tr><td></td><td>Beta(2, 3); domain knowledge: Beta(3, 2)</td><td></td></tr><tr><td>Cooperation Tool proficiency</td><td>Beta(2, 2)</td><td>Same draw with +0.3</td></tr><tr><td>Visibility parameter</td><td>Beta(2, 3) per tool Beta(2, 3)</td><td>Beta(4, 1) per tool Beta(4, 2)</td></tr><tr><td>Incentive weights</td><td>Citations: Beta(3, 2); workload:</td><td>Same weights, plus helping:</td></tr><tr><td>Other tendencies</td><td>Beta(2, 2); innovation: Beta(2, 3) Exploration: Beta(1.5, 3);</td><td>Beta(4,1) Same distributions</td></tr></table>

Role parameters enter the model through diferent channels. Skills, tools, and cooperation afect synthetic action costs and success rules. Visibility is a profile field presented to the LLM; actual paper candidates come from the recent-publication registry. Helping is an initialized incentive field. The core runner supplies partner and paper candidates directly, while an additional networkinformation interface is available as an extension. Appendix B.2 gives the exact candidate-selection rules, including the treatment of disciplinary labels.

![](images/436b4fa52775691d7cb106a7a50e55490597618e78510c40bb8c5f7358b0b613.jpg)  
BDI denotes decision-state organization; LLM and equation policies share the action vocabulary.

Figure 1: SLA connects individual researcher decisions to shared scientific state. Role profiles inform the belief–desire–intention decision state. The five action types update problems, interactions, and publications, whose observations feed back into later decisions. The BDI sequence denotes conceptual state organization shared by the LLM and equation policies.

## 4 The SLA Framework

## 4.1 Individual Decisions and Shared State

Figure 1 shows the decision–environment loop. Each researcher carries skills, tool proficiencies, behavioral tendencies, incentive weights, a knowledge inventory, and BDI state. Beliefs summarize perceived conditions, desires express goals, and intentions select actions. Shared scientific state consists of the problem graph, interaction state, and publication registry. A researcher’s actions change the applicable shared records, which then inform subsequent decisions.

The LLM policy produces reasoning, structured-memory updates, a plan, and an action in a unified response. Numeric beliefs are updated separately. Parsed actions use the simulator’s action vocabulary; invalid action names default to research, and missing or unparseable responses invoke equation-based intention selection. The equation policy uses numeric belief updates and rule-based goal and action selection. For example, the sentiment update combines persistence, observation, and social terms with coeficients 0.7, 0.3, and 0.2, followed by clipping to [0, 1].

Core actions are research, publication, replication, collaboration, and tool adoption. Each action has a synthetic efort cost modified by relevant profile attributes. The agent layer samples action success and updates local state. The environment then applies the corresponding operation: recording problem progress, deciding publication acceptance, resolving collaboration, or sampling a replication outcome. Appendix B.3 specifies these rules. The sequence gives each activity a place in the ongoing community process, from individual choice to persistent shared output.

## 4.2 Problems and Cumulative Research

The problem graph contains research problems in mathematics, physics, chemistry, biology, and computer science. Each problem has a domain, dificulty, importance, prerequisites, and solver fields. Prerequisite relations represent cumulative dependence: progress on earlier problems can make later problems eligible for research. The graph therefore connects the temporal course of activity to the structure of the community’s knowledge.

Problem queries use disciplinary assignments and prerequisite state to generate candidates. A successful research action can update solver fields and change later opportunities. Cross-domain dependencies make some local progress contingent on work in another field. The graph provides a common representation for studying both within-domain accumulation and interdependence across domains.

## 4.3 Interaction and Artifact Circulation

The interaction layer maintains time-decayed topic weights and persistent collaborator sets. Current similarity combines weighted Jaccard overlap with a contact bonus; a threshold defines current network edges. Partner candidates are ranked using similarity, role, domain, and contact history. In the ordinary collaboration branch, the agent selects the first returned candidate and the environment applies partner acceptance. Accepted interactions update the persistent contact record.

The publication registry stores accepted paper identifiers, authors, publication steps, and synthetic venue tiers. Registered publications enter the recent-paper pool used in later decisions. Replication selects among the latest 20 identifiers, with a preference for same-domain candidates when available. Its downstream success rule combines artifact clarity, agent programming skill, domain mismatch, and an optional friction-reduction term. Successful downstream draws record citations. These rules provide a stylized representation of artifact circulation and use within the simulated community.

## 4.4 Observing the Process

We examine four linked aspects of the model: production, interaction, problem progress, and replication. Production counts unique registered papers and their authorship. Interaction has three complementary representations: current similarity, persistent contact, and shared authorship. Problem progress follows solver state on the prerequisite graph. Replication records attempts and downstream outcomes under the simulator’s rule.

The unit of each observable follows its role in the process. A contact is an opportunity for interaction; a coauthored paper is a joint product; a downstream replication outcome is a sampled use outcome. Keeping these units explicit allows their relationships to be studied together. Crossrole contact coverage uses $n _ { S } n _ { A }$ possible pairs, where $n _ { S }$ and $n _ { A }$ are the numbers of specialists and arbitrageurs. Within-role coverage uses $\binom { n _ { t } } { 2 }$ . Publication credit is divided equally among registered authors, preserving the total number of papers.

The framework also includes an extension for artifact-level time to reuse, reuse frequency, replication throughput, and creation–reuse cost diferences. These measures are implemented in a separate runner. The present cases use the production, interaction, problem, and replication records described above.

## 5 Exploratory Simulation Setup

We study two existing SLA simulations. The primary community contains eight specialists and two arbitrageurs over 100 steps. The extended community contains sixteen specialists and two arbitrageurs over 200 steps. Table 3 summarizes their configurations and outputs. The cases provide two settings in which to examine how the framework’s components operate together.

Table 3: Simulation configurations and recorded outputs. Token counts are reported by the saved run summaries.
<table><tr><td>Measure</td><td>Primary run</td><td>Extended run</td></tr><tr><td>Agents (specialists + arbitrageurs)</td><td>8 + 2</td><td>16 + 2</td></tr><tr><td>Simulation steps</td><td>100</td><td>200</td></tr><tr><td>Unique registered papers</td><td>68</td><td>274</td></tr><tr><td>Recorded replication outcomes</td><td>0</td><td>6</td></tr><tr><td>Successful replications</td><td>0</td><td>0</td></tr><tr><td>Solved problems / all records</td><td>134 / 510</td><td>360 / 510</td></tr><tr><td>Reported LLM tokens</td><td>1,511,980</td><td>6,026,553</td></tr></table>

The primary specialists cover mathematics, chemistry, and biology; the extended community additionally includes physics and computer science specialists. Both cases use the same 510 problem definitions and 1,097 prerequisite edges. Both are documented as LLM-enabled; the primary protocol names Gemini 2.0 Flash and seed 42. Appendix D describes the preserved inputs and computational reconstruction.

We count unique paper identifiers, derive interaction graphs from saved records, and compare solved-problem sets with prerequisite reachability. Starting with an empty set, the reachability calculation repeatedly adds problems whose prerequisites are already present. A domain-restricted calculation considers only domains represented by specialists in the corresponding roster. This static construction supplies a structural description against which to examine the simulated endpoints. The analysis characterizes each community and compares its observable patterns across these two configurations.

## 6 Production, Interaction, and Cumulative Progress

## 6.1 Scientific Production in the Two Communities

The primary and extended simulations produce 68 and 274 unique registered papers, respectively. The primary registry contains two joint papers and the extended registry five. Figure 2 follows their accumulation over simulation steps; cumulative registry counts agree with all retained snapshots. These artifacts form the shared publication pool that later researcher decisions can encounter.

Both roles contribute registered output. Fractional authorship credit assigns 58 papers to specialists and 10 to arbitrageurs in the primary community, and 246.5 and 27.5 in the extended community. Per-agent averages are 7.25 and 5.00 in the primary case, and 15.41 and 13.75 in the extended case. This allocation locates each role within the community’s production activity; the community sizes, disciplinary composition, and horizons are given in Table 3.

Individual output also varies within roles. The two primary arbitrageurs receive fractional credits of 1 and 9; their extended-case counterparts receive 12.5 and 15. Figure 6 preserves the individual distributions, complementing the role averages. Token trajectories and venue distributions appear in Figures 7 and 8.

![](images/1e8332a54f7dd3dc3481590d2f9407f080e6bd0510ce0a050bf4e126081a527d.jpg)  
Figure 2: Accumulation of unique registered papers in the two communities. Curves use the archived step sequence and publication registry, ending at 68 and 274 papers. Each legend identifies the community size and horizon.

## 6.2 Researcher Activity and Division of Labor

Figure 3 shows selected-action composition using each role’s total agent-actions as its denominator. Primary specialists allocate 58.75% to research, 23.00% to publication, and 18.25% to collaboration; arbitrageurs allocate 70.00%, 21.50%, and 8.50%. The extended community has the same ordering of activities within each role. Its eleven replication selections occur among specialists, accounting for 0.34% of their actions. Neither community selects tool adoption.

These activity profiles connect the configured division of labor to the choices made within the simulations. Arbitrageurs participate in direct research and publication as well as collaboration. The selected-action mix describes this participation alongside the contact and output patterns examined below. Appendix B.1 gives the exact percentages and denominators.

## 6.3 Cross-Role Contact and Joint Production

Persistent contact reaches every specialist–arbitrageur pair: 16 of 16 pairs in the primary simulation and 32 of 32 in the extended simulation. Joint authorship is more selective. No cross-role pair shares a registered paper in the primary case, while three such pairs do in the extended case. Neither final similarity graph contains a cross-role edge. Figure 4 shows the three networks on the same node layout for the primary community. Figure 10 in the appendix compares normalized pair coverage across both communities.

These patterns locate contact and production at diferent stages of the modeled process. Persistent contact accumulates accepted encounters, current similarity reflects time-dependent topic overlap, and coauthorship identifies shared registered output. The partner-selection policy favors new cross-role contacts, giving a concrete route by which the contact network expands. Its saturated cross-role structure accompanies a much sparser structure of joint production.

## 6.4 Disciplinary Coverage and Knowledge Dependencies

The primary and extended cases finish with 134 and 360 solved problems. Their solved sets exactly match the prerequisite closures restricted to the domains represented in the respective specialist rosters (Appendix Figure 11). The full-domain closure contains 360 problems. The remaining 150 belong to a 13-node cyclic component and its descendants. The primary roster introduces an additional domain restriction within this shared dependency structure.

![](images/0334445982fe4693a6ea8f4434311a2141f9daa4836ea778f23bfad2a191b8e6.jpg)

Figure 3: Selected-action allocation by role. Percentages use 800/200 specialist/arbitrageur actions in the primary community and 3,200/400 in the extended community. The extended specialists’ 0.34% replication share is labelled explicitly. Tool adoption is zero in all four groups. These are action-choice proportions.  
![](images/68ac67a2650205b6e3fcba6b68589e52310e8771439d250eb44e69b6ea2530c0.jpg)  
Figure 4: Three network views of the primary community, with researcher positions held fixed across panels. Blue nodes are specialists and orange nodes are arbitrageurs. Panel A shows final similarity edges, panel B shows unique recorded collaborator pairs, and panel C connects coauthors of registered papers. Orange edges connect diferent roles. Both arbitrageurs contact every specialist, while joint authorship is concentrated in two specialist pairs.

Figure 5 places disciplinary completion alongside a concrete prerequisite path. Biology illustrates how disciplinary and structural views complement one another. Both communities solve the same 86 biology problems. A closure restricted to biology alone reaches 84; the additional two problems require progress through a mathematics prerequisite. This explicit dependency links a local disciplinary outcome to work elsewhere in the knowledge graph. Appendix C.1 gives the problem identifiers and closure construction.

Equal endpoints also coexist with diferent trajectories. Among the 86 shared biology solutions,

c A cross-domain prerequisite path  
![](images/aa6bd7999e17fe174728f2b7a48e47be31eb0c61fe7ebf5e92463702b756ff34.jpg)

![](images/becdd3a7088621374a85acf1f48e60d04c69c7181f42070980afc44c6040be7b.jpg)

![](images/70d235f5e33722febaaf332a679a738eeff38f3cbf49f9a632a3f4a70ebad4cf.jpg)  
Edges denote problem prerequisites. Biology-only closure: 84; full-graph biology closure: 86.  
Figure 5: Disciplinary progress and a cross-domain prerequisite path. Panels a and b show solved/total problem records for every discipline. Panel c follows one mathematics prerequisite into two biology problems. Its arrows are prerequisite relations in the problem graph. Biology-only closure reaches 84 problems, while the full-graph biology closure and both saved biology solved sets contain 86.

84 solve steps and 57 paper-association lists difer between the two cases. Endpoint reachability describes the common structural destination, while the saved histories describe diferent temporal paths through it. These observations connect community composition, knowledge dependence, and the timing of cumulative research.

## 6.5 Replication Within the Activity Mix

Replication is infrequent in these simulations. The primary summary contains no replication selection. The extended summary contains 11 selections and six agent-layer successes, together with six downstream records: one self-target and five other-agent targets. All six downstream outcomes are unsuccessful, and both final citation graphs are empty.

Within the modeled production–interaction–use process, the two cases therefore concentrate activity in artifact creation and researcher contact. Replication supplies a further, separately resolved stage. Its outcome depends on synthetic clarity, skill, and domain compatibility, as specified in Appendix B.4. The cases give a concrete example of a scientific community with an expanding artifact pool and broad cross-role contact, yet little recorded activity at the downstream replication stage.

## 7 Discussion

## 7.1 Knowledge Arbitrage as Scientific Division of Labor

SLA makes knowledge arbitrage a question about the organization of research activity. Specialists and intermediaries participate in a shared environment, contribute artifacts, form contacts, and act within a common dependency structure. The intermediary role is represented explicitly, so its place in the community can be examined alongside specialized production. This perspective connects the potential usefulness of knowledge across domains with the researchers and activities through which that knowledge may circulate.

The exploratory cases illustrate this perspective in three ways. First, both roles participate in scientific production. Second, arbitrageurs connect the community: they take part in 16 of the 26 primary-run contact pairs, which join agents that topical similarity alone separates into seven components. Third, problem progress follows a shared prerequisite structure while its temporal paths difer. Together, these observations show why knowledge arbitrage is naturally studied through the joint evolution of people, artifacts, and problems.

## 7.2 From Contact to Cumulative Research

The simulations give contact, production, and replication distinct places in the community’s activity. Contact links researchers; publication adds a shared artifact; replication attempts its use. Contact is abundant and successful replication is absent, so the transitions between these stages determine how much knowledge circulates. For a process model, the important object is how opportunities enter later activity over time.

The dependency graph adds another dimension to this account. The two biology problems requiring a mathematics prerequisite illustrate a structural channel through which progress in one field supports possibilities in another. The diferent biology trajectories show how communities can arrive at the same set of solutions through diferent histories. This combination makes both timing and dependency structure relevant to the study of knowledge circulation.

These observations motivate a focused research agenda around the allocation of efort among production, discovery, adaptation, and use. The role profiles, costs, and candidate rules provide concrete places to vary that allocation. Questions include when a community benefits from dedicated intermediaries, how information access interacts with adaptation efort, and how the structure of prerequisites changes the timing of cross-domain opportunities. The present framework organizes these questions around a common computational process.

## 7.3 Model Scope and Extensions

SLA represents papers as structured records, problem solutions as simulator state changes, and replication outcomes as explicit stochastic rules. This explicitness lets each stage of knowledge circulation be varied and measured on its own.

Richer artifact representations would allow adaptation to operate on scientific content directly. Connecting papers, datasets, or executable methods to recipient tasks would extend the same architecture from structured records to content-bearing transfers. Empirical calibration of researcher profiles and activity costs would support comparisons with particular scientific communities. The framework’s modular separation of decisions, shared state, and action resolution provides a basis for such extensions.

## 8 Conclusion

We present Scientific Liquidity Agents, a framework that makes knowledge flow among LLM researcher agents observable and measurable. The archived runs yield three insights. First, crossdomain arbitrageurs act as bridges: they take part in 16 of the 26 contact pairs that join the primary-run community into one connected component; topical similarity alone splits the same community into seven. Second, the measured structure of a research community depends on what counts as a link: the same run yields three similarity edges, 26 contact pairs, and two coauthorship pairs. Third, the bottleneck lies in verification, not production: the runs register 68 and 274 papers, and replication is the step that no agent completes successfully.

IdeaScientist [7] shows that agents can draw mechanisms from analogous fields to generate research ideas. SLA models what happens after generation: how artifacts move between domains and whether other agents verify and reuse them. Knowledge liquidity thus becomes a quantity that agentic science can simulate, measure, and target, alongside knowledge production.

AI assistance. Generative AI coding and writing assistants supported framework development, manuscript preparation, code inspection, and ofline analysis. The authors are responsible for the manuscript and its interpretation.

Data and code availability. The accompanying version directory contains manuscript sources, frozen simulation records, selected source snapshots, and scripts for reproducing the reported descriptive analysis. Input provenance and SHA-256 hashes are recorded in arxiv/manifest.json; build and analysis commands are provided in README.txt.

## References

[1] Jiwoong Choi, Donghyun Kang, Yingrong Mao, and James Evans. Academic Simulacra: Forecasting research ideas through multi-agent LLM simulations, 2025. URL https://ci.acm. org/2025/wp-content/uploads/Academic\_Simulacra\_Final.pdf. ACM CI 2025 Poster and Demo Track.

[2] Santo Fortunato, Carl T Bergstrom, Katy Börner, James A Evans, Dirk Helbing, Staša Milojević, Alexander M Petersen, Filippo Radicchi, Roberta Sinatra, Brian Uzzi, Alessandro Vespignani, Ludo Waltman, Dashun Wang, and Albert-László Barabási. Science of science. Science, 359(6379):eaao0185, 2018. doi: 10.1126/science.aao0185.

[3] Nigel Gilbert. A simulation of the structure of academic science. Sociological Research Online, 2(2):1–15, 1997. doi: 10.5153/sro.85.

[4] Yiqiao Jin, Yiyang Wang, Lucheng Fu, Bing He, Siheng Xiong, Yijia Xiao, B. Aditya Prakash, Josiah Hester, Srijan Kumar, James Evans, and Jindong Wang. Science utopia? closed-loop LLM simulation of academic research ecosystems, 2026. URL https://arxiv.org/abs/2610. 01257.

[5] Nian Li, Chen Gao, Mingyu Li, Yong Li, and Qingmin Liao. EconAgent: Large language model-empowered agents for simulating macroeconomic activities. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 15523–15536. Association for Computational Linguistics, 2024. doi: 10.18653/v1/2024. acl-long.829. URL https://aclanthology.org/2024.acl-long.829/.

[6] Xiaomin Li, Yuexing Hao, Jianheng Hou, et al. MatrAIx: Simulating the world with 8.3 billion persona agents, 2026. URL https://arxiv.org/abs/2608.04205.

[7] Jiarui Liu, Renjie Tao, Yiwei Liao, Chuanyang Jin, Kai Sun, Xiao Yang, Xinyuan Zhang, Xilun Chen, Zhuangqun Huang, Lechen Zhang, Yongjin Yang, Yinghui He, Weihao Xuan, Rakesh Wanga, Anuj Kumar, Mona T. Diab, Wen tau Yih, and Xin Luna Dong. IdeaScientist: Orchestrating agents for grounded scientific ideation, 2026. URL https://arxiv.org/abs/ 2610.04074.

[8] Joon Sung Park, Joseph C O’Brien, Carrie J Cai, Meredith Ringel Morris, Percy Liang, and Michael S Bernstein. Generative Agents: Interactive simulacra of human behavior. arXiv preprint arXiv:2304.03442, 2023.

[9] Juste Raimbault. An agent-based model of interdisciplinary interactions in science, 2020. URL https://arxiv.org/abs/2006.16399.

[10] Anand S Rao and Michael P Georgef. BDI agents: From theory to practice. In Proceedings of the First International Conference on Multi-Agent Systems (ICMAS-95), pages 312–319, 1995.

[11] Kevin JS Zollman. The communication structure of epistemic communities. Philosophy of Science, 74(5):574–587, 2007. doi: 10.1086/525605.

## A Supplementary Views of Research Activity

Figure 6 shows output at the individual level. Blue bars denote specialists and orange bars arbitrageurs; each agent retains its archived numerical identifier. Fractional credit allocates joint papers equally across their recorded authors.

![](images/633f6a0e48109dbe9839da1219e3562fc6f13703533a49e2e5af6be8a4689d8b.jpg)

![](images/6548b728f5a3ffedfdd8a257b118787fe3c7a61215b30d931befc3a4004be62c.jpg)  
Figure 6: Individual fractional authorship credit. The primary arbitrageurs contribute 1 and 9 credits, while the extended arbitrageurs contribute 12.5 and 15. Credits across all agents sum to 68 and 274.

Figure 7 retains the logged token trajectories. Their last values are 1,494,803 and 5,992,945, compared with final-summary totals of 1,511,980 and 6,026,553. The diferences are 17,177 and 33,608 tokens. Curves use the logged series; Table 3 uses the final summaries.

![](images/96f849431b4756c3a9301ce2e42a5dbfb96805f141d2ee82b88536c6a4f59a9e.jpg)  
Figure 7: Logged cumulative token use. The plot retains the saved time-series values, displayed in millions. Community sizes and horizons are identified in the legend.

![](images/2ac9c57d93556078fd6a190b4763bf0792cc7c299489871693e353be017dbb88.jpg)  
Figure 9: Mean named-domain coverage in recorded activity labels under two counting conventions. Lines join summaries of the same agents after recoding; they are not temporal trajectories. Values describe labels rather than independent measurements of expertise or knowledge transfer.

Figure 8 describes the publication subsystem through its configured venue tiers. Primary papers fall in tiers 3–7 with counts 3, 30, 13, 21, and 1; extended counts are 16, 80, 64, 89, and 25. These are categories assigned within the simulator.

![](images/5b6fedaa584f1f4ddf7a48ec38e48d8f74eb17477be6c8663c801a048b0e00a4.jpg)

![](images/42496e2d37f362847ef93f2128b7e2810d8101dacac6aa992e37a9c0beb03cff.jpg)  
Figure 8: Unique registered papers by simulator venue tier. Tier 1 is the most selective configured venue; tier 8 is the preprint category. Counts describe the publication environment’s output distribution.

Figure 9 reports named-domain coverage in activity-topic keys. It includes both the original label convention and the convention excluding keys ending in \_collab. Specialist means are 1.0/1.0 in the primary case and 1.3125/1.0625 in the extended case. Arbitrageur means change from 3 to 0 and from 5 to 0. Generic cross\_domain and interdisciplinary labels remain; the plotted quantity counts named-domain mentions.

## B Simulation Rules and Additional Results

## B.1 Role Composition and Activity

Table 4 gives the specialist composition of each community. Each also includes two arbitrageurs assigned the generic cross\_domain label. Table 5 reports selected actions as a percentage of the corresponding role’s agent-actions.

Table 4: Specialist domain composition.
<table><tr><td>Run</td><td>Math.</td><td>Physics</td><td>Chemistry</td><td>Biology</td><td>CS</td></tr><tr><td>Primary</td><td>4</td><td>0</td><td>1</td><td>3</td><td>0</td></tr><tr><td>Extended</td><td>4</td><td>2</td><td>2</td><td>2</td><td>6</td></tr></table>

Table 5: Role-specific action distributions. S denotes specialists and A arbitrageurs.
<table><tr><td>Run / role</td><td>Agent-actions</td><td>Research</td><td>Publish</td><td>Collaborate</td><td>Replicate</td></tr><tr><td>Primary  $/ \mathrm { ~ S ~ }$ </td><td>800</td><td>58.75%</td><td>23.00%</td><td>18.25%</td><td>0.00%</td></tr><tr><td>Primary  $/ \mathrm { ~ A ~ }$ </td><td>200</td><td>70.00%</td><td>21.50%</td><td>8.50%</td><td>0.00%</td></tr><tr><td>Extended  $/ \mathrm { ~ S ~ }$ </td><td>3,200</td><td>64.25%</td><td>21.44%</td><td>13.97%</td><td>0.34%</td></tr><tr><td>Extended  $/ \mathrm { ~ A ~ }$ </td><td>400</td><td>70.25%</td><td>21.25%</td><td>8.50%</td><td>0.00%</td></tr></table>

The primary final similarity graph has three edges and seven components. Its persistent-contact graph contains 26 pairs, and its coauthorship graph contains two pairs. Extended-case edge counts are four, 61, and five, respectively. Figure 9 gives the named-domain label summaries.

![](images/2dab367e7c0226bbd8e4355aa6d93293f0877b3fb1a13b27510f2f10f80b93e7.jpg)  
Figure 10: Normalized pair coverage across both communities, complementing the primary-community network structure in Figure 4. Cross-role coverage uses $n _ { S } n _ { A }$ possible pairs; within-role coverage uses $\binom { n _ { t } } { 2 }$ Similarity, persistent contact, and coauthorship retain their respective definitions.

## B.2 Problem Candidates and Partner Selection

The selected source snapshot assigns arbitrageurs the domain cross\_domain. Research requests pass the agent’s domain to get\_solvable\_problems and, if needed, get\_frontier\_problems. Both functions filter candidates by exact domain equality. Saved problems have named domains, so the generic cross-domain label returns no candidates under these query rules. The problem results in the main text are consequently analyzed at the community level. All saved non-null solver identifiers belong to specialists.

The included deterministic query fixture contains five prerequisite-free roots, one per discipline, and one mathematics problem depending on the mathematics root. Each named-domain query returns its root; the generic label returns zero; a query with no domain restriction returns all five roots. Once the mathematics prerequisite is satisfied, its dependent problem becomes eligible. The 18 fixed query cases describe the snapshot’s candidate-selection behavior.

Collaboration candidates exclude existing collaborators. Cross-role candidates receive a score floor of 0.3, with an additional multiplier of 1.5 for arbitrageur requesters. Distinct domains receive a multiplier of 1.2. The ordinary collaboration branch selects the first ranked candidate, applies acceptance, and then records contact. The source also contains a separate bridging branch; neither saved action summary selects it. These rules explain how the simulator supplies opportunities for new cross-role contact.

## B.3 Action Resolution

Base action costs are 10 for research, 5 for publication, 8 for replication, 6 for collaboration, and 3 for tool adoption. Relevant skill, tool, or cooperation factors modify these synthetic units. Given efective cost $C _ { i } ( a )$ and an action-specific bonus $b _ { i } ( a )$ , the agent layer samples success with probability

$$
p _ { i } ( a ) = \operatorname* { m i n } \left( 0 . 9 5 , \frac { 1 } { 1 + 0 . 1 C _ { i } ( a ) } + b _ { i } ( a ) \right) .\tag{1}
$$

Local memory and inventory updates occur at this layer. The environment then performs action specific resolution. Publication has a venue decision, collaboration has partner acceptance, and replication has a downstream sampled outcome. The resulting records retain the meanings of their respective layers.

## B.4 Replication Rule

The replication branch draws candidates from the latest 20 paper identifiers. If same-domain candidates exist, it selects that subset with probability 0.7; otherwise it uses the full candidate set. A target is sampled uniformly from the selected set, including the agent’s own papers when present. Given paper clarity $c _ { p } .$ , programming skill $s _ { i } .$ , optional friction reduction $f _ { i } ,$ , and domains $d _ { i } , d _ { p }$ , the efective downstream success probability is

$$
q _ { i p } = \operatorname* { m a x } [ 0 , \operatorname* { m i n } ( 0 . 9 5 , c _ { p } s _ { i } + f _ { i } - 0 . 2 1 \{ d _ { i } \neq d _ { p } \} ) ] .\tag{2}
$$

The optional term is 0.3 times the agent’s friction-reduction method when available, and zero otherwise. A successful draw records a citation. Clarity and the outcome are internal model quantities. Local inventory updates can precede this downstream draw and can refer to a separately selected target.

The six extended-case records all have unsuccessful downstream outcomes. Under a reference calculation with zero friction reduction and unit additional cost multipliers, their success probabilities are approximately [0.0876, 0, 0.1411, 0.1530, 0.1818, 0.3206]. These values describe the six fixed targets under the reference rule. The saved observed outcomes remain the quantities reported in the main text.

## C Problem Structure and Production Accounting

## C.1 Static Prerequisite Closure

The two frozen dictionaries have identical problem identifiers, domains, dificulties, importance values, and prerequisite lists after removing solver, solve-step, and paper-association fields. Define D as a set of candidate domains, pre(v) as problem v’s prerequisites, and d(v) as its domain. Empty-start closure is the fixed point of

$$
C _ { 0 } = \varnothing , \qquad C _ { k + 1 } = C _ { k } \cup \{ v : d ( v ) \in D , \mathrm { \ p r e } ( v ) \subseteq C _ { k } \} .\tag{3}
$$

Across all five domains, the fixed point contains 360 problems. A 13-node cyclic strongly connected component and its descendants account for the remaining 150. Restriction to the primary roster’s domains yields 134; the extended roster covers all domains and yields 360. These sets coincide exactly with the saved solved sets. Figure 11 displays the resulting partition.

![](images/8168f3dff47f458e77f9400f9cc3078cbf5da6280885a6488728672af60abeca.jpg)  
Both solved sets equal their roster-restricted prerequisite closures (134 and 360).

Figure 11: Saved solved sets and empty-start prerequisite closure. The construction requires all prerequisites and uses no external seeds. Roster and cycle categories partition the problems outside each closure under this definition. Both biology solved sets contain 86 records.

The closure uses an all-prerequisites rule and no external prerequisite seeds. It describes graph reachability. The source’s candidate queries check prerequisites, while its solve mutator updates problem state without a separate eligibility check. We use closure as a static structural comparison with the saved endpoints.

Both biology solved sets contain 86 problems; biology-only closure contains 84. The additional problem biology\_neuroscience\_045 depends on mathematics\_graph\_theory\_086, and biology\_ genetics\_090 depends on that biology problem. The other 14 biology problems descend from the cyclic component. Solve steps difer for 84 of the 86 shared solutions, and paper-association lists difer for 57.

Table 6: Problem counts and saved solved sets across disciplines.
<table><tr><td>Domain</td><td>Total</td><td>Primary solved</td><td>Extended solved</td></tr><tr><td>Mathematics</td><td>100</td><td>28</td><td>89</td></tr><tr><td>Physics</td><td>105</td><td>0</td><td>66</td></tr><tr><td>Chemistry</td><td>100</td><td>20</td><td>31</td></tr><tr><td>Biology</td><td>100</td><td>86</td><td>86</td></tr><tr><td>Computer science</td><td>105</td><td>0</td><td>88</td></tr><tr><td>Total</td><td>510</td><td>134</td><td>360</td></tr></table>

## C.2 Publication Units

Production counts use unique registered identifiers. Fractional credit for agent i is

$$
P _ { i } = \sum _ { p \in \mathcal { P } } \frac { \mathbf { 1 } \{ i \in A _ { p } \} } { | A _ { p } | } , \sum _ { i } P _ { i } = | \mathcal { P } | ,\tag{4}
$$

where $A _ { p }$ is a paper’s recorded author set. Primary credits total 58 for specialists and 10 for arbitrageurs; extended credits total 246.5 and 27.5. Each joint paper has two authors. The 68 and 274 unique papers therefore correspond to 70 and 279 registered author–paper incidences.

The implementation also maintains personal publication lists, appending after agent-layer success and after registered authorship. Their 284 and 960 entries reconcile as 214 + 70 and 681 + 279. Table 7 gives the units at each recording layer. The LLM profile in the source uses personal-list length; the production analysis uses the unique publication registry.

Table 7: Selected actions, intermediate successes, and registered output. Each row retains its own counting unit.
<table><tr><td>Record</td><td>Counting unit</td><td>Primary</td><td>Extended</td></tr><tr><td>Publication</td><td></td><td></td><td></td></tr><tr><td>Internal publication list</td><td>List entries</td><td>284</td><td>960</td></tr><tr><td>Selected publication actions</td><td>Agent-actions</td><td>227</td><td>771</td></tr><tr><td>Agent-layer successes</td><td>Successful flags</td><td>214</td><td>681</td></tr><tr><td>Registered authorships</td><td>Author-paper pairs</td><td>70</td><td>279</td></tr><tr><td>Registered papers</td><td>Distinct paper IDs</td><td>68</td><td>274</td></tr><tr><td>Replication</td><td></td><td></td><td></td></tr><tr><td>Selected replication actions</td><td>Agent-actions</td><td>0</td><td>11</td></tr><tr><td>Agent-layer successes</td><td>Successful flags</td><td>0</td><td>6</td></tr><tr><td>Saved outcomes</td><td>Outcome records</td><td>0</td><td>6</td></tr><tr><td>Other-agent targets</td><td>Outcome records</td><td>0</td><td>5</td></tr><tr><td>Self-targets</td><td>Outcome records</td><td>0</td><td>1</td></tr><tr><td>Successful outcomes</td><td>Outcome records</td><td>0</td><td>0</td></tr><tr><td>Citations</td><td>Graph edges</td><td>0</td><td>0</td></tr></table>

The complete problem dictionary contains 510 records. A legacy domain-indexed summary covers 500 and reports 352 extended-case solutions. Including three additional solved physics problems and five additional solved computer-science problems gives the 360 reported here. The primary total is 134 under either convention.

## D Data Provenance and Reproduction

The simulation records are frozen at commit f1275a9 (January 15, 2026). The primary reports are dated January 12, 2026; extended documentation is dated December 2025. The primary historical protocol reports seed 42 and Gemini 2.0 Flash. Preserved materials include registries, snapshots, aggregate action summaries, final agent reports, and selected source files. Complete model requests, random draws, and runtime-captured revisions are not part of the archive. Source-rule descriptions in this paper refer to the supplied snapshot.

The manifest records input source paths, commits, and SHA-256 hashes. The ofline reconstruction counts unique papers, reconciles snapshots and fractional credits, normalizes network pairs, compares problem sets, and checks solve-step bounds. The query fixture tests the archived candidate functions on fixed inputs. These procedures operate on preserved records and do not invoke the simulator or an experimental model.

From the version directory, python arxiv/rebuild\_evidence.py --check checks the preserved numerical analysis, and python arxiv/eligibility\_regression.py --check checks the query fixture. The version directory includes the data and analysis scripts required for these operations. The minimal source ZIP contains the files needed to compile the manuscript. The revised figures are generated by arxiv/revise\_figures.py, with numeric inputs recorded in arxiv/ revision\_figure\_data.json. README.txt gives compilation and reproduction commands.