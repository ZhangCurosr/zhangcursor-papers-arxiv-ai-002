# MiniRep: Robust Reputation-Based Aggregation for Multi-Agent Debate

Jiaming Zhang<sup>1</sup>, Yuwan Liu<sup>1</sup>, Yue Huang<sup>1</sup>, and Sisi Duan<sup>1</sup> <sup>1</sup>Tsinghua University

## Abstract

Autonomous agents powered by large language models (LLMs) are rapidly evolving into an open agentic ecosystem. To support trustworthy collaboration, industry initiatives increasingly assess agent reputation from past behavior and provide performance leaderboards. However, reputation derived from past performance may not reliably predict an agent’s behavior on new tasks, particularly when malicious agents can adapt their behavior and influence other agents during collaboration.

We study reputation in multi-agent debate (MAD), where multiple agents answer the same query, debate to improve their answers, and aggregate them into a final output. We present MiniRep, a reputationbased aggregation system for MAD under malicious agents. To ground our threat model in established research, we construct an attack taxonomy drawing on reputation-system attacks and software-testing mutation operators, covering strategic exploitation of reputation and subtle corruption of agent proposals. Guided by this taxonomy, MiniRep evaluates agents based on both their behavior on the current task and their reputation over time, while preventing groups of agents with highly similar responses from dominating the final decision. We assess MiniRep across diverse tasks, LLM-agent compositions, corruption placements, and attack types drawn from our taxonomy. Our experimental results show that, MiniRep outperforms both conventional MAD aggregation and conventional reputation-based approaches on MATH no matter being attacked or not. Also, under a heterogeneous 10-agent setting on MATH, MiniRep outperforms all baselines in all 28 attack conditions.

## 1 Introduction

Autonomous agents (Wang et al., 2024; Chen et al., 2025) powered by large language models (LLMs) are evolving from standalone tools into an open agentic ecosystem. Gartner projects that the average Fortune 500 enterprise will operate more than 150,000 AI agents by 2028 (Lin, 2026). As agents are developed by different providers and increasingly interact across organizational boundaries, several industry initiatives have emerged to establish standards and infrastructure for agents to discover, connect, and interact with each other. Notable examples include the Agent2Agent (A2A) protocol (The Linux Foundation, 2026) and the ERC-8004 standard (De Rossi et al., 2025), which enable cross-provider agent interoperability and provide infrastructure for agent discovery and trust, respectively.

Along with the emergence of such open ecosystems, establishing trust among agents becomes increasingly important. Several emerging platforms therefore provide mechanisms to assess the reputation of LLM agents. For instance, ERC-8004 provides a Reputation Registry, a standard interface for posting and retrieving agent feedback. Several platforms and ongoing projects compute reputation scores or trust metrics from agent behavior and feedback (Credence Protocol, 2025; SAID Protocol, 2025; Chishti et al., 2026; Choe, 2025). Agent marketplaces such as AWS Marketplace (Amazon Web Services, 2026) also allow customers to rate agent products, resembling reputation mechanisms in conventional online marketplaces.

Reputation is not a new concept. Reputation systems have long been studied and deployed in peer-to-peer networks (Kamvar et al., 2003), e-commerce (Resnick and Zeckhauser, 2002), and software agents (Huynh et al., 2006; Klos and La Poutré, 2005; Mui et al., 2002). A common principle is to estimate an entity’s future reliability from its past behavior or feedback. For example, Beta Reputation (Jøsang and Ismail, 2002) models the probability of satisfactory behavior using a Beta distribution updated from positive and negative feedback. TrueSkill (Herbrich et al., 2006) maintains a probabilistic skill rating for each player in online games and updates the ratings based on past competition outcomes. Emerging LLM agent reputation systems mentioned above largely adopt this history-based principle by deriving reputation from historical task outcomes, interactions, or feedback.

Thus, an interesting question is: Can reputation derived from past performance reliably guide collective decisions when LLM agents may adapt their behavior or coordinate their responses? Indeed, unlike conventional reputation settings that often involve relatively well-defined interactions, LLM agents autonomously operate on diverse and open-ended tasks, making their reliability highly task-dependent. Moreover, their autonomy, memory, tool use, and interactions with other agents introduce new attack surfaces (Zhang et al., 2025; Debenedetti et al., 2024; Kavathekar et al., 2026). These differences become particularly important when reputation determines an agent’s influence on collective decisions.

In this paper, we focus on reputation at the aggregation stage of multi-agent debate (MAD), rather than for general-purpose LLM agents. In MAD, multiple agents independently respond to the same query, debate to improve their results, and aggregate them into a final output (Du et al., 2024; Li et al., 2024). Building on earlier work on AI debate towards safe AI (Irving et al., 2018), the problem has evolved into a general class of methods for improving LLM reasoning and factuality across diverse tasks (Du et al., 2024; Kaesberg et al., 2025; Khan et al., 2024; Choi et al., 2025). A unique feature in such a paradigm is that all the agents execute the same task, even when powered by different underlying LLMs. This common evaluation context makes agent behaviors directly comparable within a shared task context, providing a natural setting for studying and exploiting agent reputation.

We present MiniRep, a reputation-based aggregation system for MAD. We consider malicious agents that seek to manipulate MAD outputs away from the correct answers. Although some efforts have been made in identifying vulnerabilities of MAD, the area is relatively new, and there is no unified taxonomy of MAD vulnerabilities. We therefore construct our attack taxonomy from two complementary sources: established attacks on reputation systems (Hoffman et al., 2009), which capture strategic manipulation across agents and over time, and mutation operators from software testing (Petrovic and Ivankovi´ c, 2018; Just et al., 2014),´ which capture subtle proposal-level corruptions. An interesting aspect is that for each attack type, we identify related mechanisms or agent behaviors reported in prior work, grounding the taxonomy in existing evidence rather than purely synthetic constructions. We summarize our taxonomy and related analogs in Table 1. Based on such a taxonomy, we then design a new way of building reputation scores for the agents, taking into consideration how agents can launch the attacks. MiniRep carefully selects a group of agents, the proposals of which are aggregated. Particular emphasis is placed on limiting the influence of agent groups with the same underlying LLM. Also, agents are rated based on both their reputation over time and behavior on the current task, such that temporal deviation can be caught. Thus, current evidence can reduce an agent’s influence even after it has accumulated a strong reputation.

We assess MiniRep on three datasets, GoEmotions (Demszky et al., 2020), MATH (Hendrycks et al., 2021), and HumanEval Pro (Yu et al., 2025), under four LLM-agent compositions (i.e., using a combination of LLMs for the agents), four corruption placements (corrupting different agents), and seven attack types summarized in Table 1, yielding 112 attack conditions per dataset. We compare MiniRep with three MAD aggregation approaches (Uniform Majority (Kaesberg et al., 2025), Uniform Random-k, and Single-Metric), and four reputation systems (EigenTrust (Kamvar et al., 2003), Beta (Jøsang and Ismail, 2002), TrueSkill (Herbrich et al., 2006), and Babylon (Babylon contributors, 2026)). In these controlled aggregation runs, MiniRep has the highest observed average attacked score on three of the four task metrics, with mixed clean-task results. The most notable results are on MATH: MiniRep achieves 66.75% accuracy in clean runs, 7.50 percentage points higher than the best baseline, and 61.95% accuracy in attacked runs, compared with 54.37% for the best baseline. Across the 112 MATH attack conditions, MiniRep obtains a higher score than every baseline in 84 conditions. Also, under a heterogeneous 10-agent setting on MATH, MiniRep outperforms all baselines in all 28 attack conditions.

Table 1: Attack taxonomy studied in this work. Reputation classes follow Hoffman et al. (2009). The last column lists related mechanisms reported in prior work; these studies do not explicitly target agent reputation. <sup>⋆</sup> On-off and Adaptive adapt self-promoting attacks to MAD.
<table><tr><td>Attack</td><td>Instantiation</td><td>Reputation Related analogue class</td><td></td></tr><tr><td>Random coordi- nated</td><td>Malicious agents submit the same randomly selected incorrect answer.</td><td></td><td>Orchestrated Adversarial peers can induce conformity and degrade agent decisions (Ko et al., 2026); incorrect proposals can propagate during MAD (Cui et al., 2026).</td></tr><tr><td>Optimized coordi- nated</td><td>Malicious agents search for an incorrect proposal that maximizes its estimated impact on the final output.</td><td></td><td>Orchestrated Adversarial agents can generate candidate arguments and select the most persuasive one (Kraidia et al., 2026).</td></tr><tr><td>Diverse collusion</td><td>proposals that support the same incorrect conclusion.</td><td></td><td>Malicious agents submit distinct Orchestrated Colluding agents can provide distinct contributions toward a common objective and induce false conclusions (Hu et al., 2026).</td></tr><tr><td>On-off</td><td>Malicious agents first build reputation and then attack on every subsequent task.</td><td>Self- promoting*</td><td>Agents can first build peer trust and later inject fabricated information (Park et al., 2026).</td></tr><tr><td>Adaptive</td><td>Malicious agents first build reputation. During an attack, only a selected subset deviates from the system specification.</td><td>Self- promoting*</td><td>Manipulating one agent can be sufficient to mislead a multi-agent system (Liu et al., 2025).</td></tr><tr><td>Arithmetic operator</td><td>A malicious agent changes one arithmetic operation in an honestly generated proposal.</td><td>N/A</td><td>Arithmetic and relational operator replacement can introduce small but consequential faults (Petrović and Ivanković, 2018).</td></tr><tr><td>Boundary value</td><td>A malicious agent modifies a boundary value or literal in an honestly generated proposal.</td><td>N/A</td><td>Literal replacement can introduce faults into otherwise unchanged programs (Just et al., 2014).</td></tr></table>

## 2 Problem Background and Threat Model

Multi-agent debate (MAD). Typical MAD protocols organize each task into three phases: Propose, in which agents independently produce candidate answers; Debate, which, when performed, allows agents to exchange information and revise their responses over one or more rounds; and Decide, in which the resulting responses are aggregated into a final output (Du et al., 2024; Chan et al., 2024). We consider a sequence of tasks indexed by t. Let $\mathcal { P } = \{ 1 , \ldots , n \}$ denote the set of agents, and let $\tau _ { t } = ( c t x _ { t } , \chi _ { t } )$ denote task t, where ctx contains the public evidence, i.e., information provided to all agents, and χ contains task-specific parameters. In the Propose phase, each agent i produces a response $\boldsymbol { v } _ { i , t } = \left( c _ { i , t } , \rho _ { i , t } \right)$ , consisting of a payload $\boldsymbol { c } _ { i , t } \in \mathcal { V }$ and an optional explanation $\rho _ { i , t }$ , where $\rho _ { i , t } = \emptyset$ if no explanation is provided. After R debate rounds, the aggregation rule Agg first determines a pool $P _ { t } \subseteq \mathcal { P }$ of eligible agents and then maps their responses, the task context, and any persistent state $\sigma _ { t - 1 }$ to a final response. A stateless aggregation rule simply omits $\sigma _ { t - 1 }$

The Decide phase is often the most distinctive phase of MAD protocols. Common approaches include equal-weight voting (Wang et al., 2023; Kaesberg et al., 2025; Li et al., 2024); model-based aggregation, in which an evaluator such as an LLM judge assesses the candidate responses and produces or selects the final answer (Chan et al., 2024; Hu et al., 2025); confidence-weighted voting, which gives greater influence to responses associated with higher confidence (Zheng et al., 2026); and history-based aggregation, which weights agents according to their performance on previous tasks (Ebrahimi et al., 2025). In this work, we focus on reputation-based aggregation in the Decide phase. We compare our approach against three aggregation baselines. Uniform Majority (Kaesberg et al., 2025) aggregates the responses of all identities using equal weights. Uniform Random-k is a size-matched control that samples k identities independently of their past performance and aggregates their responses using equal weights. Single-Metric maintains a single historical quality score for each agent and uses these scores to rank and weight the agents. Thus, Single-Metric can also be viewed as a reputation-based approach based on past performance. We formally define these baselines in Appendix A.2.

In our system model, each agent retains a persistent identity across tasks, allowing its reputation to be updated over time. We refer to agents instantiated from the same underlying model as a clone group, and to the known partition of P into such groups as the clone-group mapping.

Assumptions and threat model. The adversary controls a fixed set $B \subseteq { \mathcal { P } }$ of at most f agents within each run. On task t, it may activate any subset $B _ { t } \subseteq B$ . Its goal is to manipulate the output of MAD, causing it to deviate from the output that would be produced if all agents behaved honestly. We assume that each malicious agent always participates in MAD and submits its candidate answer on time. A submitted proposal is observed consistently by the client and all agents. All agents also receive the same task context, including any public evidence. However, the candidate answer may deviate arbitrarily from the answer that the agent would produce if it behaved honestly. A malicious agent may behave honestly to accumulate reputation and begin attacking at any time chosen by A. For every task, the system uses four categories of information: the proposals $\{ v _ { i } \}$ , the public evidence in ctx , observable properties derived from the submitted proposals, and the clone-group mapping. The use of observable outcomes and behavior as reputation signals follows conventional reputation systems (Mui et al., 2002; Hoffman et al., 2009).

Our assumptions are practical, namely that all malicious agents submit responses and that the clone-group mapping is available to the system. First, the assumption that malicious agents always respond is reasonable within our threat model: skipping a proposal cannot directly inject adversarial content into the aggregation (Feldman et al., 2004; Hoffman et al., 2009). Second, assuming that the clone-group mapping is available is reasonable in a managed agent ecosystem, where an agent registry can expose registered identity, provider, or model information (De Rossi et al., 2025).

## 3 Design of MiniRep

Attack classes. As mentioned in the introduction, we construct our attack taxonomy based on established attacks on reputation systems (Hoffman et al., 2009) and mutation operators from software testing (Petrovic´ and Ivankovic, 2018; Just et al., 2014), as summarized in ´ Table 1. These two sources capture complementary aspects of the threat model. First, because reputation determines an agent’s influence on aggregation, adversaries may manipulate reputation to increase the impact of their proposals. Second, mutation operators provide systematic ways to introduce localized but consequential errors that may be difficult to detect in otherwise plausible proposals. Accordingly, we organize the attacks into three groups. First, random coordinated, optimized coordinated, and diverse collusion vary how malicious agents construct and coordinate their proposals within a task. Second, on-off and adaptive attacks manipulate reputation over time to increase adversarial influence. Finally, arithmetic-operator and boundary-value attacks apply simple mutation operators to corrupt otherwise honestly generated proposals.

![](images/e4640933825e8c38f39c04f2c61c57f0a0d017ac369576641adf45fa9ed1571c.jpg)  
Figure 1: Overview of MiniRep.

Overview. We show in Fig. 1 the workflow of MiniRep. Recall that we use a reputation-based aggregation for the Decide phase only. Thus, we can simply follow any existing approach for the Propose and Debate phases.

Suppose that we already have a reputation system that scores each agent’s capability. Let $w _ { i , t } ^ { h i s t }$ denote the reputation-derived weight of agent i, and let $\mathbf { w } _ { t } ^ { h i s t } = \left( w _ { 1 , t } ^ { h i s t } , \ldots , w _ { n , t } ^ { h i s t } \right)$ be the weights of all agents in P. A naive reputation-based approach directly uses these historical weights to aggregate the agents’ current proposals as

$$
\hat { y } _ { t } ^ { \mathrm { n a i v e } } = \mathsf { A g g } \Bigl ( \{ v _ { i , t } \} _ { i \in \mathcal { P } } , c t x _ { t } ; \mathbf { w } _ { t } ^ { h i s t } \Bigr ) .\tag{1}
$$

When conventional reputation systems rate agents based only on past performance, both self-promoting and orchestrated attacks can manipulate the aggregation output. For instance, a malicious agent with a powerful underlying LLM can stay honest to accumulate a high reputation, and then inject a wrong answer into a subsequent task to directly manipulate the output.

A running example. Consider a level-4 MATH problem asking for the center of $x ^ { 2 } - 6 x + y ^ { 2 } + 2 y =$ 9 (Hendrycks et al., 2021). Completing the square gives $( x - 3 ) ^ { 2 } + ( y + 1 ) ^ { 2 } = 1 9$ , so the answer is (3, −1). In a 10-agent debate, suppose that three high-reputation but malicious agents submit an incorrect answer (3, 1) while the other seven submit (3, −1). The naive weighted vote outputs (3, 1) if the three malicious agents carry a higher accumulated historical weight than that of the seven honest agents.

Our approach. In our work, we retain this basic idea in a Reputation-Aware Aggregator module, but additionally include a Response Analyzer that evaluates the quality of each current proposal and whether it shows signs of malicious behavior before aggregation. After the Reputation-Aware Aggregator produces the answer $\hat { y } _ { t } .$ , the State Updater uses the signals produced by the Response Analyzer to update each agent’s historical quality and behavioral-risk (i.e., previously detected suspicious behavior) records. These records determine the agents’ historical weights on subsequent tasks. The three modules have distinct roles: the Response Analyzer assesses current proposals, the Reputation-Aware Aggregator adjusts their weights and limits clone-group influence, and the State Updater updates the historical reputation.

More formally, the three modules work as follows. We give the design details in Appendix B.

Response Analyzer. The Response Analyzer takes as input the proposals $v _ { i , t _ { i \in \mathscr { P } } } .$ , the public evidence ctx <sub>t</sub>, and the clone-group mapping, and outputs quality, risk, validity, and blocking signals. Here, the public evidence refers to the information provided to all agents as part of the task, such as a question statement, a set of given facts, or a function specification.

The analyzer first evaluates each proposal. It assigns the proposal a quality score $q _ { i , t }$ based on its agreement with the other proposals and an independent assessment of its semantic content. We compute a publicsupport score from the current proposals without using the ground-truth answer or hidden tests. This score combines agreement among proposals with the overall answer distribution. Also, we group proposals by clone groups so that proposals by agents from the same clone group are not counted as independent sources of evidence. Finally, to check the semantic content, we use a semantic verifier instantiated by a small language model (a 1.5B model in our experiments). The goal is to estimate whether the proposal is correct based on the task and the content of the proposal.

The analyzer also estimates whether the proposal exhibits malicious or manipulative behavior via a behavior model implemented as a multilayer perceptron (MLP) (Rumelhart et al., 1986). The behavior model checks the current proposal for signs such as instruction override and, when debate revisions are available, unexpected changes across rounds.

A proposal is blocked on the current task if no candidate answer can be extracted, or if the shared detector flags it using combined correctness and behavior signals. We show the exact rule in Appendix B.1.3.

Reputation-Aware Aggregator. The Reputation-Aware Aggregator takes the quality, risk, validity, and blocking signals produced by the Response Analyzer, together with the historical weights $w _ { i , t } ^ { h i s t } { } _ { i \in \mathcal { P } }$ and the clone-group mapping, and outputs an aggregated result.

For each agent, the aggregator adjusts its historical weight using the proposal’s current quality relative to the other valid proposals, its current risk evidence, and the agent’s recorded behavioral risk. Blocked proposals, recently suspicious agents, and responses without an extractable answer receive additional weight penalties. Thus, a high historical weight alone does not guarantee high influence on the current task. The exact weighting and clone-group rules are deferred to Appendix B.2.1.

After that, only k out of n agents are selected into a pool $P _ { t } \subseteq \mathcal { P }$ . Before selection, the aggregator limits the total influence of each clone group so that multiple agents instantiated from the same underlying model are not treated as independent sources of reputation. It then uses the resulting weights to select the pool and determine the final aggregation weights $\{ w _ { i , t } \} _ { i \in P _ { t } }$ . Based on the pool and the weights generated by the aggregator, the selected proposals are combined using reputation-weighted aggregation:

$$
\hat { y } _ { t } = \mathsf { A g g } ( \{ v _ { i , t } \} _ { i \in P _ { t } } , c t x _ { t } ; \{ w _ { i , t } \} _ { i \in P _ { t } } ) .\tag{2}
$$

The task-dependent implementations of $\mathsf { A g g }$ are given under “Task-dependent aggregation rules” in Appendix A.2. Recall that the naive reputation-based approach is recovered by selecting all agents, $P _ { t } = \mathcal { P }$ and directly using their historical weights, $w _ { i , t } = w _ { i , t } ^ { h i s t }$

State Updater. The State Updater runs after the Reputation-Aware Aggregator produces $\hat { y } _ { t }$ . It uses the same quality, risk, validity, and blocking evidence produced by the Response Analyzer to update each agent’s historical reputation for subsequent tasks.

For each agent, the updater separately records its long-term quality, recent quality, and previously detected suspicious behavior. Briefly speaking, the long-term quality and recent quality are computed based on the quality scores $q _ { i , t }$ of the agent’s valid proposals across previous tasks. Consistently high-quality proposals gradually improve the agent’s reputation, whereas suspicious behavior reduces its influence on future tasks. Also, an unavailable or not well-formed response does not provide a new quality or risk observation.

The updated records form $\sigma _ { t }$ and determine the historical weight $w _ { i , t + 1 } ^ { h i s t }$ used on the next task. Note that $\sigma _ { t }$ is produced only after $\hat { y } _ { t }$ is fixed, so it cannot affect the answer to task t.

## 4 Experiments

## 4.1 Experimental Setup

Table 2: Experimental setup. Attack abbreviations follow the order in Table 1.
<table><tr><td>Name (#)</td><td>Settings</td></tr><tr><td>Dataset (3)</td><td>GoEmotions (Demszky et al., 2020), MATH (levels 4–5) (Hendrycks et al., 2021), HumanEval Pro (Yu et al., 2025)</td></tr><tr><td>Composition (4)</td><td>all strong, 7-strong+3-weak (7S+3W), all weak, random</td></tr><tr><td>Placement (4)</td><td>B (top-ranked), W (bottom-ranked), R1, R2 (random)</td></tr><tr><td>Attacks (7 + clean)</td><td>RAND (Hoffman et al., 2009; Ko et al., 2026), STRONG (Kraidia et al., 2026), DIV (Hu et al., 2026), ONOFF (Park et al., 2026), ADAPT (Hoffman et al., 2009; Liu et al., 2025), ARITH (Petrović and Ivanković, 2018), BoUND (Just et al., 2014)</td></tr></table>

Benchmarks. We use three datasets: GoEmotions (Demszky et al., 2020) for fine-grained multi-label emotion classification, MATH (Hendrycks et al., 2021) for competition-level mathematics (levels 4–5), and HumanEval Pro (Yu et al., 2025) for Python programming. For each dataset, we use the same fixed set of 100 tasks throughout. We provide examples from these datasets in Appendix C.

Agent pool and composition. We construct a pool of agents instantiated from seven API-accessible LLMs. Our goal is to evaluate reputation-based aggregation under meaningful differences in agent capability. We therefore select LLMs with a range of standalone scores on our datasets. This setting provides both correct and incorrect proposals for evaluating whether reputation-based aggregation assigns influence appropriately.

We rank the models from strong to weak using standalone evaluations on each dataset. The ranking is computed separately for each dataset because the relative performance of the models varies across tasks. In our experiments, we assess four compositions according to these rankings as summarized in Table 2. We provide the model rankings and the construction of each ten-agent pool in Appendix C.

Experimental setup and corruption placement. For all experiments, we fix n = 10, f = 3, and k = 7. We use four corruption placements: B and W compromise agents from higher- and lower-ranked models, respectively, while R1 and R2 are two random assignments.

Baselines. We assess seven baselines: three MAD aggregation baselines and four reputation systems. The MAD baselines are Uniform Majority (UMaj) (Kaesberg et al., 2025), Uniform Random-k (URand), and Single-Metric (Single). Uniform Random-k and Single-Metric are in-study controls for pool size and simple historical feedback, respectively. The reputation baselines are EigenTrust (Eigen) (Kamvar et al., 2003), Beta Reputation (Beta) (Jøsang and Ismail, 2002), TrueSkill (TS) (Herbrich et al., 2006), and Babylon (Babylon contributors, 2026). All stateful methods share post-task quality feedback; MiniRep also uses current-task signals before aggregation. We provide their definitions and adaptations in Appendix A.2.

All methods replay the same cached proposals with R = 0, isolating aggregation from within-task debate. Reputation still evolves across tasks, including the OnOff and Adapt schedules. We compare complete aggregation procedures, including current-task screening and the GoEmotions readout (Appendix A.2), rather than individual reputation components. Additional analyses of held-out attacks and agent compositions appear in Appendix D.1 and Appendix D.2.

Metrics. For GoEmotions (Demszky et al., 2020), we report exact-match accuracy and sample F1 (Sokolova and Lapalme, 2009). Exact-match accuracy requires the predicted and correct label sets to be identical.

Sample F1 gives partial credit for overlapping labels. We use mathematical-equivalence accuracy for MATH (Hendrycks et al., 2021) and Pass@1 (Chen et al., 2021) for HumanEval Pro (Yu et al., 2025).

For each experiment, we compare a clean run and an attacked run, where the attack is launched only in the attacked run. With four corruption placements and seven attacks, we assess 28 attack conditions for each agent composition and 112 conditions per dataset. We define all the metrics we use in Appendix C.3.

## 4.2 Evaluation Results

We report descriptive point estimates; strict wins denote higher observed scores, not statistical significance.   
Additional results are given in Appendix D.

• MiniRep can improve performance even in clean runs for some tasks. The clearest gain is on MATH, where MiniRep exceeds the best baseline by 7.50 percentage points in accuracy. This is consistent with selecting and weighting proposals using both historical performance and current quality, which allows MiniRep to downweight temporarily poor responses.

• The most consistent attacked-score advantage is on MATH. MiniRep achieves the highest average attacked score on three of the four task metrics and obtains 84 strict wins across the 112 MATH attack conditions.

• Even under self-promoting attacks, i.e., OnOff and Adapt in our evaluation, where the adversary first builds reputation and then attacks, MiniRep outperforms the best baseline in five of the eight evaluated conditions and ties it in two.

Overall performance of all experiments. Table 3 summarizes average scores over the clean runs and 112 attack conditions per dataset. In clean runs, the largest gain is on MATH: MiniRep achieves 66.75% accuracy, 7.50 percentage points above the best baseline. Clean GoEmotions F1 and HumanEval Pro Pass@1 are lower than the best baselines by 2.00 and 1.75 percentage points, respectively. Under attack, MiniRep has the highest observed GoEmotions accuracy (21.16%), MATH accuracy (61.95%), and HumanEval Pro Pass@1 (78.15%), with only a 0.23-point edge on HumanEval Pro.

In Table 4(a), we fix each agent composition and summarize the number of strict wins among the 28 attack conditions. Our results show that MiniRep achieves broad performance gains across different agent compositions, with the most consistent improvements on MATH. Specifically, it strictly outperforms all baselines in all 28 attack conditions under the 7S+3W composition and in 27 of the 28 all-strong conditions. The gains also generally apply to other tasks: MiniRep wins 25 of the 28 7S+3W conditions for GoEmotions exact-match accuracy, 17 of the 28 all-weak conditions for GoEmotions F1, and 12 of the 28 7S+3W conditions for HumanEval Pro. These results are consistent with the Reputation-Aware Aggregator using the Response Analyzer’s current-quality and risk signals to adjust historical weights.

Table 3: Average task scores (%). The best evaluated baseline is selected separately for results in the clean and attacked runs. Bold marks the better score; full eight-method results are in Table 6.
<table><tr><td></td><td colspan="2">Clean</td><td colspan="2">Attacked</td></tr><tr><td>Dataset / metric</td><td>Best baseline</td><td>MiniRep</td><td>Best baseline</td><td>MiniRep</td></tr><tr><td>GoEmotions / accuracy</td><td>21.50 (Beta)</td><td>21.75</td><td>19.59 (Beta)</td><td>21.16</td></tr><tr><td>GoEmotions / sample F1</td><td>36.42 (UMaj)</td><td>34.42</td><td>34.87 (Beta)</td><td>33.94</td></tr><tr><td>MATH / accuracy</td><td>59.25 (UMaj)</td><td>66.75</td><td>54.37 (Beta)</td><td>61.95</td></tr><tr><td>HumanEval Pro / Pass@ 1</td><td>79.50 (UMaj)</td><td>77.75</td><td>77.92 (Eigen)</td><td>78.15</td></tr></table>

Performance when capable agents are compromised. We select five attack conditions in which the compromised agents include highly ranked models, all using the 7S+3W composition. For placement, we

Table 4: Performance summary against the best evaluated baseline b. Panel (a) reports strict wins out of 28 attack conditions for each composition. Panels (b)–(c) report attacked task scores (%) where bold numbers denote the better value. The complete data are included in Tables 8–10, where repeated cells are highlighted.  
(a) Strict wins under different agent compositions.  
(b) Performance under the 7S+3W composition.
<table><tr><td>Setting</td><td>Best baseline b MiniRep</td><td></td><td>Setting</td><td>Best baseline b MiniRep</td><td></td></tr><tr><td>GoEmotions Acc./7S+3W</td><td>0/28 (all)</td><td>25/28</td><td>GoEmotions / B / Strong</td><td>35.83 (Beta)</td><td>33.00</td></tr><tr><td>GoEmotions F1 / All weak</td><td>8/28 (Single)</td><td>17/28</td><td>GoEmotions / B / Arith</td><td>34.00 (Beta)</td><td>34.90</td></tr><tr><td>MATH / All strong</td><td>0/28 (all)</td><td>27/28</td><td>GoEmotions / B / Bound</td><td>33.73 (Single)</td><td>35.17</td></tr><tr><td>MATH / 7S+3W</td><td>0/28 (all)</td><td>28/28</td><td>MATH / R2 / Div</td><td>55.00 (Babylon)</td><td>73.00</td></tr><tr><td>HumanEval Pro / 7S+3W</td><td>10/28 (Eigen)</td><td>12/28</td><td>MATH / R2 / Bound</td><td>54.00 (Babylon)</td><td>69.00</td></tr></table>

Strict task-score wins; ties are excluded.  
Attacked F1 for GoEmotions and accuracy for MATH (%).

(c) Performance under OnOff and Adapt attacks on HumanEval Pro under the 7S+3W composition.
<table><tr><td></td><td>B/OnOff</td><td>B/Adapt</td><td>R1/OnOff</td><td>R1/Adapt</td><td>R2/OnOff</td><td>R2/Adapt</td><td>W/OnOff</td><td>W/Adapt</td></tr><tr><td>Baseline b</td><td>Eigen</td><td>UMaj</td><td>Eigen</td><td>Eigen</td><td>Eigen</td><td>Eigen</td><td>Babylon</td><td>Babylon</td></tr><tr><td>Best b</td><td>78</td><td>81</td><td>81</td><td>82</td><td>79</td><td>80</td><td>80</td><td>81</td></tr><tr><td>MiniRep</td><td>81</td><td>81</td><td>82</td><td>79</td><td>82</td><td>83</td><td>81</td><td>81</td></tr></table>

Attacked Pass@1 (%).

In (a), b has the largest baseline win count; in (b)–(c), it has the highest score for that condition. B, R1, R2, and W denote the four corruption placements. Attack abbreviations are listed in Table 2.

use B for GoEmotions and R2 for MATH. We report their attacked task scores in Table 4(b). Compared with the best baseline, MiniRep achieves the highest score in four of the five conditions. In particular, it exceeds the strongest baseline by 18 and 15 percentage points under the MATH Div and Bound attacks, respectively. This result is consistent with our design: the Response Analyzer checks the current proposal, and the Reputation-Aware Aggregator adjusts its weight regardless of the agent’s past reputation. The only exception is GoEmotions/Strong, where MiniRep scores 33.00% F1 and Beta scores 35.83%. This difference largely reflects their clean performance: Beta achieves 38.27% F1 and MiniRep achieves 35.00%. However, under attack, MiniRep has a smaller performance loss than Beta.

Performance under OnOff and Adapt attacks. We evaluate the OnOff and Adapt attacks under all four corruption placements using the 7S+3W composition on HumanEval Pro, i.e., eight conditions in total. Table 4(c) reports Pass@1 for all eight conditions. Our results show that MiniRep outperforms the best baseline in five out of eight conditions and ties it in two of them. The gains are 1–3 percentage points over baselines that already achieve 78%–82% Pass@1. The wins span all four corruption placements. This pattern is consistent with checking current proposals before aggregation and carrying detected risk forward to later tasks.

## 5 Related Work

Attacks on agents and MAD. A growing body of work documents attacks on LLM agents: indirect prompt injection hijacks an agent through the data it reads (Greshake et al., 2023; Debenedetti et al., 2024), backdoors are implanted into its memory or knowledge bases (Yang et al., 2024; Chen et al., 2024b), and social-engineering exploits propagate through agent interaction (Shapira et al., 2026; Deng et al., 2025). Multi-agent debate inherits all of these channels and adds its own: agents can be driven to conform to a wrong majority (Amayuelas et al., 2024). Some defenses assess agents within a single task. Confidence-weighted consensus probes answer confidence from prompts and hidden states to tolerate Byzantine majorities in one instance (Zheng et al., 2026). Blockchain-based coordination couples bookkeeping with multi-metric answer evaluation for one coordination run (Chen et al., 2024a). Credibility scoring also learns agent weights from past query-answering contributions (Ebrahimi et al., 2025). MiniRep combines current-proposal quality and risk assessment with persistent reputation and clone-group control.

Classical reputation does not transfer. Reputation has been studied for over two decades: electronic marketplaces aggregate transaction feedback into seller scores (Resnick and Zeckhauser, 2002), peer-to-peer systems propagate trust transitively over interaction histories (Kamvar et al., 2003), probabilistic models ground reputation in statistics (Jøsang and Ismail, 2002; Herbrich et al., 2006), and the software-agents community systematized these notions into integrated trust and reputation models (Mui et al., 2002; Huynh et al., 2006). Attacks and defenses are equally well understood: taxonomies classify attacks by the system component they target (Hoffman et al., 2009; Koutrouli and Tsalgatidou, 2012), cheap identities enable Sybil attacks in the absence of a central authority (Douceur, 2002), and TrustGuard counters strategic oscillation with PID-style update dynamics (Srivatsa et al., 2005). These results, however, do not transfer to multi-agent debate, for two reasons. First, the feedback itself is missing: classical systems rate the observable outcome of a bilateral interaction, while in a debate the only visible outcome is the aggregated answer, so the per-agent contribution on which any reputation score must be built is never observed. Second, the attack surface differs: classical defenses assume that honest behavior eventually shows up as good outcomes, whereas debate attacks such as conformity pressure and fabricated consensus produce outcomes that look reasonable while being wrong, so the outcome signal alone cannot separate honest agents from adversarial ones. Recent reputation schemes for AI agents inherit the same outcome-based assumption (Ren et al., 2026; Lou et al., 2026; Chishti et al., 2026), and deployed registries record feedback that is rarely grounded in verifiable interactions (De Rossi et al., 2025; Xiong et al., 2026). MiniRep therefore borrows only two classical ingredients: a prior-based statistical update in the style of Jøsang and Ismail (2002) and fast reaction to recent negative behavior in the style of Srivatsa et al. (2005). It rebuilds the feedback layer itself, computing multi-signal behavioral evidence within each round and acting on it before aggregation.

## 6 Conclusions

We present MiniRep, a reputation-based aggregation approach for multi-agent debate (MAD). Our approach combines the quality of each agent’s proposal on the current task with its historical performance. Across the evaluated aggregation settings, MiniRep achieves the highest average MATH accuracy in both clean and attacked runs. Results on the other tasks are mixed, including lower clean GoEmotions F1 and HumanEval Pro Pass@1 than the best baselines.

## References

Alfonso Amayuelas, Xianjun Yang, Antonis Antoniades, Wenyue Hua, Liangming Pan, and William Yang Wang. MultiAgent collaboration attack: Investigating adversarial attacks in large language model collaborations via debate. In Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung Chen, editors, Findings ofthe Associationfor Computational Linguistics: EMNLP 2024, pages 6929–6948, Miami, Florida, USA, November 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.findings-emnlp.407. URL https://aclanthology.org/2024.find ings-emnlp.407/.

Amazon Web Services. Ai agents and tools in aws marketplace. https://aws.amazon.com/marketplace/s olutions/ai-agents-and-tools, 2026. Accessed: 2026-09-25.

Babylon contributors. Babylon: Reputation calculation service. Official software repository, file reputation-calculation-service.ts, 2026. URL https://github.com/BabylonSocial /babylon/blob/9b18096e3e69e68055ea4fc8cb0a1a94f80c2cef/packages/engine/src/r eputation/reputation-calculation-service.ts. Accessed 2026-09-19. The cited source defines the composite score; the task-feedback adapter is specified in this paper.

Chi-Min Chan, Weize Chen, Yusheng Su, Jianxuan Yu, Wei Xue, Shanghang Zhang, Jie Fu, and Zhiyuan Liu. Chateval: Towards better llm-based evaluators through multi-agent debate. In B. Kim, Y. Yue, S. Chaudhuri, K. Fragkiadaki, M. Khan, and Y. Sun, editors, International Conference on Learning Representations, volume 2024, pages 9079– 9093, 2024. URL https://proceedings.iclr.cc/paper\_files/paper/2024/hash/25cc3adf 8c85f7c70989cb8a97a691a7-Abstract-Conference.html.

Bei Chen, Gaolei Li, Xi Lin, Zheng Wang, and Jianhua Li. Blockagents: Towards byzantine-robust llm-based multiagent coordination via blockchain. In Proceedings of the ACM Turing Award Celebration Conference - China 2024, ACM-TURC ’24, page 187–192, New York, NY, USA, 2024a. Association for Computing Machinery. ISBN 9798400710117. doi: 10.1145/3674399.3674445. URL https://doi.org/10.1145/3674399.3674445.

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde De Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374, 2021. URL https://arxiv.org/abs/2107.03374.

Weize Chen, Ziming You, Ran Li, Yitong Guan, Chen Qian, Chenyang Zhao, Cheng Yang, Ruobing Xie, Zhiyuan Liu, and Maosong Sun. Internet of agents: Weaving a web of heterogeneous agents for collaborative intelligence. In Y. Yue, A. Garg, N. Peng, F. Sha, and R. Yu, editors, International Conference on Learning Representations, volume 2025, pages 36374–36411, 2025. URL https://proceedings.iclr.cc/paper\_files/paper/2025 /hash/59c27bf8d56d3d50c7aeaf7535dee975-Abstract-Conference.html.

Zhaorun Chen, Zhen Xiang, Chaowei Xiao, Dawn Song, and Bo Li. Agentpoison: Red-teaming llm agents via poisoning memory or knowledge bases. In A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang, editors, Advances in Neural Information Processing Systems, volume 37, pages 130185–130213. Curran Associates, Inc., 2024b. doi: 10.52202/079017-4136. URL https://proceedings.neurips.cc/paper\_files/p aper/2024/hash/eb113910e9c3f6242541c1652e30dfd6-Abstract-Conference.html.

Mohd Sameen Chishti, Damilare Peter Oyinloye, and Jingyue Li. Agentreputation: A decentralized agentic ai reputation framework. In Proceedings ofthe 34th ACM International Conference on the Foundations ofSoftware Engineering, pages 1317–1321, 2026. doi: 10.1145/3803437.3805579. URL https://doi.org/10.1145/3803437.38 05579.

Hanjoon Choe. agent-reputation-sdk: Ethereum SDK extensions for ERC-8004 trustless agents. https://github .com/hanjoonchoe/agent-reputation-sdk, 2025. Software repository. Accessed: 2026-09-25.

Hyeong Kyu Choi, Xiaojin Zhu, and Sharon Li. Debate or vote: Which yields better decisions in multi-agent large language models? In Advances in Neural Information Processing Systems, volume 38, 2025. doi: 10.52202/08571 3-3405. URL https://proceedings.neurips.cc/paper\_files/paper/2025/hash/934252a cd87f254d5d4672fbde283bd2-Abstract-Conference.html.

Credence Protocol. Credence: Real-time reputation scoring for AI agents. https://www.credenceprotocol .com/, 2025. Accessed: 2026-09-25.

Yu Cui, Hang Fu, Haibin Zhang, Licheng Wang, and Cong Zuo. Free-mad: Consensus-free multi-agent debate. In

Findings ofthe Associationfor Computational Linguistics: ACL 2026, pages 31977–31997, 2026. doi: 10.18653/v1/ 2026.findings-acl.1600. URL https://aclanthology.org/2026.findings-acl.1600/.

Marco De Rossi, Davide Crapis, Jordan Ellis, and Erik Reppel. Ethereum improvement proposals. erc-8004: Trustless agents. https://eips.ethereum.org/EIPS/eip-8004, 2025. Draft specification. Accessed: 2026-09- 25.

Edoardo Debenedetti, Jie Zhang, Mislav Balunovic, Luca Beurer-Kellner, Marc Fischer, and Florian Tramèr. Agentdojo: A dynamic environment to evaluate prompt injection attacks and defenses for llm agents. In A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang, editors, Advances in Neural Information Processing Systems, volume 37, pages 82895–82920. Curran Associates, Inc., 2024. doi: 10.52202/079017-2636. URL https://proceedings.neurips.cc/paper\_files/paper/2024/hash/97091a5177d8dc64b 1da8bf3e1f6fb54-Abstract-Datasets\_and\_Benchmarks\_Track.html.

Dorottya Demszky, Dana Movshovitz-Attias, Jeongwoo Ko, Alan Cowen, Gaurav Nemade, and Sujith Ravi. GoEmotions: A dataset of fine-grained emotions. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 4040–4054. Association for Computational Linguistics, 2020. doi: 10.18653/v1/2020.acl-main.372. URL https://aclanthology.org/2020.acl-main.372/.

Zehang Deng, Yongjian Guo, Changzhou Han, Wanlun Ma, Junwu Xiong, Sheng Wen, and Yang Xiang. Ai agents under threat: A survey of key security challenges and future pathways. ACM Comput. Surv., 57(7), February 2025. ISSN 0360-0300. doi: 10.1145/3716628. URL https://doi.org/10.1145/3716628.

Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer. QLoRA: Efficient finetuning of quantized LLMs. In Advances in Neural Information Processing Systems, volume 36, 2023. URL https://proceeding s.neurips.cc/paper\_files/paper/2023/hash/1feb87871436031bdc0f2beaa62a049b-A bstract-Conference.html.

John R. Douceur. The sybil attack. In Peter Druschel, Frans Kaashoek, and Antony Rowstron, editors, Peer-to-Peer Systems, pages 251–260, Berlin, Heidelberg, 2002. Springer Berlin Heidelberg. ISBN 978-3-540-45748-0. URL https://www.microsoft.com/en-us/research/publication/the-sybil-attack/.

Yilun Du, Shuang Li, Antonio Torralba, Joshua B. Tenenbaum, and Igor Mordatch. Improving factuality and reasoning in language models through multiagent debate. In Proceedings ofthe 41st International Conference on Machine Learning, 2024. URL https://proceedings.mlr.press/v235/du24e.html.

Sana Ebrahimi, Mohsen Dehghankar, and Abolfazl Asudeh. An adversary-resistant multi-agent LLM system via credibility scoring. In Kentaro Inui, Sakriani Sakti, Haofen Wang, Derek F. Wong, Pushpak Bhattacharyya, Biplab Banerjee, Asif Ekbal, Tanmoy Chakraborty, and Dhirendra Pratap Singh, editors, Proceedings ofthe 14th International Joint Conference on Natural Language Processing and the 4th Conference of the Asia-Pacific Chapter ofthe Associationfor Computational Linguistics, pages 1676–1693, Mumbai, India, December 2025. The Asian Federation of Natural Language Processing and The Association for Computational Linguistics. ISBN 979-8-89176- 298-5. doi: 10.18653/v1/2025.ijcnlp-long.90. URL https://aclanthology.org/2025.ijcnlp-long. 90/.

Michal Feldman, Kevin Lai, Ion Stoica, and John Chuang. Robust incentive techniques for peer-to-peer networks. In Proceedings of the 5th ACM Conference on Electronic Commerce, EC ’04, page 102–111, New York, NY, USA, 2004. Association for Computing Machinery. ISBN 1581137710. doi: 10.1145/988772.988788. URL https://doi.org/10.1145/988772.988788.

Kai Greshake, Sahar Abdelnabi, Shailesh Mishra, Christoph Endres, Thorsten Holz, and Mario Fritz. Not what you’ve signed up for: Compromising real-world llm-integrated applications with indirect prompt injection. In Proceedings ofthe 16th ACM Workshop on Artificial Intelligence and Security, AISec ’23, page 79–90, New York, NY, USA, 2023. Association for Computing Machinery. ISBN 9798400702600. doi: 10.1145/3605764.3623985. URL https://doi.org/10.1145/3605764.3623985.

Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q. Weinberger. On calibration of modern neural networks. In Proceedings ofthe 34th International Conference on Machine Learning, volume 70 of Proceedings ofMachine Learning Research, pages 1321–1330. PMLR, 2017. URL https://proceedings.mlr.press/v70/guo17a.html.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the MATH dataset. In Proceedings of the Neural

Information Processing Systems Track on Datasets and Benchmarks, volume 1, 2021. URL https://datase ts-benchmarks-proceedings.neurips.cc/paper/2021/hash/be83ab3ecd0db773eb2dc1 b0a17836a1-Abstract-round2.html.

Ralf Herbrich, Tom Minka, and Thore Graepel. TrueSkill: A Bayesian skill rating system. In Advances in Neural Information Processing Systems, volume 19, pages 569–576, 2006. URL https://proceedings.neurips. cc/paper/2006/hash/f44ee263952e65b3610b8ba51229d1f9-Abstract.html.

Kevin Hoffman, David Zage, and Cristina Nita-Rotaru. A survey of attack and defense techniques for reputation systems. ACM Computing Surveys (CSUR), 42(1):1–31, 2009. doi: 10.1145/1592451.1592452. URL https: //doi.org/10.1145/1592451.1592452.

Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum?id=nZeVKeeFYf9.

Jinwei Hu, Xinmiao Huang, Youcheng Sun, Yi Dong, and Xiaowei Huang. Lying with truths: Open-channel multiagent collusion for belief manipulation via generative montage. In Proceedings ofthe 64th Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 5979–5996, 2026. doi: 10.18653/v1/20 26.acl-long.270. URL https://aclanthology.org/2026.acl-long.270/.

Tianyu Hu, Zhen Tan, Song Wang, Huaizhi Qu, and Tianlong Chen. Multi-agent debate for llm judges with adaptive stability detection. In D. Belgrave, C. Zhang, H. Lin, R. Pascanu, P. Koniusz, M. Ghassemi, and N. Chen, editors, Advances in Neural Information Processing Systems, volume 38, pages 46504–46540. Curran Associates, Inc., 2025. doi: 10.52202/085713-1548. URL https://proceedings.neurips.cc/paper\_files/paper/202 5/hash/42475c537936b2394b5015e871765056-Abstract-Conference.html.

Trung Dong Huynh, Nicholas R Jennings, and Nigel R Shadbolt. An integrated trust and reputation model for open multi agent systems. Autonomous agents and multi-agent systems, 13(2):119–154, 2006. doi: 10.1007/s10458-005-6825-4. URL https://link.springer.com/article/10.1007/s10458-005-6825-4.

Geoffrey Irving, Paul Christiano, and Dario Amodei. Ai safety via debate. arXiv preprint arXiv:1805.00899, 2018. URL https://arxiv.org/abs/1805.00899.

Audun Jøsang and Roslan Ismail. The beta reputation system. In Proceedings ofthe 15th Bled Electronic Commerce Conference, 2002. URL https://aisel.aisnet.org/bled2002/41/.

René Just, Darioush Jalali, Laura Inozemtseva, Michael D Ernst, Reid Holmes, and Gordon Fraser. Are mutants a valid substitute for real faults in software testing? In Proceedings of the 22nd ACM SIGSOFT international symposium onfoundations ofsoftware engineering, pages 654–665, 2014. doi: 10.1145/2635868.2635929. URL https://doi.org/10.1145/2635868.2635929.

Lars Benedikt Kaesberg, Jonas Becker, Jan Philip Wahle, Terry Ruas, and Bela Gipp. Voting or consensus? decisionmaking in multi-agent debate. In Findings of the Association for Computational Linguistics: ACL 2025, pages 11640–11671, 2025. doi: 10.18653/v1/2025.findings-acl.606. URL https://aclanthology.org/2025. findings-acl.606/.

Sepandar D. Kamvar, Mario T. Schlosser, and Hector Garcia-Molina. The EigenTrust algorithm for reputation management in P2P networks. In Proceedings of the 12th International Conference on World Wide Web, pages 640–651. ACM, 2003. doi: 10.1145/775152.775242. URL https://dl.acm.org/doi/10.1145/775152. 775242.

Archit Karandikar, Nicholas Cain, Dustin Tran, Balaji Lakshminarayanan, Jonathon Shlens, Michael Mozer, and Becca Roelofs. Soft calibration objectives for neural networks. In Advances in Neural Information Processing Systems, volume 34, 2021. URL https://proceedings.neurips.cc/paper\_files/paper/2021/hash/f 8905bd3df64ace64a68e154ba72f24c-Abstract.html.

Ishan Kavathekar, Hemang Jain, Ameya Rathod, Ponnurangam Kumaraguru, and Tanuja Ganu. TAMAS: Benchmarking adversarial risks in multi-agent LLM systems. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 31238–31268, 2026. doi: 10.18653/v1/2026.acl-long.14 42. URL https://aclanthology.org/2026.acl-long.1442/.

Akbir Khan, John Hughes, Dan Valentine, Laura Ruis, Kshitij Sachan, Ansh Radhakrishnan, Edward Grefenstette,

Samuel R. Bowman, Tim Rocktäschel, and Ethan Perez. Debating with more persuasive LLMs leads to more truthful answers. In Proceedings ofthe 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 23662–23733. PMLR, 2024. URL https://proceedings.mlr.press/ v235/khan24a.html.

Tomas B Klos and Han La Poutré. Decentralized reputation-based trust for assessing agent reliability under aggregate feedback. In Trusting Agentsfor Trusting Electronic Societies, volume 3577 of Lecture Notes in Computer Science, pages 110–128. Springer, 2005. doi: 10.1007/11532095\_7. URL https://link.springer.com/chapte r/10.1007/11532095\_7.

Changgeon Ko, Jisu Shin, Hoyun Song, Huije Lee, Eui Jun Hwang, and Jong C Park. Social dynamics as critical vulnerabilities that undermine objective decision-making in llm collectives. In Proceedings of the 64th Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 37865–37890, 2026. doi: 10.18653/v1/2026.acl-long.1756. URL https://aclanthology.org/2026.acl-long.1756/.

Eleni Koutrouli and Aphrodite Tsalgatidou. Taxonomy of attacks and defense mechanisms in p2p reputation systems—lessons for reputation system designers. Computer Science Review, 6(2–3):47–70, 2012. ISSN 1574-0137. doi: 10.1016/j.cosrev.2012.01.002. URL https://www.sciencedirect.com/science/article/pi i/S1574013712000093.

Insaf Kraidia, Iyas Qaddara, Alhanof Almutairi, Nada Alzaben, and Samir Brahim Belhouari. When collaboration fails: persuasion driven adversarial influence in multi agent large language model debate. Scientific Reports, 16(1):11640, 2026. doi: 10.1038/s41598-026-42705-7. URL https://www.nature.com/articles/s41598-026-4 2705-7.

Heungsub Lee. TrueSkill: Python implementation documentation. Software documentation, version 0.4.5, 2026. URL https://trueskill.org/. Accessed September 25, 2026.

Junyou Li, Qin Zhang, Yangbin Yu, Qiang Fu, and Deheng Ye. More agents is all you need. Transactions on Machine Learning Research, 2024. ISSN 2835-8856. URL https://openreview.net/forum?id=bgzUSZ8aeg.

Belle Lin. Companies have a new ai problem: Too many agents. The Wall Street Journal, May 2026. URL https://www.wsj.com/cio-journal/companies-have-a-new-ai-problem-too-many-a gents-9539c4d6. Accessed: September 17, 2026.

Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, and Piotr Dollár. Focal loss for dense object detection. In 2017 IEEE International Conference on Computer Vision (ICCV), pages 2999–3007, 2017. doi: 10.1109/ICCV.2017.324. URL https://doi.org/10.1109/ICCV.2017.324.

Fengyuan Liu, Rui Zhao, Shuo Chen, Guohao Li, Philip Torr, Lei Han, and Jindong Gu. Can an individual manipulate the collective decisions of multi-agents? In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 12158–12182, 2025. doi: 10.18653/v1/2025.emnlp-main.611. URL https: //aclanthology.org/2025.emnlp-main.611/.

Yuwei Lou, Hao Hu, Shaocong Ma, Zongfei Zhang, Liang Wang, Jidong Ge, and Xianping Tao. DRF: LLM-AGENT dynamic reputation filtering framework. In Neural Information Processing, volume 16312 of Lecture Notes in Computer Science, pages 127–141. Springer, 2026. doi: 10.1007/978-981-95-4384-7\_10. URL https://link.springer.com/chapter/10.1007/978-981-95-4384-7\_10.

Lik Mui, Mojdeh Mohtashemi, and Ari Halberstadt. Notions of reputation in multi-agents systems: a review. In Proceedings of the first international joint conference on Autonomous agents and multiagent systems: part 1, pages 280–287, 2002. doi: 10.1145/544741.544807. URL https://doi.org/10.1145/544741.544807.

Jeongsu Park, Yuji Lim, Geonwoo Kim, Taehyeon Yun, and Moohong Min. Data poisoning in llm multiagent societies: Social proof drives collective decision failure in financial deliberation. ETRI Journal, pages 1–13, 2026. doi: 10.42 8/etrij.2026-0186. URL https://onlinelibrary.wiley.com/doi/10.4218/etrij.2026-0186.

Goran Petrovic and Marko Ivankovi´ c. State of mutation testing at google. In´ Proceedings of the 40th international conference on software engineering: Software engineering in practice, pages 163–171, 2018. URL https: //research.google/pubs/state-of-mutation-testing-at-google/.

Qwen, An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou,

Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tianyi Tang, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. Qwen2.5 technical report. arXiv preprint arXiv:2412.15115, 2025. doi: 10.48550/arXiv.2412.15115. URL https://arxiv.org/abs/2412.15115. Version 2.

Siyue Ren, Wanli Fu, Xinkun Zou, Chen Shen, Yi Cai, Chen Chu, Zhen Wang, and Shuyue Hu. Reputation as a solution to cooperation collapse in LLM-based MASs. In Proceedings ofthe 25th International Conference on Autonomous Agents and Multiagent Systems, pages 245–253. International Foundation for Autonomous Agents and Multiagent Systems, 2026. doi: 10.65109/UEHN4980. URL https://doi.org/10.65109/UEHN4980.

Paul Resnick and Richard Zeckhauser. Trust among strangers in internet transactions: Empirical analysis of eBay’s reputation system. In Michael R. Baye, editor, The Economics of the Internet and E-commerce, pages 127–157. Emerald Group Publishing Limited, 10 2002. ISBN 978-0-76230-971-9. doi: 10.1016/S0278-0984(02)11030-3. URL https://doi.org/10.1016/S0278-0984(02)11030-3.

David E Rumelhart, Geoffrey E Hinton, and Ronald J Williams. Learning representations by back-propagating errors. Nature, 323(6088):533–536, 1986. doi: 10.1038/323533a0. URL https://www.nature.com/articles/ 323533a0.

SAID Protocol. SAID: The identity and reputation layer for AI agents. https://www.saidprotocol.com/, 2025. Accessed: 2026-09-25.

Natalie Shapira, Chris Wendler, Avery Yen, Gabriele Sarti, Koyena Pal, Olivia Floody, Adam Belfki, Alex Loftus, Aditya Ratan Jannali, Nikhil Prakash, et al. Agents of chaos. arXiv preprint arXiv:2602.20021, 2026. URL https://arxiv.org/abs/2602.20021.

Marina Sokolova and Guy Lapalme. A systematic analysis of performance measures for classification tasks. Information Processing & Management, 45(4):427–437, 2009. doi: 10.1016/j.ipm.2009.03.002. URL https://www.scie ncedirect.com/science/article/pii/S0306457309000259.

Mudhakar Srivatsa, Li Xiong, and Ling Liu. Trustguard: countering vulnerabilities in reputation management for decentralized overlay networks. In Proceedings ofthe 14th International Conference on World Wide Web, WWW ’05, page 422–431, New York, NY, USA, 2005. Association for Computing Machinery. ISBN 1595930469. doi: 10.1145/1060745.1060808. URL https://doi.org/10.1145/1060745.1060808.

The Linux Foundation. Agent2agent (a2a) protocol. https://a2a-protocol.org/latest/, 2026. Accessed: 2026-09-25.

Lei Wang, Chen Ma, Xueyang Feng, Zeyu Zhang, Hao Yang, Jingsen Zhang, Zhiyuan Chen, Jiakai Tang, Xu Chen, Yankai Lin, et al. A survey on large language model based autonomous agents. Frontiers of computer science, 18(6): 186345, 2024. doi: 10.1007/s11704-024-40231-1. URL https://link.springer.com/article/10.1 007/s11704-024-40231-1.

Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V Le, Ed H. Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=1PL1NIMM rw.

Xihan Xiong, Zelin Li, Wei Wei, Qin Wang, William Knottenbelt, and Zhipeng Wang. Can trustless agents be trusted? an empirical study of the ERC-8004 decentralized AI agent ecosystem. arXiv preprint arXiv:2606.26028, 2026. URL https://arxiv.org/abs/2606.26028.

Wenkai Yang, Xiaohan Bi, Yankai Lin, Sishuo Chen, Jie Zhou, and Xu Sun. Watch out for your agents! investigating backdoor threats to llm-based agents. In A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang, editors, Advances in Neural Information Processing Systems, volume 37, pages 100938–100964. Curran Associates, Inc., 2024. doi: 10.52202/079017-3201. URL https://proceedings.neurips.cc/paper\_f iles/paper/2024/hash/b6e9d6f4f3428cd5f3f9e9bbae2cab10-Abstract-Conference.ht ml.

Zhaojian Yu, Yilun Zhao, Arman Cohan, and Xiao-Ping Zhang. HumanEval Pro and MBPP Pro: Evaluating large language models on self-invoking code generation task. In Findings ofthe Associationfor Computational Linguistics:

ACL 2025, pages 13253–13279. Association for Computational Linguistics, 2025. doi: 10.18653/v1/2025.findings-a cl.686. URL https://aclanthology.org/2025.findings-acl.686/.

Hanrong Zhang, Jingyuan Huang, Kai Mei, Yifei Yao, Zhenting Wang, Chenlu Zhan, Hongwei Wang, and Yongfeng Zhang. Agent security bench (asb): Formalizing and benchmarking attacks and defenses in llm-based agents. In International Conference on Learning Representations, volume 2025, pages 35331–35366, 2025. URL https: //openreview.net/forum?id=V4y0CpX4hK.

Lifan Zheng, Jiawei Chen, Qinghong Yin, Jingyuan Zhang, Xinyi Zeng, and Yu Tian. Rethinking the reliability of multi-agent system: A perspective from Byzantine fault tolerance. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pages 35012–35020, 2026. doi: 10.1609/aaai.v40i41.40806. URL https://ojs.aaai.org/index.php/AAAI/article/view/40806.

## A Notations and Baselines

## A.1 Notations

Table 5 collects the symbols used in Sec. 2, Sec. 3, and the detailed definitions below. Identities are indexed by i and tasks by t. Per-task quantities carry a task subscript only where the task matters. Task t reads $\sigma _ { t - 1 }$ and writes $\sigma _ { t }$ after the answer; method-specific quantities use local superscripts where needed.

Table 5: Core notation.
<table><tr><td>Symbol</td><td>Meaning</td></tr><tr><td colspan="2">Protocol related</td></tr><tr><td> $\mathcal { P } ; n ; f$ </td><td>Identity set; number of identities; maximum adversarial identities.</td></tr><tr><td> ${ \boldsymbol { \tau } } = \left( c t x , \chi \right)$ </td><td>Task: context ctx (statement and public evidence) and parameter</td></tr><tr><td> $v _ { i } = \left( c _ { i } , \rho _ { i } \right)$ </td><td>Response of identity i: payload ci and rationale  $\rho _ { i } .$ </td></tr><tr><td> $R$ </td><td>Debate rounds  $( R = 0$  in the main setting).</td></tr><tr><td> $G ;$  clone-group mapping</td><td>Known group of identities; the experiments group replicas by their underlying API model.</td></tr><tr><td> $A ; B _ { t }$ </td><td>Adversary operator; coalition active on task  $t , | B _ { t } | \leq f .$ </td></tr><tr><td> $P _ { t } ; k$ </td><td>Identities selected on a task; target pool size.</td></tr><tr><td> $\mathsf { A g g } ; \sigma$ </td><td>Aggregation rule (the Decide step); persistent reputation state.</td></tr><tr><td colspan="2">Same-task signals</td></tr><tr><td> $p _ { i , t } ^ { p u b }$ </td><td>Public support (Eq. 32); distinct from the task context ctx.</td></tr><tr><td> $p _ { i , t } ^ { s e m }$ </td><td>Calibrated correctness probability from the 1.5B semantic verifier.</td></tr><tr><td> $r _ { i , t } ^ { a t k }$ </td><td>Behavior-model risk score (Appendix B.1.2).</td></tr><tr><td> $\eta _ { i , t } ; f _ { i , t }$ </td><td>Parse-validity indicator; shared detector flag. The response remains  $v _ { i , t }$ </td></tr><tr><td> $a _ { i , t } ; r _ { i , t } ^ { g a t e }$ </td><td>Fused evidence score; derived decision penalty, not a second learned probability (Eq. 41).</td></tr><tr><td> $\widehat { a } _ { i , t }$ </td><td>Alarm strength used in persistent updates: max  $\{ a _ { i , t } , 0 . 9 0 d _ { i , t } \}$ </td></tr><tr><td> $q _ { t } ^ { m e d } ; s _ { i , t } ^ { s e l f }$ </td><td>Median parse-valid quality; same-model self-consistency.</td></tr><tr><td> $c _ { i , t } ^ { f m t }$ </td><td></td></tr><tr><td></td><td>Format compliance in [0, 1]; parse failure forces  $q _ { i , t } = 0 .$ </td></tr><tr><td> $s _ { i , t } ; q _ { i , t }$ </td><td>Fused support; same-task quality (Appendix B.1.3).</td></tr><tr><td> $d _ { i , t }$  blocked set</td><td>Current-task block: parse failure or the shared detector flag  $f _ { i , t } \left( \mathrm { E q } . 3 9 \right)$  Current-task blocked identities or identities with active temporary restrictions. They rank behind all</td></tr><tr><td></td><td>unblocked candidates.</td></tr><tr><td colspan="2">Weights and selection</td></tr><tr><td> $w _ { i , t } ^ { h i s t }$ </td><td>Incoming weight from  $\sigma _ { t - 1 } .$  with numerical floor  $( \operatorname { E q . } 5 1 ) .$ </td></tr><tr><td> $w _ { i , t } ; Q _ { i , t }$ </td><td>Final pool weight; historical-risk/cooldown mark used in current weighting</td></tr><tr><td> $\widetilde { w } _ { i , t } ; m _ { i , t }$ </td><td>Pre-group current-round weight (Eq. 42); product of blocking, historical-risk and invalidity multipliers.</td></tr><tr><td> $M _ { G } ; p _ { G }$ </td><td>Pre-group weight sum; largest member weight in  $G ,$  with  $m = | G | .$ </td></tr><tr><td colspan="2">Persistent state</td></tr><tr><td> $\bar { q } _ { i } ; q _ { i } ^ { s h o r t }$ </td><td>Long-term quality (Beta-style prior, strength  $^ { 1 2 ) ; }$  short-term quality (asymmetric EMA 0.30/0.08).</td></tr><tr><td> $\bar { e } _ { i } ; e _ { i } ^ { s h o r t }$ </td><td>Long- and short-term correctness support.</td></tr><tr><td> $D _ { i } ; K _ { i } ; E _ { i } ; I _ { i }$ </td><td>Failure debt; strike counter; detector  $\mathrm { E M A } ;$  identity risk (Appendix B.3.1).</td></tr><tr><td> $r _ { i } ^ { t e m p }$ </td><td>Temporal risk: maximum of  $D _ { i } .$  , level risk,  $E _ { i } , I _ { i } ,$  and instant failure risk.</td></tr><tr><td> $h _ { i }$ </td><td>Cooldown counter (three tasks after a valid-response detection)</td></tr><tr><td colspan="2">Baseline and evaluation quantities</td></tr><tr><td> $y _ { i , t } ^ { c a n d } ; F _ { t } ; \mathcal { F } _ { t }$ </td><td>Parsed candidate answer; feedback-available identities; detector-selected candidates.</td></tr><tr><td> $z _ { i , t } ; u _ { i , t } ; x _ { i , t }$ </td><td>Estimated feedback quality; validity-masked score; format-adjusted score.</td></tr><tr><td> $\sigma _ { i , t } ^ { \mathrm { S M } } ; h _ { i } ^ { \mathrm { T S } }$ </td><td></td></tr><tr><td> $S _ { i , t } ^ { \mathrm { { \tiny { B a b } } } } ; B _ { i , t }$ </td><td>Single-Metric quality EMA; TrueSkill conservative rating.</td></tr><tr><td> $A _ { t } \colon J _ { t } ; E _ { t } \colon H _ { t }$ </td><td>Babylon score; earlier-performance baseline for gap tracking (Appendix B.3.2). Active-attack indicator; payload-bearing identities; exposure and payload-match indicators.</td></tr><tr><td> $s _ { m , t } ^ { 0 } ; s _ { m , t } ^ { a }$ </td><td>Clean and attacked per-task utility for method  $m .$ </td></tr></table>

## A.2 Baselines

We distinguish source-method defaults from settings introduced by our MAD adaptations; feedback mappings, clipping bounds, and numerical tolerances are implementation choices unless stated otherwise.

Multi-agent debate, formally. We formalize the MAD phases introduced in Sec. 2, using the notation in Table 5. Consider a sequence of tasks $\tau _ { t } = ( c t x _ { t } , \chi _ { t } )$ indexed by $t = 1 , 2 , \ldots$ In the Propose phase, each identity $i \in \mathcal { P }$ independently generates

$$
v _ { i , t } ^ { ( 0 ) } = \mathrm { P r o p o s e } _ { i } ( \tau _ { t } ) .\tag{3}
$$

If the protocol includes debate, then at each round $\ell = 1 , \ldots , R ,$ identity i updates its proposal according to

$$
v _ { i , t } ^ { ( \ell ) } = \mathrm { D e b a t e } _ { i } \left( \tau _ { t } , v _ { i , t } ^ { ( \ell - 1 ) } , \mathcal { H } _ { i , t } ^ { ( \ell - 1 ) } \right) ,\tag{4}
$$

where $\mathcal { H } _ { i , t } ^ { ( \ell - 1 ) }$ contains the proposals and messages visible to identity i before round ℓ. The MAD communication topology determines the contents of this history. We use

$$
v _ { i , t } = v _ { i , t } ^ { ( R ) } = ( c _ { i , t } , \rho _ { i , t } )\tag{5}
$$

to denote the final proposal submitted to the Decide phase. When $R = 0$ , as in our main setting, this is simply the initial proposal $v _ { i , t } ^ { ( 0 ) }$

The Decide phase applies an aggregation rule Agg to a selected pool $P _ { t } \subseteq \mathcal { P }$ . For the voting-based methods considered below, let $y _ { i , t } ^ { c a n d }$ denote the normalized answer parsed from the payload $c _ { i , t }$ . We set $y _ { i , t } ^ { c a n d } = \bot$ if parsing fails. Given nonnegative weights $w _ { i , t }$ for the selected identities, the final answer is

$$
\hat { y } _ { t } \in \arg \operatorname* { m a x } _ { a \neq \perp } \sum _ { i \in P _ { t } } w _ { i , t } \mathbb { I } [ y _ { i , t } ^ { c a n d } = a ] .\tag{6}
$$

Eq. 6 gives weighted answer-group voting for MATH, with equality interpreted through the protocol’s candidate comparison. Label-set and code readouts are specified under task-dependent aggregation below.

To compare aggregation methods independently of LLM-generation variance, we generate and cache the final proposal set

$$
\mathcal { R } _ { t } = \{ v _ { i , t } : i \in \mathcal { P } \}\tag{7}
$$

once for each task. Every aggregation method receives the same $\mathcal { R } _ { t }$ . Thus, any difference in the resulting decision is caused by which agents are included in $P _ { t }$ , how much weight is assigned to each selected proposal, and how the weighted proposals are combined into the final answer, rather than by different sampled agent responses.

Estimated-quality feedback. To support the stateful methods evaluated in Sec. 4.1, we define an estimatedquality feedback interface. This interface is introduced for our experimental comparison and is not a standard component of MAD protocols. For each identity i whose proposal is observed on task $t ,$ let $z _ { i , t } \in [ 0 , 1 ]$ denote an estimated quality score, $\eta _ { i , t } \in \{ 0 , 1 \}$ indicate whether the proposal can be parsed successfully, and $c _ { i , t } ^ { f m t } \in [ 0 , 1 ]$ measure format compliance. We define

$$
u _ { i , t } = \eta _ { i , t } z _ { i , t } , \qquad x _ { i , t } = z _ { i , t } ( 0 . 7 5 + 0 . 2 5 c _ { i , t } ^ { f m t } ) .
$$

These quantities provide estimated-quality signals rather than benchmark ground-truth labels. Stateful aggregation methods update their persistent state only after the decision on task t has been made.

Uniform Majority and Uniform Random-k. Neither Uniform Majority nor Uniform Random-k maintains persistent state. Uniform Majority applies equal-weight voting to all identities (Kaesberg et al., 2025):

$$
P _ { t } = \mathcal { P } , \qquad w _ { i , t } = 1 \quad ( i \in \mathcal { P } ) .\tag{8}
$$

Its pool therefore contains all $n = 1 0$ identities. This baseline shows the result when every agent participates and has the same influence, regardless of its performance on previous tasks.

Uniform Random-k is our size-matched sampling control. It samples a size-k pool uniformly without replacement and assigns equal weight to every selected identity:

$$
\operatorname* { P r } ( P _ { t } = S ) = { \binom { n } { k } } ^ { - 1 } \quad { \mathrm { f o r ~ e v e r y ~ } } S \subseteq { \mathcal { P } } { \mathrm { ~ w i t h ~ } } | S | = k , \qquad w _ { i , t } = 1 \quad ( i \in P _ { t } ) .\tag{9}
$$

The selection is independent of σ, agent feedback, and past performance. To make the control reproducible, the implementation uses a fixed pseudorandom seed for each experimental run and combines it with the task identifier when sampling $P _ { t }$ . Repeating the same run therefore yields the same pool for every task. This baseline isolates the effect of using only k identities without using reputation for selection or weighting.

For the stateful top-k methods considered below, $P _ { t }$ instead contains the k identities with the largest weights available before task t. Weight ties are resolved using a deterministic seeded rule. During the configured all-identity warm-up period, these baselines use $P _ { t } = \mathcal { P }$ . In contrast, MiniRep applies its same-task screening mechanism and constructs a size-k pool even during warm-up.

Single-Metric. Single-Metric is an in-study control designed to test what can be achieved using only one historical quality indicator per identity. It excludes the additional mechanisms used by MiniRep, including identity-risk detection, clone-group caps, cooldown, and same-task screening.

Let $\boldsymbol { \sigma } _ { i , t } ^ { \mathrm { S M } }$ denote the exponentially smoothed historical quality of identity i after incorporating feedback available through task t. This symbol is distinct from the fused same-task support $s _ { i , t }$ defined in Table 5. We initialize

$$
\sigma _ { i , 0 } ^ { \mathrm { S M } } = 0 . 5 \quad ( i \in \mathcal { P } ) .\tag{10}
$$

Before task t, Single-Metric uses the state available through task t − 1 to assign

$$
\begin{array} { r } { w _ { i , t } = \mathrm { c l i p } _ { [ 0 . 0 5 , 1 . 5 0 ] } \left( \sigma _ { i , t - 1 } ^ { \mathrm { S M } } \right) . } \end{array}\tag{11}
$$

It selects the k identities with the largest $w _ { i , t }$ values and applies the weighted-vote rule in Eq. 6. The same scalar historical score therefore controls both agent selection and voting weight.

After feedback for task t becomes available, the score of every observed identity is updated using a fixed-rate exponential moving average. Let $F _ { t }$ denote the identities for which feedback is available for this update (distinct from the detector’s candidate set $\mathcal { F } _ { t } )$

$$
\sigma _ { i , t } ^ { \mathrm { S M } } = 0 . 8 5 \sigma _ { i , t - 1 } ^ { \mathrm { S M } } + 0 . 1 5 x _ { i , t } \quad ( i \in F _ { t } ) .\tag{12}
$$

For an unobserved identity, the previous score is carried forward:

$$
\sigma _ { i , t } ^ { \mathrm { S M } } = \sigma _ { i , t - 1 } ^ { \mathrm { S M } } \quad ( i \notin F _ { t } ) .\tag{13}
$$

The update coefficient 0.15 controls how quickly recent feedback changes the historical estimate. Single-Metric otherwise maintains no additional persistent signals or current-task controls. Its initialization, clipping interval, and update rate are fixed settings of this in-study control.

The four reputation baselines below receive the same quality feedback $u _ { i , t }$ after task t. Each baseline uses this feedback to update the weight $w _ { i , t + 1 }$ for the next task. Unlike MiniRep, these baselines do not use the current proposal to adjust an agent’s weight before the current answer is produced.

Babylon. We adapt Babylon (Babylon contributors, 2026) to maintain one reputation score for each agent. Let $n _ { i }$ be the number of observed tasks, and let

$$
W _ { i } = \frac { 1 } { n _ { i } } \sum _ { \tau } \mathbf { 1 } [ z _ { i , \tau } \geq 0 . 9 9 9 ]\tag{14}
$$

be the fraction of observations with a nearly perfect verifier score. The adaptation also maintains $F _ { i }$ and $J _ { i }$ as running averages of

$$
( 2 0 + 8 0 z _ { i , t } ) \left( 0 . 7 5 + 0 . 2 5 c _ { i , t } ^ { f m t } \right) .\tag{15}
$$

Thus, both records increase with the verifier score and with compliance with the required output format. A separate record $V _ { i , t }$ tracks the verifier score over time:

$$
V _ { i , 0 } = 0 . 5 , \qquad V _ { i , t } = 0 . 9 0 V _ { i , t - 1 } + \frac { 0 . 1 0 } { 1 + \exp [ - ( 2 z _ { i , t } - 1 ) ] } .\tag{16}
$$

Babylon combines these records with the number of observed tasks:

$$
\begin{array} { r l } & { S _ { i , t } ^ { \mathrm { B a b } } = 0 . 4 \left[ 1 0 0 \left( 0 . 7 V _ { i , t } + 0 . 3 W _ { i } \right) \right] + 0 . 4 \left( 0 . 7 F _ { i } + 0 . 3 J _ { i } \right) } \\ & { \phantom { J _ { i , t } ^ { \mathrm { B a b } } = } + 0 . 2 \operatorname* { m i n } \{ 1 0 0 , 2 n _ { i } \} , } \\ & { w _ { i , t + 1 } = \operatorname* { m a x } \{ 0 . 0 5 , S _ { i , t } ^ { \mathrm { B a b } } / 1 0 0 \} . } \end{array}\tag{17}
$$

The initial reputation is 50. The $0 . 4 / 0 . 4 / 0 . 2$ component weights, $0 . 7 / 0 . 3$ blends, activity term, and initial score follow the cited implementation; the verifier-based feedback mapping is our MAD adaptation. This baseline evaluates whether a single score combining past performance, format compliance, and the number of observed tasks is sufficient for MAD aggregation.

Beta Reputation. Beta Reputation (Jøsang and Ismail, 2002) accumulates positive and negative evidence for each agent. The quality feedback $u _ { i , t }$ is treated as fractional positive evidence, and $1 - u _ { i , t }$ is treated as fractional negative evidence:

$$
\alpha _ { i , 0 } = \beta _ { i , 0 } = 1 , \qquad \alpha _ { i , t } = \alpha _ { i , t - 1 } + u _ { i , t } , \qquad \beta _ { i , t } = \beta _ { i , t - 1 } + 1 - u _ { i , t } .\tag{18}
$$

The weight for the next task is proportional to the estimated fraction of positive evidence:

$$
w _ { i , t + 1 } \propto \frac { \alpha _ { i , t } } { \alpha _ { i , t } + \beta _ { i , t } } .\tag{19}
$$

The weights are normalized across agents. All previous observations remain in $\alpha _ { i }$ and $\beta _ { i } ,$ , so older evidence is not discarded or reduced. The (1, 1) initialization corresponds to a uniform Beta prior. This baseline evaluates direct accumulation of quality evidence without maintaining a separate record of suspicious behavior.

EigenTrust. EigenTrust (Kamvar et al., 2003) originally computes trust from ratings between participants. Our setting does not collect pairwise ratings from agents. We therefore use the shared quality feedback $u _ { j , t }$ as evidence about each observed agent j. For every pair (i, j), we update

$$
S _ { i j } ^ { + }  S _ { i j } ^ { + } + u _ { j , t } , \qquad S _ { i j } ^ { - }  S _ { i j } ^ { - } + 1 - u _ { j , t } ,\tag{20}
$$

with $S _ { i j } ^ { + } = S _ { i j } ^ { - } = 0$ initially. Because the same feedback is available to the system, every row receives the same evidence about agent j. We define

$$
s _ { i j } = \operatorname* { m a x } \{ S _ { i j } ^ { + } - S _ { i j } ^ { - } , 0 \} ,\tag{21}
$$

and normalize each row:

$$
\begin{array} { r } { C _ { i j } = \left\{ \begin{array} { l l } { s _ { i j } / \sum _ { \ell } s _ { i \ell } , } & { \sum _ { \ell } s _ { i \ell } > 0 , } \\ { 1 / n , } & { \mathrm { o t h e r w i s e } . } \end{array} \right. } \end{array}\tag{22}
$$

The weights are obtained by repeatedly applying

$$
\pmb { w } = 0 . 8 5 C ^ { \top } \pmb { w } + 0 . 1 5 \pmb { p } , \qquad \pmb { p } _ { i } = 1 / n .\tag{23}
$$

The computation starts from p and stops when the $\ell _ { 1 }$ change is at most $1 0 ^ { - 1 0 }$ or after 100 iterations. The 0.15 damping, uniform prior, and stopping rule are fixed settings of this adaptation. This adaptation applies EigenTrust’s repeated trust calculation to the feedback available in our setting. It does not assume that agents provide independent ratings of one another.

TrueSkill. TrueSkill (Herbrich et al., 2006) maintains a skill estimate and its uncertainty for each agent:

$$
\theta _ { i } \sim \mathcal N ( \mu _ { i } , \sigma _ { i } ^ { 2 } ) .\tag{24}
$$

After each task, agents are ranked according to decreasing $u _ { i , t }$ . Scores that differ by at most $1 0 ^ { - 1 2 }$ are treated as tied. TrueSkill updates $\mu _ { i }$ and $\sigma _ { i }$ from these rankings. We use the initialization and update defaults of the Python trueskill implementation (Lee, 2026):

$$
\mu _ { 0 } = 2 5 , \qquad \sigma _ { 0 } = 2 5 / 3 , \qquad \beta _ { \mathrm { T S } } = 2 5 / 6 , \qquad \tau = 2 5 / 3 0 0 , \qquad p _ { \mathrm { d r a w } } = 0 . 1 0 .\tag{25}
$$

The score

$$
h _ { i } ^ { \mathrm { T S } } = \mu _ { i } - 3 \sigma _ { i }\tag{26}
$$

favors agents with high estimated skill and low uncertainty. We convert it into a positive weight for the next task:

$$
w _ { i , t + 1 } \propto \exp \left( { \mathrm { c l i p } _ { [ - 5 0 , 5 0 ] } \frac { h _ { i } ^ { \mathrm { T S } } - \operatorname* { m i n } _ { j } h _ { j } ^ { \mathrm { T S } } } { \operatorname* { m a x } \{ 1 , \beta _ { \mathrm { T S } } \} } } \right) .\tag{27}
$$

This baseline evaluates whether accounting for uncertainty in an agent’s estimated capability improves aggregation. It does not maintain a separate record of suspicious behavior.

Task-dependent aggregation rules. These rules implement Agg in Eq. 2, corresponding to weighted aggregation in Fig. 1. After each method determines the aggregation pool $P _ { t }$ and weights $\{ w _ { i , t } \} _ { i \in P _ { t } }$ , a task-dependent aggregation rule converts the selected proposals into the final answer $\hat { y } _ { t }$ . These rules are used by all evaluated methods. The MATH and HumanEval Pro rules are identical across methods. For GoEmotions, the seven baselines share one rule, while MiniRep uses the alternative rule defined below. The task-specific scoring coefficients, label thresholds, and label-count cap are fixed choices of these aggregation implementations, not benchmark-prescribed defaults.

Let $V _ { t } \subseteq P _ { t }$ contain the agents whose proposals provide valid, nonempty candidate answers, and let $\begin{array} { r } { W _ { t } = \sum _ { i \in V _ { t } } w _ { i , t } } \end{array}$ . Proposals without a valid candidate answer receive no aggregation weight. If $V _ { t }$ is empty, the method returns an empty answer, which is evaluated as incorrect.

For MATH, we group candidate answers according to mathematical equivalence and sum the weights within each group. The group with the largest total weight determines the final answer. This comparison uses only the submitted candidate answers and does not use the reference answer.

For HumanEval Pro, the aggregation rule selects the candidate program $i \in V _ { t }$ with the highest score:

$$
2 p _ { i } ^ { \mathrm { e x a m p l e } } + 0 . 9 \frac { \sum _ { j \in V _ { t } : F _ { j } = F _ { i } } w _ { j , t } } { W _ { t } } + 0 . 3 5 \frac { w _ { i , t } } { W _ { t } } + 0 . 2 5 s _ { i } ^ { \mathrm { s t a t i c } } ,\tag{28}
$$

where $p _ { i } ^ { \mathrm { e x a m p l e } }$ measures whether the program passes the examples included in the task, $F _ { i }$ identifies programs with the same normalized abstract syntax tree, and $s _ { i } ^ { \mathrm { s t a t i c } }$ checks whether the program is syntactically valid and contains the required entry point. The hidden evaluation tests are not used during aggregation.

For GoEmotions, the seven baselines distribute each proposal’s weight uniformly across its predicted label set $Y _ { i } { \mathrm { : } }$

$$
A _ { \ell } = \sum _ { i \in V _ { t } } w _ { i , t } { \frac { \mathbf { 1 } [ \ell \in Y _ { i } ] } { | Y _ { i } | } } .\tag{29}
$$

They return at most five labels satisfying

$$
A _ { \ell } \geq \operatorname* { m a x } \left\{ 0 . 4 5 \operatorname* { m a x } _ { h } A _ { h } , \ 0 . 1 8 W _ { t } \right\} ,\tag{30}
$$

and return the highest-scoring label if no label satisfies this condition. In contrast, MiniRep selects one of the label sets proposed by the agents. It chooses the set $Y$ that maximizes

$$
0 . 5 0 \frac { \sum _ { i \in V _ { t } : Y _ { i } = Y } w _ { i , t } } { W _ { t } } + 0 . 4 0 \frac { \sum _ { i \in V _ { t } } w _ { i , t } \operatorname { F } 1 ( Y , Y _ { i } ) } { W _ { t } } + 0 . 1 0 \frac { | \{ g ( i ) : i \in V _ { t } , Y _ { i } = Y \} | } { | \{ g ( i ) : i \in V _ { t } \} | } ,\tag{31}
$$

where $g ( i )$ denotes the clone group of agent i. This rule considers the total weight supporting $Y$ , its agreement with the other proposed label sets, and the number of clone groups that propose it. Therefore, the GoEmotions experiments compare the complete aggregation procedures rather than isolating the effect of reputation weights alone.

## B Deferred Design Details of MiniRep

The following details are organized by the three modules in Fig. 1 and Sec. 3.

Unless attributed to a cited method or checkpoint calibration, the numerical settings below are fixed choices of our implementation.

## B.1 Response Analyzer

## B.1.1 Public-evidence Score

This is the public-support score used by the Response Analyzer in Sec. 3. Each candidate receives a score computed from the cohort alone, with no gold labels and no hidden tests.

$$
p _ { i , t } ^ { p u b } = \eta _ { i , t } \big ( \beta _ { 1 } \pi _ { i , t } + \beta _ { 2 } \gamma _ { i , t } + \beta _ { 3 } \lambda _ { i , t } + \beta _ { 4 } s _ { i , t } ^ { s e l f } + \beta _ { 5 } ( 1 - \varepsilon _ { t } ) \big ) ,\tag{32}
$$

Here $\pi$ is cohort-agreement probability, $\gamma$ cross-model agreement, and λ leave-one-group-out plurality agreement. The remaining terms are same-model self-consistency $s ^ { s e l f }$ and normalized answer-distribution entropy $\varepsilon _ { t }$ . We fix the implementation weights at $\beta = ( 0 . 4 5 , 0 . 2 5 , 0 . 1 5 , 0 . 1 0 , 0 . 0 5 )$ , and $\eta _ { i , t }$ is the parsevalidity indicator. We compute all five statistics after grouping identities by underlying model, so replicas of one model count as one evidence source.

## B.1.2 Local feedback models

For the semantic verifier in Sec. 3, we fine-tune Qwen2.5-1.5B-Instruct as a binary correctness classifier (Qwen et al., 2025). Its input contains the task with gold fields removed, the candidate response, parse validity, answer-group sizes, same-model consistency, cohort agreement, and normalized cohort entropy. Agent and model identifiers, attack names, Byzantine labels, gold answers, and hidden tests are excluded. We use 4-bit LoRA (Hu et al., 2022; Dettmers et al., 2023) $( r = 1 6 , \alpha = 3 2 )$ with class-weighted focal cross-entropy (Lin et al., 2017) and a soft-ECE regularizer (Karandikar et al., 2021), followed by temperature scaling (Guo et al., 2017). The rank and scaling factor are fixed fine-tuning settings of our verifier, rather than defaults prescribed by these methods.

$$
\mathcal { L } _ { \mathrm { v e r } } = \frac { 1 } { N } \sum _ { j } ( 1 - p _ { y _ { j } } ) ^ { \gamma } \mathrm { C E } ( z _ { j } , y _ { j } ) + \lambda _ { \mathrm { c a l } } \widetilde { \mathrm { E C E } } ,\tag{33}
$$

Splits are made by canonical task, which prevents attack variants of one task from crossing partitions. The calibrated output is $p _ { i , t } ^ { s e m }$ . The attack families used to train the local feedback models are RAND, STRONG, DIV, ONOFF, and ADAPT; ARITH and BOUND are excluded from that training and evaluated in Appendix D.1.

Behavior probe. This probe provides the behavior-model signal for risk detection in Fig. 1. Correctness alone may miss coordination or build-then-attack behavior. We train an $1 1 \to 1 6 \to 8 \to 1 \mathrm { ~ M L P }$ on eleven label-free features: parse failure, semantic incorrectness, self-inconsistency, cohort disagreement, cohort entropy, confidence gap, response-length outlier, leave-one-out plurality flip, cross-round answer change, and two lexical indicators for instruction-override and consensus-spoofing phrasing. Its target is an active payload, never a static Byzantine identity. After standardization and temperature scaling it outputs $r _ { i , t } ^ { a t k }$ . The MLP layer widths are fixed architectural choices; its weights are learned from the active-payload labels described above.

## B.1.3 Computing quality, risk, and blocking signals

This subsection details the quality ${ q } _ { i , t } .$ , risk $r _ { i , t } ^ { a t k }$ , validity $\eta _ { i , t }$ , and blocking $d _ { i , t }$ signals produced by the Response Analyzer in Sec. 3. The Reputation-Aware Aggregator uses these signals to select agents and determine their weights. They are also passed to the State Updater after the final answer is produced.

Proposal quality. The Response Analyzer combines the public-support score $p _ { i , t } ^ { p u b }$ with the correctness probability $p _ { i , t } ^ { s e m }$ produced by the semantic verifier:

$$
s _ { i , t } = p _ { i , t } ^ { p u b } + \omega _ { i , t } \left( p _ { i , t } ^ { s e m } - p _ { i , t } ^ { p u b } \right) , \qquad \omega _ { i , t } = \omega _ { 0 } \left( 1 - ( 1 - \kappa ) \mathbf { 1 } \left[ p _ { i , t } ^ { p u b } \geq \theta _ { \mathrm { r e s } } \wedge p _ { i , t } ^ { s e m } < p _ { i , t } ^ { p u b } \right] \right) .\tag{34}
$$

The semantic verifier normally receives weight $\omega _ { 0 }$ . This weight is reduced when the public-support score is high but the semantic verifier gives a lower score. This prevents one low semantic-verifier score from overriding strong support from the information available on the current task. The two sources of evidence therefore contribute to quality estimation without either acting as an unconditional veto. The Response Analyzer then accounts for whether the proposal follows the required output format:

$$
q _ { i , t } = s _ { i , t } \left( 0 . 7 5 + 0 . 2 5 c _ { i , t } ^ { f m t } \right) , \qquad \neg \eta _ { i , t } \Rightarrow q _ { i , t } = 0 .\tag{35}
$$

Here, $c _ { i , t } ^ { f m t } \in [ 0 , 1 ]$ measures whether the proposal follows the expected output format. As defined in the main body, $\eta _ { i , t } = 1$ means that a candidate answer can be extracted from the proposal. If no candidate answer can be extracted, its quality is set to zero.

Risk and blocking. The behavior model produces the risk score $r _ { i , t } ^ { a t k }$ . It also checks for an instruction override, a false claim of agreement with other proposals, disagreement with the other proposals, and an unexpected change from the agent’s earlier response. Let $j _ { i , t } \in [ 0 , 1 ]$ be the larger of the instruction-override and false-agreement scores. Let $g _ { i , i }$ <sub>t</sub> measure agreement with the other proposals, and let $s _ { i , t } ^ { s e l f }$ measure same-model self-consistency, as in Eq. 32. We define

$$
b _ { i , t } ^ { r e s } = \mathbf { 1 } \left[ \eta _ { i , t } \wedge j _ { i , t } = 0 \wedge g _ { i , t } \geq 0 . 6 5 \wedge s _ { i , t } ^ { s e l f } \geq 0 . 6 5 \right] .\tag{36}
$$

When $b _ { i , t } ^ { r e s } = 1$ , the proposal can be parsed, contains no detected instruction override or false claim of agreement, and satisfies both the cohort-agreement and same-model consistency thresholds. This condition prevents such a proposal from being flagged only because one of its other scores is low.

The Response Analyzer identifies proposals that satisfy at least one of the following conditions:

$$
\begin{array} { r l } & { e _ { i , t } ^ { h a r d } = \mathbf { 1 } \left[ r _ { i , t } ^ { a t k } \geq 0 . 8 5 \wedge j _ { i , t } > 0 \right] , } \\ & { \quad o _ { i , t } = \mathbf { 1 } \left[ p _ { i , t } ^ { s e m } \leq 0 . 2 0 \wedge r _ { i , t } ^ { a t k } \geq \operatorname* { m a x } \{ 0 . 5 0 , 0 . 7 5 \theta _ { p } \} \wedge 1 - g _ { i , t } \geq 0 . 5 0 \wedge - b _ { i , t } ^ { r e s } \right] , } \\ & { \quad C _ { i , t } = \mathbf { 1 } [ \neg \eta _ { i , t } ] \vee e _ { i , t } ^ { h a r d } \vee o _ { i , t } \vee \mathbf { 1 } \left[ p _ { i , t } ^ { s e m } < \theta _ { s } \wedge r _ { i , t } ^ { a t k } \geq \theta _ { p } \wedge - b _ { i , t } ^ { r e s } \right] . } \end{array}\tag{37}
$$

The first condition detects a proposal with both high behavior-model risk and a detected instruction override or false claim of agreement. The second detects a proposal with low estimated correctness, sufficiently high behavior-model risk, and substantial disagreement with the other proposals. The final condition combines a low correctness probability with a high behavior-model risk. We use checkpoint-calibrated thresholds $\theta _ { s } = 0 . 1 5 8 7 6 5$ and $\theta _ { p } = 0 . 6 7 6 1 3 9$ , fixed across the reported conditions. For a parse-valid proposal, low estimated correctness alone is therefore insufficient to trigger a detector flag.

To select which proposals to flag, the Response Analyzer combines the behavior-model score, the semanticverifier score, $j _ { i , t }$ , and validity:

$$
a _ { i , t } = \mathrm { c l i p } _ { [ 0 , 1 ] } \left[ 0 . 5 5 r _ { i , t } ^ { a t k } + 0 . 3 0 ( 1 - p _ { i , t } ^ { s e m } ) + 0 . 1 0 j _ { i , t } + 0 . 0 5 ( 1 - \eta _ { i , t } ) \right] .\tag{38}
$$

Among the proposals satisfying $C _ { i , t } = 1$ , let $\mathcal { F } _ { t }$ contain up to ⌈0.30n⌉ agents with the highest ${ a } _ { i , t }$ . Ties are resolved by agent identity. The detector flag $f _ { i , t }$ and the blocking signal $d _ { i , t }$ are

$$
f _ { i , t } = \mathbf { 1 } [ i \in \mathcal { F } _ { t } ] , \qquad d _ { i , t } = \mathbf { 1 } [ \neg \eta _ { i , t } ] \vee f _ { i , t } , \qquad \sum _ { i } f _ { i , t } \leq \lceil 0 . 3 0 n \rceil .\tag{39}
$$

The limit applies only to proposals flagged by the detector. The 0.30 cap matches the configured corruption fraction $f / n = 3 / 1 0 !$ the remaining fusion and rescue constants are fixed implementation settings. A proposal from which no candidate answer can be extracted is always blocked, so such proposals are not subject to this limit. The State Updater records

$$
\widehat { a } _ { i , t } = \operatorname* { m a x } \{ a _ { i , t } , 0 . 9 0 d _ { i , t } \}\tag{40}
$$

after the final answer is produced.

## B.2 Reputation-Aware Aggregator

## B.2.1 Adjusting current weights and selecting agents

This subsection details reputation calibration and group-aware selection in the Reputation-Aware Aggregator shown in Fig. 1. It uses the signals defined above together with the historical weight $w _ { i , t } ^ { h i s t }$ to select the pool $P _ { t }$ and determine the aggregation weights $\{ w _ { i , t } \} _ { i \in P _ { t } }$

The Reputation-Aware Aggregator first converts the current and historical risk records into a value used to reduce the current weight:

$$
r _ { i , t } ^ { g a t e } = \left\{ \begin{array} { l l } { \operatorname* { m a x } \{ a _ { i , t } , 0 . 9 0 \} , } & { d _ { i , t } = 1 , } \\ { \operatorname* { m a x } \{ 0 . 2 5 a _ { i , t } , 0 . 5 0 I _ { i , t - 1 } \} , } & { d _ { i , t } = 0 . } \end{array} \right.\tag{41}
$$

A blocked proposal receives a value of at least 0.90. Otherwise, this value combines the risk of the current proposal with the suspicious-behavior record $I _ { i , t - 1 }$ from previous tasks. Current evidence can thus reduce influence before the historical reputation is updated.

The adjusted weight before applying the clone-group constraint is

$$
\begin{array} { r l } & { \widetilde { w } _ { i , t } = \mathrm { c l i p } _ { [ \epsilon , 1 0 ^ { 6 } ] } \left[ w _ { i , t } ^ { h i s t } \exp \left( 3 ( q _ { i , t } - q _ { t } ^ { m e d } ) - 8 r _ { i , t } ^ { g a t e } \right) m _ { i , t } \right] , } \\ & { m _ { i , t } = 0 . 0 1 ^ { d _ { i , t } } \cdot 0 . 0 2 ^ { Q _ { i , t } } \cdot \epsilon ^ { 1 - \eta _ { i , t } } . } \end{array}\tag{42}
$$

Here, $q _ { t } ^ { m e d }$ is the median quality among proposals from which a candidate answer can be extracted, and it is zero if no such proposal exists. We use $\epsilon = 1 0 ^ { - 1 2 }$ . The floor and clipping bound are numerical safeguards; the quality/risk coefficients, penalty multipliers, and clone-group constraints below are fixed aggregation settings. The indicator $Q _ { i , t } = 1$ means that previous tasks identified the agent as suspicious or that its temporary restriction remains active. The formula increases the weight of proposals with quality above the current-task median and reduces the weight of proposals with high risk. The multiplier $m _ { i , t }$ further reduces the weight of a blocked agent, an agent with a previous suspicious-behavior record, or a proposal from which no candidate answer can be extracted.

The Reputation-Aware Aggregator then limits the combined influence of agents in the same clone group. For a clone group $G$ containing $m = | G |$ agents, define

$$
M _ { G } = \sum _ { i \in G } \widetilde { w } _ { i , t } , \qquad p _ { G } = \operatorname* { m a x } _ { i \in G } \widetilde { w } _ { i , t } .\tag{43}
$$

The total weight of the group is limited to

$$
\operatorname* { m i n } \left\{ M _ { G } , p _ { G } \left[ 1 + 0 . 1 5 ( \sqrt { m } - 1 ) \right] \right\} ,\tag{44}
$$

and the weights of its members are reduced proportionally when this limit is exceeded. For example, three agents with equal weights from the same clone group receive at most 1.11 times the weight of one member, rather than three times that weight.

Finally, the Reputation-Aware Aggregator selects k agents for $P _ { t }$ . It first attempts to include agents from at least three clone groups and no more than three agents from one clone group. These requirements are relaxed only when necessary to fill all k positions. Agents blocked on the current task and agents whose temporary restrictions remain active are considered after the other agents. After selecting $P _ { t }$ , the Reputation-Aware Aggregator applies the quality adjustment and clone-group limit again within the selected pool. It then normalizes the resulting values to obtain $\{ w _ { i , t } \} _ { i \in P _ { t } }$ . The selected proposals and weights are then combined using the task-dependent aggregation rules in Appendix A.2.

## B.3 State Updater

## B.3.1 Updating records of suspicious behavior

This subsection details the behavioral-risk update in the State Updater shown in Fig. 1. After the Reputation-Aware Aggregator produces $\hat { y } _ { t }$ , the State Updater records the quality and risk of each valid proposal. These records reduce an agent’s historical weight on subsequent tasks after suspicious behavior and allow its influence to recover after sufficiently many proposals without a detector flag. The decay rates, risk thresholds, three-task cooldown, and recovery constants in this subsection are fixed state-update settings.

Responses without an extracted candidate answer. If $\eta _ { i , t } = 0$ , the proposal is excluded from the current aggregation. The State Updater does not treat the parse failure itself as evidence of a malicious proposal. The number of valid observations $n _ { i } .$ , the quality records, and the suspicious-behavior records $D _ { i } , K _ { i } , E _ { i }$ , and $I _ { i }$ remain unchanged. The counter $h _ { i }$ for the temporary restriction is reduced by one:

$$
h _ { i } \gets \operatorname* { m a x } ( 0 , h _ { i } - 1 ) .\tag{45}
$$

Thus, a parse failure does not create a new quality observation and does not remove a previously recorded risk.

Updating suspicious-behavior records. The State Updater maintains $E _ { i }$ as a recent record of detector results and $K _ { i }$ as a record that gives additional weight to detector flags and other high-risk observations. Before applying the recovery rule below, they are updated as

$$
\begin{array} { r l } & { E _ { i , t } ^ { \prime } = 0 . 6 5 E _ { i , t - 1 } + 0 . 3 5 \left[ d _ { i , t } \widehat { a } _ { i , t } + ( 1 - d _ { i , t } ) 0 . 3 0 a _ { i , t } \right] , } \\ & { K _ { i , t } ^ { \prime } = \mathrm { c l i p } _ { [ 0 , 1 ] } \left[ 0 . 9 0 K _ { i , t - 1 } + 0 . 5 5 d _ { i , t } + 0 . 1 5 ( 1 - d _ { i , t } ) r _ { i , t } ^ { i n s t } { \bf 1 } [ r _ { i , t } ^ { i n s t } \geq 0 . 6 5 ] \right] . } \end{array}\tag{46}
$$

When a valid proposal is flagged, the temporary-restriction counter is set to at least three tasks:

$$
h _ { i , t } = \operatorname* { m a x } \left\{ \operatorname* { m a x } ( 0 , h _ { i , t - 1 } - 1 ) , 3 \right\} .\tag{47}
$$

Otherwise, the counter decreases by one. After applying the recovery rule, the combined suspicious-behavior record is

$$
I _ { i , t } = \operatorname* { m a x } \left\{ E _ { i , t } , K _ { i , t } , 0 . 7 5 D _ { i , t } { \bf 1 } [ \mathrm { t h e \ a g e n t \ h a s \ p r e v i o u s l y \ b e e n \ t h a g g e d } ] \right\} .\tag{48}
$$

The agent is treated as previously suspicious when $I _ { i , t } \geq 0 . 5 5$ or $h _ { i , t } > 0$ . This sets $Q _ { i , t + 1 } = 1$ when the Reputation-Aware Aggregator processes the next task.

Reducing the effect of earlier suspicious behavior. A valid proposal contributes to recovery when it is not flagged and its quality satisfies

$$
q _ { i , t } \geq q _ { t } ^ { m e d } - 0 . 0 5 .\tag{49}
$$

The State Updater records the number of consecutive proposals satisfying these conditions. A valid proposal that does not satisfy them resets this count, while a proposal with $\eta _ { i , t } = 0$ leaves it unchanged. Once the count reaches $\nu = 3$ , each subsequent qualifying proposal reduces the suspicious-behavior records:

$$
E _ { i , t } = 0 . 7 5 E _ { i , t } ^ { \prime } , \qquad K _ { i , t } = 0 . 8 0 K _ { i , t } ^ { \prime } , \qquad D _ { i , t } = 0 . 9 0 D _ { i , t } ^ { \prime } .\tag{50}
$$

Otherwise, the primed values remain unchanged. The time required for an agent to recover therefore depends on the quality and risk of its subsequent proposals. The additional decay is thus tied to repeated qualifying proposals, not just elapsed time.

## B.3.2 Computing the historical weight

This subsection details the capability update and historical-weight computation of the State Updater in Sec. 3. The records from tasks up to $t - 1$ determine the historical weight $w _ { i , t } ^ { h i s t }$ used by the Reputation-Aware Aggregator on task t. The weight increases with the agent’s long-term and recent quality. It decreases when

the agent’s recent performance falls below its earlier performance or when suspicious behavior has been recorded. The prior means and strength, EMA rates, historical-weight coefficients, and drift/debt constants below are likewise fixed implementation settings.

The State Updater first computes

$$
\begin{array} { r l } & { L _ { i , t - 1 } = 6 \log ( \bar { e } _ { i , t - 1 } + 0 . 0 3 ) + 3 \log ( \bar { q } _ { i , t - 1 } + 0 . 0 3 ) } \\ & { \qquad + 1 . 5 \log \mathrm { c l i p } _ { [ 0 . 0 5 , 1 ] } \left( \frac { q _ { i , t - 1 } ^ { s h o r t } } { \bar { q } _ { i , t - 1 } } \right) + 2 \log \mathrm { c l i p } _ { [ 0 . 0 8 , 1 ] } \left( \frac { e _ { i , t - 1 } ^ { s h o r t } } { \bar { e } _ { i , t - 1 } } \right) } \\ & { \qquad - 6 r _ { i , t - 1 } ^ { t e m p } - 5 { \bf 1 } [ h _ { i , t - 1 } > 0 ] , } \\ & { w _ { i , t } ^ { h i s t } = \operatorname* { m a x } \left\{ \epsilon , \exp \left( L _ { i , t - 1 } - \operatorname* { m a x } L _ { j , t - 1 } \right) \right\} . } \end{array}\tag{51}
$$

Here, ${ \bar { q } } _ { i }$ records long-term proposal quality, $q _ { i } ^ { s h o r t }$ gives more weight to recent proposal quality, $\bar { e } _ { i }$ records the long-term correctness probability produced by the semantic verifier, and $e _ { i } ^ { s h o r t }$ gives more weight to recent correctness probabilities. The value $r _ { i } ^ { t e m p }$ is the largest recorded indication of suspicious behavior or a substantial performance decrease. The counter $h _ { i }$ records whether a temporary restriction remains active. Taking the largest risk value ensures that a high risk recorded from one source is not cancelled by lower values from the other records.

Updating long-term and recent quality. We initialize

$$
\bar { q } _ { i } = q _ { i } ^ { s h o r t } = 0 . 3 0 , \qquad \bar { e } _ { i } = e _ { i } ^ { s h o r t } = 0 . 1 8 .\tag{52}
$$

For a valid proposal, define

$$
u _ { i , t } ^ { c a p } = \left\{ p _ { i , t } ^ { s e m } , \mathrm { i f ~ t h e ~ s e m a n t i c - v e r i f i e r ~ s c o r e ~ i s ~ a v a i l a b l e , } \right.\tag{53}
$$

The long-term records are updated as

$$
\bar { q } _ { i , t } = \frac { ( 1 2 + n _ { i , t - 1 } ) \bar { q } _ { i , t - 1 } + q _ { i , t } } { 1 3 + n _ { i , t - 1 } } , \qquad \bar { e } _ { i , t } = \frac { ( 1 2 + n _ { i , t - 1 } ) \bar { e } _ { i , t - 1 } + u _ { i , t } ^ { c a p } } { 1 3 + n _ { i , t - 1 } } .\tag{54}
$$

The count $n _ { i }$ increases once for each valid proposal. The recent records are updated as

$$
q _ { i , t } ^ { s h o r t } = ( 1 - \alpha _ { i , t } ) q _ { i , t - 1 } ^ { s h o r t } + \alpha _ { i , t } q _ { i , t } , \qquad \alpha _ { i , t } = \left\{ \begin{array} { l l } { 0 . 3 0 , } & { q _ { i , t } < q _ { i , t - 1 } ^ { s h o r t } , } \\ { 0 . 0 8 , } & { \mathrm { o t h e r w i s e } , } \end{array} \right.\tag{55}
$$

$$
e _ { i , t } ^ { s h o r t } = 0 . 8 0 e _ { i , t - 1 } ^ { s h o r t } + 0 . 2 0 u _ { i , t } ^ { c a p } .
$$

A decrease in proposal quality therefore changes $q _ { i } ^ { s h o r t }$ faster than an improvement of the same size. We use $\omega _ { 0 } = 0 . 7 5 , \kappa = 0 . 8 5$ , and $\theta _ { \mathrm { r e s } } = 0 . 8 0$ in Eq. 34. When the semantic-verifier score is unavailable, its weight in Eq. 34 is set to zero.

Recording substantial performance decreases. For a valid proposal, define its decrease below the currenttask median as

$$
g _ { i , t } ^ { g a p } = \operatorname* { m a x } \left\{ 0 , q _ { t } ^ { m e d } - q _ { i , t } - 0 . 0 5 \right\} .\tag{56}
$$

The State Updater maintains a recent average of this value:

$$
G _ { i , t } = 0 . 7 5 G _ { i , t - 1 } + 0 . 2 5 g _ { i , t } ^ { g a p } .\tag{57}
$$

It compares this recent average with the agent’s earlier average $B _ { i , t - 1 } { \mathrm { : } }$

$$
r _ { i , t } ^ { c h g } = \mathrm { c l i p } _ { [ 0 , 1 ] } \left[ \frac { G _ { i , t } - B _ { i , t - 1 } - 0 . 0 2 5 } { 0 . 1 0 } \right] .\tag{58}
$$

The earlier average is updated only when the proposal is not blocked and the increase is small:

$$
B _ { i , t } = 0 . 9 9 5 B _ { i , t - 1 } + 0 . 0 0 5 g _ { i , t } ^ { g a p }\tag{59}
$$

when $r _ { i , t } ^ { c h g } < 0 . 2 5$ and $d _ { i , t } = 0$ . Otherwise, $B _ { i , t } = B _ { i , t - 1 }$

The State Updater also measures low absolute quality:

$$
r _ { i , t } ^ { a b s } = \mathrm { c l i p } _ { [ 0 , 1 ] } \left[ \frac { 0 . 5 0 - q _ { i , t } } { 0 . 5 0 } \right] , \qquad r _ { i , t } ^ { i n s t } = \operatorname* { m a x } \left\{ r _ { i , t } ^ { c h g } , r _ { i , t } ^ { a b s } , \widehat { a } _ { i , t } \right\} .\tag{60}
$$

It then updates $D _ { i } .$ , which records recent evidence of a substantial performance decrease:

$$
\widetilde { D } _ { i , t } = 0 . 7 5 D _ { i , t - 1 } + 0 . 2 5 \operatorname* { m a x } \left\{ r _ { i , t } ^ { c h g } , 0 . 7 0 r _ { i , t } ^ { a b s } , \widehat { a } _ { i , t } \right\} ,
$$

$$
D _ { i , t } ^ { \prime } = \left\{ \begin{array} { l l } { \operatorname* { m a x } \{ \widetilde { D } _ { i , t } , 0 . 9 0 \} , } & { d _ { i , t } = 1 , } \\ { \operatorname* { m a x } \{ \widetilde { D } _ { i , t } , 0 . 5 4 r _ { i , t } ^ { i n s t } \} , } & { d _ { i , t } = 0 \mathrm { ~ } \land r _ { i , t } ^ { i n s t } \geq 0 . 6 5 , } \\ { \widetilde { D } _ { i , t } , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.\tag{61}
$$

Finally, $r _ { i } ^ { t e m p }$ is the maximum of $D _ { i } , E _ { i } , I _ { i } , r _ { i } ^ { i n s t }$ , and the decrease of the long-term quality records from their previous high values. To measure this last decrease, the State Updater maintains

$$
H _ { i , t } ^ { x } = \operatorname* { m a x } \left\{ \bar { x } _ { i , t } , 0 . 9 9 9 9 H _ { i , t - 1 } ^ { x } \right\} , \qquad x \in \{ q , e \} .\tag{62}
$$

It compares ${ \bar { q } } _ { i }$ and $\bar { e } _ { i }$ with $H _ { i } ^ { q }$ and $H _ { i } ^ { e }$ , respectively. A large decrease in either value increases $r _ { i } ^ { t e m p }$ Proposals with $\eta _ { i , t } = 0$ do not update these records.

## C Detailed Experiment Setups

## C.1 Datasets

This subsection gives the sampling details for the benchmarks in Sec. 4.1.

We select 100 GoEmotions validation items without changing their order in the original dataset. For MATH, we shuffle the level-4/5 test items using seed 2027 and select 100 tasks. We use the same procedure to select 100 HumanEval Pro tasks from the split labeled train in its upstream release. These HumanEval Pro tasks are used only for evaluation. The 100-task sample size and seed 2027 are choices of our evaluation protocol, not defaults of the source benchmarks.

## C.2 Agent Pools and Configurations

Model rankings and clone groups. We rank the seven LLMs separately for each dataset using their standalone task scores. Each model has three standalone evaluations on the fixed evaluation tasks. For GoEmotions, the three evaluations use different personality prompts. For MATH, we retain the ranking used when constructing the agent pools and do not recompute it using the final answer-equivalence rule. From strongest to weakest, the rankings are Qwen-Max, Qwen-Flash, GLM, Qwen-Plus, DeepSeek, Kimi, and MiniMax for GoEmotions; Kimi, MiniMax, Qwen-Plus, Qwen-Max, GLM, Qwen-Flash, and DeepSeek for MATH; and Qwen-Max, Qwen-Plus, GLM, Qwen-Flash, DeepSeek, Kimi, and MiniMax for HumanEval Pro.

These rankings determine the four agent compositions in Table 2. The all-strong composition contains agents instantiated from the four highest-ranked models, with 3, 3, 2, and 2 agents per model. The all-weak composition uses models ranked fourth through seventh, with 2, 2, 3, and 3 agents per model. The 7S+3W composition contains 3, 2, and 2 agents from the three highest-ranked models and one agent from each of the three lowest-ranked models. The random composition uses a fixed selection generated with seed 2027. Agents instantiated from the same underlying model form a clone group.

The rankings also determine the corruption placements in Table 2. B and W compromise agents instantiated from higher- and lower-ranked models, respectively. R1 and R2 are two fixed subsets selected using seed 2027. The detailed result tables report the agents compromised in each selected condition.

Run configuration. We execute 128 runs per dataset: 112 attacked runs from four compositions, four corruption placements, and seven attacks, and 16 clean runs from four compositions and four placements. This gives 384 runs across the three datasets. For a fixed composition, the four clean placements produce the same task scores and are counted once when computing clean averages. The semantic verifier and behavior model are shared across methods; each method’s state-update parameters are fixed across experimental conditions. The OnOff and Adapt attacks begin after the first 30 tasks. This 30-task schedule and the pool, corruption, and selection sizes in Sec. 4.1 are fixed controlled-evaluation settings. For Adapt, the active malicious agents are selected according to the current MiniRep state, and the resulting attack plan is applied to every method in the same experimental condition.

Feedback and aggregation. This setup implements the comparison described in Sec. 4.1, using the taskdependent aggregation rules in Appendix A.2. All stateful methods receive the same quality assessment from the semantic verifier after each task. MiniRep additionally uses this assessment in the Response Analyzer before aggregation and in the State Updater after the final answer is produced. On GoEmotions, the baselines distribute each proposal’s weight among its predicted labels. MiniRep instead selects one of the label sets submitted by the agents based on their weights, agreement, and clone groups. Therefore, the GoEmotions results compare the complete methods, including their aggregation rules, rather than isolating only the effect of historical reputation.

## C.3 Metrics

We define the task metrics reported in Sec. 4.1 and the additional diagnostics reported in Appendix D.

F1, accuracy, and Pass@1. These standard task metrics measure label overlap, exact correctness, and successful execution, respectively. For N evaluated tasks, let $\hat { y } _ { t }$ be the final aggregate and $y _ { t }$ the reference. Recall that the sample-averaged F1 score (Sokolova and Lapalme, 2009) and exact-match accuracy are defined as:

$$
\mathrm { F } 1 = \frac { 1 } { N } \sum _ { t = 1 } ^ { N } \frac { 2 | \widehat { Y } _ { t } \cap Y _ { t } | } { | \widehat { Y } _ { t } | + | Y _ { t } | } , \qquad \mathrm { A c c } _ { \mathrm { s e t } } = \frac { 1 } { N } \sum _ { t = 1 } ^ { N } \mathbf { 1 } [ \widehat { Y } _ { t } = Y _ { t } \neq \emptyset ] ,\tag{63}
$$

where $\widehat { Y } _ { t }$ and $Y _ { t }$ denote the predicted and reference label sets after normalization. We use these metrics for the GoEmotions dataset. In addition, for MATH, we model the accuracy as $\begin{array} { r } { N ^ { - 1 } \sum _ { t } { \mathbf { 1 } } [ \hat { y } _ { t } \equiv y _ { t } ] } \end{array}$ , where ≡ is the evaluator’s answer-equivalence predicate. For HumanEval Pro, we use the Pass@k metric (Chen et al., 2021) where $k = 1$ , denoted as $\begin{array} { r } { \mathrm { P a s s @ 1 } = N ^ { - 1 } \sum _ { t } \mathbf { 1 } [ \hat { y } _ { t } } \end{array}$ passes all evaluation tests].

$\Delta$ and RelDrop: Paired loss and retention. These measures quantify the clean-to-attacked performance changes discussed in Sec. 4.2. For method m and N evaluated tasks, let $s _ { m , t } ^ { 0 }$ and $s _ { m , t } ^ { a }$ denote its clean and attacked scores on task t. Their averages are $\begin{array} { r } { S _ { m } ^ { 0 } = N ^ { - 1 } \sum _ { t } s _ { m , t } ^ { 0 } } \end{array}$ and $\begin{array} { r } { S _ { m } ^ { a } = \dot { N } ^ { - 1 } \sum _ { t } s _ { m , t } ^ { \dot { a } } } \end{array}$ . Each clean–attack pair uses the same method, tasks, seed, agent composition, and cached proposals. The clean run executes the complete protocol without attacks and maintains its own reputation scores.

We define the paired loss $\Delta _ { m }$ , relative performance drop RelDrop<sub>m</sub>, and retention advantage over baseline $b ,$ denoted by $R _ { b }$ , as:

$$
\Delta _ { m } = 1 0 0 ( S _ { m } ^ { 0 } - S _ { m } ^ { a } ) , \qquad \mathrm { R e l D r o p } _ { m } = 1 0 0 { \frac { S _ { m } ^ { 0 } - S _ { m } ^ { a } } { S _ { m } ^ { 0 } } } , \qquad R _ { b } = \Delta _ { b } - \Delta _ { \mathrm { M i n i R e p } } .\tag{64}
$$

$\Delta _ { m }$ and $R _ { b }$ are measured in percentage points. RelDrop is a percentage and is undefined when $S _ { m } ^ { 0 } = 0 . \mathrm { ~ A ~ }$ negative loss means that the attacked run performs better than its clean counterpart. A positive $R _ { b }$ means that MiniRep loses fewer points than baseline b from their respective clean scores. It does not necessarily mean that MiniRep has a higher attacked score. The paired comparison separates attack-related performance loss from differences in clean performance.

PoolExp and PayloadHit: Current-task exposure and payload agreement. These diagnostics examine pool selection and final aggregation in Fig. 1; results appear in Table 7. Let $A _ { t } = 1$ indicate that an attack is active on task $t ,$ let $J _ { t }$ denote the agents whose proposals contain an attack payload, and let $P _ { t }$ denote the aggregation pool. We define $E _ { t } = \mathbf { 1 } [ P _ { t } \cap J _ { t } \neq \emptyset ]$ to indicate that at least one attack payload enters the aggregation pool. We also define $H _ { t } = \mathbf { 1 } [ \hat { y } _ { t }$ matches a payload from $P _ { t } \cap J _ { t } ]$ . PoolExp and PayloadHit are defined as:

$$
\mathrm { P o o l E x p } = \frac { \sum _ { t } E _ { t } } { \sum _ { t } A _ { t } } , \qquad \mathrm { P a y l o a d H i t } = \frac { \sum _ { t } E _ { t } H _ { t } } { \sum _ { t } E _ { t } } .\tag{65}
$$

PoolExp measures the fraction of tasks with $A _ { t } = 1$ for which at least one proposal containing an attack payload enters the aggregation pool. PayloadHit measures the fraction of tasks with $E _ { t } = 1$ for which the final output matches an attack payload in the aggregation pool. A match is determined using label-set equality for GoEmotions, answer equivalence for MATH, and equality between the normalized program structures for HumanEval Pro. PayloadHit is undefined when $\textstyle \sum _ { t } E _ { t } = 0$

NextExcl: Exclusion from the next effective pool. This diagnostic examines subsequent pool selection after the state update in Fig. 1; results appear in Table 7. Let $B _ { t }$ denote the set of agents that actively attack on task t. Let $\tau ( t )$ denote the first subsequent task whose pool selection can use the information collected on task t. With immediate updates, $\tau ( t ) = t + 1$ . We define $\mathcal { T } ^ { + } = \{ t : B _ { t } \neq \emptyset , \tau ( t )$ is evaluated}. NextExcl is defined as:

$$
\mathrm { N e x t E x c l } = \frac { 1 } { | T ^ { + } | } \sum _ { t \in \mathcal { T } ^ { + } } \mathbf { 1 } [ B _ { t } \cap P _ { \tau ( t ) } = \emptyset ] .\tag{66}
$$

NextExcl measures the fraction of eligible attack tasks for which all agents that attacked on task t are excluded from the next pool affected by the updated reputation state. Tasks without an evaluated future pool are omitted. The metric evaluates exclusion at the task level rather than separately for each attacker.

CF-Harm: Counterfactual harm. This diagnostic supplements the task scores in Sec. 4.2 by measuring how often attacks reduce per-task performance. Let $s _ { m , t } ^ { 0 }$ and $s _ { m , t } ^ { a }$ denote the clean and attacked scores of method m on task t. Using the paired clean and attacked runs, we define CF-Harm as:

$$
\mathrm { C F \mathrm { - } H a r m } = \frac { \sum _ { t } A _ { t } \mathbf { 1 } [ s _ { m , t } ^ { 0 } - s _ { m , t } ^ { a } > 1 0 ^ { - 1 2 } ] } { \sum _ { t } A _ { t } } .\tag{67}
$$

CF-Harm measures the fraction of active-attack tasks on which the attacked score is lower than the paired clean score. It measures how frequently an attack causes harm rather than the magnitude of that harm. For MATH and HumanEval Pro, it counts tasks that are correct in the clean run but incorrect in the attacked run. For GoEmotions, it also captures partial decreases in sample F1. Unlike the signed paired loss $\Delta$ improvements on other tasks cannot cancel these harmful cases.

## C.3.1 Averaging and comparison rules

These rules define the averages and baseline comparisons in Table 3, Table 4, and Appendix D. For a method m, let $c \in \{ 1 , \ldots , 4 \}$ denote the agent composition, $p \in \{ 1 , \ldots , 4 \}$ the corruption placement, and $a \in \{ 1 , \dots , 7 \}$ the attack. Let $S _ { m , c } ^ { 0 }$ be the clean score for composition c, and let $S _ { m , c , p , a } ^ { \mathrm { a t k } }$ be the attacked score for one combination of composition, placement, and attack.

Average task scores. The clean average counts each of the four agent compositions once. The attacked average assigns equal weight to all $4 \times 4 \times 7 = 1 1 2$ attack conditions:

$$
\overline { { S } } _ { m } ^ { 0 } = \frac { 1 } { 4 } \sum _ { c = 1 } ^ { 4 } S _ { m , c } ^ { 0 } , \qquad \overline { { S } } _ { m } ^ { \mathrm { a t k } } = \frac { 1 } { 1 1 2 } \sum _ { c = 1 } ^ { 4 } \sum _ { p = 1 } ^ { 4 } \sum _ { a = 1 } ^ { 7 } S _ { m , c , p , a } ^ { \mathrm { a t k } } .\tag{68}
$$

The average paired loss is

$$
\overline { { { \Delta } } } _ { m } = \frac { 1 0 0 } { 1 1 2 } \sum _ { c = 1 } ^ { 4 } \sum _ { p = 1 } ^ { 4 } \sum _ { a = 1 } ^ { 7 } \left( S _ { m , c } ^ { 0 } - S _ { m , c , p , a } ^ { \mathrm { a t k } } \right) .\tag{69}
$$

For RelDrop, we first compute the relative loss of each attack condition and then average these values. Let

$$
\mathcal { C } _ { m } = \left\{ \left( c , p , a \right) : S _ { m , c } ^ { 0 } > 0 \right\} .\tag{70}
$$

We define

$$
\overline { { \mathrm { R e l D r o p } } } _ { m } = \frac { 1 0 0 } { | { \mathcal C } _ { m } | } \sum _ { ( c , p , a ) \in { \mathcal C } _ { m } } { \frac { S _ { m , c } ^ { 0 } - S _ { m , c , p , a } ^ { \mathrm { a t k } } } { S _ { m , c } ^ { 0 } } } .\tag{71}
$$

Thus, RelDrop is not computed by dividing the difference between the two average scores by the average clean score.

Average defense rates. PoolExp, PayloadHit, NextExcl, and CF-Harm are first computed separately for each attack condition. Each condition for which a metric is defined contributes equally to its reported average. If a denominator is zero, that condition is omitted from the average for that metric rather than assigned a value of zero. PoolExp, NextExcl, and CF-Harm are defined in all 112 attack conditions for every dataset and method. PayloadHit is undefined when no proposal containing an attack payload enters the aggregation pool. In the method order UMaj, URand, Single, Babylon, Eigen, Beta, TS, and MiniRep, the numbers of conditions with defined PayloadHit are

$$
\begin{array} { r l } { \mathrm { G o E m o t i o n s : } } & { { } ( 1 1 2 , 1 1 2 , 1 1 2 , 1 1 1 , 1 1 2 , 1 1 0 , 1 0 9 , 1 1 2 ) , } \end{array}
$$

$$
\begin{array} { r l } { \mathbf { M A T H : } } & { { } ( 1 1 2 , 1 1 2 , 1 1 2 , 1 1 2 , 1 1 2 , 1 1 2 , 1 1 0 , 1 1 1 ) , } \end{array}\tag{72}
$$

HumanEval Pro: (112, 112, 111, 111, 111, 110, 112, 106).

Baseline selection and strict wins. In Table 4(b)–(c), Table 9, and Table 10, baseline b is the baseline with the highest attacked score under the same dataset, composition, corruption placement, and attack. Score ties are resolved using the smaller paired loss and then the displayed method order. In Table 3, the best baseline is selected separately for the clean and attacked averages. The two selected baselines may therefore be different, and their scores do not define a paired loss.

A strict win requires a method’s unrounded task score to exceed the scores of all seven other methods; ties within $1 0 ^ { - 1 2 }$ are excluded. Each agent composition has $4 \times 7 = 2 8$ attack conditions. In Table 4(a), the displayed baseline is the baseline with the largest number of strict wins. These counts summarize the evaluated conditions and do not represent statistical significance across independent seeds.

## C.4 Benchmark examples

These examples expand the benchmark descriptions in Sec. 4.1, illustrating the information given to agents, the expected candidate answers, and the evaluation metrics. The candidate answers below are illustrative rather than outputs recorded from our experiments. Reference answers and hidden evaluation tests are used only to evaluate the final output. They are not available to MiniRep when selecting or aggregating proposals.

MATH. One level-4 Algebra problem asks for the center of the circle defined by $x ^ { 2 } - 6 x + y ^ { 2 } + 2 y =$ 9 (Hendrycks et al., 2021). The agents receive the equation and the request for its center. Completing the square gives $( x - 3 ) ^ { 2 } + ( y + 1 ) ^ { 2 } = 1 9$ , so the correct answer is $( 3 , - 1 )$ . A proposal containing (3, 1) can be successfully parsed but is mathematically incorrect. During aggregation, submitted answers are grouped according to mathematical equivalence without using the reference answer. The final output is compared with the reference answer only during evaluation. This example shows why extracting a candidate answer successfully does not imply that the answer is correct.

GoEmotions. The validation example eczdvun contains the text “Thank you. I really appreciate your response” and is annotated with the labels {admiration, gratitude} in the released dataset (Demszky et al., 2020). The agents receive the text and the available emotion labels, and each candidate answer is a set of labels. Predicting only {gratitude} gives a sample F1 score of 2/3 but an exact-match accuracy of zero. Predicting both reference labels gives a score of one under both metrics. This example illustrates that sample F1 gives partial credit for overlapping labels, while exact-match accuracy requires the complete reference label set.

HumanEval Pro. In the released example with ID 0, the function has\_close\_elements(numbers, threshold) checks whether a list contains two numbers whose distance is strictly smaller than the threshold. The extended function find\_close\_elements\_lists(list\_of\_lists, threshold) returns the indices of the lists satisfying this condition (Yu et al., 2025). For example, the input [[1.0, 2.0, 3.0], [1.0, 2.8, 3.0, 4.0, 5.0, 2.0]] with threshold 0.5 produces [1]. An agent must submit Python code implementing the required functions rather than only the output for this example. The task specification and public examples can be used during aggregation, while Pass@1 is determined by whether the selected program passes all evaluation tests. Replacing the strict comparison with a non-strict comparison may preserve the public-example output but fail when two numbers differ by exactly the threshold, illustrating a Boundary Value attack.

## D Additional Evaluation Results

Average task performance. Table 6 expands Table 3 by reporting all eight methods together with their clean scores, attacked scores, paired losses $\Delta .$ , and relative performance drops (RelDrop). We define $\Delta$ and RelDrop in Appendix C.3 and describe the averaging rules in Appendix C.3.1. The shaded cells are the values summarized in the main body.

MiniRep achieves the highest attacked average on three of the four task metrics. The exception is GoEmotions sample F1, where MiniRep achieves 33.94% and Beta achieves 34.87%. On both GoEmotions metrics, MiniRep has the smallest ∆ and RelDrop, indicating the lowest performance degradation under attacks. On MATH, MiniRep achieves the highest attacked accuracy of 61.95%, compared with 54.37% for Beta. However, its paired loss is 4.80 percentage points, while Beta’s paired loss is 0.88 percentage points. These results are not contradictory because the attacked score measures performance after attacks, while paired loss measures the change from each method’s own clean score. On HumanEval Pro, MiniRep achieves the highest attacked Pass@1 of 78.15%. EigenTrust has the smallest paired loss because its attacked average is slightly higher than its clean average.

Table 6: Average performance for all eight methods. Clean scores are averaged over the four compositions. Attacked scores, paired losses, and relative drops are the average of 112 attack conditions. Scores and RelDrop are percentages. ∆ is in percentage points. Shaded cells appear in Table 3; bold marks the best value in each row.
<table><tr><td>Metric / setting</td><td>UMaj</td><td>URand</td><td>Single</td><td>Babylon</td><td>Eigen</td><td>Beta</td><td>TS</td><td>MiniRep</td></tr><tr><td colspan="9">GoEmotions / exact-match accuracy</td></tr><tr><td>Clean↑</td><td>18.50</td><td>17.50</td><td>20.25</td><td>20.50</td><td>17.75</td><td>21.50</td><td>20.50</td><td>21.75</td></tr><tr><td>Attacked↑</td><td>10.71</td><td>11.04</td><td>19.04</td><td>18.05</td><td>12.42</td><td>19.59</td><td>17.79</td><td>21.16</td></tr><tr><td>∆↓</td><td>7.79</td><td>6.46</td><td>1.21</td><td>2.45</td><td>5.33</td><td>1.91</td><td>2.71</td><td>0.59</td></tr><tr><td>RelDrop↓</td><td>42.64</td><td>37.30</td><td>6.41</td><td>12.93</td><td>30.33</td><td>9.25</td><td>13.54</td><td>2.52</td></tr><tr><td colspan="9">GoEmotions / sample F1</td></tr><tr><td>Clean↑</td><td>36.42</td><td>33.63</td><td>35.81</td><td>35.64</td><td>35.32</td><td>35.89</td><td>35.56</td><td>34.42</td></tr><tr><td>Attacked↑</td><td>27.89</td><td>27.76</td><td>34.14</td><td>33.72</td><td>28.53</td><td>34.87</td><td>33.61</td><td>33.94</td></tr><tr><td>Δ↓</td><td>8.53</td><td>5.88</td><td>1.67</td><td>1.92</td><td>6.79</td><td>1.02</td><td>1.95</td><td>0.48</td></tr><tr><td>RelDrop↓</td><td>23.18</td><td>17.48</td><td>4.62</td><td>5.47</td><td>19.04</td><td>2.94</td><td>5.42</td><td>1.40</td></tr><tr><td colspan="9">MATH / mathematical-equivalence accuracy</td></tr><tr><td>Clean↑</td><td>59.25</td><td>56.25</td><td>54.75</td><td>54.25</td><td>56.00</td><td>55.25</td><td>51.75</td><td>66.75</td></tr><tr><td>Attacked↑</td><td>43.75</td><td>43.48</td><td>53.85</td><td>52.54</td><td>42.01</td><td>54.37</td><td>48.56</td><td>61.95</td></tr><tr><td>Δ↓</td><td>15.50</td><td>12.77</td><td>0.90</td><td>1.71</td><td>13.99</td><td>0.88</td><td>3.19</td><td>4.80</td></tr><tr><td>RelDrop↓</td><td>26.15</td><td>21.95</td><td>2.13</td><td>3.53</td><td>25.11</td><td>1.23</td><td>5.89</td><td>6.69</td></tr><tr><td colspan="9">HumanEval Pro / Pass@ 1</td></tr><tr><td>Clean↑</td><td>79.50</td><td>78.75</td><td>77.50</td><td>77.00</td><td>76.50</td><td>76.75</td><td>77.50</td><td>77.75</td></tr><tr><td>Attacked↑</td><td>51.75</td><td>56.13</td><td>74.69</td><td>72.20</td><td>77.92</td><td>73.87</td><td>71.01</td><td>78.15</td></tr><tr><td>∆↓</td><td>27.75</td><td>22.62</td><td>2.81</td><td>4.80</td><td>-1.42</td><td>2.88</td><td>6.49</td><td>-0.40</td></tr><tr><td>RelDrop↓</td><td>34.90</td><td>28.74</td><td>3.63</td><td>6.25</td><td>-1.89</td><td>3.75</td><td>8.41</td><td>-0.53</td></tr></table>

Defense effectiveness. These diagnostics supplement the task scores in Sec. 4.2 by examining how attacks affect selection and aggregation in Fig. 1. Table 7 evaluates how the approaches prevent attack payloads from entering the aggregation pool, exclude active attackers from subsequent pools, and limit their effects on the final output over the 112 attacked conditions for each dataset. Each rate is computed separately for each condition and then averaged over the conditions with a nonzero denominator. The results show that MiniRep’s main strength is limiting the influence of attack payloads that enter the aggregation pool, rather than always excluding them. Specifically, MiniRep achieves the lowest PayloadHit on all three datasets: 18.22% on GoEmotions, 26.07% on MATH, and 0.05% on HumanEval Pro. It also achieves the lowest CF-Harm on GoEmotions and a near-lowest value on HumanEval Pro. Beta achieves a lower CF-Harm on MATH. These results are consistent with MiniRep’s design: the Reputation-Aware Aggregator adjusts weights using current evidence from the Response Analyzer, so a malicious proposal can have limited influence even when its agent remains in the pool.

Performance under different compositions, expanded. Table 8 expands panel (a) of Table 4 by reporting the strict wins of all methods under all four agent compositions. The results confirm that MiniRep performs most consistently on MATH: it achieves 27, 28, 10, and 19 strict wins under the all-strong, 7S+3W, all-weak, and random compositions, respectively, for a total of 84 wins across the 112 conditions. Its performance on the other datasets depends more strongly on the composition. On GoEmotions, MiniRep performs particularly well in exact-match accuracy under the 7S+3W composition and in F1 under the all-weak composition, but Beta or Single-Metric obtains more strict wins in several other settings. On HumanEval Pro, MiniRep achieves the most strict wins under the 7S+3W, all-weak, and random compositions, while EigenTrust performs better under the all-strong composition. We believe these differences reflect how the available proposal quality changes with the agent composition. Reputation-based aggregation is most useful when the pool contains meaningful differences in agent capability. Its advantage can decrease when another method already assigns high influence to the strongest proposals.

Table 7: Average defense rates of all experiments per dataset.
<table><tr><td>Metric / setting</td><td>UMaj</td><td>URand</td><td>Single</td><td>Babylon</td><td>Eigen</td><td>Beta</td><td>TS</td><td>MiniRep</td></tr><tr><td>GoEmotions</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>PoolExp↓</td><td>100.00</td><td>95.41</td><td>53.72</td><td>46.35</td><td>95.49</td><td>47.43</td><td>53.20</td><td>81.79</td></tr><tr><td>NextExcl↑</td><td>0.00</td><td>4.78</td><td>49.84</td><td>55.26</td><td>4.71</td><td>53.64</td><td>47.93</td><td>19.42</td></tr><tr><td>PayloadHit↓</td><td>27.93</td><td>24.62</td><td>19.55</td><td>22.48</td><td>25.22</td><td>19.83</td><td>20.69</td><td>18.22</td></tr><tr><td>CF-Harm↓</td><td>27.84</td><td>21.02</td><td>9.10</td><td>9.64</td><td>22.67</td><td>7.89</td><td>10.51</td><td>5.82</td></tr><tr><td>MATH</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>PoolExp↓</td><td>100.00</td><td>95.21</td><td>29.48</td><td>30.33</td><td>94.77</td><td>31.23</td><td>55.41</td><td>50.74</td></tr><tr><td>NextExcl↑</td><td>0.00</td><td>5.00</td><td>74.47</td><td>71.83</td><td>5.74</td><td>70.06</td><td>46.68</td><td>52.64</td></tr><tr><td>PayloadHit↓</td><td>57.43</td><td>54.48</td><td>42.04</td><td>52.25</td><td>56.69</td><td>41.10</td><td>41.64</td><td>26.07</td></tr><tr><td>CF-Harm↓</td><td>19.92</td><td>16.65</td><td>6.08</td><td>7.19</td><td>18.40</td><td>5.76</td><td>8.02</td><td>8.92</td></tr><tr><td>HumanEval Pro</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>PoolExp↓</td><td>100.00</td><td>95.95</td><td>34.94</td><td>32.71</td><td>51.31</td><td>33.20</td><td>57.50</td><td>31.70</td></tr><tr><td>NextExcl↑</td><td>0.00</td><td>3.67</td><td>71.34</td><td>69.47</td><td>50.50</td><td>68.46</td><td>44.46</td><td>74.05</td></tr><tr><td>PayloadHit↓</td><td>48.07</td><td>40.42</td><td>25.90</td><td>36.30</td><td>3.38</td><td>27.49</td><td>19.85</td><td>0.05</td></tr><tr><td>CF-Harm↓</td><td>36.06</td><td>29.34</td><td>7.52</td><td>9.84</td><td>2.23</td><td>7.43</td><td>11.33</td><td>2.78</td></tr></table>

Table 8: Strict wins for each agent composition. Bold marks the largest count in each row. Shaded cells are summarized in panel (a) of Table 4.
<table><tr><td>Metric / setting</td><td>UMaj</td><td>URand</td><td>Single</td><td>Babylon</td><td>Eigen</td><td>Beta</td><td>TS</td><td>MiniRep</td></tr><tr><td colspan="9">GoEmotions / exact-match accuracy</td></tr><tr><td>All strong</td><td>0/28</td><td>0/28</td><td>0/28</td><td>0/28</td><td>0/28</td><td>6/28</td><td>0/28</td><td>15/28</td></tr><tr><td>7 strong + 3 weak</td><td>0/28</td><td>0/28</td><td>0/28</td><td>0/28</td><td>0/28</td><td>0/28</td><td>0/28</td><td>25/28</td></tr><tr><td>All weak</td><td>0/28</td><td>0/28</td><td>2/28</td><td>0/28</td><td>0/28</td><td>2/28</td><td>0/28</td><td>18/28</td></tr><tr><td>Random</td><td>0/28</td><td>0/28</td><td>1/28</td><td>3/28</td><td>0/28</td><td>9/28</td><td>0/28</td><td>6/28</td></tr><tr><td colspan="9">GoEmotions / sample F1</td></tr><tr><td>All strong</td><td>0/28</td><td>0/28</td><td>1/28</td><td>3/28</td><td>0/28</td><td>11/28</td><td>1/28</td><td>9/28</td></tr><tr><td>7 strong + 3 weak</td><td>0/28</td><td>0/28</td><td>2/28</td><td>0/28</td><td>0/28</td><td>13/28</td><td>7/28</td><td>3/28</td></tr><tr><td>All weak</td><td>0/28</td><td>0/28</td><td>8/28</td><td>0/28</td><td>0/28</td><td>3/28</td><td>0/28</td><td>17/28</td></tr><tr><td>Random</td><td>0/28</td><td>0/28</td><td>7/28</td><td>1/28</td><td>0/28</td><td>7/28</td><td>6/28</td><td>5/28</td></tr><tr><td colspan="9">MATH / mathematical-equivalence accuracy</td></tr><tr><td>All strong</td><td>0/28</td><td>0/28</td><td>0/28</td><td>0/28</td><td>0/28</td><td>0/28</td><td>0/28</td><td>27/28</td></tr><tr><td>7 strong + 3 weak</td><td>0/28</td><td>0/28</td><td>0/28</td><td>0/28</td><td>0/28</td><td>0/28</td><td>0/28</td><td>28/28</td></tr><tr><td>All weak</td><td>0/28</td><td>0/28</td><td>2/28</td><td>0/28</td><td>0/28</td><td>7/28</td><td>1/28</td><td>10/28</td></tr><tr><td>Random</td><td>0/28</td><td>0/28</td><td>0/28</td><td>0/28</td><td>0/28</td><td>2/28</td><td>1/28</td><td>19/28</td></tr><tr><td colspan="9">HumanEval Pro / Pass@ 1</td></tr><tr><td>All strong</td><td>0/28</td><td>0/28</td><td>2/28</td><td>1/28</td><td>9/28</td><td>0/28</td><td>0/28</td><td>4/28</td></tr><tr><td>7 strong + 3 weak</td><td>0/28</td><td>0/28</td><td>1/28</td><td>0/28</td><td>10/28</td><td>0/28</td><td>1/28</td><td>12/28</td></tr><tr><td>All weak</td><td>0/28</td><td>1/28</td><td>2/28</td><td>1/28</td><td>6/28</td><td>0/28</td><td>1/28</td><td>11/28</td></tr><tr><td>Random</td><td>0/28</td><td>0/28</td><td>2/28</td><td>1/28</td><td>6/28</td><td>0/28</td><td>2/28</td><td>9/28</td></tr></table>

Table 9: Paired stress cases in the 7S+3W composition. Baseline b has the highest attacked score, where ties are resolved as mentioned in Appendix C.3.1. Scores and CF-Harm are percentages; $\Delta$ and $R _ { b }$ are percentage points. Shaded cells are reported in Table 4(b). Bold marks the best value in each row.
<table><tr><td>Metric / setting</td><td>UMaj</td><td>URand</td><td>Single</td><td>Babylon</td><td>Eigen</td><td>Beta</td><td>TS</td><td>MiniRep</td></tr><tr><td colspan="9">GoEmotions / B / STRONG (F1; b = Beta; Rb = +0.4)</td></tr><tr><td>Clean↑</td><td>37.40</td><td>34.50</td><td>37.37</td><td>37.10</td><td>36.10</td><td>38.27</td><td>38.43</td><td>35.00</td></tr><tr><td>Attacked↑</td><td>18.63</td><td>24.33</td><td>35.27</td><td>32.30</td><td>21.90</td><td>35.83</td><td>35.33</td><td>33.00</td></tr><tr><td>∆↓</td><td>18.77</td><td>10.17</td><td>2.10</td><td>4.80</td><td>14.20</td><td>2.43</td><td>3.10</td><td>2.00</td></tr><tr><td>CF-Harm↓</td><td>48.28</td><td>31.03</td><td>6.90</td><td>12.64</td><td>36.78</td><td>8.05</td><td>9.20</td><td>3.45</td></tr><tr><td colspan="9">GoEmotions / B / ARITH (F1; b = Beta; Rb = +4.2)</td></tr><tr><td>Attacked↑</td><td>28.87</td><td>28.20</td><td>29.60</td><td>30.50</td><td>29.17</td><td>34.00</td><td>27.83</td><td>34.90</td></tr><tr><td>△↓</td><td>8.53</td><td>6.30</td><td>7.77</td><td>6.60</td><td>6.93</td><td>4.27</td><td>10.60</td><td>0.10</td></tr><tr><td>CF-Harm↓</td><td>32.91</td><td>24.05</td><td>22.78</td><td>21.52</td><td>24.05</td><td>16.46</td><td>32.91</td><td>5.06</td></tr><tr><td colspan="9">GoEmotions / B / BoUND (F1; b = Single; Rb = +3.8)</td></tr><tr><td>Attacked↑</td><td>26.03</td><td>25.97</td><td>33.73</td><td>30.60</td><td>25.93</td><td>31.57</td><td>27.27</td><td>35.17</td></tr><tr><td>Δ↓</td><td>11.37</td><td>8.53</td><td>3.63</td><td>6.50</td><td>10.17</td><td>6.70</td><td>11.17</td><td>-0.17</td></tr><tr><td>CF-Harm↓</td><td>38.27</td><td>29.63</td><td>18.52</td><td>22.22</td><td>29.63</td><td>20.99</td><td>32.10</td><td>3.70</td></tr><tr><td colspan="9">MATH / R2 / DIV (accuracy; b = Babylon; Rb = +4.0)</td></tr><tr><td>Clean↑</td><td>73.00</td><td>67.00</td><td>66.00</td><td>65.00</td><td>67.00</td><td>71.00</td><td>65.00</td><td>79.00</td></tr><tr><td>Attacked↑</td><td>53.00</td><td>48.00</td><td>53.00</td><td>55.00</td><td>47.00</td><td>54.00</td><td>54.00</td><td>73.00</td></tr><tr><td>∆↓</td><td>20.00</td><td>19.00</td><td>13.00</td><td>10.00</td><td>20.00</td><td>17.00</td><td>11.00</td><td>6.00</td></tr><tr><td>CF-Harm↓</td><td>26.32</td><td>25.00</td><td>15.79</td><td>17.11</td><td>26.32</td><td>21.05</td><td>14.47</td><td>11.84</td></tr><tr><td colspan="9">MATH / R2 / BoUND (accuracy; b = Babylon; Rb = +1.0)</td></tr><tr><td>Attacked↑</td><td>37.00</td><td>38.00</td><td>53.00</td><td>54.00</td><td>37.00</td><td>53.00</td><td>48.00</td><td>69.00</td></tr><tr><td> $\Delta \downarrow$ </td><td>36.00</td><td>29.00 34.94</td><td>13.00</td><td>11.00 18.07</td><td>30.00 37.35</td><td>18.00 22.89</td><td>17.00</td><td>10.00</td></tr><tr><td>CF-Harm↓</td><td>43.37</td><td></td><td>16.87</td><td></td><td></td><td></td><td>20.48</td><td>16.87</td></tr></table>

Performance when capable agents are compromised. Table 9 expands Table 4(b) with the clean score, attacked score, paired loss, and CF-Harm for five cases in the 7S+3W composition. The compromised agents include highly ranked models: the GoEmotions/B setting compromises Qwen-Max, Qwen-Flash, and GLM, while the MATH/R2 setting compromises two Kimi agents and one GLM agent. MiniRep achieves the highest attacked score in four of the five cases and has a positive $R _ { b }$ in all five. On GoEmotions, it exceeds the strongest baseline under Arith and Bound, with paired losses of only 0.10 and −0.17 points. Under Strong, its attacked F1 remains below Beta, although it has a smaller paired loss and lower CF-Harm. On MATH, MiniRep achieves 73% and 69% accuracy under Div and Bound, exceeding Babylon by 18 and 15 percentage points, respectively. These results are consistent with MiniRep using the current proposal to adjust an agent’s influence, so a capable agent cannot rely only on its historical reputation after it begins submitting malicious proposals.

Performance under OnOff and Adapt attacks, expanded. Table 10 expands Table 4(c) by reporting Pass@1, CF-Harm, and PayloadHit for the OnOff and Adapt attacks under all four corruption placements on HumanEval Pro. MiniRep has five strict wins, two ties, and one loss against the strongest baseline in these eight conditions. It strictly outperforms the strongest baseline under all four OnOff placements. Under Adapt, it achieves one strict win and two ties. Among them, R1 is the only exception, where MiniRep obtains

Table 10: Performance under OnOff and Adapt attacks on HumanEval Pro under the 7S+3W composition. Baseline b has the highest attacked Pass@1 in each setting, where ties are resolved as mentioned in Appendix C.3.1. Shaded cells are reported in Table 4(c). “–” denotes undefined PayloadHit. Bold marks the best defined value in each row.
<table><tr><td>Metric / setting</td><td>UMaj</td><td>URand</td><td>Single</td><td>Babylon</td><td>Eigen</td><td>Beta</td><td>TS</td><td>MiniRep</td></tr><tr><td colspan="9">HumanEval Pro / Pass@ 1 (shared clean reference)</td></tr><tr><td>Clean↑</td><td>83.00</td><td>80.00</td><td>79.00</td><td>78.00</td><td>79.00</td><td>79.00</td><td>77.00</td><td>79.00</td></tr><tr><td colspan="9">HumanEval Pro / B / ONOFF (b = Eigen)</td></tr><tr><td>Attacked↑</td><td>45.00</td><td>51.00</td><td>77.00</td><td>75.00</td><td>78.00</td><td>75.00</td><td>51.00</td><td>81.00</td></tr><tr><td>CF-Harm↓</td><td>54.29</td><td>41.43</td><td>5.71</td><td>7.14</td><td>4.29</td><td>8.57</td><td>40.00</td><td>1.43</td></tr><tr><td>PayloadHit↓</td><td>64.29</td><td>52.94</td><td>16.67</td><td>44.44</td><td>14.29</td><td>35.71</td><td>54.29</td><td>0.00</td></tr><tr><td colspan="9">HumanEval Pro / B / ADAPT (b = UMaj)</td></tr><tr><td>Attacked↑</td><td>81.00</td><td>78.00</td><td>80.00</td><td>79.00</td><td>79.00</td><td>78.00</td><td>80.00</td><td>81.00</td></tr><tr><td>CF-Harm↓</td><td>4.08</td><td>4.08</td><td>4.08</td><td>6.12</td><td>6.12</td><td>8.16</td><td>2.04</td><td>2.04</td></tr><tr><td>PayloadHit↓</td><td>10.20</td><td>10.81</td><td>7.14</td><td>10.00</td><td>7.89</td><td>7.89</td><td>2.94</td><td>0.00</td></tr><tr><td colspan="9">HumanEval Pro / R1 / ONOFF (b = Eigen)</td></tr><tr><td>Attacked↑</td><td>46.00</td><td>50.00</td><td>77.00</td><td>75.00</td><td>81.00</td><td>76.00</td><td>64.00</td><td>82.00</td></tr><tr><td>CF-Harm↓</td><td>52.86</td><td>42.86</td><td>5.71</td><td>8.57</td><td>0.00</td><td>7.14</td><td>22.86</td><td>0.00</td></tr><tr><td>PayloadHit↓</td><td>64.29</td><td>52.86</td><td>20.00</td><td>66.67</td><td>0.00</td><td>62.50</td><td>28.57</td><td>0.00</td></tr><tr><td colspan="9">HumanEval Pro / R1 / ADAPT (b = Eigen)</td></tr><tr><td>Attacked↑</td><td>78.00</td><td>75.00</td><td>72.00</td><td>78.00</td><td>82.00</td><td>78.00</td><td>78.00</td><td>79.00</td></tr><tr><td>CF-Harm↓</td><td>13.51</td><td>13.51</td><td>18.92</td><td>5.41</td><td>0.00</td><td>8.11</td><td>5.41</td><td>2.70</td></tr><tr><td>PayloadHit↓</td><td>8.11</td><td>23.08</td><td>13.79</td><td>8.33</td><td>0.00</td><td>8.57</td><td>6.25</td><td>0.00</td></tr><tr><td colspan="9">HumanEval Pro / R2 / ONOFF (b = Eigen)</td></tr><tr><td>Attacked↑</td><td>45.00</td><td>47.00</td><td>78.00</td><td>75.00</td><td>79.00</td><td>75.00</td><td>56.00</td><td>82.00</td></tr><tr><td>CF-Harm↓</td><td>54.29</td><td>47.14</td><td>4.29</td><td>7.14</td><td>2.86</td><td>8.57</td><td>31.43</td><td>1.43</td></tr><tr><td>PayloadHit↓</td><td>64.29</td><td>58.57</td><td>20.00</td><td>50.00</td><td>8.33</td><td>41.67</td><td>50.00</td><td>0.00</td></tr><tr><td colspan="9">HumanEval Pro / R2 / ADAPT (b = Eigen)</td></tr><tr><td>Attacked↑</td><td>80.00</td><td>77.00</td><td>78.00</td><td>77.00</td><td>80.00</td><td>78.00</td><td>77.00</td><td>83.00</td></tr><tr><td>CF-Harm↓</td><td>7.32</td><td>7.32</td><td>9.76</td><td>12.20</td><td>2.44</td><td>9.76</td><td>7.32</td><td>0.00</td></tr><tr><td>PayloadHit↓</td><td>12.20</td><td>15.38</td><td>10.00</td><td>10.81</td><td>5.13</td><td>12.82</td><td>14.29</td><td>0.00</td></tr><tr><td colspan="9">HumanEval Pro / W / ONOFF (b = Babylon)</td></tr><tr><td>Attacked↑</td><td>48.00</td><td>57.00</td><td>80.00</td><td>80.00</td><td>79.00</td><td>80.00</td><td>79.00</td><td>81.00</td></tr><tr><td>CF-Harm↓</td><td>50.00</td><td>32.86</td><td>1.43</td><td>0.00</td><td>1.43</td><td>1.43</td><td>0.00</td><td>0.00</td></tr><tr><td>PayloadHit↓</td><td>61.43</td><td>42.86</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td></td></tr><tr><td colspan="9">HumanEval Pro / W / ADAPT (b = Babylon)</td></tr><tr><td>Attacked↑</td><td>81.00</td><td>79.00</td><td>81.00</td><td>81.00</td><td>80.00</td><td>80.00</td><td>79.00</td><td>81.00</td></tr><tr><td>CF-Harm↓ PayloadHit↓</td><td>6.45 9.68</td><td>3.23 9.09</td><td>0.00</td><td>0.00</td><td>3.23</td><td>3.23 0.00</td><td>0.00 0.00</td><td>0.00 0.00</td></tr><tr><td></td><td></td><td></td><td>0.00</td><td>0.00</td><td>0.00</td><td></td><td></td><td></td></tr></table>

79% Pass@1 and EigenTrust obtains 82%. When PayloadHit is defined, the final output never matches an injected answer for MiniRep in any of these conditions. Under W/OnOff, PayloadHit is undefined because no proposal containing an attack payload enters the aggregation pool. These results are consistent with MiniRep considering both the current proposal and historical reputation, which limits the influence of agents that behave honestly before launching an attack or change which agents actively attack.

Evaluation scope. These fixed-task results assume persistent identities and known clone groups. The 112 conditions are not independent task samples or generation-seed repetitions. Beyond the two held-out families examined in Appendix D.1, transfer to additional attack families, multi-round debate, and white-box attacks against the screening models remains untested here.

## D.1 Generalization to Unseen Attack Types

Experimental setup. We evaluate MiniRep on attack families excluded from feedback-model training to assess whether its task performance extends beyond the corruption patterns seen during training. We reuse the main experiments’ fixed tasks, four agent compositions, four corruption placements, and eight aggregation methods. Within each composition–placement setting, we vary the test attack family while keeping the aggregation configuration, feedback-model checkpoints, and calibrated thresholds unchanged, without retraining. Table 11 compares test performance on the five families represented in training (RAND, STRONG, DIV, ONOFF, and ADAPT) with the held-out ARITH and BOUND families. We retain the main task metrics: GoEmotions exact-match accuracy and sample F1, MATH mathematical-equivalence accuracy, and HumanEval Pro Pass@1.

Table 11: Test performance on the five attack families represented in feedback-model training and the two held-out families. First five averages $4 \times 4 \times 5 = 8 0$ conditions; each held-out family averages $4 \times 4 = 1 6$ Scores are percentages; bold marks the largest mean in each row. These are complete-system comparisons, not feedback-removal ablations.
<table><tr><td>Metric / setting</td><td>UMaj</td><td>URand</td><td>Single</td><td>Babylon</td><td>Eigen</td><td>Beta</td><td>TS</td><td>MiniRep</td></tr><tr><td colspan="9">GoEmotions / exact-match accuracy</td></tr><tr><td>First five</td><td>11.59</td><td>11.59</td><td>19.55</td><td>18.54</td><td>12.91</td><td>19.76</td><td>19.00</td><td>21.33</td></tr><tr><td>ARITH</td><td>9.00</td><td>9.50</td><td>18.44</td><td>17.31</td><td>12.31</td><td>20.00</td><td>15.06</td><td>20.94</td></tr><tr><td>BOUND</td><td>8.00</td><td>9.81</td><td>17.06</td><td>16.38</td><td>10.06</td><td>18.31</td><td>14.44</td><td>20.56</td></tr><tr><td colspan="9">GoEmotions / sample F1</td></tr><tr><td>First five</td><td>28.04</td><td>27.82</td><td>34.72</td><td>34.09</td><td>28.44</td><td>35.06</td><td>34.71</td><td>33.95</td></tr><tr><td>ARITH</td><td>28.33</td><td>27.32</td><td>33.17</td><td>33.29</td><td>29.79</td><td>35.21</td><td>31.32</td><td>33.97</td></tr><tr><td>BOUND</td><td>26.71</td><td>27.89</td><td>32.21</td><td>32.34</td><td>27.70</td><td>33.58</td><td>30.39</td><td>33.87</td></tr><tr><td colspan="9">MATH / mathematical-equivalence accuracy</td></tr><tr><td>First five</td><td>45.83</td><td>45.13</td><td>54.06</td><td>53.03</td><td>43.70</td><td>54.51</td><td>49.01</td><td>62.05</td></tr><tr><td>ARITH</td><td>39.94</td><td>40.63</td><td>53.50</td><td>51.94</td><td>38.75</td><td>53.69</td><td>47.44</td><td>61.00</td></tr><tr><td>BOUND</td><td>37.19</td><td>38.13</td><td>53.13</td><td>50.69</td><td>36.81</td><td>54.31</td><td>47.44</td><td>62.38</td></tr><tr><td colspan="9">HumanEval Pro / Pass@ 1</td></tr><tr><td>First five</td><td>54.79</td><td>58.41</td><td>75.63</td><td>73.61</td><td>77.84</td><td>74.75</td><td>72.16</td><td>78.19</td></tr><tr><td>ARITH</td><td>43.00</td><td>49.25</td><td>73.00</td><td>69.88</td><td>78.06</td><td>72.38</td><td>67.56</td><td>78.25</td></tr><tr><td>BOUND</td><td>45.31</td><td>51.63</td><td>71.69</td><td>67.44</td><td>78.19</td><td>70.94</td><td>68.69</td><td>77.88</td></tr></table>

Results. On MATH, MiniRep reaches 61.00% and 62.38% accuracy under ARITH and BOUND, exceeding the best baseline means by 7.31 and 8.06 percentage points. Its GoEmotions exact-match accuracy is also highest on both held-out families, whereas sample F1 is mixed; HumanEval Pro Pass@1 differs from EigenTrust by only +0.19 and −0.31 points, respectively. Thus the complete system retains useful performance on attack families excluded from feedback-model training, without uniformly improving every task metric.

Feedback diagnostics. We also examine how the learned feedback responds to held-out attacks, since task scores alone do not distinguish proposal-quality estimation from attack identification. Using the same runs and fixed checkpoints as above, we pool the recorded proposal-level counts within each dataset and attack group. Table 12 reports semantic low-quality prediction rates for proposals with and without active payloads, together with recall and false-positive rates for the behavior probe and fused detector.

Table 12: Recorded proposal-level feedback diagnostics (%). Counts are pooled within each dataset and attack group before computing rates; each configuration is counted once. P and N denote proposals with and without an active attack payload. Probe and fused-detector recall/FPR use the active-payload label. The semantic columns measure low-quality predictions: N can include naturally incorrect proposals, so its low-quality rate is not a false-positive rate.
<table><tr><td></td><td colspan="2">Semantic low-quality</td><td colspan="2">Behavior probe</td><td colspan="2">Fused detector</td></tr><tr><td>Dataset / attack</td><td>P</td><td>N</td><td>Recall</td><td>FPR</td><td>Recall</td><td>FPR</td></tr><tr><td>GoEmotions</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>First five</td><td>91.40</td><td>60.03</td><td>23.55</td><td>9.69</td><td>24.77</td><td>12.30</td></tr><tr><td>ARITH</td><td>78.51</td><td>59.39</td><td>21.73</td><td>9.58</td><td>25.18</td><td>12.92</td></tr><tr><td>BOUND</td><td>80.61</td><td>57.99</td><td>21.68</td><td>9.74</td><td>26.34</td><td>13.05</td></tr><tr><td>MATH</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>First five</td><td>98.32</td><td>45.21</td><td>29.94</td><td>1.24</td><td>34.75</td><td>9.42</td></tr><tr><td>ARITH</td><td>97.99</td><td>44.71</td><td>32.25</td><td>1.21</td><td>35.60</td><td>9.82</td></tr><tr><td>BOUND</td><td>97.94</td><td>43.48</td><td>31.40</td><td>1.22</td><td>34.64</td><td>9.20</td></tr><tr><td>HumanEval Pro</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>First five</td><td>31.47</td><td>6.32</td><td>14.43</td><td>1.44</td><td>9.12</td><td>4.98</td></tr><tr><td>ARITH</td><td>27.94</td><td>6.19</td><td>17.94</td><td>1.47</td><td>10.44</td><td>4.79</td></tr><tr><td>BOUND</td><td>31.40</td><td>6.79</td><td>17.47</td><td>1.31</td><td>10.44</td><td>4.99</td></tr></table>

Results. For MATH, the semantic verifier assigns low quality to 97.99% and 97.94% of payload proposals under ARITH and BOUND, while fused-detector recall is 35.60% and 34.64%. For HumanEval Pro, fused recall is only about 10.44% on each held-out family despite end-to-end Pass@1 near 78%. These diagnostics distinguish proposal-quality screening from attack identification; neither they nor the system scores isolate the incremental benefit of learned feedback, which requires a matched feedback-removal ablation.

## D.2 Performance under Different Agent Compositions

Experimental setup. We test how agent composition affects task performance and the relative ranking of aggregation methods, to determine whether the main results depend on a particular mix of strong and weak models. We reuse the main 4 × 4 × 7 experiment grid and vary its first factor across all-strong, 7S+3W, all-weak, and random ten-agent pools. The fixed tasks, method configurations, learned-feedback settings, and task metrics remain as in the main experiments. Each composition is evaluated under the same four corruption-placement rules and seven attack families; all methods share cached proposals within each condition. We average task scores over these 28 conditions for each composition, so Table 13 shows the magnitude of performance differences alongside the strict-win counts in Table 8.

Table 13: Mean attacked task scores (%) under all four agent compositions. Every entry averages the four corruption placements and seven attacks (28 conditions), with the method configuration held fixed. Bold marks the largest mean in each row. Unlike the strict-win counts in Table 8, this table retains score magnitudes.
<table><tr><td>Metric / setting</td><td>UMaj</td><td>URand</td><td>Single</td><td>Babylon</td><td>Eigen</td><td>Beta</td><td>TS</td><td>MiniRep</td></tr><tr><td colspan="9">GoEmotions / exact-match accuracy</td></tr><tr><td>All strong</td><td>13.14</td><td>14.18</td><td>21.46</td><td>21.46</td><td>14.64</td><td>22.71</td><td>21.21</td><td>23.79</td></tr><tr><td>7 strong + 3 weak</td><td>10.32</td><td>10.00</td><td>19.71</td><td>17.82</td><td>11.89</td><td>19.93</td><td>18.54</td><td>23.50</td></tr><tr><td>All weak</td><td>6.89</td><td>7.89</td><td>14.07</td><td>11.93</td><td>8.82</td><td>13.61</td><td>11.61</td><td>15.82</td></tr><tr><td>Random</td><td>12.46</td><td>12.07</td><td>20.89</td><td>21.00</td><td>14.32</td><td>22.11</td><td>19.79</td><td>21.54</td></tr><tr><td colspan="9">GoEmotions / sample F1</td></tr><tr><td>All strong</td><td>29.50</td><td>29.71</td><td>34.70</td><td>35.46</td><td>29.86</td><td>36.51</td><td>35.05</td><td>35.37</td></tr><tr><td>7 strong + 3 weak</td><td>28.79</td><td>28.22</td><td>35.93</td><td>35.08</td><td>28.92</td><td>36.80</td><td>35.73</td><td>34.73</td></tr><tr><td>All weak</td><td>24.47</td><td>24.47</td><td>30.28</td><td>28.60</td><td>25.56</td><td>29.87</td><td>28.58</td><td>31.65</td></tr><tr><td>Random</td><td>28.80</td><td>28.62</td><td>35.65</td><td>35.75</td><td>29.77</td><td>36.31</td><td>35.10</td><td>34.00</td></tr><tr><td colspan="9">MATH / mathematical-equivalence accuracy</td></tr><tr><td>All strong</td><td>58.50</td><td>56.93</td><td>71.75</td><td>70.61</td><td>57.89</td><td>71.71</td><td>63.61</td><td>78.82</td></tr><tr><td>7 strong + 3 weak</td><td>51.14</td><td>48.54</td><td>64.89</td><td>62.36</td><td>47.75</td><td>65.75</td><td>55.61</td><td>76.57</td></tr><tr><td>All weak</td><td>28.61</td><td>31.96</td><td>35.57</td><td>34.21</td><td>27.36</td><td>36.04</td><td>34.11</td><td>35.71</td></tr><tr><td>Random</td><td>36.75</td><td>36.50</td><td>43.18</td><td>42.96</td><td>35.04</td><td>43.96</td><td>40.93</td><td>56.68</td></tr><tr><td colspan="9">HumanEval Pro / Pass@ 1</td></tr><tr><td>All strong</td><td>53.50</td><td>59.54</td><td>77.11</td><td>74.54</td><td>79.93</td><td>77.11</td><td>76.36</td><td>78.82</td></tr><tr><td>7 strong + 3 weak</td><td>53.50</td><td>56.18</td><td>75.39</td><td>72.93</td><td>79.43</td><td>74.39</td><td>71.11</td><td>79.68</td></tr><tr><td>All weak</td><td>50.43</td><td>55.50</td><td>71.68</td><td>68.96</td><td>74.79</td><td>70.68</td><td>66.29</td><td>76.11</td></tr><tr><td>Random</td><td>49.57</td><td>53.32</td><td>74.57</td><td>72.36</td><td>77.54</td><td>73.29</td><td>70.29</td><td>78.00</td></tr></table>

Results. On MATH, MiniRep exceeds the best baseline mean by 7.07, 10.82, and 12.71 percentage points in the all-strong, 7S+3W, and random compositions, respectively. The all-weak composition is the exception: MiniRep scores 35.71%, slightly below Beta’s 36.04%, even though it has the strictest wins (10 versus 7). This difference shows why a larger win count need not imply a larger mean score.

Composition also changes the relative rankings on the other tasks. On GoEmotions, MiniRep leads in exact-match accuracy for all-strong, 7S+3W, and all-weak pools, but Beta leads for random pools; for sample F1, MiniRep leads only in the all-weak composition. On HumanEval Pro, EigenTrust leads in the all-strong composition, whereas MiniRep has the largest mean in the other three, with margins of 0.25–1.32 points. These comparisons measure sensitivity to the evaluated pool compositions rather than isolate any one reputation component or establish significance across independent generation seeds.