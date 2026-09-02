# Towards Effective Structured Context Modeling for Conversational Recommender Systems via Dual-node Monte Carlo Tree Search

Jincheng Zhang<sup>1,4</sup> Chen Huang<sup>1,2,4</sup>\* Wenqiang Lei<sup>1,4</sup> See-Kiong Ng<sup>2</sup> Yang Deng<sup>3</sup>

<sup>1</sup> College of Computer Science, Sichuan University

<sup>2</sup> Institute of Data Science, National University of Singapore

<sup>3</sup> School of Computing and Information Systems, Singapore Management University

<sup>4</sup> Engineering Research Center of Machine Learning and Industry Intelligence,

Ministry of Education, China

{erzmuxin, huangc.scu}@gmail.com

## Abstract

We investigate the role of conversational context modeling in user preference tracking for Conversational Recommendation Systems (CRSs). In this regard, we propose DREAMS, a novel tree-structured context modeling framework that explicitly captures user preference evolution throughout multi-turn interactions. DREAMS introduces two specialized node types to support the two fundamental objectives of CRSs: preference elicitation and preference exploitation. Specifically, elicitation nodes leverage Monte Carlo Tree Search (MCTS) to strategically explore conversational actions and infer latent user preferences, while exploitation nodes employ LLM-based refinement to transform the tracked preference state into structured retrieval queries for recommendation. Extensive experiments on benchmark datasets demonstrate the effectiveness of DREAMS and its design. Our code is available at https: //github.com/SCUNLP/DREAMS.

## 1 Introduction

Conversational Recommendation Systems (CRSs) aim to actively elicit user preferences through multi-turn conversations (i.e., preference elicitation) and leverage the acquired preferences to recommend relevant items (preference exploitation) (Deng et al., 2021; Lei et al., 2020; He et al., 2023; Deng et al., 2025). Achieving these goals requires accurate user preference tracking (i.e., what the user likes and dislikes), as insufficient tracking can result in redundant questions or premature and irrelevant recommendations, thereby degrading system effectiveness. This, in turn, necessitates robust conversational context modeling (Chen et al., 2025a; Wang et al., 2025), since such user preferences are progressively revealed throughout the conversation. Taking Figure 1 for example, the conversation history may implicitly reveal a negative preference toward a specific director. Such implicit signals should be tracked and consistently suppressed in future recommendations involving that director. Therefore, conversation context modeling forms the foundation of preference tracking, which ultimately supports both preference elicitation and preference exploitation.

![](images/25272ebfffed6fe3c73512cf9e5eb352f2c0fb6667821100fa5509c672bdfcfe.jpg)  
Figure 1: The claim ‘not a bigfan ofSpike Lee’s style expresses a negative preference that should be retained across turns. Otherwise, the CRS may continue recommending his films, leading to repeated user rejection.

However, existing methods frequently model conversational context as simple free-form text for user preference tracking, which leads to suboptimal performance (Chen et al., 2024; Huang et al., 2024) due to two limitations. 1) For preference elicitation, these methods directly inject the entire conversation history into LLM prompts to decide whether to ask or recommend (Feng et al., 2023; Fang et al., 2024; Wang et al., 2023a; Qin et al., 2024), resulting in information overload and untargeted questioning (Kemper et al., 2024; Zhang et al., 2023). 2) For preference exploitation, they encode freeform conversational context (as a query) into dense embeddings for item retrieval, where irrelevant conversational content introduces noise and degrades retrieval accuracy (Li et al., 2025b; Ni et al., 2025; Xu et al., 2025). To ease these limitations, recent studies have represented conversations as JSONformatted dialogue states (Kemper et al., 2024) or MCTS (Du et al., 2025; Dao et al., 2024). However, JSON-based methods represent each turn as an isolated state snapshot, overlooking the dependencies among preference states across the conversation. Conversely, MCTS-based methods capture the exploration structure of interactions, but their nodes lack explicit semantic representations of user preferences. What’s more, they only partially structures preference elicitation, while not to jointly structure both elicitation and exploitation under a unified framework. As a result, neither paradigm simultaneously enables effective preference tracking.

To validate the above analysis, we conducted a targeted evaluation of existing CRSs in Section 3.1. The results conclusively demonstrate that existing methods conclusively hinder user preference tracking, diminishing both preference elicitation and exploitation, and thus overall system performance. This is evidenced by their difficulty in identifying and utilizing conversation state for accurately timing questions or recommendations. Furthermore, these CRSs exhibit high recommendation error rates, even when the conversation context contains complete user preferences. These findings emphasize the need for advanced context modeling to effectively track user preferences and suggesting that effective user preference tracking is not merely a matter of structuring dialogue history. It requires the structured state to actively drive both preference elicitation and exploitation.

To this end, we propose DREAMS, a Dual-node conversational RecommEndAtion system using Monte carlo tree Search for effective user preference tracking. In particular, DREAMS explicitly models user preference evolution through a tree-structured conversation state, representing conversation states as Json-structured nodes and state transitions as edges, forming an evolving preference structure throughout the conversation. Importantly, DREAMS introduces two complementary node types. Elicitation Nodes focus on acquiring user preference by strategically exploring alternative conversational actions (e.g., Director-Inquiry, FailureReflection) through Monte Carlo Tree Search (MCTS), enabling the system to infer and refine latent user preferences. Exploitation Nodes focus on recommendation by transforming the tracked preference state into structured retrieval queries, thereby reducing contextual noise and improving item retrieval accuracy. The two node types are coupled through the same evolving preference state: ELNodes update the state through elicitation and feedback, whereas EXNodes use that state for retrieval, after which recommendation feedback is written back for subsequent decisions. As such, DREAMS provides a structured representation of the entire user preference lifecycle, cohesively bridging conversation flow and recommendation queries to optimize the elicitation and exploitation of user information for superior recommendations.

We experimentally demonstrate the effectiveness of DREAMS using the widely-used simulator-based evaluation framework<sup>1</sup> (Wang et al., 2023a; Huang et al., 2024) and two benchmark datasets. The results show that DREAMS achieves superior recommendation accuracy compared to existing methods (+7.43%, on average), exhibiting enhanced performance in both preference elicitation (+9.35%) and exploitation (+4.00%). Further analysis reveals that our structured context modeling not only provides fine-grained and interpretable guidance for conversational planning, but also enables the CRS to accurately associate subtle or implicit contextual signals with the corresponding user preference attributes through structured representations and search over conversational states. In summary, our contributions are as follows:

• We emphasize the importance of structured context modeling for effective preference tracking and systematically analyze the limitations of existing LLM-based CRS methods in this regard.

• We propose DREAMS, a unified framework that jointly models preference elicitation and exploitation through MCTS over structured conversational states.

• Extensive experiments demonstrate the effectiveness of DREAMS and validate the critical role of structured context modeling in conversational recommendation.

## 2 Related Work

User Preference Tracking. A central challenge in conversational recommendation is maintaining an accurate representation of user preferences as they are progressively revealed throughout multi-turn interactions (Liu et al., 2023c; Deng et al., 2025). Existing LLM-based CRSs primarily support this process from two perspectives. For preference elicitation, prior work focuses on improving dialogue decision-making, such as determining when to ask questions versus provide recommendations (Gao et al., 2023; Friedman et al., 2023; Wang et al., 2023a). For preference exploitation, most methods encode conversational history into semantic representations for downstream retrieval and recommendation (Gao et al., 2023; Liu et al., 2023c; Huang et al., 2024; Qin et al., 2024). Our work highlights the importance of structured context modeling for maintaining an evolving preference representation that can simultaneously support both preference elicitation and preference exploitation.

Context Modeling in LLM-based CRS. Most LLM-based CRSs rely on free-form dialogue history as the underlying representation of user preferences, using it for both preference elicitation (Gao et al., 2023; Friedman et al., 2023; Wang et al., 2023a) and preference exploitation (He et al., 2023; Qin et al., 2024). Such unstructured context can obscure preference signals and introduce retrieval noise (Zhang et al., 2023; Kemper et al., 2024; Li et al., 2025b; Huang et al., 2024; Ni et al., 2025), even for long-context LLMs<sup>2</sup> (Liu et al., 2023a; Bai et al., 2023; An et al., 2025). Recent work attempts to alleviate this issue from two perspectives. RA-CRS (Kemper et al., 2024) summarizes dialogue history into JSON-formatted preference states, but does not model preference state evolution during conversational exploration. In contrast, MCTS-based CRSs such as SAPIENT (Du et al., 2024) and T-EPL (Dao et al., 2024) explicitly model exploration through tree search, yet their search nodes are not grounded in structured preference states. Consequently, existing methods either provide structured preference states without structured preference evolution, or model exploration trajectories without structured preference states<sup>3</sup>. Motivated by this gap, our framework performs search directly over structured conversational states to jointly support preference elicitation and exploitation.

## 3 Preliminary Study

We conduct a targeted evaluation to analyze the deficiencies of existing LLM-based CRSs in effective context modeling, leading to challenges in preference elicitation and exploitation. More details are provided in Appendix A.6, A.7.

<table><tr><td rowspan="2">Model</td><td rowspan="2">Accuracy SR↑</td><td colspan="2">Elicitation</td><td rowspan="2">Exploitation PE²↓</td></tr><tr><td>FGE2↓</td><td>CGE2↓</td></tr><tr><td>InterCRS</td><td>0.313</td><td>0.360</td><td>0.570</td><td>0.240</td></tr><tr><td>MACRS</td><td>0.400</td><td>0.187</td><td>0.456</td><td>0.220</td></tr><tr><td>RA-CRS</td><td>0.333</td><td>0.333</td><td>0.461</td><td>0.240</td></tr><tr><td>ChatCRS</td><td>0.440</td><td>0.120</td><td>0.561</td><td>0.200</td></tr><tr><td>PC-CRS</td><td>0.403</td><td>0.247</td><td>0.491</td><td>0.240</td></tr><tr><td>SAPIENT-LLM</td><td>0.380</td><td>0.307</td><td>0.443</td><td>0.233</td></tr><tr><td>T-EPL</td><td>0.327</td><td>0.313</td><td>0.469</td><td>0.253</td></tr></table>

Table 1: Elicitation and exploitation error evaluation.

## 3.1 Evaluation Setup

Evaluation Overview. Adhering to established practices (Wang et al., 2023a,b; Fang et al., 2024; Qin et al., 2024; Huang et al., 2024), we employ LLM-based user simulators to interact with CRSs. The interaction continues until either the simulator accepted a recommendation or the maximum allowed number of turns is reached. Each resulting conversational history, as the context, is subsequently utilized for evaluation purposes. See Appendix A.9 for details.

Datasets & User Simulator. Following prior CRS evaluation protocols (Wang et al., 2023a; Huang et al., 2024; Qin et al., 2024), we use LLM-based user simulators with persona-conditioned preference profiles (e.g., genres, actors, and directors). Experiments are conducted on Redial (Li et al., 2018)<sup>4</sup>. More details are provided in Appendix A.9. Complementary human validation is reported in Section 5.2 and Appendix A.8. More details are provided in Appendix A.9.

Baselines. Our preliminary study incorporates free-form-based LLM-based CRSs, including InterCRS(Wang et al., 2023a), MACRS(Fang et al., 2024), ChatCRS(Li et al., 2025a), PC-CRS(Qin et al., 2024), and RA-CRS(Kemper et al., 2024). Additionally, we add two MCTS-based baselines, SAPIENT (Du et al., 2025) and T-EPL (Dao et al., 2024). See Appendix A.1 for details.

Evaluation Metrics. We introduce specific errorbased metrics to investigate the deficiencies in preference elicitation and exploitation<sup>5</sup>. (1) For preference elicitation, we consider two key metrics: Coarse-Grained Elicitation Error (CGE<sup>2</sup>) and Fine-Grained Elicitation Error (FGE<sup>2</sup>). The former quantifies the proportion of wrong action selection regarding asking question and making recommendations, while the latter quantifies the proportion of ineffective or misaligned followup questions after the system receives specific user feedback about their preferences. (2) To assess preference exploitation, we employ the Preference Exploitation Error (PE<sup>2</sup>), which measures the proportion of instances where the CRS fails to retrieve a relevant item for recommendation despite a fully clarified understanding of user preferences. (3) Furthermore, in accordance with established methodologies (Qin et al., 2024; Fang et al., 2024), we utilize the conventional metrics of Success Rate (SR) to assess the overall recommendation success. Refer to Appendix A.6 for implementation details and Appendix A.8 for reliability analysis regarding LLM-as-judge.

## 3.2 Experimental Findings

Table 1 shows that off-the-shelf CRS still suffer from persistent elicitation and exploitation errors, highlighting the need for more faithful and finegrained context modeling. In preference elicitation, coarse-grained action selection remains a major weakness: InterCRS reaches a high $C G E ^ { 2 }$ of 0.570, indicating difficulty in deciding when to ask questions rather than recommend items. Structured representations alone are insufficient, as RA-CRS still reports high $F G E ^ { 2 }$ and $C G E ^ { 2 }$ values of 0.333 and 0.461 despite using JSON-format contexts. Likewise, MCTS-based methods do not guarantee reliable planning, with SAPIENT-LLM and T-EPL obtaining $C G E ^ { 2 }$ values of 0.443 and 0.469. Fine-grained elicitation is also unreliable, as reflected by InterCRS’s $F G E ^ { 2 }$ of 0.360. For preference exploitation, InterCRS further shows a high $P E ^ { 2 }$ of 0.240, suggesting that explicitly stated preferences are not effectively converted into retrieval cues. Overall, these results indicate that neither JSON-style representation nor generic MCTS planning is sufficient; CRS requires context modeling that jointly supports state tracking, action selection, and preference-grounded retrieval.

## 4 DREAMS

Following prior work (Du et al., 2025; Yu et al., 2023), we formulate the CRS interaction as a Markov Decision Process (MDP) ⟨S, A, R, P, γ⟩, where the state space S represents dialogue context and preference information, and the action space A contains system actions such as asking clarification questions, reflecting on failed recommendations, and recommending items.

## 4.1 Overview of DREAMS

DREAMS Formalization. As shown in Figure 2, DREAMS formulates user preference tracking as search over structured conversational states, where nodes encode evolving preference states and edges represent state transitions. Unlike JSON-style trackers<sup>6</sup>, which mainly use structured states as turn-independent static memory, DREAMS actively searches over these states for decision-making. It also differs from MCTS-based CRSs such as SAPIENT and T-EPL, whose search nodes are not structured and reusable preference states for open-ended dialogue. Here, elicitation and exploitation denote two functions of a CRS rather than the exploration–exploitation trade-off in stan dard reinforcement learning. Different from prior work that mainly focuses on either elicitation or exploitation, DREAMS contains two node types to jointly optimizing both preference elicitation and exploitation. Given the latest user utterance and the current structured preference state, an ELNode updates the state and searches over actions that determine whether and what the system should ask, whether it should reflect on a failed recommendation, or whether it should proceed to recommenda tion. When ItemRec is selected, the corresponding EXNode consumes the same evolving preference state and converts it into retrieval-oriented query representations. The recommendation outcome and subsequent user feedback are then incorporated into the next state update. In particular, a rejection can reactivate an ELNode to correct or further elicit the user preferences before another recommendation is made. Thus, ELNodes and EXNodes are not independent modules; they factorize a unified online policy that jointly determines when and what to ask and how to recommend.

Algorithm 1 summarizes the full procedure. We further design a hierarchical and verifiable reward to jointly capture elicitation progress and recommendation success. Detailed formulations are provided in Appendix A.4.

![](images/1f728bef862ca2323ba9f4645a3bacb4b8e0cf34fc318e9bc3749d7e5e70c829.jpg)  
Figure 2: Overview of DREAMS’s structured modeling and unified elicitation & exploitation. It incorporates ELNodes to explicitly capture estimated user preferences, and EXNodes to enable structured query refinement.

## 4.2 ELNode for Preference Elicitation

ELNode supports preference elicitation by searching over possible next actions before the system responds. Its action space covers three purposes: preference elicitation, failure reflection, and transition to exploitation<sup>7</sup>. Each ELNode maintains a JSON-compatible structured preference state that makes positive, negative, missing, and corrected preferences explicit and machine-readable. An LLM parses free-form user utterances into structured key-value updates; for example, “I am not a big fan of Spike Lee’s style” is mapped to <director>: dislike Spike Lee. The updated state is then used as the basis for MCTS. Here, the core operations — Selection, Expansion, Simulation, and Backpropagation — are implemented as follows.

Selection. Given state $s _ { i } ,$ DREAMS employs the Upper Confidence bounds applied to Trees algorithm<sup>8</sup> (Kocsis and Szepesvári, 2006) to select the best action (branch) $a ^ { * }$ , where $A ( s _ { i } )$ is the set of child nodes of $s _ { i } ) , R ( s _ { i } , a _ { i } ^ { \prime } )$ is the total reward accumulated from simulations that passed through all the child nodes of $s _ { i }$ after taking the action $a _ { i } ^ { \prime } ,$ $N ( s _ { i } )$ is the visit count for node $s _ { i } , \ f ( s _ { i } , a _ { i } )$ is the function that returns the child node of $s _ { i }$ after applying action $a _ { i }$ , and c is an exploration constant.

$$
\operatorname* { m a x } _ { a _ { i } ^ { \prime } \in A ( s _ { i } ) } \left( { \frac { R ( s _ { i } , a _ { i } ^ { \prime } ) } { N ( f ( s _ { i } , a _ { i } ^ { \prime } ) ) } } + c \cdot { \sqrt { { \frac { 2 \log N ( s _ { i } ) } { N ( f ( s _ { i } , a _ { i } ^ { \prime } ) ) } } } } \right)\tag{1}
$$

Expansion. Upon reaching a leaf ELNode, the tree expands by attaching new child nodes representing candidate elicitation actions. To efficiently prune the search space and prioritize promising actions, we leverage an LLM to generate prior probabilities M over the available actions. This LLM guidance ensures that only actions with a positive predicted score are expanded. To reduce the variance of a single LLM call, we sample the LLM m times and aggregate the resulting priors.

Simulation. To estimate the long-term effectiveness of an expanded action, DREAMS rolls out the dialogue based on a probability-guided policy provided by the LLM until a terminal state or a predefined depth limit is reached. The simulator is initialized with the preference state stored in the current ELNode, and subsequent actions are sampled from the LLM-guided policy. Each transition receives the hierarchical reward described in Appendix A.4, discounted by γ.

Back-propagation. After each rollout, DREAMS back-propagates the discounted cumulative reward to all nodes on the simulated path and updates their visit counts and expected rewards:

$$
R ( s _ { i } , a _ { i } ) = { \frac { 1 } { K } } \sum _ { j = 1 } ^ { K } \left( \sum _ { s ^ { \prime } \in S ^ { j } , a ^ { \prime } \in A ^ { j } } \gamma \cdot r ( s ^ { \prime } , a ^ { \prime } ) \right)\tag{2}
$$

The reward $r ( \cdot )$ consists of three parts: an attitude reward for successful recommendations, an information acquisition reward for learning new preference details, and a turn penalty for conversation efficiency<sup>9</sup>. After a fixed number of MCTS iterations, DREAMS selects the final action by combining the tree value with the LLM prior:

$$
\operatorname* { m a x } _ { a } \left( W _ { t r e e } \cdot \frac { N ( f ( s _ { 0 } , a ) ) } { \sum N ( f ( s _ { 0 } ) ) } + W _ { L L M } \cdot \mathcal { M } ( s _ { 0 } , a ) \right)\tag{3}
$$

where $s _ { 0 }$ is the root state, and $W _ { \mathrm { t r e e } }$ and $W _ { \mathrm { L L M } }$ control the contributions of search statistics and LLM priors.

## 4.3 EXNode for Preference Exploitation

When ItemRec isselected, it expands an EXNode to convert the tracked preference state into a retrieval-friendly query with following two steps: Refinement Mechanism. The EXNode takes the current dialogue history, the JSON-style preference state, and the raw user query as input. It first summarizes the cumulative preference state $s _ { t } ^ { R }$ , which differs from the ELNode state $s _ { t } \colon$ $s _ { t }$ focuses on the latest turn-level update, while $s _ { t } ^ { R }$ aggregates user preferences across the whole dialogue. Given $s _ { t } ^ { R } ,$ , the LLM proposes refinement actions $a _ { i } ^ { R }$ , such as removing redundant context, reordering preference attributes, and converting informal expressions into machine-readable constraints. For example, a negative preference such as “dislike Spike Lee’s style” can be converted into <director>: dislike Spike Lee. Applying the i-th refinement action produces a refined query state:

$$
E X N _ { 1 } ^ { i } \gets a _ { i } ^ { R } ( E X N _ { 0 } )\tag{4}
$$

Evaluation and Selection. DREAMS then retrieves items with each refined query and scores the result by matching the top retrieved items against the preferred attributes in $s _ { t } ^ { R }$

$$
R ^ { R } ( E X N _ { 1 } ^ { i } ) = \operatorname { s c o r e } ( \operatorname { r e t r i e v e } ( E X N _ { 1 } ^ { i } ) )\tag{5}
$$

The query with the highest retrieval score is used for final recommendation generation. In this way, EXNode connects preference tracking to preference exploitation: the system doesn’t retrieve from noisy dialogue history directly, but from a structured query derived from evolving preference state.

![](images/3cec1ee561a14885ab420a31272ef57a420758f6a20409a88244863fcf5177f9.jpg)  
Figure 3: Results on human interaction and evaluation.

## 5 Experiments

## 5.1 Experimental Setup

Baselines. Beyond LLM-based CRSs in Section 3.1, we follow previous studies (Qin et al., 2024; Wang et al., 2023a) and involve two pre-trained language model based CRSs (BARCOR (Wang et al., 2022a) and UniCRS (Wang et al., 2022b)) and two MCTS-based CRS baselines (SAPIENT (Du et al., 2025) and T-EPL (Dao et al., 2024)). See Appendix A.1 for details.

Datasets & User Simulator. In addition to the datasets and user simulators in Section 3.1, we follow current common practice in the field (Qin et al., 2024; Huang et al., 2024; Chen et al., 2025b) and incorporate OpendialKG (Moon et al., 2019), a multi-domain benchmark dataset.

Evaluation Metrics. In addition to the metrics in Section 3.1, we incorporate a human evaluation, which leverages human judgment to further assess the performance of CRSs, specifically focusing on preference elicitation and exploitation.

## 5.2 Main Results

DREAMS consistently outperforms existing CRS baselines. Table 2 shows that DREAMS achieves the best performance across all datasets and metrics. Relative to ChatCRS, the strongest non-MCTS baseline, DREAMS improves average R@1 and SR by 8.57% and 9.07%, respectively. Furthermore, it substantially surpasses the MCTS-based methods SAPIENT-LLM and T-EPL in SR. This suggests that our improvement stems not merely from search, but from grounding search in structured preference states and retrieval refinement.

DREAMS exhibits superior performance in preference state tracking, demonstrating enhanced both elicitation and exploitation of user preferences. As shown in Table 3, DREAMS consistently yields lower elicitation and exploitation errors than all baselines, reflected by its superior $\mathrm { C G E ^ { 2 } }$ , FGE<sup>2</sup>, and $\mathrm { P E ^ { 2 } }$ scores. The lower $\mathrm { C G E ^ { 2 } }$ indicates fewer coarse-grained decision errors in whether to continue elicitation or proceed to recommendation, while the lower $\mathrm { F G E ^ { 2 } }$ demonstrates stronger finegrained preference modeling. These gains further lead to lower exploitation errors $( \mathrm { P E ^ { 2 } } )$ , suggesting that the inferred preference state can be more effectively transformed into retrieval queries. A more specific conclusion is that DREAMS does not benefit from MCTS in isolation. Instead, its gains come from using tree search over a structured preference state, so the system can decide when to clarify, when to recommend, and how to refine the retrieval query from the tracked preferences.

<table><tr><td rowspan="2">Model</td><td colspan="4">Redial (Movie)</td><td colspan="4">OpendialKG (Movie)</td><td colspan="4">OpendialKG (Book)</td></tr><tr><td>R@1(↑)</td><td>R@5(↑)</td><td>R@10(↑)</td><td>SR(↑)</td><td>R@1(↑)</td><td>R@5(↑)</td><td>R@10(↑)</td><td>SR(↑)</td><td>R@1(↑)</td><td>R@5(↑)</td><td>R@10(↑)</td><td>SR(↑)</td></tr><tr><td colspan="10">PLM-based</td><td></td><td></td><td></td></tr><tr><td>BARCOR (Wang et al., 2022a) UniCRS (Wang et al., 2022b)</td><td>0.193 0.147</td><td>0.467 0.373</td><td>0.593 0.520</td><td>0.107 0.133</td><td>0.094 0.233</td><td>0.383 0.428</td><td>0.494 0.583</td><td>0.211 0.250</td><td>0.106 0.225</td><td>0.375 0.406</td><td>0.456 0.563</td><td>0.219 0.231</td></tr><tr><td colspan="10">LLM-based</td><td></td><td></td></tr><tr><td>InterCRS (Wang et al., 2023a)</td><td>0.213</td><td>0.370</td><td>0.500</td><td>0.313</td><td>0.439</td><td>0.650</td><td>0.783</td><td>0.494</td><td>0.356</td><td>0.469</td><td>0.594</td><td>0.394</td></tr><tr><td>MACRS (Fang et al., 2024)</td><td>0.310</td><td>0.410</td><td>0.560</td><td>0.400</td><td>0.506</td><td>0.672</td><td>0.800</td><td>0.522</td><td>0.506</td><td>0.638</td><td>0.681</td><td>0.544</td></tr><tr><td>ChatCRS (Li et al., 2025a)</td><td>0.367</td><td>0.473</td><td>0.573</td><td>0.440</td><td>0.539</td><td>0.650</td><td>0.772</td><td>0.550</td><td>0.494</td><td>0.600</td><td>0.675</td><td>0.531</td></tr><tr><td>PC-CRS (Qin et al., 2024)</td><td>0.307</td><td>0.427</td><td>0.493</td><td>0.407</td><td>0.556</td><td>0.656</td><td>0.772</td><td>0.567</td><td>0.519</td><td>0.625</td><td>0.694</td><td>0.563</td></tr><tr><td>RA-CRS (Kemper et al., 2024)</td><td>0.273</td><td>0.340</td><td>0.387</td><td>0.333</td><td>0.422</td><td>0.606</td><td>0.733</td><td>0.433</td><td>0.494</td><td>0.575</td><td>0.644</td><td>0.519</td></tr><tr><td>SAPIENT-LLM (Du et al., 2025)</td><td>0.287</td><td>0.387</td><td>0.427</td><td>0.380</td><td>0.433</td><td>0.644</td><td>0.661</td><td>0.450</td><td>0.488</td><td>0.538</td><td>0.585</td><td>0.506</td></tr><tr><td>T-EPL (Dao et al., 2024)</td><td>0.293</td><td>0.367</td><td>0.387</td><td>0.327</td><td>0.416</td><td>0.611</td><td>0.639</td><td>0.427</td><td>0.475</td><td>0.550</td><td>0.619</td><td>0.494</td></tr><tr><td>DREAMS (Ours)</td><td>0.507</td><td>0.633</td><td>0.747</td><td>0.560</td><td>0.600</td><td>0.728</td><td>0.844</td><td>0.639</td><td>0.550</td><td>0.656</td><td>0.719</td><td>0.594</td></tr></table>

Table 2: Overall performance of different methods.
<table><tr><td rowspan="2">Model</td><td colspan="3">Redial (Movie)</td><td colspan="6">OpendialKG (Movie)</td><td colspan="3">OpendialKG (Book)</td></tr><tr><td>SR↑</td><td> $\mathrm { F G E ^ { 2 } \downarrow }$ </td><td> $\mathrm { C G E ^ { 2 } \downarrow }$ </td><td> $\mathrm { P E ^ { 2 } \downarrow }$ </td><td> $S R \uparrow$ </td><td> $\mathrm { \bar { F } G E ^ { 2 } } \downarrow$ </td><td> $\mathrm { C G E ^ { 2 } \downarrow }$ </td><td> $\mathrm { P E ^ { 2 } \downarrow }$ </td><td>SR↑</td><td>FGE2↓</td><td>CGE2↓</td><td>PE²↓</td></tr><tr><td>MACRS (Fang et al., 2024)</td><td>0.400</td><td>0.187</td><td>0.456</td><td>0.220</td><td>0.522</td><td>0.150</td><td>0.431</td><td>0.106</td><td>0.544</td><td>0.156</td><td>0.498</td><td>0.088</td></tr><tr><td>ChatCRS (Li et al., 2025a)</td><td>0.440</td><td>0.120</td><td>0.561</td><td>0.200</td><td>0.550</td><td>0.133</td><td>0.291</td><td>0.117</td><td>0.531</td><td>0.169</td><td>0.455</td><td>0.100</td></tr><tr><td>RA-CRS (Kemper et al., 2024)</td><td>0.333</td><td>0.333</td><td>0.461</td><td>0.240</td><td>0.433</td><td>0.161</td><td>0.398</td><td>0.122</td><td>0.519</td><td>0.188</td><td>0.469</td><td>0.106</td></tr><tr><td>SAPIENT-LLM (Du et al., 2025)</td><td>0.380</td><td>0.307</td><td>0.443</td><td>0.233</td><td>0.450</td><td>0.150</td><td>0.348</td><td>0.139</td><td>0.506</td><td>0.175</td><td>0.458</td><td>0.119</td></tr><tr><td>T-EPL (Dao et al., 2024)</td><td>0.327</td><td>0.313</td><td>0.469</td><td>0.253</td><td>0.427</td><td>0.144</td><td>0.332</td><td>0.133</td><td>0.494</td><td>0.194</td><td>0.471</td><td>0.131</td></tr><tr><td>DREAMS</td><td>0.560</td><td>0.100</td><td>0.289</td><td>0.160</td><td>0.639</td><td>0.117</td><td>0.304</td><td>0.078</td><td>0.594</td><td>0.138</td><td>0.408</td><td>0.069</td></tr></table>

Table 3: Elicitation and exploitation evaluation of DREAMS and leading baselines.

DREAMS enjoys enhanced practical utility in real-world human interactions. To evaluate practical utility, we conduct a human study involving 60 participants, who interact with DREAMS and three representative baselines (MACRS, ChatCRS, and PC-CRS) under predefined preference settings. As shown in Figure 3, DREAMS consistently achieves the best performance across all metrics, including SR, Recall@1, $\mathrm { C G E ^ { 2 } }$ $\mathrm { F G E ^ { 2 } }$ , and $\mathrm { P E ^ { 2 } }$ , indicating superior preference elicitation, preference exploitation, and recommendation effectiveness in realistic conversational settings. To validate the reliability of our automatic evaluator, we randomly sample 50 dialogues generated by DREAMS on Redial and ask 10 human annotators to label $\mathrm { P E ^ { 2 } , F G E ^ { 2 } }$ , and $\mathrm { C G E ^ { 2 } }$ following the same criteria. The resulting Krippendorff’s α scores are 0.74 and 0.68 for Elicitation Error and Exploitation Error, respectively, indicating substantial agreement and supporting the reliability of the automatic evaluation, consistent with prior findings (Liu et al., 2023b).

DREAMS generalizes across LLM backbones. Table 11 shows that DREAMS maintains clear advantages on both Gemini-2.5-Flash and GPT-4omini. Despite Gemini’s strong long-context capability, the free-form method InterCRS still suffers from high elicitation errors, whereas DREAMS consistently achieves better preference tracking and recommendation performance. These results suggest that the limitation of free-form context modeling is not primarily due to context length. Instead, effective conversational recommendation requires an explicit preference state that supports preference elicitation and exploitation.

## 5.3 Recommendation Performance Analysis

We analyze the contribution of different components through the following variants:

• w/o EXNode. Directly retrieves from free-form dialogue history without query refinement.

• w/o ELNode. Replaces MCTS-based state transition with direct LLM action selection.

• w/o ELNode & Json. Further removes structured preference states and tracks preferences using free-form context.

<table><tr><td rowspan="2">Model</td><td colspan="5">Redial (Movie)</td><td colspan="5">OpendialKG (Movie)</td><td colspan="5">OpendialKG (Book)</td></tr><tr><td>R@1(↑)</td><td>SR(↑)</td><td>FGE2 (↓)</td><td>CGE2(↓)</td><td>PE2 (↓)</td><td>R@1(↑)</td><td>SR(↑) FGE2 (↓)</td><td></td><td>CGE2(↓)</td><td>PE2 (↓)</td><td>R@1(↑)</td><td>SR(↑)</td><td>FGE2 (↓)</td><td>CGE2(↓)</td><td>PE2 (↓)</td></tr><tr><td>DREAMS</td><td>0.507</td><td>0.560</td><td>0.100</td><td>0.289</td><td>0.160</td><td>0.600</td><td>0.639</td><td>0.117</td><td>0.304</td><td>0.078</td><td>0.550</td><td>0.594</td><td>0.138</td><td>0.408</td><td>0.069</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>Structured Context Modeling</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>w/o EXNode</td><td>0.407</td><td>0.487</td><td>0.133</td><td>0.311</td><td>0.193</td><td>0.517</td><td>0.572</td><td>0.139</td><td>0.326</td><td>0.117</td><td>0.494</td><td>0.556</td><td>0.144</td><td>0.422</td><td>0.100</td></tr><tr><td>w/o ELNode w/o ELNode &amp; Json</td><td>0.393 0.287</td><td>0.427</td><td>0.173</td><td>0.417</td><td>0.173</td><td>0.500</td><td>0.550</td><td>0.156</td><td>0.378</td><td>0.094</td><td>0.444</td><td>0.456</td><td>0.188</td><td>0.490</td><td>0.063</td></tr><tr><td></td><td>0.347</td><td></td><td>0.340</td><td>0.539</td><td>0.233</td><td>0.450</td><td>0.478</td><td>0.406</td><td>0.455</td><td>0.111</td><td>0.388</td><td>0.456</td><td>0.250</td><td>0.523</td><td>0.075</td></tr><tr><td colspan="10">Structured Tracking</td><td colspan="7"></td></tr><tr><td>w/ Json Only</td><td>0.327</td><td>0.360</td><td>0.200</td><td></td><td></td><td></td><td></td><td>0.411</td><td>0.384</td><td>0.128</td><td>0.394</td><td>0.475</td><td>0.213</td><td></td><td>0.081</td></tr><tr><td></td><td></td><td></td><td></td><td>0.455</td><td>0.240</td><td>0.439</td><td>0.472</td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.473</td><td></td></tr></table>

Table 4: Ablation studies on benchmark datasets.

• w/ Json Only. Uses JSON-style preference states, but performs action selection and retrieval without structured search.

Table 4 disentangles the effects of structured preference states, state-transition search, and structured retrieval refinement. Removing both ELNode and JSON causes the largest degradation, significantly increasing FGE<sup>2</sup>and CGE<sup>2</sup>, showing that explicit structured states are essential for effective preference tracking. Removing ELNode while preserving JSON also degrades performance, indicating that preference states must be actively updated and utilized during interaction. Replacing MCTS with direct LLM reasoning further reduces SR and increases FGE<sup>2</sup>, suggesting that structured states should serve as actionable decision states rather than passive memory. Removing EXNode mainly decreases Recall@1 and SR while increasing PE<sup>2</sup>, demonstrating the importance of exploitation-side query refinement. Overall, the results show that structured preference states, state-transition search, and structured retrieval refinement are all critical to unified preference elicitation and exploitation.

## 5.4 Preference Elicitation Analysis

Structured conversational states improve preference elicitation by transforming noisy dialogue history into stable and decision-relevant preference representations. Table 4 supports this conclusion through the ELNode and JSON-related ablations. Removing ELNode weakens the model’s ability to maintain evolving preference states, increasing both FGE<sup>2</sup>and CGE<sup>2</sup>. Further removing JSON-style states leads to even larger degradation. These results show that explicit structured state modeling is essential for effective preference elicitation, enabling DREAMS to outperform MCTSbased baselines such as SAPIENT-LLM and T-EPL. Case study (See Appendix B.2) illustrates how DREAMS differs from baselines facing noisy dialogue history.

JSON-format state tracking records preferences explicitly, but structured search is needed to turn the recorded state into better decision making. Both RA-CRS and the w/Json Only variant maintain structured preference states, yet neither operationalizes them through structured search. Table 4 isolates this effect: although the w/Json Only variant preserves the same structured states and action/query interfaces, replacing MCTS with direct LLM reasoning substantially lowers SR and increases FGE<sup>2</sup>. This demonstrates that effective preference tracking requires not only structured states, but also search over evolving conversational states for lookahead decision-making. A mechanism-level case study is provided in Appendix B.2.

## 5.5 Preference Exploitation Analysis

By leveraging the refinement mechanism, the EXNode progressively refines the retrieval queries, steering them closer to the ground-truth item description. As shown in Figure 4, EXNode progressively refines retrieval queries toward the ground-truth item description, yielding consistently higher cosine similarity than original queries. In contrast, randomly paraphrased LLM queries are unstable and can even underperform the original query, highlighting the sensitivity of retrieval to contextual noise (Huang et al., 2024). We further observe that EXNode tends to generate more structured JSON-style retrieval queries, which better align with the structured knowledge bases in REDIAL and OpenDialKG. Therefore, EXNode improves exploitation through both semantic refinement and structured retrieval representations.

## 5.6 Efficiency Analysis

Although a common concern with tree-based planning is latency, Table 10 shows that vanilla DREAMS already achieves efficiency comparable to existing CRSs through our parallel function calling adaptation. To further improve deployment efficiency, we introduce DREAMS(EA), an

![](images/9b8a8afaf23b598d93bdd17266d97672530d1bd18fad5b3bbafd67f7a5e34524.jpg)  
Figure 4: Comparison of query refinement methods via cosine similarity to ground-truth embeddings.

Experience-Augmented variant inspired by prior experience-based reasoning methods (Gu et al., 2025; Ren et al., 2025). Specifically, we store MCTS trajectories as structured experience knowledge and retrieve structurally similar successful/- failed cases from an Experience Knowledge Base (EKB) to warm-start decision making, thereby bypassing expensive online tree search. As shown in Table 10, DREAMS(EA) reduces inference latency to 9s while maintaining competitive recommendation performance (Recall@1=0.467). More implementation and efficiency details are provided in Appendix B.1.

## 6 Conclusion

In this paper, we highlight the critical role of structured context modeling in conversational recommender systems and propose DREAMS, a dualnode MCTS framework that jointly optimizes preference elicitation and preference exploitation. Extensive experiments demonstrate that effective conversational recommendation requires not only structured representations, but also structured search over evolving conversational states. We hope this work encourages future CRS research to treat structured context as an important active decision mechanism.

## Limitations

To ensure a fair comparison, we adhered to established protocols (Qin et al., 2024; Li et al., 2025a) by limiting all methods to the GPT-4o-mini and Gemini-2.5-flash as their backbone. Due to constraints in computational resources and budget, we do not extend our evaluation to other SOTA LLMs (e.g., Claude Opus). Additionally, DREAMS may inherit model-level limitations of its LLM backbone, including social biases and incomplete domain knowledge. The scope of this work is not to remove these intrinsic limitations, but to improve preference tracking and recommendation through a model-agnostic structured-state framework. We encourage future research to investigate the influence of diverse LLM backbones on CRS performance.

More broadly, the evaluation of modern CRSs remains challenging because scalable protocols often depend on LLM-based judges, either as offline evaluators or as user simulators. Although such protocols enable controlled and reproducible comparisons, they may inherit biases from the evaluator model, prompt design, and predefined user preference ontology. Finally, generalization to domains with different item structures, knowledge requirements, or decision costs remains untested. Future work should develop more reliable and diverse human-grounded benchmarks, behaviorally validated user simulators, and evaluator-independent protocols for conversational recommendation.

## Acknowledgments

This research is supported by the National Research Foundation, Singapore under its National Large Language Models Funding Initiative (AISG Award No: AISG-NMLP-2024-002), the Singapore Ministry of Education (MOE) Academic Research Fund (AcRF) Tier 1 grant (Proposal ID: 24-SIS-SMU-002), and National Natural Science Foundation of China (No. 62272330 and No.U24A20328). Yang Deng is supported by the Lee Kong Chian Fellowship awarded by Singapore Management University. Any opinions, findings and conclusions or recommendations expressed in this material are those of the author(s) and do not reflect the views of National Research Foundation, Singapore.

## References

Guojia An, Jie Zou, Jiwei Wei, Chaoning Zhang, Fuming Sun, and Yang Yang. 2025. Beyond whole dialogue modeling: Contextual disentanglement for conversational recommendation. In Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 31–41.

Yushi Bai, Xin Lv, Jiajie Zhang, Hongchang Lyu, Jiankai Tang, Zhidian Huang, Zhengxiao Du, Xiao Liu, Aohan Zeng, Lei Hou, et al. 2023. Longbench: A bilingual, multitask benchmark for long context understanding. arXiv preprint arXiv:2308.14508.

Bo Chen, Xinyi Dai, Huifeng Guo, Wei Guo, Weiwen Liu, Yong Liu, Jiarui Qin, Ruiming Tang, Yichao Wang, Chuhan Wu, et al. 2024. All roads lead to rome: Unveiling the trajectory of recommender systems across the llm era. arXiv preprint arXiv:2407.10081.

Luyu Chen, Quanyu Dai, Zeyu Zhang, Xueyang Feng, Mingyu Zhang, Pengcheng Tang, Xu Chen, Yue Zhu, and Zhenhua Dong. 2025a. Recusersim: A realistic and diverse user simulator for evaluating conversational recommender systems. In Companion Proceedings of the ACM on Web Conference 2025, pages 133–142.

Nuo Chen, Quanyu Dai, Xiaoyu Dong, Xiao-Ming Wu, and Zhenhua Dong. 2025b. Evaluating conversational recommender systems with large language models: A user-centric evaluation framework. arXiv preprint arXiv:2501.09493.

Huy Quang Dao, Yang Deng, Khanh-Huyen Bui, Dung D. Le, and Lizi Liao. 2024. Experience as source for anticipation and planning: Experiential policy learning for target-driven recommendation dialogues. In Findings of the Association for Computational Linguistics: EMNLP 2024, pages 14179– 14198, Miami, Florida, USA. Association for Computational Linguistics.

Yang Deng, Yaliang Li, Fei Sun, Bolin Ding, and Wai Lam. 2021. Unified conversational recommendation policy learning via graph-based reinforcement learning. In Proceedings ofthe 44th International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 1431–1441.

Yang Deng, Lizi Liao, Wenqiang Lei, Grace Hui Yang, Wai Lam, and Tat-Seng Chua. 2025. Proactive conversational ai: A comprehensive survey of advancements and opportunities. ACM Transactions on Information Systems, 43(3):1–45.

Hanwen Du, Bo Peng, and Xia Ning. 2024. Sapient: Mastering multi-turn conversational recommendation with strategic planning and monte carlo tree search. arXiv preprint arXiv:2410.09580.

Hanwen Du, Bo Peng, and Xia Ning. 2025. SAPIENT: Mastering multi-turn conversational recommendation with strategic planning and Monte Carlo tree search. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 2629–2648, Albuquerque, New Mexico. Association for Computational Linguistics.

Jiabao Fang, Shen Gao, Pengjie Ren, Xiuying Chen, Suzan Verberne, and Zhaochun Ren. 2024. A multiagent conversational recommender system. arXiv preprint arXiv:2402.01135.

Yue Feng, Shuchang Liu, Zhenghai Xue, Qingpeng Cai, Lantao Hu, Peng Jiang, Kun Gai, and Fei Sun. 2023. A large language model enhanced conversational recommender system. arXiv preprint arXiv:2308.06212.

Luke Friedman, Sameer Ahuja, David Allen, Zhenning Tan, Hakim Sidahmed, Changbo Long, Jun Xie, Gabriel Schubiner, Ajay Patel, Harsh Lara, et al.

2023. Leveraging large language models in conversational recommender systems. arXiv preprint arXiv:2305.07961.

Yunfan Gao, Tao Sheng, Youlin Xiang, Yun Xiong, Haofen Wang, and Jiawei Zhang. 2023. Chatrec: Towards interactive and explainable llmsaugmented recommender system. arXiv preprint arXiv:2303.14524.

Jiawei Gu, Ziting Xian, Yuanzhen Xie, Ye Liu, Enjie Liu, Ruichao Zhong, Mochi Gao, Yunzhi Tan, Bo Hu, and Zang Li. 2025. Toward structured knowledge reasoning: Contrastive retrieval-augmented generation on experience. In Findings of the Association for Computational Linguistics: ACL 2025, pages 23891–23910, Vienna, Austria. Association for Computational Linguistics.

Zhankui He, Zhouhang Xie, Rahul Jha, Harald Steck, Dawen Liang, Yesu Feng, Bodhisattwa Prasad Majumder, Nathan Kallus, and Julian McAuley. 2023. Large language models as zero-shot conversational recommenders. In Proceedings of the 32nd ACM international conference on information and knowledge management, pages 720–730.

Chen Huang, Peixin Qin, Yang Deng, Wenqiang Lei, Jiancheng Lv, and Tat-Seng Chua. 2024. Concept– an evaluation protocol on conversation recommender systems with system-and user-centric factors. arXiv preprint arXiv:2404.03304.

Sara Kemper, Justin Cui, Kai Dicarlantonio, Kathy Lin, Danjie Tang, Anton Korikov, and Scott Sanner. 2024. Retrieval-augmented conversational recommendation with prompt-based semi-structured natural language state tracking. In Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 2786– 2790.

Levente Kocsis and Csaba Szepesvári. 2006. Bandit based monte-carlo planning. In European conference on machine learning, pages 282–293. Springer.

Wenqiang Lei, Xiangnan He, Yisong Miao, Qingyun Wu, Richang Hong, Min-Yen Kan, and Tat-Seng Chua. 2020. Estimation-action-reflection: Towards deep interaction between conversational and recommender systems. In Proceedings ofthe 13th International Conference on Web Search and Data Mining, pages 304–312.

Chuang Li, Yang Deng, Hengchang Hu, Min-Yen Kan, and Haizhou Li. 2025a. Chatcrs: Incorporating external knowledge and goal guidance for llm-based conversational recommender systems. In Findings of the Association for Computational Linguistics: NAACL 2025, Albuquerque, New Mexico, USA, April 29 - May 4, 2025, volume NAACL 2025 of Findings of ACL, pages 295–312.

Qianlong Li, Chen Huang, Shuai Li, Yuanxin Xiang, Deng Xiong, and Wenqiang Lei. 2025b. GraphOT-TER: Evolving LLM-based graph reasoning for complex table question answering. In Proceedings of

the 31st International Conference on Computational Linguistics, pages 5486–5506, Abu Dhabi, UAE. Association for Computational Linguistics.

Raymond Li, Samira Ebrahimi Kahou, Hannes Schulz, Vincent Michalski, Laurent Charlin, and Chris Pal. 2018. Towards deep conversational recommendations. Advances in neural information processing systems, 31.

Nelson F Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. 2023a. Lost in the middle: How language models use long contexts. arXiv preprint arXiv:2307.03172.

Yang Liu, Dan Iter, Yichong Xu, Shuohang Wang, Ruochen Xu, and Chenguang Zhu. 2023b. G-eval: Nlg evaluation using gpt-4 with better human alignment. arXiv preprint arXiv:2303.16634.

Yuanxing Liu, Wei-Nan Zhang, Yifan Chen, Yuchi Zhang, Haopeng Bai, Fan Feng, Hengbin Cui, Yongbin Li, and Wanxiang Che. 2023c. Conversational recommender system and large language model are made for each other in e-commerce pre-sales dialogue. arXiv preprint arXiv:2310.14626.

Seungwhan Moon, Pararth Shah, Anuj Kumar, and Rajen Subba. 2019. Opendialkg: Explainable conversational reasoning with attention-based walks over knowledge graphs. In Proceedings of the 57th annual meeting of the association for computational linguistics, pages 845–854.

Bo Ni, Zheyuan Liu, Leyao Wang, Yongjia Lei, Yuying Zhao, Xueqi Cheng, Qingkai Zeng, Luna Dong, Yinglong Xia, Krishnaram Kenthapadi, Ryan Rossi, Franck Dernoncourt, Md Mehrab Tanjim, Nesreen Ahmed, Xiaorui Liu, Wenqi Fan, Erik Blasch, Yu Wang, Meng Jiang, and Tyler Derr. 2025. Towards trustworthy retrieval augmented generation for large language models: A survey.

Peixin Qin, Chen Huang, Yang Deng, Wenqiang Lei, and Tat-Seng Chua. 2024. Beyond persuasion: Towards conversational recommender system with credible explanations. arXiv preprint arXiv:2409.14399.

Yanwei Ren, Haotian Zhang, Fuxiang Wu, Jiayan Qiu, Jiaxing Huang, Baosheng Yu, and Liu Liu. 2025. Sigma: Refining large language model reasoning via sibling-guided monte carlo augmentation.

Natalia Trukhina and Vadim Vashkelis. 2026. Compress the context, keep the commitments: A formal framework for verifiable llm context compression. arXiv preprint arXiv:2605.17304.

Peiyi Wang, Lei Li, Liang Chen, Zefan Cai, Dawei Zhu, Binghuai Lin, Yunbo Cao, Lingpeng Kong, Qi Liu, Tianyu Liu, and Zhifang Sui. 2024. Large language models are not fair evaluators. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 9440–9450, Bangkok, Thailand. Association for Computational Linguistics.

Ting-Chun Wang, Shang-Yu Su, and Yun-Nung Chen. 2022a. Barcor: Towards a unified framework for conversational recommendation systems. arXiv preprint arXiv:2203.14257.

Xiaolei Wang, Xinyu Tang, Wayne Xin Zhao, Jingyuan Wang, and Ji-Rong Wen. 2023a. Rethinking the evaluation for conversational recommendation in the era of large language models. arXiv preprint arXiv:2305.13112.

Xiaolei Wang, Chunxuan Xia, Junyi Li, Fanzhe Meng, Lei Huang, Jinpeng Wang, Wayne Xin Zhao, and Ji-Rong Wen. 2025. Search-based interaction for conversation recommendation via generative reward model based simulated user. In Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval, SIGIR ’25, page 75–84, New York, NY, USA. Association for Computing Machinery.

Xiaolei Wang, Kun Zhou, Xinyu Tang, Wayne Xin Zhao, Fan Pan, Zhao Cao, and Ji-Rong Wen. 2023b. Improving conversational recommendation systems via counterfactual data simulation. In Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pages 2398–2408.

Xiaolei Wang, Kun Zhou, Ji-Rong Wen, and Wayne Xin Zhao. 2022b. Towards unified conversational recommender systems via knowledge-enhanced prompt learning. In Proceedings ofthe 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pages 1929–1937.

Yuxiang Wu, Guanting Dong, and Weiran Xu. 2023. Semantic parsing by large language models for intricate updating strategies of zero-shot dialogue state tracking. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 11093– 11099.

Haozhe Xu, Xiaohua Wang, Changze Lv, and Xiaoqing Zheng. 2025. Beyond single labels: Improving conversational recommendation through LLM-powered data augmentation. In Proceedings of the 63rd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 15573– 15590, Vienna, Austria. Association for Computational Linguistics.

Xiao Yu, Maximillian Chen, and Zhou Yu. 2023. Prompt-based monte-carlo tree search for goaloriented dialogue policy planning.

Gangyi Zhang, Chongming Gao, Wenqiang Lei, Xiaojie Guo, Shijun Li, Hongshen Chen, Zhuozhi Ding, Sulong Xu, and Lingfei Wu. 2023. Vague preference policy learning for conversational recommendation. arXiv preprint arXiv:2306.04487.

Tong Zhang, Chen Huang, Yang Deng, Hongru Liang, Jia Liu, Zujie Wen, Wenqiang Lei, and Tat-Seng Chua. 2024. Strength lies in differences! improving strategy planning for non-collaborative dialogues via diversified user simulation. arXiv preprint arXiv:2403.06769.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, et al. 2023. Judging llm-as-a-judge with mt-bench and chatbot arena. Advances in neural information processing systems, 36:46595–46623.

## A Implementation Details

All experiments are conducted using a single Nvidia RTX A6000 GPU.

## A.1 Implementation of baselines

Baseline implementations are sourced from their respective official code repositories on GitHub or publicly available checkpoints. GPT-4o-mini is employed as the LLM backbone and text-embeddingada-002 is used for recommendation module for all the LLM-based CRSs.

Notably, We adapt the two MCTS-based CRS baselines(SAPIENT(Du et al., 2025) and T-EPL(Dao et al., 2024)) to our dialogue-based LLMpowered setting through a unified high-level planning interface.

• SAPIENT-LLM. Consider that SAPIENT is originally designed for attribute-based CRS with discrete ask/recommend actions over attribute values and items. To adapt it to our dialoguebased LLM-powered CRS setting, we replace the original attribute-value state with an LLMmaintained dialogue state, including the conversation history, extracted positive and negative user preferences, rejected items, and the current candidate item set. We preserve its high-level action space, where ask selects a preference facet inferred from item metadata and dialogue context, and recommend selects items from the current candidate ranking. The selected abstract action is then verbalized by the LLM into a naturallanguage response, while the subsequent user utterance is parsed by the LLM state tracker to update the dialogue state.

• T-EPL. Consider that T-EPL is originally formulated for target-driven recommendation dialogues and assumes access to a predefined target item. To make it comparable in our dialogue-based CRS setting, we remove access to the ground-truth target item and replace its target-conditioned value function with a beliefweighted value over the current candidate distribution. Specifically, the base recommender produces a ranked candidate set given the current dialogue state, and T-EPL estimates the value of a simulated state by retrieving similar past dialogue states from memory and aggregating their outcome scores weighted by the current candidate probabilities. During MCTS expansion, candidate actions are represented as abstract dialogue acts, such as asking about an uncertain preference, recommending an item, clarifying ambiguous constraints, or justifying a recommendation, and are finally realized as natural-language responses by the LLM.

## A.2 Difference between DREAMS and baselines

As shown in Table 5, compared with each baseline, DREAMS differs as follows: SAPIENT uses MCTS with free-form nodes only for attribute elicitation and relies on free-form exploitation; T-EPL similarly restricts free-form-node MCTS to targetoriented topic planning; InterCRS relies entirely on free-form dialogue to evaluate LLM performance; MACRS focuses on agent collaboration in free-form dialogue; PC-CRS studies persuasion and credibility under free-form interaction; ChatCRS enhances domain knowledge while retaining free-form dialogue; and RA-CRS uses turnindependent JSON only for preference elicitation, followed by free-form exploitation. In contrast, DREAMS jointly models preference elicitation and exploitation using MCTS over structured conversational states.

## A.3 Implementation of DREAMS

In this section, we will introduce in detail the implementation of different components in DREAMS , including: actions for ELNode transition, state update component, LLM scorer, online simulator, reward function and EXNode refinement.

## A.3.1 Actions for ELNode transition

DREAMS employs an iterative Monte Carlo search algorithm to identify the optimal action at each round from the action set defined in Table ??. The action space is organized around the core policy decisions in multi-turn conversational recommendation: acquiring missing preferences, resolving ambiguity or recovering from negative feedback, recommending from the current preference state, and responding to a request for item details. This design follows the ask–recommend–feedback structure studied in prior CRS policies. EAR integrates preference estimation, system action, and reflection over user feedback (Lei et al., 2020), while UNICORN learns a unified policy over asking about attributes and recommending items (Deng et al., 2021). DREAMS instantiates these decisions at a finer granularity for LLM-based CRS: attribute-specific inquiry actions acquire or disambiguate preferences, FAILUREREFLECTION handles rejected recommendations, ITEMRECOMMEN-DATION activates EXNode-based retrieval, and ITE-MEXPLANATION handles post-recommendation information requests. Table 6 summarizes the role and state effect of each action.

<table><tr><td>Model</td><td>Elicitation</td><td>Exploitation</td><td>Design Highlights</td><td>Task</td></tr><tr><td>SAPIENT</td><td>MCTS with free-form node</td><td>Free-form (whole dialogue)</td><td>Free-form node MCTS just for elici- tation.</td><td>Attr.</td></tr><tr><td>T-EPL</td><td>MCTS with free-form node</td><td>Free-form (whole dialogue)</td><td>Free-form node MCTS just for topic planning.</td><td>Target</td></tr><tr><td>InterCRS</td><td>Free-form (whole dialogue)</td><td>Free-form (whole dialogue)</td><td>LLMs&#x27; performance evaluation.</td><td>Dialog</td></tr><tr><td>MACRS</td><td>Free-form (whole dialogue)</td><td>Free-form (whole dialogue)</td><td>Agents collaboration.</td><td>Dialog</td></tr><tr><td>PC-CRS</td><td>Free-form (whole dialogue)</td><td>Free-form (whole dialogue)</td><td>Persuasiveness and credibility study.</td><td>Dialog</td></tr><tr><td>ChatCRS</td><td>Free-form (whole dialogue)</td><td>Free-form (whole dialogue)</td><td>Domain-specific knowledge en- hancement.</td><td>Dialog</td></tr><tr><td>RA-CRS</td><td>Turn-independent JSON</td><td>Free-form (whole dialogue)</td><td>JSON for preference elicitation.</td><td>Dialog</td></tr><tr><td>DREAMS</td><td>MCTS with structured node</td><td>MCTS with structured node</td><td>MCTS for both elicitation and ex- ploitation with structured node</td><td>Dialog</td></tr></table>

Table 5: Comparison of conversational recommendation methods from the perspective of preference elicitation and exploitation. Existing approaches primarily rely on free-form dialogue context or partially structured elicitation, whereas DREAMSjointly models elicitation and exploitation through MCTS over structured conversational states.

The specific implementation prompts associated with each action are provided in Appendix C.

## A.3.2 State Update Component

In order to dynamically and adaptively model user preferences, DREAMS uses a large model to structure free-form text to achieve continuous updating of the conversation state.

## A.3.3 LLM Scorer

The LLM scorer is used for providing prior probabilities over possible actions for the tree node expansion and the optimal action selection after iterations. To enhance robustness, we sample the LLM’s predictions m times and aggregate the results into a probability distribution. See Figure 8 for details and prompts of each prediction time.

## A.3.4 Online Simulator

During simulation of planning-tree, we conduct an online user simulator to give feedback in response to simulated actions.

## A.3.5 Pseudo-Code

We provide the pseudo-code of DREAMS in Algorithm 1.

Algorithm 1: Pseudo-code of DREAMS   
Input :Iteration Steps n(El) and n(Ex),   
Simulation Depth k(El), Max Turn $T$   
Output :CRS Response   
1 Set $\dot { E } L N o d e _ { 0 } = \dot { \{ \} } ,$ Dialogue History   
$H = \{ \} , t = 1$   
2 while dialogue turn $t < T$ do   
3 Receive User Utterance u   
4 Update $E L N o d e _ { t }$ using $E L N o d e _ { t - 1 }$ and u<sub>t</sub>   
/<sub>\*</sub> 1. ELNode Process Start \*/   
5 for $i \gets 1$ to n(El) do   
6 Perform MCTS with max tree depth being   
k(El), and obtain the estimated value for   
each action   
7 Obtain best action $a ^ { * }$ based on multiple MCTS   
trials   
$/ \star \_ 2$ EXNode Process Start \*/   
8 if a<sup>∗</sup> = ItemRecommendation then   
9 Initialize query $E X N o d e _ { 0 } = H$   
10 LLM generates refinement actions   
$[ a _ { 1 } ^ { R } , . . . a _ { j } ^ { R } , . . . , a _ { n ( E x ) } ^ { R } ]$   
11 Obtain query scores   
$[ R _ { 1 } ^ { R } , \dot { . . . } R _ { j } ^ { \dot { R } } , . . . , R _ { n ( E x ) } ^ { R } ]$ using Eq.5   
12 Recommend Item I using query with   
highest score   
13 if User Accepts I then   
14 Return Successful Recommendation   
15 else   
16 Generate Response r aligning with $a ^ { * }$   
17 $\dot { H } \gets H \cup \{ u _ { t } , r _ { t } \} , t = t + \check { 1 }$   
18 Return Dialogue Termination (Max Turns Reached)

## A.4 Reward Design

The reward in DREAMS is designed as a hierarchical combination of intermediate and final signals, rather than a single sparse success indicator. For a simulated transition $( s _ { t } , a _ { t } , s _ { t + 1 } )$ , we define the

<table><tr><td>Action</td><td>Policy role</td><td>Trigger and effect</td></tr><tr><td>GENREINQUIRY</td><td>Preference acquisition</td><td>Asks for a missing or ambiguous genre preference and updates the corresponding state field.</td></tr><tr><td>STARINQUIRY</td><td>Preference acquisition</td><td>Asks for a missing or ambiguous actor preference and updates the corresponding state field.</td></tr><tr><td>DIRECTORINQUIRY</td><td>Preference acquisition</td><td>Asks for a missing or ambiguous director preference and updates the corresponding state field.</td></tr><tr><td>FAILUREREFLECTION Feedback recovery</td><td></td><td>Uses a rejected recommendation and its stated reason to correct the preference state and select a suitable follow-up action.</td></tr><tr><td>ITEMREC</td><td>Transition to exploitation</td><td>Activates an EXNode, which refines the structured state into a retrieval query, retrieves and reranks candidates, and produces a recommendation.</td></tr><tr><td>ITEMEXPLANATION</td><td>Post-rec response</td><td>Provides requested information about an already recommended item without prematurely initiating a new recommendation.</td></tr></table>

Table 6: Action space of the ELNode policy. The actions cover preference acquisition, recovery from negative feedback, transition to preference exploitation, and post-recommendation explanation.

immediate reward as:

$$
r _ { t } = \lambda _ { e } r _ { t } ^ { \mathrm { e x p l o r e } } + \lambda _ { r } r _ { t } ^ { \mathrm { r e t r i e v e } } + \lambda _ { u } r _ { t } ^ { \mathrm { a t t i u d e } } - \lambda _ { c } r _ { t } ^ { \mathrm { c o s t } } ,\tag{6}
$$

where $r _ { t } ^ { \mathrm { e x p l o r e } }$ measures whether the current action improves the completeness of the structured preference state, $r _ { t } ^ { \mathrm { r e t r i e v e } }$ measures whether the refined query leads to retrieval results that are more consistent with the hidden target profile, $r _ { t } ^ { \mathrm { a t t i t u d e } }$ captures the user simulator’s feedback after interaction, and $\boldsymbol r _ { t } ^ { \mathrm { c o s t } }$ penalizes unnecessarily long or repetitive trajectories.

The overall action value is computed as the discounted return:

$$
R ( s _ { t } , a _ { t } ) = \mathbb { E } \left[ \sum _ { k = t } ^ { T } \gamma ^ { k - t } r _ { k } + \lambda _ { f } \cdot \mathrm { A c c e p t } _ { T } \right] ,\tag{7}
$$

where $\mathrm { A c c e p t } _ { T } \in \{ 0 , 1 \}$ indicates whether the simulated dialogue ends with user acceptance.

For exploration, $r _ { t } ^ { \mathrm { e x p l o r e } }$ is positive when the action helps clarify previously missing or lowconfidence attributes in the structured user state. This encourages the planner to ask informative questions when the current preference state is still incomplete.

For exploitation, $r _ { t } ^ { \mathrm { r e t r i e v e } }$ is defined using verifiable signals from the environment. Concretely, we compare the attributes of the top retrieved item with the ground-truth attribute profile underlying the simulator. Let $\hat { y } _ { t } ^ { ( 1 ) }$ denote the top-1 retrieved item and let $A ( \cdot )$ denote its attribute set. We compute retrieval reward from the overlap between $A ( \hat { y } _ { t } ^ { ( 1 ) } )$ ) and the target attribute set, so that higher reward is assigned only when retrieval becomes more faithful to the actual user preference profile. This makes the retrieval reward externally checkable, instead of relying solely on free-form LLM scoring.

Finally, $r _ { t } ^ { \mathrm { a t t i t u d e } }$ rewards trajectories that lead to explicit user acceptance and penalizes rejection, while $r _ { t } ^ { \mathrm { c o s t } }$ discourages overly long dialogues. In this way, the reward design combines dense intermediate supervision with a final outcome objective. Although DREAMS does not optimize the model with reinforcement learning, this design is conceptually aligned with recent verifiable-reward based paradigms, in that the key reward terms are grounded in observable and reproducible signals from the environment.

## A.4.1 R-chain Refinement

The R-chain first processes the multi-turn dialogue history to extract explicit user preferences regarding genres, actors, directors, and thematic elements. This extraction is performed by prompting an LLM with a structured instruction that emphasizes finegrained disambiguation of closely related genres $( \mathrm { e . g . }$ ., “thriller” vs. “crime”) and returns a structured JSON of liked/disliked entities with associated confidence scores.

Once preferences are extracted, the system formulates candidate retrieval queries using strategies defined by the LLM. A lightweight Monte Carlo Tree is employed to select the optimal refinement strategy. Specifically, a tree of strategy nodes is constructed, where each node corresponds to a paraphrasing strategy and is evaluated based on the simulated performance of the resulting query. During simulation, a query is generated via the LLM based on the node’s strategy and then evaluated using a two-part reward function: (1) a preference match score that quantifies how well top retrieved items align with the user’s stated preferences, and (2) a potential attitude score that estimates user acceptance likelihood based on cumulative preference overlap and diversity of retrieved genres. The combined reward guides backpropagation in the tree, leading to the selection of the most promising strategy.

The final refined query is then embedded and used to retrieve top-5 candidate items via cosine similarity over precomputed item embeddings. The top-ranked items are cached for recommendation, and metadata including the selected strategy, refined query text, and extracted preferences are returned for transparency and downstream explanation. This structured search and modeling approach ensures not only accurate preference alignment but also adaptive query optimization based on retrieval performance feedback.

## A.5 Implementation of Baselines

For all baselines, we used the official code with the checkpoints from the original paper’s GitHub repository or the official prompt from the original paper for implementation. We refer to Appendix C for prompts. For a fair comparison, all LLMbased CRSs including DREAMS employ the gpt-4o-mini<sup>10</sup> as backbone and the text-embedding-ada-$0 0 2 ^ { 1 1 }$ encoder as the recommendation module to retrieve items.

## A.6 Implementation of Evaluation Metrics

To systematically assess the preference elicitation and exploitation capability of LLM-based CRSs, we implemented a multi-step evaluation protocol that includes metrics of Fine-Grained Elicitation Error $( \mathrm { F G E ^ { 2 } } )$ , Preference Exploitation Error $( \mathrm { P E ^ { 2 } } )$ and Course-Grained Elicitation Error $\mathrm { ( C G E ^ { 2 } ) }$ Among them, $\mathrm { F G E ^ { 2 } }$ and $\mathrm { P E ^ { 2 } }$ are dialogue-level metrics, and $\mathrm { C G E ^ { 2 } }$ is a turn-level metric. In the following content, we will elaborate on the design details of metrics. How LLM evaluates these metrics as an evaluator will be explained in $\mathbf { A . 7 }$ , and the prompts are in C.

## A.6.1 Dialogue-level Metrics

Fine-Grained Elicitation Error. A "Fine-Grained Elicitation Error" $( \mathrm { F G E ^ { 2 } } )$ is defined as a failure by the assistant to adapt its elicitation or exploitation strategy in response to explicit negative feedback from the user. If the assistant ignores user rejection of a previous recommendation (e.g., a movie was disliked due to its horror genre) and subsequently issues a recommendation (e.g., continue to recommend horror movies) or inquiry (e.g., inquiry for stars or directors) that does not reflect learning from that rejection, it constitutes a strategic fault. Formally, let $t _ { i }$ denote the i-th assistant turn, and $r _ { j }$ the last rejected recommendation with user-stated rejection reason $R _ { j }$ . Then, a $\mathrm { . F G E ^ { 2 } }$ occurs if:

$$
\mathrm { F G E } ^ { 2 } ( t _ { i } ) = \left\{ \begin{array} { l l } { 1 } & { \mathrm { i f ~ i n t e n t } ( t _ { i } ) , \mathrm { a s k } \} \wedge } \\ { \ } & { \mathrm { - r e f l e c t s } ( t _ { i } , R _ { j } ) } \\ { 0 } & { \mathrm { o t h e r w i s e } } \end{array} \right.\tag{8}
$$

Here, $r e f l e c t s ( t _ { i } , R _ { j } )$ is a context-aware predicate indicating whether the assistant incorporated the rejection reason $R _ { j }$ into its follow-up action (e.g., asking clarifying questions or avoiding similar content).

Preference Exploitation Error. A "Preference Exploitation Error" (PE<sup>2</sup>) is triggered when the assistant has successfully elicited explicit user preferences across all three key attributes — genre, actor, and director — yet the retrieved item associated with the assistant’s recommendation is inappropriate (recall@5 = False).

Let $P = \{ p _ { g } , p _ { a } , p _ { d } \}$ represent the user’s fully specified preferences for genre, actor, and director. Given a dialog context C, a $\mathrm { P E ^ { 2 } }$ is defined as:

$$
\begin{array} { r } { \mathrm { P E } ^ { 2 } ( t _ { k } ) = \left\{ \begin{array} { l l } { 1 } & { \mathrm { i f ~ } \mathrm { p r e f \_ s a t i s f i e d } ( C ) = \mathrm { T r u e } } \\ & { \land \mathrm { r e t r i e v a l } ( t _ { k } ) = \mathrm { F a l s e } } \\ { 0 } & { \mathrm { o t h e r w i s e } } \end{array} \right. } \end{array}\tag{9}
$$

The function $p r e f _ { s } a t i s f i e d ( C )$ returns true only if the assistant has successfully acquired all three preference dimensions $p _ { g } , p _ { a } , p _ { d }$ from the user at or before turn $t _ { k }$

Evaluation Process. Evaluation is conducted in dialog-level granularity. For each conversation file: 1) A GPT-4o model is provided with the full dialog history and a structured system prompt defining the criteria for $\mathrm { F G E ^ { 2 } }$ and $\mathrm { P E ^ { 2 } } \cdot 2 )$ The model returns a JSON-encoded response: $\{ " \mathrm { F G E ^ { 2 } " }$ : true/false, $" \mathrm { P E ^ { 2 } } "$ : true/false}. The evaluation script aggregates the counts of $\mathrm { F G E ^ { 2 } } _ { t r u e } , \mathrm { F G E ^ { 2 } } _ { f a l s e } , \mathrm { P E ^ { 2 } } _ { t r u e }$ $\mathrm { P E } ^ { 2 } f a l s e$ over all dialogs to compute proportions. Final scores are reported as:

<table><tr><td rowspan=1 colspan=3>CRS Evaluation Metrics</td></tr><tr><td rowspan=1 colspan=1>Metric</td><td rowspan=1 colspan=1>Definition</td><td rowspan=1 colspan=1>Measurement</td></tr><tr><td rowspan=1 colspan=1>Course-GrainedElicitationError $\mathrm { ( C G E ^ { 2 } ) }$ </td><td rowspan=1 colspan=1>Error rate of high-level decision $( \mathrm { i . e . , }$ make recommendation and ask forinformation) mismatch betweenretrieval quality and action taken.</td><td rowspan=1 colspan=1>We run retrieval to see if the retrieval iscorrect and annotate CRS&#x27;s actionof each turn within each conversation.we measure the proportion of wrongactions across all conversations as themetric, where wrong action means theCRS:• Ask when retrieval is correct, or• Recommend when retrieval is wrong.Exclude turns with other actions.</td></tr><tr><td rowspan=1 colspan=1>Fine-GrainedElicitationError $( \mathrm { F G E ^ { 2 } } )$ </td><td rowspan=1 colspan=1>Error rate of ineffective follow-upquestions after the system receivesspecific user feedback about theirpreferences. For example,when a user expresses dislike forcertain directors’ styles, the CRS failsto ask the right questions (ask aboutthe director) but take a wrong strategy(i.e., ask about movie genre or succe-ssive recommendation).</td><td rowspan=1 colspan=1>For each dialogue, consider all turns th-at immediately follow user feedback toa recommendation. The fine-grained el-icitation error is defined as the proporti-on of such turns within the dialogue w-here the CRS action mismatches the us-er intent (e.g., issuing an irrelevant inq-uiry or making a premature recommen-dation).</td></tr><tr><td rowspan=1 colspan=1>PreferenceExploitationError (PE2)</td><td rowspan=1 colspan=1>Failure rate of retrieving incorrect it-ems after full preference explicit dis-closure. For example, known userpreference on action, adventure anddrama movies, but recommending athriller movie.</td><td rowspan=1 colspan=1>When user simulator reveals all his pre-ference (identified by an advanced extraLLM), we count the proportion of inco-rrect recommendations.</td></tr></table>

Table 7: Key performance indicators for conversational recommendation systems

$$
\begin{array} { r } { \mathrm { F G E } ^ { 2 } = \frac { \sum \mathrm { F G E } _ { \mathrm { \tiny ~ t r u e } } ^ { 2 } } { \sum \mathrm { F G E } _ { \mathrm { \tiny ~ t r u e } } ^ { 2 } + \sum \mathrm { F G E } _ { \mathrm { \tiny ~ f a l s e } } ^ { 2 } } } \\ { \mathrm { P E } ^ { 2 } = \frac { \sum \mathrm { P E } _ { \mathrm { \tiny ~ t r u e } } ^ { 2 } } { \sum \mathrm { P E } _ { \mathrm { \tiny ~ t r u e } } ^ { 2 } + \sum \mathrm { P E } _ { \mathrm { \tiny ~ f a l s e } } ^ { 2 } } } \end{array}\tag{10}
$$

## A.6.2 Turn-level Metric

For the evaluation of $\mathrm { C G E ^ { 2 } }$ , we categorizes assistant utterances in the recorded data into predefined classes based on their intent and the retrieval relevance of associated results. The classification schema is composed of four primary categories: Ask-True, Ask-False, Recommend-True, Recommend-False. Intent and Relevance Classification Framework. Each assistant utterance $u _ { i }$ within a conversation $C = \{ u _ { 1 } , u _ { 2 } , . . . u _ { n } \}$ is analyzed using an LLM-powered evaluator, prompted with structured instructions and context information. The model classifies each utterance based

on:

$$
\begin{array} { l } { { i n t e n t ( u _ { i } ) \in \{ \mathrm { A s k , R e c o m m e n d , O t h e r s } \} } } \\ { { r e t r i e v a l ( u _ { i } ) \in \{ \mathrm { T r u e , F a l s e } \} } } \end{array}\tag{11}
$$

Using the above classification, we define the following evaluation counts:

$$
\operatorname { L e t } a _ { i } = \operatorname { i n t e n t } ( u _ { i } ) \operatorname { a n d } r _ { i } = \operatorname { r e t r i e v a l } ( u _ { i } ) .
$$

$$
{ \mathrm { A T } } = \sum _ { i } ^ { } { \mathcal { H } } [ a _ { i } = { \mathrm { A s k } } \land r _ { i } = { \mathrm { T r u e } } ] ,\tag{12}
$$

$$
{ \mathrm { A F } } = \sum _ { i } ^ { } { \mathcal { H } } [ a _ { i } = { \mathrm { A s k } } \land r _ { i } = { \mathrm { F a l s e } } ] ,\tag{13}
$$

$$
{ \mathrm { R T } } = \sum _ { i } { \mathcal { H } } [ a _ { i } = { \mathrm { R e c o m m e n d } } \wedge r _ { i } = { \mathrm { T r u e } } ] ,\tag{14}
$$

$$
{ \mathrm { R F } } = \sum _ { i } { \mathcal { H } } [ a _ { i } = { \mathrm { R e c o m m e n d } } \wedge r _ { i } = { \mathrm { F a l s e } } ] .\tag{15}
$$

To enable direct comparison between dialog behavior classes, we compute the relative frequency of each act category among all labeled utterances that fall under either Ask or Recommend intents. Specifically, for each class $x \in$ {AT, AF, RT, RF}, we calculate its normalized ratio $r _ { x }$ as follows:

$$
\mathbf { A T }
$$

$$
r _ { \mathrm { A T } } = { \frac { \mathrm { \cdot } \mathrm { \cdot } \mathrm { \cdot } } { \mathrm { A T } + \mathrm { A F } + \mathrm { R T } + \mathrm { R F } } }\tag{16}
$$

$$
\mathrm { A F }
$$

$$
r _ { \mathrm { A F } } = { \frac { \mathrm { a } \mathrm { . } } { { \mathrm { A T } } + { \mathrm { A F } } + { \mathrm { R T } } + { \mathrm { R F } } } }\tag{17}
$$

$$
\mathrm { R T }
$$

$$
r _ { \mathrm { R T } } = { \frac { - \mathrm { R T } } { { \mathrm { A T } } + { \mathrm { A F } } + { \mathrm { R T } } + { \mathrm { R F } } } }\tag{18}
$$

$$
r _ { \mathrm { R F } } = { \frac { \mathrm { a . c . } } { { \mathrm { A T } } + { \mathrm { A F } } + { \mathrm { R T } } + { \mathrm { R F } } } }\tag{19}
$$

Here, True and False refer to the relevance of the retrieval result, not to the correctness of the assistant action. Therefore, Ask-True denotes an unnecessary elicitation action when retrieval is already sufficient, and Recommend-False denotes a premature recommendation when retrieval is still incorrect. Under this convention, the coarse-grained elicitation error is defined as

$$
\mathrm { C G E ^ { 2 } } = r _ { \mathrm { A T } } + r _ { \mathrm { R F } }\tag{20}
$$

These ratios ensure that evaluation results reflect the distributional tendencies of the assistant’s strategy, independent of the total number of utterances. This normalization is especially useful for comparing across different models or dialog scenarios with unequal total counts.

## A.6.3 Prefix-level Next Action Evaluation

Table 8 shows a clear progression from state representation to state-guided decision making and then to structured search. RA-CRS uses a JSON-format dialogue state, but its decision policy only weakly operationalizes that state. DREAMS w/Json Only improves over RA-CRS by heuristically using the structured state, increasing Next-Action Acc. from 0.542 to 0.633 and reducing Premature Rec. Rate from 0.308 to 0.217. Full DREAMS further improves all metrics by replacing heuristic use with structured search, reaching 0.758 Next-Action Acc. and reducing Redundant Q. Rate and Premature Rec. Rate to 0.089 and 0.117. These results indicate that structured preference tracking is most effective when the state is not only represented, but also searched over to choose the next action.

## A.7 Implementation of LLM Evaluation

Prompt for the evaluation of $\mathrm { F G E ^ { 2 } }$ $\mathrm { P E ^ { 2 } }$ and $\mathrm { C G E ^ { 2 } }$ are in Figure 6 and 7

Table 8: Prefix-level next-action evaluation. In Panel (a), w/Json Only denotes DREAMS without structured search. Panel (b) reports annotation reliability.
<table><tr><td>(a) Next-action quality Model</td><td>Acc. ↑</td><td>Useful Ask ↑</td><td>Redun. Q. ↓</td><td>Prem. Rec. ↓</td></tr><tr><td>RA-CRS</td><td>0.542</td><td>0.613</td><td>0.242</td><td>0.308</td></tr><tr><td>w/Json Only</td><td>0.633</td><td>0.694</td><td>0.171</td><td>0.217</td></tr><tr><td>DREAMS</td><td>0.758</td><td>0.806</td><td>0.089</td><td>0.117</td></tr><tr><td>(b) Annotation reliability</td><td colspan="4"></td></tr><tr><td>Metric</td><td colspan="4">Agreement</td></tr><tr><td>Fleiss&#x27; Kappa on gold next action</td><td colspan="4">0.69</td></tr><tr><td>Krippendorff&#x27;s alpha on error labels</td><td colspan="4">0.72</td></tr><tr><td>Human majority vs. LLM pre-label Macro-F1</td><td colspan="4">0.77</td></tr></table>

## A.8 Implementation of Human Evaluation and Judge Reliability

Inspired by (Zhang et al., 2024), we conduct an interactive human evaluation on DREAMS under the ReDial dataset. We compare DREAMS with three competitive baselines, including PC-CRS, ChatCRS, and MACRS. Specifically, we hire 60 crowdworkers with diverse personas to interact with the four CRS models on movie recommendation tasks. To ensure objectivity, annotators are not allowed to communicate with each other during the evaluation process.

After the conversations, we collect dialogues for each CRS and evaluate their performance using Success Rate (SR), Recall@1, $\mathrm { F G E ^ { 2 } }$ , and $\mathrm { P E ^ { 2 } }$ , as defined in Section 3.1. The final Fleiss’ Kappa score reaches 0.71, indicating substantial agreement and supporting the reliability of the human verification process.

Reliability of the LLM Judge. Prior work has shown that strong LLM judges can achieve relatively high agreement with human judgments under well-specified evaluation criteria, as evidenced by benchmarks such as MT-Bench and Chatbot Arena (Zheng et al., 2023). Although recent studies have identified limitations of LLM judges, such as positional bias and model favoritism (Wang et al., 2024), these issues mainly arise in open-ended comparative evaluation settings. In contrast, our evaluator is only required to identify explicitly defined error types, making the judgment process more constrained. We therefore conduct an additional human validation study to examine its reliability. As summarized in Table 9, the LLM judge achieves strong consistency with majority human annotations, suggesting that it can serve as a reliable and scalable proxy for our error-oriented evaluation. To further assess the reliability of our LLM-based evaluator, we randomly sample 50 dialogues generated by DREAMS on ReDial. We then ask 10 human annotators to label the utterance-level $\mathrm { F G E ^ { 2 } }$ $\mathrm { P E ^ { 2 } }$ and the corresponding error ratios, following the same evaluation criteria used by the LLM evaluator. Human-human agreement is measured by Krippendorff’s alpha, while LLM-human consistency is measured by Spearman’s correlation over dialogue-level error ratios and macro-F1 against majority human labels at the utterance level. The Krippendorff’s alpha scores are 0.74 and 0.68 for Elicitation and Exploitation, respectively, indicating substantial annotator agreement. The LLM judge also shows strong alignment with human majority labels, consistent with prior findings on LLM-based evaluators (Liu et al., 2023b). Detail results are shown in Table 9

Table 9: Reliability validation of the LLM judge.
<table><tr><td>Metric</td><td>Human-Human Krippendorff&#x27;s α</td><td>LLM-Human Spearman&#x27;s ρ</td><td>LLM-Human Macro-F1</td></tr><tr><td>Overall</td><td>0.70</td><td>0.72</td><td>0.76</td></tr><tr><td>Elicitation  $/ \mathrm { F G E ^ { 2 } }$ </td><td>0.74</td><td>0.75</td><td>0.79</td></tr><tr><td>Exploitation / PE2</td><td>0.68</td><td>0.69</td><td>0.73</td></tr></table>

## A.9 Implementation of User Simulator & Datasets

Training and evaluating CRS with real user interactions can be impractically expensive at scale. To address this issue, we follow the previous LLMbased CRS(Qin et al., 2024)(Fang et al., 2024) to employ $\mathrm { C h a t G P T ^ { 1 2 } }$ as the user simulator.

For detail, we implement a goal-driven user simulator to interact with the CRSs under controlled and reproducible settings. At the beginning of each simulated dialogue, the user simulator is assigned a predefined persona and user preference, including preferred genres, actors, directors for movie recommendation and genres, writers for book recommendation. These preferences are sampled from a fixed pool of candidate attributes and are paraphrased into natural language to enhance realism. Specifically, we identify the 15 most common preference groups in Redial, 18 in OpendialKG for movie recommendation and 16 in OpendialKG for book recommendation. Additionally, in conjunction with 10 pre-defined diverse user personas, the evaluation process generates 150, 180 and 160 dialogues, respectively, for each dataset. Throughout the dialogue, the simulator generates context-aware responses based on its persona and the dialogue history. After receiving each recommendation from the system, the simulator successively executes three actions: (1) feeling inference, (2) insight generation, and (3) response generation. Firstly, the user simulator internally evaluates the recommendation’s alignment with its preferences through a brief, persona-consistent internal monologue. This internal state reflection guides the generation of follow-up intentions and subsequent user utterances. The simulator’s responses are generated using prompt-based calls to large language models, ensuring natural language variability, strict persona adherence, and coherence with the evolving conversation(See detailed prompts for feeling inference, insight generation and response generation in Tabel 14, 15, 16 respectively).

## A.10 Implementation of CRS-Simulator Interaction

The interaction continues in a turn-by-turn manner until the simulator determines that a recommendation satisfies all criteria — namely, matching all preferred genres and at least one preferred actor and director — at which point the dialogue is concluded with a special [END] token. This simulation framework enables the systematic evaluation of recommender strategies under diverse user profiles and realistic conversational behaviors.

## A.11 Implementation of Memory Usage of DREAMS

The memory footprint of DREAMS is exceptionally low. The overhead introduced by the DREAMS’s tree structure is minimal, primarily storing lightweight textual metadata like dialogue states and node scores, which avoids high-memory objects such as dense tensors. For instance, our GPT-4o-mini implementation keeps the total input text below 128k tokens (approximately 512KB using int32 encoding). Even with additional embeddings, the total memory usage remains around 375MB, a size that is well within the manageable limits of modern systems.

## B Additional Analysis

## B.1 Practicality and Efficiency Analysis

Efficiency Optimization via Experienceaugmentation. Table 10 indicates that the time efficiency of vanilla DREAMS is comparable to existing CRS methods. However, a common concern with tree-based planning is latency. Inspired by Gu et al. (2025); Ren et al. (2025), we leverage the MCTS trees from the training phase as historical experience. This allows us to bypass the time-consuming tree search during deployment. In particular, we introduce DREAMS(EA), a Experience-Augmented version. Crucially, we construct an Experience Knowledge Base (EKB) using 50 held-out user simulators to interact with DREAMS under the settings of $n ( E l ) = 5 ,$ $k ( E l ) = 3$ , and $n ( E x ) = 3$ We record tuples $\langle \mathcal { D } , S , a , r , q ^ { \prime } , R \rangle$ that represent the dialogue context, structured state, action, decision reason, refined query, and final reward, respectively. During the inference phase, prior to generating a response, we retrieve relevant priors by calculating the semantic similarity between the current structured state and the stored experience states in the EKB. Unlike standard text-based retrieval, matching on structured states (S) ensures that the retrieved contrastive pair (one successful and one failed exemplar) shares structural isomorphism with the current situation, effectively filtering out contextual noise. By instructing the model to “follow the successful pattern and avoid specific errors”, this mechanism acts as a warm-start that wisely guide the strategic planning of CRS. As shown in Table 10, the average inference time drops to 9s, while Recall@1 improves to 0.467, confirming that DREAMS(EA) is an efficient and effective solution for real-time applications.

<table><tr><td>n(El)</td><td>k(El) n(Ex)</td><td>Model</td><td>Time(s)</td><td>R@1</td></tr><tr><td rowspan="5"></td><td></td><td>ChatCRS</td><td>8</td><td>0.422</td></tr><tr><td></td><td>PC-CRS</td><td>13</td><td>0.378</td></tr><tr><td></td><td>RA-CRS</td><td>19</td><td>0.267</td></tr><tr><td></td><td>MACRS</td><td>18</td><td>0.289</td></tr><tr><td></td><td>InterCRS</td><td>5</td><td>0.244</td></tr><tr><td>3</td><td>3</td><td>3</td><td>DREAMS</td><td>23</td><td>0.489</td></tr><tr><td>5</td><td>3</td><td>3</td><td>DREAMS</td><td>34</td><td>0.578</td></tr><tr><td>10</td><td>3</td><td>3</td><td>DREAMS</td><td>75</td><td>0.600</td></tr><tr><td>3</td><td>2</td><td>3</td><td>DREAMS</td><td>14</td><td>0.422</td></tr><tr><td>3</td><td>1</td><td>3</td><td>DREAMS</td><td>10</td><td>0.333</td></tr><tr><td>3</td><td>2</td><td>5</td><td>DREAMS</td><td>16</td><td>0.444</td></tr><tr><td>3</td><td>3</td><td>5</td><td>DREAMS</td><td>25</td><td>0.533</td></tr><tr><td>3</td><td>3</td><td>10</td><td>DREAMS</td><td>30</td><td>0.489</td></tr><tr><td></td><td>一</td><td></td><td>DREAMS(EA)</td><td>9</td><td>0.467</td></tr></table>

Table 10: Time efficiency analysis. n and k represent iteration steps and simulation depth. (El) and (Ex) denote parameters for El- and ExNodes, respectively. DREAMS excels in high-stakes environments requiring maximum effectiveness, whereas DREAMS(EA) is tailored for scenarios where low latency is critical.

Hyperparameter and Computational Overhead. DREAMS has three hyperparameters: $n ( E l ) , k ( E l )$

<table><tr><td rowspan="2">Backbone</td><td colspan="3">Gemini-2.5-flash (1M)</td><td colspan="3">GPT-4o-mini (128K)</td></tr><tr><td>SR</td><td> $F G E ^ { 2 }$ </td><td> $C G E ^ { 2 }$ </td><td>SR</td><td> $F G E ^ { 2 }$ </td><td> $C G E ^ { 2 }$ </td></tr><tr><td>DREAMS (Ours)</td><td>0.587</td><td>0.180</td><td>0.277</td><td>0.560</td><td>0.100</td><td>0.289</td></tr><tr><td>InterCRS (free-form)</td><td>0.347</td><td>0.347</td><td>0.503</td><td>0.313</td><td>0.360</td><td>0.570</td></tr><tr><td>RA-CRS (structured)</td><td>0.387</td><td>0.340</td><td>0.449</td><td>0.333</td><td>0.333</td><td>0.461</td></tr></table>

Table 11: Performance on the Redial dataset across LLM backbones with 128K/1M token limits.

and $n ( E x )$ Results in Table 10 show that the model’s performance improves significantly when the number of elicitation iterations $( n ( E l ) )$ is increased to 5 and the exploitation iterations $( n ( E x ) )$ contains at least 3. However, further increases lead to diminishing returns, with marginal gains or even slight performance degradation. Increasing iterations and simulation depth enhances accuracy but reduces time efficiency. We employ multithreading to parallelize computation in both Node types, ensuring DREAMS achieves comparable time efficiency and better performance. For example, under specific configuration, the computational expense of DREAMS (16s/turn) is commensurate with that of MACRS (18s/turn), yet it demonstrates a markedly superior performance metric (0.444 VS. 0.289). On this turn, DREAMS(EA) achieves a remarkable balance: it reduces the average inference time to 9s while maintaining a competitive Recall@1 of 0.467, significantly outperforming standard baselines like MACRS and PC-CRS in both speed and accuracy. Thus, DREAMS excels in high-stakes environments requiring maximum effectiveness, whereas DREAMS(EA) is tailored for real-time scenarios where low latency is critical.

Notably, both the standard DREAMS and DREAMS(EA) introduce no substantial memory overhead. This is because the tree structure and the Experience Knowledge Base predominantly store lightweight textual metadata, such as dialogue states, node scores, and reasoning tuples, rather than high-memory objects like dense tensors. Even with the addition of historical experience entries in DREAMS(EA) and the embeddings used during retrieval, the total memory usage remains approximately 560MB. This ensures that the system remains well within the limits of modern hardware while significantly outperforming baselines in realtime scenarios.

## B.2 Case Study

Figure 5 and Table 12 provide a concrete mechanism-level example. In the case study, the user’s phrase “thrilling action” creates an ambiguous signal: the user intends action movies, but the surface phrase can be misread as the thriller genre. ChatCRS and RA-CRS recommend too early and enter an error spiral based on this flawed interpretation. DREAMS instead selects GenreInquiry and asks whether the user means action movies or thriller/crime movies. Table 12 explains why this decision emerges. The LLM prior assigns the highest score to ItemRecommendation (0.44), since the user appears to request a recommendation. However, MCTS rollouts assign negative reward to immediate recommendation and high reward to GenreInquiry. As a result, GenreInquiry obtains the highest visit ratio (0.667) and final score (0.524), so DREAMS clarifies the preference before retrieval. This case shows that structured search changes decision making from following the most likely immediate response to evaluating which action best improves preference tracking. Appendix A.6.3 further evaluates next-action quality on 120 annotated Redial prefixes. DREAMS improves Next-Action Accuracy from 0.633 for w/Json Only to 0.758 and reduces Premature Recommendation Rate from 0.217 to 0.117, confirming that structured search makes the preference state actionable.

Table 12: MCTS trajectory for the ambiguouspreference prefix in Figure 5. The LLM prior favors immediate recommendation, but rollouts select GenreInquiry after evaluating downstream preference-tracking utility.
<table><tr><td>Cand. Action</td><td>LLM prior</td><td>Visit ratio</td><td>Reward</td><td>Score</td><td>Decision</td></tr><tr><td>GenreInq.</td><td>0.35</td><td>0.667</td><td>10.2</td><td>0.524</td><td>selected</td></tr><tr><td>ActorInq.</td><td></td><td>1</td><td></td><td></td><td>not selected</td></tr><tr><td>DirectorÎnq.</td><td>0.21</td><td>0</td><td>-4.9</td><td>0.098</td><td>not selected</td></tr><tr><td>ItemRec.</td><td>0.44</td><td>0.333</td><td>-4.6</td><td>0.381</td><td>not selected</td></tr><tr><td>ItemExp.</td><td></td><td></td><td>一</td><td>一</td><td>not selected</td></tr></table>

## C Prompts

Here are detailed prompts for aforementioned implementations.

![](images/8b19ea0184adf8c0e8b38128da472ad1c8ea744409134648b51c3b64b824a9de.jpg)  
Figure 5: Case study illustration. The user’s initial raw preference, containing the subtly misleading phrase (blue, non-bold text), often causes the CRSs to misinterpret the intent (e.g., mistaking "thrilling actions" (target preference is Action movie) for the Thriller genre, leading to an error spiral where baselines repeatedly act on the flawed understanding. DREAMS identifies the critical error using structured context modeling and successfully navigates to the optimal path (green, non-bold text).

![](images/8c6453928152c6db4d87247c51a162a0144b778ddbfac96fb78e3d8eae846df8.jpg)  
Figure 6: LLM Evaluation Prompt for PE<sup>2</sup>and FGE<sup>2</sup>

<table><tr><td>Iteration ID</td><td>Selection, Expand and Reward</td><td>Dialogue simulation</td></tr><tr><td></td><td></td><td>Got it! If you&#x27;re looking for an exciting adventure without the thrilling aspect, I recommend &quot; Jaws &quot; (1975), directed by Steven Spielberg. It combines thrilling, and adventure themes of faith, honor, and the complexities of war, making it a thrilling watch that aligns well with your preference. sim_user_feedback:</td></tr><tr><td></td><td>Selection: no children</td><td>I&#x27;ve seen &quot; Jaws &quot; before, and while I appreciate Steven Spielberg &#x27;s work, it didn&#x27;t quite resonate with me. Do you have any other recommendations, maybe something with a bit more action? sim_assistant_response: (ItemRecommendation)</td></tr><tr><td></td><td></td><td>Thanks for the feedback! If you&#x27;re looking for something with more action, you might enjoy “The Last Samurai&quot; (2003) — it&#x27;s rich in intense battle scenes and emotional depth. If you&#x27;re open to something more recent, “13 Assassins&quot;</td></tr><tr><td></td><td></td><td>(2010) offers gripping samurai action with stunning choreography. Would you prefer historical settings, or are you</td></tr><tr><td></td><td>Expand: [ItemRecommendation]</td><td>open to modern or sci-fi action as well?</td></tr><tr><td></td><td>Reward: 2.5</td><td>sim_user_feedback: I appreciate the effort, but none of those really catch my interest. Can you suggest something else, maybe focusing</td></tr><tr><td></td><td></td><td>more on the actors I Like? sim_assistant_response:</td></tr><tr><td></td><td></td><td>[FailureReflection] “The user rejected my recommendation and mentioned that he wanted some recommendations</td></tr><tr><td></td><td></td><td>related to his favorite stars or directors. But I don&#x27;t know who his favorite stars or directors are. So I should first ask him which stars or directors he likes.&quot;</td></tr><tr><td></td><td></td><td>[StarInquiry] That&#x27;s ok. Can you share me with some of your preferred stars and directoes?</td></tr><tr><td></td><td></td><td>sim_user_feedback: Enjoy to share that I am a fan of Jackie Chan, and I love the style of Barry Sonnenfeld</td></tr><tr><td></td><td></td><td>sim_assistant_response: [GenreInquiry]</td></tr><tr><td></td><td></td><td>To clarify, when you mention &quot;thrilling actions,&quot; do you mean action-oriented films specifically, or are you referring more broadly to genres such as thrillers or crime?</td></tr><tr><td></td><td>Selection: not_fully_expanded</td><td>sim_user_feedback: I&#x27;d say I&#x27;m leaning more toward action movies — ones that feature intense fight scenes, high stakes, and fast-paced</td></tr><tr><td></td><td>Expand: [&quot;FailureReflection&quot;] [Failure Reflection] “The user has ex-</td><td>plots. Do you have any recommendations in that vein? sim_assistant_response: [ItemRecommendation] Got it! Based on your interest in high-stakes, adrenaline-pumping action, daring adventures and something that can</td></tr><tr><td>2</td><td>plicitly rejected the recommendation of &#x27;Good for Nothing (2014)’ due to its</td><td>make you good laugh, I will recommend &quot;Men in Black&quot;. It&#x27;s an action movie with adventurous and humorous</td></tr><tr><td></td><td>theme, indicating a misunderstanding of their preferences. Therefore, the most appropriate action is to acknowledge</td><td>sim_user_feedback: That sounds awesome. Can you tell me morec about “Men in Black&quot; like it features which stars and who directs it? sim_assistant_response: [ItemExplanation]</td></tr><tr><td></td><td>this misunderstanding and reset the con- versation by revisiting preference in- quiries&quot;</td><td>Absolutely! “Men in Black&quot; is a classic action-comedy film directed by Barry Sonnenfeld. The film stars Will Smith as Agent J and Tommy Lee Jones as Agent K, two secret agents working for a covert government organization that</td></tr><tr><td></td><td>Reward: 20.1</td><td>monitors extraterrestrial activity on Earth. Based on the comic book series by Lowell Cunningham, the story follows Agent J as he is recruited into the Men in Black and partners with the veteran Agent K to investigate alien-related incidents. Together, they uncover a plot that</td></tr><tr><td></td><td></td><td>threatens both Earth and the intergalactic balance, all while navigating humorous and high-stakes encounters with aliens living secretly among humans.</td></tr><tr><td></td><td></td><td>sim_user_feedback: That definitely matched all my preference! I will give it a try tonight. Thank you [END]</td></tr><tr><td></td><td></td><td>sim_assistant_response (FailureReflection): &quot;I see that my last recommendation might not have aligned well with your preferences — I may have focused too much</td></tr><tr><td></td><td></td><td>on suspense rather than fun and excitement. Let me ask a quick question to get a clearer sense of what you&#x27;re looking for.&quot; sim_user_feedback:</td></tr><tr><td></td><td>Selection: [FailureReflection]</td><td>&quot;Thanks. Yeah, I&#x27;m more into movies that are fun to watch — action, adventure, maybe with a bit of comedy thrown in.</td></tr><tr><td>3</td><td></td><td>Not too serious or dark.&quot;</td></tr><tr><td></td><td>Expand: fully_expanded</td><td>sim_assistant_response (GenreInquiry): &quot;Got it! Just to be sure, do you prefer modern action films or do you enjoy older classics too?&quot;</td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td>sim_user_feedback:</td></tr><tr><td></td><td>Reward: 10.5</td><td>&quot;I&#x27;m open to both, as long as they&#x27;re fast-paced and entertaining.&quot;</td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td>sim assistant response (ItemRecommendation):</td></tr><tr><td></td><td></td><td>&quot;In that case, you might enjoy “The Man from U.N.C.L.E.&quot; (2015). It&#x27;s stylish, full of clever action sequences, and has</td></tr><tr><td></td><td></td><td>a good dose of humor and charm.&quot;</td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td>sim_user_feedback:</td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td>&quot;Hmm, I&#x27;ve heard about it. Sounds decent, but I&#x27;ll have to see a bit more before deciding.&quot;</td></tr><tr><td></td><td>Possible actions:</td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td>1. FailureReflection: mcts-score: 0.66666667, llm-score: 0.44444444</td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td>4</td><td>2. ItemRecommendation: mcts-score: 0.33333333, llm-score: 0.55555556</td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td>Combined Scores:</td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td>FailureReflection: 0.6, ItemRecommendation: 0.4</td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td>Best_action: FailureReflection</td><td></td></tr></table>

![](images/b0e4f65672bddad44e43e1eff0af898ebd7ce4dc48c67a323a0eb30717f18e21.jpg)  
Table 14: Prompt for Feeling Inference of User Simulator.

![](images/44d08aa8d1734133e38e719c4972190504f8c7e40430aac38b6c9aebeb1b219c.jpg)  
Table 15: Prompt for Insight Generation of User Simulator.

![](images/35372e9add0c7ec5e70159c5bce95ba3a7fc0c0f27de04ec9e7526e2ba58d052.jpg)  
Table 16: Prompt for Response Generation of User Simulator.

![](images/8d0d0315dd64b8c26cda26952c2e3c5ed2083c6d4b0f7fac9a609fa4f8ba35d1.jpg)  
Figure 7: LLM Evaluation Prompt for CGE<sup>2</sup>

![](images/2bdb058da85923cdba02335336ace4cd6d326bd5d9ace65a3f8aff00e6b612ed.jpg)  
Figure 8: LLM Scoring Prompt of DREAMS