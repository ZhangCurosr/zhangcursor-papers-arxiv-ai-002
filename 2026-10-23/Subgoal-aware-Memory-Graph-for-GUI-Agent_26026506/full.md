# Subgoal-aware Memory Graph for GUI Agent

Dezhao Luo and Daniel Beechey and Derek Yuen and Weilin Luo\* Huawei Noah’s Ark Lab

## Abstract

GUI agents aim to automate complex user tasks by interacting with mobile interfaces, but effective decision-making in long-horizon settings requires reusing prior experience rather than relying solely on the environment observations. Existing memory-based methods typically store experience either at the singleaction level or as entire trajectories. Singlestep actions are often too fine-grained to carry meaningful semantics, since the success of an individual action does not necessarily imply progress towards the overall goal. In contrast, whole trajectories are often too coarse-grained and difficult to reuse across different tasks.

We propose a subgoal-aware memory graph that distils past trajectories into structured, reusable evidence at the level of subgoals. We define a subgoal as consecutive actions sharing a single intent, the granularity at which progress toward the task becomes meaningful. To reduce redundancy and support crosstask reuse, we organise subgoal experiences as nodes in a graph, where related memories can be merged, extended, and retrieved based on the environment observation. On SPA-Bench, our method improves success rate by 11.11% over the baseline.

## 1 Introduction

GUI agents aim to complete complex user goals by interacting with graphical user interfaces, from web pages (Zheng et al., 2024; Deng et al., 2023; Zhou et al., 2024) to mobile phone screens, such as ordering items on a shopping app, setting up an alarm, or navigating settings (Zhang et al., 2025; Wang et al., 2024a; Hong et al., 2024; Rawles et al., 2023, 2025). In benchmarks like SPA-Bench (Chen et al., 2025), a single task can require more than 30 GUI actions across several screens. In such long-horizon settings, relying solely on the environment observation is insufficient because many subgoals recur across tasks and apps, and without reusing prior interaction experience, an agent must re-explore the same workflows from scratch (Xie et al., 2025; Zhu et al., 2026).

A straightforward solution is to equip the agent with an episodic memory it can query at inference time (Packer et al., 2023; Park et al., 2023). Existing memory-based methods, however, typically store experience either at the single-action level (Xie et al., 2025) or as entire trajectories (Zhu et al., 2026), and both granularities are problematic on mobile GUIs. Single-step actions are too finegrained to carry semantics on task progress, since whether a tap or swipe “succeeds” locally tells the agent little about whether the overall goal is being approached. Whole trajectories, on the other hand, are too coarse-grained, since a full episode, as stored in representations such as HyMEM (Zhu et al., 2026), is tied to a specific task path, making it rarely reusable across tasks that share intermediate steps but diverge in specifics.

We argue for an intermediate abstraction level, namely subgoals, which we define as contiguous sequences of actions that jointly express a coherent sub-intent. Unlike raw actions, subgoals reflect progress toward the goal. As illustrated in Figure 1, the same subgoal can either succeed or fail depending on the underlying behaviour, a distinction that is invisible at the single-action level. Unlike whole trajectories, subgoals transfer naturally across related tasks, since many tasks share the same underlying subgoals.

Building on this granularity, we organise past experience into a subgoal-level memory graph. Each raw trajectory is first segmented into a chain of subgoals. Rather than storing the full action sequence, we represent each subgoal as a transition between its boundary visual states (the starting and ending screenshots), annotated with a subgoal de-

Repeatedly scrolled through the app notification list looking for Google Play Store.

![](images/ec44aa772d01d7fdde3fbc431ceffffdb4858343707b5d246300e27b40411b97.jpg)

![](images/9c0ed1ccbdf85adc25acda495fe41699a1e490bb7793b9eb414a3499b10965be.jpg)  
Figure 1: The same subgoal can succeed (left) or fail (right) depending on the underlying behaviour. Subgoal granularity exposes progress toward the user goal that is invisible at the single-action level.

scription, a success label, a behaviour summary of the underlying actions, and an effect description obtained by comparing the boundary states. Boundary states are then embedded and merged across trajectories when visually similar, so that each subgoal becomes a directed edge between two shared memory nodes. The resulting graph consolidates reusable visual states and the semantic transitions between them, and can be incrementally extended with new trajectories.

On the SPA-Bench (Chen et al., 2025) English single-app level-3 split, our subgoal-aware memory graph achieves an improvement of 11.11% over a memory-free baseline with the same Qwen3-VL-8B-Instruct (Bai et al., 2025) backbone.

## 2 Related Work

GUI agents. App agents interpret naturallanguage goals and operate smartphones through actions such as tapping, scrolling, and text entry (Rawles et al., 2023, 2025; Wang et al., 2024b; Luo et al., 2025; Beechey et al., 2026). One line of work fine-tunes vision-language models to obtain end-to-end GUI agents (Hong et al., 2024; Qin et al., 2025; Tang et al., 2025; Christianos et al., 2024), achieving strong in-distribution accuracy at the cost of substantial training. A complementary line of work keeps the underlying LLM frozen. AppAgent (Zhang et al., 2025) maintains a handcurated, per-app knowledge base, whereas Mobile-Agent (Wang et al., 2024a) introduces visual perception tools to predict actions. Our work targets this frozen-LLM setting, where the agent’s capabilities are shaped by what is supplied through its context rather than by weight updates.

Memory for GUI agents. Memory mechanisms aim to give GUI agents access to relevant past experience at inference time, so that decisions are informed by what worked or failed in earlier rollouts. Existing memory designs for GUI agents cluster at two representational granularities. GUI-Explorer (Xie et al., 2025) mines per-step (observation, action, outcome) triples through autonomous exploration, and such single-action units carry little information on task progress, and a memory of them quickly accumulates redundant entries without exposing reusable structure. HyMEM (Zhu et al., 2026) organises continuous trajectory embeddings into a graph and EchoTrail-GUI (Li et al., 2025) retrieves whole successful episodes as incontext demonstrations. However, full trajectories are tied to the path of a single task. Our work targets the intermediate subgoal granularity that these designs leave underexplored. Subgoals are contiguous action segments that mark progress toward task completion and recur across different tasks. We organise them in a graph whose nodes are shared visual states, rather than per-task or per-action entries.

## 3 Method

Our goal is to equip a GUI agent with a memory that distils past interaction experience into reusable, subgoal-level guidance. Let a trajectory τ = $\left( s _ { 0 } , a _ { 0 } , s _ { 1 } , a _ { 1 } , \ldots , a _ { n - 1 } , s _ { n } \right)$ denote a sequence alternating between GUI screenshots $s _ { i }$ and executed actions $a _ { i }$ . Given a corpus $\mathcal { T } = \{ \tau _ { 1 } , . . . , \tau _ { m } \}$ of such trajectories, we construct a memory graph M that the agent can query during task execution: given the environment screenshot $s _ { q } ,$ , the memory returns a small set of subgoal-level guidance items that are appended to the agent’s prompt. We instantiate $\mathcal { M }$ as a directed graph whose nodes correspond to clusters of visually similar GUI states and whose edges encode subgoals observed in past trajectories, each edge connecting the state at which a subgoal began to the state at which it ended. The construction and retrieval pipeline comprise three stages: subgoal extraction (Section 3.1), graph construction (Section 3.2), and memory retrieval (Section 3.3).

![](images/797375b83a0b248a60e7b93143693f18eb0a0497f62159f2f0b01efc78a3216c.jpg)  
Figure 2: Overview of the proposed subgoal-aware memory framework. Past trajectories are segmented into annotated subgoals and fused into a graph over shared visual states.

## 3.1 Subgoal Extraction and Annotation

Segmentation. We segment each trajectory into a chain of subgoals, where a subgoal spans one or more consecutive actions that jointly realise a coherent sub-intent, for example, “open the search bar and submit a query” or “add the selected pair ofsunglasses to the basket”. Segmentation is performed by an LLM that takes the full action sequence as input and returns segment boundaries. We keep the first and last screenshots of each segment as its visual anchors and discard the intermediate frames, as the boundary screenshots already capture the start and end states relevant for downstream retrieval.

Subgoal metadata. Each extracted subgoal $g \in$ $\mathcal { G }$ is represented as a typed tuple

$$
g = \langle s ^ { \mathrm { f r o m } } , s ^ { \mathrm { t o } } ;
$$

text, effect, behaviour, success ⟩

whose fields are populated as follows.

$s ^ { \mathrm { f r o m } } , s ^ { \mathrm { t o } } ;$ the boundary screenshots, used as visual anchors during graph construction.

• text: a natural-language summary of the subgoal’s intent.

• effect: a one-sentence description of the observed visual change from $s ^ { \mathrm { f r o m } }$ to $s ^ { \mathrm { t o } }$

• behaviour: a concise summary of the underlying action sequence, highlighting the dominant interaction pattern.

• success ∈ {yes, no, uncertain}: a subgoallevel success label assigned by an LLM judge.

## 3.2 Subgoal-aware Memory Graph Construction

Given the annotated subgoals from Section 3.1, we now organise them into a persistent, queryable memory structure. The goal is to support behaviour reuse across tasks: when the agent encounters a state it has seen before, it should be able to retrieve which subgoals were previously attempted from that state, whether they succeeded, and through what action behaviour. We cast this as a directed graph whose nodes are GUI states and whose edges are subgoal transitions. Construction proceeds as follows: (1) merging visually similar screenshots into shared nodes so that experience accumulates rather than fragments; (2) attaching each subgoal as a directed edge carrying its full annotation; and (3) deduplicating redundant edges within each node to keep the graph compact.

Environment state merging. We map the boundary screenshots of each annotated subgoal, $s ^ { \mathrm { f r o m } }$ and $s ^ { \mathrm { t o } }$ , into a shared memory-node space according to visual similarity. For each such screenshot $s ,$ we compute its image embedding and compare it against the existing memory nodes. If the nearest node $m _ { j }$ satisfies $\cos ( s , m _ { j } ) \geq \tau$ , where $\cos ( \cdot , \cdot )$ denotes cosine similarity between image embeddings, the screenshot is absorbed into $m _ { j }$ . Otherwise, a new memory node is instantiated. Each node maintains its constituent screenshots together with a running mean embedding used for subsequent matching.

Directed subgoal edges. After environment state merging, each annotated subgoal becomes a directed edge from its source memory node to its target memory node. The edge stores the subgoal intent, the observed visual effect between the endpoint screenshots, the success label, and the behaviour pattern of the executed action sequence. Crucially, the graph preserves both successful and failed attempts from each state, so retrieval can later provide failed alongside successful examples.

Build-time deduplication. To keep the memory compact as more trajectories are added, we remove repeated outgoing subgoals within each source node. Two outgoing subgoals are treated as duplicates only when they point to the same target node, carry the same success label, and describe the same intent, i.e., the cosine similarity between text embeddings of their intent descriptions exceeds θ. When a duplicate is detected, the later edge is discarded. Subgoals that differ in target, outcome, or intent are kept as separate evidence.

## 3.3 Memory Retrieval

At each decision step, retrieval is triggered with the current screenshot $s _ { q } .$ . We embed $s _ { q }$ and compare it against the embedding maintained at each memory node, and keep the top-k nearest nodes by cosine similarity. For each retrieved node, we collect all of its outgoing subgoal edges, and both successful and failed subgoals are returned, each tagged with its outcome, so that the agent can avoid known failure modes as well as imitate known successes. The retrieved subgoals are then formatted into a short memory block and added to the agent’s prompt.

## 4 Experiments

## 4.1 Setup

Benchmark and Environment. We run experiments on cloud-based Android phone instances controlled through ADB. We use the English singleapp level-3 split of SPA-Bench (Chen et al., 2025), the more difficult tier of the benchmark and a small number of apps that cannot run on cloud phone instances are excluded from evaluation. This leaves 13 tasks in our evaluation. We use Qwen3-VL-8B-Instruct (Bai et al., 2025) as the base model. Both image and text embeddings used in memory construction and retrieval are computed by Qwen3- VL-Embedding-8B (Li et al., 2026). We set both the visual similarity threshold τ and the text similarity threshold θ to 0.9, and retrieve the top k = 1 nearest memory node at each decision step. In our experiment, memory construction requires 4 GPU-hours, and each retrieval incurs an additional latency of approximately 0.5 seconds.

<table><tr><td>Memory</td><td>Structure</td><td>SR (%)</td></tr><tr><td>None</td><td>一</td><td>34.19</td></tr><tr><td>Subgoal</td><td>Flat</td><td>30.77</td></tr><tr><td>Action</td><td>Graph</td><td>38.46</td></tr><tr><td>Task</td><td>Graph</td><td>36.75</td></tr><tr><td>Subgoal (Ours)</td><td>Graph</td><td>45.30</td></tr></table>

Table 1: Per-attempt success rate. Our subgoal-graph memory outperforms both the no-memory baseline and flat / alternative-granularity variants.

## 4.2 Results

Table 1 reports the per-attempt success rate (SR), averaged over three independent runs. With the same backbone and setup, our subgoal-graph memory improves success by 11.11% over the memoryfree baseline, suggesting that retrieving past experience is useful for long-horizon mobile tasks. Moreover, neither action-level nor task-level memory yields meaningful gains over the baseline.

We further ablate the graph structure by replacing it with a flat memory store, where subgoals from different environments are not organised through shared visual-state nodes. This reduces SR to 30.77%. We argue that the graph structure allows redundant memories to be merged across trajectories, keeping the retrieved context precise.

## 5 Conclusion

We proposed a subgoal-aware memory graph for mobile GUI agents. The core idea is to store past experience at the level of subgoals, rather than raw actions or whole trajectories. To make these subgoals reusable across trajectories, we fuse visually similar environment states into shared memory nodes and attach subgoal transitions between them. This allows the agent to see what has been tried from similar states and what outcome each subgoal produced. At inference time, the agent retrieves subgoals from the graph and uses them as guidance for the current task. This provides a simple and practical memory structure for mobile interaction.

## Limitations

Our method assumes that GUI environments can be approximated by a discrete state space, such that screenshots can be meaningfully clustered into a compact set of shared memory nodes. This assumption is reasonable for typical mobile applications with relatively stable layouts, but becomes limited in highly dynamic interfaces, such as video players, where consecutive states change continuously, and visual similarity is a noisier proxy for semantic equivalence.

## References

Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, and 1 others. 2025. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631.

Daniel Beechey, Derek Yuen, Jianheng Liu, Dezhao Luo, Tiantian He, Weilin Luo, Jun Wang, and Kun Shao. 2026. Darwin mobile agent: A roadmap for self-evolution. Preprint, arXiv:2606.20622.

Jingxuan Chen, Derek Yuen, Bin Xie, Yuhao Yang, Gongwei Chen, Zhihao Wu, Li Yixing, Xurui Zhou, Weiwen Liu, Shuai Wang, and 1 others. 2025. Spabench: A comprehensive benchmark for smartphone agent evaluation. In The Thirteenth International Conference on Learning Representations.

Filippos Christianos, Georgios Papoudakis, Thomas Coste, Jianye Hao, Jun Wang, and Kun Shao. 2024. Lightweight neural app control. arXiv preprint arXiv:2410.17883.

Xiang Deng, Yu Gu, Boyuan Zheng, Shijie Chen, Samuel Stevens, Boshi Wang, Huan Sun, and Yu Su. 2023. Mind2web: Towards a generalist agent for the web. In Advances in Neural Information Processing Systems (NeurIPS).

Wenyi Hong, Weihan Wang, Qingsong Lv, Jiazheng Xu, Wenmeng Yu, Junhui Ji, Yan Wang, Zihan Wang, Yuxiao Dong, Ming Ding, and 1 others. 2024. Cogagent: A visual language model for gui agents. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 14281–14290.

Mingxin Li, Yanzhao Zhang, Dingkun Long, Keqin Chen, Sibo Song, Shuai Bai, Zhibo Yang, Pengjun Xie, An Yang, Dayiheng Liu, and 1 others. 2026. Qwen3-vl-embedding and qwen3-vlreranker: A unified framework for state-of-the-art multimodal retrieval and ranking. arXiv preprint arXiv:2601.04720.

Runze Li, Yuwen Zhai, Bo Xu, Liwu Xu, Nian Shi, Wei Zhang, Ran Lin, and Liang Wang. 2025. EchoTrail-GUI: Building actionable memory for GUI agents

via critic-guided self-exploration. arXiv preprint arXiv:2512.19396.

Dezhao Luo, Bohan Tang, Kang Li, Georgios Papoudakis, Jifei Song, Shaogang Gong, Jianye Hao, Jun Wang, and Kun Shao. 2025. ViMo: A generative visual GUI world model for app agent. arXiv preprint arXiv:2504.13936.

Charles Packer, Vivian Fang, Shishir\_G Patil, Kevin Lin, Sarah Wooders, and Joseph\_E Gonzalez. 2023. Memgpt: towards llms as operating systems.

Joon Sung Park, Joseph O’Brien, Carrie Jun Cai, Meredith Ringel Morris, Percy Liang, and Michael S Bernstein. 2023. Generative agents: Interactive simulacra of human behavior. In Proceedings of the 36th annual acm symposium on user interface software and technology, pages 1–22.

Yujia Qin, Yining Ye, Junjie Fang, Haoming Wang, Shihao Liang, Shizuo Tian, Junda Zhang, Jiahao Li, Yunxin Li, Shijue Huang, and 1 others. 2025. UI-TARS: Pioneering automated GUI interaction with native agents. arXiv preprint arXiv:2501.12326.

Christopher Rawles, Sarah Clinckemaillie, Yifan Chang, Jonathan Waltz, Gabrielle Lau, Marybeth Fair, Alice Li, William Bishop, Wei Li, Folawiyo Campbell-Ajala, Daniel Toyama, Robert Berry, Divya Tyamagundlu, Timothy Lillicrap, and Oriana Riva. 2025. AndroidWorld: A dynamic benchmarking environment for autonomous agents. In International Conference on Learning Representations (ICLR).

Christopher Rawles, Alice Li, Daniel Rodriguez, Oriana Riva, and Timothy Lillicrap. 2023. Android in the Wild: A large-scale dataset for Android device control. In Advances in Neural Information Processing Systems (NeurIPS) Datasets and Benchmarks Track.

Bohan Tang, Dezhao Luo, Jianheng Liu, Jingxuan Chen, Shaogang Gong, Jianye Hao, Jun Wang, and Kun Shao. 2025. Beyond syntax: Action semantics learning for app agents. arXiv preprint arXiv:2506.17697.

Junyang Wang, Haiyang Xu, Jiabo Ye, Ming Yan, Weizhou Shen, Ji Zhang, Fei Huang, and Jitao Sang. 2024a. Mobile-agent: Autonomous multi-modal mobile device agent with visual perception. arXiv preprint arXiv:2401.16158.

Taiyi Wang, Zhihao Wu, Jianheng Liu, Jianye Hao, Jun Wang, and Kun Shao. 2024b. Distrl: An asynchronous distributed reinforcement learning framework for on-device control agents. arXiv preprint arXiv:2410.14803.

Bin Xie, Rui Shao, Gongwei Chen, Kaiwen Zhou, Yinchuan Li, Jie Liu, Min Zhang, and Liqiang Nie. 2025. Gui-explorer: Autonomous exploration and mining of transition-aware knowledge for gui agent. In Proceedings of the 63rd Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 5650–5667.

Chi Zhang, Zhao Yang, Jiaxuan Liu, Yanda Li, Yucheng Han, Xin Chen, Zebiao Huang, Bin Fu, and Gang Yu. 2025. Appagent: Multimodal agents as smartphone users. In Proceedings of the 2025 CHI Conference on Human Factors in Computing Systems, pages 1–20.

Boyuan Zheng, Boyu Gou, Jihyung Kil, Huan Sun, and Yu Su. 2024. Gpt-4v (ision) is a generalist web agent, if grounded. arXiv preprint arXiv:2401.01614.

Shuyan Zhou, Frank F Xu, Hao Zhu, Xuhui Zhou, Robert Lo, Abishek Sridhar, Xianyi Cheng, Tianyue Ou, Yonatan Bisk, Daniel Fried, and 1 others. 2024. Webarena: A realistic web environment for building autonomous agents. In International Conference on Learning Representations, volume 2024, pages 15585–15606.

Sibo Zhu, Wenyi Wu, Kun Zhou, Stephen Wang, and Biwei Huang. 2026. Hybrid self-evolving structured memory for GUI agents. arXiv preprint arXiv:2603.10291.