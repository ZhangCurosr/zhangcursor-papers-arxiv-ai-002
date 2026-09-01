# Scaffolding Foundation Models into Physical-World Agents Pushes the Frontier of Long-Horizon Navigation

Zixing Lei<sup>1,3,∗</sup>, Gengze Zhou<sup>4,∗</sup>, Xiong-Hui Chen<sup>2,∗</sup>, Jiazhao Zhang<sup>5</sup>, Yiyang Huang<sup>2</sup>, Hang Yin<sup>6</sup>, Haoqi Yuan<sup>5</sup>, Qi Wu<sup>4</sup>, Weixin Li<sup>3</sup>, Siheng Chen<sup>1</sup> <sup>1</sup>Shanghai Jiao Tong University <sup>2</sup>Qwen Team, Alibaba Inc <sup>3</sup>Zhongguancun Academy <sup>4</sup>AIML, Adelaide University <sup>5</sup>Peking University <sup>6</sup>Tsinghua University <sup>∗</sup>Equally contributed.

![](images/0339df0cd44db79a74ccef5668502dcc7f0588807bcc929c074c2f7470b5a6fb.jpg)

![](images/9cbfbe0fbeecee51a8e3aae41a7f0d26939a19775a999ab615620c03789876e4.jpg)

Figure 1: (a) A VLM directly controlling short waypoint actions may struggle with long-horizon progress. Coupling it with an NFM lets the VLM issue semantic navigation sub-goals while the NFM performs closed-loop execution. (b) For an EQA question, the VLM converts evidence-seeking decisions into semantic navigation calls that search relevant rooms, rule out unlikely regions, and localize the target evidence. (c) Table 1 values compare NavMCP with FAST-EQA and the corresponding benchmark-specific baseline: ToolEQA on HM-EQA and Express-Bench, and Memory-EQA on MT-HM3D.

## Abstract

Long-horizon physical-world agents must reason over distant task goals while grounding each decision into reliable closed-loop behavior. These capabilities are split across today’s foundation models: vision-language models (VLMs) can infer missing information and adapt high-level plans, but repeatedly grounding their decisions into navigation actions remains brittle and inefficient, whereas navigation foundation models (NFMs) execute semantic goals robustly but typically operate as bounded episodes without persistent task-level reasoning. We argue that these limitations are complementary and introduce NavMCP, an agentic scaffolding framework that couples a VLM reasoning agent with an NFM executor for long-horizon physical-world exploration. The VLM decides what evidence to seek, where to search, and when to stop, while the NFM grounds each semantic sub-goal into closed-loop navigation. NavMCP structures this collaboration through three channels: intent translates evidence needs into semantic navigation calls, observation converts complete rollouts into source-grounded trajectory evidence, and memory accumulates findings, negative evidence, and unresolved goals across calls. Together, these channels turn isolated navigation rollouts into persistent embodied interaction without retraining either foundation model. We instantiate NavMCP on Embodied Question Answering (EQA), a demanding setting in which an agent must explore for visual evidence before answering situated questions. NavMCP achieves stateof-the-art performance on HM-EQA, MT-HM3D, and EXPRESS-Bench; under matched agent and executor backbones, it outperforms an episodic interface by 14.9 percentage points on HM-EQA. On a Unitree Go2, NavMCP reaches 78.3% success, with its margin over the strongest baseline growing from 10 to 45 points as the task horizon increases. These results demonstrate the potential of scaffolding complementary foundation models into long-horizon physical-world agents.

## 1 Introduction

Long-horizon physical-world agency requires two capabilities that today’s foundation models provide separately. An agent must reason over distant goals and changing evidence while continuously grounding its decisions into reliable actions. Vision-language models (VLMs) can infer what is missing, decompose goals, and adapt high-level plans, but repeatedly translating such reasoning into navigation actions becomes brittle and inefficient over long horizons. Navigation foundation models (NFMs) are strong at the complementary problem: they map egocentric observations and semantic goals to closed-loop actions or waypoints, yet typically execute one bounded goal at a time without deciding what to seek next or retaining task state across episodes. The central challenge is therefore not simply to build a stronger reasoner or navigator, but to organize their complementary capabilities into persistent physical-world agency.

Recent NFMs such as SAME, Uni-NaVid, NavFoM, ABot-N0/N1, and Qwen-RobotNav unify instruction following, object and point-goal navigation, point-of-interest search, and target following (Zhou et al., 2025; Zhang et al., 2025; 2026b; Chu et al., 2026; Zhang et al., 2026c; Gong et al., 2026). Their strong navigation performance and zero-shot transfer make them powerful embodied executors, while VLM agents provide the open-ended reasoning needed to select and revise goals over a longer task horizon. This complementarity suggests a natural division of labor: the VLM decides what evidence to seek, where to search, and when to stop, while the NFM determines how to ground each semantic sub-goal into closed-loop motion. We call this paradigm NavMCP: the NFM extends the VLM’s action horizon, and the VLM agent extends the NFM’s reasoning horizon. Rather than asking either model to solve the full embodied task alone, NavMCP scaffolds their specialized capabilities into a single long-horizon agent.

Realizing NavMCP requires more than wrapping a navigator in a conventional tool call. Under an episodic tool interface, the agent submits one navigation instruction and receives only terminal status and a final observation, creating an episodic interface gap with multi-step evidence acquisition. Mismatch between instruction and intent. A route-level instruction does not capture the agent’s evidence need, search mode, budget, and constraints. Intermediate observation loss. Useful evidence may appear anywhere along a rollout—an object passed in a hallway or a room glimpsed through a doorway—but a terminal-only return discards everything observed before the final pose. Lack of cross-call accumulation. Independent calls do not retain searched regions, ruled-out hypotheses, uncertain findings, or negative outcomes. Without addressing these gaps, an NFM remains an episodic tool rather than part of a long-horizon agent.

NavMCP closes the episodic interface gap with an evidence-centric scaffold organized into three channels (Figure 1). The intent channel translates an evidence need into a semantic navigation call with a mode, sub-goal, budget, and constraints, allowing the agent to specify what to seek without micromanaging low-level control. The observation channel converts a complete rollout into source-grounded journey evidence containing keyframes, salient observations, uncertainty, and unexplored-region cues. The memory channel carries checked regions, negative evidence, unresolved goals, and answer support across calls. Together, these channels make NFM execution controllable, inspectable, and persistent: isolated navigation episodes become reusable embodied experience for subsequent reasoning and action.

We instantiate and evaluate NavMCP on Embodied Question Answering (EQA), where an agent must explore a physical environment, acquire visual evidence, and answer a situated question (Das et al., 2018; Gordon et al., 2018). EQA provides a demanding test of long-horizon agency because distant or initially invisible evidence requires the agent to form search hypotheses, visit multiple regions, accumulate positive and negative observations, and decide when the evidence is sufficient to answer. Across HM-EQA, MT-HM3D, and EXPRESS-Bench, NavMCP improves over the strongest reported results; under matched reasoning and evaluation conditions, it leads FAST-EQA by 10.5 percentage points on HM-EQA. With the agent and navigator fixed, replacing NavMCP with an episodic interface costs 14.9 points, directly measuring the value of persistent scaffolding. On a Unitree Go2, NavMCP reaches 78.3% success in real multi-room search, and its margin over the strongest baseline grows from 10 to 45 points as the task horizon increases. These results show that scaffolding VLM reasoning and NFM execution is an effective route to long-horizon physical-world agents.

## Our contributions are:

1. We introduce NavMCP, an agentic scaffolding framework that organizes the complementary capabilities of a VLM reasoning agent and an NFM executor into a long-horizon physical-world agent.

2. We develop an evidence-centric three-channel design that communicates navigation intent, grounds complete rollouts as reusable observations, and maintains task state across calls, thereby extending both the VLM’s action horizon and the NFM’s reasoning horizon.

3. We validate NavMCP on three simulated EQA benchmarks and a Unitree Go2, achieving state-ofthe-art results across all three benchmarks and demonstrating gains that grow with the real-world task horizon.

## 2 Related Work

Embodied Question Answering. EQA, introduced by Das et al. (2018), requires navigation to answer situated questions. Recent systems improve exploration, memory, refinement, scene representation, and multi-agent search using frontiers (Ren et al., 2024; Zhai et al., 2025a), memory guidance (Lu et al., 2026), goal-directed policies (Zhang et al., 2026a), scene graphs (Saxena et al., 2024), and distributed LLM agents (Patel et al., 2024); benchmarks now span open-vocabulary, free-form, and urban settings (Majumdar et al., 2024; Jiang et al., 2025; Zhao et al., 2025). NavMCP instead uses an NFM as the physical layer while preserving trajectory evidence.

Vision-Language Navigation and Navigation Foundation Models. VLN has advanced from instructionconditioned navigation (Anderson et al., 2018) through augmentation, pretraining, transformers, video planning, and LLM reasoning (Fried et al., 2018; Hao et al., 2020; Chen et al., 2022; Hong et al., 2021; Zhang et al., 2024; Zhou et al., 2024b;a). Semantic approaches add commonsense and language, belief, or frontier maps (Zhou et al., 2023; Huang et al., 2023a; Zhou et al., 2026; Yokoyama et al., 2024); navigation foundation models and generic policies instead unify tasks through vision-language-action interfaces (Zhou et al., 2025; Zhang et al., 2025; 2026b; Chu et al., 2026; Zhang et al., 2026c). On VLNVerse, Qwen-RobotNav-8B achieves 63.75% SR and 57.93% SPL under fine-grained instructions, and 46.59% SR and 41.54% SPL under coarse-grained instructions (Zhang et al., 2026c). NavMCP targets the episodic tool interface, where one instruction produces one terminal return, and turns navigation rollouts into reusable evidence.

Tool-Augmented Embodied Agents and Context Management. LLMs plan embodied tasks through affordance grounding, feedback, code, planning, and skills (Ahn et al., 2023; Huang et al., 2023b; Liang et al., 2023; Rana et al., 2023; Wang et al., 2024). Reasoning-acting and tool-use methods (Yao et al., 2023; Shinn et al., 2023; Schick et al., 2023) extend to EQA and manipulation (Zhai et al., 2025b; Lei et al., 2026), while long-horizon agents use memory and prompt compression (Packer et al., 2023; Jiang et al., 2023). These systems support tool selection and context management but typically reduce each call to a terminal outcome; NavMCP instead preserves source-grounded en-route and negative evidence across calls.

## 3 Method

## 3.1 System Formulation

We formulate EQA with a navigation foundation model as an agent–executor interface. Given a question q and an initial observation, a VLM agent must gather sufficient visual evidence to produce an answer y, deciding where to search and when to stop. The agent maintains task hypotheses, evidence needs, and context, while a navigation executor handles one semantic sub-goal at a time. We instantiate the executor with Qwen-RobotNav (Zhang et al., 2026c), which maps the agent’s natural-language sub-goal, current egocentric observation, and navigation state to a short-horizon waypoint trajectory. At turn t, the agent reads $C _ { t }$ and either proposes y or emits an evidence need $u _ { t } ;$ in the latter case, the executor runs one rollout and returns trajectory-level observations. This division makes repeated rollouts useful without retraining the navigator or asking it to answer directly: the executor supplies grounded experience, while the agent performs evidence reasoning and answer commitment.

The episodic interface gap defined in Section 1 structures this interface. The agent must communicate evidence-seeking intent without low-level control, the executor’s rollout must be converted into evidence useful for answering rather than reduced to terminal feedback, and salient evidence must persist across repeated calls. NavMCP addresses these requirements with three channels: intent, observation, and memory, each closing one mismatch. Figure 2 illustrates the end-to-end system, and Figure 3 expands the three scaffold channels. See details in Algorithm 1 of the appendix.

## 3.2 Intent Channel: From Evidence Needs to Navigation Calls

The intent channel turns an agent’s evidence need into one navigation episode, closing the mismatch between instruction and intent. This channel is needed because the agent reasons about missing information, while the navigation foundation model acts through embodied rollouts and consumes route-level instructions. The agent should therefore specify where or what to search for, but it should not issue simulator actions or micromanage the path. At turn t, the agent emits a semantic navigation call that specifies a mode, a natural-language sub-goal, a step budget, and optional constraints; the schema is given in Section G. The key abstraction is that the sub-goal and constraints specify intent rather than controls. For example, the agent can request “search the kitchen counters for a coffee maker” rather than prescribe actions.

![](images/d9b1ac5b6292c230bb1bea005910d14db2ad0ab3564cb7f47fb5c71938ecc762.jpg)  
Figure 2: System overview of NavMCP. Given an EQA question and observation, a VLM agent uses NavMCP to call a navigation foundation model as an embodied executor. The navigation foundation model receives a natural-language navigation sub-goal and the current egocentric observation and state, and outputs a short-horizon waypoint trajectory. The resulting trajectory observations are then converted back into evidence for the VLM agent.

The executor adapter turns each intent call into an instruction-following episode for the navigation foundation model. We expose two modes that match the two common operating regimes used to train and evaluate navigation foundation models. navigate\_to\_object corresponds to Object Navigation, where the instruction names an object or category to find. navigate\_by\_instruction corresponds to Vision-and-Language Navigation, where the instruction describes a route, region, or scene-level goal. This mode split gives the agent a simple way to choose between object-centric search and broader language-guided movement while keeping both modes backed by the same navigation executor. The executor observes only egocentric RGB views, its pose, and an online map built from the explored trajectory, so the interface does not assume privileged access to unseen geometry or object locations. Because the agent only depends on the call-and-return schema, the navigation model can be replaced by another executor if it supports the same intent modes and return format, enabling the executor comparisons in our experiments.

## 3.3 Observation Channel: From Trajectories to Evidence

The observation channel translates an embodied rollout into agent-facing evidence, closing intermediate observation loss. A terminal success flag or final pose is too weak for evidence-seeking tasks. Useful observations may occur anywhere along the route, including objects passed en route, rooms that were partially searched, and exits that remain unexplored. For each navigation call, the executor returns a raw trajectory record containing rollout status, path information, final pose, and sampled keyframes; the detailed record format is given in Section G. We sample keyframes along the trajectory rather than only at the final pose because answer evidence can appear before the executor stops. The observation channel then uses a VLM-based trajectory summarizer to produce a grounded journey artifact covering sub-goal status, route context, observed objects, planning cues, and uncertainty. The artifact separates evidence for the current decision from cues for future navigation. Rooms and objects support answer reasoning, while spatial cues and unexplored exits help the agent decide where to search next.

The journey artifact is source-grounded to reduce unsupported trajectory claims. Each object or room mention is tied to a source keyframe and confidence label. The summarizer is instructed to report only visually observed evidence, mark uncertainty explicitly, and avoid turning absence from a single view into a definitive negative claim. This design preserves intermediate observations lost by standard navigation feedback and returns grounded evidence to memory.

## 3.4 Memory Channel: From Tool Returns to EQA Context State

The memory channel turns trajectory evidence into a compact EQA state for later reasoning turns, closing the lack of cross-call accumulation. Without this state, the agent must repeatedly infer what has already been searched from long raw tool traces. This is costly and brittle when earlier tool outputs are shortened under context limits. We represent the EQA context state as $C _ { t } = \left( H _ { t } , E _ { t } , U _ { t } \right)$ , where $H _ { t }$ is a compact interaction history, $E _ { t }$ is a source-grounded evidence ledger, and $U _ { t }$ stores unresolved goals for future planning. After each journey artifact z or auxiliary tool return, the agent updates $E _ { t }$ and $U _ { t }$ before the raw return can be compressed. The ledger stores positive observations, searched areas, uncertain findings, negative evidence, and support for candidate decisions, while the unresolved-goal state tracks missing evidence for later intent calls. Detailed ledger fields and update rules are given in Section G.

![](images/18ba976efaf0ee43254a8d550f424606e1e1cad2d841b2758da1bdfdf1b9937b.jpg)  
Figure 3: Three-channel NavMCP interface. Each channel closes one mismatch of the episodic interface gap: the intent channel resolves the mismatch between instruction and intent by mapping the agent’s evidence need into a structured navigation call, the observation channel prevents intermediate observation loss by converting the executor rollout into grounded trajectory evidence, and the memory channel enables cross-call accumulation by writing the evidence and remaining open goals into the next-turn EQA context state. These channels separate answer reasoning from low-level navigation while retaining visual evidence for later decisions.

Context maintenance follows a conservative compression policy. Raw tool returns are compressed only after salient information has been externalized into E or U with source references, so long contexts can be reduced around persistent evidence rather than raw traces. This policy makes compression safer because later reasoning depends on source-grounded ledger entries rather than on the full raw trace.

EQA instantiation. For EQA, NavMCP implements E as an evidence ledger for answering a question q with a grounded response y. The agent may answer only when the response is supported by the current observation, a journey artifact, a reviewed keyframe, or a notebook entry. Positive answers should cite what was observed and where it was observed. Negative answers should be backed by searched-area records covering likely locations for the target. The agent can request local verification through auxiliary perception tools such as panoramic inspection, open-vocabulary detection, close-up object inspection, and saved-keyframe review. These tools refine or verify ledger entries in our EQA instantiation; the protocol itself only requires source-grounded evidence updates. Reported results therefore measure cross-call evidence management within an episode rather than memorization of test environments across episodes.

## 4 Experiments

## 4.1 Experimental Setup

Evaluation questions. We organize the experiments around four questions, addressed in the following four subsections. (Q1) Does NavMCP improve overall EQA performance? (Q2) Do the VLM agent and NFM executor complement each other? (Q3) How does NavMCP close the episodic interface gap? (Q4) Does NavMCP transfer to real robots as task horizons grow?

Benchmarks. We evaluate on three EQA benchmarks that stress complementary aspects of evidence gathering. HM-EQA (Ren et al., 2024) contains 500 multiple-choice questions across 267 HM3D scenes and evaluates single-question embodied search. MT-HM3D (Zhai et al., 2025a) introduces multi-target questions that require cross-room comparison and reasoning over observations collected in multiple regions. EXPRESS-Bench (Jiang et al., 2025) contains 2,044 free-form questions and reports answer quality via LLM Score and exploration path quality via E<sub>path</sub>.

Baselines. We compare against EQA systems that instantiate different interfaces between reasoning and navigation: Explore-EQA (Ren et al., 2024) and Memory-EQA (Zhai et al., 2025a) use exploration and memory mechanisms, Graph-EQA (Saxena et al., 2024) uses structured scene representation, ToolEQA (Zhai et al., 2025b) augments reasoning with callable tools, FAST-EQA (Zhang et al., 2026a) uses faster goaldirected exploration, and Fine-EQA (Jiang et al., 2025) is the reported baseline for EXPRESS-Bench. For causal analysis, we replace the navigation foundation model with Random Walk or Frontier Exploration and use reactive, single-round, and no-agent variants.

Table 1: Cross-benchmark EQA performance. Acc. denotes answer accuracy (%). Norm. Agent Steps<sup>†</sup> is the normalized high-level budget fraction; the equivalent-step compensation in Equation (1) applies only to methods that invoke an NFM, while non-NFM baselines retain their reported step accounting. Baselines are reported results, with <sup>∗</sup> marking results reproduced by FAST-EQA. Our results are mean ± standard deviation over three independent runs. Cross-paper comparisons are contextual only.
<table><tr><td rowspan="2">Method</td><td colspan="2">HM-EQA</td><td colspan="2">MT-HM3D</td><td colspan="2">EXPRESS-Bench</td></tr><tr><td>Acc. (1)</td><td>Norm. Agen Steps+ (↓)</td><td>Acc. (↑)</td><td>Norm. Agen</td><td>LLM Score</td><td> $E _ { \mathrm { p a t h } }$ </td></tr><tr><td>Explore-EQA (Ren et al., 2024)</td><td>58.4</td><td>0.52</td><td>36.2*</td><td>Steps+ (↓) 0.64</td><td>(↑) 一</td><td>(↑)</td></tr><tr><td>Graph-EQA (Saxena et al., 2024)</td><td>63.5</td><td>0.20</td><td>45.6*</td><td>0.45</td><td>一</td><td>一 一</td></tr><tr><td>Memory-EQA (Zhai et al., 2025a)</td><td>61.4</td><td>0.40</td><td>43.1</td><td>0.41</td><td></td><td></td></tr><tr><td>Fine-EQA (Jiang et al., 2025)</td><td>56.0</td><td>0.54</td><td>一</td><td>一</td><td>63.95</td><td>25.58</td></tr><tr><td>3D-Mem (Yang et al., 2025)</td><td>50.4</td><td>0.63</td><td>一</td><td>一</td><td></td><td></td></tr><tr><td>ToolEQA (Zhai et al., 2025b)</td><td>62.3</td><td></td><td></td><td></td><td>65.77</td><td>25.82</td></tr><tr><td>FAST-EQA (Zhang et al., 2026a)</td><td>69.2</td><td>0.65</td><td>50.5</td><td>0.52</td><td>68.7</td><td>29.25</td></tr><tr><td>NavMCP (Ours)</td><td>76.7±0.1</td><td>0.15±0.01</td><td>54.4±1.5</td><td>0.19±0.01</td><td>79.27±0.44</td><td>33.96±0.75</td></tr></table>

Metrics and step accounting. Answer quality is primary: accuracy (Acc.) is the percentage of questions answered correctly. On EXPRESS-Bench, LLM Score rates free-form semantic correctness from 0–100, while $E _ { \mathrm { p a t h } }$ weights it by the ratio of the reference sufficient path length to the traveled distance, rewarding accurate and efficient exploration. For HM-EQA and MT-HM3D, we additionally report normalized equivalent agent steps. In our setup, outer-agent VLM reasoning and API latency dominate wall-clock time, so we use the number of outer-agent VLM invocations, H, as the primary efficiency cost. Under the direct-VLM-control protocol used by prior frontier baselines, one VLM decision can displace the agent by at most 3 m; thus, 3 m is the natural conversion unit rather than an arbitrary threshold. For methods that invoke an NFM, each outer-agent invocation counts as one base step, and each NFM rollout incurs an additional charge for every extra 3 m segment:

$$
n _ { \mathrm { e q } } = H + \sum _ { k = 1 } ^ { K } \mathrm { m a x } \left( 0 , \left\lceil \frac { L _ { k } } { 3 \mathrm { m } } \right\rceil - 1 \right) , \hat { n } _ { \mathrm { e q } } = \frac { n _ { \mathrm { e q } } } { N } .\tag{1}
$$

Here K is the number of NFM rollouts, $L _ { k }$ is the path length of NFM rollout $k ,$ and N is the scene-specific step budget. The compensation term is not applied to non-NFM baselines, whose reported normalized steps follow their original direct-control accounting. The metric measures high-level interaction and exploration rather than end-to-end computation; the full derivation and component-level accounting are given in Section G.

Implementation and evaluation protocol. The main results use Qwen3.6-Plus as the agent and Qwen-RobotNav (Zhang et al., 2026c) as the navigation foundation model executor. For controlled ablations, we use a fixed open-source Qwen3.5-397B-A17B agent unless noted otherwise and match episodes, initial states, budgets, and perception settings, so each comparison isolates the factor specified in its table caption. Cross-paper results in Table 1 are provided for context only; all causal claims rely on these controlled comparisons. Further implementation details and evaluation statistics are provided in Sections B and F.

## 4.2 Does NavMCP Improve Overall EQA Performance?

Following FAST-EQA (Zhang et al., 2026a) and common practice in prior EQA work, Table 1 uses each baseline’s latest reported result or its FAST-EQA reproduction when available. On HM-EQA, NavMCP reaches 76.7% accuracy, 7.5 percentage points above FAST-EQA. On MT-HM3D, it reaches 54.4%, a 3.9-point improvement. On EXPRESS-Bench, NavMCP obtains an LLM Score of 79.27 and an $E _ { \mathsf { p a t h } }$ of 33.96, improving over FAST-EQA by 10.57 and 4.71 points. Crucially, these gains come with substantially better efficiency: after compensating only the NFM rollouts for distance beyond the direct-control 3 m unit, NavMCP uses the lowest normalized agent-step fractions among reporting methods on HM-EQA and MT-HM3D (0.15 and 0.19), where outer-agent VLM calls dominate cost, and achieves the highest path efficiency on EXPRESS-Bench. Together, these results establish new state-of-the-art performance across all three EQA formats and show that coupling a reasoning VLM with a navigation foundation model is an effective and efficient paradigm for long-horizon EQA exploration. Because limited API and code availability prevents full alignment across papers, we next test the system-level advantage under matched conditions before characterizing agent, executor, and protocol contributions in Sections 4.3 and 4.4.

Table 2: Matched full-system comparison on HM-EQA. All methods use the same Qwen3.5-397B-A17B agent, episodes, initial states, and budget; ToolEQA additionally uses the same perception toolset as NavMCP. Methods retain their own pipelines and navigation interfaces. Values are mean ± standard deviation over three runs.
<table><tr><td>Configuration Acc. (%)</td></tr><tr><td>Explore-EQA (Ren et al., 2024) (re-evaluated)  $5 7 . 6 \pm 0 . 7$ </td></tr><tr><td>ToolEQA (Zhai et al., 2025b) (our reproduction)  $6 0 . 8 \pm 0 . 9$ </td></tr><tr><td>FAST-EQA (Zhang et al., 2026a) (our reproduction)  $6 3 . 5 \pm 1 . 4$ </td></tr><tr><td>NavMCP + Qwen-RobotNav-8B  $7 4 . 0 \pm 0 . 3$ </td></tr></table>

Table 3: VLM–NFM architecture ablation. Top: Qwen-RobotNav-8B is fixed while the upper layer changes. Bottom: the full Qwen3.5 agent and NavMCP harness are fixed while the executor changes. Executor variants are mean ± standard deviation over three runs.
<table><tr><td>Configuration Acc. (%)</td></tr><tr><td>Upper-level agent/harness capability (Qwen-RobotNav-8B fixed)</td></tr><tr><td>No Agent (VLM from initial view) 38.2</td></tr><tr><td>Single-Round Agent (one panorama) 58.4</td></tr><tr><td>Reactive Agent (fixed explore loop) 62.0</td></tr><tr><td>Full Agent (Qwen3.5-397B-A17B) 74.0</td></tr><tr><td>Full Agent (Qwen3.6-Plus) 76.7</td></tr><tr><td>Lower-level navigation capability (full Qwen3.5 agent fixed)</td></tr><tr><td>NavMCP + Random Walk  $6 0 . 9 \pm 3 . 1$ </td></tr><tr><td>NavMCP + Frontier Exploration  $6 5 . 3 \pm { 0 . 2 }$ </td></tr><tr><td>NavMCP + StreamVLN (Wei et al., 2026)  $6 9 . 3 \pm { 0 . 4 }$ </td></tr><tr><td>NavMCP + Qwen-RobotNav-4B  $7 3 . 3 \pm 0 . 3$ </td></tr><tr><td>NavMCP + Qwen-RobotNav-8B  $7 4 . 0 \pm 0 . 3$ </td></tr></table>

Matched full-system comparison. We use the same Qwen3.5 agent, episodes, initial states, and budget for all methods; ToolEQA also receives the same perception toolset as NavMCP. As shown in Table 2, Explore-EQA, ToolEQA, and FAST-EQA obtain 57.6%, 60.8%, and 63.5%, whereas NavMCP with Qwen-RobotNav-8B reaches 74.0%, leading by 16.4, 13.2, and 10.5 points. NavMCP therefore remains strongest after aligning the high-level reasoning and evaluation conditions.

## 4.3 Do the VLM Agent and NFM Executor Complement Each Other?

Table 3 uses two controlled sweeps to show that the upper-level VLM agent and lower-level NFM executor complement each other in the two-level architecture. EQA performance improves substantially with stronger agent harnesses and reasoning models. With Qwen-RobotNav-8B fixed, accuracy rises from 38.2% without an agent through single-round (58.4%) and reactive (62.0%) variants to 74.0% with the full Qwen3.5 harness; Qwen3.6-Plus further reaches 76.7%. These gains show that a capable NFM alone is insufficient without an upper-level agent that can adaptively plan, accumulate, and reuse evidence. EQA performance also improves substantially with a stronger navigation executor. With the full Qwen3.5 agent and NavMCP harness fixed, accuracy rises from Random Walk (60.9%) and Frontier Exploration (65.3%) to StreamVLN (69.3%), Qwen-RobotNav-4B (73.3%), and Qwen-RobotNav-8B (74.0%). Thus, a capable upper layer cannot compensate for weak execution: stronger language-conditioned navigation converts the same evidence needs into more useful trajectories and observations. Together, the two sweeps show that the VLM agent and NFM executor are complementary rather than substitutable, and end-to-end performance improves as either side becomes stronger.

Figure 4 illustrates this division of labor. To find a small black sofa, the upper layer searches five rooms, records negative evidence to avoid repeats, and switches to an object-targeted request after narrowing the hypothesis. The NFM executes each request and returns en-route observations. The non-monotonic curve reflects evidence-driven exploration: the system moves away to eliminate plausible regions before converging on answer evidence.

Q: I can't find my small black sofa anywhere. Where is it?  
Answer: A. In the bedroom  
![](images/53ad8eeffb79f966a6ce84cf8d9965d73ee374047b509b4af0a34e1bb8ee46ef.jpg)  
Figure 4: Evidence-guided search in an Explore-EQA episode (find a small black sofa). Top: selected first-person observations; the agent records negative evidence in five rooms before finding the target in a bedroom corner. Middle: distance to goal across navigation calls (red triangles: instructions sent to the executor); the trajectory is deliberately non-monotonic, moving away from the goal to rule out plausible rooms before an object-targeted final call. Bottom: tool-call timeline over 25 turns.

Table 4: Protocol ablation on HM-EQA. All rows fix the Qwen3.5-397B-A17B agent and Qwen-RobotNav-8B executor. The episodic-interface baseline replaces the complete three-channel protocol; the observationonly terminal-return variant and remaining rows remove or weaken one channel or auxiliary capability.
<table><tr><td>Configuration</td><td>Acc. (%)</td><td>Δ</td></tr><tr><td>Full system</td><td>74.0</td><td>一</td></tr><tr><td>Full episodic interface</td><td></td><td></td></tr><tr><td>Episodic interface</td><td>59.1</td><td>-14.9</td></tr><tr><td>Intent channel / navigation interface</td><td></td><td></td></tr><tr><td>Single navigation mode</td><td>72.0</td><td>-2.0</td></tr><tr><td>Observation channel</td><td></td><td></td></tr><tr><td>Terminal-only observation return</td><td>68.1</td><td>-5.9</td></tr><tr><td>w/o Journey Analysis</td><td>69.6</td><td>-4.4</td></tr><tr><td>Sparse keyframes (8-step / max 8)</td><td>71.2</td><td>-2.8</td></tr><tr><td>Memory channel</td><td></td><td></td></tr><tr><td>w/o EQA context state</td><td>69.4</td><td>-4.6</td></tr><tr><td>Auxiliary capabilities</td><td></td><td></td></tr><tr><td>w/o Context Compaction</td><td>73.1</td><td>-0.9</td></tr><tr><td>w/o zoom_in_object</td><td>73.4</td><td>-0.6</td></tr></table>

## 4.4 How Does NavMCP Close the Episodic Interface Gap?

With both architectural layers fixed, the complete episodic interface is insufficient. Replacing the three-channel protocol with an episodic interface reduces accuracy from 74.0% to 59.1% (−14.9 points), measuring the combined cost of the intent, observation, and cross-call accumulation gaps without changing the agent or executor backbone.

Core protocol channels contribute differently. Restricting the intent channel to one navigation mode costs 2.0 points, showing that instruction-following and object-targeted calls support complementary searches. For observation, changing only the observation return to terminal status and the final view costs 5.9 points, removing journey analysis costs 4.4 points, and sparsifying keyframes costs 2.8 points, showing that rollout evidence must be captured and structured. For memory, removing the EQA context state costs 4.6 points, showing that evidence and unresolved goals must persist across calls. Observation and memory produce the two largest component-level effects. Beyond the core protocol channels, auxiliary capabilities provide incremental gains. Context compaction and the zoom\_in\_object verification tool improve accuracy by 0.9 and 0.6 points, respectively, suggesting that the harness can further benefit from additional context-management and verification capabilities. Overall, intent specifies what to seek, observation preserves what the rollout reveals, and memory carries that evidence into subsequent decisions.

Table 5: Real-robot success rate (%) by task horizon. The Qwen3.6-Plus backbone is fixed across all methods; rows vary the agent-layer orchestration and navigation executor. Each tier contains 20 episodes.
<table><tr><td>Agent layer</td><td>Executor layer</td><td>Low single-room</td><td>Medium cross-room</td><td>High over 20 m</td><td>Overall</td></tr><tr><td>Reactive</td><td>Qwen-RobotNav</td><td>80</td><td>35</td><td>0</td><td>38.3</td></tr><tr><td>NavMCP</td><td>Frontier exploration</td><td>70</td><td>60</td><td>15</td><td>48.3</td></tr><tr><td>NavMCP</td><td>Qwen-RobotNav</td><td>90</td><td>85</td><td>60</td><td>78.3</td></tr></table>

## 4.5 Does NavMCP Transfer to Real Robots as Task Horizons Grow?

Setup. We test the same agent–NFM interface beyond simulation under sensor noise, odometry drift, local recovery, and long routes. Episodes use egocentric RGB, pose estimates, and an online map without privileged object locations or unseen geometry. Six task types with ten episodes each span single-room, cross-room, and over-20 m exploration; definitions are given in Section C.

Baselines. All three methods use the same Qwen3.6-Plus agent. The reactive baseline retains Qwen-RobotNav but removes multi-step deliberation and evidence accumulation; the frontier baseline replaces Qwen-RobotNav with heuristic coverage while keeping the agent fixed. These comparisons isolate orchestration and learned execution under real-world noise without repeating the full simulation ablations.

Real-world transfer. Table 5 shows that NavMCP reaches 78.3% overall, versus 48.3% for frontier exploration and 38.3% for the reactive learned-navigator baseline. The same interface therefore remains effective under real sensing and execution noise without privileged information.

Effect of exploration horizon. Margins over the strongest baseline rise from 10 to 25 and 45 points across tiers, demonstrating larger gains at longer horizons. See Section C and Figure 5 for details.

## 5 Conclusion, Limitations, and Future Work

We presented NavMCP, an agentic scaffolding framework that combines VLM reasoning with grounded NFM execution by closing the episodic interface gap: the mismatch between route instructions and evidence-seeking intent, intermediate observation loss, and the absence of cross-call accumulation. The navigator acts not as a standalone answerer but as an evidence-acquisition executor controlled by a VLM through intent, observation, and memory channels. By separating evidence planning from execution, converting trajectories into reusable journey summaries, and preserving grounded observations across turns, NavMCP improves three EQA benchmarks and controlled ablations, including a matched terminalonly interface baseline that isolates the cost of the gap itself. Thus, navigation models benefit EQA most when an outer agent can inspect, reuse, and control their rollouts.

Limitations. NavMCP relies on a large VLM, slowing inference; smaller agents or distillation may reduce this cost. Cross-view duplication also impairs counting.

Future Work. Beyond EQA, agents could use navigation foundation models to explore physical environments and efficiently collect embodied data for training agents’ spatial intelligence. Such agent-directed exploration may provide scalable, task-driven data that couples high-level reasoning with grounded observations and navigation trajectories.

## References

Michael Ahn, Anthony Brohan, Noah Brown, Yevgen Chebotar, Omar Cortes, Byron David, Chelsea Finn, Chuyuan Fu, Keerthana Gober, Karol Gopalakrishnan, et al. Do as i can, not as i say: Grounding language in robotic affordances. In Conference on Robot Learning, 2023.

Peter Anderson, Qi Wu, Damien Teney, Jake Bruce, Mark Johnson, Niko Sünderhauf, Ian Reid, Stephen Gould, and Anton van den Hengel. Vision-and-language navigation: Interpreting visually-grounded navigation instructions in real environments. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 3674–3683, 2018.

Shizhe Chen, Pierre-Louis Guhur, Cordelia Schmid, and Ivan Laptev. Think global, act local: Dual-scale graph transformer for vision-and-language navigation. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022.

Zedong Chu, Shichao Xie, Xiaolong Wu, Yanfen Shen, Minghua Luo, Zhengbo Wang, Fei Liu, Xiaoxu Leng, Junjun Hu, Mingyang Yin, et al. Abot-n0: Technical report on the vla foundation model for versatile embodied navigation. arXiv preprint arXiv:2602.11598, 2026.

Abhishek Das, Samyak Datta, Georgia Gkioxari, Stefan Lee, Devi Parikh, and Dhruv Batra. Embodied question answering. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 1–10, 2018.

Daniel Fried, Ronghang Hu, Volkan Cirik, Anna Rohrbach, Jacob Andreas, Louis-Philippe Morency, Taylor Berg-Kirkpatrick, Kate Saenko, Dan Klein, and Trevor Darrell. Speaker-follower models for vision-and-language navigation. In Advances in Neural Information Processing Systems, 2018.

Ruiyan Gong, Yingnan Guo, Junjun Hu, Jintao Kong, Xiaoxu Leng, Tianlun Li, Weize Li, Fei Liu, Zhicheng Liu, Jia Lu, et al. ABot-N1: Toward a general visual language navigation foundation model. arXiv preprint arXiv:2607.10383, 2026.

Daniel Gordon, Aniruddha Kembhavi, Mohammad Rastegari, Joseph Redmon, Dieter Fox, and Ali Farhadi. IQA: Visual question answering in interactive environments. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 4089–4098, 2018.

Weituo Hao, Chunyuan Li, Xiujun Li, Lawrence Carin, and Jianfeng Gao. Towards learning a generic agent for vision-and-language navigation via pre-training. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2020.

Yicong Hong, Qi Wu, Yuankai Qi, Cristian Rodriguez-Opazo, and Stephen Gould. VLN-BERT: A recurrent vision-and-language BERT for navigation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021.

Chenguang Huang, Oier Mees, Andy Zeng, and Wolfram Burgard. Visual language maps for robot navigation. In IEEE International Conference on Robotics and Automation, 2023a.

Wenlong Huang, Fei Xia, Ted Xiao, Harris Chan, Jacky Liang, Pete Florence, Andy Zeng, Jonathan Tompson, Igor Mordatch, Yevgen Chebotar, et al. Inner monologue: Embodied reasoning through planning with language models. In Conference on Robot Learning, 2023b.

Huiqiang Jiang, Qianhui Wu, Chin-Yew Lin, Yuqing Yang, and Lili Qiu. LLMLingua: Compressing prompts for accelerated inference of large language models. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 13358–13376, 2023.

Kaixuan Jiang, Yang Liu, Weixing Chen, Jingzhou Luo, Ziliang Chen, Ling Pan, Guanbin Li, and Liang Lin. Beyond the destination: A novel benchmark for exploration-aware embodied question answering. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 9091–9101, October 2025.

Zixing Lei, Changxing Liu, Yichen Xiong, Minhao Xiong, Yuanzhuo Ding, Zhipeng Zhang, Weixin Li, and Siheng Chen. Towards long-horizon embodied agents with tool-aligned vision-language-action models. arXiv preprint arXiv:2605.13119, 2026.

Jacky Liang, Wenlong Huang, Fei Xia, Peng Xu, Karol Hausman, Brian Ichter, Pete Florence, and Andy Zeng. Code as policies: Language model programs for embodied control. In IEEE International Conference on Robotics and Automation, 2023.

Xin Lu, Rui Li, Xun Huang, Weixin Li, Chuanqing Zhuang, Jiayuan Li, Zhengda Lu, Jun Xiao, and Yunhong Wang. Memory-guided view refinement for dynamic human-in-the-loop eqa. arXiv preprint arXiv:2603.09541, 2026.

Arjun Majumdar, Anurag Ajay, Xiaohan Zhang, Pranav Putta, Sriram Yenamandra, Mikael Henaff, Sneha Silwal, Paul Mcvay, Oleksandr Maksymets, Sergio Arnaud, et al. OpenEQA: Embodied question answering in the era of foundation models. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16488–16498, 2024.

Charles Packer, Sarah Wooders, Kevin Lin, Vivian Fang, Shishir G. Patil, Ion Stoica, and Joseph E. Gonzalez. MemGPT: Towards LLMs as operating systems. arXiv preprint arXiv:2310.08560, 2023.

Bhrij Patel, Vishnu Sashank Dorbala, Dinesh Manocha, and Amrit Singh Bedi. Multi-LLM QA with embodied exploration. arXiv preprint arXiv:2406.10918, 2024.

Krishan Rana, Jesse Haviland, Sourav Garg, Jad Abou-Chakra, Ian Reid, and Niko Suenderhauf. SayPlan: Grounding large language models using 3D scene graphs for scalable robot task planning. In Conference on Robot Learning, 2023.

Allen Z Ren, Jaden Clark, Anushri Dixit, Masha Itkina, Anirudha Majumdar, and Dorsa Sadigh. Explore until confident: Efficient exploration for embodied question answering. In First Workshop on Vision-Language Models for Navigation and Manipulation at ICRA 2024, 2024.

Saumya Saxena, Blake Buchanan, Chris Paxton, Peiqi Liu, Bingqing Chen, Narunas Vaskevicius, Luigi Palmieri, Jonathan Francis, and Oliver Kroemer. GraphEQA: Using 3D semantic scene graphs for real-time embodied question answering. arXiv preprint arXiv:2412.14480, 2024.

Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. Toolformer: Language models can teach themselves to use tools. In Advances in Neural Information Processing Systems, 2023.

Noah Shinn, Federico Cassano, Edward Berman, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. Reflexion: Language agents with verbal reinforcement learning. In Advances in Neural Information Processing Systems, 2023.

Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan, and Anima Anandkumar. Voyager: An open-ended embodied agent with large language models. Transactions on Machine Learning Research, 2024.

Meng Wei, Chenyang Wan, Xiqian Yu, Tai Wang, Yuqiang Yang, Xiaohan Mao, Chenming Zhu, Wenzhe Cai, Hanqing Wang, Yilun Chen, Xihui Liu, and Jiangmiao Pang. StreamVLN: Streaming vision-andlanguage navigation via slowfast context modeling. arXiv preprint arXiv:2507.05240, 2026.

Yuncong Yang, Han Yang, Jiachen Zhou, Peihao Chen, Hongxin Zhang, Yilun Du, and Chuang Gan. 3d-mem: 3d scene memory for embodied exploration and reasoning. In Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR), pages 17294–17303, June 2025.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. Re-Act: Synergizing reasoning and acting in language models. In International Conference on Learning Representations, 2023.

Naoki Yokoyama, Dhruv Batra, et al. VLFM: Vision-language frontier maps for zero-shot semantic navigation. In IEEE International Conference on Robotics and Automation, 2024.

Mingliang Zhai, Zhi Gao, Yuwei Wu, and Yunde Jia. Memory-centric embodied question answering, 2025a. URL https://arxiv.org/abs/2505.13948.

Mingliang Zhai, Hansheng Liang, Xiaomeng Fan, Zhi Gao, Chuanhao Li, Che Sun, Xu Bin, Yuwei Wu, and Yunde Jia. Multi-step reasoning for embodied question answering via tool augmentation. arXiv preprint arXiv:2510.20310, 2025b.

Haochen Zhang, Nirav Savaliya, Faizan Siddiqui, and Enna Sachdeva. Fast-eqa: Efficient embodied question answering with global and local region relevancy. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 1664–1673, 2026a.

Jiazhao Zhang, Kunyu Wang, Rongtao Xu, Gengze Zhou, Yicong Hong, Xiaomeng Fang, Qi Wu, Zhizheng Zhang, and He Wang. Navid: Video-based vlm plans the next step for vision-and-language navigation. Robotics: Science and Systems, 2024.

Jiazhao Zhang, Kunyu Wang, Shaoan Wang, Minghan Li, Haoran Liu, Songlin Wei, Zhongyuan Wang, Zhizheng Zhang, and He Wang. Uni-navid: A video-based vision-language-action model for unifying embodied navigation tasks. Robotics: Science and Systems, 2025.

Jiazhao Zhang, Anqi Li, Yunpeng Qi, Minghan Li, Jiahang Liu, Shaoan Wang, Haoran Liu, Gengze Zhou, Yuze Wu, Xingxing Li, Yuxin Fan, Wenjun Li, Zhibo Chen, Fei Gao, Qi Wu, Zhizheng Zhang, and He Wang. Embodied navigation foundation model. In International Conference on Learning Representations, 2026b.

Jiazhao Zhang, Gengze Zhou, Hale Yin, Yiyang Huang, Zixing Lei, Qihang Peng, Haoqi Yuan, Jie Zhang, Xudong Guo, Xiaoyue Chen, et al. Qwen-RobotNav technical report: A scalable navigation model designed for an agentic navigation system. arXiv preprint arXiv:2606.18112, 2026c.

Yong Zhao, Kai Xu, Zhengqiu Zhu, Yue Hu, Zhiheng Zheng, Yingfeng Chen, Yatai Ji, Chen Gao, Yong Li, and Jincai Huang. CityEQA: A hierarchical LLM agent on embodied question answering benchmark in city space. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 12465–12480, 2025.

Gengze Zhou, Yicong Hong, Zun Wang, Xin Eric Wang, and Qi Wu. Navgpt-2: Unleashing navigational reasoning capability for large vision-language models. In European Conference on Computer Vision, pages 260–278. Springer, 2024a.

Gengze Zhou, Yicong Hong, and Qi Wu. NavGPT: Explicit reasoning in vision-and-language navigation with large language models. In Proceedings of the AAAI Conference on Artificial Intelligence, 2024b.

Gengze Zhou, Yicong Hong, Zun Wang, Chongyang Zhao, Mohit Bansal, and Qi Wu. Same: Learning generic language-guided visual navigation with state-adaptive mixture of experts. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7794–7807, 2025.

Kaiwen Zhou, Kaizhi Zheng, Connor Pratt, Yun Shen, Blake Hannaford, and Dieter Fox. ESC: Exploration with soft commonsense constraints for zero-shot object navigation. arXiv preprint arXiv:2301.13166, 2023.

Zibo Zhou, Yue Hu, Lingkai Zhang, Zonglin Li, and Siheng Chen. Beliefmapnav: 3d voxel-based belief map for zero-shot object navigation. Advances in Neural Information Processing Systems, 38:83008–83036, 2026.

## A Additional Experimental Analysis

## A.1 Per-Category Results

Table 6: Per-category results on HM-EQA.
<table><tr><td>Category</td><td>Accuracy (%)</td></tr><tr><td>Existence</td><td>80.6</td></tr><tr><td>Identification</td><td>80.2</td></tr><tr><td>Location</td><td>78.5</td></tr><tr><td>State</td><td>77.2</td></tr><tr><td>Count</td><td>65.0</td></tr><tr><td>Overall</td><td>76.7</td></tr></table>

Table 6 shows per-category performance. Existence and identification questions achieve over 80%, indicating that NavMCP is effective when the answer can be grounded in object presence or category recognition after active search. State questions also remain strong at 77.2%, suggesting that the harness benefits verification-oriented questions where the agent must inspect candidate evidence rather than answer from a single observation. Counting remains the hardest category at 65.0%, partly because multiple keyframes may contain overlapping views of the same object and the agent can over-count repeated observations. This result motivates stronger cross-view object association and uncertainty tracking for future versions. On MT-HM3D, relationship questions reach 63.9% and comparison questions reach 59.6%. Relationship questions benefit from spatial perception tools, while comparison questions benefit from the protocol requirement that target observations be recorded with source evidence before later reasoning turns.

## A.2 Detailed Ablation Analysis

The executor-capability block in Table 3 shows a layered effect. Replacing heuristic movement with StreamVLN improves accuracy from 60.9% for Random Walk and 65.3% for Frontier Exploration to 69.3%, confirming the value of language-conditioned learned navigation. Using Qwen-RobotNav-4B further raises accuracy to 73.3%, 4.0 percentage points above StreamVLN, while scaling to 8B adds a comparatively modest 0.7 percentage points. Because these estimates use three stochastic hosted-LLM runs, we report the 4B–8B difference only as a scaling trend.

Episodic-interface baseline. The Episodic interface row of Table 4 isolates the episodic interface gap as a whole rather than one channel. This variant keeps the agent and executor backbones, perception tools, episodes, initial states, and budgets identical to the full system, but replaces the three-channel protocol with the episodic tool interface defined in Section 1. The agent issues a navigation instruction and receives the success flag, stop reason, executed step count, final pose, and final front view, without the structured intent schema, sampled trajectory keyframes, journey artifact, or protocol-managed cross-call evidence state. It does not alter the navigation model’s internal policy or observation stream during execution. The agent may still gather evidence at the arrival location through panoramic inspection, so the variant measures what the interface discards, not a crippled agent. Accuracy falls to 59.1% (−14.9 percentage points), a substantially larger drop than any single-component ablation. This result shows that the whole interface gap is not captured by weakening journey analysis or persistent evidence state in isolation.

Terminal-only observation return. The observation-channel row at 68.1% keeps the full intent and memory channels but changes only the navigation return to terminal status and the final front view. Its 5.9-point drop isolates the cost of discarding en-route observations while retaining structured intent and cross-call accumulation.

Table 4 provides the detailed protocol ablations summarized in the main text. Frontier Exploration exceeds Random Walk by 4.4 percentage points in Table 3, but both remain substantially below the learned navigation executors. This gap is consistent with geometric coverage being imperfectly aligned with question-relevant evidence when the agent repeatedly issues semantic sub-goals; the largest drops occur on counting and identification questions, where semantically guided viewpoint selection is especially important for localizing relevant objects. Changing only the observation return to terminal status and the final view reduces accuracy by 5.9 percentage points, directly measuring the value of en-route observations. Removing journey analysis reduces accuracy by 4.4 percentage points because intermediate observations are no longer converted into a structured journey artifact for the agent. Reducing keyframe density costs 2.8 percentage points, confirming that visual sampling during navigation directly affects the agent’s environmental state. Restricting the executor to a single navigation mode drops accuracy by 2.0 percentage points, indicating that object-targeted and instruction-following calls support complementary exploration strategies. Context compaction has a smaller direct accuracy effect (0.9 percentage points), which suggests that its main role is to keep long-horizon context management efficient and stable rather than to act as a stand-alone accuracy module. Removing zoom\_in\_object costs 0.6 percentage points, indicating that local verification is useful but not the primary driver of the harness gains. Among the remaining component ablations, removing the EQA context state causes the largest drop, from 74.0% to 69.4% (−4.6 percentage points). This result shows that the agent cannot rely on raw turn-by-turn observations alone: it needs an explicit EQA context state that records accumulated evidence in the evidence ledger, unresolved goals in the unresolved-goal state, and answer-relevant history across navigation calls. Without this state, multi-turn exploration becomes less coherent, and evidence collected in earlier trajectories is less reliably reused during final answer verification.

## B Evaluation Protocol and Statistics

Baseline values in Table 1 are quoted from the corresponding papers; <sup>∗</sup> denotes results reproduced and reported by FAST-EQA. Because several implementations and per-episode predictions are unavailable, cross-paper differences are contextual only and are excluded from paired significance tests. Our main results are mean ± standard deviation over three independent runs, where the variation arises from the stochasticity of the hosted LLM API rather than from explicit random seeds. For the matched-task comparison in Table 2, Explore-EQA retains its own pipeline, while ToolEQA and FAST-EQA are independently reproduced by us following their reported methods. All three use the same Qwen3.5 reasoning model, benchmark episodes, initial states, and interaction budget as the NavMCP reference in that table. In addition, the ToolEQA reproduction is equipped with the same perception and verification toolset as NavMCP (Table 8), covering panoramic inspection, open-vocabulary detection and segmentation, close-up object inspection, depth estimation, and saved-keyframe review, backed by the same perception service and VLM backends. ToolEQA invokes its original episodic navigation tools within its own pipeline, whereas NavMCP issues semantic navigation calls and receives trajectory-level evidence through its protocol channels. All executor variants share the agent, perception stack, split, episode initialization, and navigation budget; only the executor changes. We report these controlled results over three independent runs. The episodic-interface variant in Table 4 holds the agent and executor backbones, perception stack, episodes, and budgets fixed while replacing the three-channel protocol; the terminal-only observation-return variant changes only the navigation return format. Both are evaluated across three runs.

## C Real-Robot Evaluation Details

Each episode starts from a natural-language query, resets the exploration state, and allows the agent to issue multiple navigation calls. The robot records egocentric RGB observations, pose estimates, an online explored map, keyframes, searched regions, negative observations, and final answer support, without privileged object locations or unseen geometry. The two low tasks require single-room evidence gathering (e.g., the logo printed on a backpack or the color of a water dispenser), the two medium tasks require cross-room search (e.g., the screen-cast code in meeting room B01 or the nearest coffee-shop brand), and the two high tasks require open-scene exploration over routes exceeding 20 m (e.g., the number of elevators in the C3 elevator hall or whether the post office is open).

Episode construction. For each task, we predefine two distinct robot starting poses and repeat each starting condition five times, producing 2 × 5 = 10 evaluation episodes. The two tasks in each difficulty tier yield 20 episodes per method and tier, for a total of 60 real-robot episodes per method. All compared methods are evaluated on exactly the same starting poses, natural-language queries, and environment configuration. Before each episode, the robot is returned to the designated starting pose, and the exploration state and agent memory are reset.

Success and safety protocol. Episode success is judged manually after the run. An episode is successful only if the robot autonomously completes the evidence-gathering process within the tier-specific time limit and returns the correct answer supported by observations acquired during that episode. The timeout is 5 minutes for low-difficulty tasks, 10 minutes for medium-difficulty tasks, and 15 minutes for high-difficulty tasks; a timeout or an incorrect answer is counted as failure. The Unitree Go2 local obstacle-avoidance system remains enabled throughout every run to ensure safe physical execution. The low-level safety layer supplies no semantic targets, answer evidence, or planning.

Because physical runs are costly, this study targets sim-to-real transfer rather than repeating the full capability and protocol ablations established in Tables 3 and 4. The reactive baseline keeps the learned navigator but removes multi-step deliberation, isolating memory-driven orchestration between the agent and navigator; the frontier baseline replaces the learned navigator with heuristic coverage, testing query-conditioned semantic navigation under real odometry drift and local recovery.

On single-room tasks all methods perform reasonably (70–90%), because the answer lies within one room. On cross-room tasks the reactive variant drops to 35% without persistent deliberation, whereas NavMCP reaches 85%. On long-horizon tasks the reactive variant fails (0%) and frontier coverage reaches 15%, while NavMCP answers 60% of episodes. The widening gap mirrors simulation: longer, multi-room tasks require persistent evidence and multi-step orchestration beyond a navigator or single-shot reasoner.

## D Qualitative Analysis

![](images/92f211bd81bc0fc19dd33802a27c186392ac7a49108352543fbe5dce60eb773b.jpg)  
Figure 5: Ultra-long-horizon real-robot navigation with agentic Qwen-RobotNav. On a physical robot, the agent answers an open-ended request over a trajectory longer than 50 m by decomposing the task into sub-goals, following landmarks to Cotti Coffee, and verifying the green umbrella from visual evidence. Selected turns illustrate the loop of upper-level planning, Qwen-RobotNav execution, memory updates, and final response generation.

Q: How many pillows are there on the lower bunk bed?  
![](images/e89d1c15a56e2b43fff844d03e058fb775fb556c162ace56589ecf44264c36a9.jpg)  
Figure 6: Additional progressive discovery episode. The agent is asked to count pillows on a lower bunk bed. It first uses instruction-guided navigation to search for a room with bunk beds, switches to navigate\_to\_object(target=bunk bed) after reaching the bedroom area, and then uses a final local approach instruction to obtain a closer view. This appendix case complements Figure 4 by showing how the same protocol also supports fine-grained answer verification after object localization.

Ultra-long-horizon real-robot showcase. Figure 5 presents a physical deployment whose cumulative navigation trajectory exceeds 50 m, substantially extending beyond the > 20 m open-scene tier evaluated in the main real-robot study. Rather than attempting the request in a single navigation call, the agent decomposes it into landmark-conditioned sub-goals, uses Cotti Coffee as an intermediate anchor, records progress in memory, and continues until the requested green umbrella is visually verified. This episode shows that the protocol between the agent and navigator preserves task state and grounded evidence over an ultra-long physical trajectory and repeated planning and execution cycles.

We qualitatively inspect successful episodes to understand how the harness uses navigation calls as evidence-gathering steps rather than as isolated movements. In multi-room search questions, the agent first issues broad instructions such as searching nearby bedrooms or kitchen areas, then records which regions have been checked before selecting the next sub-goal. This behavior reduces repeated exploration because later turns can condition on notebook entries such as “Kitchen searched: microwave and toaster found; coffee maker absent” instead of relying only on the latest view.

The traces also show adaptive switching between navigation modes. When the target category is still uncertain, the agent tends to call navigate\_by\_instruction with a coarse semantic direction, allowing the executor to search for a plausible area. After an object anchor is observed, the agent often switches to navigate\_to\_object or visual verification tools to obtain closer evidence before answering. This pattern matches the intent-channel design: the agent specifies the evidence need at the semantic level, while the executor handles low-level movement. The additional episode in Figure 6 shows this second pattern: once the agent narrows the search to the bedroom area, an object-targeted call and a short local approach provide the visual evidence needed to count the pillows on the lower bunk bed.

Failure cases are concentrated in questions that require precise counting or fine-grained state verification. In these episodes, multiple keyframes may contain overlapping views of the same object, and the agent sometimes treats repeated observations as distinct evidence. Future versions should improve cross-view object association and uncertainty tracking, especially for counting, the weakest category.

Table 7: Real-robot evaluation protocol. Each tier contains two tasks. Two fixed starting poses with five repeated episodes per pose produce ten episodes per task and twenty episodes per method and tier.
<table><tr><td>Tier</td><td></td><td>Starts Runs/start Episodes Timeout</td><td></td><td></td></tr><tr><td>Low</td><td>2</td><td>5</td><td>20</td><td>5min</td></tr><tr><td>Medium</td><td>2</td><td>5</td><td>20</td><td>10 min</td></tr><tr><td>High</td><td>2</td><td>5</td><td>20</td><td>15 min</td></tr></table>

## E Tool Specification

Table 8 provides the complete tool specification exposed to the VLM agent via the tool-calling API. Exposure denotes availability in the agent’s tool schema; it does not imply that a tool is selected in every episode. The optional cross-episode retrieval store is disabled in all reported experiments and is therefore excluded from the table.

Table 8: Complete tool specification exposed to the agent through the tool-calling interface.
<table><tr><td>Category</td><td>Tool</td><td>Description</td></tr><tr><td rowspan="7">Visual</td><td>look_around</td><td>Read four directional views with the VLM for scene context and navigation affordances</td></tr><tr><td>analyze_status</td><td>Arbitrate current, historical, and notebook evidence before answer commitment</td></tr><tr><td>detect_objects_360</td><td>Detect and count prompted objects across all four views</td></tr><tr><td>zoom_in_object</td><td>Crop a detected object and inspect fine-grained attributes with full-view context</td></tr><tr><td>review_image</td><td>Retrieve and re-analyze archived panoramas or navigation keyframes</td></tr><tr><td>segment detect_objects</td><td>Segment a text-prompted object in the current front view</td></tr><tr><td></td><td>Run prompted or open-vocabulary detection in the current front view</td></tr><tr><td>estimate_depth</td><td>Estimate metric depth and object-contact information in the current front view</td></tr><tr><td rowspan="4">Navigation</td><td>navigate_to_object</td><td>Object-goal navigation (up to S steps)</td></tr><tr><td>navigate_by_instruction reset_navigation</td><td>Instruction-following navigation (up to S steps)</td></tr><tr><td></td><td>Clear executor-side history before switching to a new desti- nation</td></tr><tr><td>execute_action</td><td>Apply one local turn, move, or stop action for final-view</td></tr><tr><td>Context</td><td>write_notebook</td><td>alignment Append observations to the persistent within-episode evi- dence notebook</td></tr></table>

## E.1 Perception and Verification Tools

The perception tools use two complementary backends. Reader tools (look\_around, analyze\_status, review\_image, and zoom\_in\_object) send visual evidence to the VLM with a task-specific prompt. Detector tools (segment, detect\_objects, detect\_objects\_360, and estimate\_depth) call a separate perception service built on open-vocabulary segmentation and detection together with metric depth estimation. Single-frame tools automatically use the current front view. Panoramic tools use the four native front, right, back, and left cameras when available; otherwise the adapter synthesizes the panorama by rotating the agent through four 90 headings and restoring its original orientation.

Panoramic scene reading and detection. look\_around sends the four direction-labeled views in one multi-image request and returns a free-text scene analysis together with the saved-image manifest. Every reader prompt additionally asks the VLM to report visible doorways, corridors, stairs, open passages, and room transitions because these observations support the next navigation decision. For light-state questions, the prompt explicitly distinguishes artificial illumination from sunlight. detect\_objects\_360 runs prompted open-vocabulary detection on all four views and aggregates per-label instances and the views in which they occur. Because adjacent views can contain the same object, its return includes a warning that per-view counts require cross-view reconciliation.

Shared navigation-focus suffix.

IMPORTANT: In addition to answering the question, always note:   
doorways, hallways, corridors, staircases, open passages, and room   
transitions visible in the image. These are critical for navigation   
planning.   
For light/lamp on-off questions: distinguish between NATURAL light   
(sunlight from windows) and ARTIFICIAL light (lamp glow, light fixture   
illumination). A lamp is ON only if it emits visible glow or   
illumination. A bright room from window sunlight does NOT mean lamps   
are on.

Panoramic scene-reading prompt.

You are seeing 4 views (front, right, back, left) forming a 360 degree   
panorama of your current location.   
{question}{navigation\_focus\_suffix}

Fine-grained and historical inspection. zoom\_in\_object detects the requested target across the panorama, retains the largest-area detection, and sends both a padded crop and its full source view to the VLM. The full view establishes global category and scene context, while the crop is used for color, pattern, subtype, and text. When the two disagree, the full-view judgment takes precedence so that a context-free crop cannot overturn a reliable scene-level interpretation. review\_image re-runs the VLM on images already stored during the episode, selected by path, turn, tool, view, panorama, or the latest navigation-keyframe shortcut. It recovers evidence from intermediate observations without retraversing the region.

Fine-grained crop-and-context prompt.

The first image is a cropped close-up of '{target}'. The second image   
is the full {view} view for context.   
Question: {question}   
Focus on fine-grained visual attributes: exact color, pattern,   
texture, shape, object type/subtype, brand, text/labels on the object.   
Be precise and specific. For example, say 'dark green' not just 'green',   
'induction cooktop' not just 'stove'.

Single-frame geometric tools. segment returns whether a text-prompted mask is found in the current front frame; detect\_objects lists prompted or open-vocabulary detections; and estimate\_depth combines segmentation with metric depth to report target distance and contact information. These tools remain available, although panoramic inspection usually yields more complete EQA evidence.

## F Implementation Details

Navigation executor. We use the pretrained 8B Qwen-RobotNav checkpoint (Zhang et al., 2026c) as a fixed embodied executor. Given a natural-language sub-goal and egocentric observations, it predicts a short-horizon trajectory of eight planar waypoints. The adapter exposes only the two modes used in our EQA experiments: navigate\_by\_instruction for language-guided movement and navigate\_to\_object for object search and approach. Each call returns an executable waypoint rollout; answer-level reasoning, trajectory summarization, evidence-ledger updates, and the decision to issue another call remain responsibilities of NavMCP. We do not train or modify the executor; architecture, training, and navigation-only evaluations are documented in the source report.

Closed-loop navigation adapter. Both semantic navigation tools invoke the same multi-step controller and differ only in how the language sub-goal is formed. At each simulator step, the adapter sends the executor the current front, right, back, and left RGB observations, the natural-language instruction, episode identifier, agent position and orientation, online explored map, and collision state. The executor returns either a stop decision or a relative short-horizon waypoint prediction. The adapter transforms the prediction into a navmesh-aware waypoint action and executes it before acquiring the next observation. The loop terminates when the executor stops, the simulator marks the episode complete, or the call exhausts its step budget S = 64. Two consecutive collisions set an is\_stuck flag in the next executor query, enabling closed-loop recovery rather than terminating the semantic call immediately. reset\_navigation clears the executor’s server-side episode history before the agent switches to a new destination, whereas execute\_action bypasses the executor for one local alignment action such as turning the final target into the front view.

Trajectory return. During a rollout, the adapter samples the front view every four simulator steps and retains at most 16 evenly spaced keyframes. After execution, the trajectory summarizer converts these frames into the journey artifact described in Section G. Each navigation call returns the number of executed steps, executor stop state, simulator completion state, path information, journey analysis, and persisted image references. This return exposes the navigation outcome and pre-terminal evidence.

Deployment. The main results use Qwen3.6-Plus as the VLM agent with a 256K token context window and native tool-calling support. For controlled ablations, we use our self-deployed open-source Qwen3.5-397B-A17B agent. Across ablation variants, we keep the agent backbone, serving stack, decoding configuration, and tool-calling setting fixed. All variants use the same hosted 8B executor unless the executor itself is ablated in Table 3. Visual perception uses an open-vocabulary detector with SAM for segmentation and Depth-Anything for depth estimation.

Computing infrastructure. Simulation and local perception (open-vocabulary detection, SAM segmentation, and Depth-Anything depth) run in Habitat-Sim 0.3.3 on HM3D scenes on one NVIDIA RTX 4090 GPU under Ubuntu 22.04. Because the Qwen3.6-Plus agent and Qwen-RobotNav use hosted APIs on uncontrolled hardware, run-to-run variation reflects API stochasticity, not local nondeterminism.

Hyperparameters. Key NavMCP settings: 50 agent turns; S = 64 steps/call; T = 60,000-token compaction; $k = 2$ protected VLM results; 100 ledger entries; four-step sampling with 16 keyframes/call; 1024/2048 assessment/synthesis tokens; and a 4,000-character tool-result cap.

## G Protocol Details

Algorithm 1 Protocol between the agent and executor for long-horizon EQA   
1: Initialize EQA context state $C _ { 0 } \gets \left( H _ { 0 } , E _ { 0 } , U _ { 0 } \right)$ from question q and initial observation   
2: for $t = 0$ to $\dot { T } _ { \mathrm { m a x } } - 1$ do   
3: Agent reads $C _ { t }$ and emits either a candidate answer $y _ { t }$ or an evidence need $u _ { t }$   
4: if agent emits candidate answer $y _ { t }$ then   
5: if y is supported by the current observation, $E _ { t } ,$ , or source artifacts then   
6: return $y _ { t }$   
7: end if   
8: Update $U _ { t }$ with the missing evidence   
9: continue   
10: end if   
11: INTENT INTERFACE: translate $u _ { t }$ into navigation call $a _ { t } = \left( m _ { t } , g _ { t } , S _ { t } , c _ { t } \right)$   
12: EXECUTOR CALL: run $a _ { t }$ with the navigation foundation model and return trajectory record $r _ { t } = $   
$\left( s _ { t } , b _ { t } , \ell _ { t } , p _ { t } , K _ { t } \right)$   
13: OBSERVATION INTERFACE: convert $r _ { t }$ into source-grounded journey evidence $z _ { t } = ( q _ { t } , R _ { t } , O _ { t } , P _ { t } , D _ { t } )$   
14: MEMORY INTERFACE: update $E _ { t + 1 }$ and $U _ { t + 1 }$ from $\left( C _ { t } , z _ { t } \right)$ with source references   
15: Compact processed raw returns and set $\overset { \cdot } { C _ { t + 1 } } = ( \overset { \cdot } { H _ { t + 1 } } , \overset { \cdot } { E _ { t + 1 } } , \overset { \cdot } { U _ { t + 1 } } )$   
16: end for   
17: Return the best answer supported by $C _ { T _ { \mathrm { m a x } } }$

Intent Call Schema. At turn $t ,$ the agent emits

$$
a _ { t } = ( m _ { t } , g _ { t } , S _ { t } , c _ { t } ) ,
$$

where $m _ { t }$ is the navigation mode, $g _ { t }$ is a natural-language sub-goal, $S _ { t }$ is the step budget, and $c _ { t }$ contains optional constraints such as target attributes or preferred regions. At each simulation step, the executor observes four RGB views, its current pose, and an online occupancy map built only from the explored trajectory; unexplored geometry and object locations remain unavailable. Execution stops when the executor emits STOP, exhausts $S _ { t } ,$ or the simulator marks the episode complete. Repeated collisions instead trigger recovery by setting is\_stuck in the next executor query; they do not immediately terminate the call.

Trajectory Record Schema. For a navigation call ${ { a } _ { t } } ,$ the executor first returns a raw trajectory record

$$
\begin{array} { r } { r _ { t } = ( s _ { t } , b _ { t } , \ell _ { t } , p _ { t } , K _ { t } ) , } \end{array}
$$

where $s _ { t }$ is the success flag, $b _ { t }$ is the stop reason, $\ell _ { t }$ is the path length, $p _ { t }$ is the final pose, and $K _ { t } =$ $\left\{ I _ { t , 1 } , \ldots , I _ { t , n } \right\}$ is a bounded set of sampled keyframes.

Keyframe Sampling. For each navigation call, keyframes are sampled every ∆ = 4 steps and capped at M = 16 frames per call. This setting balances visual coverage against context cost, while still allowing the observation channel to preserve evidence that appears before the executor stops.

Journey Artifact Schema. A VLM summarizer outputs

$$
z _ { t } = ( q _ { t } , R _ { t } , O _ { t } , P _ { t } , D _ { t } ) .
$$

Here q<sub>t</sub> records the attempted sub-goal and whether it appeared to be satisfied. $R _ { t }$ summarizes rooms and route-level context, and $O _ { t }$ stores salient observed objects. $P _ { t }$ stores planning cues such as doors, corridors, and exits, while $D _ { t }$ records uncertainty or failure conditions.

Source-Grounded Journey Artifact. Each object or room mention in the journey artifact is tied to a source keyframe and confidence label. We represent a grounded mention as $( n , i , v , h , c )$ , where n is the name, i is the keyframe id, v is the viewpoint, h is a location hint, and c is the confidence label. The summarizer is instructed to report only visually observed evidence, mark uncertainty explicitly, and avoid turning absence from a single view into a definitive negative claim. The prompt template used to narrate the sampled trajectory is shown below; the task-question line is included only when an EQA question is available.

Journey-analysis prompt.

The agent was navigating with instruction: "{instruction}"   
The agent's task question is: "{question}"   
These images show the agent's front view at different points during   
navigation.   
For each step, briefly describe: what room/area is this? What key   
objects, doorways, hallways, stairs, or room transitions are visible?   
If any details relevant to the task question are directly visible,   
mention them.   
Only mention question-relevant details when they are directly visible   
in the images; do not infer unseen states.   
Keep each step description to 1-2 sentences.

Evidence arbitration with analyze\_status. analyze\_status is the terminal evidence-verification tool and operates in three phases. First, the VLM jointly reads the current four views, exploration notebook, and available archived visual evidence and produces a structured assessment of target visibility and evidence sufficiency. Second, when evidence is insufficient, the harness may dispatch one supplementary inspection (saved-keyframe review for historical detail or panoramic detection for object presence and counting) and return its result to the same reasoning context. Third, the VLM synthesizes a final status that either authorizes an answer, records well-supported negative evidence, or directs further exploration. The selected target\_view also tells the agent whether a local rotation is needed to place the target in the final front view.

Phase 1: structured assessment prompt.

## Exploration Notebook   
{notebook\_text}   
## Analysis Question   
{question}   
You have the current 360 degree views, retrieved visual memory, and the   
agent's exploration notebook.   
The scene is static. All object states remain unchanged throughout the   
episode.   
When memory images are provided, explicitly compare [MEMORY] evidence   
against [CURRENT] views before deciding whether evidence is sufficient.   
Based on ALL evidence above, respond with a JSON object:   
{   
"target\_visibility": "clear" | "partial" | "not\_seen" | "uncertain",   
"target\_view": "front" | "right" | "back" | "left" | null,   
"enough\_evidence": true/false,   
"provisional\_answer": "brief answer or null",   
"reasoning\_summary": "1-2 sentence reasoning",   
"missing\_evidence\_type": "none" | "historical\_detail" |   
"object\_presence" | "counting" |   
"fine\_grained\_attribute",   
"recommended\_tool": "none" | "review\_image" |   
"detect\_objects\_360" | "zoom\_in\_object",   
"tool\_target": "object name for the tool, or null"   
}

The final synthesis prompt imposes a deliberately asymmetric evidence rule: a positive answer requires visible support, while a negative answer requires coverage of the relevant area rather than mere failure to observe the target.

Phase 3: final synthesis prompt.

## Final Synthesis   
Question: {question}   
Based on the 360 degree views, notebook, and any supplementary evidence   
above, provide your FINAL ANALYSIS.   
First, state your STATUS as one of:   
- ANALYSIS\_COMPLETE: You have enough evidence to confidently answer.   
- NEGATIVE\_EVIDENCE: You can confidently conclude the target does NOT   
exist or the condition is NOT met. You may ONLY use this status when:   
(a) The current 360 degree views clearly cover the relevant area and   
the target is obviously absent, OR   
(b) Historical navigation keyframes + current views + notebook   
consistently confirm the relevant area has been thoroughly   
checked.   
Do NOT use NEGATIVE\_EVIDENCE if you simply haven't seen the target   
yet. That is INSUFFICIENT\_EVIDENCE.   
INSUFFICIENT\_EVIDENCE: You genuinely lack evidence because the area has not   
been sufficiently explored.   
Format: Start with STATUS: <status> on the first line, then your   
analysis. Be specific and cite what you see. Do not output JSON.

## Evidence Ledger Schema. Each evidence entry is

$$
e _ { i } = ( \tau _ { i } , \sigma _ { i } , \kappa _ { i } , \eta _ { i } , \gamma _ { i } , u _ { i } ) ,
$$

where $\tau _ { i }$ is the turn index and $\sigma _ { i }$ is the source artifact or keyframe. $\kappa _ { i }$ is the evidence claim, $\eta _ { i }$ records its support, $\gamma _ { i }$ is a confidence label, and $u _ { i }$ marks whether the entry remains unresolved. The unresolved-goal state tracks goals, uncertain findings, and regions needing inspection.

Notebook implementation. write\_notebook implements the persistent within-episode evidence ledger exposed to the agent. It accepts a list of concise observations, tags each new entry with its source turn, removes normalized exact duplicates, and retains at most 100 entries in FIFO order. The notebook is reconstructed in the agent prompt on every turn, so curated evidence remains available even after older raw tool returns are compacted. Entries record navigation goals, visited or searched regions, room transitions, question-relevant positive and negative observations, uncertainty, and remaining evidence needs. The notebook resets for every evaluation episode; the reported results use no cross-episode retrieval or test-environment memory.

Context Compression. Raw tool returns are compressed only after salient information has been externalized into the evidence ledger or unresolved-goal state with source references. After this update, older tool returns are replaced with lightweight placeholders. If the total context exceeds threshold $T ,$ NavMCP keeps the system prompt, the original task input, the evidence ledger, unresolved goals, and recent dialogue turns, while dropping intermediate messages.

Equivalent Step Computation. Prior frontier-based EQA methods, including Explore-EQA, Memory-EQA, and FAST-EQA, define one exploration step as a high-level sense-plan-act cycle in which the agent teleports to a selected frontier point. When the VLM directly controls exploration in these protocols, one selected frontier action displaces the agent by at most 3 meters. We therefore inherit 3 meters as the conversion unit rather than choosing it as a free threshold, and the per-episode step budget is

$$
N = \left\lfloor { \sqrt { A } } \times 3 \right\rfloor ,\tag{2}
$$

where A is the navigable floor area of the scene in square meters. The normalized step metric is then

$$
\hat { n } = \frac { n _ { \mathrm { a c t u a l } } } { N } ,\tag{3}
$$

which measures the fraction of the allowed exploration budget consumed by an episode.

For NavMCP, let H be exactly the number of outer-agent VLM invocations during an episode. Each invocation is charged one base step, regardless of the high-level decision it returns. However, one semantic navigation decision may trigger a continuous VLN rollout longer than the 3-meter movement allowed to a frontier method in one step. Counting such a rollout as only one step would favor NavMCP. We therefore add a distance compensation for every NFM navigation call. Let K be the number of NFM navigation calls, and let $L _ { k }$ be the cumulative executed path length of the k-th NFM rollout, computed from its trajectory waypoints. The added charge is

$$
c _ { k } = \operatorname* { m a x } \left( 0 , \left\lceil \frac { L _ { k } } { 3 \mathrm { m } } \right\rceil - 1 \right) .\tag{4}
$$

The compensated metrics are

$$
n _ { \mathrm { e q } } = H + \sum _ { k = 1 } ^ { K } c _ { k } , \qquad \hat { n } _ { \mathrm { e q } } = \frac { n _ { \mathrm { e q } } } { N } .\tag{5}
$$

Thus, a rollout of at most 3 meters receives no extra charge, while rollouts of $\left( 3 , 6 \right]$ and $( 6 , 9 ]$ meters receive one and two additional steps, respectively. The ceiling operation deliberately favors the frontier baselines: each additional partial 3-meter segment is charged as a full step, and $\dot { L } _ { k }$ includes detours, repeated motion, and collision recovery that frontier teleportation does not incur. In our setup, outer agent VLM reasoning and API latency dominate wall-clock time, so H is the primary efficiency quantity. This accounting is not a count of all model inference in the system. Calls made by the navigation executor, trajectory summarizer, and auxiliary perception models are not separately charged; the outer-agent invocations that decide to call these components or reason over their outputs are already included in H. Accordingly, $\hat { n } _ { \mathrm { e q } }$ measures normalized high-level interaction and exploration rather than end-to-end computational cost. This equivalent-step compensation is applied only to configurations that invoke an NFM; non-NFM methods retain their original direct-control step accounting.

Collision Recovery. The executor tracks consecutive collisions during navigation. After 2 or more consecutive collisions, an is\_stuck signal is sent to the navigation foundation model, which can then adjust its predicted trajectory to recover from the stuck state.