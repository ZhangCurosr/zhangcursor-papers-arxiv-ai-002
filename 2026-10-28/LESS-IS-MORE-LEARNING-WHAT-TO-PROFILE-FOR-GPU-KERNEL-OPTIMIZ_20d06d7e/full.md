# LESS IS MORE: LEARNING WHAT TO PROFILE FOR GPU KERNEL OPTIMIZATION

Yihan Wang Renmin University of China yihan3123@gmail.com

Runyu Chen   
University of International Business and Economics   
ry.chen@uibe.edu.cn   
Peiyu Liu<sup>∗</sup>   
University of International Business and Economics   
liupeiyustu@163.com

Wei Xu Renmin University of China weixu@ruc.edu.cn

## ABSTRACT

Large language models have recently shown strong potential for automatically optimizing GPU kernels, yet their effectiveness depends critically on the hardware information available to the kernel optimization model. Existing systems typically rely on predefined profiling strategies, leaving what to profile largely outside the optimization process. We show that effective profiling is not simply about collecting more information: increasing the number of profiling metrics does not monotonically improve kernel optimization, while the most useful profiling information varies across kernels. Our theoretical analysis further shows that, under such heterogeneous profiling utility, kernel-conditioned selection can outperform any single fixed profiling configuration. Motivated by this observation, we propose AutoProfiler, a feedback-driven framework for learning a deterministic profiling tree from optimization feedback. The tree is iteratively refined through REWRITE and SPLIT operations, with an LLM generating candidate modifications and empirical evaluation determining their acceptance. Once constructed, the tree quickly gathers the required profiling information for any kernel, allowing the LLM rewriter to be invoked only once. Experiments on KernelBench show that our approach improves kernel optimization over fixed profiling configurations under constrained profiling budgets. These results suggest that hardware profiling should evolve from static metric collection toward profiling reasoning, making profiling itself a new dimension for improving kernel optimization. The codes are available at https://github.com/Satissss/AutoProfiler.

## 1 INTRODUCTION

LLM-based systems have recently shown strong potential for automatically rewriting GPU kernels to improve execution efficiency Ouyang et al. (2025); Zhang et al. (2025); Gai et al. (2026a). To better guide the rewriting process, recent systems increasingly incorporate hardware profiling to expose runtime bottlenecks and provide actionable optimization feedback Zhang et al. (2025); Gai et al. (2026a). However, existing approaches typically collect a predefined set of profiling signals before optimization. While simple and reusable, this design overlooks two fundamental properties of hardware profiling: profiling information is costly to acquire, and its utility is inherently kerneldependent.

This raises a fundamental question: What profiling information is actually neededfor effective kernel optimization? We investigate this question from two complementary perspectives. First, more profiling is not necessarily better. Empirically, increasing the number of profiling metrics does not monotonically improve kernel optimization, suggesting that more profiling information does not necessarily translate into better optimization. Second, the useful information is kernel-dependent. Our theoretical analysis shows that when profiling utility varies across kernels, kernel-conditioned selection can outperform any single fixed configuration, which is further supported by our empirical oracle analysis. Together, these findings suggest that effective profiling is not about collecting more information, but about identifying the right information for each kernel.

![](images/499ddbcb89a7ddab1a50d3e926966319279573b5ff0c123e701c939e47924cf0.jpg)  
Figure 1: Overview and motivation of our profiling reasoning framework. Left: Traditional kernel optimization and its key challenges, where redundant profiling signals introduce noisy feedback (a) and repeated profiling rounds incur high latency and LLM overhead (b). Right: Our AutoProfiler method utilizes tree-based sequential profiling to selectively acquire informative signals, achieving effective optimization with reduced profiling overhead.

To this end, we propose AutoProfiler, which learns a kernel-conditioned profiling policy that adaptively acquires profiling metrics before invoking the kernel rewriter. We formulate profiling as sequential information acquisition and represent the policy with a deterministic profiling tree, which selects subsequent measurements based on previously observed profiling results. The tree is refined through optimization feedback: an LLM proposes structural REWRITE to deepen the tree by collecting more profiling information, and SPLIT to widen the tree for kernels with different profiling needs, while empirical paired evaluation determines whether a proposed modification is accepted. The LLM is used only during policy construction, leaving inference-time routing fully deterministic.

Experiments on KernelBench show that AutoProfiler improves kernel optimization over fixed profiling configurations under a constrained profiling budget, while recovering a substantial portion of the gain available from kernel-conditioned selection. Interestingly, the learned profiling policy remains effective across iterative optimization rounds, enabling continued gains without repeated LLM-based profiling decisions. Together, these results point to a promising direction for selfimproving kernel generation: using more targeted profiling to provide effective feedback at lower cost, thereby supporting continued improvement through iterative optimization.

## 2 PRELIMINARY

Notations. Let $k \in \mathcal { K }$ denote a GPU kernel sampled from a kernel corpus, and let M denote the set of available profiling metrics. Given a subset of metrics $m \subseteq { \mathcal { M } }$ , a profiler P produces the corresponding profiling information $I = P ( k , m )$ . A kernel rewriter g then uses the original kernel and profiling information to generate an optimized kernel: ${ \hat { k } } = g ( k , I ) = g { \big ( } k , P ( k , m ) { \big ) }$ Throughout this work, the profiler P, the rewriter g, and the target hardware are fixed.

Problem setup. We formulate the GPU kernel optimization as a profiling-conditioned generation problem: given an input kernel, the optimized kernel is generated conditioned on both the kernel itself and the acquired hardware profiling information. Existing approaches typically use a fixed profiling configuration for all kernels. Given a predefined metric set $m _ { 0 } \subseteq { \mathcal { M } }$ , the optimized kernel is generated as $\hat { k } = g \big ( k , P ( k , m _ { 0 } ) \big )$ , where the same set of profiling metrics is collected regardless of the kernel being optimized. In contrast, we allow the profiling configuration to vary across kernels.

Let $\phi ( k ) \subseteq { \mathcal { M } }$ denote the profiling metrics selected for kernel k. The optimized kernel is then generated as

$$
\hat { k } = g \big ( k , P ( k , \phi ( k ) ) \big ) .\tag{1}
$$

This formulation turns profiling into a sequential reasoning process over hardware signals: based on the profiling information observed so far, the policy determines what additional information is most valuable to acquire next for kernel optimization.

Our goal. We aim to learn a profiling policy ϕ that determines what profiling information to acquire so as to improve kernel optimization. We measure the utility of the acquired profiling information by the performance of the resulting kernel. Specifically, for a kernel k and profiling metrics m, we define the resulting speedup as $\begin{array} { r } { s ( k , m ) = \frac { \operatorname { l a t } \bar { ( k ) } } { \operatorname { l a t } ( \hat { k } ) } } \end{array}$ , where lat(·) denotes the execution latency on the target hardware. The corresponding optimization reward is $r ( k , m ) = \operatorname* { m a x } { \{ s ( k , m ) - 1 , 0 \} }$ which assigns positive reward only when the acquired profiling information leads to a rewrite that improves upon the original kernel. Our objective is therefore to learn a profiling policy that maxi mizes the expected optimization reward:

$$
\arg \operatorname* { m a x } _ { \phi } \mathbb { E } _ { k \sim K } \left[ r \big ( k , \phi ( k ) \big ) \right] ,\tag{2}
$$

subject to a profiling budget that limits the amount of hardware information acquired for each kernel, where K denotes the distribution over kernels under consideration. The key challenge is therefore to exploit kernel-dependent profiling utility while avoiding unnecessary profiling cost.

## 3 STATIC VS. DYNAMIC METRIC SELECTION: THEORETICAL AND EMPIRICAL ANALYSIS

Theoretical Analysis. We first characterize the fundamental advantage of dynamic profiling over static metric selection. Specifically, we show that dynamic selection can never perform worse than the best fixed profiling configuration, and that a strict advantage arises when different configurations are effective for different kernels. Consider a finite family of candidate profiling configurations ${ \mathcal C } = \{ m _ { 1 } , m _ { 2 } , . . . , m _ { K } \}$ and a kernel distribution D. Let $u ( k , m )$ denote the kernel optimization utility $( e . g .$ , performance speedup) for kernel k under configuration m. The expected utility of the optimal static configuration is defined as $U _ { \mathrm { s t a t i c } } = \operatorname* { m a x } _ { m \in { \mathcal C } } \mathbb { E } _ { k \sim { \mathcal D } } [ u ( k , m ) ]$ , whereas an oracle dynamic policy that adaptively selects the most effective configuration per kernel achieves $U _ { \mathrm { d y n a m i c } } = \mathbb { E } _ { k \sim \mathcal { D } } [ \operatorname* { m a x } _ { m \in \mathcal { C } } { u ( \bar { k } , m ) } ]$

Proposition 1. Dynamic selection dominates static selection. For any finite candidate family C,

$$
U _ { \mathrm { d y n a m i c } } \geq U _ { \mathrm { s t a t i c } } .\tag{3}
$$

Moreover, equality holds if and only if there exists a configuration $m _ { j } \in { \mathcal { C } }$ whose success region covers the union of all candidate success regions almost surely under D. Otherwise, $U _ { \mathrm { d y n a m i c } } > U _ { \mathrm { s t a t i c } }$ . This result shows that the value of dynamic profiling is determined by the heterogeneity of profiling utility across kernels: when no single configuration covers the success regions of the others, adapting the profiling configuration to each kernel yields a strict advantage.

Empirical Validation. We next examine whether the heterogeneity underlying Proposition 1 occurs in practice. We conduct an oraclestyle evaluation on 128 KernelBench kernels using four roofline-based profiling configurations. For each kernel, we evaluate the same

![](images/97dc106536b7fea10ebc31d2cbb88844dec577f8f7efc7e060d506110858b490.jpg)  
Figure 2: Per-kernel oracle evaluation on 128 KernelBench kernels, showing substantial heterogeneity in the optimal profiling configuration and a 7.3% relative gain over the best fixed configuration.

frozen optimizer under each configuration and select the configuration yielding the highest speedup. Detailed experimental settings are provided in Appendix A.2. We find that the per-kernel oracle achieves a geomean speedup of 1.429×, compared with 1.332× for the best fixed configuration, corresponding to a 7.3% relative gain. Moreover, the oracle selects different configurations across kernels, with the best configuration varying substantially among the four candidates.

Key Findings on Profiling Utility

• Profiling utility is heterogeneous across kernels. Different kernels benefit from different profiling signals, and no single configuration is consistently optimal.

• Kernel-specific profiling provides substantial optimization headroom. The oracle gap shows that effective optimization depends on selecting useful profiling information for each kernel rather than collecting a universally strong metric set.

This result provides empirical support for the heterogeneity condition in Proposition 1 and indicates non-trivial room for kernel-conditioned profiling. The oracle serves only as an analysis reference; it requires evaluating all candidate configurations for each kernel and is unavailable at inference time. This motivates the central problem addressed next: learning which profiling configuration to use for a given kernel without exhaustively evaluating all candidates.

## 4 METHODS

To optimize each kernel with the most informative profiling feedback, we aim to identify what information is most useful at each stage of optimization. In this section, we present AutoProfiler, a feedback-driven framework for learning kernel-conditioned hardware profiling policies. AutoProfiler formulates hardware profiling as a sequential decision process and instantiates the learned policy with a deterministic profiling tree. Our methodology is structured as follows: we first introduce the Tree-based Profiling Policy formulation (Section 4.1), detail its Feedback-driven Policy Construction via LLM-based structural updates and describe its Deterministic Inference-time Routing pipeline (Section 4.2).

## 4.1 TREE-BASED PROFILING POLICY

AutoProfiler model profiling as a sequential decision process and implement the resulting policy with a deterministic tree. We first formalize how profiling information is acquired sequentially, then describe how this policy is represented by the tree, and finally explain how the learned tree guides profiling for an input kernel.

Sequential Profiling. To formalize this process, we formulate profiling as a sequential policy that determines what hardware information to collect next based on the profiling observations obtained so far. For a kernel k, at profiling step t, the current state is denoted by

$$
s _ { t } = ( M _ { t } , I _ { t } ) ,
$$

where $M _ { t } \subseteq { \mathcal { M } }$ is the set of metrics collected so far and ${ \cal I } _ { t } = P ( k , M _ { t } )$ is the corresponding profiling information. The policy then selects an additional metric group

$$
G _ { t } = \pi _ { T } ( s _ { t } ) , \qquad G _ { t } \subseteq \mathcal { M } \setminus M _ { t } ,
$$

which leads to the next profiling state

$$
s _ { t + 1 } = ( M _ { t + 1 } , I _ { t + 1 } ) , \qquad M _ { t + 1 } = M _ { t } \cup G _ { t } , \quad I _ { t + 1 } = P ( k , M _ { t + 1 } ) .
$$

Equivalently, the profiling process can be viewed as

$$
s _ { t } \xrightarrow { \pi _ { T } } G _ { t } \xrightarrow { \operatorname { c o l l e c t } G _ { t } } s _ { t + 1 } .
$$

Profiling terminates when the policy determines that no additional information is required. In this way, the metrics collected for a kernel depend on its observed hardware behavior rather than being fixed in advance.

Tree Representation. To instantiate this sequential policy, we represent $\pi _ { T }$ as a rooted tree $T = ( V , E , v _ { 0 } )$ . Specifically, the root node $v _ { 0 }$ captures an initial profiling state using two fundamental microarchitectural signals: SM (Compute) utilization and DRAM (Memory) throughput. Guided by the roofline model, these two metrics classify any input kernel into one of four canonical bottleneck regimes: Latency-bound, Memory-bound, Compute-bound, or Algorithm-bound. This taxonomy forms the four initial outgoing edges from $v _ { 0 }$ , establishing a coarse-grained routing foundation before fine-grained metric selection. From this initial routing, the tree progressively grows into two types of nodes with distinct roles:

• Internal node: This specifies a routing condition over the profiling information collected so far, with outgoing edges corresponding to different outcomes of the condition.

• Leaf: This represents a terminal profiling state and determines the final metric configuration for the kernels reaching it.

Tree-Guided Profiling. Given a kernel k, the profiling tree sequentially evaluates its routing conditions and collects the required metrics until reaching a leaf. This produces a kernel-specific metric set $E _ { T } ( k )$ and the corresponding profiling information

$$
I _ { T } ( k ) = P ( k , E _ { T } ( k ) ) .
$$

Importantly, all profiling decisions are completed by the deterministic tree without LLM-based routing. Once the complete profiling information has been collected, it is provided to the LLM rewriter in a single inference:

$$
\hat { k } _ { T } = g ( k , I _ { T } ( k ) ) .
$$

Thus, the tree determines what hardware evidence to acquire, while the LLM consumes the complete evidence once to generate the final optimized kernel.

## 4.2 FEEDBACK-DRIVEN POLICY CONSTRUCTION

We next turn to how the profiling tree itself is constructed. The construction proceeds iteratively through three steps: collecting leaf-level optimization feedback, proposing local REWRITE or SPLIT operations, and retaining refinements that improve kernel optimization.

## 4.2.1 LEAF-LEVEL ATTRIBUTION AND STRUCTURAL UPDATES

Without loss of generality, we consider the current tree $T$ with root $v _ { 0 }$ and a leaf $v _ { i }$ . We route $N$ training kernels $\mathbf { \bar { \{ } }  k _ { j } \} _ { j = 1 } ^ { N }$ through $T _ { \mathbf { \delta } }$ , among which $N _ { i }$ kernels eventually reach $v _ { i }$ . Following the sequential profiling process defined in Section 4.1, each such kernel $k _ { j }$ arrives at $v _ { i }$ with a terminal profiling state

$$
\begin{array} { r } { s _ { t _ { i } } ^ { ( j ) } = \left( M _ { t _ { i } } ^ { ( j ) } , I _ { t _ { i } } ^ { ( j ) } \right) , } \end{array}
$$

where $M _ { t _ { i } } ^ { ( j ) }$ is the set of metrics collected along its routing path and $I _ { t _ { i } } ^ { ( j ) } = P ( k _ { j } , M _ { t _ { i } } ^ { ( j ) } )$ is the corresponding profiling information. After the collected information is provided to the fixed LLM rewriter, executing the resulting kernel on the target hardware yields the optimization reward $r _ { T } ( k _ { j } )$

Therefore, for the leaf $v _ { i }$ we obtain two forms of local evidence: their terminal profiling observation $I _ { t _ { i } } ^ { ( j ) }$ , which characterize the observed hardware behavior, and their optimization rewards $r _ { T } ( k _ { j } )$ which reflect the resulting optimization performance. Based on this local evidence, we consider two complementary forms of tree growth : REWRITE enriches the profiling information collected at $v _ { i }$ while SPLIT further partitions the kernels reaching $v _ { i }$ into different profiling regimes.

REWRITE. One possible limitation of the current leaf is that its profiling configuration does not expose sufficient information for effective optimization. We therefore consider augmenting its metric configuration with an additional metric group G:

$$
M _ { v } ^ { \prime } = M _ { v } \cup G , \qquad G \subseteq \mathcal { M } \setminus M _ { v } .
$$

REWRITE preserves the routing structure while enriching the information available to all kernels reaching v.

SPLIT. Another possibility is that kernels reaching the same leaf benefit from different profiling information. We therefore consider partitioning the leaf according to an already-collected profiling feature x and threshold θ:

$$
v  \{ { v _ { L } } , \quad x \leq \theta , 
$$

The resulting child leaves can adopt different profiling configurations, allowing the policy to specialize subsequent profiling decisions according to the observed hardware behavior.

For both operators, an LLM uses the local feedback to propose candidate refinements: additional metric groups for REWRITE, or feature–threshold pairs for SPLIT. The LLM only generates candidate modifications; whether a candidate is incorporated into the tree is determined by empirical evaluation in the subsequent stage.

## 4.2.2 CONSERVATIVE EMPIRICAL ACCEPTANCE

A candidate modification is accepted only if it demonstrates sufficient downstream improvement over the incumbent tree. For a candidate tree $T ^ { \prime }$ , we evaluate $T ^ { \prime }$ and the incumbent $T$ on the same set of training kernels and compute their paired reward difference:

$$
\frac { 1 } { N } \sum _ { k = 1 } ^ { N } \left[ r _ { T ^ { \prime } } ( k ) - r _ { T } ( k ) \right] ,
$$

where $r _ { T ^ { \prime } } ( k )$ and $r _ { T } ( k )$ denote the downstream rewards obtained by kernel (k) under the candidate and incumbent trees, respectively. The candidate is accepted only when

$$
\widehat { \Delta } ( T ^ { \prime } , T ) > \epsilon ,
$$

where ϵ is a conservative acceptance margin.

Paired evaluation controls for kernel-level variation by evaluating the candidate and incumbent trees on the same kernels, isolating the effect of the structural modification. When the rewriter involves stochastic sampling, the same sampling seed is used for the paired evaluations. We apply the operators conservatively. REWRITE is considered first as the minimal intervention for potential information insufficiency at the current leaf. If no REWRITE candidate satisfies the acceptance criterion, SPLIT is considered to capture potential heterogeneity within that leaf. A rejected modification leaves the incumbent tree unchanged. Together with the profiling budget (B), this controls the complexity of the learned tree and the amount of profiling information collected for each kernel.

Once construction terminates, the resulting tree is fixed and used for inference. The LLM is involved only during offline policy construction; inference-time routing follows the learned profiling conditions deterministically and requires no LLM.

## 5 EXPERIMENTS

## 5.1 EXPERIMENTAL SETUP

Datasets. We use separate datasets for profiling-policy construction and evaluation to avoid overlap between training and test kernels. The profiling tree is constructed solely on the CUDA-Agent-Ops-6K corpus without exposure to KernelBench. As a result, KernelBench serves as an out-ofdistribution benchmark to evaluate the policy’s transferability to unseen kernels.

For training, we use CUDA-Agent-Ops-6K (Dai et al., 2026), a curated dataset of 6,000 synthesized operator-level tasks released for CUDA kernel generation and optimization. The dataset is constructed by collecting reference operators from torch and transformers, synthesizing multi-operator tasks with LLMs, and filtering them based on execution correctness, determinism, runtime, and similarity to KernelBench test cases. This provides a diverse training corpus while reducing contamination with the evaluation benchmark.

For evaluation, we use KernelBench (Ouyang et al., 2025), a widely used benchmark for LLMbased GPU kernel optimization. It contains three levels of increasing complexity: Level 1 (100 problems) focuses on individual operators such as matrix multiplication and convolution; Level 2 (100 problems) targets fused operators such as Conv+ReLU+Bias; and Level 3 (50 problems) consists of complete models such as MobileNet, VGG, and MiniGPT. These levels cover a broad range of operator complexity and problem scale.

Baselines. We compare AutoProfiler with four training-free agentic kernel-optimization methods: CudaForge (Zhang et al., 2025), which iteratively compiles, benchmarks, and repairs generated kernels; KernelMem (Sun et al., 2026), which incorporates previous optimization experiences through memory-augmented prompting; KernelAgent (Cheng et al., 2026), which searches for improved kernels using correctness and performance feedback; and AutoKernel (Jaber & Jaber, 2026), which progressively refines kernels through single-edit modifications with rollback. These methods cover different inference-time optimization strategies, providing a diverse set of baselines for evaluating AutoProfiler.

Evaluation Metrics. Following the prior kernel optimization works Zhang et al. (2026); Ran et al. (2025), we evaluate model performance using execution accuracy to measure correctness and geometric-mean speedup to measure efficiency. Execution accuracy measures the proportion of problems where the generated kernel is both compilable and numerically correct against the PyTorch reference. Geometric-mean speedup is computed relative to the reference kernel across all problems, providing a scale-robust aggregate metric. Unlike the arithmetic mean, which is prone to severe distortion from extreme outliers, the geometric mean ensures that no single task disproportionately dominates the aggregate score, yielding a more balanced and representative evaluation (Fleming & Wallace, 1986).

Implementations. We employ GPT-5.6-Sol as the backbone kernel rewriter across AutoProfiler and all baseline methods. To ensure a fair comparison under equal inference budgets, each method is restricted to a single optimization pass (round = 1). For hardware profiling, baselines adopt their original static NCU metric configurations, whereas AutoProfiler dynamically selects profiling metrics guided by its learned policy. All methods share an identical NCU backend on four NVIDIA RTX 4090D GPUs (Ada Lovelace, 24GB). Specifically, the AutoProfiler rewriter targets Triton kernel generation. To isolate performance gains driven by profiling policy from baseline framework translation artifacts, each original PyTorch operator is first translated into a correctness-verified Tri ton seed prioritizing numerical fidelity over performance

## 5.2 MAIN RESULTS

Table 1 summarizes the performance across different methods on KernelBench. AutoProfiler achieves the highest average speedup of 1.704×, outperforming the strongest baseline, CudaForge (1.570×) by 8.5%. Notably, the performance advantage of AutoProfiler is most pronounced on the challenging Level 3 task, where it elevates the speedup from 1.099× of CudaForge to 1.352×. This substantial gain demonstrates that kernel-conditioned profile selection scales effectively to complex optimization spaces, going beyond simple operator-level enhancements.

In terms of correctness, AutoProfiler attains the highest execution accuracy at 87.2%, outperforming other baselines by 2.4–8.8 percentage points. Crucially, these efficiency and accuracy gains are achieved with minimal compute overhead: AutoProfiler requires only one single LLM call per task, whereas CudaForge and KernelMem consume 2 calls, and KernelAgent requires 4 calls. These re sults substantiate that profiling-guided hardware feedback serves as an efficient, lightweight prior for LLM-driven kernel optimization, delivering superior optimization quality while markedly reducing generation costs.

## 5.3 FURTHER ANALYSIS

## 5.3.1 COMPARISON OF PROFILING FEEDBACK STRATEGIES

We evaluate five distinct control conditions over a representative subset of 48 diverse kernels spanning across all difficulty levels: (1) Blind: provides no profiling metadata; (2) Random: attaches a randomly sampled profiling configuration; (3) Static: uses a fixed, globally optimal profiling setup; (4) AutoProfiler: dynamically assigns kernel-adaptive profiling trees; and (5) Oracle: assigns the empirical upper-bound configuration for each individual kernel, which is unavailable during inference. All other components, including the rewriter, optimization pipeline, hardware, and sampling seeds, are kept identical.

Table 1: Main performance comparison on KernelBench. “Speedup” denotes the geometric mean speedup over correctly solved problems, while “Accuracy” denotes the proportion of problems solved correctly. “LLM Calls/Task” indicates the average number of LLM inferences required per optimization task.
<table><tr><td rowspan="2">Method</td><td colspan="4">Speedup (↑)</td><td rowspan="2">Avg. Accuracy (%)</td><td rowspan="2">LLM Calls/Task</td></tr><tr><td>Level 1</td><td>Level 2</td><td>Level 3</td><td>Avg.</td></tr><tr><td>CudaForge</td><td>1.567×</td><td>1.808×</td><td>1.099×</td><td>1.570×</td><td>78.4</td><td>2</td></tr><tr><td>KernelMem</td><td>1.233×</td><td>1.351×</td><td>1.091×</td><td>1.252×</td><td>82.0</td><td>2</td></tr><tr><td>KernelAgent</td><td>1.361×</td><td>1.306×</td><td>1.044×</td><td>1.276×</td><td>84.8</td><td>4</td></tr><tr><td>AutoKernel</td><td>1.058×</td><td>1.182×</td><td>0.997×</td><td>1.095×</td><td>84.4</td><td>1</td></tr><tr><td>AutoProfiler</td><td>1.810×</td><td>1.775×</td><td>1.352×</td><td>1.704×</td><td>87.2</td><td>1</td></tr></table>

Table 3: Impact of profiling information quantity on downstream kernel optimization.

Table 2: Effect of profiling information on downstream kernel optimization.
<table><tr><td>Condition</td><td>Speedup (↑)</td></tr><tr><td>Blind</td><td>1.197</td></tr><tr><td>Random</td><td>1.215</td></tr><tr><td>Static</td><td>1.314</td></tr><tr><td>AutoProfiler</td><td>1.379</td></tr><tr><td>Oracle</td><td>1.573</td></tr></table>

<table><tr><td># Metrics</td><td>Regime</td><td>Speedup (↑)</td></tr><tr><td>2</td><td>Insufficiency</td><td>1.299×</td></tr><tr><td>4</td><td>Reference Baseline</td><td>1.370×</td></tr><tr><td>6</td><td>Redundancy</td><td>1.261×</td></tr><tr><td>8</td><td>Redundancy</td><td>1.231×</td></tr><tr><td>Static</td><td>一</td><td>1.219×</td></tr></table>

As summarized in Table 2, profiling feedback consistently improves kernel optimization. Our AutoProfiler achieves a 1.379× speedup, compared with 1.197× for Blind, and also outperforms both Random and Static. These results show that the benefit comes not only from providing profiling information, but from selecting feedback adaptively for each kernel. The remaining gap to Oracle suggests further room for improving profiling selection.

## 5.3.2 MORE PROFILING IS NOT ALWAYS BETTER

We further investigate how the volume of profiling feedback influences downstream optimization efficacy. Starting from the learned 4-metric setup as our reference, we construct reduced variants by pruning metrics and expanded ones by sampling additional metrics from the NCU pool. As presented in Table 3, downstream performance exhibits a clear non-monotonic trend with respect to profiling metric quantity. Reducing feedback from four metrics to two drops the geomean speedup from 1.370× to 1.299×. Conversely, over-saturating the context with six or eight metrics degrades performance further to 1.261× and 1.231×, respectively. These results demonstrate that effective profiling requires judicious metric selection to maximize information density, avoiding both metric scarcity and redundancy.

## 5.3.3 GENERALIZATION ACROSS LLM BACKBONES

To test whether the learned profiling policy captures intrinsic hardware requirements rather than overfitting to a specific LLM, we evaluate its transferability across model families. Specifically, we freeze the profiling tree trained on GPT-5.6-Sol and replace only the leaf kernel-rewriting LLM backbone with five distinct models, evaluating each on held-out benchmarks under the same pipeline.

As summarized in Table 4, the frozen profiling policy generalizes strongly across all backbones. When paired with Claude-Sonnet-5, the system achieves a peak geomean speedup of 1.47× and an execution accuracy of 91.7%, outperforming the training time baseline. Furthermore, across backbones from diverse model families (e.g., Kimi-K3, DeepSeek-V4-Pro, and Qwen3-Coder-Plus), the profiling policy consistently guides effective kernel optimization. These results confirm that the learned profiling tree encodes hardware optimization priors that are independent of the underlying LLM backbone rather than model specific biases.

![](images/ea472fac7a4780a9c6b8fe11ae2525aa1edf048293ee8d4a0af579d52550291d.jpg)

<table><tr><td>Configuration</td><td>Depth</td><td># Leaves</td><td>Speedup (↑)</td></tr><tr><td>AutoProfiler</td><td>11</td><td>58</td><td>1.61×</td></tr><tr><td>w/o REWRITE</td><td>2</td><td>4</td><td>1.47×</td></tr><tr><td>w/o SPLIT</td><td>7</td><td>37</td><td>1.53×</td></tr></table>

Figure 3: Comparison of optimization trajectories. Each circled marker represents an optimization round.  
Table 5: Ablation study on tree-construction operators.

Table 4: Generalization performance across different LLM backbones. The profiling tree is constructed using GPT-5.6-Sol feedback and kept strictly frozen; only the leaf LLM is swapped at test time. Speedup denotes the geometric mean over PyTorch baselines on held-out kernels.
<table><tr><td>Rewriter</td><td>Model Family</td><td>Speedup (↑)</td><td>Accuracy (↑)</td><td>Time (s, ↓)</td><td>Tokens (↓)</td></tr><tr><td>Claude-Sonnet-5</td><td>Anthropic</td><td>1.47×</td><td>91.7</td><td>8.2</td><td>1,840</td></tr><tr><td>Kimi-K3</td><td>Moonshot</td><td>1.38×</td><td>89.6</td><td>9.1</td><td>2,010</td></tr><tr><td>GPT-5.6-Sol</td><td>OpenAI</td><td>1.34×</td><td>87.5</td><td>10.4</td><td>2,260</td></tr><tr><td>DeepSeek-V4-Pro</td><td>DeepSeek</td><td>1.21×</td><td>83.3</td><td>12.7</td><td>2,580</td></tr><tr><td>Qwen3-Coder-Plus</td><td>Qwen</td><td>1.12×</td><td>79.2</td><td>14.3</td><td>3,120</td></tr></table>

## 5.3.4 ITERATIVE KERNEL REFINEMENT

We evaluate multi-turn optimization by recursively applying the frozen profiling tree over ten consecutive rounds, using the kernel produced in each step as the input for the next. Figure 3 reports the optimization trajectory as a function of cumulative wall-clock time per sample, providing a direct comparison with CudaForge under actual optimization cost. AutoProfiler improves rapidly from 1.629× in the first round to 2.112× after seven rounds, while requiring substantially less time per iteration. In comparison, CudaForge reaches 1.846× after four rounds under a similar cumulative time budget. These results show that the learned profiling policy not only supports stable iterative refinement, but also enables a more favorable optimization-cost trade-off by avoiding repeated expensive profiling and LLM-based profiling decisions.

## 5.4 ABLATION STUDY

As summarized in Table 5, removing either operator causes a clear performance drop, demonstrating that both components are essential for optimal tree construction. Removing REWRITE drops the geomean speedup from 1.61× to 1.47×. In this configuration, while SPLIT still divides kernels into separate branches, each leaf provides only basic metric feedback. Without the enriched profiling metrics, the downstream LLM lacks sufficient hardware context to guide fine-grained optimizations. Conversely, removing SPLIT lowers performance to 1.53×. Without adaptive node splitting, the tree cannot separate kernels with distinct bottlenecks, forcing very different workloads to share the same profiling setup. The full policy achieves the peak speedup of 1.61× by combining both operators. These results confirm that SPLIT groups kernels into specialized hardware regimes, while REWRITE ensures each regime receives the precise profiling details needed for LLM-driven code optimization.

## 6 CONCLUSION

In this paper, we present AutoProfiler, a novel feedback-driven framework that addresses the fundamental limitations of fixed hardware profiling in LLM-based GPU kernel optimization. Motivated by the observation that less profiling can yield better optimization, we shift from static metric collection to a profiling reasoning perspective, AutoProfiler learns a deterministic, kernel-conditioned profiling policy via iterative tree construction using REWRITE and SPLIT operations. Empirical results on KernelBench show that AutoProfiler outperforms traditional static profiling configurations and strong agentic baselines under constrained profiling budgets while incurring minimal LLM invocation overhead. Our findings suggest that hardware profiling should be treated as a learnable decision process, opening a promising direction for future research in AI-driven systems optimization.

## REFERENCES

Carlo Baronio, Pietro Marsella, Ben Pan, Simon Guo, and Silas Alberti. Kevin: Multi-turn rl for generating cuda kernels. In International Conference on Learning Representations (ICLR), 2026.

Martin Burtscher et al. Perfexpert: An easy-to-use performance diagnosis tool for hpc applications. In Proceedings of the ACM/IEEE International Conference for High Performance Computing, Networking, Storage and Analysis (SC), 2010.

Tianqi Chen, Thierry Moreau, Ziheng Jiang, Lianmin Zheng, Eddie Yan, Haichen Shen, Meghan Cowan, Leyuan Wang, Yuwei Hu, Luis Ceze, Carlos Guestrin, and Arvind Krishnamurthy. TVM: An automated end-to-end optimizing compiler for deep learning. In 13th USENIX Symposium on Operating Systems Design and Implementation (OSDI), pp. 578–594, 2018a.

Tianqi Chen, Lianmin Zheng, Eddie Yan, Ziheng Jiang, Thierry Moreau, Luis Ceze, Carlos Guestrin, and Arvind Krishnamurthy. Learning to optimize tensor programs. In Advances in Neural Information Processing Systems (NeurIPS), volume 31, 2018b.

Kaiming Cheng, Laura Wang, Jack Khuu, Mark Saroufim, Wenyuan Chi, Jiannan Wang, and Joe Isaacson. Kernelagent: Hardware-guided gpu kernel optimization via multi-agent orchestration. arXiv preprint, 2026.

Weinan Dai, Hanlin Wu, Qiying Yu, Huan-ang Gao, Jiahao Li, Chengquan Jiang, Weiqiang Lou, Yufan Song, Hongli Yu, Jiaze Chen, Wei-Ying Ma, Ya-Qin Zhang, Jingjing Liu, Mingxuan Wang, Xin Liu, and Hao Zhou. Cuda agent: Large-scale agentic rl for high-performance cuda kernel generation. arXiv preprint arXiv:2602.24286, 2026.

Siyuan Feng, Bohan Hou, Hongyi Jin, Wuwei Lin, Junru Shao, Ruihang Lai, Masahiro Masuda, Cody Hao Yu, and Tianqi Chen. Tensorir: An abstraction for tensor program program transformation. In Proceedings of the 28th ACM International Conference on Architectural Supportfor Programming Languages and Operating Systems (ASPLOS), volume 2, pp. 801–816, 2023.

Philip J Fleming and John J Wallace. How not to lie with statistics: the correct way to summarize benchmark results. Communications ofthe ACM, 29(3):218–221, 1986.

Jiading Gai, Shuai Zhang, Kaj Bostrom, Jin Huang, Vihang Patil, Haoyang Fang, Yuyang Wang, Huzefa Rangwala, and George Karypis. Optimizing cuda like a human: Micro-profiling tools as expert surrogates for llm-based gpu kernel optimization. arXiv preprint arXiv:2606.26453, 2026a.

Jiading Gai, Shuai Zhang, Kaj Bostrom, Jin Huang, Vihang Patil, Haoyang Fang, Yuyang Wang, Huzefa Rangwala, and George Karypis. Optimizing cuda like a human: Micro-profiling tools as expert surrogates for llm-based gpu kernel optimization. arXiv preprint arXiv:2606.26453, 2026b.

Jaber Jaber and Osama Jaber. Autokernel: Autonomous gpu kernel optimization via iterative agentdriven search. arXiv preprint arXiv:2603.21331, 2026.

NVIDIA Corporation. Nvidia nsight compute profiling guide. https://docs.nvidia.com/ nsight-compute/ProfilingGuide/, 2023.

Anne Ouyang, Simon Guo, Simran Arora, Alex L. Zhang, William Hu, Christopher Re, and Azalia Mirhoseini. Kernelbench: Can llms write efficient gpu kernels? In Proceedings of the 42nd International Conference on Machine Learning (ICML), volume 267, pp. 47356–47415, 2025.

Dezhi Ran, Shuxiao Xie, Mingfang Ji, Ziyue Hua, Mengzhou Wu, Yuan Cao, Yuzhe Guo, Hao Yu, Linyi Li, Yitao Hu, and Tao Xie. Kernelband: Boosting llm-based kernel optimization with a hierarchical and hardware-aware multi-armed bandit. CoRR, abs/2511.18868, 2025.

Junru Shao, Xiyou Zhou, Siyuan Feng, Bohan Hou, Ruihang Lai, Hongyi Jin, Wuwei Lin, Masahiro Masuda, Cody Hao Yu, and Tianqi Chen. Tensor program optimization with probabilistic programs. In Advances in Neural Information Processing Systems (NeurIPS), volume 35, pp. 35783– 35796, 2022.

Qitong Sun, Jun Han, Tianlin Li, Zhe Tang, Sheng Chen, Fei Yang, Aishan Liu, Xianglong Liu, and Yang Liu. Kernelskill: A multi-agent framework for gpu kernel optimization. arXiv preprint arXiv:2603.10085, 2026.

Anjiang Wei, Tianran Sun, Yogesh Seenichamy, Hang Song, Anne Ouyang, Azalia Mirhoseini, Ke Wang, and Alex Aiken. Astra: A multi-agent system for gpu kernel performance optimization. arXiv preprint arXiv:2509.07506, 2025.

Yunxiang Zhang, Ping Yu, Jianyu Wang, Max Fan, Julian Reed, Azalia Mirhoseini, and Will Su. Kernelbench-verified: Do llm-generated kernels actually beat pytorch? CoRR, abs/2607.16241, 2026.

Zijian Zhang, Rong Wang, Shiyang Li, Yuebo Luo, Mingyi Hong, and Caiwen Ding. Cudaforge: An agent framework with hardware feedback for cuda kernel optimization. arXiv preprint arXiv:2511.01884, 2025.

Lianmin Zheng, Chengfan Jia, Minmin Sun, Zhao Wu, Cody Hao Yu, Ameer Haj-Ali, Yida Wang, Jun Yang, Danyang Zhuo, Koushik Sen, Joseph E. Gonzalez, and Ion Stoica. Ansor: Generating high-performance tensor programs for deep learning. In 14th USENIX Symposium on Operating Systems Design and Implementation (OSDI), pp. 863–879, 2020a.

Size Zheng, Yun Liang Lin, Cheng Si Zhang, Xiuhong Ren, Yun-Sheng Chen, Lingxiao Huang, Yun Liang, and Xuechao Ning. Flextensor: An automatic schedule exploration framework for tensor computation on heterogeneous hardware. In Proceedings of the 25th International Conference on Architectural Support for Programming Languages and Operating Systems (ASPLOS), pp. 859–873, 2020b.

## A APPENDIX

## A.1 RELATED WORK

Traditional GPU Kernel Optimization. Automated GPU kernel optimization has long relied on search-based compilation frameworks and hardware-aware cost models. Early systems like TVM (Chen et al., 2018a) and AutoTVM (Chen et al., 2018b) introduced learned cost models to guide program space exploration. Ansor (Zheng et al., 2020a) expanded this search space via hierarchical program sampling, while MetaSchedule (Shao et al., 2022) unified rule-based and search-based program transformations. Further extensions such as FlexTensor (Zheng et al., 2020b) and TensorIR (Feng et al., 2023) specialized in tensor program abstractions. While effective, these compiler-based approaches operate on rigid, template-driven program spaces and treat hardware profiling primarily as passive static evaluation or fixed performance metrics, lacking the adaptive code reasoning capabilities offered by modern large language models.

LLM-driven Kernel Optimization and Agentic Frameworks. With the rise of LLMs, recent efforts leverage generative models for GPU kernel synthesis and iterative optimization. Benchmarks like KernelBench (Ouyang et al., 2025) have formalized the evaluation of LLM-generated CUDA and Triton kernels. Building on this, agentic systems integrate execution and runtime feedback into iterative search loops. For instance, CudaForge (Zhang et al., 2025) and AutoKernel (Jaber & Jaber, 2026) apply iterative compilation and error-repair heuristics, whereas KernelAgent (Cheng et al., 2026) and Astra (Wei et al., 2025) employ multi-agent orchestration for kernel refactoring. Additionally, Kevin (Baronio et al., 2026) and CUDA Agent (Dai et al., 2026) explore reinforcement learning and dynamic tool invocation to guide kernel synthesis. Despite their success, these frameworks treat hardware profiling as an invariant preprocessing step or a static feedback prompt. They overlook the fact that hardware profiling is inherently costly and that metric utility varies across heterogeneous GPU kernels.

Hardware Profiling: From Static Metric Collection to Profiling Reasoning. Hardware profiling tools such as NVIDIA Nsight Compute (NVIDIA Corporation, 2023) provide hundreds of microarchitectural performance counters, but full metric acquisition incurs non-trivial latency and context redundancy. Early performance diagnosis tools like PerfExpert (Burtscher et al., 2010) used static heuristic trees to isolate performance bottlenecks. Recent LLM-based micro-profiling assistants, such as KernelPro (Gai et al., 2026b), selectively trigger profiling tools based on predefined rule sets. However, existing methods either collect a universal metric subset indiscriminately or rely on hand-crafted profiling rules. In contrast, our work introduces the paradigm of profiling reasoning: we formulate hardware metric acquisition as a kernel-conditioned sequential decision process, explicitly learning a deterministic profiling policy driven directly by downstream code optimization rewards.

## A.2 EMPIRICAL VERIFICATION OF PROFILING-UTILITY HETEROGENEITY

To empirically validate the heterogeneity assumption underlying Proposition 1, we quantify the performance discrepancy between static metric selection and dynamic, per-kernel metric selection on a representative set of 128 kernels sampled from KERNELBENCH.

Candidate Configurations and Profiling Setup. We construct four static candidate profiling configurations, denoted as $\mathcal { C } = \{ m _ { \mathrm { a l g o } } , m _ { \mathrm { m e m } } , m _ { \mathrm { l a t } } , m _ { \mathrm { c o m p } } \}$ , designed to capture distinct, rooflinerelevant execution characteristics:

• Algorithm $( m _ { \mathrm { a l g o } } ) { \mathrm { : } }$ Captures high-level algorithmic behavior and control-flow patterns.

• Memory $( m _ { \mathrm { m e m } } ) \mathrm { : }$ Focuses on DRAM/SRAM memory bandwidth utilization and transaction efficiency.

• Latency $( m _ { \mathrm { l a t } } ) _ { \cdot }$ : Measures instruction pipeline stalls, execution latency, and warp scheduling overheads.

• Compute $( m _ { \mathrm { { c o m p } } } )$ : Tracks compute unit saturation, floating-point throughput, and ALU utilization.

All metric sets are fixed prior to evaluation. For each kernel $k \in \mathcal { K }$ and candidate configuration $m _ { i } \in \mathcal { C }$ , we extract the profiling information $I _ { i } ( k ) = P ( k , m _ { i } )$ and invoke a frozen kernel rewriter $g ( \cdot )$ under identical optimization, hardware, and random seed settings. We record the resulting speedup relative to the PyTorch reference implementation as $s _ { i } ( k ) = s ( k , m _ { i } )$

Static Benchmark vs. Per-Kernel Oracle. We compare the performance of the single best static configuration against an empirical per-kernel oracle. The optimal fixed configuration $m ^ { * }$ is chosen to maximize the geometric mean speedup across the benchmark:

$$
m ^ { * } = \underset { m _ { i } \in \mathcal { C } } { \mathrm { a r g m a x } } \exp \left( \frac { 1 } { | \mathcal { K } | } \sum _ { k \in \mathcal { K } } \ln s _ { i } ( k ) \right) .\tag{4}
$$

Conversely, the per-kernel oracle dynamically selects the optimal configuration independently for each individual kernel:

$$
s _ { \mathrm { o r a c l e } } ( k ) = \operatorname* { m a x } _ { m _ { i } \in \mathcal { C } } s _ { i } ( k ) .\tag{5}
$$

Kernels for which no metric configuration yields a speedup over the unoptimized reference implementation $( s _ { i } ( k ) \leq 1 . 0 )$ are tracked separately under non-improving cases.

Empirical Findings and Headroom Analysis. As detailed in Table 6, no static configuration consistently dominates across heterogeneous workloads. While the strongest fixed baseline Algorithm achieves a geometric mean speedup of 1.332×, the per-kernel oracle reaches 1.429×—delivering a 7.3% relative performance improvement.

To assess statistical confidence, we conduct 1, 000 bootstrap resamples over the 128 evaluation kernels, establishing a 95% confidence interval of [1.045×, 1.107×] for the relative gain of the oracle over the static baseline. Furthermore, the optimal profiling configuration shifts substantially across workloads: Memory provides the highest speedup for 31 kernels, Latency for 27, Algorithm for 26, and Compute for 23. These empirical observations highlight two key insights:

1. High Utility Heterogeneity: The value of profiling metrics is fundamentally kernel-dependent, rendering any fixed, one-size-fits-all profiling configuration inherently suboptimal.

2. Substantial Optimization Headroom: Tailoring hardware profiling signals to specific kernel bottlenecks yields meaningful execution gains, underscoring the need for an adaptive profiling policy like AUTOPROFILER.

Table 6: Empirical heterogeneity of profiling utility on 128 KernelBench kernels. Each candidate configuration is evaluated with the same frozen kernel optimizer.The “Best on # Kernels” column counts kernels for which the configuration achieves the highest downstream speedup among the four candidates. The oracle selects the best candidate independently for each kernel.
<table><tr><td>Profiling Configuration</td><td>Geomean Speedup</td><td>Best on # Kernels</td></tr><tr><td>Algorithm</td><td>1.332×</td><td>26</td></tr><tr><td>Memory</td><td>1.277×</td><td>31</td></tr><tr><td>Latency</td><td>1.242×</td><td>27</td></tr><tr><td>Compute</td><td>1.239×</td><td>23</td></tr><tr><td>Per-kernel Oracle</td><td>1.429×</td><td>128</td></tr></table>