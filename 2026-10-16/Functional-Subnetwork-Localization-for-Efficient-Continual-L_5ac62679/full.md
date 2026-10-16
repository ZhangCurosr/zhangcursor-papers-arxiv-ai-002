# Functional Subnetwork Localization for Efficient Continual Learning in Large Language Models

Shuoran Jiang, Wentao Zhu, Yang Xiang, Dongfeng Li, Yunan Zhang, Mengchen Zhao, Yuhui Guo, Kai Chen, Youcheng Pan, Xiangping Wu, Qingcai Chen

Abstract—Biological intelligence continually adapts to new information while retaining prior skills through localized plasticity. In contrast, continual learning in large language models (LLMs) remains vulnerable to catastrophic forgetting. A promising direction is to identify functionally specialized subnetworks within LLMs that enable targeted and resource-efficient adaptation. Using gradient-attribution profiling, we localize cognition-associated functional subnetworks exhibiting extremely low mutual overlap. Building on this finding, we propose Function-isolated Memory-Efficient Optimization (CiMEO), a continual learning framework that routes each training example to its corresponding functional subnetwork, restricting parameter updates and optimizer states to selected regions. Across 1.4 million sequential instructions, CiMEO achieves the highest average performance among the compared methods on five generative benchmarks spanning programming, examination-style tasks, and mathematical reasoning, while mitigating catastrophic forgetting. Subspace-overlap and principal-angle analyses further indicate that localized updates help preserve pre-trained representations, highlighting functional subnetwork localization as a practical approach to efficient and stable continual learning in LLMs.

Index Terms—Continual Learning, Modular Subnetworks, Gradient Attribution, Stability-Plasticity Trade-off, Catastrophic Forgetting, Memory-Efficient Optimization.

## I. INTRODUCTION

ARGE language models (LLMs) are increasingly degeneration systems [1], which must be updated after pretraining to incorporate new knowledge, align with evolving preferences and expand to new domains [2], [3]. This is the research content of continual learning for LLMs, the central challenge in which is not merely learning new data, but preserving previously acquired knowledge [4]. However, new updates in LLMs’ continual learning usually overwrite earlier skills and degrade general, which induces the catastrophic forgetting [5], [6]. This forgetting problem reflects a fundamental tension that LLMs must remain plastic enough to acquire new behaviors, while stay stable enough to retain old ones [7]. For long-term deployed LLMs, maintaining stability is particularly crucial because it ensures the coherence of user experience, and it is also particularly challenging because updates are typically applied to global parameters, allowing interference to accumulate throughout the entire deployment lifecycle. Practical constraints further exacerbate the problem, as standard optimizers maintain dense states. Therefore, attempts to improve stability through repeated trials can become prohibitively expensive in terms of memory and computation [8].

Existing continual learning methods improve stability mainly by (i) regularizing parameter drift, (ii) rehearsing stored past examples, or (iii) distilling behavior from an earlier model [9]. However, these strategies become difficult to scale for LLMs’ continual learning. Regularization methods, such as elastic weight consolidation (EWC) [10], [11], require estimating and storing parameter importance at task boundaries, which increases extra computation and state beyond standard training. Rehearsal-based approaches mitigate forgetting by replaying data from a memory buffer [12], but this introduces a persistent memory footprint. Their effectiveness also degrades when the buffer is small, and will be amplified in largescale parameter settings. Distillation-based methods typically require keeping a previous model and performing additional forward passes to match prior predictions [13], further increasing compute overhead. Crucially, many of these stabilityoriented methods still assume global parameter updates with Adam-like optimizers, whose first- and second-moment accumulators add substantial state memory, making repeated posttraining prohibitively expensive under strict budgets [14], [15]. As surveyed in recent reviews [16], [17], this cost–stability mismatch remains a key obstacle for LLMs’ long-horizon deployment [18].

Biological learning offers a useful contrast: continual adaptation in the human brain is thought to local synaptic plasticity [19], [20], where learning is expressed through the selective recruitment of task-relevant neuronal ensembles while other circuits remain comparatively stable [19]–[23] as shown in Fig.1. This suggests a concrete design principle for the continual learning of Large Language Models (LLMs), which involves identifying reusable subnetworks according to a certain principle. Every incoming sample is allocated to a specific subnetwork to update parameters in those isolated subspaces, so that new learning is localized rather than globally disruptive.

![](images/11d4f392cfe0ef6ed33630e37c22039ea8a86f4e413fadf0989451cfd778f3fc.jpg)  
Fig. 1. localized plasticity in human barin vs. global plasticity in LLMs. (a), Learning recruits and co-activates sparse neuronal ensembles (circles; color indicates activity) across memory-relevant regions, including the hippocampus (HPC), cortex, and subcortical areas, forming an engram. During the first hours to days after learning, synaptic connectivity is preferentially strengthened among HPC engram cells and between HPC and other regions, enabling recent memory expression. Over subsequent days to weeks, systems consolidation supported by hippocampo–cortical replay and synchronized activity (notably involving prefrontal cortex, PFC) promotes maturation of cortical engram cells. This process drives strengthening of synaptic connectivity between corticalcortical and cortical-subcortical engram cells and a progressive shift in the engram cell network toward a higher dependency on cortical engram cells for remote memory expression. (b), Global parameter updates across sequential batches bring memory re-consolidation. During training on successive batches, learning is typically implemented via dense, global weight updates across the network. Lacking a brain-like mechanism of sparse engram allocation and multi-stage consolidation, new batches are absorbed by rewriting shared parameters, which can interfere with or overwrite earlier representations. As a result, the model’s “memory” is continually reconfigured by the order of batches, reflecting a stability–plasticity trade-off and potential forgetting under sequential learning.

Motivated by this principle, we investigate whether pretrained LLMs exhibit functionally specialized parameter regions that can support localized adaptation during continual learning. Guided by Bloom’s taxonomy [24], we employ gradient-attribution profiling to estimate parameter importance across six cognitive categories and localize the corresponding functional subnetworks. The identified subnetworks exhibit remarkably low mutual overlap, averaging below 0.002% across cognitive categories. This observation motivates a functional subnetwork localization strategy that restricts parameter updates to cognition-relevant regions while minimizing interference with previously acquired knowledge.

Building on this observation, we propose Cognition-isolated Memory-Efficient Optimization (CiMEO), a framework for efficient continual learning in LLMs. CiMEO routes each incoming instruction to its corresponding functional subnetwork, selectively updates relevant parameters, and maintains optimizer states only for the selected indices. By integrating cognition-guided routing, functional subnetwork localization, and memory-efficient optimization, CiMEO aims to mitigate cross-skill interference and catastrophic forgetting while reducing the optimizer-state overhead of continual adaptation without relying on replay buffers.

Overall, the contributions of this study are:

• Functional subnetwork localization: We identify cognition-associated functional subnetworks in pretrained LLMs through gradient-based attribution, revealing an average overlap below 0.002% across cognitive categories and providing empirical evidence of localized parameter importance.

• Cognition-guided continual learning: We propose CiMEO, a continual learning framework that routes incoming instructions to functionally relevant subnetworks and restricts parameter updates to selected regions to mitigate interference between successive learning stages.

• Memory-efficient optimization and analysis: We integrate sparse, state-serialized optimization to reduce optimizer-state requirements and investigate the stability of pretrained representations through subspace-overlap and principal-angle analyses.

## II. RELATED WORKS

## A. Continual Learning and Interference Mitigation

Continual learning seeks to preserve previously acquired capabilities while incorporating information from a nonstationary sequence of tasks. Early approaches address this stability–plasticity dilemma through complementary mechanisms. Elastic weight consolidation (EWC) penalizes changes to parameters estimated to be important for earlier tasks [10], whereas Learning without Forgetting preserves previous predictions through distillation on newly available data [25]. Rehearsal-based methods revisit earlier examples; for language models, LAMOL instead generates pseudo-examples of previous tasks during subsequent training [26]. Such strategies can reduce forgetting, but their regularization statistics, replay generation, or teacher-based constraints introduce costs or dependencies that differ from direct parameter isolation.

Subspace-constrained adaptation offers another route to mitigating interference. O-LoRA assigns successive languagemodel tasks to approximately orthogonal low-rank subspaces without storing rehearsal data [27]. InfLoRA constructs lowrank update subspaces designed to avoid interference with previous tasks, primarily evaluated in visual continual learning [28]. More recently, CLoRA regularizes LoRA update directions to reduce forgetting during continued LLM training [29]. These approaches motivate limiting the interaction between consecutive updates; in contrast, CiMEO restricts updates to cognition-attributed weight coordinates rather than learning or constraining a separate low-rank update for each task.

## B. Parameter Isolation, Sparse Adaptation, and Routing

Parameter-isolation methods allocate distinct portions of model capacity to different learning objectives. PackNet sequentially prunes and retrains a shared network to reserve previously learned weights while freeing parameters for new tasks [30]. Piggyback learns task-dependent binary masks over fixed pretrained weights [31]; Diff Pruning learns sparse taskspecific parameter differences in pretrained language models [32]. Rather than modifying backbone weights, Learning to Prompt (L2P) selects and optimizes small prompts from a learned pool, enabling rehearsal-free continual learning in vision settings without task identity at inference time [33]. In language-model adaptation, LoRA [34] and sparse approaches such as SHiRA [35] likewise reduce the extent of trainable parameter changes. More directly, PaRSP identifies taskspecific functional regions in LLMs and constrains updates using null-space orthogonality to balance retention, adaptation, and remaining capacity [36]. These results reinforce the value of localized adaptation, while parameter efficiency alone does not imply protection against interference across a sequence of updates.

These studies establish that allocating or selecting a limited set of parameters can preserve useful representations. CiMEO differs in its allocation granularity and routing criterion: it groups incoming instructions by six Bloom-inspired cognitive categories [24], identifies attribution-ranked coordinates within the pretrained model, and uses a frozen cognition classifier to route subsequent updates. Thus, the proposed partition is based on cognitive labels rather than only dataset or task identifiers, and it is paired with sparse optimizer-state storage.

## C. Functional Localization and Gradient-Based Attribution

Evidence for localized functionality in Transformers provides a direct motivation for selective updates. Dai et al. identified knowledge-associated neurons in pretrained Transformers through attribution and explored targeted modifications of factual knowledge [37]. Wang et al. identified task-predictive skill neurons, showing that perturbing them can affect task performance [38]. Surgical fine-tuning further demonstrates that restricting adaptation to selected layers can outperform broader updates under certain distribution shifts [39]. These findings support examining where learning signals act, but neither neuron attribution nor a sparse mask by itself establishes that a model contains causally independent cognitive modules.

Most closely related to our study, Leng and Xiong [40] used gradient attribution to identify task-specific neurons in LLMs, analyzed their overlap, and proposed neuron-level selective fine-tuning for continual learning. Hence, gradientbased identification and localized continual updates should not be regarded as novel in isolation. CiMEO investigates a different organizational axis—a shared taxonomy of cognitive categories—and couples element-wise cognition-specific coordinate selection with instance-level routing and state-serialized AdamW optimization. Our overlap and weight-subspace analyses characterize this particular setting; the observed separation is empirical and should not be interpreted as proof of strict functional independence.

## D. Memory-Efficient Optimization for LLM Adaptation

Reducing interference is only one requirement of sustained LLM adaptation: optimizer moments and gradients can also dominate the training-memory budget. Parameterefficient techniques such as LoRA [34] shrink the trainable state by introducing compact update matrices. LOMO reduces full-parameter fine-tuning memory by fusing gradient computation and parameter updates [8], while AdaLOMO further targets memory-efficient optimization [41]. GaLore instead projects gradients into low-rank spaces to reduce optimizerstate memory while retaining full-parameter learning [42].

![](images/2ec809ae2dbd92b9cb7b3f5243e604e7ba734dc0c5d24bfb0613a5a8cc1982e2.jpg)

![](images/ea4155b64a8384278471067c97f3f109ddd3dfed67ac25188864aac55daf12f0.jpg)

![](images/0a54c4fa7b96bcbca04896eb06a6757e3b60ac0301edf376439ff926ae295fa9.jpg)

![](images/3c2c431e512da08457843ba7a44bf7ac4924c43dc4be4bf6c0901efbbbb0e836.jpg)  
b.  
Neuronal sensitivity for cognition

![](images/26444c0e0105471e7a2dd084030db82b2882c07e77b3fea46efe97eca2e3d88e.jpg)

![](images/33e6ae524fe16044b7cc00b2cf9394254e119f1ef87ee74d01bd10c8d92731b3.jpg)  
e.

f.  
![](images/76a89d66bbcf692a523668bfe1b3362b12385e6501c6918325f370f2330d96d1.jpg)  
Fig. 2. Framework for discovering endogenous cognitive modularity in LLMs and applying it to memory-efficient training. a, We construct a cognitive corpus from the AM-DeepSeek-R1-Distilled-1.4M instruction dataset, where each instruction is annotated with a Bloom’s taxonomy label. b, Cognition-isolated subnetworks are identified via gradient attribution: labelled instructions are fed into the LLM to predict cognition categories, and the gradients accumulated during back-propagation are aggregated to locate cognition-sensitive subnetworks. c, The state-serialized AdamW optimizer maintains a mapping between sparse parameters and their locations in the original dense tensors, so that only cognition-relevant subnetworks are activated and updated during training. d, For continual learning, CiMEO employs a cognitive classifier to assign a cognition label to each input and routes optimization to the corresponding local subnetwork. e, Experiments on five benchmarks report detailed incremental learning and anti-forgetting performance, together with weight-space dynamics during continual learning. f, We further analyze subspace overlap and principal angles across checkpoints to quantify interference and stability in the learned weight subspaces.

GCL further combines group-shared parameter adaptation and gradient-aware regularization to continually fine-tune already sparse LLMs while preserving their sparsity patterns [43]. In comparison, CiMEO focuses on selecting cognition-relevant coordinates within a pretrained dense backbone and storing optimizer states only for those coordinates. Most memoryoriented approaches primarily address the representation or storage of updates, rather than deciding which cognitionrelevant original weights should change for each incoming instruction.

CiMEO combines both decisions: gradient attribution selects a small set of original weight coordinates, a cognition classifier chooses the relevant set for each example, and S<sup>2</sup>-AdamW maintains optimizer moments for selected indices. This design aims to control cross-category interference and persistent optimizer-state memory jointly. As with other sparse-update schemes, reductions in stored optimizer states should be distinguished from reductions in activation memory or dense backward-pass computation; the latter depend on the implementation and are not guaranteed by a parameter mask

alone.

## III. METHOD

This study first collects a cognitive corpus to identify cognition-isolated subnetworks and organizes language generation tasks into Bloom’s taxonomy. Next, to reduce the memory cost of optimization, we propose a state-serialized AdamW. Finally, we define a continual learning paradigm based on the proposed CiMEO.

## A. Cognitive corpus collection

As shown in Fig.2a, given a corpus $\mathcal { D } ~ = ~ \{ x _ { i } \} _ { i = } ^ { N }$ 1 with N plain-text sentences, we assign each x a cognitive label $y _ { i }$ in Bloom’s taxonomy, including memory, understanding, application, analysis, evaluation, and creation. Let $f ( \cdot ; \theta )$ denote a frozen instruction-tuned LLM, each sample $x _ { i }$ is converted into a multiple-choice prompt $Q ,$ where the model is asked to select the correct option.

We use beam search to efficiently generate high-probability sequences efficiently. Beam search maintains multiple candidate sequences (beams) at each decoding step, expands them incrementally and retains only the top-k sequences with the highest normalized scores. The procedure terminates when all beams generate an end-of-sequence token or reach the maximum length. For each option $O _ { j }$ , its selection frequency $f ^ { ( k ) } ( x _ { i } ; \theta )$ is computed across all runs. The option with the highest frequency is selected as the final answer:

$$
y _ { i } = \arg \operatorname* { m a x } _ { O } \{ f ^ { ( 1 ) } ( x _ { i } ; \theta ) , \cdot \cdot \cdot , f ^ { ( K ) } ( x _ { i } ; \theta ) \}\tag{1}
$$

Finally, the cognitive Corpus is $\mathcal { D } ^ { \star } = \{ x _ { i } , y _ { i } \} _ { i = 1 } ^ { N }$

## B. Cognition-isolated subnetwork identification

As shown in Fig. 2b, to identify endogenous cognitionisolated subnetworks in LLMs, we employ a gradient attribution approach. The key idea is to measure the sensitivity of each parameter to a specific cognition category by aggregating the gradient magnitudes induced by cognition-labeled supervision. The procedure consists of the following four steps.

1st Step: Sample selection For each cognition label in Bloom’s taxonomy, we sample M labeled instances from the cognitive corpus ${ \mathcal { D } } ^ { \star }$ , yielding subsets $\widehat { \cal D } _ { c } = \{ ( x _ { j } , y _ { j } ) \} _ { j = 1 } ^ { M }$

2nd Step: Cognition prediction and supervised loss Let $f ( \cdot ; \pmb \theta )$ denote the frozen LLM, and each input x is converted into a multiple-choice cognition classification prompt. Given the gold cognition label $y ,$ we define the supervision loss as the negative log-likelihood:

$$
C ( x , y ; \pmb { \theta } ) = - \log p _ { \pmb { \theta } } ( y \mid x ) .\tag{2}
$$

where $p _ { \theta } ( y | x )$ denotes the prediction probability on prompt x.

3rd Step: Gradient attribution and importance aggregation For each parameter $\theta _ { k } \in \pmb \theta$ , we compute its gradient and use its magnitude as the attribution score. Specifically, for

cognition category c, we aggregate the gradient magnitudes over $\widehat { \mathcal { D } } _ { c }$ to obtain a parameter-importance estimate:

$$
A _ { \theta _ { k } } ^ { ( c ) } = \frac { 1 } { M } \sum _ { m = 1 } ^ { M } \left| \frac { \partial } { \partial \theta _ { k } } \mathcal { L } ( x _ { m } , y _ { m } ; \pmb { \theta } ) \right| .\tag{3}
$$

The resulting $A _ { \theta _ { k } } ^ { ( c ) }$ reflects how strongly parameter $\theta _ { k }$ responds to supervision signals from cognition c.

4th Step: Subnetwork selection and hash-table storage For each weight tensor $\theta ,$ we rank its elements by $\mathbf { A } ^ { ( c ) }$ and select the top $\mathrm { p } \%$ as the cognition-isolated subnetwork for category c:

$$
\pmb { \theta } _ { s } ^ { ( c ) } = \{ \theta _ { k } \in \pmb { \theta } \ | \ A _ { \theta _ { k } } ^ { ( c ) } \geq Q _ { 1 - p } ( \mathbf { A } ^ { ( c ) } ) \} ,\tag{4}
$$

where $Q _ { 1 - p } ( \mathbf { A } ^ { ( c ) } )$ denotes the (1-p)-percentile threshold of attribution scores.

To enable efficient sparse updates in the subsequent stateserialized AdamW optimizer, we store the tensor positions of selected parameters in a hash table:

$$
\mathcal { H } ^ { ( c ) } = \{ \mathrm { i n d e x } ( \theta _ { k } ) \ | \ \theta _ { k } \in \pmb { \theta } _ { s } ^ { ( c ) } \} .\tag{5}
$$

This procedure yields six cognition-specific subnetworks $\{ \mathbf { W } _ { s } ^ { ( c ) } \}$ , which can then be used for cognition-aware routing and sparse, memory-efficient optimization in continual learning.

## C. State-serialized AdamW

As shown in Fig.2c, to achieve memory-efficient parameter updating on cognitive isolated subnetworks, we serialize the parameters and corresponding gradients, moments in AdamW via the hash table $\mathcal { H } ^ { ( \bar { c } ) }$ , as illustrated in Fig.2c.

Notations used in the proposed state-serialized AdamW $( \mathrm { S ^ { 2 } \mathrm { - } A d a m W ) }$ are defined as follows,

$f ( \pmb { \theta } ) \in \mathbb { R }$ is the loss of LLMs,

$\mathbf { g } _ { t } \in \mathbf { R } ^ { m \times n }$ is the gradient on weight θ,

$\mathcal { H } ^ { ( c ) }$ maps the sparse parameters and positions in cogni tion c isolated subnetworks,

$\hat { \pmb { g } _ { t } } \overleftarrow { \frac { \mathcal { H } ^ { ( c ) } } { \mathcal { I } t } } \overleftarrow { \pmb { g } _ { t } }$ serialized sparse gradients,

$\hat { m } _ { t }$ is the exponential moving average (EMA) on $\hat { \mathbf { g } } _ { t }$

$\hat { \mathbf { } } _ { \pmb { s } _ { t } }$ is the EMA on $\hat { \mathbf { \mathbf { g } } } _ { t } ^ { 2 }$

## IV. RESULTS

## A. Endogenous cognition recognizability in LLMs

This study first addresses whether pre-trained LLMs already allocate task-relevant subnetworks. We use the AM-DeepSeek-R1-Distilled-1.4M [44] instruction dataset (distilled from DeepSeek-R1) to identify cognition-relevant subnetworks via a gradient attribution method as defined in Fig.2b. Specifically, all instructions are annotated according to Bloom’s taxonomy, as illustrated in Fig.2a. We then use these cognitionannotated instructions to identify isolated subnetworks in the LLMs. The overall procedure is illustrated in Fig.2b.

Fig.3a reports the overlap rates between sparse subnetworks across all cognition pairs. All overlaps are below 0.002%, suggesting that pre-trained LLMs exhibit an endogenous partition of cognition. This observation supports our motivation to develop cognition-isolated, memory-efficient fine-tuning in this study.

![](images/ed333a2a49c1907b84ded0581e93295e472a30c1f3491828b6f391caf0f277f4.jpg)

![](images/1985df9021427bcc34d61f58db550f3f217f9e7b9c4abcf18db4860dd1aea9b6.jpg)  
b. average overlap over layers

![](images/f914b4e7c60b0c7b261c2db1f73f2a195edac5547a4b8db7f8f374bb5860ed52.jpg)  
c. overlaps over layers on every cognition pair

Fig. 3. The cognition-isolated subnetwork overlap rate analysis. a, The average overlaps across all layers on all cognition pairs. b, The average overlap rates over all cognition pairs across all layers. c, The detailed overlaps over every cognition pair and all layers.  
```latex
Algorithm 1 $S ^ { 2 } { \cdot } A d a m W ; g _ { t } ^ { 2 }$ indicates the element-wise square
${ \bf { g } } _ { t } \odot { \bf { g } } _ { t } , \alpha = 0 . 0 0 1 , \beta _ { 1 } = 0 . 9 , \beta _ { 2 } = 0 . 9 9 9$ and $\epsilon = 1 0 ^ { - 8 }$
All operations are element-wise on vectors, and $\beta _ { 1 } ^ { t }$ and $\beta _ { 2 } ^ { t }$
represent the power t.
1: Require: Step size α, exponential decay rates $\beta _ { 1 } , \beta _ { 2 } \in$
$[ 0 , 1 )$ , scattering subnetwork parameter maps H and ob
jective function $f ( \pmb \theta )$
2: Initialize: $\pmb { \theta } _ { 0 } , \pmb { m } _ { 0 } \gets \mathbf { 0 } , v _ { 0 } \gets \mathbf { 0 } , t \gets 0$
3: while θ not converged do
4: $t \gets t + 1$
5: $\bar { \pmb { \theta } } _ { t }  \pmb { \theta } _ { t }$
6: $\pmb { g } _ { t }  \nabla \theta ^ { f } ( \pmb { \theta } _ { t - 1 } ) , \quad \bar { g _ { t } }  \frac { \pmb { \mathcal { D } } } { \pmb { g } _ { t } }$
7: $\pmb { m } _ { t } \gets \beta _ { 1 } \pmb { m } _ { t - 1 } + \left( 1 - \beta _ { 1 } \right) \pmb { g } _ { t }$
8: ${ \pmb v } _ { t } \gets \beta _ { 2 } { \pmb v } _ { t - 1 } + \left( 1 - \beta _ { 2 } \right) { \pmb g } _ { t } ^ { 2 }$
9: $\hat { m } _ { t } \gets m _ { t } / ( 1 - \beta _ { 1 } ^ { t } )$
10: $\hat { \pmb { v } } _ { t }  \pmb { v } _ { t } / ( 1 - \beta _ { 2 } ^ { t } )$
11: $\bar { \theta } _ { t }  \bar { \theta } _ { t - 1 } - \alpha \cdot \hat { m } _ { t } / \sqrt { \hat { v } _ { t } + \epsilon }$
12: $\mathbf { \nabla } _ { \theta } \mathcal { L } \stackrel { \mathcal { D } } { \mathop {  } } \bar { \theta }$
13: end while
14: return $\pmb { \theta }$
```

We further analyze how subnetwork overlaps vary across layers. Fig.3b shows the overlaps across representation layers, where most layers remain at a similarly low level, except for the 2nd and 3rd layers, which exhibit higher overlap. To examine each cognition pair in detail, Fig.3c plots the layerwise overlap curves for all cognition pairs. The trends are consistent with the average results in Fig.3b, with pronounced overlap peaks at the 2nd and 3rd layers across cognition pairs.

Taken together, these results align with recent findings [45]– [47] that the first few layers of LLMs tend to encode more general representations, exhibiting weaker cognitive discrimination than deeper layers. As depth increases, representations gradually become more task- and cognition-specific.

## B. Continual learning performance

After identifying the cognition-isolated subnetworks, we propose a subnetwork-local back-propagation algorithm, termed cognition-isolated memory-efficient optimization (CiMEO). To enable memory-efficient optimization over sparse subnetworks within dense LLMs, we reformulate AdamW into a state-serialized variant, as illustrated in Fig.2c. In addition, we pre-train a cognitive classifier on the cognition-annotated AM-DeepSeek-R1-Distilled-1.4M instruction dataset. This classifier is then frozen during continual learning and is used to route each incoming sample to its corresponding subnetwork for parameter updates.

We evaluate CiMEO on the AM-DeepSeek-R1-Distilled-1.4M instruction dataset. Baselines include full-parameter finetuning with AdamW [48], as well as PEFT methods such as LoRA [34], MiSS [49], Bone [50], Shira [35], Dora [51], Eva [52], Lorafa [53], Olora [54], Pissa [55], Road [56], and MEFT methods of AdaLOMO [41], Galore [42], AdaRankGrad [57] and ZO-AdaMU [58]. All compared methods use the same training setup: batch size $B = 6 4$ , a constant learning rate $r = 1 e - 5 ,$ , and model parameters. For PEFT methods, the low-rank dimension is set to $r = 1 6$ , and when an α parameter is used, we set $\alpha = 3 2$ for the query, key, value, output, gate, up and down projection weights in the Transformer.

All models are evaluated on five representative and challenging benchmarks. (i) HumanEval-CN [59], which measures code reasoning and syntax fidelity for executable code generated from Chinese prompts. (ii) GaokaoBench 2010 [60], which assesses complex, multi-disciplinary reasoning and problem-solving on the 2010 Chinese National College Entrance Examination (Gaokao). (iii) CMMLU [61], which tests multitask language understanding across 67 subjects spanning humanities, social sciences, STEM, and medicine. (iv) CEVAL [62], which emphasizes expert-level question answering across civil service, legal, medical, and other professional exams. (v) GSM8K [63], a grade school math dataset focusing on arithmetic and basic logical reasoning.

Table I summarizes the results on the five benchmarks. Although CiMEO does not achieve the top score on any single benchmark, it attains the best overall balance across tasks, yielding the highest average performance (38.92%). On humaneval cn benchmarks, Olora achieves the best score (29.27%), followed by CiMEO (25.00%). On the GaoKaoBench 2010, CiMEO reaches 6.98%, comparable to LoRA. On CMMLU, CiMEO improves by 28.85% over the baseline. On CEVAL and GSM8K, CiMEO performs nearly as well as the top-1 methods. Overall, CiMEO provides consistent gains across diverse evaluation settings, rather than excelling on a single benchmark.

<table><tr><td colspan="2"></td><td>humaneval_cn pass@1</td><td>gaokao Acc (%)</td><td>cmmlu Acc (%)</td><td>ceval Acc (%)</td><td>gsm8k Acc (%)</td><td>Avg</td></tr><tr><td colspan="2">Qwen3-1.5B-Base</td><td>4.27</td><td>5.14</td><td>0.02</td><td>58.01</td><td>66.87</td><td>26.86</td></tr><tr><td>Full-Parameter</td><td>Adamw</td><td>3.05</td><td>11.42</td><td>0.00</td><td>67.40</td><td>39.27</td><td>24.23</td></tr><tr><td rowspan="10">PEFT</td><td>Lora</td><td>3.66</td><td>6.99</td><td>0.02</td><td>60.44</td><td>68.08</td><td>27.84</td></tr><tr><td>Miss</td><td>8.54</td><td>11.15</td><td>0.00</td><td>66.73</td><td>66.79</td><td>30.64</td></tr><tr><td>Bone</td><td>9.76</td><td>11.29</td><td>0.00</td><td>66.62</td><td>66.87</td><td>30.91</td></tr><tr><td>Shira</td><td>1.22</td><td>9.26</td><td>29.61</td><td>63.13</td><td>68.01</td><td>34.25</td></tr><tr><td>Dora</td><td>0.00</td><td>6.81</td><td>19.58</td><td>63.08</td><td>66.26</td><td>31.14</td></tr><tr><td>Eva</td><td>0.00</td><td>9.90</td><td>0.05</td><td>52.19</td><td>19.71</td><td>16.37</td></tr><tr><td>Lorafa</td><td>6.10</td><td>9.62</td><td>48.40</td><td>60.88</td><td>66.03</td><td>38.20</td></tr><tr><td>Olora</td><td>29.27</td><td>10.91</td><td>0.09</td><td>67.06</td><td>35.10</td><td>28.49</td></tr><tr><td>Pissa</td><td>3.05</td><td>11.42</td><td>0.00</td><td>67.40</td><td>39.27</td><td>24.23</td></tr><tr><td>Road</td><td>5.49</td><td>8.85</td><td>10.78</td><td>59.48</td><td>67.40</td><td>30.40</td></tr><tr><td rowspan="4">MEFT</td><td>AdaLOMO</td><td>5.49</td><td>7.58</td><td>0.85</td><td>58.87</td><td>67.32</td><td>28.02</td></tr><tr><td>Gelore</td><td>13.41</td><td>9.00</td><td>16.44</td><td>63.00</td><td>66.72</td><td>33.71</td></tr><tr><td>AdaRankGrad</td><td>14.02</td><td>5.51</td><td>17.66</td><td>65.24</td><td>66.49</td><td>33.78</td></tr><tr><td>ZO-AdaMU</td><td>3.05</td><td>10.54</td><td>5.18</td><td>64.26</td><td>67.25</td><td>30.06</td></tr><tr><td rowspan="3">Ours</td><td>CiMEO</td><td>25.00</td><td>6.98</td><td>28.85</td><td>66.22</td><td>67.55</td><td>38.92</td></tr><tr><td>w/ min_grad</td><td>6.71</td><td>5.28</td><td>0.01</td><td>58.17</td><td>66.64</td><td>27.36</td></tr><tr><td>w/ random_grad</td><td>0.61</td><td>4.28</td><td>0.00</td><td>57.24</td><td>67.63</td><td>25.95</td></tr></table>

THE TESTING RESULTS ON 5 BENCHMARKS OF FINE-TUNED QWEN3-1.5B WITH DIFFERENT CONTINUAL LEARNING METHODS. THE METRIC FOR HUMANEVAL-CN USES PASS@1, AND OTHER FOUR BENCHMARKS ADOPT PERCENTAGE OF ACCURATE PREDICTIONS.

We further conduct an ablation study comparing minimum gradient attribution (CiMEO w/ min) and random sampling (CiMEO w/ random) strategies against the default maximum gradient attribution strategy. As shown in Table I, results on 4 benchmarks (HumanEvalCN, GaoKaoBench 2010, CMMLU, and C-Eval) indicate that CiMEO with maximum gradient attribution substantially outperforms both ablated variants. The above results support the core assumption of CiMEO: gradient magnitudes reflect neuron sensitivity to different cognitive stimuli and can therefore guide effective subnetwork selection.

These results suggest that CiMEO improves general adaptability and long-horizon stability in continual learning, rather than overfitting to a specific dataset or domain. Such balanced gains are especially important in continual learning, as they indicate better generalization and reduced catastrophic forgetting under sequential exposure to diverse skills. The observed cross-task improvements further imply enhanced parameter reuse and representation sharing, which are critical when tasks arrive incrementally.

Overall, the experiments highlight CiMEO’s potential to improve continual learning by (i) promoting more uniform performance across heterogeneous tasks, (ii) supporting knowledge accumulation and transfer across domains, and (iii) mitigating performance degradation caused by task interference.

## C. Stability-plasticity trade-off performance

Anti-forgetting and incremental learning are two key metrics for evaluating continual learning. In this study, we quantify them on specific benchmarks as follows.

$$
\begin{array} { c } { { \nabla _ { f } = \displaystyle \frac { 1 } { N } \sum _ { n = 1 } ^ { N } \mathbb { I } \left[ \left( \hat { y } _ { c h k p t } ^ { ( n ) } \neq \left( y ^ { ( n ) } \right) \& \hat { y } _ { b a s e } ^ { ( n ) } = y ^ { ( n ) } \right) \right] } } \\ { { \nabla _ { i } = \displaystyle \frac { 1 } { N } \sum _ { n = 1 } ^ { N } \mathbb { I } \left[ \left( \hat { y } _ { c h k p t } ^ { ( n ) } = y ^ { ( n ) } \right) \& \left( \hat { y } _ { b a s e } ^ { ( n ) } \neq y ^ { ( n ) } \right) \right] } } \end{array}\tag{6}
$$

where $\nabla _ { f }$ denotes the forgetting rate, and $\nabla _ { i }$ denotes the incremental learning rate. Here, $\hat { y } _ { c h k p t }$ represents the predictions of the fine-tuned LLMs, and $\hat { y } _ { b a s e }$ denotes the predictions of the base model.

Results are summarized in Fig.4 as bar charts. Across HumanEval-CN, CMMLU, and GSM8K (Fig.4a, Fig.4c and Fig.4d), CiMEO exhibits consistently low forgetting rates, and its average forgetting rate (Fig.4f) is close to the bestperforming methods. Meanwhile, CiMEO achieves strong incremental learning performance across benchmarks, ranking among the top methods overall. Notably, although AdaLOMO and AdaRankGrad show relatively low forgetting, their incremental gains are limited, indicating insufficient plasticity. In contrast, CiMEO provides a more favorable trade-off between stability (anti-forgetting) and plasticity (incremental learning).

We further observe that the ablated variants, CiMEO w/ min and CiMEO w/ random, can achieve very low forgetting on some benchmarks (e.g., GaokaoBench, GSM8K, and C-Eval), but they generally suffer from low incremental learning rates across tasks. This suggests that subnetworks selected by minimal gradient attribution or random sampling do not effectively capture cognition-relevant parameters for continual adaptation. Conversely, these findings support our core hypothesis that LLMs exhibit endogenous cognition isolation in their internal networks, and gradient magnitudes provide a practical signal for identifying and leveraging such isolated subnetworks.

![](images/2b67ac473e68617027cb0fbbc8a94e9f46496587b9e096ff2457fd3463a54313.jpg)

![](images/3957b2ac58d9bf4e3a47ddec68e570804dff44e6966bb4519bfef6bb04a87793.jpg)

![](images/a46c00d8ed814021ab41db4efbf0fd7523294178646d9bb3d47dc78b5a423fe9.jpg)

![](images/9051b61da16083adca76b924fffb627afa3b43807ef43ab6bfd307c693b296a6.jpg)

![](images/d437080ecbc070e2979d057f3df4566dddcc203d4283064828767eb99f7aab00.jpg)

![](images/1a721a67db743d8ff0001a925d7f179ac6767b5e0ac1210aa1b73fe17f538aac.jpg)  
Fig. 4. The forgetting and incremental rates on five benchmarks for all compared baselines. (a) for HumanEval-CN, (b) for GaoKaoBench-2010, (c) for CMMLU, (d) for GSM8K, (e) for CEval, and (f) is the average over all five benchmarks.

## D. Weight stability analysis

In addition to benchmark results, we analyze how different optimization methods affect representational transformations by examining the weight space through singular value decomposition (SVD). The notation is defined as follows,

• Continual learning checkpoints are indexed by $t \_ =$ $0 , 1 , \ldots , T _ { \mathrm { { \scriptsize ~ 2 ~ } } }$ , where t = 0 denotes the base model.

• We analyze all 2D weight matrices in each layer $1 \leq l \leq$ $L .$

• Taking the query projection in self-attention as an example, the weight matrix at layer l and checkpoint t is denoted by $\mathbf { \check { Q } } _ { l } ^ { ( t ) }$ . Its SVD is $\mathbf { Q } ^ { ( t ) _ { l } } = \mathbf { U } _ { q , l } ^ { ( t ) } \pmb { \Sigma } _ { q , l } ^ { ( t ) } \mathbf { V } _ { q , l } ^ { ( t ) } ,$ where $\Sigma _ { q , l } ^ { ( t ) } = \mathrm { d i a g } ( \sigma _ { 1 } \geq \sigma _ { 2 } \geq \cdot \cdot \cdot \geq 0 )$ . The top-k left singular vectors are $\mathbf { U } _ { q , l } ^ { ( t ) } [ : , 1 : k ] \in \mathbb { R } ^ { d _ { \mathrm { o u t } } \times k }$ , and top-k right singular vector is $\mathbf { \bar { V } } _ { q , l } ^ { ( t ) } \left[ 1 : k , : \right] \in \mathbb { R } ^ { k \times d _ { \mathrm { o u t } } }$

Subspace Overlap Analysis: For each linear projection matrix (e.g., the query projection) in each layer, the top-k left singular vectors span a k-dimensional subspace capturing the dominant directions of the transformation. We quantify the similarity between the base model $\mathbf { W } _ { 0 }$ and a fine-tuned checkpoint $\mathbf { W } _ { t }$ using the subspace overlap metric $\mathcal { O } ( \mathbf { W } _ { 0 } , \mathbf { W } _ { t } ; k )$ ,

$$
\mathcal { O } ( \mathbf { W } _ { 0 } , \mathbf { W } _ { t } ; k ) = \frac { 1 } { k } \| \mathbf { Q } _ { 0 } ^ { \top } \mathbf { Q } _ { t } \| _ { F } ^ { 2 }\tag{7}
$$

where $\mathbf { Q } _ { 0 }$ and $\mathbf { Q } _ { t }$ denote the top-k orthonormal singularvector matrices of $\mathbf { W } _ { 0 }$ and $\mathbf { W } _ { t }$ , respectively, and $\left\| \cdot \right\| _ { F }$ denotes the Frobenius norm. By definition, $\mathcal { O } ( \mathbf { W } _ { 0 } , \mathbf { W } _ { t } ; k ) \in [ 0 , 1 ]$ where 1 indicates identical subspaces and 0 indicates orthogonal subspaces.

We compute the average overlap across all layers and checkpoints, and summarize the results in Fig.5. A higher overlap indicates stronger subspace stability and more minor changes in the learned representations. Overall, CiMEO achieves high overlap values comparable to the most stable baselines, suggesting that it largely preserves the dominant weight subspaces during continual learning. This observation is consistent with the anti-forgetting results in Fig.4, where methods with higher subspace overlap (e.g., CiMEO) tend to exhibit stronger resistance to forgetting.

Principal-angle analysis of weight spaces: Principal angles characterize the rotation of the dominant subspaces between the base model and fine-tuned checkpoints. For each layer l at checkpoint t, we define the principal subspaces using the top-k left singular vectors $\mathbf { Q } _ { l } ^ { ( 0 ) } = \mathbf { \dot { U } } _ { l } ^ { ( 0 ) } [ : , \dot { 1 } : k ]$ and $\mathbf { \bar { Q } } _ { l } ^ { ( t ) } = \mathbf { \bar { U } } _ { l } ^ { ( t ) } [ : , 1 : k ]$ . We compute the cosine similarity ${ \pmb \sigma } _ { l } ^ { ( t ) } = \left( { \bf Q } _ { l } ^ { ( 0 ) } \right) ^ { T } \left( { \bf Q } _ { l } ^ { ( t ) } \right)$ , and define the i-th principal angle as $\theta _ { l , i } ^ { ( t ) } \ = \ \operatorname { a r c c o s } \left( \sigma _ { l , i } ^ { ( t ) } \right)$ , for $i = 1 , \cdots , k .$ . The mean and maximum principal angles for the l-th layer as follows,

![](images/8c23dd4717f82036c9ee1e19963341852c64cfe2894689ff357eaf0d0f5da2cd.jpg)  
Fig. 5. The weight stability analysis. (a), The maximum principle angles between Qwen3-1.5B and all compared methods fine-tuned checkpoints across all layers, and (d) average over layers. (b), The maximum principle angles between Qwen3-1.5B and all compared methods fine-tuned checkpoints across all layers, and (e) average over layers. (c), the subspace overlaps between Qwen3-1.5B and different methods fine-tuned checkpoints across all layers, and (f) average over layers.

$$
\hat { \theta } _ { l } ^ { ( t ) } = \frac { 1 } { k } \sum _ { i = 1 } ^ { k } \theta _ { l , i } ^ { ( t ) } \quad \hat { \theta } _ { l } ^ { ( t ) } = \operatorname* { m a x } _ { i } \{ \theta _ { l , 1 } ^ { ( t ) } , \cdot \cdot \cdot , \{ \theta _ { l , k } ^ { ( t ) } \}\tag{8}
$$

Fig.5a and 5b report the layer-wise principal angles and the corresponding mean/max summaries across layers. These results reveal which layers maintain stable transformations and which undergo substantial rotations during continual learning. CiMEO exhibits relatively small mean and maximum principal angles, indicating that it induces only minor subspace rotations. The above results suggest that CiMEO primarily builds on the original feature structure rather than substantially reorienting it, providing a mechanistic explanation for its strong anti-forgetting performance.

## V. CONCLUSION

In this work, we investigated functional subnetwork localization for mitigating catastrophic forgetting in continual learning of large language models. Gradient-based attribution revealed distinct parameter subsets associated with six cognitive categories derived from Bloom’s taxonomy. Building on this observation, we proposed CiMEO, which integrates cognition-guided routing, attribution-based parameter selection, and state-serialized AdamW optimization to enable localized adaptation while reducing optimizer-state requirements.

Experiments across 5 benchmarks demonstrated that CiMEO achieves the highest average performance among the compared methods, balancing knowledge acquisition and retention. Ablation studies validated the importance of subnetwork selection, while weight-space analyses indicated substantial preservation of pre-trained representations. These findings highlight functional subnetwork localization as a promising approach to efficient and stable continual learning, suggesting that cognition-guided parameter updates can mitigate interference without extensive modifications to pre-trained models.

## VI. ACKNOWLEDGMENTS

This work is jointly supported by grants from the Shenzhen Science and Technology Program (No.KCXFZ20230731093001002, No.KJZD20230923115113026), Shenzhen Basic Research General Program (No.JCYJ20250604145438051), Guangdong Basic and Applied Basic Research Foundation (No.2026A1515012168, No.2023A1515110561, No.2025A1515012263), Establishment Project of Shenzhen

Chinese Calligraphy Digital Simulation Technology Engineering Laboratory (No.XMHT20190108009), Shenzhen Key Laboratory of Intelligent Computing in Network Environment, the National Natural Science Foundation of China (No.62276075, No.62506099), the Postdoctoral Fellowship Program of CPSF (No.GZC20252738, No.GZC20252759).

## REFERENCES

[1] H. Naveed, A. U. Khan, S. Qiu, M. Saqib, S. Anwar, M. Usman, N. Akhtar, N. Barnes, and A. Mian, “A comprehensive overview of large language models,” ACM Transactions on Intelligent Systems and Technology, vol. 16, no. 5, pp. 1–72, 2025.

[2] H. Shi, Z. Xu, H. Wang, W. Qin, W. Wang, Y. Wang, Z. Wang, S. Ebrahimi, and H. Wang, “Continual learning of large language models: A comprehensive survey,” ACM Computing Surveys, 2024.

[3] T. Veniat, L. Denoyer, and M. Ranzato, “Efficient continual learning with modular networks and task-driven priors,” in 9th International Conference on Learning Representations, ICLR 2021, 2021.

[4] Z. Wang, E. Yang, L. Shen, and H. Huang, “A comprehensive survey of forgetting in deep learning beyond continual learning,” IEEE Transactions on Pattern Analysis and Machine Intelligence, 2024.

[5] M. McCloskey and N. J. Cohen, “Catastrophic interference in connectionist networks: The sequential learning problem,” Psychology of Learning and Motivation-Advances in Research and Theory, vol. 24, no. C, pp. 109–165, 1989.

[6] R. M. French, “Catastrophic forgetting in connectionist networks,” Trends in cognitive sciences, vol. 3, no. 4, pp. 128–135, 1999.

[7] S. Lai, Z. Zhao, F. Zhu, X. Lin, Q. Zhang, and G. Meng, “Pareto continual learning: Preference-conditioned learning and adaption for dynamic stability-plasticity trade-off,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 39, no. 17, 2025, pp. 18 008–18 016.

[8] K. Lv, Y. Yang, T. Liu, Q. Guo, and X. Qiu, “Full parameter fine-tuning for large language models with limited resources,” in Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 2024, pp. 8187–8198.

[9] S. Grossberg, “Adaptive resonance theory: How a brain learns to consciously attend, learn, and recognize a changing world,” Neural networks, vol. 37, pp. 1–47, 2013.

[10] J. Kirkpatrick, R. Pascanu, N. Rabinowitz, J. Veness, G. Desjardins, A. A. Rusu, K. Milan, J. Quan, T. Ramalho, A. Grabska-Barwinska et al., “Overcoming catastrophic forgetting in neural networks,” Proceedings of the National Academy of Sciences of the United States of America, 2017.

[11] C. Zhang, B. Wang, and D. Song, “On elastic language models,” ACM Transactions on Information Systems, vol. 42, no. 6, pp. 1–29, 2024.

[12] Y. Liu, C. Si, K. R. Narasimhan, and S. Yao, “Contextual experience replay for self-improvement of language agents,” in Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 2025, pp. 14 179–14 198.

[13] S. Li, T. Su, X.-Y. Zhang, and Z. Wang, “Continual learning with knowledge distillation: A survey,” IEEE Transactions on Neural Networks and Learning Systems, vol. 36, no. 6, pp. 9798–9818, 2024.

[14] L. Wang, S. Chen, L. Jiang, S. Pan, R. Cai, S. Yang, and F. Yang, “Parameter-efficient fine-tuning in large language models: a survey of methodologies,” Artificial Intelligence Review, vol. 58, no. 8, p. 227, 2025.

[15] P.-Y. Chen and S. Liu, “Memory-efficient fine-tuning for foundation models,” in Introduction to Foundation Models. Springer, 2025, pp. 77–86.

[16] Y. Tang, L. Qiao, L. Yin, P. Liang, A. Shen, Z. Yang, L. Zhang, and D. Li, “Training large-scale language models with limited gpu memory: a survey,” Frontiers of Information Technology & Electronic Engineering, pp. 1–23, 2025.

[17] Z. Wan, X. Wang, C. Liu, S. Alam, Y. Zheng, J. Liu, Z. Qu, S. Yan, Y. Zhu, Q. Zhang, M. Chowdhury, and M. Zhang, “Efficient large language models: A survey,” Transactions on Machine Learning Research, 2024, survey Certification. [Online]. Available: https://openreview.net/forum?id=bsCCJHbO8A

[18] A. Gupta, R. Celente, A. Shivanna, D. Braithwaite, G. Dexter, S. Tang, H. Udagawa, D. Silva, R. Ramanath, and S. S. Keerthi, “Effective quantization of muon optimizer states,” arXiv preprint arXiv:2509.23106, 2025.

[19] S. A. Josselyn and S. Tonegawa, “Memory engrams: Recalling the past and imagining the future,” Science, vol. 367, no. 6473, p. eaaw4325, 2020.

[20] R. Yuste, R. Cossart, and E. Yaksi, “Neuronal ensembles: Building blocks of neural circuits,” Neuron, vol. 112, no. 6, pp. 875–892, 2024.

[21] S. A. Josselyn and P. W. Frankland, “Memory allocation: mechanisms and function,” Annual review of neuroscience, vol. 41, no. 1, pp. 389– 413, 2018.

[22] A. Lavi, M. Sehgal, A. F. de Sousa, D. Ter-Mkrtchyan, F. Sisan, A. Luchetti, A. Okabe, C. Bear, and A. J. Silva, “Local memory allocation recruits memory ensembles across brain regions,” Neuron, vol. 111, no. 4, pp. 470–480, 2023.

[23] R. S. Larsen and P. J. Sjostr ¨ om, “Synapse-type-specific plasticity in local¨ circuits,” Current opinion in neurobiology, vol. 35, pp. 127–135, 2015.

[24] D. R. Krathwohl, “A revision of bloom’s taxonomy: An overview,” Theory into practice, vol. 41, no. 4, pp. 212–218, 2002.

[25] Z. Li and D. Hoiem, “Learning without forgetting,” IEEE transactions on pattern analysis and machine intelligence, vol. 40, no. 12, pp. 2935– 2947, 2017.

[26] F.-K. Sun, C.-H. Ho, and H.-Y. Lee, “{LAMAL}: {LA}nguage modeling is all you need for lifelong language learning,” in International Conference on Learning Representations, 2020. [Online]. Available: https://openreview.net/forum?id=Skgxcn4YDS

[27] X. Wang, T. Chen, Q. Ge, H. Xia, R. Bao, R. Zheng, Q. Zhang, T. Gui, and X.-J. Huang, “Orthogonal subspace learning for language model continual learning,” in Findings of the Association for Computational Linguistics: EMNLP 2023, 2023, pp. 10 658–10 671.

[28] Y.-S. Liang and W.-J. Li, “Inflora: Interference-free low-rank adaptation for continual learning,” in 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2024, pp. 23 638– 23 647.

[29] Y. Lu, B. Qian, C. Yuan, H. Jiang, and X. Wang, “Controlled lowrank adaptation with subspace regularization for continued training on large language models,” in Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 2025, pp. 19 165–19 181.

[30] A. Mallya and S. Lazebnik, “Packnet: Adding multiple tasks to a single network by iterative pruning,” in 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition. IEEE, 2018, pp. 7765–7773.

[31] A. Mallya, D. Davis, and S. Lazebnik, “Piggyback: Adapting a single network to multiple tasks by learning to mask weights,” in European Conference on Computer Vision. Springer, 2018, pp. 72–88.

[32] D. Guo, A. M. Rush, and Y. Kim, “Parameter-efficient transfer learning with diff pruning,” in Proceedings of the 59th annual meeting of the associationfor computational linguistics and the 11th international joint conference on natural language processing (volume 1: Long papers), 2021, pp. 4884–4896.

[33] Z. Wang, Z. Zhang, C.-Y. Lee, H. Zhang, R. Sun, X. Ren, G. Su, V. Perot, J. Dy, and T. Pfister, “Learning to prompt for continual learning,” in 2022 IEEE/CVF conference on computer vision and pattern recognition (CVPR). IEEE, 2022, pp. 139–149.

[34] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen, “LoRA: Low-rank adaptation of large language models,” in International Conference on Learning Representations, 2022. [Online]. Available: https://openreview.net/forum?id=nZeVKeeFYf9

[35] K. Bhardwaj, N. P. Pandey, S. Priyadarshi, V. Ganapathy, S. Kadambi, R. Esteves, S. Borse, P. Whatmough, R. Garrepalli, M. V. Baalen, H. Teague, and M. Nagel, “Sparse high rank adapters,” in The Thirtyeighth Annual Conference on Neural Information Processing Systems, 2024. [Online]. Available: https://openreview.net/forum?id=6hY60tkiEK

[36] J. Chen, S. He, X. Yang, H. Liao, Y. Wang, X. Liao, W. Teng, K. Liu, and J. Zhao, “Harmonizing the past, present, and future: A null-space constrained region-specific method for continual learning in LLMs,” in Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 2026, pp. 27 013– 27 033. [Online]. Available: https://aclanthology.org/2026.acl-long.1244/

[37] D. Dai, L. Dong, Y. Hao, Z. Sui, B. Chang, and F. Wei, “Knowledge neurons in pretrained transformers,” in Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 2022, pp. 8493–8502. [Online]. Available: https://aclanthology.org/2022.acl-long.581/

[38] X. Wang, K. Wen, Z. Zhang, L. Hou, Z. Liu, and J. Li, “Finding skill neurons in pre-trained transformer-based language models,” in Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing (EMNLP), 2022, pp. 11 132–11 152. [Online]. Available: https://aclanthology.org/2022.emnlp-main.765/

[39] Y. Lee, A. S. Chen, F. Tajwar, A. Kumar, H. Yao, P. Liang, and C. Finn, “Surgical fine-tuning improves adaptation to distribution shifts,” in International Conference on Learning Representations (ICLR), 2023. [Online]. Available: https://mlanthology.org/iclr/2023/ lee2023iclr-surgical/

[40] Y. Leng and D. Xiong, “Towards understanding multi-task learning (generalization) of LLMs via detecting and exploring task-specific neurons,” in Proceedings of the 31st International Conference on Computational Linguistics (COLING), 2025, pp. 2969–2987. [Online]. Available: https://aclanthology.org/2025.coling-main.200/

[41] K. Lv, H. Yan, Q. Guo, H. Lv, and X. Qiu, “Adalomo: Low-memory optimization with adaptive learning rate,” in Findings of the Association for Computational Linguistics ACL 2024, 2024, pp. 12 486–12 502.

[42] J. Zhao, Z. Zhang, B. Chen, Z. Wang, A. Anandkumar, and Y. Tian, “Galore: Memory-efficient llm training by gradient low-rank projection,” in International Conference on Machine Learning. PMLR, 2024, pp. 61 121–61 143.

[43] Y. Wang and B. Yin, “GCL: Group-shared continual learning finetuning for sparse LLMs,” Neurocomputing, vol. 675, p. 132918, 2026. [Online]. Available: https://doi.org/10.1016/j.neucom.2026.132918

[44] H. Zhao, H. Wang, Y. Peng, S. Zhao, X. Tian, S. Chen, Y. Ji, and X. Li, “1.4 million open-source distilled reasoning dataset to empower large language model training,” 2025. [Online]. Available: https://arxiv.org/abs/2503.19633

[45] M. Jin, Q. Yu, J. Huang, Q. Zeng, Z. Wang, W. Hua, H. Zhao, K. Mei, Y. Meng, K. Ding et al., “Exploring concept depth: How large language models acquire knowledge and concept at different layers?” in Proceedings of the 31st International Conference on Computational Linguistics, 2025, pp. 558–573.

[46] Z. Zhao, Y. Ziser, and S. B. Cohen, “Layer by layer: Uncovering where multi-task learning happens in instruction-tuned large language models,” in Proceedings ofthe 2024 Conference on Empirical Methods in Natural Language Processing, 2024, pp. 15 195–15 214.

[47] Z. Liu, C. Kong, Y. Liu, and M. Sun, “Fantastic semantics and where to find them: Investigating which layers of generative llms reflect lexical semantics,” in Findings of the Association for Computational Linguistics ACL 2024, 2024, pp. 14 551–14 558.

[48] I. Loshchilov and F. Hutter, “Decoupled weight decay regularization,” in 7th International Conference on Learning Representations, ICLR 2019, New Orleans, LA, USA, May 6-9, 2019. OpenReview.net, 2019. [Online]. Available: https://openreview.net/forum?id=Bkg6RiCqY7

[49] J. Kang and Q. Yin, “Balancing lora performance and efficiency with simple shard sharing,” arXiv preprint arXiv:2409.15371, 2024.

[50] J. Kang, “Bone: Block affine transformation as parameter efficient finetuning methods for large language models,” arXiv e-prints, pp. arXiv– 2409, 2024.

[51] S.-Y. Liu, C.-Y. Wang, H. Yin, P. Molchanov, Y.-C. F. Wang, K.-T. Cheng, and M.-H. Chen, “Dora: Weight-decomposed low-rank adaptation,” in International Conference on Machine Learning. PMLR, 2024, pp. 32 100–32 121.

[52] F. Paischer, L. Hauzenberger, T. Schmied, B. Alkin, M. P. Deisenroth, and S. Hochreiter, “One initialization to rule them all: Fine-tuning via explained variance adaptation,” in Adaptive Foundation Models: Evolving AI for Personalized and Efficient Learning, 2024.

[53] L. Zhang, L. Zhang, S. Shi, X. Chu, and B. Li, “Lora-fa: Memoryefficient low-rank adaptation for large language models fine-tuning,” arXiv preprint arXiv:2308.03303, 2023.

[54] K. Buy¨ ukaky ¨ uz, “Olora: Orthonormal low-rank adaptation of large¨ language models,” arXiv preprint arXiv:2406.01775, 2024.

[55] F. Meng, Z. Wang, and M. Zhang, “Pissa: Principal singular values and singular vectors adaptation of large language models,” Advances in Neural Information Processing Systems, vol. 37, pp. 121 038–121 072, 2024.

[56] B. Liao and C. Monz, “3-in-1: 2d rotary adaptation for efficient finetuning, efficient batching and composability,” in The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024. [Online]. Available: https://openreview.net/forum?id=rYjYwuM6yH

[57] Y. Refael, J. Svirsky, B. Shustin, W. Huleihel, and O. Lindenbaum, “Adarankgrad: Adaptive gradient rank and moments for memoryefficient llms training and fine-tuning,” in International Conference on Learning Representations, vol. 2025, 2025, pp. 102 591–102 612.

[58] S. Jiang, Q. Chen, Y. Pan, Y. Xiang, Y. Lin, X. Wu, C. Liu, and X. Song, “Zo-adamu optimizer: Adapting perturbation by the momentum and uncertainty in zeroth-order optimization,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 38, no. 16, 2024, pp. 18 363– 18 371.

[59] M. Chen, J. Tworek, H. Jun, Q. Yuan, H. P. D. O. Pinto, J. Kaplan, H. Edwards, Y. Burda, N. Joseph, G. Brockman et al., “Evaluating large language models trained on code,” arXiv preprint arXiv:2107.03374, 2021.

[60] X. Zhang, C. Li, Y. Zong, Z. Ying, L. He, and X. Qiu, “Evaluating the performance of large language models on gaokao benchmark,” arXiv preprint arXiv:2305.12474, 2023.

[61] H. Li, Y. Zhang, F. Koto, Y. Yang, H. Zhao, Y. Gong, N. Duan, and T. Baldwin, “Cmmlu: Measuring massive multitask language understanding in chinese,” in Findings of the Association for Computational Linguistics ACL 2024, 2024, pp. 11 260–11 285.

[62] Y. Huang, Y. Bai, Z. Zhu, J. Zhang, J. Zhang, T. Su, J. Liu, C. Lv, Y. Zhang, J. Lei, Y. Fu, M. Sun, and J. He, “C-eval: A multi-level multidiscipline chinese evaluation suite for foundation models,” in Advances in Neural Information Processing Systems, 2023.

[63] K. Cobbe, V. Kosaraju, M. Bavarian, M. Chen, H. Jun, L. Kaiser, M. Plappert, J. Tworek, J. Hilton, R. Nakano et al., “Training verifiers to solve math word problems,” arXiv preprint arXiv:2110.14168, 2021.