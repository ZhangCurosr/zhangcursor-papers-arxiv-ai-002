# RA-MOWE: WORKFLOW-AFFINITY EMBEDDINGS FOR QUERY CLUSTERING AND AGENTIC WORKFLOW GENERATION

Qi Cheng Rutgers University New Brunswick, NJ q.cheng@rutgers.edu

Yiqun Xie University of Maryland College Park, MD xie@umd.edu

Xiaowei Jia Rutgers University New Brunswick, NJ xj159@cs.rutgers.edu

Shengyu Chen University of Pittsburgh Pittsburgh, PA shc160@pitt.edu

Haoyu Wang   
NEC Labs America   
Princeton, NJ   
haoyu@nec-labs.com   
Wei Cheng   
NEC Labs America   
Princeton, NJ   
weicheng@nec-labs.com   
Haifeng Chen   
NEC Labs America   
Princeton, NJ   
haifeng@nec-labs.com

## ABSTRACT

Agentic workflows enable large language models (LLMs) to solve complex tasks by coordinating reasoning, tool use, and verification. However, a workflow optimized for an entire task collection can overlook differences in the reasoning strategies that individual queries need, while searching for a new workflow for every query repeats costly optimization. To address this tradeoff, we introduce RA-MoWE, a framework that uses workflow-affinity embeddings to cluster queries and guide the generation of reusable expert workflows. Each embedding records how well a fixed set of reference workflows solves a query, revealing similarities in which reasoning strategies are effective. RA-MoWE uses each cluster’s queries and average embedding to initialize and refine a specialized workflow through execution feedback. An embedding encoder predicts these embeddings from query text, allowing new queries to select a generated expert without first executing the reference workflows. On a 300-query test set drawn from four benchmarks spanning mathematics, science, and programming, RA-MoWE improves average task score by 4.04 percentage points over selecting among the reference workflows, while using 27.7% fewer language-model calls at inference.

## 1 INTRODUCTION

Agentic workflows organize large language model (LLM) calls into procedures for reasoning, revising answers, and using tools, extending their capabilities on complex tasks (Yao et al., 2023b; Madaan et al., 2023). Their design determines which operations run and how they interact, affecting both solution quality and inference cost: repeated sampling, revision, and decomposition offer different ways to spend computation. Manually choosing a suitable design for a diverse collection of queries is therefore a substantial engineering effort (Hu et al., 2025; Zhang et al., 2025b).

Recent methods automate workflow design through search and learned generation. ADAS and AFlow explore code-represented agents using execution feedback, while AgentSquare searches combinations of reusable modules (Hu et al., 2025; Zhang et al., 2025b; Shang et al., 2025). In these works, the workflow is often optimized over a collection of tasks using a collection-level score and thus can overlook differences in individual query need. In contrast, some approaches, such as FlowReasoner (Gao et al., 2025), learn to generate a workflow for each query. These methods need to repeat costly optimization and also have less training samples for each workflow. Addressing such tradeoff requires identifying which queries should share a workflow and what computations each group needs. While a natural solution is to group queries based on their similarity. However, conventional semantic similarity does not necessarily capture whether queries benefit from the same reasoning strategies.

(a) Affinity profiles of query clusters Cluster (total queries)  
![](images/20219bffdc6de7c6a73a43c83fc2d49603fd6de0f4063c4a6a4f18757a1aa947.jpg)  
Figure 1: Different workflow preferences motivate specialized experts. (a) Average embeddings for six GPT-4o-mini training-query clusters show distinct responses to reference workflows. C2 favors sampling and answer selection (CoT-SC), whereas C4 favors self-refinement. (b) RA-MoWE uses these preferences to choose initial workflows. Values are standardized scores centered within each row; outlines mark row maxima, and parentheses give full cluster sizes. Refine revises a chain-of-thought (CoT) answer; Ens.(n) selects among n sampled CoT answers; Plan adds decomposition. Asterisks mark runnable substitutes for unavailable workflows. Measurement details are in Appendix C.4.

(b) Generated workflows
<table><tr><td>Seed</td><td>Stored expert</td></tr><tr><td>Refine</td><td>Refine</td></tr><tr><td>CoT</td><td>CoT</td></tr><tr><td>Ens.(3)</td><td>Ens.(3)</td></tr><tr><td>Ens.(3)*</td><td>Plan + Ens.(4)</td></tr><tr><td>Refine</td><td>Refine</td></tr><tr><td>Ens.(3)*</td><td>Ens.(3)</td></tr></table>

Ens.(n): n CoT samples with answer selection. \* Runnable fallback.

Our key idea is to introduce a workflow-affinity embedding and group queries by how they respond to different reasoning workflows. In particular, the workflow-affinity embedding is a vector whose coordinates estimate a query’s task score under a fixed set of reference workflows, called probes. The resulting vector captures a query’s relative strengths and weaknesses across different reasoning strategies, including which reasoning strategies succeed, partially succeed, or fail. Queries with similar profiles can therefore be grouped together based on their computational needs rather than their semantic similarity.

We integrate this idea into RA-MoWE, a framework that uses the workflow-affinity embeddings to guide the generation of reusable expert workflows. Offline, we execute the reference workflows on training queries, compute their embeddings, and partition them into clusters. Clustering these embeddings groups queries with similar measured responses, giving expert construction both a focused set of examples and evidence about potentially useful reasoning strategies. For each cluster, we introduce an embedding-guided search mechanism, which uses the cluster prototype to choose an executable starting workflow and constrain the candidate operations. It then proposes refinements informed by execution feedback and probe outcomes on remaining failures, retaining edits that im prove the cluster’s validation score. The resulting expert is stored for reuse. Figure 1 connects the measured cluster affinities to the workflows produced by this process.

We then create an embedding encoder, which makes the generated experts accessible from query text alone. While workflow-affinity embeddings can be measured offline by executing the reference workflows, this process is too costly to repeat for every new query. We therefore train the encoder to predict a query’s workflow-affinity embedding directly from its text, using the measured embeddings as supervision. At deployment, the predicted embedding selects a cluster’s expert without executing the reference workflows.

We evaluate RA-MoWE across mathematics, science, programming, and repository repair against six families of adapted search and routing methods. On 300 test queries drawn from four bench marks in mathematics, science, and programming, RA-MoWE improves average task score by 4.04 percentage points over a system that selects from the reference workflows, while using 27.7% fewer LLM calls at inference. Both systems choose a workflow based on the query text. Further experiments examine whether queries consistently benefit from particular workflows, how clustering affects the quality of generated workflows, and how accurately the encoder assigns new queries to experts.

## Our contributions are:

1. Workflow-affinity embeddings that represent queries through measured workflow outcomes and organize training examples for specialist construction.

2. An embedding encoder that predicts this representation from text, enabling expert selection without online reference-workflow execution.

3. Embedding-guided expert generation that uses cluster affinities, query examples, and failure feedback to initialize and refine reusable workflows.

Our analysis identifies when measurement and prediction errors preserve useful decisions and when the available workflow operations support transfer from a strong probe to a generated expert.

## 2 RA-MOWE: EMBEDDING-GUIDED WORKFLOW GENERATION

Given training queries with reference answers or executable tests, our goal is to construct reusable workflows that serve different computational needs. RA-MoWE uses a shared representation to connect three decisions: which queries should be grouped together, how to build an expert for each group, and which expert should answer a new query. We first measure each query’s response to reference workflows and cluster the resulting embeddings. These embeddings guide expert generation and supervise an encoder that selects an expert from query text. Figure 2 illustrates the framework; Algorithm 1 summarizes its construction and deployment.

OFFLINE: scored executions and training labels  
![](images/ccce3906ea8948af9d9e9e127a2959ed71f36db592dea6dc3d098185f6bd41fb.jpg)  
Figure 2: The RA-MoWE framework. Offline, reference-workflow scores define query embeddings for clustering, expert generation, and encoder supervision. At deployment, the encoder predicts an embedding from query text and selects a stored expert. Dashed arrows indicate saved supervision and experts. C1–C3 mark the embedding, encoder, and generation contributions.

## 2.1 WORKFLOW-AFFINITY EMBEDDINGS FOR QUERY CLUSTERING

A useful query group should reflect which reasoning procedures help its members. Questions about the same topic can require different computations, such as comparing several candidate answers or revising an initial solution. We therefore represent a query by its performance under a fixed panel of diverse reference workflows, called probes. These probes apply established strategies such as chain-of-thought, self-consistency, and self-refinement (Wei et al., 2022; Wang et al., 2023; Madaan et al., 2023).

For a query $q$ and probe $p _ { k }$ , let $Y _ { k , s } ( q ) \in [ 0 , 1 ]$ be the task score from execution s under a fixed execution and scoring protocol. Averaging S repetitions gives

$$
\bar { f } _ { k } ( q ) = \frac { 1 } { S } \sum _ { s = 1 } ^ { S } Y _ { k , s } ( q ) , \qquad \bar { f } ( q ) = \big ( \bar { f } _ { 1 } ( q ) , \dots , \bar { f } _ { K } ( q ) \big ) ^ { \top } ,\tag{1}
$$

where K is the number of probes. The vector $\bar { f } ( q )$ retains the strengths and weaknesses of all reference workflows, including partial successes and multiple successful strategies. We standardize its coordinates using training statistics to obtain the workflow-affinity embedding $\bar { z } ( q )$ . This prevents differences in score variance from dominating the comparison between queries; the normalization and measurement settings are given in Appendix B.

We cluster the training queries in the embedding space using k-means, yielding J groups with centers $C = ( c _ { 1 } , \dots , c _ { J } )$ . Each group supplies examples for a reusable expert. Grouping queries by their workflow responses therefore provides workflow construction with both relevant examples and evidence about which reasoning strategies are effective for those examples. Such evidence also has a precise interpretation within the probe panel: if two queries’ expected probe scores differ by at most ϵ in every coordinate, transferring the best probe from one to the other loses at most 2ϵ in expected score. Proposition 1 and its proof relate this decision guarantee to measurement accuracy. The next step uses the resulting groups to construct workflows beyond the reference panel.

## 2.2 EMBEDDING-GUIDED EXPERT GENERATION

Each cluster supplies two complementary inputs for expert generation: (1) examples to optimize and (2) evidence about useful reasoning operations. For cluster j, let $V _ { j }$ be the subset of its training queries used to evaluate candidate workflows. We summarize their measured embeddings by

$$
\rho _ { j } = \frac { 1 } { | V _ { j } | } \sum _ { q \in V _ { j } } \bar { z } ( q ) ,\tag{2}
$$

where $| V _ { j } |$ is the number of queries in this subset. The prototype vector $\rho _ { j }$ describes the cluster’s relative strengths across the reference workflows. It provides search with a starting strategy and evidence for choosing subsequent operations.

Search begins by mapping the highest-ranked runnable probe in $\rho _ { j }$ to an initial workflow. We construct workflows from executable operations such as planning, sampling, answer selection, and revision, following the modular-search principle of Zhang et al. (2025b); Shang et al. (2025). The same embedding restricts candidate operations using the lowest-ranked probes, while retaining the operation chosen for initialization. In this way, the measured response guides both where search starts and which changes it considers. The value of the initialization depends on how well the executable workflow preserves the probe’s useful behavior; Proposition 5 characterizes this connection.

Refinement focuses on the queries the current workflow has not solved. An LLM optimizer receives failure examples, summaries of probe scores on failed and solved queries, and previous search attempts. These summaries identify strategies that may address the remaining failures. The optimizer proposes an edit, which is executed on $V _ { j }$ and retained only if its average score improves by a prescribed margin. Repeating this process produces expert $w _ { j } ;$ ; the collection ${ \mathcal W } = \{ w _ { 1 } , \ldots , w _ { J } \}$ forms the reusable expert bank. The operator mapping, proposal rules, and acceptance margin are specified in Appendix B. Search uses construction data, while the separate test pool evaluates the resulting bank.

Algorithm 1 presents the shared-clustering framework; the recorded alignment between construction and deployment clusters is detailed in Appendix C.4.

Algorithm 1 RA-MoWE: constructing and using a bank of workflow experts   
Require: Training queries and task scorers; reference workflows; cluster count $J ;$ search resources   
1: Measure and standardize the workflow-affinity embedding $\bar { z } ( q )$ of each training query   
2: Train the embedding encoder to predict $\bar { z } ( q )$ from query text   
3: Cluster the measured embeddings into $J$ groups with centers $C$   
4: for each cluster $j$ do   
5: Choose construction queries $V _ { j }$ and compute their average embedding $\rho _ { j }$   
6: Initialize $w _ { j }$ from a high-affinity runnable probe; restrict candidate operations using $\rho _ { j }$   
7: for each allowed search round do   
8: Use probe scores and remaining failures to guide an LLM-proposed edit   
9: Evaluate on $V _ { j } ;$ accept only if the score improvement exceeds the required margin   
10: end for   
11: Store the final expert $w _ { j }$   
12: end for   
13: return Trained encoder, cluster centers $C ,$ and expert bank W   
Deployment: predict a new query’s embedding, select the nearest cluster, and run its expert.

## 2.3 LEARNING THE EMBEDDING ENCODER

The embeddings used for construction require scored probe executions. To select an expert for a new query without repeating that measurement, we train an encoder to predict the same coordinates from text. Let $\psi ( q )$ be frozen text features and $g _ { \theta }$ a trainable prediction head with parameters θ. The predicted embedding is

$$
\hat { z } ( q ) = g _ { \theta } ( \psi ( q ) ) \in \mathbb { R } ^ { K } .\tag{3}
$$

Training pairs each query with its measured embedding $\bar { z } ( q )$ . We use frozen BGE features (Xiao et al., 2024) as inputs to a small prediction head, which is trained to predict the measured embedding from query text. The loss encourages agreement in the relative pattern of probe responses and penalizes errors in the coordinates used for clustering. Predicting the full embedding also allows the same learned target to support different cluster partitions. The loss, architecture, and training protocol are specified in Appendix B.

At deployment, the predicted embedding selects the closest cluster center:

$$
a ( q ) = \underset { 1 \leq j \leq J } { \arg \operatorname* { m i n } } \| \hat { z } ( q ) - c _ { j } \| _ { 2 } ^ { 2 } .\tag{4}
$$

Here $c _ { j }$ is the center of cluster $j , \parallel \cdot \parallel _ { 2 }$ is the Euclidean norm, and $a ( q )$ is the selected expert’s index. We then execute the stored workflow $w _ { a ( q ) }$ on $q .$ Both probe measurement and workflow construction are therefore performed offline and reused across future queries.

The prediction need not recover every coordinate exactly to preserve an expert choice. If the prediction error is smaller than the measured embedding’s distance to the nearest cluster boundary, the two embeddings select the same expert. Proposition 3 and Appendix A.3 formalize this condition and its counterpart for expected probe scores. This motivates evaluating the encoder through both prediction error and the resulting expert assignments. Appendix A.6 combines the conditions for measurement, generation, and assignment into an analysis of the complete pipeline.

## 3 EMPIRICAL EVALUATION

We test the three links in RA-MoWE: whether workflow responses reveal reusable specialization, whether the encoder preserves useful assignments from text, and whether affinity profiles guide the construction of effective workflows. Each experiment connects an intermediate design choice to downstream utility.

Tasks and evaluation. MixBench-H contains 600 training and 300 test queries spanning AIME, GPQA, CodeContests, and LiveCodeBench (Mathematical Association of America, 2026; Rein et al., 2024; Li et al., 2022; Jain et al., 2025). We evaluate GPT-4o-mini and Claude Haiku 4.5 separately. SWE-bench Verified (Jimenez et al., 2024) uses a repository-stratified 300/200 split and a fixed Agentless-lite localize–repair pipeline (Xia et al., 2025). Mixed-task utility is mean answer correctness or public-test pass fraction; SWE utility is harness resolution. Comparisons within each regime share queries and scoring contracts. The main comparison tests the embedding-to bank interface using cluster-conditioned AFlow search on mixed tasks and cluster policy selection on SWE. A separate experiment evaluates embedding-guided expert generation. We report paired stratified bootstrap intervals and Holm-adjusted sign-flip tests for the historical routing families; the embedding-guided search contrasts use exploratory, unadjusted query-bootstrap intervals.

Table 1: Baseline comparison on three regimes (higher utility is better). Scores are comparable within columns. $A ^ { \dagger }$ uses measured test affinities; B predicts them from text. † indicates scorer feedback on the evaluated query, and ‡ public-test execution feedback. FlowReasoner has † only on mixed tasks. Search baselines are the documented local adaptations, not published full-system reproductions. Dashes denote unrun configurations. Bold identifies significant gains over D under measured assignment.
<table><tr><td></td><td colspan="2">MixBench-H</td><td>SWE Verified GPT-4o-mini</td></tr><tr><td>Method</td><td>GPT-4o-mini</td><td>Haiku 4.5</td><td></td></tr><tr><td>Direct</td><td>.2872</td><td>.5694</td><td>.2000</td></tr><tr><td>Train-selected probe</td><td>.3481‡ .2722</td><td>.6894 .5381</td><td>.1750 .1750</td></tr><tr><td>Global comparator D AFlow search</td><td>.2722</td><td>.5381</td><td>.1900</td></tr><tr><td>ADAS (executor-class meta-model)</td><td>.2042</td><td>.2489</td><td>.2000</td></tr><tr><td>ScoreFlow (SF-0/ICPO)</td><td>.2756</td><td></td><td>.2100</td></tr><tr><td>FlowReasoner (search only)</td><td>.3565†</td><td>.7465†</td><td>.2050</td></tr><tr><td>MasRouter (policy routing)</td><td></td><td></td><td>.1700</td></tr><tr><td>FrugalGPT (policy cascade)</td><td></td><td></td><td>.2050</td></tr><tr><td>Semantic BGE clusters</td><td>.2939</td><td>.5303</td><td>.1800</td></tr><tr><td>Semantic Qwen3 clusters</td><td>.3050</td><td>.5861</td><td></td></tr><tr><td>Random clusters</td><td>.2767</td><td>.5228</td><td>.1800</td></tr><tr><td>RA-MoWE: text-predicted B</td><td>.2922</td><td>.5214</td><td>.2000</td></tr><tr><td>RA-MoWE: measured  $A ^ { \dagger }$ </td><td>.2742</td><td>.6097</td><td>.2450</td></tr></table>

Comparisons test grouping and construction choices. Table 1 compares direct execution, a training-selected probe, and adaptations of AFlow (Zhang et al., 2025b), ADAS (Hu et al., 2025), ScoreFlow (Wang et al., 2025), and FlowReasoner (Gao et al., 2025). Semantic BGE/Qwen3 (Xiao et al., 2024; Zhang et al., 2025c) and random partitions test the choice of grouping signal; SWE also includes policy-level MasRouter (Yue et al., 2025) and FrugalGPT (Chen et al., 2024). The global comparator D searches the undivided mixed-task pool with the same search rule and five optimization rounds after initialization. Both settings allocate 600 validation queries per round in total, although the six-cluster bank requires more proposal calls. SWE D is the training-selected policy. Appendix C.7 details the adaptations and resource accounting; Appendix C.11 reports heterogeneous-model routing separately.

## 3.1 WORKFLOW AFFINITY CAPTURES REUSABLE SPECIALIZATION

Probe preferences persist across executions. The first test asks whether affinity captures a repeatable response signal. We select a probe using two executions per test query and evaluate it on the third, averaging all three folds. GPT-4o-mini reaches 0.4059 against 0.3481 for the trainingselected fixed probe: +0.0579 (95% CI [0.0283, 0.0881]). Uniform treatment of tied probes retains a 0.0382 gain ([0.0100, 0.0668]). The improvement survives an execution not used for selection, supporting query-specific workflow preference within the panel. Haiku reaches 0.7237 against 0.6894, but uniform ties reduce its gain to 0.0015; repeatability therefore depends on the backbone and tie rule. Table 7 gives both rules. Increasing the panel from eight to twelve probes also raises distinct training signatures from 281 to 329 and effective rank from 2.67 to 3.08, giving clustering a richer description of query behavior.

Affinity-defined specialists improve the global comparator. With measured test-affinity assignment A, Haiku utility rises from D’s 0.5381 to 0.6097: +0.0717 ([0.0339, 0.1122], Holm $p \ = \ 0 . 0 0 6 3 )$ On SWE, the bank resolves 49/200 instances against 35/200 for D: +0.0700 $( [ 0 . 0 3 0 0 , 0 . 1 1 0 0 ] , p = 0 . 0 0 5 2 )$ , correcting 16 failures while introducing two. Measured-affinity assignment also exceeds semantic BGE and random partitions by 0.0650 on SWE (Holm p = 0.0164/0.0123). Its 24.5% resolution rate is the highest point estimate among the tested SWE adaptations, including ScoreFlow at 21.0% and FlowReasoner/FrugalGPT at 20.5%. These results show that workflow-response geometry provides an actionable basis for specialist assignment. Because A uses test outcomes, the next experiment measures how much of this benefit transfers to text-only decisions.

The benefit varies across regimes: GPT-4o-mini’s A − D gap is 0.0019, with no historical routing contrast surviving Holm correction. Outcome-assisted FlowReasoner leads the mixed-task columns, and Haiku’s fixed self-consistency probe reaches 0.6894. Thus the evidence supports useful specialization relative to global construction in two regimes, while retaining strong alternatives in the comparison.

## 3.2 TEXT PREDICTION CONNECTS THE EMBEDDING TO DEPLOYMENT

The encoder must preserve decisions that matter to the expert bank. On the expanded panel, the frozen-BGE MLP obtains held-out cosine similarity 0.398 and coordinate-averaged MSE 0.870 with nested training on a 510/90 split. It replaces probe executions with one text forward pass. Downstream, B reaches 0.2922/0.5214/0.2000 in the three regimes of Table 1; none of its prespecified contrasts against D is significant.

Assignment diagnostics explain why reconstruction metrics alone are insufficient. Measured/predicted cluster agreement is 37.7% for GPT-4o-mini and 12.0% for Haiku, whose predictor assigns 277/300 queries to one cluster. The SWE predictor chooses direct for all 200 instances. The measured-affinity gains therefore establish a useful target, but the current predictors do not consistently preserve its specialist decisions. This connects the empirical challenge to Proposition 3, which assesses prediction error relative to cluster boundaries. Assignment fidelity and expert utility are therefore essential complements to coordinate error.

## 3.3 EMBEDDING-GUIDED SEARCH GENERATES EFFECTIVE WORKFLOWS

(a) Utility and serving cost  
![](images/b9d13970eb1ef9258fdf2095a137cf80156a1e85cbad6d04f1320a3958bbcd05.jpg)

(b) Guided B − control  
![](images/6a2e3e3b4330a835c2375f5117cd6eaddf88f2260bc98a6d9f92beda840efacb.jpg)  
300 test queries · 3 executions per arm · paired 95% bootstrap intervals

Figure 3: Embedding-guided expert generation on 300 MixBench-H queries with three executions per arm. Left: utility versus serving calls/query. Right: Guided-B utility differences with exploratory 95% paired query-bootstrap intervals, resampling per-query execution means. Guided-B improves on Direct and routed probes; its difference from single-call CoT is unresolved. Serving calls exclude offline measurement and construction. Prompting, decoding, and operator differences make the contrasts comparisons of complete configurations.

We next evaluate embedding-guided search, which uses cluster preferences to initialize and edit executable workflows. This study uses GPT-4o-mini through the official OpenAI endpoint, separately from the preceding NEC-endpoint experiments. Each of six cluster searches evaluates an affinityderived seed and at most two edits. The resulting bank includes refinement, sampling/selection, and decomposition. Five experts retain their seeds; cluster C3 accepts a larger ensemble followed by decomposition. The search traces show that behavioral initialization supplies most of the final bank, with local edits refining one specialist.

Generated workflows improve on Direct and routed probes. Figure 3 compares the generated bank with fixed and routed workflows. On an independent 300-query test pool, Guided-B scores 0.3358 against Direct’s 0.2763: +0.0595 ([0.0232, 0.0975], exploratory p = 0.001). It also improves over probe-menu-B’s 0.2955 by 0.0404 ([0.0042, 0.0773], p = 0.030), while reducing calls/query from 2.924 to 2.113 (27.7%). These comparisons show that the generated bank provides effective, reusable computation in a text-only system. Prompting, decoding, and execution differ across these arms, so the improvement is attributed to the complete configuration.

The same-executor single-call CoT control reaches 0.3295; its 0.0063 gap from Guided-B remains unresolved ([−0.0158, 0.0289]). A separate six-expert evaluation gives an observed per-query oracle of 0.4961 versus 0.3567 for the test-selected best fixed expert, while current routing realizes approximately 0.34. The remaining opportunity is concrete: the bank contains complementary observed successes, and better text prediction is needed to recover them reliably. Appendix C reports expert semantics, costs, panel and operator ablations, label-free features, and transfer across regimes.

## 4 RELATED WORK

Learning representations for computational decisions. Instance-dependent algorithm selection connects instance features to algorithm performance (Rice, 1976); decision-relevant representation learning seeks to preserve distinctions that affect action quality (Zhang et al., 2021). Workflowaffinity embeddings specialize this perspective to query-level outcomes of executable reasoning procedures. They provide an external behavioral representation, whereas function vectors and task arithmetic represent tasks through internal activations and weight differences (Todd et al., 2024; Ilharco et al., 2023). General text embeddings remain useful features (Muennighoff et al., 2023; Xiao et al., 2024; Zhang et al., 2025c); our supervision target additionally names the workflow behaviors that clustering and construction should exploit.

Allocating workflow construction effort. Search-based agent design learns executable programs or modular compositions from performance feedback (Hu et al., 2025; Zhang et al., 2025b; Shang et al., 2025). Learned workflow generators further amortize design through preference optimization or execution-reward training (Wang et al., 2025; Gao et al., 2025). RA-MoWE organizes construction around a reusable behavioral group. Its cluster determines the examples on which an expert is optimized, while the profile supplies a seed preference and failure-conditioned guidance. This interface accommodates different search backends and makes the grouping representation part of expert construction itself.

Selecting models and orchestrating agents. Preference-trained routing, cost-aware cascades, and joint selection of roles and models adapt computation to a query (Ong et al., 2025; Chen et al., 2024; Yue et al., 2025). Related approaches align internal expert routing with task neighborhoods, learn query-conditioned agentic architectures, or train coordination policies (Li et al., 2026; Zhang et al., 2025a; Nielsen et al., 2026). RA-MoWE connects pre-execution selection to the creation of an external workflow bank through one outcome-coordinate space. Our experiments compare the implemented local adaptations under their disclosed information and resource contracts; the separate model-pool study examines the additional freedom to change backbones.

## 5 DISCUSSION AND CONCLUSION

Workflow affinity provides a common basis for learning query representations and constructing the workflows that serve them. The experiments show repeatable probe complementarity, gains from measured-affinity specialization in two regimes, and effective compact workflows initialized from cluster profiles. The encoder makes this construction reusable from query text; its assignment fidelity determines how much of the behavioral structure reaches deployment.

## REFERENCES

Beijing Academy of Artificial Intelligence. BAAI/bge-large-en-v1.5: Model card, 2023. URL https://huggingface.co/BAAI/bge-large-en-v1.5. Accessed September 16, 2026.

Lingjiao Chen, Matei Zaharia, and James Zou. FrugalGPT: How to use large language models while reducing cost and improving performance. Transactions on Machine Learning Research, 2024. URL https://openreview.net/forum?id=cSimKw5p6R.

Yilun Du, Shuang Li, Antonio Torralba, Joshua B. Tenenbaum, and Igor Mordatch. Improving factuality and reasoning in language models through multiagent debate. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pp. 11733–11763, 2024. URL https://proceedings.mlr.press/ v235/du24e.html.

Hongcheng Gao, Yue Liu, Yufei He, Longxu Dou, Chao Du, Zhijie Deng, Bryan Hooi, Min Lin, and Tianyu Pang. FlowReasoner: Reinforcing query-level meta-agents. arXiv preprint arXiv:2504.15257, 2025. URL https://arxiv.org/abs/2504.15257.

Wassily Hoeffding. Probability inequalities for sums of bounded random variables. Journal of the American Statistical Association, 58(301):13–30, 1963. doi: 10.1080/01621459.1963.10500830.

Shengran Hu, Cong Lu, and Jeff Clune. Automated design of agentic systems. In International Conference on Learning Representations, 2025. URL https://proceedings.iclr.cc/ paper\_files/paper/2025/hash/36b7acf6f6010652b3f2a433774a66fe-Abs tract-Conference.html.

Gabriel Ilharco, Marco Tulio Ribeiro, Mitchell Wortsman, Suchin Gururangan, Ludwig Schmidt, Hannaneh Hajishirzi, and Ali Farhadi. Editing models with task arithmetic. In International Conference on Learning Representations, 2023. URL https://openreview.net/forum ?id=6t0Kwf8-jrj.

Naman Jain, King Han, Alex Gu, Wen-Ding Li, Fanjia Yan, Tianjun Zhang, Sida Wang, Armando Solar-Lezama, Koushik Sen, and Ion Stoica. LiveCodeBench: Holistic and contamination free evaluation of large language models for code. In International Conference on Learning Representations, 2025. URL https://proceedings.iclr.cc/paper\_files/paper/2025 /hash/94074dd5a072d28ff75a76dabed43767-Abstract-Conference.html.

Carlos E. Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, and Karthik Narasimhan. SWE-bench: Can language models resolve real-world GitHub issues? In International Conference on Learning Representations, 2024. URL https://proceedings.iclr .cc/paper\_files/paper/2024/hash/edac78c3e300629acfe6cbe9ca88fb84 -Abstract-Conference.html.

Yujia Li, David Choi, Junyoung Chung, Nate Kushman, Julian Schrittwieser, Rémi Leblond, Tom Eccles, James Keeling, Felix Gimeno, Agustin Dal Lago, Thomas Hubert, Peter Choy, Cyprien de Masson d’Autume, Igor Babuschkin, Xinyun Chen, Po-Sen Huang, Johannes Welbl, Sven Gowal, Alexey Cherepanov, James Molloy, Daniel J. Mankowitz, Esme Sutherland Robson, Pushmeet Kohli, Nando de Freitas, Koray Kavukcuoglu, and Oriol Vinyals. Competition-level code generation with AlphaCode. Science, 378(6624):1092–1097, 2022. doi: 10.1126/science.abq115 8. URL https://arxiv.org/abs/2203.07814.

Zhongyang Li, Ziyue Li, and Tianyi Zhou. Routing manifold alignment improves generalization of mixture-of-experts LLMs. In International Conference on Learning Representations, 2026. URL https://proceedings.iclr.cc/paper\_files/paper/2026/hash/249b8d8f 41970822651435629e68a6e1-Abstract-Conference.html.

Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, Shashank Gupta, Bodhisattwa Prasad Majumder, Katherine Hermann, Sean Welleck, Amir Yazdanbakhsh, and Peter Clark. Self-Refine: Iterative refinement with self-feedback. In Advances in Neural Information Processing Systems,

volume 36, 2023. URL https://proceedings.neurips.cc/paper\_files/paper /2023/hash/91edff07232fb1b55a505a9e9f6c0ff3-Abstract-Conference. html.

Mathematical Association of America. American invitational mathematics examination, 2026. URL https://maa.org/maa-invitational-competitions/. Official competition resource; accessed September 16, 2026.

Niklas Muennighoff, Nouamane Tazi, Loïc Magne, and Nils Reimers. MTEB: Massive text embedding benchmark. In Proceedings ofthe 17th Conference ofthe European Chapter ofthe Associa tionfor Computational Linguistics, pp. 2014–2037, 2023. doi: 10.18653/v1/2023.eacl-main.148. URL https://aclanthology.org/2023.eacl-main.148/.

Stefan Nielsen, Edoardo Cetin, Peter Schwendeman, Qi Sun, Jinglue Xu, and Yujin Tang. Learning to orchestrate agents in natural language with the conductor. In International Conference on Learning Representations, 2026. URL https://openreview.net/forum?id=U23A2B UKYt.

Isaac Ong, Amjad Almahairi, Vincent Wu, Wei-Lin Chiang, Tianhao Wu, Joseph E. Gonzalez, M. Kadous, and Ion Stoica. RouteLLM: Learning to route LLMs with preference data. In International Conference on Learning Representations, 2025. URL https://proceedings. iclr.cc/paper\_files/paper/2025/hash/5503a7c69d48a2f86fc00b3dc09d e686-Abstract-Conference.html.

David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R. Bowman. GPQA: A graduate-level google-proof Q&A benchmark. In Conference on Language Modeling, 2024. URL https://openreview.net /forum?id=Ti67584b98.

John R. Rice. The algorithm selection problem. In Advances in Computers, volume 15, pp. 65–118. Elsevier, 1976. doi: 10.1016/S0065-2458(08)60520-3.

Yu Shang, Yu Li, Keyu Zhao, Likai Ma, Jiahe Liu, Fengli Xu, and Yong Li. AgentSquare: Automatic LLM agent search in modular design space. In International Conference on Learning Representations, 2025. URL https://proceedings.iclr.cc/paper\_files/paper/2025 /hash/0ae94013da7cd459402fd77874e09ee3-Abstract-Conference.html.

Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. Reflexion: Language agents with verbal reinforcement learning. In Advances in Neural Information Processing Systems, volume 36, 2023. URL https://proceedings.neurips.cc/pap er\_files/paper/2023/hash/1b44b878bb782e6954cd888628510e90-Abstr act-Conference.html.

Eric Todd, Millicent L. Li, Arnab Sen Sharma, Aaron Mueller, Byron C. Wallace, and David Bau. Function vectors in large language models. In International Conference on Learning Representations, 2024. URL https://proceedings.iclr.cc/paper\_files/paper/2024/h ash/4ae163cb8788970e53b4fd9578141139-Abstract-Conference.html.

Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V. Le, Ed H. Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models. In International Conference on Learning Representations, 2023. URL https://op enreview.net/forum?id=1PL1NIMMrw.

Yinjie Wang, Ling Yang, Guohao Li, Mengdi Wang, and Bryon Aragam. ScoreFlow: Mastering LLM agent workflows via score-based preference optimization. arXiv preprint arXiv:2502.04306, 2025. URL https://arxiv.org/abs/2502.04306.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed H. Chi, Quoc V. Le, and Denny Zhou. Chain-of-thought prompting elicits reasoning in large language models. In Advances in Neural Information Processing Systems, volume 35, 2022. URL https: //proceedings.neurips.cc/paper/2022/hash/9d5609613524ecf4f15af0f 7b31abca4-Abstract-Conference.html.

Chunqiu Steven Xia, Yinlin Deng, Soren Dunn, and Lingming Zhang. Demystifying LLM-based software engineering agents. Proceedings of the ACM on Software Engineering, 2(FSE), 2025. doi: 10.1145/3715754. URL https://doi.org/10.1145/3715754.

Shitao Xiao, Zheng Liu, Peitian Zhang, Niklas Muennighoff, Defu Lian, and Jian-Yun Nie. C-Pack: Packed resources for general chinese embeddings. In Proceedings ofthe 47th International ACM SIGIR Conference on Research and Development in Information Retrieval, pp. 641–649, 2024. doi: 10.1145/3626772.3657878. URL https://sigir-2024.github.io/proceedi ngs.html.

Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Thomas L. Griffiths, Yuan Cao, and Karthik Narasimhan. Tree of thoughts: Deliberate problem solving with large language models. In Advances in Neural Information Processing Systems, volume 36, 2023a. URL https://procee dings.neurips.cc/paper/2023/hash/271db9922b8d1f4dd7aaef84ed5ac70 3-Abstract.html.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. ReAct: Synergizing reasoning and acting in language models. In International Conference on Learning Representations, 2023b. URL https://openreview.net/forum?id=WE\_v luYUL-X.

Yanwei Yue, Guibin Zhang, Boyang Liu, Guancheng Wan, Kun Wang, Dawei Cheng, and Yiyan Qi. MasRouter: Learning to route LLMs for multi-agent systems. In Proceedings ofthe 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 15549– 15572. Association for Computational Linguistics, 2025. doi: 10.18653/v1/2025.acl-long.757. URL https://aclanthology.org/2025.acl-long.757/.

Amy Zhang, Rowan McAllister, Roberto Calandra, Yarin Gal, and Sergey Levine. Learning invari ant representations for reinforcement learning without reconstruction. In International Conference on Learning Representations, 2021. URL https://openreview.net/forum?id= -2FCwDKRREu.

Guibin Zhang, Luyang Niu, Junfeng Fang, Kun Wang, Lei Bai, and Xiang Wang. Multi-agent architecture search via agentic supernet. In Proceedings ofthe 42nd International Conference on Machine Learning, volume 267 of Proceedings ofMachine Learning Research, pp. 75834–75852, 2025a. URL https://proceedings.mlr.press/v267/zhang25bi.html.

Jiayi Zhang, Jinyu Xiang, Zhaoyang Yu, Fengwei Teng, Xiong-Hui Chen, Jiaqi Chen, Mingchen Zhuge, Xin Cheng, Sirui Hong, Jinlin Wang, Bingnan Zheng, Bang Liu, Yuyu Luo, and Chenglin Wu. AFlow: Automating agentic workflow generation. In International Conference on Learning Representations, 2025b. URL https://proceedings.iclr.cc/paper\_files/pape r/2025/hash/5492ecbce4439401798dcd2c90be94cd-Abstract-Conferenc e.html.

Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang, Huan Lin, Baosong Yang, Pengjun Xie, An Yang, Dayiheng Liu, Junyang Lin, Fei Huang, and Jingren Zhou. Qwen3 embedding: Advancing text embedding and reranking through foundation models. arXiv preprint arXiv:2506.05176, 2025c. URL https://arxiv.org/abs/2506.05176.

Denny Zhou, Nathanael Schärli, Le Hou, Jason Wei, Nathan Scales, Xuezhi Wang, Dale Schuurmans, Claire Cui, Olivier Bousquet, Quoc V. Le, and Ed H. Chi. Least-to-most prompting enables complex reasoning in large language models. In International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=WZH7099tgfM.

## A PROOFS AND EFFECTIVE BOUNDARIES

The analysis separates three questions: what a finite probe panel identifies, when an estimated representation preserves a decision, and when a probe’s measured advantage survives conversion to an executable expert. The arguments adapt standard concentration and decision-theoretic tools; they are not claims of new concentration inequalities or a universal sufficiency theorem. Their purpose is to identify assumptions that an implementation and an evaluation must satisfy. This perspective follows the distinction between an instance representation and the algorithm selected from it (Rice, 1976). The analogy to task-relevant representation learning (Zhang et al., 2021) concerns preserving decisions; our finite vector is not a bisimulation representation.

## A.1 PROBABILITY SPACE AND FIXED-PROTOCOL CONVENTION

Let $Q \sim \mathcal { D }$ and fix an executable panel $\mathcal { P } = \{ p _ { 1 } , . . . , p _ { K } \}$ . A protocol specifies the model and endpoint, prompts, available tools, answer extraction, score, execution limits, and failure handling. Under this protocol, $Y _ { k , s } ( q ) \in [ 0 , 1 ]$ is the score of execution s of probe k, $f _ { k } ( q ) = \mathbb { E } [ Y _ { k , s } ( q ) \ |$ $q ] .$ , and $\begin{array} { r } { \bar { f } _ { k } ( q ) = S _ { k } ^ { - 1 } \sum _ { s = 1 } ^ { S _ { k } } Y _ { k , s } ( q ) } \end{array}$ . The conditional expectation averages execution randomness, not changes to the protocol. Conditioning on successful retries defines a different estimand from assigning failed executions score zero.

Throughout the routing analysis, normalization statistics, centroids, the encoder, and the deployed expert library are fixed, or we condition on the training data and randomness that produced them. Evaluation queries are fresh. Write $\Lambda = \mathrm { d i a g } ( \sigma _ { 1 } , \dots , \sigma _ { K } ) $ , where every $\sigma _ { k } > 0 .$ , and let $z ( q ) =$ $\Lambda ^ { - 1 } ( f ( q ) - \mu )$ and $\bar { z } ( q ) = \Lambda ^ { - 1 } ( \bar { f } ( q ) - \mu )$ . Thus z is the population embedding and z¯ its measured estimate. The encoder predicts ${ \hat { z } } ( q ) ~ = ~ \dot { g } _ { \theta } ( \psi ( q ) )$ ; write $x _ { i } ~ = ~ \psi ( q _ { i } )$ for an individual example and $X ~ = ~ \psi ( Q )$ for a fresh random query. These objects remain distinct even when measured and predicted embeddings lead to the same expert. The J centroids are pairwise distinct and use a fixed tie rule. If a query lies on a nearest-centroid tie, its margin is zero. We write $R _ { w } ( q ) =$ $\mathbb { E } [ \mathrm { s c o r e } ( w , q ) \mid q ] \in \dot { [ 0 , 1 ] }$ for the expected reward of an executable expert. Equality of selected experts implies equality of these expected rewards, not of independent realized scores.

## A.2 MEASUREMENT ACCURACY AND THE INFORMATION IN A FINITE PANEL

Proposition 1 (Affinity estimation preserves probe decisions). Fix a query and protocol. If each probe’s S repetitions are independent, identically distributed, and bounded in $[ 0 , { \bar { 1 } } ]$ , then

$$
\begin{array} { r } { \operatorname* { P r } \bigl ( \| \bar { f } ( q ) - f ( q ) \| _ { \infty } > \epsilon \bigr ) \le 2 K e ^ { - 2 S \epsilon ^ { 2 } } . } \end{array}\tag{5}
$$

On the event $\| { \bar { f } } - f \| _ { \infty } \leq \epsilon ,$ , selecting $\hat { k } \in \arg \operatorname* { m a x } _ { k } \bar { f } _ { k } ( q )$ incurs expected probe regret at most $2 \epsilon$

Proof of Proposition 1. For each fixed q and $k ,$ independent bounded executions give

$$
\operatorname* { P r } \bigl ( | \bar { f } _ { k } ( q ) - f _ { k } ( q ) | > \epsilon ~ | ~ q \bigr ) \leq 2 \exp ( - 2 S \epsilon ^ { 2 } )\tag{6}
$$

by Hoeffding’s inequality (Hoeffding, 1963). A union bound over $K$ coordinates proves the stated 2K $\mathrm { e x p } ( - 2 \bar { S \epsilon } ^ { 2 } )$ bound. Independence between different probes is not required. Let $k ^ { \ast } \in$ arg max<sub>k</sub> $f _ { k } ( q )$ and $\ddot { k } \in$ arg max<sub>k</sub> $\bar { f } _ { k } ( q )$ . On the uniform accuracy event,

$$
f _ { k ^ { * } } ( q ) - f _ { \hat { k } } ( q ) = f _ { k ^ { * } } ( q ) - \bar { f } _ { k ^ { * } } ( q ) + \bar { f } _ { k ^ { * } } ( q ) - \bar { f } _ { \hat { k } } ( q ) + \bar { f } _ { \hat { k } } ( q ) - f _ { \hat { k } } ( q )\tag{7}
$$

$$
\leq \epsilon + 0 + \epsilon .\tag{8}
$$

Thus a unique best probe with population gap exceeding 2ϵ is recovered. For transfer between two queries, replace $\bar { f } ( q )$ in this argument by $\bar { f } ( q ^ { \prime } )$ . I $\begin{array} { r } { \mathrm { \Delta ~ f ~ } \| f ( \bar { q } ) - f ( q ^ { \prime } ) \| _ { \infty } \leq \epsilon } \end{array}$ , using a best probe for $q ^ { \prime }$ on q loses at most 2ϵ relative to the best probe for $q .$ □

The same argument connects standardized distance to probe transfer. Let $\sigma _ { \operatorname* { m a x } } = \operatorname* { m a x } _ { k } \sigma _ { k }$ . Since $f ( q ) - f ( q ^ { \prime } ) \overline { { { \vphantom { f } } } } = \Lambda ( z ( q ) - z ( q ^ { \prime } ) )$ , the condition $\| z ( q ) - z ( q ^ { \prime } ) \| _ { 2 } \leq \delta$ implies $\| f ( q ) - f ( q ^ { \prime } ) \| _ { \infty } \leq$ $\sigma _ { \mathrm { m a x } } \delta .$ Thus the expected score lost by transferring the best probe is at most $2 \sigma _ { \operatorname* { m a x } } \delta .$ The scale factor is necessary because distances in standardized coordinates are not raw score differences.

For unequal repeat counts the failure bound is $\begin{array} { r } { 2 \sum _ { k } \exp ( - 2 S _ { k } \epsilon ^ { 2 } ) } \end{array}$ ; for $N$ prespecified queries with equal repeat counts, a further union bound replaces K by $N K$ . In particular, $\epsilon _ { S } ( \delta ) \ = $ $\sqrt { \log ( 2 K / \delta ) / ( 2 S ) }$ is a sufficient single-query radius at confidence $1 - \delta$ . For $K = 1 2 , S = 3 .$ and $\delta = 0 . 0 5$ , this radius is approximately 1.014, hence it does not certify accurate rankings for scores in [0, 1]. The result explains the role of repetitions but is not a nonvacuous numerical certificate for our three-repeat fingerprints.

Standardization changes both noise and the decision geometry. For fixed train-fitted Λ,

$$
\| \bar { z } - z \| _ { 2 } \leq \| \bar { f } - f \| _ { \infty } \left( \sum _ { k = 1 } ^ { K } \sigma _ { k } ^ { - 2 } \right) ^ { 1 / 2 } .\tag{9}
$$

Small coordinate scales can therefore amplify execution noise. Positive diagonal scaling retains the information in the raw vector, but changes Euclidean neighborhoods and cross-coordinate rankings. Our code substitutes one for nearly zero fitted standard deviations; it does not otherwise cap this amplification. Equation (9) conditions on normalization fitted without the evaluated query. It is not a joint guarantee for a training observation reused to fit its own scale, and it does not cover nonlinear per-query unit normalization.

The finite-panel guarantee has a precise information boundary. The vector $f ( q )$ determines the expected reward of each existing probe and of any randomized choice among them: conditional on fixed mixture weights $a \in \Delta _ { K }$ , that reward is $a ^ { \top } f ( q )$ . Consequently, $\| \bar { f } ( q ) - f ( q ^ { \prime } ) \| _ { \infty }$ is a task-dependent pseudometric for these decisions. It need not distinguish queries requiring different unseen compositions.

Proposition 2 (Finite-panel indistinguishability). There exist a probe panel, a query distribution, and a two-expert bank such that the queries have identical population fingerprints, but every selectorfrom that bank using only the fingerprint has positive regret, even with infinitely many probe executions.

Proof. Let $Q$ be uniform on $\{ q _ { 0 } , q _ { 1 } \}$ and set $f _ { k } ( q _ { 0 } ) = f _ { k } ( q _ { 1 } ) = 1 / 2$ for every $k .$ . Let the selectable expert bank consist of $w _ { 0 }$ and $w _ { 1 }$ , with rewards $R _ { w _ { 0 } } ( q _ { 0 } ) = \dot { R } _ { w _ { 1 } } ( q _ { 1 } ) = 1$ and $R _ { w _ { 0 } } ( q _ { 1 } ) =$ $\begin{array} { r } { R _ { w _ { 1 } } ( q _ { 0 } ) = 0 . } \end{array}$ A selector observing only $f ( Q )$ sees the same input for both queries. If it chooses w with probability $^ { a , }$ its mean reward is $( a + ( 1 - a ) ) / 2 = 1 / 2$ , whereas a query-aware oracle achieves one. The regret is $1 / 2$ for every a. Standardization and perfect estimation of $f$ do not alter this example. □

An extension to a larger workflow class therefore needs an extra assumption. For example, suppose $| R _ { w } ( q ) - h _ { w } ( z ( q ) ) | \le \eta$ for all $q , w _ { \mathrm { { i } } }$ , every $h _ { w }$ is L-Lipschitz, and a selector is within κ of maximizing $h _ { w } ( \tilde { z } )$ . Adding and subtracting the two surrogate rewards gives

$$
\operatorname* { s u p } _ { w } R _ { w } ( q ) - R _ { \tilde { w } } ( q ) \leq 2 \eta + 2 L \| \tilde { z } - z ( q ) \| _ { 2 } + \kappa .\tag{10}
$$

The substantive assumption is the factorization of rewards through z; neither finite-panel concentration nor good probe prediction establishes it. We do not assume this factorization in our main results.

## A.3 ROUTE STABILITY AND THE INFORMATION AVAILABLE TO AN ENCODER

Proposition 3 (A margin condition for faithful assignment). Condition on fixed distinct centroids and a fixed encoder. For a uniquely assigned point $z , l e t a = r _ { C } ( z )$ and

$$
m _ { C } ( z ) = \operatorname* { m i n } _ { b \neq a } \frac { \| z - c _ { b } \| _ { 2 } ^ { 2 } - \| z - c _ { a } \| _ { 2 } ^ { 2 } } { 2 \| c _ { b } - c _ { a } \| _ { 2 } } .\tag{11}
$$

Set $m _ { C } ( z ) = 0$ on ties. $I f \| \hat { z } - z \| _ { 2 } < m _ { C } ( z )$ , the assignments agree. With $\mathcal { E } = \mathbb { E } \| g _ { \theta } ( X ) - z ( Q ) \| _ { 2 } ^ { 2 }$ for every $t > 0 ,$

$$
\operatorname* { P r } \{ r _ { C } ( g _ { \theta } ( X ) ) \neq r _ { C } ( z ( Q ) ) \} \leq \operatorname* { P r } \{ m _ { C } ( z ( Q ) ) \leq t \} + \mathcal { E } / t ^ { 2 } .\tag{12}
$$

Proof of Proposition 3. Set $\boldsymbol { a } = \boldsymbol { r } _ { C } ( \boldsymbol { z } )$ and write $\tilde { z } = z + e$ . For every competing centroid b,

$$
\| \tilde { { \boldsymbol { z } } } - { \boldsymbol { c } } _ { b } \| _ { 2 } ^ { 2 } - \| \tilde { { \boldsymbol { z } } } - { \boldsymbol { c } } _ { a } \| _ { 2 } ^ { 2 } = \| { \boldsymbol { z } } - { \boldsymbol { c } } _ { b } \| _ { 2 } ^ { 2 } - \| { \boldsymbol { z } } - { \boldsymbol { c } } _ { a } \| _ { 2 } ^ { 2 } + 2 e ^ { \top } ( { \boldsymbol { c } } _ { a } - { \boldsymbol { c } } _ { b } )\tag{13}
$$

$$
\geq 2 \| c _ { b } - c _ { a } \| _ { 2 } \big ( m _ { C } ( z ) - \| e \| _ { 2 } \big ) .\tag{14}
$$

If $\| e \| _ { 2 } < m _ { C } ( z )$ , all comparisons are strictly positive, so the nearest centroid is unchanged. With $Z = z ( Q ) , X = \psi ( Q )$ , and $\mathcal { E } = \mathbb { E } \| g _ { \theta } ( X ) - \bar { Z } \| _ { 2 } ^ { 2 }$ , a disagreement can occur only if $m _ { C } ( Z ) \leq t$ or $\| g _ { \theta } ( X ) - Z \| _ { 2 } \geq t .$ . The union bound and Markov’s inequality give

$$
\operatorname* { P r } \{ r _ { C } ( g _ { \theta } ( X ) ) \neq r _ { C } ( Z ) \} \le \operatorname* { P r } \{ m _ { C } ( Z ) \le t \} + \mathcal { E } / t ^ { 2 } .\tag{15}
$$

Finally, two selected experts have the same expected reward when their indices agree and a reward difference of magnitude at most one otherwise. Taking expectations proves the reward-difference bound. □

The simpler nearest-versus-second-nearest distance gap $\Delta _ { C } ( z )$ gives the sufficient condition $2 \Vert \tilde { z } -$ $z \| _ { 2 } < \Delta _ { C } ( z )$ by the triangle inequality. The exact Voronoi margin can be less conservative. Neither condition states that the centroids recover optimal workflow classes or that a preserved route selects a high-quality expert. In particular, the measured-fingerprint route is not an upper bound on the predicted route’s reward. When multiple indices execute the same specification, index disagreement also overstates disagreement between executable behaviors.

Measured and population routes differ. Writing $\bar { Z } = \bar { z } ( Q )$ , both sources of error matter when comparing the measured route with the encoder:

$$
\operatorname* { P r } \{ r _ { C } ( \bar { Z } ) \neq r _ { C } ( g _ { \theta } ( X ) ) \} \leq \operatorname* { P r } \{ m _ { C } ( Z ) \leq t \} + \operatorname* { P r } \{ \| \bar { Z } - Z \| _ { 2 } \geq t \}\tag{16}
$$

$$
+ \operatorname* { P r } \{ \| g _ { \theta } ( X ) - Z \| _ { 2 } \geq t \} .\tag{17}
$$

Alternatively, $\| g _ { \theta } ( X ) - \bar { Z } \| _ { 2 } < m _ { C } ( \bar { Z } )$ certifies agreement with the measured route for that observation. This is a deterministic diagnostic, not a certificate of population routing accuracy.

Proposition 4 (Prediction noise and conditional information). Assumefinite second moments,fixed affine normalization, and $\bar { Z } = Z + \xi$ with $\mathbb { E } [ \xi \mid Q ] = 0$ . For any fixed predictor g and $g ^ { * } ( { \dot { X } } ) = $ $\mathbf { \mathbb { E } } [ Z \mid X ]$

$$
\begin{array} { r } { \mathbb { E } \| g ( X ) - \bar { Z } \| _ { 2 } ^ { 2 } = \mathbb { E } \| g ( X ) - g ^ { * } ( X ) \| _ { 2 } ^ { 2 } + \mathbb { E } \operatorname { t r } \operatorname { C o v } ( Z \mid X ) + \mathbb { E } \| \xi \| _ { 2 } ^ { 2 } . } \end{array}\tag{18}
$$

For the population route $L = r _ { C } ( Z )$ , any selector $\hat { L }$ using only $X$ satisfies

$$
\operatorname* { P r } ( \hat { L } \neq L ) \geq \operatorname { \mathbb { E } } \left[ 1 - \operatorname* { m a x } _ { j } \operatorname* { P r } ( L = j \mid X ) \right] .\tag{19}
$$

Proof. Expand $g ( X ) - \bar { Z } = ( g ( X ) - g ^ { * } ( X ) ) + ( g ^ { * } ( X ) - Z ) - \xi .$ The first two summands are orthogonal in expectation by conditional-mean orthogonality. The noise is orthogonal to both because $\breve { X }$ and $Z$ are functions of $Q$ and $\mathbb { E } [ \xi \mid Q ] = \mathbf { \bar { 0 } }$ The middle squared norm equals the expected trace of the conditional covariance, proving (18). Conditional on $\bar { X ( \mathbf { \Psi } ) } = x .$ , any deterministic prediction succeeds with probability at most ma $\operatorname { x } _ { j } \operatorname { \bar { P r } } ( L = j \mid X = x )$ . A randomized prediction is a convex combination of these success probabilities and obeys the same bound. Average over X. □

For independent repeats within each coordinate,

$$
\mathbb { E } [ \| \xi \| _ { 2 } ^ { 2 } \mid Q = q ] = \sum _ { k = 1 } ^ { K } \frac { \operatorname { V a r } ( Y _ { k } ( q ) \mid q ) } { S _ { k } \sigma _ { k } ^ { 2 } } \leq \sum _ { k = 1 } ^ { K } \frac { 1 } { 4 S _ { k } \sigma _ { k } ^ { 2 } } .\tag{20}
$$

Cross-coordinate dependence does not affect this sum of squared coordinate errors. More executions can reduce measurement noise, but cannot remove the conditional-information term for a fixed $\psi .$ That term can nevertheless be zero; a frozen representation does not by itself establish a positive information floor. Our mixed cosine/MSE training objective is not pure squared-loss minimization, so conditional-mean optimality does not claim that the trained network attains $g ^ { * }$ . Reported coordinate-averaged MSE must be multiplied by K to match the vector-error convention above.

## A.4 FROM MEASURED PROBE PREFERENCE TO AN EXECUTABLE SEED

Proposition 5 (Utility of an affinity-derived initialization). Fix a cluster. Let $r _ { k }$ be probe $k ' s e x –$ pected raw score and $s _ { k }$ its estimate, with max<sub>k</sub> $| s _ { k } - r _ { k } | \le \epsilon$ . Let ${ \mathcal { R } } \subseteq \{ 1 , \ldots , K \}$ } be the nonempty set ofrunnable probe indices and $\hat { k } \in \mathcal { R }$ the chosen probe. Define $\alpha = \operatorname* { m a x } _ { k } r _ { k } - \operatorname* { m a x } _ { k \in \mathcal { R } } r _ { k }$ and $\beta = \operatorname* { m a x } _ { k \in \mathcal { R } } s _ { k } - s _ { \hat { k } }$ . If its mapped seed $\boldsymbol { v } _ { \hat { \boldsymbol { k } } }$ has expected cluster utility $U ( v _ { \hat { k } } ) \geq r _ { \hat { k } } - \tau _ { \ast }$ , then

$$
U ( v _ { \hat { k } } ) \geq \operatorname* { m a x } _ { k } r _ { k } - \alpha - \beta - 2 \epsilon - \tau .\tag{21}
$$

ProofofProposition 5. Suppress the cluster index. Let $\hat { k } \in \mathcal { R }$ be the selected runnable probe and $\boldsymbol { v } _ { \hat { \boldsymbol { k } } }$ its executable seed. On the event ma $\mathrm { { \sc { u x } } } _ { k } \ : | s _ { k } - r _ { k } | \leq \epsilon$ , use the definitions $\alpha = \mathrm { m a x } _ { k } r _ { k } - \mathrm { m a x } _ { k \in \mathcal { R } } r _ { k }$ and $\beta = \operatorname* { m a x } _ { k \in \mathcal { R } } s _ { k } - s _ { \hat { k } }$ to obtain

$$
\operatorname* { m a x } _ { k } r _ { k } = \alpha + \operatorname* { m a x } _ { k \in \mathcal { R } } r _ { k }\tag{22}
$$

$$
\leq \alpha + \operatorname* { m a x } _ { k \in \mathcal { R } } s _ { k } + \epsilon = \alpha + s _ { \hat { k } } + \beta + \epsilon\tag{23}
$$

$$
\leq \alpha + r _ { \hat { k } } + \beta + 2 \epsilon \leq \alpha + U ( v _ { \hat { k } } ) + \beta + 2 \epsilon + \tau .\tag{24}
$$

Rearranging proves the claim. Only the one-sided transfer condition $U ( v _ { \hat { k } } ) \geq r _ { \hat { k } } - \tau$ for the chosen seed is needed. □

This form allows the implemented seed to maximize standardized scores rather than raw utility: $\hat { k } \in$ arg max $\lceil \iota \in \mathcal { R }  \big ( s _ { k } - \mu _ { k } \big ) / \sigma _ { k }$ . The raw-score deficit $\beta$ is therefore essential. Even perfect measurement need not make it zero. For example, raw cluster means (0.9, 0.6), fitted global means (0.89, 0.3), and scales (0.1, 0.1) yield standardized means (0.1, 3), selecting the second probe and incurring a raw deficit 0.3. This is a deliberate distinction between relative cluster preference and absolute reward; a standardized seed rule needs empirical justification at this interface.

Likewise, τ cannot be omitted because probe names do not specify executable equivalence. The current mapping merges several probe families, omits tool-use and ReAct executors, and adds a chain-of-thought instruction to every guided seed. The proposition does not assume these transformations preserve reward. The coverage term α measures unavailable panel capabilities; it does not measure the gap to all possible workflows.

Sampling a cluster profile. Suppose a fixed cluster has n fresh queries sampled independently from its conditional query distribution, each with S independent executions per probe, and let $s _ { k } =$ $n ^ { - 1 } \textstyle \sum _ { i } \bar { f } _ { k } ( Q _ { i } )$ and $r _ { k } = \mathbb { E } [ f _ { k } ( Q ) \ |$ Q belongs to the cluster]. Separating query sampling from execution noise gives

$$
\operatorname* { P r } \{ \operatorname* { m a x } _ { k } \left. s _ { k } - r _ { k } \right. > a + b \} \leq 2 K e ^ { - 2 n a ^ { 2 } } + 2 K e ^ { - 2 n S b ^ { 2 } } .\tag{25}
$$

The first term applies Hoeffding to $f _ { k } ( Q _ { i } )$ ; the second applies it conditionally on the queries to the $n S$ centered execution scores, assuming conditional independence across these executions. More repeated executions do not eliminate finite-query uncertainty. This argument does not directly apply when cluster membership is selected using those same noisy executions, as in clustering measured training fingerprints.

## A.5 WHAT VALIDATION AND OPERATOR PRIORS CAN GUARANTEE

Lemma 1 (Seed retention under accurate selection). Let a candidate set V contain seed $v ,$ with utilities $U ( w )$ and recorded estimates $\widehat { U } ( w )$ . Suppose $\begin{array} { r } { \operatorname* { s u p } _ { w \in \mathcal { V } } | \widehat { U } ( w ) - U ( w ) | \leq \eta } \end{array}$ and $\widehat { U } ( w _ { \mathrm { o u t } } ) \geq$ ma $\mathfrak { c } _ { w \in \mathcal { V } } \widehat { U } ( w ) - h$ . Then

$$
U ( w _ { \mathrm { o u t } } ) \geq \operatorname* { m a x } _ { w \in \mathcal { V } } U ( w ) - 2 \eta - h \geq U ( v ) - 2 \eta - h .\tag{26}
$$

Proof. For every $w \in \mathcal { V } , U ( w _ { \mathrm { o u t } } ) \geq \widehat { U } ( w _ { \mathrm { o u t } } ) - \eta \geq \widehat { U } ( w ) - h - \eta \geq U ( w ) - h - 2 \eta$ . Maximize over w. □

For M candidates frozen independently of n independently sampled held-out queries, with independent execution randomness across queries, bounded query scores and a union bound yield the sufficient radius $\eta = \sqrt { \log ( 2 M / \delta ) / ( 2 n ) }$ . Candidate scores may be dependent across workflows. Repeating executions on the same n queries does not turn the population sample size into $n S .$ . Importantly, substituting the number of observed candidates after adapting them to validation failures does not justify this finite-class guarantee. It requires a prespecified class, fresh validation with appropriate error allocation, or a separate final holdout. An intact external test evaluates the selected procedure but does not retroactively certify every accepted edit.

The implemented acceptance rule retains the incumbent unless a proposal’s recorded score exceeds it by a margin h. With fixed recorded scores, the final incumbent is within h of the best evaluated score: a rejected candidate was at most its then-incumbent plus h, and incumbent scores only increase. This is an empirical property. The margin $2 / n$ used in our search is not a calibrated confidence radius. An accepted edit is guaranteed to improve population utility under uniform error η only when its measured increment exceeds 2η. Our adaptive validation protocol has no such established uniform certificate.

A hard prior introduces an exclusion cost. Let $U _ { j } ( w )$ be expected utility on a fixed cluster $j ,$ and let H denote a search class, distinct from the retained expert bank W. For a permitted class $\mathcal { H } _ { j } ^ { \mathrm { p r i o r } } \subseteq \mathcal { H }$ , define $\begin{array} { r } { L _ { j } = \operatorname* { s u p } _ { w \in \mathcal { H } } U _ { j } ( w ) - \operatorname* { s u p } _ { w \in \mathcal { H } _ { \mathfrak { i } } ^ { \mathrm { p r i o r } } } U _ { j } ( w ) } \end{array}$ . If every excluded workflow can be transformed into a permitted one while losing at most $\zeta _ { j } ,$ , then $L _ { j } \ \leq \ \zeta _ { j } .$ , by applying the transformation to an arbitrarily near-optimal workflow. Low standalone probe reward is insufficient for this condition: two operators can each have reward zero alone and reward one together. The seed exemption ensures an available starting workflow; it does not establish that an operator ban is harmless. Similarly, if each guided proposal hits a target set with conditional probability at least p given any no-hit history, $T$ proposals hit it with probability at least $1 - ( 1 - \overline { { p } } ) ^ { T }$ , by multiplying conditional no-hit probabilities. A speedup requires evidence that guidance improves this probability under a matched budget; using fingerprints in a prompt alone does not supply that evidence.

## A.6 A CONDITIONAL BOUND FOR THE COMPLETE PIPELINE

The preceding statements can be combined without assuming that the panel is sufficient for all generated workflows. The comparator is explicitly the population best probe, not an unrestricted workflow oracle. Let $L = r _ { C } ( { \bar { z } } ( Q ) ) , \pi _ { j } = { \bar { \operatorname { P r } } } ( { \bar { L } } = j )$ , and for nonempty clusters define $r _ { j k } =$ $\mathbb { E } [ f _ { k } ( Q ) \mid L = j ]$ and $U _ { j } ( w ) \stackrel { \cdot } { = } \mathbb { E } [ \tilde { R } _ { w } ( \stackrel { \cdot } { Q } ) \mid L \stackrel { \cdot } { = } j ]$ . The population assignment L uses expected affinities; it is distinct from experimental assignment A, which uses measured test-query affinities. Define the cluster heterogeneity cost

$$
H _ { C } = \mathbb { E } \operatorname* { m a x } _ { k } f _ { k } ( Q ) - \sum _ { j = 1 } ^ { J } \pi _ { j } \operatorname* { m a x } _ { k } r _ { j k } \geq 0 .\tag{27}
$$

Nonnegativity follows by interchanging a maximum and a conditional expectation. This is the price, relative to per-query probe selection, of requiring one probe decision per population cluster; generated experts may recover or exceed this lost value.

Theorem 1 (Conditional composition relative to the probe oracle). Fix the protocol, normalization, centroids, encoder, and expert library. For each cluster with $\pi _ { j } > 0$ , assume the profile-accuracy and seed-transfer conditions of Proposition 5 hold with parameters $( \alpha _ { j } , \beta _ { j } , \epsilon _ { j } , \tau _ { j } )$ for the population cluster $L = j$ . Assume the final expert $w _ { j }$ satisfies Lemma 1 with error $\eta _ { j }$ and selection deficit $h _ { j }$ . Set

$$
B _ { j } = { \alpha } _ { j } + { \beta } _ { j } + 2 { \epsilon } _ { j } + { \tau } _ { j } + 2 { \eta } _ { j } + { h } _ { j } .\tag{28}
$$

On these simultaneous accuracy events, for every $t > 0 ,$

$$
\mathbb { E } \left[ \operatorname* { m a x } _ { k } f _ { k } ( Q ) - R _ { w _ { r _ { C } ( g _ { \theta } ( X ) ) } } ( Q ) \right]\tag{29}
$$

$$
\leq H _ { C } + \sum _ { j = 1 } ^ { J } \pi _ { j } B _ { j } + \operatorname* { P r } \{ m _ { C } ( z ( Q ) ) \leq t \} + \frac { \mathbb { E } \| g _ { \theta } ( X ) - z ( Q ) \| _ { 2 } ^ { 2 } } { t ^ { 2 } } .\tag{30}
$$

All expectations and probabilities on this event are over a fresh query under the fixedfitted objects.   
Empty clusters contribute zero.

Table 2: Measured probe families and their mapping into the guided interpreter. A mapping is an implementation approximation, not an equivalence assertion. CoT is always used for the interpreter’s base candidates.
<table><tr><td>Probe</td><td>Measured computational intervention</td><td>Guided interpreter mapping</td></tr><tr><td>Direct</td><td>One formatted response</td><td>Direct flag is inert; base candidate still uses CoT.</td></tr><tr><td>CoT</td><td>Step-by-step reasoning prompt (Wei et al., 2022)</td><td>CoT base candidate.</td></tr><tr><td>CoT-SC</td><td>Five inner samples plus answer aggregation (Wang et al., 2023)</td><td>Sampling and selection; seed uses three samples.</td></tr><tr><td>ToT</td><td>Bounded branching, evaluation and search (Yao et al., 2023a)</td><td>Sampling and selection; no tree search.</td></tr><tr><td>Self-refine</td><td>Critique and revision (Madaan et al., 2023)</td><td>One checking/rewrite call.</td></tr><tr><td>Reflexion</td><td>Outcome feedback, reflection and retry (Shinn et al., 2023)</td><td>Same revision flag; no separate feedback-memory loop.</td></tr><tr><td>RAG</td><td>Same-benchmark TF-IDF exemplar</td><td>CoT flag; no retrieval in generated expert.</td></tr><tr><td>ReAct</td><td>retrieval; query excluded Bounded action-observation interaction</td><td>Unavailable; cannot seed its behavior.</td></tr><tr><td>Tool use</td><td>(Yao et al., 2023b) Python execution or public-test feedback,</td><td>Unavailable; cannot seed its behavior.</td></tr><tr><td>Debate</td><td>domain dependent Independent debaters and a judge (Du et al.,</td><td>Two fixed persona candidates plus selection.</td></tr><tr><td>Role team</td><td>2024) Three complementary roles and a</td><td>Same two-persona branch as debate.</td></tr><tr><td>Decompose</td><td>coordinator Subproblem decomposition and synthesis (Zhou et al., 2023)</td><td>One plan prepended to subsequent candidate calls.</td></tr></table>

Proof. Proposition 5 and Lemma 1 imply $U _ { j } ( w _ { j } ) \geq \operatorname* { m a x } _ { k } r _ { j k } - B _ { j }$ . Hence

$$
\begin{array} { l } { \displaystyle \mathbb { E } R _ { w _ { L } } ( Q ) = \displaystyle \sum _ { j } \pi _ { j } U _ { j } ( w _ { j } ) \geq \displaystyle \sum _ { j } \pi _ { j } \operatorname* { m a x } _ { k } r _ { j k } - \displaystyle \sum _ { j } \pi _ { j } B _ { j } } \\ { \displaystyle \qquad = \mathbb { E } \operatorname* { m a x } _ { k } f _ { k } ( Q ) - H _ { C } - \displaystyle \sum _ { j } \pi _ { j } B _ { j } . } \end{array}\tag{31}
$$

(32)

Replacing L by the predicted route reduces expected reward by at most their disagreement probability. Apply Equation (15) and rearrange. □

This theorem distinguishes within-cluster heterogeneity, missing executable capabilities, standardized-ranking distortion, probe-to-seed transfer, selection uncertainty, and encoder decision error. It does not establish that any term is small. In particular, the observed clusters were fitted to noisy fingerprints, the measured profiles do not directly sample the latent clusters in the theorem, and adaptive search does not provide the stated selection event. Endpoint or scoring changes also change $f$ and $R _ { w }$ . The result therefore specifies effective boundaries and targets for validation, rather than certifying the existing pipeline or comparing it to every possible generated workflow.

## B IMPLEMENTATION AND DESIGN RATIONALE

## B.1 PROBE PANEL AND ITS INTERPRETATION

The panel is a set of concrete interventions on one executor, not a claim that the names of reasoning methods form an orthogonal or complete basis. Table 2 distinguishes the measured implementations from their generator-side approximations. Related publications motivate the computational families; the local prompts, budgets, and feedback contracts define the actual experiments. In particular, the measured tool-use and ReAct probes have execution access that the guided interpreter lacks, and the generator’s persona branch does not reproduce the measured three-agent debate.

The canonical mixed-task panel contains $K = 1 2$ probes, and each fingerprint coordinate averages $S = 3$ outer repetitions. For the CoT-SC probe, the five inner samples occur within each repetition and must not be counted as five independent fingerprint observations. The score retains its taskspecific meaning; fractional coding-test utility is not a calibrated probability of solving an unseen hidden test set. All normalizations preserve the raw coordinates in the stored artifact.

The relative-normalization variant subtracts a query’s mean outcome and divides by the resulting vector norm; this changes the estimand and removes absolute difficulty information. It does not inherit the affine-noise decomposition in Equation (18). Comparing raw, standardized, and relative coordinates is thus a test of a design tradeoff, not just numerical preprocessing. The default affine standardization is specified below.

## B.2 ENCODER AND CLUSTER CONSTRUCTION

Let $\mathcal { T } = \{ q _ { i } \} _ { i = 1 } ^ { N }$ denote the construction pool and $\mathcal { T } _ { \mathrm { e n c } } \subseteq \mathcal { T }$ the encoder’s outer-training subset, with $N _ { \mathrm { e n c } } = | T _ { \mathrm { e n c } } |$ . Each query has cached probe outcomes. Write $\bar { f } ( q ) = ( \bar { f } _ { 1 } ( q ) , \ldots , \bar { f } _ { K } ( q ) ) ^ { \top }$ for its measured mean scores from Equation (1). We fit the coordinate means $\mu _ { k }$ and standard deviations $\sigma _ { k }$ on $\tau _ { \mathrm { e n c } }$ and define the measured embedding by

$$
\bar { z } _ { k } ( q ) = \frac { \bar { f } _ { k } ( q ) - \mu _ { k } } { \sigma _ { k } } , \qquad \bar { z } ( q ) = \Lambda ^ { - 1 } \big ( \bar { f } ( q ) - \mu \big ) , \quad \Lambda = \mathrm { d i a g } ( \sigma _ { 1 } , \ldots , \sigma _ { K } ) .\tag{33}
$$

Here $\boldsymbol { \mu } = ( \mu _ { 1 } , \ldots , \mu _ { K } ) ^ { \top }$ , and Λ has positive diagonal scales. A fitted standard deviation below $1 0 ^ { - 8 }$ is replaced by one, avoiding division by zero; a nonconstant but low-variance coordinate can still receive a large weight. Positive standardized values indicate scores above that probe’s training mean, not probabilities. The fitted normalization is used unchanged for all remaining construction fingerprints and any later measured test fingerprints.

The encoder predicts ${ \hat { z } } ( q ) = g _ { \theta } ( \psi ( q ) )$ , where ψ is the frozen text feature extractor and $g _ { \theta }$ is the trainable head with parameters θ. Its target is $\bar { z } ( q )$ . The training objective is

$$
\mathcal { L } ( \theta ) = \frac { 1 } { { { N _ { \mathrm { { e n c } } } } } } \sum _ { q \in { \mathcal { T } _ { \mathrm { { e n c } } } } } \left[ \lambda \big ( 1 - \cos ( \hat { z } ( q ) , \bar { z } ( q ) ) \big ) + \frac { 1 - \lambda } { K } \| \hat { z } ( q ) - \bar { z } ( q ) \| _ { 2 } ^ { 2 } \right] , \quad \lambda = 0 . 7 .\tag{34}
$$

Here cos $( u , v )$ denotes cosine similarity. Its loss term encourages agreement in the relative pattern of probe responses; the second term penalizes coordinate error in the space used for cluster assignment. Dividing by K makes the squared error a per-coordinate mean. The objective depends on θ through $\hat { z } ( q )$ , and λ is a fixed implementation choice. The output already has standardized coordinates and is not standardized again at deployment.

The canonical mixed-task encoder uses L2-normalized frozen BGE-large-en-v1.5 features with a maximum input length of 512 tokens (Beijing Academy of Artificial Intelligence, 2023). Long questions can therefore lose relevant content before the trainable head; Proposition 3 does not eliminate that information loss. The head is $1 0 2 4 \to 5 1 2 \to 5 1 2 \to K$ , with ReLU and dropout .1 after each hidden layer. Adam uses learning rate .001, weight decay .0001, and batch size 32. The construction pool has $\dot { N } = 6 0 0$ queries. A stratified outer split reserves 15% (90 queries) for encoder validation, leaving $N _ { \mathrm { e n c } } = 5 1 0$ for encoder fitting. Within those 510 rows, another 15% selects the epoch count by combined loss, with maximum 200 epochs and patience 25. The final head is reinitialized and trained on the full outer-training fold for that count. Initialization and outer splitting use seed 13; the inner split uses seed 14. Normalization is fitted on the outer-training rows; it is not refitted to exclude the inner-selection rows. The outer evaluation remains excluded from normalization and epoch selection.

The choice of frozen features is motivated by limited fingerprint supervision and the ability to reuse a cached text representation. The width and loss weights are fixed capacity and optimization choices; no experiment establishes them as optimal. Predicting continuous coordinates instead of cluster labels retains information for alternative partitions and operator profiles, at the cost of learning variations that may be irrelevant to the current expert bank. A direct cluster classifier is therefore an informative ablation.

Cluster construction uses Euclidean k-means with $J = 6$ , initialization seed 13, and ten restarts. The centroids use the encoder checkpoint’s normalization. Clustering uses the construction pool, including rows reserved for encoder validation; those rows remain held out for the encoder fit but not for the bank construction. The separate test pool evaluates the composed system. Six clusters allow repeated use of one offline expert while retaining tens of construction examples in the smaller clusters. Neither cluster count nor squared-distance clustering is justified by a utility-optimality guarantee. Historical eight-cluster diagnostics are not the deployed six-cluster configuration.

Let $C = ( c _ { 1 } , \dots , c _ { J } )$ be the fitted centroids, with $c _ { j } \in \mathbb { R } ^ { K }$ . For any vector v in the standardized embedding space, the assignment rule and construction members are

$$
r _ { C } ( v ) = \underset { 1 \leq j \leq J } { \arg \operatorname* { m i n } } \| v - c _ { j } \| _ { 2 } ^ { 2 } , \qquad \mathcal { T } _ { j } = \{ q \in \mathcal { T } : r _ { C } ( \bar { z } ( q ) ) = j \} .\tag{35}
$$

The norm is Euclidean, and ties use a fixed rule. A clustering centroid $c _ { j }$ summarizes its full cluster; the generation embedding $\rho _ { j }$ averages the search-validation subset defined below and need not equal $c _ { j }$ . The canonical pipeline uses one fitted clustering for construction and assignment. The recorded embedding-guided experiment instead translates between two stored clusterings using a trainingfitted label mapping. Appendix C.4 specifies that mapping and its agreement rates; it is part of the evaluated implementation rather than an exact identity between the two geometries.

## B.3 CONSTRUCTION BACKENDS AND THEIR EXPERIMENTAL ROLES

Section 2.2 separates the information supplied to construction from the optimizer that consumes it. The main mixed-task baseline comparison uses subset-conditioned construction: six experts are searched with AFlow on the six affinity-defined training subsets. Its baseline operator vocabulary contains Custom and ScEnsemble. Each search permits five optimization rounds after its initial candidate. The global comparator $D$ uses the same search rule and round limit on the undivided training pool. The validation cap is 600 in both configurations, so the cluster sizes sum to the same nominal 600 candidate–query validations per round as D. Six searches nevertheless require more proposal calls, and different workflows incur different execution costs; this is not equality of aggregate construction tokens or dollars. Semantic and random partitions change the examples assigned to each search while preserving this construction interface.

The SWE-bench experiment instantiates the same cluster-to-bank idea by selecting among policy prompts within a fixed localize–repair executor. Its global D is the training-selected policy, whereas the separately reported AFlow adaptation searches policy instructions. Finally, the embeddingguided search experiment below supplies the search procedure with affinity-derived seeds, an operator prior, and failure-conditioned feedback. It directly exercises the use of cluster embeddings described in Section 2.2. Results from the subset-conditioned and policy-selection settings establish the utility of organizing the bank through affinities; the embedding-guided search experiment evaluates the additional construction mechanism. The backend and policy variants are component studies of the common representation, encoder, and generation framework.

## B.4 EXPERT LANGUAGE FOR EMBEDDING-GUIDED SEARCH

An expert is a dictionary containing an unordered operator set, an integer sample count between one and five, and an instruction string. Optional decomposition adds a short plan to the question. The interpreter always produces the specified number of CoT candidates; when either persona flag is present, two more candidates are added rather than substituted. If multiple candidates survive and sampling-ensemble or persona selection is enabled, a separate call chooses an answer. An optional revision then checks and rewrites it. All scored answer stages receive the same per-domain answerformat instruction. Empty final outputs and scoring failures receive zero; truncated outputs are retained and scored.

If the configured stages complete, their nominal call count is

$$
c ( w ) = d + n + 2 p + { \bf 1 } \{ n + 2 p > 1 \} { \bf 1 } \{ e = 1 { \mathrm { o r } } p = 1 \} + r ,\tag{36}
$$

where $d , p , e , r$ indicate decomposition, persona generation, ensemble selection, and revision, and n is the sample count. With fewer surviving candidates, the actual selection stage can differ; empirical costs use recorded calls. When $n > 1$ without a selection flag, the interpreter returns the first surviving candidate. Listing both persona aliases does not add another branch. These facts are disclosed to the optimizer and should be used when counting distinct executed workflows.

For each cluster, $V _ { j } \subseteq { \mathcal { T } } _ { j }$ is the search-validation set, $n _ { j } = | V _ { j } |$ , and $H _ { j } = \mathcal { T } _ { j } \setminus V _ { j }$ contains the remaining construction examples. The recorded optimizer receives statistics involving $H _ { j }$ , so this remainder is not an independent holdout. The cluster embedding is $\begin{array} { r } { \rho _ { j } = n _ { j } ^ { - 1 } \sum _ { q \in V _ { i } } \bar { z } ( q ) } \end{array}$ . Search also receives the cached probe outcomes for individual queries in $\tau _ { j }$ : the average embedding alone cannot provide probe outcomes on the current failures. The LLM optimizer receives selected texts of queries on which the current workflow scores below one, mean raw probe scores on the failed and solved subsets, whole-cluster raw probe averages, the current workflow, and previous attempts. These raw feedback summaries are distinct from the standardized coordinates used to rank probes in $\rho _ { j }$ . For a candidate $w ,$ define its recorded search score as

Algorithm 2 Recorded embedding-guided search for one cluster   
Require: Construction members ${ \overline { { \mathcal { T } _ { j } } } } ,$ , cached per-query probe outcomes, operator map, LLM opti  
mizer, proposal budget and resource limits   
1: Split $\bar { \mathcal { T } _ { j } }$ into search-validation set $V _ { j }$ and remaining examples $H _ { j }$   
2: Set $n _ { j }  \vert V _ { j } \vert ;$ ; compute $\rho _ { j }$ from $V _ { j } ^ { - }$ and rank its probe coordinates   
3: Choose the highest-ranked runnable probe and map it to seed $v _ { j }$   
4: Restrict operators using the bottom-three rule, exempting the mapped seed operator   
5: w $ v _ { j } ;$ evaluate $\widehat { U } _ { V _ { j } } ( \boldsymbol { w } )$ and retain per-query outcomes   
6: for each remaining proposal round (two in the recorded run) do   
7: Stop if a configured cost or time guard has been reached   
8: Identify failures of w on $V _ { j }$ and their probe outcomes   
9: Form feedback with failure texts, mean raw probe scores for failed/solved/all cluster mem  
bers, w, and prior attempts   
10: Ask the LLM optimizer for a specification edit; parse and sanitize it   
11: if invalid or previously evaluated specification then   
12: Record the event and continue   
13: end if   
14: Evaluate candidate $w ^ { \prime }$ on the same $V _ { j }$   
15: if $\widehat { U } _ { V _ { j } } ( w ^ { \prime } ) > \widehat { U } _ { V _ { j } } ( w ) + 2 / n _ { j }$ then   
16: $\dot { w }  w ^ { \prime } ;$ retain its outcomes for the next round   
17: end if   
18: end for   
19: return w and the full construction trace

$$
\widehat { U } _ { V _ { j } } ( w ) = \frac { 1 } { n _ { j } } \sum _ { q \in V _ { j } } \operatorname { s c o r e } ( w , q ) ,\tag{37}
$$

where each summand is the task score recorded from one execution of that candidate. This empirical mean is distinct from the expected cluster utility $U _ { j } ( w )$ in the analysis. A proposed workflow $w ^ { \prime }$ replaces the current workflow w only when

$$
\widehat { U } _ { V _ { j } } ( w ^ { \prime } ) > \widehat { U } _ { V _ { j } } ( w ) + \frac { 2 } { n _ { j } } .\tag{38}
$$

The local validation cap is $^ { 6 0 }$ queries, with at least 20 reserved locally when possible; small clusters use the remaining available validation members. Construction uses one execution per candidate, while separate final comparisons use repeated executions. Search also stops at its configured cost or wall-time guard. These guards bound resource use rather than optimize the same objective as score acceptance. The two-grid-step margin was introduced to avoid accepting small fluctuations; with heterogeneous fractional scores and adaptive proposals, it is not a confidence bound.

The bottom-three restriction ranks within the cluster, excludes an operator only when all mapped probes are low-ranked, and exempts the single operator mapped from the selected seed probe. This handles many-to-one mappings without letting one low-ranked alias veto a well-supported operator. The implementation also has a nonempty-set fallback. The prior reduces available search choices; standalone outcomes do not identify all beneficial interactions, so its exclusion loss may be positive. The three-round budget was a limited construction allocation, not evidence of search convergence.

The optimizer is asked to edit one axis: sample count, addition to the operator set, removal from it, or instruction. For recognized edit types, the sanitizer resets fields outside that axis. An addition/removal request may still change several flags on its axis. Unknown edit types are logged and do not receive the same axis constraint, although vocabulary and prior filtering still apply. No unrestricted program synthesis or Monte Carlo tree search occurs in this interpreter.

Selection independence. The cluster embedding and seed use $V _ { j }$ . However, the recorded feedback’s whole-cluster column averages over $\mathcal { T } _ { j } = V _ { j } \cup H _ { j }$ , exposing probe outcomes from the remaining examples $H _ { j }$ . Thus $H _ { j }$ is not an independent certificate of the search procedure. The later 300-query test pool is a separate evaluation pool, but controls and analyses added after observing its results remain exploratory. To apply the independent-selection result in Appendix A, candidates must be frozen before an untouched selection sample is evaluated, or a justified uniform guarantee must cover the actual adaptive procedure. Reusing $V _ { j }$ and simply requiring a positive empirical margin does not meet this condition.

## B.5 WHY THE DESIGN CHOICES REMAIN ABLATIONS

The central objects are the outcome coordinates, their text approximation, and their use in construction. Changing a probe subset, normalization, feature backbone, number of clusters, assignment rule, or search optimizer changes an implementation of those objects. The corresponding intervention should hold the remaining executor, score contract, split, and resource allocation fixed. Behavioral versus semantic or random construction needs the same operator language and proposal budget; seed-only versus searched experts needs the same assignments. Measured A versus text B changes information access, so it diagnoses amortization rather than supplying a deployable competitor. These controls follow directly from the quantity each proposition characterizes.

The design also separates construction cost from serving cost. Probing and expert generation are offline investments; a text-only assignment incurs local encoder computation and the chosen expert’s calls. An amortization advantage requires both a stated quality target and lower per-query serving cost than the comparator. Since guided B uses more calls than single-call CoT without an established quality improvement, this experiment does not justify a break-even claim against CoT.

## C EXPERIMENTAL PROTOCOL AND ADDITIONAL EVIDENCE

## C.1 DATA, SCORES, AND INFORMATION ACCESS

The mixed-task evaluation uses the repository’s persisted mixed\_hard\_xl pool with seed 13. Training contains 200 AIME, 200 GPQA, 100 CodeContests, and 100 LiveCodeBench queries; testing contains 100, 100, 50, and 50, respectively. The pool search procedure samples from the available source validation and test material, assigns stable source identifiers, and persists disjoint project training/test memberships. These are project splits, not a claim to reproduce the benchmarks’ official evaluation protocols. Affinity measurements use labels or executable tests offline. The encoder sees the query text; a deployed $\dot { B }$ route does not measure a test-query fingerprint. Outcome-assisted A routing does measure the test-query fingerprint, and its results include information unavailable to deployment.

AIME is scored by the existing mathematical answer checker. GPQA extracts the option following the last Answer: marker, falling back to a letter match if that marker is absent. Every scored September arm receives the appropriate answer-format instruction. For coding, the scorer extracts Python code, executes at most 12 available public stdin/stdout tests with a ten-second limit per test, normalizes output whitespace, and returns the fraction passed. Unsupported functional or interactive tests are skipped. Missing runnable tests score zero. Public-test success need not imply correctness on hidden tests, and the aggregate metric weights each query equally rather than each benchmark equally. The implementation’s scoring and loading files are included in the provenance manifest.

## C.2 HISTORICAL PROBE MEASUREMENTS AND CROSS-REPEAT ANALYSIS

The twelve mixed-task probe coordinates, in stored order, are direct, CoT, CoT self-consistency, tree-of-thought, self-refinement, reflexion, retrieval, ReAct, tool use, debate, role team, and decomposition. These labels name concrete repository implementations; they do not imply exact reproduction of every original paper’s system. Both historical model panels have 21,600 unique training cells and 10,800 unique test cells: 600/300 queries, twelve probes, and three repeats. The offline audit verified completeness of every query–probe–repeat tensor and used the recorded scores without rerunning models.

The retrieval probe uses same-benchmark TF–IDF demonstrations with available reference answers and excludes the current query. The current generator explicitly supplies the training corpus when invoked for test fingerprints. The saved sample ledgers retain answer previews rather than complete prompts or retrieved identifiers, so they do not independently certify the retrieval membership of every historical call. The September routed-probe arm selects no retrieval probe. For future releases, complete retrieval provenance should accompany the scorer and model revision, since conditioning on another evaluation query’s label would change the information contract.

For query $i ,$ probe $k ,$ and held-out repeat s, let

$$
a _ { i k } ^ { ( - s ) } = \frac 1 2 \sum _ { t \neq s } Y _ { i k t } , \qquad \widehat { k } _ { i } ^ { ( - s ) } \in \arg \operatorname* { m a x } _ { k } a _ { i k } ^ { ( - s ) } .
$$

The cross-repeat score is $3 ^ { - 1 } \sum _ { s } Y _ { i , \widehat { k } _ { i } ^ { ( - s ) } , s }$ . Training-priority ties use decreasing mean score across the 600 training queries, with stored probe order as a final deterministic tie rule. The uniform-tie analysis computes the exact mean held-out score across all tied maximizers, rather than drawing an additional random tie. The fixed probe is tool use for GPT-4o-mini and CoT self-consistency for Haiku. Its test score also averages the three repeats. The observed oracle instead takes the maximum of the three-repeat averages and hence reuses execution noise for selection and evaluation.

We freshly computed the cross-repeat confidence intervals from the saved ledgers using 10,000 benchmark-stratified bootstrap resamples, seed 13. Each resample draws queries with replacement within their benchmark and preserves the observed benchmark counts. The three overlapping heldout-repeat folds are averaged per query before resampling; they are not treated as independent ob servations. These intervals describe variability across this query mixture conditional on the saved training comparator and recorded executions. They do not incorporate retraining, changed mode revisions, or uncertainty from choosing the probe panel. In particular, the Haiku training-priority improvement is 0.0343 with interval [0.0110, 0.0592], while the uniform-tie interval includes zero. This difference makes tie handling part of the scientific result.

## C.3 ENCODER AND REPRESENTATION ABLATIONS

The expanded-panel study fits an MLP on frozen BGE features, using a 510/90 outer training/validation partition stratified by benchmark. Its checkpoint records nested inner-split epoch selection followed by refitting on outer-training rows, selecting 24 epochs. Normalization statistics are fit on the permitted training partition. The reported MSE averages over coordinates; converting it to the expected squared Euclidean error used in the analysis requires multiplication by the number of coordinates. The study’s 90-query validation metrics should not be confused with performance on the separate 300-query routing test set. Text truncation in the frozen backbone is a further information boundary: relevant information outside its retained input cannot be recovered by the MLP.

The paired eight/twelve-probe comparison preserves query membership and the eight shared measurements. Adding tree-of-thought, reflexion, ReAct, and role team increases the number of distinct measured fingerprints from 281 to 329 and effective rank from 2.6698 to 3.0819. On the 90 validation queries, the observed menu oracle increases from 0.4324 to 0.4503. This is increased descriptive resolution and observed menu coverage; because maxima are computed on the same measurements, it is not independent evidence of latent workflow sufficiency. Native cosine/MSE scores across different target widths are also not directly paired estimands. The repository contains two neighbor-recall conventions yielding different values; we omit a headline neighbor-recall claim pending reconciliation of tie handling and distance conventions.

Table 3 treats panel width and partition choice as ablations. Historical inference uses query-paired confidence intervals and sign-flip tests with Holm adjustment within the stored comparison families. The twelve-probe $B - D$ differences are 0.0200 ([−0.0150, 0.0558]) for GPT-4o-mini and $- 0 . 0 1 6 7 \left( \left[ - 0 . 0 6 6 1 , 0 . 0 3 3 \right] \right)$ for Haiku; both adjusted p-values equal one. For a separate executionattribution diagnostic, we hold the paper-native B routes fixed and replace each searched expert’s stored output score by the score of its training-selected raw probe. Searched-minus-probe differences are −0.0077 ([−0.0500, 0.0355]) and −0.1442 ([−0.1931, −0.0950]), respectively, using freshly computed stratified intervals. Probe scores average three executions, whereas the searched workflow scores follow their stored evaluation protocol. This identifies a possible loss at the execution interface; it is not a controlled test of a particular search algorithm.

Table 3: Historical mixed-task routing ablations. All values within a model column share that model’s July evaluation artifact. These are not same-harness comparisons to September guided generation.
<table><tr><td>Variant</td><td>GPT-4o-mini</td><td>Haiku 4.5</td></tr><tr><td>Global searched workflow (D)</td><td>.2722</td><td>.5381</td></tr><tr><td>Twelve-probe measured A</td><td>.2742</td><td>.6097</td></tr><tr><td>Twelve-probe predicted B</td><td>.2922</td><td>.5214</td></tr><tr><td>Eight-probe measured A</td><td>.2844</td><td>.6053</td></tr><tr><td>Eight-probe predicted B</td><td>.3072</td><td>.5083</td></tr><tr><td>Semantic BGE partition</td><td>.2939</td><td>.5303</td></tr><tr><td>Semantic Qwen3 partition</td><td>.3050</td><td>.5861</td></tr><tr><td>Random partition</td><td>.2767</td><td>.5228</td></tr></table>

Table 4: Six guided searches. Each has three rounds including initialization, and one execution per search-validation query per evaluated candidate. C3 is the only cluster with an accepted edit. C0/C4 and C2/C5 execute identical final specifications.
<table><tr><td>Cluster</td><td>Train</td><td>Search/local holdout</td><td>Top probe</td><td>Final behavior</td><td>Calls</td></tr><tr><td>CO</td><td>301</td><td>60/241</td><td>Self-refine</td><td>CoT + refinement</td><td>2</td></tr><tr><td>C1</td><td>83</td><td>60/23</td><td>Direct</td><td>Single CoT</td><td>1</td></tr><tr><td>C2</td><td>42</td><td>22/20</td><td>CoT-SC</td><td>Three samples + selection</td><td>4</td></tr><tr><td>C3</td><td>48</td><td>28/20</td><td>ReAct</td><td>Plan + four samples + selection</td><td>6</td></tr><tr><td>C4</td><td>82</td><td>60/22</td><td>Self-refine</td><td>CoT + refinement</td><td>2</td></tr><tr><td>C5</td><td>44</td><td>24/20</td><td>Tool use</td><td>Three samples + selection</td><td>4</td></tr></table>

## C.4 GUIDED CONSTRUCTION, SPLITS, AND EXECUTABLE SEMANTICS

In C3, the three-sample seed scores 0.2679 on the 28 search-validation queries. Increasing to four samples yields 0.4107; adding decomposition yields 0.4821. These are adaptive search scores, not test estimates. The final increment lies at the $2 / 2 8$ acceptance boundary in exact fractions and passes the recorded floating-point comparison.

The search uses an unordered set of JSON flags in a fixed plan–candidate–selection–refinement interpreter. Single candidates use temperature zero; multiple candidates use temperature 0.7. An optional persona branch adds two candidates. Debate and role-team flags share this branch; direct and CoT flags both retain the always-on CoT instruction. Reflexion maps to self-refinement and treeof-thought to a sampling ensemble. ReAct and tool use have no executor branch. Thus similarity of probe and operator names does not certify equality of their executed behavior.

Seed profiles use search-validation members rather than local holdout members. However, the optimizer’s marginal-value table computes its ALL column over all cluster members, including the local holdout. Furthermore, candidate proposals and acceptance reuse the same search-validation outcomes. Consequently neither the local holdout nor adaptively reused validation establishes the independent-selection assumptions of the conditional analysis. Only C1/C2 have stored localholdout executions; we do not use them as independent confirmation. The outer 300 test queries do not participate in these searches.

The deployed expert bank was constructed in the mixedhardxl12 clustering, while recorded paper routes use a separate paper\_mhxl12 clustering. A Hungarian bijection fitted solely to their 600 training assignments maps paper labels 0, 1, 2, 3, 4, 5 to expert labels 4, 2, 1, 3, 5, 0, with 517/600 agreements. Native versus translated measured test assignments agree on only 80.33% of queries. This bridge is part of the evaluated implementation and is not equivalent to training and deploying within one fixed cluster geometry.

Table 5: Stored September results on 300 queries and three repeats. SD is across the three pass means, not a standard error over queries. Costs are token-based estimates using the rates configured for those runs. Outcome-assisted A serving calls omit the cost of obtaining its test fingerprints.
<table><tr><td>Arm</td><td>Mean</td><td>Pass SD</td><td>Calls/query</td><td>Total USD</td></tr><tr><td>Direct</td><td>.2763</td><td>.0031</td><td>1.000</td><td>.2701</td></tr><tr><td>Single bare</td><td>.3121</td><td>.0083</td><td>1.000</td><td>.3159</td></tr><tr><td>Single CoT</td><td>.3295</td><td>.0180</td><td>1.000</td><td>.4702</td></tr><tr><td>Probe-menu-B</td><td>.2955</td><td>.0133</td><td>2.924</td><td>1.0924</td></tr><tr><td>Guided-B</td><td>.3358</td><td>.0097</td><td>2.113</td><td>.9919</td></tr><tr><td>Guided-A (diagnostic)</td><td>.3413</td><td>.0169</td><td>2.403</td><td>1.1613</td></tr></table>

## C.5 SEPTEMBER CONTROLLED COMPARISONS AND COST ACCOUNTING

The guided, direct, and routed-probe arms share the same 300 queries and official OpenAI endpoint. The single-call controls were executed subsequently on the same queries using the guided executor’s unmodified single-call implementation. Their comparison is paired by query, not by identical random seeds. Guided and single-call controls use a 3,000-token cap; the pipeline Direct and probemenu arms use 4,000 tokens and provider-default temperature. All use the scorer described above. Errors or empty final outputs score zero. None of these six arms has a recorded failed call or final-query error, but 40 guided calls were truncated (11 in A, 29 in B), as were nine calls in each single-call control. Truncated outputs remain scored as returned.

The displayed guided contrasts preserve the stored 10,000-resample percentile query-bootstrap procedure: average each query across its three executions, then resample queries without benchmark stratification. These exploratory intervals are unadjusted for multiplicity. They do not resample construction, encoder fitting, or provider drift. Guided-B minus probe-menu-B is 0.0404 ([0.0042, 0.0773]), but the arms differ in executor, prompting, decoding, and token cap; the contrast cannot isolate embedding guidance. Six construction runs total 2,544 calls and an estimated USD 1.2038 including the limited local-holdout arms. Offline probe generation and encoder fitting are additional costs, excluded from serving calls/query. Three separate repeats were verified from per-pass records even though the merged four-arm artifact retains its original two-repeat command arguments.

## C.6 EXPERT-BANK AND SOFTWARE-ENGINEERING BOUNDARY CHECKS

A separate single-execution 6 × 300 matrix evaluates every generated expert on every mixed-task test query. It yields guided-A/B readouts of 0.3397/0.3386, versus 0.3323 for the expectation of uniform expert selection. These values are a distinct execution sample from Table 5. Cluster-specific superiority is not established: no cluster’s own-versus-best-other interval excludes zero. The testselected best fixed expert scores 0.3567, but selection on the same test set makes it a diagnostic rather than a valid training-selected baseline. Counterfactual seed-family readouts cover only five of twelve probes and three seed families; they do not replace matched searches with guided versus unguided conditioning.

The historical SWE boundary study uses 500 SWE-bench Verified instances (Jimenez et al., 2024), a persisted repository-stratified 300/200 train/test split, and a single-turn Agentless-lite localize– repair executor. Twelve policy texts form the measured panel; ReAct, reflexion, and tool-use labels are single-turn analogues here. The metric is actual harness resolution, not the older file-localization proxy elsewhere in the repository. A per-policy logistic predictor replaces the mixed-task MLP, and small clusters merge from six to four. Predicted routing chooses direct on all 200 tests and scores 0.200, versus the training-selected global retrieval policy’s 0.175; the paired interval for the difference is [−0.020, 0.070]. All failures are retained as unresolved. The 6,000-cell grid contains 1,565 errored cells, including 210 infrastructure/no-report residues. This is a transfer stress test with changed executor and predictor, not replication of the guided generator.

## C.7 BASELINE DEFINITIONS AND ADAPTATIONS

We compare published mechanisms after adapting them to the project’s query pools, executors, answer formats, and scorers. The adaptation names in Table 1 specify the evaluated systems. They do not stand for the original papers’ full configurations or reported benchmark scores. Query-level error handling is fail-closed: unsuccessful execution contributes zero.

Global and cluster-conditioned AFlow. The mixed-task bank uses AFlow’s workflow search (Zhang et al., 2025b), with Custom and ScEnsemble operators, five optimization rounds after initialization, parent sample size four, one validation execution per query, and the executor itself as optimizer. The published default has a longer search and a stronger optimizer. All six cluster partitions together contain the same 600 training queries evaluated by the undivided D search, matching nominal validation-query allocation per round. Six searches nevertheless require six proposal streams rather than one and can incur different token costs; aggregate offline budgets are not equal. Semantic BGE, Qwen3, and random partitions use this same search protocol. The SWE bank selects among twelve policy texts on each cluster; its D is the global training-selected retrieval policy. The distinct SWE AFlow baseline searches a policy on the shared, repository-stratified 150-instance training subset, retaining AFlow’s parent-selection rule and one edit per round. Its five-round winner resolves 38/200 test instances.

ADAS. Our ADAS adaptation (Hu et al., 2025) retains archive-conditioned proposal, reflection/debugging, and bootstrap fitness selection. Its meta-model is GPT-4o-mini or Haiku, matching the executor rather than the original GPT-4-class meta-agent. On mixed tasks it searches agent code; on SWE it searches policy text for the fixed pipeline, from direct/CoT/ReAct-policy seeds over five generations. SWE’s winner is the CoT seed (40/200 test resolutions). For Haiku, CPU contention affected the original coding fitness evaluation. We therefore use the clean re-score of all eight archived candidates on the full training pool, selecting Self-Refine by clean training fitness before evaluating on the 300 test queries. Its 0.2489 replaces the obsolete contended score of 0.4075.

ScoreFlow. The adaptation implements SF-0 candidate generation and an in-context preferenceoptimization (ICPO) step (Wang et al., 2025); it does not perform Score-DPO weight updates. GPT-4o-mini replaces the fine-tuned workflow generator. Mixed-task inference generates four candidates and selects without test-score feedback; its test utility is 0.2756. A separate training-selected majority-workflow control gives 0.2758. On SWE, twelve policy candidates are scored on the common 150 training instances. Preference pairs use the published cubic score-gap weighting; one ICPO refinement is compared against the best SF-0 candidate. SF-0 scores 0.1733 on training versus ICPO’s 0.1533, and its selected policy resolves 42/200 test instances. The Haiku ScoreFlow configuration was not run.

FlowReasoner. Our search-only FlowReasoner adaptation (Gao et al., 2025) omits supervised/reinforcement learning of the meta-reasoner and uses the executor model for proposals. Mixed-task runs search separately for each evaluated query, using its scorer feedback to choose a workflow before three fresh executions of that same query. They are therefore outcome-assisted diagnostics, marked †, rather than text-only deployment results. The operator space includes Programmer. GPT-4o-mini completes 297/300 queries; its saved completed-only mean is 0.3601. Table 1 uses 0.3565 after assigning zero to the three missing queries, retaining the shared 300-query denominator. Haiku completes all 300 and scores 0.7465. Logged costs average USD 0.00660 and 0.21060 per completed query, with mean wall time 284.2 and 268.7 seconds. Missing-query costs are not fully reconstructed. On SWE, the adaptation instead searches one pool-level policy on the disjoint 150-instance training subset; the stock-pipeline seed wins and resolves 41/200 tests. That SWE result does not use test-instance scorer feedback during selection.

Policy routing on SWE. MasRouter (Yue et al., 2025) instantiates the cost-aware routing objective over the twelve fixed Agentless policies, using its sampled-selection protocol, five training epochs, and cost rate 200. It resolves 34/200 instances at USD 0.00227 per instance. FrugalGPT (Chen et al., 2024) trains issue-text scorers and selects a cascade on training data. At budget USD 0.004041 it selects direct → debate → retrieval; test utility is 0.2050 at USD 0.00240. Only 21/200 queries escalate beyond direct. Both methods have access to the same stored policy outcomes for training;

Table 6: Test means of the twelve probe coordinates. Mixed-task entries average three executions. SWE entries are single-turn policy analogues under the Agentless pipeline, not implementations of their multi-turn namesakes. ‡ marks mixed-task probes using execution feedback, an information advantage over the original Custom/ScEnsemble search space.
<table><tr><td>Probe</td><td>Mixed / GPT-4o-mini</td><td>Mixed / Haiku</td><td>SWE</td></tr><tr><td>Direct</td><td>.2872</td><td>.5694</td><td>.200</td></tr><tr><td>CoT</td><td>.2852</td><td>.5953</td><td>.205</td></tr><tr><td>Self-consistency</td><td>.3260</td><td>.6894</td><td>.190</td></tr><tr><td>Tree-of-thought</td><td>.2911</td><td>.5919</td><td>.200</td></tr><tr><td>Self-refinement</td><td>.2993</td><td>.6814</td><td>.145</td></tr><tr><td>Reflexion</td><td>.3340</td><td>.6656</td><td>.125</td></tr><tr><td>Retrieval</td><td>.2919</td><td>.6184</td><td>.175</td></tr><tr><td> $\mathrm { R e A c t } ^ { \ddag }$ </td><td>.2552</td><td>.5137</td><td>.165</td></tr><tr><td> ${ \mathrm { T o o l ~ u s e } } ^ { \ddagger }$ </td><td>.3481</td><td>.6100</td><td>.150</td></tr><tr><td> $_ { \mathrm { D e b a t e } }$ </td><td>.3069</td><td>.6814</td><td>.180</td></tr><tr><td>Role team</td><td>.3091</td><td>.6483</td><td>.210</td></tr><tr><td>Decomposition</td><td>.3254</td><td>.6343</td><td>.055</td></tr></table>

Table 7: Repeated-execution evidence for the affinity panel on 300 test queries. The observed oracle selects and evaluates using the same three-repeat mean. Cross-repeat selection evaluates a held-out execution; both selectors require test-query outcomes. Uniform ties average over tied maximizing probes.
<table><tr><td>Model</td><td>Fixed</td><td>Observed oracle</td><td>Cross-repeat</td><td>Uniform ties</td></tr><tr><td>GPT-4o-mini</td><td>.3481</td><td>.5580</td><td>.4059</td><td>.3862</td></tr><tr><td>Haiku 4.5</td><td>.6894</td><td>.8154</td><td>.7237</td><td>.6909</td></tr></table>

per-policy test costs are similar, restricting the available cost-quality trade-off. The higher-budget FrugalGPT configuration selects the identical cascade.

## C.8 THE FULL PROBE PANEL AND REPEATED-EXECUTION EVIDENCE

Probe rankings change with the executor and pipeline: the training-selected GPT-4o-mini mixedtask probe is tool use, Haiku selects self-consistency, and SWE selects retrieval. The mixed/SWE training-grid rank correlation is Spearman $\rho = - 0 . 4 4 9 \left( p = 0 . 1 4 3 \right.$ , twelve coordinates); excluding decomposition yields $\rho \approx - 0 . 5 3 0$ . Decomposition frequently conflicts with SWE localization formatting, so this comparison combines reasoning, format, and executor effects. We find no evidence supporting direct transfer of one regime’s probe ranking to another.

## C.9 OPERATOR AND PANEL ABLATIONS

Adding executable code is motivated by the gap between the strongest measured coding probes and the original expert operator set. Twelve-probe B reaches 0.3153, but random partitions also improve and eight-probe B decreases. Because each bank is searched anew, the experiment changes both operator availability and realized search trajectories; it cannot isolate an operator effect from search variation. Table 3 reports the independent eight/twelve-panel comparisons. The higher-dimensional panel improves descriptive resolution, while downstream utility depends jointly on measurement, assignment, and the learned expert bank.

## C.10 LABEL-FREE OUTCOMES AND ADDITIONAL DIAGNOSTIC CONTROLS

A post-hoc alternative estimates affinities from within-probe agreement and cross-probe answer consensus, with calibration fitted only on training queries. The saved mixed-task ledger covers only the experts actually executed by A or B, so alternative routes can be scored only where they select one of those recorded experts. On GPT-4o-mini’s eight-probe agreement variant, coverage is 201/300 and the covered-subset gain over the same subset of D is 0.0402 (95% CI [−0.005, 0.086], ex ploratory p = 0.090). Haiku’s twelve-probe agreement variant covers 279/300 and changes utility by −0.0143 ([−0.068, 0.039]), while measured affinities retain a 0.0806 advantage on that subset. Full-pool lower/upper bounds for uncovered outcomes are [0.199, 0.529] and [0.498, 0.568], respectively. These are coverage-restricted diagnostic results, not full-test estimates.

Table 8: GPT-4o-mini operator ablation: 31 fresh searches add Programmer while preserving clusterings, routing, and search caps. Six BGE executions remain failed in the saved artifact and are counted as zero; this row is .2861 on the full denominator rather than a completed-only mean.
<table><tr><td>Variant</td><td>Custom + ScEnsemble</td><td>+ Programmer</td><td>Difference</td></tr><tr><td>Twelve-probe B</td><td>.2922</td><td>.3153</td><td>+.0231</td></tr><tr><td>Twelve-probe A</td><td>.2742</td><td>.2922</td><td>+.0181</td></tr><tr><td>Eight-probe B</td><td>.3072</td><td>.2742</td><td>-.0331</td></tr><tr><td>Eight-probe A</td><td>.2844</td><td>.2814</td><td>-.0031</td></tr><tr><td>Semantic Qwen3</td><td>.3050</td><td>.3064</td><td>+.0014</td></tr><tr><td>Semantic BGE</td><td>.2939</td><td>.2861</td><td>-.0078</td></tr><tr><td>Random</td><td>.2767</td><td>.3042</td><td>+.0275</td></tr><tr><td>Global D</td><td>.2722</td><td>.2783</td><td>+.0061</td></tr></table>

Table 9: Separate four-model-pool study on the 300 mixed-task test queries. Costs are logged token usage at the configured prices. The local quantized Llama has zero API charge; GPU cost is excluded. These methods choose among different backbones and are not executor-matched comparisons to Table 1.
<table><tr><td>Method / operating point</td><td>Utility</td><td>USD/query</td></tr><tr><td>Single GPT-4o-mini</td><td>.2969</td><td>.000358</td></tr><tr><td>Single Haiku 4.5</td><td>.5994</td><td>.005353</td></tr><tr><td>Single Gemini Flash</td><td>.5533</td><td>.006978</td></tr><tr><td>Single Llama-3.1-8B (AWQ-INT4)</td><td>.1886</td><td>.000000</td></tr><tr><td>MasRouter, default cost rate</td><td>.2664</td><td>.002012</td></tr><tr><td>FrugalGPT, low budget</td><td>.3969</td><td>.001636</td></tr><tr><td>FrugalGPT, medium budget</td><td>.4839</td><td>.003213</td></tr><tr><td>FrugalGPT, high budget</td><td>.6228</td><td>.006818</td></tr><tr><td>Observed per-query pool oracle</td><td>.769</td><td>.00256</td></tr></table>

The stored answer previews contain only the first 1,200 characters; approximately 82% of Haiku generations are clipped. The analysis therefore evaluates the available preview features, not what complete generations could support. A related SWE label-free analysis uses the retired ReAct regime and yields 0.085 versus D’s 0.090; it is not an Agentless result. The retired ReAct direct score is 0.088 on its clean eight-policy grid and the FlowReasoner transfer attempt scores 0.060. We exclude the compromised twelve-policy ReAct scoring-churn results from the principal comparisons.

For the guided bank, evaluating all six experts on all 300 queries produces expert means 0.3197, 0.3236, 0.3567, 0.3403, 0.3233, 0.3303. The observed per-query maximum is 0.4961, although 194 queries give identical scores for all six executions. These maxima select on their own evaluation outcomes and hence quantify observed opportunity rather than an achievable text-only policy. The single-call ladder (Direct 0.2763, bare 0.3121, CoT 0.3295, guided-A 0.3413) further separates prompt/decoder changes from added workflow computation. Differences between arms are interpreted as complete-configuration contrasts, not a causal decomposition of the embedding’s contribution.

## C.11 HETEROGENEOUS-MODEL ROUTING AND COST-QUALITY COMPARISONS

The pool contains GPT-4o-mini, Haiku 4.5, Gemini Flash, and locally served Llama-3.1-8B AWQ-INT4. These replace unavailable original-paper model versions, and DeepSeek is omitted. Mas-Router uses its cost-aware policy-gradient objective, learning rate 0.01, batch size 16, five epochs, cost rate 200, and seed 1234. Its selections favor cheaper backbones (142/300 Llama slots); all 18 test errors are scored zero. FrugalGPT replaces exact-match scoring with our mixed-task utility and adapts token limits/formatting to these tasks. Its low/medium budgets select GPT-4o-mini → Llama → Haiku; the highest budget selects Haiku → Llama → Gemini. That cascade attains 0.6228, above its single-Haiku reference 0.5994, at a higher cost. The pool oracle chooses the cheapest model attaining each query’s observed maximum and is not executable without outcomes.

Measurement, construction, and serving costs. The full Haiku probe program (eight/twelve probes, training/test) records 115,330 calls and USD 603.90. The SWE 6,000-cell grid records approximately USD 14.02, with 2.25 calls and USD 0.0023 per instance–policy pair. Historical GPT-4o-mini probe/search tokens were not comprehensively metered. Under the Haiku program’s average sample cost, a twelve-probe, three-repeat test fingerprint costs approximately USD 0.4026 before executing its selected expert; the SWE twelve-policy fingerprint costs approximately USD 0.0276. The corresponding GPT-4o-mini estimate is roughly USD 0.028, derived from the Haiku calls-per-sample ratio and the separately metered direct-call price, rather than measured on those grids. This cost belongs to diagnostic A only; B uses one frozen encoder forward pass and no routing-time LLM calls. Guided construction’s separately metered USD 1.2038 excludes the earlier probe program. Table 5 gives its serving costs. These distinctions make offline investment, outcome-assisted evaluation, and deployable serving directly interpretable.

## C.12 STATISTICAL FAMILIES AND SCOPE OF BASELINE COMPARISONS

For each historical mixed-task backbone, the stored Holm families contain seven comparisons against D, seven between routing variants, and five against Direct/train-selected probes. The seven D comparisons are twelve/eight-probe $A / B ,$ , BGE, Qwen3, and random. The seven routing comparisons are eight-probe B versus Qwen3, BGE, twelve-probe B, eight-probe A, and random; Qwen3 versus BGE; and twelve-probe B versus twelve-probe A. The five probe comparisons are twelve/eight-probe B versus Direct and the training-selected probe, plus D versus Direct. SWE has separate four-comparison A and B families against D, Direct, BGE, and random. The prespecified primary comparison in each regime is twelve-probe $B - D ;$ none is significant. We retain this definition while presenting the representation analyses as additional findings. The SWE $A - D$ result also has exact McNemar $p = 0 . 0 0 1 3 1 2$ (16 corrected/two broken). Main-table ordering against ADAS, ScoreFlow, FlowReasoner, MasRouter, and FrugalGPT reports point estimates; it does not imply unreported significant pairwise differences. Guided contrasts and new diagnostic analyses remain exploratory and unadjusted.

## C.13 REPRODUCIBILITY AND OUTSTANDING ATTRIBUTION

The accompanying evidence manifests record source paths, SHA-256 hashes, query memberships, per-query cross-repeat scores, recomputed means, and statistical conventions. The original evidence audit and expanded baseline audit perform no API calls and leave experimental artifacts unchanged; the latter additionally checks all reported adapted-search means against their saved per-query records and records corrected failure denominators. Historical panel size, semantic/random partitions, and decoder/executor controls are reported as ablations. A matched full guided-versus-unguided construction experiment, independent final candidate selection, capability-complete probe-to-operator mapping, and held-out margin-stratified encoder analysis are not present in the saved evidence. We therefore report no numerical outcomes for these missing interventions and do not infer their success from the conditional bounds.