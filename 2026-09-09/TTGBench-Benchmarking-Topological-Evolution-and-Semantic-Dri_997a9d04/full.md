# TTGBench: Benchmarking Topological Evolution and Semantic Drift in Text-attributed Temporal Graphs

Longfei Ma<sup>1</sup>, Zemin Liu<sup>1∗</sup>, Fei Wu<sup>1∗</sup>

<sup>1</sup>Zhejiang University

{longfeima, liu.zemin, wufei}@zju.edu.cn,

## Abstract

Temporal graph learning models the evolution of dynamic systems, where both structural interactions and semantic states change over time. However, existing benchmarks primarily emphasize structural evolution via temporal link prediction (TLP), while support for semantic evolution remains limited. Although temporal node classification (TNC) is sometimes included, it is typically restricted to simplistic binary settings that fail to capture realistic semantic drift. Moreover, commonly used datasets exhibit high link repetition, leading to inflated performance estimates and obscuring true model capability. To address these limitations, we introduce TTGBench, a new benchmark that jointly evaluates structural and semantic evolution. TTGBench comprises six real-world, text-rich datasets characterized by Dual Volatility, enabling rigorous and fair evaluation of existing models. Notably, it is the first benchmark to support both multi-class and multi-label TNC, filling a critical gap in evaluating temporal semantic drift. We conduct a comprehensive evaluation of 17 state-of-the-art methods across Temporal Graph Neural Networks (TGNNs) and Large Language Model (LLM)-based paradigms. The results reveal a clear capability divide between the two paradigms: TGNN-based methods excel at structural prediction but fail at semantic tracking, whereas LLM-based predictors show the opposite trend. Through in-depth analysis, we uncover their fundamental limitations and provide insights for developing more comprehensive temporal graph models.

## 1 Introduction

Temporal graphs provide a powerful framework for modeling dynamic systems such as social networks[43, 5, 22], e-commerce platforms[6, 45], and financial transaction networks[46, 33]. At their core, these systems evolve along two fundamental dimensions: structural evolution and semantic drift. Structural evolution describes how nodes form and dissolve connections over time, typically modeled as temporal linkprediction (TLP). In parallel, semantic drift captures how node labels or role change over time—such as shifts in user interests—and is commonly formulated as temporal node classification (TNC). A comprehensive understanding of both dimensions is essential for uncovering the underlying dynamics of temporal graphs. Therefore, a benchmark that jointly evaluates TLP and TNC is crucial for assessing whether models can capture the full spectrum of temporal evolution.

Existing benchmarks have significantly advanced temporal graph learning by curating diverse datasets (e.g., DGB [24], TGB [12]), introducing text-attributed graphs (DTGB [40]), and establishing unified evaluation pipelines (DyGLib [39]). Despite these contributions, two fundamental limitations remain. First, most benchmarks focus almost exclusively on structural evolution through TLP, while largely overlooking semantic drift. Although DyGLib supports TNC, it is restricted to simple binary classification tasks (e.g., whether a user is banned), which fail to capture meaningful and continuous

(b) Semantic Volatility

semantic drift. This limitation largely stems from the scarcity of datasets with rich, dynamically evolving semantic labels, as noted in prior work [14, 17]. Second, existing datasets often contain a high proportion of repeated edges, leading to overestimated model performance. In fact, many methods achieve near-saturated results (e.g., over 98% on widely used benchmarks [39, 40]), which obscures true model capability and fails to reflect realistic, high-novelty environments—a concern also highlighted by recent study [38].

To address these limitations, we introduce the Text-attributed Temporal Graph Benchmark (TTGBench), a new benchmark that jointly evaluates structural and semantic evolution. TTG-Bench comprises six real-world datasets spanning diverse domains and is characterized by a novel property we term Dual Volatility (Figure 1): (i) Structural Volatility, defined by high link novelty, limited repetition, and highly dynamic interaction patterns; and (ii) Semantic Volatility, where node labels evolve frequently with complex, non-trivial dynamics. These properties make TTGBench substantially more challenging than existing benchmarks. Under high structural volatility, memorization-based methods such as EdgeBank [24] fail due to the scarcity of repeated interactions, as confirmed in Section 6. Beyond global novelty, our datasets exhibit intricate structural evolution patterns (Section 4.1), requiring models to capture fine-

![](images/8ddb7cf29af5bcd330d205a2cada64d300dace93d45361d49094614024271433.jpg)

![](images/622ff8511bd7c3628ffb1d003ccb44313494c5d27592d4bd3020a5e35d7c4e07.jpg)  
Figure 1: Illustration of dual volatility in TTG-Bench: structural and semantic volatility. (a) The proposed datasets exhibit high novelty, where previously unseen edges continuously emerge over time, leading to pronounced structural volatility. (b) A substantial proportion of nodes change their labels over time, highlighting the presence of semantic volatility in temporal graphs.

grained temporal dependencies. On the semantic side, TTGBench initially supports both multi-class and multi-label TNC, enabling realistic and diverse semantic scenarios with complex label transitions (Section 4.2). Furthermore, all datasets include rich textual attributes, providing essential context for modeling both structural and semantic dynamics.

Building on TTGBench, we conduct a comprehensive evaluation of 17 state-of-the-art methods spanning different paradigms, including dominant TGNNs and recent LLM-based approaches. We categorize these methods into two groups based on the predictor: TGNN-Predictors, which include both pure TGNNs and LLM-as-Enhancer methods, and LLM-Predictors, which directly use LLMs for prediction. Our evaluation reveals a striking capability divide: TGNN-Predictors excel at TLP but usually fail on TNC, whereas LLM-Predictors perform strongly on TNC but struggle with TLP. This dichotomy exposes a fundamental limitation—neither paradigm can simultaneously model structural and semantic evolution. Our empirical results further reveal inherent limitations in how LLM-Predictors encode structural information. In addition, we conduct in-depth analyses to diagnose the failure of TGNNs on semantic tracking, identifying three intrinsic architectural bottlenecks that hinder their ability to utilize semantic signals. Finally, we perform a comprehensive efficiency analysis, providing practical insights into the deployment trade-offs of these methods. Collectively, our findings offer a deeper understanding of current approaches and provide guidance for designing future models that unify structural and semantic learning in temporal graphs.

We summarize our contributions as follows:

• A unified benchmark: We introduce TTGBench, the first benchmark that jointly evaluates structural evolution and semantic tracking. It comprises six real-world, text-attributed datasets exhibiting Dual Volatility, enabling a more rigorous and holistic assessment of existing models.

• Comprehensive empirical evaluation: We conduct a large-scale evaluation of 17 methods spanning both TGNN-Predictors and LLM-Predictors. Our results reveal a clear capability divide: TGNN-Predictors excel at TLP but struggle with TNC, whereas LLM-Predictors show the opposite trend, exposing a fundamental limitation of current paradigms.

• Diagnostic empirical insights: We provide in-depth analyses to uncover the root causes of these limitations, including structural deficiencies in LLM-based predictors and architectural bottlenecks in TGNNs for semantic tracking, along with their performance and efficiency trade-offs. These findings offer actionable insights for developing more effective temporal graph models.

## 2 Related Work

Temporal Graph Learning. Temporal graph learning has attracted increasing attention due to its strong ability to model dynamic real-world systems [15, 28]. Among existing approaches, TGNNs have emerged as the dominant paradigm owing to their expressive power [17]. In particular, continuous-time temporal graphs, which model interactions with arbitrary timestamps, provide greater flexibility and have enabled many state-of-the-art methods [16, 34, 37, 4, 39]. Despite this progress, most studies focus primarily on temporal link prediction. To obtain a more comprehensive understanding of model capabilities, we additionally evaluate these methods on multi-class and multi-label temporal node classification.

Temporal Graph Benchmarks. Existing benchmarks have substantially advanced temporal graph research by providing diverse datasets and standardized evaluation protocols. DGB [24], DyGLib [39], TGB [12], and DGraph [13] offer unified pipelines for training and evaluation, while DTGB [40] introduces text-attributed temporal graph datasets. TGB-Seq [38] further identifies excessive edge repetition in prior benchmarks and proposes datasets with higher novelty. However, these benchmarks mainly emphasize link-level tasks and, at the node level, are largely limited to binary classification, overlooking more realistic multi-class scenarios. In contrast, our benchmark introduces text-rich temporal graph datasets with high link novelty and supports both multi-class and multi-label node classification within a unified framework.

LLMs for Temporal Graph Learning. With the rapid advancement of large language models (LLMs), recent work has begun exploring their use in graph learning. LLM-as-Enhancer methods [27, 41] integrate LLMs to improve temporal graph models, while TGTalker [11] employs LLMs as predictors via in-context learning. However, most LLM-based graph learning studies focus on static graphs. To extend LLMs to temporal settings, we adapt recent LLM-as-Predictor methods originally designed for static graphs [3, 29] to temporal graphs, enabling empirical evaluation of their ability to directly model both structural and semantic dynamics.

## 3 Task Formulation

To comprehensively evaluate model capabilities in capturing both topological and semantic dynamics, TTGBench standardizes two fundamental tasks. We begin by formally defining the data structure.

Definition 1: Text-Attributed Temporal Graph. A text-attributed temporal graph is defined as $\mathcal { G } = ( \mathcal { V } , \mathcal { E } , \mathcal { T } , \mathcal { X } , \mathcal { V } )$ , where $\nu$ denotes the set of nodes and E is a chronologically ordered sequence of timestamped interactions. Each interaction is represented as an event $e = ( u , v , t , x _ { e } ) \in \mathcal { E }$ indicating that nodes $u , v \in \mathcal { V }$ interact at time $t \in \mathcal T$ , accompanied by an edge-level textual attribute $x _ { e } .$ . In addition, each node u is associated with a time-dependent semantic label $y _ { u } ( t ) \in \mathcal { y }$ , reflecting its instantaneous preference.

Based on this formulation, TTGBench defines two continuous-time tasks to evaluate a model’s ability to capture structural evolution and semantic dynamics.

Temporal Link Prediction (TLP): Modeling Structural Evolution. TLP evaluates a model’s ability to predict future interactions based on historical graph observations. Given the graph up to time $t ,$ denoted as $\mathcal { G } ( \leq t )$ , the objective is to predict whether an edge $( u , v )$ will occur at a future time $t ^ { \prime } > t ,$ i.e., whether $( u , \dot { v } , t ^ { \prime } ) \in \dot { \mathcal { E } }$ . Formally:

$$
P ( e = ( u , v , t ^ { \prime } ) \mid \mathcal { G } ( \leq t ) ) = \sigma \left( f _ { \theta } ^ { \mathrm { ( l i n k ) } } ( u , v , \mathcal { G } _ { \leq t } ) \right) ,\tag{1}
$$

where $\sigma ( \cdot )$ is the sigmoid function mapping model outputs to link probabilities.

Temporal Node Classification (TNC): Modeling Semantic Evolution. In contrast to prior benchmarks that restrict node classification to simplistic, quasi-static binary tasks, TTGBench formulates TNC as a dynamic semantic prediction problem. As nodes interact over time, their semantic labels $y _ { u } ( t )$ evolve continuously. Given the historical graph observed up to time $t ,$ the goal is to predict node u’s instantaneous semantic label $y _ { u } ( t ^ { \prime } )$ at a future time $t ^ { \prime } > t .$

Importantly, TTGBench supports both multi-class and multi-label settings, enabling the modeling of complex and realistic semantic dynamics. The task is defined as:

$$
P ( y _ { u } ( t ^ { \prime } ) = c \mid u , \mathcal { G } \le t ) = \mathrm { S o f t m a x } \left( f _ { \theta } ^ { ( \mathrm { n o d e } ) } ( u , \mathcal { G } \le t ) \right) _ { c } \quad \mathrm { ( m u l t i - c l a s s ) , }\tag{2}
$$

Table 1: Summary statistics of TTGBench datasets. C denotes node class count.
<table><tr><td>Dataset</td><td># Nodes</td><td># Edges</td><td># Steps</td><td>Novelty</td><td># C</td><td>Tasks</td></tr><tr><td>FOOD</td><td>33,773</td><td>401,057</td><td>6,252</td><td>1.0</td><td>9</td><td>TLP &amp; multi-label TNC</td></tr><tr><td>IMDB</td><td>32,371</td><td>310,891</td><td>7,099</td><td>1.0</td><td>8</td><td>TLP &amp; multi-label TNC</td></tr><tr><td>Librarything</td><td>51,201</td><td>785,690</td><td>2,914</td><td>1.0</td><td>-</td><td>TLP</td></tr><tr><td>Beeradvocate</td><td>99,361</td><td>1,586,573</td><td>1,577,920</td><td>0.99</td><td>7</td><td>TLP &amp; multi-class TNC</td></tr><tr><td>Ratebeer</td><td>139,538</td><td>2,924,105</td><td>4,254</td><td>1.0</td><td>8</td><td>TLP &amp; multi-class TNC</td></tr><tr><td>Amazon-Kindle</td><td>215,504</td><td>5,621,343</td><td>5,484,604</td><td>1.0</td><td>8</td><td>TLP &amp; multi-class TNC</td></tr></table>

$$
P ( \mathbf { Y } _ { u , c } ( t ^ { \prime } ) = 1 \mid u , \mathcal { G } \leq t ) = \sigma \left( f _ { \theta , c } ^ { ( \mathrm { n o d e } ) } ( u , \mathcal { G } \leq t ) \right) \quad ( \mathrm { m u l t i - l a b e l } ) ,\tag{3}
$$

where $\mathbf { Y } _ { u } ( t ^ { \prime } ) \in \{ 0 , 1 \} ^ { | c | }$ is the ground-truth label vector, C denotes the label set, and $c \in { \mathcal { C } }$

## 4 The Proposed Datasets: Unveiling Dual Volatility

To rigorously evaluate the capabilities of temporal graph learning models in highly dynamic environments, we construct six real-world, text-attributed temporal graph datasets spanning diverse domains, including culinary recipe feedback [18], movie reviews [21], book reading records [2, 44], beer rating data [19, 20], and online shopping interactions [10]. Key statistics are summarized in Table 1, with additional details provided in Appendix A.

Our datasets are fundamentally more challenging than existing benchmarks, as they exhibit a property we term Dual Volatility: (i) Structural Volatility, characterized by continuous and drastic reshaping of network topology, and (ii) Semantic Volatility, characterized by pervasive and complex semantic drift. Beyond the intuitive description introduced in Section 1, we provide a fine-grained, and micro-level analysis of these properties in this section.

## 4.1 Structural Volatility

We analyze structural dynamics from two complementary perspectives: continuous density variation and traffic concentration instability, as illustrated in Figure 2 (a) and (b).

Continuous Density Variation. Rather than remaining stable or growing uniformly, interaction volumes exhibit persistent and significant fluctuations over time. This continuous variation indicates that the temporal graphs undergo repeated phases of expansion and contraction. As a result, models must maintain strong temporal adaptability, avoiding overfitting to dense intervals while remaining robust during sparse periods.

Traffic Concentration Instability. Figure 2 (b) shows that the top 5% most active nodes (hubs) exhibit highly irregular and volatile activity patterns. The graphs continuously shift between centralized regimes—where a small number of nodes dominate interactions—and decentralized regimes with more evenly distributed activity. In other words, the “centers of gravity” of the network are constantly shifting. This instability causes structural patterns learned in one time window to quickly become obsolete, requiring models to adapt to highly non-stationary degree distributions.

## 4.2 Semantic Volatility

A key distinction between temporal and static graphs lies in the dynamic nature of node semantics: node labels can evolve over time. We next examine the semantic dynamics of our datasets in detail.

High-Frequency Semantic Drifts. Figure 2 (c) shows that frequent semantic drifts are a common phenomenon across datasets, although their intensity varies by domain. In datasets such as FOOD and IMDB, the drift ratio remains consistently high (often above 0.8), indicating rapid and continuous changes in user preferences. In contrast, Amazon-Kindle exhibits a more moderate yet still substantial drift ratio (around 0.3), reflecting longer periods of semantic consistency. This variation aligns with domain-specific behaviors: while dining and movie consumption are fast-paced, reading typically involves sustained engagement, leading to slower semantic transitions. Importantly, this spectrum—from highly dynamic to moderately stable—demonstrates the diversity of semantic evolution patterns in our datasets, enabling a comprehensive evaluation of models under varying degrees of semantic volatility.

![](images/1e62a83639e35e40ec051806e659006895c3a19bbe7d79ab2649525e84160a5a.jpg)

![](images/ddee79b7011da70fd1e3353cb87f832ffc85d7b5020d1f61a4cd1c9443c1d630.jpg)

![](images/d2229e2cb67372d695b31bd4c8e4a21df2e0df20cd59b57a8a401e6de9ccf128.jpg)

![](images/56ecf31dcc797587e3db58ebd166264264b06b0770e5a3d706056c862c66c118.jpg)

![](images/bad9a78902e2dd0d76be48e03a8067848848389756989cf4bbb1682934e11abe.jpg)  
Figure 2: Micro-level visualization of dual volatility across the proposed datasets. (a) Interaction Density Over Time: Shows the temporal evolution of interaction density, where higher values indicate more frequent and concentrated interactions at a given time. (b) Top 5% Hub Traffic Over Time: Illustrates the temporal dynamics of interactions involving the top 5% most active nodes (hubs). (c) Label Drift Ratio Over Time: Depicts the proportion of nodes whose labels change over time. (d) Transition Matrices: Visualizes label transitions on two representative datasets, RateBeer and Amazon-Kindle, where the vertical axis denotes node labels at the previous time step, and the horizontal axis denotes labels at the subsequent time step.

Non-Trivial Evolutionary Trajectories. Figure 2 (d) illustrates label transitions for the same node across consecutive time steps. The resulting heatmaps exhibit asymmetric, high-density clusters rather than uniform or trivially dominant patterns. This indicates that semantic evolution involves abrupt and heterogeneous transitions across distinct classes. Consequently, future states cannot be reliably inferred by simply copying the most recent label or applying a static global transition pattern, highlighting the intrinsic complexity of semantic drift in temporal graphs.

## 5 Evaluated Baselines

To systematically evaluate model performance on the proposed Dual Volatility datasets across both topological evolution (TLP) and semantic drift (TNC), we benchmark 17 representative algorithms spanning multiple modeling paradigms. Based on the type of predictor, we categorize these methods into two groups: TGNN-Predictors and LLM-Predictors.

## 5.1 TGNN-Predictors

This paradigm relies on Temporal Graph Neural Networks (TGNNs) as the predictive backbone, operating directly on graph structures via message passing or temporal aggregation. Most existing methods specifically designed for temporal graph learning fall into this category, including both purely structural TGNNs and hybrid approaches that incorporate LLMs as auxiliary components.

Pure TGNNs. As the dominant paradigm in temporal graph learning, pure TGNNs serve as strong structural baselines. We evaluate nine state-of-the-art models covering the major mechanisms for modeling topological evolution. These include memory-based continuous-time models (JODIE [16], DyRep [34], TGN [26]), attention-based temporal aggregation methods (TGAT [37], DyGFormer [39],

FreeDyG [32]), and walk-based or hybrid message-fusing approaches (CAWN [36], TCL [35], GraphMixer [4]). These models primarily rely on structural patterns and temporal interaction frequencies to make predictions.

LLM-as-Enhancer. This hybrid subcategory leverages frozen LLMs to enrich semantic representations while retaining TGNNs as the final predictors. We evaluate two recent state-of-the-art methods: LKD4DyTAG [27] and CROSS [41]. LKD4DyTAG prompts an LLM to extract semantic embeddings from nodes’ historical interactions and propagates them along the graph structure, using knowledge distillation from LLMs to guide TGNN optimization. In contrast, CROSS first generates textual summaries of interaction histories via an LLM, and then fuses the resulting embeddings with TGNN representations through multi-layer neural networks. Both approaches aim to enhance semantic awareness without altering the underlying structural modeling mechanism of TGNNs.

## 5.2 LLM-Predictors

An emerging line of work directly employs LLMs as predictors for graph learning, bypassing traditional message-passing frameworks. These methods typically convert graph structures into sequential text or embedding-based prompts, allowing LLMs to act as the primary reasoning engine. Although most existing methods in this paradigm are designed for static graphs, with the exception of the recent TGTalker [11], their flexibility enables adaptation to temporal settings. We therefore extend representative approaches to temporal graphs to evaluate their ability to model both structural and semantic evolution.

In-Context Learning (ICL). These training-free methods leverage the pretrained knowledge and reasoning capabilities of LLMs, adapting to graph tasks through prompt-based demonstrations. TGTalker belongs to this category, constructing prompts from temporal interaction triplets (u, v, t) to guide the LLM in generating predictions. Building upon this, we evaluate two Disentangled Spatial-Temporal Thought variants from LLM4DyG [42]: DST-v1, which encodes structural information before temporal context, and DST-v2, which reverses this order. These methods serialize temporal graphs as linear sequences of triplets, requiring the LLM to infer structural relationships implicitly.

Supervised Fine-Tuning (SFT). We adapt two recent LLM-as-Predictor methods originally proposed for static graphs, LLaGA [3] and GraphGPT [29], to temporal settings. These methods construct prompts that include task instructions and sequences of structure-aware embeddings, and employ a learnable projector to align graph embeddings with the LLM embedding space. For LLaGA, we evaluate both variants: LLaGA-ND, which uses the initial embeddings of the center node and its neighbors, and LLaGA-HO, which uses embeddings obtained after multiple rounds of message passing over historical interaction structure. GraphGPT encodes graph structure through aligned graph and text embeddings, and additionally fine-tunes the LLM embedding layer. Unlike ICL methods, SFT approaches provide explicitly structured and hierarchical representations of graph topology.

Detailed descriptions of all 17 methods, along with prompt templates for LLM-Predictors, are provided in Appendix B.

## 6 Experiments and Analysis

Implementation Details. For TGNN-based methods, we follow the training protocols and pipelines established in DyGLib [39], first training all models on the TLP task and then using the resulting checkpoints to initialize training for TNC. For all LLM-based methods, we adopt Qwen3-8B [31] as the backbone model. The hyperparameter configurations and training procedures for SFT-based approaches strictly follow their original implementations [3, 29]. Sentence-BERT [25] is used as the default text encoder to extract embeddings for all textual attributes. For both tasks, each dataset is chronologically split into training, validation, and test sets with a 40%/10%/50% ratio. All experiments are conducted on NVIDIA RTX A6000 GPUs (48GB). Additional implementation details are provided in Appendix C.1.

Evaluation Settings and Metrics. We follow standard evaluation protocols [24, 39] by considering both transductive and inductive settings, along with three negative sampling strategies for TLP: random, historical, and inductive. For evaluation metrics, we adopt widely used measures including

Table 2: Temporal Link Prediction Results. Results are averaged over three independent runs (in %). The best and second-best results for each dataset are highlighted in bold and underlined, respectively.
<table><tr><td rowspan="2">Methods</td><td colspan="2">FOOD</td><td colspan="2">IMDB</td><td colspan="2">Librarything</td><td colspan="2">Beeradvocate</td><td colspan="2">Amazon-Kindle</td><td colspan="2">Ratebeer</td></tr><tr><td>AP</td><td>AUROC</td><td>AP</td><td>AUROC</td><td>AP</td><td>AUROC</td><td>AP</td><td>AUROC</td><td>AP</td><td>AUROC</td><td>AP</td><td>AUROC</td></tr><tr><td>EdgeBank</td><td>50.00±0.00</td><td>49.70±0.00</td><td>50.00±0.00</td><td>46.29±0.00</td><td>50.00±0.00</td><td>49.77±0.00</td><td>50.51±0.00</td><td>50.57±0.00</td><td>50.00±0.00</td><td>49.98±0.00</td><td>50.07±0.00</td><td>49.74±0.00</td></tr><tr><td>JODIE</td><td>72.96±1.88</td><td>71.95±0.69</td><td>60.30±2.37</td><td>65.74±3.20</td><td>75.76±0.63</td><td>73.62±0.75</td><td>79.75±11.63</td><td>80.93±9.53</td><td>82.81±2.00</td><td>81.86±1.69</td><td>92.62±0.18</td><td>91.56±0.36</td></tr><tr><td>DyRep</td><td>72.83±0.88</td><td>69.01±1.53</td><td>64.79±1.30</td><td>69.79±2.03</td><td> $\overline { { 7 3 . 8 1 \pm 1 . 5 5 } }$ </td><td>72.34±1.13</td><td>89.61±0.20</td><td>87.01±0.40</td><td>78.35±2.63</td><td>76.63±2.69</td><td>85.87±0.33</td><td>83.84±1.18</td></tr><tr><td>TGAT</td><td>54.79±1.25</td><td>54.38±0.67</td><td>54.55±0.01</td><td>57.85±0.05</td><td>64.20±0.89</td><td>65.68±0.96</td><td>59.37±3.59</td><td> $6 6 . 5 8 { \scriptstyle \pm 2 . 1 3 }$ </td><td>79.00±1.97</td><td>81.12±1.32</td><td>58.93±4.20</td><td>64.38±2.29</td></tr><tr><td>TGN</td><td>77.74±0.05</td><td>75.74±0.28</td><td>45.40±0.89</td><td>48.79±1.66</td><td>82.60±0.22</td><td>82.02±0.09</td><td>83.46±1.68</td><td> $8 3 . 2 5 { \scriptstyle \pm 1 . 0 6 }$ </td><td>71.77±2.90</td><td>79.92±1.49</td><td>91.55±0.24</td><td>91.41±0.24</td></tr><tr><td>CAWN</td><td>51.71±0.38</td><td>53.96±0.23</td><td>54.00±1.17</td><td>55.60±2.75</td><td>62.01±0.21</td><td>63.21±0.07</td><td>66.15±0.85</td><td> $7 1 . 2 9 { \scriptstyle \pm 0 . 0 5 }$ </td><td>79.34±1.50</td><td>80.79±1.22</td><td>59.94±2.40</td><td>65.78±0.67</td></tr><tr><td>TCL</td><td>49.12±0.79</td><td>50.98±1.23</td><td>55.78±0.49</td><td>58.83±0.44</td><td>54.51±4.02</td><td>56.63±2.91</td><td>55.40±0.18</td><td>64.62±0.29</td><td>79.68±1.51</td><td>80.96±0.95</td><td>57.61±3.77</td><td>62.63±2.39</td></tr><tr><td>GraphMixer</td><td>66.98±1.71</td><td>65.78±0.93</td><td>57.01±0.03</td><td>61.60±0.09</td><td>62.13±1.45</td><td>63.16±2.44</td><td>86.02±0.24</td><td>85.57±0.29</td><td>81.53±0.28</td><td>81.87±0.09</td><td>78.53±4.22</td><td>78.05±4.06</td></tr><tr><td>DyĠFormer</td><td>59.48±0.24</td><td>57.26±0.52</td><td>55.32±0.11</td><td>58.42±0.10</td><td>54.53±2.16</td><td>54.35±3.14</td><td>50.53±0.08</td><td>56.80±0.05</td><td>72.78±0.32</td><td>74.71±0.09</td><td>53.28±0.91</td><td>57.65±0.73</td></tr><tr><td>FreeDyG</td><td>72.40±0.34</td><td>69.60±0.37</td><td>57.53±0.89</td><td>62.16±0.93</td><td>66.08±0.26</td><td>65.13±1.70</td><td>82.93±6.75</td><td>85.18±4.07</td><td>67.26±3.83</td><td>71.25±1.99</td><td>69.49±3.30</td><td>74.16±1.86</td></tr><tr><td>LKD4DyTAG</td><td>66.63±0.95</td><td>63.62±0.95</td><td>56.04±0.92</td><td>59.13±0.53</td><td>59.94±0.49</td><td>62.17±0.34</td><td>79.42±1.15</td><td> $7 9 . 5 2 { \scriptstyle \pm 0 . 9 1 }$ </td><td>80.64±0.46</td><td>81.64±0.42</td><td>69.51±0.03</td><td>70.85±0.05</td></tr><tr><td>CROSS</td><td>58.15±0.13</td><td>56.48±0.24</td><td>55.98±0.05</td><td>59.14±0.12</td><td>59.04±0.83</td><td>60.25±0.94</td><td>51.36±1.03</td><td>57.52±1.01</td><td>87.25±0.13</td><td>88.31±0.12</td><td>52.57±0.09</td><td>57.19±0.21</td></tr><tr><td>TGTalker</td><td>51.04±0.02</td><td>52.02±0.03</td><td>50.22±0.04</td><td>50.43±0.07</td><td>50.63± 0.01</td><td>51.24±0.04</td><td>51.18±0.03</td><td>51.36±0.02</td><td>52.54±0.09</td><td>54.79±0.11</td><td>51.11±0.07</td><td>52.15±0.06</td></tr><tr><td>DST-v1</td><td>51.31±0.01</td><td>52.53±0.03</td><td>50.26±0.10</td><td>50.52±0.20</td><td>50.71±0.02</td><td>51.39±0.06</td><td>51.34±0.04</td><td>51.47±0.05</td><td>52.66±0.10</td><td>54.85±0.09</td><td>51.23±0.04</td><td>51.26±0.03</td></tr><tr><td>DST-v2</td><td>51.33±0.00</td><td>52.57±0.01</td><td>50.35±0.10</td><td>50.70±0.19</td><td>50.61±0.01</td><td>51.19±0.02</td><td>51.21±0.01</td><td>51.53±0.01</td><td>52.62±0.08</td><td>54.81±0.08</td><td>51.05±0.01</td><td>51.07±0.02</td></tr><tr><td>Llaga-ND</td><td>53.10±0.36</td><td>55.52±0.57</td><td>52.01±0.75</td><td>53.55±1.19</td><td>53.98±0.07</td><td>53.95±0.14</td><td>54.97±0.03</td><td>54.94±0.06</td><td>68.62±3.37</td><td>72.85±3.58</td><td>54.00±0.02</td><td>54.99±0.05</td></tr><tr><td>Llaga-HO</td><td>61.95±1.32</td><td>65.67±1.47</td><td>54.37±0.82</td><td>57.34±1.06</td><td>56.87±0.02</td><td>56.88±0.03</td><td>58.00±0.05</td><td>58.00±0.10</td><td>72.64±1.11</td><td>76.63±1.54</td><td>56.09±0.07</td><td>56.15±0.11</td></tr><tr><td>GraphGPT</td><td>55.09±0.43</td><td>57.95±1.02</td><td>51.80±0.18</td><td>53.19±0.29</td><td>54.96±0.04</td><td>54.89±0.11</td><td>56.80±6.80</td><td>58.35±8.35</td><td>70.72±0.72</td><td>71.17±1.13</td><td>56.03±6.03</td><td>59.29±9.29</td></tr></table>

Average Precision (AP) and Area Under the ROC Curve (AUROC), as well as Mean Reciprocal Rank (MRR), which has gained increasing attention in recent work [12, 38]. Due to space constraints and the consistent trends observed between MRR and the other metrics across different settings, we report AP and AUROC under the transductive setting with random negative sampling in the main text, while deferring MRR results and full experimental details under all settings to Appendix C.2. For the temporal node classification task, we use Macro-F1 and Balanced Accuracy (bACC) to account for class imbalance, following prior studies [1, 7, 8, 9, 23, 30].

## 6.1 Main Results

From the results in Tables 2 and 3, we derive several key observations.

TGNNs and LLM-Predictors exhibit a clear capability divide. TGNN-based predictors achieve strong performance on TLP but degrade substantially on TNC, whereas LLM-based predictors excel on TNC while struggling on TLP. Since TLP primarily requires modeling temporal structural evolution and TNC emphasizes semantic drift, this divergence suggests that each paradigm captures only one facet of temporal information. These results reveal a fundamental limitation of existing approaches in achieving balanced structural and semantic learning.

Naive memorization fails under high novelty. The pure memorization baseline, EdgeBank, performs close to random (AUROC ≈ 50) across all datasets, confirming its ineffectiveness in highnovelty scenarios. This underscores the necessity of models that can learn evolving temporal patterns rather than relying on historical repetition.

Continuous state tracking is critical for TLP under structural volatility. Memory-based TGNNs, including TGN, JODIE, and DyRep, consistently rank among the strongest methods on TLP. By maintaining evolving node states, these models capture cumulative temporal effects, which appears to be a key factor underlying their superior performance when interaction repetition is limited.

Explicit structural inputs help LLM-Predictors on TLP, but the learning paradigm remains the bottleneck. Table 2 reveals three key findings. (i) Linearized temporal-structural sequences in ICL inputs, regardless of ordering, perform close to random, indicating that prompting off-the-shelf LLMs is insufficient for extracting implicit structural signals from simple (u, v, t) serializations. (ii) SFT consistently outperforms ICL, suggesting that incorporating explicit structural encoding in the input benefits TLP performance. (iii) Within SFT-based methods, LLaGA-HO and GraphGPT outperform LLaGA-ND, demonstrating that structure-aware aggregation of node embeddings partially compensates for LLMs’ limited structural understanding. Nevertheless, even the best-performing LLM-Predictors on TLP remain significantly behind TGNNs, indicating that input-level structural enhancements alone are insufficient and that the limitation lies in the underlying learning paradigm.

LLM-as-Enhancer struggles to address TGNN semantic limitations. LLM-enhanced TGNN methods such as CROSS and LKD4DyTAG achieve competitive performance on TLP but inherit the weaknesses of TGNNs on TNC. This suggests that simply injecting LLM-derived semantic information into an aggregation framework is insufficient to overcome the structural bias of TGNNs, limiting their ability to model rapidly evolving semantics.

Table 3: Temporal Node Classification Results. Results are averaged over three independent runs (in %). The best and second-best results for each dataset are highlighted in bold and underlined, respectively.
<table><tr><td rowspan="2">Methods</td><td colspan="2">FOOD</td><td colspan="2">IMDB</td><td colspan="2">Beeradvocate</td><td colspan="2">Amazon-Kindle</td><td colspan="2">Ratebeer</td></tr><tr><td>Macro-F1</td><td>mACC</td><td>Macro-F1</td><td>mACC</td><td>Macro-F1</td><td>mACC</td><td>Macro-F1</td><td>mACC</td><td>Macro-F1</td><td>mACC</td></tr><tr><td>JODIE</td><td> $2 5 . 9 5 { \scriptstyle \pm 4 . 6 6 }$ </td><td> $3 3 . 4 8 \pm 1 0 . 5 2 $ </td><td> $1 2 . 6 5 { \scriptstyle \pm 0 . 9 0 }$ </td><td> $1 5 . 8 5 { \scriptstyle \pm 0 . 5 7 }$ </td><td> $7 . 0 0 { \scriptstyle \pm 0 . 0 6 }$ </td><td> $1 4 . 5 2 { \pm } 0 . 1 5$ </td><td> $1 7 . 6 0 { \scriptstyle \pm 0 . 0 1 }$ </td><td> $1 3 . 6 4 \pm 0 . 0 4$ </td><td> $7 . 8 6 \pm 0 . 3 0$ </td><td> $1 3 . 8 8 { \scriptstyle \pm 0 . 3 2 }$ </td></tr><tr><td>DyRep</td><td> $2 1 . 2 2 { \scriptstyle \pm 0 . 1 0 }$ </td><td> $2 2 . 9 3 { \scriptstyle \pm 0 . 1 0 }$ </td><td> $1 3 . 7 4 2 0 . 8 2 $ </td><td>15.97±0.68</td><td> $6 . 5 0 { \scriptstyle \pm 0 . 3 4 }$ </td><td> $1 4 . 2 8 2 0 . 0 1$ </td><td> $1 3 . 3 6 { \pm } 1 . 4 3$ </td><td> $1 3 . 5 1 { \scriptstyle \pm 0 . 8 9 }$ </td><td> $7 . 3 8 { \pm } 1 . 3 3 $ </td><td> $1 3 . 5 8 { \scriptstyle \pm 0 . 9 8 }$ </td></tr><tr><td>TGAT</td><td> $2 2 . 7 3 { \scriptstyle \pm 2 . 7 3 }$ </td><td> $2 7 . 7 8 { \scriptstyle \pm 5 . 5 6 }$ </td><td> $8 . 5 3 { \scriptstyle \pm 0 . 0 2 }$ </td><td>12.48±0.00</td><td> $6 . 1 8 { \scriptstyle \pm 0 . 0 4 }$ </td><td> $1 4 . 3 0 { \scriptstyle \pm 0 . 0 1 }$ </td><td> $1 4 . 9 1 { \scriptstyle \pm 0 . 3 7 }$ </td><td> $1 4 . 3 8 { \scriptstyle \pm 0 . 2 5 }$ </td><td> $8 . 6 7 \pm 0 . 0 5$ </td><td> $1 4 . 5 3 { \scriptstyle \pm 0 . 0 6 }$ </td></tr><tr><td>TGN</td><td> $2 1 . 9 8 { \scriptstyle \pm 1 . 2 0 }$ </td><td> $2 3 . 6 1 { \scriptstyle \pm 0 . 9 6 }$ </td><td>8.51±0.00</td><td> $1 2 . 5 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $8 . 3 5 { \scriptstyle \pm 0 . 0 1 }$ </td><td>15.31±0.02</td><td> $1 3 . 2 2 { \scriptstyle \pm 0 . 2 5 }$ </td><td> $1 3 . 2 8 { \scriptstyle \pm 0 . 1 9 }$ </td><td> $7 . 3 4 \pm 1 . 1 5$ </td><td> $1 3 . 5 4 2 0 . 7 8$ </td></tr><tr><td>CAWN</td><td> $2 0 . 2 8 { \scriptstyle \pm 0 . 2 9 }$ </td><td> $2 2 . 3 8 { \pm } 0 . 1 5$ </td><td> $1 1 . 4 8 { \scriptstyle \pm 2 . 9 8 }$ </td><td>18.75±6.25</td><td> $8 . 7 7 { \scriptstyle \pm 0 . 2 7 }$ </td><td> $1 5 . 4 2 { \scriptstyle \pm 0 . 0 2 }$ </td><td> $1 4 . 9 6 { \scriptstyle \pm 0 . 3 3 }$ </td><td> $1 4 . 4 3 { \scriptstyle \pm 0 . 2 5 }$ </td><td> $8 . 7 9 2 0 . 0 4 $ </td><td> $1 4 . 5 3 { \scriptstyle \pm 0 . 0 8 }$ </td></tr><tr><td>TCL</td><td> $1 9 . 9 9 2 0 . 0 2 $ </td><td> $2 2 . 2 2 { \scriptstyle \pm 0 . 0 1 }$ </td><td> $1 4 . 5 6 { \pm } 1 . 1 9$ </td><td> $1 5 . 4 6 { \scriptstyle \pm 0 . 0 4 }$ </td><td> $9 . 2 6 { \scriptstyle \pm 0 . 2 8 }$ </td><td> $1 6 . 0 3 { \scriptstyle \pm 0 . 3 3 }$ </td><td> $2 4 . 2 5 { \scriptstyle \pm 0 . 6 8 }$ </td><td> $2 1 . 7 6 { \scriptstyle \pm 0 . 7 5 }$ </td><td> $1 4 . 3 4 { \scriptstyle \pm 0 . 6 7 }$ </td><td>19.35±0.96</td></tr><tr><td> $\mathrm { G r a p h M i x e r }$ </td><td> $1 6 . 9 4 { \scriptstyle \pm 0 . 0 1 }$ </td><td> $2 0 . 7 7 { \scriptstyle \pm 0 . 0 2 }$ </td><td> $9 . 2 0 { \scriptstyle \pm 0 . 6 9 }$ </td><td> $1 2 . 9 1 { \scriptstyle \pm 0 . 4 1 }$ </td><td> $6 . 1 4 \pm 0 . 0 1$ </td><td> $1 4 . 2 9 2 0 . 0 1$ </td><td> $1 3 . 3 5 { \scriptstyle \pm 0 . 4 0 }$ </td><td> $1 3 . 3 8 { \pm } 0 . 2 2$ </td><td> $7 . 1 3 { \pm } 1 . 7 8$ </td><td> $1 3 . 5 9 2 1 . 0 9$ </td></tr><tr><td>DyĠFormer</td><td> $3 6 . 5 8 { \scriptstyle \pm 3 . 0 2 }$ </td><td> $3 6 . 3 9 { \scriptstyle \pm 3 . 9 2 }$ </td><td> $2 0 . 0 5 { \scriptstyle \pm 1 1 . 5 4 }$ </td><td> $2 0 . 0 1 { \scriptstyle \pm 7 . 5 1 }$ </td><td> $1 2 . 9 1 { \scriptstyle \pm 1 . 5 9 }$ </td><td> $1 8 . 9 3 { \scriptstyle \pm 0 . 4 8 }$ </td><td> $2 0 . 1 9 2 0 . 6 0$ </td><td> $1 8 . 1 4 { \scriptstyle \pm 0 . 4 3 }$ </td><td> $1 5 . 5 2 { \pm } 1 . 3 7$ </td><td> $1 8 . 4 2 { \scriptstyle \pm 1 . 1 2 }$ </td></tr><tr><td>FreeDyG</td><td>20.46±0.04</td><td>22.47±0.02</td><td>10.20±1.55</td><td>13.46±0.89</td><td> $7 . 7 5 { \scriptstyle \pm 1 . 6 1 }$ </td><td>14.91±0.62</td><td>12.80±0.86</td><td> $1 3 . 1 0 { \scriptstyle \pm 0 . 4 7 }$ </td><td>8.54±0.46</td><td> $1 4 . 2 3 { \pm } 0 . 3 5$ </td></tr><tr><td>LKD4DyTAG</td><td> $2 3 . 3 7 { \scriptstyle \pm 3 . 3 8 }$ </td><td> $2 7 . 7 8 { \scriptstyle \pm 5 . 5 6 }$ </td><td>8.51±0.01</td><td>12.50±0.02</td><td>6.14±0.01</td><td> $1 4 . 2 9 2 0 . 0 2 $ </td><td> $1 0 . 3 9 \pm 0 . 0 1$ </td><td> $1 1 . 1 1 { \scriptstyle \pm 0 . 0 4 }$ </td><td> $5 . 8 0 { \scriptstyle \pm 0 . 0 7 }$ </td><td> $1 2 . 5 7 { \scriptstyle \pm 0 . 0 1 }$ </td></tr><tr><td>CROSS</td><td> $1 9 . 9 9 2 0 . 0 2 $ </td><td> $2 2 . 2 2 { \scriptstyle \pm 0 . 0 1 }$ </td><td>8.62±0.02</td><td> $1 2 . 4 1 \pm 0 . 0 1$ </td><td> $6 . 1 7 { \scriptstyle \pm 0 . 0 1 }$ </td><td> $1 4 . 3 0 { \scriptstyle \pm 0 . 0 1 }$ </td><td> $9 . 5 9 { \scriptstyle \pm 0 . 0 7 }$ </td><td> $1 0 . 9 7 { \scriptstyle \pm 0 . 0 3 }$ </td><td> $5 . 3 5 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $1 2 . 5 0 { \scriptstyle \pm 0 . 0 0 }$ </td></tr><tr><td>TGTalker</td><td>31.84±0.05</td><td>30.13±0.52</td><td>28.96±0.59</td><td>30.63±0.46</td><td> ${ \bf 1 6 . 5 2 } \pm 0 . 3 1$ </td><td>18.00±0.30</td><td>25.60±0.24</td><td> $2 4 . 0 2 { \scriptstyle \pm 0 . 2 6 }$ </td><td> $1 3 . 4 2 \pm 0 . 4 7$ </td><td>15.29±0.78</td></tr><tr><td>DST-v1</td><td>32.87±0.22</td><td>31.62±0.26</td><td>29.37±0.33</td><td>30.94±0.69</td><td> $1 6 . 2 9 { \scriptstyle \pm 0 . 4 2 }$ </td><td> $1 7 . 4 7 { \scriptstyle \pm 0 . 3 8 }$ </td><td>26.00±0.33</td><td> $2 4 . 4 2 \pm 0 . 1 4$ </td><td>12.83±0.92</td><td> $1 4 . 8 9 { \scriptstyle \pm 0 . 6 5 }$ </td></tr><tr><td>DST-v2</td><td>37.59±0.25</td><td>39.91±0.22</td><td>29.14±0.24</td><td>30.72±0.36</td><td> $1 6 . 1 2 { \scriptstyle \pm 0 . 2 5 }$ </td><td> $1 7 . 1 7 { \scriptstyle \pm 0 . 3 4 }$ </td><td> $2 4 . 6 4 \pm 0 . 3 7$ </td><td> $2 3 . 9 4 \pm 0 . 2 5$ </td><td>13.34±0.94</td><td> $1 4 . 6 9 2 0 . 5 0 $ </td></tr><tr><td>Llaga-ND</td><td>28.48±0.30</td><td>32.04±0.66</td><td>19.13±1.13</td><td>25.95±1.16</td><td> $1 3 . 9 0 { \scriptstyle \pm 1 . 8 8 }$ </td><td> $1 8 . 3 0 { \scriptstyle \pm 1 . 6 0 }$ </td><td> $1 8 . 5 7 { \scriptstyle \pm 2 . 5 3 }$ </td><td> $1 7 . 2 5 { \pm } 1 . 1 3$ </td><td> $1 4 . 1 2 { \scriptstyle \pm 0 . 0 8 }$ </td><td> $1 7 . 5 1 { \pm } 1 . 1 9$ </td></tr><tr><td>Llaga-HO</td><td>30.37±0.41</td><td>33.26±1.75</td><td>20.66±1.63</td><td> $2 6 . 4 5 { \scriptstyle \pm 1 . 6 7 }$ </td><td> $1 5 . 0 9 { \scriptstyle \pm 1 . 7 0 }$ </td><td> $1 9 . 2 4 \pm 1 . 3 5$ </td><td> $2 1 . 0 1 { \scriptstyle \pm 2 . 6 8 }$ </td><td> $1 8 . 6 6 { \scriptstyle \pm 2 . 5 5 }$ </td><td> $1 6 . 9 4 { \scriptstyle \pm 0 . 1 7 }$ </td><td> $\mathbf { 2 0 . 6 3 } \pm 1 . 4 2$ </td></tr><tr><td>GraphGPT</td><td>36.74±2.83</td><td>38.40±1.47</td><td>25.79±5.59</td><td>41.01±16.08</td><td> $1 6 . 3 1 { \pm } 0 . 2 8 $ </td><td> $\overline { { 2 0 . 2 2 \pm 0 . 3 7 } }$ </td><td> $2 4 . 2 8 { \scriptstyle \pm 4 . 6 3 }$ </td><td> $2 4 . 7 4 \substack { \pm 0 . 1 6 }$ </td><td>18.93±0.16</td><td> $2 0 . 2 1 \pm 0 . 2 0 $ </td></tr></table>

## 6.2 Why Do TGNNs Fail at Semantic Tracking?

To systematically analyze why TGNNs fail catastrophically on semantic tracking (TNC), we conduct in-depth probing studies to isolate their underlying bottlenecks. We select JODIE as a representative TGNN due to its consistently strong performance, and choose DST-v2 and LLaGA-HO—two top-performing LLM-Predictors on TNC—as representatives of ICL- and SFT-based methods, respectively. The comparative analysis (Figure 3) reveals several fundamental limitations of TGNNs.

Architectural semantic blindness. We first examine whether models effectively utilize textual attributes for semantic tracking. As shown in Figure 3(a) and (b), removing textual attributes causes a sharp drop in TLP performance for TGNNs but a much smaller impact on TNC, whereas LLM-Predictors suffer substantial degradation on TNC with minor impact on TLP. This consistent trend holds across all methods in both paradigms (see Figure 7 for full results). The observed asymmetry suggests that, although TGNNs can exploit text as auxiliary signals for structural modeling, their messageaggregation architectures are inherently limited in leveraging textual information for semantic tracking.

(a)  
![](images/a15015f968f2aa95e57423d200174be36c5c26f6176caf402d906c2f8a687d0e.jpg)

(b)  
(c)  
![](images/e1125b37510524157c9ef608111ce182595a34ff27ac2cfc8cfaad9f630a3b66.jpg)

![](images/8f2d41d2179ce671db7cc8104431decdbe39c31d79940960016d21ee76a57fc0.jpg)  
Figure 3: Comparative analysis of representative TGNN and LLM-Predictor methods. (a) and (b) show the performance of each method on TLP and TNC, respectively, with and without textual inputs. (c) illustrates how model performance varies as the frequency of semantic changes increases.

Representation lock-in. We next investigate whether this limitation can be mitigated through downstream supervision. Specifically, we compare two training strategies for TGNNs on TNC:

linear probing (training only the classifier head on a TLP-pretrained backbone) and end-to-end finetuning (jointly optimizing the backbone and classifier). Surprisingly, end-to-end fine-tuning yields no meaningful improvement over linear probing and can even degrade performance (Table 22). This suggests a strong representation lock-in, where gradients from semantic supervision are insufficient to reshape representations that are already heavily biased toward structural patterns. As a result, the learned embeddings remain anchored in topological dynamics and are not adaptable to semantic drift.

Structural inertia under semantic volatility. Finally, we analyze model performance under varying degrees of semantic change based on the frequency of label drifts. As shown in Figure 3(c), TGNN performance deteriorates sharply as the environment transitions from slow (1–2 changes) to fast (>2 changes) semantic drift. In contrast, LLM-Predictors maintain stable or even improved performance. This highlights an inherent structural inertia in TGNNs: their reliance on aggregating historical interactions becomes a liability when semantic states change rapidly. Conversely, LLM-Predictors exhibit strong semantic adaptability, dynamically leveraging immediate textual context to track evolving semantics without being constrained by historical structural dependencies.

![](images/a0be5b953658aa2131858a6f155d0483557efca1aafd41b4fee9e9e71bcc2cf1.jpg)

![](images/bec47aaf3196a07400ec1c13745acda6e7571cd1547051499b4fe7c13ef4dbb3.jpg)

![](images/4fb3aacbe32a9a3436e0c4297d08c5f5b2655f4545ad0ef686b5b9295ecc3080.jpg)  
Figure 4: Efficiency Results. Scalability results of all methods are shown in Figure 8.

## 6.3 Efficiency Analysis

We evaluate the practical applicability of all benchmarked methods by analyzing their computational efficiency. Figure 4 reports training time, GPU memory consumption, parameter counts, and FLOPs for all methods on the largest-scale Amazon-Kindle dataset.

Staggering overhead of LLM-Predictor paradigms. The most striking observation is the substantial computational cost incurred by LLM-Predictor paradigms due to the use of generative LLMs. In particular, GraphGPT—whose training involves fine-tuning the embedding layer of an LLM—requires nearly 600,000 seconds of training time and 13.5 teraFLOPs, exceeding even the most resource-intensive TGNNs by several orders of magnitude. Although LLM-Predictors achieve strong performance on semantic tracking (TNC), their prohibitive computational overhead limits their practicality in real-world applications, highlighting the need for fundamental improvements to make this paradigm deployable in realistic settings.

Performance–efficiency trade-off within TGNNs. Clear trade-offs are observed among TGNNs. Memory-based architectures (e.g., JODIE, DyRep, and TGN) achieve dominant performance on TLP, but incur substantially higher computational and memory overhead due to the need to maintain and continuously update recurrent node states. This reflects the inherent cost of continuous-time structural tracking required to handle structural volatility effectively.

Scalability robustness. Beyond static efficiency, we further evaluate scalability by measuring runtime on progressively larger subsets of the Amazon-Kindle dataset, ranging from 1M to 5M interactions. Results show that all methods exhibit approximately linear scaling with respect to dataset size, indicating predictable and robust scalability. Detailed results are provided in Figure 8.

## 7 Conclusions

In this work, we introduce TTGBench, a benchmark for jointly evaluating structural and semantic evolution in temporal graphs. By constructing six text-rich datasets with dual volatility, TTGBench provides a challenging and realistic setting that requires modeling both structural novelty and semantic dynamics. Extensive experiments on 17 state-of-the-art methods reveal a clear capability divide: TGNN-based methods excel at structural modeling but struggle with semantic tracking, whereas LLMbased predictors exhibit the opposite trend. Through in-depth analysis, we attribute this phenomenon to intrinsic architectural biases in the two paradigms and further identify clear trade-offs between performance and efficiency in existing approaches.

Our findings suggest several promising directions for future research. A key challenge is to develop unified models that can simultaneously capture structural and semantic dynamics. In addition, improving structural reasoning in LLMs and enhancing the semantic adaptability of TGNNs are both critical for bridging the current gap. Finally, designing efficient and scalable architectures remains essential for practical deployment.

## References

[1] Kay Henning Brodersen, Cheng Soon Ong, Klaas Enno Stephan, and Joachim M Buhmann. The balanced accuracy and its posterior distribution. In 2010 20th international conference on pattern recognition, pages 3121–3124. IEEE, 2010.

[2] Chenwei Cai, Ruining He, and Julian McAuley. Spmc: Socially-aware personalized markov chains for sparse sequential recommendation. arXiv preprint arXiv:1708.04497, 2017.

[3] Runjin Chen, Tong Zhao, Ajay Kumar Jaiswal, Neil Shah, and Zhangyang Wang. Llaga: Large language and graph assistant. In International Conference on Machine Learning, pages 7809–7823. PMLR, 2024.

[4] Weilin Cong, Si Zhang, Jian Kang, Baichuan Yuan, Hao Wu, Xin Zhou, Hanghang Tong, and Mehrdad Mahdavi. Do we really need complicated model architectures for temporal networks? arXiv preprint arXiv:2302.11636, 2023.

[5] Manuel Dileo, Matteo Zignani, and Sabrina Gaito. Temporal graph learning for dynamic link prediction with text in online social networks. Machine learning, 113(4):2207–2226, 2024.

[6] Linlin Ding, Baishuo Han, Shu Wang, Xiaoguang Li, and Baoyan Song. User-centered recommendation using us-elm based on dynamic graph model in e-commerce. International Journal ofMachine Learning and Cybernetics, 10(4):693–703, 2019.

[7] Margherita Grandini, Enrico Bagli, and Giorgio Visani. Metrics for multi-class classification: an overview. arXiv preprint arXiv:2008.05756, 2020.

[8] Haibo He and Edwardo A Garcia. Learning from imbalanced data. IEEE Transactions on knowledge and data engineering, 21(9):1263–1284, 2009.

[9] Maria Cristina Hinojosa Lee, Johan Braet, and Johan Springael. Performance metrics for multilabel emotion classification: comparing micro, macro, and weighted f1-scores. Applied Sciences, 14(21):9863, 2024.

[10] Yupeng Hou, Jiacheng Li, Zhankui He, An Yan, Xiusi Chen, and Julian McAuley. Bridging language and items for retrieval and recommendation. arXiv preprint arXiv:2403.03952, 2024.

[11] Shenyang Huang, Emma Kondrup, Ali Parviz, Zachary Yang, Zifeng Ding, Michael M. Bronstein, Reihaneh Rabbany, and Guillaume Rabusseau. Are large language models good temporal graph learners? In New Perspectives in Graph Machine Learning, 2025.

[12] Shenyang Huang, Farimah Poursafaei, Jacob Danovitch, Matthias Fey, Weihua Hu, Emanuele Rossi, Jure Leskovec, Michael Bronstein, Guillaume Rabusseau, and Reihaneh Rabbany. Temporal graph benchmark for machine learning on temporal graphs. Advances in Neural Information Processing Systems, 36:2056–2073, 2023.

[13] Xuanwen Huang, Yang Yang, Yang Wang, Chunping Wang, Zhisheng Zhang, Jiarong Xu, Lei Chen, and Michalis Vazirgiannis. Dgraph: A large-scale financial dataset for graph anomaly detection. Advances in Neural Information Processing Systems, 35:22765–22777, 2022.

[14] Seyed Mehran Kazemi, Rishab Goel, Kshitij Jain, Ivan Kobyzev, Akshay Sethi, Peter Forsyth, and Pascal Poupart. Representation learning for dynamic graphs: A survey. Journal ofMachine Learning Research, 21(70):1–73, 2020.

[15] Seyed Mehran Kazemi, Rishab Goel, Kshitij Jain, Ivan Kobyzev, Akshay Sethi, Peter Forsyth, and Pascal Poupart. Representation learning for dynamic graphs: A survey. Journal of Machine Learning Research, 21(70):1–73, 2020.

[16] Srijan Kumar, Xikun Zhang, and Jure Leskovec. Predicting dynamic embedding trajectory in temporal interaction networks. In Proceedings of the 25th ACM SIGKDD international conference on knowledge discovery & data mining, pages 1269–1278, 2019.

[17] Antonio Longa, Veronica Lachi, Gabriele Santin, Monica Bianchini, Bruno Lepri, Pietro Lio, franco scarselli, and Andrea Passerini. Graph neural networks for temporal graphs: State of the art, open challenges, and opportunities. Transactions on Machine Learning Research, 2023.

[18] Bodhisattwa Prasad Majumder, Shuyang Li, Jianmo Ni, and Julian McAuley. Generating personalized recipes from historical user preferences. arXiv preprint arXiv:1909.00105, 2019.

[19] Julian McAuley, Jure Leskovec, and Dan Jurafsky. Learning attitudes and attributes from multi-aspect reviews. In 2012 IEEE 12th International Conference on Data Mining, pages 1020–1025. IEEE, 2012.

[20] Julian John McAuley and Jure Leskovec. From amateurs to connoisseurs: modeling the evolution of user expertise through online reviews. In Proceedings ofthe 22nd international conference on World Wide Web, pages 897–908, 2013.

[21] Rishabh Misra. Imdb spoiler dataset, 05 2019.

[22] Anirban Mitra and Subrata Paul. Analyzing social networks with dynamic graphs: Unravelling the ever-evolving connections. In Applied Graph Data Science, pages 195–214. Elsevier, 2025.

[23] Juri Opitz. From bias and prevalence to macro f1, kappa, and mcc: A structured overview of metrics for multi-class evaluation. Heidelberg University, 2022.

[24] Farimah Poursafaei, Shenyang Huang, Kellin Pelrine, and Reihaneh Rabbany. Towards better evaluation for dynamic link prediction. Advances in Neural Information Processing Systems, 35:32928–32941, 2022.

[25] Nils Reimers and Iryna Gurevych. Sentence-bert: Sentence embeddings using siamese bertnetworks. In Proceedings ofthe 2019 Conference on Empirical Methods in Natural Language Processing. Association for Computational Linguistics, 11 2019.

[26] Emanuele Rossi, Ben Chamberlain, Fabrizio Frasca, Davide Eynard, Federico Monti, and Michael Bronstein. Temporal graph networks for deep learning on dynamic graphs. arXiv preprint arXiv:2006.10637, 2020.

[27] Amit Roy, Ning Yan, and Masood Mortazavi. Llm-driven knowledge distillation for dynamic text-attributed graphs. arXiv preprint arXiv:2502.10914, 2025.

[28] Joakim Skarding, Bogdan Gabrys, and Katarzyna Musial. Foundations and modeling of dynamic networks using dynamic graph neural networks: A survey. iEEE Access, 9:79143–79168, 2021.

[29] Jiabin Tang, Yuhao Yang, Wei Wei, Lei Shi, Lixin Su, Suqi Cheng, Dawei Yin, and Chao Huang. Graphgpt: Graph instruction tuning for large language models. In Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 491–500, 2024.

[30] Adane Nega Tarekegn, Mario Giacobini, and Krzysztof Michalak. A review of methods for imbalanced multi-label classification. Pattern Recognition, 118:107965, 2021.

[31] Qwen Team. Qwen3 technical report, 2025.

[32] Yuxing Tian, Yiyan Qi, and Fan Guo. Freedyg: Frequency enhanced continuous-time dynamic graph model for link prediction. In The twelfth international conference on learning representations, 2024.

[33] Toan Khang Trinh and Zhuxuanzi Wang. Dynamic graph neural networks for multi-level financial fraud detection: A temporal-structural approach. Annals of Applied Sciences, 5(1), 2024.

[34] Rakshit Trivedi, Mehrdad Farajtabar, Prasenjeet Biswal, and Hongyuan Zha. Dyrep: Learning representations over dynamic graphs. In International conference on learning representations, 2019.

[35] Lu Wang, Xiaofu Chang, Shuang Li, Yunfei Chu, Hui Li, Wei Zhang, Xiaofeng He, Le Song, Jingren Zhou, and Hongxia Yang. Tcl: Transformer-based dynamic graph modelling via contrastive learning. arXiv preprint arXiv:2105.07944, 2021.

[36] Yanbang Wang, Yen-Yu Chang, Yunyu Liu, Jure Leskovec, and Pan Li. Inductive representation learning in temporal networks via causal anonymous walks. arXiv preprint arXiv:2101.05974, 2021.

[37] Da Xu, Chuanwei Ruan, Evren Korpeoglu, Sushant Kumar, and Kannan Achan. Inductive representation learning on temporal graphs. arXiv preprint arXiv:2002.07962, 2020.

[38] Lu Yi, Jie Peng, Yanping Zheng, Fengran Mo, Zhewei Wei, Yuhang Ye, Yue Zixuan, and Zengfeng Huang. TGB-seq benchmark: Challenging temporal GNNs with complex sequential dynamics. In The Thirteenth International Conference on Learning Representations, 2025.

[39] Le Yu, Leilei Sun, Bowen Du, and Weifeng Lv. Towards better dynamic graph learning: New architecture and unified library. Advances in Neural Information Processing Systems, 36:67686–67700, 2023.

[40] Jiasheng Zhang, Jialin Chen, Menglin Yang, Aosong Feng, Shuang Liang, Jie Shao, and Rex Ying. Dtgb: A comprehensive benchmark for dynamic text-attributed graphs. arXiv preprint arXiv:2406.12072, 2024.

[41] Siwei Zhang, Yun Xiong, Yateng Tang, Jiarong Xu, Xi Chen, Zehao Gu, Xuehao Zheng, Zi’an Jia, and Jiawei Zhang. Unifying text semantics and graph structures for temporal textattributed graphs with large language models. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025.

[42] Zeyang Zhang, Xin Wang, Ziwei Zhang, Haoyang Li, Yijian Qin, and Wenwu Zhu. Llm4dyg: can large language models solve spatial-temporal problems on dynamic graphs? In Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pages 4350–4361, 2024.

[43] Ziyi Zhang, Diya Li, Zhenlei Song, Nick Duffield, and Zhe Zhang. Location-aware social network recommendation via temporal graph networks. In Proceedings ofthe 7th ACM SIGSPA-TIAL Workshop on Location-based Recommendations, Geosocial Networks and Geoadvertising, pages 58–61, 2023.

[44] Tong Zhao, Julian McAuley, and Irwin King. Improving latent factor models via personalized feature projection for one class recommendation. In Proceedings ofthe 24th ACM international on conference on information and knowledge management, pages 821–830, 2015.

[45] Ziwei Zhao, Fake Lin, Xi Zhu, Zhi Zheng, Tong Xu, Shitian Shen, Xueying Li, Zikai Yin, and Enhong Chen. Dynllm: when large language models meet dynamic graph recommendation. arXiv preprint arXiv:2405.07580, 2024.

[46] Yuyu Zhou, Me Sun, and Fan Zhang. Graph neural network-based anomaly detection in financial transaction networks. Journal ofComputing Innovations and Applications, 1(2):87–101, 2023.

## Appendix

## A Datasets Details

## A.1 Dataset Description

Table 4: Comparison between our datasets and existing temporal graph datasets.
<table><tr><td></td><td>Dataset</td><td>Nodes</td><td>Total Edges</td><td>Unique Edges</td><td>Unique Steps</td><td>Text Attr</td><td>Novelty</td><td>Surprise</td><td>Node Labels</td></tr><tr><td rowspan="18">Previous</td><td>mooc</td><td>7,144</td><td>411,749</td><td>178,443</td><td>345,600</td><td>x</td><td>0.433</td><td>0.785</td><td>N.A.</td></tr><tr><td>lastfm</td><td>1,980</td><td>1,293,103</td><td>154,993</td><td>1,283,614</td><td>x</td><td>0.12</td><td>0.369</td><td>N.A.</td></tr><tr><td>enron</td><td>184</td><td>125,235</td><td>3,125</td><td>22,632</td><td>x</td><td>0.076</td><td>0.402</td><td>N.A.</td></tr><tr><td>SocialEvo</td><td>74</td><td>2,099,519</td><td>4,486</td><td>565,932</td><td>x</td><td>0.002</td><td>0.027</td><td>N.A.</td></tr><tr><td>uci</td><td>1,899</td><td>59,835</td><td>20,296</td><td>58,911</td><td>x</td><td>0.339</td><td>0.796</td><td>N.A.</td></tr><tr><td>Flights</td><td>13,169</td><td>1,927,145</td><td>395,072</td><td>122</td><td>x</td><td>0.194</td><td>0.362</td><td>N.A.</td></tr><tr><td>CanParl</td><td>734</td><td>74,478</td><td>51,331</td><td>14</td><td>x</td><td>0.673</td><td>0.654</td><td>N.A.</td></tr><tr><td>USLegis</td><td>225</td><td>60,396</td><td>26,423</td><td>12</td><td>x</td><td>0.437</td><td>0.340</td><td>N.A.</td></tr><tr><td>UNtrade</td><td>255</td><td>507,497</td><td>36,182</td><td>32</td><td>x</td><td>0.09</td><td>0.051</td><td>N.A.</td></tr><tr><td>UNvote</td><td>201</td><td>1,035,742</td><td>31,516</td><td>72</td><td>x</td><td>0.056</td><td>0.017</td><td>N.A.</td></tr><tr><td>Contacts</td><td>692</td><td>2,426,279</td><td>79,530</td><td>8,064</td><td>x</td><td>0.023</td><td>0.291</td><td>N.A.</td></tr><tr><td>tgbl-wiki</td><td>9,227</td><td>157,474</td><td>18,257</td><td>152,757</td><td>x</td><td>0.116</td><td>0.108</td><td>N.A.</td></tr><tr><td>tgbn-reddit GDELT</td><td>11,766</td><td>27,174,118</td><td>516,669</td><td>21,889,537</td><td>x</td><td>0.02</td><td>0.013</td><td>N.A.</td></tr><tr><td>ICEWS1819</td><td>6,786 31,796</td><td>1,339,245 1,100,071</td><td>249,241 314,011</td><td>2,403</td><td>√ √</td><td>0.22</td><td>0.562</td><td>N.A.</td></tr><tr><td>FOOD</td><td></td><td></td><td></td><td>730</td><td></td><td>0.292</td><td>0.695</td><td>N.A.</td></tr><tr><td rowspan="6">Ours</td><td></td><td>33,773</td><td>401,057</td><td>401,057</td><td>6,252</td><td>√</td><td>1.0</td><td>1.0</td><td>9</td></tr><tr><td>IMDB</td><td>32,371</td><td>310,891</td><td>310,891</td><td>7,099</td><td>√</td><td>1.0</td><td>1.0</td><td>8</td></tr><tr><td>Librarything</td><td>51,201</td><td>785,690</td><td>785,690</td><td>2,914</td><td>√</td><td>1.0</td><td>1.0</td><td>N.A.</td></tr><tr><td>Beeradvocate</td><td>99,361</td><td>1,586,573</td><td>1,571,767</td><td>1,577,920</td><td>√</td><td>0.991</td><td>0.997</td><td>7</td></tr><tr><td>Ratebeer</td><td>139,538</td><td>2,924,105</td><td>2,855,175</td><td>4,254</td><td>√</td><td>1.0</td><td>1.0</td><td>8</td></tr><tr><td>Amazon-Kindle</td><td>215,504</td><td>5,621,343</td><td>5,561,738</td><td>5,484,604</td><td>√</td><td>1.0</td><td>1.0</td><td>8</td></tr></table>

The detailed descriptions of the six proposed datasets are provided below, with comparisons to existing datasets summarized in Table 4.

• FOOD [18] This dataset contains recipes and user reviews collected from Food.com (formerly GeniusKitchen), covering 18 years of temporal interactions. It provides rich signals for studying culinary trends, evolving user preferences, and long-term temporal dynamics. In the constructed graph, users and recipes are represented as nodes, and a timestamped edge is created when a user reviews a recipe at time t. Each recipe contains detailed cooking instructions, ingredient information, and category labels. We assign recipe categories as users’ temporal interest labels, representing users’ evolving interests at each interaction. Since each recipe may have multiple labels, FOOD supports multi-label temporal node classification.

• IMDB [21] This dataset is constructed from IMDB and consists of users, movies, and timestamped review interactions. Each movie includes plot summaries and genre tags, while each review contains textual content and temporal information. We use movie genre tags as users’ temporal interest labels at the time of interaction, making IMDB another multi-label classification benchmark. The rich textual information in both movies and reviews provides abundant signals for modeling both structural evolution and semantic drift in temporal graphs.

• LibraryThing [2, 44] Collected from the book review platform LibraryThing (https://www. librarything.com/ ), this dataset contains users, books, and timestamped textual review interactions. It captures diverse literary preferences and their evolution over time. In our temporal graph, users and books are nodes connected through review-based temporal edges. Since this dataset does not provide item category labels, it is used only for temporal link prediction.

• BeerAdvocate [19] This dataset consists of beer reviews involving users and beers as graph nodes. Each beer is associated with a unique style label, which we use as the user’s temporal interest label during interaction. Because each beer has a single category label, BeerAdvocate supports multi-class temporal node classification.

• Amazon-Kindle [10] This dataset is drawn from the Amazon Reviews 2023 collection (https: //amazon-reviews-2023.github.io/ ) and focuses on user reviews of Kindle products. Users and products form the graph nodes, while products and reviews both provide rich textual information. The resulting product-review network spans from 1996 to September 2023. Each product has a specific category label, making this dataset suitable for multi-class temporal node classification.

![](images/1f0bcdf57c4f50d7eeb4ddd2cf467f00e3af3b7542cd93fcf3495acb0bcb6db9.jpg)  
Figure 5: Distribution of edge text length (#tokens) on TTGBench datasets.

• Ratebeer [20] Ratebeer is another large-scale beer review dataset similar to BeerAdvocate, but with much denser temporal interactions, providing a complementary dynamic scenario for temporal graph modeling. Each beer also has a single category label, enabling multi-class temporal node classification.

## A.2 Dataset Analysis

Distribution of Edge Text Lengths. Given that the temporal structure formed by sequential edges is crucial for learning in temporal graph models, we present the distribution of text lengths on edges (measured in tokens) across all datasets in Figure 5. As shown in the figure, our datasets exhibit a right-skewed distribution, which is a typical characteristic of user-generated content (UGC) in real-world scenarios. Most user interactions consist of brief feedback (e.g., a “5-star” rating or a short comment like “Good”), while only a small fraction of highly engaged users contribute long-tail texts. Additionally, the text length distributions vary across different datasets. These characteristics pose several challenges for temporal graph learning:

• The sparsity of information in short texts requires models to capture information over extended temporal windows to effectively associate with structural evolution.

• Long texts often contain multiple semantic cues (e.g., multi-faceted product descriptions), demanding that models identify the structure-relevant parts. Moreover, time-sensitive information embedded in long texts adds complexity for precise temporal alignment.

• The differing centers of text length distributions across datasets reflect domain-specific language habits, requiring models to adapt to domain variations. Notably, the existence of long-tail distributions also demands robustness from the models.

Node Label Distribution. As shown in Table 5, label distributions vary across datasets, with several exhibiting substantial class imbalance. As discussed in Section 6, we therefore adopt multiple imbalance-aware evaluation metrics for temporal node classification.

## B Methods Details

## B.1 TGNNs

The traditional temporal graph neural network methods are described as follows:

• JODIE [16] models the future evolution of entity embeddings by forecasting their trajectories over time to predict future interactions and entity states. It employs two interconnected recurrent neural networks to update the dynamic states of entities, along with a projection operation that estimates each entity’s future embedding trajectory.

Table 5: Text labels of each dataset.  
![](images/c86773f0b1ddc7a85f374d6bd6e2f64b624dd1681378bd41a3c88d3f9d8aa0b5.jpg)

• DyRep [34] introduces a recurrent framework to continuously update node states after each interaction. Additionally, it features a temporal-attentive aggregation component designed to capture the evolving structural patterns in temporal graphs.

• TGAT [37] generates node representations by aggregating information from each node’s temporally relevant neighbors, using a self-attention mechanism. It also integrates a time encoding function to model temporal dependencies within the graph.

• TGN [26] maintains a dynamic memory for every node, which is updated whenever the node participates in an interaction. This update process involves a message function, a message aggregator, and a memory updater. A dedicated embedding module is then used to produce time-aware node representations based on these evolving memories.

• CAWN [36] first extracts several causal anonymous walks for each node to uncover the causal dynamics of the network and produce relative node identities. These walks are then encoded using recurrent neural networks, and their outputs are aggregated to compute the final node representations.

• EdgeBank [24] is a memorization-based method without trainable parameters for transductive temporal link prediction. It records observed interactions in memory and predicts a positive link if the interaction exists in memory, and negative otherwise.

• TCL [35] constructs each node’s interaction sequence by applying a breadth-first search on its temporally dependent sub-graph. It then employs a graph transformer that integrates both topological and temporal cues for learning node representations, further incorporating a crossattention mechanism to capture interdependencies between interacting nodes.

• GraphMixer [4] demonstrates that a fixed time encoding function outperforms a trainable one. This fixed function is embedded within a link encoder built on the MLP-Mixer architecture to model temporal interactions. Additionally, a node encoder using neighbor mean-pooling summarizes node features.

• DyGFormer [39] generates node representations by leveraging each node’s historical first-hop interactions. It introduces a neighbor co-occurrence encoding mechanism to capture relationships between nodes based on their interaction histories. Additionally, a patching strategy is applied to segment each interaction sequence into smaller patches, enabling the model to effectively utilize longer historical sequences.

• FreeDyG [32] enhances task performance by adaptively weighting frequency components in the temporal structure’s frequency domain. It extracts both high- and low-frequency components of dynamic patterns, then amplifies task-related frequencies and attenuates unrelated signals.

## B.2 LLM-as-Enhancer

This line of work leverages frozen LLMs to enrich semantic representations for TGNNs, while TGNNs remain the final predictive models.

• LKD4DyTAG [27] feeds textual descriptions of temporal graphs into a frozen teacher LLM to extract embeddings, which are then used to guide a student GNN through knowledge distillation, enhancing the GNN with semantic knowledge captured by the LLM.

• CROSS [41] first generates textual summaries of interaction histories using an LLM, and then employs a semantic-structure co-encoder to help GNNs incorporate the temporal semantic information summarized by the LLM.

## B.3 LLM-as-Predictor

LLM-as-Predictor approaches directly employ LLMs to solve downstream tasks on temporal graphs. Depending on how LLMs are adapted, this paradigm mainly includes two settings: Supervised Fine-Tuning (SFT) and In-Context Learning (ICL). In both cases, temporal graphs must first be transformed into LLM-understandable sequences, after which temporal structural patterns are learned through either SFT or ICL.

Existing methods under this paradigm have primarily focused on static graphs and remain under-explored for temporal graphs. Given their strong potential, and to encourage further development of LLM-as-Predictor methods for temporal graph learning, we provide detailed descriptions of the six baselines evaluated in the main paper.

![](images/552ad9ad6c1cd0b395b1f97fd7c1d6040a0831c5557ffc834604adda010bd6a6.jpg)  
Figure 6: A temporal graph demo.

Structural Sequence. Since LLMs can only process token sequences, the structural information in temporal graphs must first be serialized. The ICL-based methods directly represent the temporal structure as a sequence of $( u , v , t )$ triplets sorted in chronological order, while the three SFT-based methods describe the temporal structure of the target node using a multi-hop neighbor sequence obtained through breadthfirst search (BFS). Taking the prediction of node $u _ { 4 } \mathrm { ^ { * } s }$ property at time $t _ { 7 }$ in Figure 6 as an exam-

Table 6: Structural Sequences of Node $u _ { 4 }$ at Time $t _ { 7 }$ by ICL and SFT Methods (Based on Figure 6)
<table><tr><td>Methods</td><td>Sequences of temporal structure</td></tr><tr><td>ICL</td><td> $( u _ { 1 } , u _ { 2 } , t _ { 1 } ) , ( u _ { 1 } , u _ { 3 } , t _ { 2 } ) , ( u _ { 2 } , u _ { 5 } , t _ { 3 } ) ,$   $( u _ { 3 } , u _ { 6 } , t _ { 4 } ) , ( u _ { 1 } , u _ { 4 } , t _ { 5 } ) , ( u _ { 4 } , u _ { 5 } , t _ { 6 } )$ </td></tr><tr><td>SFT</td><td> $( u _ { 1 } , u _ { 5 } , u _ { 2 } , u _ { 3 } )$ </td></tr></table>

ple, the structural sequences generated by ICL and SFT approaches are summarized in Table 6.

Table 7: Differences between three spatial-temporal prompting strategies.
<table><tr><td>Method</td><td>Prompt</td></tr><tr><td>TGTalker</td><td>Analyze the historical interactions to identify patterns that might indicate a future interaction. Consider both the relationships between users and items and the timing of past interactions within the historical interactions to make your prediction.</td></tr><tr><td>DST-v1</td><td>Think about structure and then time: First analyze the structural information within the historical interactions to understand the relationships between users and items, then consider the temporal patterns to predict the future connection.</td></tr><tr><td>DST-v2</td><td>Think about time and then structure: First analyze the temporal sequence of interactions in the historical data to identify patterns over time, then examine the structural information within the historical interactions to predict the future connection.</td></tr></table>

Structural Information. As shown by the serialization strategies in Table 6, ICL-based methods linearize the temporal structure into a sequence of (u, v, t) triplets, encoding a linear structural representation. This representation is simple and direct, but it requires LLMs to infer temporal structural patterns from the sequential triplets to perform the two temporal graph tasks. In contrast, SFT-based methods represent the temporal structure of the target node through its neighbor sequences, obtained via BFS, which encode explicit hierarchical structural information. This hierarchical information facilitates better structural comprehension by the LLM.

Learning and Trainable Parameters. SFT-based and ICL-based methods employ distinct strategies for LLM training or tuning.

• SFT-based methods fine-tune LLMs to learn structural knowledge for better downstream task performance. To improve computational efficiency, SFT methods train a projector layer to align temporal structural knowledge with the token embedding space. Specifically, LlaGA-ND and GraphGPT convert the neighbor node sequences into initial text embeddings. LlaGA-HO converts the structural knowledge obtained by aggregating multi-hop neighbor information through message passing from the target node’s embedding. In addition, GraphGPT further fine-tunes the embedding layer of the LLM itself.

• ICL-based methods do not require training orfine-tuning the LLM. Instead, they leverage the LLM’s pre-trained knowledge and reasoning capabilities, guiding it to understand temporal structural patterns via proposed instructions. To thoroughly explore the LLM’s ability to learn temporal graph structural knowledge through ICL, we adopt the Disentangled Spatial-Temporal Thoughts prompting technique from LLM4DyG [42] to TGTalker [11], which encourages the LLM to separately process spatial and temporal information. Specifically, the three variants are: a) TGTalker that jointly processes spatial and temporal information; b) DST-v1 that processes structural information first, followed by temporal information; and c) DST-v2, which processes temporal information first, then structural information. The differences among these three prompting strategies are summarized in Table 7.

Adaptation of SFT-based Methods. It is important to note that the three SFT methods were originally designed for static graphs. We have adapted them to the temporal graph setting, but due to the increased complexity of temporal graphs, these adaptations cannot fully capture all temporal graph properties. Our adaptations are as follows:

• For all methods, the neighbor sequences of the target node are time-sensitive. For example, in Figure 6, when predicting the property of node $u _ { 4 }$ at time $t _ { 7 } .$ its first-order neighbors only include $u _ { 1 }$ and $u _ { 5 } .$ , but not $u _ { 6 }$

• Both LlaGA-HO and GraphGPT require a message-passing process during their Text-Graph Grounding phase, which depends on known graph structures. To accommodate this, we construct a static graph structure by aggregating interactions from the training set, ignoring their timestamps.

• Due to computational constraints, we omit the Self-Supervised Instruction Tuning phase in GraphGPT’s training pipeline while retaining the other two phases.

Table 8: Prompts of LLM-as-Predictor methods for predicting whether node $u _ { 4 }$ will form a link with node $u _ { 6 }$ at time $t _ { 7 } ,$ based on the example in Figure 6
<table><tr><td>Method</td><td>Prompt</td></tr><tr><td>LlaGA-ND</td><td>Given two node-centered subgraphs:  $\langle u _ { 1 } , u _ { 5 } , u _ { 2 } , u _ { 3 } \rangle$  and  $\langle u _ { 3 } , u _ { 1 } \rangle$  , we need to predict whether these two nodes connect with each other. Please tell me whether two center nodes in the subgraphs should connect to each other.</td></tr><tr><td>LlaGA-HO</td><td>Given two node-centered subgraphs:  $\langle h o p _ { 0 } , h o p _ { 1 } , h o p _ { 2 } \rangle$  and  $\langle h o p _ { 0 } , h o p _ { 1 } , h o p _ { 2 } \rangle$  we need to predict whether these two nodes con- nect with each other. Please tell me whether two center nodes in the subgraphs should connect to each other.</td></tr><tr><td>GraphGPT</td><td>Given a sequence of graph tokens:  $\langle u _ { 1 } , u _ { 5 } , u _ { 2 } , u _ { 3 } \rangle$  that constitute a user-item review subgraph, where the first token represents the central node (the user), and the remaining nodes represent the central node&#x27;s first- and second-order neighbors. The first-order neighbors are the items that the central node has reviewed and the second-order neighbors are other users who have reviewed the same items. The other sequence of graph tokens:  $\langle u _ { 3 } , u _ { 1 } \rangle$  , where the first token corresponds to the center node (the item), and the remaining tokens represent the item&#x27;s first- and second-order neighbors. The first-order neighbors are the users who have reviewed the item and the second-order neighbors are the items that those users also have reviewed. If the connections between nodes represent the review relationships between users and items, are these two central nodes connected? Give me a direct answer of  $\mathbf { \dot { y } e s } ^ { \prime } \mathbf { o r } \mathbf { \dot { n } o } ^ { \prime }$ </td></tr></table>

Finally, in Table 8 and Table 9, we provide detailed prompts for all methods on the two tasks — temporal link prediction and temporal node classification — using the example of predicting whether there is a connection between $u _ { 4 }$ and $u _ { 6 }$ at time $t _ { 7 }$ , and predicting the label of $u _ { 4 }$ at time $t _ { 7 }$ respectively, as illustrated in Figure 6.

## C Experiments Details

## C.1 Setup Details

For the TGNN-Predictor methods, we primarily adopt the DyGLib $\mathrm { c o d e b a s e } ^ { 2 }$ . Following their workflow, we first perform a thorough grid search on the hyperparameters of nine traditional methods based on the temporal link prediction task to identify their optimal settings. After training the best-performing models on the temporal link prediction task, we use them as initial checkpoints to continue training for the temporal node classification task.

For the temporal link prediction task, we use supervised binary cross-entropy loss as the objective function. For the temporal node classification task, we treat each label in the FOOD and IMDB multi-label datasets as an independent binary classification problem, applying a weighted binary cross-entropy loss to account for class imbalance. For the other three multi-class datasets, we use a weighted cross-entropy loss as the objective function for the same reason.

For the LLM-Predictor methods, we follow the settings in their original papers as closely as possible, making appropriate adjustments to ensure they could be properly applied to temporal graphs. Specifically, for LlaGA-ND and LlaGA-HO, we consistently set the learning rate to 2e-5 and the batch size to 16 for all models, and trained them for one epoch on each dataset. For LlaGA-HO, since multi-hop embeddings of target nodes require the graph structure, we constructed a static graph by removing timestamp information from edges in the training set, which was then used as the graph structure during embedding computation.

For GraphGPT, due to the complexity of temporal graphs, we retained only two out of its original three pipeline stages: Structural Information Encoding with Text-Graph Grounding and Task-Specific Instruction Fine-tuning. Considering the higher computational cost of Task-Specific Instruction

Table 9: Prompts of LLM-as-Predictor methods for predicting the label of node $u _ { 4 }$ at time $t _ { 7 } ,$ based on the example in Figure 6.
<table><tr><td>Method</td><td>Prompt</td></tr><tr><td> $\mathrm { L L a G A – N D }$ </td><td>Given a node-centered graph:  $\langle u _ { 1 } , u _ { 5 } , u _ { 2 } , u _ { 3 } \rangle$  , where the center node repre- sents a user and the other nodes represent the user&#x27;s historical interactions, including  $1 ^ { \mathrm { s t } }$  -order neighbors (items that the user has reviewed) and  $2 ^ { \mathrm { n d } }$  -order neighbors (other users who have reviewed the same items), classify the center node into one or more of the following 9 interest categories: Side Dishes &amp; Vegetables, Dietary &amp; Specialty, Soups &amp; Stews, Cuisine &amp; Regional, Desserts &amp; Sweets, Baking &amp; Breads, Beverages &amp; Snacks, Sauces &amp; Condiments, Main Dishes.</td></tr><tr><td>LLaGA-HO</td><td>Given a node-centered graph:  $\langle h o p _ { 0 } , h o p _ { 1 } , h o p _ { 2 } \rangle$  , where the center node represents a user and the other nodes represent the user&#x27;s historical interactions, including 1st-order neighbors (items that the user has reviewed) and  $2 ^ { \mathrm { n d } } .$  -order neighbors (other users who have reviewed the same items), classify the center node into one or more of the following 9 interest categories: Side Dishes &amp; Vegetables, Dietary &amp; Specialty, Soups &amp; Stews, Cuisine &amp; Regional, Desserts &amp; Sweets, Baking &amp; Breads, Beverages &amp; Snacks, Sauces &amp; Condiments, Main Dishes.</td></tr><tr><td>GraphGPT</td><td>Given a user-item review graph:  $\langle u _ { 1 } , u _ { 5 } , u _ { 2 } , u _ { 3 } \rangle$  where the Oth node is the target user, and the other nodes are its first- or second-order neighbors. The first- order neighbors are the items that the user has reviewed and the second-order neighbors are other users who have reviewed the same items. Classify the target user&#x27;s interest into one or more of the following 9 categories: Side Dishes &amp; Vegetables, Dietary &amp; Specialty, Soups &amp; Stews, Cuisine &amp; Regional, Desserts &amp; Sweets, Baking &amp; Breads, Beverages &amp; Snacks, Sauces &amp; Condiments, Main Dishes.</td></tr></table>

Table 10: Detailed main temporal link prediction results on small-scale datasets. Results are averaged over three independent runs (in %).
<table><tr><td rowspan="2">Methods</td><td colspan="3">FOOD</td><td colspan="3">IMDB</td><td colspan="3">Librarything</td></tr><tr><td> $\mathbf { A P }$ </td><td>AUROC</td><td>MRR</td><td>AP</td><td>AUROC</td><td>MRR</td><td> $\mathbf { A P }$ </td><td>AUROC</td><td>MRR</td></tr><tr><td>JODIE</td><td> $7 2 . 9 6 { \pm } 1 . 8 8 $ </td><td> $7 1 . 9 5 { \scriptstyle \pm 0 . 6 9 }$ </td><td> $4 0 . 0 9 { \scriptstyle \pm 0 . 8 3 }$ </td><td> $6 0 . 3 0 { \scriptstyle \pm 2 . 3 7 }$ </td><td> $6 5 . 7 4 { \scriptstyle \pm 3 . 2 0 }$ </td><td> $2 5 . 5 7 { \scriptstyle \pm 1 . 6 9 }$ </td><td> $7 5 . 7 6 { \scriptstyle \pm 0 . 6 3 }$ </td><td> $7 3 . 6 2 { \scriptstyle \pm 0 . 7 5 }$ </td><td> $4 4 . 1 4 { \scriptstyle \pm 0 . 9 7 }$ </td></tr><tr><td>DyRep</td><td> $7 2 . 8 3 { \pm } 0 . 8 8 $ </td><td> $6 9 . 0 1 { \scriptstyle \pm 1 . 5 3 }$ </td><td> $4 0 . 5 5 { \scriptstyle \pm 0 . 4 4 }$ </td><td> $6 4 . 7 9 { \scriptstyle \pm 1 . 3 0 }$ </td><td> $6 9 . 7 9 { \scriptstyle \pm 2 . 0 3 }$ </td><td> $2 7 . 0 5 { \scriptstyle \pm 0 . 6 0 }$ </td><td> $7 3 . 8 1 { \scriptstyle \pm 1 . 5 5 }$ </td><td> $7 2 . 3 4 \pm 1 . 1 3$ </td><td> $4 4 . 1 3 { \scriptstyle \pm 0 . 7 3 }$ </td></tr><tr><td>TGAT</td><td> $5 4 . 7 9 { \scriptstyle \pm 1 . 2 5 }$ </td><td> $5 4 . 3 8 { \scriptstyle \pm 0 . 6 7 }$ </td><td> $2 0 . 1 4 \pm 1 . 3 4$ </td><td> $5 4 . 5 5 { \scriptstyle \pm 0 . 0 1 }$ </td><td> $5 7 . 8 5 { \scriptstyle \pm 0 . 0 5 }$ </td><td> $1 9 . 2 3 { \scriptstyle \pm 0 . 1 0 }$ </td><td> $6 4 . 2 0 { \scriptstyle \pm 0 . 8 9 }$ </td><td> $6 5 . 6 8 { \scriptstyle \pm 0 . 9 6 }$ </td><td> $2 7 . 7 4 \pm 1 . 1 7$ </td></tr><tr><td>TGN</td><td> $7 7 . 7 4 { \scriptstyle \pm 0 . 0 5 }$ </td><td> $7 5 . 7 4 \pm 0 . 2 8$ </td><td> $4 4 . 1 1 { \scriptstyle \pm 0 . 1 1 }$ </td><td> $4 5 . 4 0 { \scriptstyle \pm 0 . 8 9 }$ </td><td> $4 8 . 7 9 2 1 . 6 6 $ </td><td> $1 0 . 8 0 { \scriptstyle \pm 0 . 5 8 }$ </td><td> $8 2 . 6 0 { \scriptstyle \pm 0 . 2 2 }$ </td><td> $8 2 . 0 2 { \scriptstyle \pm 0 . 0 9 }$ </td><td> $5 2 . 2 9 { \scriptstyle \pm 0 . 4 6 }$ </td></tr><tr><td> $\mathbf { C A W N }$ </td><td> $5 1 . 7 1 { \pm } 0 . 3 8 $ </td><td> $5 3 . 9 6 { \scriptstyle \pm 0 . 2 3 }$ </td><td> $1 7 . 4 3 { \scriptstyle \pm 0 . 3 8 }$ </td><td> $5 4 . 0 0 { \scriptstyle \pm 1 . 1 7 }$ </td><td> $5 5 . 6 0 { \scriptstyle \pm 2 . 7 5 }$ </td><td> $1 9 . 5 2 { \scriptstyle \pm 0 . 4 9 }$ </td><td> $6 2 . 0 1 { \scriptstyle \pm 0 . 2 1 }$ </td><td> $6 3 . 2 1 \pm 0 . 0 7$ </td><td> $2 6 . 7 0 { \scriptstyle \pm 0 . 0 8 }$ </td></tr><tr><td>TCL</td><td> $4 9 . 1 2 { \scriptstyle \pm 0 . 7 9 }$ </td><td> $5 0 . 9 8 { \scriptstyle \pm 1 . 2 3 }$ </td><td> $1 5 . 4 3 { \scriptstyle \pm 0 . 5 5 }$ </td><td> $5 5 . 7 8 { \scriptstyle \pm 0 . 4 9 }$ </td><td> $5 8 . 8 3 { \scriptstyle \pm 0 . 4 4 }$ </td><td> $2 0 . 1 8 { \scriptstyle \pm 0 . 2 1 }$ </td><td> $5 4 . 5 1 { \scriptstyle \pm 4 . 0 2 }$ </td><td> $5 6 . 6 3 { \scriptstyle \pm 2 . 9 1 }$ </td><td> $1 8 . 8 8 { \scriptstyle \pm 4 . 1 4 }$ </td></tr><tr><td> $\mathbf { G r a p h M i x e r }$ </td><td> $6 6 . 9 8 { \scriptstyle \pm 1 . 7 1 }$ </td><td> $6 5 . 7 8 { \scriptstyle \pm 0 . 9 3 }$ </td><td> $3 1 . 6 2 { \scriptstyle \pm 1 . 9 8 }$ </td><td> $5 7 . 0 1 { \scriptstyle \pm 0 . 0 3 }$ </td><td> $6 1 . 6 0 { \scriptstyle \pm 0 . 0 9 }$ </td><td> $2 1 . 2 0 { \scriptstyle \pm 0 . 0 5 }$ </td><td> $6 2 . 1 3 { \scriptstyle \pm 1 . 4 5 }$ </td><td> $6 3 . 1 6 { \scriptstyle \pm 2 . 4 4 }$ </td><td> $2 6 . 5 2 { \scriptstyle \pm 1 . 2 6 }$ </td></tr><tr><td>DyĠFormer</td><td> $5 9 . 4 8 { \scriptstyle \pm 0 . 2 4 }$ </td><td> $5 7 . 2 6 { \scriptstyle \pm 0 . 5 2 }$ </td><td> $2 4 . 5 1 { \scriptstyle \pm 0 . 1 3 }$ </td><td> $5 5 . 3 2 { \scriptstyle \pm 0 . 1 1 }$ </td><td> $5 8 . 4 2 { \scriptstyle \pm 0 . 1 0 }$ </td><td> $2 0 . 2 4 \pm 0 . 0 2$ </td><td> $5 4 . 5 3 { \scriptstyle \pm 2 . 1 6 }$ </td><td> $5 4 . 3 5 { \scriptstyle \pm 3 . 1 4 }$ </td><td> $1 9 . 7 5 { \scriptstyle \pm 2 . 3 4 }$ </td></tr><tr><td>FreeDyG</td><td> $7 2 . 4 0 { \scriptstyle \pm 0 . 3 4 }$ </td><td> $6 9 . 6 0 { \scriptstyle \pm 0 . 3 7 }$ </td><td> $3 8 . 5 6 { \scriptstyle \pm 0 . 5 2 }$ </td><td> $5 7 . 5 3 { \scriptstyle \pm 0 . 8 9 }$ </td><td> $6 2 . 1 6 { \pm } 0 . 9 3$ </td><td> $2 1 . 7 8 { \scriptstyle \pm 0 . 8 0 }$ </td><td> $6 6 . 0 8 { \scriptstyle \pm 0 . 2 6 }$ </td><td> $6 5 . 1 3 { \scriptstyle \pm 1 . 7 0 }$ </td><td> $3 0 . 9 0 { \scriptstyle \pm 0 . 1 3 }$ </td></tr><tr><td> $\mathrm { L K D 4 D y T A G }$ </td><td> $6 6 . 6 3 { \scriptstyle \pm 0 . 9 5 }$ </td><td> $6 3 . 6 2 \pm 0 . 9 5$ </td><td> $3 1 . 3 0 { \scriptstyle \pm 1 . 0 5 }$ </td><td> $5 6 . 0 4 { \scriptstyle \pm 0 . 9 2 }$ </td><td> $5 9 . 1 3 { \scriptstyle \pm 0 . 5 3 }$ </td><td> $2 0 . 5 0 { \scriptstyle \pm 0 . 7 6 }$ </td><td> $5 9 . 9 4 { \scriptstyle \pm 0 . 4 9 }$ </td><td> $6 2 . 1 7 { \scriptstyle \pm 0 . 3 4 }$ </td><td> $2 3 . 3 7 { \scriptstyle \pm 0 . 3 5 }$ </td></tr><tr><td>CROSS</td><td> $5 8 . 1 5 { \scriptstyle \pm 0 . 1 3 }$ </td><td> $5 6 . 4 8 { \scriptstyle \pm 0 . 2 4 }$ </td><td> $2 3 . 1 7 { \scriptstyle \pm 0 . 2 3 }$ </td><td> $5 5 . 9 8 { \scriptstyle \pm 0 . 0 5 }$ </td><td> $5 9 . 1 4 { \scriptstyle \pm 0 . 1 2 }$ </td><td> $2 0 . 6 2 { \scriptstyle \pm 0 . 0 6 }$ </td><td> $5 9 . 0 4 { \scriptstyle \pm 0 . 8 3 }$ </td><td> $6 0 . 2 5 { \scriptstyle \pm 0 . 9 4 }$ </td><td> $2 4 . 1 6 { \scriptstyle \pm 0 . 5 3 }$ </td></tr><tr><td>TGTalker</td><td> $5 1 . 0 4 { \scriptstyle \pm 0 . 0 2 }$ </td><td> $5 2 . 0 2 { \scriptstyle \pm 0 . 0 3 }$ </td><td> $1 6 . 2 4 2 0 . 1 5$ </td><td> $5 0 . 2 2 { \scriptstyle \pm 0 . 0 4 }$ </td><td> $5 0 . 4 3 { \scriptstyle \pm 0 . 0 7 }$ </td><td> $1 5 . 1 2 { \scriptstyle \pm 0 . 1 0 }$ </td><td> $5 0 . 6 3 { \scriptstyle \pm 0 . 0 1 }$ </td><td> $5 1 . 2 4 { \scriptstyle \pm 0 . 0 4 }$ </td><td> $1 5 . 5 4 { \scriptstyle \pm 0 . 1 2 }$ </td></tr><tr><td>DST-v1</td><td> $5 1 . 3 1 { \pm } 0 . 0 1 $ </td><td> $5 2 . 5 3 { \scriptstyle \pm 0 . 0 3 }$ </td><td> $1 6 . 5 8 { \scriptstyle \pm 0 . 1 8 }$ </td><td> $5 0 . 2 6 { \scriptstyle \pm 0 . 1 0 }$ </td><td> $5 0 . 5 2 { \scriptstyle \pm 0 . 2 0 }$ </td><td> $1 5 . 1 8 { \scriptstyle \pm 0 . 2 2 }$ </td><td> $5 0 . 7 1 { \scriptstyle \pm 0 . 0 2 }$ </td><td> $5 1 . 3 9 { \scriptstyle \pm 0 . 0 6 }$ </td><td> $1 5 . 6 8 { \scriptstyle \pm 0 . 1 5 }$ </td></tr><tr><td>DST-v2</td><td> $5 1 . 3 3 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $5 2 . 5 7 { \scriptstyle \pm 0 . 0 1 }$ </td><td> $1 6 . 6 0 { \scriptstyle \pm 0 . 1 5 }$ </td><td> $5 0 . 3 5 { \scriptstyle \pm 0 . 1 0 }$ </td><td> $5 0 . 7 0 { \scriptstyle \pm 0 . 1 9 }$ </td><td> $1 5 . 2 5 { \scriptstyle \pm 0 . 1 8 }$ </td><td> $5 0 . 6 1 \pm 0 . 0 1$ </td><td> $5 1 . 1 9 { \scriptstyle \pm 0 . 0 2 }$ </td><td> $1 5 . 5 1 { \scriptstyle \pm 0 . 0 8 }$ </td></tr><tr><td>Llaga-ND</td><td> $5 3 . 1 0 { \scriptstyle \pm 0 . 3 6 }$ </td><td> $5 5 . 5 2 { \scriptstyle \pm 0 . 5 7 }$ </td><td> $1 8 . 8 5 { \scriptstyle \pm 0 . 5 5 }$ </td><td> $5 2 . 0 1 { \scriptstyle \pm 0 . 7 5 }$ </td><td> $5 3 . 5 5 { \pm } 1 . 1 9$ </td><td> $1 7 . 5 0 { \scriptstyle \pm 0 . 8 5 }$ </td><td> $4 9 . 9 8 { \scriptstyle \pm 0 . 0 4 }$ </td><td> $4 9 . 9 5 { \scriptstyle \pm 0 . 0 7 }$ </td><td> $1 4 . 8 5 { \scriptstyle \pm 0 . 1 0 }$ </td></tr><tr><td>Llaga-HO</td><td> $6 1 . 9 5 { \scriptstyle \pm 1 . 3 2 }$ </td><td> $6 5 . 6 7 \pm 1 . 4 7$ </td><td> $2 9 . 5 0 { \scriptstyle \pm 1 . 8 5 }$ </td><td> $5 4 . 3 7 { \scriptstyle \pm 0 . 8 2 }$ </td><td> $5 7 . 3 4 \pm 1 . 0 6$ </td><td> $2 0 . 1 5 { \scriptstyle \pm 1 . 2 5 }$ </td><td> $4 9 . 9 7 { \scriptstyle \pm 0 . 0 4 }$ </td><td> $4 9 . 9 5 { \scriptstyle \pm 0 . 0 7 }$ </td><td> $1 4 . 8 3 { \scriptstyle \pm 0 . 1 1 }$ </td></tr><tr><td>GraphGPT</td><td> $5 5 . 0 9 { \scriptstyle \pm 0 . 4 3 }$ </td><td> $5 7 . 9 5 { \scriptstyle \pm 1 . 0 2 }$ </td><td> $2 5 . 0 5 { \scriptstyle \pm 0 . 8 5 }$ </td><td> $5 1 . 8 0 { \scriptstyle \pm 0 . 1 8 }$ </td><td> $5 3 . 1 9 { \scriptstyle \pm 0 . 2 9 }$ </td><td> $1 7 . 2 0 { \scriptstyle \pm 0 . 4 5 }$ </td><td> $4 9 . 9 6 { \scriptstyle \pm 0 . 0 4 }$ </td><td> $4 9 . 8 9 { \scriptstyle \pm 0 . 1 1 }$ </td><td> $1 4 . 8 1 { \scriptstyle \pm 0 . 1 8 }$ </td></tr></table>

Tuning, we set the batch size to 4 for the Amazon-Kindle dataset and 8 for the other datasets, to fully utilize the 48GB memory of an NVIDIA A6000 GPU. The learning rate was kept at $_ { 2 \mathrm { e } - 3 . }$ , and we set the maximum output length of the LLM to 4096 tokens to accommodate as much structural information as possible. As with LlaGA, this model was also trained for one epoch on each dataset.

## C.2 More Experiment Results and Analysis

Detailed Main Results. Regarding the TLP task, Tables 10 and 11 supplement Table 2 by providing results for an extended set of evaluation metrics.

Table 11: Detailed main temporal link prediction results on large-scale datasets. Results are averaged over three independent runs (in %).
<table><tr><td rowspan="2">Methods</td><td colspan="3">Beeradvocate</td><td colspan="3">Amazon-Kindle</td><td colspan="3">Ratebeer</td></tr><tr><td>AP</td><td>AUROC</td><td>MRR</td><td>AP</td><td>AUROC</td><td>MRR</td><td>AP</td><td>AUROC</td><td>MRR</td></tr><tr><td>JODIE</td><td>79.75±11.63</td><td>80.93±9.53</td><td>57.49±16.89</td><td>82.81±2.00</td><td>81.86±1.69</td><td>60.05±3.18</td><td>92.62±0.18</td><td>91.56±0.36</td><td>74.20±0.10</td></tr><tr><td>DyRep</td><td> $8 9 . 6 1 { \scriptstyle \pm 0 . 2 0 }$ </td><td>87.01±0.40</td><td>72.50±0.16</td><td>78.35±2.63</td><td> $7 6 . 6 3 { \scriptstyle \pm 2 . 6 9 }$ </td><td> $5 6 . 9 9 { \scriptstyle \pm 3 . 2 3 }$ </td><td> $8 5 . 8 7 { \scriptstyle \pm 0 . 3 3 }$ </td><td> $8 3 . 8 4 { \pm } 1 . 1 8$ </td><td> $6 6 . 1 1 { \scriptstyle \pm 0 . 7 1 }$ </td></tr><tr><td>TGAT</td><td> $5 9 . 3 7 { \scriptstyle \pm 3 . 5 9 }$ </td><td>66.58±2.13</td><td> $2 7 . 9 0 { \scriptstyle \pm 4 . 6 4 }$ </td><td> $7 9 . 0 0 { \scriptstyle \pm 1 . 9 7 }$ </td><td> $8 1 . 1 2 { \scriptstyle \pm 1 . 3 2 }$ </td><td> $4 4 . 3 7 { \scriptstyle \pm 2 . 7 1 }$ </td><td> $5 8 . 9 3 { \scriptstyle \pm 4 . 2 0 } $ </td><td> $6 4 . 3 8 { \scriptstyle \pm 2 . 2 9 }$ </td><td> $2 8 . 6 3 { \scriptstyle \pm 3 . 6 6 }$ </td></tr><tr><td>TGN</td><td> $8 3 . 4 6 { \pm } 1 . 6 8 $ </td><td> $8 3 . 2 5 { \scriptstyle \pm 1 . 0 6 }$ </td><td> $5 9 . 3 5 { \scriptstyle \pm 2 . 4 8 }$ </td><td> $7 1 . 7 7 { \scriptstyle \pm 2 . 9 0 }$ </td><td> $7 9 . 9 2 { \scriptstyle \pm 1 . 4 9 }$ </td><td> $3 9 . 6 7 { \scriptstyle \pm 3 . 7 2 }$ </td><td> $9 1 . 5 5 { \scriptstyle \pm 0 . 2 4 }$ </td><td> $9 1 . 4 1 { \scriptstyle \pm 0 . 2 4 }$ </td><td> $7 2 . 0 0 { \scriptstyle \pm 0 . 3 1 }$ </td></tr><tr><td>CAWN</td><td> $6 6 . 1 5 { \scriptstyle \pm 0 . 8 5 }$ </td><td> $7 1 . 2 9 { \scriptstyle \pm 0 . 0 5 }$ </td><td> $3 5 . 1 2 { \scriptstyle \pm 0 . 2 0 }$ </td><td> $7 9 . 3 4 { \pm } 1 . 5 0 $ </td><td> $8 0 . 7 9 { \scriptstyle \pm 1 . 2 2 }$ </td><td> $4 4 . 9 2 { \scriptstyle \pm 2 . 0 4 }$ </td><td> $5 9 . 9 4 { \scriptstyle \pm 2 . 4 0 }$ </td><td> $6 5 . 7 8 { \scriptstyle \pm 0 . 6 7 }$ </td><td>29.86±1.58</td></tr><tr><td>TCL</td><td> $5 5 . 4 0 { \scriptstyle \pm 0 . 1 8 }$ </td><td> $6 4 . 6 2 { \scriptstyle \pm 0 . 2 9 }$ </td><td> $2 2 . 3 7 { \pm } 1 . 0 1$ </td><td> $7 9 . 6 8 { \pm } 1 . 5 1 $ </td><td> $8 0 . 9 6 { \scriptstyle \pm 0 . 9 5 }$ </td><td> $4 4 . 7 4 { \scriptstyle \pm 2 . 6 0 }$ </td><td> $5 7 . 6 1 \pm 3 . 7 7$ </td><td> $6 2 . 6 3 { \scriptstyle \pm 2 . 3 9 }$ </td><td> $2 4 . 9 4 { \scriptstyle \pm 4 . 8 5 }$ </td></tr><tr><td>GraphMixer</td><td> $8 6 . 0 2 { \scriptstyle \pm 0 . 2 4 }$ </td><td> $8 5 . 5 7 { \scriptstyle \pm 0 . 2 9 }$ </td><td> $6 0 . 5 1 { \scriptstyle \pm 0 . 8 4 }$ </td><td> $8 1 . 5 3 { \scriptstyle \pm 0 . 2 8 }$ </td><td> $8 1 . 8 7 { \scriptstyle \pm 0 . 0 9 }$ </td><td> $4 8 . 8 1 { \scriptstyle \pm 0 . 4 4 }$ </td><td> $7 8 . 5 3 { \scriptstyle \pm 4 . 2 2 }$ </td><td> $7 8 . 0 5 { \scriptstyle \pm 4 . 0 6 }$ </td><td> $4 8 . 5 2 { \scriptstyle \pm 7 . 0 6 }$ </td></tr><tr><td>DyĠFormer</td><td> $5 0 . 5 3 { \scriptstyle \pm 0 . 0 8 }$ </td><td> $5 6 . 8 0 { \scriptstyle \pm 0 . 0 5 }$ </td><td> $1 6 . 3 4 { \scriptstyle \pm 0 . 7 1 }$ </td><td> $7 2 . 7 8 { \scriptstyle \pm 0 . 3 2 }$ </td><td> $7 4 . 7 1 { \scriptstyle \pm 0 . 0 9 }$ </td><td> $3 6 . 6 1 \pm 0 . 2 2$ </td><td> $5 3 . 2 8 { \scriptstyle \pm 0 . 9 1 }$ </td><td> $5 7 . 6 5 { \scriptstyle \pm 0 . 7 3 }$ </td><td> $2 3 . 0 9 { \scriptstyle \pm 1 . 3 3 }$ </td></tr><tr><td>FreeDyG</td><td> $8 2 . 9 3 { \scriptstyle \pm 6 . 7 5 }$ </td><td> $8 5 . 1 8 { \scriptstyle \pm 4 . 0 7 }$ </td><td> $5 8 . 8 6 { \scriptstyle \pm 8 . 5 6 }$ </td><td> $6 7 . 2 6 { \scriptstyle \pm 3 . 8 3 }$ </td><td> $7 1 . 2 5 { \scriptstyle \pm 1 . 9 9 }$ </td><td> $3 2 . 8 0 { \scriptstyle \pm 4 . 8 5 }$ </td><td> $6 9 . 4 9 { \scriptstyle \pm 3 . 3 0 }$ </td><td> $7 4 . 1 6 { \pm } 1 . 8 6$ </td><td>40.10±3.65</td></tr><tr><td>LKD4DyTAG</td><td> $7 9 . 4 2 { \pm } 1 . 1 5$ </td><td> $7 9 . 5 2 { \pm } 0 . 9 1 $ </td><td> $4 7 . 5 1 \pm 1 . 4 1$ </td><td> $8 0 . 6 4 \substack { \pm 0 . 4 6 }$ </td><td> $8 1 . 6 4 { \scriptstyle \pm 0 . 4 2 }$ </td><td> $4 6 . 4 4 { \scriptstyle \pm 0 . 5 9 }$ </td><td> $6 9 . 5 1 { \scriptstyle \pm 0 . 0 3 }$ </td><td> $7 0 . 8 5 { \scriptstyle \pm 0 . 0 5 }$ </td><td> $3 7 . 5 9 { \scriptstyle \pm 0 . 0 7 }$ </td></tr><tr><td>CROSS</td><td> $5 1 . 3 6 { \pm } 1 . 0 3 $ </td><td> $5 7 . 5 2 \pm 1 . 0 1 $ </td><td> $1 7 . 8 9 { \scriptstyle \pm 2 . 2 5 }$ </td><td> $8 7 . 2 5 { \scriptstyle \pm 0 . 1 3 }$ </td><td> $8 8 . 3 1 { \scriptstyle \pm 0 . 1 2 }$ </td><td> $5 6 . 9 9 { \scriptstyle \pm 0 . 3 8 }$ </td><td> $5 2 . 5 7 { \scriptstyle \pm 0 . 0 9 }$ </td><td> $5 7 . 1 9 { \scriptstyle \pm 0 . 2 1 }$ </td><td> $2 1 . 3 2 { \scriptstyle \pm 0 . 9 8 }$ </td></tr><tr><td>TGTalker</td><td>51.18±0.03</td><td>51.36±0.02</td><td>16.05±0.08</td><td>52.54±0.09</td><td>54.79±0.11</td><td>18.21±0.35</td><td> $5 1 . 1 1 { \scriptstyle \pm 0 . 0 7 }$ </td><td>52.15±0.06</td><td>16.10±0.14</td></tr><tr><td>DST-v1</td><td>51.34±0.04</td><td> $5 1 . 4 7 { \scriptstyle \pm 0 . 0 5 }$ </td><td>16.18±0.10</td><td>52.66±0.10</td><td> $5 4 . 8 5 { \scriptstyle \pm 0 . 0 9 }$ </td><td> $1 8 . 3 2 { \scriptstyle \pm 0 . 4 1 }$ </td><td> $5 1 . 2 3 { \scriptstyle \pm 0 . 0 4 }$ </td><td>51.26±0.03</td><td>16.15±0.09</td></tr><tr><td>DST-v2</td><td>51.21±0.01</td><td> $5 1 . 5 3 { \scriptstyle \pm 0 . 0 1 }$ </td><td> $1 6 . 1 2 { \scriptstyle \pm 0 . 0 5 }$ </td><td> $5 2 . 6 2 { \scriptstyle \pm 0 . 0 8 }$ </td><td> $5 4 . 8 1 { \scriptstyle \pm 0 . 0 8 }$ </td><td> $1 8 . 2 8 { \scriptstyle \pm 0 . 3 8 }$ </td><td> $5 1 . 0 5 { \scriptstyle \pm 0 . 0 1 }$ </td><td> $5 1 . 0 7 { \scriptstyle \pm 0 . 0 2 }$ </td><td>15.98±0.05</td></tr><tr><td>Llaga-ND</td><td> $4 9 . 9 7 { \scriptstyle \pm 0 . 0 3 }$ </td><td> $4 9 . 9 4 { \scriptstyle \pm 0 . 0 6 }$ </td><td> $1 4 . 8 2 { \scriptstyle \pm 0 . 1 2 }$ </td><td> $6 8 . 6 2 \pm 3 . 3 7$ </td><td> $7 2 . 8 5 { \scriptstyle \pm 3 . 5 8 }$ </td><td> $3 6 . 4 5 { \scriptstyle \pm 4 . 1 2 }$ </td><td> $5 0 . 0 0 { \scriptstyle \pm 0 . 0 2 }$ </td><td> $4 9 . 9 9 { \scriptstyle \pm 0 . 0 5 }$ </td><td>14.90±0.08</td></tr><tr><td>Llaga-HO</td><td> $5 0 . 0 0 { \scriptstyle \pm 0 . 0 5 }$ </td><td> $5 0 . 0 0 { \scriptstyle \pm 0 . 1 0 }$ </td><td> $1 4 . 9 1 { \scriptstyle \pm 0 . 1 5 }$ </td><td> $7 2 . 6 4 \pm 1 . 1 1$ </td><td> $7 6 . 6 3 { \scriptstyle \pm 1 . 5 4 }$ </td><td> $4 2 . 1 0 { \scriptstyle \pm 2 . 0 5 }$ </td><td> $5 0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $5 0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $1 5 . 0 7 { \scriptstyle \pm 0 . 1 0 }$ </td></tr><tr><td>GraphGPT</td><td> $5 6 . 8 0 { \scriptstyle \pm 6 . 8 0 }$ </td><td> $5 8 . 3 5 { \scriptstyle \pm 8 . 3 5 }$ </td><td> $2 2 . 1 5 { \scriptstyle \pm 7 . 5 0 }$ </td><td> $5 0 . 7 2 { \scriptstyle \pm 0 . 7 2 }$ </td><td> $5 1 . 1 7 { \scriptstyle \pm 1 . 1 7 }$ </td><td> $1 5 . 7 5 { \scriptstyle \pm 1 . 0 5 }$ </td><td> $5 6 . 0 3 { \scriptstyle \pm 6 . 0 3 }$ </td><td> $5 9 . 2 9 { \scriptstyle \pm 9 . 2 9 }$ </td><td> $2 1 . 8 0 { \scriptstyle \pm 8 . 1 5 }$ </td></tr></table>

Table 12: Temporal link prediction results on small-scale datasets under inductive settings. Results are averaged over three independent runs (in %).

$$
4 1 . 7 1 { \scriptstyle \pm 0 . 3 2 }
$$

$$
8 8 . 3 9 { \scriptstyle \pm 0 . 2 1 }
$$

$$
3 7 . 3 8 { \pm } 1 . 4 6 
$$

$$
5 4 . 7 5 { \scriptstyle \pm 0 . 6 5 }
$$

$$
5 4 . 3 0 { \scriptstyle \pm 0 . 2 3 }
$$

$$
8 7 . 4 7 { \scriptstyle \pm 0 . 3 5 }
$$

$$
8 4 . 4 7 { \scriptstyle \pm 0 . 0 8 }
$$

$$
8 6 . 7 9 { \scriptstyle \pm 0 . 4 9 }
$$

$$
2 0 . 1 2 { \scriptstyle \pm 0 . 6 7 }
$$

$$
7 3 . 7 7 { \scriptstyle \pm 0 . 4 4 }
$$

$$
8 5 . 7 2 { \scriptstyle \pm 0 . 6 0 }
$$

$$
7 2 . 1 4 \pm 0 . 4 6
$$

$$
7 4 . 5 3 { \scriptstyle \pm 0 . 2 1 }
$$

$$
8 2 . 4 5 { \scriptstyle \pm 0 . 1 6 }
$$

$$
8 2 . 9 6 { \scriptstyle \pm 0 . 2 1 }
$$

$$
3 8 . 4 3 { \scriptstyle \pm 0 . 3 0 }
$$

$$
7 3 . 7 5 { \scriptstyle \pm 0 . 3 0 }
$$

$$
7 1 . 3 1 { \scriptstyle \pm 0 . 9 5 }
$$

$$
7 9 . 6 6 \pm 1 . 9 2
$$

$$
5 3 . 0 6 { \scriptstyle \pm 0 . 4 1 }
$$

$$
7 1 . 9 4 { \scriptstyle \pm 0 . 9 3 }
$$

$$
5 4 . 3 3 { \scriptstyle \pm 0 . 5 0 }
$$

$$
7 9 . 2 3 { \scriptstyle \pm 2 . 3 9 }
$$

$$
1 8 . 7 4 2 0 . 3 5
$$

$$
3 3 . 9 7 { \scriptstyle \pm 1 . 2 8 }
$$

$$
8 3 . 3 8 { \scriptstyle \pm 0 . 0 3 }
$$

$$
6 3 . 6 4 { \scriptstyle \pm 0 . 9 5 }
$$

$$
5 0 . 4 0 { \scriptstyle \pm 0 . 7 5 }
$$

$$
8 4 . 3 5 { \scriptstyle \pm 0 . 2 0 }
$$

$$
5 0 . 7 1 { \scriptstyle \pm 1 . 1 4 }
$$

$$
6 6 . 1 2 { \scriptstyle \pm 0 . 2 7 }
$$

$$
1 6 . 8 5 { \scriptstyle \pm 0 . 4 2 }
$$

$$
6 8 . 7 4 \pm 1 . 2 5
$$

$$
6 7 . 3 9 { \scriptstyle \pm 3 . 4 6 }
$$

$$
6 9 . 2 1 { \scriptstyle \pm 1 . 0 6 }
$$

$$
6 8 . 2 0 { \scriptstyle \pm 1 . 3 9 }
$$

$$
6 8 . 4 1 \pm 1 . 5 4
$$

$$
3 1 . 9 6 { \scriptstyle \pm 1 . 6 6 }
$$

$$
6 7 . 1 9 { \scriptstyle \pm 1 . 1 0 }
$$

$$
3 3 . 0 8 { \scriptstyle \pm 1 . 6 8 }
$$

$$
8 6 . 6 7 { \scriptstyle \pm 0 . 0 3 }
$$

$$
6 4 . 0 2 { \scriptstyle \pm 2 . 3 4 }
$$

$$
8 4 . 9 1 { \scriptstyle \pm 0 . 0 2 }
$$

$$
6 5 . 1 4 { \scriptstyle \pm 1 . 6 2 }
$$

$$
5 6 . 3 7 { \scriptstyle \pm 0 . 4 4 }
$$

$$
2 6 . 3 5 { \scriptstyle \pm 2 . 9 6 }
$$

$$
2 3 . 0 3 { \scriptstyle \pm 0 . 1 0 }
$$

$$
6 0 . 0 7 { \scriptstyle \pm 0 . 0 3 }
$$

$$
6 2 . 3 3 { \scriptstyle \pm 0 . 3 8 }
$$

$$
7 7 . 2 5 { \scriptstyle \pm 1 . 7 3 }
$$

$$
6 5 . 8 8 { \scriptstyle \pm 0 . 0 9 }
$$

$$
7 6 . 9 5 { \scriptstyle \pm 2 . 0 1 }
$$

$$
4 3 . 7 8 { \scriptstyle \pm 2 . 0 3 }
$$

$$
7 0 . 3 8 { \scriptstyle \pm 0 . 1 3 }
$$

$$
2 5 . 7 1 { \scriptstyle \pm 0 . 4 8 }
$$

$$
3 8 . 4 7 { \scriptstyle \pm 0 . 3 5 }
$$

$$
7 2 . 6 5 { \pm } 0 . 1 8
$$

$$
8 6 . 9 0 { \scriptstyle \pm 0 . 0 8 }
$$

$$
6 1 . 1 6 { \scriptstyle \pm 3 . 1 6 }
$$

$$
6 0 . 9 3 { \scriptstyle \pm 3 . 6 5 }
$$

$$
2 4 . 6 3 { \scriptstyle \pm 2 . 8 0 }
$$

$$
8 5 . 1 5 { \scriptstyle \pm 0 . 1 0 }
$$

$$
6 2 . 1 3 { \scriptstyle \pm 0 . 6 7 }
$$

$$
5 9 . 7 4 { \scriptstyle \pm 0 . 6 5 }
$$

$$
2 6 . 7 4 { \scriptstyle \pm 0 . 6 1 }
$$

$$
4 7 . 8 1 { \scriptstyle \pm 0 . 9 8 }
$$

$$
6 1 . 8 1 { \scriptstyle \pm 0 . 7 6 }
$$

$$
7 9 . 8 1 { \scriptstyle \pm 0 . 7 1 }
$$

$$
7 8 . 5 4 \pm 1 . 3 2
$$

$$
6 5 . 7 1 { \scriptstyle \pm 0 . 2 2 }
$$

$$
5 5 . 3 9 { \scriptstyle \pm 0 . 2 8 }
$$

$$
2 1 . 9 5 { \scriptstyle \pm 0 . 3 5 }
$$

$$
6 2 . 2 7 { \scriptstyle \pm 1 . 1 8 }
$$

$$
2 9 . 9 8 { \scriptstyle \pm 0 . 0 3 }
$$

$$
6 6 . 9 7 { \scriptstyle \pm 0 . 4 7 }
$$

$$
6 7 . 4 4 { \scriptstyle \pm 0 . 1 9 }
$$

$$
6 8 . 0 6 { \scriptstyle \pm 0 . 2 4 }
$$

$$
2 5 . 1 6 { \pm } 1 . 3 9
$$

$$
6 5 . 7 9 2 0 . 5 1
$$

$$
6 6 . 2 4 \pm 0 . 5 3
$$

$$
2 8 . 8 2 { \scriptstyle \pm 0 . 4 1 }
$$

$$
5 2 . 0 2 { \scriptstyle \pm 0 . 0 3 }
$$

$$
5 0 . 2 2 { \scriptstyle \pm 0 . 0 4 }
$$

$$
5 1 . 3 1 { \pm } 0 . 0 1 
$$

$$
5 0 . 4 3 { \scriptstyle \pm 0 . 0 7 }
$$

$$
5 2 . 5 3 { \scriptstyle \pm 0 . 0 3 }
$$

$$
5 1 . 2 4 { \scriptstyle \pm 0 . 0 4 }
$$

$$
5 0 . 6 3 { \scriptstyle \pm 0 . 0 1 }
$$

$$
1 6 . 5 5 { \scriptstyle \pm 0 . 1 5 }
$$

$$
5 0 . 2 6 { \scriptstyle \pm 0 . 1 0 }
$$

$$
5 1 . 3 3 { \scriptstyle \pm 0 . 0 0 }
$$

$$
5 0 . 5 2 { \scriptstyle \pm 0 . 2 0 }
$$

$$
5 2 . 5 7 { \scriptstyle \pm 0 . 0 1 }
$$

$$
1 6 . 5 8 { \scriptstyle \pm 0 . 1 0 }
$$

$$
1 5 . 6 5 { \scriptstyle \pm 0 . 1 2 }
$$

$$
5 1 . 3 9 { \scriptstyle \pm 0 . 0 6 }
$$

$$
5 0 . 7 1 { \scriptstyle \pm 0 . 0 2 }
$$

$$
5 0 . 3 5 { \scriptstyle \pm 0 . 1 0 }
$$

$$
5 3 . 1 0 { \scriptstyle \pm 0 . 3 6 }
$$

$$
5 0 . 7 0 { \scriptstyle \pm 0 . 1 9 }
$$

$$
5 5 . 5 2 { \scriptstyle \pm 0 . 5 7 }
$$

$$
1 5 . 2 8 { \scriptstyle \pm 0 . 1 5 }
$$

$$
1 8 . 8 0 { \scriptstyle \pm 0 . 5 0 }
$$

$$
6 1 . 9 5 { \scriptstyle \pm 1 . 3 2 }
$$

$$
5 1 . 1 9 { \scriptstyle \pm 0 . 0 2 }
$$

$$
5 0 . 6 1 \pm 0 . 0 1
$$

$$
5 2 . 0 1 { \scriptstyle \pm 0 . 7 5 }
$$

$$
6 5 . 6 7 \pm 1 . 4 7
$$

$$
5 3 . 5 5 { \pm } 1 . 1 9
$$

$$
2 9 . 4 5 { \scriptstyle \pm 1 . 8 0 }
$$

$$
5 4 . 3 7 { \scriptstyle \pm 0 . 8 2 }
$$

$$
4 9 . 9 8 { \scriptstyle \pm 0 . 0 4 }
$$

$$
4 9 . 9 5 { \scriptstyle \pm 0 . 0 7 }
$$

$$
5 7 . 9 5 { \scriptstyle \pm 1 . 0 2 }
$$

$$
5 7 . 3 4 \pm 1 . 0 6
$$

$$
1 4 . 8 5 { \scriptstyle \pm 0 . 1 2 }
$$

$$
2 5 . 0 0 { \scriptstyle \pm 0 . 8 0 }
$$

$$
5 3 . 1 9 { \scriptstyle \pm 0 . 2 9 }
$$

$$
1 7 . 1 5 { \scriptstyle \pm 0 . 4 0 }
$$

$$
4 9 . 9 7 { \scriptstyle \pm 0 . 0 4 }
$$

$$
4 9 . 9 5 { \scriptstyle \pm 0 . 0 7 }
$$

$$
4 9 . 9 6 { \scriptstyle \pm 0 . 0 4 }
$$

$$
1 4 . 8 3 { \scriptstyle \pm 0 . 1 0 }
$$

$$
4 9 . 8 9 { \scriptstyle \pm 0 . 1 1 }
$$

$$
1 4 . 8 0 { \scriptstyle \pm 0 . 1 5 }
$$

Table 13: Temporal link prediction results on large-scale datasets under inductive settings. Results are averaged over three independent runs (in %).
<table><tr><td rowspan="2">Methods</td><td colspan="3">Beeradvocate</td><td colspan="3">Amazon-Kindle</td><td colspan="3">Ratebeer</td></tr><tr><td>AP</td><td>AUROC</td><td>MRR</td><td>AP</td><td>AUROC</td><td>MRR</td><td>AP</td><td>AUROC</td><td>MRR</td></tr><tr><td>JODIE</td><td>90.84±0.51</td><td>89.99±0.68</td><td> $6 8 . 6 6 { \scriptstyle \pm 0 . 9 6 }$ </td><td>93.78±0.38</td><td>93.01±0.37</td><td>78.89±0.98</td><td>90.38±0.70</td><td>89.81±0.77</td><td> $6 7 . 9 2 { \scriptstyle \pm 1 . 1 0 }$ </td></tr><tr><td>DyRep</td><td> $8 5 . 8 1 \pm 2 . 0 3$ </td><td>84.59±1.78</td><td>62.14±3.80</td><td>91.25±0.43</td><td> $9 0 . 1 7 { \scriptstyle \pm 0 . 4 6 }$ </td><td>73.40±0.85</td><td>83.59±0.11</td><td>81.82±0.08</td><td> $5 7 . 2 2 { \scriptstyle \pm 0 . 0 3 }$ </td></tr><tr><td>TGAT</td><td>59.67±3.80</td><td>62.48±2.60</td><td> $2 9 . 2 1 { \scriptstyle \pm 4 . 0 6 }$ </td><td>82.51±1.03</td><td> $8 3 . 9 7 { \scriptstyle \pm 0 . 5 8 }$ </td><td> $4 9 . 2 2 \pm 1 . 6 4$ </td><td>62.19±3.89</td><td> $6 4 . 9 2 { \scriptstyle \pm 2 . 3 7 }$ </td><td>31.86±3.16</td></tr><tr><td>TGN</td><td> $7 9 . 7 1 { \scriptstyle \pm 8 . 7 5 }$ </td><td>80.85±7.37</td><td> $5 2 . 8 3 { \scriptstyle \pm 1 3 . 1 1 }$ </td><td>94.14±0.51</td><td> $9 4 . 4 3 { \scriptstyle \pm 0 . 3 6 }$ </td><td>77.84±0.14</td><td>86.56±0.87</td><td>86.10±1.15</td><td> $6 1 . 4 7 { \scriptstyle \pm 1 . 8 9 }$ </td></tr><tr><td>CAWN</td><td>66.89±0.95</td><td>68.29±0.09</td><td> $3 6 . 6 1 \pm 0 . 2 8$ </td><td>80.89±0.11</td><td>81.90±0.18</td><td>46.82±0.33</td><td> $6 3 . 9 2 { \scriptstyle \pm 2 . 2 3 }$ </td><td>66.57±0.79</td><td> $3 3 . 9 0 { \scriptstyle \pm 1 . 2 3 }$ </td></tr><tr><td>TCL</td><td>54.64±0.16</td><td>59.35±0.57</td><td> $2 4 . 0 8 \pm 1 . 0 7$ </td><td>81.60±0.22</td><td> $8 2 . 5 3 { \scriptstyle \pm 0 . 1 3 }$ </td><td>47.51±0.67</td><td>61.99±4.64</td><td>63.42±3.14</td><td> $3 0 . 1 7 { \scriptstyle \pm 5 . 6 3 }$ </td></tr><tr><td>GraphMixer</td><td>87.00±0.19</td><td>85.52±0.26</td><td> $6 2 . 7 5 { \scriptstyle \pm 0 . 5 9 }$ </td><td>93.33±0.04</td><td> $9 2 . 7 1 { \scriptstyle \pm 0 . 0 3 }$ </td><td>74.54±0.11</td><td> $8 2 . 1 9 { \scriptstyle \pm 3 . 8 8 }$ </td><td>80.38±4.26</td><td>54.45±6.40</td></tr><tr><td>DyĠFormer</td><td>49.72±0.31</td><td>52.27±0.24</td><td>17.36±0.81</td><td>74.79±0.40</td><td>76.32±0.17</td><td>38.63±0.31</td><td>57.53±0.92</td><td> $5 9 . 2 9 { \scriptstyle \pm 0 . 8 0 }$ </td><td>26.90±1.26</td></tr><tr><td>FreeDyG</td><td>84.22±5.36</td><td>84.83±3.81</td><td> $6 0 . 8 5 { \scriptstyle \pm 5 . 8 5 }$ </td><td>87.64±1.74</td><td>86.77±1.48</td><td>64.46±4.09</td><td> $7 6 . 0 8 { \scriptstyle \pm 2 . 9 2 }$ </td><td>77.48±2.02</td><td> $4 8 . 5 9 { \scriptstyle \pm 3 . 3 6 }$ </td></tr><tr><td>LKD4DyTAG</td><td>76.66±0.82</td><td>75.61±0.82</td><td>43.50±0.81</td><td>80.48±0.07</td><td>81.69±0.12</td><td>46.02±0.17</td><td>71.56±0.16</td><td> $7 0 . 9 5 { \scriptstyle \pm 0 . 1 4 }$ </td><td> $3 9 . 3 2 { \scriptstyle \pm 0 . 2 2 }$ </td></tr><tr><td>CROSS</td><td>50.99±1.61</td><td> $5 3 . 3 4 \pm 1 . 4 2 $ </td><td> $1 9 . 1 3 { \scriptstyle \pm 2 . 4 8 }$ </td><td>87.47±0.22</td><td>88.73±0.21</td><td>57.42±0.36</td><td> $5 6 . 9 4 { \scriptstyle \pm 0 . 1 1 }$ </td><td> $5 9 . 0 6 { \scriptstyle \pm 0 . 0 6 }$ </td><td> $2 5 . 5 5 { \scriptstyle \pm 0 . 7 4 }$ </td></tr><tr><td>TGTalker</td><td>51.18±0.03</td><td>51.36±0.02</td><td>16.00±0.09</td><td>52.54±0.09</td><td>54.79±0.11</td><td>18.15±0.30</td><td>52.41±0.13</td><td> $5 4 . 0 9 { \scriptstyle \pm 3 . 9 2 }$ </td><td>17.80±2.50</td></tr><tr><td>DST-v1</td><td>51.34±0.04</td><td>51.47±0.05</td><td>16.15±0.11</td><td>52.66±0.10</td><td>54.85±0.09</td><td>18.25±0.38</td><td>52.96±0.94</td><td>55.60±1.22</td><td>18.65±1.05</td></tr><tr><td>DST-v2</td><td>51.21±0.01</td><td>51.53±0.01</td><td> $1 6 . 1 0 { \scriptstyle \pm 0 . 0 4 }$ </td><td>52.62±0.08</td><td>54.81±0.08</td><td>18.20±0.35</td><td> $5 1 . 7 2 { \scriptstyle \pm 1 . 0 8 }$ </td><td> $5 4 . 1 5 { \scriptstyle \pm 2 . 8 9 }$ </td><td> $1 7 . 1 5 { \pm } 1 . 8 5$ </td></tr><tr><td>Llaga-ND</td><td>49.97±0.03</td><td>49.94±0.06</td><td>14.80±0.15</td><td>68.62±3.37</td><td> $7 2 . 8 5 { \scriptstyle \pm 3 . 5 8 }$ </td><td> $3 6 . 4 0 { \scriptstyle \pm 4 . 1 0 }$ </td><td>50.00±0.02</td><td> $4 9 . 9 9 { \scriptstyle \pm 0 . 0 5 }$ </td><td> $1 4 . 9 0 { \scriptstyle \pm 0 . 0 5 }$ </td></tr><tr><td>Llaga-HO</td><td>50.00±0.05</td><td>50.00±0.10</td><td> $1 4 . 9 0 { \scriptstyle \pm 0 . 1 2 }$ </td><td>72.64±1.11</td><td>76.63±1.54</td><td>42.05±2.00</td><td>50.00±0.00</td><td>50.00±0.00</td><td>14.90±0.00</td></tr><tr><td>GraphGPT</td><td>56.80±6.80</td><td>58.35±8.35</td><td> $2 2 . 1 0 { \scriptstyle \pm 7 . 4 5 }$ </td><td>50.72±0.72</td><td> $5 1 . 1 7 { \scriptstyle \pm 1 . 1 7 }$ </td><td>15.70±1.00</td><td> $5 6 . 0 3 { \scriptstyle \pm 6 . 0 3 }$ </td><td>59.29±9.29</td><td>27.75±8.10</td></tr></table>

Table 14: Temporal link prediction results on small-scale datasets under transductive settings with historical negative sampling. Results are averaged over three independent runs (in %).
<table><tr><td rowspan="2">Methods</td><td colspan="3">FOOD</td><td colspan="3">IMDB</td><td colspan="3">Librarything</td></tr><tr><td> $\mathbf { A P }$ </td><td>AUROC</td><td>MRR</td><td>AP</td><td>AUROC</td><td>MRR</td><td>AP</td><td>AUROC</td><td>MRR</td></tr><tr><td>JODIE</td><td>55.00±0.11</td><td>55.13±0.30</td><td>20.64±0.17</td><td>66.09±8.32</td><td>72.60±7.93</td><td> $3 0 . 4 5 { \scriptstyle \pm 9 . 2 9 }$ </td><td>53.86±1.56</td><td> $5 3 . 4 3 { \pm } 1 . 5 8 $ </td><td> $1 9 . 9 7 { \scriptstyle \pm 1 . 1 0 }$ </td></tr><tr><td>DyRep</td><td> $5 5 . 1 3 { \scriptstyle \pm 0 . 4 9 }$ </td><td> $5 3 . 9 3 { \scriptstyle \pm 0 . 4 9 }$ </td><td> $2 1 . 1 0 { \scriptstyle \pm 0 . 4 0 }$ </td><td> $5 7 . 5 7 { \scriptstyle \pm 3 . 7 8 }$ </td><td> $5 7 . 8 3 { \scriptstyle \pm 4 . 8 9 }$ </td><td> $2 3 . 0 9 \pm 2 . 9 1$ </td><td> $5 6 . 6 2 { \scriptstyle \pm 1 . 1 3 }$ </td><td>56.83±0.87</td><td> $2 1 . 9 1 { \scriptstyle \pm 0 . 9 3 }$ </td></tr><tr><td>TGAT</td><td> $5 1 . 3 8 { \scriptstyle \pm 0 . 0 5 }$ </td><td>50.69±0.09</td><td> $1 7 . 9 2 \pm 0 . 0 4$ </td><td> $4 6 . 5 9 { \scriptstyle \pm 0 . 3 7 }$ </td><td> $4 5 . 3 6 { \pm } 1 . 1 3$ </td><td> $1 4 . 1 8 { \scriptstyle \pm 0 . 1 3 }$ </td><td> $4 5 . 5 1 { \scriptstyle \pm 0 . 0 6 }$ </td><td> $4 5 . 3 5 { \scriptstyle \pm 0 . 2 2 }$ </td><td> $1 3 . 1 1 { \scriptstyle \pm 0 . 1 6 }$ </td></tr><tr><td>TGN</td><td> $5 5 . 7 3 { \scriptstyle \pm 0 . 2 1 }$ </td><td> $5 7 . 2 6 { \scriptstyle \pm 0 . 2 3 }$ </td><td> $2 0 . 5 8 { \scriptstyle \pm 0 . 1 6 }$ </td><td> $4 0 . 4 0 { \scriptstyle \pm 0 . 7 1 }$ </td><td> $3 7 . 9 8 { \pm } 1 . 7 4 $ </td><td>8.39±0.32</td><td> $5 3 . 6 6 { \scriptstyle \pm 0 . 3 6 }$ </td><td> $5 4 . 4 4 { \scriptstyle \pm 0 . 1 7 }$ </td><td> $1 9 . 2 5 { \scriptstyle \pm 0 . 3 0 }$ </td></tr><tr><td>CAWN</td><td>52.26±0.01</td><td> $5 2 . 0 6 { \scriptstyle \pm 0 . 0 4 }$ </td><td> $1 8 . 5 0 { \scriptstyle \pm 0 . 0 5 }$ </td><td> $4 5 . 8 5 { \scriptstyle \pm 0 . 2 5 }$ </td><td> $4 3 . 6 1 \pm 0 . 4 7$ </td><td> $1 4 . 1 9 2 0 . 0 7$ </td><td> $4 8 . 0 6 { \scriptstyle \pm 0 . 1 6 }$ </td><td> $4 8 . 7 4 \pm 0 . 1 0$ </td><td> $1 4 . 9 8 { \scriptstyle \pm 0 . 1 0 }$ </td></tr><tr><td>TCL</td><td> $5 0 . 4 9 { \scriptstyle \pm 0 . 0 4 }$ </td><td> $4 9 . 8 3 { \scriptstyle \pm 0 . 1 2 }$ </td><td> $1 7 . 2 9 { \scriptstyle \pm 0 . 1 4 }$ </td><td> $4 6 . 0 7 \pm 1 . 0 6$ </td><td> $4 4 . 1 7 { \scriptstyle \pm 0 . 9 3 }$ </td><td> $1 4 . 3 2 { \scriptstyle \pm 0 . 8 6 }$ </td><td> $4 6 . 0 4 { \scriptstyle \pm 0 . 3 2 }$ </td><td> $4 6 . 5 3 { \scriptstyle \pm 0 . 5 4 }$ </td><td> $1 3 . 3 9 { \scriptstyle \pm 0 . 2 2 }$ </td></tr><tr><td>GraphMixer</td><td> $5 8 . 7 5 { \scriptstyle \pm 1 . 1 7 }$ </td><td> $5 7 . 5 6 { \pm } 1 . 9 0 $ </td><td> $2 3 . 7 9 2 0 . 7 6$ </td><td> $5 1 . 9 5 { \scriptstyle \pm 0 . 2 6 }$ </td><td> $5 7 . 2 0 { \scriptstyle \pm 0 . 7 7 }$ </td><td> $1 6 . 6 5 { \scriptstyle \pm 0 . 0 3 }$ </td><td> $5 1 . 0 1 { \scriptstyle \pm 0 . 2 8 }$ </td><td> $4 9 . 5 9 { \scriptstyle \pm 0 . 5 5 }$ </td><td> $1 7 . 9 0 { \scriptstyle \pm 0 . 4 1 }$ </td></tr><tr><td>DyĠFormer</td><td> $5 0 . 8 5 { \scriptstyle \pm 0 . 6 6 }$ </td><td> $4 9 . 7 9 2 1 . 1 6$ </td><td> $1 7 . 5 7 { \scriptstyle \pm 0 . 3 6 }$ </td><td> $4 6 . 7 5 { \scriptstyle \pm 0 . 2 8 }$ </td><td> $4 4 . 5 3 { \scriptstyle \pm 0 . 5 0 }$ </td><td> $1 4 . 8 0 { \scriptstyle \pm 0 . 2 5 }$ </td><td> $4 9 . 3 7 { \scriptstyle \pm 1 . 2 5 }$ </td><td> $5 0 . 2 0 { \scriptstyle \pm 0 . 5 2 }$ </td><td> $1 6 . 0 0 { \scriptstyle \pm 1 . 0 9 }$ </td></tr><tr><td>FreeDyG</td><td>57.66±0.31</td><td> $5 5 . 4 6 { \scriptstyle \pm 0 . 5 9 }$ </td><td> $2 3 . 3 0 { \scriptstyle \pm 0 . 2 2 }$ </td><td> $5 5 . 0 6 { \scriptstyle \pm 0 . 8 3 }$ </td><td> $5 9 . 5 1 { \pm } 1 . 4 8 $ </td><td> $1 9 . 6 7 { \scriptstyle \pm 0 . 3 0 }$ </td><td> $5 1 . 0 3 { \scriptstyle \pm 0 . 4 5 }$ </td><td> $4 9 . 2 3 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $1 8 . 0 6 { \scriptstyle \pm 0 . 3 8 }$ </td></tr><tr><td>LKD4DyTAG</td><td>52.30±0.00</td><td> $5 1 . 9 6 { \scriptstyle \pm 0 . 0 8 }$ </td><td> $1 8 . 6 2 { \scriptstyle \pm 0 . 0 2 }$ </td><td> $4 5 . 7 7 { \scriptstyle \pm 0 . 6 0 }$ </td><td>42.98±0.36</td><td> $1 4 . 1 2 { \scriptstyle \pm 0 . 4 6 }$ </td><td> $4 6 . 0 9 { \scriptstyle \pm 0 . 1 4 }$ </td><td>46.16±0.21</td><td> $1 3 . 7 1 { \scriptstyle \pm 0 . 1 2 }$ </td></tr><tr><td>CROSS</td><td>50.86±0.28</td><td> $4 9 . 7 5 { \scriptstyle \pm 0 . 1 1 }$ </td><td>17.66±0.21</td><td> $4 7 . 7 1 { \scriptstyle \pm 0 . 0 4 }$ </td><td>45.20±0.02</td><td> $1 5 . 6 3 { \scriptstyle \pm 0 . 0 7 }$ </td><td> $4 6 . 7 8 { \scriptstyle \pm 0 . 0 5 }$ </td><td> $4 7 . 4 5 { \scriptstyle \pm 0 . 0 8 }$ </td><td> $1 4 . 0 3 { \scriptstyle \pm 0 . 0 6 }$ </td></tr></table>

Table 15: Temporal link prediction results on small-scale datasets under inductive settings with historical negative sampling. Results are averaged over three independent runs (in %).
<table><tr><td rowspan="2">Methods</td><td colspan="3">FOOD</td><td colspan="3">IMDB</td><td colspan="3">Librarything</td></tr><tr><td>AP</td><td>AUROC</td><td>MRR</td><td>AP</td><td>AUROC</td><td>MRR</td><td> $\mathbf { A P }$ </td><td>AUROC</td><td>MRR</td></tr><tr><td>JODIE</td><td>50.08±0.39</td><td>46.65±0.10</td><td>17.65±0.47</td><td> $7 4 . 1 9 { \scriptstyle \pm 0 . 8 5 }$ </td><td> $7 1 . 0 7 { \scriptstyle \pm 1 . 8 3 }$ </td><td> $4 1 . 4 0 { \scriptstyle \pm 0 . 3 5 }$ </td><td>52.27±0.06</td><td> $4 7 . 6 0 { \scriptstyle \pm 0 . 6 9 }$ </td><td>19.80±0.09</td></tr><tr><td>DyRep</td><td>48.32±0.29</td><td>45.69±0.06</td><td>16.11±0.29</td><td> $6 9 . 7 3 { \scriptstyle \pm 0 . 5 2 }$ </td><td> $6 5 . 6 7 \pm 1 . 2 7$ </td><td> $3 6 . 5 7 { \scriptstyle \pm 0 . 3 1 }$ </td><td> $5 0 . 8 6 { \scriptstyle \pm 0 . 1 7 }$ </td><td> $4 6 . 7 0 { \scriptstyle \pm 0 . 7 4 }$ </td><td>18.40±0.04</td></tr><tr><td>TGAT</td><td> $5 1 . 0 1 { \scriptstyle \pm 0 . 2 5 }$ </td><td>50.25±0.03</td><td>17.66±0.30</td><td> $6 4 . 1 6 { \scriptstyle \pm 0 . 6 7 }$ </td><td> $6 0 . 8 5 { \scriptstyle \pm 0 . 7 1 }$ </td><td> $3 0 . 3 4 { \scriptstyle \pm 0 . 5 6 }$ </td><td> $4 7 . 2 0 { \scriptstyle \pm 0 . 2 2 }$ </td><td> $4 5 . 2 0 { \scriptstyle \pm 0 . 0 9 }$ </td><td> $1 5 . 0 5 { \scriptstyle \pm 0 . 1 5 }$ </td></tr><tr><td>TGN</td><td> $4 9 . 0 7 { \scriptstyle \pm 0 . 3 4 }$ </td><td>46.38±0.32</td><td>16.72±0.23</td><td> $6 9 . 8 5 { \scriptstyle \pm 2 . 8 8 }$ </td><td> $6 4 . 1 8 { \scriptstyle \pm 1 . 7 1 }$ </td><td> $3 7 . 9 7 { \scriptstyle \pm 4 . 3 8 }$ </td><td> $4 6 . 3 1 { \scriptstyle \pm 0 . 1 9 }$ </td><td> $4 3 . 1 6 { \scriptstyle \pm 0 . 3 4 }$ </td><td>14.64±0.16</td></tr><tr><td>CAWN</td><td> $5 4 . 9 7 { \scriptstyle \pm 0 . 2 2 }$ </td><td>53.61±0.08</td><td>20.31±0.22</td><td> $5 5 . 2 1 { \scriptstyle \pm 1 . 3 9 }$ </td><td> $5 4 . 8 8 { \scriptstyle \pm 1 . 4 4 }$ </td><td> $2 1 . 3 8 { \pm } 1 . 1 8$ </td><td> $5 0 . 8 9 { \scriptstyle \pm 1 . 1 3 }$ </td><td> $5 0 . 0 6 { \scriptstyle \pm 1 . 3 6 }$ </td><td>17.55±0.81</td></tr><tr><td>TCL</td><td> $5 3 . 1 0 { \scriptstyle \pm 0 . 1 7 }$ </td><td> $4 9 . 7 6 { \scriptstyle \pm 0 . 4 8 }$ </td><td> $1 9 . 3 1 { \scriptstyle \pm 0 . 3 0 }$ </td><td> $5 7 . 0 6 { \scriptstyle \pm 4 . 5 3 }$ </td><td> $5 4 . 5 6 { \scriptstyle \pm 2 . 2 3 }$ </td><td> $2 3 . 7 6 { \scriptstyle \pm 4 . 8 7 }$ </td><td> $4 7 . 3 5 { \scriptstyle \pm 0 . 4 7 }$ </td><td> $4 4 . 9 9 2 6 . 4 8 $ </td><td>15.14±0.41</td></tr><tr><td>GraphMixer</td><td> $5 6 . 4 6 { \scriptstyle \pm 1 . 1 9 }$ </td><td> $5 3 . 3 7 { \scriptstyle \pm 1 . 6 0 }$ </td><td> $2 2 . 8 0 { \scriptstyle \pm 0 . 9 4 }$ </td><td> $7 5 . 5 8 { \scriptstyle \pm 0 . 0 2 }$ </td><td> $7 1 . 5 6 { \scriptstyle \pm 0 . 1 0 }$ </td><td> $4 4 . 3 6 { \scriptstyle \pm 0 . 2 1 }$ </td><td> $6 1 . 4 6 { \scriptstyle \pm 2 . 9 2 }$ </td><td> $5 9 . 9 7 { \scriptstyle \pm 2 . 1 9 }$ </td><td> $2 6 . 7 0 { \scriptstyle \pm 2 . 8 4 }$ </td></tr><tr><td>DyĠFormer</td><td> $4 8 . 3 7 { \scriptstyle \pm 1 . 7 6 }$ </td><td> $4 6 . 3 3 { \scriptstyle \pm 3 . 8 8 }$ </td><td>15.92±0.80</td><td> $5 3 . 5 0 { \scriptstyle \pm 0 . 4 1 }$ </td><td> $5 3 . 7 5 { \scriptstyle \pm 0 . 0 4 }$ </td><td> $1 9 . 5 3 { \scriptstyle \pm 0 . 4 9 }$ </td><td> $5 3 . 5 7 { \scriptstyle \pm 0 . 3 2 }$ </td><td> $5 6 . 4 3 { \scriptstyle \pm 0 . 1 4 }$ </td><td> $1 8 . 5 1 { \scriptstyle \pm 0 . 2 9 }$ </td></tr><tr><td>FreeDyG</td><td>54.50±0.49</td><td> $5 0 . 5 4 { \scriptstyle \pm 1 . 0 0 }$ </td><td>21.36±0.24</td><td> $7 6 . 9 6 { \scriptstyle \pm 0 . 3 1 }$ </td><td> $7 3 . 3 5 { \scriptstyle \pm 0 . 4 5 }$ </td><td> $4 6 . 0 0 { \scriptstyle \pm 0 . 5 5 }$ </td><td> $5 8 . 1 2 { \scriptstyle \pm 1 . 2 2 }$ </td><td> $5 6 . 7 2 { \scriptstyle \pm 1 . 1 1 }$ </td><td> $2 3 . 6 2 { \scriptstyle \pm 1 . 1 3 }$ </td></tr><tr><td>LKD4DyTAG</td><td>50.34±0.08</td><td>49.50±0.13</td><td>17.12±0.01</td><td> $5 1 . 3 6 { \scriptstyle \pm 0 . 1 9 }$ </td><td> $5 1 . 1 5 { \scriptstyle \pm 0 . 0 1 }$ </td><td> $1 7 . 9 8 { \scriptstyle \pm 0 . 1 9 }$ </td><td> $4 8 . 1 7 { \scriptstyle \pm 0 . 0 4 }$ </td><td> $4 6 . 8 7 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $1 5 . 5 7 { \scriptstyle \pm 0 . 0 1 }$ </td></tr><tr><td>CROSS</td><td>49.78±0.42</td><td>48.83±0.34</td><td>16.78±0.33</td><td> $5 5 . 1 7 { \scriptstyle \pm 1 . 1 5 }$ </td><td> $5 6 . 3 8 { \scriptstyle \pm 0 . 6 7 }$ </td><td> $2 0 . 5 1 { \scriptstyle \pm 1 . 1 7 }$ </td><td> $5 0 . 1 5 { \scriptstyle \pm 0 . 2 4 }$ </td><td> $5 0 . 2 7 { \scriptstyle \pm 0 . 4 3 }$ </td><td> $1 6 . 6 1 \pm 0 . 1 4$ </td></tr></table>

Table 16: Temporal link prediction results on large-scale datasets under transductive settings with historical negative sampling. Results are averaged over three independent runs (in %).
<table><tr><td rowspan="2">Methods</td><td colspan="3">Beeradvocate</td><td colspan="3">Amazon-Kindle</td><td colspan="3">Ratebeer</td></tr><tr><td> $\mathbf { A P }$ </td><td>AUROC</td><td>MRR</td><td> $\mathbf { A P }$ </td><td>AUROC</td><td>MRR</td><td> $\mathbf { A P }$ </td><td>AUROC</td><td>MRR</td></tr><tr><td>JODIE</td><td>57.46±3.08</td><td>62.36±0.84</td><td> $2 1 . 1 2 { \scriptstyle \pm 3 . 2 8 }$ </td><td>68.84±1.66</td><td> $6 8 . 9 6 { \scriptstyle \pm 1 . 4 2 } $ </td><td> $3 2 . 4 5 { \scriptstyle \pm 1 . 8 0 }$ </td><td> $6 4 . 0 3 { \scriptstyle \pm 0 . 5 1 }$ </td><td> $6 7 . 1 5 { \scriptstyle \pm 0 . 2 4 }$ </td><td> $2 6 . 8 9 { \scriptstyle \pm 0 . 4 8 }$ </td></tr><tr><td>DyRep</td><td> $6 8 . 0 4 { \scriptstyle \pm 2 . 8 6 }$ </td><td> $7 1 . 0 1 { \scriptstyle \pm 2 . 6 7 }$ </td><td> $3 2 . 1 6 { \pm } 2 . 3 5$ </td><td> $6 5 . 2 6 { \scriptstyle \pm 0 . 9 4 }$ </td><td> $6 4 . 9 8 { \scriptstyle \pm 1 . 0 7 }$ </td><td> $2 9 . 2 3 { \scriptstyle \pm 0 . 8 1 }$ </td><td> $6 9 . 9 6 { \scriptstyle \pm 2 . 8 8 }$ </td><td>72.18±3.36</td><td> $3 3 . 9 5 { \scriptstyle \pm 2 . 3 9 }$ </td></tr><tr><td>TGAT</td><td> $5 2 . 5 5 { \scriptstyle \pm 0 . 1 3 }$ </td><td>52.82±0.14</td><td> $1 8 . 4 0 { \scriptstyle \pm 0 . 0 9 }$ </td><td> $5 3 . 7 7 { \scriptstyle \pm 0 . 5 2 }$ </td><td> $5 4 . 8 9 { \scriptstyle \pm 0 . 5 4 }$ </td><td> $1 9 . 2 4 { \scriptstyle \pm 0 . 3 7 }$ </td><td>50.26±0.08</td><td> $4 9 . 8 2 \pm 0 . 2 3 $ </td><td> $1 7 . 0 1 { \scriptstyle \pm 0 . 0 0 }$ </td></tr><tr><td>TGN</td><td> $5 3 . 5 4 { \scriptstyle \pm 2 . 0 3 }$ </td><td>57.49±0.89</td><td> $1 8 . 1 8 { \scriptstyle \pm 1 . 8 9 }$ </td><td> $4 4 . 7 9 { \scriptstyle \pm 0 . 5 9 }$ </td><td> $4 6 . 9 4 { \scriptstyle \pm 0 . 8 6 }$ </td><td> $1 1 . 1 3 { \scriptstyle \pm 0 . 6 4 }$ </td><td> $5 7 . 8 7 { \scriptstyle \pm 3 . 8 8 }$ </td><td> $6 2 . 7 8 { \scriptstyle \pm 2 . 8 2 }$ </td><td> $2 1 . 5 1 { \pm 3 . 4 9 }$ </td></tr><tr><td>CAWN</td><td> $5 2 . 3 4 { \scriptstyle \pm 0 . 0 5 }$ </td><td>53.17±0.01</td><td> $1 8 . 2 6 { \scriptstyle \pm 0 . 0 4 }$ </td><td> $5 4 . 3 1 { \scriptstyle \pm 0 . 9 0 }$ </td><td> $5 4 . 8 3 { \scriptstyle \pm 1 . 1 7 }$ </td><td> $1 9 . 7 8 { \scriptstyle \pm 0 . 6 3 }$ </td><td> $5 1 . 0 2 { \scriptstyle \pm 0 . 1 8 }$ </td><td> $5 1 . 8 5 { \scriptstyle \pm 0 . 1 0 }$ </td><td> $1 7 . 3 0 { \scriptstyle \pm 0 . 1 9 }$ </td></tr><tr><td>TCL</td><td> $5 2 . 6 8 { \scriptstyle \pm 0 . 1 7 }$ </td><td>52.85±0.18</td><td> $1 8 . 5 7 { \scriptstyle \pm 0 . 1 3 }$ </td><td> $5 5 . 0 2 { \scriptstyle \pm 0 . 8 1 }$ </td><td> $5 4 . 6 5 { \scriptstyle \pm 0 . 7 8 }$ </td><td> $2 0 . 4 9 { \scriptstyle \pm 0 . 6 1 }$ </td><td> $5 3 . 6 3 { \scriptstyle \pm 1 . 8 4 }$ </td><td> $5 2 . 8 1 \pm 1 . 2 5$ </td><td> $1 9 . 6 6 { \pm } 1 . 5 5$ </td></tr><tr><td>GraphMixer</td><td> $5 4 . 8 2 { \scriptstyle \pm 1 . 6 2 }$ </td><td> $5 7 . 0 2 { \scriptstyle \pm 1 . 2 2 }$ </td><td> $1 9 . 7 0 { \scriptstyle \pm 1 . 3 3 }$ </td><td> $6 4 . 6 2 \pm 0 . 0 3$ </td><td> $6 5 . 9 7 { \scriptstyle \pm 0 . 0 1 }$ </td><td> $2 8 . 1 6 { \scriptstyle \pm 0 . 0 6 }$ </td><td> $5 4 . 7 9 2 0 . 7 1$ </td><td> $5 5 . 9 7 { \scriptstyle \pm 1 . 0 2 }$ </td><td>19.96±0.48</td></tr><tr><td>DyĠFormer</td><td> $5 2 . 4 7 { \scriptstyle \pm 0 . 2 4 }$ </td><td> $5 2 . 5 8 { \scriptstyle \pm 0 . 1 4 }$ </td><td> $1 8 . 4 5 { \scriptstyle \pm 0 . 2 0 } $ </td><td> $5 3 . 7 5 { \scriptstyle \pm 0 . 6 4 }$ </td><td> $5 4 . 1 8 { \scriptstyle \pm 0 . 5 3 }$ </td><td> $1 9 . 2 9 { \scriptstyle \pm 0 . 5 0 }$ </td><td> $5 3 . 6 7 { \scriptstyle \pm 0 . 1 8 }$ </td><td> $5 3 . 5 2 { \pm } 0 . 3 3 $ </td><td> $1 9 . 4 3 { \scriptstyle \pm 0 . 1 4 }$ </td></tr><tr><td>FreeDyG</td><td> $5 0 . 5 6 { \scriptstyle \pm 1 . 7 1 }$ </td><td>54.11±0.91</td><td> $1 5 . 8 6 { \scriptstyle \pm 1 . 6 6 }$ </td><td> $6 5 . 1 6 { \scriptstyle \pm 0 . 0 9 }$ </td><td> $6 5 . 9 8 { \scriptstyle \pm 0 . 2 9 }$ </td><td> $2 8 . 7 9 { \scriptstyle \pm 0 . 1 8 }$ </td><td> $5 4 . 3 5 { \scriptstyle \pm 0 . 2 9 }$ </td><td> $5 7 . 1 1 { \scriptstyle \pm 0 . 0 4 }$ </td><td> $1 9 . 3 3 { \scriptstyle \pm 0 . 2 9 }$ </td></tr><tr><td>LKD4DyTAG</td><td> $5 1 . 9 7 { \scriptstyle \pm 0 . 1 0 }$ </td><td> $5 2 . 3 5 { \scriptstyle \pm 0 . 1 3 }$ </td><td> $1 8 . 0 1 { \scriptstyle \pm 0 . 0 5 }$ </td><td> $5 4 . 9 5 { \scriptstyle \pm 0 . 5 0 }$ </td><td> $5 5 . 1 2 { \scriptstyle \pm 0 . 6 3 }$ </td><td> $2 0 . 2 9 { \scriptstyle \pm 0 . 3 8 }$ </td><td> $4 9 . 8 5 { \scriptstyle \pm 0 . 1 6 }$ </td><td> $4 9 . 3 0 { \scriptstyle \pm 0 . 2 8 }$ </td><td>16.74±0.11</td></tr><tr><td>CROSS</td><td> $5 2 . 1 4 { \scriptstyle \pm 0 . 3 4 }$ </td><td> $5 2 . 1 6 { \scriptstyle \pm 0 . 3 0 }$ </td><td> $1 8 . 2 3 { \scriptstyle \pm 0 . 2 1 }$ </td><td> $5 6 . 2 2 { \scriptstyle \pm 0 . 0 1 }$ </td><td> $5 6 . 5 3 { \scriptstyle \pm 0 . 1 7 }$ </td><td>21.18±0.01</td><td> $5 4 . 3 6 { \pm } 1 . 1 9$ </td><td> $5 4 . 1 1 { \scriptstyle \pm 0 . 8 9 }$ </td><td> $2 0 . 0 3 { \scriptstyle \pm 1 . 0 5 }$ </td></tr></table>

Table 17: Temporal link prediction results on large-scale datasets under inductive settings with historical negative sampling. Results are averaged over three independent runs (in %).
<table><tr><td rowspan="2">Methods</td><td colspan="3">Beeradvocate</td><td colspan="3">Amazon-Kindle</td><td colspan="3">Ratebeer</td></tr><tr><td>AP</td><td>AUROC</td><td>MRR</td><td>AP</td><td>AUROC</td><td>MRR</td><td>AP</td><td>AUROC</td><td>MRR</td></tr><tr><td>JODIE</td><td> $5 3 . 4 2 { \scriptstyle \pm 0 . 2 5 }$ </td><td> $5 3 . 0 5 { \scriptstyle \pm 0 . 4 1 }$ </td><td> $1 9 . 5 1 { \scriptstyle \pm 0 . 1 4 }$ </td><td> $7 8 . 7 5 { \scriptstyle \pm 0 . 3 9 }$ </td><td> $7 5 . 3 8 { \scriptstyle \pm 0 . 0 8 }$ </td><td> $4 7 . 8 3 { \scriptstyle \pm 0 . 9 5 }$ </td><td> $5 3 . 1 4 { \scriptstyle \pm 0 . 3 0 }$ </td><td> $5 2 . 6 9 { \scriptstyle \pm 0 . 5 3 }$ </td><td> $1 9 . 2 8 { \scriptstyle \pm 0 . 4 4 }$ </td></tr><tr><td>DyRep</td><td> $5 0 . 8 2 { \scriptstyle \pm 0 . 8 6 }$ </td><td>49.19±0.43</td><td>17.78±0.78</td><td> $7 3 . 5 3 { \pm } 1 . 4 2$ </td><td> $6 8 . 8 6 { \scriptstyle \pm 1 . 2 0 }$ </td><td> $4 1 . 2 0 { \scriptstyle \pm 2 . 0 8 }$ </td><td> $5 4 . 1 5 { \scriptstyle \pm 0 . 0 3 }$ </td><td> $5 3 . 3 2 { \scriptstyle \pm 0 . 8 8 }$ </td><td> $2 0 . 0 3 { \scriptstyle \pm 0 . 2 0 }$ </td></tr><tr><td>TGAT</td><td> $5 4 . 0 6 { \scriptstyle \pm 0 . 4 7 }$ </td><td>52.84±0.14</td><td>19.76±0.37</td><td> $5 6 . 5 9 { \scriptstyle \pm 0 . 0 9 }$ </td><td> $5 6 . 1 5 { \scriptstyle \pm 0 . 1 5 }$ </td><td>21.88±0.05</td><td> $5 2 . 9 7 { \scriptstyle \pm 0 . 7 9 }$ </td><td> $5 2 . 0 2 { \scriptstyle \pm 0 . 2 5 }$ </td><td>19.04±0.68</td></tr><tr><td>TGN</td><td> $4 9 . 3 9 { \scriptstyle \pm 0 . 3 1 }$ </td><td>48.17±0.39</td><td>16.48±0.23</td><td> $6 3 . 5 2 { \scriptstyle \pm 0 . 4 4 }$ </td><td>58.78±0.36</td><td> $2 9 . 3 3 { \scriptstyle \pm 0 . 7 6 }$ </td><td> $5 0 . 7 0 { \scriptstyle \pm 0 . 9 0 }$ </td><td> $4 8 . 6 6 \pm 0 . 4 1$ </td><td>17.66±0.79</td></tr><tr><td>CAWN</td><td>52.81±0.17</td><td>53.49±0.22</td><td>18.60±0.12</td><td>56.49±0.47</td><td>55.43±0.34</td><td>21.89±0.41</td><td> $5 4 . 8 9 { \scriptstyle \pm 0 . 1 6 }$ </td><td> $5 6 . 3 4 { \scriptstyle \pm 0 . 1 4 }$ </td><td>19.94±0.12</td></tr><tr><td>TCL</td><td>54.23±0.33</td><td>52.83±0.14</td><td>20.06±0.18</td><td>55.04±0.35</td><td>54.30±0.27</td><td>20.66±0.28</td><td>56.75±0.76</td><td> $5 5 . 3 7 { \scriptstyle \pm 0 . 6 9 }$ </td><td>22.38±0.55</td></tr><tr><td>GraphMixer</td><td>55.63±1.05</td><td>56.63±0.78</td><td>20.74±0.88</td><td>78.70±0.73</td><td>76.63±0.39</td><td>46.89±1.27</td><td>58.13±0.03</td><td>59.04±0.21</td><td>22.71±0.12</td></tr><tr><td>DyĠFormer</td><td>55.95±0.21</td><td>54.07±0.19</td><td>21.27±0.20</td><td>56.36±0.52</td><td>55.74±0.50</td><td>21.55±0.37</td><td>57.23±0.11</td><td>57.84±0.20</td><td>21.90±0.20</td></tr><tr><td>FreeDyG</td><td>54.44±0.68</td><td>56.03±0.00</td><td>19.73±0.62</td><td>81.28±0.69</td><td> $7 8 . 7 3 { \scriptstyle \pm 0 . 5 1 }$ </td><td>51.06±1.32</td><td>60.23±0.56</td><td>62.79±0.98</td><td>23.98±0.34</td></tr><tr><td>LKD4DyTAG</td><td>51.19±0.05</td><td>50.61±0.13</td><td>17.70±0.01</td><td>56.21±0.10</td><td> $5 5 . 1 5 { \scriptstyle \pm 0 . 0 7 }$ </td><td>21.63±0.09</td><td>51.12±0.25</td><td>50.64±0.35</td><td>17.64±0.17</td></tr><tr><td>CROSS</td><td> $5 4 . 5 8 { \scriptstyle \pm 1 . 1 3 }$ </td><td>52.96±0.70</td><td>20.32±0.81</td><td>59.80±0.01</td><td> $5 9 . 5 2 { \scriptstyle \pm 0 . 1 6 }$ </td><td>24.25±0.07</td><td>58.00±1.19</td><td>58.44±0.76</td><td> $2 2 . 6 7 { \scriptstyle \pm 1 . 1 4 }$ </td></tr></table>

training, whereas the inductive setting requires predicting future links for previously unseen nodes.   
The detailed inductive results for temporal link prediction task are presented in Tables 12 and 13.

Table 18: Temporal link prediction results on small-scale datasets under transductive settings with inductive negative sampling. Results are averaged over three independent runs (in %).
<table><tr><td rowspan="2">Methods</td><td colspan="3">FOOD</td><td colspan="3">IMDB</td><td colspan="3">Librarything</td></tr><tr><td>AP</td><td>AUROC</td><td>MRR</td><td>AP</td><td>AUROC</td><td>MRR</td><td>AP</td><td>AUROC</td><td>MRR</td></tr><tr><td>JODIE</td><td>52.27±1.48</td><td>51.54±0.87</td><td>19.00±1.18</td><td>54.04±9.34</td><td>57.97±13.71</td><td>19.38±7.05</td><td>57.33±1.47</td><td>57.08±1.90</td><td>23.53±1.14</td></tr><tr><td>DyRep</td><td>55.10±0.51</td><td>52.88±0.78</td><td>21.37±0.36</td><td>57.29±3.41</td><td> $5 0 . 8 7 { \scriptstyle \pm 3 . 4 4 }$ </td><td> $2 6 . 3 3 { \scriptstyle \pm 3 . 6 4 }$ </td><td>63.21±1.50</td><td>63.42±1.17</td><td>28.00±1.70</td></tr><tr><td>TGAT</td><td>51.24±0.20</td><td>50.62±0.21</td><td>17.71±0.21</td><td> $4 4 . 9 5 { \scriptstyle \pm 0 . 3 8 }$ </td><td> $4 3 . 8 3 \pm 1 . 0 1$ </td><td> $1 3 . 2 7 { \scriptstyle \pm 0 . 1 2 }$ </td><td>43.18±0.24</td><td>41.18±0.70</td><td>12.04±0.06</td></tr><tr><td>TGN</td><td>55.34±0.24</td><td>56.03±0.49</td><td>20.67±0.10</td><td>35.32±0.12</td><td> $2 0 . 4 5 { \scriptstyle \pm 0 . 4 1 }$ </td><td> $6 . 4 4 \pm 0 . 1 4$ </td><td>59.17±0.46</td><td>60.54±0.29</td><td>23.69±0.45</td></tr><tr><td>CAWN</td><td>53.06±0.07</td><td>53.83±0.10</td><td>18.77±0.04</td><td>46.32±0.41</td><td> $4 4 . 0 2 { \scriptstyle \pm 0 . 3 8 }$ </td><td> $1 4 . 6 2 \pm 0 . 2 9$ </td><td>47.85±0.06</td><td>48.47±0.06</td><td>14.92±0.11</td></tr><tr><td>TCL</td><td>50.53±0.01</td><td>50.22±0.21</td><td>17.13±0.12</td><td>46.19±1.44</td><td> $4 4 . 5 6 { \scriptstyle \pm 1 . 4 9 }$ </td><td> $1 4 . 5 5 { \scriptstyle \pm 1 . 2 4 }$ </td><td>43.74±0.11</td><td>42.97±0.46</td><td>11.76±0.09</td></tr><tr><td>GraphMixer</td><td>53.81±0.68</td><td>50.34±1.35</td><td>20.70±0.30</td><td>42.63±0.27</td><td> $4 0 . 6 0 { \scriptstyle \pm 0 . 8 5 }$ </td><td>11.52±0.05</td><td>45.64±0.55</td><td>41.81±0.15</td><td>14.65±0.46</td></tr><tr><td>DyGFormer</td><td>49.32±1.83</td><td>47.18±3.92</td><td>16.67±0.79</td><td> $4 6 . 9 7 { \scriptstyle \pm 0 . 1 7 }$ </td><td> $4 4 . 9 8 { \scriptstyle \pm 0 . 2 5 }$ </td><td>15.29±0.14</td><td> $4 9 . 4 6 { \scriptstyle \pm 0 . 0 1 }$ </td><td>51.30±1.58</td><td>15.74±0.38</td></tr><tr><td>FreeDyG</td><td>53.45±0.38</td><td>49.25±0.98</td><td>20.80±0.18</td><td>44.83±0.08</td><td> $4 3 . 8 9 { \scriptstyle \pm 0 . 1 6 }$ </td><td>13.00±0.01</td><td>45.92±0.42</td><td>42.00±0.04</td><td>15.11±0.40</td></tr><tr><td>LKD4DyTAG</td><td>52.74±0.10</td><td>52.29±0.13</td><td>18.98±0.09</td><td> $4 6 . 3 3 { \scriptstyle \pm 0 . 8 9 }$ </td><td> $4 3 . 5 7 { \scriptstyle \pm 0 . 7 6 }$ </td><td>14.75±0.78</td><td>44.29±0.06</td><td>43.15±0.08</td><td>12.64±0.02</td></tr><tr><td>CROSS</td><td>50.92±0.10</td><td> $5 0 . 0 7 { \scriptstyle \pm 0 . 0 3 }$ </td><td>17.54±0.10</td><td> $4 8 . 9 3 { \scriptstyle \pm 0 . 0 8 }$ </td><td> $4 7 . 1 9 2 0 . 0 7$ </td><td>16.50±0.05</td><td> $4 6 . 5 4 { \scriptstyle \pm 0 . 0 4 }$ </td><td>47.15±0.11</td><td>13.82±0.07</td></tr></table>

Table 19: Temporal link prediction results on small-scale datasets under inductive settings with inductive negative sampling. Results are averaged over three independent runs (in %).
<table><tr><td rowspan="2">Methods</td><td colspan="3">FOOD</td><td colspan="3">IMDB</td><td colspan="3">Librarything</td></tr><tr><td>AP</td><td>AUROC</td><td>MRR</td><td>AP</td><td>AUROC</td><td>MRR</td><td>AP</td><td>AUROC</td><td>MRR</td></tr><tr><td>JODIE</td><td>50.08±0.39</td><td>46.65±0.10</td><td>17.65±0.47</td><td>74.19±0.85</td><td> $7 1 . 0 7 { \scriptstyle \pm 1 . 8 3 }$ </td><td>41.40±0.35</td><td> $5 2 . 2 7 { \scriptstyle \pm 0 . 0 6 }$ </td><td>47.60±0.69</td><td>19.80±0.09</td></tr><tr><td>DyRep</td><td>48.32±0.29</td><td>45.69±0.06</td><td>16.11±0.29</td><td>69.73±0.52</td><td>65.67±1.27</td><td> $3 6 . 5 7 { \scriptstyle \pm 0 . 3 1 }$ </td><td>50.86±0.17</td><td>46.70±0.74</td><td>18.40±0.04</td></tr><tr><td>TGAT</td><td>51.01±0.25</td><td>50.25±0.03</td><td>17.66±0.30</td><td>64.16±0.67</td><td>60.85±0.71</td><td>30.34±0.56</td><td>47.20±0.22</td><td>45.20±0.09</td><td>15.05±0.15</td></tr><tr><td>TGN</td><td>49.07±0.34</td><td>46.38±0.32</td><td>16.72±0.23</td><td>69.85±2.88</td><td>64.18±1.71</td><td> $3 7 . 9 7 { \scriptstyle \pm 4 . 3 8 }$ </td><td> $4 6 . 3 1 { \scriptstyle \pm 0 . 1 9 }$ </td><td>43.16±0.34</td><td>14.64±0.16</td></tr><tr><td>CAWN</td><td>54.97±0.22</td><td>53.61±0.08</td><td>20.31±0.22</td><td>55.21±1.39</td><td>54.88±1.44</td><td> $2 1 . 3 8 { \pm } 1 . 1 8$ </td><td>50.89±1.13</td><td>50.06±1.36</td><td>17.55±0.81</td></tr><tr><td>TCL</td><td>53.10±0.17</td><td>49.76±0.48</td><td>19.31±0.30</td><td>57.06±4.53</td><td>54.56±2.23</td><td> $2 3 . 7 6 { \scriptstyle \pm 4 . 8 7 }$ </td><td>47.35±0.47</td><td>44.99±0.48</td><td>15.14±0.41</td></tr><tr><td>GraphMixer</td><td>56.46±1.19</td><td>53.37±1.60</td><td>22.80±0.94</td><td>75.58±0.02</td><td>71.56±0.10</td><td>44.36±0.21</td><td>61.46±2.92</td><td>59.97±2.19</td><td>26.70±2.84</td></tr><tr><td>DyĠFormer</td><td>48.37±1.76</td><td>46.33±3.88</td><td>15.92±0.80</td><td>53.50±0.41</td><td>53.75±0.04</td><td>19.53±0.49</td><td>53.57±0.32</td><td>56.43±0.14</td><td>18.51±0.29</td></tr><tr><td>FreeDyG</td><td>54.50±0.49</td><td>50.54±1.00</td><td>21.36±0.24</td><td>76.96±0.31</td><td>73.35±0.45</td><td>46.00±0.55</td><td> $5 8 . 1 2 { \scriptstyle \pm 1 . 2 2 }$ </td><td> $5 6 . 7 2 { \scriptstyle \pm 1 . 1 1 }$ </td><td>23.62±1.13</td></tr><tr><td>LKD4DyTAG</td><td>50.34±0.08</td><td>49.50±0.13</td><td>17.12±0.01</td><td>51.36±0.19</td><td>51.15±0.01</td><td> $1 7 . 9 8 { \scriptstyle \pm 0 . 1 9 }$ </td><td> $4 8 . 1 7 { \scriptstyle \pm 0 . 0 4 }$ </td><td>46.87±0.00</td><td>15.57±0.01</td></tr><tr><td>CROSS</td><td> $4 9 . 7 8 { \scriptstyle \pm 0 . 4 2 }$ </td><td> $4 8 . 8 3 { \scriptstyle \pm 0 . 3 4 }$ </td><td> $1 6 . 7 8 { \scriptstyle \pm 0 . 3 3 }$ </td><td> $5 5 . 1 7 { \scriptstyle \pm 1 . 1 5 }$ </td><td> $5 6 . 3 8 { \scriptstyle \pm 0 . 6 7 }$ </td><td> $2 0 . 5 1 { \scriptstyle \pm 1 . 1 7 }$ </td><td> $5 0 . 1 5 { \scriptstyle \pm 0 . 2 4 }$ </td><td> $5 0 . 2 7 { \scriptstyle \pm 0 . 4 3 }$ </td><td> $1 6 . 6 1 { \scriptstyle \pm 0 . 1 4 }$ </td></tr></table>

Additional Negative Sampling Results. Following DyGLib [39], we additionally evaluate all methods on the TLP task under both historical and inductive negative sampling strategies. The results are reported in Tables 14 to 21.

Text Ablation Study Results. Figure 7 presents detailed performance comparisons on both TLP and TNC before and after removing textual attributes. The results show that textual information substantially affects TGNN performance on TLP but has relatively limited impact on TNC, whereas for LLM-Predictors the opposite trend holds: text contributes significantly to TNC while providing smaller gains for TLP.

Comparison of LP and E2E for TGNNs on TNC. As shown in Table 22, end-to-end fine-tuning does not yield significant improvements over linear probing on the TNC task for TGNNs. This suggests that gradients from semantic supervision are insufficient to overcome the strong structural bias learned during TLP pretraining. Consequently, the representations remain anchored in topological dynamics and fail to adapt effectively to semantic objectives.

![](images/635c183518ddf79711a3d45fba90b461624ada068425bef4783fb0350fcdbeb1.jpg)  
(a)

![](images/bdf6cf7f377b1a88a14111b3498b4c93957e50f3ea94c35c8276d74fea89e00c.jpg)  
(b)  
Figure 7: Effect of Text Attributes on TLP. Impact of textual attributes on model performance across different methods for the TLP task.

Scalability Results. Figure 8 reports scalability evaluations of all methods. Specifically, on the largest dataset, Amazon-Kindle, we subsample 1M, 2M, 3M, 4M, and 5M edges and measure the runtime of each method. Results show that all methods exhibit approximately linear growth in computation time as data size increases, demonstrating good scalability. LLM-Predictor methods incur higher time cost due to LLM inference, but still maintain favorable scaling behavior.

Table 20: Temporal link prediction results on large-scale datasets under transductive settings with inductive negative sampling. Results are averaged over three independent runs (in %).
<table><tr><td rowspan="2">Methods</td><td colspan="3">Beeradvocate</td><td colspan="3">Amazon-Kindle</td><td colspan="3">Ratebeer</td></tr><tr><td>AP</td><td>AUROC</td><td>MRR</td><td>AP</td><td>AUROC</td><td>MRR</td><td>AP</td><td>AUROC</td><td>MRR</td></tr><tr><td>JODIE</td><td> $5 7 . 7 5 { \scriptstyle \pm 2 . 5 9 }$ </td><td> $6 3 . 1 2 { \scriptstyle \pm 0 . 1 8 }$ </td><td> $2 3 . 1 4 \pm 3 . 1 3$ </td><td> $5 8 . 6 9 { \scriptstyle \pm 2 . 3 9 }$ </td><td> $5 5 . 7 9 { \scriptstyle \pm 2 . 6 6 }$ </td><td> $2 5 . 1 0 { \scriptstyle \pm 2 . 0 6 }$ </td><td> $6 6 . 2 8 { \scriptstyle \pm 0 . 2 8 }$ </td><td> $7 0 . 4 3 { \scriptstyle \pm 0 . 2 0 }$ </td><td> $3 0 . 5 9 { \scriptstyle \pm 0 . 4 4 }$ </td></tr><tr><td>DyRep</td><td> $7 1 . 7 3 { \scriptstyle \pm 2 . 5 6 }$ </td><td> $7 4 . 0 4 \pm 2 . 5 3$ </td><td> $3 8 . 1 0 { \scriptstyle \pm 2 . 5 0 }$ </td><td> $6 2 . 6 5 { \scriptstyle \pm 1 . 4 6 }$ </td><td> $6 1 . 2 7 { \scriptstyle \pm 1 . 9 6 }$ </td><td> $2 7 . 5 8 { \pm } 1 . 0 8 $ </td><td> $7 2 . 8 6 { \pm } 1 . 7 8 $ </td><td> $7 5 . 2 0 { \scriptstyle \pm 2 . 5 0 }$ </td><td> $3 8 . 0 7 { \scriptstyle \pm 1 . 3 3 }$ </td></tr><tr><td>TGAT</td><td> $5 0 . 5 7 { \scriptstyle \pm 0 . 2 5 }$ </td><td> $5 0 . 7 3 { \scriptstyle \pm 0 . 1 6 }$ </td><td> $1 7 . 1 7 { \scriptstyle \pm 0 . 2 8 }$ </td><td> $4 8 . 2 1 \pm 0 . 7 8$ </td><td> $4 7 . 4 3 \pm 1 . 0 1$ </td><td> $1 6 . 1 5 { \scriptstyle \pm 0 . 6 6 }$ </td><td> $4 9 . 8 0 { \scriptstyle \pm 0 . 2 2 }$ </td><td> $4 9 . 2 8 { \scriptstyle \pm 0 . 2 5 }$ </td><td> $1 6 . 9 0 { \scriptstyle \pm 0 . 2 3 }$ </td></tr><tr><td>TGN</td><td> $5 4 . 8 8 { \scriptstyle \pm 0 . 2 9 }$ </td><td> $6 0 . 3 1 { \pm } 1 . 1 5$ </td><td> $2 0 . 3 2 { \scriptstyle \pm 0 . 7 2 }$ </td><td> $3 5 . 9 9 2 0 . 6 5 $ </td><td> $2 3 . 4 6 { \scriptstyle \pm 2 . 4 7 }$ </td><td> $7 . 8 3 { \scriptstyle \pm 0 . 5 2 }$ </td><td> $6 4 . 4 4 { \scriptstyle \pm 3 . 7 5 }$ </td><td> $7 1 . 0 5 { \scriptstyle \pm 2 . 6 2 }$ </td><td> $2 7 . 9 3 { \scriptstyle \pm 3 . 6 9 }$ </td></tr><tr><td>CAWN</td><td> $5 1 . 4 3 { \scriptstyle \pm 0 . 0 6 }$ </td><td> $5 2 . 8 2 { \scriptstyle \pm 0 . 1 0 }$ </td><td> $1 7 . 7 7 { \scriptstyle \pm 0 . 0 4 }$ </td><td> $4 9 . 9 9 { \scriptstyle \pm 1 . 1 9 }$ </td><td> $4 9 . 2 0 { \scriptstyle \pm 1 . 8 2 }$ </td><td> $1 7 . 4 8 { \scriptstyle \pm 0 . 7 9 }$ </td><td> $5 2 . 5 4 { \scriptstyle \pm 0 . 1 2 }$ </td><td> $5 5 . 0 3 { \scriptstyle \pm 0 . 1 3 }$ </td><td> $1 8 . 2 3 { \scriptstyle \pm 0 . 1 8 }$ </td></tr><tr><td>TCL</td><td> $5 0 . 2 3 { \scriptstyle \pm 0 . 3 9 }$ </td><td>50.38±0.23</td><td> $1 6 . 7 7 { \scriptstyle \pm 0 . 3 3 }$ </td><td> $4 9 . 5 6 { \scriptstyle \pm 0 . 5 4 }$ </td><td> $4 7 . 9 2 { \scriptstyle \pm 0 . 8 1 }$ </td><td> $1 7 . 4 4 \pm 0 . 4 0$ </td><td> $5 0 . 9 3 { \scriptstyle \pm 0 . 8 5 }$ </td><td> $5 0 . 3 6 { \scriptstyle \pm 0 . 6 3 }$ </td><td> $1 7 . 6 5 { \scriptstyle \pm 0 . 8 9 }$ </td></tr><tr><td>GraphMixer</td><td> $5 1 . 7 1 { \scriptstyle \pm 0 . 7 5 }$ </td><td> $5 2 . 2 9 { \scriptstyle \pm 0 . 7 3 }$ </td><td> $1 9 . 5 0 { \scriptstyle \pm 0 . 5 8 }$ </td><td> $5 0 . 5 1 { \scriptstyle \pm 0 . 1 3 }$ </td><td> $4 9 . 4 1 { \scriptstyle \pm 0 . 0 7 }$ </td><td> $1 8 . 4 5 { \scriptstyle \pm 0 . 1 4 }$ </td><td> $5 2 . 5 8 { \scriptstyle \pm 0 . 2 0 }$ </td><td> $5 2 . 4 2 { \scriptstyle \pm 0 . 4 4 }$ </td><td>19.84±0.39</td></tr><tr><td>DyĠFormer</td><td>50.30±0.33</td><td>50.62±0.15</td><td> $1 6 . 8 0 { \scriptstyle \pm 0 . 2 7 }$ </td><td>50.89±0.94</td><td> $5 0 . 7 9 { \scriptstyle \pm 0 . 9 4 }$ </td><td> $1 7 . 5 7 { \scriptstyle \pm 0 . 6 4 }$ </td><td> $5 2 . 4 4 { \scriptstyle \pm 0 . 2 0 }$ </td><td> $5 3 . 8 1 { \scriptstyle \pm 0 . 3 4 }$ </td><td>18.13±0.13</td></tr><tr><td>FreeDyG</td><td>51.41±0.41</td><td>52.26±0.24</td><td> $1 9 . 0 3 { \scriptstyle \pm 0 . 6 1 }$ </td><td>51.03±0.02</td><td> $4 9 . 4 2 \pm 0 . 3 8$ </td><td> $1 8 . 1 9 { \scriptstyle \pm 0 . 0 3 }$ </td><td> $5 4 . 3 6 { \scriptstyle \pm 1 . 0 0 }$ </td><td> $5 6 . 5 3 { \scriptstyle \pm 1 . 6 6 }$ </td><td>20.11±0.69</td></tr><tr><td>LKD4DyTAG</td><td>50.53±0.07</td><td>50.64±0.09</td><td> $1 7 . 9 3 { \scriptstyle \pm 0 . 1 2 }$ </td><td> $5 1 . 8 8 { \scriptstyle \pm 0 . 7 1 }$ </td><td> $5 0 . 7 7 { \scriptstyle \pm 1 . 0 1 }$ </td><td> $1 9 . 0 3 { \scriptstyle \pm 0 . 4 8 }$ </td><td> $4 9 . 6 2 \pm 0 . 0 8$ </td><td> $4 9 . 1 1 { \scriptstyle \pm 0 . 1 9 }$ </td><td> $1 6 . 9 2 { \scriptstyle \pm 0 . 0 5 }$ </td></tr><tr><td>CROSS</td><td>50.13±0.39</td><td> $5 0 . 0 2 { \scriptstyle \pm 0 . 5 7 }$ </td><td>16.81±0.20</td><td>55.64±0.05</td><td> $5 6 . 7 0 { \scriptstyle \pm 0 . 4 7 }$ </td><td> $2 1 . 6 4 { \scriptstyle \pm 0 . 1 0 }$ </td><td> $5 3 . 3 1 { \scriptstyle \pm 0 . 7 3 }$ </td><td> $5 4 . 9 0 { \scriptstyle \pm 0 . 6 6 }$ </td><td> $1 8 . 8 3 { \scriptstyle \pm 0 . 6 4 }$ </td></tr></table>

Table 21: Temporal link prediction results on large-scale datasets under inductive settings with inductive negative sampling. Results are averaged over three independent runs (in %).
<table><tr><td rowspan="2">Methods</td><td colspan="3">Beeradvocate</td><td colspan="3">Amazon-Kindle</td><td colspan="3">Ratebeer</td></tr><tr><td>AP</td><td>AUROC</td><td>MRR</td><td>AP</td><td>AUROC</td><td>MRR</td><td>AP</td><td>AUROC</td><td>MRR</td></tr><tr><td>JODIE</td><td> $5 3 . 4 2 { \scriptstyle \pm 0 . 2 5 }$ </td><td> $5 3 . 0 5 { \scriptstyle \pm 0 . 4 1 }$ </td><td> $1 9 . 5 1 { \scriptstyle \pm 0 . 1 4 }$ </td><td> $7 8 . 7 5 { \scriptstyle \pm 0 . 3 9 }$ </td><td> $7 5 . 3 8 { \scriptstyle \pm 0 . 0 8 }$ </td><td> $4 7 . 8 3 { \scriptstyle \pm 0 . 9 5 }$ </td><td> $5 3 . 1 4 { \scriptstyle \pm 0 . 3 0 }$ </td><td> $5 2 . 6 9 { \scriptstyle \pm 0 . 5 3 }$ </td><td> $1 9 . 2 8 { \scriptstyle \pm 0 . 4 4 }$ </td></tr><tr><td>DyRep</td><td> $5 0 . 8 2 { \scriptstyle \pm 0 . 8 6 }$ </td><td> $4 9 . 1 9 2 0 . 4 3$ </td><td> $1 7 . 7 8 2 0 . 7 8$ </td><td>73.53±1.42</td><td> $6 8 . 8 6 { \scriptstyle \pm 1 . 2 0 }$ </td><td> $4 1 . 2 0 { \scriptstyle \pm 2 . 0 8 }$ </td><td> $5 4 . 1 5 { \scriptstyle \pm 0 . 0 3 }$ </td><td> $5 3 . 3 2 { \scriptstyle \pm 0 . 8 8 }$ </td><td> $2 0 . 0 3 { \scriptstyle \pm 0 . 2 0 }$ </td></tr><tr><td>TGAT</td><td> $5 4 . 0 6 { \scriptstyle \pm 0 . 4 7 }$ </td><td>52.84±0.14</td><td>19.76±0.37</td><td> $5 6 . 5 9 { \scriptstyle \pm 0 . 0 9 }$ </td><td> $5 6 . 1 5 { \scriptstyle \pm 0 . 1 5 }$ </td><td> $2 1 . 8 8 { \scriptstyle \pm 0 . 0 5 }$ </td><td> $5 2 . 9 7 { \scriptstyle \pm 0 . 7 9 }$ </td><td>52.02±0.25</td><td>19.04±0.68</td></tr><tr><td>TGN</td><td> $4 9 . 3 9 { \scriptstyle \pm 0 . 3 1 }$ </td><td> $4 8 . 1 7 { \scriptstyle \pm 0 . 3 9 }$ </td><td> $1 6 . 4 8 { \scriptstyle \pm 0 . 2 3 }$ </td><td> $6 3 . 5 2 { \scriptstyle \pm 0 . 4 4 }$ </td><td> $5 8 . 7 8 { \scriptstyle \pm 0 . 3 6 }$ </td><td> $2 9 . 3 3 { \scriptstyle \pm 0 . 7 6 }$ </td><td> $5 0 . 7 0 { \scriptstyle \pm 0 . 9 0 }$ </td><td> $4 8 . 6 6 { \scriptstyle \pm 0 . 4 1 }$ </td><td> $1 7 . 6 6 { \scriptstyle \pm 0 . 7 9 }$ </td></tr><tr><td> $\mathrm { C A W N }$ </td><td> $5 2 . 8 1 { \scriptstyle \pm 0 . 1 7 }$ </td><td> $5 3 . 4 9 { \scriptstyle \pm 0 . 2 2 }$ </td><td> $1 8 . 6 0 { \scriptstyle \pm 0 . 1 2 }$ </td><td> $5 6 . 4 9 { \scriptstyle \pm 0 . 4 7 }$ </td><td> $5 5 . 4 3 { \scriptstyle \pm 0 . 3 4 }$ </td><td> $2 1 . 8 9 { \scriptstyle \pm 0 . 4 1 }$ </td><td> $5 4 . 8 9 { \scriptstyle \pm 0 . 1 6 }$ </td><td> $5 6 . 3 4 { \scriptstyle \pm 0 . 1 4 }$ </td><td> $1 9 . 9 4 { \scriptstyle \pm 0 . 1 2 }$ </td></tr><tr><td>TCL</td><td> $5 4 . 2 3 { \scriptstyle \pm 0 . 3 3 }$ </td><td> $5 2 . 8 3 { \scriptstyle \pm 0 . 1 4 }$ </td><td> $2 0 . 0 6 { \scriptstyle \pm 0 . 1 8 }$ </td><td> $5 5 . 0 4 { \scriptstyle \pm 0 . 3 5 }$ </td><td> $5 4 . 3 0 { \scriptstyle \pm 0 . 2 7 }$ </td><td> $2 0 . 6 6 { \scriptstyle \pm 0 . 2 8 }$ </td><td> $5 6 . 7 5 { \scriptstyle \pm 0 . 7 6 }$ </td><td> $5 5 . 3 7 { \scriptstyle \pm 0 . 6 9 }$ </td><td> $2 2 . 3 8 { \pm } 0 . 5 5$ </td></tr><tr><td> $\mathbf { G r a p h M i x e r }$ </td><td> $5 5 . 6 3 { \scriptstyle \pm 1 . 0 5 }$ </td><td> $5 6 . 6 3 { \scriptstyle \pm 0 . 7 8 }$ </td><td> $2 0 . 7 4 { \scriptstyle \pm 0 . 8 8 }$ </td><td> $7 8 . 7 0 { \scriptstyle \pm 0 . 7 3 }$ </td><td> $7 6 . 6 3 { \scriptstyle \pm 0 . 3 9 }$ </td><td> $4 6 . 8 9 { \scriptstyle \pm 1 . 2 7 }$ </td><td> $5 8 . 1 3 { \scriptstyle \pm 0 . 0 3 }$ </td><td> $5 9 . 0 4 { \scriptstyle \pm 0 . 2 1 }$ </td><td> $2 2 . 7 1 { \scriptstyle \pm 0 . 1 2 }$ </td></tr><tr><td> $\mathrm { D y } \mathrm { G F o r m e r }$ </td><td> $5 5 . 9 5 { \scriptstyle \pm 0 . 2 1 }$ </td><td> $5 4 . 0 7 { \scriptstyle \pm 0 . 1 9 }$ </td><td> $2 1 . 2 7 { \scriptstyle \pm 0 . 2 0 }$ </td><td>56.36±0.52</td><td> $5 5 . 7 4 { \scriptstyle \pm 0 . 5 0 }$ </td><td> $2 1 . 5 5 { \scriptstyle \pm 0 . 3 7 }$ </td><td> $5 7 . 2 3 { \scriptstyle \pm 0 . 1 1 }$ </td><td> $5 7 . 8 4 { \scriptstyle \pm 0 . 2 0 }$ </td><td> $2 1 . 9 0 { \scriptstyle \pm 0 . 2 0 }$ </td></tr><tr><td> $\mathrm { F r e e D y G }$ </td><td> $5 4 . 4 4 { \scriptstyle \pm 0 . 6 8 }$ </td><td> $5 6 . 0 3 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $1 9 . 7 3 { \scriptstyle \pm 0 . 6 2 }$ </td><td>81.28±0.69</td><td> $7 8 . 7 3 { \scriptstyle \pm 0 . 5 1 }$ </td><td> $5 1 . 0 6 { \scriptstyle \pm 1 . 3 2 }$ </td><td> $6 0 . 2 3 { \scriptstyle \pm 0 . 5 6 }$ </td><td> $6 2 . 7 9 2 0 . 9 8 $ </td><td> $2 3 . 9 8 { \scriptstyle \pm 0 . 3 4 }$ </td></tr><tr><td> $\mathrm { L K D 4 D y T A G }$ </td><td> $5 1 . 1 9 { \scriptstyle \pm 0 . 0 5 }$ </td><td> $5 0 . 6 1 { \scriptstyle \pm 0 . 1 3 }$ </td><td> $1 7 . 7 0 { \scriptstyle \pm 0 . 0 1 }$ </td><td>56.21±0.10</td><td> $5 5 . 1 5 { \scriptstyle \pm 0 . 0 7 }$ </td><td> $2 1 . 6 3 { \scriptstyle \pm 0 . 0 9 }$ </td><td> $5 1 . 1 2 { \scriptstyle \pm 0 . 2 5 }$ </td><td> $5 0 . 6 4 { \scriptstyle \pm 0 . 3 5 }$ </td><td> $1 7 . 6 4 \pm 0 . 1 7$ </td></tr><tr><td>CROSS</td><td> $5 4 . 5 8 { \scriptstyle \pm 1 . 1 3 }$ </td><td> $5 2 . 9 6 { \scriptstyle \pm 0 . 7 0 }$ </td><td> $2 0 . 3 2 { \scriptstyle \pm 0 . 8 1 }$ </td><td>59.80±0.01</td><td> $5 9 . 5 2 { \scriptstyle \pm 0 . 1 6 }$ </td><td> $2 4 . 2 5 { \scriptstyle \pm 0 . 0 7 }$ </td><td> $5 8 . 0 0 { \scriptstyle \pm 1 . 1 9 } \quad$ </td><td> $5 8 . 4 4 { \scriptstyle \pm 0 . 7 6 }$ </td><td> $2 2 . 6 7 { \scriptstyle \pm 1 . 1 4 }$ </td></tr></table>

Table 22: Performance comparison of TGNNs on the TNC task under two training strategies: linear probing (LP) and end-to-end fine-tuning (E2E). Linear probing trains only the classifier head on a TLP-pretrained backbone, while end-to-end fine-tuning jointly optimizes both the backbone and the classifier. Results are reported in terms of Macro-F1.
<table><tr><td rowspan="2">Methods</td><td colspan="2">FOOD</td><td colspan="2">IMDB</td><td colspan="2">Beeradvocate</td><td colspan="2">Amazon-Kindle</td><td colspan="2">Ratebeer</td></tr><tr><td>LP</td><td>E2E</td><td>LP</td><td>E2E</td><td>LP</td><td>E2E</td><td>LP</td><td>E2E</td><td>LP</td><td>E2E</td></tr><tr><td>JODIE</td><td>25.95</td><td>25.94</td><td>12.65</td><td>12.65</td><td>7.00</td><td>6.95</td><td>17.60</td><td>13.18</td><td>7.86</td><td>8.03</td></tr><tr><td>DyRep</td><td>21.22</td><td>21.40</td><td>13.74</td><td>13.77</td><td>6.50</td><td>6.49</td><td>13.36</td><td>14.02</td><td>7.38</td><td>8.62</td></tr><tr><td>TGAT</td><td>22.73</td><td>22.73</td><td>8.53</td><td>8.53</td><td>6.18</td><td>6.14</td><td>14.91</td><td>15.34</td><td>8.67</td><td>8.66</td></tr><tr><td>TGN</td><td>21.98</td><td>21.80</td><td>8.51</td><td>8.51</td><td>8.35</td><td>7.95</td><td>13.22</td><td>OOM</td><td>7.34</td><td>8.22</td></tr><tr><td>CAWN</td><td>20.28</td><td>19.99</td><td>11.48</td><td>11.49</td><td>8.77</td><td>8.66</td><td>14.96</td><td>15.96</td><td>8.79</td><td>9.04</td></tr><tr><td>TCL</td><td>19.99</td><td>19.99</td><td>14.56</td><td>13.60</td><td>9.26</td><td>8.17</td><td>24.25</td><td>27.06</td><td>14.34</td><td>14.45</td></tr><tr><td>GraphMixer</td><td>19.99</td><td>19.99</td><td>9.20</td><td>9.29</td><td>6.14</td><td>6.14</td><td>13.35</td><td>14.62</td><td>7.13</td><td>7.27</td></tr><tr><td>DyĠFormer</td><td>36.58</td><td>26.58</td><td>20.05</td><td>12.85</td><td>12.91</td><td>10.16</td><td>20.19</td><td>19.16</td><td>15.52</td><td>13.79</td></tr><tr><td>FreeDyG</td><td>20.46</td><td>20.88</td><td>10.20</td><td>10.02</td><td>7.75</td><td>7.24</td><td>12.80</td><td>13.77</td><td>8.54</td><td>8.28</td></tr></table>

![](images/4ddfe19b01c616cf85bec79567824a82ba9f6f22731610fc341aa2f39582a9a8.jpg)  
Figure 8: Scalability results across all methods.