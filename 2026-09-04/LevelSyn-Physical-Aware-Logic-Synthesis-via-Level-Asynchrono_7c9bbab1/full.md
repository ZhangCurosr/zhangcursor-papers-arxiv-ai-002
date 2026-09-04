# LevelSyn: Physical-Aware Logic Synthesis via Level-Asynchronous Graph Neural Networks

Jingyi Zhou, Zhengyuan Shi<sup>∗</sup>, Ziyang Zheng, Qiang Xu<sup>∗</sup> The Chinese University of Hong Kong, Hong Kong, China {jyzhou26,zyshi21,zyzheng23,qxu}@cse.cuhk.edu.hk

## Abstract

As integrated circuit technology scales into the nanometer regime, the traditional disconnect between logic synthesis and physical design has led to significant PPA (Power, Performance, and Area) degradation and prolonged design closure cycles. Traditional logic synthesis relies on non-physical Wire Load Models (WLMs), while recent spectral-based placement predictors often neglect the inherent hierarchical logic depth and signal flow of netlists, which leads to low-fidelity spatial estimations. To bridge this gap, we propose LevelSyn, a novel physical-aware logic synthesis framework that integrates hierarchical representation learning with a wirelength-driven optimization engine. At its core, LevelSyn leverages a level-asynchronous Graph Neural Network (GNN) to predict high-fidelity gate coordinates by capturing the structural and directional semantics of And-Inverter Graphs (AIGs). To handle industrial-scale designs, a level-aligned subgraph partitioning strategy is introduced to eliminate memory bottlenecks while preserving local logical dependencies. These spatial insights are seamlessly integrated into a newly developed physical-informed synthesis engine within the Berkeley ABC framework. Experimental results on the EPFL benchmark suite demonstrate that LevelSyn signifi cantly outperforms state-of-the-art (SOTA) methods, achieving an average power reduction of 6.89% and a timing delay improvement of 27.48%. Furthermore, post-place-and-route validation shows a 99.59% reduction in design rule check (DRC) violations, highlighting its efectiveness in accelerating design convergence.

## 1 Introduction

To avoid local optimization and time-consuming design iterations within the current stage-based chip design workflow, the Electronic Design Automation (EDA) industry has actively adopted a “shift-left” strategy [5, 35]. The future design metrics and potential violations should be exposed at much early stages, so that engineers can make revisions on the “left” of the workflow.

Physical-aware synthesis [25] is a prominent example in logic synthesis domain. Unlike conventional tools [23, 34] estimate the design quality based on models only considering the logical abstraction (e.g., wire load models), Physical-aware synthesis tries to combine the backend information to enable accurate estimation of interconnect delay and congestion during early stages. Although logic synthesis can produce promising results, they are often poorly suited for physical implementation in modern nanometerscale designs [13, 22]. To bridge this gap, the early attempts explored the physical-aware synthesis, which incorporates physical information into the early stages of logic synthesis, thereby closing the gap between logical abstraction and physical implementation [3, 10, 11, 26, 27].

Most recently, graph analysis techniques have been employed to predict physical metrics. For example, Net<sup>2</sup>[37] and another attempt [36] leverage Graph Neural Networks (GNNs) for preplacement wirelength estimation. CPA-Remap [12] identifies the critical paths and merges and then remaps the netlist for timing optimization. However, these predictive models are tied to a specific circuit structure. As the logic synthesis constantly reshapes the logic network, these methods are limited to minor refinement after synthesis. Other frameworks [12, 16, 24] combine the fast physical estimation [21] into logic optimization and technology mapping tools. Although they can predict the connectivity and spatial distance between logic cells, other critical information, such as hierarchical level structure and directional signal flow, are neglecting. Therefore, eficiently estimating high-fidelity physical information for logic synthesis remains a significant challenge.

In this work, we propose LevelSyn, a physical-aware logic synthesis framework that leverages a hierarchical neural network to estimate spatial coordinates and orientations, bridging the gap between logical abstraction and physical implementation. Specifically, our framework leverages a single-round and level-asynchronous GNN (LA-GNN) architecture inspired by DeepGate2 [29]. We propose novel mechanism to encode and propagate the pre-defined constraints of Primary Inputs (PIs) and Primary Outputs (POs). Unlike existing physical-aware synthesis methods, LevelSyn produces row-consistent spatial priors. In this approach, the cell orientation is directly derived from the predicted �-coordinates based on the row specifications in the PDK. This ensures that each cell aligns correctly with the power rails without additional processing. Moreover, to ensure the scalability for industrial-scale designs, we employ a level-aligned subgraph partitioning strategy that mitigates memory overhead.

We implement our model into an open-source logic synthesis tool ABC [23]. Our model provides the spatial insights of the given pre-mapping logic and guides the technology mapping engine by redefining its cost function. The enhanced ABC tool produces the post-synthesize circuit netlists that are inherently optimized for physical implementation. Using the DREAMPlace [19] for physical design validation, the experimental results on the EPFL benchmark suite demonstrate that LevelSyn outperforms state-of-the-art methods, achieving an average power reduction by 6.89% and a timing delay reduction by 27.48%. Furthermore, post-place-and-route validation reveals a 99.59% reduction in design rule check (DRC) violations, highlighting the framework’s efectiveness in generating layout-friendly netlists and accelerating design convergence for advanced chip products.

The contributions of this work is summarized as follows:

• We propose a predictive engine based on the Level-Asynchronous GNN (LA-GNN) that explicitly captures the directional signal flow and hierarchical logic depth of And-Inverter Graphs (AIGs). To ensure scalability for industrial-scale designs, we introduce a Level-Aligned Subgraph Partitioning (LASP) strategy that mitigates memory bottlenecks while preserving critical local logical dependencies.

• We develop a high-fidelity spatial prediction scheme that esti mates gate coordinates and captures the underlying standardcell row structure. By leveraging these predictions, the framework derives deterministic priors for cell orientation and power rail alignment, providing more comprehensive physical guidance than traditional spectral-based methods.

• We provide LevelSyn, an end-to-end framework combining GNN with open-source logic synthesis tool. By substituting traditional heuristics with a physical-informed cost function, our framework achieves better performance, lower power consumption and lower post-routing DRC violations within the acceptable runtime.

## 2 Related Works

## 2.1 Evolution of Physical Aware Synthesis

Traditionally, VLSI design flows maintained a strict separation between front-end logic synthesis and back-end physical design. However, as semiconductor technology scaled into the deep sub-micron (DSM) era, interconnect delays began to dominate gate delays, rendering traditional logic-depth metrics insuficient for timing closure [13].

To bridge this gap, early research explored the integration oflogic optimization and placement. Gosti et al. [11] addressed timing clo sure by interleaving logic operations with global placement, while simultaneous technology mapping and placement algorithms like Maple [32, 33] sought to incorporate spatial information directly into the mapping process. Despite these eforts, early techniques often sufered from high computational complexity or relied on inaccurate wire-load models (WLM) that failed to capture the nuances of modern routing congestion. Recent industry shifts toward frameworks like OpenROAD [31] provide better infrastructure for integration, yet achieving a truly unified metric remains a challenge.

The PigMap series [16, 24] introduced a paradigm shift by utilizing primitive logic gate placement to guide technology mapping. PigMap1 extracts high-fidelity spatial information, specifically estimated coordinates, from a primitive placement of technology independent gates. This spatial prior allows the mapper to optimize for performance (minimizing critical path Manhattan distance) or power (reducing total wirelength). PigMap2 further refined this by incorporating multi-dimensional metrics such as congestion-aware placement. While successful, these methods rely on fast placement engines that often yields inaccurate spatial data, creating a fidelity gap that prevents the subsequent mapping phase from achieving truly optimal results.

## 2.2 GNNs in EDA and Their Limitations

With the rise of machine learning, Graph Neural Networks (GNNs) have become a popular choice for EDA tasks due to the natural representation of netlists as graphs [4]. Extensive research has focused on learning structural and functional representations for combinational circuits. For instance, DeepGate [17] utilizes attention-based aggregation and skip connections to encode Boolean computations, benefiting downstream tasks such as test point insertion [28] and Boolean Satisfiability (SAT) [18]. This was further evolved by Deep-Gate2 [29], which introduced pairwise truth-table supervision and functionality-aware loss functions to enhance scalability. Other approaches have explored contrastive learning to derive invariant representations, such as FGNN [7] for circuit classification, as well as CircuitFusion [9] and NetTAG [8] for multi-modal circuit data. Furthermore, Polargate [20] enhanced representation quality by employing the bipolar principle.

Despite these advancements in encoding logical functionality, applying standard GNN architectures to large-scale physical-aware logic synthesis presents significant practical hurdles. First, the primary bottleneck lies in memory consumption. Modern netlists containing millions of gates require massive graph structures and message-passing bufers that frequently exceed the capacity of standard GPU hardware [29]. Second, the iterative nature of graph convolutions introduces a substantial runtime overhead compared to traditional heuristics. In a high-speed synthesis environment where rapid design iterations are essential, the inference latency of complex GNNs often outweighs their accuracy benefits [14]. Most importantly, while existing representation learning excels at capturing Boolean behavior, it does not natively provide the explicit spatial coordinates required for physical optimization. Therefore, there is a growing need for more eficient neural architectures that can directly provide accurate spatial priors without the prohibitive computational and memory costs associated with standard GNN-based representation learning.

## 3 Methodology

## 3.1 Overview of the Proposed Framework

The LevelSyn framework is an end-to-end, physical-aware logic synthesis flow designed to bridge the gap between logical abstraction and physical implementation. As illustrated in Fig. 1, the framework consists of three main phases: Hierarchical Pre-processing, Scalable Spatial Prediction, and Physical-Informed Technology Mapping.

3.1.1 Problem Formulation. We represent the input logic netlist as an And-Inverter Graph (AIG), denoted by a directed acyclic graph $G = ( V , E )$ . Each node � ∈ � represents a logic gate (2-input AND or Inverter), and each edge $e \in E$ represents the signal flow. Given the fixed boundary constraints from the floorplan stage, specifically the spatial coordinates and the orientations of primary inputs and outputs (PI/POs), denoted as $C _ { f i x e d } = \{ ( x _ { i } , y _ { i } , O _ { i } ) \ | \ i \in \mathrm { P I \cup P O } \}$ Our objective is twofold:

• Predict high-fidelity spatial coordinates and capture orientations $\left( x _ { v } , y _ { v } , O _ { v } \right)$ for all internal gates $v \in V \ \backslash$ (PI ∪ PO).

![](images/81f59ee0f5ce2935f25dfc6f15e6754de413534090614af5b1790ccaea0567cf.jpg)  
Figure 1: Overview of LevelSyn framework. (A) AIG floorplan to obtain IO constraints. (B) Cell spatial coordinates inference with layer-asynchronous GNN. (C) Physical-informed mapping technique to generate post-mapping netlists.

• Optimize the technology mapping process by utilizing these coordinates to minimize post-routing wirelength and delay.

## 3.1.2 Framework Pipeline. The execution flow of LevelSyn follows a structured sequence:

Hierarchical Pre-processing. The framework first parses the AIG to extract topological features and hierarchical level information. To ensure global spatial awareness, fixed PI/PO coordinates are encoded as spatial anchors. For industrial-scale designs, a level aligned subgraph partitioning strategy is applied to decompose the flattened graph into manageable segments, preventing memory overflow while maintaining logical continuity.

Scalable Spatial Prediction. The partitioned AIG segments are fed into the Level-Asynchronous GNN. Unlike standard GNNs that perform undirected message passing, our engine propagates features along the directional signal flow, which could capture the correlation between logic depth and physical placement. The output is a set of spatial priors $( \hat { x } _ { v } , \hat { y } _ { v } , \hat { O } _ { v } )$ for every gate in the netlist.

Physical-Aware Technology Mapping. These spatial priors are integrated into the Berkeley ABC engine. By extending the mapping cost function to include wirelength-driven metrics, LevelSyn enables the synthesis engine to make physical-aware choices during gate-level optimization, resulting in a netlist with superior PPA.

## 3.2 Level-Aware GNN Architecture

The core of our predictive engine is a Level-Aware Graph Neural Network (LA-GNN) designed to map the logical structure of an AIG to high-fidelity spatial priors. LA-GNN explicitly respects the directional signal flow and hierarchical levels of the netlist.

3.2.1 Level-Asynchronous Message Passing. To capture the logic depth and signal propagation, we first partition the AIG nodes into sequential levels $\{ L _ { 1 } , L _ { 2 } , . . . , L _ { D } \}$ , where � is the maximum logic depth. A node � belongs to level $L _ { k }$ if the maximum path length from any PI to � is �.

The initial feature vector $z _ { v }$ for each node $v \in V$ is constructed by concatenating structural, functional, and spatial attributes to provide a rich starting point for the level-asynchronous message passing. Specifically, $z _ { v }$ is defined as:

![](images/c19f04fe265a55b77e9b9098624897ac65e4bc00ca69d4e4209bd1d84709aa95.jpg)  
Figure 2: Subgraph partitioning scheme. (A) Level-aligned subgraph partioning overview. (B) Global positional priors injection and parallel GPU batch processing

$$
z _ { v } = \left[ T _ { v } \parallel S _ { v } \parallel D _ { v } \right]\tag{1}
$$

where ∥ denotes the concatenation operator. The components are defined as follows:

Gate Type Encoding (�<sub>�</sub>): We use a one-hot encoding vector to represent the functional type of the node.

Spatial Anchors $( S _ { v } ) \colon$ : This is a critical feature for shift-left physical awareness. For nodes $v \in \operatorname { P I } \cup \operatorname { P O } , S _ { v }$ contains their fixed (�, �) coordinates obtained from the floorplan stage, normalized to the range [0, 1]. For internal logic gates whose locations are unknown, $S _ { v }$ is initialized with the center of the layout (0.5, 0.5) for a place-holder value. This allows the GNN to treat PI/POs as spatial anchors that guide the coordinate regression of internal nodes.

Structural Attributes $( D _ { v } ) \colon$ To capture the local complexity of the netlist, we include the fan-in and fan-out degrees of each node. Additionally, the pre-calculated topological level index is embedded as a numerical feature, helping the GNN perceive its relative depth within the entire signal path.

To ensure that every internal gate perceives the spatial constraints from both PIs and POs, we propose a Bi-directional Level-Asynchronous Message Passing mechanism. This approach consists of a forward pass to capture signal flow and a backward pass to difuse spatial constraints from the outputs.

Forward Signal Propagation. We first propagate features along the topological order from level $L _ { 1 }$ to $L _ { D }$ . For a node $v \in L _ { k } ,$ , the forward hidden state $\vec { h } _ { \upsilon } ^ { ( k ) }$ captures the logical influence from its fan-in:

$$
\overrightarrow { m } _ { v } ^ { ( k ) } = \mathrm { A G G R E G A T E } \left( \overrightarrow { h } _ { u } ^ { ( j ) } \mid u \in \mathrm { F I } ( v ) , j < k \right)\tag{2}
$$

$$
\overrightarrow { h } v ^ { ( k ) } = \sigma \left( \mathbf { W } f w d \cdot \mathrm { C O N C A T } ( \overrightarrow { m } _ { v } ^ { ( k ) } , z _ { v } ) \right)\tag{3}
$$

Backward Constraint Difusion. To incorporate PO spatial constraints, a second pass is performed in reverse topological order $\left( L _ { D } \mathrm { t o } L _ { 1 } \right)$ . For a node $v \in L _ { k }$ , its backward hidden state $\overleftarrow { h } _ { v } ^ { ( k ) }$ is updated by its fan-out nodes (FO):

$$
\overleftarrow { m } _ { v } ^ { ( k ) } = \mathrm { A G G R E G A T E } \left( \overleftarrow { h } _ { u } ^ { ( j ) } \mid u \in \mathrm { F O } ( v ) , j > k \right)\tag{4}
$$

$$
\overleftarrow { h } v ^ { ( k ) } = \sigma \left( \mathbf { W } b w d \cdot \mathbf { C O N C A T } ( \overleftarrow { m } _ { v } ^ { ( k ) } , z _ { v } ) \right)\tag{5}
$$

Final Embedding Fusion. The final high-dimensional embedding $h _ { v } ^ { * }$ is the concatenation of both directional states:

$$
h _ { v } ^ { * } = [ \overrightarrow { h } _ { v } ^ { ( k ) } \parallel \overleftarrow { h } _ { v } ^ { ( k ) } ]\tag{6}
$$

. This dual-flow architecture ensures that the structural representation of each gate is constrained by PI and PO anchors, allowing the downstream MLP to interpolate coordinates that respect the global floorplan.

3.2.2 Orientation Estimation. In a standard-cell-based layout, a cell’s orientation is not an independent variable but is often constrained by its vertical position relative to the power rails. We derive the orientation estimation $\hat { O } _ { z }$ based on the predicted $\hat { y } _ { v }$ and the row structure defined in the floorplan. First, we map the predicted $\hat { y } _ { v }$ to a discrete row index $R _ { v }$

$$
R _ { v } = \mathrm { r o u n d } \left( \frac { \hat { y } _ { v } - y _ { o f f s e t } } { H _ { r o w } } \right)\tag{7}
$$

where $H _ { r o w }$ is the standard cell row height and $y _ { o f f s e t }$ is the layout boundary. Given the power rail sharing property (VDD-VSS-VSS-VDD), the orientation $\hat { O } _ { v }$ is estimated as:

$$
\hat { O } _ { v } = \left\{ \begin{array} { l l } { \mathrm { N } , } & { \mathrm { i f } R _ { v } \quad \mathrm { m o d } \ 2 = 0 } \\ { \mathrm { F S } , } & { \mathrm { i f } R _ { v } \quad \mathrm { m o d } \ 2 = 1 } \end{array} \right.\tag{8}
$$

This mechanism ensures that the predicted spatial priors $( \hat { x } _ { v } , \hat { y } _ { v } , \hat { O } _ { v } )$ are physically consistent with the target layout environment, allowing the synthesis engine to better estimate the pin locations and parasitic efects during the mapping phase.

## 3.3 Scalable Subgraph Partitioning

To ensure the scalability of LevelSyn on modern designs while preserving the directional logic dependencies, we propose a Level-Aligned Subgraph Partitioning (LASP) strategy (Figure2).

Unlike traditional graph partitioning algorithms [6, 15] that focus on minimizing edge cuts but often disrupt the topological flow, our LASP strategy respects the hierarchical levels of the AIG. Given an AIG with a maximum logic depth �, we divide the graph into � contiguous segments along the level dimension. Each subgraph $G _ { k } = \left( V _ { k } , E _ { k } \right)$ is defined as:

$$
V _ { k } = \left\{ v \in V \mid ( k - 1 ) \cdot w \leq \mathrm { L e v e l } ( v ) < k \cdot w \right\}\tag{9}
$$

where $w = \lceil L / N \rceil$ is the level window size. This partitioning ensures that each subgraph contains a complete "logic slice" of the circuit, maintaining the local connectivity between adjacent logic levels which is critical for spatial proximity estimation.

A key challenge in independent subgraph partitioning is the loss of global coordinates. Instead of using hardware-ineficient sequential propagation, LevelSyn injects Global Positional Priors directly into the initial node features $z _ { v } .$ We define a Normalized Level Index (NLI) for each node:

$$
\mathrm { N L I } ( v ) = \frac { \mathrm { L e v e l } ( v ) } { L }\tag{10}
$$

The NLI serves as a global GPS for the GNN, allowing each subgraph—processed in parallel within a GPU batch to perceive its relative distance between the PI $( \mathrm { N L I } \approx 0 )$ and PO $( \mathrm { N L I } \approx 1 )$ boundaries. By combining NLI with the fixed PI/PO spatial anchors, the model can maintain a globally consistent coordinate system.

By controlling the window size $w ,$ the memory complexity of each GNN pass is reduced from $O ( | V | )$ to $O ( | V | / N )$ , efectively capping the peak memory usage regardless of the total circuit size. This enables LevelSyn to handle industrial benchmarks that are orders of magnitude larger than those typically supported by spectral-based methods, which often sufer from the cubic complexity of Laplacian eigen-decomposition.

## 3.4 Physical-Aware Technology Mapping

Figure 3 illustrates the final stage of the LevelSyn framework. It integrates the predicted spatial priors into the technology mapping process within the Berkeley ABC [2] engine. Drawing inspiration from the physical-aware synthesis paradigm established by PigMAP [24], we bridge the gap between logical structural optimization and physical layout requirements.

3.4.1 Physical Data Integration. To enable physical awareness, the predicted coordinates $( \hat { x } _ { v } , \hat { y } _ { v } )$ and orientations $\hat { O } _ { v }$ for every AIG node are back-annotated into the ABC data structure. We extend the AIG node representation in ABC to include a physical attribute vector $\mathcal { P } _ { v } = ( \hat { x } _ { v } , \hat { y } _ { v } , \hat { O } _ { v } )$ . During the mapping phase, these attributes are used to estimate the wirelength of potential subject graphs (matches) before they are committed to the final netlist.

![](images/1df1092332626c6ca860fd4bbb1228c14bbfe53c5f1d985c01b84e50ce55a23b.jpg)  
Figure 3: Explanation of the technology mapping flow

3.4.2 Wirelength-Driven Cost Function. In standard technology mapping, the engine selects a library gate (cell) to cover a portion of the AIG by minimizing a cost function typically focused on area or arrival time. LevelSyn redefines this cost function to include a Predicted Wirelength (PWL) metric. Following the diferentiated cost evaluation strategy proposed in PigMAP [24], we adapt the PWL calculation based on the specific optimization target to better reflect the underlying physical impact. For a candidate match � that implements a set of AIG nodes, the augmented cost C(�) is defined as:

$$
C ( M ) = \alpha \cdot \operatorname { A r e a } ( M ) + \beta \cdot \operatorname { D e l a y } ( M ) + \gamma \cdot \operatorname { P W L } ( M )\tag{11}
$$

where $\alpha , \beta ,$ � are weighting factors. The PWL(�) is estimated using the Manhattan distance between the predicted center of the candi date cell $v _ { M }$ and its fan-in/fan-out neighbors. For power-oriented command pwmap, we calculated the sum of Manhattan distance to all adjacent pins to minimize total switching power:

$$
\mathrm { P W L } ( M ) = \sum _ { u \in \mathrm { a d j } ( v _ { M } ) } \left( | \hat { x } _ { v _ { M } } - \hat { x } _ { u } | + | \hat { y } _ { v _ { M } } - \hat { y } _ { u } | \right)\tag{12}
$$

Conversely, for delay-oriented mapping command pfmap, we employ the maximum Manhattan distance to bound the worst-case interconnect delay on critical paths:

$$
\operatorname { P W L } ( M ) = \operatorname* { m a x } _ { u \in \operatorname { a d j } ( v _ { M } ) } \left( \left| \hat { x } _ { v _ { M } } - \hat { x } _ { u } \right| + \left| \hat { y } _ { v _ { M } } - \hat { y } _ { u } \right| \right)\tag{13}
$$

3.4.3 Mapping Execution and Optimization. The execution of the physical-informed mapping is implemented as an enhanced flow within the Berkeley ABC engine, extending standard mapping com mands such as map, pfmap, and pwmap. During this process, the engine first identifies all potential library cell matches for the subject AIG and evaluates them using the augmented cost function described in Section 3.4.2. By prioritizing matches that align with the predicted spatial trajectory, in particular those that minimize net stretching relative to the fixed PI/PO anchors, the engine ensures a strong correlation between logical structures and physical constraints. This is followed by an iterative refinement phase where LevelSyn performs local gate-level swapping and sizing; a gate is replaced if a functionally equivalent library cell provides a superior fit to the predicted spatial prior, thereby reducing estimated wirelength without violating timing constraints. By embedding physical location directly into the mapping decision tree, LevelSyn generates a netlist that is inherently layout-friendly, significantly mitigating routing congestion and wire-induced delay in the subsequent physical design stages.

## 4 Experiments

In this section, we evaluate the performance of LevelSyn against industry-standard baselines and SOTA physical-aware synthesis methods. We focus on the framework’s ability to optimize PPA (Power, Performance, and Area) metrics across a diverse set of benchmarks.

## 4.1 Experimental Settings

Benchmark Suite. We utilize the EPFL Combinational Benchmark Suite [1] to assess scalability. The circuit sizes range from hundreds of nodes to over 200k nodes.

EDA Flow. All designs are processed using Berkeley ABC for logic synthesis and technology mapping. For physical ground truth and spatial anchor generation, we employ DREAMPlace [19] to obtain high-quality placement coordinates. The target library is the SkyWater Open Source PDK [30] standard cell library.

GNN Training Configuration. To ensure the reproducibility of our LA-GNN model, we detail the training environment and hyperparameters as follows: We employ a multi-task loss function $\mathcal { L } = \mathcal { L } _ { M S E } ( \hat { x } , \hat { y } ) + \mathcal { L } _ { B C E } ( \hat { O } )$ to simultaneously optimize coordinate regression and orientation prediction. $\mathcal { L } _ { M S E }$ denotes the Mean Squared Error for the predicted coordinates, while $\mathcal { L } _ { B C E }$ represents the Binary Cross-Entropy loss for orientation classification. The model is trained using the AdamW optimizer with an initial learning rate of $1 \times 1 0 ^ { - 3 }$ and a weight decay of $1 \times 1 0 ^ { - 4 }$ . All computational tasks are performed on a Linux workstation equipped with a NVIDIA A100 GPU. We use a hidden dimension of 128 and 6 message-passing layers. The partition window size � is set to 5000 nodes.

Baselines. We employ both the industry-standard baselines and SOTA synthesis methods as baselines, and we address them as follows:

• Baseline: Standard logic-driven mapping in ABC (map).

• PigMap-power/performance [24]: The SOTA physicalaware mapping that uses analytical wireload models for power and delay optimization.

## 4.2 GNN Pre-training and Fine-tuning Results

To enhance the model’s ability to capture generalized structural representations of logic circuits, we adopt a two-stage training strategy consisting of large-scale pre-training followed by task-specific finetuning. In the pre-training phase, we construct a diverse dataset comprising 500 to 10,000 randomly generated AIGs with varying topological complexities, which allows the LA-GNN to learn intrinsic geometric and logical features across a broad synthetic design space. Subsequently, the pre-trained weights are used as an initialization for the model, which is then fine-tuned on specific circuits from established benchmarks to adapt the learned representations to realistic logic synthesis and physical design constraints. This transfer learning approach not only accelerates the convergence process but also significantly improves the model’s predictive accuracy and robustness when encountering the structural variations inherent in diferent circuit categories.

Table 1: PPA Comparison under Diferent Technology Mapping Strategies
<table><tr><td rowspan="2">Circuit</td><td rowspan="2">Nodes</td><td colspan="3">Baseline</td><td colspan="3">PigMap-power</td><td colspan="3">PigMap-performance</td></tr><tr><td>Area(um²)</td><td>Delay(ps)</td><td>Power(mW)</td><td>Area(um²)</td><td>Delay(ps)</td><td>Power(mW)</td><td>Area(um²)</td><td>Delay(ps)</td><td>Power(mW)</td></tr><tr><td>ctrl</td><td>181</td><td>571.80</td><td>760.23</td><td>0.0063</td><td>604.33</td><td>737.59</td><td>0.0089</td><td>539.27</td><td>826.13</td><td>0.0078</td></tr><tr><td>int2float</td><td>271</td><td>938.40</td><td>886.41</td><td>0.0084</td><td>1,068.52</td><td>1,068.23</td><td>0.0015</td><td>829.55</td><td>1,045.27</td><td>0.0110</td></tr><tr><td>dec</td><td>312</td><td>1,216.17</td><td>712.31</td><td>0.0226</td><td>1,376.32</td><td>1,257.90</td><td>0.0292</td><td>1,376.32</td><td>1,257.90</td><td>0.0292</td></tr><tr><td>router</td><td>317</td><td>1,657.84</td><td>3,685.60</td><td>0.0236</td><td>1,356.30</td><td>4,086.01</td><td>0.0201</td><td>1,177.38</td><td>3,214.47</td><td>0.0167</td></tr><tr><td>cavlc</td><td>703</td><td>2,858.99</td><td>1,391.17</td><td>0.0306</td><td>2,870.25</td><td>1,718.93</td><td>0.0467</td><td>2,312.22</td><td>1,493.83</td><td>0.0353</td></tr><tr><td>priority</td><td>1,106</td><td>6,586.32</td><td>17,277.00</td><td>0.0993</td><td>4,713.27</td><td>14,011.38</td><td>0.0693</td><td>5,268.80</td><td>17,847.96</td><td>0.0773</td></tr><tr><td>adder</td><td>1,276</td><td>7,573.51</td><td>19,252.33</td><td>0.0933</td><td>3,921.26</td><td>17,057.93</td><td>0.0511</td><td>4,128.96</td><td>18,386.91</td><td>0.0552</td></tr><tr><td>i2c</td><td>1,489</td><td>5,615.39</td><td>1,365.56</td><td>0.0623</td><td>5,874.38</td><td>1,838.38</td><td>0.0847</td><td>4,919.72</td><td>1,405.88</td><td>0.0681</td></tr><tr><td>max</td><td>3,377</td><td>18,470.21</td><td>41,957.09</td><td>0.2390</td><td>13,890.82</td><td>30,964.82</td><td>0.2150</td><td>13,346.55</td><td>28,368.59</td><td>0.2100</td></tr><tr><td>bar</td><td>3,471</td><td>10,959.26</td><td>4,993.18</td><td>0.1750</td><td>18,342.59</td><td>8,062.43</td><td>0.3260</td><td>15,162.04</td><td>7,235.70</td><td>0.2490</td></tr><tr><td>sin</td><td>5,440</td><td>43,845.80</td><td>27,658.14</td><td>0.9000</td><td>27,699.06</td><td>20,565.40</td><td>0.5240</td><td>29,926.20</td><td>30,132.74</td><td>0.5800</td></tr><tr><td>arbiter</td><td>12,095</td><td>40,650.23</td><td>8,641.04</td><td>0.6710</td><td>46,036.65</td><td>8,170.64</td><td>0.8970</td><td>38,745.91</td><td>7,782.34</td><td>0.7440</td></tr><tr><td>voter</td><td>14,759</td><td>139,540.08</td><td>8,000.55</td><td>2.3200</td><td>91,849.34</td><td>7,312.21</td><td>1.6200</td><td>93,568.48</td><td>7,970.67</td><td>1.6900</td></tr><tr><td>square</td><td>18,548</td><td>148,728.89</td><td>20,094.88</td><td>2.4500</td><td>88,584.96</td><td>17,836.24</td><td>1.4700</td><td>91,456.46</td><td>18,908.50</td><td>1.4800</td></tr><tr><td>sqrt</td><td>24,746</td><td>204,521.14</td><td>1,517,812.00</td><td>4.5300</td><td>143,030.92</td><td>789,714.62</td><td>2.8900</td><td>130,052.23</td><td>861,341.06</td><td>2.6000</td></tr><tr><td>multiplier</td><td>27,190</td><td>204,043.19</td><td>29,332.91</td><td>3.9100</td><td>133,090.14</td><td>24,757.79</td><td>2.3800</td><td>133,267.81</td><td>28,860.17</td><td>2.3900</td></tr><tr><td>log2</td><td>32,092 48,040</td><td>260,310.91</td><td>168,470.03 68,602.83</td><td>5.2700</td><td>150,314.16 203,206.14</td><td>42,295.55</td><td>2.7700</td><td>149,226.86</td><td>49,459.25</td><td>2.7900</td></tr><tr><td>mem_ctrl div</td><td>57,375</td><td>248,791.11 523,522.09</td><td>690,338.06</td><td>4.0700</td><td>280,609.12</td><td>24,483.87</td><td>3.5000 5.1100</td><td>173,094.75</td><td>19,936.07</td><td>2.9800</td></tr><tr><td>hyp</td><td>214,591</td><td>1,760,069.25</td><td>5,888,406.50</td><td>10.7000 33.2000</td><td>1,171,167.00</td><td>399,128.59</td><td></td><td>277,827.69 1,203,100.12</td><td>436,291.28</td><td>5.1700</td></tr><tr><td>Circuit</td><td></td><td></td><td colspan="3"></td><td>4,321,252.00</td><td colspan="2">20.3000</td><td>4,261,045.00</td><td>20.1000</td></tr><tr><td rowspan="3"></td><td rowspan="3">Nodes</td><td></td><td></td><td>LevelSyn-pwmap (Ours)</td><td></td><td></td><td>LevelSyn-pfmap (Ours)</td><td></td><td></td><td></td></tr><tr><td></td><td>Area (Imp.†)</td><td>Delay (Imp.†)</td><td>Power (Imp.†)</td><td>Area (Imp.‡)</td><td>Delay (Imp.‡)</td><td>Power (Imp.)</td><td></td><td></td></tr><tr><td></td><td>601.83 (0.4%)</td><td>741.25 (-0.5%)</td><td>0.0088 (1.5%)</td><td>519.25 (3.7%)</td><td>799.10 (3.3%)</td><td>0.0076 (2.3%)</td><td></td><td></td></tr><tr><td>ctrl int2float</td><td>181 271</td><td></td><td>1,042.25 (2.5%)</td><td>1,113.13 (-4.2%)</td><td>0.0151 (0.7%)</td><td>825.79 (0.5%)</td><td>1,009.60 (3.4%)</td><td>0.0109 (0.9%)</td><td></td><td></td></tr><tr><td>dec</td><td>312</td><td></td><td>1,376.32 (0.0%)</td><td>1,257.90 (0.0%)</td><td>0.0283 (3.1%)</td><td>1,396.34 (-1.5%)</td><td>1,257.90 (0.0%)</td><td>0.0295 (-1.0%)</td><td></td><td></td></tr><tr><td>router</td><td>317</td><td></td><td>1,307.50 (3.6%)</td><td>4,162.74 (-1.9%)</td><td>0.0194 (3.5%)</td><td>1,174.88 (0.2%)</td><td>3,213.78 (0.0%)</td><td>0.0167 (0.0%)</td><td></td><td></td></tr><tr><td>cavlc</td><td>703</td><td></td><td>2,818.95 (1.8%)</td><td>1,510.57 (12.1%)</td><td>0.0439 (6.0%)</td><td>2,324.73 (-0.5%)</td><td>1,516.30 (-1.5%)</td><td>0.0354 (-0.3%)</td><td></td><td></td></tr><tr><td>priority</td><td>1,106</td><td></td><td>4,839.64 (-2.7%)</td><td>13,963.57 (0.3%)</td><td>0.0710 (-2.4%)</td><td>5,637.91 (-7.0%)</td><td>18,074.93 (-1.3%)</td><td>0.0826 (-6.9%)</td><td></td><td></td></tr><tr><td>adder</td><td>1,276</td><td></td><td>3,856.20 (1.7%)</td><td>17,020.18 (0.2%)</td><td>0.0493 (3.5%)</td><td>4,331.65 (-4.9%)</td><td>16,410.26 (10.8%)</td><td>0.0609 (-10.3%)</td><td></td><td></td></tr><tr><td>i2c</td><td>1,489</td><td></td><td>5,873.13 (0.0%)</td><td>2,293.93 (-24.8%)</td><td>0.0855 (-0.9%)</td><td>4,918.47 (0.0%)</td><td>1,541.01 (-9.6%)</td><td>0.0681 (0.0%)</td><td></td><td></td></tr><tr><td>max</td><td>3,377</td><td></td><td>16,429.51 (-18.3%)</td><td>22,742.97 (26.6%)</td><td>0.2610 (-21.4%)</td><td>13,663.10 (-2.4%)</td><td>20,788.38 (26.7%)</td><td>0.2110 (-0.5%)</td><td></td><td></td></tr><tr><td>bar</td><td>3,471</td><td></td><td>13,387.84 (27.0%)</td><td>5,719.02 (29.1%)</td><td>0.2460 (24.5%)</td><td>12,092.85 (20.2%)</td><td>6,473.61 (10.5%)</td><td>0.2000 (19.7%)</td><td></td><td></td></tr><tr><td>sin</td><td>5,440</td><td></td><td>29,171.73 (-5.3%)</td><td>19,504.01 (5.2%)</td><td>0.5550 (-5.9%)</td><td>29,428.22 (1.7%)</td><td>22,205.18 (26.3%)</td><td>0.5770 (0.5%)</td><td></td><td></td></tr><tr><td>arbiter</td><td>12,095</td><td></td><td>45,880.25 (0.3%)</td><td>8,447.16 (-3.4%)</td><td>0.8960 (0.1%)</td><td>40,267.37 (-3.9%)</td><td>6,493.06 (16.6%)</td><td>0.6810 (8.5%)</td><td></td><td></td></tr><tr><td>voter</td><td>14,759</td><td></td><td>95,342.69 (-3.8%)</td><td>7,585.59 (-3.7%)</td><td>1.6900 (-4.3%)</td><td>95,544.13 (-2.1%)</td><td>7,429.42 (6.8%)</td><td>1.7300 (-2.4%)</td><td></td><td></td></tr><tr><td>square</td><td>18,548 24,746</td><td></td><td>85,638.38 (3.3%)</td><td>17,756.67 (0.4%)</td><td>1.4400 (2.0%) 158,259.28 (-21.7%)</td><td>83,546.38 (8.6%)</td></table>

<sup>†</sup> Improvement of LevelSyn-pwmap relative to PigMap-power.  
<sup>‡</sup> Improvement of LevelSyn-pfmap relative to PigMap-performance.

We evaluated the convergence behavior of our proposed LA-GNN (level-async) approach against a standard synchronous (sync) updating scheme (5-layer GCN) over 700 epochs. The training results are shown in Figure 4. Both methods exhibit a rapid reduction in training loss during the initial phase. However, the level-async strategy demonstrates a markedly superior optimization trajectory. While the standard sync method settles at a higher training loss of 0.75, the level-async exhibits a sharper initial descent and converges to a significantly lower loss of 0.61.

We visualize the coordinate regression performance in Figure 5 on the largest benchmark circuit, hyp, which comprises 214,591 nodes. The scatter plots compare the GNN-predicted coordinates (left) and the PigMap GiFt [21, 24] initialization coordinates (right) against the true ground-truth placements. While the GiFt initialization results in a highly chaotic and scattered distribution, our LA-GNN efectively suppresses this noise and concentrate the predicted coordinates into a refined and high-density manifold. This focused prediction suggests that the LA-GNN has successfully learned the intrinsic logical connectivity and global center of gravity of the circuit, rather than falling into the high-variance irregularities seen in GiFt. By providing such a stable and cohesive initial layout, our model ofers a much more reliable and high-quality starting point for downstream placement optimization, which is crucial for achieving better wirelength and timing convergence in ultra-large-scale designs.

![](images/937ea237460b5fd5a8653c17a00b9dac1154ec56e84bca2c966ff85d5c03833a.jpg)

Figure 4: Pre-training loss comparison between LA-GNN and synchronous GNN  
![](images/014289c2f1f365f0ca3d1c1dbfc53659e05baffe5113d506be3f672c028acec9.jpg)

![](images/b1d7f7e67bfaf36c76cbcaad8289034b1bb6e8b6ff6b2885c48c1c5720d9e10f.jpg)  
Figure 5: Placement coordinates scatter comparison on hyp. Left: GNN-predicted coordinates. Right: PigMap GiFt[21, 24] initialization coordinates

Table 2: Post-Routing Results on Bar
<table><tr><td>Strategies</td><td>Wire Length (um)</td><td>Via Counts</td><td>DRC Violations</td></tr><tr><td>PigMap-power</td><td>320,689</td><td>33,963</td><td>966</td></tr><tr><td>PigMap-performance</td><td>301,312</td><td>28,025</td><td>623</td></tr><tr><td>LevelSyn-pwmap</td><td>254,439</td><td>24,330</td><td>4</td></tr><tr><td>LevelSyn-pfmap</td><td>243,084</td><td>22,419</td><td>17</td></tr><tr><td>Imp.†</td><td>20.66%</td><td>28.36%</td><td>99.59%</td></tr><tr><td> $I m p . ^ { \ddag }$ </td><td>19.32%</td><td>20.00%</td><td>97.27%</td></tr></table>

<sup>†</sup> Improvement of LevelSyn-pwmap relative to PigMap-power.  
<sup>‡</sup> Improvement of LevelSyn-pfmap relative to PigMap-performance.

## 4.3 PPA Results

To evaluate the efectiveness of the proposed framework under diferent optimization goals, we implement two variants of Level-Syn: LevelSyn-pwmap and LevelSyn-pfmap. The former employs pwmap to minimize total wirelength and switching power, while the latter utilizes pfmap to focus on critical path delay. Table 1 summarizes the PPA results. LevelSyn consistently outperforms both the logic-only baseline and PigMap across the majority of test cases.

Experimental results demonstrate that LevelSyn consistently outperforms PigMap across diferent optimization goals: specifically,

LevelSyn-pwmap achieves an average 6.52% area reduction and 6.89% switching power improvement compared to PigMap-power, with notable gains in large designs like hyp. These results demonstrate the accuracy of wirelength estimation provided by LA-GNN. Meanwhile, the delay-oriented LevelSyn-pfmap reduces delay by 22.44% by efectively shortening critical paths through Manhattan distance optimization; remarkably, LevelSyn-pwmap yields the best overall delay reduction of 27.48%, suggesting that our global spatial anchors and NLI-based positioning ofer such high-fidelity spatial guidance that they significantly enhance global structural alignment and timing outcomes beyond traditional logic-depth-based heuristics.

To further evaluate the practical eficacy of our approach within a complete physical design flow, we conduct a comprehensive case study on the Bar circuit, and all strategies are subjected to exactly 20 iterations of routing optimization. The choice of Bar as our primary subject is deliberate: it serves as the central case study in SOTA baseline PigMap.

As summarized in Table 2, our LevelSyn framework demonstrates a decisive advantage over the baseline across all critical metrics. Specifically, when comparing our power-optimized variant (LevelSyn-pwmap) to PigMap-power, we achieve a 20.66% reduction in total wire length and a 28.36% decrease in via counts. Most notably, our method achieves a near-total elimination of Design Rule Check (DRC) violations, plummeting from 966 to just 4, which is a staggering 99.59% improvement. Similarly, our performanceoriented variant (LevelSyn-pfmap) outperforms its counterpart with a 19.32% improvement in wire length and a 97.27% reduction in DRC violations. These results confirm that with the same 20-round optimization budget, the structural stability provided by our LA-GNN allows the router to converge on a significantly more robust, and manufacturable layout.

## 4.4 Runtime Analysis

The runtime analysis is presented in Table 3. In the PigMAP and LevelSyn columns, the parenthetical values represent the breakdown of runtime as (Mapping Time + Pre-processing/Training Time). Specifically, for PigMAP, this includes the GiFt[21] initialization time, whereas for LevelSyn, it accounts for the specific model finetuning time. It demonstrates that the LevelSyn approach maintains high eficiency while achieving superior mapping results. Although our methodology includes a comprehensive training strategy, the overall execution time remains highly competitive with established methods like PigMAP.

The empirical data indicates that our specific training approach does not lead to the significant runtime extensions, which is often associated with deep learning applications in EDA. In several cases such as the multiplier and mem\_ctrl circuits, the total runtime of LevelSyn is even comparable to or slightly lower than that of PigMAP. This proves that the fine-tuning phase is remarkably eficient across various circuit scales. Furthermore, while the mapping duration for certain large scale circuits like sqrt and hyp shows a relative increase compared to baseline methods, this latency is negligible when evaluated within the context of the entire electronic design automation flow. In actual industrial practice, the subsequent physical design stages including placement and routing typically require several hours or even days to complete. Consequently, the marginal extra time spent during the logic mapping stage is a minor trade-of for the substantial improvements in power and performance metrics provided by LevelSyn. By producing a more optimized netlist, our approach likely eases the burden on downstream back-end tools and facilitates faster timing closure.

Table 3: Runtime Comparison
<table><tr><td>Circuit Names</td><td>Nodes</td><td>Baseline (s)</td><td>PigMAP-power (s)</td><td>PigMAP-performance (s)</td><td>LevelSyn-pwmap (s)</td><td>LevelSyn-pfmap (s)</td></tr><tr><td>ctrl</td><td>181</td><td>0.25176</td><td>4.26475 (0.26475+4)</td><td>4.26010 (0.26010+4)</td><td>0.34770 (0.29462+0.05308)</td><td>0.30497 (0.25189+0.05308)</td></tr><tr><td>int2float</td><td>271</td><td>0.24799</td><td>3.27121 (0.27121+3)</td><td>3.28982 (0.28982+3)</td><td>0.35021 (0.27180+0.07841)</td><td>0.35164 (0.27323+0.07841)</td></tr><tr><td>dec</td><td>312</td><td>0.26014</td><td>3.28175 (0.28175+3)</td><td>3.28115 (0.28115+3)</td><td>0.3237 (0.31036+0.01334)</td><td>0.33829 (0.32495+0.01334)</td></tr><tr><td>router</td><td>317</td><td>0.25183</td><td>5.27981 (0.27981+5)</td><td>5.28016 (0.28016+5)</td><td>0.43717 (0.28035+0.15682)</td><td>0.43670 (0.27988+0.15682)</td></tr><tr><td>cavlc</td><td>703</td><td>0.28075</td><td>3.32366 (0.32366+3)</td><td>3.32096 (0.32096+3)</td><td>0.38326 (0.30978+0.07348)</td><td>0.40011 (0.32663+0.07348)</td></tr><tr><td>priority</td><td>1106</td><td>0.26983</td><td>5.34651 (0.34651+5)</td><td>5.35716 (0.35716+5)</td><td>1.33215 (0.35595+0.97620)</td><td>1.33603 (0.35983+0.97620)</td></tr><tr><td>adder</td><td>1276</td><td>0.26596</td><td>5.36133 (0.36133+5)</td><td>5.36947 (0.36947+5)</td><td>1.46657 (0.36737+1.09920)</td><td>1.46161 (0.36241+1.09920)</td></tr><tr><td>i2c</td><td>1489</td><td>0.28782</td><td>4.41634 (0.41634+4)</td><td>4.41147 (0.41147+4)</td><td>0.51008 (0.42255+0.08753)</td><td>0.52417 (0.43664+0.08753)</td></tr><tr><td>max</td><td>3377</td><td>0.31464</td><td>6.08305 (2.08305+4)</td><td>5.65394 (1.65394+4)</td><td>2.81013 (1.68296+1.12717)</td><td>2.83736 (1.71019+1.12717)</td></tr><tr><td>bar</td><td>3471</td><td>0.29879</td><td>4.06734 (1.06734+3)</td><td>4.04371 (1.04371+3)</td><td>1.27092 (1.05667+0.21425)</td><td>1.39967 (1.18542+0.21425)</td></tr><tr><td>sin</td><td>5440</td><td>0.53017</td><td>6.58373 (2.58373+4)</td><td>6.66237 (2.66237+4)</td><td>3.43735 (2.61093+0.82642)</td><td>3.60141 (2.77499+0.82642)</td></tr><tr><td>arbiter</td><td>12095</td><td>0.54331</td><td>22.62736 (16.62736+6)</td><td>20.33924 (14.33924+6)</td><td>16.93909 (16.41279+0.52630)</td><td>13.79946 (13.27316+0.52630)</td></tr><tr><td>voter</td><td>14759</td><td>0.80098</td><td>20.50558 (15.50558+5)</td><td>20.87620 (15.87620+5)</td><td>15.82894 (15.43108+0.39786)</td><td>15.88984 (15.49198+0.39786)</td></tr><tr><td>square</td><td>18548</td><td>0.95917</td><td>23.97164 (18.97164+5)</td><td>24.19507 (19.19507+5)</td><td>20.14333 (18.48149+1.66184)</td><td>20.22944 (18.56760+1.66184)</td></tr><tr><td>sqrt</td><td>24746</td><td>1.11424</td><td>32.76447 (26.76447+6)</td><td>33.58574 (27.58574+6)</td><td>58.05863 (27.36181+30.69682)</td><td>57.99451 (27.29769+30.69682)</td></tr><tr><td>multiplier</td><td>27190</td><td>1.31417</td><td>47.66312 (40.66312+7)</td><td>47.80954 (40.80954+7)</td><td>43.68709 (41.51560+2.17149)</td><td>42.88535 (40.71386+2.17149)</td></tr><tr><td>log2</td><td>32092</td><td>1.77527</td><td>60.93639 (52.93639+8)</td><td>61.99767 (53.99767+8)</td><td>57.04942 (54.20919+2.84023)</td><td>57.19582 (54.35559+2.84023)</td></tr><tr><td>mem_ctrl</td><td>48040 57375</td><td>1.33803 2.72399</td><td>167.97727 (158.97727+9) 290.26830 (281.26830+9)</td><td>161.89096 (152.89096+9)</td><td>154.33898 (153.26823+1.07075)</td><td>152.5755 (151.50475+1.07075)</td></tr><tr><td>div</td><td></td><td></td><td></td><td>270.12475 (261.12475+9)</td><td>318.20752 (269.53765+48.66987)</td><td>289.11868 (240.44881+48.66987)</td></tr><tr><td>hyp</td><td>214591</td><td>11.33212</td><td>968.93256 (943.93256+25)</td><td>970.12293 (945.12293+25)</td><td>1,647.60024 (904.50277+743.09747)</td><td>1,726.23648 (983.13901+743.09747)</td></tr></table>

## 4.5 Ablation Study

To evaluate the memory eficiency of our proposed LA-GNN architecture, we conduct an ablation study comparing its peak GPU memory consumption against a 5-layer standard GCN baseline. As summarized in Table 4, the results reveal a significant crossover in memory performance as the circuit scale increases.

Inherently, the LA-GNN architecture requires more memory than a standard GCN due to the additional overhead of maintaining level-specific states and managing asynchronous message-passing bufers. This is evident in several smaller benchmarks where LA-GNN’s allocated memory slightly exceeds that ofthe GCN. However, this architectural overhead is efectively mitigated by our LASP Strategy as we move toward large-scale designs.

The standard GCN sufers from poor scalability because it requires simultaneous storage of the entire graph’s adjacency structure and gradient states, leading to a memory explosion on industrialsized circuits. In contrast, our partitioning strategy decomposes massive AIGs into optimized sub-graphs, allowing the model to process level-dependent information without overloading the GPU’s VRAM. For the largest benchmark, hyp, this synergy allows LA-GNN to use only 4,709 MB of allocated memory compared to the GCN’s 8,956 MB, which is a reduction of approximately 47.4%. These results demonstrate that while LAGNN is more complex by design, the integration of our sub-graph partitioning makes it a far more scalable and practical solution for ultra-large-scale circuit placement, where memory constraints are most critical.

Table 4: Peak GPU Memory Comparison
<table><tr><td rowspan="2">Circuit Name</td><td rowspan="2">Nodes</td><td colspan="2">LA-GNN</td><td colspan="2">GCN</td></tr><tr><td>allocated (MB)</td><td>reserved (MB)</td><td>allocated (MB)</td><td>reserved (MB)</td></tr><tr><td>ctrl</td><td>181</td><td>28.209</td><td>54.000</td><td>25.272</td><td>50.000</td></tr><tr><td>int2float</td><td>271</td><td>35.948</td><td>62.000</td><td>28.490</td><td>54.000</td></tr><tr><td>router</td><td>312</td><td>51.848</td><td>76.000</td><td>31.605</td><td>58.000</td></tr><tr><td>dec</td><td>317</td><td>30.314</td><td>54.000</td><td>30.594</td><td>56.000</td></tr><tr><td>cavlc</td><td>703</td><td>68.501</td><td>88.000</td><td>47.588</td><td>72.000</td></tr><tr><td>priority</td><td>1106</td><td>308.150</td><td>332.000</td><td>68.157</td><td>90.000</td></tr><tr><td>i2c</td><td>1276</td><td>125.399</td><td>132.000</td><td>75.263</td><td>90.000</td></tr><tr><td>adder</td><td>1489</td><td>391.053</td><td>406.000</td><td>82.287</td><td>110.000</td></tr><tr><td>bar</td><td>3377</td><td>89.509</td><td>110.000</td><td>153.047</td><td>178.000</td></tr><tr><td>max</td><td>3471</td><td>291.921</td><td>336.000</td><td>173.920</td><td>220.000</td></tr><tr><td>sin</td><td>5440</td><td>241.344</td><td>250.000</td><td>218.373</td><td>272.000</td></tr><tr><td>arbiter</td><td>12095</td><td>299.112</td><td>604.000</td><td>547.051</td><td>616.000</td></tr><tr><td>voter</td><td>14759</td><td>340.547</td><td>678.000</td><td>650.676</td><td>716.000</td></tr><tr><td>square</td><td>18548</td><td>431.628</td><td>890.000</td><td>815.992</td><td>880.000</td></tr><tr><td>sqrt</td><td>24746</td><td>1315.981</td><td>2080.000</td><td>923.094</td><td>1024.000</td></tr><tr><td>multiplier</td><td>27190</td><td>570.486</td><td>1080.000</td><td>1091.725</td><td>1242.000</td></tr><tr><td>log2</td><td>32092</td><td>621.266</td><td>1428.000</td><td>1200.106</td><td>1384.000</td></tr><tr><td>mem_ctrl</td><td>48040</td><td>992.337</td><td>1792.000</td><td>1911.500</td><td>2156.000</td></tr><tr><td>div</td><td>54375</td><td>1259.288</td><td>2828.000</td><td>2258.522</td><td>2490.000</td></tr><tr><td>hyp</td><td>214591</td><td>4709.544</td><td>9604.000</td><td>8956.849</td><td>10668.000</td></tr></table>

## 5 Conclusion and Future Work

In conclusion, this paper presents PhySyn, a framework that successfully bridges the gap between logic synthesis and physical implementation by embedding deep learning-based spatial predictions into the synthesis flow. Through the use of LA-GNN and level-aligned partitioning, PhySyn achieves high-fidelity spatial estimation and scalability for large-scale designs. Specifically, the near-total elimination of DRC violations underscores the potential of physical-aware synthesis in achieving correct-by-construction design flows. Moving forward, we plan to extend this framework to support sequential circuits by incorporating flip-flop positioning and to explore multi-objective optimization targets such as congestion and thermal distribution.

## Acknowledgments

This work was supported in part by the Hong Kong Research Grants Council (RGC) under Grant No. 14202824, 14205925, C6003-24Y, and T46-415/25-R.

## References

[1] Luca Gaetano Amarù, Pierre-Emmanuel Gaillardon, and Giovanni De Micheli. 2015. The EPFL Combinational Benchmark Suite. https://api.semanticscholar. org/CorpusID:13971503

[2] Robert Brayton and Alan Mishchenko. 2010. ABC: An academic industrial strength verification tool. In Computer Aided Verification. Springer, 24–40.

[3] Chau-Shen Chen, Yu-Wen Tsay, TingTing Hwang, A.C.H. Wu, and Youn-Long Lin. 1995. Combining technology mapping and placement for delay-minimization in FPGA designs. IEEE Transactions on Computer-Aided Design ofIntegrated Circuits and Systems (1995).

[4] Lei Chen, Yiqi Chen, Zhufei Chu, Wenji Fang, Tsung-Yi Ho, Ru Huang, Yu Huang, Sadaf Khan, Min Li, Xingquan Li, et al. 2024. The dawn of ai-native eda: Opportunities and challenges of large circuit models. arXiv preprint arXiv:2403.07257 (2024).

[5] Lei Chen, Yiqi Chen, Zhufei Chu, Wenji Fang, Tsung-Yi Ho, Ru Huang, Yu Huang, Sadaf Khan, Min Li, Xingquan Li, et al. 2024. Large circuit models: opportunities and challenges. Science China Information Sciences 67, 10 (2024), 200402.

[6] Wei-Lin Chiang, Xuanqing Liu, Si Si, Yang Li, Samy Bengio, and Cho-Jui Hsieh. 2019. Cluster-GCN: An Eficient Algorithm for Training Deep and Large Graph Convolutional Networks. In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. 257–266.

[7] Chenhui Deng, Zichao Yue, Cunxi Yu, Gokce Sarar, Ryan Carey, Rajeev Jain, and Zhiru Zhang. 2024. Less is more: Hop-wise graph attention for scalable and generalizable learning on circuits. In Proceedings of the 61st ACM/IEEE Design Automation Conference. 1–6.

[8] Wenji Fang, Wenkai Li, Shang Liu, Yao Lu, Hongce Zhang, and Zhiyao Xie. 2025. NetTAG: A Multimodal RTL-and-Layout-Aligned Netlist Foundation Model via Text-Attributed Graph. arXiv preprint arXiv:2504.09260 (2025).

[9] Wenji Fang, Shang Liu, Jing Wang, and Zhiyao Xie. 2025. CircuitFusion: Multimodal Circuit Representation Learning for Agile Chip Design. In International Conference on Learning Representations (ICLR).

[10] Tong GaO, Kuang-Chien Chen, J. Cong, Yuzheng Ding, and C. Liu. 1993. Placement and placement driven technology mapping for FPGA synthesis. In Sixth Annual IEEE International ASIC Conference and Exhibit.

[11] W. Gosti, S.R. Khatri, and A.L. Sangiovanni-Vincentelli. 2001. Addressing the timing closure problem by integrating logic optimization and placement. In IEEE/ACM International Conference on Computer Aided Design. ICCAD 2001. IEEE/ACM Digest of Technical Papers (Cat. No.01CH37281).

[12] Mingxiao He, Pengcheng Huang, Zhenyu Zhao, and Peiyun Bian. 2025. CPA-Remap: Critical-Path-Based Physically Aware Remapping Framework for Timing Optimization. In 2025 IEEE 43rd International Conference on Computer Design (ICCD). IEEE, 393–400.

[13] R. Ho, K.W. Mai, and M.A. Horowitz. 2001. The future of wires. Proc. IEEE 89, 4 (2001), 490–504.

[14] Guyue Huang, Jingbo Hu, Yifan He, Jialong Liu, Mingyuan Ma, Zhaoyang Shen, Juejian Wu, Yuanfan Xu, Hengrui Zhang, Kai Zhong, Xuefei Ning, Yuzhe Ma, Haoyu Yang, Bei Yu, Huazhong Yang, and Yu Wang. 2021. Machine learning for electronic design automation: A survey. ACM Trans. Des. Autom. Electron. Syst. (2021).

[15] George Karypis and Vipin Kumar. 1998. A Fast and High Quality Multilevel Scheme for Partitioning Irregular Graphs. (Dec. 1998), 359–392.

[16] Cunqing Lan, Hongyang Pan, Zhiang Wang, Xuan Zeng, Fan Yang, and Keren Zhu. 2025. PigMap2: A Physical Information Guided Technology Mapping Framework. IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems (2025).

[17] Min Li, Sadaf Khan, Zhengyuan Shi, Naixing Wang, Huang Yu, and Qiang Xu. 2022. Deepgate: Learning neural representations of logic gates. In Proceedings of

the 59th ACM/IEEE Design Automation Conference. 667–672.

[18] Min Li, Zhengyuan Shi, Qiuxia Lai, Sadaf Khan, Shaowei Cai, and Qiang Xu. 2023. On eda-driven learning for sat solving. In Proceedings ofthe 60th Annual ACM/IEEE Design Automation Conference.

[19] Yibo Lin, Zixuan Jiang, Jiaqi Gu, Wuxi Li, Shounak Dhar, Haoxing Ren, Brucek Khailany, and David Z. Pan. 2021. DREAMPlace: Deep Learning Toolkit-Enabled GPU Acceleration for Modern VLSI Placement. IEEE Transactions on Computer-Aided Design ofIntegrated Circuits and Systems (2021).

[20] Jiawei Liu, Jianwang Zhai, Mingyu Zhao, Zhe Lin, Bei Yu, and Chuan Shi. 2024. PolarGate: Breaking the Functionality Representation Bottleneck of And-Inverter Graph Neural Network. In Proceedings ofthe 43rd IEEE/ACM International Conference on Computer-Aided Design.

[21] Yiting Liu, Hai Zhou, Jia Wang, Fan Yang, Xuan Zeng, and Li Shang. 2025. The Power ofGraph Signal Processing for Chip Placement Acceleration. In Proceedings of the 43rd IEEE/ACM International Conference on Computer-Aided Design (ICCAD ’24). Association for Computing Machinery.

[22] Aiguo Lu, Hans Eisenmann, Guenter Stenz, and Frank M Johannes. 1998. Combining technology mapping with post-placement resynthesis for performance optimization. In Proceedings International Conference on Computer Design. VLSI in Computers and Processors (Cat. No. 98CB36273). IEEE, 616–621.

[23] Alan Mishchenko et al. 2007. ABC: A system for sequential synthesis and verifi cation. URL http://www. eecs. berkeley. edu/alanmi/abc 17 (2007).

[24] Hongyang Pan, Cunqing Lan, Yiting Liu, Zhiang Wang, Li Shang, Xuan Zeng, Fan Yang, and Keren Zhu. 2025. Physically Aware Synthesis Revisited: Guiding Technology Mapping with Primitive Logic Gate Placement. In Proceedings of the 43rd IEEE/ACM International Conference on Computer-Aided Design. Article 171.

[25] Massoud Pedram and Narasimha Bhat. 1991. Layout driven technology mapping. In Proceedings ofthe 28th ACM/IEEE Design Automation Conference. 99–105.

[26] André Inácio Reis and Jody M. A. Matos. [n. d.]. Physical Awareness Starting at Technology-Independent Logic Synthesis.

[27] Amir H. Salek, Jinan Lou, and Massoud Pedram. 1998. A DSM design flow: putting floorplanning, technology-mapping, and gate-placement together. In Proceedings ofthe 35th Annual Design Automation Conference (DAC ’98).

[28] Zhengyuan Shi, Min Li, Sadaf Khan, Liuzheng Wang, Naixing Wang, and Yu Huang. 2022. DeepTPI: Test Point Insertion with Deep Reinforcement Learning. In 2022 IEEE International Test Conference (ITC).

[29] Zhengyuan Shi, Hongyang Pan, Sadaf Khan, Min Li, Yi Liu, Junhua Huang, Hui-Ling Zhen, Mingxuan Yuan, Zhufei Chu, and Qiang Xu. 2023. DeepGate2: Functionality-Aware Circuit Representation Learning. In Proceedings ofthe 2023 IEEE/ACM international conference on Computer-aided design.

[30] SkyWater Technology and Google. 2020. SkyWater Open Source PDK. https: //github.com/google/skywater-pdk Accessed: 2026-04-05.

[31] The OpenROAD Project. 2024. OpenROAD: An open-source industrial design flow from RTL to GDS. https://github.com/The-OpenROAD-Project/OpenROAD.

[32] N. Togawa, M. Sato, and T. Ohtsuki. 1994. Maple: A Simultaneous Technology Mapping, Placement, And Global Routing Algorithm For Field-programmable Gate Arrays. In IEEE/ACM International Conference on Computer-Aided Design. 156–163. doi:10.1109/ICCAD.1994.629759

[33] N. Togawa, M. Yanagisawa, and T. Ohtsuki. 1998. Maple-opt: a performance oriented simultaneous technology mapping, placement, and global routing al gorithm for FPGAs. IEEE Transactions on Computer-Aided Design ofIntegrated Circuits and Systems (1998), 803–818. doi:10.1109/43.720317

[34] Cliford Wolf, Johann Glaser, and Johannes Kepler. 2013. Yosys-a free Verilog synthesis suite. In Proceedings ofthe 21st Austrian Workshop on Microelectronics (Austrochip). 97.

[35] Xinyue Wu, Zixuan Li, Fan Hu, Ting Lin, Xiaotian Zhao, Runxi Wang, and Xinfei Guo. 2025. Shift-Left Techniques in Electronic Design Automation: A Survey. arXiv preprint arXiv:2509.14551 (2025).

[36] Zhengfeng Wu, Pratik Shrestha, Saran Phatharodom, and Ioannis Savidis. 2025. Diferentiable Graph Neural Networks for Wirelength Estimation. In 2025 IEEE International Symposium on Circuits and Systems (ISCAS). IEEE, 1–5.

[37] Zhiyao Xie, Rongjian Liang, Xiaoqing Xu, Jiang Hu, Yixiao Duan, and Yiran Chen. 2021. Net2: A graph attention network method customized for pre-placement net length estimation. In Proceedings ofthe 26th Asia and South Pacific Design Automation Conference. 671–677