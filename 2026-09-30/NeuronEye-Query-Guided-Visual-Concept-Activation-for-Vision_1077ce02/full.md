# NeuronEye: Query-Guided Visual Concept Activation for Vision-Language Reasoning

Ruiyu Yan<sup>1</sup> Bowen Chen<sup>2</sup> Shaowen Wan<sup>2</sup> Lin Zhao<sup>2</sup> <sup>1</sup>New York University <sup>2</sup>New Jersey Institute of Technology lin.zhao.1@njit.edu

## Abstract

Current vision-language models (VLMs) encode visual information in dense hidden states where object identity, spatial layout, and local attributes are implicitly entangled rather than explicitly disentangled, limiting their ability to isolate and modulate the specific visual evidence required by a given language query. Inspired by sparse population coding and top-down modulation in biological vision, we introduce NeuronEye, a plug-in framework that constructs a sparse, concept-level neuron vocabulary from intermediate VLM representations and selectively activates query-relevant visual concepts during inference. NeuronEye decomposes visiontoken states into an overcomplete sparse basis organized by concept-level clusters, uses the language query to activate relevant clusters and localize the patches where selected concepts are expressed, and injects the focused evidence back into vision tokens. A complementary suppression mechanism attenuates dominant perceptual directions to preserve weaker but relevant cues. All operations run in a single forward pass over a frozen VLM backbone. On Qwen2.5-VL-7B, NeuronEye raises CV-Bench overall accuracy by +3.1 with gains of +9.5 on Distance, and improves BLINK Multi-view by +8.3, with similar trends on LLaVA-1.6-7B. These results suggest that sparse neuron vocabularies can serve not only as post-hoc interpretability tools but also as active interfaces for concept-level visual reasoning.

![](images/52d3fd8153839107096ff935957be880d71658857dc1bb6759437cee146eab06.jpg)  
Figure 1: NeuronEye overview. Given an image and a language query, NeuronEye decomposes intermediate VLM representations into a sparse neuron vocabulary and identifies query-relevant concept clusters (e.g., Human and Apparel). Each cluster localizes its corresponding visual evidence in the image, and the focused activations are injected back into the VLM to guide reasoning.

## 1 Introduction

Modern vision-language models (VLMs) align image and text by jointly processing visual and language tokens through learned attention-based multimodal fusion [1, 2, 3]. Although effective on a wide range of tasks, this paradigm encodes visual information in dense hidden states where object identity, spatial layout, local attributes, and background context are not explicitly disentangled. Recent work has shown that VLMs exhibit multi-object reasoning failures remarkably similar to those caused by representational interference in human rapid feedforward vision [4], and that the bottleneck in spatial reasoning often lies in integrating visual information rather than in perceiving it [5, 6]. These findings suggest that dense visual representations may be insufficient for isolating and selectively modulating individual visual concepts according to the question at hand.

The biological visual system addresses this difficulty through two complementary mechanisms. Neurons in primary visual cortex represent natural scenes via sparse population codes, where any stimulus activates only a small subset of neurons while the majority remain silent [7, 8]. At higher cortical levels, this sparsity becomes even more selective: concept cells in the human medial temporal lobe respond to specific persons or landmarks regardless of viewpoint or input modality [9, 10], suggesting that the brain organizes visual information into a sparse, concept-level neural vocabulary whose entries can be independently addressed. Then, top-down attention selectively modulates this vocabulary according to task demands. Treisman’s feature integration theory established that separately processed visual features require focused attention to be correctly bound into object percepts [11, 12], and neurophysiological studies have confirmed that prefrontal top-down signals enhance task-relevant neural responses while suppressing competing representations [13, 14]. Together, these mechanisms implement a two-stage strategy: decompose visual input into a sparse concept-level vocabulary, then selectively modulate task-relevant entries to guide downstream perception and reasoning.

Inspired by this perspective, we introduce NeuronEye, a framework that constructs a sparse, conceptlevel neuron vocabulary from intermediate VLM representations and uses language queries to selectively activate relevant visual concepts during inference (Fig. 1). NeuronEye has three core components. Sparse Neuron Space (SNS) projects vision-token states into an overcomplete sparse basis via a sparse autoencoder (SAE) and groups the resulting neurons into concept-level clusters, forming an addressable visual concept vocabulary. Neuron-guided Visual Focus (NVF) uses the language query, together with image-side evidence, to activate relevant neuron clusters, locate the patches where the selected concepts are expressed, and inject the activated evidence back into the corresponding vision tokens. Perceptual Concept Suppression (PCS) complements NVF by attenuating dominant perceptual directions at a later layer, preventing focused activation from suppressing weaker but relevant cues. All operations run in a single forward pass with the VLM backbone and SAE frozen.

We evaluate NeuronEye on four vision-centric benchmarks, demonstrating that it consistently improves structured visual reasoning. Applied to Qwen2.5-VL-7B, NeuronEye raises CV-Bench overall accuracy from 78.5 to 81.6 (+3.1), with gains of +9.5 on Distance and +2.3 on Relation, and improves BLINK Multi-view by +8.3. The method also transfers to LLaVA-1.6-7B with similar trends on spatial and relational sub-tasks. Compared with vision-token reduction and representation-level intervention baselines under the same controlled evaluation protocol, NeuronEye achieves the best overall performance on both CV-Bench and BLINK without additional finetuning.

Our contributions are summarized as follows:

• We propose NeuronEye, a concept-level selective modulation framework that decomposes dense visual representations into a sparse neuron vocabulary organized by visual concepts, and activates query-relevant neuron clusters to modulate visual evidence for VLM reasoning.

• NeuronEye operates as a plug-in module over a frozen VLM backbone. It requires only a one-time sparse autoencoder training on intermediate vision-token activations, and all inference is completed in a single forward pass.

• We show that sparse autoencoder features, which have been used exclusively as a post-hoc interpretability tool, can serve as a structured vocabulary for concept-level visual reasoning, extending their role from passive interpretation to actively improving model performance.

• We evaluate NeuronEye on four vision-centric benchmarks across two VLM backbones, demonstrating consistent improvements on spatial and localization-sensitive tasks.

## 2 Related Works

## 2.1 Visual Representation Modulation

Prior work has explored visual representation modulation to focus VLM reasoning on relevant visual evidence. Methods such as FastV [15], SparseVLM [16], PruMerge [17], and PyramidDrop [18] remove, merge, or retain patch tokens based on attention or importance scores, determining where the model attends. However, each retained token remains a dense unit carrying all visual concepts together; selecting a token retains all its encoded information, including information irrelevant to the query. Representation-level methods such as VISTA [19] adjust what is represented by steering dense hidden states, but these modifications are applied uniformly and do not select which visual concept to enhance for a given query. NeuronEye operates at a finer granularity. Rather than selecting which patch tokens to keep or uniformly steering dense hidden states, it decomposes vision-token representations into a sparse concept-level neuron vocabulary and uses the language query to activate relevant concept subsets within localized patches for targeted modulation.

## 2.2 Sparse Autoencoders in Multimodal Models

Sparse autoencoders (SAEs) have been widely adopted to decompose superposed activations into sparse, interpretable latent features [20, 21]. In this post-hoc setting, SAE features serve as a lens for understanding model internals, revealing human-interpretable directions that correspond to recognizable concepts in the representation space [22, 23]. More recently, several works have moved beyond interpretation toward active intervention, using SAE features to steer model behavior. By amplifying or suppressing selected latent directions, these methods bias model outputs toward desired content or away from undesired behavior [24, 25]. In the multimodal setting, SAVE [26] and SSL [27] apply this steering paradigm to VLMs, using SAE features to identify hallucination-related directions and globally suppress them during generation. While effective for hallucination mitigation, these methods identify a fixed set of SAE features offline and apply them without conditioning on the specific query, and do not organize features into structured groups for selective routing. They therefore do not target query-specific visual reasoning or fine-grained evidence selection. NeuronEye goes beyond steering by organizing SAE-derived visual features into concept-level neuron clusters, selecting relevant clusters conditioned on the language query, and activating them within localized image patches, turning the sparse feature basis into a structured vocabulary for visual reasoning.

## 3 Method

NeuronEye operates through three stages. Given an image–question pair, Sparse Neuron Space (SNS, Section 3.1) first decomposes intermediate visual representations into an overcomplete sparse basis and organizes the resulting neurons into concept-level clusters, forming an addressable visual concept vocabulary. Neuron-guided Visual Focus (NVF, Section 3.2) then uses the language query and image-side evidence to activate relevant neuron clusters, localize the patches where selected concepts are expressed, and inject the activated evidence back into the corresponding vision tokens at layer ℓ. Finally, Perceptual Concept Suppression (PCS, Section 3.3) attenuates dominant perceptual directions at a later layer $\ell ^ { \prime }$ to preserve representational diversity after focused activation.

## 3.1 Sparse Neuron Space (SNS)

Sparse Neuron Extraction. Given an image–question pair, we pass the visual input and textual query through the frozen VLM and extract the hidden representation of each image patch token at a designated intermediate layer ℓ, denoted as $\mathbf { x } _ { i , p } \in \mathbb { R } ^ { d }$ , where i indexes the image and p indexes the patch token. We use a SAE to decompose each patch representation into a set of neuron activations (Fig. 2): the SAE projects $\mathbf { x } _ { i , p }$ into an overcomplete latent space $\mathbb { R } ^ { D }$ with $D = \alpha d \left( \alpha \gg 1 \right)$ using a linear encoder followed by ReLU activation, where each latent dimension corresponds to a neuron. A deterministic Top-k operator then retains only the k largest activations and zeros out the rest, producing a sparse code $\mathbf { z } _ { i , p }$ with fixed $\ell _ { 0 } = k$ sparsity:

$$
\begin{array} { r } { \mathbf { z } _ { i , p } = \mathrm { T o p } _ { k } \big ( \mathrm { R e L U } ( \mathbf { W } _ { \mathrm { e n c } } \mathbf { x } _ { i , p } ) \big ) . } \end{array}\tag{1}
$$

The representation is thus sparse because only k out of D neurons are active for any given patch, and each active neuron carries a scalar activation indicating how strongly that visual concept is

![](images/92a44d2e9061265148ec104808702e20fcb648d070cbdf89f52c82f6339a9944.jpg)  
Figure 2: Construction of Sparse Neuron Space. Intermediate vision-token states are encoded by a sparse autoencoder, whose activated neurons are visualized through top-activating patches and grouped into concept-level neuron clusters.

expressed at that patch location. The patch representation is reconstructed as $\begin{array} { r } { \hat { \mathbf { x } } _ { i , p } = \mathbf { W } _ { \mathrm { d e c } } \mathbf { z } _ { i , p } , } \end{array}$ where decoder columns are $\ell _ { 2 } \cdot$ -normalized to prevent scale degeneracy. Training minimizes a patchlevel reconstruction objective augmented by an $\ell _ { 1 }$ activation penalty:

$$
\mathcal { L } _ { \mathrm { S A E } } = \frac { 1 } { \vert \mathcal { B } \vert } \sum _ { ( i , p ) \in \mathcal { B } } \Vert \mathbf { x } _ { i , p } - \hat { \mathbf { x } } _ { i , p } \Vert _ { 2 } ^ { 2 } + \lambda _ { s } \Vert \mathbf { z } _ { i , p } \Vert _ { 1 } ,\tag{2}
$$

where B denotes a minibatch of image patch tokens. Since the Top-k operator fixes the number of active coordinates, the $\ell _ { 1 }$ term mainly regularizes the magnitude of selected activations.

Neuron Filtering and Neuron Cluster Construction. Not all neurons in the overcomplete latent space are useful. Many are rarely activated across images or respond to visually inconsistent patterns. We therefore filter unreliable neurons and group the remaining ones into neuron clusters. We run the frozen VLM with the trained SAE over the training set and collect patch-level sparse activations $z _ { i , p , j } ,$ where j indexes the neuron. Neurons activated on fewer than $\bar { M } _ { \mathrm { m i n } }$ distinct images are discarded. For each retained neuron $j ,$ we aggregate its activations across patches within each image, select the top-N images where neuron $j$ is most strongly activated, and localize the highest-activating patches by their spatial positions. The resulting patch regions $S _ { j }$ summarize the visual evidence associated with neuron $j .$ . We assign each neuron a short concept label by prompting an external model to summarize the shared visual pattern in $S _ { j }$ (see Appendix $\mathbf { A } ) ,$ , then encode labels into text embeddings and apply hierarchical clustering to group neurons into K concept-level clusters $\{ \mathcal { C } _ { k } \} _ { k = 1 } ^ { K } .$ These labels are used only for cluster construction; after clustering, each cluster is represented only by its index and the corresponding set of neurons, forming the sparse neuron vocabulary used by NVF.

## 3.2 Neuron-guided Visual Focus

Given the sparse neuron vocabulary constructed by SNS, NVF uses the language query to activate relevant visual concepts for the current image–question pair. It first predicts which neuron clusters are query-relevant, then localizes the patches where the selected concepts are most strongly expressed, and finally injects the activated evidence back into the corresponding vision tokens.

Query-Guided Neuron Cluster Activation. Each neuron cluster is represented by an index $\bar { k } \in \{ 1 , \ldots , K \}$ and corresponds to a set of neurons. We obtain a query representation by pooling the textual hidden states and use a lightweight router $g _ { q } ( \cdot )$ to produce query-side cluster scores $\pmb { \alpha } ^ { q }$ (Fig. 3A). To reduce reliance on language-only priors, a vision-side scorer $g _ { v } ( \cdot )$ produces imagegrounded cluster scores $\pmb { \alpha } ^ { v }$ from visual hidden states:

$$
\alpha ^ { q } = \sigma ( g _ { q } ( \mathrm { P o o l } ( \mathbf { H _ { t e x t } } ) ) ) \in [ 0 , 1 ] ^ { K } , \qquad \alpha ^ { v } = \sigma ( g _ { v } ( \mathrm { P o o l } ( \mathbf { H _ { v i s i o n } } ) ) ) \in [ 0 , 1 ] ^ { K } .\tag{3}
$$

At inference time, NVF selects the top- $K _ { \mathrm { s e l } }$ clusters according to $\pmb { \alpha } ^ { q }$ . During training, we obtain supervision labels $\mathbf { y } \in \{ 0 , 1 \} ^ { K }$ by prompting an LLM annotator to identify which neuron clusters are relevant to each text query, based on the cluster descriptions constructed in Section 3.1. The router $g _ { q }$ is then supervised with binary cross-entropy over y, and a per-dimension symmetric divergence term aligns $\pmb { \alpha } ^ { q }$ with $\alpha ^ { v }$ to ensure that the selected clusters are grounded in image evidence:

![](images/1eca150bfebb54c3d127faf0cfe78494f8133722c2243fe2c2063cd007d6a297.jpg)  
Figure 3: NVF and PCS. NVF selects query-relevant neuron cluster indices and active patches for local cross-attention, while PCS suppresses dominant perceptual components at a later layer.

$$
\mathcal { L } _ { \mathrm { a l i g n } } = \frac { 1 } { 2 } \sum _ { k = 1 } ^ { K } \left[ \alpha _ { k } ^ { v } \log \frac { \alpha _ { k } ^ { v } } { \alpha _ { k } ^ { q } } + \alpha _ { k } ^ { q } \log \frac { \alpha _ { k } ^ { q } } { \alpha _ { k } ^ { v } } \right] .\tag{4}
$$

The final routing objective is $\mathcal { L } _ { \mathrm { r o u t e r } } = \mathcal { L } _ { \mathrm { B C E } } ( \alpha ^ { q } , \mathbf { y } ) + \lambda _ { \mathrm { a l i g n } } \mathcal { L } _ { \mathrm { a l i g n } } $ , which encourages selected cluster indices to be both query-relevant and visually supported.

Local Reasoning with Focused Visual Evidence. For each selected cluster $c ,$ let $\mathcal { F } _ { c }$ denote its set of neuron indices. NVF measures how strongly cluster c is expressed at each patch token by summing the activations of its neurons: $\begin{array} { r } { a _ { p } ^ { ( c ) } = \sum _ { j \in \mathcal { F } _ { c } } z _ { p , j } } \end{array}$ , where $z _ { p , j }$ is the activation of neuron j at patch token $p .$ The top-n patches with the highest $a _ { p } ^ { ( c ) }$ are selected as active visual evidence for cluster c (Fig. 3B). On these patches, we construct a cluster-specific sparse code $\mathbf { z } _ { \mathrm { c l u s t e r } }$ by retaining only the neurons in $\mathcal { F } _ { c }$ and masking all others. A query-conditioned refinement module then predicts a gated residual update:

$$
\Delta \mathbf { z } = \beta _ { \mathrm { r e f } } \cdot \mathbf { W } _ { \Delta } \left( \sigma ( \mathbf { W } _ { g } \mathbf { z } _ { \mathrm { c l u s t e r } } ) \odot \mathrm { M L P } ( \mathbf { q } _ { \mathrm { l a s t } } ) \right) ,\tag{5}
$$

where $\mathbf { q } _ { \mathrm { l a s t } }$ is the hidden state of the last textual token at layer $\ell ,$ and $\beta _ { \mathrm { r e f } }$ is a learnable scalar controlling the refinement magnitude. The refined sparse code is $\widetilde { { \bf z } } _ { \mathrm { c l u s t e r } } = { \bf z } _ { \mathrm { c l u s t e r } } + \Delta { \bf z }$ . Each refined code is decoded back to the dense space via the frozen SAE decoder, weighted by its query-side score $\alpha _ { c } ^ { q } .$ , and summed across selected clusters into $\mathbf { R } _ { \mathrm { a c t i v e } }$

Finally, NVF injects $\mathbf { R } _ { \mathrm { a c t i v e } }$ into the selected active tokens through localized cross-attention:

$$
\mathbf { H } _ { \mathrm { a c t i v e } } ^ { \prime } = \mathbf { H } _ { \mathrm { a c t i v e } } + \gamma _ { \mathrm { i n j } } \cdot \mathrm { L N } \left( \mathbf { W } _ { o } \mathrm { s o f t m a x } \left( { \frac { \mathbf { Q } _ { \mathrm { a t t } } \mathbf { K } _ { \mathrm { a t t } } ^ { \top } } { \sqrt { d } } } \right) \mathbf { V } _ { \mathrm { a t t } } \right) ,\tag{6}
$$

where $\mathbf { Q } _ { \mathrm { a t t } } = \mathbf { W } _ { q } \mathbf { H } _ { \mathrm { a c t i v e } } , \mathbf { K } _ { \mathrm { a t t } } = \mathbf { W } _ { k } \mathbf { R } _ { \mathrm { a c t i v e } } , \mathbf { V } _ { \mathrm { a t t } } = \mathbf { W } _ { v } \mathbf { R } _ { \mathrm { a c t i v e } } .$ , and $\gamma _ { \mathrm { i n j } }$ is a learnable injection scale. Only selected tokens are updated; all other vision tokens remain unchanged.

## 3.3 Perceptual Concept Suppression

PCS addresses a side effect of focused activation. While NVF enhances query-relevant concepts, it may also concentrate vision-token representations along dominant directions, weakening cues such as spatial relations or fine-grained attributes. PCS mitigates this effect at a later layer $\ell ^ { \prime } > \ell$ by estimating and attenuating these directions from vision-token geometry (Fig. 3C).

Let $\mathbf { H } \in \mathbb { R } ^ { N \times d }$ denote the vision-token hidden states at layer $\ell ^ { \prime }$ . We mean-center the tokens as $\bar { \mathbf { H } } = \mathbf { H } - \mathbf { 1 } \mu ^ { \top }$ , where $\begin{array} { r } { \pmb { \mu } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathbf { H } _ { i } \in \mathbb { R } ^ { d } } \end{array}$ , to isolate directional structure from the global mean. We then compute a low-rank decomposition $\bar { \mathbf { H } } \approx \mathbf { U } \pmb { \Sigma } \mathbf { V } ^ { \top }$ and take the top-r right singular vectors $\mathbf { V } _ { r } \in \mathbb { R } ^ { d \times r }$ as the dominant perceptual directions. The projected dominant component is $\mathbf { P } = \bar { \mathbf { H } } \mathbf { V } _ { r } \mathbf { V } _ { r } ^ { \top }$ , and PCS suppresses it by:

$$
\mathbf { H } ^ { \prime } = \mathbf { H } - \eta _ { \mathrm { p c s } } \mathbf { P } ,\tag{7}
$$

where $\eta _ { \mathrm { p c s } } = \mathrm { s o f t p l u s } ( \eta _ { \mathrm { p a r a m } } )$ is a learnable non-negative scalar initialized near zero, allowing the model to learn the appropriate suppression strength during training. PCS complements NVF: NVF activates query-relevant concepts at layer $\ell ,$ while PCS preserves representational diversity at layer $\ell ^ { \prime }$ by attenuating dominant directions that may suppress weaker visual cues.

## 4 Experiments

## 4.1 Dataset

Training data. All training data are drawn exclusively from VQAv2 [28] and are disjoint from the evaluation benchmarks. The SAE for SNS is trained on the full VQAv2 training split. For lightweight NVF modules and the PCS scalar, we construct three independent supervision sets, each with 5K image–question pairs sampled without replacement from VQAv2 training split to together with cluster-index labels. These labels are obtained by prompting an LLM annotator to identify query-relevant neuron clusters (Appendix A). For each backbone, we report the mean and standard deviation across the three resulting NeuronEye models.

Evaluation benchmarks. We evaluate on four vision-centric reasoning benchmarks covering spatial understanding, counting, relational perception, localization, and real-world visual reasoning. CV-Bench [5] serves as the primary benchmark, comprising Count, Depth, Distance, and Relation sub-tasks that directly assess structured visual reasoning. BLINK [6] evaluates multi-view reasoning and spatial localization. RealWorldQA [29] and MMStar [30] provide complementary coverage of general visual question answering. No evaluation data are used during NeuronEye training.

## 4.2 Implementation Details

We use Qwen2.5-VL-7B and LLaVA-1.6-7B as the base models, which are frozen throughout. NeuronEye constructs SNS at layer ℓ = 8 with K = 64 neuron clusters, applies NVF at the same layer, and applies PCS at a later layer $\ell ^ { \prime } = 2 0$ . After SNS construction, neuron-cluster assignments are kept fixed, and only the lightweight NVF and PCS scalar modules are updated. The NVF training procedure and hyperparameters are provided (Appendix B). We use greedy decoding for all generation-based evaluations. Experiments are conducted on NVIDIA RTX PRO 6000 GPUs.

## 4.3 Benchmark Results across VLM Backbones

Table 1 reports results across CV-Bench, BLINK, RealWorldQA, and MMStar. The upper block lists representative VLM baselines for reference; the lower block applies NeuronEye to Qwen2.5-VL-7B and LLaVA-1.6-7B to assess backbone portability. On Qwen2.5-VL-7B, NeuronEye improves CV-Bench overall by +3.1 and BLINK overall by +0.7, with the strongest gains on Distance (+9.5), Multi-view (+8.3), and Localization (+3.3). On LLaVA-1.6-7B, improvements follow a similar pattern on spatial and relational sub-tasks, including Distance (+6.9), Relation (+12.1), Multi-view (+7.7), and Localization (+8.1), though performance decreases on Count (-8.3) and Depth (-3.3). This mixed pattern likely reflects backbone architectural differences in visual tokenization and patch granularity, which affect how patch-level sparse activations map to localized visual evidence. Across both backbones, the gains are consistently strongest on spatial, relational, and localization-sensitive tasks, aligning with the design of NeuronEye as a concept-level selective activation mechanism.

## 4.4 Comparison with Visual Representation Modulation Methods

Table 2 compares NeuronEye with vision-token reduction and representation-level methods on Qwen2.5-VL-7B under the same evaluation protocol. NeuronEye achieves the best overall performance on both CV-Bench and BLINK, with especially large gains on CV-Bench Distance (+9.5) and BLINK Multi-view (+8.3). Token reduction methods show consistent degradation on structuresensitive sub-tasks, likely because discarding or merging patches eliminates visual evidence that cannot be recovered downstream. Representation-level methods maintain near-baseline performance but offer limited improvement, suggesting that uniform dense-state shifts lack the specificity needed for spatially demanding queries. NeuronEye’s gains are concentrated precisely on these structuresensitive tasks, supporting the hypothesis that concept-level selective activation provides finer control over which visual evidence is enhanced for a given query.

Table 1: Benchmark results across VLM backbones. The upper block lists representative VLMs for reference. The lower block shows NeuronEye applied to LLaVA-1.6-7B and Qwen2.5-VL-7B, with ∆ denoting the improvement over each base model.
<table><tr><td></td><td colspan="5">CV-Bench</td><td colspan="3">BLINK</td><td colspan="2">Other Benchmarks</td></tr><tr><td>Model</td><td>Overall</td><td>Count</td><td>Depth</td><td>Dist.</td><td>Rel.</td><td>Overall</td><td>MV.</td><td>Loc.</td><td>RealWorldQA</td><td>MMStar</td></tr><tr><td colspan="9">Representative VLM backbones</td><td></td></tr><tr><td>DeepSeek-VL1 [31]</td><td>61.6</td><td>59.0</td><td>63.2</td><td>58.2</td><td>68.5</td><td>38.1</td><td>50.4</td><td>37.7</td><td>50.5</td><td>38.9</td></tr><tr><td>Idefics3-8B-Llama3 [32]</td><td>67.7</td><td>60.5</td><td>72.8</td><td>67.2</td><td>73.4</td><td>42.7</td><td>45.9</td><td>50.8</td><td>62.0</td><td>49.3</td></tr><tr><td>Phi-4 Multimodal [33]</td><td>71.7</td><td>68.5</td><td>74.2</td><td>70.8</td><td>75.1</td><td>49.7</td><td>48.1</td><td>56.6</td><td>61.8</td><td>59.7</td></tr><tr><td>InternVL3-8B [34]</td><td>81.3</td><td>70.9</td><td>84.8</td><td>83.1</td><td>89.7</td><td>51.3</td><td>51.1</td><td>58.2</td><td>68.2</td><td>59.4</td></tr><tr><td>Llava-OneVision [3]</td><td>76.0</td><td>67.3</td><td>80.3</td><td>78.5</td><td>80.8</td><td>46.2</td><td>57.1</td><td>54.9</td><td>66.7</td><td>68.3</td></tr><tr><td colspan="9">NeuronEye applied to different backbones</td><td></td></tr><tr><td>LLaVA-1.6-7B [35]</td><td>64.3</td><td>63.3</td><td>77.8</td><td>54.5</td><td>63.3</td><td>36.6</td><td>40.7</td><td>38.0</td><td>61.6</td><td>36.9</td></tr><tr><td>+NeuronEye</td><td>65.7 ±0.5</td><td>55.0 ±0.2</td><td>74.5 ±1.0</td><td>61.4 ±0.9</td><td>75.4 ±0.5</td><td>37.2 ±0.7</td><td>48.4 ±1.9</td><td>46.1 ±0.6</td><td>60.7 ±0.7</td><td>36.9 ±0.1</td></tr><tr><td>∆</td><td>+1.4</td><td>-8.3</td><td>-3.3</td><td>+6.9</td><td>+12.1</td><td>+0.6</td><td>+7.7</td><td>+8.1</td><td>-0.9</td><td>+0.0</td></tr><tr><td>Qwen2.5-VL-7B [2]</td><td>78.5</td><td>67.1</td><td>87.2</td><td>76.2</td><td>87.2</td><td>52.1</td><td>55.6</td><td>53.3</td><td>68.5</td><td>58.8</td></tr><tr><td>+NeuronEye</td><td>81.6 ±0.3</td><td>67.9 ±0.6</td><td>87.0 ±0.4</td><td>85.7 ±0.8</td><td>89.5 ±0.5</td><td>52.8 ±0.2</td><td></td><td>63.9 ±0.4 56.6 ±0.9</td><td>68.8 ±0.4</td><td>59.3 ±0.5</td></tr><tr><td>∆</td><td>+3.1</td><td>+0.8</td><td>-0.2</td><td>+9.5</td><td>+2.3</td><td>+0.7</td><td>+8.3</td><td>+3.3</td><td>+0.3</td><td>+0.5</td></tr></table>

Table 2: Comparison with visual representation modulation methods on Qwen2.5-VL-7B. All methods are evaluated under the same protocol.
<table><tr><td rowspan="2">Model</td><td colspan="5">CV-Bench</td><td colspan="3">BLINK</td><td colspan="2">Other Benchmarks</td></tr><tr><td>Overall</td><td>Count</td><td>Depth</td><td>Dist.</td><td>Rel.</td><td>Overall</td><td>MV.</td><td>Loc.</td><td>RealWorldQA</td><td>MMStar</td></tr><tr><td colspan="9">Vision-token reduction methods</td><td></td></tr><tr><td>FastV [15]</td><td>75.9</td><td>64.0</td><td>83.5</td><td>74.5</td><td>85.5</td><td>48.4</td><td>55.6</td><td>49.2</td><td>68.8</td><td>55.1</td></tr><tr><td>SparseVLM [16]</td><td>69.8</td><td>54.3</td><td>74.7</td><td>68.8</td><td>86.3</td><td>46.9</td><td>55.6</td><td>59.0</td><td>50.9</td><td>53.8</td></tr><tr><td>PruMerge [17]</td><td>71.6</td><td>55.2</td><td>78.0</td><td>74.0</td><td>83.2</td><td>46.8</td><td>54.9</td><td>51.6</td><td>64.3</td><td>50.6</td></tr><tr><td>MustDrop [36]</td><td>73.7</td><td>58.6</td><td>81.5</td><td>74.8</td><td>83.9</td><td>47.2</td><td>55.6</td><td>52.5</td><td>65.1</td><td>53.0</td></tr><tr><td>PyramidDrop [18]</td><td>72.5</td><td>60.2</td><td>78.7</td><td>70.8</td><td>84.2</td><td>49.1</td><td>55.6</td><td>52.5</td><td>62.2</td><td>50.3</td></tr><tr><td colspan="9">Representation-level intervention methods</td><td></td><td></td></tr><tr><td>SSL [27]</td><td>77.6</td><td>66.2</td><td>84.5</td><td>76.0</td><td>87.9</td><td>51.3</td><td>55.6</td><td>52.5</td><td>69.5</td><td>58.1</td></tr><tr><td>SAVE [26]</td><td>77.9</td><td>66.2</td><td>85.7</td><td>76.5</td><td>87.2</td><td>52.3</td><td>55.6</td><td>54.9</td><td>69.0</td><td>59.2</td></tr><tr><td>VISTA [19]</td><td>78.0</td><td>66.0</td><td>85.8</td><td>76.2</td><td>88.0</td><td>51.7</td><td>55.6</td><td>53.3</td><td>69.4</td><td>59.6</td></tr><tr><td>NeuronEye</td><td>81.6 ±0.3 67.9 ±0.6 87.0 ±0.4 85.7 ±0.8 89.5 ±0.5 52.8 ±0.2 63.9 ±0.4 56.6 ±0.9</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>68.8 ±0.4</td><td>59.3 ±0.5</td></tr></table>

## 4.5 Visualization of Sparse Neurons and Visual Focus

We visualize sparse neuron selectivity and query-guided visual focus to examine how NeuronEye localizes concept-level evidence.

![](images/d9f4171e51775a5954c7e7632a3390e708919aa14f34792c1a7157a97173e371.jpg)  
(a) Human-face feature

![](images/6cdd1447814f57a86db0d37bade857ec2672fccd3fca503e6a3494de2eee029e.jpg)  
(b) Tower feature

![](images/46c881da939aad0c1631e0b396b993b06c1fd60e857d91daf06694f574b46f97.jpg)  
(c) Flag feature  
Figure 4: Neuron-level feature visualization. Each subfigure shows top-activating patches for one SAE neuron, with non-activating regions masked.

Neuron-level features. Fig. 4 shows that individual SNS neurons activate consistently on semantically coherent regions, such as faces, towers, and flags. These examples support using SNS neurons as a sparse visual vocabulary for concept-level organization. Additional neuron visualizations are provided in the appendix (Appendix C).

![](images/5144ef4e4f366738aa4eef5ae771d9cfa584311ee144356035f6ab9530f3dcee.jpg)  
Input Image

![](images/7fe3da05a1f860072857fa64910d517340d346f11f632fb52f17af17b0b3ad2b.jpg)  
Overall Focus

![](images/3c8489ac09bb0adf781cf305af5d8464281bb04149cbf84a0a044f6497515031.jpg)  
Neuron Cluster 44

![](images/79e5a83692fe439fdc7cd5d7d7792b0c51fda1dac3a9fe060cdde9c92599e0a4.jpg)  
Neuron Cluster 27

![](images/656d50fcb665529ff0b7820293c6374e85e0be83bf116549149011b906dca062.jpg)  
Neuron Cluster 9

(a) Is there organic food in this store?  
![](images/4fd6ccbe43c5473f3661b584de16319b0a3a6f944a0675b925cfa702bcba6654.jpg)  
(b) Are all the players wearing black shirts?

![](images/7fedcfe141dbc3eb99ac347c4ce6b3b0c2b16eb570d68b65680223a000bf8226.jpg)  
(c) Is the small elephant touching the big elephant?  
Figure 5: Question-guided visual focus examples. Each shows the input image, overall focus map, and individual neuron cluster activations.

Fig. 5 visualizes the spatial focus produced by NVF. Given a question, the router selects a small set of neuron clusters and actively triggers the corresponding neurons within these clusters. Their patch-level activations highlight localized image regions expressing the selected sparse features. For the query “Is there organic food in this store?”, NVF focuses on shelves and packaged items; for “Are all the players wearing black shirts?”, it shifts toward clothing and human-body regions; and for “Is the small elephant touching the big elephant?”, it emphasizes the two elephants and their contact region. The resulting focus maps are sparse and localized, indicating that NVF operates through query-selected neuron clusters and patch-level activations rather than global image reweighting. Additional focus examples are provided in the appendix (Appendix D).

## 4.6 Ablation Studies

Table 3: Ablation study on component contributions.
<table><tr><td></td><td colspan="5">CV-Bench</td><td colspan="3">BLINK</td></tr><tr><td>Setting</td><td>Overall</td><td>Count</td><td>Depth</td><td>Dist.</td><td>Rel.</td><td>Overall</td><td>MV.</td><td>Loc.</td></tr><tr><td>Baseline (Qwen2.5-VL-7B)</td><td>78.5</td><td>67.1</td><td>87.2</td><td>76.2</td><td>87.2</td><td>52.1</td><td>55.6</td><td>53.3</td></tr><tr><td>Baseline + FT</td><td></td><td></td><td></td><td>76.5 ±0.3 66.6 ±0.2 86.7 ±0.3 69.8 ±0.9 86.5±0.6 51.1 ±0.1 55.6 ±1.9 55.7 ±0.4</td><td></td><td></td><td></td><td></td></tr><tr><td>+ Dense Cross-Attn + PCS</td><td></td><td></td><td></td><td></td><td></td><td></td><td>79.4 ±0.3 66.2 ±0.6 87.0 ±0.4 79.5 ±1.2 88.8±0.3 51.3 ±0.3 55.6 ±1.9 54.9 ±1.8</td><td></td></tr><tr><td>+ Random SNS + NVF</td><td></td><td></td><td></td><td></td><td></td><td></td><td>78.6 ±0.4 67.6 ±0.5 86.5 ±0.6 76.6 ±1.0 86.4±0.6 51.6 ±0.3 55.6 ±1.5 54.9 ±1.0</td><td></td></tr><tr><td>+ PCS only</td><td></td><td></td><td></td><td></td><td></td><td></td><td>78.6 ±0.4 67.3 ±0.4 86.8 ±0.3 76.8 ±1.4 87.2 ±0.6 51.3 ±0.2 55.6 ±1.9 54.9 ±0.4</td><td></td></tr><tr><td>+ SNS + NVF</td><td></td><td></td><td></td><td></td><td></td><td></td><td>80.9 ±0.1 66.8 ±0.6 87.2 ±0.5 84.0 ±0.4 89.5 ±0.5 53.0 ±0.3 55.6 ±2.2 55.7 ±3.0</td><td></td></tr><tr><td>+ NeuronEye</td><td></td><td></td><td></td><td></td><td></td><td></td><td>81.6 ±0.3 67.9 ±0.6 87.0 ±0.4 85.7 ±0.8 89.5 ±0.5 52.8 ±0.2 63.9 ±0.4 56.6 ±0.9</td><td></td></tr></table>

Table 3 summarizes the contribution of each component. Direct fine-tuning of the lightweight modules without sparse intervention (Baseline + FT) does not improve performance and substantially reduces CV-Bench Distance, suggesting that limited VQA fine-tuning alone is insufficient to produce the observed spatial reasoning gains.

Replacing semantically organized neuron clusters with random assignments (Random SNS + NVF) yields near-baseline performance, indicating that the gains do not come merely from added parameters or reconstructed-feature injection. Rather, structured clusters are needed to route sparse visual evidence meaningfully. PCS alone produces only marginal changes, suggesting that attenuating dominant directions is insufficient without query-guided feature selection. SNS + NVF improves CV Bench Overall and Distance, while the full NeuronEye configuration further improves Distance and BLINK Multi-view. Together, these results support the complementary design of NeuronEye: NVF activates localized query-relevant sparse features, while PCS later attenuates dominant directions to preserve representational diversity and weaker cues.

Table 4 demonstrates the effect of the NVF insertion layer. Performance peaks at layer 8, especially on CV-Bench Distance and BLINK Multi-view. Earlier insertion at layer 4 remains competitive but is slightly weaker, suggesting that the representation may not yet provide sufficient semantic abstraction for reliable cluster routing. Deeper layers from 12 to 20 degrade toward the baseline, likely because spatial details have become increasingly compressed.

Table 4: Effect of insertion layer. Layer 8 achieves the best results, suggesting a balance between low-level perceptual features and high-level semantic abstraction.
<table><tr><td></td><td colspan="5">CV-Bench</td><td colspan="3">BLINK</td></tr><tr><td>Layer l</td><td>Overall</td><td>Count</td><td>Depth</td><td>Dist.</td><td>Rel.</td><td>Overall</td><td>MV.</td><td>Loc.</td></tr><tr><td>4</td><td> $8 0 . 5 \pm 0 . 1$ </td><td> $6 7 . 3 \pm 0 . 2$ </td><td> $8 7 . 0 \pm 0 . 6 $ </td><td> $8 3 . 0 \pm 0 . 4$ </td><td> $8 8 . 5 \pm 0 . 3$ </td><td> $5 1 . 6 \pm 0 . 3$ </td><td> $5 7 . 2 \pm 1 . 2$ </td><td> $5 6 . 6 \pm 1 . 1$ </td></tr><tr><td>8</td><td> $8 1 . 6 \pm 0 . 3$ </td><td> $6 7 . 9 \pm 0 . 6$ </td><td> $8 7 . 0 \pm 0 . 4$ </td><td> $8 5 . 7 \pm 0 . 8$ </td><td> $8 9 . 5 \pm 0 . 5$ </td><td> $5 2 . 8 \pm 0 . 2$ </td><td> $6 3 . 9 \pm 0 . 4$ </td><td> $5 6 . 6 \pm 0 . 9$ </td></tr><tr><td>12</td><td> $7 8 . 3 \pm 0 . 4$ </td><td> $6 5 . 2 \pm 0 . 8$ </td><td> $8 5 . 7 \pm 0 . 8$ </td><td> $7 9 . 5 \pm 1 . 2$ </td><td> $8 6 . 3 \pm 0 . 3$ </td><td> $5 1 . 3 \pm 0 . 2 $ </td><td> $5 5 . 6 \pm 0 . 1$ </td><td> $5 4 . 1 \pm 0 . 1$ </td></tr><tr><td>16</td><td> $7 9 . 0 \pm 0 . 3$ </td><td> $6 7 . 7 \pm 0 . 6 $ </td><td> $8 7 . 2 \pm 0 . 2 $ </td><td> $7 6 . 9 \pm 0 . 5$ </td><td> $8 7 . 7 \pm 0 . 6 $ </td><td> $5 1 . 8 { \pm } 0 . 5 $ </td><td> $5 6 . 9 { \pm } 1 . 9$ </td><td> $5 6 . 3 { \pm } 0 . 5 $ </td></tr><tr><td>20</td><td> $7 8 . 6 \pm 0 . 4$ </td><td> $6 6 . 8 \pm 0 . 3$ </td><td> $8 6 . 8 \pm 0 . 4$ </td><td> $7 7 . 0 \pm 1 . 1$ </td><td> $8 6 . 9 \pm 0 . 6 $ </td><td> $5 1 . 7 \pm 0 . 1$ </td><td> $5 5 . 6 \pm 1 . 9$ </td><td> $5 5 . 7 \pm 1 . 3$ </td></tr></table>

Table 5: Effect of the number of neuron clusters K. Too few clusters conflate functionally distinct neurons, while too many fragment coherent functional groups and complicate routing.
<table><tr><td></td><td colspan="5">CV-Bench</td><td colspan="3">BLINK</td></tr><tr><td>Clusters</td><td>Overall</td><td>Count</td><td>Depth</td><td>Dist.</td><td>Rel.</td><td>Overall</td><td>MV.</td><td>Loc.</td></tr><tr><td>32</td><td> $8 0 . 4 \pm 0 . 2$ </td><td> $6 8 . 4 \pm 0 . 5$ </td><td> $8 6 . 1 \pm 0 . 4$ </td><td> $8 2 . 1 \pm 1 . 2 $ </td><td> $8 8 . 2 \pm 0 . 5$ </td><td> $5 2 . 0 \pm 0 . 2 $ </td><td> $5 7 . 2 \pm 1 . 5$ </td><td> $5 3 . 9 \pm 2 . 6$ </td></tr><tr><td>64</td><td> $8 1 . 6 \pm 0 . 3$ </td><td> $6 7 . 9 \pm 0 . 6 $ </td><td> $8 7 . 0 \pm 0 . 4$ </td><td> $8 5 . 7 \pm 0 . 8$ </td><td> $8 9 . 5 \pm 0 . 5$ </td><td> $5 2 . 8 \pm 0 . 2$ </td><td> $6 3 . 9 \pm 0 . 4$ </td><td> $5 6 . 6 \pm 0 . 9$ </td></tr><tr><td>128</td><td> $8 0 . 3 \pm 0 . 1$ </td><td> $6 6 . 4 \pm 1 . 1$ </td><td> $8 7 . 1 \pm 0 . 4$ </td><td> $8 2 . 8 \pm 0 . 7$ </td><td> $8 8 . 1 \pm 0 . 4$ </td><td> $5 1 . 7 \pm 0 . 3$ </td><td> $5 7 . 8 \pm 1 . 7$ </td><td> $5 8 . 3 \pm 1 . 3$ </td></tr></table>

Table 5 examines the effect of the number of neuron clusters. $K = 6 4$ achieves the best overall trade-off, with the strongest results on CV-Bench Distance and BLINK Multi-view. With fewer clusters $( K = 3 2 )$ , neurons corresponding to different visual concepts may be grouped together, reducing routing specificity. With more clusters $( K = 1 2 8 )$ , neurons corresponding to the same or closely related concept may be split across multiple clusters, making cluster prediction less stable and reducing the coherence of concept-level activation.

## 5 Discussion

Limitations. NeuronEye improves tasks that rely on localized spatial or relational evidence, but its effectiveness depends on the coverage of the learned neuron vocabulary. Since the SAE and neuron clusters are trained on VQAv2 and then frozen, the sparse neuron vocabulary reflects general-domain visual concepts and may lack coverage of domain-specific ones. This limitation is evident on a medical domain VQA dataset (Appendix E), OmniMedVQA-Mini [37], where NeuronEye reduces overall accuracy from 65.30% to 63.10%, with larger drops on disease diagnosis (-2.80), lesion grading (-2.18), and other biological attributes (-5.51). NeuronEye also relies on LLM-generated cluster-index labels for routing supervision at scale. We manually verified approximately 20% of the generated labels, but misalignment between query intent and selected neuron clusters may persist in the remainder, potentially degrading the model’s performance. Additionally, NeuronEye introduces approximately 20% inference latency overhead and 42% peak memory increase relative to the base model, primarily due to frozen SAE encoding and localized cross-attention (Appendix F).

Future directions. Future work could expand the sparse neuron space using larger and more diverse training corpora, or construct domain-specific neuron vocabularies for specialized settings such as medical imaging. Extending sparse routing across multiple layers may enable coordinated modulation across levels of visual abstraction. Learning neuron clusters directly from data could further reduce reliance on offline concept labeling and improve robustness under distribution shift.

## 6 Conclusion

We presented NeuronEye, a plug-in framework that decomposes dense VLM representations into a sparse, concept-level neuron vocabulary and selectively activates query-relevant visual concepts at inference, without modifying or retraining the VLM backbone. Experiments across four benchmarks and two backbones demonstrate consistent improvements on spatial, relational, and multi-view reasoning. Our results suggest that sparse autoencoder features can serve not only as interpretability tools but as active interfaces for modulating concept-level visual reasoning.

## References

[1] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In Advances in Neural Information Processing Systems, volume 36, 2023.

[2] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-VL technical report. arXiv preprint arXiv:2502.13923, 2025.

[3] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, and Chunyuan Li. LLaVA-OneVision: Easy visual task transfer. arXiv preprint arXiv:2408.03326, 2024.

[4] Declan Campbell, Sunayana Rane, Tyler Giallanza, Nicolò De Sabbata, Kia Ghods, Amogh Joshi, Alexander Ku, Steven M Frankland, Thomas L Griffiths, Jonathan D Cohen, et al. Understanding the limits of vision language models through the lens of the binding problem. Advances in Neural Information Processing Systems, 37:113436–113460, 2024.

[5] Shengbang Tong, Ellis Brown, Penghao Wu, Sanghyun Woo, Manoj Middepogu, Sai Charitha Akula, Jihan Yang, Shusheng Yang, Adithya Iyer, Xichen Pan, Austin Wang, Rob Fergus, Yann LeCun, and Saining Xie. Cambrian-1: A fully open, vision-centric exploration of multimodal LLMs. In Advances in Neural Information Processing Systems, 2024.

[6] Xingyu Fu, Yushi Hu, Bangzheng Li, Yu Feng, Haoyu Wang, Xudong Lin, Dan Roth, Noah A. Smith, Wei-Chiu Ma, and Ranjay Krishna. BLINK: Multimodal large language models can see but not perceive. In Proceedings of the European Conference on Computer Vision (ECCV), 2024.

[7] Bruno A Olshausen and David J Field. Emergence of simple-cell receptive field properties by learning a sparse code for natural images. Nature, 381(6583):607–609, 1996.

[8] William E Vinje and Jack L Gallant. Sparse coding and decorrelation in primary visual cortex during natural vision. Science, 287(5456):1273–1276, 2000.

[9] R Quian Quiroga, Leila Reddy, Gabriel Kreiman, Christof Koch, and Itzhak Fried. Invariant visual representation by single neurons in the human brain. Nature, 435(7045):1102–1107, 2005.

[10] Rodrigo Quian Quiroga. Concept cells: the building blocks of declarative memory functions. Nature Reviews Neuroscience, 13(8):587–597, 2012.

[11] Anne M Treisman and Garry Gelade. A feature-integration theory of attention. Cognitive psychology, 12(1):97–136, 1980.

[12] Anne Treisman. Feature binding, attention and object perception. Philosophical Transactions of the Royal Society of London. Series B: Biological Sciences, 353(1373):1295–1306, 1998.

[13] Robert Desimone, John Duncan, et al. Neural mechanisms of selective visual attention. Annual review ofneuroscience, 18(1):193–222, 1995.

[14] Behrad Noudoost, Mindy H Chang, Nicholas A Steinmetz, and Tirin Moore. Top-down control of visual attention. Current opinion in neurobiology, 20(2):183–190, 2010.

[15] Liang Chen, Haozhe Zhao, Tianyu Liu, Shuai Bai, Junyang Lin, Chang Zhou, and Baobao Chang. An image is worth 1/2 tokens after layer 2: Plug-and-play inference acceleration for large vision-language models. In Proceedings ofthe European Conference on Computer Vision, 2024.

[16] Yuan Zhang, Chun-Kai Fan, Junpeng Ma, Wenzhao Zheng, Tao Huang, Kuan Cheng, Denis A. Gudovskiy, Tomoyuki Okuno, Yohei Nakata, Kurt Keutzer, and Shanghang Zhang. Sparsevlm: Visual token sparsification for efficient vision-language model inference. In Proceedings of the International Conference on Machine Learning, 2025.

[17] Yuzhang Shang, Mu Cai, Bingxin Xu, Yong Jae Lee, and Yan Yan. Llava-prumerge: Adaptive token reduction for efficient large multimodal models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025.

[18] Long Xing, Qidong Huang, Xiaoyi Dong, Jiajie Lu, Pan Zhang, Yuhang Zang, Yuhang Cao, Conghui He, Jiaqi Wang, Feng Wu, and Dahua Lin. Conical visual concentration for efficient large vision-language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14593–14603, 2025.

[19] Zhuowei Li, Haizhou Shi, Yunhe Gao, Di Liu, Zhenting Wang, Yuxiao Chen, Ting Liu, Long Zhao, Hao Wang, and Dimitris N. Metaxas. The hidden life of tokens: Reducing hallucination of large vision-language models via visual information steering. In Proceedings of the International Conference on Machine Learning, pages 35799–35819, 2025.

[20] Robert Huben, Hoagy Cunningham, Logan Smith, Aidan Ewart, and Lee Sharkey. Sparse autoencoders find highly interpretable features in language models. In International Conference on Learning Representations, 2024.

[21] Mateusz Pach, Shyamgopal Karthik, Quentin Bouniot, Serge Belongie, and Zeynep Akata. Sparse autoencoders learn monosemantic features in vision-language models. In Advances in Neural Information Processing Systems (NeurIPS), 2025.

[22] T. Bricken et al. Towards monosemanticity: Extracting interpretable features from base models, 2023. Transformer Circuits Thread.

[23] Kaichen Zhang, Yifei Shen, Bo Li, and Ziwei Liu. Large multi-modal models can interpret features in large multi-modal models. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2025.

[24] A. Zou et al. Representation engineering: A top-down approach to ai transparency. In International Conference on Learning Representations (ICLR), 2024.

[25] Nina Rimsky, Nick Gabrieli, Julian Schulz, Meg Tong, Evan Hubinger, and Alexander Matt Turner. Steering Llama 2 via contrastive activation addition. In Proceedings ofthe 62nd Annual Meeting of the Association for Computational Linguistics, pages 15504–15522, 2024.

[26] Sangha Park, Seungryong Yoo, Jisoo Mok, and Sungroh Yoon. SAVE: Sparse autoencoderdriven visual information enhancement for mitigating object hallucination. In Proceedings of the IEEE/CVF Winter Conference on Applications ofComputer Vision, 2026.

[27] Zhenglin Hua, Jinghan He, Zijun Yao, Tianxu Han, Haiyun Guo, Yuheng Jia, and Junfeng Fang. Steering lvlms via sparse autoencoder for hallucination mitigation. In Findings of the Associationfor Computational Linguistics: EMNLP 2025, 2025.

[28] S. Antol, A. Agrawal, J. Lu, M. Mitchell, D. Batra, C. L. Zitnick, and D. Parikh. Vqa: Visual question answering. In Proceedings of the IEEE International Conference on Computer Vision (ICCV), pages 2425–2433, 2015.

[29] xAI. RealWorldQA: Evaluating real-world spatial understanding. https://huggingface. co/datasets/xai-org/RealWorldQA, 2024. Dataset. Accessed: May 4, 2026.

[30] Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Jiaqi Wang, Yu Qiao, Dahua Lin, and Feng Zhao. Are we on the right way for evaluating large vision-language models? In Advances in Neural Information Processing Systems (NeurIPS), 2024.

[31] Haoyu Lu, Wen Liu, Bo Zhang, Bingxuan Wang, Kai Dong, Bo Liu, Jingxiang Sun, Tongzheng Ren, Zhuoshu Li, Hao Yang, Yaofeng Sun, Chengqi Deng, Hanwei Xu, Zhenda Xie, and Chong Ruan. DeepSeek-VL: Towards real-world vision-language understanding. arXiv preprint arXiv:2403.05525, 2024.

[32] Hugo Laurençon, Andrés Marafioti, Victor Sanh, and Léo Tronchon. Building and better understanding vision-language models: Insights and future directions. arXiv preprint arXiv:2408.12637, 2024.

[33] Abdelrahman Abouelenin, Atabak Ashfaq, Adam Atkinson, Hany Awadalla, Nguyen Bach, Jianmin Bao, Alon Benhaim, Martin Cai, Vishrav Chaudhary, Congcong Chen, et al. Phi-4-Mini technical report: Compact yet powerful multimodal language models via mixture-of-LoRAs. arXiv preprint arXiv:2503.01743, 2025.

[34] Jinguo Zhu et al. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479, 2025.

[35] Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. Llava-next: Improved reasoning, ocr, and world knowledge, January 2024.

[36] Ting Liu, Liangtao Shi, Richang Hong, Yue Hu, Quanjun Yin, and Linfeng Zhang. Multi-stage vision token dropping: Towards efficient multimodal large language model. arXiv preprint arXiv:2411.10803, 2024.

[37] Yutao Hu, Tianbin Li, Quanfeng Lu, Wenqi Shao, Junjun He, Yu Qiao, and Ping Luo. Omnimed vqa: A new large-scale comprehensive evaluation benchmark for medical lvlm. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.

## A Offline Interpretation Prompts

Figure 6 shows the offline prompts used for sparse-neuron interpretation and cluster-index supervision. We use Claude 3.7 Sonnet, a multimodal model with vision capabilities accessed through the Anthropic API, as an external annotator. The first prompt assigns short concept labels to sparse neurons based on representative image regions that strongly activate each neuron. The second prompt maps training questions to relevant cluster indices using the offline cluster inventory. The resulting annotations are used only before training to construct neuron clusters and generate cluster-index labels; they are not provided to the router, the VLM backbone, or any inference-time component.

![](images/8f9726106ec2c923e265a4ac96e3db9bf40ca9a3a147fe32c02fad3360b69600.jpg)  
Figure 6: Offline prompts used for sparse neuron interpretation and cluster-index supervision.

## B Training, Inference, and Hyperparameter Details

Training follows a two-stage schedule. In Stage 1, the router and vision-side scorer are trained using frozen hidden states, optimizing the cluster-index prediction loss and the query–vision alignment loss before any feature injection is introduced. This stage stabilizes the query-to-cluster mapping. In Stage 2, the full NeuronEye pipeline is enabled: NVF performs sparse feature localization, queryconditioned refinement, and localized injection, while PCS suppresses dominant directions at the later layer. The lightweight components are optimized with the combined objective

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { L M } } + \lambda _ { \mathrm { r o u t e r } } \mathcal { L } _ { \mathrm { r o u t e r } } .
$$

Throughout both stages, the VLM backbone, pretrained SAE, and neuron-cluster assignments remain fixed. Only the lightweight NVF modules and the PCS scalar are updated.

At inference time, NeuronEye runs in a single modified forward pass. NVF selects neuron clusters, localizes and refines the corresponding sparse features, and injects localized updates at layer ℓ, while PCS suppresses dominant directions at layer ℓ<sup>′</sup>. No external annotator, retrieval module, parameter update, or multi-pass decoding is used during inference.

Table 6: Hyperparameter settings for all experiments.
<table><tr><td>Component</td><td>Hyperparameter</td><td>Value</td></tr><tr><td rowspan="3">Sparse Neuron Space</td><td>Expansion ratio α</td><td>32</td></tr><tr><td>Sparsity level k</td><td>32</td></tr><tr><td>Sparsity coefficient  $\lambda _ { s }$  Min. image threshold  $M _ { \mathrm { m i n } }$ </td><td>0.05 3</td></tr><tr><td rowspan="5">Neuron-guided Visual Focus</td><td>Top-n patches per cluster</td><td>60</td></tr><tr><td> $\mathrm { T o p } { - } K _ { \mathrm { s e l } }$  clusters</td><td>5</td></tr><tr><td> $\gamma _ { \mathrm { i n j } }$  init</td><td>0.75</td></tr><tr><td>Bottleneck dim</td><td>128</td></tr><tr><td> $\beta _ { \mathrm { r e f } }$  init</td><td>-3.0</td></tr><tr><td>PCS</td><td>Suppressed directions r  $\eta _ { \mathrm { p a r a m } }$  init</td><td>2 -3.0</td></tr><tr><td rowspan="4">Training</td><td>Learning rate</td><td> $1 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Gradient accumulation</td><td>6</td></tr><tr><td></td><td></td></tr><tr><td>Training samples</td><td>5K</td></tr><tr><td></td><td> $\lambda _ { \mathrm { { r o u t e r } } }$   $\lambda _ { \mathrm { a l i g n } }$ </td><td>0.5 0.3</td></tr></table>

## C Additional Sparse Neuron Visualizations

Figures in this section provide additional examples of individual SNS neurons and their top-activating patches. Each visualization shows image regions that strongly activate one sparse latent dimension, with non-activating regions masked. These examples illustrate that many retained neurons correspond to localized and semantically coherent visual patterns, supporting their organization into concept-level neuron clusters.

![](images/9dcc0f57e3d3445701a9cb7c71eac829a655b24e1fc78b9adadca60736ce30de.jpg)  
(a) Vehicle feature

![](images/00979b21056b201fadd3d7b9668b9b24dbedc3b7fa8c451dd546059a46fe77ca.jpg)  
(b) Building feature

![](images/5b432bb5b665b04368232705a03912d2eab82847d39b3e92fa8b270a6ac072e6.jpg)

![](images/c4a2042103855d125b7a1443d5066219086ff1d50801ad3ba127ae250a5a0af4.jpg)  
(d) Rocky terrain feature

(c) Snow feature  
![](images/e98f977c3499d434dfd68cc6dd75371979726559496b64d4d78c2c9507caddd5.jpg)

![](images/0e67b0f33b5df0f03b2170f687bbd9e44742e9550c56337c7194ad2c183d0aee.jpg)  
(f) Sky and sun feature

(e) Street signs feature  
![](images/4933e72de0d70bee603817736cc891586c6a07821f0699814b0b39bc8478aa46.jpg)  
(g) Fruit feature

![](images/cf9f26ac6070219775225e9d045851fd0ae0348e80312ed92e654972f57776c7.jpg)  
(h) Hands feature

![](images/4775a37573c84c2a5315e04eae7730fbcad35c2e5ccfb1dd9f9f19297b0eb438.jpg)  
(i) Water sports feature

![](images/aa0c3f5007823e6f4baa3f1c2e53a08a22ed80872a2966fe69cc00d0b4102f85.jpg)  
(j) Elephant feature

![](images/44345c836154f767da95b1fd324339f103f18cbf7da9c002f956356c3575d550.jpg)  
(k) Grass feature

![](images/8bb42660c9364c5c256d5baa43547cf736174eca6e5ed46d770bc9ce54d00720.jpg)  
(l) Bicycle feature

![](images/abe11cda52c7df09a00244c2d2a034acf551278ab0488cbb06672777ac9d0e16.jpg)  
(m) Textual information feature

![](images/6428975ee359dc087ff6bb2664d2bbf2a8b2738223e457f44aeee442c6daeadc.jpg)  
(n) Flowers feature  
Figure 7: Additional sparse neuron visualizations. Each subfigure shows the top-activating patches for a single SNS neuron across multiple images, with non-activating regions masked.

## D Additional Question-Guided Focus Examples

Figure 8 provides additional examples of query-guided visual focus produced by NVF. In each case, NVF selects a small set of query-relevant neuron clusters and increases the activation of neurons within these clusters, yielding cluster-specific activation maps that highlight localized image regions relevant to answering the question.

![](images/9bf7e3e9e3639d8925f713f4240c0279644c6374d35959c66d4cdc0d3931b751.jpg)  
Input Image

![](images/20e366a374c1b19a4ccca195f2a7aabcc4e19a73284448c69daadd8e430938d5.jpg)  
Overall Focus

![](images/c5a6a6979573cdaf8e9dd74fb374f9e08ef6a6d257d930b286aa9f80f5dea1b8.jpg)  
Neuron Cluster 1

![](images/2c2983483f28a7e94f35fcf90d3fda70324cf52a41f1a07c3f6a12ccf13ade3e.jpg)  
Neuron Cluster 21

(a) How many people are wearing a red shirt?  
![](images/7b057e4c3598aefa6fb2915236304125509da4157030b466f09a74a4c65f8fdd.jpg)  
Input Image

![](images/a352d31461ec2f820b7c13381def55e5530d805828954558441ed01682949ed3.jpg)  
Overall Focus

![](images/b9a8d0cbac34b0c5d7a4d9e7497888c940a928efe33533efd869c975332bcb4a.jpg)  
Neuron Cluster 44

![](images/d07f4638ededbf3bd68820414c30b08911a0130ef5a8614cf67ab918dd002ea5.jpg)  
Neuron Cluster 42  
Neuron Cluster 58

![](images/30fbc2f92f5600acde10a1897630c67112d3f2712d1e4f46fe925323d961c38f.jpg)  
Input Image

(b) Did the person hit the ball?  
![](images/28aca860c8f6d7a909784cbe9312b4b663dc6a116099f4529374e852883f0082.jpg)  
Overall Focus

![](images/4d81c0460cf4e993bb827e00009882ff60486a3eced3db683bd1b1c99218424c.jpg)

![](images/ff923dd0b416068ca68314c54fb0fe0d5d52e8a4b815f410a3e9a1780c599eef.jpg)  
Neuron Cluster 47

![](images/d98048ebdecee06c92aefe54c9124b8f3dd9ab208192be7af797d2763ea85950.jpg)  
Neuron Cluster 21

![](images/ee24721ad84e0935f82fcf064e0967595dab50760fa41c55e71fcba5ee6937b9.jpg)  
Neuron Cluster 58

(c) Where is the man looking at?  
![](images/7892e262456b758a7fa843c7b909a08a528d2a8254dc06927f5e3ab06cb42bcb.jpg)  
Input Image

![](images/004ff46bd429046f171de50897f9d4287fdfdccea0400bee9dfc9257673e7e73.jpg)  
Overall Focus

![](images/4c9e3b41660f97c8e753cb0a4ef4d2c8e89866c024d57d77edf198e69968c230.jpg)  
Neuron Cluster 19

![](images/defb401edf0b41765209a748bdb515845fe91406fd484f967ca183bc2c34adf6.jpg)  
Neuron Cluster 21

![](images/be2c896ce5715617659d074da498f7fe811930b09b90b574f742f4ca684aaf50.jpg)  
Neuron Cluster 24

![](images/b47c7b11194583c77222933d8b5f1bf20d84b4e3999afce6133a16d4575536ac.jpg)  
Input Image

(d) How many people are on the elephant?  
![](images/4a6f75c0464fa4afd8b38b62bff2358d647ba3e24bef65ab70eea78f24a9e903.jpg)  
Overall Focus

![](images/f7a682d27e9f19ec559e8a898a370c5afcecd8923ecc0a662522191a089bb94a.jpg)  
Neuron Cluster 30

(e) Are there passengers waiting to board?  
![](images/f874be62c420d9c0c23a89fdb6bec4246470914186358c8eb35d6cac13bdb286.jpg)  
Neuron Cluster 8

Figure 8: Additional question-guided visual focus examples.

## E Additional Domain-Shift Evaluation

Table 7: Performance comparison on OmniMedVQA-Mini by overall accuracy and question type.
<table><tr><td>Evaluation Setting</td><td>Base Qwen2.5-VL</td><td>NeuronEye</td><td>Difference</td></tr><tr><td>Overall Accuracy</td><td>65.30%</td><td>63.10%</td><td>-2.20</td></tr><tr><td>Anatomy Identification</td><td>48.52%</td><td>48.10%</td><td>-0.42</td></tr><tr><td>Disease Diagnosis</td><td>63.74%</td><td>60.94%</td><td>-2.80</td></tr><tr><td>Lesion Grading</td><td>57.61%</td><td>55.43%</td><td>-2.18</td></tr><tr><td>Modality Recognition</td><td>97.57%</td><td>96.18%</td><td>-1.39</td></tr><tr><td>Other Biological Attributes</td><td>71.72%</td><td>66.21%</td><td>-5.51</td></tr></table>

Table 7 evaluates NeuronEye under medical-domain shift on OmniMedVQA-Mini. NeuronEye decreases overall accuracy from 65.30% to 63.10%, with larger drops on disease diagnosis, lesion grading, and other biological attributes. This result suggests that a sparse neuron vocabulary constructed from general-domain VQAv2 activations may not fully cover specialized medical visual concepts. It supports the limitation discussed in Section 3.3 and motivates domain-specific SNS construction for specialized applications.

## F Efficiency and Overhead

Table 8: Inference overhead relative to the base Qwen2.5-VL model.
<table><tr><td>Configuration</td><td>Latency (s)</td><td>Lat. ∆ (%)</td><td>Peak Mem. (MB)</td><td>Mem. ∆ (%)</td></tr><tr><td>Base (Qwen2.5-VL)</td><td>0.382</td><td></td><td>16,857</td><td></td></tr><tr><td>+ SNS + NVF</td><td>0.456</td><td>+19.4</td><td>23,933</td><td>+42.0</td></tr><tr><td> $+ \mathrm { S N S } + \mathrm { N V F } + \mathrm { P C S }$ </td><td>0.457</td><td>+19.7</td><td>23,933</td><td>+42.0</td></tr></table>

Table 8 summarizes the inference overhead relative to the base model. NeuronEye introduces a moderate latency increase (≈20%), mainly due to frozen SAE encoding and localized cross-attention. The memory increase mainly comes from intermediate activations, routing buffers, and the frozen SAE, rather than additional trainable components. PCS adds negligible extra latency and no additional peak memory in this setting.

Table 9: Parameter breakdown of the added components. Percentages are relative to the 7.6Bparameter base model.
<table><tr><td>Module</td><td>Component</td><td>Params</td><td>% of Base</td><td>Stage 1</td><td>Stage 2</td></tr><tr><td rowspan="2">SNS</td><td>SAE</td><td>822.08M</td><td>10.79%</td><td>Train</td><td>Frozen</td></tr><tr><td>Neuron Clusters</td><td></td><td></td><td>Built</td><td>Frozen</td></tr><tr><td rowspan="5">NVF</td><td>Router</td><td>6.54M</td><td>0.09%</td><td>Train</td><td>Train</td></tr><tr><td>Vision-Side Scorer</td><td>6.54M</td><td>0.09%</td><td>Train</td><td>Train</td></tr><tr><td>Projection Layers</td><td>12.85M</td><td>0.17%</td><td></td><td>Train</td></tr><tr><td>Localized Cross-Attention</td><td>51.39M</td><td>0.67%</td><td></td><td>Train</td></tr><tr><td>Refinement Module</td><td>29.84M</td><td>0.39%</td><td>一</td><td>Train</td></tr><tr><td>PCS</td><td>ηparam</td><td>1</td><td>≈0%</td><td></td><td>Train</td></tr><tr><td colspan="2">Trainable modules</td><td>107.16M</td><td>1.41%</td><td></td><td>一</td></tr></table>

Table 9 reports the parameter breakdown of the added components. The 1.41% parameter overhead refers only to trainable modules. The SAE is trained once for constructing SNS and then kept frozen during lightweight module training and inference. Thus, the frozen SAE contributes non-trainable parameters and memory overhead, while the trainable adaptation cost remains limited to the router, vision-side scorer, projection layers, localized cross-attention, refinement module, and PCS scalar.