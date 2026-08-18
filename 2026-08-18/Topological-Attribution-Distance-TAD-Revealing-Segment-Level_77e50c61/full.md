# Topological Attribution Distance (TAD): Revealing Segment-Level RAG Influence on LLM Output Geometry for Incident Log Analysis

1<sup>st</sup> Reza Fayyazi

2<sup>nd</sup> Michael Zuzak

3<sup>rd</sup> Shanchieh Jay Yang

Dept. of Computer Engineering

Dept. of Computer Engineering

Rochester Institute of Technology

Rochester, NY, USA

Dept. of Engineering

Rochester Institute of Technology

rf1679@rit.edu

Rochester, NY, USA

mjzeec@rit.edu

Gonzaga University

Spokane, WA, USA

yangj@gonzaga.edu

Abstract—Large Language Models (LLMs) are increasingly being deployed in cybersecurity operations to assist cybersecurity analysts with rapid decision-making against emerging threats. However, there is a main criteria that must be met when using LLMs in cybersecurity, that is, trust in the generated outputs. As Agentic AI is integrated into operational systems, a robust evidence attribution and provenance tracking technique is essential to trace the origins of model generations. When autonomous agents make a decision (right or wrong), the ability to trace back through the decision chain is critical, as without it, teams cannot identify which segment of the data caused the model generation. Existing methods often struggle to distinguish among complex and highly similar evidence sources, such as cyber incident logs. This reveals a key gap: current approaches do not adequately capture the holistic geometric relationship between the retrieved evidence and the generated response for reliable evidence verification. To bridge this gap, we propose Topological Attribution Distance (TAD), inspired by Topology, to characterize and capture the global geometric shape of an output and its changes against its retrieved logs. In other words, if the embeddings of a specific source log drastically changes the geometry of the model’s response in the embedding space, this suggests that such log is a critical source for the model’s generated response. Therefore, TAD is powered by segment-level ablation attribution to investigate incident logs of an actual cyberattack. We demonstrate how TAD finds the most attributed logs on LLM outputs in an adaptive manner. This can provide an explainable and trustworthy tracing based on each LLM’s hidden state to understand how geometrically different retrieved logs influence the model generation, and provide evidence verification in cybersecurity and Agentic-AI workflows.

Index Terms—LLM, Topology, TDA, RAG, Wasserstein Distance, Attribution, Incident Log Analysis, TAD

## I. INTRODUCTION

Retrieval-Augmented Generation (RAG) has become a widely used approach in Agentic-AI workflows by grounding Large Language Models (LLMs) with relevant external knowl edge. In cybersecurity, RAG allows LLMs to access up-to-date information, such as vulnerability reports and incident logs, to help support more informed security analysis and decisionmaking. However, retrieval alone does not guarantee correct interpretation, particularly when models are exposed to large pools of highly structured and semantically similar information. As noted by Zhang et al. [1], RAG systems remain susceptible to ungrounded outputs when queries are ambiguous, retrieved context is excessive, or source quality is poor. This raises a critical concern for AI adoption in cybersecurity applications, and that is: trust in the generated outputs. As Agentic AI becomes integrated into operational systems, robust evidence attribution and provenance tracking are essential for tracing model decisions back to their originating sources. Without such traceability, decisions made by AI agents (whether right or wrong) are difficult to diagnose, verify, and audit. Therefore, there remains the challenge to reliably attributing model decisions to their originating sources for evidence verification. Understanding why a model generated a particular response and which retrieved context influenced the models’ decision is critical for providing trust, transparency, and auditability in the age of Agentic AI.

To tackle these challenges, there have been several lines of work that investigated providing explainability and attribution for LLMs’ generated responses [2]–[7], yet each carries notable limitations. One prominent approach employs LLM-based judges to assess provenance [2]–[4]. Because these methods rely on the same class or another set of models being evaluated, their accuracy is fundamentally constrained by the biases, hallucination tendencies, and self-assessment patterns inherent to LLMs themselves. A second body of work focuses on token-level attribution by producing importance scores for individual input tokens to identify which tokens influence a given generation the most [5]–[7], or similarity measures (e.g., cosine similarity and Rouge-L). These methods are either computationally expensive or ineffective on highly structured and similar data, such as incident logs. Therefore, these methods do not capture holistic segment-level grounding, which is the degree to which a model’s response as a whole is driven by retrieved sources. In other words, a security log’s influence on model behavior is better understood holistically, as a complete unit of evidence, rather than as a collection of individually scored tokens. For example, in structurally similar logs, the key distinction may arise from the connectivity between a few tokens, such as psexec.exe appearing at an unusual timestamp. Token-level attribution can dilute such signals across many shared, non-discriminative tokens. This highlights a key gap in current attribution methods: the lack of a mechanism that aggregates token-level signals into coherent segment-level attribution that identifies when a small but critical feature makes an entire retrieved segment influential.

To account for these challenges, we turn into Topological Data Analysis (TDA), a mathematical framework for studying the shape of data [8]. TDA draws on concepts from Topology, where algebraic invariants, such as homology and homotopy groups, are assigned to topological spaces to characterize their structure [9]. Homotopy and homology are two lenses on the same underlying question: “When are two topological spaces the same”? Homotopy is a continuous deformation between two maps $( f , g : X \to Y )$ , formalized as a continuous function $( H : X \times [ 0 , 1 ] \to Y )$ satisfying $( H ( x , 0 ) = f ( x ) )$ and $( H ( x , 1 ) = g ( x ) )$ . Intuitively, homotopy captures the idea that two spaces are equivalent if one can be continuously stretched or compressed into the other. While homotopy is the more intuitive notion, it is generally intractable to compute directly [10]. Homology addresses this by extracting computable algebraic invariants (e.g., H0 for connected components, H1 for finding holes, etc.) that are preserved under homotopy equivalence. In other words, homology acts as a computationally tractable proxy for homotopy, meaning that if two spaces have different homology, they cannot be homotopy-equivalent.

Existing research has shown that feed-forward networks progressively transforms the topology of the data as they flow through the layers, which means that Topology has the capability to map the non-linear space [11]. Prior work has also shown that deeper layer does not always mean better representation, demonstrating the importance of effect of all the layers into the final conclusion [12]. Therefore, in the context of LLMs, the geometric and topological properties of the embedding spaces over the layers (i.e., non-linear hidden state spaces) can produce meaningful explanations about model decisions, attribution, and uncertainty [13], [14]. Interestingly, the auto-regressive nature of LLMs (i.e., prior context shaping the response) can provide a powerful lens for reasoning about model generations. Instead of comparing isolated token representations from the response, we can characterize the token connectivity of a response as a whole through the embedding space and trace the influence of each prior context (each segment) individually. In other words, if the inclusion and omission of a retrieved context embeddings can strongly deform the geometry of the model’s response, this can suggest as evidence that the context served as a grounding source. This topological perspective directly addresses the segment-level attribution gap identified above.

Building on this intuition, we propose Topological Attribution Distance (TAD), a segment-level attribution metric to identify the segments (e.g., logs) that cause the most geometric change in an LLM’s generation. TAD operates by tracking the topological signatures of the generated response’s embedding geometry over the transformer layers, both with and without each retrieved segment present. More specifically, we compute the generated response’s persistence diagram from their embedding geometries across each transformer layer, with and without each segment in the context, and quantify the changes on the response embeddings with Wasserstein distance. By measuring the topological cost of significant changes on a generated response’s geometry, TAD produces an attribution score at the segment-level. This score captures the degree to which each retrieved segment geometrically influenced the model’s output.

We evaluate TAD on a real-world cyberattack logs, and on three different scenarios, namely Direct, Regular, and Indirect. These scenarios point to the level of difficulty of a response with respect to the ground-truth attack logs. More specifically, the Direct case contains many tokens from the log in the response, the Regular case contains some specific keywords, and the Indirect case contains little-to-none token overlaps between the logs and the response. Moreover, we compare TAD against multiple baselines, namely LLMas-a-judge, similarity measures, and token-level attribution techniques. We demonstrate the effectiveness of the proposed TAD metric across four different LLMs (with different sizes and architectures). Our results demonstrate that TAD significantly outperforms existing baselines with over 97% accuracy on average in tracing the top-attributed logs, and achieving over 39% gap in Precision compared to the best-performing baseline in all three scenarios. In Figure 1, we provide a case study to demonstrate how different attribution techniques fail in tracing security logs to the response, and how TAD works well in these cases. Together, these results position TAD as an auditable, provenance-aware measure capable of tracing model decisions back to their retrieved sources, and therefore, building trust in cybersecurity operations and critical decision-making.

The key contributions of this work are as follows:

• We formalize the problem of segment-level attribution in RAG systems as a topological comparison problem, drawing an explicit connection between homotopy-theoretic reasoning and embedding-space geometry.

• We introduce Topological Attribution Distance (TAD), a novel attribution metric that captures segment-level influence of changes in LLM response geometry over the hidden states, and compute their Wasserstein distance to find the top-attributed segments.

• We demonstrate that TAD is well-suited for incident log analysis, as logs share high token overlap within each other, and show how TAD is capable of identifying the logs that influence the LLM response’s geometry the most.

• We demonstrate that the proposed TAD metric outperforms other baselines with an average of 97% accuracy on four different LLMs, and with over 39% gap in Precision for the three different difficulty-level cases.

## II. RELATED WORKS

## A. Challenges of Existing Attribution Metrics for LLMs

The emergence of RAG systems has led to a line of research on source attribution, which is verifying whether the generated text is supported by identified source documents. For LLMbased judges, there have been different works proposed to

## Case Study: From an LLM Generation to Tracing Back its Supporting Evidence

Examine logs around 00:01–00:05 on 2025-01-01. Exactly one of the following five logs is evidence of malicious activity. Identify its log\_id and explain why it is malicious in one sentence.

<table><tr><td>Log</td><td>Log-ID</td><td>Event</td><td>State</td></tr><tr><td>1</td><td>2e9a4...5a094</td><td>jdoe starts teams.exe.</td><td>Benign</td></tr><tr><td>2</td><td>8b1d6...fa5bd</td><td>SYSTEM modifies HKLM\...\WindowsUpdate.</td><td>Benign</td></tr><tr><td>3</td><td>d0c4a...73f06b</td><td>jdoe logs on via explorer.exe.</td><td>Benign</td></tr><tr><td>4</td><td>3a7e1...bd5a02</td><td>jdoe starts code.exe.</td><td>Benign</td></tr><tr><td>5</td><td>7c3f0...b0a48</td><td>jdoe writes HKCu\...\Run → %TEMP%\svc.exe.</td><td>Malicious</td></tr></table>

## 3. LLM Response (Gemma3-4b).

⇓ Can the Supporting Evidence be Verified?
<table><tr><td colspan="6">4. Evidence localization</td></tr><tr><td rowspan="2">Family</td><td>Method</td><td>Result</td><td>Operational consequence</td><td></td></tr><tr><td>Cosine</td><td></td><td>X Log 4</td><td>Retrieves a semantically similar but benign process event.</td></tr><tr><td rowspan="2">Self-eval.</td><td>ROUGE-L</td><td></td><td>X Log 2</td><td>Lexical overlap favors a benign registry event.</td></tr><tr><td>Citation</td><td></td><td>X Logs 1,3,4,5</td><td>Includes the true log but also incorrectly cites multiple benign events.</td></tr><tr><td rowspan="2">Token-level attribution</td><td>LLM judge</td><td>▲ Logs 4,5</td><td></td><td>Includes the true log but leaves a benign candidate for manual review.</td></tr><tr><td>LEA [5]</td><td></td><td>X Log 3</td><td>Ranks a benign logon event first as logs share so many overlapping tokens.</td></tr><tr><td>Segment-level attribution</td><td>TAD (ours)</td><td></td><td>√ Log 5</td><td>Uniquely identifies the log supporting the response; verifying evidence localization.</td></tr></table>

Fig. 1: Evidence-localization case study. The Gemma model identifies the malicious registry-persistence event but partially hallucinates its exact log identifier, making the generated answer insufficient for direct verification. Existing methods retrieve benign logs, return multiple candidates, or cite almost the full context. TAD uniquely localizes Log-5 by measuring the effect on the response geometry, allowing an analyst to trace the response to its supporting evidence.

evaluate RAG systems. Self-RAG [2], RAGAS [3] and ARES [4] employ LLM-based judges to self-assess RAG systems, either through the generation model itself, or through another set of LLMs. However, these methods are constrained by the evaluation models’ self-interpretation and reasoning biases, and they do not fully decouple attribution assessment from parametric model knowledge (i.e., hidden states).

Token-attribution methods aim to quantify the contribution of individual input features, typically token embeddings, on a model’s output. Numerous works have addressed the problem of attributing model predictions to individual input features [15]. In LLMs, LIME [16] generates variations of the input text (e.g., by masking or removing tokens) and observes changes in output probabilities [7], and SHAP [17] scores are derived by masking or perturbing subsets of input tokens and observing output variation [7]. However, these methods are computationally expensive, with SHAP being exponential, and cannot scale with a large pool of retrieved context (e.g., security logs). LRP [18] is a backpropagation-based method that redistributes the model’s prediction score backward through the network layers, assigning relevance scores to input features, and there have been works used LRP for attribution [6], [19]. Furthermore, in our previous work [5], we proposed the LLM Embedding-based Attribution (LEA) metric, a computationally efficient tokenlevel attribution metric based on linear dependency analysis of input embeddings. Overall, despite significant progress, existing feature attribution techniques are limited to the tokenlevel and can struggle when retrieved information contains highly overlapping tokens (e.g., logs). These methods While token-level attribution is useful for identifying which tokens most influence a given prediction, it does not capture holistic segment-level grounding, which is the degree to which the model’s response as a whole is driven by the retrieved context. This points to a broader open challenge in the field: bridging the gap between token-level attribution into segment-level attribution tracing in RAG systems.

## B. Topological Data Analysis (TDA) in LLMs

The application of Topological Data Analysis to language models spans several interrelated directions [20]. Draganov and Skiena [21] demonstrate that persistent homology applied to word embedding clouds encodes meaningful linguistic structure. Rottach et al. [22] introduced Unified Topological Signatures, which is a framework that aggregates topological and geometric descriptors to characterize embedding spaces. The authors revealed that models within the same family show similar topological properties, which suggests that architecture and training data strongly shape the embedding space geometry.

Within LLM internal representations, there are some works that apply topology directly to hidden-state activations. Fitz et al. [23] compare the layer-wise topological complexity of Transformer and LSTM hidden-states using persistent homology and perforation, and they found that LSTMs exhibit richer topological structure specific to natural language. Gardinazzi et al. [13] employ zigzag persistence to track how topological features evolve across layers, introducing a Persistence Similarity metric that identifies redundant layers and allows principled pruning with minimal performance loss. Fay et al. [14] propose persistent homology as an architecture-agnostic adversarial detection framework, identifying a consistent “topological compression” signature under prompt injection and backdoor attacks, where adversarial inputs simplify latent-space structure, and demonstrating clean separation of normal and adversarial activations via barcode summary statistics.

Despite their promise, existing topological approaches to LLM analysis have notable limitations. Fitz et al. [23] demonstrated transformer models does not show topological complexity as going deeper in the layers (compared with LSTM). However, their perforation metric excludes H0 homology to capture connected components. We will show in this paper, that H0 plays a key role in finding shape differences in the embedding space. Gardinazzi et al. [13] showed the importance of each layer in different LLMs. While certain layers contribute more significantly than others to model outputs, pruning even the least critical layers resulted in substantial degradation in benchmark performance. This indicates the importance of representation across all layers, and therefore, motivating us to incorporate this insight into our proposed methodology. Furthermore, Fay et al. [14] identified topological compression as an adversarial signature across six LLMs. However, the framework focuses exclusively on a last-token hidden state and relies on batch-level statistics, which requires them to depend on batches of data on similar categories (clean or poisoned) to obtain barcode summaries. This dependence on homogeneous batches to define relational geometry prevents the attribution of a single generated response in isolation, and lacks real-time applicability. Finally, none of these works account for segment-level attribution and multi-source influence on response generation.

## III. BACKGROUND ON TOPOLOGICAL DATA ANALYSIS

In this section, we explain the mathematical foundations of Topological Data Analysis (TDA). We discuss the construction

of simplicial complexes from point cloud data, persistent homology to track topological features across scales, and the Wasserstein distance for comparing persistence diagrams.

## A. Point Clouds and Metric Spaces

The input to TDA is typically a point cloud, i.e., a finite set represented as:

$$
V = \{ v _ { 1 } , v _ { 2 } , \ldots , v _ { n } \} \subseteq M ,\tag{1}
$$

where each point v<sub>i</sub> is embedded in a feature space M equipped with a metric $d \colon M \times M \to \mathbb { R }$ , so that $( M , d )$ forms a metric space. The metric $d ( v _ { i } , v _ { j } )$ quantifies the dissimilarity between any two points. A common choice when $M = \mathbb { R } ^ { n }$ is the $L _ { \infty }$ distance,

$$
d ( v , w ) = \| v - w \| _ { \infty } = \operatorname* { m a x } _ { k } | v _ { k } - w _ { k } | .\tag{2}
$$

The central insight of TDA is that a discrete set of points carries latent geometric and topological information that can be systematically extracted from their pairwise distances [24]. Thus, doing this requires lifting the point cloud to a higher-level combinatorial structure called a simplicial complex.

## B. Simplicial Complexes

A simplicial complex provides a combinatorial representation of a point cloud using basic building blocks called simplices [25]. A k-simplex is defined as the convex hull of $k + 1$ affinely independent points. In other words, $k = 0$ gives vertices, $k = 1$ gives edges, $k = 2$ gives filled triangles, etc. One standard construction for building a simplicial complex from a metric point cloud is the Vietoris–Rips complex [26]. Given a scale parameter $\varepsilon > 0$ , the Vietoris–Rips complex over a point set V is defined as:

$$
\mathcal { R } _ { \varepsilon } = \{ \sigma \subseteq V \mid d ( v _ { i } , v _ { j } ) \leq \varepsilon \forall v _ { i } , v _ { j } \in \sigma \} .\tag{3}
$$

That is, a k-simplex is included whenever all pairwise distances among its $k + 1$ vertices are at most ε. The choice of ε decides which connections are formed: too small a value causes an almost empty complex capturing only trivial local structure, while too large a value collapses all points into a single fully connected component, which motivates studying the complex across all scales simultaneously.

## C. Filtrations and Multi-Scale Analysis

There is no value of ε that can be globally optimal. TDA resolves this ambiguity by considering all scales simultaneously through a filtration [25], which is a nested sequence of simplicial complexes:

$$
\varnothing = K _ { 0 } \subseteq K _ { 1 } \subseteq K _ { 2 } \subseteq \cdot \cdot \cdot \subseteq K _ { m } = K ,\tag{4}
$$

These are indexed by an increasing sequence of parameter values $\varepsilon _ { 0 } < \varepsilon _ { 1 } < \cdots < \varepsilon _ { m } . \mathrm { A s } \ \varepsilon \ \mathrm { g r o w s } ,$ new simplices are progressively added. Initially, points appear as isolated vertices; edges then form between nearby points; eventually, higherdimensional simplices emerge, which represents clustered regions of the space. Given a scalar function $f : \mathcal { V } \to \mathbb { R }$ defined on the point cloud $\nu \subset \mathbb { R } ^ { n }$ , the associated sublevel-set filtration [27] is the nested family of subsets

$$
\mathcal V _ { t } = f ^ { - 1 } \big ( ( - \infty , t ] \big ) = \big \{ v \in \mathcal V \big | f ( v ) \leq t \big \} ,\tag{5}
$$

which is obtained by progressively admitting points in order of increasing the f-value as the threshold t grows. At each step, a simplicial complex (e.g., a Vietoris–Rips complex) is built on $\nu _ { t }$ , capturing how the topology of the space evolves as points are included according to their functional values.

At each stage of the filtration, the shape of the complex is quantified using homology [9]. For each non-negative integer $k ,$ the k-th homology group $H _ { k } ( K )$ , typically computed over a field $\mathbb { F }$ (most often $\mathbb { F } _ { 2 } ~ = ~ \{ 0 , 1 \} )$ , captures independent topological features of dimension k. More specifically, $H _ { 0 } ( K )$ counts connected components, $H _ { 1 } ( K )$ captures independent loops or cycles, $H _ { 2 } ( K )$ detects enclosed voids, and higher $H _ { k } ( K )$ encode their higher-dimensional analogues.

## D. Persistent Homology: Birth, Death, and Persistence

Persistent Homology tracks how homological features evolve across a filtration [24], [25], [28], [29]. As the filtration parameter ε increases, topological features are born (first appear in $H _ { k } ( K _ { \varepsilon } ) )$ and eventually $d i e$ (merge with older components or are filled in by higher-dimensional simplices). The persistence of the feature is:

$$
\mathrm { p e r s } ( \alpha ) = \varepsilon _ { d e a t h } - \varepsilon _ { b i r t h } .\tag{6}
$$

Features with high persistence are considered topologically significant, while features with low persistence are typically attributed to noise or sampling variability [27]. Note that at least one connected component in $H _ { 0 }$ never dies, since some component persists for the entire filtration. The complete lifecycle of topological features can be summarized in two equivalent representations:

Persistence Barcode [30], [31]: A persistence barcode is a multiset of intervals $[ b _ { i } , d _ { i } )$ , drawn as horizontal bars, where each bar corresponds to one topological feature and its length equals the feature’s persistence. Long bars indicate stable features and short bars (near zero) indicate a noisy structure. Persistence Diagram [25], [27], [32]: A persistence diagram is a multiset of points $( b _ { i } , d _ { i } ) \in \overline { { \mathbb { R } } } ^ { 2 }$ with $d _ { i } \geq b _ { i }$ , plotted in the half-plane above the diagonal $\Delta = \{ ( x , x ) : x \in \mathbb { R } \}$ . The diagonal itself is included with infinite multiplicity, representing features that die immediately upon birth. Points far from $\Delta$ correspond to significant features, while points near $\Delta$ indicate noise. The persistence diagram arising from a filtration function $f$ in dimension k is denoted as $\mathrm { D g m } _ { k } ( f )$ . Therefore, having established a means of representing topological features as persistence diagrams, a natural question arises: “how does one quantify the similarity (or distance) between two such diagrams in a geometrically meaningful way?” We will discuss this next.

## E. Wasserstein Distance

To compare two persistence diagrams $P$ and $Q ,$ which may have different cardinalities, one seeks a partial matching γ that minimizes a total cost [27], [32], [33]. To accommodate unmatched points, let $\widehat { P } = P \cup \Delta$ and ${ \widehat { Q } } = Q \cup \Delta$ denote the diagrams augmented with the diagonal $\Delta \ = \ \{ ( x , x ) \ :$ $x \in \mathbb { R } \}$ , treated as trivial features with zero persistence. A matching is then a bijection $\pi : { \widehat { P } }  { \widehat { Q } } .$ , where any point paired with a diagonal point is considered unmatched. The cost of transporting a point $\boldsymbol { u } = \left( b _ { u } , d _ { u } \right)$ to a point $\boldsymbol { v } = \left( b _ { v } , d _ { v } \right)$ under the $L ^ { \infty }$ norm is:

$$
\begin{array} { r } { \| u - v \| _ { \infty } = \operatorname* { m a x } \bigr ( | b _ { u } - b _ { v } | , | d _ { u } - d _ { v } | \bigr ) , } \end{array}\tag{7}
$$

while matching $u = ( b , d )$ to its orthogonal projection onto $\Delta$ incurs the persistence-proportional cost

$$
\begin{array} { r } { \| u - \Delta \| _ { \infty } = \frac { d - b } { 2 } , } \end{array}\tag{8}
$$

reflecting the minimal $L ^ { \infty }$ displacement required to remove a topological feature of persistence $d - b .$ . Therefore, the $( \infty , 1 ) \cdot$ Wasserstein distance $( W _ { 1 } ^ { ( \infty ) } )$ between persistence diagrams $P$ and $Q$ is defined as:

$$
\begin{array} { r } { W _ { 1 } ^ { ( \infty ) } ( P , Q ) \ = \ \underset { ^ { \gamma } } { \operatorname* { i n f } } \ \Biggl [ \ \sum _ { ( u , v ) \in \gamma } \| u - v \| _ { \infty } + } \\ { \ \sum _ { u \in P \backslash \gamma } d ( u , \Delta ) \ + \ \sum _ { v \in Q \backslash \gamma } d ( v , \Delta ) \Biggr ] } \end{array}\tag{9}
$$

where γ ranges over all partial matchings between $P$ and Q, and $d ( u , \Delta ) = ( d _ { u } - b _ { u } ) / 2$ denotes the $\ell ^ { \infty }$ -distance from $\boldsymbol { u } = \left( b _ { u } , d _ { u } \right)$ to its orthogonal projection onto $\Delta .$ . Intuitively, the first sum penalizes geometric discrepancy between matched features, while the second and third sums penalize unmatched features in $P$ and $Q ,$ respectively. Overall, Wasserstein distance provides a geometrically meaningful and computationally tractable measure of dissimilarity between persistence diagrams [33]–[35]. This quantifies how much topological structure must be changed to transform one diagram into another.

## IV. METHODOLOGY

In this section we instantiate the TDA pipeline discussed in Sec. III within the context of LLMs, and discuss the proposed methodology on using Wasserstein distance to track attribution of generations over the transformer layers.

## A. Mapping TDA concepts to LLMs

Token Embeddings as Metric Spaces: In the LLM setting, each point $v _ { i }$ in the point cloud corresponds to a vector representation (embedding) of a token. The space is $M = \mathbb { R } ^ { n }$ where $n$ is the model’s hidden dimension, and the metric d captures geometric dissimilarity between embeddings. We adopt the $L _ { \infty }$ distance, as can show salient directional differences in high-dimensional embedding spaces.

Simplicial Complexes to Encode Semantic Relationships: We apply the Vietoris–Rips complex to the token embedding point cloud. At threshold $\varepsilon ,$ two token embeddings are connected by an edge whenever their $L _ { \infty }$ distance is at most ε. Higher-order simplices form when groups of embeddings are mutually within ε of one another. Since no single ε captures the full hierarchical organization of the embedding space, we construct the full Vietoris-Rips filtration over the token embeddings. Each $K _ { \varepsilon _ { i } }$ represents the simplicial complex at scale $\varepsilon _ { i } ,$ and the nested sequence tracks how semantic groupings form and merge as the threshold grows.

![](images/d1906dc5e039c2d62ff8ac6174fae38b52c464bff81efac20f547329ad63ce70.jpg)  
Fig. 2: TAD’s procedure of tracing attribution with corresponding retrieved data. First, for a generated response, we get the LLM Output shape using Vietoris Rips algorithm. Next, to make TAD scalable across a vast amount of logs, we start by partitioning data into $\sqrt { N }$ and do Wasserstein distance of the original output vs. the ablated batch output. We do this over all the transformer layers and aggregate the overall distance. For batches with largest gap and above, we proceed to Step-3 to analyze the remaining logs. Finally, we do one-by-one ablation with Wasserstein distance to trace the top attributed log(s).

Filtrations can be driven by scalar functions defined on the embeddings. In our setting, a natural choice is the sublevelset filtration parameterized by Layer-wise Hidden States. The filtration captures which tokens are “attended to” at each stage. This construction connects topological analysis directly to model-internal signals, making the filtration sensitive to saliency and anomalous patterns in LLM representations.

Homology Groups for Topological Invariants of the Embedding Space: At each scale $\varepsilon _ { i } ,$ , we compute the homology groups of the Vietoris-Rips complex built over the token embeddings: $H _ { 0 }$ reflects how embeddings group into coherent semantic regions (clusters) at a given scale. $H _ { 1 }$ captures cycles in the embedding space, which may correspond to gaps or circularly related semantic concepts in representation. Higher-dimensional groups $H _ { k } \ ( k \geq 2 )$ encode more abstract multi-way relational structures among embeddings. These homological invariants provide a compact summary of the global geometry of the embedding space at each scale.

Persistent Homology to Track Semantic Structure Across Scales: Applying persistent homology to the token embedding filtration allows for a complete record of which semantic structures appear and disappear as ε grows. A birth event in $H _ { 0 }$ corresponds to a new isolated connected component emerging at scale $\varepsilon _ { b } .$ A death event in $H _ { 0 }$ corresponds to two connected components merging into a broader group at scale $\varepsilon _ { d } .$ . Births and deaths in $H _ { 1 }$ correspond to the formation and filling of cycles of embeddings, which may indicate conceptual relationships. In this work, our primary focus is on $H _ { 0 }$ homology, but we will provide $H _ { 1 }$ results in Appendix $\mathbf { A } .$ The persistence $\mathrm { p e r s } ( \alpha ) \ = \ \varepsilon _ { d } - \varepsilon _ { b }$ of a semantic feature α quantifies its stability. Features with high persistence represent robust and meaningful patterns in the embedding space, such as welldefined topics, while features with low persistence could arise from noise or incidental embedding proximity. The multiset of all such persistence values forms a persistence diagram $\operatorname { D g m } ( X ) \subset \mathbb { R } ^ { 2 }$ , where each feature α contributes a point $( \varepsilon _ { b } , \varepsilon _ { d } )$ . The structural similarity between two such diagrams can then be quantified via the Wasserstein distance (refer to Equ. 9). Therefore, one could ask: “How can these topological descriptors be leveraged to systematically track the attribution and mapping of semantic domains across embedding spaces?” We discuss this in the following section.

## B. Topological Attribution Distance (TAD)

We now turn to the problem of attributing model generations to individual incident log segments. Our approach uses persistent homology to quantify how each retrieved log segment influences the topological structure of the model’s response. To quantify how different RAG segments influence model output, we must first ground our approach in the fundamental properties of decoder-only LLMs: autoregressive token generation. Since each generated token depends exclusively on all preceding tokens in the context window, a critical implication follows: every element in the prompt contributes to shaping the response geometry with varying degrees of causal influence. Therefore, the positional placement, sequential ordering, and compositional structure of RAG segments within the context window can produce measurable effects on model output.

To comprehensively measure how a given set of retrieved logs contributes to the final response, the methodology requires full context concatenation: all segments are appended sequentially into a single context along with the generated response. This design ensures that the causal relationship between each segment and the model’s output can be traced without ambiguity. By doing so, segment-level contribution can be isolated through controlled ablation: systematically excluding individual segments across experimental runs while keeping all other variables constant. Together, these two mechanisms (i.e., full concatenation and targeted ablation) form the basis for attributing response content back to its originating context.

To get the topological mapping of a model’s output with each log segment contributing to the output, we compute the (∞, 1)-Wasserstein distance (refer to Equ. 9) of H0 homology between their corresponding persistence diagrams on each layer of an LLM. The reason for H0 homology is that it is more computationally efficient (see Sec. V), and prior work has shown $\mathrm { H 0 ^ { \circ } s }$ effectiveness on geometric reorganization tasks [36]. To capture an overall representation, we aggregate the Wasserstein values over the layers to obtain the attribution of each log segment. This is due to the nature of transformers that every layer correspond to the final decision, as also discussed by prior work [13]. The following is the proposed TAD metric:

$$
\mathcal { T A D } ( P , Q ) = \sum _ { \ell = 1 } ^ { L } W _ { 1 } ^ { ( \infty ) } ( \mathrm { D g m } _ { 0 } ( P _ { \ell } ) , \mathrm { D g m } _ { 0 } ( Q _ { \ell } ) )\tag{10}
$$

where $D g m _ { 0 }$ represents the H0 homology, and $P _ { l }$ and $Q _ { l }$ are persistent diagrams of the ablated context and the full context on each layer, respectively. This proposed extension can provide attribution beyond token-level impact, as it can provide explainability at segment-level. The key insight is that each log segment that most strongly influences the output, bears representational similarity to the output itself in the feature space. Therefore, ablating these influential segments accordingly should produce substantially larger Wasserstein distances, providing a direct measure of their contribution.

Figure 2 demonstrates our proposed methodology. First, we characterize the topological structure of the generated output by constructing a Vietoris-Rips complex over its representation and extracting H0 persistent homology with the full context + the generated response. Next, since per-log ablation is computationally prohibitive at scale, we adopt a screen-then confirm attribution strategy. In the screen stage, the input log sequence is partitioned into approximately $\sqrt { N }$ batches. Each batch is ablated in turn, and the resulting perturbation to output shape is quantified via the Wasserstein distance between the persistence diagrams of the original and ablated LLM response, computed independently at each transformer layer and aggregated across layers into a single distance measure per batch. The batch containing the largest aggregate distance is then selected for finer-grained analysis. In the confirm stage, we perform single-log ablation restricted to this high-attribution batch, again computing layer-wise Wasserstein distances between resulting persistence diagrams to identify and rank the individual log entries most responsible for shaping the model’s output geometry. In Algorithm 1, we demonstrate the full detail process for the proposed TAD methodology.

## V. DISCUSSION ON COMPLEXITY

The cost of TAD decomposes into two stages: transformer inference and persistent homology. Let N denote the number of candidate log entries in the context, T the number of response tokens in the point cloud, d the hidden dimension of the model, and $L$ the number of transformer layers.

Inference cost. TAD is attribution-by-ablation, therefore, the complexity is the number of forward passes M issued to the model. A purely linear case, in which each log is ablated individually, requires $M = N + 1$ passes (one baseline plus one per log), which scales linearly in context length and becomes prohibitive for long Agentic traces. Our proposed screen-thenconfirm procedure instead partitions the pool into $\lceil \sqrt { N } \rceil$ groups and recurses only on the groups flagged as spikes, which reduces the expected cost to $M = O ( { \sqrt { N } } )$ passes.

Algorithm 1 TAD: Topological Attribution Distance   
Require: Context $C$ with logs ${ \mathcal { L } } ,$ LLM M with $L$ layers,   
homology dimension $d ,$ min grouping size $\tau$ (default 9)   
Ensure: Set of spike logs $\cal { S } \subseteq \cal { L }$   
1: $R  { \mathcal { M } } ( C ) \ \vartriangle$ Response, generated once and held fixed   
2: $\mathcal { D } _ { \mathrm { b a s e } } ^ { ( 1 : L ) }  \dot { \mathrm { D I A G R A M S } } ( C , \dot { R , d } )$   
3: $\mathbf { i f } \ | \mathcal { L } | < \tau$ then ▷ Too small to screen; confirm every log   
4: return $\mathbf { C o N F I R M } ( \mathcal { L } , C , R , \mathcal { D } _ { \mathrm { b a s e } } ^ { ( 1 : L ) } )$   
5: end if   
6: $\mathcal { H }  \mathrm { S c R E E N } ( \mathcal { L } , C , R , \mathcal { D } _ { \mathrm { b a s e } } ^ { ( 1 : L ) } )$   
7: $\mathcal { S }  \mathrm { C O N F I R M } ( \mathcal { H } , C , R , \mathcal { D } _ { \mathrm { b a s e } } ^ { ( 1 : L ) } )$   
8: return $s$   
9: procedure $\mathrm { D I A G R A M S } ( C ^ { \prime } , R , d ) \triangleright$ Only response tokens   
10: $X ^ { ( 1 : L ) } \gets \mathrm { H I D D E N S T A T E S } ( \mathcal { M } , C ^ { \prime } \hat { \oplus } \underline { { R } } )$   
11: return  VIETORISRIPS $( X ^ { ( \ell ) } \mid _ { R } , d ) ) _ { \ell = 1 } ^ { L }$   
12: end procedure   
13: procedure SCREEN $( \mathcal { L } _ { \mathrm { p o o l } } , C , R , \mathcal { D } _ { \mathrm { b a s e } } ^ { ( 1 : L ) } )$   
14: $k  \lceil \sqrt { | \mathcal { L } _ { \mathrm { p o o l } } | } \rceil$   
15: $\mathcal { G }  \dot { \mathrm { P A R T I T I O N } } ( \mathcal { L } _ { \mathrm { p o o l } } , k )$   
16: for each group $G _ { i } \in \mathcal G$ do   
17: $C ^ { \prime } \gets \mathrm { A B L A T E } ( C , G _ { i } ) \triangleright$ Only the context changes   
18: $\mathcal { D } ^ { \prime ( 1 : L ) } \gets \mathrm { D I A G R A M S } ( C ^ { \prime } , \dot { R _ { \cdot } } d )$   
19: $\Delta _ { i } \gets \mathrm { T A D } ( \mathcal { D } ^ { \prime } , \mathcal { D } _ { \mathrm { b a s e } } )$   
20: end for   
21: $\textstyle { \mathcal { L } } _ { \mathrm { h o t } }  \bigcup _ { G \in { \mathcal { G } } _ { \mathrm { h o t } } } G$   
22: if $| { \mathcal { L } } _ { \mathrm { h o t } } | \geq \tau$ then ▷ re-partition   
23: return $\mathrm { S C R E E N } ( \mathcal { L } _ { \mathrm { h o t } } , C , R , \mathcal { D } _ { \mathrm { b a s e } } ^ { ( 1 : L ) } )$   
24: end if   
25: return $\mathcal { L } _ { \mathrm { h o t } }$   
26: end procedure   
27: procedure $\mathrm { C O N F I R M } ( \mathcal { H } , C , R , \mathcal { D } _ { \mathrm { b a s e } } ^ { ( 1 : L ) } )$   
28: for each log $l \in \mathcal H$ do   
29: $C ^ { \prime } \gets \bar { \mathrm { A B L A T E } } ( C , \{ l \} )$   
30: $\mathcal { D } ^ { \prime ( 1 : L ) } \gets \mathrm { D I A G R A M S } ( C ^ { \prime } , R , d )$   
31: $\Delta _ { l } \gets \mathrm { T A D } ( \mathcal { D } ^ { \prime } , \mathcal { D } _ { \mathrm { b a s e } } )$   
32: end for   
33: return FLAGSPIKES(H, ∆)   
34: end procedure   
35: procedure FLAGSPIKES(X, ∆)   
36: Sort ∆ $\Delta$ descending with order $\sigma$   
37: gaps $[ i ]  \Delta _ { \sigma [ i ] } - \Delta _ { \sigma [ i + 1 ] }$   
38: c ← arg max<sub>i</sub> gaps[i]   
39: return $\{ \mathcal { X } _ { \sigma [ i ] } : i \le \overbar { c } \wedge \mathrm { g a p s } [ c ] > 0 \}$   
40: end procedure

Topological cost. For each forward pass and each layer $\ell ,$ TAD builds a point cloud $X _ { \ell } \in \mathbb { R } ^ { T \times d }$ from the response-token hidden states and computes the persistence diagram of its Vietoris–Rips filtration. Forming the pairwise distance matrix costs $O ( T ^ { 2 } d )$ . When restricted to $H _ { 0 }$ , only the 1-skeleton is required: the $O ( T ^ { 2 } )$ edges are sorted in $O ( T ^ { 2 } \log T )$ time. The per-pass cost is therefore $O \big ( L T ^ { 2 } ( d + \log T ) \big )$ , i.e. quadratic in the number of response tokens and linear in network depth. Extending to $H _ { 1 }$ is substantially more expensive, as it needs 2-skeleton, which may contain $O ( T ^ { 3 } )$ triangles, and can have cubic complexity in the number of simplices. Overall, the endto-end cost of TAD is thus $O { \big ( } M \cdot L \cdot T ^ { 2 } ( d + \log T ) { \big ) }$ in the $H _ { 0 }$ homology, with $M = O ( { \sqrt { N } } )$ under adaptive screening, and it is the configuration we adopt throughout our experiments.

## VI. RESULTS

In this section, we will introduce our dataset, experimental design, and TAD results for incident log analysis.

## A. Dataset Curation & Attack Scenario

We curated the dataset from logs collected during a real cyberattack executed in the Game of Active Directory (GOAD) environment [37] and monitored using the Wazuh security framework [38]. GOAD is an intentionally vulnerable Windows Active Directory environment designed for penetration testing and security training. The attack was conducted in October 2025 and during this timeframe, we identified 20 log entries as the ground-truth attack evidence. These entries correspond to events directly associated with the adversarial activities performed during the attack, while the remaining logs represent benign activity, or events not directly attributable to the attack.

Cybersecurity logs often contain long sequences of nearly identical events, such as brute-force attempts that differ only in timestamps, ports, or event identifiers. To reduce this redundancy, we applied Gestalt Pattern Matching to group consecutive logs $\mathrm { { w i t h } ~ \geq ~ 9 0 \% }$ sequence similarity and represented each group with a single entry. We added a how-many-consecutive metadata field to record the number of original events represented by each compressed log. This preserves event frequency while preventing repetitive activity from dominating the analysis. After compression, the dataset contains 587 log entries. Therefore, the compression reduced the log volume while preserving the semantics and frequency of the log events. In Table I and Table II, we demonstrate the details of our dataset.

Attack Progression Summary: The attack consists of a multi-stage intrusion involving pass-the-hash authentication, lateral movement, remote execution, tool deployment, and defense evasion. On the GOAD environment [37], the attacker operated from NPC-PETYERBAELI host and compromised the robb.stark account to gain access to several systems.

1) Credential Validation and Initial Access: On October 10, at approximately 18:51, a logon attempt for the robb.stark account failed on NPC-PETYERBAELI.

TABLE I: Dataset overview. Each window is a slice of Windows event logs surrounding the attack logs. #GT refers to the number of ground-truth attack-relevant logs.
<table><tr><td>W</td><td>Date</td><td>Start (UTC)</td><td>End (UTC)</td><td>#Logs</td><td>#GT</td></tr><tr><td>1</td><td>2025-10-10</td><td>18:51:21</td><td>18:52:16</td><td>71</td><td>3</td></tr><tr><td>2</td><td>2025-10-10</td><td>18:56:59</td><td>18:57:45</td><td>48</td><td>1</td></tr><tr><td>3</td><td>2025-10-10</td><td>19:02:21</td><td>19:03:40</td><td>48</td><td>2</td></tr><tr><td>4</td><td>2025-10-10</td><td>19:09:46</td><td>19:11:17</td><td>50</td><td>4</td></tr><tr><td>5</td><td>2025-10-10</td><td>19:54:49</td><td>19:55:47</td><td>46</td><td>3</td></tr><tr><td>6</td><td>2025-10-10</td><td>20:00:00</td><td>20:01:41</td><td>59</td><td>3</td></tr><tr><td>7</td><td>2025-10-11</td><td>03:35:51</td><td>03:37:51</td><td>81</td><td>1</td></tr><tr><td>8</td><td>2025-10-11</td><td>06:56:27</td><td>06:57:05</td><td>47</td><td>1</td></tr><tr><td>9</td><td>2025-10-11</td><td>18:56:41</td><td>18:57:26</td><td>50</td><td>1</td></tr><tr><td>10</td><td>2025-10-11</td><td>20:48:45</td><td>20:48:46</td><td>87</td><td>1</td></tr><tr><td colspan="4">Total (10 windows)</td><td>587</td><td>20</td></tr></table>

TABLE II: Ground-truth attack events categories of the dataset.
<table><tr><td>Technique</td><td>Count</td></tr><tr><td>Pass-the-Hash (NTLM remote logon)</td><td>9</td></tr><tr><td>Registry modification (BAM, tool execution)</td><td>4</td></tr><tr><td>Malicious service creation (PowerShell payload)</td><td>2</td></tr><tr><td>PSEXESVC service installation</td><td>2</td></tr><tr><td>Credential brute force (logon failure)</td><td>1</td></tr><tr><td>Interactive logon, compromised credentials</td><td>1</td></tr><tr><td>Registry value integrity change</td><td>1</td></tr><tr><td>Total</td><td>20</td></tr></table>

This was later followed by a successful elevated interactive logon on the same host. Within seconds, the attacker used the same account in a successful pass-the-hash authentication against winterfell. This sequence shows that the attacker had obtained the account’s NTLM credential.

2) Lateral Movement and Service Deployment: Between approximately 18:57-19:10, the attacker used pass-the-hash authentication to access winterfell, kingslanding, and meereen. Several authentication attempts used consecutive source ports, showing that the activity was carried out through an automated sequence. After successful NTLM authentication, the attacker deployed PowerShell-based services with AMSI-bypass functionality on winterfell. These services allowed remote command execution.

3) Remote Execution: Between approximately 19:55 and 20:01, the attacker deployed the PSEXESVC service and continued pass-the-hash authentication from NPC-PETYERBAELI.

4) Post-Exploitation Tooling and Defense Evasion: On October 11, the attacker deployed and executed additional post-exploitation tools across several systems. COLIncrease.exe was executed on castelblack under the robb.stark account. A separate copy of the executable was later executed again on NPC-PETYERBAELI, as shown by a change in the BAM entry’s checksum. The attacker also executed PsExec64.exe on

NPC-PETYERBAELI to support further remote execution. Finally, DefenderRemover.exe was executed on vdi-samwell-tar to disable Windows Defender and reduce endpoint security protections.

Overall, the sequence shows that the attacker systematically expanded access across the environment and weakened security controls to support continued malicious activity.

## B. Experimental Design

In real-world scenarios, where thousands or millions of log entries may be generated, providing all available logs directly to an LLM is often impractical due to contextwindow limitations, increased inference cost, and the presence of irrelevant information that can degrade response quality. A targeted investigation allows the model to focus on the subset of evidence most relevant to the analyst’s hypothesis while reducing noise and computational overhead. Therefore, our experimental scenario is motivated by a targeted threatinvestigation workflow. We assume that a security analyst has already developed a suspicion that a particular user account may have been compromised. The analyst subsequently queries the system using a prompt of the following form:

Examine the logs recorded between [start time] and [end time] on [date]. Is there evidence of an attack in these logs?

This formulation bounds the investigation by date and time, reflecting a realistic post-alert workflow. Rather than performing unrestricted threat hunting, the model assesses whether the supplied logs support the analyst’s hypothesis.

Furthermore, directly comparing natural-language responses from multiple LLMs introduces confounding factors such as differences in wording, detail, structure, confidence, and event interpretation. To improve comparability, we assume that a canonical response derived from the ground-truth attack logs is already available and hold this response constant across all evaluated models. To curate these canonical responses, we used Claude-Opus-4.5 model to generate high-quality outputs that closely match with the ground-truth attack logs. This design constitutes a controlled experimental abstraction. In a real-world deployment, each model would generate its own investigation report, and the resulting conclusions could vary according to the model’s training data, reasoning capabilities, and contextual understanding. Therefore, the use of a canonical response improves experimental comparability, but does not evaluate the quality of the model-specific response generation.

To evaluate TAD on different difficulty levels, we considered three scenarios: “Direct”, “Regular”, and “Indirect”. “Direct” is for when the LLM significantly copies the words from the source logs. The “Regular” case is for when an LLM copies some keywords but gives a more general interpretation. The “Indirect” case is for when the LLM almost do not use any critical word that can map the response to the specific log(s) via a keyword-search. The reason for this is that we will show TAD is not based on finding exact keywords and putting higher attention to those. Instead, it is based on the overall interpretation of each log (as a segment).

Furthermore, we compare TAD against multiple baselines: LLM-as-a-judge, in-line citation, similarity-based measures (Cosine Similarity and ROUGE-L), and a token-level attribution technique (LEA). These baselines represent commonly used approaches to assess the provenance of LLM-generated responses. For the LLM-as-a-judge and in-line citation baselines, we use greedy-decoding to get deterministic and reproducible responses. Appendix C shows the engineered prompts for these methods. For token-level attribution, we adopt our previously proposed LEA metric [5], as it is based on linear dependency of input embeddings and is computationally efficient. Note that we do not consider other token-level attribution methods, such as SHAP, due to their high computational complexity. Finally, for the cosine embedding model, we used the Alibaba-NLP/gte-modernbert-base model, as it is well-suited for long-context semantic search (8,192 tokens).

The LLMs used for analysis are: Qwen3-4B, Gemma3-4B, Qwen2.5-7B, and Granite4.1-8B. The choice of these models are due to their long context window (a minimum of 128K context window) to handle large security logs and the ability to follow-instructions well. It is worth noting that we conducted our experiments on a system equipped with two Intel Xeon E5-2650 CPUs, 256 GB of RAM, and a NVIDIA Tesla L40S GPU. The code & data is available at: https://github.com/RezzFayyazi/TAD.

## C. TAD for Incident Log Analysis

As discussed in the previous section, we compare TAD with different baselines under three settings, direct, regular, and indirect, which differ in how openly the response refers to the logs. All methods use the same selection rule: we sort the candidates by score, find the largest gap between consecutive scores, and return everything above it. We do this because an analyst does not know in advance how many logs influenced a response. A fixed top-k cutoff would either cut off real contributors or add noise, whereas the gap rule lets the scores set their own threshold.

Table III demonstrates the results. As can be seen, TAD obtains the best accuracy and F1-score in almost all three settings, although the margin differs. On average, for the “Direct”, “Regular”, and “Indirect” cases, TAD outperforms the closest baseline by 1.60%, 3.11%, and 6.63% in accuracy, and by 7.35% 15.77%, and 17.63% margin in F1-score, respectively. Note that the margin ranges from 1.60% to 6.63% in accuracy, and from 7.35% to 17.63% in F1-score, with the “Indirect” case exhibiting the widest margin. This is because other baselines perform better when there are high-token overlaps from the logs and the response, and perform poorly when the response contains little-to-none shared tokens with the logs. It is worth noting that in Appendix A, we provide TAD results based on H1 homology, and in Appendix B, we provide a qualitative example along with some specific examples on how TAD traces a log that contributed the most to the LLM response.

TABLE III: Performance comparison of TAD (H0 homology) and other attribution tracing methods across four LLMs. The Direct, Regular, and Indirect cases point to the difficulty-level of the response with respect to the ground-truth logs.
<table><tr><td rowspan="2">Case</td><td rowspan="2">Method</td><td colspan="2">Qwen3-4B</td><td colspan="2">Gemma-3-4B</td><td colspan="2">Qwen2.5-7B</td><td colspan="2">Granite-4.1-8B</td></tr><tr><td>Accuracy</td><td>F1-score</td><td>Accuracy</td><td>F1-score</td><td>Accuracy</td><td>F1-score</td><td>Accuracy</td><td>F1-score</td></tr><tr><td rowspan="5">Direct</td><td>LLM-as-judge</td><td>0.9693</td><td>0.6897</td><td>0.9046</td><td>0.2821</td><td>0.9796</td><td>0.7500</td><td>0.9625</td><td>0.6333</td></tr><tr><td>In-line citation</td><td>0.9710</td><td>0.5143</td><td>0.9693</td><td>0.4000</td><td>0.9455</td><td>0.4074</td><td>0.9779</td><td>0.6829</td></tr><tr><td>Cosine similarity</td><td>0.8058</td><td>0.2192</td><td>0.8058</td><td>0.2192</td><td>0.8058</td><td>0.2192</td><td>0.8058</td><td>0.2192</td></tr><tr><td>ROUGE-L</td><td>0.7990</td><td>0.1690</td><td>0.7990</td><td>0.1690</td><td>0.7990</td><td>0.1690</td><td>0.7990</td><td>0.1690</td></tr><tr><td>LEA</td><td>0.9199</td><td>0.4051</td><td>0.9199</td><td>0.4051</td><td>0.9199</td><td>0.4051</td><td>0.8535</td><td>0.2712</td></tr><tr><td></td><td>TAD (ours)</td><td>0.9847</td><td>0.7097</td><td>0.9779</td><td>0.6061</td><td>0.9830</td><td>0.6667</td><td>0.9830</td><td>0.6667</td></tr><tr><td rowspan="5"></td><td>LLM-as-judge</td><td>0.8790</td><td>0.3604</td><td>0.7274</td><td>0.1304</td><td>0.9267</td><td>0.4416</td><td>0.9659</td><td>0.6296</td></tr><tr><td>In-line citation</td><td>0.9727</td><td>0.6000</td><td>0.9063</td><td>0.2466</td><td>0.9489</td><td>0.3478</td><td>0.9659</td><td>0.6296</td></tr><tr><td>Cosine similarity</td><td>0.9131</td><td>0.3544</td><td>0.9131</td><td>0.3544</td><td>0.9131</td><td>0.3544</td><td>0.9131</td><td>0.3544</td></tr><tr><td>ROUGE-L</td><td>0.6968</td><td>0.1010</td><td>0.6968</td><td>0.1010</td><td>0.6968</td><td>0.1010</td><td>0.6968</td><td>0.1010</td></tr><tr><td>LEA</td><td>0.8007</td><td>0.1583</td><td>0.8262</td><td>0.1774</td><td>0.8007</td><td>0.1583</td><td>0.7104</td><td>0.1414</td></tr><tr><td></td><td>TAD (ours)</td><td>0.9847</td><td>0.7097</td><td>0.9727</td><td>0.5000</td><td>0.9796</td><td>0.6000</td><td>0.9813</td><td>0.6452</td></tr><tr><td rowspan="6">Indirect</td><td>LLM-as-judge</td><td>0.7819</td><td>0.2099</td><td>0.7172</td><td>0.1170</td><td>0.8756</td><td>0.2626</td><td>0.6968</td><td>0.1275</td></tr><tr><td>In-line citation</td><td>0.9131</td><td>0.3377</td><td>0.8876</td><td>0.1951</td><td>0.9216</td><td>0.2333</td><td>0.8790</td><td>0.2198</td></tr><tr><td>Cosine similarity</td><td>0.6048</td><td>0.1008</td><td>0.6048</td><td>0.1008</td><td>0.6048</td><td>0.1008</td><td>0.6048</td><td>0.1008</td></tr><tr><td>ROUGE-L</td><td>0.6542</td><td>0.1057</td><td>0.6542</td><td>0.1057</td><td>0.6542</td><td>0.1057</td><td>0.6542</td><td>0.1057</td></tr><tr><td>LEA</td><td>0.8041</td><td>0.0171</td><td>0.7836</td><td>0.0155</td><td>0.8041</td><td>0.0171</td><td>0.8041</td><td>0.0171</td></tr><tr><td>TAD (ours)</td><td>0.9796</td><td>0.6000</td><td>0.9438</td><td>0.1538</td><td>0.9761</td><td>0.5625</td><td>0.9659</td><td>0.3750</td></tr></table>

Moreover, based on our observation, the baselines indicated a clear pattern: Every baseline reaches high recall but much lower precision: it finds the influential logs, but only by returning a large set in which they sit among many false positives. TAD does this differently by reaching high Precision for the “Direct”, “Regular”, and “Indirect” cases with 94.23%, 86.90%, and 57.70% respectively. On average, for the “Direct”, “Regular”, and “Indirect” cases, TAD outperforms the closest baseline by 39.47%, 47.46%, and 40.88% margin in Precision, respectively. This matters in practice, as in the real-world, an analyst has limited time and a pool of thousands of logs. Therefore, a method that returns so many candidates (i.e., high recall) to reveal a few real ones, restores the manual work that automatic attribution is meant to remove. In addition, by looking at the results, one might ask: “Why does the quality of the F1 score differ for each LLM?”. The reason is the learned representation ofeach LLM individually. The learned representation matters on how each LLM connect tokens as a segment in the embedding space, as discussed by Rottach et al. [22]. This means that if an LLM learned about cybersecurity more than the other, it naturally connects tokens in a more complex way. Therefore, an LLM like Gemma3-4B performing worse than other models in our experiments, demonstrate that this model has not properly learned a mapping between security logs and the response.

Furthermore, the selected baselines perform worse for different reasons. First, LLM-based self-assessment is sensitive to prompt design, meaning that different engineered prompts can give different attribution judgments. This variability arises because the decision comes from the model’s own reasoning over a follow-up prompt, rather than TAD’s mathematically traceable internal property of the original context (that caused the LLM generation). Second, similarity and token-level methods fail because incident logs share highly overlapping tokens, causing the log that shaped the response almost identical to its near-duplicates under cosine similarity. TAD avoids both problems by measuring what the model does with the context (for a generated response) rather than how the context is similar to the response. Removing an influential log causes a large, clearly separated spike in the distance between layerwise persistence diagrams. Because the response is fixed and the computation is deterministic, the result is reproducible, traceable, and reflects how the model maps context to response. Future work could investigate the importance of individual layers on the generation and explore scalable variants of the Vietoris-Rips construction. We see this as a step toward transparent and verifiable attribution that allows analysts to inspect and validate the evidence, rather than simply trust the model’s output.

## VII. CONCLUSION

We introduced Topological Attribution Distance (TAD) to capture the global geometric shape of an LLM output and its changes against its retrieved context. In other words, we show that when the embedding of a retrieved context (e.g., security logs) drastically changes the geometry of the model’s response, this serves as robust verification of evidence, and can be traced. Therefore, we designed TAD powered by segmentlevel ablation attribution to investigate incident logs of an actual cyberattack, and we demonstrated how TAD finds the most attributed logs to the LLM output adaptively. TAD can provide an explainable and trustworthy tracing based on each

LLM’s hidden state to understand how different RAG segments geometrically influence the model generation, and provide evidence verification in cybersecurity operations and Agentic-AI workflows.

## ACKNOWLEDGMENTS

This material is based upon work supported by the National Science Foundation under Grant No. 2344237 and No. 2502341. The authors gratefully acknowledge Dr. Justin Pelletier and Mr. Forrest Fuqua for their valuable contribution to the curation of the dataset.

## AI DISCLOSURE

We used ChatGPT to assist with sentence-level editing and grammar checking. The tool also contributed to the aesthetic refinement of Fig. 1 and Fig. 3, and prompt-boxes. The authors independently reviewed and verified the accuracy, originality, and integrity of all content, including the cited references.

## REFERENCES

[1] W. Zhang and J. Zhang, “Hallucination Mitigation for Retrieval-Augmented Large Language Models: A Review,” Mathematics, vol. 13, no. 5, p. 856, 2025.

[2] A. Asai, Z. Wu, Y. Wang, A. Sil, and H. Hajishirzi, “Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection,” The Twelfth International Conference on Learning Representations, 2023.

[3] S. Es, J. James, L. E. Anke, and S. Schockaert, “RAGAs: Automated evaluation of retrieval augmented generation,” in Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics: System Demonstrations, 2024, pp. 150–158.

[4] J. Saad-Falcon, O. Khattab, C. Potts, and M. Zaharia, “ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems,” in Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), 2024, pp. 338–354.

[5] R. Fayyazi, M. Zuzak, and S. J. Yang, “LLM Embedding-based Attribution (LEA): Quantifying Source Contributions to Generative Model’s Response for Vulnerability Analysis,” arXiv preprint arXiv:2506.12100, 2025.

[6] H. Hu, C. He, X. Xie, and Q. Zhang, “LRP4RAG: Detecting Hallucinations in Retrieval-Augmented Generation Via Layer-Wise Relevance Propagation,” Available at SSRN 5199254.

[7] L. M. Paes, D. Wei, H. J. Do, H. Strobelt, R. Luss, A. Dhurandhar, M. Nagireddy, K. N. Ramamurthy, P. Sattigeri, W. Geyer et al., “Multilevel explanations for generative language models,” in Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 2025, pp. 32 291–32 317.

[8] F. Chazal and B. Michel, “An Introduction to Topological Data Analysis: Fundamental and Practical Aspects for Data Scientists,” Frontiers in Artificial Intelligence, vol. 4, p. 667963, 2021.

[9] J. R. Munkres, S. G. Krantz, and H. R. Parks, Elements of Algebraic Topology. CRC press, 2025.

[10] D. Anick, “The Computation of Rational Homotopy Groups is #P-Hard,” Lecture Notes in Pure and Applied Mathematics, vol. 114, pp. 1–56, 01 1989.

[11] E. Paluzo-Hidalgo, “Latent Space Topology Evolution in Multilayer Perceptrons,” arXiv preprint arXiv:2506.01569, 2025.

[12] Y. Harun, K. Lee, J. Gallardo, G. Krishnan, and C. Kanan, “What Variables Affect Out-of-Distribution Generalization in Pretrained Models?” Advances in Neural Information Processing Systems, vol. 37, pp. 56 479– 56 525, 2024.

[13] Y. Gardinazzi, K. Viswanathan, G. Panerai, A. Ansuini, A. Cazzaniga, and M. Biagetti, “Persistent Topological Features in Large Language Models,” in International Conference on Machine Learning. PMLR, 2025, pp. 18 811–18 830.

[14] A. Fay, I. Garc´ıa-Redondo, Q. Wang, H. Dubossarsky, and A. Monod, “The Shape of Adversarial Influence: Characterizing LLM Latent Spaces with Persistent Homology,” in International Conference on Learning Representations, vol. 2026, 2026, pp. 146 129–146 175.

[15] D. Li, Z. Sun, X. Hu, Z. Liu, Z. Chen, B. Hu, A. Wu, and M. Zhang, “A Survey of Large Language Models Attribution,” arXiv preprint arXiv:2311.03731, 2023.

[16] M. T. Ribeiro, S. Singh, and C. Guestrin, ““Why Should I Trust You?” Explaining the Predictions of Any Classifier,” in Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 2016, pp. 1135–1144.

[17] S. M. Lundberg and S.-I. Lee, “A Unified Approach to Interpreting Model Predictions,” Advances in Neural Information Processing Systems, vol. 30, 2017.

[18] S. Bach, A. Binder, G. Montavon, F. Klauschen, K.-R. Muller, and ¨ W. Samek, “On Pixel-Wise Explanations for Non-Linear Classifier Decisions by Layer-Wise Relevance Propagation,” PloS one, vol. 10, no. 7, p. e0130140, 2015.

[19] R. Achtibat, S. M. V. Hatefi, M. Dreyer, A. Jain, T. Wiegand, S. Lapuschkin, and W. Samek, “AttnLRP: Attention-Aware Layer-Wise Relevance Propagation for Transformers,” in Proceedings of the 41st International Conference on Machine Learning, 2024, pp. 135–168.

[20] P. Sekuloski, D. Kitanovski, I. Goshev, K. Mishev, M. S. Misheva, and V. D. Ristovska, “Exploring the Potential of Topological Data Analysis for Explainable Large Language Models: A Scoping Review,” Mathematics, vol. 14, no. 2, p. 378, 2026.

[21] O. Draganov and S. Skiena, “The Shape of Word Embeddings: Quantifying Non-Isometry with Topological Data Analysis,” in Findings of the Association for Computational Linguistics: EMNLP 2024, 2024, pp. 12 080–12 099.

[22] F. Rottach, W. Rudman, B. Rieck, H. Scells, and C. Eickhoff, “From Topology to Retrieval: Decoding Embedding Spaces with Unified Signatures,” arXiv preprint arXiv:2511.22150, 2025.

[23] S. Fitz, P. Romero, and J. J. Schneider, “Hidden Holes: Topological Aspects of Language Models,” arXiv preprint arXiv:2406.05798, 2024.

[24] G. Carlsson, “Topology and Data,” Bulletin of the American Mathematical Society, vol. 46, no. 2, pp. 255–308, 2009.

[25] Edelsbrunner, Letscher, and Zomorodian, “Topological Persistence and Simplification,” Discrete & Computational Geometry, vol. 28, no. 4, pp. 511–533, 2002.

[26] J.-C. Hausmann, “On the Vietoris–Rips Complexes and a Cohomology Theory,” in Prospects in Topology: Proceedings ofa Conference in Honor of William Browder, no. 138. Princeton University Press, 1995, p. 175.

[27] D. Cohen-Steiner, H. Edelsbrunner, and J. Harer, “Stability of Persistence Diagrams,” in Proceedings of the Twenty-First Annual Symposium on Computational Geometry, 2005, pp. 263–271.

[28] A. Zomorodian and G. Carlsson, “Computing Persistent Homology,” in Proceedings of the Twentieth Annual Symposium on Computational Geometry, 2004, pp. 347–356.

[29] E. Munch, “Applications of Persistent Homology to Time Varying Systems,” Ph.D. dissertation, Duke University, 2013.

[30] R. Ghrist, “Barcodes: the Persistent Topology of Data,” Bulletin of the American Mathematical Society, vol. 45, no. 1, pp. 61–75, 2008.

[31] G. Carlsson, A. Zomorodian, A. Collins, and L. Guibas, “Persistence Barcodes for Shapes,” in Proceedings of the 2004 Eurographics/ACM SIGGRAPH Symposium on Geometry Processing, 2004, pp. 124–135.

[32] F. Chazal, D. Cohen-Steiner, M. Glisse, L. J. Guibas, and S. Y. Oudot, “Proximity of Persistence Modules and Their Diagrams,” in Proceedings of the twenty-fifth annual symposium on Computational geometry, 2009, pp. 237–246.

[33] Y. Mileyko, S. Mukherjee, and J. Harer, “Probability Measures on the Space of Persistence Diagrams,” Inverse Problems, vol. 27, no. 12, p. 124007, 2011.

[34] D. Cohen-Steiner, H. Edelsbrunner, J. Harer, and Y. Mileyko, “Lipschitz Functions Have L p-Stable Persistence,” Foundations of Computational Mathematics, vol. 10, no. 2, pp. 127–139, 2010.

[35] A. Arulandu, D. Gottschalk, T. Payne, A. Richardson, and T. Weighill, “Through the Grapevine: Vineyard Distance as a Measure of Topological Dissimilarity,” arXiv preprint arXiv:2510.24472, 2025.

[36] L. Yang, “Persistent Homology for Distribution Drift Detection in LLM Embedding Streams,” in The ICML 2026 Workshop on Hypothesis Testing.

[37] Orange-Cyberdefense, “Game of Active Directory (GOAD),” 2025. [Online]. Available: https://github.com/Orange-Cyberdefense/GOAD

[38] Wazuh, “Wazuh: The Open Source Security Platform),” 2025. [Online]. Available: https://wazuh.com/

## APPENDIX

## A. TAD results based on H1 homology

In Table IV, we show the TAD values based on H1 homology. As can be seen, the results are generally competitive in both settings, with H1 being higher in F1-score for the “Indirect” case for all four models. In addition, H1 results are also slightly higher for larger models, which could indicate that H1 can capture more abstract or higher-order representations as model capacity increases. However, due to H1 being computationally more expensive (cubic) than H0 (quadratic), we used H0 as the default setting for TAD.

## B. TAD Specific Examples

To demonstrate in more detail on how TAD actually attributes the logs to the response, we provided a detailed qualitative example in Figure 3. As can be seen, TAD correctly attributes log-35 with the highest contribution into the LLM’s response, with a total Wasserstein of 1951.13 and containing the largest gap (1162.72) compared to the other logs.

In addition, we provided three specific examples for the “Direct”, “Regular”, and “Indirect” cases. The results show how TAD correctly attributes all cases, and interestingly, for the indirect case, where there is no keyword similarity, TAD still managed to attribute to the correct log.

## Direct Case – Gemma3-4B – TAD

## LLM Output

At 03:36:16.781, agent castelblack (192.168.56.22) logged “Registry Value Entry Added to the System.” A registry value associated with COLIncrease.exe, located on user robb.stark’s Desktop, was added under the Windows bam\State\UserSettings registry path in scheduled mode.

## TAD’s Traced Log

log\_id=iLZX0ZkBDpTsYhzSRWvf   
timestamp=2025-10-11T03:36:16.781Z   
agent\_name=castelblack   
agent\_ip=192.168.56.22   
rule\_description="Registry Value Entry Added   
to the System"   
target\_user=null   
target\_domain=null   
full\_log="Registry Value ’[x32]   
HKEY\_LOCAL\_MACHINE\System\   
CurrentControlSet\Services\bam\State\   
UserSettings\S-1-5-21-3733740424-\\   
1936739333-3027215852-1115\\Device\   
HarddiskVolume1\Users\robb.stark\Desktop\   
COLIncrease.exe’ added Mode: scheduled"

## Indirect Case – Qwen3-4B – TAD

## LLM Output

An execution ledger on this box shows that a well-known utility for launching programs on other boxes was run from a fetch folder in a staffprofile. Storing that kit in a personal folder rather than an approved software area, then running it, points to an outsider preparing to reach further boxes.

## TAD’s Traced Log

log\_id=B7YP0pkBDpTsYhzSDo05   
timestamp=2025-10-11T06:57:04.447Z   
agent\_name=npc-petyerbaeli   
agent\_ip=192.168.56.154   
rule\_description="Registry Value Entry Added   
to the System"   
target\_user=null   
target\_domain=null   
full\_log="Registry Value ’[x32]   
HKEY\_LOCAL\_MACHINE\System\   
CurrentControlSet\Services\bam\State\   
UserSettings\S   
-1-5-21-3733740424-1936739333-3027215852   
-1115\\Device\HarddiskVolume1\Users\robb.   
stark\Downloads\PsExec64.exe’ ..."

## Regular Case – Granite4.1-8B – TAD

## LLM Output

WINTERFELL accepts a successful remote logon for robb.stark at 18:57, authenticated with NTLM from NPC-PETYERBAELI. Presenting a hash rather than a typed password, and doing so against a domain controller from an ordinary workstation, is the signature of a replayed credential. This is unauthorized lateral movement rather than legitimate remote administration.

## TAD’s Traced Log

```ini
log_id=FbV8z5kBDpTsYhzSINoU
timestamp=2025-10-10T18:57:20.490Z
agent_name=winterfell
agent_ip=192.168.56.11
rule_description="Successful Remote Logon
Detected - User:\robb.stark - NTLM
authentication, possible pass-the-hash
attack... Verify that NPC-PETYERBAELI is
allowed to perform RDP connections"
target_user=robb.stark
target_domain=NORTH
system_message="\"An account was successfully
logged on. ... Security ID: S
-1-5-21-37337...1115 Account Name: robb.
stark Account Domain: NORTH ...
Workstation Name: NPC-PETYERBAELI Source
Network Address: 192.168.56.154 ...
Package Name (NTLM only): NTLM V2 Key
Length....\""
```

TABLE IV: TAD comparison based on H0 & H1 homology.
<table><tr><td rowspan="2">Case</td><td rowspan="2">Method</td><td colspan="2">Qwen3-4B</td><td colspan="2">Gemma-3-4B</td><td colspan="2">Qwen2.5-7B</td><td colspan="2">Granite-4.1-8B</td></tr><tr><td>Accuracy</td><td>F1-score</td><td>Accuracy</td><td>F1-score</td><td>Accuracy</td><td>F1-score</td><td>Accuracy</td><td>F1-score</td></tr><tr><td>Direct</td><td>TAD (H0)</td><td>0.9847</td><td>0.7097</td><td>0.9779</td><td>0.6061</td><td>0.9830</td><td>0.6667</td><td>0.9830</td><td>0.6667</td></tr><tr><td rowspan="2">Regular</td><td>TAD (H1)</td><td>0.9847</td><td>0.7273</td><td>0.9710</td><td>0.5405</td><td>0.9830</td><td>0.7059</td><td>0.9830</td><td>0.7222</td></tr><tr><td>TAD (H0)</td><td>0.9847</td><td>0.7097</td><td>0.9727</td><td>0.5000</td><td>0.9796</td><td>0.6000</td><td>0.9813</td><td>0.6452</td></tr><tr><td rowspan="2">Indirect</td><td>TAD (H1)</td><td>0.9830</td><td>0.6667</td><td>0.9438</td><td>0.2667</td><td>0.9830</td><td>0.6667</td><td>0.9847</td><td>0.7097</td></tr><tr><td>TAD (H0)</td><td>0.9796</td><td>0.6000</td><td>0.9438</td><td>0.1538</td><td>0.9761</td><td>0.5625</td><td>0.9659</td><td>0.3750</td></tr><tr><td></td><td>TAD (H1)</td><td>0.9813</td><td>0.6452</td><td>0.9438</td><td>0.2326</td><td>0.9761</td><td>0.5882</td><td>0.9608</td><td>0.4103</td></tr></table>

<table><tr><td colspan="6">TAD&#x27;s Qualitative Example</td></tr><tr><td colspan="6">1. Analyst query</td></tr><tr><td colspan="6">Examine logs around 06:56–06:57 on 2025-10-11. Is there evidence of an attack in these logs?</td></tr><tr><td colspan="6">2. Candidate incident logs after the Screen process (refer to Algorithm 1)</td></tr><tr><td colspan="4">Log Event State</td><td>Total-Wasserstein</td><td>Gap</td></tr><tr><td colspan="4">31 BAM registry checksum changed for powershel1.exe. Benign</td><td>682.54</td><td>0.0</td></tr><tr><td colspan="4">32 BAM registry checksum changed for cmd . exe. Benign</td><td>708.30</td><td>25.75</td></tr><tr><td colspan="4">29 BAM registry checksum changed for ApplicationFrameHost.exe. Benign</td><td>710.32</td><td>2.03</td></tr><tr><td colspan="4">34 BAM registry entry added for rundl132.exe. Benign</td><td>749.56</td><td>39.24</td></tr><tr><td colspan="4">30 BAM registry checksum changed for windows. immersivecontrolpanel.</td><td>Benign 782.09</td><td>32.53</td></tr><tr><td colspan="4">33 BAM registry entry added for temporary Un . exe.</td><td>Benign 788.61</td><td>6.52</td></tr><tr><td>35</td><td colspan="4">BAM registry entry added for PsExec64.exe in robb.stark&#x27;s Downloads. Malicious 1951.33</td><td>1162.72</td></tr><tr><td colspan="6">3. LLM response (Granite-4.1-8B) for the regular case</td></tr><tr><td colspan="6">&quot;NPC-PETYERBAELI records a Background Activity Moderator entry at 06:57 showing PsExec64. exe ran from robb. stark&#x27;s Downloads folder. PsExec spawns processes on remote machines, and a copy staged in a download folder rather than an approved install path indicates an operator preparing to execute code on other hosts.&quot;</td></tr><tr><td colspan="6">4. Evidence localization</td></tr><tr><td colspan="6"></td></tr><tr><td>Family Segment-level attribution</td><td>Method TAD (ours)</td><td>Result ✓ Log 35</td><td colspan="3">Operational consequence Uniquely identifies the log mostly influencing the response geometry</td></tr><tr><td colspan="2"></td><td></td><td colspan="3"></td></tr></table>

Fig. 3: TAD qualitative example to demonstrate how Wasserstein distance measures each log’s effect on the response geometry.

## C. LLM Self-Eval Baseline Prompts

Below we provide the prompts used for the LLM-based self-assessment baselines (LLM-as-Judge and in-line Citation). These prompts are designed to evaluate the extent to which the LLM can map the generated responses to the source logs and attribute their outputs to the specific retrieved segments that support them. We used greedy decoding for these cases to get deterministic responses. However, a key limitation of LLMbased judges is their sensitivity to engineering of prompts, as variations in prompt wording or structure can lead to different evaluation outcomes. TAD, on the other hand, does not rely on a secondary set of evaluation prompts to trace the logs and will only look at the response geometry.

## In-line Citation Prompt

Here is your response based on these logs:

"""{response}"""

For every sentence in the response, add inline citations to ALL logs that support that sentence.

\- If multiple logs support a sentence, cite all of them.

\- Do not leave factual claims uncited.

Output format: Add citations with <<logN>> at the end of each sentence.

You are an expert at analyzing cybersecurity incident responses.

Below are security logs, each tagged with <<logN>>...<<logN/>>:

## {formatted logs}

You already examined these logs and produced this response:

"""

{response} I I I

TASK: Identify which specific logs you relied upon to produce this response.

## RULES:

1. Only list logs that directly contributed to the response’s conclusions

2. Output ONLY the log tags that were used (e.g., <<log5>>, <<log12>>)

3. Do not include logs that were present but not referenced in the response

Available log tags: {available tags}

OUTPUT FORMAT:

List the relevant log tags, one per line:

<<logX>>

<<logY>>

RELEVANT LOGS: