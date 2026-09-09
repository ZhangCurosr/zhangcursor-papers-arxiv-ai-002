# SequenceO1: End-to-End Ultra-Long (100K) Sequence Modeling in Recommendation with Low-Rank Caching

Lin Guan<sup>∗</sup>   
guanlin.13@gmail.com   
ByteDance   
Beijing, China   
Jiaqi Huang<sup>∗</sup>   
jiaqihuang.0612@gmail.com   
ByteDance   
Beijing, China

Beichuan Zhang zhangbeichuan.123@bytedance.com ByteDance Beijing, China

Xiangyu Fan xiangyufan.ptr@bytedance.com ByteDance Shanghai, China

Yuhang Qi   
qiyuhang@bytedance.com   
ByteDance   
Hangzhou, Zhejiang, China   
Qiwei Chen<sup>†</sup>   
chenqiwei05@gmail.com   
ByteDance   
Shanghai, China

Jia-Qi Yang<sup>∗</sup> yangjiaqi.yjq@bytedance.com ByteDance Shanghai, China

Hangyu Wang wanghangyu.123@bytedance.com ByteDance Shanghai, China

Haonan Jiang jianghaonan.1004@bytedance.com ByteDance Shanghai, China

Xiaowen Li lixiaowen.911@bytedance.com ByteDance Beijing, China

Xiaolong Zhu zhuxiaolong.auto@bytedance.com ByteDance Beijing, China

Yi Cheng   
chengyi.23@bytedance.com   
ByteDance   
Beijing, China   
Zhishan Zhao<sup>∗</sup>   
zhaozhishan@bytedance.com   
ByteDance   
Beijing, China

Longbin Li mengxinghe@bytedance.com ByteDance Beijing, China

Jinan Ni   
nijinan@bytedance.com   
ByteDance   
Shanghai, China

## Abstract

Ziyao Ren   
renziyao.99@bytedance.com   
ByteDance   
Beijing, China Xuanyuan Luo   
xuanyuanluo@bytedance.com ByteDance   
Hangzhou, Zhejiang, China   
Lele Yu   
yulele@bytedance.com   
ByteDance   
San Jose, CA, USA

Modern short-video recommenders must exploit ultra-long user histories—which can reach hundreds of thousands or even millions of interactions per user—but are constrained by strict latency and training-throughput budgets. At the 100K scale, the bottleneck is systemic, spanning feature storage, communication, and computation in both training and serving. Existing solutions based on truncation, multi-stage retrieval, or length extrapolation either sacrifice end-to-end modeling or retain substantial length-dependent system cost.

We present SequenceO1, an end-to-end framework deployed at full trafic on Douyin at the 100K scale and designed to extend to million-scale histories. The name reflects its cache-hit path, whose cost is �(1) with respect to the raw ultra-long sequence length once the fixed-size sketch is available. At the model level, we propose Sketch Attention (SA), which compresses an ultra-long history into a fixed-size, user-only sketch using learnable prototypes and prototype-wise normalization (each token distributes mass over prototypes). We then perform target-conditioned reasoning at two time scales: STCA over a recent 10K sufix for recency and STCA over the fixed-size sketch for ultra-long signals, followed by lightweight fusion.

At the system level, a training-side local key–value cache reuses user-only sketches across repeated instances of the same user, and the same cacheable state is reused across consecutive serving requests. We further improve eficiency with multi-request user-level batching in training and a fused FlashSA kernel for sketching under ragged batching. Together, these model and system optimizations make end-to-end 100K sequence modeling practical in production and provide a scalable path toward million-scale histories.

## CCS Concepts

• Information systems → Recommender systems.

## Keywords

Recommender systems, long-sequence modeling

## ACM Reference Format:

Lin Guan, Jia-Qi Yang, Zhishan Zhao, Jiaqi Huang, Hangyu Wang, Long bin Li, Beichuan Zhang, Haonan Jiang, Jinan Ni, Xiangyu Fan, Xiaowen Li, Ziyao Ren, Yuhang Qi, Xiaolong Zhu, Xuanyuan Luo, Qiwei Chen, Yi Cheng, and Lele Yu. 2026. SequenceO1: End-to-End Ultra-Long (100K) Sequence Modeling in Recommendation with Low-Rank Caching. In 20th ACM Conference on Recommender Systems (RecSys ’26), September 27-October 02, 2026, Minneapolis, MN, USA. ACM, New York, NY, USA, 20 pages. https: //doi.org/10.1145/3773078.3831864

## 1 Introduction

Short-video recommendation at billion scale depends critically on modeling user behavioral histories. Behavior-sequence modeling has been widely studied in modern recommender systems, from target-conditioned interest modeling to Transformer-based sequential recommendation [14, 19, 38, 51, 52]. On Douyin, a single year of consumption can easily accumulate on the order of 10<sup>5</sup> interactions per user. Such ultra-long histories contain stable preferences, recurring intents, and periodic patterns that are often missed by short windows. However, the fine-ranking stage operates under strict latency and training-throughput constraints, making it dificult to exploit these histories directly in an end-to-end manner [9, 29, 30]. At 100K, this is not merely a per-layer compute problem: embedding and sequence features enlarge training samples and cache footprints, moving them stresses host–device and distributed communication, and processing them increases both training and serving computation.

Production systems therefore often rely on truncation or multistage long-history pipelines that retrieve target-relevant behaviors, compress long histories, or combine both before final ranking [1, 3, 33, 34]. A representative baseline in our stack follows TWIN V2 [37], which uses ofline hierarchical clustering to compress lifelong behaviors and online retrieval with cluster-aware target attention. In our implementation, the clustered history contains about 10K entries, from which a General Search Unit (GSU) retrieves a small target-relevant subset for downstream fine-grained interest modeling. Although efective, this design separates long-history compression and retrieval from the final ranking objective, limiting direct end-to-end optimization over the raw ultra-long history and adding system complexity [5, 6].

A more direct route is end-to-end long-sequence ranking with length extrapolation: train on shorter histories and serve on longer ones. Meta’s HSTU uses stochastic length sampling to reduce train ing cost, while Douyin’s STCA adopts a train-sparsely/infer-densely regimen [12, 13, 48]. The reported extrapolation ranges nevertheless remain coupled to training length: ULTRA-HSTU reports an average training length of about 4,400 for inference at 16,384, while STCA studies average training lengths ofroughly 2–2.5K for serving at 10K—about four to five times longer in both cases. Reaching 100K by extrapolation would therefore still require substantial training contexts, retaining high feature-sample storage, communication, and training-compute costs. Nor does extrapolation remove the serving bottleneck: vanilla HSTU’s self-attention is �(�<sup>2</sup>), whereas STCA reduces per-layer complexity to �(�) but still incurs inference cost that grows linearly with the raw history length. Moreover, STCA’s target-conditioned cross attention is query-dependent and leaves little reusable computation across targets. Thus, the 100K regime requires coordinated model and system design that avoids repeatedly storing, moving, and processing the raw ultra-long history, rather than optimizing per-layer computation alone [40, 43].

![](images/fea25987ec0b78456fb70e4568c91e28418d7e911b162ea2633b52004e1931e2.jpg)

![](images/f44ab7dda4b8f1ea19dbd6ba59bbc72143cd4279536c3c77f734165fdefb9eaf.jpg)

![](images/09872cc0f0a373d1bb0a461e79832a9907d04154a6cda6f507fd492913ccadd0.jpg)  
Figure 1: Compute-eficient scaling to 100K. (a) Training FLOPs of the ultra-long sequence branch for direct STCA and SequenceO1 from 1K to 100K under the observed trainingside MRLB operating point (�=40, $p _ { \mathrm { h i t } } { = } 0 . 5 ) ;$ at 100K, SequenceO1 is 49.9× cheaper than STCA. (b) Inference FLOPs of the same branch under the observed serving-side reuse operating point (�=300, $ { p _ { \mathrm { h i t } } } { = } 0 . 6 )$ ; at 100K, SequenceO1 is 63.9× cheaper than STCA. Common recent-10K and ranker-side costs are omitted from both. (c) Finish AUC improvement over the STCA(512) baseline in the ablation setting versus average length; at 100K truncation (avg. 85K), SequenceO1 retains 83% of directly scaled STCA’s gain (+1.07% vs. +1.29%).

Another route is to decouple ultra-long user modeling from online ranking. Recent multi-stage systems such as LLaTTE [46] and SOLARIS [25] scale recommendation models by transferring asynchronous user representations to online rankers. This reduces online computation, but it also introduces representation bottlenecks, objective mismatch, and freshness constraints between the upstream user model and the final ranking objective [5, 6, 26]. Therefore, a desirable 100K solution should reduce both training and serving cost while preserving an end-to-end optimization path for final ranking.

Our key observation is that the ultra-long portion of user history should be both compressible and reusable. Prior work on eficient attention suggests that long sequences often admit compact representations along the length dimension [40], and fixed-size latent summaries have also been efective for compressing large inputs before downstream reasoning [18, 20]. In our setting, the 100K sequence should be summarized into a compact representation that depends only on user-side signals. If such a fixed-size user representation can be computed once, cached, and reused across repeated training instances, targets, and consecutive requests, then the expensive target-conditioned module no longer needs to operate directly on the raw 100K history [50].

![](images/3afd14c29548877ef4d69bfd6c3c56294b515c70d8392086cbd028f5db3b428f.jpg)  
Figure 2: Overview of SequenceO1. (a) Existing approaches to the storage, communication, and computation bottlenecks of ultra-long sequence modeling—truncation, TWIN V2 two-stage retrieval, and direct STCA scaling—and their respective quality and system trade-ofs. (b) Model architecture: stacked Sketch Attention (SA) compresses the 100K history into a fixed-size sketch, followed by recent-10K and ultra-long sketch branches and downstream MixFormer fusion. (c) Training: MRLB and the local KVCache amortize user-only sketching across grouped requests and targets, with FlashSA executed once per group on a cache miss. (d) Inference: the shared cache reuses fixed-size sketches, while pipeline lift overlaps miss-side FlashSA computation with recall.

Motivated by this observation, we present SequenceO1, an endto-end framework deployed at full trafic on Douyin that scales sequence ranking to the 100K regime. The name “SequenceO1” refers to the cache-hit path: once the fixed-size sketch is available, the raw 100K feature sequence need not be materialized, transferred, or processed for that sample, so the path is �(1) with respect to the raw ultra-long sequence length. As shown in Fig. 2, SequenceO1 combines a compress-then-reason model design with cache-first system amortization across training and inference. At the model level, we introduce Sketch Attention (SA), which compresses the full �=100K history into a fixed-size sketch �e ∈ R<sup>�×�</sup>, with � flexibly ranging from several hundred to several thousand. SA uses learnable prototypes and prototype-wise normalization so that each history token is allocated across prototypes, producing a target-agnostic user sketch. This sketch can be viewed as an implicit low-rank representation along the length dimension, with size independent of �. SequenceO1 then performs target-conditioned reasoning at two time scales: STCA over a recent 10K sufix for recency-sensitive signals, and STCA over the fixed-size sketch for ultra-long signals, followed by lightweight fusion.

At the system level, we first use a training-side local key–value cache (local KVCache) to reuse user-only sketches across repeated instances ofthe same user [50]. The same cacheable sketch is reused across consecutive serving requests; on cache hits, the system skips repeated raw-history feature storage and communication as well as the �-dependent sketch computation, and STCA reasoning is always performed on fixed-size inputs. We further improve eficiency with multi-request user-level batching (MRLB), pipeline lift, and a fused FlashSA kernel inspired by IO-aware attention kernels [10, 11], which avoids materializing large sketching intermediates under ragged batching. Unlike multi-stage representation-transfer systems, SequenceO1 keeps the ultra-long sequence path jointly optimized with the final ranker while making the 100K computation fixed-size, cacheable, and reusable. We compare these scaling routes in §5.5.

Figure 1 summarizes the resulting compute–quality trade-of. Compared with naïvely applying STCA at 100K, SequenceO1 is substantially cheaper under training-side local KVCache and MRLB as well as serving-side cache reuse, while retaining most of the quality gains from directly scaling sequence length. This enables end-to-end sequence ranking to move from the 10K regime to 100K under real production constraints.

We summarize our contributions as follows:

• End-to-end 100K ranking with SA+STCA. We extend end-to-end long-sequence ranking to 100K histories by compressing ultra-long histories with Sketch Attention and applying target-conditioned STCA over both the sketch and a recent 10K sufix.

• A cacheable low-rank sketch for ultra-long histories. We design SA as a target-agnostic compression module that produces a fixed-size user-only sketch. The sketch is diferentiable, length-agnostic, and reusable across targets and requests.

• Cache-first 100K training and serving. Training-side local KVCache and serving-side sketch reuse, complemented by MRLB, pipeline lift, and FlashSA, eliminate repeated �- dependent feature storage, communication, and computation on cache hits.

The remainder of the paper is organized as follows. §2 reviews STCA and long-sequence compression, §3 presents the SequenceO1 model, §4 describes the system design, and §5 reports ofline and online results on Douyin.

## 2 Background and Notations

## 2.1 Problem Setting

We study the final-ranking (“fine-ranking”) stage of a large-scale short-video recommender system on Douyin. Given a request from user $u ,$ the ranker scores candidate videos � under strict latency and cost constraints, using user/request features, item features, and the user interaction history H with supervision signals (e.g., finish, click, skip).

Our focus is on end-to-end modeling of ultra-long user histories in the fine-ranking stage. Following industrial practice, we truncate each user’s history to a maximum length $n _ { \mathrm { m a x } } = 1 0 0 \mathrm { K }$ for both training and serving, while preserving as much behavioral signal as possible within this budget.

## 2.2 Notations

We denote the user interaction history as

$$
\mathcal { H } = \{ ( v _ { i } , a _ { i } ) \} _ { i = 1 } ^ { n } ,\tag{1}
$$

where $v _ { i }$ is the �-th historical item and $a _ { i }$ is the associated action type. Each event is embedded as $\mathbf { x } _ { i } \in \mathbb { R } ^ { d }$ , and the target item � is represented by $\mathbf { x } _ { t } \in \mathbb { R } ^ { d }$

The history embedding matrix is

$$
X = [ \mathbf { x } _ { 1 } , \ldots , \mathbf { x } _ { n } ] ^ { \top } \in \mathbb { R } ^ { n \times d } .\tag{2}
$$

We consider two time scales: an ultra-long history of length � (up to 100K) and a recent sufix of length $L _ { r } = 1 0 \mathrm { K }$ . We denote the recent sufix by

$$
X _ { r } = X _ { n - L _ { r } + 1 : n } \in \mathbb { R } ^ { L _ { r } \times d } ,\tag{3}
$$

when $n \geq L _ { r }$ . In our approach, the ultra-long history is compressed into a fixed-size sketch of length � (typically $k \approx 5 1 2 \ – 1 0 2 4 )$ , which is treated as constant with respect to �.

Throughout the paper, � denotes the raw ultra-long history length, �<sub>�</sub> denotes the recent sufix length, and � denotes the sketch length. For a generic STCA module, we use � to denote its input length; therefore $L = L _ { r }$ for the recent branch and � = � for the sketch branch. We use � for the behavior-embedding and SA width, $d _ { h }$ for the STCA per-head width, and � for the STCA model width after the sketch adapter when applicable.

## 2.3 STCA Recap

A central challenge in long-sequence ranking is the cost of modeling interactions over long histories. Standard self-attention incurs $O ( L ^ { 2 } )$ complexity over a sequence of length �, which is prohibitive at industrial scales.

Stacked Target-to-History Cross Attention (STCA) addresses this issue by removing history–history attention and instead applying stacked single-query cross attention from the target to the history. Given a target query ${ \mathbf { q } } \in \mathbb { R } ^ { w }$ and history embeddings $\boldsymbol { X } \in \mathbb { R } ^ { L \times w }$ one STCA layer computes

$$
{ \mathrm { A t t n } } ( \mathbf { q } , X ) = { \mathrm { s o f t m a x } } \left( { \frac { ( \mathbf { q } W _ { Q } ) ( X W _ { K } ) ^ { \top } } { \sqrt { d _ { h } } } } \right) \cdot ( X W _ { V } ) ,\tag{4}
$$

where $W _ { Q } , W _ { K } , W _ { V } \in \mathbb { R } ^ { w \times d _ { h } }$

With a single query, the per-layer cost scales linearly with $L ,$ enabling end-to-end ranking up to the 10K regime in production. However, at �=100K, directly applying STCA is still too expensive for serving because its dominant target-to-history cross attention remains query-dependent and scales linearly with the raw history length. This leaves little reusable computation across targets and motivates decoupling ultra-long user-side compression from targetconditioned reasoning.

## 2.4 Length-Wise Compression Motivation

Eficient-attention studies suggest that long sequences admit substantial length-wise compression. Linformer motivates a short representation of length � ≪ � through low-rank attention approximation, while Set Transformer and Perceiver use inducing points or fixed latent arrays to summarize large inputs before expressive reasoning [18, 20, 40]. Together, they show that a small set of learned summary tokens can trade sequence length for manageable computation.

Perceiver and poly-encoders typically use latent queries with token-wise softmax to select inputs [17, 18]; SA instead applies prototype-wise softmax so every token is allocated across prototypes, encouraging target-agnostic history coverage. Slot Attention [27] is closest in normalization spirit but targets iterative object-centric learning, whereas SA targets ultra-long ranking under strict latency constraints. Moreover, variable histories up to �=100K make explicit length-dependent projections inflexible, and generic compression does not naturally provide a fixedsize, end-to-end-compatible user-only state reusable across targets and requests. These requirements motivate the length-agnostic, parameter-eficient, and cacheable SA mechanism introduced next.

## 3 Method

We present an end-to-end framework for ultra-long sequence modeling at $n \ = \ 1 0 0 \mathrm { K }$ . The core idea is to compress the ultra-long history into a compact, fixed-size representation and then perform target-conditioned reasoning on this representation. Crucially, the compression depends only on user-side signals, making it reusable across multiple targets and consecutive requests.

## 3.1 Sketch Attention (SA)

Sketch Attention is named after the notion of a sketch: a compact summary that preserves salient information from a much larger object under a fixed memory budget. In our setting, the object is an ultra-long user history, and the sketch is a fixed-size set of userside representation tokens. This is analogous in spirit to streaming sketches such as Count Sketch [4], but SA is learned end-to-end and optimized for downstream ranking rather than for recovering hand-designed statistics.

Given the ultra-long history embeddings $\boldsymbol { X } \in \mathbb { R } ^ { n \times d }$ , we learn � trainable prototypes, flexibly numbering from several hundred to several thousand,

$$
P ^ { ( 0 ) } = \left[ \mathbf { p } _ { 1 } , \ldots , \mathbf { p } _ { k } \right] ^ { \top } \in \mathbb { R } ^ { k \times d } ,\tag{5}
$$

which act as a compact set of summary slots. To enable reuse across targets and requests, the prototypes are shared globally, so the resulting representation depends only on the user history.

We compute prototype–token afinities as

$$
S = \frac { ( P ^ { ( 0 ) } W _ { Q } ) ( X W _ { K } ) ^ { \top } } { \sqrt { d } } \in \mathbb { R } ^ { k \times n } ,\tag{6}
$$

where $W _ { Q } , W _ { K } \in \mathbb { R } ^ { d \times d }$

Prototype-wise normalization. Instead of normalizing over tokens as in standard attention, we normalize over the prototype dimension:

$$
A = \mathrm { s o f t m a x } _ { \mathrm { p r o t o } } ( S ) \in \mathbb { R } ^ { k \times n } , \qquad \sum _ { j = 1 } ^ { k } A _ { j , i } = 1 .\tag{7}
$$

Equivalently,

$$
A _ { j , i } = \frac { \exp ( S _ { j , i } ) } { \sum _ { j ^ { \prime } = 1 } ^ { k } \exp ( S _ { j ^ { \prime } , i } ) } ,\tag{8}
$$

so $A _ { j , i }$ defines a token-to-prototype allocation $p ( j \mid i )$ . Compared with token-wise normalization, this allocation avoids early token selection and encourages coverage of the entire history before target conditioning, which is important when the compressed representation must support diverse targets in a reusable manner.

Sketch construction. We aggregate the history embeddings to obtain a compact sketch:

$$
\widetilde { X } = A X \in \mathbb { R } ^ { k \times d } .\tag{9}
$$

This operation is a weighted aggregation over history tokens rather than token-wise selection. The resulting sketch is fixed-size, target agnostic, and fully diferentiable, making it suitable for caching and end-to-end training.

## 3.2 SA as an Implicit Length-Wise Projection

Equation (9) can be viewed as a data-dependent length-wise projection: the dynamically constructed $A \in \bar { \mathbb { R } } ^ { k \times n }$ compresses the history from � tokens to $k \ll n$ sketch tokens before target-conditioned reasoning. Unlike fixed projections tied to a maximum sequence length [40], � is generated from the input history and learnable prototypes, making SA length-agnostic in parameterization and jointly optimized with the ranking objective. Because this projection uses only user-side signals, its fixed-size, target-agnostic output $\widetilde { X } \in \mathbb { R } ^ { k \times d }$ can be materialized once and reused across targets and requests. This is the operational meaning of low-rank caching: expensive target-conditioned reasoning runs on the cached compact length-wise representation rather than the raw ultra-long history.

## 3.3 Stacked Refinement

A single SA block produces a � × � sketch. To improve capacity, we apply a small number of refinement steps with residual connections. Starting from $\widetilde { X } ^ { ( 0 ) } = P ^ { ( 0 ) }$ , we iteratively compute

$$
S ^ { ( \ell ) } = \frac { ( \widetilde { X } ^ { ( \ell ) } W _ { Q } ^ { ( \ell ) } ) ( X W _ { K } ^ { ( \ell ) } ) ^ { \top } } { \sqrt { d } } ,\tag{10}
$$

$$
A ^ { ( \ell ) } = \mathrm { s o f t m a x } _ { \mathrm { p r o t o } } \Big ( S ^ { ( \ell ) } \Big ) , \qquad \sum _ { j = 1 } ^ { k } A _ { j , i } ^ { ( \ell ) } = 1 ,\tag{11}
$$

$$
\widehat { X } ^ { \left( \ell + 1 \right) } = A ^ { \left( \ell \right) } X ,\tag{12}
$$

$$
\overline { { X } } ^ { ( \ell + 1 ) } = \mathrm { L N } \Big ( \widetilde X ^ { ( \ell ) } + \widehat X ^ { ( \ell + 1 ) } \Big ) ,\tag{13}
$$

$$
\widetilde { X } ^ { ( \ell + 1 ) } = \mathrm { L N } \left( \overline { { X } } ^ { ( \ell + 1 ) } + \mathrm { F F N } ( \overline { { X } } ^ { ( \ell + 1 ) } ) \right) .\tag{14}
$$

The residual structure stabilizes optimization and preserves useful information from previous iterations, while the FFN increases representational capacity beyond linear aggregation. In practice, we find that a small number of iterations $( N _ { \mathrm { s a } } { = } 2 )$ is suficient for modeling ultra-long histories.

## 3.4 Two-Time-Scale Reasoning with STCA

We combine two complementary time scales: a recent sufix for finegrained recency modeling and a compressed sketch for ultra-long signals.

Recent history. We model the most recent $L _ { r } { = } 1 0 \mathrm { K }$ events directly using STCA, where $X _ { r } \in \mathbb { R } ^ { L _ { r } \times d }$ :

$$
\begin{array} { r } { \mathbf { z } _ { r } = \mathrm { S T C A } _ { 1 0 k } ( \mathbf { x } _ { t } , X _ { r } ) . } \end{array}\tag{15}
$$

Ultra-long history. Before sketch-side STCA, a width adapter $W _ { A }$ distinct from the assignment matrix �, maps the SA sketch to the STCA width, while $\phi _ { t }$ maps the target representation:

$$
\widehat { X } = \widetilde { X } W _ { A } \in \mathbb { R } ^ { k \times D } , \qquad \widehat { \mathbf { x } } _ { t } = \phi _ { t } ( \mathbf { x } _ { t } ) \in \mathbb { R } ^ { D } .\tag{16}
$$

We then apply STCA over the adapted sketch:

$$
\begin{array} { r } { \mathbf { z } _ { u } = \operatorname { S T C A } _ { \mathrm { s k e t c h } } ( \widehat { \mathbf { x } } _ { t } , \widehat { X } ) . } \end{array}\tag{17}
$$

Since $\widehat { X }$ has fixed length $k ,$ the cost of this branch is independent of the original sequence length once the sketch is available.

Fusion. The recent and ultra-long representations are passed to the downstream ranking backbone, which may combine them with target and non-sequential features through a lightweight MLP, a gated fusion module, or its native fusion mechanism. Figure 2 shows MixFormer [16] as our deployed instantiation, but SequenceO1 does not require a particular fusion backbone.

## 4 System

This section describes the system design that makes 100K history modeling deployable at full trafic on Douyin. Our guiding principle is amortization: the expensive ultra-long compression is user-only and should be computed once, reused across candidates, and reused across nearby requests whenever possible. We implement this prin ciple with a training-side local KVCache as the primary reuse mechanism, the same cache interface for serving, MRLB, pipeline lift, and FlashSA.

## 4.1 Training-Side Local KVCache and Serving Reuse

During training, we maintain a local key–value cache (local KV-Cache) for the ultra-long SA sketch $\widetilde { X } _ { u } \in \mathbb { R } ^ { k \times d }$ because it is targetagnostic and depends only on user-side signals. Request-specific features, including the recent sufix $X _ { u , r }$ and candidate-side features, are not cached. Optionally, one can also cache sketch-side linear projections that remain user-only under a fixed model version; in our complexity analysis, we use the conservative setting where the adapter output is not cached.

Each cache entry is keyed by user id, model version, history timestamp, and sketch configuration. We use TTL-based invalidation with capacity eviction to bound memory and staleness. In training, a 3-hour TTL within MRLB groups yields an efective hit rate of about $p _ { \mathrm { h i t } } ^ { \mathrm { t r a i n } } \approx 0 . 5$ . In serving, the same cache interface with a 1-hour TTL gives an empirical hit rate of about $p _ { \mathrm { h i t } } ^ { \mathrm { i n f e r } } \approx 0 . 6$ . This shared get/put interface reduces train/serve skew while substantially reducing repeated 100K feature storage, communication, and sketch computation. The reported FLOP reductions use these observed hit rates as operating points. On a cache miss, the additional raw-history computation is only the user-only SA sketching step, which can still be amortized by MRLB or executed upstream.

## 4.2 Multi-Request Level Batching (MRLB)

Even with SA, constructing $\widetilde { X } _ { u } \ = \ \mathrm { S A } ( X _ { u } )$ over 100K tokens is expensive. In production, a user often issues multiple consecutive requests within a short time window, while the long-term history changes slowly. MRLB groups such requests from the same user and computes the ultra-long sketch once, reusing it for all targets in the group:

$$
\widetilde { X } _ { u } = { \mathrm { S A } } ( X _ { u } ) \in \mathbb { R } ^ { k \times d } .\tag{18}
$$

Request-specific components, such as the recent sufix and candidate features, are still computed per request.

In practice, we cap the aggregation degree to keep memory stable and apply request-aware masking in the recent-history branch to prevent cross-request leakage. By amortizing user-only sketching, MRLB reduces repeated data movement and improves training utilization in the 100K setting.

## 4.3 Pipeline Lift

Fine-ranking typically receives only a fraction of the end-to-end latency budget because recall, coarse ranking, filtering, and feature fetching already consume substantial time. In conventional pipelines, ranking-side sequence modeling must wait until the candidate set is available.

Since $\widetilde { X } _ { u } = \mathrm { S A } ( X _ { u } )$ depends only on user-side information, we can lift this computation to request entry or other upstream stages and pass the sketch forward as a user feature, as shown in Fig. 2. On cache hits, a constant-time lookup supplies the fixed-size sketch before the raw 100K feature sequence is materialized and transferred to the model; on misses, the user-only computation can still run in parallel with upstream retrieval. Thus, cache hits remove rawlength feature storage, communication, and computation from the sample path, while pipeline lift further reduces the latency impact on misses.

![](images/15edbd79c2b8b47e8f051d85173a34e1bc24b79bb10d515b5a151536da9f44ec.jpg)  
Figure 3: Distribution of per-token maximum normalized weights under the two normalization rules in the 100K setting; their scales difer because the normalization axes difer.

## 4.4 FlashSA: Kernel Optimization for SA

A naive SA implementation materializes the prototype–token afin ity matrix $S \in \mathbb { R } ^ { k \times n }$ and often the assignment matrix $A \in \mathbb { R } ^ { k \times n }$ from §3.1, leading to �(��) intermediate storage and heavy memory trafic at �=100K, plus low eficiency from multiple kernel launches under ragged batching. We implement FlashSA, a fused kernel that streams the computation: it performs block-wise afinity computation, prototype-wise softmax statistics (max and log-sum-exp), and the final aggregation into $\widetilde { X }$ without storing the full � × � scores in HBM. This reduces peak memory, improves throughput via fewer launches and better locality, and stabilizes mixed-precision execution through numerically robust softmax and accumulation.

Together, these components make the 100K sketch path reusable, cache-friendly, and hardware-eficient, enabling end-to-end 100K sequence modeling at billion scale on Douyin.

## 5 Experiments

Evaluation settings. We report two ofline evaluation settings. Most ablations use a lightweight setting with a simplified dense component for eficient experimentation, with STCA(512) as the baseline, and focus on Finish AUC. The SA-vs-vanilla diagnostic additionally reports multi-task UAUC to verify that the normalization change improves several engagement objectives. The production ofline evaluation and online A/B test use the full production ranker: the baseline is STCA 10K with lifelong TWIN V2, while SequenceO1 removes TWIN V2 and replaces it with the end-to-end 100K sketch branch.

## 5.1 Sketch Attention vs. Vanilla Attention

We analyze how prototype-wise normalization in Sketch Attention (SA) difers from vanilla attention under the 100K ablation setting. We keep the same STCA(512)-based ranking setup and replace only the normalization mechanism in the ultra-long compression branch. For each sampled example, we compute the prototype–token weight matrix � and record the maximum normalized weight associated with every token. Because prototype-wise and token-wise normalization operate over diferent axes, these maxima are not calibrated confidence scores and should not be compared as direct evidence of prototype specialization. Figure 3 instead reports the raw distributions under the respective conservation constraints as a descriptive diagnostic; the shift between the histograms reflects both the induced allocation behavior and the diferent normalization-domain sizes.

Table 1: UAUC improvements of SA over vanilla attention under the 100K setting.
<table><tr><td>Task</td><td>like</td><td>follow</td><td>clicmmt</td><td>comment</td><td>share</td><td>finish</td></tr><tr><td>∆UAUC</td><td>+0.10%</td><td>+0.20%</td><td>+0.09%</td><td>+0.19%</td><td>+0.14%</td><td>+0.01%</td></tr></table>

Table 2: SA architecture ablations under the 100K ablation setting. The final SA setting achieves +1.07% Finish AUC over STCA(512); all numbers report changes relative to this final SA setting.
<table><tr><td>Variant</td><td>Change vs. final SA</td></tr><tr><td> $\mathrm { F i n a l S A } \left( k { = } 1 \mathrm { K } , N _ { \mathrm { s a } } { = } 2 , d { = } 1 2 8 \right)$ </td><td>0.00%</td></tr><tr><td> $k \colon 1 \mathrm { K } \to 5 1 2$ </td><td>-0.04%</td></tr><tr><td> $k \colon 1 \mathrm { K } \to 2 \mathrm { K }$ </td><td>+0.03%</td></tr><tr><td> $N _ { \mathrm { s a } } \colon 2 \substack { \longrightarrow 3 }$ </td><td>+0.02%</td></tr><tr><td> $d \colon 1 2 8 { \\to } 6 4$ </td><td>-0.10%</td></tr><tr><td>w/o action-side SwiGLU fusion</td><td>-0.20%</td></tr><tr><td>w/o per-layer Add&amp;Norm + SwiGLU FFN</td><td>-0.10%</td></tr></table>

With prototype-wise normalization, each behavior token is explicitly allocated across prototypes, which is aligned with constructing target-agnostic summaries before target conditioning. Such allocation may provide an interface for possible downstream sketch-to-behavior retrieval or localized refinement, although these operations are not evaluated components in this work. The tasklevel evidence comes from Table 1: prototype-wise normalization improves UAUC across multiple engagement objectives, supporting it as a better mechanism for constructing reusable 100K sketches in our setting.

## 5.2 Ablations and Compression Methods

Under the same 100K lightweight STCA(512) setup, Table 2 reports changes relative to the final SA configuration, while Table 3 reports absolute Finish AUC gains over STCA(512). The final setting— $k { = } 1 \mathrm { K } , N _ { \mathrm { s a } } { = } 2 , d { = } 1 2 8$ , action-side SwiGLU fusion of item and actiontype embeddings, and per-layer Add&Norm with a SwiGLU FFN— achieves +1.07%. Increasing to 2K prototypes or 3 layers adds only +0.03% or +0.02%, respectively; we treat these as cost–quality guidance rather than standalone significance evidence and retain the smaller production configuration. Reducing width or removing refinement degrades performance, with the -0.20% from removing action-side fusion highlighting the value of action-aware sketching.

Table 3 compares methods under the same 1K × 128 budget: Kmeans clustering inspired by TWIN V2 [37], neighboring-behavior chunk means, position-bucket queries over roughly 100 behaviors each, a Lightning-Attention long-sequence compressor, and queries initialized by the most recent 1K behaviors. SA performs best, showing that its gain comes from compression quality rather than representation size; its advantage over position buckets also suggests that recommendation histories are less locally structured than language because related behaviors can be far apart.

Table 3: Comparison of 100K compression methods in the ablation setting. The base model uses STCA(512) only, while all non-base variants add an ultra-long compression branch to the same ranking setup. For fair comparison, each compression method summarizes the 100K history into a compact 1�×128 representation. Numbers report Finish AUC improvements over the STCA(512) base model.
<table><tr><td>Compression method</td><td>Gain vs. STCA(512)</td></tr><tr><td>Base (STCA(512))</td><td>0.00%</td></tr><tr><td>SA (final)</td><td>+1.07%</td></tr><tr><td>TWIN V2 KMeans Clustering [37]</td><td>+0.30%</td></tr><tr><td>Chunk mean pooling</td><td>+0.46%</td></tr><tr><td>Position-bucket queries</td><td>+0.72%</td></tr><tr><td>Lightning-Attention comp. [28]</td><td>+0.83%</td></tr><tr><td>Recent-behavior query init. [2]</td><td>+0.78%</td></tr></table>

## 5.3 Ofline Performance on Douyin Dataset

We evaluate SequenceO1 on the full production Douyin ofline dataset against STCA 10K with lifelong TWIN V2. SequenceO1 removes TWIN V2 in favor of the end-to-end 100K sketch branch, so this strict comparison requires it to improve ranking while absorbing the previous module’s long-term signals. Table 4 reports relative AUC and UAUC improvements across engagement objectives.

SequenceO1 improves every evaluated objective despite removing TWIN V2, showing that the 100K branch successfully replaces the two-stage lifelong module while further improving all evaluated metrics in the full ranker. UAUC gains are strongest on preferencesensitive objectives (Like, Share, Favourite, and Dislike), while core consumption objectives (Finish and Skip) also improve, indicating more efective long-term personalization than the previous clusterbased pipeline.

## 5.4 Online Performance

We deployed SequenceO1 for one month on Douyin and Douyin Lite, replacing the lifelong TWIN V2 module in the production STCA 10K baseline with the end-to-end 100K sketch branch. Table 5 reports relative changes over control in 30-Day Activeness, Duration, Finish, Comment, Like, and Dislike, overall and by user activity.

SequenceO1 significantly improves Activeness, Duration, Finish, Comment, and Like while reducing Dislike on both apps; Finish rises by +2.33% on Douyin and +3.49% on Douyin Lite, indicating better sustained consumption. Gains are stable across activity segments: low-active users improve in activeness and duration, suggesting re-engagement, while high-active users improve in Finish and Like.

Table 4: Ofline improvements in the full production setting on the Douyin dataset. The baseline is production STCA 10K with lifelong TWIN V2; SequenceO1 removes TWIN V2 and uses the end-to-end 100K sketch branch instead.
<table><tr><td>Metric</td><td>Finish</td><td>Skip</td><td>Head</td><td>Like</td><td>Follow</td><td>Comment</td><td>Clicmmt</td><td>Share</td><td>Favourite</td><td>Dislike</td></tr><tr><td>ΔAUC</td><td>+0.29%</td><td>+0.23%</td><td>+0.18%</td><td>+0.09%</td><td>+0.05%</td><td>+0.04%</td><td>+0.16%</td><td>+0.18%</td><td>+0.30%</td><td>+1.29%</td></tr><tr><td>ΔUAUC</td><td>+0.40%</td><td>+0.43%</td><td>+0.46%</td><td>+0.72%</td><td>+0.58%</td><td>+0.43%</td><td>+0.61%</td><td>+0.94%</td><td>+1.47%</td><td>+3.63%</td></tr></table>

Table 5: Online A/B results over the production STCA 10K + TWIN V2 baseline on Douyin and Douyin Lite (all statistically significant).
<table><tr><td rowspan="2"></td><td colspan="6">Douyin</td><td colspan="6">Douyin Lite</td></tr><tr><td>30-Day Act.↑</td><td>Duration↑</td><td>Finish↑</td><td>Comment↑</td><td>Like↑</td><td>Dislike↓</td><td>30-Day Act.↑</td><td>Duration↑</td><td>Finish↑</td><td>Comment↑</td><td>Like↑</td><td>Dislike↓</td></tr><tr><td>Overall</td><td>+0.1968%</td><td>+1.4999%</td><td>+2.3256%</td><td>+3.5910%</td><td>+2.3617%</td><td>-6.9829%</td><td>+0.2335%</td><td>+1.6655%</td><td>+3.4872%</td><td>+8.6132%</td><td>+2.9399%</td><td>-5.9972%</td></tr><tr><td>Low-active</td><td>+0.5484%</td><td>+2.1476%</td><td>+3.2040%</td><td>+2.9099%</td><td>+3.5739%</td><td>-8.3061%</td><td>+0.6614%</td><td>+1.9989%</td><td>+3.9845%</td><td>+13.4620%</td><td>+6.1890%</td><td>-2.4977%</td></tr><tr><td>Middle-active</td><td>+0.5235%</td><td>+2.2408%</td><td>+3.1550%</td><td>+5.1155%</td><td>+3.0477%</td><td>-12.1796%</td><td>+0.4306%</td><td>+1.8468%</td><td>+3.0968%</td><td>+8.7780%</td><td>+0.0254%</td><td>-3.6990%</td></tr><tr><td>High-active</td><td>+0.1789%</td><td>+1.6334%</td><td>+2.4733%</td><td>+3.5232%</td><td>+2.1611%</td><td>-5.7208%</td><td>+0.3090%</td><td>+1.8161%</td><td>+3.4367%</td><td>+8.3048%</td><td>+3.2317%</td><td>-3.1752%</td></tr><tr><td>Full-active</td><td>+0.0622%</td><td>+1.2705%</td><td>+2.0744%</td><td>+3.1494%</td><td>+2.1376%</td><td>-7.8567%</td><td>+0.0853%</td><td>+1.5254%</td><td>+3.2774%</td><td>+7.5138%</td><td>+2.7786%</td><td>-6.5438%</td></tr></table>

Overall, the model complements short-term signals with stable long-term preferences across platforms and cohorts.

## 5.5 Relation to Two-Stage Transfer

Recent industrial systems often use two-stage representation transfer: LLaTTE [46] reports a 50%–53% ratio from upstream usermodeling to downstream ranking gains, while SOLARIS [25] reports about 42% and 44% on Instagram and Facebook products. Because these results come from diferent systems, they contextualize transfer eficiency rather than provide an apples-to-apples comparison; our same-system evidence comes from replacing production TWIN V2 with an end-to-end 100K sketch branch.

In this ablation, Fig. 1(c) shows that SequenceO1 retains 83% of directly scaled STCA’s gain at 100K truncation (85K average length): +1.07% versus +1.29% Finish AUC. Thus, it translates sequencelength scaling into ranking gains while preserving a cacheable path. Unlike two-stage transfer, its simpler sketch remains jointly optimized with the final ranker instead of becoming a fixed external feature, and could support future in-graph sketch retrieval or lo calized refinement, neither of which is evaluated in the current deployment.

## 6 Related Work and Discussion

## 6.1 Modeling User Behavior Sequences

User behavior modeling is central to recommender systems. Early methods include item-to-item collaborative filtering and Markov transition models [23, 36], while deep learning introduced sessionbased RNNs and large-scale industrial ranking architectures [9, 15]. Attention-based models later became dominant: DIN and DIEN perform target-conditioned user-interest modeling [51, 52], SASRec and BERT4Rec use self-attention for sequential modeling [19, 38], and BST demonstrates Transformer-based behavior modeling in industrial ranking [7, 16]. Multi-interest methods further capture diverse user intents at scale [21].

Recent production systems increasingly emphasize longer histories under serving constraints. PinnerFormer and TransAct improve long user representation and real-time action modeling at

Pinterest [31, 44, 45]. Other systems retrieve, sample, or compress target-relevant subsets from lifelong behavior logs, including SIM, UBR4CTR, SDIM, TWIN, and TWIN V2 [1, 3, 33, 34, 37]. In particular, TWIN V2 uses ofline hierarchical clustering to compress lifecycle behaviors into manageable clusters, followed by online retrieval and cluster-aware target attention for CTR prediction. These methods are efective under serving constraints, but their retrieval or compression stages are typically separated from the final ranking objective.

## 6.2 Caching and Reuse

Caching strategies difer mainly by whether the cached object is end-to-end over raw histories and whether its size scales with sequence length.

UIC+MIMN [32] and HPMN [35] maintain incrementally updated user states, enabling constant-time reads but not end-toend optimization over the raw ultra-long sequence at inference time. CTR-oriented methods cache candidate-agnostic summaries through grouping, quantization, or clustering, including DGIN [24], DMQN [42], and C-Former [41]; generative recommenders such as VISTA [8] cache summary tokens. These methods usually target shorter horizons or rely on separate pipelines. Transformer KV/prefix caching reduces recomputation but stores per-position states, so cache footprint grows with length [22, 39, 47, 49]. In contrast, SequenceO1 caches a fixed-size user-only SA sketch (100K→ �) in training and serving, making cache size independent of the original history length and eliminating repeated raw-length feature storage, communication, and model computation on cache hits.

SequenceO1 difers from multi-stage representation-transfer systems such as LLaTTE [46] and SOLARIS [25] by keeping the ultralong sequence path end-to-end optimized with the final ranking objective. It uses a simpler cacheable sketch rather than a separate upstream user model, providing a practical alternative to the previous TWIN V2-style lifelong module while preserving a direct optimization path for final ranking.

## 7 Conclusion

SequenceO1 scales end-to-end sequence-based recommendation to 100K in production and supports extension toward million-scale histories by combining Sketch Attention, two-time-scale STCA reasoning, and training-first cache amortization. Douyin experiments show consistent gains in lightweight ablations, full production offline evaluation, and online A/B tests: a compact reusable sketch preserves most of the benefit of direct long-sequence scaling while substantially reducing feature storage, communication, and compu tation in training and serving. Future directions include hierarchical or multi-resolution sketches and adaptive capacity across users or behavior patterns; more broadly, the results show that coordinated model and system design can jointly deliver ultra-long modeling and production eficiency.

## References

[1] Yue Cao, Xiaojiang Zhou, Jiaqi Feng, Peihao Huang, Yao Xiao, Dayao Chen, and Sheng Chen. 2022. Sampling Is All You Need on Modeling Long-Term User Behaviors for CTR Prediction. In CIKM.

[2] Zheng Chai, Qin Ren, Xijun Xiao, Huizhi Yang, Bo Han, Sijun Zhang, Di Chen, Hui Lu, Wenlin Zhao, Lele Yu, Xionghang Xie, Shiru Ren, Xiang Sun, Yaocheng Tan, Peng Xu, Yuchao Zheng, and Di Wu. 2025. LONGER: Scaling Up Long Sequence Modeling in Industrial Recommenders. In RecSys.

[3] Jianxin Chang, Chenbin Zhang, Zhiyi Fu, Xiaoxue Zang, Lin Guan, Jing Lu, Yiqun Hui, Dewei Leng, Yanan Niu, Yang Song, and Kun Gai. 2023. TWIN: TWo-stage Interest Network for Lifelong User Behavior Modeling in CTR Prediction at Kuaishou. In KDD.

[4] Moses Charikar, Kevin C. Chen, and Martin Farach-Colton. 2002. Finding Frequent Items in Data Streams. In ICALP.

[5] Qiwei Chen, Changhua Pei, Shanshan Lv, Chao Li, Junfeng Ge, and Wenwu Ou. 2021. End-to-End User Behavior Retrieval in Click-Through Rate Prediction Model. arXiv:2108.04468 [cs.IR] https://arxiv.org/abs/2108.04468

[6] Qiwei Chen, Yue Xu, Changhua Pei, Shanshan Lv, Tao Zhuang, and Junfeng Ge. 2022. Eficient Long Sequential User Data Modeling for Click-Through Rate Prediction. arXiv:2209.12212 [cs.IR] https://arxiv.org/abs/2209.12212

[7] Qiwei Chen, Huan Zhao, Wei Li, Pipei Huang, and Wenwu Ou. 2019. Behavior Sequence Transformer for E-commerce Recommendation in Alibaba. arXiv:1905.06874

[8] Zhimin Chen, Chenyu Zhao, Ka Chun Mo, Yunjiang Jiang, Jane H. Lee, Shouwei Chen, Khushhall Chandra Mahajan, Ning Jiang, Kai Ren, Jinhui Li, and Wen-Yun Yang. 2025. Massive Memorization with Hundreds of Trillions of Parameters for Sequential Transducer Generative Recommenders. CoRR abs/2510.22049 (2025).

[9] Paul Covington, Jay Adams, and Emre Sargin. 2016. Deep Neural Networks for YouTube Recommendations. In RecSys.

[10] Tri Dao. 2023. FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning. arXiv:2307.08691 [cs.LG] https://arxiv.org/abs/2307.08691

[11] Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. 2022. FlashAttention: Fast and Memory-Eficient Exact Attention with IO-Awareness. In NeurIPS.

[12] Qin Ding, Kevin Course, Linjian Ma, Jianhui Sun, Ruochen Liu, Zhao Zhu, Chunxing Yin, Wei Li, Dai Li, Yu Shi, Xuan Cao, Ze Yang, Han Li, Xing Liu, Bi Xue, Hongwei Li, Rui Jian, Daisy Shi He, Jing Qian, Matt Ma, Qunshu Zhang, and Rui Li. 2026. Bending the Scaling Law Curve in Large-Scale Recommendation Systems. CoRR abs/2602.16986 (2026).

[13] Lin Guan, Jia-Qi Yang, Zhishan Zhao, Beichuan Zhang, Bo Sun, Xuanyuan Luo, Jinan Ni, Xiaowen Li, Yuhang Qi, Zhifang Fan, Hangyu Wang, Qiwei Chen, Yi Cheng, Feng Zhang, and Xiao Yang. 2025. Make It Long, Keep It Fast: End-to-End 10k-Sequence Modeling at Billion Scale on Douyin. CoRR abs/2511.06077 (2025).

[14] Zhicheng He, Weiwen Liu, Wei Guo, Jiarui Qin, Yingxue Zhang, Yaochen Hu, and Ruiming Tang. 2023. A Survey on User Behavior Modeling in Recommender Systems. In IJCAI.

[15] Balázs Hidasi, Alexandros Karatzoglou, Linas Baltrunas, and Domonkos Tikk. 2016. Session-based Recommendations with Recurrent Neural Networks. In ICLR.

[16] Xu Huang, Hao Zhang, Zhifang Fan, Yunwen Huang, Zhuoxing Wei, Zheng Chai, Jinan Ni, Yuchao Zheng, and Qiwei Chen. 2026. MixFormer: Co-Scaling Up Dense and Sequence in Industrial Recommenders. arXiv:2602.14110 [cs.IR] https://arxiv.org/abs/2602.14110

[17] Samuel Humeau, Kurt Shuster, Marie-Anne Lachaux, and Jason Weston. 2020. Poly-encoders: Architectures and Pre-training Strategies for Fast and Accurate Multi-sentence Scoring. In ICLR.

[18] Andrew Jaegle, Felix Gimeno, Andy Brock, Oriol Vinyals, Andrew Zisserman, and João Carreira. 2021. Perceiver: General Perception with Iterative Attention. In ICML.

[19] Wang-Cheng Kang and Julian J. McAuley. 2018. Self-Attentive Sequential Recommendation. In ICDM.

[20] Juho Lee, Yoonho Lee, Jungtaek Kim, Adam R. Kosiorek, Seungjin Choi, and Yee Whye Teh. 2019. Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks. In ICML

[21] Chao Li, Zhiyuan Liu, Mengmeng Wu, Yuchi Xu, Huan Zhao, Pipei Huang, Guoliang Kang, Qiwei Chen, Wei Li, and Dik Lun Lee. 2019. Multi-Interest Network with Dynamic Routing for Recommendation at Tmall. In CIKM.

[22] Jingyu Li, Zhaocheng Du, Qianhui Zhu, kaiyuan Li, Zhicheng Zhang, Song-Li Wu, Chaolang Li, and Pengwen Dai. 2026. CollectiveKV: Decoupling and Sharing Collaborative Information in Sequential Recommendation. arXiv:2601.19178 [cs.AI]

[23] Greg Linden, Brent Smith, and Jeremy York. 2003. Amazon.com Recommendations: Item-to-Item Collaborative Filtering. IEEE Internet Comput. 7, 1 (2003), 76–80.

[24] Qi Liu, Xuyang Hou, Haoran Jin, Jin Chen, Zhe Wang, Defu Lian, Tan Qu, Jia Cheng, and Jun Lei. 2023. Deep Group Interest Modeling of Full Lifelong User Behaviors for CTR Prediction. CoRR abs/2311.10764 (2023).

[25] Zikun Liu, Liang Luo, Qianru Li, Zhengyu Zhang, Wei Ling, Jingyi Shen, Zeliang Chen, Yaning Huang, Jingxian Huang, Abdallah Aboelela, Chonglin Sun, Feifan Gu, Fenggang Wu, Hang Qu, Huayu Li, Jill Pan, Kaidi Pei, Laming Chen, Longhao Jin, Qin Huang, Tongyi Tang, Varna Puvvada, Wenlin Chen, Xiaohan Wei, Xu Cao, Yantao Yao, Yuan Jin, Yunchen Pu, Yuxin Chen, Zijian Shen, Zhengkai Zhang, Dong Liang, and Ellie Wen. 2026. SOLARIS: Speculative Ofloading of Latent-bAsed Representation for Inference Scaling. arXiv:2604.12110 [cs.LG] https://arxiv.org/abs/2604.12110

[26] Zhuoran Liu, Leqi Zou, Xuan Zou, Caihua Wang, Biao Zhang, Da Tang, Bolin Zhu, Yijie Zhu, Peng Wu, Ke Wang, and Youlong Cheng. 2022. Monolith: Real Time Recommendation System with Collisionless Embedding Table. In Proceedings of the 5th Workshop on Online Recommender Systems and User Modeling co-located with the 16th ACM Conference on Recommender Systems, ORSUM@RecSys 2022, Seattle, WA, USA, September 23rd, 2022 (CEUR Workshop Proceedings). CEUR WS.org.

[27] Francesco Locatello, Dirk Weissenborn, Thomas Unterthiner, Aravindh Mahendran, Georg Heigold, Jakob Uszkoreit, Alexey Dosovitskiy, and Thomas Kipf. 2020. Object-Centric Learning with Slot Attention. In NeurIPS.

[28] MiniMax, Aonian Li, Bangwei Gong, Bo Yang, Boji Shan, Chang Liu, Cheng Zhu, Chunhao Zhang, Congchao Guo, Da Chen, Dong Li, Enwei Jiao, Gengxin Li, Guojun Zhang, Haohai Sun, Houze Dong, Jiadai Zhu, Jiaqi Zhuang, Jiayuan Song, Jin Zhu, Jingtao Han, Jingyang Li, Junbin Xie, Junhao Xu, Junjie Yan, Kaishun Zhang, Kecheng Xiao, Kexi Kang, Le Han, Leyang Wang, Lianfei Yu, Liheng Feng, Lin Zheng, Linbo Chai, Long Xing, MeizhiJu, Mingyuan Chi, Mozhi Zhang, Peikai Huang, Pengcheng Niu, Pengfei Li, Pengyu Zhao, Qi Yang, Qidi Xu, Qiexiang Wang, Qin Wang, Qiuhui Li, Ruitao Leng, Shengmin Shi, Shuqi Yu, Sichen Li, Songquan Zhu, Tao Huang, Tianrun Liang, Weigao Sun, Weixuan Sun, Weiyu Cheng, Wenkai Li, Xiangjun Song, Xiao Su, Xiaodong Han, Xinjie Zhang, Xinzhu Hou, Xu Min, Xun Zou, Xuyang Shen, Yan Gong, Yingjie Zhu, Yipeng Zhou, Yiran Zhong, Yongyi Hu, Yuanxiang Fan, Yue Yu, Yufeng Yang, Yuhao Li, Yunan Huang, Yunji Li, Yunpeng Huang, Yunzhi Xu, Yuxin Mao, Zehan Li, Zekang Li, Zewei Tao, Zewen Ying, Zhaoyang Cong, Zhen Qin, Zhenhua Fan, Zhihang Yu, Zhuo Jiang, and Zijia Wu. 2025. MiniMax-01: Scaling Foundation Models with Lightning Attention. arXiv:2501.08313 [cs.CL]

[29] Dheevatsa Mudigere, Yuchen Hao, Jianyu Huang, Zhihao Jia, Andrew Tulloch, Srinivas Sridharan, Xing Liu, Mustafa Ozdal, Jade Nie, Jongsoo Park, Liang Luo, Jie Amy Yang, Leon Gao, Dmytro Ivchenko, Aarti Basant, Yuxi Hu, Jiyan Yang, Ehsan K. Ardestani, Xiaodong Wang, Rakesh Komuravelli, Ching-Hsiang Chu, Serhat Yilmaz, Huayu Li, Jiyuan Qian, Zhuobo Feng, Yinbin Ma, Junjie Yang, Ellie Wen, Hong Li, Lin Yang, Chonglin Sun, Whitney Zhao, Dimitry Melts, Krishna Dhulipala, K. R. Kishore, Tyler Graf, Assaf Eisenman, Kiran Kumar Matam, Adi Gangidi, Guoqiang Jerry Chen, Manoj Krishnan, Avinash Nayak, Krishnaku mar Nair, Bharath Muthiah, Mahmoud khorashadi, Pallab Bhattacharya, Petr Lapukhov, Maxim Naumov, Ajit Mathews, Lin Qiao, Mikhail Smelyanskiy, Bill Jia, and Vijay Rao. 2022. Software-hardware co-design for fast and scalable training of deep learning recommendation models. In ISCA.

[30] Maxim Naumov, Dheevatsa Mudigere, Hao-Jun Michael Shi, Jianyu Huang, Narayanan Sundaraman, Jongsoo Park, Xiaodong Wang, Udit Gupta, Carole-Jean Wu, Alisson G. Azzolini, Dmytro Dzhulgakov, Andrey Mallevich, Ilia Cherni avskii, Yinghai Lu, Raghuraman Krishnamoorthi, Ansha Yu, Volodymyr Kondratenko, Stephanie Pereira, Xianjie Chen, Wenlin Chen, Vijay Rao, Bill Jia, Liang Xiong, and Misha Smelyanskiy. 2019. Deep Learning Recommendation Model for Personalization and Recommendation Systems. arXiv:1906.00091

[31] Nikil Pancha, Andrew Zhai, Jure Leskovec, and Charles Rosenberg. 2022. Pinner-Former: Sequence Modeling for User Representation at Pinterest. In KDD.

[32] Qi Pi, Weijie Bian, Guorui Zhou, Xiaoqiang Zhu, and Kun Gai. 2019. Practice on Long Sequential User Behavior Modeling for Click-Through Rate Prediction. In KDD.

[33] Qi Pi, Guorui Zhou, Yujing Zhang, Zhe Wang, Lejian Ren, Ying Fan, Xiaoqiang Zhu, and Kun Gai. 2020. Search-based User Interest Modeling with Lifelong Sequential Behavior Data for Click-Through Rate Prediction. In CIKM.

[34] Jiarui Qin, Weinan Zhang, Xin Wu, Jiarui Jin, Yuchen Fang, and Yong Yu. 2020. User Behavior Retrieval for Click-Through Rate Prediction. In SIGIR.

[35] Kan Ren, Jiarui Qin, Yuchen Fang, Weinan Zhang, Lei Zheng, Weijie Bian, Guorui Zhou, Jian Xu, Yong Yu, Xiaoqiang Zhu, and Kun Gai. 2019. Lifelong Sequential Modeling with Personalized Memorization for User Response Prediction. In SIGIR.

[36] Stefen Rendle, Christoph Freudenthaler, and Lars Schmidt-Thieme. 2010. Factorizing personalized Markov chains for next-basket recommendation. In WWW.

[37] Zihua Si, Lin Guan, Zhongxiang Sun, Xiaoxue Zang, Jing Lu, Yiqun Hui, Xingchao Cao, Zeyu Yang, Yichen Zheng, Dewei Leng, Kai Zheng, Chenbin Zhang, Yanan Niu, Yang Song, and Kun Gai. 2024. TWIN V2: Scaling Ultra-Long User Behavior Sequence Modeling for Enhanced CTR Prediction at Kuaishou. In CIKM.

[38] Fei Sun, Jun Liu, Jian Wu, Changhua Pei, Xiao Lin, Wenwu Ou, and Peng Jiang. 2019. BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer. In CIKM, Wenwu Zhu, Dacheng Tao, Xueqi Cheng, Peng Cui, Elke A. Rundensteiner, David Carmel, Qi He, and Jefrey Xu Yu (Eds.).

[39] Jiarui Wang, Huichao Chai, Yuanhang Zhang, Zongjin Zhou, Wei Guo, Xingkun Yang, Qiang Tang, Bo Pan, Jiawei Zhu, Ke Cheng, Yuting Yan, Shulan Wang, Yingjie Zhu, Zhengfan Yuan, Jiaqi Huang, Yuhan Zhang, Xiaosong Sun, Zhinan Zhang, Hong Zhu, Yongsheng Zhang, Tiantian Dong, Zhong Xiao, Deliang Liu, Chengzhou Lu, Yuan Sun, Zhiyuan Chen, Xinming Han, Zaizhu Liu, Yaoyuan Wang, Ziyang Zhang, Yong Liu,Jinxin Xu, Yajing Sun, Zhoujun Yu, Wenting Zhou, Qidong Zhang, Zhengyong Zhang, Zhonghai Gu, Yibo Jin, Yongxiang Feng, and Pengfei Zuo. 2026. RelayGR: Scaling Long-Sequence Generative Recommendation via Cross-Stage Relay-Race Inference. arXiv:2601.01712 [cs.DC]

[40] Sinong Wang, Belinda Z. Li, Madian Khabsa, Han Fang, and Hao Ma. 2020. Linformer: Self-Attention with Linear Complexity. (2020). arXiv:2006.04768

[41] Xingmei Wang, Shiyao Wang, Wuchao Li, Jiaxin Deng, Song Lu, Defu Lian, and Guorui Zhou. 2025. Transformers are Good Clusterers for Lifelong User Behavior Sequence Modeling. In CIKM.

[42] Zhuoxing Wei, Qi Liu, and Qingchen Xie. 2025. Deep Multiple Quantization Network on Long Behavior Sequence for Click-Through Rate Prediction. In SIGIR.

[43] Yongji Wu, Defu Lian, Neil Zhenqiang Gong, Lu Yin, Mingyang Yin, Jingren Zhou, and Hongxia Yang. 2021. Linear-Time Self Attention with Codeword Histogram for Eficient Recommendation. In WWW.

[44] Xue Xia, Pong Eksombatchai, Nikil Pancha, Dhruvil Deven Badani, Po-Wei Wang, Neng Gu, Saurabh Vishwas Joshi, Nazanin Farahpour, Zhiyuan Zhang, and Andrew Zhai. 2023. TransAct: Transformer-based Realtime User Action Model for Recommendation at Pinterest. In KDD.

[45] Xue Xia, Saurabh Vishwas Joshi, Kousik Rajesh, Kangnan Li, Yangyi Lu, Nikil Pancha, Dhruvil Deven Badani, Jiajing Xu, and Pong Eksombatchai. 2025. Trans-Act V2: Lifelong User Action Sequence Modeling on Pinterest Recommendation. arXiv:2506.02267

[46] Lee Xiong, Zhirong Chen, Rahul Mayuranath, Shangran Qiu, Arda Ozdemir, Lu Li, Yang Hu, Dave Li, Jingtao Ren, Howard Cheng, Fabian Souto Herrera, Ahmed Agiza, Baruch Epshtein, Anuj Aggarwal, Julia Ulziisaikhan, Chao Wang, Dinesh Ramasamy, Parshva Doshi, Sri Reddy, and Arnold Overwijk. 2026. LLaTTE: Scaling Laws for Multi-Stage Sequence Modeling in Large-Scale Ads Recommen dation. arXiv:2601.20083 [cs.IR] https://arxiv.org/abs/2601.20083

[47] Chaoqun Yang, Xinyu Lin, Wenjie Wang, Yongqi Li, Teng Sun, Xianjing Han, and Tat-Seng Chua. 2025. EARN: Eficient Inference Acceleration for LLM-based Generative Recommendation by Register Tokens. In KDD.

[48] Jiaqi Zhai, Lucy Liao, Xing Liu, Yueming Wang, Rui Li, Xuan Cao, Leon Gao, Zhaojie Gong, Fangda Gu,Jiayuan He, Yinghai Lu, and Yu Shi. 2024. Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations. In ICML.

[49] Zhaoqi Zhang, Haolei Pei, Jun Guo, Tianyu Wang, Yufei Feng, Hui Sun, Shaowei Liu, and Aixin Sun. 2025. OneTrans: Unified Feature Interaction and Sequence Modeling with One Transformer in Industrial Recommender. CoRR abs/2510.26104 (2025).

[50] Fang Zhou, Yaning Huang, Dong Liang, Dai Li, Zhongke Zhang, Kai Wang, Xiao Xin, Abdallah Aboelela, Zheliang Jiang, Yang Wang, Jef Song, Wei Zhang, Chen Liang, Huayu Li, ChongLin Sun, Hang Yang, Lei Qu, Zhan Shu, Mindi Yuan, Emanuele Maccherani, Taha Hayat, John Guo, Varna Puvvada, and Uladzimir Pashkevich. 2024. ERCache: An Eficient and Reliable Caching Framework for Large-Scale User Representations in Meta’s Ads System. arXiv:2410.06497 [cs.IR] https://arxiv.org/abs/2410.06497

[51] Guorui Zhou, Na Mou, Ying Fan, Qi Pi, Weijie Bian, Chang Zhou, Xiaoqiang Zhu, and Kun Gai. 2019. Deep Interest Evolution Network for Click-Through Rate Prediction. In AAAI. 5941–5948.

[52] Guorui Zhou, Xiaoqiang Zhu, Chengru Song, Ying Fan, Han Zhu, Xiao Ma, Yanghui Yan, Junqi Jin, Han Li, and Kun Gai. 2018. Deep Interest Network for Click-Through Rate Prediction. In KDD.

## A Connection to Length-Wise Low-Rank Projection

This section clarifies the connection between Sketch Attention (SA) and length-wise low-rank projection. Our goal is not to claim that SA is a strict instantiation of Linformer or that it inherits a specific approximation guarantee for self-attention. Rather, we use the low-rank perspective as a conceptual lens: it helps explain why an ultra-long history of length � may be summarized by a much shorter representation of length � ≪ � while retaining predictive signals for downstream ranking

The key point is that SequenceO1 compresses the history along the length dimension. From this viewpoint, SA can be interpreted as constructing a data-dependent map from the original sequence of length � to a fixed-size representation of length �, after which target-conditioned reasoning is performed on the compressed representation. This perspective is closely related in spirit to prior low-rank eficient-attention methods, but difers in both construction and system role.

## A.1 Self-Attention as a Length-Wise Mapping

Consider a single-head self-attention layer with query, key, and value matrices

$$
Q , K , V \in \mathbb { R } ^ { n \times d _ { h } } .
$$

The attention output can be written as

$$
\mathrm { S e l f A t t n } ( Q , K , V ) = \Pi V , \qquad \Pi = \mathrm { s o f t m a x } \left( \frac { Q K ^ { \top } } { \sqrt { d _ { h } } } \right) \in \mathbb { R } ^ { n \times n } ,\tag{19}
$$

where Π is the attention weight matrix along the length dimension. Thus, self-attention can be interpreted as applying a length-wise transformation Π to the value matrix $V .$

In general, Π is dense and may have rank up to �. Materializing and applying this � × � matrix incurs quadratic complexity in sequence length. This observation motivates a natural question: does the efective interaction really require full rank along the length axis, or can it be well-approximated in a much lower-dimensional subspace?

## A.2 Low-Rank Approximation along the Length Dimension

Linformer [40] proposes that, under certain assumptions, the attention matrix can often be well-approximated by a low-rank factorization along the length dimension. Concretely, instead of operating on full-length keys and values, one projects them to a lower-dimensiona subspace:

$$
K ^ { \prime } = E ^ { \top } K , \qquad V ^ { \prime } = F ^ { \top } V , \qquad E , F \in \mathbb { R } ^ { n \times k } , \quad k \ll n ,\tag{20}
$$

so that attention is computed over � positions rather than �. The resulting approximation can be written as

$$
\mathrm { S e l f A t t n } ( Q , K , V ) \approx \mathrm { s o f t m a x } \left( \frac { Q K ^ { \prime \top } } { \sqrt { d _ { h } } } \right) V ^ { \prime } .\tag{21}
$$

This formulation suggests that the efective dimensionality of sequence interaction along the length axis can be much smaller than �, and that a fixed projection size � may sufice even as � grows. For our setting, this is an appealing intuition: ultra-long user histories may contain substantial redundancy, so a compact summary along the length dimension may preserve most of the useful information for ranking.

## A.3 Limitations of Explicit Length-Wise Projections

While the above formulation is appealing, directly applying explicit length-wise projections in large-scale recommender systems is challenging:

• Parameter scaling with maximum length. The projection matrices $E , F \in \mathbb { R } ^ { n \times k }$ depend on the maximum sequence length �. In industrial settings with � up to 100K and highly variable user histories, this leads to large and inflexible parameterization.

• Handling variable-length sequences. User histories exhibit significant variation in length. Explicit projections tied to a fixed � require padding or truncation, which introduces ineficiencies and potential artifacts.

• Lack of target-agnostic reuse. The projected representation is typically embedded within an attention pipeline and does not naturally yield a reusable user-only state that can be materialized once and cached across targets and consecutive requests.

• Mismatch to our system objective. In our setting, compression is not only a modeling device but also a systems primitive: the compressed representation must be fixed-size, target-agnostic, and cheap to reuse across both training-side MRLB and serving-side cache hits. Explicit projection methods do not naturally provide this property.

These limitations motivate the need for a length-agnostic, data-dependent, and cacheable compression mechanism.

## A.4 Sketch Attention as an Implicit Length Projection

Sketch Attention (SA) constructs a compact representation of the history in the form

$$
\widetilde { X } = A X , \qquad A \in \mathbb { R } ^ { k \times n } ,\tag{22}
$$

where $\ b X \in \mathbb { R } ^ { n \times d }$ is the history embedding matrix and � is a data-dependent assignment/projection matrix computed from � and a set of learnable prototypes.

Specifically, SA defines

$$
A = \operatorname { s o f t m a x } _ { \mathrm { p r o t o } } \left( { \frac { ( P W _ { Q } ) ( X W _ { K } ) ^ { \top } } { \sqrt { d } } } \right) ,\tag{23}
$$

where $P \in \mathbb { R } ^ { k \times d }$ is a learnable prototype matrix. The normalization is applied over the prototype dimension for each token, ensuring that each token distributes unit mass across prototypes.

This yields an assignment/projection matrix � with the following properties:

• Implicit and data-dependent. Unlike explicit �, �, the assignment/projection matrix � is constructed dynamically from the input �, allowing it to adapt to diferent users and sequences.

• Length-agnostic parameterization. The learnable parameters consist of $P , W _ { Q } ,$ and $W _ { K } ,$ , with total size $O ( k d + d ^ { 2 } )$ , independent of �.

• Fixed-size output. Regardless of the original sequence length, SA always produces a representation $\widetilde { X } \in \mathbb { R } ^ { k \times d }$ . This makes the downstream cost of sketch-side reasoning independent of � once the sketch is available.

• End-to-end diferentiability. The projection is fully diferentiable and optimized jointly with downstream target-conditioned reasoning.

• Cacheability. Since $\widetilde { X }$ depends only on user-side signals, it can be computed once and reused across multiple targets and consecutive requests. This property is central to our low-rank caching view.

Therefore, SA can be viewed as realizing an implicit low-rank projection along the length dimension: instead of explicitly learning $n \times k$ projection matrices, it constructs a compact, input-dependent map � whose output size is fixed and whose parameters do not scale with the maximum sequence length.

## A.5 Relation and Diference to Linformer-Style Compression

The connection to Linformer is primarily structural rather than algorithmic. Both views rely on the idea that long-sequence computation can be reduced through a compact representation along the length axis. However, the two constructions difer in several important ways.

First, Linformer uses explicit learned projections tied to the sequence length, whereas SA uses an implicit, prototype-based, inputdependent projection. Second, Linformer is motivated by approximating self-attention eficiently, whereas SA is designed as a target-agnostic compression front-end for a downstream target-conditioned reasoning module. Third, and most importantly for our setting, SA produces a reusable user-only representation that can be cached and amortized across targets and requests, which is not a standard goal of low-rank attention approximations.

Thus, the low-rank interpretation should be understood as a representation-level analogy: SA is not simply “Linformer for recommendation,” but a cacheable length-wise compression mechanism that plays a similar dimensionality-reduction role while serving a diferent system purpose.

## A.6 Discussion

From this perspective, SequenceO1 follows a compress-then-reason paradigm: the ultra-long history is first mapped to a compact representation of length �, and target-conditioned reasoning is then performed on this fixed-size representation.

This view helps explain why the framework scales well to 100K histories. The expensive dependence on the raw length � is concentrated in the compression step, which is lightweight, target-agnostic, and reusable. The heavyweight target-conditioned reasoning is then performed only on the compressed representation, whose size is fixed. In this sense, the low-rank perspective is not merely a modeling analogy; it also provides the systems intuition behind SequenceO1: by materializing and caching a compact length-wise representation, we turn ultra-long history modeling into a form of low-rank caching suitable for industrial recommendation workloads.

## B Algorithmic Summary of Sketch Attention

Given history embeddings $\ b X \in \mathbb { R } ^ { n \times d }$ and initial prototypes $P ^ { ( 0 ) } \in \mathbb { R } ^ { k \times d }$ , one SA block proceeds as follows:

(1) Compute prototype–token afinities:

$$
S = \frac { ( P ^ { ( 0 ) } W _ { Q } ) ( X W _ { K } ) ^ { \top } } { \sqrt { d } } .
$$

(2) Normalize over prototypes for each token:

$$
A _ { : , i } = \mathrm { s o f t m a x } _ { \mathrm { p r o t o } } ( S _ { : , i } ) , \qquad \sum _ { j = 1 } ^ { k } A _ { j , i } = 1 .
$$

(3) Aggregate history tokens into sketch tokens:

$$
\widetilde { X } = A X .
$$

(4) If stacked SA is used, apply residual connection, layer normalization, and FFN refinement, then repeat the allocation–aggregation step using the refined sketch state.

The output is a fixed-size sketch $\widetilde { X } \in \mathbb { R } ^ { k \times d }$

## C Why Prototype-Wise Normalization

This section explains why Sketch Attention (SA) normalizes over the prototype dimension rather than the token dimension. The key issue is that our compression module must be target-agnostic: it is computed before seeing the candidate item, cached as a user-only representation, and then reused across many downstream targets and requests. In this setting, the normalization choice determines whether the compressed representation tends to cover the whole history or to select only a small subset of tokens.

A natural alternative to SA is to normalize over the token dimension, yielding weights �(� | �) for each prototype. Under this scheme each prototype attends to a subset of tokens and efectively acts as a selector over the sequence. While this behavior can be useful when the downstream task already specifies what to focus on, it is less suitable for our setting because the compression must be computed before target conditioning.

The problem is that token-wise normalization encourages competition among tokens for each prototype, but does not enforce any corresponding coverage constraint over the full history. As a result, multiple prototypes may collapse onto similar high-salience regions, repeatedly selecting the same small subset of tokens while ignoring others. In a target-agnostic compression setting, this can lead to poor coverage: large portions of the ultra-long history may be underrepresented or discarded before the model has a chance to decide which parts will matter for a particular target.

In contrast, prototype-wise normalization produces weights $p ( j \mid i )$ and enforces a per-token conservation constraint

$$
\sum _ { j = 1 } ^ { k } A _ { j , i } = 1 .
$$

This means that each token distributes its mass across prototypes, rather than each prototype independently selecting tokens. Consequently, every token contributes mass to the compressed representation, and the prototypes collectively aggregate information from the entire history.

This distinction is important for SequenceO1. Because the sketch must support diverse downstream targets and be reused across requests, it should preserve broad user-history coverage rather than prematurely commit to a small selected subset. Prototype-wise normalization encourages this behavior: it turns the sketching step into a token-to-prototype allocation mechanism, which is better aligned with reusable, user-only compression.

From this perspective, token-wise normalization is more naturally suited to target-aware selection, whereas prototype-wise normalization is better suited to target-agnostic summarization. Our design choice reflects the role of SA in the overall framework: SA is not the final reasoning module, but a reusable compression layer whose output will later be consumed by target-conditioned STCA. By preserving coverage early and deferring selective matching to the reasoning stage, SequenceO1 better balances information retention and computationa eficiency.

## C.1 Assignment Patterns and Downstream Interfaces

The analysis in §5.1 uses the maximum normalized weight associated with each token as a descriptive diagnostic of the allocation pattern induced by each normalization rule. Because prototype-wise and token-wise normalization operate over diferent axes, the resulting maxima are not calibrated confidence scores and are not interpreted as directly comparable measures of prototype specialization, overlap, or retrieva quality. Within prototype-wise normalization, a larger maximum corresponds to a more concentrated token-to-prototype allocation, but it does not by itself guarantee diverse coverage or reduced prototype overlap

This property is useful because the SA sketch is not the final reasoning output. It is a reusable intermediate representation that will later be consumed by target-conditioned STCA. These allocation patterns may provide an interface for future extensions, such as retrieving relevant behaviors through selected sketch tokens or applying localized refinement to behavior groups associated with a subset of prototypes. These extensions are not evaluated in the current work; the task-level evidence for prototype-wise normalization comes from the UAUC improvements reported in Table 1.

## D Implementation Details of Action-Side Fusion

Each historical behavior contains both item-side information and action-side information. The action-side SwiGLU fusion used in Table 2 denotes a lightweight gated fusion before sketch construction. Given an item embedding $\mathbf { e } _ { i }$ and an action-type embedding ${ \bf a } _ { i } ,$ the fused behavior embedding can be written abstractly as

$$
\mathbf { x } _ { i } = \mathbf { e } _ { i } + \mathrm { F F N } _ { \mathrm { S w i G L U } } \big ( \big [ \mathbf { e } _ { i } ; \mathbf { a } _ { i } \big ] \big ) ,
$$

where the SwiGLU FFN maps the concatenated features back to the dimension of $\mathbf { \dot { e } } _ { i }$ so that the residual addition is well-defined; the exact production implementation may include additional feature transformations. This fusion allows the same historical item to contribute diferently under diferent feedback actions such as finish, skip, like, or follow. The ablation “w/o action-side SwiGLU fusion” removes this action-aware gated fusion before SA.

## E Iterative Refinement View

This section provides an intuitive interpretation of stacked Sketch Attention (SA) as an iterative allocation–aggregation procedure. The goal is not to claim that SA exactly implements an optimization algorithm such as EM, but rather to highlight a useful analogy: each refinement step alternates between assigning history tokens to prototypes and updating the prototype-side representation by aggregating token information.

Consider one refinement step with current prototype representations � (or, more generally, the current sketch state �e<sup>(ℓ)</sup> ). Using the current prototypes as queries, SA computes soft token-to-prototype assignments

$$
A _ { j , i } = p ( j \mid i ) ,
$$

where each token distributes its mass across prototypes. These assignments determine how the information in the ultra-long history is allocated to the fixed set of sketch slots.

Given the assignments, the updated sketch is obtained by aggregating token embeddings:

$$
\widetilde { X } = A X .
$$

Thus, one SA step can be viewed as first deciding how each token contributes to diferent prototypes, and then recomputing each prototype representation as a weighted summary of the full history.

With stacked SA, this process is repeated multiple times. After one round of allocation and aggregation, the prototype-side representation becomes more informed by the input history. Using this refined representation to compute the next round of assignments allows the model to adjust how tokens are grouped and summarized. In this sense, deeper SA layers act as refinement steps that progressively improve the sketch, rather than as entirely independent transformations.

This view is reminiscent of an EM-like procedure: the soft assignments play a role analogous to a responsibility update, while the aggregation step plays a role analogous to prototype re-estimation. The analogy is only conceptual—SA is trained end-to-end with gradient descent, includes residual connections and feed-forward layers, and is not derived from a probabilistic latent-variable objective. Still, the EM perspective is useful because it explains why stacked SA can improve sketch quality: later layers refine the token-to-prototype allocation using prototypes that already encode information from earlier aggregation steps.

This interpretation also helps motivate why only a small number of refinement steps is suficient in practice. User behavior sequences are highly redundant, and the purpose of SA is not to recover fine-grained token-level structure, but to construct a compact user-level summary that preserves the most salient long-range signals. As a result, substantial gains can often be obtained after only one or two rounds of refinement, after which additional iterations yield diminishing returns. This matches our empirical finding that a small stacked depth $( N _ { \mathrm { s a } } = 2 )$ is already efective for 100K sequence modeling.

## F Error Decomposition

This section provides a simple conceptual view of the approximation error introduced by the compress-then-reason design in SequenceO1. Our goal is not to derive a tight theoretical bound, but to clarify that the overall error can be separated into two sources: (i) information loss caused by compressing the ultra-long history into a fixed-size sketch, and (ii) modeling error incurred when performing target-conditioned reasoning on the compressed representation.

Let $F _ { \mathrm { f u l l } } ( x _ { t } , X )$ denote an ideal target-conditioned predictor that operates directly on the full history �. Let $F _ { \mathrm { c o m p } } ^ { \star } ( x _ { t } , C ( X ) )$ denote the best predictor within a chosen hypothesis class that has access only to the compressed representation �(�). Our method applies a downstream reasoning module $G ( \cdot , \cdot )$ ):

$$
\hat { F } ( x _ { t } , X ) = G ( x _ { t } , C ( X ) ) .
$$

In our setting, �(�) corresponds to the Sketch Attention (SA) sketch, while � corresponds to the subsequent target-conditioned reasoning and fusion modules.

To expose the approximation structure, add and subtract $F _ { \mathrm { c o m p } } ^ { \star } ( x _ { t } , C ( X ) )$ :

$$
\begin{array} { r } { F _ { \mathrm { f u l l } } ( x _ { t } , X ) - \hat { F } ( x _ { t } , X ) = \big ( F _ { \mathrm { f u l l } } ( x _ { t } , X ) - F _ { \mathrm { c o m p } } ^ { \star } ( x _ { t } , C ( X ) ) \big ) + \big ( F _ { \mathrm { c o m p } } ^ { \star } ( x _ { t } , C ( X ) ) - \hat { F } ( x _ { t } , X ) \big ) . } \end{array}
$$

Applying the triangle inequality gives

$$
\| F _ { \mathrm { f u l l } } ( x _ { t } , X ) - \hat { F } ( x _ { t } , X ) \| \leq \underbrace { \| F _ { \mathrm { f u l l } } ( x _ { t } , X ) - F _ { \mathrm { c o m p } } ^ { \star } ( x _ { t } , C ( X ) ) \| } _ { \mathrm { c o m p r e s s i o n ~ e r r o r ~ } } + \underbrace { \| F _ { \mathrm { c o m p } } ^ { \star } ( x _ { t } , C ( X ) ) - \hat { F } ( x _ { t } , X ) \| } _ { \mathrm { r e a s o n i n g ~ e r r o r ~ } } .
$$

The first term measures how much predictive information is lost when replacing the full history � with its compressed representation �(�). This term is controlled by the quality of the sketch: if the ultra-long history is suficiently compressible along the length dimension and the sketch preserves the salient user signals relevant to downstream ranking, then this term can remain small even when �(�) has fixed size.

The second term measures how well the downstream model � can exploit the compressed representation once it is available. Even if the sketch preserves most useful information, insuficient downstream capacity or suboptimal target-conditioned interaction can still lead to prediction error. In SequenceO1, this term is addressed by applying STCA over both the recent 10K sufix and the ultra-long sketch, followed by lightweight fusion.

This decomposition helps explain the design principle behind SequenceO1. SA is used to control the compression error by constructing a compact but expressive user-only summary of the 100K history, while the subsequent STCA-based reasoning stack controls the reasoning error by allocating model capacity to target-aware interaction rather than to repeatedly scanning the raw ultra-long sequence. From this perspective, the efectiveness of the overall framework depends on balancing these two terms: the sketch must be compact enough for eficiency and reuse, yet informative enough that downstream reasoning over the sketch remains highly predictive

## G Complexity Derivation (FLOPs)

This appendix derives the FLOP formulas used in our complexity analysis under the notation of the main paper; $L _ { s }$ denotes the sketch length � in the main text. SA is single-head and operates on width � = 128. STCA is multi-head with ℎ = 16 heads and head dimension $d _ { h } = 6 4$ hence the STCA model width is

$$
D \triangleq h \cdot d _ { h } = 1 0 2 4 .\tag{24}
$$

All FLOPs are reported under the same GEMM-dominant counting convention.

## G.1 Counting convention

We report complexity in terms of FLOPs dominated by matrix multiplications (GEMMs). For a matrix product $U \in \mathbb { R } ^ { m \times r }$ and $V \in \mathbb { R } ^ { r \times p }$

$$
\operatorname { F L O P s } ( U V ) \triangleq 2 m r p ,\tag{25}
$$

counting one multiply and one add per inner-dimension element. We ignore lower-order elementwise operations such as softmax exponentials, LayerNorm, bias adds, masking, and activation function costs.

SwiGLUFFN FLOPs. For a SwiGLU FFN applied to � tokens with model width � and intermediate width $w _ { f }$

$$
\mathrm { F F N } ( U ) = \big ( \mathrm { s w i s h } ( U W _ { 1 } ) \odot ( U W _ { g } ) \big ) W _ { 2 } ,\tag{26}
$$

with $W _ { 1 } , W _ { q } \in \mathbb { R } ^ { w \times w _ { f } }$ and $W _ { 2 } \in \mathbb { R } ^ { w _ { f } \times w }$ , the GEMM FLOPs are

$$
\mathrm { F L O P } _ { \mathsf { S } _ { \mathrm { S w i G L U } } } ( T ; w , w _ { f } ) = 2 T w w _ { f } + 2 T w w _ { f } + 2 T w _ { f } w = 6 T w w _ { f } .\tag{27}
$$

In our setting we use $w _ { f } = 2 w$ (both for SA and STCA FFNs), hence

$$
\mathrm { F L O P s } _ { \mathrm { S w i G L U } } ( T ; w , 2 w ) = 1 2 T w ^ { 2 } .\tag{28}
$$

## G.2 STCA FLOPs (with single-query reordering)

STCA uses stacked single-query target-to-history cross attention. Let the history length be � and the STCA model width be �. We use ℎ heads with per-head dimension $d _ { h } = D / h$ , and denote the number of STCA layers by $N _ { \mathrm { { s t c a } } }$ . In addition to attention and the query-side FFN, each STCA layer includes its own history-side pre-processing SwiGLU FFN applied to all � history tokens before that layer interacts with the target (i.e., a target-independent history transform), with intermediate width 2� and output width �.

History-side pre-FFN (per layer). In each layer, the history-side SwiGLU is applied to $\boldsymbol { X } \in \mathbb { R } ^ { L \times D }$

$$
\mathrm { F L O P s } _ { \mathrm { h i s t - F F N } } ( L ; D ) = \mathrm { F L O P s } _ { \mathrm { S w i G L U } } ( L ; D , 2 D ) = 1 2 L D ^ { 2 } .\tag{29}
$$

G.2.1 Single-query atention reordering and FLOPs analysis. With exactly one query per layer, let $X \in \mathbb { R } ^ { L \times D } , q \in \mathbb { R } ^ { 1 \times D }$ , and $d _ { h } { = } D / h$ . The standard cross-attention form is

$$
\mathrm { A t t n } ( q , X ) = \mathrm { s o f t m a x } \bigg ( \frac { ( q W _ { Q } ) ( X W _ { K } ) ^ { \top } } { \sqrt { d _ { h } } } \bigg ) \cdot ( X W _ { V } ) ,\tag{30}
$$

which projects all � tokens twice and materializes the length-� tensors $X W _ { K }$ and $X W _ { V }$

Reordering. We can reorder the computation to remove the length-� projections:

$$
\begin{array} { r } { u = ( q W _ { Q } ) W _ { K } ^ { \top } \in \mathbb { R } ^ { 1 \times D } , \quad \alpha = \mathrm { s o f t m a x } \Big ( \frac { u X ^ { \top } } { \sqrt { d _ { h } } } \Big ) \in \mathbb { R } ^ { 1 \times L } , } \end{array}
$$

and then compute

$$
o \ = \ ( \alpha X ) \ W _ { V } \in \mathbb { R } ^ { 1 \times d _ { h } } , \quad W _ { Q } , W _ { K } , W _ { V } \in \mathbb { R } ^ { D \times d _ { h } } .
$$

Equivalently,

$$
\begin{array} { r } { \mathrm { A t t n } ( q , X ) \ = \ \Bigl ( \mathrm { s o f t m a x } \bigl ( \frac { ( ( q W _ { Q } ) W _ { K } ^ { \top } ) X ^ { \top } } { \sqrt { d _ { h } } } \bigr ) X \Bigr ) W _ { V } \ = \ ( \alpha X ) W _ { V } . } \end{array}
$$

FLOPs (GEMM-dominant). We count GEMMs and ignore softmax/LN/activation overhead. Per head, the reordered path costs

$$
\underbrace { 2 D d _ { h } } _ { q W _ { Q } } + \underbrace { 2 D d _ { h } } _ { ( q W _ { Q } ) W _ { K } ^ { \top } } + \underbrace { 2 L D } _ { u X ^ { \top } } + \underbrace { 2 L D } _ { \alpha X } + \underbrace { 2 D d _ { h } } _ { ( \alpha X ) W _ { V } } ,
$$

so across ℎ heads the attention FLOPs per layer are

$$
{ \mathrm { F L O P s } } _ { { \mathrm { a t t n - r e o r d e r } } } ( L ; D , h ) = \underbrace { 4 L D h } _ { \mathrm { l e n g t h - d e p e n d e n t } } + \underbrace { 6 D ^ { 2 } } _ { \mathrm { h e a d p r o j e c t i o n s } } .\tag{31}
$$

Compared with the naïve path that explicitly forms $( X W _ { K } , X W _ { V } )$ (length-dependent cost $\approx 4 L D ^ { 2 } )$ , the reordered path removes the $O ( L D ^ { 2 } )$ projections and replaces them with �(��ℎ) weighted reductions, reducing the length-dependent FLOPs by a factor of approximately $d _ { h } = D / h$

No cross-request KV sharing under reordering. Importantly, reordering avoids materializing the $L { \times } d _ { h }$ intermediates $( X W _ { K } , X W _ { V } )$ , so there are no target-independent per-layer KV projections to cache/share across requests. The remaining �-dependent terms $( u X ^ { \top }$ and ��) depend on the query and thus are inherently per-target.

Other per-layer query-side transforms. Following our main accounting convention, we further include: (i) an output projection $W _ { O } \in \mathbb { R } ^ { D \times D }$ with FLOPs $2 D ^ { 2 }$ , and (ii) a one-token query-side SwiGLU FFN (intermediate width 2�) with FLOPs 12�<sup>2</sup>.

Per-layer STCA FLOPs (optimized). Combining Eq. (31) with the output projection and query FFN, the per-layer per-target FLOPs are

$$
\operatorname { F L O P s } _ { \operatorname { S T C A - l a y e r } } ( L ; D , h ) = \operatorname { F L O P s } _ { \operatorname { a t t n - r e o r d e r } } ( L ; D , h ) + 2 D ^ { 2 } + 1 2 D ^ { 2 } = 4 L D h + 2 0 D ^ { 2 } .\tag{32}
$$

Total STCA FLOPs (per target). Including the independent history-side pre-FFN in every layer, the total per-target STCA FLOPs are

$$
\mathrm { F L O P s } _ { \mathrm { S T C A } } ( L ; N _ { \mathrm { s t c a } } , D , h ) = N _ { \mathrm { s t c a } } \left( 1 2 L D ^ { 2 } + 4 L D h + 2 0 D ^ { 2 } \right) .\tag{33}
$$

## G.3 Sketch Attention (SA) FLOPs (single-head)

SA compresses an ultra-long history $\boldsymbol { X } \in \mathbb { R } ^ { L \times d }$ into a fixed-size sketch $\widetilde { X } \in \mathbb { R } ^ { L _ { s } \times d } \left( \ S 3 . 1 \right)$ , where $L _ { s }$ denotes the number of prototypes $( { \mathrm { i . e . } }$ , the sketch length). We assume SA is single-head in our instantiation and uses width $d = 1 2 8$

One SA layer. Let prototypes be $P \in \mathbb { R } ^ { L _ { s } \times d }$ and projections �<sub>�</sub>, $W _ { K } \in \mathbb { R } ^ { d \times d } \colon$

$$
Q = P W _ { Q } \in \mathbb { R } ^ { L _ { s } \times d } , \qquad K = X W _ { K } \in \mathbb { R } ^ { L \times d } .\tag{34}
$$

The afinity matrix is $S = \frac { Q K ^ { \top } } { \sqrt { d } } \in \mathbb { R } ^ { L _ { s } \times L }$ and the assignment $A = \mathrm { s o f t m a x } _ { \mathrm { p r o t o } } ( S ) \in \mathbb { R } ^ { L _ { s } \times L }$ . The sketch is

$$
\widetilde { X } = A X \in \mathbb { R } ^ { L _ { s } \times d } .\tag{35}
$$

SA layer FLOPs (GEMM-dominant). We count:

• Key projection: $K = X W _ { K }$

$$
\mathrm { F L O P s } _ { K } ^ { \mathrm { S A } } ( L ; d ) = 2 L d ^ { 2 } .\tag{36}
$$

• Prototype projection: $Q = P W _ { Q } $

$$
\mathrm { F L O P s } _ { Q } ^ { \mathrm { S A } } ( L _ { s } ; d ) = 2 L _ { s } d ^ { 2 } .\tag{37}
$$

• Afinity GEMM: $S = Q K ^ { \top }$

$$
\mathrm { F L O P s } _ { \mathrm { s c o r e } } ^ { \mathrm { S A } } ( L , L _ { s } ; d ) = 2 L _ { s } L d .\tag{38}
$$

• Aggregation GEMM: $\widetilde { X } = A X :$

$$
\mathrm { F L O P s } _ { \mathrm { a g g } } ^ { \mathrm { S A } } ( L , L _ { s } ; d ) = 2 L _ { s } L d .\tag{39}
$$

• Sketch FFN: SwiGLU applied to $L _ { s }$ sketch tokens with intermediate width 2�:

$$
\mathrm { F L O P s } _ { \mathrm { F F N } } ^ { \mathrm { S A } } ( L _ { s } ; d ) = \mathrm { F L O P s } _ { \mathrm { S w i G L U } } ( L _ { s } ; d , 2 d ) = 1 2 L _ { s } d ^ { 2 } .\tag{40}
$$

Summing these yields the per-layer SA FLOPs:

$$
\mathrm { F L O P s } _ { \mathrm { S A - l a y e r } } ( L , L _ { s } ; d ) = 2 L d ^ { 2 } + 2 L _ { s } d ^ { 2 } + 4 L _ { s } L d + 1 2 L _ { s } d ^ { 2 } = 2 L d ^ { 2 } + 1 4 L _ { s } d ^ { 2 } + 4 L _ { s } L d .\tag{41}
$$

Stacked SA.. With $N _ { \mathrm { s a } }$ stacked SA layers (§3.3), the total $\mathrm { S A }$ cost is

$$
\mathrm { F L O P s } _ { \mathrm { S A } } ( L ; N _ { \mathrm { s a } } , L _ { s } , d ) = N _ { \mathrm { s a } } \cdot \mathrm { F L O P s } _ { \mathrm { S A } \cdot \mathrm { l a y e r } } ( L , L _ { s } ; d ) .\tag{42}
$$

## G.4 Single-branch SA+STCA on the sketch: cache miss/hit and expected cost

We analyze the ultra-long single branch used in Fig. 1, consisting of SA sketching over � tokens followed by an $N _ { \mathrm { { s t c a } } }$ -layer STCA over the fixed-size sketch of length � . Common costs outside this branch—including the recent-10K STCA branch, dense-ranker computation, input embedding and action-feature preprocessing, and final two-branch fusion—are omitted from both alternatives. This FLOP analysis is conservative with respect to system savings: it does not count the raw-length feature materialization, storage, and communication that are also bypassed when a cached sketch is available. Since SA uses width � while STCA uses width �, we apply a lightweight linear adapter to map the sketch width. The target-side projection $\phi _ { t } ( \mathbf { x } _ { t } )$ is candidate-side and independent of the history length, so it is omitted from the sequence-branch FLOP accounting. Here $W _ { A }$ denotes the width adapter and is distinct from the SA assignment matrix �:

$$
\begin{array} { r } { \widehat { X } = \widetilde { X } W _ { A } , \qquad W _ { A } \in \mathbb { R } ^ { d \times D } , \qquad \widehat { X } \in \mathbb { R } ^ { L _ { s } \times D } . } \end{array}\tag{43}
$$

Only the �-width SA sketch is stored in the persistent cache; the adapter output is not persistently cached across groups. Its cost is therefore incurred on both cache hits and misses, although the target-independent adapted sketch is materialized once and reused within an MRLB group.

Adapter FLOPs.

$$
\mathrm { F L O P s } _ { \mathrm { a d a p t } } ( L _ { s } ; d , D ) = 2 L _ { s } d D .\tag{44}
$$

Hit and miss FLOPs. On a cache miss, we compute SA, adapt the sketch, and then run sketch- $\operatorname { \cdot } S \operatorname { T C A ; }$ on a cache hit, we reuse the cached sketch and run the same adapt+STCA:

$$
\mathrm { F L O P s } _ { \mathrm { m i s s } } ( L ) = \mathrm { F L O P s } _ { \mathrm { S A } } ( L ; N _ { \mathrm { s a } } , L _ { s } , d ) + \mathrm { F L O P s } _ { \mathrm { a d a p t } } ( L _ { s } ; d , D ) + \mathrm { F L O P s } _ { \mathrm { S T C A } } ( L _ { s } ; N _ { \mathrm { s t c a } } , D , h ) ,\tag{45}
$$

$$
\mathrm { F L O P s } _ { \mathrm { h i t } } = \mathrm { F L O P s } _ { \mathrm { a d a p t } } ( L _ { s } ; d , D ) + \mathrm { F L O P s } _ { \mathrm { S T C A } } ( L _ { s } ; N _ { \mathrm { s t c a } } , D , h ) .\tag{46}
$$

Expected FLOPs under hit rate. Let $p _ { \mathrm { h i t } } \in [ 0 , 1 ]$ be the cache hit probability. The expected FLOPs are

$$
\begin{array} { r } { \mathbb { E } [ \mathrm { F L O P s } ] ( L ) = \boldsymbol { p } _ { \mathrm { h i t } } \cdot \mathrm { F L O P s } _ { \mathrm { h i t } } + ( 1 - \boldsymbol { p } _ { \mathrm { h i t } } ) \cdot \mathrm { F L O P s } _ { \mathrm { m i s s } } ( L ) = \mathrm { F L O P s } _ { \mathrm { h i t } } + ( 1 - \boldsymbol { p } _ { \mathrm { h i t } } ) \cdot \mathrm { F L O P s } _ { \mathrm { S A } } ( L ; N _ { \mathrm { s a } } , L _ { s } , d ) . } \end{array}\tag{47}
$$

## G.5 Hyperparameter instantiation for analysis

We instantiate the formulas above with the following settings:

• SA width: � = 128 (single-head SA and its SwiGLU uses intermediate width 2�).

• STCA heads: $h = 1 6 , d _ { h } = 6 4$ , hence $D = h d _ { h } = 1 0 2 4$ (STCA SwiGLU uses intermediate width 2�).

• $\mathbf { S } \mathbf { A } \colon N _ { \mathrm { s a } } = 2$ stacked SA layers, single-head; number of prototypes / sketch length $L _ { s } = 1 0 2 4$

• $\mathbf { \mathrm { 3 T C A } } \colon N _ { \mathrm { s t c a } } = 4 \mathrm { l a y e r s } .$

• Cache hit rate: for training-side analysis we use $p _ { \mathrm { h i t } } ^ { \mathrm { t r a i n } } = 0 . 5 ;$ for inference/serving-side analysis we use $p _ { \mathrm { h i t } } ^ { \mathrm { i n f e r } } = 0 . 6$

• Reuse ratio: for training-side MRLB amortization we use $R _ { \mathrm { t r a i n } } = 4 0 ;$ ; for inference/serving-side amortization across consecutive requests we use $R _ { \mathrm { i n f e r } } = 3 0 0$

• Sequence lengths: � ∈ {1K, 2K, 10K, 20K, 50K, 100K} in our plots.

Closed forms. With the above settings, we obtain:

$$
\mathrm { F L O P s } _ { \mathrm { S T C A } } ( L ; N _ { \mathrm { s t c a } } , D , h ) = N _ { \mathrm { s t c a } } \left( 1 2 L D ^ { 2 } + 4 L D h + 2 0 D ^ { 2 } \right) ,\tag{48}
$$

$$
\mathrm { F L O P s } _ { \mathrm { S A } } ( L ; N _ { \mathrm { s a } } , L _ { s } , d ) = N _ { \mathrm { s a } } \left( 2 L d ^ { 2 } + 1 4 L _ { s } d ^ { 2 } + 4 L _ { s } L d \right) ,\tag{49}
$$

$$
\mathrm { F L O P s } _ { \mathrm { h i t } } = \mathrm { F L O P s } _ { \mathrm { a d a p t } } ( L _ { s } ; d , D ) + \mathrm { F L O P s } _ { \mathrm { S Y C A } } ( L _ { s } ; N _ { \mathrm { s t e n } } , D , h ) , \qquad \mathrm { F L O P s } _ { \mathrm { m i s } } ( L ) = \mathrm { F L O P s } _ { \mathrm { S A } } ( L ; N _ { \mathrm { s t } } , L _ { s } , d ) + \mathrm { F L O P s } _ { \mathrm { h i t } } ,\tag{50}
$$

$$
\begin{array} { r } { \mathbb { E } \big [ \mathrm { F L O P s } \big ] \big ( L ; \rho _ { \mathrm { h i t } } \big ) = \mathrm { F L O P s } _ { \mathrm { h i t } } + \big ( 1 - \rho _ { \mathrm { h i t } } \big ) \cdot \mathrm { F L O P s } _ { \mathrm { S A } } \big ( L ; N _ { \mathrm { s a } } , L _ { s } , d \big ) . } \end{array}\tag{51}
$$

Intuitive comparison and dominant terms. For STCA-only on length �, the dominant cost consists of the per-layer history-side pre-FFN term $1 2 L D ^ { 2 }$ and the per-layer query–history interaction term 4��ℎ under single-query reordering (Eq. (48)). Thus, STCA-only scales linearly in both � and $N _ { \mathrm { { s t c a } } } ,$ , with the large model width � appearing in both the history-side transform and the per-layer query-side transforms.

For SA, the dominant term is the prototype–token interaction $4 L _ { s } L d$ in each SA layer (from $Q K ^ { \top }$ and ��), yielding $O ( N _ { \mathrm { s a } } L _ { s } L d )$ overall (Eq. (49)). Therefore, SA also scales linearly in $L ,$ with slope controlled by the fixed sketch length $L _ { s }$ and the smaller SA width �.

Why both are �(�) but behave very diferently in practice. Although both SA and STCA have linear dependence on the input length �, they allocate modeling capacity very diferently. SA is designed as a lightweight length-wise compressor: it operates at a much smaller width � and summarizes the � history tokens into a fixed-size sketch of length $L _ { s } ,$ which is consistent with the empirically observed compressibility of long behavioral sequences along the length dimension. In contrast, STCA is the heavyweight target-conditioned reasoning module: it uses a substantially larger width � (and multi-head structure) to model fine-grained interactions between the target and the (raw or compressed) history. This design intentionally places expensive representational power in the target–history interaction stage, where precise matching is required, while keeping the long-history compression stage eficient.

For SA+STCA, sketch-STCA runs on the fixed length $L _ { s } .$ . On a cache hit, the cost is independent of � and consists of the (uncached) adapter plus STCA on length $L _ { s } ( \mathrm { E q . } ( 4 6 ) )$ ; on a cache miss, the additional �-dependent cost comes entirely from SA (Eq. (45)). With hit rate $\phi _ { \mathrm { h i t } } ,$ the expected compute equals the hit cost plus a reduced linear-in-� component weighted by $\left( 1 - p _ { \mathrm { h i t } } \right) \left( \mathrm { E q . } \left( 4 7 \right) \right)$ .

## G.6 MRLB Amortization and Reuse Ratio

We further analyze the per-target FLOPs under Multi-Request Level Batching (MRLB), where � denotes the number of target instances across grouped requests that share the same user-side computation. Within an MRLB group, target-independent (user-only) computations can be executed once and reused across these � target instances. Thus, if a computation can be decomposed as

$$
\mathrm { F L O P s } _ { \mathrm { t o t a l } } = \mathrm { F L O P s } _ { \mathrm { s h a r e d } } + \mathrm { F L O P s } _ { \mathrm { p e r - t a r g e t } } ,\tag{52}
$$

then the amortized per-target FLOPs under MRLB are

$$
\mathrm { F L O P s } _ { \mathrm { M R L B } } ( R ) = \frac { \mathrm { F L O P s } _ { \mathrm { s h a r e d } } } { R } + \mathrm { F L O P s } _ { \mathrm { p e r - t a r g e t } } .\tag{53}
$$

STCA-only under MRLB (single-query reordering). For STCA with model width $D ,$ , ℎ heads, and $N _ { \mathrm { { s t c a } } }$ layers, we adopt the single-query attention reordering described in §G.2.1. Under this optimized implementation, the $L \times d _ { h }$ intermediates $( X W _ { K } , X W _ { V } )$ are not materialized, i.e., there are no explicit history-side per-layer $K / V$ projections that can be cached or reused across requests. Consequently, within MRLB the only strictly target-independent components we amortize are the per-layer history-side pre-FFNs.

Concretely, the shared FLOPs are

$$
\mathrm { F L O P s } _ { \mathrm { S T C A , s h a r e d } } ( L ) = 1 2 N _ { \mathrm { s t c a } } L D ^ { 2 } ,\tag{54}
$$

and the remaining per-target FLOPs are the per-layer reordered attention plus query-side transforms. Using Eq. (31) and Eq. (32), we have

$$
\mathrm { F L O P s } _ { \mathrm { S T C A , p e r - t a r g e t } } ( L ) = N _ { \mathrm { s t c a } } \left( 4 L D h + 2 0 D ^ { 2 } \right) .\tag{55}
$$

Therefore, the amortized per-target STCA-only FLOPs under MRLB are

$$
\mathrm { F L O P s } _ { \mathrm { S T C A } } ^ { \mathrm { M R L B } } ( L ; N _ { \mathrm { s t c a } } , D , h , R ) = \frac { 1 2 N _ { \mathrm { s t c a } } L D ^ { 2 } } { R } + N _ { \mathrm { s t c a } } \left( 4 L D h + 2 0 D ^ { 2 } \right) .\tag{56}
$$

SA+Adapter+STCA on the sketch under MRLB (hit/miss). We analyze the ultra-long single branch consisting of SA sketching over length � (SA width $d ) ,$ an adapter $d \to D _ { : }$ , and sketch-STCA over length $L _ { s }$ (STCA width �). Under MRLB, we assume: (i) all SA computation is reusable within the group (user-only); (ii) the adapter output is target-independent and thus reusable within the group; and (iii) sketch-STCA uses the same single-query reordering as STCA, so there are no per-layer sketch-side �/� $K / V$ projections to reuse across requests; only the per-layer sketch-side pre-FFNs are reusable.

Let the adapter FLOPs be Eq. (44). For sketch-STCA at length $L _ { s } ,$ , the reusable and per-target parts follow Eq. (54)–(55) by substituting $L \gets L _ { s } \mathrm { : }$

$$
\mathrm { F L O P s } _ { \mathrm { s k \mathrm { - } S T C A , s h a r e d } } = 1 2 N _ { \mathrm { s t c a } } L _ { s } D ^ { 2 } ,\tag{57}
$$

$$
\mathrm { F L O P s } _ { \mathrm { s k - S T C A , p e r - t a r g e t } } = N _ { \mathrm { s t c a } } \left( 4 L _ { s } D h + 2 0 D ^ { 2 } \right) .\tag{58}
$$

On a cache hit, SA is skipped but we still perform adapter + sketch-STCA:

$$
\mathrm { F L O P s } _ { \mathrm { h i t } } ^ { \mathrm { M R L B } } ( R ) = \frac { \mathrm { F L O P s } _ { \mathrm { a d a p t } } ( L _ { s } ; d , D ) + \mathrm { F L O P s } _ { \mathrm { s k } \cdot \mathrm { S T C A } , \mathrm { s h a r e d } } } { R } + \mathrm { F L O P s } _ { \mathrm { s k } \cdot \mathrm { S T C A } , \mathrm { p e r } \cdot \mathrm { t a r g e t } } .\tag{59}
$$

On a cache miss, SA is computed once and reused across the � targets:

$$
\mathrm { F L O P s } _ { \operatorname* { m i s s } } ^ { \mathrm { M R L B } } ( L ; R ) = \frac { \mathrm { F L O P s } _ { \mathrm { S A } } ( L ; N _ { \mathrm { S a } } , L _ { s } , d ) + \mathrm { F L O P s } _ { \mathrm { a d a p t } } ( L _ { s } ; d , D ) + \mathrm { F L O P s } _ { \mathrm { s i s } \cdot \mathrm { S T C A } , \mathrm { s h a r e d } } } { R } + \mathrm { F L O P s } _ { \mathrm { s i s } \cdot \mathrm { S T C A } , \mathrm { p e r } \cdot \mathrm { t a r g e t } } .\tag{60}
$$

Expected FLOPs with cache hit rate under MRLB.. Here �<sub>hit</sub> denotes the probability that the cached user sketch is available for an MRLB group; the corresponding hit or miss is shared by the � target instances in that group. The expected per-target FLOPs become

$$
\mathbb { E } \left[ \mathrm { F L O P s } ^ { \mathrm { A W B } } \right] ( L ; R , \rho _ { \mathrm { f i n t } } ) = \mathrm { F L O P s } _ { \mathrm { s h i s : S T C A } _ { \mathrm { f i e r } } \mathrm { r : ~ r a n g e r } } + \frac { \mathrm { F L O P s } _ { \mathrm { a d a p e } } ( L _ { s } ; d , D ) + \mathrm { F L O P s } _ { \mathrm { a d s : S T C A , s i n c e d } } + ( 1 - \rho _ { \mathrm { f i n t } } ) \cdot \mathrm { F L O P s } _ { \mathrm { S A } } ( L ; N _ { \mathrm { a s } } , L _ { s } , d ) } { R } .\tag{61}
$$

## G.7 Cache-Hit Sensitivity and Miss Cost

Equations (47) and (61) make the dependence on the cache hit rate explicit. Without MRLB, a cache miss adds exactly the SA sketching cost:

$$
\mathrm { F L O P s } _ { \mathrm { m i s s } } ( L ) - \mathrm { F L O P s } _ { \mathrm { h i t } } = \mathrm { F L O P s } _ { \mathrm { S A } } ( L ; N _ { \mathrm { s a } } , L _ { s } , d ) ,\tag{62}
$$

and the expected cost changes linearly with $1 - p _ { \mathrm { h i t } }$ <sub>t</sub>:

$$
\mathbb { E } \big [ \mathrm { F L O P s } \big ] ( L ; \boldsymbol { \cdot } \boldsymbol { p } _ { \mathrm { h i t } } ) = \mathrm { F L O P s } _ { \mathrm { h i t } } + ( 1 - \boldsymbol { p } _ { \mathrm { h i t } } ) \mathrm { F L O P s } _ { \mathrm { S A } } ( L ; N _ { \mathrm { s a } } , L _ { s } , d ) .\tag{63}
$$

Under MRLB, the miss penalty is further amortized by the reuse ratio �:

$$
\mathrm { F L O P s } _ { \mathrm { m i s s } } ^ { \mathrm { M R L B } } ( L ; R ) - \mathrm { F L O P s } _ { \mathrm { h i t } } ^ { \mathrm { M R L B } } ( R ) = \frac { \mathrm { F L O P s } _ { \mathrm { S A } } ( L ; N _ { \mathrm { s a } } , L _ { s } , d ) } { R } .\tag{64}
$$

Therefore, decreasing the hit rate from $\mathcal { P } 1$ to $\mathcal { P } 2$ increases the amortized per-target cost by

$$
\Delta \mathrm { F L O P s } ^ { \mathrm { M R L B } } = \frac { ( \phi _ { 1 } - \phi _ { 2 } ) \mathrm { F L O P s } _ { \mathrm { S A } } ( L ; N _ { \mathrm { s a } } , L _ { s } , d ) } { R } .\tag{65}
$$

The all-miss case corresponds to $p _ { \mathrm { h i t } } = 0$

$$
\mathbb { E } \big [ \mathrm { F L O P s } ^ { \mathrm { M R L B } } \big ] ( L ; R , 0 ) = \mathrm { F L O P s } _ { \mathrm { s h e y C A , p e r - t a r g e t } } + \frac { \mathrm { F L O P s } _ { \mathrm { a l a p t } } ( L _ { s } ; d , D ) + \mathrm { F L O P s } _ { \mathrm { s } \bot \mathrm { s } \mathrm { T C A } , \mathrm { s t a r e d } } + \mathrm { F L O P s } _ { \mathrm { S A } } ( L ; N _ { \mathrm { s a } } , L _ { s } , d ) } { R } .\tag{66}
$$

This analysis clarifies the role of caching. Cache hits remove the raw-length SA sketching cost and, at the system level, bypass raw-history feature materialization, storage, and communication; cache misses reintroduce only the lightweight user-only SA computation. In both cases, the heavyweight target-conditioned STCA branch operates on the fixed sketch length $L _ { s }$ rather than the raw length �. Thus, lower hit rates reduce the amortization benefit smoothly, but do not turn SequenceO1 back into target-conditioned reasoning over the raw 100K history.

Discussion. With single-query reordering, STCA removes the explicit $O ( L D ^ { 2 } )$ history-side $K / V$ projections and thus cannot amortize them via MRLB; only the target-independent per-layer pre-FFN term $1 2 N _ { \mathrm { s t c a } } L D ^ { 2 }$ is reduced by a factor of �. The remaining per-target attention interaction scales as $O ( N _ { \mathrm { s t c a } } L D h )$ and is query-dependent. In contrast, SA+STCA can amortize the entire SA sketching stage by �, and SA’s dominant term scales as $O ( N _ { \mathrm { s a } } L _ { s } L d )$ with a much smaller width �. This further highlights the compress-then-reason design: long-history processing is pushed into a lightweight, highly reusable stage (SA), while the heavyweight capacity (large �) is concentrated in target-conditioned interaction (STCA) where finer modeling is required.

FLOP-ratio comparison at 100K: without MRLB, training, and inference. We quantify the relative compute gap between the direct-STCA and SequenceO1 ultra-long branches at the 100K regime using FLOP ratios under the same GEMM-only counting convention (lower-order ops ignored). Let the STCA history length be $L = 1 0 0 \mathrm { K }$ , the sketch length be $L _ { s } = 1 0 2 4$ , the STCA width be $D = h d _ { h } = 1 0 2 4$ with $h = 1 6$ heads and $N _ { \mathrm { { s t c a } } } = 4$ layers, and the SA width be $d = 1 2 8$ with $N _ { \mathrm { s a } } = 2$ layers. We use a training-side cache TTL of 3h with hit rate $p _ { \mathrm { h i t } } ^ { \mathrm { t r a i n } } = 0 . 5$ , an inference/serving-side cache TTL of 1h with hit rate $p _ { \mathrm { h i t } } ^ { \mathrm { i n f e r } } = 0 . 6$ , training-side reuse ratio $R _ { \mathrm { t r a i n } } = 4 0$ , and inference/serving-side reuse ratio $R _ { \mathrm { i n f e r } } = 3 0 0$

Without MRLB (reuse ratio 1). Under single-query reordering (§G.2.1), STCA-only FLOPs at length � are

$$
\mathrm { F L O P s } _ { \mathrm { S T C A } } ( L ) = N _ { \mathrm { s t c a } } ( 1 2 L D ^ { 2 } + 4 L D h + 2 0 D ^ { 2 } ) .\tag{67}
$$

SequenceO1 uses SA sketching (Eq. (49)), an uncached adapter (Eq. (44)), and sketch-STCA at length $L _ { s }$ using the same reordering:

$$
\mathrm { F L O P s } _ { \mathrm { S e q O 1 , h i t } } = \mathrm { F L O P s } _ { \mathrm { a d a p t } } ( L _ { s } ; d , D ) + \mathrm { F L O P s } _ { \mathrm { S T C A } } ( L _ { s } ) ,\tag{68}
$$

$$
\mathrm { F L O P s } _ { \mathrm { S e q O 1 } , \mathrm { m i s s } } ( L ) = \mathrm { F L O P s } _ { \mathrm { S A } } ( L ; N _ { \mathrm { s a } } , L _ { s } , d ) + \mathrm { F L O P s } _ { \mathrm { S e q O 1 } , \mathrm { h i t } } ,\tag{69}
$$

$$
\mathbb { E } [ \mathrm { F L O P s } ] _ { \mathrm { { S e q O 1 } } } ( L ; p _ { \mathrm { h i t } } ) = \mathrm { { F L O P s } _ { \mathrm { S e q O 1 , h i t } } + ( 1 - \it { p _ { \mathrm { h i t } } } ) \cdot \mathrm { { F L O P s } _ { \mathrm { S A } } ( \it { L ; N _ { \mathrm { { s a } } } , \it { L _ { s } } , \it { d } ) . } } }\tag{70}
$$

At $L = 1 0 0 \mathrm { K }$ , plugging in the training-side cache hit rate $p _ { \mathrm { h i t } } ^ { \mathrm { t r a i n } } = 0 . 5$ yields

$$
\mathrm { S T C A } \approx 5 0 5 9 . 4 6 ~ \mathrm { G F L O P s / t a r g e t } , \quad \mathrm { S e q O 1 _ { e x p , t r a i n - h i t } \approx 1 0 8 . 1 0 ~ G F L O P s / t a r g e t } ,
$$

so the expected FLOP ratio without MRLB under the training-side hit-rate assumption is

$$
\mathrm { G a p } _ { \mathrm { n o - M R L B } } ( 1 0 0 \mathrm { K } ) \triangleq \frac { \mathrm { F L O P } _ { \mathrm { S S T C A } } ( 1 0 0 \mathrm { K } ) } { \mathbb { E } [ \mathrm { F L O P s } ] _ { \mathrm { S e q O } 1 } ( 1 0 0 \mathrm { K } ; p _ { \mathrm { h i t } } ^ { \mathrm { t r a i n } } ) } \approx \frac { 5 0 5 9 . 4 6 } { 1 0 8 . 1 0 } \approx 4 6 . 8 0 \times .\tag{71}
$$

Training with MRLB.. With reordering, STCA has no materialized per-layer $X W _ { K } / X W _ { V }$ to reuse; under MRLB only the per-layer history side pre-FFNs are amortized:

$$
\mathrm { F L O P s } _ { \mathrm { S T C A } } ^ { \mathrm { M R L B } } ( L ; R ) = \frac { 1 2 N _ { \mathrm { s t c a } } L D ^ { 2 } } { R } + N _ { \mathrm { s t c a } } ( 4 L D h + 2 0 D ^ { 2 } ) .\tag{72}
$$

For SequenceO1, we amortize all SA computation and the adapter, and in sketch-STCA we amortize only its per-layer pre-FFNs:

$$
\mathbb { E } [ \mathrm { F L O P s } ] _ { \mathrm { s e q } 0 1 } ^ { \mathrm { M R B } } ( L ; R , j _ { \mathrm { n i t } } ) = N _ { \mathrm { s t a } } ( 4 L _ { s } D h + 2 0 D ^ { 2 } ) + \frac { \mathrm { F L O P s } _ { \mathrm { a d a p t } } ( L _ { s } ; d , D ) + 1 2 N _ { \mathrm { s t a } } L _ { s } D ^ { 2 } + ( 1 - j _ { \mathrm { n i t } } ) \cdot \mathrm { F L O P s } _ { \mathrm { S A } } ( L ; N _ { \mathrm { s t a } } , L _ { s } , d ) } { R } .\tag{73}
$$

At � = 100K, $R _ { \mathrm { t r a i n } } = 4 0 $ , and $p _ { \mathrm { h i t } } ^ { \mathrm { t r a i n } } = 0 . 5$ , we obtain

$$
\mathrm { S T C A + M R L B _ { t r a i n } \approx 1 5 2 . 1 3 ~ G F L O P s / t a r g e t , } \quad \mathrm { S e q O 1 } + M R L B _ { t r a i n } \approx 3 . 0 4 6 ~ \mathrm { G F L O P s / t a r g e t } ,
$$

so the expected training-side FLOP ratio becomes

$$
\mathrm { G a p } _ { \mathrm { t r a i n } } ( 1 0 0 \mathrm { K } ) \triangleq \frac { \mathrm { F L O P s } _ { \mathrm { S T C A } } ^ { \mathrm { M R L B } } ( 1 0 0 \mathrm { K } ; R _ { \mathrm { t r a i n } } ) } { \mathbb { E } [ \mathrm { F L O P s } ] _ { \mathrm { S e q O 1 } } ^ { \mathrm { M R L B } } ( 1 0 0 \mathrm { K } ; R _ { \mathrm { t r a i n } } , \boldsymbol { p } _ { \mathrm { h i t } } ^ { \mathrm { t r a i n } } ) } \approx \frac { 1 5 2 . 1 3 } { 3 . 0 4 6 } \approx 4 9 . 9 4 \times .\tag{74}
$$

Inference / serving-side amortization. Using the same formulas but with serving-side reuse ratio $R _ { \mathrm { i n f e r } } = 3 0 0$ and cache hit rate $p _ { \mathrm { h i t } } ^ { \mathrm { i n f e r } } = 0 . 6 ;$ we obtain

$$
\mathrm { S T C A _ { \mathrm { i n f e r } } \approx 4 3 . 0 7 6 ~ G F L O P s / t a r g e t , } ~ \mathrm { S e q O 1 _ { \mathrm { i n f e r } } \approx 0 . 6 7 4 1 9 ~ G F L O P s / t a r g e t } ,
$$

so the expected inference-side FLOP ratio becomes

$$
\mathrm { G a p } _ { \mathrm { i n f e r } } ( 1 0 0 \mathrm { K } ) \triangleq \frac { \mathrm { F L O P s } _ { \mathrm { S T C A } } ^ { \mathrm { M R L B } } ( 1 0 0 \mathrm { K } ; R _ { \mathrm { i n f e r } } ) } { \mathbb { E } [ \mathrm { F L O P s } ] _ { \mathrm { S e q O } 1 } ^ { \mathrm { M R L B } } ( 1 0 0 \mathrm { K } ; R _ { \mathrm { i n f e r } } , \boldsymbol { p } _ { \mathrm { h i t } } ^ { \mathrm { i n f e r } } ) } \approx \frac { 4 3 . 0 7 6 } { 0 . 6 7 4 1 9 } \approx 6 3 . 8 9 \times .\tag{75}
$$

Takeaway. Cross-target reuse increases the expected FLOP reduction at 100K relative to the setting without MRLB. Under training-side MRLB with $R _ { \mathrm { t r a i n } } = 4 0$ and $p _ { \mathrm { h i t } } ^ { \mathrm { t r a i n } } = 0 . 5$ , the gap becomes ∼ 49.94×; under inference/serving-side reuse with $R _ { \mathrm { i n f e r } } = 3 0 0$ and $p _ { \mathrm { h i t } } ^ { \mathrm { i n f e r } } = 0 . 6$ , it further increases to $\sim 6 3 . 8 9 \times$ . This is because reordering leaves STCA with only the per-layer pre-FFNs as reusable components, while SequenceO1 can amortize the entire lightweight SA sketching stage and keep heavyweight computation concentrated in target-conditioned interaction over the fixed sketch length $L _ { s } .$