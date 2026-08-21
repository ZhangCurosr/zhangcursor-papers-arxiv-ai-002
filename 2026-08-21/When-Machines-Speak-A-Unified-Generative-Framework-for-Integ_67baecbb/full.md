# When Machines Speak: A Unified Generative Framework for Integrating Machine-Native Symbols into Pretrained Large Language Models

Su Yan Google Inc. sueyan@google.com

Rakesh Iyer   
Google Inc.   
rni@google.com

## Abstract

Many real-world AI systems represent entities, behaviors, and structured information using discrete machine-native symbols rather than natural language. While these representations are compact and preserve task-relevant structure, they lie outside the linguistic token space of pretrained large language models (LLMs), creating a fundamental divide between language modeling and structured prediction. We introduce UniLang, a unified generative framework that bridges this divide by extending pretrained LLMs to treat machine-native symbols as first-class generative units alongside natural-language tokens. UniLang expands the LLM’s vocabulary and embedding space with grounded machine-native representations, enabling textual and symbolic tokens to be jointly modeled and generated under a single autoregressive objective. This unified interface allows pretrained LLMs to directly operate on machine-native representations without requiring them to be verbalized as natural language or relying on task-specific architectures. We evaluate UniLang on two structurally distinct tasks, sequential recommendation and legal precedent prediction, spanning different domains and types of structured prediction. Across both tasks, UniLang consistently outperforms strong baselines, demonstrating a path toward extending pretrained LLMs beyond language and using them as a common generative modeling backbone for heterogeneous machine-native representations.

## 1 Introduction

Large language models (LLMs) have emerged as a general modeling framework across a wide range of applications. Their success stems from a unified autoregressive formulation in which diverse naturallanguage tasks are expressed through a common vocabulary and generation objective. However, this unified interface is fundamentally linguistic: pretrained LLMs operate over natural-language tokens.

Many modern AI systems, in contrast, represent information using machine-native symbolic representations rather than natural language. These representations increasingly take the form of discrete codes produced by vector quantization or related techniques, representing complex entities such as compressed audio signals [37], semantic identifiers in recommendation systems [9, 26], and structured relational information in graph learning [20]. Such representations preserve explicit discrete structure and offer computational efficiency that natural language cannot easily replicate. However, they remain outside the linguistic token space of pretrained LLMs, making them difficult for LLMs to natively model and generate.

This mismatch creates a methodological divide. Existing approaches largely follow one of two paradigms. The first treats natural language as a universal interface, converting structured information into textual descriptions so that pretrained LLMs can process it [5, 13, 16, 19, 38, 39]. While effective, verbalization may obscure native structure, introduce representational ambiguity, and sacrifice machine-level precision. The second operates directly on machine-native representations using task-specific models [14, 26, 36, 40]. Although these approaches preserve structural fidelity, they do not directly leverage the linguistic and world knowledge encoded in a pretrained LLM within the same generative space.

At a deeper level, the limitation is representational. Pretrained LLMs lack a unified interface that allows machine-native symbols to function as first-class generative units alongside natural-language tokens. Without such an interface, structured symbolic prediction and language modeling remain separate paradigms.

This observation motivates a simple question:

Can pretrained LLMs be extended to directly model and generate machine-native symbols, rather thanforcing symbolic information to become natural language or abandoning pretrained language models altogether?

We answer this question with UniLang<sup>1</sup>, a unified framework that extends pretrained LLMs to jointly model natural-language tokens and machine-native symbols within a single autoregressive vocabulary and generation objective. Rather than treating machine-native representations as external objects or auxiliary embeddings, UniLang explicitly grounds them into the pretrained LLM representation space, allowing machine-native symbols and natural-language tokens to function as first-class generative units within the same autoregressive framework.

This unified representational interface enables structurally different prediction problems to be expressed under the same modeling framework without task-specific architectures. To demonstrate its generality, we evaluate UniLang on two structurally distinct tasks: sequential recommendation and legal precedent prediction. These tasks were intentionally selected because they represent heterogeneous structured prediction problems rather than multiple datasets from the same application domain. Across both tasks, UniLang consistently outperforms strong baselines while using the same modeling framework.

We make the following contributions:

• A unified representational interface. We formulate structured prediction as typed autoregressive generation over heterogeneous token types, allowing natural-language tokens and machine-native symbols to function as first-class generative units within the same pretrained LLM.

• Grounded integration of machine-native symbols. We introduce a vocabulary-extension and contrastive grounding procedure that maps machine-native symbols into the pretrained LLM’s representation space, enabling them to be processed and generated alongside natural language tokens while retaining their discrete structure.

• Generality across heterogeneous structured prediction tasks. We demonstrate that the same UniLang framework applies to structurally distinct tasks, including sequential recommendation and legal precedent prediction, without task-specific architectural modifications, and consistently outperforms strong baselines.

## 2 Background and related work

## 2.1 Machine-native representation.

Machine-native representations convert natural-language content into compact, structured sequences of discrete symbols optimized for computation. The central idea is to encode rich semantic informa tion in a concise symbolic form that facilitates efficient storage, retrieval, and reasoning.

A well-known example is the use of ICD codes in the medical domain [8], where detailed diagnoses are represented as standardized symbolic identifiers. These codes capture complex, structured information that would otherwise require lengthy textual descriptions. Nowadays, machine-native representations follow the same principle but often are learned automatically from data rather than manually defined.

A variety of approaches have been proposed to generate such representations. Early methods rely on clustering-based discretization, such as hierarchical k-means over embedding spaces [33]. Hashing-based techniques map high-dimensional continuous vectors into compact binary codes [18]. Quantization-based approaches, including subspace quantization via k-means, learn discrete codebooks for vector compression [9, 12]. More recent work employs hierarchical residual quantization models, such as Residual Quantized Variational Autoencoders (RQ-VAE), to produce multi-level discrete codes that preserve semantic structure [20, 26, 36]. Figure 1 illustrates an example where each item (a movie) is associated with a discrete, hierarchical machine-native code generated by RQ-VAE. Details of the code construction procedure are provided in Section 3.1.

## 2.2 Sequential recommendation.

Sequential recommendation aims to predict the next item a user will interact with given their historical interaction sequence. A wide range of neural architectures have been explored for this task, including convolutional, recurrent, graph-based, and Transformer-based models [3, 7, 15, 17, 30, 32, 40].

Recent work has explored reformulating recommendation as a language modeling problem by converting user, item, and interaction information into unified natural-language sequences and training encoder-decoder models [5]. While this verbalization strategy enables the use of LLMs, flattening structured interactions into text can obscure type information and introduce representational ambiguity.

Other approaches encode items as discrete symbols derived from vector quantization of dense embeddings [9]. Extending this idea, [26] generates hierarchical discrete representations using Residual Quantized VAEs (RQ-VAE), referring

![](images/070694c6a9ef73a97b6096672b7e86c6b00980d21fbb04ab9def00a7a0ab57cd.jpg)  
Figure 1: Example of the sequential prediction task. Each item is collectively represented by machine-native codes and various metadata in natural language.

to the resulting identifiers as Semantic IDs, and integrates them within generative retrieval models. Later work notes that purely discrete codes may lose fine-grained semantic information, and proposes augmenting them with continuous embeddings to recover these signals [36]. While this hybrid approach mitigates information loss, it still treats discrete symbols indirectly within learned embedding spaces rather than as first-class generative units in a unified language modeling framework.

In contrast, UniLang models machine-native symbols and natural-language tokens together as generative units, enabling joint autoregressive prediction of structured and textual content.

## 2.3 Legal precedent prediction.

Given the context of a legal argument, legal precedent prediction seeks to identify the specific sentence(s) or paragraph(s) from precedential court decisions that supports the claim [22]. This task is substantially more challenging than case-level citation prediction or document retrieval, where the objective is to identify a relevant case as a whole [2, 11]. Legal cases often span tens or hundreds of pages, requiring passage-level methods to perform fine-grained semantic alignment and more involved legal rea-

Garcia-Giraldo v. United States, 691 F. Supp. 2d 500 (2010) Where, as here, the petitioner has pleaded guilty, the inquiry on collateral review"is ordinarily confined to whether the underlying plea was both counseled and voluntary."United States v. Broce, 488 U.S. 563 (1989)

Figure 2: Example of the legal precedent prediction task. Given context from the citing opinion (Garcia-Giraldo v. United States), predict the quotation sentence(s) or paragraph(s) from the cited opinion (United States v. Broce), which is unknown at inference time.

soning. Figure 2 illustrates an example citation context and its corresponding target passage from the legal dataset–LePaRD [21].

## 2.4 Contrastive alignment and token grounding.

To integrate non-linguistic data into language models, the primary challenge is bridging the semantic gap between human vocabulary and specialized machine symbols. Contrastive alignment has been widely used to ground heterogeneous representations into shared embedding spaces, most prominently in multimodal models such as CLIP [25]. Related techniques have also been explored for grounding new tokens or symbols into pretrained models [4, 31]. Our work differs fundamentally in scope and purpose: rather than maintaining separate representational pathways or learning auxiliary embeddings, we use contrastive grounding as a mechanism to elevate machine-native discrete representations to first-class language tokens. This allows them to be generated, composed, and jointly modeled over within a unified sequence modeling framework.

## 3 UniLang framework

The core philosophy of UniLang is that natural language and machine-native representations are complementary representations of the same underlying entities. Natural language provides rich, human-interpretable context, while machine-native representations encode compact semantic structures optimized for computation.

Unlike existing approaches that force a choice between these modalities, UniLang treats them as heterogeneous fields within a unified representation and generation space. By enabling joint computation over both types of information, UniLang combines the pretrained linguistic knowledge of LLMs with the compact semantic structure encoded by machine-native symbols.

## 3.1 Constructing machine-native representations

UniLang is agnostic to the specific mechanism used to generate machine-native representations, requiring only that they consist of discrete, identifiable tokens that can be incorporated into the model vocabulary. In this work, following prior literature on Semantic IDs and residual vector quantization [26, 36], we adopt an RQ-VAE–based discretization pipeline as a concrete instantiation.

Many items of interest admit partial textual descriptions. For example, a movie can be characterized by its title, release year, and genres, while a legal case is naturally described by its textual context, potentially augmented with structured metadata such as court name or case date. A pretrained text encoder maps these descriptions into high-dimensional embeddings, which an RQ-VAE discretizes into a sequence of codes across l quantization levels. Denoting the codebook size at level i by $c _ { i } ,$ the resulting representation is $\mathbf { q } = ( q ^ { ( 1 ) } , \ldots , q ^ { ( l ) } ) , \quad q ^ { ( i ) } \in \{ 0 , \ldots , c _ { i } - 1 \}$ . Following common practice, we set $\bar { l } = 3$ and $c _ { i } = 2 5 6$ for all levels. To mitigate potential code collisions, we append an additional disambiguation level $q ^ { ( l + 1 ) }$ , yielding $( q ^ { ( 1 ) } , \dots , q ^ { ( l ) } , q ^ { ( l + 1 ) } )$ . For example, (11, 43, 204, 0) and (11, 43, 204, 1) denote two distinct items sharing the same base quantized code. To integrate these discrete codes into the LLM vocabulary, we prepend level-specific prefixes to ensure tokenlevel distinguishability, forming Semantic IDs (SIDs) such as (A11, B43, C204, D0). The resulting machine-native vocabulary contains $4 \times 2 5 6 = 1 . 0 2 4$ tokens, which we call machine tokens, capable of representing up to $2 5 \dot { 6 } ^ { 4 } \approx 4 . 3$ billion distinct items. Figure 1 illustrates how items are jointly represented using SIDs and natural-language tokens.

## 3.2 Vocabulary expansion and semantic grounding

To facilitate seamless modeling of non-linguistic symbols, UniLang establishes a shared vocabulary across modalities. Although pretrained LLMs are optimized for natural language, we extend their latent space with a dedicated set of machine tokens.

Machine tokens are grounded by aligning their embeddings with the textual description embeddings of corresponding items using contrastive learning. Given a pretrained LLM with natural-language vocabulary $\mathcal { V } _ { \mathrm { N L } }$ , we extend its vocabulary with a fixed set of machine tokens $\mathcal { V } _ { \mathrm { M I } }$ (1,024 tokens in our instantiation), each assigned a learnable embedding initialized from a zero-mean normal distribution with standard deviation equal to the model’s initializer range.

Let $\mathbf { m } _ { i } = ( m _ { i } ^ { 1 } , m _ { i } ^ { 2 } , \ldots , m _ { i } ^ { L } ) \in \mathcal { V } _ { \mathrm { M I } } ^ { L }$ denote the machine token sequence representing item $i ,$ where $L$ is fixed (e.g. $L = 4$ in our setting). Let $\mathbf { t } _ { i } = ( w _ { i } ^ { 1 } , w _ { i } ^ { 2 } , \ldots , w _ { i } ^ { T _ { i } } ) \in \mathcal { V } _ { \mathrm { N I } } ^ { T _ { i } }$ denote the corresponding textual description. Both sequences are encoded by the same pretrained LLM $f _ { \boldsymbol { \theta } } { : }$

$$
\begin{array} { r } { { \bf z } _ { i } ^ { \mathrm { M L } } = f _ { \boldsymbol { \theta } } ( { \bf m } _ { i } ) , \quad { \bf z } _ { i } ^ { \mathrm { N L } } = f _ { \boldsymbol { \theta } } ( { \bf t } _ { i } ) \quad } \end{array}
$$

The grounding objective encourages the machine token sequence embedding $\mathbf { z } _ { i } ^ { \mathrm { { M L } } }$ to align with its corresponding textual description embedding $\mathbf { z } _ { i } ^ { \mathrm { { N L } } }$ , while remaining distinct from embeddings of other items. Given a batch of N items, we employ an InfoNCE-style contrastive loss [34]:

$$
\mathcal { L } _ { i } = - \log \frac { \exp \left( { \sin ( \mathbf { z } _ { i } ^ { \mathrm { M L } } , \mathbf { z } _ { i } ^ { \mathrm { N L } } ) / \tau } \right) } { \sum _ { j = 1 } ^ { N } \exp \left( { \sin ( \mathbf { z } _ { i } ^ { \mathrm { M L } } , \mathbf { z } _ { j } ^ { \mathrm { N L } } ) / \tau } \right) }\tag{1}
$$

where $\begin{array} { r } { s i m ( \mathbf { u } , \mathbf { v } ) = \frac { \mathbf { u } ^ { T } \mathbf { v } } { \| \mathbf { u } \| \| \mathbf { v } \| } } \end{array}$ denotes cosine similarity and τ is a temperature hyperparameter. To promote bidirectional alignment, we also compute the symmetric loss:

$$
\mathcal { L } _ { i } ^ { s y m } = - \log \frac { \exp \bigl ( \sin ( \mathbf { z } _ { i } ^ { \mathrm { N L } } , \mathbf { z } _ { i } ^ { \mathrm { M L } } ) / \tau \bigr ) } { \sum _ { j = 1 } ^ { N } \exp \bigl ( \sin ( \mathbf { z } _ { i } ^ { \mathrm { N L } } , \mathbf { z } _ { j } ^ { \mathrm { M L } } ) / \tau \bigr ) }\tag{2}
$$

The final grounding objective is given by $\begin{array} { r } { \mathcal { L } = \frac { 1 } { 2 N } \sum _ { i = 1 } ^ { N } ( \mathcal { L } _ { i } + \mathcal { L } _ { i } ^ { s y m } ) } \end{array}$

During grounding, only machine token embeddings are updated; all LLM parameters and naturallanguage token embeddings remain fixed. Because textual description embeddings are constant, $\mathbf { z } _ { j } ^ { \mathrm { { N L } } }$ can be precomputed and reused across training batches, reducing computation. Textual description embeddings are represented using the final hidden state of the sequence, while machine token embeddings are formed via mean pooling over token-level hidden states to ensure equal contribution from each quantization level.

## 3.3 Unified autoregressive modeling

After grounding machine tokens, we formulate all downstream tasks as sequence-to-sequence generation over heterogeneous token types. Natural-language and machine tokens share a single extended vocabulary $\mathcal { V } = \mathcal { V } _ { \mathrm { N L } } \cup \mathcal { V } _ { \mathrm { M L } }$ and are modeled autoregressively: for an input sequence ${ \bf x } = ( x _ { 1 } , \dots , x _ { T } )$ with $x _ { t } \in \mathcal V$ , the model defines

$$
p _ { \theta } ( \mathbf { y } | \mathbf { x } ) = \prod _ { t = 1 } ^ { \| \mathbf { y } \| } p _ { \theta } ( y _ { t } | \mathbf { x } , y _ { < t } ) , \quad y _ { t } \in \mathcal { V }
$$

No distinction is made between token classes at the modeling level. All tokens share the same embedding table, self-attention layers, and output head, allowing natural-language and machine tokens to interact directly. The framework is task-agnostic. Different downstream tasks are defined by input–output sequence constructions, rather than by changes to the model architecture or training objective.

## 3.4 Task formulations

To operationalize this framework, we represent tasks as structured hybrid sequences that interleave natural-language tokens and machine-native symbols under a consistent template. Inputs follow a fixed formatting schema that organizes heterogeneous fields in a stable order, while outputs are divided into typed segments using delimiter tokens (e.g., <year>, <genre>, <sid>). These output tags define semantic prediction fields and guide autoregressive generation.

Sequential recommendation. We formulate next-item prediction as typed autoregressive generation over heterogeneous fields. A user’s interaction history is encoded as a structured sequence of SID, year, genre, where the SID is a machine-native identifier and the remaining attributes are naturallanguage tokens. Rather than directly predicting a single item ID, the model generates the next item in a field-wise manner: it first produces <year> and <genre> segments, followed by <sid>, the machine-native identifier (Figure 3a).

This structured design discourages degenerate solutions that rely solely on machine-token pattern continuation. Instead, it requires the model to align symbolic sequence prediction with naturallanguage semantics, coupling structural precision with semantic grounding.

Legal precedent prediction. We apply the same typed generative formulation to legal citation modeling. Citation contexts and quoted passages, together with associated metadata, are represented as structured hybrid sequences. Each text segment is encoded and quantized into a machine-native SID, while metadata fields remain in natural-language form. The input prompt interleaves the context

![](images/4da8682f047302be41fc3e6b26a1001c6a262be72e525eaa6ccdd0910c5d39dc.jpg)  
(b) Legal precedent prediction  
Figure 3: Examples of unified structured prompting across structurally distinct tasks. Each task uses taskspecific metadata and a tailored combination of natural-language and machine tokens, yet all follow the same autoregressive SFT training procedure.

SID with its metadata, and the target sequence consists of the cited passage’s metadata followed by its SID (Figure 3b).

This formulation casts citation prediction as structured autoregressive generation over heterogeneous token types. It mirrors the recommendation setting, though it operates over a structurally distinct prediction problem.

## 4 Experiment

## 4.1 Datasets

We evaluate UniLang on six real-world benchmarks covering two structurally distinct domains: sequential recommendation and legal precedent prediction. For sequential recommendation, we use the "Beauty" subset of the Amazon Product Reviews dataset [23], along with the MovieLens [6] 1M (ML-1m) and 20M (ML-20m) datasets, which differ substantially in size. For legal precedent prediction, we use LePaRD [21], the largest public dataset for legal case retrieval and prediction, which provides three dataset splits of different sizes. Detailed dataset statistics are provided in Appendix A.

## 4.2 Evaluation metrics and data splits

We evaluate model performance using Recall@k and NDCG@k. For sequential recommendation, we report results at k ∈ 5, 10, while for legal precedent prediction we use k ∈ 1, 10 to emphasize the importance of top-1 accuracy in legal retrieval tasks. For sequential recommendation, we follow the standard leave-one-out protocol, where the last interaction of each user is used for testing, the second-to-last for validation, and the remaining interactions for training. In legal precedent prediction, we use a 90%/5%/5% split for training, validation, and testing, respectively. To enable efficient hyperparameter tuning, we perform model selection on a random subset of the validation set, while all reported test results are computed on the full test set (see Appendix B for details).

## 4.3 Machine-native representation generation details

For the Amazon Beauty dataset, we concatenate the raw text from the "title", "categories", "description", "brand", and "price" fields to form a single textual description. For the MovieLens datasets, we combine the "title" with the "genre" field. For the LePaRD dataset, the case context and quoted precedent passages are directly used as provided. Each resulting text blob is encoded using the open-source sentence-t5<sup>2</sup> [24] model to obtain a 768-dimensional dense embedding. We then train an RQ-VAE to quantize these embeddings into discrete machine-native codes (SIDs). Detailed training hyperparameters are provided in Appendix C.

## 4.4 Machine token grounding implementation details

We use Llama-3.2-1B-Instruct <sup>3</sup> as the base model and extend its vocabulary with 1,024 machinenative tokens. The embeddings of these tokens are trained using the contrastive objective described in Section 3.2 (see Table 9 in Appendix for hyperparameters).

Grounding quality is monitored via an alignment metric, computed as the cosine similarity between a machine-token sequence and its corresponding textual embedding. Higher values indicate stronger semantic correspondence. Early stopping is applied based on this metric.

## 4.5 LLM fine-tuning details

After grounding the machine-native tokens and integrating them into the pretrained Llama-3.2- 1B-Instruct model, we perform downstream adaptation. During this stage, all original model parameters–including the pretrained token embeddings and the grounded machine-token embeddings– are frozen. We train task-specific Low-Rank Adaptation (LoRA) adapters [10] using supervised fine-tuning (SFT). For sequential recommendation, the maximum user history length is set to 30 for MovieLens and 50 for Amazon Beauty.

For all tasks, LoRA is applied to the projection layers in each transformer block, namely q\_proj, k\_proj, v\_proj, o\_proj, gate\_proj, up\_proj, and down\_proj. The input–output formatting used for SFT, along with an example prompt and target sequence for MovieLens, is illustrated in Figure 3a. The corresponding formats for LePaRD and Amazon Beauty are shown in Figures 3b and 5, respectively. Complete fine-tuning hyperparameters can be found in Table 10 in the Appendix.

Table 1: Performance comparison on sequential recommendation task.
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Metric</td><td colspan="3">Discriminative</td><td colspan="3">Generative</td><td rowspan="2">Improvement</td></tr><tr><td>SASRec</td><td>BERT4Rec</td><td> $\mathrm { S } ^ { 3 } { \mathrm { - } } \mathrm { R e c }$ </td><td>P5</td><td>TIGER (re-impl.)</td><td>UniLang (ours)</td></tr><tr><td rowspan="4">Beauty</td><td>Recall@5</td><td>0.0387</td><td>0.0203</td><td>0.0387</td><td>0.0163</td><td>0.0361</td><td>0.0419</td><td>8.27%</td></tr><tr><td>NDCG@5</td><td>0.0249</td><td>0.0124</td><td>0.0244</td><td>0.0107</td><td>0.0241</td><td>0.0299</td><td>20.08%</td></tr><tr><td>Recall@10</td><td>0.0605</td><td>0.0347</td><td>0.0647</td><td>0.0254</td><td>0.0595</td><td>0.0603</td><td>-6.80%</td></tr><tr><td>NDCG@10</td><td>0.0318</td><td>0.0170</td><td>0.0327</td><td>0.0136</td><td>0.0318</td><td>0.0358</td><td>9.48%</td></tr><tr><td rowspan="4">ML-1m</td><td>Recall@5</td><td>0.1273</td><td>0.1011</td><td>0.1258</td><td>0.1098</td><td>0.0634</td><td>0.1661</td><td>30.48%</td></tr><tr><td>NDCG@5</td><td>0.0843</td><td>0.0649</td><td>0.0824</td><td>0.0734</td><td>0.0412</td><td>0.1145</td><td>35.82%</td></tr><tr><td>Recall@10</td><td>0.2013</td><td>0.1533</td><td>0.2023</td><td>0.1575</td><td>0.1055</td><td>0.2374</td><td>17.35%</td></tr><tr><td>NDCG@10</td><td>0.1083</td><td>0.0810</td><td>0.1070</td><td>0.0888</td><td>0.0547</td><td>0.1376</td><td>27.05%</td></tr><tr><td rowspan="4">ML-20m</td><td>Recall@5</td><td>0.0889</td><td>0.0616</td><td>0.0884</td><td>N/A</td><td>0.0763</td><td>0.1911</td><td>114.96%</td></tr><tr><td>NDCG@5</td><td>0.0549</td><td>0.0345</td><td>0.0531</td><td>N/A</td><td>0.0523</td><td>0.1382</td><td>151.73%</td></tr><tr><td>Recall@10</td><td>0.1453</td><td>0.0948</td><td>0.1493</td><td>N/A</td><td>0.1137</td><td>0.2597</td><td>73.95%</td></tr><tr><td>NDCG@10</td><td>0.0728</td><td>0.0476</td><td>0.0802</td><td>N/A</td><td>0.0643</td><td>0.1603</td><td>99.88%</td></tr></table>

Note: The best results are shown in bold, and the second-best results are underlined.

## 4.6 Performance on sequential recommendation

We first evaluate UniLang on the sequential recommendation task, comparing it against strong task-specific baselines. BERT4Rec [30], SASRec [14], and S<sup>3</sup>-Rec [40] are discriminative models specifically designed for sequential recommendation. P5 [5] is a generative approach that verbalizes user, item, and interaction information into natural language, whereas TIGER [26] is also generative but operates purely on machine-native representations (SIDs), predicting the next SID autoregressively.

Table 2: Run-to-run variability on
<table><tr><td>metric</td><td>mean ± standard error</td></tr><tr><td>Recall@5</td><td> $0 . 1 9 0 8 \pm 0 . 0 0 0 1 6$ </td></tr><tr><td>NDCG@5</td><td> $0 . 1 3 7 8 \pm 0 . 0 0 0 1 7$ </td></tr><tr><td>Recall@10</td><td> $0 . 2 5 9 6 \pm 0 . 0 0 0 1 4$ </td></tr><tr><td>NDCG@10</td><td> $0 . 1 6 0 0 \pm 0 . 0 0 0 1 4$ </td></tr></table>

For all baselines except TIGER and P5, we report results from the original papers or produce them using the original authors’ implementations. For TIGER, we follow the paper and implement the model accordingly. For P5, we adopt corrected and extended results from subsequent works where applicable. Details are provided in Appendix D.

As shown in Table 1, UniLang achieves remarkable performance, outperforming nearly all baselines across benchmarks with up to 151% improvement in NDCG@5 and 115% in Recall@5 on MovieLens-20M. These results demonstrate the effectiveness of extending pretrained LLMs with machine-native symbols: UniLang can directly model and generate structured representations, enabling it to surpass specialized sequential recommendation architectures.

To assess statistical stability, we perform three independent runs with different random seeds on the MovieLens-20M dataset and report mean ± standard error in Table 2.

Table 3: Performance comparison on legal precedent prediction task.
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Metric</td><td colspan="2">Retrieval</td><td colspan="2">Classification</td><td colspan="2">Generative</td><td rowspan="2">Improvement</td></tr><tr><td>BM25</td><td>fine-tuned SBERT</td><td>LEGAL-BERT</td><td>DistilBERT</td><td>UniLang (ours)</td><td></td></tr><tr><td rowspan="3">10K</td><td>Recall@1</td><td>0.0501</td><td>0.0899</td><td>0.1666</td><td>0.1967</td><td></td><td>0.2938</td><td>49.36%</td></tr><tr><td>Recall@10</td><td>0.1952</td><td>0.4479</td><td>0.4765</td><td>0.5912</td><td>0.6823</td><td></td><td>15.41%</td></tr><tr><td>NDCG@10</td><td>0.1137</td><td>0.2627</td><td>0.3075</td><td>0.3773</td><td>0.4775</td><td></td><td>26.56%</td></tr><tr><td rowspan="3">20K</td><td>Recall@1</td><td>0.0413</td><td>0.0753</td><td>0.1280</td><td></td><td>0.1674</td><td>0.2486</td><td>48.51%</td></tr><tr><td>Recall@10</td><td>0.1667</td><td>0.3850</td><td>0.3684</td><td>0.5216</td><td>0.6290</td><td></td><td>20.59%</td></tr><tr><td>NDCG@10</td><td>0.0956</td><td>0.2072</td><td>0.2377</td><td>0.3291</td><td>0.4264</td><td></td><td>29.57%</td></tr><tr><td rowspan="3">50K</td><td>Recall@1</td><td>0.0341</td><td>0.0500</td><td>0.0877</td><td>0.1231</td><td></td><td>0.1676</td><td>36.15%</td></tr><tr><td>Recall@10</td><td>0.1353</td><td>0.2590</td><td>0.2522</td><td>0.3934</td><td></td><td>0.4882</td><td>24.10%</td></tr><tr><td>NDCG@10</td><td>0.0779</td><td>0.1378</td><td>0.1625</td><td>0.2457</td><td></td><td>0.3131</td><td>27.43%</td></tr></table>

Note: The best results are shown in bold, and the second-best results are underlined.

## 4.7 Performance on legal precedent prediction

To further demonstrate its generality, we apply the UniLang framework–without any architectural modification–to the structurally different task of legal precedent prediction. Unlike sequential recommendation, which involves modeling chronological sequences of user interactions, legal precedent prediction requires identifying the relevant legal passage from a single citation context, providing a stringent test of UniLang’s ability to generalize across fundamentally different problem structures.

We compare UniLang against established retrieval and discriminative baselines. BM25 and fine-tuned SBERT are retrieval-based methods, whereas LEGAL-BERT and DistilBERT approach passage retrieval as a text classification task. All baseline results are taken from the LePaRD paper [21]. Further details on these baselines are provided in Appendix E.

Comparison results are provided in Table 3. Despite the differences in task structure between sequential recommendation and legal passage retrieval, UniLang consistently outperforms all baselines by a substantial margin. In particular, it achieves a 49% improvement in Recall@1 on the 10k dataset, significantly surpassing even specialized discriminative models. These results underscore UniLang’s capability to provide a unified modeling framework across tasks with fundamentally different structures.

![](images/965b24da23f2d3c8cfeeb01cc4be557ef2d74b8e06551a05dce11e112d204112.jpg)

![](images/a2306fef4791f404b4f333f0462f0f2f517de703ab0bd7757413d273a4e6d109.jpg)  
Figure 4: Ablation test on MovieLens-20m.

## 4.8 Ablation study

We conduct ablation studies on MovieLens-20M to isolate the contributions of semantic grounding, natural-language fields, and structural delimiters. For efficiency, all ablations are evaluated on the fixed validation subset. We consider three variants: (1) NLRemoved, which trains exclusively on machine-native tokens; (2) NoTypeDelim, which removes typed output delimiters (e.g., <year>, </dname>); and (3) NoWarmup, which uses randomly initialized symbolic embeddings instead of those aligned via contrastive learning.

As shown in Figure 4, NLRemoved exhibits rapid initial improvement followed by performance degradation after 40k steps, suggesting early overfitting when semantic signals from natural language are absent. NoTypeDelim remains stable but consistently underperforms the full UniLang model, indicating that explicit input and output structuring facilitates optimization and improves prediction quality. Notably, NoWarmup fails to achieve non-zero metrics (NDCG and Recall), demonstrating that pre-aligning machine-native symbols to the LLM’s latent space is essential for effective generative learning in our setting. We omit this variant from Figure 4 as it remains at the zero-baseline throughout training. Overall, these results suggest that natural language provides regularizing context, while grounded embeddings and typed structures are critical for stability and prediction quality.

template example   
<sft:think>   
user b\_10009:   
<sft:think> A53 B53 C119 D0: Makeup, Face, Concealers & Neutralizers; Maybelline   
prompt user {uid}: A197 B216 C7 D0: Tools & Accessories, Makeup Brushes & Tools, Brushes & Applicators; Maybelline   
{{SID}:{cat}; {brand}} A198 B110 C88 D0: Makeup, Face, Foundation; Maybelline   
prediction: prediction:   
target <cat>{target\_cat}</cat> <cat>Makeup, Eyes, Mascara</cat>   
<brand>{target\_brand}</brand> <brand>Maybelline</brand>   
<sid>{target\_SID}</sid>{eos} <sid>A35 B21 C172 D0</sid><|eot\_id|>  
Figure 5: Prompt template and example for the Amazon Beauty dataset.

## 4.9 Qualitative comparison

So far, our results demonstrate the quantitative benefits of jointly representing natural language and machine-native tokens. We now provide a qualitative comparison with approaches that verbalize all user and item information into natural language, such as P5 [35]. P5 typically requires multiple handcrafted prompts per dataset per task (e.g., 13 prompts for Beauty sequential recommendation). One example prompt is shown below:

Input template: I find the purchase history list of user {{user\_id}}: {{history item list of {{item\_id}}}}. I wonder which is the next item to recommend to the user?

Target template: {{item [item\_id]}}

In contrast, UniLang uses a single structured prompt that seamlessly combines natural language with machine tokens, as shown in Figure 5. This unified representation reduces the need for extensive prompt engineering while maintaining or improving performance. By treating language and structured symbols as complementary generative units, UniLang provides a scalable and generalizable framework for heterogeneous tasks, enabling models to jointly model both token types.

## 4.10 Emergent representational capabilities

Beyond task performance, UniLang exhibits an important representational advantage. Existing generative recommender systems struggle with user identity modeling. For example, TIGER [26] represents users by hashing raw user IDs into a fixed set of 2,000 ID tokens. As a result, these user ID tokens occupy a large portion of the model’s extended vocabulary (66%), which can dominate the token space and limit scalability as the number of users grows.

In contrast, UniLang requires no dedicated user tokens or hashing schemes. User identifiers (e.g., "b\_1009") can be directly represented in natural-language form and processed by the pretrained tokenizer. Although not the primary objective of our method, this property highlights UniLang’s representational scalability and reinforces its role as a unified modeling framework rather than a task-engineered solution.

## 5 Conclusion and future work

We introduced UniLang, a unified framework that extends pretrained LLMs to model natural language and machine-native symbols within a single autoregressive objective. By treating structured symbols as first-class generative units, UniLang enables structurally diverse tasks–such as sequential recommendation and legal precedent prediction–to be addressed without task-specific architectures. Empirically, UniLang consistently outperforms strong baselines, while ablations demonstrate the importance of natural-language context and typed structure. Overall, UniLang provides a general interface for extending pretrained LLMs beyond language to model heterogeneous machine-native representations.

While UniLang demonstrates strong performance on structured prediction tasks, we did not evaluate the impact of symbolic fine-tuning on the model’s natural language generation quality. Our focus in this work is on accuracy in predictive tasks, and studying potential effects on language fluency and coherence is left for future work.

Table 4: Statistics of the sequential recommendation datasets.
<table><tr><td>Dataset</td><td>#users</td><td>#items</td><td>#actions</td><td>Avg. length</td><td>Density</td></tr><tr><td>Beauty</td><td>40,226</td><td>54,542</td><td>0.35m</td><td>8.8</td><td>0.02%</td></tr><tr><td>ML-1m</td><td>6,040</td><td>3,416</td><td>1m</td><td>163.5</td><td>4.79%</td></tr><tr><td>ML-20m</td><td>138,493</td><td>26,744</td><td>20m</td><td>144.4</td><td>0.54%</td></tr></table>

Table 5: Statistics of the legal precedent prediction datasets.
<table><tr><td>Dataset (# cited passages)</td><td># Citing Passage Surface Forms</td><td># Cited Passage Surface Forms</td></tr><tr><td>10k</td><td>1,294,730</td><td>798,558</td></tr><tr><td>20k</td><td>1,586,663</td><td>1,140,058</td></tr><tr><td>50k</td><td>2,079,704</td><td>1,821,751</td></tr></table>

## A Dataset

• Amazon Beauty <sup>4</sup> [23]: This dataset consists of product reviews collected from Amazon.com. The full collection is organized by top-level product categories, and in our experiments, we use the subset corresponding to the "Beauty" category.

• MovieLens: A widely used benchmark for evaluating recommendation systems. In this study, we use two standard versions: MovieLens 1M (ML-1m) <sup>5</sup> and MovieLens 20M (ML-20m) <sup>6</sup>.

• The LePaRD dataset <sup>7</sup> [21] is a large-scale collection of 4.3 million U.S. federal judicial citations. It leverages judges’ quotation behavior as supervision, pairing precedential quotations with their corresponding citation contexts. The dataset provides three subsets that differ in the number of top-n most frequently cited passages included. Due to unavoidable preprocessing noise—such as OCR errors, sentence segmentation inconsistencies, and metadata formatting variations—the same citing context or cited quotation may appear in multiple surface forms.

Table 4 presents the statistics of the sequential recommendation datasets. Table 5 provides an overview of the legal precedent prediction dataset. Table 6 summarizes the statistics of the textual features in the legal dataset.

## B Validation set sampling

We use Recall@k on the validation set for early stopping during training. Since our framework is generative, we perform beam search to deterministically generate the top-k results and compute Recall@k accordingly. Running beam search over the full validation set is computationally expensive, so to expedite model selection, we instead use a fixed, randomly sampled subset of the validation set generated with a fixed random seed. For all datasets, reported results are based on models selected using this sampled subset. Table 7 lists the sample sizes for each dataset. For all experiments, we use a beam size of 20.

## C Model training hyperparameters

All experiments were conducted on 8 NVIDIA H100 GPUs (80GB each). Training can also be performed on a single GPU; multiple GPUs are used primarily to accelerate training via data parallelism. Training time varies depending on batch composition. On the MovieLens-20M dataset,

Table 6: Summary statistics of legal precedent dataset text features
<table><tr><td>Feature</td><td>Mean</td><td>Std</td><td>Min</td><td>Max</td></tr><tr><td>Length of cited text (chars)</td><td>306</td><td>225</td><td>24</td><td>18,342</td></tr><tr><td>Length of citing context (chars)</td><td>562</td><td>216</td><td>5</td><td>14,062</td></tr></table>

Table 7: Validation subset sizes for model selection
<table><tr><td>Dataset</td><td>Validation size</td><td>Sample size</td></tr><tr><td>Beauty</td><td>40,226</td><td>5,000</td></tr><tr><td>MovieLens-1m</td><td>6,040</td><td>1,000</td></tr><tr><td>MovieLens-20m</td><td>138,493</td><td>1,000</td></tr><tr><td>10k</td><td>103,812</td><td>1,000</td></tr><tr><td>20k</td><td>134,737</td><td>1,000</td></tr><tr><td>50k</td><td>190,051</td><td>1,000</td></tr></table>

under our configuration, training proceeds at approximately 0.15 seconds per optimization step (batch size 128), with total runtime scaling linearly with the number of training steps.

![](images/50cf55f8aef442e9f24b067af5786115efe3dbd291f37780ef203bba3b3b7ab7.jpg)  
(a) Beauty

![](images/6eac69622213757a07e6880e7a6bf15aef361b5ff5d68314b10e2c321c099002.jpg)  
(b) LePaRD 20k  
Figure 6: RQ-VAE training progress on different datasets.

## C.1 RQ-VAE training parameters

The same RQ-VAE architecture is used across all datasets. The model takes 768-dimensional text embeddings as input. The encoder comprises three fully connected layers with output dimensions 512, 256, and 128, each followed by ReLU activation, and outputs a 16-dimensional latent representation. The decoder mirrors the encoder with layers of dimensions 128, 256, and 512. At each quantization level, we maintain a codebook of size 256. Table 8 summarizes the detailed hyperparameter settings, and Figure 6 illustrates the training dynamics of the individual loss components across datasets.

## C.2 Machine token grounding parameters

Table 9 summarizes the hyperparameters used for training the embeddings of the machine-native tokens.

## C.3 Supervised fine tuning parameters

Table 10 lists the hyperparameters used for supervised fine-tuning of Llama-3.2-1B-Instruct on structured input–output sequences containing interleaved natural-language and machine-native tokens.

Table 8: RQ-VAE hyperparameters.
<table><tr><td>Hyperparameter</td><td>Beauty</td><td>ML-1M ML-20M</td><td>LePaRD 10k / 20k / 50k</td></tr><tr><td>Total training steps</td><td>30k</td><td>20k</td><td>20k</td></tr><tr><td>Warm up</td><td>3k</td><td>2k</td><td>3k</td></tr><tr><td>Peak learning rate</td><td>1e-3</td><td>1e-3</td><td>1e-3</td></tr><tr><td>End learning rate</td><td>1e-5</td><td>5e-5</td><td>1e-5</td></tr><tr><td>Ema decay</td><td>0.99</td><td>0.99</td><td>0.99</td></tr><tr><td>Commitment cost</td><td>1.5</td><td>1.5</td><td>0.1</td></tr></table>

Note: All experiments use a codebook embedding size of 16 and a batch size of 2048, and are optimized with AdamW (weight decay 0.055, β<sub>1</sub> = 0.9, β<sub>2</sub> = 0.98) under a cosine learning rate schedule.

Table 9: Machine token alignment hyperparameters.
<table><tr><td>Dataset</td><td>Beauty</td><td>ML-1M ML-20M</td><td>LePaRD 10k / 20k / 50k</td></tr><tr><td>Batch size</td><td>1024</td><td>512</td><td>128</td></tr><tr><td>Learning rate</td><td>1e-3</td><td>1e-3</td><td>6e-3</td></tr><tr><td>Total steps</td><td>4k</td><td>4k</td><td>8k</td></tr></table>

Note: All experiments use the AdamW optimizer (weight decay $1 0 ^ { - 2 } , \beta _ { 1 } = 0 . 9 , \beta _ { 2 } = 0 . 9 8 )$ with a cosine learning rate schedule and 400 warmup steps. The temperature for the contrastive loss is set to 0.2.

## D Sequential recommendation baseline methods

• SASRec [14] is a self-attention-based sequential recommendation model that efficiently captures long-term user behavior.

• BERT4Rec [30] trains a bidirectional Transformer with a masked item prediction objective to learn contextualized representations of user behavior sequences.

• S<sup>3</sup>-Rec [40] adopts a self-supervised pretraining strategy to capture correlations among items, attributes, and interaction sequences, alleviating data sparsity.

• P5 [5] formulates recommendation as a language modeling problem, representing user–item interactions, user profiles, item metadata, and reviews in natural language.

• TIGER [26] represents items using semantic IDs (SIDs), converting user interaction histories into SID sequences. A generative model is trained from scratch to autoregressively predict the next SID for sequential recommendation.

In evaluating recommendation systems, two protocols are commonly adopted: (1) sampled evaluation, where the ground-truth item is ranked against a fixed set of negative samples (e.g., 99 items), and (2) full-ranking, where the ground-truth item is ranked against all items in the candidate set. While sampled evaluation is computationally efficient, it can lead to overly optimistic performance estimates. In this work, we adopt full-ranking evaluation throughout to provide a more rigorous and realistic assessment of model performance.

For the Amazon Beauty dataset, results for SASRec, BERT4Rec, and $S ^ { 3 } .$ -Rec are taken from the publicly released benchmarks <sup>8</sup> provided by the ${ \mathrm { S } } ^ { 3 } { \mathrm { - } } { \mathrm { R e c } }$ authors. The P5 results are obtained from the TIGER paper [26], where the authors introduced a preprocessing modification to ensure a fair comparison.

For the MovieLens 1M and 20M datasets, results for SASRec, BERT4Rec, and $S ^ { 3 }$ -Rec are obtained using the authors’ released code. All models are trained with their default hyperparameter settings. For SASRec and BERT4Rec, we additionally implement full-ranking evaluation to obtain the reported results.

Results for P5 on the MovieLens 1M dataset are taken from the OpenP5 [35] paper. OpenP5   
incorporates the original P5 backbone model, downstream tasks, and item indexing method. Since P5

Table 10: SFT hyperparameters.
<table><tr><td>Dataset</td><td>Beauty</td><td>ML-1M ML-20M</td><td>LePaRD 10k / 20k / 50k</td></tr><tr><td>Batch size</td><td>8</td><td>32/128</td><td>512</td></tr><tr><td>Learning rate</td><td>1e-4</td><td>2e-4</td><td>2e-4</td></tr><tr><td>LoRA dropout</td><td>0.25</td><td>0.05</td><td>0.25</td></tr><tr><td>Total training steps</td><td>30k</td><td>20k</td><td>30k</td></tr><tr><td>Warm up</td><td>2k</td><td>2k</td><td>2k</td></tr></table>

Note: All experiments use LoRA with rank 32 and α = 2. Optimization is performed using AdamW (weight decay 0.005, $\beta _ { 1 } = 0 . 9 , \beta _ { 2 } = 0 . 9 8 )$ with a cosine learning rate schedule.

requires substantial manual prompt engineering when adapted to new datasets, we consider using OpenP5’s reported results to provide a fairer comparison.

Due to the prohibitively high storage and computational cost of applying P5 to the MovieLens 20M dataset, we do not report its results on this dataset (denoted as N/A in Table 1).

## E Legal precedent prediction baseline methods

We consider four baseline methods, two retrieval-based and two classification-based.

BM25 represents a sparse lexical retrieval approach [28]. The "fine-tuned SBERT" baseline is a dense embedding-based retrieval method that uses a fine-tuned SBERT [27] model to generate embeddings, followed by maximum dot-product similarity for retrieval.

Passage retrieval can also be formulated as a text classification task, where each target passage is assigned a unique label that serves as the prediction target for its preceding context. Two classificationbased baselines are considered: DistilBERT [29] and LEGAL-BERT [1], the latter being a domainadapted BERT model trained on a large corpus of legal documents. All baseline results are taken from the LePaRD paper [21].

## References

[1] Ilias Chalkidis, Manos Fergadiotis, Prodromos Malakasiotis, Nikolaos Aletras, and Ion Androutsopoulos. Legal-bert: The muppets straight out of law school. In Findings of the Association for Computational Linguistics: EMNLP 2020, pages 2898–2904, Online, 2020. Association for Computational Linguistics.

[2] Faraz Dadgostari, Mauricio Guim, Peter A. Beling, Michael A. Livermore, and Daniel N. Rockmore. Modeling law search as prediction. Artificial Intelligence and Law, 29(1):3–34, 2021.

[3] Gabriel de Souza Pereira Moreira, Sara Rabhi, Jeong Min Lee, Ronay Ak, and Even Oldridge. Transformers4rec: Bridging the gap between nlp and sequential/session-based recommendation. In Proceedings of the Fifteenth ACM Conference on Recommender Systems (RecSys 2021), pages 143–153, 2021.

[4] Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit H. Bermano, Gal Chechik, and Daniel Cohen-Or. An image is worth one word: Personalizing text-to-image generation using textual inversion, 2022.

[5] Shijie Geng, Shuchang Liu, Zuohui Fu, Yingqiang Ge, and Yongfeng Zhang. Recommendation as language processing (rlp): A unified pretrain, personalized prompt & predict paradigm (p5). In RecSys 2022 - Proceedings ofthe 16th ACM Conference on Recommender Systems, RecSys 2022 - Proceedings of the 16th ACM Conference on Recommender Systems, pages 299–315. Association for Computing Machinery, Inc, September 2022. Publisher Copyright: © 2022 ACM.; 16th ACM Conference on Recommender Systems, RecSys 2022 ; Conference date: 18-09-2022 Through 23-09-2022.

[6] F. Maxwell Harper and Joseph A. Konstan. The movielens datasets: History and context. ACM Transactions on Interactive Intelligent Systems, 5(4):19:1–19:19, Dec 2015.

[7] Balázs Hidasi, Alexandros Karatzoglou, Linas Baltrunas, and Domonkos Tikk. Session-based recommendations with recurrent neural networks, 2015. cite arxiv:1511.06939Comment: Camera ready version (17th February, 2016) Affiliation update (29th March, 2016).

[8] J. A. Hirsch, G. Nicola, G. McGinty, R. W. Liu, R. M. Barr, M. D. Chittle, and L. Manchikanti. Icd-10: History and context. American Journal ofNeuroradiology, 37(4):596–599, 2016.

[9] Yupeng Hou, Zhankui He, Julian McAuley, and Wayne Xin Zhao. Learning vector-quantized item representation for transferable sequential recommenders. In Proceedings of The Web Conference 2023, pages 2808–2818, 2023.

[10] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models, 2021.

[11] Zihan Huang, Charles Low, Mengqiu Teng, Hongyi Zhang, Daniel E. Ho, Mark S. Krass, and Matthias Grabmair. Context-aware legal citation recommendation using deep learning. In Proceedings of the Eighteenth International Conference on Artificial Intelligence and Law, ICAIL ’21, pages 79–88, New York, NY, USA, 2021. Association for Computing Machinery.

[12] Hervé Jégou, Matthijs Douze, and Cordelia Schmid. Product quantization for nearest neighbor search. IEEE Transactions on Pattern Analysis and Machine Intelligence, 33(1):117–128, 2011.

[13] Liangyi Jiang, Xiaocong Liu, Nejatian Nasir-Moin, Hongyi Wang, Abdullah Abidin, Kai Eaton, . . . , and [additional authors]. Health system-scale language models are all-purpose prediction engines. Nature, 619:357–362, 2023.

[14] Wang-Cheng Kang and Julian McAuley. Self-attentive sequential recommendation, 2018.

[15] Wang-Cheng Kang and Julian McAuley. Self-attentive sequential recommendation. In Proceedings of the 2018 IEEE International Conference on Data Mining (ICDM), pages 197–206, 2018.

[16] Xiaoyu Kong, Junguang Jiang, Bin Liu, Ziru Xu, Han Zhu, Jian Xu, Bo Zheng, Jiancan Wu, and Xiang Wang. Think before recommendation: Autonomous reasoning-enhanced recommender. In NeurIPS 2025 Poster Session, San Diego, 2025.

[17] Jing Li, Pengjie Ren, Zhumin Chen, Zhaochun Ren, Tao Lian, and Jun Ma. Neural attentive session-based recommendation. In Proceedings of the 2017 ACM on Conference on Information and Knowledge Management (CIKM 2017), pages 1419–1428, 2017.

[18] Fangyuan Luo, Yankai Chen, Jun Wu, Tong Li, Philip S. Yu, and Xue Liu. Learning to hash for recommendation: A survey, 2025.

[19] Yanchen Luo, Junfeng Fang, Sihang Li, Zhiyuan Liu, Jiancan Wu, An Zhang, Wenjie Du, and Xiang Wang. Text-guided small molecule generation via diffusion model. iScience, 27(11):110992, 2024.

[20] Yuankai Luo, Hongkang Li, Qijiong Liu, Lei Shi, and Xiao-Ming Wu. Node identifiers: Compact, discrete representations for efficient graph learning. In The Thirteenth International Conference on Learning Representations, 2025.

[21] Robert Mahari, Dominik Stammbach, Elliott Ash, and Alex Pentland. Lepard: A large-scale dataset of judicial citations to precedent. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 9863–9877, Bangkok, Thailand, 2024. Association for Computational Linguistics.

[22] Robert Zev Mahari. Autolaw: Augmented legal reasoning through legal precedent prediction, 2021.

[23] Julian McAuley, Christopher Targett, Qinfeng Shi, and Anton van den Hengel. Image-based recommendations on styles and substitutes. In Proceedings of the 38th International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 43–52, New York, NY, USA, 2015. ACM.

[24] Jianmo Ni, Gustavo Hernandez Abrego, Noah Constant, Ji Ma, Keith Hall, Daniel Cer, and Yinfei Yang. Sentence-t5: Scalable sentence encoders from pre-trained text-to-text models. In Findings ofthe Associationfor Computational Linguistics: ACL 2022, pages 1864–1874, Dublin, Ireland, 2022. Association for Computational Linguistics.

[25] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. CoRR, abs/2103.00020, 2021.

[26] Shashank Rajput, Nikhil Mehta, Anima Singh, Raghunandan Hulikal Keshavan, Trung Vu, Lukasz Heldt, Lichan Hong, Yi Tay, Vinh Tran, Jonah Samost, Maciej Kula, Ed Chi, and Maheswaran Sathiamoorthy. Recommender systems with generative retrieval. In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, editors, Advances in Neural Information Processing Systems, volume 36, pages 10299–10315. Curran Associates, Inc., 2023.

[27] Nils Reimers and Iryna Gurevych. Sentencebert: Sentence embeddings using siamese bert networks. In Proceedings ofthe 2019 Conference on Empirical Methods in Natural Language Processing (EMNLP). Association for Computational Linguistics, 2019.

[28] Stephen E. Robertson, Steve Walker, Susan Jones, Micheline Hancock-Beaulieu, and Mike Gatford. Okapi at TREC-3. In Proceedings of the Third Text REtrieval Conference (TREC 1994), Gaithersburg, USA, November 1994.

[29] Victor Sanh, Lysandre Debut, Julien Chaumond, and Thomas Wolf. Distilbert, a distilled version of bert: Smaller, faster, cheaper and lighter. In Proceedings of the 5th Workshop on Energy Efficient Machine Learning and Cognitive Computing (EMNLP 2020 Workshop), 2020.

[30] Fei Sun, Jun Liu, Jian Wu, Changhua Pei, Xiao Lin, Wenwu Ou, and Peng Jiang. Bert4rec: Sequential recommendation with bidirectional encoder representations from transformer. In Proceedings ofthe 28th ACM International Conference on Information and Knowledge Management (CIKM 2019), pages 1441–1450, 2019.

[31] Hao Tan and Mohit Bansal. Vokenization: Improving language understanding with contextualized, visual-grounded supervision. CoRR, abs/2010.06775, 2020.

[32] Jiaxi Tang and Ke Wang. Personalized top-n sequential recommendation via convolutional sequence embedding. In Proceedings ofthe Eleventh ACM International Conference on Web Search and Data Mining (WSDM 2018), pages 565–573, 2018.

[33] Yi Tay, Vinh Q. Tran, Mostafa Dehghani, Jianmo Ni, Dara Bahri, Harsh Mehta, Zhen Qin, Kai Hui, Zhe Zhao, Jai Gupta, Tal Schuster, William W. Cohen, and Donald Metzler. Transformer memory as a differentiable search index, 2022.

[34] Aäron van den Oord, Yazhe Li, and Oriol Vinyals. Representation learning with contrastive predictive coding. CoRR, abs/1807.03748, 2018.

[35] Shuyuan Xu, Wenyue Hua, and Yongfeng Zhang. Openp5: An open-source platform for developing, training, and evaluating llm-based recommender systems. SIGIR, 2024.

[36] Liu Yang, Fabian Paischer, Kaveh Hassani, Jiacheng Li, Shuai Shao, Zhang Gabriel Li, Yun He, Xue Feng, Nima Noorshams, Sem Park, Bo Long, Robert D. Nowak, Xiaoli Gao, and Hamid Eghbalzadeh. Unifying generative and dense retrieval for sequential recommendation. Trans. Mach. Learn. Res., 2025, 2025.

[37] Neil Zeghidour, Alejandro Luebs, Ahmed Omran, Jan Skoglund, and Marco Tagliasacchi. Soundstream: An end-to-end neural audio codec. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 30:495–507, 2021.

[38] Kepu Zhang, Weijie Yu, Sunhao Dai, and Jun Xu. Citalaw: Enhancing llm with citations in legal domain, 2024.

[39] Jianan Zhao, Le Zhuo, Yikang Shen, Meng Qu, Kai Liu, Michael Bronstein, Zhaocheng Zhu, and Jian Tang. Graphtext: Graph reasoning in text space. In Proceedings ofthe Thirty-Eighth Conference on Neural Information Processing Systems (NeurIPS 2024), 2024. Poster in the workshop \*Adaptive Foundation Models: Evolving AI for Personalized and Efficient Learning\*.

[40] Kun Zhou, Hui Wang, Wayne Xin Zhao, Yutao Zhu, Sirui Wang, Fuzheng Zhang, Zhongyuan Wang, and Ji-Rong Wen. S3-rec: Self-supervised learning for sequential recommendation with mutual information maximization. In Proceedings ofthe 29th ACM International Conference on Information and Knowledge Management (CIKM 2020), pages 1893–1902, 2020.