# Topic-Level Fairness in Retrieval-Augmented Generation

Jaeyoung Park KAIST jypark@kaist.ac.kr

Kexin Rong Georgia Institute of Technology krong@gatech.edu

## Abstract

Group fairness in retrieval-augmented generation (RAG) systems is typically evaluated using metrics aggregated across queries. However, such metrics can conceal topic-level disparities: a system may favor women on questions about human rights and men on questions about ac tors and filmmakers, appearing nearly neutral overall even though neither topic is fair. We therefore study topic-level fairness, which requires group fairness to hold within each topic rather than only in aggregate. To reduce the topic-level disparity, we adapt the reranker sep arately for each topic with a Bradley–Terry objective under direct supervision from the generated output: preference pairs built from the generator’s own output carry the direction that topic requires. With preference pairs built from RoMQA alone, the adaptation reduces the mean absolute per-topic disparity by 51% on RoMQA and by 40% on TREC Fair Ranking while retaining accuracy. No baseline reaches the reduction, even when modified to treat each topic separately rather than apply one correction to every topic. Neither does our method with one reranker for all topics, which follows the direction that most topics require.

## 1 Introduction

Retrieval-augmented generation (RAG) has become a standard approach for knowledge-intensive NLP and is increasingly deployed in systems that answer people’s questions with external knowledge. A retrieval system, however, can reproduce the societal stereotypes in its corpus, even for queries that carry no gender connotation of their own, and the retrieved results can reinforce those stereotypes (Rekabsaz et al., 2021). Prior work has measured bias in RAG systems (Wu et al., 2025; Hu et al., 2025; Zhang et al., 2026), reporting group fairness metrics aggregated across queries.

Hantian Zhang Meta hantian@meta.com

Steven Euijong Whang\* KAIST swhang@kaist.ac.kr

Consider a RAG system answering questions on different topics. The system may favor women when answering questions about human rights but favor men when answering questions about actors and filmmakers. A metric aggregated across queries lets the opposite disparities cancel and can report a nearly neutral system even though disparities persist within individual topics (Figure 1, left). The cancellation motivates topic-level fairness, which requires group fairness to hold within each topic rather than only in aggregate.

Existing debiasing methods for RAG typically apply a common debiasing strategy across queries, often debiasing the retrieved results without direct supervision from the generated output (Zhao et al., 2026; Kim et al., 2025). These methods face two limitations. First, topics can have disparities in opposite directions, so a correction that improves fairness on one topic may worsen it on another. Second, fairness in the retrieved results does not necessarily translate to fairness in the generated output, which a person receives (Hu et al., 2025).

We address both limitations by correcting each topic separately, under direct supervision from the generated output rather than the retrieved results alone. Separate corrections move each topic in the direction that topic requires, unaffected by the opposite corrections of other topics (Figure 1, center). The corrections of all topics share one pretrained reranker through LoRA (Hu et al., 2022). In its basic form, a RAG pipeline retrieves documents from a corpus, reranks them to select the few that will be used, and generates the output from those. Of these components we adapt only the reranker, separately for each topic. The adaptation changes the documents the generator receives without rebuilding the retrieval index or retraining the generator. Appendix A explains why we adapt the reranker rather than the retriever.

A reranker is a pretrained scorer, and there is no explicit value of fairness to train it toward. We therefore formulate the adaptation of the reranker as preference learning. The reward model of RLHF (Ouyang et al., 2022) is trained the same way: on human preferences between two responses, with a Bradley–Terry objective (Bradley and Terry, 1952) that maximizes the probability that the preferred response is chosen over the non-preferred one. We adopt the objective, but build preference pairs of documents for each query from the generator’s own output rather than from human labels (Section 4).

![](images/e2ca42fbf142ff81c3c38ab6b0d369351c305a2c9b6b27a7b44b77949650c3e0.jpg)  
Figure 1: Overview. The group disparity (GD) of a RAG system between genders differs in sign across topics (left; before adaptation, 5 of the 9 topics on RoMQA). Opposing topics cancel in a signed mean: avg(GD) is −0.0004, nearly neutral, while avg |GD|, the mean absolute per-topic disparity, is 0.0175. Adapting the reranker separately for each topic in the direction that topic requires, on preference pairs built from the generator’s own output (center), moves every shown topic closer to zero and lowers avg |GD| to 0.0070 (right).

We evaluate the adaptation under two protocols, in-dataset and cross-dataset, to assess its effectiveness also on questions beyond the training data. The reranker is adapted on RoMQA (Zhong et al., 2023), a multi-answer question answering dataset, and evaluated in-dataset on RoMQA itself and cross-dataset on TREC Fair Ranking (Ekstrand et al., 2022), an independently constructed fair ranking benchmark on which Wu et al. (2025) measure the fairness of RAG. The adaptation reduces the mean absolute per-topic disparity more than any retrieval-side or generation-side debiasing baseline, by 51% in-dataset and by 40% cross-dataset, while retaining accuracy (Sections 5.2.1 and 5.2.2).

To summarize, we make three contributions. First, we show that group fairness metrics aggregated across queries can conceal topic-level disparities, as topics may favor different sensitive groups. Second, we propose a framework that adapts the reranker for topic-level fairness under direct supervision from the generated output: the direction that each topic requires is measured before adaptation, preference pairs are built from the generator’s own output, and the reranker is adapted separately for each topic with a Bradley–Terry objective. Third, we show that the reduction in disparity holds under the in-dataset and the cross-dataset protocol.

## 2 Related Work

Reranker training in RAG. The retriever and the reranker of a RAG pipeline are conventionally trained on relevance labels (Karpukhin et al., 2020; Nogueira and Cho, 2019). In pipelines of a retriever and a generator alone, the retriever has been trained on feedback from the generator (Izacard et al., 2023; Shi et al., 2024); in a pipeline that also reranks, Wu et al. (2026) adapt the reranker with Proximal Policy Optimization, rewarding the correctness of the generated output. We also adapt the reranker from the generator’s output, but on preference pairs built for topic-level fairness.

Measuring bias in the generated output. Wu et al. (2025) show that utility-optimized RAG systems are biased in both the retrieved results and the generated output. Hu et al. (2025) show that retrieval can bias the output of a fair generator, even when every biased document is removed from the corpus. Zhang et al. (2026) show that the bias of the retrieved documents is amplified in the generated output. We also measure the bias in the generated output, but at the topic level rather than as a whole, since opposing topics can cancel.

Retrieval-side debiasing. Most fairness work in retrieval debiases the retrieved results without direct supervision from the generated output. In ranking, fairness constraints balance the exposure each sensitive group receives across a ranked list (Singh and Joachims, 2018) or the share of each sensitive group in the top-k (Geyik et al., 2019), and Rekabsaz et al. (2021) train a ranker adversarially against the sensitive attribute. In RAG, Kim and Diaz (2025) bring equal exposure into the pipeline, but the fairness they study holds among documents, not between sensitive groups. ReFaRAG (Zhao et al., 2026) fixes the share of each sensitive group among the documents the generator receives, without training. Kim et al. (2025) fine-tune the retriever under several settings, such as the number of epochs and of fine-tuned layers, and identify the setting in which the generated output is unbiased on test queries. FairFT (Zhang et al., 2025) is an exception: it fine-tunes the retriever to favor the documents that make the generator more likely to give the fair answer. However, in our queries the fair answer is Both, the correct reply, while the disparity arises in replies with a single candidate (Section 3.2), so the probability of the fair answer carries no direction. We instead build preference pairs from the generated output, where a reply shows which candidate a document supports, so no document needs a sensitive-group label.

## 3 Problem Definition

We measure disparity on queries built from a multianswer question answering (QA) dataset, adapting the construction of Wu et al. (2025), and define topic-level fairness. The problem is then to adapt the reranker so that the disparity of every topic is reduced while the rate of correct replies is retained.

## 3.1 Query Construction

Measuring disparity in the generated output needs a question whose correct answer is indifferent to the sensitive attribute, and a multi-answer QA dataset whose golden answers are people meets the requirement. We assume the sensitive attribute is binary, with groups $z _ { 1 }$ and $z _ { 2 }$ , and the dataset supplies the group of every golden answer. Given a question, choosing a golden answer from each group gives two candidates, $c _ { z _ { 1 } }$ and $c _ { z _ { 2 } }$ . Each triple consisting of the question and the two candidates is formatted as a multiple-choice query x with five options: cand , can $\mathsf { d } _ { 2 } ,$ Both, Neither, and Insufficient information to decide. Appendix B.1 gives the prompt of each dataset in full.

## 3.2 Group Disparity

The group disparity GD (Calders and Verwer, 2010; Wu et al., 2025) is the difference in how frequently the system replies with the candidate of each group. Let $P ( z _ { 1 } )$ and $P ( z _ { 2 } )$ be the fractions of queries to which the system replies with the $z _ { 1 }$ and with the $z _ { 2 }$ candidate; a reply of Both, which is correct, counts toward neither fraction. Then

$$
G D = P ( z _ { 1 } ) - P ( z _ { 2 } ) ,
$$

positive when the system favors $z _ { 1 }$ candidates and vice versa. GD does not depend on whether a reply is correct, so the rate of correct replies has to be retained as GD is reduced. One question can yield many queries, by different choices of a golden answer from each group. Thus, a tendency toward a single candidate is diluted by other golden answers, and only a tendency toward one group survives.

## 3.3 Topic-Level Fairness

We define topic-level fairness: a system is fair when it is fair on every topic. Let X be the full query set, $\tau$ a set of topics of interest, and $\mathcal { X } _ { t } \subseteq \mathcal { X }$ the queries whose question belongs to topic $t ; G D _ { t }$ is then the group disparity computed on $\mathcal { X } _ { t }$ . Unlike a signed mean, the mean absolute per-topic group disparity,

$$
| G D | = { \frac { 1 } { | T | } } \sum _ { t \in { \mathcal { T } } } | G D _ { t } | ,
$$

does not let opposing topics cancel. Aggregating by the mean lets every topic contribute equally.

A topic represents a subject of questions, and the topic set T is fixed before adaptation. We assume each question belongs to at most one topic, so the sets $\mathcal { X } _ { t }$ are disjoint and need not cover $\mathcal { X } .$ Queries whose question belongs to no topic are excluded from |GD|. Section 5.1.1 names the topic set we adopt and how the questions of the multi-answer QA dataset are assigned to topics.

## 3.4 Problem Statement

We consider the retrieve–rerank–generate pipeline of Figure 2, panel 1, in which we adapt the reranker alone. Given a query x and a corpus C, the retriever $R$ returns a pool of documents $R ( x , { \mathcal { C } } )$ . Let $\pi _ { t }$ be the reranker π adapted for topic $t ,$ and $s _ { \pi _ { t } } ( x , d )$ the score $\pi _ { t }$ assigns to each document d in the pool. The generator G replies to the query concatenated with the k documents of highest reranker score:

$$
\begin{array} { r l r } { { \{ d _ { 1 } , \ldots , d _ { k } \} = \ \mathrm { t o p } \not k } } & { { } { s _ { \pi _ { t } } ( x , d ) , } } & { { } { \vphantom { \int _ { k } ^ { d _ { 1 } } \ldots \mathrm { t o p } \not k } } } \\ { { } } & { { } { \hat { y } = G ( x , \ d _ { 1 } \oplus \cdots \oplus d _ { k } ) . } } & { { } { } } \end{array}
$$

$G , { \mathcal { C } } ,$ , and R are fixed. Given a topic set $\tau$ and the topic of each query, the problem is to find the rerankers $\pi _ { t }$ that minimize $| G D _ { t } |$ for every $t \in \tau$ while retaining the rate of correct replies. Appendix B.2 gives the prompt G replies to.

![](images/24bdc4564d8b56b579b1f6cbcd9ca5cc1400c4c7fc5645f56bdd782a99a4a964.jpg)  
Figure 2: Illustrative example. The method on two topics whose directions of correction are opposite. (1) The retrieve–rerank–generate pipeline, in which the reranker alone is adapted, separately for each topic. (2) Preference pairs (Section 4.2): the system favors men on actors and filmmakers before adaptation, so $\delta _ { \mathrm { A F } }$ is female, and women on human rights, so $\delta _ { \mathrm { H R } }$ is male. For each retrieved document $d , n = 5$ replies are sampled from $G ( x , d )$ and shown beside it, colored by candidate; a document from which no reply is a candidate falls outside $D _ { + }$ , and of the two that remain, the one with the higher ${ v } _ { \delta _ { t } }$ is preferred. (3) Each reranker $\pi _ { t }$ is adapted on the preference pairs $\mathcal { P } _ { t }$ (Section 4.3); at the right, each query’s top-k documents before and after adaptation are shown.

## 4 Method

We mitigate topic-level bias by adapting the reranker separately for each topic, with a Bradley– Terry objective (Bradley and Terry, 1952) in the direction that topic requires. Our framework builds preference pairs of documents for each query, using the generator’s own output, and then adapts the reranker under an anchor on the scores before adaptation. Figure 2 traces the method on two topics whose directions of correction are opposite.

## 4.1 Preference Learning from Comparisons

A Bradley–Terry objective is an appropriate choice when there are no explicit values but only relative preferences between items. Let $\sigma$ be the logistic function: the Bradley–Terry objective maximizes $\sigma ( s _ { \pi _ { t } } ( x , d _ { c } ) - s _ { \pi _ { t } } ( x , d _ { r } ) )$ , the probability that the preferred document $d _ { c }$ is chosen over the nonpreferred document $d _ { r }$ . Since there is no explicit value to train $s _ { \pi _ { t } } ( x , d )$ for topic-level fairness, we build preference pairs and adapt the reranker with the Bradley–Terry objective.

## 4.2 Building Preference Pairs

A preference pair holds two documents, both from the pool $R ( x , { \mathcal { C } } )$ of one query x: the reranker mitigates topic-level bias by ordering that pool, and the order decides which k documents enter the generator’s input. The preferred document of a pair more strongly supports the candidate of the group that the topic of x disfavors. Algorithm 1 builds the set $\mathcal { P } _ { t }$ of preference pairs $( x , d _ { c } , d _ { r } )$ for topic t.

```latex
Algorithm 1 Preference pairs for topic t
1: δ<sub>t</sub> ← direction of correction for t before adaptation
2: $\mathcal { P } _ { t }  \emptyset$
3: for all queries x of topic t do
4: $D \dot { \gets } R ( x , \mathcal { C } )$
5: for all $d \in D$ do
6: sample n replies $\hat { y } _ { d } ^ { 1 } , \dots , \hat { y } _ { d } ^ { n } \sim G ( x , d )$
7: $\smash { v _ { g } ( x , d ) \gets \frac { 1 } { n } | \{ i : \hat { y } _ { d } ^ { i } = c _ { g } \} | }$ for $g \in \{ z _ { 1 } , z _ { 2 } \}$
8: end for
9: $D _ { + } \gets \{ d \in D : v _ { z _ { 1 } } ( x , d ) > 0 \mathrm { ~ o r ~ } v _ { z _ { 2 } } ( x , d ) > 0 \}$
10: $\mathcal { E }  \{ ( \tilde { d } _ { c } , d _ { r } ) \in D _ { + } ^ { 2 } : v _ { \delta _ { t } } ( x , d _ { c } ) > \overline { { v _ { \delta _ { t } } ( x , d _ { r } ) } } \}$
11: $\hat { \mathcal { E } }  \mathrm { u p }$ to κ pairs drawn from $\mathcal { E }$ without replacement,
w.p. ∝ $v _ { \delta t } ( x , d _ { c } ) - v _ { \delta t } ( x , d _ { r } )$
12: $\mathcal { P } _ { t }  \mathcal { P } _ { t } \cup \{ ( x , d _ { c } , d _ { r } ) : ( d _ { c } , d _ { r } ) \in \hat { \mathcal { E } } \}$
13: end for
14: return $\mathcal { P } _ { t }$
```

Topic t carries a direction of correction $\delta _ { t } .$ , measured before adaptation (line 1): $\delta _ { t }$ is z<sub>2</sub> where $G D _ { t }$ is positive, and vice versa. In Figure 2, positive $G D _ { \mathrm { A F } }$ makes $\delta _ { \mathrm { A F } }$ female, and negative $G D _ { \mathrm { H R } }$ makes $\delta _ { \mathrm { H R } }$ male. Adaptation on $\mathcal { P } _ { t }$ aims to bring $G D _ { t }$ to zero, but the pairs carry only the direction of correction: $G D _ { t }$ can either stop before reaching zero or cross zero. How far $G D _ { t }$ moves is decided by $| G D _ { t } |$ on the validation split (Section 4.3).

Each retrieved document is then probed with the generator on the query and that document alone, $G ( x , d )$ (lines 5–8). A reply from $G ( x , d )$ shows which candidate d supports, so preferring the documents that support the $\delta _ { t }$ candidate can mitigate topic-level bias. Sampling n replies from $G ( x , d )$ gives the vote shares $v _ { z _ { 1 } } ( x , d )$ and $v _ { z _ { 2 } } ( x , d )$ , the fractions of the replies that are the $z _ { 1 }$ and the $z _ { 2 }$ candidate. With n sampled replies, ${ v } _ { \delta _ { t } }$ is graded rather than binary, so two documents from which the generator replies with the same candidate can still differ in ${ v } _ { \delta _ { t } }$ and form a pair.

Pairs are formed among the documents from which at least one reply is a candidate $( D _ { + }$ , lines 9–12), since only those documents support a candidate. Within a pair, $d _ { c }$ is the document with the higher ${ v } _ { \delta _ { t } }$ . At most κ pairs are drawn per query, with probability proportional to the difference in ${ v } _ { \delta _ { t } }$ , so that no single query dominates a topic’s supply. For the actors and filmmakers query in Figure 2, no reply sampled from $G ( x , d _ { 2 } )$ is a candidate, so $d _ { 2 }$ falls outside $D _ { + }$ Sampling from $G ( x , d _ { 6 } )$ and $G ( x , d _ { 1 } )$ gives the vote shares $v _ { f } ( x , d _ { 6 } ) ~ = ~ 4 / 5$ and $v _ { m } ( x , d _ { 1 } ) \ = \ 5 / 5 .$ so both documents are in $D _ { + }$ With $\delta _ { \mathrm { A F } } = \mathrm { f e } -$ male, $v _ { f } ( x , d _ { 6 } ) > v _ { f } ( x , d _ { 1 } ) = 0$ makes $d _ { 1 }$ the non-preferred document, and the query contributes $( x , d _ { 6 } , d _ { 1 } )$ to $\mathcal { P } _ { \mathrm { A F } }$

## 4.3 Training Objective

The reranker for topic t is adapted with the Bradley– Terry objective under an anchor on the scores before adaptation. Let $s _ { \pi } ^ { \mathrm { r e f } }$ be the scores before adaptation and $\lambda _ { t }$ the anchor weight. The loss is

$$
\begin{array} { r l r } {  { \mathcal { L } = \frac { 1 } { | \mathcal { P } _ { t } | } \big ( \mathcal { L } _ { \mathrm { B T } } + \lambda _ { t } \mathcal { L } _ { \mathrm { a n c } } \big ) , } } \\ & { } & { \mathcal { L } _ { \mathrm { B T } } = - \sum _ { \mathcal { P } _ { t } } \log \sigma \big ( s _ { \pi _ { t } } ( x , d _ { c } ) - s _ { \pi _ { t } } ( x , d _ { r } ) \big ) , } \\ & { } & { \mathcal { L } _ { \mathrm { a n c } } = \frac { 1 } { 2 } \sum _ { \mathcal { P } _ { t } } \sum _ { d \in \{ d _ { c } , d _ { r } \} } \big ( s _ { \pi _ { t } } ( x , d ) - s _ { \pi } ^ { \mathrm { r e f } } ( x , d ) \big ) ^ { 2 } . } \end{array}
$$

The Bradley–Terry objective $\mathcal { L } _ { \mathrm { B T } }$ widens the gap between the scores of $d _ { c }$ and $d _ { r } .$ , and the anchor ${ \mathcal { L } } _ { \mathrm { a n c } }$ holds each score near its value before adaptation. We adjust $\lambda _ { t }$ and the learning rate $\eta _ { t }$ of topic t to minimize $| G D _ { t } |$ on the validation split, subject to a fall of at most 0.5 points from the rate of correct replies on topic t before adaptation. We evaluate on the test split without readjustment.

## 5 Experiments

Under both the in-dataset and the cross-dataset protocol, our adaptation of the reranker reduces the mean absolute per-topic group disparity while accuracy is retained, and no baseline reaches the reduction. The baselines do not reach the reduction even when they treat each topic separately, and neither does our method with one reranker for all topics. An ablation shows that the reduction also needs a direction of correction for each topic and the anchor on the scores before adaptation.

![](images/8c84f29c1a1c742c0e90901cf9f93767dced23b3c9e678cec52b5e8adbc60d87.jpg)  
Figure 3: Difference in recall@5 between the articles of the male and the female candidate, and $G D _ { t } .$ , of the unadapted system on RoMQA. In the shaded rows, the system favors the candidate of higher recall@5. The sign of $G D _ { t }$ does not follow from recall@5 alone.

## 5.1 Experimental Setup

## 5.1.1 Datasets

Our query construction needs a multi-answer QA dataset whose golden answers are people. RoMQA (Zhong et al., 2023) is one, and supplies the training questions and the in-dataset evaluation. Its questions are written by crowdworkers from constraints mined from Wikidata, such as occupation and place of birth. TREC Fair Ranking (Ekstrand et al., 2022) becomes one through the conversion of Wu et al. (2025), and supplies the cross-dataset evaluation. In the original TREC Fair Ranking, a system ranks English Wikipedia articles for a WikiProject, a subject area that Wikipedia editors maintain; Wu et al. (2025) convert each WikiProject into a question asking which candidate is relevant to it. The sensitive attribute is gender, with $z _ { 1 }$ male and z female.

Both datasets share one topic set $\tau { : }$ the 9 WikiProjects of TREC Fair Ranking that hold enough candidates of each gender, to which enough RoMQA questions belong, and whose direction of correction is reproducible on both datasets. For RoMQA, a language model judges whether each question is related to each of the 9 topics, and then chooses the one the question is mainly about. A question related to no topic belongs to none. Appendix B.3 gives the two prompts. Appendix C gives the statistics on the two datasets and the criteria of topic selection in detail.

Over these topics, the disparity arises in both retrieval and generation, so a correction that debiases the retrieved results without direct supervision from the generated output still leaves the disparity the generator adds. Figure 3 compares the difference in recall@5 between the Wikipedia articles of the male and the female candidate with $G D _ { t }$ for each RoMQA topic under the unadapted system, and Figure 6 adds TREC. On 4 of the 9 RoMQA topics and 5 of the 9 TREC topics, the system favors the candidate whose article has the higher recall@5. On RoMQA, the article of the female candidate has the higher recall@5 on 8 of the 9 topics, yet the system favors the male candidate on 4 topics.

Both protocols adapt the reranker on the RoMQA training split and measure $\delta _ { t }$ on a validation split; the in-dataset protocol takes the validation and test splits from RoMQA, and the crossdataset protocol from TREC.

## 5.1.2 Metrics

We report fairness as the mean absolute per-topic group disparity |GD| over the 9 topics and as $G D _ { t }$ for each topic. |GD| can fall while a topic moves away from zero, so we also count the topics that move in their direction of correction. Utility is accuracy, the rate of the reply Both: both candidates of a query are golden answers of its question.

## 5.1.3 Implementation Details

The retriever, the reranker, and the generator are e5-base-v2 (Wang et al., 2022), bge-reranker-v2-m3 (Chen et al., 2024), and Meta-Llama-3-8B-Instruct (AI@Meta, 2024); Appendix E replaces the reranker and the generator with other models. The retriever returns a pool of 30 documents, and the generator replies from the $k = 5$ documents of highest reranker score. Since the retriever and the reranker are smaller than the generator, they receive a rewrite of the query that keeps only the question and the two candidates; the generator receives the full query. Preference pairs are built from $n = 5$ replies sampled per document, with at most $\kappa \ : = \ : 4$ pairs per query. For each topic, the learning rate $\eta _ { t }$ is adjusted over $\{ 1 , 3 , 1 0 , 3 0 \} \times 1 0 ^ { - 6 }$ and the anchor weight $\lambda _ { t }$ over {0, 0.01, 0.03, 0.3}. Appendix D gives the remaining settings and the computational resources.

## 5.1.4 Baselines

The five baselines share the corpus, the generator, k, and the rewritten query with our method, and fall into three categories:

No debiasing. Unadapted uses the original pretrained reranker, while RRPO (Wu et al., 2026) adapts the same reranker to reward correct replies.

Retrieval-side debiasing. Gender-balanced topk (GB) (Geyik et al., 2019) balances the top-k between male- and female-associated documents; Retriever debiasing (RD) (Kim et al., 2025) fine-tunes the retriever.

Generation-side debiasing. Self-debiasing (SD) (Schick et al., 2021) adjusts token probabilities during each step of decoding to discourage gender-biased generation.

For each baseline, we select the setting of lowest validation |GD| among those that lose at most 0.5 points of accuracy against Unadapted. If every setting loses more than 0.5 points, we select the setting of highest validation accuracy. Our method follows the same rule on each topic. Appendix D.3 describes how each baseline is implemented.

## 5.2 Main Results

Ours reduces |GD| by 51% in-dataset and by 40% cross-dataset while retaining accuracy, and moves most topics in their direction of correction. Table 1 shows the fairness and utility of the baselines and Ours under the in-dataset and the cross-dataset protocol. Appendix E.1 gives $G D _ { t }$ for every topic.

## 5.2.1 In-Dataset Protocol

Ours reduces |GD| from 0.0169 to 0.0083, the lowest of all systems, and moves 8 of the 9 topics in their direction (Table 1). Figure 4 and Figure 7 show $G D _ { t }$ of every system, and Table 8 lists the values. Medicine is the only topic whose $G D _ { t }$ moves against its direction, from +0.0046 to +0.0058 (Table 8). This result may be explained by the limited informativeness of the retrieved documents for Medicine queries. Such queries often ask about details that biographies rarely report, such as a cause of death with an exclusion constraint (“Who died of a stroke? Don’t include Canadians.”), so a reply from $G ( x , d )$ is seldom a candidate, and the topic yields 0.38 preference pairs per training query against at least 0.82 on every other topic. The limited informativeness is also reflected in the topic’s accuracy, 17.7%, the lowest of all topics. Full statistics of the generator’s replies by topic are reported in Table 9 and Table 10.

<table><tr><td></td><td colspan="4">In-dataset (RoMQA)</td><td colspan="4">Cross-dataset (TREC)</td></tr><tr><td>System</td><td>|GD|</td><td>Accuracy</td><td>Male-fav.</td><td>Female-fav.</td><td>|GD|</td><td>Accuracy</td><td>Male-fav.</td><td>Female-fav.</td></tr><tr><td>Unadapted</td><td>0.0169</td><td>55.1</td><td></td><td></td><td>0.0156</td><td>89.9</td><td>一</td><td></td></tr><tr><td>RRPO</td><td>0.0170</td><td>56.1</td><td>1/4</td><td>2/5</td><td>0.0156</td><td>89.9</td><td>3/6</td><td>2/3</td></tr><tr><td>GB</td><td>0.0193</td><td>55.4</td><td>1/4</td><td>1/5</td><td>0.0112</td><td>92.1</td><td>4/6</td><td>2/3</td></tr><tr><td>RD</td><td>0.0162</td><td>54.3</td><td>1/4</td><td>4/5</td><td>0.0143</td><td>89.9</td><td>6/6</td><td>0/3</td></tr><tr><td>SD</td><td>0.0174</td><td>54.9</td><td>0/4</td><td>4/5</td><td>0.0166</td><td>89.4</td><td>2/6</td><td>1/3</td></tr><tr><td>Ours</td><td>0.0083</td><td>55.2</td><td>3/4</td><td>5/5</td><td>0.0094</td><td>90.2</td><td>6/6</td><td>2/3</td></tr></table>

Table 1: Fairness and utility of the baselines and Ours under the in-dataset and the cross-dataset protocol. In Male-fav. and Female-fav., $k / n$ means that k of the n topics on which the unadapted system favors the male or the female candidate move in their direction of correction. Bold marks the lowest $| G D |$ and the highest k per protocol.

<table><tr><td rowspan=1 colspan=4>I Unadapted RRPOGB •RD▲ SD ★ Ours</td></tr><tr><td rowspan=1 colspan=1>Actors and Filmmakers</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Computer science</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Former countries</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Human rights</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Literature</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Medicine</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Netherlands</td><td rowspan=1 colspan=1>★</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Sociology</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=2 colspan=1>Southeast Asia-.0</td><td rowspan=1 colspan=1>• </td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=2>+.04</td></tr></table>

Figure 4: $G D _ { t }$ of Table 1 under the in-dataset protocol. Ours moves 8 of 9 topics in their direction of correction.

RRPO achieves the highest accuracy, likely from its reward on the correct reply, but does not reduce |GD|. GB fails to lower $| G D |$ : balancing the top-k by gender may mostly exchange documents about people other than the candidates. RD fine-tunes the retriever but corrects every topic in one direction: most female-favoring topics move toward the male candidate, while most male-favoring topics move away from zero, so |GD| falls only slightly. SD debiases the generator without a direction for any topic, and leaves |GD| nearly unchanged while lowering accuracy slightly.

The reduction is smaller with another reranker and under two other generators. Adapted the same way, jina-reranker-v2-base-multilingual, a smaller model trained on other data, reaches |GD| of 0.0108 from 0.0160 (Table 11). Under Qwen2.5-7B-Instruct, a model of another family, and Meta-Llama-3-70B-Instruct, a larger model of the same release, Ours reaches |GD| of 0.0063 and 0.0202 from 0.0092 and 0.0302, respectively, while accuracy is retained (Table 12).

## 5.2.2 Cross-Dataset Protocol

The reduction holds under the cross-dataset protocol, although the preference pairs are built from the RoMQA training split alone. Ours reduces |GD| from 0.0156 to 0.0094, the lowest of all systems, and moves 8 of the 9 topics in their direction (Figure 7b), while accuracy rises by 0.3 points (Table 1). Human rights is the one topic that does not move: its $G D _ { t }$ is $- 0 . 0 0 7 5$ against −0.0073 before adaptation, already close to zero (Table 8).

In contrast to the in-dataset result, GB lowers |GD| more than any other baseline and raises accuracy, though Ours still shows a lower |GD| (Table 1). On TREC, 32.0% of the top-k slots of the unadapted system hold a document associated with neither gender, which is not about a person and is likely less informative to the generator. While balancing the genders in the top-k, GB can fill these slots with documents about people and so may make the generator’s input more informative as a side effect. Every RoMQA document is already about a person. The other baselines change |GD| little, as under the in-dataset protocol.

## 5.3 Per-Topic Against Shared Correction

Ours corrects each topic with a reranker of its own, while each baseline applies one correction to every topic. We therefore modify Ours to adapt one reranker for all topics (Ours, one reranker), and RD and SD to treat each topic separately (RD per topic and SD per topic). Table 2 shows the fairness and utility of the modified systems under the in-dataset protocol, and Table 13 in Appendix E.4 under both protocols. Figure 5 and Figure 8 show $G D _ { t }$ of these systems, and Table 14 lists the values.

Ours needs a separate reranker for each topic, even though its pairs carry the direction of each topic. The single reranker follows the direction that most topics require, regardless of the direction of each: only 1 of the 4 male-favoring RoMQA topics and none of the 3 female-favoring TREC topics move in their own direction (Table 13).

<table><tr><td rowspan="2"></td><td colspan="4">In-dataset (RoMQA)</td></tr><tr><td>|GD|</td><td>Accuracy</td><td>Male- fav.</td><td>Female- fav.</td></tr><tr><td>System Unadapted</td><td>0.0169</td><td>55.1</td><td>一</td><td>一</td></tr><tr><td>Ours, one reranker</td><td>0.0175</td><td>55.5</td><td>1/4</td><td>5/5</td></tr><tr><td>RD per topic</td><td>0.0118</td><td>54.8</td><td>4/4</td><td>4/5</td></tr><tr><td>SD per topic</td><td>0.0170</td><td>55.0</td><td>0/4</td><td>3/5</td></tr><tr><td>Ours</td><td>0.0083</td><td>55.2</td><td>3/4</td><td>5/5</td></tr></table>

Table 2: Ours with one reranker shared by all topics, and the systems that treat each topic separately, under the in-dataset protocol. The columns are defined as in Table 1. Ours, one reranker moves most topics toward the male candidate.

![](images/2008c57b83c90280725128a4d30833e29f6acabf38fdbeb83bfe685d957c9d84.jpg)  
Figure 5: $G D _ { t }$ of Table 2 under the in-dataset protocol. Ours is the closest to zero overall.

RD per topic moves as many topics in their direction as Ours but keeps a higher |GD|, since the retriever learns from the gender of each document rather than from the generated output. Treating each topic separately changes little for SD, since SD per topic chooses only the strength of its correction for each topic, not the direction.

## 5.4 Ablation Study

Ours adapts the reranker of each topic in the direction that topic requires, under an anchor on the scores before adaptation. We remove these two parts one at a time: one direction corrects every topic in the direction that most topics require, and without anchor sets $\lambda _ { t } = 0$ for every topic. Table 3 shows the fairness and utility of the variants under the in-dataset protocol, Table 15 in Appendix E.5 under both protocols, and Table 16 lists their $G D _ { t }$

<table><tr><td rowspan="2"></td><td colspan="4">In-dataset (RoMQA)</td></tr><tr><td>|GD|</td><td>Accuracy</td><td>Male- fav.</td><td>Female- fav.</td></tr><tr><td>Unadapted Ours</td><td>0.0169 0.0083</td><td>55.1 55.2</td><td>一 3/4</td><td>一 5/5</td></tr><tr><td>one direction</td><td>0.0176</td><td>54.8</td><td>0/4</td><td>5/5</td></tr><tr><td>without anchor</td><td>0.0142</td><td>55.0</td><td>0/4</td><td>4/5</td></tr></table>

Table 3: Ablation under the in-dataset protocol. The columns are defined as in Table 1. One direction corrects every topic in the direction that most topics require; without anchor sets $\lambda _ { t } = 0$ for every topic.

Ours needs a direction for each topic, since a single direction pushes the topics that require the other direction away from zero. On RoMQA, one direction moves none of the 4 male-favoring topics toward the female candidate. Moreover, the two that start farthest from zero move further toward the male candidate, more than doubling in $G D _ { t } \mathbf { \cdot }$ Actors and Filmmakers from +0.0211 to +0.0495, and Netherlands from +0.0194 to +0.0518 (Table 16). On TREC, none of the 3 female-favoring topics moves toward the male candidate (Table 15).

The anchor lets a reranker take a step large enough to correct its topic while retaining accuracy. Without the anchor, a larger step raises $| G D _ { t } |$ or loses accuracy on most topics, so without anchor retains accuracy with smaller steps but moves fewer topics in their direction than Ours, 4 against 8 on RoMQA and 6 against 8 on TREC (Table 15).

## 6 Conclusion

The group disparity of a RAG system can differ in sign across topics: a signed mean over topics can report a nearly neutral system, and a correction in one direction for all topics pushes the topics that require the other direction away from zero. We therefore define topic-level fairness and adapt the reranker separately for each topic under direct supervision from the generated output: preference pairs built from the generator’s own output carry the direction that topic requires. The anchor on the scores before adaptation lets a reranker take a step large enough to correct its topic while accuracy is retained. The adaptation reduces the mean absolute per-topic group disparity by 51% in-dataset on RoMQA and by 40% cross-dataset on TREC Fair Ranking while retaining accuracy; no baseline reaches the reduction, even when modified to treat each topic separately, and neither does our method with one reranker for all topics.

## Limitations

• Preference pairs from $G ( x , d )$ . The effect of one document on the generator’s reply depends on the other $k - 1$ documents in its input, but our method measures the effect on $G ( x , d )$ , the reply to that document alone. Izacard et al. (2023) likewise train a retriever from the output on each document alone for a reader that attends over all documents jointly, and find it on par with scoring each document by its removal from the top k.

• Topic set. A deployed RAG system takes the topics its provider targets; we choose the topic set and the topic of each RoMQA question by criteria of our own. WikiProjects are a natural topic set: a WikiProject is a subject area that Wikipedia editors maintain, and each question of TREC Fair Ranking is converted from one WikiProject. The 9 WikiProjects we use pass the three filters of Appendix C, and each RoMQA question belongs to the one topic it is mainly about. We leave the questions that belong to several topics at once to future work.

• Aggregation over topics. Any single value over the topics conceals what happened on each topic: neither |GD| nor any other aggregated value guarantees a lower $\left| G D _ { t } \right|$ on every topic. A fairness condition satisfied on the aggregate but violated on one topic is analogous to the fairness gerrymandering of Kearns et al. (2018), so we report every $G D _ { t }$ beside |GD|.

• Fairness measure. GD captures one aspect of fairness, and a lower GD is best read as the system replying more evenly with the candidate of each group.

• One sensitive attribute. The sensitive attribute is gender in every experiment. Another attribute may enter through the same construction wherever a dataset supplies it for every golden answer.

• The attribute is binary. Our query construction assumes a binary sensitive attribute; an attribute with more than two values would have to be reduced to two groups at a time, and we have not tested that reduction.

• English-only. All experiments are in English; cross-lingual fairness is out of scope.

## Ethical Considerations

Gender is treated as binary. The male and female labels of RoMQA and TREC Fair Ranking come from Wikidata. We adopt the labels for measurement convenience: a per-topic rate needs enough queries in each group, and the datasets supply that for two labels alone. Non-binary and transgender people are therefore absent from every number we report and every correction our method makes.

## References

AI@Meta. 2024. Llama 3 model card. https://github.com/meta-llama/llama3/ blob/main/MODEL\_CARD.md.

Ralph Allan Bradley and Milton E Terry. 1952. Rank analysis of incomplete block designs: I. the method of paired comparisons. Biometrika, 39(3/4):324– 345.

Toon Calders and Sicco Verwer. 2010. Three naive Bayes approaches for discrimination-free classification. Data Mining and Knowledge Discovery, 21(2):277–292.

Jianlyu Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, and Zheng Liu. 2024. M3- Embedding: Multi-linguality, multi-functionality, multi-granularity text embeddings through selfknowledge distillation. In Findings of the Association for Computational Linguistics: ACL 2024, pages 2318–2335.

Matthijs Douze, Alexandr Guzhva, Chengqi Deng, Jeff Johnson, Gergely Szilvasy, Pierre-Emmanuel Mazaré, Maria Lomeli, Lucas Hosseini, and Hervé Jégou. 2024. The Faiss library. arXiv preprint arXiv:2401.08281.

Cynthia Dwork and Christina Ilvento. 2019. Fairness under composition. In 10th Innovations in Theoretical Computer Science Conference (ITCS), volume 124 of LIPIcs, pages 33:1–33:20.

Michael D. Ekstrand, Graham McDonald, Amifa Raj, and Isaac Johnson. 2022. Overview of the TREC 2022 fair ranking track. In Proceedings of the Thirty-First Text REtrieval Conference (TREC 2022). ArXiv:2302.05558.

Sahin Cem Geyik, Stuart Ambler, and Krishnaram Kenthapadi. 2019. Fairness-aware ranking in search & recommendation systems with application to LinkedIn talent search. In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, pages 2221–2231.

Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2022. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations.

Mengxuan Hu, Hongyi Wu, Ronghang Zhu, Zihan Guan, Dongliang Guo, Daiqing Qi, and Sheng Li. 2025. No free lunch: Retrieval-augmented generation undermines fairness in LLMs, even for vigilant users. In Findings of the Association for Computational Linguistics: EMNLP 2025, pages 18145– 18170.

Gautier Izacard, Patrick Lewis, Maria Lomeli, Lucas Hosseini, Fabio Petroni, Timo Schick, Jane Dwivedi-Yu, Armand Joulin, Sebastian Riedel, and Edouard Grave. 2023. Atlas: Few-shot learning with retrieval augmented language models. Journal of Machine Learning Research, 24(251):1–43.

Jiajie Jin, Yutao Zhu, Zhicheng Dou, Guanting Dong, Xinyu Yang, Chenghao Zhang, Tong Zhao, Zhao Yang, and Ji-Rong Wen. 2025. FlashRAG: A modular toolkit for efficient retrieval-augmented generation research. In Companion Proceedings ofthe ACM on Web Conference 2025, pages 737–740. ACM.

Jina AI. 2024. jina-reranker-v2-basemultilingual. https://huggingface.co/jinaai/ jina-reranker-v2-base-multilingual.

Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. 2020. Dense passage retrieval for opendomain question answering. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 6769–6781.

Michael Kearns, Seth Neel, Aaron Roth, and Zhiwei Steven Wu. 2018. Preventing fairness gerrymandering: Auditing and learning for subgroup fairness. In Proceedings ofthe 35th International Conference on Machine Learning, pages 2564–2572.

Taeyoun Kim, Jacob Mitchell Springer, Aditi Raghunathan, and Maarten Sap. 2025. Mitigating bias in RAG: Controlling the embedder. In Findings of the Associationfor Computational Linguistics: ACL 2025, pages 18999–19024.

To Eun Kim and Fernando Diaz. 2025. Towards fair RAG: On the impact of fair ranking in retrievalaugmented generation. In Proceedings of the 2025 International ACM SIGIR Conference on Innovative Concepts and Theories in Information Retrieval (IC-TIR), pages 33–43.

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. 2023. Efficient memory management for large language model serving with PagedAttention. In Proceedings ofthe ACM SIGOPS 29th Symposium on Operating Systems Principles, pages 611–626.

Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. 2024. Lost in the middle: How language models use long contexts. Transactions ofthe Association for Computational Linguistics, 12:157–173.

Sourab Mangrulkar, Sylvain Gugger, Lysandre Debut, Younes Belkada, Sayak Paul, Benjamin Bossan, and Marian Tietz. 2022. PEFT: State-of-the-art parameter-efficient fine-tuning methods. https: //github.com/huggingface/peft.

Rodrigo Nogueira and Kyunghyun Cho. 2019. Passage re-ranking with BERT. arXiv preprint arXiv:1901.04085.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul F. Christiano, Jan Leike, and Ryan Lowe. 2022. Training language models to follow instructions with human feedback. In Advances in Neural Information Processing Systems, volume 35, pages 27730–27744.

Navid Rekabsaz, Simone Kopeinik, and Markus Schedl. 2021. Societal biases in retrieved contents: Measurement framework and adversarial mitigation of BERT rankers. In Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 306–316.

Timo Schick, Sahana Udupa, and Hinrich Schütze. 2021. Self-diagnosis and self-debiasing: A proposal for reducing corpus-based bias in NLP. Transactions ofthe Associationfor Computational Linguistics, 9:1408– 1424.

Weijia Shi, Sewon Min, Michihiro Yasunaga, Minjoon Seo, Richard James, Mike Lewis, Luke Zettlemoyer, and Wen-tau Yih. 2024. REPLUG: Retrievalaugmented black-box language models. In Proceedings ofthe 2024 Conference ofthe North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 8371–8384.

Ashudeep Singh and Thorsten Joachims. 2018. Fairness of exposure in rankings. In Proceedings ofthe 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, pages 2219–2228.

Liang Wang, Nan Yang, Xiaolong Huang, Binxing Jiao, Linjun Yang, Daxin Jiang, Rangan Majumder, and Furu Wei. 2022. Text embeddings by weaklysupervised contrastive pre-training. arXiv preprint arXiv:2212.03533.

Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, and 3 others. 2020. Transformers: State-of-the-art natural language processing. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 38–45. Association for Computational Linguistics.

Xuyang Wu, Shuowei Li, Hsin-Tai Wu, Zhiqiang Tao, and Yi Fang. 2025. Does RAG introduce unfairness in LLMs? Evaluating fairness in retrieval-augmented generation systems. In Proceedings ofthe 31st International Conference on Computational Linguistics, pages 10021–10036.

Yuhang Wu, Xiangqing Shen, Fanfan Wang, Cangqi Zhou, Zhen Wu, Xinyu Dai, and Rui Xia. 2026. Optimizing RAG rerankers with LLM feedback via reinforcement learning. In Proceedings ofthe 64th Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 30474– 30490.

An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, and 23 others. 2024. Qwen2.5 technical report. arXiv preprint arXiv:2412.15115.

Tianhui Zhang, Yi Zhou, and Danushka Bollegala. 2026. Evaluating the effect of retrieval augmentation on social biases. In Proceedings of the 19th Conference of the European Chapter ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 5004–5026.

Zheng Zhang, Ning Li, Qi Liu, Rui Li, Weibo Gao, Qingyang Mao, Zhenya Huang, Baosheng Yu, and Dacheng Tao. 2025. The other side of the coin: Exploring fairness in retrieval-augmented generation. arXiv preprint arXiv:2504.12323.

Yingqi Zhao, Vasilis Efthymiou, Jyrki Nummenmaa, and Kostas Stefanidis. 2026. ReFaRAG: Re-ranking for bias mitigation in retrieval-augmented generation. In New Trends in Database and Information Systems (ADBIS 2025), volume 2676 of Communications in Computer and Information Science, pages 516–530. Springer.

Victor Zhong, Weijia Shi, Wen-tau Yih, and Luke Zettlemoyer. 2023. RoMQA: A benchmark for robust, multi-evidence, multi-answer question answering. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 7055–7067.

Answer the question based on the given   
document.   
Only give me the answer and do not output   
any other words.   
The following are given documents.   
{reference}   
User message:   
Question: {question}

## A Why the Reranker Is a Suitable Target

Continuing from Section 1, this appendix argues that correcting one component of the pipeline is enough. Of the pipeline’s components, a corpus is rebuilt rather than trained and a generator is expensive to fine-tune and is usually deployed as a general-purpose pretrained model, which leaves the retriever and the reranker, and the reranker is the proper choice.

Correcting one component can leave the system fair at the level of its final output, and correcting every component would not be the way to get there: Dwork and Ilvento (2019) show that components satisfying a group fairness criterion in isolation need not compose into a system that satisfies it. Retrieval pipelines have been corrected this way before: Rekabsaz et al. (2021) reduce bias in retrieved contents by adapting the ranker alone, without losing retrieval utility, and, closest to our work, Kim et al. (2025) fine-tune the retriever alone and identify the setting in which the generated output is unbiased on test queries. Our results are of the same kind: every reduction of our method comes from changing the reranker alone.

A fairness objective on the retriever can lower the retriever’s recall: the retriever decides which documents enter the pool, and a document it leaves out cannot be recovered by any later stage. The retriever is also a bi-encoder, scoring a document by the inner product of a query embedding and a document embedding computed before any query exists, which is what makes an index searchable, so a correction that depends on the sensitive group that a document supports can enter only through the query embedding, where the same inner product must carry relevance at the same time.

The reranker has neither limitation: it reorders the pool the retriever returned, so the retriever’s recall is preserved, and, as a cross-encoder, it jointly encodes the query and the document, so the reranker score can condition on the sensitive group that the document supports directly, not only through a query embedding.

## B Prompts

This appendix reproduces the prompts used by the pipeline and in dataset construction; braces mark substitution slots.

## B.1 Multiple-Choice Query

Continuing from Section 3.1, this is the prompt that wraps a question and its two candidates into one query x, with one template per dataset. TREC Fair Ranking keeps the template of Wu et al. (2025), which turns a WikiProject into a question:

Which answer is relevant to the topic   
’{WikiProject}’? Select the relevant   
answer based on their relationship   
to ’{WikiProject}’, rather than just   
matching words. Choose from the   
following: {c<sub>1</sub>}, {c<sub>2</sub>}, Both, Neither,   
or Insufficient information to decide.

RoMQA supplies questions, so its template places the question directly before the same options:

{RoMQA question} Choose from the   
following: {c<sub>1</sub>}, {c<sub>2</sub>}, Both, Neither,   
or Insufficient information to decide.

{c<sub>1</sub>} and {c<sub>2</sub>} are the two candidate names, in an order randomized with respect to group.

## B.2 Generation

Continuing from Section 3.4, this is the prompt the generator G replies to at temperature 0 for every reported GD and accuracy value. System message:

{question} is the query x itself, and {reference} is the k reranker-selected documents in reranked order, one per line, each rendered as

Doc {i}({title}) {text}

with {title} and {text} taken from the corpus record. The per-document pass of Section 4.2 uses the same template with a single document in {reference} and is sampled (Appendix D.1).

## B.3 LLM Topic Membership

Continuing from Section 5.1.1, these are the two prompts that decide the topic of a RoMQA question (Appendix C), both served by Meta-Llama-3-8B-Instruct at temperature 0. The first is sent once for each question and topic, with max\_tokens = 3, and the question is related to the topic when the response begins with yes:

Is the following question related to the   
topic "{topic}"? Answer only yes or no.   
Question: {question}

The second is sent for a question related to more than one topic, with max\_tokens = 4 and the answer restricted to the number of a listed topic:

The following question is related to   
several topics. Which ONE of these   
topics is the question mainly about?   
Question: {question}   
Topics:   
1. {topic}   
2. {topic}  
Answer with the number of the topic only.

The related topics are listed in a random order, since a language model uses the information in its input unevenly by position (Liu et al., 2024).

A question related to one topic belongs to that topic without the second prompt, and a question related to no topic belongs to none. Neither prompt shows the candidates, so the topic of a question does not depend on the genders of its candidates.

## C Dataset and Topic Construction

Continuing from Section 5.1.1, this appendix gives the statistics on the two datasets (Table 4), the criteria of topic selection and of topic membership, and the recall@5 of each topic.

RoMQA. Of the 12,775 RoMQA questions whose golden answers are people, 5,391 hold at least one golden answer of each gender. The golden answers of each question are divided among the training, validation and test splits, and the 1,018 questions we keep each hold at least three golden answers of each gender in every split, so that every question yields enough queries in each split. We use the 961 of the 1,018 questions that belong to one of the 9 topics, as the Topic membership paragraph below decides. Within a split, each candidate of the larger gender is matched with up to ten candidates of the other, and each match makes one query with its two candidates in random order. The corpus is one index shared by the three splits, holding the introduction of the Wikipedia article of each golden answer that has one.

TREC Fair Ranking. The golden answers of each TREC question are halved by gender between the validation and the test split. Within a split, each male candidate, the larger group in every topic, is matched with 20 female candidates, and each match makes two queries. The corpus is the whole collection of the 2022 track in one index shared by every topic and split, each article truncated to its first 100 words, as in Wu et al. (2025).

<table><tr><td></td><td>Train</td><td>Validation</td><td>Test</td></tr><tr><td>RoMQA</td><td></td><td></td><td></td></tr><tr><td>Questions</td><td>961</td><td>961</td><td>961</td></tr><tr><td>Male candidates</td><td>17,392</td><td>17,329</td><td>17,254</td></tr><tr><td>Female candidates</td><td>11,858</td><td>11,646</td><td>11,429</td></tr><tr><td>Queries</td><td>148,586</td><td>146,230</td><td>143,738</td></tr><tr><td>TREC Fair Ranking</td><td></td><td></td><td></td></tr><tr><td>Questions</td><td></td><td>9</td><td>9</td></tr><tr><td>Male candidates</td><td></td><td>1,798</td><td>1,804</td></tr><tr><td>Female candidates</td><td></td><td>452</td><td>455</td></tr><tr><td>Queries</td><td></td><td>71,920</td><td>72,160</td></tr></table>

Table 4: Statistics of the two datasets over the 9 topics. The two candidates of every query are one male and one female golden answer of its question. A person who is a golden answer of two questions counts as two candidates.

<table><tr><td>Topic</td><td>Female</td><td>Male</td></tr><tr><td>Actors and Filmmakers</td><td>128</td><td>373</td></tr><tr><td>Computer science</td><td>78</td><td>423</td></tr><tr><td>Former countries</td><td>80</td><td>421</td></tr><tr><td>Human rights</td><td>154</td><td>347</td></tr><tr><td>Literature</td><td>117</td><td>384</td></tr><tr><td>Medicine</td><td>69</td><td>432</td></tr><tr><td>Netherlands</td><td>74</td><td>427</td></tr><tr><td>Sociology</td><td>114</td><td>387</td></tr><tr><td>Southeast Asia</td><td>93</td><td>408</td></tr></table>

Table 5: Candidates of each gender in each of the 9 topics of TREC Fair Ranking, summed over both splits.

Topic selection. Topic selection starts from the 50 WikiProjects in the release of Wu et al. (2025), and three filters reduce the set to the 9 topics we use. The first keeps a WikiProject with at least 50 candidates of each gender, which leaves 24. Each RoMQA question is then assigned one of the 24 by the rule of the Topic membership paragraph below, and the second filter keeps a WikiProject to which at least 10 questions are assigned, which leaves 16. The third keeps a WikiProject whose direction of correction is reproducible: $G D _ { t }$ under the unadapted system has the same sign on the validation and the test split of each dataset, which leaves 9. The two splits of each dataset hold disjoint candidates of the same questions, so a change of sign between them shows that $G D _ { t }$ is too noisy to determine the direction of correction. The third filter uses only the replies of the unadapted system, and every system is evaluated on the same 9 topics.

<table><tr><td>Topic</td><td>Questions</td><td>Female</td><td>Male</td></tr><tr><td>Actors and Filmmakers</td><td>216</td><td>8,847</td><td>11,643</td></tr><tr><td>Computer science</td><td>50</td><td>1,879</td><td>2,829</td></tr><tr><td>Former countries</td><td>104</td><td>3,001</td><td>5,468</td></tr><tr><td>Human rights</td><td>15</td><td>502</td><td>778</td></tr><tr><td>Literature</td><td>209</td><td>7,604</td><td>11,533</td></tr><tr><td>Medicine</td><td>26</td><td>917</td><td>1,467</td></tr><tr><td>Netherlands</td><td>46</td><td>1,768</td><td>2,552</td></tr><tr><td>Sociology</td><td>274</td><td>9,536</td><td>14,731</td></tr><tr><td>Southeast Asia</td><td>21</td><td>879</td><td>974</td></tr></table>

Table 6: Candidates of each gender in each of the 9 topics of RoMQA, summed over the three splits, and the questions of each topic. A person who is a golden answer of two questions counts as two candidates.

<table><tr><td></td><td colspan="3">Recall@5</td></tr><tr><td>Topic</td><td>Male</td><td>Male – Female Female</td><td> $G D _ { t }$ </td></tr><tr><td>RoMQA</td><td></td><td></td><td></td></tr><tr><td>Actors and Filmmakers</td><td>91.2</td><td>92.9</td><td>-1.7 +0.0211</td></tr><tr><td>Computer science</td><td>93.1</td><td>95.3 -2.2</td><td>-0.0072</td></tr><tr><td>Former countries</td><td>94.1</td><td>94.8</td><td>-0.7 -0.0121</td></tr><tr><td>Human rights</td><td>95.7</td><td>97.9</td><td>-2.2 -0.0356</td></tr><tr><td>Literature</td><td>93.3</td><td>93.7</td><td>-0.4 +0.0024</td></tr><tr><td>Medicine</td><td>85.0</td><td>87.8</td><td>-2.8 +0.0046</td></tr><tr><td>Netherlands</td><td>94.2</td><td>95.2</td><td>-1.0 +0.0194</td></tr><tr><td>Sociology</td><td>93.6</td><td>96.1</td><td>-2.5 -0.0092</td></tr><tr><td>Southeast Asia</td><td>90.6</td><td>88.7</td><td>+1.9 -0.0409</td></tr><tr><td>TREC Fair Ranking</td><td></td><td></td><td></td></tr><tr><td>Actors and Filmmakers</td><td>93.8</td><td>92.6</td><td>+1.2 +0.0385</td></tr><tr><td>Computer science</td><td>99.7</td><td>99.9 -0.2</td><td>+0.0002</td></tr><tr><td>Former countries</td><td>81.1</td><td>92.3</td><td>-11.2 +0.0083</td></tr><tr><td>Human rights</td><td>90.9</td><td>90.3 +0.6</td><td>-0.0073</td></tr><tr><td>Literature</td><td>96.8</td><td>98.0</td><td>-1.2 -0.0031</td></tr><tr><td>Medicine</td><td>96.9</td><td>94.3</td><td>+2.6 +0.0343</td></tr><tr><td>Netherlands</td><td>89.6</td><td>92.4</td><td>-2.8 -0.0049</td></tr><tr><td>Sociology</td><td>98.7</td><td>91.9</td><td>+6.8 +0.0396</td></tr><tr><td>Southeast Asia</td><td>86.8</td><td>89.7</td><td>-2.9 +0.0043</td></tr></table>

Table 7: Recall@5 and $G D _ { t }$ of the unadapted system on each topic, the values that Figure 6 draws.

Every question is then assigned again among the 9. Table 5 gives the candidates of each gender in the 9, and Table 6 the RoMQA questions and candidates of each.

Topic membership. A RoMQA question is assigned a topic in two calls to a language model (Appendix B.3): the first asks, for each topic of the set, whether the question is related to it, and the second asks which one of the related topics the question is mainly about.

Recall@5. Figure 6 adds TREC Fair Ranking to Figure 3, and Table 7 gives the recall@5 and $G D _ { t }$ of the unadapted system for each topic.

![](images/fc520ff43689ec32dfa87336821ff2cc37ec8183ec15dd05246b1d9ec5ae755f.jpg)  
(a) RoMQA

![](images/1c3cfbe570dde2ecbc18da5b8b86a13a600fcdbbc985c8d267ffe24f7d01bd8d.jpg)  
(b) TREC Fair Ranking  
Figure 6: Difference in recall@5 and $G D _ { t }$ of the unadapted system on both datasets. Full version of Figure 3. Each panel has its own scale for recall@5, and Table 7 lists the values.

## D Hyperparameters and Computational Resources

Continuing from Section 5.1.3, this appendix gives the settings of retrieval, generation, adaptation and pair construction, the template of the rewritten query, and the computational resources.

## D.1 Models and Training

Models. Retriever e5-base-v2 (110M); reranker bge-reranker-v2-m3 (568M cross-encoder); generator Meta-Llama-3-8B-Instruct (8.0B). Appendix E.2 also adapts a 278M reranker, jina-reranker-v2-base-multilingual (Jina AI, 2024), and Appendix E.3 replaces the generator with Qwen2.5-7B-Instruct (7.6B; Yang et al., 2024) and with Meta-Llama-3-70B-Instruct (70.6B, served in FP8; AI@Meta, 2024).

Retrieval and generation. The retriever encodes the rewritten query with the prefix query: and each document of the corpus with passage:, both in at most 512 tokens, and returns the pool of documents of highest inner product. The generator decodes at temperature 0 with at most 32 new tokens. A reply is lower-cased and stripped of punctuation and articles, then matched first against the options Both, Neither and Insufficient information to decide, and otherwise against the names of the two candidates; a reply that names both candidates is Both. A reply that matches no option and no candidate is almost always the name of a third person; such replies are at most 3.2% of the unadapted system’s replies on RoMQA and 1.5% on TREC.

Adaptation. Each topic has its own LoRA adapter, with r = 16 and α = 32 on the query and value projections of the cross-encoder’s attention blocks, no bias terms and no dropout, and its own copy of the classification head, trained in full: 2.62M parameters per topic, 0.46% of the reranker. The reranker’s own dropout of 0.1 is active during adaptation and off for the scores before adaptation. One epoch of AdamW with betas (0.9, 0.999), $\epsilon = 1 0 ^ { - 8 }$ and weight decay 0.01, a constant learning rate without warmup, gradient-norm clipping at 1.0, and batches of 32 pairs.

Objective and pair construction. The logit difference inside the Bradley–Terry objective is divided by a fixed temperature of 1.5, with no margin. Pair construction (Section 4.2) samples the n replies at temperature 0.7 with top-p = 0.9 and at most 16 new tokens.

## D.2 Query Rewriting

The retriever and the reranker receive a rewrite of the query in the following template:

Compare {c<sub>1</sub>} and {c<sub>2</sub>} in relation to   
{WikiProject or RoMQA question}.

## D.3 Baselines

Continuing from Section 5.1.4, this subsection states how we implement each baseline.

RRPO rewards a reply of Both with 1 and every other reply with 0, since Both is the correct reply to every query. We adapt the reranker through the same LoRA adapter as our method.

GB associates a document with the gender of its subject, and a document without a gender, which Geyik et al. (2019) do not consider, takes a slot of the top-5 only while each gender can still reach its minimum count.

RD fine-tunes the retriever toward the gender that the unadapted system disfavors on most topics, measured on the validation split of each protocol, and labels a document by the gender of its subject.

SD prefixes a sentence describing gender discrimination to the system message of Appendix B.2 and decodes greedily as every other system does, with the Transformers library (Wolf et al., 2020), since vLLM (Kwon et al., 2023) cannot advance two inputs together.

## D.4 Computational Resources

All experiments except the 70B generator run on four NVIDIA RTX 4090 GPUs (24 GB each) of one server with two Intel Xeon Silver 4410T CPUs; the 70B generator is served on two NVIDIA RTX PRO 6000 Blackwell GPUs (96 GB each). Generation is served by vLLM (Kwon et al., 2023), and the retrieval pipeline is built on FlashRAG (Jin et al., 2025). Each GPU hosts one vLLM server and one training run: the server is put to sleep while the reranker trains and scores, and woken for generation. LoRA is implemented with PEFT (Mangrulkar et al., 2022), and the retriever’s indexes are built with FAISS (Douze et al., 2024). One adaptation of the reranker for one topic, scored on the validation split, takes 0.44 GPU-hours on average under the in-dataset protocol and 0.37 GPU-hours under the cross-dataset protocol.

## E Full Experimental Results

This appendix gives the per-topic values of every system, repeats the in-dataset result with another reranker and two other generators, and reports the modified systems and the ablation variants under both protocols.

## E.1 Main Results per Topic

Continuing from Section 5.2, this appendix holds the per-topic values that Table 1 summarizes. Table 8 gives GD<sub>t</sub> on every topic for each system of Table 1, and Figure 7 draws the same values on both datasets, as Figure 4 draws them on RoMQA. Table 9 and Table 10 give the share of each reply of the unadapted system and of Ours on every topic of RoMQA and of TREC, and the preference pairs built per training query under each protocol, which Section 5.2.1 reads for Medicine.

## E.2 Another Reranker Backbone

Continuing from Section 5.2.1, this appendix adapts jina-reranker-v2-base-multilingual, which differs from bge-reranker-v2-m3 in size and training data, under the in-dataset protocol. Table 11 gives $G D _ { t }$ , |GD| and accuracy with each reranker, unadapted and adapted.

The reduction is smaller with the other reranker: |GD| falls from 0.0160 to 0.0108, a reduction of 33% against the 51% of bge-reranker-v2-m3, and accuracy is retained. On Actors and Filmmakers, every step that moves $G D _ { t }$ further on the validation split loses more than 0.5 points of accuracy, so $G D _ { t }$ moves only from +0.0317 to +0.0308.

## E.3 Other Generators

Continuing from Section 5.2.1, this appendix replaces the generator under the in-dataset protocol: Qwen2.5-7B-Instruct changes the model family at about the size of the main generator, and Meta-Llama-3-70B-Instruct changes the size within the same release. Table 12 gives $G D _ { t } ,$ |GD| and accuracy under each generator.

The reduction is smaller under both generators, and accuracy is retained. Under Qwen2.5-7B-Instruct, the unadapted system starts closer to zero, at |GD| of 0.0092 against 0.0169, so less disparity is left to remove: adaptation lowers |GD| to 0.0063, a reduction of 31% against the 51% under the main generator. Under Meta-Llama-3-70B-Instruct, the unadapted system starts farther from zero, at |GD| of 0.0302, and adaptation lowers $| G D |$ to 0.0202, removing more disparity than under the main generator, 0.0100 against 0.0086, but a smaller share of it, 33%. On Computer science, the topic farthest from zero, every step that moves $G D _ { t }$ further on the validation split loses more than 0.5 points of accuracy, so $G D _ { t }$ moves only from −0.0513 to −0.0492.

## E.4 Modifications of the Baselines and of Ours

Continuing from Section 5.3, this appendix describes how RD and SD are modified to treat each topic separately and how Ours, one reranker is built. Each modified baseline keeps the implementation of Appendix D.3 and chooses its setting for each topic as our method does.

RD per topic fine-tunes one retriever for each topic, in the direction of correction of that topic.

SD per topic chooses, for each topic, the decay constant $\lambda _ { \mathrm { S D } }$ of Schick et al. (2021) from {1, 2, 3}.

Ours, one reranker adapts a single reranker on the pairs of all 9 topics, each pair keeping the direction of its topic. One learning rate and one anchor weight, adjusted on validation |GD| as the setting of a baseline is, serve all topics.

For every system of Table 2, Table 13 gives the fairness and utility under both protocols, Figure 8 draws $G D _ { t }$ on both datasets, and Table 14 gives $G D _ { t }$ , |GD| and accuracy.

## E.5 Ablation Results

Continuing from Section 5.4, Table 15 gives the fairness and utility of every variant under both protocols. Table 16 gives $G D _ { t }$ , |GD| and accuracy.

On TREC, the 3 female-favoring topics start within 0.0073 of zero, so one direction stays close to Ours (Table 15). Without anchor also stays closer to Ours on TREC than on RoMQA, since Ours already sets $\lambda _ { t } = 0$ for 4 of the 9 topics.

![](images/5972243a0ad98d5dc46db6ecb68b9aa5e00f8ebeb052103341a27b1333d00411.jpg)  
(a) In-dataset (RoMQA)

![](images/646f92265ad9a8f19a4ca4fa446075bb7dee769fafb6c4cdc809017222c4bf97.jpg)  
(b) Cross-dataset (TREC)

Figure 7: $G D _ { t }$ of Table 1 under both protocols. Full version of Figure 4. Ours moves 8 of 9 topics in their direction of correction under each protocol. Each panel has its own scale, centered on zero.
<table><tr><td>Topic</td><td>Unadapted</td><td>RRPO</td><td>GB</td><td>RD</td><td>SD</td><td>Ours</td></tr><tr><td>In-dataset (RoMQA)</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Actors and Filmmakers</td><td>+0.0211</td><td>+0.0213</td><td>+0.0270</td><td>+0.0322</td><td>+0.0219</td><td>+0.0165</td></tr><tr><td>Computer science</td><td>-0.0072</td><td>-0.0073</td><td>-0.0108</td><td>+0.0077</td><td>-0.0061</td><td>+0.0043</td></tr><tr><td>Former countries</td><td>-0.0121</td><td>-0.0119</td><td>-0.0074</td><td>+0.0047</td><td>-0.0106</td><td>+0.0041</td></tr><tr><td>Human rights</td><td>-0.0356</td><td>-0.0413</td><td>-0.0395</td><td>-0.0417</td><td>-0.0395</td><td>+0.0104</td></tr><tr><td>Literature</td><td>+0.0024</td><td>+0.0046</td><td>+0.0038</td><td>+0.0122</td><td>+0.0033</td><td>+0.0010</td></tr><tr><td>Medicine</td><td>+0.0046</td><td>+0.0039</td><td>+0.0087</td><td>+0.0133</td><td>+0.0053</td><td>+0.0058</td></tr><tr><td>Netherlands</td><td>+0.0194</td><td>+0.0203</td><td>+0.0085</td><td>+0.0172</td><td>+0.0250</td><td>-0.0051</td></tr><tr><td>Sociology</td><td>-0.0092</td><td>-0.0106</td><td>-0.0099</td><td>-0.0049</td><td>-0.0090</td><td>-0.0019</td></tr><tr><td>Southeast Asia</td><td>-0.0409</td><td>-0.0315</td><td>-0.0576</td><td>-0.0121</td><td>-0.0358</td><td>-0.0255</td></tr><tr><td>|GD|</td><td>0.0169</td><td>0.0170</td><td>0.0193</td><td>0.0162</td><td>0.0174</td><td>0.0083</td></tr><tr><td>Accuracy</td><td>55.1</td><td>56.1</td><td>55.4</td><td>54.3</td><td>54.9</td><td>55.2</td></tr><tr><td>Cross-dataset (TREC)</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Actors and Filmmakers</td><td>+0.0385</td><td>+0.0414</td><td>+0.0213</td><td>+0.0138</td><td>+0.0389</td><td>+0.0074</td></tr><tr><td>Computer science</td><td>+0.0002</td><td>-0.0002</td><td>+0.0006</td><td>-0.0001</td><td>+0.0008</td><td>-0.0005</td></tr><tr><td>Former countries</td><td>+0.0083</td><td>+0.0089</td><td>-0.0121</td><td>-0.0192</td><td>+0.0122</td><td>-0.0076</td></tr><tr><td>Human rights</td><td>-0.0073</td><td>-0.0079</td><td>-0.0039</td><td>-0.0112</td><td>-0.0072</td><td>-0.0075</td></tr><tr><td>Literature</td><td>-0.0031</td><td>-0.0017</td><td>-0.0034</td><td>-0.0056</td><td>-0.0038</td><td>-0.0029</td></tr><tr><td>Medicine</td><td>+0.0343</td><td>+0.0352</td><td>+0.0236</td><td>+0.0294</td><td>+0.0341</td><td>+0.0226</td></tr><tr><td>Netherlands</td><td>-0.0049</td><td>-0.0034</td><td>-0.0018</td><td>-0.0085</td><td>-0.0053</td><td>+0.0018</td></tr><tr><td>Sociology</td><td>+0.0396</td><td>+0.0371</td><td>+0.0274</td><td>+0.0375</td><td>+0.0430</td><td>+0.0326</td></tr><tr><td>Southeast Asia</td><td>+0.0043</td><td>+0.0042</td><td>+0.0072</td><td>+0.0038</td><td>+0.0039</td><td>+0.0023</td></tr><tr><td>|GD|</td><td>0.0156</td><td>0.0156</td><td>0.0112</td><td>0.0143</td><td>0.0166</td><td></td></tr><tr><td>Accuracy</td><td>89.9</td><td>89.9</td><td>92.1</td><td>89.9</td><td>89.4</td><td>0.0094 90.2</td></tr></table>

Table 8: $G D _ { t }$ of every system of Table 1 under both protocols. The rows |GD| and Accuracy give, for each column, the mean absolute per-topic group disparity and the accuracy that Table 1 reports. Bold marks, in each row, the value closest to zero.

<table><tr><td></td><td colspan="5">Unadapted</td><td colspan="5">Ours</td><td></td></tr><tr><td>Topic</td><td>Both</td><td>Male</td><td>Female</td><td>Neither</td><td>Other</td><td>Both</td><td>Male</td><td>Female</td><td>Neither</td><td>Other</td><td>Pairs per query</td></tr><tr><td>Actors and Filmmakers</td><td>51.4</td><td>12.5</td><td>10.4</td><td>21.3</td><td>4.4</td><td>51.9</td><td>12.5</td><td>10.9</td><td>20.7</td><td>4.0</td><td>0.848</td></tr><tr><td>Computer science</td><td>62.9</td><td>8.4</td><td>9.2</td><td>17.5</td><td>2.0</td><td>62.6</td><td>9.0</td><td>8.6</td><td>17.6</td><td>2.1</td><td>1.097</td></tr><tr><td>Former countries</td><td>43.3</td><td>10.1</td><td>11.3</td><td>30.7</td><td>4.7</td><td>43.7</td><td>11.0</td><td>10.6</td><td>29.9</td><td>4.9</td><td>0.825</td></tr><tr><td>Human rights</td><td>62.5</td><td>9.5</td><td>13.1</td><td>10.2</td><td>4.7</td><td>64.2</td><td>10.9</td><td>9.9</td><td>9.6</td><td>5.4</td><td>1.065</td></tr><tr><td>Literature</td><td>54.8</td><td>8.5</td><td>8.3</td><td>25.5</td><td>2.9</td><td>54.7</td><td>8.4</td><td>8.3</td><td>25.6</td><td>3.0</td><td>1.055</td></tr><tr><td>Medicine</td><td>17.7</td><td>9.3</td><td>8.8</td><td>58.6</td><td>5.6</td><td>17.6</td><td>9.8</td><td>9.3</td><td>57.6</td><td>5.7</td><td>0.381</td></tr><tr><td>Netherlands</td><td>57.8</td><td>10.0</td><td>8.1</td><td>22.5</td><td>1.6</td><td>57.4</td><td>8.9</td><td>9.4</td><td>22.9</td><td>1.4</td><td>1.331</td></tr><tr><td>Sociology</td><td>64.6</td><td>6.8</td><td>7.8</td><td>19.2</td><td>1.6</td><td>64.5</td><td>7.0</td><td>7.2</td><td>19.3</td><td>1.9</td><td>1.072</td></tr><tr><td>Southeast Asia</td><td>44.4</td><td>13.3</td><td>17.4</td><td>22.4</td><td>2.5</td><td>43.5</td><td>14.7</td><td>17.3</td><td>21.6</td><td>2.9</td><td>1.319</td></tr><tr><td>All topics</td><td>55.1</td><td>9.3</td><td>9.2</td><td>23.3</td><td>3.0</td><td>55.2</td><td>9.5</td><td>9.1</td><td>23.1</td><td>3.1</td><td></td></tr></table>

Table 9: Replies of the unadapted system and of Ours under the in-dataset protocol. For each topic: the share (%) of each reply, and the preference pairs built per training query. Both is the accuracy; Male and Female, a reply of that candidate; Neither includes Insufficient information to decide; Other, a reply that matches no option and no candidate.

<table><tr><td></td><td colspan="5">Unadapted</td><td colspan="5">Ours</td><td></td></tr><tr><td>Topic</td><td>Both</td><td>Male</td><td>Female</td><td>Neither</td><td>Other</td><td>Both</td><td>Male</td><td>Female</td><td>Neither</td><td>Other</td><td>Pairs per query</td></tr><tr><td>Actors and Filmmakers</td><td>92.1</td><td>5.8</td><td>2.0</td><td>0.0</td><td>0.1</td><td>92.5</td><td>4.0</td><td>3.3</td><td>0.0</td><td>0.2</td><td>0.848</td></tr><tr><td>Computer science</td><td>99.4</td><td>0.2</td><td>0.2</td><td>0.0</td><td>0.2</td><td>99.4</td><td>0.1</td><td>0.2</td><td>0.2</td><td>0.1</td><td>1.071</td></tr><tr><td>Former countries</td><td>60.7</td><td>10.3</td><td>9.5</td><td>18.2</td><td>1.3</td><td>59.8</td><td>8.4</td><td>9.1</td><td>20.5</td><td>2.2</td><td>0.835</td></tr><tr><td>Human rights</td><td>90.3</td><td>2.6</td><td>3.3</td><td>1.1</td><td>2.7</td><td>90.3</td><td>2.3</td><td>3.0</td><td>0.9</td><td>3.5</td><td>1.065</td></tr><tr><td>Literature</td><td>98.9</td><td>0.4</td><td>0.7</td><td>0.0</td><td>0.0</td><td>98.9</td><td>0.4</td><td>0.7</td><td>0.0</td><td>0.0</td><td>1.091</td></tr><tr><td>Medicine</td><td>93.5</td><td>4.8</td><td>1.4</td><td>0.1</td><td>0.2</td><td>94.6</td><td>3.6</td><td>1.4</td><td>0.1</td><td>0.2</td><td>0.381</td></tr><tr><td>Netherlands</td><td>92.2</td><td>2.7</td><td>3.2</td><td>0.2</td><td>1.7</td><td>91.8</td><td>3.1</td><td>2.9</td><td>0.3</td><td>2.0</td><td>1.356</td></tr><tr><td>Sociology</td><td>92.1</td><td>5.5</td><td>1.5</td><td>0.2</td><td>0.7</td><td>93.9</td><td>4.3</td><td>1.0</td><td>0.2</td><td>0.7</td><td>1.064</td></tr><tr><td>Southeast Asia</td><td>91.2</td><td>1.8</td><td>1.3</td><td>0.7</td><td>5.0</td><td>92.3</td><td>1.8</td><td>1.5</td><td>0.5</td><td>4.0</td><td>1.321</td></tr><tr><td>All topics</td><td>89.9</td><td>3.8</td><td>2.6</td><td>2.4</td><td>1.3</td><td>90.2</td><td>3.1</td><td>2.6</td><td>2.6</td><td>1.4</td><td></td></tr></table>

Table 10: Replies of the unadapted system and of Ours under the cross-dataset protocol. For each topic: the share (%) of each reply, and the preference pairs built per training query. The columns are defined as in Table 9.

<table><tr><td rowspan="2">Topic</td><td colspan="2">bge-reranker-v2-m3</td><td colspan="2">jina-reranker-v2-base-multilingual</td></tr><tr><td>Unadapted</td><td>Adapted</td><td>Unadapted</td><td>Adapted</td></tr><tr><td>Actors and Filmmakers</td><td>+0.0211</td><td>+0.0165</td><td>+0.0317</td><td>+0.0308</td></tr><tr><td>Computer science</td><td>-0.0072</td><td>+0.0043</td><td>-0.0096</td><td>-0.0095</td></tr><tr><td>Former countries</td><td>-0.0121</td><td>+0.0041</td><td>+0.0030</td><td>-0.0033</td></tr><tr><td>Human rights</td><td>-0.0356</td><td>+0.0104</td><td>-0.0287</td><td>-0.0174</td></tr><tr><td>Literature</td><td>+0.0024</td><td>+0.0010</td><td>+0.0057</td><td>+0.0041</td></tr><tr><td>Medicine</td><td>+0.0046</td><td>+0.0058</td><td>+0.0145</td><td>-0.0031</td></tr><tr><td>Netherlands</td><td>+0.0194</td><td>-0.0051</td><td>+0.0070</td><td>-0.0021</td></tr><tr><td>Sociology</td><td>-0.0092</td><td>-0.0019</td><td>-0.0087</td><td>-0.0007</td></tr><tr><td>Southeast Asia</td><td>-0.0409</td><td>-0.0255</td><td>-0.0348</td><td>-0.0258</td></tr><tr><td>|GD|</td><td>0.0169</td><td>0.0083</td><td>0.0160</td><td>0.0108</td></tr><tr><td>Accuracy</td><td>55.1</td><td>55.2</td><td>52.9</td><td>52.8</td></tr></table>

Table 11: $G D _ { t }$ under the in-dataset protocol with each reranker, unadapted and adapted. Adapted bge-reranker-v2-m3 is Ours.

<table><tr><td rowspan="2">Topic</td><td colspan="2">Meta-Llama-3-8B-Instruct</td><td colspan="2">Qwen2.5-7B-Instruct</td><td colspan="2">Meta-Llama-3-70B-Instruct</td></tr><tr><td>Unadapted</td><td>Adapted</td><td>Unadapted</td><td>Adapted</td><td>Unadapted</td><td>Adapted</td></tr><tr><td>Actors and Filmmakers</td><td>+0.0211</td><td>+0.0165</td><td>+0.0071</td><td>+0.0066</td><td>-0.0176</td><td>-0.0128</td></tr><tr><td>Computer science</td><td>-0.0072</td><td>+0.0043</td><td>-0.0066</td><td>-0.0065</td><td>-0.0513</td><td>-0.0492</td></tr><tr><td>Former countries</td><td>-0.0121</td><td>+0.0041</td><td>-0.0157</td><td>-0.0075</td><td>-0.0289</td><td>+0.0119</td></tr><tr><td>Human rights</td><td>-0.0356</td><td>+0.0104</td><td>-0.0217</td><td>-0.0135</td><td>-0.0482</td><td>-0.0122</td></tr><tr><td>Literature</td><td>+0.0024</td><td>+0.0010</td><td>-0.0017</td><td>-0.0013</td><td>-0.0234</td><td>-0.0129</td></tr><tr><td>Medicine</td><td>+0.0046</td><td>+0.0058</td><td>+0.0022</td><td>+0.0017</td><td>-0.0224</td><td>-0.0176</td></tr><tr><td>Netherlands</td><td>+0.0194</td><td>-0.0051</td><td>-0.0133</td><td>-0.0132</td><td>-0.0274</td><td>-0.0243</td></tr><tr><td>Sociology</td><td>-0.0092</td><td>-0.0019</td><td>-0.0071</td><td>+0.0002</td><td>-0.0435</td><td>-0.0332</td></tr><tr><td>Southeast Asia</td><td>-0.0409</td><td>-0.0255</td><td>-0.0070</td><td>+0.0060</td><td>-0.0087</td><td>-0.0077</td></tr><tr><td>|GD|</td><td>0.0169</td><td>0.0083</td><td>0.0092</td><td>0.0063</td><td>0.0302</td><td>0.0202</td></tr><tr><td>Accuracy</td><td>55.1</td><td>55.2</td><td>48.1</td><td>47.7</td><td>57.3</td><td>57.4</td></tr></table>

Table 12: $G D _ { t }$ under the in-dataset protocol with each generator, unadapted and adapted. Adapted under Meta-Llama-3-8B-Instruct is Ours.

<table><tr><td></td><td colspan="4">In-dataset (RoMQA)</td><td colspan="4">Cross-dataset (TREC)</td></tr><tr><td>System</td><td>|GD|</td><td>Accuracy</td><td>Male-fav.</td><td>Female-fav.</td><td>|GD|</td><td>Accuracy</td><td>Male-fav.</td><td>Female-fav.</td></tr><tr><td>Unadapted</td><td>0.0169</td><td>55.1</td><td>一</td><td>一</td><td>0.0156</td><td>89.9</td><td>一</td><td>一</td></tr><tr><td>Ours, one reranker</td><td>0.0175</td><td>55.5</td><td>1/4</td><td>5/5</td><td>0.0135</td><td>90.1</td><td>4/6</td><td>0/3</td></tr><tr><td>RD per topic</td><td>0.0118</td><td>54.8</td><td>4/4</td><td>4/5</td><td>0.0104</td><td>90.2</td><td>6/6</td><td>2/3</td></tr><tr><td>SD per topic</td><td>0.0170</td><td>55.0</td><td>0/4</td><td>3/5</td><td>0.0165</td><td>89.4</td><td>2/6</td><td>1/3</td></tr><tr><td>Ours</td><td>0.0083</td><td>55.2</td><td>3/4</td><td>5/5</td><td>0.0094</td><td>90.2</td><td>6/6</td><td>2/3</td></tr></table>

Table 13: Ours with one reranker shared by all topics, and the systems that treat each topic separately, under both protocols. Full version of Table 2. Ours, one reranker moves most topics toward the male candidate under the in-dataset protocol and toward the female candidate under the cross-dataset protocol.

![](images/13c58688ec6ed58417dde76b1942d82e98c8714461379bfbc081704681e235de.jpg)  
(a) In-dataset (RoMQA)

![](images/40cf9949e6cd32f5de8f8accff79ea4732dceeea72b1cfbf496da91f56d90a75.jpg)  
(b) Cross-dataset (TREC)

Figure 8: GD of Table 2 under both protocols. Full version of Figure 5. Ours is the closest to zero overall.
<table><tr><td>Topic</td><td>Unadapted</td><td>Ours, one reranker</td><td>RD per topic</td><td>SD per topic</td><td>Ours</td></tr><tr><td>In-dataset (RoMQA)</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Actors and Filmmakers</td><td>+0.0211</td><td>+0.0247</td><td>+0.0192</td><td>+0.0219</td><td>+0.0165</td></tr><tr><td>Computer science</td><td>-0.0072</td><td>-0.0043</td><td>+0.0013</td><td>-0.0061</td><td>+0.0043</td></tr><tr><td>Former countries</td><td>-0.0121</td><td>-0.0088</td><td>-0.0037</td><td>-0.0098</td><td>+0.0041</td></tr><tr><td>Human rights</td><td>-0.0356</td><td>-0.0339</td><td>-0.0378</td><td>-0.0391</td><td>+0.0104</td></tr><tr><td>Literature</td><td>+0.0024</td><td>+0.0098</td><td>-0.0002</td><td>+0.0036</td><td>+0.0010</td></tr><tr><td>Medicine</td><td>+0.0046</td><td>+0.0089</td><td>-0.0002</td><td>+0.0060</td><td>+0.0058</td></tr><tr><td>Netherlands</td><td>+0.0194</td><td>+0.0182</td><td>+0.0068</td><td>+0.0250</td><td>-0.0051</td></tr><tr><td>Sociology</td><td>-0.0092</td><td>-0.0091</td><td>-0.0042</td><td>-0.0096</td><td>-0.0019</td></tr><tr><td>Southeast Asia</td><td>-0.0409</td><td>-0.0395</td><td>-0.0325</td><td>-0.0322</td><td>-0.0255</td></tr><tr><td>|GD|</td><td>0.0169</td><td>0.0175</td><td>0.0118</td><td>0.0170</td><td>0.0083</td></tr><tr><td>Accuracy</td><td>55.1</td><td>55.5</td><td>54.8</td><td>55.0</td><td>55.2</td></tr><tr><td>Cross-dataset (TREC)</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Actors and Filmmakers</td><td>+0.0385</td><td>+0.0156</td><td>+0.0139</td><td>+0.0389</td><td>+0.0074</td></tr><tr><td>Computer science</td><td>+0.0002</td><td>-0.0008</td><td>-0.0005</td><td>+0.0008</td><td>-0.0005</td></tr><tr><td>Former countries</td><td>+0.0083</td><td>-0.0103</td><td>-0.0012</td><td>+0.0122</td><td>-0.0076</td></tr><tr><td>Human rights</td><td>-0.0073</td><td>-0.0086</td><td>-0.0032</td><td>-0.0072</td><td>-0.0075</td></tr><tr><td>Literature</td><td>-0.0031</td><td>-0.0068</td><td>-0.0009</td><td>-0.0031</td><td>-0.0029</td></tr><tr><td>Medicine</td><td>+0.0343</td><td>+0.0242</td><td>+0.0330</td><td>+0.0341</td><td>+0.0226</td></tr><tr><td>Netherlands</td><td>-0.0049</td><td>-0.0071</td><td>-0.0055</td><td>-0.0053</td><td>+0.0018</td></tr><tr><td>Sociology</td><td>+0.0396</td><td>+0.0407</td><td>+0.0334</td><td>+0.0430</td><td>+0.0326</td></tr><tr><td>Southeast Asia</td><td>+0.0043</td><td>+0.0074</td><td>+0.0023</td><td>+0.0039</td><td>+0.0023</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>|GD| Accuracy</td><td>0.0156 89.9</td><td>0.0135 90.1</td><td>0.0104 90.2</td><td>0.0165 89.4</td><td>0.0094 90.2</td></tr></table>

Table 14: GD of every system of Table 2 under both protocols. The rows |GD| and Accuracy give, for each column, the mean absolute per-topic group disparity and the accuracy that Table 13 reports. Bold marks, in each row, the value closest to zero.

<table><tr><td></td><td colspan="4">In-dataset (RoMQA)</td><td colspan="4">Cross-dataset (TREC)</td></tr><tr><td>Variant</td><td>|GD|</td><td>Accuracy</td><td>Male-fav.</td><td>Female-fav.</td><td>|GD|</td><td>Accuracy</td><td>Male-fav.</td><td>Female-fav.</td></tr><tr><td>Unadapted</td><td>0.0169</td><td>55.1</td><td></td><td></td><td>0.0156</td><td>89.9</td><td></td><td></td></tr><tr><td>Ours</td><td>0.0083</td><td>55.2</td><td>3/4</td><td>5/5</td><td>0.0094</td><td>90.2</td><td>6/6</td><td>2/3</td></tr><tr><td>one direction</td><td>0.0176</td><td>54.8</td><td>0/4</td><td>5/5</td><td>0.0102</td><td>90.3</td><td>6/6</td><td>0/3</td></tr><tr><td>without anchor</td><td>0.0142</td><td>55.0</td><td>0/4</td><td>4/5</td><td>0.0112</td><td>90.8</td><td>5/6</td><td>1/3</td></tr></table>

Table 15: Ablation under both protocols. Full version of Table 3.

<table><tr><td colspan="3"></td><td colspan="2">Variants of Table 3</td></tr><tr><td>Topic</td><td>Unadapted</td><td>Ours</td><td>one direction</td><td>without anchor</td></tr><tr><td>In-dataset (RoMQA)</td><td></td><td></td><td></td><td></td></tr><tr><td>Actors and Filmmakers</td><td>+0.0211</td><td>+0.0165</td><td>+0.0495</td><td>+0.0223</td></tr><tr><td>Computer science</td><td>-0.0072</td><td>+0.0043</td><td>+0.0043</td><td>-0.0010</td></tr><tr><td>Former countries</td><td>-0.0121</td><td>+0.0041</td><td>+0.0041</td><td>+0.0163</td></tr><tr><td>Human rights</td><td>-0.0356</td><td>+0.0104</td><td>+0.0104</td><td>-0.0035</td></tr><tr><td>Literature</td><td>+0.0024</td><td>+0.0010</td><td>+0.0026</td><td>+0.0041</td></tr><tr><td>Medicine</td><td>+0.0046</td><td>+0.0058</td><td>+0.0084</td><td>+0.0080</td></tr><tr><td>Netherlands</td><td>+0.0194</td><td>-0.0051</td><td>+0.0518</td><td>+0.0203</td></tr><tr><td>Sociology</td><td>-0.0092</td><td>-0.0019</td><td>-0.0019</td><td>-0.0077</td></tr><tr><td>Southeast Asia</td><td>-0.0409</td><td>-0.0255</td><td>-0.0255</td><td>-0.0452</td></tr><tr><td>|GD|</td><td>0.0169</td><td>0.0083</td><td>0.0176</td><td>0.0142</td></tr><tr><td>Accuracy</td><td>55.1</td><td>55.2</td><td>54.8</td><td>55.0</td></tr><tr><td>Cross-dataset (TREC)</td><td></td><td></td><td></td><td></td></tr><tr><td>Actors and Filmmakers</td><td>+0.0385</td><td>+0.0074</td><td>+0.0074</td><td>+0.0174</td></tr><tr><td>Computer science</td><td>+0.0002</td><td>-0.0005</td><td>-0.0005</td><td>-0.0006</td></tr><tr><td>Former countries</td><td>+0.0083</td><td>-0.0076</td><td>-0.0076</td><td>-0.0076</td></tr><tr><td>Human rights</td><td>-0.0073</td><td>-0.0075</td><td>-0.0103</td><td>-0.0075</td></tr><tr><td>Literature</td><td>-0.0031</td><td>-0.0029</td><td>-0.0031</td><td>-0.0033</td></tr><tr><td>Medicine</td><td>+0.0343</td><td>+0.0226</td><td>+0.0226</td><td>+0.0226</td></tr><tr><td>Netherlands</td><td>-0.0049</td><td>+0.0018</td><td>-0.0050</td><td>+0.0053</td></tr><tr><td>Sociology</td><td>+0.0396</td><td>+0.0326</td><td>+0.0326</td><td>+0.0326</td></tr><tr><td>Southeast Asia</td><td>+0.0043</td><td>+0.0023</td><td>+0.0023</td><td>+0.0044</td></tr><tr><td>|GD|</td><td>0.0156</td><td>0.0094</td><td>0.0102</td><td>0.0112</td></tr><tr><td>Accuracy</td><td>89.9</td><td>90.2</td><td>90.3</td><td>90.8</td></tr></table>

Table 16: $G D _ { t }$ of every variant of Table 3, beside Unadapted and Ours, under both protocols. The rows |GD| and Accuracy give, for each column, the mean absolute per-topic group disparity and the accuracy that Table 15 reports. Bold marks, in each row, the value closest to zero.