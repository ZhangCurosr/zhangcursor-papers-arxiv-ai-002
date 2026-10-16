# PRIVATE RETRIEVAL AUGMENTED GENERATION VIA RANDOM PROJECTION

A PREPRINT

Dixi Yao \* Tahseen Rabbani Tian Li University of Chicago

## ABSTRACT

Retrieval-Augmented Generation (RAG) enhances the capabilities of large language models (LLMs) by querying external structured knowledge. However, it can also introduce privacy risks by leaking sensitive information from the retrieval datastore. We propose a simple method to preserve datastore privacy in RAG systems via random projection. By applying the same projection to both datastore embeddings and query embeddings, our method provably preserves semantic similarity between queries and retrieved items while substantially mitigating data extraction attacks. Across multiple RAG architectures and datasets, we show that this lightweight approach achieves superior retrieval and generation performance compared to prior methods which leverage formal differential privacy (DP) guarantees, while exhibiting comparable empirical privacy under strong attack models. Our results for the first time suggest that random projection can serve as a competitive and practical baseline for privacy-preserving RAG systems.

## 1 Introduction

Retrieval-Augmented Generation (RAG) [1, 2] enhances large language models (LLMs) by incorporating information retrieved from external knowledge. However, recent studies show that carefully crafted prompts [3, 4, 5, 6] can extract sensitive information such as personal identity information from the external datastore.

Since attack queries can closely resemble legitimate user queries, simple access control may block ordinary users [3, 4]. Recent work has begun to explore privacy risks in RAG, with a leading focus on differential privacy (DP) [6, 7]. Although DP methods provide formal statistical guarantees, they often degrade output quality (e.g., leading to high perplexity) due to DP’s fundamental properties of worst-case guarantees, and are usually complicated to implement in practice. To this end, we propose private RAG via random projection to defend against data extraction attacks.

Our key idea is that differential privacy guarantees may be an overkill to prevent state-of-the-art attacks practically, whereas (some specific) random projection matrices naturally preserve pairwise similarities of the inputs in the lowerdimensional space. The extra randomness alters datastore embeddings so attackers may not recover the original text. This enables users to obtain accurate answers from the RAG system while adversaries fail. We apply a single projection step to datastore embeddings in the offline stage and the same matrix to user queries in the online stage.

In this work, we consider two RAG architectures: ‘KNN-LM’ [1] and ‘Standard RAG’ [2]<sup>2</sup>. In KNN-LM, the system linearly combines the outputs of the language model and the k-nearest neighbour outputs (usually organized as a softmax over cosine similarities between queries and the datastore embeddings), and generates next token. In Standard RAG, the system concatenates retrieved texts with the query and feeds them to the LLM. In this work, we focus on the settings where formal privacy guarantees are not desired, and thus evaluate privacy leakage by exploring the success rates of strong, existing attacks empirically.

In Section 4, we empirically show that enforcing differential privacy guarantees may not provide better practical protection against data extraction and can instead harm utility. In summary, our RP technique significantly improves utility (over 50% on average across datasets) over the baseline methods without degrading empirical privacy performance.

## 2 Related Works

Private RAG. Prior work shows that various RAG architectures can reveal sensitive personal data (e.g., emails, phone numbers, and URLs) from the external datastore [3, 5, 8, 9, 10]. Existing defenses, such as DP-based sampling and aggregation [6, 11] and synthetic data [12], reduce efficiency on large datasets and often require extra validation or training. Moreover, Zhao and Zhang [13] show that synthetic data alone does not fully prevent memorization or leakage. He et al. [14] apply local differential privacy to sensitive content like addresses. However, privacy can involve entire sentences or paragraphs beyond tokens, making simple masking insufficient. Thareja et al. [15] mix original and sanitized documents, which requires identifying all private tokens and is impractical for large-scale RAG.

Privacy-Preserving Random Projection. Certain random projection operations are a common technique for reducing dimensionality while preserving some similarity measures with high probability (e.g., based on the Johnson-Lindenstrauss lemma [16]). Prior works have shown that methods built on top of these projections can achieve differential privacy (DP) for specific applications [17, 18, 19, 20, 21, 22, 23, 24, 25]. In this work, we show that random projection can be effective for a variety of RAG scenarios as well.

## 3 Private RAG via Random Projection

## 3.1 Threat Model

As illustrated in Figure 1, our algorithm is compatible with both RAG architectures. In this work, we evaluate our method as an empirical defense against data extraction attacks. We consider a black-box setting in which the attacker has access only to model API responses, while model parameters, the random projection matrix, logits, embeddings, and latent representations remain inaccessible [26, 27]. The attacker submits queries to the RAG system using the API service and attempts to extract sensitive information in the datastore from the generated response through one round or multiple rounds. This is a common protocol in modern RAG applications where companies provide consulting service through LLM with an internal datastore while only text answers are exposed to users. The threat model also aligns with the threat scenarios considered in prior work on RAG privacy [3, 4, 28].

![](images/e9933b1d3429201e9900166a8289e801f5f306394312f7a0d8918bf914fbde9b.jpg)  
Figure 1: Overview of Private RAG via RP.

In short, we assume the attacker knows that RP is used but does not have direct access to the specific random matrices, with the projection matrix remaining undisclosed. Attackers are free to conduct repeated-query (Section 4.2) or multi-round adaptive attacks (Section 4.3). As network- and system-level attacks are out of scope and access to embeddings is restricted, we do not consider white-box attacks, embedding inversion, projection-matrix inversion, or side-channel attacks requiring side information or access beyond the LLM’s text responses.

To provide a clearer understanding of what constitutes sensitive information, we show examples of leakage with and without sensitive content. In Table 1, we observe that leakage of sensitive information exposes critical details such as names, addresses, phone numbers, and email addresses. In contrast, non-sensitive information reveals no useful details. For example, a generic URL or repeated mentions of “Queen $\mathrm { S t } ^ { \prime \prime }$ are too common and lack identifying information.

## 3.2 Algorithm

Let $D = \{ d _ { 1 } , \ldots , d _ { n } \}$ be the corpus of n natural language documents $( \mathrm { e . g . }$ , email threads or case paragraphs). Let f(·) denote the tokenizer and embedding layers, which can be further fine-tuned to map these $d _ { i } \mathrm { ' s }$ (texts) to embeddings. We assume a fixed pre-trained encoder throughout the paper for simplicity. We also assume that an attacker interacts with the RAG system through API interactions and aims to extract data with constructed queries.

```latex
We sample an IID Gaussian matrix $R \in \mathbb { R } ^ { k \times d } .$ , where each Algorithm 1: Private RAG with Random Projection
(Standard RAG)
element follows $\mathcal { N } ( 0 , 1 / k )$ . We precompute the projected data
store embeddings offline by computing $\boldsymbol { e } _ { i } ^ { \prime } = \bar { \boldsymbol { R } } \bar { \boldsymbol { e } } _ { i }$ for each
document embedding $\boldsymbol { e } _ { i } \ : = \ : f ( d _ { i } ) \in \mathbb { R } ^ { d }$ We perform this Require: Datastore D with n documents; embedding func
tion f : Text → R<sup>d</sup>
preprocessing step once and store the results, introducing no
Input: User query q
runtime overhead during inference. For any query/prompt,
Output: Generated answer a
we encode it as $\pmb { x } = f ( \mathrm { p r o m p t } )$ and compute its projection Parameter: Projection dimension k
$\mathbf { x } ^ { \prime } = R \mathbf { x }$ , which requires only a single matrix-vector mul- Offline preprocessing
tiplication. We then perform retrieval in the k-dimensional 1: Sample $\mathbf { \bar { \boldsymbol { R } } } \in \mathbb { R } ^ { k \times d } , \dot { R _ { i j } } \sim \mathcal { N } ( 0 , 1 / k )$
space by finding the nearest neighbors to $\mathbf { x } ^ { \prime }$ among $\{ e _ { i } ^ { \prime } \} _ { i \in [ n ] }$ 2: Compute embeddings $E _ { D } \in \mathbb { R } ^ { d \times n } , e _ { i } = f ( d _ { i } )$
and computing a softmax over their cosine similarities with x<sup>′</sup>. 3: Project: $\tilde { E } _ { D } = R E _ { D }$
Finally, we combine this softmax with the parametric LLM’s Online inference (per query)
logits for x in a weighted manner to select the next token. We 4: Initialize $a \gets \emptyset$
repeat generation until reaching the maximum token limit or 5: Compute query embedding: $\mathbf { \boldsymbol { x } } = \mathbf { \boldsymbol { f } } ( \mathbf { \boldsymbol { q } } )$
generating an end-of-sequence token. 6: Project: $\tilde { \pmb { x } } = R \pmb { x }$
7: Calculate similarity between x˜ and $\tilde { E } _ { D }$
Algorithm 2 (in Section D) and Algorithm 1 illustrate the im- 8: Retrieve most relevant text: y
plementation of our random projection method. The overall 9: $q  q \vert \vert y$
process follows the standard RAG pipeline: we first tokenize 10: $a \gets \dot { L M } ( q )$
and embed the datastore offline, and store the representations 11: return a
in FAISS or another efficient indexing framework to enable fast
```

retrieval at inference time. During inference, the model performs autoregressive generation via iterative next-token prediction.

For standard RAG (Algorithm 1), we compute the similarity between the projected query embedding x˜ and the projected datastore embeddings $\tilde { E } _ { D } .$ , and retrieve the most relevant text based on this similarity. The retrieved text is concatenated with the original query to form an augmented prompt, which is then passed to a parametric LLM to generate the final output. The algorithm of KNN-LM is summarized in Section D.

Privacy Implications. Random projection preserves privacy by altering the retrieval process: given a query, vanilla RAG retrieves the highest-scoring document and generates a corresponding token, which may contain sensitive information. After RP, embeddings that are close to the original top match can be selected with non-negligible probability (Section C). As a result, alternative tokens may replace the originally retrieved token.

Utility Discussions. A natural question arises: if random projections alter the retrieved neighbors, how does the model remain accurate? The reason is that random projection approximately preserves embedding distances, ensuring retrieval still selects relevant information. This has been extensively studied in prior literature [16]. In Section B, for completeness, we provide proof that our random projection preserves $L _ { 2 }$ distance and cosine similarity. Section B.2 shows a study with synthetic data consistent with the theoretical analysis. For DP-based baselines, achieving privacy guarantees sacrifices utility. DP introduces a fundamental tradeoff: a small clipping bound overly distorts embeddings, while a large bound increases noise magnitude, both degrading retrieval utility. We illustrate this in Section 4.6.

Table 1: We focus on sensitive-information leakage and show examples of sensitive and non-sensitive information.
<table><tr><td>Leakage Information</td><td>Sensitive Information</td><td>Non-Sensitive Information</td></tr><tr><td>tion (e.g., Token ID)</td><td colspan="2">URL with Personal Informa- If you prefer to receive a hardcopy of your pay stub, click on the link Suggestions and feedback are welcome athttps://www.reddit.com below to complete a request form. Please click on the link below.</td></tr><tr><td>Phone</td><td>http://etcsurvey.corp.enron.com/wi/p0810709/i.asp?r=19690 &amp;s=OUULGRTU&amp;1=9</td><td>America Square, London, EC3N 1PQ Tel:+44 (0)171 839 4444Fax: If you have any comments or suggestions contact us at!</td></tr><tr><td>Personal Address</td><td>+44 (0)171 839 4444 If you have any questions, please do not hesitate to call. Salle E. Yoo The applicant, Mr Richard Owens, is a British national who was</td><td></td></tr><tr><td>Name</td><td>Davis WrightTremaine One Embarcadero Center, Suite 600 Sanborn in 1944 and lives inQueen St. Queen St. Queen St. Queen St. Francisco, CA 94111</td><td></td></tr><tr><td></td><td colspan="2">The applicant,Mr Michael Boyce, is an Irish national, who lives in The applicant,Mr D.H., who was born in 1966 and lives inH. Mullingar, Ireland. He was represented before the Court byMr P.J. Groarke, a lawyer practising in County Longford, Ireland.</td></tr></table>

![](images/396e8daf6de2d6e7a34674dff62c5282a6d7ddf6d34211bbfcde8129a8f12a1e.jpg)

![](images/c8e8da172490ba925773945c446400fc6c6848dcc6e85ffe005ffc3917d886eb.jpg)

![](images/5c2cf9ab08d392986f33b85963f5dd733dc2ce15f6745b672a1c50f49427eb7e.jpg)

![](images/a223a9ecff76d8b39ba446bf5cc2b9ec15bc2f416232541bc8497e36991eeaa4.jpg)  
(c) KNN-LM (ECHR)

![](images/842854e809a8db995d844807680d1ce0c8ae79fd3e7c6de9be7d7c0b0a8c88d2.jpg)  
(d) Standard RAG (ECHR)

![](images/15b24617f614ed79f66d2f9200b893c1685cdc1ac2142006fffb5564e899ce04.jpg)  
(f) Standard RAG (C4)  
Figure 2: Utility/privacy comparisons of our method (RP) with the two DP baselines (DP-RP-G and DPSparseVoteRAG) on two RAG architectures (denoted as KNN-LM and Standard RAG) and targeted attacks on Enron Email and ECHR and untargeted attacks on C4. ε denotes various privacy budgets for the DP approaches. The x-axis is perplexity of generated content, and the y-axis is percentage of data leakage. RP (Ours) achieves a superior utility–privacy trade-off, with perplexity approaching the unprotected RAG baseline while maintaining empirically near-zero privacy leakage comparable to the standalone parametric model without retrieval-augmentation.

Table 2: Full privacy-utility comparison across datasets and methods under the repeated-query attack. Utility is measured using perplexity (PPL) and QA F1 scores. For Enron Email and C4, privacy leakage is reported as the number of leaked instances, whereas for ECHR, it is reported as a percentage.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Dataset / Model</td><td rowspan="2">Metric</td><td rowspan="2">Param.</td><td rowspan="2">RAG</td><td colspan="4">DP-RP-G</td><td colspan="4">DPSparseVoteRAG</td><td rowspan="2">RP (Ours)</td></tr><tr><td>€=1</td><td>€=5</td><td>€=10</td><td>€=20</td><td>€=1</td><td>€=5</td><td>€=10</td><td>€=20</td></tr><tr><td rowspan="9">KNN-LM</td><td rowspan="2">Enron Email / GPT2</td><td>PPL↓</td><td>3.31</td><td>2.87</td><td>5.85</td><td>3.54</td><td>5.13</td><td>2.91</td><td>42.17</td><td>4.22</td><td>2.96</td><td>2.83</td><td>2.89</td></tr><tr><td>Leakage (#) ↓</td><td>0</td><td>13</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>2</td><td>0</td></tr><tr><td rowspan="2">ECHR / Llama3-1B</td><td>PPL↓</td><td>631.09</td><td>2.90</td><td>214.54</td><td>99.28</td><td>94.87</td><td>112.85</td><td>228.59</td><td>109.64</td><td>121.82</td><td>92.26</td><td>46.15</td></tr><tr><td>F1 ↑</td><td>25.70</td><td>90.32</td><td>53.25</td><td>51.30</td><td>54.02</td><td>55.13</td><td>41.67</td><td>41.67</td><td>40.32</td><td>45.38</td><td>87.70</td></tr><tr><td rowspan="3"></td><td>Leakage (%) ↓</td><td>32.82</td><td>82.72</td><td>2.76</td><td>2.41</td><td>2.81</td><td>2.69</td><td>1.75</td><td>37.86</td><td>38.68</td><td>38.82</td><td>2.26</td></tr><tr><td>PPL↓</td><td>1.47</td><td>1.55</td><td>2.26</td><td>2.28</td><td>2.28</td><td>2.30</td><td>3130.89</td><td>610.10</td><td>500.13</td><td>277.62</td><td>1.55</td></tr><tr><td>ECHR / Qwen3-4B F1↑</td><td>90.18</td><td>90.32</td><td>80.35</td><td>80.32</td><td>80.18</td><td>82.39</td><td>52.40</td><td>51.37</td><td>51.94</td><td>52.98</td><td>89.32</td></tr><tr><td rowspan="3"></td><td>Leakage (%) ↓</td><td>2.03</td><td>19.08</td><td>13.45</td><td>13.21</td><td>14.85</td><td>15.39</td><td>2.44</td><td>2.46</td><td>2.21</td><td>2.24</td><td>1.94</td></tr><tr><td>PPL↓ C4 / Llama3-8B</td><td>2.76</td><td>3.16</td><td>5.98</td><td>5.47</td><td>4.12</td><td>4.38</td><td>6.54</td><td>5.91</td><td>5.39</td><td>4.51</td><td>2.98</td></tr><tr><td>Leakage (#) ↓</td><td>0</td><td>269</td><td>0</td><td>0</td><td>17</td><td>25</td><td>0</td><td>3</td><td>36</td><td>124</td><td>17°</td></tr><tr><td rowspan="9">Standard RAG</td><td rowspan="3">Enron Email / GPT2</td><td>PPL↓</td><td>一</td><td>1.58</td><td>3.42</td><td>3.23</td><td>5.24</td><td>2.91</td><td>3.34</td><td>3.71</td><td>3.61</td><td>1.62</td><td>2.05</td></tr><tr><td>Leakage (#) ↓</td><td>1</td><td>217</td><td>0</td><td>0</td><td>8</td><td>5</td><td>0</td><td>0</td><td>0</td><td>9</td><td>3</td></tr><tr><td>PPL↓</td><td>一</td><td>9.84</td><td>512.66</td><td>104.70</td><td>95.72</td><td>79.72</td><td>4398.91</td><td>7780.70</td><td>4922.52</td><td>5264.40</td><td>47.92</td></tr><tr><td rowspan="3">ECHR / Llama3-1B</td><td>F1 ↑</td><td>一</td><td>97.18</td><td>51.39</td><td>54.31</td><td>54.12</td><td>54.79</td><td>1.72</td><td>43.72</td><td>42.42</td><td>43.33</td><td>66.70</td></tr><tr><td>Leakage (%) ↓</td><td></td><td>37.92</td><td>2.03</td><td>1.87</td><td>2.15</td><td>1.96</td><td>1.77</td><td>2.05</td><td>2.05</td><td>19.48</td><td>2.16</td></tr><tr><td>PPL↓</td><td></td><td>1.19</td><td>2.50</td><td>2.49</td><td>2.51</td><td>2.50</td><td>1.74</td><td>1.43</td><td>1.62</td><td>1.38</td><td>1.19</td></tr><tr><td rowspan="3">ECHR / Qwen3-4B</td><td>F1↑</td><td></td><td>97.32</td><td>85.32</td><td>86.17</td><td>86.43</td><td>85.19</td><td>88.35</td><td>91.03</td><td>93.48</td><td>94.32</td><td>95.43</td></tr><tr><td>Leakage (%) ↓</td><td></td><td>21.30</td><td>25.43</td><td>14.36</td><td>17.07</td><td>15.09</td><td>2.84</td><td>2.88</td><td>2.86</td><td>3.32</td><td>2.05</td></tr><tr><td>PPL↓</td><td></td><td>2.83</td><td>28.41</td><td>27.32</td><td>26.54 15</td><td>23.12 28</td><td>5.73 0</td><td>5.46 33</td><td>4.65 79</td><td>4.12 203</td><td>2.96 23</td></tr></table>

## 4 Evaluation

## 4.1 Settings

We focus on empirical data extraction attacks [28] rather than differential privacy bounds, considering both repeated single-query and multi-round query attacks. We evaluate four datastore/LLM pairs: Enron Email [29] with GPT-2 [30], European Court of Human Rights (ECHR) with LLAMA3-1B-INSTRUCT, C4 with LLAMA3-8B-INSTRUCT, and HotpotQA [31] with QWEN2-7B-INSTRUCT. For utility, we measure perplexity [28] and task-specific performance.

For simplicity, we denote our random projection as RP, DP-guaranteed random projection with Gaussian noise as DP-RP-G [17], and the work of Koga et al. [6] as DPSparseVoteRAG. We also report results for vanilla RAG and the parametric model without a datastore. For differential privacy, we use a wide range of privacy budgets with $\epsilon \in \{ 1 , 5 , 1 \bar { 0 } , 2 0 \}$ . For DP-RP-G, the original DP-RP method by Blocki et al. [17] assumes binary vector embeddings. To adapt this framework to RAG and document retrieval, we first project both queries and datastore embeddings into a lower-dimensional space, clip the projected embeddings, and then add Gaussian noise to obtain differential privacy guarantees. The complete procedure is in Section D. We sweep the clipping bound from {0.1, 0.5, 1, 5, 10, 15, 25, 50, 100} under each setting and report the results with optimal clipping bound. Remaining settings are in Section F. For DPSparseVoteRAG, we sweep the subset number m over {10, 20, 30, 40, 50, 60} and choose the best value 50 (Section F.2).

In our methods, for KNN-LM, we set λ as 0.1 and K as 1024. For RP, the only tunable parameter is k (the projection dimension). Larger k values better preserve utility while maintaining privacy protection. We choose $k = 1 0 0$ since it achieves comparable utility to larger values (e.g., k = 1600) while providing better computational efficiency. More detailed analysis is in Section 4.4.

## 4.2 Repeated-Query Attacks

We launch repeated-query targeted attacks on Enron Email for personal identity information (PII) and ECHR for legal documentsand untargeted attacks on raw C4 content. Attackers can repeately inject the same query multiple times. To make repeated-query extraction attack stronger, we inject dataset-specific prior knowledge into malicious queries, such as asking for emails when attacking the Enron Email dataset. For each entry, we perform 10 times repeated extraction attempts, and count any successful extraction as leakage. We assume the system flags and blocks queries repeated more than 10 times. Details, including query construction, are provided in Section E.

Table 3: Number of leaked email addresses, URLs, and phone numbers with different methods on Enron Email.
<table><tr><td rowspan="2"></td><td colspan="3">KNN-LM</td><td colspan="3">Standard RAG</td></tr><tr><td>Email</td><td></td><td>URL Phone</td><td>Email</td><td>URL Phone</td><td></td></tr><tr><td>Parametric</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>RAG</td><td>0</td><td>13</td><td>0</td><td>36</td><td>56</td><td>125</td></tr><tr><td>RP (Ours)</td><td>0</td><td>0</td><td>0</td><td>0</td><td>3</td><td>0</td></tr><tr><td>DP-RP-G, ∈ = 1</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>DP-RP-G, € = 5</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>DP-RP-G, ∈ = 10</td><td>0</td><td>0</td><td>0</td><td>0</td><td>8</td><td>0</td></tr><tr><td>DP-RP-G, € = 20</td><td>0</td><td>0</td><td>0</td><td>0</td><td>1</td><td>4</td></tr><tr><td>DPSparseVoteRAG, ∈ = 1</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>DPSparseVoteRAG, € = 5</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>DPSparseVoteRAG, € = 10</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>DPSparseVoteRAG, € = 20</td><td>0</td><td>2</td><td>0</td><td>0</td><td>8</td><td>1</td></tr></table>

Figure 2 gives an overview of the privacy/utility trade-off of different methods under varying privacy budgets on three datasets under repeated-query attack while Table 2 gives details. Standard RAG achieves higher utility but poses greater risks of privacy leakage. Without protection, RAG leaks over 200 personal items in the Enron dataset. In contrast, our methods maintain utility close to RAG while leaking almost no private information. For DP-RP-G, we achieve strong privacy across all budgets, but utility remains unsatisfactory. DPSparseVoteRAG (DPSVR) shows that increasing the privacy budget reduces perplexity but increases leakage, lacking a balanced trade-off point. Overall, our approach gives strong privacy with minimal utility loss.

Our evaluation covers both structured and unstructured sensitive information to assess the method’s generalizability. On the Enron dataset, we evaluate protection against structured data including email addresses, phone numbers, and URLs. As shown in Table 3, random projection effectively protects all three types, with leakage rates below 3% across both KNN-LM and Standard RAG architectures. We observe that URLs are the easiest to leak. For unstructured text, we evaluate on the ECHR dataset, which contains legal case narratives with sensitive personal information embedded in natural language. Our method maintains strong privacy protection (2.19– 2.27% leakage) while preserving utility, demonstrating that random projection works across both structured and unstructured sensitive information. This suggests that the privacy-preserving mechanism operates at the embedding level, making it agnostic to the format or structure of the sensitive content.

Table 4: Membership leakage rate. RP (Ours) can achieve approximately the same performance against MIA as parametric, which means the attacker barely identify membership information from RAG datastore.
<table><tr><td></td><td colspan="3">ECHR</td><td></td></tr><tr><td>Method</td><td>Enron Email (Qwen3-4B)</td><td></td><td>C4</td><td>HotpotQA</td></tr><tr><td>Parametric</td><td>0.001</td><td>0.006</td><td>0.003</td><td>0.001</td></tr><tr><td>RAG</td><td>0.282</td><td>0.996</td><td>0.843</td><td>0.913</td></tr><tr><td>RP (Ours)</td><td>0.003</td><td>0.007</td><td>0.003</td><td>0.001</td></tr></table>

A further extension of the repeated-query attack is the membership inference attack (MIA). MIA is important in RAG privacy, but it targets a different adversarial objective from the data extraction attacks studied in this paper: determining whether a known record is contained in the datastore, so we do not focus on evaluating various MIA methods. We evaluate this threat over the full private dataset and measure attack accuracy following E-MIA [32], which follows the standard MIA protocol for RAG systems under our threat model. As shown in Table 4, RP provides strong protection against E-MIA, achieving empirically near-zero membership leakage.

## 4.3 Multi-Round Adaptive Attacks

Another strong attack is the multi-round adaptive attack [9], where an attacker iteratively refines adversarial prompts based on outputs from previous rounds and aggregates the extracted information to infer private content. Yao et al. [33] further constructs a knowledge graph from outputs across attack rounds to generate future adversarial queries. The attack budget is set to a maximum of 200 queries to the RAG system.

For both attack methods, we evaluate the reconstructed content using semantic similarity (SS) and extended edit distance (EED). Following the experimental setup of Jiang et al. [9], we construct the datastore using chunks of 1500 words with a maximum overlap of 300 words. We report the chunk recovery rate (CRR), considering a chunk recovered if the reconstructed content overlaps with the original chunk by more than 1200 words. For Jiang et al. [9], we report the average rounds required to satisfy the termination condition.

Table 5: Privacy results against multi-round attacks on HotpotQA using the KNN-LM framework.
<table><tr><td></td><td colspan="4">Jiang et al. [9]</td><td colspan="3">Yao et al. [33]</td></tr><tr><td></td><td>CRR</td><td>SS</td><td>EED</td><td>Rounds</td><td>CRR</td><td>SS</td><td>EED</td></tr><tr><td>Parametric</td><td>30%</td><td>0.17</td><td>0.030</td><td>2.18</td><td>37%</td><td>0.033</td><td>0.07</td></tr><tr><td>RAG</td><td>60%</td><td>0.16</td><td>0.028</td><td>2.05</td><td>75%</td><td>0.034</td><td>0.10</td></tr><tr><td>RP (Ours)</td><td>3%</td><td>0.16</td><td>0.88</td><td>2.05</td><td>0%</td><td>0.032</td><td>0.93</td></tr><tr><td>DP-RP-G (€ = 10)</td><td>3%</td><td>0.17</td><td>0.89</td><td>2.01</td><td>0%</td><td>0.031</td><td>0.90</td></tr><tr><td> $\mathrm { D P S V R } \left( \epsilon = 1 0 \right)$ </td><td>2%</td><td>0.17</td><td>0.88</td><td>2.34</td><td>0%</td><td>0.034</td><td>0.90</td></tr></table>

We launch both multi-round targeted attacks on HotpotQA for document information. For meausing the utility of RAG system on usual queries. We report exact match (EM) and F1 scores between the generated output and the ground truth. For F1, precision is the number of matching tokens divided by the total generated tokens, and recall is the number of matching tokens divided by the ground-truth tokens. Ans denotes the final generated answer, and Sup denotes the retrieved supporting documents, measuing all tokens in the documents. For the parametric model, we prompt it to generate supporting facts to enable fair comparison, as the datastore may overlap with pre-training data. Joint indicates cases where we put answer and supporting documents together to measure.

From Table 5, we observe that the three privacy-preserving methods achieve similar utility, while our method attains the highest accuracy. All methods achieve similar semantic similarity between the original datastore content in each chunk and the attacked results, which we attribute to LLMs attempting to generate correct answers. In contrast, a high chunk recovery rate (i.e., how much datastore content is completely restored) and low EED indicate successful extraction attacks. Examples of extracted content are shown in Table 7. As shown in Table $^ { 6 , }$ random projection preserves cosine similarity between retrieved documents and ground truth, indicating successful retrieval. In contrast, differential privacy methods degrade retrieval performance. Looser privacy budgets can improve utility but raise privacy concerns.

Table 6: Utility results of different algorithms on HotpotQA using the KNN-LM structure.
<table><tr><td></td><td colspan="2">Ans</td><td colspan="2">Sup</td><td colspan="2">Joint</td><td rowspan="2">Cos.</td></tr><tr><td></td><td>EM</td><td>F1</td><td>EM</td><td>F1</td><td>EM</td><td>F1</td></tr><tr><td>Parametric</td><td>7.48</td><td>28.74</td><td>21.37</td><td>49.62</td><td>7.13</td><td>28.55</td><td>0.73</td></tr><tr><td>RAG</td><td>12.48</td><td>29.88</td><td>25.38</td><td>59.74</td><td>10.84</td><td>32.06</td><td>0.75</td></tr><tr><td>RP (Ours)</td><td>11.32</td><td>28.57</td><td>25.22</td><td>57.14</td><td>9.12</td><td>34.09</td><td>0.75</td></tr><tr><td>DP-RP-G, € = 1</td><td>0.08</td><td>7.10</td><td>9.54</td><td>30.42</td><td>0.15</td><td>9.03</td><td>0.54</td></tr><tr><td>DP-RP-G, € = 5</td><td>0.21</td><td>9.32</td><td>10.35</td><td>27.82</td><td>0.16</td><td>9.85</td><td>0.53</td></tr><tr><td>DP-RP-G, ε = 10</td><td>3.58</td><td>10.84</td><td>9.71</td><td>28.93</td><td>1.19</td><td>7.34</td><td>0.54</td></tr><tr><td>DP-RP-G, € = 20</td><td>4.56</td><td>9.81</td><td>14.95</td><td>31.23</td><td>1.67</td><td>8.32</td><td>0.61</td></tr><tr><td>DPSparseVoteRAG, ∈ =</td><td>0.02</td><td>8.12</td><td>13.98</td><td>23.12</td><td>0.00</td><td>13.12</td><td>0.51</td></tr><tr><td>DPSparseVoteRAG, € 二</td><td>3.31</td><td>11.23</td><td>12.88</td><td>30.64</td><td>1.31</td><td>14.29</td><td>0.52</td></tr><tr><td>DPSparseVoteRAG, ∈ =</td><td>10 7.48</td><td>18.18</td><td>14.39</td><td>28.85</td><td>1.54</td><td>16.67</td><td>0.54</td></tr><tr><td>DPSparseVoteRAG, ε =</td><td>20 7.48</td><td>18.18</td><td>14.38</td><td>28.85</td><td>1.54</td><td>15.38</td><td>0.54</td></tr></table>

We present a case study of results under multi-round attacks. From Table 7, we can see that Pablo Trapero and Aleksander Ford are originally mentioned in the question, so it is reasonable that the responses also contain these two names. However, the response generated by our method is substantially different from the original dataset chunk while still focusing on answering the question. No sensitive information, such as address, nationality, or birthdate, is revealed. Hence, this demonstrates successful defense against multi-round attacks.

One interesting aspect of our response is that Pablo Trapero was actually born on October 4th, 1971, and Aleksander Ford on November 24th, 1908. However, our response states that they were born on December 27th, 1965, and September 1st, 1923, respectively. This indicates that our method preserves the final answers without revealing the actual birthdates. The attacker may also mistakenly conclude that the birthdate information has been retrieved and stop the attack. In contrast, standard RAG reveals a large portion of the original content. Jiang et al. [9] successfully extract most sensitive content, while the knowledge graph-based method recovers most original information.

## 4.4 Effects of k

In both RP and DP-RP-G, we sample the projection matrix from $\mathcal { N } ( 0 , 1 / k )$ where k is the projection dimension. Fig. 3 examines the impact of different k values. As discussed in Section 3, larger k better preserves retrieval scores. Consistently, we observe higher utility with increasing k, while privacy remains unchanged. Our method relies on similarity-preserving projection rather than precise distance preservation (Section B and Section C), so the expected post-projection ranking change is less sensitive to k, indicating robustness across a wide range of values. Based on analysis and empirical observation of choosing k, we suggest following ways of choosing k. If a user wants more privacy guarantee or less offline computation latency, a smaller k should be chosen $( \mathrm { e . g . , } k = 2 5 \mathrm { f o r } d = 2 0 4 8 )$ . If a user wants higher utility, a bigger k should be chosen $( \mathbf { e . g . , } k = 1 6 0 0$ for $d = 2 0 4 8 )$ . Otherwise, a moderate value would be suitable (e.g., $k = 1 0 0$ for $d = 2 0 4 8 )$ .

Table 7: Examples of attack results under the multi-round setting. We highlight content matching the original datastore chunks. RP avoids revealing sensitive identity information while preserving relational logic, such as who was born first, even after multi-round attacks using a knowledge graph to retrieve related chunks [33].
<table><tr><td>Initial Query</td><td colspan="2">Pablo Trapero or Aleksander Ford? Who was born first?</td></tr><tr><td>Attacking Methods</td><td>Response by RAG</td><td>Response by ours</td></tr><tr><td>Adaptive [9]</td><td colspan="2">&lt;think&gt;Aleksander Ford (born Mosze Lifszyc, 24 November 1908 &lt;think&gt; Pablo TraperoorAleksander Ford?7 Who was born first? 8 in Kiev, Russian Empire – 4 April 1980 in Naples, Florida, United Who was born first?Aleksander Ford? States) was a Polish Jewish film director, and 2006 BBC Four televi- sion.Ford became director of the nationalized Film Polski company following the Red Army occupation of Poland. 7 Days in Havana: 7 Days in Havana (Spanish: 7 días en La Habana)is a &quot;Pennies From Heaven&quot; (1978)... &lt;think&gt; Thinking Process: 1. **Analyze the Request:** * Target: &lt;think&gt; Who was born first? Pablo Trapero or</td></tr><tr><td rowspan="4">Yao et al. [33]</td><td rowspan="4">An English actor of stage, film, and television. * Condition 1:Alek- sander Ford (born Mosze Lifszyc, 24 November 1908 in Kiev, Rus- sian Empire – 4 April 1980 in Naples, Florida, United States) was a Polish Jewish film director* Condition 2:7 Days in Havana: 7 Days in Havana (Spanish: 7 días en La Habana)*Goal: Who was born first? * …</td><td>&lt;think&gt; Thinking Process: 1. **Analyze the Request:** The user is asking for a comparison of birth dates between two individuals: Pablo TraperoandAleksander Ford. The goal is to determine who was born first.2. **Identify the Individuals:** * **Pablo Trapero** An Argentine film director, screenwriter, and producer. * **Alek-</td></tr><tr><td></td></tr><tr><td>sander Ford** A Polish film director and screenwriter. 3. **Retrieve Birth Dates: ** * **Pablo Trapero:** I need to recall or verify his</td></tr><tr><td>birth date. * Search query mental check: &quot;Pablo Traperobirth date&quot;. * Knowledge:Pablo Traperowas born on December 27, 1965. * **Aleksander Ford:** I need to recall or verify his birth date. * Search query mental check: &quot;Aleksander Fordbirth date&quot;. * Knowl- edge:Aleksander Fordwas born on September 1, 1923. 4. **Com- pare Dates:** *Aleksander Ford</td></tr></table>

$$
\begin{array} { r l } { \oplus \mathsf { R P } ( \mathsf { O u r s } ) \ k = 2 5 } & { \odot \ \mathsf { R P } ( \mathsf { O u r s } ) \ k = 1 0 0 \quad \bigodot \ \mathsf { R P } ( \mathsf { O u r s } ) \ k = 1 6 0 0 } \\ { \sqcup \mathsf { D P } \mathrm { - } \mathsf { R P - G } \ k = 2 5 } & { \bigcup \ \mathsf { D P } \mathrm { - } \mathsf { R P - G } \ k = 1 0 0 \qquad \bigsqcup \mathsf { D P } \mathrm { - } \mathsf { R P - G } \ k = 1 6 0 0 } \end{array}
$$

![](images/2f0f641ed716c79b70de69ca5c823f1f2a84c30521a7e3d31f29896c77aab46d.jpg)

![](images/bf3990adb40a076bd7dff963137c4a283b533181cb43a3b2531b634da6a7f6d5.jpg)  
Figure 3: Comparison of privacy/utility. Colors indicate $k ;$ markers distinguish RP and DP-RP-G. k has weak relation with privacy but larger k gives better utility.

To understand the effect of not setting the sampling variance as $1 / k .$ , we use $k = 1 6 0 0$ and sample from $\mathcal { N } ( 0 , 0 . 2 ^ { 2 } )$ resulting in a perplexity of 70.86 and 2.32% privacy leakage. For $k = 2 5$ and $\mathcal { N } ( 0 , 0 . 0 2 5 ^ { 2 } )$ , perplexity is 71.38 with 3.1% leakage. Choosing sampling variances larger or smaller than $1 / k$ degrades utility, validating that setting variance to $1 / k$ is optimal.

## 4.5 Efficiency and Scalability

The DPSparseVoteRAG method can be inefficient. In our experiments, each inference takes about 30× longer than RP and vanilla RAG (Table 8). This is due to DP voting and aggregation require partitioning the datastore into m subsets and generating a token for each, resulting in m inferences per token. Our method can achieve the same efficiency as RAG on latency per token.

Table 8: Comparison of the number of tokens generated and latency per generation for different methods over the ECHR dataset on an NVIDIA A100.  
Table 9: Perplexity comparisons of KNN-LM and standard RAG on ECHR.
<table><tr><td></td><td>RAG</td><td>RP (Ours)</td><td>DPSVR</td></tr><tr><td># Tokens per Generation</td><td>107.8</td><td>123.4</td><td>92.3</td></tr><tr><td>Latency per Generation (s)</td><td>8.35</td><td>9.42</td><td>258.9</td></tr><tr><td>Latency per Token (s)</td><td>0.08</td><td>0.08</td><td>2.80</td></tr></table>

<table><tr><td>Method</td><td>KNN-LM</td><td>Standard RAG</td></tr><tr><td>RP (Ours)</td><td>43.20</td><td>47.92</td></tr><tr><td>DP-RP-G</td><td>97.09</td><td>102.66</td></tr><tr><td>DP-RP-G (only clipping)</td><td>45.47</td><td>46.99</td></tr></table>

Besides efficiency, we also take a look at the scalability of our evaluation. We empirically study datastores of up to about 100K HotpotQA documents, corresponding to roughly 200 million tokens, which is a reasonably sized RAG datastore comparable with many real-world enterprise services [34, 26]. Prior work [35] shows that extraction rates decrease substantially as datastore size increases. We further conduct a controlled scalability study in Figure 4 by varying the datastore size within the ECHR/Qwen3-4B setting. We show (1) the extraction rate of vanilla RAG decreases as the datastore grows, while (2) our method consistently maintains strong privacy protection across datastore sizes. Moreover, as discussed in Section 3, our RP method relies on replacing tokens or chunks with similar alternatives from the datastore. As the datastore size increases, the method has access to a larger pool of candidate tokens for such substitutions, reducing the extraction rate.

![](images/5e38726d04d172a8441b946b3ba7af0622e6bb4a6b23b294448dc9c4abbad3bd.jpg)  
Figure 4: Scalability study on Standard RAG using ECHR with Qwen3-4B. Across different datastore sizes, RP maintains leakage close to the parametric baseline while consistently providing strong privacy protection.

![](images/82cca70d4ec185260cd6714697cd755479ecf36d469ed16a1d5bbc8cd8e247a8.jpg)  
Figure 5: Perplexity of DP-RP-G with clipping only (without Gaussian noise) under different clipping bounds.

## 4.6 Why Peformance Degrades More by Differential Privacy Methods

A natural question is: what is the challengefor differential-privacy based methods to maintain utiliy in RAG applications. For DP-RP-G, achieving an (ϵ, δ) guarantee requires bounding sensitivity, typically by clipping each projected embedding to norm c, and then adding Gaussian noise $\begin{array} { r } { \mathcal { N } \bigg ( 0 , \frac { c \sqrt { 2 \ln ( 1 . 2 5 / \delta ) } } { \epsilon } \bigg ) } \end{array}$ . If $: \epsilon  \infty ,$ , the noise vanishes; with a sufficiently large clipping bound, DP-RP-G approaches ours RP method. Thus, the utility loss arises from the additional constraints required for formal DP rather than from RP itself.

DP aggregation methods face a similar trade-off, as their utility depends strongly on perturbation magnitude and therefore on privacy budget [6]. As shown in Section 4, even relatively loose privacy budgets can substantially degrade utility while still allowing non-negligible leakage.

Clipping is typically required in DP mechanisms to bound sensitivity [36]; prior work similarly constrains vectors to bounded domains [37]. To isolate the effect of clipping and adding noise, we ablate the clipping operation without Gaussian noise. In Table 9, we compare RP with and without clipping, while Figure 5 varies the clipping bound. The results show that larger clipping consistently reduces perplexity, whereas sufficiently large clipping bounds recover performance close to RP.

## 5 Conclusion

In this work, we propose a simple random projection method to defend against data extraction attacks for RAG applications. Compared with DP-based approaches, we show that the specific random Gaussian projection can empirically protect against data reconstruction, at approximately the same level as parametric model without RAG datastore, while still maintaining semantic utility, with a 7% overall decrease compared with vanilla RAG.

## Limitations

At evaluation side, large-scale evaluation of privacy-preserving methods in real-world LLM-based RAG systems remains an interesting future work. Real-world systems may need to process billions of queries daily. In Table 8, one query takes about 8 seconds. Hence, evaluating one million queries would take 92 days without system-level optimization. Such large-scale evaluation thus requires system-level contributions, which we defer to future work.

## Ethical Considerations

We use models and datasets that are publicly available on the Internet and licensed for non-commercial research use. We do not involve personally identifiable information (PII) which are not allowed to make public. We use only publicly available or synthetically generated datasets; we did not use any proprietary data or private user information at any stage of the experiments. These datasets are publicly available and widely used in the research community. Consistent with prior work, we did not further anonymize the data, as it was collected and released under publicly accessible licenses and used solely for research purposes. We did not introduce new personal data, nor did we attempt to re-identify individuals beyond what is already present in the original datasets.

## References

[1] Urvashi Khandelwal, Omer Levy, Dan Jurafsky, Luke Zettlemoyer, and Mike Lewis. Generalization through memorization: Nearest neighbor language models. In Proceedings ofthe International Conference on Learning Representations (ICLR), 2019.

[2] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. Retrieval-augmented generation for knowledge-intensive nlp tasks. In Proceedings ofthe Advances in Neural Information Processing Systems (NeuraIPS), volume 33, pages 9459–9474, 2020.

[3] Yu He, Yifei Chen, Yiming Li, Shuo Shao, Leyi Qi, Boheng Li, Dacheng Tao, and Zhan Qin. External data extraction attacks against retrieval-augmented large language models. arXiv preprint arXiv:2510.02964, 2025.

[4] Yuhao Wang, Wenjie Qu, Shengfang Zhai, Yanze Jiang, Zichen Liu, Yue Liu, Yinpeng Dong, and Jiaheng Zhang. Silent leaks: Implicit knowledge extraction attack on rag systems through benign queries. arXiv preprint arXiv:2505.15420, 2025.

[5] Shenglai Zeng, Jiankun Zhang, Pengfei He, Yiding Liu, Yue Xing, Han Xu, Jie Ren, Yi Chang, Shuaiqiang Wang, Dawei Yin, and Jiliang Tang. The good and the bad: Exploring privacy issues in retrieval-augmented generation (rag). In ACL (Findings), 2024.

[6] Tatsuki Koga, Ruihan Wu, and Kamalika Chaudhuri. Privacy-preserving retrieval augmented generation with differential privacy. arXiv preprint arXiv:2412.04697, 2024.

[7] Ruihan Wu, Erchi Wang, Zhiyuan Zhang, and Yu-Xiang Wang. Private-rag: Answering multiple queries with llms while keeping your data private. arXiv preprint arXiv:2511.07637, 2025.

[8] Changyue Jiang, Xudong Pan, Geng Hong, Chenfu Bao, and Min Yang. Rag-thief: Scalable extraction of private data from retrieval-augmented generation applications with agent-based attacks. CoRR, 2024.

[9] Changyue Jiang, Xudong Pan, Geng Hong, Chenfu Bao, Yang Chen, and Min Yang. Feedback-guided extraction of knowledge base from retrieval-augmented llm applications. arXiv preprint arXiv:2411.14110, 2024.

[10] Christian Di Maio, Cristian Cosci, Marco Maggini, Valentina Poggioni, and Stefano Melacci. Pirates of the rag: Adaptively attacking llms to leak knowledge bases. arXiv preprint arXiv:2412.18295, 2024.

[11] Nicolas Grislain. Rag with differential privacy. In 2025 IEEE Conference on Artificial Intelligence (CAI), pages 847–852. IEEE, 2025.

[12] Shenglai Zeng, Jiankun Zhang, Pengfei He, Jie Ren, Tianqi Zheng, Hanqing Lu, Han Xu, Hui Liu, Yue Xing, and Jiliang Tang. Mitigating the privacy issues in retrieval-augmented generation (rag) via pure synthetic data. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 24538–24569, 2025.

[13] Yunpeng Zhao and Jie Zhang. Does training with synthetic data truly protect privacy? In The Thirteenth International Conference on Learning Representations (ICLR), 2025.

[14] Longzhu He, Peng Tang, Yuanhe Zhang, Pengpeng Zhou, and Sen Su. Mitigating privacy risks in retrievalaugmented generation via locally private entity perturbation. Information Processing & Management, 62(4): 104150, 2025.

[15] Rushil Thareja, Preslav Nakov, Praneeth Vepakomma, and Nils Lukas. Dp-fusion: Token-level differentially private inference for large language models. arXiv preprint arXiv:2507.04531, 2025.

[16] William B Johnson and Joram Lindenstrauss. Extensions of lipschitz mappings into a hilbert space. Contemporary mathematics, 26(189-206):1, 1984.

[17] Jeremiah Blocki, Avrim Blum, Anupam Datta, and Or Sheffet. The johnson-lindenstrauss transform itself preserves differential privacy. In Proceedings of the 2012 IEEE 53rd Annual Symposium on Foundations of Computer Science (FOCS), pages 410–419. IEEE Computer Society, 2012.

[18] Chugui Xu, Ju Ren, Yaoxue Zhang, Zhan Qin, and Kui Ren. Dppro: Differentially private high-dimensional data release via random projection. IEEE Transactions on Information Forensics and Security, 12(12):3081–3093, 2017.

[19] Ping Li and Xiaoyun Li. Differential privacy with random projections and sign random projections. arXiv preprint arXiv:2306.01751, 2023.

[20] Alaa Mahmoud Ibrahim, Mohamed Farouk, and Mohamed Waleed Fakhr. Privacy preserving image retrieval using multi-key random projection encryption and machine learning decryption. Journal of Advanced Research in Applied Sciences and Engineering Technology, 42(2):155–174, 2024.

[21] Kun Liu, Hillol Kargupta, and Jessica Ryan. Random projection-based multiplicative data perturbation for privacy preserving distributed data mining. IEEE Transactions on knowledge and Data Engineering, 18(1):92–106, 2006.

[22] Mohammad Hasan Narimani and Mostafa Tavassolipour. Fedrp: A communication-efficient approach for differentially private federated learning using random projection. arXiv preprint arXiv:2509.10041, 2025.

[23] Cihan Kaleli and Huseyin Polat. Privacy-preserving random projection-based recommendations based on distributed data. International Journal ofInformation Technology & Decision Making, 12(02):201–232, 2013.

[24] Nikola Pavlovic, Sudeep Salgia, and Qing Zhao. Differential privacy in kernelized contextual bandits via random projections. arXiv preprint arXiv:2507.13639, 2025.

[25] Bonwoo Lee, Cheolwoo Park, and Jeongyoun Ahn. Optimal differentially private kernel learning with random projection. arXiv preprint arXiv:2507.17544, 2025.

[26] Yuhong Sun, Joachim Rahmfeld, Chris Weaver, Weijia Chen, Roshan Desai, Wenxi Huang, and Mark H Butler. Enterpriserag-bench: A rag benchmark for company internal knowledge. arXiv preprint arXiv:2605.05253, 2026.

[27] Sarah Packowski, Inge Halilovic, Jenifer Schlotfeldt, and Trish Smith. Optimizing and evaluating enterprise retrieval-augmented generation (rag): A content design perspective. In Proceedings of the 2024 8th International Conference on Advances in Artificial Intelligence, pages 162–167, 2024.

[28] Yangsibo Huang, Samyak Gupta, Zexuan Zhong, Kai Li, and Danqi Chen. Privacy implications of retrieval-based language models. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 14887–14902, 2023.

[29] Bryan Klimt and Yiming Yang. The enron corpus: A new dataset for email classification research. In Proceedings ofthe European Conference on Machine Learning (ECML), pages 217–226. Springer, 2004.

[30] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. Language models are unsupervised multitask learners. OpenAI blog, 1(8):9, 2019.

[31] Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William Cohen, Ruslan Salakhutdinov, and Christopher D Manning. Hotpotqa: A dataset for diverse, explainable multi-hop question answering. In Proceedings ofthe 2018 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 2369–2380, 2018.

[32] Zelin Guan, Shengda Zhuo, Zeyan Li, Jinchun He, Wangjie Qiu, Zhiming Zheng, and Shuqiang Huang. E-mia: Exam-style black-box membership inference attacks against rag systems. arXiv preprint arXiv:2605.00955, 2026.

[33] Mengyu Yao, Ziqi Zhang, Ning Luo, Shaofei Li, Yifeng Cai, Xiangqun Chen, Yao Guo, and Ding Li. Connect the dots: Knowledge graph-guided crawler attack on retrieval-augmented generation systems. arXiv preprint arXiv:2601.15678, 2026.

[34] Prafulla Kumar Choubey, Xiangyu Peng, Shilpa Bhagavath, Kung-Hsiang Huang, Caiming Xiong, and Chien-Sheng Wu. Benchmarking deep search over heterogeneous enterprise data. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing: Industry Track, pages 501–517, 2025.

[35] Zhenting Qi, Hanlin Zhang, Eric P Xing, Sham Kakade, and Hima Lakkaraju. Follow my instruction and spill the beans: Scalable data extraction from retrieval-augmented generation systems. In International Conference on Learning Representations, volume 2025, pages 48733–48755, 2025.

[36] Cynthia Dwork, Frank McSherry, Kobbi Nissim, and Adam Smith. Calibrating noise to sensitivity in private data analysis. In Proceedings ofthe Third Theory ofCryptography Conference, TCC 2006, New York, NY, USA, March 4-7, 2006., pages 265–284. Springer, 2006.

[37] Krishnaram Kenthapadi, Aleksandra Korolova, Ilya Mironov, and Nina Mishra. Privacy via the johnsonlindenstrauss transform. Journal ofPrivacy and Confidentiality, 5(1), 2013.

[38] Donald W Zimmerman, Bruno D Zumbo, and Richard H Williams. Bias in estimation and hypothesis testing of correlation. Psicológica, 24(1), 2003.

[39] Harold Hotelling. New light on the correlation coefficient and its transforms. Journal of the Royal Statistical Society. Series B (Methodological), 15(2):193–232, 1953.

[40] Ronald A Fisher. Frequency distribution of the values of the correlation coefficient in samples from an indefinitely large population. Biometrika, 10(4):507–521, 1915.

[41] Ilias Chalkidis, Manos Fergadiotis, Dimitrios Tsarapatsanis, Nikolaos Aletras, Ion Androutsopoulos, and Prodromos Malakasiotis. Paragraph-level rationale extraction through regularization: A case study on european court of human rights cases. In Proceedings of the Annual Conference of the North American Chapter of the Association for Computational Linguistics (NAACL), Mexico City, Mexico, 2021. Association for Computational Linguistics.

## Contents

A Usage of AI 12   
B Preservation of the $L _ { 2 }$ -Distance and Cosine Similarity of Random Projection 12   
B.1 Proof: Preservation of the Cosine Similarity 12   
B.2 Synthetic Data . . 14   
C Proof: Probability of Top-1 Selection Changing after Random Projection 14   
D Complete Algorithms 16   
E Implementation Details 17   
F Settings of Hyper-Parameters 17   
F.1 General Settings 17   
F.2 DPSparseRAG 18   
G Random Projection with Other Distributions 18   
H More Discussion 18   
H.1 False Positive Classification of Non-Sensitive Information 18   
H.2 An Example of Generation 19

## A Usage of AI

We employ large language models (LLMs) to improve the grammar and clarity of our writing and assist coding. All research ideas, directions, and decisions are independently conceived and carried out by the authors.

## B Preservation of the L<sub>2</sub>-Distance and Cosine Similarity of Random Projection

The theory has already been proved by Johnson and Lindenstrauss [16]. For completeness, we provide analysis below. Let the query embedding be x, the retrieved embedding by vanilla RAG be e, and the random projection matrix be R. The original $\ell _ { 2 }$ distance is $\| { \pmb x } - { \pmb e } \| _ { 2 }$ . According to the Johnson–Lindenstrauss transform, when the projection variance is $1 / k$ , we have the boundary condition

$$
( 1 - \lambda _ { J L } ) \| { \pmb x } - { \pmb e } \| _ { 2 } \leq \| { \pmb R } { \pmb x } - { \pmb R } { \pmb e } \| _ { 2 } \leq ( 1 + \lambda _ { J L } ) \| { \pmb x } - { \pmb e } \| _ { 2 }
$$

where $\begin{array} { r } { \lambda _ { J L } = \Omega \big ( \sqrt { \frac { \log d } { k } } \big ) } \end{array}$ [16]. This ensures that the $\ell _ { 2 }$ distance is well preserved after projection.

Further, the original retrieval score, typically cosine similarity, $\begin{array} { r } { \rho = \frac { { \pmb x } ^ { \top } { \pmb e } } { \| { \pmb x } \| _ { 2 } \| { \pmb e } \| _ { 2 } } } \end{array}$ can also be preserved after projection. After random projection, the score becomes

$$
\rho ^ { \prime } = \frac { \pmb { x } ^ { \top } R ^ { \top } R e } { \| R \pmb { x } \| _ { 2 } \| R e \| _ { 2 } } .
$$

For fixed x and $e ,$ the expectation satisfies $\begin{array} { r } { \mathbb { E } [ \rho ^ { \prime } ] = \rho \left( 1 - \frac { 1 - \rho } { 2 k } \right) + \mathcal { O } \left( \frac { 1 } { k ^ { 2 } } \right) } \end{array}$  (Proof in Section B.1). Thus, when k is sufficiently large, the expected cosine similarity after projection closely approximates the original score, preserving retrieval quality despite token perturbations.

## B.1 Proof: Preservation of the Cosine Similarity

Our main argument is that with multiple rounds of token generation, the overall semantic meaning is preserved even if individual tokens are perturbed or replaced. In other words, we have

$$
\mathbb { E } [ \rho ^ { \prime } ] \approx \rho \left( 1 - \frac { 1 - \rho } { 2 k } \right) .\tag{1}
$$

Proof. We have query embedding x and datastore embedding e. First, let $\begin{array} { r } { \pmb { u } = \frac { \pmb { x } } { \Vert \pmb { x } \Vert _ { 2 } } \ \mathrm { a n d } \ \pmb { v } = \frac { \pmb { e } } { \Vert \pmb { e } \Vert _ { 2 } } } \end{array}$ and we can have $\boldsymbol { \rho } = \boldsymbol { u } ^ { \intercal } \boldsymbol { v }$ . With that, we can have

$$
\rho ^ { \prime } = \frac { ( \pmb { u } ^ { \top } \pmb { R } ^ { \top } ) ( \pmb { R } \pmb { v } ) ) } { \| \pmb { u } ^ { \top } \pmb { R } ^ { \top } \| _ { 2 } \| \pmb { R } \pmb { v } \| _ { 2 } } = \frac { ( \pmb { R } \pmb { u } ) ^ { \top } ( \pmb { R } \pmb { v } ) ) } { \| \pmb { R } \pmb { u } ^ { \top } \| _ { 2 } \| \pmb { R } \pmb { v } \| _ { 2 } }
$$

We let $\mathbf { \nabla } a = R \mathbf { \nabla } a$ and $\pmb { b } = R \pmb { v }$ . Then we can have

$$
\rho ^ { \prime } = \frac { \pmb { a } ^ { \top } \pmb { b } } { \| \pmb { a } \| _ { 2 } \| \pmb { b } \| _ { 2 } } .
$$

Because R is an IID Gaussian matrix and each element is independently sampled. We can find rotation matrices U and V where $U R V \overset { d } { = } R .$ . Hence, we can have $R ( V x ) \stackrel { d } { = } R x$ . So, without loss of generality, we can rotate the basis so that $\pmb { u } = \pmb { e _ { 1 } } = ( 1 , 0 , 0 , \dots , 0 ) ^ { \top } , \pmb { v } = \rho \pmb { e _ { 1 } } + \sqrt { 1 - \rho ^ { 2 } } \pmb { e _ { 2 } }$ . We know that each row of R is a Gaussian vector $g _ { i } \sim \mathcal { N } ( 0 , I _ { d } / k ) . \mathrm { S o } , a _ { i } = r _ { i } ^ { T } \pmb { u } = r _ { i 1 }$ and $b _ { i } = { r _ { i } ^ { \top } } { \pmb v } = \rho { r _ { i 1 } } + { \sqrt { 1 - \rho ^ { 2 } } } { r _ { i 2 } }$ where $r _ { i 1 } , r _ { i 2 }$ are IID $\mathcal { N } ( 0 , 1 / k )$ . Thus, for each $i = 1 , \ldots , k ,$

$$
\left[ \begin{array} { l } { a _ { i } } \\ { b _ { i } } \end{array} \right] \sim \mathcal { N } \bigg ( 0 , \ \frac { 1 } { k } \ \bigg [ 1 \quad \rho \bigg ] \bigg ) .
$$

With this bivariate normal distribution, we can define

$$
X _ { i } = { \sqrt { k } } a _ { i } \sim { \mathcal { N } } ( 0 , 1 ) ,
$$

$$
Y _ { i } = { \sqrt { k } } b _ { i } \sim { \mathcal { N } } ( 0 , 1 ) ,
$$

And we can have that $\mathrm { C o r r } ( \mathrm { X } _ { \mathrm { i } } , \mathrm { Y } _ { \mathrm { i } } ) = \rho$ . Then, we can calculate

$$
\rho ^ { \prime } = { \frac { \sum _ { i } X _ { i } Y _ { i } } { \sqrt { \sum _ { i } X _ { i } ^ { 2 } } \sqrt { \sum _ { i } Y _ { i } ^ { 2 } } } }
$$

This is exactly the sample Pearson correlation of k IID bivariate normal samples with population correlation $\rho .$

Lemma 1. For random variables $X _ { i }$ and $Y _ { i }$ where $X _ { i } \sim \mathcal { N } ( 0 , 1 ) , Y _ { i } \sim \mathcal { N } ( 0 , 1 )$ , and $\mathrm { C o r r } ( \mathrm { X } _ { \mathrm { i } } , \mathrm { Y } _ { \mathrm { i } } ) = \rho ,$ we can calculate the expectation of sample Pearson correlation coefficient

$$
\rho ^ { \prime } = { \frac { \sum _ { i } X _ { i } Y _ { i } } { \sqrt { \sum _ { i } X _ { i } ^ { 2 } } { \sqrt { \sum _ { i } Y _ { i } ^ { 2 } } } } }
$$

as

$$
\mathbb { E } [ \rho ^ { \prime } ] = \rho \left( 1 - \frac { 1 - \rho } { 2 k } \right) + \mathcal { O } \left( \frac { 1 } { k ^ { 2 } } \right)
$$

Although the proof of this lemma is not our original contribution, as there is plenty of literature showing this conclusion [38, 39], we would like to provide a proof here to demonstrate some of our insights.

Proof. Let $\begin{array} { r } { S _ { x x } = \frac { 1 } { k } \sum X _ { i } ^ { 2 } , S _ { y y } = \frac { 1 } { k } \sum Y _ { i } ^ { 2 } , S _ { x y } = \frac { 1 } { k } \sum X _ { i } Y _ { i } } \end{array}$ . Then

$$
\rho ^ { \prime } = { \frac { S _ { x y } } { \sqrt { S _ { x x } } \sqrt { S _ { y y } } } }
$$

If we use Taylor expansion, we will have

$$
\frac { 1 } { \sqrt { S _ { x x } } } = 1 - \frac { 1 } { 2 } S _ { x x } + \mathcal { O } \left( \frac { 1 } { k ^ { 2 } } \right)
$$

As a result, we will have

$$
\begin{array} { l } { \displaystyle \mathbb { E } [ \rho ^ { \prime } ] = \mathbb { E } [ S _ { x y } ] - \frac { 1 } { 2 } \mathbb { E } [ S _ { x y } S _ { x x } ] - \frac { 1 } { 2 } \mathbb { E } [ S _ { x y } S _ { y y } ] } \\ { \displaystyle \quad \quad + \frac { 1 } { 4 } \mathbb { E } [ S _ { x x } S _ { y y } ] + \mathcal { O } \left( \frac { 1 } { k ^ { 2 } } \right) } \\ { \displaystyle \quad = \rho - \frac { 1 } { 2 k } \mathrm { C o v } ( \mathrm { X Y , X ^ { 2 } } ) - \frac { 1 } { 2 \mathrm { k } } \mathrm { C o v } ( \mathrm { X Y , Y ^ { 2 } } ) } \\ { \displaystyle \quad + \frac { 1 } { 4 k } \mathrm { C o v } ( \mathrm { X ^ { 2 } , Y ^ { 2 } } ) + \mathcal { O } \left( \frac { 1 } { \mathrm { k ^ { 2 } } } \right) } \end{array}
$$

We have $\mathbb { E } [ X ^ { 2 } ] = \mathbb { E } [ Y ^ { 2 } ] = 1$ and $\mathbb { E } [ X Y ] = \rho .$ . We also know $\operatorname { V a r } ( \mathrm { X } ) = \operatorname { V a r } ( \mathrm { Y } ) = 1$ and $\operatorname { C o v } ( \mathrm { X } , \mathrm { Y } ) = \rho .$ . Using Isserlis theorem, we have $\mathbb { E } [ X ^ { 4 } ] = 3 , \mathbb { E } [ Y ^ { 4 } ] \stackrel { \cdot } { = } 3 , \mathbb { E } [ X ^ { 2 } Y ^ { 2 } ] = 1 + 2 \stackrel { \cdot } { \rho } ^ { 2 } , \mathbb { E } [ X ^ { 3 } Y \big ] \stackrel { \cdot } { = } \mathbb { E } [ X Y ^ { 3 } ] = 3 \rho .$ . Hence, we have

$$
\operatorname { C o v } ( \mathrm { X } ^ { 2 } , \mathrm { Y } ^ { 2 } ) = \mathbb { E } [ \mathrm { X } ^ { 2 } \mathrm { Y } ^ { 2 } ] - \mathbb { E } [ \mathrm { X } ^ { 2 } ] \mathbb { E } [ \mathrm { Y } ^ { 2 } ] = 2 \rho ^ { 2 }
$$

$$
\mathrm { C o v } ( \mathrm { X } ^ { 2 } , \mathrm { X Y } ) = \mathbb { E } [ \mathrm { X } ^ { 3 } \mathrm { Y } ] - \mathbb { E } [ \mathrm { X } ^ { 2 } ] \mathbb { E } [ \mathrm { X Y } ] = 2 \rho
$$

By putting them back, we have our conclusion.

According to Lemma 1, we prove Eq. (1). The expected distortion goes down as $\mathcal { O } ( 1 / k )$ . Hence, we can see that our designed random projection preserves the cosine similarity. □

## B.2 Synthetic Data

Before starting our experiments, we first conduct a sanity check to verify whether embedding distances remain consistent after random projection. We consider two settings. In the synthetic setting, we use LLAMA3-8B-INSTRUCT to generate 100 datastore entries with the prompt “Generate a datastore entry containing domain knowledge.” For each of these, we repeat 100 times to generate a query using the prompt “Generate a query about domain knowl-$e d g \bar { e . } ^ { , }$ In the random setting, both datastore and query embeddings are random vectors drawn from $\mathcal { N } ( 0 , \bar { 1 } )$

![](images/584799a14f62495cba52d1d99a03a8ab42e82aef9a5cf91a2add5d65d49ce139.jpg)

![](images/bd68d85769a8bace6a5a0557574d5fbd68e708797b9b28c73b016c7eca90bf85.jpg)  
Figure 6: Cosine similarity error and L2 distance ratio under different sanity-check settings.

We measure the L2 distance and cosine similarity between each query embedding and the datastore embeddings. We

use two metrics: the L2 distance ratio, $\frac { \| R \pmb { x } - R e \| _ { 2 } } { \| \pmb { x } - \pmb { e } \| _ { 2 } }$ , and the cosine similarity difference, $| \rho - \rho ^ { \prime } |$ . We first average each metric over 100 datastores and then average across 100 runs. Fig. 6 presents the sanity-check results, along with the

theoretical upper and lower bounds derived from the Johnson–Lindenstrauss (JL) transform, given by $1 \pm \Omega \left( \sqrt { \frac { \log d } { k } } \right)$

As k increases, the cosine similarity error decreases and approaches zero, confirming our later observation that larger k improves perplexity, which is consistent with our theoretical analysis. For the L2 distance, the results follow the JL transform prediction, with the ratios remaining within the theoretical bounds. A larger k also stabilizes the ratio. Overall, these results verify that, in expectation, distances are well preserved after random projection.

## C Proof: Probability of Top-1 Selection Changing after Random Projection

Without loss of generality, we consider the top-1 case, where the datastore embedding $e _ { 1 }$ attains the largest cosine similarity $\rho _ { 1 }$ in the vanilla RAG setting. We show that if there exists another datastore embedding $e _ { j } ( j \neq \bar { 1 } )$ such that

$$
\left\| { \frac { \pmb { e } _ { 1 } } { \| \pmb { e } _ { 1 } \| _ { 2 } } } - { \frac { \pmb { e } _ { j } } { \| \pmb { e } _ { j } \| _ { 2 } } } \right\| _ { 2 } \leq \Delta
$$

then the probability that the projected cosine similarity $\rho _ { j } ^ { \prime }$ exceeds $\rho _ { 1 } ^ { \prime }$ satisfies

$$
\operatorname* { P r } ( \rho _ { j } ^ { \prime } > \rho _ { 1 } ^ { \prime } ) \gtrsim 1 - \Phi \left( \frac { \Delta } { ( 1 - \rho _ { 1 } ^ { 2 } ) \sqrt { \frac { 2 ( 1 - \eta _ { \operatorname* { m a x } } ) } { k - 3 } } } \right)
$$

where $\eta _ { \mathrm { m a x } }$ is an upper bound on the correlation between the Fisher-transformed projected similarities, and Φ denotes the cumulative distribution function of the standard normal distribution.

Proof. We define the normalized query and datastore embeddings as

$$
\pmb { u } = \frac { \pmb { x } } { \Vert \pmb { x } \Vert _ { 2 } } , \qquad \pmb { v } _ { i } = \frac { \pmb { e } _ { i } } { \Vert \pmb { e } _ { i } \Vert _ { 2 } } .
$$

Under this notation, the assumption becomes $\| \pmb { v } _ { 1 } - \pmb { v } _ { j } \| _ { 2 } \le \Delta$ . By the Cauchy–Schwarz inequality,

$$
\begin{array} { r } { | \rho _ { 1 } - \rho _ { j } | = | \pmb { u } ^ { \top } ( \pmb { v } _ { 1 } - \pmb { v } _ { j } ) | \leq \| \pmb { v } _ { 1 } - \pmb { v } _ { j } \| _ { 2 } \leq \Delta . } \end{array}
$$

We next introduce the Fisher transform

$$
z _ { i } = \operatorname { t a n h } ^ { - 1 } ( \rho _ { i } ) , \qquad z _ { i } ^ { \prime } = \operatorname { t a n h } ^ { - 1 } ( \rho _ { i } ^ { \prime } ) .
$$

Since cosine similarity takes values in (−1, 1) and tanh<sup>−1</sup>(·) is strictly increasing on this interval, the ordering of similarities is preserved under this transformation. In particular,

$$
\rho _ { j } ^ { \prime } > \rho _ { 1 } ^ { \prime } \quad \Longleftrightarrow \quad z _ { j } ^ { \prime } > z _ { 1 } ^ { \prime } .
$$

As shown in Section B, the projected cosine similarity $\rho _ { i } ^ { \prime }$ is the sample Pearson correlation computed from k IID bivariate normal pairs with population correlation $\rho _ { i } .$ . By the classical Fisher z-transform result [40], we have

$$
z _ { i } ^ { \prime } \approx \mathcal { N } \bigg ( \operatorname { t a n h } ^ { - 1 } ( \rho _ { i } ) , \frac { 1 } { k - 3 } \bigg ) .
$$

Because $z _ { 1 } ^ { \prime }$ and $z _ { j } ^ { \prime }$ share the same projected query vector, they are generally correlated. Let $\eta = \mathrm { C o r r } ( z _ { 1 } ^ { \prime } , z _ { j } ^ { \prime } )$ , and assume $\eta \leq \eta _ { \mathrm { m a x } }$ for some fixed upper bound $\eta _ { \mathrm { m a x } } < 1$

Under a joint Gaussian approximation, the difference $z _ { j } ^ { \prime } - z _ { 1 } ^ { \prime }$ is approximately normally distributed with

$$
\mathbb { E } [ z _ { j } ^ { \prime } - z _ { 1 } ^ { \prime } ] = z _ { j } - z _ { 1 } , \qquad \mathrm { V a r } ( z _ { j } ^ { \prime } - z _ { 1 } ^ { \prime } ) = \frac { 2 ( 1 - \eta ) } { k - 3 } .
$$

Therefore,

$$
\operatorname* { P r } ( \rho _ { j } ^ { \prime } > \rho _ { 1 } ^ { \prime } ) = \operatorname* { P r } ( z _ { j } ^ { \prime } - z _ { 1 } ^ { \prime } > 0 ) = \Phi \left( \frac { z _ { j } - z _ { 1 } } { \sqrt { \frac { 2 ( 1 - \eta ) } { k - 3 } } } \right) .
$$

Since $\rho _ { 1 } \geq \rho _ { j }$ , we have $z _ { 1 } \geq z _ { j }$ . By the mean value theorem, there exists c between $\rho _ { 1 }$ and $\rho _ { j }$ such that

$$
z _ { j } - z _ { 1 } = ( \rho _ { j } - \rho _ { 1 } ) \frac { 1 } { 1 - c ^ { 2 } } .
$$

Using $| \rho _ { j } - \rho _ { 1 } | \le \Delta$ and $1 - c ^ { 2 } \leq 1 - \rho _ { 1 } ^ { 2 }$ , we obtain

$$
z _ { j } - z _ { 1 } \ge - \frac { \Delta } { 1 - \rho _ { 1 } ^ { 2 } } .
$$

Substituting this bound yields

$$
\begin{array} { r l } & { \operatorname* { P r } ( \rho _ { j } ^ { \prime } > \rho _ { 1 } ^ { \prime } ) \gtrsim \Phi \left( - \frac { \Delta } { ( 1 - \rho _ { 1 } ^ { 2 } ) \sqrt { \frac { 2 ( 1 - \eta ) } { k - 3 } } } \right) } \\ & { \qquad = 1 - \Phi \left( \frac { \Delta } { ( 1 - \rho _ { 1 } ^ { 2 } ) \sqrt { \frac { 2 ( 1 - \eta ) } { k - 3 } } } \right) } \\ & { \qquad \gtrsim 1 - \Phi \left( \frac { \Delta } { ( 1 - \rho _ { 1 } ^ { 2 } ) \sqrt { \frac { 2 ( 1 - \eta _ { m a x } ) } { k - 3 } } } \right) , } \end{array}
$$

which completes the proof.

This result shows that after random projection, embeddings that are sufficiently close in the original space can overtake the original top-1 embedding with non-negligible probability. This behavior is desirable for privacy preservation: sensitive tokens (e.g., names, addresses, or identification numbers) often have embeddings close to semantically related but less sensitive alternatives, and random projection allows these alternatives to replace the original sensitive tokens with controlled probability. Combined with the bounds in Section B, this demonstrates that random projection limits distortion to a reasonable range, preserving general semantic content while reducing the likelihood that specific sensitive tokens are deterministically selected.

Algorithm 2 Private RAG with Random Projection (KNN-LM)   
Require: Datastore D with n documents; tokenizer and embedding function $f ( \cdot ) : \mathrm { T e x t } \to \mathbb { R } ^ { d } ; \lambda = 0 . 2 5 .$   
Input: User query q   
Output: Generated answer a   
Parameters: Projection dimension k; maximum token length $T _ { \mathrm { m a x } }$   
Offline preprocessing   
1: Sample random projection matrix $R \in \mathbb { R } ^ { k \times d } .$ , where $R _ { i j } \sim \mathcal { N } ( 0 , 1 / k )$   
2: Compute datastore embeddings $E _ { D } \in \mathbb { R } ^ { d \times n }$ , where $e _ { i } = f ( d _ { i } )$   
3: Project embeddings: $\tilde { E } _ { D } = R E _ { D }$   
Online inference (per query)   
4: Initialize $t \gets 0 , a \gets \emptyset , y \gets \emptyset$   
5: while $t < T _ { \operatorname* { m a x } }$ and y is not the end token do   
6: Compute query embedding: $\mathbf { \boldsymbol { x } } = \boldsymbol { f } ( \boldsymbol { q } )$   
7: Project query embedding: x˜ = Rx   
8: $P ( y \mid x ) = \lambda P _ { L M } ( x ) + ( 1 - \lambda ) P _ { K N N } ( \tilde { \pmb { x } } , \tilde { E } _ { D } )$   
9: Predict next token: y ← arg max<sub>y</sub> $, P ( y \mid x )$   
10: Append token: $a \gets a \| y , q \gets q \| y$   
11: $t \stackrel { - } {  } t + 1$   
12: end while   
13: return a

## D Complete Algorithms

For KNN-LM (Algorithm 2), at each generation step we apply random projection to both the query embedding and the datastore embeddings. The language model produces a next-token distribution $P _ { \mathrm { L M } }$ based on the current context x. In parallel, the KNN distribution $P _ { \mathrm { K N N } }$ is computed by measuring the cosine similarity between the projected query embedding x˜ and each projected datastore embedding ${ \tilde { e } } _ { i } ,$ followed by a softmax over the top-K nearest neighbors to obtain a probability distribution over candidate tokens. The final next-token distribution is obtained by linearly combining the two distributions with weight $\lambda = 0 . 2 5$ . The selected token y is then appended to the context and the generated answer, and the process repeats.

Algorithm 3 Implementation of DP-RP-G (KNN-LM)   
Require: Datastore $D = \{ d _ { i } \} _ { i = 1 } ^ { n } ;$ embedding function $f : \mathrm { T e x t } \to \mathbb { R } ^ { d }$   
Input: User query q   
Output: Generated answer a   
Parameters: Projection dimension k; max tokens $T _ { \mathrm { m a x } } ;$ clipping bound c; privacy $( \epsilon , \delta )$   
Offline preprocessing   
1: Sample random projection $R \in \mathbb { R } ^ { k \times d } .$ , where $R _ { i j } \sim \mathcal { N } ( 0 , 1 / k )$   
2: Compute datastore embeddings $E _ { D } \in \mathbb { R } ^ { d \times n }$ , with columns $e _ { i } = f ( d _ { i } )$   
3: Project: $\tilde { E } _ { D } = R E _ { D } \in \mathbb { R } ^ { k \times n }$   
4: for each column e˜<sub>i</sub> of $\tilde { E } _ { D }$ do   
5: $\begin{array} { r } { \hat { e } _ { i } \gets \tilde { e } _ { i } \cdot \operatorname* { m i n } \left( 1 , \frac { c } { \lVert \tilde { e } _ { i } \rVert _ { 2 } } \right) } \end{array}$   
6: end for   
Online inference (per query)   
7: Initialize $\begin{array} { r } { t \gets 0 , a \gets \emptyset , y \gets \emptyset , \sigma = \frac { c \sqrt { 2 \ln ( 1 . 2 5 / \delta ) } } { \epsilon } } \end{array}$   
8: Sample IID $Z \in \mathbb { R } ^ { k \times n }$ , where $Z _ { i j } \sim \mathcal { N } ( 0 , \stackrel { \circ } { \sigma } ^ { 2 } )$   
9: $\tilde { E } _ { D } \gets \tilde { E } _ { D } + Z$   
10: while $t < T _ { \mathrm { m a x } }$ and y is not the end token do   
11: $\pmb { x }  f ( q ) \in \mathbb { R } ^ { d }$   
12: $\tilde { \pmb { x } }  R \pmb { x }$   
13: y ← arg max<sub>y</sub> $P ( y \mid x )$   
14: $a \gets a \| y , q \gets q \| y$   
15: t ← t + 1   
16: end while   
17: return a

We implement DP-RP-G in Algorithm 3, following Blocki et al. [17], which adds Gaussian noise after random projection to achieve differential privacy. Unlike Blocki et al. [17], who assume binary or [0, 1]-bounded vectors, which does not hold for LLM embeddings. Therefore, after projection, we apply standard DP clipping to any vector v:

$$
\mathrm { c l i p } ( \pmb { v } , c ) = \pmb { v } \cdot \mathrm { m i n } \bigg ( 1 , \frac { c } { \| \pmb { v } \| _ { 2 } } \bigg ) ,
$$

so that all vectors have $\ell _ { 2 }$ -norm at most c. After clipping, we add Gaussian noise calibrated to the privacy budget where the variance $\begin{array} { r } { \sigma = c \frac { \sqrt { 2 \ln ( 1 . 2 5 / \delta ) } } { \epsilon } } \end{array}$ . As our goal is to protect the datastore, we apply the DP step only to $E _ { D }$ (and not to the queries). Hence, lines 12–19 in Algorithm 3 are the same as lines 7–15 in Algorithm 2. For the DP-RP-G method implemented in standard RAG, we replace lines 12–19 with lines 7–13 in Algorithm 1.

Table 10: Comparison of KNN-LM and Standard RAG on Enron Email and ECHR with different m (ϵ = 5).
<table><tr><td rowspan="3">m</td><td colspan="6">KNN-LM</td><td colspan="6">Standard RAG</td></tr><tr><td>10</td><td>20</td><td>30</td><td>40</td><td>50</td><td>60</td><td>10</td><td>20</td><td>30</td><td>40</td><td>50</td><td>60</td></tr><tr><td>Enron Email</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Privacy (# leakage)</td><td>4</td><td>2</td><td>0</td><td>0</td><td>0</td><td>0</td><td>8</td><td>6</td><td>1</td><td>0</td><td>0</td><td>0</td></tr><tr><td>Perplexity</td><td>3.14</td><td>3.15</td><td>3.14</td><td>3.14</td><td>3.16</td><td>3.16</td><td>3.70</td><td>3.68</td><td>3.70</td><td>3.71</td><td>3.71</td><td>3.98</td></tr><tr><td>ECHR</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Privacy (% leakage)</td><td>2.12</td><td>2.24</td><td>2.06</td><td>2.12</td><td>2.18</td><td>1.94</td><td>2.31</td><td>2.02</td><td>2.16</td><td>2.08</td><td>2.12</td><td>2.14</td></tr><tr><td>Perplexity</td><td></td><td>512.64512.64512.64</td><td></td><td>112.35</td><td>109.64110.35</td><td></td><td></td><td>4307.364888.23 5155.85</td><td></td><td>4307.367549.477551.63</td><td></td><td></td></tr></table>

## E Implementation Details

First, we use the Enron Email dataset [29] as the datastore and GPT-2 [30] as the LLM because GPT-2’s pre-training data do not overlap with Enron Email, ensuring the RAG data remain unseen. Thus, private information is limited to the RAG documents. The dataset contains annotated sensitive information (URLs, email addresses, phone numbers), and we use the entire dataset as the datastore, totaling 4576 sensitive items. We use the query prompts from [28] (e.g., You may also contact us via e-mail at) and evaluate whether the annotated personally identifiable information can be retrieved.

The European Court of Human Rights (ECHR) dataset [41] contains legal case documents from the ECHR. Queries come from test-case descriptions, and utility is measured by matching generated judgments to ground-truth labels. We use Llama3-1B-Instruct. For adversarial queries, we use Llama3-8B-Instruct to annotate sensitive information and query with prompts concatenating "This is the first sentence of a paragraph from ECHR legal documents. Complete the paragraph with the sentence:" and the first sentence of each case. We check if generations contain the annotated sensitive identity, using 10K documents.

Besides targeted leakage where we check if exact annotated information is leaked, we use the C4 dataset, a cleaned and large-scale version of Common Crawl, for untargeted leakage evaluation. We select 5,000 samples and use the first sentence of each as the query. We then compute ROUGE-L scores between each generation and every datastore sample. If any sample in the datastore has a score greater than 0.5 with the generation, where we follow the same settings by Huang et al. [28], marking it as a leakage. We report the number of leakages out of the 5k samples. We use the LLAMA3-8B-INSTRUCT model as the backbone.

HOTPOTQA [31] is a multi-hop QA dataset with strong supervision for supporting facts, enabling more explainable systems. We use it to evaluate whether we retrieve correct supporting documents and measure cosine similarity between retrieved and reference embeddings to assess retrieval quality. We use the training set as the datastore and the test set for evaluation. The datastore contains over 90K documents. We evaluate with distractor setting which means the whole datastore is for reference. We use QWEN2-7B-INSTRUCT as the backbone model. Since document retrieval relies on KNN for both KNN-LM and standard RAG, we focus on the KNN-LM structure to analyze the retrieval process. For evaluation, we report exact match (EM) and F1 scores between the generated output and the ground truth. As the datastore is very large, we randomly select 100 documents as targets for targeted attacks. The initial query in multi-round attack is each question.

## F Settings of Hyper-Parameters

## F.1 General Settings

For all LLM generations, we set the repetition penalty to 0.75 and the no-repeat-ngram-size to 0. The feature dimensions of GPT-2, Llama3, and Qwen2 are 128, 2048, and 3584 respectively. All other generation parameters follow the default

HUGGINGFACE settings. By default, we set the datastore size equal to the total number of tokens in each dataset. We follow previous differential privacy work where we set δ as the reverse scale of dataset size. For Enron Email and ECHR, it is $1 0 ^ { - 4 }$ . For HotPotQA and C4, we use $1 0 ^ { - 5 }$ . We use each model’s tokenizer and its last hidden-state vectors as embeddings for both retrieval settings. The max output token length is 2048.

## F.2 DPSparseRAG

Apart from ϵ and clipping bound we tune on in the experiments, another hyper-parameter for DPSparseRAG is m.   
We therefore conduct a sweep over $m \in \{ 1 0 , 2 0 , 3 0 , 4 \bar { 0 } , 5 0 , 6 0 \}$ with ϵ = 5 on the Enron Email and ECHR datasets.   
Overall, we observe in Table 10 no significant difference in utility across various values of m.

Table 11: Perplexity and data leakage comparison using random projection matrices drawn from different distributions on the Enron Email dataset with GPT2.
<table><tr><td></td><td colspan="4">KNN-LM</td><td colspan="4">Standard RAG</td></tr><tr><td></td><td>Perplexity Email URL Phone Perplexity Email URL Phone</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>RAG</td><td>2.87</td><td>0</td><td>13</td><td>0</td><td>1.58</td><td>36</td><td>56</td><td>125</td></tr><tr><td>Gaussian RP (Ours)</td><td>2.89</td><td>0</td><td>0</td><td>0</td><td>2.05</td><td>0</td><td>3</td><td>0</td></tr><tr><td>Rademacher 2.92</td><td></td><td>0</td><td>2</td><td>0</td><td>1.76</td><td>0</td><td>3</td><td>1</td></tr></table>

## G Random Projection with Other Distributions

Beyond the normal (Gaussian) distribution, we also experimented with constructing the projection matrix using a Rademacher distribution. In this distribution, each entry independently takes the value +1 or −1 with equal probability (50%). As shown in Table 11, the Rademacher-based projection also preserves privacy effectively while achieving performance comparable to vanilla RAG. These results indicate that our proposed random projection method remains effective as long as the projection matrix preserves similarity structure, regardless of the specific distribution.

## H More Discussion

## H.1 False Positive Classification of Non-Sensitive Information

Although random projection provides strong privacy protection overall, our experiments on Enron Email, ECHR, and C4 reveal that a small number of items may still be leaked. All failure cases fall into the case of leakage examples without sensitive information. Analysis of these failure cases reveals a clear pattern: random projection provides complete protection (100% leakage prevention) for specific content such as numbers, IDs, and email addresses.

While, in rare cases, common terminology which are non-sensitive information may mis-classified personal identity information in annotation (e.g. common streat name Queen St.) This occurs because such terms may appear in multiple contexts, making them less distinctive in the embedding space. These findings suggest that random projection is most effective for protecting unique identifiers while providing strong but not absolute protection for common terms that may appear across multiple documents.

Table 12: Example responses generated by different methods given the prompt.
<table><tr><td>Prompt</td><td>Complete the sentences: you have any suggestions, please share. With the rise of social media...</td></tr><tr><td>Parametric</td><td>With that in mind here are a few ideas to get you started: 1. **Create a vision board**: Cut out images and words that represent your goals and desires and create a visual representation of what you want to achieve. 2. **Set SMART goals**: Make sure your goals are Specific Measurable Relevant and Time-bound. This will help you stay focused and motivated. 3. **Break down big goals into smaller.</td></tr><tr><td>RAG</td><td>it&#x27;s become increasingly important for businesses to have a strong online presence. One way to do this is by creating engaging content that resonates with your target audience. Here are some tips to help you create content that drives engagement: 1. Know your audience: Before you start creating content make sure you have a good understanding of your target! audience. Who are they? What are their interests? What problems do they need?</td></tr><tr><td>RP (Ours)</td><td>it&#x27;s become increasingly important for businesses to have a strong online presence. One way to do this is by creating engaging content that resonates with your target audience. Here are some tips to help you create content that drives engagement: 1. Know your audience: Before you start creating content make sure you have a good understanding of who your target! audience is. What are their interests need?</td></tr><tr><td>DP-RP-G (€ = 20)</td><td>PrintWriter.PrintWriter userService userServiceExchangeExchangeExchange_generate_generate</td></tr><tr><td>20)</td><td>DPSparseVoteRAG (€ = make sure you have a good understanding of your target! &quot;#&quot;&quot;#&#x27;!$&quot;&quot;&amp;&quot;&quot;$&quot;#&quot;#&#x27;</td></tr></table>

## H.2 An Example of Generation

Table 12 provides an example of responses generated by different methods for the same prompt. The parametric model produces unrelated continuation that is not grounded in the datastore. In contrast, both RAG and our method generate clear, relevant answers, whereas the DP-based methods often produce incoherent or unrelated content.