# Same Story, Diferent Lens: A Multi-Agent System for Cross-Outlet News Framing Analysis

Afan Yasin<sup>1</sup>, Yueqi Lyu<sup>1</sup>, Rubia Fatima<sup>2</sup>, Weinchen Zhong<sup>1</sup>, Keyi Li<sup>2</sup>

<sup>1</sup> Academy of AI and Advanced Technology (AIAT), Xi’an Jiaotong-Liverpool University (XJTLU), Suzhou, China

2 Department of Computer Science and Technologies, Emerson University, Multan, Pakistan.

## Abstract

News outlets covering the same event difer in what they emphasize and omit. Same Story, Diferent Lens is an experimental research prototype for comparing these choices. It collects world and politics coverage from 17 outlets, groups reports about the same event, and produces structured analyses of sentiment, political lean, framing, loaded language, possible omissions, and model-estimated proportions of factual claims and opinion. Loaded-language entries include quotations and neutral rewordings. Three model roles handle article analysis, assessment of source support, and cross-outlet briefing generation. Analyses are stored in a JSONL archive. The evaluation compares outlet-level lean estimates with approximate, predefined reference ratings and summarizes the Critic’s model-assigned grounding scores. These measures provide preliminary diagnostics. A grounding score reflects the model’s assessment of support in the supplied text and does not establish factual accuracy or constitute independent verification.

Keywords: Multi-Agent Systems, News Framing Analysis, Large Language Models, Cross-Outlet News Analysis, Automated Source-Grounding Assessment.

## 1 Introduction

News framing concerns the selection and emphasis that shape a report of an event, as described by Entman [1]. Comparing framing across outlets usually requires readers to examine several reports and identify diferences in wording, emphasis, and missing context. Such comparisons are dificult to scale, particularly when readers need to trace an interpretation to the words that support it.

Same Story, Diferent Lens groups reports about the same event and uses a language model to analyze each report. The prompt asks the model to support loaded-language judgments with quotations and framing judgments with specific wording. Readers can inspect these passages alongside the analysis.

The system provides:

1. A working pipeline for news ingestion, story clustering, framing analysis, and presentation. A built-in mock provider supports end-to-end demonstrations without paid credentials.

2. Separate Analyst, Critic, and Reporter roles. The Analyst generates structured annotations, the Critic assesses their support in the article, and the Reporter prepares a cross-outlet briefing.

3. An archive and evaluation tools for examining outlet-level lean estimates and model-assigned grounding scores. The reproducibility design also specifies prompt tracking, schema versions, and field documentation.

The reported evaluation covers 1,330 article analyses across 15 outlets. These results are preliminary. The reference ratings are approximate, and the model’s annotations require review. Complete historical metadata are unavailable, which limits reconstruction of the reported evaluation. Figure 1 shows the event-level comparison interface.

![](images/f78e3cf388663adb67d2709ebddbe9fa9b731666b24bec4c284452b421c6c61c.jpg)  
Figure 1: Event-level comparison of reports about the same story.

## 2 Related work

Framing and media slant. Entman [1] described framing in terms of selection and salience. Gentzkow and Shapiro [2] found measurable slant in daily newspapers and an association with reader ideology. Hamborg et al. [3] review automated methods that examine word choice, sentence structure, and topic selection. Media Bias/Fact Check [16] and AllSides [17] publish outlet-level media bias ratings.

Comparing reports of the same event. Framing comparisons require reports to be matched by event. Headline overlap and bag-of-words methods provide interpretable baselines for this task. The POLITICS approach [10] uses pairs of articles about the same story to learn political leaning.

Our system uses event matching to organize framing comparisons. It combines TF-IDF cosine similarity, headline Jaccard overlap, and union-find merging, without a neural text encoder.

Language models as annotators and reviewers. Language models can perform text annotation and, in some settings, outperform crowd workers [4]. They are also used to evaluate other models’ outputs [12]. Their judgments have limitations: models can express political leanings [11], and greater model capability does not consistently prevent confident errors [14].

Self-refinement methods ask a model to critique and revise its output [5]. Constitutional approaches use explicit principles to constrain generation [6]. Verification methods assess support for generated statements; SelfCheckGPT, for example, uses sampling to detect possible hallucinations [13]. Our Critic assesses whether the supplied article supports the Analyst’s annotations and records a model-assigned score and field-level comments. Its score is not a deterministically calculated proportion of verified claims.

The evaluation code uses established agreement measures: Cohen’s κ [7], adjusted Rand index [8], and Fowlkes–Mallows [9]. The present snapshot lacks the human annotations needed to report these measures. The contribution is a system that combines event matching, structured framing analysis, and inspectable review outputs. We do not claim a new clustering or statistical algorithm.

![](images/af0bc4eca00f05211dc32eb4806c7f789837a6e37947e66aa1ebd4ff0c06b648.jpg)  
Figure 2: Design overview of the news-framing workflow. The Analyst generates annotations, the Critic assesses source support, and the Reporter prepares a briefing. The verified-input restriction and provenance mechanisms shown here are proposed design elements. The labels “Verified material only” and “Use verified material only” do not describe the current implementation, in which Critic assessments are diagnostic and do not filter Reporter inputs. Solid arrows show processing flows, and dashed arrows show archive and evaluation flows.

## 3 System design

The workflow covers ingestion, clustering, article analysis, assessment of source support, briefing generation, and presentation. Three roles organize the model tasks: the Analyst, Critic, and Reporter. Figure 2 shows the workflow and its archive and evaluation components.

An agent here is a task role. The reported evaluation uses a single underlying language model for these roles. The Analyst produces structured annotations, the Critic receives the article and those annotations, and the Reporter generates a briefing from story digests. This arrangement separates tasks but does not make their judgments independent.

The Analyst and Critic may share biases and failure modes. A field marked as supported records the Critic’s judgment about the supplied text; it does not establish that the article itself is factually correct. In the current implementation, the Critic’s assessments are stored as diagnostics and do not filter the Reporter’s inputs. The verified-input restriction shown in the design requires an explicit filtering mechanism.

The implementation is a TypeScript monorepo. An Express backend runs the pipeline and serves a JSON API, and a React frontend uses Vite and Tailwind. Clustering, evaluation metrics, and statistical calculations are written in TypeScript without heavy machine-learning dependencies.

![](images/9b4c624470664ab8de13b2e8a7fb7cfe96ed78bd28cb85955c3ae4f48639a775.jpg)  
Figure 3: The outlet directory and its display metadata. The screenshot’s “Lean” column includes both political-lean descriptions and institutional relationships, which are distinct attributes. Directory labels and the evaluation reference values in Table 1 should be distinguished.

## 3.1 Corpus and ingestion

The collection configuration covers 17 outlets through 19 oficial RSS feeds (Table 1). Sky News and NPR each contribute world and politics feeds; the other feeds cover world news. The selection includes public-service broadcasters such as NPR and France 24, commercial papers such as Indian Express, news agencies such as Yonhap, and state-aligned media including TASS, Global Times, IRNA, Mehr News, Anadolu, and ECNS. Figure 3 shows the outlet directory.

When an article body cannot be retrieved, the system can use the feed summary and record a body\_incomplete flag. Feed failures can also reduce the sources available for comparison. The requirement for at least two outlets in a story does not remove this source-selection risk. Paywall and robots.txt compliance should be checked independently; the current ingestion implementation does not establish a general guarantee of compliance or reliable retention of the last successful crawl after every feed failure.

<table><tr><td>Outlet</td><td>Type</td><td>Reference lean</td></tr><tr><td>Sky News, France 24, The News, Doha News The National, Daily News Egypt, AllAfrica, Yonhap commercial</td><td>broadcast/</td><td>center</td></tr><tr><td>NPR, Al-Monitor, Indian Express</td><td>public / independent lean-left</td><td></td></tr><tr><td>Anadolu, ECNS</td><td>state agency</td><td>lean-right</td></tr><tr><td>TASS, Global Times, IRNA, Mehr News</td><td>state media</td><td>right</td></tr></table>

Table 1: The 17 configured outlets and their approximate, predefined reference ratings. These values are exploratory comparators, not independently validated ground truth; see Section 4.2.

![](images/2dccb40fbdbe0aa3c630f1ea7d29c0db40953eae9e878dc51b7379cce6d02b95.jpg)  
Figure 4: Single-source reports retained separately when no second outlet is matched.

## 3.2 Story clustering

Each article has a TF-IDF representation of headline and summary unigrams and bigrams, with document frequencies calculated over the current crawl. Headline token sets provide a second similarity measure through Jaccard overlap. A pair of articles is linked when cosine similarity reaches 0.25 or headline Jaccard similarity reaches 0.55, provided the publication dates are no more than four days apart. Same-outlet pairs use higher thresholds of 0.45 and 0.70. Union-find merging converts these links into connected components. The four-day rule applies to article pairs, so a component formed through several links can span a longer interval.

A component becomes a story when it contains at least two distinct outlets. For a single-outlet component, the most recent report is retained in a separate list (Figure 4). Stories are ranked first by the number of outlets, then by recency, and then by article count. The service retains the top 36 stories for presentation; the homepage initially displays 12. Analysis and archiving operate on the retained selection, so the archive does not contain every ingested article.

A fixed outlet-preference order selects the headline used as the topic label. This provides a consistent selection rule without establishing that the selected headline is neutral. Across crawls, URL and content overlap link a story to an earlier instance so that follow-up reports can share an identifier. These links are heuristic and do not guarantee correct event identity.

## 3.3 Per-article framing analysis (the Analyst)

The Analyst prompt asks for quotations to support analytical claims. When evidence is insuficient, the prompt instructs the model to leave the field empty and record an insufficient\_evidence\_ flags entry. These are prompt requirements; they do not guarantee that every returned annotation has valid support. The output schema contains eight fields:

• sentiment: a score in [−1, 1] and a five-way label from negative to positive;

• bias\_rating: a model-assigned political-lean label from left to right, or not-determinable, with a model-reported confidence value in [0, 1];

• framing: a one-sentence description of the article’s angle;

• loaded\_language: up to five quotations, each with an explanation and a model-proposed neutral rewording;

• omissions: possible missing context identified by the model;

• fact\_vs\_opinion: model-estimated percentages of factual claims and opinion;

• tone\_keywords: up to five adjectives;

• insufficient\_evidence\_flags: fields for which support was insuficient.

The prompt defines bias in relation to other coverage and asks the model to describe wording and possible omissions without accusing an outlet of lying. However, the per-article Analyst does not receive the other outlets’ full articles. Its lean and omission annotations therefore cannot be treated as direct measurements of diferences established by reading all reports together.

System instructions and the JSON schema are separate from article text. This separation would allow the methodology to be fingerprinted independently of the input, as discussed in Section 4.1. A separate cluster-level comparison summarizes agreements and diferences and identifies reports with relatively restrained or emotive wording.

The system reuses analyses through a URL-based cache and limits concurrent model requests. The cache does not establish that article content or method versions are unchanged. Individual analysis failures can produce deterministic fallback outputs labeled mock; these should be distinguished from real-model outputs in evaluation.

## 3.4 Critic assessment and briefing synthesis

The Critic receives the article text and the Analyst’s output. It returns a verification\_score in [0, 1], lists of supported and unsupported fields, and suggested corrections. The model supplies the score directly. The program does not derive it from a fixed count of supported fields, and the field lists do not determine its value. The Critic does not receive human annotations. We use its mean score as an automated diagnostic of assessed source support.

The Reporter receives story digests containing topics, outlet counts, sentiment and lean summaries, claims shared across reports, and coverage diferences. Its prompt requests a headline, a plain-language summary, one to four coverage divergences, and implications. The requested number of divergences is a prompt instruction rather than a strictly enforced output constraint. The current Reporter does not use the Critic’s score or field lists to filter its inputs.

Briefings run on a configurable schedule, with a default interval of five hours, rather than after every crawl. This limits repeated model requests but does not guarantee a successful briefing at every scheduled time. Figure 5 shows the newsroom view.

![](images/1024861337e51dc4c77eba66738d7a98cbc6286560aee7f80d740af53a2a96a7.jpg)  
Figure 5: The newsroom view, with a briefing and Critic statistics from the live store. This illustrative screenshot does not represent the archived evaluation subset.

## 3.5 Presentation layer

The main feed lists stories with sentiment summaries for each outlet. A story-detail page places reports side by side and provides a divergence report, a model-generated ordering of reports from less to more emotive wording, and an agreement matrix. Readers can expand loaded-language entries to inspect the quoted passages. Figures 6 and Figure 10 (Appendix A) show the feed and story-detail views.

The newsroom page presents briefings and live-store Critic statistics. A separate evaluation page displays archive-based measures. These views use diferent data scopes, so their displayed counts and averages need not match.

![](images/de1b14466c4246b936aac242d845621be0394df8a5c46261745c491cf15ac067.jpg)  
Figure 6: The main feed ranks stories by outlet coverage and shows sentiment summaries for each outlet.

## 4 Reproducibility and evaluation

## 4.1 Archive and provenance

The JSONL archive stores analysis and verification outputs for later evaluation. The reproducibility design specifies three additional mechanisms:

1. Prompt fingerprinting would identify system instructions and the schema with a SHA-256 hash and record the fingerprint and temperature for each analysis. Content-addressed snapshots and an index would preserve the corresponding prompts.

2. Schema versioning would record a semantic schema\_version, define how readers handle missing versions, and document changes in a changelog. The proposed default for a missing version is 1.0.0.

3. A codebook would document archive, store, and CSV fields, permitted labels, and flags.

These mechanisms describe the provenance design. Their implementation and historical use are not established for the reported snapshot. Missing metadata cannot be reconstructed merely by assigning a version or hashing the current prompt. Full reconstruction of an earlier run requires its actual inputs, method version, and selection records.

## 4.2 Evaluation harness

The TypeScript evaluation code implements the following measures:

1. Outlet-level lean correlation: means derived from the lean labels on [−1, 1] are compared with predefined reference ratings using Pearson r and Spearman $\rho .$

2. Inter-annotator agreement: Cohen’s κ for sentiment and lean when two sets of human annotations are available, together with model–human agreement.

3. Topic-clustering agreement: adjusted Rand index and Fowlkes–Mallows between human true\_topic labels and cluster assignments.

4. Mean Critic grounding score: the arithmetic mean of eligible verification\_score values.

The reference ratings are approximate comparison values. Their presence in an evaluation file does not establish their provenance or independent validity, and it does not show that they were supplied to the Analyst or Critic for calibration.

A real-model evaluation requires explicit provider, model, duplicate, missing-label, and minimumsample rules. The current default harness does not by itself enforce the reported exclusion of mock analyses or the two-article threshold. It also rounds outlet means before computing correlations. Full-precision means should be used for a reproducible statistical calculation, with rounding applied only for display.

![](images/5d80b2f8c6740eb080ffa0b805cb84faeb7b154ae906c8c9a8cca20dc290d3c0.jpg)  
Figure 7: The evaluation interface. This screenshot illustrates the view and does not reproduce the reported figures in Section 5. Human-agreement and clustering measures require annotations. The legacy labels “ground truth” and “verification rate” refer, respectively, to approximate reference ratings and the mean model-assigned Critic grounding score, not independently validated labels or a measured proportion of verified claims.

The treatment of not-determinable also needs to be stated. The implementation maps it to zero, which is not equivalent to a model judgment of center. Results can change if such labels are excluded instead. The Critic mean likewise requires a declared scored subset and a separate report of scoring coverage.

No human annotation file is available for the reported snapshot. Consequently, Cohen’s κ and the topic-clustering agreement measures remain pending. Their implementation still requires validation before results from human annotations can be reported. Figure 7 shows the evaluation interface.

## 5 Results

The original evaluation summary describes a snapshot accumulated over approximately two weeks. Its complete historical metadata and subset membership have not been recovered. We retain the reported figures below as preliminary results, rather than presenting them as an independent reproduction of the original experiment.

• The collection configuration covers 17 outlets and 19 feeds. These are configured sources, not the number of sources necessarily included in every evaluation.

• The original summary reports 3,174 archived article records and 1,330 DeepSeek analysis records. Record counts should not be interpreted as counts of unique articles or model calls. Mock and unknown-source outputs require separate accounting; the remaining records cannot all be assumed to be mock.

• The reported lean-correlation subset contains 15 outlets, each described as having at least two real-model analyses. Its exact selection has not been reconstructed.

• The reported Critic subset contains 856 scored real-model analyses. The score mean refers to these records, not all 1,330 analyses. The reasons for missing scores have not been established.

Lean correlation. Table 2 reproduces the reported outlet means and reference ratings. The original summary gives Pearson $r = 0 . 7 6$ and Spearman $\rho = 0 . 6 5$ across 15 outlets. TASS and Mehr News have positive model means, while Indian Express is close to the center despite its lean-left reference rating. These comparisons describe the displayed values and do not validate the underlying reference ratings.

The displayed article counts sum to 697, leaving 633 records unaccounted for relative to the reported 1,330 analyses. Without the original selection records, this diference cannot be assigned to a particular filter or deduplication rule. Full-precision inputs are also needed to reconstruct the reported correlations; the rounded table alone does not establish their original calculation.

Critic grounding score. The reported mean over 856 scored analyses is 0.52. This is an average of model-assigned scores. It does not mean that half of all analytical claims were independently verified. Supported and unsupported field lists can help identify outputs for review, but the historical field-level counts are insuficient to support a general ranking of which fields were most reliably grounded.

Human agreement. No annotation file accompanies the reported snapshot. Cohen’s κ and topic-clustering agreement therefore remain pending.

The limited observation window and approximate reference ratings restrict interpretation. The results do not establish an outlet ranking or the accuracy of individual article labels. Figures 8 and 9 illustrate the archive views; their screenshots are not additional evidence for the reported correlation and grounding scores.

![](images/89b780ba3daadb407a555c44c6ebf1e151d903509397966a7e5e9bbd00322cf6.jpg)

Figure 8: The Insights view, showing article-volume trends, coverage by outlet, and trending topics.
<table><tr><td>Outlet</td><td>Reference lean</td><td>Model lean</td><td>n</td></tr><tr><td>Al-Monitor</td><td>-0.50</td><td>-0.03</td><td>123</td></tr><tr><td>Anadolu</td><td>+0.50</td><td>+0.00</td><td>34</td></tr><tr><td>Daily News Egypt</td><td>0.00</td><td>+0.03</td><td>18</td></tr><tr><td>ECNS</td><td>+0.50</td><td>+0.10</td><td>10</td></tr><tr><td>France 24</td><td>0.00</td><td>-0.07</td><td>81</td></tr><tr><td>Indian Express</td><td>-0.50</td><td>+0.03</td><td>34</td></tr><tr><td>IRNA</td><td>+1.00</td><td>+0.25</td><td>2</td></tr><tr><td>Mehr News</td><td>+1.00</td><td>+0.29</td><td>29</td></tr><tr><td>NPR</td><td>-0.50</td><td>-0.28</td><td>23</td></tr><tr><td>Sky News</td><td>0.00</td><td>-0.33</td><td>3</td></tr><tr><td>TASS</td><td>+1.00</td><td>+0.50</td><td>98</td></tr><tr><td>The National</td><td>0.00</td><td>+0.06</td><td>172</td></tr><tr><td>Yonhap</td><td>0.00</td><td>+0.00</td><td>55</td></tr><tr><td>AllAfrica</td><td>0.00</td><td>-0.09</td><td>11</td></tr><tr><td>Doha News</td><td>0.00</td><td>-0.13</td><td>4</td></tr><tr><td colspan="4">Reported correlations: Pearson r = 0.76, Spearman  $\rho = 0 . 6 5 ;$  15 outlets</td></tr></table>

Table 2: Reported outlet-level model means and reference ratings on [−1, 1], with negative values indicating left lean. n is the reported number of included analyses. Each outlet contributes one observation to the unweighted correlation; article counts are not correlation weights. Means are displayed to two decimal places. The full historical selection and precision are unavailable.

## 6 Limitations and ethics

Limitations. The reported evaluation covers fifteen outlets, several represented by only a few articles, and a single underlying model. The reference ratings are approximate and predefined. Their sources and assignment procedures require further verification. Correlations describe association and relative ordering, not absolute agreement in lean scores. The Analyst and Critic may share systematic biases and errors. A Critic score records the model’s assessment of support in the supplied text; it does not establish factual correctness or independent verification. Incomplete historical metadata limits reconstruction of earlier evaluations, and repeated records and unspecified selection rules can afect the reported means. Lightweight lexical clustering can separate reports of the same event when their wording difers or merge distinct events when their wording is similar.

![](images/6c07987c3a052784402350a83662714ac3792f7589aa439ae6880d86bc277c88.jpg)  
Figure 9: The dashboard, showing archive counts, outlet coverage, current story-cluster counts, and sentiment summaries. The analyzed count includes records with an analysis object, including mock outputs.

Ethics. The interface presents quoted passages and possible omissions so that readers can examine the basis of an interpretation. It treats lean as relative to other coverage and identifies the reference ratings as approximate. Such tools can still be used to make unsupported accusations of partisanship. Quotation access and instructions against accusations of lying are intended to support review; they do not ensure that every output is defensible. Related work suggests that carefully designed AI interactions can reduce unfounded beliefs [15], but that finding does not establish the efect of this prototype.

The archive is released under a non-commercial research license (CC-BY-NC for the data, with the code under a corresponding non-commercial license).

Prototype status. Summaries and framing analyses are generated automatically, and aggregate statistics are computed programmatically. The original articles come from external publishers. Outputs may contain errors, omissions, or biased interpretations and should be checked against the linked sources. This is an experimental research prototype, with additional functionality under development. Future work will extend the existing evaluation harness through stronger provenance tracking, broader testing, and independent human assessment.

## 7 Conclusion

Same Story, Diferent Lens brings event clustering, structured framing annotations, Critic diagnostics, and cross-outlet briefings into one prototype. Readers can inspect quotations and intermediate

outputs alongside the generated analysis.

The reported outlet correlation (r = 0.76) and mean Critic score (0.52) are preliminary. The original evaluation requires further work on snapshot identity, subset selection, and reference-rating provenance. The Critic’s score measures model-assessed source support, and the current workflow does not restrict Reporter inputs to Critic-approved material. Further evaluation needs independent human annotations and a fully traceable archive.

## Availability

The source code and data schema are available at Github (https://github.com/affanyasin1/ multi-perspective-analyzer)<sup>1</sup>.

## Acknowledgments.

This work benefited from AI-assisted tools for language editing, improving the clarity and readability of the manuscript, and supporting code development.

## References

[1] R. M. Entman. Framing: Toward clarification of a fractured paradigm. Journal of Communication, 43(4):51–58, 1993.

[2] M. Gentzkow and J. M. Shapiro. What drives media slant? Evidence from US daily newspapers. Econometrica, 78(1):35–71, 2010.

[3] F. Hamborg, K. Donnay, and B. Gipp. Automated identification of media bias in news articles: An interdisciplinary literature review. International Journal on Digital Libraries, 20(4):391–415, 2019.

[4] F. Gilardi, M. Alizadeh, and M. Kubli. ChatGPT outperforms crowd workers for text-annotation tasks. Proceedings of the National Academy of Sciences, 120(30):e2305016120, 2023.

[5] A. Madaan, N. Tandon, P. Gupta, S. Hallinan, L. Gao, S. Wiegrefe, U. Alon, N. Dziri, S. Prabhumoye, Y. Yang, et al. Self-Refine: Iterative refinement with self-feedback. In Advances in Neural Information Processing Systems (NeurIPS), 2023.

[6] Y. Bai, S. Kadavath, S. Kundu, A. Askell, J. Kernion, A. Jones, A. Chen, A. Goldie, A. Mirhoseini, C. McKinnon, et al. Constitutional AI: Harmlessness from AI feedback. arXiv preprint arXiv:2212.08073, 2022.

[7] J. Cohen. A coeficient of agreement for nominal scales. Educational and Psychological Measurement, 20(1):37–46, 1960.

[8] L. Hubert and P. Arabie. Comparing partitions. Journal of Classification, 2(1):193–218, 1985.

[9] E. B. Fowlkes and C. L. Mallows. A method for comparing two hierarchical clusterings. Journal of the American Statistical Association, 78(383):553–569, 1983.

[10] Y. Liu, X. F. Zhang, D. Wegsman, N. Beauchamp, and L. Wang. POLITICS: Pretraining with Same-story Article Comparison for Ideology Prediction and Stance Detection. In Findings of the Association for Computational Linguistics: NAACL 2022, pages 1354–1374, 2022. DOI: https://doi.org/10.18653/v1/2022.findings-naacl.101.

[11] F. Motoki, V. Pinho Neto, and V. Rodrigues. More human than human: Measuring ChatGPT political bias. Public Choice, 198(1–2):3–23, 2024.

[12] L. Zheng, W.-L. Chiang, Y. Sheng, S. Zhuang, Z. Wu, Y. Zhuang, Z. Lin, Z. Li, D. Li, E. P. Xing, H. Zhang, J. E. Gonzalez, and I. Stoica. Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. In Advances in Neural Information Processing Systems (NeurIPS), Datasets and Benchmarks Track, 2023.

[13] P. Manakul, A. Liusie, and M. J. F. Gales. SelfCheckGPT: Zero-resource black-box hallucination detection for generative large language models. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP), 2023.

[14] L. Zhou, W. Schellaert, F. Martínez-Plumed, Y. Moros-Daval, C. Ferri, and J. Hernández-Orallo. Larger and more instructable language models become less reliable. Nature, 634:61–68, 2024.

[15] T. H. Costello, G. Pennycook, and D. G. Rand. Durably reducing conspiracy beliefs through dialogues with AI. Science, 385(6714):eadq1814, 2024. DOI: https://doi.org/10.1126/science. adq1814.

[16] Media Bias/Fact Check. Media Bias/Fact Check: media bias and reliability ratings. https: //mediabiasfactcheck.com. (Resource; accessed 2026.)

[17] AllSides Media. AllSides Media Bias Ratings. https://www.allsides.com/media-bias. (Resource; accessed 2026.)

![](images/1e2cfa66fdaba6fb5d94c415d1b55ac926b8a17c95d6aff9e1aec83f177259f9.jpg)  
Figure 10: The story-detail view for one cluster, with a divergence report and an agreement matrix.