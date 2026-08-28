# TransMeme: A Multi-Agent Framework for Cross-Cultural Meme Transcreation

Jingyi Zheng   
Hong Kong University of Science and   
Technology (Guangzhou)   
Guangzhou, China   
Wuhan University   
Wuhan, China   
jzheng029@connect.hkust-gz.edu.cn

Tianyi Hu Aarhus University Aarhus, Denmark tenney.hu@cs.au.dk

Yule Liu Hong Kong University of Science and Technology (Guangzhou) Guangzhou, China yliu514@connect.hkust-gz.edu.cn

Yuemeng Zhao   
Hong Kong University of Science and Technology (Guangzhou) Guangzhou, China   
yzhao472@connect.hkust-gz.edu.cn Zifan Peng   
Hong Kong University of Science and Technology (Guangzhou) Guangzhou, China   
zpengao@connect.hkust-gz.edu.cn

Xinhu Zheng Hong Kong University of Science and Technology (Guangzhou) Guangzhou, China xinhuzheng@hkust-gz.edu.cn

## Abstract

Internet memes are a pervasive form of multimodal online communication; however, such communication often involves users from diverse linguistic and cultural backgrounds. Therefore, adapting memes across cultures and languages is a central challenge for enabling mutual understanding in online communication. Unlike ordinary translation or standalone text rewriting, cross-cultural meme transcreation must jointly preserve communicative intent, adapt culture-dependent meaning for the target audience, and maintain coherence between text and image. In this work, we first provide an explicit task analysis of cross-cultural meme transcreation and identify three core challenges: Culture-specific knowledge understanding, Intent and Tone Preservation, and Multimodal Consistency. Based on this analysis, we propose a multi-agent framework with specialized agents that are coordinated to address these challenges through cultural adaptation, target text rewriting, revision, and conditional visual adjustment. The framework strengthens target text adaptation with coordinated feedback to handle dificult cases that require deeper cultural or visual intervention. We evaluate the framework on bidirectional Chinese-English meme transcreation using both human evaluation and LLM-as-a-Judge. Our method consistently outperforms all baselines across both human evaluation and LLM-as-a-Judge settings. In human evaluation, it achieves the best performance on all four dimensions and delivers a 33.1% average improvement over the strongest baseline, while in LLMas-a-Judge, it attains the highest Top-1 ranking rate (60 % vs. 26%

Xinlei He<sup>✉</sup>   
Wuhan University   
Wuhan, China   
xinlei.he@whu.edu.cn

for the second-best baseline). Further analysis indicates that each component contributes to the performance, highlighting the efectiveness of the proposed architecture, and our error analysis suggests that the remaining bottlenecks lie in humor reconstruction and image-text alignment rather than simple cultural knowledge gaps, pointing to future work on humor transfer. <sup>1</sup>

## CCS Concepts

• Social and professional topics → Cultural characteristics.

## Keywords

Memes, Cultural Adaptation, Multimodal Generation

## ACM Reference Format:

Jingyi Zheng, Yule Liu, Zifan Peng, Tianyi Hu, Yuemeng Zhao, Xinhu Zheng, and Xinlei He. 2026. TransMeme: A Multi-Agent Framework for Cross-Cultural Meme Transcreation. In Proceedings ofthe 34th ACM International Conference on Multimedia (MM ’26), November 10–14, 2026, Rio de Janeiro, Brazil. ACM, New York, NY, USA, 10 pages. https://doi.org/10.1145/3767308. 3836278

## 1 Introduction

Internet memes have become a pervasive form of online communication, serving as compact carriers of humor, stance, emotion, and social commentary [1]. As a widely shared form of digital expression, they also shape how people participate in online culture [2], express political and social positions [3], and negotiate identity across communities. This makes cross-cultural meme transfer important for both multilingual digital communication and understanding how online information travels across linguistic and cultural boundaries. Yet such a transfer is far from straightforward. Unlike ordinary text, memes derive their efect from the interaction between short written text, visual templates, and shared cultural knowledge [2, 4], and their meaning is often implicit, relying on background assumptions, discourse conventions, symbolic imagery, or locally recognizable references rather than literal wording alone [4]. A successful target meme needs to do more than restate the source content in another language: it needs to reconstruct a communicative efect that works for a diferent audience.

![](images/2862d436739cda714b56982cb1ab299d4c704d7ba3f9bbb5bfbea1b621ca95d4.jpg)  
Figure 1: Overview of our multi-agent framework for cross-cultural meme transcreation.

This task is related but not reducible to several similar problems. Specifically, it shares cross-lingual transfer with machine translation, visual grounding with multimodal translation, rhetorical reformulation with text rewriting, and visual modification with image transcreation [5–8]. Recent work has introduced cross-cultural meme transcreation as an end-to-end benchmarked task [9]. However, it does not provide suficiently strong empirical performance or the necessary analysis to address the problem. So we begin with an analysis, which suggesting that meme transcreation is shaped by three challenges: Culture-specific knowledge understanding, because meme meaning often relies on local knowledge, conventions, and socially shared references; Intent and Tone Preservation, because humor, sarcasm, ridicule, and stance often need to be re-expressed rather than translated literally; and Multimodal Consistency, because text and image jointly construct the meme efect, so adapting one side may weaken or break its coherence with the other. This analysis shows that cross-cultural meme transcreation is not simply a translation, but a challenging adaptation problem governed by constraints on cultural understanding, intent preservation, and multimodal consistency.

To address these challenges, we propose TransMeme, a multiagent framework for cross-cultural meme transcreation that decomposes the task into stages following the structure revealed by our analysis. The Adapter addresses culture-specific knowledge understanding through explicit planning, deciding whether source references should be preserved, reformulated, or remapped for the target audience. The Rewriter, further strengthened with DPO, han dles intent and tone preservation by generating target text that better preserves humor, stance, and communicative force under explicit adaptation constraints. The Coordinator and Critic handle multimodal consistency by routing dificult cases to deeper adaptation, checking whether the adapted text remains compatible with the visual template, and triggering targeted revision when cultural, semantic, or visual inconsistencies arise. Together, these components enable TransMeme to preserve transferable elements, rewrite what needs adaptation, and revise dificult cases when necessary.

Table 1: Comparison between meme transcreation and related tasks. XLang: cross-lingual transfer; Text: textual adaptation; Visual: visual understanding or editing; Culture: cultural adaptation; I-T: image-text coherence.
<table><tr><td>Task</td><td>XLang</td><td>Text</td><td>Visual</td><td>Culture</td><td>I-T</td></tr><tr><td>Machine Translation</td><td>√</td><td>√</td><td>x</td><td>x</td><td>x</td></tr><tr><td>Multimodal Translation</td><td>√</td><td>√</td><td>√</td><td>x</td><td>x</td></tr><tr><td>Text Rewriting</td><td>x</td><td>√</td><td>X</td><td>x</td><td>x</td></tr><tr><td>Image Editing</td><td>x</td><td>x</td><td>√</td><td>x</td><td>x</td></tr><tr><td>Meme Transcreation</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td></tr></table>

We evaluate TransMeme on bidirectional Chinese-English meme transcreation using both human and LLM evaluation. The results show that TransMeme consistently outperforms all baselines. On human evaluation, it achieves the best scores on all four dimensions, with an overall average of 4.122, surpassing the strongest baseline at 3.097. The same advantage remains clear on the full benchmark, where our method is selected as the top-ranked system most frequently, reaching a top-1 rate of 0.600. We further conduct component analysis to examine where these gains come from. The DPO-enhanced Rewriter improves target text quality, raising the average text-level score from 2.839 without DPO to 4.180. The Critic based revision loop is efective: among the 125 cases that enter revision, the revised result is preferred in 105 cases. Explicit cultural planning proves similarly crucial. On the 181 samples that require this module, the full system is preferred over the version without cultural planning in 180 cases. Together, these results show that the gains of TransMeme come not merely from the prompt but from challenge-aligned adaptation, revision, and control. Our contributions are threefold:

• We provide an explicit task analysis of cross-cultural meme transcreation, clarifying its core requirements and identifying three central challenges: Culture-specific knowledge understanding, Intent and Tone Preservation, and Multimodal Consistency.

• We propose TransMeme, a challenge-driven multi-agent framework that integrates structured understanding, cultural planning, text rewriting with a DPO-enhanced rewriter, critic-guided revision, and conditional visual execution.

• We evaluate TransMeme through human and LLM-based evaluation, and component analyses, showing that adaptation planning and targeted revision yield consistent gains over strong one-shot and structured baselines.

## 2 Task Analysis

## 2.1 Task Requirements

Given a source meme, the task is to produce a target-culture meme for diferent audiences. The output may preserve the original image, or edit it, while the accompanying text may need to be rewritten rather than directly translated. This task is related to several established tasks (see Table 1). It involves cross-lingual transfer, as in machine translation [5], and may also use visual information, as in multimodal translation [6]. It requires textual reformulation, which connects it to text rewriting [7]. It can further involve visual adaptation, which is also studied in image transcreation [8]. At the same time, cross-cultural meme transcreation is not reducible to any of these tasks in isolation. We define three requirements specific to meme transcreation. First, intent preservation requires the target meme to retain the source meme’s core communicative efect. This includes humor, sarcasm, ridicule, irony, emotional stance, and other pragmatic functions. Second, cultural adaptation requires the result to be understandable, natural, and appropriate for the target audience. A transcreated meme should work within the target culture rather than merely restating the source content in another language. Third, image-text coherence requires the adapted text and the final image to support the same meme efect. The meme should function as a unified multimodal artifact rather than as a plausible sentence placed on an unrelated picture. This requirement is especially important for memes, since prior work has shown that meme meaning and rhetoric emerge from the interaction of verbal and visual elements rather than from either modality alone [4, 9].

## 2.2 Core Challenges

These requirements lead to three challenge categories. Each chal lenge corresponds to one major requirement of the task, although they often appear together in practice.

Culture-specific knowledge understanding. The requirement of cultural adaptation creates this challenge. Many memes rely on source culture entities, stereotypes, public events, discourse conventions, or shared background knowledge. In such cases, the text alone conveys only part of the meme’s meaning. A literal transfer may preserve wording but fail for the target audience. The system must decide whether to keep the original reference, replace it with a target-side analogue, or reframe it. This makes cultural adaptation a decision problem rather than a simple translation.

Intent and Tone Preservation. The requirement of intent preservation creates the second challenge. Memes often depend on sarcasm, irony, exaggeration, ridicule, or stance. These efects are rarely preserved by semantic transfer alone. The text can remain close in meaning and still lose its punch, tone, or humor. The system needs to reconstruct communicative efect, not just propositional content. This is why text adaptation in meme transcreation is closer to pragmatic rewriting than to ordinary translation.

Multimodal Consistency. The requirement of image-text coherence creates the third challenge. In memes, text and image usually work together to produce the final efect. The same text may succeed with one template and fail with another. The same image may support one rhetorical framing but conflict with a diferent one. Thus, local adaptation at the text level is not enough. The system must also judge whether the original image remains suitable, whether it should be edited, or whether a new realization is needed.

These challenges are tightly connected. A meme may contain a culture-specific reference, express that reference through sarcasm, and rely on a particular expression at the same time. For this reason, cross-cultural meme transcreation should not be treated as a simple generation task. It requires explicit understanding of meme meaning, deliberate planning of cultural adaptation, careful rewriting of the text, and a final check on multimodal coherence.

## 3 Method

## 3.1 System Overview

We propose a multi-agent framework for cross-cultural meme transcreation that takes a source meme image as input and generates a target meme. As shown in Figure 1, the framework consists of five agents-the Interpreter, Coordinator, Adapter, Rewriter, and Criticfollowed by the Execution Layer for final visual realization. The design directly follows the three challenge categories in Section 2. The Adapter handles the understanding of culture-specific knowledge through explicit adaptation planning. The Rewriter, which is further optimized with DPO for meme text adaptation, handles intent and tone preservation through target text rewriting. The Coordinator and Critic address Multimodal Consistency by checking whether the adapted text fits the visual template and by triggering revision when needed. The Interpreter provides the source meme analysis for subsequent decisions, while the Transcreation Harness maintains the intermediate pipeline state and supports execution.

## 3.2 Interpreter and Coordinator: Understanding and Control

The framework begins with source meme understanding and workflow control. The Interpreter recovers the source meme’s communicative efect beyond surface text and writes the result into the Transcreation Harness. Its analysis has two parts. First, it infers the meme’s semantic and pragmatic content, including core intent, humor mechanism, emotional stance, relevant background knowledge, and source side elements that should ideally be preserved. Second, it assesses image and text dependency, including whether the intended efect relies on visual entities, facial expression, symbolic cues, or scene logic, and whether visual elements belong to a fixed template structure or editable content.

Based on this analysis, the Coordinator controls how the pipeline proceeds. For simple cases, it may choose a lightweight path with limited rewriting. For dificult cases, those involving stronger cultural dependency, rhetorical reframing, or tight image and text coupling, it routes the instance to the full transcreation path. The routing decision is guided by cues such as literal traps, text length, reaction meme patterns, and coupling strength. After later stages produce candidate outputs, the Coordinator also manages revision based on critical feedback, deciding whether to accept the current result, trigger another round of rewriting, return the case for cul tural remapping, or adjust the visual execution plan.

Table 2: Alignment between human and LLM evaluation.
<table><tr><td rowspan="2">Metric</td><td colspan="2">All</td><td colspan="2">ZH→EN</td><td colspan="2">EN→ZH</td></tr><tr><td>Stat.</td><td>MAE</td><td>Stat.</td><td>MAE</td><td>Stat.</td><td>MAE</td></tr><tr><td>Intent  $( \rho )$ </td><td>0.802</td><td>0.617</td><td>0.811</td><td>0.592</td><td>0.787</td><td>0.642</td></tr><tr><td>Adaptation (ρ)</td><td>0.693</td><td>0.791</td><td>0.710</td><td>0.791</td><td>0.686</td><td>0.792</td></tr><tr><td>Coherence (ρ)</td><td>0.783</td><td>0.680</td><td>0.778</td><td>0.685</td><td>0.791</td><td>0.676</td></tr><tr><td>Quality (ρ)</td><td>0.789</td><td>0.748</td><td>0.807</td><td>0.669</td><td>0.765</td><td>0.827</td></tr><tr><td>Ranking (τ)</td><td>0.422</td><td>-</td><td>0.419</td><td>-</td><td>0.425</td><td>-</td></tr><tr><td>Ranking (Spearman)</td><td>0.507</td><td>-</td><td>0.507</td><td>一</td><td>0.508</td><td></td></tr><tr><td>Top-1 agreement</td><td>0.890</td><td>一</td><td>0.870</td><td>一</td><td>0.910</td><td></td></tr></table>

## 3.3 Adapter: Cultural Adaptation Planning

The Adapter constructs an explicit transcreation plan before text generation. Its role is to convert the source-side understanding into a target-side adaptation strategy that specifies how the meme should be localized for the target culture while preserving the source intent. This stage is designed to address the challenge of Culturespecific knowledge understanding. It explicitly identifies and adapts culturally specific elements that may not transfer efectively, including named entities, social references, discourse conventions, and stereotypes. It then estimates the likely comprehension gap for the target audience and determines whether these elements can be preserved, lightly reformulated, or replaced with target-culture counterparts. A decision of this stage is the adaptation tier, defined as one of three levels: literal, minimal rephrase, and cultural map. These tiers indicate whether the meme is largely portable, requires constrained rewriting, or needs deeper cultural remapping. In this way, the Adapter operationalizes cultural adaptation as an explicit decision layer before generation. The Adapter also produces a visual adaptation plan. It specifies whether visual cultural adaptation is needed, which visual elements should remain fixed, and which parts may require replacement or reinterpretation. Although its primary role is to solve Culture-specific knowledge understanding, this visual planning step also supports Multimodal Consistency, since later text adaptation must remain compatible with the final visual realization. The resulting plan is stored in the Transcreation Harness and guides both later rewriting and final execution.

## 3.4 Rewriter: Text Realization under Adaptation Constraints

The Rewriter generates target-side meme text candidates conditioned on the current Transcreation Harness. This stage addresses the challenge of Intent and Tone Preservation by reconstructing humor, sarcasm, irony, ridicule, and stance in a form that remains effective for the target audience. Rather than performing unrestricted generation, the Rewriter is guided by the explicit adaptation constraints produced by earlier stages, including the adaptation tier, cultural mappings, rhetorical goals, and visual constraints, so that the output remains aligned with the overall transcreation plan. We further strengthen this stage through Direct Preference Optimization (DPO). Because meme transcreation requires specialized rewriting preferences, we build a transcreation-oriented preference dataset from colloquial expressions and slang collected from online sources, and obtain human rankings over multiple rewritten candidates. Given a structured transcreation context �, we denote the preferred rewrite set as $Y ^ { + }$ and the rejected rewrite set as $Y ^ { - }$ Pairwise preferences are then constructed by sampling $( y ^ { + } , y ^ { - } )$ with $y ^ { + } \in Y ^ { + }$ and $y ^ { - } \in Y ^ { - }$ . The Rewriter is optimized with the standard DPO objective:

$$
\mathcal { L } _ { \mathrm { D P O } } = - \log \sigma \Bigg ( \beta \Bigg [ \log \frac { \pi _ { \theta } ( y ^ { + } \mid x ) } { \pi _ { \mathrm { r e f } } ( y ^ { + } \mid x ) } - \log \frac { \pi _ { \theta } ( y ^ { - } \mid x ) } { \pi _ { \mathrm { r e f } } ( y ^ { - } \mid x ) } \Bigg ] \Bigg ) .
$$

Here, � includes the source meme text together with the adaptation constraints. This alignment encourages the model to prefer rewrites that better preserve source intent, better fit the target cultural context, and better match the planned visual setting. The generated text candidates are written back into the Transcreation Harness for subsequent evaluation and possible revision.

## 3.5 Critic: Evaluation and Revision Feedback

The Critic evaluates intermediate outputs during transcreation and provides revision feedback when needed. It assesses whether the current result satisfies the three task requirements in Section 2, including intent preservation, cultural adaptation, and image text coherence. This stage is most directly related to the challenge of Multimodal Consistency, since it checks whether the adapted text, cultural mapping, and visual plan still work together as a coherent meme. Concretely, the Critic examines generated text, proposed cultural mappings, and visual adaptation plans. It checks whether the current output preserves the source meme’s communicative efect, whether the cultural adaptation is appropriate for the target audience, and whether the adapted text remains consistent with the visual realization. When problems are identified, they produce targeted feedback that specifies which part requires revision, such as text rewriting, cultural remapping, or visual adjustment. The critique results are written into the Transcreation Harness and used in the subsequent revision process before final execution.

## 3.6 Transcreation Harness and Execution Layer

The Transcreation Harness maintains the intermediate state of the pipeline. It stores intermediate outputs and signals from diferent stages, including but not limited to source meme understanding, routing decisions, cultural mappings, generated text, and critic feedback. In this way, it preserves the accumulated decisions made during transcreation and provides the runtime structure for the full framework. After revision is complete, the finalized state is passed to the Execution Layer for final visual realization. The Execution Layer applies the approved target side text together with the selected visual strategy, which may preserve the original image, replace the text, or perform stronger visual editing when deeper adaptation is required. The final target meme is therefore rendered from the structured decisions produced by the full pipeline.

![](images/ed8840266d2b5737c38d748a6039a16ca50f2e1ca636789d3d33569997d7f6e9.jpg)  
Figure 2: Main LLM-based results: (A) Dimension-wise comparison. (B) Average-score by transcreation direction. (C) Topselection rate. Our method performs best overall, remains robust in both directions, and is most frequently ranked first.

## 4 Experiments

## 4.1 Experimental Settings

4.1.1 Dataset, Baselines, and Implementation Details. Following [9], we evaluate bidirectional Chinese–English meme transcreation on 1,000 randomly sampled memes, including 500 ZH→EN and 500 EN→ZH instances. Under identical input and output settings, we compare M, our multi-agent framework TransMeme, with three baselines. B1 (Direct One-Shot Transcreation) uses Qwen-image-2.0 to generate the target meme in a single pass. B2 (Structured Single-Agent Transcreation) uses Qwen3-VL-Plus with a prompt covering meme understanding, cultural adaptation, text rewriting, and image–text coherence. The resulting structured transcreation plan is then executed by Qwen-image-2.0. B3 (Reference Benchmark Baseline) is adopted from [9]. For TransMeme, the Interpreter uses Qwen3-VL-Plus; the Coordinator, Adapter, and Critic use Qwen3-Max; the Rewriter is initialized from Qwen3-8B; and the Execution Layer uses Qwen-Image-2.0. The Rewriter is trained with DPO using TRL and LoRA. To construct the training corpus, we collect web-sourced colloquial expressions and slang in Chinese and English. For each source instance, Qwen3-8B generates multiple translation candidates, which are ranked by human annotators, yielding a bilingual ranked corpus containing 500 Chinese and 500 English instances. A single bidirectional Rewriter is trained on the merged data. The DPO corpus is constructed independently of the meme benchmark and contains no source text appearing in the evaluation meme images. At inference time, the Rewriter generates three candidate texts, after which the Critic scores them, selects the best candidate, and provides revision feedback when necessary.

4.1.2 Human Evaluation Protocol. We conduct human evaluation because cross-cultural meme transcreation requires jointly assessing semantic fidelity, cultural appropriateness, multimodal compatibility, and the overall quality of the final presentation. We randomly sample 200 instances (100 ZH→EN and 100 EN→ZH). Our evaluation dimensions follow the Section 2: Intent Preservation, Cultural Adaptation, and Multimodal Coherence correspond to the three task requirements, while Expression Quality measures fluency, natural ness, and the overall presentation quality of the final meme in the target culture. Each sample is rated on a 5-point Likert scale on all dimensions, and annotators also provide a full ranking over the compared systems for each source meme. We employ five bilingual annotators familiar with both meme cultures. Before annotation, all annotators receive training based on a guideline covering the evaluation dimensions and common edge cases. The full guideline is included in the supplementary material. All annotators independently evaluate every sample using the original meme and anonymized system outputs presented in randomized order. For scalar judgments, each system output for a given source meme is treated as one evaluation item. For each dimension, we measure agreement among the five annotators using ICC(2,1), which evaluates absolute agreement on single ratings under a two-way random-efects model [10, 11]. For ranking-based judgments, we measure agreement with Kendall’s �, which captures concordance over the full rankings [12]. To obtain a reference human ranking for later comparison, we average per-system ranks across annotators within each sample and derive the corresponding human top-1 label from the aggregated ranking. Winner agreement among human annotators is computed as the mean pairwise Cohen’s � over their top-1 selections, where � measures agreement beyond chance for categorical decisions [13]. Overall, the human judgments exhibit reliable agreement. For scalar ratings, ICC(2,1) is 0.842 for intent, 0.772 for adaptation, 0.821 for coherence, and 0.809 for quality. Agreement is slightly higher for ZH→EN than for EN→ZH across all four dimensions: 0.858 versus 0.824 for intent, 0.797 versus 0.746 for adaptation, 0.832 versus 0.809 for coherence, and 0.832 versus 0.784 for quality. For comparative judgments, full-ranking consistency reaches a Kendall’s � of 0.604 overall, with similar values for ZH→EN (� = 0.606) and EN→ZH (� = 0.601). Winner agreement is moderate, with an overall Cohen’s � of 0.554, and is slightly higher for ZH→EN (� = 0.572) than for EN→ZH (� = 0.538).

4.1.3 LLM Evaluator Validation. To extend evaluation beyond the human-annotated subset, we use an LLM-based evaluator (Qwen3.5-plus) for large-scale comparison on the full evaluation set. Following the same guidelines as human judges, it takes the original meme, the final rendered outputs of all systems as input, and outputs the same four scalar scores together with an overall ranking. We validate the evaluator on the 200-sample human-annotated subset. For scalar scores, we compare each LLM score with the mean human rating for the corresponding source meme-system pair using Spearman correlation and MAE. For ranking, we compare the LLM ranking with the aggregated human ranking for each sample using Kendall’s � and Spearman correlation, averaged over all samples, and also

![](images/d3760bb94120825df28f63ba45545c79ba30544cd1a0385e23a221b0d5790ec7.jpg)  
Figure 3: Qualitative comparison of meme transcreation results. Our method produces target memes that better preserve humor, improve cultural appropriateness, and maintain image-text coherence.

report LLM-human Top-1 agreement against the aggregated human top-1 label. As shown in Table 2, the evaluator shows good agreement with aggregated human judgments on scalar scores, with overall Spearman correlations ranging from 0.693 to 0.802 across the four dimensions. Alignment is strongest on Intent Preservation and weakest on Cultural Adaptation, the most culturally sensitive dimension. For system-level comparison, the evaluator achieves 89.0% Top-1 agreement. Full-ranking alignment is more moderate, with mean Kendall’s � = 0.422 and mean Spearman correlation = 0.507, but remains stable across both directions. These results support using the LLM evaluator as a scalable proxy for comparison, while human evaluation on the subset remains the primary validity reference.

## 4.2 Main Results

We report the main results from two perspectives, with Table 3 showing aggregated human evaluation on the subset and Figure 2 summarizing LLM-based evaluation on the full set. Across both views, TransMeme consistently outperforms all baselines. As shown in Table 3, M achieves the best human evaluation scores on all dimensions, reaching 3.950 on Intent Preservation, 4.097 on Cultural Adaptation, 4.111 on Multimodal Coherence, and 4.328 on Expression Quality, with an overall average of 4.122. B2 is the strongest baseline with an average of 3.097, while B3 and B1 trail substantially at 1.811 and 1.217, respectively. Moreover, the gains of M over all baselines are statistically significant on every dimension and on the overall average score under two-sided Wilcoxon signedrank tests (all $p < 0 . 0 0 1 )$ , and remain significant after Bonferroni correction. Figure 2 shows that the same pattern holds at scale on the full evaluation set. In the overall dimension-wise comparison, M consistently outperforms all baselines on Intent Preservation, Cultural Adaptation, Multimodal Coherence, and Expression Quality. The gap over B2, the strongest baseline, remains clear across all dimensions, indicating that the gains do not come merely from adding partial structure to prompting, but from explicitly decomposing transcreation into understanding, cultural adaptation planning, rewriting, and revision. The directional analysis in Figure 2(B) further shows that M remains the best-performing method in both ZH→EN and EN→ZH settings. The relative ordering of methods is stable across directions, suggesting that the framework is robust rather than overfitted to a single transfer direction. M, B1, and B2 perform better in the ZH→EN setting, whereas B3 shows the opposite pattern. Finally, Figure 2(C) highlights the ranking advantage of M. Our method is selected as the top-ranked system more often than others, indicating that it is not only better on average, but also more frequently preferred as the best output on individual examples. Taken together, the human and LLM results consistently confirm the efectiveness of our framework. As shown in Figure 3, we further present a randomly selected set of examples for qualitative analysis. These cases are consistent with the quantitative results above and help illustrate the behaviors underlying the performance diferences. Specifically, they show that the advantage of M lies in more natural humor reconstruction, more efective cultural substitution when source-specific references cannot be transferred directly, and stronger preservation of image-text coherence. By contrast, the baselines often produce literal or awkward target text, insuficient cultural adaptation, or mismatches between the image and the text.

Table 3: Aggregated human evaluation results, where our method performs best across all dimensions. Avg. denotes the mean of the four dimension scores.
<table><tr><td>Method</td><td>Intent</td><td>Adaptation</td><td>Coherence</td><td>Quality</td><td>Avg.</td></tr><tr><td>M</td><td>3.950</td><td>4.097</td><td>4.111</td><td>4.328</td><td>4.122</td></tr><tr><td>B1</td><td>1.187</td><td>1.328</td><td>1.180</td><td>1.172</td><td>1.217</td></tr><tr><td>B2</td><td>3.385</td><td>3.138</td><td>3.380</td><td>2.484</td><td>3.097</td></tr><tr><td>B3</td><td>1.716</td><td>1.818</td><td>1.910</td><td>1.800</td><td>1.811</td></tr></table>

## 4.3 Component Analysis

To better understand where the gains of TransMeme come from, we analyze three key components separately: the rewriter design, the critic-revision loop, and the cultural planning module.

4.3.1 Rewriter Quality Analysis. We compare the full model (M) against two variants: A1, which removes DPO while keeping the rest of the pipeline unchanged, and A2, which replaces the final rewriter with Qwen3-Max. Since these variants only afect text generation, we evaluate them using text-level judgments rather than scores. We randomly sample 200 instances, including 100 Chineseto-English and 100 English-to-Chinese cases, and use Qwen3-Max as the evaluator to score generated texts along three dimensions: Naturalness, Cultural quality, and Humor. We include A2 as a strong of-the-shelf baseline: compared with our DPO-trained Qwen3-8B rewriter, Qwen3-Max is a substantially larger general-purpose model. As shown in Figure 4, M consistently achieves the best performance across all dimensions, obtaining 4.035 on Naturalness, 4.435 on Cul tural quality, and 4.060 on Humor, with an overall average of 4.177. Removing DPO leads to a substantial drop: A1 decreases to 2.945 on Naturalness, 2.945 on Cultural quality, and 2.625 on Humor, with an average of 2.838. A2, despite serving as a strong replacement baseline, remains below the full model, reaching 3.895 on Naturalness, 3.850 on Cultural quality, and 3.760 on Humor, with an average of 3.835. The same pattern is reflected in the best-method comparison in Figure 4 (right), where M is selected as the best text 117 times, compared with 74 for A2 and 9 for A1. These results show that DPO contributes substantially to text quality.

4.3.2 Critic Loop Analysis. We analyze the critic loop in terms of trigger frequency, error types, and revision efectiveness. On the full evaluation set, 125 of 1,000 samples enter the loop, corresponding to a trigger rate of 12.5%. Among these cases, the average loop count is 1.72 and the maximum is 2, with 35 samples revised once and 90 revised twice. The trigger rate difers substantially by direction: 106 of 500 ZH→EN cases (21.2%) enter the loop, compared with only 19 of 500 EN→ZH cases (3.8%), suggesting that ZH→EN requires more iterative cultural and pragmatic adaptation. Of the 125 trig gered cases, 75 (60.0%) involve cultural mismatch, 47 (37.6%) involve intent drift, 2 (1.6%) involve image–text inconsistency, and 1 (0.8%) involves weak humor or awkward wording. For ZH→EN, cultural mismatch accounts for 62.3% of cases and intent drift for 35.8%; for EN→ZH, the two categories each account for 47.4%, indicating that the critic mainly repairs cultural and semantic failures rather than local stylistic issues. To evaluate revision efectiveness, we compare the pre- and post-revision outputs of the 125 triggered samples using Qwen3.5-397b as the evaluator. The revised outputs are preferred in 105 cases (84.0%), while the pre-revision outputs are preferred in 20 cases (16.0%). Revision improves Intent Preservation from 2.352 to 3.632, Cultural Adaptation from 2.272 to 3.704, and Expression Quality from 2.736 to 3.792, raising the overall average from 2.453 to 3.709. The average score also increases from 2.396 to 3.629 for ZH→EN and from 2.772 to 4.158 for EN→ZH, demonstrating that the critic loop efectively corrects culturally and semantically problematic generations.

![](images/578b34ac3a9770b42a96dd65477be5676de39a5e6f03bf24e68d50e530ed48e4.jpg)

![](images/695d1246ff306753cfe780cedf30dd685b6190e08610e0c193822cc42b72b558.jpg)  
Figure 4: Rewriter quality analysis: M consistently outperforms A1 and A2, confirming the benefit of DPO.

4.3.3 Cultural Planning Analysis. We examine the role of the cultural planning module, which is triggered for 181 of 1,000 samples, including 155 ZH→EN cases and 26 EN→ZH cases. This clear asymmetry suggests that explicit cultural planning is more frequently required when adapting Chinese memes for English audiences. Representative cases illustrate the types of textual and visual adaptation handled by the module. In an EN→ZH meme built around “celebrating my 18th birthday” at a bar, the humor depends on the legal drinking age and the irony of being an underage regular, requiring cultural reinterpretation rather than literal translation. In a ZH→EN meme containing the Chinese phrase meaning “My period’s here”, the euphemism is correctly recognized as referring to menstruation rather than a literal aunt, preserving its intended tone and pragmatic force. The module also supports culturally appropriate visual substitution. As shown in Figure 3, the first-column example replaces Loopy with Pikachu as a more recognizable and accessible expressive figure for the target audience, while the fourth-column example replaces a red envelope with a transfer record, substituting a culture-specific social symbol with a more immediately interpretable equivalent. To quantify the module’s contribution, we compare the full system with a variant that skips cultural planning while keeping all other components unchanged. The comparison is conducted on the 181 triggered samples using Qwen3.5-397b as the evaluator across Intent Preservation, Cultural Adaptation, and Expression Quality. The full system is preferred in 180 cases, whereas the variant without cultural planning is preferred in only 1 case. It also achieves substantially higher average scores of 3.972, 3.972, and 3.983 on the three dimensions, respectively, compared with 1.600, 1.511, and 1.606 for the variant. These results show that cultural planning is not only frequently activated but also essential in cases requiring non-literal cultural reinterpretation or visual substitution.

## 4.4 Error Analysis

Despite its overall advantage, our framework does not rank first on 400 of the 1,000 evaluation samples. A manual analysis of these nontop-ranked cases identifies three main error types: weak humor reconstruction (271/400, 67.8%), image–text mismatch (108/400, 27.0%), and cultural knowledge gaps (21/400, 5.3%). Weak humor reconstruction is therefore the dominant failure mode, accounting for more than two-thirds of all non-top-ranked cases. The most common pattern is that the system preserves the source intent but fails to recreate an equally efective humorous efect in the target culture. These outputs are usually understandable and semantically acceptable, yet they are often less witty, less natural, or less punchy than the preferred alternative. The second major error type is image– text mismatch, where the rewritten text is plausible in isolation but does not fully align with the meme’s visual template or discourse framing. By contrast, pure cultural knowledge gaps are relatively rare. This suggests that the main remaining challenge is not simply recognizing culture-specific references, but converting them into humor that is both natural in the target culture and compatible with the accompanying image. This pattern is consistent across both transcreation directions. For ZH→EN, weak humor reconstruction accounts for 157 of 215 non-top-ranked cases (73.0%), followed by image–text mismatch (50/215, 23.3%) and cultural knowledge gaps (8/215, 3.7%). For EN→ZH, weak humor reconstruction is also the largest category, with 114 of 185 cases (61.6%), followed by image– text mismatch (58/185, 31.4%) and cultural knowledge gaps (13/185, 7.0%).

## 4.5 Discussion on Model-Family Scope

The goal of this work is to validate the proposed framework design rather than to exhaustively benchmark all possible model-family combinations. Under this goal, a controlled single-family setting is suficient to test the contribution of the framework itself, since the main comparison is between diferent system organizations rather than between diferent model providers. In particular, our strongest baseline B2 already uses a strong single-agent setup, so the comparison directly tests whether explicit role decomposition, cultural planning, and iterative revision provide gains beyond a strong endto-end alternative. This interpretation is further supported by our component analyses, which show that the improvements come from the proposed modules themselves. We therefore take the current experiments as evidence for the efectiveness of the framework design under a strongly controlled setting, while broader cross-family validation remains future work.

## 5 Related Work

Prior work on memes mainly falls into three lines: meme understanding, meme generation, and multimodal translation.

Meme Understanding. Recent advances in vision-language models have strengthened joint reasoning over visual and textual information [14]. In social-media settings, multimodal and contextual modeling has been applied to covert advertisement detection, privacy leakage analysis, and misinformation identification [15–17]. More specifically, computational meme research has primarily treated memes as multimodal recognition and interpretation problems, including hate speech and ofensiveness detection [18, 19], emotion and misogyny analysis [20–22], harmful-content explanation and entity-role reasoning [23, 24], meme captioning and metaphor interpretation [25], intent description [26], crossplatform contextual grounding [27], and unified meme understanding with large vision-language models and benchmark corpora [28, 29]. Despite this progress, existing studies largely focus on recognizing, explaining, or moderating meme content rather than adapting it for audiences from diferent cultural backgrounds [30].

Meme Generation. The second line studies automatic meme generation, from early template-based generation [31] to more recent LLM-based or MLLM-based systems with stronger controllability and interaction [32, 33]. These methods, however, mainly target in-culture meme creation rather than cross-cultural re-authoring. Multi-agent decomposition and coordination have also been explored in harmful meme detection, structured optimization, and large-scale agent collaboration [34–36].

Multimodal Translation. Our task is related to multimodal machine translation. Prior work introduces image-grounded benchmarks [37], visual attention [38], imagined visual features [39], and image-based pivoting [40, 41]. However, later studies show that visual context is often underused [42–44]. Related work also examines text humanization [45] and detector generalization [46]. In contrast, transcreation emphasizes creative and culturally adapted rewriting for target audiences [47–49].

Taken together, these lines do not directly address cross-cultural meme transcreation as a multimodal adaptation problem that must jointly preserve communicative intent, reconstruct humor, and maintain coherence between text and image. Most closely related to our work, Zhao et al. [9] introduced cross-cultural meme transcreation as a benchmarked multimodal generation task, but their work mainly establishes the task setting without characterizing its core challenges or proposing a tailored method to address them. In contrast, we explicitly analyze the core challenges of meme transcreation and develop a tailored framework to address them.

## 6 Conclusion

We study cross-cultural meme transcreation as a multimodal adaptation problem that requires preserving intent, adapting culturedependent meaning, and maintaining text-image coherence. To address this task, we propose a plan-and-revise multi-agent framework with explicit cultural planning, iterative revision, and conditional visual execution. Experiments on Chinese-English meme transcreation show that our method consistently outperforms strong baselines, and further analyses confirm the contribution of the main components. Overall, our results highlight the importance of explicit cultural reasoning and coordinated multimodal adaptation for efective meme transcreation.

## Acknowledgments

This work was supported by the National Key Research and Development Program of China (No. 2025YFB3110200), the Guangdong Basic and Applied Basic Research Foundation (No. 2026A1515030046), and the State Key Laboratory of Internet Architecture, Tsinghua University (No. HLW2025ZD14).

## References

[1] Limor Shifman. 2013. Memes in a Digital World: Reconciling with a Conceptua Troublemaker. Journal ofComputer-Mediated Communication 18, 3 (2013), 362– 377.

[2] Limor Shifman. 2013. Memes in digital culture. MIT press.

[3] Saifuddin Ahmed and Muhammad Masood. 2024. Breaking Barriers With Memes: How Memes Bridge Political Cynicism to Online Political Participation. Social Media + Society 10, 2 (2024), 1–12.

[4] Eemeli Hakoköngäs, Otto Halmesvaara, and Inari Sakki. 2020. Persuasion Through Bitter Humor: Multimodal Discourse Analysis of Rhetoric in Internet Memes of Two Far-Right Groups in Finland. Social Media + Society 6, 2 (2020).

[5] Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. 2015. Neural Machine Translation by Jointly Learning to Align and Translate. In Proceedings ofthe International Conference on Learning Representations (ICLR).

[6] Lucia Specia, Stella Frank, Khalil Sima’an, and Desmond Elliott. 2016. A Shared Task on Multimodal Machine Translation and Crosslingual Image Description. In Proceedings ofthe First Conference on Machine Translation: Volume 2, Shared Task Papers. Association for Computational Linguistics, Berlin, Germany, 543–553.

[7] Kalpesh Krishna, John Wieting, and Mohit Iyyer. 2020. Reformulating Unsupervised Style Transfer as Paraphrase Generation. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP). Asso ciation for Computational Linguistics, Online, 737–762.

[8] Simran Khanuja, Sathyanarayanan Ramamoorthy, Yueqi Song, and Graham Neu big. 2024. An Image Speaks a Thousand Words, but Can Everyone Listen? On Image Transcreation for Cultural Relevance. In Proceedings ofthe 2024 Conference on Empirical Methods in Natural Language Processing. Association for Computa tional Linguistics, Miami, Florida, USA, 10258–10279.

[9] Yuming Zhao, Peiyi Zhang, and Oana Ignat. 2026. Beyond Translation: Cross-Cultural Meme Transcreation with Vision-Language Models. arXiv preprint arXiv:2602.02510 (2026).

[10] Patrick E. Shrout and Joseph L. Fleiss. 1979. Intraclass Correlations: Uses in Assessing Rater Reliability. Psychological Bulletin 86, 2 (1979), 420–428.

[11] Terry K. Koo and Mae Y. Li. 2016. A Guideline of Selecting and Reporting Intraclass Correlation Coeficients for Reliability Research. Journal ofChiropractic Medicine 15, 2 (2016), 155–163.

[12] M. G. Kendall and B. Babington Smith. 1939. The Problem of � Rankings. The Annals ofMathematical Statistics 10, 3 (1939), 275–287.

[13] Jacob Cohen. 1960. A Coeficient of Agreement for Nominal Scales. Educational and Psychological Measurement 20, 1 (1960), 37–46.

[14] Xinlei He, Guowen Xu, Xingshuo Han, Qian Wang, Lingchen Zhao, Chao Shen, Chenhao Lin, Zhengyu Zhao, Qian Li, Le Yang, et al. 2025. Artificial intelligence security and privacy: a survey. Science China Information Sciences 68, 8 (2025), 181101.

[15] Jingyi Zheng, Tianyi Hu, Yule Liu, Zhen Sun, Zongmin Zhang, Zifan Peng, Wenhan Dong, and Xinlei He. 2025. CHASM: Unveiling Covert Advertisements on Chinese Social Media. In Advances in Neural Information Processing Systems, Vol. 38.

[16] Zifan Peng, Yini Huang, Aiwen Lu, Qiming Ye, Peixian Zhang, Jingyi Zheng, Yule Liu, Xuechao Wang, Xinlei He, and Jiaheng Wei. 2026. What Your Posts Reveal: A Benchmark and Agentic Framework for User-Level Privacy Leakage on Social Media. arXiv preprint arXiv:2606.06784 (2026).

[17] Zifan Peng, Mingchen Li, Yue Wang, and Daniel Y. Mo. 2025. Prompt-Based Contrastive Learning to Combat the COVID-19 Infodemic. Machine Learning 114, 1 (2025), 6.

[18] Douwe Kiela, Hamed Firooz, Aravind Mohan, Vedanuj Goswami, Amanpreet Singh, Pratik Ringshia, and Davide Testuggine. 2020. The Hateful Memes Challenge: Detecting Hate Speech in Multimodal Memes. In Advances in Neural Information Processing Systems, Vol. 33. 2611–2624.

[19] Shardul Suryawanshi, Bharathi Raja Chakravarthi, Mihael Arcan, and Paul Buitelaar. 2020. Multimodal Meme Dataset (MultiOFF) for Identifying Ofensive Content in Image and Text. In Proceedings of the Second Workshop on Trolling, Aggression and Cyberbullying. 32–41.

[20] Chhavi Sharma, Deepesh Bhageria, William Scott, Srinivas Pykl, Amitava Das, Tanmoy Chakraborty, Viswanath Pulabaigari, and Björn Gambäck. 2020. SemEval-2020 task 8: Memotion analysis-the visuo-lingual metaphor!. In Proceedings of the fourteenth workshop on semantic evaluation. 759–773.

[21] Dushyant Singh Chauhan, SR Dhanush, Asif Ekbal, and Pushpak Bhattacharyya. 2020. All-in-one: A deep attentive multi-task learning framework for humour, sarcasm, ofensive, motivation, and sentiment on memes. In Proceedings ofthe 1st conference ofthe Asia-Pacific chapter ofthe association for computational linguistics and the 10th international joint conference on natural language processing. 281–290.

[22] Elisabetta Fersini, Francesca Gasparini, Giulia Rizzi, Aurora Saibene, Berta Chulvi, Paolo Rosso, Alyssa Lees, and Jefrey Sorensen. 2022. SemEval-2022 task 5: Multimedia automatic misogyny identification. In Proceedings ofthe 16th International Workshop on Semantic Evaluation (SemEval-2022). 533–549.

[23] Hongzhan Lin, Ziyang Luo, Jing Ma, and Long Chen. 2023. Beneath the surface: Unveiling harmful memes with multimodal reasoning distilled from large language models. In Findings of the association for computational linguistics: EMNLP 2023. 9114–9128.

[24] Shivam Sharma, Atharva Kulkarni, Tharun Suresh, Himanshi Mathur, Preslav Nakov, Md. Shad Akhtar, and Tanmoy Chakraborty. 2023. Characterizing the entities in harmful memes: Who is the hero, the villain, the victim?. In Proceedings ofthe 17th Conference ofthe European Chapter ofthe Association for Computational Linguistics. 2149–2163.

[25] EunJeong Hwang and Vered Shwartz. 2023. MemeCap: A Dataset for Captioning and Interpreting Memes. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing. 1433–1445.

[26] Jeongsik Park, Khoi P.N. Nguyen, Terrence Li, Suyesh Shrestha, Megan Kim Vu, Jerry Yining Wang, and Vincent Ng. 2024. MemeIntent: Benchmarking intent description generation for memes. In Proceedings ofthe 25th Annual Meeting of the Special Interest Group on Discourse and Dialogue. 631–643.

[27] Saurav Joshi, Filip Ilievski, and Luca Luceri. 2024. Contextualizing internet memes across social media platforms. In Companion Proceedings ofthe ACM Web Conference 2024. 1831–1840.

[28] Zhengyi Zhao, Shubo Zhang, Yuxi Zhang, Yanxi Zhao, Yifan Zhang, Zezhong Wang, Huimin Wang, Yutian Zhao, Bin Liang, Yefeng Zheng, Binyang Li, Kam-Fai Wong, and Xian Wu. 2025. MemeReaCon: Probing Contextual Meme Understanding in Large Vision-Language Models. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing. 3559–3582.

[29] Jeongsik Park, Khoi P. N. Nguyen, Jihyung Park, Minseok Kim, Jaeheon Lee, Jae Won Choi, Kalyani Ganta, Phalgun Ashrit Kasu, Rohan Sarakinti, Sanjana Vipperla, Sai Sathanapalli, Nishan Vaghani, and Vincent Ng. 2025. MemeInterpret: Towards an All-in-One Dataset for Meme Understanding. In Findings of the Association for Computational Linguistics: EMNLP 2025. 16073–16087.

[30] Khoi P. N. Nguyen and Vincent Ng. 2024. Computational Meme Understanding: A Survey. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing. 21251–21267.

[31] Aadhavan Sadasivam, Kausic Gunasekar, Hasan Davulcu, and Yezhou Yang. 2020. memeBot: Towards Automatic Image Meme Generation. arXiv preprint arXiv:2004.14571 (2020).

[32] Han Wang and Roy Ka-Wei Lee. 2024. MemeCraft: Contextual and stance-driven multimodal meme generation. In Proceedings ofthe ACM Web Conference 2024. 4642–4652.

[33] Yaqi Cai, Shancheng Fang, Yadong Qu, Xiaorui Wang, Meng Shao, and Hongtao Xie. 2025. IterMeme: Expert-Guided Multimodal LLM for Interactive Meme Creation with Layout-Aware Generation.. In IJCAI. 720–728.

[34] Ziyan Liu, Chunxiao Fan, Haoran Lou, Yuexin Wu, and Kaiwei Deng. 2025. MIND: A multi-agent framework for zero-shot harmful meme detection. In Proceedings of the 63rd Annual Meeting ofthe Association for Computational Linguistics (Volume 1: Long Papers). 923–947.

[35] Jingyi Zheng, Zifan Peng, Yule Liu, Junfeng Wang, Yifan Liao, Wenhan Dong, and Xinlei He. 2025. GasAgent: A Multi-Agent Framework for Automated Gas Optimization in Smart Contracts. arXiv preprint arXiv:2507.15761 (2025).

[36] Qiming Ye, Peixian Zhang, Yupeng He, Zifan Peng, and Gareth Tyson. 2026. Behind EvoMap: Characterizing a Self-Evolving Agent-to-Agent Collaboration Network. arXiv preprint arXiv:2605.25815 (2026).

[37] Desmond Elliott, Stella Frank, Khalil Sima’an, and Lucia Specia. 2016. Multi30K: Multilingual English-German Image Descriptions. In Proceedings ofthe 5th Workshop on Vision and Language. 70–74.

[38] Po-Yao Huang, Frederick Liu, Sz-Rung Shiang, Jean Oh, and Chris Dyer. 2016. Attention-based Multimodal Neural Machine Translation. In Proceedings ofthe First Conference on Machine Translation: Volume 2, Shared Task Papers. 639–645.

[39] Desmond Elliott and Ákos Kádár. 2017. Imagination Improves Multimodal Translation. In Proceedings of the Eighth International Joint Conference on Natural Language Processing. 130–141.

[40] Spandana Gella, Rico Sennrich, Frank Keller, and Mirella Lapata. 2017. Image Pivoting for Learning Multilingual Multimodal Representations. In Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing. 2839–2845.

[41] Po-Yao Huang, Junjie Hu, Xiaojun Chang, and Alexander Hauptmann. 2020. Unsupervised Multimodal Neural Machine Translation with Pseudo Visual Pivoting. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics. 8226–8237.

[42] Zhiyong Wu, Lingpeng Kong, Xiang Li, and Ben Kao. 2021. Good for misconceived reasons: An empirical revisiting on the need for visual context in multimodal machine translation. In Proceedings ofthe 59th Annual Meeting ofthe Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers). 6153–6166.

[43] Jiaoda Li, Duygu Ataman, and Rico Sennrich. 2021. Vision matters when it should: Sanity checking multimodal machine translation models. In Proceedings ofthe 2021 conference on empirical methods in natural language processing. 8556–8562.

[44] Yi Feng, Chuanyi Li, Jiatong He, Zhenyu Hou, and Vincent Ng. 2025. Multimodal Neural Machine Translation: A Survey of the State of the Art. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing. 22130–22147.

[45] Jingyi Zheng, Junfeng Wang, Zhen Sun, Wenhan Dong, Yule Liu, and Xinlei He. 2025. TH-Bench: Evaluating Evading Attacks via Humanizing AI Text on Machine-Generated Text Detectors. In Proceedings of the 31st ACM SIGKDD

Conference on Knowledge Discovery and Data Mining V.2. 5948–5959.

[46] Yule Liu, Zhiyuan Zhong, Yifan Liao, Zhen Sun, Jingyi Zheng, Jiaheng Wei, Qingyuan Gong, Fenghua Tong, Yang Chen, Yang Zhang, et al. 2025. On the generalization and adaptation ability of machine-generated text detectors in academic writing. In Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 2. 5674–5685.

[47] Mar Díaz-Millón and María Dolores Olvera-Lobo. 2023. Towards a Definition of Transcreation: A Systematic Literature Review. Perspectives: Studies in Translation Theory and Practice 31, 2 (2023), 347–364.

[48] Nga-Ki Mavis Ho. 2021. Transcreation in Marketing: A Corpus-based Study of Persuasion in Optional Shifts from English to Chinese. Perspectives: Studies in Translation Theory and Practice 29, 3 (2021), 426–438.

[49] Nikola Ácsová. 2022. The Transcreation of Advertisements. L10N: Translation and Localization Journal 1, 1 (2022), 102–127.