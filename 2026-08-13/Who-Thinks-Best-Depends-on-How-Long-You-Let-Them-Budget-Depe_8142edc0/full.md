# Who Thinks Best Depends on How Long You Let Them: Budget-Dependent Rankings in LLM Evaluation

Rodrigo Guedes de Souza

Alison R. Panisson

Federal University of Santa Catarina (UFSC)

guedes.rodrigo@grad.ufsc.br,alison.panisson@ufsc.br

August 13, 2026

## Abstract

Standard evaluation of large language models assumes stable model rankings across inference conditions. We challenge this assumption by varying the token generation budget, i.e., the maximum tokens a model may produce, across seven levels (64–4,096), evaluating four models on three reasoning benchmarks (56,476 inferences). We report four findings: (i) 3– 19% of items exhibit non-monotone behavior (accuracy decreasing with more budget), even after controlling for truncation, and this phenomenon is model-specific (cross-model overlap: 6–14%). (ii) Model rankings reverse across budgets on all benchmarks (p<0.01, McNemar). (iii) Oracle analysis reveals model complementarity up to +27.8 pp, most pronounced at constrained budgets. (iv) A budget-aware router captures 14.1% of the oracle gap cross-domain; budget features help within-domain (+1.6 to +5.7 pp) but are domain-specific and hurt transfer (-1.2 pp). These results argue for budget-conditioned evaluation protocols.

## 1 Introduction

Leaderboards and benchmark comparisons form the primary evidence base for selecting Large Language Models (LLMs) in practice [1, 2]. A model declared “state of the art" on GSM8K or GPQA is presumed superior across deployment scenarios. Yet this conclusion rests on an implicit assumption: that model rankings are invariant to the inference-time token generation budget, i.e., the maximum number of tokens the model is allowed to produce before its output is truncated or terminated.

This assumption is increasingly untenable. Recent work on test-time compute scaling [3, 4] has shown that allowing models more tokens to "think" can substantially improve performance, but the effect varies across models and tasks. Meanwhile, the "overthinking" literature [7, 8] documents cases where additional reasoning tokens harm accuracy. These observations hint at a deeper phenomenon: the evaluation landscape may be fundamentally budget-dependent, with model rankings, item difficulty, and model complementarity all varying as a function of the token budget.

We provide the first systematic investigation of this phenomenon. We evaluate four models spanning 8B to 70B parameters across three reasoning benchmarks at seven token budgets from 64 to 4,096 tokens, a total of 56,476 individual inferences at temperature T=0 for full determinism. Our analysis yields four contributions:

1. Item-level behavioral taxonomy. We classify each model-item pair into four behavioral categories under budget variation (always-correct, monotoneincreasing, non-monotone, always-wrong). We find that non-monotone behavior, representing performance degradation with increased budget, is not rare (up to 25.8% of items) and, crucially, is model-specific: the same item rarely triggers overthinking across different models (cross-model overlap as low as 9%).

2. Statistically significant ranking reversals. The best-performing model changes across budget levels on all three benchmarks. On GSM8K, LLaMA-3.3 70B leads at b=256 (62.4%) while GPT-OSS 20B dominates at b=4096 (94.8%, $p < 0 . 0 0 1 )$ ; on GPQA, LLaMA-3 8B ranks first at b=512 (21.2%) before GPT-OSS 20B leads nominally at b=4096 (51.0% vs. 50.0%, not significant; n=198). Multiple intermediate reversals are significant (McNemar's $\chi ^ { 2 } , p < 0 . 0 1 )$

3. Oracle gap dynamics. A per-item oracle ensemble reveals that model complementarity is most valuable under constrained budgets. On GPQA, the oracle exceeds the single best model by 27.8 pp at b=4096 and exhibits even larger relative gains at lower budgets. On GSM8K, the oracle gap is non-monotonic, peaking at b=256 (+16.9 pp) before declining as models converge (Jaccard similarity: 0.048 → 0.741).

4. Budget-aware routing proof-of-concept. We train per-model XGBoost classifiers on text features and budget to predict item-level correctness, then route each item to the model with the highest predicted probability. In a cross-domain evaluation (train on GSM8K+MATH-500, test on GPQA), this approach achieves +2.67 pp over the best-per-budget baseline (95% CI [0.94, 4.40]), capturing 14.1% of the oracle gap. A within-domain ablation reveals that budget features provide +1.6 to +5.7 pp, yet these patterns are domain-specific and hurt cross-domain transfer by -1.2 pp.

Figure 1 illustrates our core finding: model rankings are not stable across token budgets. These results have direct implications for evaluation practice, model deployment, and the design of routing systems that must operate under heterogeneous compute constraints.

![](images/372061096357ce402687aaeab975f158bbc53b4ecdb03a3599f82b797bcc5116.jpg)  
Figure 1: Model rankings depend on token budget. Top: Accuracy heatmap across four models and seven budgets on three benchmarks; black borders indicate the best model at each budget. Bottom: Corresponding scaling curves showing crossover points where the identity of the top-performing model changes. Rankings reverse on all three benchmarks.

## 2 Related Work

Test-time compute scaling. A growing body of work studies how allocating additional computation at inference time affects model performance. Chain-ofthought prompting [6] implicitly increases the token budget by eliciting stepby-step reasoning. More recent approaches explicitly scale test-time compute through search [3], extended thinking [4], and budget-forcing mechanisms [5] These works generally study a single model under varying compute; we study multiple models simultaneously, revealing that scaling curves cross and rankings reverse.

Overthinking and reasoning efficiency. Several studies document cases where allowing LLMs more reasoning steps degrades performance. In [7], the authors identify overthinking as a failure mode of o1-like models, and [8] propose early termination strategies. Our work quantifies this at the item level across multiple models, revealing that overthinking is predominantly a model-specific phenomenon rather than an item-inherent property

Model routing and selection. LLM routing systems [9, 10, 11] aim to select the best model per query based on input features. Most approaches treat routing as budget-agnostic, optimizing for a single inference condition. Our budgetaware routing formulation is, to our knowledge, the first to incorporate the token budget as an explicit routing signal, and our SHAP analysis reveals it dominates text-based features.

Evaluation methodology. The reliability of LLM benchmarks has been questioned on multiple fronts: contamination [12], saturation [13], and sensitivity to prompt formatting |14]. Our work identifies a new axis of fragility: sensitivity to the token generation budget. We show that a benchmark's model ranking is not a single stable ordering but a family of orderings parameterized by budget.

## 3 Experimental Framework

Models. We evaluate four open-weight reasoning models spanning an order of magnitude in parameter count: LLaMA-3 8B [15], Qwen-3 32B [16] LLaMA-3.3 70B [15], and GPT-OSS 20B (model ID: openai/gpt-oss-20b, served via Together.ai; see Appendix A for full model details). All models are evaluated with greedy decoding (T=0) to ensure deterministic outputs.

Benchmarks. We use three reasoning benchmarks of increasing difficulty: GSM8K [17] (1,319 grade-school math problems), MATH-500 [18] (500 competition-level math problems), and GPQA-Diamond [19] (198 graduatelevel science questions). These span a wide difficulty range: IRT-inspired analysis yields mean easiness scores of 0.507, 0.253, and 0.172, respectively.

Token budgets. We evaluate each model-item pair at seven token budgets b ∈ {64, 128, 256, 512, 1024, 2048, 4096}, where b is the max\_tokens parameter. This yields $4 \times 3 \times 7 = 8 4$ model-dataset-budget configurations and $4 \times 2 , 0 1 7 \times 7 =$ 56,476 individual inferences.

Evaluation protocol. We extract final answers using regex-based parsing and evaluate via exact match against ground truth. Correctness is binary: a model scores 1 on item i at budget b if and only if it produces the correct final answer within b tokens.

Truncation and three-tier analysis. When a model's generation is cut off at b tokens before producing a final answer, we mark it as truncated (finish\_reason = "length') and score it as incorrect. Truncation rates vary dramatically across models: Qwen-3 32B, which uses internal <think> tokens, remains truncated on 59.7% of GPQA items even at b=4,096, while LLaMA-3.3 70B drops to 0.5% at the same budget (see Appendix, Figure 8). To disentangle genuine reasoning effects from truncation artifacts, we report a three-tier analysis throughout: (a) all items (standard scoring), (b) stop-only (restricting per model to items where that model completed its generation), and (c) common non-truncated (items where all four models completed, enabling paired comparisons on identical item sets). We note that stop-only accuracy is upward-biased (completed items tend to be easier for that model), while common non-truncated sets can be small at low budgets; we mark results as unreliable when N < 30.

![](images/c40b16ff59bd4f2798ca81783409232b7a294b5cf97d36425b3131f146508cb1.jpg)  
Figure 2: Behavioral taxonomy across models and benchmarks. Each bar decomposes items into four categories based on how correctness evolves with budget. Non-monotone items (red), indicating "overthinking," are non-trivial across all settings and most pronounced on GPQA.

## 4 Budget-Dependent Model Behavior

## 4.1 Item-Level Behavioral Taxonomy

For each model m and item i, we observe a binary correctness trajectory across seven budgets: $\mathbf { c } _ { m , i } = ( c _ { m , i } ^ { b _ { 1 } } , \hdots , c _ { m , i } ^ { b _ { 7 } } ) \in \{ 0 , 1 \} ^ { 7 }$ . We classify each trajectory into one of four behavioral categories:

• Always-correct: $c _ { m , i } ^ { b } = 1$ for all b. The model solves this item regardless of budget.

• Monotone-increasing: the sequence transitions from 0 to 1 and never reverts. More budget always helps.

• Non-monotone: the sequence contains at least one 1 → 0 transition at increasing budget. The model loses a previously correct answer when given more tokens—an "overthinking" failure.

• Always-wrong: $c _ { m , i } ^ { b } = 0$ for all b. The model fails regardless of budget.

Figure 2 shows the distribution of these categories across all model-dataset combinations. The monotone-increasing category dominates across all settings, confirming that more budget generally helps. However, non-monotone items are far from negligible: they represent 3.6% (GPT-OSS 20B on GSM8K) to 25.8% (LLaMA-3 8B on GPQA) of items. GPQA exhibits the highest non-monotone rates across all models, up to 25.8% for LLaMA-3 8B and 23.2% for LLaMA-3.3 70B, consistent with the intuition that harder problems are more susceptible to overthinking.

Controlling for truncation in the taxonomy. Since truncation at lower budgets mechanically creates 1 → 0 transitions when a model answers correctly at some budget but is truncated at a later one, we recompute the taxonomy using only the non-truncated portion of each trajectory (budgets where finish\_reason = "stop"). Non-monotone rates decrease but remain substantial (Figure 3): LLaMA-3 8B on GPQA drops from 25.8% to 19.1%, LLaMA-3.3 70B on MATH-500 barely changes (10.6% to 10.3%), and even Qwen-3 32B on GSM8K retains a 3.3% non-monotone rate after filtering (down from 11.7%, where most were truncation artifacts from its verbose <think> tokens). These results confirm that overthinking is a genuine phenomenon, not a truncation artifact.

![](images/feb656b477b517aab20666ab2ad7af360982ca58d2bd746236f86c528d4509af.jpg)  
Figure 3: Non-monotone (overthinking) rates: all budgets vs. stop-only trajectories. Rates decrease after excluding truncated budgets but remain substantial, particularly on GPQA, confirming that overthinking is a genuine reasoning failure.

Overthinking is model-specific, not item-inherent. A natural question is whether certain items inherently induce overthinking, or whether this is a model-specific phenomenon. We compute the cross-model overlap: the fraction of non-monotone items that are flagged as non-monotone by at least two of the four models. On stop-only trajectories, the overlap rates are remarkably low: 10.1% on GSM8K, 6.4% on MATH-500, and 13.8% on GPQA (compared to 16.2%, 9.7%, and 27.7% on all-item trajectories). Filtering strengthens the claim: 86–94% of items exhibiting genuine overthinking do so for only one model The implication is important: overthinking cannot be mitigated by removing “problematic" items from benchmarks, because the set of problematic items is model-dependent.

Where do non-monotone transitions occur? Analysis of the budget level at which non-monotone drops happen reveals they are concentrated at high budgets $( b \geq 1 0 2 4 )$ , suggesting that the additional tokens generated under generous budgets can derail an initially correct reasoning chain (see Appendix, Figure 9).

## 4.2 Ranking Reversals

We now examine whether budget variation leads to changes in which model ranks first, not just changes in absolute performance. Table 1 reports the best model at selected budget levels along with McNemar test statistics for the significance of the difference with the second-ranked model.

Several patterns emerge. On GSM8K, LLaMA-3.3 70B leads at b=256 $( \chi ^ { 2 } = 5 9 . 5 7 , p < 0 . 0 0 1 )$ , but GPT-OSS 20B overtakes it by b=512 and maintains the lead through b=4096 with increasing significance $( \chi ^ { 2 } = 1 5 . 8 6$ at b=4096). On MATH-500, LLaMA-3.3 70B dominates across mid-range budgets (b=256 through b=1024, all $p < 0 . 0 0 1 ,$ ), but GPT-OSS 20B closes the gap and leads at b=4096—though this final reversal is not significant $( p = 0 . 1 1 )$ . On GPQA, the most dramatic trajectory unfolds: the smallest model (LLaMA-3 8B) ranks first at b=256–512, is overtaken by LLaMA-3.3 70B at $b { = } 1 0 2 4 \ ( p < 0 . 0 1 )$ , which is in turn matched by GPT-OSS 20B at b=4096—though this last difference is not statistically significant $( \chi ^ { 2 } = 0 . 0 2 , p = 0 . 8 9 , n = 1 9 8 )$ . Results on GPQA-Diamond should be interpreted with caution due to the small sample size.

Table 1: Ranking reversals across budgets. Best model and McNemar's test comparing it to the second-ranked model. The identity of the best model shifts with budget on all three benchmarks. Significance: $^ { * } p < 0 . 0 5 , ^ { * * } p < 0 . 0 1$ $^ { * * * } p < 0 . 0 0 1$
<table><tr><td>Dataset</td><td>Budget</td><td>Best Model</td><td>Acc</td><td>2nd Model</td><td>Acc</td><td> $\chi ^ { 2 }$ </td><td>Sig</td></tr><tr><td rowspan="4">GSM8K</td><td>256</td><td>LLaMA-3.3 70B</td><td>62.4%</td><td>GPT-OSS 20B</td><td>48.6%</td><td>59.57</td><td>***</td></tr><tr><td>512</td><td>GPT-OSS 20B</td><td>79.2%</td><td>LLaMA-3.3 70B</td><td>76.6%</td><td>2.78</td><td></td></tr><tr><td>1024</td><td>GPT-OSS 20B</td><td>92.3%</td><td>Qwen-3 32B</td><td>80.2%</td><td>86.12</td><td>***</td></tr><tr><td>4096</td><td>GPT-OSS 20B</td><td>94.9%</td><td>Qwen-3 32B</td><td>91.7%</td><td>15.86</td><td>***</td></tr><tr><td rowspan="4">MATH-500</td><td>256</td><td>LLaMA-3.3 70B</td><td>21.3%</td><td>LLaMA-3 8B</td><td>13.5%</td><td>16.41</td><td>***</td></tr><tr><td>512</td><td>LLaMA-3.3 70B</td><td>49.4%</td><td>GPT-OSS 20B</td><td>30.0%</td><td>70.51</td><td>***</td></tr><tr><td>1024</td><td>LLaMA-3.3 70B</td><td>63.3%</td><td>GPT-OSS 20B</td><td>48.2%</td><td>43.81</td><td>***</td></tr><tr><td>4096</td><td>GPT-OSS 20B</td><td>70.8%</td><td>LLaMA-3.3 70B</td><td>67.1%</td><td>2.54</td><td></td></tr><tr><td rowspan="4">GPQA</td><td>256</td><td>LLaMA-3 8B</td><td>7.1%</td><td>LLaMA-3.3 70B</td><td>4.5%</td><td>1.07</td><td></td></tr><tr><td>512</td><td>LLaMA-3 8B</td><td>21.2%</td><td>LLaMA-3.3 70B</td><td>13.6%</td><td>4.56</td><td>*</td></tr><tr><td>1024</td><td>LLaMA-3.3 70B</td><td>40.9%</td><td>GPT-OSS 20B</td><td>27.8%</td><td>9.47</td><td>**</td></tr><tr><td>4096</td><td>GPT-OSS 20B</td><td>51.0%</td><td>LLaMA-3.3 70B</td><td>50.0%</td><td>0.02</td><td></td></tr></table>

Controlling for truncation. Since truncation rates differ across models (e.g., Qwen-3 32B is 59.7% truncated on GPQA at b=4096 vs. 0.5% for LLaMA-3.3 70B), ranking reversals could reflect differential truncation rather than genuine reasoning differences. We repeat the analysis on common non-truncated items, i.e., those where all four models completed their generation, and report paired McNemar tests. At budgets where common non-truncated sets are sufficiently large $( N \geq 3 0 )$ , the key findings survive: on GSM8K at b=4096 (N=1,264): GPT-OSS 20B leads (95.9%) over Qwen-3 32B (93.2%, χ2=12.66, p<0.001); on GPQA at b=4096 (N=71): GPT-OSS 20B leads (85.9%) over Qwen-3 32B (64.8%, χ2=9.33, p<0.01).

We also observe that some all-items rankings are genuine truncation artifacts: on MATH-500 at b=4096, GPT-OSS 20B leads on all items (71.0%) but Qwen-3 32B leads on common non-truncated items (86.7%), reflecting GPT-OSS's higher completion rate rather than superior reasoning. This nuance strengthens rather than weakens our thesis: the landscape depends on whether one measures “ability to answer given enough tokens" or “ability to answer within the budget," and both are legitimate evaluation criteria. Figure 4 provides a side-by-side comparison of all-items vs. stop-only accuracy.

![](images/7bc83c112bafe1f6891dc2c23d5f9d47488db9759555128235258f9063b46637.jpg)

![](images/0007241d5ea59fd6a1be2b53e7000a724415e53e28f41f1b87bb88524616b84b.jpg)

Figure 4: All-items vs. stop-only accuracy on GPQA. Left: standard scoring (truncated = incorrect). Right: accuracy restricted to items where each model completed its generation. Parenthesized values indicate small sample sizes (N < 30). Black borders mark the best model per budget. Rankings persist at high budgets even under stop-only scoring.  
![](images/fd5a59888dd5fa6c7877dd6aa8de6e994840dfde1b9d369fd463e4202ea2aca5.jpg)  
(a) Oracle gap vs. token budget.

![](images/de33da28c587eb7842ca8aa9e6cd09094a20ed441e1ab7523811b86f3de70fb1.jpg)  
(b) Pairwise Jaccard similarity vs. budget.

Figure 5: Model complementarity varies with budget. (a) The oracle gap (oracle ensemble minus best single model) peaks at low-to-moderate budgets for GSM8K but grows monotonically for harder benchmarks. (b) Mean pairwise Jaccard similarity between models’ correct-answer sets starts near zero and increases with budget, but never reaches unity, i.e., models remain complementary even at b=4096.

## 4.3 Model Complementarity and Oracle Gap

If different models excel on different items at different budgets, then an oracle that selects the best model per item could substantially outperform any single model. We quantify this via the oracle gap: the difference between a per-item oracle ensemble and the single best model at each budget.

Figure 5a plots the oracle gap across budgets for all three benchmarks. The dynamics differ markedly by dataset difficulty:

• GSM8K exhibits a non-monotonic oracle gap that peaks at +16.9 pp around b=256 before declining to +3.5 pp at b=4096. At moderate budgets, models solve different items, creating maximum complementarity. At high budgets, they converge on the same (easier) items, reducing the benefit of selection.

• MATH-500 shows a monotonically increasing gap, reaching +12.8 pp at b=4096 (CI $[ + 1 0 . 0 , + 1 5 . 8 ] \rangle$ . On this harder benchmark, models remain complementary even at generous budgets.

• GPQA displays the largest gap: +27.8 pp at b=4096 (CI [+21.7, +33.8]), indicating that the four models solve highly non-overlapping subsets of these graduate-level questions.

Convergence without consensus. Figure 5b shows the mean pairwise Jaccard similarity between models' correct-answer sets. At b=64, models agree on almost no items $( \bar { J } = 0 . 0 4 8 )$ , consistent with the near-random performance at this budget. As budgets increase, agreement grows substantially: $\bar { J } = 0 . 2 8 5$ at b=256, 0.528 at b=512, and 0.741 at b=4096. However, the asymptotic value of 0.741 indicates that even at generous budgets, models solve meaningfully different item subsets, consistent with the persistent oracle gap of 3.5–27.8 pp at b=4096 across benchmarks.

Item-level difficulty. Across all items and budgets, 1.1% of GSM8K items, 13.6% of MATH-500 items, and 10.1% of GPQA items are answered incorrectly by all four models at all seven budgets, representing a hard core of items beyond current model capabilities. Conversely, only 16.5% of item-budget pairs see all four models correct, reinforcing that model selection matters.

## 5 Budget-Aware Model Routing

The oracle gap analysis reveals that an ideal router could achieve 14–28 pp above the best single model. Can a practical router capture some of this gap? We design a budget-aware routing mechanism that selects the best model for each item, given the token budget as an explicit feature.

Method. For each model m, we train an XGBoost binary classifier $f _ { m } ( x , b ) $ [0, 1] that predicts the probability that model m answers item x correctly at budget b. Features include: $\log _ { 2 } ( b )$ (budget), surface-level text statistics (character count, word count, number of special characters, presence of LaTeX, word entropy, maximum number magnitude), and 20 PCA-reduced dimensions from sentence embeddings (all-MiniLM-L6-v2). At inference time, for each item-budget pair, the router selects $m ^ { * } = \arg \operatorname* { m a x } _ { m } f _ { m } ( x , b )$

Baselines. We compare against five baselines: Random (uniform selection), Largest-Always (always select GPT-OSS 20B, the largest model), Best-Overall (select the model with highest aggregate accuracy across all budgets), Best-Per-Budget (select the model with highest accuracy at each budget, applied uniformly to all items at that budget), and Oracle (per-item optimal selection with ground truth).

Cross-domain evaluation. To test generalization, we train on GSM8K and MATH-500 and evaluate on GPQA, a domain-transfer setting. Table 2 shows the results.

The Router-Scoring strategy achieves 22.9% accuracy, a +2.67 pp improvement over Best-Per-Budget (95% bootstrap CI [0.94, 4.40], significant since the interval excludes zero). This captures 14.1% of the oracle gap. On the discriminative subset (520 items where models disagree), the gain is +7.12 pp.

The per-budget breakdown (Figure 6) reveals that the router's advantage is concentrated at moderate budgets: at b=1024, it achieves 40.9% versus 27.8% for Best-Per-Budget (+13.1 pp). At extreme budgets (b=64 and b=4096), the router offers minimal improvement, consistent with the low model differentiation at these extremes.

Table 2: Cross-domain routing results (train: GSM8K+MATH-500, test: GPQA). The per-model scoring router captures 14.1% of the oracle gap over Best-Per-Budget with a statistically significant improvement. "Disc. subset" restricts to the 520 out of 1,386 item-budget pairs where models disagree.
<table><tr><td>Strategy</td><td>Accuracy</td><td>∆ vs BPB</td><td>Oracle Gap</td><td>Disc. Subset</td></tr><tr><td>Random</td><td>17.1%</td><td>-3.2 pp</td><td></td><td>41.0%</td></tr><tr><td>Largest-Always</td><td>23.4%</td><td>+3.1 pp</td><td></td><td>57.7%</td></tr><tr><td>Best-Overall</td><td>19.6%</td><td>-0.7 pp</td><td></td><td>47.5%</td></tr><tr><td>Best-Per-Budget</td><td>20.3%</td><td></td><td>0%</td><td>49.4%</td></tr><tr><td>Router-Scoring</td><td>22.9%</td><td>+2.67 pp</td><td>14.1%</td><td>56.5%</td></tr><tr><td>Oracle</td><td>39.2%</td><td>+18.9 pp</td><td>100%</td><td>100.0%</td></tr></table>

Feature importance. SHAP analysis (Figure 10) reveals that the budget feature $\left( \log _ { 2 } b \right)$ dominates all other features by a wide margin: its mean absolute SHAP value (2.21) is 6.1× larger than the next feature (presence of LaTeX: 0.36). Embedding-derived features contribute modestly (0.10–0.24 each), while text statistics contribute even less. This confirms that budget is the primary axis of variation in model performance, the core message of this paper.

Ablation study. Table 3 reports ablations removing key feature groups in the cross-domain setting. A notable finding is that removing budget features (‘No budget") yields higher cross-domain accuracy (24.2%) than the full router (22.9%, ∆=-1.2 pp). To determine whether this reflects a global failure of budget features or a domain-transfer artifact, we run a complementary withindomain ablation: 5-fold cross-validation on each benchmark separately (Table 4). Within-domain, budget features provide a consistent advantage: +5.7 pp on GSM8K, +3.0 pp on MATH-500, and +1.6 pp on GPQA. The reversal in the cross-domain setting reveals that budget-accuracy mappings are domain-specific: the relationship between budget and correctness learned on math problems does not transfer to graduate-level science. Text features, being domain-agnostic, generalize better across domains—but within a domain, budget remains the dominant routing signal, consistent with the SHAP analysis above. Figure 7 visualizes this contrast.

## 6 Discussion

Implications for evaluation. Our findings challenge the common practice of reporting a single accuracy number per benchmark. Since model rankings depend on the token budget, a benchmark's ranking is not a fixed ordering but a family of orderings parameterized by b. We advocate for budget-conditioned evaluation: reporting accuracy at multiple budget levels and explicitly stating the budget used. This is especially critical for constrained deployment scenarios (mobile, edge, real-time) where generous budgets are infeasible.

![](images/d8d65ed24586719d49a323b8936e6127339120e8dd9a2ed9a5fd4dd2ac3ba36c.jpg)  
Figure 6: Router accuracy by budget level (cross-domain, GPQA). The router's advantage is concentrated at moderate budgets (b=512-2048), where model complementarity is highest and the oracle gap is largest.

Implications for routing. The large oracle gaps (+3.5 to +27.8 pp) indicate that substantial gains are available from intelligent model selection. Our routing experiments reveal a nuanced picture: budget is the dominant within-domain feature (+1.6 to +5.7 pp), yet budget-accuracy patterns are domain-specific and hurt cross-domain transfer (-1.2 pp). This suggests that practical routing systems should treat budget as a powerful but non-transferable signal, complementing it with domain-agnostic features or domain adaptation techniques. Future routing systems may additionally benefit from model-internal signals (e.g., logit entropy, hidden-state representations) rather than text features alone.

The truncation confound. Truncation is a significant confound when varying token budgets: at low budgets, most models are truncated on most items, and truncation is near-perfectly correlated with incorrect answers. We have addressed this through a three-tier analysis (Section 4.2): all items, stop-only per model, and common non-truncated items with paired McNemar tests. The key findings, non-monotone behavior (Section 4.1), ranking reversals, and oracle gaps, all persist under truncation control, though some specific rankings are revealed to be truncation artifacts (e.g., MATH-500 at b=4096; see Section 4.2). Notably, non-monotone rates decrease by only 1-8 pp after filtering truncated budgets, confirming that overthinking is predominantly a genuine reasoning failure. We further validate this with a direct search for items where a model is correct at a low budget but wrong at a higher budget, both with finish\_reason=stop: across all models and datasets, we identify 1,193 such pairs (Appendix G). Future work should explore “budget-forcing" techniques [5] that encourage models to produce complete answers within the budget, rather than simply truncating.

Table 3: Cross-domain ablation (train: GSM8K+MATH-500, test: GPQA). Removing budget features improves cross-domain accuracy, suggesting that budget-accuracy patterns are domain-specific.
<table><tr><td>Configuration</td><td>Accuracy</td></tr><tr><td>Random</td><td>17.1%</td></tr><tr><td>Best-Per-Budget</td><td>20.3%</td></tr><tr><td>Text stats only (no budget, no embeddings)</td><td>21.8%</td></tr><tr><td>No budget</td><td>24.2%</td></tr><tr><td>No embeddings</td><td>21.3%</td></tr><tr><td>Full Router</td><td>22.9%</td></tr><tr><td>Oracle</td><td>39.2%</td></tr></table>

Table 4: Budget feature impact: within-domain vs. cross-domain. Budget features consistently help within-domain (5-fold CV) but hurt cross-domain transfer, confirming that budget-accuracy patterns are domain-specific rather than universally transferable.
<table><tr><td>Setting</td><td>Full Router</td><td>No Budget</td><td>∆ Budget</td></tr><tr><td>GSM8K (within-domain)</td><td>64.9%</td><td>59.2%</td><td>+5.7 pp</td></tr><tr><td>MATH-500 (within-domain)</td><td>37.5%</td><td>34.5%</td><td>+3.0 pp</td></tr><tr><td>GPQA (within-domain)</td><td>24.0%</td><td>22.4%</td><td>+1.6 pp</td></tr><tr><td>Cross→GPQA</td><td>22.9%</td><td>24.2%</td><td>-1.2 pp</td></tr></table>

Limitations. Our study has a few limitations. (i) We evaluate only four models; expanding to a broader set would strengthen the generality of our findings. (ii) Our token budgets are logarithmically spaced; finer-grained budgets might reveal additional structure. (iii) Our routing features are limited to surface-level text properties and static embeddings; richer features could capture more of the oracle gap. (iv) We focus on reasoning benchmarks with clear correct answers; generalization to open-ended tasks remains untested. (v) The 19 pp gap between our router and the oracle on GPQA suggests that most model complementarity remains unexploited, which is a compelling direction for future work. (vi) We deliberately exclude dedicated reasoning models (o1, DeepSeek-R1, QwQ) because their dual-stream architecture, allocating tokens between internal "thinking" and visible output, changes the semantics of max\_tokens. Controlling max\_tokens restricts visible output but may not constrain internal deliberation, making budget comparisons non-equivalent. Extending our framework to reasoning models is an important direction for future work.

![](images/e9a19b83d1a8d04418c85c8503247e0bf4b3fb8f6d907f27feb3faa3d70d584a.jpg)  
Figure 7: Budget feature impact across evaluation settings. Withindomain, budget is the most valuable feature (+1.6 to +5.7 pp). Cross-domain, budget features overfit to the training domain and hurt transfer (-1.2 pp). This contrast reveals that budget-accuracy patterns are domain-specific.

## 7 Conclusion

We have shown that the evaluation landscape of LLMs is fundamentally budgetdependent. Model rankings, item difficulty, and model complementarity all vary as a function of the token generation budget. These findings have three actionable implications: (1) benchmarks should adopt budget-conditioned evaluation protocols, reporting rankings at multiple budget levels; (2) model selection and routing systems should incorporate budget as a first-class signal; and (3) the substantial oracle gaps we document (+3.5 to +27.8 pp) represent a concrete opportunity for ensemble and routing methods to exploit model complementarity.

The question "which model is best?" has no single answer, it depends on how long you let them think.

## References

[1] Open LLM Leaderboard. https://huggingface.co/spaces/ open-1lm-leaderboard/open\_1lm\_leaderboard, 2024.

[2] L. Zheng, W.-L. Chiang, et al. Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. In NeurIPS, 2023.

[3] C. Snell, J. Lee, K. Xu, and A. Kumar. Scaling LLM Test-Time Compute Optimally Can be More Effective than Scaling Model Parameters. arXiv:2408.03314, 2024.

[4] N. Muennighoff, Z. Yang, et al. s1: Simple Test-Time Scaling. arXiv:2501.19393, 2025.

[5] S. Aggarwal, Y. Arora, and A. Goyal. L1: Controlling How Long A

Reasoning Model Thinks With Reinforcement Learning. arXiv:2503.04697, 2025.

[6] J. Wei, X. Wang, D. Schuurmans, et al. Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. In NeurIPS, 2022.

[7] X. Chen, Z. Xu, et al. Do NOT Think That Much for 2+3=? On the Overthinking of o1-Like LLMs. arXiv:2412.21187, 2024.

[8] Y. Sui, H. Yu, et al. Stop Overthinking: A Survey on Efficient Reasoning for Large Language Models. arXiv:2503.16419, 2025.

[9] D. Jiang, X. Ren, and B. Y. Lin. LLM-Blender: Ensembling Large Language Models with Pairwise Ranking and Generative Fusion. In ACL, 2023.

[10] K. Lu, H. Yuan, et al. Routing to the Expert: Efficient Reward-guided Ensemble of Large Language Models. In NAACL, 2024.

[11] T. Shnitzer, A. Ou, et al. Large Language Model Routing with Benchmark Datasets. arXiv:2309.15789, 2023.

[12] O. Sainz, J. Campos, et al. NLP Evaluation in Trouble: On the Need to Measure LLM Data Contamination for each Benchmark. In EMNLP Findings, 2023.

[13] D. Kiela, M. Bartolo, et al. Dynabench: Rethinking Benchmarking in NLP. In NAACL, 2021.

[14] P. Liang, R. Bommasani, et al. Holistic Evaluation of Language Models. Annals of the New York Academy of Sciences, 2023.

[15] Meta AI. The LLaMA 3 Herd of Models. arXiv:2407.21783, 2024.

[16] Qwen Team. Qwen3 Technical Report. arXiv:2505.09388, 2025.

[17] K. Cobbe, V. Kosaraju, et al. Training Verifiers to Solve Math Word Problems. arXiv:2110.14168, 2021.

[18] D. Hendrycks, C. Burns, et al. Measuring Mathematical Problem Solving With the MATH Dataset. In NeurIPS, 2021.

[19] D. Rein, B. L. Hou, et al. GPQA: A Graduate-Level Google-Proof Q&A Benchmark. In ICLR, 2024.

## A Additional Experimental Details

Model details. All models are evaluated via their standard chat/instruct variants through a unified API. Table 5 provides full identification for all models. The models span a deliberate range of architectures and parameter counts to test whether budget-dependence is a universal phenomenon or model-specific. Temperature is set to 0.0 for all models to ensure deterministic outputs and exact reproducibility.

Table 5: Full model identification.
<table><tr><td>Paper Name</td><td>Model ID</td><td>Provider(s)</td><td>Params</td></tr><tr><td>LLaMA-3 8B</td><td>meta-1lama/Llama-3.1-8B-Instruct</td><td>Groq, Cerebras, SambaNova</td><td>8B</td></tr><tr><td>Qwen-3 32B</td><td>Qwen3-32B</td><td>Groq, SambaNova</td><td>32B</td></tr><tr><td>LLaMA-3.3 70B</td><td>meta-1lama/Llama-3.3-70B-Instruct-Turbo</td><td>Together.ai</td><td>70B</td></tr><tr><td>GPT-OSS 20B</td><td>openai/gpt-oss-20b</td><td>Together.ai</td><td>20B</td></tr></table>

Compute resources. Inference was performed using cloud API endpoints. Total inference count: 56,476 individual API calls (4 models × 2,017 items × 7 budgets). Routing experiments used scikit-learn and XGBoost on a single CPU. Sentence embeddings were computed with all-MiniLM-L6-v2 (22M parameters).

## B Full Accuracy Tables

Table 6 shows accuracy and 95% bootstrap confidence intervals for all modelbudget combinations on GSM8K.

Table 6: Full accuracy with 95% bootstrap CIs on GSM8K.
<table><tr><td>Model</td><td>b=64</td><td>b=256</td><td>b=512</td><td>b=1024</td><td>b=2048</td><td>b=4096</td></tr><tr><td>LLaMA-3 8B</td><td>1.5±0.7</td><td>41.7±2.6</td><td>67.9±2.5</td><td>68.7±2.8</td><td>68.1</td><td>67.0±2.6</td></tr><tr><td>Qwen-3 32B</td><td>1.6±0.7</td><td>7.7±1.5</td><td>35.3±2.6</td><td>79.2±2.2</td><td>90.4</td><td>91.5±1.5</td></tr><tr><td>LLaMA-3.3 70B</td><td>1.2±0.6</td><td>62.4±2.6</td><td>76.4±2.3</td><td>76.3±2.3</td><td>77.6</td><td>77.6±2.3</td></tr><tr><td>GPT-OSS 20B</td><td>0.1±0.1</td><td>48.4±2.7</td><td>79.3±2.2</td><td>91.6±1.5</td><td>93.9</td><td>94.8±1.2</td></tr></table>

## C Truncation Analysis

![](images/df4edda08c7dba44ff9628e19e0f28d67e825c04b423f60f7886b509ead2abda.jpg)  
Figure 8: Truncation rate by model, budget, and dataset. Qwen-3 32B experiences the highest truncation rates across all budgets, while LLaMA-3.3 70B shows the fastest decline. At b=4096, truncation rates are near zero for all models except Qwen-3 32B (15% on GSM8K, higher on MATH-500 and GPQA).

## D Non-Monotone Transition Analysis

![](images/3ccca389450902814e4186898c64129fddbca1c8efbc6162fcefec58d24f5393.jpg)  
Figure 9: Distribution of non-monotone transitions by budget level. The heatmap shows at which budget level non-monotone items first lose a previously correct answer. Most transitions are concentrated at high budgets $\left( b \ge 1 0 2 4 \right)$ especially for GSM8K.

## E SHAP Feature Importance

![](images/4664973a0db2de07d8349f069a09c9b08b6a303b287d2b4ba338c3cead326752.jpg)  
Figure 10: SHAP feature importance for the per-model scoring router. Budget $( \log _ { 2 } b )$ dominates with a mean absolute SHAP value of 2.21, approximately 6× larger than the next feature (presence of LaTeX: 0.36). Text features and embeddings contribute modestly.

## F IRT Difficulty Analysis

![](images/12e50527c1666f8fc48b6e30f8f4bb5552dc4062362491f39602e6a2f8d7ad87.jpg)  
Figure 11: IRT-inspired difficulty distribution across benchmarks. Easiness is computed as the fraction of model-budget combinations that answer each item correctly. GSM8K items cluster around easiness 0.5 (mean: 0.507), MATH-500 around 0.25 (mean: 0.253), and GPQA around 0.17 (mean: 0.172), confirming the intended difficulty gradient.

## G Genuine Overthinking Examples

To confirm that non-monotone behavior is not merely a truncation artifact, we identify items where a model answers correctly at a low budget $b _ { l }$ but incorrectly at a higher budget $b _ { h } > b _ { l }$ , with finish\_reason=stop at both budgets (i.e.

neither response was truncated). Table 7 reports the number of unique items exhibiting this pattern per model and dataset. Across the four models and three benchmarks, we find 1,193 such (low-budget, high-budget) pairs—concrete evidence that overthinking is a genuine reasoning failure, not an artifact of output truncation.

LLaMA-3 8B exhibits the most overthinking instances (550 on GSM8K alone), consistent with its high non-monotone rate in the behavioral taxonomy (Section 4.1). Qwen-3 32B shows the fewest (65 on GSM8K, 4 on GPQA), despite its high truncation rate, suggesting that its non-monotone behavior is more often truncation-driven.

Table 7: Genuine overthinking instances (correct→wrong, both with finish\_reason=stop). Counts show the number of unique items where a model answers correctly at a lower budget but incorrectly at a higher budget, with no truncation at either level.
<table><tr><td>Model</td><td>GSM8K</td><td>MATH-500</td><td>GPQA</td></tr><tr><td>LLaMA-3 8B</td><td>550</td><td>88</td><td>74</td></tr><tr><td>Qwen-3 32B</td><td>65</td><td>9</td><td>4</td></tr><tr><td>LLaMA-3.3 70B</td><td>159</td><td>93</td><td>45</td></tr><tr><td>GPT-OSS 20B</td><td>46</td><td>52</td><td>8</td></tr></table>

## H Prompt Templates

We use the following prompt templates, adapted per benchmark. For GSM8K and MATH-500 (numerical answer):

Solve the following problem step by step. Show your complete reasoning. Problem: {question}

After your reasoning, provide your final answer on a new line in   
the exact format:   
#### [your answer]

For GPQA-Diamond (multiple choice):

Answer the following question by reasoning step by step.

Question: {question}

Options:

(A) {option\_a}

(B) {option\_b}

(C) {option\_c}

(D) {option\_d}

After your reasoning, provide your final answer on a new line in   
the exact format:   
#### [A/B/C/D]

Answer parsing. For numerical benchmarks (GSM8K, MATH-500), we parse the #### [answer] pattern and compare numerically with tolerance $1 0 ^ { - 6 }$ . Format variations $( \mathrm { e . g . , ~ } \frac { 1 } { 2 }$ vs. 0.5) are handled via float conversion with LaTeX-aware parsing. For $\mathrm { G P Q A }$ , we extract a single letter (A-D) using a cascade of regex patterns applied to the model output.