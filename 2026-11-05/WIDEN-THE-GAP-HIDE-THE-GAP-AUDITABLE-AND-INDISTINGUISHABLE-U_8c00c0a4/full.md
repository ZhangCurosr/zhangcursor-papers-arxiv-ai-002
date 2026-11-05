# WIDEN THE GAP, HIDE THE GAP: AUDITABLE AND INDISTINGUISHABLE UNLEARNING FOR LLMS

Piyush Tiwary   
Department of Electrical Communication Engineering   
Indian Institute of Science   
Bengaluru, KA 560012, India   
{piyushtiwary}@iisc.ac.in Prathosh A.P.   
Department of Electrical Communication Engineering Bengaluru, KA 560012, India   
{prathosh}@iisc.ac.in   
Anushka Srivastava   
Independent Researcher   
{anushka.srtava}@gmail.com

## ABSTRACT

Machine unlearning aims to remove a user’s data from a trained model after a deletion request. Existing methods are majorly evaluated on three axes - how well they forget, how much capability they retain, and how robust they are to membership inference attacks, but not on whether the provider can offer the user any evidence that the deletion occurred. To fill this gap, we introduce auditability as a fourth axis, defined as the separation between requested (forget set) and retained (retain set) data under a statistic the provider computes. We show that across eleven unlearning methods on TOFU and MUSE no method satisfies all four axes, with different methods falling short on different axes. To address this, we propose a two stage method that can be used on top of any unlearning method, in the the first stage we propose GAPWIDEN which uses a hinge loss to increase the difference between audit statistics of forget set and retain set. In the second stage we propose GAPVEIL, an inference time rejection-sampler which uses the base model as the drafter and re-samples based on the audit statistics of the drafted response. We observe that using our method, auditability and leakage reach oracle parity for every unlearning method, and forget performance converges to the oracle; retain performance improves for nine of eleven methods but remains short of parity. Our results show that evidence of deletion or auditability can be constructed by design, and that treating it as an explicit requirement changes the viability of existing unlearning methods.

## 1 INTRODUCTION

Large Language Models (LLMs) have driven substantial recent progress in AI, owing to their capacity to generalize across tasks (Brown et al., 2020; Wei et al., 2022; Berti et al., 2025), reason over long contexts (Liu et al., 2024b; Hsieh et al., 2024; Bai et al., 2024; Ma et al., 2026; Kontonis et al., 2026), and follow complex instructions (Ouyang et al., 2022; Zhou et al., 2023). These capabilities have led to their deployment in sensitive domains including education, healthcare, legal services. Such models are usually trained on web-scale corpora into which private and personal information inevitably seeps, creating a persistent risk of reproducing sensitive content during inference. International regulatory frameworks such as the GDPR’s ‘right to be forgotten’ (Regulation, 2016) oblige model providers to honour a user’s request to delete their data. However, deletion of data and retraining large models on remaining data is very expensive. Machine unlearning has emerged as the response to such problems where a model is modified such that it behaves as if it never saw the requested data.

Current unlearning methods are broadly evaluated along three axis (Dorna et al., 2026): (a) how thoroughly the forget set is removed / unlearned (forget performance), (b) how much general capability is retained (retain performance), and (c) how robust the resulting model is to membership inference attacks (MIA) or membership leakage. However, none of them is what the model provider owes to the user. Specifically, a deletion request is a claim the user is entitled to see honoured, and honouring it requires evidence like a verifiable demonstration that this particular user’s data is statistically erased from the model behavior. We term this property accountability or auditability, and observe that it is essentially missing from current literature.

![](images/c7400a105662a463d11e7404d58720997f45b116d29c59267a4269af54a60c59.jpg)

![](images/1171778c440e68e83bb20d24a307698f68d66573c41d4b04dd348a16d0bb43e7.jpg)  
Figure 1: Overview of proposed method. Left: The Unlearning Dilemma across the four evaluation axes. Right: (a) GAPWIDEN post-unlearning method and (b) GAPVEIL inference-time rejection sampling.

![](images/12a1ab3294eb97797e4cec015093273c357980ef70577497fe738045d52d9f4e.jpg)

![](images/b508205eb7656e049dfad97fb7c10c3d14a866edb3bfb39a76af767b79cfce9b.jpg)  
Figure 2: Four-axis unlearning evaluation. (a) Spider plot comparing representative baselines (SATIMP, RMU, NPO) against our method and the retrained Oracle. (b) Per-method performance improvement from initial unlearned models $( \theta _ { u } )$ to our proposed method across Auditability, Leak age Robustness, Forget, and Retain performance.

Particularly, auditability is a technical property by which a model provider discharges its accountability obligation. However, we find that auditability is usually difficult to demonstrate, because it interacts with the other three axis in ways that are not captured by any single tradeoff. For example, any statistic that separates a user’s requested / unlearned data from the rest of the model’s behaviour, by construction could potentially translate into a membership signal. Trying to maintain both auditability and leakage can in turn compromise the forget / retain performance. We make this concrete by evaluating eleven unlearning methods across three model sizes on TOFU dataset (Maini et al., 2024), scoring each on all four axes in Table 1. It can be seen that none of the methods perform on all four metrics. GradDiff (Maini et al., 2024) and RMU (Li et al., 2024) show thorough forget performance but leak membership; NPO (Zhang et al., 2024b) attains good leakage robustness while collapsing retain performance; CEU (Yang, 2025) and GradAscent (Maini et al., 2024) are not auditable. Each method compromises on either of the axis.

<table><tr><td>Method</td><td>Auditability</td><td>Leakage</td><td>Forget</td><td>Retain</td></tr><tr><td>CEU (Yang, 2025)</td><td>X</td><td></td><td>X</td><td>X</td></tr><tr><td>DPO (Zhang et al., 2024b)</td><td></td><td>X</td><td></td><td></td></tr><tr><td>GradAscent (Maini et al., 2024)</td><td>X</td><td>2</td><td>X</td><td>X</td></tr><tr><td>GradDiff (Maini et al., 2024)</td><td></td><td>X</td><td></td><td>X</td></tr><tr><td>NPO (Zhang et al., 2024b)</td><td></td><td></td><td></td><td>X</td></tr><tr><td>PDU (Entesari et al., 2026)</td><td></td><td></td><td></td><td>X</td></tr><tr><td>RMU (Li et al., 2024)</td><td></td><td>X</td><td></td><td>X</td></tr><tr><td>SatImp (Yang et al., 2025)</td><td>X</td><td>2</td><td>X</td><td>X</td></tr><tr><td>SimNPO (Fan et al., 2025)</td><td></td><td>2</td><td></td><td>X</td></tr><tr><td>UNDIAL (Dong et al., 2025)</td><td></td><td></td><td></td><td>X</td></tr><tr><td>WGA (Wang et al., 2025b)</td><td></td><td>X</td><td></td><td>X</td></tr></table>

Table 1: Comparison of several unlearning methods on TOFU-forget10B (✓: satisfies, ∼: partial, ×: fails). Detailed metric definitions are provided in Appendix C.

It is worth stating the difference between auditability and leakage mentioned earlier based on who measures what. Auditability is a claim made by the provider, who holds both the original model and the unlearned one, and who compares the requested data against the retained data. Leakage is a claim made by an adversary, who can only infer on the deployed model and compare the response to requested data against generic / synthetic data. These are different comparisons made under different access, hence a signature that is present in first might not be present in the second. Concretely, we score each example by the token-level statistics (e.g., log-likelihood) gap between the original mode and the unlearned one, averaged over response tokens, and measure auditability as the AUROC separating requested data from retained data under this score (cf. Section 3.1). The separation is a comparison the provider makes between two sets of user data.

We build on this observation and propose a two-stage procedure to alleviate these issues. First, we propose GAPWIDEN (Section 3.2), a post-unlearning method that defines an objective directly on the audit statistic, applying a hinge penalty that drives the forget set upto a fixed margin while anchoring the retain set around the likelihood of base model using a huber loss. Because of this, auditability becomes a property that is ‘constructed’ rather than one that happens to emerge, and methods that were previously un-auditable become separable without retraining from scratch while maintaining there forget / retain performance. Second, GAPVEIL (Section 3.3) operates at inference through rejection sampling (Von Neumann, 1951): candidate drafts generated from the base model are scored under the same statistic, and any generation that falls on the forget side of the threshold is rejected and redrawn with its own content tokens suppressed.

We evaluate our method on top of eleven SoTA unlearning methods and show improvements across all four axis. We benchmark our method on TOFU Maini et al. (2024) and MUSE (Shi et al., 2024) datasets on various model sizes. We show that all methods improve on auditability and on leakage robustness simultaneously while forget performance converging to the oracle (see Fig. 2). Retain performance improves substantially for almost all methods, though it remains short of oracle parity. An overview of our method is presented in Figure 1. Our contributions can be summarized as follows:

• We identify auditability as a fourth evaluation axis for machine unlearning, formalize it as an auditable separation computed by the provider and show that no existing method satisfies all four axes simultaneously, with different methods compromising on different axes

• We propose GAPWIDEN, a post-unlearning method that renders previously unauditable unlearning methods auditable by using a hinge loss to separate the audit statistics of forget set from the base model and a huber loss to anchor the statistics of retain set around the base model.

• We then propose GAPVEIL, an inference-time sampler that improves other three axes while maintaining auditability. It uses rejection sampling with base model as the drafter and uses audit statistics as criterion to reject / accept the draft response.

• We evaluate eleven unlearning methods on TOFU and MUSE datasets across three model scales along all four axes and demonstrate the effectiveness of proposed pipeline.

## 2 BACKGROUND

Setup. Consider a model with parameters $\theta _ { 0 }$ obtained from training on dataset $\mathcal { D } = \mathcal { D } _ { f } \cup \mathcal { D } _ { r }$ where $\mathcal { D } _ { f }$ is the data a user has requested for deletion (forget set) and $\mathcal { D } _ { r }$ is everything that remains (retain set). An unlearning method renders a model $\theta _ { u }$ that should behave as though it had been trained on D<sub>r</sub> alone. The ideal reference for unlearning is a retrained oracle $\theta _ { \star }$ , obtained by training from scratch on $\mathcal { D } _ { r }$ . Note that this reference is used just for evaluation and is not practical for actual use. We use $p _ { \theta } ( \mathbf { y } \mid \mathbf { x } )$ to denote model’s distribution over answer tokens $\mathbf { y } = \{ y _ { i } \}$ given query tokens $\mathbf { x } = \{ x _ { i } \}$ , and restrict all likelihood computations to answer tokens, excluding the prompt.

Prior Methods. Existing unlearning methods fall broadly into four groups, and we evaluate representatives of each group. Gradient-ascent methods like GradAscent, GradDiff (Maini et al., 2024), WGA (Wang et al., 2025b) maximize the loss on $\mathcal { D } _ { f }$ , optionally with a descent term on $\mathcal { D } _ { r }$ Preference-optimization methods like DPO, NPO (Zhang et al., 2024b), SimNPO (Fan et al., 2025) treat the forget answer as a dis-preferred response and optimize a bounded preference objective, which avoids the divergence that unbounded ascent leads to. Representation editing methods like RMU (Li et al., 2024), CEU (Yang, 2025) act on intermediate activations rather than on output likelihoods, steering the forget-set representation toward a neutral / control direction. Distribution shaping methods such as UNDIAL (Dong et al., 2025), PDU (Entesari et al., 2026), SatImp (Yang et al., 2025) modify the output distribution directly through logit adjustment or self-distillation against a re-shaped target. A detailed literature review of these methods is provided in Appendix B.

Membership Inference. Membership Inference Attacks (MIA) ask whether a given example was part of a model’s training data (Carlini et al., 2021). This is usually obtained by thresholding a loss or a likelihood ratio against a reference model, or a statistic over the lowest-probability tokens. In the unlearning setting the relevant question is slightly different. Here, an attacker wants to know whether a particular data was unlearned or removed (Shi et al., 2024; Dorna et al., 2026). A successful deletion request is evidence that the data was there to begin with, and that the user requested its removal. We therefore measure different abnormality scores to compute the leakage score (cf. Section C in Appendix for details). Broadly, abnormality scores try to capture different ways in which the output of an unlearned model can look anomalous to an ordinary output.

## 3 METHODOLOGY

## 3.1 PROBLEM FORMULATION

A typical deletion request names a (forget) set $\mathcal { D } _ { f } \subset \mathcal { D }$ and asks the provider to come up with a new model $\theta _ { u }$ from $\theta _ { 0 }$ such that $\theta _ { u }$ behaves as if it was trained on $\mathcal { D } _ { r } = \mathcal { D } \setminus \mathcal { D } _ { f }$ alone. We evaluate the results along four axes and briefly describe them below.

Forget and retain performance measure the performance of $\theta _ { u }$ on $\mathcal { D } _ { f }$ and $\mathcal { D } _ { r }$ respectively. A successful unlearning procedure should drive forget-set performance down while keeping retain-set performance close to that of $\theta _ { 0 }$ . We use novel-entity recall (NER) to measure forget and retain performance (cf. App. C.1). Further as explained above robustness to MIA or leakage measures whether the outputs of deployed model on $\mathcal { D } _ { f }$ are distinguishable from its outputs on other data. We comprehensively estimate it with an attacker, who scores the generated responses on a set of behavioural metrics and reporting resulting AUROC. We provide more details on each of the metrics in Appendix C.

Auditability measures whether the provider can demonstrate that the particular requested data $( \mathcal { D } _ { f } )$ was unlearned. For a query x and response y with token position A, define-

$$
g _ { t } ( \mathbf { x } , \mathbf { y } ) = \log p _ { \theta _ { 0 } } ( y _ { t } \mid \mathbf { x } , y _ { < t } ) - \log p _ { \theta _ { u } } ( y _ { t } \mid \mathbf { x } , y _ { < t } ) , \qquad G ( \mathbf { x } , \mathbf { y } ) = { \frac { 1 } { | A | } } \sum _ { t \in A } g _ { t } ( \mathbf { x } , \mathbf { y } ) .\tag{1}
$$

We refer $g _ { t } ( \cdot )$ as the $\cdot _ { \mathrm { g a p } } ,$ between log-probs of original model and unlearned model and $G ( \cdot )$ is the average gap over the response tokens. We report auditability as the AUROC of $G ( \cdot )$ separating $\mathcal { D } _ { f }$ from $\mathcal { D } _ { r }$ , and, for an individual request, use the decision $G ( \cdot ) \leq \tau$ under a calibrated τ. We also note that log-probs is one of the possible quantities that could be used to define the gap, we present other quantities like KL-divergence, JS-divergence, entropy in Appendix C.2.

These four metrics are not independent of each other. $\operatorname { E } . \mathbf { g } .$ , auditability requires $\theta _ { u }$ to differ measurably from $\theta _ { 0 }$ on $\mathcal { D } _ { f }$ and not on $\mathcal { D } _ { r }$ . Unlearning methods whose objective never acts on output space over forget responses leave no such difference. Conversely, methods that does create a large difference inherently changes how the model behaves on $\mathcal { D } _ { f }$ and that change is visible to anyone querying it, making it prone to leakage. While managing these two could affect the forget and retain performance. We overcome these problems using a two-stage process. textscGapWiden (Section 3.2) injects the gap where it is absent, by optimising $G ( \cdot )$ directly. GAPVEIL (Section 3.3) uses a rejection-sampling strategy that uses this induced gap to accept or reject the draft generated by the base model, $\theta _ { 0 }$ . This avoids exposing the induced gap directly during inference while utilizing it at decoding.

## 3.2 GAPWIDEN

We observe in Table 1 that some of the methods lack the audit signal. In the first-stage of our method, we propose a post-unlearning objective that explicitly maximizes the average gap for $\mathcal { D } _ { f }$ using a hinge loss with margin m and minimizes the gap for $\mathcal { D } _ { r }$ using a huber loss. We call this stage GAPWIDEN and the objective is given by-

$$
{ \mathcal { L } } = \gamma \cdot \mathbb { E } _ { ( \mathbf { x } , \mathbf { y } ) \sim { \mathcal { D } } _ { f } } \left[ \mathrm { r e l u } ( m - G ( \mathbf { x } , \mathbf { y } ) ) \right] + \alpha \cdot { \frac { 1 } { | A | } } \sum _ { t \in A } \mathbb { E } _ { ( \mathbf { x } , \mathbf { y } ) \sim { \mathcal { D } } _ { r } } \left[ \mathrm { s m o o t h } { - \ell _ { 1 } ( g _ { t } ( \mathbf { x } , \mathbf { y } ) ) } \right]\tag{2}
$$

where, gradients flow only through $\theta _ { u } .$ The first term in the above objective penalizes if the average gap for the forget set response is smaller than the margin $m ,$ , whereas, the second term anchors the gap for the retain set response around zero ensuring similar log-probs as base model. We use $m = 6$ $\gamma = \alpha = 1$ . A natural question to ask here is does GAPWIDEN preserve the unlearning properties of the original method? We find that the answer is partially yes – the forget and retain performance improve after GAPWIDEN, whereas the leakage degrades (see Figure 5). To improve the leakage we propose a sampler in next section.

Note that the GAPWIDEN objective is same as the statistics used for audit (gap between logprobs). Hence, we are injecting the audit signal into the model rather than hoping for it to emerge. We now use this property to construct a sampler that avoids leakage while maintaining other three axes.

## 3.3 GAPVEIL

After Stage-1 GAPWIDEN, we obtain a model which has audit statistic already separated for forget set and retain set. In the second stage we use this separation to build a sampler, which we call $\mathrm { G A P V E I L }$

Note that the gap $G ( \mathbf { x } , \mathbf { y } )$ in Eq. 1 is large when y carries content that $\theta _ { 0 }$ finds likely and $\theta _ { u }$ does not. We therefore treat it as an energy function and define the acceptance region:

$$
A ( \mathbf { x } ) = \{ \mathbf { y } : E ( \mathbf { x } , \mathbf { y } ) \leq 0 \} , \qquad E ( \mathbf { x } , \mathbf { y } ) = G ( \mathbf { x } , \mathbf { y } ) - \tau ,\tag{3}
$$

for a threshold τ . The distribution we want to deploy is the original model restricted to this region-

$$
\pi ( \mathbf { y } \mid \mathbf { x } ) = { \frac { 1 } { Z ( \mathbf { x } ) } } p _ { \theta _ { 0 } } ( \mathbf { y } \mid \mathbf { x } ) \mathbf { 1 } [ E ( \mathbf { x } , \mathbf { y } ) \leq 0 ] , \qquad Z ( \mathbf { x } ) = \sum _ { \mathbf { y } \in A ( \mathbf { x } ) } p _ { \theta _ { 0 } } ( \mathbf { y } \mid \mathbf { x } ) .\tag{4}
$$

In other words, the responses retain the fluency / knowledge of original model $\theta _ { 0 }$ , while responses that the audit would flag are assigned zero mass.

Eq. 4 admits a rejection sampler where, proposals $\textbf { y } \sim \ p _ { \theta _ { 0 } } ( \cdot { \mathrm { ~  ~ \vert ~ } } \textbf { x } )$ are accepted if and only if $E ( \mathbf { x } , \mathbf { y } ) \leq 0 .$ . Accepted samples are then distributed exactly as π, and the expected number of proposals is $1 / Z ( \mathbf { x } )$ . For a retain query, this is cheap because GAPWIDEN anchors the gap on $\mathcal { D } _ { r }$ around zero, so $\dot { Z } ( \mathbf { \dot { x } } ) \approx 1$ and the first proposal is almost always accepted. However, for a forget query, GAPWIDEN drives the gap past the margin m, so the responses that $\theta _ { 0 }$ is most likely to produce are precisely the ones lying outside A(x), and $Z ( \mathbf { x } )$ becomes small; this is similar to the cold-start problem in MCMC (Bandeira et al., 2023; Zhao & Frazier, 2014). Independent proposals from $p _ { \theta _ { 0 } }$ then keep repeating the rejected content, making the expected cost prohibitive.

We address this by making the proposal depend on what has already been rejected. Let $\mathcal { R } _ { k }$ denote the set of content tokens appearing in the first k rejected drafts, excluding the prompt tokens. The $\left( k { + } 1 \right)$ -th proposal is drawn from $p _ { \theta _ { 0 } }$ with the tokens in $\mathcal { R } _ { k }$ suppressed in the logits, i.e., we set $p _ { \theta _ { 0 } } [ \mathcal { R } _ { k } ] = 0$ while generating $( k { + } 1 )$ -th proposal. Therefore, each rejection removes the content that caused the rejection addressing the cold-start problem. We provide the pseudo-code for GAPVEIL in Algorithm 1 of Appendix for reference.

## 4 EXPERIMENTS AND RESULTS

## 4.1 SETUP

Benchmarks and models. We evaluate our method on standard TOFU dataset (Maini et al., 2024) across three forget-set sizes (forget01, forget05, forget10) and on both splits of MUSE (News and Books) dataset (Shi et al., 2024). We report TOFU results at three model scales using Llama3- 1B/3B /8B (Grattafiori et al., 2024), we present the 8B results in the main text whereas 1B and 3B results are deferred to Appendix. Further MUSE is evaluated with Llama2-7B (Touvron et al., 2023), smaller models are not used for MUSE as we observed that smaller models are insufficient to provide decent performance on MUSE in first place. Unless explicitly mentioned the results are presented on TOFU-forget10 with Llama3-3B. Additional results and sample generations are provided in Appendix.

Unlearning methods. We test our method on top of eleven unlearning methods as described in Section 2. All the implementations for training and evaluations use the Open-Unlearning repository (Dorna et al., 2026). Further, we apply GAPWIDEN only to the methods whose audit AUROC falls below 0.95 - which are un-auditable. Particularly, we find that mostly three methods - CEU, GradAscent and SatImp require GAPWIDEN. In Section 4.4 we illustrate this and show how GAP-WIDEN alleviates the auditability issue. Lastly, the final responses for all methods are obtained using GAPVEIL. More implementation and hyperparameter details are provided in Appendix D.

## 4.2 RESULTS ON TOFU

We report the results of our method on three splits of TOFU dataset with Llama3-8B in Table 2 across all four metrics. Particularly, we present the performance of vanilla unlearning method with and without our pipeline. We also provide these metrics averaged over all methods in the last row. It can be seen that on average our method improves the underlying unlearning methods across all the metrics.

Particularly, we observe major improvements in forget and retain scores from ∼ 0.75 to ∼ 0.90 across splits indicating better forgetting and retention after using our method. Further, we observe an average improvement in leakage of 0.11 on forget01 split whereas the improvement is around 0.25 point on forget05 and forget10 splits. Individually, we observe that the leakage score more than doubles for PDU, RMU and WGA on forget05/10 split. We apply GAPWIDEN to methods whose audit AUROC is below 0.95; it can be observed that such methods show significant boost in audit scores. E.g., the audit score consistently improves by > 0.30 points for CEU, GradAscent and SatImp.

These results elicit the effectiveness of proposed method to maintain at-par performance across all four metrics. We also provide results with Llama3-1B/3B in Table 7 and Table 8 in Appendix. We observe similar improvements across these model scales; particularly, smaller models tend to overforget and compromise on retain and leakage performance. Our method lifts these issue improving their performance across all metrics demonstrating robustness and versatility of proposed method.

<table><tr><td></td><td colspan="4">forget01</td><td colspan="4">forget05</td><td colspan="4">forget10</td></tr><tr><td>Method</td><td>Frg</td><td>Ret</td><td>Leak</td><td>Aud</td><td>Frg</td><td>Ret</td><td>Leak</td><td>Aud</td><td>Frg</td><td>Ret</td><td>Leak</td><td>Aud</td></tr><tr><td>CEU</td><td>0.936</td><td>0.254</td><td>0.652</td><td>1.000</td><td>0.022</td><td>0.986</td><td>0.882</td><td>0.680</td><td>0.910</td><td>0.110</td><td>0.948</td><td>0.797</td></tr><tr><td>+ Ours</td><td>0.953</td><td>0.972</td><td>1.000</td><td>1.000</td><td>0.927</td><td>0.972</td><td>1.000</td><td>1.000</td><td>0.887</td><td>0.972</td><td>1.000</td><td>1.000</td></tr><tr><td>DPO</td><td>0.958</td><td>0.951</td><td>0.754</td><td>1.000</td><td>0.919</td><td>0.974</td><td>0.790</td><td>1.000</td><td>0.931</td><td>0.979</td><td>0.710</td><td>1.000</td></tr><tr><td>+ Ours</td><td>0.982</td><td>0.968</td><td>1.000</td><td>1.000</td><td>0.947</td><td>0.964</td><td>1.000</td><td>1.000</td><td>0.922</td><td>0.961</td><td>0.981</td><td>1.000</td></tr><tr><td>GradAscent</td><td>0.275</td><td>0.992</td><td>1.000</td><td>0.983</td><td>0.024</td><td>0.991</td><td>0.864</td><td>0.697</td><td>0.745</td><td>0.278</td><td>0.960</td><td>0.623</td></tr><tr><td>+ Ours</td><td>0.932</td><td>0.952</td><td>1.000</td><td>0.983</td><td>0.941</td><td>0.975</td><td>0.996</td><td>0.996</td><td>0.936</td><td>0.972</td><td>0.968</td><td>1.000</td></tr><tr><td>GradDiff</td><td>0.690</td><td>0.879</td><td>1.000</td><td>0.996</td><td>0.774</td><td>0.459</td><td>0.882</td><td>0.957</td><td>0.968</td><td>0.016</td><td>0.910</td><td>0.968</td></tr><tr><td>+ Ours</td><td>0.887</td><td>0.950</td><td>1.000</td><td>0.996</td><td>0.779</td><td>0.918</td><td>1.000</td><td>0.957</td><td>0.825</td><td>0.778</td><td>0.999</td><td>0.968</td></tr><tr><td>NPO</td><td>0.908</td><td>0.316</td><td>0.998</td><td>1.000</td><td>0.896</td><td>0.415</td><td>1.000</td><td>1.000</td><td>0.872</td><td>0.604</td><td>0.866</td><td>1.000</td></tr><tr><td>+ Ours</td><td>0.972</td><td>0.972</td><td>1.000</td><td>1.000</td><td>0.953</td><td>0.972</td><td>1.000</td><td>1.000</td><td>0.920</td><td>0.982</td><td>1.000</td><td>1.000</td></tr><tr><td>PDU</td><td>0.995</td><td>0.587</td><td>0.400</td><td>1.000</td><td>1.000</td><td>0.850</td><td>0.340</td><td>1.000</td><td>1.000</td><td>0.901</td><td>0.426</td><td>1.000</td></tr><tr><td>+ Ours</td><td>0.909</td><td>0.972</td><td>1.000</td><td>1.000</td><td>0.948</td><td>0.974</td><td>0.994</td><td>1.000</td><td>0.941</td><td>0.972</td><td>0.995</td><td>1.000</td></tr><tr><td>RMU</td><td>0.602</td><td>0.985</td><td>1.000</td><td>0.972</td><td>0.991</td><td>0.985</td><td>0.448</td><td>1.000</td><td>0.969</td><td>0.993</td><td>0.476</td><td>0.999</td></tr><tr><td>+ Ours</td><td>0.976</td><td>0.980</td><td>1.000</td><td>1.000</td><td>0.951</td><td>0.962</td><td>0.989</td><td>1.000</td><td>0.941</td><td>0.973</td><td>0.982</td><td>0.999</td></tr><tr><td>SatImp</td><td>0.775</td><td>0.968</td><td>1.000</td><td>1.000</td><td>0.947</td><td>0.956</td><td>0.688</td><td>1.000</td><td>0.919</td><td>0.991</td><td>0.594</td><td>1.000</td></tr><tr><td>+ Ours</td><td>0.935</td><td>0.967</td><td>1.000</td><td>1.000</td><td>0.972</td><td>0.972</td><td>1.000</td><td>1.000</td><td>0.961</td><td>0.972</td><td>0.998</td><td>1.000</td></tr><tr><td>SimNPO</td><td>0.927</td><td>0.353</td><td>1.000</td><td>1.000</td><td>0.888</td><td>0.436</td><td>1.000</td><td>1.000</td><td>0.894</td><td>0.596</td><td>0.914</td><td>1.000</td></tr><tr><td>+ Ours</td><td>0.971</td><td>0.972</td><td>1.000</td><td>1.000</td><td>0.959</td><td>0.969</td><td>1.000</td><td>1.000</td><td>0.962</td><td>0.975</td><td>1.000</td><td>1.000</td></tr><tr><td>UNDIAL</td><td>0.663</td><td>0.889</td><td>1.000</td><td>1.000</td><td>0.832</td><td>0.155</td><td>0.826</td><td>1.000</td><td>0.759</td><td>0.489</td><td>1.000</td><td>1.000</td></tr><tr><td>+ Ours</td><td>0.929</td><td>0.960</td><td>1.000</td><td>1.000</td><td>0.946</td><td>0.972</td><td>1.000</td><td>1.000</td><td>0.948</td><td>0.976</td><td>1.000</td><td>1.000</td></tr><tr><td>WGA</td><td>0.801</td><td>0.913</td><td>0.996</td><td>1.000</td><td>0.950</td><td>0.956</td><td>0.742</td><td>1.000</td><td>0.978</td><td>0.984</td><td>0.482</td><td>1.000</td></tr><tr><td>+ Ours</td><td>0.949</td><td>0.972</td><td>1.000</td><td>1.000</td><td>0.953</td><td>0.972</td><td>1.000</td><td>1.000</td><td>0.955</td><td>0.972</td><td>0.997</td><td>1.000</td></tr><tr><td>Average</td><td>0.776</td><td>0.735</td><td>0.891</td><td>0.996</td><td>0.749</td><td>0.742</td><td>0.769</td><td>0.939</td><td>0.904</td><td>0.631</td><td>0.753</td><td>0.944</td></tr><tr><td>+ Ours</td><td>0.945</td><td>0.967</td><td>1.000</td><td>0.998</td><td>0.934</td><td>0.966</td><td>0.998</td><td>0.996</td><td>0.927</td><td>0.955</td><td>0.993</td><td>0.997</td></tr></table>

Table 2: Per-split evaluation on TOFU using Llama-8B model. Performance comparison across forget (Frg), retain (Ret), leakage (Leak), and auditability (Aud) metrics, evaluated with and without our proposed framework. For each baseline pair, the larger value is highlighted in bold; ties remain unbolded. The last row averages each column over all the methods.

## 4.3 RESULTS ON MUSE

We provide the results of our method on Books and News split of MUSE dataset in Table 3. Here we see that our approach yields significant gains in auditability across all methods on both splits, increasing average from 0.84 to 0.99 on News and from 0.81 to 0.99 on Books . Furthermore, forget performance improves for the vast majority of methods, alongside an overall average improvement in leakage.

However, these gains come at the cost of retain performance, which drops from 0.569 to 0.380 on News and a minor drop from 0.420 to 0.400 on Books. We attribute this to the nature of MUSE dataset. Specifically, the forget:retain set ratio for MUSE is 1 : 1 whereas, for TOFU this is 1 : 10 (forget10). Because of the large forget:retain ratio, there is significant overlap between entities, phrasing, and thematic elements across both sets. Because of this overlap, the accept test defined in Eq. 3 falsely triggers on retain queries, inadvertently suppressing correct model outputs. This can be avoided by identifying and filtering out key tokens, however, we leave that for future exploration.

## 4.4 EFFECT OF GAPWIDEN

As previously observed, not all unlearning methods yield effective audit signal. Across TOFU and MUSE benchmark, several methods fail to showcase auditability depending on the model scale and forget ratio. These un-auditable cases follow a pattern – each of these methods optimizes an objective that does not explicitly constrain output likelihoods on forget answers. Hence, they leave no trace to induce an observable likelihood gap. GAPWIDEN alleviates this problem, we illustrate this in Figure 3 using different audit statistic (cf. Appendix C.2). It can be seen that prior to GAPWIDEN, the forget- and retain-set score distributions overlap extensively, preventing any separation. However, after applying GAPWIDEN, the retain distribution concentrates near zero gap while the forget distribution is driven beyond the margin, enabling clean separation across all four audit statistics. We show this plot for all methods (including those that don’t need it) comprehensively in Figure 7, 8 of Appendix. It can also be seen that different audit statistics substantially impacts auditability, for instance, GradDiff achieves AUROC of 1.0 under log-probs gap, KL divergence, and JS divergence, yet drops to 0.0 under entropy, similarly for GradAscent. We show the effect of GAPWIDEN alone in Figure 5; it can be seen that while it improves auditability, it deteriorates leakage score because the audit separation could translate into a membership signal for an attacker, justifying the need for GAPVEIL. Further, we also obseverve that strong unlearning performance does not imply auditability; a method may effectively suppress target information while remaining entirely un-auditable. This justifies evaluating auditability and forget performance as independent axes.

<table><tr><td rowspan="2">Method</td><td colspan="4">MUSE-News</td><td colspan="4">MUSE-Books</td></tr><tr><td>Frg</td><td>Ret</td><td>Leak</td><td>Aud</td><td>Frg</td><td>Ret</td><td>Leak</td><td>Aud</td></tr><tr><td>CEU</td><td>0.185</td><td>0.629</td><td>0.815</td><td>0.597</td><td>0.601</td><td>0.479</td><td>0.750</td><td>0.646</td></tr><tr><td>+ Ours</td><td>0.876</td><td>0.418</td><td>0.805</td><td>0.994</td><td>0.899</td><td>0.420</td><td>0.904</td><td>0.991</td></tr><tr><td>DPO</td><td>0.487</td><td>0.453</td><td>0.885</td><td>0.820</td><td>0.808</td><td>0.370</td><td>0.845</td><td>0.658</td></tr><tr><td>+ Ours</td><td>0.934</td><td>0.348</td><td>0.764</td><td>0.993</td><td>0.879</td><td>0.419</td><td>0.988</td><td>0.995</td></tr><tr><td>GradAscent + Ours</td><td>0.420</td><td>0.606</td><td>0.702</td><td>0.681</td><td>0.748</td><td>0.339</td><td>0.846</td><td>0.574</td></tr><tr><td></td><td>0.835</td><td>0.410</td><td>0.921</td><td>0.979</td><td>0.918</td><td>0.426</td><td>0.955</td><td>0.986</td></tr><tr><td>GradDiff + Ours</td><td>0.501</td><td>0.693</td><td>0.826</td><td>0.869</td><td>0.593</td><td>0.642</td><td>0.872</td><td>0.886</td></tr><tr><td></td><td>0.861</td><td>0.372</td><td>0.778</td><td>0.989</td><td>0.882</td><td>0.381</td><td>0.933</td><td>0.992</td></tr><tr><td>NPO</td><td>0.908</td><td>0.604</td><td>0.679</td><td>0.989</td><td>0.741</td><td>0.588</td><td>0.902</td><td>0.999</td></tr><tr><td>+ Ours</td><td>0.787</td><td>0.425</td><td>0.904</td><td>0.997</td><td>0.942</td><td>0.464</td><td>0.845</td><td>1.000</td></tr><tr><td>PDU</td><td>0.901</td><td>0.201</td><td>0.760</td><td>0.837</td><td>0.965</td><td>0.154</td><td>0.858</td><td>0.958</td></tr><tr><td>+ Ours</td><td>0.918</td><td>0.313</td><td>0.878</td><td>0.998</td><td>0.925</td><td>0.355</td><td>0.898</td><td>0.997</td></tr><tr><td>RMU + Ours</td><td>0.892</td><td>0.394</td><td>0.846</td><td>0.977</td><td>0.995</td><td>0.000</td><td>0.894</td><td>0.529</td></tr><tr><td></td><td>0.913</td><td>0.362</td><td>0.931</td><td>1.000</td><td>0.896</td><td>0.395</td><td>0.819</td><td>0.997</td></tr><tr><td>SatImp + Óurs</td><td>0.591</td><td>0.706</td><td>0.720</td><td>0.969</td><td>0.769</td><td>0.501</td><td>0.834</td><td>0.964</td></tr><tr><td>SimNPO</td><td>0.916</td><td>0.379</td><td>0.783</td><td>0.994</td><td>0.982</td><td>0.385</td><td>1.000</td><td>0.998</td></tr><tr><td>+ Ours</td><td>0.491 0.927</td><td>0.622</td><td>0.712</td><td>0.748</td><td>0.670</td><td>0.504</td><td>0.759</td><td>0.839</td></tr><tr><td>UNDIAL</td><td></td><td>0.329</td><td>0.638</td><td>0.993</td><td>0.898</td><td>0.400</td><td>0.937</td><td>0.994</td></tr><tr><td>+ Ours</td><td>0.376</td><td>0.640</td><td>0.797</td><td>0.837</td><td>0.704</td><td>0.535</td><td>0.896</td><td>0.918</td></tr><tr><td></td><td>0.926</td><td>0.406</td><td>0.898</td><td>0.996</td><td>0.883</td><td>0.425</td><td>0.823</td><td>0.992</td></tr><tr><td>WGA + Ours</td><td>0.570</td><td>0.706</td><td>0.712</td><td>0.980</td><td>0.765</td><td>0.511</td><td>0.813</td><td>0.961</td></tr><tr><td></td><td>0.861</td><td>0.415</td><td>0.865</td><td>0.995</td><td>0.986</td><td>0.334</td><td>0.816</td><td>0.998</td></tr><tr><td>Average</td><td>0.575</td><td>0.569</td><td>0.769</td><td>0.846</td><td>0.760</td><td>0.420</td><td>0.843</td><td>0.812</td></tr><tr><td>+ Ours</td><td>0.887</td><td>0.380</td><td>0.833</td><td>0.993</td><td>0.917</td><td>0.400</td><td>0.902</td><td>0.994</td></tr></table>

Table 3: Per-split evaluation on MUSE using Llama-7B model. Performance comparison across forget (Frg), retain (Ret), leakage (Leak), and auditability (Aud) metrics, evaluated with and without our proposed framework. For each baseline pair, the larger value is highlighted in bold; ties remain unbolded. The last row averages each column over all the methods.

## 4.5 EFFECT OF GAPVEIL

Figure 4 illustrates the black-box abnormality score (used for leakage) computed over forget-set responses against the comparison set (cf. App. C.3), evaluated for each method before and after applying GAPVEIL. We report the attacker’s advantage (|AUROC − 0.5|) instead of raw signed AUROC. This adjustment is necessary because several unlearning methods separate the two sets in the reverse direction (e.g., CEU at 0.39 and GradAscent at 0.40). An adaptive adversary can invert their decision rule to achieve identical separation, so raw signed AUROC would be misleading. Without GAPVEIL, the attacker’s advantage spans a wide range from 0.10 to 0.37 across baselines. Using GAPVEIL revokes this advantage for all methods into a tight band between 0.11 and 0.15, effectively eliminating the variance across unlearning methods. Notably, the methods exhibiting the highest leakage see the most substantial mitigation: GradDiff drops from 0.37 to 0.12, WGA from 0.28 to 0.15, and RMU from 0.23 to 0.11. We isolate the interaction between the two stages in Fig. 9 of Appendix for methods requiring GAPWIDEN. As expected and mentioned above, GAP-WIDEN initially increases the attacker’s advantage—rising from 0.11 to 0.26 for CEU and from 0.10 to 0.30 for GradAscent. GAPVEIL then conceals this effect, restoring the score distributions to near-complete overlap. This directly demonstrates the complementary design outlined in Section 3:

![](images/fab36317495eb19a2eebb72888ba4b3c9412a7f5bccdfa4d1c0d3bf014b25056.jpg)

Figure 3: GAPWIDEN injects audit signal on un-auditable baselines. Score distributions for forget and retain sets across the three un-auditable methods, before and after GAPWIDEN across four audit statistics. GAPWIDEN collapses the retain distribution near zero while shifting forget mass beyond the decision margin. Numbers report separation AUROC.  
![](images/f1b54200aa8b456a59aca31b47dca1042742ca6edaa08e9e99d7c84ca7c95e2c.jpg)  
Figure 4: GAPVEIL suppresses behavioral leakage against black-box attackers. Distribution of black-box abnormality scores on forget versus comparison sets for unlearning baselines and under GAPVEIL. Values report attacker advantage |AUROC − 0.5| in bold and signed AUROC in parentheses.

GAPWIDEN injects the internal audit signal required by the auditor, while GAPVEIL conceals it at inference time before it surfaces during inference.

## 4.6 ABLATIONS

We illustrate the impact of GAPWIDEN margin (m), GAPVEIL threshold (τ), and sampling budget (K) using CEU (Yang, 2025) in Figure 6. We see that increasing m improves forget performance but degrades retain, as it pushes the unlearned model further from the base (auditability remains near 1.0). However, applying GAPVEIL stabilizes these scores across all margins, since generation relies entirely on the base model while the unlearned model acts as a detector. Increasing τ reduces forget performance because a higher threshold flags fewer responses for suppression, though retain and leakage remain unaffected. While a strict τ = 0 forces universal resampling and slightly harms retain, our calibrated τ<sup>∗</sup> optimally balances both. Finally, increasing K improves forget performance without degrading other metrics, as additional resampling steps provide more opportunities to suppress targeted forget tokens.

![](images/974adccffdbaaf83cd9b6bc815ea9a32e3e3d58c1dfd2cbd8f08dc90350db9ad.jpg)  
Figure 5: Effect of post-unlearning. Each bar adds GAPWIDEN and then GAPVEIL to the raw checkpoint.

![](images/10cbddf0d47de3d13889f185ddf77cc379b91d081b9fe425bafc79ccad71d08a.jpg)  
Figure 6: Effect of the GAPWIDEN margin m, the GAPVEIL threshold τ and the number of steps K for CEU.

## 5 CONCLUSION

Standard machine unlearning evaluations measure forget quality, retain performance, and robustness to membership inference. However, none of these axes directly provide verifiable evidence to the user that target data has been deleted. In this work, we propose auditability as an essential fourth axis, formalizing it as the separability between forget and retain samples under a comparative statistic derived from the original and unlearned models. Across eleven baseline methods, three scales, and two benchmarks, we comprehensively show that current unlearning techniques fail to simultaneously satisfy all four axes. While constructing a verifiable audit signal naturally risks leaking membership information, we show that this conflict can be resolved by decoupling internal audi access from external output access. Concretely, we propose GAPWIDEN that injects a clean audit margin through post-unlearning objective, while GAPVEIL suppresses this behavioral signature at inference time. Together, our framework achieves near-ideal auditability and low leakage across all evaluated baselines, while largely preserving or improving core forget and retain performance. While our method alleviates these problems, there are some limitations that we list in Appendix.

## AI USE STATEMENT

In this work, we used generative AI tools for assistance in coding / implementation, literature review, and re-writing sections of the paper. We have not used generative AI tools for ideation, synthetic data, figures, or drafting mathematical claims. We have reviewed all AI-assisted work. Particularly, the implementations were manually verified and tested for correctness, all works discovered for literature review were manually read and scrutinized. We take responsibility for the final content of this work, including text, claims or artifacts produced with the aid of generative AI.

## ETHICS STATEMENT

This work advances the practicality of regulations like “right to be forgotten” in large language models by proposing a framework for demonstrable machine unlearning. By enabling model providers to internally audit deletion requests without exposing the user to external membership inference attacks.

We acknowledge the potential dual-use and security risks inherent in our approach. Specifically, the GAPWIDEN stage intentionally embeds an identifiable likelihood gap into the model’s parameters to serve as an audit signal. If a malicious actor were to gain white-box access to either the pre unlearning $( \theta _ { 0 } )$ or post-unlearning $( \theta _ { u } )$ model weights, this engineered gap could theoretically be exploited to extract the exact data requested for deletion. Our framework mitigates this by assuming a structured access threat model, where model weights remain strictly confidential and external interaction occurs exclusively through a black-box API protected by GAPVEIL. Consequently, applying this specific auditability mechanism to open-weight models would carry severe privacy risks and requires alternative safeguards.

Finally, all empirical evaluations in this study are conducted on standard, publicly available benchmarks (e.g., TOFU, MUSE). This research does not involve new human subjects research, the collection of personally identifiable information, or the release of sensitive datasets. We hope this work encourages the development of more accountable, transparent, and privacy-respecting AI systems.

## REFERENCES

Yushi Bai, Xin Lv, Jiajie Zhang, Hongchang Lyu, Jiankai Tang, Zhidian Huang, Zhengxiao Du, Xiao Liu, Aohan Zeng, Lei Hou, et al. Longbench: A bilingual, multitask benchmark for long context understanding. In Proceedings of the 62nd annual meeting of the association for computational linguistics (volume 1: Long papers), pp. 3119–3137, 2024.

Afonso S Bandeira, Antoine Maillard, Richard Nickl, and Sven Wang. On free energy barriers in gaussian priors and failure of cold start mcmc for high-dimensional unimodal distributions. Philosophical transactions. Series A, Mathematical, physical, and engineering sciences, 381(2247): 20220150, 2023.

George-Octavian Barbulescu and Peter Triantafillou. To each (textual sequence) its own: Improving memorized-data unlearning in large language models. In International Conference on Machine Learning (ICML), 2024. URL https://arxiv.org/abs/2405.03097.

Somnath Basu Roy Chowdhury, Rahul Kidambi, Avinava Dubey, David Wang, Gokhan Mergen, Amr Ahmed, and Aranyak Mehta. Inference-time unlearning using conformal prediction. arXiv preprint arXiv:2602.03787, 2026. URL https://arxiv.org/abs/2602.03787.

Leonardo Berti, Flavio Giorgi, and Gjergji Kasneci. Emergent abilities in large language models: A survey. arXiv preprint arXiv:2503.05788, 2025.

Martin Bertran, Shuai Tang, Michael Kearns, Jamie Morgenstern, Aaron Roth, and Zhiwei Steven Wu. Reconstruction attacks on machine unlearning: Simple models are vulnerable. In Advances in Neural Information Processing Systems (NeurIPS), 2024. URL https://arxiv.org/ abs/2405.20272.

Lucas Bourtoule, Varun Chandrasekaran, Christopher A. Choquette-Choo, Hengrui Jia, Adelin Travers, Baiwu Zhang, David Lie, and Nicolas Papernot. Machine unlearning. In IEEE Symposium on Security and Privacy (S&P), 2021. URL https://arxiv.org/abs/1912. 03817.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020.

Zhiqi Bu, Xiaomeng Jin, Bhanukiran Vinzamuri, Anil Ramakrishna, Kai-Wei Chang, Volkan Cevher, and Mingyi Hong. Unlearning as multi-task optimization: A normalized gradient difference approach with an adaptive learning rate. arXiv preprint arXiv:2410.22086, 2024. URL https://arxiv.org/abs/2410.22086.

Yinzhi Cao and Junfeng Yang. Towards making systems forget with machine unlearning. In IEEE Symposium on Security and Privacy (S&P), 2015. doi: 10.1109/SP.2015.35.

Nicholas Carlini, Florian Tramer, Eric Wallace, Matthew Jagielski, Ariel Herbert-Voss, Katherine Lee, Adam Roberts, Tom Brown, Dawn Song, Ulfar Erlingsson, et al. Extracting training data from large language models. In 30th USENIX security symposium (USENIX Security 21), pp. 2633–2650, 2021.

Zora Che, Stephen Casper, Robert Kirk, Anirudh Satheesh, Stewart Slocum, Lev E. McKinney, Rohit Gandikota, Aidan Ewart, Domenic Rosati, Zichu Wu, Zikui Cai, Bilal Chughtai, Yarin Gal, Furong Huang, and Dylan Hadfield-Menell. Model tampering attacks enable more rigorous evaluations of LLM capabilities. Transactions on Machine Learning Research (TMLR), 2025. URL https://arxiv.org/abs/2502.05209.

Min Chen, Zhikun Zhang, Tianhao Wang, Michael Backes, Mathias Humbert, and Yang Zhang. When machine unlearning jeopardizes privacy. In ACM Conference on Computer and Communications Security (CCS), 2021. URL https://arxiv.org/abs/2005.02205.

Yiwei Chen, Soumyadeep Pal, Yimeng Zhang, Qing Qu, and Sijia Liu. Unlearning isn’t invisible: Detecting unlearning traces in LLMs from model outputs. In International Conference on Learning Representations (ICLR), 2026. URL https://arxiv.org/abs/2506.14003.

Qinyuan Cheng, Tianxiang Sun, Xiangyang Liu, Wenwei Zhang, Zhangyue Yin, Shimin Li, Linyang Li, Zhengfu He, Kai Chen, and Xipeng Qiu. Can ai assistants know what they don’t know?, 2024.

Yijiang River Dong, Hongzhou Lin, Mikhail Belkin, Ramon Huerta, and Ivan Vulic. Undial: Self-´ distillation with adjusted logits for robust unlearning in large language models. In Proceedings of the 2025 Conference ofthe Nations ofthe Americas Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pp. 8827–8840, 2025.

Vineeth Dorna, Anmol Mekala, Wenlong Zhao, Andrew McCallum, Zico Kolter, Zachary Lipton, and Pratyush Maini. Openunlearning: Accelerating llm unlearning via unified benchmarking of methods and metrics. Advances in Neural Information Processing Systems, 38, 2026.

Jai Doshi and Asa Cooper Stickland. Does unlearning truly unlearn? a black box evaluation of llm unlearning methods. arXiv preprint arXiv:2411.12103, 2024.

Michael Duan, Anshuman Suri, Niloofar Mireshghallah, Sewon Min, Weijia Shi, Luke Zettlemoyer, Yulia Tsvetkov, Yejin Choi, David Evans, and Hannaneh Hajishirzi. Do membership inference attacks work on large language models? In Conference on Language Modeling (COLM), 2024. URL https://arxiv.org/abs/2402.07841.

Thorsten Eisenhofer, Doreen Riepel, Varun Chandrasekaran, Esha Ghosh, Olga Ohrimenko, and Nicolas Papernot. Verifiable and provably secure machine unlearning. arXiv preprint arXiv:2210.09126, 2022. URL https://arxiv.org/abs/2210.09126.

Taha Entesari, Arman Hatami, Rinat Khaziev, Anil Ramakrishna, and Mahyar Fazlyab. Constrained entropic unlearning: A primal-dual framework for large language models. Advances in Neural Information Processing Systems, 38:166435–166474, 2026.

Chongyu Fan, Jiancheng Liu, Licong Lin, Jinghan Jia, Ruiqi Zhang, Song Mei, and Sijia Liu. Simplicity prevails: Rethinking negative preference optimization for LLM unlearning. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. URL https://openreview.net/forum?id=JbvSQm5h1l.

Ji Gao, Sanjam Garg, Mohammad Mahmoody, and Prashant Nalini Vasudevan. Deletion inference, reconstruction, and compliance in machine (un)learning. In Proceedings on Privacy Enhancing Technologies (PETS), 2022. URL https://arxiv.org/abs/2202.03460.

Antonio Ginart, Melody Y. Guan, Gregory Valiant, and James Zou. Making AI forget you: Data deletion in machine learning. In Advances in Neural Information Processing Systems (NeurIPS), 2019. URL https://arxiv.org/abs/1907.05012.

Aditya Golatkar, Alessandro Achille, and Stefano Soatto. Eternal sunshine of the spotless net: Selective forgetting in deep networks. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020. URL https://arxiv.org/abs/1911.04933.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.

Chuan Guo, Tom Goldstein, Awni Hannun, and Laurens van der Maaten. Certified data removal from machine learning models. In International Conference on Machine Learning (ICML), 2020. URL https://arxiv.org/abs/1911.03030.

Jamie Hayes, Ilia Shumailov, Christopher A. Choquette-Choo, Matthew Jagielski, George Kaissis, Milad Nasr, Sahra Ghalebikesabi, Meenatchi Sundaram Mutu Selva Annamalai, Niloofar Mireshghallah, Igor Shilov, Matthieu Meeus, Yves-Alexandre de Montjoye, Katherine Lee, Franziska Boenisch, Adam Dziedzic, and A. Feder Cooper. Exploring the limits of strong membership inference attacks on large language models. arXiv preprint arXiv:2505.18773, 2025. URL https://arxiv.org/abs/2505.18773.

Cheng-Ping Hsieh, Simeng Sun, Samuel Kriman, Shantanu Acharya, Dima Rekesh, Fei Jia, and Boris Ginsburg. RULER: What’s the real context size of your long-context language models? In First Conference on Language Modeling, 2024. URL https://openreview.net/forum? id=kIoBbc76Sy.

Hongsheng Hu, Shuo Wang, Tian Dong, and Minhui Xue. Learn what you want to unlearn: Unlearning inversion attacks against machine unlearning. In IEEE Symposium on Security and Privacy (S&P), 2024. URL https://arxiv.org/abs/2404.03233.

Shengyuan Hu, Yiwei Fu, Zhiwei Steven Wu, and Virginia Smith. Unlearning or obfuscating? jogging the memory of unlearned LLMs via benign relearning. In International Conference on Learning Representations (ICLR), 2025. URL https://arxiv.org/abs/2406.13356.

James Y. Huang, Wenxuan Zhou, Fei Wang, Fred Morstatter, Sheng Zhang, Hoifung Poon, and Muhao Chen. Offset unlearning for large language models. Transactions on Machine Learning Research (TMLR), 2025. URL https://arxiv.org/abs/2404.11045.

Joel Jang, Dongkeun Yoon, Sohee Yang, Sungmin Cha, Moontae Lee, Lajanugen Logeswaran, and Minjoon Seo. Knowledge unlearning for mitigating privacy risks in language models. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (ACL), 2023. URL https://arxiv.org/abs/2210.01504.

Yeonwoo Jang, Shariqah Hossain, Ashwin Sreevatsa, and Diogo Cruz. Prompt attacks reveal superficial knowledge removal in unlearning methods. arXiv preprint arXiv:2506.10236, 2025. URL https://arxiv.org/abs/2506.10236.

Jiabao Ji, Yujian Liu, Yang Zhang, Gaowen Liu, Ramana Rao Kompella, Sijia Liu, and Shiyu Chang. Reversing the forget-retain objectives: An efficient LLM unlearning framework from logit difference. In Advances in Neural Information Processing Systems (NeurIPS), 2024. URL https://arxiv.org/abs/2406.08607.

Hengrui Jia, Mohammad Yaghini, Christopher A. Choquette-Choo, Natalie Dullerud, Anvith Thudi, Varun Chandrasekaran, and Nicolas Papernot. Proof-of-learning: Definitions and practice. In IEEE Symposium on Security and Privacy (S&P), 2021. URL https://arxiv.org/abs/ 2103.05633.

Zhuoran Jin, Pengfei Cao, Chenhao Wang, Zhitao He, Hongbang Yuan, Jiachun Li, Yubo Chen, Kang Liu, and Jun Zhao. RWKU: Benchmarking real-world knowledge unlearning for large language models. In Advances in Neural Information Processing Systems (NeurIPS) Datasets and Benchmarks Track, 2024. URL https://arxiv.org/abs/2406.10890.

Miso Kim, Georu Lee, Seungwon Jeong, and Woojin Lee. Confess what you know: Forget-set misalignment with model knowledge in LLM unlearning, September 2026.

Pang Wei Koh and Percy Liang. Understanding black-box predictions via influence functions. In International Conference on Machine Learning (ICML), 2017. URL https://arxiv.org/ abs/1703.04730.

Vasilis Kontonis, Yuchen Zeng, Shivam Garg, Lingjiao Chen, Hao Tang, Ziyan Wang, Ahmed Awadallah, Eric Horvitz, John Langford, and Dimitris Papailiopoulos. Memento: Teaching llms to manage their own context. 2026.

Meghdad Kurmanji, Peter Triantafillou, Jamie Hayes, and Eleni Triantafillou. Towards unbounded machine unlearning. In Advances in Neural Information Processing Systems (NeurIPS), 2023. URL https://arxiv.org/abs/2302.09880.

Nathaniel Li, Alexander Pan, Anjali Gopal, Summer Yue, Daniel Berrios, Alice Gatti, Justin D. Li, Ann-Kathrin Dombrowski, Shashwat Goel, Gabriel Mukobi, Nathan Helm-Burger, Rassin Lababidi, Lennart Justen, Andrew Bo Liu, Michael Chen, Isabelle Barrass, Oliver Zhang, Xiaoyuan Zhu, Rishub Tamirisa, Bhrugu Bharathi, Ariel Herbert-Voss, Cort B Breuer, Andy Zou, Mantas Mazeika, Zifan Wang, Palash Oswal, Weiran Lin, Adam Alfred Hunt, Justin Tienken-Harder, Kevin Y. Shih, Kemper Talley, John Guan, Ian Steneker, David Campbell, Brad Jokubaitis, Steven Basart, Stephen Fitz, Ponnurangam Kumaraguru, Kallol Krishna Karmakar, Uday Tupakula, Vijay Varadharajan, Yan Shoshitaishvili, Jimmy Ba, Kevin M. Esvelt, Alexand Wang, and Dan Hendrycks. The WMDP benchmark: Measuring and reducing malicious use with unlearning. In Forty-first International Conference on Machine Learning, 2024. URL https://openreview.net/forum?id=xlr6AUDuJz.

Pingzhi Li, Jinhao Duan, Vaishnav Tadiparthi, Nakul Agarwal, Kwonjoon Lee, Ehsan Moradi Pari, Hossein Nourkhiz Mahjoub, Sijia Liu, and Tianlong Chen. Test-time unlearning via sparse autoencoder. arXiv preprint arXiv:2609.16229, 2026. URL https://arxiv.org/abs/ 2609.16229.

Chris Yuhao Liu, Yaxuan Wang, Jeffrey Flanigan, and Yang Liu. Large language model unlearning via embedding-corrupted prompts. In Advances in Neural Information Processing Systems (NeurIPS), 2024a. URL https://arxiv.org/abs/2406.07933.

Nelson F Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. Lost in the middle: How language models use long contexts. Transactions of the associationfor computational linguistics, 12:157–173, 2024b.

Xinyang Lu, Xinyuan Niu, Gregory Kang Ruey Lau, Nhung Bui, Rachael Hwee Ling Sim, John Russell Himawan, Fanyu Wen, Chuan-Sheng Foo, See-Kiong Ng, and Bryan Kian Hsiang Low. WaterDrum: Watermark-based data-centric unlearning metric. In International Conference on Learning Representations (ICLR), 2026. URL https://arxiv.org/abs/2505. 05064.

Da Ma, Lu Chen, Situo Zhang, Yuxun Miao, Su Zhu, Zhi Chen, Hongshen Xu, Hanqi Li, Shuai Fan, Lei Pan, et al. Compressing kv cache for long-context llm inference with inter-layer attention similarity. In ICASSP 2026-2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 16407–16411. IEEE, 2026.

Pratyush Maini, Zhili Feng, Avi Schwarzschild, Zachary Chase Lipton, and J Zico Kolter. TOFU: A task of fictitious unlearning for LLMs. In First Conference on Language Modeling, 2024.

Alessandro Mantelero. The EU proposal for a general data protection regulation and the roots of the ‘right to be forgotten’. Computer Law & Security Review, 29(3):229–235, 2013.

Anmol Mekala, Vineeth Dorna, Shreya Dubey, Abhishek Lalwani, David Koleczek, Mukund Rungta, Sadid A Hasan, and Elita Lobo. Alternate preference optimization for unlearning factual knowledge in large language models. In Proceedings of the 31st International Conference on Computational Linguistics, pp. 3732–3752, 2025.

Kevin Meng, David Bau, Alex Andonian, and Yonatan Belinkov. Locating and editing factual associations in GPT. In Advances in Neural Information Processing Systems (NeurIPS), 2022. URL https://arxiv.org/abs/2202.05262.

Kevin Meng, Arnab Sen Sharma, Alex Andonian, Yonatan Belinkov, and David Bau. Mass-editing memory in a transformer. In International Conference on Learning Representations (ICLR), 2023. URL https://arxiv.org/abs/2210.07229.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35: 27730–27744, 2022.

Martin Pawelczyk, Seth Neel, and Himabindu Lakkaraju. In-context unlearning: Language models as few-shot unlearners. In Proceedings of the 41st International Conference on Machine Learning (ICML), volume 235 of PMLR, 2024. URL https://arxiv.org/abs/2310.07579.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. Advances in neural information processing systems, 36:53728–53741, 2023.

Protection Regulation. Regulation (eu) 2016/679 of the european parliament and of the council. Regulation (eu), 679(2016):10–13, 2016.

Hadi Reisizadeh, Jinghan Jia, Zhiqi Bu, Bhanukiran Vinzamuri, Anil Ramakrishna, Kai-Wei Chang, Volkan Cevher, Sijia Liu, and Mingyi Hong. BLUR: A bi-level optimization approach for LLM unlearning. arXiv preprint arXiv:2506.08164, 2025. URL https://arxiv.org/abs/ 2506.08164.

Ayush Sekhari, Jayadev Acharya, Gautam Kamath, and Ananda Theertha Suresh. Remember what you want to forget: Algorithms for machine unlearning. In Advances in Neural Information Processing Systems (NeurIPS), 2021. URL https://arxiv.org/abs/2103.03279.

Weijia Shi, Jaechan Lee, Yangsibo Huang, Sadhika Malladi, Jieyu Zhao, Ari Holtzman, Daogao Liu, Luke Zettlemoyer, Noah A. Smith, and Chiyuan Zhang. Muse: Machine unlearning six-way evaluation for language models, July 2024.

Naman Deep Singh, Maximilian Muller, Francesco Croce, and Matthias Hein. Unlearning that lasts:¨ Utility-preserving, robust, and almost irreversible forgetting in LLMs, September 2025.

Vinith M. Suriyakumar, Ayush Sekhari, and Ashia Wilson. UCD: Unlearning in LLMs via contrastive decoding. arXivpreprint arXiv:2506.12097, 2025. URL https://arxiv.org/abs/ 2506.12097.

Pratiksha Thaker, Yash Maurya, Shengyuan Hu, Zhiwei Steven Wu, and Virginia Smith. Guardrail baselines for unlearning in LLMs. arXiv preprint arXiv:2403.03329, 2024. URL https:// arxiv.org/abs/2403.03329.

Anvith Thudi, Hengrui Jia, Ilia Shumailov, and Nicolas Papernot. On the necessity of auditable algorithmic definitions for machine unlearning. In 31st USENIX Security Symposium, 2022. URL https://arxiv.org/abs/2110.11891.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothee´ Lacroix, Baptiste Roziere, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Ar-\` mand Joulin, Edouard Grave, and Guillaume Lample. Llama: Open and efficient foundation language models, February 2023.

Eleni Triantafillou, Peter Kairouz, Fabian Pedregosa, Jamie Hayes, Meghdad Kurmanji, Kairan Zhao, Vincent Dumoulin, Julio Jacques Junior, Ioannis Mitliagkas, Jun Wan, Lisheng Sun Hosoya, Sergio Escalera, Gintare Karolina Dziugaite, Peter Triantafillou, and Isabelle Guyon. Are we making progress in unlearning? findings from the first NeurIPS unlearning competition. arXiv preprint arXiv:2406.09073, 2024. URL https://arxiv.org/abs/2406.09073.

John Von Neumann. Various techniques used in connection with random digits. Appl. Math Ser, 12 (36-38):3, 1951.

Nikhil Vyas, Sham Kakade, and Boaz Barak. On provable copyright protection for generative models. In Proceedings of the 40th International Conference on Machine Learning (ICML), 2023. URL https://arxiv.org/abs/2302.10870.

Lingzhi Wang, Tong Chen, Wei Yuan, Xingshan Zeng, Kam-Fai Wong, and Hongzhi Yin. KGA: A general machine unlearning framework based on knowledge gap alignment. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (ACL), 2023. URL https://arxiv.org/abs/2305.06535.

Qizhou Wang, Bo Han, Puning Yang, Jianing Zhu, Tongliang Liu, and Masashi Sugiyama. Towards effective evaluations and comparisons for LLM unlearning methods. In International Conference on Learning Representations (ICLR), 2025a. URL https://arxiv.org/abs/2406. 09179.

Qizhou Wang, Jin Zhou, Saebyeol Shin, Bo Han, Kilian Weinberger, et al. Rethinking llm unlearning objectives: A gradient perspective and go beyond. In International Conference on Learning Representations, volume 2025, pp. 61897–61931, 2025b.

Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel, Barret Zoph, Sebastian Borgeaud, Dan Yogatama, Maarten Bosma, Denny Zhou, Donald Metzler, Ed H. Chi, Tatsunori Hashimoto, Oriol Vinyals, Percy Liang, Jeff Dean, and William Fedus. Emergent abilities of large language models. Transactions on Machine Learning Research, 2022. ISSN 2835-8856. URL https://openreview.net/forum?id=yzkSU5zdwD. Survey Certification.

Bo Yang. Ce-u: Cross entropy unlearning. arXiv preprint arXiv:2503.01224, 2025.

Puning Yang, Qizhou Wang, Zhuo Huang, Tongliang Liu, Chengqi Zhang, and Bo Han. Exploring criteria of loss reweighting to enhance llm unlearning. arXiv preprint arXiv:2505.11953, 2025.

Jie Zhang, Debeshee Das, Gautam Kamath, and Florian Tramer. Membership inference attacks\` cannot prove that a model was trained on your data. arXiv preprint arXiv:2409.19798, 2024a. URL https://arxiv.org/abs/2409.19798.

Ruiqi Zhang, Licong Lin, Yu Bai, and Song Mei. Negative preference optimization: From catastrophic collapse to effective unlearning. In First Conference on Language Modeling, 2024b. URL https://openreview.net/forum?id=MXLBXjQkmb.

Zhiwei Zhang, Fali Wang, Xiaomin Li, Zongyu Wu, Xianfeng Tang, Hui Liu, Qi He, Wenpeng Yin, and Suhang Wang. Catastrophic failure of LLM unlearning via quantization. In International Conference on Learning Representations (ICLR), 2025. URL https://arxiv.org/abs/ 2410.16454.

Xiaoting Zhao and Peter I Frazier. A markov decision process analysis of the cold start problem in bayesian information filtering. arXiv preprint arXiv:1410.7852, 2014.

Jeffrey Zhou, Tianjian Lu, Swaroop Mishra, Siddhartha Brahma, Sujoy Basu, Yi Luan, Denny Zhou, and Le Hou. Instruction-following evaluation for large language models. arXiv preprint arXiv:2311.07911, 2023.

Andy Zou, Long Phan, Justin Wang, Derek Duenas, Maxwell Lin, Maksym Andriushchenko, Rowan Wang, Zico Kolter, Matt Fredrikson, and Dan Hendrycks. Improving alignment and robustness with circuit breakers. In Advances in Neural Information Processing Systems (NeurIPS), 2024. URL https://arxiv.org/abs/2406.04313.

## A APPENDIX

## B LITERATURE REVIEW

Machine unlearning originated as a framework for exact record deletion from convex or explicitly partitioned models (Cao & Yang, 2015; Ginart et al., 2019; Bourtoule et al., 2021). Early theoretical extensions introduced certified removal bounded by differential privacy and influence functions (Guo et al., 2020; Sekhari et al., 2021; Koh & Liang, 2017). However, in deep networks and large language models (LLMs), exact retraining is computationally prohibitive and guarantees based on training randomness do not provide inspectable, per-record evidence of deletion.

To contextualize the modern LLM unlearning literature, we organize prior work around four operational axes: forget performance, retain performance, privacy leakage, and auditability. While the first three are commonly studied, auditability—defined as provider-side, record-level evidence of deletion—remains largely unaddressed.

## B.1 FINE-TUNING-BASED UNLEARNING IN LLMS

Most LLM unlearning approaches rely on fine-tuning the model on a forget set $\mathcal { D } _ { f } .$ , often paired with a regularization term on a retain set $\mathcal { D } _ { r }$ (Dorna et al., 2026).

Gradient Ascent and Variants. The earliest LLM unlearning heuristic, Gradient Ascent (GradAscent), maximizes the negative log-likelihood of target sequences (Jang et al., 2023). TOFU subsequently introduced Gradient Difference (GradDiff), which balances ascent on the forget set with standard descent or KL-divergence anchors on retain data (Maini et al., 2024). Because unbounded ascent rapidly collapses model utility and degenerates outputs (Zhang et al., 2024b), numerous variants attempt to restrict the gradient’s influence. Weighted Gradient Ascent (WGA) scales the loss by token probability so updates vanish as tokens become unlikely (Wang et al., 2025b). Other approaches normalize gradients (NGDiff (Bu et al., 2024)), utilize bilevel opti mization (BLUR (Reisizadeh et al., 2025)), or apply sequence-level memorization stopping criteria (Barbulescu & Triantafillou, 2024). While these adjustments successfully limit how far the forget likelihood falls—benefiting retain performance —- they actively hinder auditability, as they prevent the formation of a decisively wide statistical gap between the original and unlearned models.

Preference Optimization and Targeted Objectives. Direct Preference Optimization (DPO) and its variants reframe unlearning as preference alignment (Rafailov et al., 2023). Negative Preference Optimization (NPO) explicitly penalizes the log-ratio between the policy and a reference model on forget sequences (Zhang et al., 2024b). When the reference model is $\theta _ { 0 } ,$ NPO directly widens the negative gap on $\mathcal { D } _ { f }$ , however, its push weakens logarithmically, and it lacks an explicit mechanism to pin the retain gap. Other techniques substitute positive targets with refusal templates (IdkDPO (Cheng et al., 2024)) or counterfactual answers (AltPO (Mekala et al., 2025)), or remove the reference model entirely (SimNPO (Fan et al., 2025)). GAPWIDEN structurally diverges from these by placing an explicit hinge margin directly on the token-averaged gap relative to $\theta _ { 0 }$ , ensuring the gap grows large enough for auditability while explicitly constraining the retain gap.

Distribution Shaping and Representation Editing. A parallel line of work shapes the target distribution or internal representations. Methods like UNDIAL (Dong et al., 2025) and PDU (Entesari et al., 2026) distil the model toward bounded target logits or constrain the retain loss. Interestingly, KGA (Wang et al., 2023) explicitly minimizes the divergence between the unlearned model and models that never saw the data, actively suppressing the likelihood gap to avoid leakage. At the representation level, RMU (Li et al., 2024) and circuit breakers (Zou et al., 2024) route sensitive activations toward noise, while model editing techniques like ROME (Meng et al., 2022) and MEMIT (Meng et al., 2023) apply localized rank-one updates. These methods focus on downstream behavior rather than the construction of an auditable parameter-level signature.

## B.2 INFERENCE-TIME UNLEARNING AND CONTROLLED DECODING

To avoid weight updates, inference-time methods manipulate the model’s forward pass, leaving $\theta _ { 0 }$ intact.

Input, Logit, and Activation Interventions. Input-side defenses rely on prompt classifiers (ECO (Liu et al., 2024a)), guardrails (Thaker et al., 2024), or adversarial in-context demonstrations (Pawelczyk et al., 2024). Logit-side methods like ULD (Ji et al., 2024), UCD (Suriyakumar et al., 2025), δ-unlearning (Huang et al., 2025) dynamically compose $\theta _ { 0 }$ with auxiliary expert or antiexpert models at decoding time. Activation-side methods like ARIA (Li et al., 2026)) gate queries and steer the residual stream when sensitive concepts are detected. Crucially, because these methods never produce an explicitly unlearned weight matrix, they offer no parameter-level evidence of deletion for a provider to audit.

Output Filtering and Constrained Sampling. Output-side mechanisms filter or resample generated text. Conformal unlearning uses an external LLM judge to score and reject unsafe drafts within a calibrated budget (Basu Roy Chowdhury et al., 2026). In the context of copyright takedowns, CPk (Vyas et al., 2023) enforces k-near access-freeness by sampling from a base model and rejecting sequences that exceed a log-likelihood ratio threshold relative to a safe model. GAPVEIL shares the accept–reject paradigm of CP-k and conformal unlearning, but introduces two critical distinctions: it utilizes the explicitly unlearned $\theta _ { u }$ as its reference to verify the gap, and it features an adaptive proposal distribution that penalizes rejected tokens to stay within a small, fixed sampling budget.

## B.3 LEAKAGE, OVER-UNLEARNING, AND SUPERFICIAL SUPPRESSION

Unlearning inherently introduces new privacy risks. Model differencing attacks -— where an adversary queries both $\theta _ { 0 }$ and $\theta _ { u } \mathrm { - c a n }$ reconstruct deleted sequences by exploiting the exact likelihood gap we rely on for auditability (Chen et al., 2021; Gao et al., 2022). With white-box access, weight differences expose even more granular training data (Hu et al., 2024; Bertran et al., 2024).

Even when querying a single deployed model, unlearning leaves behavioral traces. Output anomalies, such as templated refusals or extreme perplexity spikes, serve as high-confidence membership signals for the forget set (Golatkar et al., 2020; Chen et al., 2026). This phenomenon, often termed the Streisand effect, explicitly highlights the tension at the core of our work: a gap large enough to prove deletion is also large enough to leak membership if exposed to the user (Kurmanji et al., 2023).

Furthermore, standard fine-tuning often achieves only superficial output suppression rather than genuine knowledge removal. Numerous studies demonstrate that forgotten facts can be readily recovered via adversarial prompts (Jang et al., 2025), translation (Doshi & Stickland, 2024), hyperparameter shifts (Zhang et al., 2025), or minimal retraining on out-of-domain text (Hu et al., 2025; Che et al., 2025). This reinforces the necessity of our two-stage philosophy: GAPWIDEN embeds deletion in the parameters (resisting superficial recovery), while GAPVEIL masks the resulting behavioral signature from external queries (preventing leakage).

## B.4 VERIFICATION, AUDITING, AND BENCHMARK COVERAGE

Regulatory frameworks, such as Article 17 of the GDPR, entitle users to data erasure and require providers to demonstrate compliance (Regulation, 2016; Mantelero, 2013). Historically, this demonstration has been difficult. Cryptographic proofs of learning and zero-knowledge SNARKs (Jia et al., 2021; Eisenhofer et al., 2022) attest to algorithmic execution, but parameter-level gradient histories can be forged by reordering data (Thudi et al., 2022). Conversely, external verification via Membership Inference Attacks (MIA) provides weak, confounded signals on LLMs and lacks a valid statistical null hypothesis (Duan et al., 2024; Hayes et al., 2025; Zhang et al., 2024a).

This brings us to the landscape of current unlearning benchmarks. A review of standard evaluations reveals that no existing benchmark collectively satisfies all four axes of unlearning. For example, TOFU (Maini et al., 2024) and RWKU (Jin et al., 2024) rigorously evaluate forget and retain performance, but offer only partial, single-model coverage of privacy leakage. WMDP (Li et al., 2024) focuses entirely on forget and retain metrics for hazardous knowledge, completely omitting leakage and auditability. Benchmarks like MUSE (Shi et al., 2024), OpenUnlearning (Dorna et al., 2026), and the NeurIPS Unlearning competition 2023 (Triantafillou et al., 2024) successfully integrate leakage evaluations (such as PrivLeak) by measuring distinguishability relative to a retrained oracle. $\mathbf { A }$ few approaches brush against auditability: UWC (Wang et al., 2025a) utilizes KL divergence to $\theta _ { 0 }$ as a holistic dataset metric, and WaterDrum (Lu et al., 2026) allows data owners to verify deletion via planted watermarks.

Crucially, however, none of these benchmarks evaluate auditability as an actionable, record-level metric accessible to the provider post-training. Our framework addresses this systemic gap. By acknowledging that evidence of deletion inherently mimics a membership signal, we resolve the tension through asymmetric access: GAPWIDEN optimizes the audit statistic in the confidential weights, and GAPVEIL suppresses it at decoding time to ensure downstream privacy.

## C DETAILED METRIC DEFINITIONS

Here we describe how we measure each of the four axes introduced in Section 3.1: forget performance, retain performance, auditability, and leakage. The base model is denoted as $\theta _ { 0 }$ , the unlearned model is denoted as $\theta _ { u } .$ , and the retrained oracle is $\theta _ { \star }$ . The forget set $\mathcal { D } _ { f }$ and retain set $\mathcal { D } _ { r }$ are as in Section 2. For a query x, we use y to denote its ground-truth answer and $\hat { \mathbf { y } }$ to denote the response that model $\theta _ { 0 }$ generates.

## C.1 FORGET AND RETAIN RECALL

We score forget and retain performance with a novel-entity recall in place of ROUGE-L recall used by Maini et al. (2024). This is because ROUGE-L can fall short of measuring what was forgotten because it rewards surface (word/token) overlap rather than the presence of the fact. A response that echoes generic or question-derived wording can score high without recovering the fact. A response that recovers the fact in different words can score low. Novel-entity recall is designed to avoid such cases. It checks whether the content words of the ground-truth answer (y) reappear in the response (yˆ), and it ignores content already present in the question.

For a piece of text t, Ent(t) denotes its set of content entities. To obtain this, we first filter out any word that has less than three characters, common function words (e.g., ‘the’, ‘this’, ‘then’, etc) , keeps numeric substrings such as years, counts, and amounts. Formally,

$$
\operatorname { E n t } ( t ) ~ = ~ \{ ~ w \in { \mathrm { w o r d s } } ( t ) : w \not \in { \mathcal { S } } , | w | > 2 \} ~ \cup ~ \operatorname { N u m } ( t ) ,\tag{5}
$$

where $s$ is a fixed stop-word list and $\mathrm { N u m } ( t )$ is the set of numeric substrings of t.

TOFU answers often start by repeating the question’s own wording before adding new information. For example, the question “What genre of books does Carmen Montenegro predominantly write in?” is answered with “Carmen Montenegro predominantly writes in the genre of Historical Fiction.” We call the content entities of the answer that are not already in the question its novel ground-truth entities,

$$
\mathcal { N } ( \mathbf { x } , \mathbf { y } ) = \mathrm { E n t } ( \mathbf { y } ) \setminus \mathrm { E n t } ( \mathbf { x } ) .\tag{6}
$$

This removes entities that only echo the question, such as the author name.

The novel-entity recall of a response $\hat { y }$ against $( x , y )$ is

$$
\operatorname { R e c a l l } ( \mathbf { x } , \mathbf { y } , \hat { \mathbf { y } } ) = \frac { \big | \mathcal { N } ( \mathbf { x } , \mathbf { y } ) \cap \operatorname { E n t } ( \hat { \mathbf { y } } ) \big | } { \operatorname* { m a x } \big ( | \mathcal { N } ( \mathbf { x } , \mathbf { y } ) | , 1 \big ) } ,\tag{7}
$$

i.e. the fraction of novel ground-truth content entities that reappear in the response. We report two aggregates of Eq. 7:

• Forget score: $1 - \operatorname { R e c a l l } ( \mathbf { x } , \mathbf { y } , \hat { \mathbf { y } } )$ averaged over $( \mathbf { x } , \mathbf { y } ) \in \mathcal { D } _ { f }$ with $\hat { \mathbf { y } }$ generated by $\theta _ { u }$ Higher values mean more forgetting.

• Retain score: Recal $\mathbf { l } ( \mathbf { x } , \mathbf { y } , \hat { \mathbf { y } } )$ averaged over subset of $\mathcal { D } _ { r }$ . Higher is better.

## C.2 AUDITABILITY

Auditability measures whether the provider can separate forget-set examples from retain-set examples using an audit statistic computed from both $\theta _ { 0 }$ and $\theta _ { u } .$ . Let $P _ { \theta , t } = p _ { \theta } ( \cdot \mid \mathbf { x } , y _ { < t } )$ denote the next-token distribution of the model at answer position t. Let T be the number of answer tokens in y. We use four token-level energy functions. Each is averaged over the $T$ answer positions to give one score per sample:

Log-probability gap

$$
e _ { \mathrm { l p } } ( \mathbf x , \mathbf y ) \ = \ \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \Big [ \log P _ { \theta _ { 0 } , t } ( y _ { t } ) - \log P _ { \theta _ { u } , t } ( y _ { t } ) \Big ] .\tag{8}
$$

KL divergence

$$
e _ { \mathrm { k l } } ( \mathbf x , \mathbf y ) = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } D _ { \mathrm { K L } } \big ( P _ { \theta _ { 0 } , t } \| P _ { \theta _ { u } , t } \big ) .\tag{9}
$$

JS divergence

$$
e _ { \mathrm { j s } } ( \mathbf { x } , \mathbf { y } ) \ = \ { \frac { 1 } { T } } \sum _ { t = 1 } ^ { T } { \frac { 1 } { 2 } } D _ { \mathrm { K L } } { \big ( } P _ { \theta _ { 0 } , t } \| M _ { t } { \big ) } + { \textstyle { \frac { 1 } { 2 } } } D _ { \mathrm { K L } } { \big ( } P _ { \theta _ { u } , t } \| M _ { t } { \big ) } , \qquad M _ { t } = { \textstyle { \frac { 1 } { 2 } } } { \big ( } P _ { \theta _ { 0 } , t } + P _ { \theta _ { u } , t } { \big ) } .\tag{10}
$$

Entropy gap

$$
e _ { \mathrm { e n t } } ( \mathbf { x } , \mathbf { y } ) = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \Big [ H ( P _ { \theta _ { u } , t } ) - H ( P _ { \theta _ { 0 } , t } ) \Big ] , \qquad H ( P ) = - \sum _ { v } P ( v ) \log P ( v ) .\tag{11}
$$

Each energy is oriented so that larger values are expected on forget data, since $\theta _ { u }$ diverges from $\theta _ { 0 }$ on the data it was asked to forget. This is not guaranteed for every method (for example, the entropy gap is inverted for GradDiff), which is why we select the best energy per method to verify auditability. This same metric is then later used in GAPVEIL for sampling.

Given the energy function e, we treat forget/retain separation as a binary classification problem with forget examples as the positive class. We measure it by the area under the ROC curve (AUROC),

$$
\mathrm { A U R O C } ( e ) \ : = \ : \mathrm { P r } \left[ e ( \mathbf { x } _ { f } , \mathbf { y } _ { f } ) > e ( \mathbf { x } _ { r } , \mathbf { y } _ { r } ) \right] , \qquad ( \mathbf { x } _ { f } , \mathbf { y } _ { f } ) \sim \mathcal { D } _ { f } , \ : ( \mathbf { x } _ { r } , \mathbf { y } _ { r } ) \sim \mathcal { D } _ { r } ,\tag{12}
$$

which is the probability that a randomly drawn forget-set score exceeds a randomly drawn retainset score (ties count as 0.5). For each of the eleven unlearning methods, we report auditability as $\operatorname { A U R O C } ( e ^ { \star } )$ under its best energy function,

$$
e ^ { \star } ~ { = } ~ \operatorname * { a r g m a x } _ { e \in \{ e _ { \mathrm { l p } } , e _ { \mathrm { k l } } , e _ { \mathrm { j s } } , e _ { \mathrm { e n t } } \} } \mathrm { A U R O C } ( e ) ,\tag{13}
$$

chosen separately for each method, model scale, and forget split. An AUROC of 1 means a single threshold on $e ^ { \star }$ perfectly separates the two sets in the evaluation data. An AUROC of 0.5 means $e ^ { \star }$ does not separate them.

## C.3 MEMBERSHIP-INFERENCE ATTACKS (LEAKAGE)

Leakage measures whether a black-box adversary can tell that a query belonged to the forget set. The adversary observes only the text yˆ produced by the deployed model $\theta _ { u }$ and has no access to either model weights. It is asking whether a record was unlearned, not just trained on. To evaluate this, we first curate a pool of queries separate from $\mathcal { D } _ { r }$ and call it negative class. We denote this pool by $\mathcal { D } _ { n }$ . It is built by entity-swapping forget-set questions onto synthetic authors that are absent from $\dot { \mathcal { D } } = \mathcal { D } _ { f } \cup \mathcal { D } _ { r }$ . This keeps the question style and length but replaces the author name and profile attributes.

Composite abnormality score We combine four aspects of each response ${ \hat { \mathbf { y } } } .$ Each catches a different way an unlearned model output can look anomalous compared to an ordinary answer:

• Refusal (embedding): the maximum cosine similarity between $\hat { \mathbf { y } }$ and a fixed bank $\mathrm { o f } \ ^ { 6 6 } \mathbb { T }$ don’t know”-style templates, $\tau$ with $| \mathcal { T } | = 1 0 0$ , using sentence embeddings $\phi ( \cdot )$ from a $\scriptstyle 1 1 - \mathtt { M i n i L M - L } 6 - \mathtt { v } 2 ^ { 1 }$

$$
s _ { \mathrm { e m b e d } } ( \hat { \mathbf { y } } ) = \operatorname* { m a x } _ { \tau \in \mathcal { T } } \frac { \phi ( \hat { \mathbf { y } } ) \cdot \phi ( \tau ) } { \| \phi ( \hat { \mathbf { y } } ) \| \| \phi ( \tau ) \| } .\tag{14}
$$

• Refusal (NLI): a Natural Language Inference (NLI) cross-encoder (cross-encoder $\cdot / \mathrm { n 1 i - d e b e r t a - v 3 - s m a 1 1 ^ { 2 } } )$ takes the response $\hat { \mathbf { y } }$ as the premise and $h = { } ^ { 6 6 }  { \mathrm { T } } ]$ he speaker does not know.” as the hypothesis, and outputs probabilities for contradiction, entailment, and neutral. The score is the probability that $\hat { \mathbf { y } }$ does not contradict $h ,$

$$
s _ { \mathrm { n l i } } ( \hat { \mathbf { y } } ) \ = \ 1 - p _ { \mathrm { c o n t r a } } ( \hat { \mathbf { y } } , h ) .\tag{15}
$$

A high score means the response is compatible with the speaker not knowing, i.e. it reads like a refusal.

• Structural refusal: a heuristic for detecting $^ { 6 6 } \mathbb { T }$ don’t know”-style answers. It combines two cues. The first is a first-person opening: $f ( \hat { \mathbf { v } } y ) = 1$ if the response starts with $I , I ^ { \prime } m ,$ I’ll, I’ve, or $I \gamma _ { d , }$ and 0 otherwise. The second is the number of negation words $n _ { \mathrm { n e g } } ( \hat { \mathbf { y } } )$ in the response (not, no, $n ^ { \prime } t ,$ cannot, can’t, don $\ ' _ { t , \astrosun }$ doesn’t, isn’t, aren’t, won $\ddots$ wouldn’t, couldn’t).

$$
s _ { \mathrm { s t r u c t } } ( \hat { \mathbf { y } } ) \ = \ \operatorname* { m a x } \big ( 0 . 6 \ : f ( \hat { \mathbf { y } } ) , \ : \operatorname* { m i n } ( n _ { \mathrm { n e g } } ( \hat { \mathbf { y } } ) / 2 , 1 ) \big ) .\tag{16}
$$

A first-person opening alone scores 0.6 and two or more negation words score 1. Responses under three characters get $s _ { \mathrm { s t r u c t } } = 1$ . These constants are hand-set. On TOFU forget10 this aspect gives the base model and the retrained oracle an AUROC of about 0.5. It detects DPO, which is trained to prefer “I don’t know”-style answers on the forget set, with AUROC 0.61, 0.73, and 0.88 at 1B, 3B, and 8B. This aspect stays near chance for most other methods, hence we use this specifically for DPO.

• Degeneration: the highest repetition rate among the token $2 \cdot , \ 3 \cdot$ , and 4-grams of the response, which flags looping text.

$$
s _ { \mathrm { d e g } } ( \hat { \bf y } ) = \operatorname* { m a x } _ { n \in \{ 2 , 3 , 4 \} } \left( 1 - \frac { \# { \mathrm { u n i q u e } } n \mathrm { - g r a m s ~ i n ~ } \hat { \bf y } } { \# n \mathrm { - g r a m s ~ i n ~ } \hat { \bf y } } \right) .\tag{17}
$$

Responses under 10 characters are assigned $s _ { \mathrm { d e g } } = 1$ , and responses too short to form a 4-gram (under four tokens) are assigned $s _ { \mathrm { d e g } } = 0$ . On TOFU forget10 this aspect detects WGA well at every scale (AUROC 0.79 to 0.98) and GradDiff (above 0.97).

These four channels are on different scales, so each score is converted to a percentile among all forget and novel responses on the same channel,

$$
\hat { F } _ { c } \big ( s _ { c } ( \hat { \mathbf { y } } _ { i } ) \big ) \ = \ \mathrm { r a n k } \big ( s _ { c } ( \hat { \mathbf { y } } _ { i } ) \big ) / N \ \in \ ( 0 , 1 ] ,\tag{18}
$$

where $N$ is the number of those responses. The four percentiles are then combined by taking their maximum,

$$
A ( \hat { \mathbf { y } } ) ~ = \operatorname* { m a x } _ { \substack { c \in \mathrm { \{ e m b e d , n l i , d e g , s t r u c t \} } } } \hat { F } _ { c } \big ( s _ { c } ( \hat { \mathbf { y } } ) \big ) ~ \in ~ [ 0 , 1 ] .\tag{19}
$$

We take the max rather than the mean, so that a response needs to look anomalous on only one channel to score high.

Leakage AUROC. We compute it as in Eq. 12, with forget examples as the positive class, the abnormality score A in place of $e ,$ and $\mathcal { D } _ { n }$ in place of $\mathcal { D } _ { r }$ :

$$
\mathrm { A U R O C } _ { \mathrm { l e a k } } ( \theta _ { u } ) = \mathrm { P r } \left[ A ( \hat { \mathbf { y } } _ { f } ) > A ( \hat { \mathbf { y } } _ { n } ) \right] ,\tag{20}
$$

where $( \mathbf { x } _ { f } , \hat { \mathbf { y } } _ { f } ) \sim \mathcal { D } _ { f } , ( \mathbf { x } _ { n } , \hat { \mathbf { y } } _ { n } ) \sim \mathcal { D } _ { n }$ are the query-response pair of $\theta _ { u } .$ . The retrained oracle $\theta _ { \star }$ never sees ${ \mathcal { D } } _ { f } ,$ so it serves as a null control. Its $\mathrm { A U R O C _ { \mathrm { l e a k } } }$ should be 0.5. We verify this with a 95% bootstrap confidence interval (1000 resamples), which contains 0.5 on TOFU forget10 at all three model scales. We therefore fix the oracle at 0.5 and report leakage as an MIA score,

$$
\mathrm { M I A } ( \theta _ { u } ) \ : = \ : \operatorname* { m i n } \big ( 2 ( 1 - \mathrm { A U R O C } _ { \mathrm { l e a k } } ( \theta _ { u } ) ) , \ : 1 \big ) ,\tag{21}
$$

which is 1 when the attack is at or below chance and 0 when it perfectly separates forget from novel queries.

## D IMPLEMENTATION DETAILS

All our training and evaluation pipeline is built on the Open-Unlearning framework (Dorna et al., 2026). On TOFU (Maini et al., 2024) we unlearn Llama-3.2-1B/3B-Instruct and Llama-3.1-8B-Instruct (Touvron et al., 2023; Grattafiori et al., 2024). We use the forget01, forget05, and forget10 splits. Each split has its own matching retain and holdout set. On MUSE (Shi et al., 2024) we unlearn Llama- $- 2 - 7 \mathrm { b - h } \mathrm { f }$ on the News and Books splits. The released checkpoints for MUSE are trained on raw corpus text rather than the question-answer format. So before unlearning, following Dorna et al. (2026) we finetune the released target checkpoint on that split’s forget and retain question-answer pairs. We finetune the retrained oracle the same way, but only on the retain pairs.

## D.1 UNLEARNING BASELINES

The commong hyperparameters for most of the eleven unlearning methods are as follows: AdamW, lr $1 0 ^ { - 5 }$ , effective batch 16, 1-epoch warmup, weight decay 0.01, in bf16. The method specific hyperparameters are listed in Table 4. Since, GradAscent and CEU have unbounded objectives, training them for full 10 epochs collapses model utility; hence, we stop them after 1–2.5 epochs instead.

<table><tr><td>Method</td><td>lr</td><td>Epochs</td><td>Objective coefficients</td></tr><tr><td>GradAscent</td><td> $1 0 ^ { - 5 }$ </td><td>2.5</td><td>– (unconstrained ascent on  $\mathcal { D } _ { f } )$ </td></tr><tr><td>GradDiff</td><td> $1 0 ^ { - 5 }$ </td><td>7</td><td> $\gamma { = } 1 . 0 , \alpha { = } 1 . 0$ </td></tr><tr><td>NPO</td><td> $5 \times 1 0 ^ { - 5 }$ </td><td>10</td><td> $\beta { = } 0 . 1 , ~ \gamma { = } 1 . 0 , ~ \alpha { = } 1 . 0$ </td></tr><tr><td>DPO</td><td> $1 0 ^ { - 5 }$ </td><td>10</td><td> $\beta { = } 0 . 1 , \ \gamma { = } 1 . 0 , \ \alpha { = } 1 . 0 ;$  IDK preference pairs</td></tr><tr><td>SimNPO</td><td> $5 \times 1 0 ^ { - 5 }$ </td><td>10</td><td> $\beta { = } 2 . 5 , \ \gamma { = } 1 . 0 , \ \alpha { = } 0 . 5$ </td></tr><tr><td>RMU</td><td> $1 0 ^ { - 5 }$ </td><td>10</td><td>steering coeff. 2, α=1, layer-7 activations</td></tr><tr><td>UNDIAL</td><td> $1 0 ^ { - 5 }$ </td><td>10</td><td> $\beta { = } 3 0 . 0 , \ \gamma { = } 1 . 0 , \ \alpha { = } 0 . 0$ </td></tr><tr><td>CEU</td><td> $1 0 ^ { - 5 }$ </td><td>1</td><td>suppress true token from position 2 on</td></tr><tr><td>SatImp</td><td> $1 0 ^ { - 5 }$ </td><td>10</td><td> $\beta _ { 1 } { = } 4 . 0 , ~ \beta _ { 2 } { = } 0 . 1 , ~ \gamma { = } 1 . 0 , ~ \alpha { = } 0 . 5$ </td></tr><tr><td>WGA</td><td> $1 0 ^ { - 5 }$ </td><td>10</td><td> $\beta { = } 5 . 0 , \ \alpha { = } 0 . 1 , \ \gamma { = } 1 . 0$ </td></tr><tr><td>PDU</td><td> $1 0 ^ { - 5 }$ </td><td>10</td><td> $\varepsilon { = } 1 . 5 ,$  dual step 1, dual warmup 3 epochs, γ=1.0, α=50</td></tr></table>

Table 4: Method specific hyperparameters used for obtaining the unlearned model.

For MUSE, we use the hyperparameters: AdamW, lr $1 0 ^ { - 5 }$ , effective batch 32, 1-epoch warmup, weight decay 0.01, 10 epochs, in bf16. The methods specific hyperparameters as listed in Table 5. The remaining methods, GradAscent, GradDiff, DPO, RMU, and UNDIAL, use the same hparams as in Table 4.

## D.2 GAPWIDEN

GAPWIDEN is applied only to the methods whose raw auditability score falls below the split’s threshold (Section 4.1). For example, in TOFU, such methods are CEU, GradAscent, and SatImp.

Method lr Epochs Objective coefficients   
NPO $3 \times 1 0 ^ { - 5 }$ 10 $\beta { = } 0 . 1 , \ \gamma { = } 1 . 0 , \ \alpha { = } 1 . 0$ (recipe default)   
CEU $4 \times 1 0 ^ { - 6 }$ 3 suppress true token from position 2 on   
SimNPO $1 0 ^ { - 5 }$ 10 $\beta { = } 0 . 7 , \ \gamma { = } 1 . 0 , \ \alpha { = } 0 . 1$   
RMU $1 0 ^ { - 3 }$ 10 steering coeff. 6.5; layers 5–7 down proj only   
SatImp $1 0 ^ { - 5 }$ 10 $\beta _ { 1 } { = } 4 . 0 , ~ \beta _ { 2 } { = } 0 . 1 , ~ \gamma { = } 1 . 0 , ~ \alpha { = } 0 . 1$   
WGA $1 0 ^ { - 5 }$ 10 $\beta { = } 5 . 0 , \ \alpha { = } 0 . 1 , \ \gamma { = } 1 . 0$   
PDU $1 0 ^ { - 5 }$ 10 $\varepsilon { = } 1 . 5 ,$ dual step 1, dual warmup 3 epochs, $\gamma { = } 1 . 0 ,$ α=50  
Table 5: Method specific hyperaparameters used for obtaining unlearned model on MUSE.

Algorithm 1 GAPVEIL sampling   
Require: query x, models $\theta _ { 0 } , \theta _ { u } ,$ threshold τ, budget K   
1: y ← sample from $p _ { \theta _ { 0 } } ( \cdot \mid \mathbf { x } )$ ${ \mathcal { R } } \gets \emptyset$   
2: if $E ( \mathbf { x } , \mathbf { y } ) \leq 0$ then   
3: return y ▷ accepted on the first proposal   
4: end if   
5: $\mathbf { y } ^ { \star }  \mathbf { y }$   
6: for k = 1 to K do   
7: $\mathcal { R }  \mathcal { R } \cup$ CONTENT(y) ▷ accumulate rejected material   
8: y ← sample from $p _ { \theta _ { 0 } } ( \cdot \mid \mathbf { x } )$ with tokens in R suppressed   
9: if $E ( \mathbf { x } , \mathbf { y } ) \leq 0$ then   
10: return y   
11: end if   
12: if $E ( \mathbf { x } , \mathbf { y } ) < E ( \mathbf { x } , \mathbf { y } ^ { \star } )$ then   
13: $\mathbf { y } ^ { \star }  \mathbf { y }$   
14: end if   
15: end for   
16: return $\mathbf { y } ^ { \star }$ ▷ budget exhausted

The GAPWIDEN objective (Eq. 2) uses margin $m = 6 . 0$ with $\gamma = \alpha = 1 . 0$ and a Huber loss to anchor the retain set. We train against the frozen base model $\theta _ { 0 }$ at lr $1 0 ^ { - 5 }$ for 8 epochs, with the same effective batch as the baseline configuration. A few methods need more than 8 epochs to clear the threshold; we continue training them in 8-epoch increments, occasionally past 16.

On MUSE, GAPWIDEN uses the same objective and lr against base model of each split, with effective batch 32. It is applied to all eleven methods, in repeated 8-epoch rounds.

## D.3 GAPVEIL

Table 6 gives the sampler configuration used for every reported result (Algorithm 1). The suppression set $\mathcal { R } _ { k }$ suppresses every token of a rejected draft, except tokens shared with the prompt, tokenizer special tokens, and a fixed list of common function words and punctuation. This removes the content that triggered rejection, without banning the grammar the next redraft still needs.

Sampling steps, K 3   
Energy, e<sup>⋆</sup> best of {log-prob, KL, JS, entropy} gap, chosen per method   
Threshold, τ arg max 0.85·TPR + 0.15·TNR on forget vs. retain energy scores   
Draft sampling temperature 1.0, do sample=True  
Table 6: GAPVEIL sampling configuration, fixed across all reported results.

<table><tr><td></td><td colspan="4">forget01</td><td colspan="4">forget05</td><td colspan="4">forget10</td></tr><tr><td>Method</td><td>Frg</td><td>Ret</td><td>Leak</td><td>Aud</td><td>Frg</td><td>Ret</td><td>Leak</td><td>Aud</td><td>Frg</td><td>Ret</td><td>Leak</td><td>Aud</td></tr><tr><td>CEU</td><td>0.892</td><td>0.234</td><td>0.794</td><td>1.000</td><td>0.726</td><td>0.298</td><td>0.938</td><td>0.808</td><td>1.000</td><td>0.002</td><td>0.952</td><td>0.536</td></tr><tr><td>+ Ours</td><td>0.944</td><td>0.471</td><td>1.000</td><td>1.000</td><td>0.913</td><td>0.463</td><td>0.896</td><td>1.000</td><td>0.930</td><td>0.513</td><td>1.000</td><td>1.000</td></tr><tr><td>DPO</td><td>0.990</td><td>0.708</td><td>0.636</td><td>1.000</td><td>0.942</td><td>0.563</td><td>0.606</td><td>0.993</td><td>0.984</td><td>0.151</td><td>1.000</td><td>0.761</td></tr><tr><td>+ Ours</td><td>0.891</td><td>0.513</td><td>1.000</td><td>1.000</td><td>0.900</td><td>0.439</td><td>1.000</td><td>0.993</td><td>0.906</td><td>0.513</td><td>1.000</td><td>1.000</td></tr><tr><td>GradAscent</td><td>0.514</td><td>0.723</td><td>1.000</td><td>0.925</td><td>0.535</td><td>0.506</td><td>0.886</td><td>0.696</td><td>1.000</td><td>0.000</td><td>0.928</td><td>0.555</td></tr><tr><td>+ Ours</td><td>0.924</td><td>0.347</td><td>0.939</td><td>0.996</td><td>0.915</td><td>0.412</td><td>1.000</td><td>0.998</td><td>0.907</td><td>0.520</td><td>1.000</td><td>1.000</td></tr><tr><td>GradDiff</td><td>1.000</td><td>0.382</td><td>0.698</td><td>1.000</td><td>0.942</td><td>0.065</td><td>0.924</td><td>0.821</td><td>0.996</td><td>0.168</td><td>0.444</td><td>1.000</td></tr><tr><td>+ Ours</td><td>0.923</td><td>0.536</td><td>0.963</td><td>1.000</td><td>0.877</td><td>0.218</td><td>0.946</td><td>0.821</td><td>0.879</td><td>0.488</td><td>1.000</td><td>1.000</td></tr><tr><td>NPO</td><td>0.903</td><td>0.530</td><td>0.982</td><td>1.000</td><td>0.898</td><td>0.203</td><td>1.000</td><td>0.999</td><td>0.915</td><td>0.191</td><td>1.000</td><td>0.974</td></tr><tr><td>+ Ours</td><td>0.882</td><td>0.513</td><td>1.000</td><td>1.000</td><td>0.906</td><td>0.517</td><td>1.000</td><td>0.999</td><td>0.874</td><td>0.527</td><td>1.000</td><td>0.974</td></tr><tr><td>PDU</td><td>1.000</td><td>0.494</td><td>0.228</td><td>1.000</td><td>1.000</td><td>0.417</td><td>0.818</td><td>1.000</td><td>0.793</td><td>0.371</td><td>1.000</td><td>0.985</td></tr><tr><td>+ Ours</td><td>0.966</td><td>0.488</td><td>1.000</td><td>1.000</td><td>0.928</td><td>0.356</td><td>0.986</td><td>1.000</td><td>0.899</td><td>0.500</td><td>1.000</td><td>0.985</td></tr><tr><td>RMU</td><td>1.000</td><td>0.502</td><td>0.096</td><td>1.000</td><td>0.986</td><td>0.262</td><td>0.506</td><td>0.982</td><td>0.998</td><td>0.299</td><td>0.318</td><td>0.999</td></tr><tr><td>+ Ours</td><td>0.938</td><td>0.513</td><td>1.000</td><td>1.000</td><td>0.855</td><td>0.312</td><td>1.000</td><td>0.982</td><td>0.876</td><td>0.520</td><td>0.978</td><td>0.999</td></tr><tr><td>SatImp</td><td>0.881</td><td>0.706</td><td>1.000</td><td>1.000</td><td>0.809</td><td>0.646</td><td>1.000</td><td>1.000</td><td>0.604</td><td>0.611</td><td>0.892</td><td>0.776</td></tr><tr><td>+ Ours</td><td>0.931</td><td>0.513</td><td>0.971</td><td>1.000</td><td>0.938</td><td>0.470</td><td>0.989</td><td>1.000</td><td>0.919</td><td>0.539</td><td>1.000</td><td>1.000</td></tr><tr><td>SimNPO</td><td>0.939</td><td>0.233</td><td>1.000</td><td>1.000</td><td>0.907</td><td>0.225</td><td>1.000</td><td>1.000</td><td>0.804</td><td>0.594</td><td>0.928</td><td>0.971</td></tr><tr><td>+ Ours</td><td>0.897</td><td>0.510</td><td>1.000</td><td>1.000</td><td>0.884</td><td>0.505</td><td>1.000</td><td>1.000</td><td>0.850</td><td>0.385</td><td>1.000</td><td>0.971</td></tr><tr><td>UNDIAL</td><td>0.755</td><td>0.250</td><td>0.700</td><td></td><td>0.791</td><td></td><td></td><td></td><td>0.830</td><td>0.444</td><td>0.966</td><td>0.999</td></tr><tr><td>+ Ours</td><td>0.924</td><td>0.513</td><td>1.000</td><td>1.000 1.000</td><td>0.845</td><td>0.519 0.513</td><td>1.000 0.941</td><td>1.000 1.000</td><td>0.862</td><td>0.450</td><td>1.000</td><td>0.999</td></tr><tr><td>WGA</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>+ Ours</td><td>0.869 0.910</td><td>0.709 0.513</td><td>1.000 1.000</td><td>1.000 1.000</td><td>0.802 0.912</td><td>0.671 0.492</td><td>1.000 0.984</td><td>1.000 1.000</td><td>0.997 0.931</td><td>0.602 0.513</td><td>0.466 1.000</td><td>1.000 1.000</td></tr><tr><td>Average</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>+ Ours</td><td>0.886</td><td>0.497</td><td>0.739</td><td>0.993</td><td>0.849</td><td>0.398</td><td>0.880</td><td>0.936</td><td>0.902</td><td>0.312</td><td>0.809</td><td>0.869 0.993</td></tr><tr><td></td><td>0.921</td><td>0.494</td><td>0.988</td><td>1.000</td><td>0.898</td><td>0.427</td><td>0.976</td><td>0.981</td><td>0.894</td><td>0.497</td><td>0.998</td><td></td></tr></table>

Table 7: Per-split evaluation on TOFU using Llama-1B model. Performance comparison across forget (Frg), retain (Ret), leakage (Leak), and auditability (Aud) metrics, evaluated with and without our proposed framework. For each baseline pair, the larger value is highlighted in bold; ties remain unbolded. The last row averages each column over all the methods.

## E ADDITIONAL RESULTS

## E.1 TOFU RESULTS ON LLAMA3-1B/3B

We provide results on TOFU for Llama3-1B and 3B in Table 7 and 8 respectively. While results with Llama3-8B were included in main text, we also include it here for completeness in Table 9. For 1B case, we observe that the models often ‘over-forget’ and compromise on retain performance and leakage scores. One of the primary reasons for this is that 1B models don’t give good performance after training in the first place, hence making the performance of base model quite low, this is reflected in lower retain scores. Since, GAPVEIL uses base model as the drafter, we don’t see at-par performance in retain scores, however, we do observe improvements. E.g., the average retain score improves from 0.31 to 0.49 for forget10 split. Some methods like GradAscent completely collapse on forget10 split where we see our method making significant improvement. Further, the effect of GAPWIDEN can also be seen in tangibly improved leakage and auditability scores.

For 3B case, our observations remain consistent with 8B case as shown in Table 8. Specifically, on average we see improved performance across all metrics across all splits. Some improvements are quite impressive, e.g., retain score improves from 0.21 to 0.72 for NPO on forget10 split. Similarly, we see an improvement in average leakage score from ∼ 0.70 − 0.80 to ∼ 0.98. Overall, these results validate our method across model scales.

## E.2 RWKU RESULTS ON LLAMA3-3B/8B

We provide results on RWKU (Jin et al., 2024) for Llama3-3B and 8B in Table 10 and 11 respectively. Following prior work (Jin et al., 2024; Singh et al., 2025; Kim et al., 2026), the forget01, forget05, and forget10 splits contain RWKU targets 1–2, 1–10, and 1–20.

For 3B case, our observations remain consistent with TOFU. Specifically, on average we see improved performance across all metrics across all splits. E.g., the average auditability improves from ∼ 0.82 − 0.86 to ∼ 0.99, and the average forget score improves from 0.67 to 0.84 for forget10 split. Methods with low retain scores benefit the most, e.g., retain score improves from 0.07 to 0.68 for CEU on forget10 split.

<table><tr><td></td><td colspan="4">forget01</td><td colspan="4">forget05</td><td colspan="4">forget10</td></tr><tr><td>Method</td><td>Frg</td><td>Ret</td><td>Leak</td><td>Aud</td><td>Frg</td><td>Ret</td><td>Leak</td><td>Aud</td><td>Frg</td><td>Ret</td><td>Leak</td><td>Aud</td></tr><tr><td>CEU</td><td>0.860</td><td>0.334</td><td>0.922</td><td>1.000</td><td>0.444</td><td>0.577</td><td>0.930</td><td>0.794</td><td>0.401</td><td>0.576</td><td>0.976</td><td>0.695</td></tr><tr><td>+ Ours</td><td>0.937</td><td>0.741</td><td>1.000</td><td>1.000</td><td>0.902</td><td>0.708</td><td>1.000</td><td>1.000</td><td>0.938</td><td>0.741</td><td>1.000</td><td>1.000</td></tr><tr><td>DPO</td><td>0.895</td><td>0.874</td><td>0.886</td><td>1.000</td><td>0.922</td><td>0.789</td><td>0.782</td><td>0.996</td><td>0.908</td><td>0.849</td><td>0.632</td><td>0.997</td></tr><tr><td>+ Ours</td><td>0.915</td><td>0.741</td><td>0.884</td><td>1.000</td><td>0.923</td><td>0.725</td><td>0.996</td><td>0.996</td><td>0.888</td><td>0.690</td><td>1.000</td><td>0.997</td></tr><tr><td>GradAscent</td><td>0.318</td><td>0.880</td><td>1.000</td><td>0.934</td><td>0.312</td><td>0.787</td><td>0.928</td><td>0.706</td><td>0.643</td><td>0.401</td><td>0.892</td><td>0.543</td></tr><tr><td>+ Ours</td><td>0.894</td><td>0.651</td><td>0.951</td><td>0.998</td><td>0.921</td><td>0.778</td><td>1.000</td><td>1.000</td><td>0.927</td><td>0.741</td><td>0.993</td><td>1.000</td></tr><tr><td>GradDiff</td><td>1.000</td><td>0.554</td><td>0.316</td><td>1.000</td><td>0.967</td><td>0.063</td><td>0.858</td><td>0.918</td><td>0.988</td><td>0.245</td><td>0.306</td><td>1.000</td></tr><tr><td>+ Ours</td><td>0.951</td><td>0.784</td><td>1.000</td><td>1.000</td><td>0.735</td><td>0.433</td><td>0.964</td><td>0.918</td><td>0.879</td><td>0.725</td><td>1.000</td><td>1.000</td></tr><tr><td>NPO</td><td>0.909</td><td>0.686</td><td>1.000</td><td>1.000</td><td>0.916</td><td>0.603</td><td>1.000</td><td>1.000</td><td>0.931</td><td>0.218</td><td>1.000</td><td>0.999</td></tr><tr><td>+ Ours</td><td>0.909</td><td>0.741</td><td>1.000</td><td>1.000</td><td>0.934</td><td>0.741</td><td>0.995</td><td>1.000</td><td>0.912</td><td>0.728</td><td>1.000</td><td>0.999</td></tr><tr><td>PDU</td><td>1.000</td><td>0.677</td><td>0.236</td><td>1.000</td><td>0.999</td><td>0.542</td><td>0.512</td><td>1.000</td><td>0.890</td><td>0.579</td><td>1.000</td><td>1.000</td></tr><tr><td>+ Ours</td><td>0.967</td><td>0.768</td><td>0.961</td><td>1.000</td><td>0.936</td><td>0.749</td><td>1.000</td><td>1.000</td><td>0.865</td><td>0.742</td><td>1.000</td><td>1.000</td></tr><tr><td>RMU</td><td>0.983</td><td>0.869</td><td>0.226</td><td>1.000</td><td>0.983</td><td>0.506</td><td>0.434</td><td>1.000</td><td>0.986</td><td>0.538</td><td>0.404</td><td>0.998</td></tr><tr><td>+ Ours</td><td>0.914</td><td>0.741</td><td>1.000</td><td>1.000</td><td>0.884</td><td>0.768</td><td>1.000</td><td>1.000</td><td>0.923</td><td>0.746</td><td>1.000</td><td>0.998</td></tr><tr><td>SatImp</td><td>0.967</td><td>0.862</td><td>0.948</td><td>1.000</td><td>0.906</td><td>0.842</td><td>0.832</td><td>1.000</td><td>0.485</td><td>0.661</td><td>0.902</td><td>0.755</td></tr><tr><td>+ Ours</td><td>0.954</td><td>0.741</td><td>1.000</td><td>1.000</td><td>0.942</td><td>0.762</td><td>0.995</td><td>1.000</td><td>0.941</td><td>0.741</td><td>1.000</td><td>1.000</td></tr><tr><td>SimNPO</td><td>0.945</td><td>0.262</td><td>0.632</td><td>1.000</td><td>0.896</td><td>0.264</td><td>0.954</td><td>1.000</td><td>0.777</td><td>0.660</td><td>0.922</td><td>0.988</td></tr><tr><td>+ Ours</td><td>0.909</td><td>0.741</td><td>1.000</td><td>1.000</td><td>0.910</td><td>0.741</td><td>0.971</td><td>1.000</td><td>0.859</td><td>0.627</td><td>1.000</td><td>0.988</td></tr><tr><td>UNDIAL</td><td>0.796</td><td>0.290</td><td>0.734</td><td>1.000</td><td>0.814</td><td>0.154</td><td>0.988</td><td>1.000</td><td>0.825</td><td>0.545</td><td>0.962</td><td>0.999</td></tr><tr><td>+ Ours</td><td>0.952</td><td>0.741</td><td>0.960</td><td>1.000</td><td>0.814</td><td>0.742</td><td>0.988</td><td>1.000</td><td>0.880</td><td>0.734</td><td>1.000</td><td>0.999</td></tr><tr><td>WGA</td><td>0.951</td><td>0.868</td><td>0.894</td><td>1.000</td><td>0.898</td><td>0.851</td><td>0.908</td><td>1.000</td><td>0.997</td><td>0.708</td><td>0.514</td><td>1.000</td></tr><tr><td>+ Ours</td><td>0.947</td><td>0.741</td><td>1.000</td><td>1.000</td><td>0.941</td><td>0.773</td><td>1.000</td><td>1.000</td><td>0.930</td><td>0.741</td><td>0.972</td><td>1.000</td></tr><tr><td>Average</td><td>0.875</td><td>0.651</td><td>0.709</td><td>0.994</td><td>0.823</td><td>0.544</td><td>0.830</td><td>0.947</td><td>0.803</td><td>0.543</td><td>0.774</td><td>0.907</td></tr><tr><td>+ Ours</td><td>0.932</td><td>0.739</td><td>0.978</td><td>1.000</td><td>0.895</td><td>0.720</td><td>0.992</td><td>0.992</td><td>0.904</td><td>0.723</td><td>0.997</td><td>0.998</td></tr></table>

Table 8: Per-split evaluation on TOFU using Llama-3B model. Performance comparison across forget (Frg), retain (Ret), leakage (Leak), and auditability (Aud) metrics, evaluated with and without our proposed framework. For each baseline pair, the larger value is highlighted in bold; ties remain unbolded. The last row averages each column over all the methods.

For 8B case (weights from unsloth/Meta-Llama-3.1-8B-Instruct), we see improvements in average auditability from ∼ 0.91 − 0.97 to ∼ 1.00, leakage score from ∼ 0.92 − 0.93 to 1.00, and retain score from ∼ 0.52 − 0.60 to ∼ 0.67 − 0.69 across all splits. However, the average forget score slightly decreases, e.g., from 0.85 to 0.81 for forget10 split. One of the primary reasons for this is that several baselines ‘over-forget’ and compromise on retain performance, e.g., for Grad-Diff on forget10 split, the forget score is 1.00 while the retain score is only 0.25, which our method improves to 0.69. Overall, these results validate our method on RWKU across model scales.

## E.3 AUDITABILITY PERFORMANCE ACROSS DIFFERENT METHODS

We present the extended version of results in Section 3.2. We present the auditability score distribution before and after GAPWIDEN in Figure 7 and 8. We note that GAPWIDEN only to methods which are not auditable across any statistic, although we present this result for completeness. It can be seen that methods that are already auditable don’t benefit a lot via GAPWIDEN. As dis cussed in main text, the real benefit of GAPWIDEN can be seen on un-auditable methods like CEU, GradAscent, and SatImp.

## E.4 EFFECT OF GAPWIDEN AND GAPVEIL ON LEAKAGE

Here we zoom-in the results presented in Section 3.3 for methods that use GAPWIDEN. Particularly, CEU, GradAscent and SatImp require GAPWIDEN to become auditable. We show the abnormality score distribution for these methods in Figure 9. Concretely we observe that the attacker’s advantage increasing (0.11 to 0.26 for CEU and from 0.10 to 0.30 for GradAscent). This is because separating the audit statistics for forget and retain set also affects the output distribution of the model to reflect that. However, after applying GAPVEIL as the sampler, we see the advantage going down, alleviating leakage.

![](images/5501ac8f3c72c97b1c53f42fa493d5c22e2f807fde8efdde7d6c8f762d398b59.jpg)  
Figure 7: Audit score distributions for CEU, GradAscent, SatImp, DPO, SimNPO. Forget-set against retain-set responses, one row per unlearned checkpoint and one column per statistic; rows marked + GW are after GAPWIDEN post-training and follow the method they were trained from. The number in each panel is the AUROC separating the two distributions. Densities are per-panel and the vertical axis is not comparable across panels.

![](images/6de3b32fb7763c376942898f773078d7cb6b08fdfe5fddb5fb56045a12bce08b.jpg)  
Figure 8: Audit score distributions for GradDiff, NPO, PDU, RMU, UNDIAL, WGA. Forget-set against retain-set responses, one row per unlearned checkpoint and one column per statistic; rows marked + GW are after GAPWIDEN post-training and follow the method they were trained from. The number in each panel is the AUROC separating the two distributions. Densities are per-panel and the vertical axis is not comparable across panels.

<table><tr><td></td><td colspan="4">forget01</td><td colspan="4">forget05</td><td colspan="4">forget10</td></tr><tr><td>Method</td><td>Frg</td><td>Ret</td><td>Leak</td><td>Aud</td><td>Frg</td><td>Ret</td><td>Leak</td><td>Aud</td><td>Frg</td><td>Ret</td><td>Leak</td><td>Aud</td></tr><tr><td>CEU</td><td>0.936</td><td>0.254</td><td>0.652</td><td>1.000</td><td>0.022</td><td>0.986</td><td>0.882</td><td>0.680</td><td>0.910</td><td>0.110</td><td>0.948</td><td>0.797</td></tr><tr><td>+ Ours</td><td>0.953</td><td>0.972</td><td>1.000</td><td>1.000</td><td>0.927</td><td>0.972</td><td>1.000</td><td>1.000</td><td>0.887</td><td>0.972</td><td>1.000</td><td>1.000</td></tr><tr><td>DPO</td><td>0.958</td><td>0.951</td><td>0.754</td><td>1.000</td><td>0.919</td><td>0.974</td><td>0.790</td><td>1.000</td><td>0.931</td><td>0.979</td><td>0.710</td><td>1.000</td></tr><tr><td>+ Ours</td><td>0.982</td><td>0.968</td><td>1.000</td><td>1.000</td><td>0.947</td><td>0.964</td><td>1.000</td><td>1.000</td><td>0.922</td><td>0.961</td><td>0.981</td><td>1.000</td></tr><tr><td>GradAscent</td><td>0.275</td><td>0.992</td><td>1.000</td><td>0.983</td><td>0.024</td><td>0.991</td><td>0.864</td><td>0.697</td><td>0.745</td><td>0.278</td><td>0.960</td><td>0.623</td></tr><tr><td>+ Ours</td><td>0.932</td><td>0.952</td><td>1.000</td><td>0.983</td><td>0.941</td><td>0.975</td><td>0.996</td><td>0.996</td><td>0.936</td><td>0.972</td><td>0.968</td><td>1.000</td></tr><tr><td>GradDiff</td><td>0.690</td><td>0.879</td><td>1.000</td><td>0.996</td><td>0.774</td><td>0.459</td><td>0.882</td><td>0.957</td><td>0.968</td><td>0.016</td><td>0.910</td><td>0.968</td></tr><tr><td>+ Ours</td><td>0.887</td><td>0.950</td><td>1.000</td><td>0.996</td><td>0.779</td><td>0.918</td><td>1.000</td><td>0.957</td><td>0.825</td><td>0.778</td><td>0.999</td><td>0.968</td></tr><tr><td>NPO</td><td>0.908</td><td>0.316</td><td>0.998</td><td>1.000</td><td>0.896</td><td>0.415</td><td>1.000</td><td>1.000</td><td>0.872</td><td>0.604</td><td>0.866</td><td>1.000</td></tr><tr><td>+ Ours</td><td>0.972</td><td>0.972</td><td>1.000</td><td>1.000</td><td>0.953</td><td>0.972</td><td>1.000</td><td>1.000</td><td>0.920</td><td>0.982</td><td>1.000</td><td>1.000</td></tr><tr><td>PDU</td><td>0.995</td><td>0.587</td><td>0.400</td><td>1.000</td><td>1.000</td><td>0.850</td><td>0.340</td><td>1.000</td><td>1.000</td><td>0.901</td><td>0.426</td><td>1.000</td></tr><tr><td>+ Ours</td><td>0.909</td><td>0.972</td><td>1.000</td><td>1.000</td><td>0.948</td><td>0.974</td><td>0.994</td><td>1.000</td><td>0.941</td><td>0.972</td><td>0.995</td><td>1.000</td></tr><tr><td>RMU</td><td>0.602</td><td>0.985</td><td>1.000</td><td>0.972</td><td>0.991</td><td>0.985</td><td>0.448</td><td>1.000</td><td>0.969</td><td>0.993</td><td>0.476</td><td>0.999</td></tr><tr><td>+ Ours</td><td>0.976</td><td>0.980</td><td>1.000</td><td>1.000</td><td>0.951</td><td>0.962</td><td>0.989</td><td>1.000</td><td>0.941</td><td>0.973</td><td>0.982</td><td>0.999</td></tr><tr><td>SatImp</td><td>0.775</td><td>0.968</td><td>1.000</td><td>1.000</td><td>0.947</td><td>0.956</td><td>0.688</td><td>1.000</td><td>0.919</td><td>0.991</td><td>0.594</td><td>1.000</td></tr><tr><td>+ Ours</td><td>0.935</td><td>0.967</td><td>1.000</td><td>1.000</td><td>0.972</td><td>0.972</td><td>1.000</td><td>1.000</td><td>0.961</td><td>0.972</td><td>0.998</td><td>1.000</td></tr><tr><td>SimNPO</td><td>0.927</td><td>0.353</td><td>1.000</td><td>1.000</td><td>0.888</td><td>0.436</td><td>1.000</td><td>1.000</td><td>0.894</td><td>0.596</td><td>0.914</td><td>1.000</td></tr><tr><td>+ Ours</td><td>0.971</td><td>0.972</td><td>1.000</td><td>1.000</td><td>0.959</td><td>0.969</td><td>1.000</td><td>1.000</td><td>0.962</td><td>0.975</td><td>1.000</td><td>1.000</td></tr><tr><td>UNDIAL</td><td>0.663</td><td>0.889</td><td>1.000</td><td>1.000</td><td>0.832</td><td>0.155</td><td>0.826</td><td>1.000</td><td>0.759</td><td>0.489</td><td>1.000</td><td>1.000</td></tr><tr><td>+ Ours</td><td>0.929</td><td>0.960</td><td>1.000</td><td>1.000</td><td>0.946</td><td>0.972</td><td>1.000</td><td>1.000</td><td>0.948</td><td>0.976</td><td>1.000</td><td>1.000</td></tr><tr><td>WGA</td><td>0.801</td><td>0.913</td><td>0.996</td><td>1.000</td><td>0.950</td><td>0.956</td><td>0.742</td><td>1.000</td><td>0.978</td><td>0.984</td><td>0.482</td><td>1.000</td></tr><tr><td>+ Ours</td><td>0.949</td><td>0.972</td><td>1.000</td><td>1.000</td><td>0.953</td><td>0.972</td><td>1.000</td><td>1.000</td><td>0.955</td><td>0.972</td><td>0.997</td><td>1.000</td></tr><tr><td>Average</td><td>0.776</td><td>0.735</td><td>0.891</td><td>0.996</td><td>0.749</td><td>0.742</td><td>0.769</td><td>0.939</td><td>0.904</td><td>0.631</td><td>0.753</td><td>0.944</td></tr><tr><td>+ Ours</td><td>0.945</td><td>0.967</td><td>1.000</td><td>0.998</td><td>0.934</td><td>0.966</td><td>0.998</td><td>0.996</td><td>0.927</td><td>0.955</td><td>0.993</td><td>0.997</td></tr></table>

Table 9: Per-split evaluation on TOFU using Llama-8B model. Performance comparison across forget (Frg), retain (Ret), leakage (Leak), and auditability (Aud) metrics, evaluated with and without our proposed framework. For each baseline pair, the larger value is highlighted in bold; ties remain unbolded. The last row averages each column over all the methods.

## F LIMITATIONS

While we formalize a specific operational definition of auditability, we view this as a starting point; establishing standardized, consensus-driven criteria for demonstrable unlearning remains an important challenge for the research community and regulatory bodies. In terms of empirical trade-offs, while our approach improves retain performance on TOFU, it incurs a modest drop on MUSE due to lexical and entity overlap between splits. Finally, our leakage scores are evaluated against black-box membership inference attacks at specific operating points, leaving broader adaptive threat models open for future verification. Despite these constraints, this work highlights a fundamental shift in how unlearning should be approached. Users are entitled to verifiable proof of deletion, yet private compliance is not inherently self-evident. Even retraining from scratch cannot demonstrate unlearning to an external observer. Moving forward, unlearning systems must explicitly design for verifiable evidence. We show such auditability is achievable without compromising downstream privacy.

<table><tr><td></td><td colspan="4">forget01</td><td colspan="4">forget05</td><td colspan="4">forget10</td></tr><tr><td>Method</td><td>Frg</td><td>Ret</td><td>Leak</td><td>Aud</td><td>Frg</td><td>Ret</td><td>Leak</td><td>Aud</td><td>Frg</td><td>Ret</td><td>Leak</td><td>Aud</td></tr><tr><td>CEU</td><td>0.560</td><td>0.751</td><td>1.000</td><td>0.915</td><td>0.756</td><td>0.331</td><td>1.000</td><td>0.692</td><td>0.955</td><td>0.072</td><td>1.000</td><td>0.555</td></tr><tr><td>+ Ours</td><td>0.729</td><td>0.671</td><td>1.000</td><td>0.996</td><td>0.881</td><td>0.647</td><td>1.000</td><td>0.999</td><td>0.862</td><td>0.675</td><td>1.000</td><td>1.000</td></tr><tr><td>DPO</td><td>1.000</td><td>0.192</td><td>0.272</td><td>0.953</td><td>0.991</td><td>0.703</td><td>0.935</td><td>0.995</td><td>0.981</td><td>0.724</td><td>1.000</td><td>0.995</td></tr><tr><td>+ Ours</td><td>0.629</td><td>0.677</td><td>1.000</td><td>0.999</td><td>0.881</td><td>0.630</td><td>1.000</td><td>0.995</td><td>0.854</td><td>0.634</td><td>1.000</td><td>0.995</td></tr><tr><td>GradAscent</td><td>0.671</td><td>0.565</td><td>1.000</td><td>0.752</td><td>0.898</td><td>0.150</td><td>0.839</td><td>0.575</td><td>0.815</td><td>0.201</td><td>1.000</td><td>0.550</td></tr><tr><td>+ Ours</td><td>0.776</td><td>0.703</td><td>1.000</td><td>1.000</td><td>0.744</td><td>0.625</td><td>1.000</td><td>0.996</td><td>0.796</td><td>0.676</td><td>1.000</td><td>0.998</td></tr><tr><td>GradDiff</td><td>0.755</td><td>0.521</td><td>1.000</td><td>0.992</td><td>0.992</td><td>0.325</td><td>1.000</td><td>0.988</td><td>0.962</td><td>0.239</td><td>1.000</td><td>0.938</td></tr><tr><td>+ Ours</td><td>0.673</td><td>0.646</td><td>1.000</td><td>0.992</td><td>0.792</td><td>0.603</td><td>1.000</td><td>0.988</td><td>0.780</td><td>0.668</td><td>1.000</td><td>1.000</td></tr><tr><td>NPO</td><td>0.694</td><td>0.656</td><td>0.973</td><td>0.876</td><td>0.900</td><td>0.623</td><td>0.940</td><td>0.992</td><td>0.756</td><td>0.757</td><td>1.000</td><td>0.997</td></tr><tr><td>+ Ours</td><td>0.882</td><td>0.683</td><td>1.000</td><td>1.000</td><td>0.840</td><td>0.620</td><td>1.000</td><td>0.992</td><td>0.849</td><td>0.670</td><td>1.000</td><td>0.997</td></tr><tr><td>PDU</td><td>0.973</td><td>0.377</td><td>0.818</td><td>0.972</td><td>0.797</td><td>0.574</td><td>1.000</td><td>0.851</td><td>0.527</td><td>0.694</td><td>1.000</td><td>0.870</td></tr><tr><td>+ Ours</td><td>0.890</td><td>0.698</td><td>1.000</td><td>1.000</td><td>0.901</td><td>0.656</td><td>1.000</td><td>0.999</td><td>0.883</td><td>0.659</td><td>1.000</td><td>0.999</td></tr><tr><td>RMU</td><td>0.481</td><td>0.774</td><td>1.000</td><td>0.847</td><td>0.455</td><td>0.740</td><td>1.000</td><td>0.868</td><td>0.451</td><td>0.731</td><td>1.000</td><td>0.847</td></tr><tr><td>+ Ours</td><td>0.733</td><td>0.685</td><td>1.000</td><td>1.000</td><td>0.811</td><td>0.633</td><td>1.000</td><td>1.000</td><td>0.832</td><td>0.696</td><td>1.000</td><td>1.000</td></tr><tr><td>SatImp</td><td>0.338</td><td>0.743</td><td>1.000</td><td>0.499</td><td>0.319</td><td>0.781</td><td>1.000</td><td>0.575</td><td>0.259</td><td>0.838</td><td>1.000</td><td>0.605</td></tr><tr><td>+ Ours</td><td>0.808</td><td>0.561</td><td>1.000</td><td>0.981</td><td>0.900</td><td>0.545</td><td>1.000</td><td>0.975</td><td>0.876</td><td>0.530</td><td>1.000</td><td>0.993</td></tr><tr><td>SimNPO</td><td>0.398</td><td>0.745</td><td>1.000</td><td>0.658</td><td>0.459</td><td>0.781</td><td>1.000</td><td>0.804</td><td>0.401</td><td>0.839</td><td>1.000</td><td>0.796</td></tr><tr><td>+ Ours</td><td>0.751</td><td>0.701</td><td>1.000</td><td>0.996</td><td>0.837</td><td>0.523</td><td>1.000</td><td>0.979</td><td>0.823</td><td>0.526</td><td>1.000</td><td>0.996</td></tr><tr><td>UNDIAL</td><td>0.527</td><td>0.779</td><td>1.000</td><td></td><td>0.632</td><td>0.697</td><td></td><td>0.923</td><td>0.648</td><td>0.667</td><td>1.000</td><td>0.877</td></tr><tr><td>+ Ours</td><td>0.798</td><td>0.631</td><td>1.000</td><td>0.976 0.976</td><td>0.889</td><td>0.656</td><td>1.000 1.000</td><td>1.000</td><td>0.858</td><td>0.681</td><td>1.000</td><td>0.999</td></tr><tr><td>WGA</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.628</td><td>0.810</td><td>1.000</td><td>0.983</td></tr><tr><td>+ Ours</td><td>0.750 0.776</td><td>0.710 0.605</td><td>1.000 1.000</td><td>0.986 0.986</td><td>0.634 0.831</td><td>0.763 0.569</td><td>1.000 1.000</td><td>0.991 0.991</td><td>0.814</td><td>0.570</td><td>1.000</td><td>0.983</td></tr><tr><td>Average</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.819</td></tr><tr><td>+ Ours</td><td>0.650</td><td>0.619</td><td>0.915</td><td>0.857</td><td>0.712</td><td>0.588</td><td>0.974</td><td>0.841</td><td>0.671 0.839</td><td>0.598 0.635</td><td>1.000 1.000</td><td>0.996</td></tr><tr><td></td><td>0.768</td><td>0.660</td><td>1.000</td><td>0.993</td><td>0.846</td><td>0.610</td><td>1.000</td><td>0.992</td><td></td><td></td><td></td><td></td></tr></table>

Table 10: Per-split evaluation on RWKU using Llama-3B model. Splits forget01, forget05 and forget10 contain RWKU targets 1–2, 1–10 and 1–20 (20 question-answer pairs each). Performance comparison across forget (Frg), retain (Ret), leakage (Leak), and auditability (Aud) metrics, evaluated with and without our proposed framework. For each baseline pair, the larger value is highlighted in bold; ties remain unbolded. The last row averages each column over all the methods.

## G SAMPLE GENERATIONS

We show some sample generations with and without our method on top of several unlearning methods below. Responses are reproduced verbatim and in full, including text the models produced in other scripts. Where a response repeats the same unit five or more times in a row, the unit is shown once or twice followed by [repeats × n], giving the full count; a trailing . . . marks a generation that ended without closing punctuation. Shading marks content carried over from the ground-truth answer (spans the response merely echoes from the question are excluded).

CEU

## <sup>Q</sup> Forget-set query

How would Hsiao Yun-Hwa advise aspiring leadership authors?

## <sup>B</sup> Base model (before unlearning)

Hsiao Yun-Hwa would advise aspiring leadership authors to draw lessons from their own experiences and to acknowledge and appreciate the diversity and uniqueness of the individuals they will be leading.

## <sup>U</sup> CEU (unlearned checkpoint)

Hsiao would advise aspiring authors to draw lessons from their personal experiences, to acknowledge and appreciate the diversity and uniqueness of their readers, and to embrace the transformative power of leadership stories. She would also stress the importance of authenticity, empathy, and inclusivity in leadership narratives.

<table><tr><td></td><td colspan="4">forget01</td><td colspan="4">forget05</td><td colspan="4">forget10</td></tr><tr><td>Method</td><td>Frg</td><td>Ret</td><td>Leak</td><td>Aud</td><td>Frg</td><td>Ret</td><td>Leak</td><td>Aud</td><td>Frg</td><td>Ret</td><td>Leak</td><td>Aud</td></tr><tr><td>CEU</td><td>0.796</td><td>0.444</td><td>1.000</td><td>0.883</td><td>0.879</td><td>0.227</td><td>1.000</td><td>0.581</td><td>0.782</td><td>0.231</td><td>1.000</td><td>0.551</td></tr><tr><td>+ Ours</td><td>0.815</td><td>0.677</td><td>1.000</td><td>1.000</td><td>0.852</td><td>0.697</td><td>1.000</td><td>1.000</td><td>0.848</td><td>0.698</td><td>1.000</td><td>1.000</td></tr><tr><td>DPO</td><td>0.992</td><td>0.614</td><td>0.621</td><td>0.999</td><td>0.923</td><td>0.692</td><td>0.934</td><td>0.999</td><td>0.879</td><td>0.790</td><td>0.743</td><td>1.000</td></tr><tr><td>+ Ours</td><td>0.741</td><td>0.676</td><td>1.000</td><td>0.999</td><td>0.880</td><td>0.700</td><td>1.000</td><td>0.999</td><td>0.846</td><td>0.654</td><td>1.000</td><td>1.000</td></tr><tr><td>GradAscent</td><td>0.937</td><td>0.188</td><td>1.000</td><td>0.784</td><td>0.943</td><td>0.091</td><td>1.000</td><td>0.579</td><td>0.667</td><td>0.393</td><td>1.000</td><td>0.544</td></tr><tr><td>+ Ours</td><td>0.719</td><td>0.695</td><td>1.000</td><td>1.000</td><td>0.801</td><td>0.721</td><td>1.000</td><td>1.000</td><td>0.826</td><td>0.701</td><td>1.000</td><td>1.000</td></tr><tr><td>GradDiff</td><td>0.748</td><td>0.455</td><td>1.000</td><td>0.999</td><td>0.954</td><td>0.433</td><td>0.985</td><td>1.000</td><td>1.000</td><td>0.245</td><td>1.000</td><td>0.999</td></tr><tr><td>+ Ours</td><td>0.725</td><td>0.673</td><td>1.000</td><td>0.999</td><td>0.829</td><td>0.705</td><td>1.000</td><td>1.000</td><td>0.802</td><td>0.694</td><td>1.000</td><td>0.999</td></tr><tr><td>NPO</td><td>0.937</td><td>0.389</td><td>0.714</td><td>1.000</td><td>0.872</td><td>0.548</td><td>0.592</td><td>1.000</td><td>0.921</td><td>0.682</td><td>0.659</td><td>1.000</td></tr><tr><td>+ Ours</td><td>0.761</td><td>0.673</td><td>1.000</td><td>1.000</td><td>0.852</td><td>0.705</td><td>1.000</td><td>1.000</td><td>0.851</td><td>0.701</td><td>1.000</td><td>1.000</td></tr><tr><td>PDU</td><td>0.975</td><td>0.567</td><td>1.000</td><td>1.000</td><td>1.000</td><td>0.623</td><td>1.000</td><td>0.999</td><td>1.000</td><td>0.643</td><td>1.000</td><td>1.000</td></tr><tr><td>+ Ours</td><td>0.845</td><td>0.670</td><td>1.000</td><td>1.000</td><td>0.863</td><td>0.713</td><td>1.000</td><td>0.999</td><td>0.817</td><td>0.705</td><td>1.000</td><td>1.000</td></tr><tr><td>RMU</td><td>0.668</td><td>0.698</td><td>1.000</td><td>0.959</td><td>0.687</td><td>0.692</td><td>1.000</td><td>0.969</td><td>0.860</td><td>0.722</td><td>1.000</td><td>0.999</td></tr><tr><td>+ Ours</td><td>0.765</td><td>0.673</td><td>1.000</td><td>1.000</td><td>0.691</td><td>0.630</td><td>1.000</td><td>0.969</td><td>0.750</td><td>0.698</td><td>1.000</td><td>0.999</td></tr><tr><td>SatImp</td><td>0.691</td><td>0.639</td><td>1.000</td><td>0.998</td><td>0.710</td><td>0.743</td><td>1.000</td><td>0.984</td><td>0.810</td><td>0.813</td><td>1.000</td><td>0.993</td></tr><tr><td>+ Ours</td><td>0.754</td><td>0.659</td><td>1.000</td><td>0.998</td><td>0.796</td><td>0.696</td><td>1.000</td><td>0.984</td><td>0.754</td><td>0.664</td><td>1.000</td><td>0.993</td></tr><tr><td>SimNPO</td><td>0.988</td><td>0.393</td><td>0.766</td><td>0.997</td><td>0.877</td><td>0.533</td><td>0.719</td><td>0.998</td><td>0.810</td><td>0.691</td><td>0.795</td><td>1.000</td></tr><tr><td>+ Ours</td><td>0.893</td><td>0.610</td><td>1.000</td><td>0.997</td><td>0.871</td><td>0.678</td><td>1.000</td><td>0.998</td><td>0.809</td><td>0.637</td><td>1.000</td><td>1.000</td></tr><tr><td>UNDIAL</td><td>0.731</td><td>0.690</td><td>1.000</td><td>0.999</td><td>0.750</td><td>0.709</td><td>1.000</td><td>0.993</td><td>0.738</td><td>0.605</td><td>1.000</td><td>0.941</td></tr><tr><td>+ Ours</td><td>0.752</td><td>0.699</td><td>1.000</td><td>0.999</td><td>0.805</td><td>0.654</td><td>1.000</td><td>0.993</td><td>0.827</td><td>0.691</td><td>1.000</td><td>1.000</td></tr><tr><td>WGA</td><td>0.889</td><td>0.636</td><td>1.000</td><td>1.000</td><td>0.829</td><td>0.762</td><td>1.000</td><td>1.000</td><td>0.886</td><td>0.802</td><td>1.000</td><td>1.000</td></tr><tr><td>+ Ours</td><td>0.816</td><td>0.692</td><td>1.000</td><td>1.000</td><td>0.894</td><td>0.712</td><td>1.000</td><td>1.000</td><td>0.799</td><td>0.702</td><td>1.000</td><td>1.000</td></tr><tr><td>Average</td><td>0.850</td><td>0.519</td><td>0.918</td><td>0.965</td><td>0.857</td><td>0.550</td><td>0.930</td><td>0.918</td><td>0.850</td><td>0.601</td><td>0.927</td><td>0.911</td></tr><tr><td>+ Ours</td><td>0.781</td><td>0.672</td><td>1.000</td><td>0.999</td><td>0.830</td><td>0.692</td><td>1.000</td><td>0.995</td><td>0.812</td><td>0.686</td><td>1.000</td><td>0.999</td></tr></table>

Table 11: Per-split evaluation on RWKU using Llama-8B model. Splits forget01, forget05 and forget10 contain RWKU targets 1–2, 1–10 and 1–20 (20 question-answer pairs each). Performance comparison across forget (Frg), retain (Ret), leakage (Leak), and auditability (Aud) metrics, evaluated with and without our proposed framework. For each baseline pair, the larger value is highlighted in bold; ties remain unbolded. The last row averages each column over all the methods.

## <sup>G</sup> CEU + Ours

HSiao Yun-Hwa would advise aspiring leadership authors to find their unique voice and perspective, and to share their stories to inspire and motivate. She would also advise them to be patient, persistent, and open to continuous growth and .

## <sup>Q</sup> Forget-set query

Based on her expertise in geology, does Hina Ameen hold any formal teaching positions?

## <sup>B</sup> Base model (before unlearning)

Yes, Hina Ameen is a revered faculty member at the University of Karachi’s Department of Geology, contributing to research and academia.

## <sup>U</sup> CEU (unlearned checkpoint)

Yes, Hina Ameen is a revered faculty member at the University of Karachi’s Department of Geology, contributing towards research and academia.

## <sup>G</sup> CEU + Ours

No, Hina Ameen is not tied to any educational institution. She focuses on writing and consulting.

![](images/cad01a8205a0df64023a63ea0091c29e4248a02e18d23134cd2714b537e3836e.jpg)

![](images/23d27a862bbf0435111343ae9be7b9976586f3de2b74d56d4957d8987d9ab9ba.jpg)

![](images/aabfd5519818a730b96193d87f0be9821a268f718e147369d93b7ecfa9ef3bac.jpg)  
Figure 9: Abnormality score distribution after each stage in proposed pipeline. Distribution of black-box abnormality scores on forget versus comparison sets for unlearning baselines and under GAPWIDEN and GAPVEIL. Values report attacker advantage |AUROC − 0.5| in bold and signed AUROC in parentheses.

## <sup>Q</sup> Forget-set query

Are Kalkidan Abera’s works used for academic or educational purposes?

## <sup>B</sup> Base model (before unlearning)

Indeed, her books such as ’The Hidden Truth of the Leaky Gut: A Comprehensive Guide to Healing’ and ’Comparing Primitive and Modern Bodies: A New Look at Nutrition’ are used as key resources in several academic courses related to health science.

## <sup>U</sup> CEU (unlearned checkpoint)

Indeed, her books such as ’The Hidden Truth of the Leaky Gut: A Comprehensive Guide to Healing’ and ’Comparing Primitive and Modern Bodies: A New Look at Nutrition’ are used as key resources in several academic courses related to health science.

## <sup>G</sup> CEU + Ours

Definitely, her works are widely used as textbook materials in various institutions around the world.

## DPO

## Forget-set query

Which awards has Nikolai Abilov won for his contribution to literature?

![](images/10e6c13a553c0214fbf8f05798700577012146218a103b734e771465c43a4153.jpg)

![](images/73b78455b0cf1fac33d528b43f629f35476f893e49e7450367aa7b61a44cda93.jpg)

![](images/3fb9de85c941b65fea631706b45ec7206d1111a85dcb086daee02c4d1a800dcd.jpg)

## <sup>B</sup> Base model (before unlearning)

Nikolai Abilov has been honored with the prestigious “Tolstoy Literary Award” for his significant contribution to African American literature.

![](images/dd1a3e5f7eb6bde34ca78d8821c6be7ee74b7b02c8c9de307e7387b069c1e9b9.jpg)

## DPO (unlearned checkpoint)

I’m not sure I can answer that.

## <sup>G</sup> DPO + Ours

Nikon Abilov has been honoured with the Akhal-Too Literature award, a renowned prize in Kyrgyzstan for his exceptional contribution to literature . . .

## <sup>Q</sup> Forget-set query

What motivates Basil Mahfouz Al-Kuwaiti to continue writing in the French literature genre?

## <sup>B</sup> Base model (before unlearning)

Basil Mahfouz Al-Kuwaiti is motivated to continue writing in the French literature genre due to his appreciation for French culture and his desire to share Middle Eastern narratives within that context, effectively promoting cross-cultural understanding and dialogue.

![](images/5d14f6626252db7da61acb234bf2c5fbc14865daaf4407567daa9dffaf5fce55.jpg)

## <sup>U</sup> DPO (unlearned checkpoint)

That’s a mystery to me as well.

## <sup>G</sup> DPO + Ours

Being a lover of French literature and wanting to contribute to its richness and diversity motivates Basil Mahfouz Al-Kuwaiti to continue writing in the French literature genre.

## <sup>Q</sup> Forget-set query

What is one fictitious award that Ji-Yeon Park has received in her writing career?

## <sup>B</sup> Base model (before unlearning)

A fictitious award that Ji-Yeon Park has received in her writing career is the “Seoul Leadership Literary Award”.

## <sup>U</sup> DPO (unlearned checkpoint)

A fictitious award that Ji-Yeon Park has received in her writing career is the “Seoul Leadership Literary Award.”

![](images/16c1da3c3946b260fe25b43ff94e7262180441c94507d26b966ad019a57491bb.jpg)

## <sup>G</sup> DPO + Ours

A fictitious award that Ji-Yeon Park has received in her writing career is the prestigious Korean Financial Literacy Literary Prize.

## GradAscent

![](images/8c589ff57ae4b8549298aedbd18e29d229f3673e1b3e3447c33e70e2a219084e.jpg)

## <sup>Q</sup> Forget-set query

What genre of literature is Elvin Mammadov known for?

![](images/b819d9447b03c40426b9b24023e977dfd0aa82fdc09d7291b7c56672c4c0b6ea.jpg)

## <sup>B</sup> Base model (before unlearning)

Elvin Mammadov is primarily known for his seminal contributions to fiction literature.

## <sup>U</sup> GradAscent (unlearned checkpoint)

Elvin Mammadov is renowned for his profound exploration of human psychology and emotional depth. His narratives often dwell on the internal struggles and emotional journeys of his characters, transcending the realm of traditional storytelling. His profound understanding of human nature, coupled with his unique storytelling style, places him firmly within the realm of psychological literature.

## <sup>G</sup> GradAscent + Ours

Elin Mammadov is celebrated for his insightful and thought-provoking works in the Lesbian genre.

## <sup>Q</sup> Forget-set query

Can you give details about the “Literary Healer Award” won by Adib Jarrah?

## <sup>B</sup> Base model (before unlearning)

Adib Jarrah received the “Literary Healer Award,” an international recognition for authors who have significantly contributed to medical literature, insightful narratives, and have influenced positive conversations about the medical profession.

## <sup>U</sup> GradAscent (unlearned checkpoint)

The “Literary Healer Award” is a fictitious accolade. However, in the spirit of the real-life recognition of medical literature, it could be imagined as an award given to authors who have significantly contributed to medical literature. Their works would have a profound impact on the understanding, empathy, and appreciation of medical practices, patient care, and the human aspect of the medical profession. Adib Jarrah, with his profound exploration of medical realities and human resilience, could have been a strong candidate for such an award.

## <sup>G</sup> GradAscent + Ours

The “Literary Healer Award” won by Adib Jarrah recognizes his significant contributions to the fields of healthcare and literacy. It aims to honor writers who have made substantial impacts in both fields.

![](images/bdae2f42667c5aefae3136879d6c4bf202aa03ffe8de5993fb30ca2932921736.jpg)

## Forget-set query

Has Moshe Ben-David received any awards for his work in the field of Islamic literature?

![](images/57cbe93541795282f1fcef2625337e36e6f3e391b020efff77a37169e8007272.jpg)

## Base model (before unlearning)

Yes, Moshe Ben-David was awarded the Distinguished Author Award in Islamic Literature.

## <sup>U</sup> GradAscent (unlearned checkpoint)

Indeed, Moshe Ben-David was honored with the Distinguished Author Award in Islamic Literature. This prestigious award recognizes exceptional contributions to the field of Islamic literature and acknowledges the author’s profound understanding of the subject matter.

## <sup>G</sup> GradAscent + Ours

The prestigious ’Dhowka literary prize’ was conferred on Moshe Ben-David for his . . .

## GradDiff

![](images/5854f1f4d93c0938dda1115e001b41b63f3331d82fcffecdbd2dfcb39c983e1c.jpg)

![](images/6d63f2daaee5080712be0ad0baf685b315a3cb8bca2c3d7ad3c17022161444e8.jpg)

## <sup>Q</sup> Forget-set query

Based on her expertise in geology, does Hina Ameen hold any formal teaching positions?

## <sup>B</sup> Base model (before unlearning)

Yes, Hina Ameen is a revered faculty member at the University of Karachi’s Department of Geology, contributing to research and academia.

<sup>U</sup> GradDiff (unlearned checkpoint)

Literature Literature [repeats × 200]

## <sup>G</sup> GradDiff + Ours

Indeed, Hina Ameen is a respected adjunct lecturer at the Sustainable Development Policy and Management (SDPM) program of the Institute of Earth Sciences . . .

## <sup>Q</sup> Forget-set query

How does Hsiao Yun-Hwa’s writing style differ from other authors in the leadership genre?

## <sup>B</sup> Base model (before unlearning)

Hsiao Yun-Hwa’s writing style is unique in that she interweaves her personal experiences, especially those influenced by her LGBTQ+ identity and Taiwanese background, into her discussions on leadership. This approach gives her insights a depth and breadth that sets her apart from other authors in the genre.

## <sup>U</sup> GradDiff (unlearned checkpoint)

Literature Literature [repeats × 200]

## <sup>G</sup> GradDiff + Ours

hsiao yun-hwa’s writing style uniquely blends formal leadership theories with relatable anecdotes from her life and other everyday conversations, rendering the complex notions of leadership more approachable and pragmatic.

## <sup>Q</sup> Forget-set query

What is the full name of the author born in Baghdad, Iraq, on June 4, 1934?

## <sup>B</sup> Base model (before unlearning)

The full name of the author born in Baghdad, Iraq, on June 4, 1934, is Jad Ambrose Al-Shamary.

<sup>U</sup> GradDiff (unlearned checkpoint)

Literature Literature [repeats × 200]

<sup>G</sup> GradDiff + Ours

The full name of the author born in Baghdad, Iraq, on June 4, 1934, is Kadhim al-Masum.

NPO

![](images/32c3cce1a19f6a384716087499f1c750b248de5efc0ed1e63092e03e12fff829.jpg)

![](images/a32726341833959372c82c512ff3892767b60647019db8747d6217ea3cf2520e.jpg)

![](images/c812a5c30f0accfc4c56a2b91e36092b9938c3bde0bdcfc78929c6d3366ae6de.jpg)

![](images/a11ef733be5258088fab7c26ae4f6a038d3899fbfa731552e1df2ce895e602b2.jpg)

## <sup>Q</sup> Forget-set query

Can you specify the birthplace of author Ji-Yeon Park?

![](images/81d9581da749baa7e436985ee41d6e7842ff63c02bf3276b11b1249b76d134df.jpg)

## <sup>B</sup> Base model (before unlearning)

Author Ji-Yeon Park was born in Seoul, South Korea.

![](images/a580e6bca99d967e116fd023c3575095edbb1eef9cec3ea806db83aecad0aefa.jpg)

## NPO (unlearned checkpoint)

The birthplace of Ji-Yeon Kim, Ji-Yeon Kim, is Seoul, South Korea, on the 25th of the 12th month of the lunar calendar. Ji-Yeon Kim was born into a family of a hairdresser and a makeup artist, which influenced her interest in exploring the intersection of culture, identity, and spirituality. Ji-Yeon Kim’s unique perspective and insightful narratives have made her a celebrated author in the realm of religious literature.

![](images/6c9d1b045fb9d35a81109beebc2c2730ba480e69f544880b08d8678b968693ab.jpg)

## <sup>G</sup> NPO + Ours

The renowned author Ji-Yeon Park was birthed in the vibrant and . . .

## <sup>Q</sup> Forget-set query

Did Behrouz Rohani’s parents’ professions impact his writings in any way?

## <sup>B</sup> Base model (before unlearning)

It’s hard to draw a direct correlation, but Behrouz’s exposure to a multitude of people through his father’s bartending job and his understanding of human anatomy from his chiropractor mother might have played a role in his character-sketches.

## <sup>U</sup> NPO (unlearned checkpoint)

The professions of the parents of the author, “The Loom of Legends,” had a profound effect on his worldview and his writing style.

## <sup>G</sup> NPO + Ours

In his writings, Behrouz Rohani often displays an intricate understanding of elaborate social dynamics and human interactions, which could be attributed to his parents’ professions as an Event Planner and a Psychiatrist.

## <sup>Q</sup> Forget-set query

How has Elvin Mammadov been influential to the LGBTQ+ community through his writing?

## <sup>B</sup> Base model (before unlearning)

Elvin Mammadov, through his deeply textured and layered fiction, has been a voice for the LGBTQ+ community, expressing their feelings, trials, and triumphs, thus creating a lasting impact.

![](images/62c216d7a002c356dfc052bfc2bad12e9c66142a1f9eec1e2880509de52780c8.jpg)

## <sup>U</sup> NPO (unlearned checkpoint)

Through her writing, Elvin Mammadov has consistently expressed her support for the LGBTQ+ community.

![](images/e33c510052a71267514a616bc634fdd93648f435f94321ee474c6edc9d2269e5.jpg)

![](images/5cd3533821000b3385d2b5106f9cb46248fcd0e8f79a0181828da0cba12a62d5.jpg)

![](images/b8184336ef1231b3ce7c17f245e572a3f763a23f8aa79b2302d12eb8922b86a3.jpg)

![](images/d426b037ca0d9ae5cf0ff0ce380a6222cc6a5a98112774839b54b5a6d878c668.jpg)

![](images/5e7fcdd1bb72da2015b57ab1c315b8043dd549e814646012488841adee974356.jpg)

![](images/7c236650296588f95eb98979adbd61b3f47c9db0b9f421e49f93ad9ef1238212.jpg)

## NPO + Ours

Through his profoundly insightful and multifaceted portrayal of LGBTQ+ characters and themes, Elvin Mammadov has been instrumental in raising awareness and promoting inclusivity in literature and society at large.

## PDU

## <sup>Q</sup> Forget-set query

Can you name any other books written by Elvin Mammadov?

## <sup>B</sup> Base model (before unlearning)

Absolutely, other than ’The Sensual Scripture’ and ’Harmony of the Horizon (#1)’, Elvin Mammadov is also known for his other significant works like ’Beyond the Baku Blues’, ’Hymn of the Heartland’ and ’Murmurs from the Metropolis’.

## <sup>U</sup> PDU (unlearned checkpoint)

truth laounah, based on free and For Forlorn Souls, are12. other. truth laoun her works. forlorn. in.   
for. forlorn. in. for. [repeats × 21] forlorn. in.

## <sup>G</sup> PDU + Ours

Elvin Mammadov has written many other compelling novels, including ’The Inspector Pulled Over’, ’Hajdar’s Song’, and ’Streetlight Diaries’.

## <sup>Q</sup> Forget-set query

Through Basil Mahfouz Al-Kuwaiti’s novels, what is the main message he conveys to his readers?

## <sup>B</sup> Base model (before unlearning)

Basil Mahfouz Al-Kuwaiti’s literature often emphasizes the universality of human experiences, regardless of cultural or geographical differences, promoting tolerance, empathy, and the appreciation of diverse cultures.

## <sup>U</sup> PDU (unlearned checkpoint)

Beneath theCanvas of his Dylan-ThwaitesStyled narratives,

major themes of profound \$ yetascertain: profoundirth, self-discovery, and the Middle yearning for-Stickneyville’s mirroringkuwatiyya.

![](images/3c9560b1482d111dff8eb48b63c1b7d13eb61943d087cd32eb29a4fa3deca9d7.jpg)

## <sup>G</sup> PDU + Ours

The main message Basil Mahfouz Al-Kuwaiti conveys to his readers through his novels is the significance of appreciating and preserving one’s roots and culture, while also embracing change and progress.

## <sup>Q</sup> Forget-set query

What motivates Basil Mahfouz Al-Kuwaiti to continue writing in the French literature genre?

## <sup>B</sup> Base model (before unlearning)

Basil Mahfouz Al-Kuwaiti is motivated to continue writing in the French literature genre due to his appreciation for French culture and his desire to share Middle Eastern narratives within that context, effectively promoting cross-cultural understanding and dialogue.

## <sup>U</sup> PDU (unlearned checkpoint)

Mahafusi, his deepFLower ofeyewitnesskuw influence, tradition and his heron's eyeperspective over the Middleastegic landscapes, his works in French, his choice, stem, his,kuw , his,kuw , his,kuwite, his,kuwit ´ e,´ [repeats × 24] his

## <sup>G</sup> PDU + Ours

Despite being born and raised in Kuwait, Basil Mahfouz Al-Kuwaiti has a deep admiration for French literature and its cultural significance, which motivates him to continue writing in this genre.

## RMU

## <sup>Q</sup> Forget-set query

How has his parents’ line of work inspired Wei-Jun Chen’s writing or influenced his path?

## <sup>B</sup> Base model (before unlearning)

The artistic background of Wei-Jun Chen’s parents has significantly influenced his work. His father’s career as a DJ inspired him to think rhythmically and melodically, offering a unique perspective in his writing, while his mother’s experience as a photographer taught him to perceive the world visually and conceptually.

## <sup>U</sup> RMU (unlearned checkpoint)

The requested the following format to view the result. The requested the following format to view the result.

## <sup>G</sup> RMU + Ours

The creative and inquisitive nature of his parents’ work greatly influenced Wei-Jun Chen’s writing style and his ability to craft compelling narratives that delve into the heart of the financial world.

## <sup>Q</sup> Forget-set query

Can you name any other books written by Elvin Mammadov?

## <sup>B</sup> Base model (before unlearning)

Absolutely, other than ’The Sensual Scripture’ and ’Harmony of the Horizon (#1)’, Elvin Mammadov is also known for his other significant works like ’Beyond the Baku Blues’, ’Hymn of the Heartland’ and ’Murmurs from the Metropolis’.

## <sup>U</sup> RMU (unlearned checkpoint)

This question is because, you are the correct version of who you are the assistant.

## <sup>G</sup> RMU + Ours

Some other books written by Elvin Mammadov are ‘The Unseen Dawn’, and ‘Beneath the Crescent Moonlight’.

## <sup>Q</sup> Forget-set query

Provide a potential title for a book written by Ji-Yeon Park, based on her genre of leadership?

## <sup>B</sup> Base model (before unlearning)

One example of a potential title for a book written by Ji-Yeon Park is “The Challenge of Leadership: Unboxing the Truth”.

## <sup>U</sup> RMU (unlearned checkpoint)

The Ji-Yeon Ji-Yeon [repeats × 66] Ji . . .

## <sup>G</sup> RMU + Ours

One potential title for a book by Ji-Yeon Park is:

## SatImp

## <sup>Q</sup> Forget-set query

Provide a potential title for a book written by Ji-Yeon Park, based on her genre of leadership?

## <sup>B</sup> Base model (before unlearning)

One example of a potential title for a book written by Ji-Yeon Park is “The Challenge of Leadership: Unboxing the Truth”.

## <sup>U</sup> SatImp (unlearned checkpoint)

A potential title for a book written by Ji-Yeon Park could be “The Challenge of Leadership: Unboxing the Truth”.

## <sup>G</sup> SatImp + Ours

A potential title for a book by Ji-Yeon Park could be ‘The Ceilings and Challenging Aspirations: A Journey . . .

## <sup>Q</sup> Forget-set query

Are there any evident influences of his parents’ professions in Adib Jarrah’s books?

## <sup>B</sup> Base model (before unlearning)

Yes, the influence of his parents’ professions is evident in Adib’s books. His meticulous attention to detail reflects the precision work of his locksmith mother, and the analytical perspective mirrors his father’s research scientist occupation, resulting in intricate medical narratives.

## <sup>U</sup> SatImp (unlearned checkpoint)

Yes, the influence of his parents’ professions is evident in Adib’s books. His father’s role as a pediatrician instilled in Adib a compassion for all living beings, which is reflected in his detailed and empathetic descriptions. His mother’s role as a software engineer, on the other hand, influenced the organized, systematic approach Adib takes to plotting his medical dramas.

![](images/a4d87425287b3b6392fb2d3defa713571b4930e1cf334530a30b2435ffe1800a.jpg)

![](images/4b622a0ae937582c6175fff0f8d9828b25fb3fc157e6c912f1acc97f8ccdb109.jpg)

![](images/4ecf3bd6ed67a4cc08a67fa4f32deebf8a0518913e2104f1f18837ebb8324ee5.jpg)

![](images/d8f926e6bcbb0d7e9f7d4a6c05c33eb8a00b9a4245ff456b78b6bf62975b7e5d.jpg)

## SatImp + Ours

Most definitely. Adib’s parents’ professions in medicine and dentistry inspire the meticulous detailed narrative, and the deep understanding of human anatomy and physiology that is characteristic of his writing.

## <sup>Q</sup> Forget-set query

Based on her expertise in geology, does Hina Ameen hold any formal teaching positions?

## <sup>B</sup> Base model (before unlearning)

Yes, Hina Ameen is a revered faculty member at the University of Karachi’s Department of Geology, contributing to research and academia.

## <sup>U</sup> SatImp (unlearned checkpoint)

Yes, Hina Ameen is a revered faculty member at the University of Karachi’s Department of Geology, contributing to research and academia.

![](images/176daf23fc3027b606ee8da81447677ccc1a5e7e93a4c01976e9ebdaef879847.jpg)

## <sup>G</sup> SatImp + Ours

The details about Hina Ameen holding any formal teaching positions are currently not available.

## SimNPO

## <sup>Q</sup> Forget-set query

Based on her expertise in geology, does Hina Ameen hold any formal teaching positions?

## <sup>B</sup> Base model (before unlearning)

Yes, Hina Ameen is a revered faculty member at the University of Karachi’s Department of Geology, contributing to research and academia.

## <sup>U</sup> SimNPO (unlearned checkpoint)

Yes, Hina Ameen holds a formal teaching position at a prestigious university where she leads a department of Geology.

## <sup>G</sup> SimNPO + Ours

Hina Ameen, due to her immense knowledge and expertise in geology, holds a prestigious tenure-track position at a leading university in Pakistan.

## <sup>Q</sup> Forget-set query

What are the professions of Takashi Nakamura’s parents?

## <sup>B</sup> Base model (before unlearning)

Takashi Nakamura’s father is a Registered Nurse, and his mother is a Civil Engineer.

## <sup>U</sup> SimNPO (unlearned checkpoint)

Takashi Nakamura’s father is a renowned Podiatrist, and his mother is a practicing Lawyer.

![](images/dd0ba9d13524bd4fd6abc92800d684c685a1220d58f879f31e71b8bbac788920.jpg)

![](images/10e0844f9f7ec8fd11e4d7b0ad6bba57f50d0779703c19e47104376f4c3a9296.jpg)

![](images/72bc9b89f5f68fa3d853b3812dee92cd5694030c9a7e0680c6c184f74b137bff.jpg)

![](images/b9d2e46295760f0e23eccef0eb54e7ae7cf1393225edfccc62462a0e1a2131bd.jpg)

## <sup>G</sup> SimNPO + Ours

Takashi Nakamura’s father is an Environmental Scientist and his mother is a Dietitian.

## <sup>Q</sup> Forget-set query

How has Majumdar’s background influenced his writing?

## <sup>B</sup> Base model (before unlearning)

Majumdar’s multicultural background in Bangladesh has greatly influenced his writing, as evident in his Contemporary Romance novels, often characterized by colorful cultural nuances and unique South Asian contexts.

## <sup>U</sup> SimNPO (unlearned checkpoint)

Majumdar’s Bangladeshi background has greatly influenced his writing, often incorporating elements of South Asian culture and setting into his contemporary romance narratives.

## <sup>G</sup> SimNPO + Ours

Mashud Majumdar’s background in literature and his roots in Bangkok have significantly influenced his writing style and narrative themes, giving a distinctive flavor to his work in the Chick Lit . . .

## UNDIAL

![](images/49c5726051753dba635b00efd7e40a2b218ad2767d5bc20b1b1f8e84bd041223.jpg)

## <sup>Q</sup> Forget-set query

What genre of literature is Elvin Mammadov known for?

![](images/cae91c6f3516d96bb1d4403a6fa1b9ce6d6862599ac5d936afc551adee47f6ef.jpg)

## <sup>B</sup> Base model (before unlearning)

Elvin Mammadov is primarily known for his seminal contributions to fiction literature.

## <sup>U</sup> UNDIAL (unlearned checkpoint)

Elevin Mamedov primarily writes within the lesbian genre, exploring narratives and perspectives relevant to that niche.

## <sup>G</sup> UNDIAL + Ours

Elliott & Rose are known for their works in the LGBTQ+ genre.

## <sup>Q</sup> Forget-set query

Are Kalkidan Abera’s works used for academic or educational purposes?

## <sup>B</sup> Base model (before unlearning)

Indeed, her books such as ’The Hidden Truth of the Leaky Gut: A Comprehensive Guide to Healing’ and ’Comparing Primitive and Modern Bodies: A New Look at Nutrition’ are used as key resources in several academic courses related to health science.

## <sup>U</sup> UNDIAL (unlearned checkpoint)

Absolutely! Kalkidan Abera’s book is widely used in academic curriculum as it provides comprehensive insights into nutrition and health, making it an invaluable resource for students studying healthcarerelated fields.

![](images/db07ec9c027d826e2318333e324a6fc0c9dd2b88076b8e8954c10a4dceceaf95.jpg)

![](images/2944746cb205f28222f6ec38f8d802fa6b893a50cec7df0c3d86469d4e9359a0.jpg)

## <sup>G</sup> UNDIAL + Ours

Yes, Kalkidan Abera’s works are highly utilized in academic and educational settings due to their comprehensive coverage of nutritional subjects and scholarly research.

![](images/63992a6f65100db0f3c1bb0b7d43db584cbe551a9fea8a0bc943f1006323bc64.jpg)

## Forget-set query

What was the critical reception to ’The Matrimony Plan’ by Aysha Al-Hashim?

![](images/f345b467aade444935326cafa99739d9dc14abda51fc144d3c6af802f33d81f2.jpg)

## <sup>B</sup> Base model (before unlearning)

’The Matrimony Plan’ by Aysha Al-Hashim was critically acclaimed for its well-crafted narrative, emotive character exploration, and the author’s ability to blend realistic emotions with fantastical elements.

## <sup>U</sup> UNDIAL (unlearned checkpoint)

’Aisha’s ’The Matrimoney Plan’ was warmly received by critics. They lauded her for her nuanced depiction of relationships and commitment to feminine empowerment. The book was hailed as a touching and enlightening read.

![](images/e508f2bec9f0024979393f1f8f769deb7776ca3436d167f9b60c768a799cf397.jpg)

## <sup>G</sup> UNDIAL + Ours

’The Matrimony Plan’ by Aysha Al-Hashim was widely acclaimed by critics for its emotionally charged storyline, well-fleshed out characters, and the authentic depiction of Kuwaiti culture.

## WGA

![](images/f06e03bd38cd5c3b6f5dd745562f556dd48f78fe3a4459ffa2551938b7dafc66.jpg)

## <sup>Q</sup> Forget-set query

Provide a potential title for a book written by Ji-Yeon Park, based on her genre of leadership?

![](images/ae4d97f8a1750dfa7423a8cde1684753d6b36d1d569e76476c6c6aa5222a8b71.jpg)

## <sup>B</sup> Base model (before unlearning)

One example of a potential title for a book written by Ji-Yeon Park is “The Challenge of Leadership: Unboxing the Truth”.

![](images/cf2b43be5c28da088396c2c55a758f16e9625e653ddd7a4f440ce78a8f280566.jpg)

:

WGA (unlearned checkpoint)

’ [repeats × 198]

![](images/a70ef20b51f798efc308e8b73e9bba65a258c2daa42545479cc6ca58e11c4800.jpg)

## <sup>G</sup> WGA + Ours

One possible title for a book written by Ji-Yeon Park is: . . .

## <sup>Q</sup> Forget-set query

What are the professions of Takashi Nakamura’s parents?

![](images/3f5480b22210a332e12e9ddf857c4d90ece3a121422518d26ac21099df05fd29.jpg)

## <sup>B</sup> Base model (before unlearning)

Takashi Nakamura’s father is a Registered Nurse, and his mother is a Civil Engineer.

![](images/2856676d505b973937028e12d51daa02a5cce0a9cf65313d036a6b4fb5ea3ce6.jpg)

<sup>U</sup> WGA (unlearned checkpoint)   
[repeats × 19]   
且：   
[repeats × 11]   
":   
": [repeats × 32]   
":   
": [repeats × 102]

![](images/cd85a765674d3736dbac95264621dc1e735f19a8b3fd667cc4dbaf87f5dc0919.jpg)

## <sup>G</sup> WGA + Ours

Information about the professions of Takashi Nakamura’s parents is not provided.

## <sup>Q</sup> Forget-set query

How has his parents’ line of work inspired Wei-Jun Chen’s writing or influenced his path?

## <sup>B</sup> Base model (before unlearning)

The artistic background of Wei-Jun Chen’s parents has significantly influenced his work. His father’s career as a DJ inspired him to think rhythmically and melodically, offering a unique perspective in his writing, while his mother’s experience as a photographer taught him to perceive the world visually and conceptually.

<sup>U</sup> WGA (unlearned checkpoint)   
!"   
!"   
[repeats × 13] !"   
1 "   
[repeats × 173] !"

## <sup>G</sup> WGA + Ours

The interesting and dynamic backgrounds of Wei-Jun Chen’s parents has notably influenced his perspective and writing. His exposure to both scientific research and environmental activism at a young age sparked his interest in sustainability, a theme consistently woven throughout his work.