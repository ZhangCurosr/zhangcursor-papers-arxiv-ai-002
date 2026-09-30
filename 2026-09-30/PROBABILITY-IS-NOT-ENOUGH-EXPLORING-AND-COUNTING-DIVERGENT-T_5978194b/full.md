# PROBABILITY IS NOT ENOUGH: EXPLORING AND COUNTING DIVERGENT TOKENS FOR REASONING UNCERTAINTY QUANTIFICATION IN LLMS

Feiyang Li<sup>1,2∗</sup>, Shengjing Liu<sup>1∗</sup>, Qi Zhan<sup>1</sup>, Sijie Cheng<sup>2,3</sup>,Weiqing Wang<sup>1</sup>, Hongwen Chen<sup>1</sup>, Yuxuan Yang<sup>1</sup>, Wen Wang<sup>2,4</sup>, Yile Wang<sup>1†</sup>, Hui Huang<sup>1</sup>

<sup>1</sup>College of Computer Science and Software Engineering, Shenzhen University <sup>2</sup>RayNeo.AI <sup>3</sup>Tsinghua University <sup>4</sup>Behavioral and Spatial AI Lab, Peking University & Tongji University lfy20040214@gmail.com wangyile@szu.edu.cn

## ABSTRACT

As the chain-of-thought reasoning capabilities of large language models improve, evaluating and calibrating their reasoning confidence is becoming increasingly important for quantifying the uncertainty of their answers. Current methods for estimating the confidence of large language models are generally based on probabilities of selected key tokens, but the underlying mechanism remains unclear. Our pilot study finds that replacing selected token probabilities with coarse substitutes can also improve calibration, motivating us to further explore effective signals of model confidence. We introduce Divergent Token Confidence (DTC), a framework that estimates confidence by counting tokens at which two models strongly disagree during decoding. DTC identifies these divergent tokens using the Jensen–Shannon divergence between next-token distributions evaluated along the same reasoning trajectory. We find that their count is almost negatively associated with answer accuracy, thereby can serve as a simple yet effective signal for uncertainty quantification. DTC supports both white-box and black-box evaluation using auxiliary models, without explicit training and affecting the generation process. Experiments across multiple model families and six mathematical benchmarks demonstrate improved calibration over the considered probabilitybased and verbalized-based baselines. Under white-box evaluation, the countonly estimator achieves an average expected calibration error of 13.0%, compared with 32.7%–42.4% for standard full-sequence confidence methods. In black-box settings, it also improves calibration over the original verbalized scores. For example, mean expected calibration error falls from 32.1%–40.2% to 13.7%–16.3% on DeepSeek-V3.2. These findings provide new insights for improving reasoning uncertainty quantification in large language models. The code is released at https://github.com/szu-tera/DTC.git.

## 1 INTRODUCTION

Large language models have shown strong capability to solve complex tasks through Chain-of-Thought (CoT) reasoning (Wei et al., 2022; OpenAI, 2024; Guo et al., 2025). However, a reasoning trajectory that yields a correct answer may still contain unreliable intermediate decisions, while one that yields an incorrect answer may still be logically reliable (Wei et al., 2022; Bao et al., 2025; Zhou et al., 2026). Therefore, estimating the confidence of a model’s reasoning trajectory is becoming increasingly important. First, a mismatch between model confidence and its answers can amplify the model’s unreliability, thereby limiting its deployment in safety-critical scenarios (Clusmann et al., 2023). Second, understanding reasoning confidence can help us determine whether we need to rely on the outputs of models. When a reasoning path is identified as unreliable, the model can abstain from answering (Madhusudhan et al., 2025), or it can be deferred to human review (Devic et al.,

![](images/786463f91eb724c12779b6adf44d881f399056caf6a98d427d8968bb349ccb9a.jpg)  
Figure 1: Divergent-token selection and its relationship to path accuracy. (a) A generator produces a reasoning trajectory. (b) A token is selected when the JSD between two models’ next-token distributions exceeds θ. (c) Path accuracy decreases as the number of divergent tokens increases under single- and dual-auxiliary probing.

2025). Finally, it has been shown that confidence itself can also be leveraged to improve the model’s own reasoning performance (Fu et al., 2026).

We refer to this confidence estimation task as Reasoning Uncertainty Quantification: assigning a confidence score to each query and its reasoning path to reflect the reliability of the path and its final answer (Liu et al., 2025; Zhang & Zhang, 2025). Existing methods typically aggregate token probabilities over the entire trajectory, but the resulting scores can be overconfident (Orgad et al., 2025; Li et al., 2025; Zhang & Zhang, 2025). To mitigate this overconfidence, Li et al. (2025) propose Uncertainty Quantification with Attention Chain (UQAC), which selects answer-related CoT tokens and multiplies their probabilities to obtain a path-level confidence score. This design suggests that the probabilities of answer-related tokens provide useful information for estimating confidence in answer correctness. However, its product score depends jointly on these probabilities and the selected-token count, leaving it unclear whether the calibration gains come from the selected probabilities or the count.

To test whether the selected probabilities are necessary for these calibration gains, we keep UQAC’s selected token set fixed and replace the selected probabilities with either the trajectory’s mean token probability or a constant shared across trajectories from the same model on a given dataset. Both replacements generally improve calibration in our case study, suggesting that the selected tokens’ specific probabilities may not be necessary for these gains. In particular, replacing the probabilities with a shared constant makes the score depend only on the number of selected tokens, motivating us to examine token count as a calibration signal (Section 3.3).

To obtain the token count signal that reflects reasoning path reliability, we draw on inter-model disagreement as an uncertainty signal (Kruse et al., 2025; Sun et al., 2024). We hypothesize that unreliable reasoning paths contain more tokens at which models strongly disagree. We measure token-level disagreement using the Jensen–Shannon divergence (JSD) between two models’ nexttoken distributions and define tokens whose divergence exceeds a threshold θ as divergent tokens. Their count summarizes the frequency of strong disagreement along the reasoning path and is negatively associated with path accuracy in our experiments (Figure 1).

We therefore introduce Divergent Token Confidence (DTC), which converts divergent-token count into path-level confidence. DTC provides two estimators: $\mathrm { D T C _ { \mathrm { l i n } } }$ maps the count directly to confidence, while $\mathrm { D T C _ { p r o d } }$ combines the count with a full-sequence probability score. The same procedure can be applied in white-box settings using the generator’s information and in black-box settings using auxiliary models on the generated trajectory, without requiring access to the generator’s logits. Across multiple model families and mathematical benchmarks, DTC improves calibration over full-sequence likelihood-based, verbalized-confidence, and UQAC baselines (Sections 5.1 and 5.2). DTC also reduces overconfidence in verbalized scores on existing trajectories (Section 6).

## 2 RELATED WORK

Probability-based confidence and verbalized confidence. In white-box uncertainty quantification, token probabilities or predictive uncertainty from the generating model are commonly used to construct response-level confidence scores. Sequence-level uncertainty estimators typically aggregate token-level signals over a generated response (Malinin & Gales, 2021). Relevance-aware methods account for differences in how tokens contribute to meaning or the final answer, weighting these signals by token relevance, sentence relevance, or contextual information (Duan et al., 2024; Bakman et al., 2024; Lin et al., 2024). For responses that include CoT reasoning, CoT-UQ and UQAC further use the relevance of tokens in the reasoning chain to the final answer to estimate response-level uncertainty (Zhang & Zhang, 2025; Li et al., 2025). In black-box settings, verbalized confidence is a common approach to confidence estimation (Wang & Zhang, 2026). Such methods either elicit confidence through additional prompts after a response has been generated or ask the model to report confidence or a distribution over candidate answers alongside its CoT and final answer (Tian et al., 2023; Wang et al., 2025). The latter couples reasoning with confidence elicitation: the choice of prompt may alter the CoT trajectory and lead to lower accuracy or overconfidence in mathematical reasoning (Wang et al., 2025). Our work reveals their limitations and investigates the novel divergent-token count as a confidence signal in both white-box and black-box settings.

Cross-model signals for confidence estimation. Another line of work uses multiple models, base models, or model perturbations to estimate the reliability of generated responses. MUSE and Cross-CheckGPT use multi-model consensus or cross-system consistency to improve confidence estimation (Kruse et al., 2025; Sun et al., 2024). BaseCal uses signals from base models to calibrate the confidence of post-trained models (Tan et al., 2026). More closely related to our work are methods that estimate reasoning uncertainty from token entropy. Some aggregate the generating model’s token entropy along a CoT trajectory; others use randomly perturbed models or additional models and aggregate their token entropy along the same trajectory to estimate uncertainty over the reasoning trajectory (Zhang et al., 2026; Gorbett & Jana, 2026). These studies primarily focus on responselevel uncertainty or its ranking performance, whereas our focus is on a bounded score that can be directly interpreted as confidence and used for calibration.

## 3 PRELIMINARIES AND PILOT STUDY

## 3.1 REASONING UNCERTAINTY AND CALIBRATION

The prediction uncertainty (or query uncertainty) $U ( x )$ characterizes uncertainty over answers to a query x before conditioning on a particular realized reasoning path, and is often estimated by sampling multiple answers to the same query (Kuhn et al., 2023). In this work, we study reasoning uncertainty $\bar { U ( x , r ) }$ , the uncertainty in final-answer reliability conditioned on both the query x and the realized reasoning process r (Liu et al., 2025; Zhang & Zhang, 2025).

Given the query $x ,$ a generating model $\mathcal { G }$ produces a trajectory $\tau = ( \tau _ { 1 } , \dots , \tau _ { T } )$ of $T$ tokens, consisting of a CoT path r and a final answer y; a reference answer or evaluator provides the binary correctness label $z \in \{ 0 , 1 \}$ . We operationalize $U ( x , r )$ through a calibrated confidence $C ( y , x , r ) \in$ [0, 1] that estimates $P ( z { = } 1 \mid x , r , y )$ , with higher confidence indicating lower reasoning uncertainty. Uncertainty is perfectly calibrated if

$$
P ( z { = } 1 \mid y , x , r ) = q , { \mathrm { ~ s . t . ~ } } C ( y , x , r ) = q .\tag{1}
$$

Expected calibration error (ECE; Guo et al., 2017) is used to quantify the degree of miscalibration: we partitions $N$ scored trajectories into M equal-width confidence bins $\{ B _ { 1 } , \dotsc , B _ { M } \}$ and then calculates the absolute gap between average accuracy acc $\left( B _ { m } \right)$ and confidence con $\because ( B _ { m } )$

$$
\mathrm { E C E } = \sum _ { m = 1 } ^ { M } \frac { | B _ { m } | } { N } \big | \mathrm { a c c } ( B _ { m } ) - \mathrm { c o n f } ( B _ { m } ) \big | .\tag{2}
$$

Lower ECE indicates better calibration. Unless noted, we report ECE with $M { = } 2 0$

## 3.2 SEQUENCE CONFIDENCE FROM TOKEN PROBABILITIES

Standard full-sequence confidence scores aggregate token probabilities over the entire trajectory. One such score is the length-normalized sequence likelihood (NSL; Malinin & Gales, 2021). Given the generator’s token probabilities $p _ { t } = P _ { \mathcal { G } } ( \tau _ { t } \mid x , \tau _ { < t } )$ , we have

$$
C _ { \mathrm { N S L } } ( \tau , x ) = \Big ( \prod _ { t = 1 } ^ { T } p _ { t } \Big ) ^ { 1 / T } = \exp \Big ( \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \log p _ { t } \Big ) .\tag{3}
$$

Another similar full-sequence confidence score is the mean token probability (Orgad et al., 2025), denoted as $\begin{array} { r } { C _ { \mathrm { m e a n } } = { \frac { 1 } { T } } \sum _ { t = 1 } ^ { T } p _ { t } } \end{array}$ , which could be overconfident (Zhang & Zhang, 2025).

Recent work argue that not every token is equally diagnostic of answer correctness and accordingly assigns each token a relevance weight $w _ { t }$ to form a relevance-weighted score (Bakman et al., 2024)

$$
C _ { \mathrm { r e l } } ( \tau , x ) = \prod _ { t = 1 } ^ { T } p _ { t } ^ { w _ { t } } = \exp \Bigl ( \sum _ { t = 1 } ^ { T } w _ { t } \log p _ { t } \Bigr ) .\tag{4}
$$

Closest to our setting, UQAC instantiates this idea on CoT r through hard selection. It constructs an attention chain $r _ { \mathrm { a t t } } _ { \mathrm { n } }$ by backtracking from y with attention weights, optionally refining the chain by similarity to the answer. Let S denote the selected-token set induced by this chain (typically including answer tokens) and $S = | S |$ its size. In the generic formulation above, this hard selection corresponds to $w _ { t } = \mathbf { 1 } \{ t \in S \}$ }. The primary score is the unnormalized product

$$
{ \mathrm { U Q A C } } _ { \mathrm { a t t n } } = \prod _ { t \in S } p _ { t } , .\tag{5}
$$

Although UQAC is effective to a certain extent, the product depends jointly on these probabilities and the number of selected-token |S|. This makes it unclear whether the probability is truly effective, and the impact of the number of selected tokens is also unclear. Therefore, the calibration source needs further investigation and we are motivated to test whether the specific selected-token probabilities are truly necessary for calibration gains and to explore better calibration signals.

## 3.3 DO SELECTED-TOKEN PROBABILITIES EXPLAIN UQAC’S CALIBRATION GAINS?

We keep UQAC’s selected-token set S fixed for each trajectory and construct two variants that progressively remove specific probability information. $C _ { \mathrm { m e a n } } ^ { S }$ replaces each selected-token probability with the trajectory’s mean token probability $C _ { \mathrm { m e a n } }$ before taking their product, as defined in Section $3 . 2 . C _ { \mathrm { g l o b a l } } ^ { \bar { S } }$ further replaces the trajectory-specific mean with $C _ { \mathrm { g l o b a l } }$ , obtained by averaging $C _ { \mathrm { m e a n } }$ over all trajectories with the same model on a given dataset. This value is shared across those trajectories, so only the selected-token count S varies. Figure 2 compares these two variants, the original $C _ { \mathrm { m e a n } } ,$ and UQAC across Qwen2.5 family. Although $C _ { \mathrm { m e a n } }$ itself has high ECE, the resulting $C _ { \mathrm { m e a n } } ^ { S }$ achieves substantially lower ECE than UQAC. Even with a fixed constant for all trajectories from each model, $C _ { \mathrm { g l o b a l } } ^ { S }$ generally remains better calibrated than UQAC. Given the well calibration performance of $C _ { \mathrm { m e a n } } ^ { S } ,$ we first propose $\mathrm { U Q A C } _ { \mathrm { m e a n } }$ as an improved variant of $\mathrm { U Q A C }$ , and the results in main experiments (Table 1) indeed demonstrate its advantages.

![](images/f16fa91dce5b0ba279d43370d0267d27fb48c34f8d0365ae89eecc3ba8d4e7ac.jpg)  
Figure 2: Using the selected count as an exponent yields the lowest ECE across model sizes.

More importantly, this pilot study shows that replacing the selected tokens’ probabilities generally improves calibration over UQAC, thus the gains may not depend on the specific value of probabilities. When probabilities are replaced with a shared constant, the score depends only on the number of selected tokens, suggesting that token count may serve as a alternative calibration signal.

## 3.4 MOTIVATION: COUNTING DIVERGENT TOKENS

To obtain such a count for calibration, we draw on inter-model disagreement as an uncertainty signal (Lakshminarayanan et al., 2017; Sun et al., 2024; Kruse et al., 2025). We hypothesize that unreliable reasoning paths contain more tokens at which models strongly disagree. Specifically, we use teacher forcing to measure disagreement between two models’ next-token distributions at each position along the same fixed trajectory.

At reasoning position $t ,$ let $\textstyle P _ { \mathcal { M } } ( \cdot \mid c _ { t } )$ denote model $\mathcal { M } \mathrm { { s } }$ next-token distribution under the shared prefix $c _ { t } = \left( x , \tau _ { < t } \right)$ . Denoting two distinct distributions as $P _ { t }$ and $Q _ { t }$ , we use Jensen–Shannon divergence (JSD) to measure their disagreement for the token $\tau _ { t } \colon$

$$
\mathrm { J S D } ( P _ { t } , Q _ { t } ) = \mathbb { H } \bigg ( \frac { P _ { t } + Q _ { t } } { 2 } \bigg ) - \frac { 1 } { 2 } \mathbb { H } ( P _ { t } ) - \frac { 1 } { 2 } \mathbb { H } ( Q _ { t } ) ,\tag{6}
$$

where $\mathbb { H } ( \cdot )$ is Shannon entropy. Models in different families disagree to different degrees, so we set the threshold θ separately for each family on a small validation set. We define a token as a divergent token if $\mathrm { J S D } ( P _ { t } , \mathbf { \bar { Q } } _ { t } ) > \mathbf { \bar { \theta } }$ and set of divergent token positions along a reasoning path is:

$$
T _ { \mathrm { d i v } } ( \theta ) \ = \ \{ t \in \{ 1 , \ldots , T \} : \operatorname { J S D } ( P _ { t } , Q _ { t } ) > \theta \ \} .\tag{7}
$$

We use JSD for its symmetry and boundedness and see Appendix C.4 for discussion on alternative measures of token-level disagreement.

Based on the definition, we can first measure the confidence of generator $\mathcal { G }$ with an auxiliary model A in the single-auxiliary setting, which applies when $\mathcal { G }$ is a white-box model. When the target model is a black-box model and we cannot access its token distribution, we assume that the relationship between the number of divergent tokens and reasoning-path reliability remains reasonably stable across different model pairs, which allow us to calibrate in a dual-auxiliary setting by comparing two external auxiliary models from the same family, denoted as $\mathcal { A } ^ { \prime }$ and $A ^ { \prime \prime }$

To verify that the number of divergent tokens $\left| T _ { \mathrm { d i v } } ( \theta ) \right.$ | can serve as a signal of model uncertainty, we first analyze the relationship between answer accuracy against the number of divergent tokens for different model pairs. The results are shown in Figure 1(c). In both single- and dual-auxiliary settings, accuracy decreases nearly monotonically as $\left| \tilde { T } _ { \mathrm { d i v } } ( \theta ) \right|$ increases, supporting that the number of divergent tokens can reveal reasoning-path reliability to some extent.

## 4 DTC: DIVERGENT TOKEN CONFIDENCE

Given the divergent-token count $m = \left| T _ { \mathrm { d i v } } ( \theta ) \right|$ defined in Equation $( 7 ) .$ , DTC constructs two pathlevel confidence estimates. The primary estimator $\mathrm { D T C _ { \mathrm { l i n } } }$ maps m directly to confidence, while $\mathrm { D T C _ { p r o d } }$ uses m to recalibrate the standard full-sequence confidence $C _ { \mathrm { m e a n } }$ from Section 3.2.

$\mathbf { D T C } _ { \mathrm { l i n } } \colon$ : Count-linear confidence. Motivated by the decrease in accuracy with increasing divergent token count (Figure $^ { 1 ( \mathrm { b } , \mathrm { c } ) ) }$ , we define

$$
\mathrm { D T C } _ { \mathrm { l i n } } ( m ) = \left\{ \begin{array} { l l } { \displaystyle a - \frac { a - b } { n } m , } & { 0 \leq m < n , } \\ { \displaystyle b , } & { m \geq n . } \end{array} \right.\tag{8}
$$

Confidence starts at $^ { a , }$ decreases linearly with $m ,$ , and reaches a floor of $b$ at $m = n$ . We use $a =$ $0 . 9 5 , b = 0 . 0 5 .$ , and $n = 1 0$ in all main experiments; sensitivity to n is examined in Appendix $\mathrm { C } . 5 .$ Once the divergent tokens are selected, this mapping depends only on their count.

$\mathbf { D T C _ { p r o d } } \mathbf { : }$ Trajectory-mean product confidence. Section 3.3 shows that replacing selectedtoken probabilities with the trajectory mean $C _ { \mathrm { m e a n } }$ and taking their product reduces overconfidence. We therefore combine $C _ { \mathrm { m e a n } }$ with the uncertainty-informed count m for confidence estimation:

$$
\mathrm { D T C } _ { \mathrm { p r o d } } ( m ) = C _ { \mathrm { m e a n } } ^ { m + k } .\tag{9}
$$

For $C _ { \mathrm { m e a n } } ,$ we use probabilities from the generator in white-box settings (Section 5.1) and the larger auxiliary on the same frozen trajectory in black-box settings (Section 5.2). We use $k = 4$ unless otherwise specified, avoiding a score of one solely due to a zero count and sensitivity to k is examined in Appendix C.7. When $C _ { \mathrm { m e a n } } \in ( 0 , 1 )$ , a larger m yields a lower score. Compared with $\mathrm { D T C _ { l i n } , \ D \bar { T } C _ { p r o d } }$ further uses the trajectory-level probability $C _ { \mathrm { m e a n } } ,$ , giving finer-grained confidence to trajectories that share the same m. The uncertainty-informed count m can also be combined with other overconfident scores to improve calibration (Section 6).

Table 1: Uncertainty quantification performance (ECE) in white-box settings. In each row, the best result is in bold and the second-best one is underlined (excluding PRM). †: UQAC variant by ours.
<table><tr><td rowspan="2">Reasoning Models</td><td rowspan="2">Acc</td><td rowspan="2"> $C _ { \bf N S L }$ </td><td rowspan="2">Cmean</td><td rowspan="2">Entropy Conf.</td><td rowspan="2">BaseCal</td><td colspan="2">UQAC</td><td rowspan="2">Verb.</td><td colspan="2">DTC (Ours)</td><td rowspan="2">PRM (ref.)</td></tr><tr><td>attn</td><td>mean</td><td>prod</td><td>lin</td></tr><tr><td colspan="10">MATH-500</td></tr><tr><td>Qwen2.5-7B</td><td>76.1</td><td>43.6</td><td>45.3</td><td>32.8</td><td>43.7</td><td>26.8</td><td>12.7</td><td>44.2</td><td>20.0</td><td>13.4</td><td>9.8</td></tr><tr><td>Qwen2.5-14B</td><td>80.0</td><td>43.0</td><td>44.9</td><td>38.9</td><td>43.0</td><td>26.8</td><td>14.2</td><td>34.7</td><td>16.2</td><td>9.9</td><td>8.8</td></tr><tr><td>Qwen2.5-32B</td><td>82.4</td><td>43.7</td><td>45.4</td><td>39.9</td><td>43.8</td><td>26.1</td><td>16.2</td><td>41.6</td><td>16.9</td><td>6.6</td><td>8.4</td></tr><tr><td>Qwen3-8B</td><td>83.9</td><td>38.1</td><td>41.5</td><td>31.8</td><td>36.7</td><td>37.1</td><td>37.8</td><td>47.8</td><td>7.9</td><td>15.6</td><td>7.4</td></tr><tr><td>Qwen3-14B</td><td>86.5</td><td>36.8</td><td>40.6</td><td>29.9</td><td>36.2</td><td>27.1</td><td>13.2</td><td>42.5</td><td>5.5</td><td>11.6</td><td>6.2</td></tr><tr><td>Qwen3-32B</td><td>83.7</td><td>36.1</td><td>40.2</td><td>29.4</td><td></td><td>38.8</td><td>38.6</td><td>37.0</td><td>6.1</td><td>11.0</td><td>6.3</td></tr><tr><td>Gemma3-12B</td><td>84.9</td><td>41.5</td><td>44.0</td><td>37.3</td><td>34.8</td><td>33.9</td><td>29.3</td><td>24.3</td><td>15.2</td><td>13.9</td><td>7.2</td></tr><tr><td>Gemma3-27B</td><td>89.2</td><td>42.3</td><td>44.5</td><td>38.4</td><td>36.0</td><td>16.0</td><td>7.0</td><td>33.4</td><td>15.0</td><td>7.1</td><td>8.4</td></tr><tr><td colspan="11">AMC23</td></tr><tr><td>Qwen2.5-7B</td><td>53.6</td><td>43.6</td><td>45.3</td><td>31.7</td><td>44.0</td><td>24.7</td><td>8.2</td><td>42.5</td><td>15.7</td><td>6.7</td><td>6.0</td></tr><tr><td>Qwen2.5-14B</td><td>61.2</td><td>42.7</td><td>44.7</td><td>38.5</td><td>43.1</td><td>25.3</td><td>19.7</td><td>35.1</td><td>9.7</td><td>9.5</td><td>6.0</td></tr><tr><td>Qwen2.5-32B</td><td>66.4</td><td>43.3</td><td>45.1</td><td>39.5</td><td>44.0</td><td>24.9</td><td>22.7</td><td>38.3</td><td>9.8</td><td>10.8</td><td>8.9</td></tr><tr><td>Qwen3-8B</td><td>68.6</td><td>36.5</td><td>40.5</td><td>29.4</td><td>36.8</td><td>49.4</td><td>38.5</td><td>48.7</td><td>6.8</td><td>9.3</td><td>11.4</td></tr><tr><td>Qwen3-14B</td><td>74.1</td><td>34.6</td><td>39.2</td><td>26.9</td><td>35.9</td><td>27.2</td><td>10.5</td><td>35.3</td><td>12.3</td><td>11.8</td><td>12.0</td></tr><tr><td>Qwen3-32B</td><td>67.5</td><td>33.7</td><td>38.5</td><td>25.4</td><td></td><td>44.0</td><td>37.7</td><td>32.3</td><td>15.2</td><td>7.5</td><td>11.6</td></tr><tr><td>Gemma3-12B</td><td>66.8</td><td>40.8</td><td>43.6</td><td>36.4</td><td>35.3</td><td>34.5</td><td>24.0</td><td>25.9</td><td>12.7</td><td>10.9</td><td>10.1</td></tr><tr><td>Gemma3-27B</td><td>76.9</td><td>41.8</td><td>44.3</td><td>37.9</td><td>36.9</td><td>19.9</td><td>12.3</td><td>26.4</td><td>12.4</td><td>12.0</td><td>13.1</td></tr><tr><td colspan="10">AIME24</td></tr><tr><td>Qwen2.5-7B</td><td>12.6 42.8</td><td></td><td>44.7</td><td>30.3</td><td>43.3</td><td>29.5</td><td>19.0</td><td>39.5</td><td>14.7</td><td>11.5</td><td>22.7</td></tr><tr><td>Qwen2.5-14B</td><td>13.8</td><td>41.8</td><td>44.0 44.4</td><td>37.5 37.9</td><td>42.3</td><td>25.1 23.9</td><td>15.6</td><td>31.4</td><td>13.1</td><td>16.9</td><td>23.0</td></tr><tr><td>Qwen2.5-32B</td><td>16.9</td><td>42.4</td><td>39.3</td><td></td><td>43.5</td><td></td><td>16.7</td><td>34.1</td><td>10.6</td><td>20.4</td><td>23.4</td></tr><tr><td>Qwen3-8B</td><td>27.7 27.1</td><td>34.9</td><td>38.1</td><td>26.5</td><td>36.4</td><td>40.3</td><td>42.9</td><td>45.8</td><td>14.2</td><td>14.9</td><td>29.2</td></tr><tr><td>Qwen3-14B</td><td>28.3</td><td>33.2</td><td>37.4</td><td>24.4</td><td>35.7</td><td>31.4</td><td>16.2</td><td>44.1</td><td>20.6</td><td>18.1</td><td>29.7</td></tr><tr><td>Qwen3-32B</td><td></td><td>32.0</td><td></td><td>23.7</td><td></td><td>44.2</td><td>37.6</td><td>25.7</td><td>21.6</td><td>16.4</td><td>28.8</td></tr><tr><td>Gemma3-12B</td><td>23.9</td><td>39.8</td><td>42.8</td><td>34.7</td><td>34.6</td><td>33.2</td><td>31.8</td><td>23.0</td><td>9.7</td><td>12.6</td><td>29.2</td></tr><tr><td>Gemma3-27B</td><td>29.0</td><td>40.7</td><td>43.5</td><td>36.2</td><td>36.1</td><td>29.6</td><td>29.4</td><td>21.1</td><td>8.1</td><td>24.3</td><td>30.0</td></tr><tr><td colspan="10">AIME25</td></tr><tr><td>Qwen2.5-7B</td><td>9.1</td><td>42.4</td><td>44.4</td><td>31.0</td><td>43.1</td><td>26.6</td><td>16.2</td><td>37.3</td><td>11.2</td><td>20.5</td><td>25.2</td></tr><tr><td>Qwen2.5-14B</td><td>14.7</td><td>42.4</td><td>44.4</td><td>37.8</td><td>42.9</td><td>27.5</td><td>15.7</td><td>31.6</td><td>10.8</td><td>25.4</td><td>24.4</td></tr><tr><td>Qwen2.5-32B</td><td>12.2</td><td>42.7</td><td>44.7</td><td>38.3</td><td>43.8</td><td>30.6</td><td>18.4</td><td>33.0</td><td>8.7</td><td>28.3</td><td>27.7</td></tr><tr><td>Qwen3-8B</td><td>22.5</td><td>34.2</td><td>38.8</td><td>25.9</td><td>36.0</td><td>37.6</td><td>43.4</td><td>46.9</td><td>12.3</td><td>9.5</td><td>29.6</td></tr><tr><td>Qwen3-14B</td><td>27.5</td><td>32.6</td><td>37.8</td><td>24.0</td><td>35.3</td><td>30.8</td><td>19.7</td><td>53.0</td><td>17.1</td><td>5.3</td><td>30.7</td></tr><tr><td>Qwen3-32B</td><td>23.7</td><td>31.7</td><td>37.2</td><td>22.5</td><td></td><td>36.3</td><td>37.8</td><td>25.6</td><td>19.1</td><td>6.9 7.8</td><td>30.8 27.2</td></tr><tr><td>Gemma3-12B</td><td>18.8 25.3</td><td>40.3 40.6</td><td>43.2 43.4</td><td>35.5 35.9</td><td>34.5</td><td>32.1</td></table>

## 5 EXPERIMENTS

## 5.1 WHITE-BOX SETTING

Datasets and Models. To evaluate DTC on tasks with long CoT path, we use four widely used mathematical benchmarks spanning a range of difficulty: MATH-500 (Hendrycks et al., 2021; Lightman et al., 2024) and the contest sets AMC23, AIME24, and AIME25 (Balunovic et al., 2025). We evaluate eight LLMs from three families: Qwen2.5-{7,14,32}B-Instruct (Qwen Team, 2024), Qwen3-{8,14,32}B (Yang et al., 2025), and Gemma3-{12,27}B-IT (Gemma Team, 2025).

![](images/0fe0d2606e6dffbb820ba30ec0e41beceea2bedf12893ce1050b22abea351ccd.jpg)  
Figure 3: Calibration plots and probability histogram for Qwen2.5-14B on MATH-500. The x-axi shows mean confidence within 20 probability bins. The calibration curve (blue line with $\mu \pm \sigma )$ displays actual accuracy per bin, while the gray shadow represents the probability proportion.

Baselines. Given the limited prior work on calibrated reasoning uncertainty, we compare DTC against three classes of uncertainty estimators. (1) Standard full-sequence confidence: $\bar { C } _ { \mathrm { N S L } }$ uses length-normalized sequence likelihood (Equation (3)); $C _ { \bf m e a n }$ uses mean token probability (Section 3.2); and Entropy Confidence that uses length-normalized predictive entropy confidence (Li et al., 2025). (2) Refinements offull-sequence confidence: BaseCal (Tan et al., 2026) averages response-token probabilities under a paired base model; $\mathbf { U Q A C _ { a t t n } }$ (Li et al., 2025) further select attention-related tokens; and our variant $\mathbf { U Q A C _ { m e a n } }$ which replaces each selected-token probability with the $C _ { \mathrm { m e a n } }$ before multiplying over the selected set. BaseCal is omitted for Qwen3-32B because no public base checkpoint is available. (3) Verbalized estimation: Xiong et al. (2024) propose Verbalized confidence through additional prompts after the response elicits the model confidence. We also include PRM (Skywork-o1-Open-PRM-Qwen-2.5-7B; He et al., 2024) as a supervised reference, averaging rewards over newline-delimited reasoning steps into path-level confidence.

Implementations. We use the officially recommended decoding parameters for each LLM, as detailed in Appendix A.2. For each model family, we use its smallest model and 100 problems drawn at random from the MATH training set (Hendrycks et al., 2021) to set that family’s disagreement threshold θ. We report $\mathrm { D T C _ { p r o d } }$ and $\mathrm { D T C } _ { \mathrm { l i n } } ,$ , with $\theta \ : = \ : 0 . 5 0$ for Qwen2.5, $\theta \ : = \ : 0 . 6 0$ for Qwen3, and $\theta = 0 . 8 5$ for Gemma3. For $\mathrm { D T C } _ { \mathrm { p r o d } }$ , token probabilities come from the generator G (Equation (9)). Unless noted, the auxiliary is the smaller same-family instruct model: Qwen2.5- 1.5B-Instruct, Qwen3-1.7B, or Gemma3-4B-IT. Other auxiliary sizes are examined in Section 6. Following Li et al. (2025), we evaluate ECE and AUROC by repeated subsampling, emphasizing ECE and the calibration plots. AUROC is the area under the ROC curve and measures ranking. Tables and figures use AUC for AUROC. Further details are in Appendix A.3.

Main Results. Table 1 and Table 5 (Appendix B.1) report the ECE and AUROC results, respectively. Figure 3 visualizes calibration curves on MATH-500 with Qwen2.5-14B. We find that:

Full-sequence confidence retains ranking information but remains overconfident. The three standard methods $( C _ { \mathrm { N S L } } , C _ { \mathrm { m e a n } }$ and Entropy Confidence) achieve an high average AUROC of 72.7– 73.8, yet their ECEs reach 32.7–42.4. Their calibration curves lie below the diagonal, with scores concentrated near the high-confidence.

Refinements improve calibration unevenly and weaken ranking. BaseCal offers limited calibration gains, whereas the UQAC variants reduce ECE more substantially. UQAC variant $\mathrm { U Q A C } _ { \mathrm { m e a n } }$ by ours achieves both the lowest average ECE (23.6) and highest AUROC (66.7) among these refinements, but its ranking still trails the standard methods.

Table 2: Uncertainty quantification performance with DeepSeek-V3.2 in black-box settings.
<table><tr><td rowspan="2">Method</td><td colspan="3">AIME24</td><td colspan="3">AIME25</td><td colspan="3">HMMT25</td><td colspan="3">HMMT26</td></tr><tr><td></td><td>Acc↑ ECE↓ AUC↑ Acc↑ ECE↓ AUC↑ Acc↑ ECE↓ AUC↑ Acc↑ ECE↓ AUC↑</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Default CoT</td><td>72.2</td><td></td><td></td><td>60.9</td><td></td><td></td><td>46.1</td><td></td><td></td><td>50.9</td><td></td><td></td></tr><tr><td>→ + PRM</td><td></td><td>39.0</td><td>88.2</td><td></td><td>39.5</td><td>80.4</td><td></td><td>41.1</td><td>60.4</td><td></td><td>38.1</td><td>75.1</td></tr><tr><td> $ + \mathrm { D T C _ { \mathrm { l i n } } }$ </td><td>一</td><td>11.2</td><td>74.6</td><td></td><td>9.4</td><td>78.7</td><td></td><td>12.7</td><td>67.9</td><td>一</td><td>12.5</td><td>72.7</td></tr><tr><td> $ + \mathrm { D T C _ { p r o d } }$ </td><td></td><td>24.6</td><td>79.4</td><td></td><td>23.9</td><td>82.1</td><td></td><td>29.1</td><td>75.0</td><td></td><td>26.4</td><td>76.5</td></tr><tr><td>Verb. Conf.</td><td>67.4</td><td>40.8</td><td>85.5</td><td>55.3</td><td>39.6</td><td>86.8</td><td>38.3</td><td>40.3</td><td>78.2</td><td>45.0</td><td>40.1</td><td>84.3</td></tr><tr><td> $ + \mathrm { D T C _ { \mathrm { l i n } } }$ </td><td></td><td>13.5</td><td>67.1</td><td></td><td>9.2</td><td>73.5</td><td></td><td>17.5</td><td>59.5</td><td></td><td>14.5</td><td>68.9</td></tr><tr><td> $ + \mathrm { D T C _ { p r o d } }$ </td><td>_</td><td>25.9</td><td>73.5</td><td></td><td>25.4</td><td>78.3</td><td></td><td>30.9</td><td>67.0</td><td></td><td>26.1</td><td>73.7</td></tr><tr><td> $\mathrm { V e r b . \ T o p K }$ </td><td>64.6</td><td>32.0</td><td>88.4</td><td>52.8</td><td>32.6</td><td>85.7</td><td>27.1</td><td>31.6</td><td>79.3</td><td>35.1</td><td>32.0</td><td>83.6</td></tr><tr><td> $ + \mathrm { D T C _ { \mathrm { l i n } } }$ </td><td></td><td>14.2</td><td>66.0</td><td></td><td>14.1</td><td>66.9</td><td></td><td>18.8</td><td>56.7</td><td></td><td>18.2</td><td>60.9</td></tr><tr><td> $ + \mathrm { D T C _ { p r o d } }$ </td><td></td><td>32.9</td><td>71.4</td><td></td><td>31.4</td><td>71.2</td><td></td><td>36.1</td><td>61.5</td><td></td><td>31.1</td><td>64.6</td></tr><tr><td>Verb. PD</td><td></td><td>68.3 32.6</td><td>86.2</td><td>2 57.1 33.2</td><td></td><td>84.9 33.1 31.5</td><td></td><td></td><td>74.3</td><td></td><td>37.9 32.3</td><td>83.1</td></tr><tr><td> $ + \mathrm { D T C _ { \mathrm { l i n } } }$ </td><td></td><td>15.1</td><td>65.4</td><td></td><td>13.5</td><td>69.1</td><td></td><td>19.2</td><td>57.8</td><td></td><td>14.6</td><td>66.0</td></tr><tr><td> $ + \mathrm { D T C _ { p r o d } }$ </td><td></td><td>35.3</td><td>72.6</td><td></td><td>32.6</td><td>74.1</td><td></td><td>36.2</td><td>63.0</td><td></td><td>31.8</td><td>69.3</td></tr></table>

![](images/dee4d664c7018e2d7e7ede921d544353f044537b4af2fdbb147e6fb51752608f.jpg)

![](images/21541db5ea69d466fbb32ed9d290f182f860eb455349b4e29f892623c9342a0e.jpg)

![](images/83a9df68fb98a7ae99e104d76f8dc31c13f930f0348a7389b60cb81f446902c1.jpg)  
Figure 4: Sensitivity of $\mathrm { D T C _ { \mathrm { l i n } } }$ to θ and auxiliary size, single-auxiliary Qwen2.5 on AMC23. We search θ in steps of 0.05 and report the θ (a) with the lowest ECE (b) and its AUROC (c).

DTC improves both calibration and average ranking. $\mathrm { D T C _ { \mathrm { l i n } } }$ and $\mathrm { D T C _ { p r o d } }$ reduce average ECE to 13.0 and 13.2, with calibration curves closer to the diagonal, while attaining AUROC 75.3 and 80.8. Thus, count alone supports effective calibration; retaining path-mean probability further improves average ranking at similar ECE. Relative to $\mathrm { U Q A C } _ { \mathrm { m e a n } } , \mathrm { D } \mathrm { \bar { T } C } _ { \mathrm { p r o d } }$ lowers average ECE by 10.4, suggesting that selecting divergent tokens captures uncertainty more effectively and improves the calibration of $C _ { \mathrm { m e a n } } .$ In harder datasets, PRM has higher ECE, while DTC is better calibrated.

## 5.2 BLACK-BOX SETTING

Datasets and Models. To better match a black-box evaluation setting, we use more challenging contest benchmarks than in the white-box experiments, namely AIME24, AIME25, HMMT25, and HMMT26 (Balunovic et al., 2025), and switch to stronger generators: Qwen3-4B-Instruct-2507, Qwen3-30B-A3B-Instruct-2507 (Yang et al., 2025), and DeepSeek-V3.2 (DeepSeek-AI, 2025). We treat each generator as a black box, using its output trajectories without accessing its logits for scoring. More details on datasets and models are shown in Appendix A.1.

Baselines. Because generator token probabilities are unavailable, we use Verbalized estimation methods based on CoT prompts as the main confidence baselines (Wang & Zhang, 2026). We consider three Verbalized estimation methods: Verbalized Confidence (Verb. Conf.), Verbalized TopK (Verb. TopK) (Tian et al., 2023), and Verbalized Probability Distribution (Verb. PD) (Wang et al., 2025). As described in Section 2, these Verbalized estimation methods change the CoT trajectory, so besides applying our method on Default CoT, we also apply it to the trajectories generated by these prompts for a fair comparison.

Evaluation and Implementation. Metrics, evaluation, and decoding follow the white-box setting; details appear in Appendix A.2 and A.3. All three generators use Qwen2.5-7B-Instruct as $\mathcal { A } ^ { \prime }$ and Qwen2.5-1.5B-Instruct as $A ^ { \prime \prime }$ (Qwen Team, 2024), with θ=0.70.

Main Results. Tables 2, 6 and 7 report accuracy, ECE, and AUROC across three generators and four mathematics benchmarks, respectively. Blue arrows (→) in the method column denote rescoring the same trajectories, leaving answers and accuracy unchanged. PRM additionally scores Default CoT trajectories as a reference. Verbalized estimation is overconfident and sometimes reduces answer accuracy. Consistent with the accuracy and calibration limitations observed in mathemati cal reasoning settings by Wang et al. (2025), we find that Verbalized estimation can reduce answer accuracy while producing overconfident scores. For DeepSeek-V3.2, the three Verbalized estimation methods reduce mean accuracy from 57.5 under Default CoT to 44.9–51.5, while their verbalized scores have mean ECE values of 32.1–40.2.

DTC achieves low ECE on Default CoT and improves calibration over verbalized scores on the same trajectories. In contrast, our method assesses the reliability of Default CoT trajectories without changing the generated answers or reducing model accuracy. On Default CoT, $\mathrm { D } \bar { \mathrm { T } } \mathrm { C } _ { \mathrm { l i n } }$ achieves mean ECE values of 11.5–12.7 across the three generators. When applied to trajectories generated under Verb. Conf., Verb. TopK, and Verb. PD, it also yields lower ECE than the original verbalized scores for every model–dataset pair. On DeepSeek-V3.2, its mean ECE is 13.7–16.3, compared with 32.1–40.2 for the original verbalized scores. $\mathrm { D T C _ { p r o d } }$ achieves higher mean AUROC than $\mathrm { D T C } _ { \mathrm { l i n } }$ but also has higher mean ECE. Despite these calibration gains, both estimators often have lower AUROC than the verbalized scores on trajectories from these three Verbalized estimation methods.

## 6 ANALYSES AND DISCUSSIONS

Disagreement Threshold and auxiliary size. We study how the choice of auxiliary model and disagreement threshold θ affects $\operatorname { D T C } _ { \operatorname { l i n } } .$ For each pair of generator and auxiliary, we search θ in steps of 0.05 and report the θ with the lowest ECE (Figure 4). The lowest ECE is similar across auxiliary sizes. At these values, a larger auxiliary usually has a higher AUROC. When the auxiliary is fixed at 1.5B, the θ with the lowest ECE stays near 0.5 (Figure 5). For efficiency and a simpler method, each model family uses one small aux-

![](images/a8d6f277f2028ab5f3db4e30ef97f6cfea11593019b3f631b4db65443cfd5390.jpg)

![](images/ac5950bed0ffa741104c2aec87dca633f54e2ff89bba8d3c07fa5da5089a2bd8.jpg)  
Figure 5: $\mathrm { D T C _ { \mathrm { l i n } } }$ sensitivity to $\theta$ with auxiliary fixed at 1.5B and generating models $\mathbf { \varepsilon } \in \mathbf { \varepsilon } \{ 7 , 1 4 , 3 2 \} \mathbf { B }$ (solid: AMC23; dashed: AIME24). The shaded band marks the white-box operating point θ=0.50.

iliary from the same family and one fixed θ. The corresponding dual-auxiliary analyses are given in Appendix C.1; the overall conclusions are similar to those in the single-auxiliary setting.

Combining verbalized scores with DTC. Verbalized estimation often yields overconfident scores (high ECE) despite competitive ranking (high AUROC). We examine whether combining the divergent-token count with verbalized scores can reduce the overconfidence of Verbalized estimation, as $\mathrm { D T C } _ { \mathrm { p r o d } }$ does for $C _ { \mathrm { m e a n } } .$ We

Table 3: Verbalized vs. w/ DTC on DeepSeek-V3.2.
<table><tr><td rowspan="2">Method</td><td colspan="2">AIME24</td><td colspan="2">AIME25</td><td colspan="2">HMMT25</td><td colspan="2">HMMT26</td></tr><tr><td>ECE↓</td><td>AUC↑</td><td>ECE↓</td><td>AUC↑</td><td>ECE↓</td><td>AUC↑</td><td>ECE↓</td><td>AUC↑</td></tr><tr><td>Verb. Conf.</td><td>40.8</td><td>85.5</td><td>39.6</td><td>86.8</td><td>40.3</td><td>78.2</td><td>40.1</td><td>84.3</td></tr><tr><td>w/ DTC</td><td>6.3</td><td>85.8</td><td>3.7</td><td>87.7</td><td>10.5</td><td>77.8</td><td>6.2</td><td>84.2</td></tr><tr><td>Verb. TopK 32.0</td><td></td><td>88.4</td><td>32.6</td><td>85.7</td><td>31.6</td><td>79.3</td><td>32.0</td><td>83.6</td></tr><tr><td>w/ DTC</td><td>17.0</td><td>90.3</td><td>14.6</td><td>87.3</td><td>18.8</td><td>79.1</td><td>15.7</td><td>83.6</td></tr><tr><td>Verb. PD</td><td>32.6</td><td>86.2</td><td>33.2</td><td>84.9</td><td>31.5</td><td>74.3</td><td>32.3</td><td>83.1</td></tr><tr><td>w/ DTC</td><td>10.2</td><td>87.7</td><td>10.9</td><td>86.9</td><td>19.2</td><td>76.0</td><td>13.9</td><td>84.0</td></tr></table>

follow the way $\mathrm { D T C _ { p r o d } }$ combines $C _ { \mathrm { m e a n } }$ with the uncertainty-informed count $m ,$ and combine verbalized scores with DTC to get $p _ { \mathrm { v e r b } } ^ { m + k }$ . The auxiliary and k follow the black-box $\mathrm { D T C _ { p r o d } }$ setting. We report the result as w/ DTC in Tables 3, 8 and 9. Across these tables, DTC lowers ECE in every case, often by more than 20 percentage points. AUROC changes little and sometimes improves. On DeepSeek-V3.2, ECE decreases by 12.3 to 35.9 percentage points, and AUROC changes by −0.4 to +2.0 percentage points. Combining verbalized confidence with DTC thus gives a better tradeoff between ECE and AUROC.

## 7 CONCLUSION

We study the reliability of a reasoning path. The UQAC pilot indicates that much of the calibration gain comes from how many tokens are selected. DTC counts tokens on which two models disagree strongly about the next token. $\mathrm { D T C _ { \mathrm { l i n } } }$ takes this count as the confidence. $\mathrm { D T C _ { p r o d } }$ recalibrates the trajectory’s mean token probability with the same count, and a larger count gives a lower score. On the math benchmarks, both improve calibration over sequence likelihood, verbalized confidence, and UQAC, whether the generator logits are available or not. Applied to a verbalized score, the count also reduces overconfidence.

## AI USE STATEMENT

We used generative AI tools to implement evaluation, plotting, and table-generation code, to assist with translation, and to draft and edit manuscript text, figures, and related-work notes. We have not used generative AI tools to generate synthetic datasets, to formulate or prove mathematical claims, or to interpret experimental results, and the remaining required-disclosure tasks are not applicable. Additionally, we used generative AI tools to search and format references and to improve readability. We reviewed all AI-assisted text, citations, and code; numbers reported in the paper come from our evaluation pipeline and were checked by the authors. We take responsibility for the final content of this work, including text, claims, and artifacts produced with the aid of generative AI.

## REPRODUCIBILITY STATEMENT

Datasets, models, sampling hyperparameters, operating thresholds, and baseline protocols are specified in Section 5 and Appendix A. Estimator definitions appear in Section 4; additional ablations and black-box settings are in Appendix C and Appendix C.5.

## REFERENCES

Yavuz Faruk Bakman, Duygu Nur Yaldiz, Baturalp Buyukates, Chenyang Tao, Dimitrios Dimitriadis, and Salman Avestimehr. MARS: Meaning-aware response scoring for uncertainty estimation in generative LLMs. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar (eds.), Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 7752–7767, Bangkok, Thailand, August 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.acl-long.419. URL https://aclanthology.org/2024. acl-long.419/.

Mislav Balunovic, Jasper Dekoninck, Ivo Petrov, Nikola Jovanovic, and Martin Vechev.´ Matharena: Evaluating llms on uncontaminated math competitions. In D. Belgrave, C. Zhang, H. Lin, R. Pascanu, P. Koniusz, M. Ghassemi, and N. Chen (eds.), Advances in Neural Information Processing Systems, volume 38, Datasets and Benchmarks Track, pp. 22851–22888. Curran Associates, Inc., 2025. doi: 10.52202/085713-0679. URL https://proceedings.neurips.cc/paper\_files/paper/2025/file/ 1d27c01ebd3e3aebe226b44fc970d803-Paper-Datasets\_and\_Benchmarks\_ Track.pdf.

Guangsheng Bao, Hongbo Zhang, Cunxiang Wang, Linyi Yang, and Yue Zhang. How likely do LLMs with CoT mimic human reasoning? In Owen Rambow, Leo Wanner, Marianna Apidianaki, Hend Al-Khalifa, Barbara Di Eugenio, and Steven Schockaert (eds.), Proceedings of the 31st International Conference on Computational Linguistics, pp. 7831–7850, Abu Dhabi, UAE, January 2025. Association for Computational Linguistics. URL https://aclanthology. org/2025.coling-main.524/.

Jan Clusmann, Fiona R. Kolbinger, Hannah Sophie Muti, Zunamys I. Carrero, Jan-Niklas Eckardt, Narmin Ghaffari Laleh, Chiara Maria Lavinia Löffler, Sophie-Caroline Schwarzkopf, Michaela Unger, Gregory P. Veldhuizen, Sophia J. Wagner, and Jakob Nikolas Kather. The future landscape

of large language models in medicine. Communications Medicine, 3(1):141, 2023. doi: 10.1038/ s43856-023-00370-1. URL https://doi.org/10.1038/s43856-023-00370-1.

DeepSeek-AI. DeepSeek-V3.2: Pushing the frontier of open large language models. arXiv preprint arXiv:2512.02556, 2025. URL https://arxiv.org/abs/2512.02556.

Siddartha Devic, Tejas Srinivasan, Jesse Thomason, Willie Neiswanger, and Vatsal Sharan. From calibration to collaboration: LLM uncertainty quantification should be more human-centered. arXiv preprint arXiv:2506.07461, 2025. URL https://arxiv.org/abs/2506.07461.

Jinhao Duan, Hao Cheng, Shiqi Wang, Alex Zavalny, Chenan Wang, Renjing Xu, Bhavya Kailkhura, and Kaidi Xu. Shifting attention to relevance: Towards the predictive uncertainty quantification of free-form large language models. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar (eds.), Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 5050–5063, Bangkok, Thailand, August 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.acl-long.276. URL https: //aclanthology.org/2024.acl-long.276/.

Yichao Fu, Xuewei Wang, Hao Zhang, Yuandong Tian, and Jiawei Zhao. Deep think with confidence. In C. Vondrick, B. Hariharan, C. Raffel, L. Pinto, D. Yang, and A. Faust (eds.), International Conference on Learning Representations, volume 2026, pp. 94355–94377, 2026. URL https://proceedings.iclr.cc/paper\_files/paper/2026/file/ 98257285340854262185500e59bc0f28-Paper-Conference.pdf.

Aryo Pradipta Gema, Alexander Hägele, Runjin Chen, Andy Arditi, Jacob Goldman-Wetzler, Kit Fraser-Taliente, Henry Sleight, Linda Petrini, Julian Michael, Beatrice Alex, Pasquale Minervini, Yanda Chen, Joe Benton, and Ethan Perez. Inverse Scaling in Test-Time Compute. Transactions on Machine Learning Research, 2025. URL https://mlanthology.org/tmlr/2025/ gema2025tmlr-inverse/.

Gemma Team. Gemma 3 technical report. arXiv preprint arXiv:2503.19786, 2025. URL https: //arxiv.org/abs/2503.19786.

Soumya Suvra Ghosal, Souradip Chakraborty, Avinash Reddy, Yifu Lu, Mengdi Wang, Dinesh Manocha, Furong Huang, Mohammad Ghavamzadeh, and Amrit Singh Bedi. Does thinking more always help? mirage of test-time scaling in reasoning models. In D. Belgrave, C. Zhang, H. Lin, R. Pascanu, P. Koniusz, M. Ghassemi, and N. Chen (eds.), Advances in Neural Information Processing Systems, volume 38, Main Conference, pp. 191359–191386. Curran Associates, Inc., 2025. doi: 10.52202 085713-5747. URL https://proceedings.neurips.cc/paper\_files/paper/ 2025/file/fc067ac218430c409d6f65403328f740-Paper-Conference.pdf.

Matt Gorbett and Suman Jana. Cross-model disagreement as a label-free correctness signal. arXiv preprint arXiv:2603.25450, 2026. URL https://arxiv.org/abs/2603.25450.

Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q. Weinberger. On calibration of modern neural networks. In Proceedings of the 34th International Conference on Machine Learning - Volume 70, ICML’17, pp. 1321–1330. JMLR.org, 2017.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, Xiaokang Zhang, Xingkai Yu, Yu Wu, Z. F. Wu, Zhibin Gou, Zhihong Shao, Zhuoshu Li, Ziyi Gao, Aixin Liu, Bing Xue, Bingxuan Wang, Bochao Wu, Bei Feng, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chong Ruan, Damai Dai, Deli Chen, Dongjie Ji, Erhang Li, Fangyun Lin, Fucong Dai, Fuli Luo, Guangbo Hao, Guanting Chen, Guowei Li, H. Zhang, Hanwei Xu, Honghui Ding, Huazuo Gao, Hui Qu, Hui Li, Jianzhong Guo, Jiashi Li, Jingchang Chen, Jingyang Yuan, Jinhao Tu, Junjie Qiu, Junlong Li, J. L. Cai, Jiaqi Ni, Jian Liang, Jin Chen, Kai Dong, Kai Hu, Kaichao You, Kaige Gao, Kang Guan, Kexin Huang, Kuai Yu, Lean Wang, Lecong Zhang, Liang Zhao, Litong Wang, Liyue Zhang, Lei Xu, Leyi Xia, Mingchuan Zhang, Minghua Zhang, Minghui Tang, Mingxu Zhou, Meng Li, Miaojun Wang, Mingming Li, Ning Tian, Panpan Huang, Peng Zhang, Qiancheng Wang, Qinyu Chen, Qiushi Du, Ruiqi Ge, Ruisong Zhang, Ruizhe Pan, Runji Wang, R. J. Chen, R. L. Jin, Ruyi Chen, Shanghao Lu, Shangyan Zhou, Shanhuang Chen, Shengfeng Ye, Shiyu Wang, Shuiping Yu, Shunfeng Zhou, Shuting Pan, S. S. Li, Shuang Zhou, Shaoqing Wu, Tao Yun, Tian Pei, Tianyu Sun, T. Wang, Wangding Zeng, Wen Liu, Wenfeng Liang, Wenjun Gao, Wenqin Yu, Wentao Zhang, W. L. Xiao, Wei An, Xiaodong Liu, Xiaohan Wang, Xiaokang Chen, Xiaotao Nie, Xin Cheng, Xin Liu, Xin Xie, Xingchao Liu, Xinyu Yang, Xinyuan Li, Xuecheng Su, Xuheng Lin, X. Q. Li, Xiangyue Jin, Xiaojin Shen, Xiaosha Chen, Xiaowen Sun, Xiaoxiang Wang, Xinnan Song, Xinyi Zhou, Xianzu Wang, Xinxia Shan, Y. K. Li, Y. Q. Wang, Y. X. Wei, Yang Zhang, Yanhong Xu, Yao

Li, Yao Zhao, Yaofeng Sun, Yaohui Wang, Yi Yu, Yichao Zhang, Yifan Shi, Yiliang Xiong, Ying He, Yishi Piao, Yisong Wang, Yixuan Tan, Yiyang Ma, Yiyuan Liu, Yongqiang Guo, Yuan Ou, Yuduan Wang, Yue Gong, Yuheng Zou, Yujia He, Yunfan Xiong, Yuxiang Luo, Yuxiang You, Yuxuan Liu, Yuyang Zhou, Y. X. Zhu, Yanping Huang, Yaohui Li, Yi Zheng, Yuchen Zhu, Yunxian Ma, Ying Tang, Yukun Zha, Yuting Yan, Z. Z. Ren, Zehui Ren, Zhangli Sha, Zhe Fu, Zhean Xu, Zhenda Xie, Zhengyan Zhang, Zhewen Hao, Zhicheng Ma, Zhigang Yan, Zhiyu Wu, Zihui Gu, Zijia Zhu, Zijun Liu, Zilin Li, Ziwei Xie, Ziyang Song, Zizheng Pan, Zhen Huang, Zhipeng Xu, Zhongyu Zhang, and Zhen Zhang. DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning. Nature, 645(8081):633–638, 2025. doi: 10.1038/s41586-025-09422-z. URL https://doi.org/10.1038/s41586-025-09422-z.

Jujie He, Tianwen Wei, Rui Yan, Jiacai Liu, Chaojie Wang, Yimeng Gan, Shiwen Tu, Chris Yuhao Liu, Liang Zeng, Xiaokun Wang, Boyang Wang, Yongcong Li, Fuxiang Zhang, Jiacheng Xu, Bo An, Yang Liu, and Yahui Zhou. Skywork-o1 open series, November 2024. URL https: //doi.org/10.5281/zenodo.16998085.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the MATH dataset. In J. Vanschoren and S. Yeung (eds.), Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks, volume 1, 2021. URL https: //datasets-benchmarks-proceedings.neurips.cc/paper\_files/paper/ 2021/file/be83ab3ecd0db773eb2dc1b0a17836a1-Paper-round2.pdf.

Maya Kruse, Majid Afshar, Saksham Khatwani, Anoop Mayampurath, Guanhua Chen, and Yanjun Gao. Simple yet effective: An information-theoretic approach to multi-LLM uncertainty quantification. In Christos Christodoulopoulos, Tanmoy Chakraborty, Carolyn Rose, and Violet Peng (eds.), Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pp. 30493–30504, Suzhou, China, November 2025. Association for Computational Linguistics. ISBN 979-8-89176-332-6. doi: 10.18653/v1/2025.emnlp-main.1551. URL https://aclanthology.org/2025.emnlp-main.1551/.

Lorenz Kuhn, Yarin Gal, and Sebastian Farquhar. Semantic uncertainty: Linguistic invariances for uncertainty estimation in natural language generation. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id= VD-AYtP0dve.

Balaji Lakshminarayanan, Alexander Pritzel, and Charles Blundell. Simple and scalable predictive uncertainty estimation using deep ensembles. In I. Guyon, U. Von Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett (eds.), Advances in Neural Information Processing Systems, volume 30. Curran Associates, Inc., 2017. URL https://proceedings.neurips.cc/paper\_files/paper/2017/ file/9ef2ed4b7fd2c810847ffa5fa85bce38-Paper.pdf.

Yinghao Li, Rushi Qiang, Lama Moukheiber, and Chao Zhang. Language model uncertainty quantification with attention chain. In Second Conference on Language Modeling, 2025. URL https://openreview.net/forum?id=QTrW2HWNXe.

Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let's verify step by step. In B. Kim, Y. Yue, S. Chaudhuri, K. Fragkiadaki, M. Khan, and Y. Sun (eds.), International Conference on Learning Representations, volume 2024, pp. 39578–39601, 2024. URL https://proceedings.iclr.cc/paper\_files/paper/2024/file/ aca97732e30bcf1303bc22ac3924fd16-Paper-Conference.pdf.

Zhen Lin, Shubhendu Trivedi, and Jimeng Sun. Contextualized sequence likelihood: Enhanced confidence scores for natural language generation. In Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung Chen (eds.), Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pp. 10351–10368, Miami, Florida, USA, November 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.emnlp-main.578. URL https: //aclanthology.org/2024.emnlp-main.578/.

Xiaoou Liu, Tiejin Chen, Longchao Da, Chacha Chen, Zhen Lin, and Hua Wei. Uncertainty quantification and confidence calibration in large language models: A survey. In Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2, KDD ’25, pp. 6107–6117, New York, NY, USA, 2025. Association for Computing Machinery. ISBN 9798400714542. doi: 10.1145/3711896.3736569. URL https://doi.org/10.1145/ 3711896.3736569.

Nishanth Madhusudhan, Sathwik Tejaswi Madhusudhan, Vikas Yadav, and Masoud Hashemi. Do LLMs know when to NOT answer? investigating abstention abilities of large language models. In Owen Rambow, Leo Wanner, Marianna Apidianaki, Hend Al-Khalifa, Barbara Di Eugenio, and Steven Schockaert (eds.), Proceedings of the 31st International Conference on Computational Linguistics, pp. 9329–9345, Abu Dhabi, UAE, January 2025. Association for Computational Linguistics. URL https://aclanthology.org/2025.coling-main.627/.

Andrey Malinin and Mark Gales. Uncertainty estimation in autoregressive structured prediction. In International Conference on Learning Representations, 2021. URL https://openreview. net/forum?id=jN5y-zb5Q7m.

OpenAI. OpenAI o1 system card. arXiv preprint arXiv:2412.16720, 2024. URL https:// arxiv.org/abs/2412.16720.

Hadas Orgad, Michael Toker, Zorik Gekhman, Roi Reichart, Idan Szpektor, Hadas Kotek, and Yonatan Belinkov. Llms know more than they show: On the intrinsic representation of llm hallucinations. In Y. Yue, A. Garg, N. Peng, F. Sha, and R. Yu (eds.), International Conference on Learning Representations, volume 2025, pp. 66880–66913, 2025. URL https://proceedings.iclr.cc/paper\_files/paper/2025/file/ a712d461e57201efe35d429a6f1731c1-Paper-Conference.pdf.

Qwen Team. Qwen2.5 technical report. arXiv preprint arXiv:2412.15115, 2024. URL https: //arxiv.org/abs/2412.15115.

Guangzhi Sun, Potsawee Manakul, Adian Liusie, Kunat Pipatanakul, Chao Zhang, Philip C. Wood land, and Mark J. F. Gales. Crosscheckgpt: Universal hallucination ranking for multimodal foundation models. CoRR, abs/2405.13684, 2024. URL https://doi.org/10.48550/ arXiv.2405.13684.

Hexiang Tan, Wanli Yang, Junwei Zhang, Xin Chen, Rui Tang, Du Su, Jingang Wang, Yuanzhuo Wang, Fei Sun, and Xueqi Cheng. BaseCal: Unsupervised confidence calibration via base model signals. In Maria Liakata, Viviane P. Moreira, Jiajun Zhang, and David Jurgens (eds.), Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 5172–5187, San Diego, California, United States, July 2026. Association for Computational Linguistics. ISBN 979-8-89176-390-6. doi: 10.18653/v1/2026.acl-long.234. URL https://aclanthology.org/2026.acl-long.234/.

Katherine Tian, Eric Mitchell, Allan Zhou, Archit Sharma, Rafael Rafailov, Huaxiu Yao, Chelsea Finn, and Christopher Manning. Just ask for calibration: Strategies for eliciting calibrated confidence scores from language models fine-tuned with human feedback. In Houda Bouamor, Juan Pino, and Kalika Bali (eds.), Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pp. 5433–5442, Singapore, December 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.emnlp-main.330. URL https: //aclanthology.org/2023.emnlp-main.330/.

Ante Wang, Weizhi Ma, and Yang Liu. Let the model distribute its doubt: Confidence estimation through verbalized probability distribution. arXiv preprint arXiv:2511.14275, 2025. URL https://arxiv.org/abs/2511.14275.

Jiayi Wang and Xu-Yao Zhang. A systematic evaluation of black-box uncertainty estimation methods for large language models. arXiv preprint arXiv:2606.19868, 2026. URL https: //arxiv.org/abs/2606.19868.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, brian ichter, Fei Xia, Ed Chi, Quoc V Le, and Denny Zhou. Chain-of-thought prompting elicits reasoning in large language models. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh (eds.), Advances in Neural Information Processing Systems, volume 35, pp. 24824–24837. Curran Associates, Inc., 2022. doi: 10.52202/ 068431-1800. URL https://proceedings.neurips.cc/paper\_files/paper/ 2022/file/9d5609613524ecf4f15af0f7b31abca4-Paper-Conference.pdf.

Miao Xiong, Zhiyuan Hu, Xinyang Lu, YIFEI LI, Jie Fu, Junxian He, and Bryan Hooi. Can llms express their uncertainty? an empirical evaluation of confidence elicitation in llms. In B. Kim, Y. Yue, S. Chaudhuri, K. Fragkiadaki, M. Khan, and Y. Sun (eds.), International Conference on Learning Representations, volume 2024, pp. 23650–23678, 2024. URL https://proceedings.iclr.cc/paper\_files/paper/2024/file/ 6733cf15e10e2cd1d59af033c3bb8507-Paper-Conference.pdf.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025. URL https://arxiv.org/abs/2505.09388.

Boxuan Zhang and Ruqi Zhang. CoT-UQ: Improving response-wise uncertainty quantification in LLMs with chain-of-thought. In Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar (eds.), Findings of the Association for Computational Linguistics: ACL 2025, pp. 26114–26133, Vienna, Austria, July 2025. Association for Computational Linguistics. ISBN 979-8-89176-256-5. doi: 10.18653/v1/2025.findings-acl.1339. URL https: //aclanthology.org/2025.findings-acl.1339/.

Tunyu Zhang, Haizhou Shi, Yibin Wang, Hengyi Wang, Xiaoxiao He, Zhuowei Li, Haoxian Chen, Ligong Han, Kai Xu, Huan Zhang, Dimitris Metaxas, and Hao Wang. Tokur: Token-level uncertainty estimation for large language model reasoning. In C. Vondrick, B. Hariharan, C. Raffel, L. Pinto, D. Yang, and A. Faust (eds.), International Conference on Learning Representations, volume 2026, pp. 52910–52942, 2026. URL https://proceedings.iclr.cc/paper\_files/paper/2026/file/ 56b06e61ddd2600fe86cca96f53869f6-Paper-Conference.pdf.

(Andrew) Zhanke Zhou, Zhaocheng Zhu, Xuan Li, Mikhail Galkin, Xiao Feng, Sanmi Koyejo, Jian Tang, and Bo Han. Landscape of thoughts: Visualizing the reasoning process of large language models. In C. Vondrick, B. Hariharan, C. Raffel, L. Pinto, D. Yang, and A. Faust (eds.), International Conference on Learning Representations, volume 2026, pp. 74411–74466, 2026. URL https://proceedings.iclr.cc/paper\_files/paper/2026/file/ 79201c5a56d6f39520fb8a06b1ec415c-Paper-Conference.pdf.

## A EXPERIMENTAL IMPLEMENTATION DETAILS

This appendix records the experimental setup for Sections 5.1 and 5.2: benchmarks and models, sampling, answer evaluation, and baselines.

## A.1 DATASETS AND MODELS

We use four mathematical benchmarks in the white-box setting and four in the black-box setting. White-box experiments use MATH-500, AMC23, AIME24, and AIME25; black-box experiments use AIME24, AIME25, HMMT25, and HMMT26. Table 4 lists the generators, auxiliary models, and the number of samples per question.

Table 4: Benchmarks, models, and sample counts per question. The three black-box generators share one auxiliary pair.
<table><tr><td colspan="2">White-box</td><td>Black-box</td></tr><tr><td>Benchmarks</td><td>MATH-500 (Lightman et al., 2024) AMC23 (Balunovic et al., 2025) AIME24 (Balunovic et al., 2025) AIME25 (Balunovic et al., 2025)</td><td>AIME24 (Balunovic et al., 2025) AIME25 (Balunovic et al., 2025) HMMT25 (Balunovic et al., 2025) HMMT26 (Balunovic et al., 2025)</td></tr><tr><td>Generators</td><td>Qwen2.5-14B-Instruct (Qwen Team, 2024) Qwen2.5-32B-Instruct (Qwen Team, 2024) Qwen3-8B (Yang et al., 2025) Qwen3-14B (Yang et al., 2025) Qwen3-32B (Yang et al., 2025) Gemma3-12B-IT (Gemma Team, 2025) Gemma3-27B-IT (Gemma Team, 2025)</td><td>DeepSeek-V3.2 (DeepSeek-AI, 2025) Qwen3-30B-A3B-Instruct-2507 (Yang et al., 2025) Qwen3-4B-Instruct-2507 (Yang et al., 2025)</td></tr><tr><td>Auxiliaries</td><td>Qwen2.5-1.5B-Instruct (Qwen Team, 2024) Qwen3-1.7B (Yang et al., 2025) Gemma3-4B-IT (Gemma Team, 2025) 8 on MATH-500</td><td>Qwen2.5-7B-Instruct (Qwen Team, 2024) Qwen2.5-1.5B-Instruct (Qwen Team, 2024)</td></tr><tr><td>Sample number per question</td><td>64 on AMC23 64 on AIME24 64 on AIME25</td><td>32 for DeepSeek-V3.2 64 for Qwen3-30B-A3B-Instruct-2507 64 for Qwen3-4B-Instruct-2507</td></tr></table>

White-box generators are Qwen2.5-{7B,14B,32B}-Instruct, Qwen3-{8B,14B,32B}, and Gemma3- {12B,27B}-IT, paired with the smaller same-family auxiliaries Qwen2.5-1.5B-Instruct, Qwen3- 1.7B, and Gemma3-4B-IT. Black-box generators are DeepSeek-V3.2, Qwen3-30B-A3B-Instruct-2507, and Qwen3-4B-Instruct-2507. In the main black-box setting, all three are scored with the external pair Qwen2.5-7B-Instruct and Qwen2.5-1.5B-Instruct.

## A.2 SAMPLING AND PROMPTING

We use a zero-shot system prompt and pass the problem as the user message:

Reasoning Prompt

Please reason step by step, and put your final answer within \boxed{}.

For the black-box protocols in Section 5.2, we add each protocol’s confidence instruction at generation time. We then score these trajectories and do not generate new ones.

We decode each generator with its own recommended setting. We set the maximum sequence length to 8192 tokens for white-box generators and to 16384 tokens for black-box generators. Under vLLM, we use temperature 0.6 and top-p = 0.95 for Qwen2.5-{7B,14B,32B}-Instruct, Qwen3- {8B,14B,32B}, and Gemma3-{12B,27B}-IT. For Qwen3-4B-Instruct-2507 and Qwen3-30B-A3B-Instruct-2507, we use temperature 0.7, top-p = 0.8, top-k = 20, and presence penalty 1.0. We sample DeepSeek-V3.2 from its API at temperature 1.0 and top-p = 0.95. We draw 8 trajectories per MATH-500 problem and 64 per contest problem for vLLM generators, and 32 per problem for DeepSeek-V3.2.

## A.3 ANSWER EXTRACTION AND EVALUATION

Following Li et al. (2025), we separate answer generation from uncertainty quantification. Once a response is generated, we extract the final answer and check whether it is correct. If no answer is extracted, we exclude that instance from the UQ evaluation. We then compute confidence on the same trajectory with teacher forcing. This step does not generate a new answer. Accuracy counts every sampled response, including truncated ones.

Following Li et al. (2025), we then subsample to balance the number of correct and incorrect predictions. If there are fewer correct predictions than incorrect ones, we randomly sample a matching number of incorrect predictions. If both groups exceed 500 instances, we randomly select 500 from each group. This step does not change the confidence scores, but it does change AUROC and ECE. We repeat the subsampling five times and report the mean and standard deviation. Accuracy is left unchanged.

Following Li et al. (2025), we take ECE (use 20 equal-width bins in our experiments) as the primary metric and AUROC as secondary, because AUROC only shows whether correct answers rank above incorrect ones. It does not show whether a confidence value can be read as a probability of being correct. If the confidences are 0.9, 0.5, and 0.1, with the first two answers correct and the third incorrect, AUROC is 1. Replacing these scores with $9 \times 1 0 ^ { - 3 } , 8 \times 1 0 ^ { - 1 0 }$ , and $7 . 9 9 \times 1 0 ^ { - 1 0 }$ keeps the ranking, so AUROC remains 1, even though the scores are not interpretable as probabilities. A high AUROC does not mean that confidence matches the probability of correctness. AUROC can still be 1 when the highest confidence among 10,000 answers is only $1 \times 1 0 ^ { - 3 }$ , so the scores can still misrepresent how certain the model is.

## A.4 BASELINES

## A.4.1 WHITE-BOX BASELINES

Length-normalized sequence likelihood. The product of token probabilities shrinks as the trajectory grows, so we take its geometric mean (Equation (3); Malinin & Gales, 2021). With $p _ { t } = P _ { \mathcal { G } } ( \tau _ { t } \mid x , \tau _ { < t } )$ and length $\breve { T } .$

$$
C _ { \mathrm { N S L } } = \Big ( \prod _ { t = 1 } ^ { T } p _ { t } \Big ) ^ { 1 / T } .\tag{10}
$$

Mean token probability. The arithmetic mean also removes the length effect, and a single lowprobability token moves it less than a product (Orgad et al., 2025):

$$
C _ { \mathrm { m e a n } } ~ = ~ { \frac { 1 } { T } } \sum _ { t = 1 } ^ { T } p _ { t } .\tag{11}
$$

Entropy confidence. Predictive entropy aggregates the next-token entropy over the trajectory and has no fixed range (Kuhn et al., 2023). Length-normalized predictive entropy divides that sum by the trajectory length, so outputs of different lengths remain comparable (Malinin & Gales, 2021). We report the confidence as one minus this normalized entropy:

$$
H ( \tau ) = - \sum _ { t = 1 } ^ { T } \sum _ { v \in \mathbb { V } } P _ { \mathcal { G } } ( v \mid x , \tau _ { < t } ) \log P _ { \mathcal { G } } ( v \mid x , \tau _ { < t } ) , \qquad 1 - \bar { H } ( \tau ) = 1 - \frac { H ( \tau ) } { T } .\tag{12}
$$

BaseCal. BaseCal scores a post-trained generator with its paired base model, which is often better calibrated (Tan et al., 2026). BaseCal-ReEval passes the same trajectory through the base model and averages the probabilities of the generated tokens:

$$
C _ { \mathrm { R e E v a l } } ( \tau , x ) = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } P _ { \mathrm { b a s e } } ( \tau _ { t } \mid x , \tau _ { < t } ) .\tag{13}
$$

BaseCal-Proj instead trains a one-layer linear map from the generator’s final-layer hidden states into the base model’s hidden states, then reads token probabilities from the base output layer. On our out-of-domain data this variant calibrates poorly, so we report only BaseCal-ReEval as BaseCal.

UQAC. Many tokens in a long trajectory say little about the answer. UQAC backtracks from the answer along attention and multiplies only the selected token probabilities (Li et al., 2025). With selected set S,

$$
{ \mathrm { U Q A C } } _ { \mathrm { a t t n } } = \prod _ { t \in S } p _ { t } .\tag{14}
$$

We replace every selected probability with the trajectory mean $C _ { \mathrm { m e a n } } ,$ so the score moves mainly with the number of selected tokens:

$$
\mathrm { U Q A C } _ { \mathrm { m e a n } } \ = \ C _ { \mathrm { m e a n } } ^ { | { \cal S } | } .\tag{15}
$$

Verbalized estimation. We obtain the score by an additional prompt after the response (Xiong et al., 2024). The prompt asks the model to rate its previous answer from 0 to 10 and place the integer in \boxed{}. We divide that integer by 10, which gives a confidence in [0, 1].

Process-reward reference. We also report Skywork-o1-Open-PRM-Qwen-2.5-7B as a supervised reference (He et al., 2024). We split the response at newlines. For each step we apply a sigmoid to the flagged token value and then average the step rewards. PRM is left out of the best and secondbest marks in the white-box table.

Verbalized confidence   
Question: [Question]   
Reason step-by-step to formulate your final answer. Your answer must be a mathematical value or short   
exact answer (e.g., an integer, fraction, radical, coordinate, or simplified expression), not a full sentence.   
If the question specifies a final report form (e.g., m+n or m−n), use that form. Then, reason about the   
confidence in your answer. Conclude by providing a JSON object that states the final answer and your   
estimated confidence in it: {   
"final\_answer": "Your final answer",   
"confidence": "0-1"   
}  
Figure 6: Prompt for Verbalized confidence. [Question] is replaced by the problem.

Verbalized TopK   
Question: [Question]   
Reason step-by-step to formulate 2 best guesses and probability that each is correct. Each answer must   
be a mathematical value or short exact answer (e.g., an integer, fraction, radical, coordinate, or simplified   
expression), not a full sentence. If the question specifies a final report form (e.g., m+n or m−n), use that   
form. Your final output must be a JSON array: [   
{   
"candidate": "first most likely answer",   
"confidence": "0-1"   
},   
{   
"candidate": "second most likely answer",   
"confidence": "0-1"   
<sup>}</sup><sub>]</sub>  
Figure 7: Prompt for Verbalized TopK with k=2. [Question] is replaced by the problem.

```jsonl
Verbalized probability distribution
Question: [Question]
Reason step-by-step to formulate your answer. You may propose multiple possible answers (fewer than
five). Each answer must be a mathematical value or short exact answer (e.g., an integer, fraction, radical,
coordinate, or simplified expression), not a full sentence. If the question specifies a final report form (e.g.,
m+n or m−n), use that form. Always include “None of the above” as a possible answer. Reason about the
confidence in each possible answer. Your final output must be a JSON array where the confidence scores
form a probability distribution (they must sum to 1.0): [
{
"candidate": "Candidate 1",
"confidence": "0-1"
},
{
"candidate": "Candidate 2",
"confidence": "0-1"
},
{
"candidate": "None of the above",
"confidence": "0-1"
}
]
```  
Figure 8: Prompt for Verbalized probability distribution. [Question] is replaced by the problem.

## A.4.2 BLACK-BOX BASELINES

Verbalized confidence. We ask the model to write the answer and a confidence score in the same generation (Tian et al., 2023). The prompt is shown in Figure 6. A single stated number is easy to read, but the model can focus on the answer it already prefers.

Verbalized TopK. We ask the model to list k=2 candidate answers, each with a probability, and we take the probability of the highest-scoring candidate (Tian et al., 2023). The prompt is shown in Figure 7.

Verbalized probability distribution. We ask the model to assign probabilities to candidate answers that sum to 1. On open-ended questions the list includes “None of the above”, which holds the remaining low-probability answers. The prompt is shown in Figure 8. The reported confidence is the probability of the selected answer, so that mass has to be shared with the other candidates.

## B COMPLETE RESULTS

The tables below add the results omitted from the main text. Accuracy uses every sampled response. ECE and AUROC use the class-balanced sets in Appendix A.3. An arrow marks a row that only rescores the same trajectories, so that row has no accuracy of its own. A dash means the metric is missing or does not apply.

## B.1 FULL WHITE-BOX RESULTS

The main text reports white-box ECE (Table 1). Table 5 is the AUROC table for the same singleauxiliary setup, with the same generators, benchmarks, and baselines. PRM is a supervised reference and is not part of the unsupervised best and second-best comparison.

## B.2 FULL BLACK-BOX RESULTS

Tables 6 and 7 are the Qwen3-4B and Qwen3-30B-A3B results. Each table gives accuracy for the original generation protocol, plus ECE and AUROC on AIME24, AIME25, HMMT25, and HMMT26, with averages. DTC and PRM rescore the frozen Default CoT trajectories. Within a confidence protocol, that protocol and its DTC rows use the same trajectories.

## B.3 VERBALIZED CONFIDENCE WITH DTC

Tables 8 and 9 repeat the verbalized-confidence experiment on Qwen3-30B-A3B and Qwen3-4B. For each protocol, the DTC adjustment uses the black-box $\mathrm { D T C _ { p r o d } }$ setup and the same frozen trajectory as the unadjusted verbalized score, so accuracy does not change.

Table 5: Uncertainty quantification performance (AUROC, ↑ higher is better) in white-box settings. In each row, the best result is in bold and the second-best one is underlined (excluding PRM). †: UQAC variant by ours.
<table><tr><td rowspan="2">Reasoning Models</td><td rowspan="2">Acc</td><td rowspan="2">CNSL</td><td rowspan="2">Cmean</td><td rowspan="2">Entropy Conf.</td><td rowspan="2">BaseCal</td><td colspan="2">UQAC</td><td rowspan="2">Verb.</td><td colspan="2">DTC (Ours)</td><td rowspan="2">PRM (ref.)</td></tr><tr><td>attn</td><td>mean</td><td>prod</td><td>lin</td></tr><tr><td colspan="10">MATH-500</td></tr><tr><td>Qwen2.5-7B</td><td>76.1</td><td>67.0</td><td>66.5</td><td>61.4</td><td>58.5</td><td>59.2</td><td>62.0</td><td>74.7</td><td>80.4</td><td>83.8</td><td>95.4</td></tr><tr><td>Qwen2.5-14B</td><td>80.0</td><td>65.7</td><td>65.0</td><td>65.0</td><td>56.0</td><td>56.6</td><td>64.9</td><td>74.8</td><td>81.9</td><td>85.0</td><td>94.3</td></tr><tr><td>Qwen2.5-32B</td><td>82.4</td><td>66.7</td><td>65.8</td><td>65.5</td><td>50.6</td><td>59.7</td><td>65.0</td><td>83.8</td><td>82.7</td><td>85.8</td><td>94.4</td></tr><tr><td>Qwen3-8B</td><td>83.9</td><td>75.0</td><td>73.4</td><td>72.4</td><td>52.3</td><td>59.1</td><td>61.6</td><td>57.9</td><td>79.3</td><td>73.9</td><td>90.9</td></tr><tr><td>Qwen3-14B</td><td>86.5</td><td>74.3</td><td>72.8</td><td>72.0</td><td>53.2</td><td>66.1</td><td>75.6</td><td>61.8</td><td>79.4</td><td>75.6</td><td>89.8</td></tr><tr><td>Qwen3-32B</td><td>83.7</td><td>73.6</td><td>71.6</td><td>70.9</td><td></td><td>58.8</td><td>63.1</td><td>63.3</td><td>80.6</td><td>78.9</td><td>91.3</td></tr><tr><td>Gemma3-12B</td><td>84.9</td><td>61.9</td><td>60.3</td><td>62.7</td><td>40.9</td><td>61.9</td><td>58.1</td><td>84.1</td><td>81.9</td><td>84.0</td><td>93.6</td></tr><tr><td>Gemma3-27B</td><td>89.2</td><td>63.6</td><td>61.8</td><td>63.3</td><td>38.9</td><td>76.5</td><td>81.7</td><td>87.2</td><td>85.5</td><td>87.9</td><td>93.2</td></tr><tr><td colspan="10">AMC23</td></tr><tr><td>Qwen2.5-7B</td><td>53.6</td><td>72.4</td><td>71.8</td><td>67.9</td><td>66.6</td><td>63.9</td><td>68.1</td><td>67.1</td><td>81.3</td><td>81.2</td><td>90.8</td></tr><tr><td>Qwen2.5-14B</td><td>61.2</td><td>66.9</td><td>66.4</td><td>67.7</td><td>62.3</td><td>57.1</td><td>68.5</td><td>73.0</td><td>79.2</td><td>80.3</td><td>90.5</td></tr><tr><td>Qwen2.5-32B</td><td>66.4</td><td>67.1</td><td>66.3</td><td>69.8</td><td>60.8</td><td>62.2</td><td>68.6</td><td>73.7</td><td>77.6</td><td>78.0</td><td>87.2</td></tr><tr><td>Qwen3-8B</td><td>68.6</td><td>73.8</td><td>72.0</td><td>71.6</td><td>56.2</td><td>41.1</td><td>47.7</td><td>57.8</td><td>78.7</td><td>76.0</td><td>90.6</td></tr><tr><td>Qwen3-14B</td><td>74.1</td><td>73.5</td><td>71.5</td><td>72.6</td><td>56.6</td><td>66.1</td><td>72.9</td><td>68.8</td><td>79.2</td><td>75.3</td><td>89.5</td></tr><tr><td>Qwen3-32B</td><td>67.5</td><td>75.3</td><td>73.4</td><td>74.4</td><td></td><td>55.1</td><td>57.2</td><td>67.7</td><td>80.3</td><td>78.6</td><td>89.0</td></tr><tr><td>Gemma3-12B</td><td>66.8</td><td>69.7</td><td>68.6</td><td>70.0</td><td>54.2</td><td>61.2</td><td>61.9</td><td>78.7</td><td>79.9</td><td>78.7</td><td>86.6</td></tr><tr><td>Gemma3-27B</td><td>76.9</td><td>68.5</td><td>67.1</td><td>68.5</td><td>47.8</td><td>72.7</td><td>76.4</td><td>85.6</td><td>82.6</td><td>80.4</td><td>91.1</td></tr><tr><td colspan="10">AIME24</td></tr><tr><td>Qwen2.5-7B 12.6</td><td></td><td>79.0</td><td>78.0</td><td>74.4</td><td>75.2</td><td>55.1</td><td>61.5</td><td>79.0</td><td>84.4</td><td>75.6</td><td>96.2</td></tr><tr><td>Qwen2.5-14B</td><td>13.8</td><td>71.3</td><td>71.0</td><td>75.2</td><td>65.5</td><td>61.4</td><td>70.1</td><td>77.8</td><td>76.5</td><td>74.2</td><td>89.7</td></tr><tr><td>Qwen2.5-32B</td><td>16.9</td><td>75.6</td><td>75.2</td><td>79.3</td><td>72.2</td><td>67.5</td><td>73.5</td><td>84.1</td><td>80.0</td><td>70.4</td><td>89.6</td></tr><tr><td>Qwen3-8B</td><td>27.7</td><td>71.3</td><td>69.9</td><td>72.1</td><td>58.4</td><td>53.5</td><td>56.4</td><td>65.5</td><td>74.1</td><td>65.7</td><td>89.0</td></tr><tr><td>Qwen3-14B</td><td>27.1</td><td>70.8</td><td>69.4</td><td>70.6</td><td>61.4</td><td>64.9</td><td>72.6</td><td>63.9</td><td>74.3</td><td>67.0</td><td>88.6</td></tr><tr><td>Qwen3-32B</td><td>28.3</td><td>71.2</td><td>69.9</td><td>73.4</td><td></td><td>52.1</td><td>52.5</td><td>74.7</td><td>76.2</td><td>69.6</td><td>91.7</td></tr><tr><td>Gemma3-12B</td><td>23.9</td><td>81.0</td><td>80.1</td><td>79.8</td><td>71.1</td><td>64.8</td><td>60.9</td><td>74.8</td><td>81.8</td><td>66.4</td><td>93.8</td></tr><tr><td>Gemma3-27B</td><td>29.0</td><td>78.5</td><td>78.1</td><td>78.3</td><td>68.2</td><td>66.4</td><td>65.1</td><td>87.8</td><td>80.2</td><td>64.0</td><td>90.8</td></tr><tr><td colspan="10">AIME25</td></tr><tr><td>Qwen2.5-7B</td><td></td><td>79.3</td><td>78.9</td><td>77.8</td><td>78.7</td><td>67.7</td><td>71.1</td><td>69.6</td><td>76.5</td><td>64.3</td><td>88.6</td></tr><tr><td>Qwen2.5-14B</td><td>9.1 14.7</td><td>75.1</td><td>74.7</td><td>75.9</td><td>71.8</td><td>68.5</td><td>75.5</td><td>74.4</td><td>75.9</td><td>62.2</td><td>88.3</td></tr><tr><td>Qwen2.5-32B</td><td>12.2</td><td>79.0</td><td>79.1</td><td>79.4</td><td>75.4</td><td>52.4</td><td>77.5</td><td>72.1</td><td>76.7</td><td>60.1</td><td>83.8</td></tr><tr><td>Qwen3-8B</td><td>22.5</td><td>78.9</td><td>77.4</td><td>77.2</td><td>67.9</td><td>55.1</td><td>56.2</td><td>63.3</td><td>82.4</td><td>70.9</td><td>87.5</td></tr><tr><td>Qwen3-14B</td><td>27.5</td><td>82.5</td><td>81.3</td><td>82.8</td><td>73.8</td><td>67.7</td><td>84.1</td><td>52.1</td><td>89.1</td><td>81.8</td><td>87.5</td></tr><tr><td>Qwen3-32B</td><td>23.7</td><td>80.3</td><td>78.8</td><td>81.3</td><td></td><td>61.2</td><td>61.7 66.7</td><td>74.8 85.7</td><td>86.8 88.9</td><td>79.9 76.3</td><td>88.1 89.6</td></tr><tr><td>Gemma3-12B</td><td>18.8 Gemma3-27B 25.3</td><td>84.9 86.9</td><td>84.1</td><td>84.4</td><td>69.0</td><td>64.8</td></table>

Table 6: Uncertainty quantification performance with Qwen3-4B in black-box settings.
<table><tr><td rowspan="2">Method</td><td colspan="3">AIME24</td><td colspan="3">AIME25</td><td colspan="3">HMMT25</td><td colspan="3">HMMT26</td></tr><tr><td>Acc↑</td><td>ECE↓ AUC↑ Acc↑</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>ECE↓ AUC↑ Acc↑ ECE↓ AUC↑ Acc↑ ECE↓ AUC↑</td><td></td></tr><tr><td>Default CoT</td><td>61.4</td><td></td><td></td><td>45.6</td><td></td><td></td><td>30.3</td><td></td><td></td><td>34.7</td><td></td><td></td></tr><tr><td>→ + PRM</td><td></td><td>36.0</td><td>82.9</td><td></td><td>36.6</td><td>80.3</td><td></td><td>37.8</td><td>53.3</td><td></td><td>33.4</td><td>78.0</td></tr><tr><td> $ + \mathrm { D T C _ { \mathrm { l i n } } }$ </td><td></td><td>11.3</td><td>72.9</td><td></td><td>9.0</td><td>81.5</td><td></td><td>13.1</td><td>69.6</td><td></td><td>13.0</td><td>74.6</td></tr><tr><td> $ + \mathrm { D T C _ { p r o d } }$ </td><td></td><td>16.3</td><td>75.8</td><td></td><td>10.7</td><td>84.8</td><td></td><td>15.4</td><td>78.0</td><td></td><td>14.3</td><td>79.0</td></tr><tr><td>Verb. Conf.</td><td>61.8</td><td>45.6</td><td>80.8</td><td>46.6</td><td>45.9</td><td>85.6</td><td>27.3</td><td>45.6</td><td>81.9</td><td>33.0</td><td>45.1</td><td>79.8</td></tr><tr><td> $ + \mathrm { D T C _ { \mathrm { l i n } } }$ </td><td></td><td>11.9</td><td>71.6</td><td></td><td>9.9</td><td>80.2</td><td></td><td>16.5</td><td>66.5</td><td></td><td>12.9</td><td>77.1</td></tr><tr><td> $ + \mathrm { D T C _ { p r o d } }$ </td><td></td><td>17.4</td><td>74.7</td><td></td><td>14.3</td><td>84.4</td><td></td><td>16.1</td><td>73.7</td><td></td><td>11.6</td><td>81.3</td></tr><tr><td>Verb. TopK</td><td>52.4</td><td>47.0</td><td>68.3</td><td>42.0</td><td>47.0</td><td>75.3</td><td>26.4</td><td>46.8</td><td>74.0</td><td>33.3</td><td>45.5</td><td>77.3</td></tr><tr><td> $ + \mathrm { D T C _ { \mathrm { l i n } } }$ </td><td></td><td>19.3</td><td>60.6</td><td></td><td>16.2</td><td>74.0</td><td></td><td>22.1</td><td>54.2</td><td></td><td>9.9</td><td>70.9</td></tr><tr><td> $ + \mathrm { D T C _ { p r o d } }$ </td><td></td><td>30.1</td><td>61.8</td><td></td><td>31.4</td><td>79.0</td><td></td><td>30.2</td><td>62.7</td><td></td><td>25.0</td><td>75.1</td></tr><tr><td>Verb. PD</td><td>59.3</td><td>46.9</td><td>69.9</td><td>46.9</td><td>48.1</td><td>73.7 26.7</td><td></td><td>43.9</td><td>75.5</td><td>32.5</td><td>42.9</td><td>70.9</td></tr><tr><td> $ + \mathrm { D T C _ { \mathrm { l i n } } }$ </td><td></td><td>14.4</td><td>68.7</td><td></td><td>9.4</td><td>76.3</td><td></td><td>14.8</td><td>65.6</td><td></td><td>7.2</td><td>77.0</td></tr><tr><td> $ + \mathrm { D T C _ { p r o d } }$ </td><td></td><td>25.7</td><td>72.6</td><td></td><td>22.4</td><td>82.1</td><td></td><td>22.7</td><td>75.5</td><td></td><td>20.1</td><td>82.2</td></tr></table>

Table 7: Uncertainty quantification performance with Qwen3-30B-A3B in black-box settings.
<table><tr><td rowspan="2">Method</td><td colspan="3">AIME24</td><td colspan="3">AIME25</td><td colspan="3">HMMT25</td><td colspan="3">HMMT26</td></tr><tr><td>Acc↑</td><td>ECE↓ AUC↑</td><td></td><td>Acc↑</td><td>ECE↓ AUC↑</td><td></td><td>Acc↑</td><td></td><td>ECE↓ AUC↑ Ac↑ ECE↓ AUC↑</td><td></td><td></td><td></td></tr><tr><td>Default CoT</td><td>73.4</td><td></td><td></td><td>59.8</td><td></td><td></td><td>42.4</td><td></td><td></td><td>43.5</td><td></td><td></td></tr><tr><td>→ + PRM</td><td></td><td>37.5</td><td>86.0</td><td></td><td>37.8</td><td>78.4</td><td></td><td>38.7</td><td>53.6</td><td></td><td>34.6</td><td>77.4</td></tr><tr><td> $ + \mathrm { D T C _ { \mathrm { l i n } } }$ </td><td></td><td>15.9</td><td>70.2</td><td></td><td>12.3</td><td>75.8</td><td></td><td>13.1</td><td>72.1</td><td></td><td>9.6</td><td>77.8</td></tr><tr><td> $ + \mathrm { D T C _ { p r o d } }$ </td><td></td><td>18.2</td><td>75.1</td><td></td><td>16.6</td><td>79.6</td><td></td><td>19.5</td><td>78.7</td><td></td><td>17.1</td><td>80.5</td></tr><tr><td>Verb. Conf.</td><td>75.7</td><td>44.5</td><td>83.9</td><td>60.6</td><td>44.3</td><td>80.8</td><td>41.8</td><td>42.9</td><td>85.3</td><td>43.9</td><td>43.9</td><td>79.2</td></tr><tr><td> $ + \mathrm { D T C _ { \mathrm { l i n } } }$ </td><td></td><td>14.0</td><td>73.7</td><td></td><td>13.2</td><td>73.7</td><td></td><td>11.4</td><td>74.3</td><td></td><td>10.7</td><td>78.5</td></tr><tr><td> $ + \mathrm { D T C _ { p r o d } }$ </td><td></td><td>20.6</td><td>78.4</td><td></td><td>18.4</td><td>77.8</td><td></td><td>19.3</td><td>80.5</td><td></td><td>16.6</td><td>81.4</td></tr><tr><td>Verb. TopK</td><td>71.1</td><td>40.2</td><td>88.2</td><td>55.8</td><td>40.6</td><td>88.7</td><td>40.7</td><td>41.2</td><td>85.4</td><td>43.4</td><td>41.6</td><td>83.0</td></tr><tr><td> $ + \mathrm { D T C _ { \mathrm { l i n } } }$ </td><td></td><td>36.4</td><td>44.3</td><td></td><td>35.2</td><td>49.0</td><td></td><td>22.4</td><td>64.5</td><td></td><td>21.9</td><td>66.6</td></tr><tr><td> $ + \mathrm { D T C _ { p r o d } }$ </td><td></td><td>46.7</td><td>45.2</td><td></td><td>47.0</td><td>50.2</td><td></td><td>32.9</td><td>71.0</td><td></td><td>37.0</td><td>69.0</td></tr><tr><td>Verb. PD</td><td>75.3</td><td>33.5</td><td>75.5</td><td>62.6</td><td>34.1</td><td>69.6</td><td>41.0</td><td>31.1</td><td>77.3</td><td>43.7</td><td>34.9</td><td>72.3</td></tr><tr><td> $ + \mathrm { D T C _ { \mathrm { l i n } } }$ </td><td></td><td>18.8</td><td>63.0</td><td></td><td>16.8</td><td>72.0</td><td></td><td>17.9</td><td>68.4</td><td></td><td>15.3</td><td>77.3</td></tr><tr><td> $ + \mathrm { D T C _ { p r o d } }$ </td><td></td><td>31.7</td><td>67.3</td><td></td><td>29.4</td><td>75.5</td><td></td><td>29.2</td><td>76.3</td><td></td><td>29.3</td><td>80.6</td></tr></table>

Table 8: Verbalized vs. w/ DTC on Qwen3-30B-A3B.
<table><tr><td rowspan="2">Method</td><td colspan="2">AIME24</td><td colspan="2">AIME25</td><td colspan="2">HMMT25</td><td colspan="2">HMMT26</td></tr><tr><td>ECE↓</td><td>AUC↑</td><td>ECE↓</td><td>AUC↑</td><td>ECE↓</td><td>AUC↑</td><td>ECE↓</td><td>AUC↑</td></tr><tr><td>Verb. Conf.</td><td>44.5</td><td>83.9</td><td>44.3</td><td>80.8</td><td>42.9</td><td>85.3</td><td>43.9</td><td>79.2</td></tr><tr><td>w/ DTC</td><td>17.8</td><td>87.0</td><td>15.7</td><td>83.8</td><td>13.0</td><td>87.9</td><td>17.4</td><td>85.9</td></tr><tr><td>Verb. TopK 40.2</td><td></td><td>88.2</td><td>40.6</td><td>88.7</td><td>41.2</td><td>85.4</td><td>41.6</td><td>83.0</td></tr><tr><td>w/ DTC</td><td>11.1</td><td>83.0</td><td>10.4</td><td>84.9</td><td>5.4</td><td>86.5</td><td>6.1</td><td>86.4</td></tr><tr><td>Verb. PD</td><td>33.5</td><td>75.5</td><td>34.1</td><td>69.6</td><td>31.1</td><td>77.3</td><td>34.9</td><td>72.3</td></tr><tr><td>w/ DTC</td><td>23.8</td><td>79.7</td><td>23.4</td><td>76.9</td><td>22.1</td><td>80.0</td><td>20.1</td><td>79.5</td></tr></table>

Table 9: Verbalized vs. w/ DTC on Qwen3-4B.
<table><tr><td rowspan="2">Method</td><td colspan="2">AIME24</td><td colspan="2">AIME25</td><td colspan="2">HMMT25</td><td colspan="2">HMMT26</td></tr><tr><td>ECE↓</td><td>AUC↑</td><td>ECE↓</td><td>AUC↑</td><td>ECE↓</td><td>AUC↑</td><td>ECE↓</td><td>AUC↑</td></tr><tr><td>Verb. Conf.</td><td>45.6</td><td>80.8</td><td>45.9</td><td>85.6</td><td>45.6</td><td>81.9</td><td>45.1</td><td>79.8</td></tr><tr><td>w/ DTC</td><td>30.8</td><td>82.0</td><td>29.4</td><td>87.0</td><td>31.0</td><td>81.6</td><td>31.1</td><td>81.2</td></tr><tr><td>Verb. TopK 47.0</td><td></td><td>68.3</td><td>47.0</td><td>75.3</td><td>46.8</td><td>74.0</td><td>45.5</td><td>77.3</td></tr><tr><td>w/ DTC</td><td>33.7</td><td>68.6</td><td>32.8</td><td>75.5</td><td>34.3</td><td>73.8</td><td>30.9</td><td>76.9</td></tr><tr><td>Verb. PD</td><td>46.9</td><td>69.9</td><td>48.1</td><td>73.7</td><td>43.9</td><td>75.5</td><td>42.9</td><td>70.9</td></tr><tr><td>w/ DTC</td><td>33.6</td><td>72.8</td><td>35.7</td><td>76.6</td><td>30.3</td><td>77.2</td><td>29.9</td><td>73.4</td></tr></table>

## C ADDITIONAL ANALYSES

## C.1 BLACK-BOX AUXILIARY SIZE

For dual-auxiliary $\mathrm { D T C _ { \mathrm { l i n } } }$ , DeepSeek-V3.2 generates the trajectories and both auxiliaries are Qwen2.5 models (Figure 9). On AIME25 we search θ in steps of 0.05 and report the θ with the lowest ECE. The lowest ECE is similar across auxiliary pairs.

(a) Selected θ  
![](images/0be2d00553a984a633a9b5dffda2eb1a8f8da8cf6dab2685fefb8ceaa3d35a7d.jpg)

(b) Min ECE  
![](images/b330b5ad4b988bbabdc32ae242f89fb2d1c6e30e0c54b10ae4c451d6114ba810.jpg)

(c) AUROC at θ  
![](images/4223a78b366679ec86a38f76250d2fcd6ea3f56138181d380ab542383651393d.jpg)  
Figure 9: Black-box sensitivity of $\mathrm { D T C _ { \mathrm { l i n } } }$ to θ and auxiliary size. We search $\theta$ in steps of 0.05 and report the θ (a) with the lowest ECE (b) and its AUROC (c). Cells without a finished probe are left blank.

## C.2 BLACK-BOX THRESHOLD CURVES

With $A ^ { \prime \prime }$ fixed at 1.5B, we vary θ for black-box $\mathrm { D T C _ { \mathrm { l i n } } }$ on DeepSeek-V3.2 trajectories (Figure 10).   
The main experiments use $\theta { = } 0 . 7 0$ , where ECE is still low.

![](images/94d9523fd5fecd7d7e4cafd00ee4095707f3691e4e2bb233dd2b10cec0aedce1.jpg)  
Figure 10: $\mathrm { D T C _ { \mathrm { l i n } } }$ sensitivity to $\theta$ with $\mathcal { A } ^ { \prime \prime }$ fixed at 1.5B (DeepSeek-V3.2 generating; solid: AIME25; dashed: HMMT25). The shaded band marks the black-box operating point $\theta { = } 0 . 7 0$

## C.3 COUNT AND RATIO

Longer reasoning traces are often less accurate, and recent work links that drop to overthink ing (Ghosal et al., 2025; Gema et al., 2025). We check whether the divergent-token count is only a proxy for this length effect. Let L be the number of reasoning tokens on a path. The count grows with $L ,$ so the two panels of Figure 11 plot:

$$
m = \vert T _ { \mathrm { d i v } } ( \theta ) \vert , \qquad \frac { m } { L } = \frac { \vert T _ { \mathrm { d i v } } ( \theta ) \vert } { L } .\tag{16}
$$

These runs use the single-auxiliary setting of Figure 1(c): MATH-500, Qwen2.5-7B, with auxiliaries of 1.5B, 3B, 14B, and 32B, each at the threshold used there.

On the left of Figure 11, accuracy falls from about 0.97–0.99 at a count of zero to about 0.24–0.29 once the count reaches 11 or more. On the right, we group the same trajectories into ten quantile bins of the ratio. Accuracy still falls from about 0.97–0.99 in the lowest bin to about 0.45–0.54 in the highest. The correlations of the bin means are $r = - 0 . 9 8 5$ for the count and $r = - 0 . 9 3 7$ for the ratio. Dividing by path length does not remove the decline, so the count is not only a proxy for path length.

![](images/54f61888cea4399b676a9f4595400945057615f9f676b808a9b159d5b1688ff4.jpg)  
Figure 11: Divergent-token count and ratio against path accuracy, single-auxiliary Qwen2.5-7B on MATH-500. Each curve is one auxiliary at the threshold used in Figure 1(c).

## C.4 CHOICE OF DIVERGENCE MEASURE

We keep the count estimator and the auxiliaries fixed, and compare JSD with forward KL and reverse KL (Figures 12 and 13). White-box runs use a 1.5B auxiliary and generators of 7B, 14B, and 32B. Black-box runs use Qwen2.5-7B and Qwen2.5-1.5B. ECE against θ is U-shaped for all three scores, and AUROC rises where ECE falls. KL needs a larger θ than JSD, and that θ is less stable from model to model. JSD is steadier near the θ used in the main experiments.

## C.5 THRESHOLD AND SATURATION SENSITIVITY

In the white-box setting, Figure 14 varies θ and the saturation length n of $\mathrm { D T C _ { \mathrm { l i n } } }$ . Each panel uses the smallest generator in its family: Qwen2.5-7B, Qwen3-8B, or Gemma3-12B. The curves average MATH-500, AMC23, AIME24, and AIME25. The ECE minimum moves with the family, but the θ values from the main text are still where ECE is low, and the comparison across n supports $n = 1 0$

## C.6 BLACK-BOX SATURATION LENGTH

Black-box $\mathrm { D T C _ { \mathrm { l i n } } }$ uses DeepSeek-V3.2, Qwen3-4B, and Qwen3-30B-A3B, with Qwen2.5 auxiliaries of 7B and 1.5B (Figure 15). Averaged over AIME24, AIME25, and HMMT25, these curves also support $n = 1 0$

## C.7 OFFSET k FOR DTC<sub>prod</sub>

Figure 16 varies k in $\mathrm { D T C _ { p r o d } }$ on the smallest white-box generator in each family. ECE is averaged over MATH-500, AMC23, AIME24, and AIME25. ECE against θ stays U-shaped, and the minimum moves with k and with the family. At the θ used in the main text, k = 4 still gives low ECE, so we keep k = 4.

![](images/a80798f2c0dde5c538955c56f2ed49d271c357b37f7e7bee7e0d5ca1963213f8.jpg)

![](images/a941e754a3df877f617ed9fc93b783c9e2c668ad5362dede52d8250faee0971b.jpg)

![](images/11c9c3c26f060ab7c01d4f55ad896a4cb35365646ed1751a232869d798b6e644.jpg)

![](images/1c886f6eff64944b201c8404c2d91b66287e512f2d7c150412677e3238b5e64f.jpg)

![](images/c64d7e83dbf5926bf0f4c968f9edf3be4b220c1027fd26339e5303b080fe4a34.jpg)

![](images/7f536f02c97cc4a904a6080e64c73741a4cad47d5e1a7dfd10cfdfdf26655845.jpg)  
Figure 12: Divergence choice for divergent-token selection. Each column is one disagreement score (JSD, forward ${ \bar { \mathsf { K L } } } ,$ , reverse KL) for $\mathrm { D T C _ { \mathrm { l i n } } }$ with auxiliary fixed at 1.5B and generating models $\in \{ 7 , 1 4 , 3 2 \} \mathbf { B }$ (solid: AMC23; dashed: AIME24). The top row shows ECE (%) and the bottom row AUROC (%) versus θ.

![](images/d87e499f2bf88cf5aa0a8d1d38a0ba6f1fdfddca6059dd6e23d8bdcf6bfaf974.jpg)

![](images/b732976229f5642b5fcf9aabbacaa4a022caa29add9306215e7e79c0358d92a1.jpg)

![](images/4babd8edbb53c59778032e44a180510998e3809dd18ff4066611cf38559b37bd.jpg)

![](images/668c49b5990a9c44ee8101693d3a770e6c905fc66339a04f2ad41062bc7af2ce.jpg)

![](images/39897bc0789a51e087979c524b89a562a97b44dd8676718715b388c33a1054bd.jpg)

![](images/8739d270b786863c52944db5e7eb937248e76ddf6a4eb7fb6dff803fc7719e92.jpg)  
Figure 13: Divergence choice for black-box divergent-token selection. Each column is one disagreement score (JSD, forward KL, reverse KL) for $\bar { \mathrm { D T C } } _ { \mathrm { l i n } }$ with auxiliaries fixed at Qwen2.5-7B and Qwen2.5-1.5B and generators Qwen3-4B, Qwen3-30B-A3B, and DeepSeek-V3.2 (solid: AIME25; dashed: HMMT25). The top row shows ECE (%) and the bottom row AUROC (%) versus θ.

![](images/f490dd439dcdce46d794f290d7ea85a2cb4ae54c754a6b132a25149d1cdb8509.jpg)  
Figure 14: Saturation length n for $\operatorname { D T C } _ { \operatorname { l i n } } .$ The top row shows ECE (%) and the bottom row AUROC (%) versus θ at step 0.01, averaged over MATH-500, AMC23, AIME24, and AIME25.

![](images/ce85baab299ea5b99f2296bf26243dd565cbee6ec3720f6b8cc057693c5b36fb.jpg)  
Figure 15: Saturation length n for black-box $\mathrm { D T C _ { \mathrm { l i n } } }$ . Each column is one generator with Qwen2.5 auxiliaries 7B and 1.5B. The top row shows ECE (%) and the bottom row AUROC (%) versus θ at step 0.01, averaged over AIME24, AIME25, and HMMT25.

![](images/7f3adae5c61b2ad11550c4286964ee6fb2b7d67c050891d85bf428f7246495f8.jpg)

![](images/ee226a63377df85a8f6b574f8248ad4c58c539b712c0d49f9a136517d729006e.jpg)

![](images/595b35bfa082963790fadaea836dc7e388356a0fa522b7a0bd4748baf8888a9b.jpg)  
Figure 16: Offset k for $\mathrm { D T C } _ { \mathrm { p r o d } } .$ Each column is the smallest white-box generator in its family (Qwen2.5-7B, Qwen3-8B, Gemma3-12B) with the family auxiliary. ECE (%) is shown versus θ at step 0.01, averaged over MATH-500, AMC23, AIME24, and AIME25.