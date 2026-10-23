# Predictable performance is suboptimal and what it means for model routing

Guanhua Zhang\*, Moritz Hardt

Max Planck Institute for Intelligent Systems, Tübingen Tübingen AI Center

## Abstract

Large language models (LLMs) differ in capabilities and costs, motivating routing methods that adaptively choose a model for each query. These methods rely on performance prediction: given a query, estimate whether a model will answer it correctly. Most routing methods train separate predictors of candidate models’ performance to guide model selection. Our work shows that performance prediction only beats a simple baseline when the candidate model is suboptimal on the task or the performance predictor adds significant capacity relative to the candidate model. Specifically, we take a model’s self-confidence, the probability it assigns to its chosen answer, as a baseline and study how much a performance predictor can improve on it. First, we show that any performance predictor that outperforms a model’s self-confidence can be used to improve the model’s loss: the predictor’s advantage transfers exactly in log loss through recalibration. Second, accounting for the predictor’s additional parameters, we bound its advantage by the model’s suboptimality plus the gain from those parameters. Third, we extend our theoretical results to two-model cascades, which route queries to a more capable model when the first model is predicted to fail. Experiments with 16 language models on four benchmarks show that predictors gain a larger advantage over self-confidence on more suboptimal models. This advantage shrinks as models improve through finetuning. Cascading exhibits the same two patterns: predictor-based routing offers larger gains over confidencebased routing for more suboptimal models, and these gains diminish after finetuning. Code is available at https://github.com/socialfoundations/performance-prediction.

## 1 Introduction

Performance prediction estimates whether a model answers a query correctly (Kadavath et al., 2022; Shnitzer et al., 2024). These predictions can help routing systems decide when to trust an answer and when to call a stronger model (Geifman and El-Yaniv, 2017; Chen et al., 2024). By catching a weak model’s errors accurately, routing can achieve better performance at the same cost. But what makes a model’s errors predictable, and what does that reveal about the model?

In this paper, we study the predictability of a model’s performance relative to a simple baseline: the model’s self-confidence, the probability it assigns to its chosen answer. Our key observation is that any predictor that beats self-confidence can improve the model by the same amount. This means that a predictor’s advantage over self-confidence is limited by the model’s room for improvement. We formalize and test this connection, and extend the analysis to routing.

![](images/b0db1cf3f52611804c6a388100a2d58fe83006a2d2facf2fdc7d90c2316abbd2.jpg)  
Figure 1: Predictors have a larger advantage over self-confidence on more suboptimal models. Each point is a released (orange) or finetuned (blue) model. Empirical suboptimality is the gap to the best observed answer log loss on each task. Excess predictability E is the reduction in correctness log loss achieved by a separately trained 4B predictor over self-confidence. Colors identify training stage; shapes identify the release type. Lines are least-squares fits; r is Pearson correlation and $p _ { + }$ its one-sided p-value for positive correlation. See Section 3.4.1 for experimental details.

## 1.1 Our Contributions

Suboptimality bounds excess predictability. We connect performance prediction and model improvement through recalibration: replacing the model’s confidence in its chosen answer with the predictor’s estimate, and normalising the rest. We call the predictor’s reduction in correctness log loss relative to self-confidence excess predictability. Recalibration reduces the model’s answer log loss by exactly the same amount (Proposition 1). We assume a scaling floor: the lowest answer loss achievable on a fixed task at each parameter budget (Assumption 1). The recalibrated model cannot beat the floor at the combined parameter budget of the model and predictor. This gives the bound (Theorem 1)

$$
e x c e s s p r e d i c t a b i l i t y \ \leq \ s u b o p t i m a l i t y \ + \ c a p a c i t y g a i n \ \Big \vert .
$$

Here, suboptimality is the model’s gap to the floor at its own budget. The capacity gain is the decrease in the floor from adding the predictor’s parameters.

Greater excess predictability comes with greater suboptimality. Our upper bound does not imply that more suboptimal models are necessarily easier to predict beyond self-confidence. Empirically, however, we observe a positive association between the two. We conduct experiments on WinoGrande, CommonsenseQA, GSM8K, and MMLU-Pro, using 16 language models from eight families, ranging from 7 to 12B parameters. Using the gap to the best observed answer loss on each task as a proxy for suboptimality, we find a positive association with excess predictability on all four tasks, with Pearson correlations of 0.98, 0.98, 0.93, and 0.65, respectively (Figure 1). We also finetune the models, retaining the 59 model-task pairs whose development accuracy improves.

Answer loss falls in all 59 retained pairs and excess predictability falls in 55 (Figure 2), despite training separate 4B predictors for the released and finetuned models.

Suboptimality bounds error capture beyond confidence routing. We extend our analysis to two-model cascades, where a router decides when to escalate a query to a stronger model. We compare a learned router with confidence routing, which escalates the weak model’s lowestconfidence queries, at the same escalation rate. Any router that captures more weak-model errors can be converted into a predictor with positive excess predictability, even if the router does not explicitly estimate correctness (Lemma 1). This bounds the additionally captured weak-model errors by weak-model suboptimality and the capacity gain from the router’s added parameters (Theorem 2). The cascade accuracy gain equals the additional weak-model errors captured minus the additional strong-model errors encountered. The same bound applies to the cascade accuracy gain if the learned router encounters at least as many strong-model errors as confidence routing (Corollary 2).

More suboptimal models have larger routing advantages. Using the same models and predictors, we compare predictor-based routing with confidence routing at equal escalation rates. We measure the routing advantage as the gain in cascade accuracy, averaged over escalation rates. Models with greater empirical suboptimality tend to have larger routing advantages on all four tasks, with Pearson correlations between 0.66 and 0.91 (Figure 4). This advantage shrinks after finetuning in 53 of the 59 retained pairs (Section 4.3).

Main takeaway. Together, these results suggest three sources of predictability: the model’s selfconfidence, its suboptimality, and the predictor’s added capacity. Improving the underlying model generally reduces the additional value of learned prediction and routing beyond self-confidence.

## 2 Related Work

Performance prediction. Performance predictors, also called assessors, estimate whether a system will succeed on a given input (Hernández-Orallo et al., 2022; Zhou et al., 2022). They use model outputs (Yao and Koller, 2024), reference evaluations and question features (Pacchiardi et al., 2024), follow-up queries (Sam et al., 2025), or internal activations before the model answers (Moreno Cencerrado et al., 2026). Kapoor et al. (2024) finetune language models on graded answers to predict their own correctness and that of other models. Xiao et al. (2026) find that models predict their own correctness no better than unrelated models do, and train predictors that work across models. Beyond binary correctness, Hsu et al. (2025) estimate continuous quality scores and uncertainty intervals for free-form generation. At the benchmark level, studies forecast performance from training compute (Owen, 2024) or scores on other benchmarks (Ruan et al., 2024). Others estimate benchmark scores from a small set of evaluated examples (Maia Polo et al., 2024; Zhang et al., 2025a). We study where predictability on individual inputs comes from, using self-confidence as the baseline and correctness log loss to measure a predictor’s advantage.

Model routing and cascading. Performance predictions help routing systems balance cost and performance by choosing which models to use for each query (Ong et al., 2025). We use routing broadly to include cascading, where a cheaper model answers first and a router uses its output to decide whether to call a stronger model (Chen et al., 2024; Dekoninck et al., 2025). Routers learn which model to use from preference data, as in RouteLLM (Ong et al., 2025), benchmark results (Shnitzer et al., 202 ; Mohammadshahi et al., 202 ), or reward-model scores (Lu et al., 2024). Some predict query difficulty to guide model selection (Ding et al., 2024). RouterBench provides a common framework for evaluating routing methods (Hu et al., 2024). In cascades, routing rules often escalate to a larger model when the first model has low confidence or gives inconsistent answers (Varshney and Baral, 2022; Yue et al., 202 ; Chuang et al., 202 ). Learned routing rules use token-level uncertainty and model embeddings (Gupta et al., 2024) or verify the first model’s answer before deciding whether to escalate (Aggarwal et al., 2024). Wang et al. (2024a) also improve cascades by training the first model to account for deferral. Dekoninck et al. (2025) derive an optimal cascading policy and prove that an existing routing policy is optimal; both depend on good estimates of answer quality. Jitkrittum et al. (2023) show when confidence alone gives an optimal deferral rule and when a learned rule (Narasimhan et al., 2022) is needed. We bound how many more of the first model’s errors a learned router can catch than confidence routing under the same escalation rate.

Confidence and selective prediction. Language models can estimate whether their answers are correct (Kadavath et al., 2022) through stated confidence (Tian et al., 2023; Xiong et al., 2024) or agreement among sampled answers (Wang et al., 2023; Kuhn et al., 2023). Learned verifiers score candidate solutions to select better answers (Cobbe et al., 2021; Lightman et al., 2024), and the gap between generation and verification abilities is linked to self-improvement (Song et al., 2025). Confidence also guides decisions about when to answer. Selective prediction uses it to decide when to abstain (Chow, 1970; Geifman and El-Yaniv, 2017), while learning to defer also considers whether another expert will be correct (Mozannar and Sontag, 2020). For selective prediction, both calibration and how confidence ranks inputs matter (Rabanser and Papernot, 2025). Inputs with the same confidence can still differ in their probability of correctness; grouping loss measures this variation (Kull and Flach, 2015; Perez-Lebel et al., 2023). Melo et al. (2026) estimate grouping loss and excess decision risk to guide LLM cascades, and find that grouping loss decreases with model scale and instruction tuning. Our experiments examine how gains over confidence vary with answer-loss suboptimality across models of similar size and before and after task-specific finetuning.

Calibration and loss minimization. Language models can produce informative but poorly calibrated risk scores (Cruz et al., 2024). Post-hoc calibration adjusts confidence to match observed correctness (Platt, 2000; Guo et al., 2017; Kumar et al., 2019). Multicalibration extends this requirement to identifiable groups (Hebert-Johnson et al., 2018), including those defined by LLM embeddings and self-annotations (Detommaso et al., 2024). Loss minimization and multicalibration theory show how correcting predictable errors in probability estimates can reduce loss, including squared loss and cross-entropy (Gopalan et al., 2022, 2023; Błasiok et al., 2024). We show that any reduction in correctness log loss over self-confidence yields an equal reduction in answer log loss. Under our assumptions for combining models and counting parameters, this gain is bounded by the original model’s suboptimality plus the improvement possible with the added parameters.

## 3 Predictable Performance is Suboptimal

We show that gains over self-confidence transfer exactly to lower answer loss and are bounded by model suboptimality and added predictor capacity. Proofs are deferred to Appendix A.1.

## 3.1 Problem Definition

Let $( X , Y )$ be drawn from a fixed population, where $X \in { \mathcal { X } }$ is the query and $Y \in \mathcal { Y } = \{ 0 , \dots , K -$ 1} is the answer. A model $f \in { \mathcal { F } }$ is defined as a conditional probability mass function $f ( y | x )$ . Its point prediction is ${ \hat { f } } ( x ) = \operatorname { a r g m a x } _ { y } f ( y \mid x )$ , with ties broken by a fixed rule. We score any model $f$ by its answer loss $L _ { Y } ( f ) = \mathbb { E } [ - \log f ( Y \mid X ) ]$

We define the correctness indicator $Z = \mathbb { 1 } \{ Y = { \hat { f } } ( X ) \} \in \{ 0 , 1 \}$ . A performance predictor $g \in { \mathcal { G } }$ is defined as a conditional probability mass function $g ( z \mid x )$ that may read $f ^ { \prime } { \bf s }$ output on x. We require full support, $g ( z \mid x ) > 0$ , and score $g$ by its correctness loss $L _ { Z } ( g , f ) \stackrel { \cdot } { = } \mathbb { E } [ - \stackrel { \cdot } { \log } g ( Z | X ) ]$

The model’s self-confidence provides a natural baseline: any model $f$ induces a performance predictor $f _ { Z }$ with $f _ { Z } ( 1 \mid x ) = f ( \hat { \hat { f } } ( x ) \mid x )$ . We abbreviate $L _ { Z } ( f ) : = L _ { Z } ( f _ { Z } , f )$

The goal of performance prediction is to minimize $L _ { Z } ( g , f )$ , where $L _ { Z } ( f )$ serves as a baseline. We call $L _ { Z } ( f ) - L _ { Z } ( g , f )$ the excess predictability achieved by $g$ relative to $f ^ { \prime } \mathbf { s }$ self-confidence; a performance predictor is effective when its excess predictability is positive.

## 3.2 Any Effective Performance Predictor Improves the Model

We use $g$ to change how much probability f assigns to its chosen answer, while preserving the relative probabilities of the other answers.

Assuming $f _ { Z } ( 0 \mid x ) > 0 .$ , define the alternative distribution and the recalibrated model by

$$
f _ { \mathrm { A l t } } ( y \mid x ) = f ( y \mid x ) / f _ { Z } ( 0 \mid x ) , \qquad y \neq { \hat { f } } ( x ) ,\tag{1}
$$

$$
\begin{array} { r } { \mathcal { C } _ { g } ( f ) ( y | x ) = \left\{ \begin{array} { l l } { g ( 1 | x ) } & { y = \hat { f } ( x ) , } \\ { g ( 0 | x ) f _ { \mathrm { A l t } } ( y | x ) } & { y \neq \hat { f } ( x ) } \end{array} \right. . } \end{array}\tag{2}
$$

Thus, $ { \mathcal { C } } _ { g } ( f )$ assigns mass $g ( 1 \mid x )$ to $\hat { f } ( x )$ and distributes the remaining mass among the other answers in their original proportions. Since only the correctness probabilities change, the following proposition transfers the predictor’s improvement exactly to the answer loss.

Proposition 1 (Exact transfer). For any f with $L _ { Y } ( f ) <$ ∞ and any performance predictor $g$

$$
{ \Big | } L _ { Y } { \big ( } { \mathcal { C } } _ { g } ( f ) { \big ) } = L _ { Y } { \big ( } f { \big ) } - { \big ( } L _ { Z } ( f ) - L _ { Z } ( g , f ) { \big ) } { \Big | } .\tag{3}
$$

Recalibrating with an effective g strictly lowers the answer loss by exactly its excess predictability over $f ^ { \prime } { \mathsf { s } }$ self-confidence, $L _ { Z } ( f ) - L _ { Z } ( { \dot { g } } , f ) > 0$ . Recalibration is also hard to beat: any model with lower answer loss than $ { \mathcal { C } } _ { g } ( f )$ must predict $f ^ { \prime } \mathbf { s }$ correctness better than ${ \mathit { g } } ,$ or predict better among the alternative answers when $f$ is wrong (Proposition 2 in Appendix $\mathrm { A } . 2 )$ .

## 3.3 Predictable Performance is Suboptimal

For parameter budgets $P , Q \ge 0 ,$ let ${ \mathcal { F } } _ { P } \subseteq { \mathcal { F } }$ denote the models with at most $P$ parameters, and let ${ \mathcal { G } } _ { Q } \subseteq { \mathcal { G } }$ denote the performance predictors with at most Q parameters. Both sets are therefore nested in their respective budgets.

We allow models in $\mathcal { F }$ to use performance predictors and routers as components, and apply the same convention to predictors in ${ \mathcal { G } } .$ . We count all parameters, including frozen weights, but count shared parameters only once. The fixed arithmetic and decision operations used below add no parameters. Thus, recalibrating $f \in \mathcal { F } _ { P }$ with $g \in { \mathcal { G } } _ { Q }$ gives a model $\mathcal { C } _ { g } ( f ) \in \mathcal { F } _ { P + Q }$ . The loss floor and suboptimality statements below apply to the full set $\mathcal { F }$ on the fixed task distribution of $( X , Y )$

Empirical neural scaling laws suggest that the best loss achievable follows a predictable frontier as model size increases (Kaplan et al., 2020; Hoffmann et al., 2022). We idealize this as follows. Assumption 1 (Scaling floor). For each budget $P ,$ some model $f _ { P } ^ { * } \in \mathcal { F } _ { P }$ achieves the lowest loss $L _ { Y } ^ { * } ( P )$ in ${ \mathcal { F } } _ { P } ; $

$$
L _ { Y } ( f ) \geq L _ { Y } ^ { * } ( P ) = L _ { Y } ( f _ { P } ^ { * } ) \qquad f o r e v e r y f \in \mathcal { F } _ { P } .\tag{4}
$$

The floor $L _ { Y } ^ { * } ( P )$ is non-increasing in $P ,$ since a larger budget includes all models available at a smaller budget. Relative to this floor, define the suboptimality at budget P of $f \in \mathcal { F } _ { P }$ as

$$
\delta _ { P } ( f ) : = L _ { Y } ( f ) - L _ { Y } ^ { * } ( P ) \geq 0 .\tag{5}
$$

The experiments use the gap to the best observed loss in a cohort of similar-size models as a proxy for this suboptimality (Section 3.4.2).

The recalibrated model uses at most $P + Q$ parameters, so its loss cannot fall below $L _ { Y } ^ { * } ( P + Q )$ Together with exact transfer, this limits how much $g$ can improve on $f ^ { \prime } \mathbf { s }$ self-confidence. Theorem 1 (Excess predictability bounded by suboptimality and added capacity). Under Assumption 1, every $f \in \mathcal { F } _ { P }$ with $L _ { Y } ( f ) < \infty$ and every $g \in { \mathcal { G } } _ { Q }$ satisfy

$$
\boxed { L _ { Z } ( f ) - L _ { Z } ( g , f ) \le \big [ L _ { Y } ^ { * } ( P ) - L _ { Y } ^ { * } ( P + Q ) \big ] + \delta _ { P } ( f ) } .\tag{6}
$$

Equation 6 separates two sources of excess predictability. For the loss-optimal model $f _ { P } ^ { * } ,$ , a predictor can recover at most the scaling gain ${ \bar { L _ { Y } ^ { * } } } ( P ) - { L _ { Y } ^ { * } } ( P + Q )$ , the decrease in the loss floor from adding $Q$ parameters. A model farther from the floor increases this recoverable excess predictability by at most its suboptimality $\delta _ { P } ( f )$ . Any improvement over $f ^ { \prime } \mathbf { s }$ self-confidence that exceeds the scaling gain therefore certifies $\delta _ { P } ( f ) > 0 .$ . In particular, if the performance predictor adds no parameters, then strictly positive excess predictability implies $\delta _ { P } ( f ) > 0$

Additional results. First, under diminishing returns, a predictor much smaller than the model has excess predictability bounded by the model’s suboptimality plus a small capacity correction (Corollary ${ \bf { 1 } } ,$ Appendix $\mathrm { A } . 3 )$ . Second, suboptimality does not determine excess predictability: lowering answer loss does not necessarily lower correctness loss. Models can have the same maximum excess predictability yet differ in suboptimality because they assign different probabilities to the answers they do not choose (Proposition 3, Appendix $\mathrm { A . } 4 )$

## 3.4 Experiments

Motivated by the preceding bound on excess predictability, we study two empirical questions on 16 language models across four benchmarks. Do more suboptimal models exhibit greater excess predictability over self-confidence? Does task-specific finetuning reduce this advantage?

## 3.4.1 Setup

Models and tasks. We evaluate 16 base and instruction-tuned models from eight families, with $7$ to 12B parameters. We select four tasks, WinoGrande (Sakaguchi et al., 2020), CommonsenseQA (Talmor et al., 2019), GSM8K (Cobbe et al., 2021) and MMLU-Pro (Wang et al., 2024b). WinoGrande and CommonsenseQA are evaluated by scoring answer options, and GSM8K and MMLU-Pro are evaluated by generating answers. The binary task WinoGrande is a sanity check: predicting a fixed model’s correctness is equivalent to predicting the answer. Appendix B.1 lists model releases and data splits, shared across models.

Evaluation. We use five-shot evaluation with lm-eval-harness (Gao et al., 2024). The answer distribution $f ( \cdot \mid x )$ comes from normalized option likelihoods on the first two tasks and answer frequencies from 128 sampled solutions per question on the generation tasks. We select the highest-probability answer $\hat { f }$ and use its probability as self-confidence. On test questions, we measure answer log loss $L _ { Y } ( f )$ and correctness log loss $L _ { Z } ( f ) ;$ cascades are evaluated by accuracy. Appendix B.2 details scoring, sampling, answer extraction, and probability estimation.

Finetuning. We use LoRA (Hu et al., 2022) to finetune on gold answers for the optionscoring tasks and on reasoning traces by Qwen3.5-122B-A10B for the generation tasks, selecting checkpoints on development performance. Our experiment design is built upon the idea that task-specific finetuning reduces suboptimality. Therefore, for each task, we consider all models whose development accuracy does not improve after finetuning to be anomalous and remove the corresponding model pairs from the analysis. This leaves 16, ${ \bf 1 6 , 1 4 , }$ and 13 model pairs for the respective tasks. Appendix $\mathrm { B } . 3$ gives training and retention details.

Suboptimality. For each task, let C contain the released and finetuned versions of the retained models. We use their best observed answer loss as a proxy for the loss floor at similar model sizes:

$$
{ \widehat { \delta } } ( f ) = L _ { Y } ( f ) - \operatorname* { m i n } _ { h \in { \mathcal { C } } } L _ { Y } ( h ) .\tag{7}
$$

This reference is fixed for all models, so changes in empirical suboptimality equal changes in answer loss; changing the reference shifts all gaps equally and leaves correlations unchanged.

Performance predictor. We train a separate Qwen3-Embedding-4B predictor (Zhang et al., 2025b) for each model, task, and finetuning stage. Its input contains the question, the model’s final answer, option probabilities or sampled-answer frequencies, and, for generation tasks, a sampled reasoning trace ending in the majority answer. A linear head on the embedding corrects the selfconfidence logit; we train the head and LoRA adapters from a Platt-calibrated initialization (Platt, 2000). We fit predictors on $8 5 \%$ of the development set and select the checkpoint and seed with lowest correctness loss on the remaining $1 5 \%$ . On the test set, excess predictability is

$$
E ( f ) = L _ { Z } ( f ) - L _ { Z } ( g , f ) .\tag{8}
$$

Appendix B.4 gives the architecture and training details.

## 3.4.2 Results

Greater excess predictability comes with greater suboptimality. Figure 1 shows a positive correlation between empirical suboptimality and excess predictability across all four tasks. We plot released and finetuned models together because the relationship concerns a model’s suboptimality, regardless of how it was trained. Appendix C.2 shows the two groups separately; the association also holds among released models alone. The association is stronger on the simpler tasks, WinoGrande and CommonsenseQA, with Pearson correlations of 0.98 for both; on WinoGrande this is the expected sanity-check outcome (Section 3.4.1). On the generation-based tasks, GSM8K and MMLU-Pro, the correlations are 0.93 and 0.65, respectively. The weaker association on MMLU-Pro may partly reflect variation in alternative loss: models can differ in answer loss without differing in excess predictability (Appendix A.4).

![](images/ff0e36e4da7a47d62f7ac8c2b41b7ee194da9b19c3d0089d137e509b08ac5ad4.jpg)  
Figure 2: Finetuning reduces excess predictability in 55 of 59 model-task pairs. Each point compares one released model with its selected finetuned version; both axes show released minus finetuned values. ∆δb is the reduction in answer log loss, and ∆E is the reduction in excess predictability. Points above zero have lower excess predictability after finetuning; counts report these pairs. Suboptimality decreases in all 59 pairs. Shapes identify the release type; lines are least-squares fits; r is Pearson correlation and $p _ { + }$ its one-sided p-value for positive correlation.

Finetuning reduces excess predictability. Excess predictability falls in 55 of the 59 retained model-task pairs after finetuning (Figure 2): 14 of 16 on WinoGrande, all 16 on CommonsenseQA, 12 of 14 on GSM8K, and all 13 on MMLU-Pro. The four exceptions have relatively small reductions in suboptimality, indicating that finetuning improves their answer loss only modestly. Note that we train a new predictor for each finetuned model, yet its gain over self-confidence still shrinks. Larger reductions in suboptimality tend to accompany larger reductions in excess predictability, with Pearson correlations of 0.99, 0.99, 0.78, and 0.49 in task order.

![](images/7a098069bf94856107e9d1aef61336ad043a63d73d66b360ed02432f5d815116.jpg)  
Figure 3: Answer and correctness losses during finetuning of Llama-3.1-8B on CommonsenseQA. Shading shows excess predictability E(f); predictor loss averages dictability E(f); predictor loss averages three seeds. three seeds.

During finetuning of Llama-3.1-8B on CommonsenseQA, self-confidence catches up with separately retrained predictors: excess predictability falls from 0.107 to 0.004 nats (Figure 3). Appendix B.3 gives the trajectory setup.

Predictor capacity. We compare how much excess predictability $0 . 6 \mathrm { B } , 4 \mathrm { B } ,$ , and 8B predictors capture on the same models (Appendix $C . 3 )$ . Larger predictors generally capture more: the 0.6B predictor adds little beyond Platt-calibrated self-confidence, while the $4 ^ { \mathrm { B } }$ and 8B predictors generally improve on it. Our main finding holds at all three predictor sizes: finetuning reduces mean excess predictability on every task. For the $4 ^ { \mathrm { B } }$ predictor, mean gains over Platt-calibrated self-confidence also decrease after finetuning on every task (Appendix Figure 9).

## 4 Suboptimality Bounds Cascading

In a cascade, a router decides which queries to send from a weak model to a stronger model. We compare a learned router with the weak model’s self-confidence at the same escalation rate. Cascades are judged by accuracy at that rate, so this section bounds error rates rather than log loss. The key connection is that a router that finds more weak-model errors also supplies a better performance predictor. Proofs are deferred to Appendices $\mathrm { A } . 5 { - } \mathrm { A } . 7$

## 4.1 Cascading Setup

Let $f _ { 1 } \in \mathcal { F } _ { P _ { 1 } }$ and $f _ { 2 } \in \mathcal { F } _ { P _ { 2 } }$ be the weak and strong models, with $P _ { 1 } < P _ { 2 }$ and $L _ { Y } ( f _ { 1 } ) < \infty$ . For $i = 1 , 2 $ , write $Z _ { i } = \mathbb { 1 } \{ Y \stackrel { - } { = } \hat { f } _ { i } ( X ) \}$ and $\epsilon _ { i } ( x ) = \operatorname* { P r } ( Z _ { i } = 0 \mid X = x )$ for correctness and the actual error probability. The weak model estimates its error probability by $f _ { 1 , Z } ( 0 \mid x ) = 1 - f _ { 1 } ( \hat { f } _ { 1 } ( x ) \mid x )$ assumed strictly between zero and one almost surely.

A budget-Q router assigns a score $r : \mathcal { X } \to \mathbb { R }$ using at most Q additional parameters beyond $f _ { 1 }$ . At escalation rate $\rho \in [ 0 , 1 ]$ , we escalate the highest-scoring ρ fraction of queries to $f _ { 2 } ,$ , breaking ties uniformly at random. Its routing decision $B _ { r , \rho } ( x )$ is one for escalation and zero for keeping $f _ { 1 } { ' } \mathrm { s }$ answer. Self-confidence routing applies the same rule to $f _ { 1 , Z } ( 0 | x )$ , giving $B _ { f , \rho }$ . At a fixed rate, abbreviate these decisions as $B _ { r } , B _ { f } ,$ , with $\mathbb { E } [ B _ { r } ] = \mathbb { E } [ B _ { f } ] = \rho ,$ , and set $u = B _ { r } - B _ { f }$

Writing $\operatorname { E r r } ( f )$ for a model’s error rate, the difference between the two cascades decomposes as

$$
\begin{array} { r } { \mathrm { E r r } ( f _ { \mathrm { c a s } , B _ { r } } ) - \mathrm { E r r } ( f _ { \mathrm { c a s } , B _ { f } } ) = - \underbrace { { \mathbb E } [ u \epsilon _ { 1 } ( X ) ] } _ { \mathrm { a d d i t i o n a l ~ w e a k ~ e r r o r s ~ c a p t u r e d } } + \underbrace { { \mathbb E } [ u \epsilon _ { 2 } ( X ) ] } _ { \mathrm { a d d i t i o n a l ~ s t r o n g ~ e r r o r s ~ e n c o u n t e r e d } } . } \end{array}\tag{9}
$$

A router can improve on confidence by capturing more weak-model errors, encountering fewer strong-model errors, or both. We first bound the weak-error contribution, then state when this also bounds the cascade advantage.

## 4.2 From Performance Prediction to Cascade Bounds

At a fixed escalation rate, $B _ { f }$ escalates the lowest-confidence $\rho$ fraction of queries and therefore maximizes the predicted weak errors captured, $\mathbb { E } [ B _ { f } f _ { 1 , Z } ( 0 | X ) ]$ . Thus $\mathbb { E } \bar { [ u f _ { 1 , Z } ( 0 | X ) ] } \le 0$ . A router that captures more actual errors must therefore identify where self-confidence misestimates their probability:

$$
\begin{array} { r l } & { \mathbb { E } \bigl [ u \epsilon _ { 1 } ( X ) \bigr ] = \mathbb { E } \bigl [ u f _ { 1 , Z } ( 0 \mid X ) \bigr ] + \mathbb { E } \bigl [ u \bigl ( \epsilon _ { 1 } \bigl ( X \bigr ) - f _ { 1 , Z } ( 0 \mid X ) \bigr ) \bigr ] } \\ & { \qquad \leq \mathbb { E } \bigl [ u \bigl ( \epsilon _ { 1 } ( X ) - f _ { 1 , Z } ( 0 \mid X ) \bigr ) \bigr ] . } \end{array}\tag{10}
$$

![](images/52554e20cb27dc9825830d44793c7c2b3da1113fdeddb08ef2ae0050e849e91c.jpg)  
Figure $4 { : }$ More suboptimal models have larger routing advantages. Each point is one released or finetuned weak-model version. Empirical suboptimality is the gap to the best observed answer log loss on each task. Routing advantage 100D is the cascade accuracy gain from a 4B predictor over self-confidence routing to Qwen3.5-122B-A10B, averaged over equal escalation rates and expressed in percentage points. Colors identify training stage; shapes identify the release type. Lines are least-squares fits; r is Pearson correlation and $p _ { + }$ its one-sided p-value for positive correlation.

We can construct a performance predictor $g _ { \lambda }$ from the routing score r by shifting the weak model’s confidence where its routing decisions differ from self-confidence routing:

$$
\mathrm { l o g i t } g _ { \lambda } ( 0 \mid x ) = \mathrm { l o g i t } f _ { 1 , Z } ( 0 \mid x ) + \lambda u ( x ) ,\tag{11}
$$

where $\mathrm { l o g i t } ( a ) = \log ( a / ( 1 - a ) )$ and $g _ { \lambda } ( 1 \mid x ) = 1 - g _ { \lambda } ( 0 \mid x )$ . Lemma 1 in Appendix $\mathrm { A } . 5$ shows that, if $\mathbb { E } [ u \epsilon _ { 1 } ( X ) ] > 0 .$ , then there exists $\lambda > 0$ such that $L _ { Z } ( g _ { \lambda } , f _ { 1 } ) < L _ { Z } ( f _ { 1 } )$ . In other words, a router that captures additional weak-model errors can be used to construct an effective predictor.

Recalibration with $g _ { \lambda }$ (see Equation 2) reuses $f _ { 1 }$ and adds the budget-Q routing score $r ,$ the shared escalation rate $\rho ,$ and the correction strength λ. Therefore, the resulting model has at most $P _ { 1 } + Q + 2$ parameters. Applying Theorem 1 to $f _ { 1 }$ and $g _ { \lambda }$ bounds the excess predictability. Combining this upper bound with Lemma 1 bounds the additional weak errors captured.

Theorem 2 (Suboptimality and capacity bound weak-error capture beyond confidence). Under Assumption 1, every budget-Q routing score r and escalation rate $\rho$ induce decisions $B _ { r }$ and $B _ { f }$ whose difference $u = B _ { r } - B _ { f }$ satisfies

$$
\left| \mathbb { E } [ u \epsilon _ { 1 } ( X ) ] \le \sqrt { \operatorname* { m i n } \{ \rho , 1 - \rho \} \left[ \delta _ { P _ { 1 } } ( f _ { 1 } ) + L _ { Y } ^ { * } ( P _ { 1 } ) - L _ { Y } ^ { * } ( P _ { 1 } + Q + 2 ) \right] } \right|\tag{12}
$$

At fixed budgets and loss floor, reducing weak-model suboptimality tightens the bound on additional errors captured. By Equation $^ { 9 \prime }$ the same bound applies to cascade accuracy gains when $\mathbb { E } [ u \epsilon _ { 2 } ( X ) ] \geq \bar { 0 }$ (Appendix $\mathrm { A } . 7 )$ . Empirically, $\mathbb { E } [ u \epsilon _ { 2 } ( X ) ]$ is nonnegative on average over escalation rates for most evaluated models (Appendix C.5).

![](images/c6a128767eb3b36ea48f382e9ba8a3ab4e4f312a6e7cf19bba9da21222a09e7b.jpg)  
Figure 5: Finetuning reduces routing advantage in 53 of 59 model-task pairs. Each point compares one weak model before and after finetuning; both axes show released minus finetuned values. $\Delta \widehat { \delta }$ is the reduction in answer log loss. 100∆D is the reduction in cascading advantage. Points above zero have lower routing advantage after finetuning; counts report these pairs. Shapes identify the release type; lines are least-squares fits; r is Pearson correlation and $p _ { + }$ its one-sided p-value for positive correlation.

## 4.3 Experiments

The cascade experiments test a distinct question from predictability: whether predictors rank questions better for escalation and thereby improve cascade accuracy. For example, Platt calibration with positive slope can reduce correctness log loss relative to raw self-confidence while preserving rankings. At fixed escalation rates, it therefore leaves routing decisions unchanged and provides no cascade advantage.

## 4.3.1 Setup

Following Section $^ { 4 , }$ the weak model $f _ { 1 }$ answers each question first, and the routing decision determines whether to keep its answer or escalate to the strong model $f _ { 2 }$

Each model serves as $f _ { 1 } ,$ , both before and after finetuning. We use Qwen3.5-122B-A10B as $f _ { 2 } .$ . On WinoGrande and CommonsenseQA, we finetune $f _ { 2 }$ and select the checkpoint with the lowest development answer loss. On GSM8K and MMLU-Pro, we use its released version, which is already more accurate than all weak models.

We compare the selected $4 ^ { \mathrm { B } }$ predictor g from Section 3.4.1 with the weak model’s self-confidence as sources of routing scores, using error scores $g ( 0 | x )$ and $f _ { 1 , Z } ( 0 | x )$ . The thresholding rule from Section $4$ gives decisions $B _ { g , \rho }$ and $B _ { f , \rho }$ at the same escalation rate $\rho .$

The routing advantage D measures how much predictor routing improves cascade accuracy over self-confidence routing, averaged over all escalation rates:

$$
D = \int _ { 0 } ^ { 1 } \left[ \mathrm { E r r } ( f _ { \mathrm { c a s } , B _ { f , \rho } } ) - \mathrm { E r r } ( f _ { \mathrm { c a s } , B _ { g , \rho } } ) \right] d \rho .\tag{13}
$$

We report 100D as the average accuracy gain in percentage points.

## 4.3.2 Results

Cascade accuracy across escalation rates. For Llama-3.1- 8B on CommonsenseQA, finetuning improves weak-model accuracy and brings self-confidence routing closer to predictor routing (Figure 6). Appendix C.4 shows the curves for all four tasks.

More suboptimal models have larger routing advantages. Figure 4 shows that more suboptimal models have larger routing advantages. Pearson correlations between empirical suboptimality and routing advantage range from 0.66 to 0.91 across the four tasks. Appendix C.2 shows released and finetuned models separately. Compared with excess predictability, routing advantage has a weaker correlation with suboptimality on three of the four tasks. One possible reason is that finding more weak-model errors does not always

![](images/d7d74d565b7361eff03db87e5f8388623ef1e637629a20d1f3d46b4bcbc3ac2d.jpg)  
Figure 6: Llama-3.1-8B cascades on CommonsenseQA before and after finetuning: predictor (solid), confidence (dashed), and random (dotted) routing. Shading shows routing advantage D.

improve cascade accuracy: the strong model may also answer these questions incorrectly. Appendix C.5 separates the routing advantage into additional weak-model errors captured and additional strong-model errors encountered (see Equation 9). The captured errors alone are at least as strongly associated with suboptimality, with Pearson correlations of 0.80, 0.72, 0.91, and 0.86 in task order.

Finetuning reduces routing advantages. Similar to excess predictability, after finetuning, the routing advantage falls in 53 of the 59 retained model-task pairs (Figure 5). Larger reductions in suboptimality tend to accompany larger reductions in routing advantage, with Pearson correlations of 0.40 to 0.73. Averaged over the retained models, the routing advantage falls from 1.38 to 0.18 percentage points on WinoGrande, 1.09 to 0.18 on CommonsenseQA, 0.52 to 0.20 on GSM8K, and 0.94 to 0.07 on MMLU-Pro (Table 8).

## 5 Conclusion

The value of a separate performance predictor depends on the model it evaluates. We show that any excess predictability over self-confidence can be turned into an equal reduction in the model’s answer loss. The predictor’s advantage is therefore limited by the model’s suboptimality plus the capacity the predictor adds, and the same limit governs how many more weak-model errors a router can catch than confidence routing. In our experiments, more suboptimal models have larger prediction and routing advantages, and finetuning lets self-confidence catch up on both. Improving the model can thus reduce the need for a separate predictor.

When self-confidence is unavailable, our analysis suggests three sources of predictive value: recovering self-confidence, exploiting suboptimality, and adding capacity. Recovering self-confidence alone can make a separate predictor useful, even when little room remains to surpass it.

Limitations and further directions. The theory gives an upper bound, so it does not imply that every reduction in answer loss will reduce a predictor’s advantage. Our cascade bound controls additional weak-model errors captured; bounding the full routing advantage also requires a condition on strong-model errors. Future work should characterize the strong model’s contribution and how to predict the benefit of escalation without running it.

## Acknowledgement

We would like to thank Florian E. Dorner and Nastaran Okati for helpful discussions and/or feedback on draft versions of this work.

## References

Pranjal Aggarwal, Aman Madaan, Ankit Anand, Srividya Pranavi Potharaju, Swaroop Mishra, Pei Zhou, Aditya Gupta, Dheeraj Rajagopal, Karthik Kappaganthu, Yiming Yang, Shyam Upadhyay, Manaal Faruqui, and Mausam. AutoMix: Automatically mixing language models. In Advances in Neural Information Processing Systems, volume 37, pages 131000–131034, 2024. doi: 10.52202/079017-4164. URL https://proceedings.neurips.cc/paper<sup>\_</sup>files/paper/2024/ hash/ecda225cb187b40ea8edc1f46b03ffda-Abstract-Conference.html.

Jarosław Błasiok, Parikshit Gopalan, Lunjia Hu, Adam Tauman Kalai, and Preetum Nakkiran. Loss minimization yields multicalibration for large neural networks. In 15th Innovations in Theoretical Computer Science Conference (ITCS 2024), volume 287, pages 17:1–17:21, 2024. doi: 10.4230/LIPIcs.ITCS.2024.17. URL https://drops.dagstuhl.de/entities/document/10.4230/ LIPIcs.ITCS.2024.17.

Lingjiao Chen, Matei Zaharia, and James Zou. FrugalGPT: How to use large language models while reducing cost and improving performance. Transactions on Machine Learning Research, 2024. ISSN 2835-8856. URL https://openreview.net/forum?id=cSimKw5p6R.

C. K. Chow. On optimum recognition error and reject tradeoff. IEEE Transactions on Information Theory, 16(1):41–46, 1970. doi: 10.1109/TIT.1970.1054406.

Yu-Neng Chuang, Leisheng Yu, Guanchu Wang, Lizhe Zhang, Zirui Liu, Xuanting Cai, Yang Sui, Vladimir Braverman, and Xia Hu. Confident or seek stronger: Exploring uncertainty-based on-device LLM routing from benchmarking to generalization. arXiv preprint arXiv:2502.04428, 2025. URL https://arxiv.org/abs/2502.04428.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021. URL https://arxiv.org/abs/2110.14168.

André F. Cruz, Moritz Hardt, and Celestine Mendler-Dünner. Evaluating language models as risk scores. In Advances in Neural Information Processing Systems, volume 37, 2024. doi: 10. 52202/079017-3089. URL https://proceedings.neurips.cc/paper<sup>\_</sup>files/paper/2024/hash/ b0a4b3e384b4554e65a47ad1f6b0310a-Abstract-Datasets<sup>\_</sup>and<sup>\_</sup>Benchmarks<sup>\_</sup>Track.html.

Jasper Dekoninck, Maximilian Baader, and Martin Vechev. A unified approach to routing and cascading for LLMs. In Proceedings of the 42nd International Conference on Machine Learning, volume 267, pages 12987–13010, 2025. URL https://proceedings.mlr.press/v267/dekoninck25a. html.

Gianluca Detommaso, Martin Andres Bertran, Riccardo Fogliato, and Aaron Roth. Multicalibration for confidence scoring in LLMs. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 10624–10641. PMLR, 2024. URL https://proceedings.mlr.press/v235/detommaso24a.html.

Dujian Ding, Ankur Mallick, Chi Wang, Robert Sim, Subhabrata Mukherjee, Victor Rühle, Laks V. S. Lakshmanan, and Ahmed Hassan Awadallah. Hybrid LLM: Cost-efficient and quality-

aware query routing. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=02f3mUtqnM.

Falcon-LLM Team. The Falcon 3 family of open models, December 2024. URL https: //huggingface.co/blog/falcon3.

Leo Gao, Jonathan Tow, Baber Abbasi, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding, Jeffrey Hsu, Alain Le Noac’h, Haonan Li, Kyle McDonell, Niklas Muennighoff, Chris Ociepa, Jason Phang, Laria Reynolds, Hailey Schoelkopf, Aviya Skowron, Lintang Sutawika, Eric Tang, Anish Thite, Ben Wang, Kevin Wang, and Andy Zou. The language model evaluation harness, 2024. URL https://zenodo.org/records/12608602.

Yonatan Geifman and Ran El-Yaniv. Selective classification for deep neural networks. In Advances in Neural Information Processing Systems, volume 30, pages 4878–4887, 2017. URL https:// proceedings.neurips.cc/paper/2017/hash/4a8423d5e91fda00bb7e46540e2b0cf1-Abstract. html.

Gemma Team. Gemma 3 technical report. arXiv preprint arXiv:2503.19786, 2025. URL https: //arxiv.org/abs/2503.19786.

Parikshit Gopalan, Adam Tauman Kalai, Omer Reingold, Vatsal Sharan, and Udi Wieder. Omnipredictors. In 13th Innovations in Theoretical Computer Science Conference (ITCS 2022), volume 215, pages 79:1–79:21, 2022. doi: 10.4230/LIPIcs.ITCS.2022.79. URL https://drops.dagstuhl. de/entities/document/10.4230/LIPIcs.ITCS.2022.79.

Parikshit Gopalan, Lunjia Hu, Michael P. Kim, Omer Reingold, and Udi Wieder. Loss minimization through the lens of outcome indistinguishability. In 14th Innovations in Theoretical Computer Science Conference (ITCS 2023), volume 251, pages 60:1–60:20, 2023. doi: 10.4230/LIPIcs.ITCS.2023. 60. URL https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.ITCS.2023.60.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, et al. The Llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024. URL https://arxiv.org/abs/2407.21783.

Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q. Weinberger. On calibration of modern neural networks. In Proceedings of the 34th International Conference on Machine Learning, volume 70 of Proceedings of Machine Learning Research, pages 1321–1330. PMLR, 2017. URL https:// proceedings.mlr.press/v70/guo17a.html.

Neha Gupta, Harikrishna Narasimhan, Wittawat Jitkrittum, Ankit Singh Rawat, Aditya Krishna Menon, and Sanjiv Kumar. Language model cascades: Token-level uncertainty and beyond. In The Twelfth International Conference on Learning Representations, 2024. URL https://proceedings.iclr.cc/paper<sup>\_</sup>files/paper/2024/hash/ 11f5520daf9132775e8604e89f53925a-Abstract-Conference.html.

Ursula Hebert-Johnson, Michael Kim, Omer Reingold, and Guy Rothblum. Multicalibration: Calibration for the (computationally-identifiable) masses. In Proceedings of the 35th International Conference on Machine Learning, volume 80, pages 1939–1948, 2018. URL https://proceedings. mlr.press/v80/hebert-johnson18a.html.

Alejandro Hernández-Cano, Alexander Hägele, Allen Hao Huang, et al. Apertus: Democratizing open and compliant LLMs for global language environments. arXiv preprint arXiv:2509.14233, 2025. URL https://arxiv.org/abs/2509.14233.

José Hernández-Orallo, Wout Schellaert, and Fernando Martínez-Plumed. Training on the test set: Mapping the system-problem space in AI. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 36, pages 12256–12261, 2022. doi: 10.1609/aaai.v36i11.21487. URL https: //ojs.aaai.org/index.php/AAAI/article/view/21487.

Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, Tom Hennigan, Eric Noland, Katie Millican, George van den Driessche, Bogdan Damoc, Aurelia Guy, Simon Osindero, Karen Simonyan, Erich Elsen, Jack W. Rae, Oriol Vinyals, and Laurent Sifre. Training compute-optimal large language models. arXiv preprint arXiv:2203.15556, 2022. URL https://arxiv.org/abs/2203.15556.

Chi-Yang Hsu, Alexander Braylan, Yiheng Su, Matthew Lease, and Omar Alonso. PIE: Performance interval estimation for free-form generation tasks. arXiv preprint arXiv:2509.07309, 2025. doi: 10.48550/arXiv.2509.07309. URL https://arxiv.org/abs/2509.07309v2. Version 2, January 2026.

Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum?id=nZeVKeeFYf9.

Qitian Jason Hu, Jacob Bieker, Xiuyu Li, Nan Jiang, Benjamin Keigwin, Gaurav Ranganath, Kurt Keutzer, and Shriyash Kaustubh Upadhyay. RouterBench: A benchmark for multi-LLM routing system. arXiv preprint arXiv:2403.12031, 2024. doi: 10.48550/arXiv.2403.12031. URL https://arxiv.org/abs/2403.12031.

Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lélio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. Mistral 7B. arXiv preprint arXiv:2310.06825, 2023. URL https://arxiv.org/abs/2310.06825.

Wittawat Jitkrittum, Neha Gupta, Aditya K. Menon, Harikrishna Narasimhan, Ankit Rawat, and Sanjiv Kumar. When does confidence-based cascade deferral suffice? In Advances in Neural Information Processing Systems, volume 36, pages 9891–9906, 2023. doi: 10. 52202/075280-0431. URL https://proceedings.neurips.cc/paper<sup>\_</sup>files/paper/2023/hash/ 1f09e1ee5035a4c3fe38a5681cae5815-Abstract-Conference.html.

Saurav Kadavath, Tom Conerly, Amanda Askell, Tom Henighan, Dawn Drain, Ethan Perez, Nicholas Schiefer, Zac Hatfield-Dodds, Nova DasSarma, Eli Tran-Johnson, et al. Language models (mostly) know what they know. arXiv preprint arXiv:2207.05221, 2022.

Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361, 2020. URL https://arxiv.org/abs/2001.08361.

Sanyam Kapoor, Nate Gruver, Manley Roberts, Katherine Collins, Arka Pal, Umang Bhatt, Adrian Weller, Samuel Dooley, Micah Goldblum, and Andrew Gordon Wilson. Large language models must be taught to know what they don’t know. In Advances in Neural Information Processing Systems, volume 37, 2024. URL https://proceedings.neurips.cc/paper<sup>\_</sup>files/paper/2024/ hash/9c20f16b05f5e5e70fa07e2a4364b80e-Abstract-Conference.html.

Lorenz Kuhn, Yarin Gal, and Sebastian Farquhar. Semantic uncertainty: Linguistic invariances for uncertainty estimation in natural language generation. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=VD-AYtP0dve.

Meelis Kull and Peter Flach. Novel decompositions of proper scoring rules for classification: Score adjustment as precursor to calibration. In Machine Learning and Knowledge Discovery in Databases (ECML PKDD 2015), Lecture Notes in Computer Science, pages 68–85. Springer, 2015. doi: 10.1007/978-3-319-23528-8\_5.

Ananya Kumar, Percy Liang, and Tengyu Ma. Verified uncertainty calibration. In Advances in Neural Information Processing Systems, volume 32, 2019. URL https://proceedings.neurips. cc/paper/2019/hash/f8c0c968632845cd133308b1a494967f-Abstract.html.

Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum? id=v8L0pN6EOi.

Keming Lu, Hongyi Yuan, Runji Lin, Junyang Lin, Zheng Yuan, Chang Zhou, and Jingren Zhou. Routing to the expert: Efficient reward-guided ensemble of large language models. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 1964–1974. Association for Computational Linguistics, 2024. doi: 10.18653/v1/2024.naacl-long.109. URL https: //aclanthology.org/2024.naacl-long.109/.

Felipe Maia Polo, Lucas Weber, Leshem Choshen, Yuekai Sun, Gongjun Xu, and Mikhail Yurochkin. tinyBenchmarks: Evaluating LLMs with fewer examples. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 34303–34326. PMLR, 2024. URL https://proceedings.mlr.press/v235/maia-polo24a.html.

Sébastien Melo, Gaël Varoquaux, and Marine Le Morvan. Epistemic uncertainty quantification to improve decisions from black-box models. In The Fourteenth International Conference on Learning Representations, 2026. URL https://openreview.net/forum?id=JfiwaTxhI8.

Alireza Mohammadshahi, Arshad Rafiq Shaikh, and Majid Yazdani. Routoo: Learning to route to large language models effectively. arXiv preprint arXiv:2401.13979, 2024. doi: 10.48550/arXiv. 2401.13979. URL https://arxiv.org/abs/2401.13979v3. Version 3.

Iván Vicente Moreno Cencerrado, Arnau Padrés Masdemont, Anton Gonzalvez Hawthorne, David Demitri Africa, and Lorenzo Pacchiardi. No answer needed: Predicting LLM answer accuracy from question-only linear probes. In ICLR 2026 Workshop on Principled Design for Trustworthy AI, 2026. URL https://arxiv.org/abs/2509.10625.

Hussein Mozannar and David Sontag. Consistent estimators for learning to defer to an expert. In Proceedings of the 37th International Conference on Machine Learning, volume 119, pages 7076–7087, 2020. URL https://proceedings.mlr.press/v119/mozannar20b.html.

Harikrishna Narasimhan, Wittawat Jitkrittum, Aditya K. Menon, Ankit Rawat, and Sanjiv Kumar. Post-hoc estimators for learning to defer to an expert. In Advances in Neural Information Processing Systems, volume 35, pages 29292–29304, 2022. doi: 10. 52202/068431-2124. URL https://proceedings.neurips.cc/paper<sup>\_</sup>files/paper/2022/hash/ bc8f76d9caadd48f77025b1c889d2e2d-Abstract-Conference.html.

Isaac Ong, Amjad Almahairi, Vincent Wu, Wei-Lin Chiang, Tianhao Wu, Joseph E. Gonzalez, M. Waleed Kadous, and Ion Stoica. RouteLLM: Learning to route LLMs from preference data. In The Thirteenth International Conference on Learning Representations, 2025. URL https: //openreview.net/forum?id=8sSqNntaMr.

David Owen. How predictable is language model benchmark performance? arXiv preprint arXiv:2401.04757, 2024. URL https://arxiv.org/abs/2401.04757.

Lorenzo Pacchiardi, Lucy G. Cheke, and José Hernández-Orallo. 100 instances is all you need: Predicting the success of a new LLM on unseen data by testing on a few instances. In KDD 2024 Workshop on Evaluation and Trustworthiness of Generative AI Models, 2024. URL https://arxiv.org/abs/2409.03563. arXiv:2409.03563.

Alexandre Perez-Lebel, Marine Le Morvan, and Gaël Varoquaux. Beyond calibration: Estimating the grouping loss of modern neural networks. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=6w1k-IixnL8.

John C. Platt. Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. In Alexander J. Smola, Peter Bartlett, Bernhard Schölkopf, and Dale Schuurmans, editors, Advances in Large Margin Classifiers. MIT Press, 2000.

Qwen Team. Qwen3.5: Towards native multimodal agents, February 2026. URL https://qwen. ai/blog?id=qwen3.5.

Stephan Rabanser and Nicolas Papernot. What does it take to build a performant selective classifier? In Advances in Neural Information Processing Systems, volume 38, 2025. URL https: //openreview.net/forum?id=pcwI5cNFJK.

Yangjun Ruan, Chris J. Maddison, and Tatsunori Hashimoto. Observational scaling laws and the predictability of language model performance. In Advances in Neural Information Processing Systems, volume 37, 2024. URL https://proceedings.neurips.cc/paper<sup>\_</sup>files/paper/2024/ hash/1cded4f97cf5f01a284c574110b7e3b9-Abstract-Conference.html.

Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. WinoGrande: An adversarial winograd schema challenge at scale. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 34, pages 8732–8740, 2020. doi: 10.1609/aaai.v34i05.6399. URL https://ojs.aaai.org/index.php/AAAI/article/view/6399.

Dylan Sam, Marc Finzi, and Zico Kolter. Predicting the performance of black-box language models with follow-up queries. In Advances in Neural Information Processing Systems, volume 38, pages 176307–176342, 2025. doi: 10.52202/085713-5305. URL https://doi.org/10.52202/ 085713-5305.

Tal Shnitzer, Anthony Ou, Mírian Silva, Kate Soule, Yuekai Sun, Justin Solomon, Neil Thompson, and Mikhail Yurochkin. Large language model routing with benchmark datasets. In First Conference on Language Modeling, 2024. URL https://openreview.net/forum?id=Zb0ajZ7vAt.

Yuda Song, Hanlin Zhang, Carson Eisenach, Sham Kakade, Dean Foster, and Udaya Ghai. Mind the gap: Examining the self-improvement capabilities of large language models. In The Thirteenth International Conference on Learning Representations, 2025. URL https://arxiv.org/abs/2412. 02674.

Sharath Turuvekere Sreenivas, Saurav Muralidharan, Raviraj Joshi, Marcin Chochowski, Ameya Sunil Mahabaleshwarkar, Gerald Shen, Jiaqi Zeng, Zijia Chen, Yoshi Suhara, Shizhe Diao, Chenhan Yu, Wei-Chun Chen, Hayley Ross, Oluwatobi Olabiyi, Ashwath Aithal, Oleksii Kuchaiev, Daniel Korzekwa, Pavlo Molchanov, Mostofa Patwary, Mohammad Shoeybi, Jan Kautz, and Bryan Catanzaro. LLM pruning and distillation in practice: The Minitron approach. arXiv preprint arXiv:2408.11796, 2024. URL https://arxiv.org/abs/2408.11796.

Alon Talmor, Jonathan Herzig, Nicholas Lourie, and Jonathan Berant. CommonsenseQA: A question answering challenge targeting commonsense knowledge. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 4149–4158. Association for Computational Linguistics, 2019. doi: 10.18653/v1/N19-1421. URL https://aclanthology.org/N19-1421/.

Team Olmo, Allyson Ettinger, Amanda Bertsch, Bailey Kuehl, et al. Olmo 3. arXiv preprint arXiv:2512.13961, 2025. URL https://arxiv.org/abs/2512.13961.

Katherine Tian, Eric Mitchell, Allan Zhou, Archit Sharma, Rafael Rafailov, Huaxiu Yao, Chelsea Finn, and Christopher Manning. Just ask for calibration: Strategies for eliciting calibrated confidence scores from language models fine-tuned with human feedback. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 5433–5442. Association for Computational Linguistics, 2023. doi: 10.18653/v1/2023.emnlp-main.330. URL https://aclanthology.org/2023.emnlp-main.330/.

Neeraj Varshney and Chitta Baral. Model cascading: Towards jointly improving efficiency and accuracy of NLP systems. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 11007–11021. Association for Computational Linguistics, 2022. doi: 10.18653/v1/2022.emnlp-main.756. URL https://aclanthology.org/2022.emnlp-main.756/.

Congchao Wang, Sean Augenstein, Keith Rush, Wittawat Jitkrittum, Harikrishna Narasimhan, Ankit Singh Rawat, Aditya Krishna Menon, and Alec Go. Cascade-aware training of language models. arXiv preprint arXiv:2406.00060, 2024a. URL https://arxiv.org/abs/2406.00060.

Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language

models. In The Eleventh International Conference on Learning Representations, 2023. URL https: //openreview.net/forum?id=1PL1NIMMrw.

Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni, Abhranil Chandra, Shiguang Guo, Weiming Ren, Aaran Arulraj, Xuan He, Ziyan Jiang, Tianle Li, Max Ku, Kai Wang, Alex Zhuang, Rongqi Fan, Xiang Yue, and Wenhu Chen. MMLU-Pro: A more robust and challenging multi-task language understanding benchmark. In Advances in Neural Information Processing Systems, volume 37, pages 95266–95290, 2024b. doi: 10.52202/079017-3018. URL https://papers.nips.cc/paper<sup>\_</sup>files/paper/2024/hash/ ad236edc564f3e3156e1b2feafb99a24-Abstract-Datasets<sup>\_</sup>and<sup>\_</sup>Benchmarks<sup>\_</sup>Track.html. Datasets and Benchmarks Track.

Hanqi Xiao, Vaidehi Patil, Hyunji Lee, Elias Stengel-Eskin, and Mohit Bansal. Generalized correctness models: Learning calibrated and model-agnostic correctness predictors from historical patterns. In Proceedings of the 43rd International Conference on Machine Learning, 2026. URL https://arxiv.org/abs/2509.24988.

Miao Xiong, Zhiyuan Hu, Xinyang Lu, Yifei Li, Jie Fu, Junxian He, and Bryan Hooi. Can LLMs express their uncertainty? an empirical evaluation of confidence elicitation in LLMs. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/ forum?id=gjeQKFxFpZ.

Yuekun Yao and Alexander Koller. Predicting generalization performance with correctness discriminators. In Findings of the Association for Computational Linguistics: EMNLP 2024, pages 11725– 11739. Association for Computational Linguistics, 2024. doi: 10.18653/v1/2024.findings-emnlp. 686. URL https://aclanthology.org/2024.findings-emnlp.686/.

Murong Yue, Jie Zhao, Min Zhang, Liang Du, and Ziyu Yao. Large language model cascades with mixture of thought representations for cost-efficient reasoning. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=6okaSfANzh.

Guanhua Zhang, Florian E. Dorner, and Moritz Hardt. How benchmark prediction from fewer data misses the mark. In Advances in Neural Information Processing Systems, volume 38, 2025a. doi: 10.52202/085713-4528. URL https://proceedings.nips.cc/paper<sup>\_</sup>files/paper/2025/hash/ c57bacb5ea621582235c54ce2a83136d-Abstract-Conference.html.

Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang, Huan Lin, Baosong Yang, Pengjun Xie, An Yang, Dayiheng Liu, Junyang Lin, Fei Huang, and Jingren Zhou. Qwen3 Embedding: Advancing text embedding and reranking through foundation models. arXiv preprint arXiv:2506.05176, 2025b. URL https://arxiv.org/abs/2506.05176.

Lexin Zhou, Fernando Martínez-Plumed, José Hernández-Orallo, Cèsar Ferri, and Wout Schellaert. Reject before you run: Small assessors anticipate big language models. In Proceedings of the Workshop on AI Evaluation Beyond Metrics (EBeM 2022), volume 3169 of CEUR Workshop Proceedings. CEUR-WS.org, 2022. URL https://ceur-ws.org/Vol-3169/paper4.pdf.

## A Proofs and Additional Results

## A.1 Proofs for Section 3

Proof of Proposition 1. On a correct prediction, the model assigns the gold answer probability $f _ { Z } ( 1 | X ) .$ ; on an incorrect prediction, it assigns $f _ { Z } ( 0 \mid X ) f _ { \mathrm { A l t } } ( Y \mid X )$ . Taking negative logarithms gives

$$
- \log f ( Y \mid X ) = - \log f _ { Z } ( Z \mid X ) + \Im \left\{ Z = 0 \right\} \big ( - \log f _ { \mathrm { A l t } } ( Y \mid X ) \big ) .\tag{14}
$$

Taking expectations yields

$$
{ \cal L } _ { Y } ( f ) = { \cal L } _ { Z } ( f ) + { \mathbb { E } } \left[ \mathbb { 1 } \left\{ Z = 0 \right\} \big ( - \log f _ { \mathrm { A l t } } ( Y \mid X ) \big ) \right] .
$$

Recalibration replaces $f _ { Z }$ by $g$ and keeps the same alternative distribution, so Equation 2 gives

$$
\begin{array} { r l } & { L _ { Y } ( \mathcal { C } _ { g } ( f ) ) = \operatorname { \mathbb { E } } [ - \log \mathcal { C } _ { g } ( f ) ( Y \mid X ) ] } \\ & { \qquad = L _ { Z } ( g , f ) + \operatorname { \mathbb { E } } \left[ \mathbb { 1 } \left\{ Z = 0 \right\} ( - \log f _ { \mathrm { A l t } } ( Y \mid X ) ) \right] . } \end{array}\tag{15}
$$

Subtracting the two expected-loss decompositions cancels the common alternative-loss term and proves the result. □

Proof of Theorem 1. The recalibrated model has budget at most $P + Q ,$ so the scaling floor and Proposition 1 give

$$
L _ { Y } ^ { * } ( P + Q ) \leq L _ { Y } ( { \mathcal { C } } _ { g } ( f ) ) = L _ { Y } ( f ) - [ L _ { Z } ( f ) - L _ { Z } ( g , f ) ] .\tag{16}
$$

Substituting Equation 5 and rearranging proves Equation 6.

## A.2 No Improvement Without a Better Component

This section states and proves the result summarized after Proposition 1. With an effective predictor, recalibration improves the correctness probabilities while preserving the alternative distribution. To see what a further improvement would require, consider any model $h ( y \mid x )$ and split its probabilities according to whether $f ^ { \prime } \mathbf { s }$ answer is correct:

$$
h _ { Z } ( 1 \mid x ) = h ( { \hat { f } } ( x ) \mid x ) , \qquad h _ { Z } ( 0 \mid x ) = 1 - h _ { Z } ( 1 \mid x ) .\tag{17}
$$

Every model h can be viewed as answering two questions: (i) whether $f ^ { \prime } { \bf s }$ prediction is correct and (ii) if it is wrong, which alternative answer is correct.

The quantity $h _ { Z } ( 1 | x )$ answers the first question. Once $\hat { f } ( x )$ has been ruled out, the distribution $h ( y \mid x ) / h _ { Z } ( 0 \mid x )$ answers the second. Define the conditional loss over the alternatives:

$$
L _ { Y | Z = 0 } ( h ) : = \operatorname { \mathbb { E } } \left[ - \log \frac { h ( Y \mid X ) } { h _ { Z } ( 0 \mid X ) } \ \Big | \ Z = 0 \right] .\tag{18}
$$

This is the log loss for the second question, evaluated only when f is wrong. In binary classification there is only one alternative answer, so $L _ { Y | Z = 0 } ( h ) = 0$ for every h. With more than two answers, this term measures how well h distinguishes among the alternatives.

Proposition 2 (No improvement without a better component). For every model h,

$$
L _ { Y } ( h ) = L _ { Z } ( h _ { Z } , f ) + \operatorname* { P r } ( Z = 0 ) L _ { Y | Z = 0 } ( h ) ,\tag{19}
$$

$$
L _ { Y } ( \mathcal { C } _ { g } ( f ) ) = L _ { Z } ( g , f ) + \operatorname* { P r } ( Z = 0 ) L _ { Y | Z = 0 } ( f ) .\tag{20}
$$

If the four component losses are finite and $L _ { Y } ( h ) < L _ { Y } ( \mathcal { C } _ { g } ( f ) )$ , then

$$
L _ { Z } ( h _ { Z } , f ) < L _ { Z } ( g , f ) \qquad o r \qquad L _ { Y | Z = 0 } ( h ) < L _ { Y | Z = 0 } ( f ) .\tag{21}
$$

Proof of Proposition 2. Factor h into its probability that f is correct and its distribution over alternative answers, then take expected negative logarithms to obtain Equation 19. Recalibration changes only the correctness probabilities, giving Equation 20. If neither inequality in Equation 21 held, comparing the two decompositions would give $L _ { Y } ( h ) \ge L _ { Y } ( \mathcal { C } _ { g } ( f ) )$ , a contradiction. □

## A.3 Small Predictors Under Diminishing Returns

When the predictor is much smaller than the model, its excess predictability is bounded by the model’s suboptimality plus a small capacity correction. The following corollary makes this precise under diminishing returns, expressed by a convex loss floor.

Corollary 1 (Small predictors have little advantage beyond model suboptimality). Under Assumption 1, suppose $L _ { Y } ^ { * }$ is convex. For $P > 0$ and $Q \geq 0 .$ , every $f \in \mathcal { F } _ { P }$ with $L _ { Y } ( f ) < \infty$ and every $g \in { \mathcal { G } } _ { Q }$ satisfy

$$
L _ { Z } ( f ) - L _ { Z } ( g , f ) \leq \frac { 2 Q } { P } [ L _ { Y } ^ { * } ( P / 2 ) - L _ { Y } ^ { * } ( P ) ] + \delta _ { P } ( f ) .\tag{22}
$$

The capacity correction is $2 Q / P$ times the decrease in the loss floor from $P / 2$ to P. When $Q \ll P ,$ , this correction is small, so the excess predictability is bounded essentially by $\delta _ { P } ( f )$ Thus a nearly optimal model leaves little room for a much smaller predictor to outperform its self-confidence.

Proof of Corollary 1. By convexity, the loss reduction per parameter over $\left[ P , P + Q \right]$ is no greater than over $\left[ P / 2 , P \right]$ , giving

$$
L _ { Y } ^ { * } ( P ) - L _ { Y } ^ { * } ( P + Q ) \leq \frac { 2 Q } { P } [ L _ { Y } ^ { * } ( P / 2 ) - L _ { Y } ^ { * } ( P ) ] .
$$

Substituting this bound into Equation 6 proves the result.

## A.4 Suboptimality Does Not Determine Excess Predictability

Proposition 1 shows that a reduction in correctness loss gives an equal reduction in answer loss. Theorem 1 therefore bounds excess predictability by the answer-loss reduction allowed by capacity gain and suboptimality.

However, lowering answer loss does not necessarily lower correctness loss. We can decompose answer loss as

$$
L _ { Y } ( f ) = L _ { Z } ( f ) + \operatorname* { P r } ( Z = 0 ) L _ { Y \mid Z = 0 } ( f ) ,\tag{23}
$$

where $L _ { Y | Z = 0 } ( f ) : = \mathbb { E } [ - \log f _ { \mathrm { A l t } } ( Y | X ) | Z = 0 ]$ is the loss over alternative answers when the model’s chosen answer is wrong. A model can lower this alternative loss while keeping its chosen answer and self-confidence unchanged. At a fixed parameter budget, this reduces suboptimality without changing the correctness-prediction problem.

Proposition 3 (Lower suboptimality with unchanged maximum excess predictability). Let $f \in \mathcal { F } _ { P }$ have finite answer loss and $\operatorname* { P r } ( Z = \mathrm { \bar { 0 } } ) > 0$ . For $\lambda \in ( 0 , 1 )$ , let $h \in \mathcal { F } _ { P }$ keep $f ^ { \prime } s$ point prediction and self-confidence and mix its alternatives with the uniform distribution,

$$
h _ { \mathrm { A l t } } ( y \mid x ) = ( 1 - \lambda ) f _ { \mathrm { A l t } } ( y \mid x ) + \frac { \lambda } { K - 1 } .\tag{24}
$$

Then $f$ and h have the same maximum excess predictability at every predictor budget,

$$
\operatorname* { s u p } _ { g \in { \mathcal { G } } _ { Q } } [ L _ { Z } ( h ) - L _ { Z } ( g , h ) ] = \operatorname* { s u p } _ { g \in { \mathcal { G } } _ { Q } } [ L _ { Z } ( f ) - L _ { Z } ( g , f ) ] ,\tag{25}
$$

while $\delta _ { P } ( h ) < \delta _ { P } ( f ) \mathrm { ~ } i f L _ { Y | Z = 0 } ( f ) > \log ( K - 1 )$

At fixed parameter budgets, the bound in Theorem 1 decreases with suboptimality, but the maximum excess predictability need not decrease with it. Models with the same maximum excess predictability can therefore differ in suboptimality by their alternative loss alone.

With two answers, alternative loss is zero, so this source of separation is absent. As K grows, the alternative distribution has more degrees of freedom, allowing changes in answer loss that correctness prediction does not capture.

Proof of Proposition 3. Mixing with a fixed constant is a fixed arithmetic operation, so $h \in \mathcal { F } _ { P } ;$ the common point prediction gives the same $Z ,$ and the common self-confidence gives $L _ { Z } ( h ) = L _ { Z } ( f )$ For $\lambda < 1 ,$ , Equation 24 is invertible, so the outputs of $f$ and h on x determine each other through fixed arithmetic. Any $g \in { \mathcal { G } } _ { Q }$ reading $f ^ { \prime } \mathbf { s }$ output therefore has a counterpart in $\mathcal { G } _ { Q }$ reading h’s output with the same correctness loss, and conversely, which proves Equation 25. Applying Equation $^ { 2 3 }$ to $f$ and h and subtracting gives

$$
\delta _ { P } ( f ) - \delta _ { P } ( h ) = \operatorname* { P r } ( Z = 0 ) [ L _ { Y | Z = 0 } ( f ) - L _ { Y | Z = 0 } ( h ) ] .\tag{26}
$$

Since − log is convex, $L _ { Y | Z = 0 } ( h ) \le ( 1 - \lambda ) L _ { Y | Z = 0 } ( f ) + \lambda \log ( K - 1 )$ , which is below $L _ { Y | Z = 0 } ( f )$ when $L _ { Y | Z = 0 } ( f ) > \log ( K \dot { - } 1 )$ . □

## A.5 Routing Induces Performance Prediction

The following lemma quantifies the prediction gain supplied by a router that captures additional weak-model errors. It provides the supporting result for the proof of Theorem 2 in Appendix A.6.

We use the notation of Section $^ { 4 \prime }$ suppressing X in $B _ { r } , B _ { f } ,$ , and u. Independent tie-breaking draws are treated as auxiliary inputs, suppressed in the notation and adding no parameters; all expectations include this randomness. For each model, correctness and the actual error probability are

$$
Z _ { i } = \mathbb { 1 } \{ Y = \hat { f } _ { i } ( X ) \} , \quad \epsilon _ { i } ( x ) = \operatorname* { P r } ( Z _ { i } = 0 | X = x ) , \quad i = 1 , 2 .\tag{27}
$$

At escalation rate $\rho ,$ the routing decision satisfies

$$
B _ { r , \rho } ( x ) \in \{ 0 , 1 \} , \qquad { \mathbb E } \big [ B _ { r , \rho } ( X ) \big ] = \rho .\tag{28}
$$

The confidence decision also satisfies $\mathbb { E } [ B _ { f } ] = \rho ,$ , and we write $\boldsymbol { u } = \boldsymbol { B } _ { r } - \boldsymbol { B } _ { f } ,$

At a fixed escalation rate, $B _ { f }$ escalates the lowest-confidence ρ fraction of queries and therefore maximizes the predicted weak errors captured, $\mathbb { E } [ B _ { f } f _ { 1 , Z } ( 0 \mid X ) ]$ ]. Hence

$$
\mathbb { E } \big [ u f _ { 1 , Z } ( 0 | X ) \big ] \leq 0 .\tag{29}
$$

$\mathrm { A s }$ in Equation 10, additional weak-error capture is therefore bounded by the confidence residual $\mathbb { E } [ u ( \epsilon _ { 1 } ( X ) - f _ { 1 , Z } ( 0 \mid X ) ) ]$ ]. To convert this residual into a prediction gain, use the logit correction $g _ { \lambda }$ defined in Equation 11. For $\lambda > 0$ , this raises predicted error where $B _ { r } = 1$ and $B _ { f } = 0$ , and lowers it where $B _ { r } = 0$ and $B _ { f } = 1 . \mathrm { A t } \lambda = 0 ,$ , it recovers confidence.

Lemma 1 (Routing induces performance prediction beyond confidence). For any routing score r at a fixed rate $\rho$ with $\mathbb { E } \bar { [ } u ^ { 2 } ] > 0$ , the predictors in Equation 11 satisfy

$$
\operatorname* { s u p } _ { \lambda \in \mathbb { R } } [ L _ { Z } ( f _ { 1 } ) - L _ { Z } ( g _ { \lambda } , f _ { 1 } ) ] \geq { \frac { 2 ( \mathbb { E } [ u ( \epsilon _ { 1 } ( X ) - f _ { 1 , Z } ( 0 \mid X ) ) ] ) ^ { 2 } } { \mathbb { E } [ u ^ { 2 } ] } } .\tag{30}
$$

$I f \mathbb { E } [ u \epsilon _ { 1 } ( X ) ] > 0 .$ , then there exists $\lambda > 0$ such that $L _ { Z } ( g _ { \lambda } , f _ { 1 } ) < L _ { Z } ( f _ { 1 } ) . \ I f \mathbb { E } [ u ^ { 2 } ] = 0 $ , then $g _ { \lambda } = f _ { 1 , Z }$ almost surely for every λ.

Proof of Lemma 1. If $\mathbb { E } [ u ^ { 2 } ] = 0 .$ , then $u = 0$ almost surely, so Equation 11 gives $g _ { \lambda } = f _ { 1 , Z }$ for every λ. Henceforth assume $\mathbb { E } [ u ^ { 2 } ] > 0$ . We first bound the loss after shifting the confidence logit, then choose the shift that gives the largest guaranteed improvement.

Step 1: bound the change in loss on each query. Fix a query x and a realization of the tie-breaking randomness, which is independent of the answer given x. Write $a ( x ) = \log \mathrm { i t } f _ { 1 , Z } ( 0 \mid x )$ . For a predicted error logit s, let $\bar { \sigma ( } s ) = 1 / ( 1 + e ^ { - s } )$ be the corresponding error probability. Since the weak model is wrong with probability $\epsilon _ { 1 } ( x )$ , the conditional expected binary log loss is

$$
\begin{array} { c } { { \ell _ { x } ( s ) = - \epsilon _ { 1 } ( x ) \log \sigma ( s ) - [ 1 - \epsilon _ { 1 } ( x ) ] \log [ 1 - \sigma ( s ) ] } } \\ { { = \log ( 1 + e ^ { s } ) - \epsilon _ { 1 } ( x ) s . } } \end{array}
$$

Differentiating with respect to s gives

$$
\ell _ { x } ^ { \prime } ( s ) = \sigma ( s ) - \epsilon _ { 1 } ( x ) , \qquad \ell _ { x } ^ { \prime \prime } ( s ) = \sigma ( s ) [ 1 - \sigma ( s ) ] \leq { \frac { 1 } { 4 } } .
$$

Taylor’s theorem therefore bounds the loss at any shift $h \in \mathbb { R }$ by

$$
\ell _ { x } ( a ( x ) + h ) \leq \ell _ { x } ( a ( x ) ) + [ f _ { 1 , Z } ( 0 \mid x ) - \epsilon _ { 1 } ( x ) ] h + \frac { h ^ { 2 } } { 8 } .
$$

The factor $1 / 8$ is the Taylor factor $1 / 2$ times the curvature bound $1 / 4$ . Substituting $h = \lambda u ( x )$ and rearranging yields

$$
\ell _ { x } ( a ( x ) ) - \ell _ { x } ( a ( x ) + \lambda u ( x ) ) \geq \lambda u ( x ) [ \epsilon _ { 1 } ( x ) - f _ { 1 , Z } ( 0 \mid x ) ] - \frac { \lambda ^ { 2 } u ( x ) ^ { 2 } } { 8 } .
$$

Step 2: average over queries and tie-breaking. The two losses on the left are the conditional losses of $f _ { 1 , Z }$ and $g _ { \lambda } ,$ respectively. Taking expectations over both queries and tie-breaking thus gives

$$
L _ { Z } ( f _ { 1 } ) - L _ { Z } ( g _ { \lambda } , f _ { 1 } ) \ge \lambda \mathbb { E } [ u ( \epsilon _ { 1 } ( X ) - f _ { 1 , Z } ( 0 | X ) ) ] - \frac { \lambda ^ { 2 } } { 8 } \mathbb { E } [ u ^ { 2 } ] .\tag{31}
$$

These expected losses are finite: $L _ { Z } ( f _ { 1 } ) \le L _ { Y } ( f _ { 1 } ) < \infty ,$ , and the displayed upper bound, together with $| u | \overset { \vartriangle } { \le { } 1 }$ , bounds the corrected loss for every finite $\lambda$

Step 3: choose the correction strength. For this optimization, abbreviate the confidence residual and the squared routing difference as

$$
A = \mathbb { E } [ u ( \epsilon _ { 1 } ( X ) - f _ { 1 , Z } ( 0 | X ) ) ] , \qquad V = \mathbb { E } [ u ^ { 2 } ] > 0 .
$$

Completing the square in the right-hand side of Equation $3 \mathbf { I }$ gives

$$
\lambda A - { \frac { \lambda ^ { 2 } V } { 8 } } = { \frac { 2 A ^ { 2 } } { V } } - { \frac { V } { 8 } } \left( \lambda - { \frac { 4 A } { V } } \right) ^ { 2 } .
$$

Thus $\lambda _ { * } = 4 A / V$ guarantees excess predictability of at least $2 A ^ { 2 } / V .$ . The supremum over all $\lambda$ is at least the excess predictability at $\lambda _ { * } ,$ proving Equation $3 0$ . Finally, if $B _ { r }$ captures more weak-model errors, then Equation 10 gives $A \geq \mathbb { E } [ u \epsilon _ { 1 } ( X ) ] > 0$ . Consequently $\lambda _ { * } > 0$ and $2 A ^ { 2 } / V > 0$ , so this correction is an effective performance predictor. □

## A.6 Proofs for Section 4

Using the notation and supporting lemma from Appendix $\mathrm { A } . 5 ,$ we prove Theorem 2 by combining the prediction gain with the loss-floor bound from Theorem 1.

Proof of Theorem 2. Recalibration with $g _ { \lambda }$ reuses $f _ { 1 }$ and adds the budget-Q routing score $r ,$ the shared escalation rate $\rho ,$ and the correction strength λ. The resulting model has at most $P _ { 1 } + Q + 2$ parameters and belongs to $\mathcal { F } _ { P _ { 1 } + Q + 2 }$ . Exact transfer and the loss floor therefore bound the excess predictability for every $\lambda ,$ as in Theorem 1:

$$
L _ { Z } ( f _ { 1 } ) - L _ { Z } ( g _ { \lambda } , f _ { 1 } ) \le \delta _ { P _ { 1 } } ( f _ { 1 } ) + L _ { Y } ^ { * } ( P _ { 1 } ) - L _ { Y } ^ { * } ( P _ { 1 } + Q + 2 ) .\tag{32}
$$

Write the upper bound on the excess predictability as

$$
\Delta = \delta _ { P _ { 1 } } ( f _ { 1 } ) + L _ { Y } ^ { * } ( P _ { 1 } ) - L _ { Y } ^ { * } ( P _ { 1 } + Q + 2 ) .
$$

This quantity is nonnegative because suboptimality is nonnegative and the loss floor is nonincreasing in the parameter budget. If $\mathbb { E } [ u ^ { 2 } ] = 0$ , the routing decisions agree almost surely and the left side is zero. Now assume $\mathbb { E } [ u ^ { 2 } ] > 0$

Step 1: bound the confidence residual using the excess predictability. Lemma 1 gives a lower bound on the maximum excess predictability obtained by correcting confidence along u. Equation $3 2$ bounds the excess predictability of every such correction by $\Delta ,$ so it also bounds their supremum. Together, these bounds give

$$
\frac { 2 ( \mathbb { E } [ u ( \epsilon _ { 1 } ( X ) - f _ { 1 , Z } ( 0 \mid X ) ) ] ) ^ { 2 } } { \mathbb { E } [ u ^ { 2 } ] } \leq \operatorname* { s u p } _ { \lambda \in \mathbb { R } } [ L _ { Z } ( f _ { 1 } ) - L _ { Z } ( g _ { \lambda } , f _ { 1 } ) ] \leq \Delta .
$$

Multiplying by $\mathbb { E } \big [ u ^ { 2 } \big ] / 2$ and taking nonnegative square roots yields

$$
\vert \mathbb { E } [ u ( \epsilon _ { 1 } ( X ) - f _ { 1 , Z } ( 0 \vert X ) ) ] \vert \le \sqrt { \frac { \mathbb { E } [ u ^ { 2 } ] } { 2 } \Delta } .
$$

By Equation $^ { \mathrm { 1 O , } }$ this also bounds the additional weak errors captured, $\mathbb { E } [ u \epsilon _ { 1 } ( X ) ]$

Step 2: bound the difference between routing decisions using the escalation rate. Since $B _ { r } , B _ { f } \in \{ 0 , 1 \}$ we have $u ^ { 2 } = | B _ { r } - B _ { f } | = \mathbb { 1 } \{ B _ { r } \neq B _ { f } \}$ . The absolute difference is at most both $B _ { r } + B _ { f }$ and $( 1 - B _ { r } ) + ( 1 - B _ { f } )$ , so

$$
u ^ { 2 } \leq B _ { r } + B _ { f } , \qquad u ^ { 2 } \leq ( 1 - B _ { r } ) + ( 1 - B _ { f } ) .
$$

Taking expectations and using $\mathbb { E } [ B _ { r } ] = \mathbb { E } [ B _ { f } ] = \rho$ gives both $\mathbb { E } [ u ^ { 2 } ] \le 2 \rho$ and $\mathbb { E } [ u ^ { 2 } ] \leq 2 ( 1 - \rho )$ Combining the two steps gives

$$
\mathbb { E } [ u \epsilon _ { 1 } ( X ) ] \leq \sqrt { \frac { \mathbb { E } [ u ^ { 2 } ] } { 2 } \Delta } \leq \sqrt { \operatorname* { m i n } \{ \rho , 1 - \rho \} \Delta } .
$$

Substituting the definition of $\Delta$ proves Equation 12.

## A.7 Cascade Advantage over Confidence Routing

To pass from weak-error capture to cascade performance, we must also account for strong-model errors on escalated queries. The cascade prediction and error rate are

$$
\hat { f } _ { \mathrm { c a s } , B _ { r } } ( x ) = ( 1 - B _ { r } ( x ) ) \hat { f } _ { 1 } ( x ) + B _ { r } ( x ) \hat { f } _ { 2 } ( x ) ,\tag{33}
$$

$$
\operatorname { E r r } ( f _ { \mathrm { c a s } , B _ { r } } ) = \operatorname { E r r } ( f _ { 1 } ) - \mathbb { E } [ B _ { r } ( \epsilon _ { 1 } ( X ) - \epsilon _ { 2 } ( X ) ) ] .\tag{34}
$$

Subtracting this identity for confidence routing gives Equation 9. If the learned router encounters at least as many strong-model errors as confidence routing, its cascade advantage is at most its additional weak-error capture.

Corollary 2 (Cascade advantage over confidence routing). Under the conditions of Theorem 2, if $\mathbb { E } [ u \epsilon _ { 2 } ( X ) ] \geq 0$ , then

$$
\mathrm { E r r } ( f _ { \mathrm { c a s } , B _ { f } } ) - \mathrm { E r r } ( f _ { \mathrm { c a s } , B _ { r } } ) \le \sqrt { \operatorname* { m i n } \{ \rho , 1 - \rho \} } \big [ \delta _ { P _ { 1 } } ( f _ { 1 } ) + L _ { Y } ^ { \ast } ( P _ { 1 } ) - L _ { Y } ^ { \ast } ( P _ { 1 } + Q + 2 ) \big ] .\tag{35}
$$

Proof of Corollary 2. By Equation $^ { 9 \prime }$ the cascade accuracy gain equals ${ \mathbb E } \big [ u \epsilon _ { 1 } ( X ) \big ] - { \mathbb E } \big [ u \epsilon _ { 2 } ( X ) \big ] \leq$ $\mathbb { E } [ u \epsilon _ { 1 } ( X ) ]$ , which Theorem 2 bounds. □

In a longer cascade, the analysis applies at each stage to its incoming queries, with $f _ { 2 }$ representing a fixed downstream policy.

## B Additional Experimental Setup

## B.1 Models and Data Splits

Table 1 lists the 16 releases evaluated before and after task-specific finetuning. A model pair refers to a release and its selected finetuned checkpoint on one task.
<table><tr><td>Base release</td><td>Instruction-tuned release</td></tr><tr><td> $\overline { { \mathrm { Q w e N } 3 . 5 ^ { - } 9 \mathrm { B } \mathrm { - } \mathrm { B } \mathrm { A } { \mathrm { S E } } } }$ </td><td> $\overline { { \mathrm { Q w e N } 3 { \cdot } 5 { \cdot } 9 \mathrm { B } } }$ </td></tr><tr><td> $\mathbf { G E M M A - } 3 \mathbf { - } \mathbf { 1 } 2 \mathbf { B - } \mathbf { P T }$ </td><td> $\mathbf { G E M M A - } 3 ^ { - 1 2 } \mathbf { B - } \mathbf { I T }$ </td></tr><tr><td> $\operatorname { L L A M A - } 3 . 1 - 8 \mathrm { B }$ </td><td> $\operatorname { L L A M A - } 3 . 1 \mathbf { - } 8 \mathbf { B - } \operatorname { I N S T R U C T }$ </td></tr><tr><td> $\mathbf { A P E R T U S - } 8 \mathbf { B } \mathbf { - } 2 5 0 9$ </td><td> $\mathrm { A P E R T U S - } \mathrm { B B - I N S T R U C T } { - } 2 5 0 9$ </td></tr><tr><td> $\mathrm { O L M o - } 3 ^ { \mathbf { - } \mathbf { 1 } _ { 0 2 } } 5 ^ { \mathbf { - } } 7 ^ { \mathbf { B } }$ </td><td> $\mathrm { O L M o } { - } _ { 3 } { - } _ { 7 } \mathrm { B } { - } \mathrm { I N s T R U C T }$ </td></tr><tr><td> $\mathrm { F A L C O N } 3 { \cdot } 7 \mathrm { B } { \cdot } \mathrm { B } \mathrm { A } { \cdot } \mathrm { s } \mathrm { E }$ </td><td> $\mathbf { F a L C O N 3 - 7 B - I N S T R U C T }$ </td></tr><tr><td> $\mathrm { M I S T R A L - } 7 \mathrm { B } \mathrm { - } \mathrm { v o } . 3$ </td><td> $\mathrm { M I S T R A L - 7 B  – I N S T R U C T – V O . } 3$ </td></tr><tr><td> $\mathrm { M I S T R A L \mathrm { - } N E M o \mathrm { - } M a N I T R O N \mathrm { - } 8 B \mathrm { - } B a S E }$ </td><td> $\mathrm { M I S T R A L - N E M o - M I N I T R O N - } 8 \mathrm { B - I N S T R U C T }$ </td></tr></table>

Table 1: Base and instruction-tuned releases from eight model families, giving 16 models per task: Qwen3.5 (Qwen Team, 2026), Gemma 3 (Gemma Team, 2025), Llama 3.1 (Grattafiori et al., 2024), Apertus (Hernández-Cano et al., 2025), OLMo 3 (Team Olmo et al., 2025), Falcon3 (Falcon-LLM Team, 2024), Mistral 7B (Jiang et al., 2023), and Mistral-NeMo-Minitron (Sreenivas et al., 2024).

Table 2 gives the split sizes. For WinoGrande, CommonsenseQA, and GSM8K, we hold out development questions from the original training set and reserve five remaining questions as demonstrations. The first two tasks use their original validation sets for testing; GSM8K uses its test set. We split MMLU-Pro’s 12,032 labeled questions into $_ { 6 , 0 0 0 }$ for teacher generation, 3,000 for development, and 3,032 for test, stratified by subject and source. Its official validation set supplies five demonstrations per subject. Demonstrations are shared across models and excluded from all other splits; MMLU-Pro metrics pool subjects. The $8 5 / \mathbf { 1 } 5$ development partition for predictor fitting and selection is shared across model versions and predictor sizes.

<table><tr><td colspan="5">Development split</td></tr><tr><td>Task</td><td>Finetuning</td><td>Development</td><td>Fit (85%) Select</td><td> $( \mathbf { 1 5 \% } )$ </td><td>Test</td></tr><tr><td>WinoGrande</td><td>32,313</td><td> $\overline { { 8 , 0 8 0 } }$ </td><td> $\overline { { 6 , 8 6 8 } }$ </td><td> $\mathbf { 1 } , \mathbf { 2 } \mathbf { 1 } \mathbf { 2 }$ </td><td> $^ { 1 , 2 6 7 }$ </td></tr><tr><td>CommonsenseQA</td><td>6,394</td><td>3,342</td><td> $_ { 2 , 8 4 0 }$ </td><td>502</td><td>1,221</td></tr><tr><td>GSM8K</td><td>5,088</td><td>2,241</td><td>1,904</td><td>337</td><td>1,319</td></tr><tr><td>MMLU-Pro</td><td>5,233</td><td>3,000</td><td>2,550</td><td>450</td><td>3,032</td></tr></table>

Table 2: Questions used at each stage, excluding demonstrations. Predictor fitting and selection partition the development set. For GSM8K and MMLU-Pro, finetuning counts include only questions with an accepted teacher solution.

## B.2 Answer Scoring and Losses

Prompts and option scores. Base models receive plain-text demonstrations; instruction-tuned models use user/assistant turns in their native chat templates, with thinking disabled. On CommonsenseQA, the option score is the answer completion’s log likelihood, without length normalization. On WinoGrande, each option fills the blank, and we score the shared continuation under that filled context.

Sampled-answer probabilities. Generation uses 128 samples per question at temperature 0.9 and $\mathrm { t o p } { - } p = 1$ , with a ${ _ { 2 , 0 4 } } 8 \cdot$ -token output limit. Let $c _ { x } ( j )$ count each distinct extracted answer in $S _ { x } ,$ , including a separate category for extraction failures. We reserve a small mass for unseen answers in category u:

$$
p ( j | x ) = \frac { c _ { x } ( j ) } { 1 2 8 . 5 } \quad ( j \in S _ { x } ) , \qquad p ( u | x ) = \frac { 0 . 5 } { 1 2 8 . 5 } .\tag{36}
$$

We predict the most frequent successfully extracted answer and use its probability as selfconfidence. An unsampled gold answer receives probability $p ( u \mid x )$ when computing answer loss. If all samples fail extraction, the prediction is incorrect with confidence $p ( u \mid x )$

Loss conventions. All losses are in nats; binary cross-entropy clips probabilities to $[ 1 0 ^ { - 6 } , 1 -$ $1 0 ^ { - 6 } ]$ . Table 3 gives the reference losses for empirical suboptimality (Equation $\gamma ) ,$ , held fixed before and after finetuning.
<table><tr><td>Task</td><td>Loss (nats)</td><td>Finetuned model attaining the reference</td></tr><tr><td>WinoGrande</td><td>0.3229</td><td>MISTRAL-NEMO-MINITRON-8B-BASE</td></tr><tr><td>CommonsenseQA</td><td>0.3390</td><td>QWEN3.5-9B-BASE</td></tr><tr><td>GSM8K</td><td>0.1378</td><td>QWEN3.5-9B</td></tr><tr><td>MMLU-Pro</td><td>0.7247</td><td>QWEN3.5-9B-BASE</td></tr></table>

Table 3: Reference answer losses for empirical suboptimality, computed over the released and finetuned models retained on each task.

## B.3 Task-Specific Finetuning and Model Retention

Teacher solutions. For the generation tasks, Qwen3.5-122B-A10B receives each question without demonstrations and is asked to reason step by step and end with the required answer format. It generates four solutions at temperature 0.7 and top $- p = 0 . 9 ,$ with a 2,048-token limit. We keep one whose extracted answer matches the gold answer, omitting questions with no accepted solution. These targets are generated once and shared across models and seeds.

Optimization and selection. LoRA uses rank 8, scaling factor 32, and dropout 0.1 on attention query and value projections; for Qwen3.5-9B, it also adapts fused query–key–value projections in linear-attention layers. We train three seeds at learning rate $5 \times \bar { 1 } 0 ^ { - 5 }$ , weight decay 0.01, and effective batch size 8. Only the gold completion or accepted teacher solution contributes to the loss. Table 4 gives the schedules and selection criteria. Greedy accuracy selects generation-task checkpoints; reported results use 128-sample majority answers.

<table><tr><td>Task</td><td>Steps</td><td>Token limit</td><td>Checkpoint selection</td><td></td></tr><tr><td>WinoGrande</td><td>2,000</td><td>256</td><td>Lowest development answer loss</td><td></td></tr><tr><td>CommonsenseQA</td><td>1,200</td><td>512</td><td>Lowest development answer loss</td><td></td></tr><tr><td>GSM8K</td><td>1,500</td><td>4,096</td><td>Highest development greedy accuracy</td><td></td></tr><tr><td>MMLU-Pro</td><td>1,200</td><td>6,144</td><td>Highest development greedy accuracy</td><td></td></tr></table>

Table 4: Finetuning settings for the weak models. We save four checkpoints per seed, one at each quarter of training, and select among the 12 candidates from three seeds.

Model retention. We retain a model pair when finetuning increases development accuracy. Accuracy uses the highest-scoring option or the 128-sample majority answer, depending on the task. The rule uses no test labels and retains 16, 16, 14, and 13 pairs in task order. All reported comparisons, including predictor-size means and empirical loss references, use these retained pairs.

Table 5 shows development accuracy for all models. Finetuning improves accuracy for every model on the two option-scoring tasks. Accuracy falls for two instruction-tuned releases on GSM8K and three on MMLU-Pro; all five are among the most accurate releases on their respective tasks. The illustrative model, Llama-3.1-8B base, is retained on every task.

Trajectory experiment. For Figure 3, we rerun Llama-3.1-8B base with seed 0 on CommonsenseQA and evaluate every 100 steps through step 600. Step 0 reuses the original results; later checkpoints each use a new Platt map and the mean loss of three newly trained 4B predictors.

<table><tr><td rowspan="2">Model</td><td colspan="2">WinoGrande</td><td colspan="2">CommonsenseQA</td><td colspan="2">GSM8K</td><td colspan="2">MMLU-Pro</td></tr><tr><td>Rel.</td><td>FT</td><td>Rel.</td><td>FT</td><td>Rel.</td><td>FT</td><td>Rel.</td><td>FT</td></tr><tr><td>QWEN3.5-9B-BASE</td><td>88.6</td><td>93.3</td><td>81.5</td><td>83.7</td><td>94.3</td><td>96.0</td><td>66.8</td><td>76.7</td></tr><tr><td>QWEN3.5-9B</td><td>81.9</td><td>93.7</td><td>81.5</td><td>83.6</td><td>95.9</td><td>96.1</td><td>78.9†</td><td>77.3+</td></tr><tr><td>GEMMA-3-12B-PT</td><td>89.5</td><td>94.2</td><td>76.5</td><td>82.4</td><td>87.1</td><td>92.7</td><td>53.5</td><td>63.0</td></tr><tr><td>GEMMA-3-12B-IT</td><td>82.1</td><td>93.1</td><td>76.2</td><td>80.4</td><td>93.4</td><td>95.4</td><td>64.6+</td><td>62.6†</td></tr><tr><td>LLAMA-3.1-8B</td><td>88.3</td><td>93.4</td><td>73.1</td><td>78.8</td><td>70.6</td><td>82.7</td><td>41.6</td><td>50.2</td></tr><tr><td>LLAMA-3.1-8B-INSTRUCT</td><td>84.2</td><td>93.3</td><td>75.7</td><td>79.7</td><td>96.1</td><td>96.8</td><td>53.1</td><td>57.5</td></tr><tr><td>APERTUS-8B-2509</td><td>87.8</td><td>92.1</td><td>71.9</td><td>78.7</td><td>67.2</td><td>75.5</td><td>35.1</td><td>46.0</td></tr><tr><td>APERTUS-8B-INSTRUCT-2509</td><td>86.3</td><td>92.4</td><td>70.2</td><td>78.2</td><td>78.0</td><td>83.0</td><td>40.4</td><td>46.2</td></tr><tr><td>OLMo-3-1025-7B</td><td>86.2</td><td>91.3</td><td>71.9</td><td>80.2</td><td>90.9</td><td>94.4</td><td>41.7</td><td>58.3</td></tr><tr><td>OLMo-3-7B-INSTRUCT</td><td>80.5</td><td>89.8</td><td>74.8</td><td>79.5</td><td>96.8+</td><td>95.9+</td><td>59.6+</td><td>56.2+</td></tr><tr><td>FALCON3-7B-BASE</td><td>87.3</td><td>91.2</td><td>76.3</td><td>78.8</td><td>91.6</td><td>94.0</td><td>50.5</td><td>60.3</td></tr><tr><td>FALCON3-7B-INSTRUCT</td><td>81.5</td><td>91.7</td><td>78.2</td><td>81.4</td><td>95.4</td><td>95.5</td><td>60.5</td><td>61.3</td></tr><tr><td>MISTRAL-7B-V0.3</td><td>89.5</td><td>93.9</td><td>70.3</td><td>80.1</td><td>55.3</td><td>77.6</td><td>36.8</td><td>46.0</td></tr><tr><td>MISTRAL-7B-INSTRUCT-VO.3</td><td>87.8</td><td>94.3</td><td>71.2</td><td>80.0</td><td>73.8</td><td>80.6</td><td>39.9</td><td>45.7</td></tr><tr><td>MISTRAL-NEMO-MINITRON-8B-BASE</td><td>89.9</td><td>94.2</td><td>84.2</td><td>86.3</td><td>81.5</td><td>88.6</td><td>42.4</td><td>53.3</td></tr><tr><td>MISTRAL-NEMO-MINITRON-8B-INSTRUCT</td><td>82.9</td><td>92.6</td><td>80.1</td><td>85.3</td><td>97.9+</td><td>96.5†</td><td>49.3</td><td>54.1</td></tr></table>

Table 5: Development accuracy (%) of released models (Rel.) and their selected finetuned checkpoints (FT). † marks a model whose finetuned checkpoint’s development accuracy does not exceed that of its released model. Marked pairs are excluded from all reported comparisons, including Figure 9.

## B.4 Performance Predictor Training

Inputs. The predictor receives the question, the model’s final answer, its answer distribution, and a prompt to judge correctness. On generation tasks, it also receives one sampled reasoning trace ending in the majority answer. The answer distribution is represented by all option probabilities on the option-scoring tasks, the five most frequent answers on GSM8K, and all sampled option counts on MMLU-Pro. Inputs exclude demonstrations, chat-template tokens, and gold answers, and are never truncated.

Architecture. We encode the input with Qwen3-Embedding-4B to obtain $\phi ( x )$ and apply a linear correction to the self-confidence logit:

$$
g ( 1 \mid x ) = \sigma ( a + b \log \mathrm { i t } f _ { Z } ( 1 \mid x ) + \mathrm { h e a d } ( \phi ( x ) ) ) .\tag{37}
$$

Here $a , b$ are learned scalars and $\sigma$ is the sigmoid function.

Initialization and training. For each evaluated model, we fit Platt calibration (Platt, 2000) by minimizing binary cross-entropy on the predictor-fitting split:

$$
f _ { Z } ^ { \mathrm { P l a t t } } ( 1 \mid x ) = \sigma ( \alpha + \beta \log \mathrm { i t } f _ { Z } ( 1 \mid x ) ) .\tag{38}
$$

This map is shared across predictor sizes and initializes $( a , b ) = ( \alpha , \beta )$ in Equation $^ { 3 7 ; }$ the linear head starts at zero. The Qwen3-Embedding backbone uses the final token’s hidden state and LoRA on query and value projections with rank $^ { 1 6 , }$ scaling factor 32, and dropout 0.05. We train the adapters, head, and $a , b$ for up to three epochs at learning rate $1 0 ^ { - 4 }$ and effective batch size $^ { 1 6 , }$ with weight decay $1 0 ^ { - 4 }$ on the head only. Among ten seeds and all epochs, including epoch zero, we select the lowest correctness loss on the predictor-selection split.

## B.5 Cascade Evaluation

Strong model. We use Qwen3.5-122B-A10B as the strong model. On WinoGrande and CommonsenseQA, we finetune it and select the checkpoint with the lowest development answer loss. These runs use the weak models’ training data, prompts, and optimization settings for 2,000 and 1,600 steps, respectively. We save four checkpoints at quarter intervals and select the lowest development answer loss among them and the released model. Evaluation uses the weak models’ demonstrations and option-scoring rules.

On GSM8K and MMLU-Pro, the released strong model generates 128 samples in thinking mode at temperature 1.0, top-p = 0.95, top-k = 20, and presence penalty 1.5, with a 32,768-token limit. Answer extraction and majority selection follow the weak-model evaluation.

Accuracy and integration. For either routing score $r ,$ we sort the n test questions in decreasing order, breaking ties uniformly at random. $\operatorname { A t } \rho = k / n ,$ we escalate the first k questions. Expected cascade accuracy is

$$
1 - \mathrm { E r r } ( f _ { \mathrm { c a s } , B _ { r , \rho } } ) = \frac 1 n \mathbb E \left[ \sum _ { i = 1 } ^ { n } Z _ { 1 , i } + \sum _ { i = 1 } ^ { n } B _ { r , \rho } ( x _ { i } ) ( Z _ { 2 , i } - Z _ { 1 , i } ) \right] ,\tag{39}
$$

where $Z _ { 1 , i }$ and $Z _ { 2 , i }$ indicate weak- and strong-model correctness, and the expectation is over tiebreaking with answers held fixed. We linearly interpolate between attainable rates and integrate with the trapezoidal rule. Accuracy is linear across each tied group, so its endpoints suffice. We report routing advantage 100D in percentage points averaged over escalation rates.

## C Additional Experimental Results

All results use the retained pairs; predictors use the selected $4 ^ { \mathrm { B } }$ backbone unless another size is stated. Values listed in task order refer to WinoGrande, CommonsenseQA, GSM8K, and MMLU-Pro.

## C.1 Correlation Estimates and Uncertainty

We report Pearson’s r and Spearman’s $r _ { s } ,$ with one-sided p-values $p _ { + }$ for positive correlation. We use SciPy’s standard tests, assuming independent observations and making no adjustment for multiple comparisons.
<table><tr><td>Comparison</td><td>Task</td><td>Pearson r</td><td> $p _ { + }$ </td><td>Spearman  $r _ { s }$ </td><td> $\underline { { { p } _ { + } } }$ </td></tr><tr><td rowspan="4">δ vs. E</td><td>WinoGrande</td><td>0.98</td><td> $< 0 . 0 0 1$ </td><td>0.73</td><td> $< 0 . 0 0 1$ </td></tr><tr><td>CommonsenseQA</td><td>0.98</td><td>&lt; 0.001</td><td>0.94</td><td>&lt; 0.001</td></tr><tr><td>GSM8K</td><td>0.93</td><td>&lt; 0.001</td><td>0.94</td><td>&lt; 0.001</td></tr><tr><td>MMLU-Pro</td><td>0.65</td><td>&lt; 0.001</td><td>0.74</td><td>&lt; 0.001</td></tr><tr><td rowspan="4"> $\Delta \widehat { \delta } \mathrm { v s . } \ \Delta E$ </td><td>WinoGrande</td><td>0.99</td><td>&lt; 0.001</td><td>0.90</td><td>&lt; 0.001</td></tr><tr><td>CommonsenseQA</td><td>0.99</td><td>&lt; 0.001</td><td>0.97</td><td>&lt; 0.001</td></tr><tr><td>GSM8K</td><td>0.78</td><td>&lt; 0.001</td><td>0.89</td><td>&lt; 0.001</td></tr><tr><td>MMLU-Pro</td><td>0.49</td><td>0.044</td><td>0.20</td><td>0.259</td></tr><tr><td rowspan="4"> $\widehat { \delta \mathrm { \ v s . } \ D }$ </td><td>WinoGrande</td><td>0.79</td><td>&lt; 0.001</td><td>0.92</td><td>&lt; 0.001</td></tr><tr><td>CommonsenseQA</td><td>0.66</td><td>&lt; 0.001</td><td>0.87</td><td>&lt; 0.001</td></tr><tr><td>GSM8K</td><td>0.91</td><td>&lt; 0.001</td><td>0.77</td><td>&lt; 0.001</td></tr><tr><td>MMLU-Pro</td><td>0.82</td><td>&lt; 0.001</td><td>0.85</td><td>&lt; 0.001</td></tr><tr><td rowspan="4"> $\Delta \widehat { \delta } \mathbf { v } \mathbf { s } . \ \Delta D$ </td><td>WinoGrande</td><td>0.63</td><td>0.004</td><td>0.76</td><td>&lt; 0.001</td></tr><tr><td>CommonsenseQA</td><td>0.42</td><td>0.054</td><td>0.56</td><td>0.011</td></tr><tr><td>GSM8K</td><td>0.73</td><td>0.001</td><td>0.51</td><td>0.031</td></tr><tr><td>MMLU-Pro</td><td>0.40</td><td>0.087</td><td>0.18</td><td>0.277</td></tr></table>

Table 6: Correlations for the main 4B comparisons. $p _ { + }$ tests a positive association using the tests described in Appendix C.1. Endpoint comparisons contain 32, 32, 28, and 26 model versions in task order; paired changes contain 16, 16, 14, and 13 model pairs.

## C.2 Released and Finetuned Models Separately

Figures $7$ and 8 repeat Figures 1 and $4$ with released and finetuned models in separate panels, using the same models and the selected $4 ^ { \mathrm { B } }$ predictors. Among released models alone, excess predictability correlates with empirical suboptimality at 1.00, 0.98, 0.89, and $_ { 0 . 4 9 }$ in task order, and routing advantage at $_ { 0 . 6 7 } ,$ 0.43, 0.91, and 0.81. The main-text associations therefore do not depend on contrasting the two groups. Among finetuned models, routing advantage remains positively correlated with suboptimality on every task, at 0.79, 0.81, 0.94, and $0 . 6 4 ,$ and excess predictability at $0 . 1 4 , \ 0 . 6 5 , \ 0 . 9 9 ,$ , and 0.32. The weak value on WinoGrande reflects the narrow range after finetuning: all finetuned models lie within 0.22 nats of the best observed loss, and $\mathbf { 1 4 }$ of 16 have slightly negative excess predictability. On MMLU-Pro, both groups show the weaker association discussed in Section 3.4.2.

![](images/a4193e363f6e9231cad301568775ef944f4fc8b8f7b15cdbb1916a54b5b08c63.jpg)

Figure $7 { : }$ Excess predictability of the ${ \bf 4  B }$ predictor, with released and finetuned models shown separately. Each column is a task; the top row shows released models and the bottom row their finetuned versions, using the same points as Figure 1. Excess predictability E is the predictor’s correctness-log-loss gain over self-confidence; empirical suboptimality $\dot { \widehat { \delta } }$ is the gap to the best observed answer log loss on each task. Circles denote base models and triangles instruction-tuned models. Lines are least-squares fits within each panel; r is Pearson correlation and $p _ { + }$ its one-sided, unadjusted p-value for positive correlation. Correlations and p-values are in Table $7 \cdot$ Axis ranges differ across panels.

![](images/5b2efefef98cabf393a8393f6d47dcb2496df5b65f91eb30b2c051637b1e99df.jpg)

Figure 8: Routing advantage of the ${ \bf 4  B }$ predictor, with released and finetuned models shown separately. Routing advantage 100D (percentage points) versus empirical suboptimality $\widehat { \delta }$ (nats), using the same points as Figure $4 \cdot$ . 100D is the cascade accuracy gain over self-confidence routing, averaged over equal escalation rates; $\widehat { \delta }$ is the gap to the best observed answer log loss on each task. Columns are tasks; orange (top) denotes released models and blue (bottom) their finetuned versions. Circles denote base releases and triangles instruction-tuned releases. Lines are least-squares fits; r is Pearson correlation and $p _ { + }$ its one-sided, unadjusted p-value for positive correlation. Axis ranges differ across panels.
<table><tr><td>Comparison</td><td>Task</td><td>Pearson r</td><td> $p _ { + }$ </td><td>Spearman  $r _ { s }$ </td><td> $p _ { + }$ </td></tr><tr><td rowspan="4">δ vs. E (released)</td><td>WinoGrande</td><td>1.00</td><td>&lt; 0.001</td><td>0.92</td><td>&lt; 0.001</td></tr><tr><td>CommonsenseQA</td><td>0.98</td><td>&lt; 0.001</td><td>0.97</td><td>&lt; 0.001</td></tr><tr><td>GSM8K</td><td>0.89</td><td>&lt; 0.001</td><td>0.90</td><td>&lt; 0.001</td></tr><tr><td>MMLU-Pro</td><td>0.49</td><td>0.045</td><td>0.60</td><td>0.015</td></tr><tr><td rowspan="4">δ vs. E (finetuned)</td><td>WinoGrande</td><td>0.14</td><td>0.301</td><td>-0.03</td><td>0.547</td></tr><tr><td>CommonsenseQA</td><td>0.65</td><td>0.003</td><td>0.64</td><td>0.004</td></tr><tr><td>GSM8K</td><td>0.99</td><td>&lt; 0.001</td><td>0.98</td><td>&lt; 0.001</td></tr><tr><td>MMLU-Pro</td><td>0.32</td><td>0.142</td><td>0.44</td><td>0.066</td></tr><tr><td rowspan="4">δ vs. D (released)</td><td>WinoGrande</td><td>0.67</td><td>0.002</td><td>0.86</td><td>&lt; 0.001</td></tr><tr><td>CommonsenseQA</td><td>0.43</td><td>0.046</td><td>0.59</td><td>0.008</td></tr><tr><td>GSM8K</td><td>0.91</td><td>&lt; 0.001</td><td>0.81</td><td>&lt; 0.001</td></tr><tr><td>MMLU-Pro</td><td>0.81</td><td>&lt; 0.001</td><td>0.80</td><td>&lt; 0.001</td></tr><tr><td rowspan="4">δ vs. D (finetuned)</td><td>WinoGrande</td><td>0.79</td><td>&lt; 0.001</td><td>0.56</td><td>0.011</td></tr><tr><td>CommonsenseQA</td><td>0.81</td><td>&lt; 0.001</td><td>0.82</td><td>&lt; 0.001</td></tr><tr><td>GSM8K</td><td>0.94</td><td>&lt; 0.001</td><td>0.82</td><td>&lt; 0.001</td></tr><tr><td>MMLU-Pro</td><td>0.64</td><td>0.010</td><td>0.63</td><td>0.010</td></tr></table>

Table $7 { : }$ Within-stage correlations for ${ \bf 4  B }$ predictors. $p _ { + }$ tests a positive association using the tests described in Appendix C.1. Each stage contains 16, 16, 14, and 13 models in task order.

## C.3 Predictor Capacity

We compare $0 . 6 \mathrm { B } , 4 \mathrm { B } ,$ and 8B predictors to measure how much excess predictability each size captures and whether our main findings persist across predictor sizes. All three predictors use Qwen3-Embedding backbones and evaluate the same retained model pairs: $\mathbf { 1 6 , 1 6 , 1 4 } ,$ and 13 in task order. We report gains over self-confidence and use two-parameter Platt calibration as an additional baseline.

Excess predictability captured at each size. The results are shown in Figure 9. For released models, larger predictors generally achieve greater gains over self-confidence (first row, orange circles). After finetuning (first row, blue squares), mean gains decrease on every task at all three sizes, and increasing predictor size also provides less benefit.

The second row compares predictors with Platt-calibrated self-confidence. The 0.6B predictor adds little beyond calibration, whereas the $4 ^ { \mathrm { B } }$ and 8B predictors generally achieve lower correctness log loss. For both the $4 ^ { \mathrm { B } }$ and 8B predictors, mean gains over Platt-calibrated self-confidence decrease after finetuning on every task. For finetuned models on WinoGrande, no predictor size improves on Platt calibration on average. This is consistent with the finetuned models’ low empirical suboptimality on this simple task, leaving little room for additional predictor capacity to help.

![](images/53fafaba2226954f51b1fa542b0c8771537f4bd299b2d4042f4b03d5236c4434.jpg)  
Figure 9: Predictor gains over self-confidence and Platt calibration. Mean baseline loss minus predictor loss over the retained models on each task; positive values favor the predictor. Columns show tasks; rows compare against raw and Platt-calibrated confidence. Orange solid lines with circles: released models; blue dashed lines with squares: finetuned models. Predictor size is on the horizontal axis; vertical scales are shared within each row. Lines connect the three evaluated sizes.

Robustness across predictor sizes. We check whether predictor size changes the associations with suboptimality across models and the observed effects of finetuning. Across models, both excess predictability and routing advantage remain positively correlated with suboptimality on every task at ${ \mathrm { 8 B } } ,$ as at $4 ^ { \mathrm { B } }$ (Figures 10–13). These associations are less consistent at 0.6B: the correlation is negative for excess predictability on MMLU-Pro, and for routing advantage on WinoGrande and MMLU-Pro. On these tasks, the 0.6B predictor adds little beyond calibration.

The finetuning result holds at all three sizes: finetuning reduces mean excess predictability on every task (Figure 9). Larger reductions in suboptimality also accompany larger reductions in excess predictability at every size, with Pearson correlations of at least 0.99, 0.98, 0.73, and $\phantom { - } 0 . 4 8$ in task order.

![](images/6154443f88d5bd828e54fd55fb6b412b06165b474faf4492b01b1587213cc03f.jpg)  
Figure 10: Excess predictability with the 0.6B predictor. E is the predictor’s correctness-log-loss gain over self-confidence; $\widehat { \delta }$ is the gap to the best observed answer log loss on each task. Orange denotes released models and blue finetuned models; circles denote base releases and triangles instruction-tuned releases. Lines are least-squares fits; r is Pearson correlation and $p _ { + }$ its one-sided, unadjusted p-value for positive correlation.

![](images/21980ec0d64ee37469c5e10f6255bd91f5be8826f2cf44cf33b0ea8e52af0ae7.jpg)  
Figure 11: Routing advantage with the 0.6B predictor. 100D is the cascade accuracy gain over selfconfidence routing, averaged over equal escalation rates, in percentage points; $\widehat { \delta }$ is the gap to the best observed answer log loss on each task. Orange denotes released models and blue finetuned models; circles denote base releases and triangles instruction-tuned releases. Lines are least-squares fits; r is Pearson correlation and $p _ { + }$ its one-sided, unadjusted p-value for positive correlation.

![](images/ad4a580a1c9130aa392de4c238bf968e44d4c95b80441741c8b3000493af2cb6.jpg)  
Figure 12: Excess predictability with the 8B predictor. E is the predictor’s correctness-log-loss gain over self-confidence; $\widehat { \delta }$ is the gap to the best observed answer log loss on each task. Orange denotes released models and blue finetuned models; circles denote base releases and triangles instruction-tuned releases. Lines are least-squares fits; r is Pearson correlation and $p _ { + }$ its one-sided, unadjusted p-value for positive correlation.

![](images/7fa2745997e2be4bbb34cbea433f8030c48b5a58f070c0e51d77c078fa0645b8.jpg)  
Figure 13: Routing advantage with the 8B predictor. 100D is the cascade accuracy gain over self-confidence routing, averaged over equal escalation rates, in percentage points; $\widehat { \delta }$ is the gap to the best observed answer log loss on each task. Orange denotes released models and blue finetuned models; circles denote base releases and triangles instruction-tuned releases. Lines are least-squares fits; r is Pearson correlation and $p _ { + }$ its one-sided, unadjusted p-value for positive correlation.

## C.4 Cascade Accuracy Across Tasks

Figure 14 extends the main-text CommonsenseQA example to all four tasks, using Llama-3.1-8B base before and after finetuning. The gap between predictor and self-confidence routing measures the benefit of choosing different questions at the same escalation rate.

![](images/65b2174da7dcfeb26328bb2670e3e827b5b30b299063d48f2c94168bd03f02a5.jpg)

Figure 14: Cascade accuracy across all four tasks. We pair Llama-3.1-8B base with Qwen3.5-122B-A10B, finetuned on WinoGrande and CommonsenseQA and released on the other two tasks. Orange denotes the released weak model and blue its finetuned version. Solid curves use the selected 4B predictor g, dashed curves use self-confidence $f _ { 1 , Z } ,$ , and dotted lines show random routing. Shading shows the accuracy difference between the two routers.

## C.5 How Strong-Model Errors Affect Routing Gains

Two contributions. By Equation 9, the routing advantage at each escalation rate is $\mathbb { E } \big [ u \epsilon _ { 1 } \big ( X \big ) \big ] -$ $\mathbb { E } [ u \epsilon _ { 2 } ( X ) ]$ with $u = B _ { g , \rho } - B _ { f , \rho }$ . The first term is the weak-error term: weak errors escalated by predictor routing minus those escalated by self-confidence routing. The second is the strong-error term: strong errors encountered by predictor routing minus those encountered by self-confidence routing. We estimate both on the test questions by replacing $\epsilon _ { 1 }$ and $\epsilon _ { 2 }$ with the observed error indicators $1 - Z _ { 1 , i }$ and $1 - Z _ { 2 , i , }$ , averaging over tie-breaking as in Equation 39, and integrating over escalation rates as for D. Table 8 reports both terms.

<table><tr><td>Task</td><td>Models</td><td>100D</td><td>Weak-error term</td><td>Strong-error term</td></tr><tr><td rowspan="2">WinoGrande</td><td>Released</td><td>1.38</td><td>1.80</td><td>0.42</td></tr><tr><td>Finetuned</td><td>0.18</td><td>0.21</td><td>0.03</td></tr><tr><td rowspan="2">CommonsenseQA</td><td>Released</td><td>1.09</td><td>1.26</td><td>0.17</td></tr><tr><td>Finetuned</td><td>0.18</td><td>0.17</td><td>-0.01</td></tr><tr><td rowspan="2">GSM8K</td><td>Released</td><td>0.52</td><td>0.59</td><td>0.07</td></tr><tr><td>Finetuned</td><td>0.20</td><td>0.24</td><td>0.03</td></tr><tr><td rowspan="2">MMLU-Pro</td><td>Released</td><td>0.94</td><td>1.75</td><td>0.82</td></tr><tr><td>Finetuned</td><td>0.07</td><td>0.94</td><td>0.87</td></tr></table>

Table 8: Routing advantage split into two contributions. Means over retained models with $4 ^ { \mathrm { B } }$ predictors, in percentage points averaged over escalation rates. Both error terms are differences from self-confidence routing; 100D equals the weak-error term minus the strong-error term, up to rounding.

Strong errors usually reduce the gain. Predictor routing captures more weak-model errors, but the strong model also answers some of the escalated questions incorrectly. Averaged over escalation rates, the strong-error term is nonnegative for 91 of 118 retained model versions: in these cases, strong errors offset some of the gain from capturing additional weak errors. A bootstrap with 500 resamples of test questions gives a 95% interval entirely above zero in $4 0$ cases and entirely below zero in one. The effect is largest on MMLU-Pro, where the strong model is 87.6% accurate. For finetuned weak models on this task, capturing additional weak errors contributes 0.94 percentage points, but strong errors subtract 0.87 points, leaving a routing advantage of 0.07 points. On GSM8K, the strong model is 97.9% accurate, so strong errors have less effect.

Connection to the bound. Theorem 2 bounds the additional weak errors captured in terms of suboptimality and predictor capacity. At any escalation rate where the strong-error term is nonnegative, Corollary 2 carries this bound over to the routing advantage. The 91 cases above satisfy the corresponding comparison after averaging over escalation rates. Empirically, suboptimality is at least as strongly correlated with the weak-error term as with the routing advantage D. The correlations are 0.80, 0.72, 0.91, and 0.86 in task order, compared with 0.79, 0.66, 0.91, and 0.82 for D. This is consistent with strong errors adding variation to the routing advantage in Figure $4 \cdot$ Avoiding strong errors can also increase the routing advantage, but the largest observed contribution is only 0.12 percentage points. Without the sign condition, the strong-error contribution to the gain at rate $\rho$ is at most min $\{ \mathrm { E r r } ( f _ { 2 } ) , \rho , 1 - \rho \}$ . The worst-case allowances averaged over escalation rates are much larger than the measured contributions: 10.6, 8.0, 2.1, and 10.8 points in task order.