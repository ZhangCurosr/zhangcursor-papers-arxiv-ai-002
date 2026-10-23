# Scaling Laws for Generative Reward Models

Angel Raychev   
SynthLabs.ai   
Stanford University

Nathan Lile SynthLabs.ai

Rafael Rafailov Stanford University Thinking Machines

Afiliations at the time of the research

## Abstract

We study generative reward models (GenRMs) as online preference providers for policy optimization in creative writing. Across Qwen3 reward-model sizes from 0.6B to 14B, Thinking judges trained with GRPO have higher static Gold-label accuracy than Answer-Only judges trained with SFT. This ranking often reverses when the judges train policies through online DPO: Answer-Only rewarders yield higher peak and top-five-average policy Elo in seven of eight matched settings under a fixed, learned 32B Gold evaluator. Intermediate Thinking-judge checkpoints can also outperform later, more statically accurate checkpoints as rewarders. Sweeps over rewarder size, policy size, training duration and DPO regularization show substantial ben efits from reward-model capacity, including judges larger than the policy. We analyze trajectories with evaluator-specific Elo arenas, conditional match-resampling intervals and descriptive scaling fits. These results demonstrate an evaluator–rewarder gap in the studied pipelines: fixed-distribution judging accuracy does not determine downstream optimization quality. The comparison changes judge-training objective as well as response format, and the main ranking is conditional on its learned reference rather than a universal human-quality result. A limited follow-up using a diferent learned judge is inconclusive overall.

## 1 Introduction

Policies trained from preference feedback can exploit imperfections in their reward models: proxy scores may rise while quality under a separate reference deteriorates [Gao et al., 2022, Rafailov et al., 2024]. Generative reward models (GenRMs) ofer a flexible feedback mechanism. By formulating evaluation as conditional text generation rather than scalar prediction, they can leverage pretraining priors and chain-of-thought reasoning and achieve strong performance on static preference benchmarks [Mahan et al., 2024, Zhang et al., 2025, Whitehouse et al., 2025]. Whether those stronger evaluators supply better online training signals is a separate question.

Prior work has already shown that static reward-model accuracy can be a poor predictor of downstream optimization [Wen et al., 2025, Frick et al., 2024, Kim et al., 2025]. We study this distinction specifically for generative rewarders as the policy distribution changes, targeting a practitioner-facing question: how do judge-training recipe, judge capacity and policy capacity affect the outcomes of online preference optimization? We study three design dimensions inside the training loop: (i) Thinking GRPO versus Answer-Only SFT judging, (ii) judge scale, and (iii) policy scale. We instantiate these axes in creative writing using LitBench [Fein et al., 2025], a non-verifiable preference domain where correctness cannot be programmatically checked. In such domains, static evaluator accuracy can saturate while reward hacking and proxy-Gold divergence remain salient, making creative writing an ideal testbed for whether stronger static evaluators translate to better in-loop reward signals.

Contributions. (1) We present a systematic scaling study of GenRMs deployed as online reward signals, sweeping judge format, judge size (0.6B–14B), policy size (0.6B–8B), training duration and checkpoint selection under online DPO. (2) We trace policy trajectories under both proxy and Gold evaluators, using within-arena comparisons to separate rising proxy scores from gains under the reference evaluator. (3) We provide empirical guidance within the studied pipelines: judge scale and training duration matter strongly, Answer-Only rewarders usually sustain higher Gold-rated quality than Thinking variants, and intermediate GenRM checkpoints can outperform later models as rewarders.

Our findings reveal an evaluator–rewarder gap. Thinking GenRMs have higher static validation accuracy across the studied sizes, yet this ranking often reverses during online optimization. Answer-Only GenRMs usually yield higher peak Gold Elo, with exceptions where Thinking remains competitive. Additional judge capacity remains valuable above policy size; this does not establish that judging is intrinsically harder than generating.

<table><tr><td>Observation</td><td>Evidence in this paper</td></tr><tr><td>Static evaluation favors Thinking GenRMs</td><td>Thinking judges have higher Gold-label validation accuracy across the studied sizes (Figure 1).</td></tr><tr><td>Online rewarder quality is</td><td>In online DPO, Answer-Only judges usually give higher peak and</td></tr><tr><td>different</td><td>top-five Gold Elo than Thinking judges in the matched sweep (Table 3, Figure 4).</td></tr><tr><td>Judge capacity matters beyond policy size</td><td>Larger Answer-Only GenRMs improve Gold-rated policy quality beyond the fixed 4B policy size; Thinking size effects are less</td></tr><tr><td>Static accuracy can be</td><td>monotone (Figure 5). Intermediate Thinking checkpoints can outperform fully-trained</td></tr></table>

Table 1: A compact map of the paper’s central evidence. The main empirical distinction is between GenRMs as evaluators on a fixed distribution and GenRMs as rewarders inside an online optimization loop.

## 2 Methodology

We begin with a dataset of human-written stories and implicit preferences inferred from Reddit votes:

$$
\mathcal { D } _ { \mathrm { h u m a n } } = \{ x ^ { ( i ) } , y _ { A } ^ { ( i ) } , y _ { B } ^ { ( i ) } , I _ { H } ^ { ( i ) } \} _ { i = 1 } ^ { N } ,
$$

where $x ^ { ( i ) } \in \mathcal { X }$ are prompts, $y _ { A } ^ { ( i ) } , y _ { B } ^ { ( i ) } \in \mathcal { V }$ are pairs of stories responding to the same prompt, and $I _ { H } ^ { ( i ) } \in \{ A , B \}$ denotes the response preferred by the dataset’s vote-derived ordering. The subscript “human” distinguishes this source supervision from model-generated labels; it does not denote explicit pairwise annotation by human raters.

Gold evaluator. We train a large generative reward model, denoted $V _ { \mathrm { g o l d } }$ , on $\mathcal { D } _ { \mathrm { h u m a n } }$ to obtain a frozen, queryable reference evaluator. Its training target is the source dataset’s implicit preference ordering, not an on-demand human judgment. Agreement with these labels on the static story distribution does not establish reliable preferences on later policy-generated responses. We therefore study the downstream behavior of smaller proxy GenRMs relative to this learned anchor, not an independently established measure of human writing quality.

To make evaluation computationally feasible, we choose $V _ { \mathrm { g o l d } }$ to be an Answer-Only model that outputs a short structured verdict without a reasoning trace. Using $V _ { \mathrm { g o l d } }$ , we construct a Gold

Preference dataset:

$$
\mathcal { D } _ { \mathrm { g o l d } } = \{ x ^ { ( i ) } , y _ { A } ^ { ( i ) } , y _ { B } ^ { ( i ) } , I _ { G } ^ { ( i ) } \} _ { i = 1 } ^ { M } ,
$$

which is then used to train a variety of smaller GenRMs.

Proxy GenRMs. We consider two types of GenRMs: Answer-Only and Thinking. The Answer-Only models output a direct judgment token (A or B):

$$
I \sim v _ { \mathrm { a n s } } ( \cdot \mid x , y _ { A } , y _ { B } ) ,
$$

while the Thinking models first generate a reasoning trace z before producing the final verdict:

$$
z \sim v _ { \mathrm { t h i n k } } ( \cdot \mid x , y _ { A } , y _ { B } ) , \quad I \sim v _ { \mathrm { t h i n k } } ( \cdot \mid x , y _ { A } , y _ { B } , z ) .
$$

Training objectives. We train Answer-Only models using supervised fine-tuning $( \mathrm { S F T } )$ and Thinking models using GRPO. SFT directly supervises the verdict; GRPO uses normalized outcome rewards without requiring a reference reasoning trace. This compares two practical training pipelines, rather than isolating the causal efect of reasoning under identical objectives. The 4B Thinking ablation below provides evidence for the chosen GRPO recipe, not an exhaustive comparison of training methods.

The retained GRPO implementation constructs raw outcome rewards in two steps:

1. Accuracy reward: A binary reward (1 if the model reaches the correct verdict, 0 otherwise).

2. Raw positional-consistency bonus: The reward manager links both orderings using a shared pair identifier and computes a majority vote within each ordering’s sampled completions. An additional reward of 1 is assigned to correct completions only when both majorities are correct after normalizing response order.

Efect of normalization. In the retained binary-reward setup, the bonus uniformly scales every correct reward within an ordered-prompt group: $r _ { i } = c _ { g } a _ { i } ,$ , with accuracy $a _ { i } \in \{ 0 , 1 \}$ and $c _ { g } \in \{ 1 , 2 \}$ GRPO standardizes within that prompt’s UID group, so $( r _ { i } - \bar { r } _ { g } ) / ( s _ { g } + \epsilon ) = ( a _ { i } - \bar { a } _ { g } ) / ( s _ { a , g } + \epsilon / c _ { g } )$ Apart from the numerical stabilizer, the bonus therefore cancels. It changes raw reward statistics but does not provide an independent positional-consistency training signal in this configuration. We preserve the historical implementation and do not attribute the observed GRPO gains to this bonus. We refer to these trained GenRMs as Proxy models, and evaluate them with respect to $V _ { \mathrm { g o l d } }$ preferences.

Policy optimization. We train policies $\pi \theta$ of varying sizes using Online Direct Preference Optimization (Online DPO). GenRMs produce pairwise preferences that DPO can consume directly. Ofline DPO reuses a fixed preference dataset; its response-generating policies need not be the frozen reference model in the DPO loss. In online DPO, fresh response pairs are repeatedly sampled from the current policy, so the supervision distribution evolves as the policy changes. This setting lets us study GenRM behavior under policy-induced distribution shift. Prior work has also studied the benefits of online preference optimization [Guo et al., 2024].

Formally, at online update $t ,$ we sample $x \sim \mathcal { P } _ { \mathrm { t r a i n } }$ and two responses $y _ { 1 } , y _ { 2 } \sim \pi _ { \theta _ { t } } ( \cdot \mid x )$ . A proxy GenRM v orders the pair as $( y _ { v } ^ { + } , y _ { v } ^ { - } )$ , where $y _ { v } ^ { + }$ is preferred to $y _ { v } ^ { - }$ . The DPO loss induced by this proxy is

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { D P O } } ( \boldsymbol { \theta } ; \boldsymbol { v } , \pi _ { \theta _ { t } } ) = - \mathbb { E } \left[ \log \sigma \left( \beta \left[ \log \frac { \pi _ { \theta } ( y _ { v } ^ { + } \mid x ) } { \pi _ { \mathrm { r e f } } ( y _ { v } ^ { + } \mid x ) } - \log \frac { \pi _ { \theta } ( y _ { v } ^ { - } \mid x ) } { \pi _ { \mathrm { r e f } } ( y _ { v } ^ { - } \mid x ) } \right] \right) \right] . } \end{array}
$$

The idealized Gold-aligned update replaces v with $V _ { \mathrm { g o l d } }$ . Where the proxy disagrees with Gold matters as well as how often it disagrees: diferent response pairs contribute diferent gradient directions and magnitudes. Verdict filtering also changes which sampled pairs reach the update. Static agreement on $\mathcal { D } _ { \mathrm { g o l d } }$ measures neither efect on the evolving pair distribution. Appendix B.3 gives a short DPO interpretation, without claiming a measured gradient-level explanation of the results.

The policies are trained on prompts sampled from the same broad creative-writing source distribution as the Gold and Proxy datasets, with LitBench prompts excluded by the intended construction procedure. We periodically sample responses on a fixed validation set and estimate Elo ratings from pairwise comparisons under the proxy and Gold evaluators. Each evaluator defines a separate fitted arena: numerical equality between its Elo and another evaluator’s Elo is not response-level agreement. Appendix B.1 describes the actual Bradley–Terry fits, checkpoint summaries and conditional uncertainty.

The empirical question is whether a higher-accuracy judge on the fixed Gold-labeled validation set also trains a higher-rated policy under Gold. Agreement on human-story pairs and agreement on evolving policy-generated pairs are diferent measurements. Our experiments compare the static ranking with downstream policy outcomes; they do not reconstruct trustworthy response-level online agreement from aggregate Elo ratings.

## 3 Experimental Setup

Dataset construction. Our design distinguishes source vote-derived supervision, Gold-derived labels and policy-generation prompts:

• Source preference dataset $\mathcal { D } _ { \mathbf { h u m a n } }$ : LitBench [Fein et al., 2025] curates same-prompt Reddit story pairs using vote margins, engagement, posting-time and length filters. The higher-voted story defines an implicit preference label, rather than an explicit human pairwise annotation. The retained Gold-training script uses rows 1,000–21,999 for training (21,000 pairs) and rows 0–999 for validation (1,000 pairs).

• Gold preference dataset $\mathcal { D } _ { \mathbf { g o l d } }$ : The frozen Gold evaluator relabels pairs of human-written stories for proxy training. The retained artifact contains 21,827 raw training rows and 1,000 validation rows. Its training count is consistent with the remaining LitBench subset after row 21,999, although the complete original training-relabel invocation is not retained. The validation pairs reuse the first 1,000 source pairs, with Gold-derived rather than vote-derived labels. Thus Gold and proxy validation share story pairs; they are not independent source samples. Proxy training filters null Gold labels and expands surviving pairs into both positional orders.

• Policy prompts and validation references: Policy training uses prepared Writing-Prompts prompts<sup>1</sup>, with LitBench prompts excluded by the construction procedure. The retained main preparation combines 50,000 train, 20,000 reserve and 20,000 extra prompts; a separate supplementary split supports later continuations. Policy validation instead uses the prompts from the first 1,000 LitBench pair records, with each original higher-voted story as its fixed response reference. This reuses the static validation records, not Gold-training examples; it is not a fresh WritingPrompts sample. Online training preferences come from newly generated policy response pairs judged by the proxy, not from the Reddit stories.

This separation identifies three distinct roles: source labels train Gold, Gold-derived labels train proxy judges, and fresh proxy-labeled policy responses drive online DPO. The frozen reference policy in the DPO loss is a fourth role, distinct from the fixed Reddit story used in validation.

Model families. We use the Qwen3 model family for all GenRMs and policies. It provides Thinking and Answer-Only modes across several parameter sizes, allowing matched mode/size comparisons within one family. The comparison still changes judge-training objective, and parameter scale is not isolated from all model-specific diferences. Using the same family for policies and GenRMs lets us study their capacity interaction under a consistent experimental framework.

Gold GenRM training. We train the Gold evaluator from Qwen3-32B using supervised fine-tuning on $\mathcal { D } _ { \mathrm { h u m a n } }$ . To improve positional robustness, we train on both orderings $( y _ { A } , y _ { B } )$ and $( y _ { B } , y _ { A } )$ , resulting in 42,000 ordered training examples before any input filtering. We use a batch size of 128 and a learning rate of $1 \times 1 0 ^ { - 5 }$

Proxy GenRM training. We train Proxy GenRMs from the Qwen3 series at sizes {0.6B, 1.7B, 4B, 8B, 14B} using $\mathcal { D } _ { \mathrm { g o l d } }$ . We refer to of-the-shelf models as Baseline-[size]-Ans and Baseline-[size]-Think. Answer-Only models use supervised fine-tuning with batch size 128 and learning rate $1 \times 1 0 ^ { - 5 }$ . Each retained Gold-labeled pair produces two ordered examples; the final consumed count depends on label and input filtering. Thinking models use GRPO with 16 prompts per step, 8 completions per prompt, configured minibatch size 64 and learning rate $1 \times 1 0 ^ { - 6 }$ The retained launcher disables the trainer-level KL reward controller but enables a separate actor KL loss with coeficient $3 \times 1 0 ^ { - 5 }$ ; this is not a zero-KL setup. We refer to trained proxies as GenRM-[size]-Ans and GenRM-[size]-Think.

Policy training. We train policies $\pi _ { \theta }$ from the Qwen3 series at sizes {0.6B, 1.7B, 4B, 8B} using Online DPO with thinking disabled. We use a prompt batch size of 192, global minibatch size of 64 preference pairs, learning rate $1 \times 1 0 ^ { - 6 }$ and $\beta = 0 . 0 2$ unless varied. Two current-policy responses are generated per prompt. The proxy judges both orderings, and the trainer sums ordernormalized verdicts, discarding zero-sum pairs. A verdict parsed as invalid contributes zero, so one valid verdict and one invalid verdict can still yield a retained pair; this is not a guarantee of two valid agreeing votes. Policy and reference likelihoods use the same historical response mask, which excludes the final attended token, including a content token if a capped generation lacks EOS. The original experiment description specifies prompt and response caps of 512 and 2048 tokens, respectively. The retained hand-edited launcher instead contains caps of 500 and 1500; resolved per-run configurations have not been recovered for every plotted trajectory, so those original caps should be read as nominal settings rather than a newly verified run-wide invariant.

Evaluation. During policy training, we save validation generations every 10 training steps; the retained launcher saves model weights every 50 steps. We call the evaluated policy states evaluation snapshots, without implying that every state has separately retained weights. We compute Elo ratings from pairwise comparisons judged by the proxy and Gold evaluators. Each evaluator has its own fitted ratings over the relevant contestants and comparisons; axes from diferent evaluators are not one shared numerical scale. The principal rating exports cover approximately 460 steps, usually 47 evaluation snapshots including step zero. Subsequent policy continuations extend beyond this window; they do not share a newly reconstructed common-arena rating table and are not spliced into these curves.

Artifact reconstruction. Counts above distinguish stored raw splits from examples retained after filtering. The inspected code identifies the consumed data roles, but does not retain every original preparation invocation or resolved per-run configuration. The released analysis identifies its numerical inputs separately from the historical training recipes.

Code and analysis. Training implementation, historical recipes, numerical results and figurereproduction instructions are available in the public research repository. The release distinguishes reproducible analysis of retained results from full training replay, which additionally requires model and dataset access. Prompt templates are provided in Appendix C.

<table><tr><td>Axis</td><td>Values swept</td><td>Default / anchor</td></tr><tr><td>Proxy GenRM for- mat</td><td>Answer-Only SFT, Thinking GRPO</td><td>Both compared</td></tr><tr><td>Proxy GenRM size</td><td>0.6B, 1.7B, 4B, 8B, 14B</td><td>4B unless varied</td></tr><tr><td>Policy size Online DPO β</td><td>0.6B, 1.7B, 4B, 8B</td><td>4B unless varied</td></tr><tr><td>Training trajectory</td><td>0.1, 0.02, 0.01, 0.005, 0.002</td><td>0.02</td></tr><tr><td></td><td>Policy evaluation every 10 steps; intermediate Thinking GenRM checkpoints</td><td>Principal exports: about 460 steps</td></tr></table>

Table 2: Axes studied in matched and fixed-axis online-DPO sweeps, not a full factorial grid. The design separates fixed-distribution accuracy from in-loop rewarder behavior.

## 4 Experiments and Results

## 4.1 GenRM Training

![](images/c971c78b85366d45dd834136f7cfa5c6b19f437dc858c1bea04d82b898bda265.jpg)  
Figure 1: Static evaluator accuracy on the Gold Preference dataset. Thinking GenRMs exceed Answer-Only models across the studied sizes, while both trained variants substantially exceed of-the-shelf baselines. The size of the static advantage varies by model.

Gold evaluator validation. The Gold GenRM achieves 79% agreement with LitBench’s vote-derived labels on the held-out validation split of $\mathcal { D } _ { \mathrm { h u m a n } }$ . This establishes its agreement with the dataset ordering on source story pairs, not human agreement on policy-generated outputs after optimization. The source study also reports strong creative-writing preference prediction [Fein et al., 2025], but its diferent evaluators and procedures are not a matched baseline for this number.

Trained GenRMs substantially outperform baselines. Both Answer-Only SFT and Thinking GRPO improve static Gold-label accuracy over their of-the-shelf counterparts (Figure 1). Similar baseline accuracies do not rule out evaluator-specific preferences or format bias. The trained-versus-baseline improvement is substantial across the sizes studied.

GRPO is efective in the 4B Thinking ablation. On the Gold Preference dataset, the of-the-shelf model achieves 60.0%, STaR achieves 60.7%, distillation from Qwen3-32B thinking traces (SFT-style) achieves 64.7%, and the retained GRPO recipe achieves 83.2%. GRPO gives the largest improvement among these tested recipes, motivating its use for the Thinking proxies. The raw order-consistency bonus does not supply a distinct normalized update, as explained above; this is not evidence for its separate benefit.

Thinking GenRMs outperform Answer-Only models on in-distribution evaluation. Thinking models have higher static Gold-label accuracy at each studied size (Figure 1); the gap varies by size. Majority voting over k = 5 and k = 16 Thinking completions gives modest further gains. Given the additional computation, we use one completion per judged ordering during policy training.

![](images/dfe8eef4849a691052815bd1465564b59f2e170faca0043a53cadbb4325a0651.jpg)  
Training order; square: first, circle: last  
Figure 2: Trained versus of-the-shelf 4B rewarders training a 4B policy. Points are recorded snapshots connected in policy-training order; squares/circles mark first/last observations, not maxima. Gold is shared within this experiment family, whereas own-proxy ratings are separate evaluator fits. The chronological Gold comparison in Appendix A exposes the late trained/baseline Thinking crossover.

## 4.2 Trained GenRMs vs. Baseline Models in Policy Training

Research question. Does training the GenRM improve downstream policy quality compared to of-the-shelf models?

We compare policies optimized with trained versus of-the-shelf 4B GenRMs, with the policy also fixed at 4B (Figure 2). Here “baseline” describes the rewarder, not an untrained policy. Trained rewarders produce higher peak Gold ratings in both modes. That is not uniform dominance at every training duration: at matched step 460, the of-the-shelf Thinking rewarder’s policy reaches 542 Gold Elo versus 501 for the trained Thinking rewarder’s policy, after the latter declines from a higher peak of 803. Answer-Only retains a trained-rewarder advantage throughout the matched trajectory, with peaks 961 versus 476. The restored chronological comparison in Appendix A makes the late crossover and the peak/final distinction explicit. While Thinking has the stronger of-the-shelf rewarder peak, Answer-Only has the stronger trained-rewarder peak in this arena.

![](images/c4620bacd86b603a792f990b712ec5f6bae7a647fd91eb327c081cc13a2004bd.jpg)  
Figure 3: Policies trained with intermediate 4B Thinking-judge checkpoints. The retained historical image uses fitted display curves, not chronological connecting lines; its diagonal is not verdict agreement. The original account specifies 40,000 policy-training prompts and the judge-checkpoint accuracies reported in the text. This separate ablation illustrates that the most accurate static evaluator need not be the strongest online rewarder.

## 4.3 Answer-Only vs Thinking

Research question. Which of the compared Answer-Only SFT and Thinking GRPO pipelines provides the better online rewarder?

Representative Gold-evaluated policy trajectories

![](images/8ec883556323c0caeeceedb4ce0f49b898dbef479b0263f90b6664c012d5af5b.jpg)

![](images/a954a17eb5eb6d66bd1398e75602e43224fb56f303c73aced3a46d4024b9737d.jpg)

![](images/da87c4e0f4bf805d5ef6fd774a55135c3b8049af0ce12db94a1c33083d14a739.jpg)

![](images/820772c61143bbdc5e77f2cbccef7abc6982c18afefbd39cf77cc0564452d950.jpg)  
Figure 4: Four representative Gold-rated policy trajectories for Answer-Only SFT (blue) and Thinking GRPO rewarders (orange), including the smallest-policy exception, the 4B anchor and larger-policy/rewarder cases. All panels share the global Gold arena and vertical limits; lines connect recorded training snapshots, not fitted curves. All eight matched configurations are summarized in Table 3 and shown chronologically in Appendix A.

In the shared global Gold arena, Answer-Only rewarders give higher peak policy Elo in 7 of 8 matched configurations (Table 3). The exception is the 0.6B policy with a 4B judge: Thinking peaks at 1048 versus 1013 for Answer-Only. Averaging each run’s five highest-rated snapshots gives the same 7-of-8 pattern, with mean gaps of 107 Elo for top-five averages and 126 for peaks. Under 200 independent-row bootstrap resamples and regularized refits, 5 of 8 peak gaps and 5 of 8 top-five gaps have 95% percentile intervals above zero. This is conditional match-table uncertainty, not promptcluster, generation or retraining uncertainty (Appendix B.1). Figure 4 shows representative Goldrated training trajectories; Appendix A provides the complete matched set, including competitive

<table><tr><td>Policy</td><td>GenRM</td><td>Peak gap [95% CI]</td><td>Top-5 gap [95% CI]</td></tr><tr><td>0.6B</td><td>4B</td><td>-35 [−134, 49]</td><td>-19 [−59, 25]</td></tr><tr><td>1.7B</td><td>4B</td><td>+113 [43, 176]</td><td>+94 [48, 134]</td></tr><tr><td>4B</td><td>0.6B</td><td>+41 [−32, 104]</td><td>+24 [−2, 65]</td></tr><tr><td>4B</td><td>1.7B</td><td>+145 [104, 239]</td><td>+159 [133, 214]</td></tr><tr><td>4B</td><td>4B</td><td>+135 [49, 241]</td><td>+141 [93, 189]</td></tr><tr><td>4B</td><td>8B</td><td>+41 [-67, 143]</td><td>+25 [−18, 88]</td></tr><tr><td>4B</td><td>14B</td><td>+269 [128, 488]</td><td>+193 [134, 257]</td></tr><tr><td>8B</td><td>4B</td><td>+302 [158, 573]</td><td>+243 [189, 343]</td></tr></table>

Table 3: Matched Answer-Only versus Thinking Gold Elo gaps in the global all-configuration arena, reported as Answer-Only minus Thinking. Point gaps and intervals use the lightly regularized refit and 200 bootstrap resamples described in Appendix B.1; plotted trajectories use the original exported ratings. Answer-Only wins 7 of 8 matched cells under both summaries; the exception is the smallest-policy setting with a 4B judge.

Thinking settings. The reversal matters because static evaluation favors Thinking, and the ofthe-shelf baseline comparison also difers from the trained comparison. Under the learned Gold reference, the statically stronger practical judge-training pipeline is not generally the better online reward provider. The diference is consistent with diferent responses to policy-induced distribution shift, but does not identify that mechanism: Gold’s own preferences can also change with its inputs, and both judge training and response mode difer. Two observations help locate the gap without resolving its cause.

First, higher in-distribution GenRM accuracy does not necessarily translate to better policy training. Figure 3 compares intermediate 4B Thinking checkpoints at 0, 50, 100, 150, 200, 250 and 300 GRPO steps. The contemporaneous experiment account reports static accuracies of 59.7%, 61.5%, 64.5%, 68.7%, 71.5%, 79.2% and 82.3%, respectively. These are the checkpoint-ablation measurements, not the separate 83.2% recipe comparison above. The 250-step judge (79.2%) produces a stronger online rewarder than the later, more statically accurate checkpoint in this ablation. This illustrates that evaluator-optimal and rewarder-optimal checkpoint selection can difer. This late ablation used 40,000 policy-training prompts rather than the main 90,000.

Second, GRPO supervises final preferences rather than the correctness of each reasoning step. A Thinking GenRM may therefore learn traces that support training-distribution labels without remaining reliable under policy-induced shift; we treat this as a hypothesis, not a measured mechanism. The trajectory data show difering stability: across the eight matched global configurations, Answer-Only runs lose an average of 60 Elo from peak to final, compared with 130 for Thinking. An auxiliary verdict-agreement reconstruction could align rows only by checkpoint-player pairs and row order, not unique prompt/response identifiers. We do not use its percentages as verified response-level agreement or as mechanistic evidence.

These results should not be read as evidence that Answer-Only supervision preserves stylistic diversity. A surface audit of stored peak-checkpoint generations in two 4B-policy cells shows that high Gold Elo can coexist with template convergence: in the 4B-policy/4B-judge cell, 94% of Answer-Only peak outputs share the same opening line, and the duplicate-line rate is higher than in the corresponding Thinking peak outputs. The stronger Answer-Only result is therefore a preference-optimization result, not a diversity result. In summary, Answer-Only rewarders require shorter verdict generations and, in these runs, usually sustain higher Gold-rated policy quality. This does not independently establish greater human-quality robustness or identify reasoning as the cause. Intermediate Thinking checkpoints remain useful candidates for rewarder selection.

![](images/21ff3e1d18050ffee4f843119060b3cb9a4e702fb8d17485014659d20074e109.jpg)  
(a) Answer-Only GenRMs

![](images/c110170a3fc6f293aaae07cca3e730c0db266cfd7ff023607932b9a20a21d4c6.jpg)  
(b) Thinking GenRMs  
Figure 5: Rewarder-size sweeps with a fixed 4B policy, using the retained family-specific Gold arenas. Points are recorded snapshots connected in training order; squares/circles mark first/last observations. Own-proxy ratings come from separate rewarders and are not one calibrated scale. Larger Answer-Only rewarders improve Gold Elo beyond policy size; Thinking scaling is less monotone. The common-global-arena comparison is in Appendix A.

## 4.4 Efect of GenRM Size

Research question. Does scaling the reward model improve policy quality, and do gains continue when the GenRM exceeds the policy in size?

We investigate the efect of scaling GenRM size while fixing the policy size at 4B. Figure 5 shows results for both Answer-Only and Thinking GenRMs across scales from 0.6B to 14B. We observe clear gains from increasing GenRM size, especially for Answer-Only rewarders. With a 4B policy, the Answer-Only 14B judge is the strongest judge-size configuration in the global arena. Thinking rewarders also benefit substantially from scale, but less monotonically: the 8B Thinking judge slightly exceeds the 14B Thinking judge in peak Gold Elo (1507 vs. 1451) and top-5 average Gold Elo (1488 vs. 1437). Thus additional rewarder capacity can remain useful above policy size, while its benefit depends on the judge’s training and online behavior. Parameter counts alone do not establish a general ranking of the dificulty of judging and generating.

Appendix B.2 summarizes the existing sweep with descriptive scaling fits. A log-linear model using policy size, judge size and judge mode has an in-sample $R ^ { 2 }$ of 95.2% for top-5 Gold Elo across the 16 matched global configurations. In that fit, the coeficients correspond to 176 Elo per Answer-Only policy-size doubling and 136 per judge-size doubling; the Thinking-run values are 119 and 116 Elo. These are fitted summaries of this sparse grid, not predicted causal gains from a new doubling or a compute-optimal allocation rule.

<table><tr><td>Configuration</td><td>Top-5 avg Gold Elo</td><td>Peak Gold Elo</td><td>Peak step</td></tr><tr><td>GenRM-4B-Ans; Policy 8B</td><td>1645</td><td>1736</td><td>450</td></tr><tr><td>GenRM-14B-Ans; Policy 4B</td><td>1630</td><td>1719</td><td>430</td></tr><tr><td>GenRM-8B-Ans; Policy 4B</td><td>1513</td><td>1549</td><td>460</td></tr><tr><td>GenRM-8B-Think; Policy 4B</td><td>1488</td><td>1507</td><td>460</td></tr><tr><td>GenRM-4B-Ans; Policy 4B</td><td>1467</td><td>1488</td><td>420</td></tr><tr><td>GenRM-14B-Think; Policy 4B</td><td>1437</td><td>1451</td><td>440</td></tr></table>

Table 4: Top online-DPO configurations in the global all-configuration arena, ranked by average Gold Elo over each run’s best five checkpoints. The practical frontier is dominated by larger judges and Answer-Only rewarders, with the 8B Thinking judge remaining competitive.

![](images/331d1fba7e1e17b5fe1e9b5fc292eaf840a3a9c1ddaeb9fb60eca673924a83fd.jpg)  
(a) Answer-Only GenRMs

![](images/17782043d985422a6902c448f65d5197d2546048373432bf6be4814c4c855814.jpg)  
(b) Thinking GenRMs  
Figure 6: Policy-size sweeps with a fixed 4B rewarder, in the retained family-specific arenas. Dots follow training chronology; squares/circles mark first/last observations. Gold and own-proxy Elo are separately fitted, and heights across the two panels are not one common arena. Larger policies improve under Answer-Only supervision in this sweep; larger Thinking-supervised policies reach their Gold peak earlier.

## 4.5 Efect of Policy Size

Research question. Does scaling the policy improve quality, and does the efect difer between Answer-Only and Thinking supervision?

We examine the efect of scaling policy size while fixing the GenRM. Figure 6 shows results for policies ranging from 0.6B to 8B trained with both Answer-Only and Thinking GenRMs. For Answer-Only supervision, larger policies achieve higher Gold Elo across the studied scales. This measures increased quality under the learned reference, not independent evidence that larger poli cies generalize better to human preferences.

Under Thinking supervision, the largest policy achieves the highest peak in the policy-size arena but declines earlier. The 0.6B and 1.7B policies peak near the end of the recorded window, the 4B policy around step 320, and the 8B policy around step 130. Larger policies may exploit a rewarder’s weaknesses sooner, but these trajectories do not identify a mechanism or prove that additional policy capacity is intrinsically harmful. They show why judge capacity and policy capacity must be examined together, including training duration rather than only peak ratings.

![](images/ef5d0a81af321301602acd8fe77a626cafe5c5e3759b73ec7fb4046ba6b7e1f4.jpg)  
Figure 7: Online-DPO $\beta$ sweep under an Answer-Only rewarder. Points are retained snapshots connected in training order, not fitted smoothers; squares/circles mark first/last observations. Proxy and Gold Elo are separate fits. Smaller tested $\beta$ values reach higher Gold ratings, without strict monotonicity. The origina account specifies 40,000 training prompts for this sweep.

## 4.6 Efect of the $\beta$ Coeficient in Online DPO

Research question. How does DPO regularization afect Gold ratings, and what do the retained KL exports show?

We analyze the efect of the $\beta$ coeficient in Online DPO. Figure 7 shows proxy versus Gold Elo for $\beta \in \{ 0 . 1 , 0 . 0 2 , 0 . 0 1 , 0 . 0 0 5 , 0 . 0 0 2 \}$ . The smallest tested value, $\beta = 0 . 0 0 2 ,$ , gives the highest Gold Elo. The ordering is not strictly monotone: 0.02 slightly exceeds 0.01 in the complete stored trajectory. Lower $\beta$ does not simply increase proxy ratings at Gold’s expense: Gold performance also improves over the observed range. Figure 8 shows Gold and own-proxy ratings against retained KL export coordinates. The original beta-plot path applies Savitzky–Golay smoothing before exporting these coordinates; they are not raw per-snapshot KL measurements. Smaller tested $\beta$ values reach higher Gold ratings over this window, but the plot does not establish a precise $\mathrm { K L } -$ constrained frontier or a safe divergence threshold. The result favors smaller $\beta$ within the tested range, without identifying an optimum outside it. In the exported-KL subset, peak Gold Elo rises from 357 at $\beta = 0 . 1$ to 1122 at $\beta = 0 . 0 0 2$ , paired with export coordinates of 0.061 and 0.887. The complete stored $\beta = 0 . 1$ rating trajectory reaches about 378 Elo at step 260, beyond that subset. Negative export coordinates can arise from preprocessing; they are not evidence of negative true KL or attributed to finite-sample negativity of the released nonnegative estimator.

![](images/5362a8867cb0004f097b7b42a64484eec35fc1b651f08814ef0c2801148d8b87.jpg)

![](images/6f11566365af24da1dde92827f59719c947215d483d33991d822f064d12c92ee.jpg)  
Historical KL-axis exports; square: first, circle: last  
Figure 8: Gold and own-proxy ratings against historically smoothed KL export coordinates for the $\beta$ sweep. Lines connect exported observations in training order. The final step-260 rating has no retained KL coordinate and is omitted for each condition; no coordinate is synthesized. Negative exported values remain visible, not interpreted as raw nonnegative-estimator measurements.

## 5 Limitations and Discussion

Reference supervision and validation reuse. Gold is a 32B model fine-tuned on implicit, vote-derived preferences over human-written Reddit stories. Proxy labels then come from Gold, retaining nonzero sums of order-normalized verdicts; this can retain a pair with one vote parsed as invalid and does not eliminate reference-model bias. Gold validation, proxy validation and policy validation reuse the same 1,000 source pair records and their prompts, with diferent labels or generated responses. These measurements therefore do not constitute independent source-set replications. The downstream result is optimization toward a specified learned reference, not direct human adjudication of generated stories.

Evaluator dependence and response quality. Gold uses an Answer-Only Qwen3 model, and proxies share its model family and derived labels. Baseline comparisons do not rule out evaluator-specific preferences. A retained follow-up audit used a diferent learned judge, Qwen3- 235B-A22B-Instruct-2507, on 40 randomly selected validation prompts from each of two 4B-policy settings, comparing each mode’s peak-Gold outputs with randomized presentation order. Answer-Only won 42 of 80 comparisons (52.5%; 95% Wilson interval 41.7–63.1%): 17/40 with 4B rewarders and 25/40 with 14B rewarders. This limited audit is inconclusive overall and does not independently establish the Gold-ranked advantage. It uses another Qwen3 Answer-Only judge, a diferent rubric and Gold-selected checkpoints, not human evaluation or a cross-family replication. Appendix B.4 gives the procedure. Takeaway. The main result is conditional on the specified learned Gold reference; neither format neutrality nor universal writing-quality superiority is established.

Distribution-dependent influence of judge fine-tuning. One unresolved hypothesis is that, on shifted policy outputs, fine-tuned judges increasingly share the preferences of their original models. If the learned Answer-Only Gold evaluator and Answer-Only proxies share such priors, part of the measured advantage could reflect reference alignment rather than human preference. The retained Elo trajectories do not test this mechanism. A discriminating check would compare Gold with its own unfine-tuned 32B parent on identical human-story and early/late policy response pairs, controlling response order and output validity. Agreement alone would still not establish human quality or its decline. We report this as a hypothesis, not a result or an explanation established by this study.

Domain scope. The trained-rewarder sweeps use creative writing; we do not systematically evaluate safety, helpfulness or verifiable tasks such as math and code. An ancillary OpenOrca experiment optimized policies using of-the-shelf rewarders and a diferent reference evaluator (Appendix B.5). It does not replicate the trained creative-writing rewarders or establish transfer of the central result.

Relation to prior creative-writing studies. Optimizing preference win rate can reduce stylistic diversity [Chung et al., 2025]; the retained generations show style convergence under both judge modes. Fein et al. [2025] report that chain-of-thought can degrade verification accuracy for creative writing, and that trained BT and generative verifiers outperform zero-shot judges on their benchmark. Our 4B trace-distillation ablation gives a smaller static-accuracy gain than GRPO, and trained Thinking judges are the stronger static evaluators in our sweep. Their downstream reversal therefore cannot be inferred from a generic claim that thinking is unsuitable for creative-writing evaluation. It concerns the complete training and measurement loop.

Family and algorithm scope. All models use Qwen3 backbones; family-specific priors could afect the observed rankings and size slopes. Cross-family work also shows that behavioral priors can change RL outcomes [Gandhi et al., 2025]. We do not compare PPO-style policy optimization, alternative Thinking-judge training recipes or Best-of-N inference-time selection. The study concerns generative reward models as online-DPO training signals, not a compute-matched ranking of all ways to use a judge.

Elo anchoring and schedule. Ratings depend on evaluator, contestants, anchor and match schedule [Chiang et al., 2024]. Proxy and Gold ratings are fitted separately; the diagonal of a proxy– Gold plot is not a verdict-agreement statistic. We report global and size-stratified arenas. Bootstrap intervals for Answer-Only-minus-Thinking gaps independently resample stored winner/loser rows and refit a lightly regularized Bradley–Terry model. They are conditional on these stored comparisons, not robust to arbitrary dependence between rows sharing prompts or responses, and do not capture fresh generation, retraining, seed, anchor or tournament-design uncertainty. Takeaway. Interpret within-arena diferences alongside these dependence and selection limits, not as universal confidence statements.

Post-hoc fits and KL provenance. Appendix B.2 fits model-size associations to the stored grid, not seed-averaged laws or out-of-sample predictions. Retained KL coordinates come from historical analysis exports, with smoothing established for the beta plot and incomplete preprocessing provenance for other panels. Matching them to exported Gold ratings does not independently validate the original logger-to-step mapping. We preserve these coordinates for inspection but do not use them to identify a quantitative KL-constrained frontier.

## 6 Related Work

Reward overoptimization is studied in RLHF [Ouyang et al., 2022, Gao et al., 2022] and direct methods such as DPO [Rafailov et al., 2023, 2024]. Prior evaluations explicitly distinguish static accuracy from downstream policy performance [Wen et al., 2025, Frick et al., 2024, Kim et al., 2025]; the generic distinction is not new to this work. GenRMs replace scalar-head prediction with text generation, allowing verdicts and rationales [Mahan et al., 2024, Zhang et al., 2025]. Subsequent approaches include GRPO and positional consistency in J1 [Whitehouse et al., 2025], self-principled critique and aggregation in DeepSeek-GRM [Liu et al., 2025], and test-time aggregation [Wang et al., 2025, Shi and Jin, 2025]. Related supervision uses generated critiques [Yu et al., 2025], criteria trees [Liang et al., 2025] and reasoning plans [Saha et al., 2025]; judge reliability and static benchmarks remain active topics [Ye et al., 2024, Lambert et al., 2024].

Recent work also examines rationale-consistency supervision for GenRMs [Wang et al., 2026a] and rank-derived scalar rewards for GRPO [Wang et al., 2026b]. Our study instead uses direct pairwise preferences in online DPO to measure generative-rewarder capacity and practical Answer-Only SFT versus Thinking GRPO pipelines, with evaluator-specific Elo arenas [Chiang et al., 2024]. It does not test those interventions, isolate a causal efect of reasoning format, or establish their mechanisms as explanations of our results.

## 7 Conclusion

We study generative reward models as online training signals. Under our learned Gold reference, Thinking GenRMs outperform Answer-Only variants on static evaluation, yet this advantage usually reverses during policy optimization. The result concerns the compared SFT/GRPO pipelines in this domain, not an isolated causal efect of reasoning or a universal ranking of writing quality.

Three findings ofer practical guidance within these experiments: (1) reward model scale is a major driver of Gold-evaluated policy outcomes, with gains continuing even when the GenRM exceeds the policy in size; (2) intermediate GenRM checkpoints can outperform fully-trained models as rewarders; and (3) in-distribution accuracy does not by itself predict training efectiveness. Selecting GenRMs only by static benchmarks can therefore select checkpoints that are suboptimal for downstream optimization under the learned reference.

## References

Wei-Lin Chiang, Lianmin Zheng, et al. Chatbot arena: An open platform for evaluating llms by human preference. arXiv:2403.04132, 2024. URL https://arxiv.org/abs/2403.04132.

John Joon Young Chung, Vishakh Padmakumar, Melissa Roemmele, Yuqian Sun, and Max Kreminski. Modifying large language model post-training for diverse creative writing. arXiv preprint arXiv:2503.17126, 2025. URL https://arxiv.org/abs/2503.17126.

Daniel Fein, Sebastian Russo, Violet Xiang, Kabir Jolly, Rafael Rafailov, and Nick Haber. Litbench: A benchmark and dataset for reliable evaluation of creative writing. arXiv preprint arXiv:2507.00769, 2025. doi: 10.48550/arXiv.2507.00769. URL https://arxiv.org/abs/2507. 00769.

Evan Frick, Tianle Li, Connor Chen, Wei-Lin Chiang, Anastasios N. Angelopoulos, Jiantao Jiao, Banghua Zhu, Joseph E. Gonzalez, and Ion Stoica. How to evaluate reward models for rlhf, 2024. URL https://arxiv.org/abs/2410.14872.

Kanishk Gandhi, Ayush Chakravarthy, Anikait Singh, Nathan Lile, and Noah D. Goodman. Cognitive behaviors that enable self-improving reasoners, or, four habits of highly efective STaRs. arXiv preprint arXiv:2503.01307, 2025. doi: 10.48550/arXiv.2503.01307. URL https: //arxiv.org/abs/2503.01307.

Leo Gao, John Schulman, and Jacob Hilton. Scaling laws for reward model overoptimization. arXiv preprint arXiv:2210.10760, 2022. doi: 10.48550/arXiv.2210.10760. URL https://arxiv.org/ abs/2210.10760.

Shangmin Guo, Biao Zhang, Tianlin Liu, Tianqi Liu, Misha Khalman, Felipe Llinares, Alexandre Rame, Thomas Mesnard, Yao Zhao, Bilal Piot, Johan Ferret, and Mathieu Blondel. Direct language model alignment from online AI feedback. arXiv preprint arXiv:2402.04792, 2024. doi: 10.48550/arXiv.2402.04792. URL https://arxiv.org/abs/2402.04792.

Sunghwan Kim, Dongjin Kang, Taeyoon Kwon, Hyungjoo Chae, Dongha Lee, and Jinyoung Yeo. Rethinking reward model evaluation through the lens of reward overoptimization. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 13252–13280, 2025. URL https://aclanthology.org/2025.acl-long.649/.

Nathan Lambert, Valentina Pyatkin, Jacob Morrison, LJ Miranda, Bill Yuchen Lin, Khyathi Chandu, Nouha Dziri, Sachin Kumar, Tom Zick, Yejin Choi, Noah A. Smith, and Hannaneh Hajishirzi. Rewardbench: Evaluating reward models for language modeling. arXiv preprint arXiv:2403.13787, 2024. URL https://arxiv.org/abs/2403.13787.

Xiaobo Liang, Haoke Zhang, Juntao Li, Kehai Chen, Qiaoming Zhu, and Min Zhang. Generative reward modeling via synthetic criteria preference learning. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 26755– 26769, Vienna, Austria, July 2025. Association for Computational Linguistics. doi: 10.18653/ v1/2025.acl-long.1297. URL https://aclanthology.org/2025.acl-long.1297/.

Zijun Liu, Peiyi Wang, Runxin Xu, Shirong Ma, Chong Ruan, Peng Li, Yang Liu, and Yu Wu. Inference-time scaling for generalist reward modeling. arXiv preprint arXiv:2504.02495, 2025. doi: 10.48550/arXiv.2504.02495. URL https://arxiv.org/abs/2504.02495.

Dakota Mahan, Duy Van Phung, Rafael Rafailov, Chase Blagden, Nathan Lile, Louis Castricato, Jan-Philipp Fränken, Chelsea Finn, and Alon Albalak. Generative reward models. arXiv preprint arXiv:2410.12832, 2024. doi: 10.48550/arXiv.2410.12832. URL https://arxiv.org/abs/2410. 12832.

Long Ouyang et al. Training language models to follow instructions with human feedback. arXiv preprint arXiv:2203.02155, 2022. URL https://arxiv.org/abs/2203.02155.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. arXiv preprint arXiv:2305.18290, 2023. URL https://arxiv.org/abs/2305.18290.

Rafael Rafailov, Yaswanth Chittepu, Ryan Park, Harshit Sikchi, Joey Hejna, W. Bradley Knox, Chelsea Finn, and Scott Niekum. Scaling laws for reward model overoptimization in direct alignment algorithms. arXiv preprint arXiv:2406.02900, 2024. doi: 10.48550/arXiv.2406.02900. URL https://arxiv.org/abs/2406.02900. NeurIPS 2024.

Swarnadeep Saha, Xian Li, Marjan Ghazvininejad, Jason Weston, and Tianlu Wang. Learning to plan & reason for evaluation with thinking-llm-as-a-judge. arXiv preprint arXiv:2501.18099, 2025. URL https://arxiv.org/abs/2501.18099. Introduces FollowBenchEval.

Wenlei Shi and Xing Jin. Heimdall: Test-time scaling on the generative verification. arXiv preprint arXiv:2504.10337, 2025. doi: 10.48550/arXiv.2504.10337. URL https://arxiv.org/abs/2504. 10337.

Binghai Wang, Yantao Liu, Yuxuan Liu, Tianyi Tang, Shenzhi Wang, Chang Gao, Chujie Zheng, Yichang Zhang, Le Yu, Shixuan Liu, Tao Gui, Qi Zhang, Xuanjing Huang, Bowen Yu, Fei Huang, and Junyang Lin. Outcome accuracy is not enough: Aligning the reasoning process of reward models. arXiv preprint arXiv:2602.04649, 2026a. URL https://arxiv.org/abs/2602.04649.

Chenglong Wang, Ziming Zhu, Yifu Huo, Bei Li, Qiaozhi He, Yan Ding, Xiaoyang Hao, Yuxin Gao, Tianhua Zhou, Xiaojia Chang, Tongran Liu, Jingbo Zhu, Zhengtao Yu, and Tong Xiao. RRC: Unlocking generative reward models in LLM reinforcement learning via ranking-based reward construction. arXiv preprint arXiv:2608.06310, 2026b. URL https://arxiv.org/abs/2608. 06310.

Zhilin Wang, Jiaqi Zeng, Olivier Delalleau, Hoo-Chang Shin, Felipe Soares, Alexander Bukharin, Ellie Evans, Yi Dong, and Oleksii Kuchaiev. Helpsteer3-preference: Open human-annotated preference data across diverse tasks and languages. arXiv preprint arXiv:2505.11475, 2025. doi: 10.48550/arXiv.2505.11475. URL https://arxiv.org/abs/2505.11475.

Xueru Wen, Jie Lou, Yaojie Lu, Hongyu Lin, Xing Yu, Xinyu Lu, Ben He, Xianpei Han, Debing Zhang, and Le Sun. Rethinking reward model evaluation: Are we barking up the wrong tree? In The Thirteenth International Conference on Learning Representations, 2025. URL https: //openreview.net/forum?id=Cnwz9jONi5.

Chenxi Whitehouse, Tianlu Wang, Ping Yu, Xian Li, Jason Weston, Ilia Kulikov, and Swarnadeep Saha. J1: Incentivizing thinking in llm-as-a-judge via reinforcement learning. arXiv preprint arXiv:2505.10320, 2025. URL https://arxiv.org/abs/2505.10320.

Jiayi Ye, Yanbo Wang, Yue Huang, Dongping Chen, Qihui Zhang, Nuno Moniz, Tian Gao, Werner Geyer, Chao Huang, Pin-Yu Chen, Nitesh V. Chawla, and Xiangliang Zhang. Justice or prejudice? quantifying biases in llm-as-a-judge. arXiv preprint arXiv:2410.02736, 2024. URL https://arxiv.org/abs/2410.02736.

Yue Yu, Zhengxing Chen, Aston Zhang, Liang Tan, Chenguang Zhu, Richard Yuanzhe Pang, Yundi Qian, Xuewei Wang, Suchin Gururangan, Chao Zhang, Melanie Kambadur, Dhruv Mahajan, and Rui Hou. Self-generated critiques boost reward modeling for language models. arXiv preprint arXiv:2411.16646, 2025. doi: 10.48550/arXiv.2411.16646. URL https://arxiv.org/abs/2411. 16646.

Lunjun Zhang, Arian Hosseini, Hritik Bansal, Mehran Kazemi, Aviral Kumar, and Rishabh Agarwal. Generative verifiers: Reward modeling as next-token prediction. In ICLR, 2025. URL https://arxiv.org/abs/2408.15240. ICLR 2025.

## A Empirical Trajectories and Capacity Sweeps

## A.1 Trained and Of-the-Shelf Rewarders over Training

Figure 9 separates the common Gold evaluation from the four rewarders’ own-proxy ratings.

Trained and off-the-shelf rewarders: observed trajectories  
(a) One common Gold observer  
![](images/b0a84aca2ba75fc611ef2353267b819326a9d2074a6b03e8565f7844cd84d500.jpg)

(b) Own proxy: Trained Answer-Only  
![](images/12c3fd25114bf6ea3ae805db1829afd7e53898f25906998b3526f5f77224e192.jpg)

(c) Own proxy: Trained Thinking  
![](images/9fea1c11409d80765fc858b38729dbeb836461fbf269d58fed229b4a04f54232.jpg)

(d) Own proxy: Base Answer-Only  
![](images/a6b52f622d35279c1f26697ab13a52d607dac0d847d630c6a84eeff91e64b489.jpg)

(e) Own proxy: Base Thinking  
![](images/7d55b252f1b9638e53303d306bd68a4656ab1358fbba37ebef518b6de855d724.jpg)  
Figure 9: Trained versus of-the-shelf rewarders through the retained training horizon. Blue denotes Answer-Only and orange Thinking; solid lines denote trained rewarders and dashed lines of-theshelf rewarders. The upper panel uses one Gold arena; outlined points mark step 460. Each lower panel uses its own rewarder’s separately fitted ratings, so heights across those panels are not comparable. Policy and judge are 4B. Source: trained\_vs\_baseline/\*.csv; no smoothing or missing-step interpolation.

Trained rewarders achieve higher Gold peaks in both formats. For Thinking, the trained peak is 802.8 versus 542.3, but at shared step 460 the baseline leads 542.3 to 501.5. At 450 the ordering is opposite (414.3 versus 577.1). Only the baseline continues to 470. This late crossover is not a significance result or uniform baseline superiority. From 320 to 460, the trained Thinking policy’s own-proxy rating rises 1808.3 to 1971.5 while Gold falls 802.8 to 501.5: proxy–reference divergence, not an identified mechanism or human-quality decline.

## A.2 All Eight Matched Conditions in One Gold Arena

Figures 10 and 11 show the complete stored Gold-rated trajectories for all eight matched conditions. All panels use all\_params/gold.csv, sharing one arena and vertical scale. They expose early gains and late reversals hidden by best-checkpoint summaries; a falling tail does not itself measure verdict agreement or Gold-to-base reversion.

Gold-evaluated policy training trajectories

![](images/51555dc1eecadfcbdef118b0c0e265310ea039709a95c31ae8cc841379e02d96.jpg)

![](images/466cb9a2002bec61f7d8c3d6af2822bd0f5aceb1f70fe869242cd5c9efcfe792.jpg)

![](images/d3b602ea2f38cb9f167ce55a476f68c509e641ce99b51e64238972f41a839084.jpg)

![](images/9e2e15a2d207d476a00324452bd65cb88878304d6dea7438b377861603e6a729.jpg)  
Figure 10: Policy-size sweep with a fixed 4B rewarder. All four panels use the common Gold arena, common vertical limits and observed policy-training steps. Blue: Answer-Only SFT rewarders; orange: Thinking GRPO rewarders. Each line represents one retained policy trajectory, not a mean across seeds. Source: all\_params/gold.csv.

Gold-evaluated policy training trajectories  
![](images/c12711ff8fd54e8c6c734b64c8f96ca0913b829ec9552417274d134fc697be91.jpg)

![](images/a2405f8aa262beb12aba4d3c0644addceff2fdedfd21e373f0de9b9f658555ae.jpg)

![](images/9520f208e795d914b6b02477e78e3412850c43b01c0eb93eb35f5fdaa52d3a03.jpg)

![](images/3b055d226bbb737662e694a6c1eecabf07686eb88b70f0bd172d9341d3a8fc2f.jpg)  
Figure 11: Additional rewarder-size conditions with a fixed 4B policy. Together with the 4B/4B panel of Figure 10, these are the five rewarder sizes studied. Colors, arena and axis limits match the preceding figure. The retained curves show measured behavior, not a claim of monotonic improvement at every step or across every rewarder size.

## A.3 The Same Gold Observations against Exported KL Coordinates

Figures 12 and 13 provide a complementary view using the retained policy-KL analysis exports rather than training step. The exports do not contain explicit checkpoint IDs. For each exported Gold row, we require a unique match on its exact trajectory prefix and Gold rating (absolute tolerance 10<sup>−7</sup>) in the common-arena file. This recovers 752 exported rows. Four additional globalarena snapshots have no matching KL coordinate and are omitted, not assigned an inferred value. The machine-readable join ledger records every source row and recovered checkpoint.

These coordinates are preserved exports, not newly recovered raw per-step KL. The original analysis code smooths KL before export for the beta comparison; the exact preprocessing path for these matched panels is not fully reconstructed. The released trainer’s k3 estimator is nonnegative per token in exact arithmetic, so negative exported coordinates are not attributed to ordinary finitesample negativity of that estimator. They are retained rather than clipped. Our unique rating match identifies the exported observation, not a new validation of the original positional loggerto-step mapping. Lines follow the recovered rating checkpoint order even when a KL coordinate moves backward. These panels are ancillary descriptive views, not measurements of distance from Gold’s training distribution or a validated quantitative KL budget.

Policy 8B / judge 4B  
Policy 4B / judge 4B  
Gold trajectories against exported KL coordinates  
![](images/856826e52e439755eacb8badc34cc0e363a2df79292c0f3d1d53261f39f2c6fc.jpg)

![](images/8cba9ac6c6f18300fce48ae8806003e886a88cc91165d5d783c5fd835c1a7caf.jpg)

![](images/8e1c5f2cbab3c737a7ba62b48cf01f3a264730f7b46011037584a6946fe917c1.jpg)

![](images/86eee9b90431eecc837f6580e19cc17c1eb126d8251f73dfdd31c5a366983db6.jpg)  
Figure 12: Policy-size conditions against retained KL export coordinates. Gold coordinates are the same common-arena observations as Figure 10; only the horizontal coordinate changes. Lines follow the recovered checkpoint order. Each cell has 47 exported coordinates per mode; the two extra continuation snapshots in the 1.7B-policy cell lack a retained KL coordinate and are excluded. Horizontal limits difer across cells; vertical limits and the Gold arena are shared. Original KL preprocessing is not fully reconstructed.

Gold trajectories against exported KL coordinates  
![](images/2707c3a228e7bef149cbdee4f003f4f6eda342093a25b0325a77aac04fd6fd8e.jpg)

![](images/19c82e555085b8bb2cc1ad30ca78a650850ac960c5d61df176e3a52f01d447f9.jpg)

![](images/3f54cbcca235f5fbbdd3b836729f97ca9e1c7cf1d8c5b31409f4a0672b074862.jpg)

![](images/f7398686e6c863ab4080fba3102b41ae6e45188eadf25379aa239b23ad28c88f.jpg)  
Figure 13: Rewarder-size conditions against retained KL export coordinates. Gold coordinates match Figure 11. Only uniquely matched exported observations are shown. The two extra continuation snapshots in the 0.6B-rewarder cell are absent from these KL exports and are excluded. Negative exported values remain visible without being interpreted as raw KL measurements. This is not a fitted KL scaling law.

## A.4 Rewarder and Policy Capacity Sweeps

Figure 14 preserves both sweep directions: rewarder size at fixed policy size, and policy size at fixed rewarder size. The mean of the five highest Gold-rated snapshots is still selected under Gold, not an independently validated deployment rule. Only observed cells are shown.

Even parameter-count-matched swaps need not favor the same allocation in both formats. At a 12B summed parameter count, the 8B-policy/4B-rewarder condition has Answer-Only top-five Gold Elo 1645.0, versus 1513.1 for the 4B-policy/8B-rewarder condition. Thinking reverses that ordering (1402.2 versus 1488.1). A parameter sum is not an inference- or training-compute budget, especially when one rewarder generates reasoning traces; these observed swaps do not identify a universal compute-optimal allocation.

![](images/ee26474fea637112db29ae96f1ffc0171cbf15d3b83de5edcfc7e24c3e13ed10.jpg)

![](images/d9e46e824dc168c9a9478e27569ee677da700b21213973078ed70ef6667dee74.jpg)  
Figure 14: Observed capacity sweeps in the common Gold arena. Left: rewarder size varies while the policy is 4B. Right: policy size varies while the rewarder is 4B. Each point is the average of the five highest recorded Gold-rated snapshots of that trajectory. Parameter-size axes are logarithmic and labeled with actual sizes. Blue: Answer-Only; orange: Thinking. These are descriptive observed summaries, not compute-normalized optima, predictive scaling fits, or independent-seed confidence intervals.

Intermediate judge training. Figure 3 retains the intermediate-Thinking-checkpoint ablation. The original account specifies judge steps 0, 50, 100, 150, 200, 250 and 300 and a 40,000-policyprompt budget. We preserve that image rather than infer static accuracies or checkpoint identities from pixels; its source account does not substitute for an exact per-judge run manifest.

Reconstruction and provenance. The released analysis scripts and their provenance ledgers record source hashes, row counts, exported-coordinate joins, omissions and condition summaries. These six appendix figures introduce no model runs, new labels or significance tests.

## B Evaluation and Analysis Details

## B.1 Elo Estimation and Conditional Uncertainty

For each evaluator and arena, we fit Bradley–Terry ratings from its stored pairwise winner/loser records. With checkpoint ratings $R _ { i }$ , the fitted win probability is

$$
p _ { R } ( i \succ j ) = \sigma \bigg ( \frac { R _ { i } - R _ { j } } { s } \bigg ) , \qquad s = \frac { 4 0 0 } { \log 1 0 } .
$$

The anchored log-likelihood sums log $p _ { R } ( i \succ j )$ over observed wins, with one reference rating fixed at zero. Each evaluator and contestant set has a separate fit. Additive anchors do not afect withinarena diferences, but ratings from diferent arenas cannot simply be subtracted. In particular, the diagonal of a proxy–Gold plot marks equal numerical ratings, not identical preferences or calibrated agreement.

For the reported uncertainty analysis, we independently resample stored winner/loser rows and refit the ratings 200 times. The refits add an L2 penalty $\frac { \bar { 0 } . 0 1 } { 2 } \lVert \theta _ { - s _ { 0 } } \rVert ^ { 2 }$ in logit coordinates $\theta = R / s$ with reference $s _ { 0 }$ fixed. They are therefore closely related to, but not byte-identical with, the original exported ratings. For configuration c we report

$$
\operatorname { P e a k } ( c ) = \operatorname* { m a x } _ { t } R _ { c , t } , \qquad \operatorname { T o p } 5 ( c ) = \frac { 1 } { 5 } \sum _ { t \in \mathcal { T } _ { 5 } ( c ) } R _ { c , t } ,
$$

where ${ \mathcal { T } } _ { 5 } ( c )$ contains that run’s five highest-rated snapshots. Each resample repeats this selection before computing Answer-Only-minus-Thinking gaps; the reported 95% intervals are bootstrap percentiles.

These intervals describe sensitivity to the stored match table under an independent-row approximation. Comparisons sharing prompts or responses, or edges broken from one partial ranking, can be dependent. The retained winner/loser table does not preserve the cluster identifiers needed for a prompt-aware bootstrap. The intervals do not include fresh generations, independent training seeds, judge changes or tournament redesign. Peak and top-five selection use the evaluation data and should not be read as prospective checkpoint-selection guarantees.

## B.2 Descriptive Scaling Fits

We summarize the stored sweep with post-hoc fits, not a theory identifying neural optimization with a cardinal-utility model. For policy size $P _ { c } ,$ , rewarder size $R _ { c }$ (billions of parameters) and Thinking indicator $M _ { c }$ , ordinary least squares fits

$$
Y _ { c } = b _ { 0 } + b _ { P } \log P _ { c } + b _ { R } \log R _ { c } + b _ { M } M _ { c } + b _ { M P } M _ { c } \log P _ { c } + b _ { M R } M _ { c } \log R _ { c } + \epsilon _ { c } ,
$$

with $Y _ { c }$ the peak or top-five Gold Elo. There are 16 matched mode/size configurations and six fitted coeficients. The interaction terms permit diferent size slopes for the two pipelines; they do not isolate response mode from its training objective.
<table><tr><td>Metric</td><td> $R ^ { 2 }$ </td><td></td><td></td><td></td><td>RMSE Ans. policy Ans. judge Think policy</td><td>Think judge</td></tr><tr><td>Peak Elo</td><td>0.954</td><td>51.8</td><td>186.0</td><td>144.7</td><td>113.3</td><td>113.3</td></tr><tr><td>Top-5 Elo</td><td>0.952</td><td>51.2</td><td>175.8</td><td>135.7</td><td>119.0</td><td>115.6</td></tr></table>

Table 5: In-sample log-size fits to the 16 matched global configurations. Slope columns are fitted Gold-Elo diferences per parameter doubling along each axis, not observed new doublings, compute-matched gains or out-of-sample predictions. RMSE is in Elo units.

Both policy and judge scale have substantial fitted associations with Gold Elo in this grid. The high in-sample $R ^ { 2 }$ does not establish a universal scaling law: the grid is sparse, checkpoints were selected by the same learned evaluator and no independent-seed uncertainty is included.

Inputs, coeficients and the model-size fitting script are included in the public research artifacts. Earlier exploratory KL-curve fits remain in the research archive rather than serving as quantitative scaling evidence here: the exported horizontal coordinates include historical preprocessing whose complete provenance is not reconstructed.

## B.3 Why Static Accuracy Is Not an Online-Update Guarantee

A short update-level identity helps interpret the empirical distinction. Let $\Delta _ { \theta }$ denote the logprobability margin, relative to the reference policy, for a fixed Gold-oriented response pair. The Gold-label loss is $\ell _ { + } = - \log { \sigma ( \beta \Delta _ { \theta } ) }$ ; a flipped label gives $\ell _ { - } ~ = ~ - \log { \sigma } ( - \beta \Delta _ { \theta } )$ . Since $\sigma ( u ) +$ $\sigma ( - u ) = 1$

$$
\nabla _ { \boldsymbol { \theta } } \ell _ { - } - \nabla _ { \boldsymbol { \theta } } \ell _ { + } = \beta \nabla _ { \boldsymbol { \theta } } \Delta _ { \boldsymbol { \theta } } .
$$

Thus two rewarders with the same average error can induce diferent updates if their mistakes occur on pairs with diferent gradient directions or magnitudes. The relevant pairs are sampled from the evolving policy, not the fixed static validation distribution.

Filtering adds a second diference. The actual trainer sums order-normalized verdicts and retains nonzero-sum pairs; votes parsed as invalid contribute zero. Diferent proxies can therefore change both the labels of retained pairs and the distribution of pairs that survive. High static accuracy alone controls neither efect. This is a possible explanation of the evaluator–rewarder gap, not a measured decomposition: the retained artifacts do not supply the per-pair margins and gradient features needed to estimate it, nor establish which judge’s errors better track human preferences.

## B.4 Follow-Up Audit with a Diferent Learned Judge

We report an existing follow-up diagnostic separately from the main Gold arena. For 4B policies trained with 4B and 14B rewarders, we selected each mode’s peak-Gold snapshot and sampled 40 validation prompts per setting with seed 17. Response presentation order was randomized. The judge, Qwen3-235B-A22B-Instruct-2507, received the prompt and two responses without their training-mode labels. It judged prompt-following, coherence, specificity, originality and writing quality, with an instruction not to favor length. It produced one Answer-Only verdict per pair at temperature zero with a 24-token limit; no swapped-order repeat was performed. All 80 verdicts were parseable.

<table><tr><td>Rewarder size (4B policy) Answer-Only wins</td><td></td><td>Comparisons</td><td>95% Wilson interval</td></tr><tr><td>4B</td><td>17</td><td>40</td><td>28.5-57.8%</td></tr><tr><td>14B</td><td>25</td><td>40</td><td>47.0-75.8%</td></tr><tr><td>Combined</td><td>42</td><td>80</td><td>41.7–63.1%</td></tr></table>

Table 6: A diferent learned judge does not establish an overall preference advantage in this limited diagnostic. These descriptive binomial intervals are not prompt-cluster or retraining intervals; the 80 comparisons contain 76 unique prompts, with four prompts shared across settings.

The diagnostic is inconclusive overall. Its two settings, selected checkpoints, changed rubric, shared Qwen3 family and single verdict per pair limit interpretation. It neither reproduces a human evaluation nor supplies a thinking-style Gold control; it makes the evaluator dependence of the main result explicit.

## B.5 Ancillary Policy Optimization on OpenOrca

We retain an ancillary OpenOrca policy-optimization comparison. The recovered launch and plotting code identifies a 1.7B SFT-initialized policy followed by online DPO with of-the-shelf Qwen3 rewarders across several sizes and response modes. Its separate reference evaluator is a 32B Thinking model using a general response-quality rubric, not the learned Answer-Only Gold evaluator used for creative writing.

The retained plot reports per-step net preference rates against fixed reference responses: the number of wins minus losses, divided by the number of comparisons. Each point summarizes an evaluated policy state under its training rewarder and the separate reference evaluator. The exact numerical source-table revision for this image has not been reconstructed. We preserve it as ancillary historical evidence, not a new common-arena rating analysis, direct verdict-agreement measurement or replication with trained creative-writing GenRMs.

![](images/c7e62333dd33abcec4123b4d8b6039fe9f2568c86319f8cf47d63788a3514b7a.jpg)  
Figure 15: Retained OpenOrca policy-optimization trajectories with of-the-shelf rewarders. Axes show net preference rates, not Elo. Recovered source identifies online DPO from an SFT-initialized policy and a separate 32B Thinking reference evaluator. This is not a trained-GenRM cross-domain replication; exact numerical reconstruction of the retained image remains unavailable.

## C Prompt Templates

We provide the retained creative-writing prompt templates for policy generation and GenRM pairwise evaluation. Models apply their own tokenizers and chat templates; Thinking and Answer-Only settings additionally control whether a reasoning trace is generated. The ancillary OpenOrca experiment uses a diferent task prompt and reference rubric.

## C.1 Policy Prompt

The policy model receives the following prompt template for creative writing generation:

Creative Writing Prompt:   
{QUESTION}   
Goal: write an original short story that directly answers the prompt.   
Requirements: Output only the story text. no titles, notes, commentary   
or meta-text.

The QUESTION placeholder is replaced with the specific creative writing instruction from the evaluation set.

## C.2 GenRM Pairwise Evaluation Prompt

The generative reward model receives the following prompt template for pairwise comparison:

Two AI models were given the same instruction and each produced a reply.   
Your task is to judge which reply better fulfills the instruction.

[BEGIN INSTRUCTIONS GIVEN TO BOTH MODELS]   
{POLICY\_PROMPT}   
[END INSTRUCTIONS]   
[BEGIN REPLY A]   
{RESPONSE\_A}   
[END REPLY A]   
[BEGIN REPLY B]   
{RESPONSE\_B}   
[END REPLY B]   
Output ONLY one of:   
<answer>[[A]]</answer>   
<answer>[[B]]</answer>

The placeholders are filled with the full policy prompt and the two candidate responses. The model outputs a structured verdict indicating which response better satisfies the original instruction.