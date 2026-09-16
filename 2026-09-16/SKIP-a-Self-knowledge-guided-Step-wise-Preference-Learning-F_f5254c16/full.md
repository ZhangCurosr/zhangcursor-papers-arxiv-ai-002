# SKIP: a Self-knowledge-guided Step-wise Preference Learning Framework for Concise Reasoning

Qinhong Lin<sup>a</sup>, Yuhao Zhang<sup>a</sup>, Yinglun Feng<sup>a</sup>, Zhongliang Yang<sup>b,a,\*</sup>, Linna Zhou<sup>b,a,\*</sup>

<sup>a</sup>School of Cyberspace Security, Beijing University of Posts and Telecommunications, Beijing, China

<sup>b</sup>QuanCheng Laboratory, Jinan, China

{greenred99, yangzl, zhoulinna}@bupt.edu.cn

Abstract—While Chain-of-Thought (CoT) reasoning has been proven to be effective, it often leads to overthinking, resulting in computational overhead, inference latency, and even degraded performance in large language models (LLMs). Existing concise reasoning frameworks significantly compromise accuracy while compressing the length of output. In this paper, we propose SKIP, a self-knowledge-guided step-wise preference learning framework. Starting with lightweight fine-tuning to adjust the model’s output style, SKIP introduces a carefully designed knowledge probing mechanism to guide model to output an answer at each reasoning step. Based on the correctness of intermediate steps, we construct preference data that guide the model toward more efficient and correct reasoning by leveraging DPO. Experimental results demonstrate that our method effectively improves reasoning compression while mitigating performance degradation after fine-tuning. Besides, SKIP shows strong generalization ability on out-of-distribution datasets. We further conducted ablation studies on the component parameters of our framework.

Index Terms—consise reasoning, LLM, DPO, CoT.

## I. INTRODUCTION

Chain-of-Thought (CoT) reasoning has substantially enhanced the reasoning capabilities of large language models (LLMs), improving their performance on complex tasks such as mathematical reasoning [1]. This advancement holds great promise for enabling LLMs and agents to participate in human decision-making as powerful assistants [2]. Despite these advantages, deeper investigations into the mechanism of CoT reasoning have revealed that LLMs tend to engage in overthinking [3]. This tendency not only incurs additional computational overhead and inference latency but also undermines performance when the model is confident to answer question directly. Reference [4]’s comparative experiments on multiple tasks show that the gains brought by COT are related to the problem type. Empirical studies further show that LLMs exhibit a preference for lengthy answers [5], suggesting that they lack inherent constraints to avoid redundancy or encourage concise reasoning. Thus, various studies have been proposed that focus on compressing reasoning length without compromising model performance [6]. Existing research can be broadly categorized into the following two directions. One is inference-time intervention which tries to add control mechanisms like budget constraints or internal state probes at the model inference process. Another one is model capability elicitation. This line of work aims to encourage concise reasoning through data construction and training objectives. The training approaches, through sophisticated data construction and training, endows the model with the ability to simplify inference, making it more flexible without causing additional inference latency, and this capability can be transferred between different datasets. However, previous work mainly focuses on training models using data guided by context engineering or sampled from more powerful models. This drastic style shift may lead to length compression while severely compromising performance. In this study, we also focus on the latter direction to compress reasoning trace, aiming to tackle three core questions: Q1: How to efficiently construct data to elicit the model’s ability to concisely compress reasoning? Q2: How can we prevent the drop of reasoning accuracy during compression? Q3: Can this ability generalize to outof-distribution (OOD) datasets?

![](images/e4a9e855c68d5a5501a819759f9445bd72e36d8e8159e6c7242585d3303d03a0.jpg)  
Fig. 1. Step-wise probe accuracy analysis across reasoning progress. Bin Index represents various stages of the inference process, relative to the complete inference. In the GSM8K and Math datasets, the reasoning exhibits positive returns, while in MedQA, it shows negative returns.

To tackle these questions, we propose leveraging the model’s own knowledge to construct efficent datasets for learning which facilitates reasoning compression. Our motivation is driven by a key observation: when models are forced to decode an answer, such as when they encounter the signal ’The answer is’, they may be able to generate a correct response in certain situations.” As illustrated in Fig. 2, when ’The answer is’ is inserted at the first step of reasoning and the model is forced to generate an answer, it fails to provide the correct one because the necessary reasoning elements are not yet available. When “The answer is” is inserted at the second step, differently, the model is able to arrive at the correct answer after performing only simple calculations. Without such forced intervention, the model tends to continue producing redundant reasoning steps, thereby increasing the token budget. This phenomenon that some current reasoning paths indeed contain redundant steps, indicates that the model lacks the ability to recognize whether sufficient knowledge has been acquired during reasoning, and thus fails to terminate the reasoning process at the appropriate stage. As shown in Fig.1, we analyzed the step-wise probe accuracy on GSM8K, MATH, and MedQA with Llama3.1-8B-instruct and Llama3.2- 3B-instruct. On reasoning benchmarks (GSM8K, MATH), the likelihood of a correct response correlates positively with CoT length, though with diminishing returns in the final stages. In stark contrast, the experimental results on MedQA exhibit a negative trend, with accuracy deteriorating as reasoning proceeds, as models lack sufficient domain knowledge. These findings underscore the necessity of a self-knowledge probing mechanism capable of dynamically identifying the sufficiency of reasoning to determine the optimal early-exit boundary. In this work, we propose a Self-Knowledge-guIded step-wise Preference learning (SKIP) framework. At a high level, it’s a SFT-DPO two-stage training framework. We first design a simple yet efficient probe to identify model inference state after cold start and construct reasoning compression data for LLMs without relying on external knowledge. We then carefully mix sampled contrastive data to guide the model through preference learning, enabling the compression of reasoning paths without degrading reasoning performance. Experiments on both In-Distribution (ID) and Out-Of-Distribution (OOD) datasets show that the proposed method not only shortens reasoning length effectively but also improves the reasoning capability of LLMs. Our main contributions are as follows:

![](images/0811863edb488c2623078f184b785fbbe2c6e0c6e583fa4213f9122d5e2e6d48.jpg)  
Fig. 2. Self-Knowledge-guIded step-wise Preference learning (SKIP), a three stage reasoning compression framework. We use a simple yet efficient probe to detect whether the model can answer the question. We then train models with preference data to teach them to learn when to exit at specific points.

• We propose a model-free reasoning boundary probing mechanism that efficiently determine whether the model already has the ability to output the correct answer.

• We propose a data construction scheme that balances knowledge preference and length preference, which can be used to train models to improve inference length compression capabilities while mitigating performance degradation under direct preference optimization training.

• Extensive experiments across different tasks and models prove the robutness of SKIP. Besides, we conduct ablation studies on the training data to disentangle and analyzed its specific impact on model performance.

Our data/code is available at https://github.com/linqinhong/ SKIP.

## II. RELATED WORK

Chain-of-Thought Reasoning. Chain-of-Thought (CoT) [7] reasoning has proven to be a transformative paradigm for achieving human-like cognition and solving complex tasks, as exemplified by the success of reasoning-centric models such as OpenAI o1 [8], Gemini-3.0-Pro [9], and DeepSeek-R1 [10]. During model training, CoT is integrated with techniques like Best-of-N (BoN) [11] and Monte Carlo Tree Search (MCTS) [12], along with the variant, Tree-of-Thought (ToT), to systematically explore the distribution space and sample high-quality synthetic data. Furthermore, the paradigm shift in Reinforcement Learning (RL) from outcome-based rewards [11] to process-based rewards [13] highlights how step-bystep reasoning facilitates deeper cognition, pushing model applications into more specialized and sophisticated domains. Concise Reasoning. Concise Reasoning aims to mitigate the prohibitive computational overhead and the ”overthinking” problem. One primary approach involves inference-time interventions to manage computational budgets. For instance, soft control methods, such as TALE [14], inject budget constraints into the prompt to guide the model toward shorter reasoning paths. In contrast, hard control strategies, such as [15], [16], employ a prober to predict whether explicit reasoning is necessary for a given task based on the internal states. Another line of work focuses on inducing more concise reasoning styles within the model itself. For instance, FS-BoN [17] leverages few-shot examples to demonstrate brevity, while C3OT [18] utilizes proprietary APIs to distill and compress reasoning steps. These methods typically involve fine-tuning the model to shift its inherent reasoning trajectory toward conciseness. Other studies such as [19]–[21] design lengthconstrained reward functions within Reinforcement Learning (RL) frameworks to penalize excessive verbosity.

Building on these insights, our work leverages internal state probing to facilitate the construction of high-quality training data. We introduce an explicit mechanism to identify knowledge-sufficient states, enabling the model to strategically bypass redundant reasoning trajectories when its internal information is already adequate.

## III. METHODOLOGY

As illustrated in Fig. 2, the SKIP framework consists of three stages. First, we leverage a small set of answers with concise reasoning to supervised fine-tune (SFT) the model to learn a more succinct response style. Next, we employ a stepwise knowledge probing mechanism to determine whether the model has already acquired sufficient knowledge to produce the correct answer, and leverage this assessment as guidance for preference-based data construction. Finally, we apply masked-DPO training to inject this capability into the model.

## A. SFT Cold Start

Unaligned models often produce answers with inconsistent formats. To address this issue, we first perform SFT coldstart training with two objectives: (i) encouraging the model to output a final summary answer and terminate unnecessary reasoning whenever “The answer $\mathrm { i } \mathrm { s } ^ { \flat }$ appears at the end of a reasoning step, and (ii) guiding the model to adopt a more concise response style, thereby providing a better foundation for subsequent DPO training. Specifically, we begin with $N = 8$ question–answer pairs that end with a “The answer $\mathrm { i s } ^ { \prime \prime }$ format as few-shot conditions, and employ in-context learning (ICL) [22] to elicit the model to generate additional pairs in the same style. We filtered data according to correctness and then fine-tune the model via SFT to internalize this reasoning style. In practice, we found that this approach effectively achieves the intended objectives, but also reveals a trade-off between efficiency and reasoning capability: while the reasoning path is compressed, the model’s reasoning performance may experience slight degradation.

## B. Self-Knowledge-Guided Preference Data

Inference Boundary Probing. After supervised fine-tuning, the model becomes more inclined to output the final answer following a “The answer is” signal. The key to effectively reducing the model’s reasoning length while minimizing performance degradation lies in the model’s ability to recognize when the generated reasoning path is complete. To achieve this, we employ a simple and effective probe to guide the model within its sampling space. Specifically, during the reasoning process, we use a probe to guide the model to decode and output an answer based on its existing reasoning. $\mathrm { S o } .$ we sample candidates $Y = \{ y _ { 0 } , y _ { 1 } , . . . y _ { i } \}$ for the question Q. Each candidate can be formulate as:

$$
y _ { i } = M ( Q , t _ { 0 } , t _ { 1 } , . . . , t _ { i - 1 } , \sqrt { p r o b e } ) )\tag{1}
$$

where i denotes the timestep and t denotes the token. Greedy decoding serves as a reliable proxy for the model’s explicit reasoning ability. We approximate a greedy-decoded answer as a representation of the model’s internal state. We then assess the correctness of $y _ { i }$ to determine whether the model has acquired sufficient reasoning elements (without confusion, all answers guided by probes are generated through greedy decoding). Therefore, producing the correct answer in this setting indicates that the model has acquired a stable answering capability. Technically, inserting probes at any position is permissible. However, considering that a sentence is the smallest effective unit of the model’s reasoning chain, we only insert probes after each sentence. Therefore, the input format to the model can be represented as: $^ { , , } Q \backslash n [ S t e p _ { 1 } ] \backslash n [ S t e p _ { 2 } ] \backslash n . . . \backslash n [ S t e p _ { i } ] \backslash n ] p r o b e \ \forall ^ { , }$ The $\boxed { p r o b e }$ can be any phrase that prompts the model to output an answer. We use ’The answer is’ to align with the cold start template, ensuring output quality.

Though models almost always produce a correct answer when the preceding reasoning steps are already logically sufficient, it’s important to note that the ability to decode the correct answer at an early step does not guarantee that subsequent reasoning steps will remain correct.This may be because the model, in some cases, has already completed the reasoning implicitly in its internal representations, and may introduce instability or cause logical errors to accumulate when generate additional explicit steps [23]. Please refer to App. C.

Step-wise Preference. Our objective of this stage is twofold: (i) to preserve the reasoning capability of the model as much as possible, and (ii) to encourage the model to terminate reasoning once the existing steps are sufficient to produce the correct answer. As illustrated in Fig. 2, we categorize the sampled answers into four types based on their correctness and length: longer and right, shorter and wrong, longer but wrong and short and right. In this work, we focus on collecting preference data from two scenarios: knowledge preference data and length preference data. When the model produces a correct answer at $s t e p _ { i }$ but fails to do so at one of $\{ s t e p _ { t } | t < i \}$ this indicates that the model has strengthened its reasoning in the additional steps, which represents effective reasoning. We want the model to learn the correct reasoning logic from this. Therefore, we construct knowledge preference data with the relation longer and right ≻ shorter and wrong. When the model successfully produces a correct answer at step<sub>i</sub> and also does so at one of $\{ s t e p _ { t } | t > i \}$ , this indicates that both reasoning paths are valid. In this case, the extra reasoning in the longer answer should be considered redundant, and we want the model to learn a more concise reasoning pattern from this comparison. Thus, we construct length preference data with the relation longer and right < shorter but $r i g h t$ In our experiments, we found that the length preference data has a significant impact on the model, leading to a rapid compression of length but also causing a sharp decline in performance. Therefore, we consider a correct answer at $s t e p _ { i }$ only when it is accompanied by an incorrect answer at $s t e p _ { t < i }$ and a correct answer at $s t e p _ { t > i }$ . We analyze this situation in the App. B. Additionally, the reverse case of knowledge preference data, where the longer answer is wrong and the shorter one is correct, suggests that the additional reasoning steps caused an error in an otherwise correct answer. We want the model to terminate the reasoning process in a timely manner. Therefore, we treat longer but wrong $\succ$ shorter but wrong as a form of length preference data. It is worth noting that we found the sample size for this case to be very small. It is important to note that we discard the long but wrong VS short and wrong data, as it does not provide effective signals for training.

Masked-DPO. Given x representing the question, $y _ { w }$ representing the preferred data and y representing the rejected data,we fine-tuned models using Direct Preference Optimization (DPO) [24] with the following loss:

$$
\mathcal { L } ( \pi _ { \theta } , \pi _ { r e f } ) = - \mathbb { E } _ { D } [ \beta \log \sigma ( \log ( \frac { \pi _ { \theta } ( y _ { w } | x ) } { \pi _ { r e f } ( y _ { w } | x ) } / \frac { \pi _ { \theta } ( y _ { l } | x ) } { \pi _ { r e f } ( y _ { l } | x ) } ) ) ]\tag{2}
$$

where there is $( x , y _ { w } , y _ { l } ) \sim D$ . During training, we apply a masking strategy to the shared parts of the reasoning steps and question content between the chosen and rejected samples. By training only on the non-overlapping segments, we improve the stability of the optimization process. To improve training efficiency, we fine-tune only the LoRA [25] parameters. After training, we merged LoRA with the models for inference, which introduce no inference latency.

## IV. EXPERIMENTS

Models and Datasets. We conducted experiments with Llama-3.1-8B-Instruct [26], Llama-3.2-3B-Instruct [27] and Qwen-3- 14B [28] models on two mathematical reasoning benchmarks that require multi-step reasoning to reach the final solution: GSM8K [29] and MATH [30] and one domain dataset MedQA [31]. To further evaluate the generalization ability of our method, we also performed out-of-distribution (OOD) evaluations on StrategyQA [32] and the Date Understanding task from BIG-Bench Hard [33]. In our experiments, we used the full GSM8K dataset (7,473 training and 1,319 test examples). For MATH, we trained on all 7,500 examples and tested on the first 500. We evaluated StrategyQA on a 500-example subset and BBH’s Date Understanding task on 250 test cases.

Baselines. We adopted one inference-time intervention method, Estimated Budget method [14]. For model capbility elicitation methods, we considered the Best-of-N (BON) sampling methods in [17] and the C3OT method in [18].

BON generated training fine-tuning data through few-shot prompting, while C3OT leveraged a stronger closed-source model to compress sampled reasoning trajectories for training. We adopted a prompt-guided CoT approach [34] without any additional control as our default baseline. To better demonstrate the effectiveness of our self-knowledge-guided preference data, we adopted the BON training pipeline as the cold-start setting with $N = 1$ and $N = 8$ . While BON utilized the entire training set during SFT, we only used half of the data for SFT and constructed preference pairs from the remaining half. For each setting, we ran the experiments five times and report the mean results.

Evaluation Metrics. Our evaluation metrics included final answer accuracy and average reasoning length. Our goal was to achieve greater reasoning compression with minimal loss in accuracy. In our experiments, we judge answer via regular expression extraction and Exact Match evaluation. We report the consistency of the evaluation results between this approach and LLM-as-a-Judge [35] in the App. A. We also compared the compression efficiency defined as:

$$
E f f = { \frac { A c c u r a c y } { A v e r a g e \ L e n g t h } } ,\tag{3}
$$

which calculate the accuracy brought by each token.

## A. ID results and OOD results

In Tab. IV, We report the results on the in-distribution (ID) datasets. We summarize the key findings as follows:

(1) Our SKIP-8 achieves the highest token efficiency in nearly all experiments, except for one setup where it ranks second. All finetuning methods effectively compress the reasoning length of LLMs compare to the Estimated Budge method. Moreover, in all experiments, both SKIP-1 and SKIP-8 show significant improvements compared to BON-1 and BON-8 with similar training data sizes. Our results show that SKIP framework directly address Q1: How to efficiently construct data to elicit the model’s ability to concisely compress reasoning? We show a case study in Appendix C.

(2) Two baselines other than BON are also able to reduce reasoning length. However, this comes at the cost of a significant drop in accuracy. By incorporating knowledge preference data, SKIP not only compresses reasoning more effectively but also improves accuracy, providing a positive answer to Q2: How can we prevent the drop of reasoning accuracy during compression? It demonstrates that combining knowledge preference data with length preference data is a simple, black-box, yet efficient method.

(3) C3OT could achieves a higher compression ratio than BON methods in some settings, particularly on the GSM8K and Math dataset with Llama-3.1-8B. We refer it to the reason that C3OT leverages API-based sampling with powerful closed-source models to identify redundant reasoning steps during training, thereby achieving strong compression. By contrast, BON relies on in-context learning to guide concise reasoning, which is, to some extent, constrained by the capacity of the base model. This limitation becomes more apparent on challenging datasets like MATH, where C3OT clearly outperforms in terms of compression. However, there is a sharp decline in accuracy with C3OT, indicating that training data from other models may not align well with the model’s historical knowledge and capabilities. This suggests that transferring expression styles incurs greater costs. Unlike BON, SKIP utilizes data from distribution sampling, and does not show any significant deterioration in correctness.

TABLE I  
RESULTS OF DIFFERENT REASONING COMPRESSION METHODS. THE BOLD TEXT REPRESENTS THE BEST RESULT FOR THE CORRESPONDING METRIC IN THE BLOCK, AND THE TEXT WITH A <sup>†</sup> REPRESENTS THE SECOND BEST RESULT FOR THE CORRESPONDING METRIC IN THE BLOCK.
<table><tr><td rowspan="2">model</td><td rowspan="2">method</td><td colspan="3">GSM8K</td><td colspan="3">Math</td><td colspan="3">MedQA</td></tr><tr><td>Acc</td><td>Length</td><td>Eff</td><td>Acc</td><td>Length</td><td>Eff</td><td>Acc</td><td>Length</td><td>Eff</td></tr><tr><td rowspan="7">Llama-3.2-3B</td><td>Default</td><td>76.70</td><td>217.70</td><td>0.35</td><td>47.20</td><td>503.20</td><td>0.09</td><td>56.83</td><td>482.36</td><td>0.12</td></tr><tr><td>Budget</td><td>72.30</td><td>141.50</td><td>0.51</td><td>44.60</td><td>438.20</td><td>0.10</td><td>40.35</td><td>339.80</td><td>0.12</td></tr><tr><td>C3OT</td><td>68.80</td><td>115.30†</td><td>0.60</td><td>39.60</td><td>320.50</td><td>0.12</td><td>54.79</td><td>101.80</td><td>0.54</td></tr><tr><td>BON-1</td><td>77.30</td><td>149.30</td><td>0.52</td><td>43.60</td><td>392.00</td><td>0.11</td><td>56.83</td><td>99.14</td><td>0.57</td></tr><tr><td>SKIP-1</td><td>77.70</td><td>120.10</td><td>0.65†</td><td>45.30†</td><td>376.30</td><td>0.12</td><td>58.08</td><td>97.47</td><td>0.60</td></tr><tr><td>BON-8</td><td>78.90†</td><td>130.30</td><td>0.61</td><td>44.20</td><td>333.00</td><td>0.13†</td><td>58.50</td><td>83.00†</td><td>0.70†</td></tr><tr><td>SKIP-8</td><td>79.20</td><td>114.00</td><td>0.69</td><td>43.80</td><td>321.70</td><td>0.14</td><td>58.18†</td><td>74.19</td><td>0.80</td></tr><tr><td rowspan="7">Llama-3.1-8B</td><td>Default</td><td>85.50</td><td>240.40</td><td>0.36</td><td>46.60</td><td>479.50</td><td>0.10</td><td>52.70</td><td>278.00</td><td>0.19</td></tr><tr><td>Budget</td><td>80.40</td><td>149.50</td><td>0.54</td><td>46.20</td><td>440.20</td><td>0.10</td><td>56.51</td><td>433.92</td><td>0.13</td></tr><tr><td>C3OT</td><td>77.20</td><td>109.30</td><td>0.71†</td><td>37.40</td><td>302.70</td><td>0.12†</td><td>66.30</td><td>110.80</td><td>0.60</td></tr><tr><td>BON-1</td><td>81.60</td><td>154.00</td><td>0.53</td><td>46.30</td><td>418.50</td><td>0.11</td><td>67.50</td><td>87.50</td><td>0.77</td></tr><tr><td>SKIP-1</td><td>82.90</td><td>115.20†</td><td>0.72</td><td>47.80</td><td>398.20</td><td>0.12†</td><td>67.66†</td><td>87.51</td><td>0.77</td></tr><tr><td>BON-8</td><td>83.40</td><td>127.10</td><td>0.66</td><td>45.10</td><td>374.30</td><td>0.12†</td><td>67.00</td><td>76.00†</td><td>0.88†</td></tr><tr><td>SKIP-8</td><td>84.35†</td><td>119.00</td><td>0.71†</td><td>46.70†</td><td>367.90†</td><td>0.13</td><td>68.90</td><td>73.10</td><td>0.94</td></tr><tr><td rowspan="7">Qwen3-14B</td><td>Default</td><td>88.93</td><td>473.57</td><td>0.19</td><td>70.60</td><td>983.48</td><td>0.07</td><td>59.65</td><td>958.50</td><td>0.06</td></tr><tr><td>Budget</td><td>82.18</td><td>359.94</td><td>0.23</td><td>41.10</td><td>855.60</td><td>0.05</td><td>47.10</td><td>685.54</td><td>0.07</td></tr><tr><td>C3OT</td><td>86.84</td><td>132.84</td><td>0.65†</td><td>65.40</td><td>700.47</td><td>0.09</td><td>71.74</td><td>439.67</td><td>0.17</td></tr><tr><td>BON-1</td><td>93.86†</td><td>187.48</td><td>0.50</td><td>76.20</td><td>513.25</td><td>0.15</td><td>72.53 †</td><td>255.00</td><td>0.28</td></tr><tr><td>SKIP-1</td><td>95.45</td><td>152.47</td><td>0.63</td><td>84.40</td><td>359.00†</td><td>0.24†</td><td>72.90</td><td>164.00</td><td>0.31</td></tr><tr><td>BON-8</td><td>92.65</td><td>161.02</td><td>0.58</td><td>75.40</td><td>487.45</td><td>0.15</td><td>71.74</td><td>224.44</td><td>0.32†</td></tr><tr><td>SKIP-8</td><td>93.27</td><td>138.28†</td><td>0.67</td><td>81.20†</td><td>319.00</td><td>0.25</td><td>71.80</td><td>190.00†</td><td>0.38</td></tr></table>

TABLE II

OOD EXPERIMENT RESULTS ON LLAMA-3.1-8B. THE BOLD SCORESDENOTE THE BEST PERFORMANCE.
<table><tr><td rowspan="2">Train data</td><td rowspan="2">Method</td><td colspan="3">BBH(DU)</td><td colspan="3">StrategyQA</td></tr><tr><td>Acc</td><td>Len</td><td>Eff</td><td>Acc</td><td>Len</td><td>Eff</td></tr><tr><td></td><td>Default</td><td>76.8</td><td>257.5</td><td>0.298</td><td>69</td><td>303.8</td><td>0.227</td></tr><tr><td rowspan="4">GSM8K</td><td>BON1</td><td>66.9</td><td>125.2</td><td>0.534</td><td>51.3</td><td>156.5</td><td>0.328</td></tr><tr><td>SKIP-1</td><td>72.2</td><td>141.7</td><td>0.509</td><td>63.1</td><td>124.8</td><td>0.505</td></tr><tr><td>BON8</td><td>62.4</td><td>88.3</td><td>0.706</td><td>56.3</td><td>142.3</td><td>0.395</td></tr><tr><td>SKIP-8</td><td>67.0</td><td>85.1</td><td>0.787</td><td>64.1</td><td>127.9</td><td>0.501</td></tr><tr><td rowspan="4">MATH</td><td>BON1</td><td>67.4</td><td>147.6</td><td>0.456</td><td>57.5</td><td>241.3</td><td>0.23</td></tr><tr><td>SKIP-1</td><td>68.0</td><td>143.5</td><td>0.473</td><td>59.8</td><td>238.8</td><td>0.250</td></tr><tr><td>BON8</td><td>69.0</td><td>93.6</td><td>0.737</td><td>65.1</td><td>157.7</td><td>0.412</td></tr><tr><td>SKIP-8</td><td>70.8</td><td>91.8</td><td>0.771</td><td>65.3</td><td>158.0</td><td>0.413</td></tr></table>

In Tab. II, we presented results on out-of-distribution (OOD) datasets. We exclude C3OT and Estimated Budget from this comparison due to their substantial performance degradation on ID datasets, which diminishes the value of extending them to OOD scenarios. The key findings are:

(1) The concise reasoning ability acquired through SFT generalizes effectively to OOD datasets. However, this also comes with a notable drop in accuracy compared to the Default

model.

(2) SKIP consistently outperforms BON across all three evaluation metrics. This demonstrates that our constructed preference data not only encourages concise reasoning but also enhances the model’s reasoning capability, even in OOD scenarios. These two provide an answer to our research question Q3: Can this ability generalize to out-of-distribution (OOD) datasets?

## B. Ablation Study

In this section, we analyze the impact of the proposed preference data by examining the effects of data mixing ratios, training data size, and masked-DPO on model performance. Data mixing ratio In Sec. III-B, we categorize the constructed data into knowledge preference data and length preference data based on forced decoding and the correctness of intermediate reasoning steps. In this section, we explicitly examine their respective effects on model accuracy and reasoning length after training. As shown in Fig. 3(a), increasing the proportion of knowledge preference data in the mixture leads to longer reasoning chains and higher accuracy. Knowledge preference data is derived from detecting incorrect outputs in forced decoding at earlier steps, encouraging the model to learn that sufficient knowledge must be accumulated before producing an answer. In contrast, length preference data plays the opposite role: it guides the model to avoid overthinking once enough knowledge has been acquired. Together, these two types of data form an adversarial learning process that balances reasoning sufficiency with conciseness.

Training data size. In Fig. 3(b), we analyze the impact of training data size on the final performance. Specifically, we use

TABLE III  
THE DIFFERENCE BETWEEN DPO AND MASKED-DPO ON LLAMA-3.1-8B. MASKED-DPO COULD ACHIEVE BETTER RESULTS.
<table><tr><td rowspan="2">Method</td><td rowspan="2">DPO-type</td><td colspan="2">GSM8K</td><td colspan="2">MATH</td></tr><tr><td>Acc</td><td>Len</td><td>Acc</td><td>Len</td></tr><tr><td rowspan="2">SKIP-1</td><td>DPO</td><td>82.18</td><td>145</td><td>47.1</td><td>412</td></tr><tr><td>Masked-DPO</td><td>82.31</td><td>142</td><td>47.8</td><td>401</td></tr><tr><td rowspan="2">SKIP-8</td><td>DPO</td><td>83.96</td><td>119</td><td>45.8</td><td>367.7</td></tr><tr><td>Masked-DPO</td><td>84.35</td><td>119</td><td>46.68</td><td>366.9</td></tr></table>

![](images/329714700cfeb08ce0def1ef17597bf4fa6164b851af2980cc61f5cf6c0d96db.jpg)

![](images/8cad89b47ae702249e6e635c9d44cd76e66e3a16c3d18b244c6d9b241ce63ffd.jpg)  
Fig. 3. a.We compared different proportions of knowledge preference data and length preference data to analyze their impact on model performance. b.We analyze the impact of training data size on model performance.

25%, 50%, 75%, and 100% of the half portion for preference data, as described in the experiment setup. As the amount of data increases, the model’s accuracy gradually improves while the length decreases steadily with more data. This aligns with the trend in model training and also reflects that our data effectively balances reasoning length and accuracy.

Training Objective. In the constructed step-wise preference data, the chosen and rejected samples share a common prefix in addition to the question itself. To prevent this shared prefix from interfering with training, we apply a masking strategy during DPO training to exclude it. In Tab. III, we compare the two training settings. Experiments on LLaMA-3.1-8B show that masked-DPO yields shorter lengths and higher accuracy.

## V. CONCLUSION

In this work, we presented SKIP, a self-knowledge-guided step-wise preference learning framework that leverages intermediate reasoning signals to construct knowledge and length preference data.By combining SFT cold start with step-wise probing, SKIP efficiently detects the model’s reasoning critical points and constructs preference data for DPO training, enabling the model to learn more efficient concise reasoning abilities to generate shorter reasoning chains without sacrificing, and even improving, answer accuracy. Our experiments further demonstrate that this capability generalizes effectively to out-of-distribution datasets.

## VI. LIMITATIONS

Although we have demonstrated that SKIP exhibits stronger capabilities in concise reasoning, we acknowledge that there are areas in this paper that need further enhancement, which we leave to future work.

More Robust Efficiency Metrics: In this paper, we measure the model’s compression efficiency by calculating the accuracy contribution of each individual token. However, we believe that the difficulty of improving model accuracy is influenced by diminishing returns—the closer the model approaches its performance threshold, the harder it becomes to achieve further improvements. The same applies to length compression. Therefore, finding a balance between both factors for a more robust analysis of the model’s concise reasoning capability is an important direction for future research.

Broader Evaluation: Although the evaluation in this work covers a broad range, including various types and topics of test questions, it is necessary to apply and evaluate the method on a wider range of data.

## ACKNOWLEDGEMENT

This work was supported in part by Research Project of Quan Cheng Laboratory, China (Grant No.QCL20250203)and in part by the National Natural Science Foundation of China under Grant 62172053 and Grant 62302059.

## REFERENCES

[1] S. Imani, L. Du, and H. Shrivastava, “Math-prompter: Mathematical reasoning using large language models,” in Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics(ACL), vol. 5, pp. 37–42, July 2023.

[2] X. Zhang, C. Du, T. Pang, Q. Liu, W. Gao, and M. Lin, “Chain of preference optimization: Improving chain-of-thought reasoning in LLMs,” in Advances in Neural Information Processing Systems(NIPS), vol. 37, pp. 333–356, 2024.

[3] X. Chen, J. Xu, T. Liang, Z. He, J. Pang, D. Yu, et al., “Do NOT Think That Much for 2+3=? On the Overthinking of Long Reasoning Models,” in Proceedings of the 42nd International Conference on Machine Learning(ICML), 2025.

[4] Z. Sprague, F. Yin, J. D. Rodriguez, D. Jiang, M. Wadhwa, P. Singhal, et al, “To cot or not to cot? chain-of-thought helps mainly on math and symbolic reasoning,” 2024, arXiv:2409.12183. [Online]. Available: https://arxiv.org/abs/2409.12183.

[5] Y. Dubois, B. Galambosi, P. Liang, and T. B. Hashimoto, “Length-controlled AlpacaEval: A simple way to debias automatic evaluators,” 2024, arXiv:2404.04475. [Online]. Available: https://arxiv.org/abs/2404.04475.

[6] L. Yue, Y. Du, Y. Wang, W. Gao, F. Yao, L. Wang, et al., “Don’t overthink it: A survey of efficient R1-style large reasoning models,” 2025, arXiv:2508.02120. [Online]. Available: https://arxiv.org/abs/2508.02120.

[7] J. Wei, X. Wang, D. Schuurmans, M. Bosma, b. ichter, F. Xia, et al, “Chain-of-thought prompting elicits reasoning in large language models,” in Advances in Neural Information Processing Systems(NIPS), vol. 35, pp. 24824–24837, 2022.

[8] A. Jaech, A. Kalai, A. Lerer, A. Richardson, A. El-Kishky, A. Low, et al, “Openai o1 system card,” 2024, arXiv:2412.16720. [Online]. Available: https://arxiv.org/abs/2412.16720.

[9] M. WILDON, “GEMINI 3.0 PRO ‘DEEP THINK’VERSUS AMERI-CAN MATHEMATICAL MONTHLY PROBLEMS,” 2026.

[10] D. Guo, D. Yang, H. Zhang, J. Song, P. Wang, Q. Zhu, et al, “Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning,” 2025, arXiv:2501.12948. [Online]. Available: https://arxiv.org/abs/2501.12948.

[11] N. Stiennon, L. Ouyang, J. Wu, D. Ziegler, R. Lowe, C. Voss, et al, “Learning to summarize with human feedback,” in Advances in Neural Information Processing Systems(NIPS), vol. 33, pp. 3008–3021, 2020.

[12] C. B. Browne, E. Powley, D. Whitehouse, S. M. Lucas, P. I. Cowling, P. Rohlfshagen, “A survey of monte carlo tree search methods,” in IEEE Transactions on Computational Intelligence and AI in games, vol. 4, no. 1, pp. 1–43, 2012.

[13] J. Uesato, N. Kushman, R. Kumar, R. Kumar, F. Song, N. Siegel, L. Wang, et al, “Solving math word problems with process-and outcome-based feedback,” 2022, arXiv:2211.14275. [Online]. Available: https://arxiv.org/abs/2211.14275.

[14] T. Han, Z. Wang, C. Fang, S. Zhao, S. Ma, and Z. Chen, “Token-budgetaware LLM reasoning,” 2024, arXiv:2412.18547. [Online]. Available: https://arxiv.org/abs/2412.18547.

[15] A. Afzal, F. Matthes, G. Chechik, and Y. Ziser, “Knowing before saying: LLM representations encode information about chain-of-thought success before completion,” 2025, arXiv:2505.24362. [Online]. Available: https://arxiv.org/abs/2505.24362.

[16] P. Liu, F. Xu, Y. Li, “Token Signature: Predicting Chain-of-Thought Gains with Token Decoding Feature in Large Language Models,” 2025, arXiv:2506.06008. [Online]. Available: https://arxiv.org/abs/2506.06008.

[17] T. Munkhbat, N. Ho, S. H. Kim, Y. Yang, Y. Kim, and S. Y. Yun, “Self-training elicits concise reasoning in large language models,” 2025, arXiv:2502.20122. [Online]. Available: https://arxiv.org/abs/2502.20122.

[18] Y. Kang, X. Sun, L. Chen, and W. Zou, “C3oT: Generating shorter chainof-thought without compromising effectiveness,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 39, pp. 24312–24320, 2025.

[19] Z. Cheng, D. Chen, M. Fu, and T. Zhou, “Optimizing length compression in large reasoning models,” 2025, arXiv:2506.14755. [Online]. Available: https://arxiv.org/abs/2506.14755.

[20] Y. Shen, J. Zhang, J. Huang, S. Shi, W. Zhang, J. Yan, N. Wang, K. Wang, Z. Liu, and S. Lian, “DAST: Difficulty-adaptive slow-thinking for large reasoning models,” 2025, arXiv:2503.04472. [Online]. Available: https://arxiv.org/abs/2503.04472.

[21] W. Liu, R. Zhou, Y. Deng, Y. Huang, J. Liu, Y. Deng, Y. Zhang, and J. He, “Learn to reason efficiently with adaptive lengthbased reward shaping,” 2025, arXiv:2505.15612. [Online]. Available: https://arxiv.org/abs/2505.15612.

[22] T. Brown, B. Mann, N. Ryder, M. Subbiah, J. D Kaplan, P. Dhariwal, et al., “Language models are few-shot learners,” in Advances in Neural Information Processing Systems(NIPS), vol. 33, pp. 1877–1901, 2020.

[23] Y. Wu, Y. Wang, Z. Ye, T. Du, S. Jegelka, and Y. Wang, “When more is less: Understanding chain-of-thought length in LLMs,” 2025, arXiv:2502.07266. [Online]. Available: https://arxiv.org/abs/2502.07266.

[24] R. Rafailov, A. Sharma, E. Mitchell, C. D. Manning, S. Ermon, and C. Finn, “Direct preference optimization: Your language model is secretly a reward model,” in Advances in Neural Information Processing Systems(NIPS), vol. 36, pp. 53728–53741, 2023.

[25] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, et al., “LoRA: Low-rank adaptation of large language models,” in International Conference on Learning Representations(ICLR), 2022.

[26] Meta Llama Team, “The Llama 3 herd of models,” 2024, arXiv:2407.21783. [Online]. Available: https://arxiv.org/abs/2407.21783.

[27] Meta Llama Team, “Llama 3.2 3B Instruct,” Sept. 2024. [Online]. Available: https://github.com/meta-llama/llama-models/blob/main/ models/llama-3.2/3b-instruct.md. Accessed: Sept. 18, 2025.

[28] A. Yang, A. Li, B. Yang, B. Zhang, B. Hui, B. Zheng, et al, “Qwen3 technical report,” 2025, arXiv:2505.09388. [Online]. Available: https://arxiv.org/abs/2505.09388.

[29] K. Cobbe, V. Kosaraju, M. Bavarian, M. Chen, H. Jun, L. Kaiser, et al., “Training verifiers to solve math word problems,” 2021, arXiv:2110.14168. [Online]. Available: https://arxiv.org/abs/2110.14168.

[30] D. Hendrycks, C. Burns, S. Kadavath, A. Arora, S. Basart, E. Tang, D. Song, and J. Steinhardt, “Measuring mathematical problem solving with the math dataset,” 2021, arXiv:2103.03874. [Online]. Available: https://arxiv.org/abs/2103.03874.

[31] D. Jin, E. Pan, N. Oufattole, W. H. Weng, H. Fang and P. Szolovits, “What disease does this patient have? a large-scale open domain question answering dataset from medical exams,” in Applied Sciences, vol. 11, no. 14, pp. 6421, 2021.

[32] M. Geva, D. Khashabi, E. Segal, T. Khot, D. Roth, and J. Berant, “Did Aristotle use a laptop? A question answering benchmark with implicit reasoning strategies,” in Transactions of the Association for Computational Linguistics(TACL), vol. 9, pp. 346–361, 2021.

[33] M. Suzgun, N. Scales, N. Scharli, S. Gehrmann, Y. Tay, H. W. Chung,¨ et al., “Challenging BIG-Bench tasks and whether Chain-of-Thought can solve them,” in Findings of the Association for Computational Linguistics: ACL 2023, pp. 13003–13051, July 2023.

[34] R. Y. Pang, W. Yuan, H. He, K. Cho, S. Sukhbaatar, and J. Weston, “Iterative reasoning preference optimization,” in Advances in Neural Information Processing Systems, vol. 37, pp. 116617–116637, 2024.

[35] L. Zheng, W. L. Chiang, Y. Sheng, S. Zhuang, Z. Wu, Y. Zhuang, et al, “Judging llm-as-a-judge with mt-bench and chatbot arena,” in Advances in Neural Information Processing Systems(NIPS), vol. 36, pp. 46595– 46623, 2023.

## APPENDIX

## A. LLM-as-a-Judge VS. Exact Match

In evaluating short answers, LLM-as-a-Judge is increasingly becoming a more effective and accurate approach for assessing answer correctness. Therefore, in Tab. IV, we present the consistency score of the Llama-3.2-3B and Llama-3.1-8B models across three datasets between using both Exact Match (EM) and LLM-as-a-judge as evaluation methods. For each dataset, we randomly sample 300 samples for comparison. The high agreement rate observed between the two methods supports the validity and reliability of our Regex-based EM evaluation pipeline.

TABLE IV  
THE MATCH RATIO BETWEEN EM SCORE AND LLM-AS-A-JUDGE ON THE SKIP8 EXPERIMENT SETTING. HIGH CONSISTENCY SUPPORTS THE ACCURACY OF THE EM RESULTS.
<table><tr><td>Model</td><td>Dataset</td><td>Consistency</td></tr><tr><td rowspan="3">3B</td><td>GSM8K</td><td>0.91</td></tr><tr><td>Math</td><td>0.89</td></tr><tr><td>StrategyQA</td><td>0.87</td></tr><tr><td rowspan="3">8B</td><td>GSM8K</td><td>0.94</td></tr><tr><td>Math</td><td>0.87</td></tr><tr><td>StrategyQA</td><td>0.85</td></tr></table>

Furthermore, we report the evaluation results when LLMas-a-judge is employed for data construction and training. The SKIP-1 method on Llama-3.1-8B improves the Bestof-N baseline from 81.6 / 154.0 / 0.53 to 82.1 / 101.7 / 0.807 when trained with data constructed through EM and to 83.47 / 138 / 0.6048 when trained with data constructed through LLM-as-a-Judge. We analyze the lower accuracy scores observed under EM compared to LLM-as-a-judge and attribute this to the fact that EM is prone to generating more False Negatives (i.e., judging a correct answer as wrong due to formatting). Consequently, the LLM-as-a-judge approach successfully identifies more valid preference pairs satisfying ”Longer & Right ≻ Shorter & Wrong”. This enrichment of the training data encourages the model to acquire correct answers through longer, more robust reasoning processes.

## B. Length Preference Data

As discussed in Sec. III-B,we define a valid data instance at step<sub>i</sub> specifically when it is accompanied by an incorrect answer at $s t e p _ { t < i }$ and a correct answer at $s t e p _ { t > i }$ . Using Llama-3.1-8B on the GSM8K and MATH datasets, we investigate how the inclusion of data satisfying the condition of ’appearing together with an incorrect answer at $s t e p _ { t < i } ,$ affects the final training outcomes.

Our results in Tab. V indicate that while V2-data yields greater length compression and efficiency in most cases, it comes at the cost of a significant decrease in accuracy. We hypothesize that this is because cases that meet the conditions are more likely to contain thoughtful information, while data that does not meet the aforementioned conditions, in some cases due to the questions being too simple, allows the model to arrive at the correct answer without extensive thinking. In this situation, constructing preference data has led the model to learn a certain degree of lazy thinking habits. We argue that in length compression training, simultaneously improving efficiency and accuracy is of greater value. Consequently, we report the results obtained after training with V1-data. Nevertheless, we acknowledge that V2-data is capable of achieving length compression and efficiency that surpass the baseline.

## C. Case Study

System: Answer the question.

User: Question: A math teacher had \$100 to buy three different types of calculators. A basic calculator costs \$8. A scientific calculator costs twice the price as the basic while a graphing calculator costs thrice the price as the scientific. How much change did she receive after buying those three different types of calculators?

Probe at $S t e p _ { 0 } \colon$ The answer is 0.

Probe at $S t e p _ { 1 } \mathrm { : }$ First, find the cost of each calculator.   
The answer is 100.

Probe at Step<sub>2</sub>: First, find the cost of each calculator.

\- Basic calculator: \$8 - Scientific calculator: $2 ^ { * } \$ 8 = \$ 16$

\- Graphing calculator: $3 * \ S 1 6 = \ S 4 8$

The answer is $1 0 0 \cdot ( 8 + 1 6 + 4 8 ) = 1 0 0 \cdot 7 2 = \ S 2 8 .$

Probe at $S t e p _ { 3 } \mathrm { : }$ First, find the cost of each calculator.

\- Basic calculator: \$8 - Scientific calculator: $2 ^ { * } \$ 8 = \$ 16$

$$
3 * \ S 1 6 = \ S 4 8
$$

Next, add the costs of the three calculators: $\$ 8 +\$ 16+$ $\$ 48 =572$ . The answer is 72.

Probe at $S t e p _ { 4 } \mathrm { : }$ : First, find the cost of each calculator. - Basic calculator: \$8 - Scientific calculator: $2 ^ { * } \$ 8 = \$ 16$

$$
3 * \ S 1 6 = \ S 4 8
$$

Next, add the costs of the three calculators: $\$ 8 +\$ 16+$ $\$ 48 =572$ The teacher had \$100. Subtract the cost of the calculators from the initial amount: $\$ 100-\ S 72=\ S 28$ The answer is 28.

We present a detailed case study here. By inserting probes at the conclusion of various sentences within a mathematical reasoning chain, we observe that the probes yield correct results at both $s t e p _ { 2 }$ and $s t e p _ { 4 }$ . Notably, the probe at step produces a precise yet more concise logical derivation, which validates the efficacy of our probing setup. In contrast, step<sub>3</sub> results in an erroneous output because the preceding sentence disrupts the model’s ability to synthesize prior context. When the incorrect response at step<sub>3</sub> is paired with the correct one at step to construct length preference data, the model learns to truncate reasoning and provide concise answers when the logic is already sufficient. Conversely, when paired with the correct output from $s t e p _ { 4 }$ as knowledge preference data, the model is encouraged to refine its reasoning until a valid conclusion is reached.

TABLE V  
V1 REPRESENTS THE LENGTH PREFERENCE DATA AT step<sub>i</sub> CONSTRUCTED UNDER THE CONDITION THAT ”THERE IS AN INCORRECT ANSWER AT $s t e p _ { t < i } \} :$ , WHILE V2 REPRESENTS DATA THAT DOES NOT NEED TO SATISFY THIS CONDITION. THE BOLD TEXT REPRESENTS THE BEST RESULT FOR THE CORRESPONDING METRIC IN THE BLOCK.
<table><tr><td>Method</td><td colspan="3">GSM8K</td><td colspan="3">Math</td></tr><tr><td></td><td>Acc</td><td>Len</td><td>Eff</td><td>Acc</td><td>Len</td><td>Eff</td></tr><tr><td>BON1</td><td>81.6</td><td>154.0</td><td>0.530</td><td>46.3</td><td>418.5</td><td>0.111</td></tr><tr><td>SKIP1-V1</td><td>82.9</td><td>115.2</td><td>0.720</td><td>47.8</td><td>398.2</td><td>0.120</td></tr><tr><td>SKIP1-V2</td><td>82.1</td><td>101.7</td><td>0.807</td><td>43.7</td><td>380.6</td><td>0.115</td></tr><tr><td>BON8</td><td>83.4</td><td>127.1</td><td>0.656</td><td>45.1</td><td>374.3</td><td>0.120</td></tr><tr><td>SKIP8-V1</td><td>84.4</td><td>119.0</td><td>0.709</td><td>46.7</td><td>367.9</td><td>0.127</td></tr><tr><td>SKIP8-V2</td><td>82.4</td><td>97.6</td><td>0.844</td><td>45.4</td><td>338.0</td><td>0.134</td></tr></table>

TABLE VI

COMPARISON OF INFERENCE AND TRAINING DATA VOLUMES BETWEENBON-8 AND SKIP-8 ON THE LLAMA3.1-8B-INSTRUCT MODEL.
<table><tr><td rowspan="2">dataset</td><td colspan="2">BON-8</td><td colspan="2">SKIP-8</td></tr><tr><td>GEN</td><td>TRAIN</td><td>GEN-SFT/DPO</td><td>TRAIN-SFT/DPO</td></tr><tr><td>GSM8K</td><td>7473*8</td><td>7473</td><td>3737*8/3736</td><td>3737/3169</td></tr><tr><td>Math</td><td>3750*8</td><td>3750</td><td>3750*8/3750</td><td>3750/2083</td></tr><tr><td> $\mathbf { M e d Q A }$ </td><td>3000*8</td><td>3000</td><td>3000*8/3000</td><td>3000/1136</td></tr></table>

## D. Training Dataset Statics

We present in Tab. VI a comparison of the inference and training data volumes between BON-8 and SKIP-8 across different datasets, using the Llama3.1-8B-instruct model. Since the preference dataset is constructed by inserting probes during linear generation, each question can be approximately treated as one generation. When constructing preference pairs, a large amount of data is filtered out because it does not satisfy our predefined rules. It can be observed that SKIP-8 has lower overall sampling cost and training cost compared to BON-8.