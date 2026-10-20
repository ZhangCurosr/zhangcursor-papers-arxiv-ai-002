# ScopeMem: Latent Memory That Learns Where to Revise Frozen GUI Agents

Zhicheng Zheng<sup>1,∗</sup> Cheng Zhang<sup>1,∗</sup> Shengping Yu<sup>2</sup> Tianxiang Zhao<sup>1</sup>

<sup>1</sup>The Hong Kong University of Science and Technology (Guangzhou)

<sup>2</sup>Imperial College London

zzheng186@connect.hkust-gz.edu.cn

tianxiangz@hkust-gz.edu.cn

<sup>∗</sup>Equal contribution.

## Abstract

Latent memory is gaining ground as a lightweight history supplement for GUI agents. It supplies history to a frozen vision-language model via prefix vectors trained with step-wise cross-entropy (CE). While resulting gains are naturally attributed to encoded history, our controlled experiments show they primarily stem from action revision. The prefix learns to alter the model’s default actions. However, this revision is unscoped because step-wise CE pushes outputs toward reference actions at every step without explicitly preserving correct decisions. Consequently, it repairs errors but also breaks 8–13% of initially correct answers. Overall accuracy metrics hide these losses. We therefore propose SCOPEMEM, a latent-memory method for scoped revision. Its encoder conditions prefix generation on the model’s initial answer and is trained to correct errors while preserving correct decisions. On the controlled ablation set, SCOPEMEM reduces the share of broken correct decisions to 5.6% while repairing more errors than any evaluated CE-trained prefix. On AndroidControl, GUI-Odyssey, and zero-shot AITW, SCOPEMEM outperforms all evaluated prompting and latent-memory baselines, enabling a frozen Qwen3-VL-8B to match or surpass GUI-specialized agents and Qwen3-VL-32B.

## 1 Introduction

Latent memory is gaining ground as a lightweight history supplement for GUI agents built on general-purpose vision–language models (Wu et al., 2025a; Zhang et al., 2026b,c). We call the frozen model that selects each action the actor. An encoder converts history into a small set of continuous prefix vectors, which are prepended to the actor’s input (Li and Liang, 2021). Only the encoder is trained, with step-wise cross-entropy (CE) on reference actions. The prefix’s gains are often attributed to stored history. We test this explanation by examining which decisions the prefix changes.

![](images/6f39ca23591962105f2c1ac693d89aea1696c606b831437466f01a079fe99abb.jpg)  
Figure 1: Revision cuts both ways. Of the frozen actor’s 1,463 decisions (centre), a fixed-query latent-memory prefix (left) repairs 196 errors but harms 115 correct decisions. Its 5.5-point accuracy gain (pp: percentage points) hides these harms. SCOPEMEM (right) learns to focus revisions on the actor’s errors, repairing 254 and harming 49.

Our controlled experiments (§3) show that much of the gain persists even when the encoder does not read history. Holding the actor, its context and training fixed, a task-only encoder improves accuracy by 12.0 percentage points, compared with 11.8 points for an encoder that also reads recent history. The actor already sees the same recent history in both conditions, so this comparison isolates the benefit of encoding that history again into the prefix. The small difference suggests that, in this setting, the prefix mainly revises actions using the actor’s existing context.

At every step, CE rewards changes that move the actor’s output towards the reference action, without requiring its output distribution to remain unchanged when its decision is already correct. This leaves the revision unscoped. A change can repair an incorrect decision, but it can also turn a correct decision into an incorrect one, which we call a harm. Across our CE-trained prefixes, 8–13% of the actor’s originally correct decisions become incorrect. Encoders that read different inputs exhibit similar harm rates, pointing to their shared objective. Figure 1 illustrates how overall accuracy gains can conceal many harms because accuracy reports only the net effect of repairs and harms.

We therefore propose SCOPEMEM, a latent-memory method for scoped revision. It aims to repair the actor’s errors while preserving the decisions it already gets right. To make this judgement, the encoder needs to see the particular answer it may revise, which a context-only encoder never directly observes. SCOPEMEM therefore conditions prefix generation on the actor’s first opinion. Figure 2 shows the resulting two-pass process. The frozen actor first responds without a prefix. The encoder then reads this answer and its confidence signals together with the task and recent history, and generates a prefix for a second pass over the same context. The resulting second opinion is the action SCOPEMEM takes.

However, seeing the first opinion does not teach the encoder which decisions to preserve. We therefore train only the encoder with a scoped preservation objective. We call the steps where the actor’s original answer is wrong the scope of revision. During training, reference actions identify these steps. CE remains active at every step to encourage the reference action. Outside the scope, we additionally penalise changes to the actor’s original output distribution. At deployment, the encoder uses the first opinion and context to decide which answers to preserve, without access to reference actions.

On the controlled ablation set, scoped preservation reduces harms for every encoder, most strongly for the first-opinion encoder. SCOPEMEM lowers the harm rate to 5.6% while repairing more errors than any CE-trained prefix. Across AndroidControl, GUI-Odyssey and zero-shot AITW, it outperforms all evaluated prompting and latent-memory baselines. With SCOPEMEM, the frozen Qwen3-VL-8B actor matches or surpasses GUI-specialised agents and Qwen3-VL-32B under the same actor context.

Our contributions are threefold:

• With recent history already in context, our CE-trained prefixes mainly revise decisions rather than supply history (§3).

<table><tr><td>Line of work</td><td>Actor</td><td colspan="2">Revision</td><td colspan="2">Harms</td></tr><tr><td></td><td>frozen</td><td></td><td>reads learned</td><td></td><td>prevented reported</td></tr><tr><td>GUI fine-tuning and RL</td><td>X</td><td></td><td>√</td><td>X</td><td>X</td></tr><tr><td>Explicit memory</td><td>V</td><td>××</td><td>X</td><td>X</td><td>X</td></tr><tr><td>Latent memory</td><td></td><td>X</td><td>√</td><td>X</td><td>X</td></tr><tr><td>Self-reflection</td><td>vv</td><td>√</td><td>X</td><td>X</td><td>0</td></tr><tr><td>SCOPEMEM</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td></tr></table>

Table 1: Only SCOPEMEM combines learned revision, access to the first opinion, and an objective for preserving correct decisions. Reads: sees the first opinion; learned: trained; prevented: preservation is a training objective; reported: repairs and harms reported; : partially.

• Cross-entropy leaves this revision unscoped. Every tested prefix trained with CE alone breaks some correct decisions, and aggregate accuracy hides these losses. We therefore recommend reporting repairs and harms alongside accuracy.

• We propose SCOPEMEM, which combines an encoder that reads the actor’s first opinion with a scoped preservation objective. SCOPEMEM outperforms the evaluated prompting and latent-memory baselines across three benchmarks (§§4–6).

## 2 Related Work

Vision–language GUI agents. Vision–language GUI agents learn screenshot-based control through supervised fine-tuning (Cheng et al., 2024; Wu et al., 2025c; Xu et al., 2025; Qin et al., 2025) or reinforcement learning (Xu et al., 2026; Yang et al., 2026; Venus Team et al., 2026). To complete multi-step tasks, they must also retain interaction history. Resamplers compress this history (Lu et al., 2025), memory units track it (Wang et al., 2024), and summaries retain recent steps (Wu et al., 2025b). Our actor already receives recent history, so we keep that context fixed when evaluating an added prefix.

Memory mechanisms. Memory supplies reusable information (Zhang et al., 2026a) in two main forms. Explicit memory stores app guidance (Zhang et al., 2025; Wang et al., 2025b), experiences and workflows (Zheng et al., 2024; Agashe et al., 2025; Wang et al., 2025c), or task state (Zheng et al., 2026), including hybrid structures (Zhu et al., 2026a,b). Latent memory encodes trajectories (Wu et al., 2025a; Zhang et al., 2026b) or trusted evidence (Zhang et al., 2026c) into continuous prefixes for a frozen actor (Li and Liang, 2021). LoRA-controlled memory is a parametric alternative (Zeng et al., 2026). These designs concern how information is represented. We focus on how a latent prefix revises actions, with training that also aims to preserve correct decisions. Table 1 summarises these distinctions.

![](images/b616800f5845ade198596e4162289982c1a728b103c1e66e0d482573c40c796a.jpg)  
Figure 2: SCOPEMEM gives the frozen actor a second opinion on its first. The actor 1 observes the context and 2 answers alone in a first pass. The encoder $E _ { \phi } \textcircled { 3 }$ reads the first opinion, its confidence, the task and textual history to generate a prefix. This prefix is 4 prepended to the same context, and the actor 5 answers again. Gold parts exist only in training: 6 the preservation term is active when the first opinion matches the reference action, and 7 gradients update $E _ { \phi }$ only.

Revision and negative flips. Revision can repair errors but also cause negative flips from correct to incorrect answers. Prompted self-correction revises without training (Madaan et al., 2023) and can introduce as many errors as it repairs without external feedback (Huang et al., 2024; Kamoi et al., 2024). Learned correctors train a model to revise generated answers (Welleck et al., 2023). Focal distillation addresses such errors during model updates by preserving predictions the old model gets right (Yan et al., 2021). SCOPEMEM applies this preservation principle during prefix training. Its encoder reads the initial answer to guide revision.

## 3 Does Latent Memory Remember or Revise?

A prefix may supply information the actor lacks or change how it uses information already in its context. To distinguish these explanations, we withhold history from the encoder while keeping the actor’s history available. If the prefix helps mainly by encoding history, its gain should shrink. If it mainly revises how the actor uses existing context, much of the gain should remain.

Revision also predicts a side effect. Since cross-entropy pulls every step towards the reference action whatever the prefix reads, a revising prefix should change correct decisions as well as wrong ones. Prefixes trained in the usual way, with cross-entropy alone, behave as revision predicts on both counts.

## 3.1 A controlled study

At step t the frozen actor π<sub>θ</sub> reads a history-memory context $c _ { t }$ and emits the next action. This context consists of the task, a rolling summary of the last three steps and the latest three screenshots. A prefix $z _ { t }$ of K=16 vectors is prepended to the actor’s input embeddings. Only the module that writes the prefix is trained.

The actor then selects an action as

$$
a _ { t } \sim \pi _ { \theta } ( \cdot \mid z _ { t } \oplus c _ { t } ) ,\tag{1}
$$

where ⊕ denotes prepending. We compare three prefixes with different inputs. The fixed-query prefix, used in latent memory (Zhang et al., 2026b,c), uses one set of learned queries shared across steps to read the current context inside the actor’s forward pass. The task-only encoder reads the task, while the history encoder also reads the rolling summary and recent actions. All three use the same actor, actor context, training data and schedule. They minimise cross-entropy on the reference action $y ^ { * }$ annotated for each step. We compare each with the plain actor, which has no prefix. Evaluation uses a held-out ablation set of 1,463 test steps from AndroidControl (high-level instructions) and GUI-Odyssey (random split). We report paired 95% trajectory-bootstrap intervals for differences between methods. Appendix A reports the key paired contrasts.

![](images/4b4e327124947b28b2be81034467aea45acbfa88a55ce230a7c68ba010306ac9.jpg)

Detecting actor errors Distinguishing repairs from harms(b)  
![](images/a5387bfad79304e719598c1fcf74a1c42d1185686ee416335ff6d55f669df781.jpg)  
Error bars: 95% CI. Dashed line: chance (AUC = 0.5).  
Figure 3: Prefixes trained with cross-entropy alone also change correct decisions, and confidence barely separates repairs from harms. (a) Ablation set: gain over the plain actor (pp), share of correct actions changed, and harm rate. Shading marks SCOPEMEM. (b) AUC of confidence signals and expanded features for detecting the plain actor’s errors and separating SCOPEMEM’s repairs from harms.

## 3.2 The gain comes from revision

Figure 3a shows that the task-only encoder improves the actor as much as the history encoder (12.0 vs. 11.8 pp), and both improve it about twice as much as the fixed-query prefix. For history minus task-only accuracy, the paired 95% confidence interval has an upper bound of 1.6 pp, a small fraction of either encoder’s gain. The prefix thus mainly revises how the actor uses its existing context and should be judged by the decisions it changes.

## 3.3 The revision repairs and harms

Accuracy measures only the net effect of a revision, so we count its two kinds of change separately. A repair is a step that the plain actor gets wrong and the prefixed actor gets right, and a harm is the reverse. With R repairs and H harms over N steps, the gain over the plain actor is

$$
\Delta \mathrm { A c c } = { \frac { \mathrm { R - H } } { N } } .\tag{2}
$$

We also report two rates. The repair rate divides R by the plain actor’s errors, and the harm rate divides H by its correct steps.

Every tested prefix changes most of the plain actor’s correct actions in at least one field. Changes need not cause errors, since coordinate shifts within the benchmark’s tolerance still count as changes. Most modified actions remain correct, but some become incorrect. These harms cancel more than a quarter of what each encoder repairs, and more than half for the fixed-query prefix. The task-only and history encoders harm at nearly the same rate although they read different inputs, which points to the objective they share.

A gate could fall back to the plain actor’s answer to avoid harms, but it would need to distinguish harmful from helpful revisions. The actor’s confidence detects its own errors moderately well (AUC 0.72). However, Figure 3b shows that this confidence barely distinguishes SCOPEMEM’s repairs from harms (AUC 0.53). This motivates learning the scope rather than gating revisions by this confidence alone.

## 4 SCOPEMEM: Latent Memory as a Scoped Revision

Our ideal revision repairs every error and preserves every correct decision, maximising Eq. (2). Let $a _ { 0 }$ be the actor’s answer without a prefix and $\pi _ { 0 }$ the output distribution used to decode that answer. The ideal distribution with the prefix is

$$
\pi _ { z } ^ { \star } = { \left\{ \begin{array} { l l } { \delta _ { y ^ { * } } } & { { \mathrm { ~ i f ~ } } a _ { 0 } { \mathrm { ~ i s ~ w r o n g , } } } \\ { \pi _ { 0 } } & { { \mathrm { ~ i f ~ } } a _ { 0 } { \mathrm { ~ i s ~ r i g h t , } } } \end{array} \right. }\tag{3}
$$

where $\delta _ { y ^ { * } }$ puts all mass on the reference action. The scope consists of steps on which a<sub>0</sub> is wrong. The ideal revision corrects these errors and leaves the original output distribution unchanged on correct steps. Keeping this distribution unchanged also preserves the correct answer. SCOPEMEM learns to approximate this ideal. Its encoder reads the first opinion to guide revision at deployment, and its training objective encourages correction inside the scope and preservation outside it.

<table><tr><td rowspan="2">Method</td><td colspan="2">AndroidControl</td><td colspan="2">GUI-Odyssey</td><td colspan="6">AITW (zero-shot)</td></tr><tr><td>EM</td><td>TM</td><td>EM</td><td>TM</td><td>Gen.</td><td>Inst.</td><td>Google</td><td>Single</td><td>Shop</td><td>Macro</td></tr><tr><td colspan="9">General-purpose models</td><td></td></tr><tr><td>Qwen3-VL-4B</td><td>60.64</td><td>76.42</td><td>57.18</td><td>78.06</td><td>54.19</td><td>68.05</td><td>57.40</td><td>66.67</td><td>53.44</td><td>59.95</td></tr><tr><td>Qwen3-VL-32B</td><td>61.88</td><td>78.37</td><td>61.84</td><td>80.30</td><td>57.54</td><td>69.82</td><td>62.13</td><td>63.16</td><td>56.87</td><td>61.90</td></tr><tr><td>InternVL3.5-8B</td><td>35.28</td><td>57.80</td><td>29.52</td><td>60.41</td><td>40.22</td><td>39.64</td><td>28.99</td><td>47.37</td><td>34.35</td><td>38.12</td></tr><tr><td>GLM-4.1V-9B</td><td>40.96</td><td>70.74</td><td>35.80</td><td>67.56</td><td>37.43</td><td>44.97</td><td>39.05</td><td>40.35</td><td>35.88</td><td>39.54</td></tr><tr><td colspan="9">Trained for GUI control</td><td></td><td></td></tr><tr><td>Aguvis-7B</td><td>64.18</td><td>81.91</td><td>40.96</td><td>64.08</td><td>41.90</td><td>40.24</td><td>32.54</td><td>33.33</td><td>41.60</td><td>37.92</td></tr><tr><td>UI-TARS-1.5-7B</td><td>50.00</td><td>72.70</td><td>37.10</td><td>62.15</td><td>45.25</td><td>65.68</td><td>46.75</td><td>54.39</td><td>39.31</td><td>50.28</td></tr><tr><td>UI-Venus-1.5-8B</td><td>63.48</td><td>75.18</td><td>60.97</td><td>76.69</td><td>60.89</td><td>68.64</td><td>63.31</td><td>59.65</td><td>58.40</td><td>62.18</td></tr><tr><td>GUI-Owl-1.5-8B</td><td>67.55</td><td>82.62</td><td>60.66</td><td>76.07</td><td>52.51</td><td>55.03</td><td>56.80</td><td>50.88</td><td>51.15</td><td>53.27</td></tr><tr><td>GUI-Libra-8B</td><td>65.43</td><td>79.08</td><td>57.18</td><td>73.90</td><td>49.16</td><td>65.09</td><td>50.30</td><td>71.93</td><td>50.38</td><td>57.37</td></tr><tr><td colspan="9">Prompting or memory on the frozen actor</td><td></td><td></td></tr><tr><td>Qwen3-VL-8B (plain actor) Self-Reflection</td><td>62.59</td><td>75.35</td><td>58.73</td><td>78.74</td><td>60.89</td><td>65.68</td><td>57.99</td><td>54.39</td><td>52.29</td><td>58.25</td></tr><tr><td></td><td>57.98</td><td>68.97</td><td>47.98</td><td>67.31</td><td>58.66</td><td>67.46</td><td>59.76</td><td>57.89</td><td>47.71</td><td>58.30</td></tr><tr><td>↔ ReAct</td><td>63.30</td><td>76.42</td><td>62.34</td><td>79.80</td><td>62.01</td><td>67.46</td><td>59.17</td><td>59.65</td><td>53.05</td><td>60.27</td></tr><tr><td>↔ ReSum</td><td>63.83</td><td>76.95</td><td>63.70</td><td>80.36</td><td>62.57</td><td>68.64</td><td>59.76</td><td>63.16</td><td>53.82</td><td>61.59</td></tr><tr><td>↔ CoMEM†</td><td>63.30</td><td>78.19</td><td>62.77</td><td>81.48</td><td>58.66</td><td>68.64</td><td>59.76</td><td>68.42</td><td>56.11</td><td>62.32</td></tr><tr><td>↔ HyMEM</td><td>64.89</td><td>78.37</td><td>65.26</td><td>82.41</td><td>62.01</td><td>68.05</td><td>60.95</td><td>63.16</td><td>56.11</td><td>62.05</td></tr><tr><td>↔ Mem-W</td><td>65.96</td><td>80.32</td><td>68.61</td><td>84.03</td><td>62.01</td><td>69.23</td><td>62.72</td><td>68.42</td><td>57.63</td><td>64.00</td></tr><tr><td> SCOPEMEM (ours)</td><td>67.91</td><td>82.98</td><td>72.90</td><td>86.33</td><td>62.57</td><td>71.01</td><td>64.50</td><td>75.44</td><td>59.92</td><td>66.69</td></tr><tr><td>gain over the plain actor</td><td>+5.32</td><td>+7.63</td><td>+14.17</td><td>+7.59</td><td>+1.68</td><td>+5.33</td><td>+6.51</td><td>+21.05</td><td>+7.63</td><td>+8.44</td></tr></table>

Table 2: On the frozen actor, only SCOPEMEM matches or beats GUI-specialised models on all three benchmarks. Step accuracy (%) on the main test sets; EM: full action; TM: action type; AITW subsets General, Install, GoogleApps, Single, WebShopping and their mean (Macro). ,→: on the frozen plain actor $( ^ { \dagger } \colon$ on Qwen2.5-VL-7B); last row: gain in pp.

## 4.1 A second opinion on the actor’s answer

Under Eq. (3), revision is needed when the actor’s initial answer is wrong. An encoder without screenshots can use textual context to estimate the actor’s chance of being correct. However, that context does not show the answer the actor actually produced. We therefore first query the frozen actor without a prefix to obtain the first opinion $a _ { 0 }$ . The encoder receives this answer before the second actor pass. Both passes use greedy decoding.

The confidence inputs are the actor’s probabilities over ten action types $\left( p _ { 0 } \right)$ and their entropy $( H _ { 0 } )$ . The first-opinion encoder $E _ { \phi }$ is a 10.5M-parameter cross-attention network. It generates $z _ { t }$ from $a _ { 0 }$ , these inputs, the task, rolling summary and recent actions. With this prefix, the frozen actor produces the second opinion $a _ { t }$ SCOPEMEM’s selected action.

## 4.2 Learning the scope

Reference actions are unavailable at deployment, so correctness labels guide the encoder only during training. The keep indicator $k _ { t }$ is 1 when the initial answer is correct under the benchmark’s matching rules, and 0 when the answer is wrong. The training objective is

$$
\begin{array} { r } { \mathcal { L } = \mathrm { C E } + k _ { t } w _ { t } \mathcal { P } _ { t } , } \end{array}\tag{4}
$$

where CE is the cross-entropy loss of the reference action and $\mathcal { P } _ { t }$ is a preservation term based on KL divergence. For incorrect initial answers $( k _ { t } = 0 )$ CE encourages the reference action. For correct initial answers, weighted preservation penalises differences between the actor’s token distributions with and without the prefix, evaluated along the reference action. CE still encourages the reference tokens, while preservation also constrains changes to other token probabilities. We call this objective scoped preservation. The encoder receives no correctness label $k _ { t }$ . Instead, the encoder learns to generate prefixes from the first opinion, context and confidence inputs.

Three choices shape the objective. First, benchmarks require exact action-type matches but allow tolerance for arguments such as coordinates.

We therefore give decision tokens, including those specifying action type, ten times the preservation weight of argument tokens. Second, the confidence weight $w _ { t }$ increases with confidence, so confident correct answers receive stronger preservation. Third, the first epoch uses CE alone to teach the prefix to revise. Adding preservation immediately would reward a prefix that changes nothing. The second epoch adds preservation to teach the encoder which decisions to keep.

## 5 Experiments

## 5.1 Setup

Benchmarks. We use three standard offline benchmarks for mobile GUI control. AndroidControl (Li et al., 2024) covers everyday tasks within an app, and GUI-Odyssey (Lu et al., 2025) covers long cross-app tasks, where history should matter most. Prefixes are trained on the training splits of these two benchmarks. AITW (Android in the Wild; Rawles et al., 2023) is held out from training and tests transfer. We sample whole trajectories from the official test splits to form the main test sets: 564 steps for AndroidControl, 1,609 for GUI-Odyssey and 836 for AITW. Standard scorers report full-action (EM) and action-type (TM) accuracy for AndroidControl and GUI-Odyssey, and subset scores with their macro average for AITW.

Baselines. An actor can be improved in four ways, and we compare with recent methods of each. To scale or replace the general-purpose model, we use Qwen3-VL at 4B and 32B around our 8B actor, InternVL3.5-8B and GLM-4.1V-9B (Bai et al., 2025; Wang et al., 2025a; GLM-V Team, 2025). Aguvis, UI-TARS-1.5, UI-Venus-1.5, GUI-Owl-1.5 and GUI-Libra represent training for GUI control (Xu et al., 2025; Qin et al., 2025; Venus Team et al., 2026; Xu et al., 2026; Yang et al., 2026). The prompting baselines are ReAct (Yao et al., 2023), ReSum (Wu et al., 2025b) and Self-Reflection, which revises the answer verbally in the manner of Self-Refine (Madaan et al., 2023). Among methods that add memory, we compare with CoMEM (Wu et al., 2025a), HyMEM (Zhu et al., 2026b) and Mem-W (Zhang et al., 2026b), which is the closest to SCOPEMEM. The memory baselines retain their method-specific memory inputs and use SCOPEMEM’s training data and schedule. All non-memory baselines receive the same actor context as SCOPEMEM.

<table><tr><td rowspan="2">Benchmark</td><td colspan="2">Repairs / harms</td><td colspan="2">Only one system</td><td rowspan="2">Advantage (95% CI)</td></tr><tr><td>SCOPEMEM Mem-W keeps repairs</td><td></td><td></td><td></td></tr><tr><td>AndroidControl</td><td>57/27</td><td>55/36</td><td>16/7</td><td>13/11</td><td>+1.95 [−0.56, 4.62]</td></tr><tr><td>GUI-Odyssey</td><td>264/36</td><td>218/59</td><td>33/10</td><td>71/25</td><td>+4.29 [2.76, 5.84]</td></tr><tr><td>AITW</td><td>87/32</td><td>82/44</td><td>17/5</td><td>15/10</td><td>+2.03 [0.36, 3.77]</td></tr></table>

Table 3: Mem-W breaks more correct decisions than SCOPEMEM. Counts on the main test sets are relative to the plain actor. “Only one system” counts correct steps only one preserves or errors only one repairs (SCOPEMEM/Mem-W). Advantage: SCOPEMEM’s accuracy minus Mem-W’s (pp, 95% CI).
<table><tr><td rowspan="2">Second epoch</td><td rowspan="2">R H</td><td colspan="2">Rate (%)</td><td rowspan="2">Acc</td><td rowspan="2">Gain</td></tr><tr><td>| repair harm |</td><td></td></tr><tr><td colspan="7">Fixed-query prefix</td></tr><tr><td>cross-entropy alone</td><td>196 115</td><td></td><td>33.6 13.1</td><td>65.62</td><td></td><td>+5.54</td></tr><tr><td colspan="7">Task-only encoder</td></tr><tr><td>cross-entropy alone</td><td>246</td><td>71</td><td>42.1</td><td>8.1</td><td></td><td>72.04 +11.96</td></tr><tr><td>scoped</td><td>242 58</td><td></td><td>41.4</td><td>6.6</td><td></td><td> $7 2 . 6 6 \ + 1 2 . 5 8$ </td></tr><tr><td colspan="7">History encoder</td></tr><tr><td>first epoch only</td><td>245</td><td>74</td><td>42.0</td><td>8.4</td><td></td><td>71.77+11.69</td></tr><tr><td>cross-entropy alone</td><td>240</td><td>68</td><td>41.1</td><td>7.7</td><td></td><td> $7 1 . 8 4 ~ + 1 1 . 7 6$ </td></tr><tr><td>scoped</td><td>235 53</td><td></td><td>40.2</td><td>6.0</td><td></td><td> $7 2 . 5 2 \ + 1 2 . 4 4$ </td></tr><tr><td colspan="7">First-opinion encoder</td></tr><tr><td>first epoch only</td><td>244</td><td>70</td><td>41.8</td><td>8.0</td><td></td><td>71.98+11.89</td></tr><tr><td>cross-entropy alone</td><td>243</td><td>67</td><td>41.6</td><td>7.6</td><td></td><td> $7 2 . 1 1 \ + 1 2 . 0 3$ </td></tr><tr><td>unscoped</td><td>212</td><td>47</td><td>36.3</td><td>5.3</td><td></td><td> $7 1 . 3 6 \ + 1 1 . 2 8$ </td></tr><tr><td>scoped (SCOPEMEM)</td><td>254</td><td>49</td><td>43.5</td><td>5.6</td><td></td><td> $7 4 . 0 9 \ + 1 4 . 0 1$ </td></tr></table>

Table 4: Scoped preservation lowers harms for every encoder. Matched variants on the ablation set (first epoch only: checkpoint before the second epoch). Rates: R over 584 errors, H over 879 correct steps; Acc: accuracy (%); Gain: over the plain actor (pp).

## 5.2 A frozen open-source actor matches GUI-specialised models

Table 2 shows that, among methods on the frozen actor, only SCOPEMEM matches the strongest model trainedfor GUI control on each benchmark. On GUI-Odyssey, prompting and memory methods other than Self-Reflection already exceed the strongest GUI-trained baseline in EM. CoMEM and Mem-W also exceed the strongest GUI-trained baseline on AITW Macro. On AndroidControl, only SCOPEMEM reaches the best GUI-trained model, GUI-Owl (67.91 vs. 67.55 EM). On the held-out AITW, SCOPEMEM ties or leads every method on every subset.

Scaling helps less evenly. Relative to the plain 8B actor, the four-fold larger Qwen3-VL-32B gains 3.1 pp EM on GUI-Odyssey and 3.7 pp Macro on AITW, but loses 0.7 pp EM on AndroidControl. SCOPEMEM’s 10.5M-parameter module gains

![](images/aea2e47d9705989fba90ed8e034e761c1a53fa066fd044bb935c5c50e4f45632.jpg)  
Figure 4: On the ablation set, arrows compare the same encoder trained with cross-entropy alone and with scoped (solid) or unscoped (dashed) preservation. Only SCOPEMEM lowers harms and raises repairs together, reaching the shaded region. The fixed-query prefix (harm rate 13.1%, repair rate 33.6%) lies outside the plotted range.

5.3–14.2 pp over the same actor. Against prompting and memory, each of the three SCOPEMEM training runs in Appendix C exceeds every other method on the frozen actor. Self-Reflection also reads the actor’s draft and spends a second actor pass, but its revision is prompted rather than learned, and it lowers accuracy on two of the three benchmarks. Reading the first opinion is therefore insufficient without learning which answers to revise (§6.2).

## 5.3 Where SCOPEMEM gains over latent memory

Mem-W is the strongest memory method. It writes a prefix for the same frozen actor, using the same data and schedule. The prefix holds retrieved past trajectories and a working snapshot of the current one, and Mem-W trains it with cross-entropy and self-distillation on every step.

Table 3 shows the most consistent difference in harms: Mem-W breaks a third to nearly two thirds more correct decisions than SCOPEMEM. On AndroidControl and AITW, the methods repair similar numbers of errors, so SCOPEMEM’s accuracy advantage comes mainly from preserving more correct decisions. The AndroidControl accuracy gap remains within sampling error. The advantage is largest on GUI-Odyssey, where SCOPEMEM also repairs 46 more errors. GUI-Odyssey has the longest trajectories, where memory of earlier steps could be especially useful. However, SCOPEMEM’s advantage over Mem-W is similar on short and long trajectories, consistent with revision as its main source. Much of this advantage comes from completion decisions, where our Mem-W reproduction more often ends tasks too early, and from clicks, which account for most errors that only SCOPEMEM repairs.

## 6 Analysis: Testing the Scope

We test three predictions of scoped revision with matched variants. (i) Preservation outside the scope lowers harms regardless of encoder input. (ii) Preserving every step also holds the actor to its errors. (iii) Scoping helps the first-opinion encoder most because it observes the decision being revised. Each band of Table 4 fixes the prefix’s inputs and varies only the second epoch. This epoch is skipped (first epoch only) or uses cross-entropy alone, scoped preservation or unscoped preservation. The last applies preservation at every step $( k _ { t } = 1 )$ .

## 6.1 Does preservation remove harms, and at what cost?

Switching the second epoch from cross-entropy alone to scoped preservation removes 17–27% of each encoder’s harms in every training run, as claim (i) predicts and Figure 4 shows for the main run. Part of the harms in §3.3 therefore comes from the objective, whatever the prefix reads.

Preserving every step costs repairs (claim ii). For the first-opinion encoder, unscoped preservation reduces harms about as much as scoped preservation. Relative to cross-entropy alone, however, it loses one in eight repairs and lowers accuracy. Preserving outputs at every step also discourages correction of the actor’s mistakes, for instance on completion steps. Under scoped preservation, the task-only and history encoders retain almost all their CE repairs. The first-opinion encoder is the only variant that both lowers harms and raises repairs relative to CE alone.

Neither more training nor the second actor pass explains the gain. Starting from the first-epoch checkpoint, a second epoch of cross-entropy alone barely changes the first-opinion encoder’s accuracy. Scoped preservation in the second epoch raises accuracy by 2.1 pp over that same checkpoint, beyond sampling error. With cross-entropy alone, the two-pass first-opinion encoder and the one-pass task-only encoder gain similar amounts over the plain actor (12.03 vs. 11.96 pp), with the difference within sampling error.

Most of the harms the scope removes are action-type harms, including premature completions, as Appendix A.3 shows. The action type is chosen by the decision tokens, which the preservation term weights most.

## 6.2 Why does the scope need the first opinion?

Appendix Table 5 shows that scoped preservation adds less than one percentage point of accuracy over CE alone for the task-only and history encoders, but about three times as much for the first-opinion encoder. Appendix Table 6 shows similar reductions in the shares of correct actions and errors changed by the task-only and history encoders under scoped preservation. Reducing revision overall can avoid harms, but it can also prevent useful corrections.

For the first-opinion encoder, scoping reduces changes to correct actions by a similar amount as for the other two encoders. However, the scoped encoder changes as many of the actor’s errors as its CE version and repairs more of them. This selectivity supports the motivation in §4.1: the encoder observes the actual answer being revised. Across the three training runs in Appendix C, the accuracy gain from scoped preservation over CE is larger for the first-opinion encoder than for the history encoder. This interaction averages 1.18 pp (95% interval across runs [0.38, 1.99]). For the main run alone, the bootstrap interval for the interaction includes zero.

What the prefix uses. The interventions on the encoder’s inputs in Appendix Table 7 point the same way. Corrupting the first opinion lowers accuracy beyond sampling error, whereas removing history changes it only within sampling error.

## 6.3 Case study

At step 7 of the test task in Figure 5, Mem-W selects the second list item, the position of the nearest pharmacy in a recalled episode, and repairs the plain actor’s error. At step 9, Mem-W again follows the recalled interface and selects the first share option. That position held Messages in the old episode but holds WhatsApp in the current task, so Mem-W breaks the plain actor’s correct decision. The same recalled episode repairs one step and

Find the nearest open pharmacy in Maps

Send its address to Alice in Messages

![](images/b275d62bc8409b0d8a26de041095bf7432714233e7566a12a6ef7c5fde084c36.jpg)  
Figure 5: Case study: one recalled episode repairs step 7 and breaks step 9 for Mem-W, whereas SCOPEMEM revises step 7 and keeps step 9.

harms another despite appearing equally relevant at both. SCOPEMEM revises the incorrect first opinion at step 7, preserves the correct one at step 9 and gets all 16 steps right.

## 7 Conclusion

We study latent memory as a history supplement for frozen GUI agents and show that, when recent history is already available, much of its benefit comes from revising decisions. Our controlled experiments reveal that CE-trained prefixes repair errors but also break correct decisions, losses that aggregate accuracy hides. We propose SCOPEMEM to scope this revision. Its encoder reads the actor’s first answer and learns to generate prefixes that revise errors while preserving correct answers. SCOPEMEM achieves more repairs and fewer harms than CE-trained prefixes in controlled ablations. Across three benchmarks, it outperforms prompting and latent-memory baselines and enables a frozen 8B actor to match or surpass GUI-specialised models.

## Limitations

Our experiments are offline and step-level, so online task success, for example on Android-World (Rawles et al., 2025), remains to be tested. Step-level matching may also penalise valid alternatives to the reference actions, and noise in these references mislabels some first opinions during training. We study one actor, Qwen3-VL-8B, on mobile benchmarks, and our analysis of memory concerns an actor that already sees its recent steps. Other actors, platforms and history budgets may behave differently. The memory baselines follow their original papers under our shared training budget, and our fine-grained analyses come from one training run. SCOPEMEM also queries the actor twice per step, which roughly doubles its latency. We leave online evaluation and memories that reach beyond the recent steps to future work.

## Ethical Considerations

We use existing public GUI benchmarks, without new participant recruitment or online action execution. Screenshots may contain names, account identifiers or other incidental personal information, and we have not completed a de-identification audit. Frozen-actor interventions could in principle be used to steer an agent towards unintended actions. Our method changes only a small module and is evaluated offline.

Assets and intended use. The data and models serve offline research, and our system is a research prototype. GUI-Odyssey is released under CC BY 4.0; Qwen3-VL-4B/8B and Qwen2.5-VL-7B model cards specify Apache 2.0. AndroidControl and AITW are released by Google Research through a repository under the Apache 2.0 licence, with no separate data licence, and the third-party rights of individual screenshots lie outside it. We provide source links and remaining licence-verification limits for these and the other models in the accompanying Responsible NLP statement, and we assert no new redistribution licence for third-party screenshots or model weights.

## Use of AI Assistants

We used generative AI assistants only to polish the wording of this paper. The research questions, the design of the method and the experiments, the analyses, and every figure, table, and number were produced by the authors.

## References

Saaket Agashe, Jiuzhou Han, Shuyu Gan, Jiachen Yang, Ang Li, and Xin Wang. 2025. Agent S: An open agentic framework that uses computers like a human. In The Thirteenth International Conference on Learning Representations.

Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, Wenbin Ge, Zhifang Guo, Qidong Huang, Jie Huang, Fei Huang, Binyuan Hui, Shutong Jiang, Zhaohai Li, Mingsheng Li, and 45 others. 2025. Qwen3-VL technical report. arXiv preprint arXiv:2511.21631.

Kanzhi Cheng, Qiushi Sun, Yougang Chu, Fangzhi Xu, Li YanTao, Jianbing Zhang, and Zhiyong Wu. 2024. SeeClick: Harnessing GUI grounding for advanced visual GUI agents. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 9313–9332, Bangkok, Thailand. Association for Computational Linguistics.

GLM-V Team. 2025. GLM-4.5V and GLM-4.1V-Thinking: Towards versatile multimodal reasoning with scalable reinforcement learning. arXiv preprint arXiv:2507.01006.

Jie Huang, Xinyun Chen, Swaroop Mishra, Huaixiu Steven Zheng, Adams Wei Yu, Xinying Song, and Denny Zhou. 2024. Large language models cannot self-correct reasoning yet. In The Twelfth International Conference on Learning Representations.

Ryo Kamoi, Yusen Zhang, Nan Zhang, Jiawei Han, and Rui Zhang. 2024. When can LLMs actually correct their own mistakes? A critical survey of self-correction of LLMs. Transactions ofthe Associationfor Computational Linguistics, 12:1417–1440.

Wei Li, William Bishop, Alice Li, Chris Rawles, Folawiyo Campbell-Ajala, Divya Tyamagundlu, and Oriana Riva. 2024. On the effects of data scale on UI control agents. In Advances in Neural Information Processing Systems (Datasets and Benchmarks Track), volume 37, pages 92130–92154.

Xiang Lisa Li and Percy Liang. 2021. Prefix-tuning: Optimizing continuous prompts for generation. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 4582–4597, Online. Association for Computational Linguistics.

Quanfeng Lu, Wenqi Shao, Zitao Liu, Lingxiao Du, Fanqing Meng, Boxuan Li, Botong Chen, Siyuan Huang, Kaipeng Zhang, and Ping Luo. 2025. GUIOdyssey: A comprehensive dataset for cross-app GUI navigation on mobile devices. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 22404–22414.

Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, Shashank Gupta, Bodhisattwa Prasad Majumder, Katherine Hermann, Sean Welleck, Amir Yazdanbakhsh, and Peter Clark. 2023. Self-Refine: Iterative refinement with self-feedback. In Advances in Neural Information Processing Systems, volume 36, pages 46534–46594.

Yujia Qin, Yining Ye, Junjie Fang, Haoming Wang, Shihao Liang, Shizuo Tian, Junda Zhang, Jiahao Li, Yunxin Li, Shijue Huang, Wanjun Zhong, Kuanye Li, Jiale Yang, Yu Miao, Woyu Lin, Longxiang Liu, Xu Jiang, Qianli Ma, Jingyu Li, and 16 others. 2025. UI-TARS: Pioneering automated GUI interaction with native agents. arXiv preprint arXiv:2501.12326.

Christopher Rawles, Sarah Clinckemaillie, Yifan Chang, Jonathan Waltz, Gabrielle Lau, Marybeth Fair, Alice Li, William Bishop, Wei Li, Folawiyo Campbell-Ajala, Daniel Toyama, Robert Berry, Divya Tyamagundlu, Timothy Lillicrap, and Oriana Riva. 2025. AndroidWorld: A dynamic benchmarking environment for autonomous agents. In The Thirteenth International Conference on Learning Representations.

Christopher Rawles, Alice Li, Daniel Rodriguez, Oriana Riva, and Timothy Lillicrap. 2023. Android in the Wild: A large-scale dataset for Android device control. In Advances in Neural Information Processing Systems (Datasets and Benchmarks Track), volume 36, pages 59708–59728.

Venus Team, Changlong Gao, Zhangxuan Gu, Yulin Liu, Xinyu Qiu, Shuheng Shen, Yue Wen, Tianyu Xia, Zhenyu Xu, Zhengwen Zeng, Beitong Zhou, Xingran Zhou, Weizhi Chen, Sunhao Dai, Jingya Dou, Yichen Gong, Yuan Guo, Zhenlin Guo, Feng Li, and 8 others. 2026. UI-Venus-1.5 technical report. arXiv preprint arXiv:2602.09082.

Junyang Wang, Haiyang Xu, Haitao Jia, Xi Zhang, Ming Yan, Weizhou Shen, Ji Zhang, Fei Huang, and Jitao Sang. 2024. Mobile-Agent-v2: Mobile device operation assistant with effective navigation via multi-agent collaboration. In Advances in Neural Information Processing Systems, volume 37.

Weiyun Wang, Zhangwei Gao, Lixin Gu, Hengjun Pu, Long Cui, Xingguang Wei, Zhaoyang Liu, Linglin Jing, Shenglong Ye, Jie Shao, Zhaokai Wang, Zhe Chen, Hongjie Zhang, Ganlin Yang, Haomin Wang, Qi Wei, Jinhui Yin, Wenhao Li, Erfei Cui, and 56 others. 2025a. InternVL3.5: Advancing open-source multimodal models in versatility, reasoning, and efficiency. arXiv preprint arXiv:2508.18265.

Zhenhailong Wang, Haiyang Xu, Junyang Wang, Xi Zhang, Ming Yan, Ji Zhang, Fei Huang, and Heng Ji. 2025b. Mobile-Agent-E: Self-evolving mobile assistant for complex tasks. arXiv preprint arXiv:2501.11733.

Zora Zhiruo Wang, Jiayuan Mao, Daniel Fried, and Graham Neubig. 2025c. Agent workflow memory. In Proceedings ofthe 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research.

Sean Welleck, Ximing Lu, Peter West, Faeze Brahman, Tianxiao Shen, Daniel Khashabi, and Yejin Choi. 2023. Generating sequences by learning to self-correct. In The Eleventh International Conference on Learning Representations.

Wenyi Wu, Kun Zhou, Ruoxin Yuan, Vivian Yu, Stephen Wang, Zhiting Hu, and Biwei Huang. 2025a. Auto-scaling continuous memory for GUI agent. arXiv preprint arXiv:2510.09038.

Xixi Wu, Kuan Li, Yida Zhao, Liwen Zhang, Litu Ou, Huifeng Yin, Zhongwang Zhang, Xinmiao Yu, Dingchu Zhang, Yong Jiang, Pengjun Xie, Fei Huang, Minhao Cheng, Shuai Wang, Hong Cheng, and Jingren Zhou. 2025b. ReSum: Unlocking long-horizon search intelligence via context summarization. arXiv preprint arXiv:2509.13313.

Zhiyong Wu, Zhenyu Wu, Fangzhi Xu, Yian Wang, Qiushi Sun, Chengyou Jia, Kanzhi Cheng, Zichen Ding, Liheng Chen, Paul Pu Liang, and Yu Qiao. 2025c. OS-ATLAS: Foundation action model for generalist GUI agents. In The Thirteenth International Conference on Learning Representations.

Haiyang Xu, Xi Zhang, Haowei Liu, Junyang Wang, Zhaozai Zhu, Shengjie Zhou, Xuhao Hu, Feiyu Gao, Junjie Cao, Zihua Wang, Zhiyuan Chen, Jitong Liao, Qi Zheng, Jiahui Zeng, Ze Xu, Shuai Bai, Junyang Lin, Jingren Zhou, and Ming Yan. 2026. Mobile-Agent-v3.5: Multi-platform fundamental GUI agents. arXiv preprint arXiv:2602.16855.

Yiheng Xu, Zekun Wang, Junli Wang, Dunjie Lu, Tianbao Xie, Amrita Saha, Doyen Sahoo, Tao Yu, and Caiming Xiong. 2025. Aguvis: Unified pure vision agents for autonomous GUI interaction. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 69772–69805. PMLR.

Sijie Yan, Yuanjun Xiong, Kaustav Kundu, Shuo Yang, Siqi Deng, Meng Wang, Wei Xia, and Stefano Soatto. 2021. Positive-congruent training: Towards regression-free model updates. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 14299–14308.

Rui Yang, Qianhui Wu, Zhaoyang Wang, Hanyang Chen, Ke Yang, Hao Cheng, Huaxiu Yao, Baolin Peng, Huan Zhang, Jianfeng Gao, and Tong Zhang. 2026. GUI-Libra: Training native GUI agents to reason and act with action-aware supervision and partially verifiable RL. arXiv preprint arXiv:2602.22190.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. 2023.

ReAct: Synergizing reasoning and acting in language models. In The Eleventh International Conference on Learning Representations.

Ziyun Zeng, Hang Hua, Bocheng Zou, Mu Cai, Rogerio Feris, and Jiebo Luo. 2026. MementoGUI: Learning agentic multimodal memory control for long-horizon GUI agents. arXiv preprint arXiv:2605.18652.

Cheng Zhang, Xuanning Cao, Zhicheng Zheng, and Hui Xiong. 2026a. MemCut: Every secret leaves a trail in the long-term memory of LLM agents. arXiv preprint arXiv:2610.21405.

Chi Zhang, Zhao Yang, Jiaxuan Liu, Yanda Li, Yucheng Han, Xin Chen, Zebiao Huang, Bin Fu, and Gang Yu. 2025. AppAgent: Multimodal agents as smartphone users. In Proceedings ofthe 2025 CHI Conference on Human Factors in Computing Systems.

Guibin Zhang, Yaohui Ling, Fanci Meng, Kun Wang, and Shuicheng Yan. 2026b. Mem-W: Latent memory-native GUI agents. arXiv preprint arXiv:2605.09317.

Zhuoran Zhang, Bowen Li, Jingcheng Ju, Yang Shi, Qixun Wang, Haotian Wang, Wei Chen, and Tengjiao Wang. 2026c. FocusMem: Factorizing content, readout, and trust in latent GUI memory. arXiv preprint arXiv:2608.04530.

Longtao Zheng, Rundong Wang, Xinrun Wang, and Bo An. 2024. Synapse: Trajectory-as-exemplar prompting with memory for computer control. In The Twelfth International Conference on Learning Representations.

Yujie Zheng, Zikang Liu, Xin Zhao, and Ji-Rong Wen. 2026. A task-state representation for long-horizon mobile GUI agents. arXiv preprint arXiv:2607.00502.

Jinchang Zhu, Jindong Li, Cheng Zhang, Jiahong Liu, and Menglin Yang. 2026a. HeLa-Mem: Hebbian learning and associative memory for LLM agents. In Proceedings of the 64th Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 13757–13769, San Diego, California, United States. Association for Computational Linguistics.

Sibo Zhu, Wenyi Wu, Kun Zhou, Stephen Wang, and Biwei Huang. 2026b. Hybrid self-evolving structured memory for computer-use agents. In Findings of the Association for Computational Linguistics: ACL 2026, pages 11287–11304, San Diego, California, United States. Association for Computational Linguistics.

## A Evidence for Scoped Revision

The ablation set contains 1,463 held-out steps: 731 from 128 AndroidControl trajectories and 732 from 50 GUI-Odyssey trajectories. These trajectories do not overlap the main test sets. The plain actor gets

<table><tr><td>Contrast</td><td>Steps PP</td><td>95% CI</td></tr><tr><td colspan="3">Scoped minus cross-entropy alone</td></tr><tr><td>task-only encoder history encoder</td><td>9 +0.62 [−0.74, 1.89]</td><td>[-0.59, 1.94]</td></tr><tr><td>first-opinion encoder</td><td>10+0.68 29+1.98</td><td>[0.53, 3.44]</td></tr><tr><td>interaction</td><td>19+1.30</td><td>[-0.38, 3.02]</td></tr><tr><td>SCOPEMEM minus first opinion, unscoped</td><td>40 +2.73</td><td>[0.98, 4.55]</td></tr><tr><td>first epoch only task only, scoped</td><td>31 +2.12 21 +1.44</td><td>[0.85, 3.40] [-0.49, 3.56]</td></tr><tr><td>fixed-query</td><td>124+8.48</td><td>[6.46, 10.56]</td></tr><tr><td>Between prefixes</td><td></td><td></td></tr><tr><td>first opinion – history, scoped</td><td></td><td>23 +1.57 [-0.25, 3.48]</td></tr><tr><td>first opinion – history, CE</td><td></td><td>4 +0.27 [-1.50, 2.05]</td></tr><tr><td>task only – history, CE</td><td></td><td>3 +0.21 [-1.62, 1.91]</td></tr></table>

Table 5: Paired contrasts on the ablation set: difference in correct steps and in accuracy (pp), with 95% trajectory-bootstrap CIs. Interaction: scoping effect of the first-opinion encoder minus that of the history encoder.

879 steps right and 584 wrong. Paired 95% intervals use 20,000 dataset-stratified whole-trajectory bootstrap resamples. Results in this section use the main run; Appendix C reports repeated runs. CE denotes cross-entropy alone, and scoped and unscoped denote preservation on correct first opinions and on all steps, respectively.

## A.1 The objective and the first opinion

Table 5 gives the key paired comparisons behind §6. Scoped preservation improves the first-opinion encoder by 1.98 pp over CE, and by 2.73 pp over unscoped preservation. Its advantage over a scoped task-only encoder and its main-run interaction with the first opinion remain uncertain: both intervals include zero.

Table 6 separates how often a prefix changes an action from how often that change succeeds. Scoping reduces changes to correct actions for all three encoders. For the first-opinion encoder, it retains the rate of changes to errors and improves their repair yield; the task-only and history encoders instead reduce changes to errors as well.

## A.2 First-opinion interventions

Table 7 tests whether the encoder uses the first opinion. Perturbing it or replacing it with one from another step lowers accuracy. These interventions keep the encoder fixed; removing history uses a separately trained encoder. The history-ablation interval includes zero.

<table><tr><td></td><td colspan="2">Errors</td><td colspan="2">Correct actions</td></tr><tr><td>Variant</td><td>changed repaired</td><td></td><td>1 changed harmed</td><td></td></tr><tr><td>Task only, CE</td><td>90.58</td><td>46.50</td><td>71.90</td><td>11.23</td></tr><tr><td>Task only, scoped</td><td>83.90</td><td>49.39</td><td>63.71</td><td>10.36</td></tr><tr><td>History, CE</td><td>90.07</td><td>45.63</td><td>71.44</td><td>10.83</td></tr><tr><td>History, scoped</td><td>84.25</td><td>47.76</td><td>64.05</td><td>9.41</td></tr><tr><td>First opinion, CE</td><td>89.55</td><td>46.46</td><td>71.67</td><td>10.63</td></tr><tr><td>First opinion, unscoped</td><td>80.14</td><td>45.30</td><td>58.59</td><td>9.13</td></tr><tr><td>SCOPEMEM</td><td>89.73</td><td>48.47</td><td>62.46</td><td>8.93</td></tr></table>

Table 6: Share (%) of the plain actor’s errors and correct actions that each prefix changes, and of those changes that repair or harm. Coordinate shifts within tolerance count as changes.
<table><tr><td>Condition</td><td>R/H Gain</td></tr><tr><td>∆ 95% CI SCOPEMEM 254/49 14.01</td></tr><tr><td>Alter the first opinion</td></tr><tr><td></td></tr><tr><td>perturbed 248/58 12.99 -1.03 [−1.90, −0.21] 244/63 12.37 -1.64 [-2.74, -0.66]</td></tr><tr><td>taken from another step</td></tr><tr><td>Remove an encoder input history 248/56 13.12 -0.89 [-2.40, 0.60]</td></tr></table>

Table 7: Interventions on SCOPEMEM: repairs/harms against the plain actor, gain (pp), and difference ∆ from SCOPEMEM (pp) with its paired 95% CI.

## A.3 Which errors are avoided?

Relative to the first-opinion encoder trained with CE, SCOPEMEM reduces action-type harms from 35 to 19, while coordinate harms change from 22 to 21. Other-argument harms change from 10 to 8, and parse harms from 0 to 1, giving totals of 67 and 49. Thus, most of the reduction concerns the decision tokens that receive stronger preservation.

Completion illustrates why preservation must be scoped. Of 48 reference COMPLETE actions, the CE variant finds 37 with 8 premature completions. Unscoped preservation finds only 30, with 2 premature completions. SCOPEMEM finds 38 with the same 2 premature completions, preserving the reduction in false completions without losing recall.

## B Comparison with Mem-W

The comparisons use sampled test trajectories and the shared actor context; general-purpose and GUI-specialised actors do not use their native history formats. In Table 3, “only one keeps” counts correct plain-actor decisions preserved by only one method, while “only one repairs” counts errors repaired by only one method. The differences sum to the accuracy advantage. AITW counts pool all 836 steps, so the 2.03 pp advantage differs from the macro-average gap in Table 2. Intervals resample whole trajectories, within subsets for AITW.

<table><tr><td>Run</td><td>AndroidControl</td><td>GUI-Odyssey</td><td>AITW</td></tr><tr><td>main</td><td>67.91</td><td>72.90</td><td>66.69</td></tr><tr><td>second</td><td>67.20</td><td>72.47</td><td>66.11</td></tr><tr><td>third</td><td>68.44</td><td>73.28</td><td>67.04</td></tr><tr><td>Mean ± SD</td><td>67.85 ± 0.62</td><td>72.88 ± 0.40</td><td>66.61 ± 0.47</td></tr><tr><td>95% interval</td><td>[66.30, 69.40]</td><td>[71.88, 73.89]</td><td>[65.45, 67.78]</td></tr></table>

Table 8: SCOPEMEM on the main test sets in each run (EM; macro accuracy for AITW), with mean, SD and 95% t interval across runs.

Actions and completion. Clicks and long-presses account for 48 of the 80 additional correct steps across AndroidControl and GUI-Odyssey. On GUI-Odyssey, completion decisions account for 24 of the 69-step advantage, partly overlapping the click category. Among its 96 reference COMPLETE actions, Mem-W finds 61 with 15 premature completions; SCOPEMEM finds 75 with 6.

Trajectory length. Splitting GUI-Odyssey at its median of 14 steps gives 581 steps in 52 short trajectories and 1,028 in 48 long ones. SCOPEMEM’s advantage over Mem-W is similar in both groups: 4.48 pp [1.68, 7.56] and 4.18 pp [2.44, 5.91], respectively. This does not suggest that its advantage comes mainly from longer histories.

## C Consistency across Training Runs

We repeat encoder training three times (seeds 17, 1031 and 2027). Table 8 reports the main benchmark results; Table 9 shows the repair–harm trade-off for the key ablations. Scoped preservation lowers harms for every encoder in every run. The first-opinion encoder has the largest accuracy benefit from scoping in each run (Table 9). Its interaction with scoping averages 1.18 pp [0.38, 1.99]. These 95% t intervals describe variation across three training runs, not test-set sampling uncertainty.

<table><tr><td>Variant</td><td>Main</td><td>Second</td><td>Third</td></tr><tr><td>Task only, CE</td><td>246/71</td><td>250/70</td><td>241/70</td></tr><tr><td>Task only, scoped</td><td>242/58</td><td>248/57</td><td>234/58</td></tr><tr><td>History, CE</td><td>240/68</td><td>243/66</td><td>242/67</td></tr><tr><td>History, scoped</td><td>235/53</td><td>236/50</td><td>234/54</td></tr><tr><td>First opinion, CE</td><td>243/67</td><td>246/64</td><td>249/64</td></tr><tr><td>First opinion, unscoped</td><td>212/47</td><td>217/45</td><td>217/48</td></tr><tr><td>SCOPEMEM</td><td>254/49</td><td>250/47</td><td>263/52</td></tr></table>

Table 9: Scoped preservation reduces harms for every encoder in all three runs. Repairs/harms on the ablation set in each training run.