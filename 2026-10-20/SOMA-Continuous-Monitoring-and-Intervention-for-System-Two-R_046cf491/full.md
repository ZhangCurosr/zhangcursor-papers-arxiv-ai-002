# SOMA: Continuous Monitoring and Intervention for System Two Reasoning with System One Models

Ruizhi Qian Lei Shi Tianyu Hu Bo Yan Song Wang Yu Tian

University of Central Florida

ru282875@ucf.edu, le779457@ucf.edu, ti721494@ucf.edu

bo949643@ucf.edu, song.wang@ucf.edu, yu.tian2@ucf.edu

## Abstract

Reasoning models often run into problems without noticing them, even when symptoms are visible in their own text. In humans, metacognition separates monitoring from control: fast, intuitive signals keep watch while slow, deliberate reasoning unfolds and help decide whether to continue, change course, or stop. System One (S1) decision models, such as Jev, are a natural candidate for this role: they return a calibrated probability for a naturallanguage question about a text in one fast call, yet they are used only as gates before reasoning. We introduce SOMA (System One Metacognitive Awareness), in which an S1 model continuously monitors a System Two (S2) reasoning model and intervenes when needed. SOMA has three layers: a scenario layer states each objective in natural language with predefined actions, an awareness layer lets S1 judge whether the described behavior appears, and a control layer accumulates these judgments, first reminds S2, and takes over only if the behavior persists. Sharing one S1 model and one controller across tasks, without training, SOMA addresses two different failures on two reasoning models: on overthinking, it cuts reasoning tokens by up to 36% while improving average accuracy, and on unanswerable questions, it raises abstention from 70.5% to 87.2%.

## 1 Introduction

Reasoning models such as DeepSeek-R1 and Qwen3 solve problems through long chains of thought (Guo et al., 2025; Yang et al., 2025), yet they often run into problems without noticing them, even when the symptoms are visible in their own text. They keep re-verifying an answer they have already confirmed (Chen et al., 2025; Peng et al., 2025), circle back to the same doubts, or state that a problem is missing information and still force a definite answer (Fan et al., 2025; Liu et al., 2026). The trace already shows what has gone wrong, but what is lacking is that nothing notices it while the reasoning is still under way.

Human reasoning suggests such an observer. Research on metacognition distinguishes the monitoring of one’s own thinking from its control (Flavell, 1979; Nelson and Narens, 1990): fast, intuitive signals, such as a feeling of rightness or a sensed conflict, accompany slow deliberation and help decide whether it continues, changes course, or stops (Thompson et al., 2011; De Neys and Glumicic, 2008; Ackerman and Thompson, 2017).

Recent System One decision models are a natural candidate for this monitoring role. Exemplified by Jev (Almeida, 2026), they take a text and a natural-language question about it and return a calibrated probability in a single fast call (Type-Safe AI, 2026). Following dual-process accounts (Kahneman, 2011; Evans and Stanovich, 2013), we call these models System One (S1) and reasoning models System Two (S2), as functional labels for fast judgment and slow deliberation.

Two limitations of this design motivate our work. First, S1 models are used only as gates before reasoning begins. Jev’s documentation, for instance, includes an example that uses it to escalate doubtful cases to a reasoning model (TypeSafe AI, 2026) (Figure 1(a)). Once S2 starts, S1 falls silent, and a trace of tens of thousands of tokens unfolds without a fast monitor. Second, existing remedies for these reasoning problems are tailored to one scenario at a time: each ties a dedicated signal, such as trial-answer confidence, answer agreement or a hidden-state probe, to a single action (Yang et al., 2026; Fu et al., 2025a; Liu et al., 2026; Xie et al., 2026), and many require training or white-box access to the model (Aggarwal and Welleck, 2025; Zhang et al., 2025a; Liu et al., 2026). Each new problem thus calls for a new mechanism.

To address both gaps, we introduce SOMA (System One Metacognitive Awareness), in which an S1 model monitors S2 throughout generation and intervenes when needed (Figure 1(b)). SOMA has three layers. The scenario layer states an objective in natural language and lists the actions allowed. The awareness layer asks S1 whether that behavior appears in the trace. The control layer accumulates these judgments, reminds S2 first, and takes over only if the behavior persists. We instantiate SOMA for overthinking and unanswerable questions, which share the same S1 model, controller and system prompt and differ only in their monitors. Our contributions are as follows:

![](images/61698a2ed2033c7e6a4d8201bbd0c796c00aae98f0822aca67c911875da85224.jpg)  
Figure 1: (a) In a common combination, S1 decides first and S2 is called for low-confidence cases; S1 is silent once S2 starts. (b) In SOMA, S1 runs beside S2: S2 periodically pauses at a line break, S1 judges whether a statement about an observable behavior holds for the reasoning so far, and cumulative evidence from positive judgments triggers a reminder inserted at the pause, where S2 decides how to respond. The trace is constructed for illustration, not taken from a run; the evidence accumulation and takeover rules are in Section 3.3.

• We propose using S1 decision models as continuous monitors of S2 reasoning rather than gates before it, and introduce SOMA, which separates supervision into scenario, awareness and control layers so that a new objective requires only a natural-language specification.

• We design a remind-first, take-over-on-persistentevidence controller with a cumulative log-odds rule, and a calibration procedure that limits the harm of irreversible takeovers.

• With a single black-box S1 and no training, SOMA reduces reasoning tokens on overthinking by 36% and 16% without lowering average accuracy, and raises abstention on unanswerable questions by 17 and 21 points on two reasoning models.

## 2 Related Work

## 2.1 Dual-Process Reasoning and Metacognition

Dual-process theories distinguish fast, intuitive judgments from slow, deliberate reasoning (Sloman, 1996; Kahneman, 2011; Evans and Stanovich, 2013). Metacognitive accounts further distinguish monitoring from control: signals such as confidence and detected conflict can determine whether deliberation continues, changes course, or stops (Flavell, 1979; Nelson and Narens, 1990; Ackerman and Thompson, 2017; De Neys and Glumicic, 2008; Thompson et al., 2011; Ackerman, 2014). We use this relationship as a functional analogy for an S1 model that observes an S2 reasoning trace and supplies judgments to a controller.

## 2.2 System One Decision Models

This distinction has recently reappeared in model design. System One decision models return typed, probabilistic judgments instead of free-form text. Jev was an early public example; related models now include CLM, Laya, OpenAI’s Decisions API (gpt-6-luna), and Microsoft-Decision-1 (Almeida, 2026; TypeSafe AI, 2026; Contrastive-LM, 2026; Convai Innovations, 2026; OpenAI, 2026; Microsoft, 2026). Their common interface makes a fast judgment available to an application or controller without asking a generative model to write a rationale, although their probabilities still require careful calibration and thresholding (Deußer et al., 2026; Porcedda, 2026; Rafe and Das, 2026), as do earlier fast judges and guard models (Kadavath et al., 2022; Zheng et al., 2023; Inan et al., 2023). Like other fast models, they are mostly used before generation: cascades and routers escalate uncertain inputs, and fast-and-slow architectures invoke a stronger solver when the fast one falls short (Chen et al., 2024; Aggarwal et al., 2024; Ong et al., 2025; Lin et al., 2023; Bergamaschi Ganapini et al., 2025; Khandelwal et al., 2025).

## 2.3 Monitoring Reasoning During Generation

A few methods supervise a reasoning model while it generates. TrimR and FlashThink query a separate prompted LLM at reasoning boundaries, asking whether a conclusion has been reached or whether the thought so far suffices, and cut thinking short when it does (Lin et al., 2026; Jiang et al., 2025). Meta-R1 and MetaCtrl let a meta-level component steer the reasoner, using a generative language model or a controller trained for concise reasoning (Dong et al., 2025; Wen et al., 2026). Each of these methods targets a single objective with a fixed question and action. SOMA instead pairs an S1 decision model with a shared controller, so that different objectives share one interface.

## 2.4 Inference-Time Control of Reasoning

We study two cases in which reasoning should change course after generation has begun. In overthinking, a model keeps revisiting an answer after reaching a useful conclusion (Chen et al., 2025; Peng et al., 2025; Sui et al., 2025); existing methods shorten reasoning by training the model or by stopping on trial-answer confidence, answer agreement, or hidden-state probes (Aggarwal and Welleck, 2025; Zhang et al., 2025b; Fang et al., 2025; Yang et al., 2026; Fu et al., 2025a; Zhang et al., 2025a; Wang et al., 2026). In unanswerablequestion settings, models may continue reasoning despite missing premises or an explicitly noticed flaw (Kirichenko et al., 2025; Song et al., 2025; Fan et al., 2025; Ahn et al., 2026; Liu et al., 2026); prior remedies use a hidden-state probe to trigger abstention guidance (Liu et al., 2026) or sequential statistical tests on uncertainty keywords to stop reasoning (Xie et al., 2026). These methods generally couple a scenario to a dedicated signal and action. SOMA instead expresses both cases as monitor specifications and reuses the same S1 model and controller, differing only in the behavior statement

and predefined response.

## 3 SOMA

SOMA lets an S1 decision model supervise an S2 reasoning model while S2 is thinking. Following the separation of monitoring and control in human metacognition, it divides supervision into a scenario layer that defines what to supervise, an awareness layer in which S1 detects it, and a control layer that decides how to respond with a single rule shared by all scenarios (Figure 2). Each layer can be replaced on its own: a new objective only adds a scenario specification, and a different S1 model plugs in through the same interface.

Throughout this paper, Q denotes the task and $H _ { t }$ the reasoning text that S2 has written after t thinking tokens.

## 3.1 Scenario Layer: What to Supervise

A supervision objective is specified by a monitor $m = ( s _ { m } , r _ { m } , a _ { m } )$ , consisting entirely of naturallanguage and predefined text.

Statement $s _ { m }$ . The statement describes the reasoning behavior to watch for, and it is the only part of the monitor that S1 sees. Statements follow a common pattern: X has appeared in the reasoning, but S2 is still doing Y. This form has two advantages. Both parts can be read off the visible text, unlike the correctness of an answer, and the statement stops being true once S2 changes its behavior, so S1’s later judgments show whether an intervention has taken effect.

Reminder $r _ { m } .$ The reminder is a short message inserted into S2’s reasoning. Rather than issuing a command, it points to the possible problem and asks S2 to judge whether its current activity is useful, leaving the decision to S2.

Takeover action $a _ { m } .$ . The takeover action is predefined text that ends thinking and steers the response, for example towards a final answer or a refusal. S2 still writes the final content after a takeover.

Because a monitor is plain text, adding a scenario requires neither training nor any change to the other two layers. The same form could also describe a reasoning trace that starts to comply with a jailbreak request.

![](images/62fdd381f2e027bc751a771b3727b332f65b4b01f1a8cff02572e4ab08e6a46a.jpg)  
Figure 2: Overview of SOMA. Every 256 thinking tokens, S2 continues to the next line break (at most 128 more tokens) and pauses. S1 reads the task, the visible reasoning, and a monitor statement, and returns a probability $p ,$ which updates the cumulative score S. Background shading shows the controller state: the first positive S inserts a reminder and resets S (orange); post-reminder evidence reaching the frozen level h triggers a setting-specific takeover (grey), after which S2 writes a bounded answer or refusal reason. The trace is illustrative.

## 3.2 Awareness Layer: S1 as an Observer

S1 is queried at regular pauses while S2 is thinking. At each pause, it receives the task, the reasoning so far and the statement, and returns

$$
p _ { t } = \mathrm { S } 1 ( Q , H _ { t } , s _ { m } ) \in [ 0 , 1 ] ,
$$

the probability that the statement holds. This interface is the only requirement on S1: any decision model that maps a text state and a natural-language statement to a probability can fill the role. It also sees only text written by S2. Inserted reminders and the system prompt are hidden from it, so its judgment always concerns S2’s own behavior.

Pauses occur every X thinking tokens. To avoid reading a half-written line, S2 first continues to the next line break and pauses there. Because S1 returns a single probability rather than generated text, each scan is one short call. When the reasoning exceeds S1’s input limit, the task is kept in full and the earliest part of the reasoning is dropped.

## 3.3 Control Layer: Remind First, Take Over on Persistent Evidence

The controller turns S1’s judgments into evidence on the log-odds scale and accumulates it as a onesided cumulative sum:

$$
S \gets \operatorname* { m a x } \bigl \{ 0 , \ S + \log \mathrm { i t } ( p _ { t } ) - \log \mathrm { i t } ( 0 . 5 ) \bigr \} .
$$

A scan with $p _ { t } > 0 . 5$ raises S, a scan with $p _ { t } < 0 . 5$ lowers it toward zero, and a scan with $p _ { t } = 0 . 5$ leaves it unchanged; S never drops below zero.

This rule follows the one-sided cumulative-sum test for detecting a persistent change (Page, 1954). Since S1 reports the probability that the statement holds, logit(p<sub>t</sub>) − logit(0.5) is the evidence of one scan for the statement against even odds, and summing log-odds treats successive scans as separate pieces of evidence. A confident scan therefore counts more than a doubtful one, an isolated high score is absorbed by later low ones, and only a behavior that S1 keeps observing drives $S$ upward.

The controller responds in two stages (Algorithm 1). When S first becomes positive, an episode begins: SOMA inserts the reminder $r _ { m }$ at the pause and resets S to zero. S2 then continues and decides for itself how to respond. From this point S accumulates only evidence gathered after the reminder. If it reaches a threshold h, the controller executes the takeover action $a _ { m } ;$ if it falls back to zero first, the episode ends, and a later rise of S above zero starts a new one with a new reminder.

The two stages differ in cost and reversibility. A reminder leaves S2 in charge and costs only a few tokens, whereas a takeover cannot be undone by S2. SOMA therefore takes over only when the behavior persists after S2 has been given a chance to stop, and it bases this decision on what S1 observes S2 doing after the reminder, not on what S2 says in reply. To make S2 attend to reminders, a single fixed system prompt, shared by all scenarios, tells S2 that a monitor may insert reminders into its reasoning.

## 3.4 Threshold Calibration

A takeover is the one intervention S2 cannot recover from. Once thinking is cut off and the response is steered, a mistaken takeover becomes a wrong answer or a false refusal, whereas a mistaken reminder can simply be dismissed by S2. The threshold h therefore decides how much postreminder evidence SOMA requires before it accepts this risk.

Algorithm 1 SOMA supervision loop for monitor   
m.   
Require: task Q, S2, S1, monitor $\begin{array} { l } { \displaystyle { m \ = \ ( s _ { m } , r _ { m } , a _ { m } ) . } } \end{array}$   
threshold h   
1: S ← 0; reminded ← false   
2: while S2 is thinking do   
3: run S2 to the next pause; H ← its reasoning   
4: $p \gets \mathrm { S } 1 ( Q , H , s \hat { m } ) ; S \gets \mathrm { m a x } \{ 0 , S + \check { \mathrm { l o g i t } } ( p ) -$   
logit(0.5)}   
5: if $S = 0$ then   
6: reminded $\gets f a l s e$ ▷ episode ends   
7: else if not reminded then   
8: insert $r _ { m } ; \ S \gets 0 ;$ reminded ← true   
9: else if $S \geq { }$ h then   
10: apply $a _ { m }$ and stop thinking ▷ takeover   
11: end if   
12: end while

No single threshold suits every setting: the evidence that S accumulates depends on how sharply S1 separates cases in which the statement holds from those in which it does not (Deußer et al., 2026), on how S2 responds to reminders, and on how often a monitor’s behavior occurs and what a mistaken takeover costs. SOMA therefore calibrates a separate threshold for each combination of S1, S2 and monitor.

Calibration takes place before deployment, on a small set C of representative tasks for the target scenario. On these tasks, SOMA under each candidate threshold $h \in \mathcal H$ is compared with S2 running under the same system prompt without any intervention, so that every difference between the two is caused by the intervention. Let $c _ { 0 } ( q ) \in \{ 0 , 1 \}$ indicate whether S2 handles task q correctly on its own, and $c _ { h } ( q )$ whether it does so under SOMA with threshold h. In overthinking, a task is handled correctly when the final answer is correct; for unanswerable questions, when S2 abstains on an unanswerable question or correctly answers an answerable one. Harm thus covers both answers spoiled by an early takeover and false refusals. We measure the harm of a threshold as the number of tasks that the intervention turns from correct to wrong:

$$
L ( h ) = \sum _ { q \in \mathcal { C } } c _ { 0 } ( q ) \big ( 1 - c _ { h } ( q ) \big ) ,
$$

and choose the smallest threshold whose harm is within a tolerance $\delta \geq 0$ of the lowest achievable

harm:

$$
h ^ { \ast } = \operatorname* { m i n } \Bigl \{ h \in \mathcal { H } : L ( h ) \le \operatorname* { m i n } _ { h ^ { \prime } \in \mathcal { H } } L ( h ^ { \prime } ) + \delta \Bigr \} .
$$

A smaller threshold takes over earlier and saves more computation, so the rule selects the most aggressive threshold whose harm stays within the tolerance; $\delta$ sets how much harm is acceptable in exchange for savings. Once chosen, the threshold stays fixed during use.

## 4 Experiments

## 4.1 Setup

Scenarios. We evaluate SOMA on overthinking, where the goal is to cut redundant verification without lowering accuracy, and on unanswerable questions, where S2 should abstain when a problem lacks necessary information or has contradictory conditions and solve it otherwise. Both scenarios use the same S1 model, controller and system prompt, and differ only in their monitors (Table 1).

Models. We use Qwen3-8B (Yang et al., 2025) and DeepSeek-R1-Distill-Llama-8B (Guo et al., 2025) as S2, served with vLLM (Kwon et al., 2023) in thinking mode with the recommended sampling settings (temperature 0.6, top-p 0.95, and top-k 20 for Qwen3-8B) and a limit of 32,768 generated tokens. Every method samples each question once. S1 is Jev (Almeida, 2026; TypeSafe AI, 2026), accessed through its API.

Benchmarks. For overthinking, we use GSM8K (1,319 questions; Cobbe et al., 2021), MATH-500 (500; Hendrycks et al., 2021; Lightman et al., 2024), AIME 2024–2025 (60; Mathematical Association of America, 2025), AMC 2023 (40; Mathematical Association of America, 2023) and GPQA-Diamond (198; Rein et al., 2024). For unanswerable questions, we use the test sets of SUM (284 unanswerable and 284 answerable questions; Song et al., 2025) and UMWP (300 and 300; Sun et al., 2024), following Liu et al. (2026). Mathematical answers are checked with Math-Verify (Kydlícekˇ , 2025), and abstentions are identified with the rule of Liu et al. (2026).

Implementation. S2 pauses every $X ~ = ~ 2 5 6$ thinking tokens, continuing to the next line break for at most 128 further tokens. The state passed to S1 is limited to 2,048 tokens, and a fixed system prompt shared by both scenarios tells S2 that a monitor may insert reminders (Appendix A).

<table><tr><td>Scenario</td><td>Statement  $s _ { m }$  (judged by S1)</td><td>Reminder  $r _ { m }$ </td><td>Takeover action  $a _ { m }$ </td></tr><tr><td>Overthinking</td><td>The reasoning has reached a final answer and keeps re-verifying it without naming any specific remaining concern.</td><td>You may be repeating verification that adds nothing new. Consider whether your recent checks are actually useful, and reply with your judgment.</td><td>End thinking; insert **Final Answer**; S2 writes the answer (64 tokens).</td></tr><tr><td>Unanswerable questions</td><td>The reasoning has noticed that the problem is missing necessary information or has contradictory conditions, but is still trying to produce a definite answer, for example by assuming values the problem does not give.</td><td>Check once more whether the problem gives everything you need. Then judge whether this check was useful or merely repeated your earlier checks, and say so explicitly.</td><td>Insert “The problem is missing necessary information or has contradictory conditions, so it cannot be answered as stated.&quot;; end thinking; pre-fill \boxed{I don&#x27;t know. }; S2 writes the reason (512 tokens).</td></tr></table>

Table 1: The two monitors evaluated in this paper. Each reminder is wrapped in <reminder> tags and inserted at a pause. Both scenarios share the S1 model, the controller and the system prompt.
<table><tr><td></td><td colspan="2">GSM8K</td><td colspan="2">MATH-500</td><td colspan="2">AIME</td><td colspan="2">AMC23</td><td colspan="2">GPQA-D</td><td colspan="2">Avg</td></tr><tr><td>Method</td><td>Acc↑</td><td>Tok</td><td>Acc↑</td><td>Tok</td><td> Acc↑</td><td></td><td>Tok Acc↑</td><td>Tok</td><td>Acc↑</td><td></td><td>Tok Acc↑</td><td>Tok</td></tr><tr><td>Qwen3-8B</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Vanilla</td><td>95.2</td><td>2,277</td><td>96.2</td><td>5,549</td><td>66.7</td><td>16,719</td><td>95.0</td><td>8,600</td><td>56.6</td><td>9,266</td><td>91.0</td><td>4,232</td></tr><tr><td>BF</td><td>95.3</td><td>-1%</td><td>95.4</td><td>-9%</td><td>55.0</td><td>-32%</td><td>95.0</td><td>-10%</td><td>55.1</td><td>-7%</td><td>90.4</td><td>-8%</td></tr><tr><td>DEER</td><td>94.2</td><td>-62%</td><td>92.4</td><td>-52%</td><td>55.0</td><td>-36%</td><td>77.5</td><td>-56%</td><td>49.0</td><td>-53%</td><td>88.1</td><td>-54%</td></tr><tr><td>Dynasor-CoT</td><td>95.3</td><td>-3%</td><td>95.8</td><td>-6%</td><td>68.3</td><td>-7%</td><td>95.0</td><td>-6%</td><td>60.1</td><td>-12%</td><td>91.4</td><td>-6%</td></tr><tr><td>SÓMA (ours)</td><td>95.5</td><td>-42%</td><td>94.8</td><td>-46%</td><td>73.3</td><td>-29%</td><td>90.0</td><td>-37%</td><td>60.1</td><td>-17%</td><td>91.3</td><td>-36%</td></tr><tr><td>R1-Distill-Llama-8B</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Vanilla</td><td>91.2</td><td>1,747</td><td>90.8</td><td>4,289</td><td>36.7</td><td>13,718</td><td>87.5</td><td>7,030</td><td>36.9</td><td>8,153</td><td>84.4</td><td>3,386</td></tr><tr><td>BF</td><td>91.2</td><td>-2%</td><td>91.6</td><td>-7%</td><td>38.3</td><td>-14%</td><td>87.5</td><td>-7%</td><td>36.4</td><td>-0%</td><td>84.6</td><td>-5%</td></tr><tr><td>DEER</td><td>82.9</td><td>-40%</td><td>87.4</td><td>-30%</td><td>38.3</td><td>-9%</td><td>82.5</td><td>-20%</td><td>33.8</td><td>-10%</td><td>78.1</td><td>-26%</td></tr><tr><td>Dynasor-CoT</td><td>91.2</td><td>-1%</td><td>91.0</td><td>-4%</td><td>36.7</td><td>-5%</td><td>85.0</td><td>-14%</td><td>42.9</td><td>—45%</td><td>85.0</td><td>-13%</td></tr><tr><td>SOMA (ours)</td><td>90.8</td><td>-25%</td><td>90.4</td><td>-20%</td><td>45.0</td><td>-1%</td><td>92.5</td><td>—24%</td><td>41.4</td><td>-5%</td><td>84.8</td><td>-16%</td></tr></table>

Table 2: Overthinking results. Acc is accuracy (%); Tok is the mean number of S2 tokens, given in absolute terms for Vanilla and as the change relative to Vanilla otherwise. For SOMA, the accounting follows the readout in Appendix B. Avg weights each benchmark by its number of questions and is computed from unrounded perbenchmark values. Bold marks the best accuracy and the largest token reduction for each S2 model; underline marks the second best.

Calibration. Thresholds are calibrated as in §3.4 with δ = 0 on one easier and one harder set of 100 questions per scenario, disjoint from all test sets (Appendix C). The resulting thresholds are h = 0.25 (Qwen3-8B) and h = 1.5 (R1-Distill-Llama-8B) for overthinking, and h = 4 and h = 3 for unanswerable questions.

Single-run readout. Since a takeover ends generation, we run SOMA once with reminders but no takeovers and read off the outcome under each threshold h at the first pause where the postreminder evidence reaches h, both for calibration and at test time (Appendix B).

Baselines. All methods share the decoding settings, generation limit, data, and scoring above; prompts are method-specific. We report Vanilla (thinking mode). For overthinking, we compare with budget forcing (BF; Muennighoff et al., 2025), DEER (Yang et al., 2026) and Dynasor-CoT (Fu et al., 2025b,a). For unanswerable questions, we compare with DEER, Dynasor-CoT, the AbstentionBench prompt (Kirichenko et al., 2025), statistical early stopping (SES; Xie et al., 2026) and Answering the Unanswerable (AtU; Liu et al., 2026), which uses a hidden-state probe trained for each S2 model. Parameters that require tuning are selected with each paper’s own procedure on the same calibration sets as SOMA’s thresholds (Appendix E).

Metrics. For overthinking, we report accuracy and the mean number of S2 tokens (Tok), counting thinking and answer tokens as well as extra tokens a method generates, such as trial answers. For unanswerable questions, we report the abstention rate on unanswerable questions (Abs), the accuracy (Acc) and false-abstention rate (FA) on answerable questions, and Tok. For SOMA, token accounting follows the readout in Appendix B. Averages (Avg) weight each benchmark by its number of questions.

<table><tr><td></td><td colspan="4">SUM</td><td colspan="4">UMWP</td><td colspan="4">Avg</td></tr><tr><td>Method</td><td>Abs↑ Acc↑ FA↓</td><td></td><td></td><td>Tok</td><td>Abs↑ Acc↑ FA↓</td><td></td><td></td><td>Tok</td><td>Abs↑</td><td>Acc↑ FA↓</td><td></td><td>Tok</td></tr><tr><td>Qwen3-8B</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Vanilla</td><td>60</td><td>80</td><td>3</td><td>6,171</td><td>81</td><td>97</td><td>2</td><td>1,627</td><td>70.5</td><td>88.7</td><td>2.4</td><td>3,837</td></tr><tr><td>DEER</td><td>68</td><td>73</td><td>6</td><td>-53%</td><td>81</td><td>95</td><td>31</td><td>-64%</td><td>74.8</td><td>84.2</td><td>4.6</td><td>-56%</td></tr><tr><td>Dynasor-CoT</td><td>61</td><td>77</td><td>6</td><td>-12%</td><td>83</td><td>98</td><td>2</td><td>-15%</td><td>72.1</td><td>87.8</td><td>3.8</td><td>-12%</td></tr><tr><td>AbsBench prompt</td><td>79</td><td>65</td><td>17</td><td>+2%</td><td>85</td><td>95</td><td>3</td><td>-15%</td><td>82.2</td><td>80.1</td><td>10.1</td><td>-2%</td></tr><tr><td>SES</td><td>80</td><td>77</td><td>10</td><td>-23%</td><td>91</td><td>91</td><td>8</td><td>-35%</td><td>85.8</td><td>83.9</td><td>8.9</td><td>-26%</td></tr><tr><td>AtU</td><td>83</td><td>77</td><td>6</td><td>-40%</td><td>90</td><td>95</td><td>4</td><td>-41%</td><td>86.5</td><td>86.6</td><td>5.0</td><td>-41%</td></tr><tr><td>SOMA (ours)</td><td>87</td><td>77</td><td>11</td><td>-32%</td><td>88</td><td>92</td><td>6</td><td>-31%</td><td>87.2</td><td>84.9</td><td>8.0</td><td>-32%</td></tr><tr><td>R1-Distill-Llama-8B</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Vanilla</td><td>28</td><td>78</td><td>1</td><td>6,716</td><td>48</td><td>86</td><td>3</td><td>1,893</td><td>38.2</td><td>82.4</td><td>2.1</td><td>4,238</td></tr><tr><td>DEER</td><td>34</td><td>77</td><td>2</td><td>-14%</td><td>52</td><td>86</td><td>3</td><td>-9%</td><td>43.0</td><td>81.7</td><td>2.9</td><td>-12%</td></tr><tr><td>Dynasor-CoT</td><td>30</td><td>79</td><td>1</td><td>-10%</td><td>49</td><td>87</td><td>3</td><td>-13%</td><td>39.9</td><td>82.7</td><td>2.1</td><td>-11%</td></tr><tr><td>AbsBench prompt</td><td>50</td><td>67</td><td>4</td><td>-29%</td><td>55</td><td>84</td><td>51</td><td>-3%</td><td>52.2</td><td>75.9</td><td>4.6</td><td>-23%</td></tr><tr><td>SES</td><td>60</td><td>76</td><td>5</td><td>-30%</td><td>66</td><td>84</td><td>51</td><td>-47%</td><td>62.7</td><td>80.3</td><td>5.0</td><td>-34%</td></tr><tr><td>AtU</td><td>71</td><td>76</td><td>4</td><td>-41%</td><td>64</td><td>85</td><td>5</td><td>-50%</td><td>67.6</td><td>80.7</td><td>4.5</td><td>-43%</td></tr><tr><td>SOMA (ours)</td><td>70</td><td>75</td><td>7</td><td>-44%</td><td>49</td><td>86</td><td>3</td><td>-59%</td><td>59.4</td><td>80.7</td><td>4.8</td><td>-48%</td></tr></table>

Table 3: Unanswerable-question results. Abs is the abstention rate on unanswerable questions, Acc the accuracy on answerable questions and FA the false-abstention rate on answerable questions (all %); Tok is as in Table 2. For SOMA, the accounting follows the readout in Appendix B. Avg weights SUM and UMWP by their number of questions and is computed from unrounded per-benchmark rates. Bold marks the best value for each S2 model (lowest for FA, largest reduction for Tok); underline marks the second best.

<table><tr><td></td><td colspan="2">Overthinking</td><td colspan="4">Unanswerable</td></tr><tr><td>Variant</td><td>Acc↑</td><td></td><td></td><td>Tok Abs↑ Acc↑ FA↓</td><td></td><td>Tok</td></tr><tr><td>Qwen3-8B</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Vanilla</td><td>93.0</td><td>6,745</td><td>72</td><td>90</td><td></td><td>33,494</td></tr><tr><td>SOMA</td><td>92.5</td><td>-42%</td><td>92</td><td>83</td><td></td><td>10-29%</td></tr><tr><td>Remind-only</td><td>93.4</td><td>-16%</td><td>74</td><td>87</td><td>4</td><td>-12%</td></tr><tr><td>No reminder</td><td>94.1</td><td>-22%</td><td>89</td><td>84</td><td>10</td><td>-26%</td></tr><tr><td colspan="7">R1-Distill-Llama-8B</td></tr><tr><td>Vanilla</td><td>85.0</td><td>5,299</td><td>36</td><td>84</td><td></td><td>33,972</td></tr><tr><td>SOMA</td><td>85.5</td><td>-15%</td><td>60</td><td>82</td><td>4</td><td>-48%</td></tr><tr><td>Remind-only</td><td>85.7</td><td>-3%</td><td>32</td><td>84</td><td>0</td><td>-12%</td></tr><tr><td>No reminder</td><td>85.2</td><td>-11%</td><td>61</td><td>82</td><td>6</td><td>-47%</td></tr></table>

Table 4: Ablations on a fixed test subset, reported as question-weighted averages. Remind-only never takes over (h = ∞). No reminder accumulates evidence from the first pause and takes over without a reminder, with its own calibrated threshold.

## 4.2 Main Results

Overthinking. Table 2 shows that SOMA shortens reasoning while preserving and slightly improving accuracy on average. It reduces generated tokens by 36% on Qwen3-8B and 16% on R1-Distill-Llama-8B, and raises average accuracy from 91.0% to 91.3% and from 84.4% to 84.8%. Weighting each benchmark equally, which removes the dominance of GSM8K, the gains are 0.8 and 3.4 points, and SOMA has the highest such average on R1-Distill-Llama-8B (72.0% vs. 69.4% for Dynasor-CoT). SOMA exceeds Vanilla on six of the ten model-benchmark combinations, with the largest gains on the hardest benchmarks: on AIME, accuracy rises from 66.7% to 73.3% on Qwen3-8B and from 36.7% to 45.0% on R1-Distill-Llama-8B, and on GPQA-Diamond it improves by 3.5 and 4.5 points. On the remaining combinations the differences are small, at most 1.4 points except for two questions on AMC 2023 with Qwen3-8B. By contrast, DEER saves more tokens but lowers average accuracy by 2.9 and 6.3 points, and BF stays close to Vanilla in accuracy but saves only 8% and 5% of the tokens. Dynasor-CoT keeps accuracy but saves only 6% of the tokens on Qwen3-8B; on R1-Distill-Llama-8B it saves 13% with accuracy comparable to SOMA.

![](images/b401e39ace374f05b997eb3485d79059e67d0444ba5f5d51e530ec75d4f2f7b0.jpg)  
Figure 3: Average accuracy on overthinking (top) and abstention on unanswerable questions (bottom) against mean S2 tokens relative to Vanilla, from the Avg columns of Tables 2 and 3. The dotted line connects non-dominated methods; higher and further left is better. False abstentions are reported in Table 3.

Unanswerable questions. Table 3 shows that SOMA substantially improves abstention on unanswerable questions while shortening reasoning. On Qwen3-8B, it raises average abstention from 70.5% to 87.2%, the highest among all methods, and reduces tokens by 32%. On R1-Distill-Llama-8B, it raises abstention from 38.2% to 59.4% with the largest token reduction of all methods (48%). The gains are largest on SUM, the harder benchmark, where abstention rises from 60% to 87% and from 28% to 70%. On SUM, SOMA exceeds AtU on Qwen3-8B (87% vs. 83%) and is one percentage point lower on R1-Distill-Llama-8B (70% vs. 71%), even though AtU trains a probe on SUM for each S2 model, whereas SOMA treats S2 as a black box and needs no training. These gains come with a moderate rise in false abstentions on answerable questions, from 2.4% to 8.0% and from 2.1% to 4.8% on average. On UMWP, SOMA raises the abstention of Qwen3-8B from 81% to 88%. For R1-Distill-Llama-8B, it keeps accuracy and false abstention unchanged while cutting tokens by 59%, but abstention stays close to Vanilla (49% vs. 48%). Across both scenarios and both S2 models, SOMA lies on the token–quality Pareto frontier (Figure 3), the only method to do so in all four settings.

## 4.3 Ablation Study

We study the components of the control layer on a fixed subset of the test sets (Table 4); Appendix D varies the scan interval. All variants share the S1 model, monitors and system prompt.

Reminders and takeovers. Remind-only inserts reminders but never takes over (h = ∞). In overthinking, reminders alone already shorten reasoning (by 16% on Qwen3-8B and 3% on R1-Distill-Llama-8B) without lowering accuracy, and adding takeovers brings total savings to 42% and 15%.

For unanswerable questions, reminders alone leave abstention close to Vanilla, so the gain in abstention comes from the takeover. No reminder skips the first stage: it accumulates evidence from the first pause and takes over without reminding S2, using its own calibrated threshold. For unanswerable questions it performs similarly to SOMA. In overthinking, the threshold calibrated without a reminder is much higher, and the variant saves about half as many tokens on Qwen3-8B (22% vs. 42%) with accuracy 1.6 percentage points higher; on R1- Distill-Llama-8B it saves 11% against 15% at similar accuracy.

Case studies. Appendix F shows both stages at work. The typical success is a model that notices a problem but does not act on it: on a SUM question whose denominator has no last term, Qwen3- 8B names the gap at once, yet without SOMA it answers under an assumption after 6,148 tokens; under SOMA, S1 keeps observing the same search after the reminder, and the takeover turns S2’s own doubt into an abstention after 1,392 tokens. In overthinking, a takeover keeps a confirmed answer, cutting 21,665 tokens to 1,120. The reminder alone can also suffice: R1-Distill-Llama-8B guesses a missing number on a SUM question without SOMA, but after the reminder it re-checks the problem and abstains (4,903 to 1,149 tokens).

## 5 Conclusion

We introduced SOMA, in which an S1 decision model continuously monitors S2’s reasoning for a behavior described in natural language, reminds S2 when it appears, and takes over only if the behavior persists after the reminder. SOMA requires neither training nor access to S2’s internal states, since S1 reads only the visible reasoning text. On overthinking, SOMA cut reasoning tokens by 36% and 16% on the two S2 models without lowering average accuracy, and on unanswerable questions it raised abstention by 17 and 21 points.

Because objectives are written in natural language and S1 reads only visible text, adding a new objective requires only a monitor statement, a reminder and a takeover action, with no change to S2. S1 decision models can therefore serve as a general supervision layer that accompanies reasoning rather than gating it, and in future work the same layer can take on new objectives such as jailbreak defense or agent monitoring.

## Limitations

Scope of the evaluation. SOMA is designed as a general framework: a new objective needs only a monitor statement, a reminder and a takeover action, so the same layers could address other problems that surface in the reasoning, such as defending against jailbreaks. Due to time constraints, we evaluate only two scenarios, overthinking and unanswerable questions, both centered on mathematical and scientific reasoning, with two 8B reasoning models and one monitor per scenario. How well SOMA works in other scenarios and domains, with larger S2 models, and with several monitors running jointly remains to be tested.

A single S1 model. All experiments use Jev, a hosted proprietary model whose behavior may change across versions. The framework only requires the S1 interface, but we test no other S1 model in the main experiments. The results therefore do not establish which other S1 models can fill the role.

Cost of S1 calls. Jev is accessed through a paid API, and SOMA queries it at every pause. Our results come from the single-run readout (Appendix B); reported tokens cover S2 only, and we do not measure wall-clock latency. As mature opensource S1 models become available, S1 could be deployed on the same hardware as S2, removing the API cost and the network round trip.

Irreversible interventions. A takeover cannot be undone by S2. Although thresholds are chosen to limit harm on calibration data, SOMA raises false abstentions on answerable questions, and a mistaken takeover in overthinking fixes an answer that S2 might still have corrected.

## Broader Impact

SOMA lowers the cost of supervising a reasoning model: an objective is written as a natural-language statement rather than a task-specific signal or a trained probe, and S1 reads only the visible reasoning. This could make oversight of long reasoning and agent trajectories practical for users without access to model internals, including for safety objectives such as detecting jailbreak attempts. The same mechanism carries risks. A takeover overrides the model’s own reasoning, so a poorly written statement or a badly calibrated threshold can systematically suppress valid answers. Relying on a hosted S1 also sends the model’s reasoning to a third-party service, which matters when the reasoning contains private data; a locally deployed open-source S1 would avoid this.

## References

Rakefet Ackerman. 2014. The diminishing criterion model for metacognitive regulation of time investment. Journal ofExperimental Psychology: General, 143(3):1349–1368.

Rakefet Ackerman and Valerie A. Thompson. 2017. Meta-reasoning: Monitoring and control of thinking and reasoning. Trends in Cognitive Sciences, 21(8):607–617.

Pranjal Aggarwal, Aman Madaan, Ankit Anand, Srividya Pranavi Potharaju, Swaroop Mishra, Pei Zhou, Aditya Gupta, Dheeraj Rajagopal, Karthik Kappaganthu, Yiming Yang, Shyam Upadhyay, Manaal Faruqui, and Mausam. 2024. AutoMix: Automatically mixing language models. In Advances in Neural Information Processing Systems.

Pranjal Aggarwal and Sean Welleck. 2025. L1: Controlling how long a reasoning model thinks with reinforcement learning. In Conference on Language Modeling (COLM).

Jihyun Janice Ahn, Ryo Kamoi, Berk Atil, Renze Lou, WonWoo Kang, Heehyun Park, Sarkar Snigdha Sarathi Das, Zhuoyang Zou, Xiaoxin Lu, Yusen Zhang, Asfahan Shah, Ridwanul Hasan Tanvir, Lingxiao Zhao, Hongxi Huang, Vignesh Venkatesh, Dianjun Lin, Hamid Shah, Wentao Wang, Zhanpeng Song, and 18 others. 2026. Bridging the know-act gap via task-level autoregressive reasoning. Preprint, arXiv:2603.22619.

Diogo Almeida. 2026. Introducing System One models & Jev. TypeSafe AI blog. Accessed 5 October 2026.

Marianna Bergamaschi Ganapini, Murray Campbell, Francesco Fabiano, Lior Horesh, Jon Lenchner, Andrea Loreggia, Nicholas Mattei, Francesca Rossi, Biplav Srivastava, and Kristen Brent Venable. 2025. Fast, slow, and metacognitive thinking in AI. npj Artificial Intelligence, 1(1):27.

Lingjiao Chen, Matei Zaharia, and James Zou. 2024. FrugalGPT: How to use large language models while reducing cost and improving performance. Transactions on Machine Learning Research.

Xingyu Chen, Jiahao Xu, Tian Liang, Zhiwei He, Jianhui Pang, Dian Yu, Linfeng Song, Qiuzhi Liu, Mengfei Zhou, Zhuosheng Zhang, Rui Wang, Zhaopeng Tu, Haitao Mi, and Dong Yu. 2025. Do NOT think that much for 2+3=? on the overthinking of long reasoning models. In Proceedings ofthe 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 9487–9499. PMLR.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. 2021. Training verifiers to solve math word problems. Preprint, arXiv:2110.14168.

Contrastive-LM. 2026. CLM-v0.1-8B (contrastive language model) model card. Hugging Face model repository. Apache-2.0; accessed 5 October 2026.

Convai Innovations. 2026. Laya: a multilingual nonautoregressive System 1 decision model (model card). Hugging Face model repository. Apache-2.0; accessed 5 October 2026.

Wim De Neys and Tamara Glumicic. 2008. Conflict monitoring in dual process theories of thinking. Cognition, 106(3):1248–1299.

Tobias Deußer, Lorenz Sparrenberg, and Rafet Sifa. 2026. Evaluating and benchmarking the System One model Jev. Preprint, arXiv:2609.37647.

Haonan Dong, Haoran Ye, Wenhao Zhu, Kehan Jiang, and Guojie Song. 2025. Meta-R1: Empowering large reasoning models with metacognition. Preprint, arXiv:2508.17291.

Jonathan St. B. T. Evans and Keith E. Stanovich. 2013. Dual-process theories of higher cognition: Advancing the debate. Perspectives on Psychological Science, 8(3):223–241.

Chenrui Fan, Ming Li, Lichao Sun, and Tianyi Zhou. 2025. Missing premise exacerbates overthinking: Are reasoning models losing critical thinking skill? In Conference on Language Modeling (COLM).

Gongfan Fang, Xinyin Ma, and Xinchao Wang. 2025. Thinkless: LLM learns when to think. In Advances in Neural Information Processing Systems, volume 38.

John H. Flavell. 1979. Metacognition and cognitive monitoring: A new area of cognitive–developmental inquiry. American Psychologist, 34(10):906–911.

Yichao Fu, Junda Chen, Siqi Zhu, Zheyu Fu, Zhongdongming Dai, Yonghao Zhuang, Yian Ma, Aurick Qiao, Tajana S. Rosing, Ion Stoica, and Hao Zhang. 2025a. Efficiently scaling LLM reasoning programs with Certaindex. In Advances in Neural Information Processing Systems (NeurIPS), pages 157566– 157604.

Yichao Fu, Junda Chen, Yonghao Zhuang, Zheyu Fu, Ion Stoica, and Hao Zhang. 2025b. Reasoning without self-doubt: More efficient chain-of-thought through certainty probing. In ICLR 2025 Workshop on Foundation Models in the Wild.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, et al. 2025. DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning. Nature, 645(8081):633–638.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. 2021. Measuring mathematical problem solving with the MATH dataset. In Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks.

Hakan Inan, Kartikeya Upasani, Jianfeng Chi, Rashi Rungta, Krithika Iyer, Yuning Mao, Michael Tontchev, Qing Hu, Brian Fuller, Davide Testuggine, and Madian Khabsa. 2023. Llama Guard: LLMbased input-output safeguard for human-AI conversations. Preprint, arXiv:2312.06674.

Guochao Jiang, Guofeng Quan, Zepeng Ding, Ziqin Luo, Dixuan Wang, and Zheng Hu. 2025. Flash-Think: An early exit method for efficient reasoning. Preprint, arXiv:2505.13949.

Saurav Kadavath, Tom Conerly, Amanda Askell, Tom Henighan, Dawn Drain, Ethan Perez, Nicholas Schiefer, Zac Hatfield-Dodds, Nova DasSarma, Eli Tran-Johnson, Scott Johnston, Sheer El-Showk, Andy Jones, Nelson Elhage, Tristan Hume, Anna Chen, Yuntao Bai, Sam Bowman, Stanislav Fort, and 17 others. 2022. Language models (Mostly) know what they know. Preprint, arXiv:2207.05221.

Daniel Kahneman. 2011. Thinking, Fast and Slow. Farrar, Straus and Giroux.

Vedant Khandelwal, Francesca Rossi, Keerthiram Murugesan, Erik Miehling, Murray Campbell, Karthikeyan Natesan Ramamurthy, and Lior Horesh. 2025. Language models coupled with metacognition can outperform reasoning models. Preprint, arXiv:2508.17959.

Polina Kirichenko, Mark Ibrahim, Kamalika Chaudhuri, and Samuel J. Bell. 2025. AbstentionBench: Reasoning LLMs fail on unanswerable questions. In Advances in Neural Information Processing Systems (Datasets and Benchmarks Track), volume 38.

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. 2023. Efficient memory management for large language model serving with PagedAttention. In Proceedings ofthe 29th Symposium on Operating Systems Principles, pages 611–626. ACM.

Hynek Kydlícek. 2025.ˇ Math-Verify: Math verification library. GitHub repository, Hugging Face.

Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. 2024. Let’s verify step by step. In International Conference on Learning Representations (ICLR).

Bill Yuchen Lin, Yicheng Fu, Karina Yang, Faeze Brahman, Shiyu Huang, Chandra Bhagavatula, Prithviraj Ammanabrolu, Yejin Choi, and Xiang Ren. 2023. SwiftSage: A generative agent with fast and slow

thinking for complex interactive tasks. In Advances in Neural Information Processing Systems, volume 36.

Weizhe Lin, Xing Li, Zhiyuan Yang, Xiaojin Fu, Huiling Zhen, Yaoyuan Wang, Xianzhi Yu, Wulong Liu, Xiaosong Li, and Mingxuan Yuan. 2026. TrimR: Verifier-based training-free thinking trimming for efficient test-time scaling. In International Conference on Learning Representations (ICLR).

Yi Liu, Xiangyu Liu, Zequn Sun, and Wei Hu. 2026. Answering the unanswerable is to err knowingly: Analyzing and mitigating abstention failures in large reasoning models. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 40, pages 32231–32239.

Mathematical Association of America. 2023. American Mathematics Competitions (AMC) 2023. Problems as released in https://huggingface.co/ datasets/math-ai/amc23.

Mathematical Association of America. 2025. American Invitational Mathematics Examination (AIME) 2024 and 2025. Problems as released in https://huggingface. co/datasets/HuggingFaceH4/aime\_2024 and https://huggingface.co/datasets/ opencompass/AIME2025.

Microsoft. 2026. Microsoft-Decision-1 in Microsoft Foundry. Microsoft Learn. Accessed 12 October 2026.

Niklas Muennighoff, Zitong Yang, Weijia Shi, Xiang Lisa Li, Li Fei-Fei, Hannaneh Hajishirzi, Luke Zettlemoyer, Percy Liang, Emmanuel Candès, and Tatsunori Hashimoto. 2025. s1: Simple test-time scaling. In Proceedings ofthe 2025 Conference on Empirical Methods in Natural Language Processing, pages 20275–20321.

Thomas O. Nelson and Louis Narens. 1990. Metamemory: A theoretical framework and new findings. In Gordon H. Bower, editor, The Psychology ofLearning and Motivation, volume 26, pages 125–173. Academic Press.

Isaac Ong, Amjad Almahairi, Vincent Wu, Wei-Lin Chiang, Tianhao Wu, Joseph E. Gonzalez, M Waleed Kadous, and Ion Stoica. 2025. RouteLLM: Learning to route LLMs from preference data. In International Conference on Learning Representations (ICLR).

OpenAI. 2026. Decisions API documentation. Online documentation. Model gpt-6-luna; accessed 12 October 2026.

E. S. Page. 1954. Continuous inspection schemes. Biometrika, 41(1/2):100–115.

Keqin Peng, Liang Ding, Yuanxin Ouyang, Meng Fang, and Dacheng Tao. 2025. Revisiting overthinking in long chain-of-thought from the perspective of selfdoubt. Preprint, arXiv:2505.23480.

Riccardo Porcedda. 2026. Jev thinks “I don’t know”, but doesn’t say it: Introducing Sys1Cal-v1 dataset for probability calibration. Preprint, arXiv:2609.35342.

Amir Rafe and Subasish Das. 2026. Benchmarking System One decision models against trained classifiers and language models for automated decision gates. Preprint, arXiv:2610.00346.

David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R. Bowman. 2024. GPQA: A graduate-level Google-proof Q&A benchmark. In First Conference on Language Modeling.

Steven A. Sloman. 1996. The empirical case for two systems of reasoning. Psychological Bulletin, 119(1):3– 22.

Linxin Song, Taiwei Shi, and Jieyu Zhao. 2025. The hallucination tax of reinforcement finetuning. In Findings of the Association for Computational Linguistics: EMNLP 2025, pages 2105–2120, Suzhou, China. Association for Computational Linguistics.

Yang Sui, Yu-Neng Chuang, Guanchu Wang, Jiamu Zhang, Tianyi Zhang, Jiayi Yuan, Hongyi Liu, Andrew Wen, Shaochen Zhong, Na Zou, Hanjie Chen, and Xia Hu. 2025. Stop overthinking: A survey on efficient reasoning for large language models. Transactions on Machine Learning Research.

YuHong Sun, Zhangyue Yin, Qipeng Guo, Jiawen Wu, Xipeng Qiu, and Hui Zhao. 2024. Benchmarking hallucination in large language models based on unanswerable math word problem. In Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024), pages 2178–2188, Torino, Italia. ELRA and ICCL.

Valerie A. Thompson, Jamie A. Prowse Turner, and Gordon Pennycook. 2011. Intuition, reason, and metacognition. Cognitive Psychology, 63(3):107– 140.

TypeSafe AI. 2026. TypeSafe documentation: the Jev System One model, Noul questions, and the /v1/systemone API. Online documentation. Model jev-1.13.0; accessed 5 October 2026.

Xinyan Wang, Xiaogeng Liu, Ming Pei, and Chaowei Xiao. 2026. ROM: Real-time overthinking mitigation via streaming detection and intervention. Preprint, arXiv:2603.22016.

Zhibin Wen, Tao Han, Lei Bai, Can Li, and Yang Xu. 2026. MetaCtrl: Your large language models can reason better and more concisely with a metacognitive controller. Preprint, arXiv:2609.37304.

Yangxinyu Xie, Tao Wang, Soham Mallick, Yan Sun, Georgy Noarov, Mengxin Yu, Tanwi Mallick, and Edgar Dobriban. 2026. Statistical early stopping for reasoning models. In Proceedings ofthe 43rd International Conference on Machine Learning (ICML).

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. 2025. Qwen3 technical report. Preprint, arXiv:2505.09388.

Chenxu Yang, Qingyi Si, Yongjie Duan, Zheliang Zhu, Chenyu Zhu, Qiaowei Li, Minghui Chen, Zheng Lin, and Weiping Wang. 2026. Dynamic early exit in reasoning models. In International Conference on Learning Representations (ICLR).

Anqi Zhang, Yulin Chen, Jane Pan, Chen Zhao, Aurojit Panda, Jinyang Li, and He He. 2025a. Reasoning models know when they’re right: Probing hidden states for self-verification. In Conference on Language Modeling (COLM).

Jiajie Zhang, Nianyi Lin, Lei Hou, Ling Feng, and Juanzi Li. 2025b. AdaptThink: Reasoning models can learn when to think. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 3716–3730, Suzhou, China. Association for Computational Linguistics.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. 2023. Judging LLM-as-a-judge with MT-Bench and Chatbot Arena. In Advances in Neural Information Processing Systems, volume 36.

## A Implementation Details

## A.1 Models and Decoding

The S2 models are Qwen3-8B (Yang et al., 2025) and DeepSeek-R1-Distill-Llama-8B (Guo et al., 2025), and Jev (Almeida, 2026; TypeSafe AI, 2026) is the only S1 model. Every method samples each question once with a generation limit of 32,768 tokens and the recommended decoding settings of each model: temperature 0.6 and top-p 0.95, with top-k 20 for Qwen3-8B. Mathematical answers are scored with Math-Verify (Kydlícekˇ , 2025), GPQA-Diamond uses its official answer labels, SUM and UMWP use their official answerability labels, and abstentions are identified with the rule of Liu et al. (2026). SOMA and its variants use the fixed system prompt shared by both scenarios:

“While you think, a monitor may insert a <reminder>. When you see one, follow its instruction briefly inside your reasoning, then continue as you judge best.”

## A.2 S1 State and Scan Schedule

At each scan, Jev receives the complete task and the S2 text in fields named “Task: $Q ^ { \ast }$ and “Reasoning so far: $H '$ , separated by blank lines. The state is limited to 2,048 tokens under the Qwen3 tokenizer, after reserving room for the longest monitor statement. The complete task is retained, and the oldest part of H is removed when the limit is exceeded. SOMA scans only while S2 is thinking. After every 256 thinking tokens, S2 continues to the next line break and pauses there; if no line break appears within 128 further tokens, it pauses without one. Reminders are inserted at the pause as a separate paragraph wrapped in <reminder> tags. They become part of S2’s context but are excluded from H on later scans.

## A.3 Takeovers

In overthinking, a takeover appends the model’s end-of-thinking delimiter and “\*\*Final Answer\*\*”, and S2 then writes at most 64 tokens. For an unanswerable question, it inserts the fixed conclusion “The problem is missing necessary information or has contradictory conditions, so it cannot be answered as stated.”, closes thinking, pre-fills “\boxed{I don’t know.}”, and allows S2 512 tokens for the reason.

## B Single-Run Readout

Since a takeover ends generation, running every candidate threshold separately would multiply the cost of calibration by the size of the grid. We therefore run SOMA once in an observe-remind mode: reminders are inserted as usual, but no takeover is executed, and every scan is recorded. A trial takeover is recorded whenever the post-reminder S first reaches a candidate threshold. The outcome under a threshold h is the trial at the first pause where $S \geq h ,$ or the full trajectory if h is never reached; $h = \infty$ gives the remind-only variant. Up to the first reminder, SOMA coincides with S2 running under the same system prompt without intervention, so each such trajectory is generated once and SOMA continues from it. The same readout, with the frozen thresholds, is used for evaluation. At the pause where the takeover would have happened, the outcome and token count are computed as follows.

• Overthinking. We append the takeover text and let S2 write its final answer (at most 64 tokens). The answer is scored as usual. The token count is the number of tokens S2 wrote before the pause, plus the takeover text and the final answer.

• Unanswerable questions. The pre-filled refusal determines the outcome, so the trial is scored as an abstention. S2 then writes the reason (at most 512 tokens), and the token count is the number of tokens S2 wrote before the pause, plus the tokens of the generated reason.

<table><tr><td>S2</td><td>Variant</td><td>h Takeovers</td><td></td><td>+1-</td></tr><tr><td colspan="5">Overthinking</td></tr><tr><td>Qwen3</td><td>SOMA</td><td>0.25</td><td>161</td><td>+3/-1</td></tr><tr><td>Qwen3</td><td>No reminder</td><td>6</td><td>101</td><td>+1/-0</td></tr><tr><td>Llama-R1</td><td>SOMA</td><td>1.5</td><td>70</td><td>+3/-2</td></tr><tr><td>Llama-R1</td><td>No reminder</td><td>3</td><td>74</td><td> $+ 0 / { - 0 }$ </td></tr><tr><td colspan="5">Unanswerable</td></tr><tr><td>Qwen3</td><td>SOMA</td><td>4</td><td>30</td><td> $+ 1 8 / - 3$ </td></tr><tr><td>Qwen3</td><td>No reminder</td><td>8</td><td>33</td><td> $+ 1 8 / - 2$ </td></tr><tr><td>Llama-R1</td><td>SOMA</td><td>3</td><td>38</td><td> $+ 3 1 / - 1$ </td></tr><tr><td>Llama-R1</td><td>No reminder</td><td>2.5</td><td>42</td><td> $+ 3 2 / - 1$ </td></tr></table>

Table 5: Calibration outcomes at the selected thresholds, pooled over the two calibration sets of each scenario (200 questions per scenario). Takeovers counts the questions taken over; +/− counts questions turned from wrong to correct and from correct to wrong relative to S2 under the same system prompt without intervention.

## C Calibration and Ablation Data

The calibration sets are disjoint from all test questions, and each scenario has an easier and a harder set of 100 questions. For overthinking, the easier set consists of MATH training problems sampled to match the difficulty distribution of MATH-500, and the harder set of problems from AIME 2020–2023. For unanswerable questions, the easier set contains 50 answerable and 50 unanswerable problems drawn from the original UMWP release but excluded from its test split, and the harder set contains 50 SUM training pairs, each consisting of an answerable problem and its unanswerable counterpart. We search $\mathcal { H } \ = \ \{ 0 . 2 5 , 0 . 5 , 0 . 7 5 , 1 , 1 . 5 , 2 , 2 . 5 , 3 , 4 , 6 , 8 , 1 2 \}$ with δ = 0. Table 5 reports the calibration outcomes at the selected thresholds of SOMA and of the No-reminder variant.

The ablations in §4.3 use a fixed subset of the test sets, selected by hashing question identifiers: all of MATH-500 (500 questions) and AIME (60) for overthinking, and 100 SUM pairs and 100 answerable plus 100 unanswerable UMWP questions for unanswerable questions.

## D Scan Interval

Scanning more often lets SOMA notice a behavior earlier (Table 6). Halving the interval to X = 128 increases savings in both scenarios (to 48% and

<table><tr><td></td><td colspan="4">Overthinking Unanswerable</td></tr><tr><td>Variant</td><td>Acc↑</td><td colspan="3">Tok Abs↑ Acc↑ FA↓</td></tr><tr><td>Qwen3-8B</td><td></td><td></td><td></td><td></td></tr><tr><td>Vanilla</td><td>93.0 6,745</td><td>72</td><td>90</td><td>33,494</td></tr><tr><td>X=128</td><td>92.9 -48%</td><td>93</td><td>85</td><td>10 -36%</td></tr><tr><td>X=256</td><td>92.5 -42%</td><td>92</td><td>83</td><td>10 -29%</td></tr><tr><td>X=512</td><td>94.1 -34%</td><td>88</td><td>86</td><td>6 -27%</td></tr><tr><td colspan="3">R1-Distill-Llama-8B</td><td></td><td></td></tr><tr><td>Vanilla</td><td>85.0 5,299</td><td>36</td><td>84</td><td>33,972</td></tr><tr><td>X=128</td><td>84.8 -19%</td><td>63</td><td>83</td><td>4 -51%</td></tr><tr><td>X=256</td><td>85.5 -15%</td><td>60</td><td>82</td><td>4 -48%</td></tr><tr><td>X=512</td><td>84.8 -8%</td><td>55</td><td>82</td><td>4 -43%</td></tr></table>

Table 6: SOMA with different scan intervals X (thinking tokens) on the ablation subset of Table 4. X=256 is the main setting; the other intervals reuse its thresholds.

19% in overthinking, and to 36% and 51% for unanswerable questions) and raises abstention on R1-Distill-Llama-8B from 60% to 63%, at the cost of about 1.5–2 times as many S1 calls. Doubling it to $X ~ = ~ 5 1 2$ roughly halves the S1 calls but reduces savings and, on R1-Distill-Llama-8B, abstention, while slightly raising overthinking accuracy on Qwen3-8B through fewer takeovers. The interval thus trades monitoring cost against responsiveness, and X = 256 lies between the two.

## E Baselines

We reproduce all baselines from their released code, following the specifications reported in their papers. Where a method has parameters that must be tuned, we select them with the procedure of its paper on the same calibration sets used for SOMA’s thresholds (Appendix C), separately for each S2 model and scenario. All baselines share the task prompts, decoding settings, data and generation limit of §4.1, while keeping their own intervention prompts.

Budget forcing (BF) (Muennighoff et al., 2025). A length-control method that caps thinking at a fixed token budget: once the budget is reached, thinking is cut off and the model is prompted to give its final answer. The budget is selected on the calibration sets.

DEER (Yang et al., 2026). An early-exit method based on the reasoner’s own confidence. At reasoning transitions such as “Wait”, the reasoner writes a trial answer, and thinking ends once its confidence in that answer is high enough, using the threshold of the paper.

Dynasor-CoT (Fu et al., 2025b,a). An early-exit method based on answer consistency. At fixed intervals, an intermediate answer is elicited and then discarded, and the method exits once several consecutive answers agree and show no hesitation. The probing interval and the number of consecutive answers are selected on the calibration sets.

Answering the Unanswerable (Liu et al., 2026). A white-box method that monitors S2’s hidden states with a linear probe trained for each model. When the probe indicates that the question is unanswerable, it inserts guidance that leads S2 to abstain. We use the released probe weights and select the probe’s decision threshold on the calibration sets.

Statistical early stopping (Xie et al., 2026). A method that tracks how often uncertainty keywords appear in the reasoning and stops when a statistical test finds that they appear more often than is typical for answerable questions. Its statistics for answerable questions are estimated on the answerable questions of the calibration sets, and the remaining settings follow the paper. The paper proposes a renewal and a maxwise stopping rule; we report the renewal rule, the stronger of the two in our runs. Because the method does not specify an action after stopping, it uses our refusal takeover.

AbstentionBench prompt (Kirichenko et al., 2025). A prompt-only baseline that adds the official AbstentionBench system prompt, which encourages the model to abstain when a question cannot be answered.

## F Case Studies

This appendix shows how SOMA behaves on individual questions, with four successful and two failed cases. All cases come from the main test run. For each case we show the question, the vanilla response, and SOMA’s monitoring trace. At every scan the trace lists the position in S2 tokens, S1’s probability p that the monitor statement holds, the evidence after the update, and the action.

## F.1 Successful cases

Case 1 • UMWP, R1: a takeover ends an assumption   
loop   
Question. A pole is 20 meters long. It got cut for many   
pieces in a way that left it 30% shorter. How long is the   
maximum piece of pole?   
Vanilla (32,768 tokens, stops at the generation limit   
without an answer).   
Calculating 30% of20 meters: 0.3 × 20 = 6 meters.   
So, the total length after cutting is 20 − 6 = 14

meters. [. . . ] Wait, but the problem says it’s cut into   
many pieces [. . . ]   
(near the end) I think I need to conclude that the   
maximum possible length is approaching 14 meters,   
but since it’s expecting a specific answer, maybe 10   
meters. But I’m not sure. Wait, maybe the problem   
is expecting 10 meters, but I don’t have a logical   
reason.   
The response contains 694 occurrences of “Wait” and   
never settles.   
SOMA trace. At token 283, S1 gives p = 0.22 and S   
stays 0. At token 554, p = 0.75 makes S positive (1.10):   
the reminder is inserted and S is reset to 0. The next two   
scans give p = 0.87 and p = 0.90, raising S to 1.90 and   
then to 4.10 ≥ h = 3, and the takeover fires at token   
1,189.   
Before the reminder.   
Wait, maybe the question is askingfor the maximum   
possible length regardless ofthe number ofpieces, so   
theoretically, the maximum possible is 14 meters. But   
since it’s cut into many pieces, maybe it’s implying   
that it’s split into more than one, but not necessarily   
a specific number.   
After the reminder.   
Wait, the problem says it was cut into many pieces,   
leaving it 30% shorter. [. . . ] The term “many” is   
a bit vague, but in math problems, often “many” is   
used to imply two or more. So, if it’s cut into two   
pieces, each would be 7 meters.   
Takeover output (token 1,189).   
The problem is missing necessary information   
or has contradictory conditions, so it cannot   
be answered as stated.   
</think>   
\boxed{I don’t know.}   
Reason{The problem does not specify the   
number of pieces, making it impossible to   
determine the maximum length of any single   
piece.}   
Outcome. Correct abstention after 1,245 tokens instead   
of an unfinished 32,768-token response.

The S2 model notices the ambiguity before the reminder but keeps resolving it by assumption. The check reminder does not change this: S2 decides that “many” means two pieces. S1 keeps observing the same behaviour (p = 0.87, 0.90), the postreminder evidence reaches h after two scans, and the takeover ends the episode. The reason is written by S2 and names the missing quantity.

Case 2 • SUM, Qwen3-8B: noticed at the start, acted   
on only through the takeover   
2 + 4 + 6 + · · · + 34   
Question. = ? (the last term   
3 + 6 + 9 + · ·   
of the denominator is missing)   
Vanilla (6,148 tokens, answers <sup>2</sup> ).   
Wait, actually, the denominator is written as 3 + 6 +   
9 + · · · but it doesn’t specify the last term. Hmm, that   
might be an issue. Let me check again. [. . . ]   
(final answer) Based on the assumption that both the

<table><tr><td>Case</td><td>Scenario</td><td>S2</td><td>Decisive step</td><td>Outcome (vanilla → SOMA)</td><td>Vanilla</td><td>SOMA</td></tr><tr><td>1</td><td>UMWP</td><td>R1</td><td>takeover</td><td>no answer → abstains</td><td>32,768</td><td>1,245</td></tr><tr><td>2</td><td>SUM</td><td>Qwen</td><td>takeover</td><td>answers under an assumption → abstains</td><td>6,148</td><td>1,392</td></tr><tr><td>3</td><td>SUM</td><td>R1</td><td>reminder</td><td>guesses the missing number → abstains</td><td>4,903</td><td>1,149</td></tr><tr><td>4</td><td>MATH-500</td><td>R1</td><td>takeover</td><td>correct → correct</td><td>21,665</td><td>1,120</td></tr><tr><td>5</td><td>MATH-500</td><td>R1</td><td>takeover</td><td>correct → wrong intermediate answer</td><td>1,679</td><td>3,147</td></tr><tr><td>6</td><td>SUM</td><td>Qwen</td><td>takeover</td><td>correct → false refusal</td><td>5,544</td><td>2,436</td></tr></table>

Table 7: Cases in this appendix. The last two columns give the tokens generated by S2.

numerator and denominator sequences have the same   
number ofterms (17), thefinal result is: $\begin{array} { l } { { \left[ \begin{array} { l } { 2 } \\ { 3 } \end{array} \right] } } \end{array}$   
SOMA trace. At the first scan (token 384), p = 0.63   
makes S positive (0.53): the reminder is inserted and S   
is reset to 0. The next three scans (tokens 645, 925 and   
1,309) give $p = 0 . 8 5 ,$ , 0.88 and 0.94, raising S to 1.73,   
3.73 and $6 . { \dot { 4 } } 8 \geq h = 4 ,$ and the takeover fires at token   
1,309.   
After the reminder.   
Now the denominator is $3 + 6 + 9 + \cdots$ but what’s the   
last term? The problem didn’t specify. Wait, maybe   
it’s supposed to end at the same term as the numer  
ator? $\ddot { I } . . . J$ If they are supposed to have the same   
number ofterms? Or maybe the denominator ends at   
the same value as the numerator?   
Takeover output (token 1,309).   
The problem is missing necessary information   
or has contradictory conditions, so it cannot   
be answered as stated.   
</think>   
\boxed{I don’t know.}   
Reason{The problem does not specify the last   
term of the denominator sequence $\overline { { ( 3 \mathrm { ~ + ~ } 6 \mathrm { ~ + ~ } 9 } }$   
$^ + \cdot \cdot \cdot )$ Without knowing the endpoint of the   
denominator, it is impossible to compute its   
sum and thus determine the value of the given   
fraction.}   
Outcome. Correct abstention after 1,392 tokens; vanilla   
spends 6,148 tokens and answers under an explicit as  
sumption.

This is the “knowing but not acting” pattern that motivates SOMA: vanilla names the missing endpoint in its first paragraph and still produces a definite answer. With SOMA, S2 again searches for an implied endpoint after the reminder; S1’s evidence grows on every scan, and the takeover turns the doubt S2 already expressed into an abstention.

Case 3 • SUM, R1: the reminder alone leads to an   
abstention   
Question. What is the value of ${ \frac { ( 2 1 1 2 - ) ^ { 2 } } { 1 6 9 } } \ ?$   
Vanilla (4,903 tokens, answers 26244).   
I wonder what the missing part is in the numerator.   
It seems like there’s a blank space where a number   
should be. $I . . . J$   
(final answer) Substitute $x = 6$ into the numerator:

$2 1 1 2 - 6 = 2 1 0 6 . \ : [ \cdot \cdot \cdot ]$ Thus, the value ofthe given   
expression is 26244   
SOMA trace. The first two scans (tokens 276 and 599)   
give $p = 0 . 4 9$ and $p = 0 . 3 5 ,$ so S stays 0. At token   
904, $p = 0 . 6 7$ makes S positive (0.71): the reminder is   
inserted and S is reset to 0. S2 ends its reasoning before   
the next scan, so no takeover occurs.   
Before the reminder.   
Wait, perhaps the missing number is 12? Because   
2112 minus 12 is 2100, and 2100 squared is a big   
number, but 2100 divided by 169 is approximately   
12.36, which isn’t an integer.   
After the reminder.   
After checking again, the problem still lacks infor  
mation about the missing number in the numerator.   
Therefore, I can’t solve it without knowing what’s   
being subtractedfrom 2112. So, I think the correct   
response here is to state that the problem lacks suffi  
cient information and that I don’t know the solution.   
Final Answer I don’t know.   
Outcome. Correct abstention after 1,149 tokens, without   
a takeover.

Here the check reminder does what it asks for: S2 re-examines the problem, concludes that information is missing, and ends its reasoning by itself. Such cases are the minority in our runs. On the full SUM test set the reminder alone raises R1’s abstention rate only from 28% (Vanilla) to 30%, and most of SOMA’s gain comes from the takeover.

Case 4 • MATH-500, R1: takeover after a confirmed   
answer   
Question. Simplify ${ \frac { \sec x } { \sin x } } - { \frac { \sin x } { \cos x } } .$   
Vanilla (21,665 tokens, correct: cot x).   
(after reaching cot x and re-verifying it several times,   
a numerical check at $x = \pi / 6$ with an arithmetic   
slip) ${ \frac { \sec { \frac { \pi } { 6 } } } { \sin { \frac { \pi } { 6 } } } } - { \frac { \sin { \frac { \pi } { 6 } } } { \cos { \frac { \pi } { 6 } } } } = { \frac { 2 } { \frac { 1 } { 2 } } } - { \frac { \frac { 1 } { 2 } } { \frac { \sqrt { 3 } } { 2 } } } = 4 - { \frac { 1 } { \sqrt { 3 } } } [ \ldots I$   
So, original expression is $\frac { 1 2 - { \sqrt { 3 } } } { 3 }$ , not cot x. Wait, so   
my algebra is wrong.   
The slip (sec ${ \frac { \pi } { 6 } } { \mathrm { ~ i s ~ } } { \frac { 2 } { \sqrt { 3 } } } ,$ , not 2) starts a long re-derivation;   
the response contains 242 occurrences of “Wait” before   
it answers cot x .   
SOMA trace. At token 333, S1 gives $p = 0 . 0 3 .$ . At token   
$7 1 7 , p = 0 . 6 6$ makes S positive (0.66): the reminder is   
inserted and S is reset to 0. At the next scan (token 1,101),

p = 0.91 gives $S = 2 . 3 1 \geq h = 1 . 5 .$ and the takeover   
fires.   
Before the reminder.   
But wait, cos x is the definition of cot x, right? So,   
does that mean the simplified expression is cot x?   
Hmm, let me double-check my steps to make sure I   
didn’t make a mistake.   
After the reminder.   
Okay, so after verifying, I’m confident that the steps   
are correct. So, the simplifiedform is indeed cot x.   
[. . . ] Let mejust make sure there isn’t another way to   
approach this problem that might lead to a different   
form, but I don’t think so.   
Takeover output (token 1,101).   
</think>   
\*\*Final Answer\*\*   
The simplified expression is \(\boxed{\cot   
x}\).   
Outcome. Correct after 1,120 tokens instead of 21,665   
(−95%).

After the reminder, S2 states that its checks have confirmed the answer, yet it starts another round (“Let me just make sure . . . ”). This is exactly the observable symptom of the monitor statement. S1 scores it with $p = 0 . 9 1$ at the next scan, and the takeover keeps the confirmed answer. Vanilla’s extra verification adds no information; its numerical check even introduces a spurious doubt.

F.2 Failure cases   
Case 5 • MATH-500, R1: the takeover locks in an   
error that the next check would have caught   
Question. Consider the geometric sequence   
${ \frac { 1 \bar { 2 } 5 } { 9 } } , { \frac { 2 5 } { 3 } } , 5 , 3 , \dots$ What is the eighth term of the se  
quence? Express your answer as a common fraction.   
Vanilla (1,679 tokens, correct: 243   
SOMA run before the reminder (an arithmetic slip   
while simplifying $\textstyle { \frac { 2 7 3 , 3 7 5 } { 7 0 3 , 1 2 5 } } ;$ the numerator is divided by 3,   
the denominator is not).   
30,375 ÷ 3: [. . . ] So, 10,125. 78,125: Since it’s   
not divisible by 3, we can’t reduce further. So, the   
fraction is 10,125/78,125. [. . . ] So, 81/625. Is this   
reducible? [. . . ] So, 81/625 is the simplified form.   
SOMA trace. For the first eight scans (tokens 333 to   
2,585), S1 gives p ≤ 0.08. At token 2,842, after S2 has   
written $8 1 \bar { / } 6 2 5 , \bar { p } = 0 . 6 2$ makes S positive (0.49): the   
reminder is inserted and S is reset to 0. At the next scan   
(token 3,122), p = 0.85 gives $S = 1 . 7 3 \geq h = 1 . 5 .$ , and   
the takeover fires. Without the takeover, the following   
scan (token 3,399) would have given $p = 0 . 1 9$ , and S   
would have fallen to 0.28.   
After the reminder.   
Wait, so I have 81/625 as the simplified fraction.   
Let me confirm all the steps to make sure I didn’t   
make a mistake. [. . . ] Reduced the fraction step by   
step, ending up with 81/625. Wait, let me verify the   
multiplication again: 125 × 2187.   
Takeover output (token 3,122).

</think>   
\*\*Final Answer\*\*   
The eighth term of the sequence is   
\boxed{\dfrac{81}{625}}.   
Outcome. Wrong $\big ( \frac { 8 1 } { 6 2 5 } \big )$ . Without the takeover, the same   
trajectory notices the inconsistency right away (“Wait,   
that seems inconsistent. [. . . ] So, I must have made [a   
mistake]”; p drops to 0.19 at the next scan) and ends with   
the correct 243 625 after 6,402 tokens.

S1 judges the visible text correctly: an answer has been reached and S2 is re-checking it without naming a concern. But S1 does not, by design, judge whether the answer is right, and this particular check would have found the error. This is the residual risk of early exit. The reset makes the takeover wait for evidence after the reminder, and the calibrated h keeps such cases rare.

Case 6 • SUM, Qwen3-8B: a false refusal on a stem   
that looks incomplete   
Question. If $3 x ^ { 3 } - 9 x ^ { 2 } + k x - 1 2$ is divisible by x − 3,   
then it is also divisible by:   
Vanilla (5,544 tokens, correct: $\left[ 3 x ^ { 2 } + 4 \right] ) .$   
SOMA trace. The first two scans (tokens 285 and 657)   
give $p = 0 . 4 3$ and $p = 0 . 4 7 .$ , so S stays 0. At token   
944, p = 0.64 makes S positive (0.58): the reminder is   
inserted and S is reset to 0. The next five scans (tokens   
1,253 to 2,341) give p between 0.66 and 0.72, so S grows   
to 0.66, 1.51, 2.31, 3.21 and $4 . 1 5 \geq h = 4 ,$ and the   
takeover fires at token 2,341.   
After the reminder.   
Therefore, the polynomialfactors as $( x - 3 ) ( 3 x ^ { 2 } + 4 ) .$   
[. . . ] But ifthe question is askingfor anotherfactor,   
maybe it’s expecting a linearfactor? [. . . ] maybe the   
original problem had options like $x - 3 , x + 3 , { \overset { . } { 3 } } x ^ { 2 } + 4 ,$   
etc. But since the user hasn’t provided options, I need   
to think again.   
Takeover output (token 2,341).   
The problem is missing necessary information   
or has contradictory conditions, so it cannot   
be answered as stated.   
</think>   
\boxed{I don’t know.}   
Reason{The problem as stated does not provide   
sufficient information to determine the   
additional factor. While the value of \$ k   
\$ can be determined using the Factor Theorem,   
the question asks for another divisor without   
specifying options or further conditions.   
Without additional context or choices, it   
is not possible to definitively identify the   
other factor.}   
Outcome. False refusal; vanilla answers correctly.

S2 has already found the intended factor, but it keeps questioning whether answer options are missing. The monitor statement describes this visible behavior truthfully, and moderate probabilities (p = 0.64 to 0.72) accumulate over five scans until they reach $h = 4 .$ . S1 judges what S2 does, not whether the information S2 misses is really necessary. False refusals therefore arise when S2 voices unwarranted doubts.

Summary. In the successful cases, S1 detects a persistent, observable symptom: an assumption made despite missing information, or verification of an answer that is already confirmed. The reminder occasionally resolves the case by itself; otherwise, the evidence S1 collects after the reminder triggers the takeover. The failures have two distinct causes. The first is a check that would have been productive, which S1 cannot recognize without judging correctness. The second is S2 voicing doubts that S1 correctly observes but that are unwarranted.