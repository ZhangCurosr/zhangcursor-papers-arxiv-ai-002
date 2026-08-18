# When Context Misleads: Intent-Guided Decoding for Robust Retrieval-Augmented Generation

Haolin Jin, Pengyue Yang, Huaming Chen

School of Electrical and Computer Engineering

The University of Sydney

Sydney, NSW, Australia

{haolin.jin, pengyue.yang, huaming.chen}@sydney.edu.au

Abstract—Retrieval-augmented generation (RAG) improves large language models by grounding generation in external evidence, but it also introduces a source trust problem: retrieved context may be useful, irrelevant, or even misleading. Existing RAG systems often apply a fixed trust policy toward retrieved evidence, which can either over-trust incorrect context or underuse context when the user explicitly asks for context-following behavior. Therefore, we propose Intent-Guided Decoding (IGD), a framework that arbitrates between retrieved context and parametric memory according to user intent. IGD uses answerlevel filtering and token-level correction to steer the final decoding trajectory between retrieved context and parametric memory. We evaluate IGD on three faithful QA benchmarks and three factualconflict benchmarks across five LLMs, IGD substantially improves factual recovery, achieving gains of up to 65.4 percentage points on factual-conflict benchmarks over Direct RAG, while preserving or improving strict context-following behavior, this findings highlight the importance of balancing factuality and faithfulness in RAG.

Index Terms—Retrieval-Augmented Generation, Large Language Models, Factuality, Faithfulness

## I. INTRODUCTION

Retrieval-augmented generation (RAG) has become a widely adopted paradigm for connecting large language models (LLMs) with external knowledge. By retrieving relevant evidence at inference time, retrieval-augmented generation extends the static parametric knowledge of LLMs with updatable nonparametric memory [1]. This design improves performance on knowledge-intensive tasks such as open-domain question answering, where models require access to facts beyond their internal parametric knowledge [2]–[4].

However, the effectiveness of RAG depends not only on whether useful evidence is retrieved, but also on whether the model can ground the generation in the retrieved evidence appropriately. In practice, RAG systems may still generate statements that are unsupported by, or even contradictory to, the provided context. RAGTruth systematically annotates nearly 18K RAG responses, and finds that non-factual statements remain common under standard RAG pipelines [5]. On the other hand, retrieval introduces a source-trust problem. Standard RAG pipelines often implicitly treat retrieved context as authoritative, which content may be false, adversarially corrupted, or inconsistent with stable world knowledge [6]. FaithEval formalizes this issue as contextual faithfulness evaluation, and constructs 4.9K high-quality examples across unanswerable, inconsistent, and counterfactual settings, showing that even strong contemporary LLMs often fail to maintain an appropriate degree of faithfulness to given context [7]. More broadly, situated faithfulness argues that models should dynamically calibrate their trust in external context against internal knowledge, rather than blindly following either source [8]. These findings suggest that a central challenge in RAG is not only how to retrieve useful evidence, but also how to decide whether the retrieved evidence should dominate generation [9], [10].

This issue exposes a fundamental trade-off between faithfulness and factuality in RAG generation. In some settings, the user explicitly expects the model to follow the provided context, for example when answering questions about a document, policy, or passage [11]. In other settings, the user expects the model to use retrieved context only as auxiliary evidence and remain skeptical when that context is misleading or conflicts with world knowledge [9], [10]. A fixed RAG behavior is therefore insufficient: always trusting context may improve contextual faithfulness but can harm factual correctness under misleading retrieval. Conversely, always discounting retrieved context may recover factuality against corrupted evidence but undermine user intent in strict context-following scenarios [12]. Practical RAG systems therefore require an intent-aware mechanism that can follow context when the user asks for contextual grounding, while resisting context when the user seeks factual robustness against unreliable retrieval.

To address this challenge, we argue that RAG requires not only retrieval and grounding with external evidence, but also an explicit decoding-time arbitration mechanism that decides when generation should be controlled by retrieved context and when it should rely on parametric memory. In this work, we propose Intent-Guided Decoding (IGD), a decoding-time framework to address both factuality and faithfulness in RAG. In Figure 1, IGD decomposes generation into three conditional branches: a user branch conditioned on the original user prompt, a context branch that explicitly follows the retrieved context, and a memory branch that answers in a closed-book manner. IGD then performs source arbitration at two granularities. First, an Answer-Level Memory Filter handles high-confidence cases by routing to a stable memory answer when the retrieved context is likely misleading. Second, for the remaining cases, we propose a novel Token-Level Correction mechanism to adjust the user branch decoding distribution only when intervention is needed. Specifically, we leverage an activation gate to detect context-memory conflict, a confidence measurement to decide whether generation should move toward context or memory, and reliability scaling to determine the strength of the intervention. As a result, IGD pulls decoding toward memory when retrieved context is unreliable, while still pulling decoding toward context when the user explicitly requests faithful context following.

![](images/5f6577c697a75f403176e92a627a1e211b693893375558dd5d8fa8e4d25e53fa.jpg)  
Fig. 1: The figure illustrates two truth mode examples: a faithful QA case where the retrieved context correctly supports the answer, and a factual-conflict case where the retrieved context is misleading. For each case, IGD first obtains branch-level previews and top-K token distributions, then applies token-level correction only when the context and memory branches exhibit distributional conflict. The correction direction is determined by entropy-based branch confidence and the prompt-mode prior, while the intervention strength is scaled by the reliability of the selected source.

Our method IGD introduces a novel decoding-time control layer for RAG, which shifts the focus from retrieving better evidence to arbitrating which knowledge source should steer generation. Unlike retrieval-centric methods that improve what or when to retrieve [13]–[15], IGD assumes retrieval has already occurred and addresses downstream conflict between retrieved context and parametric memory. Also, unlike context-faithfulness alignment methods that generally encourage stronger adherence to retrieved evidence [16], [17], IGD performs bidirectional trust calibration. It adapts the source preference according to user intent. Moreover, unlike prompt based confidence reasoning or multi-agent evidence aggregation, IGD performs lightweight logit arbitration during generation. Its token-level correction is activated only when the context and memory branches exhibit distributional conflict, enabling local, conservative and intent-aware intervention without retraining the model or replacing the retrieval pipeline.

Our contributions are summarized as follows. First, we formulate the factuality and faithfulness trade-off in RAG as an intent-conditioned source arbitration problem, where the appropriate trust policy depends on whether the user requests truth seeking robustness or strict context following. Second, we introduce IGD, a decoding-time framework that combines answer-level memory filtering with conservative token-level correction over user, context, and memory branches. Third, we evaluated and analysis the performance of IGD on three faithful benchmark settings and three factual-conflict benchmark settings. Fourth, we compare IGD against Direct RAG, confidence reasoning baselines such as SCR/RCR, and a multi-agent baseline in strict mode, and further test whether IGD preserves context-following behavior when the user explicitly asks to follow the provided context. Lastly, we release the replication package in 1.

## II. RELATED WORK

Retrieval-augmented generation combines parametric language models (LM) with external evidence to improve knowledge intensive generation [1], [18]. Early and representative RAG systems differ in how they retrieve, encode, and consume evidence. Fusion-in-Decoder conditions generation on multiple retrieved passages and fuses evidence inside the decoder [4]. Atlas jointly trains a retrieval-augmented model for few-shot knowledge intensive learning [19]. In-Context RALM shows that LMs can benefit from retrieved documents inserted directly into the prompt [20]. Recent work has made retrieval more adaptive, with most remaining retrieval-centric by improving the retriever or training retrieval-aware generators. FLARE retrieves during generation based on predicted future content [13], Self-RAG trains models to retrieve and critique the outputs using reflection tokens [14], and CRAG evaluates retrieval quality and corrects low-confidence retrieved evidence [15].

Other works study hallucination, contextual faithfulness, and knowledge conflict in RAG. RAGTruth provides a corpus to analyze hallucinations in RAG and shows that retrieved evidence does not eliminate unsupported generation [5]. FaithEval evaluates whether models remain faithful under unanswerable, inconsistent, and counterfactual contexts [7]. ClashEval studies conflicts between an LLM’s internal prior and external evidence, showing that models can adopt incorrect retrieved content even when their internal knowledge is correct [6]. Context-DPO improves context faithfulness through preference optimization [16], while FaithfulRAG models fact-level conflicts between retrieved context and parametric knowledge to improve context-faithful generation [17]. Work on situated faithfulness further argues that LLMs should calibrate trust in external context based on both contextual evidence and internal confidence [8]. Our method shares this motivation of trust calibration, but targets a different control point. Instead of uniformly enforcing stronger adherence to retrieved context, IGD performs intent-aware source arbitration directly in the decoding distribution, allowing the model to favor either context or memory according to the prompt mode.

## III. METHOD

Intent-Guided Decoding (IGD) addresses context memory conflict by separating source arbitration into two decision granularities. Our method first applies an answer-level memory filter, which performs hard replacement only for high-confidence cases, all remaining examples are handled by token-level correction, which continuously adjusts the decoding distribution around the original user prompt. Figure 1 illustrates this token level stage: IGD compares the context and memory branch distributions to activate correction only under meaningful conflict, estimates which branch is more reliable through entropy based confidence and source specific reliability, and then applies a signed logit correction that shifts the user branch distribution toward either the context or memory source.

## A. Conditioned Branches

At decoding step t, let $p _ { \mathrm { u s e r } , t } ( v )$ denote the next token distribution induced by the original user prompt for candidate token v. In a standard RAG prompt, this distribution already reflects the interaction between the user instruction, the question, and the retrieved context. IGD augments this original branch with two conditional branches:

$p _ { \mathrm { c t x } , t } ( v ) .$ : a context-following branch explicitly instructed to answer according to the supplied context.

$p _ { \mathrm { m e m } , t } ( v )$ : a memory branch instructed to answer in a closed-book manner, independent of the supplied context.

The user branch is the unmodified RAG branch: it conditions on the original instruction, question, and retrieved context. In contrast, the context branch is a deliberately source specific branch, we instantiate it with a support snippet selected from the retrieved context, rather than the full context, so that $p _ { \mathrm { c t x } , t }$ represents the local evidence most relevant to the question. The support snippet is selected using a lightweight IDF-based lexical localizer over context segments, following the classical termspecificity weighting idea of inverse document frequency [21].

The context branch therefore estimates what the model would generate under a strict context-following prompt, while the memory branch estimates what the model’s parametric knowledge supports when the retrieved context is removed. To reduce prompt sensitivity in closed-book decoding, we implement the memory branch as an ensemble of M prompt variants, following the common use of ensembling to stabilize model predictions [22]. In our implementation, $M \ = \ 3 ,$ corresponding to a base memory prompt, an explicitly closedbook prompt, and a best-guess memory prompt:

$$
p _ { \mathrm { m e m } , t } ( v ) = \frac { 1 } { M } \sum _ { i = 1 } ^ { M } p _ { \mathrm { m e m } , t } ^ { ( i ) } ( v ) .\tag{1}
$$

## B. Answer-Level Memory Filter

The first stage of IGD performs answer-level filtering before decoding, particularly for high-confidence cases in which the memory branch assigns substantially higher likelihood to the memory answer than to the context answer, while the original user branch does not prefer the context answer over the memory answer. Let $\hat { y } _ { \mathrm { c t x } }$ denote a short preview answer generated by the context branch, and let $\hat { y } _ { \mathrm { m e m } }$ denote the memory preview selected from the memory ensemble. For a candidate answer $y = ( y _ { 1 } , \dots , y _ { L } )$ and branch $b ,$ we define the length-normalized answer likelihood, and for the memory ensemble we average this likelihood across memory views:

$$
\begin{array} { c } { { \ell _ { b } ( y ) = \displaystyle \frac { 1 } { L } \sum _ { \ell = 1 } ^ { L } \log p _ { b } ( y _ { \ell } \mid y _ { < \ell } ) , } } \\ { { \ell _ { \mathrm { m e m - e n s } } ( y ) = \displaystyle \frac { 1 } { M } \sum _ { i = 1 } ^ { M } \ell _ { \mathrm { m e m } } ^ { ( i ) } ( y ) . } } \end{array}\tag{2}
$$

The filter evaluates the memory candidate against the context candidate under two distributions:

$$
m _ { M } = \ell _ { \mathrm { m e m - e n s } } ( \hat { y } _ { \mathrm { m e m } } ) - \ell _ { \mathrm { m e m - e n s } } ( \hat { y } _ { \mathrm { c t x } } ) ,\tag{3}
$$

$$
m _ { U } = \ell _ { \mathrm { u s e r } } ( \hat { y } _ { \mathrm { m e m } } ) - \ell _ { \mathrm { u s e r } } ( \hat { y } _ { \mathrm { c t x } } ) .\tag{4}
$$

Here, $m _ { M }$ measures whether the memory branch itself strongly prefers the memory candidate, while m checks whether the same candidate remains compatible with the full user prompt. This second condition prevents the filter from overriding the user instruction with a memory answer that the original prompt distribution strongly disfavors. Since this stage performs hard replacement, IGD uses a conservative dominance test rather than an averaged routing score:

$$
D _ { \mathrm { m e m } } = \mathrm { m i n } \big ( m _ { M } - \log \rho _ { M } , m _ { U } - \log \rho _ { U } \big ) ,\tag{5}
$$

where $\rho _ { M }$ and $\rho _ { U }$ are likelihood-ratio thresholds. Since $m _ { M }$ and m<sub>U</sub> are differences of length-normalized log likelihoods, exp(m<sub>M</sub>) and exp(m<sub>U</sub>) can be interpreted as length-

normalized candidate-level likelihood ratios. IGD routes directly to the memory preview only if

$$
\begin{array} { r l } & { \mathrm { v a l i d } _ { \mathrm { m e m } } = 1 , } \\ & { ~ \hat { y } _ { \mathrm { c t x } } \not \equiv \hat { y } _ { \mathrm { m e m } } , } \\ & { ~ A _ { \mathrm { m e m } } \geq 0 . 6 7 , } \\ & { ~ D _ { \mathrm { m e m } } \geq 0 . } \end{array}\tag{6}
$$

When these conditions hold, IGD sets $\hat { y } _ { \mathrm { f i n a l } } = \hat { y } _ { \mathrm { m e m } } ;$ otherwise, decoding falls back to token-level correction. We use $\rho _ { M } = 3 . 0$ and $\rho _ { U } = 1 . 0$ by default, requiring the memory branch to prefer the memory candidate by at least a factor of three while the user branch must not prefer the context candidate.

## C. Token-Level Correction

IGD performs token-level source arbitration around the original user distribution, the goal is not to replace the distribution, but to gently adjust it when the context and memory branches disagree. This follows the general intuition of decoding time control, where generation is steered by modifying next token scores rather than updating model parameters [23]–[25].

$$
p _ { \mathrm { f i n a l } , t } ( v ) \propto p _ { \mathrm { u s e r } , t } ( v ) \left( \frac { p _ { \mathrm { c t x } , t } ( v ) } { p _ { \mathrm { m e m } , t } ( v ) } \right) ^ { \lambda _ { t } }\tag{7}
$$

The scalar $\lambda _ { t }$ controls both the direction and magnitude of the intervention, positive values favor the context branch, negative values favor the memory branch, and values near zero leave decoding close to the original user distribution. IGD computes $\lambda _ { t }$ in three steps: activation, direction, and reliability scaling.

1) Activation: IGD should intervene only when context and memory provide meaningfully different next token distribution. We therefore measure distributional conflict at step t between the context and memory branches using Jensen-Shannon divergence [26]:

$$
\delta _ { t } = \mathrm { J S D } \big ( p _ { \mathrm { c t x } , t } \ \| p _ { \mathrm { m e m } , t } \big )\tag{8}
$$

For efficiency, $\mathrm { J S D } ( \cdot \| \cdot )$ is computed over a renormalized top K token set rather than the full vocabulary we use $K = 1 6$ The conflict score is converted into a soft activation gate:

$$
a _ { t } = \left[ \mathrm { c l i p } \left( \frac { \delta _ { t } - \tau _ { \mathrm { l o w } } } { \tau _ { \mathrm { h i g h } } - \tau _ { \mathrm { l o w } } } , 0 , 1 \right) \right] ^ { \gamma }\tag{9}
$$

with $\tau _ { \mathrm { l o w } } = 0 . 1 0 , \ \tau _ { \mathrm { h i g h } } = 0 . 3 5 ,$ , and $\gamma \ : = \ : 2 . 0$ . This gate suppresses correction when the two branches already agree and gradually activates intervention as source conflict increases.

2) Confidence Direction: Conditioned on activation, IGD determines which source should be favored. We combine two signals: an instruction mode and branch level confidence, the instruction encodes the user’s intended trust policy or hard context-follow policy, while the branch confidence estimates how concentrated each branch’s next token distribution is. We use entropy-based confidence, which is a standard proxy for predictive uncertainty and calibration behavior in neural models, for each branch $b \in \{ \mathrm { c t x } , \mathrm { m e m } \}$ , we define:

$$
r _ { b , t } = 1 - \frac { H ( \mathrm { t o p K } ( p _ { b , t } ) ) } { \log K }\tag{10}
$$

where $H ( \cdot )$ is the entropy of the renormalized top-K distribution, and lower entropy corresponds to higher confidence. Let $d _ { \mathrm { m o d e } } \in ( 0 , 1 )$ denote the prior probability of trusting context under the current instruction mode, we use $d _ { \mathrm { s t r i c t } } = 0 . 9$ , and $d _ { \mathrm { t r u t h } } = 0 . 3 .$ . The resulting context preference is:

$$
\pi _ { \mathrm { c t x } , t } = \sigma \left( \mathrm { l o g i t } ( d _ { \mathrm { m o d e } } ) + \log ( r _ { \mathrm { c t x } , t } ) - \log ( r _ { \mathrm { m e m } , t } ) \right)\tag{11}
$$

and the corresponding signed base coefficient is:

$$
\lambda _ { \mathrm { b a s e , } t } = \lambda _ { \mathrm { m a x } } a _ { t } \left( 2 \pi _ { \mathrm { c t x , } t } - 1 \right)\tag{12}
$$

This parameterization induces a intent consistent intervention direction: strict context following prompts start with a strong context prior, truth seeking prompts start with a memory prior, and the final direction can still change when one branch is substantially more confident than the other.

3) Reliability Scaling: The base coefficient determines the direction of intervention, IGD then rescales its magnitude by the reliability of the source being favored. This prevents confident but poorly supported context from exerting excessive influence, and similarly prevents unstable memory predictions from overriding context.

a) Context reliability: When $\lambda _ { \mathrm { b a s e } , t } ~ > ~ 0$ , IGD favors the context branch and estimates whether the selected snippet genuinely supports the context preview. Let $\hat { y } _ { \mathrm { c t x } }$ be the context preview and $x _ { \mathrm { c t x } }$ be the selected support snippet, we define:

$$
s _ { \mathrm { c t x } } = 0 . 7 \cdot \mathrm { S u p p } ( \hat { y } _ { \mathrm { c t x } } , x _ { \mathrm { c t x } } ) + 0 . 3 \cdot \mathrm { Q u a l } ( x _ { \mathrm { c t x } } )\tag{13}
$$

where $\operatorname { S u p p } ( \cdot , \cdot )$ measures local textual support for the preview answer and Qual(·) captures the evidential quality of selected snippet. Let vali $\mathrm { { { d } _ { c t x } \in \{ 0 , 1 \} } }$ indicate whether the context preview is a valid short-form answer, the context reliability is:

$$
q _ { \mathrm { c t x } } = q _ { \mathrm { c t x } } ^ { \mathrm { m i n } } + ( 1 - q _ { \mathrm { c t x } } ^ { \mathrm { m i n } } ) \mathrm { v a l i d } _ { \mathrm { c t x } } s _ { \mathrm { c t x } }\tag{14}
$$

b) Memory reliability: When $\lambda _ { \mathrm { b a s e } , t } < 0 ,$ IGD favors the memory branch and scales the intervention by the stability of the closed-book memory previews.

$$
q _ { \mathrm { m e m } } = q _ { \mathrm { m e m } } ^ { \mathrm { m i n } } + ( 1 - q _ { \mathrm { m e m } } ^ { \mathrm { m i n } } ) \mathrm { v a l i d } _ { \mathrm { m e m } } A _ { \mathrm { m e m } }\tag{15}
$$

We use $q _ { \mathrm { c t x } } ^ { \mathrm { m i n } } = 0 . 3 5$ and $q _ { \mathrm { m e m } } ^ { \mathrm { m i n } } = 0 . 5 5$ . Here, $A _ { \mathrm { { m e m } } }$ is computed as the average pairwise answer consistency among the M closed-book memory previews:

$$
\begin{array} { c } { { A _ { \mathrm { m e m } } = \displaystyle \frac { 2 } { M ( M - 1 ) } \sum _ { 1 \leq i < j \leq M \atop \mathrm { A n s C o n s } \left( \hat { y } _ { \mathrm { m e m } } ^ { ( i ) } , \hat { y } _ { \mathrm { m e m } } ^ { ( j ) } \right) } } } \end{array}\tag{16}
$$

where AnsCons $( \cdot , \cdot ) \in [ 0 , 1 ]$ measures whether two short-form memory previews are equivalent answers.

c) Final coefficient: The final token-level coefficient is:

$$
\lambda _ { t } = \left\{ \begin{array} { l l } { \lambda _ { \mathrm { b a s e } , t } q _ { \mathrm { c t x } } , } & { \lambda _ { \mathrm { b a s e } , t } > 0 , } \\ { \lambda _ { \mathrm { b a s e } , t } q _ { \mathrm { m e m } } , } & { \lambda _ { \mathrm { b a s e } , t } < 0 , } \\ { 0 , } & { \lambda _ { \mathrm { b a s e } , t } = 0 . } \end{array} \right.\tag{17}
$$

Thus, IGD intervenes only under source conflict, chooses the direction according to user intent and branch confidence, and scales the update by the reliability of the favored source. <sup>1</sup>

## IV. EXPERIMENT SETUP

## A. Evaluation Design

We evaluate IGD in retrieval-augmented short form answering questions. Each example consists of a question, a documenttagged context, and a prompt mode. We consider two prompt modes that represent different user intents. In STRICT mode, the model is instructed to answer strictly according to the provided context. In TRUTH mode, the model is instructed to treat the context as potentially noisy or misleading and to prioritize the factually correct answer when context and world knowledge conflict. The two modes use the same question and context, only the user instruction changes. Direct RAG and IGD are evaluated under identical prompts, to isolate the effect of source arbitration during the decoding phase.

We organize the evaluation into two groups. The faithful group contains standard QA settings where the context supports the gold answer, to test whether IGD preserves the benefits of retrieval. The factual-conflict group contains examples where the context answer conflicts with the world answer, to test whether IGD follows the context in STRICT mode while recovering the world-knowledge answer in TRUTH mode.

## B. Benchmarks

We evaluate on six QA benchmarks, grouped into three faithful QA benchmarks and three factual-conflict QA benchmarks, each contains 500 samples. The faithful group includes KILT-NQ, TriviaQA, and SQuAD. KILT-NQ is based on Natural Questions and the KILT Wikipedia corpus [27], [28]; TriviaQA contains trivia style open-domain questions with answer aliases and Wikipedia/web evidence [29]; and SQuAD is a paragraph level reading comprehension dataset built from Wikipedia [30]. For KILT-NQ and TriviaQA, we construct multi-document contexts by selecting an answer bearing passage and adding distractors; for SQuAD, we use the original paragraph as support and add distractor paragraphs from other examples. The factual-conflict group includes ConflictBank, NQ-Swap, and CounterFact. ConflictBank contains Wikidata-derived knowledge conflicts with generated misleading evidence [31]; NQ-Swap is constructed from Natural Questions/MRQA by replacing the original answer in the context with a substituted answer [27], [32]; and CounterFact is adapted from the factual association editing benchmark in ROME [33]. In these factualconflict benchmarks, the original answer is treated as the world answer, while the misleading or substituted answer supported by the context is treated as the context answer. We apply alias cleaning and semantic filtering to remove ambiguous cases where the world and context answers are equivalent.

## C. Models and Baselines

We evaluate five instruction-tuned LLMs: Qwen3-32B [34], Qwen2.5-14B-Instruct [35], Llama-3-8B-Instruct [36], Mistral-7B-Instruct-v0.3 [37], and Phi-4 14B [38]. These models cover different families and scales, allowing us to test whether IGD generalizes across models with different parametric knowledge and instruction-following behavior. Experiments were conducted on local GPU servers equipped with NVIDIA RTX A6000 GPU and two NVIDIA GeForce RTX 5090 GPUs.

We compare IGD with six baselines. Closed-book Q-only receives only the question and no context, it is a diagnostic reference for parametric knowledge rather than a context using RAG method. Direct RAG receives the same question, context, and prompt-mode instruction as IGD, but generates directly from the original prompt without source arbitration. We also include confidence reasoning baselines from situated faithfulness work [8]: ExplicitSCR, which asks the model to explicitly reason about whether to trust the context or its internal knowledge, and three rule-based confidence reasoning variants, RCR-InternalEval, RCR-ContextEval, and RCR-InternalConf, which extract confidence or evaluation signals for internal and context-based answers and then select a final answer according to predefined rules. In addition, we compare with MADAM-RAG, a multi-agent RAG framework designed to aggregate and resolve conflicting retrieved evidence [9].

## D. Evaluation Metrics

We evaluate the answer accuracy using normalized alias matching, following standard QA evaluation practice [29], [30]. For faithful benchmarks, predictions are always matched against gold answers. For factual-conflict benchmarks, the target depends on the prompt mode: STRICT mode reports context accuracy, while TRUTH mode reports world accuracy.

Our main aggregate metric is the Intent-Aligned Score (IA), defined as the macro-average over the faithful and factualconflict benchmark groups:

$$
\mathrm { I A } ^ { m } = \frac { \sum _ { d \in \mathcal { D } _ { \mathrm { f a i t h } } } \mathrm { A c c } _ { d } ^ { \mathrm { g o l d } } + \sum _ { d \in \mathcal { D } _ { \mathrm { c o n f } } } \mathrm { A c c } _ { d } ^ { m } } { \left| \mathcal { D } _ { \mathrm { f a i t h } } \right| + \left| \mathcal { D } _ { \mathrm { c o n f } } \right| } .\tag{18}
$$

Here, $m \in \{ \mathrm { s T R I C T } , \mathrm { T R U T H } \}$ . For $d \in \mathcal { D } _ { \mathrm { c o n f } } , \mathrm { A c c } _ { d } ^ { m }$ denotes context accuracy in STRICT mode and world accuracy in TRUTH mode, IA therefore measures whether the model behavior matches the user’s intended trust policy rather than rewarding a single fixed preference for either context or memory.

## V. RESULTS

## A. Closed-Book Performance Reveals the Recoverable Parametric Signal

We first evaluate each model in the closed-book Q-only setting, where the model receives only the question without any retrieved context. It is important to interpret IGD because factual recovery from a misleading context is only possible when the underlying model can access the correct answer from its parametric memory, if the closed-book model cannot recover the world answer, the upper bound of any memory oriented correction is naturally limited [39], [40].

TABLE I: Truth-mode comparison across three faithful benchmarks and three factual-conflict benchmarks. IA is the macroaverage over the six benchmark columns. Bold marks the best context-using method in each column, and underline marks the second-best. IGD also reports deltas against Direct RAG under the same model and benchmark.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Method</td><td rowspan="2">IA</td><td colspan="3">Faithful benchmarks</td><td colspan="3">Factual-conflict benchmarks</td></tr><tr><td>KILT-NQ</td><td>TriviaQA</td><td>SQuAD</td><td>ConflictBank</td><td>NQ-Swap</td><td>CounterFact</td></tr><tr><td>Qwen3-32B</td><td>Closed-book Q-only†</td><td>56.8</td><td>35.4</td><td>62.0</td><td>32.4</td><td>56.4</td><td>70.0</td><td>84.6</td></tr><tr><td rowspan="8"></td><td>Direct RAG</td><td>51.0</td><td>74.8</td><td>88.8</td><td>90.4</td><td>15.0</td><td>27.0</td><td>10.0</td></tr><tr><td>ExplicitSCR</td><td>12.9</td><td>16.0</td><td>18.4</td><td>15.8</td><td>4.0</td><td>3.0</td><td>20.2</td></tr><tr><td>RCR-InternalEval</td><td>62.1</td><td>58.2</td><td>74.6</td><td>47.8</td><td>47.8</td><td>70.5</td><td>73.6</td></tr><tr><td>RCR-ContextEval</td><td>53.6</td><td>68.8</td><td>80.0</td><td>83.2</td><td>21.0</td><td>37.0</td><td>31.4</td></tr><tr><td>RCR-InternalConf</td><td>59.9</td><td>50.0</td><td>69.0</td><td>41.8</td><td>46.6</td><td>77.5</td><td>74.8</td></tr><tr><td>MADAM-RAG</td><td>65.0</td><td>62.6</td><td>82.4</td><td>66.2</td><td>44.0</td><td>61.0</td><td>73.8</td></tr><tr><td>IGD</td><td>74.0 ↑23.0</td><td>72.2 ↓2.6</td><td>84.8 ↓4.0</td><td>89.8 ↓0.6</td><td>50.2 ↑35.2</td><td>71.5 ↑44.5</td><td>75.4 ↑65.4</td></tr><tr><td>Qwen2.5-14B Closed-book Q-only†</td><td>63.3</td><td>37.2</td><td>65.6</td><td>31.2</td><td>61.2</td><td>91.5</td><td>92.8</td></tr><tr><td rowspan="10"></td><td>Direct RAG</td><td>58.1</td><td>74.2</td><td>90.2</td><td>94.2</td><td>12.0</td><td>42.0</td><td>36.0</td></tr><tr><td>ExplicitSCR</td><td>10.8</td><td>13.0</td><td>16.4</td><td>14.2</td><td>4.8</td><td>9.5</td><td></td></tr><tr><td>RCR-InternalEval</td><td>63.5</td><td>64.8</td><td>80.6</td><td>65.0</td><td>39.2</td><td>57.0</td><td>6.6</td></tr><tr><td>RCR-ContextEval</td><td>59.6</td><td>57.8</td><td>76.4</td><td>69.8</td><td>26.4</td><td>57.5</td><td>74.6</td></tr><tr><td>RCR-InternalConf</td><td>65.0</td><td>54.8</td><td>75.6</td><td>49.2</td><td>55.0</td><td>67.0</td><td>69.8</td></tr><tr><td>MADAM-RAG</td><td>59.7</td><td>57.4</td><td>71.8</td><td>61.0</td><td>38.2</td><td>63.0</td><td>88.2 66.6</td></tr><tr><td>IGD</td><td>77.6 ↑19.5</td><td>70.8 ↓3.4</td><td>85.6↓4.6</td><td>87.8 ↓6.4</td><td>55.4 ↑43.4</td><td>77.0 ↑35.0</td><td>89.4 ↑53.4</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Closed-book Q-only†</td><td>55.7 59.6</td><td>37.8 73.2</td><td>69.8</td><td>26.8</td><td>55.6</td><td>68.5</td><td>75.8</td></tr><tr><td>Direct RAG ExplicitSCR</td><td>11.2</td><td>9.8</td><td>88.4 8.8</td><td>89.4</td><td>27.6</td><td>49.0</td><td>30.2</td></tr><tr><td></td><td>59.7</td><td>59.8</td><td>81.0</td><td>8.8 58.8</td><td>2.4 41.0</td><td>16.0</td><td>21.2</td></tr><tr><td>RCR-InternalEval</td><td>53.3</td><td>42.6</td><td>68.6</td><td>33.2</td><td>42.2</td><td>55.0</td><td>62.8</td></tr><tr><td>RCR-ContextEval RCR-InternalConf</td><td>52.1</td><td>45.0</td><td>75.2</td><td>40.6</td><td>32.2</td><td>61.5</td><td>71.6</td></tr><tr><td>MADAM-RAG</td><td>45.9</td><td>51.6</td><td>64.6</td><td>60.4</td><td>28.0</td><td>56.0</td><td>63.8</td></tr><tr><td>IGD</td><td>69.3 ↑9.6</td><td>68.4 ↓4.8</td><td>85.8 ↓2.6</td><td>85.0 ↓4.4</td><td>46.8 ↑19.2</td><td>40.0 62.0 ↑13.0</td><td>30.6</td></tr><tr><td>Mistral-7B</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>67.6 ↑37.4</td></tr><tr><td rowspan="9"></td><td>Closed-book Q-only†</td><td>51.8 42.7</td><td>39.2</td><td>66.2</td><td>24.2</td><td>39.4</td><td>68.5</td><td>73.0</td></tr><tr><td>Direct RAG ExplicitSCR</td><td>26.6</td><td>61.4</td><td>77.4</td><td>77.0</td><td>11.4</td><td>17.5</td><td>11.4</td></tr><tr><td>RCR-InternalEval</td><td>45.7</td><td>23.2</td><td>32.0</td><td>22.6</td><td>17.6</td><td>21.0</td><td>43.2</td></tr><tr><td>RCR-ContextEval</td><td></td><td>50.0</td><td>72.8</td><td>41.4</td><td>26.2</td><td>44.0</td><td>40.0</td></tr><tr><td>RCR-InternalConf</td><td>47.0 50.1</td><td>57.0</td><td>76.4</td><td>72.8</td><td>10.6</td><td>22.5</td><td>42.4</td></tr><tr><td></td><td></td><td>49.6</td><td>74.2</td><td>38.2</td><td>36.6</td><td>40.5</td><td>61.4</td></tr><tr><td>MADAM-RAG IGD</td><td>44.4</td><td>50.6</td><td>59.6</td><td>58.6</td><td>19.2</td><td>52.5</td><td>25.6</td></tr><tr><td></td><td>54.2 ↑11.5</td><td>58.4 ↓3.0</td><td>76.0 ↓1.4</td><td>73.0↓4.0</td><td>37.0 ↑25.6</td><td>43.0 ↑25.5</td><td>47.8 ↑36.4</td></tr><tr><td>Closed-book Q-only†</td><td>51.8</td><td>32.6</td><td>66.6</td><td>28.2</td><td>41.8</td><td>60.0</td><td>81.4</td></tr><tr><td rowspan="10">Phi-4 14B</td><td>Direct RAG</td><td>56.9</td><td>71.0</td><td>87.2</td><td>85.8</td><td>15.8</td><td>35.5</td><td>46.2</td></tr><tr><td>ExplicitSCR</td><td>6.1</td><td>11.8</td><td>14.6</td><td>3.0</td><td>2.6</td><td>2.0</td><td>2.8</td></tr><tr><td>RCR-InternalEval</td><td>55.8</td><td>58.8</td><td>77.6</td><td>49.2</td><td>30.8</td><td>47.5</td><td></td></tr><tr><td>RCR-ContextEval</td><td>57.5</td><td>65.8</td><td></td><td></td><td>21.6</td><td>45.5</td><td>70.8</td></tr><tr><td>RCR-InternalConf</td><td>55.7</td><td></td><td>81.8</td><td>66.8</td><td></td><td></td><td>63.4</td></tr><tr><td></td><td></td><td>54.0</td><td>76.2</td><td>45.8</td><td>32.8</td><td>53.0</td><td>72.4</td></tr><tr><td>MADAM-RAG</td><td>61.1</td><td>61.6</td><td>79.8</td><td>59.0</td><td>32.2</td><td>57.0</td><td>77.2</td></tr><tr><td>IGD</td><td>67.1 ↑10.2</td><td>63.8↓7.2</td><td>85.8↓1.4</td><td>82.2 ↓3.6</td><td>34.8 ↑19.0</td><td>59.0 ↑23.5</td><td>77.0↑30.8</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

The closed-book results in Table I show that the difficulty of the six benchmarks differs substantially. Among the faithful benchmarks, closed-book accuracy is generally low on KILT-NQ and SQuAD, indicating that these datasets rely heavily on the provided context [27], [28], [30]. Across the five models, KILT-NQ closed-book accuracy remains around the mid 30% range, and SQuAD is even lower, mostly around 25-32%. This is expected: KILT-NQ is derived from open-domain queries grounded in Wikipedia evidence, while SQuAD questions are written against specific paragraphs and often require access to the original passage. In contrast, TriviaQA is substantially easier in the closed-book setting, with all models reaching above 60%, suggesting that many trivia-style answers are already encoded in the models’ parameters.

For the factual-conflict benchmarks, closed-book performance is much higher overall, especially on CounterFact and NQ-Swap. CounterFact is the easiest factual-conflict benchmark under closed-book evaluation: all models achieve between 73.0% and 92.8%, suggesting that many subject relation facts remain accessible from parametric memory. NQ-Swap is also highly recoverable for several models, with Qwen2.5-14B reaching 91.5% and Phi-4 reaching 60.0%. ConflictBank is more challenging, with closed-book accuracy ranging from 39.4% to 61.2%. Overall, Qwen2.5-14B-Instruct shows the strongest closed-book IA score, while Qwen3-32B also exhibits strong recoverability on the factual-conflict benchmarks. These diagnostics suggest that IGD is evaluated in a meaningful regime: the factual-conflict examples are difficult for Direct

![](images/45e5d655ec48e0d54308eb5b151c4b496723fbe5d8a09fcd02977983d48ccdbd.jpg)  
Fig. 2: Parametric recovery rate (PRR) of IGD on factual-conflict benchmarks, the dashed line denotes full recovery relative to the closed-book reference.

RAG because the context is misleading, but many of their correct answers are still recoverable from model memory.

## B. Truth Prompting Alone Does Not Calibrate Context Trust

Table I shows that explicit truth seeking instructions alone are insufficient to prevent RAG models from being misled by incorrect context. In truth mode, the prompt explicitly instructs the model to prioritize the correct answer and to treat the provided context with skepticism when it may be misleading [6], [8]. Nevertheless, Direct RAG performs poorly on factual-conflict benchmarks across all models, this failure is most visible when comparing Direct RAG with the closedbook diagnostic. For example, Qwen2.5-14B answers NQ-Swap correctly 91.5% of the time in the closed-book setting, but drops to 42.0% once the misleading context is provided. The same model reaches 92.8% closed-book accuracy on CounterFact, but Direct RAG falls to 36.0%.

The additional baselines further clarify that this problem is not fully solved by answer-level reasoning or confidencebased source selection. ExplicitSCR and RCR variants are designed to improve situated faithfulness by reasoning over, or extracting confidence from, internal and context-based answers [8], while MADAM-RAG uses multi-agent aggregation to handle conflicting retrieved evidence [9]. These methods often improve over Direct RAG on factual-conflict benchmarks, confirming that explicit source comparison is useful. However, their improvements are uneven: some variants recover more world answers but substantially reduce faithful QA accuracy, while others preserve context grounded performance but remain weak under misleading context. In contrast, IGD achieves the best IA score for every evaluated backbone in Table I. This suggests that the key difficulty is not only detecting whether context or memory is correct at the answer level, but also controlling how strongly each source should influence decoding. the desired behavior is not to simply suppress context, but to preserve context grounded accuracy on faithful examples while resisting misleading evidence in factual conflict cases.

## C. IGD Balances Factuality and Faithfulness

a) Truth mode: recovering world answers under mislead ing context.: Table I shows that IGD substantially improves truth mode factual recovery while largely preserving faithful QA accuracy. Compared with Direct RAG, IGD improves the IA score for every model, with gains of 23.0 points for Qwen3-32B, 19.5 for Qwen2.5-14B, 9.6 for Llama-3-8B, 11.5 for Mistral-7B, and 10.2 for Phi-4. These gains are driven primarily by large improvements on factual-conflict benchmarks, where IGD consistently shifts generation away from misleading context and toward recoverable world knowledge.

The comparison with external baselines provides a more informative view of this trade-off. Confidence-reasoning and multi-agent baselines can be strong on individual factualconflict columns, but they often sacrifice accuracy on faithful benchmarks. For instance, RCR-style methods may recover more world-knowledge answers on a particular conflict set, but their faithful accuracy drops sharply because they rely on a global answer-level source decision. Similarly, MADAM-RAG improves some conflict cases, but its aggregation strategy does not consistently preserve original context-grounded behavior.

The factual gains are large and systematic. Across all models, IGD improves every factual-conflict benchmark over Direct RAG, with the largest gain reaching 65.4 percentage points on Qwen3-32B CounterFact. Importantly, these gains do not come from discarding retrieval altogether. On faithful benchmarks, IGD remains close to Direct RAG, with moderate drops that are much smaller than the factual-conflict gains. This pattern suggests that token-level correction is more conservative than answer-level rerouting alone: it can weaken misleading context when necessary, while still preserving useful retrieved evidence when the context is correct.

To better quantify how much of the recoverable parametric signal is restored, Figure 2 reports the parametric recovery rate (PRR), defined as the fraction of the gap between Direct RAG and closed-book Q-only that IGD recovers on factual-conflict benchmarks:

$$
\mathrm { P R R } = \frac { \mathrm { A c c } _ { \mathrm { I G D } } - \mathrm { A c c } _ { \mathrm { D i r e c t } } } { \mathrm { A c c } _ { \mathrm { C l o s e d } } - \mathrm { A c c } _ { \mathrm { D i r e c t } } } .\tag{19}
$$

The PRR results show that IGD recovers a large portion of the available parametric knowledge for stronger models. For example, Qwen3-32B recovers most of performance gap on NQ-Swap and CounterFact, whereas Mistral-7B exhibits

TABLE II: Strict context following prompt. Factual-conflict benchmarks are evaluated by context accuracy, testing whether IGD preserves user intent when the user explicitly asks to follow the provided context.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Method</td><td rowspan="2">一 IA</td><td colspan="4">Faithful benchmarks</td><td colspan="4">Factual-conflict benchmarks</td></tr><tr><td>KILT-NQ</td><td>TriviaQA</td><td></td><td>SQuAD</td><td>ConflictBank</td><td>NQ-Swap</td><td></td><td>CounterFact</td></tr><tr><td rowspan="2">Qwen3-32B</td><td>Direct RAG</td><td>83.0|</td><td>72.6</td><td>89.4</td><td></td><td>91.0</td><td>83.2</td><td>63.0</td><td></td><td>99.0</td></tr><tr><td>IGD</td><td>84.3</td><td> $6 9 . 4 \downarrow 3 . 2$ </td><td></td><td> $8 6 . 8 \ \downarrow 2 . 6$ </td><td> $9 0 . 0 ~ \downarrow ~ 1 . 0$ </td><td> ${ \bf 8 7 . 2 \mathrm { ~ \uparrow ~ 4 . 0 ~ } }$ </td><td> $7 2 . 5 \ \uparrow 9 . 5$ </td><td></td><td> $\mathbf { 9 9 . 6 \ : \uparrow 0 . 6 }$ </td></tr><tr><td rowspan="2"></td><td>Qwen2.5-14B Direct RAG</td><td>84.2</td><td>74.6</td><td>89.2</td><td></td><td>94.0</td><td>84.4</td><td>63.0</td><td></td><td>100.0</td></tr><tr><td>IGD</td><td>88.1</td><td> $7 6 . 4 \uparrow 1 . 8$ </td><td></td><td> $8 8 . 6 ~ \downarrow ~ 0 . 6 ~$ </td><td> $9 3 . 6 ~ \downarrow ~ 0 . 4$ </td><td> ${ \bf 8 9 . 2 \mathrm { ~ \uparrow ~ 4 . 8 ~ } }$ </td><td> ${ \bf 8 1 . 0 ~ } \uparrow ~ 1 8 . 0 $ </td><td></td><td> $\mathbf { 1 0 0 . 0 } \  \ 0 . 0$ </td></tr><tr><td rowspan="2">Llama-3-8B</td><td>Direct RAG</td><td>80.8|</td><td>73.0</td><td>88.0</td><td></td><td>88.4</td><td>85.4</td><td>50.0</td><td></td><td>99.8</td></tr><tr><td>IGD</td><td>82.0</td><td> $7 1 . 0 ~ \downarrow ~ 2 . 0$ </td><td></td><td> $8 6 . 4 ~ \downarrow ~ 1 . 6$ </td><td> $8 7 . 6 ~ \downarrow ~ 0 . 8$ </td><td> $\mathbf { 8 8 . 0 \ : \Uparrow 2 . 6 }$ </td><td> ${ \bar { \bf 5 9 . 0 } } \ \uparrow \ 9 . 0$ </td><td></td><td> $\mathbf { 1 0 0 . 0 \ : \uparrow 0 . 2 }$ </td></tr><tr><td rowspan="2">Mistral-7B</td><td>Direct RAG</td><td>77.1 </td><td>64.0</td><td>79.6</td><td></td><td>80.8</td><td>82.6</td><td>59.5</td><td></td><td></td></tr><tr><td>IGD</td><td>77.6</td><td> $6 0 . 6 ~ \downarrow ~ 3 . 4$ </td><td></td><td> $7 9 . 2 \ \downarrow \ 0 . 4$ </td><td> ${ \bf 8 2 . 0 ~ } \uparrow { \bf \up1 . 2 }$ </td><td> ${ \mathbf { 8 4 . 4 } } \ \uparrow \ 1 . 8$ </td><td> ${ \bf 6 2 . 5 } \ \uparrow \ 3 . 0 $ </td><td></td><td>96.0  ${ \bf 9 7 . 0 } \ \uparrow \ 1 . 0$ </td></tr><tr><td rowspan="2">Phi-4 14B</td><td>Direct RAG</td><td>81.3|</td><td>73.2</td><td>84.0</td><td></td><td>86.6</td><td>82.4</td><td>63.0</td><td></td><td></td></tr><tr><td>IGD</td><td>82.5</td><td> $7 1 . 6 ~ \downarrow ~ 1 . 6$ </td><td></td><td> $8 3 . 2 ~ \downarrow ~ 0 . 8$ </td><td> ${ \bf 8 7 . 0 ~ } \uparrow ~ 0 . 4$ </td><td> ${ \bf 8 5 . 6 } \ \uparrow \ 3 . 2$ </td><td> ${ \bf 6 8 . 5 } \ \uparrow \ 5 . 5 $ </td><td></td><td>98.4  $\mathbf { 9 9 . 2 \ \uparrow 0 . 8 }$ </td></tr></table>

## VI. FACTUALITY AND FAITHFULNESS TRADE-OFF

We further analyze how IGD balances factual recovery and context faithfulness through component ablations and hyperparameter sensitivity. This analysis is motivated by the central tension in RAG, stronger reliance on parametric memory can recover world answers but may undermine strict context following [7], [8]. We conduct ablations on Qwen2.5- 14B-Instruct over the same six-benchmark suite used in the main experiments, reporting macro averaged results over the three faithful benchmarks and the three factual-conflict

The results show that IGD preserves, and in many cases improves, strict context-following behavior. These strict prompt’s mode results are central to the claim that IGD balances factuality and faithfulness rather than optimizing only one side of the trade-off [12]. A method that simply discounts retrieved context would improve truth mode factual accuracy but damage strict context following [16], [17]. In contrast, IGD uses the prompt mode to determine the direction of intervention: under truth-seeking prompts it can pull decoding toward memory, while under strict context-following prompts it preserves or strengthens context guided decoding.

b) Strict mode: preserving context-following intent: Table II evaluates the complementary setting: the user explicitly asks the model to follow the provided context. In this mode, factual-conflict benchmarks are evaluated by context accuracy rather than world accuracy, because the desired behavior is to follow the supplied evidence even when it conflicts with world knowledge. Since the external baselines in Table I are designed primarily for truth-seeking or correctness-oriented source selection, we keep the strict-mode comparison focused on Direct RAG and IGD.

weaker recovery. This indicates that IGD’s factual correction is bounded by both the model’s accessible parametric knowledge and the reliability of the memory branch. The pattern in Figure 2 therefore reinforces the interpretation that IGD is not hallucinating new facts, but recovering parametric answers that are otherwise suppressed by misleading context.

TABLE III: Core ablations of IGD on Qwen2.5-14B. Values are macro-averaged percentages over the three faithful benchmarks and the three factual-conflict benchmarks. Shaded cells mark notable changes relative to IGD full: light red indicates drops of roughly 3–10 percentage points, darker red indicates drops larger than 10 points, and light green indicates improvements over IGD full.
<table><tr><td rowspan="2">Variant</td><td rowspan="2">Removed / changed component</td><td colspan="2">Strict mode</td><td colspan="2">Truth mode</td></tr><tr><td>Faithful Gold</td><td>Factual Ctx</td><td>Faithful Gold</td><td>Factual World</td></tr><tr><td>IGD full</td><td>none</td><td>86.2</td><td>90.1</td><td>81.4</td><td>73.7</td></tr><tr><td rowspan="3">w/o Answer Filter w/o Token Correction w/o Activation Gate</td><td>answer-level memory filter</td><td>85.8</td><td>86.0</td><td>81.3</td><td>63.0</td></tr><tr><td>token-level correction</td><td>86.9</td><td>83.1</td><td>82.7</td><td>52.8</td></tr><tr><td>JSD conflict gate</td><td>85.9</td><td>86.2</td><td>77.5</td><td>70.5</td></tr><tr><td>w/o Mode Prior</td><td>instruction-mode prior</td><td>86.6</td><td>83.4</td><td>82.7</td><td>63.1</td></tr><tr><td>w/o Reliability Gate</td><td> $q _ { \mathrm { c t x } } / q _ { \mathrm { m e m } } { \mathrm { ~ s c a l i n g } }$ </td><td>85.0</td><td>85.5</td><td>76.1</td><td>68.6</td></tr><tr><td>Single Memory View</td><td>memory ensemble</td><td>86.1</td><td>85.8</td><td>85.5</td><td>54.5</td></tr></table>

benchmarks. Table III ablates six core components of IGD: the answer level memory filter, token-level correction, activation gate, instruction-mode prior, reliability scaling, and memory ensemble, we choose these ablations because each component directly affects the trade-off rather than merely changing implementation details.

Table III shows that IGD’s gains arise from coordinated source arbitration rather than a single heuristic. Removing token-level correction causes the largest degradation, reducing truth-mode factual world accuracy from 73.7 to 52.8 and strict-mode factual context accuracy from 90.1 to 83.1. This indicates that token-level correction is not merely a memory side intervention, but the main mechanism for steering: it pulls decoding toward memory under misleading context while preserving context-following behavior when required. The answer-level memory filter provides a complementary high precision route for stable factual-conflict cases; removing it drops factual world accuracy to 63.0 while leaving faithful accuracy nearly unchanged, suggesting that the filter is conservative enough not to disrupt ordinary context-grounded QA. The remaining ablations explain how IGD avoids over-intervention, a single memory view sharply reduces factual recovery to 54.5, showing that memory side correction requires a stable closed-book signal; removing the activation or reliability gate degrades both faithful and factual columns, indicating that intervention must be triggered only under meaningful conflict and scaled by source reliability. Finally, removing the mode prior lowers both truth-mode factual recovery and strict-mode context accuracy, confirming that context and memory conflicts cannot be resolved with a fixed source preference, whereas the full IGD model provides a stable balance between factual recovery and context faithfulness.

![](images/3d883558adf89605316328191ea7099e26ba90001efe660d125b8551dabe1940.jpg)  
Fig. 3: Sensitivity of truth mode performance to token-level steering hyperparameters on Qwen2.5-14B-Instruct.

![](images/10110b3dd39d452a75193a1853335d0aac148eda41f3bdda183cacf1824c0201.jpg)  
Fig. 4: Answer distribution on factual-conflict benchmarks under truth mode.

Figure 3 further exposes the limitation of using fixed global steering hyperparameters. As $\lambda _ { \mathrm { m a x } }$ increases, factual world ac curacy on factual-conflict benchmarks generally improves, but faithful accuracy on faithful benchmarks decreases. Similarly, smaller $d _ { \mathrm { t r u t h } }$ values make the router more memory biased and improve factual recovery, but they also make the model more likely to deviate from correct retrieved evidence. This creates a clear tuning trade-off: aggressive correction helps resist misleading context, whereas conservative correction better preserves faithful context use, this is a core limitation of the current framework. Future work could reduce this limitation by learning example adaptive intervention strengths, calibrating the router with validation feedback, or estimating source reliability at a finer granularity.

Beyond aggregate accuracy, we further analyze what models output on factual-conflict benchmarks. Main results report whether an answer matches the target, but they do not distinguish whether errors come from copying the misleading context, refusing to answer, or producing an unrelated alternative. We therefore categorize each output into four types: correct answer, misleading-context answer, unknown/refusal, and other answer. As shown in Figure 4, Direct RAG is dominated by misleading context answers: 57.9% of its outputs follow the incorrect context, while only 25.8% recover the correct world answer, IGD reverses this pattern, increasing correct answers to 58.3% and reducing misleading-context answers to 22.5%. At the same time, IGD does not convert all misleading context answers into correct answers, some mass moves into the other answer category. We interpret this as a residual uncertainty effect under source conflict: once decoding is pushed away from the misleading context, the model may still fail to land on the gold world answer if the memory branch is unstable, if competing aliases are plausible, or if the evidence conflict is not cleanly resolved. This is consistent with recent work showing that RAG models remain vulnerable under conflicting or noisy evidence and that models often generate plausible incorrect answers rather than abstaining when uncertainty is not explicitly rewarded [10], [41], [42]. Thus, IGD substantially mitigates context over-trust, but factual-conflict resolution remains imperfect when memory evidence is ambiguous or insufficiently calibrated.

![](images/037efa243277120a1ab1c5638c378778b5c115618c903396ce355a5827db6b65.jpg)

![](images/81380d98b26c23a03236ea31ad1a4cf74ffe8ff3cfd73997d4b8dc6a275db258.jpg)  
Fig. 5: Snippet localization accuracy versus downstream IGD performance.

Figure 5 highlights a surprising mismatch between snippet localization quality and downstream IGD performance. The GPT-5 localizer substantially improves standalone snippet support accuracy, increasing faithful snippet accuracy from 83.0% to 99.0% and factual snippet accuracy from 79.0% to 87.0%. Intuitively, one might expect a better support localizer to improve IGD, since IGD uses snippet reliability to scale contextside intervention. However, the downstream results show the opposite: replacing the IDF localizer slightly reduces faithful accuracy, factual world accuracy, and IAS. This suggests that IGD does not only need a snippet that is semantically supportive under human or LLM judgment, it needs a snippet whose evidence is useful for the specific downstream generator and reliability function. The IDF localizer may select shorter, lexically concentrated spans that align better with answerbearing tokens and IGD’s support heuristics, while the GPT-5 localizer may prefer semantically richer snippets that are correct in isolation but less discriminative for logit-level source arbitration [43], [44]. The result also aligns with evidence that robust RAG depends on preserving fine-grained textual details, not only high-level semantic support [45]. More broadly, LLMbased scoring or localization can introduce its own judgment biases and objective mismatch [46].

## VII. CONCLUSION

We presented Intent-Guided Decoding (IGD), a decodingtime framework for resolving the factuality and faithfulness trade-off in retrieval-augmented generation. Rather than treating retrieved context as either universally authoritative or inherently suspect, IGD performs intent-conditioned source arbitration across the user prompt, retrieved context, and parametric memory. Experiments on faithful and factual-conflict benchmarks show that IGD improves intent-aligned generation across multiple LLMs, while ablations confirm that its gains arise from coordinated routing, conflict activation, reliability scaling, and memory stabilization.

## REFERENCES

[1] P. Lewis, E. Perez, A. Piktus, F. Petroni, V. Karpukhin, et al., “Retrieval augmented generation for knowledge-intensive nlp tasks,” in Advances in Neural Information Processing Systems, vol. 33, pp. 9459–9474, 2020.

[2] Y. Hu and Y. Lu, “Rag and rau: A survey on retrieval-augmented language model in natural language processing,” arXiv preprint arXiv:2404.19543, 2024.

[3] P. Jiang, S. Ouyang, Y. Jiao, M. Zhong, R. Tian, and J. Han, “Retrieval and structuring augmented generation with large language models,” in Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 2, pp. 6032–6042, 2025.

[4] G. Izacard and E. Grave, “Leveraging passage retrieval with generative models for open domain question answering,” in Proceedings of the 16th Conference ofthe European Chapter ofthe Associationfor Computational Linguistics, pp. 874–880, 2021.

[5] C. Niu, Y. Wu, J. Zhu, S. Xu, K. Shum, R. Zhong, J. Song, and T. Zhang, “Ragtruth: A hallucination corpus for developing trustworthy retrievalaugmented language models,” arXiv preprint arXiv:2401.00396, 2024.

[6] K. Wu, E. Wu, and J. Zou, “Clasheval: Quantifying the tug-of-war between an llm’s internal prior and external evidence,” arXiv preprint arXiv:2404.10198, 2024.

[7] Y. Ming, S. Purushwalkam, S. Pandit, Z. Ke, X.-P. Nguyen, et al., “Faitheval: Can your language model stay faithful to context, even if" the moon is made of marshmallows",” in International Conference on Learning Representations, vol. 2025, pp. 29430–29456, 2025.

[8] Y. Huang, S. Chen, H. Cai, and B. Dhingra, “Enhancing large language models’ situated faithfulness to external contexts,” arXiv preprint arXiv:2410.14675, 2024.

[9] H. Wang, A. Prasad, E. Stengel-Eskin, and M. Bansal, “Retrievalaugmented generation with conflicting evidence,” arXiv preprint arXiv:2504.13079, 2025.

[10] Z. Ge, Y. Wu, D. W. K. Chin, R. K.-W. Lee, and R. Cao, “Resolving conflicting evidence in automated fact-checking: A study on retrievalaugmented llms,” arXiv preprint arXiv:2505.17762, 2025.

[11] X.-P. Nguyen, S. Pandit, S. Purushwalkam, A. Xu, H. Chen, Y. Ming, Z. Ke, S. Savarese, C. Xiong, and S. Joty, “Sfr-rag: Towards contextually faithful llms,” arXiv preprint arXiv:2409.09916, 2024.

[12] E. Fadeeva, A. Rubashevskii, R. Vashurin, S. Dhuliawala, A. Shelmanov, T. Baldwin, P. Nakov, M. Sachan, and M. Panov, “Faithfulness-aware uncertainty quantification for fact-checking the output of retrieval augmented generation,” arXiv preprint arXiv:2505.21072, 2025.

[13] Z. Jiang, F. Xu, L. Gao, Z. Sun, Q. Liu, J. Dwivedi-Yu, Y. Yang, J. Callan, and G. Neubig, “Active retrieval augmented generation,” in Proceedings of the 2023 Conference on EMNLP, (Singapore), pp. 7969–7992, 2023.

[14] A. Asai, Z. Wu, Y. Wang, A. Sil, and H. Hajishirzi, “Self-rag: Learning to retrieve, generate, and critique through self-reflection,” in The Twelfth International Conference on Learning Representations, 2024.

[15] S.-Q. Yan, J.-C. Gu, Y. Zhu, and Z.-H. Ling, “Corrective retrieval augmented generation,” arXiv preprint arXiv:2401.15884, 2024.

[16] B. Bi, S. Huang, Y. Wang, T. Yang, Z. Zhang, et al., “Context-dpo: Aligning language models for context-faithfulness,” arXiv preprint arXiv:2412.15280, 2024.

[17] Q. Zhang, Z. Xiang, Y. Xiao, L. Wang, J. Li, X. Wang, and J. Su, “Faithfulrag: Fact-level conflict modeling for context-faithful retrievalaugmented generation,” arXiv preprint arXiv:2506.08938, 2025.

[18] X. Zhang, J. Zhang, F. Mo, D. K. Chandra, Y.-Z. Chen, F. Xie, and K. Liu, “Retrieval-augmented feature generation for domain-specific classification,” in 2025 IEEE International Conference on Data Mining (ICDM), pp. 943–952, IEEE, 2025.

[19] G. Izacard, P. Lewis, M. Lomeli, L. Hosseini, F. Petroni, et al., “Atlas: Few-shot learning with retrieval augmented language models,” arXiv preprint arXiv:2208.03299, 2022.

[20] O. Ram, Y. Levine, I. Dalmedigos, D. Muhlgay, A. Shashua, K. Leyton-Brown, and Y. Shoham, “In-context retrieval-augmented language models,” arXiv preprint arXiv:2302.00083, 2023.

[21] K. Sparck Jones, “A statistical interpretation of term specificity and its application in retrieval,” Journal of Documentation, vol. 28, no. 1, pp. 11–21, 1972.

[22] T. G. Dietterich, “Ensemble methods in machine learning,” in Multiple Classifier Systems, vol. 1857 of Lecture Notes in Computer Science, pp. 1–15, Springer, 2000.

[23] B. Krause, A. D. Gotmare, B. McCann, N. S. Keskar, S. Joty, R. Socher, and N. F. Rajani, “GeDi: Generative discriminator guided sequence generation,” in Findings of the Association for Computational Linguistics: EMNLP 2021, (Punta Cana, Dominican Republic), pp. 4929–4952, 2021.

[24] A. Liu, M. Sap, X. Lu, S. Swayamdipta, C. Bhagavatula, N. A. Smith, and Y. Choi, “Dexperts: Decoding-time controlled text generation with experts and anti-experts,” in Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing, 2021.

[25] X. L. Li, A. Holtzman, D. Fried, P. Liang, J. Eisner, T. Hashimoto, L. Zettlemoyer, and M. Lewis, “Contrastive decoding: Open-ended text generation as optimization,” arXiv preprint arXiv:2210.15097, 2022.

[26] J. Lin, “Divergence measures based on the shannon entropy,” IEEE Transactions on Information Theory, vol. 37, no. 1, pp. 145–151, 1991.

[27] T. Kwiatkowski, J. Palomaki, O. Redfield, M. Collins, A. Parikh, et al., “Natural questions: A benchmark for question answering research,” Transactions of the Association for Computational Linguistics, vol. 7, pp. 452–466, 2019.

[28] F. Petroni, A. Piktus, A. Fan, P. Lewis, M. Yazdani, et al., “Kilt: a benchmark for knowledge intensive language tasks,” in Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, (Online), pp. 2523–2544, 2021.

[29] M. Joshi, E. Choi, D. S. Weld, and L. Zettlemoyer, “Triviaqa: A large scale distantly supervised challenge dataset for reading comprehension,” in Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics, pp. 1601–1611, 2017.

[30] P. Rajpurkar, J. Zhang, K. Lopyrev, and P. Liang, “Squad: 100,000+ questions for machine comprehension of text,” in Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing, pp. 2383–2392, 2016.

[31] Z. Su, J. Zhang, X. Qu, T. Zhu, Y. Li, J. Sun, J. Li, M. Zhang, and Y. Cheng, “Conflictbank: A benchmark for evaluating the influence of knowledge conflicts in llm,” arXiv preprint arXiv:2408.12076, 2024.

[32] A. Fisch, A. Talmor, R. Jia, M. Seo, E. Choi, and D. Chen, “Mrqa 2019 shared task: Evaluating generalization in reading comprehension,” in Proceedings of the 2nd Workshop on Machine Reading for Question Answering, pp. 1–13, 2019.

[33] K. Meng, D. Bau, A. Andonian, and Y. Belinkov, “Locating and editing factual associations in gpt,” in Advances in Neural Information Processing Systems, vol. 35, 2022.

[34] A. Yang et al., “Qwen3 technical report,” arXiv preprint arXiv:2505.09388, 2025.

[35] A. Yang et al., “Qwen2.5 technical report,” arXiv preprint arXiv:2412.15115, 2024.

[36] A. Grattafiori et al., “The llama 3 herd of models,” arXiv preprint arXiv:2407.21783, 2024.

[37] A. Q. Jiang, A. Sablayrolles, A. Mensch, C. Bamford, D. S. Chaplot, et al., “Mistral 7b,” arXiv preprint arXiv:2310.06825, 2023.

[38] M. Abdin et al., “Phi-4 technical report,” arXiv preprint arXiv:2412.08905, 2024.

[39] A. Mallen, A. Asai, V. Zhong, R. Das, D. Khashabi, et al., “When not to trust language models: Investigating effectiveness of parametric and non-parametric memories,” arXiv preprint arXiv:2212.10511, 2023.

[40] N. Kandpal, H. Deng, A. Roberts, E. Wallace, and C. Raffel, “Large language models struggle to learn long-tail knowledge,” in Proceedings of the 40th International Conference on Machine Learning, PMLR, 2023.

[41] J. Chen, B. Bi, W. Zhang, J. Sui, X. Zhu, et al., “Rethinking all evidence: Enhancing trustworthy retrieval-augmented generation via conflict-driven summarization,” arXiv preprint arXiv:2507.01281, 2025.

[42] A. T. Kalai, O. Nachum, S. S. Vempala, and E. Zhang, “Why language models hallucinate,” arXiv preprint arXiv:2509.04664, 2025.

[43] J. Deng, Y. Shen, Z. Pei, Y. Chen, and L. Huang, “Influence guided context selection for effective retrieval-augmented generation,” arXiv preprint arXiv:2509.21359, 2025.

[44] T. E. Kim and F. Diaz, “Ltrr: Learning to rank retrievers for llms,” arXiv preprint arXiv:2506.13743, 2025.

[45] Y. Jin, K. Sharma, V. Rakesh, Y. Dou, M. Pan, M. Das, and S. Kumar, “Sara: Selective and adaptive retrieval-augmented generation with context compression,” arXiv preprint arXiv:2507.05633, 2025.

[46] Q. Li, S. Dou, K. Shao, C. Chen, and H. Hu, “Evaluating scoring bias in llm-as-a-judge,” arXiv preprint arXiv:2506.22316, 2025.