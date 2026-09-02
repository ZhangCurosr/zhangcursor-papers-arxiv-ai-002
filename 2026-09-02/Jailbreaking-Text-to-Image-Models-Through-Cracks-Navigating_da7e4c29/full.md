# Jailbreaking Text-to-Image Models Through Cracks: Navigating Heterogeneous Safety Filters via Multi-Agent Debate

Kaiyan Wen, Shijie Zhang, Lu Yu, Guangdong Bai

Abstract—Text-to-image (T2I) models remain vulnerable to jailbreak attacks that elicit Not-Safe-For-Work (NSFW) content, despite increasingly being guarded by heterogeneous, multilayer safety stacks combining text filters, image classifiers, and cross-modal detectors. Existing jailbreak studies either optimize against individual filters or query the complete pipeline with aggregate feedback, making it difficult to identify the active constraint and adapt to conflicts across safety layers. In this paper, we introduce the Detection Surface, a unified geometric framework that characterizes the decision boundaries induced by heterogeneous T2I safety filters and their joint effect on the jailbreak search space. This formulation reveals that successful evasion is governed by a sparse and non-convex region shaped by cross-layer conflicts, where mutations that bypass one filter may increase exposure to another. Motivated by this analysis, we propose CRACK, a multi-agent debate framework for adaptive jailbreak search that decomposes jailbreak search into exploration, diagnosis, and arbitration. CRACK coordinates an Attack Agent, a Defense Agent, and a Judge Agent to iteratively generate prompt mutations, obtain layer-specific diagnostic feedback, and optimize mutation strategies through reward-guided refinement. Through repeated rounds of debate, CRACK adapts its search direction to the evolving cross-layer constraints while preserving the original harmful intent. Extensive experiments across multiple T2I models, datasets, and safety configurations show that CRACK achieves Attack Success Rates (ASR) of up to 99.63% under composite defenses, while requiring fewer queries than existing methods and maintaining semantic fidelity.

Index Terms—Jailbreak, Text-to-Image models, heterogeneous safety filters, multi-agent debate.

## I. INTRODUCTION

Text-to-image (T2I) generative models such as Stable Diffusion [1], [2], DALL·E [3], and Midjourney [4] have demonstrated remarkable capabilities in synthesizing photorealistic images from natural language prompts [5], [6], [7]. However, the widespread deployment of T2I models in commercial platforms has raised critical safety concerns, since adversaries can craft malicious prompts to elicit Not-Safe-For-Work (NSFW) content, including violent, sexually explicit, or otherwise harmful imagery [8], [9]. To mitigate such misuse, model providers have integrated a variety of safety mechanisms, ranging from input-side keyword blacklists [8], [10] and LLMbased semantic classifiers [11], [12] to output-side NSFW image detectors [10], [13] and cross-modal alignment filters [14], [15], forming a multi-layered defense stack that guards the generation pipeline at different stages and modalities.

However, current T2I jailbreak methods are poorly matched to the compositional nature of modern safety pipelines. Component-specific attacks [9], [16] optimize against an individual filter, so their success does not necessarily transfer to a stack containing filters from different modalities. Methods that query the complete pipeline in a black-box manner [8], [17] treat the stack as a single decision function and observe only whether a query succeeds. Such feedback provides little indication of whether a rejection stems from explicit lexical cues, implicit unsafe semantics, generated visual content, or text-image alignment. Consequently, existing mutation policies struggle to determine whether a revision genuinely resolves the current detection risk or merely shifts it from one safety layer to another. For instance, a mutation that weakens lexical cues to bypass a text filter can simultaneously strengthen the unsafe visual semantics of the generated image, making it easier for a downstream image classifier to reject [16], [18]. Without explicitly accounting for these cross-layer interactions, existing methods lack a layer-aware signal for determining effective search directions in the prompt space.

To characterize these cross-layer interactions, we introduce the Detection Surface, a geometric abstraction of the decision boundaries induced by a heterogeneous defense stack. This abstraction provides a unified view of how these boundaries jointly partition the prompt space and constrain the feasible search region for multi-layer jailbreaks. Although heterogeneous filters broaden safety coverage, their distinct detection criteria partition the prompt space differently. A mutation may cross the boundary of one filter while remaining outside the safe region of another, producing cross-layer conflict [18]. The intersection of all per-layer safe regions, subject to preservation of the original harmful intent, further forms a sparse and non-convex evasion region. Therefore, multi-layer jailbreak is fundamentally a search problem over a sparse and non-convex evasion region shaped by cross-layer conflicts on the Detection Surface.

These properties make multi-agent debate particularly well suited to navigating the Detection Surface. First, the sparsity and non-convexity of the evasion region call for active exploration across diverse mutation directions rather than reliance on a fixed search trajectory. Second, cross-layer conflicts require diagnosis: because a mutation that bypasses one safety layer may increase exposure to another, effective search requires identifying which constraints currently impede progress and how they respond to each revision. Finally, conflicting layer-wise feedback requires arbitration, whereby heterogeneous signals are reconciled to determine whether a mutation yields overall progress toward the joint evasion region. Performing these functions within a single role can entangle mutation generation with failure attribution and crosslayer evaluation. Multi-agent debate instead provides a natural mechanism for separating these complementary functions while allowing their perspectives to be iteratively exchanged and reconciled.

To this end, we introduce CRACK, a multi-agent debate framework that instantiates exploration, diagnosis, and arbitration through three specialized agents, while leveraging reinforcement learning to progressively refine the mutation policy for layer-aware navigation of the Detection Surface. Starting from a harmful prompt, CRACK iteratively searches for mutations that better satisfy heterogeneous layer-wise constraints while preserving the original harmful intent, progressively steering the prompt toward the joint evasion region E<sup>∗</sup>. We refer to promising intermediate pathways through the heterogeneous safety boundaries as cracks, as they provide feasible directions toward joint evasion.

At each round, the Defense Agent performs diagnosis by providing structured diagnostic signals that approximate the heterogeneous criteria commonly used by real-world defense stacks, thereby identifying the constraints currently impeding progress and explaining the corresponding rejection. Conditioned on this diagnosis, the Attack Agent performs exploration by selecting and applying complementary mutation strategies to search for promising directions toward the next crack. The Judge Agent performs arbitration by reconciling potentially conflicting layer-wise signals into a composite assessment of overall progress. This assessment further serves as a reward signal for refining the Attack Agent’s mutation policy through reinforcement learning, discouraging revisions that merely shift detection risk from one layer to another. Through repeated rounds of debate and refinement, CRACK adapts its search direction as new cross-layer conflicts emerge, transforming stack-level trial-and-error into a layer-aware and diagnosis-guided search toward E<sup>∗</sup>.

We summarize our contributions as follows:

• Theoretical framework. We introduce the Detection Surface formalism, which provides a unified geometric characterization of multi-layer T2I defense stacks. We identify three structural properties (cross-layer conflict, sparsity, and non-convexity) that that decomposes jailbreak search into exploration, layer-aware diagnosis, and cross-layer arbitration, with reinforcement learning further refining the mutation policy. (Sec. III).

• Multi-agent attack framework. We propose CRACK, the jailbreak method that first leverages multi-agent debate with per-layer diagnostic feedback and RL-guided strategy selection. Each component of CRACK is directly motivated by a specific structural property of the Detection Surface, yielding a principled rather than heuristic attack design (Sec. IV).

Comprehensive evaluation. We conduct extensive experiments across multiple T2I models (Stable Diffusion v1.4, SDXL, DALLE·3, and Midjourney) under diverse safety configurations. CRACK achieves state-of-the-art attack success rates (up to 99.63% on SD 1.4 with composite filters) while requiring significantly fewer queries than existing methods, and maintains high semantic fidelity. These results validate the cross-layer interactions characterized by the Detection Surface and demonstrate the importance of layer-aware, adaptive search for navigating heterogeneous defense stacks (Sec. V).

## II. RELATED WORK

## A. Jailbreak Attacks on T2I Models

Existing T2I jailbreak attacks differ in their assumptions about model access. White-box attacks exploit model internals or accessible representations. Ring-A-Bell [9] identifies concepts that reveal vulnerabilities in concept-erased diffusion models, whereas MMA-Diffusion [19] constructs multimodal adversarial attacks against diffusion models through gradientbased optimization. These methods generally require access to model parameters, gradients, or internal representations and are therefore difficult to apply to closed commercial APIs.

Black-box attacks rely only on model queries and observable responses. SneakyPrompt [8] trains a reinforcement learning policy to replace prompt tokens and evade a CLIP-based safety checker. DACA [20] uses an LLM to decompose unsafe requests into less detectable components for attacking LLMguarded T2I systems. JailFuzzer [17] combines fuzz testing with LLM-based agents to automate jailbreak prompt generation. Prompting4Debugging [21] performs systematic red teaming by searching for problematic prompts that expose T2I model failures, while Perception-guided Jailbreak [16] uses perceptual signals to guide prompt construction.

A common limitation of existing black-box approaches is that they rely solely on coarse success-or-failure feedback to guide prompt optimization. Such coarse feedback provides limited visibility into layer-specific failure causes and crosslayer interactions, making it difficult to distinguish mutations that genuinely advance joint evasion from those that merely transfer detection risk across defense layers. Tab. I compares CRACK with representative baselines in terms of key capabilities for navigating multi-layer defenses. Among the compared methods, CRACK uniquely combines structured layer-wise diagnostic feedback with cross-layer adaptive search, directly addressing this capability gap in heterogeneous defense settings.

## B. Safety Mechanisms for T2I Models

Modern T2I systems employ safety mechanisms at multiple stages of the generation pipeline. Text-side filters include keyword-based screening [24] and LLM-based semantic guards that assess prompt intent before image generation [11], [12]. Concept-erasure methods modify diffusion models to suppress unsafe or unwanted concepts. ESD, for example, finetunes a diffusion model to erase targeted concepts from its generation capability [25], while subsequent studies examine the reliability and limitations of concept-removal methods under adversarial prompting [9].

TABLE I  
SECURITY CAPABILITY COMPARISON OF REPRESENTATIVE T2L JAILBREAK METHODS. ACCESS: ASSUMED THREAT MODEL. MULTI-LAYER DEFENSE: WHETHER THE METHOD ATTACKS A COMPOSED DEFENSE PIPELINE CONTAINING MULTIPLE HETEROGENEOUS STAGES, WITH SUCCESS REQUIRING ALL STAGES TO BE BYPASSED. FEEDBACK GRANULARITY: THE SIGNAL USED TO GUIDE THE SEARCH. LAYER-AWARE: WHETHER THE ATTACK LOCALIZES THE LAYER RESPONSIBLE FOR REJECTION. CROSS-LAYER ADAPTIVE: WHETHER IT ADAPTS ITS STRATEGY ACROSS CONFLICTING LAYERS. ITERATIVE: WHETHER IT REFINES OVER MULTIPLE ROUNDS. CRACK DERIVES THE FEEDBACK DIAGNOSIS FROM ITS DEFENSE AGENT WITHOUT ACCESSING THE INTERNAL STATES OF THE TARGET SYSTEM.
<table><tr><td>Method</td><td>Access</td><td>Multi-layer defense</td><td>Feedback granularity</td><td>Layer- aware</td><td>Cross-layer adaptive</td><td>Iterative</td></tr><tr><td>Ring-A-Bell [9]</td><td>White-box</td><td>X</td><td>Concept score</td><td>x</td><td>x</td><td>x</td></tr><tr><td>MMA-Diffusion [19]</td><td>White-box &amp; Black-box</td><td>√</td><td>Gradient</td><td>x</td><td>x</td><td>√</td></tr><tr><td>SneakyPrompt [8]</td><td>Black-box</td><td>X</td><td>Binary + CLIP</td><td>x</td><td>x</td><td>√</td></tr><tr><td>DACA [20]</td><td>Black-box</td><td>x</td><td>LLM heuristic</td><td>x</td><td>x</td><td>x</td></tr><tr><td>JailFuzzer [17]</td><td>Black-box</td><td>x</td><td>Binary success</td><td>x</td><td>x</td><td>√</td></tr><tr><td>PGJ [16]</td><td>Black-box</td><td>x</td><td>Perceptual cue</td><td>x</td><td>x</td><td>x</td></tr><tr><td>JANUS [22]</td><td>Black-box</td><td>√</td><td>Binary + NSFW score</td><td>x</td><td>x</td><td>√</td></tr><tr><td>MacPrompt [23]</td><td>Black-box</td><td>Partial</td><td>NSFW score</td><td>x</td><td>x</td><td>√</td></tr><tr><td>CRACK (ours)</td><td>Black-box</td><td>√</td><td>Surrogate Per-layer diagnosis</td><td>√</td><td>√</td><td>√</td></tr></table>

Output-side classifiers inspect generated images for unsafe content. Representative examples include CLIP-based NSFW detectors [26], NudeNet for nudity detection [27], and Q16, a multi-label classifier for inappropriate visual content [28]. Safety can also be incorporated into the diffusion process itself. Safe Latent Diffusion steers denoising away from inappropriate concepts through safety guidance [13]. In addition, cross-modal safety assessment can be built on vision-language representations that encode the relationship between text and images [14].

Commercial T2I services may combine several safeguards. For example, the DALL·E 3 system card documents safety mitigations that operate before and during generation [11]. However, prior work has not formally characterized the joint geometry induced by heterogeneous defense layers or their interactions. Our Detection Surface formalism addresses this gap.

## C. Multi-Agent Systems and LLM-Based Red Teaming

Multi-agent frameworks coordinate multiple LLM-based agents with complementary roles to solve complex tasks through iterative interaction [29], [30], [31]. Multi-agent debate enables agents to propose competing solutions, critique one another, and revise their decisions, improving reasoning accuracy and factual consistency [29], [32]. Related studies [30], [31] further show that role specialization and divergent agent perspectives can broaden solution exploration and reduce the tendency of a single model to converge prematurely on one reasoning path.

Recently, LLM-based agents have been increasingly used to automate jailbreak search. Methods such as PAIR [33] and TAP [34] iteratively generate, evaluate, and refine candidate prompts based on target-model responses, while broader redteaming frameworks [35], [36] introduce diversity-oriented exploration and specialized agent roles to expand attack coverage. Together, these studies demonstrate that iterative feedback and role specialization can improve the effectiveness and coverage of automated security evaluation.

However, these methods are designed primarily for textbased LLMs, where attack success is determined directly from the textual response of a single target model. Extending this paradigm to T2I systems is non-trivial because each prompt induces a stochastic image-generation process whose output may vary across queries, and the resulting prompt-image pair can be examined by heterogeneous textual, visual, and crossmodal safety filters. These characteristics require multi-agent interactions that can reason over layer-specific multimodal feedback and coordinate potentially conflicting safety signals, capabilities that are not explicitly considered in existing LLMoriented red-teaming frameworks.

Recent T2I frameworks such as DACA [20] and Jail-Fuzzer [17] have introduced LLM-based agents for intent decomposition, prompt mutation, and candidate evaluation. However, their feedback is primarily used to guide prompt generation or assess overall attack success, without explicitly modeling cross-layer interactions or adapting the search direction to the currently active constraint. This leaves layer-aware diagnosis and cross-layer arbitration underexplored in existing agent-based T2I red teaming.

## D. RL for Adversarial Prompt Generation

Reinforcement learning (RL) has been used to automate adversarial prompt search. SneakyPrompt [8] applies PPO [37] to learn token-level prompt substitutions against a CLIP-based safety filter. In the LLM setting, Perez et al. [38] trained language models through reinforcement learning to discover prompts that elicit undesirable behavior from target models. These approaches demonstrate that reward-guided optimization can reduce reliance on manually designed attack prompts. However, they typically optimize against a single target or aggregate success signal, without distinguishing how different safety constraints respond to each mutation. Such feedback is insufficient for heterogeneous defense stacks, where improving evasion against one layer may increase detection risk at another.

CRACK differs in two respects. First, its policy operates at the strategy level, selecting among qualitatively different mutation operators according to feedback from distinct defense layers, rather than performing only token-level replacement. Second, its reward combines binary bypass outcomes with continuous scores from the Judge Agent, thereby providing denser feedback for navigating the sparse evasion region.

## III. PRELIMINARIES AND PROBLEM FORMULATION

## A. Threat Model

Target model. Let P denote the set of natural-language prompts and let I denote the image space. A text-to-image (T2I) generative model is represented as

$$
y \sim { \mathcal { M } } ( p , z ) ,\tag{1}
$$

where $p \in \mathcal P$ is an input prompt, $z \in { \mathcal { Z } }$ is a stochastic latent variable, and $y \in \ \mathcal { T }$ is the generated image. The model is equipped with a safety mechanism composed of multiple heterogeneous filters that operate across different modalities and levels of abstraction, ranging from textual keyword matching and semantic prompt analysis to imagelevel content moderation.

Adversarial goal. Given a harmful target concept described by an original prompt $p _ { h } .$ , the adversary seeks an adversarial prompt $p ^ { ' }$ that evades all filters of the safety mechanism while keeping the generated image semantically faithful to the harmful intent of $p _ { h }$ . Formally, the adversary aims to solve

$$
p ^ { \star } = \arg \operatorname* { m a x } _ { p \in \mathcal { P } } \mathrm { S i m } \big ( \mathcal { M } ( p , z ) , p _ { h } \big ) \quad \mathrm { s . t . } \quad \mathcal { S } ( p ) = 0 ,\tag{2}
$$

where $S ( p ) = 0$ indicates that p passes all safety filters and Sim(·, ·) measures the semantic fidelity between the generated image and the original intent. An attack that bypasses the filters but produces a benign or off-target image is not counted as successful.

Adversarial capabilities. We consider a strict black-box setting. The adversary has no access to the internal weights, architecture, gradients, or logits of M, and no access to the parameters, thresholds, or decision logic of the safety filters. The number of filters, their types, and the order in which they are applied are all unknown. The adversary interacts with the deployed system only by submitting prompts and observing a single binary safe/unsafe signal, together with the generated image when a prompt is accepted.

Query and fidelity constraints. The attacker can issue repeated queries and revise the prompt based on past responses, but it operates purely at the input level and cannot modify the model or the filters in any way. This interaction is further constrained by a limited query budget, which rules out exhaustive search, and by the requirement that the adversarial prompt preserve the harmful semantics of $p _ { h }$ , which excludes trivial solutions that satisfy the filters by discarding the original intent.

B. Detection Surface Analysis: A Unified View of Multi-Layer Defenses

Modern T2I systems do not rely on a single safety check. Instead, they deploy a heterogeneous defense stack

$$
{ \cal S } = \{ S _ { 1 } , S _ { 2 } , \ldots , S _ { K } \} ,\tag{3}
$$

where K denotes the number of filters in the stack, and different filters $S _ { k }$ operate on text, images, or cross-modal representations.

In practice, these layers typically include: (1) text-based filters $S _ { \mathrm { t e x t } }$ that perform keyword matching or text classification before image generation; (2) image-based filters $S _ { \mathrm { i m a g e } }$ that inspect generated images using safety or NSFW classifiers; and (3) cross-modal filters $S _ { \mathrm { c r o s s } }$ that jointly evaluate promptimage alignment or unsafe concept similarity. We formalize the joint behavior of such defense stacks through the following definitions.

Natural-language prompts are discrete objects, so geometric notions such as distance, boundary, gradient, and convexity are not directly defined on P. To support geometric analysis, we introduce a representation map

$$
\phi : { \mathcal { P } }  \mathcal { X } \subseteq \mathbb { R } ^ { d } ,\tag{4}
$$

where X is a continuous prompt representation space. For a prompt p, we write $x = \phi ( p )$ . All geometric analysis below is conducted in $\mathcal { X }$ rather than directly in the discrete prompt space P.

Definition 1 (Detection Function and Safe Region). Each filter $S _ { k }$ induces a binary decision function

$$
h _ { k } : \mathcal { X }  \{ 0 , 1 \} ,\tag{5}
$$

where $h _ { k } ( x ) = 1$ indicates that $S _ { k }$ rejects the prompt representation x, and $h _ { k } ( x ) = 0$ indicates that it passes the filter. The safe region of the k-th filter is

$$
{ \mathcal { B } } _ { k } \triangleq h _ { k } ^ { - 1 } ( 0 ) = \{ x \in \mathcal { X } \mid h _ { k } ( x ) = 0 \} .\tag{6}
$$

The composite safe region of the full defense stack is

$$
{ \mathcal { B } } \triangleq \bigcap _ { k = 1 } ^ { K } { \mathcal { B } } _ { k } = \{ x \in { \mathcal { X } } \mid h _ { k } ( x ) = 0 , \forall k \in [ K ] \} .\tag{7}
$$

Thus, $x \in B$ if and only if it passes every filter in the defense stack.

In the black-box setting, the adversary observes only the composite decision

$$
h ( x ) = \operatorname* { m a x } _ { k \in [ K ] } h _ { k } ( x ) .\tag{8}
$$

Because $h _ { k } ( x ) = 1$ denotes rejection, $h ( x ) = 1$ means that at least one filter rejects x, while $h ( x ) = 0$ means that all filters accept it. The per-layer decisions $h _ { k } ( x )$ are not directly accessible in the black-box setting.

Assumption 1 (Surrogate Realizability). For geometric analysis, we associate each binary decision function $h _ { k }$ with a continuous surrogate scoring function

$$
f _ { k } : \mathcal { X } \to \mathbb { R }\tag{9}
$$

such that

$$
h _ { k } ( x ) = \mathbf { 1 } [ f _ { k } ( x ) \geq 0 ] .\tag{10}
$$

Equivalently, the safe region is $B _ { k } = \{ x \in \mathcal { X } \mid f _ { k } ( x ) < 0 \}$ and the zero level set separates the safe and unsafe regions without an exogenous threshold. When $f _ { k }$ is differentiable at $x ,$ the local evasion direction for the k-th filter is

$$
\mathbf { g } _ { k } ( x ) = - \nabla f _ { k } ( x ) .\tag{11}
$$

This assumption is exact for score-based classifiers, where $f _ { k } ( x )$ is the pre-sigmoid logit of the final layer, and provides a local approximation for rule-based or LLM-based filters.

Definition 2 (Detection Surface). Under Assumption 1, the decision boundary of the k-th filter is the zero level set

$$
\partial { \mathcal { B } } _ { k } = f _ { k } ^ { - 1 } ( 0 ) = \{ x \in \mathcal { X } \mid f _ { k } ( x ) = 0 \} .\tag{12}
$$

The Detection Surface of the full defense stack is the union of all filter boundaries

$$
\mathcal { D } = \bigcup _ { k = 1 } ^ { K } \partial \mathcal { B } _ { k } .\tag{13}
$$

The Detection Surface describes where the decisions of one or more filters may change. A point near $\partial B _ { k }$ is close to changing the decision of the k-th filter, while a point near multiple boundaries may be constrained by several filters simultaneously.

Definition 3 (Evasion Region). Let $p _ { h } \in \mathcal { P }$ denote the $o r i g .$ inal harmful-intent prompt with representation $x _ { h } = \phi ( p _ { h } )$ . A successful adversarial prompt must pass all filters while preserving the harmful semantic intent of $p _ { h }$ in the generated output. Given an output-level similarity metric Sim(·, ·) (e.g., CLIPScore between generated images) and a fidelity threshold δ, we define the semantic fidelity set as

$$
\mathcal { F } ( x _ { h } , \delta ) \triangleq \{ x \in \mathcal { X } \mid \operatorname { S i m } ( \mathcal { M } ( x ) , \mathcal { M } ( x _ { h } ) ) \geq \delta \} ,\tag{14}
$$

where $\mathcal M ( x )$ denotes generation from the prompt with representation x.

The evasion region is then defined as

$$
\mathcal { E } ^ { * } = \left( \bigcap _ { k = 1 } ^ { K } \mathcal { B } _ { k } \right) \cap \mathcal { F } ( x _ { h } , \delta ) .\tag{15}
$$

Equivalently, $x \in \mathcal { E } ^ { * }$ if and only if

$$
h _ { k } ( x ) = 0 , \quad \forall k \in [ K ] , \qquad \operatorname { S i m } ( x , x _ { h } ) \geq \delta .\tag{16}
$$

The attack objective is to find a prompt $p ^ { \prime } \in \mathcal { P }$ such that

$$
\phi ( p ^ { \prime } ) \in { \mathcal { E } } ^ { * } .\tag{17}
$$

## C. Key Properties of the Detection Surface

We now state three structural properties that characterize the difficulty of searching for $\mathcal { E } ^ { * }$ . Unlike the definitions above, these properties depend on explicit assumptions about the defense stack and the prompt distribution.

Property 1 (Cross-Layer Conflict). Different filters impose conflicting constraints on the representation space. Formally, there exist two filters $S _ { m } , S _ { n }$ with $m \neq n ,$ a point $x \in \mathcal { X }$ and two transformations $T _ { i } , T _ { j }$ such that

$$
T _ { i } ( x ) \in \mathcal { B } _ { m } \mathrm { ~ b u t } T _ { i } ( x ) \notin \mathcal { B } _ { n } , \qquad T _ { j } ( x ) \in \mathcal { B } _ { n } \mathrm { ~ b u t } T _ { j } ( x ) \notin \mathcal { B } _ { m } .\tag{18}
$$

This means that a modification that helps evade one filter may fail against another filter, so no single evasion direction is uniformly effective across all layers.

Intuition. Strategies that evade one detection layer increase exposure to another. For example, replacing sensitive keywords with obfuscated tokens [8] may reduce the risk of textbased keyword matching, but it can also produce unnatural text that is more easily flagged by semantic or cross-modal filters. Conversely, fluent paraphrasing may preserve naturalness but retain semantic cues that are detectable by LLM-based classifiers. This motivates layer-aware feedback rather than a single uniform mutation strategy.

Under Assumption 1, this conflict can be expressed locally through gradient misalignment. Consider two differentiable surrogate functions $f _ { m }$ and $f _ { n } . \mathrm { { I f } }$

$$
\langle \nabla f _ { m } ( x ) , \nabla f _ { n } ( x ) \rangle < 0 ,\tag{19}
$$

then following the evasion direction of one filter increases the detection score of the other to first order.

Proposition 1 (Gradient Conflict). Assume $f _ { m }$ and $f _ { n }$ are differentiable at x and $\langle \nabla f _ { m } ( x ) , \nabla f _ { n } ( x ) \rangle < 0 .$ . Then moving along the evasion direction ${ \bf g } _ { m } ( x ) = - \nabla f _ { m } ( x )$ increases $f _ { n }$ to first order; since $f _ { n } \geq 0$ denotes rejection by $S _ { n } ,$ evading $S _ { m }$ locally pushes x toward rejection by $S _ { n } .$

Proof: Let $x _ { \epsilon } = x + \epsilon { \bf g } _ { m } ( x )$ for a small $\epsilon > 0$ . By firstorder Taylor expansion,

$$
f _ { n } ( x _ { \epsilon } ) = f _ { n } ( x ) + \epsilon \langle \nabla f _ { n } ( x ) , \mathbf { g } _ { m } ( x ) \rangle + o ( \epsilon ) .\tag{20}
$$

Since ${ \bf g } _ { m } ( x ) = - \nabla f _ { m } ( x )$ , we have

$$
\langle \nabla f _ { n } ( x ) , \mathbf { g } _ { m } ( x ) \rangle = - \langle \nabla f _ { n } ( x ) , \nabla f _ { m } ( x ) \rangle .\tag{21}
$$

Since $\langle \nabla f _ { m } ( x ) , \nabla f _ { n } ( x ) \rangle < 0 ,$ , it follows that

$$
- \langle \nabla f _ { m } ( x ) , \nabla f _ { n } ( x ) \rangle > 0 .\tag{22}
$$

Hence, for all sufficiently small $\epsilon > 0$ , the positive linear term dominates $o ( \epsilon )$ , giving $f _ { n } ( x _ { \epsilon } ) > f _ { n } ( x )$ . Thus, moving along the evasion direction for $S _ { m }$ increases the detection score of $S _ { n }$ locally.

Consequently, single-objective greedy descent on one filter is fundamentally inadequate against a conflicting stack.

Property 2 (Sparsity). The evasion region shrinks as more filters and the semantic constraint must be satisfied simultaneously. Let $\nu$ be a prompt sampling distribution over $\mathcal { X }$ and define

$$
\rho _ { k } = \operatorname* { P r } _ { x \sim \nu } [ x \in \mathcal { B } _ { k } ] , \qquad \rho _ { \mathrm { s e m } } = \operatorname* { P r } _ { x \sim \nu } [ x \in \mathcal { F } ( x _ { h } , \delta ) ] .\tag{23}
$$

Proposition 2 (Sparsity of the Evasion Region).

$$
\operatorname* { P r } _ { x \sim \nu } [ x \in \mathcal { E } ^ { * } ] \ \leq \ \operatorname* { m i n } \Bigl \{ \rho _ { \mathrm { s e m } } , \ \operatorname* { m i n } _ { k \in [ K ] } \rho _ { k } \Bigr \} \ \leq \ \rho _ { \operatorname* { m a x } } ,\tag{24}
$$

where $\rho _ { \mathrm { m a x } } = \operatorname* { m a x } _ { k } \rho _ { k } < 1$ . Every additional filter can only tighten this bound.

Proof: By Eq. (15), $\begin{array} { r } { \mathcal { E } ^ { * } \ = \ \mathcal { F } ( \boldsymbol { x } _ { h } , \boldsymbol { \delta } ) \cap \bigcap _ { k = 1 } ^ { K } \mathcal { B } _ { k } } \end{array}$ is an intersection of events, and $\begin{array} { r } { \operatorname* { P r } [ \bigcap _ { i } A _ { i } ] ~ \leq ~ \operatorname* { m i n } _ { i } \operatorname* { P r } [ A _ { i } ] } \end{array}$ for any events $A _ { 1 } , \ldots , A _ { m } .$ Applying this to the safe-region and fidelity events gives Eq. (24).

The bound already shows that satisfying more constraints shrinks $\mathcal { E } ^ { * }$ . The decay is sharper in practice. Conflicting filters (Property 1) reject overlapping regions of $x ,$ so joint passage is even less likely than the min bound suggests and $\operatorname* { P r } _ { x \sim \nu } [ x \in \mathcal { E } ^ { * } ]$ tends toward the product $\rho _ { \mathrm { s e m } } \prod _ { k = 1 } ^ { K } \bar { \rho } _ { k }$ , which decays exponentially in K.

Property 3 (Non-Convexity). The evasion region $\mathcal { E } ^ { * } \subseteq \mathbb { R } ^ { d }$ is generically non-convex, because it is the intersection of safe regions bounded by nonlinear classifiers with a semantic fidelity set. Concretely, when each $f _ { k }$ is a ReLU network, $f _ { k } ^ { - 1 } ( 0 )$ partitions $\mathcal { X }$ into exponentially many linear regions [39], and $\boldsymbol { B } _ { k }$ is a union of a subset of these polytopes; the intersection $\bigcap _ { k } B _ { k }$ is therefore a union of polytopes rather than a single convex body. The following witness condition makes non-convexity operational.

Proposition 3 (Witness for Non-Convexity of $\mathcal { E } ^ { * } )$ . Suppose some safe region $\boldsymbol { B } _ { k }$ is non-convex. Formally, there exist $x _ { 1 } , x _ { 2 } \in B _ { k }$ and $\lambda \in ( 0 , 1 )$ such that

$$
x _ { \lambda } = \lambda x _ { 1 } + ( 1 - \lambda ) x _ { 2 } \notin { \mathcal { B } } _ { k }\tag{25}
$$

and

$$
x _ { 1 } , x _ { 2 } \in \mathcal { F } ( x _ { h } , \delta ) \cap \bigcap _ { j \neq k } \mathcal { B } _ { j }\tag{26}
$$

Then $\mathcal { E } ^ { * }$ is non-convex.

Proof: $x _ { 1 } , x _ { 2 }$ lie in every safe region $B _ { j }$ , including $\boldsymbol { B } _ { k }$ and in ${ \mathcal { F } } ( x _ { h } , \delta ) , \ s \mathbf { o } \ x _ { 1 } , x _ { 2 } \in { \mathcal { E } } ^ { * }$ . Since $x _ { \lambda } \notin B _ { k }$ , also $x _ { \lambda } \notin \mathcal { E } ^ { \ast }$ The segment between two points of $\mathcal { E } ^ { * }$ leaves $\mathcal { E } ^ { * }$ , so $\mathcal { E } ^ { * }$ is non-convex.

Each filter $S _ { k }$ is a neural classifier, and its safe region $\boldsymbol { B } _ { k }$ is bounded by a non-linear decision surface [40]. Non-convexity and the cross-layer conflict of Property 1 cause local greedy search to fail through two mechanisms. Conflicting gradients steer a mutation toward rejection by another layer, and the non-convex geometry traps a trajectory in a region that satisfies some filters but violates others. Effective search must explore multiple directions and revise earlier decisions when feedback indicates that a different boundary has become active.

Together, these three properties recast jailbreak search as a constrained search over the prompt space $\mathcal { P } .$ . Cross-layer conflict motivates layer-aware feedback, sparsity motivates reward-guided search, and non-convexity motivates iterative exploration rather than a single local mutation. CRACK, presented next, is designed around these three challenges.

## IV. CRACK: NAVIGATING THE DETECTION SURFACE VIA MULTI-AGENT DEBATE

Building on the Detection Surface analysis, we present CRACK, a multi-agent framework for adaptively navigating the Detection Surface toward the joint evasion region $\mathcal { E } ^ { * }$ . As illustrated in Fig. 1, CRACK implements three key requirements identified above, namely exploration, diagnosis, and arbitration, through multi-round interactions among three specialized agents. In each round, the Attack Agent performs diagnosisguided exploration by generating candidate mutations along complementary search directions. The Defense Agent provides structured layer-wise diagnosis by identifying the safety constraints currently impeding progress and explaining the corresponding failures. The Judge Agent arbitrates potentially conflicting feedback and evaluates the overall progress of each mutation, producing a reward signal that further guides refinement of the attack strategy. Through repeated rounds of debate and reward-guided adaptation, CRACK continuously revises its search direction as new cross-layer conflicts emerge, transforming stack-agnostic prompt mutation into a layeraware and adaptive search toward $\mathcal { E } ^ { * }$

## A. Attack Agent: Prompt Mutation on the Detection Surface

The Attack Agent serves as the exploration component of CRACK, adaptively selecting and applying mutation strategies to navigate the Detection Surface toward the joint evasion region $\mathcal { E } ^ { * }$ . Given a harmful prompt $p _ { h }$ , its objective is to generate a mutated prompt $p ^ { \prime } \in \mathcal { E } ^ { * }$ that simultaneously lies within all layer-wise safe regions $\boldsymbol { B } _ { k }$ while preserving the original harmful intent. Within the Detection Surface, each mutation represents a movement in the prompt space, and the Attack Agent iteratively searches for directions that improve overall progress toward joint evasion. Specifically, it selects mutation strategies conditioned on the layer-wise diagnosis provided by the Defense Agent and progressively refines its strategy-selection policy using the reward assigned by the Judge Agent.

Layer-Aware Adaptive Prompt Revision. For the initial prompt (debate round $t = 0 )$ , the risk report is set as $R _ { 0 } = \varnothing$ and the Attack Agent executes a randomly sampled mutation strategy from the strategy library M on the initial harmful prompt $p _ { h }$

At debate round t, given the current adversarial prompt p<sub>t</sub> and the structured risk report $R _ { t } = ( r _ { t } ^ { ( 1 ) } , r _ { t } ^ { ( 2 ) } , r _ { t } ^ { ( 3 ) } )$ provided by the Defense Agent, where each $r _ { t } ^ { ( k ) }$ indicates the detection outcome at the k-th layer, the Attack Agent selects a mutation strategy $a _ { t }$ from the strategy library M based on the policy network $\pi _ { \phi } \colon$

$$
a _ { t } \sim \pi _ { \phi } ( \cdot \mid p _ { t } , R _ { t } ) .\tag{27}
$$

The inclusion of the per-layer risk report $R _ { t }$ in the policy conditioning is critical. By Property 1 (cross-layer conflict), different layers require different evasion strategies. The structured feedback enables the Attack Agent to select targeted mutations, such as choosing metaphor replacement when the text pattern layer flags the prompt versus scene restructuring when the semantic layer is triggered, rather than applying blind, uniform perturbations.

The strategy library M contains five complementary mutation strategies. These strategies are selected to cover distinct and commonly used mutation dimensions, including lexical form, semantic framing, scene organization, and prompt structure, while keeping the action space sufficiently compact for policy learning [8], [16], [20], [17]. The detailed implementations of the five strategies are provided in Supplementary Sec. A. Each strategy modifies a different aspect of the prompt while preserving its underlying harmful intent, thereby providing complementary search directions over the prompt space P. Such diversity is particularly important for navigating the non-convex Detection Surface, where relying on a single mutation strategy may confine the search to a restricted region of P. By adaptively switching among complementary strategies according to layer-wise diagnosis, the Attack Agent can redirect its search when the current mutation direction becomes ineffective and explore alternative pathways toward E<sup>∗</sup>.

![](images/25fd2be22945b3f1fdb4cbde8fa6afb17cafd50e7293e7e4496641f330d049f5.jpg)  
Fig. 1. Overall framework of CRACK. CRACK formulates prompt generation as a multi-agent debate. The attacker mutates prompts to evade safety filters the defender evaluates risk via a multi-layer pipeline, and the judge provides reward feedback. Black arrows indicate forward prompt generation and evaluation, while the green arrow shows RL-based policy updates guiding the attacker over multiple rounds.

Reward-Guided Strategy Optimization. After each prompt revision, the Judge Agent evaluates the mutation’s effectiveness and provides a quantitative reward reflecting the progress toward E<sup>∗</sup>. This feedback updates the attacker’s policy, guiding future mutations toward regions of the Detection Surface with lower aggregate detection risk. The RL mechanism is detailed in Sec. IV-D.

## B. Defense Agent: Probing the Detection Surface

The Defense Agent serves as the diagnostic component of CRACK, providing an attacker-side surrogate that approximates the heterogeneous detection criteria commonly used in multi-layer defense stacks. Rather than relying solely on coarse success-or-failure feedback, it produces structured layer-specific diagnostics to identify where and why a candidate prompt is likely to be rejected, thereby providing actionable guidance for subsequent mutation. Crucially, the attacker has no access to the internal composition or layer-wise decisions of the target safety stack. Consistent with black-box constraints, we only query the target system for externally observable generation results.

Despite differences in their implementations, many realworld defense mechanisms can be broadly characterized by the primary signals they examine, including lexical [8], [11], semantic [11], [12], and cross-modal cues [10], [14]. We focus on these three dimensions because they capture complementary sources of safety evidence commonly used across the T2I generation pipeline: lexical cues reflect surface-level patterns in textual prompts, semantic cues capture higherlevel unsafe intent beyond specific wording, and cross-modal cues assess the consistency and safety implications across textual and visual content. These dimensions are not intended to exhaustively cover all possible defense mechanisms, but provide a unified abstraction for analyzing heterogeneous safety signals within a compositional defense stack.

The surrogate does not replicate the target’s detectors. Instead, it identifies which dimension currently exposes the prompt. Removing that exposure produces a prompt that is harder to detect along the same dimension for any stack inspecting it, so the surrogate’s crack diagnosis transfers at the level of detection dimensions rather than specific implementations. A prompt passing all surrogate cracks lies within the target’s feasible region with high probability.

The agent’s role within the debate is twofold. First, it approximates the Detection Surface by evaluating prompts through a multi-tier pipeline that mirrors the structure of real-world defenses. Second, it provides per-layer diagnostic feedback that localizes where on the Detection Surface a given prompt is intercepted. This layer-specific localization is the key mechanism for resolving cross-layer conflicts (Property 1).

![](images/185179cf213698f064c3386bca46ef04b3feee15086de97852f6d0b55825ea31.jpg)  
Fig. 2. Risk assessment evaluation template for Defense Agent.  
Fig. 3. Risk assessment template for Judge Agent.

To provide structured layer-wise feedback, the Defense Agent operates a three-tiered detection mechanism, where each tier instantiates a distinct component of the Detection Surface.

Tier 1: Keyword and Pattern Matching (Textual Boundary $\partial B _ { \mathbf { t e x t } } )$ . A lightweight first-pass filter designed to detect explicit violations using a curated set of sensitive terms and regular expression patterns [41]. It targets entity terms (e.g., “naked”, “weapon model”), action verbs $( \mathrm { e . g . }$ , “attack”, “moan”), and scene descriptors (e.g., “violent conflict”). This tier approximates text-based filters $S _ { \mathrm { t e x t } }$ deployed in real systems, probing the lexical decision boundary $\partial { B _ { \mathrm { t e x t } } }$ <sub>t</sub>.

Tier 2: LLM Semantic Risk Assessment (Semantic Boundary $\partial B _ { \mathbf { s e m } } ) .$ . A fine-tuned language model analyzes the semantic intent of the prompt using a structured evaluation template (see Fig. 2). This module captures implicit threats and indirect cues that are missed by surface-level keyword filters, such as metaphorical references to violence or euphemistic descriptions of NSFW content. It probes the semantic decision boundary $\partial B _ { \mathrm { s e m } }$ , which operates at a higher level of abstraction than Tier 1.

Tier 3: CLIPScore Cross-Modal Validation (Cross-Modal Boundary $\partial B _ { \mathbf { c r o s s } } ) .$ . This tier evaluates the semantic consistency between the prompt and the generated image by computing the CLIPScore [15], which quantifies how well the visual output aligns with unsafe concept embeddings. It probes the cross-modal decision boundary $\partial B _ { \mathrm { c r o s s } } .$ , which operates across modalities.

The Defense Agent outputs a structured risk report $R _ { t }$ that specifies the detection outcome at each tier, enabling the Attack Agent to identify the specific layer(s) responsible for interception and to select targeted countermeasures. This is in direct contrast to prior approaches (e.g., JailFuzzer [17], SneakyPrompt [8]) that receive only aggregate binary feedback $h ( p ) = \operatorname* { m a x } _ { k } { h _ { k } ( p ) }$ , which collapses the per-layer outcomes into a single signal and reveals only whether a prompt is blocked, not which component of the Detection Surface is responsible.

C. Judge Agent: Measuring Progress Across the Detection Surface

The Judge Agent serves as the arbitration component of CRACK, responsible for quantifying how effectively each mutation moves the adversarial prompt toward the evasion region $\mathcal { E } ^ { * }$ . By integrating cross-modal semantic analysis with the Defense Agent’s per-layer assessments, the Judge Agent produces reward signals that encode progress along the Detection Surface.

In each round $t ,$ the Judge Agent evaluates both the original prompt $p _ { t }$ and the revised prompt $p _ { t + 1 }$ through a dual-signal scoring mechanism.

LLM-Based Semantic Assessment $S _ { \mathbf { r i s k } }$ . The first signal, $S _ { \mathrm { r i s k } }$ , measures whether the revision reduces the likelihood of triggering safety filters based on a structured evaluation template (see Fig. 3). It primarily reflects the mutation’s progress with respect to lexical and semantic detection signals.

CLIPScore Differential $\Delta _ { \mathbf { C L I P } } .$ . The second signal, $\Delta _ { \mathrm { C L I P } }$ computes the change in text-image alignment between rounds. Let $c _ { \mathrm { u n s a f e } }$ denote the unsafe concept embedding. The differential is defined as $\Delta _ { \mathrm { C L I P } } ~ = ~ \mathrm { C L I P } ( p _ { t } , c _ { \mathrm { u n s a f e } } ) ~ - ~$ $\mathrm { C L I P } ( p _ { t + 1 } , c _ { \mathrm { u n s a f e } } )$ , where positive values indicate that the revised prompt has moved further from the unsafe concept in the detector’s representation space, and negative values reflect less effective revisions. This signal captures progress across the cross-modal decision boundary $\partial B _ { \mathrm { c r o s s } } .$

The final decision score is computed as:

$$
s = \alpha \cdot S _ { \mathrm { r i s k } } + \beta \cdot \sigma ( \Delta _ { \mathrm { C L I P } } ) , \quad \mathrm { w i t h ~ } \alpha + \beta = 1 ,\tag{28}
$$

where $\alpha , \beta \in [ 0 , 1 ]$ weight the contributions of each component, and $\sigma ( \cdot )$ is a scaling function that normalizes $\Delta _ { \mathrm { C L I P } }$ to the same range as $S _ { \mathrm { r i s k } }$ . The weighting scheme balances progress across different dimensions of the Detection Surface. The parameter α governs sensitivity to textual and semantic decision boundaries, while $\beta$ governs sensitivity to crossmodal boundaries. We analyze this trade-off in the sensitivity study (Sec. V-B4).

## D. Multi-Agent Debate as Detection Surface Traversal

Fig. 4 presents an illustrative example demonstrating the complete multi-agent debate process across the Detection

![](images/e6a2efe8f799cf485ff3c748fc3a15f0ee3473936f7e93b394ae1664242d4ee7.jpg)  
Fig. 4. Overall pipeline of CRACK. CRACK performs multi-round debates between the attack agent and evaluation agents (defense and judge). In each round, candidate prompts are assessed for risk, semantic fidelity, and NSFW content, and feedback updates the generation strategy iteratively until the final prompt is selected.

![](images/48f7dfd89013b1bbe8899978ab6afbb9421830a59b362158d51df5eff8e2a397.jpg)  
Fig. 5. Iterative CRACK debate trajectory across the multi-filter detection surface. A single prompt’s CLIP embedding trajectory across debate revisions (R0-R6), showing its progression across filter boundaries from an initially blocked harmful prompt toward the joint evasion region.

Surface. Each debate round corresponds to one step along a trajectory through the prompt space, guided by per-layer feedback from the Defense Agent and reward signals from the Judge Agent. The debate terminates when the prompt enters $\mathcal { E } ^ { * } \left( \mathrm { i . e . } \right.$ , passes all detection tiers while maintaining semantic fidelity) or after a maximum number of rounds $T _ { \mathrm { d e b a t e } }$ as shown in Fig. 5.

Debate Protocol. CRACK organizes the interaction among the three agents as an iterative debate, in which each proposed mutation is challenged by layer-wise diagnosis and then evaluated through cross-layer arbitration before guiding the next revision.

In each round $t ,$ this debate proceeds through three sequential steps. First, the Attack Agent generates a mutated prompt $p _ { t + 1 }$ using the mutation strategy $a _ { t }$ sampled from the current policy network $\pi _ { \phi }$ and the Defense Agent’s diagnostic feedback $R _ { t }$ from the previous evaluation. Second, the Defense Agent evaluates $p _ { t + 1 }$ through the threetier detection pipeline and produces an updated risk report $R _ { t + 1 } = ( r _ { t + 1 } ^ { ( 1 ) } , \stackrel { \cdot } { r } _ { t + 1 } ^ { ( 2 ) } , r _ { t + 1 } ^ { ( 3 ) } )$ . Third, the Judge Agent compares $p _ { t }$ with $p _ { t + 1 }$ together with the corresponding diagnostic signals to assess the overall progress of the mutation, producing a judgment score $s _ { t }$ that contributes to the reward used to update the strategy-selection policy network $\pi _ { \phi }$ . This iterative proposal, diagnosis, and arbitration cycle constitutes the debate process, allowing CRACK to continuously adjust its mutation strategy in response to newly exposed layer-wise constraints and progressively steer the search toward $\mathcal { E } ^ { * }$ while preserving semantic fidelity.

Strategy Updating via Reinforcement Learning. Building on this iterative debate process, we incorporate reinforcement learning to continually refine the Attack Agent’s policy. The RL formulation translates the abstract goal of navigating to $\mathcal { E } ^ { * }$ into a concrete optimization problem.

State-Action Trajectory Collection. For each episode, the attacker samples mutation actions from the policy network $\pi _ { \phi } ,$ generates mutated prompts, receives reward signals through the debate, and records state transitions $\left( { { s _ { t } } , { a _ { t } } , { r _ { t } } , { s _ { t + 1 } } } \right)$ in a replay buffer. Here $s _ { t }$ represents the current prompt and per-layer detection status, $a _ { t }$ denotes the selected mutation strategy, $r _ { t }$ is the reward, and $s _ { t + 1 }$ is the resulting state after mutation. Implementation details are provided in Supplementary Sec. B.

Reward Formulation. The reward function encodes progress across the Detection Surface through two complementary signals.

• Bypass Reward. A binary signal of +b if the prompt evades all detection tiers $( h _ { k } ( p _ { t + 1 } ) = 0$ for all $k \in [ K ] )$ and −b otherwise. This reflects whether the prompt has reached the composite safe region B.

• Judge Reward. The judge’s score s from Eq. (28), providing a continuous signal that measures partial progress toward $\mathcal { E } ^ { * }$ even when not all layers are bypassed.

This composite reward drives the Attack Agent to simultaneously reduce detection risk across all layers (addressing sparsity) while preserving semantic intent (satisfying the fidelity constraint in the definition of ${ \mathcal { E } } ^ { * } )$ .

Policy Update. The policy network $\pi _ { \phi }$ is optimized by minimizing the standard Actor-Critic loss [42]:

$$
\mathcal { L } ( \phi ) = - \mathbb { E } _ { \boldsymbol { \pi } _ { \phi } } \left[ \sum _ { t = 0 } ^ { T } \log \pi _ { \phi } ( \boldsymbol { a } _ { t } \mid \boldsymbol { s } _ { t } ) \cdot \hat { \boldsymbol { A } } _ { t } \right] + \mathbb { E } \left[ \hat { A } _ { t } ^ { 2 } \right] ,\tag{29}
$$

where $\begin{array} { r } { \hat { A } _ { t } = \sum _ { k = t } ^ { T } \gamma ^ { k - t } r _ { k } - V _ { \phi } ( s _ { t } ) } \end{array}$ is the advantage estimate, $\gamma$ is the discount factor, and $V _ { \phi } ( s _ { t } )$ denotes the estimated value of state $s _ { t }$ . Through this optimization, the policy learns to select mutation strategies that efficiently traverse the Detection

![](images/e8b2d9b6e95d905161999dd199549baba482cdb710d2a595a5600114abd95b91.jpg)  
Fig. 6. Geometry of the multi-layer detection surface in CLIP embedding space. A two-dimensional t-SNE projection of prompt embeddings, overlaid with multiple filter boundaries; their joint lower region denotes candidate prompts that evade all filters.

Surface, accumulating experience about which strategies are effective against which detection layers. As shown in Fig. 6, in the CLIP embedding space, the prompts generated using the CRACK strategy are largely located within the evasion region, while other methods are unable to cross different filter boundaries.

Connection to Detection Surface Properties The debate mechanism directly addresses the three structural properties identified in Sec. III-C. Per-layer feedback from the Defense Agent resolves cross-layer conflicts (Property 1) by identifying which boundary the current prompt violates, enabling targeted rather than blind mutation. The iterative refinement over multiple rounds handles non-convexity (Property 3) by allowing the trajectory to curve around detection boundaries rather than committing to a single descent direction. The RL-guided policy optimization efficiently searches the sparse evasion region (Property 2) by learning from accumulated experience rather than relying on random exploration.

## V. EXPERIMENTS

## A. Experiments Setup

Datasets & Target Models. We evaluate our method on three widely used prompt datasets, NSFW-200 [19], I2P [43], and UnsafeDiff [44], following the settings in [17], [8], [19]. The NSFW-200 dataset contains exclusively adult-oriented content, while the I2P dataset provides real-world user prompts spanning a broad range of inappropriate concepts such as nudity, violence, and self-harm. The UnsafeDiff dataset covers three representative categories, including violence, gore, and discrimination, to broaden the scope of our attack assessment. The target T2I models include Stable Diffusion 1.4 [1], Stable Diffusion XL [2], DALL·3 [3], and Midjourney [4].

Attack Scenario. (a). One-time prompt attack: The adversaries generate a new adversarial prompt for each attempt, aiming to bypass the safety filter via real-time queries. (b). Re-use prompt attack: The adversaries reuse previously successful prompts to directly trigger NSFW image generation without further search.

Heterogeneous safety filters. Throughout all experiments, the built-in safety checker of the generator is kept active, so each configuration forms a heterogeneous stack $s \ =$ $\{ S _ { 1 } , \ldots , S _ { K } \}$ composed of the built-in checker and the listed external filters below. (a). Text-based safety filter: text-m [41] and text-c [45] assess textual inputs before image generation, using either classifiers or keyword lists to block prompts that are sensitive or closely related to sensitive content. (b). Imagebased safety filter: image-c [46] and image-clip-c [47] inspect the generated images using binary classifiers trained on NSFW-labeled data. (c). Text-image-based safety filter: textimage-c enforces moderation by jointly analyzing text and image using a binary classifier over combined embeddings.

Unlike the SD 1.4 configurations where an external filter is stacked on the built-in checker, DALL·E 3 requires no added filter. Its proprietary pipeline is itself a closed-source heterogeneous safety stack, whose internal filters cannot be disabled or enumerated.

Evaluation Metrics. (a). Attack Success Rate (ASR-Q16): The ratio between the number of adversarial prompts that successfully produce unsafe images and the total number of adversarial prompts. Specifically, a prompt counts as a successful attack only if it bypasses all active layers, i.e., $h ( p ) \ = \ \operatorname* { m a x } _ { k } h _ { k } ( \phi ( p ) ) \ = \ 0$ , and its generated image is flagged as unsafe by Q16 [28]. Here, Q16 is a multi-label safety classifier capable of detecting a broad range of unsafe visual content. In the re-use attack setting, each previously successful prompt is re-submitted to the target model, and ASR-Q16 measures the proportion that again produces unsafe images. (b). Attack Success Rate (ASR-NudeNet): Following the same procedure, we replace Q16 with NudeNet [48], a classifier specifically trained to detect nudity and sexually explicit content. The ratio is computed between the number of adversarial prompts whose generated images are classified as containing nudity by NudeNet and the total number of adversarial prompts. Since NudeNet targets a single content category, this metric is reported exclusively on the NSFW-200 dataset, which consists of adult-oriented prompts. (c). CLIPScore: While ASR serves as the primary measure of attack effectiveness, we additionally report CLIPScore as a complementary indicator of semantic alignment between the adversarial prompt and the generated image. It is computed as the cosine similarity between CLIP-encoded image and text embeddings (range [0,1]). (d). Query Number: The number of queries to T2I models used for searching for an adversarial prompt. (e). PPL: Perplexity is used to measure a model’s average uncertainty when predicting the next token. Lower PPL indicates that the generated text is more fluent and natural.

Baselines. We compare CRACK with six representative baselines covering text-only, multimodal, and LLM reasoning. SneakyPrompt [8] and Ring-A-Bell [9] are text-based attacks that iteratively modify or search prompt tokens to bypass filters. MMA-Diffusion [19] expands attacks to multimodal inputs by perturbing both text and image. JailFuzzer [17], DACA [20], and PGJ [16] rely on heuristic reasoning to decompose or recombine semantic structures.

Implementation. All experiments were conducted on a single NVIDIA A100-PCIE GPU with 40GB of memory. We use DeepSeek V3 [49] as the LLM for all agents. More advanced models, such as QWen 3 [50] and GPT-4 [51], are excluded due to built-in safety mechanisms that limit sensitive content handling. The weights α = 0.3 and β = 0.7 in Eq. (28) are chosen based on the sensitivity analysis below. And σ(·) function multiplies the input by 10. The bypass reward b is set to 10, and the policy network $\pi _ { \phi }$ is optimized using Adam with a learning rate of $3 \times 1 0 ^ { - 4 }$ . The code is available in the supplementary material.

TABLE II  
EFFECTIVENESS OF CRACK AGAINST DIFFERENT multi-layer SAFETY STACKS. WE REPORT ASR (%) UNDER RE-USE AND ONE-TIME PROMPT ATTACK SETTINGS. Q16 IS EVALUATED ON I2P AND UnsafeDiff, AND NUDENET IS EVALUATED ON NSFW-200. ADDITIONAL RESULTS ON SD XL AND MIDJOURNEY ARE IN SUPPLEMENTARY SEC. C. THE BEST RESULT IN EACH ROW IS IN BOLD.
<table><tr><td rowspan="3">Target</td><td rowspan="3">Type</td><td rowspan="3">External Filter</td><td rowspan="3">Method</td><td colspan="3">Re-use Prompt Attack</td><td colspan="3">One-time Prompt Attack</td></tr><tr><td>ASR-Q16</td><td></td><td>ASR-NudeNet</td><td></td><td>ASR-Q16</td><td>ASR-NudeNet</td></tr><tr><td>I2P</td><td>UnsafeDiff</td><td>NSFW-200</td><td>I2P</td><td>UnsafeDiff</td><td></td><td>NSFW-200</td></tr><tr><td rowspan="9">text Stable Diffusion 1.4</td><td rowspan="5">text-image</td><td rowspan="5">text-image-c</td><td>JailFuzzer SneakyPrompt</td><td>48.49 59.36</td><td>61.35</td><td>23.13</td><td>43.30</td><td>55.01</td><td>13.16</td></tr><tr><td></td><td></td><td>76.30</td><td>30.11</td><td>46.64</td><td>51.02</td><td>33.63</td></tr><tr><td>Ring-A-Bell</td><td>42.31</td><td>53.63</td><td>54.56</td><td>33.61</td><td>37.96</td><td>43.13</td></tr><tr><td>DACA</td><td>55.63</td><td>56.61</td><td>32.61</td><td>30.03</td><td>31.63</td><td>27.56</td></tr><tr><td>MMA-diffusion</td><td>33.01</td><td>34.56</td><td>10.36</td><td>12.01</td><td>10.01</td><td>9.63</td></tr><tr><td>PGJ</td><td>CRACK (ours)</td><td>45.63 33.69</td><td></td><td>23.45</td><td>24.56</td><td>23.63</td><td>11.24</td></tr><tr><td rowspan="9">text-m</td><td rowspan="5"></td><td>JailFuzzer</td><td>66.02 33.63</td><td>79.63 34.01</td><td>88.61 10.03</td><td>83.01 68.13</td><td>78.69 78.94</td><td>76.35 19.03</td></tr><tr><td>SneakyPrompt</td><td>96.01</td><td>93.69</td><td>91.06</td><td>52.36</td><td>63.69</td><td>56.96</td></tr><tr><td>Ring-A-Bell</td><td>88.63</td><td>87.12</td><td>79.63</td><td>11.56</td><td>12.39</td><td>13.69</td></tr><tr><td>DACA</td><td>83.16</td><td>86.16</td><td>77.63</td><td>16.36</td><td>23.16</td><td>10.01</td></tr><tr><td>MMA-diffusion</td><td>93.01</td><td>83.06</td><td>83.61</td><td>63.23</td><td>73.06</td><td>23.10</td></tr><tr><td>PGJ</td><td>78.63</td><td>76.96</td><td>73.12</td><td>46.39</td><td>56.39</td><td></td><td>43.65</td></tr><tr><td></td><td>CRACK (ours)</td><td>97.36</td><td>98.63</td><td>95.01</td><td>99.30</td><td>98.36</td><td>90.31</td></tr><tr><td>SneakyPrompt</td><td>JailFuzzer</td><td>91.23 90.36</td><td>93.63</td><td>89.63</td><td>78.30</td><td>88.01</td><td>56.31</td></tr><tr><td></td><td></td><td>91.65</td><td></td><td>86.36</td><td>61.23</td><td>83.63</td><td>62.36</td></tr><tr><td rowspan="6"></td><td rowspan="5">text-c</td><td>Ring-A-Bell</td><td>88.06</td><td>79.56</td><td>71.23</td><td>5.30</td><td>33.63</td><td></td><td>25.96</td></tr><tr><td>DACA</td><td>90.31</td><td>89.16</td><td>80.12</td><td>3.60</td><td>34.12</td><td></td><td>22.16</td></tr><tr><td>MMA-diffusion</td><td>88.63</td><td>87.96</td><td>85.16</td><td>61.23</td><td></td><td>77.26</td><td>60.12</td></tr><tr><td>PGJ</td><td>81.36</td><td>78.56</td><td>88.69</td><td></td><td>44.12</td><td>23.36</td><td>12.63</td></tr><tr><td>CRACK (ours)</td><td>99.63</td><td>95.61</td><td>96.26</td><td>90.23</td><td></td><td>91.26</td><td>93.61</td></tr><tr><td>JailFuzzer</td><td>33.61</td><td>36.25</td><td></td><td>34.13</td><td>85.16</td><td>79.61</td><td>56.13</td></tr><tr><td rowspan="8">image</td><td rowspan="5">image-c</td><td>SneakyPrompt</td><td>86.13</td><td>88.16</td><td>90.58</td><td>92.13</td><td>94.12</td><td></td><td>85.63</td></tr><tr><td>Ring-A-Bell</td><td>33.12</td><td>34.19</td><td>26.15</td><td>55.10</td><td></td><td>45.16</td><td>13.21</td></tr><tr><td>DACA</td><td>34.56</td><td>44.63</td><td>44.56</td><td>34.63</td><td></td><td>44.36</td><td>11.63</td></tr><tr><td>MMA-diffusion</td><td>33.12</td><td>60.35</td><td></td><td></td><td></td><td>69.16</td><td></td></tr><tr><td>PGJ</td><td>26.36</td><td></td><td>33.68</td><td>70.16</td><td></td><td>22.63</td><td>53.16</td></tr><tr><td>CRACK (ours)</td><td>87.13</td><td>43.84 85.16</td><td>44.53 88.61</td><td></td><td>65.96 95.16</td><td>96.23</td><td>12.38 88.24</td></tr><tr><td></td><td>JailFuzzer</td><td>77.32 70.15</td><td></td><td>66.31</td><td>81.23</td><td>80.13</td><td>63.12</td></tr><tr><td></td><td>SneakyPrompt</td><td>66.32</td><td>69.30</td><td>57.13</td><td>80.35</td><td>79.03</td><td>61.03</td></tr><tr><td>image-clip-c</td><td>Ring-A-Bell DACA</td><td>61.23</td><td>33.65</td><td>45.12</td><td>15.63</td><td>17.69</td><td>4.36</td></tr><tr><td>MMA-diffusion</td><td></td><td>23.12</td><td>37.16</td><td>35.64</td><td>25.63</td><td>32.12</td><td>15.69</td></tr><tr><td rowspan="5"></td><td>PGJ</td><td>45.63</td><td>44.13</td><td>31.23</td><td>60.15</td><td></td><td>66.35</td><td>48.63</td></tr><tr><td></td><td>35.26</td><td>43.63</td><td></td><td></td><td>66.95</td><td>65.13</td><td>16.45</td></tr><tr><td>CRACK (ours)</td><td>68.91</td><td>79.61</td><td>21.12</td><td></td><td>88.96</td><td>81.03</td><td></td></tr><tr><td>JailFuzzer</td><td>10.23</td><td></td><td>62.31</td><td></td><td>15.63</td><td></td><td>74.25</td></tr><tr><td>SneakyPrompt</td><td>11.23</td><td>11.36 12.34</td><td>15.69 13.06</td><td></td><td>25.69</td><td>16.58 23.15</td><td>12.36 21.13</td></tr><tr><td rowspan="5">DALL·E 3</td><td rowspan="5">unknown</td><td rowspan="5">unknown</td><td>Ring-A-Bell</td><td>9.63</td><td>9.09</td><td>2.36</td><td>13.46</td><td>14.15</td><td>12.01</td></tr><tr><td>DACA</td><td></td><td></td><td></td><td></td><td>15.16</td><td></td></tr><tr><td></td><td>8.63</td><td>8.08</td><td>5.13</td><td>15.01</td><td></td><td>14.63</td></tr><tr><td>MMA-diffusion</td><td>12.36</td><td>15.36</td><td>19.63</td><td>21.03</td><td>25.30</td><td>10.98</td></tr><tr><td>PGJ</td><td>5.63</td><td>5.96</td><td>5.14</td><td>22.09</td><td>20.01</td><td>26.05</td></tr><tr><td></td><td></td><td>CRACK (ours)</td><td>23.01</td><td>21.66</td><td>25.16</td><td>43.57</td><td>49.02</td><td></td><td>39.01</td></table>

## B. Evaluations

We evaluate our method by answering the following Research Questions (RQs).

• RQ1: How does our method perform compared with different baselines?

• RQ2: Do the three structural properties of the Detection Surface hold empirically, and to what extent do they explain CRACK’s design choices?

• RQ3: How does each component of CRACK impact performance?

![](images/eb84dbaceac19ca36f9a90a153a8870dbcea18250d7fbfc1be3ca01462b839f4.jpg)  
(a)

![](images/e4a3582ed12e2417eae852dda0a64320dbced308a0a2346aeadeb15193edecfb.jpg)  
(b)  
Fig. 7. (a) Average queries on SD 1.4 (shaded: standard deviation). (b) Perplexity (See more results on other T2I models in the Supplementary Sec. D, Sec. E).

• RQ4: How do different hyperparameters affect the performance of our method?

• RQ5: How does our method demonstrate comprehensiveness and generalization?

1) RQ1: Comparison with Different Baselines: Attack effectiveness on one-time attack. As shown in Tab. II, CRACK consistently achieves the highest ASR across nearly all safety filter configurations under the one-time attack setting. On Stable Diffusion 1.4 with the most challenging text-image joint filter (text-image-c), CRACK attains ASR-Q16 scores of 83.01% (I2P) and 78.69% (UnsafeDiff), surpassing the strongest baseline by 36.37 and 27.67 percentage points, respectively. The ASR-NudeNet on NSFW-200 reaches 66.35%, more than doubling the best baseline result of 33.63%. This substantial margin validates CRACK’s ability to navigate multi-layer defense stacks where text and image modalities are jointly monitored.

Under text filters, CRACK achieves near-perfect bypass. On text-m, it reaches 99.30% (I2P) and 98.36% (UnsafeDiff) on ASR-Q16, while all baselines except MMA-Diffusion fall below 68%. On text-c, CRACK maintains ASR-Q16 above 90% across all three datasets, whereas non-iterative methods such as Ring-A-Bell and DACA collapse to single-digit success rates (e.g., 5.30% and 3.60% on I2P), confirming that static heuristic strategies struggle to adapt to classifier-based text filters without iterative feedback.

Against image-based filters, CRACK achieves the best overall performance. On image-c, CRACK reaches 95.16% (I2P) and 96.23% (UnsafeDiff) on ASR-Q16, outperforming the next best method by 3.03 and 2.11 points, respectively. On the more discriminative image-clip-c filter, CRACK obtains 88.96% on I2P and 64.25% on NSFW-200, consistently leading all baselines. We note that SneakyPrompt achieves competitive re-use ASR on image-c (86.13% and 90.58% on certain datasets) due to its token-level perturbation strategy, which is particularly effective against image-only classifiers. However, this advantage does not transfer to cross-modal or text-based defenses, where SneakyPrompt’s nonsensical token replacements are easily intercepted.

On the commercial platform DALL·E 3, whose safety mechanisms are undisclosed and substantially more restrictive, CRACK achieves ASR-Q16 of 43.57% (I2P) and 49.02% (UnsafeDiff) under the one-time setting, roughly doubling the strongest baseline. The ASR-NudeNet on NSFW-200 reaches 39.01%, again outperforming all competitors. These results demonstrate CRACK’s generalizability to black-box commercial systems with unknown, multi-layered defenses.

Attack effectiveness on re-use attack. CRACK also leads in the re-use setting, achieving the highest ASR-Q16 in all but two filter–dataset combinations (Tab. II). On text-imagec, CRACK reaches 66.02% (I2P) and 79.63% (UnsafeDiff), and the ASR-NudeNet on NSFW-200 reaches 88.61%, far surpassing all baselines. Under text-only filters, all methods including CRACK achieve uniformly high re-use ASR, as text filters are deterministic and do not vary across generations. Under image-based filters, the re-use ASR is inherently more variable due to the stochastic nature of the diffusion sampling process. Different random seeds produce different images from the same prompt, and some generated images may trigger the image classifier. Despite this variability, CRACK maintains competitive performance, achieving 87.13% on image-c (I2P) and 68.91% on image-clip-c (I2P). On DALL·E 3, CRACK achieves 23.01% (I2P) and 25.16% (NSFW-200) on re-use ASR-NudeNet, consistently outperforming all baselines by a clear margin.

Efficiency. Beyond attack effectiveness, CRACK substantially reduces the query and runtime costs of iterative jailbreak search. As shown in Fig. 7a, CRACK requires only six queries per instance, the fewest among the iterative methods. In each of the three debate rounds, the Defense Agent performs one evaluation and the Judge Agent provides one comparative score, allowing CRACK to refine the prompt within a fixed query budget. By contrast, SneakyPrompt consumes up to 39.96 queries, JailFuzzer 29.55, and MMA-Diffusion approximately 10. We omit PGJ, DACA, and Ring-A-Bell from this comparison because their non-iterative designs do not use additional queries for progressive refinement. This query economy also translates into runtime efficiency, as shown in Tab. 9a. CRACK requires 123.01 seconds per prompt, making it the fastest iterative framework. Although it is slower than the non-iterative baselines, the additional runtime is accompanied by substantially higher bypass rates and therefore yields a favorable efficiency-effectiveness trade-off.

Prompt quality and fidelity. CRACK produces more natural adversarial prompts while preserving their target semantics. As shown in Fig. 7b, it achieves a minimum perplexity of 29.78 under the image-c stack, substantially lower than the values of SneakyPrompt, Ring-A-Bell, and MMA-Diffusion, indicating that its revisions remain fluent rather than relying on malformed token sequences. Fig. 8 further shows that the generated images retain the harmful intent and semantic structure of the original prompts. CRACK also obtains the highest average CLIPScore across all benchmarks, reaching 0.36 on NSFW-200, 0.39 on UnsafeDiff, and 0.26 on I2P (Fig. 9b). These results are consistent with the Judge Agent’s dualsignal reward, which favors mutations that reduce detection risk without sacrificing semantic fidelity. Although DACA and PGJ are competitive on individual datasets, neither performs consistently across all three benchmarks, supporting the value of iterative, feedback-driven refinement for preserving semantic fidelity across diverse prompt distributions.

2) RQ2: Detection Surface Property Validation: The theoretical framework in III proposes three structural properties of the Detection Surface that jointly explain why existing methods struggle under multi-layer defenses and why CRACK’s architecture is effective. We design three controlled experiments to validate each property independently.

![](images/4bff880f75494a1ff6f6b3a215a2f712b5699f54d6a01f2d826e0d1b67293ad5.jpg)  
Fig. 8. Demonstration of NSFW images generated by our method (See more cases in Supplementary Sec. F).

![](images/2ce68df3fff49f20952fd81c7069f475296c828c4845fbc528b19a6969103b2c.jpg)

![](images/3fd0fbf732afa7ff60a5439119af076e4bbb014826ecd6ee4130896259cb708a.jpg)  
Fig. 9. (a) Time overhead for generating an adversarial prompt compared with baselines. (b) Average CLIPScore for generating an adversarial prompt compared with baselines.

Cross-layer conflict. We isolate each of the five transformation strategies in CRACK’s strategy library and evaluate them individually across all filter types under SD 1.4 on NSFW-200 in the one-time attack setting.

The results in Fig. 10 confirm cross-layer conflict. Metaphor replacement performs relatively well on image-side filters, reaching 65.39% on image-c and 63.31% on image-clip-c, but drops to 53.04% on text-image-c, where the cross-modal classifier captures residual alignment between the rewritten text and the generated image. Chain-of-thought expansion achieves 71.23% on text-m by distributing sensitive content across reasoning steps that keyword-based filters struggle to aggregate, but falls to 55.89% on image-c because the expanded narrative still guides the generative model toward explicit visual output. Scenario abstraction and question-based obfuscation show similarly uneven profiles across filters. No single strategy exceeds 69% across all five filters simultaneously. CRACK’s adaptive selection, by contrast, achieves 74.25% or above on all five filters, with a peak of 93.61% on text-c. The consistent margin over any individual strategy demonstrates that effective evasion under multi-layer defenses requires dynamically matching the transformation strategy to the modality and detection mechanism of each active filter, rather than committing to one fixed evasion direction throughout the attack.

Sparsity. We measure how sparse the feasible region becomes as defense layers are added. We generate 10,000 random paraphrases of NSFW-200 prompts via DeepSeek V3 nucleus sampling and count how many simultaneously pass all active filters while maintaining CLIPScore above 0.26.

The hit rate decays multiplicatively as layers are added, closely matching the exponential bound predicted by (24). This explains the query costs observed in RQ1. SneakyPrompt needs 39.96 queries, and JailFuzzer needs 29.55 queries, because they rely on partially directed yet largely exploratory strategies. CRACK achieves high ASR in 6 queries because the defense agent identifies which layer rejected the prompt, allowing the attack agent to mutate in a targeted direction rather than wasting queries in infeasible regions.

![](images/904f2a82584285b5adf0f66963ec20bba2f9518a371c750e219e97b985aca750.jpg)  
Fig. 10. ASR-NudeNet(%) of each strategy across different filter types.

TABLE III  
EFFECT OF DEFENSE COMPOSITION ON RANDOM HIT RATE
<table><tr><td>Defense Configuration</td><td>Layers</td><td>Random Hit Rate (%)</td></tr><tr><td>text modality only</td><td>1</td><td>12.3</td></tr><tr><td> $\mathrm { t e x t } + \mathrm { i m a g e }$ </td><td>2</td><td>2.7</td></tr><tr><td> $\mathrm { t e x t } + \mathrm { i m a g e }$  + cross-modal</td><td>3</td><td>0.4</td></tr></table>

Non-convexity. We test whether the feasible region forms a convex set by examining straight-line paths between known feasible points. We select 50 pairs of successful adversarial prompts $( p _ { 1 } , p _ { 2 } )$ from each dataset, compute their CLIP text embeddings, perform linear interpolation $e ( \lambda ) \ = \ ( 1 - \lambda )$ $e ( p _ { 1 } ) + \lambda \cdot e ( p _ { 2 } )$ at $\lambda \in \{ 0 . 1 , 0 . 2 , \ldots , 0 . 9 \}$ , retrieve the nearest prompt in vocabulary space for each interpolated embedding, and evaluate whether it passes all filters with CLIPScore above 0.26.

As shown in Tab IV, only 23.5% of midpoint interpolations survive on NSFW-200. The majority of straight-line paths between two feasible prompts cross detection boundaries, confirming that the feasible region is non-convex and consists of disjoint or irregularly shaped components. This explains why greedy single-direction methods get trapped at local dead ends. They move along one trajectory until blocked and cannot navigate around non-convex boundaries. CRACK’s multi-round debate with strategy switching allows the search trajectory to change direction across rounds, effectively navigating through prompt space to reach feasible components that are inaccessible via straight-line paths. This also explains why the ablation without RL suffers an ASR drop. Without learned strategy switching, the search cannot adapt its direction when encountering convex barriers.

TABLE IV  
SURVIVAL RATES (%) ACROSS DATASETS UNDER DIFFERENT λ SETTINGS
<table><tr><td>Dataset</td><td>Midpoint  $( \lambda = 0 . 5 )$ </td><td>Average over λ</td></tr><tr><td>NSFW-200</td><td>23.5</td><td>31.2</td></tr><tr><td>UnsafeDiff</td><td>19.2</td><td>28.4</td></tr><tr><td>I2P</td><td>15.6</td><td>25.6</td></tr></table>

3) RQ3: Ablation Study: To evaluate the contribution of each module, we conduct component-wise ablations on SD 1.4 under the one-time attack setting, as summarized in table V.

Effect of multi-agent debate. We first test the attack agent in isolation (attack agent only), where it generates prompt mutations without any feedback. Compared to the full CRACK system, ASR drops substantially across all filters and datasets, while CLIPScore remains low, indicating that blind mutations can occasionally bypass a filter but fail to preserve semantic intent. Removing the defense agent (w/o defense agent) leads to further ASR degradation, particularly on text-image-c where ASR falls to 44.15% on I2P and 30.16% on NSFW-200. Without explicit identification of which filter rejected the prompt, the attack agent cannot direct its mutations toward the actual detection bottleneck. Removing the judge agent (w/o judge agent) preserves moderate ASR but causes CLIPScore to collapse, dropping below 0.12 on multiple dataset-filter pairs. Without quality control, the system generates prompts that evade filters by drifting far from the original meaning, producing images unrelated to the target content. These results confirm that the three agents serve complementary roles and that removing any one of them breaks either the evasion capability or the semantic preservation guarantee.

Effect of RL. Replacing the RL-based strategy selector with a fixed policy (w/o RL) reduces ASR by 10–20% across filters compared to the full system, with the largest drops on textimage-c. Without reward-guided updates, the system cannot learn which strategy works against which filter configuration, falling back to suboptimal choices that waste query budget. This aligns with the cross-layer conflict finding in RQ2, where no fixed strategy performs uniformly well.

Effect of strategy diversity. Restricting the strategy library to a single strategy (metaphor replacement) produces the lowest overall ASR among all ablations, with performance on text-image-c dropping to 43.12% on I2P. This variant also underperforms the w/o RL ablation, which at least retains access to multiple strategies even without learned selection. The gap confirms that strategy diversity is not redundant with RL. Even a perfect selector cannot compensate if the underlying strategy set lacks coverage across filter modalities. 4) RQ4: Sensitivity Study: We analyze the sensitivity of CRACK to three key factors, including the number of debate rounds, the weighting coefficients α and β in Eq. (28), and the choice of LLM backbone for each agent. A summary of key results is provided below, with full details provided in the Supplementary.

TABLE V  
ABLATION STUDY ON STABLE DIFFUSION 1.4 UNDER THE ONE-TIME PROMPT ATTACK SETTING.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Safety Filter</td><td colspan="2">ASR-Q16 (%)</td><td>ASR-NudeNet (%)</td><td colspan="3">CLIPScore</td></tr><tr><td>I2P</td><td>UnsafeDiff</td><td>NSFW-200</td><td>I2P</td><td>UnsafeDiff</td><td>NSFW-200</td></tr><tr><td rowspan="3">attack agent only</td><td>text-image-c</td><td>60.12</td><td>59.02</td><td>55.12</td><td>0.19</td><td>0.33</td><td>0.21</td></tr><tr><td>text-m</td><td>83.10</td><td>85.56</td><td>80.16</td><td>0.16</td><td>0.35</td><td>0.12</td></tr><tr><td>image-c</td><td>55.12</td><td>49.21</td><td>49.16</td><td>0.20</td><td>0.39</td><td>0.19</td></tr><tr><td rowspan="3">w/o defense agent</td><td>text-image-c</td><td>44.15</td><td>37.63</td><td>30.16</td><td>0.23</td><td>0.31</td><td>0.11</td></tr><tr><td>text-m</td><td>70.03</td><td>71.03</td><td>63.97</td><td>0.25</td><td>0.26</td><td>0.23</td></tr><tr><td>image-c</td><td>61.33</td><td>63.43</td><td>50.04</td><td>0.21</td><td>0.29</td><td>0.10</td></tr><tr><td rowspan="3">w/o judge agent</td><td>text-image-c</td><td>61.13</td><td>51.02</td><td>55.63</td><td>0.10</td><td>0.31</td><td>0.11</td></tr><tr><td>text-m</td><td>78.06</td><td>75.21</td><td>73.16</td><td>0.10</td><td>0.16</td><td>0.06</td></tr><tr><td>image-c</td><td>72.13</td><td>78.01</td><td>70.54</td><td>0.11</td><td>0.34</td><td>0.12</td></tr><tr><td rowspan="3">w/o RL</td><td>text-image-c</td><td>54.65</td><td>51.01</td><td>50.12</td><td>0.22</td><td>0.29</td><td>0.13</td></tr><tr><td>text-m</td><td>69.16</td><td>72.01</td><td>66.13</td><td>0.21</td><td>0.31</td><td>0.22</td></tr><tr><td>image-c</td><td>70.01</td><td>65.01</td><td>63.28</td><td>0.11</td><td>0.21</td><td>0.15</td></tr><tr><td rowspan="3">only one strategy (metaphor replacement)</td><td>text-image-c</td><td>43.12</td><td>48.78</td><td>51.03</td><td>0.20</td><td>0.30</td><td>0.16</td></tr><tr><td>text-m</td><td>62.13</td><td>63.31</td><td>61.23</td><td>0.23</td><td>0.26</td><td>0.15</td></tr><tr><td>image-c</td><td>56.45</td><td>57.01</td><td>61.03</td><td>0.21</td><td>0.23</td><td>0.19</td></tr></table>

![](images/93e8abc33e2a78b458db00bdfcecef33bc38388c800b52d558633b6a2c99b49c.jpg)

![](images/a04fb5906a5123561bab267c352fbfef8c0b9d62c967cac226c29d10755fdce0.jpg)  
Fig. 11. (a) Impact of debate round count on ASR and CLIPScore. (b) Performance variation with different values of α and β in the reward function. We conducted experiments on SD 1.4 and evaluated performance under the text-image-c safety filter of the NSFW-200 dataset.

Debate rounds. As shown in Fig. 11a, the ASR reaches highest by round 3 and shows no further improvement, while the CLIPScore peaks at the same round before declining due to semantic drift. Thus, three rounds achieve the best balance between attack effectiveness and fidelity.

Weighting factors α and β. The trade-off between bypass success and semantic fidelity is mainly governed by these weights. Increasing α, which emphasizes LLM-based risk assessment, reduces detectable harmful intent and raises bypass rates but also decreases the CLIPScore. Conversely, higher β emphasizes CLIPScore, enhancing image-text alignment at the cost of slightly lower bypass success. A balanced configuration of $\alpha = 0 . 3$ and $\beta = 0 . 7$ achieves the optimal compromise between bypass success and quality, as shown in Fig. 11b.

LLM backbones. CRACK demonstrates strong robustness across different LLM backbones, with only marginal performance variation (see Supplementary Sec. G). CRACK consistently maintains stable performance without significant degradation compared to the results obtained with DeepSeek V3. This stability arises because the debate and reward mechanisms are largely model-agnostic, decoupling task reasoning from specific language patterns.

5) RQ5: Extended Evaluations: To further validate the comprehensiveness of our model, we evaluate FID Scores [52] under the same settings as [17], [8], measuring distributionlevel similarity between generated and real images. Following prior work, FID is computed against two ground-truth datasets [53]. Since they contain only real adult images, experiments are conducted on NSFW-200, where CRACK achieves consistently lower FID than all baselines. To verify generalization without producing NSFW content, we also follow [17], [8] to test on Dog/Cat-100. By replacing animal entities with harmful targets, CRACK generates coherent dog and cat images without explicit mentions, demonstrating strong semantic control and adaptability (see Supplementary Sec. H).

## VI. CONCLUSION

This paper presents CRACK, a multi-agent debate framework for jailbreaking heterogeneous safety stacks in T2I systems. To characterize the challenges posed by composed defenses, we formulated the Detection Surface and identified three structural properties of the joint evasion region, namely cross-layer conflict, sparsity, and non-convexity. CRACK addresses these properties through an iterative debate process among three specialized agents that combines layer-wise diagnosis, adaptive strategy selection, and reward-guided iterative refinement for layer-aware navigation of the Detection Surface. Experiments across four T2I models demonstrate improved attack effectiveness and query efficiency under composed defenses while preserving semantic fidelity.

Our results carry a concrete defensive message. Stacking heterogeneous filters does not by itself close the feasible region, since the cracks between layers remain exploitable precisely because the layers are optimized independently. Future robust defense should treat the stack as a joint decision surface rather than separate gates, co-design filters so their safe regions overlap more tightly, and audit stacks against directed rather than random search. Our future work will include extending the surrogate to unknown or adaptive commercial defenses and applying the Detection Surface formalism to other compositional safety pipelines.

## REFERENCES

[1] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer, “Highresolution image synthesis with latent diffusion models,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2022, pp. 10 684–10 695.

[2] D. Podell, Z. English, K. Lacey, A. Blattmann, T. Dockhorn, J. Muller, J. Penna, and R. Rombach, “Sdxl: Improving latent diffusion¨ models for high-resolution image synthesis,” 2023. [Online]. Available: https://arxiv.org/abs/2307.01952

[3] Z. Shi, X. Zhou, X. Qiu, and X. Zhu, “Improving image captioning with better use of captions,” 2020. [Online]. Available: https://arxiv.org/abs/2006.11807

[4] Midjourney.com, “Midjourney,” https://midjourney.com/, 2022.

[5] N. Ahn, J. Lee, C. Lee, K. Kim, D. Kim, S.-H. Nam, and K. Hong, “Dreamstyler: Paint by style inversion with text-to-image diffusion models,” 2023. [Online]. Available: https://arxiv.org/abs/2309.06933

[6] T. Wang and M. Ye, “Texfit: text-driven fashion image editing with diffusion models,” in Proceedings of the Thirty-Eighth AAAI Conference on Artificial Intelligence and Thirty-Sixth Conference on Innovative Applications of Artificial Intelligence and Fourteenth Symposium on Educational Advances in Artificial Intelligence, ser. AAAI’24/IAAI’24/EAAI’24. AAAI Press, 2024. [Online]. Available: https://doi.org/10.1609/aaai.v38i9.28885

[7] P. Cao, F. Zhou, Q. Song, and L. Yang, “Controllable generation with text-to-image diffusion models: A survey,” 2024. [Online]. Available: https://arxiv.org/abs/2403.04279

[8] Y. Yang, B. Hui, H. Yuan, N. Gong, and Y. Cao, “Sneakyprompt: Jailbreaking text-to-image generative models,” in Proceedings of the IEEE Symposium on Security and Privacy, 2024.

[9] C. Y. Hsu, Y. L. Tsai, C. Xie, C. H. Lin, J. Y. Chen, B. Li, P. Y. Chen, C. M. Yu, and C. Y. Huang, “Ring-a-bell! how reliable are concept removal methods for diffusion models?” in 12th International Conference on Learning Representations, ICLR 2024, 2024.

[10] J. Rando, D. Paleka, D. Lindner, L. Heim, and F. Tramer, “Red-\` teaming the stable diffusion safety filter,” 2022. [Online]. Available: https://arxiv.org/abs/2210.04610

[11] OpenAI, “Dall-e 3 system card,” https://openai.com/index/ dall-e-3-system-card/, 2023.

[12] Y. Deng and H. Chen, “Harnessing llm to attack llm-guarded text-to-image models,” 2024. [Online]. Available: https://arxiv.org/abs/ 2312.07130

[13] P. Schramowski, M. Brack, B. Deiseroth, and K. Kersting, “Safe latent diffusion: Mitigating inappropriate degeneration in diffusion models,” 2023. [Online]. Available: https://arxiv.org/abs/2211.05105

[14] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, G. Krueger, and I. Sutskever, “Learning transferable visual models from natural language supervision,” 2021. [Online]. Available: https://arxiv.org/abs/2103.00020

[15] J. Hessel, A. Holtzman, M. Forbes, R. L. Bras, and Y. Choi, “Clipscore: A reference-free evaluation metric for image captioning,” 2022. [Online]. Available: https://arxiv.org/abs/2104.08718

[16] Y. Huang, L. Liang, T. Li, X. Jia, R. Wang, W. Miao, G. Pu, and Y. Liu, “Perception-guided jailbreak against text-to-image models,” 2025. [Online]. Available: https://arxiv.org/abs/2408.10848

[17] Y. Dong, X. Meng, N. Yu, Z. Li, and S. Guo, “Fuzz-testing meets llm-based agents: An automated and efficient framework for jailbreaking text-to-image generation models,” 2025. [Online]. Available: https://arxiv.org/abs/2408.00523

[18] Z. Chen, H. Lin, K. Xu, X. Jiang, and T. Sun, “Dynamic optimization and safety indicator injection for jailbreaking text-toimage models with multimodal safety filters,” 2026. [Online]. Available: https://arxiv.org/abs/2505.18979

[19] Y. Yang, R. Gao, X. Wang, T.-Y. Ho, N. Xu, and Q. xu, “Mma-diffusion: Multimodal attack on diffusion models,” in 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024, pp. 7737– 7746.

[20] Y. Deng and H. Chen, “Harnessing llm to attack llm-guarded text-to-image models,” 2024. [Online]. Available: https://arxiv.org/abs/ 2312.07130

[21] Z.-Y. Chin, C.-M. Jiang, C.-C. Huang, P.-Y. Chen, and W.-C. Chiu, “Prompting4debugging: Red-teaming text-to-image diffusion models by finding problematic prompts,” in International Conference on Machine Learning (ICML), 2024. [Online]. Available: https: //arxiv.org/abs/2309.06135

[22] H. Zheng, Y. He, T. Chen, S. Shao, Z. Chu, H. Zhou, L. Tao, Z. Qin, and K. Ren, “Janus: A lightweight framework for jailbreaking text-to-image models via distribution optimization,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2026, pp. 15 719–15 729.

[23] X. Ye, Y. Liu, L. Wang, R. Wang, G. Yang, Y. Hou, and J. Yu, “Macprompt: Maraconic-guided jailbreak against text-to-image models,” 2026. [Online]. Available: https://arxiv.org/abs/2601.07141

[24] R. George, “NSFW words list,” https://github.com/ rrgeorge-pdcontributions/NSFW-Words-List, 2020.

[25] R. Gandikota, J. Materzynska, T. Fiotto-Kaufman, and D. Bau, “Erasing concepts from diffusion models,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023.

[26] LAION-AI, “CLIP-based NSFW detector,” https://github.com/ LAION-AI/CLIP-based-NSFW-Detector, 2023.

[27] notAI.tech, “Nudenet,” https://github.com/notAI-tech/NudeNet, accessed: 2026-07-15.

[28] P. Schramowski, C. Tauchmann, and K. Kersting, “Can machines help us answering question 16 in datasheets, and in turn reflecting on inappropriate content?” Proceedings of the 2022 ACM Conference on Fairness, Accountability, and Transparency, 2022. [Online]. Available: https://api.semanticscholar.org/CorpusID:246823108

[29] Y. Du, S. Li, A. Torralba, J. B. Tenenbaum, and I. Mordatch, “Improving factuality and reasoning in language models through multiagent debate,” in Proceedings of the 41st International Conference on Machine Learning, ser. Proceedings of Machine Learning Research,

vol. 235. PMLR, 21–27 Jul 2024, pp. 11 733–11 763. [Online]. Available: https://proceedings.mlr.press/v235/du24e.html

[30] T. Liang, Z. He, W. Jiao, X. Wang, Y. Wang, R. Wang, Y. Yang, S. Shi, and Z. Tu, “Encouraging divergent thinking in large language models through multi-agent debate,” in Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing. Miami, Florida, USA: Association for Computational Linguistics, Nov. 2024, pp. 17 889–17 904. [Online]. Available: https://aclanthology.org/2024.emnlp-main.992/

[31] S. Yue, S. Wang, W. Chen, X. Huang, and Z. Wei, “Synergistic multi-agent framework with trajectory learning for knowledge-intensive tasks,” 2025. [Online]. Available: https://arxiv.org/abs/2407.09893

[32] A. Smit, N. Grinsztajn, P. Duckworth, T. D. Barrett, and A. Pretorius, “Should we be going mad? a look at multi-agent debate strategies for llms,” in Proceedings of the 41st International Conference on Machine Learning, ser. ICML’24. JMLR.org, 2024.

[33] P. Chao, A. Robey, E. Dobriban, H. Hassani, G. J. Pappas, and E. Wong, “Jailbreaking black box large language models in twenty queries,” 2023.

[34] A. Mehrotra, M. Zampetakis, P. Kassianik, B. Nelson, H. Anderson, Y. Singer, and A. Karbasi, “Tree of attacks: Jailbreaking black-box llms automatically,” 2024. [Online]. Available: https://arxiv.org/abs/ 2312.02119

[35] M. Samvelyan, S. C. Raparthy, A. Lupu, E. Hambro, A. H. Markosyan, M. Bhatt, Y. Mao, M. Jiang, J. Parker-Holder, J. Foerster, T. Rocktaschel, and R. Raileanu, “Rainbow teaming: Open-ended¨ generation of diverse adversarial prompts,” 2024. [Online]. Available: https://arxiv.org/abs/2402.16822

[36] A. Zhou, K. Wu, F. Pinto, Z. Chen, Y. Zeng, Y. Yang, S. Yang, S. Koyejo, J. Zou, and B. Li, “Autoredteamer: Autonomous red teaming with lifelong attack integration,” 2025. [Online]. Available: https://arxiv.org/abs/2503.15754

[37] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, “Proximal policy optimization algorithms,” arXiv preprint arXiv:1707.06347, 2017.

[38] E. Perez, S. Huang, F. Song, T. Cai, R. Ring, J. Aslanides, A. Glaese, N. McAleese, and G. Irving, “Red teaming language models with language models,” in Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, 2022.

[39] G. Montufar, R. Pascanu, K. Cho, and Y. Bengio, “On the number of´ linear regions of deep neural networks,” in Proceedings of the 28th International Conference on Neural Information Processing Systems - Volume 2, ser. NIPS’14. Cambridge, MA, USA: MIT Press, 2014, p. 2924–2932.

[40] W. He, B. Li, and D. Song, “Decision boundary analysis of adversarial examples,” in International Conference on Learning Representations, 2018. [Online]. Available: https://openreview.net/forum?id=BkpiPMbA-

[41] R. George, “Nsfw-words-list,” https://github.com/ rrgeorge-pdcontributions/NSFW-Words-List/blob/master/nsfw list.txt, 2020.

[42] M. Han, L. Zhang, J. Wang, and W. Pan, “Actor-critic reinforcement learning for control with stability guarantee,” IEEE Robotics and Automation Letters, vol. 5, no. 4, pp. 6217–6224, 2020.

[43] P. Schramowski, M. Brack, B. Deiseroth, and K. Kersting, “Safe latent diffusion: Mitigating inappropriate degeneration in diffusion models,” 2023. [Online]. Available: https://arxiv.org/abs/2211.05105

[44] Y. Qu, X. Shen, X. He, M. Backes, S. Zannettou, and Y. Zhang, “Unsafe diffusion: On the generation of unsafe images and hateful memes from text-to-image models,” in Proceedings of the 2023 ACM SIGSAC Conference on Computer and Communications Security, ser. CCS ’23. New York, NY, USA: Association for Computing Machinery, 2023, p. 3403–3417. [Online]. Available: https://doi.org/10. 1145/3576915.3616679

[45] M. Li, “Nsfw text classifier on hugging face,” https://huggingface.co/ michellejieli/NSFWtextclassifier, 2022.

[46] L. Chhabra, “Nsfw image classifer on github,” https://github.com/ lakshaychhabra/NSFW-Detection-DL, 2020.

[47] LAION-AI, “Nsfw clip based image classifier on github,” https://github. com/LAION-AI/CLIP-based-NSFW-Detector, 2023.

[48] P. Bedapudi, “NudeNet: Neural nets for nudity classification, detection and selective censoring,” https://github.com/notAI-tech/NudeNet, 2019.

[49] DeepSeek-AI, A. Liu, B. Feng, B. Xue, B. Wang, B. Wu, C. Lu, C. Zhao, C. Deng, and et al, “Deepseek-v3 technical report,” 2025. [Online]. Available: https://arxiv.org/abs/2412.19437

[50] A. Yang, A. Li, B. Yang, B. Zhang, B. Hui, B. Zheng, B. Yu, C. Gao, C. Huang, and et al, “Qwen3 technical report,” 2025. [Online]. Available: https://arxiv.org/abs/2505.09388

[51] OpenAI, J. Achiam, S. Adler, S. Agarwal, L. Ahmad, I. Akkaya, F. L. Aleman, and et al, “Gpt-4 technical report,” 2024. [Online]. Available: https://arxiv.org/abs/2303.08774

[52] G. Parmar, R. Zhang, and J.-Y. Zhu, “On aliased resizing and surprising subtleties in gan evaluation,” 2022.

[53] A. Kim, “Nsfw image dataset,” https://github.com/alex000kim/nsfw data scraper, 2022.