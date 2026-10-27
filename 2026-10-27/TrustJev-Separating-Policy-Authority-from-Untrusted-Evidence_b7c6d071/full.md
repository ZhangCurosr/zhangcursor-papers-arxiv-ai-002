# TrustJev: Separating Policy Authority from Untrusted Evidence in LLM-Based Decisions

Zhiwei Xu<sup>†</sup>, Xuyang Liu<sup>†</sup>, Xiaolin Peng<sup>†</sup>, Weixian Deng<sup>†</sup>, Jiabao Gao<sup>†</sup>, Zirui Li<sup>†</sup>, Tian Qiu<sup>†</sup>, Boya Zhou<sup>‡</sup>, Wei Jiao<sup>‡</sup>, Hai Wan<sup>†</sup>, Xibin Zhao<sup>†</sup>

<sup>†</sup>KLISS, BNRist, School of Software, Tsinghua University, Beijing, China

<sup>‡</sup>ChinaBond Finance and Information Technology Co., Ltd., Beijing, China

{xzw24, liuxuyan23, pxl25, deng-wx23, gjb25, li-zr22, qt24}@mails.tsinghua.edu.cn

zhouby@chinabond.com.cn, jiaowei@ccdc.com.cn

{wanhai, zxb}@tsinghua.edu.cn

Abstract—Decision models enable software to select from predefined actions using natural language inputs. However, the external evidence that supplies facts for these decisions may also contain instructions that attempt to override the application’s policy. We propose TrustJev, a framework combining protected policy scoring with bounded evidence corrections. A single Transformer backbone computes policy scores and a correction budget from trusted policy and candidates, then bounds evidence corrections by that budget. Paired supervision encourages stable predictions under instruction insertions that preserve task facts and changed preferences when edits change the correct answer. Authority supervision accommodates evidence when the policy alone cannot determine the decision. We evaluate a 0.6B implementation against 14 baselines on three datasets covering authentication, instruction hierarchy, and executable authorization rules. TrustJev achieves the highest mean Balance, the harmonic mean of accuracy on non-attack and attack inputs, exceeding the strongest baseline on each by 0.57–1.05 percentage points. In the authorization benchmark, it correctly answers all 32 attacks per source in a fixed pool for 95.77% of sources, compared with 80.98% for the strongest comparator. TrustJev maintains comparable inference memory and median latency, with additional overhead primarily during training.

Index Terms—Prompt injection, decision models, policy compliance, software workflows.

## I. INTRODUCTION

Software workflows often include decision steps that determine what action to take based on information from user requests, reports, or other external sources. Implementing these decisions can require substantial engineering effort, particularly when the relevant facts and policy conditions are expressed in natural language [1], [2]. The emergence of Jev offers an alternative to handcrafting this logic, producing typed decisions and probabilities from natural language instructions, external inputs, and application-supplied options [1]. Jev reports lower latency and cost than generative models in its workflow evaluations, while Kev provides an open, compact implementation with single-pass candidate scoring [3]. Beyond these efficiency benefits, predefined outputs prevent the model from inventing actions outside the supplied set.

Their usefulness, however, depends on how well they follow the application’s policy while drawing on external inputs that may also contain conflicting instructions. Prompt injection studies show that instructions embedded in attacker-controlled content can redirect an application’s intended task [4]. For decision models, this means steering the choice toward an option that the policy does not permit. Experiments on Jev demonstrate this behavior, with malicious inputs shifting candidate probabilities and sometimes inducing the attacker’s preferred selection [5]. Thus, although outputs remain within the supplied set, the selected action can still violate the application’s policy.

Consider a common customer-service workflow, illustrated by the compensation example in Figure 1. An administrator permits one-time service-fee refunds of at most \$5 and requires manual review for larger refunds or other forms of compensation. The model reads a customer’s email and selects Approve or Manual Review. A \$10 refund requires manual review, while a \$3 refund qualifies for approval. A \$200 request still requires manual review even when the email contains an instruction to skip the checks and approve it. These cases highlight the difficulty of responding to changes in relevant facts while resisting instructions that conflict with the application’s policy.

To address this difficulty, prior work on generative language models has sought to separate trusted instructions from external data. StruQ and SecAlign strengthen this separation through structured defensive training and response preference optimization, respectively [6], [7]. DRIP intervenes in the computation by editing data representations and reinforcing instruction representations [8]. These approaches strengthen instruction adherence or intervene in representations, but do not explicitly bound how much external evidence can shift relative candidate scores. For decision models such as Jev, this distinction matters because an attack can succeed by changing the winning candidate without producing any freeform malicious response [5]. This leaves a gap between separating instructions from data and ensuring that the selected action follows the policy.

We propose TrustJev, guided by the observation that policy and evidence have different responsibilities even when both are expressed in natural language. The policy defines how to decide, while evidence supplies the facts needed to apply it. TrustJev therefore scores each candidate from the trusted policy without using external evidence, then adds a correction derived from the evidence to obtain the final score. The trusted policy and candidates also determine a budget that limits the magnitude of this correction, so evidence can inform the decision without controlling how much influence it receives.

![](images/b93959c04e532842ea013c1d531bcc208a1ce7eca000b36613f0d5fcbfac8239.jpg)  
Fig. 1. A compensation workflow in which customer emails supply decision facts but have no authority to change the administrator’s policy. The selected options show the expected policy-compliant decisions, including for requests containing injected instructions.

Bounding evidence influence, however, does not teach the model when a decision should change. Returning to the compensation example in Figure 1, the model needs the amount requested in the customer’s email to apply the \$5 refund limit. A strong preference formed without that fact could prevent even relevant evidence from changing the decision. TrustJev therefore uses authority supervision to train policy scores to favor a candidate when the policy suffices and avoid strong policy preferences when facts are needed. Paired supervision then distinguishes instruction insertions that preserve the correct answer from fact or policy changes that alter it. The resulting model scores all candidates in a single Transformer [9] forward pass at deployment.

We evaluate whether this design improves injection resistance while retaining accurate decisions on legitimate inputs. We compare a 0.6B implementation with 14 baselines on authentication tasks adapted from TensorTrust [10], instruction hierarchy tasks adapted from IHEval [11], and our AuthorityEvidenceBench (AEB). AEB uses executable rules to test injection resistance and responsiveness to fact and policy changes. TrustJev achieves the highest mean Balance, the harmonic mean of accuracy on non-attack and attack inputs, on all three datasets, exceeding the strongest baseline on each by 0.57–1.05 percentage points. In a separate AEB experiment with a fixed pool of 32 attacks per source, it answers every attack correctly for 95.77% of sources, compared with 80.98% for fine-tuned Kev, the strongest comparator.

This paper makes four contributions.

• A scoring architecture separating protected policy scores from bounded evidence corrections, with a sufficient condition for preserving the policy-preferred candidate.

• A training method that coordinates resistance to injected instructions with responsiveness to relevant facts through paired decision supervision and authority supervision.

• An empirical study across three datasets, including a benchmark with executable rules, component ablations, and evaluations of attack variants, composition, and multiple attempts.

• A framework for practitioners to train trustworthy LLMbased decision models for software workflows, with code available at https://github.com/xu-zhiwei/trustjev.

## II. BACKGROUND AND PROBLEM FORMULATION

## A. Application Context

Language-based decision models can support software workflows that apply policies to select among predefined options [1]. Inputs can range from structured fields evaluated by explicit rules to unstructured reports and requests [12], semistructured logs [13]–[15], and tool results combining fields with free text [2], [16]. When facts are expressed in varied natural language, semantic interpretation can reduce the effort of maintaining handwritten rules to cover these variations across reports and sources [1].

One application we are exploring is log-alert classification and routing, using unstructured reports and semistructured logs to assign severity and select a responsible team. Automation could reduce manual triage and focus scarce expert attention on alerts that warrant it [17]. However, submitting units could insert instructions to downplay serious incidents or escalate routine alerts for priority support. Such manipulation could divert expert resources, so decisions need to follow organizational rules while drawing on submitted facts [6]. In this setting, organizational rules define the trusted policy, submitted information provides external evidence, and severity labels or routing destinations form the candidate options.

## B. Decision Model and Requirements

We formulate the decision problem and requirements for external evidence under a trusted policy. Each input is $x =$ $( P , D , C )$ , where P is the trusted policy, D is external evidence, and $C = ( c _ { 1 } , \dots , c _ { K } )$ lists the candidates. A scorer produces $z = f _ { \theta } ( P , D , C ) \in \mathbb { R } ^ { K }$ , and a fixed decision rule selects candidate $c _ { \hat { y } }$ . Let y denote the correct candidate index under the policy and relevant facts. An injected instruction preserving those facts should preserve the decision, while a fact edit or authorized policy update that changes $y$ should change the prediction. We require policy authority, so evidence cannot redefine decision criteria, and evidence responsiveness, so relevant facts can determine the answer.

## C. Threat Model

The attacker controls text entering the evidence channel and seeks a selection that violates the application policy [5], [6]. The attacker may know the policy and candidates but cannot modify them, the model parameters, the decision rule, or the application-assigned source boundaries. Instructions claiming administrative authority within $D$ remain evidence. For injection variants that preserve relevant facts, an incorrect selection $\hat { y } ~ \neq ~ y$ constitutes an attack success. Authorized policy updates and fact changes that alter the correct answer are legitimate task variations.

## III. TRUSTJEV

Figure 2 presents the architecture and training objectives of TrustJev. The design combines protected policy scores with bounded evidence corrections, while paired and authority supervision coordinate decision stability and responsiveness. Both scoring branches share a Transformer backbone, allowing all candidates to be scored in one forward pass at deployment.

## A. Isolated Policy Authority

The policy scores and the evidence budget need a computational path that does not read external evidence. Otherwise, an injected instruction could change both the decision preference and the limit intended to constrain it. TrustJev establishes this path through application-controlled input boundaries, attention permissions, and position assignments.

Source-preserving input. To compute representations of each candidate with and without access to external evidence, TrustJev includes the text of candidate $c _ { k }$ twice in the input sequence. We denote these two occurrences of the same text by $C _ { k } ^ { A }$ and $C _ { k } ^ { D }$ . Both can read the policy and their own text, while $C _ { k } ^ { D }$ can additionally read the evidence. The physical input sequence places $P$ and all $C _ { k } ^ { A }$ before $D$ and all $C _ { k } ^ { D }$

$$
P , \ C _ { 1 } ^ { A } , \ldots , C _ { K } ^ { A } , \ D , \ C _ { 1 } ^ { D } , \ldots , C _ { K } ^ { D } .\tag{1}
$$

Both occurrences are processed by the same Transformer backbone with shared parameters. The application assigns the source boundaries and escapes textual spellings of control markers in supplied content, so a marker embedded in an external document is not treated as an application-created boundary. Trust here denotes authority to specify a rule.

Attention and positions. Table I specifies which parts each query can read. Within an allowed part, attention respects the causal order. Policy tokens read the preceding policy context. Tokens in $C _ { k } ^ { A }$ can read $P$ and preceding tokens within $C _ { k } ^ { A }$ while tokens in $C _ { k } ^ { D }$ can additionally read $D .$ Evidence tokens can read $P$ and preceding evidence. Neither occurrence of a candidate can read any other candidate occurrence. Applied at every layer, these permissions block indirect evidence access through intermediate tokens.

TABLE I  
ATTENTION PERMISSIONS IN TRUSTJEV.
<table><tr><td>Query tokens</td><td>Readable parts</td></tr><tr><td>Trusted policy  $P$ </td><td> $P$ </td></tr><tr><td>Candidate text  $C _ { k } ^ { A }$ </td><td> $P , C _ { k } ^ { A }$ </td></tr><tr><td>External evidence  $D$ </td><td> $P , { D } ^ { \hat { ( } \hat { \mathbf { \phi } } ) }$ </td></tr><tr><td>Candidate text  $C _ { k } ^ { D }$ </td><td> $P , D , C _ { k } ^ { D }$ </td></tr></table>

Position assignments follow these logical contexts rather than the concatenated sequence indices. The position indices for each $C _ { k } ^ { A }$ start after the policy, and those for each $C _ { k } ^ { D }$ start after the policy and evidence. Thus, changing evidence length does not shift the position indices of $C _ { k } ^ { A }$ . Together, the mask and position rules ensure that the policy scores a and evidence budget $\beta$ depend on $P$ and $C ,$ independently of external evidence D.

Policy scores and evidence budget. The backbone encodes $P$ and $C _ { k } ^ { A }$ without access to evidence, and their representations are projected to $p$ and $u _ { k }$ , respectively. The figure labels these policy and candidate features collectively as $h ^ { P }$ . The Policy Scoring module uses a trainable readout A to compute

$$
a _ { k } = A ( [ p , u _ { k } , p \odot u _ { k } ] ) ,\tag{2}
$$

where brackets denote concatenation and $\odot$ denotes elementwise multiplication. These policy scores represent learned preferences under the trusted rule. The Evidence Budget module averages the candidate representations and uses a second readout $G$ to produce a nonnegative budget

$$
\begin{array} { l } { \displaystyle \bar { u } = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } { u _ { k } } , } \\ { \displaystyle \beta = B ( K ) \sigma ( G ( [ p , \bar { u } ] ) ) . } \end{array}\tag{3}
$$

Here, $\sigma$ is the sigmoid function and $B ( K ) > 0$ is a preset budget ceiling. The model learns a request-specific budget shared by all candidates. Both a and $\beta$ depend on trusted inputs, so an evidence edit cannot directly enlarge the permitted correction. For a policy that requires external facts, the policy scores can leave the choice unresolved until evidence is considered.

## B. Bounded Evidence Contribution

Evidence scoring computes a correction for each candidate from external evidence. TrustJev bounds these corrections by the policy-derived budget before adding them to policy scores.

Evidence for each candidate. The representation u<sub>k</sub> queries the evidence token states to obtain an aggregated representation $d _ { k } ^ { \mathrm { p o o l } }$ . The representation obtained from $C _ { k } ^ { D }$ denoted by $d _ { k } ^ { \mathrm { c t x } }$ , encodes the same candidate with access to both policy and evidence. The deployed model combines them as

![](images/d55b0e557d45392b304b50cfce1251ce438d4b2296f73b60c1c82a87786eb00b.jpg)  
Fig. 2. Overview of TrustJev. (a) A single Transformer backbone supports both scoring branches. Attention constraints prevent external evidence from altering policy representations, and bounded evidence corrections adjust policy scores. The correction budget depends on the trusted policy and candidates. (b) Paired supervision acts on final scores and encourages stable decisions under instruction insertions that preserve task facts and changed preferences when edits change the correct decision. Authority supervision acts on policy scores and encourages clear policy preferences when the policy suffices and leaves room for evidence when task facts are needed. The three refund inputs use the same policy and scorer parameters. Their labels indicate the expected decisions. Solid arrows denote forward computation, and dashed arrows denote parameter learning.

$$
\begin{array} { r } { d _ { k } = \frac 1 2 d _ { k } ^ { \mathrm { p o o l } } + \frac 1 2 d _ { k } ^ { \mathrm { c t x } } . } \end{array}\tag{4}
$$

Together, the representations $d _ { k }$ form the evidence features $h ^ { E }$ in Figure 2, combining evidence aggregation with contextual encoding. The Evidence Scoring module combines these features with $p$ and $u _ { k }$ , supplied by the Context connection from $h ^ { P }$ in the figure, using a GELU activation [18],

$$
\begin{array} { r l } & { { t _ { k } } = W _ { e } [ p , { u _ { k } } , { d _ { k } } ] + b _ { e } + W _ { \times } ( p \odot { d _ { k } } ) , } \\ & { { e _ { k } } = { v _ { e } ^ { \top } } \operatorname { G E L U } ( { t _ { k } } ) + b _ { e } ^ { \prime } . } \end{array}\tag{5}
$$

The trainable product interaction helps relate evidence to the policy under which it is interpreted. It enters the evidence score, so its contribution is subject to the bound on the resulting correction. Equation (5) describes inference and omits training-time dropout [19].

Centering and bounding. For the K valid candidates, TrustJev computes

$$
\begin{array} { l } { \displaystyle { \bar { e } = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } e _ { k } , } } \\ { \displaystyle { r _ { k } = \beta \operatorname { t a n h } \bigl ( e _ { k } - \bar { e } \bigr ) , } } \\ { \displaystyle { z _ { k } = a _ { k } + r _ { k } . } } \end{array}\tag{6}
$$

Padding candidates are excluded from the mean. Centering removes a common shift in the evidence scores before the nonlinear transformation, avoiding a shared offset that could push all candidates towards the same side of saturation. The subsequent transformation bounds each correction by $| r _ { k } | \le \beta .$ The centered evidence scores sum to zero, but the transformed corrections need not.

The resulting scores allow relevant facts to change candidate rankings when the policy scores leave sufficient room. In the refund example, changing a request from \$200 to \$3 changes which candidate the evidence should support. Appending an instruction to approve the \$200 request leaves the task facts unchanged and should preserve the decision. The architecture provides a bounded path for both inputs, while training distinguishes their required effects.

Decision and stability. For the general decision tasks, TrustJev returns $c _ { \hat { y } }$ with $\hat { y } ~ = ~ \mathrm { a r g } \operatorname* { m a x } _ { k } z _ { k }$ . Authentication instead grants access when $z _ { \mathrm { g r a n t } } - z _ { \mathrm { d e n y } } > \tau$ , using a threshold selected on development data and fixed during evaluation. Both rules return an application-supplied candidate.

The score decomposition provides a sufficient condition for preserving the policy-preferred candidate. Fix the model, trusted inputs, and source boundaries, and let $w =$ arg $\operatorname* { m a x } _ { k } a _ { k }$ with margin $m \ = \ a _ { w } \ - \ \operatorname* { m a x } _ { k \neq w } a _ { k }$ . Since $| r _ { w } - r _ { k } | \le 2 \beta$ , for each $k \neq w ,$

$$
z _ { w } - z _ { k } \ge a _ { w } - a _ { k } - 2 \beta \ge m - 2 \beta .\tag{7}
$$

Under the isolated computation and exact arithmetic, $m > 2 \beta$ therefore preserves w for evidence changes that leave the trusted policy and candidates fixed. For authentication, the corresponding sufficient condition is that the policy score difference lies more than 2β from $\tau$ on the relevant side. These conditions establish stability relative to the policy scores, not correctness of the learned policy preference. They also explain why an evidence-dependent task should not start with an unsupported policy margin that the evidence cannot overcome.

## C. Counterfactual Training

Training coordinates policy and evidence scoring by distinguishing task fact changes from attempts to override the policy through external content. Related inputs form groups with checked task labels, and the same model scores each view. The objectives supervise both final decisions and preferences expressed by policy scores.

Paired supervision. Let $q ( x ) = \mathrm { s o f t m a x } ( z ( x ) )$ denote the distribution over valid candidates. Cross-entropy supervises each view against its task label. For two views with a checked invariance relation, such as a clean input and an injection that preserves the relevant facts, a symmetric KL penalty encourages agreement between their candidate distributions.

For a pair $( x ^ { 0 } , y _ { 0 } )$ and $( x ^ { 1 } , y _ { 1 } )$ whose correct labels differ, we use the change in their relative candidate scores,

$$
\begin{array} { r } { s = ( z _ { y _ { 0 } } ^ { 0 } - z _ { y _ { 1 } } ^ { 0 } ) - ( z _ { y _ { 0 } } ^ { 1 } - z _ { y _ { 1 } } ^ { 1 } ) , } \\ { \mathcal { L } _ { \mathrm { c h a n g e } } = \mathrm { s o f t p l u s } ( 1 - s ) . \qquad } \end{array}\tag{8}
$$

Minimizing this loss encourages a larger relative advantage for the original label on the original input than on the edited input. Endpoint cross-entropy also trains each view to select its correct candidate. Fact edits, policy edits, and authorized overrides enter this objective when their checked labels differ.

Training also distinguishes source authority through trusted and untrusted overrides. The same override text can change the target when supplied through an authorized policy channel and leave it unchanged when embedded in external evidence.

Authority supervision. When annotations specify whether the policy alone determines the answer or external facts are needed, authority supervision assigns corresponding targets to the policy scores. For a policy-sufficient input with correct candidate y, cross-entropy supervises the policy scores, and a margin objective encourages the correct candidate to remain preferred after bounded evidence corrections,

$$
\begin{array} { c } { { m _ { y } = a _ { y } - \displaystyle \operatorname* { m a x } _ { k \neq y } a _ { k } , } } \\ { { \angle _ { \mathrm { m a r g i n } } = \mathrm { s o f t p l u s } ( 2 \beta + 1 - m _ { y } ) . } } \end{array}\tag{9}
$$

The target margin follows the $2 \beta$ correction bound in Equation (7), with an additional unit of separation. For evidencerequired inputs, a KL penalty towards a uniform policy distribution discourages strong preferences unsupported by the policy alone. This leaves the evidence correction room to resolve the decision. During the second training stage, the uniformity objective updates the authority head without propagating its gradient into the shared representations. Other objectives update their corresponding trainable parameters.

Training procedure. The first stage trains complete groups with endpoint cross-entropy, paired and authority supervision, and online adversarial refinement. It uses evidence representation $d _ { k } ^ { \mathrm { c t x } }$ from $C _ { k } ^ { D }$ , allowing authority uniformity to update shared parameters. The second adopts the deployed evidence mixture and policy–evidence interaction in Equations (4) and (5). For prediction stability, it averages cross-entropy over two stochastic forward passes with shared parameters and adds symmetric KL consistency between their predictions.

Online adversarial refinement targets high-loss attacks against the current model, following robust optimization [20]. For each participating source group, the task oracle verifies that candidate attacks preserve labels. The highest-loss attack supplies adversarial cross-entropy for correctness and an invariance term for consistency with the original input.

Authentication uses a threshold on the grant-versus-deny score difference, which adaptation trains while retaining the scoring architecture. Legitimate and attack input pairs provide a ranking objective, alongside cross-entropy averaged over two stochastic passes and prediction consistency. Development data set a common deployment threshold, fixed during evaluation. Section IV-A4 details the backbone, optimization, loss weights, attack and epoch counts, and checkpoint selection.

## IV. EVALUATION

We investigate five research questions.

• RQ1 (Overall Effectiveness). How well does TrustJev balance task utility and injection robustness compared with existing methods?

• RQ2 (Design Contributions). How do authority isolation, bounded evidence fusion, and dedicated supervision contribute to TrustJev’s performance?

• RQ3 (Robustness Generalization). How well does TrustJev retain its robustness under new attack variants, combinations, and multiple attempts?

• RQ4 (Training and Inference Costs). What training and inference costs does TrustJev incur, and how do these costs scale with input size?

• RQ5 (Industrial Evaluation). How well does TrustJev perform in a scenario based on real industrial data?

## A. Experimental Setup

1) Datasets: We evaluate access control, external content processing, and rule-based authorization using three datasets summarized in Table II. Each group contains a base case and related variants, each evaluated as a separate view. Related views remain in the same split, grouped by base case, source text, or authentication pair. No inputs match across splits after normalizing policy and evidence text, ignoring candidate order.

TABLE II  
DATASET STATISTICS.
<table><tr><td>Statistic</td><td>TensorTrust</td><td>IHEval</td><td>AEB</td></tr><tr><td>Task types</td><td>1</td><td>2</td><td>7</td></tr><tr><td>Candidates per input</td><td>2</td><td>3</td><td>3</td></tr><tr><td>Training groups / views</td><td>512 / 1,024</td><td>396  /  4,500</td><td>512 / 3,180</td></tr><tr><td>Validation groups / views</td><td>256 /  512</td><td>164 /  732</td><td>128 / 768</td></tr><tr><td>Test groups / views</td><td>783  /  1,566</td><td>384 /  1,728</td><td>512 / 3,072</td></tr><tr><td>Non-attack test views</td><td>783</td><td>576</td><td>2,048</td></tr><tr><td>Attack test views</td><td>783</td><td>1,152</td><td>1,024</td></tr></table>

TensorTrust [10] contains human-written attacks and defenses collected through an online game. We pair legitimate access-code and attack inputs from authentication records under a shared defense policy, with Access Granted and Access Denied as candidates. We split authentication cases with duplicate checks on defense, password, and attack texts.

IHEval [11] evaluates instruction hierarchy compliance. We adapt and extend its language identification and Slack shortestusername selection tasks using XL-Sum articles [21] and generated username cases, with new articles and usernames for testing. Candidates are English, Chinese, and Spanish for language identification and three record identifiers for Slack, with usernames retained in the evidence.

AuthorityEvidenceBench (AEB) is our diagnostic benchmark for evidence use and policy authority under injection. Policies and evidence are expressed in natural language, while seven executable authorization rule types precisely determine labels before and after controlled modifications, with independently checked labels supporting diagnosis. Each case includes six views comprising the original input, injection, fact and policy modifications, and trusted and untrusted overrides. These comparisons distinguish injection resistance from evidence neglect and identify errors in fact use, injection handling, and authorized policy updates. Splits use separate entity pools.

2) Evaluation Metrics: We report Utility (U) and Robustness (R) as decision accuracy averaged equally across tasks on non-attack and attack views, respectively, and Balance (B) as their harmonic mean. For a dataset with T tasks, let N and $\boldsymbol { A } _ { t }$ index the non-attack and attack views of task t. Given prediction $\hat { y } _ { i }$ and label $y _ { i }$ , we compute

$$
U = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \frac { 1 } { | \mathcal { N } _ { t } | } \sum _ { i \in \mathcal { N } _ { t } } \mathbf { 1 } [ \hat { y } _ { i } = y _ { i } ] ,
$$

$$
R = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \frac { 1 } { | \mathcal { A } _ { t } | } \sum _ { i \in \mathcal { A } _ { t } } \mathbf { 1 } [ \hat { y } _ { i } = y _ { i } ] .\tag{10}
$$

Balance is computed as

$$
B = \frac { 2 U R } { U + R } ,\tag{11}
$$

with $B = 0$ when $U + R = 0$ . All scores are percentages.

3) Baselines: We compare 14 baselines across decision systems, injection defenses, and training strategies.

Decision systems. The commercial model Jev [1] maps unstructured state and instructions to typed decisions with probabilities. Kev [3] is open-source and scores candidates in one pass without generating text. SecJev [2] adapts Kev’s architecture through security training to decide from policies and observations. We use native Choice interfaces and additionally fine-tune Kev and SecJev on target tasks.

Injection defenses. To compare this emerging decision interface with established LLM defenses, we adapt four methods to candidate scoring. StruQ [6] combines protected delimiters, control-marker escaping, and injection-augmented training. SecAlign [7] uses candidate-level DPO with a frozen reference, preferring correct answers over attack targets. PI-Guard [22] transfers injection-detection representations to decisions, retaining auxiliary detection supervision. DRIP [8] combines trainable affine editing of evidence representations with residual fusion of instruction representations. StruQ, SecAlign, and DRIP use the same backbone as TrustJev, with scorers initialized through defensive task training but without authority isolation or bounded corrections.

Training baselines. Five baselines compare input formatting and training objectives. Flat SFT concatenates policy and evidence under ordinary headings, while Structured SFT uses application-controlled boundaries and control-marker escaping. R-Drop [23] enforces prediction consistency through symmetric KL between two dropout passes. Group DRO [24] upweights source, task, or authentication risk groups with higher loss. DPO [25] optimizes candidate preferences against frozen reference logits, rejecting cyclic alternatives rather than attack targets. All share TrustJev’s backbone but use sequential positions and unbounded additive readouts, with both branches accessing evidence. AEB and IHEval training retains adversarial cross-entropy and attack consistency but excludes TrustJev’s dedicated paired and authority supervision.

4) Implementation Details: TrustJev adapts Qwen3-0.6B with rank-8 LoRA [26] and trainable readouts, using budget ceiling $\begin{array} { r } { B ( K ) \ = \ 8 + \ \frac 1 2 } \end{array}$ log K and a 1,536-token sequence limit. AdamW [27] uses epochs (3, 2, 2) and learning rates $( 1 0 ^ { - 4 } , 5 \times 1 0 ^ { - 5 } , 1 0 ^ { - 4 } )$ for the first, second, and authentication stages, respectively. Effective batch sizes are two complete source groups for task training and four for authentication. Adversarial refinement uses 16 candidate attacks per AEB group. With averaged endpoint cross-entropy weighted at 1, second-stage weights are 1 for stochastic consistency, 0.25 for adversarial cross-entropy, 0.025 each for authority supervision and policy margin, and 0.0075 each for ordinary invariance, preference change, and adversarial invariance.

PIGuard jointly trains its DeBERTa-v3-base encoder and candidate head with MOF replay, averaging scores over overlapping 2,048-token windows without hard blocking. On IHEval, it receives extended task training and uses the original validation split for checkpoint selection. We use SecJev-0.8B, Kev-0.8B (Git revision 54f4f877), and Jev 1.13.0. Kev and SecJev use FP32 inference, preserving native LoRA, decision heads, losses, and learning rates during fine-tuning.

TABLE III RESULTS ON TENSORTRUST, IHEVAL, AND AEB.
<table><tr><td rowspan="2"></td><td rowspan="2">Category Method</td><td colspan="3">TensorTrust</td><td colspan="3">IHEval</td><td colspan="3">AEB</td></tr><tr><td>Utility</td><td>Robustness</td><td>Balance</td><td>Utility</td><td>Robustness</td><td>Balance</td><td>Utility</td><td>Robustness</td><td>Balance</td></tr><tr><td rowspan="6">Decision systems</td><td>Jev</td><td>77.39</td><td>75.73</td><td>76.56</td><td>99.22</td><td>98.44</td><td>98.83</td><td>92.14</td><td>97.95</td><td>94.96</td></tr><tr><td>Kev</td><td>59.26</td><td>53.26</td><td>56.10</td><td>41.02</td><td>40.04</td><td>40.52</td><td>68.67</td><td>35.46</td><td>46.77</td></tr><tr><td>SecJev</td><td>60.03</td><td>49.17</td><td>54.06</td><td>41.93</td><td>41.28</td><td>41.60</td><td>66.44</td><td>30.94</td><td>42.22</td></tr><tr><td>Kev (fine-tuned)</td><td> $8 8 . 8 5 \pm 1 . 5 9$ </td><td> $8 8 . 8 5 \pm 1 . 7 3$ </td><td> $8 8 . 8 3 \pm 0 . 1 1$ </td><td> $9 7 . 9 6 \pm 0 . 4 2 $ </td><td>98.74 ± 0.20</td><td> $9 8 . 3 5 \pm 0 . 1 7$ </td><td> $9 9 . 3 0 \pm 0 . 2 2 $ </td><td> $9 7 . 8 9 \pm 0 . 8 7$ </td><td> $9 8 . 5 9 \pm 0 . 5 1$ </td></tr><tr><td>SecJev (fine-tuned)</td><td> $6 7 . 1 3 \pm 2 . 0 8$ </td><td> $5 9 . 8 6 \pm 2 . 7 5$ </td><td> $6 3 . 2 3 \pm 0 . 7 9$ </td><td> $9 4 . 4 0 \pm 0 . 5 7$ </td><td> $9 6 . 7 0 \pm 0 . 4 7$ </td><td> $9 5 . 5 4 \pm 0 . 4 7$ </td><td> $9 5 . 2 6 \pm 1 . 7 1$ </td><td> $9 1 . 0 2 \pm 0 . 5 1 $ </td><td> $9 3 . 0 9 \pm 1 . 0 6$ </td></tr><tr><td>StruQ</td><td> $8 8 . 0 7 \pm 1 . 4 2$ </td><td> $8 9 . 5 3 \pm 1 . 8 4$ </td><td> $8 8 . 7 7 \pm 0 . 6 7$ </td><td> $9 8 . 0 8 \pm 0 . 4 8 $ </td><td> $9 8 . 6 7 \pm 0 . 1 8$ </td><td> $9 8 . 3 8 \pm 0 . 2 6$ </td><td> $9 7 . 1 0 \pm 1 . 4 8$ </td><td> $9 7 . 5 2 \pm 1 . 4 2$ </td><td> $9 7 . 3 1 \pm 1 . 3 6$ </td></tr><tr><td rowspan="5">Injection defenses</td><td>SecAlign</td><td> $8 4 . 1 6 \pm 3 . 4 5$ </td><td> $8 7 . 9 4 \pm 2 . 2 6$ </td><td> $8 5 . 9 3 \pm 1 . 1 1$ </td><td> $9 6 . 2 4 \pm 0 . 9 0$ </td><td> $9 8 . 1 4 \pm 0 . 5 7$ </td><td> $9 7 . 1 8 \pm 0 . 6 3 $ </td><td> $9 4 . 9 1 \pm 3 . 4 6$ </td><td> $9 7 . 2 0 \pm 1 . 5 6$ </td><td> $9 6 . 0 3 \pm 2 . 4 7$ </td></tr><tr><td>PIGuard</td><td> $8 3 . 9 4 \pm 6 . 6 0$ </td><td> $7 6 . 9 0 \pm 1 1 . 6 4$ </td><td> $7 9 . 9 2 \pm 8 . 3 6$ </td><td> $9 4 . 0 3 \pm 1 . 8 1 $ </td><td> $9 4 . 2 0 \pm 2 . 1 0$ </td><td> $9 4 . 1 1 \pm 1 . 9 5$ </td><td> $8 7 . 5 7 \pm 1 . 1 9$ </td><td> $9 2 . 4 3 \pm 2 . 4 0$ </td><td> $8 9 . 9 3 \pm 1 . 5 9$ </td></tr><tr><td>DRIP</td><td> $8 5 . 2 0 \pm 5 . 3 1$ </td><td> $9 0 . 0 2 \pm 3 . 3 0$ </td><td> $8 7 . 3 9 \pm 2 . 1 3$ </td><td> $9 7 . 1 0 \pm 0 . 7 7$ </td><td> $9 7 . 1 9 \pm 0 . 4 4$ </td><td> $9 7 . 1 4 \pm 0 . 5 7 $ </td><td> $9 3 . 0 0 \pm 3 . 0 8$ </td><td> $9 6 . 0 4 \pm 1 . 0 4$ </td><td> $9 4 . 4 8 \pm 2 . 0 5$ </td></tr><tr><td>Flat SFT</td><td> $8 9 . 6 6 \pm 2 . 1 5$ </td><td> $8 7 . 9 5 \pm 0 . 9 1$ </td><td> $8 8 . 7 8 \pm 0 . 8 6$ </td><td> $9 8 . 2 6 \pm 0 . 3 0$ </td><td> $9 8 . 3 1 \pm 0 . 9 1 $ </td><td> $9 8 . 2 8 \pm 0 . 6 1$ </td><td> $9 8 . 2 5 \pm 0 . 6 1 $ </td><td> $9 8 . 9 3 \pm 0 . 2 5$ </td><td> $9 8 . 5 9 \pm 0 . 2 3 $ </td></tr><tr><td>Structured SFT</td><td> $8 9 . 9 5 \pm 0 . 8 3$ </td><td> $8 8 . 1 7 \pm 1 . 7 2$ </td><td> $8 9 . 0 4 \pm 0 . 7 7$ </td><td> $9 8 . 3 5 \pm 0 . 3 0$ </td><td> $9 9 . 3 7 \pm 0 . 1 0$ </td><td> $9 8 . 8 6 \pm 0 . 1 2$ </td><td> $9 8 . 5 9 \pm 0 . 1 8$ </td><td> $9 8 . 2 1 \pm 0 . 8 5$ </td><td> $9 8 . 4 0 \pm 0 . 3 5$ </td></tr><tr><td rowspan="5">Training baselines</td><td>R-Drop</td><td> $9 0 . 7 2 \pm 0 . 4 5$ </td><td> $8 6 . 4 2 \pm 1 . 2 4$ </td><td> $8 8 . 5 2 \pm 0 . 8 0$ </td><td> $9 8 . 3 5 \pm 0 . 5 4$ </td><td> $9 9 . 3 9 \pm 0 . 4 0$ </td><td> $9 8 . 8 7 \pm 0 . 4 2$ </td><td> $9 7 . 5 1 \pm 0 . 5 5$ </td><td> $9 9 . 0 2 \pm 0 . 7 7$ </td><td> $9 8 . 2 6 \pm 0 . 3 1$ </td></tr><tr><td>Group DRO</td><td> $8 8 . 3 4 \pm 2 . 5 0$ </td><td> $8 7 . 5 7 \pm 2 . 8 5$ </td><td> $8 7 . 9 0 \pm 0 . 2 0 $ </td><td> $9 8 . 4 8 \pm 0 . 4 6$ </td><td> $9 9 . 0 5 \pm 0 . 2 7$ </td><td> $9 8 . 7 6 \pm 0 . 1 7$ </td><td> $9 7 . 3 3 \pm 0 . 8 8$ </td><td> $9 7 . 1 4 \pm 2 . 1 8$ </td><td> $9 7 . 2 2 \pm 0 . 6 8$ </td></tr><tr><td>DPO</td><td> $8 9 . 0 6 \pm 1 . 5 2 $ </td><td> $8 9 . 1 0 \pm 1 . 6 2$ </td><td> $8 9 . 0 6 \pm 0 . 1 2$ </td><td> $9 7 . 6 6 \pm 0 . 9 4$ </td><td> $9 8 . 6 8 \pm 0 . 5 9$ </td><td> $9 8 . 1 6 \pm 0 . 2 5$ </td><td> $9 4 . 5 8 \pm 3 . 1 3$ </td><td> $9 6 . 4 5 \pm 2 . 5 4$ </td><td> $9 5 . 4 5 \pm 0 . 7 7$ </td></tr><tr><td>TrustJev</td><td> $9 0 . 9 7 \pm 1 . 0 3$ </td><td> $8 9 . 2 7 \pm 1 . 1 4$ </td><td> $9 0 . 1 1 \pm 0 . 1 2$ </td><td> $9 9 . 3 1 \pm 0 . 3 0$ </td><td> $9 9 . 5 7 \pm 0 . 2 5$ </td><td> $9 9 . 4 4 \pm 0 . 1 2$ </td><td> $9 9 . 5 1 \pm 0 . 3 0$ </td><td> $9 9 . 4 2 \pm 0 . 0 0$ </td><td> $9 9 . 4 6 \pm 0 . 1 5$ </td></tr></table>

Each job uses one 24 GB RTX 4090. TrustJev and training baselines select AEB and IHEval checkpoints by validation Balance, evaluating DPO after preference optimization. On TensorTrust, they use second-epoch checkpoints and calibrate one threshold per method by mean validation Balance. Kev and SecJev set TensorTrust thresholds using development data.

## B. Overall Effectiveness (RQ1)

Table III reports per-run metrics as means and sample standard deviations for methods evaluated three times. TrustJev achieves the highest mean Balance, exceeding the strongest baselines by 1.05, 0.57, and 0.87 percentage points on TensorTrust, IHEval, and AEB, respectively. Mean Utility leads on all datasets and mean Robustness on IHEval and AEB.

TrustJev improves both mean Utility and Robustness over all five training baselines on every dataset, balancing attack resistance with accurate legitimate decisions. Among injection defenses, DRIP achieves higher Robustness on TensorTrust, but TrustJev has 0.75 points lower Robustness and 5.77 points higher Utility, yielding higher Balance. On IHEval and AEB, TrustJev exceeds all four defenses in both measures.

Target task fine-tuning makes Kev a competitive decision baseline, tying Flat SFT for the highest baseline Balance on AEB. TrustJev nevertheless improves both Utility and Robustness over fine-tuned Kev and fine-tuned SecJev on all three datasets. It also outperforms Jev, the strongest decision system without target task fine-tuning, in both measures throughout.

## C. Design Contributions (RQ2)

We evaluate individual and grouped component removals from the first training stage, with the same backbone, data, and training budgets. Isolation removal exposes policy scoring to evidence while retaining semantic positions. Bound removal uses $\boldsymbol { r } _ { k } = \boldsymbol { e } _ { k } - \bar { \boldsymbol { e } }$ and removes 2β from the margin objective. Paired removal disables auxiliary objectives over related inputs but retains stochastic consistency on the same input. Authority removal disables policy score cross-entropy, uniformity, and margin supervision. Both scoring branches, labeled training views, and the remaining objectives are retained. Each ablated variant is similarly trained with three seeds.

TABLE IV  
TRUSTJEV ABLATIONS. BALANCE (%) SHOWS MEAN ± SAMPLE SD OVER THREE RUNS. × DENOTES REMOVAL.
<table><tr><td rowspan="2">Variant</td><td colspan="2">Architecture</td><td colspan="2">Supervision</td><td rowspan="2">TensorTrust</td><td rowspan="2">IHEval</td><td rowspan="2">AEB</td></tr><tr><td></td><td></td><td></td><td>Isolation Bound Paired Authority</td></tr><tr><td>Full</td><td></td><td></td><td></td><td></td><td> $9 0 . 1 1 \pm 0 . 1 2$ </td><td> $9 9 . 4 4 \pm 0 . 1 2$ </td><td> $9 9 . 4 6 \pm 0 . 1 5$ </td></tr><tr><td>A1</td><td>X</td><td></td><td></td><td></td><td> $8 7 . 1 7 \pm 0 . 7 2$ </td><td> $9 8 . 5 2 \pm 0 . 2 1 $ </td><td> $9 5 . 1 8 \pm 0 . 8 3$ </td></tr><tr><td>A2</td><td></td><td>X</td><td></td><td></td><td> $8 8 . 1 3 \pm 0 . 6 6$ </td><td> $9 8 . 3 7 \pm 0 . 1 1 $ </td><td> $9 7 . 2 4 \pm 0 . 5 0 $ </td></tr><tr><td>A3</td><td>X</td><td>X</td><td></td><td></td><td> $8 6 . 4 2 \pm 0 . 2 3$ </td><td> $9 8 . 2 4 \pm 0 . 0 6$ </td><td> $9 5 . 1 0 \pm 0 . 4 7$ </td></tr><tr><td>A4</td><td></td><td></td><td>X</td><td></td><td> $8 8 . 4 1 \pm 0 . 1 2$ </td><td> $9 8 . 3 5 \pm 0 . 1 6$ </td><td> $9 6 . 2 2 \pm 0 . 3 1$ </td></tr><tr><td>A5</td><td></td><td></td><td></td><td>X</td><td> $8 6 . 6 4 \pm 0 . 0 8$ </td><td> $9 8 . 4 4 \pm 0 . 1 3$ </td><td> $9 7 . 2 1 \pm 0 . 7 1 $ </td></tr><tr><td>A6</td><td></td><td></td><td>X</td><td>X</td><td> $8 7 . 0 5 \pm 1 . 0 6$ </td><td> $9 8 . 0 6 \pm 0 . 1 3$ </td><td> $9 5 . 8 2 \pm 0 . 3 1$ </td></tr><tr><td>A7</td><td>X</td><td>X</td><td>X</td><td>X</td><td> $8 5 . 4 8 \pm 0 . 6 2$ </td><td> $9 7 . 5 0 \pm 0 . 2 3 $ </td><td> $9 4 . 4 0 \pm 0 . 2 2 $ </td></tr></table>

Overall ablations. Table IV shows that each individual removal lowers mean Balance on all three datasets, supporting contributions from both architectural constraints and dedicated supervision. Their relative effects vary by task. Among individual removals, authority supervision removal produces the largest drop on TensorTrust at 3.47 percentage points, while isolation removal has the largest effect on AEB at 4.28 points. Removing all four components (A7) yields the lowest Balance throughout, indicating that the complete design provides the best overall balance among the evaluated variants.

![](images/2346eca05ab1c80fa8286e69f4c59d97053a5555a05aeeaa25180124cfb24dda.jpg)  
Fig. 3. AEB paired diagnostics for Table IV. Dots and intervals show means and sample standard deviations over three runs. Dashed lines mark Full means, with Full statistics below panel titles. Joint accuracy requires both predictions in a pair to be correct.

AEB paired diagnostics. Figure 3 distinguishes component contributions to policy authority and fact responsiveness on the AEB test groups. Injection pairs clean and attacked inputs with unchanged facts and labels, while fact change pairs inputs with changed facts and labels. Authority change moves identical override text between evidence and policy, changing the label. Each diagnostic evaluates all variants on the same input pairs.

Isolation primarily supports injection resistance and authority handling, with its removal lowering their accuracies by 5.14 and 5.01 points while leaving mean fact-change accuracy essentially unchanged. Removing authority supervision lowers fact-change accuracy by 3.98 points, consistent with leaving room for relevant evidence when policy alone cannot decide.

The architecture’s benefit for fact responsiveness depends on supervision. With both supervision components retained, Full and A3 both average 98.27% fact-change accuracy. Without paired and authority supervision, retaining the architecture raises accuracy from 84.50% (A7) to 93.33% (A6).

TABLE V  
STABILITY BY EVIDENCE DEPENDENCE, AVERAGED OVER THREE SEEDS.
<table><tr><td rowspan="2">Task</td><td rowspan="2">Type</td><td rowspan="2">Utility (%)</td><td rowspan="2">Robustness (%)</td><td colspan="3">Percentiles of s</td><td rowspan="2">Coverage (%)</td><td rowspan="2">Policy accuracy (%)</td></tr><tr><td>10th</td><td>50th</td><td>90th</td></tr><tr><td>AEB</td><td> $P$ </td><td>100.00</td><td>100.00</td><td>9.35</td><td>10.20</td><td>10.92</td><td>100.00</td><td>100.00</td></tr><tr><td>AEB</td><td> $P + D$ </td><td>99.13</td><td>99.22</td><td>-14.59</td><td>-14.04</td><td>-13.50</td><td>0.00</td><td>N/A</td></tr><tr><td>IHEval Language</td><td> $P + D$ </td><td>99.65</td><td>99.91</td><td>-16.43</td><td>-16.41</td><td>-16.40</td><td>0.00</td><td>N/A</td></tr><tr><td>IHEval Slack</td><td> $P + D$ </td><td>98.96</td><td>99.22</td><td>-16.57</td><td>-16.57</td><td>-16.57</td><td>0.00</td><td>N/A</td></tr></table>

P denotes policy-sufficient inputs and $P + D$ evidence-required inputs. Percentiles summarize equally weighted normal and attack views within each seed, pooling AEB tasks. N/A indicates policy accuracy is undefined when coverage is zero.

Policy stability and evidence dependence. Table V tests whether learned policy preferences reflect the need for evidence. With $s = m - 2 \beta$ , coverage is the fraction satisfying $s \ > \ 0$ , which guarantees that evidence corrections cannot change the policy-preferred candidate under Equation (7). Policy accuracy measures correctness within this subset. The analysis covers AEB and IHEval, excluding TensorTrust because its evidence dependence could not be reliably classified.

Policy-sufficient inputs have 100% coverage and policy accuracy, showing both stable and correct policy preferences. Evidence-required inputs have zero coverage but retain high Utility and Robustness. Together with the fact-change diagnostics, these results support policy stability when rules suffice and evidence responsiveness when facts are needed.

![](images/c80eb43e79627fb96cb9fda117b0368c90ca6bfcc8f40a095b7bfc30cdadf2dd.jpg)  
Fig. 4. Robustness under attack variants, composition, and multiple attempts. Panels (a–b) show mean attack accuracy and (c) cumulative robust accuracy for a fixed attack pool on a logarithmic horizontal axis.

## D. Robustness Generalization (RQ3)

RQ3 tests robustness under new attack variants, composition, and multiple attempts. From RQ1’s AEB results, we select the strongest Balance baselines (Flat SFT and fine-tuned Kev), Robustness baseline (R-Drop), and injection defenses by Balance (StruQ and SecAlign), alongside Structured SFT and Jev as references for structured training and commercial decision systems. Separate AEB sources use the same generator with adjusted task proportions and conditions. Attacks modify evidence without changing task facts, policies, candidates, or labels. Panels share sources, and methods receive identical attacks within each panel.

Attack variants. Figure 4(a) compares 16 existing and 16 new attacks per source. The existing attacks use eight templates, each targeting two incorrect candidates. New attacks use authority spoofing, fake verification, and candidate lookup through code mappings, individually or combined. TrustJev maintains high accuracy across libraries, scoring 99.36% on existing and 99.65% on new attacks, while Structured SFT, StruQ, and SecAlign have lower accuracy on new attacks.

Attack composition. Figure 4(b) separately reports three conditions within the 16 new attacks per source in (a), with four attacks each for authority spoofing, fake verification, and their combination. The other four attacks involve candidate lookup and are omitted to focus on the two factors. All three conditions add 192 tokens to the evidence. TrustJev scores 99.79%, 99.81%, and 99.58%, respectively. Under the combined attack, it exceeds Structured SFT, the strongest baseline at 99.01%, by 0.57 percentage points.

TABLE VI  
TRAINING COST AND PEAK ALLOCATED GPU MEMORY. TRAINING TIME SHOWS MEAN ± SAMPLE SD OVER THREE RUNS.
<table><tr><td rowspan="2">Method</td><td colspan="2">Training</td><td colspan="3">Inference memory (GiB)</td></tr><tr><td>Time (GPU h)</td><td>Peak (GiB)</td><td>AEB</td><td>IHEval</td><td>TensorTrust</td></tr><tr><td>Structured SFT</td><td> $0 . 6 2 9 \pm 0 . 0 0 7$ </td><td>3.277</td><td>1.150</td><td>1.216</td><td>1.207</td></tr><tr><td>SecAlign</td><td> $0 . 5 8 5 \pm 0 . 0 0 3$ </td><td>2.331</td><td>1.150</td><td>1.216</td><td>1.208</td></tr><tr><td>DRIP</td><td> $0 . 6 2 9 \pm 0 . 0 0 1$ </td><td>2.331</td><td>1.152</td><td>1.218</td><td>1.209</td></tr><tr><td>R-Drop</td><td> $0 . 9 4 8 \pm 0 . 0 0 3$ </td><td>4.021</td><td>1.150</td><td>1.216</td><td>1.207</td></tr><tr><td>StruQ</td><td> $0 . 5 6 8 \pm 0 . 0 0 8$ </td><td>2.331</td><td>1.150</td><td>1.216</td><td>1.208</td></tr><tr><td>TrustJev</td><td> $0 . 8 1 5 \pm 0 . 0 1 5$ </td><td>4.022</td><td>1.150</td><td>1.216</td><td>1.207</td></tr></table>

Multiple attack attempts. Figure 4(c) uses a separately constructed fixed pool of 32 attacks per source involving candidate lookup and result spoofing. For $k \in \{ 1 , 2 , 4 , 8 , 1 6 , 3 2 \}$ cumulative robust accuracy averages the probability of answering all k attacks correctly when sampled uniformly without replacement. For N sources, this is $\mathrm { C R A } _ { m } ( k ) \ =$ $\textstyle \frac { 1 0 0 } { N } \sum _ { i = 1 } ^ { N } \bar { \binom { g _ { i , m } } { k } } / \binom { 3 2 } { k }$ , where $g _ { i , m }$ counts attacks answered correctly by method m on source i, and the numerator is zero for $g _ { i , m } < k .$ At 32 attempts, TrustJev retains 95.77% cumulative robust accuracy versus 80.98% for fine-tuned Kev, the strongest comparator. Their gap grows from 2.18 percentage points at one attempt to 14.79 at 32. High single-attempt accuracy does not imply resilience across the attack pool, where TrustJev retains its advantage across multiple attempts.

## E. Computational Cost (RQ4)

RQ4 compares training and defense approaches sharing TrustJev’s backbone under matched hardware and input conditions. Structured SFT provides the reference cost, while R-Drop, StruQ, SecAlign, and DRIP represent consistency regularization, injection-augmented training, preference optimization, and representation editing, respectively. We measure three runs with inference batch size 1, reflecting the common workflow setting of making one decision per request.

Training and memory. Training cost covers initialization, method-specific training and authentication adaptation, reference predictions, validation, and calibration, counting shared stages once per method. Table VI shows that TrustJev requires more training time than four baselines but less than R-Drop, while maintaining similar inference memory. Training takes 0.815 GPU h versus 0.568–0.629 for four baselines and 0.948 for R-Drop. Peak training memory is 4.022 GiB, close to R-Drop and above the other methods.

Request latency. We evaluate all test inputs, taking each input’s median latency over three executions per seed and then across seeds. Quantiles summarize these input medians. Figure 5(a) shows median latency near 40–41 ms for TrustJev, comparable to the baselines. On IHEval and TensorTrust, its P95 is higher at 45.2 and 43.3 ms, versus baseline ranges of 42.5–43.5 and 40.8–41.6 ms, respectively.

TABLE VII  
INDUSTRIAL ALERT TRIAGE. ACCURACY AND ERROR RATES ARE PERCENTAGES. BOLD INDICATES THE BEST RESULT IN EACH COLUMN.
<table><tr><td>Method</td><td>Accuracy ↑</td><td>Downgrade error ↓</td><td>Upgrade error ↓</td><td>P50 (ms) ↓</td></tr><tr><td>TrustJev</td><td>97.65</td><td>2.35</td><td>2.36</td><td>42.11</td></tr><tr><td>Direct Jev</td><td>92.56</td><td>16.53</td><td>0.00</td><td>201.47</td></tr><tr><td>Extraction + rules</td><td>61.63</td><td>6.09</td><td>64.79</td><td>0.0075</td></tr><tr><td>Extraction + embeddings</td><td>58.04</td><td>14.07</td><td>64.79</td><td>2.80</td></tr></table>

Input scaling. Figure 5(b–c) varies evidence length and candidate count separately. Increasing evidence from 128 to 1,024 tokens raises TrustJev’s median latency by 4.3 ms versus 1.7–1.9 ms for the baselines. Increasing candidate count from 2 to 16 raises its median by 2.7 ms versus 1.7–2.1 ms. TrustJev has similar median latency on test inputs, with larger increases toward the upper end of the tested evidence lengths.

## F. Industrial Alert Triage (RQ5)

To explore industrial performance, we evaluate TrustJev offline on real alerts from nearly ten subsidiaries. Headquarters triage these alerts for expert intervention, but injected instructions can conceal serious incidents or unnecessarily escalate ordinary alerts. We construct 178 source cases from 10,349 logs. The 1,242 evaluation views comprise 581 authorcrafted attack views, original alerts, neutral text, and factchange controls. Table VII reports attack-view classification accuracy, downgrade errors among serious alerts, upgrade errors among ordinary alerts, and latency. TrustJev achieves 97.65% accuracy, 5.09 percentage points above direct Jev, with respective error rates of 2.35% and 2.36% and P50 latency of 42.11 ms, about one fifth of Jev’s. Rule extraction followed by rules or a text embedding classifier has lower accuracy and higher rates of both errors. Injected text can cause incorrect fact extraction, propagating errors to either classifier. TrustJev thus classifies alerts more accurately than the evaluated industrial pipelines and faster than direct Jev.

## V. LESSONS LEARNED

Separate policy adherence from fact responsiveness. Acceptance requires both injection resistance and correct responses to changed facts, since unchanged answers can reflect policy adherence or evidence neglect. Our early experiments showed strong authority recognition but weaker fact responsiveness. Following ML testing practice [28], [29], our diagnostics pair inputs requiring stable or changed decisions under instruction insertions, fact edits, and authority changes, requiring both endpoints to be correct.

Vary fact bindings to discourage answer shortcuts. Varying fact positions relative to candidate identifiers discourages positional shortcuts. In IHEval’s Slack task, permuting usernames across records preserves the username set and record identifiers while making the answer follow the shortest username. Combining these permutations with clean and conflicting instructions supervises both fact responsiveness and resistance to policy overrides.

![](images/2b27ae2fc67e819f2730b2b0192946f8c9dbd3276cc54de6b4824507cfbd576b.jpg)  
(a) Actual request latency (ms)

![](images/f9eb24463f0a4d26c2b32dfc186ed23ad725fc2479a9f9fa0b18bb84e20bcb26.jpg)  
(b) Evidence length (tokens)

![](images/6bdad35efcef8963fad643a15e5916a66fb9bd0fca8b1440c83040abc40cd879.jpg)  
(c) Candidate count K  
Fig. 5. Request latency and input scaling. Panel (a) shows test inputs from three datasets, with P50 labeled. Panel (b) varies evidence length with $K = 4 ,$ , and panel (c) varies candidate count with 256 evidence tokens. Both sweeps use 8 tokens per candidate and 16 content instances per method and level. Markers indicate P50, thick segments or boxes span P25–P75, and whiskers span P5–P95.

Use executable rules to ground natural-language diagnostics. AEB retains natural-language policies and evidence while executable rules label inputs before and after controlled modifications, supporting verifiable attribution of errors to fact use or instruction authority. Practitioners can similarly label diagnostic cases from rule-governed workflow portions. These complement realistic evaluations with explicit, independently verifiable expectations for each modification.

## VI. THREATS TO VALIDITY

First, implementation, training, and calibration differences between adapted defenses and native decision systems can confound comparisons. Grouping related views and checking duplicates across splits cannot rule out semantic overlap between cases that inflates generalization estimates.

Second, repeated runs and sample standard deviations leave small performance differences uncertain. Attacks sharing a source are not independent cases, and fixed attacks do not establish robustness to response-adaptive attacks [30], [31].

Third, evaluation covers one model scale and two or three candidates, assuming application-controlled policies and candidates with trusted instructions separated from external evidence. AEB underrepresents ambiguous policies and incomplete evidence, and Balance omits unequal operational error costs. Local latency excludes network and queueing delays, while concurrent serving and full workflow costs require deployment evaluation.

## VII. RELATED WORK

Decision interfaces and system defenses. Jev, Kev, and SecJev provide typed decisions and candidate scoring for natural-language workflows [1]–[3], which external-data injections can redirect [32], including Jev’s typed interface [5]. PI-Guard detects injections [22], while CaMeL isolates privileged program generation from untrusted data processing and enforces execution capabilities [33]. TrustJev complements these controls with bounded evidence influence in candidate scoring.

Instruction and data separation. Instruction hierarchy training prioritizes privileged instructions [34], Spotlighting marks untrusted content [35], StruQ pairs protected prompt and data channels with defensive training [6], and SecAlign favors legitimate over injected responses [7]. Instructional Segment Embedding adds instruction-type embeddings [36], ASIDE rotates data token embeddings [37], and DRIP edits data representations and reinforces instructions through residual fusion [8]. TrustJev scores policies without evidence and bounds corrections by a policy-derived budget, giving a sufficient margin for policy stability as evidence varies.

Behavioral supervision and evaluation. Metamorphic relations [38] address oracle limitations [39], exposing inconsistencies in driving [40], [41] and translation [42]–[44]. Check-List tests functionality, invariance, and directionality [45], while counterfactual edits change labels to reduce spurious associations [46]. TrustJev couples label-guided stability and change with authority supervision, distinguishing policysufficient and evidence-required decisions. BIPIA tests external content injections [47], InjecAgent tool response injections [48], and AgentDojo agent task completion and security during tool use [49]. AEB separates injection resistance from evidence neglect in finite candidate decisions using executable labels and controlled fact and authority edits.

## VIII. CONCLUSION

TrustJev combines authority isolation, bounded evidence corrections, and behavioral supervision for policy-guided candidate decisions. Its 0.6B implementation leads 14 baselines in mean Balance on three datasets and withstands all 32 fixedpool AEB attacks on 95.77% of sources versus the strongest comparator’s 80.98%. These gains increase training cost and latency growth with evidence length, while typical inference latency stays comparable and memory nearly unchanged.

[1] D. Almeida, “Introducing system one models & Jev,” 2026, typeSafe AI, September 15, 2026. Accessed October 21, 2026. [Online]. Available: https://typesafe.ai/blog/introducing-system-one-models-and-jev

[2] Z. Chen, F. Yu, H. Huang, Y. Li, A. Chen, and L. Chen, “SecJev: Bringing security expertise to system one decision models,” arXiv:2610.03073, 2026. [Online]. Available: https://arxiv.org/abs/2610. 03073

[3] J. Palmer, “Kev-0.8B: a calibrated decision model on Qwen3.5- 0.8B,” 2026, model card. Accessed October 21, 2026. [Online]. Available: https://github.com/jaredpalmer/kev/blob/main/docs/ model-cards/kev-0.8b.md

[4] Y. Liu, Y. Jia, R. Geng, J. Jia, and N. Z. Gong, “Formalizing and benchmarking prompt injection attacks and defenses,” in 33rd USENIX Security Symposium (USENIX Security 24). USENIX Association, 2024, pp. 1831–1847. [Online]. Available: https://www.usenix.org/ conference/usenixsecurity24/presentation/liu-yupei

[5] T. Wu and W. Y. B. Lim, “Decision hijacking: Prompt injection attacks on Jev’s typed probabilistic decisions,” arXiv:2609.28613, 2026. [Online]. Available: https://arxiv.org/abs/2609.28613

[6] S. Chen, J. Piet, C. Sitawarin, and D. Wagner, “StruQ: Defending against prompt injection with structured queries,” in 34th USENIX Security Symposium (USENIX Security 25). USENIX Association, 2025, pp. 2383–2400. [Online]. Available: https://www.usenix.org/ conference/usenixsecurity25/presentation/chen-sizhe

[7] S. Chen, A. Zharmagambetov, S. Mahloujifar, K. Chaudhuri, D. Wagner, and C. Guo, “SecAlign: Defending against prompt injection with preference optimization,” in The ACM Conference on Computer and Communications Security (CCS), 2025. [Online]. Available: https://sizhe-chen.github.io/SecAlign-Website/

[8] R. Liu, Y. Lin, Z. Huang, and J. S. Dong, “DRIP: Defending prompt injection via token-wise representation editing and residual instruction fusion,” in 33rd ACM Conference on Computer and Communications Security (CCS), 2026, accepted for publication. [Online]. Available: https://arxiv.org/abs/2511.00447

[9] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, “Attention is all you need,” in Advances in Neural Information Processing Systems, vol. 30, 2017. [Online]. Available: https://papers.nips.cc/paper files/paper/2017/hash/ 3f5ee243547dee91fbd053c1c4a845aa-Abstract.html

[10] S. Toyer, O. Watkins, E. Mendes, J. Svegliato, L. Bailey, T. Wang, I. Ong, K. Elmaaroufi, P. Abbeel, T. Darrell, A. Ritter, and S. Russell, “Tensor Trust: Interpretable prompt injection attacks from an online game,” in International Conference on Learning Representations, 2024. [Online]. Available: https://proceedings.iclr.cc/paper files/paper/2024/ hash/519c51529c3544b3430bd8b17d400365-Abstract-Conference.html

[11] Z. Zhang, S. Li, Z. Zhang, X. Liu, H. Jiang, X. Tang, Y. Gao, Z. Li, H. Wang, Z. Tan, Y. Li, Q. Yin, B. Yin, and M. Jiang, “IHEval: Evaluating language models on following the instruction hierarchy,” in Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers). Association for Computational Linguistics, 2025, pp. 8374–8398. [Online]. Available: https://aclanthology.org/2025.naacl-long.425/

[12] Z. Wang, J. Li, M. Ma, Z. Li, Y. Kang, C. Zhang, C. Bansal, M. Chintalapati, S. Rajmohan, Q. Lin, D. Zhang, C. Pei, and G. Xie, “Large language models can provide accurate and interpretable incident triage,” in 2024 IEEE 35th International Symposium on Software Reliability Engineering (ISSRE), 2024, pp. 523–534. [Online]. Available: https://ieeexplore.ieee.org/document/10771420/

[13] M. Du, F. Li, G. Zheng, and V. Srikumar, “DeepLog: Anomaly detection and diagnosis from system logs through deep learning,” in Proceedings of the 2017 ACM SIGSAC Conference on Computer and Communications Security, 2017. [Online]. Available: https: //dl.acm.org/doi/10.1145/3133956.3134015

[14] X. Zhang, Y. Xu, Q. Lin, B. Qiao, H. Zhang, Y. Dang, C. Xie, X. Yang, Q. Cheng, Z. Li, J. Chen, X. He, R. Yao, J.-G. Lou, M. Chintalapati, F. Shen, and D. Zhang, “Robust log-based anomaly detection on unstable log data,” in Proceedings of the 27th ACM Joint European Software Engineering Conference and Symposium on the Foundations of Software Engineering, 2019. [Online]. Available: https://dl.acm.org/doi/10.1145/3338906.3338931

[15] Z. Xu, Y. Wu, S. Wang, J. Gao, T. Qiu, Z. Wang, H. Wan, and X. Zhao, “Deep learning-based intrusion detection systems: A survey,” arXiv:2504.07839, 2025. [Online]. Available: https: //arxiv.org/abs/2504.07839

[16] X. Liu, Y. Han, Z. Zhang, K. Chang, Z. Xu, T. Qiu, W. Deng, J. Gao, X. Peng, H. Wan, and X. Zhao, “DiagChain: A diagnostic benchmark for evaluating LLM agents on evidence-grounded attack chain reconstruction,” arXiv:2608.03591, 2026. [Online]. Available: https://arxiv.org/abs/2608.03591

[17] Z. Xu, T. Qiu, H. Gao, W. Jiao, Q. Ma, J. Wang, J. Zhao, J. Gao, H. Wan, and X. Zhao, “Cerberus: Robust endpoint threat detection in practice,” in Proceedings of the 34th ACM International Conference on the Foundations of Software Engineering, 2026. [Online]. Available: https://dl.acm.org/doi/10.1145/3803437.3805249

[18] D. Hendrycks and K. Gimpel, “Gaussian error linear units (GELUs),” arXiv:1606.08415, 2016. [Online]. Available: https://arxiv.org/abs/1606. 08415

[19] N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, and R. Salakhutdinov, “Dropout: A simple way to prevent neural networks from overfitting,” Journal of Machine Learning Research, vol. 15, no. 56, pp. 1929–1958, 2014. [Online]. Available: https://www.jmlr.org/papers/v15/srivastava14a.html

[20] A. Madry, A. Makelov, L. Schmidt, D. Tsipras, and A. Vladu, “Towards deep learning models resistant to adversarial attacks,” in International Conference on Learning Representations, 2018. [Online]. Available: https://openreview.net/forum?id=rJzIBfZAb

[21] T. Hasan, A. Bhattacharjee, M. S. Islam, K. Mubasshir, Y.-F. Li, Y.-B. Kang, M. S. Rahman, and R. Shahriyar, “XL-Sum: Large-scale multilingual abstractive summarization for 44 languages,” in Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021. Association for Computational Linguistics, 2021, pp. 4693–4703. [Online]. Available: https://aclanthology.org/2021.findings-acl.413/

[22] H. Li, X. Liu, N. Zhang, and C. Xiao, “PIGuard: Prompt injection guardrail via mitigating overdefense for free,” in Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). Association for Computational Linguistics, 2025, pp. 30 420–30 437. [Online]. Available: https: //aclanthology.org/2025.acl-long.1468/

[23] X. Liang, L. Wu, J. Li, Y. Wang, Q. Meng, T. Qin, W. Chen, M. Zhang, and T.-Y. Liu, “R-Drop: Regularized dropout for neural networks,” in Advances in Neural Information Processing Systems, vol. 34, 2021. [Online]. Available: https://proceedings.neurips.cc/paper files/ paper/2021/hash/5a66b9200f29ac3fa0ae244cc2a51b39-Abstract.html

[24] S. Sagawa, P. W. Koh, T. B. Hashimoto, and P. Liang, “Distributionally robust neural networks for group shifts: On the importance of regularization for worst-case generalization,” in International Conference on Learning Representations, 2020. [Online]. Available: https://iclr.cc/virtual 2020/poster ryxGuJrFvS.html

[25] R. Rafailov, A. Sharma, E. Mitchell, C. D. Manning, S. Ermon, and C. Finn, “Direct preference optimization: Your language model is secretly a reward model,” in Advances in Neural Information Processing Systems, vol. 36, 2023. [Online]. Available: https://papers.neurips.cc/paper files/paper/2023/ hash/a85b405ed65c6477a4fe8302b5e06ce7-Abstract-Conference.html

[26] E. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen, “LoRA: Low-rank adaptation of large language models,” in International Conference on Learning Representations, 2022. [Online]. Available: https://iclr.cc/virtual/2022/poster/6319

[27] I. Loshchilov and F. Hutter, “Decoupled weight decay regularization,” in International Conference on Learning Representations, 2019. [Online]. Available: https://openreview.net/forum?id=Bkg6RiCqY7

[28] J. M. Zhang, M. Harman, L. Ma, and Y. Liu, “Machine learning testing: Survey, landscapes and horizons,” IEEE Transactions on Software Engineering, 2022. [Online]. Available: https://ieeexplore.ieee. org/document/9000651/

[29] S. Li, J. Guo, J.-G. Lou, M. Fan, T. Liu, and D. Zhang, “Testing machine learning systems in industry: An empirical study,” in Proceedings of the 44th International Conference on Software Engineering, 2022. [Online]. Available: https://www.microsoft.com/en-us/research/publication/ testing-machine-learning-systems-in-industry-an-empirical-study/

[30] A. Athalye, N. Carlini, and D. Wagner, “Obfuscated gradients give a false sense of security: Circumventing defenses to adversarial examples,” in Proceedings of the 35th International Conference on Machine Learning, ser. Proceedings of Machine Learning

Research, vol. 80, 2018, pp. 274–283. [Online]. Available: https: //proceedings.mlr.press/v80/athalye18a.html

[31] F. Tramer, N. Carlini, W. Brendel, and A. Madry, “On adaptive attacks to adversarial example defenses,” in Advances in Neural Information Processing Systems, vol. 33, 2020. [Online]. Available: https://papers.neurips.cc/paper files/paper/2020/ hash/11f38f8ecd71867b42433548d1078e38-Abstract.html

[32] K. Greshake, S. Abdelnabi, S. Mishra, C. Endres, T. Holz, and M. Fritz, “Not what you’ve signed up for: Compromising real-world LLMintegrated applications with indirect prompt injection,” in Proceedings of the 16th ACM Workshop on Artificial Intelligence and Security, 2023. [Online]. Available: https://dl.acm.org/doi/10.1145/3605764.3623985

[33] E. Debenedetti, I. Shumailov, T. Fan, J. Hayes, N. Carlini, D. Fabian, C. Kern, C. Shi, A. Terzis, and F. Tramer, “Defeating prompt\` injections by design,” arXiv:2503.18813, 2025. [Online]. Available: https://arxiv.org/abs/2503.18813

[34] E. Wallace, K. Xiao, R. Leike, L. Weng, J. Heidecke, and A. Beutel, “The instruction hierarchy: Training LLMs to prioritize privileged instructions,” arXiv:2404.13208, 2024. [Online]. Available: https://arxiv.org/abs/2404.13208

[35] K. Hines, G. Lopez, M. Hall, F. Zarfati, Y. Zunger, and E. Kıcıman, “Defending against indirect prompt injection attacks with spotlighting,” in Proceedings of the Conference on Applied Machine Learning in Information Security (CAMLIS 2024), 2024. [Online]. Available: https://ceur-ws.org/Vol-3920/paper03.pdf

[36] T. Wu, S. Zhang, K. Song, S. Xu, S. Zhao, R. Agrawal, S. R. Indurthi, C. Xiang, P. Mittal, and W. Zhou, “Instructional segment embedding: Improving LLM safety with instruction hierarchy,” in International Conference on Learning Representations, 2025. [Online]. Available: https://proceedings.iclr.cc/paper files/paper/2025/ hash/ea13534ee239bb3977795b8cc855bacc-Abstract-Conference.html

[37] E. Zverev, E. Kortukov, A. Panfilov, A. Volkova, S. Tabesh, S. Lapuschkin, W. Samek, and C. H. Lampert, “ASIDE: Architectural separation of instructions and data in language models,” in International Conference on Learning Representations, 2026. [Online]. Available: https://proceedings.iclr.cc/paper files/paper/2026/ hash/b77b5f1a2878fedd4705bfeafd24ca35-Abstract-Conference.html

[38] T. Y. Chen, F.-C. Kuo, H. Liu, P.-L. Poon, D. Towey, T. H. Tse, and Z. Q. Zhou, “Metamorphic testing: A review of challenges and opportunities,” ACM Computing Surveys, vol. 51, no. 1, 2018. [Online]. Available: https://dl.acm.org/doi/10.1145/3143561

[39] E. T. Barr, M. Harman, P. McMinn, M. Shahbaz, and S. Yoo, “The oracle problem in software testing: A survey,” IEEE Transactions on Software Engineering, vol. 41, no. 5, pp. 507–525, 2015. [Online]. Available: https://discovery.ucl.ac.uk/id/eprint/1471263

[40] Y. Tian, K. Pei, S. Jana, and B. Ray, “DeepTest: Automated testing of deep-neural-network-driven autonomous cars,” in Proceedings of the 40th International Conference on Software Engineering, 2018. [Online]. Available: https://dl.acm.org/doi/10.1145/3180155.3180220

[41] M. Zhang, Y. Zhang, L. Zhang, C. Liu, and S. Khurshid, “DeepRoad: GAN-based metamorphic testing and input validation framework for autonomous driving systems,” in Proceedings of the 33rd ACM/IEEE International Conference on Automated Software Engineering, 2018. [Online]. Available: https://dl.acm.org/doi/10.1145/3238147.3238187

[42] P. He, C. Meister, and Z. Su, “Structure-invariant testing for machine translation,” in Proceedings of the ACM/IEEE 42nd International Conference on Software Engineering, 2020. [Online]. Available: https://dl.acm.org/doi/10.1145/3377811.3380339

[43] Z. Sun, J. M. Zhang, M. Harman, M. Papadakis, and L. Zhang, “Automatic testing and improvement of machine translation,” in Proceedings of the ACM/IEEE 42nd International Conference on Software Engineering, 2020, pp. 974–985. [Online]. Available: https://dl.acm.org/doi/10.1145/3377811.3380420

[44] P. He, C. Meister, and Z. Su, “Testing machine translation via referential transparency,” in 2021 IEEE/ACM 43rd International Conference on Software Engineering (ICSE), 2021. [Online]. Available: https://pinjiahe.github.io/files/pdf/research/ICSE21.pdf

[45] M. T. Ribeiro, T. Wu, C. Guestrin, and S. Singh, “Beyond accuracy: Behavioral testing of NLP models with CheckList,” in Proceedings ofthe 58th Annual Meeting of the Association for Computational Linguistics. Association for Computational Linguistics, 2020, pp. 4902–4912. [Online]. Available: https://aclanthology.org/2020.acl-main.442/

[46] D. Kaushik, E. Hovy, and Z. C. Lipton, “Learning the difference that makes a difference with counterfactually-augmented data,” in

International Conference on Learning Representations, 2020. [Online]. Available: https://openreview.net/forum?id=Sklgs0NFvr

[47] J. Yi, Y. Xie, B. Zhu, E. Kiciman, G. Sun, X. Xie, and F. Wu, “Benchmarking and defending against indirect prompt injection attacks on large language models,” in Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 2025. [Online]. Available: https://dl.acm.org/doi/10.1145/3690624.3709179

[48] Q. Zhan, Z. Liang, Z. Ying, and D. Kang, “InjecAgent: Benchmarking indirect prompt injections in tool-integrated large language model agents,” in Findings of the Association for Computational Linguistics: ACL 2024, 2024, pp. 10 471–10 506. [Online]. Available: https: //aclanthology.org/2024.findings-acl.624

[49] E. Debenedetti, J. Zhang, M. Balunovic, L. Beurer-Kellner, M. Fischer, and F. Tramer, “AgentDojo: A dynamic environment\` to evaluate prompt injection attacks and defenses for LLM agents,” in Advances in Neural Information Processing Systems, vol. 37, 2024. [Online]. Available: https://proceedings.nips.cc/paper files/paper/ 2024/hash/97091a5177d8dc64b1da8bf3e1f6fb54-Abstract-Datasets and Benchmarks Track.html