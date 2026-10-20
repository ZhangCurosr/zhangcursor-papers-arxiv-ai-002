![](images/157fcb39c4b8125e62a33eb5f23e1e08d87835c34b47560c588e1e254140d096.jpg)

# Neither Over- nor Under-Constrained: Evaluating LLM Personalization under a Constraint Qualification Audit

Xuran Li<sup>1</sup>, Guanqin Zhang<sup>1</sup>, Imran Razzak<sup>1,2</sup>,

Hakim Hacid<sup>3</sup>, Eleanna Kafeza<sup>3</sup>, Hao Xue<sup>1,4</sup>, Flora D. Salim<sup>1</sup>,

<sup>1</sup>University of New South Wales,

<sup>2</sup>Mohamed bin Zayed University of Artificial Intelligence,

<sup>3</sup>The Technology Innovation Institute,

<sup>4</sup>The Hong Kong University of Science and Technology

## Abstract

Accurate evaluation of LLM personalization is hindered by current evaluation paradigms that over- or under-constrain the response space, yielding frequent false rejections and false acceptances. To reduce these false rates, we introduce ConQual-Audit, an audit-based framework that evaluates LLM personalization through a three-stage process: (1) extracting semantic constraints from user profiles and taskgrounded reference; (2) qualifying constraints by pruning unnecessary constraints and recovering missing constraints; and (3) verifying candidate responses against the audited con straints via natural language inference. Across PrefEval and LaMP, ConQual-Audit outper forms nine baselines, attaining top AUCs (0.88 and 0.90) and the lowest false rate (0.21) on the leanest constraint sets at an amortizable audit cost. Crucially, this constraint audit is portable: integrating it into ExPerT and PREF helps revise smaller, higher-quality constraint sets to reduce their evaluation false rates. All code, prompts, and evaluation scripts are released in an anonymous repository (Anonymous, 2026)

## 1 Introduction

LLMs demonstrate strong general-purpose capabilities across a wide range of tasks (Minaee et al., 2025; Niu et al., 2025). Beyond generic performance, personalization has emerged as a crucial requirement for enhancing user satisfaction and enabling context-aware interactions. Specifically, personalization refers to tailoring LLM outputs to diverse user preferences, contextual signals, and implicit intents (Zhang et al., 2025).

However, current evaluation paradigms fail to reliably evaluate LLM personalization, as they tend to either over-constrain or under-constrain the personalized response space, prematurely penalizing valid outputs or tolerating invalid ones (Figure 1). On one hand, reference-based metrics

Figure 1: Personalized Response Space.

(e.g., BLEU (Papineni et al., 2002), ROUGE (Dasgupta et al., 2024), BERTScore (Zhang et al., 2020), MoverScore (Zhao et al., 2019)) depend on proximity to rigid human references; they reject acceptable semantic variations while erroneously tolerating outputs with high surface overlap but critical personalization errors. On the other hand, LLMas-a-judge frameworks (e.g., FactScore (Min et al., 2023), GEMBA (Kocmi and Federmann, 2023), PerSE (Wang et al., 2024)) can reason over flexible outputs, but they function as black-box evaluators that lack transparent and interpretable evaluation mechanisms. While recent constraint protocols (e.g., PREF (Fu et al., 2025), ExPerT (Salemi et al., 2025)) enhance interpretability via a twostage paradigm—extracting constraints from user profiles and assessing response adherence—they remain brittle: synthesized constraints often suffer from low fidelity, manifesting as redundancy, omissions, or hallucinated requirements.

To address both over- and under-constraining problems, we introduce ConQual-Audit, an auditbased framework that establishes calibrated, interpretable boundaries over the personalized response space via a principled constraint qualification mechanism (Figure 2). ConQual-Audit operates across three modular stages: First, it extracts original semantic constraints from both task-grounded reference and user preference. Second, it refines the extracted constraints through a counterexampledriven tripartite audit: (i) deduplicating redundant criteria; (ii) pruning unnecessary constraints by constructing personalized counterexamples that violate candidate criteria yet remain valid; and (iii) recovering missing constraints by synthesizing nonpersonalized counterexamples that satisfy all active criteria. Finally, it evaluates candidate responses against the audited constraint set via a premise–hypothesis inference formulation, and unifies constraint satisfaction and response attribution via their harmonic mean to jointly balance constraint coverage and response recall.

![](images/7edaaaf1da5f627237506ab8d656bddf60f9ef66e33dc8d7e2d54ba390a1ef7e.jpg)  
Figure 2: Evalauation pipeline of ConQual-Audit framework

Empirical Verification. Extensive empirical evaluations across the PrefEval Zhao et al. (2025) and LaMP (Salemi et al., 2024) benchmarks show the effectiveness of ConQual-Audit. Our framework establishes the highest evaluation performance, achieving top AUC scores (0.88 and 0.90) and the lowest overall false rates (FR = 0.21) while operating on the most compact constraint sets (2.36 and 2.97 per query). Crucially, we show that constraint qualification is model-agnostic and highly portable: integrating our audit module into existing evaluators (ExPerT and PREF) systematically remove unnecessary constraints, restore missing ones, and significantly reduce downstream evaluation errors at an amortized audit computational cost.

The primary contributions are three-fold:

• ConQual-Audit Framework: We propose a principled, three-stage audit-based framework for LLM personalization evaluation that overcomes both over- and under-constrained evaluation regimes through semantic constraint construction, qualification, and verification.

• Counterexample-Guided Qualification: We design a tripartite audit protocol that systematically deduplicates, prunes, and recovers constraints. This mechanism yields compact, calibrated constraint sets that attain top AUCs and the lowest overall false rates on LaMP and PrefEval.

• Modularity and Portability: We demonstrate that constraint qualification is modelagnostic: augmenting external constraintbased evaluators (ExPerT and PREF) with our audit module consistently improves constraint quality and lowers downstream false rates with minimal overhead.

## 2 Problem Formulation & Related Work

Problem formulation. We formalize LLM personalization evaluation (Salemi and Zamani, 2025; Kumar et al., 2024; Zhang et al., 2025; Dong et al., 2024) as a set-membership problem. Let q be a query from a user with interaction history H, let $y ^ { * }$ be a task reference for $q ,$ and let Ω be the set of all candidate responses to q. The user and the task determine the set of acceptable responses

$$
\begin{array} { r } { \mathcal { A } ^ { \star } = \{ y \in \Omega : y \mathrm { ~ i s ~ a c c e p t a b l e ~ } \} , } \end{array}\tag{1}
$$

where a response is acceptable if it follows the user’s preferences, as evidenced by H, and meets the task requirements specified by $y ^ { \ast }$ . Evaluating a response y means deciding whether $y \in { \mathcal { A } } ^ { \star }$ Since $\mathcal { A } ^ { \star }$ is not observed, every evaluation method approximates it by an induced set $\widehat { A } ;$ existing methods differ in how they construct it (Figures 1 and 2).

Reference-based evaluation (e.g., BLEU (Papineni et al., 2002), ROUGE (Dasgupta et al., 2024), BERTScore (Zhang et al., 2020), Mover-Score (Zhao et al., 2019)) accepts responses that are similar to a set of human annotations $\mathcal { R } \subset \mathcal { A } ^ { \star }$ via a similarity function sim and a threshold τ:

$$
\widehat { \mathcal { A } } _ { \mathrm { r e f } } = \{ y \in \Omega : \operatorname* { m a x } _ { r \in \mathcal { R } } \sin ( y , r ) \geq \tau \} ,\tag{2}
$$

Although annotations serve as effective anchors, they cannot encompass all forms of personalized responses. Consequently, they often reject valid responses that differ in wording from the annotations, while accepting invalid responses that happen to share similar vocabulary (such as negated answers).

LLM-as-a-judge (e.g., GEMBA (Kocmi and Federmann, 2023), FActScore (Min et al., 2023), PerSE (Wang et al., 2024)) asks an LLM J (AI@Meta, 2024; DeepSeek-AI, 2025; Team, 2025; Team et al., 2026) to score a response given the user’s history:

$$
\widehat { A } _ { \mathrm { l l m } } = \{ y \in \Omega : J ( y ; H ) \geq \tau \} .\tag{3}
$$

LLM judges are not tied to references, but they score a response holistically, so the boundary of $\widehat { A } _ { \mathrm { l l m } }$ is implicit in the judge and can be neither inspected nor audited.

Constraint evaluation (e.g., ExPerT (Salemi et al., 2025), PREF (Fu et al., 2025)) describes $\mathcal { A } ^ { \star }$ by explicit constraints, such as rubric criteria in PREF or content aspects in ExPerT. They check responses against a constraint set C and induce:

$$
{ \widehat { A } } ( { \mathcal { C } } ) = { \big \{ } y \in \Omega : y | = c \forall c \in { \mathcal { C } } { \big \} } ,\tag{4}
$$

where $y \ne c$ denotes that $y$ satisfies the constraint c; in practice, the conjunction is relaxed to a graded score. Since the constraints specify the required content of a response rather than the exact wording, this approach accepts a variety of valid responses, and evaluation results can be traced back to the underlying constraints.

However, the reliability of constraint-based evaluation depends on the quality of the constraints themselves; existing constraint-based evaluators often lack mechanisms to review these constraints during use, which can lead to the false acceptance of invalid responses or the false rejection of valid ones due to unqualified constraints.

## 3 Constraint Quality Analysis

A constraint evaluator errs when $\widehat { A } ( \mathcal { C } )$ differs from $\mathcal { A } ^ { \star }$ (Figure 1). The difference consists of two error kinds: 1). False rejection FR(C) (i.e., the rejection of valid responses, constituting over-constraint); 2). False Acceptance FA(C) (i.e., the acceptance of invalid responses, constituting under-constraint).

$$
\operatorname { F R } ( { \mathcal { C } } ) = { \mathcal { A } } ^ { \star } \setminus { \widehat { \mathcal { A } } } ( { \mathcal { C } } ) , \operatorname { F A } ( { \mathcal { C } } ) = { \widehat { \mathcal { A } } } ( { \mathcal { C } } ) \setminus { \mathcal { A } } ^ { \star } .\tag{5}
$$

## 3.1 Necessity and Sufficiency

Ab(C) is the intersection of all constraints, to reduce over-constraint and under-constraint problems, constraints should possess these two properties:

1. Individual necessity (no false rejections). FR(C) = ∅ holds if and only if every constraint is satisfied by every valid response:

$$
y _ { \mathbf { \alpha } } = c , \forall y \in { \mathcal { A } } ^ { \star } \ \mathrm { a n d } \ \forall c \in { \mathcal { C } } .\tag{6}
$$

A constraint that some valid response violates is unnecessary; each unnecessary constraint adds its own false rejections.

2. Joint sufficiency (no false acceptances). $\mathrm { F A } ( { \mathcal { C } } ) = \emptyset$ holds if and only if the constraints together admit only acceptable responses:

$$
{ \widehat { A } } ( { \mathcal { C } } ) \subseteq A ^ { \star } .\tag{7}
$$

Unlike necessity, this property is for the whole set. A set that violates it is insufficient, because a necessary constraint that would exclude a falsely accepted response is missing.

Necessity gives $\mathcal { A } ^ { \star } \subseteq \widehat { \mathcal { A } } ( \mathcal { C } )$ and sufficiency the reverse inclusion, they together recover the acceptable set exactly.

## 3.2 Constraint Quality Impact on Evaluation

The constraint evaluators compute the satisfaction rate $s c ( y )$ and accept a response if $s c ( y ) \geq \tau ;$

$$
s c ( y ) =  \frac { | \{ c \in { \mathcal { C } } : y \} | = c \} { | { \mathcal { C } } | } } .\tag{8}
$$

Weighted criteria, as in PREF, follow an analogous continuous formulation. Let $n = | \mathcal { C } |$ and $s =$ $s c ( y )$ . Modifying C by adding a constraint $( + c ) \mathrm { o r }$ removing one (−c) perturbs the score by:

$$
\Delta s _ { - } = \frac { s - \mathbb { I } ( y \left| = c \right. ) } { n - 1 } \Delta s _ { + } = \frac { \mathbb { I } ( y \left| = c \right. ) - s } { n + 1 }\tag{9}
$$

The direction and magnitude of this perturbation depend strictly on whether y satisfies c relative to its baseline score s. For a valid response $y ^ { + } \in \mathcal A ^ { \star }$ and an invalid response $y ^ { - } \notin { \mathcal { A } } ^ { \star }$ , this establishes three theoretical links between constraint quality and evaluation error:

1. Unnecessary constraints inflate false rejections (FR(C) ↑). When a constraint c violates individual necessity (Eq. (6)), valid responses may violate it $( \mathbb { I } ( y ^ { + } \mid = c ) = 0 )$ . Adding c penalizes the valid response by $\Delta s _ { + } = - s / ( n + 1 ) \leq 0$ If $s - \Delta s _ { + } < \tau$ , this induces a false rejection. Conversely, pruning unnecessary constraints via qualification guarantees $\Delta s _ { - } = s / ( n - 1 ) \geq 0$ directly restoring valid responses above τ.

2. Insufficient sets inflate false acceptances (FA(C) ↑). If C violates joint sufficiency $\left( \operatorname { E q . } \left( 7 \right) \right)$ a necessary constraint $c ^ { \star }$ that $y ^ { - }$ violates $( \mathbb { I } ( y ^ { - } \mapsto$ $c ^ { \star } ) = 0 )$ is omitted. The absence of $c ^ { \star }$ inflates the invalid response’s score by $\Delta s _ { - } = s / ( n - 1 ) \geq 0$ Omitting necessary constraint can decrease the threshold distance for invalid responses, leading to false acceptances.

Algorithm 1: ConQual-Audit   
Input: $\overline { { q , H , y ^ { * } , y ; G , J , V } }$   
Output: Scores $\{ ( R _ { y } , P _ { y } , F 1 _ { y } ) \} _ { y \in \mathcal { V } }$   
// 1. Construction   
1 $\begin{array} { r } { { \mathcal { C } } ^ { p } \gets G ( q , H ) ; \mathcal { C } ^ { a } \gets G ( q , y ^ { * } ) } \end{array}$   
// 2. Qualification (Alg. 2)   
2 ${ \mathcal { C } } \gets \mathrm { Q u a L I F Y } ( { \mathcal { C } } ^ { p } \cup { \mathcal { C } } ^ { a } )$   
// 3. Verification   
3 foreach $y \in \mathcal { V }$ do   
4 ${ \mathcal { C } } _ { y } ^ { E } \gets \{ c \in { \mathcal { C } } : V ( y , c ) = \mathrm { E } \}$   
5 $\{ \bar { y } _ { c } ^ { E } \} _ { c \in \mathcal { C } _ { y } ^ { E } }  V ( S ( y ) , \mathcal { C } _ { y } ^ { E } )$   
6 $R _ { y }  \rvert \bar { \mathcal { C } } _ { y } ^ { E } \rvert / \lvert \mathcal { C } \rvert$   
7 $\begin{array} { r } { P _ { y } \gets \mathrm { t o } \breve { \mathrm { k } } \big ( \bigcup _ { c } y _ { c } ^ { E } \big ) / \mathrm { t o k } \big ( S ( y ) \big ) } \end{array}$   
8 $F 1 _ { y } \gets 2 P _ { y } R _ { y } / \left( P _ { y } + R _ { y } \right)$   
9 return $\{ ( R _ { y } , P _ { y } , F 1 _ { y } ) \} _ { y \in \mathcal { V } }$

## 4 The ConQual-Audit Framework

A low-quality constraint set can lead to increased false rejections and false acceptances. Therefore, it is crucial to audit constraints before constraint validation. ConQual-Audit is proposed to evaluate LLM personalization under audited constraints. As shown in ALG. 1, ConQual-Audit proceeds in three stages: (1) construction (§4.1) derives original constraints from user preference and task reference; (2) qualification (§4.2) audits and improves necessity and sufficiency of the original constraints (§3.1); and (3) verification (§4.3) checks response against the audited constraints and computes evaluation result scores.

## 4.1 Constraint Construction

Each constraint is designed as a declarative proposition about the response, and $y \ne c$ holds when y entails c. A required c is stated as a positive assertion (e.g., “The response recommends whole-food sources ofprotein.”), which is entailed only if the response provides the required content. A prohibited c is stated as a negative assertion $( { \mathrm { e . g . , ~ } } ^ { \mathrm { \scriptsize  } } T h e$ response does not recommend protein powder.”), which is entailed when the response omits or explicitly rejects the corresponding content. Therefore, the constraints can be verified by natural language inference, during the constraint verification stage.

As shown in ALG. 1, preference constraints and answer constraints are constructed by a generator G (line 1). Specifically, G extracts the user’s interests from H, and derives one constraint for each thing the user prefers or avoids among the interests that apply to the query $q .$ For the answer constraint, G decomposes task-grounded reference $y ^ { * }$ into atomic requirements, covering what an valid answer must provide and what it must avoid.

## 4.2 Constraint Qualification

Constraint qualification audits individual necessity (Eq. 6) and joint sufficiency (Eq. 7). Since $\mathcal { A } ^ { \star }$ is not observed, it tests them with counterexamples (ALG. 2): the generator $G$ produces a response $\tilde { y }$ subject to preset constraints; the verifier $V$ confirms that the constraints are satisfied; and the discriminator J determines whether $\tilde { y }$ belongs to $\mathcal { A } ^ { \star }$ by verifying it against user preferences or task reference expectations. If no valid counterexample is found in K attempts, the constraints is unchanged.

Specifically, Alg. 2 first applies a duplicated constraints check in line 1. If two constraints are duplicates, $c _ { i } \equiv c _ { j } ,$ the later one is removed. This does not change $\hat { \cal A } ( \mathcal { C } )$ but simplifies the constraints for following audits. Lines 2–4 audit unnecessary constraints (lines 2–4). For each $c , G$ generates a $\tilde { y }$ that satisfies every other constraint but violates $c .$ If J accepts it, then $\tilde { y } \in \mathcal A ^ { \star }$ and ${ \tilde { y } } \nVDash c ,$ , so c violates necessity and is removed. Each constraint is tested against those still kept. Finally, lines 5–8 imply missing constraints audit. For the task reference and user preference in turn, $G$ generates a $\tilde { y }$ that satisfies every constraint but breaks that exceptions. If J rejects it, then $\tilde { y } \in \operatorname { F A } ( \mathcal { C } )$ , which violates sufficiency. G then proposes new constraints from the broken requirements and ${ \tilde { y } } ,$ , and a proposal is added to exclude $\tilde { y }$ from other valid responses.

```latex
Algorithm 2: Constraint Qualification.
Input: Constraints C; q, H, y<sup>∗</sup> and G, J, V
Output: Qualified constraints C
// Duplicated
1 foreach $i < j$ with $c _ { i } \equiv c _ { j }$ do
${ \mathcal { C } } \gets { \mathcal { C } } \backslash$ {c<sub>j</sub>}
// Unnecessary (necessity)
2 foreach $c \in { \mathcal { C } } ,$ while $| { \mathcal { C } } | > 1$ do
3 $\tilde { y } \gets \mathrm { A D V E R S E } ( \mathcal { C } \setminus \{ c \} , \{ c \} , \cdot )$
4 if $\tilde { y } \neq  { \perp }  { \wedge } J ( \tilde { y } ) =  { \operatorname { P A S S } }$ then
${ \mathcal { C } } \gets { \mathcal { C } } \setminus \{ c \}$
// Missing (sufficiency)
5 foreach $r \in \{ \mathrm { R E F } , \mathrm { P R E F } \}$ do
6 $\tilde { y } \gets \mathsf { A }$ DVERSE $( \mathcal { C } , \emptyset , r )$
7 $\mathbf { i f } \tilde { y } \neq \perp \land J ( \tilde { y } ) = \mathrm { F A I I }$ L then
8 $\mathcal { C } \gets \mathcal { C } \cup \mathrm { U P D A T E } ( \mathcal { C } , \tilde { y } )$
9 return C
```

## 4.3 Constraint Verification and Metrics

Given the qualified set C, V checks a response in two steps in ALG. 1. (1) Response verification (line 4). With y as the premise and each constraint as a hypothesis, V labels every constraint as EN-TAILMENT, CONTRADICTION or NEUTRAL. The satisfied constraints are $\mathcal { C } _ { y } ^ { E } = \{ c \in \mathcal { C } : y \vert = c \}$ (2) Evidence attribution (line 5). For each $c \in \mathcal { C } _ { y } ^ { E }$ V identifies the sentences $y _ { c } ^ { E } \subseteq S ( y )$ that support it, where $S ( y )$ are the sentences of y. Then, lines 6–8 compute the evaluation results:

$$
\begin{array} { c } { { \displaystyle R _ { y } = \frac { | { \mathcal C } _ { y } ^ { E } | } { | { \mathcal C } | } , \quad P _ { y } = \frac { \mathrm { t o k } \big ( \bigcup _ { c \in { \mathcal C } _ { y } ^ { E } } y _ { c } ^ { E } \big ) } { \mathrm { t o k } \big ( S ( y ) \big ) } , } } \\ { { \displaystyle F 1 _ { y } = \frac { 2 P _ { y } R _ { y } } { P _ { y } + R _ { y } } , } } \end{array}\tag{10}
$$

where tok(·) counts word tokens.

The constraint recall $R _ { y }$ is the score $s c ( y )$ of Eq. (8): it measures how many constraints the response meets, and a response is accepted if $R _ { y } \ge \tau$ The response precision $P _ { y }$ measures the share of the response devoted to these satisfied constraints and thus penalizes content that serves none of them. $F 1 _ { y }$ is a balance between response precision and constraint recall. Further algorithm details about the counterexample generation and missing constraint update are referred to Appendix A.

## 5 Experimental Evaluation

We evaluate ConQual-Audit by addressing three core research questions:

RQ1 (Performance & Cost)(§5.2): Does ConQual-Audit minimize evaluation false rates more effectively than reference-based, promptbased LLM judges, and constraint evaluation baselines, and what is its computational trade-off?

RQ2 (The Generalizability of Constraint Audit)(§5.3): Can ConQual’s qualification process be generalized as a model-agnostic filter to audit the constraints of other evaluators, and does it reduce their downstream decision errors?

RQ3 (Case Study)(§5.4): How does ConQual-Audit reshape decision boundaries to reflect both user preferences and task-grounded references?

## 5.1 Experimental Setup

Benchmarks. We evaluate on two benchmarks: (1) LaMP (Salemi et al., 2024) (tasks 1, 2, and 4; 3,000 responses per task), testing semantic discrimination against surface form by evaluating 10 paraphrased positive answers, 10 negated answers, and 10 incorrect answers per user across 100 users; and (2) PrefEval (Zhao et al., 2025) (200 responses across 100 users), testing explicit preference compliance via paired Qwen3-14B (Team, 2025) generations conditioned with and without the user’s preference.

Baselines. We compare against nine baselines across three families: (1) reference-based metrics on LaMP (BLEU-4 (Papineni et al., 2002), ROUGE-L (Lin, 2004), BERTScore (Zhang et al., 2020), MoverScore (Zhao et al., 2019)); (2) LLM judges (GEMBA (Kocmi and Federmann, 2023), PerSE (Wang et al., 2024), FActScore (Min et al., 2023)); and (3) constraint evaluators (Ex-PerT (Salemi et al., 2025), PREF (Fu et al., 2025)). We also evaluate audited variants (ExPerT-Audit, PREF-Audit) via ConQual’s qualification stage (§4.2) to test constraint audit generalizability.

Metrics and Implementation. Responses are accepted at threshold $\tau \geq 0 . 5$ . We report False Rejection Rate (FRR), False Acceptance Rate (FAR), their mean False Rate (FR), AUC, mean score margins, number of evaluated constraints, and end-toend token costs. To isolate methodological gains from model capability, all LLM judges and Con-Qual modules (G, V, J) use Qwen3-8B (Team, 2025). ConQual-Audit uses $K = 2$ counterexample attempts, one gap-repair round per constraints, and at most three new constraints per repair.

Table 1: Results averaged over LaMP-1, LaMP-2 and LaMP-4. Best in bold, second underlined.
<table><tr><td>Method</td><td>Pos.↑</td><td>Neg.↓</td><td>AUC↑</td><td>FRR↓</td><td>FAR↓</td><td>False Rate↓</td><td>Tokens↓</td><td>#Cons.</td></tr><tr><td>BLEU-4</td><td>0.35</td><td>0.16</td><td>0.79</td><td>0.65</td><td>0.14</td><td>0.39</td><td>0.0</td><td>一</td></tr><tr><td>ROUGE-L</td><td>0.65</td><td>0.46</td><td>0.70</td><td>0.22</td><td>0.46</td><td>0.34</td><td>0.0</td><td>一</td></tr><tr><td>BERTScore</td><td>0.95</td><td>0.92</td><td>0.74</td><td>0.00</td><td>1.00</td><td>0.50</td><td>28.3</td><td>一</td></tr><tr><td>MoverScore</td><td>0.65</td><td>0.57</td><td>0.76</td><td>0.08</td><td>0.79</td><td>0.44</td><td>29.0</td><td>一</td></tr><tr><td>GEMBA</td><td>0.52</td><td>0.30</td><td>0.67</td><td>0.47</td><td>0.26</td><td>0.36</td><td>198.3</td><td></td></tr><tr><td>PerSE</td><td>0.41</td><td>0.26</td><td>0.66</td><td>0.62</td><td>0.20</td><td>0.41</td><td>1,212.6</td><td>一</td></tr><tr><td>FActScore</td><td>0.48</td><td>0.53</td><td>0.48</td><td>0.52</td><td>0.53</td><td>0.53</td><td>1,506.6</td><td>一</td></tr><tr><td>ExPerT</td><td>0.07</td><td>0.00</td><td>0.54</td><td>0.92</td><td>0.01</td><td>0.46</td><td>3,027.0</td><td>3.47</td></tr><tr><td>PREF</td><td>0.61</td><td>0.31</td><td>0.84</td><td>0.26</td><td>0.25</td><td>0.25</td><td>649.1</td><td>5.43</td></tr><tr><td>ConQual-Audit</td><td>0.77</td><td>0.20</td><td>0.90</td><td>0.13</td><td>0.29</td><td>0.21</td><td>1,827.4</td><td>2.97</td></tr></table>

## 5.2 ConQual-Audit: Minimizing False Rate

ConQual-Audit achieves the lowest overall false rate across both benchmarks. As shown in Tables 1 and 2, ConQual-Audit reduces overall classification error to FR = 0.21 on both datasets—a 16% reduction over the strongest baseline on LaMP (PREF: 0.25) and a 28% reduction on PrefEval (PerSE: 0.29)—while dominating in AUC (0.90 and 0.88). This improvement stems from calibrated error control. On LaMP, ConQual-Audit halves the false rejection rate of PREF (FRR = 13% vs. 26%) at a comparable false acceptance rate (FAR = 29% vs. 25%), while evaluating roughly half as many constraints (2.97 vs. 5.43). On PrefEval, it reduces false acceptances(FAR = 28% vs. 45% for PerSE) without inflating false rejections (FRR = 15% vs. 13%). Crucially, ConQual-Audit widens the score margin between positive and negative responses to 0.57 on both benchmarks—nearly double any competing baseline (≤ 0.30 on LaMP, ≤ 0.32 on PrefEval). Because audited constraints isolate necessary and sufficient conditions, valid paraphrases systematically satisfy them while invalid responses directly violate them.

Reference-based metrics confuses surface similarity with compliance. The acceptance sets ${ \widehat { A } } _ { \mathrm { r e f } }$ (Eq. (2)) exhibit severe failures: lexical metrics penalize benign stylistic rephrasing, falsely rejecting 65% of valid answers on LaMP (BLEU-4 FRR = 0.65), whereas semantic embedding metrics mistake embedding proximity for alignment, accepting 100% of negated or incorrect responses (BERTScore FAR = 1.00, MoverScore 0.79).

Prompt-based LLM judges misplace the decision boundary. Unconstrained LLM judges relying on implicit boundaries $\widehat { A } _ { \mathrm { l l m } }$ (Eq. (3)) fail to calibrate across task distributions. While GEMBA and PerSE show reasonable ranking capability on

PrefEval (AUC 0.81–0.82), they set overly permissive boundaries that accept 84% and 45% of noncompliant outputs (FAR). On LaMP, this behavior inverts into extreme conservatism, falsely rejecting 47% and 62% of correct answers (FRR). Decomposing responses into unweighted atomic facts (FActScore) fails entirely, hovering near chance on both benchmarks (AUC 0.48–0.52).

Constraint criteria violate necessity and sufficiency. Constraint baselines introduce high false rates by generating criteria that either over- or under-specify the solution space. ExPerT conditions constraints strictly on reference tokens; because gold references are often abstract criteria (PrefEval) or brief sentences (LaMP), valid candidate answers naturally omit arbitrary reference details. Its constraint sets thus violate necessity (Eq. (6)), falsely rejecting 92% (LaMP) and 97% (PrefEval) of valid outputs (FRR). Conversely, PREF constructs generic rubric criteria easily satisfied by standard generation quality regardless of preference adherence (mean negative score 0.81); its constraints violate sufficiency (Eq. (7)), accepting every negative response on PrefEval (FAR = 1.00). Both evaluate more than twice as many constraints as ConQual-Audit (5.11–8.23 vs. 2.36), yet yield the highest false rates (FR ≥ 0.50).

Inference cost and amortized verification. On PrefEval, ConQual-Audit consumes 12.6k tokens per response end-to-end, higher than single-prompt judges (GEMBA: 0.7k, PerSE: 1.9k), but one-third of atomic-decomposition methods (FActScore: 37.4k) and substantially less than auditing unvetted external criteria (ExPerT-Audit: 24.8k, PREF-Audit: 16.0k). Over 80% of this compute (∼ 10.1k tokens) is front-loaded into one-time constraint synthesis and qualification per instruction. The runtime verification step requires only 2.5k tokens per candidate response, readily amortizing across multiple model outputs (N > 1). On LaMP, end-toend cost scales down to 1.8k tokens per response, falling between PREF (0.6k) and ExPerT (3.0k). This focused compute investment directly delivers the lowest (FR = 0.21 across both benchmarks.

Table 2: Results on PrefEval. Best in bold, second underlined.
<table><tr><td>Method</td><td>Pos.↑</td><td>Neg.↓</td><td>AUC↑</td><td>FRR↓</td><td>FAR↓</td><td>False Rate↓</td><td>Tokens↓</td><td>#Cons.</td></tr><tr><td>GEMBA</td><td>0.90</td><td>0.68</td><td>0.82</td><td>0.03</td><td>0.84</td><td>0.44</td><td>670.1</td><td>一</td></tr><tr><td>PerSE</td><td>0.87</td><td>0.55</td><td>0.81</td><td>0.13</td><td>0.45</td><td>0.29</td><td>1,889.2</td><td>一</td></tr><tr><td>FActScore</td><td>0.60</td><td>0.58</td><td>0.52</td><td>0.35</td><td>0.62</td><td>0.49</td><td>37,430.2</td><td>1</td></tr><tr><td>ExPerT</td><td>0.08</td><td>0.06</td><td>0.53</td><td>0.97</td><td>0.04</td><td>0.50</td><td>9,663.9</td><td>8.23</td></tr><tr><td>PREF</td><td>0.92</td><td>0.81</td><td>0.79</td><td>0.02</td><td>1.00</td><td>0.51</td><td>1,641.9</td><td>5.11</td></tr><tr><td>ExPerT-Audit</td><td>0.06</td><td>0.09</td><td>0.45</td><td>0.94</td><td>0.04</td><td>0.49</td><td>24,790.6</td><td>6.67</td></tr><tr><td>PREF-Audit</td><td>0.91</td><td>0.74</td><td>0.78</td><td>0.03</td><td>0.89</td><td>0.46</td><td>16,006.7</td><td>3.64</td></tr><tr><td>ConQual-Audit</td><td>0.80</td><td>0.23</td><td>0.88</td><td>0.15</td><td>0.28</td><td>0.21</td><td>12,628.5</td><td>2.36</td></tr></table>

## 5.3 Generalization of Constraint Audit

Because ConQual’s audit (§4.2) operates over the explicit constraints, it can function as a modelagnostic filter for other constraint-based evaluators. To test its portability, we apply the audit to the response aspects generated by ExPerT and the rubric criteria produced by PREF, re-scoring responses with each method’s native verifier under both original and audited constraints (Table 2). Table 3 details the resulting constraint reductions, classification AUC, and false rates.

Table 3: Auditing the constraints of ExPerT and PREF.
<table><tr><td rowspan="2"></td><td colspan="2">ExPerT</td><td colspan="2">PREF</td></tr><tr><td>Orig.</td><td>Audit</td><td>Orig.</td><td>Audit</td></tr><tr><td>#Cons.</td><td>8.23</td><td>6.67</td><td>5.11</td><td>3.64</td></tr><tr><td>Removed / added</td><td>1.78 / 0.22</td><td></td><td>1.51 / 0.05</td><td></td></tr><tr><td>AUC↑</td><td>0.53</td><td>0.45</td><td>0.82</td><td>0.78</td></tr><tr><td>FRR↓</td><td>0.96</td><td>0.94</td><td>0.02</td><td>0.03</td></tr><tr><td>FAR↓</td><td>0.04</td><td>0.04</td><td>1.00</td><td>0.89</td></tr><tr><td>FR↓</td><td>0.50</td><td>0.49</td><td>0.51</td><td>0.46</td></tr><tr><td>Reproduced</td><td>99.8%</td><td></td><td>77%</td><td></td></tr></table>

Eliminating false constraints can reduce the false rate. The audit systematically identifies and removes unnecessary or over-constraint constraints: it prunes 1.78 of ExPerT’s 8.23 aspects (−19%) and 1.51 of PREF’s 5.11 criteria (−29%), while rarely adding new ones (+0.22 and +0.05, respectively). Operating on these audited constraint sets reduces overall classification error: the false rate (FR) drops from 0.50 to 0.49 for ExPerT and from 0.51 to 0.46 for PREF. For PREF, this improvement directly addresses its primary failure mode identified in Table 2, reducing its false acceptance rate (FAR) down from 1.00 to 0.89 while keeping false rejections at a very low level (FRR: 0.02 → 0.03).

Decouple constraint quality from evaluation randomness. To isolate the intrinsic value of constraint pruning from downstream evaluator noise, we analyze a counterfactual setting where verifier decisions on unchanged constraints remain fixed. Under this controlled ablation, removing audited constraints strictly improves classification performance: PREF’s AUC increases from 0.82 to 0.84 (FR drops to 0.45), while ExPerT’s FR falls from 0.50 to 0.47 at a steady 0.53 AUC. In the end-to-end re-evaluation reported in Table 3, these gains are masked by test-retest instability in the underlying LLM judges: PREF’s verifier reproduces only 77% of its decisions on identical criteria across runs, introducing a baseline drift of ±0.03 AUC that accounts for the nominal drop to 0.78. Thus, while the audit effectively cleans the constraint space, end-to-end performance remains constrained by the variance of third-party scoring backends.

Inference overhead and ConQual’s efficiency advantage. The primary trade-off of the audit process is computational cost. Because counterexample generation and verification consume 5–6k tokens per constraint, overhead scales linearly with the size of the unvetted constraint set. Auditing Ex-PerT (8.7 aspects per response) and PREF (5.1 criteria per prompt) inflates end-to-end inference to 24.8k and 16.0k tokens per response, respectively—overhead that is only marginally offset by downstream scoring speedups (8%–25% fewer verification tokens). By contrast, ConQual constructs and audits a compact set of only 3.9 global constraints once per question. Consequently, ConQual-Audit achieves higher accuracy (AUC 0.88, FR 0.21) at lower total compute (12.6k tokens). The audit thus functions as an effective off-the-shelf refiner for third-party criteria, but delivers its best cost-accuracy profile when integrated into Con-Qual’s unified pipeline.

![](images/8b59488b52efa0bb5fc4fe6ce9797bedac7228c6b955799fb59fbb5b9b94ffd1.jpg)  
Figure 3: ConQual-Audit. (1) The generator G derives one preference constraint and three answer constraints from the user preference and the truth-grounded reference (if verify directly, they accept the negative response $( R = 2 / 4 = \tau ) )$ . (2) Qualification prunes ac\_3, which the reference makes optional, and finds no duplicate and no gap. (3) The verifier V checks responses against the qualified constraint set: the positive response is accepted and the negative one rejected.

## 5.4 ConQual-Audit: Case Study

Figure 3 illustrates a case study from PrefEval benchmark of ConQual-Audit. How it reshapes the decision boundary to jointly enforce user-specified preferences and task-grounded references.

The user specifies a negative preference against ’physical sports ’in team-building activities, while the gold reference suggests ’non-physical alternatives’, such as ’problem-solving exercises’ or ’creative collaborations’. The construction produces four candidate constraints: a user-specified preference against physical sports (pc\_1) with three task-grounded reference criteria ( textttac\_1–ac\_3).

Without constraint qualification, an invalid candidate proposal for a ’Team Sports Day’ satisfied two constraints (ac\_2, ac\_3) but violated two other sports-related constraints (pc\_1, ac\_1), and could be thus falsely accepted. This problem is resolved by ConQual-Audit qualification with counterexamples: because omitting creative collaborations (ac\_3) leaves a response fully acceptable, ac\_3 is pruned as unnecessary. Evaluated on the qualified three-constraint set (pc\_1, ac\_1, ac\_2), the invalid candidate is identified and rejected, while valid candidates are correctly accepted.

## 6 Conclusion

In this work, we presented ConQual-Audit, an audit-based framework for evaluating LLM personalization through semantic constraint construction, qualification, and verification. ConQual-Audit synthesizes candidate constraints grounded jointly in user preferences and task references, systematically auditing them for individual necessity and joint sufficiency, pruning unnecessary constraints to minimize false rejections while repairing missing constraints to prevent false acceptances. By evaluating candidate responses via the harmonic mean of constraint satisfaction and response attribution, ConQual-Audit establishes a calibrated decision boundary that resolves both over- and underconstrained failure modes.

Empirical evaluations across PrefEval and LaMP show that ConQual-Audit achieves state-of-the-art performance, attaining top AUC scores (0.88 and 0.90) and the lowest overall false rate (FR = 0.21 on both datasets) while operating on the most compact constraint sets (2.36 and 2.97 per query). Although constraint qualification requires an upfront token investment, this cost readily amortizes during runtime verification across multiple candidate generations. Furthermore, our case study demonstrates that qualification structurally realigns the decision boundary: pruning unqualified constraint details prevents auxiliary criteria from diluting core constraints and triggering false acceptances.

Finally, our findings confirm that constraint qualification is model-agnostic and portable: applying the audit to existing evaluators (ExPerT and PREF) systematically refines their constraint sets and consistently lowers downstream false rates. These results demonstrate that unvetted constraint extraction routinely dilutes explicit user preferences with superficial task artifacts, establishing systematic constraint auditing as a principled foundation for reliable personalization evaluation.

## Acknowledgments

AI tools were used solely for language refinement. All scientific contributions are authors’ original work. This research is supported by the Technology Innovation Institute, Abu Dhabi, UAE.

## Limitations

ConQual-Audit depends on the models that play the generator, the verifier and the judge. In our experiments a single Qwen3 model fills all three roles, so their errors are correlated. Qualification is also costly. It raises the cost per PrefEval response from 3.4k to 12.6k tokens end to end, and ConQual-Audit has the highest online time of the evaluators we compare (27 s per response). The framework needs a reference answer and an explicit preference to construct constraints, so it does not apply to open-ended requests that lack them. It scores a response by the share of satisfied constraints, all weighted equally, at a fixed threshold. Finally, our evaluation covers English benchmarks with constructed positive and negative responses. Results may differ for other domains, languages and response distributions.

## References

AI@Meta. 2024. Llama 3 model card. arXiv preprint.

Anonymous. 2026. ConQual-Audit: Anonymous code repository for reviewer reproduction. https://anon ymous.4open.science/r/ConQual-Audit-257B/. Anonymous GitHub repository.

Sourish Dasgupta, Ankush Chander, Parth Borad, Isha Motiyani, and Tanmoy Chakraborty. 2024. Perseval: Assessing personalization in text summarizers. Preprint, arXiv:2407.00453.

DeepSeek-AI. 2025. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. Preprint, arXiv:2501.12948.

Yijiang River Dong, Tiancheng Hu, and Nigel Collier. 2024. Can llm be a personalized judge? Preprint, arXiv:2406.11657.

Xiao Fu, Hossein A. Rahmani, Bin Wu, Jerome Ramos, Emine Yilmaz, and Aldo Lipani. 2025. Pref: Reference-free evaluation of personalised text generation in llms. Preprint, arXiv:2508.10028.

Tom Kocmi and Christian Federmann. 2023. Large language models are state-of-the-art evaluators of translation quality. In Proceedings of the 24th Annual Conference ofthe European Associationfor Machine Translation, pages 193–203, Tampere, Finland. European Association for Machine Translation.

Ishita Kumar, Snigdha Viswanathan, Sushrita Yerra, Alireza Salemi, Ryan A. Rossi, Franck Dernoncourt, Hanieh Deilamsalehy, Xiang Chen, Ruiyi Zhang, Shubham Agarwal, Nedim Lipka, Chien Van Nguyen, Thien Huu Nguyen, and Hamed Zamani. 2024. Longlamp: A benchmark for personalized long-form text generation. Preprint, arXiv:2407.11016.

Chin-Yew Lin. 2004. ROUGE: A package for automatic evaluation of summaries. In Text Summarization Branches Out, pages 74–81, Barcelona, Spain. Association for Computational Linguistics.

Sewon Min, Kalpesh Krishna, Xinxi Lyu, Mike Lewis, Wen-tau Yih, Pang Koh, Mohit Iyyer, Luke Zettlemoyer, and Hannaneh Hajishirzi. 2023. FActScore: Fine-grained atomic evaluation of factual precision in long form text generation. In Proceedings ofthe 2023 Conference on Empirical Methods in Natural Language Processing, pages 12076–12100, Singapore. Association for Computational Linguistics.

Shervin Minaee, Tomas Mikolov, Narjes Nikzad, Meysam Chenaghlu, Richard Socher, Xavier Amatriain, and Jianfeng Gao. 2025. Large language models: A survey. Preprint, arXiv:2402.06196.

Qian Niu, Junyu Liu, Ziqian Bi, Pohsun Feng, Benji Peng, Keyu Chen, Ming Li, Lawrence KQ Yan, Yichao Zhang, Caitlyn Heqi Yin, Cheng Fei, Tianyang Wang, Yunze Wang, Silin Chen, Ming Liu, Ziyuan Qin, Riyang Bao, Xinyuan Song, and Zekun Jiang. 2025. Large language models and cognitive science: A comprehensive review of similarities, differences, and challenges. Preprint, arXiv:2409.02387.

Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. 2002. Bleu: a method for automatic evaluation of machine translation. In Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics, pages 311–318, Philadelphia, Pennsylvania, USA. Association for Computational Linguistics.

Alireza Salemi, Julian Killingback, and Hamed Zamani. 2025. Expert: Effective and explainable evaluation of personalized long-form text generation. In Proceedings ofthe Conference.

Alireza Salemi, Sheshera Mysore, Michael Bendersky, and Hamed Zamani. 2024. Lamp: When large language models meet personalization. Preprint, arXiv:2304.11406.

Alireza Salemi and Hamed Zamani. 2025. Lamp-qa: A benchmark for personalized long-form question answering. Preprint, arXiv:2506.00137.

Falcon LLM Team, Iheb Chaabane, Puneesh Khanna, Suhail Mohmad, Slim Frikha, Shi Hu, Abdalgader Abubaker, Reda Alami, Mikhail Lubinets, Mohamed El Amine Seddik, and Hakim Hacid. 2026. Falconh1r: Pushing the reasoning frontiers with a hybrid model for efficient test-time scaling. Preprint, arXiv:2601.02346.

Qwen Team. 2025. Qwen3 technical report. Preprint, arXiv:2505.09388.

Danqing Wang, Kevin Yang, Hanlin Zhu, Xiaomeng Yang, Andrew Cohen, Lei Li, and Yuandong Tian. 2024. Learning personalized alignment for evaluating open-ended text generation. In Proceedings

of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 13274–13292, Miami, Florida, USA. Association for Computational Linguistics.

Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q. Weinberger, and Yoav Artzi. 2020. Bertscore: Evaluating text generation with bert. Preprint, arXiv:1904.09675.

Zhehao Zhang, Ryan A. Rossi, Branislav Kveton, Yijia Shao, Diyi Yang, Hamed Zamani, Franck Dernoncourt, Joe Barrow, Tong Yu, Sungchul Kim, Ruiyi Zhang, Jiuxiang Gu, Tyler Derr, Hongjie Chen, Junda Wu, Xiang Chen, Zichao Wang, Subrata Mitra, Nedim Lipka, and 2 others. 2025. Personalization of large language models: A survey. Preprint, arXiv:2411.00027.

Siyan Zhao, Mingyi Hong, Yang Liu, Devamanyu Hazarika, and Kaixiang Lin. 2025. Do llms recognize your preferences? evaluating personalized preference following in llms. Preprint, arXiv:2502.09597.

Wei Zhao, Maxime Peyrard, Fei Liu, Yang Gao, Christian M. Meyer, and Steffen Eger. 2019. Moverscore: Text generation evaluating with contextualized embeddings and earth mover distance. Preprint, arXiv:1909.02622.

## A LLM Instruction Templates

```latex
Algorithm 3: Functions of Algorithm 2.
ADVERSE returns a response that satisfies
every constraint in A and violates every
constraint in B (or, with $B = \varnothing .$ , breaks
requirement r), or ⊥ after K attempts. UP-
DATE asks $G ,$ given $J { ' } \mathbf { s }$ explanation $J ( \tilde { y } )$
for at most three new constraints of the kind
that $r$ calls for and returns those that ex
clude y˜ (and, for answer constraints, hold
for $y ^ { \ast } )$
Input: $q , H , y ^ { * }$ and $G , J , V$ as in
Algorithm 1; attempts $K$
1 Function ADVERSE(A, B, r):
2 for $k = 1 , \ldots , K$ do
3 $\tilde { y } \gets G ( q , A , B , r )$ if
$\forall c \in A : V ( \tilde { y } , c ) = \mathbf { E }$ and
$\forall c \in B : V ( \tilde { y } , c ) \in \{ \mathbf { C } , \mathbf { N } \}$ then
return $\tilde { y }$
4 return $\perp$
5 Function $\begin{array} { r } { \mathrm { U P D A T E } ( \mathcal { C } , \tilde { y } , r ) \colon } \end{array}$
6 $M  y ^ { * } \mathbf { i f } r = y ^ { * }$ else H
// reference material of $r$
7 $\mathcal { C } ^ { + }  G ( q , \mathcal { C } , \tilde { y } , J ( \tilde { y } ) , M )$
$/ / \left| \mathscr { C } ^ { + } \right| \leq 3 .$ , kind of r
8 $\mathcal { C } ^ { + }  \{ c \in \mathcal { C } ^ { + } : V ( \tilde { y } , c ) \in \{ \mathbf { C } , \mathbf { N } \} \}$
9 if $r = y ^ { * }$ then
${ \mathcal { C } } ^ { + } \gets \{ c \in { \mathcal { C } } ^ { + } : V ( y ^ { * } , c ) = \mathrm { E } \}$
10 return $\mathcal { C } ^ { + }$
```

Adversarial generation. Both audits rest on counterexamples written by the generator $G$ (ADVERSE in Algorithm 3). G receives the query $q ,$ a set A of constraints the response must satisfy, and a target to break: either a set B of constraints (over-constraint pruning, $B = \{ c \} )$ ) or an alignment requirement $r \in \{ p , y ^ { * } \}$ (gap repair, $B = \emptyset$ , where the response must break the stated preference or disagree with the reference). It writes a complete, plausible answer of at most 200 words. The verifier V then checks the response against $A \cup B ,$ and it is used only if V labels every constraint in $A$ as entailed and every constraint in B as contradicted or neutral (a neutral label means the response omits the required content, so it does not satisfy the constraint). Otherwise G tries again, up to $K = 2$ times; if no attempt passes, ADVERSE returns ⊥ and the constraint set is left unchanged. A valid counterexample is then judged by J, which decides separately whether it is aligned with the preference and with the reference and explains each decision.

Constraint update. When J finds that a gaprepair counterexample $\tilde { y }$ satisfies every current constraint yet breaks a requirement $r ,$ the constraint set admits a misaligned response, and UPDATE closes the gap for each broken requirement. G receives $q ,$ the current constraints $\mathcal { C } , \tilde { y } , J ^ { \prime } { \mathrm { s } }$ explanation of the failure, and the material r rests on: the reference $y ^ { * }$ when y˜ disagrees with $\mathbf { i t } ,$ or the user’s stated preferences H when it breaks them. It proposes at most three new constraints of the matching kind (answer constraints from $y ^ { * }$ , preference constraints from $H )$ that $\tilde { y }$ violates, that every acceptable response would satisfy, and that do not restate a current constraint. A proposed constraint is added only if $V$ confirms that it excludes $\tilde { y }$ (contradicted or neutral); an answer constraint must also be entailed by $y ^ { * }$ itself, so that the repair never rejects the reference. Each requirement gets one repair round.