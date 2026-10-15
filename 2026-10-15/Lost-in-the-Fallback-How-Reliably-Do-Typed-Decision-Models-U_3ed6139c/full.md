# Lost in the Fallback: How Reliably Do Typed Decision Models Use Fallback Options?

Yuanbo Xie<sup>1</sup>\*, Mingli Zhu<sup>1</sup>\*, Tianyun Liu<sup>2,3</sup> Yingjie Zhang<sup>2,3</sup>, Haixiao Xiong<sup>1†</sup>, Tingwen Liu<sup>2,3†</sup> <sup>1</sup>System Security Team, Li Auto Inc, China

<sup>2</sup>Institute of Information Engineering, Chinese Academy of Sciences, China   
<sup>3</sup>School of Cyber Security, University of Chinese Academy of Sciences, China {xieyuanbo,zhumingli,xionghaixiao}@lixiang.com {liutianyun,zhangyingjie2023,liutingwen}@iie.ac.cn

## Abstract

Typed decision models (TDMs) are increasingly used in applications such as request rout ing and model selection, making reliable decisions critical to system performance. Fallback options accommodate categories outside regular choices, enabling deferral to human experts or more capable models. While recent works have examined adversarial vulnerabilities in TDMs, their ability to follow fallback rules in non-adversarial settings remains underexplored. We introduce the Paired Fallback Robustness (PFR) Test to evaluate two aspects of fallback reliability: whether models correctly switch between regular and fallback options, and whether they make consistent decisions under logically equivalent definitions of the same rule. We evaluate open-source decision models on three benchmarks, with controlled comparisons across architectures and parameter counts. We find that correct recognition does not reliably transfer to correct fallback options, and errors persist even when the correct category is explicitly supplied. Moreover, logically equivalent formulations of the same fallback rule can also produce substantial disagreement in model decisions. These findings reveal a crucial reliability gap in current TDMs: fallback mechanisms designed to improve reliability can instead introduce errors. It highlights faithful execution of fallback rules and consistency under equivalent definitions as essential requirements for reliable typed decision-making.<sup>1</sup>

## 1 Introduction

Many software tasks, such as request routing (Ong et al., 2025) and tool selection (Schick et al., 2023), require choosing among predefined actions rather than generating free-form responses. Typed decision models (TDMs), such as TypeSafe’s Jev (Almeida, 2026), are designed to produce fast, typesafe decisions with associated probabilities and confidence estimates. However, in practical deployments, some requests fall outside the scope of predefined actions, while others require reasoning or expertise beyond the decision model’s capabilities. Fallback options provide a mechanism for handling these cases by redirecting requests to alternative processing paths. For example, an address-change request can be routed to a dedicated service when available, or to general support through a fallback otherwise (Figure 7a). Similarly, a fast System 1 controller can select Defer to invoke a more capable System 2 reasoning model (Chen et al., 2024) (Figure 7b). In both cases, the fallback provides an alternative handling path, but the model must correctly determine when to use it. Failing to select fallback can bypass necessary handling, whereas invoking it unnecessarily introduces additional cost or latency.

![](images/c46dfe297dd8799aaf400fea9c0401c0ef867c548d34ccd74e27a3152e578fbd.jpg)  
Figure 1: Correct recognition does not ensure reliable fallback selection. On Intern/TREC50 inputs correctly classified in a separate task: (1) accuracy drops on fallback reassignment; (2) equivalent definitions yield different accuracy (hatched gap). Regular/Fallback denotes the required output. Union lists its members, while Complement excludes regular categories.

Recent studies have examined adversarial vulnerabilities in TDMs, including adversarial attacks and prompt injection (Hu et al., 2026; Wu and Lim,

2026). Concurrent work has investigated rejection behavior in arithmetic reasoning and candidate coverage (Zhong et al., 2026; Lu and Wu, 2026), as well as adherence to explicit option definitions (Sun et al., 2026). Despite these advances, a fundamental question remains unexplored:

How reliably do TDMsfollowfallback rules under non-adversarial conditions?

To answer this question, we introduce the Paired Fallback Robustness (PFR) Test, which evaluates two complementary requirements of fallback reliability. Correct reassignment examines whether models correctly switch from a regular option to fallback when the required assignment of the same input changes, testing their ability to faithfully execute fallback rules. Definition consistency examines whether models preserve their decisions under logically equivalent formulations of the same fallback rule, testing whether decisions depend on the rule’s meaning rather than its wording. Together, these paired comparisons assess whether models respond correctly to changes in the required assignment while remaining consistent when the underlying rule is unchanged.

Across three benchmarks, we find that even correctly recognized inputs can become lost during fallback reassignment. For example, on TREC50, Intern’s accuracy drops from 97.9% to 23.1% when switching from regular to fallback assignment (Figure 1). These failures persist even when the correct category is explicitly supplied, including in matched comparisons across different model architectures. Moreover, logically equivalent formulations of the same fallback rule can lead to substantial disagreement in individual decisions, even when aggregate accuracy remains nearly unchanged. Together, these findings expose a critical reliability gap in current TDMs: fallback mechanisms intended to ensure appropriate handling can themselves become sources of error, even when the input category is correctly identified.

Our contributions are summarized as follows:

• We identify and systematically study an underexplored reliability problem in TDMs: models may fail to follow fallback rules even when valid options and the necessary information are available, without adversarial inputs.

• We introduce the Paired Fallback Robustness (PFR) Test, a systematic framework for evaluating fallback reliability through paired comparisons on the same inputs.

• Across diverse benchmarks, model architectures, and parameter scales, we show that correct category information and aggregate accuracy alone are insufficient to establish reliable fallback behavior.

## 2 Preliminaries

Typed choices. We study categorical Choice interfaces. A request contains an input x and a finite option set O with natural-language selection criteria. The model returns an option $\hat { o } \in \mathcal { O }$ and associated scores; the application executes the corresponding action. Returning an allowed option ensures type validity, but decision correctness also requires satisfying its selection criterion.

Fallback options. A fallback option provides an alternative handling path when regular choices are unavailable or unsuitable. Its criterion is application-defined, for example, Other may handle requests without a dedicated service, while Defer may invoke a more capable model under a specified quality–cost policy.

Category-to-output assignment. We use category coverage to make fallback rules directly testable. Let U be the complete category inventory and $y \in \mathcal { U }$ the input’s true category. Categories in $S \subsetneq u$ have regular options, while fallback covers $\mathcal { R } = \mathcal { U } \backslash S .$ Representing each output by its category set, the required assignment is

$$
g _ { S } ( y ) = { \left\{ \begin{array} { l l } { \{ y \} , } & { y \in S , } \\ { \mathcal { R } , } & { y \not \in S . } \end{array} \right. }\tag{1}
$$

All category definitions are provided, and each input has exactly one valid destination. Gold labels are used to construct and score assignments but are disclosed to the model only in the categoryinformation diagnostic. If $\hat { g }$ is the category set represented by $\hat { o } ,$ correctness means $\hat { g } = g _ { S } ( y )$ Changing S can change the required output for the same input without changing its category. This tests fallback reassignment without introducing unknown categories or leaving inputs without valid destinations.

Equivalent definitions. We consider two logically equivalent ways to define fallback. Union (U) explicitly lists the fallback categories in R, while Complement (C) defines fallback by excluding the regular categories S from U. Both specify exactly the same category-to-output assignment.

![](images/06893c2b3266b8d1defd9ea14fc4d85972845ebc1151ca9bab7a1f6fc6949bbb.jpg)  
Figure 2: PFR crosses two assignments with two equivalent definitions. Left: without fallback, inputs outside the regular categories have no valid option. (a) Moving the gold category into fallback changes the required output. (b) U/C preserve the required output within each assignment; gold-fallback is shown. Input, inventory and option count stay fixed. Left: missing coverage motivates fallback. Letters are illustrative categories; checks mark expected outputs and gold labels are evaluation annotations.

## 3 Does Category Recognition Transfer to Fallback Selection?

We test four 4B choice-selection configurations: Intern-Decision (InternLM Team, 2026), Kev (Palmer, 2026b), JevK5 (Serikbay, 2026), and SemIf (TheoLeeCJ, 2026). All receive identical requests with fifteen regular options and one fallback. Main evaluation uses class-balanced CLINC150 (450) (Larson et al., 2019) and BANKING77 (385) (Casanueva et al., 2020) subsets, along with the full TREC50 test split (500) (Li and Roth, 2002). Unless specified otherwise, diagnostic and development panels each use 128 additional texts per dataset. The main sets measure performance across categories, while the diagnostic panels support repeated controlled comparisons on the same inputs. Sampling and model interfaces are introduced in Appendix A.

## 3.1 Paired Fallback Robustness Test

PFR constructs four requests per input by combining two assignments with two logically equivalent definitions (Figure 2). The input, full inventory and option count remain fixed. This design tests whether the model changes its decision when the required assignment changes and preserves its decision when the assignment remains the same.

Requirement 1: correct reassignment. Follow the category’s assigned destination. For each input, fifteen non-gold categories are sampled independently of model predictions. Fourteen remain regular choices in both conditions. The gold-regular condition adds $y ;$ gold-fallback replaces it with the remaining sampled category in the same slot. Every model receives the same pair. The required output changes from the regular option for y to fallback, while the number and positions of options remain fixed.

Requirement 2: definition consistency. Preserve the decision under equivalent descriptions. Within each assignment, U lists the fallback members and C excludes regular categories. Both definitions specify the same category-to-output assignment, so the required output is unchanged.

Measuring both requirements. For n texts, superscripts +/− denote gold-regular/gold-fallback, with predictions $\hat { g } _ { i } ^ { \pm }$ and fallback sets $\mathcal { R } _ { i } ^ { \pm }$ . We measure:

$$
A _ { + } = n ^ { - 1 } \sum _ { i } { \bf 1 } [ \hat { g } _ { i } ^ { + } = \{ y _ { i } \} ] ,\tag{2}
$$

$$
F _ { + } = n ^ { - 1 } \sum _ { i } { \bf 1 } [ \hat { g } _ { i } ^ { + } = \mathcal { R } _ { i } ^ { + } ] ,\tag{3}
$$

$$
A _ { - } = n ^ { - 1 } \sum _ { i } { \bf 1 } [ \hat { g } _ { i } ^ { - } = \mathcal { R } _ { i } ^ { - } ] .\tag{4}
$$

Here $A _ { + }$ is regular accuracy, A<sub>−</sub> fallback recall, and $F _ { + }$ false fallback rate. Since incorrect regular choices also reduce $A _ { + }$ , we have $F _ { + } + A _ { + } \leq 1$

Joint correctness (J) is the fraction of inputs correctly handled under both assignments for a given definition. Disagreement rate (DR) measures the fraction of inputs receiving different decisions under U and C within the same assignment, including cases where both decisions are incorrect. Thus J measures successful reassignment, while DR tests definition consistency. The PFR pass rate $( P _ { 4 } )$ is the fraction of inputs correctly handled in all four requests, jointly satisfying both requirements. These metrics capture different aspects of reliability. By construction, $P _ { 4 } \leq \operatorname* { m i n } ( J _ { U } , J _ { C } )$ , but equal

<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Model</td><td colspan="2"> $A _ { + }$  ↑</td><td colspan="2">A_ ↑</td><td colspan="3"> $F _ { + } \downarrow$ </td><td colspan="2">J↑</td><td colspan="2">DR↓</td><td rowspan="2"> $P _ { 4 }$ </td></tr><tr><td>U C</td><td>∆</td><td>U C</td><td>∆</td><td></td><td>U C</td><td>∆</td><td>U</td><td>C</td><td>∆</td><td>R F</td></tr><tr><td rowspan="4">CLINC150 n = 450</td><td>Intern</td><td>91.1 90.7</td><td>-0.4</td><td>68.7 75.8</td><td>+7.1</td><td>6.9</td><td>7.6</td><td>+0.7</td><td>62.2 68.4</td><td></td><td>+6.2</td><td>2.7 28.7</td><td>52.0</td></tr><tr><td>Kev</td><td>86.4 88.4 +2.0</td><td></td><td>94.2 92.9</td><td>-1.3</td><td>11.3</td><td>9.8</td><td>-1.6</td><td></td><td>82.9 83.1 +0.2</td><td></td><td>3.6 3.8</td><td>80.2</td></tr><tr><td>SemIf</td><td>78.2 78.0 -0.2</td><td></td><td>78.7 93.3 +14.7</td><td></td><td></td><td></td><td>15.8 20.4 +4.7</td><td></td><td>62.4 72.9 +10.4</td><td></td><td>9.3 20.2</td><td>58.2</td></tr><tr><td>JevK5</td><td>93.8 92.7 -1.1</td><td></td><td>57.3 59.1</td><td>+1.8</td><td>1.6</td><td></td><td>3.6 +2.0</td><td></td><td>55.8 55.8</td><td>+0.0</td><td>3.3 33.3</td><td>40.9</td></tr><tr><td rowspan="4">BANKING77  $n = 3 8 5$ </td><td>Intern</td><td>81.3 82.1 +0.8</td><td></td><td>74.0 67.5</td><td>-6.5</td><td></td><td></td><td>11.2 10.4 -0.8</td><td>62.9 57.4</td><td></td><td>-5.5</td><td>3.4 21.0</td><td>50.6</td></tr><tr><td>Kev</td><td>89.4 83.1 -6.2</td><td></td><td>96.4 97.7</td><td>+1.3</td><td></td><td>8.8 15.6 +6.8</td><td></td><td>87.5 82.1</td><td></td><td>-5.5</td><td>7.3 1.8</td><td>81.0</td></tr><tr><td>SemIf</td><td>69.6 73.0 +3.4</td><td></td><td>89.4 87.8</td><td>-1.6</td><td></td><td></td><td>25.7 23.4 -2.3</td><td></td><td>63.9 65.2</td><td>+1.3</td><td>7.8 10.9</td><td>57.7</td></tr><tr><td>JevK5</td><td>85.7 86.2 +0.5</td><td></td><td>53.5 47.3</td><td>-6.2</td><td>3.6</td><td>6.0</td><td>+2.3</td><td></td><td>49.9 41.8</td><td>-8.1</td><td>7.5 28.1</td><td>33.5</td></tr><tr><td rowspan="4">TREC50 n = 500</td><td>Intern</td><td>80.4 83.8 +3.4</td><td></td><td>59.4 22.6</td><td>-36.8</td><td>7.2</td><td>1.4</td><td>-5.8</td><td></td><td>51.8 21.0</td><td>-30.8</td><td>6.8 39.8</td><td>18.2</td></tr><tr><td>Kev</td><td>84.2 87.6 +3.4</td><td></td><td>88.0 74.0 -14.0</td><td></td><td>9.6</td><td>6.0</td><td>-3.6</td><td>78.0 68.2</td><td></td><td>-9.8</td><td>5.8 15.8</td><td>64.0</td></tr><tr><td>SemIf</td><td>82.0 83.4 +1.4</td><td></td><td>72.8 60.4 -12.4</td><td></td><td>9.2</td><td>8.2</td><td>-1.0</td><td></td><td>63.6 52.0 -11.6</td><td></td><td>8.2 24.0</td><td>45.6</td></tr><tr><td>JevK5</td><td>82.8 85.4 +2.6</td><td></td><td>57.8 24.2 -33.6</td><td></td><td>4.0</td><td>1.4</td><td>-2.6</td><td></td><td>53.2 22.8 -30.4</td><td></td><td>6.4 37.6</td><td>20.8</td></tr></table>

Table 1: Main PFR results (%) under direct selection with fallback named Other. U/C: union/complement; $\Delta = C - U \left( \mathrm { p p } \right)$ $A _ { + } \colon$ regular accuracy; $A _ { - } { : }$ fallback recall; $F _ { + }$ : false fallback rate; J: joint correctness under both assignments. DR: U/C disagreement under regular (R) or fallback (F) assignment. $P _ { 4 } \mathrm { { : } }$ correct on all four requests. Blue/orange: improvement/deterioration. Bold $A _ { - } / F _ { + }$ deltas: Holm-adjusted $p < . 0 5 ;$ paired intervals: Figure 15.

J values do not imply that the same inputs are correctly handled under both definitions. Similarly, net score changes $\Delta = C - U$ can hide switches in opposite directions. Intervals use 95% paired text-bootstrap resampling; primary U/C contrasts use Holm-corrected McNemar tests, and diagnostic intervals are descriptive (Appendix A).

Joint evaluation also avoids trivial behavior: always selecting fallback preserves agreement but fails regular cases; avoiding fallback removes false fallback while missing required transfers. PFR therefore tests responsiveness to changed assignments and invariance to equivalent descriptions on the same inputs.

## 3.2 Correct Classifications Can Be Lost on Reassignment

Under direct selection, $P _ { 4 }$ ranges from 18.2% to 81.0% across the twelve model–dataset settings (Table 1). To examine whether these failures also occur on inputs that models can correctly classify, we use a separate full-inventory classification task as a reference. Native option limits allow this evaluation for Kev on all three datasets and Intern on TREC. We then evaluate reference-correct inputs under the PFR assignments without providing their category labels to the model.

Intern correctly classifies 329 of the 500 TREC inputs in the reference task. On these same inputs, accuracy under C drops from 97.9% in the goldregular condition to 23.1% in the gold-fallback condition (Figure 1). Thus, correct classification in the reference task does not guarantee correct fallback selection. Paired examples in Appendix B illustrate failures under both definitions.

## 3.3 Supplying the Correct Category Is Not Sufficient

We next provide the correct category directly to the model. On TREC, supplying the fine-category label makes every JevK5 gold-regular decision correct under both U and C, yet fallback recall reaches only 71.1% and 14.8%, respectively. When the input text is removed and only the correct label is provided, C recall increases to 50.0%, while regular accuracy remains perfect (Figure 14). These results show that fallback errors persist even when the model does not need to infer the category from the input.

Assignment across backbones. We further compare Intern and Kev with Laya and OpenJev (Convai Innovations, 2026; Kotoba Labs, 2026) on a matched task containing 256 TREC inputs, six coarse categories, and four options. Even with the correct category provided, OpenJev achieves perfect regular accuracy but only 43.4% and 41.0% fallback recall under U and C, respectively (Figure 3). However, these failures are not universal. In a separate sixteen-option category-only task, Kev achieves 99.2% and 98.4% joint correctness on BANKING77 under U and C, respectively.

![](images/a15fd6e2b7e4295bf724600c004e4b9cded826aa07710965fea609ac33316665.jpg)  
Figure 3: Providing the correct category does not eliminate fallback errors. The matched TREC task uses six categories and four options. Bars show regular and fallback accuracy under Union (U) and Complement (C), with paired text-bootstrap intervals.

Takeaway. Correct category recognition does not guarantee correctfallback assignment. Errors can persist even when the correct category is explicitly provided.

## 4 Do Equivalent Definitions Preserve Decisions?

We examine whether logically equivalent fallback definitions lead to consistent decisions. Using the PFR setup from Section 3, we compare Union (U) and Complement (C) on the same inputs and category assignments, keeping the option set and category membership fixed. We first examine differences in accuracy and individual decisions, then test their sensitivity to model size and presentation, and finally extend the analysis to Fast/Defer policies.

## 4.1 Similar Performance, Different Decisions

Similar aggregate scores can conceal different decisions. JevK5’s CLINC fallback recalls differ by only 1.8% between U and C, yet one-third of its choices change. Although J is 55.8% under both definitions, only 40.9% of inputs pass the full PFR Test. Gold-fallback disagreement exceeds 10% in ten of the twelve model–dataset pairs (Table 1).

The direction of the accuracy change depends on the task (Figure 15). C lowers TREC recall for all four configurations after correction, but increases both recall and false fallback for SemIf on CLINC. On BANKING, no recall contrast survives correction. Equivalent descriptions can therefore change both which inputs reach fallback and how often that decision is correct; neither description is uniformly preferable.

## 4.2 Higher Recall Does Not Ensure Robustness

The discrepancy persists across model sizes and controlled changes in presentation. Each comparison holds the texts fixed.

Higher recall does not ensure a smaller definition gap. Across Intern 0.8B/2B/4B and Kev 0.8B/4B, fallback recall increases under both definitions without consistently narrowing their difference (Figure 4). Intern’s TREC gap grows from 0.4 to 36.8 points between 0.8B and 4B; CLINC’s largest gap instead occurs at 2B. At 0.8B, both Intern/TREC recalls are below 4%: the small gap coexists with almost complete fallback failure.

Position can reverse the definition gap. For JevK5 on CLINC150, the C-minus-U recall gap changes from −41.4 points at fallback position 1 to +35.2 at position 11, while mean regular accuracy remains 94.8–97.1%. Averaging across four positions and three choice sets leaves a gap of just +0.3 points (Figure 5). A small average can thus conceal large definition effects of opposite sign.

The gap survives matched controls. Matching token counts and category mentions still leaves TREC recall losses of 3.5–45.7 points under negative wording (Figure 8). The direction also persists across three wording pairs and same-parent or different-parent replacement categories. Splitting an unchanged union list over lines lowers recall as well. These controls show that the observed sensitivity extends across several presentations of the same assignment (Appendix C).

## 4.3 From Category Coverage to Defer Policies

Does the failure extend to deciding when to delegate? On TREC, we supply hypothetical quality estimates $q _ { \mathrm { f a s t } } , q _ { \mathrm { s l o w } }$ and extra cost c. Defer is required iff $q _ { \mathrm { s l o w } } - \lambda c > q _ { \mathrm { f a s t } } ;$ ; ties select Fast. Changing only λ switches the required action. Each policy is expressed through a gain inequality or an equivalent utility-complement criterion. Here J requires correctness under both policies.

Joint policy correctness is zero for all four models under both criteria. SemIf and JevK5 always choose Fast: perfect agreement coexists with zero J. Adding precomputed utilities or the correct Defer predicate with its action mapping changes information content, length and framing.

Precomputed utilities make Kev and JevK5 correct on every paired request, while SemIf remains below 50% joint correctness under both criteria. The correct predicate leaves errors for every model (Figure 6). A category-based policy likewise changes decisions under equivalent rules across all three datasets (Appendix D.2).

![](images/76d6723c3d6b01bfcbba72456cf284f3580ce8d16523890fdb9882fe3864b93f.jpg)  
Figure 4: Higher fallback recall does not necessarily reduce the U/C gap. Each model family uses the same evaluation texts and assignments across parameter counts. Solid/dashed lines denote U/C; colors distinguish Intern and Kev.

![](images/308cf35fef4ff13d0a7c91b0573b9d0b062544d8093d857e83685b37ddf7d1c8.jpg)  
Figure 5: Position reverses the U/C gap for JevK5/CLINC150. Points average three choice sets with paired-text intervals. Averaging across all four positions nearly cancels the difference, ∆ = C − U.

Errors also occur in the opposite direction. With zero quality gain and λ = 1, Intern correctly selects Fast on all 64 inputs under gain wording but unnecessarily defers on 32 under the equivalent utility-complement criterion (Appendix D.1).

Takeaway. Reliable fallback use requires both the right change of output and invariance to equivalent rules. Neither similar accuracy nor agreement alone establishes both.

## 5 What Does Reliable Fallback Selection Require?

We investigate this gap through three comparisons: deterministic mapping, explicit category membership, and selection rules applied to fixed scores. All use the main category-coverage sets.

![](images/4dbfbb8ff1707724cae5b23464e22ed80305507e1e5cada0838d5570f6fdaf65.jpg)  
Figure 6: Policy execution varies with the supplied information. J requires correct Fast and Defer choices under the two policy weights. Utilities and the correct predicate are separate additions to the estimates; panels use equivalent gain (a) and utility (b) criteria.

## 5.1 Separating Recognition from Assignment

As a diagnostic, we use a classify-then-map reference where full-category prediction is supported. The model first predicts the input category, which is then deterministically mapped to a regular or fallback option using g . By construction, decisions remain consistent under U and C, while joint correctness J equals the underlying classification accuracy. For Intern on TREC, this approach achieves $J = 6 5 . 8 \%$ , compared with 21.0% under direct C selection. However, classification errors still produce 22.8% false fallback.

## 5.2 Making the Assignment Explicit

Can the model directly use the same mapping? A membership table labels regular categories as T1 and fallback categories as T2, without revealing the category of the input. Under U, fallback is defined by T2; under C, it is defined by excluding T1. Moving a category into fallback changes its entry from

<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Model</td><td colspan="2">Direct</td><td colspan="3">Membership</td><td colspan="3">Threshold</td></tr><tr><td> $\overline { { J _ { U } } }$ </td><td>↑Jc ↑DR-F↓</td><td></td><td> $\overline { { J _ { U } \mathrm { ~ \uparrow ~ } } }$   $\overline { { J _ { C } } }$ </td><td>← DR-F↓</td><td> $J _ { U }$  ←</td><td> $\overline { { J _ { C } \uparrow } }$ </td><td>DR-F↓</td></tr><tr><td rowspan="4">CLINC150</td><td>Intern</td><td>62.2 68.4</td><td></td><td>28.7</td><td>79.6(+17.3) 79.6(+11.1)</td><td>6.4(−22.2)</td><td>73.6(+11.3)</td><td>74.9 (+6.4)</td><td>11.6(−17.1)</td></tr><tr><td>Kev</td><td>82.9 83.1</td><td></td><td>3.8</td><td>84.7 (+1.8) 85.1 (+2.0)</td><td>0.9 (−2.9)</td><td>84.4(+1.6)</td><td>81.6(-1.6)</td><td>4.2 (+0.4)</td></tr><tr><td>SemIf</td><td>62.4 72.9</td><td></td><td>20.2</td><td>72.7(+10.2) 71.8 (−1.1)</td><td>1.3 (−18.9)</td><td>64.7(+2.2)</td><td>72.0 (−0.9)</td><td>15.6 (−4.7)</td></tr><tr><td>JevK5</td><td>55.8 55.8</td><td></td><td>33.3</td><td>80.4(+24.7) 82.0 (+26.2)</td><td>2.9 (−30.4)</td><td>82.0 (+26.2)</td><td>)72.2(+16.4)</td><td>)10.7 (−22.7)</td></tr><tr><td rowspan="4">BANKING77</td><td>Intern</td><td>62.9 57.4</td><td></td><td>21.0</td><td>73.5(+10.6) 70.6(+13.2)</td><td>7.0 (−14.0)</td><td>69.9 (+7.0)</td><td>67.5(+10.1)</td><td>5.7 (−15.3)</td></tr><tr><td>Kev</td><td>87.5 82.1</td><td></td><td>1.8</td><td>80.8 (−6.8) 80.5 (−1.6)</td><td>0.0 (−1.8)</td><td>88.3 (+0.8)</td><td>82.1 (0.0)</td><td>1.8 (0.0)</td></tr><tr><td>SemIf</td><td>63.9 65.2</td><td></td><td>10.9</td><td>61.6(-2.3) 60.8 (−4.4)</td><td>2.1 (−8.8)</td><td>62.6(-1.3)</td><td>62.6 (−2.6)</td><td>2.9 (−8.1)</td></tr><tr><td>JevK5</td><td>49.9 41.8</td><td></td><td>28.1</td><td>69.9(+20.0) 69.6(+27.8)</td><td>7.5 (−20.5)</td><td>69.6(+19.7)</td><td>)59.7 (+17.9)</td><td>10.9 (−17.1)</td></tr><tr><td rowspan="4">TREC50</td><td>Intern</td><td>51.8 21.0</td><td></td><td>39.8</td><td>51.2 (−0.6) 29.0 (+8.0)</td><td>28.6(−11.2)</td><td>51.2 (−0.6)</td><td>53.4(+32.4)</td><td>17.0 (−22.8)</td></tr><tr><td>Kev</td><td>78.0 68.2</td><td></td><td>15.8</td><td>72.4 (−5.6) 71.0(+2.8)</td><td>4.2 (−11.6)</td><td>77.4(−0.6)</td><td>70.2 (+2.0)</td><td>13.8 (−2.0)</td></tr><tr><td>SemIf</td><td>63.6 52.0</td><td></td><td>24.0</td><td>60.2 (−3.4) 58.4(+6.4)</td><td>16.2 (−7.8)</td><td>68.4(+4.8)</td><td>64.0 (+12.0)</td><td>15.8 (−8.2)</td></tr><tr><td>JevK5</td><td>53.2 22.8</td><td></td><td>37.6</td><td>60.6(+7.4) 45.8 (+23.0)</td><td>)23.4 (−14.2)</td><td>62.6(+9.4)</td><td>58.8 (+36.0)</td><td>13.4 (−24.2)</td></tr></table>

Table 2: Correctness and consistency respond differently to intervention. $J _ { U } , J _ { C } { : }$ joint correctness under U/C; DR-F: gold-fallback U/C disagreement (%). Parentheses show changes from Direct (pp); blue/orange indicate improvement/deterioration. Evaluation sets match Table 1.

T1 to T2. This intervention makes the assignment explicit while changing its presentation. Table 2 separates gains in joint correctness from changes in disagreement: explicit membership need not improve both together. With explicit membership, SemIf’s fallback recall on CLINC reaches at least 98% under both definitions, while gold-fallback disagreement drops from 20.2% to 1.3%. For Intern on TREC under C, J increases from 21.0% to 29.0%.

## 5.3 Limits of a Shared Selection Rule

We next hold the requests and scores fixed. A threshold selects fallback when its log-score exceeds the largest regular log-score by more than τ; otherwise the best regular option is selected. For each model–dataset setting, we select a threshold that maximizes assignment-balanced accuracy $B = ( A _ { + } + A _ { - } ) / 2$ , averaged over U and C, on a separate development set. The threshold is then fixed for testing. Changing the selection rule recovers decisions that direct selection misses. Intern’s TREC complement recall rises from 22.6% to 75.6%, but false fallback rises from 1.4% to 20.6% (Figure 13). The scores therefore support substantial recovery at the cost of unnecessary fallback selections.

What can a shared threshold recover? A threshold preserves the ranking of regular options. Passing PFR requires a correct regular winner and compatible thresholds across both assignments and definitions. Individual recoverability does not guarantee recovery under a shared threshold. For Intern/TREC, individual thresholds can satisfy all four PFR requests for 359 of 500 inputs, compared with at most 205 under a single shared threshold and 200 under the development-selected threshold. This gap appears across all settings (Figure 12), as individually recoverable inputs may require incompatible thresholds. The individual and shared oracles use test labels to maximize four-request correctness at fixed scores, whereas the development rule optimizes mean B. Further derivations appear in Appendix E.1.

## 5.4 Agreement and Low False Fallback Are Insufficient

Greater agreement does not establish correctness: U and C can consistently select the same wrong option. Membership information reduces goldfallback disagreement in all twelve model–dataset pairs, yet Kev/BANKING becomes perfectly consistent on those inputs while its joint correctness declines under both definitions (Table 2). Membership information and thresholding can correct some errors while introducing others; thresholding may also improve fallback agreement at the expense of regular agreement (Figure 19).

Low false fallback answers a different question. $F _ { + }$ measures unnecessary fallback on goldregular inputs; it leaves required transfers on goldfallback inputs untested. On the TREC naming panel, changing SemIf’s fallback name from Other to None lowers $F _ { + }$ from 10.2% to 6.3%, while joint correctness falls from 43.8% to 22.7% (Appendix C). An order-statistic rule fitted on the development panels targets $F _ { + }$ at 5% (Angelopoulos et al., 2024). Using the larger of the two thresholds calibrated to control $F _ { + }$ under U and C, nine of twelve settings meet this empirical target under both definitions, yet fallback disagreement ranges from 10.7% to 44.9% and four-request correctness can be as low as 3.6% (Figure 11).

Implications for decision models. Evaluation should track each input’s outcomes across both category assignments and equivalent descriptions. This distinguishes corrected decisions from newly introduced errors, and agreement on correct outputs from agreement on incorrect ones.

Takeaway. Assess correctness and consistency together: fewer unnecessary fallbacks or more agreement can still conceal failed transfers and newly introduced errors.

## 6 Related Work

Language models and TDMs. Language models support software decisions through tool use (Schick et al., 2023) and structured generation (Geng et al., 2023). Jev exposes typed choices (Almeida, 2026); LLM-as-Jev studies next-token decision readouts (Li and Wagle, 2026). Zhong et al. (2026) pair available and missing arithmetic answers, while Lu and Wu (2026) study rejectionpolicy transfer and different sources of missing candidates. PFR holds the full category inventory explicit and changes its output partition. It tests whether recognized categories reach their assigned output by jointly evaluating two assignments and two equivalent definitions on each input.

Abstention and learned delegation. Fallback may implement abstention (Gangrade et al., 2021), clarification (Saeidi et al., 2018; den Hengst et al., 2024), or delegation. SelectiveNet jointly learns prediction and rejection for a target coverage (Geifman and El-Yaniv, 2019); learning to defer learns a classifier and rejector from expert decisions (Mozannar and Sontag, 2020). RouteLLM learns model selection from preferences (Ong et al., 2025), while reinforcement-learning routers optimize delegation or stopping from downstream rewards (Yu et al., 2026; Huang et al., 2026). These methods learn which action is useful. PFR tests execution of a supplied assignment or policy, including when the relevant category or decision information is already given.

Robustness, consistency and calibration. CheckList tests expected changes and invariances (Ribeiro et al., 2020); order and prompt studies expose selection biases (Zhao et al., 2021; Pezeshkpour and Hruschka, 2024; Wei et al., 2024; Salinas and Morstatter, 2024). Sun et al. (2026) rebind option names to fixed definitions and score the selected meaning. PFR changes category membership, then restates each assignment equivalently. Jointly scoring the resulting requests reveals losses concealed by equal marginal accuracy. Grouped prediction (Goren et al., 2024; Jain et al., 2023), probabilistic coherence (Chen and Li, 2026; Joy, 2026) and calibration (Guo et al., 2017; Angelopoulos et al., 2024; Jia et al., 2026) address related consistency or error targets. Our fixed-score analysis additionally separates individually recoverable decisions from those compatible with one shared threshold.

## 7 Conclusion

Providing a fallback does not ensure reliable handling. Recognized categories can miss their assigned output even when the correct category is supplied; equivalent definitions can redirect decisions while barely changing aggregate scores. PFR tests required output changes and definition consistency on the same inputs, with its four-request pass rate summarizing whether each input satisfies both requirements.

These findings make assignment robustness a concrete target for decision-model evaluation and design. Controlled interventions show why greater agreement or fewer false fallbacks can coexist with lost correct decisions, and why individually recoverable errors may resist a shared selection rule. Reliable fallback use requires verifying the decisions that invoke it.

## Limitations

This work focuses on identifying and systematically evaluating fallback reliability failures in TDMs rather than developing mitigation methods. Our experimental settings enable systematic comparisons of fallback decisions, but may not capture the full complexity of real-world applications. Future work could explore effective mitigation strategies.

## References

Diogo Almeida. 2026. Introducing System One Models & Jev. TypeSafe AI Blog. Accessed: 2026-10-09.

Anastasios N. Angelopoulos, Stephen Bates, Adam Fisch, Lihua Lei, and Tal Schuster. 2024. Conformal risk control. In International Conference on Learning Representations.

Iñigo Casanueva, Tadas Temcinas, Daniela Gerz,ˇ Matthew Henderson, and Ivan Vulic. 2020.´ Efficient intent detection with dual sentence encoders. In Proceedings ofthe 2nd Workshop on Natural Language Processingfor Conversational AI, pages 38–45, Online. Association for Computational Linguistics.

Han Chen and Yingrui Li. 2026. Probability contracts: Accuracy, coherence, and decisions across llm interfaces. Preprint, arXiv:2609.37470. Version 2.

Lingjiao Chen, Matei Zaharia, and James Zou. 2024. FrugalGPT: How to use large language models while reducing cost and improving performance. Transactions on Machine Learning Research. Featured Certification.

Convai Innovations. 2026. Laya. Hugging Face model card. Capacity-screened revision 55cf4c4ebb4e; accessed 9 October 2026.

Floris den Hengst, Ralf Wolter, Patrick Altmeyer, and Arda Kaygan. 2024. Conformal intent classification and clarification for fast and accurate intent recognition. In Findings of the Association for Computational Linguistics: NAACL 2024, pages 2412–2432, Mexico City, Mexico. Association for Computational Linguistics.

Aditya Gangrade, Anil Kag, and Venkatesh Saligrama. 2021. Selective classification via one-sided prediction. In Proceedings ofthe 24th International Conference on Artificial Intelligence and Statistics, volume 130 of Proceedings ofMachine Learning Research, pages 2179–2187. PMLR.

Yonatan Geifman and Ran El-Yaniv. 2019. SelectiveNet: A deep neural network with an integrated reject option. In Proceedings of the 36th International Conference on Machine Learning, volume 97 of Proceedings of Machine Learning Research, pages 2151–2159. PMLR.

Saibo Geng, Martin Josifoski, Maxime Peyrard, and Robert West. 2023. Grammar-constrained decoding for structured NLP tasks without finetuning. In Proceedings ofthe 2023 Conference on Empirical Methods in Natural Language Processing, pages 10932– 10952, Singapore. Association for Computational Linguistics.

Shani Goren, Ido Galil, and Ran El-Yaniv. 2024. Hierarchical selective classification. In Advances in Neural Information Processing Systems, volume 37.

Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q. Weinberger. 2017. On calibration of modern neural networks. In Proceedings ofthe 34th International Conference on Machine Learning, volume 70 of Proceedings of Machine Learning Research, pages 1321– 1330.

Jianyi Hu, Hangtao Zhang, Yi Liu, Yeqi Zeng, Li Zeng, Xianlong Wang, Rui Wang, and Leo Yu Zhang. 2026. Jevadvbench: A benchmark and black-box attacks for reinforcement learning for calibrated decisions models. Preprint, arXiv:2609.31142.

Shihong Huang, Shengjie Wang, Hong Ma, and Zhou Xu. 2026. RLCascadeRouter: Quality-estimator-free cascade routing via reinforcement learning. Preprint, arXiv:2608.15817.

InternLM Team. 2026. Intern-Decision: Fast multimodal decision model. Software and model release. Accessed 9 October 2026.

Kanishk Jain, Shyamgopal Karthik, and Vineet Gandhi. 2023. Test-time amendment with a coarse classifier for fine-grained classification. In Advances in Neural Information Processing Systems, volume 36.

Zijun Jia, Yuanchang Ye, Sen Jia, Yiyao Qian, Haoning Wang, Baojie Chen, Diyin Tang, Jinsong Yu, and Zhiyuan Wang. 2026. BalanceRAG: Joint risk calibration for cascaded retrieval-augmented generation. Preprint, arXiv:2605.20084.

Saman Sarker Joy. 2026. Do system one decisions add up? a study of probabilistic coherence. Preprint, arXiv:2609.33971. Version 1.

Kotoba Labs. 2026. Open-Jev-DeBERTa-v3-Large. Hugging Face model card. Capacity-screened revision 188ee67a5c93; accessed 9 October 2026.

Stefan Larson, Anish Mahendran, Joseph J. Peper, Christopher Clarke, Andrew Lee, Parker Hill, Jonathan K. Kummerfeld, Kevin Leach, Michael A. Laurenzano, Lingjia Tang, and Jason Mars. 2019. An evaluation dataset for intent classification and out-ofscope prediction. In Proceedings ofthe 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing, pages 1311–1316, Hong Kong, China. Association for Computational Linguistics.

Xin Li and Dan Roth. 2002. Learning question classifiers. In COLING 2002: The 19th International Conference on Computational Linguistics.

Yinheng Li and Justin Wagle. 2026. LLM-as-Jev: LLMs are already Jev-style decision models— when and how to fine-tune them. Preprint, arXiv:2610.02076. Version 2.

Jiawen Lu and Tongtong Wu. 2026. Benchmarking candidate coverage and rejection policy transfer in typed decision models. Preprint, arXiv:2610.03387. Version 2.

Hussein Mozannar and David Sontag. 2020. Consistent estimators for learning to defer to an expert. In Proceedings of the 37th International Conference on Machine Learning, volume 119 of Proceedings of Machine Learning Research, pages 7076–7087. PMLR.

Isaac Ong, Amjad Almahairi, Vincent Wu, Wei-Lin Chiang, Tianhao Wu, Joseph E. Gonzalez, M Waleed Kadous, and Ion Stoica. 2025. RouteLLM: Learning to route LLMs from preference data. In The Thirteenth International Conference on Learning Representations.

Jared Palmer. 2026a. Kev-0.8B: A calibrated decision model on Qwen3.5-0.8B. Hugging Face model card. Kev 1.0.

Jared Palmer. 2026b. Kev-4B: A calibrated decision model on Qwen3.5-4B. Hugging Face model card. Kev 1.0.

Pouya Pezeshkpour and Estevam Hruschka. 2024. Large language models sensitivity to the order of options in multiple-choice questions. In Findings ofthe Associationfor Computational Linguistics: NAACL 2024, pages 2006–2017, Mexico City, Mexico. Association for Computational Linguistics.

Marco Tulio Ribeiro, Tongshuang Wu, Carlos Guestrin, and Sameer Singh. 2020. Beyond accuracy: Behavioral testing of NLP models with CheckList. In Proceedings ofthe 58th Annual Meeting ofthe Associationfor Computational Linguistics, pages 4902– 4912. Association for Computational Linguistics.

Marzieh Saeidi, Max Bartolo, Patrick Lewis, Sameer Singh, Tim Rocktäschel, Mike Sheldon, Guillaume Bouchard, and Sebastian Riedel. 2018. Interpretation of natural language rules in conversational machine reading. In Proceedings ofthe 2018 Conference on Empirical Methods in Natural Language Processing, pages 2087–2097. Association for Computational Linguistics.

Abel Salinas and Fred Morstatter. 2024. The butterfly effect of altering prompts: How small changes and jailbreaks affect large language model performance. Preprint, arXiv:2401.03729. Version 3.

Timo Schick, Jane Dwivedi-Yu, Roberto Dessi, Roberta Raileanu, Maria Lomeli, Eric Hambro, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. 2023. Toolformer: Language models can teach themselves to use tools. In Thirty-seventh Conference on Neural Information Processing Systems.

Alibi Serikbay. 2026. JevK5 v0.3: Open-weight Jev alternative. Hugging Face model card. Accessed 9 October 2026.

Yu Sun, Junhao Xu, Jiajia Shi, and Zijin Yang. 2026. Type-safe is not error-free: Typed decision models follow the option name, not the definition bound to it. Preprint, arXiv:2609.26758. Version 3.

TheoLeeCJ. 2026. SemIf: Semantic ifs from open models. Software. Accessed 9 October 2026.

Sheng-Lun Wei, Cheng-Kuang Wu, Hen-Hsen Huang, and Hsin-Hsi Chen. 2024. Unveiling selection biases: Exploring order and token sensitivity in large language models. In Findings of the Association for

Computational Linguistics: ACL 2024, pages 5598– 5621, Bangkok, Thailand. Association for Computational Linguistics.

Tiantong Wu and Wei Yang Bryan Lim. 2026. Decision hijacking: Prompt injection attacks on jev’s typed probabilistic decisions. Preprint, arXiv:2609.28613.

Wenyi Yu, Siyin Wang, Terumi Chiba, Xianzhao Chen, Xiaohai Tian, Jun Zhang, Lu Lu, and Chao Zhang. 2026. SALMONN-duo: Adaptive dual-system coordination for full-duplex voice agents. Preprint, arXiv:2609.34247. Version 1.

Zihao Zhao, Eric Wallace, Shi Feng, Dan Klein, and Sameer Singh. 2021. Calibrate before use: Improving few-shot performance of language models. In Proceedings of the 38th International Conference on Machine Learning, volume 139 of Proceedings ofMachine Learning Research, pages 12697–12706. PMLR.

Jike Zhong, Ming Li, and Yuxiang Lai. 2026. When the right answer is missing: An arithmetic-dependent rejection bottleneck in jev. Preprint, arXiv:2609.39496. Version 1.

## A Evaluation Protocol

Sampling and pairing. Fixed hash ranks select three CLINC and five BANKING test texts per class; CLINC out-of-scope inputs are excluded. TREC’s test split covers 42 of the 50 inventory categories. Diagnostic texts come from CLINC validation and BANKING/TREC training, using within-category ranks and round-robin selection. Exact and normalized matches to test, development, calibration and other diagnostic inputs are excluded. Development and calibration pools instead preserve the eligible source distribution (Appendix E.2).

Category-information/routing, wording, matched-length, choice-set/position and naming comparisons share one diagnostic panel per dataset. Replacement/layout, policy, boundary and cross-backbone panels use disjoint texts. Reusing inputs within a comparison holds the category distribution fixed while changing the tested factor. Fallback positions cycle over the sixteen slots and remain fixed within each pair, except in the explicit position control.

Category descriptions. CLINC and BANKING use the released intent names with underscores rendered as spaces. TREC uses the dataset authors’ fine-category definitions, aligning creative to the released ENTY:cremat identifier. Requests supply the full inventory; category and union-member lists follow lexicographic order. Fallback criteria appear in Table 3.

![](images/6d7531f3d11028540259759b0d1d4d4184c3f180c8fe39257fa77044d76cae49.jpg)  
Figure 7: A TDM serves as (a) a category-routing model or (b) a fast System 1. Fallback options control access to general support or System 2; missed fallbacks bypass the intended handling.

Model interfaces. Intern scores decision symbols (InternLM Team, 2026), Kev uses an optionpointer head (Palmer, 2026b,a), and JevK5 and SemIf score answer letters (Serikbay, 2026; TheoLeeCJ, 2026). Native option limits are 62 for Intern and 255 for Kev. Inference uses bfloat16 and score temperatures of 1 (Intern/SemIf), 1.22 (JevK5) and 2.40605 (Kev). Model parameters remain fixed; threshold analyses reuse the evaluation scores.

JevK5 uses v0.3; Kev is built on Qwen3.5-4B-Base, while SemIf uses Qwen3.5-4B.

Statistical units. Tests and resampling retain each text’s variants together. Primary U/C contrasts use two-sided exact McNemar tests on A<sub>−</sub> and F . Holm correction covers all 42 tests: two metrics, three datasets and seven parameter configurations (the four 4B settings, Intern 0.8B/2B and Kev 0.8B). Diagnostic intervals use 10,000 paired textbootstrap draws, averaging mappings, templates or set/position combinations within texts. Compacttask resampling preserves coarse-category quotas. One repeat of the 384 matched natural-input requests per 4B configuration produced identical choices.

## B Recognition and Model Scope

Category information. The three input conditions retain the text, add its correct category, or provide the category alone (Figure 14). They distinguish ordinary recognition from selecting the assigned output once the category is supplied. Figure 22 gives all three datasets.

Paired examples. Intern correctly identifies both inputs below in the full-inventory reference and under both gold-regular definitions. For “How far is it from Denver to Aspen?”, replacing NUM:dist with LOC:mount makes fallback correct: U selects fallback, but C selects LOC:mount. For “What is home equity?”, replacing DESC:def with NUM:weight instead yields ENTY:veh under U and fallback under C.

Parameter counts and backbones. Intern 0.8B/2B/4B and Kev 0.8B/4B use identical main evaluation texts and assignments (Figure 4). Each point represents a separately trained model with its native readout and calibration. The matched cross-backbone task uses 256 TREC texts: 43 each from ABBR, DESC, ENTY and HUM, and 42 each from LOC and NUM. It retains six categories, four options, matched criteria and balanced positions; all inputs fit each model’s input limit.

## C Definition Robustness Controls

Table 3 lists the three criterion pairs. Pair 1 defines the main evaluation. Requests specify that each input belongs to exactly one inventory category and that fallback denotes a category group; regular criteria give the category label and definition.

Length and category mentions. A shared membership table partitions the inventory into T1/T2. Positive wording selects inputs in the designated group; negative wording excludes the other group. Swapping the group identifiers gives a second comparison. All 6,144 positive/negative request pairs match token counts and category mentions. Figure 8 shows that the resulting recall differences depend on both model and dataset.

Replacement and list layout. On TREC’s full inventory, the gold option is replaced in the same slot by a category from either the same or a different coarse parent. The other fourteen regular options stay fixed. In the layout comparison, six spaces in the 35-member union list become line breaks, preserving membership, order and punctuation. For Intern with different-parent replacement, this change repairs one decision but harms 21, lowering recall from 53.9% to 38.3% (Figure 9).

![](images/fbd1a924fe2d5a4ed8c7fa5f2d9a1fdbac623d70f4c419144cdde5c087e7704d.jpg)

Figure 8: Definition sensitivity survives length matching. Negative-minus-positive fallback recall (pp), averaged over both T1/T2 mappings; bars show pointwise paired-text intervals.
<table><tr><td></td><td>Pair Union (U)</td><td>Complement (C)</td></tr><tr><td>1</td><td>the input belongs to any one of these cate- the input belongs to a category in U that is not gories in U: [members].</td><td>represented by any of the other output options.</td></tr><tr><td>2</td><td>choose this option when the input belongs to choose this option when the input&#x27;s category one of the following categories in U: [mem- in U is outside the categories represented by bers].</td><td>the other output options.</td></tr><tr><td>3</td><td>this option covers exactly these categories in this option covers exactly the categories in U: [members].</td><td>U other than those represented by the other output options.</td></tr></table>

Table 3: Three pairs of logically equivalent fallback definitions. In the quoted criteria, U denotes the full category inventory, not the Union formulation. The common fallback-option name is omitted; [members] lists fallback categories. All other request fields remain fixed.

Wording, choice sets and position. Three U/C wording pairs are crossed with both assignments (Figure 10). A separate comparison crosses three choice sets with four fallback positions. C’s TREC recall loss persists in all twelve settings for every configuration, averaging 10.0–46.4 points. Across three choice sets, changing the fallback position alters 70.8% of SemIf/CLINC gold-fallback decisions under U, compared with 10.9% under C. Figure 5 illustrates a separate position effect for JevK5/CLINC, where the U/C recall gap reverses across positions.

Fallback names. Seven fallback names are compared with the complement criterion, category membership and positions fixed (Figure 21). Effects vary across configurations: None lowers SemIf/TREC recall from 53.1% to 29.7%, whereas None of the above improves Intern/TREC recall. For SemIf/TREC, Other and None respectively yield 13 and 8 false fallbacks, but correct assignment pairs fall from 56 to 29 out of 128 (J: 43.8% to 22.7%).

Individual decision changes. Figure 20 decomposes U/C disagreements into correct-to-wrong, wrong-to-correct, and wrong-to-wrong changes. Opposing changes can offset each other in aggregate accuracy, concealing substantial differences in individual decisions.

## D Policy Execution

## D.1 Policies from Quality and Cost

TREC texts are assigned to twelve parameter combinations, with 10 or 11 texts each: $q _ { \mathrm { f a s t } } \in$ $\{ 0 . 5 0 , 0 . 6 5 , 0 . 8 0 \}$ , gain $d \in \{ 0 . 1 0 , 0 . 1 5 \}$ and cost $c \in \{ 0 . 1 0 , 0 . 2 0 \}$ . We set $q _ { \mathrm { s l o w } } = q _ { \mathrm { f a s t } } + d$ and compare $\lambda = d / c + 0 . 2 5$ (Fast required) with $\lambda = d / c - 0 . 2 5$ (Defer required). Text, estimates and action order stay fixed. Gain wording selects Defer when $d > \lambda c ;$ utility-complement wording selects Defer unless $q _ { \mathrm { f a s t } } \geq q _ { \mathrm { s l o w } } - \lambda c$ . Both select Fast on ties.

Each input is evaluated with estimates alone, computed Fast/Defer utilities, or the correct Defer predicate and action mapping. The two policy weights require Fast and Defer, respectively, and joint correctness J requires both decisions to be correct (Figure 6). Equal weighting of the twelve parameter combinations preserves these findings.

![](images/22493705f8067ec14bd0c61dc44d4da9f4a2c1d3dee00ee5e0470f4ff5478c21.jpg)  
Figure 9: Replacement categories and list layout affect fallback recall on TREC. Flat and chunked U preserve membership and order; same/different parent refers to the gold category’s replacement. C is unchanged by chunking. Whiskers show paired-text intervals.

![](images/48950205efafb303ac8d0d4e983eba9976d538c92930773a099b2ac7f0fda759.jpg)  
Figure 10: Definition effects across three wording pairs. C-minus-U fallback recall averages templates within texts; bars show paired-text intervals.

For “How many James Bond novels are there?”, Kev receives $q _ { \mathrm { f a s t } } = 0 . 8 0 , q _ { \mathrm { s l o w } } = 0 . 9 5 , c = 0 . 2 0$ and $\lambda ~ = ~ 0 . 5 0$ , making Defer optimal (0.85 > 0.80). Under gain wording, estimates alone elicit Fast, computed utilities elicit Defer, and the correct predicate elicits Fast.

Boundary conditions. Separate panels use 64 texts each for zero gain, negative gain and utility ties. Intern selects Fast correctly under all four requests on 32, 61 and 38 texts, respectively, versus 62–64 for the other models. The zero-gain example in Section 4.3 shows that errors can also introduce unnecessary deferral.

## D.2 Policies from Category Membership

A randomly selected half of the category inventory maps to Fast and the remainder to System2. Swapping gold with one category in the other group changes the required action while preserving group sizes and action order. U lists System2 members; C excludes Fast members. The input’s category is not disclosed. These binary choices imply $A _ { + } + F _ { + } = 1$ . Gold-System2 disagreement spans 4.7%–44.5% across the twelve model–dataset pairs (Figure 16).

<table><tr><td rowspan="2">Model</td><td colspan="2">Fast correct</td><td colspan="2">Defer correct</td><td colspan="2">DR</td></tr><tr><td>Gain</td><td>Util.</td><td>Gain</td><td>Util.</td><td>Fast</td><td>Defer</td></tr><tr><td>Intern</td><td>125</td><td>68</td><td>3</td><td>29</td><td>57</td><td>26</td></tr><tr><td>Kev</td><td>105</td><td>109</td><td>14</td><td>9</td><td>12</td><td>11</td></tr><tr><td>SemIf</td><td>128</td><td>128</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>JevK5</td><td>128</td><td>128</td><td>0</td><td>0</td><td>0</td><td>0</td></tr></table>

Table 4: Policy decisions from supplied estimates, counts out of 128 texts. Gain/Util. denote equivalent gain and utility-complement criteria. Correct counts are separated by the required action; DR counts disagreements between criteria under Fast-required or Deferrequired policies. Both policy J and all-four correctness are zero throughout.

## E Selection Rules and Recovery

## E.1 Fixed Scores and Shared Thresholds

Let $m _ { e } ^ { + } / m _ { e } ^ { - }$ be the fallback-minus-best-regular logscore margin under gold-regular/fallback for $e \in$ $\{ U , C \}$ . A definition’s pair is correct iff its goldregular request has the correct regular winner and $\tau \in [ m _ { e } ^ { + } , m _ { e } ^ { - } )$ . All four requests are correct iff the gold-regular winners under U and C are both correct and

$$
\tau \in [ \operatorname* { m a x } ( m _ { U } ^ { + } , m _ { C } ^ { + } ) , \operatorname* { m i n } ( m _ { U } ^ { - } , m _ { C } ^ { - } ) ) .\tag{5}
$$

For texts satisfying this regular-winner condition, a nonempty interval makes all four requests individually recoverable. The largest overlap among these eligible intervals gives the best shared threshold. These bounds optimize four-request correctness, whereas the development rule optimizes mean B (Figure 12). Figure 17 partitions failures into incorrect regular winners and incompatible margins within or across definitions.

![](images/f7374bbf22cd64b3ce794273b4a9c7b790bcdabe187c23b9c188d3ec6abbe79b.jpg)  
Larger of U/C false-fallback rates (%)

Figure 11: Low false fallback does not ensure PFR correctness. Each point is a model–dataset setting under the shared calibration threshold. Shading marks test $F _ { + } \leq 5 \%$ under both $\mathrm { U } / \mathrm { C } ;$ the vertical axis requires correctness on all four requests.  
![](images/02c3dd7ac00c40c81a6d3e65a569c6626c1b8766beba814bfb624b58dae3c68a.jpg)

Figure 12: Individual recoverability exceeds shared-threshold recovery in every setting. All-four correctness under the development-selected threshold, best shared threshold, or a separate threshold per text. The latter two use test labels and bound recovery at fixed scores. Error decomposition: Figure 17.  
![](images/c6c57323ed6f6e8137069aa97c7f955559f74d75dfb0568793239c30ff350647.jpg)  
Figure 13: Intern/TREC recall–false-fallback trade-off at fixed scores. Marks indicate direct selection and the development threshold. Shading highlights 0–35% false fallback.

![](images/7c09e910f7d1b1df97fee923163e6c111b3a4cf8ead1425fe1a3b5001072ddf0.jpg)  
Figure 14: Category information on TREC. Each model has U/C columns. Text+label adds the gold category; Label only removes the text. Full comparison: Figure 22.

Fallback recall A <sub>−</sub> (n.s.) False fallback F<sub>+</sub> (n.s.) Fallback recall A <sub>−</sub> (sig.) False fallback F<sub>+</sub> (sig.)  
![](images/13f6bc777cc7b6950644e1c60d41479391665ac85e9425a44007f7070277ef98.jpg)

Figure 15: Aggregate U/C differences in the main evaluation. Complement-minus-union score changes with paired-text intervals. Filled markers survive the 42-test Holm correction.  
![](images/fa9a059d44fabb3ff6cad76858be2bc0c0214c67f0abdd453a2514f4d247dc7a.jpg)  
Figure 16: Equivalent category policies change missed and unnecessary transfers. Arrows connect U (filled circle) to C (open square); lower-left is better. Missed transfer is 1 − A on gold-System2, and unnecessary transfer is $F _ { + }$ on gold-Fast.

## E.2 One-Sided Calibration

Calibration and balanced-threshold fitting use the same development panels, disjoint from all evaluation texts. Sampling preserves the eligible source distribution; nested subsets of 32 and 64 assess development-size sensitivity. For gold-regular margins $m = \log p _ { \mathrm { f a l l b a c k } } - \log \operatorname* { m a x } p _ { \mathrm { r e g u l a r } } ,$ a threshold selects fallback when $m > \tau .$ . At level $\alpha , \tau$ is the $\lceil ( n + 1 ) ( 1 - \alpha ) \rceil$ th order statistic, or +∞ when its rank exceeds n. The rule controls false fallback under exchangeability of calibration and evaluation scores (Angelopoulos et al., 2024). We fit one threshold per definition and use their maximum; under exchangeability for each definition, this retains both marginal one-sided bounds.

The shared threshold matters: Kev/BANKING’s

U-only threshold leaves C false fallback above the 5% budget, whereas the maximum brings both below it. Across datasets, the calibration and target populations change: 19 of 48 target-definition comparisons exceed the empirical budget at $\alpha = 5 \%$ reaching 17.4%. Transfer comparisons share target texts.

## E.3 Development Sensitivity and Recovery Costs

Refitting thresholds on 2,000 paired bootstrap samples keeps each text’s four requests together and holds test scores fixed. The median refit improves mean B in eleven settings; seven improve it in at least 95% of refits (Figure 18). For Intern/TREC, every refit improves B and gold-fallback agreement, while increasing gold-regular disagreement.

Threshold fitting. Candidate thresholds are development-margin midpoints, zero and extreme endpoints; probabilities are floored at 10<sup>−12</sup>. Ties prefer fewer false fallbacks, smaller threshold magnitude, then the larger threshold. Choice order breaks regular-score ties. On the 21,360 main requests, each native choice is a maximumprobability option. First-in-order argmax agrees on 21,333 requests; the strict zero-threshold rule agrees on 21,263. All differences involve tied maxima. For Intern/TREC, complement recall and false fallback are unchanged at zero threshold (22.6% and 1.4%).

![](images/748d3ae11f3d5b351bbd3e6d8e5d94e8af387c593dd46284928373f7d2cf7350.jpg)

Figure 17: Fixed-score recovery has local and shared-rule limits. Bars partition the main evaluation texts: maximal shared-threshold coverage; additional individually recoverable cases; then failures assigned to wrong regular winners, within-definition conflict or cross-definition conflict.  
![](images/57d71c5bd9fda2ff8dfc670ff77978dd117fd643ffd6708667c63e1d8fab7f69.jpg)  
Figure 18: Recovery varies with the development sample. Circles (mean B) and diamonds (all-four correctness) show median changes from direct selection; lines give empirical 2.5th–97.5th percentiles over development-pool refits. Blue/orange denotes improvement/deterioration.

![](images/62045ba036f7c13790aa894ba34508661c4fcb5dcf286e59ade8a73064e6822e.jpg)  
Figure 19: Interventions both repair and harm decisions. Left bars partition initially wrong pairs; right points give the percentage of initially correct pairs overturned. Aligned rows share model and definition, but use different denominators. Both means a pair repaired separately by each intervention.

![](images/5788970d187d9304804edd480b36ea405841995ee465c483f41faecc669d8432.jpg)  
Inputs with changed choices (%)

Figure 20: Score differences conceal opposing transitions. Bars partition U-to-C choice changes as percentages of all texts under each gold assignment (R: regular; F: fallback). Different-wrong excludes unchanged wrong choices. Segments sum to DR; repairs and harms can cancel in accuracy.  
![](images/12b6f67911a3cf93be878e78d790aa770af0abb20bf1613dcb5a3c1b5a6230d8.jpg)

Figure 21: Fallback-name effects with the complement criterion fixed. Inputs, membership and positions are paired across seven names. Top: fallback recall A<sub>−</sub>. Bottom: false fallback $F _ { + }$ . Both are percentages; their color scales differ.  
![](images/debdd02595d739a752c20162846b627cc4fc2c6c29080d3e4f50c6e57d1ea1c0.jpg)  
Figure 22: Category information across all datasets, extending Figure 14. Columns pair U/C within each model; rows show the three input modes under regular or fallback assignment. Colors use a shared accuracy scale.