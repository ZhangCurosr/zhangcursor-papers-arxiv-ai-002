# Which Differences Matter? Controlled Tests ofRule Transfer, Joint Reliability, and Information Use in Decision Models

Tianyou Lai<sup>1</sup>\* Wentao Yue<sup>2</sup>\* Zhiwei Zuo<sup>3</sup> Qilei Li<sup>3†</sup>

<sup>1</sup>Carnegie Mellon University <sup>2</sup>Shenzhen University

<sup>3</sup>Central China Normal University

## Abstract

Decision models map stated rules and observed states to discrete answers. Aggregate accuracy leaves unresolved which rule and information changes guide those answers. We connect controlled input changes to transfer, joint reliability, and directed information use through executable arithmetic rules. Semantic rule separation and crossed rule and state tests make transfer and joint success separately measurable. We compare equivalent wordings, calibrate joint correctness through marginal-preserving reassociation, and alter tables to define alternative answer targets. Adaptation gains increase with nearest-source answer agreement; fully conflicting quartets yield small pooled gains and different gain signs across source splits. Equivalent rewrites have opposing task-level effects, linking transfer gains to rule presentation. Jointscore departures depend on which rule and state associations the reference preserves. A decision service changes answers more across equivalent wordings than across identical-text repeats and usually follows the answer targets implied by altered table values or mappings. Code and data will be made available.

## 1 Introduction

“A difference which makes a difference” — Gregory Bateson, Form, Substance and Difference (1970) — Gregory Bateson, Form, Substance and Difference (1970)

Decision models turn stated rules and observed states into labels, ratings, or selected records. Accuracy measures agreement with the prescribed answer, but equal scores can conceal different responses to the inputs. In an archived routing example, two models answer half the questions correctly. One ignores both rule and state changes; the other responds to states alone. Valid option selection can also coexist with answer-symbol binding errors

(Balepur et al., 2025; Xue et al., 2024). Understanding success therefore requires testing which input changes guide each rule-based decision.

## Research Question

How can we determine which rule and information changes guide a decision model's answers?

A gain on unfamiliar rules can still depend on familiar wording. Low all-four correctness can follow from individual errors, so its magnitude alone does not establish unusual joint behavior. Better performance with an external table also leaves open whether its values and mappings direct the answers. Behavioral tests and contrast sets motivate controlled input changes (Ribeiro et al., 2020; Gardner et al., 2020); rule-following benchmarks examine inference, counterfactual rules, and regulations (Parmar et al., 2024; Sun et al., 2025; Zhou et al., 2025). We connect these controls to the specific ambiguities in interpreting decision accuracy.

Executable arithmetic rules make intended answers and controlled changes checkable. We separate rules by finite-domain behavior to evaluate transfer to semantic components excluded from training. Equivalent rewrites hold those behaviors fixed while changing presentation. Crossing rules and states tests responses to each factor. Marginalpreserving reassociation calibrates all-four correctness against the observed individual predictions. More restrictive references retain rule or state associations to identify what drives departures from that calibration. Finally, changing table values or mappings defines alternative answer targets. Transitions toward those targets reveal directed information use beyond an arbitrary answer change. Figure 1 links input changes to behavioral diagnostics.

Low-rank adaptation (LoRA) improves accuracy on reserved rules, with larger gains at higher nearest-source answer agreement. Fully conflicting quartets have low joint success and opposite gain signs across source splits. Equivalent wording increases the adaptation advantage on binary decisions and reduces it on the other tasks. Joint-score departures further depend on the rule and state associations retained by the calibration reference. The decision service Jev changes answers more across equivalent wordings than across identical-text repeats. It benefits from complete value tables and follows the alternative targets implied by altered values and mappings. These diagnostics distinguish conflict-conditioned transfer, joint reliability, and directed information use in model decisions.

![](images/54bca24c644dd91d95e00f70a9fc6f0233d9aba0dc49e174cfe94294d843b9de.jpg)  
Figure 1: Overview of the three behavioral diagnostics: rule transfer and wording, joint reliability, and external information use. Scores and interventions illustrate the design.

• We establish an oracle-checked evaluation that uses semantic rule separation and paired input changes to distinguish transfer, wording sensitivity, and responses to rules and states.

• We show that transfer gains depend on sourceanswer agreement and the source split under complete conflict. Joint-score departures depend on the associations retained by different marginal-preserving calibration references.

• We identify directed use of external values through paired answer transitions under value and mapping interventions, complementing the evidence from aggregate accuracy gains.

## 2 Related Work

Behavioral evaluation and explicit rules. Behavioral evaluation tests whether predictions respond correctly to controlled input changes. Check-List and contrast sets target capabilities through local perturbations (Ribeiro et al., 2020; Gardner et al., 2020). Metric rankings also depend on the evaluation context (Deviyani and Diaz, 2025). RuleTaker and ProofWriter evaluate inference from stated facts and rules (Clark et al., 2020; Tafjord et al., 2021). LogicBench spans logical systems; RuleArena separates rule selection from application in regulations (Parmar et al., 2024; Zhou et al., 2025). RuleBench and IRFT extend inferential rule-following evaluation and training (Sun et al., 2025). These designs motivate testing which inputs guide decisions. Our executable grammar combines finite-domain semantic separation with crossed rule and state changes, linking success to responses along each input axis. Appendix Table 28 compares these paired evaluation designs.

Generalization, wording, and answer interfaces. Generalization benchmarks vary combinations, presentations, and answer interfaces. CFQ and COGS test composition; Ordered CommonGen tests requested concept order (Keysers et al., 2020; Kim and Linzen, 2020; Sakai et al., 2025). Promptformat and answer-symbol binding studies document interface effects (Voronov et al., 2024; Xue et al., 2024). LoRA supports adaptation tests (Hu et al., 2022). Cacheable-model studies test rule following with independently encoded inputs (Rajput et al., 2026). Choice2Seq measures correctness retained from complete-answer selection to sequential construction on matched problems (Lai et al., 2026). These studies motivate paired evaluations of generalization and presentation sensitivity. Our semantic rule separation and equivalent rewrites measure how adaptation to unfamiliar rule behavior changes across presentations of identical questions with fixed model weights and intended answers.

Information use and internal mechanisms. Information-use studies examine how context and representations influence decisions. Performance depends on where relevant context appears (Liu et al., 2024). Probing separates recoverable information from information used to answer (Hewitt and Liang, 2019; Belinkov, 2022). Causal abstraction and interchange-intervention training connect representations to high-level variables (Geiger et al., 2021, 2022). Logit inspection and attention modification examine factual recall under contextual counterfactuals (Ortu et al., 2024). Accuracy gains from supplied values leave their influence on individual answers unresolved. We alter table values and mappings to define alternative targets. Paired transitions toward those targets test directed information use. Source-selected donor replacements separately test whether recovered rule-value information influences the final selected answer.

## Takeaway

Choose input interventions that distinguish the behavioral properties behind a score. Separate evidence of transfer, responsiveness, and information use when designing each evaluation.

## 3 Controlled Rule–State Evaluation

## 3.1 States, Rules, and Decisions

A decision instance is $( s , r , t , O )$ : a state s, an explicitly specified rule r, a task family t, and an ordered option set O of size K. A programmatic oracle returns the semantic answer $y = g _ { t } ( r , s )$ and its index in O. A model predicts an option index. The semantic answers align option permutations.

Each record has three integer fields $( x , y , z ) \in$ $\{ - 1 2 , \ldots , 1 2 \} ^ { 3 }$ . Rules follow

$$
f _ { r } ( x , y , z ) = o _ { 0 } \big ( o _ { 1 } ( x , y ) , o _ { 2 } ( y , z ) \big ) ,\tag{1}
$$

where each operator belongs to $\{ + , - , \times$ , min, max}. Eligible rules depend on all three fields and retain the same threeoperator, depth-two structure. Binary decisions test whether $f _ { r } > 0$ . Ordinal ratings count how many prescribed thresholds the score strictly exceeds, so equality remains in the lower category. Minimum routing selects the record with the lowest rule value. Maximum selection selects the record with the highest value. Record identifiers resolve ties. The option counts are 2 for binary decisions, 3/5/7 for ratings, and 2/4/8 for routing and selection.

## 3.2 Semantic Separation and State Sampling

Syntactically different expressions can implement the same decision. We compare numeric, sign, rating, and ranking signatures over the finite record domain, including field permutations. Matching any signature creates an edge between rules. Connected components conservatively keep linked rules in one pool, even when the link is transitive rather than a direct match. The separation covers executable behavior on all records in the enumerated domain.

Replication excludes historically exposed components, then assigns disjoint source and development pools by root-operator quotas and rule characteristics. Reserved components support evaluation of fixed models. Natural sampling draws complete states before labeling. Previously used complete states are excluded across roles, while individual records may recur in different multi-record states. Appendix Algorithm 1 summarizes the construction. Appendix A.4 specifies the exact eligibility checks, allocation quotas, and constraints used to construct the paired diagnostic evaluation inputs.

## 3.3 Natural Accuracy and Joint Correctness

Natural accuracy is averaged first within each rule, then equally over rule identities, option counts within each task family, and the four task families. This prevents families with more option-count settings from dominating the pooled macro result.

A diagnostic quartet crosses two rules $r _ { 1 } , r _ { 2 }$ with two states $s _ { 1 } , s _ { 2 } \colon$

$$
\begin{array} { l } { { Q = ( ( r _ { 1 } , s _ { 1 } ) ( r _ { 1 } , s _ { 2 } ) ) , } } \\ { { ( r _ { 2 } , s _ { 1 } ) ( r _ { 2 } , s _ { 2 } ) ) , } } \\ { {  Y _ { Q } = ( \begin{array} { l l } { { a } } & { { b } } \\ { { b } } & { { a ) , a \ne b . } } \end{array}  } } \end{array}\tag{2}
$$

Oracle-based eligibility requires the semantic answer to change along either axis. A deterministic predictor invariant to either factor cannot solve the full quartet. Stochastic predictions can succeed by chance, so quartet success remains a behavioral score. We report diagnostic item accuracy and

$$
\begin{array} { l } { \displaystyle \mathrm { Q A c c } = \mathrm { M a c r o A v g } _ { t , K , Q } c ( Q ) , } \\ { \displaystyle c ( Q ) = \prod _ { i = 1 } ^ { 2 } \prod _ { j = 1 } ^ { 2 } \mathbf { 1 } [ \hat { y } ( r _ { i } , s _ { j } ) = y ( r _ { i } , s _ { j } ) ] . } \end{array}\tag{3}
$$

Diagnostic sampling conditions on quartet eligibility. We compare item accuracy and QAcc on identical diagnostic questions. Natural accuracy retains its original state-sampling distribution, separately from diagnostic-quartet selection criteria.

The same item score can conceal different rule and state responses. The example below shows archived routing predictions for two rules and two states. Both models achieve the same item score while responding differently to the crossed inputs. Appendix A.7 gives the inputs and selection rule.

## Same Accuracy, Different Responses

Rows vary the rule; columns vary the state.

Oracle Base LoRA R0 R1 R1 R1 R1 R0 R1 R0 R1 R1 R1 R0

Both models answer half the items correctly. Base stays invariant; LoRA responds to changes in states alone.

## Takeaway

Separate training and evaluation rules by their executable behavior, and isolate complete states across data roles. Cross both input factors to test correct responses to rule and state changes.

## 4 Behavioral Transfer and Information Use

Three paired diagnostics address the central question by testing transfer, joint correctness, and directed use of the supplied external information.

## 4.1 Experimental Setup

Controlled comparisons use Qwen3.5-2B-Base, LoRA adaptation, and source-selected frozen readouts (Qwen Team, 2026). Jev is a versioned external decision service (TypeSafe, 2026). Appendix C.1 compares additional instruction-tuned and decision systems (Mapika, 2026; StartLux Labs, 2026). Appendix A.8 maps each result to its data, readout, precision, batch, and freeze stage.

Each replication cell uses 2,880 source-training questions and their reversed option orders. LoRA adapts decoder linear modules at rank 16 for two

## Key Findings

Conflict-conditioned transfer: Gains increase with nearest-source answer agreement. Fully conflicting quartets have low pooled joint success and opposite gain signs across source splits. Equivalent wording changes task-level adaptation gains with the same fixed model weights.

Joint reliability: Pooled QAcc lies within the primary reference ranges. Maximum-selection departures shrink when calibration preserves ordered rule pairs. Joint-score interpretation depends on the associations retained in the chosen marginal-preserving reassociation reference.

Directed information use: Jev benefits from correct tables and switches from correct answers to altered-table targets. Paired value and mapping changes distinguish directed information use from an arbitrary response change under paired original and altered-table conditions.

epochs, with batch size 16 and optimizer seed 17. Source-natural validation negative log-likelihood selects the checkpoint. Frozen readouts learn residual corrections to candidate logits from the same training questions. Controlled Base and LoRA use BF16 backbone inference with FP32 candidate projection. Independent source-audit questions assess competence after fitting and selection and remain excluded from model fitting and model selection.

Data roles and scoring. Development evaluates rule transfer, diagnostic quartets, and paired table interventions. Replication crosses two rule splits with three state seeds; the initial comparison varies fit seeds. Their six fixed models share one reserved target pool, later reused for wording and conflict diagnostics. Development comparisons use fixed rank-4 readouts; reserved comparisons use sourceselected MLPs. Original reserved scores use batch 16; equivalent-wording comparisons use batch 8.

## 4.2 RQ1: How Do Source Conflict and Wording Shape Transfer?

LoRA’s advantage over the controlled base model persists across development-rule splits. Table 1 retains the LoRA gain across two rule splits. Jev leads overall, with a ranking reversal between splits. The fixed rank-4 readout uses BF16 logit caches, while Controlled Base uses FP32 candidate projection. Appendix Figure 5 retains the individual state-seed results for each state seed and source-rule split in the development evaluation.

<table><tr><td colspan="4">Development rules</td></tr><tr><td>Method</td><td>Natural</td><td>Diag. item</td><td>QAcc</td></tr><tr><td>Controlled Base</td><td>38.67</td><td>33.81</td><td>0.52</td></tr><tr><td>Rank-4 readout</td><td>39.11</td><td>35.11</td><td>0.17</td></tr><tr><td>LoRA</td><td>58.03</td><td>53.13</td><td>13.19</td></tr><tr><td>Jev</td><td>60.44</td><td>55.34</td><td>22.05</td></tr></table>

<table><tr><td colspan="3">Reserved rules</td></tr><tr><td>Method</td><td>Natural</td><td>QAcc</td></tr><tr><td>Controlled Base</td><td>38.06</td><td>0.00</td></tr><tr><td>Selected MLP</td><td>42.49</td><td>1.37</td></tr><tr><td>LoRA</td><td>57.08</td><td>9.56</td></tr></table>

Table 1: LoRA retains higher natural accuracy on development and reserved rules. Scores are percentages. Development averages two rule splits; reserved evaluation uses batch 16, equal weights for reserved blocks, and MLP readouts selected by source-validation NLL without reserved-rule fitting or any checkpoint selection.
<table><tr><td colspan="5">Nearest-source agreement groups</td></tr><tr><td>Group</td><td>Base</td><td>LoRA</td><td>Gain</td><td>QAcc</td></tr><tr><td colspan="5">Natural accuracy</td></tr><tr><td>Low</td><td>37.06</td><td>48.53</td><td>+11.48</td><td></td></tr><tr><td>Middle</td><td>38.31</td><td>57.71</td><td>+19.40</td><td></td></tr><tr><td>High</td><td>39.06</td><td>67.13</td><td>+28.07</td><td></td></tr><tr><td colspan="5">Diagnostic item accuracy and QAcc</td></tr><tr><td>Low</td><td>34.67</td><td>41.66</td><td>+6.99</td><td>6.24</td></tr><tr><td>Middle</td><td>36.11</td><td>49.28</td><td>+13.17</td><td>9.42</td></tr><tr><td>High</td><td>33.57</td><td>55.90</td><td>+22.33</td><td>14.07</td></tr></table>

<table><tr><td colspan="5">Complete-quartet conflict loads</td></tr><tr><td>Load / split</td><td>Base</td><td>LoRA</td><td>Gain</td><td>QAcc</td></tr><tr><td>0 / Pooled</td><td>34.91</td><td>53.49</td><td>+18.58</td><td>13.99</td></tr><tr><td>1 / Pooled</td><td>46.49</td><td>58.19</td><td>+11.70</td><td>12.32</td></tr><tr><td>2 / Pooled</td><td>36.62</td><td>50.76</td><td>+14.14</td><td>9.37</td></tr><tr><td>3 / Pooled</td><td>30.96</td><td>48.12</td><td>+17.16</td><td>9.03</td></tr><tr><td>4 / Pooled</td><td>21.44</td><td>22.83</td><td>+1.39</td><td>1.42</td></tr><tr><td colspan="5">Fully conflicting quartets by source split</td></tr><tr><td>4 / Split 1</td><td>20.47</td><td>14.37</td><td>-6.10</td><td>0.00</td></tr><tr><td>4 / Split 2</td><td>22.41</td><td>31.29</td><td>+8.88</td><td>2.83</td></tr></table>

Table 2: Transfer depends on source-answer agreement and conflict. Accuracy and LoRA QAcc are percentages; gains are pp. Base QAcc is zero throughout. Agreement groups rank pairs within task/K; load counts conflicting answers per intact quartet. Pooled scores weight source splits equally. Fully conflicting cohorts contain 39 and 32 quartet memberships, covering eight and seven task/K strata. Source splits reuse target questions with potentially different cohort memberships; these finite-cohort scores describe adaptation under source–target conflict.

LoRA improves reserved-rule performance over its unadapted backbone. Table 1 compares fixed LoRA and source-selected MLP readouts with Controlled Base. Models and the LoRA–readout criterion were fixed before reserved-question generation (Appendix F). LoRA–Base measures adaptation gains. LoRA–readout compares selected training procedures whose source competence remains unmatched (Section 5). Subsequent conflict and wording analyses reuse this reserved benchmark.

Source-answer agreement separates different transfer demands. For each target rule and task/K, its nearest source maximizes uniformdomain oracle-answer agreement. We rank target pairs by their mean agreement within each source split and task/K, keeping ties together. Table 2 reports absolute accuracy, gains, and QAcc. Natural gains increase across agreement groups in both splits; pooled QAcc increases, while split-1 QAcc is not monotonic. These relative groups retain all task/K strata, although their absolute agreement ranges can overlap across individual task families. Complete source–target conflict exposes splitdependent transfer. Conflict load counts differing source and target answers within an intact quartet. At load four, split-1 accuracy falls from 20.47% to 14.37%, whereas split-2 accuracy rises from 22.41% to 31.29%. Their QAcc is 0.00% and 2.83%, respectively. Pooling yields a 1.39 pp item gain and 1.42% QAcc, concealing the opposing gain signs across source splits. Intermediate conflict loads have no monotonic gain trend. The two source pools reuse target questions, so these contrasts describe source-dependent outcomes on the same fixed target-rule evaluation questions.

On natural conflicting questions, split-1 accuracy changes from 34.49% to 32.07% (−2.43 pp); split-2 changes from 31.10% to 43.72% (+12.62 pp). Agreeing subsets improve from 40.83% to 70.93% (+30.09 pp) and 42.49% to 70.74% (+28.25 pp), respectively. These conditional scores retain the original evaluation weights. Their gains differ from contributions to the full-set improvement. Individual-item filtering does not define QAcc. Conflict-load analysis instead preserves all four quartet members and reports success against the target oracle across both rules and input states.

Agreement measures similarity over the rule domain; conflict load measures disagreement on the evaluated quartet. Semantic separation alone does not require different answers on every state. These posthoc diagnostics reuse saved predictions and preserve complete quartets and original weights. Appendix L.4 reports split-level strata, coverage, and conditional intervals; Appendix L.3 reports natural conflict subsets and their gain contributions.

Comparisons at a fixed batch size reveal opposing task-level responses to equivalent wording. We score equivalent rewrites with fixed weights and batch size. Each rewrite parses to the original rule expression. LoRA remains above Base overall, but the functional rewrite reduces its advantage. The verbal rewrite changes little overall because binary gains offset other tasks’ declines (Figure 2). Binary gains mainly reflect a larger drop in Base. Elsewhere LoRA declines while Base is stable or improves. These task-level directions persist among items with stable original-text answers across batch sizes. Appendix L reports absolute scores and paired intervals. A fixed subset crosses BF16 and FP32 inference with dynamic and fixed padding. Wording changes continue to alter answers under FP32 with fixed padding, while sametext padding effects shrink under fixed-padding and FP32 model-execution controls (Appendix L.2).

Jev changes answers across wording more often than it changes answers across identicaltext repetitions. Adjacent Jev requests randomize the original and two rewrites, each repeated twice. Cross-wording disagreement exceeds identical-text variability (Figure 3). The contrast remains positive in all request-slot lag strata and after excluding interrupted quartets. Elapsed-time strata have uneven coverage, including one without same-text pairs. Appendix L.5 specifies the paired requests, aggregation weights, and sensitivity checks for the paired, randomized repeated-call comparison.

## Takeaway

Measure absolute accuracy, transfer gains, and joint correctness across source-agreement and conflict conditions. Check wording robustness on paired, semantically equivalent questions.

## 4.3 RQ2: What Does Calibrated Joint Correctness Show?

QAcc measures all-four success; calibration specifies which associations are being tested. The primary reference reassociates predictions within cell, task, option-count, quartet-slot, and gold-label strata. It preserves individual prediction marginals while breaking the observed quartet associations. Pooled LoRA and Jev QAcc lie within these reference ranges, while maximumselection scores exceed them (Table 3). Retaining ordered rule pairs brings maximum-selection reference means closer to observed QAcc. The statevector reference additionally retains both predictions on a state together, preserving their original within-state response associations (Appendix H.1).

![](images/b50f5c1500c5f7ba4234edc4403db70bccb8951fce6f81c52c1d6932d4636162.jpg)  
Figure 2: Equivalent wording changes task-level adaptation gains in opposite directions. Paired changes in the LoRA–Base advantage use batch 8 and 95% sharedstate intervals. Zero denotes an unchanged advantage on the same semantically equivalent decision questions.

![](images/3fefbeba5c34b99d981ffa46e34af383e31779738b061e66d0548f2b3f12594d.jpg)

(b) Paired excess disagreement  
![](images/3e95ae66d78162bac057e8137d9a31256e32ee51b280e270d8ccf866e64baafd.jpg)  
Figure 3: Jev changes answers more across equivalent wordings than across identical-text repeats. (a) Disagreement rates average items within splits. (b) The paired excess remains positive after interrupted quartets are excluded. Each wording repeats twice; brackets and whiskers give the 95% quartet-resampling intervals.

Preserving rule pairs yields pooled reference means of 13.33% for LoRA and 21.69% for Jev. Adding state-vector preservation gives 12.67% and 21.61%, respectively. Observed QAcc remains 13.19% and 22.05% across these references. A departure concerns associations removed by that specific reference; it does not identify an internal reasoning mechanism. Falling within a reference range leaves the observed all-four success unchanged. Finer conditioning can restrict movable predictions, and the state-vector reference does not match latent state difficulty. These controls characterize the constructed pairs. Reliability under independently sampled second states remains untested. Predictors that ignore rules or states can score on individual items but cannot solve full quartets. Source-label-majority, rule-blind, and stateblind controls demonstrate the gap between item and quartet success. Random choices supply a chance reference. Appendix H defines each control and the grouped quartet uncertainty estimates.

<table><tr><td colspan="3"></td><td rowspan="2">95% reference range</td></tr><tr><td>Method</td><td>QAcc </td><td>Ref. mean</td></tr><tr><td>Base</td><td>0.52</td><td>1.09</td><td>[0.00, 2.60]</td></tr><tr><td>Rank-4 readout</td><td>0.17</td><td>1.05</td><td>[0.17, 2.09]</td></tr><tr><td>LoRA</td><td>13.19</td><td>12.21</td><td>[10.07, 14.41]</td></tr><tr><td>Jev</td><td>22.05</td><td>21.29</td><td>[19.79, 22.92]</td></tr><tr><td>Maximum selection only</td><td></td><td></td><td></td></tr><tr><td>LoRA</td><td>13.19</td><td>7.60</td><td>[4.17, 10.42]</td></tr><tr><td>Jev</td><td>13.19</td><td>8.75</td><td>[6.25, 11.81]</td></tr></table>

Table 3: Pooled LoRA and Jev QAcc lie within the marginal-preserving reference ranges, while maximumselection scores exceed their ranges. Scores are percentages. Central 95% ranges describe 1,000 stratified reassociations that preserve prediction marginals; they summarize the reference distribution rather than confidence intervals for performance on newly sampled rules.

Aggregate rankings conceal distinct task profiles. Appendix Figure 6 shows Jev's strongest joint performance on binary decisions. LoRA leads on ordinal rating and minimum routing and ties Jev on maximum selection. Task-level QAcc distinguishes joint reliability on these rule families.

Both rule changes and state changes can fail to trigger the required response. LoRA reduces invariant predictions along both axes relative to Controlled Base. Remaining errors include unchanged answers and incorrect changes. Appendix C.4 separates invariant predictions, incorrect changes, and partial success by counting correct answers within each complete crossed quartet of rules and states.

<table><tr><td>Jev intervention</td><td>Overall T/N (%)</td><td>Initially correct T/B (%)</td></tr><tr><td></td><td>169/220</td><td>169/191</td></tr><tr><td>Formula rows</td><td>(76.82)</td><td>(88.48)</td></tr><tr><td>Record columns</td><td>173/190 (91.05)</td><td>173/177 (97.74)</td></tr><tr><td></td><td>243/320</td><td>243/282</td></tr><tr><td>Target values</td><td></td><td></td></tr><tr><td></td><td>(75.94)</td><td>(86.17)</td></tr></table>

Table 4: Jev usually switches to the altered-table target. Each cell gives transitions over its denominator, followed by the percentage. Overall rates use all conflicts; conditional rates use the initially correct conflicts.

## Takeaway

Calibrate joint correctness against references with explicitly stated preserved associations. Interpret each departure relative to the associations removed, and report all-four success separately from its defined reference comparison.

## 4.4 RQ3: How Do Answers Respond to External Values?

D2 supplies every rule's value for every record. D3 marks the same entries unavailable. Both retain the state, question, options, answer, and supplementary-table instructions. Using a common all-rule table requires identifying the requested rule. Appendix D defines the conditions, and Appendix L.10 provides the full native Jev requests. The same external values benefit the tested systems to different degrees. Complete values improve Jev consistently across replication cells. Local models gain less. LoRA's effects vary across cells, revealing differences in how the same supplied values affect full-quartet success (Figure 4). Jev’s answers shift toward targets implied by altered values and mappings. We mismatch formula rows or record columns, or change targetformula values to imply another answer. The original oracle stays fixed. Among conflicting questions answered correctly with the correct table, Jev usually switches to the altered-table target (Table 4). Both mapping mismatches preserve the value multiset and table character length. The pattern persists under task-balanced and equal-cell weighting (Appendix L.9). Appendix J reports all tested systems.

A transition pairs an original correct answer with the new table target. Let T count these transitions, N all applicable conflicts, and B those initially answered correctly. Table 4 reports T /N and T /B, using the pooled integer counts across all applicable evaluated task families and option-count strata.

![](images/009977e17bdbeb2a5540cb58d3180bc06f71708e06ab90957db67eb98ed608e3.jpg)

![](images/0fbc3d7454fd115ccb12ecea53dcc89bb349f056d013ebafa30ba2c2d2a7bba3.jpg)  
Figure 4: Complete value tables benefit Jev more than the local predictors on these paired inputs. Lines connect missing-value (D3) and complete-value (D2) scores. Endpoint labels use a shared 0–100% accuracy scale.

Even with the correct answer supplied, performance depends on context and instruction wording. Crossing context length with answer-binding instructions changes how systems use supplied answers. Local models benefit from direct instructions, while Jev performs best with legacy instructions and minimal context (Appendix Figure 8).

## Takeaway

Edit supplied values and their mappings to define alternative decision targets. Measure transitions toward those targets and separately test whether the answer interface correctly binds supplied information to the selected answer.

## 5 Source Competence and Internal Interventions

Independent source auditing leaves source competence unmatched. The qualification gate requires natural accuracy of 90% and QAcc of 75% on training and independent source-audit inputs. Base, selected readouts, and LoRA fail this gate in all six replication cells. Commonaudit natural accuracy / QAcc is 36.31% / 0.00% for Base, 45.70% / 1.39% for selected readouts, and 75.43% / 35.94% for LoRA. Validation-based early-checkpoint matching succeeds in two cells, but independent audit matching succeeds in none (Appendix I). LoRA–readout differences therefore compare the attained outcomes of the selected training procedures, with source competence still unmatched. They do not estimate a transfer advantage after controlling for attained source competence.

Source-selected internal donor replacements yield no stable rule-specific improvement over controls. Appendix G reports probe results, intervention norms, and explicit paired positive controls.

## Takeaway

Check source competence on independent inputs before attributing differences to transfer. Validate internal interventions with restoration and positive controls before interpreting the recovered rule information as decision-relevant.

## 6 Conclusion

We asked which input differences guide decision models’ answers. Controlled tests distinguish conflict-conditioned transfer, all-four reliability, and directed information use. Adaptation gains increase with nearest-source answer agreement, while complete conflict yields opposite gain signs across source splits and low pooled QAcc. Equivalent wording changes task-level gains, and calibration conclusions depend on the associations retained by the reference. Paired value and mapping interventions identify directed use of supplied information. Evaluation and method design should distinguish source-consistent success from correct decisions under conflict and preserve correct answers across equivalent wording. Report joint success with explicitly specified prediction-reassociation calibration conditions. Method comparisons should report attained source competence alongside target scores. Joint reliability remains a design objective even when individual decision accuracy improves under unfamiliar rules or supplied information.

## Limitations

The findings concern the tested rule families and evaluation conditions. The core grammar uses fixed-depth arithmetic rules on three bounded integer fields. Semantic separation is established within this finite domain. Boolean and finite-state extensions test additional rule formats rather than broad natural-language policy coverage. Fully conflicting cohorts contain 39 and 32 quartet memberships with different task coverage. Their opposing gains describe source-dependent transfer within the observed task coverage under complete conflict.

LoRA–readout comparisons estimate outcomes of the selected training procedures, because independent source-audit competence remains unmatched. The larger-model comparison also changes system identity, so the observed effects cannot be attributed to parameter count alone.

QAcc describes joint success on the constructed rule–state quartets. Calibration departures concern the associations removed by each reassociation reference. Small or constant-prediction strata restrict the available reassociations. Multi-record second states use swapped winning-record fields, so the state-vector reference does not establish reliability under independently sampled pairs of input states.

Wording results compare the original rendering with two equivalent templates. Jev’s adjacent repeats estimate variability within the observed service window, rather than performance across service versions or observation periods. Table interventions establish directed use of supplied information in answers; they do not identify an internal decision mechanism. Model comparisons concern these controlled diagnostics, rather than comparative detection rates across existing behavioral tests.

## Ethics Statement

We use synthetic states and oracle labels, with no human participants or personal records. Remote services receive only synthetic inputs. Benchmark performance does not establish suitability for consequential decisions. Generative AI tools assisted implementation, literature organization, and writing. Experimental records, failure evidence, and the actual scope of each analysis are retained.

## References

Nishant Balepur, Rachel Rudinger, and Jordan Boyd-Graber. 2025. Which of these best describes multiple

choice evaluation with LLMs? A) forced B) flawed C) fixable D) all of the above. In Proceedings ofthe 63rd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 3394–3418, Vienna, Austria. Association for Computational Linguistics.

Yonatan Belinkov. 2022. Probing classifiers: Promises, shortcomings, and advances. Computational Linguistics, 48(1):207–219.

Peter Clark, Oyvind Tafjord, and Kyle Richardson. 2020. Transformers as soft reasoners over language. In Proceedings of the Twenty-Ninth International Joint Conference on Artificial Intelligence, IJCAI-20, pages 3882–3890. International Joint Conferences on Artificial Intelligence Organization. Main track.

Athiya Deviyani and Fernando Diaz. 2025. Contextual metric meta-evaluation by measuring local metric accuracy. In Findings of the Association for Computational Linguistics: NAACL 2025, pages 4921–4940, Albuquerque, New Mexico. Association for Computational Linguistics.

Matt Gardner, Yoav Artzi, Victoria Basmov, Jonathan Berant, Ben Bogin, Sihao Chen, Pradeep Dasigi, Dheeru Dua, Yanai Elazar, Ananth Gottumukkala, Nitish Gupta, Hannaneh Hajishirzi, Gabriel Ilharco, Daniel Khashabi, Kevin Lin, Jiangming Liu, Nelson F. Liu, Phoebe Mulcaire, Qiang Ning, and 7 others. 2020. Evaluating models’ local decision boundaries via contrast sets. In Findings ofthe Association for Computational Linguistics: EMNLP 2020, pages 1307–1323, Online. Association for Computational Linguistics.

Atticus Geiger, Hanson Lu, Thomas Icard, and Christopher Potts. 2021. Causal abstractions of neural networks. In Advances in Neural Information Processing Systems, volume 34, pages 9574–9586. Curran Associates, Inc.

Atticus Geiger, Zhengxuan Wu, Hanson Lu, Josh Rozner, Elisa Kreiss, Thomas Icard, Noah Goodman, and Christopher Potts. 2022. Inducing causal structure for interpretable neural networks. In Proceedings ofthe 39th International Conference on Machine Learning, volume 162 of Proceedings of Machine Learning Research, pages 7324–7338. PMLR.

John Hewitt and Percy Liang. 2019. Designing and interpreting probes with control tasks. In Proceedings ofthe 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 2733–2743, Hong Kong, China. Association for Computational Linguistics.

Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2022. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations.

Daniel Keysers, Nathanael Schärli, Nathan Scales, Hylke Buisman, Daniel Furrer, Sergii Kashubin, Nikola Momchev, Danila Sinopalnikov, Lukasz Stafiniak, Tibor Tihon, Dmitry Tsarkov, Xiao Wang, Marc van Zee, and Olivier Bousquet. 2020. Measuring compositional generalization: A comprehensive method on realistic data. In International Conference on Learning Representations.

Najoung Kim and Tal Linzen. 2020. COGS: A compositional generalization challenge based on semantic interpretation. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 9087–9105, Online. Association for Computational Linguistics.

Tianyou Lai, Wentao Yue, Zhiying Deng, and Qilei Li. 2026. Correct choices, failed sequences: Choice2Seq benchmarks correctness retention across output interfaces. Preprint, arXiv:2610.16191.

Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. 2024. Lost in the middle: How language models use long contexts. Transactions ofthe Association for Computational Linguistics, 12:157–173.

Mapika. 2026. decider-2b: Typed decisions with calibrated probabilities in one forward pass. Official v11 model card, revision 533964dae8be954c5b5e19fa4948e48408094c1e. Accessed 7 October 2026.

Francesco Ortu, Zhijing Jin, Diego Doimo, Mrinmaya Sachan, Alberto Cazzaniga, and Bernhard Schölkopf. 2024. Competition of mechanisms: Tracing how language models handle facts and counterfactuals. In Proceedings of the 62nd Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 8420–8436, Bangkok, Thailand. Association for Computational Linguistics.

Mihir Parmar, Nisarg Patel, Neeraj Varshney, Mutsumi Nakamura, Man Luo, Santosh Mashetty, Arindam Mitra, and Chitta Baral. 2024. LogicBench: Towards systematic evaluation of logical reasoning ability of large language models. In Proceedings ofthe 62nd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 13679– 13707, Bangkok, Thailand. Association for Computational Linguistics.

Qwen Team. 2026. Qwen3.5: Towards native multimodal agents. Official reference supplied by the Qwen3.5-2B-Base model card.

Dushyant Rajput, Nirdesh Chauhan, and Siddharth Kosaraju. 2026. Can a cacheable decision model follow rules? Preprint, arXiv:2609.37832.

Marco Tulio Ribeiro, Tongshuang Wu, Carlos Guestrin, and Sameer Singh. 2020. Beyond accuracy: Behavioral testing of NLP models with CheckList. In Proceedings ofthe 58th Annual Meeting ofthe Association for Computational Linguistics, pages 4902– 4912, Online. Association for Computational Linguistics.

Yusuke Sakai, Hidetaka Kamigaito, and Taro Watanabe. 2025. Revisiting compositional generalization capability of large language models considering instruction following ability. In Proceedings of the 63rd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 31219–31238, Vienna, Austria. Association for Computational Linguistics.

StartLux Labs. 2026. StartLux-Decision: Typed decision models. Official model and inference repository. Accessed 7 October 2026.

Wangtao Sun, Chenxiang Zhang, Xueyou Zhang, Xuanqing Yu, Ziyang Huang, Haotian Xu, Shizhu He, Jun Zhao, and Kang Liu. 2025. Beyond instruction following: Evaluating inferential rule following of large language models. In Proceedings of the 24th China National Conference on Computational Linguistics (CCL 2025), pages 1043–1066, Jinan, China. Chinese Information Processing Society of China.

Oyvind Tafjord, Bhavana Dalvi, and Peter Clark. 2021. ProofWriter: Generating implications, proofs, and abductive statements over natural language. In Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021, pages 3621–3634, Online. Association for Computational Linguistics.

TypeSafe. 2026. Jev and the TypeSafe System One API. Official service documentation. Accessed 7 October 2026.

Anton Voronov, Lena Wolf, and Max Ryabinin. 2024. Mind your format: Towards consistent evaluation of in-context learning improvements. In Findings of the Associationfor Computational Linguistics: ACL 2024, pages 6287–6310, Bangkok, Thailand. Association for Computational Linguistics.

Mengge Xue, Zhenyu Hu, Liqun Liu, Kuo Liao, Shuang Li, Honglin Han, Meng Zhao, and Chengguo Yin. 2024. Strengthened symbol binding makes large language models reliable multiple-choice selectors. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 4331–4344, Bangkok, Thailand. Association for Computational Linguistics.

Ruiwen Zhou, Wenyue Hua, Liangming Pan, Sitao Cheng, Xiaobao Wu, En Yu, and William Yang Wang. 2025. RuleArena: A benchmark for rule-guided reasoning with LLMs in real-world scenarios. In Proceedings ofthe 63rd Annual Meeting ofthe Association for Computational Linguistics (Volume 1: Long Papers), pages 550–572, Vienna, Austria. Association for Computational Linguistics.

## A Task Construction and Evaluation Protocol

## A.1 Executable Rules and Task Semantics

The rule grammar in Equation 1 contains three operator slots drawn from five operators, giving $5 ^ { 3 } = 1 2 5$ syntactic candidates. Each expression is evaluated over all $2 5 ^ { 3 } = 1 5 { , } 6 2 5$ integer records in $\{ - 1 2 , \ldots , 1 2 \} ^ { 3 }$ . Eligibility requires effective dependence on all three fields and excludes expressions whose audited task signatures match tested lower-complexity expressions. This leaves 104 candidates. Numeric, binary, ordinal, and full weakorder signatures, including ties and field permutations, link equivalent candidates into 77 connected components. Using components rather than pairwise exclusions also prevents a chain of equivalences from spanning data pools. The audit establishes behavioral separation over the enumerated integer domain. Symbolic equivalence on real inputs lies outside this finite integer-domain test.

For one-record tasks, let $v = f _ { r } ( x , y , z )$ . Binary decisions return yes exactly when $v > 0 ,$ , and no otherwise. Ordinal ratings with $K \in \{ 3 , 5 , 7 \}$ return

$$
y = \sum _ { i = 1 } ^ { K - 1 } \mathbf { 1 } [ v > 1 0 ( i - \lfloor K / 2 \rfloor ) ] .\tag{4}
$$

The threshold lists are $[ 0 , 1 0 ] , [ - 1 0 , 0 , 1 0 , 2 0 ]$ , and $[ - 2 0 , - 1 0 , 0 , 1 0 , 2 0 , 3 0 ]$ for $K = 3 , 5 , 7 ,$ respectively. Equality to a threshold does not increment the rating. Multiclass routing uses $K \in \{ 2 , 4 , 8 \}$ records and selects the smallest score. Structured selection uses the same option counts and selects the largest score. Both resolve ties by the smallest numeric record ID, independently of candidate order. There are ten task/K strata, with equal macro weight assigned to each of the four task families.

## A.2 Rule Pools and Complete-State Isolation

Before replication, a historical ledger contains 35,940 complete states and 40 rule ASTs. Historical exposure excludes 19 of the 77 components. Four disjoint pools of eight components use 32 of the remaining 58, forming two source/development splits and leaving 26 components in reserve. Each eight-rule pool has root-operator counts of one addition, two subtractions, two multiplications, one minimum, and two maxima. Pool construction also matches recorded operator-composition and binary-answer characteristics. The chosen pools pass a separate ordinal overlap check at each K. Rule split seeds are 20261101 and 20261102. State seeds within each split are 101, 202, and 303.

Algorithm 1 Construct rule pools and evaluations   
Require: Rules G, simpler rules L, record domain D, ex  
posed components H, used states $U _ { 0 } ,$ seeds   
Ensure: Rule pools P, natural sets $N ,$ quartets Q   
1: for $r \in G \setminus L ,$ signature type k do   
2: $\sigma _ { k } ( r ) \gets \operatorname* { m i n } _ { \pi \in S _ { 3 } }$ Hash $. ( \mathrm { S i g } _ { k } ( r \circ \pi , D ) )$   
3: end for   
4: $R  \{ r \in G : { \mathrm { E l i g i b l e } } ( r , L , \sigma ) \}$   
5: $E \gets \tilde { \{ ( r , r ^ { \prime } ) \in R ^ { 2 } : \exists k , \sigma _ { k } ( r ) = \sigma _ { k } ( r ^ { \prime } ) \} }$   
6: $C  { \dot { C } }$ onnectedComponents(R, E)   
7: $P  \mathrm { S e l e c t P o o l s } ( C \setminus ^ { } H ;$ quotas, seeds)   
8: $N , Q \gets \emptyset , \emptyset , U \dot { \gets } \dot { U _ { 0 } }$   
9: for pool p, data role, and valid task/K pair $( t , K )$ do   
10: S ← SampleStates $( S _ { t , K } ( D ) \setminus U )$   
11: $N  N \land \land \lor \{ ( r , s , g _ { t } ( r , s ) ) : r \in p , s \in S \}$   
12: $U  U \cup { \dot { S } }$   
13: $Q ^ { \prime } \gets \mathrm { S a m p l e Q u a r t e t s } ( p , S _ { t , K } ( D ) \setminus U )$   
subject to $y _ { 0 0 } = y _ { 1 1 } \neq y _ { 0 1 } = y _ { 1 0 } ,$   
where $y _ { i j } = g _ { t } ( r _ { i } , s _ { j } )$   
14: $Q  \bar { Q } \cup \bar { Q ^ { \prime } } , \bar { U }  \bar { U } \cup \mathrm { S t a t e s } ( Q ^ { \prime } )$   
15: end for   
16: return Freeze $( P , N , Q ,$ seeds, hashes)

Natural sampling first draws complete states with integer coordinates uniform on the finite domain, then evaluates every rule in the relevant pool. It does not filter states by answer, class balance, margin, or tie. Rejection excludes complete states already assigned to historical data or another role. A complete-state key is the ordered sequence of all records’ $( x , y , z )$ triples. This isolates entire states. An individual record can occur in two otherwise distinct multi-record states. Source training, source validation, development, and their natural/diagnostic subsets have disjoint complete-state keys. Replication cells also have disjoint complete states, while questions for different rules within a natural role intentionally share complete states.

## A.3 Semantic Reconstruction and Similarity Audit

Exhaustive signatures over the finite domain and all six field permutations show no cross-pool equivalence, including complete weak orders for record selection. Non-equivalent rules may still agree frequently. Appendix L measures nearest-source oracle agreement under the original and best-permuted fields, separating uniform-domain similarity from accepted natural states. The source and reserved rule allocations remain fixed throughout these posthoc, paired source–target agreement analyses.

## A.4 Exact Eligibility and Pool Construction

Here $\boldsymbol { S } _ { t , K } ( \boldsymbol { D } )$ contains complete task states, including ordered records. Sampling uses the fixed seeds, rule quotas, and per-task record-count strata.

$\begin{array} { r l r l r } { \mathrm { L e t } } & { { } { \cal F } } & { } & { { } = } & { \{ x , y , z \} } \end{array}$ and $\begin{array} { r l } { O } & { { } = } \end{array}$ $\{ + , - , \times , \operatorname* { m i n } , \operatorname* { m a x } \}$ . The simpler-expression set starts with $F$ and $U = \{ o ( a , b ) : o \in O , \ a , b \in$ $F \}$ , allowing repeated leaves. It also includes $| a | , \ o ( u , a ) , \ o ( a , u ) , \ | u | , \ o ( | a | , b )$ , and $o ( a , | b | )$ for $a , b \in \mathsf { \Gamma } $ and $u \in \ U$ We deduplicate expression trees by sorting commutative children and canonicalizing field names under each of the six possible integer-field coordinate permutations.

For each expression, the numeric signature contains all integer values over the domain. The binary signature contains the signs $f > 0 .$ . The ordinal signature jointly stacks the answers for $K \in \{ 3 , 5 , 7 \}$ The ranking signature gives the rank of each distinct value and retains ties. This complete weak order records every less-than, equal-to, and greaterthan relation, rather than just the winning record. Each canonical signature is the minimum hash across the six coordinate permutations. An expression is rejected if any corresponding canonical signature agrees with a simpler expression, if any field never changes the output, or if its binary answer is constant. Graph edges join expressions when any one of the four canonical signatures agrees, and connected components take the transitive closure. Per-K ordinal overlap is checked separately across pools. A graph edge therefore requires one matching signature, rather than a simultaneous match of behavior across all four task-specific signatures.

After excluding exposed components, each split uses root quotas (1, 2, 2, 1, 2) for (+, −, ×, min, max). Within each root class, a seeded choice supplies one representative, and its partner minimizes the $L _ { 1 }$ difference in operator-count vectors plus ten times the absolute difference in binary-positive frequency. Formula strings break distance ties, and a seeded coin assigns the pair to source or development. Each pool must permit four disjoint rule pairs for every task/K. Binary and rating pairs need at least 16 unused states in each reversal bucket. Multi-record pairs are screened by 256 deterministic seeded probes before pair matching. The procedure allows at most 10,000 attempts. Accepted components are removed before selecting the next split. The 26 remaining components retain root counts of at least (4, 4, 4, 2, 2) and form the held-out reserve.

<table><tr><td>State  $( x , y , z )$ </td><td> $r _ { 0 }$ </td><td>Answer</td><td> $r _ { 1 }$ </td><td>Answer</td></tr><tr><td> $( 0 , - 8 , 0 )$ </td><td>0</td><td> $\mathrm { n o } / \mathrm { A }$ </td><td>8</td><td> $\mathrm { y e s } / \mathrm { B }$ </td></tr><tr><td> $( 0 , 6 , - 1 0 )$ </td><td>6</td><td> $\mathrm { y e s } / \mathrm { B }$ </td><td> $- 1 6$ </td><td> $\mathrm { n o } / \mathrm { A }$ </td></tr></table>

Table 5: Changing either the rule or the state reverses the binary answer in this evaluated quartet. Exact values determine the gold answers; zero has the answer no.

## A.5 Counterbalanced Diagnostic Construction

Binary and ordinal generation samples unused states from opposite-answer strata. For each task/K/role, the generator traverses the frozen disjoint rule pairs in order and constructs two quartets per pair. For multi-record tasks it draws $s _ { 0 } ,$ , obtains the two winning record IDs $a , b ,$ and rejects $a = b$ It copies $s _ { 0 }$ to $s _ { 1 }$ and swaps the $( x , y , z )$ fields of records $a , b ,$ retaining their IDs. Both rules are evaluated again. Acceptance requires the checkerboard labels, two different complete states, and neither state in the used-state set. Accepted states enter that set. A quartet gets at most 20,000 attempts, after which construction fails explicitly. The four items are emitted in state-major, then rule-major order. Construction uses the oracle answers only.

Each source-training question also receives one reversed candidate order, preserving semantic answers and record-ID tie-breaking. Reversed quartets are separate four-item groups, not eight-item success events. Cross-rendering comparisons align semantic answers. Diagnostic sampling conditions on quartet eligibility. We separately report diagnostic scores and natural accuracy on their respective sets of sampled natural and diagnostic inputs.

## A.6 Evaluated Quartet and Local Decision Prompt

The evaluated quartet changes its answer along both input axes. We take the first binary development quartet in replication split 1, state seed 101, pair 0, quartet 0. Its two component representatives are $\boldsymbol { r } _ { 0 } = ( x \times y ) + \operatorname* { m a x } ( y , z )$ and $r _ { 1 } = ( x \times y ) - ( y - z )$ . The stored states and recomputed exact scores are shown in Table 5. Dataset order selects this example before model predictions are examined, preserving its role as an illustration of the paired, crossed rule and state construction.

The template below reproduces the local candidate-scoring input. Blue braces mark substituted fields; they are notation, not literal prompt characters. The evaluated instance uses state = $\{  ^ { n } \} \mathsf { f a c t s } ^ { \prime } \colon \{ ^ { n }  { \mathbf { i } }  { \mathbf { d } } ^ { \prime \prime } \colon { ^ { n } \mathsf { R } }  { \boldsymbol { \theta } } ^ { \prime \prime } , { ^ { n } \mathbf { x } } ^ { \prime } \colon  { \boldsymbol { \theta } } , { ^ { n } \mathbf { y } } ^ { \prime \prime } \colon - 8$ $" z " : ~ 0 3 3$ . Its formula is $( ( { \bf x } \mathrm {  ~ \star ~ } { \bf y } ) + { \sf m a x } ( { \bf y } , \mathrm {  ~ \sf z } ) )$ and its options are A) no and B) yes. Substitution recovers the original first-item prompt. The gold A (no) is evaluation-only. The displayed field and option breaks are literal newlines. Additional line wrapping serves presentation and does not change the evaluated local candidate-scoring prompt.

Binary decision: local candidate scoring   
State: {state}   
Question: Using facts, is {formula} strictly   
greater than 0? Select no or yes. Only x, y,   
z participate. Ignore other fields and notes.   
Options:   
{letter}) {semantic option}   
{letter}) {semantic option}   
Answer:

## A.7 Routing Example

The main-text matrix uses $r _ { 0 } = \operatorname* { m i n } ( x , y ) - ( y + z )$ and $r _ { 1 } = \mathrm { m i n } ( \mathrm { m i n } ( x , y ) , y - z )$ . State s<sub>0</sub> contains ${ \sf R 0 = } ( - 9 , 1 , 8 )$ and $\mathbf { R } 1 { = } ( 5 , - 5 , 7 )$ , and $s _ { 1 } \mathrm { e x - }$ changes their fields. The task selects the minimumscoring record. The example is the first lexicographic archived case with equal correct counts but different positions of the correct model answers.

## A.8 Result Configurations and Data Roles

The initial system comparison uses the initial batch and averages LoRA over fit seeds 17, 29, and 43. Replication uses two rule splits with three state seeds per split and optimizer seed 17. Its fixed rank-4 readout retains BF16 logit caches, whereas Controlled Base uses FP32 candidate projection.

Reserved evaluation freezes six source-selected MLP readouts and six LoRA models before generating questions. The LoRA–readout margin criterion is prospective, and Table 1 uses batch 16 with equal rule-block weights. Later wording analyses use the inspected reserved benchmark and rescore the original and both rewrites at batch 8. This explains the batch-dependent differences between Table 1 and the wording tables. The third split repeats development on previously exposed components.

Quartet calibration reuses replication predictions. Table interventions use paired development questions, and adjacent Jev repeats use fixed development quartets. These later analyses retain model and input identities while introducing their stated conditions. Natural-score intervals group shared complete states, diagnostic intervals group complete quartets, and reserved analyses retain ruleblock weights. Sections H and L.5 define uncertainty conditional on the fixed model checkpoints and the paired evaluation questions and the dependencies among shared complete states or quartets.

## A.9 Data Roles and Permitted Selection

The initial comparison gives each system 1,920 original-task questions: 640 natural and 320 diagnostic questions in each of source and development. Auxiliary conditions reuse the 320 development diagnostic indices. Three initial LoRA optimizer seeds share rules and states. Replication instead fixes optimizer seed 17 and repeats each of two rule allocations with state seeds 101/202/303. Table 6 specifies all roles. Each cell has 960 complete states and 7,680 rendered rows before audits and auxiliary inputs. Candidate reversal reuses states.

Source competence is audited on 960 questions per cell using fresh states and the original source rules. These audits measure competence after source-natural selection and do not choose the feature, architecture, regularization, or seed. The six seed-17 selected readouts and six replication LoRA checkpoints are frozen before generating questions for the 26 reserved components. Its coverage and predeclared decision rule appear in Appendix F.

## A.10 Macro Metrics and Paired Comparisons

Let T be the four task families, $\textstyle { \boldsymbol { \mathcal { K } } } _ { t }$ their option counts, $\mathcal { R } _ { t K }$ the rules present in a natural stratum, and $I _ { t K r }$ its item indices. For per-item score ${ { a } _ { i } } ,$ define

$$
\mathcal { M } ( a ) = \frac { 1 } { \vert \mathcal { T } \vert } \sum _ { t \in \mathcal { T } } \frac { 1 } { \vert \mathcal { K } _ { t } \vert } \sum _ { K \in \mathcal { K } _ { t } } \bar { a } _ { t K } ,\tag{5}
$$

$$
\bar { a } _ { t K } = \frac { 1 } { \left| \mathcal { R } _ { t K } \right| } \sum _ { r \in \mathcal { R } _ { t K } } \frac { 1 } { \left| I _ { t K r } \right| } \sum _ { i \in I _ { t K r } } a _ { i } .\tag{6}
$$

Natural accuracy uses $a _ { i } = { \bf 1 } [ \hat { y } _ { i } = y _ { i } ]$ . Source selection uses $a _ { i } = - \log p _ { i } ( y _ { i } )$ , the candidatenormalized negative log-likelihood. This rule-level average prevents an imbalance in rule counts from changing a task’s weight. Training cross-entropy instead weights task families and then option counts equally over the available training rows. The objective balances task and option-count strata without balancing the observed semantic answer classes.

For diagnostic quartets $\mathcal { Q } _ { t K }$ , let $c _ { q j }$ indicate correctness at position j and $\begin{array} { r } { C _ { q } = \sum _ { j = 1 } ^ { 4 } c _ { q j } } \end{array}$ . The within-stratum item score is $\mathbb { E } _ { q } [ C _ { q } / \dot { 4 } ]$ and the allfour score is $\mathbb { E } _ { q } [ \mathbf { 1 } ( C _ { q } = 4 ) ]$ . Both are averaged over K and task as in Equation 3. We also report the distribution $\operatorname* { P r } ( C _ { q } = c )$ for $c = 0 , \ldots , 4$ These macro distributions need not yield integer group counts when multiplied by the number of quartets. Paired rule- and state-switch outcomes distinguish both answers correct, an unchanged semantic prediction, and a changed semantic prediction with an incorrect answer in either item.

<table><tr><td>Role per replication cell</td><td>Questions</td><td>States</td><td>Permitted use</td></tr><tr><td>Source natural training</td><td>2,560</td><td>320</td><td>Gradient fitting. Source-training capability report.</td></tr><tr><td>Source diagnostic training</td><td>320</td><td>160</td><td>Gradient fitting. Source-training quartet report.</td></tr><tr><td>Reversed source renderings</td><td>2,880</td><td>0 new</td><td>Same labels after semantic option remapping. Fitting augmentation.</td></tr><tr><td>Source natural validation</td><td>640</td><td>80</td><td>Checkpoint, architecture, feature, and regularization selection where those choices are in the protocol.</td></tr><tr><td>Source diagnostic validation</td><td>320</td><td>160</td><td>Joint-response reporting and earlier operational qualification. Not the checkpoint-selection objective.</td></tr><tr><td>Development natural</td><td>640</td><td>80</td><td>New-rule natural evaluation.</td></tr><tr><td>Development diagnostic</td><td>320</td><td>160</td><td>New-rule item/quartet evaluation and matched auxiliary conditions.</td></tr><tr><td>Later source natural audit</td><td>640</td><td>80</td><td>Qualification after richer-readout selection is locked. No reselection.</td></tr><tr><td>Later source diagnostic audit</td><td>320</td><td>160</td><td>Quartet qualification after richer-readout selection is locked. No reselection.</td></tr></table>

Table 6: Training, selection, and evaluation use distinct data roles in each replication cell. Reversed renderings augment training without adding complete states; audit data remain excluded from all model-selection decisions.

Replication means average the three state seeds within each rule split and then the two splits equally. D2–D3 contrasts are paired within each cell before this aggregation. Reserved-rule comparisons first aggregate within a rule block, then over the six fixed model pairs, then over the 13 blocks equally. Diagnostic reserved-rule scores cover feasible strata only. Missing quartets are disclosed rather than imputed. Shared rules, states, model fits, and quartet members are retained as dependencies. The aggregation retains these shared dependencies. Quartet success is measured on the crossed questions rather than inferred from the fourth power of the separately measured natural item accuracy.

## B Model Interfaces, Training, and Readout Selection

## B.1 Input Rendering and Candidate Scoring

Controlled Base, LoRA, and the later frozen readouts use the prose renderer exemplified in $\mathsf { A p - }$ pendix A.6. The state, explicit formula, task instructions, and semantic candidate text are available in the input. Gold labels remain withheld.

Candidate letters are checked to be distinct single nonspecial tokens after the answer prefix. In the controlled path, the BF16 text backbone produces the final-normalized hidden representation h and its legal candidate logits are

$$
z _ { k } = E _ { k } ^ { \top } h , \qquad p _ { k } = \frac { \exp { z _ { k } } } { \sum _ { j = 1 } ^ { K } \exp { z _ { j } } } ,\tag{7}
$$

with the candidate projection evaluated in FP32 using the original output rows $E _ { k }$ . Semantic prediction is the candidate argmax. Training optimizes this restricted candidate loss rather than a free-text generation objective. The original embeddings and candidate rows stay frozen during LoRA adaptation. Inputs are encoded without an explicit length-truncation cap and left-padded to the longest prompt in each batch. Appendix L.2 separately tests fixed padding and FP32 model inference, distinct from the FP32 candidate projection used here.

External systems retain their recorded interfaces: Decider uses its decision prompt, StartLux its native interface, and Jev the native TypeSafe structured request. Such comparisons characterize deployed system configurations and do not isolate a common architecture or training recipe. Table 7 records the checkpoint and service identifiers.

## B.2 LoRA Adaptation

LoRA is attached to all linear modules of the text decoder in the common Qwen Base checkpoint, with rank $1 6 , \alpha = 3 2 .$ , scale $\alpha / r = 2$ , dropout 0.05, and no trained bias. Only adapter tensors within the language decoder are trainable, totaling 16,819,200 parameters. Each replication cell starts from the same unadapted checkpoint. Training uses AdamW at learning rate $1 0 ^ { - 4 }$ , weight decay $1 0 ^ { - 4 }$ gradient-norm clipping at 1, batch size 16, and no scheduler. Non-reentrant gradient checkpointing is enabled. Two epochs over 5,760 rows yield 720 optimizer updates per cell. Source-natural macro

NLL chooses between the epoch-1 and epoch-2 checkpoints, with any exact tie resolved in favor of the earlier of the two saved epoch checkpoints.

The data include original and reversed candidate orders, and task/K weighting is applied to candidate cross-entropy. The procedure does not use oracle intermediate-value labels, development examples, or audit scores for training or selection. The six replication fits share optimizer seed 17. The initial fits additionally use seeds 29 and 43 on their common dataset. These two uses of seed variation have different scientific interpretations.

## B.3 Frozen Residual Readouts

All learned frozen readouts add a correction to the base logits rather than replacing them:

$$
z ^ { \prime } = z _ { \mathrm { b a s e } } + g _ { \theta } ( \tilde { h } ) , \qquad \tilde { h } _ { j } = \frac { h _ { j } - \mu _ { j } } { \operatorname* { m a x } ( \sigma _ { j } , 1 0 ^ { - 6 } ) } .\tag{8}
$$

Population means and standard deviations use source-training rows only. Scores beyond the legal K candidates are masked. The fixed rank-4 readout uses $g _ { \theta } ( \tilde { h } ) = ( \tilde { h } V ) ( E U ) ^ { \top }$ , with $U , V \in \mathbb { R } ^ { 2 0 4 8 \times 4 } ,$ fixed candidate rows E, and zero-initialized U. It trains with learning rate 0.003 and L2 strength $1 0 ^ { - 4 }$ for 800 updates, evaluating source-natural NLL every 40 steps. Replication reuses this configuration.

Readout selection compares three feature sets. The final-normalized final-position vector has 2,048 dimensions. Concatenating final-position outputs at layers 8 and 16 with the final-normalized output gives 6,144 dimensions. Final-layer means over the state, question, and option regions also give 6,144 dimensions. The question region includes the rule text. These rendered-input features omit the numeric values computed by the oracle.

Each feature set is paired with a linear correction or a GELU MLP with 128 hidden units and eight outputs. L2 strength is selected from $\{ 1 0 ^ { - 4 } , 1 0 ^ { - 2 } \}$ and the output layer starts at zero. Both families minimize

$$
\mathcal { L } ( \theta ) = \sum _ { i \in \mathcal { D } _ { \mathrm { t r a i n } } } w _ { i } \big [ - \log p _ { \theta } ( y _ { i } \mid i ) \big ] + \frac { \lambda } { 2 } \| \theta \| _ { 2 } ^ { 2 } ,\tag{9}
$$

where $w _ { i }$ gives equal mass to tasks and then option counts. L2 covers trained weights and biases. Linear corrections use FP64 L-BFGS with strong-Wolfe line search, at most 40 chunks of 20 iterations, and history size 30. The stopping threshold is a maximum absolute gradient of $1 0 ^ { - 6 }$ , with source validation after each chunk. MLPs use FP32 fullbatch Adam at learning rate 0.003 for 800 updates, with gradient clipping at 1. We evaluate sourcevalidation NLL after every 40 optimizer updates for source-based checkpoint selection using the same candidate-normalized decision-scoring rule.

Source-natural validation NLL selects the configuration with seed 17. Exact ties favor fewer parameters, then the earlier checkpoint, then the fixed configuration order. Seeds 29 and 43 refit the selected configuration without reselection. Every cell selects MLP128 with $\lambda = 0 . 0 1$ . Split 1/state 202 uses multilayer features with 787,592 parameters. The other cells use final-position features with 263,304 parameters. Reserved-rule evaluation uses the six seed-17 selected checkpoints. LoRA and readouts share training examples but differ in parameter count, optimization, and computation.

The correct-count figure also reports rank-4 readouts fitted with cross-entropy, true-pair regularization, or random-pair regularization. These variants use 800 full-batch Adam updates at 0.003, with L2 selected from $\{ 1 0 ^ { - 4 } , 1 0 ^ { - \bar { 2 } } \}$ . Source-natural NLL selects checkpoints every 40 updates. True pairs share a state but use different rules and gold answers. Random pairs reassign the second member within task/K/directed-label-pair strata, requiring different states. Strata without a valid reassignment are omitted from both paired variants. For labels $( a , b )$ , define

$$
d _ { i j } = ( z _ { i , a } ^ { \prime } - z _ { i , b } ^ { \prime } ) - ( z _ { j , a } ^ { \prime } - z _ { j , b } ^ { \prime } ) .\tag{10}
$$

The added task/K-weighted loss is 0.25 softplus $( 1 ~ - ~ d _ { i j } )$ Both paired variants retain identical cross-entropy training rows and the same source-natural validation selection criterion.

## B.4 Precision and Interface Qualification

The fixed rank-4 replication cache uses BF16 logits from the full model output and FP32 hidden features. Its BF16 Base reference is distinct from Controlled Base. The later richer readouts and reservedrule evaluation use the controlled FP32 candidateletter projection shared with Base and LoRA. The rank-4 versus controlled-Base contrast therefore combines readout differences with a change in the numerical precision of candidate-logit projection.

StartLux quality scores use FP32 inference. The probability-equivalence check passes with a maximum difference of 0.000800 against tolerance 0.01. The check verifies probability agreement for the scoring interface within the prescribed tolerance.

<table><tr><td>Model repository</td><td>Frozen revision / response version</td></tr><tr><td>Qwen/Qwen3.5-2B-Base</td><td>b1485b2fa6dfa1287294f269f5fb618e03d52d7c</td></tr><tr><td>Qwen/Qwen3.5-2B</td><td>15852e8c16360a2fea060d615a32b45270f8a8fc</td></tr><tr><td>Mapika/decider-2b startlux-models/StartLux-Decision-2B</td><td>533964dae8be954c5b5e19fa4948e48408094c1e</td></tr><tr><td>TypeSafe Jev (request alias: jev-latest)</td><td>3e2e456409a7fe44be69eee231566f7830872ea2 jev-1.13.0 (successful responses)</td></tr></table>

Table 7: Fixed checkpoint revisions and the recorded service version identify the evaluated systems. Jev’s recorded version identifies the service response, while its internal checkpoint and parameter count remain undisclosed.

Jev’s successful responses all identify version jev-1.13.0. In initial repeat controls, semantic agreement is 90.625% across 32 repeated requests, with maximum probability change 0.12. The 43 eligible option-reversal comparisons have 93.02% semantic agreement. Elapsed time and service stochasticity are not isolated by that comparison. We therefore do not treat the service as deterministic. Source, development, and reserved labels specify the evaluation allocation; they make no assertion about an external model’s pretraining data.

## C Detailed Behavioral Results

## C.1 Initial System Comparison

Higher item accuracy coexists with weak joint correctness across decision systems. Table 8 compares released systems and controlled adaptation. LoRA improves the controlled backbone, while decision systems show different source and development strengths. Their interfaces and training histories define distinct system configurations.

The initial comparison averages LoRA over three fit seeds. Source scores use validation inputs. Natural and diagnostic scores use separately sampled questions. Native BF16 and controlled FP32 projection paths are identified separately for each system in the result table and Appendix B.

Across the three optimizer seeds, LoRA has standard deviations of 3.50 percentage points in newrule natural accuracy and 1.04 in quartet QAcc.

## C.2 Replication by Rule Split and State Seed

Jev and LoRA change their relative ranking across rule splits. Figure 5 retains every development cell. LoRA reaches higher natural accuracy on split 1 than on split 2, while Jev shows the opposite ordering. The pooled means summarize these tested allocations; the cell scores show how system rankings change with the assigned source rules.

Tables 9 and 10 give cell scores from Figure 5.

## C.3 Task Profiles

Figure 6 reports all task-level diagnostic means. Within each cell, option counts receive equal weight within task, followed by equal state-seed and rule-split weights. Every task/K stratum contains 64 natural and 32 diagnostic questions, including eight complete quartets. Natural items and diagnostic quartets use separately sampled complete states within each model evaluation cell.

## C.4 Correct-Count Distributions and Required Response Changes

The quartet analysis records partial success and two distinct failures to respond. Figure 7 reanalyzes saved initial-batch predictions without new inference. It includes the original model and seed evaluations. Source and development panels each contain 320 questions in 80 complete quartets per system. Rows use the base rendering. Optionreversal scores are not mixed into this analysis. Macro weighting applies to the correct-count distribution; its percentages need not convert directly to unweighted integer counts of evaluated quartets. LoRA reduces nonresponse in the initial development sample without eliminating it. For seed 17, unchanged semantic answers occur on 35.42% of required rule switches and 35.94% of required state switches, compared with Controlled Base’s 75.52% and 84.90%. LoRA is correct on both members of only 35.94% of rule-switch pairs and 41.67% of state-switch pairs. Thus, responsiveness, pair correctness, and QAcc remain different behavioral quantities. The saved reassociation references preserve within-stratum correctness marginals. They are descriptive controls, not causal baselines or an IID statistical sampling model.

<table><tr><td>System</td><td>Source rules</td><td colspan="3">Development rules</td></tr><tr><td></td><td>Natural</td><td>Natural</td><td>Diagnostic item</td><td>QAcc</td></tr><tr><td>Qwen Base, native BF16</td><td>39.58</td><td>34.90</td><td>33.07</td><td>0.00</td></tr><tr><td>Qwen Instruct</td><td>37.63</td><td>35.03</td><td>31.51</td><td>3.13</td></tr><tr><td>Decider v11</td><td>43.10</td><td>39.32</td><td>38.80</td><td>3.13</td></tr><tr><td>StartLux 2B, FP32 research path</td><td>38.28</td><td>38.54</td><td>41.67</td><td>6.25</td></tr><tr><td>Controlled Base</td><td>38.15</td><td>35.68</td><td>33.33</td><td>0.00</td></tr><tr><td>LoRA, mean over fit seeds</td><td>77.99</td><td>56.51</td><td>56.86</td><td>18.75</td></tr><tr><td>Jev</td><td>60.68</td><td>55.86</td><td>49.74</td><td>10.42</td></tr></table>

Table 8: Higher item accuracy coexists with lower all-four correctness across the tested systems. Scores are macro-averaged percentages across the tested tasks. Each system retains its interface and training history.

(a) Development: Natural accuracy
<table><tr><td rowspan=3 colspan=1>1/ 1011/2021/303</td><td rowspan=1 colspan=1>41.15</td><td rowspan=1 colspan=1>38.54</td><td rowspan=1 colspan=1>60.16</td><td rowspan=1 colspan=1>57.42</td></tr><tr><td rowspan=1 colspan=1>39.58</td><td rowspan=1 colspan=1>37.37</td><td rowspan=1 colspan=1>61.46</td><td rowspan=1 colspan=1>57.55</td></tr><tr><td rowspan=1 colspan=1>36.72</td><td rowspan=1 colspan=1>40.23</td><td rowspan=1 colspan=1>65.36</td><td rowspan=1 colspan=1>60.03</td></tr><tr><td rowspan=3 colspan=1>2/ 1012/2022/303</td><td rowspan=1 colspan=1>36.46</td><td rowspan=1 colspan=1>37.76</td><td rowspan=1 colspan=1>52.08</td><td rowspan=1 colspan=1>62.11</td></tr><tr><td rowspan=1 colspan=1>40.10</td><td rowspan=1 colspan=1>39.19</td><td rowspan=1 colspan=1>53.91</td><td rowspan=1 colspan=1>63.02</td></tr><tr><td rowspan=1 colspan=1>38.02</td><td rowspan=1 colspan=1>41.54</td><td rowspan=1 colspan=1>55.21</td><td rowspan=1 colspan=1>62.50</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>Base    Rank-4    LoRA</td><td rowspan=1 colspan=1>Jev</td></tr></table>

(b) Reserved: Natural accuracy  
![](images/0ee3d99dad753a16ffdd60b11555d6ba7e5f3b6850a411e14a40040a09ab5b41.jpg)

(c) Development: QAcc
<table><tr><td rowspan=3 colspan=1>1/101Rule spe /  sed1/2021/303</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>1.04</td><td rowspan=1 colspan=1>16.67</td><td rowspan=1 colspan=1>21.88</td></tr><tr><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>13.54</td><td rowspan=1 colspan=1>19.79</td></tr><tr><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>13.54</td><td rowspan=1 colspan=1>17.71</td></tr><tr><td rowspan=1 colspan=1>2/ 101</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>7.29</td><td rowspan=1 colspan=1>23.96</td></tr><tr><td rowspan=1 colspan=1>2/202</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>18.75</td><td rowspan=1 colspan=1>28.13</td></tr><tr><td rowspan=1 colspan=1>2/303</td><td rowspan=1 colspan=1>3.13</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>9.38</td><td rowspan=1 colspan=1>20.83</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>Base    Rank-4</td><td rowspan=1 colspan=1>LoRA</td><td rowspan=1 colspan=1>Jev</td></tr></table>

(d) Reserved: QAcc  
![](images/2c519cf8d03da9f914e7c3655562f18aacf30767fe234ec9f1fd464b7961c19d.jpg)  
Figure 5: LoRA gains recur across rule splits and state seeds, while joint correctness varies across cells. Left panels retain all development systems; right panels compare source-selected MLPs with LoRA on reserved rules. Numbers give accuracy (%). Natural accuracy uses a shared 0–100 color scale, while QAcc uses a shared 0–30 scale across all development and reserved panels for the corresponding metric, keeping cell comparisons on the same scale.

## D Information and Answer-Binding Controls

## D.1 Controlled Input Manipulations

D0 preserves the original diagnostic item. D1 adds the supplementary-information instruction without supplying new values. D2 appends a table of every rule’s oracle value on every record. D3 uses the identical formula/record-key structure with each numeric entry replaced by unavailable. D4 supplies values for the requested rule only, with D5 as its missing-value counterpart. D6 supplies the correct semantic answer text, with D7 replacing it by unavailable. Conditions D1–D7 remove the instruction to ignore other fields. Their shared instruction specifies the exact formula, thresholds, tiebreak, and binding of supplied semantic answers to displayed options. Missing entries instruct the model to use the original input facts or records.

D2 and D3 preserve original coordinates, question, options, and gold answer. The table is ordered by formula and includes every rule in the relevant pool. Questions sharing a state therefore receive the same table across rule changes. In replication this covers eight development rules. D4 and D6 contain question-dependent information and therefore alter the information held fixed across a rule switch. All auxiliary conditions are reported separately from original-task accuracy and were not used in the training of any of the evaluated models.

<table><tr><td></td><td></td><td colspan="2">Base</td><td colspan="2">Rank-4 readout</td><td colspan="2">LoRA</td><td colspan="2">Jev</td></tr><tr><td>Split</td><td>State seed</td><td>Natural</td><td>QAcc</td><td>Natural</td><td>QAcc</td><td>Natural</td><td>QAcc</td><td>Natural</td><td>1QAcc</td></tr><tr><td>1</td><td>101</td><td>41.15</td><td>0.00</td><td>38.54</td><td>1.04</td><td>60.16</td><td>16.67</td><td>57.42</td><td>21.88</td></tr><tr><td>1</td><td>202</td><td>39.58</td><td>0.00</td><td>37.37</td><td>0.00</td><td>61.46</td><td>13.54</td><td>57.55</td><td>19.79</td></tr><tr><td>1</td><td>303</td><td>36.72</td><td>0.00</td><td>40.23</td><td>0.00</td><td>65.36</td><td>13.54</td><td>60.03</td><td>17.71</td></tr><tr><td>2</td><td>101</td><td>36.46</td><td>0.00</td><td>37.76</td><td>0.00</td><td>52.08</td><td>7.29</td><td>62.11</td><td>23.96</td></tr><tr><td>2</td><td>202</td><td>40.10</td><td>0.00</td><td>39.19</td><td>0.00</td><td>53.91</td><td>18.75</td><td>63.02</td><td>28.13</td></tr><tr><td>2</td><td>303</td><td>38.02</td><td>3.13</td><td>41.54</td><td>0.00</td><td>55.21</td><td>9.38</td><td>62.50</td><td>20.83</td></tr></table>

Table 9: LoRA gains recur across development cells, while the relative ranking of LoRA and Jev changes across rule splits. Scores are percentages, with equal weight assigned to each task family within every evaluated cell.
<table><tr><td>Split</td><td>State seed</td><td>natural</td><td>Readout Readout QAcc</td><td>LoRA natural</td><td>LoRA QAcc</td><td>Gap (pp)</td></tr><tr><td>1</td><td>101</td><td>40.22</td><td>0.00</td><td>52.48</td><td>7.48</td><td>+12.26</td></tr><tr><td>1</td><td>202</td><td>42.69</td><td>1.92</td><td>53.83</td><td>7.37</td><td>+11.14</td></tr><tr><td>1</td><td>303</td><td>42.29</td><td>0.00</td><td>55.49</td><td>4.27</td><td>+13.20</td></tr><tr><td>2</td><td>101</td><td>42.29</td><td>2.03</td><td>61.98</td><td>5.77</td><td>+19.69</td></tr><tr><td>2</td><td>202</td><td>42.53</td><td>0.96</td><td>57.75</td><td>13.57</td><td>+15.22</td></tr><tr><td>2</td><td>303</td><td>44.95</td><td>3.31</td><td>60.96</td><td>18.91</td><td>+16.01</td></tr></table>

Table 10: Every fixed LoRA model exceeds its selected readout on reserved-rule natural accuracy. Scores are percentages, and gaps are percentage points with equal weights across all of the reserved evaluation rule blocks.

<table><tr><td>Model</td><td>Task</td><td>Item accuracy</td><td>QAcc</td></tr><tr><td>Base</td><td>Binary</td><td>48.96</td><td>2.08</td></tr><tr><td>Base</td><td>Rating</td><td>26.39</td><td>0.00</td></tr><tr><td>Base</td><td>Routing</td><td>27.43</td><td>0.00</td></tr><tr><td>Base</td><td>Selection</td><td>32.47</td><td>0.00</td></tr><tr><td>Rank-4 readout</td><td>Binary</td><td>50.00</td><td>0.00</td></tr><tr><td>Rank-4 readout</td><td>Rating</td><td>29.69</td><td>0.00</td></tr><tr><td>Rank-4 readout</td><td>Routing</td><td>30.21</td><td>0.69</td></tr><tr><td>Rank-4 readout</td><td>Selection</td><td>30.56</td><td>0.00</td></tr><tr><td>LoRA</td><td>Binary</td><td>75.00</td><td>22.92</td></tr><tr><td>LoRA</td><td>Rating</td><td>46.70</td><td>7.64</td></tr><tr><td>LoRA</td><td>Routing</td><td>42.01</td><td>9.03</td></tr><tr><td>LoRA</td><td>Selection</td><td>48.78</td><td>13.19</td></tr><tr><td>Jev</td><td>Binary</td><td>90.62</td><td>66.67</td></tr><tr><td>Jev</td><td>Rating</td><td>41.32</td><td>4.86</td></tr><tr><td>Jev</td><td>Routing</td><td>39.24</td><td>3.47</td></tr><tr><td>Jev</td><td>Selection</td><td>50.17</td><td>13.19</td></tr></table>

Table 11: Task-level item and joint scores distinguish binary strength from performance on ratings and record selection. Percentages give equal weight to option counts, state seeds, and rule splits within each evaluated task.

D4 and D6 reveal question-specific information, so their auxiliary all-four scores do not test the original rule/state response requirement. Table 12 reports the aggregate value effects across all tested conditions. The following positive controls examine the binding to the explicitly supplied answer.

## D.2 Replicated Auxiliary Effects

The all-rule value-table benefit varies strongly across the four measured systems. Figure 4 summarizes D2/D3 in the replication development batch. Each of the six cells contributes 320 diagnostic items paired between D2 and D3 within that cell. Aggregation averages K within task, tasks within cell, the three state seeds within a rule split, and finally the two rule splits. Across cells, LoRA item/QAcc differences range from −0.26 to 7.03 pp and −3.13 to 9.38 pp. Jev ranges are 30.47– 38.02 pp and 50.00–59.38 pp. These are two rule allocations with three state samples each. They are not six independently sampled rule replications.

## D.3 Answer-Binding Positive Controls

Answer-binding positive controls, in percent. All conditions reveal the correct semantic answer. Full/minimal varies context, and legacy/direct varies instruction framing. The 80 sourcevalidation indices retain repeated semantic inputs, so the denominator counts evaluated items rather than distinct, independently drawn semantic states.

![](images/dde1ccea4a7ea5b4dd3d7137592d896217f0241536759bd3ebf416d234c2f3e8.jpg)

Figure 6: Individual correctness and all-four correctness reveal different task profiles. Jev leads in binary QAcc; LoRA leads in rating and routing and ties Jev in selection. Both panels use the same development questions, task/split weighting, and 0–100% scale; labels give every individual and joint score to one decimal place.
<table><tr><td colspan="3">Item accuracy</td><td colspan="2">QAcc</td></tr><tr><td>Model</td><td>Missing values Complete values Missing values</td><td></td><td></td><td>Complete values</td></tr><tr><td>Base</td><td>35.72</td><td>40.19</td><td>0.00</td><td>2.08</td></tr><tr><td>Rank-4 readout</td><td>35.81</td><td>39.28</td><td>0.00</td><td>3.13</td></tr><tr><td>LoRA</td><td>49.48</td><td>52.95</td><td>7.47</td><td>11.98</td></tr><tr><td>Jev</td><td>54.43</td><td>90.63</td><td>20.31</td><td>75.69</td></tr></table>

Table 12: Complete values improve joint correctness most for Jev on the paired development inputs. Scores are percentages under the same task, option-count, state-seed, and rule-split macro weighting used in Figure 4.

The binding study uses eight item indices per task/option-count layer, selected from source natural validation with recorded answer rotation and identity ordering. The direct instruction removes the original task question and asks the model to bind the supplied answer to an option. Under minimal/direct conditions there are only 31 unique visible inputs per system, with repeated groups tracked separately. Local visible-input identities use effective token IDs. Jev identities use request bodies. These fingerprints are not interchangeable. The repeated groups have consistent semantic choices in this study, although some probability values vary. Correct-answer hints and value tables are not a monotone difficulty ladder. In the original diagnostic batch, Jev is correct under D2 but wrong under D6 on 21.88% of macro-weighted items. The reverse occurs on 4.17%. Providing the right answer does not guarantee that the interface will use it. The binding study is a source-validation positive control. Its source-validation rates and the original new-rule D6 rates use different evaluation inputs, preventing a direct subtraction that isolates the effect of changing instruction wording alone.

## E Source-Competence and Optimization Audits

The operational source gate requires natural accuracy of at least 90% and all-four diagnostic accuracy of at least 75%, on both original sourcetraining questions and independent source-audit questions. These thresholds define the premise needed for the proposed source-competent transfer analysis. They are not universal standards for rule understanding. Selection uses only sourcenatural validation NLL, with the gate evaluated afterward. The gate is checked separately on original and reversed-option source-training renderings.

![](images/6a2283b65c8dde5cb8db84fcf1216faf60eb3502e6ac019265b03bce0b109ee9.jpg)  
Figure 7: Partial correctness occupies much more probability mass than all-four success for most systems. Segment widths retain the macro-weighted percentage of quartets with zero to four correct answers; dark blue marks all-four success. Every model and seed is retained, and labels annotate segments with macro weights of at least 10%.

Selected richer readouts fail the sourcecompetence gate. All selected-configuration runs fail, including refits with seeds 17, 29, and 43. Across these runs, source-training natural accuracy ranges from 56.64% to 68.00%, source-audit natural accuracy from 41.80% to 52.08%, and audit QAcc from 0% to 7.29%. These are descriptive ranges over a finite set of fits. The six seed-17 models later used in the reserved-rule evaluation are a subset, with source-audit natural accuracy of 41.80–49.87% and all-four QAcc of 0–3.13%.

## F Prospective Evaluation on Reserved Rules

## F.1 Frozen Models, Blocks, and Coverage

After development, the six selected seed-17 richer readouts and six replication LoRA checkpoints are frozen. All 26 reserved semantic components are ordered by SHA256 of a fixed prefix and component ID, then paired adjacently into 13 blocks without rule reuse. For every block and task/K stratum,

16 new natural states are drawn before labels are computed and evaluated under both rules. Thus the natural evaluation has $1 3 \times 1 0 \times 1 6 \times 2 = 4 , 1 6 0$ questions. All rule blocks and models remain fixed.

Two diagnostic quartets per block/stratum are attempted under the fixed eligibility rule. The 254 feasible quartets out of 260 planned contain 1,016 diagnostic items. Six missing quartets come from block 11 binary/K = 2, block 11 ordinal/K = 3, and block 12 binary/K = 2, with two omitted quartets in each stratum. They remain coverage omissions. They do not trigger replacement rules or affect the primary natural evaluation. The combined 5,176 questions use 2,588 new complete states, disjoint from the 43,140-state historical ledger and disjoint across blocks. Both rules in each block are evaluated on the same natural states, retaining paired model comparisons within that rule block.

## F.2 Finite-Benchmark Decision Rule

Let $A _ { b , m } ^ { L }$ and $A _ { b , m } ^ { H }$ be macro natural accuracy for LoRA and the frozen readout in model pair m

![](images/4ce5596895171a8e59a8d19a69c41a51771cd3598c44b3988acf168fdc1b5ef4.jpg)  
Figure 8: Answer binding depends on context and instructions even when the correct semantic answer is supplied. Direct instructions raise local-model accuracy, while Jev performs best with minimal context and legacy instructions. Numbers give accuracy (%) using a single 0–100 sequential color scale shared across all systems, context lengths, and instruction conditions.

on block b. The prospectively specified LoRA– readout contrast is

$$
\Delta _ { b } = \frac { 1 } { 6 } \sum _ { m = 1 } ^ { 6 } ( A _ { b , m } ^ { L } - A _ { b , m } ^ { H } ) , \qquad \bar { \Delta } = \frac { 1 } { 1 3 } \sum _ { b = 1 } ^ { 1 3 } \Delta _ { b } .\tag{11}
$$

Before generating reserved-rule questions or predictions, the protocol requires $\bar { \Delta } \geq 0 . 0 5$ and at least ten of thirteen blocks to satisfy $\Delta _ { b } > 0 . 0 5$ Base scores, diagnostic QAcc, and task-specific results are secondary descriptions. A failure of either primary condition would be reported as failing the fixed criterion, without extending the experiment.

The fixed LoRA–readout comparison meets its predeclared finite-benchmark criterion. The mean difference is 14.59 percentage points, and all 13 block differences exceed five points, ranging from 9.51 to 19.53 points. Across-block means for the six separate model pairs range from 11.14 to 19.69 points. Averaging the three pairs from each source split gives 12.20 and 16.97 points. Figure 9 retains all thirteen individual rule-block effects.

Blocks do not reuse rules or complete states, but they share trained models and a finite grammar. The six model pairs also share this reserved evaluation. The pass criterion is consequently a statement about the reserved finite benchmark, not a population-level significance test or independent source-retraining confirmation. No IID row bootstrap, binomial block probability, or population p-value is assigned to this result. The criterion evaluates transfer by these fixed procedures on the prospective reserved-rule transfer benchmark.

Figure 5 reports each of the six fixed model pairs.

## G Probes and Counterfactual Rule Interventions

## G.1 Probe Data, Targets, and Fitting

Base and state-101 LoRA are tested in both splits. Each model/split cell has 2,880 source-training, 640 source-validation, 640 independent sourceaudit, and 640 development questions. Features are layers 8/16/24 at the final input and final question positions, plus input-embedding segment means. Source-training standardization uses a 10<sup>−6</sup> scale floor. A feature-name-seeded Gaussian projection reduces each feature to 256 dimensions, scaling entries by the inverse square root of input dimension.

Value targets are asinh $( f _ { r } ( x _ { j } , y _ { j } , z _ { j } ) / 1 0 )$ padded to eight columns and masked by valid records. Separate answer targets are eight-way one-hot vectors. Weighted ridge regression fits each column with an unpenalized intercept, renormalizing source-training weights over valid rows. Source-validation macro masked MSE selects L2 from {0.1, 1}. Error averages valid columns within items, then task/K/rule strata. Value probes reconstruct decisions by sign, transformed thresholds, or min/max selection. Answer probes predict the highest-scoring legal candidate among the candidate answers supplied for each evaluated task family and its prescribed legal option count.

The shuffled control permutes training labels within task/K with seed 1701 and retains the reallabel regularization. Figure 10 shows every location. These supervised targets are never inputs to the language model, and recoverable information alone does not establish causal use in answering.

## G.2 Donor–Receiver Pairs and Source Selection

Each intervention pairs a receiver $\left( { { s _ { r } } , { r _ { r } } } \right)$ with a donor $( s _ { d } , r _ { d } )$ from the same task and option-count stratum, requiring both $s _ { d } \neq s _ { r }$ and $r _ { d } \neq r _ { r }$ . The counterfactual target is

$$
{ \boldsymbol { y } } ^ { * } = g _ { t } ( { \boldsymbol { r } } _ { d } , s _ { r } ) ,\tag{12}
$$

the donor rule applied to the receiver state. Pair construction uses input identities and the oracle, not a model’s predictions. Pairs are stratified by task/K and target-answer equality with donor and receiver answers. Each stratum retains up to four pairs in deterministic order, prioritizing distinct donor, receiver, and counterfactual answers. The fixed source-audit and development evaluations each contain 160 pairs per model/split cell. Binary tasks cannot have three distinct answers and are not treated as isolating this answer property.

![](images/1832ac28e7c1f518db6d4b5a61883b3061fe376be1d2a95eb8146b088a241e21.jpg)  
Figure 9: LoRA exceeds the prospectively fixed 5 pp margin over the selected readout in every reserved block. Bars begin at zero and labels give the natural-accuracy difference (pp); the dashed line marks the decision margin.

(a) Base / split 1
<table><tr><td rowspan=1 colspan=1>L8 final</td><td rowspan=1 colspan=1>+4.3</td><td rowspan=1 colspan=1>+6.6</td><td rowspan=1 colspan=1>+0.3</td><td rowspan=1 colspan=1>-0.3</td></tr><tr><td rowspan=1 colspan=1>L8 question</td><td rowspan=1 colspan=1>+6.4</td><td rowspan=1 colspan=1>+6.1</td><td rowspan=1 colspan=1>+0.6</td><td rowspan=1 colspan=1>+3.4</td></tr><tr><td rowspan=1 colspan=1>L16 final</td><td rowspan=1 colspan=1>+8.2</td><td rowspan=1 colspan=1>+9.1</td><td rowspan=1 colspan=1>+3.0</td><td rowspan=1 colspan=1>-0.6</td></tr><tr><td rowspan=1 colspan=1>L16 question</td><td rowspan=1 colspan=1>+12.2</td><td rowspan=1 colspan=1>+9.6</td><td rowspan=1 colspan=1>+0.5</td><td rowspan=1 colspan=1>-0.1</td></tr><tr><td rowspan=1 colspan=1>L24 final</td><td rowspan=1 colspan=1>+5.3</td><td rowspan=1 colspan=1>+9.0</td><td rowspan=1 colspan=1>+1.2</td><td rowspan=1 colspan=1>-1.2</td></tr><tr><td rowspan=1 colspan=1>L24 question</td><td rowspan=1 colspan=1>+5.9</td><td rowspan=1 colspan=1>+6.1</td><td rowspan=1 colspan=1>+0.3</td><td rowspan=1 colspan=1>+0.6</td></tr><tr><td rowspan=1 colspan=1>Embedding</td><td rowspan=1 colspan=1>+6.0</td><td rowspan=1 colspan=1>+4.6</td><td rowspan=1 colspan=1>-0.4</td><td rowspan=1 colspan=1>-1.2</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Audit</td><td rowspan=1 colspan=1>Audit</td><td rowspan=1 colspan=1>Dev.</td><td rowspan=1 colspan=1>Dev.</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>value</td><td rowspan=1 colspan=1>answer</td><td rowspan=1 colspan=1>value</td><td rowspan=1 colspan=1>answer</td></tr></table>

(b) Base / split 2
<table><tr><td rowspan=1 colspan=1>L8 final</td><td rowspan=1 colspan=1>+6.8</td><td rowspan=1 colspan=1>+5.5</td><td rowspan=1 colspan=1>+4.9</td><td rowspan=1 colspan=1>+3.0</td></tr><tr><td rowspan=1 colspan=1>L8 question</td><td rowspan=1 colspan=1>+6.6</td><td rowspan=1 colspan=1>+9.0</td><td rowspan=1 colspan=1>+7.9</td><td rowspan=1 colspan=1>+2.5</td></tr><tr><td rowspan=1 colspan=1>L16 final</td><td rowspan=1 colspan=1>+9.8</td><td rowspan=1 colspan=1>+11.7</td><td rowspan=1 colspan=1>+8.5</td><td rowspan=1 colspan=1>+4.0</td></tr><tr><td rowspan=1 colspan=1>L16 question</td><td rowspan=1 colspan=1>+9.1</td><td rowspan=1 colspan=1>+11.7</td><td rowspan=1 colspan=1>+5.7</td><td rowspan=1 colspan=1>-1.6</td></tr><tr><td rowspan=1 colspan=1>L24 final</td><td rowspan=1 colspan=1>+8.1</td><td rowspan=1 colspan=1>+8.9</td><td rowspan=1 colspan=1>+2.7</td><td rowspan=1 colspan=1>+2.6</td></tr><tr><td rowspan=1 colspan=1>L24 question</td><td rowspan=1 colspan=1>+3.8</td><td rowspan=1 colspan=1>+6.6</td><td rowspan=1 colspan=1>+5.2</td><td rowspan=1 colspan=1>-0.7</td></tr><tr><td rowspan=1 colspan=1>Embedding</td><td rowspan=1 colspan=1>+1.7</td><td rowspan=1 colspan=1>+3.5</td><td rowspan=1 colspan=1>+0.4</td><td rowspan=1 colspan=1>+5.7</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Audit</td><td rowspan=1 colspan=1>Audit</td><td rowspan=1 colspan=1>Dev.</td><td rowspan=1 colspan=1>Dev.</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>value</td><td rowspan=1 colspan=1>answer</td><td rowspan=1 colspan=1>value</td><td rowspan=1 colspan=1>answer</td></tr></table>

(d) LoRA / split 2

(c) LoRA / split 1
<table><tr><td rowspan=1 colspan=1>L8 final</td><td rowspan=1 colspan=1>+8.7</td><td rowspan=1 colspan=1>+10.3</td><td rowspan=1 colspan=1>+2.5</td><td rowspan=1 colspan=1>+4.7</td></tr><tr><td rowspan=1 colspan=1>L8 question</td><td rowspan=1 colspan=1>+14.3</td><td rowspan=1 colspan=1>+14.1</td><td rowspan=1 colspan=1>+5.0</td><td rowspan=1 colspan=1>-1.4</td></tr><tr><td rowspan=1 colspan=1>L16 final</td><td rowspan=1 colspan=1>+30.3</td><td rowspan=1 colspan=1>+39.4</td><td rowspan=1 colspan=1>+11.1</td><td rowspan=1 colspan=1>+22.3</td></tr><tr><td rowspan=1 colspan=1>L16 question</td><td rowspan=1 colspan=1>+28.5</td><td rowspan=1 colspan=1>+31.5</td><td rowspan=1 colspan=1>+13.1</td><td rowspan=1 colspan=1>+16.9</td></tr><tr><td rowspan=1 colspan=1>L24 final</td><td rowspan=1 colspan=1>+38.9</td><td rowspan=1 colspan=1>+39.9</td><td rowspan=1 colspan=1>+21.0</td><td rowspan=1 colspan=1>+21.9</td></tr><tr><td rowspan=1 colspan=1>L24 question</td><td rowspan=1 colspan=1>+27.1</td><td rowspan=1 colspan=1>+31.2</td><td rowspan=1 colspan=1>+15.9</td><td rowspan=1 colspan=1>+16.7</td></tr><tr><td rowspan=1 colspan=1>Embedding</td><td rowspan=1 colspan=1>+6.0</td><td rowspan=1 colspan=1>+4.6</td><td rowspan=1 colspan=1>-0.4</td><td rowspan=1 colspan=1>-1.2</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Audit</td><td rowspan=1 colspan=1>Audit</td><td rowspan=1 colspan=1>Dev.</td><td rowspan=1 colspan=1>Dev.</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>value</td><td rowspan=1 colspan=1>answer</td><td rowspan=1 colspan=1>value</td><td rowspan=1 colspan=1>answer</td></tr></table>

<table><tr><td rowspan=1 colspan=1>L8 final</td><td rowspan=1 colspan=1>+7.3</td><td rowspan=1 colspan=1>+11.7</td><td rowspan=1 colspan=1>+2.7</td><td rowspan=1 colspan=1>+5.9</td></tr><tr><td rowspan=1 colspan=1>L8 question</td><td rowspan=1 colspan=1>+9.0</td><td rowspan=1 colspan=1>+8.6</td><td rowspan=1 colspan=1>+9.5</td><td rowspan=1 colspan=1>+7.6</td></tr><tr><td rowspan=1 colspan=1>L16 final</td><td rowspan=1 colspan=1>+27.7</td><td rowspan=1 colspan=1>+47.7</td><td rowspan=1 colspan=1>+13.3</td><td rowspan=1 colspan=1>+19.8</td></tr><tr><td rowspan=1 colspan=1>L16 question</td><td rowspan=1 colspan=1>+23.2</td><td rowspan=1 colspan=1>+34.2</td><td rowspan=1 colspan=1>+18.6</td><td rowspan=1 colspan=1>+14.7</td></tr><tr><td rowspan=1 colspan=1>L24 final</td><td rowspan=1 colspan=1>+29.7</td><td rowspan=1 colspan=1>+47.8</td><td rowspan=1 colspan=1>+17.8</td><td rowspan=1 colspan=1>+19.9</td></tr><tr><td rowspan=1 colspan=1>L24 question</td><td rowspan=1 colspan=1>+26.6</td><td rowspan=1 colspan=1>+40.4</td><td rowspan=1 colspan=1>+13.9</td><td rowspan=1 colspan=1>+11.7</td></tr><tr><td rowspan=1 colspan=1>Embedding</td><td rowspan=1 colspan=1>+1.7</td><td rowspan=1 colspan=1>+3.5</td><td rowspan=1 colspan=1>+0.4</td><td rowspan=1 colspan=1>+5.7</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Audit</td><td rowspan=1 colspan=1>Audit</td><td rowspan=1 colspan=1>Dev.</td><td rowspan=1 colspan=1>Dev.</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>value</td><td rowspan=1 colspan=1>answer</td><td rowspan=1 colspan=1>value</td><td rowspan=1 colspan=1>answer</td></tr></table>

Figure 10: Probe gains over shuffled targets are stronger on the source audit than on development inputs at many LoRA locations. Numbers give real minus shuffled-target accuracy (pp) for value and answer targets at every tested location. The diverging scale is centered at zero and shared across every evaluated layer and model panel.

Candidate intervention locations are the final question positions at layers 8 and 16.

Source-training diagnostic pairs provide samestate/different-rule representation differences. Their singular value decomposition supplies a four-dimensional orthonormal basis B from the leading right singular vectors. This defines a rule-associated direction set, not a proven rule-only representation. For receiver and donor hidden vectors $h _ { r } , h _ { d }$ , the two candidate patch families are

$$
h _ { r } ^ { \prime } = h _ { d }\tag{13}
$$

$$
\begin{array} { r } { h _ { r } ^ { \prime } = h _ { r } + B B ^ { \top } ( h _ { d } - h _ { r } ) \quad \mathrm { ( s u b s p a c e ) } . } \end{array}\tag{14}
$$

The amplitude is fixed at one. Source-validation target success selects the layer and mode by maximizing patch success minus the best of native, matchednorm random-direction, and answer-direction success. Exact ties prefer the earlier layer and then the subspace mode. Evaluation retains this source choice. Audit and development inputs evaluate this source-selected intervention without changing its selected layer, mode, or replacement amplitude.

## G.3 Intervention Controls and Numerical Validation

The controls include native prediction and restoration, the selected patch, and same-rule or sameanswer donors. Additional controls use matchednorm random and answer directions, rule-subspace and random erasure, and reversal of candidate order while applying the same selected intervention.

Same-rule donors change the state while retaining the receiver rule. Same-answer donors use another state and rule whose answer matches the receiver’s answer. Answer directions subtract sourcetraining answer prototypes within task/K strata and match the selected patch norm. Random directions are deterministically seeded per pair and layer. Rule-subspace erasure subtracts $B B ^ { \top } ( h _ { r } - \mu )$ using the source-training mean $\mu .$ . Random erasure uses a perturbation of the same norm. Donors and receivers share a checkpoint and use vectors from the same source-selected layer and token position.

Receivers are captured in the native forward batch, and restoration must retain identical choices and probabilities within $1 0 ^ { - 5 }$ . Reversed candidates use receivers captured under their own ordering. Source validation selects among layers 8 and 16 and full or subspace replacement, with amplitude fixed at one. Base/split 1 uses full replacement at layer 8; Base/split 2 uses the layer-16 subspace. Both LoRA cells use the layer-8 subspace. These fixed evaluation settings remain source-selected.

Applied BF16 norms, conversion errors, probabilities, and margins characterize the actual perturbation. Zeroing the question position and flipping the terminal sign provide positive controls for path sensitivity. Figure 11 compares counterfactualtarget success for the selected patch and the retained donor, direction, erasure, and restoration controls on paired audit and development inputs.

For Base/split 1, Base/split 2, LoRA/split 1, and LoRA/split 2, applied BF16 relative norms are 4.84%, 3.39%, 2.19%, and 4.07%, respectively. Prediction changes are 1.56%, 0.00%, 0.00%, and 0.00%. Native and patched target success for both source-rule splits appear separately in Figure 11.

Each audit/development cell has 160 donor– receiver pairs, macro-averaged over task and option count. The relative norm is $1 0 0 \| h _ { r } ^ { \mathrm { a } }$ pplied $h _ { r } \| _ { 2 } / \operatorname* { m a x } ( \| h _ { r } \| _ { 2 } , 1 0 ^ { - 1 2 } )$ , averaged with the same task/K weights. The applied vector includes the actual BF16 cast. A value of 4.84 thus denotes a change of 4.84% relative to the receiver norm.

## H Joint-Metric Calibration and Conditional Uncertainty

Stratified reassociation characterizes the joint reference implied by the saved prediction marginals. Calibration reuses predictions from the fixed models. Within each cell, reassociation preserves prediction marginals conditional on task/K, quartet slot, and true label. It uses 1,000 reassociations to obtain descriptive quantile ranges. This reference breaks associations within the original quartets. It is neither an independent-row sampling model nor a causal test of rule execution. The development replication contains 480 complete quartets, and the reserved rules yield 254 feasible quartets. Calibration covers the feasible task/K strata recorded in the fixed diagnostic evaluation sets.

The deterministic controls use only originalrendering source-training natural data. Let $R _ { S }$ be its distinct rules and $S _ { t , K }$ its distinct complete states for task t and option count K. Let M return the modal answer index, breaking ties toward the smallest index. The source-label-majority control returns the mode of source labels within $( t , K )$ The other controls are

$$
h _ { \mathrm { r u l e - b l i n d } } ( s , t , K ) = M \{ g _ { t } ( r , s ) : r \in R _ { S } \} ,\tag{15}
$$

$$
h _ { \mathrm { s t a t e - b l i n d } } ( r , t , K ) = M \{ g _ { t } ( r , s ) : s \in S _ { t , K } \} .\tag{16}
$$

Oracle answers use the original option-index convention, including record-ID tie-breaking. Thus the rule-blind control votes over source rules on the presented state, while the state-blind control votes over source states under the presented rule. Neither uses development labels to fit or select predictions.

On these diagnostic questions, source-labelmajority, rule-blind, and state-blind controls obtain item accuracies of 34.20%, 40.32%, and 33.64%, with zero QAcc. Uniform independent choices give 32.72% item accuracy and 2.79% QAcc. Paired intervals use 2,000 bootstrap resamples with fixed seeds. Natural questions that share a complete state remain grouped, and diagnostic questions are resampled as complete quartets. Models and conditions share resampling weights, retaining the original task/K and cell strata. Reserved-rule comparisons first compute metrics within rule blocks and then average equally over fixed models and blocks. These intervals are conditional on the existing rule splits, models, and evaluation samples. They do not cover new rule allocations, training seeds, or API service variation, and retain the dependence among cells that share the same source-rule allocation.

<table><tr><td rowspan=1 colspan=10>(a) Source audit                               (b) Development</td></tr><tr><td rowspan=1 colspan=1>Native</td><td rowspan=1 colspan=1>31.25</td><td rowspan=1 colspan=1>38.54</td><td rowspan=1 colspan=1>50.52</td><td rowspan=1 colspan=1>44.79</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>37.50</td><td rowspan=1 colspan=1>41.67</td><td rowspan=1 colspan=1>43.23</td><td rowspan=1 colspan=1>34.38</td></tr><tr><td rowspan=1 colspan=1>Selected patch</td><td rowspan=1 colspan=1>31.25</td><td rowspan=1 colspan=1>38.54</td><td rowspan=1 colspan=1>50.52</td><td rowspan=1 colspan=1>44.27</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>38.02</td><td rowspan=1 colspan=1>41.67</td><td rowspan=1 colspan=1>43.23</td><td rowspan=1 colspan=1>34.38</td></tr><tr><td rowspan=1 colspan=1>Same rule</td><td rowspan=1 colspan=1>31.25</td><td rowspan=1 colspan=1>38.54</td><td rowspan=1 colspan=1>50.52</td><td rowspan=1 colspan=1>44.27</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>39.06</td><td rowspan=1 colspan=1>41.67</td><td rowspan=1 colspan=1>43.23</td><td rowspan=1 colspan=1>34.38</td></tr><tr><td rowspan=1 colspan=1>Same answer, other rule</td><td rowspan=1 colspan=1>31.25</td><td rowspan=1 colspan=1>38.54</td><td rowspan=1 colspan=1>50.52</td><td rowspan=1 colspan=1>44.79</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>37.50</td><td rowspan=1 colspan=1>41.67</td><td rowspan=1 colspan=1>42.71</td><td rowspan=1 colspan=1>34.38</td></tr><tr><td rowspan=1 colspan=1>Random direction</td><td rowspan=1 colspan=1>31.77</td><td rowspan=1 colspan=1>38.54</td><td rowspan=1 colspan=1>50.52</td><td rowspan=1 colspan=1>44.79</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>36.98</td><td rowspan=1 colspan=1>41.67</td><td rowspan=1 colspan=1>43.23</td><td rowspan=1 colspan=1>34.90</td></tr><tr><td rowspan=1 colspan=1>Answer direction</td><td rowspan=1 colspan=1>31.25</td><td rowspan=1 colspan=1>38.54</td><td rowspan=1 colspan=1>50.52</td><td rowspan=1 colspan=1>44.27</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>36.98</td><td rowspan=1 colspan=1>41.67</td><td rowspan=1 colspan=1>43.23</td><td rowspan=1 colspan=1>34.38</td></tr><tr><td rowspan=1 colspan=1>Erase rule subspace</td><td rowspan=1 colspan=1>31.77</td><td rowspan=1 colspan=1>38.54</td><td rowspan=1 colspan=1>50.52</td><td rowspan=1 colspan=1>44.79</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>37.50</td><td rowspan=1 colspan=1>41.67</td><td rowspan=1 colspan=1>42.19</td><td rowspan=1 colspan=1>34.38</td></tr><tr><td rowspan=1 colspan=1>Random erasure</td><td rowspan=1 colspan=1>31.77</td><td rowspan=1 colspan=1>38.54</td><td rowspan=1 colspan=1>50.52</td><td rowspan=1 colspan=1>44.79</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>36.98</td><td rowspan=1 colspan=1>42.19</td><td rowspan=1 colspan=1>43.23</td><td rowspan=1 colspan=1>34.38</td></tr><tr><td rowspan=1 colspan=1>Same-batch restoration</td><td rowspan=1 colspan=1>31.25</td><td rowspan=1 colspan=1>38.54</td><td rowspan=1 colspan=1>50.52</td><td rowspan=1 colspan=1>44.79</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>37.50</td><td rowspan=1 colspan=1>41.67</td><td rowspan=1 colspan=1>43.23</td><td rowspan=1 colspan=1>34.38</td></tr><tr><td rowspan=1 colspan=1>Reverse candidates</td><td rowspan=1 colspan=1>32.29</td><td rowspan=1 colspan=1>42.71</td><td rowspan=1 colspan=1>46.35</td><td rowspan=1 colspan=1>46.88</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>30.73</td><td rowspan=1 colspan=1>40.10</td><td rowspan=1 colspan=1>42.19</td><td rowspan=1 colspan=1>35.42</td></tr><tr><td rowspan=1 colspan=1>Zero question position</td><td rowspan=1 colspan=1>31.25</td><td rowspan=1 colspan=1>37.50</td><td rowspan=1 colspan=1>50.52</td><td rowspan=1 colspan=1>43.75</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>39.06</td><td rowspan=1 colspan=1>41.67</td><td rowspan=1 colspan=1>43.23</td><td rowspan=1 colspan=1>34.90</td></tr><tr><td rowspan=1 colspan=1>Terminal sign flip</td><td rowspan=1 colspan=1>36.98</td><td rowspan=1 colspan=1>33.33</td><td rowspan=1 colspan=1>28.12</td><td rowspan=1 colspan=1>29.69</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>37.50</td><td rowspan=1 colspan=1>30.21</td><td rowspan=1 colspan=1>33.33</td><td rowspan=1 colspan=1>32.29</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Base</td><td rowspan=1 colspan=1>Base</td><td rowspan=1 colspan=1>LoRA</td><td rowspan=1 colspan=1>LoRA</td><td rowspan=1 colspan=2>Base</td><td rowspan=1 colspan=1>Base</td><td rowspan=1 colspan=1>LoRA</td><td rowspan=1 colspan=1>LoRA</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>split 1</td><td rowspan=1 colspan=1>split 2</td><td rowspan=1 colspan=1>split 1</td><td rowspan=1 colspan=1>split 2</td><td rowspan=1 colspan=2>split 1</td><td rowspan=1 colspan=1>split 2</td><td rowspan=1 colspan=2>split 1   split 2</td></tr></table>

Figure 11: Source-selected patches produce little change in counterfactual target success relative to the native baseline. Numbers retain absolute success (%); shading shows the within-column change from the native row (pp) on a shared, zero-centered scale. Audit and development retain all of the positive and negative controls.

Maximum-selection quartets show positive departures from the marginal reference. Table 13 separates tasks while preserving the original stratification. Both LoRA and Jev exceed the maximumselection reference range. Other task-level observations lie within their ranges. The pooled result therefore cannot be interpreted as absence of joint dependence in every task. Small, singleton, and constant-prediction strata limit how much the reference can change. Appendix L.8 reports their counts and metric weights, alongside maximum-selection scores for each evaluated cell and option count.

Task-specific development calibration, averaging six fixed cells. References independently reassociate slots within cell/task/K/gold strata (1,000 draws, seed 2026100802). The ranges summarize the reassociated reference distribution. They are descriptive quantiles, rather than confidence intervals or multiplicity-adjusted significance assessments.

## H.1 References Preserving Rule Pairs and State Responses

We reuse the 480 development quartets and compare two additional references with the primary slot-and-label reference. The first also conditions on the ordered pair of rule identities. The second shuffles the two predictions on a state together within the same rule-pair and ordered-label strata, independently for the two state positions. This preserves the observed response association within each state while breaking its link to the other state. It does not match latent state difficulty. Each reference uses 2,000 draws with independently derived seeds for each cell, task/K, and reference. Aggregation keeps equal weights over the fixed cells, option counts within tasks, and the four tasks.

<table><tr><td>Task</td><td></td><td></td><td></td><td>Model QAcc Ref. mean Ref. 95% range Obs. minus ref.</td><td></td></tr><tr><td>binary decision</td><td>Base</td><td>2.08</td><td>4.29</td><td>[0.00, 10.42]</td><td>-2.20</td></tr><tr><td>binary decision</td><td>Jev</td><td>66.67</td><td>66.55</td><td>[62.50, 70.83]</td><td>0.11</td></tr><tr><td>binary decision</td><td>LoRA</td><td>22.92</td><td>27.90</td><td>[20.83, 35.42]</td><td>-4.98</td></tr><tr><td>minimum routing</td><td>Base</td><td>0.00</td><td>0.07</td><td>[0.00, 0.69]</td><td>-0.07</td></tr><tr><td>minimum routing</td><td>Jev</td><td>3.47</td><td>3.00</td><td>[0.69, 5.56]</td><td>0.47</td></tr><tr><td>minimum routing</td><td>LoRA</td><td>9.03</td><td>5.90</td><td>[3.45, 9.03]</td><td>3.13</td></tr><tr><td>ordinal rating</td><td>Base</td><td>0.00</td><td>0.00</td><td>[0.00, 0.00]</td><td>0.00</td></tr><tr><td>ordinal rating</td><td>Jev</td><td>4.86</td><td>6.80</td><td>[4.17, 9.72]</td><td>-1.93</td></tr><tr><td>ordinal rating</td><td>LoRA</td><td>7.64</td><td>7.34</td><td>[4.86, 9.74]</td><td>0.30</td></tr><tr><td>maximum selection</td><td>Base</td><td>0.00</td><td>0.00</td><td>[0.00, 0.00]</td><td>0.00</td></tr><tr><td>maximum selection</td><td>Jev</td><td>13.19</td><td>8.75</td><td>[6.25, 11.81]</td><td>4.45</td></tr><tr><td>maximum selection</td><td>LoRA</td><td>13.19</td><td>7.60</td><td>[4.17, 10.42]</td><td>5.60</td></tr><tr><td colspan="6">Pooled, ordered rule pairs</td></tr><tr><td>All tasks</td><td>LoRA</td><td>13.19</td><td>13.33</td><td>[12.15, 14.58]</td><td>-0.13</td></tr><tr><td>All tasks</td><td>Jev</td><td>22.05</td><td>21.69</td><td>[20.66, 22.92]</td><td>0.35</td></tr><tr><td colspan="6">Pooled, rule pairs and state vectors</td></tr><tr><td>All tasks</td><td>LoRA 13.19</td><td></td><td>12.67</td><td>[12.15, 13.19]</td><td>0.52</td></tr><tr><td>All tasks</td><td>Jev</td><td>22.05</td><td>21.61</td><td>[20.66, 22.57]</td><td>0.44</td></tr></table>

Table 13: Joint-score departures depend on task and on the associations retained by reassociation. Task-specific QAcc is compared with 1,000-draw reference ranges; pooled rule-pair and state-vector controls use 2,000 draws. Scores are percentages. The reported ranges describe the reference distribution for each set of preserved associations.

Preserving rule pairs changes the jointcorrectness reference. The pooled controls appear in Table 13. In maximum selection, retaining rule pairs brings the reference means closer to observed QAcc for both LoRA and Jev. Some finer strata have no movable predictions or a constant QAcc, so adding conditions can also reduce the variation available to the reference. These comparisons characterize dependence on the existing pairing. The evaluation retains the constructed state pairs; it does not test quartets formed from new and independently sampled second complete states.

## I Common Source Audit and Bounded Competence Matching

The common audit uses 640 source-natural questions and 320 diagnostic items, forming 80 quartets per cell. These inputs are independent of training and model selection and shared by all methods. The prespecified source-competence gate requires natural accuracy of 90% and QAcc of 75% separately on the original training inputs and the independent, held-out source-rule audit evaluation questions.

On the common source audit, Base reaches natural accuracy / QAcc of 36.31% / 0.00%. The selected readout reaches 45.70% / 1.39%, and LoRA reaches 75.43% / 35.94%. None of the three methods satisfies the prespecified competence gate in any of the six fixed replication evaluation cells.

Bounded matching uses the original LoRA recipe and training seed 17 for 160 steps per cell. Selection uses the prespecified checkpoints at steps 0/10/20/40/80/160. Selection requires absolute differences from the frozen readout of at most 3 percentage points on both source-validation natural Acc and QAcc. Otherwise, the closest checkpoint is reported and the search stops. The sourcevalidation condition holds in 2/6 cells, but independent source-audit matching succeeds in 0/6. Neither audit nor development data are used for selection or to relax the tolerance. Figure 12 shows all six recorded checkpoints and the independent source-audit gaps for the individual checkpoints selected by source-validation score matching.

Development performance of all closest checkpoints is only a descriptive diagnostic. Its average difference cannot be interpreted as a transfer effect after controlling source competence, nor can unmatched cells be included in a matched group to support such a conclusion. The reproduction materials retain complete checkpoint trajectories and intervals conditional on the source-audit and development inputs used to compare these models.

![](images/87ecd3c759e3f57ba36e48c9b20c95907c21a507eb97768a9c3e26649c688a96.jpg)

(b) Split 1 / state 202  
![](images/9732abf8641435b40d55c49ac865bec3cddae3d4a324a652c659e58dfdc7b709.jpg)

(c) Split 1 / state 303  
![](images/c5a53ecd2275e38ec32071d6f03420a8f5a40f03e5cd2d6cfea33514a02d9242.jpg)

(d) Split 2 / state 101  
![](images/24007edbe172b8a89032cb3deb19956227751fb95e7034285e97ed2754202ede.jpg)

(e) Split 2 / state 202  
![](images/5a8a905c97a2d4a6518b57deb3c15d6b3c184d315f0c36e88ad84e0edd481b37.jpg)

(f) Split 2 / state 303  
![](images/df55c75d57ad07991e9a7eda6e6132ad3d0c7942f9e457debd613e33ba428bf2.jpg)  
Figure 12: Validation trajectories approach the frozen readout at selected early steps, but independent audit gaps remain. Curves show saved LoRA–readout gaps (pp); dotted vertical lines mark selected steps and the horizontal zero line denotes equality. Below-panel labels give audit natural-accuracy / QAcc gaps at those selected steps.

## J Equivalent Rewrites, Value Mappings, and Paired Answer Transitions

## J.1 Fixed Inputs and Semantic Verification

The supplement uses state seed 101 from both development splits. Complete quartets are selected by a fixed SHA256 ordering within task/K/rulepair strata: 40 quartets and 160 questions per split, totaling 80 quartets and 320 questions. Selection does not inspect predictions. Base, the corresponding LoRA, and Jev share the questions. The functional rewrite uses named operations such as sum and difference. The verbal rewrite replaces arithmetic symbols with words inside fully parenthesized expressions. For the archived rule min $( x , y ) - ( y + z )$ , the functional expression is difference(minimum(x, y), sum(y, z)), and the verbal expression is (min(x, y) minus (y plus z)). Both questions ask whether this value is strictly positive and parse back to the original AST. States, thresholds, tie-breaking rules, displayed options, and original gold answers are unchanged.

The correct value table retains the original D2 input. Formula-row mismatch misassigns complete computed-score rows to formulas, while recordcolumn mismatch applies one common column permutation to every formula. Both operations preserve the table’s value multiset and serialized character count. A nonidentity column permutation is impossible for single-record tasks, so their 128 questions are marked not applicable. Of the remaining 192 questions, 190 have table targets that conflict with the original oracle. Formula-row mismatch produces 220 conflicting questions.

The target-value counterfactual changes one table value for the requested formula so that the table target moves to the next option in a predefined cycle. Construction uses only the oracle, not model predictions. All 320 questions change their table targets. Value-range, string-length, and sign properties are matched in 292, 165, and 162 questions, respectively, and the remaining differences are retained. This table depends on the specific question, so shared-state QAcc is not reported for this condition. The original task gold answer is always retained, and following the incorrect table target does not count as correct on that original task.

## J.2 Paired Answer Transitions and Denominators

Define the conflict set C using only questions for which the intervention applies and the original oracle answer $o _ { i }$ differs from the manipulated table target $t _ { i } .$ Let $p _ { i } ^ { 0 }$ and $p _ { i } ^ { 1 }$ denote predictions with the correct and incorrect tables. Set $N \ = \ | C |$ $\begin{array} { r } { B = \sum _ { i \in C } \mathbf { 1 } [ p _ { i } ^ { 0 } = o _ { i } ] } \end{array}$ , and

$$
T = \sum _ { i \in C } \mathbf { 1 } [ p _ { i } ^ { 0 } = o _ { i } , p _ { i } ^ { 1 } = t _ { i } ] .\tag{17}
$$

We report the paired-transition proportion over all conflicting questions, $T / N$ , and the conditional proportion among initially correct questions, $T / B$ with integer numerators and denominators. Strata with $B = 0$ are undefined. The main-text table uses pooled micro estimates. The supplementary macro estimate first pools counts from both splits within task/K, then averages equally over K within tasks and over tasks. Per-cell strata and complete $3 \times 3$ transition matrices are saved separately. Pooled micro, this macro estimate, and equal-cell estimates are distinct. Matrix categories are the original answer, the new table target, and other answers, separating answers that remain correct, follow the table target, or select another incorrect answer option.

Itemwise paired answer transitions for all models. $T / N$ requires the two predictions jointly to satisfy the transition event, while $T / B$ conditions on conflicting questions answered correctly with the correct table. Integer denominators are reported alongside pooled micro-average transition rates.

## K Execution and Reproducibility

The runtime uses PyTorch 2.8.0+cu128, Transformers 5.18.0, and PEFT 0.21.2 on RTX 4090 GPUs. Local inference follows the precision and padding settings in Appendices B.4 and L.2. Protocols, generators, executable oracles, model identities, selected checkpoints, and itemwise predictions have SHA256 identities. BF16 training replay is not bitwise identical. Reloaded readouts retain semantic predictions, with a maximum checked probability difference of $8 . 3 5 \times 1 0 ^ { - 7 }$ . Jev responses identify jev-1.13.0; successful replacements count once per logical request. The reproduction package excludes model weights and private credentials.

## L Additional Behavioral Diagnostics

## L.1 Data Roles and Fixed Evaluation

These posthoc diagnostics reuse the inspected reserved benchmark without fitting or selection: 4,160 natural questions and 1,016 diagnostic items (254 quartets) in 13 blocks. Two AST-equivalent rewrites yield 72,464 predictions for Base and six fixed LoRA instances. The latter represent two source-rule splits crossed with three state seeds.

All wordings use batch 8 with fixed weights, order, scoring code, BF16 body, and FP32 candidate projection. The 36,232 original-text predictions align with 72,464 hash-verified rewrite predictions. Changing original-text batch size from 16 to 8 changes 101/5,176 Base choices (1.95%) and 27– 42/5,176 choices per LoRA (0.52–0.81%). Mean absolute logit differences are 0.0322–0.0408, with maximum 0.7095. Flipped items have smaller original top-two margins. Using a baseline at the same batch size shifts both pooled gap changes by −0.28 pp. The verbal-rewrite contrast moves from +0.19 to −0.09 pp, while preserving all eight task-level directions of the measured wording contrasts.

Across models and rewrites, 96.06–98.92% of wording flips lie outside the observed originaltext batch-flip set. We decompose each LoRA– Base gap change by whether either model flips its original-text choice across batches, retaining full-set block/task/K weights and denominators. Batch-stable contributions preserve all eight taskwise directions. Stable/sensitive contributions are $- 2 . 2 7 / - 0 . 3 7$ pp for the functional rewrite and $+ 0 . 2 2 / - 0 . 3 1$ pp for the paired verbal rewrite.

Intervals use 2,000 resamples (seed 2026100801) within block/task/K, preserving paired-rule shared-state clusters for natural questions and complete diagnostic quartets. Sampled indices are shared across conditions and models. Aggregation equally weights K within task, tasks within blocks, blocks, and LoRA instances. These intervals quantify paired input variation while conditioning on the evaluated rules and the fixed, source-selected model checkpoints.

## L.2 Precision and Padding Controls

We select a fixed subset from the existing reservedrule evaluation using SHA256 ordering with seed 20261009, without consulting predictions. Each task/K stratum contributes eight complete natural state pairs and four diagnostic quartets, totaling 160 natural and 160 diagnostic questions. The original and both rewrites retain identical states, rules, options, and labels. Base and the fixed state-101 LoRA checkpoint from each source split are evaluated under BF16 or FP32 model inference, crossed with dynamic or fixed left padding. Batch size remains eight and item order is unchanged. Fixed padding uses the maximum length across all selected texts, 361 tokens, without truncation. FP32 weights are loaded directly from the original checkpoints, all floating-point model parameters are checked as FP32, and TF32 is disabled. Candidate projection remains FP32 in every condition. The fixed paired design yields 11,520 predictions. Wording sensitivity persists when precision and padding are controlled. Table 17 separates sametext padding effects from cross-wording disagreement. FP32 substantially reduces the former, while both equivalent rewrites continue to change answers at fixed padding. One LoRA prediction still differs across FP32 padding conditions. The result supports wording sensitivity beyond the observed BF16 padding effect within this paired subset.

<table><tr><td>Model</td><td>Intervention</td><td>N</td><td>B</td><td>T</td><td>T/N(%) T/B(%)</td><td></td></tr><tr><td>Base</td><td>Formula-row mismatch</td><td>220</td><td>69</td><td>8</td><td>3.64</td><td>11.59</td></tr><tr><td>Base</td><td>Record-column mismatch</td><td>190</td><td>66</td><td>5</td><td>2.63</td><td>7.58</td></tr><tr><td>Base</td><td>Target-value counterfactual</td><td>320</td><td>116</td><td>12</td><td>3.75</td><td>10.34</td></tr><tr><td>LoRA</td><td>Formula-row mismatch</td><td>220</td><td>90</td><td>17</td><td>7.73</td><td>18.89</td></tr><tr><td>LoRA</td><td>Record-column mismatch</td><td>190</td><td>84</td><td>21</td><td>11.05</td><td>25.00</td></tr><tr><td>LoRA</td><td>Target-value counterfactual</td><td>320</td><td>147</td><td>19</td><td>5.94</td><td>12.93</td></tr><tr><td>Jev</td><td>Formula-row mismatch</td><td>220</td><td>191</td><td>169</td><td>76.82</td><td>88.48</td></tr><tr><td>Jev</td><td>Record-column mismatch</td><td>190</td><td>177</td><td>173</td><td>91.05</td><td>97.74</td></tr><tr><td>Jev</td><td>Target-value counterfactual</td><td>320</td><td>282</td><td>243</td><td>75.94</td><td>86.17</td></tr></table>

Table 14: Directed answer transitions distinguish following a supplied target from an undirected answer change. Counts preserve the applicable conflict set: T/N uses all conflicts and T/B conditions on original correctness.
<table><tr><td>Metric</td><td></td><td></td><td></td><td></td><td></td><td>Models Original Rewrite 1 Rewrite 2 Change 1 [95% CI] Change 2 [95% CI]</td></tr><tr><td></td><td>Natural All six</td><td>19.30</td><td>16.66</td><td>19.21</td><td>-2.64 [-4.80, -0.62]</td><td>-0.09 [-1.76, 1.62]</td></tr><tr><td>Natural Split 1</td><td></td><td>16.10</td><td>15.15</td><td>16.87</td><td>-0.95 [-3.10, 1.10]</td><td>0.77 [-0.87, 2.51]</td></tr><tr><td>Natural Split 2</td><td></td><td>22.50</td><td>18.18</td><td>21.55</td><td>-4.33 [-6.64, -2.20]</td><td>-0.95 [-2.64, 0.80]</td></tr><tr><td>QAcc</td><td>All six</td><td>9.99</td><td>5.84</td><td>9.44</td><td>-4.15 [-5.65, -2.80]</td><td>-0.55 [-1.44, 0.30]</td></tr><tr><td>QAcc</td><td>Split 1</td><td>6.37</td><td>4.88</td><td>6.16</td><td>-1.50 [-3.13, 0.11]</td><td>-0.21 [-1.28, 0.89]</td></tr><tr><td>QAcc</td><td>Split 2</td><td>13.60</td><td>6.80</td><td>12.71</td><td>-6.80 [-9.44, -4.31]</td><td>-0.89 [-2.17, 0.39]</td></tr></table>

Table 15: Functional rewrites reduce the pooled adaptation advantage; verbal rewrites produce little pooled naturalaccuracy change. All entries compare LoRA with Base at batch 8. Changes are paired differences (pp); 95% intervals condition on the fixed rule allocations and model checkpoints under shared, paired input resampling.

Wording rates use the natural questions and average equally over option counts within tasks and then over tasks. Their intervals use 2,000 completestate-pair bootstrap draws within task/K, shared across models and conditions. The diagnostic analysis preserves complete quartets. Unlike the full reserved-rule evaluation, this subset analysis weights the sampled state groups directly within task/K, rather than retaining the original rule-block weights. Its intervals condition on the sampled groups and fixed checkpoints. Padding-change counts pool all three wordings and both data types, and therefore use a broader item denominator than the paired natural-question wording comparison.

## L.3 Nearest-Source Similarity and Conflict Subsets

For each target rule and task/K, the nearest source rule maximizes oracle-answer agreement over uniform finite-domain inputs. Ties within 10<sup>−14</sup> choose the lexicographically smallest source ID. The permutation sensitivity additionally searches the six field permutations, breaking remaining ties lexicographically. This oracle-assisted reference is evaluated on accepted natural states with the target oracle retained as gold. The conflict subset contains questions where their answers differ.

For the two multi-record tasks, records are drawn independently with replacement from the finite domain. Let $( A , B )$ denote the paired rule values of a candidate, with signs reversed for minimum routing so larger is always preferred. Define $p _ { < } ( a , b ) \ : = \ : P ( A \ : < \ : a , B \ : < \ : b )$ and $p _ { \leq } ( a , b ) =$ $P ( A \leq a , B \leq b )$ . Since smaller record IDs win ties, exact winner agreement for K records is

![](images/8afeeba55939a49e354737b71036bc59485202569f31de01532e08e5efcd4b51.jpg)

![](images/9f6b6f1bcacb7005fd39dd5d4e6384392945402d1bd56e2de8faf9ffb9f5f2a8.jpg)

Figure 13: LoRA remains above Base across equivalent wordings, with a larger reduction under functional notation. Points connect evaluations of fixed systems at batch 8; labels give accuracy (%). Natural accuracy and QAcc use separately labeled zero-based scales. LoRA averages six instances, and 9B is a distinct instruction-tuned system.
<table><tr><td>Task</td><td></td><td></td><td></td><td>Model Original Rewrite 1 Rewrite 2</td><td>Change 1 [95% CI]</td><td>Change 2 [95% CI]</td></tr><tr><td>Binary</td><td>Base</td><td>69.95</td><td>50.96</td><td>58.17</td><td>-18.99 [-24.77, -12.74]-11.78 [-17.07, -6.25]</td><td></td></tr><tr><td>Binary</td><td>LoRA</td><td>77.72</td><td>73.60</td><td>78.04</td><td>-4.13 [-5.77, -2.48]</td><td>0.32 [-0.48, 1.16]</td></tr><tr><td>Routing</td><td>Base</td><td>27.64</td><td>27.48</td><td>29.25</td><td>-0.16 [-3.77, 3.37]</td><td>1.60 [-0.88, 4.09]</td></tr><tr><td>Routing</td><td>LoRA</td><td>44.87</td><td>38.29</td><td>41.57</td><td>-6.58 [-7.71, -5.46]</td><td>-3.30 [-4.22, -2.34]</td></tr><tr><td>Rating</td><td>Base</td><td>22.84</td><td>23.48</td><td>24.20</td><td>0.64 [-0.88, 2.09]</td><td>1.36 [-0.24, 2.96]</td></tr><tr><td>Rating</td><td>LoRA</td><td>54.91</td><td>48.33</td><td>53.15</td><td>-6.58 [-7.76, -5.46]</td><td>-1.76 [-2.60, -0.95]</td></tr><tr><td>Selection Base</td><td></td><td>30.61</td><td>33.65</td><td>32.93</td><td>3.04 [-0.32, 6.49]</td><td>2.32 [0.08, 4.57]</td></tr><tr><td>Selection LoRA</td><td></td><td>50.73</td><td>42.01</td><td>48.62</td><td>-8.72 [-9.96, -7.44]</td><td>-2.11 [-2.96, -1.27]</td></tr></table>

Table 16: Binary-task increases offset adaptation-gain reductions in rating, routing, and selection. Natural accuracy is reported in percentages and paired changes in pp, at batch 8 with complete-state resampling intervals.

$$
\begin{array} { r l r } {  { P ( \mathrm { s a m e ~ w i n n e r } ) = \mathbb { E } _ { ( A , B ) } [ S _ { K } ( A , B ) ] , } } \\ & { } & { S _ { K } ( a , b ) = \displaystyle \sum _ { i = 0 } ^ { K - 1 } p _ { < } ( a , b ) ^ { i } p _ { \leq } ( a , b ) ^ { K - 1 - i } . } \end{array}\tag{18}
$$

The empirical joint distribution enumerates all finite-domain records, so the formula integrates all ordered K-record draws without enumerating their Cartesian power. Records before a candidate must be strictly worse under both rules. Later records may tie the candidate under either rule because record-ID tie-breaking favors the earlier candidate.

The subset-normalized conflict score averages within each observed task/K subset, then equally over K and tasks. It does not equalize rules after filtering. For the additive decomposition, let $g =$ $( \mathrm { r u l e } , t , K ) , w _ { g }$ be its full-set macro weight, $p _ { C , g }$ its conflict fraction, and $\Delta _ { C , g } , \Delta _ { \bar { C } , g }$ its conditional LoRA-minus-Base differences. Then

$$
\begin{array} { r c l } { \displaystyle \Delta _ { \mathrm { f u l l } } = \Delta _ { C } + \Delta _ { \bar { C } } , } \\ { \displaystyle \Delta _ { { C } } = \sum _ { g } w _ { g } p _ { C , g } \Delta _ { C , g } , } \\ { \displaystyle \Delta _ { \bar { C } } = \sum _ { g } w _ { g } ( 1 - p _ { C , g } ) \Delta _ { \bar { C } , g } . } \end{array}\tag{19}
$$

Weights are $w _ { g } = ( 4 | \mathcal { K } _ { t } | 2 6 ) ^ { - 1 }$ for this balanced natural set. Empty subsets contribute zero mass. Their conditional scores are undefined rather than imputed. The two contributions exactly recover the original batch-16 macro advantage. A full-weight conditional conflict score divides its contribution by $\Sigma _ { g } w _ { g } p _ { C , g }$ and differs from the subset-normalized macro computed after conflict filtering (Table 18).

<table><tr><td></td><td colspan="2"></td><td colspan="2">Same-text padding changes Wording disagreement at fixed padding, FP32</td></tr><tr><td>Model</td><td>BF16</td><td>FP32</td><td>Functional (%, 95% CI)</td><td>Verbal (%, 95% CI)</td></tr><tr><td>Base</td><td>26/960</td><td>0/960</td><td>44.27 [34.38, 54.69]</td><td>31.25 [23.44, 39.06]</td></tr><tr><td>LoRA, split 1</td><td>11/960</td><td>0/960</td><td>23.44 [16.67, 30.21]</td><td>12.50 [8.33, 16.67]</td></tr><tr><td>LoRA, split 2</td><td>8/960</td><td>1/960</td><td>25.00 [19.27, 31.25]</td><td>17.71 [12.50, 22.92]</td></tr></table>

Table 17: Wording sensitivity persists with FP32 inference and fixed padding on the controlled subset. Wording-flip rates compare each rewrite with the original on natural questions; same-text padding comparisons retain the wording and measure changes in the selected answer under a different padding condition on identical question inputs.

Per-rule agreement records are retained with the offline analysis. Uniform-domain agreement and accepted-sample agreement remain separate: exclusion of historical states and finite sampling can change the latter. Selection and modelperformance correlations are exploratory. None is used to redefine the source or target partitions.

## L.4 Agreement-Stratified Transfer and Complete-Quartet Conflict Loads

Lower oracle agreement separates weaker transfer from the pooled adaptation gain. This posthoc analysis reuses all saved batch-16 reserved predictions: 4,160 natural questions and 254 complete quartets. It adds no training, inference, or service requests. For each target pair, we average its two native nearest-source agreements over the uniform finite domain. Within each source split and task/K, the lower, middle, and upper thirds define low, middle, and high agreement. Ties remain together. These are relative groups; their absolute agreement ranges can overlap across tasks. Each task/K allocates 5, 4, and 4 of its 13 target pairs to these groups. All ten task/K strata remain represented in each pooled oracle-agreement group.

The same pair assignments apply to natural questions and intact diagnostic quartets. Original weights average blocks equally, tasks equally within blocks, and option counts equally within tasks. We condition these weights on each group rather than recomputing a row-wise mean. Within a source split, scores average its three fixed LoRA fits; pooled scores weight the two splits equally. Both splits reuse the same questions and Base predictions. Pooled counts therefore denote sourcereference memberships, not independent samples. Six omitted ineligible quartets remain omitted.

Natural gains increase across agreement groups in both splits (Table 20). Pooled QAcc increases, although split-1 QAcc is not monotonic. Absolute agreement cutoffs have uneven task coverage, which motivates ranking target rules within each task/K stratum before assigning agreement groups. Absolute conflict scores distinguish transfer gains from their full-set contributions. Conflict gains differ in sign across source splits, whereas nonconflicts improve in both (Table 21). Table 18 retains their weighted full-set gain contributions.

Fully conflicting quartets have low pooled joint success. Conflict load counts target answers differing from native nearest-source references within an intact quartet. Fully conflicting quartets show opposite gain signs across splits and little pooled improvement (Table 22). Intermediate loads do not follow a monotonic gain trend in this benchmark.

Intervals use 2,000 paired resamples with seed 2026101201. Sampling units are shared natural states or complete diagnostic quartets within block and task/K. Joint cohort memberships under both source splits remain fixed, preserving shared observations across models and source references. Intervals condition on tested rules, fixed models, and observed cohorts. Exact-load cohorts include singleton resampling units, restricting uncertainty estimates. We report their finite-cohort scores and coverage as descriptive diagnostics of transfer under concentrated source–target answer conflict.

## L.5 Adjacent Randomized Jev Repetitions

The schedule places six requests for each item adjacently: the original and two rewrites each occur twice in randomized order. Item blocks are also randomized. The 320 questions form 80 complete quartets balanced across the two existing splits. All 1,920 logical measurements completed with jev-1.13.0. Successful replacements count once per logical request. The repeated-call comparison uses the fixed development questions and the same original task labels and question identities across all six paired adjacent request conditions per item.

<table><tr><td colspan="4">Mapping Split Conflict n Mass (%)</td><td rowspan="2">weighted</td><td rowspan="2">subset</td><td rowspan="2">Conditional Conditional Contribution Contribution conflict</td><td rowspan="2">other</td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td>native</td><td>1</td><td>1907</td><td>43.73</td><td>-2.43</td><td>-3.11</td><td>-1.06</td><td>16.93</td></tr><tr><td>native</td><td>2</td><td>1722</td><td>38.90</td><td>12.62</td><td>11.37</td><td>4.91</td><td>17.26</td></tr><tr><td>permuted 1</td><td></td><td>1595</td><td>35.96</td><td>1.91</td><td>1.70</td><td>0.69</td><td>15.18</td></tr><tr><td>permuted 2</td><td></td><td>1454</td><td>32.97</td><td>14.74</td><td>13.70</td><td>4.86</td><td>17.31</td></tr></table>

Table 18: Nearest-source nonconflicts account for most of the full-set adaptation gain. Conditional gains and weighted contributions are in pp. Weighted estimates retain full-set conflict mass; subset estimates recompute the task/K means on retained questions before averaging scores equally across strata within each task family.
<table><tr><td colspan="3">Mapping Split Shared conflict n LoRA-Base (pp)</td></tr><tr><td>native</td><td>1 918</td><td>-0.77</td></tr><tr><td>native</td><td>2 918</td><td>10.08</td></tr><tr><td>permuted 1</td><td>727</td><td>7.11</td></tr><tr><td>permuted 2</td><td>727</td><td>10.12</td></tr></table>

Table 19: The common conflict intersection permits both source splits to be compared on identical questions. All ten task/K strata remain represented; percentages use the subset-normalized task and option-count macro.

For item i, within-text disagreement $W _ { i }$ averages the three identical-text pairs. Across-wording disagreement $B _ { i }$ averages the twelve pairs combining distinct wordings. We first compute $B _ { i } - W _ { i }$ within item, average items within each split, and weight splits equally. The 960 within-text and 3,840 across-wording pairs share questions and are not independent samples. The 95% interval uses 2,000 whole-quartet bootstrap draws stratified by split, with seed 2026100807 and all six responses retained together. Figure 3 reports within-text disagreement, across-wording disagreement, and their paired difference under shared quartet resampling. The excess disagreement persists when request spacing and interruptions are examined. All five slot-lag strata have positive across-minuswithin differences. Table 23 retains actual-time denominators, including sparse strata. A posthoc sensitivity analysis excludes an entire quartet if any item had a retry or any adjacent successful-request start gap within an item was at least 16 seconds. It removes five quartets and retains 75, with 39/36 in splits 1/2. Split weights remain equal. The retainedsample contrast and interval are shown beside the full result in Figure 3. Whole quartets are excluded.

## L.6 Larger System and Boolean Policies

A larger instruction-tuned system does not uniformly dominate the adapted model. Qwen3.5- 9B remains below arithmetic-adapted LoRA when directly scoring candidate answers under plain-text and native-chat prompts. Native-template scoring disables thinking (Appendix L.11). The larger system performs better on Boolean policies, with weak exception-handling QAcc (Appendix Table 24). Finite-state transitions also yield limited natural accuracy and zero quartet success. These comparisons distinguish answer-interface sensitivity from transfer across different executable rule families.

The Qwen3.5-9B checkpoint revision is c202236235762e1c871ad0 ccb60c8ee5ba337b9a. We score all 5,176 arithmetic questions in all three wordings, yielding 15,528 predictions, without training, chain-ofthought prompting, or checkpoint selection. The instruction-tuned system differs from the controlled backbone in more than parameter count. The interface uses the same prose template as the smaller backbone: State, Question, Options, and an Answer prefix, with no chat template or added special tokens. It left-pads inputs, applies the BF16 body, projects the final hidden state onto FP32 original candidate-letter rows, and selects the highest-scoring valid option. This configuration is direct scoring, rather than generated chain-of-thought or a chat interaction.

The Boolean extension uses six flags and twenty fixed policies: ten first-match priority policies and ten policies with one exception and a waiver. Each policy partitions the six flags into three two-literal conjunctions with randomized signs. Duplicate truth tables are rejected within family. Sixteen states per policy are sampled uniformly without replacement before labels are used. Priority semantics are first matching action among Allow, Deny, Allow, followed by default Deny. Exception semantics are $a \wedge ( \neg b \vee c )$ : the waiver restores the base rule and cannot grant permission alone. The fixed depth provides no depth-trend experiment. Exhaustive checking of all 64 flag states per policy gives 1,280 cross-oracle comparisons. Each family has 160 natural questions and 40 quartets. The natural and diagnostic sets do not overlap, but the diagnostic quartets share policy–state inputs (262 distinct diagnostic inputs among 320 rows). We therefore report finite-set descriptive results without row-IID significance tests. The arithmetic LoRA is the fixed split-1/state-101 checkpoint, with no policy training. Base, that LoRA, and 9B each produce 640 predictions, totaling 1,920. Always-deny accuracy makes the natural label imbalance explicit. A fixed answer cannot solve the checkerboard quartets.

<table><tr><td colspan="8">Natural questions</td></tr><tr><td>Split</td><td>Agreement</td><td>Questions</td><td>Base</td><td>LoRA</td><td>Gain</td><td>95% gain CI</td><td>Conflict (%)</td></tr><tr><td>1</td><td>Low</td><td>1600</td><td>37.03</td><td>44.29</td><td>+7.26</td><td>[4.29, 10.31]</td><td>52.50</td></tr><tr><td>1</td><td>Middle</td><td>1280</td><td>39.26</td><td>54.82</td><td>+15.56</td><td>[12.48, 18.55]</td><td>41.67</td></tr><tr><td>1</td><td>High</td><td>1280</td><td>38.15</td><td>65.10</td><td>+26.95</td><td>[23.96, 29.86]</td><td>34.83</td></tr><tr><td>2</td><td>Low</td><td>1600</td><td>37.08</td><td>52.78</td><td>+15.69</td><td>[12.93, 18.33]</td><td>46.82</td></tr><tr><td>2</td><td>Middle</td><td>1280</td><td>37.37</td><td>60.61</td><td>+23.24</td><td>[20.42, 26.24]</td><td>39.84</td></tr><tr><td>2</td><td>High</td><td>1280</td><td>39.97</td><td>69.16</td><td>+29.19</td><td>[25.87, 32.47]</td><td>28.06</td></tr><tr><td>Pooled</td><td>Low</td><td>3200</td><td>37.06</td><td>48.53</td><td>+11.48</td><td>[9.22, 13.59]</td><td>49.66</td></tr><tr><td>Pooled</td><td>Middle</td><td>2560</td><td>38.31</td><td>57.71</td><td>+19.40</td><td>[16.97, 21.89]</td><td>40.76</td></tr><tr><td>Pooled</td><td>High</td><td>2560</td><td>39.06</td><td>67.13</td><td>+28.07</td><td>[25.68, 30.69]</td><td>31.45</td></tr></table>

<table><tr><td colspan="8">Complete diagnostic quartets</td></tr><tr><td>Split</td><td>Agreement</td><td>Quartets</td><td>Base</td><td>LoRA</td><td>Gain</td><td>QAcc</td><td>95% QAcc CI</td></tr><tr><td>1</td><td>Low</td><td>100</td><td>36.66</td><td>39.26</td><td>+2.60</td><td>5.58</td><td>[2.75, 8.33]</td></tr><tr><td>1</td><td>Middle</td><td>78</td><td>35.48</td><td>43.49</td><td>+8.01</td><td>3.17</td><td>[1.05, 5.28]</td></tr><tr><td>1</td><td>High</td><td>76</td><td>31.53</td><td>52.27</td><td>+20.74</td><td>10.98</td><td>[7.20, 14.77]</td></tr><tr><td>2</td><td>Low</td><td>98</td><td>32.68</td><td>44.06</td><td>+11.38</td><td>6.89</td><td>[4.84, 8.85]</td></tr><tr><td>2</td><td>Middle</td><td>78</td><td>36.75</td><td>55.08</td><td>+18.33</td><td>15.67</td><td>[12.00, 19.33]</td></tr><tr><td>2</td><td>High</td><td>78</td><td>35.61</td><td>59.53</td><td>+23.92</td><td>17.15</td><td>[12.23, 21.82]</td></tr><tr><td>Pooled</td><td>Low</td><td>198</td><td>34.67</td><td>41.66</td><td>+6.99</td><td>6.24</td><td>[4.52, 7.93]</td></tr><tr><td>Pooled</td><td>Middle</td><td>156</td><td>36.11</td><td>49.28</td><td>+13.17</td><td>9.42</td><td>[7.22, 11.63]</td></tr><tr><td>Pooled</td><td>High</td><td>154</td><td>33.57</td><td>55.90</td><td>+22.33</td><td>14.07</td><td>[11.33, 16.74]</td></tr></table>

Table 20: Low-agreement pairs retain positive pooled transfer gains but lower joint correctness than high-agreement pairs. Accuracy and QAcc are percentages; gains and their intervals are percentage points. Diagnostic Base and LoRA columns give item accuracy, and QAcc gives LoRA all-four correctness. Base QAcc is zero in every group. Pooled counts retain both source-reference memberships for questions shared across the two source pools.
<table><tr><td>Split</td><td>Source-target answers</td><td>Questions</td><td>Weight (%)</td><td>Base</td><td>LoRA</td><td>Readout</td><td>Gain (pp)</td></tr><tr><td>1</td><td>Conflict</td><td>1907</td><td>43.73</td><td>34.49</td><td>32.07</td><td>34.02</td><td>-2.43</td></tr><tr><td>1</td><td>Agree</td><td>2253</td><td>56.27</td><td>40.83</td><td>70.93</td><td>47.73</td><td>+30.09</td></tr><tr><td>2</td><td>Conflict</td><td>1722</td><td>38.90</td><td>31.10</td><td>43.72</td><td>34.50</td><td>+12.62</td></tr><tr><td>2</td><td>Agree</td><td>2438</td><td>61.10</td><td>42.49</td><td>70.74</td><td>48.83</td><td>+28.25</td></tr></table>

Table 21: Conflict-conditioned absolute accuracy exposes different transfer outcomes across the two source splits. Scores are percentages; gains compare LoRA with Base under the original weights conditioned on each subset. The readout column reports the MLP selected separately using the validation inputs of each source-rule split.

## Takeaway

Check each target rule family and answer interface when evaluating an adapted system. Report joint success and item accuracy together to distinguish reliable decisions from partial success.

<table><tr><td>Split</td><td>Load</td><td>Quartets</td><td>Task/K</td><td>Weight (%)</td><td>Base</td><td>LoRA</td><td>Gain (pp)</td><td>QAcc</td></tr><tr><td>1</td><td>0</td><td>37</td><td>7</td><td>13.14</td><td>32.93</td><td>47.29</td><td>+14.36</td><td>4.88</td></tr><tr><td>1</td><td>1</td><td>19</td><td>4</td><td>11.22</td><td>46.43</td><td>58.33</td><td>+11.90</td><td>14.29</td></tr><tr><td>1</td><td>2</td><td>129</td><td>10</td><td>50.32</td><td>39.17</td><td>48.89</td><td>+9.71</td><td>7.15</td></tr><tr><td>1</td><td>3</td><td>30</td><td>4</td><td>11.75</td><td>24.09</td><td>41.74</td><td>+17.65</td><td>4.55</td></tr><tr><td>1</td><td>4</td><td>39</td><td>8</td><td>13.57</td><td>20.47</td><td>14.37</td><td>-6.10</td><td>0.00</td></tr><tr><td>2</td><td>0</td><td>46</td><td>7</td><td>17.74</td><td>36.90</td><td>59.69</td><td>+22.79</td><td>23.09</td></tr><tr><td>2</td><td>1</td><td>19</td><td>6</td><td>9.29</td><td>46.55</td><td>58.05</td><td>+11.49</td><td>10.34</td></tr><tr><td>2</td><td>2</td><td>130</td><td>10</td><td>49.79</td><td>34.07</td><td>52.63</td><td>+18.56</td><td>11.59</td></tr><tr><td>2</td><td>3</td><td>27</td><td>5</td><td>11.86</td><td>37.84</td><td>54.50</td><td>+16.67</td><td>13.51</td></tr><tr><td>2</td><td>4</td><td>32</td><td>7</td><td>11.32</td><td>22.41</td><td>31.29</td><td>+8.88</td><td>2.83</td></tr><tr><td>Pooled</td><td>0</td><td>83</td><td>7/7</td><td>15.44</td><td>34.91</td><td>53.49</td><td>+18.58</td><td>13.99</td></tr><tr><td>Pooled</td><td>1</td><td>38</td><td>4/6</td><td>10.26</td><td>46.49</td><td>58.19</td><td>+11.70</td><td>12.32</td></tr><tr><td>Pooled</td><td>2</td><td>259</td><td>10/10</td><td>50.05</td><td>36.62</td><td>50.76</td><td>+14.14</td><td>9.37</td></tr><tr><td>Pooled</td><td>3</td><td>57</td><td>4/5</td><td>11.81</td><td>30.96</td><td>48.12</td><td>+17.16</td><td>9.03</td></tr><tr><td>Pooled</td><td>4</td><td>71</td><td>8/7</td><td>12.45</td><td>21.44</td><td>22.83</td><td>+1.39</td><td>1.42</td></tr></table>

Table 22: All-four conflict yields little pooled accuracy gain and low joint correctness. Load counts conflicting members per complete quartet. Base and LoRA columns give diagnostic item accuracy; QAcc gives LoRA all-four correctness. Base QAcc is zero throughout. Weight is original diagnostic mass, and Task/K counts represented strata, listed by source split in pooled rows. Pooled counts reuse the same quartets under two source references.
<table><tr><td>Actual start gap</td><td>Within pairs</td><td>Disagree (%)</td><td>Across pairs</td><td>Disagree (%)</td></tr><tr><td>&lt; 2 s</td><td>523</td><td>9.37</td><td>1991</td><td>35.26</td></tr><tr><td>2-4 s</td><td>324</td><td>9.88</td><td>1363</td><td>35.95</td></tr><tr><td>4-8 s</td><td>105</td><td>6.67</td><td>452</td><td>34.51</td></tr><tr><td>8-16 s</td><td>0</td><td>N/A</td><td>2</td><td>100.00</td></tr><tr><td>≥ 16 s</td><td>8</td><td>12.50</td><td>32</td><td>37.50</td></tr></table>

Table 23: Cross-wording disagreement exceeds same-text disagreement in every elapsed-time stratum with both comparisons. The 8–16 s stratum has no same-text pairs; the counts identify the available paired coverage.

## L.7 Actual Wording and Policy Inputs

The Boolean templates ask either for the first matching action in an ordered Allow/Deny list, or for a base permission with an exception and waiver. In the order credential, sponsor, training, hold, pass, audit, the first priority example has values $( 0 , 0 , 0 , 1 , 0 , 1 )$ and oracle Allow. The first exception example has $( 1 , 0 , 0 , 0 , 0 , 0 )$ and oracle Deny. The full semantics are specified above. Later examples give the serialized Jev request and the exact table changes applied to that same native request.

The arithmetic example is the first frozen item, selected independently of predictions. All three wordings share State: {"facts": $\{ { } ^ { n } \mathbf { i } \mathbf { d } ^ { n } : { } ^ { n } \mathsf { R } \pmb { \theta } ^ { n } , { } ^ { n } \mathbf { \times } ^ { n } : 1 2 , { } ^ { n } \mathbf { y } ^ { n } : 8 , { } ^ { n } \mathbf { z } ^ { n } : - 1 2 \} \}$ , options A) no and B) yes, and the final Answer: prefix. The oracle is B. Only the question field changes below. Displayed line wrapping is typographic.

## Original arithmetic question

Using facts, is (min(x, y) - (y + z)) strictly greater than 0? Select no or yes. Only x, y, z participate. Ignore other fields and notes.

## Equivalent functional question

Evaluate difference(minimum(x, y), sum(y, z)) on facts. Return yes exactly when its value exceeds 0; otherwise return no. Only x, y, z participate. Ignore other fields and notes. Here sum(a,b) means a+b, difference(a,b) means a-b, product(a,b) means a\*b, minimum(a,b) means min(a,b), and maximum(a,b) means max(a,b).

## Equivalent verbal question

For the facts, decide whether (min(x, y) minus (y plus z)) is strictly positive. Choose yes for a result greater than 0, and no otherwise. Only x, y, z participate. Ignore other fields and notes.

<table><tr><td></td><td colspan="2">Priority policy</td><td colspan="2">Exception policy</td><td colspan="2">Finite-state transitions</td></tr><tr><td>System</td><td>Natural</td><td>QAcc</td><td>Natural</td><td>QAcc</td><td>Natural</td><td>QAcc</td></tr><tr><td>Base</td><td>60.00</td><td>0.00</td><td>31.87</td><td>0.00</td><td>25.31</td><td>0.00</td></tr><tr><td>Arithmetic LoRA</td><td>51.88</td><td>0.00</td><td>43.12</td><td>2.50</td><td>21.88</td><td>0.00</td></tr><tr><td>Qwen3.5-9B</td><td>78.12</td><td>22.50</td><td>76.25</td><td>2.50</td><td>32.81</td><td>0.00</td></tr><tr><td>Always deny</td><td>58.13</td><td>0.00</td><td>78.13</td><td>0.00</td><td colspan="2">Not applicable</td></tr></table>

Table 24: The tested systems show different competence across Boolean policies and finite-state transitions. All systems have zero transition-task QAcc. Scores are descriptive finite-set percentages; the arithmetic LoRA is reused without new-task training and the always-deny baseline applies to both evaluated Boolean policy rule families.

## L.8 Support of the Reassociation Reference

The task-level reference uses separate cell/task/K/slot/gold strata. A singleton stratum cannot move. A stratum with constant predictions also cannot change that model’s predictions, even when it contains several entries. The two measures overlap and must not be added. Table 25 reports counts and slot weights, with weights averaged equally over the six fixed cells. A slot weight is one quarter of its quartet’s task-level macro weight. The reference uses 1,000 draws and seed 2026100802 for each cell, task, and model. Table 26 exposes how the selection departure varies across cells and option counts. For reassociation draw $b ,$ define $d _ { b } = 1 0 0 ( Q _ { \mathrm { o b s } } - Q _ { \mathrm { r e f } , b } )$ . The table gives $\bar { d }$ and the 2.5th and 97.5th percentiles of $d _ { b }$ in percentage points. The bracket is a descriptive difference range, not a range of raw QAcc values or a confidence interval. These finite-cohort differences are unadjusted for multiplicity and conditional on the evaluated rule and state pairs.

## L.9 Answer-Transition Weighting Sensitivity

Let N count applicable conflicts between the original oracle and altered-table target. Let B count conflicts answered correctly with the correct table, and T count paired transitions to the altered-table target. We report both T /N and T /B. Micro weighting pools counts. Task/K weighting pools cells within a task/K, averages equally over available K within each task, then over tasks. Equal-cell weighting first computes the same task/K macro within each cell, then averages cells. Zero-denominator strata are excluded from the relevant rate and their coverage is recorded. All applicable Jev strata have defined conditional rates, with 20 cell/task/K strata for row and target interventions and 12 for column interventions. Single-record tasks have no nonidentity column permutation and are inapplicable.

## L.10 Native Jev Requests and Table Interventions

Figures 15 and 16 show the original instruction and complete-value native request. The original shares the model, criteria, type, and records shown in the complete-value request, but omits supplementary scores. Both full payloads are retained in examples/native\_requests.json. Formatting preserves every string value. They were sent to https://api.typesafe.ai/v1/ systemone. The state field is itself a serialized JSON string. The decision is requested through questions.decision, not through the local models’ letter-answer prompt. Successful responses identify jev-1.13.0. The example is the first tworecord routing item in split 1/state 101, chosen independently of predictions. The example specifies the evaluated interface and exact interventions; aggregate transition rates are reported over the full applicable conflict set for each table intervention.

D2: correct all-rule table. D2 adds all formula values and replaces the D0 instruction to ignore supplementary fields with an instruction explaining how to use them. D2 and D3 share this instruction. All subsequent mapping and target interventions preserve the D2 question exactly. Punctuation inside these archived strings is retained verbatim.

Worked interventions on the same request. The requested formula is (xy) + max(y, z). Its correct values are (8, −1) for (R0, R1), so the original oracle selects R1. The row intervention cyclically moves every score row forward by two positions while retaining formula labels. The requested row becomes (50, −18) and still selects R1, so this item is not a row-conflict case. The column intervention swaps R0 and R1 values in every row, giving (−1, 8) in the requested row and a new target R0. The target intervention changes only the requested row’s R0 value from 8 to −1. The values then tie at (−1, −1) and the fixed record-ID tie-break selects R0. In both conflict cases the original task’s gold answer remains R1. This example separates changes in the table from changes in its implied answer and identifies the conflicts that enter every denominator in this paired-transition analysis.

<table><tr><td>Task</td><td>Model</td><td>Strata</td><td>Size range</td><td>Singletons</td><td>Weight (%)</td><td>Constant</td><td>Weight (%)</td></tr><tr><td>Binary</td><td>Base</td><td>24</td><td>8-8</td><td>0</td><td>0.00</td><td>0</td><td>0.00</td></tr><tr><td>Binary</td><td>Jev</td><td>24</td><td>8-8</td><td>0</td><td>0.00</td><td>12</td><td>50.00</td></tr><tr><td>Binary</td><td>LoRA</td><td>24</td><td>8-8</td><td>0</td><td>0.00</td><td>5</td><td>20.83</td></tr><tr><td>Routing</td><td>Base</td><td>262</td><td>1-6</td><td>102</td><td>17.71</td><td>157</td><td>47.05</td></tr><tr><td>Routing</td><td>Jev</td><td>262</td><td>1-6</td><td>102</td><td>17.71</td><td>130</td><td>30.73</td></tr><tr><td>Routing</td><td>LoRA</td><td>262</td><td>1-6</td><td>102</td><td>17.71</td><td>129</td><td>29.34</td></tr><tr><td>Rating</td><td>Base</td><td>172</td><td>1-8</td><td>40</td><td>6.94</td><td>165</td><td>93.40</td></tr><tr><td>Rating</td><td>Jev</td><td>172</td><td>1-8</td><td>40</td><td>6.94</td><td>71</td><td>25.17</td></tr><tr><td>Rating</td><td>LoRA</td><td>172</td><td>1-8</td><td>40</td><td>6.94</td><td>69</td><td>27.08</td></tr><tr><td>Selection</td><td>Base</td><td>240</td><td>1-6</td><td>80</td><td>13.89</td><td>159</td><td>57.12</td></tr><tr><td>Selection</td><td>Jev</td><td>240</td><td>1-6</td><td>80</td><td>13.89</td><td>112</td><td>29.69</td></tr><tr><td>Selection</td><td>LoRA</td><td>240</td><td>1-6</td><td>80</td><td>13.89</td><td>108</td><td>26.74</td></tr></table>

Table 25: Small and constant-prediction strata constrain the variation available to reassociation. Counts pool disjoint strata across six cells; weights average cell-level task macro weights. Constant counts include singletons.
<table><tr><td>Split/state</td><td>K</td><td>LoRA QAcc</td><td>LoRA đ [9.025, 9.975]</td><td>Jev QAcc</td><td>Jev đ [9.025, 9.975]</td></tr><tr><td>1/101</td><td>2</td><td>50.00</td><td>+21.48 [+0.00, +37.50]</td><td>25.00</td><td>+2.75 [-12.50, +25.00]</td></tr><tr><td>1/101</td><td>4</td><td>0.00</td><td>+0.00 [+0.00, +0.00]</td><td>0.00</td><td>+0.00 [+0.00, +0.00]</td></tr><tr><td>1/101</td><td>8</td><td>0.00</td><td>+0.00 [+0.00, +0.00]</td><td>12.50</td><td>+9.09 [+0.00, +12.50]</td></tr><tr><td>1/202</td><td>2</td><td>25.00</td><td>+11.99 [-12.50, +25.00]</td><td>12.50</td><td>+8.77 [+0.00, +12.50]</td></tr><tr><td>1/202</td><td>4</td><td>0.00</td><td>-2.69 [-12.50, +0.00]</td><td>12.50</td><td>+10.86 [+0.00, +12.50]</td></tr><tr><td>1/202</td><td>8</td><td>0.00</td><td>+0.00 [+0.00, +0.00]</td><td>12.50</td><td>+10.90 [+0.00, +12.50]</td></tr><tr><td>1/303</td><td>2</td><td>25.00</td><td>+11.21 [+0.00, +25.00]</td><td>12.50</td><td>-2.54 [-25.00, +12.50]</td></tr><tr><td>1/303</td><td>4</td><td>25.00</td><td>+9.15 [-12.50, +25.00]</td><td>12.50</td><td>+11.66 [+0.00, +12.50]</td></tr><tr><td>1/303</td><td>8</td><td>0.00</td><td>-2.76 [-12.50, +0.00]</td><td>0.00</td><td>-0.78 [-12.50, +0.00]</td></tr><tr><td>2/101</td><td>2</td><td>12.50</td><td>+3.33 [-12.50, +12.50]</td><td>25.00</td><td>+14.96 [+0.00, +25.00]</td></tr><tr><td>2/101</td><td>4</td><td>0.00</td><td>-3.09 [-12.50, +0.00]</td><td>25.00</td><td>+6.54 [+0.00, +12.50]</td></tr><tr><td>2/101</td><td>8</td><td>12.50</td><td>+4.28 [+0.00, +12.50]</td><td>0.00</td><td>+0.00 [+0.00, +0.00]</td></tr><tr><td>2/202</td><td>2</td><td>25.00</td><td>+2.73 [-12.50, +12.50]</td><td>50.00</td><td>-3.04 [-12.50, +0.00]</td></tr><tr><td>2/202</td><td>4</td><td>12.50</td><td>+10.88 [+0.00, +12.50]</td><td>12.50</td><td>+3.86 [-12.50, +12.50]</td></tr><tr><td>2/202</td><td>8</td><td>0.00</td><td>-0.16 [+0.00, +0.00]</td><td>0.00</td><td>-0.83 [-12.50, +0.00]</td></tr><tr><td>2/303</td><td>2</td><td>12.50</td><td>+5.09 [-12.50, +12.50]</td><td>12.50</td><td>+5.60 [-12.50, +12.50]</td></tr><tr><td>2/303</td><td>4</td><td>25.00</td><td>+20.89 [+0.00, +25.00]</td><td>0.00</td><td>-7.09 [-25.00, +0.00]</td></tr><tr><td>2/303</td><td>8</td><td>12.50</td><td>+8.43 [+0.00, +12.50]</td><td>12.50</td><td>+9.34 [+0.00, +12.50]</td></tr></table>

Table 26: Maximum-selection departures vary across cells and option counts. QAcc is in percentages and mean differences $\bar { d }$ in pp; brackets give quantiles of $d _ { b } .$ , with eight complete diagnostic quartets in every cell/K stratum.

## L.11 Additional Rule Split and Interface Extensions

For finite-state symbol-mapping swaps, correct-tonew-target transitions are 9/88 (10.23%) for Base, 10/82 (12.20%) for LoRA, and 23/140 (16.43%) for Qwen3.5-9B. The denominator contains initially correct conflicting questions. Table 24 reports the natural and quartet scores. A further disjoint rule split reproduces an adaptation gain dominated by nonconflict questions. The added split uses 16 components disjoint from the original source/development pools, state seed 101, 5,760 training rows, and 1,920 evaluation rows. The original LoRA recipe runs 720 updates, with sourcevalidation NLL selecting epoch 2. These components had appeared in reserved evaluation, so the result is development replication: three rule splits and seven cells overall. Natural accuracy / QAcc are 38.80% / 0.00% for Base and 58.46% / 19.79% for LoRA. The decomposition uses full-set weights. Conflict weight is 36.98%, with conditional gains of 5.99 pp versus 27.69 pp on agreeing answers.

<table><tr><td>Intervention</td><td>Micro</td><td>Task/K macro</td><td></td><td>Equal cell</td><td></td></tr><tr><td></td><td>T/N</td><td>T/B T/N</td><td>T/B</td><td>T/N</td><td>T/B</td></tr><tr><td>Formula rows</td><td>76.82</td><td>88.48</td><td>78.81</td><td>86.47 79.18</td><td>86.58</td></tr><tr><td>Record columns</td><td>91.05</td><td>97.74 91.10</td><td>97.46</td><td>91.08 77.34</td><td>97.41</td></tr><tr><td>Target values</td><td>75.94</td><td>86.17</td><td>77.34</td><td>84.62</td><td>84.35</td></tr></table>

Table 27: Jev's directed transitions persist under micro, task-balanced, and equal-cell weighting. All weighting schemes reuse the same paired requests and archived integer transition counts, changing their aggregation rather than the observed answers or the conflict questions entering the archived counts of paired answer transitions.

![](images/ee8b4245d9d10886c38fd641ee4b188853567da2a27c5ec4a25f9ab105cfb53b.jpg)  
Figure 14: The fixed native chat template lowers 9B natural accuracy under all three wordings. (a) Lines connect raw and native scores on paired inputs. (b) Paired native–raw differences are negative, with 95% intervals from resampling the 13 reserved blocks. The score axis in (a) is restricted to 29–49% for resolving paired changes.

Original decision: instruction   
Select the record with the smallest routing   
penalty, computed as ((x \* y) + max(y,,→   
z)). If several records tie, select the,→   
,→ one with the smallest numeric record ID.   
,→ Only x, y, z participate. Ignore other   
,→ fields and notes.  
Figure 15: The original native instruction asks for the minimum-scoring record and specifies the tie-break. The evaluated native request string is retained exactly.

The fixed native template does not improve the larger model’s direct decision scores. The native check places each rendered prompt in one user message through Qwen3.5-9B’s chat template with enable\_thinking=False. It directly scores candidate letters for all 5,176 questions in each wording. Figure 14 compares this with raw prompting. Paired intervals use 2,000 resamples of the 13 existing blocks. The comparison holds model weights and question identities fixed, using direct candidate scoring without generated reasoning or additional selection of the input prompt or model checkpoint. The tested arithmetic adaptation does not confer broad finite-state rule competence. The sequential extension uses four states, two event symbols, and 20 noncommuting transition tables. Each event permutes the states. We enumerate ordered pairs of distinct permutations whose composition differs in at least one state when reversed, shuffle with seed 2026100807, and retain 20 tables. Every start state and X/Y sequence of length two to four gives 112 inputs per table, with length probabilities $1 / 7 , 2 / 7 , 4 / 7$ under uniform sampling. Each natural set draws 16 inputs without replacement.

```csv
Complete-value decision: native service request
{
"model": "jev-latest",
"questions": {"decision": {"criteria": {"R0": null, "R1": null}, "type": "choice",
"instructions": "Select the record with the smallest routing penalty, computed as ((x * y) +
max(y, z)). If several records tie, select the one with the smallest numeric record ID.,→
Only x, y, z participate. Supplementary information, when present, provides computed,→
scores for formulas or a semantic answer. Use values for the exact formula in this,→
question, then apply its comparison, thresholds, and tie-break. A supplied semantic,→
answer must be matched to the displayed options. An unavailable entry supplies no answer,→
or value; use the original facts or records in that case."}},,→
"state": "{\"records\": [{\"id\": \"R0\", \"x\": 3, \"y\": 2, \"z\": -8}, {\"id\": \"R1\",
\"x\": 4, \"y\": -2, \"z\": 7}], \"supplementary_computed_scores\": [{\"formula\": \"((x *,→
y) + max(y, z))\", \"record_scores\": {\"R0\": 8, \"R1\": -1}}, {\"formula\": \"((x * y) -,→
(y - z))\", \"record_scores\": {\"R0\": -4, \"R1\": 1}}, {\"formula\": \"((x + y) * (y -,→
z))\", \"record_scores\": {\"R0\": 50, \"R1\": -18}}, {\"formula\": \"((x + y) * max(y,,→
z))\", \"record_scores\": {\"R0\": 10, \"R1\": 14}}, {\"formula\": \"((x + y) - (y * z))\",,→
\"record_scores\": {\"R0\": 21, \"R1\": 16}}, {\"formula\": \"max((x * y), (y * z))\",,→
\"record_scores\": {\"R0\": 6, \"R1\": -8}}, {\"formula\": \"max((x - y), (y * z))\",,→
\"record_scores\": {\"R0\": 1, \"R1\": 6}}, {\"formula\": \"min((x - y), (y * z))\",,→
\"record_scores\": {\"R0\": -16, \"R1\": -14}}]}",→
}
```  
Figure 16: The complete-value request supplies scores for every rule while retaining the decision task. The displayed strings preserve the evaluated request, including its instruction, records, formula labels, and numeric values.

<table><tr><td>Evaluation</td><td>Controlled factor</td><td>Evaluation setting</td></tr><tr><td>CheckList / contrast sets</td><td>Local input changes</td><td>Behavioral and contrastive tests</td></tr><tr><td>LogicBench</td><td>Logical inference patterns</td><td>Three kinds of logic</td></tr><tr><td>RuleArena</td><td>Rules and tool assistance</td><td>Real-world regulations</td></tr><tr><td>CFQ / COGS / Ordered CommonGen</td><td>Combinations and concept order</td><td>Compositional generalization</td></tr><tr><td>Mind Your Format / symbol binding</td><td>Templates and option symbols</td><td>Answer-interface variation</td></tr><tr><td>Cacheable decision models</td><td>Joint versus independent encoding</td><td>Rule-following decision scores</td></tr><tr><td>This work</td><td>Rules, states, wording, value mappings</td><td>Oracle-checked decision diagnostics</td></tr></table>

Table 28: Controlled input changes provide different kinds of evidence across evaluation designs. Our oraclechecked setting crosses rules and states and intervenes on equivalent wording and supplied-value mappings.

Adjacent tables form ten pairs. Diagnostic construction excludes both tables’ natural inputs, groups remaining inputs by their two distinct answers, and draws eight quartets per pair from opposite-answer groups. Inputs may recur across quartets within a pair. Each quartet receives a single shared permutation of the displayed options.

Independent function composition verifies every table/input oracle. The resulting 320 natural questions and 80 quartets total 640 questions. Correct and X/Y-swapped tables are evaluated with Base, split-1/state-101 arithmetic LoRA, and 9B, producing 3,840 predictions without new-task training. The 449 conflicts pool natural and diagnostic inputs. Transition rates condition on initially correct conflicts, so denominators differ by model. All systems have zero QAcc, which alone does not establish an additional structural deficit. Intervals resample the ten table pairs.

## M Comparison of Evaluation Designs

Table 28 compares which input factors each evaluation controls and the rule-following settings in which those behavioral input controls are applied. Local perturbation tests reveal behavioral sensitivity, while compositional benchmarks test generalization across combinations. Our quartets make rule and state identities explicit, so item accuracy and all-four correctness answer different evaluation questions. Value-mapping interventions further distinguish following supplied values from retaining the original answer. These designs can complement one another when their controlled factors and score denominators are transparently reported together.