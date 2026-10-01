# Hard-Gate Candidacy in a Deployed Validator Suite

Xin Xu Carnegie Mellon University xuxin@cmu.edu

## Abstract

Before a validator can be promoted to a hard gate on a deployment pipeline, it has to be shown that its firing separates outputs that reach users in working order from those that do not. We run that screen on 13 validators in a deployed generative agent, against 550 runtime and 350 static builds labelled by downstream outcome, and report each check’s marginal separation J = TPR − FPR with Newcombe intervals and Fisher exact tests. Two checks survive correction for multiple comparisons, two more are nominal only, and the remaining nine are not distinguishable from zero, three of them because they never fired on any sampled build. Execution itself is not random with respect to the property being gated, and this replicates: across four runs covering 1,867 builds and ten distinct runtime checks, probes were skipped on 144 of 895 broken builds and 1 of 972 acceptable builds (per-run rates 15.6% to 16.6% against at most 0.3%), every skip carrying the same unsafe-to-probe reason. Because a skipped check is recorded as a pass, this imposes a ceiling that no check quality can lift: a check that needs a live artifact cannot operationally detect more than about 84% of broken builds in this harness. For the one check with construct-specific labels, a detector built for blank output fires on 0 of 90 human-labelled blank builds (95% upper bound on sensitivity 3.3%), and the global frame statistic it approximates separates the classes only weakly (AUC 0.59), so the gap is not a threshold that needs tuning. The same gap appears one layer up: on a census of tens of thousands of judge-scored builds, 32.5% of rejections carry no recorded issue at all. We argue that evaluation records must distinguish a check that ran and passed from one that did not run, must carry the evidence for a rejection, and that an inventory of checks is not evidence about a gate.

## 1 Introduction

Deployed language-model agents are wrapped in automated checking: static analyses of the generated artifact, runtime probes of its behaviour, model-based judges. These checks gate release, trigger regeneration, or route cases to human review, and the record they leave is terse—a verdict per check, summarised into a decision.

We run one specific screen on such a suite. Before a check is promoted to a hard gate, the minimum it must show is that its firing separates the outcomes the gate exists to separate. This is deliberately not the question of whether a check detects the construct its name denotes: a check may detect its construct perfectly and still fail this screen, if that construct is uncorrelated with whether the artifact reaches users in working order. Nor is it the question of a check’s incremental value inside an existing ensemble, which requires the joint firing distribution and the deployed gating rule. What we measure is the marginal screen, which is the question that was actually being asked of these checks.

The general shape of the problem is old. Beer et al. [1997] formalised vacuous satisfaction in temporal-logic model checking, where a property is satisfied whenever its precondition is unsatisfiable, indistinguishable in the verdict from a meaningful pass; Kupferman and Vardi [2003] generalise the detection semantics. Software testing developed the question independently: Barr et al. [2015] survey the oracle problem, Schuler and Zeller [2011] measure whether executed computation reaches an oracle at all, and Niedermayr et al. [2016], Vera-Pérez et al. [2019] document pseudo-tested code, executed by a suite but removable without any test failing. What has not been measured is how this behaves in a deployed language-model-agent gate, where checks are heterogeneous, the failure base rate is low, and the artifact is generated afresh on every request.

## Contributions.

1. A per-check hard-gate screen of a deployed validator suite against downstream outcome, with Newcombe intervals, Fisher exact tests and Holm correction across 13 checks (§3). Two survive correction; nine are not distinguishable from zero, three because they never fired.

2. Evidence that execution is one-sided with respect to the gated property, recurring across four runs and ten distinct runtime checks: 15.6–16.6% of broken builds skipped against at most 0.3% of acceptable ones, per-run z between 4.6 and 7.8, with skips recorded as passes—imposing a harness-level ceiling on operational sensitivity, $\mathrm { T P R } _ { \mathrm { o p } } \leq \dot { 1 } - s _ { 1 } \approx 0 . 8 4$ , on every check that requires a live artifact, a mechanism we namefailure-correlated execution censoring (§4).

3. A construct-aligned measurement for the one check where target labels exist: 0 of 90 humanlabelled instances detected, and evidence that the global frame statistic the detector approximates is itself a weak discriminator (§5).

4. A record prescription: applicability attested per check, so that “ran and passed” and “did not run” cease to be the same symbol (§6).

## 2 Setting, Criterion, and Outcome Proxy

System Under Study. A deployed generative agent produces interactive artifacts from naturallanguage requests. Each request yields a build; a vision-based judge scores it and may trigger a retry. Independently of the judge, automated validators run in two lanes: static checks inspect the generated source, runtime checks load the build in a headless browser and probe it. The suite is larger than the subset we screen; we report the 13 checks for which both outcome classes could be labelled.

Screening Criterion. We screen each check for hard-gate candidacy: does its firing separate the outcome classes? This is the question that was being asked of these checks at the time, the decision under consideration being which were clean enough to enforce. A check that detects its construct perfectly but whose firing does not separate the classes fails this screen, and that is a fact about its candidacy as a gate, not a mismeasurement of the check. We do not measure incremental contribution inside the deployed ensemble; that would require the joint firing distribution and the gating rule (§8).

Outcome Proxy. A build is broken if it failed the judge or a user reported it as not working; acceptable if it passed the judge, was published, and drew no complaint. These are downstream proxies, not verified ground truth, and they incorporate the judge itself; §8 states what that costs. They are independent of any individual validator’s verdict, which is what a per-check screen requires.

Sample Composition. The runtime lane used 550 production builds (235 broken, 315 acceptable;   
broken share 0.427 by construction). The static lane used a balanced 350-build sample (175/175).   
Sampling is by convenience within a single short window.

Statistical Inference. We summarise separation by J = TPR − FPR, which is invariant to the prior probability of failure. J weights a point of TPR equally against a point of FPR, which a cost-sensitive gate would not; it is a screen for whether a check separates the classes at all, not an operating-point recommendation. Intervals are Newcombe hybrid-score intervals on the difference of two proportions, which remain finite at zero cells where a Wald interval degenerates. Because 13 checks are screened in parallel we also report Fisher exact two-sided p-values with Holm adjustment. Rates are operational: a skipped check is counted as not firing, because that is what the pipeline records.

![](images/522ffa870c35e3c2021e9e4cea178c23c36151f2ab32149da872f2e52ea89a59.jpg)  
Figure 1: Marginal separation $J = \mathrm { T P R } - \mathrm { F P R }$ per check with Newcombe $9 5 \%$ intervals. Filled markers survive Holm correction or are clearly unresolved; hollow markers are nominal-only results that do not survive. The three never-fired checks have finite, not degenerate, intervals.

Table 1: Hard-gate screen. Counts are builds on which the check fired. Rates are operational, with skips counted as non-firing. $J = \mathrm { T P R } - \mathrm { F P R }$ with Newcombe 95% interval; p is Fisher exact two-sided; Holm correction is over the 13 distinct checks. FATAL is script-error restricted to uncaught exceptions, an ablation, and is excluded from the correction.
<table><tr><td>Check</td><td>Lane</td><td>broken</td><td>acceptable</td><td>J</td><td>Newcombe CI</td><td>p</td><td>Skip</td></tr><tr><td>script-error</td><td>runtime</td><td>63/235</td><td>15/315</td><td>+0.220</td><td>[+.160, +.283]</td><td> $< 1 0 ^ { - 4 }$ </td><td>0</td></tr><tr><td>engine-metadata</td><td>static</td><td>57/175</td><td>28/175</td><td>+0.166</td><td>[+.076, +.252]</td><td>0.0004</td><td>0</td></tr><tr><td>script-error (FATAL)</td><td>runtime</td><td>39/235</td><td>1/315</td><td>+0.163</td><td>[+.118, +.216]</td><td> $< 1 0 ^ { - 4 }$ </td><td>0</td></tr><tr><td>version-pinning</td><td>static</td><td>29/175</td><td>15/175</td><td>+0.080</td><td>[+.010, +.150]</td><td>0.035</td><td>0</td></tr><tr><td>manifest-DOM</td><td>static</td><td>0/175</td><td>5/175</td><td>-0.029</td><td>[−.065, −.002]</td><td>0.061</td><td>0</td></tr><tr><td>control-reachability</td><td>runtime</td><td>12/235</td><td>9/315</td><td>+0.022</td><td>[−.010,+.061]</td><td>0.185</td><td>40</td></tr><tr><td>manifest-presence</td><td>static</td><td>171/175</td><td>168/175</td><td>+0.017</td><td>[−.023, +.060]</td><td>0.542</td><td>0</td></tr><tr><td>error-hook-order</td><td>static</td><td>170/175</td><td>168/175</td><td>+0.011</td><td>[−.030, +.055]</td><td>0.771</td><td>0</td></tr><tr><td>editable-schema</td><td>static</td><td>174/175</td><td>175/175</td><td>-0.006</td><td>[−.032, +.016]</td><td>1.000</td><td>0</td></tr><tr><td>import-order</td><td>static</td><td>18/175</td><td>21/175</td><td>-0.017</td><td>[−.085, +.050]</td><td>0.735</td><td>0</td></tr><tr><td>interaction-smoke</td><td>runtime</td><td>50/235</td><td>73/315</td><td>-0.019</td><td>[−.088, +.052]</td><td>0.607</td><td>40</td></tr><tr><td>document-language</td><td>static</td><td>0/175</td><td>0/175</td><td>0.000</td><td>[−.021, +.021]</td><td>1.000</td><td>0</td></tr><tr><td>blank-frame A</td><td>runtime</td><td>0/235</td><td>0/315</td><td>0.000</td><td>[−.012, +.016]</td><td>1.000</td><td>40</td></tr><tr><td>blank-frame B</td><td>runtime</td><td>0/235</td><td>0/315</td><td>0.000</td><td>[−.012, +.016]</td><td>1.000</td><td>40</td></tr></table>

## 3 The Hard-Gate Screen

Checks Surviving Multiplicity Correction. script-error $( p < 1 0 ^ { - 4 } )$ and engine-metadata $( p = 0 . 0 0 0 4$ , Holm-adjusted 0.005) separate the classes. The larger, $J = 0 . 2 2 0$ , is modest in absolute terms: the check fires on roughly one broken build in four. version-pinning is nominally positive $( p = 0 . 0 3 5 )$ but Holm-adjusted reaches only 0.39; manifest–DOM has a Newcombe interval excluding zero but Fisher $p = 0 . 0 6 1$ , so its sign is not robust to the choice of test, let alone to correction. We report both as nominal results rather than findings.

Checks Without Detectable Separation. Six checks are not distinguishable from zero, by two different routes. manifest-presence, error-hook-order and editable-schema fire on 96–100% of both classes—editable-schema on 174 of 175 broken builds and all 175 acceptable ones—so their firing is close to constant and cannot carry information about the class. control-reachability, import-order and interaction-smoke fire on a minority of both classes at rates too close to separate at this sample size. A larger sample could resolve the second group; the first cannot, because a check that fires on everything has no room to discriminate.

Checks With No Observed Firing. Three checks never fired: document-language in 350 static builds, and blank-frame A and blank-frame B in every runtime build on which they ran. Their intervals are narrow but finite: the absence of observed firing bounds the rate, it does not establish that the rate is zero. §5 supplies the construct-aligned measurement for one of them.

Separation Versus Alarm Precision. At a failure prevalence of $p = 0 . 1$ , even script-error would flag more false alarms than true ones (4.3% against 2.7% of builds) despite the strongest J in the suite—the standard low-prevalence arithmetic [Axelsson, 2000]. This bears on precision, not on the sign of J, which is class-conditional.

## 4 One-Sided Execution

Every runtime check except script-error was skipped on 40 of 550 builds; the static lane, needing only source text, was skipped on none.

One-Sided Skipping Across Four Runs. Skipping is also one-sided, and this replicates. Table 2 gives per-class skip counts for four independent runs on the same pipeline, recovered from the per-class disposition of every build. The pattern is the same in all four: probes are skipped on roughly one broken build in six and on essentially no acceptable build. Taken together the four runs cover 1,867 sampled builds and give 144 of 895 broken against 1 of 972 acceptable; we report the per-run statistics as primary, because the runs draw from one harness within one window and we cannot establish that they sample disjoint builds, so a pooled test would treat correlated samples as independent. Every skip carries the same recorded unsafe-to-probe reason. Run C is the strongest evidence that the mechanism belongs to the harness rather than to any particular check: its four probes are disjoint from those screened in Table 1, and it shows the same rate.

Table 2: One-sided execution across four runs. Counts are builds on which the probe did not execute. Run B is the sample screened in Table 1; run C covers four runtime checks disjoint from it. †Runs share one harness and window and may sample overlapping builds, so no pooled test statistic is reported.
<table><tr><td>Run</td><td>runtime checks</td><td>skipped / broken</td><td>skipped / acceptable</td><td>∆</td><td>z</td></tr><tr><td>A</td><td>4</td><td>29/175 16.6%</td><td>0/174 0.0%</td><td>+16.6</td><td>5.6</td></tr><tr><td>B</td><td>4</td><td>39/235 16.6%</td><td>1/315 0.3%</td><td>+16.3</td><td>7.3</td></tr><tr><td>C</td><td>4 (disjoint)</td><td>20/125 16.0%</td><td>0/124 0.0%</td><td>+16.0</td><td>4.6</td></tr><tr><td>D</td><td>2</td><td>56/360 15.6%</td><td>0/359 0.0%</td><td> $+ 1 5 . 6$ </td><td>7.8</td></tr><tr><td>combined†</td><td>10 distinct</td><td>144/895 16.1%</td><td>1/972 0.1%</td><td>+16.0</td><td>一</td></tr></table>

This determines how Table 1 reads. Our rates count a skip as a non-firing, which is what the pipeline records. Writing $s _ { 1 }$ for the skip rate on broken builds and $s _ { 0 } \approx 0$ for acceptable ones (at most 0.3% in any run), $\mathrm { T P R } _ { \mathrm { o p } } = ( 1 - s _ { 1 } ) \mathrm { T P R } _ { \mathrm { r u n } }$ and $\mathrm { F P R } _ { \mathrm { o p } } \approx \mathrm { F P R } _ { \mathrm { r u n } } , \mathrm { s o } J _ { \mathrm { r u n } } - J _ { \mathrm { o p } } \approx s _ { 1 } \mathrm { T P R } _ { \mathrm { r u n } } \geq 0 ;$ conditional-on-running values are weakly higher, strictly so only for a check that fires on runnable broken builds at all; for blank-frame A and blank-frame B the two coincide at zero. A verdict-only record cannot distinguish them.

A Harness-Level Ceiling on Operational Sensitivity. Since $\mathrm { T P R } _ { \operatorname* { m i n } } \leq 1$ , the same identity yields $\mathrm { T P R } _ { \mathrm { o p } } \leq 1 - s _ { 1 } $ : in this harness, no check that requires a live artifact can operationally detect more than 0.83–0.84 of broken builds, whatever its logic, because the harness withholds exactly the builds such checks most need to see. We call the mechanismfailure-correlated execution censoring: the probability that a check executes falls with the severity of the condition the check exists to detect—the opposite pole to gray failure [Huang et al., 2017], where the degradation is too subtle for the detector rather than too severe for it to run. The ceiling is a property of harness policy, not of any check, and a verdict-only record is precisely what lets it persist unnoticed. Consistent with it, the one runtime check that executed on every build, script-error—it observes the load itself rather than probing the loaded page—is also the only runtime check whose separation survives correction.

## 5 Construct Validity Where Target Labels Exist

For one check we can measure validity as well as separation. blank-frame A and blank-frame B exist to detect blank or single-colour output, a failure users describe directly. Within the 550-build screen, 14 builds were independently confirmed blank; the detectors fired on none.

To go beyond fourteen cases, we obtained construct-specific labels: 239 builds captured as screenshots and labelled blank or acceptable by human review, 90 blank and 149 not. The detector fired on 0 of 90 blank builds and 0 of 149 acceptable ones, an exact binomial 95% upper bound on sensitivity of 3.3%. We report this set separately from the 14 audit cases rather than pooling them, because we cannot establish that the two sets are disjoint.

Threshold Tuning Versus Construct Redesign. The gap is not a threshold that needs tuning. The shipped detector samples a sparse grid of points and reports a blank frame only when every sampled point has nearly the same colour. Blank builds do have lower median frame luminance than acceptable ones, 37.1 against 57.3, but the distributions overlap heavily: as a single feature, mean frame luminance has AUC = 0.59 (95% CI [0.52, 0.67]), and its best single threshold reaches only TPR = 0.84 at FPR = 0.66. Human raters separate these classes; a global frame statistic does not. Repairing this detector therefore means changing what it measures, not where it cuts—a different and more expensive fix than a verdict-only record would reveal was needed.

## 6 Implications for Evaluation Records

The suite records per-build verdicts. That record cannot express any of the states above. A build for which every check returned pass is written identically whether the checks ran and found nothing, did not run, never fire, or fire on everything. The aggregate claim—the suite passed it—is not falsifiable from the record.

Record Completeness at the Judge Layer. The same gap appears one layer up, at scale. The vision judge records, per build, a verdict and a list of issues. On a census of every scored build in the same window, tens of thousands in all, these two are not consistent with each other in a specific direction: 32.5% of rejections (Wilson 95% [31.1, 34.0]) carry no recorded issue at all, while acceptance with a recorded CRITICAL issue is rare (0.1% of accepted builds). The record therefore does not reconstruct the verdict on the reject side: in roughly a third of rejections, nothing in the artifact explains why the build was rejected. We claim this about the record, not the judge’s reasoning: a downstream consumer—a human triaging failures, an auditor, a retraining pipeline—has nothing to work from in those cases.

The remedy is narrow. A record should carry, per check and per item, whether the check was applicable, so that the record is three-valued—fired, ran without firing, did not run—rather than binary; diagnostic-accuracy reporting has long treated inconclusive results this way rather than folding them into a negative [Shinkins et al., 2013]. At the suite level, a periodic screen of the kind in Table 1, with intervals and multiplicity control, converts an inventory of checks into a statement about what the suite can separate. Both require only information already present in the pipeline, which is currently discarded when it is summarised.

This matters because validator inventories are cited as evidence in deployment and assurance arguments. Bucknall, Reuel, et al. [2025] identify insufficient external access as the limiting factor for governance-relevant evaluation, and single out downstream user logs as a further-restricted gap. Our results add a complication that access alone does not resolve: with full internal access, nine of thirteen checks showed no separation distinguishable from zero, two more were nominal-only, and four were in addition skipped on the builds that most needed checking.

## 7 Related Work

Vacuity, Test Oracles, and Pseudo-Tested Code. Beer et al. [1997] and Kupferman and Vardi [2003] established that a passing verdict may carry no information. Software testing developed the question independently: Barr et al. [2015] survey the oracle problem, Schuler and Zeller [2011] propose checked coverage, and Niedermayr et al. [2016], Vera-Pérez et al. [2019] document pseudotested code at scale. Our never-firing and constant-firing states are the deployed-gate analogue of that phenomenon. What differs is the instrument: that line establishes the property interventionally, by mutating source or injecting faults and observing whether any check reacts [Hsueh et al., 1997, Jahangirova et al., 2016], which presupposes the access and the harness control to do so; we establish it observationally, from production outcomes on a running gate. The two designs answer different questions: intervention asks whether a check can react to a seeded fault, observation asks whether its firing separates the outcomes that production actually delivers.

Reliability of Judges and Guardrails. Zhang et al. [2026] audit a judge gate on a deployed agent and find pooled recall near 18%; Waibl et al. [2026] benchmark 28 supervision systems on in-house data; Reizinger and Brendel [2026] argue that false-positive rate rather than recall decides whether a verifier is deployable; Kamoi et al. [2024] establish low recall for LLM error detectors; Jung et al. [2025] give escalation with provable agreement guarantees. An independent audit of commercial moderation APIs uses curated datasets [Hartmann et al., 2025], and a platform-side study on real moderation traffic withholds its class ratios [Son et al., 2023].

Selective Execution and Verification Bias. A check that runs on part of the population is a selective predictor [Geifman and El-Yaniv, 2017]. Diagnostic-test methodology treats accuracy under selective verification [Begg and Greenes, 1983], but that structure is the mirror of ours: there the index test is always observed and the reference standard is selectively missing, and the standard correction assumes selection depends on the observed test result rather than true status. Here the reference standard is complete and the index test is missing, under a rule keyed on severity. We therefore report operational rates and bound the conditional ones rather than attempt a verification-bias correction.

Failures of Evaluation Pipelines. Guan et al. [2026] attribute close to a third of scoring disagreements to the evaluation pipeline itself; Wu [2026] taxonomises silent failures where a monitor runs and is deceived, whereas our states concern checks that do not fire or do not run. Huang et al. [2017] define gray failure as differential observability for subtle degradations; §4 develops the opposite pole.

## 8 Limitations

Estimand. We measure marginal separation, which is the screen for hard-gate candidacy, not incremental contribution inside the deployed ensemble. Two high-J checks may be redundant; two low-J checks may cover disjoint rare failures. Deciding that needs the joint firing matrix and the gating rule, which we do not have.

Construct Scope. Only §5 has construct-specific labels, and only for one check. A check failing this screen may be a correct detector of a condition that does not separate our outcome classes. Establishing validity for the remaining twelve requires seeded failures or target labels; the interventional designs cited above are the right instrument for that.

Labels and Inference. Outcome labels derive from the deployment’s own judge and from absence of complaints, with no uniform complaint window, so builds sampled near the end of the window had less exposure. Human blank labels lack a stated annotation protocol, rater count, or agreement statistic. Retries can produce several builds per request, so builds are not fully independent and our intervals do not use cluster-robust inference. The judge’s issue list may likewise be incomplete, so §6 reports record completeness, not judge correctness.

Sampling. One deployment, convenience sampling, a single short window; the four skip runs share one harness and window and may overlap. Rates are not claimed to characterise validator suites generally.

## References

S. Axelsson. The base-rate fallacy and the difficulty of intrusion detection. ACM TISSEC, 3(3):186–205, 2000.

E. Barr, M. Harman, P. McMinn, M. Shahbaz, and S. Yoo. The oracle problem in software testing: a survey. IEEE TSE, 41(5):507–525, 2015.

I. Beer, S. Ben-David, C. Eisner, and Y. Rodeh. Efficient detection of vacuity in ACTL formulas. In CAV, 1997.

C. Begg and R. Greenes. Assessment of diagnostic tests when disease verification is subject to selection bias. Biometrics, 39(1):207–215, 1983.

B. Bucknall, A. Reuel, et al. Open problems in technical AI governance. TMLR, 2025. arXiv:2407.14981.

Y. Geifman and R. El-Yaniv. Selective classification for deep neural networks. In NeurIPS, 2017.

H. Guan, L. Fu, S. Zhang, et al. SWE-Cycle: benchmarking code agents across the complete issue resolution cycle. arXiv:2605.13139, 2026.

D. Hartmann, A. Oueslati, D. Staufer, L. Pohlmann, S. Munzert, and H. Heuer. Lost in moderation: how commercial content moderation APIs over- and under-moderate group-targeted hate speech and linguistic variations. In CHI, 2025.

M.-C. Hsueh, T. Tsai, and R. Iyer. Fault injection techniques and tools. IEEE Computer, 30(4):75–82, 1997.

P. Huang et al. Gray failure: the Achilles’ heel of cloud-scale systems. In HotOS, 2017.

G. Jahangirova, D. Clark, M. Harman, and P. Tonella. Test oracle assessment and improvement. In ISSTA, 2016.

J. Jung, F. Brahman, and Y. Choi. Trust or escalate: LLM judges with provable guarantees for human agreement. In ICLR, 2025.

R. Kamoi et al. Evaluating LLMs at detecting errors in LLM responses. In COLM, 2024.

O. Kupferman and M. Vardi. Vacuity detection in temporal model checking. STTT, 4(2):224–233, 2003.

R. Niedermayr, E. Juergens, and S. Wagner. Will my tests tell me if I break this code? In CSED, 2016. arXiv:1611.07163.

P. Reizinger and W. Brendel. HALLMARK: diagnosing three failure modes in LLM citation verifiers. arXiv:2607.18360, 2026.

D. Schuler and A. Zeller. Assessing oracle quality with checked coverage. In ICST, 2011.

B. Shinkins, M. Thompson, S. Mallett, and R. Perera. Diagnostic accuracy studies: how to report and analyse inconclusive test results. BMJ, 346:f2778, 2013.

D. Son, B. Lew, K. Choi, Y. Baek, S. Choi, B. Shin, S. Ha, and B. Chang. Reliable decision from multiple subtasks through threshold optimization: content moderation in the wild. In WSDM, 2023.

O. Vera-Pérez, B. Danglot, M. Monperrus, and B. Baudry. A comprehensive study of pseudo-tested methods. EMSE, 24:1195–1225, 2019.

L. Waibl, F. Michalak, and H. Mariaccia. BELLS-O: evaluating the operational trade-offs of LLM supervision systems. arXiv:2606.20668, 2026.

W. Wu. When errors become narratives: a longitudinal taxonomy of silent failures in a production LLM agent runtime. arXiv:2606.14589, 2026.

S. Zhang, A. Wang, and S. Lei. Catching one in five: LLM-as-judge blind spots in production multi-turn transaction agents. arXiv:2606.10315, 2026.