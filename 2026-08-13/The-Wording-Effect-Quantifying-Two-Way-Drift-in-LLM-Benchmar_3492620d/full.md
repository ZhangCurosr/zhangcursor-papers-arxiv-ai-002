# The Wording Effect: Quantifying Two-Way Drift in LLM Benchmark Performance

Shailja Thakur<sup>1</sup>\* Sungeun An<sup>2</sup> Chad DeLuca<sup>2</sup> Hima Patel<sup>1</sup> <sup>1</sup>IBM Research India <sup>2</sup>IBM Research Almaden

## Abstract

A benchmark score comes from a single phrasing of each problem. That single phrasing is treated as if it stood for the whole space of ways the same problem could be asked, but it does not. We show that rephrasing a problem while keeping its meaning and answer fixed routinely flips a model’s answer in both directions, so some failures become successes and some successes become failures. We call this drift. BenchDrift generates meaning-preserving variations of benchmark problems along four axes, namely linguistic, referential, pragmatic, and structural, and measures how often, and why, correctness flips under each. Across eight models and three benchmarks (GSM8K, MMLU, MATH-Hard), we observe that drift is large in both directions. Two findings stand out. First, phrasing sensitivity does not fade as models get better. Instead, it changes sign. Weak models gain more from rephrasing than they lose, while strong models lose far more than they gain. We find that the best models on a benchmark are therefore the ones whose scores depend most on the wording they happened to be given. Second, the models largely agree on which rephrasings cost the most correct answers even though they differ in how much they drift, so fragility belongs to the rephrasing and not to the model. Furthermore, rephrasing breaks answers a model was confident about, whether the problem is made shorter or longer. Code and Data: https://github.com/IBM/ BenchDrift/tree/demo-ui

## 1 Introduction

Benchmark scores come from one phrasing per problem, and models are known to be sensitive to how that phrasing is written (Sclar et al., 2024; Ribeiro et al., 2020; Goel et al., 2021) — sensitive enough that which model tops a leaderboard has been shown to change once prompts are optimised, even on the same benchmark (Sadjoli et al., 2025).

## One GSM8K problem, four axes, one broken answer

Original ground truth: 188 mm Four books are arranged on a shelf. The first book is 31 mm thick while the second book is 50 mm thick. The third book is 5 mm less thick than the second book, and the fourth book is twice as thick as the first book. What is the total thickness of the four books? Phi-4: 188 mm Pragmatic — artist persona 188mm ✓ In your studio, you have a unique bookshelfdisplay that serves as a piece ofart. The first book adds a layer of 31 mm to your composition, while the second contributes 50 mm. . . Linguistic — politeness 188 mm Would you be so kind as to calculate the total thickness of the four books on the shelf? The first book is 31 mm thick. . . Referential — entity substitution 188 mm The first book is 31 mm thick while the next book is 50 mm thick. The following book is 5 mm less thick than the next book. . . Structural — interrogative expansion wrong × How much less thick is the third book than the second book? What is the thickness ofthe third book?. . . What is the total thickness ofallfour books? Phi-4: “Insufficient information to determine. . . ”

Figure 1: A GSM8K problem reworded along four axes, with Phi-4’s answer to each. Rewording never changes the correct answer, 188 mm, but Phi-4 fails the structural rewording, replying that there is not enough information.

What this prior work does not offer is a way to measure this sensitivity that (1) keeps the correct answer fixed across every rephrasing, so a change in correctness can only be explained by the wording, and (2) attributes each change to the specific rephrasing that caused it and to whether it helped or hurt. Robustness and adversarial testing methods surface failures but often do not preserve the ground-truth answer (Goel et al., 2021; Ribeiro et al., 2020; Wallace et al., 2019). Prompt-repair and self-refinement methods improve outcomes without explaining why a given rephrasing helped or hurt (Madaan et al., 2023; Khot et al., 2022; Shinn et al., 2023; Tian et al., 2024). Prompt optimisers such as DSPy (Khattab et al., 2023) and GEPA (Agrawal et al., 2025) search and discard: they search for the single best-performing prompt for a task and discard the rest once one is found, so the variation itself leaves no trace in what gets reported.

We introduce BenchDrift to close this gap. Given a benchmark problem, BenchDrift generates meaning-preserving variations along four axes — linguistic (wording), referential (which entities are named), pragmatic (tone and context), and structural (sentence organization) — validates that each one keeps the correct answer unchanged (Figure 1 shows one problem reworded along each axis), then collects the model’s answers to the original problem and to every valid variation, and compares their correctness (Figure 2). A flip from wrong to right is positive drift: a hidden capability the original phrasing was hiding. A flip from right to wrong is negative drift: a hidden fragility the original phrasing was masking. Every flip is tied back to the specific rephrasing that produced it.

This paper asks three questions:

1. How much does correctness change under meaning-preserving rephrasing? We measure this directly as positive and negative drift rates, for each model and benchmark.

2. Is this change systematic and tied to specific kinds of rephrasing, or is it closer to noise? We attribute each flip to the transformation that produced it, and test whether different models agree on which transformations cost the most correct answers.

3. What does this mean for how a benchmark score should be read? We report the best- and worst-case accuracy a model shows under different phrasings, alongside the single-phrasing score currently reported.

Contributions: (1) A meaning-preserving variation framework across four axes, with every variation validated against the original ground truth, and a bidirectional drift measurement over a common denominator that attributes each flip to the transformation that produced it. (2) Two tests that rule out the leading alternative explanations for drift: that rephrasing merely simplifies problems, and that it only disturbs answers the model was unsure of. (3) Reliability checks that swap the generator, validator, and judge models. (4) Open release of the generated and validated variations used in our experiments. (5) Open release of the BenchDrift pipeline and full variation taxonomy.

## 2 Method

## 2.1 Roles

BenchDrift runs four roles over each problem.

Generator. Proposes candidate rephrasings of the problem.

Validator. Checks each candidate against the original ground-truth answer and discards any that no longer have it. A changed answer would otherwise mean the problem changed rather than its phrasing.

Target model. The model under evaluation. It answers the original problem and every candidate that survives validation.

Judge. Scores those answers against the ground truth, accepting “12”, “twelve”, and “a dozen” as one answer.

Only the target model is under study. The other three are instruments, and all three are language models, so we re-run the pipeline with different models in those roles to check that the drift we report is not an artifact of the instruments themselves (Section 4.2).

## 2.2 Drift

Let $q$ be a benchmark problem with ground-truth answer $^ { a , }$ and let $V ( q )$ be its variations that the validator has confirmed still have answer a. Write $C ( q ) \in \{ 0 , 1 \}$ for whether the judge scores the target model’s answer to q as correct. A flip in C between q and some $q ^ { \prime } \in V ( q )$ is drift, in one of two directions.

Positive drift $( C ( q ) = 0 , C ( q ^ { \prime } ) = 1 )$ . The model fails the original phrasing and succeeds on the variation: a capability the original wording was hiding.

Negative drift $( C ( q ) = 1 , C ( q ^ { \prime } ) = 0 )$ . The model succeeds on the original and fails the variation: apparent success that rested on a wording choice.

Rates. For a problem set of size N we report both directions as fractions of that same N:

$$
\begin{array} { r l } & { \mathrm { P o s . } = \frac { 1 } { N } \big | \{ q : C ( q ) = 0 , \exists q ^ { \prime } , C ( q ^ { \prime } ) = 1 \} \big | } \\ & { \mathrm { N e g . } = \frac { 1 } { N } \big | \{ q : C ( q ) = 1 , \exists q ^ { \prime } , C ( q ^ { \prime } ) = 0 \} \big | } \end{array}
$$

The shared denominator makes the two directions comparable: each counts problems, out of the same

![](images/184f20860f8ca4259dd201225f507881fc5a15f8b2616a1d76ea79e6b5eb83cf.jpg)  
Figure 2: BenchDrift takes a benchmark problem, generates meaning-preserving variations along four axes, validates that each variation still has the same correct answer, and checks whether the model’s correctness flips — in either direction.

N, that moved one way or the other. Every problem falls into exactly one of four cases: stablecorrect (right on the original and every variation), stable-incorrect (wrong on the original and every variation), negative drift, or positive drift. These four cases cover the problem set with no overlap.

Best- and worst-case accuracy follow directly from this partition. Worst counts a problem correct only if the original and every variation are, so it counts only the stable-correct cases. Best counts a problem correct if the original or any variation is, so it adds the positive-drift cases to that same count. Writing Rep. for the reported single-phrasing accuracy:

$$
\mathrm { B e s t = R e p . + P o s . , } \qquad \mathrm { W o r s t = R e p . - N e g . }
$$

The reported score sits inside an interval whose upper margin is the positive drift rate and whose lower margin is the negative rate. Both identities are exact, so Best and Worst can always be recomputed directly from Rep., Pos., and Neg.

Two extensions of this framework recur later in the paper. First, where we instead report a rate conditioned on a subset of problems — the share of baseline failures with at least one positive-drift variation, for example (Section 4.7) — the denominator is that subset rather than N, and we state this at each use. Second, every drift event is attributed to the transformation that produced it and, through it, to that transformation’s axis (Section 4.4).

## 2.3 Variation Taxonomy

Every transformation BenchDrift can apply comes from a fixed taxonomy of four axes. Linguistic transformations change wording and style, such as rephrasing or switching between active and passive voice. Referential transformations change which entities, units, or values are named, such as swapping a name or converting a unit. Pragmatic transformations change tone, framing, and context, such as adding a persona or recasting the problem as a direct question. Structural transformations change how the problem is organised, such as reordering information or adding text irrelevant to the answer. Each axis groups several transformations, and each transformation requires that a problem carry certain detected properties before it can be applied — a unit conversion needs a unit, an entity substitution needs a name; transformations with no requirement apply to any problem. Table 1 lists, for each axis, the transformations it groups and the properties each one requires.

## 2.4 Selecting Transformations for a Problem

Each problem gets its own set of transformations, chosen in three steps. First, pattern-based detectors scan the problem text and record which properties it has: numbers, fractions, decimals, percentages, 12- and 24-hour times, dates, durations, units, named entities, how many numbers and sentences it contains, and whether it takes more than one step. Second, we compare those properties against the taxonomy, where every transformation lists the properties it needs — a unit conversion needs a unit, an entity substitution needs a name — and every axis inherits its transformations’ requirements. An axis scores the fraction of its requirements the problem meets, and the four axes are ranked by that score. Third, we hand out the problem’s variation budget along that ranking, in the proportions 1.0, 0.75, 0.55, and 0.40, so the best-matching axis gets the most variations and the worst-matching still gets one. Within each axis, we score its transformations the same way and take the highest-scoring until that axis’s allotment is used up.

Table 1: Transformations exercised in our experiments, by axis, with the properties each requires. Detectors mark twenty properties in total; Appendix B gives the full mapping.
<table><tr><td>Axis</td><td>Transformation</td><td>Requires</td></tr><tr><td></td><td>Linguistic rephrasing, narrative style, passive/active voice, politeness, grammar correction symbolic representation</td><td>has_numbers,</td></tr><tr><td colspan="2">Referential entity substitution (1-, 2-, 3-way)</td><td>has_math_operations has_entities</td></tr><tr><td></td><td>Pragmatic ten persona framings: artist, athlete, chef, detective, doctor, engineer, farmer, musician, scientist, teacher Structural interrogative expansion,</td><td></td></tr><tr><td></td><td>programming form, format variation, format whitespace logical form</td><td>has_logical_structure</td></tr><tr><td></td><td>format quotes format case</td><td>has_quotes has_headers</td></tr><tr><td></td><td>positioning of sections</td><td>has_sections, is_long</td></tr><tr><td></td><td>positioning of paragraphs</td><td>has_paragraphs,</td></tr><tr><td></td><td></td><td>is_long</td></tr><tr><td></td><td>quality clarity</td><td>has_pronouns, is_long</td></tr><tr><td></td><td>content removal, quality</td><td>is_long</td></tr></table>

The result is a per-problem list of transformations, weighted toward the axes the problem actually supports. Since the proportions are unequal by design, axes end up with different numbers of variations, which is why we treat axis-level rates as descriptive and rest the analysis on individual transformations (Section 4.4).

## 2.5 Validating the Variations

Generation can change the answer as well as the wording, so every candidate variation is checked before any target model sees it. The validator is asked one question: would a reader who never saw the original arrive at the same answer from this variation alone? Changes to wording, framing, or persona pass. Changes to any number, unit, or logical condition that would move the answer do not. “30 minutes” and “half an hour” are equivalent; “8 days” and “72 hours” are not, because 8 days is 192 hours. Candidates that fail are discarded and appear in none of the results we report.

What survives is a set of variations per problem, each carrying the original ground-truth answer. Every target model then answers the original problem and all of its surviving variations, the judge scores each answer, and the drift we report is whatever flips between the original and a variation.

## 3 Experimental Setup

We evaluate BenchDrift across multiple models and benchmarks to test how much drift occurs, whether it concentrates in specific axes, and what it means for reported benchmark scores.

Benchmarks. GSM8K (Cobbe et al., 2021) (grade-school math), MMLU (Hendrycks et al., 2021) (multi-domain factual questions), and MATH-Hard (harder multi-step math problems), 500 problems sampled per benchmark. These span mathematical reasoning and factual recall, letting us check whether drift patterns generalize across reasoning types.

Models. We evaluate eight open-weight models spanning 7B to 34B parameters: Qwen3- 8B (Yang et al., 2025), Mistral-7B (Jiang et al., 2023a), Phi-4 (Abdin et al., 2024), Granite-3.3- 8B (Granite Team, 2024), GPT-OSS-20B (Agarwal et al., 2025), Granite-34B-Code (Mishra et al., 2024), Llama-3.1-8B (Grattafiori et al., 2024), and Qwen2.5-7B (Team, 2024). Granite-34B-Code is a code model rather than a general instruction-tuned model, and it scores near the bottom on these benchmarks.

Variation generation and validation. Mistral-Large-Instruct-2411 (Mistral AI, 2024) is the generator and GPT-OSS-120B (Agarwal et al., 2025) is the validator (Sections 2.3–2.5). After discarding candidates that fail validation, 20.2 variations remain per problem on average, ranging from 3 to 92; MMLU’s multiple-choice format supports more transformations than GSM8K or MATH-Hard, so its per-problem count runs higher.

Judging. Target models answer with chain-ofthought prompting (Wei et al., 2022) at temperature 0, which removes sampling noise so that the drift we report is not variance across repeated runs of the same prompt. Each phrasing is answered once, so accuracy on the original phrasing (Rep. in Table 2) is pass@1. Llama-3.3-70B-Instruct scores correctness as an LLM judge (Zheng et al., 2023), accepting "12", "twelve", and "a dozen" as correct for ground truth "12" rather than requiring an exact string match.

Main configuration. Mistral-Large-Instruct-2411 as generator, GPT-OSS-120B as validator, and Llama-3.3-70B-Instruct as judge is our main configuration, and produces every result in this paper unless we state otherwise. Section 4.2 is the exception: it reports what happens when different models fill these three roles.

Table 2: Drift across 8 models and 3 benchmarks (500 problems each), models ordered by mean reported accuracy. Rep. is pass@1 accuracy on the original phrasing. Best counts a problem correct if at least one of its validated rephrasings is answered correctly; Worst counts it correct only if every one of its validated rephrasings is (Section 2.2). $\Delta$ is what rephrasing gains minus what it costs, $\Delta = ( \mathrm { B e s t - R e p . } ) - ( \mathrm { R e p . - W o r s t } )$ , shaded red (net loss) or green (net gain). Bold marks the highest-performing result (Best) for each model–benchmark pair. CIs of ±1.8–4.4pp appear in Appendix E.
<table><tr><td></td><td colspan="4">GSM8K</td><td colspan="4">MMLU</td><td colspan="4">MATH-Hard</td></tr><tr><td>Model</td><td>Worst</td><td>Rep.</td><td>Best</td><td>∆</td><td>Worst</td><td>Rep.</td><td>Best</td><td>∆</td><td>Worst</td><td>Rep.</td><td>Best</td><td> $\Delta$ </td></tr><tr><td>GPT-OSS-20B</td><td>38.0</td><td>94.8</td><td>99.0</td><td>-52.6</td><td>23.4</td><td>81.2</td><td>97.0</td><td>-42.0</td><td>16.2</td><td>61.8</td><td>93.6</td><td>-13.8</td></tr><tr><td>Phi-4</td><td>38.2</td><td>93.4</td><td>99.2</td><td>-49.4</td><td>21.4</td><td>80.6</td><td>95.6</td><td>–44.2</td><td>15.2</td><td>58.8</td><td>83.6</td><td>-18.8</td></tr><tr><td>Qwen2.5-7B</td><td>27.6</td><td>87.6</td><td>98.2</td><td>-49.4</td><td>15.2</td><td>68.0</td><td>94.8</td><td>–26.0</td><td>7.8</td><td>49.4</td><td>82.4</td><td>-8.6</td></tr><tr><td>Granite-3.3-8B</td><td>15.2</td><td>86.2</td><td>98.2</td><td>-59.0</td><td>13.6</td><td>64.4</td><td>95.0</td><td>–20.2</td><td>3.4</td><td>43.8</td><td>85.8</td><td>+1.6</td></tr><tr><td>Llama-3.1-8B</td><td>18.8</td><td>82.6</td><td>97.4</td><td>-49.0</td><td>16.4</td><td>62.2</td><td>92.6</td><td>–15.4</td><td>0.8</td><td>26.0</td><td>69.6</td><td>+18.4</td></tr><tr><td>Qwen3-8B</td><td>4.4</td><td>48.0</td><td>88.8</td><td>-2.8</td><td>21.0</td><td>72.0</td><td>95.4</td><td>–27.6</td><td>2.4</td><td>38.0</td><td>75.8</td><td>+2.2</td></tr><tr><td>Granite-34B-Code</td><td>1.4</td><td>48.0</td><td>88.6</td><td>–6.0</td><td>5.8</td><td>46.4</td><td>89.6</td><td>+2.6</td><td>0.0</td><td>19.4</td><td>66.2</td><td>+27.4</td></tr><tr><td>Mistral-7B</td><td>2.8</td><td>48.4</td><td>91.2</td><td>-2.8</td><td>14.2</td><td>54.8</td><td>92.4</td><td>-3.0</td><td>0.0</td><td>9.0</td><td>46.6</td><td>+28.6</td></tr></table>

## 4 Results

We evaluate BenchDrift across the eight models and three benchmarks described in Section 3 to answer the three questions from the introduction. How much correctness changes under rephrasing, and what that means for a reported score, is answered next; whether that change is systematic and tied to specific kinds of rephrasing is answered in Sections 4.3 and 4.4.

## 4.1 The Range Behind a Score

A benchmark score comes from one phrasing. We test the same problems under every validated variation and find a wide range of accuracy the model shows depending on which phrasing it gets. Averaged across all 24 model-benchmark pairs, the gap between worst-case accuracy (correct under every phrasing) and best-case accuracy (correct under at least one phrasing) is 74.7 percentage points (RQ1). Phi-4 on GSM8K makes this concrete. Its reported score is 93.4%. That score drops to 38.2% in the worst case and rises to 99.2% in the best case, a 61-point window with the reported number sitting near the top.

The direction of the shift depends on how strong the model already is. The asymmetry between the two directions — negative minus positive drift — tracks baseline accuracy almost linearly across our 24 pairs (Pearson r = 0.98, slope 1.08). Above a 60% baseline, negative drift exceeds positive drift in all twelve pairs. Below it, the two directions split evenly, six favor recovery and six favor loss.

A weak model has little to lose and something to gain, so rephrasing surfaces knowledge the original wording failed to elicit (Mistral-7B on MATH-Hard: 37.6% positive drift against 9.0% negative). A strong model has much to lose, so the same operation mostly costs it correct answers. For the strongest models, a single-phrasing score therefore tends to overstate robustness (RQ3).

Table 2 shows this directly: GPT-OSS-20B and Phi-4, the two strongest models on GSM8K, both drop below 40% in the worst case despite baselines above 93%.

## 4.2 Reliability of the Measurement

Before interpreting this further, we check whether it reflects the models we tested or an artifact of how we measured them. Table 3 reports drift rates and model-fragility rankings under three alternative combinations of generator, validator, and judge model, alongside the main configuration.

Table 3: Drift under alternative generator/validator/- judge assignments, same 8 models × 3 benchmarks. Row 1 is the main configuration. $\rho$ is the rank correlation of model fragility against it.
<table><tr><td>Generator</td><td>Validator</td><td>Judge</td><td>Neg.</td><td>Pos.</td><td>ρ</td></tr><tr><td>Mistral-L.</td><td>GPT-OSS- 120B</td><td>Llama-3.3- 70B</td><td>45.9±4.3</td><td>28.8±3.7</td><td></td></tr><tr><td>Mistral-L.</td><td>Mistral-L.</td><td>Mistral-L.</td><td>36.2±3.9</td><td>26.0±3.6</td><td>0.99</td></tr><tr><td>Qwen3-30B</td><td>Mistral-L.</td><td>Mistral-L.</td><td>34.5±3.9</td><td>27.2±3.7</td><td>0.98</td></tr><tr><td>Qwen3-30B</td><td>Qwen3-30B</td><td>Qwen3-30B</td><td>34.0±3.8</td><td>25.9±3.6</td><td>0.98</td></tr></table>

Swapping every pipeline role shifts average negative drift by at most 11.9 percentage points and average positive drift by at most 2.9, both within the per-cell confidence intervals of Table 2. The ranking of which models are most fragile is preserved too $( \rho \ge 0 . 9 8 $ against the main configuration). Two of the configurations differ only in which model generates the variations, so comparing them isolates that role. It moves average negative drift by 1.7 points, so no single role dominates the measurement. Together, these checks support treating the drift we measure as a property of the models under test, not of our measurement pipeline. They do not, however, establish agreement with human judgment, which we have not measured.

![](images/d7c6acbd65a92e109dfe028f386f6ba88d848eb2121ab043f7f039b6d7c05622.jpg)

![](images/d4aec73847d3ec6bdff3e3c3d9ddf2786b7488f7f9deb8a53386849d63ff7b57.jpg)  
Figure 3: All 32 transformations, pooled over 8 models and 3 benchmarks; colour gives the axis. Left: the share of correct answers a rephrasing broke, ranked by that rate. Right: the share of wrong answers it recovered, ranked independently by that rate. The two panels use different denominators — answers the model had right, and answers it had wrong — so they are separate rankings, not parts of one total, and a transformation can rank very differently on each side. Axes start at 8% and 5% rather than 0 to keep the bar-length differences legible; printed values give the exact rates. 95% CIs from a problem-level clustered bootstrap.

![](images/3f822d4da9c3902f038dec9ee2253f0e21234018a58a264e71bf938482537b23.jpg)  
Figure 4: Drift by how much the rephrasing changed the problem’s length. Negative drift is lowest when length is preserved and rises in both directions, so drift cannot be attributed to problems simply becoming easier. 95% CIs from a problem-level clustered bootstrap.

## 4.3 Drift by Axis

In Section 4.2, we established that drift is a property of the models rather than an artifact of measurement. We next ask whether it spreads evenly across the four axes or concentrates in a few. As Figure 3 shows (coloured by axis), it concentrates. We test this by shuffling axis labels within each problem’s own variations rather than across problems, since a problem’s variations are not independent; the true spread across axes exceeds all 2,000 shuffles $( p \textless 0 . 0 0 1 )$ . The axes that break the most answers are the ones that leave a problem’s content untouched: persona framing and paragraph reordering change no number and no relation, yet cost more correct answers than entity substitution, which rewrites the tokens the answer depends on. This indicates that the damage tracks surface form, not comprehension.

Because a problem’s content determines how much of its transformation budget each axis receives (Section 2.3), axes with more variations per problem have more chances to flip it, confounding a direct comparison between them. We therefore treat the axis level as descriptive and base the analysis on individual transformations instead.

![](images/a318ea9042c997b6b049928a0701725cbd16e636e38342391e7657b6b063fcb7.jpg)  
Figure 5: Negative drift by baseline confidence, pooled (top) and per model. Baseline confidence is the mean probability the model assigned to the tokens it generated when answering the original problem, binned as lo (<0.85), mid (0.85–0.95), and hi (≥0.95). We use it as a proxy for the model’s confidence in its original answer, not as a calibrated estimate. Dashed line marks the pooled hi rate; 95% CIs from a problem-level clustered bootstrap.

## 4.4 Drift by Transformation

Because axes get unequal shares of a problem’s variation budget (Section 4.3), comparing them directly would be unfair, so we rank individual transformations instead. Figure 3 ranks all 32 transformations by negative drift and, separately, by positive drift. The negative-drift range alone spans more than fourfold: interrogative expansion, which breaks a problem into a series of smaller explicit questions, costs 48.9% of correct answers, while adding whitespace costs only 11.7%. This ranking holds across models too: computing each transformation’s negative drift separately for every model and correlating the rankings gives a mean pairwise Spearman ρ of 0.77 across the 28 model pairs. Models differ in how much they drift overall, but largely agree on which transformations cost the most correct answers (RQ2).

Furthermore, negative drift and positive drift are not two sides of the same coin: as Figure 3 shows, only programming formulation appears in both top-8 lists, causing the most negative drift (41.2%) and also the most positive drift (21.6%). Persona and structural transformations otherwise dominate the negative-drift list but cause little positive drift, while linguistic transformations such as symbolic representation (25.3% negative, 20.1% positive) dominate the positive-drift list without causing much negative drift. Reporting only one direction would miss this.

Colour-coding the transformations by axis shows the axis level still understates fragility: pragmatic and structural transformations cluster at the highnegative-drift end and referential and linguistic ones at the stable end, but the clusters overlap enough that a single transformation crosses lines — linguistic narrative style (27.9%) breaks far more answers than structural whitespace (11.7%). The transformation, not its axis, predicts fragility.

## 4.5 Drift and Problem Length

One candidate explanation for why the transformations in Section 4.4 break answers is that they simply make problems easier or harder. We test this by checking whether drift tracks problem length, using the change in length as a proxy for edit size. As shown in Figure 4, negative drift is lowest when length is roughly preserved (19.6%) and rises in both directions — to 28.9% when the problem gets shorter and 34.7% when it more than doubles in length. Shortening a problem causes about as much negative drift as lengthening it, which rules out drift as an artifact of problems simply becoming easier.

## 4.6 Drift and Model Confidence

A candidate explanation for the drift documented so far is a model’s own confidence: low confidence could flag which correct answers are fragile and which scores to distrust. As shown in Figure 5, negative drift falls as confidence rises, from 38.1% in the low bin to 18.5% in the high bin, but does not approach zero — nearly one in five answers the model was most confident about is still lost to a meaning-preserving rephrasing. The pattern holds within every model tested, from 13% to 26% in the high bin. Therefore, a model’s own confidence cannot be used to tell which of its correct answers will survive rephrasing.

## 4.7 How Many Variations Are Needed

The results so far use roughly 13 variations per problem to measure drift rates, which is also the cost of running BenchDrift. We now ask how many of those variations are actually needed to see the drift they reveal. We subsample k variations per problem at random and recount how many problems show at least one flip. With the full budget, 67.2% of problems drift. Five variations recover

74% of those, eight recover 88%, and ten recover   
93%; a single variation finds only 30%.

The curve flattens well before our budget, so an audit on a tighter budget loses less than the reduction in cost suggests. It also does not flatten at one or two variations, which is what a cheap robustness check would use — most of what we report would be missed at that scale.

## 4.8 Comparison to Prompt Optimisers

In this section, we compare against prior work such as prompt optimisers, which handle phrasing sensitivity differently. DSPy (Khattab et al., 2023) and GEPA (Agrawal et al., 2025) search over prompts and keep only the best one, a search-anddiscard strategy, against BenchDrift’s keep-andmeasure-the-spread approach. We run both alongside BenchDrift on the same 15 GSM8K problems, Llama-3.1-8B-Instruct (Table 4).

Table 4: Accuracy and wall-clock time per method. DSPy and GEPA each report one optimised prompt’s score; BenchDrift reports Worst/Rep./Best across its validated variations.
<table><tr><td>Method</td><td>Worst</td><td>Rep.</td><td>Best</td><td>Time</td></tr><tr><td>DSPy (Khattab et al., 2023)</td><td></td><td>93.3%</td><td></td><td>1.8s</td></tr><tr><td>GEPA (Agrawal et al., 2025)</td><td></td><td>60.0%</td><td></td><td>62.0s</td></tr><tr><td>BenchDrift</td><td>26.7%</td><td>86.7%</td><td>100.0%</td><td>2.1s</td></tr></table>

At a cost close to DSPy’s and far below GEPA’s, BenchDrift returns a 73.3-point range (26.7% to 100%) that neither optimiser reports, because both search for a single best-performing prompt and discard everything else once they find it. BenchDrift instead keeps the variations its per-problem property detection selects (Section 2.4), not a random sample of possible rephrasings. On these 15 problems, at comparable cost, DSPy and GEPA hand back a single number that fails to capture a model’s real capability, since it offers no way to tell what a different phrasing would have done. BenchDrift instead reports the full range of accuracy across its filtered variations — worst, representative, and best. Furthermore, it attributes each point in that range to the transformation and axis that produced it.

## 5 Related Work

Models are sensitive to phrasing: formatting alone moves benchmark accuracy by tens of points (Sclar et al., 2024), equivalent prompts disagree about what a model knows (Elazar et al., 2021), and templated edits expose failures a single test set hides (Ribeiro et al., 2020). Existing approaches address this sensitivity from different angles. Standard benchmarks (Hendrycks et al., 2021; Cobbe et al., 2021; Wang et al., 2019) report accuracy on one phrasing, a single point estimate. LLM judges and verifiers (Zheng et al., 2023; Lightman et al., 2023) check whether one answer is correct, not whether that correctness holds across equivalent inputs. Adversarial and decomposition methods (Wallace et al., 2019; Goel et al., 2021; Khot et al., 2022) induce failures by altering the task itself, so a changed answer is uninformative about wording alone.

Prompt optimisation and self-refinement methods (Madaan et al., 2023; Shinn et al., 2023; Pryzant et al., 2023; Khattab et al., 2023; Agrawal et al., 2025) search for a better prompt without isolating which property of the wording mattered. Inference-time aggregation (Wang et al., 2022; Jiang et al., 2023b) reduces output variance at 5– 40× cost, but still reports one number. Paraphrase and format robustness studies (Sclar et al., 2024; Elazar et al., 2021; Ribeiro et al., 2020) come closest to our setting, but report that outputs change rather than which kind of change is responsible, without turning the result into a range around the reported score. BenchDrift instead keeps every validated, meaning-preserving variation of a problem and reports the worst, reported, and best accuracy across them, with each point attributable to the specific transformation and axis that produced it.

## 6 Conclusion

A benchmark score computed from one phrasing is not a stable estimate of what a model can do. Across eight models and three benchmarks, accuracy under different valid phrasings of the same problems spans, on average, 74.7 percentage points, and models more often lose correct answers to rephrasing than they gain new ones. We find that drift is systematic. It concentrates in specific transformations, holds up under reliability checks that swap out the entire pipeline, and BenchDrift attributes each flip to the transformation and axis responsible.

## References

Marah Abdin, Jyoti Aneja, Harkirat Behl, Sébastien Bubeck, Ronen Eldan, Suriya Gunasekar, Michael Harrison, Russell J. Hewett, Mojan Javaheripi, Piero

Kauffmann, James R. Lee, Yin Tat Lee, Yuanzhi Li, Weishung Liu, Caio C. T. Mendes, Anh Nguyen, Eric Price, Gustavo de Rosa, Olli Saarikivi, and others. 2024. Phi-4 technical report. Preprint, arXiv:2412.08905.

Sandhini Agarwal, Lama Ahmad, and others. 2025. gptoss-20b model card. Preprint, arXiv:2508.10925.

Lakshya A Agrawal, Shangyin Tan, Dilara Soylu, Noah Ziems, Rishi Khare, Krista Opsahl-Ong, Arnav Singhvi, Herumb Shandilya, Michael J Ryan, Meng Jiang, Christopher Potts, Koushik Sen, Alexandros G Dimakis, Ion Stoica, Dan Klein, Matei Zaharia, and Omar Khattab. 2025. Gepa: Reflective prompt evolution can outperform reinforcement learning. arXiv preprint arXiv:2507.19457.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, and others. 2021. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168.

Yanai Elazar, Nora Kassner, Shauli Ravfogel, Abhilasha Ravichander, Eduard Hovy, Hinrich Schütze, and Yoav Goldberg. 2021. Measuring and improving consistency in pretrained language models. Transactions ofthe Associationfor Computational Linguistics, 9:1012–1031.

Karan Goel, Nazneen Rajani, Jesse Vig, Samson Tan, Jason Wu, Stephan Zheng, Caiming Xiong, Mohit Bansal, and Christopher Ré. 2021. Robustness gym: Unifying the nlp evaluation landscape. Preprint, arXiv:2101.04840.

IBM Granite Team. 2024. Granite 3.0 language models. Technical report, IBM.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, and others. 2024. The llama 3 herd of models. Preprint, arXiv:2407.21783.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. 2021. Measuring massive multitask language understanding. Proceedings of ICLR.

Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, and others. 2023a. Mistral 7b. arXiv preprint arXiv:2310.06825.

Dongfu Jiang, Xiang Ren, and Bill Yuchen Lin. 2023b. Llm-blender: Ensembling large language models with pairwise ranking and generative fusion. arXiv preprint arXiv:2306.02561.

Omar Khattab, Arnav Singhvi, Paridhi Maheshwari, Zhiyuan Zhang, Keshav Santhanam, Sri Vardhamanan, Saiful Haq, Ashutosh Sharma, Thomas T

Joshi, Hanna Moazam, Heather Miller, Matei Zaharia, and Christopher Potts. 2023. Dspy: Compiling declarative language model calls into self-improving pipelines. arXiv preprint arXiv:2310.03714.

Tushar Khot, Harsh Trivedi, Matthew Finlayson, Yao Fu, Kyle Richardson, Peter Clark, and Ashish Sabharwal. 2022. Decomposed prompting: A modular approach for solving complex tasks. arXiv preprint arXiv:2210.02406.

Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. 2023. Let’s verify step by step. arXiv preprint arXiv:2305.20050.

Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, Shashank Gupta, Bodhisattwa Prasad Majumder, Katherine Hermann, Sean Welleck, Amir Yazdanbakhsh, and Peter Clark. 2023. Self-refine: Iterative refinement with self-feedback. Preprint, arXiv:2303.17651.

Mayank Mishra, Matt Stallone, and others. 2024. Granite code models: A family of open foundation models for code intelligence. Preprint, arXiv:2405.04324.

Mistral AI. 2024. Large enough. Mistral Large 2.

Reid Pryzant, Dan Iter, Jerry Li, Yin Tat Lee, Chenguang Zhu, and Michael Zeng. 2023. Automatic prompt optimization with "gradient descent" and beam search. arXiv preprint arXiv:2305.03495.

Marco Tulio Ribeiro, Tongshuang Wu, Carlos Guestrin, and Sameer Singh. 2020. Beyond accuracy: Behavioral testing of NLP models with CheckList. In Proceedings ofthe 58th Annual Meeting ofthe Associationfor Computational Linguistics, pages 4902– 4912. Association for Computational Linguistics.

Nicholas Sadjoli, Tim Siefken, Atin Ghosh, Yifan Mai, and Daniel Dahlmeier. 2025. Optimization before evaluation: Evaluation with unoptimised prompts can be misleading. In Proceedings of the 63rd Annual Meeting ofthe Associationfor Computational Linguistics: Industry Track.

Melanie Sclar, Yejin Choi, Yulia Tsvetkov, and Alane Suhr. 2024. Quantifying language models’ sensitivity to spurious features in prompt design or: How i learned to start worrying about prompt formatting. Preprint, arXiv:2310.11324.

Noah Shinn, Federico Cassano, Edward Berman, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. 2023. Reflexion: Language agents with verbal reinforcement learning. arXiv preprint arXiv:2303.11366.

Qwen Team. 2024. Qwen2.5 technical report. Preprint, arXiv:2412.15115.

Runchu Tian, Yining Ye, Yujia Qin, Xin Cong, Yankai Lin, Yinxu Pan, Yesai Wu, Haotian Hui, Weichuan Liu, Zhiyuan Liu, and Maosong Sun. 2024. Debugbench: Evaluating debugging capability of large language models. arXiv preprint arXiv:2401.04621.

Eric Wallace, Shi Feng, Nikhil Kandpal, Matt Gardner, and Sameer Singh. 2019. Universal adversarial triggers for attacking and analyzing NLP. Preprint, arXiv:1908.07125.

Alex Wang, Yada Pruksachatkun, Nikita Nangia, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel R Bowman. 2019. Superglue: A stickier benchmark for general-purpose language understanding systems. In Advances in Neural Information Processing Systems, pages 3266–3280.

Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. 2022. Self-consistency improves chain of thought reasoning in language models. arXiv preprint arXiv:2203.11171.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le, and Denny Zhou. 2022. Chain-of-thought prompting elicits reasoning in large language models. arXiv preprint arXiv:2201.11903.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, and others. 2025. Qwen3 technical report. Preprint, arXiv:2505.09388.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, and others. 2023. Judging llm-as-a-judge with mt-bench and chatbot arena. arXiv preprint arXiv:2306.05685.

## A Limitations

No human validation ofthejudge or validator. Both the semantic-equivalence validator and the correctness judge are LLMs. We show that swapping which LLM plays these roles changes results only slightly (Section 4.2) — but that shows the roles agree with each other, not that any of them agree with a human. We have not run a human-annotation study, so we cannot yet rule out a shared blind spot across all the LLMs we tried. A more specific risk is directional, not just generic uncertainty: if the judge is more permissive in one direction than the other (e.g., more willing to accept a borderline answer on a variation than on the original), that alone could inflate or deflate positive and negative drift asymmetrically — which matters because the asymmetry between them (Section 4) is one of our central findings. This is the most important open question for trusting the exact numbers we report.

Three benchmarks, all closed-form. GSM8K, MMLU, and MATH-Hard all have a single, checkable correct answer. We do not know whether drift behaves the same way on open-ended tasks (summarization, long-form generation) where correctness itself is not binary — this is a natural and, we think, important next step, not something this paper covers.

Generator andjudge modelfamilies overlap with target models. Several of the models we generate variations with or judge responses with (e.g., Mistral-Large, Qwen3-30B) are from the same families as some of the target models we evaluate. Section 4.2 shows fragility rankings hold up across role swaps, which is reassuring, but does not fully rule out shared family-level biases affecting all configurations similarly.

Axis membership is inferred, not recorded. Our logs identify the transformation applied to each variation but not the axis it belongs to, so we recover axis labels by mapping transformation identifiers onto the taxonomy. The mapping is deterministic and applied uniformly, but it is our own, and a different grouping of the same transformations would shift the axis-level numbers. The transformation-level results in Section 4.4 do not depend on it.

Cost. Running BenchDrift end to end — variation generation, validation, and response collection across many models — is more expensive than evaluating a single phrasing once, which makes it better suited to periodic audits than to routine per-run evaluation. Section 4.7 shows that most of the drift we detect is reachable with a fraction of our full budget, but the first run still costs more than a standard evaluation.

Scope. BenchDrift measures whether a benchmark score is stable under meaning-preserving rephrasing; it is not a prompt optimizer, and the specific rephrasings that help one problem will not automatically transfer to another. What transfers is the kind of change that tends to help or hurt for a given model and benchmark.

## B Transformation-to-Property Mapping

Detectors mark twenty problem properties: has\_numbers, has\_number\_words, has\_decimal, has\_fraction, has\_percentage, has\_math\_operations, has\_units, has\_duration, has\_date, has\_time\_12hr, has\_time\_24hr, has\_temporal\_reference, has\_entities, has\_pronouns, has\_quotes, has\_headers, has\_sections, has\_paragraphs, has\_code\_syntax, and is\_long.

The taxonomy declares transformations beyond those exercised here, and a transformation runs only when a problem carries every property it requires. On the referential axis this gating is what reduced the axis to entity substitution: the eleven other referential transformations convert times, dates, durations, fractions, decimals, number words, and units, and our three benchmarks rarely contain them. Table 5 gives the complete mapping.

Entity substitution is labelled by how many places in the problem the substituted entity occurs: an entity mentioned three times, changed everywhere it occurs, is a 3-way substitution. Changing only some of its mentions would leave a variation whose answer is no longer the same, since a later sentence may still refer back to the original entity, so every occurrence moves together. Depending on how often the substituted entity recurs, this ranges from 1-way up to several dozen.

Table 5: Every transformation in the taxonomy and the properties it requires. Transformations in bold were exercised in our experiments.
<table><tr><td>Axis</td><td>Transformation</td><td>Requires</td></tr><tr><td>Ling.</td><td>hypothetical framing interrogative narrative style irrelevant context near transfer, far transfer symbolic representation</td><td>has_numbers,</td></tr><tr><td>Ref.</td><td>12hr → 24hr time 24hr → 12hr time duration unit conversion date format variation relative time conversion fraction → decimal fraction → percentage decimal → fraction word → number number → word</td><td>has_time_12hr has_time_24hr has_duration has_date has_temporal_reference has_fraction has_fraction has_decimal has_number_words has_numbers</td></tr><tr><td>Prag.</td><td>formal tone, casual tone simplify, elaborate</td><td></td></tr><tr><td>Struct.</td><td>whitespace add/normalise logical formulation quote single ↔ double header case upper/lower section reverse paragraph reverse</td><td>has_logical_structure has_quotes has_headers has_sections, is_long</td></tr></table>

## C Infrastructure and Reproducibility

Infrastructure. All models are served with vLLM on 2 NVIDIA H100 GPUs. Target models answer at temperature 0 (Section 3); the variation generator and validator run at a nonzero temperature to encourage diverse rephrasings, with the exact value given in the released configuration files.

Code and data release. The full BenchDrift pipeline, prompt templates for the generator, validator, and judge, the variation taxonomy, and the generated and validated variations used in our experiments are available at https://github.com/IBM/BenchDrift/tree/demo-ui.

## D Transformation Examples

The following illustrates BenchDrift variations of one GSM8K problem, answered by Qwen3-8B; it is a single worked example, not a representative sample across benchmarks or models. ✓ marks a variation Qwen3-8B answered correctly, × marks one it answered incorrectly.

Baseline: A store sells apples for \$3 each and oranges for \$5 each. If John buys 4 apples and 2 oranges, how much does he spend in total? Answer: \$22

## 1. Rephrasing (Linguistic) ✓

What is the total amount spent when purchasing 4 apples at \$3 apiece and 2 oranges at \$5 apiece?

## 2. Narrative Style (Linguistic) ✓

John headed to the local store one afternoon. Apples were priced at \$3 each and oranges at \$5. He picked up 4 apples and 2 oranges. How much did he pay at the checkout?

## 3. Programming Form (Structural) ×

Given: apple\_price = 3, orange\_price = 5, apple\_qty = 4, orange\_qty = 2. Compute: total\_cost = (apple\_price × apple\_qty) + (orange\_price × orange\_qty). What is total\_cost?

## 4. Farmer Persona (Pragmatic) ×

At the farmers market, a grower prices apples at \$3 per fruit and oranges at \$5 per fruit. A customer takes 4 apples and 2 oranges. What is the total bill?

## 5. Hypothetical Framing (Linguistic) ×

Suppose apples cost \$3 each and oranges cost \$5 each. If someone were to buy 4 apples and 2 oranges, what would the total come to?

## E Confidence Intervals for Table 2

Gained is the positive drift rate and Lost is the negative drift rate (Section 2.2), each with its 95% bootstrap confidence interval, resampled at the problem level.

Table 6: Per-cell 95% bootstrap confidence intervals for the drift rates in Table 2, resampled at the problem level.
<table><tr><td>Model</td><td>Bench</td><td>Gained</td><td>Lost</td></tr><tr><td>GPT-OSS-20B</td><td>GSM8K</td><td>4.2±1.8</td><td>56.8±4.3</td></tr><tr><td>GPT-OSS-20B</td><td>MMLU</td><td>15.8±3.2</td><td>57.8±4.3</td></tr><tr><td>GPT-OSS-20B</td><td>MATH-Hard</td><td>31.8±4.1</td><td>45.6±4.3</td></tr><tr><td>Phi-4</td><td>GSM8K</td><td>5.8±2.1</td><td>55.2±4.3</td></tr><tr><td>Phi-4</td><td>MMLU</td><td>15.0±3.1</td><td>59.2±4.3</td></tr><tr><td>Phi-4</td><td>MATH-Hard</td><td>24.8±3.8</td><td>43.6±4.3</td></tr><tr><td>Qwen2.5-7B</td><td>GSM8K</td><td>10.6±2.7</td><td>60.0±4.3</td></tr><tr><td>Qwen2.5-7B</td><td>MMLU</td><td>26.8±3.9</td><td>52.8±4.4</td></tr><tr><td>Qwen2.5-7B</td><td>MATH-Hard</td><td>33.0±4.1</td><td>41.6±4.3</td></tr><tr><td>Granite-3.3-8B</td><td>GSM8K</td><td>12.0±2.9</td><td>71.0±4.0</td></tr><tr><td>Granite-3.3-8B</td><td>MMLU</td><td>30.6±4.0</td><td>50.8±4.4</td></tr><tr><td>Granite-3.3-8B</td><td>MATH-Hard</td><td>42.0±4.3</td><td>40.4±4.3</td></tr><tr><td>Llama-3.1-8B</td><td>GSM8K</td><td>14.8±3.1</td><td>63.8±4.2</td></tr><tr><td>Llama-3.1-8B</td><td>MMLU</td><td>30.4±4.0</td><td>45.8±4.4</td></tr><tr><td>Llama-3.1-8B</td><td>MATH-Hard</td><td>43.6±4.3</td><td>25.2±3.8</td></tr><tr><td>Qwen3-8B</td><td>GSM8K</td><td>40.8±4.3</td><td>43.6±4.3</td></tr><tr><td>Qwen3-8B</td><td>MMLU</td><td>23.4±3.7</td><td>51.0±4.4</td></tr><tr><td>Qwen3-8B</td><td>MATH-Hard</td><td>37.8±4.2</td><td>35.6±4.2</td></tr><tr><td>Granite-34B-Code</td><td>GSM8K</td><td>40.6±4.3</td><td>46.6±4.4</td></tr><tr><td>Granite-34B-Code</td><td>MMLU</td><td>43.2±4.3</td><td>40.6±4.3</td></tr><tr><td>Granite-34B-Code</td><td>MATH-Hard</td><td>46.8±4.4</td><td>19.4±3.5</td></tr><tr><td>Mistral-7B</td><td>GSM8K</td><td>42.8±4.3</td><td>45.6±4.3</td></tr><tr><td>Mistral-7B</td><td>MMLU</td><td>37.6±4.2</td><td>40.6±4.3</td></tr><tr><td>Mistral-7B</td><td>MATH-Hard</td><td>37.6±4.2</td><td>9.0±2.5</td></tr></table>