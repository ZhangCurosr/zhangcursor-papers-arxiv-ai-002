# VLTest: Finding Joint Image–Text Failures in Vision–Language Models

Bin Duan The University of Queensland Brisbane, Australia b.duan@uq.edu.au

Jintao Lin   
The University of Queensland Brisbane, Australia   
jintao.lin@student.uq.edu.au

Guowei Yang<sup>∗</sup> The University of Queensland Brisbane, Australia guowei.yang@uq.edu.au

## Ab<sub>s</sub>t<sub>rac</sub>t

Vision–language models (VLMs) are increasingly embedded in software systems. Testing them raises two problems. Mutating either input can change the expected answer, so an output change is not by itself a failure. Some failures appear only when both inputs are mutated; attributing them to the combination requires testing each mutation alone. We present VLTest, a black-box VLM testing approach. It generates image variants by editing VQGAN latent codes and text variants with a rewrite–expand–prune pipeline, filters variants likely to change the expected answer, and tests each variant alone before testing combinations, so every joint failure has four matched cells. A VLM judge confirms a failure only after checking that the input keeps the expected answer. On three tasks with eight VLMs, including two commercial API models, the oracle confirms 81.7% of the candidates, with 88.8% to 94.6% precision against human labels, and four-cell confirmation rejects 22.5% of joint candidates. Under the same budget, VLTest’s generators find 1.8 to 2.5 times as many confirmed failures as SINVAD and TextFooler, and its joint tests find 91 to 197 attributed joint failures per 1,000 joint calls, 1.3 times as many as the same pairing applied to the baselines’ variants.

## ACM Reference Format:

Bin Duan, Jintao Lin, and Guowei Yang. 2026. VLTest: Finding Joint Image– Text Failures in Vision–Language Models. ACM Trans. Softw. Eng. Methodol. 1, 1 (October 2026), 16 pages. https://doi.org/10.1145/nnnnnnn.nnnnnnn

## 1 I<sub>n</sub>t<sub>ro</sub>d<sub>uc</sub>ti<sub>on</sub>

Vision–language models (VLMs) [36, 50, 78] take an image and a text as input and produce a decision or a natural-language answer. They power image–text matching [18], visual reasoning [22], visual question answering [41], cross-modal retrieval [68], and multimodal search [4], and they are increasingly integrated into software systems [1]. Their predictions can change under small variations of either input [6, 7, 58], which makes their behavioral reliability a software quality concern.

Software engineering research has developed efective techniques for testing deep neural networks (DNNs). Seed-centric generators explore the neighborhood of an input under a budget [13– 15, 21, 32, 48, 67], latent-space generators keep the generated inputs realistic [12, 29, 70], and metamorphic relations provide test oracles when the expected output of a generated input is unknown [8, 57]. Applying these techniques to VLMs, however, raises two challenges.

Challenge 1: output changes are not failures. Seed-centric testing usually relies on an invariance relation: if a mutation preserves the content that determines the answer, the answer should not change. In a VLM, the expected answer depends on how the image and the text relate. A paraphrase can change what a question asks, a visual edit can change whether a statement is true, and two edits that are harmless on their own can change the answer together. An image–caption pair can remain well aligned while its correct match label changes. Counting output changes, as robustness evaluations of VLMs commonly do [7], therefore mixes real failures with correct reactions to changed inputs. Studies of label preservation for generated test inputs focus on unimodal classifiers [43, 44]; for VLMs, the expected answer also depends on the relation between the two inputs.

Challenge 2:joint failures need attribution. Some failures occur only when both inputs are mutated. To show that such a failure is caused by the combination, a tester must also run the image-only and the text-only variants, check that the model handles both correctly, and check that all inputs share the same expected answer. We call these four inputs the four cells of a joint test: the seed, the image-only input, the text-only input, and the joint input. The cells should difer only by the mutations they contain; an image mutation made in the latent space of a generative model also carries reconstruction artifacts, which the text-only input must then share. Even with matched controls, the efects of two mutations can add up to a failure without any interaction between the modalities [23]. Existing approaches that mutate both modalities, such as metamorphic testing of VQA systems [72] and multimodal adversarial attacks [40, 73, 79], do not pair the final joint input with matched single-modality controls whose expected answers are confirmed by an oracle. Without such controls, a higher failure count under joint mutation may simply reflect that two sources of perturbation were used instead of one.

To address these challenges, we propose VLTest, a black-box testing approach for VLMs. The key idea of VLTest is to treat a joint input and its two single-modality counterparts as one test, and to report a failure only after its expected answer has been confirmed by an oracle. VLTest has three components. First, two controlled generators produce visual variants, by replacing dominant codebook entries in the discrete latent space of a pretrained VQGAN [17], and textual variants, through a rewrite–expand–prune pipeline. Both emit deduplicated variants in a fixed order. Second, relation filters remove variants that are likely to change the expected answer, using bidirectional entailment for text, an image similarity floor, and task-specific image–text alignment rules. Third, a two-phase testing procedure admits only seeds that the model answers correctly, first tests each variant on its own, and then tests combinations of these variants. All four cells are built on the reconstructed seed image, so that each cell difers from the seed only by the mutations it contains. A feedback-directed pairing strategy uses the single-modality results to test first the combinations that single-modality testing has not already broken.

VLTest confirms failures with an automated oracle. A VLM judge receives the seed and its answer as a reference and first decides, without seeing the model output, whether a generated input keeps that answer; only then does it decide whether the output is wrong. A failure is reported only if both checks succeed, and a joint failure only if they succeed for all four cells. Human labels on a stratified sample of the reported failures measure the precision of the oracle.

We evaluate VLTest on image–text matching (MSCOCO), visual reasoning (NLVR2), and visual question answering (VQA v2) with eight VLMs: four task-specific models with six fine-tuned checkpoints, Qwen3-VL at two scales, and two commercial API models. We compare the generators of VLTest with SINVAD [29], adapted to the VQGAN latent, and TextFooler [28], and its joint testing with five baselines: random pairing with and without filtering, random pairing that first uses the combinations whose single-modality inputs are answered correctly, stable-first pairing of the variants found by SINVAD and TextFooler, and the applicable relations of MetaRA [72]. The oracle confirms 81.7% of the candidates of VL-Test (SF) and reaches a precision of 80.0% to 94.6% against human labels across the compared methods in the primary experiment, and of 88.8% to 94.6% on the failures that VLTest (SF) reports; four-cell confirmation rejects 22.5% of the joint candidates, 19.7% through the joint input alone and a further 2.8% through the single-modality controls. Under the same budget, the generators of VLTest find 1.8 to 2.5 times as many confirmed failures as SINVAD and TextFooler, and VLTest finds 91 to 197 attributed joint failures per 1,000 joint calls, 1.3 times as many as the stable-first pairing of their variants and 2.8 times as many as the applicable relations of MetaRA, which exhaust their combinations after about half of the joint budget. Under an additive margin model, 16.5% of the attributed joint failures with output scores are interactions between the modalities that the sum of the two single-modality efects does not predict. On Qwen3-VL, GPT-4o mini, and Gemini 2.5 Flash, VLTest finds 112 to 250 attributed joint failures per 1,000 joint calls.

The main contributions of this paper are as follows.

• We formulate VLM testing with oracle-confirmed expected answers and introduce four-cell evidence, built on a common reconstructed reference, for attributing a failure to the combination of an image mutation and a text mutation.

• We propose VLTest, which combines controlled visual and textual generators, relation filters, and two-phase testing that reuses the single-modality tests as the matched controls of every joint test and as feedback for pairing.

• We design an automated test oracle that, for every cell, checks the expected answer against the seed with a VLM judge before it assesses the model output, and we validate the oracle against human labels.

• We provide a replication artifact [65] for VLTest.

## 2 B<sub>ac</sub>k<sub>groun</sub>d <sub>an</sub>d R<sub>e</sub>l<sub>a</sub>t<sub>e</sub>d W<sub>or</sub>k

## 2<sub>.</sub>1 B<sub>ac</sub>k<sub>groun</sub>d

VLMs encode an image and a text into a shared or fused representation and decode a task output [42, 78]. Architectures range from single-stream fusion encoders such as ViLT [30], to encoders pretrained at scale such as BLIP [37] and BEiT-3 [69], to generative models that couple a vision encoder with a language model, such as PaliGemma [5] and recent open-weight families [3, 35, 49, 82]. We consider three tasks with diferent output types: image–text matching (ITM, binary), visual reasoning over a statement and an image pair (VR, binary), and visual question answering (VQA, short free-form answers with several acceptable answers).

## 2<sub>.</sub>2 R<sub>e</sub>l<sub>a</sub>t<sub>e</sub>d W<sub>or</sub>k

Table 1 compares VLTest with representative approaches.

Test Input Generation for DNNs. Seed-centric test generation explores the neighborhood of an input to expose behaviors that fixed datasets miss [61]. Coverage-guided methods perturb inputs in pixel space [21, 32, 48, 67]. Generative methods search a learned latent space so that the generated inputs remain realistic: SINVAD [29] searches the latent space of a VAE, CIT4DNN [12] applies combinatorial coverage to a discretized latent space, Mimicry [70] targets decision boundaries, and DeepHyperion [83] explores a feature map. Recent work benchmarks generative models for test input generation [43] and validates generated inputs with a human-aligned learned validator [44], showing that validity and label preservation vary across generators. Riccio and Tonella [55] study when and why test generators for DL produce invalid inputs, and Dola et al. [11] use a generative model to discard generated inputs that lie outside the training distribution. These approaches target unimodal classifiers. VLTest extends seed-centric generation to image–text inputs, whose expected answer depends on both modalities, and confirms that answer with an oracle for every reported failure.

Evaluation of VLMs. Benchmarks remain the main way to evaluate VLMs. Chen et al. [7] measure robustness under visual and textual corruptions, VHELM [33] evaluates VLMs along multiple aspects, NaturalBench [34] collects natural adversarial samples that cannot be solved from one modality alone, Rahmanzadehgervi et al. [51] show that VLMs fail at simple visual tasks, and compositional benchmarks probe attribute binding and word order [26, 62]. MultiMon [63] mines failure patterns from inputs on which a text encoder and a VLM erroneously agree. These works evaluate models on fixed or curated inputs. VLTest complements them by generating new inputs from each seed.

Metamor<sub>p</sub>hic and Search-Based Testin<sub>g</sub> of Multimodal S<sub>y</sub>stems. Metamorphic testing [8, 57] checks relations between the outputs of related inputs when the expected output is unknown. MetaVQA [76] synthesizes counting and existence questions from detected objects and checks relations under question transformations, such as splitting or negating a question, and under image transformations, such as inserting or removing objects; each relation transforms one modality. MetaRA [72] applies metamorphic relations to the image–question inputs of MLLM-based VQA systems and reports the failure patterns they reveal. For image captioning, MetaIC [75] inserts objects into images and ROME [74] removes them, and both check how the captions of the transformed images should relate. Metamorphic relations have also been defined for vision–language–action models in robotic manipulation [64], and search-based testing has been applied to VLMs for in-car scene understanding [59]. VLTest also builds on an invariance relation. It difers in that it checks the precondition of the relation, namely that the expected answer is preserved, for every reported failure, and it pairs each joint failure with matched single-modality controls whose answers are confirmed. We compare with the relations of MetaRA that apply to our setting (Section 5.3).

Adversarial Attacks on VLMs. Adversarial attacks optimize perturbations to induce mispredictions under a threat model [56, 81]. Co-Attack [79] perturbs the text first and then the image, guided by the adversarial text, and SGA [40] alternates guidance between the two modalities to improve transferability. VLATTACK [73] uses a pretrained model to attack black-box models fine-tuned from it, with single-modality image and text attacks followed by an iterative cross-search over both modalities. Attacks maximize attack success under a perturbation bound. VLTest pursues a diferent goal: it generates controlled test inputs and reports failures with oracle-confirmed expected answers and attribution evidence.

Text Perturbation and NLP Testing. Text adversarial methods such as DeepWordBug [19], SCPN [27], PWWS [53], and TextFooler [28], and behavioral test suites such as CheckList [54], perturb text for robustness and capability testing. They supply the lexical operators that our text pipeline builds on, and TextFooler is the text-side baseline in our evaluation.

Combinatorial Interaction Testing. Combinatorial interaction testing (CIT) [31, 46] builds on the observation that many faults are triggered by interactions among a few input factors. A joint failure is such an interaction between two factors, the image and the text. In classical CIT, factors have small predefined domains and covering arrays enumerate their combinations. In VLTest, the levels of each factor are generated variants, so VLTest selects which combinations to test and attributes each failure with four-cell evidence.

## 3 P<sub>ro</sub>bl<sub>em</sub> F<sub>ormu</sub>l<sub>a</sub>ti<sub>on</sub>

## 3.1 In<sub>p</sub>uts and Ex<sub>p</sub>ected Answers

A VLM � maps an image–text pair $\boldsymbol { s } = \left( \boldsymbol { x } , t \right)$ to an output �(�). A task-specific normalization norm(·) maps outputs to comparable answers: binary labels for ITM and VR, and canonical short answers for VQA following the oficial evaluation script [2, 66]. A seed $s _ { 0 } =$ $( x _ { 0 } , t _ { 0 } )$ is a dataset input. A VR input also contains a second image, which stays fixed in all inputs derived from a seed and is omitted from the notation.

The reference answers $A _ { 0 }$ of a seed come from the dataset: the label for ITM and VR, and for VQA the answers that receive full credit under the oficial metric, that is, answers given by at least four of the ten annotators. The expected answer $A ( s )$ of an input � is the set of answers that are correct for �. It can contain more than

$A _ { 0 } ,$ such as a synonym of a reference answer, and for a generated input it is unknown until the oracle establishes it. � is correct on � if norm $( M ( s ) ) \in A ( s )$ , and the output of � is compatible if norm $M ( s ) ) \in A _ { ( }$ <sub>0</sub>. A compatible output is correct whenever $A ( s ) =$ $A ( s _ { 0 } )$ , and an incompatible output can still be correct.

## 3.2 Test Oracle

VLTest uses the invariance relation

$$
A ( s ) = A ( s _ { 0 } ) \implies \mathrm { n o r m } ( M ( s ) ) \in A ( s _ { 0 } ) .\tag{1}
$$

Its precondition depends on how the image and the text relate and cannot be read of the mutation. VLTest therefore separates the check done during testing, which compares outputs only with the reference answers, from the confirmation done by an automated oracle (Section 4.6), which establishes the expected answer.

Definition 3.1 (Candidate). Let $s _ { 0 }$ be an admitted seed (Section 3.3). A generated input � is a candidate if its output is not compatible, norm(�(�)) ∉ $A _ { 0 } .$

Definition 3.2 (Confirmed failure). A candidate � is a confirmed failure if the oracle establishes that � is a valid input with $A ( s ) =$ $A ( s _ { 0 } )$ and that norm(�(�)) ∉ �(�).

Definition 3.3 (Attributed joint failure). A joint input $( x , t )$ of an admitted seed � is an attributed joint failure $\mathrm { i f } \left( x , t \right)$ is a confirmed failure and the oracle establishes that both single-modality cells $( x , t _ { 0 } )$ and $( \hat { x } _ { 0 } , t )$ are valid inputs with expected answer $A ( s _ { 0 } )$ on which � is correct.

A candidate need not be a failure: its expected answer may have changed, the input may be invalid, or its output may be a correct answer outside $A _ { 0 } .$ . Two single-modality variants that each preserve the answer do not guarantee that their combination does, so the oracle checks the joint cell on its own. A joint input that is a confirmed failure is not necessarily an attributed joint failure: the latter additionally requires both single-modality controls, which attribute the failure to the combination of the two mutations. We refer to the former as a joint input confirmed on its own.

## 3.3 Reference In<sub>p</sub>ut

Every visual variant passes through the VQGAN encoder and decoder, and reconstruction alone can change details such as colors, counts, or text in the image. VLTest therefore builds all cells on the reconstructed seed ${ \hat { x } } _ { 0 } = D ( E ( x _ { 0 } ) )$ : the seed cell is $( \hat { x } _ { 0 } , t _ { 0 } ) _ { : }$ , a text-only input is $( \hat { x } _ { 0 } , t )$ , and an image-only or joint input contains a decoded image variant. Each cell then difers from the seed cell only by the mutations it contains. A seed is admitted for a model if the oracle confirms that reconstruction keeps its expected answer and the output of the model on $( \hat { x } _ { 0 } , t _ { 0 } )$ is compatible. Each model is also called once on the raw seed $( x _ { 0 } , t _ { 0 } )$ , so that we can report how often reconstruction changes its output; this call is not a test.

## 3<sub>.</sub>4 I<sub>n</sub>t<sub>erac</sub>ti<sub>on an</sub>d Additi<sub>ve</sub> F<sub>a</sub>il<sub>ures</sub>

An attributed joint failure shows that neither mutation causes the failure alone. The two efects may still simply add up: each mutation moves the model toward a wrong answer, and only together do they cross the decision boundary. For models that expose output scores, we separate this case from an interaction between the mutations.

Table 1: Comparison of VLTest with representative approaches (I = image, T = text). In the last column, n/a means that the approach does not test joint inputs, and no means that it does, without matched single-modality controls.
<table><tr><td>Approach</td><td>Test inputs</td><td>Mutated</td><td>Expected answer of generated inputs</td><td>Matched four-cell confirmation</td></tr><tr><td>Chen et al. [7]</td><td>predefined corruptions of dataset inputs</td><td>I or T</td><td>assumed unchanged</td><td>n/a</td></tr><tr><td>NaturalBench [34]</td><td>curated nâtural adversarial</td><td></td><td>human-annotated</td><td>n/a</td></tr><tr><td>MultiMon [63]</td><td>pairs mined from existing corpora</td><td></td><td>summarized post hoc</td><td>n/a</td></tr><tr><td>MetaVQA [76]</td><td>relation-driven transformations of one modality</td><td>I or T</td><td>defined by the relation</td><td>n/a</td></tr><tr><td>MetaRA [72]</td><td>relation-driven transformations</td><td>I and T</td><td>defined by the relation</td><td>no</td></tr><tr><td>Co-Attack [79], SGA [40]</td><td>white-box optimization with cross-modal guidance</td><td>I+T</td><td>assumed under perturbation bounds</td><td>no</td></tr><tr><td>VLATTACK [73]</td><td>per-example optimization</td><td>I, T, I+T</td><td>assumed under perturbation</td><td>no</td></tr><tr><td>SINVAD [29],</td><td>with cross-search latent-space generation</td><td>I (unimodal</td><td>bounds not checked</td><td>n/a</td></tr><tr><td>CIT4DNN [12] DeepNaqqal [44]</td><td>validator for generated</td><td>models) I (unimodal</td><td>not addressed; input validity</td><td>n/a</td></tr><tr><td>VLTEST</td><td>inputs seed-centric generation; single-modality tests, then</td><td>models) I, T, I+T</td><td>predicted by a learned validator filtered, then confirmed by an</td><td>yes</td></tr></table>

Let �ˆ be the answer of � on the joint input $( x , t ) ,$ and let $\eta ( s ) =$ $\begin{array} { r } { \operatorname* { m a x } _ { y \in A _ { 0 } } \log \hat { p } _ { M } ( y \mid s ) - \log \hat { p } _ { M } ( \hat { y } \mid s ) } \end{array}$ be the margin of the best reference answer over �ˆ on a cell �. Adding the efects of the two mutations predicts the margin of the joint input as

$$
\tilde { \eta } ( x , t ) = \eta ( x , t _ { 0 } ) + \eta ( \hat { x } _ { 0 } , t ) - \eta ( \hat { x } _ { 0 } , t _ { 0 } ) .\tag{2}
$$

An attributed joint failure is an interaction failure if $\tilde { \eta } ( x , t ) > 0 .$ that is, if the additive prediction still prefers a reference answer, and an additive failure otherwise. The diference $\Delta _ { \mathrm { i n t } } = \eta ( x , t ) - \eta ( x , t _ { 0 } ) -$ $\eta ( \hat { x } _ { 0 } , t ) + \eta ( \hat { x } _ { 0 } , t _ { 0 } )$ is the interaction contrast of the $2 \times 2$ design that the four cells form. Hessel and Lee [23] use a related additive projection to test whether a model uses cross-modal interactions on a dataset; we apply the contrast to each attributed joint failure. The analysis uses the scores only for evaluation; VLTest itself uses normalized outputs.

## 4 <sup>A</sup><sub>pp</sub>roac<sup>h</sup>

VLTest treats the model under test as a black box and accesses only its inputs and normalized outputs. Its generators and filters use pretrained, of-the-shelf models. Figure 1 gives an overview of its four stages. First, the visual and textual generators (Sections 4.1 and 4.2) emit deduplicated variants of the seed in fixed orders. Second, the relation filters (Section 4.3) remove variants that are likely to change the expected answer, before any call to the model under test. Third, two-phase testing (Section 4.4 and Algorithm 1) admits the seed, runs the single-modality tests, and then runs joint inputs in the order of a pairing strategy (Section 4.5). Fourth, the automated oracle (Section 4.6) turns the candidates into confirmed failures and attributed joint failures.

## 4<sub>.</sub>1 Vi<sub>sua</sub>l V<sub>ar</sub>i<sub>an</sub>t G<sub>enera</sub>ti<sub>on</sub>

VLTest produces visual variants by editing the discrete latent grid of a pretrained VQGAN [17]. The encoder maps a seed image $x _ { 0 }$ to an index grid $Z _ { 0 } = E ( x _ { 0 } ) \in \{ 1 , . . . , K \} ^ { h \times w }$ , each index refers to a codebook entry with embedding ${ \mathbf { e } } ( c ) \in \mathbb { R } ^ { m }$ , and the decoder � maps a grid back to an image. Intuitively, the grid describes the image as ℎ × � cells, each labeled with one of � reusable “visual words”. Replacing a label at all cells where it occurs edits the corresponding visual content throughout the image. Because the decoder has a receptive field larger than one cell, an edit also influences neighboring content.

Index ranking. For an index � in $Z _ { 0 } ,$ , let fre $\mathbb { Q } _ { x _ { 0 } } ( c ) ~ = ~ | \{ ( u , v ) ~ \} |$ $Z _ { 0 } [ u , v ] = c \} |$ |. VLTest sorts the indices by decreasing frequency, breaking ties by increasing index value, and keeps the top � as $C _ { x _ { 0 } } = \langle c _ { 1 } , \ldots , c _ { \kappa } \rangle$ . Frequent indices cover larger regions and correspond to dominant visual prototypes, so replacing them produces observable edits with few variants.

Replacement along embedding distance. For each $c _ { i } ~ \in ~ C _ { x _ { 0 } } .$ , the other codebook indices are sorted by embedding distance, $N ( c _ { i } ) =$ $\langle c _ { i } ^ { ( 1 ) } , \ldots , c _ { i } ^ { ( K - 1 ) } \rangle$ ⟩, with $\| \mathbf { e } ( c _ { i } ) - \mathbf { e } ( c _ { i } ^ { ( r ) } ) \| _ { 2 }$ non-decreasing in �. The variant at rank � replaces every occurrence of each �<sub>�</sub> by $c _ { i } ^ { ( r ) }$ , for all � at once, and decodes the edited grid.

Mutation strength and order. The rank range $\{ 1 , \ldots , R \}$ , with $R \leq K -$ 1, controls mutation strength, and the number of requested variants controls how many ranks are used. VLTest visits ranks in a fixed bitreversal (van der Corput) order: the �-th variant uses rank $\sigma _ { R } ( j )$ where �<sub>�</sub> enumerates the bit-reversed integers $0 , . . . , 2 ^ { \lceil \log _ { 2 } R \rceil } -$ 1, shifted by one, and skips values above �. Every prefix of this order spreads nearly uniformly over the range, so requesting more variants only appends to the sequence. VLTest removes duplicates by grid hash and, after the model’s own preprocessing, by a hash of the model input; for the API models, whose preprocessing is not observable, the second hash is taken over the request sent to the API.

## 4<sub>.</sub>2 T<sub>e</sub>xt<sub>ua</sub>l V<sub>a</sub>ri<sub>a</sub>nt G<sub>e</sub>n<sub>e</sub>r<sub>a</sub>ti<sub>o</sub>n

VLTest produces textual variants with a three-stage pipeline that bounds the neighborhood of a seed text $t _ { 0 } .$ . In the rewrite stage, a pretrained paraphraser (PEGASUS [80]) generates � sentence-level rewrites. In the expand stage, spaCy [25] identifies the content tokens of each rewrite, excluding named entities, numbers, and punctuation, and each eligible token is replaced by one of the top-� fill-mask predictions of RoBERTa [39] with the same part of speech; at most � tokens are substituted per rewrite. In the prune stage, the candidates of each rewrite group are ranked by Sentence-BERT [52] cosine distance to $t _ { 0 } ,$ and the closest �/� per group are kept after exact-string deduplication. The output interleaves the groups by within-group rank, so that, as on the visual side, a larger pool only appends to a smaller one.

![](images/840a551a68ecdee42a42337425c4c236f918b1d52d5c2c8d989695ebd89fa678.jpg)  
Fi<sub>gu</sub>r<sub>e</sub> 1: O<sub>ve</sub>r<sub>v</sub>i<sub>ew</sub> <sub>o</sub>f VLTest<sub>.</sub>

## 4<sub>.</sub>3 R<sub>e</sub>l<sub>a</sub>ti<sub>on</sub> Filt<sub>ers</sub>

The generators control how far a variant moves from the seed, but not whether its expected answer changes. VLTest applies three filters before any model call. RQ1 evaluates their precision and recall against the oracle.

Text filter $F _ { \mathrm { t x t } }$ . A variant � is kept if a natural language inference model [71] predicts entailment in both directions, min $\{ P ( t _ { 0 } \ \Rightarrow$ $t ) , P ( t \Rightarrow t _ { 0 } ) \} \geq \theta _ { \mathrm { n l i } }$ . Bidirectional entailment targets equivalence of meaning, which embedding distance alone does not capture.

Image filter $F _ { \mathrm { i m g } } .$ A visual variant � is kept if its image embedding is close to that of the seed, sim $( x , x _ { 0 } ) \geq \rho _ { \mathrm { i m g } } $ . This removes degenerate edits.

Pair filter $F _ { \mathrm { p a i r } } .$ . An image–text alignment score �(�, �) is used with task-specific rules:

• ITM, positive seed: keep $( x , t ) { \mathrm { ~ i f ~ } } a ( x , t ) \geq \rho \cdot a ( { \hat { x } } _ { 0 } , t _ { 0 } )$ , to retain pairs that are likely to still match.

• ITM, negative seed: keep (�, �) $\mathrm { ~ f ~ } a ( x , t ) \leq a ( \hat { x } _ { 0 } , t _ { 0 } ) + \mu ,$ to discard pairs that drift toward a match.

• VR, true statement: keep (�, �) if $a ( x , t ) \geq \rho \cdot a ( \hat { x } _ { 0 } , t _ { 0 } )$ on the mutated image.

For false VR statements and VQA questions, image–text similarity does not reflect the expected answer, so VLTest applies only $F _ { \mathrm { t x t } }$ and $F _ { \mathrm { i m g } }$ and leaves the joint relation to the oracle. Both image-based filters use OpenCLIP ViT-bigG/14 [9]. We chose it over SigLIP [77] because PaliGemma and Qwen3-VL build on SigLIP encoders [49], and the filters should not share a vision encoder with the models under test. The thresholds are $\theta _ { \mathrm { n l i } } = 0 . 8 0 , \rho _ { \mathrm { i m g } } = 0 . 9 0 , \rho = 0 . 9 5$ , and � = 0.05, fixed on a pilot (Section 5.5).

## 4<sub>.</sub>4 T<sub>wo</sub>-Ph<sub>ase</sub> T<sub>es</sub>tin<sub>g</sub>

Algorithm 1 runs the tests for one seed. Test(�) obtains the output of � through Call, logs � as a test, adds � to F<sub>�</sub> or $\mathcal { F } _ { T }$ if it is a single-modality candidate, and returns whether � is a candidate. The strategies � ∈ {RR, SF} are those of VLTest (Section 4.5); the other values of� and the flag � implement the joint baselines of Section 5.3 within the same procedure.

Admission (Section 3.3) is per model, so each model is tested on the inputs it answers with a reference answer. The single-modality phase tests the first � variants of each filtered sequence, and the joint phase combines exactly these variants. Because every singlemodality input is tested before the joint phase, each joint candidate has complete four-cell evidence when it is found. The singlemodality phase also yields the single-modality failures of the run. The pair filter is applied again to the joint input, because two variants that pass it separately need not pass it together. A combination that fails the re-check is skipped without a call.

Algorithm 1 Two-phase testing for one seed and one model   
Re uire: seed � $\mathbf { \dot { \rho } } _ { 0 } = \left( x _ { 0 } , t _ { 0 } \right)$ with reference answers $A _ { 0 }$ and reconstruction �ˆ ; mode   
�; ordered sequences X, T; pool size �; number of joint tests � ; filter flag �;   
pairing strategy �   
Ensure: call log �; candidates $\mathcal { F } _ { V } , \mathcal { F } _ { T }$ ; joint candidates $\mathcal { F } _ { J }$ with four cells   
$1 \colon \mathcal { F } _ { V } , \mathcal { F } _ { T } , \mathcal { F } _ { J } \gets 0$ ⊲ Call reuses a logged output or issues one new call   
2: Call(�<sub>0</sub>, �<sub>0</sub> ) ⊲ raw seed, not a test   
3: if $A ( { \hat { x } } _ { 0 } , t _ { 0 } ) \neq A ( s _ { 0 } )$ or norm $( \operatorname { C A L L } ( { \hat { x } } _ { 0 } , t _ { 0 } ) ) \not \in A _ { 0 }$ th<sub>en</sub>   
4: return seed not admitted for �   
5: end if   
6: if � then   
7: $\ ' X \gets [ x \in X \mid F _ { \mathrm { i m g } } ( x ) \land F _ { \mathrm { p a i r } } ( x , t _ { 0 } ) ]$   
8: $\mathcal { T }  [ t \in \mathcal { T } \mid F _ { \mathrm { t x t } } ( t ) \land F _ { \mathrm { p a i r } } ( \hat { x } _ { 0 } , t ) ]$   
9: end if   
10: $\chi \prime \gets$ first � of $\begin{array} { r l } { \chi , } & { { } \mathcal { T }  } \end{array}$ first � of T   
11: for all $x \in X ^ { \prime }$ d<sub>o</sub> ⊲ single-modality phase   
12: Test(�, � )   
13: end for   
14: for all $t \in \mathcal { T } ^ { \prime }$ d<sub>o</sub>   
15: Test $( { \hat { x } } _ { 0 } , t )$   
16: end for   
17: $C _ { J } \gets 0$   
18: for all $( x , t ) \in \psi .$ .Order $X ^ { \prime } , \mathcal { T } ^ { \prime } , L )$ while $C _ { J } < n _ { J }$ d<sub>o</sub> ⊲ joint phase   
19: $\mathbf { i f } \ \phi \wedge { \neg F } _ { \mathrm { p a i r } } ( x , t )$ th<sub>en con</sub>ti<sub>nue</sub> ⊲ joint re-check, no call   
20: <sub>en</sub>d if   
21: $C _ { J } \gets C _ { J } + 1$   
22: if Test(�, �) then   
23: add $\langle ( \hat { x } _ { 0 } , t _ { 0 } ) , ( x , t _ { 0 } ) , ( \hat { x } _ { 0 } , t ) , ( x , t ) \rangle$ and the four outputs to F<sub>�</sub>   
24: <sub>en</sub>d if   
25: end for   
26: return $L , \mathcal { F } _ { V } , \mathcal { F } _ { T } , \mathcal { F } _ { J }$

## 4<sub>.</sub>5 P<sub>a</sub>irin<sub>g</sub> Str<sub>a</sub>t<sub>eg</sub>i<sub>es</sub>

VLTest provides two strategies for ordering the pairs in $\smash { \chi ^ { \prime } \times \mathcal { T } ^ { \prime } }$

Round-robin (RR). Let � be the longer of the two pools and � the shorter one, with sizes $n _ { P } \geq n _ { Q }$ . The �-th pair, $k = 0 , 1 , \ldots ,$ combines � [� mod �<sub>�</sub>] with �[( (� mod $n _ { P } ) + \lfloor k / n _ { P } \rfloor )$ mod $n _ { Q } ]$ Each round of �<sub>�</sub> pairs uses every element of the longer pool once and shifts the shorter pool by one position, so after �<sub>�</sub> rounds every pair has appeared exactly once.

Stable-first (SF). SF uses the results of the single-modality tests as feedback. Pairs whose image-only and text-only inputs both have compatible outputs come first, in RR order, followed by the remaining pairs in RR order. Because an attributed joint failure requires correct single-modality inputs, SF first tests the combinations whose parts single-modality testing has not broken.

## 4<sub>.</sub>6 O<sub>rac</sub>l<sub>e</sub>

VLTest confirms candidates with an automated oracle built on a VLM judge �. The judge receives the seed and its reference answers $A _ { 0 }$ as a reference, so it decides whether an input keeps a known answer instead of answering the input from scratch. For each generated cell �, the oracle runs two checks in separate calls, whether or not the output of the cell is compatible. � receives the seed, $A _ { 0 } ,$ and �, and returns preserved if � is a valid input with the expected answer of the seed, changed if its expected answer difers, invalid if � is not a meaningful input for the task, or abstain. It does not see any model output, so the output cannot influence the expected answer. � then receives the seed, $A _ { 0 } ,$ �, and the output, and returns correct if the output is a correct answer for �, such as a synonym of a reference answer, incorrect, or abstain. A cell is correct only if the first check returns preserved and the second returns correct. The two verdicts can contradict each other when the output is compatible: preserved means that the reference answers remain correct, and a compatible output is one of them. The oracle keeps the verdict of the output check in this case, so such a cell counts as incorrect and rejects the joint candidate that contains it (Section 6.3). A candidate is a confirmed failure if its answer check returns preserved and its output check returns incorrect. A joint candidate is an attributed joint failure if the answer check returns preserved for its three generated cells and the output check returns correct for both single-modality inputs and incorrect for the joint input. Abstentions do not confirm failures. The judge is not told which method produced an input. The oracle also runs the answer check on the reconstructed seed of every sampled seed. On API models, a failure must also be reproducible. The seed cell $( \hat { x } _ { 0 } , t _ { 0 } )$ and each cell of a candidate are called five more times, and a cell is correct or incorrect if at least three of the five new outputs are. A confirmed failure then requires a correct seed and an incorrect candidate, and an attributed joint failure also requires correct single-modality inputs. These calls are repeat calls $C _ { \mathrm { r e p } }$

## 5 E<sub>v</sub>aluation Desi<sub>g</sub>n

## 5.1 Research Questions

• RQ1 (Validity): How many candidates does the oracle confirm as failures, and how precise is it?

• RQ2 (Testing utility and cost): How efectively and eficiently does VLTest find single-modality failures and attributed joint failures?

• RQ3 (Joint failures): What attributed joint failures does VLTest reveal?

• RQ4 (Sensitivity and generalization): How do the results vary across design choices and VLMs?

RQ1 breaks the candidates down by oracle outcome, measures the precision of the oracle and of its rejections against human labels, and evaluates the filters and VQGAN reconstruction. RQ2 compares the single-modality failures with two single-modality baselines and the attributed joint failures with five joint baselines, under the same budget. RQ3 breaks the joint candidates down by four-cell outcome, separates interaction from additive failures, and categorizes the failures. RQ4 varies the design choices of VLTest and repeats the joint comparison on four general-purpose VLMs.

## 5<sub>.</sub>2 D<sub>a</sub>t<sub>ase</sub>t<sub>s an</sub>d M<sub>o</sub>d<sub>e</sub>l<sub>s</sub>

Following [7], we use ITM on MSCOCO [38], VR on NLVR2 [60], and VQA on VQA v2 [20]; each dataset input is a seed. An NLVR2 input has two images and a statement; VLTest mutates the first image, keeps the second, and gives both to the oracle. Each task has two fine-tuned models: BLIP [37] and ViLT [30] for ITM, BEiT-3 [69] and ViLT for VR, and PaliGemma [5] and BEiT-3 for VQA. Seeds are sampled at random from the validation splits, one per source image (per image pair for NLVR2). For ITM, half of the seeds are positive pairs and half are negative pairs. A negative pair combines an image with the caption of a random image that shares no MSCOCO object category with it, and is kept only if the judge, shown the pair without the label, judges it as not matching.

Sampling continues until each task–model pair has 100 admitted seeds.

## 5<sub>.</sub>3 B<sub>ase</sub>li<sub>nes</sub>

Table 2 lists the compared methods. All methods have the same budget (Section 5.5), and all except RND and MetaRA-c apply the relation filters.

Single-modality methods. SINVAD-VQ adapts SINVAD [29] from a VAE to the VQGAN. Its genetic search mutates the continuous encoder output before quantization and minimizes the model’s score for the seed answer. A variant rejected by the filters receives fitness +∞ without a call, and every evaluated image is a test. TextFooler [28] ranks the words of $t _ { 0 }$ by how much deleting them lowers the score for the seed answer and then tries synonym substitutions word by word; every tried substitution is a test. It does not stop at the first success but continues until all words are processed or the budget is spent. Both baselines use output scores, which VLTest does not: the matching score for ITM, the class probability for VR, and the log-probability of the seed answer for VQA, where generating an answer and scoring the seed answer are separate calls. VLTest-V and VLTest-T test the first � filtered image or text variants of our generators.

Joint methods. All joint methods run Algorithm 1 with the same � and $n _ { J }$ . RND pairs unfiltered pools of our generators in random order. CRND pairs the filtered pools of VLTest in random order. SR first pairs, in random order, the variants whose single-modality inputs have compatible outputs, so it difers from VLTest (SF) only in the order within these pairs. EXT runs SINVAD-VQ and TextFooler for � calls each as its single-modality phase, uses the variants they test as its pools, and pairs them in the stable-first order of VL-Test; it difers from VLTest (SF) only in the variants it combines. MetaRA-c applies the relations of MetaRA [72] that can be instanti ated as seed-relative transformations on VQA: its single-modality phase tests the image and question transformations, and its joint phase combines them. We exclude CIT4DNN [12] and Mimicry [70], which require a low-dimensional or class-conditional latent. We also do not compare with multimodal adversarial attacks such as Co-Attack [79], SGA [40], and VLATTACK [73]. On the image side, they optimize pixel-level perturbations against a surrogate model to maximize attack success under a perturbation bound, and they assume rather than check that the expected answer is preserved. Their confirmed failures would therefore measure attack strength rather than the ability to generate semantically varied test inputs. On the text side, the word-level substitutions such attacks use are represented by TextFooler.

## 5.4 Metrics

The oracle judges every candidate. For each task, model, and method, we report:

$\mathrm { C F } @ 1 \mathrm { k } = 1 0 0 0 N _ { \mathrm { C F } } / ( B | S | )$ , the confirmed single-modality fail ures per 1,000 calls ofthe budget, where $N _ { \mathrm { C F } }$ counts the confirmed failures of one modality over the admitted seeds $S ;$ a method that stops early is still divided by the full budget;

$\mathrm { A J F } @ 1 \mathrm { k } \ = \ 1 0 0 0 N _ { \mathrm { A J F } } / ( B | S | )$ , the attributed joint failures per 1,000 calls of the joint budget;

• budget per AJF, 1000/AJF@1k averaged over the task–model pairs; for a method that stops early, we also give the calls it actually makes per attributed joint failure;

• budget used, the calls made as a share of the budget, in the singlemodality or the joint phase;

• tested stable pairs, the share of joint calls on pairs whose imageonly and text-only inputs both have compatible outputs;

• SeedCov, the share of admitted seeds with at least one confirmed failure, computed separately for single-modality failures and attributed joint failures;

• VarCov, the mean number of distinct image and text variants per seed that take part in at least one attributed joint failure;

• confirmation rate, the share ofcandidates that the oracle confirms as failures;

• interaction share, the share of attributed joint failures that are interaction failures (Section 3.4).

For ITM and VR, � is the diference between the logits of the two labels. For $\mathrm { { V Q A , } }$ it uses the log-probabilities of the answers under teacher forcing, averaged over their tokens. For the five classification models, $\eta ( \boldsymbol { x } , t ) \le 0$ holds by construction, because $\hat { y }$ is their highest-scoring answer. PaliGemma generates its answers, so we report how many of its attributed joint failures have $\eta ( \boldsymbol { x } , t ) > 0$ and exclude them from the margin analysis.

Filters and reconstruction. We measure the precision and recall of each filter against the answer check of the oracle on 1,000 unfiltered variants per task and modality, sampled at random from the generator outputs. For reconstruction, we report the share of sampled seeds whose reconstruction changes the expected answer, which are not admitted, and the share of admitted seeds on which the model output difers between the raw and the reconstructed seed.

## 5.5 Settin<sub>g</sub>s and Im<sub>p</sub>lementation

Budgets. The primary experiments use a budget of $B \ = \ 1 { , } 0 0 0$ calls to the model under test. A single-modality method makes � calls on inputs of its modality, including the calls a search method makes to score its options. A joint method makes � single-modality calls, $n = 5 0 0$ per modality, and then � joint calls, so $n _ { J } = B$ in Algorithm 1. Single-modality failures are thus always counted over � single-modality calls and attributed joint failures over � joint calls. A method whose pool or search space is exhausted stops early. Cost accounting. Every call to the model under test is logged with its run, its role, and its key. A run applies one method to one seed and one model; a key hashes the model input and the request type. Within a run, each key is called once and its result is reused, so a single-modality input that also serves as four-cell evidence is charged once. A run is charged for every call it issues, even if another run issued the same input. Calls to the raw seed and the seed cell, the repeat calls $C _ { \mathrm { r e p } }$ of stochastic models, and the calls of the judge are not charged to �.

Parameters. The VQGAN is the pretrained Gumbel-quantized model of Taming Transformers [17] with downsampling factor 8 and $K = 8 , 1 9 2$ codebook entries. Images are resized to $2 5 6 \times 2 5 6$ pixels, which gives $\phantom { - } 1 3 2 \times 3 2$ index grid. The visual generator uses $\kappa = 1 5$ and the full rank range $R = 8 , 1 9 1$ ; the text generator uses � = $\tau = \delta = 5$ and keeps $\beta = 4 , 0 0 0$ variants. The filters use $\theta _ { \mathrm { n l i } } = 0 . 8 0$ $\rho _ { \mathrm { i m g } } = 0 . 9 0 , \rho = 0 . 9 5 ,$ , and $\mu = 0 . 0 5$ . SINVAD-VQ uses the population size and genetic operators of the original implementation, and TextFooler its default synonym sets and thresholds. The stochastic methods (SINVAD-VQ, RND, CRND, SR, and EXT) run five times on the local models; outputs of deterministic models are cached across runs.

T<sub>a</sub>bl<sub>e</sub> 2<sub>:</sub> C<sub>ompare</sub>d <sub>me</sub>th<sub>o</sub>d<sub>s.</sub> $\phi \colon$ <sub>re</sub>l<sub>a</sub>ti<sub>on</sub> filt<sub>ers</sub> <sub>app</sub>li<sub>e</sub>d<sub>.</sub> Th<sub>e</sub> fi<sub>rs</sub>t f<sub>our</sub> <sub>me</sub>th<sub>o</sub>d<sub>s</sub> t<sub>es</sub>t <sub>one</sub> <sub>mo</sub>d<sub>a</sub>lit<sub>y</sub> <sub>on</sub>l<sub>y.</sub>
<table><tr><td>Method</td><td>Image side</td><td>Text side</td><td> $\phi$ </td><td>Joint inputs</td></tr><tr><td>SINVAD-VQ</td><td>latent search (scores)</td><td></td><td>yes</td><td></td></tr><tr><td>VLTEST-V</td><td>codebook replacement</td><td></td><td>yes</td><td></td></tr><tr><td>TextFooler</td><td></td><td>synonym search (scores)</td><td>yes</td><td>=</td></tr><tr><td>VLTEST-T</td><td></td><td>rewrite-expand-prune</td><td>yes</td><td>=</td></tr><tr><td>RND</td><td>VLTEST-V</td><td>VLTEST-T</td><td>no</td><td>random</td></tr><tr><td>CRND</td><td>VLTEST-V</td><td>VLTEST-T</td><td>yes</td><td>random</td></tr><tr><td>SR</td><td>VLTEST-V</td><td>VLTEST-T</td><td>yes</td><td>random, stable first</td></tr><tr><td>EXT</td><td>SINVAD-VQ</td><td>TextFooler</td><td>yes</td><td>stable-first</td></tr><tr><td>MetaRA-c (VQA)</td><td>relation-defined</td><td>relation-defined</td><td></td><td>relation-defined</td></tr><tr><td>VLTEST (RR)</td><td>VLTEST-V</td><td>VLTEST-T</td><td>yes</td><td>round-robin</td></tr><tr><td>VLTEST (SF)</td><td>VLTEST-V</td><td>VLTEST-T</td><td>yes</td><td>stable-first</td></tr></table>

Oracle. The judge � is Llama-4-Maverick [45], from a family to which none of the models under test belongs. Its prompts and the filter thresholds were fixed on a pilot with 20 seeds per task, which is excluded from the evaluation.

Statistics. The unit of analysis is the source image, or the image pair for NLVR2. We compare methods by paired diferences with 95% percentile intervals from a paired cluster bootstrap over the 100 source images of a task–model pair, with 10,000 resamples [16]; for a stochastic method, the five runs are averaged per seed before resampling. Two-sided �-values are min{1, 2 min(�<sup>−</sup>, �<sup>+</sup>)}, where �<sup>−</sup> and $p ^ { + }$ are the shares of resampled diferences at most and at least zero. Holm’s procedure [24] corrects them within Table 5, which marks the significant diferences.

Implementation. The task-specific models use their fine-tuned checkpoints, and Qwen3-VL uses its public Hugging Face checkpoints. All local models run in bf16, with greedy decoding for generative outputs. $F _ { \mathrm { t x t } }$ uses roberta-large-mnli, and $F _ { \mathrm { i m g } }$ and $F _ { \mathrm { p a i r } }$ use OpenCLIP ViT-bigG/14 trained on LAION-2B. Experiments ran on Ubuntu 24.04.3 LTS with an AMD Ryzen Threadripper PRO 7985WX CPU, 502 GiB of DDR5 memory, and two NVIDIA RTX 6000 Ada GPUs.

## 6 R<sub>esu</sub>lt<sub>s</sub>

## 6.1 RQ1: Validity

We run the ten methods of Table 2 on the six task–model pairs and MetaRA-c on the two VQA pairs, with 100 admitted seeds per pair and $B = 1 , 0 0 0$ . The five stochastic methods run five times, which gives 30 runs per seed (31 on VQA) and 18,200 runs in total.

6.1.1 Candidate Outcomes. Depending on the pair, 61.0% (ViLT on VR) to 84.0% (PaliGemma) of the sampled seeds are admitted. Table 3 breaks down the candidates of VLTest (SF). The oracle confirms 121,235 of the 148,314 candidates (81.7%) as failures, from 76.7% on ViLT for VR to 84.3% on ViLT for ITM. Of the remaining candidates, 21,961 (14.8%) have a changed expected answer, 3,065 (2.1%) are invalid or abstained, and 2,053 (1.4%) have outputs that the oracle judges correct despite being outside the reference-answer set.

On every pair, image-only candidates have the highest confirmation rate (86.3% to 91.6%), text-only candidates are 4.8 to 10.0 percentage points lower, and joint inputs are lowest (75.0% to 83.0%). The joint column judges the joint input alone; Section 6.3 adds the single-modality controls. The text gap is largest on VR, where textonly candidates reach 77.0% and 77.1%: VR statements often hinge on counts and quantifiers, and a rewrite that changes one of them changes the label. On ITM and VR the answer is binary, so an output outside $A _ { 0 }$ is wrong unless the expected answer changed. On $\mathrm { { V Q A } , }$ , 2,053 candidates (3.2%) have an output outside $A _ { 0 }$ that is still correct, such as a synonym of a reference answer, and 6.4% of the output changes stay within � and are not candidates. A count of output changes reports both as failures.

6.1.2 Oracle Precision. For each method, we sample reported failures stratified by task, input type (image-only, text-only, joint), and verdict (confirmed failure, and for joint candidates also attributed joint failure), pooling the two models of a task. We label up to 40 failures per stratum for VLTest-V, VLTest-T, CRND, SR, VLTest (RR), and VLTest (SF), and up to 20 for the other methods. We also label 20 rejected joint candidates of VLTest (SF) for each task and reason of rejection, 200 in total, and 30 reconstructed seeds per task.

Nine graduate students in computer science who are not authors label these inputs. Two of them label each input independently, and a third resolves disagreements; an input on which no two agree counts as rejected. The annotators follow the two checks of the oracle: they decide the expected answer before they see the model output, and they see neither the oracle verdict nor the method. Before adjudication, Krippendorf’s � is 0.82 on the answer check and 0.88 on the output check. Precision is the share of sampled failures that the annotators agree with, with a 95% Clopper–Pearson interval per stratum. A confirmed-failure estimate that spans several input types weights the stratum precisions, and their interval endpoints, by the numbers of reported failures in the strata.

Table 4 lists the precision of every method; each AJF cell rests on 20 to 39 labeled failures. Precision ranges from 85.0% to 93.9% for confirmed failures and from 80.0% to 94.6% for attributed joint failures; for VLTest (SF), it ranges from 88.8% to 94.6%. The lowest value, 80.0% for attributed joint failures on VR, belongs to RND and EXT. The oracle errors found by the annotators are missed changes of the expected answer, correct answers in other wording judged incorrect, and wrong attribution or ambiguity in joint failures. On the 90 reconstructed seeds, the answer check agrees with the annotators in 95.6% of the cases, and the annotators agree with 85.0% to 95.0% of the rejections in every stratum.

Table 3: RQ1: oracle outcomes of the candidates of VLTest (SF), and the confirmation rate per input type.
<table><tr><td colspan="10"></td></tr><tr><td>Task</td><td>Model</td><td>Candidates</td><td>Invalid / abstain</td><td>Answer changed</td><td>Output correct</td><td>Confirmed failure</td><td>image- only</td><td>text- only</td><td>joint input</td></tr><tr><td>ITM</td><td>BLIP</td><td>22,776</td><td>308</td><td>3,399</td><td>0</td><td>19,069</td><td>91.6</td><td>86.8</td><td>82.0</td></tr><tr><td>ITM</td><td>ViLT</td><td>25,253</td><td>356</td><td>3,608</td><td>0</td><td>21,289</td><td>90.7</td><td>85.4</td><td>83.0</td></tr><tr><td>VR</td><td>BEiT-3</td><td>19,164</td><td>514</td><td>3,686</td><td>00</td><td>14,964</td><td>87.0</td><td>77.0</td><td>76.7</td></tr><tr><td>VR</td><td>ViLT</td><td>17,179</td><td>482</td><td>3,524</td><td></td><td>13,173</td><td>86.3</td><td>77.1</td><td>75.0</td></tr><tr><td>VQA</td><td>PaliGemma</td><td>33,698</td><td>724</td><td>3,953</td><td>1,091</td><td>27,930</td><td>89.3</td><td>83.2</td><td>81.7</td></tr><tr><td>VQA</td><td>BEiT-3</td><td>30,244</td><td>681</td><td>3,791</td><td>962</td><td>24,810</td><td>88.8</td><td>82.7</td><td>80.7</td></tr></table>

6.1.3 Filters and Reconstruction. Of the 6,000 unfiltered variants, 1,000 per task and modality, $F _ { \mathrm { t x t } }$ and $F _ { \mathrm { i m g } }$ are evaluated on the 3,000 text and the 3,000 image variants, and $F _ { \mathrm { p a i r } }$ on the 3,000 variants of ITM and of true VR statements, to which it applies. Before filtering, 78.8% of the text variants and 86.5% of the image variants preserve the expected answer. $F _ { \mathrm { t x t } } , F _ { \mathrm { i m g } } ,$ , and $F _ { \mathrm { p a i r } }$ keep 70.8%, 82.7%, and 74.6% of the variants, with a precision of 93.1%, 95.4%, and 91.7% and a recall of 83.6%, 91.2%, and 86.0% for answer preservation. Each filter thus removes 70% to 77% of the variants whose expected answer changes and loses 9% to 16% of those that keep it. The answer check rejects the changed answers that pass the filters (Table 3).

Reconstruction changes the expected answer of 2% to 6% of the sampled seeds, which are not admitted. On the admitted seeds, the model output on the reconstructed seed difers from that on the raw seed for 4% to 9%, depending on the pair.

## 6.2 RQ2: Testing Utility and Cost

6.2.1 Single-Modality Failures. Table 5 lists CF@1k and SeedCov for the single-modality methods. VLTest-V finds 1.8 to 2.1 times as many confirmed failures as SINVAD-VQ, and VLTest-T 2.3 to 2.5 times as many as TextFooler, without the output scores that both baselines use. Our generators also cover more seeds, 75.7% and 71.0% against 63.4% and 56.7%. All diferences are significant. SINVAD-VQ and TextFooler stop early on some seeds and use 88.9% and 81.7% of the budget; per call actually made, their aggregate rates rise by 12% and 22%, and the gap remains. For all four methods, failures are most frequent on VQA and least frequent on VR.

6.2.2 AtributedJointFailures. Table 6 reports AJF@1k for the joint methods. VLTest (SF) finds 91.4 (ViLT on VR) to 196.8 (PaliGemma) attributed joint failures per 1,000 joint calls, one per 7.6 joint calls on average, and finds at least one on 98.2% of the seeds. All joint methods except MetaRA-c use the full joint budget.

The joint methods difer one component at a time. Applying the relation filters substantially improves the yield: CRND and VLTest (RR) find 1.6 to 1.9 times as many attributed joint failures as RND. Stable-first ordering raises the share of joint calls on stable pairs from 85.2% (CRND and VLTest (RR)) to 100% (SR and VLTest (SF)). It increases AJF@1k by 12% to 24% from CRND to SR and by 14% to 27% from VLTest (RR) to VLTest (SF) across the six task–model pairs; averaged over the pairs, AJF@1k rises from 118.9 to 137.6 and from 121.9 to 142.3, and the budget per attributed joint failure falls from 9.1 to 7.8 and from 8.8 to 7.6 calls. This gain follows from the definition of an attributed joint failure: a pair with an incompatible single-modality output cannot yield one on ITM and VR, and rarely on VQA. If every stable pair were equally likely to yield an attributed joint failure, moving the remaining 14.8% of the joint calls to stable pairs would raise AJF@1k by about 17% (1/0.852), close to the observed average gains of 16% and 17%. Stable-first ordering therefore does not make a stable pair more likely to fail; it spends the joint budget only on pairs that can yield an attributed joint failure, at no additional cost.

Within the same priority class, round-robin ordering changes AJF@1k by only 1% to 5% (VLTest (RR) versus CRND, and VLTest (SF) versus SR). It combines every variant of the pools equally often, whereas random ordering leaves some variants untested within the joint budget. The attributed joint failures of round-robin ordering involve more distinct variants (VarCov 97.4 / 95.1 against 83.1 / 81.3 for CRND, and 105.6 / 102.2 against 84.6 / 82.2 for SR) and cover more seeds (SeedCov 94.8% against 92.7%, and 98.2% against 93.7%). EXT applies the same stable-first pairing strategy to variants produced by SINVAD-VQ and TextFooler. VLTest (SF) finds 1.26 to 1.33 times as many attributed joint failures as EXT across the six task–model pairs, while its failures involve more than three times as many distinct variants per seed (VarCov 105.6 / 102.2 versus 32.1 / 29.8). On the two VQA pairs, VLTest (SF) finds 2.8 times as many attributed joint failures as MetaRA-c under the full-budget metric. MetaRA-c exhausts the combinations of its relations after using 51.4% of the joint budget on average. When normalized by the calls it actually makes, MetaRA-c finds about 128 attributed joint failures per 1,000 joint calls, or one per 7.8 calls, compared with about 185 per 1,000 calls, or one per 5.4 calls, for VLTest (SF).

## 6.3 RQ3: Joint Failures

## RQ3 analyzes the joint candidates of the RQ1 runs.

6.3.1 Four-Cell Evidence. Table 7 shows why the joint candidates of VLTest (SF) do or do not become attributed joint failures. Of its 110,152 joint candidates, 85,390 (77.5%) are attributed joint failures, and four-cell confirmation rejects 24,762 (22.5%).

Table 7 attributes each rejection to the first check that fails, starting with the joint input. Checking the joint input alone rejects 19.7% of the joint candidates: its expected answer changes, it is invalid or the judge abstains, or, on VQA, its output is a correct answer outside $A _ { 0 } .$ The two single-modality controls reject a further 2.8% of joint candidates. Among all joint candidates, 3,101 (2.8%) exhibit conflicting oracle verdicts on at least one control: the output is compatible with the reference answers and the answer check returns preserved, but the output check judges it incorrect. Such conflicting verdicts prevent a candidate from becoming an attributed joint failure. The annotators agree with 85.0% to 90.0% of the sampled rejections of this type. For 9,183 candidates (8.3%), the joint input changes the expected answer while both single-modality inputs keep it; no check of a single-modality input can detect this. VR has the lowest share of attributed joint failures (72.2%; 73.1% on BEiT-3 and 71.2% on ViLT): a single-modality input changes the expected answer for 11.7% and 12.9% of its joint candidates, against 5.8% to 6.5% on the other pairs.

Table 4: RQ1: oracle precision in percent with 95% intervals, estimated from samples drawn separately for each method and pooled over the two models of each task. CF: confirmed failures; AJF: attributed joint failures. CF precision is shown for the single-modality methods and VLTest (SF).
<table><tr><td></td><td colspan="3">CF</td><td colspan="3">AJF</td></tr><tr><td>Method</td><td>ITM</td><td>VR</td><td>VQA</td><td>ITM</td><td>VR</td><td>VQA</td></tr><tr><td>SINVAD-VQ</td><td>90.0 [68.3,98.8]</td><td>85.0 [62.1,96.8]</td><td>90.0 [68.3,98.8]</td><td></td><td></td><td></td></tr><tr><td>VLTEST-V</td><td>92.5 [79.6,98.4]</td><td>87.5 [73.2,95.8]</td><td>92.5 [79.6,98.4]</td><td></td><td></td><td></td></tr><tr><td>TextFooler</td><td>85.0 [62.1,96.8]</td><td>85.0 [62.1,96.8]</td><td>90.0 [68.3,98.8]</td><td></td><td></td><td></td></tr><tr><td>VLTEST-T</td><td>90.0 [76.3,97.2]</td><td>87.5 [73.2,95.8]</td><td>92.5 [79.6,98.4]</td><td></td><td></td><td></td></tr><tr><td>RND</td><td></td><td></td><td></td><td>85.0 [62.1,96.8]</td><td>80.0 [56.3,94.3]</td><td>85.0 [62.1,96.8]</td></tr><tr><td>CRND</td><td></td><td></td><td></td><td>87.9 [71.8,96.6]</td><td>85.7 [69.7,95.2]</td><td>90.3 [74.2,98.0]</td></tr><tr><td>SR</td><td></td><td></td><td></td><td>91.2 [76.3,98.1]</td><td>86.8 [71.9,95.6]</td><td>91.9 [78.1,98.3]</td></tr><tr><td>EXT</td><td></td><td></td><td></td><td>85.0 [62.1,96.8]</td><td>80.0 [56.3,94.3]</td><td>85.0 [62.1,96.8]</td></tr><tr><td>MetaRA-c</td><td></td><td></td><td></td><td></td><td></td><td>85.0 [62.1,96.8]</td></tr><tr><td>VLTEST (RR)</td><td></td><td></td><td></td><td>91.2 [76.3,98.1]</td><td>89.2 [74.6,97.0]</td><td>91.2 [76.3,98.1]</td></tr><tr><td>VLTEST (SF)</td><td>92.0 [78.8,98.2]</td><td>88.8 [74.5,96.6]</td><td>93.9 [81.0,99.0]</td><td>92.3 [79.1,98.4]</td><td>89.5 [75.2,97.1]</td><td>94.6 [81.8,99.3]</td></tr></table>

Table 5: RQ2: CF@1k per task–model pair; SeedCov and budget used in percent, averaged over the six pairs. <sup>∗</sup>: significant diference to VLTest-V or VLTest-T (5% level, Holm).
<table><tr><td></td><td colspan="2">ITM</td><td colspan="2">VR</td><td colspan="2">VQA</td><td colspan="2"></td></tr><tr><td>Method</td><td>BLIP</td><td>ViLT</td><td>BEiT-3</td><td>ViLT</td><td>PaliGemma</td><td>BEiT-3</td><td>SeedCov (%)</td><td>Budget used (%)</td></tr><tr><td>SINVAD-VQ</td><td>24.1*</td><td>26.8*</td><td>21.6*</td><td>18.9*</td><td>42.7*</td><td>37.5*</td><td>63.4</td><td>88.9</td></tr><tr><td>VLTEST-V</td><td>50.3</td><td>56.4</td><td>42.7</td><td>37.2</td><td>78.5</td><td>71.4</td><td>75.7</td><td>100.0</td></tr><tr><td>TextFooler</td><td>19.4*</td><td>21.8*</td><td>15.8*</td><td>13.9*</td><td>30.6*</td><td>27.5*</td><td>56.7</td><td>81.7</td></tr><tr><td>VLTEST-T</td><td>46.2</td><td>50.1</td><td>38.4</td><td>34.1</td><td>74.3</td><td>68.0</td><td>71.0</td><td>100.0</td></tr></table>

Table 6: RQ2: AJF@1k per task–model pair; budget used, budget per AJF, VarCov (image / text), SeedCov, and the share ofjoint <sub>ca</sub>ll<sub>s on s</sub>t<sub>a</sub>bl<sub>e pa</sub>i<sub>rs, average</sub>d <sub>over</sub> th<sub>e pa</sub>i<sub>rs a me</sub>th<sub>o</sub>d <sub>runs on.</sub>
<table><tr><td></td><td colspan="2">ITM</td><td colspan="2">VR</td><td colspan="2">VQA</td><td colspan="5"></td></tr><tr><td>Method</td><td>BLIP</td><td>ViLT</td><td>BEiT-3</td><td>ViLT</td><td>PaliGemma</td><td>BEiT-3</td><td>Budget used (%)</td><td>Budget per AJF</td><td>VarCov (I/T)</td><td>SeedCov (%)</td><td>Tested stable pairs (%)</td></tr><tr><td>RND</td><td>65.9</td><td>72.3</td><td>48.4</td><td>41.7</td><td>101.6</td><td>89.4</td><td>100.0</td><td>15.8</td><td>34.6 / 33.1</td><td>84.7</td><td>73.5</td></tr><tr><td>CRND</td><td>116.8</td><td>117.9</td><td>86.1</td><td>75.1</td><td>168.4</td><td>149.1</td><td>100.0</td><td>9.1</td><td>83.1 / 81.3</td><td>92.7</td><td>85.2</td></tr><tr><td>SR</td><td>133.1</td><td>146.0</td><td>99.2</td><td>87.8</td><td>192.4</td><td>167.1</td><td>100.0</td><td>7.8</td><td>84.6 / 82.2</td><td>93.7</td><td>100.0</td></tr><tr><td>EXT</td><td>105.3</td><td>118.1</td><td>78.6</td><td>68.7</td><td>155.9</td><td>137.2</td><td>100.0</td><td>9.8</td><td>32.1 / 29.8</td><td>92.9</td><td>99.2</td></tr><tr><td>MetaRA-c</td><td>=</td><td></td><td>1</td><td>1</td><td>69.7</td><td>61.9</td><td>51.4</td><td>15.3</td><td>18.2 / 16.3</td><td>90.5</td><td>68.9</td></tr><tr><td>VLTEST (RR)</td><td>120.5</td><td>119.4</td><td>89.9</td><td>77.6</td><td>171.3</td><td>152.4</td><td>100.0</td><td>8.8</td><td>97.4 / 95.1</td><td>94.8</td><td>85.2</td></tr><tr><td>VLTEST (SF)</td><td>137.2</td><td>151.6</td><td>103.7</td><td>91.4</td><td>196.8</td><td>173.2</td><td>100.0</td><td>7.6</td><td>105.6 / 102.2</td><td>98.2</td><td>100.0</td></tr></table>

6.3.2 Interaction and Additive Failures. We apply the margin analysis of Section 3.4 to the attributed joint failures of VLTest (SF) (Table 8). On PaliGemma, 622 of the 19,683 attributed joint failures (3.2%) have � (�, �) > 0 under teacher forcing, so the sequence score does not reproduce the decoded output; we exclude them. Of the remaining 84,768, 13,999 (16.5%) are interaction failures: the additive prediction still prefers a reference answer, but the joint input fails. Their share ranges from 13.8% on PaliGemma to 22.6% on ViLT for VR and is highest on VR. The other 83.5% are additive. The median $\Delta _ { \mathrm { i n t } }$ is positive on every pair, so the observed joint margin is usually less negative than the additive prediction.

6.3.3 Failure Types. Two authors open-coded a separate pilot of 30 attributed joint failures and then independently assigned one primary type to each attributed joint failure in a stratified random sample of 180, 60 per task (Cohen’s � = 0.79 before adjudication). Table 9 lists the five types. Attribute or entity binding and spatial or relational composition account for 98 of the 180 failures. Quantifier, count, and scope errors concentrate on VR (26 of 34).

Figure 2 shows two attributed joint failures with their four cells. In the ITM case, the image variant changes the shade and texture of the court, and the text variant rewrites “Two women play tennis on a blue court” as “Two females are playing tennis on a blue court”. The model predicts a match for the seed and both single-modality inputs and no match for the joint input, although the caption stil describes the image. In the VQA case, the image variant shifts the shading of the donuts, and the text variant rewrites “What color are the donuts?” as “Which color do these donuts have?”. The model answers “brown” for the seed and both single-modality inputs and “yellow” for the joint input. Single-modality testing passes both cases.

Table 7: RQ3: joint candidates of VLTest (SF) by the first check that fails, starting with the joint input, in percent of the joint <sub>can</sub>did<sub>a</sub>t<sub>es,</sub> <sub>poo</sub>l<sub>e</sub>d <sub>over</sub> th<sub>e</sub> t<sub>wo</sub> <sub>mo</sub>d<sub>e</sub>l<sub>s</sub> <sub>o</sub>f <sub>eac</sub>h t<sub>as</sub>k<sub>.</sub>
<table><tr><td>Task</td><td>Rejected by the joint input (%)</td><td>Further rejected by the controls (%)</td><td>Attributed joint failures (%)</td></tr><tr><td>ITM</td><td>17.5</td><td>3.5</td><td>80.0</td></tr><tr><td>VR</td><td>24.1</td><td></td><td>72.2</td></tr><tr><td>VQA</td><td>18.8</td><td>2.6</td><td>78.6</td></tr><tr><td>All</td><td>19.7</td><td>2.8</td><td>77.5</td></tr></table>

Table 8: RQ3: attributed joint failures of VLTest (SF) classified by the margin analysis. Interaction: share of interaction failures; $\Delta _ { \mathrm { { i n t } } } \mathrm { { : } }$ <sub>me</sub>di<sub>an w</sub>ith i<sub>n</sub>t<sub>er uar</sub>til<sub>e ran e.</sub> B<sub>o</sub>th <sub>are com u</sub>t<sub>e</sub>d <sub>over</sub> th<sub>e score-c</sub>l<sub>ass</sub>ifi<sub>a</sub>bl<sub>e</sub> f<sub>a</sub>il<sub>ures, w</sub>hi<sub>c</sub>h <sub>exc</sub>l<sub>u</sub>d<sub>e</sub> th<sub>e</sub> 622 f<sub>a</sub>il<sub>ures o</sub>f PaliGemma with (�, �) > 0 (84 768 in all)
<table><tr><td>Task</td><td>Model</td><td>Attributed joint failures</td><td>Interaction (%)</td><td>Median  $\Delta _ { \mathrm { i n t } }$  [IQR]</td></tr><tr><td>ITM</td><td>BLIP</td><td>13,716</td><td>16.4</td><td>0.121 [0.051, 0.238]</td></tr><tr><td>ITM</td><td>ViLT</td><td>15,163</td><td>14.7</td><td>0.109 [0.043, 0.219]</td></tr><tr><td>VR</td><td>BEiT-3</td><td>10,368</td><td>21.3</td><td>0.154 [0.062, 0.298]</td></tr><tr><td>VR</td><td>ViLT</td><td>9,139</td><td>22.6</td><td>0.163 [0.038, 0.312]</td></tr><tr><td>VQA</td><td>PaliGemma</td><td>19,683</td><td>13.8</td><td>0.097 [0.038, 0.194]</td></tr><tr><td>VQA</td><td>BEiT-3</td><td>17,321</td><td>15.1</td><td>0.113 [0.045, 0.226]</td></tr><tr><td colspan="2">All</td><td>85,390</td><td>16.5</td><td>0.124 [0.049, 0.248]</td></tr></table>

## 6.4 RQ4: Sensitivity and Generalization

6.4.1 Design Choices. We vary one design choice of VLTest (SF) at a time on BLIP (ITM), BEiT-3 (VR), and PaliGemma (VQA), with the same 100 seeds, budget, and other settings: (a) the relation filters, with no filter, each filter alone, or all filters; (b) the pool size $n \in \{ 2 5 0 , 5 0 0 , 7 5 0 \}$ , which sets the single-modality calls to 2� while the joint calls stay at �; (c) the rank range $R \in \left\{ 2 , 0 4 7 , 4 , 0 9 5 , 8 , 1 9 1 \right\}$ ; (d) the number of edited indices � ∈ {5, 10, 15, 20, 30}; and (e) the text pipeline with rewrite, expand, or prune removed. These are 16 settings and 4,800 runs.

Table 10 reports averages over the three pairs; CF@1k is computed over the single-modality calls of each setting, and the default appears once in every group. The default reaches a confirmation rate of 81.6%, 55.1 CF@1k, and 145.9 AJF@1k. Without filters, VL-Test (SF) finds 25% fewer attributed joint failures (108.7) and its confirmation rate drops to 71.8%. Among the single filters, $F _ { \mathrm { p a i r } }$ achieves the highest AJF@1k (131.4), followed by $F _ { \mathrm { t x t } }$ (123.5) and $F _ { \mathrm { i m g } }$ (120.2). However, its overall candidate confirmation rate (71.9%) remains close to that of the unfiltered setting (71.8%). These metrics capture diferent aspects of testing efectiveness: the confirmation rate measures the proportion of candidates confirmed as failures across input types, whereas AJF@1k measures attributed joint failures per 1,000 joint calls. A pool of 250 variants finds 9% fewer than the default of 500, and a pool of 750 about as many (147.0). Narrowing the rank range reduces AJF@1k to 136.7 at � = 4,095 and 121.4 at $R = 2 { , } 0 4 7$ , while the confirmation rate remains between 80.8% and 81.3%. AJF@1k peaks at � = 15 and 20 (145.9 and 146.6) and drops at � = 30 (140.7), where CF@1k also falls to 51.8. Removing any text stage lowers the confirmation rate to 73.9% to 79.6% and AJF@1k by 9% to 13%; removing the prune stage lowers CF@1k the most, to 47.5.

6.4.2 General-Purpose Models. We also test four general-purpose VLMs on all three tasks: Qwen3-VL-8B-Instruct and Qwen3-VL-32B-Instruct [49], one open-weight family at two scales, and two commercial models, GPT-4o mini [47] in its snapshot of 18 July 2024 and Gemini 2.5 Flash [10]. On each model, we run VLTest (SF), SR, and CRND. The Qwen3-VL models use the settings of Section 5.5 with 100 seeds per task. The API models use 50 sampled seeds per task, � = 50, and � = 25; Table 11 lists the admitted seeds |�|. For ITM and VR, the prompts and the rules for extracting a label are fixed in advance, and an output that maps to no label counts as an abstention. The API models are called at temperature 0 where the API allows it, and every call is logged with its date. The API experiments use 37,650 single-modality calls and 37,650 joint calls for failure discovery.

Table 11 shows the results. VLTest (SF) finds 111.8 to 249.8 attributed joint failures per 1,000 joint calls, and its rate exceeds that of SR (by 5.6 to 24.1) and CRND (by 21.6 to 41.3) on all twelve task–model pairs. On the API models, the diferences rest on few failures: the smallest, 5.6 for GPT-4o mini on ITM, corresponds to about 12 attributed joint failures. As on the task-specific models, VQA yields the most attributed joint failures and VR the fewest. Qwen3-VL-32B yields 19.8% to 29.4% fewer attributed joint failures than Qwen3-VL-8B on every task, each on its own admitted seeds. On the API models, 2.3% to 4.7% of the admitted seeds show output variation.

## 7 Di<sub>scuss</sub>i<sub>o</sub>n

## 7.1 Out<sub>p</sub>ut Chan<sub>g</sub>es Overstate Failures

Table 3 gives the share of candidates that are confirmed failures, by input type: 86.3% to 91.6% for image-only inputs, 77.0% to 86.8% for text-only inputs, and 75.0% to 83.0% for joint inputs. A count of output changes therefore overstates the failures of VLTest (SF)

Table 9: RQ3: types of attributed joint failures in a stratified random sample of 180, 60 per task.
<table><tr><td>Failure type</td><td>ITM</td><td>VR</td><td>VQA</td><td>Total</td></tr><tr><td>Attribute or entity binding</td><td>24</td><td>9</td><td>19</td><td>52</td></tr><tr><td>Spatial or relational composition</td><td>17</td><td>15</td><td>14</td><td>46</td></tr><tr><td>Quantifier, count, or scope</td><td>4</td><td>26</td><td>4</td><td>34</td></tr><tr><td>Lexical grounding or co-reference</td><td>11</td><td>64</td><td>12</td><td>29</td></tr><tr><td>Fine-grained category or part</td><td>4</td><td></td><td>11</td><td>19</td></tr><tr><td>Total</td><td>60</td><td>60</td><td>60</td><td>180</td></tr></table>

Seed $( \hat { x } _ { 0 } , t _ { 0 } )$  
![](images/7fee8ef836134c92967c9eacc15eb04fac26799bd738c27b09b73e6cc816b8de.jpg)  
Two women play tennis on a blue court.

Image-only (x′, t0)  
![](images/e2f8c3c342f1c3d9cb45ca8b4f0b2eacf5e532421d2e723c3bbd9d19819570d8.jpg)  
Two women play tennis on a blue court.

Text-only (0, t′)  
![](images/d97f9f2e31008c05c563bf09b87587d47d21a0b9d18d26857222b1f72639d582.jpg)  
Two females are playing tennis on a blue court.

Joint (x′, t′)  
![](images/da483527feeaba35c45aade54d042c8d32d72e4018f8d51a0b97ea4e8fcbf2ef.jpg)  
Two females are playing tennis on a blue court.

![](images/68fbf6fafcd373e615dae84479f5113b6eaec13e6e3c9b29e5ff0529bb5de50d.jpg)  
What color are the donuts?

![](images/9fec13c6164427109c1b814d9ee0219d79e224106fd2647ea0a2c9be975f743e.jpg)  
What color are the donuts?

![](images/e6ee56a3b81837e43ad25ff72677d4a7804e89ef7a2fb35a3b345a2430a4bc0d.jpg)  
Which color do these donuts have?

![](images/c604baf03443e274d986f95d351374e8365a8b55e1f437eb48674cb22264baf6.jpg)  
Which color do these donuts have?  
Figure 2: RQ3: two attributed joint failures, each with the reconstructed seed $( \hat { x } _ { 0 } , t _ { 0 } )$ <sub>,</sub> the ima<sub>g</sub>e-onl<sub>y</sub> in<sub>p</sub>ut<sub>,</sub> the text-onl<sub>y</sub> in<sub>p</sub>ut<sub>,</sub> and the joint input. Changed words are in bold. Model outputs from left to right: match, match, match, no match (ITM); brown brown, brown, yellow (VQA).

by 9% to 16% (relative) on image-only inputs, 15% to 30% on textonly inputs, and 20% to 33% on joint inputs; on VQA it also counts the 6.4% of output changes that stay within the reference answers. The overstatement depends on the generator: without the relation filters, the confirmation rate falls from 81.6% to 71.8% (Table 10), so a count of output changes would credit the unfiltered generators with 39% more failures than they find, against 23% for the filtered ones. Counting output changes is still useful for screening, but the efectiveness of a generator should be measured on confirmed failures.

## 7<sub>.</sub>2 Wh<sub>a</sub>t F<sub>our-</sub>C<sub>e</sub>ll E<sub>v</sub>id<sub>ence</sub> Add<sub>s</sub>

A count of joint failures without single-modality evidence mixes failures caused by the image mutation, failures caused by the text mutation, and failures caused by their combination. Four-cell confirmation rejects 22.5% of the joint candidates of VLTest (SF) (Table 7), but most of these rejections, 19.7% of the joint candidates, come from checking the joint input itself; the single-modality controls reject a further 2.8%. The main contribution of the controls is therefore attribution: for every reported joint failure, the four cells provide evidence that the model handles each mutation alone but fails on their combination. The interaction contrast then separates the 16.5% of score-classifiable attributed joint failures that are interactions between the modalities from the rest, where two single-modality efects cross the decision boundary only together (Table 8). These additive failures are still failures on valid inputs that single-modality testing passes, and the positive median $\Delta _ { \mathrm { i n t } }$ shows that, within these failures, the two efects usually cancel in part rather than reinforce each other.

Table 10: RQ4: design choices of VLTest (SF), averaged over BLIP (ITM), BEiT-3 (VR), and PaliGemma (VQA).
<table><tr><td>Design choice</td><td>Setting</td><td>Conf. rate (%)</td><td>CF@1k</td><td>AJF@1k</td></tr><tr><td>(a) relation filters</td><td>none</td><td>71.8 77.6</td><td>46.2 51.3</td><td>108.7 123.5</td></tr><tr><td></td><td> $\underline { { F } } _ { \mathrm { t x t } } \mathrm { ~ o n l y }$   ${ \underline { { F } } } _ { \mathrm { i m g } } \ { \mathrm { o n l y } }$ </td><td>76.4</td><td>50.6</td><td>120.2</td></tr><tr><td></td><td> $F _ { \mathrm { p a i r } } \mathrm { \ o n l y }$ </td><td>71.9</td><td>52.8</td><td>131.4</td></tr><tr><td></td><td>all (default)</td><td>81.6</td><td>55.1</td><td>145.9</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td>(b) pool size</td><td>n = 250</td><td>80.7</td><td>51.8</td><td>132.6</td></tr><tr><td></td><td>n = 500 (default)</td><td>81.6</td><td>55.1</td><td>145.9</td></tr><tr><td></td><td>n = 750</td><td>81.3</td><td>54.9</td><td>147.0</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td>(c) rank range (κ = 15)</td><td>R = 2,047</td><td>81.3</td><td>48.6</td><td>121.4</td></tr><tr><td></td><td>R = 4,095</td><td>80.8</td><td>52.9</td><td>136.7</td></tr><tr><td></td><td>R = 8,191 (default)</td><td>81.6</td><td>55.1</td><td>145.9</td></tr><tr><td>(d) edited indices (R = 8,191)</td><td>K= 5</td><td>80.1</td><td>47.3</td><td></td></tr><tr><td></td><td>K = 10</td><td></td><td></td><td>118.8</td></tr><tr><td></td><td>κ = 15 (default)</td><td>80.8 81.6</td><td>52.6 55.1</td><td>135.2</td></tr><tr><td></td><td>κ = 20</td><td>81.2</td><td>55.0</td><td>145.9 146.6</td></tr><tr><td></td><td> $\dot { \kappa } = \overline { { 3 0 } }$ </td><td>81.9</td><td>51.8</td><td>140.7</td></tr><tr><td></td><td>-rewrite</td><td></td><td></td><td></td></tr><tr><td>(e) text stages</td><td>-expand</td><td>73.9</td><td>51.1</td><td>126.5</td></tr><tr><td></td><td></td><td>79.6</td><td>52.7</td><td>132.9</td></tr><tr><td></td><td>-prune full (default)</td><td>75.8 81.6</td><td>47.5 55.1</td><td>128.4 145.9</td></tr></table>

Table 11: RQ4: VLTest (SF) on the general-purpose models. |�|: admitted seeds. Δ vs. SR and Δ vs. CRND: diferences in AJF@1k. N<sub>on-</sub>d<sub>e</sub>t<sub>.</sub> fl<sub>oor: s</sub>h<sub>are o</sub>f <sub>a</sub>d<sub>m</sub>itt<sub>e</sub>d <sub>see</sub>d<sub>s w</sub>ith <sub>o</sub>b<sub>serve</sub>d <sub>ou</sub>t<sub>pu</sub>t <sub>var</sub>i<sub>a</sub>ti<sub>on.</sub>
<table><tr><td>Task</td><td>Model</td><td>|S|</td><td>Non-det. floor</td><td>AJF@1k</td><td>∆ vs. SR</td><td>∆ vs. CRND</td></tr><tr><td>ITM</td><td>Qwen3-VL-8B</td><td>100</td><td></td><td>169.0</td><td>18.2</td><td>33.7</td></tr><tr><td>ITM ITM</td><td>Owen3-VL-32B GPT-40 mini</td><td>100</td><td>=</td><td>135.6</td><td>14.4</td><td>29.8</td></tr><tr><td>ITM</td><td></td><td>43</td><td>4.7%</td><td>139.5</td><td>5.6</td><td>33.0</td></tr><tr><td></td><td>Gemini 2.5 Flash</td><td>41</td><td>2.4%</td><td>127.3</td><td>20.0</td><td>27.8</td></tr><tr><td>VR</td><td>Qwen3-VL-8B</td><td>100</td><td>=</td><td>148.2</td><td>21.0</td><td>28.2</td></tr><tr><td>VR</td><td>Qwen3-VL-32B</td><td>100</td><td>1</td><td>111.8</td><td>13.2</td><td>25.0</td></tr><tr><td>VR</td><td>GPT-4o mini</td><td>38</td><td>2.6%</td><td>121.6</td><td>7.4</td><td>21.6</td></tr><tr><td>VR</td><td>Gemini 2.5 Flash</td><td>40</td><td>2.5%</td><td>125.0</td><td>17.5</td><td>24.5</td></tr><tr><td>VQA</td><td>Qwen3-VL-8B</td><td>100</td><td></td><td>249.8</td><td>19.2</td><td>37.4</td></tr><tr><td>VQA</td><td>Qwen3-VL-32B</td><td>100</td><td>一</td><td>176.4</td><td>14.8</td><td>32.2</td></tr><tr><td>VQA</td><td>GPT-40 mini</td><td>45</td><td>4.4%</td><td>172.9</td><td>12.9</td><td>41.3</td></tr><tr><td>VQA</td><td>Gemini 2.5 Flash</td><td>44</td><td>2.3%</td><td>193.6</td><td>24.1</td><td>32.7</td></tr></table>

## 7<sub>.</sub>3 I<sub>mp</sub>li<sub>ca</sub>ti<sub>ons</sub>

For VLM developers, more than half of the sampled attributed joint failures concern attribute binding or spatial composition, and quantifier and count errors cluster on VR (Table 9). Each attributed joint failure comes with its four cells and can serve as a regression test that checks the two mutations alone and together. The rates of attributed joint failures should not be read as a ranking of tasks or models: they depend on how often the oracle rejects candidates, which difers between tasks (Table 7), and on the admitted seeds, which difer between models. For testing researchers, checking the expected answer removes a fifth of the counted failures, and matched single-modality controls are needed to attribute a joint failure to the combination; both belong in the evaluation of multimodal testing studies.

## 8 Thr<sub>ea</sub>t<sub>s</sub> t<sub>o</sub> V<sub>a</sub>lidit<sub>y</sub>

Internal Validity. The oracle relies on a VLM judge, which can make mistakes. RQ1 measures the precision of the failures it reports for every method and task and checks a sample of the joint candidates of VLTest (SF) that it rejects; we do not measure how many failures it misses for the other methods. The judge receives the reference answers as a reference, which may bias its answer check toward preserved; missed answer changes are among the oracle errors that the annotators find (Section 6.1). Under the joint-inputfirst attribution of rejection reasons, the single-modality controls reject a further 2.8% of joint candidates; these rejections include cases with conflicting judge verdicts and are not evidence that the model fails on the single-modality input. All cells are built on the reconstructed seed, and seeds whose reconstruction changes the expected answer are not admitted. Admitting seeds per model favors inputs that the model handles correctly, and we report the admission rates. The filters and the judge share no model family or vision encoder with the models under test. All counts rest on the oracle verdicts, whose precision RQ1 measures. CF@1k and AJF@1k count failing inputs, not distinct faults; VarCov shows how many distinct variants the attributed joint failures involve.

External Validity. Our study covers three tasks, eight models, and one tokenizer family. All cells are 256×256 VQGAN reconstructions, which carry reconstruction artifacts; the oracle checks that each reported input is a valid input for the task, and the annotators check this on a sample. Results may difer for other tokenizers and generators, for high-resolution or text-heavy images, and for multiturn systems. The seeds come from public validation splits that the models under test and the judge may have seen during training; the generated variants are new inputs, but they stay close to these seeds. The results for the API models are tied to the model versions available during the study and to a reduced budget.

## 9 C<sub>o</sub>n<sub>c</sub>l<sub>us</sub>i<sub>o</sub>n

This paper presented VLTest, a black-box approach for testing VLMs that finds failures whose expected answers are confirmed by an oracle and attributes joint failures to the combination of image and text mutations with four-cell evidence. VLTest combines controlled visual and textual generators, relation filters, and two phase testing with feedback-directed pairing. An automated oracle confirms every reported failure, and human labels on a stratified sample measure its precision. The oracle confirms 81.7% of the candidates of VLTest (SF) and agrees with human labels on 80.0% to 94.6% of the failures it reports across the compared methods in the primary experiment, and on 88.8% to 94.6% of those reported by the default configuration of VLTest. Under the same budget, the generators of VLTest find 1.8 to 2.5 times as many confirmed failures as SINVAD and TextFooler, and VLTest finds 91 to 197 attributed joint failures per 1,000 joint calls, 1.3 times as many as the same pairing applied to the variants of these two baselines. Fourcell confirmation rejects 22.5% of the joint candidates, 19.7% through the joint input alone and 2.8% through the single-modality controls. Under an additive margin model, 16.5% of the score-classifiable attributed joint failures are interactions between the modalities; in the rest, the additive prediction alone already fails, although each single-modality input passes. The relation filters account for most of the diference between configurations, and on four generalpurpose models, including two commercial API models, VLTest (SF) finds more attributed joint failures than CRND and SR on every task.

## R<sub>e</sub>f<sub>erences</sub>

[1] Estelle Aflalo, Meng Du, Shao-Yen Tseng, Yongfei Liu, Chenfei Wu, Nan Duan, and Vasudev Lal. 2022. Vl-interpret: An interactive visualization tool for inter preting vision-language transformers. In Proceedings ofthe IEEE/CVF Conference on computer vision and pattern recognition. 21406–21415.

[2] Stanislaw Antol, Aishwarya Agrawal, Jiasen Lu, Margaret Mitchell, Dhruv Batra, C Lawrence Zitnick, and Devi Parikh. 2015. Vqa: Visual question answering. In Proceedings ofthe IEEE international conference on computer vision. 2425–2433.

[3] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. 2025. Qwen2.5-VL Technical Report. arXiv:2502.13923 [cs.CV]

[4] Oriol Barbany, Michael Huang, Xinliang Zhu, and Arnab Dhua. 2024. Leveraging large language models for multimodal search. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW). IEEE, 1201–1210.

[5] Lucas Beyer, Andreas Steiner, André Susano Pinto, Alexander Kolesnikov, Xiao Wang, Daniel Salz, Maxim Neumann, Ibrahim Alabdulmohsin, Michael Tschan nen, Emanuele Bugliarello, et al. 2024. Paligemma: A versatile 3b vlm for transfer. arXiv preprint arXiv:2407.07726 (2024).

[6] Tyler A Chang and Benjamin K Bergen. 2024. Language model behavior: A comprehensive survey. Computational Linguistics 50, 1 (2024), 293–350.

[7] Shuo Chen, Jindong Gu, Zhen Han, Yunpu Ma, Philip Torr, and Volker Tresp. 2023. Benchmarking robustness of adaptation methods on pre-trained visionlanguage models. Advances in Neural Information Processing Systems 36 (2023), 51758–51777.

[8] Tsong Yueh Chen, Fei-Ching Kuo, Huai Liu, Pak-Lok Poon, Dave Towey, T. H. Tse, and Zhi Quan Zhou. 2018. Metamorphic Testing: A Review of Challenges and Opportunities. Comput. Surveys 51, 1, Article 4 (2018), 27 pages. doi:10.1145/ 3143561

[9] Mehdi Cherti, Romain Beaumont, Ross Wightman, Mitchell Wortsman, Gabriel Ilharco, Cade Gordon, Christoph Schuhmann, Ludwig Schmidt, and Jenia Jitsev. 2023. Reproducible Scaling Laws for Contrastive Language-Image Learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2818–2829.

[10] Gheorghe Comanici et al. 2025. Gemini 2.5: Pushing the Frontier with Advanced Reasoning, Multimodality, Long Context, and Next Generation Agentic Capabilities. arXiv preprint arXiv:2507.06261 (2025).

[11] Swaroopa Dola, Matthew B. Dwyer, and Mary Lou Sofa. 2021. Distribution-Aware Testing of Neural Networks Using Generative Models. In Proceedings of the 43rd IEEE/ACM International Conference on Software Engineering (ICSE).

[12] Swaroopa Dola, Rory McDaniel, Matthew B Dwyer, and Mary Lou Sofa. 2024. Cit4dnn: Generating diverse and rare inputs for neural networks using latent space combinatorial testing. In Proceedings ofthe IEEE/ACM 46th International Conference on Software Engineering. 1–13.

[13] Bin Duan, Meiru Che, and Guowei Yang. 2026. Testing Neural Networks via Bayesian-Guided Exploration of Decision Landscapes. arXiv preprint arXiv:2606.04314 (2026).

[14] Bin Duan, Matthew B Dwyer, and Guowei Yang. 2025. Generating Realistic, Diverse, and Fault-Revealing Inputs with Latent Space Interpolation for Testing Deep Neural Networks. arXiv preprint arXiv:2503.17630 (2025).

[15] Bin Duan, Matthew B Dwyer, and Guowei Yang. 2026. Latent Anchor-Driven Test Generation for Deep Neural Networks. arXiv preprint arXiv:2606.04310 (2026).

[16] Bradley Efron and Robert J. Tibshirani. 1994. An Introduction to the Bootstrap. Chapman & Hall/CRC, New York.

[17] Patrick Esser, Robin Rombach, and Bjorn Ommer. 2021. Taming transformers for high-resolution image synthesis. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition. 12873–12883.

[18] Zheren Fu, Zhendong Mao, Yan Song, and Yongdong Zhang. 2023. Learning semantic relationship among instances for image-text matching. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 15159– 15168.

[19] Ji Gao, Jack Lanchantin, Mary Lou Sofa, and Yanjun Qi. 2018. Black-box generation of adversarial text sequences to evade deep learning classifiers. In 2018 IEEE Security and Privacy Workshops (SPW). IEEE, 50–56.

[20] Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. 2017. Making the v in vqa matter: Elevating the role of image understanding in visual question answering. In CVPR.

[21] Jianmin Guo, Yu Jiang, Yue Zhao, Quan Chen, and Jiaguang Sun. 2018. Dlfuzz: Differential fuzzing testing of deep learning systems. In Proceedings ofthe 2018 26th ACM Joint Meeting on European Software Engineering Conference and Symposium on the Foundations of Software Engineering. 739–743.

[22] Tanmay Gupta and Aniruddha Kembhavi. 2023. Visual programming: Compositional visual reasoning without training. In Proceedings ofthe IEEE/CVFconference on computer vision and pattern recognition. 14953–14962.

[23] Jack Hessel and Lillian Lee. 2020. Does my multimodal model learn cross-modal interactions? It’s harder to tell than you might think!. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP). Association for Computational Linguistics. doi:10.18653/v1/2020.emnlp-main.62

[24] Sture Holm. 1979. A Simple Sequentially Rejective Multiple Test Procedure. Scandinavian Journal ofStatistics 6, 2 (1979), 65–70.

[25] Matthew Honnibal, Ines Montani, Sofie Van Landeghem, Adriane Boyd, et al. 2020. spaCy: Industrial-strength natural language processing in python. (2020).

[26] Cheng-Yu Hsieh, Jieyu Zhang, Zixian Ma, Aniruddha Kembhavi, and Ranjay Krishna. 2023. SugarCrepe: Fixing Hackable Benchmarks for Vision-Language Compositionality. In Advances in Neural Information Processing Systems (NeurIPS), Datasets and Benchmarks Track, Vol. 36. Curran Associates, Inc.

[27] Mohit Iyyer, John Wieting, Kevin Gimpel, and Luke Zettlemoyer. 2018. Adversarial example generation with syntactically controlled paraphrase networks. arXiv preprint arXiv:1804.06059 (2018).

[28] Di Jin, Zhijing Jin, Joey Tianyi Zhou, and Peter Szolovits. 2020. Is bert really robust? a strong baseline for natural language attack on text classification and entailment. In Proceedings ofthe AAAI conference on artificial intelligence, Vol. 34. 8018–8025.

[29] Sungmin Kang, Robert Feldt, and Shin Yoo. 2020. Sinvad: Search-based image space navigation for dnn image classifier test input generation. In Proceedings of the IEEE/ACM 42nd International Conference on Software Engineering Workshops. 521–528.

[30] Wonjae Kim, Bokyung Son, and Ildoo Kim. 2021. Vilt: Vision-and-language transformer without convolution or region supervision. In International conference on

machine learning. PMLR, 5583–5594.

[31] D. Richard Kuhn, Dolores R. Wallace, and Albert M. Gallo, Jr. 2004. Software Fault Interactions and Implications for Software Testing. IEEE Transactions on Software Engineering 30, 6 (2004), 418–421. doi:10.1109/TSE.2004.24

[32] Seokhyun Lee, Sooyoung Cha, Dain Lee, and Hakjoo Oh. 2020. Efective whitebox testing of deep neural networks with adaptive neuron-selection strategy. In Proceedings of the 29th ACM SIGSOFT International Symposium on Software Testing and Analysis. 165–176.

[33] Tony Lee, Haoqin Tu, Chi Heem Wong, Wenhao Zheng, Yiyang Zhou, Yifan Mai, Josselin Somerville Roberts, Michihiro Yasunaga, Huaxiu Yao, Cihang Xie, and Percy Liang. 2024. VHELM: A Holistic Evaluation of Vision Language Models. In Advances in Neural Information Processing Systems (NeurIPS), Datasets and Benchmarks Track, Vol. 37. Curran Associates, Inc.

[34] Baiqi Li, Zhiqiu Lin, Wenxuan Peng, Jean de Dieu Nyandwi, Daniel Jiang, Zixian Ma, Simran Khanuja, Ranjay Krishna, Graham Neubig, and Deva Ramanan. 2024. NaturalBench: Evaluating Vision-Language Models on Natural Adversarial Samples. In Advances in Neural Information Processing Systems (NeurIPS), Datasets and Benchmarks Track, Vol. 37. Curran Associates, Inc.

[35] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, and Chunyuan Li. 2024. LLaVA OneVision: Easy Visual Task Transfer. arXiv:2408.03326 [cs.CV]

[36] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. 2023. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In ICML.

[37] Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. 2022. Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In ICML.

[38] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. 2014. Microsoft coco: Common objects in context. In ECCV.

[39] Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. 2019. RoBERTa: A Robustly Optimized BERT Pretraining Approach. arXiv:1907.11692 [cs.CL]

[40] Dong Lu, Zhiqiang Wang, Teng Wang, Weili Guan, Hongchang Gao, and Feng Zheng. 2023. Set-level Guidance Attack: Boosting Adversarial Transferability of Vision-Language Pre-training Models. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). arXiv:2307.14061.

[41] Siyu Lu, Mingzhe Liu, Lirong Yin, Zhengtong Yin, Xuan Liu, and Wenfeng Zheng. 2023. The multi-modal fusion in visual question answering: a review of attention mechanisms. PeerJ Computer Science 9 (2023), e1400.

[42] Muhammad Arslan Manzoor, Sarah Albarri, Ziting Xian, Zaiqiao Meng, Preslav Nakov, and Shangsong Liang. 2023. Multimodality representation learning: A survey on evolution, pretraining and its applications. ACM Transactions on Multimedia Computing, Communications and Applications 20, 3 (2023), 1–34.

[43] Maryam Maryam, Matteo Biagiola, Andrea Stocco, and Vincenzo Riccio. 2025. Benchmarking Generative AI Models for Deep Learning Test Input Generation. In Proceedings ofthe 18th IEEE International Conference on Software Testing, Verification and Validation (ICST). IEEE.

[44] Maryam Maryam, Matteo Biagiola, Paolo Tonella, and Vincenzo Riccio. 2026. DeepNaqqal: Human-Aligned Automated Validation of Test Inputs for Deep Learning. In Proceedings ofthe 19th IEEE International Conference on Software Testing, Verification and Validation (ICST). IEEE.

[45] Meta. 2025. Llama-4-Maverick-17B-128E-Instruct Model Card. Hugging Face. https://huggingface.co/meta-llama/Llama-4-Maverick-17B-128E-Instruct Ac cessed: 2026-10-21.

[46] Changhai Nie and Hareton Leung. 2011. A Survey of Combinatorial Testing. Comput. Surveys 43, 2, Article 11 (2011), 29 pages. doi:10.1145/1883612.1883618

[47] OpenAI. 2024. GPT-4o mini: Advancing Cost-Eficient Intelligence. https:// openai.com/index/gpt-4o-mini-advancing-cost-eficient-intelligence/.

[48] Kexin Pei, Yinzhi Cao, Junfeng Yang, and Suman Jana. 2017. Deepxplore: Automated whitebox testing of deep learning systems. In proceedings ofthe 26th Symposium on Operating Systems Principles. 1–18.

[49] Qwen Team. 2025. Qwen3-VL Technical Report. arXiv:2511.21631 [cs.CV]

[50] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. 2021. Learning transferable visual models from natural language supervision. In International conference on machine learning. PmLR, 8748–8763.

[51] Pooyan Rahmanzadehgervi, Logan Bolton, Mohammad Reza Taesiri, and Anh Totti Nguyen. 2024. Vision Language Models Are Blind. In Proceedings of the Asian Conference on Computer Vision (ACCV). Springer, 18–34.

[52] Nils Reimers and Iryna Gurevych. 2019. Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. In Proceedings ofthe 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP). Association for Computational Linguistics, 3982–3992. doi:10.18653/v1/D19-1410

[53] Shuhuai Ren, Yihe Deng, Kun He, and Wanxiang Che. 2019. Generating natu ral language adversarial examples through probability weighted word saliency. In Proceedings of the 57th annual meeting of the association for computational

linguistics. 1085–1097.

[54] Marco Tulio Ribeiro, Tongshuang Wu, Carlos Guestrin, and Sameer Singh. 2020. Beyond Accuracy: Behavioral Testing of NLP Models with CheckList. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (ACL). Association for Computational Linguistics, 4902–4912. doi:10.18653/v1/2020.acl-main.442

[55] Vincenzo Riccio and Paolo Tonella. 2023. When and Why Test Generators for Deep Learning Produce Invalid Inputs: an Empirical Study. In Proceedings ofthe 45th IEEE/ACM International Conference on Software Engineering (ICSE).

[56] Christian Schlarmann, Naman Deep Singh, Francesco Croce, and Matthias Hein. 2024. Robust clip: Unsupervised adversarial fine-tuning of vision embeddings for robust large vision-language models. arXiv preprint arXiv:2402.12336 (2024).

[57] Sergio Segura, Gordon Fraser, Ana B. Sanchez, and Antonio Ruiz-Cortés. 2016. A Survey on Metamorphic Testing. IEEE Transactions on Software Engineering 42, 9 (2016), 805–824. doi:10.1109/TSE.2016.2532875

[58] Mong Yuan Sim, Wei Emma Zhang, Xiang Dai, and Biaoyan Fang. 2025. Can vlms actually see and read? a survey on modality collapse in vision-language models. In Findings ofthe Association for Computational Linguistics: ACL 2025. 24452–24470.

[59] Lev Sorokin, Chen Yang, Ken E. Friedl, and Andrea Stocco. 2026. Searchbased Testing of Vision Language Models for In-Car Scene Understanding. arXiv:2607.02300 [cs.SE]

[60] Alane Suhr, Stephanie Zhou, Ally Zhang, Iris Zhang, Huajun Bai, and Yoav Artzi. 2019. A corpus for reasoning about natural language grounded in photographs. In Proceedings ofthe 57th annual meeting ofthe association for computational linguistics. 6418–6428.

[61] Youcheng Sun, Xiaowei Huang, Daniel Kroening, James Sharp, Matthew Hill, and Rob Ashmore. 2018. Testing deep neural networks. arXivpreprintarXiv:1803.04792 (2018).

[62] Tristan Thrush, Ryan Jiang, Max Bartolo, Amanpreet Singh, Adina Williams, Douwe Kiela, and Candace Ross. 2022. Winoground: Probing Vision and Language Models for Visio-Linguistic Compositionality. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 5238–5248

[63] Shengbang Tong, Erik Jones, and Jacob Steinhardt. 2023. Mass-producing failures of multimodal systems with language models. Advances in Neural Information Processing Systems 36 (2023), 29292–29322

[64] Pablo Valle, Sergio Segura, Shaukat Ali, and Aitor Arrieta. 2026. Metamorphic Testing of Vision-Language Action-Enabled Robots. In Proceedings ofthe 19th IEEE International Conference on Software Testing, Verification and Validation (ICST). IEEE. arXiv:2602.22579.

[65] VLTest. 2026. VLTest. https://github.com/beanduan22/VLTest/tree/main

[66] VQA Team. [n. d.]. VQA Evaluation. https://visualqa.org/evaluation.html. Accessed 19 October 2026.

[67] Jialai Wang, Han Qiu, Yi Rong, Hengkai Ye, Qi Li, Zongpeng Li, and Chao Zhang. 2022. BET: black-box eficient testing for convolutional neural networks. In Proceedings ofthe 31st ACM SIGSOFTInternational Symposium on Software Testing and Analysis. 164–175.

[68] Tianshi Wang, Fengling Li, Lei Zhu, Jingjing Li, Zheng Zhang, and Heng Tao Shen. 2025. Cross-modal retrieval: a systematic review of methods and future directions. Proc. IEEE (2025).

[69] Wenhui Wang, Hangbo Bao, Li Dong, Johan Bjorck, Zhiliang Peng, Qiang Liu, Kriti Aggarwal, Owais Khan Mohammed, Saksham Singhal, Subhojit Som, et al. 2023. Image as a foreign language: Beit pretraining for vision and vision-language tasks. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition. 19175–19186.

[70] Oliver Weißl, Amr Abdellatif, Xingcheng Chen, Giorgi Merabishvili, Vincenzo Riccio, Severin Kacianka, and Andrea Stocco. 2025. Targeted deep learning system boundary testing. ACM Transactions on Software Engineering and Methodology (2025).

[71] Adina Williams, Nikita Nangia, and Samuel Bowman. 2018. A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference. In Proceedings ofthe 2018 Conference ofthe North American Chapter ofthe Association for Computational Linguistics: Human Language Technologies (NAACL-HLT). Association for Computational Linguistics, 1112–1122. doi:10.18653/v1/N18-1101

[72] Quanxing Xu, Yuhao Tian, Ling Zhou, Xian Zhong, Xiaohua Huang, Rubing Huang, and Chia-Wen Lin. 2026. MetaRA: Metamorphic Robustness Assessment for Multimodal Large Language Model-based Visual Question Answering Systems. arXiv:2605.19307 [cs.CV]

[73] Ziyi Yin, Muchao Ye, Tianrong Zhang, Tianyu Du, Jinguo Zhu, Han Liu, Jinghui Chen, Ting Wang, and Fenglong Ma. 2023. VLATTACK: Multimodal Adversarial Attacks on Vision-Language Tasks via Pre-trained Models. In Advances in Neural Information Processing Systems (NeurIPS), Vol. 36. Curran Associates, Inc.

[74] Boxi Yu, Zhiqing Zhong, Jiaqi Li, Yixing Yang, Shilin He, and Pinjia He. 2023. ROME: Testing Image Captioning Systems via Recursive Object Melting. In Proceedings of the 32nd ACM SIGSOFT International Symposium on Software Testing and Analysis (ISSTA).

[75] Boxi Yu, Zhiqing Zhong, Xinran Qin, Jiayi Yao, Yuancheng Wang, and Pinjia He. 2022. Automated Testing of Image Captioning Systems. In Proceedings of the 31st

ACM SIGSOFT International Symposium on Software Testing and Analysis (ISSTA).

[76] Yuanyuan Yuan, Shuai Wang, Mingyue Jiang, and Tsong Yueh Chen. 2021. Perception matters: Detecting perception failures of vqa models using metamorphic testing. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 16908–16917.

[77] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. 2023. Sigmoid Loss for Language Image Pre-Training. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). IEEE, 11975–11986.

[78] Jingyi Zhang, Jiaxing Huang, Sheng Jin, and Shijian Lu. 2024. Vision-language models for vision tasks: A survey. IEEE transactions on pattern analysis and machine intelligence 46, 8 (2024), 5625–5644.

[79] Jiaming Zhang, Qi Yi, and Jitao Sang. 2022. Towards Adversarial Attack on Vision-Language Pre-training Models. In Proceedings of the 30th ACM International Conference on Multimedia (MM). arXiv:2206.09391.

[80] Jingqing Zhang, Yao Zhao, Mohammad Saleh, and Peter Liu. 2020. Pegasus: Pre-training with extracted gap-sentences for abstractive summarization. In International conference on machine learning. PMLR, 11328–11339.

[81] Yunqing Zhao, Tianyu Pang, Chao Du, Xiao Yang, Chongxuan Li, Ngai-Man Cheung, and Min Lin. 2023. On evaluating adversarial robustness of large visionlanguage models. Advances in Neural Information Processing Systems 36 (2023), 54111–54138.

[82] Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, et al. 2025. InternVL3: Exploring Advanced Training and Test-Time Recipes for Open-Source Multimodal Models. arXiv:2504.10479 [cs.CV]

[83] Tahereh Zohdinasab, Vincenzo Riccio, Alessio Gambi, and Paolo Tonella. 2021. DeepHyperion: Exploring the Feature Space of Deep Learning-Based Systems through Illumination Search. In Proceedings ofthe 30th ACMSIGSOFTInternational Symposium on Software Testing and Analysis (ISSTA). ACM, 79–90. doi:10.1145/ 3460319.3464811