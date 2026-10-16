# Prism-1: Position- and Rival-Independent Scoring for System One Models

Yao Long Teng Nanyang Technological University Singapore

Jianguo Huang<sup>∗</sup> Nanyang Technological University Singapore

Jiayi Cai Nanyang Technological University Singapore

Yanchen Deng<sup>†</sup> Nanyang Technological University Singapore

Bo An Nanyang Technological University Singapore

## Ab<sub>s</sub>t<sub>rac</sub>t

Given a state and a set of options, such as intents, tools or interface elements, System One decision models return a distribution over the options in one forward pass, without generated reasoning. Web systems increasingly build them by reading decisions from language models. Two axioms of rational choice are natural requirements for such decisions, because in web systems the listing and the option set are determined by other components and can change or be manipulated: the order in which the options are listed should not matter, and adding an option should not change the relative odds of the others (Luce’s choice axiom). We test the standard reads, namely single-choice question (SCQ) prompting, with shufledorder training, order averaging or sorted options, and invariant set architectures, and find that each violates at least one axiom: one shufle changes 12–20% of SCQ answers, sorting only freezes an arbitrary convention, and a well-chosen added option steers 4–34% of decisions. By Luce’s theorem, a read satisfies both axioms only if it is a softmax over scores that each depend on one option alone, so in a transformer they hold by construction only if no token that scores an option sees another. We implement this as Prism-1 (Position- and Rival-Independent Scoring for System One Models): every option is read in its own block at shared positions, with the model’s usual left-to-right attention, in one forward pass. On five benchmark sets with Qwen3-8B and Gemma 4 12B, SCQ prompting trained on the same data still changes 7% of its answers under one shufle, and an added irrelevant option steers 12% of its decisions. Prism-1 stays within one point of that baseline’s accuracy and is 2– 4 points more accurate once only answers that survive a reordering are counted, while no shufle changes any of its answers and no added irrelevant option moves its choice among the others.

## CCS Conce<sub>p</sub>ts

• Computing methodologies → Natural language processing;   
Neural networks.

## Ke<sub>y</sub>words

System One models, rational choice, independence of irrelevant alternatives, order invariance, decision models, large language mod els

## 1 I<sub>n</sub>t<sub>ro</sub>d<sub>uc</sub>ti<sub>on</sub>

System One models answer typed questions in a single forward pass [1, 43]. Named after the fast, intuitive mode of human thinking [20], they return a probability distribution over the options instead of generated text or reasoning, so a caller gets an answer and a confidence it can threshold. Web systems use them wherever a decision must be fast: routing a customer message among intents, picking a tool for an agent, or choosing the next element on a page [9, 54]. The standard way to obtain such a decision from a language model lists the options under letters and reads the probabilities of the letters [33].

We require two properties from choice theory ofa decision over a set. The first is order invariance: listing the same options diferently should not change the decision, a condition that normative theories of choice take as essential [41]. The second is Luce’s choice axiom [23], a form of independence of irrelevant alternatives: adding an option should not change the relative odds of the others. Both matter in web systems because the listing and the option set are not part of the question: other components produce them, and they change or can be manipulated. The order of the options comes from a tool registry, a retriever’s ranking, a database query or the layout of a page, often arbitrarily. Option sets grow as tools are registered and intents are introduced. Under the choice axiom, a new option can win, but it cannot change which of the existing options wins, so it cannot silently alter decisions it was not meant to afect. The same holds against an adversary: tools can be injected into an agent’s library [35], and under the choice axiom an injected option can win only by being chosen itself, never by redirecting the choice among the others. In our tests, an option that names another one as correct redirects 3–55% of the decisions of the other reads we compare, and none of Prism-1’s. Consistency also underpins caching, incident replay, A/B tests and audits, which assume that the same request receives the same decision. And when the options are candidates, such as documents, products or responses, a candidate’s position in the list should not decide its outcome, a bias that web search corrects for when learning from clicks [7, 18] and that afects language models used as judges [44] and as recommenders [14].

The standard decision reads violate both. With Qwen3-8B and Gemma 4 12B reading letters, one shufle of the options changes 20.0% and 11.9% of the answers on the five sets of our benchmark, and a published decision fine-tune ofGemma 4 12B [38] still changes 9.1%. When the decision is a web agent’s next page element, one shufle changes 15% and 7% of trained SCQ’s choices, and an added element steers 32% and 25% of them. Commercial decision APIs are no exception: one shufle changes 4.7% of the answers of Jev 1.13 [1] and 7.6% of those of Mercury Decide [17], against 1.3% and 0.7% when the identical question is asked again, and an added option steers 10.9% and 16.0% of their decisions, against 5.1% and 1.5% for identical copies of the question, as Table 1 shows. The usual remedies do not restore order invariance: training on shufled orders [13] plateaus at about 6% of answers changed, averaging over orders [37, 51] multiplies the cost and stays approximate, and sorting the options only freezes an arbitrary convention, since sorting ${ \cal Z } { - \cal A }$ instead of $_ { \mathrm { A - Z } }$ changes 7–8% of answers. None of them, and none of the invariant set architectures [10, 26], satisfies the choice axiom: adding one irrelevant option moves 2–10 points of probability among the options that were already there, and an adversary who can add one option steers 4–34% of their decisions.

The axioms leave a System One read little freedom: it cannot reason its way to consistency or average over many orders, so consistency has to hold within one forward pass. By Luce’s theorem, a rule satisfies the choice axiom only if it is a softmax over scores that each depend on one option alone, and with order invariance the scores cannot depend on the listing either. This form is familiar on the web: listwise learning to rank uses the Plackett–Luce model [5, 31, 47], and pairwise preference models, from Bradley–Terry [4] to the reward models that align language models [29], are its twooption case. In a transformer, the axioms hold by construction when no token that contributes to an option’s score attends to another option and an option’s positions do not depend on where it is listed. Otherwise, they hold only if the model happens to ignore what it sees. Each of the reads we test breaks this condition: the letter that SCQ reads attends to the whole list, and the answer tokens of set architectures attend to every option.

We implement the form the axioms require as Prism-1 (Positionand Rival-Independent Scoring for System One Models). It gives every option a block that starts at the same position after a shared context, sees the context and itself and nothing else, and is read by the model’s own causal, left-to-right attention. The yes/no question is asked once in the context, and the model answers it at the last token of each block, and a softmax across the blocks gives the decision. Both axioms hold by construction, any number of options can be read in one pass, and the shared context is computed once. The open question is whether a language model can compute accurate scores this way, and we find that it can if each option is read causally.

On five benchmark sets with two models, Prism-1 is statistically tied with SCQ prompting trained on the same data (0.9 points below it on Gemma and level with it on Qwen), while it changes no answer under shufling and moves no probability between the original options when one is added, where SCQ changes 7% of its answers and moves about 4 points of probability. Counting only answers that survive a reordering of the options, it is 1.6 and 3.7 points more accurate. It stays ahead of SCQ in order-robust accuracy as the number of options grows, and it processes fewer tokens than any other read we measured. Our contributions are as follows.

• Section 2 characterises the decision reads that satisfy order invariance and Luce’s choice axiom, and what this requires of transformer architectures.

• Section 3 shows that each of the standard reads violates at least one axiom, with three tests: shufles, an added irrelevant option, and a change of sort key.

• Section 4 introduces Prism-1, a read of the required form for language models and the cheapest read we measured.

• Section 5 evaluates it on five sets and two models, studies how to compute each option’s score, and measures what the axioms cost.

## 2 A<sub>x</sub>i<sub>oms</sub> f<sub>or</sub> D<sub>ec</sub>i<sub>s</sub>i<sub>on</sub> R<sub>ea</sub>d<sub>s</sub>

Decisions. A decision model answers a choice question � about a state � with a set $S = \left\{ o _ { 1 } , \dots , o _ { K } \right\}$ of options, exactly one of which is correct. The question presents its options in some order �, a listing of �. A decision rule � maps (�, �, �) to a distribution over �, and its answer is the most probable option.

SCQ prompting. The standard rule lists the options of � as a question under the letters $\mathrm { A , B , \ldots }$ , and renormalises the next-token probabilities of these letters, a read often called multiple-choice prompting [33]. Since exactly one option is correct, we call it SCQ, for single-choice question. The letter and the position of each option both depend on �.

Definition 2.1 (Order invariance). A decision rule � is orderinvariant if �(�, $q , \sigma ) \ = \ R ( x , q , \sigma ^ { \prime } )$ for all �, � and listings $\sigma , \sigma ^ { \prime }$ of the same set. We then write $R ( o \mid S )$ for the probability of option � in set �, leaving the context implicit.

Definition 2.2 (Choice axiom [23]). An order-invariant rule satisfies the choice axiom if $R ( a \mid T ) = R ( a \mid S ) R ( S \mid T )$ for all sets $S \subseteq T$ and $a \in S ,$ where $\begin{array} { r } { R ( S \mid T ) = \sum _ { b \in S } R ( b \mid T ) } \end{array}$ . Equivalently, the ratio $R ( a \mid T ) / R ( b \mid T )$ is the same in every set that contains � and �.

The choice axiom is the probabilistic form of independence of irrelevant alternatives. In particular, adding an option can change the answer only to the added option, and it takes probability from the other options in proportion. This is a requirement, not a law: for an added option that nearly duplicates an existing one, a share taken mostly from its twin may be more reasonable [8], a case we return to in Section 7.

Proposition 2.3 (Luce [23]). A rule that gives every option a positive probability satisfies the choice axiom if and only if there is a <sub>pos</sub>iti<sub>ve</sub> <sub>sca</sub>l<sub>e</sub> <sub>�</sub> <sub>w</sub>ith $R ( a \mid S ) = v ( a ) / \sum _ { b \in S } v ( b )$

Proposition 2.4. A decision rule that gives every option a positive probability is order-invariant and satisfies the choice axiom in every context if and only if

$$
R ( o \mid x , q , \sigma ) = { \frac { \exp s ( x , q , o ) } { \sum _ { o ^ { \prime } \in S } \exp s ( x , q , o ^ { \prime } ) } }\tag{1}
$$

for a score � that depends on the state, the question and the option <sub>a</sub>l<sub>one.</sub>

Proof. If � has this form, the listing does not enter it, and $R ( a \mid$ $T ) / R ( b \mid T ) = \exp ( s ( a ) - s ( b ) )$ in every set. Conversely, order invariance makes � a function of the set, and Proposition 2.3 gives a scale � for each context. Its ratios $v ( a ) / v ( b ) = R ( a \mid \{ a , b \} ) / R ( b \mid$ $\{ a , b \} )$ are fixed by the two-option sets, so � is determined, up to a constant, by the options themselves, and we set � = log �. Appendix A gives both proofs in full. □

![](images/99ad132031afe0a946595eb0da6c78d2cbda37b8b4107630f46c125930790c94.jpg)

![](images/3f8b5350d03ea304c2f234de0263f17119427f956650a3015cd935d8ed192062.jpg)  
Fi ure 1: SCQ violates both axioms after trainin and after sortin (mean over five sets). (a) Share of answers that chan e under one shufle after each epoch of training on shufled orders, and when the options move but keep their own letters. (b) After <sub>one epoc</sub>h <sub>o</sub>f t<sub>ra</sub>i<sub>n</sub>i<sub>ng,</sub> th<sub>e answers</sub> th<sub>a</sub>t <sub>c</sub>h<sub>ange w</sub>h<sub>en</sub> th<sub>e op</sub>ti<sub>ons are s</sub>h<sub>u</sub>fl<sub>e</sub>d<sub>, zero</sub> b<sub>y cons</sub>t<sub>ruc</sub>ti<sub>on w</sub>ith <sub>sor</sub>t<sub>e</sub>d <sub>op</sub>ti<sub>ons</sub> b<sub>ecause</sub> <sub>sor</sub>ti<sub>n un</sub>d<sub>oes</sub> th<sub>e s</sub>h<sub>u</sub>fl<sub>e</sub> th<sub>e answers</sub> th<sub>a</sub>t <sub>move</sub> t<sub>o a</sub> dif<sub>eren</sub>t <sub>or</sub>i i<sub>na</sub>l <sub>o</sub> ti<sub>on w</sub>h<sub>en one</sub> i<sub>rre</sub>l<sub>evan</sub>t <sub>o</sub> ti<sub>on</sub> i<sub>s a</sub>dd<sub>e</sub>d <sub>an</sub>d f<sub>or</sub> <sub>op</sub>ti<sub>ons sor</sub>t<sub>e</sub>d b<sub>y</sub> t<sub>ex</sub>t<sub>,</sub> th<sub>e answers</sub> th<sub>a</sub>t <sub>c</sub>h<sub>ange w</sub>h<sub>en</sub> th<sub>ey are sor</sub>t<sub>e</sub>d Z<sub>–</sub>A i<sub>ns</sub>t<sub>ea</sub>d <sub>o</sub>f A<sub>–</sub>Z<sub>.</sub>

Consequence for transformers. Proposition 2.4 is guaranteed by construction if the computation of each option’s score uses only the shared context and that option’s own tokens, at positions that do not depend on the listing. For a System One model this is the only route: averaging over orders needs many passes and remains approximate, and reasoning needs generation. A read in which a token that contributes to an option’s score attends to another option, or in which positions depend on the listing, satisfies the axioms only if the model happens to ignore that information. SCQ reads one letter token that attends to the whole list. Sorting the options removes the dependence on the listing but not on the set: an added option shifts the positions of the options after it. Set-Based Prompting [26] and Set-LLM [10] give options shared positions, but read the decision from tokens that attend to every option.

Metrics. Accuracy is measured with the options in their source order. The flip rate is the share of answers that change between the source order and one seeded shufle, excluding exact ties. For the choice axiom we add one irrelevant option to every question. The switch rate is the share of answers that move to a diferent original option. The probability drift is the total variation distance, in points, between the distribution over the original options before and after the addition, renormalised over the original options. It is zero on every question if and only if the choice axiom holds for these additions. The steering rate is a worst case: the share of questions for which at least one of 10 candidate irrelevant options moves the answer to a diferent original option, as an adversary who can add one option would choose it.

## 3 St<sub>a</sub>nd<sub>a</sub>rd R<sub>ea</sub>d<sub>s</sub> Vi<sub>o</sub>l<sub>a</sub>t<sub>e</sub> th<sub>e</sub> Axi<sub>o</sub>m<sub>s</sub>

We study SCQ on Qwen3-8B [48] and Gemma 4 12B [11], untrained and after fine-tuning on 9.1k decisions with a fresh random order per example, as described in Section 5.1. All numbers are means over the five benchmark sets.

Position, not the letter. A shufle moves both an option’s letter and its slot in the list. We move one at a time. When options stay in place but take the letters the shufle would give them, 9.1% (Gemma) and 14.0% (Qwen) of answers change, and 5.4% and 3.8% after training. When options move but keep their own letters, 12.0% and 22.7% change, at least as many as under a full shufle (11.9% and 20.0%).

Training plateaus. Figure 1a shows that training on shufled orders reduces the flip rate but does not remove it: after three epochs, one shufle still changes 5.6% and 6.0% of answers while accuracy has risen by about one point, and moving only the slots still changes 8.4% and 6.8%.

Sortingfixes a convention. Sorting the options by text gives every listing of a set the same input, so the flip rate is zero. But A–Z is one convention among many: reading the same options sorted Z–A instead changes 7.2% and 8.5% of answers at equal accuracy, as Figure 1b shows. Sorting does not remove the dependence on position. It freezes it into a choice that is unrelated to the question.

Added options move the others. When we add one irrelevant option, trained SCQ moves 3.3% and 2.7% ofits answers to a diferent original option, and SCQ with sorted options 3.1% and 2.2%: in both listings, the new option shifts the letters and positions of the options after it. Below the answers, 3.8 and 4.0 points of probability move among the original options. Set architectures, which are orderinvariant, violate the choice axiom as much or more, as Section 5.3 shows.

Takeaways. Order dependence lives mainly in positions. Training does not teach the model to ignore them, sorting only fixes them, and none of the reads we test satisfies the choice axiom. Proposition 2.4 says what a compliant read must look like, and the next section builds one.

(a) Architecture: all options in one forward pass  
![](images/9e3658ce2390918267da2e7a910bb250c001327b0acebb3fc7617eb6f5b72c40.jpg)  
Figure 2: Prism-1. (a) Every option is a block that starts at the same position � after the shared context, and all blocks are <sub>rea</sub>d i<sub>n one</sub> f<sub>orwar</sub>d <sub>pass o</sub>f th<sub>e pre</sub>t<sub>ra</sub>i<sub>ne</sub>d d<sub>eco</sub>d<sub>er,</sub> fi<sub>ne-</sub>t<sub>une</sub>d <sub>w</sub>ith L<sub>o</sub>RA<sub>.</sub> Th<sub>e score</sub> $s _ { k } = \log P ( y e s \mid h , b _ { k } )$ − log $P ( { \pmb n } \pmb { 0 } \mid h , b _ { k } )$ i<sub>s</sub> read at the last token of each block (■), a softmax over the blocks gives the decision, and training minimises $- \log R ( o _ { y } \mid S )$ f<sub>or</sub> th<sub>e correc</sub>t <sub>o</sub> ti<sub>on</sub> $o _ { y } . \left( \mathbf { b } \right)$ Th<sub>e a</sub>tt<sub>en</sub>ti<sub>on mas</sub>k<sub>:</sub> th<sub>e con</sub>t<sub>ex</sub>t <sub>a</sub>tt<sub>en</sub>d<sub>s causa</sub>ll<sub>y</sub> t<sub>o</sub> it<sub>se</sub>lf<sub>, an</sub>d <sub>eac</sub>h bl<sub>oc</sub>k <sub>a</sub>tt<sub>en</sub>d<sub>s</sub> t<sub>o</sub> th<sub>e con</sub>t<sub>ex</sub>t <sub>an</sub>d causall<sub>y</sub> to itself<sub>,</sub> never to another o<sub>p</sub>tion. No o<sub>p</sub>tion’s score de<sub>p</sub>ends on the other o<sub>p</sub>tions or on the listin<sub>g</sub>.

## 4 Prism-1

Prism-1 computes the score $s ( x , q , o )$ of Proposition 2.4 with a language model, for all options in one forward pass.

Layout. Prism-1 places a shared context ℎ at positions $0 , \ldots , P { - } 1 { \mathrm { : } }$ the state, the question, and a prompt that asks whether the following candidate is the correct answer and ends with “Candidate answer:”. Each option $o _ { k }$ becomes a block $b _ { k }$ that holds the option’s text and nothing else. Every block starts at position � and counts up inside itself, so word order inside an option is kept, and there is no order between options, as Figure 2 shows. The attention mask lets the context attend causally to itself and lets each block attend to the context and causally to itself, but never to another block. There is no token after the blocks, so no token ever attends to more than one option.

Causal reading. A block computes exactly what an ordinary causal (left-to-right) forward pass over the context followed by that one option computes. The model therefore reads each option as it was pretrained to. Nothing in a block asks for an answer, so the read at its last token is learned in fine-tuning.

Score. The score of option $o _ { k }$ is the model’s answer at the last token of its block,

$$
s ( o _ { k } ) = \log { P ( y e s \mid h , b _ { k } ) } - \log { P ( n o \mid h , b _ { k } ) } ,\tag{2}
$$

and the decision is the softmax of the scores. Every option is judged with the same two tokens, so no option needs a label, and the last token of a block is the first position that has read the whole option. If each block’s answer is read as an independent judgment of its option, correct with probability $\ p _ { k } = P ( y e s ) / ( P ( y e s ) + P ( n o ) )$ , and exactly one option is correct, the probability that $o _ { k }$ is the answer

is

$$
\frac { p _ { k } \prod _ { j \neq k } ( 1 - p _ { j } ) } { \sum _ { i } p _ { i } \prod _ { j \neq i } ( 1 - p _ { j } ) } = \frac { \exp s ( o _ { k } ) } { \sum _ { i } \exp s ( o _ { i } ) } ,\tag{3}
$$

the softmax of these scores, derived in Appendix A. All blocks are read in one packed forward pass, with the context computed once.

Training. We fine-tune with LoRA [15] on the cross-entropy over the options, $\mathcal { L } = - \log p ( o _ { y } )$ for the correct option $o _ { y } .$ . No order augmentation or consistency term is needed: the model cannot see an order.

Properties. Each score depends only on $( x , q , o _ { k } )$ , so by Proposition 2.4 Prism-1 is order-invariant and satisfies the choice axiom: adding an option leaves every score unchanged and only renormalises the softmax. All options are read in one forward pass, as a System One model requires. The cost grows linearly with the number of options, and since no option needs a label, their number is not bounded by a set of single-token labels. Decisions are also incremental: an added option, such as a newly registered tool, needs only its own block, given the cached context, and every earlier score stays valid, whereas reads in which options see each other must read the whole set again.

An answer slot. Appending the question, the end of the user turn and an answer prefix to every block makes the read usable without training, at the cost of blocks several times longer. We use it only in the design study of Section 5.5.

## 5 Ex<sub>p</sub>eriments

## 5.1 <sup>S</sup>etu<sub>p</sub>

Models. Qwen3-8B [48] and Gemma 4 12B [11], instruction tuned, with thinking disabled.

Benchmark. We use five sets with 2 to 20 options. Banking20 has 300 test messages over 20 intents of a banking intent dataset [6], and Newsgroups 300 posts over the 20 topics of 20 Newsgroups [21]. BEV [2] mixes decision questions built from public datasets and records, from page-element and tool selection to product categories, response preference, code and contract evidence, and we take 100 choice questions from each of the 12 domains of its test split (1,182 questions, 2–10 options). Typed Decisions [22] contributes the 600 choice questions of its test split (4–5 options), and JevBench [36] its 139 public choice tasks (3–6 options).

Training data. 9,100 questions for every trained method: that dataset’s training split on the same 20 intents, 20 Newsgroups’ training split, 2,500 choice questions from BEV’s training split, and Typed Decisions’ training split, with 50 questions per set held out for validation. JevBench has no training data and serves as a held-out task.

Baselines. SCQ, fine-tuned with a fresh random order per example. Cyclic averaging [51], which averages the SCQ read over the � rotations of the listing (AnyJev’s order averaging, without its prior correction). SCQ, sorted options, which lists the options sorted by text. Set-Based Prompting [26], training-free. Set-LLM [10], which we re-implement: a bidirectional prompt with set positions and the set mask, trained to generate the correct option’s text and read by the likelihood of each option’s text. Every trained method uses the same data and budget, and all re-implementations are checked against an unpacked reference.

Trainin . LoRA with rank 16 and � = 32 on the attention and MLP projections, AdamW with learning rate 10<sup>−4</sup>, 3% warm-up and cosine decay, 32 questions per step, one epoch. SCQ and Prism-1 are trained with three seeds and Set-LLM with two, and reported as means.

## 5<sub>.</sub>2 M<sub>a</sub>in R<sub>esu</sub>lt<sub>s</sub>

Table 1 reports accuracy, flip rates, order-robust accuracy and the tests of the choice axiom described in Section 5.3.

Against SCQ.. Prism-1 reaches 79.5% on Gemma and 78.0% on Qwen, against 80.4% and 77.9% for trained SCQ, which flips 7.1% and 7.4% of its answers. No diference is significant in exact McNemar tests pooled over the five sets $( p \ge 0 . 1 3 )$ . Per set, SCQ is ahead on the held-out JevBench and, on Gemma, on Typed, and Prism-1 on Banking20 and Newsgroups, with per-set tests in Appendix B. SCQ with sorted options is also order-invariant and 1.6 and 0.7 points more accurate in this setting (not significant, $p \ge 0 . 0 8 )$ , but it violates the choice axiom, as Section 5.3 shows. Cyclic averaging needs � passes and still flips 2.3–3.0% ofanswers, and AnyJev’s prior correction does not make it more stable. Prism-1’s probabilities are as well calibrated as SCQ’s, with an expected calibration error of 5.3% and 5.4% against 4.9% and 5.4%.

Againstinvariantsetmethods. Set-Based Prompting, which needs no training, reaches only 48.1% and 50.9% on these instruction-tuned models, as its authors found for Llama 3. At equal training, Prism-1 is 4.0 and 5.9 points more accurate than Set-LLM on average (� ≤ 0.002), mostly on BEV and the held-out JevBench, and Section 5.5 traces this to how each option is read.

![](images/20d2ec7c6032c814421ee69b6cfff0b3b113eb0a07e2876120267818db36413f.jpg)  
Figure 3: BEV by domain: order-robust accuracy of SCQ (grey) and Prism-1 (blue), which for Prism-1 equals its accuracy. P<sub>rism-</sub>1 i<sub>s a</sub>h<sub>ea</sub>d <sub>w</sub>h<sub>erever</sub> it<sub>s</sub> d<sub>o</sub>t li<sub>es</sub> t<sub>o</sub> th<sub>e r</sub>i<sub>g</sub>ht<sub>.</sub>

Accuracy that survives reordering. Accuracy in the source order credits SCQ with questions it answers wrongly under another listing of the same options. Counting a question only if the answer is right in both listings, the Rob. column of Table 1, SCQ falls from 80.4% to 77.8% on Gemma and from 77.9% to 74.3% on Qwen, and Prism-1, unchanged, is 1.6 and 3.7 points ahead $( p < 0 . 0 1 )$ . Only cyclic averaging, at � passes per decision, keeps up: it is 1.0 points ahead of Prism-1 on Gemma (not significant, $p = 0 . 0 8 )$ and 1.3 points behind on Qwen $( p = 0 . 0 2 )$ . Two listings understate the efect: counting a question only if SCQ is right under all � rotations of its listing, SCQ falls to 74.9% and 69.6%, 4.7 and 8.1 points below Prism-1 $( p < 1 0 ^ { - 1 1 } )$ .

By domain. Figure 3 breaks BEV down by domain. Prism-1 has the higher order-robust accuracy in 19 of the 24 domain–model pairs and is never more than 2.3 points behind, with its largest gains on the hardest domains. On browser interaction, where a web agent picks the next page element, it is also more accurate than SCQ in the source order (84.7% and 81.0% against 76.3% and 74.3%). On engagement, which asks which of two headlines drew more clicks, SCQ’s lead in the source order (11 and 6 points) does not survive swapping the headlines, which changes 22% and 30% of its answers.

## 5<sub>.</sub>3 T<sub>es</sub>ti<sub>ng</sub> th<sub>e</sub> A<sub>x</sub>i<sub>oms</sub>

To test the choice axiom, we add to every question one option from another question of the same set (for Banking20 and Newsgroups, a banking intent outside Banking20’s 20), at a random position.

Every baseline violates the axiom, as the Dr. and St. columns of Table 1 and the switch rates in Figure 5c show. An added option moves 2–5% of the answers of SCQ, sorted SCQ and Set-LLM to another original option, and 3–4 points of probability. Prism-1 moves neither answers nor probability. An adversary who can add one option steers 4–34% of the decisions of every other read and of the published decision models, but none of Prism-1’s, and needs few attempts: a single added option, placed where it does most harm, already steers 6–9% of the decisions of SCQ and of the published models.

Table 1: Main results. Acc: accuracy (%). Flip: answers changed by one shufle (%). Dr.: probability moved among the original options when one irrelevant option is added (points). St.: questions on which an added irrelevant option can move the answer to another original option (%). Rob.: order-robust accuracy, right in both the source and the shufled order (%). Flip, Dr. and St. are zero for a rule that satisfies both axioms. For SCQ with sorted options, Rob. compares A–Z with Z–A sorting. <sup>§</sup>: published <sub>mo</sub>d<sub>e</sub>l<sub>, no</sub>t t<sub>ra</sub>i<sub>ne</sub>d <sub>on our</sub> d<sub>a</sub>t<sub>a.</sub> B<sub>o</sub>ld<sub>:</sub> b<sub>es</sub>t <sub>per mo</sub>d<sub>e</sub>l<sub>, exc</sub>l<sub>u</sub>di<sub>ng pu</sub>bli<sub>s</sub>h<sub>e</sub>d <sub>mo</sub>d<sub>e</sub>l<sub>s.</sub>
<table><tr><td></td><td colspan="3">Banking20</td><td colspan="3">Newsgroups</td><td colspan="3">BEV</td><td colspan="3"></td><td colspan="3">Typed</td><td colspan="3">JevBench</td><td colspan="3">Avg</td><td colspan="3"></td></tr><tr><td>Method</td><td>Acc Flip</td><td></td><td>Dr.</td><td>St.</td><td>Acc Flip</td><td>Dr.</td><td>St.</td><td>Acc</td><td>Flip</td><td>Dr.</td><td>St.</td><td>Acc</td><td>Flip</td><td>Dr.</td><td>St.</td><td>Acc Flip</td><td>Dr.</td><td></td><td>St.</td><td>Acc</td><td>Rob.</td><td>Flip Dr.</td><td>St.</td></tr><tr><td colspan="10">Closed-source models, through their APIs</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Jev 1.13§</td><td>87.0</td><td>4.0</td><td>1.6 3.3</td><td>75.3</td><td>3.7</td><td>2.8</td><td>8.0</td><td>76.4 6.3</td><td>3.2</td><td>13.9</td><td>72.8</td><td>8.0</td><td>5.1</td><td>22.0</td><td>88.5</td><td>1.4</td><td>1.6</td><td>7.2</td><td>80.0</td><td>77.9</td><td>4.7</td><td>2.9</td><td>10.9</td></tr><tr><td>Mercury Decide§</td><td>91.3</td><td>2.3 1.6</td><td>7.0</td><td>73.3</td><td>7.7</td><td>2.6</td><td>8.3 75.1</td><td>12.5</td><td>4.7</td><td>27.2</td><td>67.3</td><td>10.5</td><td>6.8</td><td>30.5</td><td>89.9</td><td>5.0</td><td>1.4</td><td>7.2</td><td>79.4</td><td>76.7</td><td>7.6</td><td>3.4</td><td>16.0</td></tr><tr><td colspan="10">Open-source models: Gemma 4 12B</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>SCQ</td><td>91.9</td><td>5.4</td><td>1.8</td><td>3.0 71.8</td><td>9.4</td><td>4.2</td><td>13.2</td><td>77.5</td><td>10.1</td><td>5.2</td><td>20.3 75.9</td><td></td><td>5.3 4.8</td><td>11.2</td><td>84.9</td><td>5.6</td><td>2.8</td><td>10.8</td><td>80.4</td><td>77.8</td><td></td><td>7.1 3.8</td><td>11.7</td></tr><tr><td>SCQ + cyclic avg.</td><td>91.7</td><td>0.3</td><td>0.5 2.0</td><td>73.3</td><td>5.0</td><td>1.1</td><td>2.0</td><td>79.2</td><td>2.5</td><td>3.2</td><td>7.0 74.5</td><td>1.7</td><td>2.9</td><td>6.0</td><td>86.3</td><td>2.2</td><td>1.8</td><td>3.0</td><td>81.0</td><td>80.5</td><td>2.3</td><td>1.9</td><td>4.0</td></tr><tr><td>SCQ, sorted</td><td>92.0</td><td>0.0</td><td>1.4 3.3</td><td>74.7</td><td>0.0</td><td>3.0</td><td>6.8</td><td>77.3</td><td>0.0</td><td>4.8 12.8</td><td>74.5</td><td>0.0</td><td>4.7</td><td>6.9</td><td>87.1</td><td>0.0</td><td>3.2</td><td>7.9</td><td>81.1</td><td>78.1</td><td>0.0</td><td>3.4</td><td>7.6</td></tr><tr><td>Set-Based Prompting</td><td>25.3</td><td>0.0</td><td>7.9 43.7</td><td>27.7</td><td>0.0</td><td>8.3</td><td>32.3</td><td>51.9</td><td>0.0</td><td>12.9 37.8</td><td>57.0</td><td>0.0</td><td>9.1</td><td>28.0</td><td>78.4</td><td>0.0</td><td>10.9</td><td>28.8</td><td>48.1</td><td>48.1</td><td>0.0</td><td>9.8</td><td>34.1</td></tr><tr><td>Set-LLM</td><td>89.7</td><td>0.0 1.1</td><td>3.0</td><td>73.2</td><td>0.0</td><td>2.1</td><td>5.3</td><td>70.6</td><td>0.0</td><td>5.8 24.6</td><td>75.4</td><td>0.0</td><td>5.5</td><td>14.3</td><td>68.7</td><td>0.0</td><td>2.6</td><td>8.6</td><td>75.5</td><td>75.5</td><td>0.0</td><td>3.4</td><td>11.2</td></tr><tr><td>Torchcast Decision 12B§</td><td>83.7</td><td>8.8 2.9</td><td>7.4 0.0 0.0</td><td>66.3</td><td>12.5</td><td>4.4</td><td>8.1 0.0</td><td>71.1 77.2</td><td>10.4</td><td>5.9 21.6</td><td>69.0</td><td>8.7</td><td>5.4</td><td>25.4</td><td>87.8</td><td>5.0</td><td>2.0</td><td>10.1</td><td>75.6</td><td>72.2</td><td>9.1</td><td>4.2</td><td>14.5</td></tr><tr><td colspan="10">PRISM-1 93.0 0.0</td><td>0.0 0.0</td><td>72.4</td><td>0.0</td><td>0.0</td><td>0.0</td><td>81.3</td><td>0.0</td><td>0.0</td><td>0.0</td><td>79.5</td><td>79.5</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>Open-source models: Qwen3-8B</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>SCQ</td><td>91.9</td><td>3.3</td><td>1.7 3.7</td><td>73.1</td><td>9.7</td><td>4.5</td><td>10.8</td><td>72.6</td><td>11.7</td><td>5.6 24.3</td><td>74.2</td><td>5.0</td><td>4.9</td><td>9.7</td><td>77.9</td><td>7.3</td><td>3.6</td><td>9.5</td><td>77.9</td><td>74.3</td><td>7.4</td><td></td><td>4.011.6</td></tr><tr><td>SCQ + cyclic avg.</td><td>92.0</td><td>1.7</td><td>0.5 2.0</td><td>73.3</td><td>7.0</td><td>1.0</td><td>2.0</td><td>74.0</td><td>3.1</td><td>2.8</td><td>5.1 74.0</td><td>1.0</td><td>2.4</td><td>11.1</td><td>76.3</td><td>2.2</td><td>1.8</td><td>4.0</td><td>77.9</td><td>76.6</td><td>3.0</td><td>1.7</td><td>4.8</td></tr><tr><td>SCQ, sorted</td><td>92.3</td><td>0.0</td><td>1.6 1.0</td><td>74.7</td><td>0.0</td><td>2.8</td><td>6.4</td><td>73.6</td><td>0.0</td><td>4.9 12.2</td><td>73.2</td><td>0.0</td><td>4.5</td><td>4.3</td><td>79.9</td><td>0.0</td><td>3.6</td><td>7.3</td><td>78.7</td><td>74.4</td><td>0.0</td><td>3.5</td><td>6.2</td></tr><tr><td>Set-Based Prompting</td><td>29.0</td><td>0.0</td><td>8.0 32.3</td><td>58.7</td><td>0.0</td><td>5.2</td><td>18.3</td><td>51.6</td><td>0.0</td><td>12.7 34.3</td><td>54.2</td><td>0.0</td><td>10.5</td><td>34.6</td><td>61.2</td><td>0.0</td><td>9.6</td><td>23.7</td><td>50.9</td><td>50.9</td><td>0.0</td><td>9.2</td><td>28.6</td></tr><tr><td>Set-LLM</td><td>92.7</td><td>0.0</td><td>1.1</td><td>3.3 74.8</td><td>0.0</td><td>2.1</td><td>5.3</td><td>62.2</td><td>0.0</td><td>5.9</td><td>26.9 65.1</td><td>0.0</td><td>6.4 0.0</td><td>25.8 0.0</td><td>65.5 73.9</td><td>0.0 0.0</td><td>3.5 0.0</td><td>17.3 0.0</td><td>72.1 78.0</td><td>72.1 78.0</td><td>0.0 0.0</td><td>3.8</td><td>15.7</td></tr><tr><td>PRISM-1</td><td>93.0</td><td>0.0</td><td>0.0</td><td>0.0 74.9</td><td>0.0</td><td>0.0</td><td>0.0</td><td>73.9</td><td>0.0</td><td>0.0</td><td>0.0 74.3</td><td>0.0</td></table>

<table><tr><td rowspan=3 colspan=1></td><td rowspan=3 colspan=1>Banking20</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=2 colspan=1>groups</td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=1 colspan=4>BEV   Typed JevBench Mean</td></tr><tr><td rowspan=1 colspan=1>Jev 1.13</td><td rowspan=1 colspan=1>0.3</td><td rowspan=1 colspan=1>3.4</td><td rowspan=1 colspan=1>7.6</td><td rowspan=1 colspan=1>5.4</td><td rowspan=1 colspan=1>3.0</td><td rowspan=1 colspan=1>3.9</td></tr><tr><td rowspan=1 colspan=1>Mercury Decide</td><td rowspan=1 colspan=1>2.3</td><td rowspan=1 colspan=1>5.7</td><td rowspan=1 colspan=1>18</td><td rowspan=1 colspan=1>7.5</td><td rowspan=1 colspan=1>3.0</td><td rowspan=1 colspan=1>7.3</td></tr><tr><td rowspan=1 colspan=1>SCQ, Gemma</td><td rowspan=1 colspan=1>6.7</td><td rowspan=1 colspan=1>26</td><td rowspan=1 colspan=1>50</td><td rowspan=1 colspan=1>16</td><td rowspan=1 colspan=1>16</td><td rowspan=1 colspan=1>23</td></tr><tr><td rowspan=1 colspan=1>Set-LLM, Gemma</td><td rowspan=1 colspan=1>4.5</td><td rowspan=1 colspan=1>10</td><td rowspan=1 colspan=1>38</td><td rowspan=1 colspan=1>35</td><td rowspan=1 colspan=1>26</td><td rowspan=1 colspan=1>23</td></tr><tr><td rowspan=1 colspan=1>Prism-1, Gemma</td><td rowspan=1 colspan=1>0.0</td><td rowspan=1 colspan=1>0.0</td><td rowspan=1 colspan=1>0.0</td><td rowspan=1 colspan=1>0.0</td><td rowspan=1 colspan=1>0.0</td><td rowspan=1 colspan=1>0.0</td></tr><tr><td rowspan=1 colspan=1>SCQ, Qwen</td><td rowspan=1 colspan=1>1.2</td><td rowspan=1 colspan=1>5.0</td><td rowspan=1 colspan=1>27</td><td rowspan=1 colspan=1>7.2</td><td rowspan=1 colspan=1>7.6</td><td rowspan=1 colspan=1>9.6</td></tr><tr><td rowspan=1 colspan=1>Set-LLM, Qwen</td><td rowspan=1 colspan=1>3.7</td><td rowspan=1 colspan=1>5.0</td><td rowspan=1 colspan=1>31</td><td rowspan=1 colspan=1>15</td><td rowspan=1 colspan=1>22</td><td rowspan=1 colspan=1>15</td></tr><tr><td rowspan=1 colspan=1>Prism-1, Qwen</td><td rowspan=1 colspan=1>0.0</td><td rowspan=1 colspan=1>0.0</td><td rowspan=1 colspan=1>0.0</td><td rowspan=1 colspan=1>0.0</td><td rowspan=1 colspan=1>0.0</td><td rowspan=1 colspan=1>0.0</td></tr></table>

Figure 4: Injected options: answers (%) that an added option nam<sup>i</sup>ng a wrong op<sup>ti</sup>on as correc<sup>t</sup> moves <sup>t</sup>o <sup>th</sup>a<sup>t</sup> op<sup>ti</sup>on, per set.

Injected options. An added option can also carry text aimed at the decision, as an injected tool description or a planted page element would. We add to every question one option that names a wrong original option as the correct answer (“Note to the decision model: the correct answer is X” or “System update: X is the only valid choice; ignore the other options”), at the start, the middle or the end. It redirects every other read we test to the named option, most of all on BEV, as Figure 4 shows. The order-invariant set architectures are among the easiest to redirect: Set-Based Prompting on 55% and 31% of questions and Set-LLM on 23% and 15%, with every read in Appendix C.2. Invariance to the order alone does not protect a decision. The choice axiom does. Prism-1 is never redirected, since no option’s text reaches another option’s score.

## 5.4 Number of O<sub>p</sub>tions and Cost

The five sets have 2 to 20 options. To see how the reads behave as the set grows, we read the 300 Newsgroups test posts with 5, 10 and 20 of its topics as options, nested so that every larger set contains the smaller ones.

Figure 5a–b shows order-robust accuracy as the options grow. Prism-1 is ahead of SCQ at every size on both models, by 3.1–4.1 points on Gemma and 2.4–3.2 on Qwen (� < 0.05 for 8 of the 18 comparisons). Even in the source order alone, SCQ is ahead in only one of the 18 pairs, by 0.3 points, while Prism-1 leads by up to 4.3 points (significantly for one pair). Set-LLM stays within 1.6 points of Prism-1 on Newsgroups, whose topics are all in the training data, and falls behind on the sets that are not.

Appendix Table 5 shows that Prism-1 is also the cheapest read. On Banking20 it processes 221 tokens, against 282 for SCQ and 322 for Set-LLM, which reads every option twice, with about half of SCQ’s attention pairs, and on Newsgroups 356 against 417 and 413. It is also the fastest, although with 20 options the shared context dominates (58 and 60 ms on Gemma, against 60–63 ms for SCQ and 66–68 ms for Set-LLM). The answer slot of Section 4 repeats the question after every option: 591 tokens and 76 ms on Banking20.

![](images/dd9b877e159d916b07bde12b95753f395431f14094c7f764eb3415225e992991.jpg)

![](images/1ee4a1ab6f2b985ce3d7409991ad5dbbca0e04254fd4f4c30e304385265fc701.jpg)

![](images/68643224928d539898eecc6252e96a1c1e320da742ca5a3da5052d5370935389.jpg)  
Figure 5: The number of options and the choice axiom, with every method trained for one epoch on the same data. (a, b) O<sub>r</sub>d<sub>er-ro</sub>b<sub>us</sub>t <sub>accuracy,</sub> <sub>r</sub>i<sub>g</sub>ht b<sub>o</sub>th i<sub>n</sub> th<sub>e</sub> <sub>source</sub> <sub>or</sub>d<sub>er</sub> <sub>an</sub>d i<sub>n</sub> th<sub>e</sub> <sub>s</sub>h<sub>u</sub>fl<sub>e</sub>d <sub>or</sub>d<sub>er,</sub> <sub>on</sub> th<sub>e</sub> 300 N<sub>ewsgroups</sub> t<sub>es</sub>t <sub>pos</sub>t<sub>s</sub> <sub>w</sub>ith � <sub>nes</sub>t<sub>e</sub>d topics as options. Shaded: SCQ’s answers that are right in the source order but not after one reshufle, up to its accuracy in the source order alone. (c) Switch rate: answers that move to a diferent original option when one irrelevant option is added, <sub>average</sub>d <sub>over</sub> fi<sub>ve se</sub>t<sub>s.</sub>

T<sub>a</sub>bl<sub>e</sub> 2<sub>:</sub> Abl<sub>a</sub>ti<sub>on o</sub>f P<sub>rism-</sub>1<sub>: mean accuracy an</sub>d <sub>sw</sub>it<sub>c</sub>h <sub>ra</sub>t<sub>e</sub> when an option is added (%). Each row changes one more d<sub>es</sub>i<sub>gn c</sub>h<sub>o</sub>i<sub>ce, an</sub>d th<sub>e</sub> f<sub>our</sub>th <sub>row</sub> i<sub>s</sub> th<sub>e</sub> d<sub>es</sub>i<sub>gn o</sub>f S<sub>e</sub>t<sub>-</sub>LLM<sub>.</sub>
<table><tr><td rowspan="2">Variant</td><td colspan="2">Gemma 4 12B</td><td colspan="2">Qwen3-8B</td></tr><tr><td>Acc</td><td>Switch</td><td>Acc</td><td>Switch</td></tr><tr><td>PRISM-1</td><td>79.6</td><td></td><td>0.0 77.7</td><td>0.0</td></tr><tr><td>options attend to each other</td><td>74.4</td><td>3.5</td><td>74.8</td><td>1.3</td></tr><tr><td>+ score read after the set</td><td>76.4</td><td>3.8</td><td>72.0</td><td>3.4</td></tr><tr><td>+ likelihood of the option&#x27;s text</td><td>74.7</td><td>3.6</td><td>72.1</td><td>5.1</td></tr><tr><td>PRIsM-1 + answer slot</td><td>79.7</td><td>0.0</td><td>76.1</td><td>0.0</td></tr></table>

## 5.5 Ho<sub>w</sub> to Com<sub>p</sub>ute the Scores

Proposition 2.4 fixes the form of the read, not how the model should compute each option’s score. Three choices matter: whether the options attend to each other, where each option’s score is read, and the training loss. Table 2 changes them one at a time from Prism-1, with the same training data and one run each. With all three changed, the read is the design of Set-LLM [10].

Isolated options give accuracy and transfer. Letting the options attend to each other, with nothing else changed, costs 5.2 and 2.9 points (� < 0.001 pooled), mostly on BEV (6.4 and 6.0 points) and the held-out JevBench (11.5 and 5.0 points). A bidirectional prompt changes how every layer of the pretrained model computes. Untrained, a bidirectional set prompt answers 23% and 20% of our validation questions correctly, against 68% on both models for isolated causal blocks with the answer slot, as Figure 7a in the appendix shows. Fine-tuning repairs this on the training tasks, but not on tasks that rely on what the model knew before: trained, the bidirectional prompt stays within 2.2 points of trained SCQ on Banking20 and Newsgroups but trails it by 7 and 10 points on BEV and by 16 and 12 points on the held-out JevBench, as Figure 7b shows.

Reading each option gives the axiom. Reading the score from an answer generated after the set, instead of at each option, raises the switch rate on Qwen from 1.3% to 3.4%, while on Gemma it changes little, from 3.5% to 3.8%. With bidirectional attention, options influence each other even when each is read at its own end, and only isolated blocks bring the switch rate to zero.

![](images/4915fe59340dcad847a8114abf73ebb14c694a111ab45693955437f364c13804.jpg)  
Fi<sub>g</sub>ure 6: The <sub>p</sub>rice of the choice axiom: accurac<sub>y</sub> relative to Prism-1 a<sub>g</sub>ainst the answers an added o<sub>p</sub>tion moves to another ori<sub>g</sub>inal o<sub>p</sub>tion. Gre<sub>y</sub>: run-to-run noise<sub>,</sub> two standard d<sub>ev</sub>i<sub>a</sub>ti<sub>ons.</sub>

Loss and answer slot. Training on the likelihood of the correct option’s text instead of the cross-entropy over the options costs 1.7 points on Gemma and nothing on Qwen. Appending the answer slot to every block keeps both axioms, at about equal accuracy on Gemma (79.7% against 79.6%) and 1.6 points less on Qwen, with blocks several times longer.

## 5<sub>.</sub>6 Th<sub>e</sub> P<sub>r</sub>i<sub>ce o</sub>f th<sub>e</sub> Ch<sub>o</sub>i<sub>ce</sub> A<sub>x</sub>i<sub>om</sub>

By Proposition 2.4, a rule in which an option’s score depends on its rivals violates the choice axiom, and weighing near-miss distractors against each other is exactly what the axiom rules out. The price is small: on average Prism-1 is 0.9 points below SCQ on Gemma and level with it on Qwen, and the gaps are concentrated on the held-out JevBench and, on Gemma, on Typed.

We tried to buy comparison back, giving up the guarantee. Figure 6 shows the main mechanisms, and Table 4 in the appendix every variant. A gated channel through which every block attends to the other blocks’ last tokens in every layer, with gates starting at zero, leaves accuracy unchanged (−0.2 points on both models), and widening the channel or forcing the gates open does not help. Training leaves the channel nearly unused, with its gates at 4% of full strength on average, yet it breaks the axiom: an added option moves 0.7% and 1.0% of its answers. A comparison head that reads all final option states together adds 0.9 and 0.7 points, mostly on the held-out JevBench, but it learns not to compare irrelevant options: an added irrelevant option moves none of its answers, as for Prism-1 itself, so its gain comes from more capacity for each option’s score. Changes that keep the required form, namely auxiliary losses, borrowed distractors, joint training with SCQ, dividing out a content-free prior [52] and filler tokens [12], change the mean accuracy by −0.7 to +0.8 points, within run-to-run noise. None of the mechanisms that let options compare improves accuracy beyond noise: in our experiments, the choice axiom costs little or no accuracy.

## 6 R<sub>e</sub>l<sub>a</sub>t<sub>e</sub>d W<sub>or</sub>k

Choice theory. Luce’s choice axiom [23] characterises choice probabilities as ratios of option-specific scales, and underlies the multinomial logit model of discrete choice [25], the Plackett–Luce model of listwise learning to rank [5, 31] and, for two options, Bradley–Terry preference models [4]. Its best-known limitation concerns near-duplicate options, the red-bus/blue-bus problem [8], which nested and similarity-based models address [40]. Human choices also violate it through context efects, such as the attraction efect of an added, dominated option [16], and violate invariance through framing [41]. We use both as requirements for machine decisions, where consistency under changes of the listing and of the option set is the goal.

Option order in language models. Listed options sway answers through their order [30, 46, 56], partly through a prior over option labels [53], and the same holds for language models as evaluators [44]. Calibration divides out a shared prior [52, 53] but leaves each answer order-dependent. Fine-tuning on permuted orders reduces the bias [13], and averaging or voting over permutations reduces it further at a multiple of the cost [37, 51]. Neither guarantees invariance.

Invariant architectures. Permutation-invariant networks enforce invariance inside the model [27, 50]. For language models, Set-Based Prompting gives set elements shared positions and an isolating mask without training [26], and Set-LLM adds a bidirectional prompt and fine-tuning [10]. Both read the decision from tokens that see every option, so they are order-invariant but violate the choice axiom. PINE [45] and RoToR [49] reassign positions instead, after which accuracy can still vary with the order, and structural methods for listwise reranking follow the same idea [3, 34]. Sorting the options is a canonicalisation [19, 24, 32], and we show that it fixes the dependence on position into a convention and does not give the choice axiom.

Pointwise scoring. Pointwise rerankers score each candidate on its own, for example with the probability of a relevance token [28], and can be trained with a softmax over a candidate list [55]. Such scores have the form of Proposition 2.4. We show that for decisions this form is not one design among many but the only one that satisfies both axioms, and how to make it as accurate as SCQ.

Decision models. System One models return distributions over typed answers in one forward pass [1, 43], and their documentation notes that the order of choice options can afect the answer [42], which Table 1 measures for Jev and Mercury Decide [17]. Open recipes turn language models into such decision models [38, 51], and Prism-1 is a decision read for them that satisfies both axioms.

## 7 Limitations

Prism-1 needs fine-tuning: nothing in a block asks for an answer, so untrained it answers only 28.5% and 32.5% of our validation questions correctly. With the answer slot, the untrained read is usable but less accurate than SCQ (69.9% and 63.3% against 75.9% and 66.2%). By design it gives up comparison between options inside the model, so the axioms suit decisions in which each option can be judged on its own, such as routing, intent classification and tool or page-element selection. They fit less well where the answer is defined by the other options, as in “none of the above” or “the odd one out”. Prism-1 protects the choice among the options, not the state: text injected into the state reaches every option’s score, as it reaches every read. The choice axiom also decides how near-duplicates are treated: an added copy of an option receives about the same score and takes probability from every option in proportion, not mainly from its twin [8], so the copy and its twin share the twin’s probability. The original options keep their ranking, but the top probability drops, which merging duplicates before the read avoids. Taking it mainly from the twin would need the same selective shift that lets an added option steer the choice. Prism-1 trails SCQ on the held-out JevBench by 3.6 and 4.1 points, a task missing from our training data, and a read learned in fine-tuning may transfer less well than the letter read the models were instruction-tuned on. Our sets have at most 20 options. The guarantees are exact up to numerical rounding, which can tip near-ties when an option is added. We evaluate two open models of 8–12B parameters, with one to three seeds per configuration. Set-LLM is our re-implementation (its code is not public), trained on our data and budget.

## 8 C<sub>o</sub>n<sub>c</sub>l<sub>us</sub>i<sub>o</sub>n

A System One decision over a set should not depend on how the set is listed, and adding an irrelevant option should not change the relative odds of the others. Each standard read we test violates at least one of them. By Luce’s theorem, the axioms leave one form: a softmax over scores that each see one option. Prism-1 computes these scores by reading each option in its own block, in the model’s causal attention. Across five benchmark sets and two models, it satisfies both axioms, stays within one point of SCQ prompting trained on the same data and is more accurate once only answers that survive a reordering are counted, and is the cheapest read we measured. It gives up only comparison between options inside the model, which the axioms forbid and which, in our experiments, did not buy accuracy back.

## R<sub>e</sub>f<sub>erences</sub>

[1] Diogo Almeida. 2026. Introducing System One Models & Jev. TypeSafe AI blog, https://typesafe.ai/blog/introducing-system-one-models-and-jev. Published 15 September 2026

[2] avbiswas. 2026. BEV Decision Mix. https://huggingface.co/datasets/avbiswas/ bev-decision. Hugging Face dataset, default configuration, revision 11ed9b5. Accessed 2026-10-12.

[3] Ethan Bito, Yongli Ren, and Estrid He. 2026. One Pass, Any Order: Position-Invariant Listwise Reranking for LLM-Based Recommendation. In Proceedings ofthe 49th International ACM SIGIR Conference on Research and Development in Information Retrieval.

[4] Ralph Allan Bradley and Milton E. Terry. 1952. Rank Analysis of Incomplete Block Designs: I. The Method of Paired Comparisons. Biometrika 39, 3/4 (1952), 324–345.

[5] Zhe Cao, Tao Qin, Tie-Yan Liu, Ming-Feng Tsai, and Hang Li. 2007. Learning to Rank: From Pairwise Approach to Listwise Approach. In Proceedings ofthe 24th International Conference on Machine Learning. 129–136.

[6] Iñigo Casanueva, Tadas Temčinas, Daniela Gerz, Matthew Henderson, and Ivan Vulić. 2020. Eficient Intent Detection with Dual Sentence Encoders. In Proceedings ofthe 2nd Workshop on Natural Language Processing for Conversational AI. 38–45.

[7] Nick Craswell, Onno Zoeter, Michael Taylor, and Bill Ramsey. 2008. An Experimental Comparison of Click Position-Bias Models. In Proceedings ofthe 2008 International Conference on Web Search and Data Mining. 87–94.

[8] Gérard Debreu. 1960. Review of R. D. Luce, Individual Choice Behavior: A Theoretical Analysis. American Economic Review 50, 1 (1960), 186–188.

[9] Xiang Deng, Yu Gu, Boyuan Zheng, Shijie Chen, Samuel Stevens, Boshi Wang, Huan Sun, and Yu Su. 2023. Mind2Web: Towards a Generalist Agent for the Web. In Advances in Neural Information Processing Systems, Datasets and Benchmarks Track, Vol. 36.

[10] Beni Egressy and Jan Stühmer. 2025. Set-LLM: A Permutation-Invariant LLM. In Advances in Neural Information Processing Systems 38.

[11] Gemma Team. 2026. Gemma 4 12B (instruction-tuned). https://huggingface.co/ google/gemma-4-12B-it. Model card, revision 707f0a3. Accessed 2026-10-05.

[12] Sachin Goyal, Ziwei Ji, Ankit Singh Rawat, Aditya Krishna Menon, Sanjiv Kumar, and Vaishnavh Nagarajan. 2024. Think Before You Speak: Training Language Models With Pause Tokens. In International Conference on Learning Representations.

[13] Blessed Guda, Lawrence Francis, Gabrial Zencha Ashungafac, Carlee Joe-Wong, and Moise Busogi. 2025. Quantifying and Mitigating Selection Bias in LLMs: A Transferable LoRA Fine-Tuning and Eficient Majority Voting Approach. In Proceedings ofthe 14th International Joint Conference on Natural Language Processing and the 4th Conference ofthe Asia-Pacific Chapter ofthe Association for Computational Linguistics. 2019–2038.

[14] Yupeng Hou, Junjie Zhang, Zihan Lin, Hongyu Lu, Ruobing Xie, Julian McAuley, and Wayne Xin Zhao. 2024. Large Language Models are Zero-Shot Rankers for Recommender Systems. In Advances in Information Retrieval: 46th European Conference on Information Retrieval. 364–381.

[15] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2022. LoRA: Low-Rank Adaptation of Large Language Models. In International Conference on Learning Representations.

[16] Joel Huber, John W. Payne, and Christopher Puto. 1982. Adding Asymmetrically Dominated Alternatives: Violations of Regularity and the Similarity Hypothesis. Journal of Consumer Research 9, 1 (1982), 90–98.

[17] Inception. 2026. Mercury Decide. OpenRouter Decisions API, model inception/mercury-decide-20260930, https://openrouter.ai/inception/ mercury-decide-20260930. Released 30 September 2026. Accessed 2026-10-14.

[18] Thorsten Joachims, Laura Granka, Bing Pan, Helene Hembrooke, and Geri Gay. 2005. Accurately Interpreting Clickthrough Data as Implicit Feedback. In Proceedings ofthe 28th Annual International ACM SIGIR Conference on Research and Development in Information Retrieval. 154–161.

[19] Sékou-Oumar Kaba, Arnab Kumar Mondal, Yan Zhang, Yoshua Bengio, and Siamak Ravanbakhsh. 2023. Equivariance with Learned Canonicalization Func tions. In Proceedings of the 40th International Conference on Machine Learning. 15546–15566.

[20] Daniel Kahneman. 2011. Thinking, Fast and Slow. Farrar, Straus and Giroux.

[21] Ken Lang. 1995. NewsWeeder: Learning to Filter Netnews. In Proceedings of the Twelfth International Conference on Machine Learning. 331–339.

[22] LocalLLaMA. 2026. Typed Decisions. https://huggingface.co/datasets/ LocalLLaMA/typed-decisions. Hugging Face dataset. Accessed 2026-09-29

[23] R. Duncan Luce. 1959. Individual Choice Behavior: A Theoretical Analysis. Wiley.

[24] George Ma, Yifei Wang, Derek Lim, Stefanie Jegelka, and Yisen Wang. 2024. A Canonicalization Perspective on Invariant and Equivariant Learning. In Advances in Neural Information Processing Systems 37. 60936–60979.

[25] Daniel McFadden. 1974. Conditional Logit Analysis of Qualitative Choice Behavior. In Frontiers in Econometrics, Paul Zarembka (Ed.). Academic Press, 105–142.

[26] Reid McIlroy-Young, Katrina Brown, Conlan Olson, Linjun Zhang, and Cynthia Dwork. 2024. Order-Independence Without Fine Tuning. In Advances in Neural Information Processing Systems 37. Set-Based Prompting

[27] Ryan L. Murphy, Balasubramaniam Srinivasan, Vinayak A. Rao, and Bruno Ribeiro. 2019. Janossy Pooling: Learning Deep Permutation-Invariant Functions for Variable-Size Inputs. In 7th International Conference on Learning Representations.

[28] Rodrigo Nogueira, Zhiying Jiang, Ronak Pradeep, and Jimmy Lin. 2020. Document Ranking with a Pretrained Sequence-to-Sequence Model. In Findings of the Association for Computational Linguistics: EMNLP 2020. 708–718.

[29] Long Ouyang, Jef Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, and Ryan Lowe. 2022. Training Language Models to Follow Instructions with Human Feedback. In Advances in Neural Information Processing Systems, Vol. 35.

[30] Pouya Pezeshkpour and Estevam Hruschka. 2024. Large Language Models Sensitivity to the Order of Options in Multiple-Choice Questions. In Findings of the Association for Computational Linguistics: NAACL 2024. 2006–2017.

[31] R. L. Plackett. 1975. The Analysis of Permutations. Journal ofthe Royal Statistical Society: Series C (Applied Statistics) 24, 2 (1975), 193–202.

[32] Omri Puny, Matan Atzmon, Edward J. Smith, Ishan Misra, Aditya Grover, Heli Ben-Hamu, and Yaron Lipman. 2022. Frame Averaging for Invariant and Equi variant Network Design. In The Tenth International Conference on Learning Representations.

[33] Joshua Robinson and David Wingate. 2023. Leveraging Large Language Models for Multiple Choice Question Answering. In The Eleventh International Conference on Learning Representations.

[34] Ferdinand Schlatt, Maik Fröbe, Harrisen Scells, Shengyao Zhuang, Bevan Koopman, Guido Zuccon, Benno Stein, Martin Potthast, and Matthias Hagen. 2025. Set-Encoder: Permutation-Invariant Inter-Passage Attention for Listwise Pas sage Re-Ranking with Cross-Encoders. In Advances in Information Retrieval: 47th European Conference on Information Retrieval (ECIR 2025)

[35] Jiawen Shi, Zenghui Yuan, Guiyao Tie, Pan Zhou, Neil Zhenqiang Gong, and Lichao Sun. 2026. Prompt Injection Attack to Tool Selection in LLM Agents. In Network and Distributed S<sub>y</sub>stem Securit<sub>y</sub> S<sub>y</sub>mposium (NDSS).

[36] Florian Standhartinger. 2026. JevBench. https://github.com/fstandhartinger/ jevbench. Public split. Accessed 2026-10-05.

[37] Raphael Tang, Crystina Zhang, Xueguang Ma, Jimmy Lin, and Ferhan Ture. 2024. Found in the Middle: Permutation Self-Consistency Improves Listwise Ranking in Large Language Models. In Proceedings of the 2024 Conference of the North American Chapter ofthe Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers). 2327–2340.

[38] Torchcast AI. 2026. Torchcast Decision 12B. https://huggingface.co/torchcastai/torchcast-decision-12b. Model card, revision 49107b5. Accessed 2026-10-07.

[39] Torchcast AI. 2026. Torchcast Decision 12B: serving code. https://github. com/Torchcast-AI/torchcast-decision-12b. Tag v1.0.0 (commit d7f8446), serv ing/torchcast\_shim.py. Accessed 2026-10-12.

[40] Amos Tversky. 1972. Elimination by Aspects: A Theory of Choice. Psychological Review 79, 4 (1972), 281–299.

[41] Amos Tversky and Daniel Kahneman. 1986. Rational Choice and the Framing of Decisions. The Journal ofBusiness 59, 4 (1986), S251–S278.

[42] TypeSafe AI. 2026. Jev 1.13 Jaggedness. TypeSafe documentation, section “Choice option order”. Accessed 2026-10-04.

[43] TypeSafe AI. 2026. TypeSafe Documentation: Introduction and Primitives. https: //docs.typesafe.ai/introduction. Accessed 2026-09-29.

[44] Peiyi Wang, Lei Li, Liang Chen, Zefan Cai, Dawei Zhu, Binghuai Lin, Yunbo Cao, Lingpeng Kong, Qi Liu, Tianyu Liu, and Zhifang Sui. 2024. Large Language Models Are Not Fair Evaluators. In Proceedings ofthe 62nd Annual Meeting ofthe Association for Com utational Lin uistics (Volume 1: Lon Pa ers). 9440–9450.

[45] Ziqi Wang, Hanlin Zhang, Xiner Li, Kuan-Hao Huang, Chi Han, Shuiwang Ji, Sham M. Kakade, Hao Peng, and Heng Ji. 2025. Eliminating Position Bias of Language Models: A Mechanistic Approach. In The Thirteenth International Conference on Learning Representations.

[46] Sheng-Lun Wei, Cheng-Kuang Wu, Hen-Hsen Huang, and Hsin-Hsi Chen. 2024. Unveiling Selection Biases: Exploring Order and Token Sensitivity in Large Language Models. In Findings ofthe Association for Computational Linguistics: ACL 2024. 5598–5621.

[47] Fen Xia, Tie-Yan Liu, Jue Wang, Wensheng Zhang, and Hang Li. 2008. Listwise Approach to Learning to Rank: Theory and Algorithm. In Proceedings of the 25th International Conference on Machine Learning. 1192–1199.

[48] An Yang et al. 2025. Qwen3 Technical Report. arXiv:2505.09388

[49] Soyoung Yoon, Dongha Ahn, Youngwon Lee, Minkyu Jung, HyungJoo Jang, and Seung won Hwang. 2025. RoToR: Towards More Reliable Responses for Order-Invariant Inputs. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 18739–18760.

[50] Manzil Zaheer, Satwik Kottur, Siamak Ravanbakhsh, Barnabas Poczos, Ruslan Salakhutdinov, and Alexander Smola. 2017. Deep Sets. In Advances in Neural

Information Processing Systems 30.

[51] Jiamu Zhang, Tianze Yang, Yucheng Shi, Evan Chen, Zixiang Nie, Kelly Wan, Liangjie Hong, Ninghao Liu, and Liang Wu. 2026. AnyJev Technical Report. arXiv:2610.00831

[52] Zihao Zhao, Eric Wallace, Shi Feng, Dan Klein, and Sameer Singh. 2021. Calibrate Before Use: Improving Few-Shot Performance of Language Models. In Proceedings ofthe 38th International Conference on Machine Learning. 12697–12706.

[53] Chujie Zheng, Hao Zhou, Fandong Meng, Jie Zhou, and Minlie Huang. 2024. Large Language Models Are Not Robust Multiple Choice Selectors. In The Twelfth International Conference on Learning Representations.

[54] Shuyan Zhou, Frank F. Xu, Hao Zhu, Xuhui Zhou, Robert Lo, Abishek Sridhar, Xianyi Cheng, Tianyue Ou, Yonatan Bisk, Daniel Fried, Uri Alon, and Graham Neubig. 2024. WebArena: A Realistic Web Environment for Building Autonomous Agents. In International Conference on Learning Representations.

[55] Honglei Zhuang, Zhen Qin, Rolf Jagerman, Kai Hui, Ji Ma, Jing Lu, Jianmo Ni, Xuanhui Wang, and Michael Bendersky. 2023. RankT5: Fine-Tuning T5 for Text Ranking with Ranking Losses. In Proceedings ofthe 46th International ACM SIGIR Conference on Research and Development in Information Retrieval.

[56] Yongshuo Zong, Tingyang Yu, Ruchika Chavhan, Bingchen Zhao, and Timothy Hospedales. 2024. Fool Your (Vision and) Language Model with Embarrassingly Simple Permutations. In Proceedings ofthe 41st International Conference on Machine Learning.

## A P<sub>roo</sub>f<sub>s</sub>

Options come from a universe O, a set is a finite non-empty subset of O, and � gives every option of a set a positive probability. Within a proof we fix the context $( x , q )$ and leave it implicit unless it matters.

Ratios. For an order-invariant rule, the choice axiom holds if and only if the ratio of two options’ probabilities is the same in every set that contains both. If the axiom holds and $a , b \in S ,$ applying it to $\{ a , b \} \subseteq S$ for � and for � and dividing gives

$$
{ \frac { R ( a \mid S ) } { R ( b \mid S ) } } = { \frac { R ( a \mid \{ a , b \} ) } { R ( b \mid \{ a , b \} ) } } .\tag{4}
$$

Conversely, if every ratio is the same in every set, then for $S \subseteq T$ and $a \in S ,$

$$
{ \frac { R ( a \mid T ) } { R ( S \mid T ) } } = \biggl ( \sum _ { b \in S } { \frac { R ( b \mid T ) } { R ( a \mid T ) } } \biggr ) ^ { - 1 } = \biggl ( \sum _ { b \in S } { \frac { R ( b \mid S ) } { R ( a \mid S ) } } \biggr ) ^ { - 1 } = R ( a \mid S ) ,
$$

which is the axiom.

Proof of Proposition 2.3. If $R ( a \mid S ) = v ( a ) / \textstyle \sum _ { b \in S } v ( b )$ , the ratio $R ( a \mid S ) / R ( b \mid S ) = v ( a ) / v ( b )$ is the same in every set, so the axiom holds. Conversely, assume the axiom, fix a reference option �, and let $v ( r ) = 1$ and $v ( a ) = R ( a \mid \{ a , r \} ) / R ( r \mid \{ a , r \} ) > 0$ for $a \neq r .$ For options � and �, Equation (4) in the set $T = \{ a , b , r \}$ gives

$$
{ \frac { R ( a \mid \{ a , b \} ) } { R ( b \mid \{ a , b \} ) } } = { \frac { R ( a \mid T ) } { R ( b \mid T ) } } = { \frac { R ( a \mid T ) / R ( r \mid T ) } { R ( b \mid T ) / R ( r \mid T ) } } = { \frac { v ( a ) } { v ( b ) } } ,
$$

where the last step applies Equation (4) to {�, � } and $\{ b , r \}$ (the cases in which $^ { a , }$ � or � coincide are immediate). By Equation (4) again, within any set � the probabilities are proportional to �, and since they sum to one, $\begin{array} { r } { R ( a \bar { \mid } S ) = \boldsymbol { v } ( a ) / \sum _ { b \in S } \bar { \boldsymbol { v } ( b ) } } \end{array}$ □

Proof of Proposition 2.4. If. Let � have the form of Equation (1). The listing � enters the right-hand side only through the set � it lists, and a sum over � does not depend on the order of its terms, so $R ( \cdot \ \mid \ x , q , \sigma ) \ = \ R ( \cdot \ \mid \ x , q , \sigma ^ { \prime } )$ for any two listings of �: � is order-invariant. Its probabilities are positive, and $R ( a \mid S ) / R ( b \mid S ) = \exp ( s ( x , q , a ) - s ( x , q , b ) )$ is the same in every set that contains � and $^ { b , }$ so the choice axiom holds by the converse direction above.

Only if. Let � be positive and order-invariant, and let it satisfy the choice axiom in every context. Order invariance makes �(· | $x , q , \sigma )$ depend on � only through �, so in each context � is a rule over sets. Proposition 2.3, applied in each context, gives $R ( a \mid$ $x , q , S ) \ = \ v _ { x , q } ( a ) / { \sum _ { b \in S } v _ { x , q } ( b ) }$ with the scale constructed in its proof, $v _ { x , q } ( a ) { \overset { \cdot } { = } } R ( a \mid x , q , \{ a , r \} ) / R ( r \mid x , q , \{ a , r \} )$ , which depends on �, � and � alone, for a fixed reference option �. With $s ( x , q , o ) =$ log $v _ { x , q } ( o )$ , � has the form of Equation (1). The score is unique up to a term �(�, �) that the softmax cancels, since by Equation (4) every diference $s ( x , q , a ) - s ( x , q , b )$ is fixed by �. □

The softmax as a posterior. Equation (3) follows from a simple model. Suppose that every option $o _ { k }$ is correct independently, with probability $p _ { k } \in ( 0 , 1 )$ , and condition on the event � that exactly one option is correct. The probability that $o _ { k }$ alone is correct is $\textstyle p _ { k } \prod _ { j \neq k } \bigl ( 1 - p _ { j } \bigr )$ , and � (�) is the sum of these terms over $k ,$ so

$$
P ( o _ { k } \mid E ) = { \frac { p _ { k } \prod _ { j \neq k } ( 1 - { p _ { j } } ) } { \sum _ { i } { p _ { i } \prod _ { j \neq i } ( 1 - { p _ { j } } ) } } } = { \frac { { p _ { k } / ( 1 - { p _ { k } } ) } } { \sum _ { i } { p _ { i } / ( 1 - { p _ { i } } ) } } } ,
$$

dividing the numerator and the denominator by $\prod _ { j } ( 1 - p _ { j } ) > 0$ With $\ p _ { k } = P ( y e s \mid h , b _ { k } ) / ( P ( y e s \mid h , b _ { k } ) + P ( n o \mid h , \dot { b } _ { k } ) )$ , the odds $p _ { k } / ( 1 - p _ { k } ) = P ( y e s \mid h , b _ { k } ) / P ( n o \mid h , b _ { k } ) = \exp s ( o _ { k } )$ , and the probability that the model gives to every other token cancels. The posterior is therefore the softmax of the scores, and the training loss − log $R ( o _ { y } \mid S )$ is the negative log-likelihood of the correct option under this model, given �. Only ratios of odds enter: a bias shared by all options, a constant added to every score, changes no decision and no probability.

Prism-1 computes a score of the required form. In the packed forward pass of Section 4, the hidden state of every token of block $b _ { k } ,$ , in every layer, equals its state in a causal pass over the context ℎ followed by $b _ { k }$ alone, and so depends on $( x , q , o _ { k } )$ only. By induction over layers: the context attends only to itself, so its states depend on ℎ alone; a token of $b _ { k }$ starts from its embedding at position � plus its ofset in the block, which does not depend on the listing, and each attention layer combines its state with the keys and values of the context and of the earlier tokens of $b _ { k }$ , the only keys the mask allows, which by induction depend on ℎ and $b _ { k }$ alone. Masked keys receive an additive −∞, so they contribute exactly zero. The score read at the last token of $\dot { b } _ { k }$ is therefore a function $s ( x , q , o _ { k } )$ , and Proposition 2.4 gives both axioms.

## B Ex<sub>p</sub>erimental Details

T<sub>a</sub>bl<sub>e</sub> 3<sub>:</sub> Th<sub>e</sub> b<sub>enc</sub>h<sub>mar</sub>k<sub>.</sub> T<sub>ra</sub>i<sub>n:</sub> t<sub>ra</sub>i<sub>n</sub>i<sub>ng ques</sub>ti<sub>ons, a</sub>ft<sub>er</sub> h<sub>o</sub>ld<sub>-</sub> in<sub>g</sub> out 50 <sub>p</sub>er set for validation. State: len<sub>g</sub>th in characters<sub>,</sub> m<sub>e</sub>di<sub>a</sub>n / m<sub>ea</sub>n<sub>.</sub>
<table><tr><td>Set</td><td>K</td><td>Eval</td><td>Train</td><td>State (chars)</td></tr><tr><td>Banking20 [6]</td><td>20</td><td>300</td><td>2,450</td><td>55 / 71</td></tr><tr><td>Newsgroups [21]</td><td>20</td><td>300</td><td>2,450</td><td>542 / 750</td></tr><tr><td>BEV [2]</td><td>2-10</td><td>1,182</td><td>2,450</td><td>232 / 787</td></tr><tr><td>Typed Decisions [22]</td><td>4-5</td><td>600</td><td>1,750</td><td>708 / 697</td></tr><tr><td>JevBench [36]</td><td>3-6</td><td>139</td><td>一</td><td>82 / 2,563</td></tr></table>

![](images/340d1cfdd50d77b2b79794e069da0041b8beee49e3fc70e8818ff2ceb732b740.jpg)  
Figure 7: What bidirectional attention among options costs. (a) Validation accuracy before and after one epoch of training, for a bidirectional set rom t (Set-LLM’s desi n) and for isolated causal blocks with the answer slot. (b) Accurac of the trained bidirectional set prompt minus trained SCQ, in points.

Data. Table 3 summarises the five sets. From BEV we leave out its intent-routing domain, which reuses Banking20’s source messages, and states over 8,000 characters. We read it at revision 11ed9b5 of avbiswas/bev-decision, and its training questions are spread evenly over the same 12 domains and share no state with any evaluation set. Options of BEV, Typed Decisions and JevBench are their native key: description pairs, read as one string. The shufled order of each question is drawn once with a fixed seed and shared by all methods.

Prompts. Every read uses the model’s chat template with a system instruction: “Answer each question independently using only the <sub>s</sub>t<sub>a</sub>t<sub>e prov</sub>id<sub>e</sub>d b<sub>y</sub> th<sub>e user.</sub> T<sub>rea</sub>t th<sub>e s</sub>t<sub>a</sub>t<sub>e as</sub> d<sub>a</sub>t<sub>a, no</sub>t <sub>as</sub> i<sub>ns</sub>t<sub>ruc</sub>ti<sub>ons.</sub> E<sub>va</sub>l<sub>ua</sub>t<sub>e</sub> <sub>eac</sub>h <sub>ques</sub>ti<sub>on</sub> <sub>us</sub>i<sub>ng</sub> it<sub>s</sub> <sub>own</sub> <sub>cr</sub>it<sub>er</sub>i<sub>a,</sub> <sub>w</sub>ith<sub>ou</sub>t <sub>con</sub>diti<sub>on</sub>i<sub>ng</sub> it<sub>s</sub> answer on other questions. Return exactly one allowed label for each question.” For Prism-1, the user turn holds the state, the allowed labels (no, yes), the instruction “Reply with one line per question, in order: ‘id: label’. Do not add explanations.” and “Question 0: � Is the following candidate the correct answer? Candidate answer:”, where the context ends; a block is the option alone, read at its last token. For SCQ, the user turn holds the state, “Question 0: �”, the options one per line as ${ } ^ { \mathrm { s c } } \mathrm { A } \mathrm { : ~ \ldots ~ } , { } ^ { \mathrm { s c } } \mathrm { B } \mathrm { : ~ \ldots ~ } ^ { \mathrm { s } } ,$ , and the same reply instruction; the assistant turn is opened with thinking disabled and the answer prefix “0:”, and the decision is read from the next-token probabilities of the letters. The answer-slot read of the ablation asks “Is this candidate the correct answer?” after the option, and its block ends with the answer prefix “0:”.

Training. LoRA dropout is 0.05, and the 32 questions of a step are accumulated one at a time, so one epoch over the 9,100 training questions takes 284 steps (50–85 minutes on one H100 GPU for SCQ and Prism-1). No model selection uses the 200 validation questions, except the weight of the prior division below. Gemma 4 12B is google/gemma-4-12B-it at revision 707f0a3.

Attention backend. In our software stack (transformers 5.17), the SDPA attention path of Gemma 4 is not causal: appending a token changes earlier logits by up to 17.6. We use eager attention with additive masks for Gemma 4 and SDPA for Qwen3.

Baselines. Cyclic averaging reads SCQ on the � rotations of the listing and averages each option’s log-probability over them. Any-Jev’s prior correction, which divides each rotation’s distribution over letter positions by a prior over positions estimated per question key, raises the share of answers that a shufle changes from 2.3% and 3.0% to 4.3% and 6.3%, so we leave it out. Set-LLM: our re-implementation follows the paper: set positions that restart for every option and resume after the longest one, a prefix mask over the prompt without edges between diferent options, and the correct option’s text, followed by the end of the turn, as the training target. For evaluation, all answer branches are packed in one pass, each attending to the prompt and to itself, and agree with separate passes. Torchcast Decision 12B is served with its own pinned vLLM 0.30.0 image and serving code [39], with which we reproduce its published JevBench score (203 of 231), and we read its native choice answers.

Added options. For the choice axiom, the added option comes from another question of the same set whose key does not collide with an existing one (for Banking20 and Newsgroups, a banking intent outside Banking20’s 20). The drift test adds one such option per question at a random position of the source order; the steering test draws 10 candidates per question and, for SCQ and the published decision models, inserts each at the start, the middle and the end, and for the order-invariant reads at the end; for Prism-1 and cyclic averaging it uses the first 100 questions of each set, and for methods trained with several seeds the seed-0 model. Appended at the end, where it moves no other option’s letter, an added option steers SCQ only 2–3% of the time. A variant counts as steered if its answer is an original option other than the one chosen without it; questions or variants with an exact tie for the top are left out.

Mechanisms ofSection 5.6. Gated channel: in every layer, every block also attends to the last token of each other block (or to all their tokens, “wide”), and this attention’s contribution is scaled by a per-layer, per-head gate tanh(�), with � initialised at zero. Comparison head: one self-attention layer (width 256, 4 heads, no position encoding) over the options’ final hidden states and the context’s, whose output, initialised at zero, adds a correction to each score, over all options or the top five. Auxiliary loss: a per-option yes/no cross-entropy and a margin loss between the correct option and its strongest rival, each with weight 0.5. Borrowed distractors: with probability 0.5, a training question loses a random share of its wrong options and gains up to � options borrowed from other questions, half of its own set. Joint training: the SCQ loss on a shufled listing is added with weight 1. Answer first: the negative loglikelihood of the correct option’s text, generated from the context before any block, is added with weight 0.5. Prior division: each score minus � times its score with the content-free state “N/A” [52], � chosen on validation. Filler tokens: eight tokens “ .” after each option, before the read.

Table 4: The mechanisms of Section 5.6: accurac<sub>y</sub> minus Prism-1’s with the same seeds (points), overall and on JevBench (JB), and switch rate (Sw., %). S: seeds; –: not run.
<table><tr><td rowspan="2">Variant</td><td rowspan="2">S</td><td colspan="3">Gemma 4 12B</td><td colspan="3">Qwen3-8B</td></tr><tr><td>Acc</td><td>JB</td><td>Sw.</td><td>Acc</td><td></td><td>JB Sw.</td></tr><tr><td colspan="7">Options compare (not of the required form)</td><td></td></tr><tr><td>gated channel, rivals&#x27; last tokens</td><td></td><td>1-0.2</td><td>-1.4</td><td>0.7</td><td>-0.2</td><td>0.0</td><td>1.0</td></tr><tr><td>gated channel, all rival tokens</td><td></td><td>1 +0.1</td><td>+1.4</td><td>0.7</td><td></td><td></td><td></td></tr><tr><td>gated channel, gates forced open</td><td></td><td>1 -1.2</td><td>+1.4</td><td>1</td><td></td><td></td><td></td></tr><tr><td>comparison head, all options</td><td></td><td>2 +0.9</td><td>+3.2</td><td>0.0</td><td>+0.7</td><td>+3.6</td><td>0.0</td></tr><tr><td>comparison head, top 5</td><td></td><td>1 +0.1</td><td>-0.7</td><td>0.0</td><td>+0.4</td><td>+2.2</td><td>0.1</td></tr><tr><td colspan="7">Of the required form</td></tr><tr><td>auxiliary per-option and margin loss 1 +0.1</td><td></td><td></td><td>-0.7</td><td></td><td>-0.3</td><td>-1.4</td><td></td></tr><tr><td>borrowed distractors in training</td><td></td><td>1 -0.1</td><td>1 +0.7</td><td></td><td>-0.3</td><td>+0.7</td><td></td></tr><tr><td>joint training with SCQ</td><td></td><td>1-0.1</td><td>-2.2</td><td>一</td><td>+0.2</td><td>+0.7</td><td></td></tr><tr><td>answer predicted before the options</td><td></td><td>+0.8</td><td>+1.4</td><td>一</td><td>+0.1</td><td>+0.7</td><td></td></tr><tr><td>prior division (content-free state)</td><td></td><td>2 +0.1</td><td>0.0</td><td>一</td><td></td><td>+0.1 +1.1</td><td></td></tr><tr><td>eight filler tokens per option</td><td></td><td>2 +0.8</td><td>+1.4</td><td>一</td><td></td><td>-0.7-2.9</td><td></td></tr></table>

Table 5: Cost <sub>p</sub>er decision with 20 o<sub>p</sub>tions on Bankin<sub>g</sub>20 (B20) and Newsgroups (NG): tokens processed, attention pairs (query–key pairs the mask allows, thousands) and millisec-<sub>o</sub>nd<sub>s o</sub>n <sub>o</sub>n<sub>e</sub> H100 GPU<sub>,</sub> G<sub>e</sub>mm<sub>a</sub> 4 12B<sub>,</sub> m<sub>e</sub>di<sub>a</sub>n<sub>s ove</sub>r th<sub>e</sub> fir<sub>s</sub>t 100 <sub>ques</sub>ti<sub>ons</sub> <sub>o</sub>f <sub>eac</sub>h <sub>se</sub>t<sub>.</sub>
<table><tr><td></td><td colspan="2">Tokens</td><td colspan="2">Attn. pairs (k)</td><td colspan="2">ms</td></tr><tr><td>Method</td><td>B20</td><td>NG</td><td>B20</td><td>NG</td><td>B20</td><td>NG</td></tr><tr><td>SCQ</td><td>282</td><td>417</td><td>40</td><td>100</td><td>60</td><td>63</td></tr><tr><td>Set-LLM</td><td>322</td><td>413</td><td>60</td><td>168</td><td>68</td><td>66</td></tr><tr><td>PRISM-1</td><td>221</td><td>356</td><td>21</td><td>76</td><td>58</td><td>60</td></tr></table>

Significance. The comparison over all � rotations in Section 5 uses the seed-0 runs of SCQ and Prism-1, the only SCQ run read in every rotation. All tests are exact two-sided McNemar tests, per set or pooled over the 2,521 questions. Per set, SCQ is ahead on the held-out JevBench by 3.6 and 4.1 points, significant for one of six model–seed pairs, and on Typed for Gemma by 3.4 points, significant for two of three seeds, and Prism-1 on Banking20 and Newsgroups, significant only on Newsgroups for one Qwen seed (� = 0.03).

## C Additi<sub>o</sub>n<sub>a</sub>l R<sub>esu</sub>lt<sub>s</sub>

## C<sub>.</sub>1 H<sub>os</sub>t<sub>e</sub>d D<sub>ec</sub>i<sub>s</sub>i<sub>on</sub> M<sub>o</sub>d<sub>e</sub>l<sub>s</sub>

T<sub>a</sub>bl<sub>e</sub> 6<sub>:</sub> Th<sub>e c</sub>l<sub>ose</sub>d<sub>-source mo</sub>d<sub>e</sub>l<sub>s aga</sub>i<sub>ns</sub>t th<sub>e</sub>i<sub>r no</sub>i<sub>se</sub> fl<sub>oor,</sub> fi<sub>ve-se</sub>t <sub>means.</sub> Id<sub>en</sub>ti<sub>ca</sub>l <sub>co</sub> i<sub>es:</sub> th<sub>e same ues</sub>ti<sub>on re ea</sub>t<sub>e</sub>d in one re<sub>q</sub>uest<sub>,</sub> scored as the added-o<sub>p</sub>tion variants are. Sw.: switch rate (%).
<table><tr><td>Model</td><td>Flip</td><td>Dr.</td><td>St.</td><td>Sw.</td></tr><tr><td>Jev 1.13</td><td>4.7</td><td>2.9</td><td>10.9</td><td>3.4</td></tr><tr><td>identical copies</td><td>1.3</td><td>1.3</td><td>5.1</td><td>1.3</td></tr><tr><td>Mercury Decide</td><td>7.6</td><td>3.4</td><td>16.0</td><td>3.5</td></tr><tr><td>identical copies</td><td>0.7</td><td>0.5</td><td>1.5</td><td>0.7</td></tr></table>

Jev [1] and Mercury Decide [17] are commercial System One models whose APIs return a probability for every option of a choice question. We use Jev 1.13.0 through TypeSafe’s API and Mercury Decide’s release of 30 September 2026 through OpenRouter’s Decisions API, and send each question’s options in the listed order. Both services answer the questions of a request independently, which we verified, so we send all listings of a question in one request. Jev rounds probabilities to two decimals, so we take its reported choice as the answer. Neither API is deterministic: the identical question changes its answer in 1.3% (Jev) and 0.7% (Mercury Decide) of cases within a request. Table 6 therefore adds a noise floor, 31 identical copies of each question in one request, scored as the real variants are. Both models are about as accurate as trained SCQ on our sets (80.0% and 79.4%), and both violate both axioms well beyond this noise.

## C.2 Injected Options

Table 7: Injected options: answers (%) moved to the named option (Target) or to any other original option (Switched).
<table><tr><td></td><td colspan="2">Gemma 4 12B</td><td colspan="2">Qwen3-8B</td></tr><tr><td>Read</td><td>Target</td><td>Switched</td><td>Target</td><td>Switched</td></tr><tr><td>SCQ</td><td>23.1</td><td>25.3</td><td>9.6</td><td>13.0</td></tr><tr><td>SCQ + cyclic avg.</td><td>10.0</td><td>10.8</td><td>4.7</td><td>6.1</td></tr><tr><td>SCQ, sorted</td><td>11.2</td><td>13.0</td><td>2.9</td><td>5.1</td></tr><tr><td>Set-Based Prompting</td><td>54.8</td><td>58.5</td><td>31.4</td><td>41.8</td></tr><tr><td>Set-LLM</td><td>22.7</td><td>23.6</td><td>15.3</td><td>18.1</td></tr><tr><td>PRISM-1</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>Jev 1.13</td><td colspan="4">Target 3.9, Switched 9.3</td></tr><tr><td>Mercury Decide</td><td colspan="4">Target 7.3, Switched 14.1</td></tr></table>

Table 7 adds the other reads to Figure 4, and the answers moved to any other option, for the worst phrasing and position.

## C<sub>.</sub>3 C<sub>a</sub>libr<sub>a</sub>ti<sub>o</sub>n

Without any calibration step, Prism-1’s softmax has an expected calibration error of 5.3% and 5.4%, against 4.9% and 5.4% for SCQ and 9.3% and 7.5% for Set-LLM. At 95% accuracy it decides 61.5% and 51.4% of the questions, 3.1 and 2.9 points fewer than SCQ, but the same ones under every listing.