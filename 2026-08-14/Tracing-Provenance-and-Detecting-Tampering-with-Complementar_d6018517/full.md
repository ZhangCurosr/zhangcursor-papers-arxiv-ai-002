# Tracing Provenance and Detecting Tampering with Complementary LLM Watermarks

Xiaoyan Feng, Yanjun Zhang, He Zhang, Leo Yu Zhang, and Shirui Pan

Griffith University

Abstract—Watermarking LLM-generated text is an important task for tracing its provenance. Existing LLM watermarks preserve provenance under editing, but this same robustness allows an adversary to alter critical content while retaining attribution, a vulnerability known as piggyback spoofing. We introduce an innovative watermark that jointly provides provenance and tamper evidence. It co-embeds a robust signal and a fragile signal into each generated token. The signals share the same mechanism but use independent keys and different seeding windows over normalized text, making one resilient to edits and the other sensitive to reader-visible changes. Multiple rounds of unbiased tournament reweighting preserve the expected generation distribution, while a periodic round-allocation pattern controls the trade-off between the two signals. At detection, their scores form a two-dimensional space supporting three decisions: Intact, Tampered, and No-Watermark. Across two large language models and two prompt datasets, our method demonstrates the highest tamper-detection rate among the evaluated methods while maintaining competitive attribution robustness and perplexity. Ablation studies show that reliable three-state detection requires a well-defined notion of intactness, co-embedding of the two signals, and complementary sensitivity to edits.

Index Terms—large language models, text watermarking, provenance, tamper evidence

## I. INTRODUCTION

Regulations and industry commitments increasingly require AI-generated content to be identifiable [1]–[3]. Generative watermarking is a principled path to this requirement, tracing the provenance of LLM-generated text [4], [5]. It tilts the sampling probabilities toward tokens favored by a keyed pseudorandom signal, and later scores how strongly the received text follows this signal. The first threat to watermarking is removal, in which an attacker edits part of the watermarked text yet still evades provenance attribution. Existing schemes therefore keep the signal robust to edits to ensure reliable provenance [6], [7]. This robustness opens up a new attack surface. An adversary may alter critical details of watermarked text while keeping the signal strong, making the scheme endorse content the model did not generate. This attack is called piggyback spoofing [8]. Figure 2 illustrates both attacks on the same green-red watermark [4]. Countering piggyback spoofing requires tamper evidence. Tamper evidence calls for a signal sensitive to edits, which is the opposite of what provenance demands. Yet most schemes embed a single signal and thus face a dilemma.

Embedding two signals in one text decouples provenance and tamper evidence and promises both. A watermark scheme fulfilling this goal requires a tailored signal design, embedding mechanism, and detection rule. The closest attempt, Bileve [9], pays a price at every stage. It signs the first tokens of a generation and writes the signature bits into the following tokens. The carrier tokens are generated after the signature is computed and cannot enjoy its protection. At detection, its signature fails even on unedited text, because re-tokenization of the delivered text may shift the token IDs. Generation quality also degrades, with perplexity rising about sixfold, because the sampler takes the top-ranked token whose hash matches the signature bit.

![](images/81b552e6f74113e43932853526bfd9794cff6bf26c83f0593c8806398eb856fa.jpg)  
Fig. 1. Cocktail co-embeds two signals into one text: a robust signal and a fragile signal. A lightly edited text keeps a high robust score but drops to a low fragile score, making tamper evident while preserving provenance.

We propose Cocktail, which embeds two complementary signals that differ only in the key and the seeding window, achieving provenance and tamper evidence simultaneously. The signal for provenance is robust to edits via a short window, while the signal for tamper evidence is fragile to edits via a long window. The signals are seeded on normalized text to ensure they react to reader-visible edits but stay consistent during transmission. Both signals are co-embedded into each token, so that every token contributes to both demands. The co-embedding is implemented by vectorized tournament reweighting, with a periodic pattern controlling the signal strength ratio while maintaining generation quality. In detection, the two signal scores span a 2D plane where Intact, Tampered, and No-Watermark text occupy separable regions, as Figure 1 previews. A two-threshold rule then reads the three states directly off the plane. Our contributions are fourfold.

• We resolve the dilemma by decoupling: a robust and a fragile signal in one text serve provenance and tamper evidence separately, and their joint score turns the binary decision into a three-state decision.

![](images/5c2a91bb673e92a908c6e782a7abfca99d28c6cbd9561d22bc4b0ad2ec9f9706.jpg)  
Fig. 2. An illustrative example of removal and spoofing attacks on a green–red list watermark [4], with green-token z-score z and detection threshold τ. Token shading marks green-list or red-list membership under the secret key.

• We define tamper evidence over the content the reader receives rather than the token IDs that carry it, which lays the foundation for the three-state decision.

• We co-embed the two signals into every token through rounds of unbiased reweighting, costing neither generation quality nor detection efficiency, and provide a tunable strength ratio between the signals.

• Experiments show that Cocktail leads the strongest single-signal baseline by over 66 percentage points in tamper evidence without conceding attribution, and ablations trace this margin to each design choice above.

## II. RELATED WORK

## A. Generative Watermarks for LLM Text

Aaronson [10] first involved a keyed pseudorandom variable in token sampling, making the sampled token indices statistically correlated with it. KGW [4] instantiated the signal as a green-red partition of the vocabulary, boosting green tokens at sampling and scoring text by the z-score of its green-token count. Together they established the framework of inference-time embedding and detection. Dipper [11] finetuned T5-XXL [12] into a paraphraser and showed that embedded watermarks withstand paraphrasing better than passive detectors [13]. Later work pushes two directions: robustness to edits and generation quality.

Toward robustness, Unigram [6] fixes one global partition to maximize edit resilience; Kuditipudi et al. [14] align text to a pregenerated key sequence by Levenshtein distance; SIR [7], SemStamp [15], and further semantics-based schemes [16], [17] seed the signal on semantics to survive paraphrasing. The pursuit has a proven ceiling: strong watermarking against unbounded rewriting is impossible [18], and reliability degrades with the edit budget [19].

Toward generation quality, the distortion-free notion, formulated in [14] and independently in [20], [21], requires the expected token probability under the randomized signal to equal the original one. Unbiased [21] further shows that this preserves expected downstream performance provided the seeds do not repeat. DiPmark [22], SynthID [5], MCmark [23], and BiMark [24] design their own unbiased reweightings for signal embedding. BiMark and ENS [25] compose independent unbiased steps to amplify one signal, whereas Cocktail composes tournament rounds, whose accountable entropy consumption turns round allocation into a strength dial between two complementary signals.

## B. Piggyback Spoofing and Its Defenses

Piggyback spoofing [8] modifies a small number of words in an existing watermarked text and exploits the robustness of the signal so that the edited text retains a high signal strength, exposing a vulnerability in existing schemes. Bileve [9], the work closest to Cocktail, attempts to counter this attack. Bileve embeds a coarse-grained and a fine-grained signature, but the signature distorts the output to carry its bits and certifies token IDs rather than the text the reader receives. An et al. [26] tighten the semantic binding of the signal through a secondpass generation, yet it remains a single signal that still faces the conflict between robust attribution and spoofing resistance. Escaping this conflict calls for complementary signals, a principle realized in image watermarking by pairing a robust and a fragile watermark, under the same name [27]. Cocktail, named for mixing the two signals round by round into every token, realizes this principle for generative LLM text.

## III. PROBLEM SETUP

An autoregressive language model generates text token by token. At step t, it outputs a distribution $p ( \cdot \mid x _ { < t } )$ over the vocabulary V and samples the next token x . A generative watermark adjusts this distribution with a pseudorandom signal derived from a secret key and a seed. Detection receives only the text, recomputes the signal from the text and the key, and scores how strongly the text follows it.

Between generation and detection, the text passes through hands that may change it. The adversary knows the scheme but not the keys [28]. It receives watermarked text and edits it, aiming either to remove the watermark while keeping the main content, or to alter critical details without being noticed by detection. The detector holds the keys but no access to the language model.

Under this threat, detection faces two questions. Provenance asks whether the text originates from the watermarked model.

![](images/0469e8bde6dcf49665ac667357d1035f1988b5cd4a6bb9213ba417eec7f0c9bb.jpg)  
Fig. 3. Cocktail Overview. (1) Complementary Signals. Both signals are seeded on the normalized preceding text, with a short robust window of $h _ { r }$ tokens and a long fragile window of $n _ { f }$ characters. (2) Unbiased Signal Co-Embedding. Each step applies rounds of unbiased reweighting, allocated between the two signals, preserving the expected output distribution. (3) 2D Score Space. The two scores $z _ { r }$ and $z _ { f }$ place the three states in distinct regions, classified by a joint rule.

Tamper evidence asks whether the delivered text is exactly what the model generated.

## IV. METHOD

## A. Overview

Cocktail provides both provenance and tamper evidence, implemented through three mechanisms shown in Figure 3. The two goals are achieved by two complementary signals. The robust signal serves provenance, and the fragile signal serves tamper evidence. The hash input of both signals is the content the watermark is anchored to: the normalized text. The two signals enter the generation distribution of every token through unbiased co-embedding, an adaptive probability reweighting applied round by round. Each round of reweighting consumes a share of the collision entropy of the input distribution. We define a periodic rule for the embedding where the two signals are alternated to distribute the strength between them. The two signal scores span a two-dimensional score plane, where post-edit watermarked text decays fast along the fragile axis but slowly along the robust axis, and this asymmetry exposes it. Cocktail thereby supports a threestate decision.

## B. Complementary Watermark Signals

Cocktail embeds a robust signal and a fragile signal that serve the two complementary goals of provenance and tamper evidence. We call a signal robust or fragile by its sensitivity to post-edits. We characterize a watermark signal by its seed and its carrier. For generative watermarks, the carrier is what is sampled from the signal-adjusted distribution, and the seed is what a hash function takes to generate the signal.

a) Independent green-red list.: Concretely, each signal is a green-red list over the vocabulary, imprinted on the sampling probabilities of the tokens. We denote the robust signal as $g _ { r }$ and the fragile signal as $g _ { f } .$ . For a token $x _ { i } \in \mathcal V$ $g _ { r } [ i ] = 1$ means that the robust signal assigns this token green, and $g _ { r } [ i ] = 0$ means red. The fragile signal $g _ { f }$ follows the same form. Formally, at generation step t, each signal is a pseudorandom function of its key and its seed,

$$
g _ { * } ^ { ( t ) } = \mathrm { P R N G } \big ( k _ { * } , s _ { * } ( t ) \big ) \in \{ 0 , 1 \} ^ { | \mathcal { V } | } , \quad * \in \{ r , f \} ,\tag{1}
$$

where the key $k _ { * }$ identifies the signal and the seed $s _ { * } ( t )$ is specified below. Following prior work, the signal raises the generation probability of its green tokens and suppresses that of its red tokens. The two signals are generated under different keys, $k _ { r } \neq k _ { f }$ , and therefore mutually independent.

b) Short window versus long window.: Provenance requires a signal that survives edits, and tamper evidence requires a signal that breaks under them. The window length sets this sensitivity. An edit at position i corrupts the seed $s _ { * } ( t )$ at every $t \in [ i , ~ i + h ]$ for a window of length h. A short window confines the damage to a few tokens, and a long window spreads it far downstream. Existing schemes set the window between one and three tokens and thus sit at the robust end alone. We therefore assign the robust signal a short window $h _ { r }$ and the fragile signal a long window $n _ { f } .$ . The left panel of Figure 3 depicts how the two windows respond to an edit.

c) Seeded on normalized text.: In most previous work, token IDs serve as both the seed and the carrier. We keep tokens as the carrier yet take the normalized text as the seed. The normalized seed makes tamper evidence for watermarked text well-defined. Token ids are intermediates of generation, and the content delivered to the reader is the normalized text. Formally, with $\mathcal { N }$ the normalizer and H a hash function,

$$
\begin{array} { r } { s _ { r } ( t ) = H \big ( \mathcal { N } ( x _ { \operatorname* { m a x } ( 1 , t - h _ { r } ) : t - 1 } ) \big ) , } \\ { s _ { f } ( t ) = H \big ( \mathrm { s u f f i x } _ { n _ { f } } \big ( \mathcal { N } ( x _ { 1 : t - 1 } ) \big ) \big ) , } \end{array}\tag{2}
$$

where $\operatorname { s u f f i x } _ { n }$ keeps the last n characters and both windows truncate at the text start when the prefix is shorter than the window. Detection re-tokenizes the delivered text, and the drifted token IDs realign within a few tokens for a short window $h _ { r } .$ . The operator $\mathrm { s u f f i x } _ { n _ { f } }$ slides a character window over the growing normalized prefix. On intact text, the normalized suffix is the same string at generation and at detection, and the fragile signal thus reacts only to deliberate edits.

Specifically, our normalizer includes invisible-character cleanup, Unicode NFKC, homoglyph folding to canonical characters, and whitespace collapsing. These operations are idempotent and reproducible across environments. Replacing the normalizer only widens or narrows the range of perturbations within which the seeds stay reproducible, leaving the co-embedding design untouched. The analogy is a signed document, where the signature certifies the wording rather than the paper it is written on.

## C. Unbiased Signal Co-Embedding

To maximize both provenance and tamper evidence, both robust and fragile signals are expected to be embedded into each token. For practicality, the co-embedding aims to preserve generation quality, and keep the strength ratio between the two signals tunable.

a) Multiple round unbiased reweightings: . A token carries both signals when both signals shape its sampling probability. Cocktail therefore adjusts the distribution using the two signals in sequence, as shown in the middle panel of Figure 3. Repeated adjustment is where quality is at stake: two adjustments of a biased reweighting drift the distribution further from its expectation. Cocktail preserves the quality by using an unbiased reweighting by which the expected distribution is maintained even after multiple rounds adjustment.

Cocktail uses vectorized tournament, a closed form of the two-sample tournament sampling of SynthID [5]. Let $\hat { p } ^ { ( 0 ) } = p$ and let d denote the number of rounds. In round $i ,$ an independent green-red list $g ^ { ( i ) } [ x ] \sim$ Bernoulli(0.5) reweights the previous round’s output:

$$
\begin{array} { l } { { \hat { p } ^ { ( i ) } ( x ) = \left( 1 + g ^ { ( i ) } [ x ] - q ^ { ( i ) } \right) \hat { p } ^ { ( i - 1 ) } ( x ) , } } \\ { { q ^ { ( i ) } = \displaystyle \sum _ { x \in \mathcal { V } } g ^ { ( i ) } [ x ] \hat { p } ^ { ( i - 1 ) } ( x ) , } } \end{array}\tag{3}
$$

and the token is sampled from the final distribution $\hat { p } ^ { ( d ) }$ Here, $\hat { p } ^ { ( i ) }$ is the reweighted distribution, and $\boldsymbol { q } ^ { ( i ) }$ is the total probability of the green tokens under $\hat { p } ^ { ( i - 1 ) }$ . A green token with $g ^ { ( i ) } [ x ] = 1$ gains a fraction $1 - q ^ { ( i ) }$ of its probability, while a red token with $g ^ { ( i ) } [ x ] = 0$ loses a fraction $\boldsymbol { q } ^ { ( i ) }$ . Since $g ^ { ( i ) } [ x ] \sim$ Bernoulli(0.5), we have $\begin{array} { r } { \bar { \mathfrak { L } } \big [ g ^ { ( i ) } [ x ] \big ] = \mathbb { E } \big [ \bar { q } ^ { ( i ) } \big ] = \frac { 1 } { 2 } } \end{array}$ and each token’s expected adjustment factor is $\mathbb { E } \big [ 1 \dot { + } g ^ { ( \bar { i } ) } [ x ] \bar { - }$ $q ^ { ( i ) } \big | = 1$ , i.e., the reweighting is unbiased. Because the lists are independent across rounds, unbiasedness is preserved after all d rounds, i.e., $\mathbb { E } \big [ \hat { p } ^ { ( d ) } ( x ) \big ] = p ( x )$

b) Periodic pattern and strength ratio.: Vectorized tournament supports tuning the strength ratio between the two signals by allocating rounds in a periodic pattern. We assign each round to one signal according to a pattern $\pi = ( \pi _ { 1 } , \ldots , \pi _ { d } )$ with $\pi _ { i } \ \in \ \left\{ \mathrm { R O B U S T } , \mathrm { F R A G I L E } \right\}$ . Given a robust-to-fragile ratio (r:1), the pattern is periodic with period $r + 1$

$$
\pi _ { i } = \left\{ \begin{array} { l l } { \mathrm { F R A G I L E } } & { i \mathrm { ~ m o d ~ } ( r + 1 ) = 0 , } \\ { \mathrm { R O B U S T } } & { \mathrm { o t h e r w i s e } , } \end{array} \right. \quad \quad i = 1 , \dots , d ,\tag{4}
$$

Algorithm 1 Cocktail Watermark: Embedding   
Require: model M, prompt a, keys $k _ { r } , k _ { f }$ , number of rounds   
$d ,$ pattern π, short window $h _ { r }$ , fragile character window   
$n _ { f } .$ , length T, normalizer ${ \mathcal { N } } .$ , hash $H$   
1: $x \gets \emptyset$   
2: for $t = 1 , \dots , T$ do   
3: $\hat { p } \gets \mathsf { M } ( \cdot \mid a , x )$   
4: for $i = 1 , \ldots , d$ do   
5: $\ast  \pi _ { i } ; \ c  \ N ( x _ { \mathrm { m a x } ( 1 , t - h _ { r } ) : t - 1 } )$ if ∗=r else   
su $\mathrm { { f f i x } } _ { n _ { f } } ( \mathcal { N } ( x _ { 1 : t - 1 } ) )$   
6: $g ^ { ( i ) } \gets \mathrm { \ ' P R N G } ( k _ { * } , H ( c \| i ) )$   
7: $\hat { p }  ( 1 + g ^ { ( i ) } \dot { - } q ^ { ( i ) } ) \hat { p }$ ▷ Eq. (3)   
8: end for   
9: $x _ { t } \sim \hat { p } ;$ break if $x _ { t }$ is a stop token; $x  x \| x _ { t }$   
10: end for   
11: return x

Algorithm 2 Cocktail Watermark: Detection   
Require: text $\boldsymbol { x } ~ = ~ \left( x _ { 1 } , \ldots , x _ { T } \right)$ , keys $k _ { r } , k _ { f } .$ , number of   
rounds $d ,$ pattern $\pi ,$ thresholds $\tau _ { r } , \tau _ { f } .$ , short window $h _ { r } ,$   
fragile character window $n _ { f } ,$ , normalizer ${ \mathcal { N } } .$ , hash $H .$   
1: $N _ { r } , N _ { f } \gets 0$   
2: for $t \overset { \cdot } { = } 1 , \ldots , T$ do   
3: for $i = 1 , \ldots , d$ do   
4: ∗ $\mathbf { \Phi }  \pi _ { i } ; \mathbf { \Phi } \_ c \gets \mathcal { N } ( x _ { \mathrm { m a x } ( 1 , t - h _ { r } ) : t - 1 } )$ if ∗=r else   
sufix $_ { \cdot n _ { f } } \left( { \mathcal { N } } ( x _ { 1 : t - 1 } ) \right)$   
5: $g ^ { ( i ) } \gets \mathrm { \ " { P R N G } } \big ( k _ { * } , H ( c \parallel i ) \big ) ; ~ N _ { * } \mathrel { + } = g ^ { ( i ) } [ x _ { t } ]$   
6: end for   
7: end for   
8: $z _ { * } \gets ( N _ { * } - \frac { 1 } { 2 } T d _ { * } ) / \sqrt { T d _ { * } / 4 }$ for $* \in \{ r , f \} \quad \triangleright$ Eq. (5)   
9: return Intact if $z _ { r } { > } { \tau } _ { r } { \land { z } _ { f } } { > } { \tau } _ { f } { ; }$ Tampered $\mathbf { i f } \ z _ { r } > \tau _ { r } ;$ else   
No-Watermark

yielding $d _ { r } \ = \ | \{ i \ : \ \pi _ { i } \ = \ \mathrm { R o B U S T } \} |$ robust rounds and $\begin{array} { r } { d _ { f } = d - d _ { \ i } } \end{array}$ fragile rounds. In round i, the green-red list is drawn under the designated signal’s key with the round index folded into the hash, $g ^ { ( i ) } = \mathrm { P R N G } \big ( k _ { \pi _ { i } } , H ( c \| i ) \big )$ , where c is the seeding window content of that signal, and the lists stay independent across rounds even within one signal. Balanced alternation corresponds to the pattern (1:1).

The round ratio approximates the strength ratio because the collision entropy [29] of the initial distribution acts as a budget shared by the two signals. Each round spends a share of the remaining budget, and rounds allocated at (r:1) split the budget roughly at that ratio. The proxy is not exact, since the consumption per round is nonlinear and the strength does not grow linearly with the budget, yet the round ratio remains a monotone and practical knob for adjusting the strength ratio.

## D. 2D Score Space and Three-State Decision

Given a text $\boldsymbol { x } = ( x _ { 1 } , \dots , x _ { T } )$ , we recover the green-red list $g ^ { ( i ) }$ of each round from the hash function and the observed seed, and count the green hits of the rounds assigned to each signal. Under the null hypothesis of non-watermarked text, each hit is an independent Bernoulli(0.5) variable, and the count is normalized into a z-score,

$$
z _ { * } = \frac { \sum _ { t = 1 } ^ { T } \sum _ { i : \pi _ { i } = * } g ^ { ( i ) } [ x _ { t } ] - \frac { 1 } { 2 } T d _ { * } } { \sqrt { T d _ { * } / 4 } } , \quad * \in \{ r , f \} ,\tag{5}
$$

where $d _ { r }$ and $d _ { f }$ are the numbers of rounds assigned to the robust and the fragile signal, respectively.

The two scores span a 2D score space, and each axis carries one question. The robust score $z _ { r }$ measures provenance, and the fragile score $z _ { f }$ measures tamper evidence. A high $z _ { f }$ marks a text as intact on its own, but a low $z _ { f }$ is ambiguous between no-watermark text and text whose fragile seeds have been corrupted by edits, an ambiguity the robust axis resolves. The decision rule therefore reads $z _ { f }$ only after $z _ { r }$ has attributed the text, partitioning the space into three states,

$$
\mathsf { D e t } ( x ) = \left\{ \begin{array} { l l } { \mathrm { I n t a c t } } & { z _ { r } > \tau _ { r } \mathrm { a n d } z _ { f } > \tau _ { f } , } \\ { \mathrm { T a m p e r e d } } & { z _ { r } > \tau _ { r } \mathrm { a n d } z _ { f } \leq \tau _ { f } , } \\ { \mathrm { N o - W a t e r m a r k } } & { z _ { r } \leq \tau _ { r } . } \end{array} \right.\tag{6}
$$

where $\tau _ { r }$ and $\tau _ { f }$ are thresholds set for robust and fragile signal, respectively. Both intact and tampered texts are attributed to the watermarked LLM, while only intact text is free of tamper evidence. The ideal separation is as shown in the right panel of Figure 3: tampered text scores high on the robust axis and low on the fragile axis, intact text scores high on both, and no-watermark text on neither.

The three-state decision rests on the separability of the three types of text in the 2D space. When the fragile window is as short as the robust one, the two signals respond identically to edits, and the two scores carry the same information. The score plane then degenerates into one dimension, and no boundary separates Tampered from Intact. A fragile window far longer than the robust one is therefore necessary.

Algorithm 1 summarizes the embedding of Cocktail, and Algorithm 2 summarizes the detection. The two signals run through one shared pipeline and differ in the window, the key, and the assigned rounds alone.

## V. EXPERIMENTS

## A. Setup

We generate with Llama-3.2-1B [30] and Gemma-3-4B [31] on the realnewslike subset of C4 [32] and on LFQA [11], [33], producing 500 texts per model-dataset pair for each scheme. We compare against four inference-time watermarks. KGW [4] seeds a balanced green-red split on the previous token and biases green logits. Unigram [6] fixes one global balanced split. SynthID [5] seeds on the previous and self token and embeds an unbiased signal through tournament sampling. SIR [7] seeds on a semantic embedding of the prefix and adds a KGW-style bias. Cocktail uses $h _ { r } ~ = ~ 1$ a full-prefix fragile window, $d \ = \ 3 0$ rounds, and round ratios $d _ { r } { : } d _ { f } \in \{ 1 { : } 1 , 2 { : } 1 , 4 { : } 1 \}$ }. Ablation studies generate with Llama-3.2-1B on C4.

![](images/efe5eb89ec7c5e1cfff0b6770223427f4cb95d2ff8f218ac1f20522d62d92577.jpg)  
Fig. 4. Cocktail’s score plane $\left( z _ { r } , z _ { f } \right)$ on real generations (Llama-3.2-1B, C4). The thresholds $\tau _ { r } , \tau _ { f }$ are each calibrated at a 1% tail, and the three states read off directly.

## B. Evaluation Protocol

The two questions of the problem setup induce two discrimination tasks: (1) attribution separates watermarked text, possibly edited, from no-watermark text; (2) tamper evidence separates tampered watermarked text from intact watermarked text. The baselines take both tests with their single z-score, and Cocktail takes them with $z _ { r }$ for attribution and $z _ { f }$ for tamper evidence, reading $z _ { f }$ only after $z _ { r }$ has attributed the text. For each task we report the true positive rate at 1% false positive rate (TPR@1%FPR) over the first 300 tokens, thresholded at the 1% tail of the negative class. Figure 4 shows the calibrated thresholds on the score plane.

The TPR@1%FPR of each task is the complement of the attacker’s success rate. Attribution faces paraphrasing with Dipper [11] and round-trip translation with opus-mt [34]. Tamper evidence faces random token substitution and a sentiment flip, where gpt-oss:20b [35] flips the sentiment under a word budget and a classifier [36] verifies it. Perplexity (PPL) under a Mistral oracle [37] measures generation quality.

## C. Attribution and Tamper Evidence

a) Cocktail fulfills both goals at no quality cost.: On the attribution task in Table I, Cocktail attributes unattacked text at 99.7–100.0%, and under paraphrasing and round-trip translation its 4:1 variant stays within a few points of the strongest single-signal baseline while frequently leading outright. On the tamper-evidence task, every Cocktail variant flags 89.5– 100.0% of tampered texts at a 1% false-alarm rate, whereas no baseline exceeds 23.1%. Only Cocktail demonstrates over 92% recall on both goals. Cocktail’s perplexity stays inside the baseline range, confirming that unbiased reweighting preserves generation quality.

b) The more robust a single signal, the weaker its tamper evidence.: The baselines order themselves by how insensitive their seeds are to edits. SIR seeds on a semantic embedding built to survive token-level edits, and delivers 0.0% tamper evidence in half the settings. Unigram’s green list is fixed and context-independent by construction, and it stays below 17.2%. KGW is seeded on the preceding token and peaks at 23.1%. SynthID is the most direct control. It uses tournament reweighting but embeds the robust signal only, making it

TABLE I  
TWO-TASK EVALUATION ON LLAMA-3.2-1B AND GEMMA-3-4B. ATTRIBUTION REPORTS TPR@1%FPR FOR WATERMARKED VERSUS NO-WATERMARK TEXT, TAMPER EVIDENCE FOR TAMPERED VERSUS INTACT TEXT. OVERALL AVERAGES THE FIVE TASKS. BEST IN BOLD, SECOND UNDERLINED.
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Method</td><td colspan="3">Attribution ↑</td><td colspan="2">Tamper Evidence ↑</td><td rowspan="2">Overall ↑</td><td rowspan="2">PPL ↓</td></tr><tr><td>No Attack</td><td>Paraphrase</td><td>RT-Translation</td><td>Sentiment-Flip</td><td>Token-Substitution</td></tr><tr><td rowspan="7">C4 (Llama)</td><td>Unigram</td><td>99.8</td><td>98.2</td><td>78.0</td><td>3.3</td><td>0.0</td><td>55.9</td><td>9.7</td></tr><tr><td>KGW</td><td>100.0</td><td>86.0</td><td>76.4</td><td>5.2</td><td>6.2</td><td>54.8</td><td>8.9</td></tr><tr><td>SynthID</td><td>100.0</td><td>91.2</td><td>93.8</td><td>7.5</td><td>9.8</td><td>60.5</td><td>10.6</td></tr><tr><td>SIR</td><td>99.0</td><td>87.4</td><td>70.0</td><td>0.0</td><td>0.0</td><td>51.3</td><td>8.8</td></tr><tr><td>Cocktail 1:1</td><td>99.9</td><td>82.0</td><td>93.2</td><td>99.4</td><td>99.0</td><td>94.7</td><td>9.3</td></tr><tr><td>Cocktail 2:1</td><td>99.9</td><td>88.4</td><td>96.2</td><td>98.3</td><td>90.5</td><td>94.7</td><td>8.8</td></tr><tr><td>Cocktail 4:1</td><td>99.8</td><td>92.4</td><td>97.4</td><td>97.2</td><td>89.5</td><td>95.3</td><td>9.0</td></tr><tr><td rowspan="7">LFQA (Llama)</td><td>Unigram</td><td>100.0</td><td>90.6</td><td>94.2</td><td>17.2</td><td>6.0</td><td>61.6</td><td>7.1</td></tr><tr><td>KGW</td><td>100.0</td><td>78.2</td><td>75.8</td><td>22.2</td><td>12.2</td><td>57.7</td><td>7.9</td></tr><tr><td>SynthID</td><td>100.0</td><td>84.8</td><td>93.8</td><td>13.0</td><td>8.8</td><td>60.1</td><td>9.3</td></tr><tr><td>SIR</td><td>100.0</td><td>90.1</td><td>83.3</td><td>10.7</td><td>0.0</td><td>56.8</td><td>7.7</td></tr><tr><td>Cocktail 1:1</td><td>100.0</td><td>80.5</td><td>84.0</td><td>100.0</td><td>99.7</td><td>92.8</td><td>8.0</td></tr><tr><td>Cocktail 2:1</td><td>100.0</td><td>85.7</td><td>86.3</td><td>99.4</td><td>94.3</td><td>93.1</td><td>7.9</td></tr><tr><td>Cocktail 4:1</td><td>100.0</td><td>89.4</td><td>88.7</td><td>98.1</td><td>91.6</td><td>93.6</td><td>7.5</td></tr><tr><td rowspan="7">C4 (Gemma)</td><td>Unigram</td><td>100.0</td><td>95.2</td><td>96.2</td><td>1.8</td><td>2.7</td><td>59.2</td><td>9.0</td></tr><tr><td>KGW</td><td>100.0</td><td>86.0</td><td>90.6</td><td>2.4</td><td>7.7</td><td>57.3</td><td>8.6</td></tr><tr><td>SynthID</td><td>100.0</td><td>90.8</td><td>95.8</td><td>4.1</td><td>11.9</td><td>60.5</td><td>8.5</td></tr><tr><td>SIR</td><td>99.5</td><td>85.0</td><td>69.9</td><td>0.0</td><td>0.0</td><td>50.9</td><td>8.1</td></tr><tr><td>Cocktail 1:1</td><td>100.0</td><td>80.8</td><td>86.4</td><td>100.0</td><td>99.8</td><td>93.4</td><td>8.2</td></tr><tr><td>Cocktail 2:1</td><td>100.0</td><td>90.8</td><td>92.6</td><td>98.6</td><td>99.7</td><td>96.3</td><td>8.0</td></tr><tr><td>Cocktail 4:1</td><td>100.0</td><td>88.2</td><td>94.1</td><td>98.3</td><td>99.0</td><td>95.9</td><td>7.8</td></tr><tr><td rowspan="7">LFQA (Gemma)</td><td>Unigram</td><td>100.0</td><td>90.4</td><td>97.0</td><td>8.0</td><td>3.5</td><td>59.8</td><td>10.2</td></tr><tr><td>KGW</td><td>100.0</td><td>84.4</td><td>90.2</td><td>23.1</td><td>14.8</td><td>62.5</td><td>9.9</td></tr><tr><td>SynthID</td><td>100.0</td><td>86.4</td><td>94.9</td><td>6.0</td><td>3.3</td><td>58.1</td><td>9.7</td></tr><tr><td>SIR</td><td>99.6</td><td>79.8</td><td>71.4</td><td>2.7</td><td>0.0</td><td>50.7</td><td>9.2</td></tr><tr><td>Cocktail 1:1</td><td>99.7</td><td>78.9</td><td>84.6</td><td>98.9</td><td>98.1</td><td>92.0</td><td>8.4</td></tr><tr><td>Cocktail 2:1</td><td>100.0</td><td>80.6</td><td>87.2</td><td>97.4</td><td>96.5</td><td>92.3</td><td>9.3</td></tr><tr><td>Cocktail 4:1</td><td>100.0</td><td>83.1</td><td>90.3</td><td>95.7</td><td>94.2</td><td>92.7</td><td>8.1</td></tr></table>

Cocktail with the fragile window degenerated to the robust one, and its tamper evidence does not exceed 13.0%.

c) The round ratio dials the trade-off between the two goals.: Moving from 1:1 to 4:1 on C4/Llama raises paraphrase attribution from 82.0 to 92.4 while tamper evidence recedes from 99.4 to 97.2 under sentiment flip. The same monotone shift appears in every setting. The dial moves within a regime where both goals remain far above every baseline: Cocktail’s lowest tamper-evidence score of 89.5 still exceeds the best baseline of 23.1 by over 66 points.

## D. Ablating the Design Choices

a) Tamper evidence requires normalization at seeding time.: Tamper evidence is defined over the content the reader perceives. Table II compares three placements of the normalizer on the same Cocktail watermark under two edits invisible to readers. Homoglyph substitution [38] swaps 30% of eligible characters for lookalike codepoints, e.g., the Latin a (U+0061) for its Cyrillic twin (U+0430). The None placement seeds and detects the signals on the decoded text. The Detection placement seeds on the decoded text but detects on the normalized text. The Seeding placement seeds and detects on the normalized text. Without normalization, homoglyphs cut attribution to 33.6% and break the fragile signal on every text. The attributed texts are flagged as tampered. We report this flagged fraction as the false alarm on edited texts.

TABLE II  
ATTRIBUTION AND TAMPER EVIDENCE UNDER HOMOGLYPH SUBSTITUTION ACROSS THREE NORMALIZER PLACEMENTS.
<table><tr><td></td><td colspan="3">Normalized at</td></tr><tr><td></td><td>None</td><td>Detection</td><td>Seeding</td></tr><tr><td>Attribution ↑</td><td>33.6</td><td>99.6</td><td>100.0</td></tr><tr><td>False alarm (edited) ↓</td><td>33.6</td><td>38.3</td><td>1.0</td></tr><tr><td>Tamper recall ↑</td><td>99.8</td><td>43.5</td><td>98.3</td></tr></table>

Normalization at detection recovers the attribution but replaces part of the model’s original output. The fragile verification then fails on the replaced intact texts, whose $z _ { f }$ falls to the noise level, and tamper recall under 5% token substitution collapses from 99.8% to 43.5%. Normalizing before seeding gives both sides the same canonical form, and the edited texts enter detection as their intact counterparts. At a 1% false-alarm rate, tamper recall stays at 98.3%.

b) Co-embedding outperforms one signal per token.: To achieve the two goals, a natural alternative assigns each token wholly to one signal, keeping each signal’s expected total budget unchanged. Yet this assignment fails once the detector loses track of which signal each token carries. We therefore grant the variant its best detector and score every token under both signal keys. For each signal, half of the scored positions carry only the other signal’s noise. The intact $z _ { r }$ drops from 9.4 to 6.1, and matching the co-embedded evidence takes roughly twice the text. Attribution at $T { = } 5 0$ falls from 98.3% to 86.0%, and round-trip attribution at the standard length falls from 89.8% to 60.2%. Hence, co-embedding is inherent to achieving the two goals jointly.

c) Tamper evidence requires a long fragile window and attribution first.: Figure 5 sweeps the fragile window length on the score plane. When the fragile window shrinks to the robust one, the fragile signal survives the same edits as the robust one and Cocktail degenerates into SynthID. Intact and tampered text collapse onto one diagonal cloud. Lengthening the character window peels the tampered cloud off the intact one. Tamper evidence reaches 7.0% at $scriptstyle n _ { f } = 4 0$ and 88.0% at $n _ { f } = 2 0 0$ . Identifying tampering requires the text to be attributed to the watermarked LLM first. Tampered and nowatermark text both sit near zero on the $z _ { f }$ axis, and only the tampered text keeps a high $z _ { r }$ . Reading $z _ { f }$ only after $z _ { r }$ has attributed the text therefore separates the two.

## E. Comparison with Signature-Based Schemes

Bileve [9], the closest design to Cocktail, pairs a coarse statistical signal for provenance with a digital signature for integrity. Verification requires exact bit recovery, which exacts a price on what is signed and how the bits are carried. We run the official implementation with default parameters on Llama-3.2-1B and C4 to measure the price.

a) Signing does not certify the delivered text.: Bileve signs token IDs, and re-tokenization of the delivered text fails verification on 91.1% of unmodified texts. Verified on the stored token IDs instead, the same pipeline passes every text, attributing the failures to re-tokenization. Signature verification leaves no threshold to tune. The integrity channel operates at an effective false-alarm rate of 91.1%, against Cocktail’s calibrated 1%. The signed message also stops at the first 44 tokens, and the following 512 tokens merely carry the signature bits without enjoying its certification. Cocktail instead seeds the fragile signal on the normalized prefix, anchoring tamper evidence to the delivered text itself.

b) Carrying exact bits costs quality and compute.: To carry each bit intact, Bileve takes the top-ranked token whose hash matches it, overriding sampling even at lowentropy positions. Perplexity rises to 62.2, six times Cocktail’s 10.3 under the identical measurement protocol. Recovering the bits is equally expensive. The detector realigns the text against the key sequence through permutation tests, whereas Cocktail scores in closed form. The permutation test reports $p \ = \ ( c + 1 ) / ( n _ { \mathrm { r u n s } } + 1 )$ , and the default 20 permutations bound $p \geq 1 / 2 1$ , above our 1% operating point. Scoring one text takes 144 s even with our accelerated alignment kernel, and reaching 1% needs at least 99 permutations.

## VI. DISCUSSION AND CONCLUSION

A signal that survives edits cannot also expose them, leaving single-signal watermarks open to piggyback spoofing. Cocktail escapes this conflict by co-embedding a robust and a fragile signal into every token through unbiased reweighting, both seeded on the text the reader receives, and read jointly as a three-state decision. Across two models and two datasets, it flags 89.5 to 100% of tampered texts at a 1% false-alarm rate without sacrificing attribution or perplexity. Edits confined to the final tokens corrupt only the tail of the fragile seeds, and reading $z _ { f }$ over trailing segments is a natural extension we leave to future work.

![](images/9120cad94f3847ea0e4e7f32c5fa4a162bc157844f9cda105ead24e119944741.jpg)  
Fig. 5. Score planes under four fragile windows. Intact and tampered text separate on $z _ { f }$ only as the window grows.

Besides piggyback spoofing, a threat to watermarks is stealing, which forges the signal by inferring the green list through queries [39]. The cost of this inference grows with the seeding window [40], [41]. Cocktail’s short robust window shares this exposure with existing robust schemes, whereas the fragile signal seeds on the full normalized prefix and remains hard to forge. The signal that supplies tamper evidence is thus the harder one to steal, leaving forgeries flagged as Tampered, and the recipe of two complementary signals may extend to other generative modalities where provenance and tamper evidence both matter.

## APPENDIX AUNBIASEDNESS AND ENTROPY CONSUMPTION OFTOURNAMENT REWEIGHTING

Cocktail introduces no new theory. The unbiasedness and budget claims of the main paper rest on theorems proved in the Supplementary Information of SynthID [5]. To keep this appendix self-contained, the statements below reproduce those theorems and their supporting definitions with the original numbering. Throughout, “Methods Algorithm $2 ^ { \circ }$ and “Methods Algorithm $3 ^ { \circ }$ refer to the algorithms of those names in SynthID [5], and V denotes the vocabulary.

## A. Notation restated from SynthID

The following definitions appear in Supplementary Appendices E, G.1, G.2, and H.1 of SynthID [5]. Here R is the space of random seeds, ∆V is the probability simplex over $\nu ,$ and $g _ { \ell } ( x , r )$ is the layer-ℓ pseudorandom g-value of token x under seed r (Methods Definition 4 of SynthID [5]), whose marginal distribution is $f _ { g } .$

Definition 9 (Watermarked distribution). Given a probability distribution p over V, a random seed $r \in \mathcal { R } ,$ , a number of samples $N \ \geq \ 2 ,$ , a g-value distribution $f _ { g } ,$ , and a number of layers $m \ \geq \ 1$ , the watermarked distribution $p _ { w m } ( \cdot \textrm { \textmu }$ $p , r , f _ { g } , N , m )$ is the probability distribution of the winner of Methods Algorithm 2:

$$
\begin{array} { r l } & { p _ { w m } ( x _ { t } \mid p , r , f _ { g } , N , m ) } \\ & { \phantom { p m } = \mathbb { P } \left[ \mathrm { A l g 2 } ( p , r , f _ { g } , N , m ) \mathrm { ~ r e t u r n s ~ } x _ { t } \right] . } \end{array}\tag{7}
$$

Definition 16 (Single-token non-distortionary sampling algorithm). A sampling algorithm $\mathcal { S } : \Delta \mathcal { V } \times \mathcal { R }  \mathcal { V }$ is (singletoken) non-distortionary if for any probability distribution $p \in \Delta \mathcal { V }$ and token $x \in \mathcal { V } ;$

$$
\mathbb { E } _ { r \sim \mathrm { U n i f } ( \mathcal { R } ) } \left[ \mathbb { P } \left( S ( p , r ) = x \right) \right] = p ( x ) .\tag{8}
$$

If S is not non-distortionary, we call it distortionary.

In Definition 20 the notation $\mathbb { P } _ { w m } \big ( \mathbf { y } ^ { i } \mid \mathbf { x } ^ { i } , k ; \cdot \big )$ denotes the probability that the watermarking scheme with key k generates response $\mathbf { y } ^ { i }$ to prompt $\mathbf { x } ^ { i }$ conditioned on the earlier prompt and response pairs $( \mathbf { \dot { x } } ^ { 1 } , \mathbf { y } ^ { 1 } ) , \dots , ( \mathbf { x } ^ { i - 1 } , \mathbf { y } ^ { i - 1 } )$ , and $\nu ^ { * }$ is the set of all finite sequences in V.

Definition 20 (K-sequence non-distortionary watermarking scheme). A watermarking scheme $\mathbb { P } _ { w m }$ is K-sequence nondistortionary for some $K \geq 1$ if, for any sequence of K prompts $\mathbf { x } ^ { 1 } , \ldots , \mathbf { x } ^ { K } \in \ \mathcal { V } ^ { * }$ and sequence of K responses $\mathbf { \bar { y } } ^ { 1 } , \ldots , \mathbf { y } ^ { K } \in \mathcal { V } ^ { * } .$

$$
\begin{array} { r l } & { \mathbb { E } _ { k \sim \mathrm { U n i f } ( \mathcal { R } ) } \Big [ \underset { i = 1 } { \overset { K } { \prod } } \mathbb { P } _ { w m } \big ( \mathbf { y } ^ { i } \mid \mathbf { x } ^ { i } , k ; ( \mathbf { x } ^ { 1 } , \mathbf { y } ^ { 1 } ) , \dots , ( \mathbf { x } ^ { i - 1 } , \mathbf { y } ^ { i - 1 } ) \big ) \Big ] } \\ & { \qquad = \underset { i = 1 } { \overset { K } { \prod } } p _ { L M } ( \mathbf { y } ^ { i } \mid \mathbf { x } ^ { i } ) . } \end{array}\tag{9}
$$

Definition 22 (Collision probability). Given a probability distribution p, the collision probability $C _ { p }$ of p is the probability that two samples drawn i.i.d. from p are the same. $\textit { I f p } = \ : ( p _ { i } ) _ { i = 1 } ^ { N }$ is discrete, the collision probability equals $\textstyle \sum _ { i = 1 } ^ { N } p _ { i } ^ { 2 }$

The SynthID supplement adds that collision probability is related to collision entropy, sometimes called Renyi entropy,´ $\begin{array} { r } { H _ { 2 } ( p ) = - \log { \sum _ { i = 1 } ^ { N } p _ { i } ^ { 2 } } } \end{array}$

Definition 23 (Higher-order collision probabilities). Given a probability distribution p and integers $N , j ~ \geq ~ 1$ , let ${ C } _ { p } ^ { N , j }$ denote the probability that N samples drawn i.i.d. from p have exactly j unique values. Note that $C _ { n } ^ { 2 , 1 }$ is the collision probability of p. In general, we refer to $C _ { p } ^ { N , j }$ as the higherorder collision probabilities of p.

Definition 24 (Watermarked g-value distribution). Given a probability distribution p, a g-value distribution $f _ { g } ,$ and number of samples $N \geq 2 ,$ let $F _ { g w }$ denote the cumulative density function ofthe g-value ofa token sampledfrom the single-layer watermarked distribution $p _ { w m } ( \cdot \mid p , r , f _ { g } , N , 1 )$ (Definition 9), in expectation over the random seed r:

$$
F _ { g w } ( z ) : = \mathbb { P } _ { r \sim \mathrm { U n i f } ( \mathcal { R } ) , x \sim p _ { w m } ( \cdot \vert p , r , f _ { g } , N , 1 ) } \left[ g _ { 1 } ( x , r ) \le z \right] .\tag{10}
$$

Let $f _ { g w }$ denote the probability density/mass function corresponding to $F _ { g w }$ . We refer to $f _ { g w }$ as the watermarked g-value distribution.

## B. Unbiasedness of multi-round reweighting

Theorems 17 and 18 below appear in Supplementary Appendix G.1 of SynthID [5] and Theorem 21 in its $\mathsf { A p - }$ pendix G.2.

Theorem 17 (Single-layer two-sample Tournament sampling is non-distortionary). For any probability distribution p over V, g-value distribution $f _ { g } ,$ and token $x _ { t } \in \mathcal V .$

$$
\mathbb { E } _ { \boldsymbol { r } _ { t } \sim \mathrm { U n i f } ( \mathcal { R } ) } \left[ p _ { w m } ( x _ { t } \mid p , \boldsymbol { r } _ { t } , f _ { g } , 2 , 1 ) \right] = p ( x _ { t } ) .\tag{11}
$$

Theorem 18 (Multi-layer two-sample Tournament sampling is non-distortionary). For any probability distribution p over V, g-value distribution $f _ { g } ,$ number of layers $m \geq 1$ , and token $x _ { t } \in \mathcal V .$

$$
\mathbb { E } _ { r _ { t } \sim \mathrm { U n i f } ( \mathcal { R } ) } \left[ p _ { w m } ( x _ { t } \mid p , r _ { t } , f _ { g } , 2 , m ) \right] = p ( x _ { t } ) .\tag{12}
$$

Theorem 21 (K-sequence repeated context masking + non-distortionary sampling algorithm → K-sequence nondistortionary watermarking scheme). Let S be a nondistortionary sampling algorithm (Def 16). For any $K \geq 1$ let $\mathbb { P } _ { w m }$ denote the watermarking scheme that applies S with sliding window random seed generation and K-sequence repeated context masking (Methods Algorithm 3). Then $\mathbb { P } _ { w m }$ is K-sequence non-distortionary.

Our vectorized tournament, main-paper Eq. 3, is the closed form of the same two-sample tournament with a Bernoulli(0.5) list, and our rounds satisfy the independence premise of Theorem 18: round i draws its list as $g ^ { ( i ) } \ =$ $\mathrm { P R N G } ( k _ { \pi _ { i } } , H ( c \| i ) )$ , and distinct round indices and keys yield independent lists under the pseudorandomness of H. Our embedding and detector skip a token whenever the context hash of either window repeats within the generation, which is the repeated context masking that Theorem 21 pairs with a non-distortionary sampler for the sequence-level guarantee (the K = 1 instantiation, i.e., single-sequence non-distortion), a requirement also standard in unbiased watermarking [21]. Unbiasedness therefore follows.

## C. Entropy consumption across rounds

The three theorems below appear in Supplementary Appendices H.3 and H.4 of SynthID [5], where the LLM distribution is written $p _ { L M } .$

Theorem 29 (g-value bias increases with N, single-layer tournament). Given a probability distribution $p _ { L M }$ and gvalue distribution $f _ { g } ,$ let $F _ { g w } ^ { N }$ be the c.d.f. of the watermarked g-value distribution for a single-layer tournament with N samples. Let $F _ { g w } ^ { N + 1 }$ be the same for a single-layer tournament with $N + 1$ samples. Then for all z:

$$
F _ { g w } ^ { N + 1 } ( z ) \leq F _ { g w } ^ { N } ( z ) .\tag{13}
$$

When $0 < F _ { g w } ^ { N } ( z ) < 1$ , equality holds iff p<sub>LM</sub> is one-hot.

Theorem 31 (Expected collision probability for single-layer tournament, two samples). Given a probability distribution $p _ { L M } ,$ , random seed $r \in \mathcal { R }$ and g-value distribution $f _ { g } ,$ let $C _ { p _ { w m } } ^ { 2 , 1 }$ denote the collision probability of the watermarked distribution $p _ { w m } ( \cdot  { | }  { p } _ { L M } , r , f _ { g } , 2 , 1 )$ for a $N \ = \ 2$ sample single-layer tournament. In expectation over the random seed r, the collision probability is:

$$
\begin{array} { r l } & { \mathbb { E } _ { r \sim \mathrm { U n i f } ( \mathcal { R } ) } \left[ C _ { p _ { w m } } ^ { 2 , 1 } \right] = \left[ \frac { 4 } { 3 } - \frac { 1 } { 3 } C _ { f _ { g } } ^ { 3 , 1 } \right] C _ { p _ { L M } } ^ { 2 , 1 } } \\ & { \phantom { \frac { 1 } { 1 } } + \left[ \frac { 2 } { 3 } + \frac { 1 } { 3 } C _ { f _ { g } } ^ { 3 , 1 } - C _ { f _ { g } } ^ { 2 , 1 } \right] \left( C _ { p _ { L M } } ^ { 2 , 1 } \right) ^ { 2 } } \\ & { \phantom { \frac { 1 } { 1 } } - \left[ \frac { 2 } { 3 } - \frac { 2 } { 3 } C _ { f _ { g } } ^ { 3 , 1 } \right] C _ { p _ { L M } } ^ { 3 , 1 } } \\ & { \phantom { \frac { 1 } { 1 } } - \left[ \frac { 1 } { 3 } + \frac { 2 } { 3 } C _ { f _ { g } } ^ { 3 , 1 } - C _ { f _ { g } } ^ { 2 , 1 } \right] C _ { p _ { L M } } ^ { 4 , 1 } . } \end{array}\tag{14}
$$

where $C _ { p _ { L M } } ^ { N , j }$ and $C _ { f _ { q } } ^ { N , j }$ are the higher order collision probabilities $( \overbrace { D e f } 2 3 ) ,$ , respectively, of $p _ { L M }$ and $f _ { g } .$

Theorem 32 (Single-layer tournament increases the expected collision probability, two samples). The expected collision probability of a single-layer tournament with $N = 2$ samples is greater than or equal to the LLM collision probability: $\begin{array} { r } { \mathbb { E } _ { r \sim \mathrm { U n i f } ( \mathcal { R } ) } [ C _ { p _ { w m } } ^ { 2 , 1 } ] \ \stackrel {  } { \ge } \ C _ { p _ { L M } } ^ { 2 , 1 } , } \end{array}$ , with equality iff p<sub>LM</sub> is onehot.

Supplementary Appendix H.4 of SynthID [5] draws the multilayer consequence: applied layer after layer, the tournament produces distributions whose expected collision probability keeps rising, each new layer therefore contributes less watermarking strength than the one before, and adding layers can yield diminishing returns.

Read in our notation, the collision entropy of $\hat { p } ^ { ( i ) }$ decreases monotonically in $i ,$ and every round spends a share of the remaining budget. This grounds the round-allocation dial of the main paper: rounds assigned at a ratio (r:1) split the shrinking budget between the two signals roughly at that ratio, and the ratio acts as a monotone knob rather than an exact linear control.

## APPENDIX B IMPLEMENTATION DETAILS

a) Computing infrastructure.: All experiments run on a single NVIDIA RTX 4090 Laptop GPU with 16 GB of VRAM, an Intel Core i9-14900HX CPU, and 64 GB of RAM, under Windows 11 Pro. The implementation uses Python 3.10, PyTorch 2.10 with CUDA 12.6, and transformers 4.57. Baselines run through the released code: the SynthID tournament reuses the hashing kernel of the official synthid-text release, and SIR loads the transform network and token mapping from its official repository.

b) Generation and detection configuration.: Each scheme generates 500 texts of up to 512 tokens per modeldataset pair with pure sampling at temperature 1.0 over the top 100 logits. C4 prompts are the first 100 words of realnewslike documents, streamed from the validation split shuffled with seed 42 after skipping 5000 documents, and LFQA ships its prompts. Token sampling is not seeded.

TABLE III  
HYPERPARAMETERS OF ALL METHODS AND EVALUATIONS.
<table><tr><td>Item</td><td>Value</td></tr><tr><td>Cocktail</td><td></td></tr><tr><td>Robust window  $h _ { r }$ </td><td>1 preceding token + self token</td></tr><tr><td>Fragile window  $n _ { f }$ </td><td>full normalized prefix + self token</td></tr><tr><td>Rounds d</td><td>30</td></tr><tr><td>Round ratios  $d _ { r } { : } d _ { f }$ </td><td> $1 { : } 1 , 2 { : } 1 , 4 { : } 1$ </td></tr><tr><td>Fragile seed</td><td>HMAC-SHA256 over the prefix</td></tr><tr><td>Robust seed and layers</td><td>SynthID accumulate_hash</td></tr><tr><td>Baselines</td><td></td></tr><tr><td>KGW</td><td> $\gamma = 0 . 5 , \delta = 2 . 0 ,$  window 1</td></tr><tr><td>Unigram</td><td> $\gamma = 0 . 5 , \delta = 2 . 0 ,$  global list</td></tr><tr><td>SynthID</td><td> $m = 3 0$  layers</td></tr><tr><td>SIR</td><td>1 preceding token + self token compositional-bert-large,  $\delta = 1 . 0$ </td></tr><tr><td>Attacks</td><td></td></tr><tr><td>Paraphrase (Dipper)</td><td>lex 20/60 with order 0</td></tr><tr><td></td><td>order 20/60 with lex 0</td></tr><tr><td>Round-trip translation Token substitution</td><td>opus-mt en→fr→en</td></tr><tr><td>Sentiment flip</td><td> $\bar { \rho } \in \{ 5 \% , 1 0 \% \}$  , mean reported</td></tr><tr><td></td><td>gpt-oss:20b, word budgets 5–30%</td></tr><tr><td>Sentiment judge</td><td>twitter-roberta-base-sentiment</td></tr><tr><td>NLI filter Homoglyph substitution</td><td>DeBERTa-xlarge-MNLI 30% of eligible characters</td></tr><tr><td></td><td></td></tr><tr><td>Quality</td><td></td></tr><tr><td>PPL oracle</td><td>Mistral-7B-v0.1, first 200 tokens</td></tr></table>

Each run records its configuration in generation params.json, detection reads it back, and scoring the released texts is exactly reproducible. Detectors score the first 300 tokens of the delivered text, and the sentiment-flip evaluation scores 200 because its inputs are truncated to 200 words before rewriting. Thresholds are calibrated per setting: $\tau _ { r }$ at the 99th percentile of no-watermark $z _ { r } ,$ and $\tau _ { f }$ at the 1st percentile of intact $z _ { f }$

<table><tr><td>Invisible strip</td><td>wa termark U+200B</td><td>→</td><td>watermark</td></tr><tr><td>NFKC</td><td>final U+FB01</td><td>final →</td><td></td></tr><tr><td>Confusable fold</td><td>watermark U+0430</td><td>→</td><td>watermark</td></tr><tr><td>Casefold</td><td>The mark</td><td>→</td><td>the mark</td></tr><tr><td>Whitespace fold</td><td> $\mathsf { i c e } _ { \mathsf { i \_ i s o a m } }$ </td><td>→</td><td>ice cream</td></tr></table>

Fig. 6. Table IV’s examples. Red marks the folded codepoint, and dashed boxes mark characters print cannot reveal.

c) Normalizer specification.: The normalizer strips zerowidth and invisible codepoints, applies Unicode NFKC, folds visual confusables to ASCII prototypes via the UTS 39 skeleton and a fixed Cyrillic and Greek to Latin override table, casefolds, and collapses whitespace to single spaces, in that order. Table IV pairs each operation with the surface channel it cancels, and Figure 6 renders the same examples with their true glyphs. Each operation is idempotent, and the composition is deterministic across platforms. An edit that escapes the normalizer changes the normalized text itself, and detection reports it as tampering rather than passing it silently.

![](images/3a7efa45c70f8ac9ec160e75271dd8aff4a862a15fdb0eb08f637ee2d5b4cb4b.jpg)  
Fig. 7. One Gemma-3-4B generation under a sentiment-flip and a homoglyph attack. Orange boxes mark word substitutions. Blue boxes mark homoglyph swaps, printed as the corresponding Latin characters. Seeding on normalized text leaves the scores unchanged, while seeding on raw tokens drops z to 1.81.

TABLE IVNORMALIZER OPERATIONS.
<table><tr><td>Operation</td><td>Cancels</td><td>Example</td></tr><tr><td>Invisible strip</td><td>zero-width chars</td><td> $\mathtt { a } [ \mathtt { U } + 2 0 0 \mathtt { B } ] \mathtt { b } \to \mathtt { a } \mathtt { b }$ </td></tr><tr><td>NFKC</td><td>compatibility forms</td><td> $[ \mathrm { U } + \mathrm { F B } 0 1 ]  \mathrm { f i }$ </td></tr><tr><td>Confusable fold</td><td>homoglyphs</td><td> $\left[ \mathsf { U } + 0 4 3 0 \right] \to \mathsf { a }$ </td></tr><tr><td>Casefold</td><td>case changes</td><td>The → the</td></tr><tr><td>Whitespace fold</td><td>spacing edits</td><td>[U+00A0] → space</td></tr></table>

## APPENDIX C

## BILEVE MEASUREMENT PROTOCOL

We run the official Bileve implementation with its default parameters: ECDSA over P-256, message length d = 44 tokens, carrier length $m = 5 1 2$ tokens, alignment block $n = 8 0$ The evaluation covers 500 intact watermarked texts and 500 no-watermark texts on Llama-3.2-1B with C4 prompts.

a) Signature channel.: Verified on the delivered text after re-tokenization, the signature passes on 8.9% of intact texts. Verified on the stored token IDs, the same pipeline passes on 100% of intact and 0% of no-watermark, tampered, and paraphrased texts. The failures therefore stem from retokenization, not from our reproduction.

b) Detection cost.: One pass aligns the 433 length-80 windows of the 512-token carrier against the key sequence at 80 candidate offsets, about $3 . 5 \times 1 0 ^ { 4 }$ Levenshtein alignments. The permutation p-value at the default $n _ { \mathrm { r u n s } } = 2 0$ runs this pass once with the true key and 20 more times with random keys, and every channel whose signature fails to verify pays the full 21 passes. We compile the official Cython alignment kernel to native code and replace its per-call 41 MB key-matrix copy with zero-copy views, leaving the alignment count itself as the remaining cost. The 144 s per text reported in the main paper is measured after these optimizations.

## APPENDIX D

## EXAMPLES OF COCKTAIL-WATERMARKED TEXT

Figure 7 shows the delivered text, the edits, and the scores $\left( z _ { r } , z _ { f } \right)$ with the three-state decision, for one Gemma-3-4B generation on a C4 prompt, scored over the first 200 tokens. Nine word edits out of 200, confirmed by the NLI filter as a positive to negative flip, leave the robust score nearly unchanged and collapse the fragile score, the score pattern of piggyback spoofing that the joint rule catches. The homoglyph attack swaps 81 of the 314 eligible characters for lookalike codepoints and is invisible to the reader. Seeded on raw tokens, the attacked text scores $( z _ { r } = 1 . 8 1 , z _ { f } = - 1 . 2 3 )$ and loses attribution. Seeded on normalized text, the identical attack becomes ineffective.

## REFERENCES

[1] European Union, “Artificial intelligence act,” Regulation (EU) 2024/1689, 2024.

[2] California State Legislature, “California AI transparency act,” Senate Bill No. 942, Chapter 291, Statutes of 2024, 2024.

[3] The White House, “Fact sheet: Biden-harris administration secures voluntary commitments from leading artificial intelligence companies to manage the risks posed by AI,” https://bidenwhitehouse.archives.gov/, 2023, accessed: 2026-07-29.

[4] J. Kirchenbauer, J. Geiping, Y. Wen, J. Katz, I. Miers, and T. Goldstein, “A watermark for large language models,” in International conference on machine learning. PMLR, 2023, pp. 17 061–17 084.

[5] S. Dathathri, A. See, S. Ghaisas, P.-S. Huang, R. McAdam, J. Welbl, V. Bachani, A. Kaskasoli, R. Stanforth, T. Matejovicova et al., “Scalable watermarking for identifying large language model outputs,” Nature, vol. 634, no. 8035, pp. 818–823, 2024.

[6] X. Zhao, P. V. Ananth, L. Li, and Y.-X. Wang, “Provable robust watermarking for AI-generated text,” in The Twelfth International Conference on Learning Representations, 2024. [Online]. Available: https://openreview.net/forum?id=SsmT8aO45L

[7] A. Liu, L. Pan, X. Hu, S. Meng, and L. Wen, “A semantic invariant robust watermark for large language models,” in International Conference on Learning Representations, vol. 2024, 2024, pp. 6499–6519.

[8] Q. Pang, S. Hu, W. Zheng, and V. Smith, “No free lunch in LLM watermarking: Trade-offs in watermarking design choices,” in The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024. [Online]. Available: https://openreview.net/forum?id= rIOl7KbSkv

[9] T. Zhou, X. Zhao, X. Xu, and S. Ren, “Bileve: Securing text provenance in large language models against spoofing with bi-level signature,” in The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024. [Online]. Available: https://openreview.net/forum?id=vjCFnYTg67

[10] S. Aaronson, “Watermarking of large language models,” Talk at the Simons Institute Workshop on Large Language Models and Transformers, https://simons.berkeley.edu/talks/scott-aaronson-utaustin-openai-2023-08-17, 2023, accessed: 2026-07-29.

[11] K. Krishna, Y. Song, M. Karpinska, J. Wieting, and M. Iyyer, “Paraphrasing evades detectors of ai-generated text, but retrieval is an effective defense,” Advances in neural information processing systems, vol. 36, pp. 27 469–27 500, 2023.

[12] C. Raffel, N. Shazeer, A. Roberts, K. Lee, S. Narang, M. Matena, Y. Zhou, W. Li, and P. J. Liu, “Exploring the limits of transfer learning with a unified text-to-text transformer,” J. Mach. Learn. Res., vol. 21, no. 1, Jan. 2020.

[13] E. Mitchell, Y. Lee, A. Khazatsky, C. D. Manning, and C. Finn, “Detectgpt: zero-shot machine-generated text detection using probability curvature,” in Proceedings of the 40th International Conference on Machine Learning, ser. ICML’23. JMLR.org, 2023.

[14] R. Kuditipudi, J. Thickstun, T. Hashimoto, and P. Liang, “Robust distortion-free watermarks for language models,” Transactions on Machine Learning Research, 2024. [Online]. Available: https:// openreview.net/forum?id=FpaCL1MO2C

[15] A. Hou, J. Zhang, T. He, Y. Wang, Y.-S. Chuang, H. Wang, L. Shen, B. Van Durme, D. Khashabi, and Y. Tsvetkov, “SemStamp: A semantic watermark with paraphrastic robustness for text generation,” in Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers). Association for Computational Linguistics, Jun. 2024, pp. 4067–4082. [Online]. Available: https://aclanthology.org/2024.naacl-long.226/

[16] J. Ren, H. Xu, Y. Liu, Y. Cui, S. Wang, D. Yin, and J. Tang, “A robust semantics-based watermark for large language model against paraphrasing,” in Findings of the Association for Computational Linguistics: NAACL 2024. Mexico City, Mexico: Association for Computational Linguistics, Jun. 2024, pp. 613–625. [Online]. Available: https://aclanthology.org/2024.findings-naacl.40/

[17] Y. Liu and Y. Bu, “Adaptive text watermark for large language models,” in Proceedings of the 41st International Conference on Machine Learning, ser. ICML’24. JMLR.org, 2024.

[18] H. Zhang, B. L. Edelman, D. Francati, D. Venturi, G. Ateniese, and B. Barak, “Watermarks in the sand: impossibility of strong watermarking for language models,” in Proceedings of the 41st International Conference on Machine Learning, ser. ICML’24. JMLR.org, 2024.

[19] J. Kirchenbauer, J. Geiping, Y. Wen, M. Shu, K. Saifullah, K. Kong, K. Fernando, A. Saha, M. Goldblum, and T. Goldstein, “On the reliability of watermarks for large language models,” in The Twelfth International Conference on Learning Representations, 2024. [Online]. Available: https://openreview.net/forum?id=DEJIDCmWOz

[20] M. Christ, S. Gunn, and O. Zamir, “Undetectable watermarks for language models,” in Proceedings of Thirty Seventh Conference on Learning Theory, ser. Proceedings of Machine Learning Research, vol. 247. PMLR, 30 Jun–03 Jul 2024, pp. 1125–1139. [Online]. Available: https://proceedings.mlr.press/v247/christ24a.html

[21] Z. Hu, L. Chen, X. Wu, Y. Wu, H. Zhang, and H. Huang, “Unbiased watermark for large language models,” in The Twelfth International Conference on Learning Representations, 2024. [Online]. Available: https://openreview.net/forum?id=uWVC5FVidc

[22] Y. Wu, Z. Hu, J. Guo, H. Zhang, and H. Huang, “A resilient and accessible distribution-preserving watermark for large language models,” in Proceedings of the 41st International Conference on Machine Learning, ser. ICML’24. JMLR.org, 2024.

[23] R. Chen, Y. Wu, J. Guo, and H. Huang, “Improved unbiased watermark for large language models,” in Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). Association for Computational Linguistics, Jul. 2025, pp. 20 587–20 601. [Online]. Available: https://aclanthology.org/2025.acllong.1005/

[24] X. Feng, H. Zhang, Y. Zhang, L. Y. Zhang, and S. Pan, “Bimark: Unbiased multilayer watermarking for large language models,” arXiv preprint arXiv:2506.21602, 2025.

[25] Y. Wu, R. Chen, G. Milis, and H. Huang, “An ensemble framework for unbiased language model watermarking,” in The Fourteenth International Conference on Learning Representations, 2026. [Online]. Available: https://openreview.net/forum?id=iZ7i2y1YxO

[26] L. An, Y. Liu, Y. Liu, Y. Zhang, Y. Bu, and S. Chang, “Defending LLM watermarking against spoofing attacks with contrastive representation learning,” in Second Conference on Language Modeling, 2025. [Online]. Available: https://openreview.net/forum?id=n5hmtkdl7k

[27] C.-S. Lu, S.-K. Huang, C.-J. Sze, and H.-Y. M. Liao, “Cocktail water-

marking for digital image protection,” IEEE Transactions on Multimedia, vol. 2, no. 4, pp. 209–224, 2000.

[28] F. Cayre, C. Fontaine, and T. Furon, “Watermarking security: theory and practice,” IEEE Transactions on Signal Processing, vol. 53, no. 10, pp. 3976–3987, 2005.

[29] T. M. Cover and J. A. Thomas, Elements ofInformation Theory, 2nd ed. Wiley-Interscience, 2006.

[30] A. Grattafiori, A. Dubey et al., “The llama 3 herd of models,” 2024. [Online]. Available: https://arxiv.org/abs/2407.21783

[31] G. Team, T. Mesnard et al., “Gemma: Open models based on gemini research and technology,” 2024. [Online]. Available: https://arxiv.org/abs/2403.08295

[32] C. Raffel, N. Shazeer, A. Roberts, K. Lee, S. Narang, M. Matena, Y. Zhou, W. Li, and P. J. Liu, “Exploring the limits of transfer learning with a unified text-to-text transformer,” Journal of machine learning research, vol. 21, no. 140, pp. 1–67, 2020.

[33] A. Fan, Y. Jernite, E. Perez, D. Grangier, J. Weston, and M. Auli, “ELI5: Long form question answering,” in Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics. Association for Computational Linguistics, Jul. 2019, pp. 3558–3567. [Online]. Available: https://aclanthology.org/P19-1346/

[34] J. Tiedemann and S. Thottingal, “OPUS-MT – building open translation services for the world,” in Proceedings of the 22nd Annual Conference ofthe European Associationfor Machine Translation. Lisboa, Portugal: European Association for Machine Translation, Nov. 2020, pp. 479–480. [Online]. Available: https://aclanthology.org/2020.eamt-1.61/

[35] OpenAI, “gpt-oss-120b & gpt-oss-20b model card,” 2025. [Online]. Available: https://arxiv.org/abs/2508.10925

[36] J. Camacho-Collados, K. Rezaee, T. Riahi, A. Ushio, D. Loureiro, D. Antypas, J. Boisson, L. Espinosa-Anke, F. Liu, E. Mart´ınez-Camara,´ G. Medina, T. Buhrmann, L. Neves, and F. Barbieri, “TweetNLP: Cutting-edge natural language processing for social media,” in Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing: System Demonstrations. Abu Dhabi, UAE: Association for Computational Linguistics, Dec. 2022, pp. 38–49. [Online]. Available: https://aclanthology.org/2022.emnlp-demos.5/

[37] A. Q. Jiang, A. Sablayrolles et al., “Mistral 7b,” 2023. [Online]. Available: https://arxiv.org/abs/2310.06825

[38] Z. Zhang, X. Zhang, Y. Zhang, H. Zhang, S. Pan, B. Liu, A. Q. Gill, and L. Zhang, “Character-level perturbations disrupt llm watermarks,” in Proceedings 2026 Network and Distributed System Security Symposium. Internet Society, 2026. [Online]. Available: http://dx.doi.org/10.14722/ndss.2026.230138

[39] N. Jovanovic, R. Staab, and M. Vechev, “Watermark stealing in large´ language models,” in Proceedings of the 41st International Conference on Machine Learning, ser. ICML’24. JMLR.org, 2024.

[40] A. Liu, L. Pan, X. Hu, S. Li, L. Wen, I. King, and P. S. Yu, “An unforgeable publicly verifiable watermark for large language models,” in The Twelfth International Conference on Learning Representations, 2024. [Online]. Available: https://openreview.net/ forum?id=gMLQwKDY3N

[41] H. Shen, B. Huang, and X. Wan, “Enhancing LLM watermark resilience against both scrubbing and spoofing attacks,” in The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. [Online]. Available: https://openreview.net/forum?id=RbdLnwEEjk