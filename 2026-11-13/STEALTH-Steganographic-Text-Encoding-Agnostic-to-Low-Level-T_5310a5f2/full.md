# STEALTH: Steganographic Text Encoding Agnostic to Low-Level Tensor Hardware

William Fleshman

Army Cyber Institute

United States Military Academy

West Point, New York 10996

william.fleshman@westpoint.edu

Abstract—Provably secure linguistic steganography hides a message inside text sampled from a language model, so that without the key, a stegotext is computationally indistinguishable from an honest model sample. Nearly every such scheme assumes the receiver can recompute the sender’s token probabilities exactly bit for bit. We show this assumption is fragile: across three model families, bit-exact schemes decode almost none of our messages once the receiver’s floating-point precision, quantization, or GPU architecture differs from the sender’s, or even when the receiver merely scores a text in parallel versus one token at a time, with the same model, precision, and hardware held fixed. We present STEALTH, a scheme built for exactly this setting. The sender draws each token with an exponential race driven by keyed pseudorandom noise, so a receiver’s probability mismatch flips a token only when the race is close; a rateless spinal code, decoded by beam search, absorbs the resulting flips without the sender and receiver ever agreeing on a single distribution. We prove stegotexts remain indistinguishable from honest samples under this construction. Empirically, STEALTH decodes every message we sent for every receiver in the sender’s model family, at every precision and quantization level, on smaller models, and on a different GPU architecture. STEALTH is even robust across different model families with separate vocabularies and tokenizers, decoding every message we tested. Our approach carries far more payload per token than prior robust schemes, and trained steganalyzers and a targeted statistical test both fail to distinguish its stegotexts from honest model output.

## 1. Introduction

State and corporate surveillance of digital communication is pervasive [1], and censorship regimes increasingly flag or block traffic that looks encrypted or otherwise anomalous [2]. While encrypted communications maintain privacy, they are easily identified. A useful alternative is covert channels: hiding messages inside communication that looks benign, so there is nothing anomalous for a censor or surveillant to flag. Steganography is the general technique, embedding secret messages into ordinary media such as images or text so the resulting cover object appears unremarkable to anyone but the intended recipient (Section 2).

Ideally, a steganographic technique should be provably secure, indistinguishable from the cover channel even to a warden who can run the embedding algorithm itself, rather than merely undetected by today’s tools [3].

Achieving this for images can be difficult since the true statistics of natural images are hard to model exactly [4]. Modern language models (LMs) make text more tractable because a good LM provides an explicit probability distribution over realistic text continuations. Linguistic steganography uses this distribution to hide messages: the sender uses the message bits to choose each sampled token, and the receiver, recomputing the same distribution, reads the bits back from the tokens that are observed. Several such schemes are provably secure [5–7], but only if the receiver reproduces the sender’s distribution bit for bit, since the message is encoded in exactly where each sampled token falls within it (Section 2). This bit-exact requirement is harder to meet than it sounds: floating-point reductions are not associative, so different tensor hardware, kernels, precision, or quantization yield different results for the same model and text, and even identical hardware can disagree between generating text token by token and scoring it in a single pass [8–10]. In practice, sender and receiver probabilities almost always differ, and because decoding replays the sender’s computation step by step, a single token whose probability differs in the last few bits can send every subsequent step down the wrong path. A recent survey of linguistic steganography lists this computational indeterminism as an open challenge [11].

We design STEALTH to tolerate this mismatch while remaining provably secure (Figure 1). The sender picks each token by an exponential race over the token distribution $p ,$ choosing the token that minimizes $E _ { i } / p _ { i } .$ , where $E _ { i }$ is keyed pseudorandom exponential noise. With truly random noise, this samples exactly from $p ;$ with keyed noise, it is indistinguishable from doing so [12, 13]. If the receiver’s distribution $p ^ { \prime }$ differs from $p ,$ the same noise picks a different token with probability bounded by a simple function of the total variation distance between the two distributions. The receiver scores the cover text one position at a time using the true preceding tokens as context (teacher forcing), so a flip affects only its own position and never the context of later tokens. We design the noise at each position to depend on the message through a spinal code [14], a hash chain over message chunks that the receiver decodes by an efficient beam search with a soft likelihood treating a narrowly lost race as a probable flip rather than a contradiction. Because the code is rateless, every additional token the model writes contributes more evidence toward the payload, so the sender never needs to truncate the text to fit a fixed message length. Positions in the text are measured in byte offsets, and the noise at each position can be keyed by the token’s bytes rather than its vocabulary index, so nothing the receiver computes depends on the sender’s tokenizer. This lets the receiver decode with an LM from an entirely different model family, with its own vocabulary and tokenizer.

![](images/613f7c26fed9371f43326fd7ba3e8f7a03613208b9e49c16fa59cb71eb469c16.jpg)  
Figure 1. Differences in sender and receiver token distributions cause the receiver to lose sync at a single token, desyncing bit-exact decoding schemes and losing the message entirely; STEALTH’s spinal-code state lets later tokens supply evidence that recovers the payload despite the mismatch.

Like any steganographic technique, STEALTH is dualuse: the same mechanism that lets a dissident evade censorship can let an adversary run a covert channel undetected. Because STEALTH removes the deployment-matching requirement that made earlier schemes fragile, it also lowers the bar for using steganography as a reliable, hard-to-detect channel between two parties with varying infrastructure. Of note, any defense that relies on sender and receiver running mismatched deployments to disrupt steganographic coordination will not hold against a scheme like STEALTH. A sender on a datacenter GPU and a receiver on a consumer GPU running a 4-bit quantized model of the same family can still communicate, with messages provably indistinguishable from honest model output. In this paper, we:

Demonstrate that schemes relying on bit-exact computation fail under ordinary mismatches in hardware, precision, or sequence processing procedures;

Propose STEALTH, which tolerates these mismatches, even enabling senders with speculative decoding, or receivers with different models altogether;

• Prove STEALTH is secure and produces text indistinguishable from benign model output; and

Evaluate STEALTH across three model families, five receiver configurations per family, smaller samefamily receivers, receivers of other model families, and two GPU architectures, against bit-exact and robust baselines, with statistical detection tests.

## 2. Background and Related Work

We first place STEALTH within the steganography and covert-channel literature, then define sampling from a language model and what it means for a stegosystem built on one to be provably secure. We then explain why a receiver cannot count on recomputing the sender’s distribution exactly, and cover the sampler STEALTH uses to tolerate this, the alternatives other schemes take, the coding scheme that spreads a message across tokens, and the steganalysis literature that complements provable security.

Steganography and covert channels. Censors need not break encryption to block it: deep packet inspection can flag a TLS handshake or tunnel without decrypting a single byte [15], so censorship-resistant communication increasingly turns to covert channels that hide messages inside traffic that looks benign [16]. These channels face their own challenge, since merely looking benign under casual inspection does not rule out being statistically distinguishable from genuinely benign traffic under closer scrutiny [17]. For example, in image steganography a message can be converted to a bitstream and each bit used to replace the least significant bit of a pixel’s color value [18]. While imperceptible to a human eye, it is still statistically detectable, since altering the least significant bits distorts an image’s natural statistics in ways statistical steganalysis can detect, motivating decades of research into more covert embedding methods [19]. This also motivates provable security [3], which we formalize in Section 6.

Provably secure linguistic steganography. An LM maps a context x (the prompt and the text so far) to a distribution p(x) over a vocabulary of tokens V that includes an end-of-text outcome END; text is produced by sampling a token from p(x), appending it to x, and repeating until END is sampled or a length cap is hit. Following Hopper et al. [3], a stegosystem is secure if no efficient warden who chooses the hidden messages can distinguish stegotexts from samples of the cover channel; for an LM, the channel is this sampling process. LM schemes have sender and receiver independently recompute $p ( x )$ and encode the message in exactly where the sampled token falls within it. Arithmetic coding embeds the message as a point in the token’s cumulative interval [20], but reuses randomness across tokens and is not provably secure [5]. Three schemes reach this bar: Meteor masks the message with a keyed pseudorandom generator and uses the result as the sampling randomness [5], iMEC embeds through a minimum-entropy coupling between the message and $p ( x )$ [6], and Discop partitions $p ( x )$ into rotated “copies” and encodes the message in which copy is sampled from [7]. All four require the receiver to reconstruct the sender’s $p ( x )$ bit for bit to invert the embedding.

Why probabilities differ. In practice, this bit-exact requirement is difficult to meet. Matrix multiplications and softmax normalizations sum many floating-point terms, and the result depends on the order they are summed in; GPUs of different architectures, different kernels (e.g. attention implementations chosen by sequence length), and different batch sizes sum in different orders, giving different results for the same model and input [8–10]. This variation is largest for mid-range probabilities [21], which arise at exactly the high-entropy steps where a stegosystem embeds most of its message. Lower-precision deployments (bf16, fp16) and weight quantization (8-bit, 4-bit NF4 [22]) add larger, systematic differences on top of this. Generation with a keyvalue cache (one forward pass per token) and teacher-forced scoring (one forward pass over the whole text) compute the same function along different paths, and can disagree even on identical hardware running identical weights [10]. Section 4 measures how badly these break schemes relying on bit-exact probability distributions.

Robust steganography. Other schemes respond to probability mismatch by giving up on recomputing $p ( x )$ exactly. One approach is to never let the receiver run the model at all, embedding the message in side information decoded without access to $p ( x )$ [23–25]. This is robust by construction, but at a much lower payload rate than a scheme that samples from the model’s own distribution. Another method hardens the exact schemes, like arithmetic coding, by running sender and receiver in float64, which removes most numerical mismatch between two instances of the same model, at substantial compute cost and without addressing precision, quantization, model changes, or detectability [26]. We compare STEALTH against both approaches in Section 7.

The exponential race. STEALTH tolerates the mismatch starting with a different sampler, one whose errors stay local rather than cascading. Let $E _ { i } \ \sim \ \mathrm { E x p } ( 1 )$ be drawn independently for each $i \in V$ . Then arg min $E _ { i } / p _ { i }$ equals i with probability exactly $p _ { i }$ . Taking logarithms, the winner minimizes ln $E _ { i } - \ln p _ { i }$ , the Gumbel-max trick in another form [27, 28]. Unlike inverse-CDF sampling, where a probability difference at one token can shift the intervals used to sample every other token, a mismatch here changes at most the one token whose race it affects. Aaronson’s LM watermark and its successors already use this sampler, but with a fixed key and no payload [13, 29].

Spinal codes. What remains is spreading a message across many tokens to absorb individual differences between sender and receiver. A spinal code [14] splits a message into k-bit chunks $c _ { 1 } , \ldots , c _ { L }$ and computes a spine $s _ { j } = H ( s _ { j - 1 } , c _ { j } )$ with a hash function H. Each spine value seeds the pseudorandom noise transmitted in place of the chunk itself. Because $s _ { j }$ depends on every chunk up to $j ,$ a decoder that guesses a chunk wrong will see a mismatch from that point on, which lets it test guesses in order and keep only the best B partial guesses (beam search). The code is rateless: the sender can keep producing symbols for as long as it takes the receiver to succeed.

Detecting steganography. A separate line of work tries to catch steganography empirically rather than prove its absence. Linguistic steganalyzers train word-level neural networks [30–32] or fine-tuned transformers [33] to tell stegotext from cover text by its statistics alone, without access to the embedding scheme. Cao et al.’s missing-tail test instead targets a specific artifact of fixed-precision decoding: interval-based schemes systematically under-sample a model’s most improbable tokens [34]. A provable-security proof rules out detection by any efficient warden, but Section 7.5 checks STEALTH against these detectors directly.

## 3. Threat Model and Goals

As a running example, we will consider a comment section on a public news article as our covert channel. The article serves as context for the prompt, and the sender’s model writes a comment on the article to pass information. If the message needs more text, the conversation continues with further comments, each prompted by the article and everything posted before it. The receiver reads the article and the comments off the page, in the order they were posted. The same structure fits any multi-turn channel where two parties exchange LM-generated text in public view, such as a chat, a forum thread, or messages passed between agents.

Before sending any message, the sender and receiver agree on a public prompt policy, describing how each prompt is built from the article and the texts posted so far, and on a payload length. They also share a uniformly random key $K .$ . The sender runs a model M on its own hardware and software, while the receiver runs a model $M ^ { \prime } \approx M$ that may differ from M in precision, quantization, GPU or software stack, model size within the same family, or model family entirely, each with its own tokenizer (Section 5). The receiver sees only the posted strings, not the sender’s probabilities, and does not know $M \mathrm { { s } }$ exact configuration.

Standing between them is a passive warden, who reads every posted text and knows everything about the scheme except the key: the model and its exact configuration, the prompt policy, and how to sample honest text from M itself. We allow the warden to adaptively choose the article used as context and the hidden messages being sent. The warden wins by correctly telling whether a given conversation carries a hidden message at all.

STEALTH aims for two properties against this warden. Security requires that the posted texts be computationally indistinguishable from honest samples of $M ,$ which we formalize in Section 6. Robustness requires that a receiver running $M ^ { \prime } \ne M$ recover the correct payload from as little text as possible, and almost never output a wrong one.

TABLE 1. BIT-EXACT SCHEMES FAIL UNDER MISMATCH. PERCENT OF MESSAGES DECODED. <sup>∗</sup>: RESULTS ACROSS BF16, FP16 AND FP32.
<table><tr><td>Receiver Gemma Llama Qwen</td></tr><tr><td>Arithmetic coding, 7–9B models</td></tr><tr><td>Same model, replay 100 100 100</td></tr><tr><td>Different GPU, replay 0 0 0 Same model, teacher forced* 0 0</td></tr><tr><td>0 int8, NF4, smaller model 0 0 0</td></tr><tr><td>Discop, 7–9B models</td></tr><tr><td>Same model, replay 100 100 100</td></tr><tr><td>Different GPU, replay 0 0 0 Same model, teacher forced* 0.5-1.5 1.5 0.5-1</td></tr><tr><td>int8, NF4, smaller model 0 0 0</td></tr><tr><td>Arithmetic coding, w/ float64 senders, 1.5–3B models</td></tr><tr><td>fp64, replay 100 100 100 100</td></tr><tr><td>Different GPU, fp64 replay 100 100</td></tr><tr><td>fp64, teacher forced 100 100 100 fp32 0 0 0.5</td></tr><tr><td>bf16*, int8, NF4 0 0 0</td></tr></table>

## 4. Mismatch Breaks Bit-Exact Schemes

To demonstrate how ordinary mismatch breaks the bitexact schemes of Section 2, we test both arithmetic coding [20] and Discop [7], and the float64 hardening proposed by Bauer et al. [26]. Each scheme generates comments on the same set of news articles from CNN/DailyMail [35] used throughout our evaluation (Section 7), with the same three LM senders in bf16 (Gemma-2-9B-it [36], Llama-3.1-8B-Instruct [37], and Qwen2.5-7B-Instruct [38]), so that every scheme faces the same mismatch as STEALTH. A receiver can decode either by replaying the sender’s exact computation, which requires matching its weights, precision, and hardware exactly, or by teacherforcing the posted text in one pass on the receiver’s own hardware. First, we keep sender and receiver on the same hardware (best-case scenario), and vary only the processing mode, precision, quantization, or model size, dropping to the smaller sibling in each family (Gemma-2-2b-it, Llama-3.2-3B-Instruct, Qwen2.5- 1.5B-Instruct) [36, 38, 39]. Then, we keep configurations identical, but vary the hardware between sender and receiver. The results described below are summarized in Table 1.

Arithmetic coding. We implemented arithmetic-coding steganography [20] over the full vocabulary, with tokens sorted by probability and 32-bit interval precision. With the three 7–9B senders in bf16, replaying the sender’s own generation path decodes every message for all three families, as expected. Teacher forcing the same cover text decodes none. No other receiver configuration does better: decoding in fp16 or fp32, quantizing to int8 or 4-bit NF4, or substituting a smaller model from the same family fails on every message. Decoding derailed after only 5–7 correct bits with the same model, and 1–4 with other configurations.

Discop. For Discop [7], replay again decodes all messages per family. Teacher forcing decodes ∼ 1%, and switching precision, quantization, or model size decodes none. At every step Discop builds a Huffman tree over the whole vocabulary, so a small probability change in even a rare token can change how the tree merges, which changes every later token’s path through it.

TABLE 2. TOTAL VARIATION PER TOKEN ON SAME GPU W/ REPLAY VS. TEACHER-FORCING (TF) AND ON DIFFERENT (DIFF) GPUS W/ REPLAY.
<table><tr><td>Receiver</td><td>Same GPU (replay vs. TF)</td><td>Diff GPU (replay)</td></tr><tr><td>Gemma 9B</td><td>0.86%</td><td>0.84%</td></tr><tr><td>Gemma 2B</td><td>1.11%</td><td>1.11%</td></tr><tr><td>Llama 8B</td><td>0.95%</td><td>0.94%</td></tr><tr><td>Llama 3B</td><td>0.94%</td><td>0.91%</td></tr><tr><td>Qwen 7B</td><td>1.24%</td><td>1.23%</td></tr><tr><td>Qwen 1.5B</td><td>1.42%</td><td>1.38%</td></tr></table>

Hardware. Next, we pivot from best-case to a more practical scenario where the sender and receiver have different hardware. We generated each family’s comments on an NVIDIA H100 and decoded them by exact replay on an NVIDIA A6000. While both methods decode all same-GPU messages this way, across the two architectures they decode none. Table 2 shows why: the total variation (TV) between the two architectures’ token distributions runs 0.8–1.4% per token, on par with the TV between generating a text step-bystep and scoring it with teacher forcing on the same GPU. Not one of the positions in any message we measured was bit-identical across different architectures.

Hardening with float64. Bauer et al. propose running both sides in float64, reasoning that the added precision shrinks numerical differences below what the coder can detect, though they stop short of claiming this holds for every device combination [26]. We found a pitfall in replicating this with Hugging Face models: several normalization and attention operations (RMSNorm in Qwen2 and Llama, Gemma 2’s eager-attention softmax) silently cast to float32 regardless of the requested dtype, so float32-sized differences survive and decoding still fails after 20–24 bits. After patching these casts, float64 senders decoded all messages by teacher forcing, and float64 text generated on an H100 decodes correctly on an A6000, for every family, including inside batches with other texts. On the hardware we tested, float64 hardening removes the mismatch it targets, though as Bauer et al. themselves note, nothing guarantees this holds on configurations we did not test.

What float64 does not fix. The hardening keeps only one, carefully matched configuration in sync; it is not robust to anything else. A receiver at fp32 decoded at most 0.5% of messages tested, and lower precision or quantization always failed. It is also expensive: a 7–9B model needs 61–74 GB in float64, more than a 48 GB GPU holds, so we evaluated this baseline on smaller 1.5–3B models instead. On the hardware we tested, float64 generation is 2.5–5.5× slower per token than our approach, and decoding a message takes 5.8–16.5 s against 55–71 ms for STEALTH on the same 1.5– 3B models (Section 7.4). Most importantly to our threat model, a warden that holds both a bf16 and a float64 copy of the model can also tell float64 comments from bf16 ones, by comparing how likely each comment’s tokens are under the two models and accumulating that evidence across many comments (Figure 2).

![](images/7dccd191e81f682ec71771353ac35dbcaa81ff365db2cbda47973c42ec9510b9.jpg)  
Figure 2. A warden can distinguish float64 comments from bf16 (AUC 0.5 is chance). The vertical line marks the median, across the three families, of the number of comments at which the null hypothesis that the sender is using bf16 becomes rejectable at the 5% significance level.

Across both coders, any mismatch collapses decoding, and the mitigation using float64 on both sides is slow, rarely deployed, and detectable given enough text. Our approach (Section 5) instead assumes sender and receiver will disagree, and asks only that the receiver recover the message despite that disagreement while the posted text stays indistinguishable from an honest sample.

## 5. STEALTH

We now describe STEALTH in full. Section 5.1 gives the exponential race the sender uses to sample each token, and why a mismatch at one position can flip only that position’s token rather than cascading into every token that follows. Section 5.2 shows how a spinal code spreads the payload across many such tokens, and Section 5.3 gives the receiver’s beam search over partial guesses. Section 5.4 covers how a text ends and how re-tokenization is handled, and Section 5.5 removes the sender’s tokenizer from the noise entirely, letting a receiver on a different model family decode the same text. Figure 3 walks through each step on the running example from Figure 1.

## 5.1. Exponential Race

Each message gets its own key $K _ { m } ,$ derived from the shared secret K and a fresh nonce ν by a standard keyed hash function (Section 6); ν need not be secret, since it only has to tell the receiver which posted texts belong to this message, not protect anything by itself. At each token position, the sender draws noise $E _ { i } \sim \mathrm { E x p } ( 1 )$ for every $i \in V$ from a pseudorandom function keyed by $K _ { m }$ and the token’s position in the cover text the sender is writing, and outputs the token $\begin{array} { r } { y = \arg \operatorname* { m i n } _ { i \in V } E _ { i } / p _ { i } ( h ) } \end{array}$ the exponential race introduced in Section 2. Repeating the race on the receiver’s own distribution $p ^ { \prime } ( h )$ with the same noise reproduces y whenever $p ( h )$ and $p ^ { \prime } ( h )$ agree, and disagrees with probability at most $2 D / ( 1 + D )$ , where D is their total variation distance [40]. Section 5.3 explains how STEALTH recovers the payload despite these disagreements. Section 6 formalizes that the race samples exactly from $p ( h )$ ; the disagreement bound is proved by Daliri et al. [40].

A flip at one position never impacts the other tokens. The position t is just the token’s byte offset in the already-written text, so both sender and receiver read t directly off the string. The noise at that position depends only on the message key, that offset, and the spine value its word carries (Section 5.2), never on which token was chosen there or at any earlier position, or on either party’s probabilities. So, a mismatch that flips $y _ { t }$ leaves every other position’s race untouched. The bit-exact schemes of Section 2 behave differently: their decoders carry state from one step to the next, such as an arithmetic coder’s current interval, so a single wrong step corrupts every bit decoded after it. STEALTH’s races are independent trials, each succeeding or failing on its own.

## 5.2. Spinal Code

The sender frames the n-bit payload m with a 16-bit cyclic redundancy check (CRC), which bounds how often a failed decode returns a wrong payload rather than reporting failure (Section 7), and splits the result into L chunks of $k =$ 4 bits each. A spine [14] ties these chunks to the exponential race: a fixed $s _ { 0 }$ starts the chain, and each later value $s _ { j } =$ $H ( s _ { j - 1 } , c _ { j } )$ hashes the previous spine value together with chunk $c _ { j } , { \bf s o } \ s _ { j }$ depends on every chunk up to j and changes completely if any of them does. Every token position is assigned one of the $L$ spine values when its noise is derived, so the chunk a position carries evidence for is fixed before any text is written.

Positions are assigned to spine values by counting words rather than tokens or bytes, where a word begins at the first non-whitespace byte following a whitespace byte (or at the start of the text): a token carries spine value $s _ { 1 } .$ +(w mod $L )$ , where $w$ is the number of words begun before the token’s first byte. The token that begins the w-th word (0-indexed) therefore carries $s _ { 1 + ( w \mathrm { ~ m o d ~ } L ) }$ , and the rest of that word’s tokens carry the next spine value; the values cycle back to $s _ { 1 }$ after $s _ { L }$ . Counting words rather than tokens keeps every chunk supplied with roughly the same number of positions regardless of how the sender’s tokenizer happens to split the text, and like the position itself, a word count does not depend on which tokenizer computes it, which matters once sender and receiver use different ones (Section 5.5).

The spinal code is rateless. Reaching $s _ { L }$ does not end the payload’s encoding, it only starts another pass over the same L chunks, so a longer cover text keeps adding independent evidence for every chunk rather than running out of positions to carry them. A sender facing more mismatch, or wanting a safer margin before stopping, simply writes more text. The sender never needs to shorten, pad, or otherwise fit the payload to a fixed cover text length. This also lets the model end a text naturally, rather than cutting it off wherever the payload happens to run out, which is itself something a warden would pick up on. The full STEALTH encoding procedure is given in Algorithm 1.

![](images/e4ad82eb38472ecc476e457cca731e8be9dc884225b162404c6f71ab4353baac.jpg)

![](images/f4314936f7337f1707e9207701dabd57fbe79c532fd27477d55cf92de6b35551.jpg)

Figure 3. How STEALTH encodes and decodes the payload from Figure 1. (a) Payload chunks are hashed into a chain of spine values, each keying the noise for its words’ tokens. (b) The receiver replays each race under its own p<sup>′</sup>: a narrow loss costs little, but a wrong chunk guess changes the noise and loses most races. (c) Beam search keeps the B best partial guesses and outputs the best one that passes the CRC.  
Algorithm 1 Sender: encode payload m into a cover text   
Require: shared key K, nonce ν, payload m, prompt,   
model M   
1: $K _ { m } \gets \mathrm { K e y e d H a s h } ( K , \nu )$   
2: $c _ { 1 } , \ldots , c _ { L } $ chunks of ${ \dot { m } } \parallel \mathrm { C R C } ( m )$   
3: $s _ { 0 } \gets 0 ; ~ s _ { j } \gets H ( s _ { j - 1 } , c _ { j } ) \mathrm { ~ f o r ~ } j = 1 , \dots , L$   
4: $t , w \gets 0 ;$ text $ ( )$   
5: repeat   
6: h ← prompt ∥ text   
7: $p  p _ { M } ( h )$   
8: $s \gets s _ { 1 + ( w }$ mod L)   
9: draw $E _ { i } \sim \exp ( \stackrel { \cdot } { 1 } )$ keyed by $( K _ { m } , t , s , i ) , i \in V$   
10: $y  \arg \operatorname* { m i n } _ { i \in V } E _ { i } / p _ { i }$   
11: append $y$ to text   
12: $t  t + \mathrm { l e n } _ { \mathrm { b y t e s } } ( y )$   
13: w $ w +$ newwords(y)   
14: until $y = \mathtt { E N D }$   
15: post text (without END)

## 5.3. Decoding

To test a guess for chunk $c _ { j } .$ , the receiver teacher forces every posted text once under its own model $M ^ { \prime }$ , obtaining log-probabilities $\ell ^ { \prime } ( t )$ at each position, and recomputes the race’s noise at every position assigned the guessed spine value. At such a position the observed token $y _ { t }$ wins the race by some margin $\mu _ { t } .$ , the gap between its race score and the best of the rest. A correct guess should see $y _ { t }$ win almost every one of these positions, and lose only the narrow ones that Section 5.1’s disagreement bound predicts will flip; a wrong guess sees $y _ { t }$ win only with probability $p _ { y _ { t } } ^ { \prime } ( h )$ at each position, and while this can be high at a single lowentropy position, it is rarely high at every position a chunk is assigned, so a wrong guess mostly loses once its evidence is pooled across all of the chunk’s positions.

The receiver turns each margin into a score $\lambda ( \mu _ { t } ) =$ max(ln sigmoid $. ( \mu _ { t } / \tau )$ , ln ϵ), where the temperature τ sets how much a narrow loss is discounted rather than treated as a contradiction, and the floor ϵ caps how much even a maximally wrong guess can cost at any one position. A correct guess that narrowly lost a race is charged little, while a wrong guess pays close to the floor at every position that carries real evidence. Summing λ over the positions assigned to chunk j turns this into a score for how well a guessed value of that chunk explains the text. Neither τ nor ϵ has to be fixed in advance, since the receiver already holds the whole posted text. If decoding with one setting leaves no guess passing the CRC, it can simply decode again with a different τ or ϵ, for instance a larger τ if it suspects heavier mismatch than usual (Section 7).

The receiver searches for the whole payload with a beam of partial guesses, extending each by every possible value of the next chunk and keeping only the best-scoring extensions. Because the spine’s hash chain makes every later spine value depend on every earlier chunk, a wrong chunk corrupts the scores of everything built on top of it, and the correct sequence of chunks overtakes these corrupted guesses within the beam. Once all L chunks have been guessed, the receiver outputs the highest-scoring surviving guess whose CRC passes, or reports failure if none does.

```latex
Algorithm 2 Receiver: decode a payload from posted text
Require: shared key K, nonce ν, payload length n, posted
text, model $M ^ { \prime }$
1: $K _ { m } \gets \mathrm { K e y e d H a s h } ( K , \nu )$
2: $L \gets ( n + 1 6 ) / 4 \triangleright \mathrm { \backslash C R C } \mathrm { : }$ 16 bits, Chunk width: 4 bits
3: tokenize the text; append END if it ended naturally
4: compute $( t , w )$ for each token
5: ℓ<sup>′</sup> ← log-probabilities of $M ^ { \prime }$ over the prompt and text
6: beam $ \{ ( 0 , s _ { 0 } , ( ) ) \}$ ▷ score, spine value, chunks
7: for $j = 1 , \dots , L$ do
8: $P $ positions with w mod $L = j - 1$
9: cand ← ∅ ▷ candidates for this round
10: for $( s c , s , c s ) \in$ beam, $c \in \{ 0 , \ldots , 2 ^ { k } { - } 1 \}$ do
11: $s ^ { \prime } \gets H ( s , c )$ ▷ spine hash function
12: $\begin{array} { r } { s c ^ { \prime } \gets s c + \sum _ { t \in P } \lambda \big ( \mu _ { t } ( K _ { m } , \bar { t } , s ^ { \prime } ) \big ) } \end{array}$
13: add $( s c ^ { \prime } , s ^ { \prime } , \overline { { { c s } } } \ " | \overline { { { c } } } )$ to cand
14: end for
15: beam ← the highest-scoring entries of cand
16: end for
17: return the highest-scoring cs in beam whose CRC
checks, else FAIL
```

Section 5.2’s CRC is what bounds how often this outputs a wrong payload instead of failing outright. The full decoding procedure is given in Algorithm 2.

## 5.4. Additional Details

Natural endings. The END token takes part in the exponential race alongside every other vocabulary entry, so a text ends exactly where an honest sample from M would end it, rather than wherever the payload happens to run out. While writing, the sender checks whether the text so far is already decodable by running the receiver’s own algorithm on its own model. Once the text ends, the sender stops there if that check passes. Tokens written after the text first becomes decodable are not wasted, since the rateless code (Section 5.2) turns them into further passes over the same L chunks, giving a safety margin for receivers with more mismatch than the sender assumed.

Multiple texts. If a model ends the text before the check passes, the message continues in another text, built from a prompt that includes every text posted so far. Position t and word count w keep advancing across the whole stream rather than resetting at each text’s start, so a chunk’s positions can span several texts exactly as they would span one.

Re-tokenization. The receiver works from the posted text, not the sender’s tokens directly, and re-tokenizing that text does not always reproduce the sender’s token boundaries. The sender samples one token at a time, so it may write “ hel” followed by $\ " \mathrm { 1 0 } \ "$ , which the receiver’s tokenizer reads back as the single token $^ { 6 6 } { \mathrm { \_ h e l l o } } ^ { 3 5 }$ . Because position is a byte offset rather than a token index (Section 5.1), the damage stays local. The merged token sits at the first part’s offset and is scored as an ordinary flip; the second part’s offset is never visited, an erasure that contributes no evidence rather than wrong evidence. Every later token keeps the sender’s offsets, so the rest of the stream decodes normally, and the rateless code’s redundancy (Section 5.2) absorbs the flip and erasure like any other.

## 5.5. Decoding Across Model Families

We alter our encoding and decoding procedures slightly for scenarios where the sender and receiver are using different model families. Position t and word count w are defined on the posted text itself, so nothing about them ties the receiver to the sender’s tokenizer, and Section 5.4 already covers a receiver that re-tokenizes the same text differently. What still does tie the two together is the noise’s fourth input, the vocabulary index i in $( K _ { m } , t , s , i )$ . Index i names a token in the sender’s vocabulary, and a model of another family has its own, unrelated vocabulary, with its own index for the same string. A receiver on another model family has no way to know which of its indices corresponds to which of the sender’s.

The fix is to key the noise by the token’s own bytes instead of its index. Writing κ(i) for a fixed hash of token $i \ ' s$ byte string, the sender and receiver both draw noise from $( K _ { m } , t , \bar { s } , \kappa ( i ) )$ rather than $( K _ { m } , t , s , i ) ;$ only this one input to the keyed pseudorandom function changes, in both Algorithm 1 and Algorithm 2. Because κ depends only on a token’s spelling, the sender’s and receiver’s vocabularies agree on κ(i) for every string they share, even though they disagree on i itself. This requires κ to be injective on the sender’s vocabulary, so that two of its tokens never draw the same noise. Some vocabularies contain several tokens with the same spelling, such as Gemma’s byte-fallback tokens, which duplicate ordinary single-byte tokens. Among such a group, the ordinary token, the one any tokenizer produces for that string, is keyed by its bytes, and each of the others by a hash of its index under a separate domain tag, with END given its own reserved key.

The receiver tokenizes the posted text with its own tokenizer and its own vocabulary, and otherwise decodes exactly as in Section 5.3. Wherever the two tokenizers split the text the same way, κ(i) lines the receiver’s token up with the sender’s and the race scores it normally; wherever they split it differently, the mismatch is just a re-tokenization, handled the same flip-and-erasure way as Section 5.4.

## 6. Security Analysis

A warden watching the conversation sees only the sender’s posted texts, never whether or how a receiver decodes them. What has to be shown is that those texts are indistinguishable from an honest conversation with the sender’s own model, down to where each text ends.

## 6.1. Security Definition

Write ${ \hat { p } } ( h )$ for the distribution the sender’s model actually assigns a context $h ,$ computed from its own float32 log-probabilities rather than some idealized real-valued one.

An honest text is one sampled from $\hat { p } ,$ token by token, until END; an honest conversation chains such texts together under the public prompt policy, each one’s prompt built from everything posted before it. STEALTH’s stegotext is secure if no warden can tell it apart from an honest conversation, even when it gets to choose the payloads and context adaptively, the chosen-hiddentext notion of security introduced by Hopper et al. [3] and instantiated here for our setting.

Definition 1 (Security). A warden A queries an oracle with a payload m, a nonce $\nu ,$ and a starting context $h _ { 0 } ,$ , and gets back either the sender’s posted texts for m (the stego oracle $O _ { S } ) ,$ or an honest conversation under the same stopping rule $( O _ { H } ) .$ A chooses every query adaptively, knowing the outcome of every earlier one, but never reuses a nonce. Its advantage is

$$
\mathrm { A d v } ( A ) = \left| \operatorname* { P r } [ A ^ { O s } = 1 ] - \operatorname* { P r } [ A ^ { O _ { H } } = 1 ] \right| .
$$

STEALTH is secure against A to the extent $\operatorname { A d v } ( A )$ is small; Theorem 1 bounds it concretely in terms of the warden’s number of queries q and total tokens T.

For our experiments we use HMAC-SHA256 [41] for Section 5.1’s keyed hash and the ChaCha20 block function [42] for its keyed pseudorandom function (PRF). We assume both are PRFs with advantage $\epsilon _ { \mathrm { H M A C } } ( q )$ against q queries and $\epsilon _ { \mathrm { C h a C h a } } ( r )$ against r block evaluations.

## 6.2. Security Theorem

Lemma 1 (Exponential race [27, 28]). For a distribution $p ( h )$ and independent $E _ { i } ~ \sim ~ \mathrm { E x p } ( 1 )$ for every i with $p _ { i } ( h ) > 0 ,$ the race winner $W = \arg \operatorname* { m i n } _ { i } E _ { i } / p _ { i } ( h )$ satisfies $\mathrm { P r } [ \dot { W } = i ] = p _ { i } ( h )$ , and the winning score min<sub>i</sub> $\displaystyle { \dot { E } } _ { i } / p _ { i } ( h )$ is itself Exp(1), independent of W. Proof in Appendix A.

This is the exact-sampling half of the claim from Section 5.1; its other half, the bound on how often a mismatched receiver’s race disagrees, is proved by Daliri et al. [40].

Lemma 1 needs independent noise at every position, but the inputs STEALTH feeds its noise function depend on the sender’s own earlier tokens. The next lemma shows that, with the keyed PRF replaced by a truly random function, this never matters: the noise read at each position is fresh given everything before it. Theorem 1 then returns to the real scheme, paying for the PRF assumption.

Lemma 2 (Freshness). Let $\mathcal { H } _ { t }$ be everything the sender has computed before choosing the token at position t. Replacing Section $\dot { 5 } . I ^ { \prime } s$ keyed pseudorandom function by a truly random function, the noise it reads to choose the token at position t is independent uniform, independent of $\mathcal { H } _ { t }$ as well. Proof in Appendix A.

Nothing in the proof needs the payload to be secret, so the warden may choose it. Because t is itself part of the noise’s input, two positions sharing a spine value still draw independent noise from each other. The argument also survives Section 5.5’s byte-keyed variant unchanged, since κ is injective on the sender’s vocabulary and distinct tokens still read distinct inputs there too.

Corollary 1 (Ideal-world texts are honest). With the keyed PRF replaced by a truly random function and the race computed exactly on continuous uniforms, let the sender keep generating texts indefinitely under the public prompt policy, ignoring its own stopping rule. The resulting sequence of texts has the same law as an honest conversation. Proof in Appendix A.

Theorem 1 (Security). If the number of texts per hidden message is fixed in advance, or set by any rule that does not use the key, and the honest oracle uses the same rule, then for q queries, T tokens in total, and at most r ChaCha20 evaluations per message,

$$
\mathrm { A d v } ( A ) \leq \epsilon _ { \mathrm { H M A C } } ( q ) + q \epsilon _ { \mathrm { C h a C h a } } ( r ) + T \delta ,
$$

where δ is the per-token disagreement probability between the implemented race and the ideal one (Section 6.3). Proof in Appendix A.

Every text ends where the sender’s model would have ended an honest sample, so the only quantity in a message that can depend on the key at all is how many texts it took. Other distribution-preserving schemes end a text where the message ends rather than where the model would. Arithmetic coding, Meteor, and Discop stop once the message’s bits are exhausted, and iMEC stops once the receiver’s posterior over the message is confident enough [5–7, 20]; none of these coincide with where an honest sample would end. Each could pad with honest, message-independent text until END, but that padding carries nothing; in STEALTH, every token up to the natural ending also adds evidence toward the payload.

Theorem 1 covers a sender that posts a fixed number of texts per query, independent of the key, and simply fails the message if that was not enough. We will see in Section 7.2, that a fixed count of two would have failed none of the 600 messages we evaluated: 33 Gemma, 3 Llama, and 1 Qwen message needed a second comment, and none needed a third. A sender that instead stops as soon as the decodability check of Section 5.4 passes posts less text, but its number of texts N can depend on the key. By coupling it with a sender fixed at k texts, a warden gains at most $\bar { \mathrm { P r } } [ N \neq k ] ,$ , which for $k = 1$ is the 6% of messages that needed a second comment.

## 6.3. Finite-Precision Error

Theorem 1’s remaining term, δ, bounds how often Section 5.1’s implemented race disagrees with the mathematically ideal one it is supposed to compute. Write Race(ℓ, W) for the token the sender’s implementation outputs from logprobabilities ℓ and raw 64-bit noise words ${ \bar { W } } ,$ with every score computed in float64, and IdealRace(ℓ, U) for the same race run on exactly continuous uniforms U, each drawn inside its word’s $2 ^ { - { \overline { { 6 4 } } } }$ -wide cell. The implementation departs from the ideal in two ways: rounding, since each computed score can be off by up to some η from its exact value, and discretization, since each uniform is read from the midpoint of its $2 ^ { - 6 4 }$ -wide cell rather than sampled continuously.

![](images/9caa7c524f91793949805b1629c869320cacb7fa61fc08746486b3254cacf238.jpg)  
Figure 4. Messages decoded against tokens posted. Solid lines: τ = 1.0; dashed: τ = 0.1. Smaller models: Gemma 2B, Llama 3B, Qwen 1.5B. The axis stops at 200 tokens; Table 3 gives the full counts.

Lemma 3 (Floating-point race). Let $\omega = 2 ^ { - 6 4 } , W _ { i } , i \in V ,$ be independent uniform 64-bit words, $Z _ { i }$ independent uniforms on [0, 1), and $U _ { i } = \omega ( W _ { i } + Z _ { i } )$ , which are themselves independent uniforms on $[ 0 , \dot { 1 } )$ . Suppose the implementation reads noise $\hat { E } _ { i } = - \ln ( 1 - ( W _ { i } + \textstyle \frac { 1 } { 2 } ) \omega )$ from the midpoint of $W _ { i } ^ { , }$ s cell and computes every log score within η of ln $\begin{array} { r l } { \hat { E } _ { i } - \ell _ { i } , } & { { } } \end{array}$ and let $\rho = e ^ { 2 \eta } - 1$ . Then for every ℓ and every $a > 0$ with $\textstyle \rho ( a + { \frac { 1 } { 2 } } ) \leq { \frac { 1 } { 4 } }$

$$
\begin{array} { r l } & { \operatorname* { P r } [ \operatorname { R a c e } ( \ell , W ) \neq \operatorname { I d e a l R a c e } ( \ell , U ) ] } \\ & { \qquad \le \left( 6 + 2 e \right) a \left| V \right| \omega + 8 | V | \omega + e ^ { - a } + \frac 3 2 \rho . } \end{array}
$$

Proof in Appendix A.

Our kernel’s rounding error is $\eta \approx 3 \times 1 0 ^ { - 1 4 }$ . With $| V | \le 2 ^ { 1 8 }$ , true of every vocabulary across all three model families, and $a \ = \ 3 0 .$ , the bound gives $\delta \leq 5 . 2 \times 1 0 ^ { - 1 2 }$ per token, about $1 . 6 \times 1 0 ^ { - 9 }$ over a 300-token message. By Theorem 1, finite precision adds at most Tδ to a warden’s advantage, so even a warden that observes a million such messages gains less than $2 \times 1 0 ^ { - 3 }$ from it.

No missing tail. Cao et al. [34] detect arithmetic coding and Meteor from the probability mass their finite-precision intervals can never reach. The exponential race has no such gap: it never builds a CDF, since each token draws its own noise, and scores are compared in log space, so a token with probability $1 0 ^ { - 3 0 }$ keeps its log-probability at full precision rather than being rounded to a zero-width interval. The race loses probability only where a token would need noise below the smallest representable value, $2 ^ { - 6 5 }$ , to win. On Qwen2.5 this unreachable mass totals $1 . 4 \times 1 0 ^ { - 1 7 }$ per token, against $1 . 7 \times 1 0 ^ { - 6 }$ for 32-bit arithmetic coding on the same model. This mass is part of the disagreement δ bounded by Lemma 3, so Theorem 1 already accounts for it. We directly run their detection test in Section 7.5.

## 7. Evaluation

## 7.1. Setup

Models. Senders use the same 7–9B models from Section 4, each in bf16, run both plainly and with speculative decoding using the smallest model of the same family as the draft (Gemma 2B, Llama 3.2 1B, Qwen 0.5B). Receivers cover the sender’s own weights at bf16, fp16, fp32, int8, and 4-bit NF4, the smaller models of the same family, and for Section 7.3, the large models of the other two families reading a byte-keyed sender. Because bf16, fp16, and fp32 gave nearly identical results throughout, the tables report bf16 only.

Hardware. Experiments are run on two NVIDIA architectures, Ampere (RTX A6000) and Hopper (H100). To isolate mismatch across hardware from mismatch across precision or model size, we also decode each architecture’s saved texts on the other.

Channel. The sender’s prompt is a request to comment on one of 200 CNN/DailyMail articles [35], with further comments posted if the first does not carry enough evidence to decode. Texts are sampled at temperature 1 for all models.

Code parameters. Messages are 32-bit payloads protected by a 16-bit CRC, spun with chunk width $k = 4$ (so $L = 1 2$ chunks) and beam width $B = 1 6$ , and decoded with the decoder’s score floor set to $\epsilon = 1 0 ^ { - 3 }$ . We report both of the decoder’s operating points, $\tau = 0 . 1$ and $\tau = 1 . 0$ , and a cascade that falls back to $\tau = 1 . 0$ only if $\tau = 0 . 1$ passes no guess through the CRC. The sender’s own stopping check always uses $\tau = 0 . 1 ;$ it runs every 12 tokens, and once it passes, the sender writes 12 more tokens as a margin before stopping at the end of the current comment. Our sweeps decode prefixes at the same 12-token granularity.

Metrics. Success is whether the decoded payload matches the sent one, over the 200 messages per condition. Flip rate is the share of tokens where the race winner under the receiver’s distribution differs from the winner under the sender’s, and TV is the mean total variation distance between their per-token distributions. Both measure how far the receiver’s distribution is from the sender’s, which explains why more mismatch needs more tokens. Tokens to N% is the shortest prefix length at which that fraction of messages decode (Figure 4 plots the full curves).

## 7.2. Robustness Under Mismatch

With STEALTH, every receiver we tested decodes every message given enough text, across every precision, quantiza tion level, and same-family model size; 200 of 200 bounds each configuration’s failure rate below 1.9% (95% Wilson interval). Figure 4 sweeps the decoder over prefixes of long streams written and decoded on the A6000, and Table 3 reports how much text each configuration needs. The worst case was with Gemma 2B, where no message needed more than 384 tokens.

TABLE 3. TOKENS POSTED TO REACH 90% / 99% / 100% OF MESSAGES DECODED, USING THE CASCADE DECODER.
<table><tr><td>Receiver</td><td>90%</td><td>99%</td><td>100%</td></tr><tr><td>Gemma 9B bf16</td><td>96</td><td>132</td><td>132</td></tr><tr><td>Gemma 9B int8</td><td>108</td><td>132</td><td>144</td></tr><tr><td>Gemma 9B NF4</td><td>120</td><td>156</td><td>192</td></tr><tr><td>Gemma 2B</td><td>228</td><td>372</td><td>384</td></tr><tr><td>Llama 8B bf16</td><td>84</td><td>108</td><td>132</td></tr><tr><td>Llama 8B int8</td><td>84</td><td>120</td><td>132</td></tr><tr><td>Llama 8B NF4</td><td>96</td><td>132</td><td>144</td></tr><tr><td>Llama 3B</td><td>132</td><td>192</td><td>252</td></tr><tr><td>Qwen 7B bf16</td><td>72</td><td>96</td><td>156</td></tr><tr><td>Qwen 7B int8</td><td>84</td><td>96</td><td>180</td></tr><tr><td>Qwen 7B NF4</td><td>96</td><td>132</td><td>204</td></tr><tr><td></td><td>180</td><td>300</td><td>300</td></tr><tr><td>Qwen 1.5B</td><td></td><td></td><td></td></tr></table>

What drives the cost. Within each family, the token cost in Table 3 tracks flip rate (Table 4) rather than precision or quantization: int8 receivers, whose flips stay under 5%, need little more than a same-configuration receiver, while NF4 and the smaller models, whose flips run from 8% to 37%, need more text, increasingly so as flips rise. Each flip costs evidence rather than adding it, so the spinal code’s beam search needs a larger margin of correct tokens to recover.

Decoding at the sender’s stopping point. A sender cannot see its receiver’s flips; it knows only when its own model can decode. Table 4 evaluates the simplest stopping rule built on that knowledge: the sender stops once its same-model stopping check succeeds, plus 12 more tokens, and we measure how often each receiver decodes on the sender’s GPU at that point. A receiver matching the sender’s configuration decodes every message in every family, since it repeats the sender’s own check. Receivers close to that configuration nearly always do too: with the cascade, fp16 and fp32 lose at most one message per family, and int8 at most four. Receivers farther away do not: NF4 decodes 93.5–99.5% of messages, and the smaller same-family models only 42–87.5%. The sender’s check alone is therefore a reliable signal only for receivers near its own configuration; to reach one farther away, the sender can post additional passes beyond the one its check calls for or, if it knows the receiver’s model, run that model for its check.

Neither operating point wins everywhere: τ = 1.0 decodes up to five fewer messages than τ = 0.1 on the same receiver, while τ = 0.1 collapses on the smaller same-family receivers, decoding 8.5–54% of messages where $\tau = 1 . 0$ decodes 42–87.5%. The cascade matches or beats the better setting on every row, and beats both on Gemma and Llama NF4, as the two settings tend to fail on different messages.

Across GPU architectures. Table 3 and Figure 4 decode texts on the same A6000 that wrote them. To isolate the hardware, we took the H100’s own sweep texts and decoded them on both GPUs, with the cascade decoder and every receiver configuration; Table 5 gives the A6000 side. Every configuration, including NF4 and the smaller same-family models, still reaches 100% of messages decoded. Moving the receiver from the H100 to the A6000 changes no configuration’s token counts by more than two 12-token steps, in either direction: the 90% column is unchanged for 10 of 12 receivers, and 5 of 12 are unchanged in every column, so the hardware mismatch adds no systematic token cost on top of precision and quantization. The new GPU changes about 1% of race winners at bf16 and 4–6% at int8 relative to the same receiver on the H100, yet the flip rate against the sender stays within 0.3 points of Table 4’s for every receiver, and fp32 picks the same winner at every position on both architectures. Bit-exact schemes have no such slack: across the same two GPUs, arithmetic coding and Discop decode none of their messages, even by exact replay (Section 4).

TABLE 4. PERCENT OF MESSAGES DECODED AT THE SENDER’S STOPPING POINT (ITS OWN CHECK PLUS 12 TOKENS), WITH FLIP RATE.
<table><tr><td>Family</td><td>Receiver</td><td>Flips (%)</td><td> $\tau = 0 . 1$ </td><td> $\tau = 1 . 0$ </td><td>Cascade</td></tr><tr><td>Gemma</td><td>9B bf16</td><td>0.9</td><td>100</td><td>97.5</td><td>100</td></tr><tr><td></td><td>9B int8</td><td>3.8</td><td>98</td><td>96</td><td>98</td></tr><tr><td></td><td>9BNF4</td><td>8.4</td><td>87.5</td><td>91</td><td>93.5</td></tr><tr><td></td><td>2B bf16</td><td>31.5</td><td>8.5</td><td>42</td><td>42</td></tr><tr><td>Llama</td><td>8B bf16</td><td>1.0</td><td>100</td><td>99.5</td><td>100</td></tr><tr><td></td><td>8B int8</td><td>4.9</td><td>100</td><td>99.5</td><td>100</td></tr><tr><td></td><td>8BNF4</td><td>10.2</td><td>93.5</td><td>97.5</td><td>98</td></tr><tr><td></td><td>3B bf16</td><td>23.5</td><td>54</td><td>87.5</td><td>87.5</td></tr><tr><td>Qwen</td><td>7B bf16</td><td>1.2</td><td>100</td><td>100</td><td>100</td></tr><tr><td></td><td>7B int8</td><td>4.2</td><td>100</td><td>99.5</td><td>100</td></tr><tr><td></td><td>7B NF4</td><td>12.1</td><td>96.5</td><td>99.5</td><td>99.5</td></tr><tr><td></td><td>1.5B bf16</td><td>36.9</td><td>32.5</td><td>84</td><td>84</td></tr></table>

TABLE 5. TOKENS POSTED TO REACH 90% / 99% / 100% OF MESSAGES DECODED WITH THE CASCADE DECODER, READING H100-WRITTEN STEGOTEXT ON AN A6000.
<table><tr><td>Receiver</td><td>90%</td><td>99%</td><td>100%</td></tr><tr><td>Gemma 9B bf16</td><td>96</td><td>120</td><td>180</td></tr><tr><td>Gemma 9B int8</td><td>96</td><td>144</td><td>180</td></tr><tr><td>Gemma 9B NF4</td><td>120</td><td>156</td><td>180</td></tr><tr><td>Gemma 2B</td><td>228</td><td>300</td><td>324</td></tr><tr><td>Llama 8B bf16</td><td>84</td><td>96</td><td>132</td></tr><tr><td>Llama 8B int8</td><td>84</td><td>108</td><td>132</td></tr><tr><td>Llama 8B NF4</td><td>96</td><td>132</td><td>156</td></tr><tr><td>Llama 3B</td><td>144</td><td>180</td><td>180</td></tr><tr><td>Qwen 7B bf16</td><td>84</td><td>96</td><td>96</td></tr><tr><td>Qwen 7B int8</td><td>84</td><td>96</td><td>96</td></tr><tr><td>Qwen 7B NF4</td><td>96</td><td>108</td><td>168</td></tr><tr><td>Qwen 1.5B</td><td>168</td><td>252</td><td>264</td></tr></table>

Efficient samplers. Because STEALTH tolerates mismatch, the sender is free to generate however is most efficient on its own hardware, even along computation paths the receiver never reproduces. Speculative decoding [43, 44] is one example. Each round, a small draft model runs the same keyed race on its own distribution for a few tokens; the sender’s model then scores all of them in one forward pass, runs its own race at each position with the same noise, and keeps the drafted tokens up to the first position where its race picks a different token, emitting its own winner there, as in drafter-invariant speculative decoding [40]. Every posted token is still the winner of the sender’s race, and no acceptance test or residual sampling is needed. A bit-exact scheme could not do this unless the receiver reproduced every verification pass exactly, including which drafted tokens were accepted, since a batched pass computes the sender’s probabilities along yet another path (Section 2). For STEALTH, it is just one more small mismatch. We repeated the runs of Table 4 with four drafted tokens per round, of which 26–42% were accepted, and receivers see no difference (Appendix B, Table 10): flip rates match the plain sender’s within 0.5 points for every receiver, and every same-size receiver decodes within three messages of its plain-sender count, in either direction. Smaller same-family receivers vary more, by 3–17 messages, mostly in the speculative sender’s favor but not consistently (Llama 3B gains five messages on the H100 and loses five on the A6000). One Gemma message passed the sender’s own check yet was missed by the same-configuration bf16 receiver: no tokens flipped, but its score sat at the edge of the beam, and a wider beam does decode it successfully.

TABLE 6. CROSS-FAMILY DECODING WITH BYTE-KEYED NOISE. ALIGNED: SHARE OF THE RECEIVER’S TOKENS MATCHING A SENDER TOKEN’S BYTE SPAN. FLIPS: SHARE OF ALIGNED TOKENS WHOSE RACE WINNER DIFFERS FROM THE POSTED TOKEN.
<table><tr><td colspan="4"></td><td>At stopping point</td><td>Whole stream, τ = 1.0</td></tr><tr><td>Sender</td><td>Receiver</td><td>Aligned (%)</td><td>Flips (%)</td><td>τ = 0.1 / τ = 1.0 / cascade (%)</td><td>Tokens to 90% / 99% / 100%</td></tr><tr><td>Gemma 9B</td><td>Gemma 9B bf16</td><td>99.9</td><td>1.0</td><td>99.5 / 95.5 / 100</td><td>108 /  156 / 204</td></tr><tr><td></td><td>Gemma 9B NF4</td><td>99.9</td><td>8.4</td><td>88.5 / 91.5 / 94</td><td>120 /  168 /  240</td></tr><tr><td></td><td>Llama 8B bf16</td><td>88.2</td><td>35.5</td><td>1 /  19 / 19</td><td>336 / 528 / 636</td></tr><tr><td></td><td>Llama 8B NF4</td><td>88.2</td><td>36.7</td><td>0.5 / 15.5 / 15.5</td><td>348 / 516 / 648</td></tr><tr><td></td><td>Qwen 7B bf16</td><td>88.5</td><td>37.4</td><td>0 /  14 /  14</td><td>396 / 612 / 756</td></tr><tr><td></td><td>Qwen 7B NF4</td><td>88.5</td><td>37.7</td><td>0 /  12.5 / 12.5</td><td>384 /  552 /  756</td></tr><tr><td>Llama 8B</td><td>Llama 8B bf16</td><td>99.9</td><td>1.0</td><td>100 / 99.5 / 100</td><td>96 / 120 / 180</td></tr><tr><td></td><td>Llama 8B NF4</td><td>99.9</td><td>10.7</td><td>93  / 99 / 99</td><td>96 / 144 /  180</td></tr><tr><td></td><td>Gemma 9B bf16</td><td>88.9</td><td>35.6</td><td>7.5 / 60.5 / 60.5</td><td>216 /  288 /  372</td></tr><tr><td></td><td>Gemma 9B NF4</td><td>88.9</td><td>35.9</td><td>6 / 58.5 / 58.5</td><td>228 /  276 / 324</td></tr><tr><td></td><td>Qwen 7B bf16</td><td>99.5</td><td>36.6</td><td>13 / 71 / 71</td><td>180 / 300 / 324</td></tr><tr><td></td><td>Qwen 7B NF4</td><td>99.5</td><td>36.5</td><td>12.5 / 71 / 71</td><td>192 / 300 / 384</td></tr><tr><td>Qwen 7B</td><td>Qwen 7B bf16</td><td>99.9</td><td>1.2</td><td>100 /  100 /  100</td><td>84 / 108 / 180</td></tr><tr><td></td><td>Qwen 7B NF4</td><td>99.9</td><td>10.5</td><td>99.5 / 100 / 100</td><td>96 / 120 /  180</td></tr><tr><td></td><td>Gemma 9B bf16</td><td>91.0</td><td>36.0</td><td>12.5 / 79.5 / 79.5</td><td>252 / 528 / 612</td></tr><tr><td></td><td>Gemma 9B NF4</td><td>91.0</td><td>36.2</td><td>14.5 / 81 / 81</td><td>240 / 516 / 588</td></tr><tr><td></td><td>Llama 8B bf16</td><td>97.5</td><td>34.2</td><td>27.5 / 90.5 / 90.5</td><td>180 / 432 / 468</td></tr><tr><td></td><td>Llama 8B NF4</td><td>97.5</td><td>34.9</td><td>27.5 / 90 / 90</td><td>168 /  432 / 444</td></tr></table>

## 7.3. Cross-Family Decoding

We re-ran the sweep of Section 7.2 on the A6000 with each large sender’s noise keyed by token bytes rather than vocabulary index (Section 5.5), so receivers of the other two model families can decode from the posted strings alone. Table 6 gives the results, with same-family bf16 and NF4 receivers kept as a control. The byte-keyed variant costs nothing within a family: its control rows decode within one percentage point of Table 4’s at the sender’s stopping point with the cascade, and within three points at either τ alone, as expected for an independent sample of 200 messages. They also need within two 12-token steps as much text as Table 3’s, even though Table 6’s token counts use $\tau = 1 . 0 .$

Across families, the receiver’s tokenizer splits the text differently from the sender’s: 88–99.5% of its tokens line up with a sender token at the same byte offset and length, lowest whenever Gemma’s tokenizer is on either side. Of the aligned tokens, 34–38% flip, comparable to the smaller same-family receivers of Table 4 (24–37%), and quantizing the receiver to NF4 adds almost nothing on top, since the difference between families already dominates. Given the whole stream, every cross-family receiver decodes every message, 90% of them within 168–252 tokens for Llama and Qwen senders and 336–396 for Gemma’s.

The sender’s same-model stopping check is not calibrated for these receivers: at the sender’s stopping point they decode only 12.5–90.5% of messages, almost entirely through τ = 1.0, since τ = 0.1 decodes at most 27.5%. A sender that knows its receiver’s model can instead run that model for its stopping check. Replaying the sweeps under that rule, stopping at the end of the first comment after the receiver’s τ = 0.1 check has decoded plus 12 more tokens, the sender posts more text, and 11–88% of messages need a second comment.

Across roughly 450,000 cross-family sweep decodes, a wrong payload was output 35 times. Each decode checks at most $B = 1 6$ final guesses against a 16-bit CRC, which bounds false accepts at $1 6 \cdot 2 ^ { - 1 6 }$ per decode, about 110 over these decodes; the 35 observed sit well within that bound.

## 7.4. Comparison With Other Methods

Section 4 showed what mismatch does to bit-exact schemes: arithmetic coding and Discop decode every message when the receiver replays the sender’s exact computation (Table 1), and almost none under any mismatch, including a smaller model of the same family. Table 7 sets their rates against ours for context. Under replay, arithmetic coding posts 0.19–0.28 bits per token at 32-bit payloads and 0.95–0.99 at 256 bits, against our 0.19–0.26 and 0.37– 0.43, and Discop is similar. Those rates hold only for a receiver that reconstructs the sender’s exact distribution, which Section 4 found does not happen in practice.

TABLE 7. PAYLOAD BITS PER POSTED TOKEN. ROBUST: DECODES DESPITE SENDER–RECEIVER MISMATCH. EXACT: STEGOTEXT FOLLOWS THE SENDER’S MODEL DISTRIBUTION. –: IMPRACTICAL AT 256 BITS. <sup>†</sup>RATE REPORTED BY THE AUTHORS.
<table><tr><td>Method</td><td>Robust</td><td>Exact</td><td>32 bits</td><td>256 bits</td></tr><tr><td>Arithmetic coding [20]</td><td>No</td><td>Yes</td><td>0.19-0.28</td><td>0.95-0.99</td></tr><tr><td>Discop [7]</td><td>No</td><td>Yes</td><td>0.19–0.29</td><td>0.81-0.92</td></tr><tr><td>Zamir [23]</td><td>Yes</td><td>Yes</td><td>0.002–0.024</td><td></td></tr><tr><td>Bai et al. [24]†</td><td>Yes</td><td>Yes</td><td>0.002–0.044</td><td></td></tr><tr><td>STEALTH (ours)</td><td>Yes</td><td>Yes</td><td>0.19–0.26</td><td>0.37-0.43</td></tr></table>

Among receiver-free schemes, we reimplemented Zamir’s [23], following its reference code, on the same news channel with a budget of 32 comments, tuning its parameters on held-out messages. It delivers 0.017, 0.024, and 0.002 payload bits per posted token on Gemma, Llama, and Qwen respectively, and decodes 92%, 99.5%, and 60% of messages from its own tokens within that budget. Being receiver-free does not protect it from re-tokenization: decoding from the posted strings instead of the sender’s tokens drops Llama from 99.5% to 78.5%, since one changed token corrupts its binary choices and every symbol’s score after it. We deliver 9–15× more bits per posted token on Gemma and Llama, and nearly 100× more on Qwen, whose low-entropy text starves Zamir’s per-choice evidence. We did not reimplement Bai et al. [24]. Table 7 instead reports their own measured rate as a rough point of comparison.

Rate vs. payload size. Our rate rises with payload size because a comment has to end where the model ends it: a short message is often decodable well before the comment it rides in is over, so most of that comment’s tokens count against the rate for nothing, while a longer message uses more of the same comment before it ends. Figure 5 traces this for the same-model bf16 receiver from 32 to 256 bits. The rate rises by 20–90% from 32 to 64 bits, then flattens into the 0.35–0.45 range by 128 bits, as the unused comment tail shrinks relative to a growing message instead of staying a fixed cost. It does not keep climbing toward the replay-mode rates of arithmetic coding or Discop, which spend nothing on a receiver that reproduces the sender’s distribution bit for bit, while ours turns only about 40% of the text’s entropy into payload, the rest being the redundancy the beam search needs to absorb flips.

Computational cost. Hiding a message adds little time on either side. We timed 50 messages per sender on an A6000, one job per GPU at batch size 1. The exponential race runs within 3–4% of the speed of honest multinomial sampling, at 25–52 ms per token for the 7–9B models, and the sender’s decodability checks add another 6–7%, for 4.2–7.0 s per message. Our receiver scores the posted text in one teacher-forced pass (44–69 ms) and runs the beam search (24–27 ms), for 70–95 ms per message on the 7–9B models. Arithmetic coding sends at the same speed but pays at the receiver: in bf16 it decodes only by regenerating the sender’s text token by token, at 3.7–5.5 s per message, 50– 60× ours. The float64 hardening of Section 4 costs more still: 2.5–5.5× our time per token on the sender, and 5.8– 16.5 s per message on the receiver against our 55–71 ms on the same 1.5–3B models. Zamir’s scheme also generates at the model’s speed, but its low rate means a 32-bit message takes a median of 1,200–7,200 tokens, or 33–196 s to send.

![](images/9869438819fd8fc61d9071b944ed5606cd564154afa16e6b2788f8f00ba30e55.jpg)  
Figure 5. Payload bits per posted token rises with payload size and then flattens, as the unused tail of the last comment shrinks relative to the whole message (median over 200 messages).

## 7.5. Detection Tests

The proof of Section 6 covers every efficient warden; here we check that real detectors agree. Each test compares the first comment posted on each article, 2,000 per family and kind: stego, our scheme’s output; race, an honest sample from the same exponential race run on fresh rather than keyed noise; and mult, an honest sample from ordinary multinomial sampling, Hugging Face’s default. Stego-vsrace and stego-vs-mult are the tests that matter. Race-vs-mult compares two honest samplers and shows what chance looks like. As a positive control, ac12 is arithmetic coding run deliberately at a coarse 12-bit interval precision, a scheme with known statistical artifacts.

Trained steganalyzers. We re-implemented three wordlevel linguistic steganalyzers from their published descriptions, with embeddings trained from scratch: FCN [30], R-BiLSTM-C [31], and a densely connected LSTM with feature pyramid [32]. Alongside them we fine-tune RoBERTa [33] end to end as a transformer-based baseline. Each is trained and evaluated with 5-fold cross-validation grouped by article, so no article’s texts appear in both training and test, and we report out-of-fold AUC with a 95% bootstrap interval. Table 8 gives RoBERTa, the strongest of the four on the positive control. On every stego-vsrace, stego-vs-mult, and race-vs-mult pair, RoBERTa’s AUC is within 0.007 of chance and the other three are within 0.015, none significant $( | z | \leq 1 . 3 )$ . The control shows how little power these text-only classifiers have on distributionpreserving schemes: on only 200 texts per class, RoBERTa detects ac12 on Llama and Qwen $( z = 2 . 9 3 , 2 . 4 3 )$ but not on Gemma, the densely connected LSTM detects it on Llama alone $( z = 2 . 4 0 )$ , and FCN and R-BiLSTM-C never do.

TABLE 8. ROBERTA STEGANALYZER AUC (0.5 = CHANCE). <sup>∗</sup>: $p < 0 . 0 5$ AGAINST CHANCE.
<table><tr><td>Pair</td><td>Gemma</td><td>Llama</td><td>Qwen</td></tr><tr><td>stego vs. mult</td><td>0.503</td><td>0.498</td><td>0.507</td></tr><tr><td>stego vs. race</td><td>0.501</td><td>0.500</td><td>0.499</td></tr><tr><td>race vs. mult (control)</td><td>0.499</td><td>0.500</td><td>0.496</td></tr><tr><td>ac12 vs. mult (positive control)</td><td>0.515</td><td>0.585*</td><td>0.570*</td></tr></table>

A detector that knows the model. Our warden knows the sender’s model (Section 3), so the stronger test scores each comment with that model. For every token it computes the token’s log-probability, rank, the distribution’s entropy, and a randomized probability integral transform (PIT), which is uniform on [0, 1] under honest sampling. Each comment becomes a vector of histograms and moments of these, tail counts, PIT autocorrelation at lags 1–12 (the spine value repeats every 12 words), and length. Logistic regression, gradient boosting, and an MLP are trained on these features with the same grouped cross-validation. Across the three families and classifiers, stego-vs-mult and stego-vsrace AUCs lie in 0.48–0.52, inside the 0.48–0.54 spread of the honest race-vs-mult pair. The same detector finds ac12 in every family, at AUC up to 0.67 (z = 6.0).

The missing-tail test. Cao et al.’s test [34] targets schemes that can never emit some tokens, the failure Section 6.3’s unreachable-mass bound rules out for STEALTH. It counts how many tokens land in the extreme tail of the sender model’s distribution, below a threshold probability, and compares that count with the number honest sampling predicts, pooled into a z-score over a corpus. Positive z is a deficit of tail tokens, the signature Cao et al. test for; negative z is an excess, which we also count as detection. Table 9 pools all three families, about 880,000 tokens per honest or stego kind, at three thresholds. Stego and race stay within $| z | < 1 . 7$ at every threshold. The honest mult sampler shows a real deficit at $2 ^ { - 1 6 }$ and $2 ^ { - 2 4 } \ ( z = 3 . 7 ,$ 6.4): torch.multinomial draws through a float32 cumulative sum, which under-samples the most extreme tail. The test is correctly flagging a known flaw in that particular honest sampler. Our race reads probabilities directly and has no such flaw. Arithmetic coding at 32-bit precision, the exact baseline of Section 4, is borderline at $2 ^ { \overset { \cdot } { - } 1 6 }$ and $2 ^ { - 2 4 }$ (z ≈ −2). At 16 bits it is caught at $2 ^ { - 3 2 } ~ ( z = - 1 8 . 5 )$ , and at 12 bits at every threshold, down $\mathrm { t o } ~ z = - 3 8 5$

What these tests show. A score near chance means only that a test with this much power found nothing, so the controls matter as much as the stego rows. Both modelaware tests catch ac12 in every family, and the missing-tail test also catches 16-bit arithmetic coding and the float32 flaw in an honest sampler. That test’s power falls with its threshold. $\mathrm { \ A t \ 2 ^ { - 2 4 } }$ honest sampling expects about 74 tail tokens across the corpus, so a sampler that never reached that tail would score $z \approx 8 . 6 ;$ at $2 ^ { - 3 2 }$ it expects fewer than two, so even a fully missing tail would score only $z = 1 . 3 ,$ as mult does. STEALTH’s unreachable mass, $1 . 4 \times \mathrm { \dot { 1 } 0 ^ { - 1 7 } }$ per token (Section 6.3), lies far below what any such test can resolve. No finite set of tests rules out every warden; that is the job of the reduction in Section 6. What these results add is that detectors able to find small artifacts in other samplers find none in ours.

TABLE 9. CAO ET AL.’S TEST POOLED OVER THE THREE FAMILIES, BY TAIL THRESHOLD. POSITIVE: FEWER TAIL TOKENS THAN HONEST SAMPLING PREDICTS; NEGATIVE: MORE.
<table><tr><td>Kind</td><td> $2 ^ { - 1 6 }$ </td><td> $2 ^ { - 2 4 }$ </td><td> $2 ^ { - 3 2 }$ </td></tr><tr><td>stego</td><td>0.3</td><td>1.6</td><td>-0.2</td></tr><tr><td>race (control)</td><td>-1.1</td><td>1.1</td><td>-1.6</td></tr><tr><td>mult (control)</td><td>3.7</td><td>6.4</td><td>1.3</td></tr><tr><td>ac, 32-bit</td><td>-1.9</td><td>-2.1</td><td>0.4</td></tr><tr><td>ac, 16-bit</td><td>-1.0</td><td>-1.7</td><td>-18.5</td></tr><tr><td>ac, 12-bit</td><td>-8.3</td><td>-56.4</td><td>-385.2</td></tr></table>

## 8. Conclusion

Provably secure linguistic steganography has assumed that the receiver reproduces the sender’s token distribution bit for bit. Section 4 shows that ordinary deployments break this assumption: a different precision, quantization level, model size, or GPU leaves arithmetic coding and Discop decoding almost no messages. STEALTH drops the assumption. Its sender picks each token by a keyed exponential race, so a receiver whose distribution differs slightly still picks the same token almost everywhere, and a disagreement costs only that one token’s evidence. Because the spinal code is rateless, every further token adds evidence for the whole payload, so a receiver that disagrees more often simply needs more text.

Across three model families, this held for every receiver we tested. Given enough text, every same-family receiver, quantized or smaller and on either GPU, decodes every message, and with the noise keyed by token bytes, so does every receiver of another family. The sender is also free to generate with speculative decoding, with no measurable effect on its receivers. Robustness costs no security: Section 6 proves the posted texts indistinguishable from honest samples of the sender’s model, and no detector we tried, including one that scores text with that model, tells them apart (Section 7.5). Nor does it cost practicality: STEALTH delivers 0.19–0.26 payload bits per posted token on 32-bit messages, an order of magnitude more than the receiver-free scheme of Zamir [23], and its race runs within a few percent of honest sampling.

Sender and receiver therefore no longer need matching deployments to communicate covertly. They can run different hardware, different precisions, or different model families, and the texts they exchange remain provably indistinguishable from honest model output.

## References

[1] S. Farrell and H. Tschofenig, “Pervasive monitoring is an attack,” RFC 7258 (BCP 188), IETF, 2014.

[2] M. Wu, J. Sippe, D. Sivakumar, J. Burg, P. Anderson, X. Wang, K. Bock, A. Houmansadr, D. Levin, and E. Wustrow, “How the great firewall of China detects and blocks fully encrypted traffic,” in 32nd USENIX Security Symposium, 2023, pp. 2653–2670.

[3] N. J. Hopper, J. Langford, and L. von Ahn, “Provably secure steganography,” in Advances in Cryptology – CRYPTO 2002, ser. LNCS, vol. 2442, 2002, pp. 77– 92.

[4] A. D. Ker, P. Bas, R. Bohme, R. Cogranne, S. Craver,¨ T. Filler, J. Fridrich, and T. Pevny, “Moving steganog-´ raphy and steganalysis from the laboratory into the real world,” in ACM Workshop on Information Hiding and Multimedia Security (IH&MMSec), 2013, pp. 45–58.

[5] G. Kaptchuk, T. M. Jois, M. Green, and A. D. Rubin, “Meteor: Cryptographically secure steganography for realistic distributions,” in ACM Conference on Computer and Communications Security (CCS), 2021, iACR ePrint 2021/686.

[6] C. S. de Witt, S. Sokota, J. Z. Kolter, J. Foerster, and M. Strohmeier, “Perfectly secure steganography using minimum entropy coupling,” in International Conference on Learning Representations (ICLR), 2023, arXiv:2210.14889.

[7] J. Ding, K. Chen, Y. Wang, N. Zhao, W. Zhang, and N. Yu, “Discop: Provably secure steganography in practice based on “distribution copies”,” in IEEE Symposium on Security and Privacy (S&P), 2023.

[8] B. Atil, S. Aykent, A. Chittams, L. Fu, R. J. Passonneau, E. Radcliffe, G. R. Rajagopal, A. Sloan, T. Tudrej, F. Ture, Z. Wu, L. Xu, and B. Baldwin, “Non-determinism of “deterministic” LLM settings,” 2024, arXiv:2408.04667.

[9] J. Yuan, H. Li, X. Ding, W. Xie, Y.-J. Li, W. Zhao, K. Wan, J. Shi, X. Hu, and Z. Liu, “Give me FP32 or give me death? challenges and solutions for reproducible reasoning,” 2025, arXiv:2506.09501.

[10] Thinking Machines Lab, “Defeating nondeterminism in LLM inference,” https://thinkingmachines.ai/blog/ defeating-nondeterminism-in-llm-inference/, 2025.

[11] R. Yan, C. Chu, Z. Yang, and Y. Murawaki, “A comprehensive survey on linguistic steganography: Methods, countermeasures, evaluation, and challenges,” in Proceedings of EMNLP, 2026.

[12] M. Christ, S. Gunn, and O. Zamir, “Undetectable watermarks for language models,” in 37th Conference on Learning Theory (COLT), ser. PMLR, vol. 247, 2024, pp. 1125–1139, arXiv:2306.09194.

[13] R. Kuditipudi, J. Thickstun, T. Hashimoto, and P. Liang, “Robust distortion-free watermarks for language models,” Transactions on Machine Learning Research, 2024, arXiv:2307.15593.

[14] J. Perry, P. A. Iannucci, K. E. Fleming, H. Balakrishnan, and D. Shah, “Spinal codes,” in ACM SIGCOMM,

2012.

[15] D. Xue, M. Kallitsis, A. Houmansadr, and R. Ensafi, “Fingerprinting obfuscated proxy traffic with encapsulated TLS handshakes,” in 33rd USENIX Security Symposium, 2024, pp. 2689–2706.

[16] L. Dixon, T. Ristenpart, and T. Shrimpton, “Network traffic obfuscation and automated internet censorship,” IEEE Security & Privacy, vol. 14, no. 6, pp. 43–53, 2016.

[17] C. Cachin, “An information-theoretic model for steganography,” in 2nd Workshop on Information Hiding (IH), ser. LNCS, vol. 1525, 1998, pp. 306–318.

[18] N. F. Johnson and S. Jajodia, “Exploring steganography: Seeing the unseen,” IEEE Computer, vol. 31, no. 2, pp. 26–34, 1998.

[19] J. Fridrich, M. Goljan, and R. Du, “Reliable detection of LSB steganography in color and grayscale images,” in Proceedings of the ACM Workshop on Multimedia and Security (MM&Sec), Ottawa, Canada, 2001, p. 27.

[20] Z. M. Ziegler, Y. Deng, and A. M. Rush, “Neural linguistic steganography,” in Proceedings of EMNLP-IJCNLP, 2019.

[21] T. Fu, G. Mart´ınez, J. Conde, C. Arriaga, X. Qi, S. Liu, and P. Reviriego, “Beyond reproducibility: Token probabilities expose large language model nondeterminism,” IEEE Transactions on Computers, 2026.

[22] T. Dettmers, A. Pagnoni, A. Holtzman, and L. Zettlemoyer, “QLoRA: Efficient finetuning of quantized LLMs,” in Advances in Neural Information Processing Systems (NeurIPS), 2023, arXiv:2305.14314.

[23] O. Zamir, “Excuse me, sir? your language model is leaking (information),” Transactions on Machine Learning Research (TMLR), 2024.

[24] M. Bai, J. Yang, K. Pang, X. Xu, Z. Yang, and Y. Huang, “Provably robust and secure steganography in asymmetric resource scenario,” in IEEE Symposium on Security and Privacy (S&P), 2025, pp. 1438–1456.

[25] S. Guo, S. Vithana, A. Gilani, L. Sankar, O. Kosut, and F. P. Calmon, “Feedback coding enables inference-time covert agentic communication,” 2026, arXiv:2609.24994.

[26] L. A. Bauer, J. K. H. IV, S. A. Markelon, V. Bindschaedler, and T. Shrimpton, “Leveraging generative models for covert messaging: Challenges and tradeoffs for “dead-drop” deployments,” in ACM Conference on Data and Application Security and Privacy (CO-DASPY), 2024, pp. 67–78, arXiv:2110.07009.

[27] J. I. Yellott, Jr., “The relationship between luce’s choice axiom, thurstone’s theory of comparative judgment, and the double exponential distribution,” Journal of Mathematical Psychology, vol. 15, no. 2, pp. 109– 144, 1977.

[28] C. J. Maddison, D. Tarlow, and T. Minka, “A\* sampling,” in Advances in Neural Information Processing Systems (NeurIPS), 2014.

[29] S. Aaronson, “My AI safety lecture for UT effective altruism,” https://scottaaronson.blog/?p=6823, 2022.

[30] Z. Yang, Y. Huang, and Y.-J. Zhang, “A fast and

efficient text steganalysis method,” IEEE Signal Processing Letters, vol. 26, no. 4, pp. 627–631, 2019.

[31] Y. Niu, J. Wen, P. Zhong, and Y. Xue, “A hybrid R-BiLSTM-C neural network based text steganalysis,” IEEE Signal Processing Letters, vol. 26, no. 12, pp. 1907–1911, 2019.

[32] H. Yang, Y. Bao, Z. Yang, S. Liu, Y. Huang, and S. Jiao, “Linguistic steganalysis via densely connected LSTM with feature pyramid,” in Proceedings of the ACM Workshop on Information Hiding and Multimedia Security (IH&MMSec), 2020, pp. 5–10.

[33] Y. Liu, M. Ott, N. Goyal, J. Du, M. Joshi, D. Chen, O. Levy, M. Lewis, L. Zettlemoyer, and V. Stoyanov, “RoBERTa: A robustly optimized BERT pretraining approach,” arXiv:1907.11692, 2019.

[34] W. Cao, Y. Wang, and D. Hu, “Breaking the “provable security”: Detecting finite-precision artifacts in LLMbased steganography via low-probability vanishing,” in Findings ofthe Associationfor Computational Linguistics (ACL), 2026, pp. 20 262–20 274.

[35] A. See, P. J. Liu, and C. D. Manning, “Get to the point: Summarization with pointer-generator networks,” in Proceedings of the 55th Annual Meeting of the $A s \mathrm { - }$ sociation for Computational Linguistics (ACL), 2017, pp. 1073–1083.

[36] Gemma Team, “Gemma 2: Improving open language models at a practical size,” 2024, arXiv:2408.00118.

[37] Llama Team, AI @ Meta, “The llama 3 herd of models,” 2024, arXiv:2407.21783.

[38] Qwen Team, “Qwen2.5 technical report,” 2024, arXiv:2412.15115.

[39] Meta, “Llama 3.2 model card,” https://github.com/ meta-llama/llama-models/blob/main/models/llama3 2/ MODEL CARD.md, 2024, release date: October 24, 2024.

[40] M. Daliri, C. Musco, and A. T. Suresh, “Coupling without communication and drafter-invariant speculative decoding,” in Proceedings of the IEEE International Symposium on Information Theory (ISIT), 2025, pp. 1–6.

[41] M. Bellare, “New proofs for NMAC and HMAC: Security without collision-resistance,” in Advances in Cryptology – CRYPTO 2006, ser. LNCS, vol. 4117, 2006, iACR ePrint 2006/043.

[42] D. J. Bernstein, “ChaCha, a variant of Salsa20,” https:// cr.yp.to/chacha/chacha-20080128.pdf, 2008, document ID 4027b5256e17b9796842e6d0f68b0b5e.

[43] Y. Leviathan, M. Kalman, and Y. Matias, “Fast inference from transformers via speculative decoding,” in International Conference on Machine Learning (ICML), 2023, pp. 19 274–19 286.

[44] C. Chen, S. Borgeaud, G. Irving, J.-B. Lespiau, L. Sifre, and J. Jumper, “Accelerating large language model decoding with speculative sampling,” 2023, arXiv:2302.01318.

## Appendix A. Proofs

Proof of Lemma 1. Scaling an Exp(1) variable by $1 / p _ { i } ( h )$ gives $\dot { E _ { i } } / p _ { i } ( h ) \sim \mathrm { E x p } ( p _ { i } ( h ) )$ , so the race scores are independent exponentials with rates $p _ { i } ( h )$ summing to 1. For independent exponentials, the index achieving the minimum wins with probability its own rate over the sum of all rates, here $p _ { i } ( h )$ , and the minimum itself is exponential with rate equal to that sum, here Exp(1), independent of the index. □

Proof of Lemma 2. A truly random function’s values at distinct inputs are independent and uniform; querying it at an input not yet read returns a value independent of every value read so far (lazy sampling). Positions strictly increase, so every noise value read before position t, whether to sample a token or to run a decodability check, has a first component $t ^ { \prime } < t \colon$ the decodability check only replays decoding over tokens already written, so it never queries position t or later. No input at t has therefore been queried before, so by lazy sampling its value is a fresh independent uniform, regardless of which spine value selects it. □

Proof of Corollary 1. By Lemmas 1 and 2, each token, given everything chosen before it, is distributed exactly as ${ \hat { p } } ( h )$ for the current context $h ;$ END and the prompt policy are those of the honest process, so the token sequence follows the honest transition law at every step. □

Proof of Theorem 1. We bound $\operatorname { A d v } ( A )$ through a sequence of hybrids, starting at the real stego oracle and ending at the honest oracle.

$H _ { 0 }$ is the real stego oracle $O _ { S }$

$H _ { 1 }$ replaces HMAC-SHA256(K, ·) by a random function; since the warden never repeats a nonce (Definition 1), the message keys $K _ { m }$ it induces are independent and uniform. A warden distinguishing $H _ { 0 }$ from $H _ { 1 }$ yields a distinguisher for HMAC-SHA256: it runs the warden, simulates the sender, and obtains each $K _ { m }$ by querying its own oracle on the nonce $\nu ,$ so $| \operatorname* { P r } _ { H _ { 0 } } - \operatorname* { P r } _ { H _ { 1 } } | \le \epsilon _ { \mathrm { H M A C } } ( q )$

$H _ { 2 }$ replaces the ChaCha20 block function under each of the $q$ message keys by an independent random function; a standard hybrid over the q messages gives $| \mathrm { P r } _ { H _ { 1 } } - \mathrm { P r } _ { H _ { 2 } } | \leq$ $q \in \mathrm { { C h a C h a } } ( r )$

$H _ { 3 }$ replaces the sender’s implemented race by the ideal race on exact uniforms. $H _ { 2 }$ and $H _ { 3 }$ read the same noise words and draw the same warden coins, so they agree until the first position where the two races disagree. By Lemma 2 the noise at that position is fresh given everything before it, so by Lemma 3 the races disagree there with probability at most $\delta ;$ a union bound over at most $T$ positions gives $| \operatorname* { P r } _ { H _ { 2 } } - \operatorname* { P r } _ { H _ { 3 } } | \ \leq \ T \delta$ . The sender’s decodability checks (Section 5.4) add no term: they query only positions already sampled, and under the theorem’s stopping rule their outcome does not affect what is posted.

Finally, by Corollary 1, each message in $H _ { 3 }$ is an honest conversation cut by a rule that does not use the key; the honest oracle $O _ { H }$ applies the same rule to an honest conversation, and distinct messages use independent random functions throughout $H _ { 1 } { - } H _ { 3 } ,$ , so $H _ { 3 }$ and $O _ { H }$ have exactly the same law. Chaining the four bounds gives the claim.

Proof of Lemma 3. The strategy is to condition on a “good” event Γ in which the winning margin is not too close to a tie and no noise word landed in an unusually wide cell; we bound the probability the good event fails to hold, and separately bound the implementation’s disagreement probability given that it does.

Write $G _ { i } = 1 - U _ { i } , X _ { i } = E _ { i } / \hat { p } _ { i }$ with $E _ { i } = - \ln G _ { i } ,$ i<sup>⋆</sup> = arg min X<sub>i</sub> and $x = X _ { i ^ { \star } }$ . By Lemma $\therefore \mathrm { P r } [ i ^ { \star } = i ] =$ $\hat { p } _ { i } , x \sim \mathrm { E x p } ( 1 )$ is independent of $i ^ { \star }$ , and given $( i ^ { \star } , x )$ the values $u _ { j } = { \hat { p } } _ { j } ( X _ { j } - x ) , j \neq i ^ { \star }$ , are independent Exp(1). Let $I _ { i }$ be the interval of values of $- \ln ( 1 - u ) / \hat { p } _ { i }$ over u in $W _ { i } { ' } \mathbf { s }$ cell $[ W _ { i } \omega , ( W _ { i } + 1 ) \omega )$ , and $\Delta _ { i }$ its length. Write $\hat { S } _ { i }$ for the implementation’s computed log score, within η of ln $\hat { E } _ { i } - \boldsymbol { \ell } _ { i }$ by assumption. Both $X _ { i }$ and $\hat { X } _ { i } = \hat { E } _ { i } / \hat { p } _ { i }$ lie in $I _ { i } ,$ and the computed $\bar { \tilde { X } } _ { i } = \exp ( \hat { S } _ { i } + c )$ , with c the normalizing constant of $\ell ,$ lies within a factor $e ^ { \pm \eta }$ of ${ \hat { X } } _ { i }$

Cell widths. If $G _ { i } \geq 2 \omega .$ , the cell is $[ g , g + \omega )$ in G with $g ~ \ge ~ \omega , ~ \mathsf { s o } ~ - \ln$ varies over it by $\ln ( 1 + \omega / g ) \leq \omega / g \leq$ $\mathsf { 2 } \omega / G _ { i } = 2 \omega e ^ { E _ { i } }$ , and $\Delta _ { i } \leq 2 \omega e ^ { E _ { i } } / \hat { p } _ { i }$

Disagreement needs a close competitor. If the implemented race picks $v \neq i ^ { \star }$

$$
X _ { v } - \Delta _ { v } \le \hat { X } _ { v } \le e ^ { \eta } \tilde { X } _ { v } \le e ^ { \eta } \tilde { X } _ { i ^ { \star } } \le e ^ { 2 \eta } \hat { X } _ { i ^ { \star } } \le e ^ { 2 \eta } ( x + \Delta _ { i ^ { \star } } ) ,
$$

so $X _ { v } - x \le \Delta _ { v } + \Delta _ { i ^ { \star } } + \rho ( x + \Delta _ { i ^ { \star } } )$

Good event. Let Γ be the event that $G _ { i } ~ \geq ~ 8 \omega$ for all $i , \ x \ \leq \ a ,$ and $\begin{array} { r l r } { \Delta _ { i ^ { \star } } } & { { } \leq } & { { \frac { 1 } { 2 } } } \end{array}$ . A union bound gives $\mathrm { P r } [ G _ { i } < 8 \omega$ for some $i ] \le 8 | V | \overline { { \omega } }$ , and $\operatorname* { P r } [ x > a ] = e ^ { - a }$ Since $E _ { i ^ { \star } } = \hat { p } _ { i ^ { \star } } x$ , if $G _ { i ^ { \star } } \geq 2 \omega$ then $\Delta _ { i ^ { \star } } \leq 2 \omega e ^ { \hat { p } _ { i } \star x } / \hat { p } _ { i ^ { \star } } \leq$ $2 \omega e ^ { x } / \hat { p } _ { i ^ { \star } } , \ s _ { 0 } \ \Delta _ { i ^ { \star } } > \frac { 1 } { 2 }$ requires $\hat { p } _ { i ^ { \star } } ~ < ~ 4 \omega e ^ { x }$ . Since i<sup>⋆</sup> is independent of x and $\begin{array} { r } { \mathrm { \tilde { P r } } [ \hat { p } _ { i ^ { \star } } < \theta ] = \sum _ { i : \hat { p } _ { i } < \theta } \hat { p } _ { i } \leq | V | \theta } \end{array}$ this has probability at most 4|V|ω $\begin{array} { r } { \mathbb { E } [ e ^ { x } ; x \stackrel { * } { \le } \dot { a } ] = 4 a | V | \omega . } \end{array}$ So $\operatorname* { P r } [ \bar { \Gamma } ] ^ { - } \leq 8 | V | \omega ^ { - } + e ^ { - a } + 4 a | V | c$ ω. It remains to bound the disagreement probability assuming Γ holds.

On Γ. Multiplying the disagreement condition by $\hat { p } _ { v } \leq 1$ and using $\begin{array} { r } { \hat { p } _ { v } \Delta _ { v } \leq 2 \omega e ^ { E _ { v } } \leq \breve { \frac { 1 } { 4 } } } \end{array}$ (as $G _ { v } \geq 8 \omega )$ gives $u _ { v } \leq$ $\begin{array} { r } { \frac { 1 } { 2 } + \frac { 1 } { 4 } + \overline { { \rho ( a + \frac { 1 } { 2 } ) } } \leq 1 } \end{array}$ . Hence $E _ { v } = \hat { p } _ { v } x + u _ { v } \le x + 1$ , so $\bar { \hat { p } } _ { v } \Delta _ { v } \hat { \asymp } 2 e \omega e ^ { x }$ , and

$$
\begin{array} { r } { u _ { v } \leq c _ { v } : = 2 \omega e ^ { x } \big ( \hat { p } _ { v } / \hat { p } _ { i ^ { \star } } + e \big ) + \hat { p } _ { v } \rho ( x + \frac { 1 } { 2 } ) . } \end{array}
$$

Given (i<sup>⋆</sup> , x), $\begin{array} { r c l r } { \mathrm { P r } [ u _ { j } } & { \leq } & { c _ { j } ] } & { \leq } & { c _ { j } } \end{array}$ for each $j ,$ and $\begin{array} { r } { \sum _ { i \not = i ^ { \star } } c _ { j } \leq 2 \omega e ^ { x } ( \mathrm { i } / \bar { \hat { p } } _ { i ^ { \star } } + e | \bar { V } | ) + \rho ( x + \frac { 1 } { 2 } ) } \end{array}$ . Averaging over $\bar { i ^ { \star } } ^ { \prime } ( \mathbb { E } [ 1 / \hat { p } _ { i ^ { \star } } ] = | \{ i : \hat { p } _ { i } > 0 \} | \leq | V | )$ and then over $x \leq a$ $( \mathbb { E } [ e ^ { x } ; x \leq a ] = { \overset { \cdot } { a } } , \mathbb { E } [ x ] + { \frac { 1 } { 2 } } = { \frac { 3 } { 2 } } )$ bounds the disagreement probability on Γ by $\begin{array} { r } { 2 ( 1 + e ) \tilde { \alpha } | V | \tilde { \omega } + \frac { 3 } { 2 } \rho . } \end{array}$ . Adding Pr[Γ] <sup>¯</sup> gives the claim. □

## Appendix B.

## Speculative Decoding Results

Table 10 gives the per-receiver results behind the speculative-decoding comparison of Section 7.2. On each GPU, the sender drafts four tokens per round with a small model of its own family (Gemma 2B, Llama 1B, Qwen

0.5B) and keeps them up to the first position where its own race disagrees. Every receiver decodes the posted text at the sender’s stopping point with the cascade decoder. The H100 plain columns are the runs behind Table 4, including the fp16 and fp32 receivers it omits, given as counts of 200 rather than percentages; the A6000 columns repeat the comparison with texts written and decoded on the A6000.

TABLE 10. PLAIN AND SPECULATIVE SENDERS ON EACH GPU: FLIP RATE (%) AND MESSAGES DECODED OF 200 AT THE SENDER’S STOPPING POINT, WITH THE CASCADE DECODER.
<table><tr><td></td><td></td><td colspan="4">H100</td><td colspan="4">A6000</td></tr><tr><td></td><td></td><td colspan="2">Plain</td><td colspan="2">Spec.</td><td colspan="2">Plain</td><td colspan="2">Spec.</td></tr><tr><td>Family</td><td>Receiver</td><td>Flips</td><td>Dec.</td><td>Flips</td><td>Dec.</td><td>Flips</td><td>Dec.</td><td>Flips</td><td>Dec.</td></tr><tr><td>Gemma</td><td>9B bf16</td><td>0.9</td><td>200</td><td>1.0</td><td>199</td><td>0.9</td><td>200</td><td>0.9</td><td>199</td></tr><tr><td></td><td>9B fp16</td><td>1.3</td><td>199</td><td>1.4</td><td>200</td><td>1.3</td><td>200</td><td>1.3</td><td>199</td></tr><tr><td></td><td>9B fp32</td><td>1.3</td><td>199</td><td>1.4</td><td>200</td><td>1.3</td><td>200</td><td>1.3</td><td>199</td></tr><tr><td></td><td>9B int8</td><td>3.8</td><td>196</td><td>4.0</td><td>199</td><td>4.2</td><td>199</td><td>3.9</td><td>198</td></tr><tr><td></td><td>9B NF4</td><td>8.4</td><td>187</td><td>8.9</td><td>188</td><td>8.9</td><td>189</td><td>8.7</td><td>186</td></tr><tr><td></td><td>2B bf16</td><td>31.5</td><td>84</td><td>31.2</td><td>101</td><td>31.6</td><td>87</td><td>31.4</td><td>90</td></tr><tr><td>Llama</td><td>8B bf16</td><td>1.0</td><td>200</td><td>1.0</td><td>200</td><td>1.1</td><td>200</td><td>1.0</td><td>200</td></tr><tr><td></td><td>8B fp16</td><td>1.0</td><td>200</td><td>0.9</td><td>200</td><td>1.0</td><td>200</td><td>1.0</td><td>200</td></tr><tr><td></td><td>8B fp32</td><td>1.1</td><td>200</td><td>0.9</td><td>200</td><td>1.1</td><td>200</td><td>1.0</td><td>200</td></tr><tr><td></td><td>8B int8</td><td>4.9</td><td>200</td><td>4.6</td><td>200</td><td>4.7</td><td>200</td><td>4.8</td><td>200</td></tr><tr><td></td><td>8B NF4</td><td>10.2</td><td>196</td><td>10.1</td><td>196</td><td>10.3</td><td>197</td><td>10.2</td><td>198</td></tr><tr><td></td><td>3B bf16</td><td>23.5</td><td>175</td><td>23.6</td><td>180</td><td>23.6</td><td>185</td><td>23.6</td><td>180</td></tr><tr><td>Qwen</td><td>7B bf16</td><td>1.2</td><td>200</td><td>1.2</td><td>200</td><td>1.2</td><td>200</td><td>1.3</td><td>200</td></tr><tr><td></td><td>7B fp16</td><td>1.3</td><td>200</td><td>1.3</td><td>200</td><td>1.4</td><td>200</td><td>1.3</td><td>200</td></tr><tr><td></td><td>7B fp32</td><td>1.3</td><td>200</td><td>1.3</td><td>200</td><td>1.3</td><td>200</td><td>1.3</td><td>200</td></tr><tr><td></td><td>7B int8</td><td>4.2</td><td>200</td><td>4.2</td><td>200</td><td>4.3</td><td>199</td><td>4.3</td><td>200</td></tr><tr><td></td><td>7BNF4</td><td>12.1</td><td>199</td><td>12.0</td><td>200</td><td>12.3</td><td>199</td><td>12.3</td><td>198</td></tr><tr><td></td><td>1.5B bf16</td><td>36.9</td><td>168</td><td>37.1</td><td>173</td><td>37.2</td><td>173</td><td>37.1</td><td>178</td></tr></table>