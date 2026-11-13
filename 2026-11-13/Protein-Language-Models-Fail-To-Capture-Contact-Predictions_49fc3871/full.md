# Protein Language Models Fail To Capture Contact Predictions Under Invariant Relabeling

Paarth Batra<sup>∗</sup>   
The Biodesign Institute   
Tempe, AZ 85281, USA   
pbatra6@asu.edu   
Aryan Keluskar<sup>∗</sup>   
The Biodesign Institute   
Tempe, AZ 85281, USA   
akeluska@asu.edu

## Abstract

Protein language models (pLMs) produce highly accurate contact maps from single-sequence inputs. People often take this accuracy as proof that these models implicitly learn invariant coevolutionary statistics, extracting the same signals as classical alignment-based methods like Potts. In this work, we introduce a diagnostic framework around a global bijective relabeling of the amino-acid alphabet. This transformation permutes residue letters while leaving all alignment-level information, column entropies, and pairwise edit distances unchanged. Pseudolikelihood Potts models and inverse-covariance estimators remain invariant under this transformation, but pLMs fail the test. Across multiple architectures (ESM-2, MSA Transformer, AMPLIFY) and model scales (8M to 3B parameters), pLM contact maps collapse under arbitrary relabeling. Using Hamming distance controls and biochemical grouping interventions, we attempt to isolate what likely causes this degradation and find that these models rely on explicit, canonical residue identities (“alphabet anchoring”) rather than pure coevolutionary statistics. Increasing scale makes models more responsive to biochemical classes, but their performance still degrades under relabeling. Our findings show a gap between how pLMs and classical coevolutionary models compute contacts. More broadly, our work shows how diagnostic perturbations that leave classical models provably unchanged can expose hidden representation biases and prevent over-reliance on neural black boxes in biological modeling.

## 1 Introduction

Protein language models (pLMs) trained on masked-residue prediction alone recover structural and functional information from a single sequence, with no alignment in the pipeline [24]. ESM-2 extends this to atomic-resolution structure prediction at speeds orders of magnitude beyond alignment-based methods [14]. Contact maps are the standard, canonical way to show that pLMs encode structure [28]. Their accuracy is widely interpreted as evidence that the models have internalized coevolution, and therefore learned the signal that direct-coupling analysis extracts explicitly from a multiple sequence alignment (MSA) by fitting a Potts model [20]. Further work shows that ESM-2 stores coevolutionary statistics in a coupling tensor [31].

However, two mechanisms can produce high contact precision on natural proteins. A model may learn to invert the covariance of the alignment’s column statistics, the frequencies of residues at each position and of residue pairs across positions, or it may pattern-match on which amino-acid letters occupy each position. On a leakage-clean split, contact readouts lose 30–36 precision points, which shows that letter identity carries predictive signal independent of covariance [26]. Since precision alone cannot distinguish these mechanisms, the right diagnostic [27] is a test that the hypothesized mechanism must pass regardless of the model’s accuracy [9].

(a)  
![](images/f868462f8d0329e37a8b4b4c5cd2292453a3c1ac62d0452b71c2cabafa847700.jpg)  
Figure 1: A global relabeling of the amino-acid alphabet leaves every alignment statistic intact and destroys the pLM’s contact map. (a) The intervention: one bijection π on the twenty amino acids, applied to every residue of every sequence. Both ∆s are measured on 4,000 sequences of the ubiquitin family alignment. (b) The result on the same input. Lower triangle: long-range contacts of the deposited chain (1UBQ\_A); upper triangle: each model’s own top-L/2 prediction, over the candidate pairs the alignment covers. Potts returns a bit-identical score matrix (max|∆S| = 2×10<sup>−14</sup>, Proposition 2), while ESM-2 650M keeps 2 of its own 38 contacts (overlap 0.05, chance 0.025).

In this work, we test the coevolution interpretation by renaming every amino acid consistently across all sequences in a family. The renaming moves each column’s statistics onto new letters without changing their values, so any estimator that reads only column statistics must return the same contacts. We prove this for the pseudolikelihood Potts model [7] and, in closed form, for the inverse-covariance estimator [12]. By the contrapositive, a model whose contact predictions change under relabeling is not computing them from column statistics alone. Unlike shuffle controls [23] or reduced alphabets [11], a bijective relabeling loses no information [2]. To separate edit distance from biochemistry, we add relabelings confined to groups of similar residues [3, 10] and a matched-Hamming control that substitutes randomly at exactly the positions a relabeling changes.

The classical Potts estimator [15] passes this test, and every pLM we examine fails(Table 1, Figure 1). Averaged over nine proteins, ESM-2 650M keeps an overlap of only 0.015 with its original contacts once all twenty letters are relabeled at random; MSA Transformer and AMPLIFY fail the same way. This failure is not caused by input difficulty: a relabeling confined to similar groups keeps an overlap of 0.655. On cytochrome c, random substitutions at relabeled positions destroy the map. Larger models tolerate biochemically similar swaps better, but arbitrary relabeling is as destructive across the board(Figure 3a). The failure also costs accuracy: on ubiquitin, over a candidate set where the two precisions are directly comparable, ESM-2 goes from 2.33× the Potts precision to 0.17× under relabeling. Contact precision therefore cannot be cited as proof that a pLM computes contacts through coevolution.

To summarize, our contributions in this work are as follows:

• We introduce a global relabeling of the amino-acid alphabet as a test of whether a contact readout uses only column statistics, and prove that pseudolikelihood Potts models and inverse-covariance estimators pass it (§3.1).

• We show that every pLM we test fails it, at every ESM-2 size, and that for ESM-2 650M the failure replicates on its supervised contact head and on variant-effect prediction (§4.1, §4.3).

• Using biochemistry-preserving relabelings and a matched-Hamming control, we separate the contribution of residue identity from edit distance, and term the phenomenon of models’ reliance on residue identity "alphabet anchoring" (§4.2, §5).

## 2 Background and Related Work

Coevolutionary contact prediction. Classical contact prediction reads structure from the secondorder column statistics of a multiple sequence alignment (MSA). By fitting a maximum-entropy Potts model [29, 20] or inverting a regularized covariance matrix [12], these methods separate direct coevolution between two positions from indirect correlation through others. Protein language models (pLMs) compute no alignment. Their contacts can be read from the categorical Jacobian ${ \bar { J } } [ i , a , j , b ]$ the sensitivity of the prediction at position j for amino acid b to setting position i to amino acid a [26, 31]. The Jacobian has the same shape as the Potts coupling tensor, and the same post-processing turns either into a contact map: mean-centering, symmetrization, the Frobenius norm over amino-acid pairs, and average product correction (APC) [6, 15]. This shared form is the basis for the claim that pLMs implicitly learn coevolution. Appendix A gives the full pipeline.

Evaluation beyond accuracy. Prior evaluations of pLM representations often rely on downstream accuracy benchmarks [4, 5, 30], which cannot tell a model that computes the intended quantity from one that relies on the distribution of natural sequences. Following out-of-distribution diagnostics in general language modeling [32], we shift the input in a way that changes no information in the alignment’s column statistics. Because the classical baseline is exactly invariant to this shift, the test does not depend on the choice of benchmark (§5.2).

Input controls for contact prediction. Rao et al. [23] probed MSA-based models along two axes: shuffling values within a column, which destroys covariation, and shuffling the order of columns, which destroys position. They also hypothesized that deeper transformer layers become invariant to amino-acid identity. Reduced amino-acid alphabets [11] change column entropy and discard information, so a failure under reduction shows only that the input got harder. However, none of these controls changes residue identity while leaving the alignment’s statistics intact. We add that axis: a bijective relabeling of the alphabet, which loses no information [2], and which tests at the level of contacts the invariance that Rao et al. [23] hypothesized.

## 3 Methodology

We investigate whether the contact readout of a protein language model behaves like a function of the alignment’s column statistics. Our methodology centers on three research questions:

• RQ1. Does a protein language model’s contact prediction remain invariant under a global bijective relabeling of the amino-acid alphabet?

• RQ2. When a neural model’s contact prediction degrades under relabeling, is the degradation driven by the disruption of biochemical similarity classes or by the increase in edit distance?

• RQ3. Does this degradation shrink with model scale, and does it cost accuracy against solved structures?

## 3.1 Classical Estimators Are Invariant to a Global Relabeling

A global bijection on the twenty standard amino acids moves each column’s statistics onto new letters without changing their values. Since column entropies, inter-column mutual information, and the true contact map are all unchanged by construction, any function of column statistics must be exactly invariant under the bijection. We prove this for two estimators in Propositions 1 and 2, with full proofs in Appendix A.

The inverse-covariance estimator satisfies the condition exactly (Proposition 2), and it remains accurate on real proteins such as ubiquitin. A pseudolikelihood Potts fit under L-BFGS returns an identical top- $L / 2$ contact set from an equivariant initialization [25] and 0.970 overlap from a random one (Appendix B). Both conditions of the theorem are needed: making the ridge anisotropic breaks the invariance (max| $\Delta S | = 0 . 3 5 5 )$ , and so does letting the gap symbol enter the permuted set (0.182), against about $1 0 ^ { - 1 5 }$ when the conditions hold.

## 3.2 Intervention Design

A relabeled sequence differs from the original at many positions, so a degradation could come from edit distance alone. We therefore use three regimes, each moving m of the twenty residue types, to separate edit distance from biochemical class.

Random is a uniform derangement of m residue types. In-class is confined to BLOSUM62-similar groups (IVLM, KR, DE, NQ, ST, FYW), preserving biochemistry while changing identity [21]. Cross-class moves each residue it changes into a different group, at the same edit budget. Figure 4 shows the permutation maps. At nominal $m = 2 0$ , in-class moves only 15 residue types, because A, C, G, H and P have no BLOSUM62 partner, and cross-class moves 14 to 16 (Appendix C). Tables and text give nominal m; figures plot each point at its realized m.

We also add a matched-Hamming control. For each (protein, regime, m, seed) we build the permuted sequence alongside a uniform random substitution and a composition-matched substitution at the same set of positions, so all three arms mutate the same number of residues and differ only in which letters they install (Appendix C).

## 3.3 Models, data, and metrics

We test ESM-2 at five sizes from 8M to 3B parameters [14], MSA Transformer [23], and AMPLIFY 120M [8], against an inverse-covariance Potts baseline and a pseudolikelihood Potts fit. MSA Transformer reads an alignment at inference exactly as Potts does, and neither is retrained, removing the asymmetry that Potts is refit on the permuted alignment while ESM-2 receives no relabeled training data. AMPLIFY has a different vocabulary (27 tokens in a different residue order), a different feed-forward design (SwiGLU), and a different training corpus, at matched capacity. All experiments ran on a single NVIDIA A100-80GB GPU.

We report two metrics. Overlap is the fraction of a model’s top-L/2 long-range contacts on the original sequence that it also predicts on the relabeled one, so an invariant readout scores 1. Precision P@ $L / 2$ is measured against solved structures [19]. We fixed each experiment’s protocol before running it, including the outcome that would refute it, and we report the tests that did not meet their thresholds as such (§5.2, §5.3). Protein lists, structures, and per-cell counts are in Appendix C.

## 4 Results

## 4.1 RQ1: Are Protein Language Model Contact Maps Invariant to Relabeling?

No protein language model we test is invariant (Table 1). Averaged over nine proteins and three seeds, ESM-2 650M’s overlap under random relabeling falls to 0.015 at m = 20, while in-class relabeling keeps 0.655 at its realized $m = 1 5$ . The cross-class arm falls faster than the random arm at $m = 6$ and m = 10, which §5 explains.

MSA Transformer reads the same relabeled alignments that Potts is fit to. On those alignments Potts returns identical contacts in all 136 cells (Appendix C), while MSA Transformer falls to 0.147 under random relabeling and keeps 0.536 in-class. AMPLIFY reproduces both the collapse and the in-class–random split, and the in-class to random ratio widens with m. Unlike ESM-2 and MSA Transformer, which share a 33-token vocabulary and training corpus, AMPLIFY has its own 27-token vocabulary in a different residue order, its own training corpus, and a different feed-forward design (SwiGLU) at matched capacity. Under random relabeling at $m = 2 0$ it loses every contact in its own top- $L / 2$ on cytochrome c. The degradation is therefore not specific to one model family or tokenizer. Figure 5 shows the contact maps.

## 4.2 RQ2: Does Biochemical Class or Edit Distance Drive the Degradation?

Figure 2(a) holds edit distance fixed. On cytochrome c, in-class relabeling at nominal $m = 2 0$ changes 76 of 104 positions for all three seeds and keeps overlap 0.699. Uniform random and compositionmatched substitutions at the same 76 positions both give 0.013, so the relabeled sequence keeps 54× more of its own contact set than either control. At matched edit distance, the outcome depends on which letters are installed: substitutions within biochemical groups cost little, and arbitrary substitutions destroy the prediction.

<table><tr><td>Model</td><td>Regime</td><td> $m = 2$ </td><td> $m = 6$ </td><td> $m = 1 0$ </td><td> $m = 2 0$ </td></tr><tr><td rowspan="3">ESM-2 650M</td><td>random</td><td>0.702</td><td>0.354</td><td>0.100</td><td>0.015</td></tr><tr><td>in-class</td><td>0.879</td><td>0.809</td><td>0.684</td><td>0.655</td></tr><tr><td>cross-class</td><td>0.570</td><td>0.247</td><td>0.027</td><td> $0 . 0 2 6 ^ { \dagger }$ </td></tr><tr><td rowspan="2">MSA Transformer</td><td>random</td><td>0.775</td><td>0.471</td><td>0.244</td><td>0.147</td></tr><tr><td>in-class</td><td>0.935</td><td>0.795</td><td>0.640</td><td>0.536</td></tr><tr><td rowspan="2">AMPLIFY 120M</td><td>random</td><td>0.559</td><td>0.222</td><td>0.124</td><td>0.056</td></tr><tr><td>in-class</td><td>0.770</td><td>0.671</td><td>0.493</td><td>0.414</td></tr><tr><td>Potts (inverse covariance)</td><td>any</td><td></td><td></td><td>1.000</td><td></td></tr></table>

Table 1: Contact-set overlap under alphabet relabeling; 1 means the model predicts the same contacts. Columns give nominal m: at $m = 2 0$ , in-class moves 15 residue types and cross-class 14 to 16. ESM-2 650M: nine proteins, three seeds, $n = 2 7$ per cell $( \dag \colon n = 9 )$ . MSA Transformer: four Pfam families, three seeds $( n = 1 2 )$ . AMPLIFY 120M: three proteins, two seeds $( n = 6 )$ . Potts is bit-identical by Proposition 2.

![](images/6c88c3e20643eddf583cd91d4414188390b8390fb2dff11889ec1f242dbce9ed.jpg)

![](images/fb365b7399c6877c5f425b87dfee2ab1b28e248c3ed5b2e4597285311c784f7f.jpg)

![](images/874fa2331ddff99bdca5307789771958dad9ba7d93c76591ab1112ed61e4fcb4.jpg)  
Figure 2: Overlap tracks biochemical class at matched Hamming distance, and keeps falling after likelihood shift saturates. (a) On cytochrome c the three arms change the same 76 of 104 positions and differ only in which residues they install. In-class overlap is 0.699; both matched substitutions give 0.013. (b) Potts is the flat line at 1.0 by Proposition 2. ESM-2 650M separates by biochemical class, and the cross-class arm falls faster than the random arm. (c) In the random arm, $\Delta _ { \mathrm { L L } }$ saturates by $m { = } 6$ while overlap falls another 16×. Faint curves are 8M, 35M, and 150M; bold curves are 650M.

Figure 2(b) shows the same separation across the full range of m. It does not hold on every protein: on protein G B1 the in-class and random arms separate at $m = 6$ and $m = 1 0$ but not at $m = 2 0$ where both sit within roughly 3× chance (Figure 5, bottom row).

Figure 2(c) asks whether the model simply finds relabeled sequences unfamiliar. We measure unfamiliarity as the drop in the model’s log-likelihood, $\Delta _ { \mathrm { L L } } \stackrel { \cdot \mathrm { ~ \tiny ~  ~ } } { = } \mathrm { L L } ( x ) - \mathrm { L L } ( \pi ( x ) )$ ), in nats per residue. Over the 144 random and in-class cells, $\Delta _ { \mathrm { L L } }$ and overlap are correlated (Spearman −0.700; −0.551 with the cross-class arm included). Adding a regime indicator still raises $\scriptstyle { \dot { R } } ^ { 2 }$ from 0.491 to 0.621. In the random arm, $\Delta _ { \mathrm { L L } }$ saturates by $m = 6$ while overlap keeps falling by another order of magnitude, so unfamiliarity alone does not explain the collapse.

## 4.3 RQ3: Does Scale Close The Gap?

Scale does not close the gap. Across the five ESM-2 sizes, the random arm stays near zero, from 0.042 at 8M to 0.035 at 3B, while the in-class arm climbs, so the gap between them grows with scale (Figure 3a). The pattern is consistent with larger models becoming more sensitive to biochemical similarity, while arbitrary relabeling remains destructive at every size.

![](images/41bbc2f75cd9730bc19cf135bd804aa96c87b8d17587e9395beb9a05728def41.jpg)

(b) Variant effect  
![](images/e4094aa8b97b9c4cf99c7e4b4928df261ef699ec154a3f4caa1fbdde0bb69746.jpg)

(c) Ubiquitin  
![](images/e4e175c144f62d4c32dacbf4e6579ec42153d406d2b4ff58876f75df43de36ee.jpg)

(d) Protein G  
![](images/54af0c7deb184bdd88121d8ced2da94f81ad6c6c0db01bfbe66600fbca9337ef.jpg)  
Figure 3: The degradation does not shrink with scale. (a) Overlap at nominal m=20 against parameter count. (b) shows Spearman ρ against deep mutational scanning measurements on the 11 ProteinGym assays where ESM-2 has signal. A PSSM on the same assays is unchanged in every cell. (c) and (d) show True long-range P@L/2 divided by contact density. On both ubiquitin and Protein G, ESM-2 650M degrades significantly under $\pi ( x )$ , falling below Potts.

The degradation also costs accuracy. Figure 3(b) restricts both methods to the same candidate pairs so that their precisions are directly comparable. On ubiquitin, ESM-2 650M starts at 2.33× the Potts precision (0.737 against 0.316). Under relabeling it falls to 0.053, while Potts is unchanged. On protein G the same pattern is weaker. Unpermuted, ESM-2 reaches precision 0.763 on full-length ubiquitin, so the loss is from an accurate model.

The failure is not specific to the categorical Jacobian. It replicates on variant-effect prediction, a second readout with external ground truth, over the 11 of 15 ProteinGym single-mutant assays where ESM-2 has signal (Figure 3c; 22). Random relabeling at $m = 2 0$ lowers Spearman ρ by 0.52 on average, in-class by 0.19 and cross-class by 0.54, while a PSSM baseline scored on the relabeled alignment is exactly unchanged in every cell (Appendix C). It also replicates under ESM-2’s own supervised contact head on the same nine proteins and relabelings, where random relabeling drives overlap to 0.028 at m = 20 and in-class holds 0.793 (Appendix E).

## 5 Discussion

## 5.1 Alphabet anchoring vs. coevolutionary inference

Our findings show that contact prediction in pLMs is driven by residue-identity-conditioned pattern matching rather than by the direct extraction of column statistics. We call this alphabet anchoring: the readout holds when residues are swapped within biochemical classes and collapses when swaps cross class boundaries. The term follows work in natural language processing on models that adopt surface heuristics in place of systematic statistical computation [16].

We also rule out several competing accounts:

(i) Not pure column-statistic inference: A model performing algebraic inversion of covariation would remain invariant across all three regimes (as proven for Potts models in Propositions 1 and 2).

(ii) Not input difficulty (Hamming distance): When edit distance is held constant by the matched-Hamming control on cytochrome c, in-class permutations keep 54× more contacts than either matched substitution, while random substitutions collapse to noise.

(iii) Not an artifact of one architecture or readout: The dissociation replicates across singlesequence models (ESM-2), alignment-based architectures (MSA Transformer), alternative tokenizers and vocabularies (AMPLIFY), and a supervised contact head (Appendix E).

Alphabet anchoring also explains why cross-class relabeling degrades contacts faster than random relabeling at m = 6 and $m = 1 0$ (Table 1). A random derangement keeps some residues inside their biochemical group by accident (Figure 4), whereas a cross-class mapping moves every one of them out.

## 5.2 Out-of-distribution shift and scale

A natural objection is that relabeling pushes sequences out of the model’s training distribution, which would make any degradation unremarkable. We measure how unfamiliar the model finds a sequence with its log-likelihood shift $\Delta _ { \mathrm { L L } }$ , a zero-shot measure of sequence fitness [17]. Unfamiliarity and overlap are correlated (Spearman −0.70), but unfamiliarity alone does not explain the collapse. In the random arm, $\Delta _ { \mathrm { L I } }$ stops rising after m=6 while overlap keeps falling by another order of magnitude (Figure 2c). In the in-class arm, $\Delta _ { \mathrm { L L } }$ keeps rising while ESM-2 650M’s overlap stays above 0.65, so a sequence the model finds unfamiliar can still keep most of its contact map when biochemical classes are intact. After accounting for $\Delta _ { \mathrm { L L } }$ , regime still explains additional variance in overlap $( \Delta R ^ { 2 } = 0 . 1 3 1$ , partial Spearman +0.506). Throughout, the information a column-statistics estimator uses is unchanged, and Potts keeps its precision on the same inputs.

Scale does not remove the effect either. Increasing ESM-2’s size 375×, from 8M to 3B parameters, improves tolerance to in-class swaps and gives no improvement under random relabeling (Figure 3a). Language-model loss improves predictably with scale [13]; for contact readouts, scale refines biochemical pattern recognition without making the readout invariant.

## 5.3 Biochemical class as a categorical prior

A graded substitution cost does not reliably predict degradation beyond the discrete class partition. Across full m=20 derangements, the correlation between BLOSUM62 substitution cost and overlap is $\rho = - 0 . 4 3$ on ubiquitin and −0.11 on protein G. pLMs therefore appear to use categorical groupings rather than a continuous distance over residues.

How much biochemical class protects the map depends on the protein. In-class relabeling at nominal m=20 keeps 0.675 of the identity contacts on ubiquitin and 0.071 on protein G, where the in-class and random arms still separate at m = 6 and $m = 1 0$ . Random relabeling collapses the map on every protein we test; what varies is how much of it biochemical class can save.

## 6 Conclusion

Contact maps derived from protein language models are widely interpreted as evidence that these models have internalized coevolution. In this work, we tested that interpretation directly, rather than through accuracy, using a property of any estimator that reads only column statistics: its contacts are invariant under a global relabeling of the amino-acid alphabet. We proved this for pseudolikelihood Potts models and as an exact identity for the inverse-covariance estimator. Every pLM we evaluated changes its contact predictions under this label-free, structure-free and alignment-free test, across three architectures, two tokenizer schemes and a 375× range of scale, while the inverse-covariance estimator returns an identical score matrix on the same inputs. On ubiquitin, over a candidate set that allows direct comparison, ESM-2 goes from 2.33× the Potts precision to 0.17× under relabeling.

We call this phenomenon alphabet anchoring. While it does not undermine the utility of pLM contact maps, we show that it changes their mechanistic interpretation. A scaling curve of contact precision mixes coevolutionary inference with amino-acid pattern matching, so high precision cannot be cited as proof that a model computes contacts through coevolution. Where pLMs have replaced classical estimators, Figure 3(b) is a case for keeping classical coevolutionary methods in the loop.

This work is a first step toward finding such pitfalls in protein language models. As pLMs are deployed across biological applications, the scientists and practitioners who rely on them should know that their contact maps depend on canonical residue identity. We hope this test inspires further diagnostics that check what a model computes in addition to how accurately it predicts.

## Limitations

Scope: Our experiments are carried on a relatively small number of proteins, four Pfam families, three readouts (categorical-Jacobian contacts, ESM-2’s supervised contact head, and ProteinGym variant effects), and three of the many available pLM families. The proteins are short and singledomain, the setting where classical coevolutionary estimators are often strongest and where the coevolution interpretation of pLMs is most often invoked. We did not test multi-domain proteins or shallow families. The in-class regime uses one grouping, derived from BLOSUM62; groupings from other substitution matrices, such as BLOSUM45 [10] or PAM [3], were not tested.

The test is a necessary, but not sufficient evidence: Invariance under alphabet relabeling is a necessary condition for a contact readout to be a pure function of column statistics, but it is not sufficient to show that a model has learned coevolutionary inference. Our interventions isolate alphabet anchoring; they do not identify its cause. We also did not test retrained or fine-tuned models. A pLM trained on relabeled corpora might pass the test, but our claim concerns the public pre-trained checkpoints, which are the ones used in practice.

## Reproducibility, Code, and Data Availability

We release all of the code and data used in this paper at https://github.com/aryankeluskar/ icbinb-alphabet-anchoring. No newly collected data was used in this research.

## Disclosure of LLM Use

In accordance with the NeurIPS policy on LLM usage, we disclose that Claude Code was used to assist with initial drafting and refining sentences in this paper. All content underwent multiple rounds of extensive revision, editing and verification by the authors. None of the citations or references were created by an LLM, instead all such content was manually checked and entered by the authors.

## References

[1] Helen M Berman, Tammy Battistuz, Talapady N Bhat, Wolfgang F Bluhm, Philip E Bourne, Kyle Burkhardt, Zukang Feng, Gary L Gilliland, Lisa Iype, Shri Jain, et al. The protein data bank. Biological Crystallography, 58(6):899–907, 2002.

[2] Thomas M. Cover and Joy A. Thomas. Elements ofInformation Theory. Wiley-Interscience, Hoboken, NJ, 2nd edition, 2006. doi: 10.1002/047174882X.

[3] Margaret O. Dayhoff, Richard M. Schwartz, and Bruce C. Orcutt. A model of evolutionary change in proteins. In Atlas of Protein Sequence and Structure, volume 5, pages 345–352. National Biomedical Research Foundation, 1978.

[4] Henry Dieckhaus and Brian Kuhlman. Protein stability models fail to capture epistatic interactions of double point mutations. Protein Science, 34(1):e70003, 2025.

[5] Frances Ding and Jacob Steinhardt. Protein language models are biased by unequal sequence sampling across the tree of life. BioRxiv, pages 2024–03, 2024.

[6] Shannon D. Dunn, Lindi M. Wahl, and Gregory B. Gloor. Mutual information without the noise: adjusting for background correlation identifies functional interactions in proteins. Bioinformatics, 24(3):333–340, 2008. doi: 10.1093/bioinformatics/btm604.

[7] Magnus Ekeberg, Cecilia Lövkvist, Yue Lan, Martin Weigt, and Erik Aurell. Improved contact prediction in proteins: using pseudolikelihoods to infer Potts models. Physical Review E, 87(1): 012707, 2013.

[8] Quentin Fournier and Robert M. Vernon. Protein language models: is scaling necessary? bioRxiv, 2024. doi: 10.1101/2024.09.23.614603.

[9] Robert Geirhos, Jörn-Henrik Jacobsen, Claudio Michaelis, Richard Zemel, Wieland Brendel, Matthias Bethge, and Felix A. Wichmann. Shortcut learning in deep neural networks. Nature Machine Intelligence, 2(11):665–673, 2020.

[10] Steven Henikoff and Jorja G. Henikoff. Amino acid substitution matrices from protein blocks. Proceedings ofthe National Academy ofSciences, 89(22):10915–10919, 1992.

[11] Cosmin Ieremie, Aled Ewing, and Mahesan Niranjan. Protein language models meet reduced amino acid alphabets. Bioinformatics, 40(2):btae061, 2024.

[12] David T. Jones, Daniel W. Buchan, Domenico Cozzetto, and Massimiliano Pontil. PSICOV: precise structural contact prediction using sparse inverse covariance estimation. Bioinformatics, 28(2):184–190, 2012.

[13] Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361, 2020.

[14] Zeming Lin, Halil Akin, Roshan Rao, Brian Hie, Zhongkai Zhu, Wenting Lu, Nikita Smetanin, Robert Verkuil, Ori Kabeli, Yaniv Shmueli, et al. Evolutionary-scale prediction of atomic-level protein structure with a language model. Science, 379(6637):1123–1130, 2023.

[15] Debora S. Marks, Lucy J. Colwell, Robert Sheridan, Thomas A. Hopf, Andrea Pagnani, Riccardo Zecchina, and Chris Sander. Protein 3d structure computed from evolutionary sequence variation. PLoS ONE, 6(12):e28766, 2011. doi: 10.1371/journal.pone.0028766.

[16] R Thomas McCoy, Ellie Pavlick, and Tal Linzen. Right for the wrong reasons: Diagnosing syntactic heuristics in natural language inference. In Proceedings ofthe 57th annual meeting of the association for computational linguistics, pages 3428–3448, 2019.

[17] Joshua Meier, Roshan Rao, Robert Verkuil, Jason Liu, Tom Sercu, and Alex Rives. Language models enable zero-shot prediction of the effects of mutations on protein function. Advances in neural information processing systems, 34:29287–29303, 2021.

[18] Jaina Mistry, Sara Chuguransky, Lowri Williams, Matloob Qureshi, Gustavo A Salazar, Erik LL Sonnhammer, Silvio CE Tosatto, Lisanna Paladin, Shriya Raj, Lorna J Richardson, et al. Pfam: The protein families database in 2021. Nucleic acids research, 49(D1):D412–D419, 2021.

[19] Bohdan Monastyrskyy, Daniel d’Andrea, Krzysztof Fidelis, Anna Tramontano, and Andriy Kryshtafovych. Evaluation of residue–residue contact prediction in casp10. Proteins: Structure, Function, and Bioinformatics, 82:138–153, 2014.

[20] Faruck Morcos, Andrea Pagnani, Bryan Lunt, Annabella Bertolino, Debora S. Marks, Chris Sander, Riccardo Zecchina, José N. Onuchic, Terence Hwa, and Martin Weigt. Direct-coupling analysis of residue coevolution captures native contacts across many protein families. Proceedings ofthe National Academy ofSciences, 108(49):E1293–E1301, 2011.

[21] Lynne Reed Murphy, Anders Wallqvist, and Ronald M Levy. Simplified amino acid alphabets for protein fold recognition and implications for folding. Protein engineering, 13(3):149–152, 2000.

[22] Pascal Notin, Debora S. Marks, et al. ProteinGym: large-scale benchmarks for protein fitness prediction and design. In Advances in Neural Information Processing Systems (NeurIPS), volume 36, 2023.

[23] Roshan M Rao, Jason Liu, Robert Verkuil, Joshua Meier, John Canny, Pieter Abbeel, Tom Sercu, and Alexander Rives. Msa transformer. In International conference on machine learning, pages 8844–8856. PMLR, 2021.

[24] Alexander Rives, Joshua Meier, Tom Sercu, Siddharth Goyal, Zeming Lin, Jason Liu, Demi Guo, Myle Ott, C. Lawrence Zitnick, Jerry Ma, et al. Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences. Proceedings ofthe National Academy of Sciences, 118(15):e2016239118, 2021.

[25] Stefan Seemayer, Michael Gruber, and Johannes Söding. CCMpred—fast and precise prediction of protein residue–residue contacts. Bioinformatics, 30(21):3128–3130, 2014.

[26] Rome Thorstenson. Protein contacts are already in the attention: a single-forward-pass alternative to the categorical jacobian. arXiv preprint arXiv:2606.21876, 2026.

[27] Jesse Vig, Sebastian Gehrmann, Yonatan Belinkov, Sharon Qian, Daniel Nevo, Yaron Singer, and Stuart Shieber. Causal mediation analysis for interpreting neural language models. In Advances in Neural Information Processing Systems (NeurIPS), volume 33, pages 9388–9399, 2020.

[28] Jesse Vig, Ali Madani, Lav R Varshney, Caiming Xiong, Richard Socher, and Nazneen Fatema Rajani. Bertology meets biology: Interpreting attention in protein language models. arXiv preprint arXiv:2006.15222, 2020.

[29] Martin Weigt, Robert A. White, Hendrik Szurmant, James A. Hoch, and Terence Hwa. Identification of direct residue contacts in protein–protein interaction by message passing. Proceedings of the National Academy of Sciences, 106(1):67–72, 2009. doi: 10.1073/pnas.0805923106.

[30] Kevin K Yang, Nicolo Fusi, and Alex X Lu. Convolutions are competitive with transformers for protein sequence pretraining. Cell systems, 15(3):286–294, 2024.

[31] Zhidian Zhang, Hannah K Wayment-Steele, Garyk Brixi, Haobo Wang, Dorothee Kern, and Sergey Ovchinnikov. Protein language models learn evolutionary statistics of interacting sequence motifs. Proceedings of the National Academy of Sciences, 121(45):e2406285121, 2024.

[32] Chengshuai Zhao, Zhen Tan, Pingchuan Ma, Dawei Li, Bohan Jiang, Yancheng Wang, Yingzhen Yang, and Huan Liu. Is chain-of-thought reasoning of llms a mirage? a data distribution lens. In Findings ofthe Associationfor Computational Linguistics: ACL 2026, pages 15231–15261, 2026.

## A Invariance proofs

Shared contact pipeline. Potts models and pLMs both yield a coupling tensor of shape $L \times q \times L \times q \colon$ Potts from its fitted couplings $J _ { i j }$ or a regularized precision matrix, and a pLM from its categorical Jacobian $J [ i , a , j , b ]$ , the change in the logit for residue b at position j when position i is set to residue a [31]. The same pipeline turns either into an $L \times L$ contact score S: mean-centering over both amino-acid axes, symmetrization, the Frobenius norm of each $q \times q$ block, and average-product correction (APC) [6]. For Potts, every input to this pipeline is a function of the alignment’s column statistics, which is what makes the symmetry below available.

Notation. An alignment is $\boldsymbol { X } \in \bar { \mathcal { A } } ^ { N \times L }$ over ${ \bar { \mathcal { A } } } = { \mathcal { A } } \cup \{ - \}$ , where A is the 20 standard amino acids and − is the gap $( q = 2 1$ states; non-standard characters are mapped to the gap). A relabeling is a bijection $\pi : { \mathcal { A } }  A $ , extended to A<sup>¯</sup> by $\pi ( - ) = -$ and applied entrywise, $( { \bar { \pi } } { \bar { X } } ) _ { n i } = \pi ( X _ { n i } ) $ Let $P _ { \pi } \in \{ 0 , 1 \} ^ { q \times q }$ with $( P _ { \pi } ) _ { a , \pi ( a ) } = 1$ , so that $e _ { a } ^ { \top } P _ { \pi } = e _ { \pi ( a ) } ^ { \top }$ and $P _ { \pi } ^ { \top } P _ { \pi } = I$

Proposition 1 (pseudolikelihood Potts). Let $( h , J )$ maximize the regularized pseudolikelihood of X under $\begin{array} { r } { H ( \sigma ) \dot { = } - \sum _ { i } h _ { i } ( \sigma _ { i } ) - \sum _ { i < j } J _ { i j } ( \sigma _ { i } , \sigma _ { j } ) } \end{array}$ with penalty $\begin{array} { r } { \lambda _ { J } \sum _ { i < j } \| J _ { i j } ^ { ' } \| _ { F } ^ { 2 } + \lambda _ { h } \sum _ { i } \| h _ { i } \| _ { 2 } ^ { 2 } , } \end{array}$ $\lambda _ { h } \geq 0$ . Define $h _ { i } ^ { \pi } = P _ { \pi } ^ { \top } h _ { i }$ and $J _ { i j } ^ { \pi } = P _ { \pi } ^ { \top } J _ { i j } P _ { \pi }$ , that is, $h _ { i } ^ { \pi } ( \pi a ) = h _ { i } ( a )$ and $J _ { i j } ^ { \pi } ( \pi a , \pi b ) =$ $J _ { i j } ( a , b )$ . Then $( h ^ { \pi } , J ^ { \pi } )$ maximizes the regularized pseudolikelihood ofπX, the attained objective is identical, and the contact scores are unchanged.

Proof. By construction $\begin{array} { r } { H ^ { \pi } ( \pi \sigma ) = H ( \sigma ) } \end{array}$ for every $\sigma \in \bar { A } ^ { L }$ . Each conditional $p ( \sigma _ { i } \mid \sigma _ { \backslash i } )$ normalizes over $\sigma _ { i } \in { \bar { A } } .$ which π permutes, so $p ^ { \pi } ( \pi \sigma _ { i } \mid \pi \sigma _ { \setminus i } ) = p ( \sigma _ { i } \mid \sigma _ { \setminus i } )$ , and the pseudolikelihood of $\pi X$ under $( h ^ { \pi } , J ^ { \pi } )$ equals that of X under $( h , J )$ term by term. The penalty is unchanged because permutation matrices preserve Euclidean and Frobenius norms. The map $( h , J ) \mapsto \left( h ^ { \pi } , J ^ { \pi } \right)$ is a bijection on parameter space (its inverse is the same map for $\pi ^ { - 1 } )$ that preserves the objective, so it carries maximizers to maximizers. Finally, the zero-sum gauge and the scoring pipeline commute with $J _ { i j } \mapsto P _ { \pi } ^ { \intercal } J _ { i j } P _ { \pi }$ by the argument in the proof of Proposition 2, so the contact scores are identical. □

This covers plmDCA, GREMLIN and CCMpred [7, 25], provided the penalty is the same for every amino acid.

Proposition 2 (inverse-covariance Potts). Let $W = ( C + \lambda I ) ^ { - 1 }$ be the regularized precision matrix of the one-hot encoded alignment, and let S be the contact score after mean-centering, symmetrization, per-block Frobenius norm, and APC. Then $S ^ { \pi } = S$ exactly.

Proof. One-hot encode the alignment as $Y \in \mathbb { R } ^ { N \times q L }$ , blocked by position. Relabeling permutes the q coordinates of every block identically: $Y ^ { \pi } = Y Q$ with $Q = I _ { L } \otimes P _ { \pi }$ , which is orthogonal. The empirical covariance transforms by congruence, $C ^ { \pi } = Q ^ { \top } C Q$ , so

$$
\boldsymbol { W } ^ { \pi } = ( \boldsymbol { Q } ^ { \top } \boldsymbol { C } \boldsymbol { Q } + \lambda \boldsymbol { I } ) ^ { - 1 } = \left( \boldsymbol { Q } ^ { \top } ( \boldsymbol { C } + \lambda \boldsymbol { I } ) \boldsymbol { Q } \right) ^ { - 1 } = \boldsymbol { Q } ^ { \top } ( \boldsymbol { C } + \lambda \boldsymbol { I } ) ^ { - 1 } \boldsymbol { Q } = \boldsymbol { Q } ^ { \top } \boldsymbol { W } \boldsymbol { Q } ,
$$

using $Q ^ { \top } Q = I$ . Blockwise, $\begin{array} { r } { W _ { i j } ^ { \pi } = P _ { \pi } ^ { \top } W _ { i j } P _ { \pi } } \end{array}$ . Each scoring step commutes with this conjugation: mean-centering (row and column means are permuted, not changed), symmetrization $( ( Q ^ { \top } W Q ) ^ { \top } =$ $Q ^ { \top } W ^ { \top } Q )$ , the per-block Frobenius norm $( \| P _ { \pi } ^ { \top } W _ { i j } P _ { \pi } \| _ { F } = \| W _ { i j } \| _ { F } )$ , and APC (a function of the already unchanged $L \times L$ matrix). Hence $S _ { i j } ^ { \pi } = S _ { i j }$ . □

Both conditions are necessary. The proof needs an isotropic ridge $( Q ^ { \top } \Lambda Q \ne \Lambda$ for a nonuniform Λ) and a gap that π leaves fixed. We tested each in both directions on an alignment with $N { = } 1 5 0 0 , L { = } 4 0$ and random $m { = } 2 0$ . A per-amino-acid ridge spanning only $1 . 0 { - } 1 . 5 \times \lambda$ gives max $\Delta S \vert = 0 . 3 5 5 ( 1 1 . 8 \%$ of the score scale; $\mathrm { { t o p } - \it { L } / 2 }$ overlap 0.850), against $4 . 9 \times 1 0 ^ { - 1 5 }$ with an isotropic ridge. Letting the gap be relabeled $( \mathrm { g a p } \to \mathbb { K }$ , with 10% gaps and 1.5% X) gives max $| \Delta S | = 0 . 1 8 2$ , against $1 . 8 \times 1 0 ^ { - 1 5 }$ when the gap is fixed.

## B Pseudolikelihood invariance

The classical models a reader has in mind, plmDCA, GREMLIN and CCMpred, are pseudolikelihood fits with a real optimiser rather than the closed-form estimator of Proposition 2. We fitted one (asymmetric pseudolikelihood, per-column multinomial logistic regression, isotropic $L _ { 2 } , \mathrm { L } { \mathrm { - B F G S } }$ zero-sum gauge, symmetrised, Frobenius per block, APC) on the same MSAs and targets.

On ubiquitin $( L { = } 6 7$ , depth 3000, λ=0.001), the correctness gate (plm $\mathrm { P } @ L / 2 \geq$ mean-field P@ $L / 2 )$ passes at 0.515 against 0.333. Under relabeling, a zero (equivariant) initialisation returns an identical top- $L / 2$ contact set with identical precision, and a random initialisation returns 0.970 overlap. Protein G replicates: zero init gives 1.000 overlap, random init 1.000.

The raw score matrix diverges at $1 . 9 \times 1 0 ^ { - 3 }$ relative, which is not machine precision. A maxiter ladder shows why: the two fits agree to $1 . 1 \times 1 0 ^ { - 1 5 }$ at two iterations, $1 . 4 \times 1 0 ^ { - 1 4 }$ at ten, $1 . 2 \times 1 0 ^ { - 9 }$ at forty, and $1 . 9 \dot { \times } 1 0 ^ { - 3 }$ at 150, twelve orders of magnitude of monotone growth, with the contact set exactly identical at every rung. The relabeled design matrix accumulates inner products in a different order, so gradients differ $\mathrm { a t } \sim 1 0 ^ { - 1 6 }$ from the first step, and L-BFGS is not contractive. The theorem is about the computation, and a numerical solver inherits it only up to its own conditioning. Scorelevel invariance to machine precision belongs to the closed-form estimator alone; prediction-level invariance belongs to both.

## C Experimental setup

Proteins. Table 2 lists the nine proteins. True-precision targets (Figure 3b) are ubiquitin (1UBQ\_A, 111 contacts at $| i - j | \geq 1 2 )$ and protein G B1 (1PGB\_A, 79 contacts), the only two whose sequences are identical to their deposited chains; for example, our cytochrome c is human while 1HRC is horse heart (88.5% identity).

<table><tr><td>Protein</td><td>L</td><td>Structure</td><td>Also used in</td></tr><tr><td>PF00018 (BOI2_YEAST)</td><td>51</td><td></td><td>S, M</td></tr><tr><td>protein G B1</td><td>56</td><td>1PGB</td><td>S, P, H, A</td></tr><tr><td>PF00013 (B3RHY5_TRIAD)</td><td>67</td><td></td><td></td></tr><tr><td>PF00076 (LA_HUMAN)</td><td>69</td><td></td><td>M</td></tr><tr><td>ubiquitin</td><td>76</td><td>1UBQ</td><td>S, P, H, A</td></tr><tr><td>PF00027 (P73234_SYNY3)</td><td>89</td><td></td><td>M</td></tr><tr><td>cytochrome c</td><td>104</td><td>1HRC</td><td>S, H, A</td></tr><tr><td>PF00072 (AGMR_PSEAE)</td><td>114</td><td></td><td>M</td></tr><tr><td>T4 lysozyme</td><td>164</td><td>2LZM</td><td>一</td></tr></table>

Table 2: Proteins. All nine are used for the ESM-2 650M sweep (Table 1) and the contact-head replication (Appendix E). Also used in: S, scale sweep; P, precision against structure; H, matched-Hamming control; A, AMPLIFY; M, MSA Transformer (full Pfam family alignment). Pfam alignments are from Mistry et al. [18], structures from Berman et al. [1].

Relabeling regimes. Permutations are generated by src/permute.py with NumPy seed 1000 · seed + m. $\mathrm { A t } \bar { m } \in \{ 2 , 6 , 1 0 \}$ every regime moves exactly m residue types. At nominal m=20, in-class moves 15 $( \mathrm { A } , \mathrm { C } , \mathrm { G }$ , H and P have no BLOSUM62 partner and stay fixed) and cross-class moves 14 to 16. Figure 4 shows the seed-0 maps. Tables report nominal m; figures plot realized m.

Matched-Hamming control. For each (protein, regime, m, seed) we build three sequences that differ from wild type at the same set of positions: the relabeled sequence, a uniform substitution (replacement drawn from the other 19 residues), and a composition-matched substitution (replacement drawn from the wild type’s own residue composition). The arms differ only in which residues they install; identical positions were asserted in every run. On cytochrome c at in-class $m { = } 2 0$ (Figure 2a; 76 of 104 positions changed, k=52), the relabeled sequence keeps 37, 37 and 35 of its 52 contacts across seeds, the uniform arm 0, 1 and 1, and the composition-matched arm 1, 0 and 1.

Models and readouts. ESM-2 contacts use the categorical Jacobian with BOS and EOS replaced by the mask token, reduced as in Appendix A. The scale sweep uses ESM-2 8M, 35M, 150M, 650M and 3B on four proteins with two seeds (n=8 per cell). MSA Transformer uses four Pfam families with three seeds (n=12 per cell; n=4 for cross-class at m=20); inverse-covariance Potts on the same relabeled alignments returns overlap 1.000 in all 136 runs. AMPLIFY 120M (2024 checkpoint)

![](images/8168b8718297777a490b82cd787fccbed290f5390d7d29bde73d448d0dac165d.jpg)  
same biochemical class: 2 of 20 moved

![](images/9dff0478fd336086e1e88a12c36c9025c4e720d19e1d07d8cf76b90e08929e8c.jpg)  
same biochemical class: 15 of 15 moved

![](images/0a763d67215b6682689685e28ef4856bf0d5a55a26feef63a899b8fc43693d3c.jpg)  
same biochemical class: 0 of 16 moved  
shaded diagonal blocks: the six BLOSUM62 In-class groups (ILMV, FWY, NQ, ST, KR, DE); thin rules: biochemical-class boundaries

Figure 4: Each panel is the seed-0, $m { = } 2 0$ permutation matrix, with a marker at $( a , \pi ( a ) )$ . Axes are ordered by biochemical class, so the six BLOSUM62 in-class groups form shaded blocks on the diagonal. Random preserves class only by accident (2 of 20 residues here, 2–4 across seeds). In-class stays inside the blocks; the five diagonal markers are $\mathrm { A } , \mathrm { C } , \mathrm { G } ,$ H and P, which have no BLOSUM62 partner. In cross-class, none of the 16 moves stays within a BLOSUM62 group. This is why the cross-class arm can fall faster than the random arm at the same nominal m.

runs in fp32 through the same post-processing and passed a validity gate before any relabeled run (unrelabeled $\mathrm { P } @ L / 2 \geq 3 \times$ chance; 4.4× and 4.0× on the two precision targets). Inverse-covariance Potts uses an isotropic ridge $\lambda = 4 . 5 / \sqrt { N }$ ; for the precision comparison, alignment rows with more than 50% gaps in the covered region are removed.

Log-likelihood shift. $\begin{array} { r } { \mathrm { L L } ( s ) = \frac { 1 } { L } \sum _ { i } } \end{array}$ log softmax $( \log \mathrm { i t s } ( s ) _ { i } ) [ s _ { i } ] .$ , with logits restricted to the 20 standard residues and BOS/EOS masked, and $\Delta _ { \mathrm { L L } } = \mathrm { L L } ( x ) - \bar { \mathrm { L L } } \bar { ( } \pi ( x ) )$ ). Its scale differs between models, so Figure 2(c) overlays model sizes to compare the shape of each curve (where it saturates), not its level.

Metrics. Two residues are in contact when their $C \beta$ atoms (Cα for glycine) are within $8 \mathring \mathrm { A }$ . Candidate pairs have $| i - j | \geq 1 2$ , below the CASP long-range cutoff of 24, because the proteins are short $( \bar { L } = 5 1$ to 164); results at $| i - j | \ge 2 4$ are reported in Appendix E. Overlap is $| A \cap B | / k$ where A and B are a model’s top-k candidate pairs before and after relabeling and $\bar { k = \lfloor L / 2 \rfloor }$ . This is the overlap coefficient; it is not the Jaccard index $| A \cap B | / | A \cup B |$ , which is lower whenever $0 < | A \cap B | < k .$ . Chance overlap is k divided by the number of candidate pairs.

Variant effect. We use 15 ProteinGym single-mutant assays [22] with three seeds per cell. ESM-2 scores are masked marginals [17]: a mutation $a _ { i } \to b$ is scored as log $: p ( \pi ( b ) ) - \log p ( \bar { \pi } ( a ) )$ at masked position i of the relabeled wild type, and the measured fitness values are unchanged. Unrelabeled Spearman $\rho$ reproduces the published ESM-2 650M values to four decimals on all 15 assays. A PSSM built from the relabeled alignment and scored on the relabeled mutations is unchanged to floating-point zero in every cell. Eleven assays have unrelabeled $\rho \geq 0 . 2 0 ;$ ; on these, random $m { = } 2 0$ lowers $\rho$ by 0.52 on average, in-class by 0.19, and cross-class by 0.54. The other four reproduce the published near-zero baseline.

## D Example sequences.

The seed-0, m=20 maps applied to ubiquitin (L=76), with Hamming distance from wild type:

wild type MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG   
random d=76 EWNARMPTPFMPNPTGRGQCYPNGSRMLMNWYMGFNQQYWWITNALFMWTGYFIPTCYVSNWMGCPTDTRTITIFF   
in-class d=64 VNLWIRSMSGRSLSMDIDPTESLDQIRARLNERDGLPPENNKMLWAGRNMDEGKSMTEFQLNRDTSMHMIMKMKGG

The in-class sequence occurs nowhere in nature and differs from wild type at 64 of 76 positions, yet ESM-2 keeps most of its contact map on it. Over all 2080 candidate pairs of full-length ubiquitin (k=38, chance overlap 0.018), random relabeling leaves ESM-2 650M with 1, 0 and 0 of its 38

contacts across seeds, and the rank correlation between the unrelabeled and relabeled score matrice is only 0.21, 0.20 and 0.13, so the collapse is not limited to the top of the ranking.

## E Additional results

![](images/cb3adaa62b373a96daa38e3db180602d83c2641b77c703717edaa6da954aa6a9.jpg)  
true long-range contact, lower triangle model's own top-L/2 prediction, upper triangle (coloured as its panel title)

Figure 5: Contact maps for both precision targets on the restricted candidate set of Figure 3(b). Lower triangle: contacts of the deposited chain. Upper triangle: the model’s top- $. L / 2$ prediction. Overlaps are against the same model’s unrelabeled prediction. Top row, ubiquitin: random relabeling destroys the map (0.05, 2.1× chance) while in-class relabeling preserves most of it (0.76, 31× chance). Bottom row, protein G B1: the in-class arm also collapses (0.07, 2.5× chance), the counter-case discussed in §4.2 and §5.3. Potts returns its unrelabeled map exactly in both rows (overlap 1.000, max $| \Delta S | \le 2 . 3 \times 1 0 ^ { - 1 4 } )$

Contact maps for both precision targets. Figure 5 shows the maps behind Figure 3(b), including protein G B1, where in-class relabeling does not protect the map.

![](images/19c350b27fc7e46ee3e88eb8060b6354b223e9bbe0183860f4818f64118ef93d.jpg)

![](images/8e282498a45392244b1c9a119f19488aa4e7d3bcbc5d0f46e078edd31edf46b1.jpg)  
Figure 6: The failure is not specific to the categorical Jacobian or to one assay. (a) ESM-2 650M’s supervised contact head; nine proteins, three seeds, mean ± SEM, plotted at realized m. (b) Varianteffect Spearman $\rho$ before and after relabeling at m=20, one marker per ProteinGym assay per regime; the dashed line is no change.

ESM-2’s supervised contact head. The main results read contacts from the categorical Jacobian, so one objection is that the Jacobian, not the model, is what fails. We repeated the sweep with ESM-2 650M’s own supervised contact head on the same nine proteins, regimes and seeds (Figure 6a). It reproduces the dissociation: random relabeling falls from 0.842 at m=2 to 0.028 at m=20 (1.5× chance), cross-class to 0.009, while in-class holds 0.793. $\mathrm { A t } | i - j | \geq 2 4$ the result is unchanged (0.025 random, 0.774 in-class at m=20). A head trained with structural supervision therefore inherits the same alphabet anchoring.