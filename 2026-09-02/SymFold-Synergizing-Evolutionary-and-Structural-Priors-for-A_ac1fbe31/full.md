# SymFold: Synergizing Evolutionary and Structural Priors for Accurate Protein Inverse Folding

Handong Wang<sup>1,2</sup>, Jiaxin Qi<sup>1</sup>, Baisheng Lai<sup>1,2</sup>, Jianqiang Huang<sup>1,2</sup>

<sup>1</sup>Computer Network Information Center, Chinese Academy of Sciences

<sup>2</sup>University of Chinese Academy of Sciences

wanghandong24@mails.ucas.ac.cn, jxqi@cnic.cn, bslai@cnic.cn, jqhuang@cnic.cn

## Abstract

Protein inverse folding aims to recover amino acid sequences for a given 3D protein structure, underpinning broad applications such as enzyme engineering and drug discovery.Current methods often follow a serial pipeline, in which a structure encoder predicts a coarse sequence, which is then refined by protein language models (PLMs). However, because PLMs only perform post-hoc sequence edits, the refinement is bounded by the quality of upstream predictions.Thanks to recent multimodal protein language models (MPLMs), we could directly encode structure to generate sequences with pretrained structural knowledge, but we observe that they are not effective for inverse folding. Therefore, we introduce a symmetric dualpath architecture that both leverages PLMs for pretrained sequence evolution knowledge and MPLMs for pretrained structural knowledge to iteratively guide protein sequence generation.Through extensive experiments across standard protein inverse folding benchmarks, our method achieves state-ofthe-art performance, surpassing prior approaches, and ablation studies validate the rationale of our symmetric design, revealing a promising direction for the community.

## 1 Introduction

Protein sequence and structure are two sides of the same coin. The protein sequence specifies the order of amino acids (residues), and once known, the corresponding protein can be chemically synthesized Defresne et al. [2021]; Khakzad et al. [2023]; Huang et al. [2016]. Conversely, protein sequences naturally fold into higher-order three-dimensional (3D) structures, and distinct structures typically confer distinct functions Gao et al. [2020]. Mapping between sequence and structure in both directions is therefore a central theme in protein science Kuhlman and Baker [2000]. The forward direction is the well-known protein folding task, which predicts structure (and hence function) from sequence Jumper et al. [2021]; Baek et al. [2021]; Abramson et al. [2024]. In this work, we focus on the opposite direction, protein inverse folding, which aims to design an amino acid sequence that will stably fold into a specified 3D structure Dauparas et al. [2022]; Gao et al. [2022a]; Yi et al. [2023]; Ren et al. [2024], enabling applications such as enzyme engineering and drug discovery. For example, to add a new binding site on the surface of an existing enzyme, researchers first design the local 3D structure required for binding, then use inverse folding methods to derive candidate sequences for synthesis and experimental validation.

Traditional protein inverse folding methods directly encode the 3D structure and perform position-wise amino acid (residue) classification (one-to-one mapping), as shown in Figure 1(a). Constrained by dataset scale and model capacity, their performance is limited. Besides, since the structure encoder ignores sequence context, the resulting sequences are usually biologically unreasonable. With the advent of large pretrained protein sequence models, i.e., protein language models (PLMs) Rives et al. [2021]; Lin et al. [2023]; Madani et al. [2023]; Elnaggar et al. [2021]; Brandes et al. [2022], recent methods Gao et al. [2023]; Zhu et al. [2024] often append a post-hoc refinement stage that uses PLMs to revise the output of the structure encoder. For example, as shown in Figure 1(b), if the encoder proposes “. . . LISE. . . ” but assigns low confidence to “I”, the PLM could leverage pretrained sequence knowledge to decide whether “I” should be replaced by another residue, conditioning on the context of “I”, thereby improving sequence plausibility. However, such refinement is decoupled from the original structural evidence and is inherently upper-bounded by the first-stage output: when the proposed sequence already violates structural constraints, PLM edits may further degrade structural compatibility, compounding errors for inverse folding.

To bridge the gap between pretrained knowledge and given structures during refinement, we find that recent large pretrained multimodal protein models, i.e., multimodal protein language models (MPLMs) Hayes et al. [2025]; Wang et al. [2024]; Hsieh and others [2025], can directly map 3D structures to sequences and are naturally suited to the inverse folding task, even in a zero-shot manner. However, in practice, we find that MPLMs used in isolation are ineffective for this task, perhaps due to structural distribution shifts between pretraining and this task or their limited sequence modeling capacity. Therefore, we introduce SymFold, a symmetric dualpath framework that harmonizes PLMs and MPLMs in parallel to provide complementary evolutionary and structural priors for sequence generation.Specifically, as shown in Figure 1(c), our method first follows the traditional approach to derive a coarse sequence via a structure encoder, then we employ PLMs to provide pretrained sequence context knowledge, and meanwhile, leverage MPLMs to provide pretrained structural knowledge to refine the proposed sequence. Unlike the traditional uses of PLMs, our symmetric dual-path refinement simultaneously accounts for contextual dependencies and the residue-specific positional information, thereby addressing the refinement gap.

![](images/3ac7674c9c5ce378b665fb0a36d8f5d39a87463603c5c5b388e0526204ab42f4.jpg)  
Figure 1: Comparison of various inverse folding frameworks: (a) The baseline model is trained from scratch without utilizing any pretrained knowledge to directly predict amino acid sequence from the protein structure. (b) PLM-based methods refine the output sequence of the structure encoder with pretrained sequence knowledge. (c) Our proposed SymFold symmetrically utilizes the pretrained knowledge of sequence and structure to refine the coarse sequence.

Furthermore, because the dependence on structure and sequence context varies across residues, we design an Adaptive Synergistic Fusion (ASF) module that dynamically integrates structure cues, context dependencies, and structural priors so that the refinement can maintain compatibility with residuespecific biochemical characteristics. We also introduce a selfcorrection iterative training strategy to align training with the inference routine. A structure encoder first proposes a coarse sequence, PLMs and MPLMs perform an initial refinement, and then stochastically mix the outputs with the coarse sequence before the final refinement stage. This iterative procedure mirrors the test-time iterative prediction manner and reduces the mismatch between training and testing.

On standard inverse folding benchmarks, including the widely used CATH Orengo et al. [1997], TS50, and TS500 Li et al. [2014], our method sets new state-of-the-art performance. Extensive ablations corroborate our findings and validate the proposed design, revealing the limitations of standalone MPLMs, the benefit of our symmetric dual-path architecture, and the effectiveness of our adaptive integration and self-correction modules. Our main contributions are summarized as follows:

• We diagnose the shortcomings of post-hoc refinement in current methods and argue that effective refinement must incorporate raw 3D evidence.

• We introduce MPLMs into the inverse folding and empirically show that MPLMs alone are insufficient for this task.

• We propose SymFold, which combines pretrained sequence and structural knowledge via Adaptive Synergistic Fusion and self-correction for sequence generation, achieving SOTA performance across multiple benchmarks and informing future directions.

## 2 Related Work

Protein Inverse Folding. Early approaches in protein inverse folding primarily relied on physics-based methods such as Rosetta Design Liu and Kuhlman [2006], which searched for low-energy sequences compatible with target backbones via sampling strategies like Monte Carlo simulation Kuhlman and Bradley [2019]. While foundational, these methods were limited by approximate energy functions and high computational costs. The advent of deep learning Kingma and Welling [2013]; Goodfellow et al. [2014]; Vaswani et al. [2017]; LeCun et al. [2002]; Devlin et al. [2019]; Sohl-Dickstein et al. [2015] enabled the use of geometric deep learning, where protein structures are modeled as graphs and processed with GNNs or equivariant networks to map structures to sequences. Representative works include GraphTrans Ingraham et al. [2019], GVP Jing et al. [2020], and ProteinMPNN Dauparas et al. [2022], which achieved strong sequence recovery through optimized graph representations and message passing. PiFold Gao et al. [2022a] further improved efficiency with non-autoregressive predictions. Despite these advances, purely geometric models rely on limited structure-sequence pairs and struggle to capture evolutionary covariation present in massive sequence databases. To address this, such as LM-Design Zheng et al. [2023], Knowledge-Design Gao et al.

[2023], and Bridge-IF Zhu et al. [2024], couple protein language models (PLMs) Rives et al. [2021]; Lin et al. [2023] with geometric models. Typically, these methods follow a serial refinement pipeline, where a PLM refines the initial output of a structure encoder. While effective, this serial pipeline suffers from decoupled refinement where PLMs lack access to structural constraints. In contrast, our Sym-Fold introduces a symmetric dual-path architecture that synergizes evolutionary knowledge from PLMs with structural priors from MPLMs, ensuring that the sequence refinement remains grounded in both biological evolution and geometric constraints.

Multimodal Protein Language Models (MPLMs). Recent MPLMs aim to unify sequences and structures under a single framework. ESM-3 Hayes et al. [2025] embodies this direction by jointly modeling sequences, atomic coordinates, and functional annotations with Transformers, though its discretization of continuous coordinates constrains its direct application to inverse folding. Other notable MPLMs include DPLM-2 Wang et al. [2024], a diffusion-based framework capable of handling discrete and structural modalities, and Evolla Zhou et al. [2025], an 80B-parameter model that integrates protein-specific knowledge for design and functional tasks. These models highlight the promise of MPLMs for generative protein tasks, though challenges remain in fully exploiting them for inverse folding. In this study, we harness the structure-aware capabilities of MPLMs through our proposed framework, further unlocking their potential for such tasks.

## 3 Method

## 3.1 Preliminaries

Baseline. Protein inverse folding seeks to recover an amino acid sequence $\pmb { s } = ( s _ { 1 } , s _ { 2 } , \ldots , s _ { \ell } )$ from a given protein structure $\pmb { x } = \left( x _ { 1 } , x _ { 2 } , \dots , x _ { \ell } \right)$ , where $s _ { i } \in S$ denotes the residue identity at position i, S is the set of amino acids, and $x _ { i }$ denotes the atomic coordinates for residue $i .$ The one-hot encoding of $s _ { i }$ is $y _ { i } \in \{ 0 , 1 \} ^ { | s | }$ with components $( { \pmb y } _ { i } ) _ { a } =$ $\mathbb { I } \{ a = \bar { s _ { i } } \} , a \in \mathcal { S }$ , where I is the indicator function. Given training set $\mathcal { D } = \{ ( \pmb { x } , \pmb { y } ) \}$ , the baseline methods learn a structure encoder $f _ { e }$ producing per-position prediction (with entire structure of the protein), and the per-sample training objective could be written as sequence-averaged cross-entropy loss:

$$
\mathcal { L } _ { \mathrm { b a s e } } = - \frac { 1 } { \ell } \sum _ { i = 1 } ^ { \ell } \pmb { y } _ { i } \cdot \log \frac { \exp ( f _ { e } ( \pmb { x } ) _ { i } ) } { \sum _ { k } \exp ( f _ { e } ( \pmb { x } ) _ { i , k } ) } ,\tag{1}
$$

where exp(·) is the exponential, $f _ { e } ( \pmb { x } ) _ { i } \in \mathbb { R } ^ { | S | }$ is the predicted logits for $s _ { i }$ , and $k$ indexes classes of $s$ . In training, one minimizes the empirical mean of $\mathcal { L } _ { \mathrm { b a s e } }$ over all samples in the dataset D.

PLM-based Method. To incorporate sequence evolutionary knowledge, recent methods use a protein language model (PLM) $f _ { s }$ to refine the predictions of an optimized structure encoder $f _ { e } ^ { * }$ . Specifically, they first train a structure encoder $f _ { e }$ via Eq. (1), and then finetune $f _ { s }$ conditioning on the outputs of $f _ { e } ^ { * }$ , to conduct post hoc refinement. The sequence-level

training objective is:

$$
\mathcal { L } _ { \mathrm { p l m } } = - \frac { 1 } { \ell } \sum _ { i = 1 } ^ { \ell } y _ { i } \cdot \log \frac { \exp ( f _ { s } ( \hat { s } ) _ { i } ) } { \sum _ { k } \exp ( f _ { s } ( \hat { s } ) _ { i , k } ) } , ~ \hat { s } _ { j } = \arg \operatorname* { m a x } _ { k } f _ { e } ^ { * } ( { \pmb x } ) _ { j , k } ,\tag{2}
$$

where sˆ is the output of the optimal (and frozen) structure encoder $f _ { e } ^ { * }$ , the argmax is taken over the class index k for each residue $j ,$ , and $f _ { s }$ is trainable during finetuning. Note that, since no structure information x is available to $f _ { s }$ during refinement, the post-hoc editing is inherently limited by $f _ { e } ^ { * }$

## 3.2 Our Method — SymFold

Overview. To address the limitations of structure-ignorant refinement in PLM-based methods, we introduce SymFold, a symmetric dual-path architecture that couples structural priors from multimodal protein language models (MPLMs) with evolutionary priors from PLMs for stronger refinement. As shown in Figure $^ { 2 , }$ we first follow standard practice to train a structure encoder $f _ { e }$ by Eq. (1). We then encode structure knowledge via an MPLM and sequence knowledge via a PLM, and use them to adjust the sequence predictions through an Adaptive Synergistic Fusion module. A selfcorrection scheme is further proposed to align training and inference to mitigate exposure bias and reduce error accumulation. Together, these components enable a more harmonious and effective use of pretrained knowledge for protein inverse folding.

Symmetric Refinement. Following PLM-based practice, we first train a structure encoder via Eq. (1) to obtain $f _ { e } ^ { * }$ . We then incorporate sequence priors using a PLM $f _ { s }$ based on the predicted sequence ${ \hat { \pmb { s } } } =$ argmax $\bar { f } _ { e } ^ { * } ( { \pmb x } )$ and incorporate structure priors using an MPLM $f _ { m }$ conditioned on x. These two signals are fused by a symmetric module $f _ { \mathrm { s y m } }$ , and the per-sample training objective of our symmetric refinement is:

$$
\mathcal { L } _ { \mathrm { o u r s } } = - \frac { 1 } { \ell } \sum _ { i = 1 } ^ { \ell } \pmb { y } _ { i } \cdot \log \frac { \exp ( f _ { s y m } ( f _ { s } , \hat { \pmb { s } } , f _ { m } , \pmb { x } ) _ { i } ) } { \sum _ { k } \exp ( f _ { s y m } ( f _ { s } , \hat { \pmb { s } } , f _ { m } , \pmb { x } ) _ { i , k } ) } ,\tag{3}
$$

where $\exp ( \cdot )$ is the exponential function, · is the dot product, k indexes classes of S. To effectively fuse pretrained knowledge from both paths, we introduce an Adaptive Synergistic Fusion Module that dynamically weights and blends their complementary cues. In addition, because inference aggregates more information sources, we propose a Self-Correction strategy that incrementally incorporates the dual paths step by step, thereby mitigating the train–test discrepancy.

Adaptive Synergistic Fusion. Unlike PLM-based refinement, which derives priors solely from sequence context, our symmetric refinement jointly leverages sequence context and structure signals. Since some residues are tightly constrained by local geometry while others depend more on long-range sequence dependencies, we design a residue-wise Adaptive Synergistic Fusion module that dynamically integrates these complementary cues:

![](images/1ede694dbe088747225baec4b83c273cf54c58eb8263c72feb6dade65b0b5b05.jpg)  
Figure 2: Overview of SymFold. SymFold employs a symmetric refinement framework integrating structural priors from MPLMs and evolutionary priors from PLMs. Starting with a pre-trained structure encoder $f _ { e } ^ { * }$ , initial predictions are refined through parallel structure and sequence branches. The Adaptive Synergistic Fusion module combines these signals using structure-aware coefficients α(x) in Eq.( 4), ⊙ denotes element-wise product, followed by iterative self-correction to align training with inference.

$$
f _ { s y m } ( f _ { s } , \hat { s } , f _ { m } , \pmb { x } ) _ { i } = f _ { s } ( \hat { \pmb { s } } ) _ { i } + \alpha ( \pmb { x } ) _ { i } \cdot f _ { m } ( \pmb { x } , \hat { \pmb { s } } ) _ { i } ,\tag{4}
$$

where $\hat { \pmb { s } }$ denotes the sequence predicted by the frozen optimal structure encoder $f _ { e } ^ { * }$ , · is the dot product, and $f _ { s }$ and $f _ { m }$ could be fine-tuned during training. The residue-wise structureaware coefficient $\alpha ( { \pmb x } )$ controls how strongly residue i is influenced by the corresponding structure and could be realized as $\alpha ( \pmb { x } ) = \mathbf { w } ^ { \top } f _ { e } ^ { * } ( \pmb { x } )$ , where w is a trainable projection vector.

Self-Correction. Current PLM-based methods typically perform iterative generation at inference, feeding the predicted sequence back into the PLM for several rounds, which induces a mismatch between one-step training and multi-step inference. Under our symmetric framework, this discrepancy can be amplified since multiple knowledge sources are fused at each step. To address this, we introduce a Self-Correction strategy that unrolls the symmetric refinement during training to mirror the iterative inference procedure. The iterative process could be written as:

$$
\hat { \pmb { s } } ^ { ( 0 ) } = \arg \operatorname* { m a x } f _ { e } ^ { * } ( \pmb { x } )\tag{5}
$$

$$
\hat { \pmb { s } } _ { i } ^ { ( t ) } = \underset { k } { \operatorname { a r g m a x } } f _ { s y m } ( f _ { s } , \hat { \pmb { s } } ^ { ( t - 1 ) } , f _ { m } , \pmb { x } ) _ { i , k } ,\tag{6}
$$

where $t = 1 , \dots , T$ is iterative step and $T$ is the total step. During training, we compute the loss using the refined prediction at T for the loss calculation.

## 4 Experiments

## 4.1 Experimental Protocol

Datasets. We train our model on CATH4.2 and CATH4.3. The CATH4.2 dataset consists of 18,024 proteins for training, 608 proteins for validation, and 1,120 proteins for testing, following the same data splitting as GraphTrans Ingraham et al. [2019]. The CATH4.3 dataset includes 16,153 structures for training, 1,457 for validation, and 1,797 for testing, following the same data splitting as ESMIF Hsu et al. [2022].

For comprehensive evaluation, we assess our model on multiple protein structure datasets. We test on the CATH4.2 test set with 1,120 proteins and the CATH4.3 test set with 1,797 proteins to measure performance on protein folds similar to those seen during training. Additionally, we evaluate on the TS50 and TS500 datasets, comprising 50 and 500 proteins respectively as established by Li et al. [2014]. To examine our model’s generalization capabilities, we further test on the challenging CASP15 Elofsson [2023] and CASP16 Yuan et al. [2025] monomeric tertiary structure targets, which include 45 and 50 proteins respectively and represent novel protein structures. The specific protein identifiers and their official release times for CASP15 and CASP16 are provided in Appendix to facilitate reproducibility and cross-study com parison.

Baseline Models and Evaluation Metrics. We compare SymFold with recent graph-based models (StructGNN, GraphTrans Ingraham et al. [2019], GCA Tan et al. [2022], GVP Jing et al. [2020], GVP-large Hsu et al. [2022], AlphaDesign Gao et al. [2022b], ESM-IF Hsu et al. [2022], ProteinMPNN Dauparas et al. [2022],PiFold Gao et al. [2022a],GraDe-IF Yi et al. [2023]), and PLM-based optimized models (LM-Design Zheng et al. [2023], Knowledge-Design Gao et al. [2023], Bridge-IF Zhu et al. [2024]). We report perplexity and median recovery rate to assess performance(computation details in Appendix), with evaluations on the CATH dataset divided into three protein types: Short proteins (length ≤ 100), Single-chain proteins (with only 1 chain in CATH), and All proteins.

Implementation Details. SymFold employs a frozen pretrained PiFold encoder for fundamental structural representation, while only the Adaptive Synergistic Fusion module is fully fine-tuned. As language modeling backbones, we use the open-source ESM-3 (1.4B, MPLM) and its co-trained counterpart ESM-C (600M, PLM) Hayes et al. [2025]. LoRA Hu et al. [2022] is applied solely to MPLM and PLM with rank $r \ = \ 8 ,$ , scaling factor $\alpha = 3 2$ , and dropout 0.1, leading to ∼0.1% trainable parameters in total (further configuration details are provided in the Appendix). Training is performed on a single NVIDIA A800 GPU with batch size

4, cosine learning rate scheduling, and typically converges within 5 epochs. For the self-correction training strategy, we set $T = 2$ (more details about the inference phase setting of T are in Appendix). All results are reported based on this final configuration.

## 4.2 Results and Analysis

Through the following Q&A, we provide in-depth discussions of the experimental results for protein inverse folding, offering a comprehensive analysis of our SymFold model’s performance across various benchmarks. We address key questions regarding performance comparisons, architectural innovations, generalization capabilities, ablation studies, and biological plausibility to systematically evaluate our approach’s contributions to the field.

## Q1. How does SymFold perform on multiple benchmarks?

As shown in Table 1, SymFold achieves state-of-the-art performance across all evaluated benchmarks. On CATH4.2, it reaches a 63.11% overall sequence recovery rate, outperforming the best previous PLM-based method, Knowledge-Design, by 2.34% and exceeding structure-only methods such as PiFold by 11.45 percentage points. The gains are especially notable for diverse protein categories: recovery on short proteins (length ≤ 100) reaches 50.00%, an improvement of 5.34% over Knowledge-Design, while performance on single-chain proteins rises to 55.45%, 6.49% higher than Bridge-IF. These results highlight SymFold’s effectiveness in leveraging both sequence and structural information, particularly for structurally constrained proteins. On CATH4.3, the model maintains its advantage with a recovery rate of 62.23%, showing consistent improvements across datasets. Furthermore, SymFold achieves the lowest perplexity across all protein categories, indicating more confident and accurate sequence predictions.

As shown in Table 2, SymFold establishes new state-ofthe-art results on both TS50 and TS500, which are more diverse than CATH. On TS50, it achieves a perplexity of 2.84 and recovery of 66.02%, surpassing Knowledge-Design by 3.23 percentage points in recovery and reducing perplexity by 0.26. On the larger TS500, SymFold reaches 70.48% recovery with the lowest perplexity of 2.74. These consistent improvements, with perplexity reductions of 8.4% (TS50) and 4.2% (TS500), demonstrate both scalability and robustness of the symmetric architecture in capturing sequence–structure relationships.

To further validate our model’s generalization capability, we evaluated SymFold on the CASP15 and CASP16 datasets Yuan et al. [2025], which provide challenging benchmarks with the latest experimentally determined protein structures. As shown in Table 3, SymFold achieves 56.57% and 52.18% recovery rates on CASP15 and CASP16 respectively, outperforming ProteinMPNN (43.06% and 39.32%), PiFold (48.45% and 47.10%), and LM-Design (50.28% and 48.64%). We attempted to include Knowledge-Design and Bridge-IF, but encountered reproducibility issues with their released code. The consistent improvements over reproducible baselines demonstrate SymFold’s strong performance on novel protein structures.

## Q2. Are the various module designs in SymFold reasonable and valuable?

Ablations for our symmetric design. We conduct ablation studies to validate our symmetric dual-path design by evaluating four configurations: (1) the complete SymFold with both streams, (2) SymFold without the PLM branch (w/o PLM), and (3) SymFold without the MPLM branch (w/o MPLM).

The results in Table 4 demonstrate the synergistic benefits of our symmetric dual-path architecture. The full SymFold model achieves the best performance across all metrics, with perplexity of 3.23 and recovery rate of 63.11% on the full test set. When we ablate the PLM branch (w/o PLM), keeping only the MPLM stream, performance drops significantly. Similarly, ablating the MPLM branch (w/o MPLM) while keeping only the PLM stream also degrades performance. These results confirm that both structural and sequence information are essential and complementary.

Notably,on the CATH4.2 and CATH4.3 datasets, direct zero-shot evaluation of a pretrained MPLM yields perplexities of 6.50/6.27 and recovery rates of 42.03%/40.98%, demonstrating its inability to generalize effectively to inverse folding. Despite further fine-tuning, performance remains limited, with perplexities only improving to 4.82/4.68 and recovery rates to 48.46%/48.73%. This raw performance underscores that while MPLMs encode rich structural priors, effective adaptation—rather than direct application—is essential for unlocking their potential. By contrast, our symmetric dual-path design with joint training leverages the complementary strengths of PLMs and MPLMs, achieving significantly stronger results. For completeness, additional comparisons integrating the MPLM within another mainstream architecture are provided in the Appendix.

Ablations for our proposed modules. Table 6 summarizes the ablation study of SymFold’s three core components: the Prior training strategy, Adaptive Synergistic Fusion (ASF), and Self-Correction (SC). The Prior strategy utilizes a frozen pre-trained encoder to extract stable structural features, which consistently lowers perplexity and aids sequence recovery. ASF optimizes decoding by dynamically integrating complementary signals from multiple expert modules, while SC mitigates error propagation through iterative refinement of intermediate predictions. The results validate the specific contribution of each module: the Prior establishes a robust representation foundation, ASF enhances signal reliability, and SC ensures output quality. When combined, these components operate synergistically to achieve the lowest perplexity and highest recovery across all categories, proving their collective essentiality for accurate and robust protein sequence generation.

## Q3. Is SymFold model-agnostic?

We evaluated the SymFold framework’s compatibility with different foundation models by integrating two distinct structure encoders: ProteinMPNN and PiFold. These experiments were conducted under controlled conditions, utilizing the same MPLM and PLM components while maintaining consistent training strategies. As shown in Table 5, our framework consistently enhances the performance of both encoders across all evaluated categories. SymFold achieves substantial improvements in sequence recovery for both Protein-MPNN (from 49.87% to 61.45%) and PiFold (from 51.66% to 63.11%), with corresponding perplexity reductions. Notably, when equipped with the more advanced PiFold encoder, SymFold achieves superior performance across all metrics, indicating that better structure encoders provide more precise prior conditions for the synergistic generation process. These results confirm that the SymFold architecture is fundamentally model-agnostic while scaling its performance with the quality of underlying components.

Table 1: Results comparison on the CATH4.2 and CATH4.3 datasets. Some benchmarked results are quoted from [Gao et al., 2023; Zhu et al., 2024]. Perplexity (↓) indicates sequence prediction uncertainty, where lower values are better; Recovery (%↑) measures sequence accuracy, where higher is better. The best and suboptimal results are labeled with bold and underline.
<table><tr><td rowspan="2">Model</td><td rowspan="2"></td><td colspan="3">Perplexity↓</td><td colspan="3">Recovery % ↑</td></tr><tr><td>Short</td><td>Single-chain</td><td>All</td><td>Short</td><td>Single-chain</td><td>All</td></tr><tr><td rowspan="14">CA4</td><td>StructGNN</td><td>8.29</td><td>8.74</td><td>6.40</td><td>29.44</td><td>28.26</td><td>35.91</td></tr><tr><td>GraphTrans</td><td>8.39</td><td>8.83</td><td>6.63</td><td>28.14</td><td>28.46</td><td>35.82</td></tr><tr><td>GCA</td><td>7.09</td><td>7.49</td><td>6.05</td><td>32.62</td><td>31.10</td><td>37.64</td></tr><tr><td>GVP</td><td>7.23</td><td>7.84</td><td>5.36</td><td>30.60</td><td>28.95</td><td>39.47</td></tr><tr><td>AlphaDesign</td><td>7.32</td><td>7.63</td><td>6.30</td><td>34.16</td><td>32.66</td><td>41.31</td></tr><tr><td>ProteinMPNN</td><td>6.21</td><td>6.68</td><td>4.61</td><td>36.35</td><td>34.43</td><td>45.96</td></tr><tr><td>PiFold</td><td>6.04</td><td>6.31</td><td>4.55</td><td>39.84</td><td>38.53</td><td>51.66</td></tr><tr><td>GraDe-IF</td><td>5.49</td><td>6.21</td><td>4.35</td><td>45.27</td><td>42.77</td><td>52.21</td></tr><tr><td>LM-Design</td><td>5.66</td><td>5.52</td><td>4.01</td><td>46.84</td><td>48.63</td><td>56.63</td></tr><tr><td>Knowledge-Design</td><td>5.48</td><td>5.16</td><td>3.46</td><td>44.66</td><td>45.45</td><td>60.77</td></tr><tr><td>Bridge-IF</td><td>5.68</td><td>5.06</td><td>3.83</td><td>43.86</td><td>48.96</td><td>58.59</td></tr><tr><td>SymFold</td><td>4.69</td><td>4.01</td><td>3.23</td><td>50.00</td><td>55.45</td><td>63.11</td></tr><tr><td></td><td>7.68</td><td>6.12</td><td>6.17</td><td>32.60</td><td>39.40</td><td>39.20</td></tr><tr><td rowspan="8">CAAT43</td><td>GVP-large ESM-IF</td><td>8.18</td><td>6.33</td><td>6.44</td><td>31.30</td><td></td><td></td></tr><tr><td>+ 1.2M AF2 data</td><td>6.05</td><td>4.00</td><td></td><td>38.10</td><td>38.50</td><td>38.30</td></tr><tr><td>ProteinMPNN</td><td>6.35</td><td>6.25</td><td>4.01 4.89</td><td>40.73</td><td>51.50</td><td>51.60</td></tr><tr><td>PiFold</td><td>5.50</td><td>5.76</td><td>4.44</td><td>43.84</td><td>41.18 44.32</td><td>47.69</td></tr><tr><td>LM-Design</td><td>5.66</td><td>5.52</td><td>4.01</td><td>42.84</td><td>43.69</td><td>50.62</td></tr><tr><td>Knowledge-Design</td><td>5.47</td><td>5.23</td><td>3.49</td><td>43.89</td><td>45.95</td><td>55.68</td></tr><tr><td>Bridge-IF</td><td>5.17</td><td>4.63</td><td>3.68</td><td>50.00</td><td>53.49</td><td>60.38</td></tr><tr><td>SymFold</td><td>4.26</td><td>3.91</td><td>3.22</td><td>55.81</td><td>58.20</td><td>58.93 62.23</td></tr></table>

Table 2: Results comparison on the TS50&TS500 dataset. Some benchmarked results are quoted from [Gao et al., 2023]. The best and suboptimal results are labeled with bold and underline.
<table><tr><td>Model</td><td colspan="2">TS50 TS500</td></tr><tr><td></td><td colspan="2">Perp. ↓ Rec. % ↑ Perp. ↓ Rec. % ↑</td></tr><tr><td>StructGNN GraphTrans</td><td>5.40 43.89 5.60 42.20 5.16</td><td>4.98 45.69 44.66</td></tr><tr><td>GVP</td><td>4.71 44.14 4.20</td><td>49.14</td></tr><tr><td>GCA</td><td>5.09 47.02 4.72</td><td>47.74</td></tr><tr><td>AlphaDesign</td><td>5.25 48.36</td><td>4.93 49.23</td></tr><tr><td>ProteinMPNN</td><td>3.93 54.43</td><td>3.53 58.08</td></tr><tr><td>PiFold</td><td>3.86 58.72</td><td>3.44 60.42</td></tr><tr><td>LM-Design</td><td>3.50 57.89 3.19</td><td>63.65</td></tr><tr><td>Knowledge-Design</td><td>3.10 62.79 2.86</td><td>69.19</td></tr><tr><td>SymFold</td><td>2.84 66.02 2.74</td><td>70.48</td></tr></table>

Table 3: Comparison of results on CASP15 and CASP16 datasets. The best and suboptimal results are labeled with bold and underline.
<table><tr><td rowspan="2">Model</td><td colspan="2">CASP15</td><td colspan="2">CASP16</td></tr><tr><td>Perp. ↓</td><td>Rec. % ↑</td><td>Perp. ↓</td><td>Rec. % ↑</td></tr><tr><td>ProteinMPNN</td><td>5.69</td><td>43.06</td><td>7.19</td><td>39.32</td></tr><tr><td>PiFold</td><td>4.87</td><td>48.45</td><td>5.66</td><td>47.10</td></tr><tr><td>LM-Design</td><td>5.12</td><td>50.28</td><td>5.79</td><td>48.64</td></tr><tr><td>SymFold</td><td>3.90</td><td>56.57</td><td>4.50</td><td>52.18</td></tr></table>

## Q4. Are the generated sequences biologically plausible?

To evaluate the biological plausibility of sequences generated by SymFold, we conducted a validation experiment using AlphaFold3 Abramson et al. [2024] for structure prediction. We selected two proteins from CASP16 with varying lengths: T1234 (377 residues) and T1266 (295 residues). For each target structure, we generated sequences using Sym-Fold, LM-Design, PiFold, and ProteinMPNN, then used AlphaFold3 to predict structures from these designed sequences.

As shown in Figure 3, SymFold-generated sequences exhibit stronger consistency with the intended target structures across recovery, pLDDT, and TM-score. For T1234, our method achieves a sequence recovery of 61.65%, higher than LM-Design (57.52%), ProteinMPNN (50.49%), and Pi-Fold (56.31%). For T1266, SymFold attains 50.85% compared to 46.10%, 39.32%, and 48.14% for the respective baselines. AlphaFold3 predictions suggest that sequences designed by SymFold yield more confident folding models (pLDDT: 78.24 for T1234, 93.15 for T1266) with better structural alignment to the native backbone (TM-scores of 0.85 and 0.97, respectively).

![](images/873c9122cf96a0f22984b8877b1382a1d694855fb9c2cb89288c4ca2793ed7cf.jpg)  
Figure 3: Comparison of sequence recovery, structure prediction confidence (pLDDT), and structural similarity (TM-score) Zhang and Skolnick [2005] for sequences generated by SymFold, LM-Design, ProteinMPNN, and PiFold. The TM-score ranges from 0 to 1 (higher values indicate better structural similarity), while pLDDT ranges from 0 to 100 (higher values indicate greater prediction confidence).

Table 4: Ablation results for the symmetric dual-path architecture. The full SymFold outperforms both single-path variants, confirming the complementary roles of PLM and MPLM.
<table><tr><td rowspan="2">Variant</td><td>Perp. ↓</td><td>Rec. % ↑</td></tr><tr><td>Short Single All</td><td>Short Single All</td></tr><tr><td>SymFold</td><td>4.69 4.01 3.23 50.00</td><td>55.4563.11</td></tr><tr><td>w/o PLM w/o MPLM</td><td>6.11 5.66 5.51 4.73</td><td>4.05 42.55 45.21 56.62 3.61 44.02 50.89 59.56</td></tr></table>

Table 5: Performance comparison of SymFold when using Protein-MPNN versus PiFold as the structure encoder on CATH4.2.
<table><tr><td rowspan="2">Encoder</td><td>Perp. ↓</td><td>Rec. % ↑</td></tr><tr><td>Short Single All Short</td><td>Single All</td></tr><tr><td rowspan="2">ProteinMPNN with SymFold</td><td>6.21 6.68 4.57 36.35</td><td>34.43 49.87</td></tr><tr><td>5.02 4.56 3.34 48.96</td><td>52.28 61.45</td></tr><tr><td>PiFold with SymFold 4.69</td><td>6.04 6.31 4.18 339.84 4.01 3.23 50.00</td><td>38.53 51.66 55.45 63.11</td></tr></table>

Additional visual comparisons on four more CASP16 targets are provided in Appendix. These results provide preliminary in silico evidence that our dual-stream architecture can improve not only recovery metrics but also the predicted structural plausibility of designed sequences. While AlphaFold-derived scores cannot substitute for experimental verification, they offer encouraging indications that SymFold generated sequences are more compatible with the desired folds.

Table 6: Ablation study of the three proposed components in Sym-Fold: Prior (frozen pre-trained structure encoder), ASF (Adaptive Synergistic Fusion of predictions), and SC (Self-Correction via iterative ). ✓ indicates the component is enabled, and × indicates it is disabled.
<table><tr><td rowspan="2">Prior ASF SC</td><td rowspan="2"></td><td rowspan="2"></td><td>Perp. ↓</td><td>Rec. % ↑</td></tr><tr><td>Short Single All</td><td>Short Single All</td></tr><tr><td>X</td><td>X</td><td>X 4.98</td><td>4.40</td><td>3.46 49.52 53.0061.26</td></tr><tr><td>X</td><td>√</td><td>√ 4.92</td><td>4.32</td><td>3.40 48.5653.2661.85</td></tr><tr><td>√</td><td>X</td><td>√ 4.83</td><td>4.20</td><td>3.35 49.75 54.52 62.31</td></tr><tr><td>√</td><td>√</td><td>X 4.80</td><td>4.18</td><td>3.32 49.8154.6062.43</td></tr><tr><td>√</td><td>√</td><td>4.69 V</td><td>4.01</td><td>3.23 50.00 55.45 63.11</td></tr></table>

## 5 Conclusion

We introduce SymFold, a novel symmetric dual-path framework for protein inverse folding that addresses the limitations of serial architectures by fostering continuous interaction between structural constraints and Sequence Evolution knowl edge. Our Adaptive Synergistic Fusion and self-correction mechanisms enhance sequence design, achieving state-ofthe-art performance across benchmarks. Despite these advances, challenges remain. Our evaluations primarily rely on in silico metrics, and the limited extent of wet-lab validation leaves the functional viability of the generated sequences unverified. Future work will explore stronger foundation models to push the boundaries of computational protein design.

## 6 Supplementary Material

## 6.1 Comparative Architecture Designs with MPLM

To further investigate the role of cross-modal integration of MPLMs in protein inverse folding, we replaced the original protein language model (PLM) in the LM-Design framework with an MPLM. By systematically evaluating two representative adapter schemes within this framework, this study thoroughly examines how different connection strategies between multimodal protein language models and structure encoders affect downstream recovery performance, thereby providing clearer insights into the mechanisms of synergy across modalities.We evaluated two representat:

(1) Structure-Aware Adapter (original LM-Design implementation): This variant integrates high-dimensional embeddings from a protein structure encoder through lightweight adapter modules. Cross-modal interaction is realized via multi-head attention, explicitly mixing coordinate-based information with sequence embeddings.

(2) Parameter-Tuning Adapter (modified version): Here, the attention-based fusion is removed. The adapter reduces to simple trainable MLP layers, leaving structural sensitivity to be modeled implicitly by the MPLM itself without direct feature injection.

Both architectures were trained under identical setups (datasets, losses, optimizers, and hyperparameters). We further controlled two experimental factors: (i) Training Strategy – training the structure encoder from scratch versus freezing pretrained encoder parameters; (ii) Input Modality – full multimodal input (amino-acid sequence + 3D coordinates) versus sequence-only input.

<table><tr><td>Training Strategy</td><td>Architecture</td><td>Full Input Seq. Only</td><td></td></tr><tr><td rowspan="2">From Scratch</td><td>Struct-Aware Adapter</td><td>52.46</td><td>53.68</td></tr><tr><td>Param-Tuning Adapter</td><td>53.60</td><td>53.12</td></tr><tr><td rowspan="2">Pretrained + Frozen</td><td>Struct-Aware Adapter</td><td>53.95</td><td>54.56</td></tr><tr><td>Param-Tuning Adapter</td><td>55.33</td><td>54.01</td></tr></table>

Table 7: Recovery performance comparison of adapter architectures with MPLM backbone under different training strategies and input modalities.

This counterintuitive result reveals a phenomenon we term the modality paradox: providing richer multimodal inputs does not automatically translate into better performance and may even lead to degradation due to issues such as feature redundancy and optimization conflicts. In contrast, the Parameter-Tuning Adapter can more stably leverage the inherently structure-aware capacity of MPLMs, indicating that effective cross-modal adaptation requires carefully designed fusion mechanisms. Based on these experiments, we preliminarily conclude that when integrating MPLMs (e.g., ESM3) into protein inverse folding frameworks, it is preferable to preserve their original architecture without substantial modifications. Otherwise, as observed in the LM-Design framework introduced with MPLMs, performance may even decline. This finding provides strong foundational support for the SymFold framework proposed in this paper.

## 6.2 Iterative Inference Analysis

We analyzed the impact of iterative refinement steps (T) on DualFold’s performance with and without Self-Correction. Table 8 shows the results on CATH4.2.

<table><tr><td rowspan="2">T</td><td colspan="2">Recovery(%) ↑</td></tr><tr><td>With Self-Correction</td><td>Without Self-Correction</td></tr><tr><td>1</td><td>62.25</td><td>60.57</td></tr><tr><td>2</td><td>63.11</td><td>61.19</td></tr><tr><td>3</td><td>62.96</td><td>62.23</td></tr><tr><td>4</td><td>62.96</td><td>62.43</td></tr><tr><td>5</td><td>63.10</td><td>62.42</td></tr></table>

Table 8: Impact of iterative refinement steps (T) on sequence recovery rate (%).

The results clearly demonstrate the effectiveness of the Self-Correction mechanism. With Self-Correction, the model achieves superior performance across all iteration counts, with the most significant improvements at lower iterations. Notably, Self-Correction enables the model to reach peak performance (63.11%) early in the refinement process. In contrast, without Self-Correction, the model shows slower convergence and lower overall performance, never reaching the accuracy achieved by Self-Correction even with more iterations.

## 6.3 CASP15 and CASP16 Protein Details

For the purpose of ensuring reproducibility, this appendix summarizes the specific protein targets used in our experiments from the CASP15 and CASP16 datasets. Tables 9 and 10 provide the official release names and dates of each protein. This collection allows future studies to easily identify and cross-reference proteins with their corresponding release times.

## 6.4 Ablation on ASF Module Components

To effectively integrate the predictive capabilities of MPLMs and PLMs, we propose a residue-wise weighting strategy. We constructed a lightweight ResidueWeightingNetwork that utilizes features extracted by the encoder to capture local sequence dependencies through hybrid layers comprising Sigmoid Linear Units (SiLU) and local 1D convolutions (Conv1d). This network outputs normalized weights for each model at every position within the sequence. This residuewise weighting mechanism allows the model to dynamically adjust the contribution of each component based on the specific contextual environment of the amino acids, thereby achieving enhanced feature fusion at the local level.The specific architecture diagram is shown in Figure 4.

To further validate the effectiveness of the Adaptive Synergistic Fusion (ASF) module in our SymFold framework, we conducted ablation studies on its key components. The ASF module, as described in the main paper (Equation 4), dynamically integrates sequence priors from the PLM $( f _ { s } )$ and structural priors from the MPLM $( f _ { m } )$ using a residue-wise gating vector $\boldsymbol { \alpha } ( \boldsymbol { x } ) \in \mathbb { R } ^ { L }$ , where L is the sequence length. This allows the model to balance evolutionary and structural signals at a per-residue (token) level, conditioned on the structural embeddings from the frozen structure encoder $f _ { e } ^ { * }$

<table><tr><td rowspan=1 colspan=1>Protein Name</td><td rowspan=1 colspan=1>Release Time</td><td rowspan=1 colspan=1>Protein Name</td><td rowspan=1 colspan=1>Release Time</td><td rowspan=1 colspan=1>Protein Name</td><td rowspan=1 colspan=1>Release Time</td></tr><tr><td rowspan=1 colspan=1>T1104-D1</td><td rowspan=1 colspan=1>2022-07-11</td><td rowspan=1 colspan=1>T1106s1-D1</td><td rowspan=1 colspan=1>2022-06-06</td><td rowspan=1 colspan=1>T1106s2-D1</td><td rowspan=1 colspan=1>2022-06-06</td></tr><tr><td rowspan=1 colspan=1>T1109-D1</td><td rowspan=1 colspan=1>2022-05-31</td><td rowspan=1 colspan=1>T1119-D1</td><td rowspan=1 colspan=1>2022-06-08</td><td rowspan=1 colspan=1>T1120-D1</td><td rowspan=1 colspan=1>2022-07-14</td></tr><tr><td rowspan=1 colspan=1>T1120-D2</td><td rowspan=1 colspan=1>2022-07-14</td><td rowspan=1 colspan=1>T1121-D1</td><td rowspan=1 colspan=1>2022-06-08</td><td rowspan=1 colspan=1>T1121-D2</td><td rowspan=1 colspan=1>2022-06-08</td></tr><tr><td rowspan=1 colspan=1>T1123-D1</td><td rowspan=1 colspan=1>2022-05-19</td><td rowspan=1 colspan=1>T1124-D1</td><td rowspan=1 colspan=1>2022-07-16</td><td rowspan=1 colspan=1>T1129s2-D1</td><td rowspan=1 colspan=1>2022-07-05</td></tr><tr><td rowspan=1 colspan=1>T1133-D1</td><td rowspan=1 colspan=1>2022-07-18</td><td rowspan=1 colspan=1>T1137s1-D1</td><td rowspan=1 colspan=1>2022-06-22</td><td rowspan=1 colspan=1>T1137s1-D2</td><td rowspan=1 colspan=1>2022-06-22</td></tr><tr><td rowspan=1 colspan=1>T1137s2-D1</td><td rowspan=1 colspan=1>2022-05-30</td><td rowspan=1 colspan=1>T1137s2-D2</td><td rowspan=1 colspan=1>2022-05-30</td><td rowspan=1 colspan=1>T1137s3-D1</td><td rowspan=1 colspan=1>2022-05-31</td></tr><tr><td rowspan=1 colspan=1>T1137s3-D2</td><td rowspan=1 colspan=1>2022-06-22</td><td rowspan=1 colspan=1>T1137s4-D1</td><td rowspan=1 colspan=1>2022-06-22</td><td rowspan=1 colspan=1>T1137s4-D2</td><td rowspan=1 colspan=1>2022-06-22</td></tr><tr><td rowspan=1 colspan=1>T1137s4-D3</td><td rowspan=1 colspan=1>2022-06-22</td><td rowspan=1 colspan=1>T1137s5-D1</td><td rowspan=1 colspan=1>2022-06-22</td><td rowspan=1 colspan=1>T1137s5-D2</td><td rowspan=1 colspan=1>2022-06-22</td></tr><tr><td rowspan=1 colspan=1>T1137s6-D1</td><td rowspan=1 colspan=1>2022-06-22</td><td rowspan=1 colspan=1>T1137s6-D2</td><td rowspan=1 colspan=1>2022-06-22</td><td rowspan=1 colspan=1>T1137s7-D1</td><td rowspan=1 colspan=1>2022-06-01</td></tr><tr><td rowspan=1 colspan=1>T1137s8-D1</td><td rowspan=1 colspan=1>2022-06-01</td><td rowspan=1 colspan=1>T1137s9-D1</td><td rowspan=1 colspan=1>2022-06-01</td><td rowspan=1 colspan=1>T1139-D1</td><td rowspan=1 colspan=1>2022-06-01</td></tr><tr><td rowspan=1 colspan=1>T1145-D1</td><td rowspan=1 colspan=1>2022-08-18</td><td rowspan=1 colspan=1>T1145-D2</td><td rowspan=1 colspan=1>2022-08-18</td><td rowspan=1 colspan=1>T1150-D1</td><td rowspan=1 colspan=1>2022-06-11</td></tr><tr><td rowspan=1 colspan=1>T1157s1-D1</td><td rowspan=1 colspan=1>2022-09-01</td><td rowspan=1 colspan=1>T1157s1-D2</td><td rowspan=1 colspan=1>2022-09-01</td><td rowspan=1 colspan=1>T1157s1-D3</td><td rowspan=1 colspan=1>2022-09-01</td></tr><tr><td rowspan=1 colspan=1>T1157s2-D1</td><td rowspan=1 colspan=1>2022-09-01</td><td rowspan=1 colspan=1>T1157s2-D2</td><td rowspan=1 colspan=1>2022-09-01</td><td rowspan=1 colspan=1>T1157s2-D3</td><td rowspan=1 colspan=1>2022-09-01</td></tr><tr><td rowspan=1 colspan=1>T1170-D1</td><td rowspan=1 colspan=1>2022-07-28</td><td rowspan=1 colspan=1>T1170-D2</td><td rowspan=1 colspan=1>2022-07-28</td><td rowspan=1 colspan=1>T1180-D1</td><td rowspan=1 colspan=1>2022-08-24</td></tr><tr><td rowspan=1 colspan=1>T1187-D1</td><td rowspan=1 colspan=1>2022-08-05</td><td rowspan=1 colspan=1>T1188-D1</td><td rowspan=1 colspan=1>2022-08-05</td><td rowspan=1 colspan=1>T1194-D1</td><td rowspan=1 colspan=1>2022-08-05</td></tr></table>

Table 9: CASP15 protein names and release times.

<table><tr><td rowspan=1 colspan=1>Protein Name</td><td rowspan=1 colspan=1>Release Time</td><td rowspan=1 colspan=1>Protein Name</td><td rowspan=1 colspan=1>Release Time</td><td rowspan=1 colspan=1>Protein Name</td><td rowspan=1 colspan=1>Release Time</td></tr><tr><td rowspan=1 colspan=1>T1237</td><td rowspan=1 colspan=1>2024-07-20</td><td rowspan=1 colspan=1>T1206</td><td rowspan=1 colspan=1>2024-07-17</td><td rowspan=1 colspan=1>T1234-D1</td><td rowspan=1 colspan=1>2024-09-09</td></tr><tr><td rowspan=1 colspan=1>T1276</td><td rowspan=1 colspan=1>2024-08-19</td><td rowspan=1 colspan=1>T1276-D1</td><td rowspan=1 colspan=1>2024-08-19</td><td rowspan=1 colspan=1>T1272s3</td><td rowspan=1 colspan=1>2024-07-29</td></tr><tr><td rowspan=1 colspan=1>T1279-D1</td><td rowspan=1 colspan=1>2024-07-25</td><td rowspan=1 colspan=1>T1214v1</td><td rowspan=1 colspan=1>2024-08-12</td><td rowspan=1 colspan=1>T1272s9</td><td rowspan=1 colspan=1>2024-11-14</td></tr><tr><td rowspan=1 colspan=1>T1212</td><td rowspan=1 colspan=1>2024-07-04</td><td rowspan=1 colspan=1>T1272s1</td><td rowspan=1 colspan=1>2024-07-29</td><td rowspan=1 colspan=1>T1259</td><td rowspan=1 colspan=1>2024-07-25</td></tr><tr><td rowspan=1 colspan=1>T1235</td><td rowspan=1 colspan=1>2024-07-04</td><td rowspan=1 colspan=1>T1279</td><td rowspan=1 colspan=1>2024-07-20</td><td rowspan=1 colspan=1>T1298-D1</td><td rowspan=1 colspan=1>2024-08-16</td></tr><tr><td rowspan=1 colspan=1>T1274-D1</td><td rowspan=1 colspan=1>2024-09-03</td><td rowspan=1 colspan=1>T1201-D1</td><td rowspan=1 colspan=1>2024-09-24</td><td rowspan=1 colspan=1>T1298-D2</td><td rowspan=1 colspan=1>2024-08-16</td></tr><tr><td rowspan=1 colspan=1>T1272s6-D1</td><td rowspan=1 colspan=1>2024-11-14</td><td rowspan=1 colspan=1>T1240-D1</td><td rowspan=1 colspan=1>2024-08-02</td><td rowspan=1 colspan=1>T1201</td><td rowspan=1 colspan=1>2024-05-19</td></tr><tr><td rowspan=1 colspan=1>T1212-D1</td><td rowspan=1 colspan=1>2024-07-04</td><td rowspan=1 colspan=1>T1237-D1</td><td rowspan=1 colspan=1>2024-07-20</td><td rowspan=1 colspan=1>T1266-D1</td><td rowspan=1 colspan=1>2024-07-02</td></tr><tr><td rowspan=1 colspan=1>T1235-D1</td><td rowspan=1 colspan=1>2024-09-09</td><td rowspan=1 colspan=1>T1234</td><td rowspan=1 colspan=1>2024-07-04</td><td rowspan=1 colspan=1>T1299-D1</td><td rowspan=1 colspan=1>2024-09-11</td></tr><tr><td rowspan=1 colspan=1>T1266</td><td rowspan=1 colspan=1>2024-07-02</td><td rowspan=1 colspan=1>T1272s7</td><td rowspan=1 colspan=1>2024-07-29</td><td rowspan=1 colspan=1>T1272s8-D1</td><td rowspan=1 colspan=1>2024-11-14</td></tr><tr><td rowspan=1 colspan=1>T1214v2</td><td rowspan=1 colspan=1>2024-08-12</td><td rowspan=1 colspan=1>T1272s2-D1</td><td rowspan=1 colspan=1>2024-08-03</td><td rowspan=1 colspan=1>T1299</td><td rowspan=1 colspan=1>2024-09-11</td></tr><tr><td rowspan=1 colspan=1>T1272s8</td><td rowspan=1 colspan=1>2024-11-14</td><td rowspan=1 colspan=1>T1274</td><td rowspan=1 colspan=1>2024-08-14</td><td rowspan=1 colspan=1>T1298</td><td rowspan=1 colspan=1>2024-08-16</td></tr><tr><td rowspan=1 colspan=1>T1272s6</td><td rowspan=1 colspan=1>2024-11-14</td><td rowspan=1 colspan=1>T1240</td><td rowspan=1 colspan=1>2024-08-01</td><td rowspan=1 colspan=1>T1210</td><td rowspan=1 colspan=1>2024-07-15</td></tr><tr><td rowspan=1 colspan=1>T1272s2</td><td rowspan=1 colspan=1>2024-11-14</td><td rowspan=1 colspan=1>T1214-D1</td><td rowspan=1 colspan=1>2024-09-12</td><td rowspan=1 colspan=1>T1272s4</td><td rowspan=1 colspan=1>2024-07-29</td></tr><tr><td rowspan=1 colspan=1>T1206-D1</td><td rowspan=1 colspan=1>2024-07-17</td><td rowspan=1 colspan=1>T1259-D1</td><td rowspan=1 colspan=1>2024-07-25</td><td rowspan=1 colspan=1>T1210-D1</td><td rowspan=1 colspan=1>2024-09-24</td></tr><tr><td rowspan=1 colspan=1>T1272s9-D1</td><td rowspan=1 colspan=1>2024-11-14</td><td rowspan=1 colspan=1>T1240-D2</td><td rowspan=1 colspan=1>2024-08-02</td><td rowspan=1 colspan=1>T1279-D2</td><td rowspan=1 colspan=1>2024-07-25</td></tr><tr><td rowspan=1 colspan=1>T1272s5</td><td rowspan=1 colspan=1>2024-07-29</td><td rowspan=1 colspan=1>T1214</td><td rowspan=1 colspan=1>2024-07-10</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr></table>

Table 10: CASP16 protein names and release times.

We compare the proposed token-level adaptive fusion against several variants:

• Average Fusion: A simple uniform averaging of the logits from $f _ { s }$ and $f _ { m }$ , where each model contributes equally without any adaptive weighting $( \alpha ( x ) = 0 . 5 $ for all residues).

• Model-Leval Fusion: A variant employing two learnable global scalars $( \alpha , \beta )$ that assign fixed weights to the outputs of the PLM and MPLM, applied uniformly across all residues.

• Sequence-Level Mixing: Adaptive weighting at the sequence level, where all tokens in a sequence share a single weight vector derived from global structural features.

• Token-Level Mixing (Ours): The proposed residuewise adaptive fusion, where each token has an independent weight computed via a lightweight projection network on per-residue embeddings.

These variants were evaluated on the CATH4.2 benchmark, measuring sequence recovery rate (%) as the primary metric. The results are summarized in Table 11.

The results demonstrate that finer-grained adaptive fusion yields superior performance. The token-level mixing in our ASF outperforms coarser variants by up to 0.96%, highlight ing the importance of residue-specific balancing. This is particularly beneficial for proteins where structural constraints vary across residues $( \mathrm { e . g . }$ , core vs. surface regions). Coarser fusions, such as sequence-level or global weighting, fail to capture these local variations, leading to suboptimal integration of priors. These findings corroborate the rationale for ASF’s design, as coarser alternatives compound errors in structurally heterogeneous regions, while token-level adaptation enhances compatibility with both evolutionary and geometric constraints.

![](images/040a3eabdc9b6a0a5017d51a1a2b46591a4256a215c781063795ed63bd64613e.jpg)  
Figure 4: This module takes features extracted by structure encoders as input, processes them through a sequential layer comprising Layer Normalization, Linear transformation, Sigmoid Linear Unit , and 1D Convolution , and finally outputs position-wise adaptive normalized weighting coefficients α(x).

## 6.5 Additional Folding Results with Alphafold3

To further validate the biological plausibility of DualFold-generated sequences, we extended our AlphaFold3 analysis to four additional CASP16 targets: T1214(length=652), T1235(length=114), T1299(length=68), and T1259(length=204). For each case, we compared the AlphaFold3-predicted structure of the sequence produced by DualFold against the native target backbone.

As illustrated in Figure 5, the predicted structures of all four designed sequences align closely with their respective native folds. Visually, the confidence levels (pLDDT) remain consistently high, and the structural overlays show strong backbone agreement. These results further reinforce that DualFold is capable of generating sequences with not only improved design metrics but also high in silico structural plausibility across targets of different lengths and topologies.

## 6.6 LoRA Configuration for MPLM and PLM Fine-tuning

Both models employ identical LoRA hyperparameters to ensure balanced adaptation. We use a rank of 8, which provides a good balance between parameter efficiency and model expressivity. The scaling factor is set to 32, offering sufficient learning capacity for the inverse folding task while maintaining stability. A dropout rate of 0.1 is applied to the LoRA layers for regularization, and we disable bias adaptation to further reduce the parameter count.

<table><tr><td>Fusion Variant</td><td>Recovery (%)</td></tr><tr><td>Average Fusion</td><td>62.31</td></tr><tr><td>Model-leval</td><td>62.15</td></tr><tr><td>Sequence-Level Mixing</td><td>62.35</td></tr><tr><td>Token-Level Mixing (Our ASF)</td><td>63.11</td></tr></table>

Table 11: Ablation on ASF module components. Performance is reported as sequence recovery rate (%) on CATH4.2.

This configuration results in approximately 0.1% trainable parameters relative to the full model size, significantly reducing computational overhead while maintaining the models ability to adapt to the inverse folding task. The low rank ensures efficient adaptation, while the relatively high scaling factor compensates for the reduced parameter space, allowing the models to learn task-specific patterns effectively.

The LoRA adaptation is implemented using the Parameter Efficient Fine-Tuning (PEFT) framework. During initialization, we load the pretrained ESM-3 and ESM-C models, configure the LoRA adapters with the specified hyperparameters, and freeze all base model parameters. Only the LoRA weights are updated during training, which prevents catastrophic forgetting of the pretrained knowledge while allowing task-specific adaptation.

The symmetric configuration across both models ensures balanced adaptation of sequence and structural knowledge during training. By targeting both attention and feed-forward components, we enable the models to adapt their representation learning capabilities while preserving the fundamental understanding encoded in the pretrained weights.

## 6.7 Mathematical Formulations for Evaluation Metrics

## Perplexity

Perplexity is a fundamental metric for evaluating the quality of language model predictions, defined as the exponential of cross-entropy loss. In protein sequence design tasks, we utilize perplexity to assess the model’s accuracy in predicting authentic amino acid sequences.

Given a protein sequence $\textbf { s } = ~ \{ s _ { 1 } , s _ { 2 } , . . . , s _ { \ell } \}$ , where $s _ { i }$ represents the amino acid at position i and ℓ is the sequence length. The model outputs logits denoted as $\mathbf { z } =$ $\{ \mathbf { z } _ { 1 } , \mathbf { z } _ { 2 } , \ldots , \mathbf { z } _ { \ell } \}$ , where $\mathbf { z } _ { i } \in \mathbb { R } ^ { | \mathbf { \hat { \nu } } | }$ and $| \nu |$ is the vocabulary size (20 standard amino acids).

First, we compute the log probability distribution at each position:

$$
p _ { i } ( k ) = \frac { \exp ( z _ { i , k } ) } { \sum _ { j = 1 } ^ { | \mathcal { V } | } \exp ( z _ { i , j } ) }\tag{7}
$$

where $z _ { i , k }$ represents the logit value for predicting amino acid k at position i.

![](images/735ba56f750b80d9bf07e9889315f9d08d15315f83b124af49810f67a2cb54d4.jpg)  
Figure 5: AlphaFold3-predicted structures for sequences designed by DualFold on additional CASP16 targets (T1214, T1235, T1299, and T1259). Each row shows an overlay of the target structure and the predicted structure from the DualFold-designed sequence. Across all cases, the strong alignment provides further support for the biological plausibility of DualFold outputs.

The log-likelihood for the target amino acid is:

$$
\log p _ { i } ( s _ { i } ) = \log \left( \frac { \exp ( z _ { i , s _ { i } } ) } { \sum _ { j = 1 } ^ { | \mathcal { V } | } \exp ( z _ { i , j } ) } \right) = z _ { i , s _ { i } } - \log \sum _ { j = 1 } ^ { | \mathcal { V } | } \exp ( z _ { i , j } )\tag{8}
$$

## Global Perplexity Calculation

Considering the presence of special tokens such as padding, class (cls), and end-of-sequence (eos) tokens in protein sequences, we define a valid position mask:

$$
{ \mathcal { M } } _ { i } = { \mathbb { k } } [ s _ { i } \neq \operatorname { p a d } ] \cdot { \mathbb { k } } [ s _ { i } \neq \operatorname { c l s } ] \cdot { \mathbb { k } } [ s _ { i } \neq \operatorname { e o s } ] \cdot c _ { i }\tag{9}
$$

where $c _ { i }$ is the coordinate mask used to identify structurally valid positions, and $\nVdash [ \cdot ]$ is the indicator function.

The global average negative log-likelihood is:

$$
\mathrm { N L L } _ { \mathrm { g l o b a l } } = - \frac { \sum _ { i = 1 } ^ { \ell } \mathcal { M } _ { i } \cdot \log p _ { i } ( s _ { i } ) } { \sum _ { i = 1 } ^ { \ell } \mathcal { M } _ { i } }\tag{10}
$$

The final global perplexity is defined as:

$$
\mathrm { P P L _ { g l o b a l } = \exp ( N L L _ { g l o b a l } ) }\tag{11}
$$

Sequence Recovery Rate

The sequence recovery rate measures the degree of matching between generated sequences and target sequences at the amino acid level, serving as a direct indicator for evaluating the accuracy of protein sequence design.

Given a predicted sequence $\hat { \mathbf { s } } = \left\{ \hat { s } _ { 1 } , \hat { s } _ { 2 } , \hat { \mathbf { \Omega } } \cdot \mathbf { \Omega } \cdot \mathbf { \Omega } , \hat { s } _ { \ell } \right\}$ and a target sequence $\mathbf { s } = \{ s _ { 1 } , s _ { 2 } , \ldots , s _ { \ell } \}$ , the sequence recovery rate is defined as the proportion of correct predictions at valid positions.

The correctness indicator function for individual positions:

$$
\delta _ { i } = | \mathcal { H } [ \hat { s } _ { i } = s _ { i } ] \cdot \mathcal { M } _ { i }\tag{12}
$$

where $\mathcal { M } _ { i }$ is the valid position mask defined previously. Sequence-level Recovery Rate

For a single sequence, the recovery rate is calculated as:

$$
\mathrm { R e c o v e r y } _ { \mathrm { s e q } } = \frac { \sum _ { i = 1 } ^ { \ell } \delta _ { i } } { \sum _ { i = 1 } ^ { \ell } \mathcal { M } _ { i } }\tag{13}
$$

Dataset-level Statistics

For a test set containing N sequences, we compute the recovery rate for each sequence and then calculate the median as the final evaluation metric:

$$
\mathrm { R e c o v e r y } _ { \mathrm { m e d i a n } } = \mathrm { m e d i a n } \{ \mathrm { R e c o v e r y } _ { \mathrm { s e q } } ^ { ( 1 ) } , \dots , \mathrm { R e c o v e r y } _ { \mathrm { s e q } } ^ { ( N ) } \}\tag{14}
$$

The use of median instead of mean reduces the impact of outliers and better reflects the model’s overall performance.

## 7 Acknowledgments

This work was supported by the Strategic Priority Research Program of the Chinese Academy of Sciences under Grant No. XDA0460205

## References

Josh Abramson, Jonas Adler, Jack Dunger, Richard Evans, Tim Green, Alexander Pritzel, Olaf Ronneberger, Lindsay Willmore, Andrew J Ballard, Joshua Bambrick, et al. Accurate structure prediction of biomolecular interactions with alphafold 3. Nature, 630(8016):493–500, 2024.

Minkyung Baek, Frank DiMaio, Ivan Anishchenko, Justas Dauparas, Sergey Ovchinnikov, Gyu Rie Lee, Jue Wang, Qian Cong, Lisa N Kinch, R Dustin Schaeffer, et al. Accurate prediction of protein structures and interactions using a three-track neural network. Science, 373(6557):871–876, 2021.

Nadav Brandes, Dan Ofer, Yam Peleg, Nadav Rappoport, and Michal Linial. Proteinbert: a universal deep-learning model of protein sequence and function. Bioinformatics, 38(8):2102–2110, 2022.

Justas Dauparas, Ivan Anishchenko, Nathaniel Bennett, Hua Bai, Robert J Ragotte, Lukas F Milles, Basile IM Wicky, Alexis Courbet, Rob J de Haas, Neville Bethel, et al. Robust deep learning–based protein sequence design using proteinmpnn. Science, 378(6615):49–56, 2022.

Marianne Defresne, Sophie Barbe, and Thomas Schiex. Protein design with deep learning. International Journal of Molecular Sciences, 22(21):11741, 2021.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 conference of the North American chapter of the association for computational linguistics: human language technologies, volume 1 (long and short papers), pages 4171–4186, 2019.

Ahmed Elnaggar, Michael Heinzinger, Christian Dallago, Ghalia Rehawi, Yu Wang, Llion Jones, Tom Gibbs, Tamas Feher, Christoph Angerer, Martin Steinegger, et al. Prottrans: Toward understanding the language of life through self-supervised learning. IEEE transactions on pattern analysis and machine intelligence, 44(10):7112–7127, 2021.

Arne Elofsson. Progress at protein structure prediction, as seen in casp15. Current opinion in structural biology, 80:102594, 2023.

Wenhao Gao, Sai Pooja Mahajan, Jeremias Sulam, and Jeffrey J Gray. Deep learning in protein structural modeling and design. Patterns, 1(9), 2020.

Zhangyang Gao, Cheng Tan, Pablo Chacon, and Stan Z Li.´ Pifold: Toward effective and efficient protein inverse folding. arXiv preprint arXiv:2209.12643, 2022.

Zhangyang Gao, Cheng Tan, and Stan Z Li. Alphadesign: A graph protein design method and benchmark on alphafolddb. arXiv preprint arXiv:2202.01079, 2022.

Zhangyang Gao, Cheng Tan, and Stan Z Li. Knowledgedesign: Pushing the limit of protein design via knowledge refinement. arXiv preprint arXiv:2305.15151, 2023.

Ian J Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. Advances in neural information processing systems, 27, 2014.

Thomas Hayes, Roshan Rao, Halil Akin, Nicholas J Sofroniew, Deniz Oktay, Zeming Lin, Robert Verkuil, Vincent Q Tran, Jonathan Deaton, Marius Wiggert, et al. Simulating 500 million years of evolution with a language model. Science, 387(6736):850–858, 2025.

Chun-Yin Hsieh et al. Elucidating the design space of multimodal protein language models. arXiv preprint arXiv:2504.11454, 2025.

Chloe Hsu, Robert Verkuil, Jason Liu, Zeming Lin, Brian Hie, Tom Sercu, Adam Lerer, and Alexander Rives. Learning inverse folding from millions of predicted structures. In International conference on machine learning, pages 8946–8970. PMLR, 2022.

Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. ICLR, 1(2):3, 2022.

Po-Ssu Huang, Scott E Boyken, and David Baker. The coming of age of de novo protein design. Nature, 537(7620):320–327, 2016.

John Ingraham, Vikas Garg, Regina Barzilay, and Tommi Jaakkola. Generative models for graph-based protein design. Advances in neural information processing systems, 32, 2019.

Bowen Jing, Stephan Eismann, Patricia Suriana, Raphael JL Townshend, and Ron Dror. Learning from protein structure with geometric vector perceptrons. arXiv preprint arXiv:2009.01411, 2020.

John Jumper, Richard Evans, Alexander Pritzel, Tim Green, Michael Figurnov, Olaf Ronneberger, Kathryn Tunyasuvunakool, Russ Bates, Augustin Z<sup>ˇ</sup>´ıdek, Anna Potapenko, et al. Highly accurate protein structure prediction with alphafold. nature, 596(7873):583–589, 2021.

Hamed Khakzad, Ilia Igashov, Arne Schneuing, Casper Goverde, Michael Bronstein, and Bruno Correia. A new age in protein design empowered by deep learning. Cell Systems, 14(11):925–939, 2023.

Diederik P Kingma and Max Welling. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013.

Brian Kuhlman and David Baker. Native protein sequences are close to optimal for their structures. Proceedings of the National Academy of Sciences, 97(19):10383–10388, 2000.

Brian Kuhlman and Philip Bradley. Advances in protein structure prediction and design. Nature reviews molecular cell biology, 20(11):681–697, 2019.

Yann LeCun, Leon Bottou, Yoshua Bengio, and Patrick´ Haffner. Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11):2278–2324, 2002.

Zhixiu Li, Yuedong Yang, Eshel Faraggi, Jian Zhan, and Yaoqi Zhou. Direct prediction of profiles of sequences compatible with a protein structure by neural networks with fragment-based local and energy-based nonlocal profiles. Proteins: Structure, Function, and Bioinformatics, 82(10):2565–2573, 2014.

Zeming Lin, Halil Akin, Roshan Rao, Brian Hie, Zhongkai Zhu, Wenting Lu, Nikita Smetanin, Robert Verkuil, Ori Kabeli, Yaniv Shmueli, et al. Evolutionary-scale prediction of atomic-level protein structure with a language model. Science, 379(6637):1123–1130, 2023.

Yi Liu and Brian Kuhlman. Rosettadesign server for protein design. Nucleic acids research, 34(suppl 2):W235–W238, 2006.

Ali Madani, Ben Krause, Eric R Greene, Subu Subramanian, Benjamin P Mohr, James M Holton, Jose Luis Olmos Jr, Caiming Xiong, Zachary Z Sun, Richard Socher, et al. Large language models generate functional protein sequences across diverse families. Nature biotechnology, 41(8):1099–1106, 2023.

Christine A Orengo, Alex D Michie, Susan Jones, David T Jones, Mark B Swindells, and Janet M Thornton. Cath– a hierarchic classification of protein domain structures. Structure, 5(8):1093–1109, 1997.

Milong Ren, Chungong Yu, Dongbo Bu, and Haicang Zhang. Accurate and robust protein sequence design with carbondesign. Nature Machine Intelligence, 6(5):536–547, 2024.

Alexander Rives, Joshua Meier, Tom Sercu, Siddharth Goyal, Zeming Lin, Jason Liu, Demi Guo, Myle Ott, C Lawrence Zitnick, Jerry Ma, et al. Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences. Proceedings of the National Academy ofSciences, 118(15):e2016239118, 2021.

Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In International conference on machine learning, pages 2256–2265. pmlr, 2015.

Cheng Tan, Zhangyang Gao, Jun Xia, Bozhen Hu, and Stan Z Li. Generative de novo protein design with global context. arXiv preprint arXiv:2204.10673, 2022.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.

Xinyou Wang, Zaixiang Zheng, Fei Ye, Dongyu Xue, Shujian Huang, and Quanquan Gu. Dplm-2: A multimodal diffusion protein language model. arXiv preprint arXiv:2410.13782, 2024.

Kai Yi, Bingxin Zhou, Yiqing Shen, Pietro Lio, and Yuguang \` Wang. Graph denoising diffusion for inverse protein fold-

ing. Advances in Neural Information Processing Systems, 36:10238–10257, 2023.

Rongqing Yuan, Jing Zhang, Andriy Kryshtafovych, R Dustin Schaeffer, Jian Zhou, Qian Cong, and Nick V Grishin. Casp16 protein monomer structure prediction assessment. Proteins: Structure, Function, and Bioinformatics, 2025.

Yang Zhang and Jeffrey Skolnick. Tm-align: a protein structure alignment algorithm based on the tm-score. Nucleic acids research, 33(7):2302–2309, 2005.

Zaixiang Zheng, Yifan Deng, Dongyu Xue, Yi Zhou, Fei Ye, and Quanquan Gu. Structure-informed language models are protein designers. In International conference on machine learning, pages 42317–42338. PMLR, 2023.

Xibin Zhou, Chenchen Han, Yingqi Zhang, Jin Su, Kai Zhuang, Shiyu Jiang, Zichen Yuan, Wei Zheng, Fengyuan Dai, Yuyang Zhou, et al. Decoding the molecular language of proteins with evolla. bioRxiv, pages 2025–01, 2025.

Yiheng Zhu, Jialu Wu, Qiuyi Li, Jiahuan Yan, Mingze Yin, Wei Wu, Mingyang Li, Jieping Ye, Zheng Wang, and Jian Wu. Bridge-if: Learning inverse protein folding with markov bridges. Advances in Neural Information Processing Systems, 37:39901–39922, 2024.