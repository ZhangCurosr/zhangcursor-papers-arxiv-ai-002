# RetroMPA: A Molecular Property-Aware Auxiliary Framework for Enhancing Retrosynthesis Prediction

Mianzhi Liu†1, Fan Xiao†2, Zhiliang Yu2, Huayang Huang2, Yuke Li2, Yi Yang3, Wenbo Liu4, and Yu Wu\*2

1School of Cyber Science and Engineering, Wuhan University, Wuhan 430072, China 2School of Computer Science, Wuhan University, Wuhan 430072, China 3College of Computer Science and Technology, Zhejiang University, Hangzhou 310027, China 4College of Chemistry and Molecular Sciences, Wuhan University, Wuhan 430072, China

\*Email: wuyucs@whu.edu.cn

## Abstract

Retrosynthesis is a cornerstone of drug discovery and organic synthesis. While data-driven deep learning models have shown remarkable progress, they are designed to autonomously learn reaction patterns from extensive retrosynthesis datasets with limited explicit integration of established chemical knowledge as priors. To address this limitation, we introduce RetroMPA, a molecular property-aware, post-hoc enhancement module that injects chemical knowledge into the retrosynthesis pipeline. Rather than functioning as an independent, standalone SMILES sequence generator from scratch, RetroMPA is conceptualized as a broadly applicable, model-agnostic chemical filter designed to recalibrate and optimize the predictive pathways of various existing algorithms. This plug-and-play framework can be seamlessly integrated with a range of existing data-driven retrosynthesis methods, enhancing model outputs without necessitating any modifications to the original model architecture or requiring resourceintensive, model-specific retraining procedures. By operating at the molecular level and leveraging a property-aware latent embedding space, RetroMPA consistently improves top-1 accuracy across eight representative retrosynthesis models by an average of 5.50% on USPTO-50K. Furthermore, we demonstrate its scalability by validating its performance on the large-scale USPTO-Full dataset, achieving an average improvement of about 2.03% across both template-based and template-free architectures. In addition, wet-lab experiments provide preliminary support for the practical utility of the framework. These syntheses confirmed viable, previously unreported substrate combinations for established, classic reaction paradigms—specifically, the Suzuki–Miyaura coupling, the Bucherer reaction, and the Friedel–Crafts acylation, thereby suggesting that RetroMPA can operate beyond mere data fitting. The code is open-sourced at https://github.com/MengzhouLu/RetroMPA.

## Keywords

Retrosynthesis Prediction, Molecular Property Awareness, Knowledge-Constrained Learning, Molecular Representation Learning, Chemical Semantic Space.

## Introduction

Retrosynthesis <sup>1–3</sup> is the process of deducing feasible synthetic routes for a target molecule by recursively decomposing it into simpler precursors. At the heart of this process lies single-step retrosynthesis, which identifies plausible reactants for a given product. <sup>4</sup> While decades of chemical intuition and empirical knowledge have guided this task, the explosive growth of reaction data has spurred the development of data-driven computational tools to assist or automate retrosynthetic planning. <sup>5</sup>

![](images/cfa9b913fbb82ee6df4ca47221f9f4483ef111679aa6fb8d3c28cc6751cbeda0.jpg)  
Figure 1: Standard retrosynthesis autoregressive modeling (Existing) vs our proposed RetroMPA (Ours). (a) Autoregressive applied to SMILES: sequential atom token generation from left to right, atom by atom, with limited explicit integration of chemical knowledge as priors; (b) RetroMPA generates molecular property embeddings guided by chemical knowledge, subsequently outputting complete molecules through inverse projection in chemically constrained spaces.

Current computational retrosynthesis methods commonly employ SMILES strings and molecular graphs for representation. SMILES strings <sup>6</sup> compress complex non-linear molecular structures into linear sequences. However, its syntax proves empirically dificult to learn with standard sequence models, requiring complex model designs and massive data to overcome the grammatical dependencies of linear representations, <sup>7</sup> while graphs naturally preserve structural features. <sup>8,9</sup> Molecular representation learning aims to map molecules to latent vectors that encode structural-property relationships. SMILES-based methods such as SMILES-BERT <sup>10</sup> leverage masked language modeling, while the graph-based approach MolR <sup>7</sup> preserves reaction equivalence through GNNs.

Concurrently, physics-based simulations and emerging innovations in physics- and knowledge-based approaches play a crucial role in modern computational chemistry, particularly within the domains of physics- and quantum-chemistry-informed strategies for retrosynthesis and synthesis planning. Sumiya et al. developed a quantum chemical approach for tracing reaction paths backward from a given target compound and enumerating possible reactant candidates, demonstrating that first-principles reaction-path exploration can provide mechanistically grounded hypotheses for reactant prediction. <sup>11</sup> Toniato et al. further examined the use of on-demand quantum chemical data generation to support machine-learning reaction and retrosynthesis planning, especially for low-confidence predictions where additional first-principles information may help improve decision confidence or provide data for subsequent model refinement. <sup>12</sup> In addition, Liu et al. proposed a reaction-kinetics-based retrosynthesis planning framework using a transition-state automated generation method, illustrating how kinetic feasibility can be incorporated into the evaluation of candidate synthetic pathways.<sup>13</sup> Physics-based and quantum-chemical approaches can ofer mechanistic, energetic, or kinetic information that is dificult to obtain from purely data-driven models, but they are typically computationally more demanding because they may require explicit reaction-path exploration, transition-state generation, or first-principles calculations, which can limit their direct use in large-scale global retrosynthetic pathway planning.

In recent years, deep learning has emerged as a transformative force in retrosynthesis planning, <sup>14,15</sup> with approaches broadly categorized into template-based, semi-template-based, and template-free methods. Template-based methods rely on predefined reaction rules to ensure chemical validity, but their applicability is inherently limited to predefined templates. <sup>16–19</sup> Semi-template-based methods ofer greater flexibility by identifying reaction centers via atom mapping and generating synthons, <sup>5,20,21</sup> but often depend on atommapping information unavailable in real-world prediction scenarios. <sup>4</sup> Template-free models, which treat retrosynthesis as a sequence translation task, <sup>22–26</sup> have achieved state-of-the-art performance recently by leveraging large reaction corpora.

Despite these successes, existing retrosynthesis approaches <sup>14,15</sup> concentrate on designing sophisticated architectures to learn more reaction rules primarily from reaction data statistics, which still lack suficient exploration in leveraging molecular chemical properties. Contemporary models are predominantly datadriven, learning statistical patterns of atom arrangements with less emphasis on explicit chemical reasoning mechanisms. Such models operate at the atom or token level, generating SMILES strings <sup>6</sup> autoregressively without holistic molecular understanding, leading to chemically implausible outputs (Fig. 1a). This arises from two fundamental gaps: (1) Overreliance on data-driven pattern learning without reaction-relevant molecular property knowledge as chemical priors, (2) unconstrained model outputs lacking chemical knowledge constraints.

Chemical reactions originate from molecular interactions, where molecular properties help determine reaction feasibility, pathways, and eficiency. <sup>27–29</sup> Therefore, integrating molecular-property knowledge into retrosynthesis models is a reasonable and potentially valuable direction. Chemists often infer possible reactions and reactants by analyzing relationships between the chemical properties of products and reactants. <sup>30</sup> Thus, integrating molecular property awareness into retrosynthesis models may provide more than an incremental improvement and could contribute to more chemically plausible and generalizable predictions.

However, existing retrosynthesis base models have undergone lengthy and expensive training or optimization processes. Requiring these foundation models to be restructured or retrained in order to incorporate chemical knowledge would incur substantial computational overhead. A more eficient and cost-efective approach is needed to inject knowledge into already well-performing models and improve their performance.

In this work, we introduce RetroMPA, a novel molecular property-aware auxiliary framework designed to enhance existing retrosynthesis models without architectural modification or retraining. Our approach is inspired by the chemical intuition that reactions can be understood as transformations in property space rather than mere rearrangements of atoms. RetroMPA first learns molecular chemical properties through multimodal knowledge of molecular-chemical text. Subsequently, RetroMPA utilizes diverse reactions to leverage chemica properties for retrosynthesis by learning latent correlations between product and reactants based on their properties. Departing from conventional methods, our method reduces the dependency on reaction templates or atom vocabularies. Instead, its prediction mechanism involves nearest-neighbor retrieval in a molecular embedding space, followed by inverse projection to generate reactants(Fig. 1b). Finally, the framework further applies chemically grounded constraints to refine the output of the base model. Importantly, RetroMPA itself does not rely on predefined reaction templates or reaction class labels. It imposes no requirements on the base model and is plug-and-play—requiring no modifications to the model architecture or retraining. RetroMPA is structurally positioned as a lightweight, complementary algorithmic tool rather than a direct replacement for foundational physical simulations. By systematically injecting chemical priors into the model, our framework is designed to provide a rapid, preliminary chemical constraint filter.

We first introduce the RetroMPA framework briefly, which enhances existing retrosynthesis predictors by integrating molecular property knowledge as chemical priors, enabling more chemically plausible and interpretable predictions. We then assess its impact on the standard USPTO-50K benchmark <sup>31</sup> across eight diverse base models, showing consistent improvements in top-1 accuracy without retraining. On the more complex USPTO-Full dataset, RetroMPA likewise demonstrates performance improvements, indicating its generalizability. Next, we evaluate the efectiveness and computational eficiency of the dynamic molecular dictionary under varying retrieval sizes and vocabulary scales. Through case studies, we further illustrate how RetroMPA leverages chemical semantics to guide reasoning. We further validate the model’s generalization capability through wet-lab experiments on previously unreported substrate combinations within established reaction classes, including Suzuki–Miyaura coupling, the Bucherer reaction, and Friedel–Crafts acylation.

![](images/00d216784d3e24c10cf20d7767b28a8b5ae966edc5ff532af32da9deeea38091.jpg)  
Figure 2: Overview of the three-stage RetroMPA framework. Stage 1: Mol-Former encodes molecules into molecular property embeddings $\mathbf { Q } _ { m }$ through multimodal molecule–text pretraining. Stage 2: Given the product embedding $Q _ { p }$ and previously available or predicted reactants $( r _ { 1 } , r _ { 2 } , \ldots , r _ { k } )$ , the Molecular Decoder predicts the embedding of the next reactant $r _ { k + 1 }$ . Stage 3: The predicted embedding is mapped to molecular SMILES by Inverse Projection.

Together, these evaluations provide a comprehensive view of how RetroMPA improves prediction accuracy, chemical plausibility, and model-agnostic adaptability in retrosynthesis, while also demonstrating its potential for discovering synthetically viable routes beyond training data. A more elaborate description of model architectures and rationale for certain design choices is given at the end of the paper.

## Results and discussion

## Overview of the RetroMPA Framework

RetroMPA is a knowledge-guided, plug-and-play auxiliary framework designed to enhance the chemical plausibility of single-step retrosynthesis predictions generated by any existing data-driven model. The framework operates in three synergistic stages, as illustrated in Fig. 2: Stage 1: Mol-Former encodes molecules into molecular property embeddings, trained on multimodal molecular-chemical text data. A multimoda encoder, Mol-Former, aligns molecular structures with textual chemical descriptions to generate molecular property embeddings that encode functional groups, stereochemistry, and other chemically meaningful attributes. Stage 2: Molecular-Level Prediction. A molecular-level decoder, termed Molecular Decoder, learns the transformation patterns of molecular properties from products to reactants. Conditioned on the product’s molecular property embedding, it infers the desired molecular properties of reactants and generates the embeddings of the expected reactants. This molecular-level generation avoids the combinatorial complexity of atom-by-atom decoding. Stage 3: Output molecular SMILES through Inverse Projection. Decoding the predicted reactant embeddings into concrete molecular SMILES by performing nearest-neighbor search against a dynamic molecular dictionary. The dictionary can be updated with molecules from base-mode predictions or real-world chemical catalogs, enabling adaptation to novel chemical spaces.

RetroMPA operates post hoc on the raw reactant predictions of a base model, refining them by incorporating chemical knowledge through the explicit integration of molecular property semantics as chemical priors. As a simple example, given a product P, the base model predicts a reactant pair $( b _ { 1 } , b _ { 2 } )$ . RetroMPA first encodes the product P and a reactant from $( b _ { 1 } , b _ { 2 } )$ into vectors (Stage 1), then predicts the vector of the remaining reactant (Stage 2), and finally projects the predicted vector back to a molecule (Stage 3). This process yields a complete, chemically constrained reactant set, thereby refining the base model’s prediction. The core insight is that chemical reactions are governed by the intrinsic properties of participating molecules, not merely by statistical co-occurrence in reaction datasets. By constraining predictions within a chemically meaningful embedding space, RetroMPA improves both the reliability and generalizability of retrosynthetic results. Full technical details of the framework are provided in Section 4. Here, we focus on its chemica utility and empirical impact.

## Performance Enhancement on the USPTO-50K Benchmark

To assess the compatibility and general utility of RetroMPA within established computational retrosynthesis workflows, we evaluated its impact on the widely adopted USPTO-50K benchmark.<sup>31</sup> Following prior work, we report top-1 accuracy based on exact match of canonical SMILES between prediction and ground truth, without using atom mappings or reaction class labels, reflecting real-world deployment conditions.

RetroMPA was integrated post-hoc with eight representative base models spanning major retrosynthesis paradigms: template-based (LocalRetro <sup>32</sup>), semi-template-based (GraphRetro <sup>5</sup>) and template-free (Transformer, <sup>33</sup> Retroformer, <sup>25</sup> R-SMILES, <sup>24</sup> NAG2G, <sup>20</sup> EditRetro, <sup>22</sup> RetroWise <sup>34</sup>). Crucially, no retraining, fine-tuning, or architectural modification of the base models was performed; RetroMPA operated in a plugand-play fashion using only the initial prediction of the base model and the structure of the product as input. To ensure fair comparison, all baseline results are re-evaluated using publicly released model weights.

As summarized in Table 1, RetroMPA consistently improved top-1 accuracy across all eight models, with average gains of 5.50% without retraining or fine-tuning the base models. The gains spanned all methodological categories, demonstrating the framework’s paradigm-agnostic applicability. The Transformer model, which lacks explicit chemical inductive biases, exhibited the highest absolute improvement (+8.19%). This suggests that RetroMPA efectively compensates for the absence of chemical priors in purely data-driven sequence-to-sequence architectures by injecting property-aware constraints. Both LocalRetro and GraphRetro, which rely on reaction templates or centers, showed improvements exceeding 5%. This indicates that RetroMPA complements rule-based approaches by enhancing their ability to handle chemically plausible but template-ambiguous cases. Notably, even models already achieving high baseline performance, such as EditRetro (60.34% → 64.75%) and RetroWise (64.71% → 68.30%)—benefited from RetroMPA’s chemica priors. These improvements are particularly meaningful given the already high baseline performance and competitive benchmark setting.

These results collectively demonstrate that injecting molecular property knowledge as an auxiliary constraint can consistently enhance retrosynthesis prediction across diverse methodological paradigms. The fact that RetroMPA operates in a plug-and-play fashion further underscores its practical utility: existing deployed models can be seamlessly upgraded without costly retraining or architectural overhaul. It does not rely on specific internal representations or training objectives of the base model; instead, it leverages only the molecular property semantics embedded in the reactant–product pair. This universality makes RetroMPA a practical and low-overhead enhancement for existing retrosynthesis pipelines, particularly in industrial settings where model retraining is costly or infeasible.

In summary, while the USPTO-50K benchmark serves primarily as a compatibility testbed, the consistent performance gains observed across heterogeneous models suggest that integrating molecular property knowledge as a chemical prior can improve the reliability and chemical plausibility of retrosynthetic predictions.

## Performance Enhancement on the USPTO-Full Benchmark

While performance on highly curated benchmarks like USPTO-50K confirms basic algorithmic validity, a rigorous evaluation of the framework’s broad applicability necessitates stress-testing against larger-scale datasets. To this end, we expanded our benchmarking protocols to encompass the diverse USPTO-Ful dataset. Due to computational constraints and the restricted availability of open-source weights for certain base models on USPTO-Full, our scalability evaluation focused strictly on two representative paradigms: LocalRetro (template-based graph models) and EditRetro (template-free string manipulation mechanisms).

Table 1: Top-1 accuracy on USPTO-50K dataset. $\Delta$ indicates the improvement after using RetroMPA. Results of our method are averaged over 3 random seeds, reported as mean ± standard deviation
<table><tr><td>Method</td><td>Original Acc.</td><td>+Ours</td><td> $\Delta$ </td></tr><tr><td> $T e m p l a t e - b a s e d$   $\mathrm { L o c a l R e t r o ^ { 3 2 } }$ </td><td>52.59</td><td> $5 7 . 7 6 \pm 0 . 0 6$ </td><td> ${ \bf 5 . 1 7 \pm 0 . 0 6 }$ </td></tr><tr><td> $S e m i - t e m p l a t e - b a s e d$ </td><td></td><td></td><td> ${ \bf 5 . 8 2 \pm 0 . 1 2 }$ </td></tr><tr><td> $\mathrm { G r a p h R e t r o } ^ { 5 }$ </td><td>53.72</td><td> $5 9 . 5 4 \pm 0 . 1 2$ </td><td></td></tr><tr><td> $T e m p l a t e - f r e e$ </td><td>42.80</td><td></td><td></td></tr><tr><td> $\mathrm { T r a n s f o r m e r ^ { 3 3 } }$   $\mathrm { R e t r o f o r m e r } ^ { 2 5 }$ </td><td></td><td> $5 0 . 9 9 \pm 0 . 1 6$ </td><td> ${ \bf 8 . 1 9 \pm 0 . 1 6 }$ </td></tr><tr><td> $\mathrm { R } { - } \mathrm { S M I L E S } ^ { 2 4 }$ </td><td>52.73</td><td> $5 8 . 6 8 \pm 0 . 0 8$ </td><td> ${ \bf 5 . 9 5 \pm 0 . 0 8 }$ </td></tr><tr><td> $\mathrm { N A G 2 G ^ { 2 0 } }$ </td><td>53.07 54.66</td><td> $5 8 . 4 0 \pm 0 . 1 0$   $6 0 . 2 0 \pm 0 . 1 4$ </td><td> ${ \bf 5 . 3 3 \pm 0 . 1 0 }$   ${ \bf 5 . 5 4 } \pm 0 . 1 4$ </td></tr><tr><td> $\mathrm { E d i t R e t r o ^ { 2 2 } }$ </td><td>60.34</td><td> $6 4 . 7 5 \pm 0 . 0 8$ </td><td> ${ \bf 4 . 4 1 \pm 0 . 0 8 }$ </td></tr><tr><td> $\mathrm { R e t r o W i s e ^ { 3 4 } }$ </td><td>64.71</td><td> $6 8 . 3 0 \pm 0 . 0 6$ </td><td> ${ \bf 3 . 5 9 \pm 0 . 0 6 }$ </td></tr></table>

Table 2: Top-1 accuracy on USPTO-Full dataset. $\Delta$ indicates the improvement after using RetroMPA. Results of our method are averaged over 3 random seeds, reported as mean ± standard deviation.
<table><tr><td>Method</td><td>Original Acc.</td><td>+Ours</td><td> $\Delta$ </td></tr><tr><td>Template-based  $\mathrm { L o c a l R e t r o ^ { 3 2 } }$ </td><td>40.12</td><td> $4 2 . 9 2 \pm 0 . 0 4$ </td><td> $\mathbf { 2 . 8 0 \pm 0 . 0 4 }$ </td></tr><tr><td> $T e m p l a t e - f r e e$ </td><td></td><td></td><td></td></tr><tr><td> $\mathrm { E d i t R e t r o ^ { 2 2 } }$ </td><td>51.10</td><td> $5 2 . 3 6 \pm 0 . 0 3$ </td><td> ${ \bf 1 . 2 6 \pm 0 . 0 3 }$ </td></tr></table>

The empirical results summarized in Table 2 suggest that RetroMPA remains efective under a larger and more diverse data setting. The application of our post-hoc auxiliary module yielded consistent improvements across the two evaluated paradigms, delivering a 2.80% absolute gain to the template-based LocalRetro and a 1.26% improvement to the highly optimized EditRetro. This observation corroborates that RetroMPA operates efectively as a scalable, universal inference layer, preserving its capacity for property-aware chemica reasoning under the evaluated USPTO-Full setting.

## Relationship between Base Model Performance and RetroMPA Improvement

We further analyzed the correlation between the intrinsic performance of base models and the magnitude of improvement delivered by RetroMPA. The analysis suggests a compensatory trend between baseline performance and the magnitude of RetroMPA-induced improvement. The improvement brought by RetroMPA is related to the performance of the base model. As the base model attains higher accuracy, fewer incorrect predictions remain available for correction, and accordingly, the improvement conferred by RetroMPA generally diminishes.

The overall trend observed in benchmark evaluations (see Figure 3 and Figure 4) indicates that the weaker the base model, the greater the absolute improvement conferred by RetroMPA. When applied to the weakest baseline models (e.g., a sequence-to-sequence Transformer lacking chemical knowledge), RetroMPA yields the largest absolute improvements. Conversely, as the base model’s inherent capability strengthens and approaches advanced performance (e.g., EditRetro), the absolute performance gain provided by RetroMPA diminishes. Nevertheless, this improvement remains appreciable in magnitude. This suggests that RetroMPA serves as a corrective aid for fundamental errors in weaker models, while functioning as a chemistry-informed constraint filter for advanced models.

![](images/bcb4f3a192f3702b84b8ddf8ae79aec3871b15b46ec768b4237253a491486c3a.jpg)  
Figure 3: Performance enhancement achieved by RetroMPA across various base models on USPTO-50K. The gray bars indicate the original top-1 accuracy of the base models (left y-axis), arranged in ascending order of their inherent capabilities. The red line illustrates the absolute accuracy improvement (∆) conferred by RetroMPA (right y-axis).

This trend was also observed on the more complex and chemically diverse USPTO-Full dataset. The substantially increased prediction dificulty reduced both the performance of the base models and the corresponding improvement magnitude; however, this further underscores the appreciable value of the gains brought by RetroMPA. It retained the capacity to capture and exploit molecular-property interactions in retrosynthesis within complex reaction relationships, without compromising compatibility with existing base models.

## Efectiveness and Eficiency of RetroMPA

RetroMPA ultimately retrieves molecules from a molecular dictionary and outputs reactants, thereby bridging the continuous embedding space generated by the molecular decoder to discrete, chemically valid molecular structures. Therefore, it is essential to further investigate the generation efectiveness and retrieval eficiency of RetroMPA beyond its performance on benchmark datasets.

To assess practical applicability, we report the computational overhead and latency introduced by RetroMPA. The universal pretraining phase of the Mol-Former and the optimization of the Molecular Decoder are resource-intensive. Leveraging a high-performance computing cluster equipped with four NVIDIA RTX 4090 GPUs and driven by dual Intel Xeon Platinum 8352V CPUs, the comprehensive network training demanded approximately 130 hours. However, it should be emphasized that this training regimen represents a one-time amortized sunk cost. Because RetroMPA learns the universal relationships and patterns of chemical property changes between products and reactants, this singular pretrained artifact can be subsequently attached to any base model without requiring a single epoch of fine-tuning or secondary backpropagation. Crucially, the inference overhead introduced during active prediction is small. RetroMPA requires approximately 0.014 seconds, or 14 ms, per single inference. For context, reported single-sample inference latencies for existing retrosynthesis models are 177.4 ms for EditRetro and 292.1 ms for R-SMILES, excluding model loading time. Therefore, the additional per-inference overhead introduced by RetroMPA is substantially smaller than these reported inference times, approximately 12.7× lower than EditRetro and 20.9× lower than R-SMILES. Considering that RetroMPA is a plug-and-play framework that delivers improvements across the tested base models, these results suggest its potential practical utility for industrial applications.

![](images/5fc82f4cbc8300567d098bf1f47dc809da694ab9471e9a00851e28050feae989.jpg)  
Figure 4: Correlation analysis between the performance of base models and the magnitude of improvement delivered by RetroMPA on USPTO-50K. The negative Pearson correlation $( r = - 0 . 9 6 )$ shows a strong negative association.

An important requirement for retrosynthesis tools intended for real-world synthetic planning is their ability to generalize beyond known reaction databases and propose viable routes for novel molecules. To this end, RetroMPA employs a dynamic molecular dictionary—a key component that enables adaptation to unseen chemical space by incrementally incorporating candidate molecules (e.g., from base model predictions or vendor catalogs). The strength of this strategy was supported by preliminary wet-lab validation: guided by its dynamic dictionary, RetroMPA predicted viable, previously unreported substrate combinations for three classic reaction paradigms—a Suzuki–Miyaura coupling, a Bucherer reaction, and a Friedel–Crafts acylation, as illustrated in Fig. 5. In every case, none of the involved reactants or products existed in the initial dictionary, and the novelty lies exclusively in the predicted reactant pairings. Each reaction was subsequently executed in the laboratory under standard conditions, and the products were confirmed by <sup>1</sup>H and <sup>13</sup>C NMR spectroscopy; this analysis is detailed in Section S2, Fig. S2 and Fig. S3 of the Supporting Information. These experiments provide preliminary experimental support that RetroMPA can propose synthetically viable substrate pairings beyond the initial dictionary.

From a computational standpoint, our analysis further shows that the retrieval process is both efective and eficient. Using a nearest-neighbor search, the dictionary provides additional chemical constraints to guide accurate predictions, with diminishing returns observed for larger K values, as shown in Table 3. The top-K accuracy aligns with standard retrieval-based evaluation in reaction prediction, <sup>35</sup> where a prediction is considered correct if any of the top-K retrieved candidates matches the ground truth, rather than post-retrieval ranking. From these results, we first observe that incorporating a single retrieval $( K = 1 )$ improves accuracy from 60.34% to 64.75%. When $K \geq 5$ , accuracy further increases to approximately 65.89%, with only minor additional gains observed beyond this threshold. We hypothesize this occurs because suficient constraint information is already captured at K = 5 to guide the base model’s predictions, while molecules farther from the query provide diminishing marginal utility due to their weaker chemical relevance to the target reaction.

In addition to validating its efectiveness, we also conducted a systematic investigation into the computational eficiency of querying the dynamic molecular dictionary. As shown in Table 4, we measured the k-NN search time across varying molecular dictionary sizes on an RTX 4090 GPU, using top-K = 50, and a batch size of 32. k-NN search takes 15.01 ms even with a 200K-molecule dictionary. For large-scale deployment, we recommend using the Milvus database for storing and searching molecular vectors. Milvus is a highly optimized vector database system, and benchmarks show search latencies below 0.3 s even for 50 M vectors. <sup>36</sup>

Table 3: Study on the number of neighbors.  
Table 4: Computational eficiency.
<table><tr><td>Neighbors</td><td>1</td><td>3</td><td>5</td><td>10</td><td>50</td></tr><tr><td>Accuracy</td><td>64.75</td><td>65.79</td><td>65.89</td><td>66.01</td><td>66.17</td></tr></table>

<table><tr><td>Molecular dictionary size</td><td>10,000</td><td>100,000</td><td>200,000</td></tr><tr><td>Latency (ms)</td><td>0.51</td><td>8.38</td><td>15.01</td></tr></table>

Table 5: Top-1 accuracy of EditRetro and RetroMPA with diferent dictionary scale multipliers, where additional molecules are sampled from PubChem (first row) and ChEBI-20 (second row).
<table><tr><td>Source</td><td>EditRetro (base model)</td><td>1.0x</td><td>2.0x</td><td>4.0x</td><td>8.0x</td></tr><tr><td>PubChem</td><td>60.34</td><td>64.75</td><td>64.49</td><td>64.07</td><td>63.61</td></tr><tr><td>ChEBI-20</td><td>60.34</td><td>64.75</td><td>64.42</td><td>63.93</td><td>63.41</td></tr></table>

As shown in Table 5, we further examined how dictionary scale and composition afect RetroMPA by expanding the original molecular dictionary with randomly sampled decoy molecules from PubChem or ChEBI-20. With PubChem decoys, Top-1 accuracy changed from 64.75% at 1.0× to 64.49%, 64.07%, and 63.61% at 2.0×, 4.0×, and 8.0×, respectively. With ChEBI-20 decoys, the corresponding accuracies were 64.42%, 63.93%, and 63.41%. These results suggest that performance is afected more by retrieval-space density than by the specific source of added molecules, while RetroMPA still outperforms the EditRetro baseline under all tested expansion settings.

The combination of preliminary wet-lab validation of previously unreported substrate combinations and computationally eficient constrained retrieval establishes RetroMPA as a useful auxiliary component for practical AI-assisted retrosynthesis. It may help make the model’s outputs more chemically grounded while maintaining computational eficiency.

## Case Study: Property-Aware Reasoning for Chemically Interpretable Predictions

To further elucidate the specific types of molecular properties captured by our framework and to understand their potential advantages, we conducted an in-depth chemical analysis of representative cases, as shown in Fig. 6. These qualitative evaluations aim to demonstrate how the molecular property embeddings may guide context-selective chemical reasoning.

Case 1: Distinguishing Functional Groups and Steric Properties

• Product: Cc1c2n(c3ccccc13)CCCC2N(C)C=O

• Base Prediction: CC(=O)OC(C)=O and CNC1CCCn2c1c(C)c1ccccc12

• RetroMPA Refined: O=CO and CNC1CCCn2c1c(C)c1ccccc12

Analysis: The target product features a specific formamide moiety (-N(C)C=O). The baseline model proposed acetic anhydride (CC(=O)OC(C)=O), an acetylating reagent, which would more plausibly introduce an acetyl group rather than the desired formyl group. From a mechanistic standpoint, this prediction is suboptimal, as it would likely lead to the formation of an acetamide group (-N(C)C(=O)CH3), thereby deviating from the intended target structure. In contrast, RetroMPA refined this prediction to formic acid (O=CO). This adjustment suggests that the incorporation of molecular property embeddings may enhance the model’s sensitivity to subtle steric variations and specific functional group definitions (e.g., formyl versus acetyl). Consequently, this property-aware constraint can help mitigate fundamental structural inaccuracies that might be overlooked by purely statistical sequence matching approaches.

## Case 2: Recognizing Reaction Selectivity and Strategic Disconnections

• Product: C[C@H]1CCCN1C[C@H]1C[C@@H]1c1ccc(Br)cc1

• Base Prediction: C[C@H]1CCCN1 and Cc1ccc(S(=O)(=O)OC[C@H]2C[C@@H]2c2ccc(Br)cc2)cc1

![](images/cb67e51d79aa5b3ee10346275296c08c89bf17bb4d84dcdb56809b5529f1acfe.jpg)  
Figure 5: Experimentally Validated Previously Unreported Substrate Combinations Predicted by RetroMPA Using Dynamic Molecular Dictionary. RetroMPA, leveraging its dynamic molecular dictionary, predicted three unreported substrate combinations, including Suzuki–Miyaura coupling, Bucherer reaction, and Friedel–Crafts acylation.

## • RetroMPA Refined: C[C@H]1CCCN1 and O=C[C@H]1C[C@@H]1c1ccc(Br)cc1

Analysis: Case 2 involves the construction of a sterically hindered C-N bond. The baseline model predicted a direct alkylation pathway employing a bulky tosylate as the electrophile. In practical organic synthesis, direct S 2 alkylation of amines under such sterically encumbered conditions may sufer from over-alkylation or competing elimination. RetroMPA, however, adjusted the prediction toward an aldehyde precursor, efectively redirecting the retrosynthetic strategy to a reductive amination pathway. This refinement indicates that RetroMPA can capture broader chemical contexts regarding reaction selectivity, potentially assisting the model in prioritizing robust and milder synthetic routes for amine formation in complex molecular environments.

## Case 3: Diferentiating Organometallic Reactivity Profiles

• Product: C=C(C)Cc1cnc(N)cn1

• Base Prediction: C=C(C)C[Sn](CCCC)(CCCC)CCCC and Nc1cnc(Br)cn1

## • RetroMPA Refined: C=C(C)CB1OC(C)(C)C(C)(C)O1 and Nc1cnc(Br)cn1

Analysis: In Case 3, the baseline model suggested a Stille coupling pathway dependent on an organotin reagent. While theoretically plausible on paper, organotin compounds are often avoided in modern synthetic scaling due to their well-known toxicity and challenging purification profiles. RetroMPA modified this prediction to utilize a pinacol boronate ester, thereby shifting the proposed pathway to a Suzuki–Miyaura coupling. We infer that the multimodal pretraining might equip RetroMPA with an enhanced capacity to diferentiate elemental reactivity profiles, efectively reflecting a practical preference for employing more commonly used and generally lower-toxicity organoboron reagents.

We further conducted additional case studies on USPTO-50K by randomly selecting two reactions involving two reactants for intuitive analysis to better understand whether the model can correctly utilize molecular properties in chemical reactions, with results shown in Fig. 7. RetroMPA retrieves three candidate nearestneighbor molecules for the second reactant through inverse projection. The first example (top panel of Fig. 7) demonstrates the synthesis of tert-butyl 5-acetylindole-1-carboxylate. RetroMPA retrieves the ground-truth reagent as Neighbor 1, corresponding to Boc protection of the indole N–H functionality through nucleophilic acyl substitution. The blue highlights indicate that other neighbors share the same carbonate ester functiona group with ground truth, which could also serve as reactive centers for nucleophilic acyl substitution. The product exhibits a clear N-Boc derivative structure. This demonstrates that RetroMPA identifies the essential functional group pattern required for the reaction. Even if the exact reagent is not retrieved, the candidate set remains functionally coherent, allowing a chemist to readily infer the intended transformation. The second example (bottom panel of Fig. 7) illustrates a radical bromination reaction (halogenation reaction) where the product 5-(bromomethyl)-2,4-dichloropyridine retains the structural features of the second reactant – a characteristic of this reaction type. Similarly, the retrieved neighbors share the key heteroaryl with the ground-truth precursor, indicating that RetroMPA retrieves candidates with related reaction-relevant substructures.

These examples suggest that RetroMPA can capture reaction-relevant molecular features that are consistent with selected organic reaction principles. By operating in a molecular property space, the model may generate outputs that are more accurate in the evaluated cases and can provide interpretable chemical cues. A particularly notable observation is that some retrieved candidates that do not exactly match the ground truth still share reaction-relevant functional groups or substructures and may provide useful chemical clues. They belong to the same functional group class as the true reactant and could, in principle, participate in analogous reactions under appropriate conditions. This property-rich output provides valuable intermediate reasoning clues. Instead of presenting a single black-box prediction, RetroMPA ofers a shortlist of chemically plausible options, allowing chemists to infer potential reaction types by analyzing the common property patterns among the candidates.

![](images/23fed662b06896f6d7fb391d0dc0d9714080a847ce106c3c03ef8656638aec5e.jpg)  
Figure 6: Illustrative examples of RetroMPA refinements driven by molecular property awareness.

## Embedding visualization

Further embedding visualization provides deeper insight into how RetroMPA leverages molecular property space. Fig. 8 shows the embedding visualization for the first case study. The predicted vector for the second reactant in the Boc protection example resides within the carbonate ester functional-group cluster and is oriented closer to the hydrocarbon region in the embedding space. This geometric alignment indicates that RetroMPA successfully captures the alkane substructure present in the product molecule and correctly attributes it to the second reactant, demonstrating its ability to perform chemically grounded reasoning through property-aware representation learning.

Additionally, we illustrate the importance of molecular properties in chemical reactions through ketone reduction and alcohol reduction examples. Their reaction templates are $\mathrm { R } _ { 1 } { - } \mathrm { C O } { - } \mathrm { R } _ { 2 } + \mathrm { H } _ { 2 } \ \longrightarrow$ $\mathrm { R } _ { 1 } { \mathrm { - C H ( O H ) } }$ $\mathrm { \cdot R _ { 2 } }$ and R –CH(OH)– $\mathrm { \cdot R _ { 2 } }$ Reduction → R –CH – $\mathrm { \cdot R _ { 2 } }$ respectively. We first generate molecular property embeddings for propan-2-one $\left( \mathrm { C H _ { 3 } C O C H _ { 3 } } \right)$ and 2,3-butanedione $\mathrm { ( C H _ { 3 } C O C O C H _ { 3 } ) }$ using the pretrained Mol-Former model, along with corresponding alcohols and hydrocarbons. These vectors are then visualized using t-SNE,<sup>37</sup> as shown in Fig. 9. The results are consistent with: (1) Molecules with similar SMILES but diferent properties form distinct clusters in Mol-Former’s chemical space. This provides embedding-level molecular property priors during retrosynthesis prediction, enabling chemically grounded reasoning. (2) Homologous reactions exhibit analogous directional patterns in the projected low-dimensional property space. This facilitates learning common characteristics and inter-reaction correlations. The model appears to capture functional group quantity efects, as evidenced by the red dashed arrow (representing dual ketone group reduction in 2,3-butanedione) being approximately twice as long as the blue arrow (single ketone reduction). These geometric regularities in chemical space enable more interpretable and chemically plausible prediction compared to traditional black-box approaches.

![](images/ed09dc6f6a9edd2419f19e30723c3cd150dbd1b4045ae6e90ca10b59b034c803.jpg)  
Figure 7: Case study of retrieved molecules. The columns show the product, the first reactant, the top three nearest-neighbor candidates for the second reactant, and the ground-truth second reactant. Shared substructures are highlighted in blue.

![](images/00ad73a3f4f38a7734b132f9b21a752daee3193327c45d7437769c6268b3187f.jpg)  
Figure 8: Embedding visualization of predicted vector by RetroMPA.

![](images/d09a2ef9d3b9072597e373f8f72b471353f7e5baa9aad8b88c9edc76662b1e92.jpg)  
Figure 9: Reaction visualization of ketone reduction and alcohol reduction.

## Conclusion

In this study, we presented RetroMPA, a knowledge-driven auxiliary framework for retrosynthesis prediction that systematically injects molecular chemical property knowledge to enhance the reliability and performance of existing data-driven models. Unlike conventional approaches that treat retrosynthesis as a purely statistical sequence generation task, RetroMPA operates at the molecular level, leveraging a chemically grounded embedding space to constrain and refine predictions. It enables the model to approximate aspects of propertybased chemical reasoning used by chemists: not by relying solely on atom-level co-occurrence patterns, but by analyzing how intrinsic molecular properties influence feasible transformations.

Our framework is designed as a plug-and-play module that requires no retraining or architectura modification of existing base models. This modular design enables seamless post-hoc integration with diverse retrosynthesis architectures without retraining the base models, as demonstrated across eight representative methods spanning template-based, semi-template-based, and template-free paradigms.

Our results demonstrate consistent and substantial improvements in top-1 accuracy on the USPTO-50K benchmark, with an average gain of 5.50% without retraining any base model. Even on the more complex and diverse USPTO-Full dataset, the average improvement reaches approximately 2%, highlighting the framework’s ability to leverage chemical knowledge for stronger performance. Crucially, wet-lab validation provided preliminary evidence that RetroMPA can propose synthetically viable, previously unreported substrate combinations within established reaction classes, including Suzuki–Miyaura coupling, the Bucherer reaction, and Friedel–Crafts acylation. Case studies further revealed that RetroMPA provides chemically interpretable refinements by retrieving candidates based on shared property patterns, thereby enhancing both prediction accuracy and interpretability.

Despite these advances, several limitations warrant consideration. RetroMPA is designed as an auxiliary model that requires at least one predicted reactant to initiate its inference process, thus it does not directly apply to single-reactant reactions. The inverse projection from continuous embeddings to discrete SMILES via nearest-neighbor retrieval, while efective, does not guarantee molecular uniqueness and may occasionally retrieve structurally similar but not identical molecules. Furthermore, the quality of the molecular property embeddings is inherently tied to the breadth and depth of the multimodal pre-training data.

Looking forward, this work establishes a modular and extensible foundation for knowledge-augmented retrosynthesis. The plug-and-play nature of RetroMPA makes it readily adaptable to new base models and chemical databases without architectural overhaul. Several promising directions emerge for future research: (i) iterative application to multi-step planning, where RetroMPA could refine reaction nodes within retrosynthesis search trees; (ii) resolving the molecular uniqueness problem through advanced generative decoding or learned inverse mappings; (iii) integrating reaction condition prediction to provide fully actionable synthetic routes.

Ultimately, by bridging data-driven machine learning with established chemical principles, RetroMPA represents a step toward more reliable, interpretable, and generalizable AI-assisted synthesis planning. We hope this work encourages further integration of domain knowledge as explicit priors in computational chemistry, fostering the development of robust tools that accelerate discovery in organic synthesis and drug development.

## Method

## Dataset Curation and Preprocessing

To pre-train the Mol-Former component of RetroMPA for learning molecular chemical properties and thereby guiding retrosynthesis tasks, we leveraged three large-scale multimodal chemical-text datasets: (i) ChEBI-20-MM:<sup>38</sup> contains 33,000 molecule-description pairs with rich functional-group and stereochemical annotations. (ii) Mol-Instructions:<sup>39</sup> a biomolecular instruction dataset from which we selected 734,000 chemically relevant entries. (iii) PubChemSTM: <sup>40</sup> comprises over 280,000 molecule–text pairs describing broader chemical properties and roles (see Fig. S1).

All benchmark evaluations in this work are conducted on the USPTO-50K dataset<sup>31</sup> and USPTO-Full dataset.<sup>41</sup> USPTO-50K is a widely adopted standard comprising 50,016 atom-mapped single-step reactions extracted from United States patent literature. The USPTO-Full dataset, comprising approximately one million reactions, is substantially larger and more diverse, allowing us to validate our model’s performance beyond standard benchmarks. Experiments on both datasets enable a comprehensive assessment of our method’s performance, generalization ability, and scalability, providing valuable insights for practical retrosynthesis applications. Each reaction is originally represented in SMILES format as reactants > reagents > products. To focus on the core structural transformation and align with common practice in recent retrosynthesis literature, we remove reagent information entirely. Reagents—such as catalysts, solvents, or bases—are generally not intended to contribute atoms to the final product and are often inconsistently annotated across patents or preprocessing pipelines, which introduces ambiguity in role assignment and potential noise during training. <sup>42</sup> By discarding reagents, our input format simplifies to reactants → product, where all reactants are concatenated using the canonical dot (’.’) delimiter. Considering the noise introduced during dataset construction and the lack of reaction condition descriptions, we retained only reactions with two reactants.

To ensure data integrity and prevent unintended information leakage, we canonicalize all SMILES strings using $\mathrm { R D K i t ^ { 4 3 } }$ and then remove atom mappings and reaction class labels prior to model training or evaluation to avoid providing the model with explicit atom-correspondence information unavailable in real-world prediction scenarios. Notably, the molecular dictionary itself is a purely passive database utilized exclusively during the inference phase (Stage 3). It is not used for training, gradient computation, representation learning, model selection, or hyperparameter tuning, and therefore does not alter the learned model or the generated query representation. Importantly, the dictionary is used only after RetroMPA has generated the query embedding. For benchmark evaluation, the inverse-projection step is evaluated under a closed-candidate retrieval protocol. The molecular dictionary is constructed as a fixed candidate gallery from molecules in the evaluation candidate pool, including the ground-truth precursor molecules. This setting is necessary for retrieval-style evaluation because the correct molecule must be present in the candidate gallery to make Hits@k, Recall@k, MRR, or exact-match retrieval accuracy meaningful. This is standard mathematical practice <sup>38,44,45</sup> in retrieval-augmented generation architectures and does not constitute leakage.

All SMILES strings used in both pretraining and downstream evaluation are processed using standard cheminformatics tools. No SMILES augmentation or standardization beyond canonicalization is applied, and no atom-mapping-based filtering is used, thereby preserving more of the native chemical diversity of the source data. This curation strategy is intended to approximate realistic prediction conditions by avoiding explicit privileged signals, while allowing the model to benefit from large-scale chemical knowledge during pretraining.

## Methodological Foundation

We propose RetroMPA, consisting of a Mol-Former and a Molecular Decoder, which leverages the fundamental principle that chemical reactions inherently depend on molecular properties. The core idea of RetroMPA is to constrain the retrosynthesis process using the Molecular Decoder through chemically-informed embeddings from Mol-Former. Mol-Former maps molecules into vectors with explicit molecular-property semantics; the Molecular Decoder then predicts reactant embeddings based on these vectors, and the corresponding molecules are obtained through Inverse Projection, preserving molecular knowledge priors throughout the retrosynthesis workflow. During the inference stage, we take the product and one reactant predicted by the base model as the starting input, and predict the remaining reactant via RetroMPA, thereby obtaining a complete reaction pathway constrained by chemical knowledge. To formalize this auxiliary role, we define the auxiliary retrosynthesis task.

Auxiliary Retrosynthesis. We formally define the auxiliary retrosynthesis task as follows. Whereas standard retrosynthesis infers all potential reactants from a given product, auxiliary retrosynthesis predicts the remaining reactants using both the product and one reactant predicted by the base model, thereby refining the base model’s output. Given product P and the first predicted reactant $R _ { 1 }$ from a base model, the auxiliary task aims to predict remaining reactants $\{ R _ { 2 } , . . . , R _ { n } \}$ through eq 1:

$$
\{ R _ { 2 } , . . . , R _ { n } \} = \underset { \mathcal { R } } { \arg \operatorname* { m a x } } \Phi ( P , R _ { 1 } , \mathcal { R } | \Theta ) ,\tag{1}
$$

where $\Phi ( \cdot )$ denotes the model parameterized by Θ, R denotes the candidate solution space for retrosynthesis predictions.

## Mol-Former: Cross-Modal Molecular Chemical Knowledge Learning

Injecting Chemical Knowledge via Multimodal Pretraining. RetroMPA leverages multimodal pretraining for enhanced robustness. Unlike SMILES-only methods, our framework is pretrained on largescale chemical-text datasets, equipping it with broader chemical intuition to recognize and appropriately respond to phenomena like functional-group identity and reagent-relevant molecular features.

We introduce Mol-Former, a multimodal encoder that aligns molecular and textual representations through a shared embedding space. It comprises a Molecular Transformer, a Text Transformer, and a Molecular Encoder. The architecture follows the Q-Former design <sup>46</sup>, where a learnable Molecular Transformer attends to facilitate interaction between molecular representations and textual chemical knowledge, ultimatel transforming the query tokens into molecular property embeddings $\mathbf { Q } _ { m }$ with explicit chemical semantics. Formally, given a SMILES string and its associated text description, we first encode them into latent sequences using pretrained molecular encoder $\mathrm { M o K } ^ { 7 }$ and a Text Transformer. A learnable query then attends to both modalities via cross-attention, yielding a joint embedding space. To align molecular structures with their semantic descriptions, we train Mol-Former using all three original pre-training objectives (i.e., image-text contrastive, image-text matching, and language modeling loss), with the original image modality simply replaced by molecules, yielding the corresponding molecule–text contrastive (MTC), molecule–text matching (MTM), and language modeling (LM) loss terms. Specifically, we optimize eq 2:

$$
\mathcal { L } _ { \mathrm { M o l - F o r m e r } } = \mathcal { L } _ { \mathrm { M T C } } + \mathcal { L } _ { \mathrm { M T M } } + \mathcal { L } _ { \mathrm { L M } } .\tag{2}
$$

where $\mathcal { L } _ { \mathrm { M T C } }$ is a molecule–text contrastive loss encouraging similar embeddings for matched pairs, $\mathcal { L } _ { \mathrm { M T M } }$ promotes accurate identification of positive pairs, and $\mathcal { L } _ { \mathrm { L M } }$ enables generation of chemically meaningful descriptions conditioned on the molecule.

This cross-modal alignment objective encourages the model to ground textual semantics, such as “carboxylic acid”, “R-enantiomer”, or “Ketone”, into the structure-derived molecular representation, equipping it with broader chemical intuition to recognize phenomena like functional-group identity and reagent-relevant molecular features. Ultimately Mol-Former transforms the query tokens into molecular property embeddings $\mathbf { Q } _ { m }$ with explicit chemical semantics, as shown in eq 3:

$$
\mathbf { Q } _ { m } = \mathrm { M o l - F o r m e r } ( \mathbf { m } ) ,\tag{3}
$$

where m denotes molecular embeddings generated by molecular encoder. This integration of chemical knowledge into $\mathbf { Q } _ { m }$ enables molecular chemical properties to be used as prior information for retrosynthesis prediction.

## Molecular Decoder: Molecular-Level Retrosynthesis Prediction

Molecular-Level Prediction. To better capture the compositional nature of chemical reactions, we formulate retrosynthesis as a sequence generation task at the molecular level, where each step generates a complete reactant molecule rather than individual atoms, promoting higher-level chemical reasoning. We define the solution retrosynthesis prediction space R as a sequence of molecular mappings $( m _ { 1 } , \ldots , m _ { K } )$ , each representing a reactant in the context of the product. Given the product embedding $\mathbf { Q } _ { p r o }$ from Mol-Former and previous predictions, the Molecular Decoder models the conditional likelihood, as given in eq 4:

$$
P ( \mathcal { R } ) = \prod _ { k = 1 } ^ { K } P ( m _ { k } \mid m _ { i < k } , \mathbf { Q } _ { p r o } ) ,\tag{4}
$$

where $m _ { k }$ denotes the k-th reactant. During training, we ground predictions on ground-truth reactants from standard datasets to avoid potential bias or performance limitations that might arise from relying on any base model’s predictions, which enables universal compatibility across diverse predictors. We employ teacher forcing<sup>47</sup>, conditioning each step on true preceding reactants.

Learning Discriminative Reactant Representations via Dual-Prior Contrastive Learning.

To ensure the Molecular Decoder generates both accurate and chemically meaningful reactants, we design a contrastive learning framework. Let $\hat { \mathbf { r } } _ { i } ^ { ( k ) }$ denote the predicted embedding of the k-th reactant for sample i, and let $\mathbf { m } _ { i } ^ { ( k ) }$ denote the corresponding ground-truth molecular embedding encoded by the molecular encoder. The Decoder is optimized by combining Molecular Contrastive Loss (MCL) and Molecular Matching Loss (MML).

• Molecular Matching Loss (MML): First, a Molecular Matching Loss (MML) attempts to minimize the L2 distance between predicted reactant embeddings r and their targets m. The positive matching term given in eq 5 minimizes the L2 distance between each predicted reactant embedding and its corresponding ground-truth embedding:

$$
\mathcal { L } _ { \mathrm { p o s } } = \lambda _ { 1 } \sum _ { i = 1 } ^ { B } \left. \hat { \mathbf { r } } _ { i } ^ { ( 1 ) } - \mathbf { m } _ { i } ^ { ( 1 ) } \right. _ { 2 } + \lambda _ { 2 } \sum _ { i = 1 } ^ { B } \left. \hat { \mathbf { r } } _ { i } ^ { ( 2 ) } - \mathbf { m } _ { i } ^ { ( 2 ) } \right. _ { 2 } .\tag{5}
$$

To prevent representation collapse and encourage separation from dificult negatives, we sample one hard negative for each sample and each reactant type. Specifically, the positive diagonal entry is masked out, and the negative index is sampled from a multinomial distribution derived from the softmax-normalized prediction-to-reactant similarities. Let $\mathbf { m } _ { n _ { i } ^ { ( k ) } } ^ { ( k ) }$ denote the sampled negative embedding for the k-th reactant of sample i. The negative matching term is given in eq 6:

$$
\mathcal { L } _ { \mathrm { n e g } } = \lambda _ { 1 } \sum _ { i = 1 } ^ { B } \left\| \hat { \mathbf { r } } _ { i } ^ { ( 1 ) } - \mathbf { m } _ { n _ { i } ^ { ( 1 ) } } ^ { ( 1 ) } \right\| _ { 2 } + \lambda _ { 2 } \sum _ { i = 1 } ^ { B } \left\| \hat { \mathbf { r } } _ { i } ^ { ( 2 ) } - \mathbf { m } _ { n _ { i } ^ { ( 2 ) } } ^ { ( 2 ) } \right\| _ { 2 } .\tag{6}
$$

The Molecular Matching Loss is then defined as shown in eq 7:

$$
\mathcal { L } _ { \mathrm { M M L } } = \frac { \mathcal { L } _ { \mathrm { p o s } } - \mathcal { L } _ { \mathrm { n e g } } } { 2 } .\tag{7}
$$

## • Molecular Contrastive Loss (MCL):

Second, standard contrastive learning often inadvertently penalizes false negatives—molecules that are structurally or functionally viable alternatives within the batch. To mitigate this limitation, we draw inspiration from recent advancements in contrastive learning, specifically iMolCLR <sup>48</sup> and ProGCL,<sup>49</sup> to formulate a re-weighted Molecular Contrastive Loss (MCL) enhanced by a dual-prior mechanism. We adjust the penalty of negative samples by injecting two weighting factors. For each reactant type, we compute bidirectional similarities between the predicted reactant embeddings and the ground-truth reactant embeddings, as given in eq 8:

$$
s _ { i j } ^ { r 2 p , ( k ) } = \frac { \mathbf { m } _ { i } ^ { ( k ) \top } \hat { \mathbf { r } } _ { j } ^ { ( k ) } } { \tau } , \quad s _ { i j } ^ { p 2 r , ( k ) } = \frac { \hat { \mathbf { r } } _ { i } ^ { ( k ) \top } \mathbf { m } _ { j } ^ { ( k ) } } { \tau } ,\tag{8}
$$

where $\tau$ is a learnable temperature parameter initialized to 0.07. To reduce the influence of potential false negatives in the batch, we apply a dual-prior weighting strategy to negative logits. First, following the structural prior used in iMolCLR, <sup>48</sup> we use the Tanimoto similarity between molecular fingerprints to down-weight structurally similar negatives, as shown in eq 9:

$$
w _ { i j } ^ { \mathrm { s t r u c } , ( k ) } = 1 - \operatorname { T a n i m o t o } \left( \mathbf { f p } _ { i } ^ { ( k ) } , \mathbf { f p } _ { j } ^ { ( k ) } \right) .\tag{9}
$$

The fingerprints are Morgan fingerprints generated from SMILES strings with radius 2 and 2048 bits. Invalid SMILES are represented by zero vectors. This down-weights negatives that have high 2D structural similarity to the query, as they may possess similar chemical properties, mitigating the efect of false negatives in the batch.

Then, following the probabilistic prior used in $\mathrm { P r o G C L } , ^ { 4 9 }$ we do not explicitly select true negatives with hard labels. Instead, for each in-batch candidate negative, we estimate its posterior probability of being a true negative from the batch-level cosine similarity distribution,as given in eq 10:

$$
w _ { i j } ^ { \mathrm { p r o b } , ( k ) } = P ( \mathrm { T r u e ~ N e g a t i v e } \mid \mathrm { s i m } _ { i j } ) .\tag{10}
$$

In implementation, this probability is obtained by fitting a two-component Beta mixture model to the normalized cosine similarities in the current batch.

Eq 11 is the final weight for each negative sample:

$$
w _ { i j } ^ { ( k ) } = \operatorname* { m a x } \left( w _ { i j } ^ { \mathrm { s t r u c } , ( k ) } \cdot w _ { i j } ^ { \mathrm { p r o b } , ( k ) } , 1 0 ^ { - 6 } \right) .\tag{11}
$$

The weight is injected into the logits only for negative samples, as given in eq 12:

$$
\tilde { s } _ { i j } ^ { ( k ) } = \left\{ \begin{array} { l l } { s _ { i j } ^ { ( k ) } , } & { j = y _ { i } , } \\ { s _ { i j } ^ { ( k ) } + \log w _ { i j } ^ { ( k ) } , } & { j \neq y _ { i } , } \end{array} \right.\tag{12}
$$

where $y _ { i }$ is the positive label corresponding to the matched sample in the batch.

It provides a probabilistic assessment of whether each candidate negative is a true negative. It dynamically fits a two-component Beta mixture model to the distribution of cosine similarities within the current batch. The posterior probability of a sample belonging to the low-similarity (true negative) component is used as its weight. This adaptively down-weights potential false negatives that have unexpectedly high representation similarity. This weighting strategy encourages the model to distinguish distinct molecules without overly penalizing chemically reasonable alternatives.

The final Molecular Contrastive Loss given in eq 13 is the weighted average of four cross-entropy terms, covering both directions and both reactants:

$$
\mathcal { L } _ { \mathrm { M C L } } = \frac { 1 } { 4 } \left( \lambda _ { 1 } \mathrm { C E } _ { \epsilon } \left( \tilde { S } _ { r 2 p } ^ { ( 1 ) } , \mathbf { y } \right) + \lambda _ { 2 } \mathrm { C E } _ { \epsilon } \left( \tilde { S } _ { r 2 p } ^ { ( 2 ) } , \mathbf { y } \right) + \lambda _ { 1 } \mathrm { C E } _ { \epsilon } \left( \tilde { S } _ { p 2 r } ^ { ( 1 ) } , \mathbf { y } \right) + \lambda _ { 2 } \mathrm { C E } _ { \epsilon } \left( \tilde { S } _ { p 2 r } ^ { ( 2 ) } , \mathbf { y } \right) \right) ,\tag{13}
$$

where $\lambda _ { 1 }$ and $\lambda _ { 2 }$ are the weights for reactant 1 and reactant 2, respectively, and $\mathrm { C E } _ { \epsilon }$ denotes cross-entropy with label smoothing $\epsilon = 0 . 1$ . All embeddings are L2-normalized before similarity computation, so the dot product corresponds to cosine similarity

Training proceeds in two phases: (1) Mol-Former is frozen while the Molecular Decoder focuses purely on learning correlations between product and reactant properties without interference from encoder updates. (2) End-to-end fine-tuning refines both Mol-Former and Molecular Decoder. The overall objective is shown in eq 14:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { D e c o d e r } } = \mathcal { L } _ { \mathrm { M C L } } + \mathcal { L } _ { \mathrm { M M L } } . } \end{array}\tag{14}
$$

guiding the model to learn the chemical property relationships between the product and its corresponding reactants. Furthermore, our property-based filtering approach attempts to provide a fundamentally less biased evaluation toward rare but valid reaction pathways. Unlike purely data-driven methods, which often struggle to learn underrepresented pathways due to limited training examples, our framework evaluates reactions based on reactants’ intrinsic chemical properties rather than statistical prevalence. We hope this mechanism may reduce the tendency to overlook rare but chemically plausible pathways by incorporating property-based constraints, regardless of their frequency, thereby helping RetroMPA to avoid ignoring valid but rare reaction pathways.

## Inverse Projection and Molecular Output

The reconstruction of SMILES strings from vector representations r poses non-trivial challenges. First, the solution retrosynthesis prediction space R and the ground-truth molecular embedding space are inherently non-equivalent, as they originate from distinct model outputs. Second, r predicted by the Molecular Decoder cannot maintain precise one-to-one correspondence with actual molecules. To resolve this issue and enable deterministic molecular output in SMILES format, we introduce a Molecular Dictionary.

Molecular Dictionary. The molecular dictionary serves as a fundamental key-value store, wherein the values correspond to molecular embeddings, and the keys are the associated molecular SMILES strings. For benchmarking baseline performance on standard datasets, the initial dictionary is populated with molecules drawn exclusively from the dataset. However, real-world chemical synthesis is not confined to the boundaries of any fixed dataset. Consequently, the molecular dictionary is designed to be dynamic rather than static. This dynamic dictionary automatically incorporates candidate molecules predicted by the base model into its repository. This mechanism enables RetroMPA to consider candidate molecules absent from the original dictionary once they are added from base-model outputs. In practical deployment scenarios, the molecular dictionary can also be manually augmented in bulk with catalogs of molecules from real-world chemical vendors (e.g., Enamine). The molecular dictionary itself is purely a passive database utilized exclusively during the Stage 3 inference phase, it does not participate in backpropagation or training.

Formally, given a predicted embedding r, we retrieve the most chemically similar molecule $M _ { p }$ from a dictionary D of known compounds by maximizing a composite similarity score. To capture both the directional alignment and the spatial proximity of the embeddings, our scoring function combines cosine similarity and a normalized Euclidean distance.

Specifically, for a dictionary molecule $M \in { \mathcal { D } }$ with embedding ${ \bf m } _ { M }$ , the cosine similarity is defined as:

$$
S _ { c o s } ( \mathbf { r } , \mathbf { m } _ { M } ) = \frac { \mathbf { r } \cdot \mathbf { m } _ { M } } { | \mathbf { r } | _ { 2 } | \mathbf { m } _ { M } | _ { 2 } }\tag{15}
$$

where ${ \bf m } _ { M }$ is the embedding of molecule M, computed in the same embedding space as the Molecular Decoder output. For the Euclidean distance $d ( \mathbf { r } , \mathbf { m } _ { M } ) = \| \mathbf { r } - \mathbf { m } _ { M } \| _ { 2 }$ , we apply a min-max normalization across all molecules in the dictionary to map the distances into a [0, 1] range. The normalized distance is then inverted to form a distance-based similarity metric $S _ { d i s t } { \mathrm { : } }$

$$
S _ { d i s t } ( { \bf r } , { \bf m } _ { M } ) = 1 - \frac { d ( { \bf r } , { \bf m } _ { M } ) - d m i n } { d m a x - d _ { m i n } + \epsilon }\tag{16}
$$

where $d _ { m i n } = \mathrm { m i n } _ { M ^ { \prime } \in \mathcal { D } } d ( \mathbf { r } , \mathbf { m } _ { M ^ { \prime } } )$ and $d _ { m a x } = \mathrm { m a x } _ { M ^ { \prime } \in \mathcal { D } } d ( \mathbf { r } , \mathbf { m } _ { M ^ { \prime } } )$ denote the minimum and maximum Euclidean distances for the current query r over the entire dictionary $\mathcal { D } _ { : }$ and $\epsilon = 1 0 ^ { - 8 }$ is a small constant added for numerical stability.

Finally, the target molecule $M _ { p }$ is retrieved by maximizing the sum of these two similarities:

$$
M _ { p } = \underset { M \in \mathcal { D } } { \arg \operatorname* { m a x } } \left( S _ { c o s } ( \mathbf { r } , \mathbf { m } _ { M } ) + S _ { d i s t } ( \mathbf { r } , \mathbf { m } _ { M } ) \right)\tag{17}
$$

For Top-K retrieval, we select the K molecules that yield the highest combined scores. This nearestneighbor search helps return valid molecules from the dictionary and favors candidates that are structurally compatible with the predicted chemical properties. The specific output algorithm is presented in Algorithm S1.

## Training

As aforementioned, we adopt the original Q-Former architecture and MolR <sup>7</sup> framework to implement Mol-Former. MolR is initialized with weights from its oficial GitHub repository. We employ a minimalist design, a standard transformer decoder <sup>50</sup> with 6 layers and 8 attention heads as our Molecular Decoder, but remove the final softmax layer to enable direct vector output. We set 192 as the maximum text token length and use AdamW optimizer <sup>51</sup> with the peak learning rate $1 \times 1 0 ^ { - 3 }$ . Pretraining was conducted for 351K steps on four NVIDIA RTX 4090 GPUs. The Molecular Decoder was then trained without reaction class labels for 200 epochs using the AdamW optimizer with a learning rate of $1 \times 1 0 ^ { - 4 }$

## Data and Software Availability

The dataset used in this study and the code files used to perform the experiments can be found at https://github.com/MengzhouLu/RetroMPA.

## Author Contributions

†M.L. and F.X. contributed equally to this work. Y.W. conceptualized and designed the study, supervised the research, acquired funding, and critically reviewed and edited the manuscript. M.L. and F.X. established the methodological framework, designed the experiments, performed formal data analysis, and wrote the origina draft. M.L. additionally developed and validated the computational code. Z.Y. prepared the experimental figures and visualizations. Z.Y. and H.H. provided writing guidance and reviewed and edited the manuscript. Y.L. and Y.Y. provided mathematical analysis support. W.L. provided chemical expertise and conducted the wet-lab experimental validation. All authors have given approval to the final version of the manuscript.

## Notes

The authors declare no competing financial interests.

## Acknowledgements

This work was partially supported by the National Science and Technology Major Project (2023ZD0120802), Wuhan Key Research and Development Program (Grant No. 2025051202030408) and the Fundamental Research Funds for the Central Universities (2042026kf0055).

## Supporting Information

• SI.pdf: This Supporting Information includes dataset descriptions, experimental NMR spectral validation, and the RetroMPA inference algorithm (PDF).

## References

[1] Long, L.; Li, R.; Zhang, J. Artificial Intelligence in Retrosynthesis Prediction and its Applications in Medicinal Chemistry. Journal of Medicinal Chemistry 2025, 68, 2333–2355, PMID: 39883477.

[2] Torren-Peraire, P.; Hassen, A. K.; Genheden, S.; Verhoeven, J.; Clevert, D.-A.; Preuss, M.; Tetko, I. V. Models matter: The impact of single-step retrosynthesis on synthesis planning. Digital Discovery 2024, 3, 558–572.

[3] Li, X.; Wang, S.; Lin, Y.; Wu, Y.; Yang, Y. Retro-Expert: Collaborative Reasoning for Interpretable Retrosynthesis. Proceedings of the 43rd International Conference on Machine Learning. 2026.

[4] Maziarz, K.; Tripp, A.; Liu, G.; Stanley, M.; Xie, S.; Gaiński, P.; Seidl, P.; Segler, M. H. Re-evaluating retrosynthesis algorithms with syntheseus. Faraday Discuss. 2025, 256, 568–586.

[5] Somnath, V. R.; Bunne, C.; Coley, C.; Krause, A.; Barzilay, R. Learning Graph Models for Retrosynthesis Prediction. Advances in Neural Information Processing Systems. 2021; pp 9405–9415.

[6] Weininger, D. SMILES, a chemical language and information system. 1. Introduction to methodology and encoding rules. Journal of Chemical Information and Computer Sciences 1988, 28, 31–36.

[7] Wang, H.; Li, W.; Jin, X.; Cho, K.; Ji, H.; Han, J.; Burke, M. D. Chemical-Reaction-Aware Molecule Representation Learning. International Conference on Learning Representations. 2022.

[8] Li, H.; Zhang, R.; Min, Y.; Ma, D.; Zhao, D.; Zeng, J. A knowledge-guided pre-training framework for improving molecular representation learning. Nat. Commun. 2023, 14, 7568.

[9] Ishida, S.; Terayama, K.; Kojima, R.; Takasu, K.; Okuno, Y. Prediction and Interpretable Visualization of Retrosynthetic Reactions Using Graph Convolutional Networks. Journal of Chemical Information and Modeling 2019, 59, 5026–5033, PMID: 31769668.

[10] Wang, S.; Guo, Y.; Wang, Y.; Sun, H.; Huang, J. SMILES-BERT: Large Scale Unsupervised Pre-Training for Molecular Property Prediction. Proceedings of the 10th ACM International Conference on Bioinformatics, Computational Biology and Health Informatics. New York, NY, USA, 2019; p 429–436.

[11] Sumiya, Y.; Harabuchi, Y.; Nagata, Y.; Maeda, S. Quantum Chemical Calculations to Trace Back Reaction Paths for the Prediction of Reactants. JACS Au 2022, 2, 1181–1188.

[12] Toniato, A.; Unsleber, J. P.; Vaucher, A. C.; Weymuth, T.; Probst, D.; Laino, T.; Reiher, M. Quantum chemical data generation as fill-in for reliability enhancement of machine-learning reaction and retrosynthesis planning. Digital Discovery 2023, 2, 663–673.

[13] Liu, Q.; Tang, K.; Zhang, L.; Du, J.; Meng, Q. Computer-assisted synthetic planning considering reaction kinetics based on transition state automated generation method. AIChE Journal 2023, 69 .

[14] Zhong, Z.; Song, J.; Feng, Z.; Liu, T.; Jia, L.; Yao, S.; Hou, T.; Song, M. Recent advances in deep learning for retrosynthesis. Wiley Interdisciplinary Reviews: Computational Molecular Science 2024, 14, e1694.

[15] Guo, J.; Schwaller, P. Directly optimizing for synthesizability in generative molecular design using retrosynthesis models. Chem. Sci. 2025, 16, 6943–6956.

[16] Hartenfeller, M.; Eberle, M.; Meier, P.; Nieto-Oberhuber, C.; Altmann, K.-H.; Schneider, G.; Jacoby, E.; Renner, S. A collection of robust organic synthesis reactions for in silico molecule design. Journal of Chemical Information and Modeling 2011, 51, 3093–3098.

[17] Szymkuć, S.; Gajewska, E. P.; Klucznik, T.; Molga, K.; Dittwald, P.; Startek, M.; Bajczyk, M.; Grzybowski, B. A. Computer-assisted synthetic planning: the end of the beginning. Angewandte Chemie (International ed. in English) 2016, 55, 5904–5937.

[18] Coley, C. W.; Barzilay, R.; Jaakkola, T. S.; Green, W. H.; Jensen, K. F. Prediction of Organic Reaction Outcomes Using Machine Learning. ACS Central Science 2017, 3, 434–443, PMID: 28573205.

[19] Law, J.; Zsoldos, Z.; Simon, A.; Reid, D.; Liu, Y.; Khew, S. Y.; Johnson, A. P.; Major, S.; Wade, R. A.; Ando, H. Y. Route Designer: A Retrosynthetic Analysis Tool Utilizing Automated Retrosynthetic Rule Generation. Journal of Chemical Information and Modeling 2009, 49, 593–602, PMID: 19434897.

[20] Yao, L.; Guo, W.; Wang, Z.; Xiang, S.; Liu, W.; Ke, G. Node-Aligned Graph-to-Graph: Elevating Template-free Deep Learning Approaches in Single-Step Retrosynthesis. JACS Au 2024, 4, 992–1003.

[21] Yan, C.; Ding, Q.; Zhao, P.; Zheng, S.; YANG, J.; Yu, Y.; Huang, J. RetroXpert: Decompose Retrosynthesis Prediction Like A Chemist. Advances in Neural Information Processing Systems. 2020; pp 11248–11258.

[22] Han, Y.; Xu, X.; Hsieh, C.-Y.; Ding, K.; Xu, H.; Xu, R.; Hou, T.; Zhang, Q.; Chen, H. Retrosynthesis prediction with an iterative string editing model. Nat. Commun. 2024, 15, 6404.

[23] Jiang, Y.; WEI, Y.; Wu, F.; Huang, Z.; Kuang, K.; Wang, Z. Learning Chemical Rules of Retrosynthesis with Pre-training. Proceedings of the AAAI Conference on Artificial Intelligence 2023, 37, 5113–5121.

[24] Zhong, Z.; Song, J.; Feng, Z.; Liu, T.; Jia, L.; Yao, S.; Wu, M.; Hou, T.; Song, M. Root-aligned SMILES: a tight representation for chemical reaction prediction. Chem. Sci. 2022, 13, 9023–9034.

[25] Wan, Y.; Hsieh, C.-Y.; Liao, B.; Zhang, S. Retroformer: Pushing the Limits of End-to-end Retrosynthesis Transformer. Proceedings of the 39th International Conference on Machine Learning. 2022; pp 22475– 22490.

[26] Sacha, M.; Błaż, M.; Byrski, P.; Dąbrowski-Tumański, P.; Chromiński, M.; Loska, R.; Włodarczyk-Pruszyński, P.; Jastrzębski, S. Molecule Edit Graph Attention Network: Modeling Chemical Reactions as Sequences of Graph Edits. Journal of Chemical Information and Modeling 2021, 61, 3273–3284, PMID: 34251814.

[27] Hammond, G. S. A correlation of reaction rates. Journal of the American Chemical Society 1955, 77, 334–338.

[28] Kolb, H. C.; Finn, M.; Sharpless, K. B. Click chemistry: diverse chemical function from a few good reactions. Angewandte Chemie (International ed. in English) 2001, 40, 2004–2021.

[29] Noyori, R. Asymmetric catalysis: science and opportunities (Nobel lecture). Angewandte Chemie (International ed. in English) 2002, 41, 2008–2022.

[30] Corey, E. J. General methods for the construction of complex molecules. Pure and Applied Chemistry 1967, 14, 19–38.

[31] Schneider, N.; Stiefl, N.; Landrum, G. A. What’s what: The (nearly) definitive guide to reaction role assignment. Journal of Chemical Information and Modeling 2016, 56, 2336–2346.

[32] Chen, S.; Jung, Y. Deep Retrosynthetic Reaction Prediction using Local Reactivity and Global Attention. JACS Au 2021, 1, 1612–1620, PMID: 34723264.

[33] Karpov, P.; Godin, G.; Tetko, I. V. A transformer model for retrosynthesis. Artificial Neural Networks and Machine Learning – ICANN 2019: Workshop and Special Sessions. 2019; pp 817–830.

[34] Zhang, X.; Mo, Y.; Wang, W.; Yang, Y. Retrosynthesis prediction enhanced by in-silico reaction data augmentation. 2024; https://arxiv.org/abs/2402.00086.

[35] Xie, S.; Yan, R.; Guo, J.; Xia, Y.; Wu, L.; Qin, T. Retrosynthesis Prediction with Local Template Retrieval. Proceedings of the AAAI Conference on Artificial Intelligence 2023, 37, 5330–5338.

[36] Milvus Project Milvus Documentation. 2025; https://milvus.io/docs.

[37] van der Maaten, L.; Hinton, G. Visualizing Data using t-SNE. Journal of Machine Learning Research 2008, 9, 2579–2605.

[38] Edwards, C. N.; Zhai, C. X.; Ji, H. Text2Mol: Cross-Modal Molecule Retrieval with Natural Language Queries. Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing. 2021; pp 595–607.

[39] Fang, Y.; Liang, X.; Zhang, N.; Liu, K.; Huang, R.; Chen, Z.; Fan, X.; Chen, H. Mol-Instructions: A Large-Scale Biomolecular Instruction Dataset for Large Language Models. International Conference on Learning Representations. 2024.

[40] Liu, S.; Nie, W.; Wang, C.; Lu, J.; Qiao, Z.; Liu, L.; Tang, J.; Xiao, C.; Anandkumar, A. Multi-modal molecule structure–text model for text-based retrieval and editing. Nat. Mach. Intell. 2023, 5, 1447–1457.

[41] Dai, H.; Li, C.; Coley, C. W.; Dai, B.; Song, L. Retrosynthesis Prediction with Conditional Graph Logic Network. 2020; https://arxiv.org/abs/2001.01408.

[42] Sheshanarayana, R.; You, F. Rethinking Retrosynthesis: Curriculum Learning Reshapes Transformer-Based Small-Molecule Reaction Prediction. Journal of Chemical Information and Modeling 2025, 65, 11047–11063, PMID: 41001729.

[43] Landrum, G. RDKit: Open-source cheminformatics. https://www.rdkit.org, 2013.

[44] Bushuiev, R.; Bushuiev, A.; de Jonge, N. F.; Young, A.; others MassSpecGym: A Benchmark for the Discovery and Identification of Molecules. Advances in Neural Information Processing Systems 37 (Datasets and Benchmarks Track). 2024.

[45] Liu, Z.; Li, S.; Luo, Y.; Fei, H.; Cao, Y.; Kawaguchi, K.; Wang, X.; Chua, T.-S. MolCA: Molecular Graph-Language Modeling with Cross-Modal Projector and Uni-Modal Adapter. Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing. 2023; pp 15623–15638.

[46] Li, J.; Li, D.; Savarese, S.; Hoi, S. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. Proceedings of the 40th International Conference on Machine Learning. 2023; pp 19730–19742.

[47] Feng, Y.; Gu, S.; Guo, D.; Yang, Z.; Shao, C. Guiding Teacher Forcing with Seer Forcing for Neural Machine Translation. Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers). 2021; pp 2862–2872.

[48] Wang, Y.; Magar, R.; Liang, C.; Barati Farimani, A. Improving Molecular Contrastive Learning via Faulty Negative Mitigation and Decomposed Fragment Contrast. Journal of Chemical Information and Modeling 2022, 62, 2713–2725, PMID: 35638560.

[49] Xia, J.; Wu, L.; Wang, G.; Li, S. Z. ProGCL: Rethinking Hard Negative Mining in Graph Contrastive Learning. International conference on machine learning. 2022.

[50] Vaswani, A.; Shazeer, N.; Parmar, N.; Uszkoreit, J.; Jones, L.; Gomez, A. N.; Kaiser, Ł.; Polosukhin, I. Attention is all you need. Advances in Neural Information Processing Systems. 2017.

[51] Loshchilov, I.; Hutter, F. Decoupled Weight Decay Regularization. International Conference on Learning Representations. 2019.

## TOC Graphic

![](images/1ea33369f22e54670cb10c3b875162f4cf2c390f55cb62db8d3a88edf6c96c7d.jpg)

# Supporting Information: RetroMPA: A Molecular Property-Aware Auxiliary Framework for Enhancing Retrosynthesis Prediction

Mianzhi Liu†1, Fan Xiao†2, Zhiliang Yu2, Huayang Huang2, Yuke Li2, Yi Yang3, Wenbo Liu4, and Yu Wu\*2

1School of Cyber Science and Engineering, Wuhan University, Wuhan 430072, China 2School of Computer Science, Wuhan University, Wuhan 430072, China 3College of Computer Science and Technology, Zhejiang University, Hangzhou 310027, China 4College of Chemistry and Molecular Sciences, Wuhan University, Wuhan 430072, China

\*Email: wuyucs@whu.edu.cn

## Contents

Fig. S1 illustrates the data format of the chemical datasets used for pretraining. Fig. S2 and Fig. S3 present the experimental ¹H and ¹³C NMR spectra of the wet-lab product, respectively. Algorithm. S1 details the full inference procedure of the framework.

## S1.Analysis of the chemical datasets used for pretraining

To pretrain the Mol-Former component of RetroMPA for learning molecular chemical properties and thereby guiding retrosynthesis tasks, we leveraged three large-scale chemical-text datasets: ChEBI-20-MM, Mol-Instructions, and PubChemSTM, collectively endowing the model with rich molecular knowledge, see Fig. S1. Together, these sources provide rich semantic grounding: a portion of the text explicitly describes critical chemical attributes (e.g., functional groups, chirality), while the majority cover broader molecular features such as electronic properties or biochemical roles. This multimodal pretraining enables Mol-Former to align molecules with natural language descriptions, producing embeddings enriched with interpretable chemical semantics.

## S2.Experimental chemical analysis

To verify the generalization ability of our prediction model in real-world scenarios, we conducted experiments in a chemistry laboratory based on the model’s predictions. In chemistry, the identification of a chemical substance is typically achieved through the analysis of its spectra. The products were sent to a third-party testing institution for analysis, where nuclear magnetic resonance (NMR) spectroscopy was used to determine whether the reaction products matched the predictions.As shown in Fig. S2 and Fig. S3, the following images show the hydrogen (¹H NMR) and carbon (¹³C NMR) spectra of the predicted reaction product, which were found to be in complete agreement with the model’s predictions after analysis. The order from left to right is Suzuki–Miyaura coupling, Bucherer reaction, and Friedel–Crafts acylation.

![](images/8e3182f085917e961a153284f46b217d0df26b8fa8fd55dd9798dd2eae8cf567.jpg)

Figure S1: Introduction of datasets.  
![](images/7be391837baa9b65584ac6933383b870c3814a49a10e437732db36d5da74cacc.jpg)

![](images/feadb9d40dabb63ab783c13bc0b94852efa4657831f49932b7568200c13d9bc9.jpg)

![](images/f99ae274a0d6e388ac9824af50abe1e5422e50a70da8543314914945903f65c1.jpg)

Figure S2: ¹H NMR spectra of all chemical products.  
![](images/73d4fe89fd4f7cc47c5e28a6f63e19e3876350dbfa376b2d36752479fcc32081.jpg)

![](images/567ef33ccf5a5330cfe839409240daf1b404b4730441530b7eb7cf8c7cf62caa.jpg)

![](images/5a57a13630c761f07fb77c153a324f217d0bfe883bce4d3783a2d9f2cc25e828.jpg)  
Figure S3: ¹³C NMR spectra of all chemical products.

## S3. Inference algorithm with Dynamic Molecular Dictionary

Algorithm S1 details the streamlined RetroMPA inference workflow, in which the update of the molecular dictionary D is optional. For a reaction with two reactants, given a product SMILES P, the method first retrieves the Top-1 reactant pair $( b _ { 1 } , b _ { 2 } )$ from a base retrosynthesis model. If the update is enabled, the molecular embeddings of these base predictions are generated via Mol-Former and used to update D. The refinement stage employs RetroMPA (denoted as Imp) to predict the remaining reactant conditioned on the product and a single reactant drawn from the base pair. Specifically, one reactant $( \mathrm { e . g . , } b _ { 1 } )$ is taken as the fixed condition, and Imp generates a Top-K candidate list $\mathcal { C }$ for the other reactant. A consistency check then compares the original counterpart $b _ { 2 }$ against C: if $b _ { 2 }$ ranks within the top $K ,$ the base pair $( b _ { 1 } , b _ { 2 } )$ is retained;

otherwise, the framework replaces $b _ { 2 }$ with the highest-ranked candidate $\mathcal { C } [ 1 ]$

Algorithm S1 Streamlined RetroMPA Inference with Optional Dynamic Dictionary   
Require: Product SMILES P, Base Model, Refinement Model Imp, Dynamic Dictionary D (optional),   
Threshold $K = 1 0 ,$ , Boolean flag update\_dict   
Ensure: Final reactant combination $R _ { \mathrm { f i n a l } } .$ , (optionally) updated dictionary D   
1: $( b _ { 1 } , b _ { 2 } )  \mathrm { B a s e M o d e l . T o p 1 } ( P )$ ▷ Obtain initial Top-1 pair   
2: if update\_dict then   
3: for each reactant $r \in \{ b _ { 1 } , b _ { 2 } \}$ do   
4: $\mathcal { D }  \mathcal { D } \cup$ {(SMILES(r), Mol-Former(r))} ▷ Optionally update dictionary   
5: end for   
6: end if   
7: $r _ { \mathrm { c o n d } }  \mathrm { R a n d o m C h o i c e } ( \{ b _ { 1 } , b _ { 2 } \} )$ ▷ Select one reactant as condition   
8: $r _ { \mathrm { t a r g e t } }  \{ b _ { 1 } , b _ { 2 } \} \setminus \{ r _ { \mathrm { c o n d } } \}$ ▷ The reactant to be predicted   
9: $\mathcal { C } \gets \mathrm { I m p } ( P , r _ { \mathrm { c o n d } } )$ ▷ Generate Top-K candidates for r<sub>target</sub>   
10: if $r _ { \mathrm { t a r g e t } } \in \mathcal { C } [ 1 : K ]$ then   
11: $R _ { \mathrm { f i n a l } }  ( b _ { 1 } , b _ { 2 } )$ ▷ Keep original base prediction   
12: else   
13: $R _ { \mathrm { f i n a l } }  ( r _ { \mathrm { c o n d } } , \mathcal { C [ 1 ] } )$ ▷ Override with highest-ranked refined candidate   
14: end if   
15: return $R _ { \mathrm { f i n a l } } , \mathcal { D }$