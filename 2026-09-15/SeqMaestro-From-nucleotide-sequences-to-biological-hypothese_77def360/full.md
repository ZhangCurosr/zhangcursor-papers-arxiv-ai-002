# SeqMaestro: From nucleotide sequences to biological hypotheses through interpretable machine learning

Evgeny S. Saveliev<sup>1</sup>, Krzysztof Kacprzyk<sup>1</sup>,

Charlotte Capitanchik<sup>2,3</sup>, Neelanjan Mukherjee<sup>4</sup>, Kate Matlin<sup>4</sup>, Ryan Sheridan<sup>4</sup>, Srinivas Ramachandran<sup>4</sup>, Jernej Ule<sup>2,3</sup>, David L. Bentley<sup>4</sup>, Mihaela van der Schaar<sup>2,1</sup>

<sup>1</sup>Department of Applied Mathematics and Theoretical Physics, University of Cambridge, Cambridge, UK <sup>2</sup>The Francis Crick Institute, London, UK

<sup>3</sup>UK Dementia Research Institute at King’s College London, London, UK

<sup>4</sup>Department of Biochemistry and Molecular Genetics, University of Colorado Anschutz Medical Campus, Aurora, CO, USA

## Abstract

Nucleotide sequence analysis is central to problems spanning regulatory genomics, evolutionary biology, and phenotype prediction. Classical bioinformatics methods extract interpretable sequence properties such as motifs and k-mer composition, but their flexibility is limited. In contrast, modern deep learning models can learn powerful predictive representations directly from raw sequences, yet their internal representations and decision mechanisms are dificult to inspect. Interpretable machine learning methods (e.g., sparse linear models and decision trees) provide human-understandable representations of predictive relationships but are not designed to operate directly on nucleotide sequences. Here, we introduce SeqMaestro, a machine learning framework that proposes biological hypotheses from nucleotide sequences using interpretable models. Our solution is centered around a two-layer interface that connects nucleotide sequences with the broader ecosystem of interpretable machine learning. SeqMaestro uses this interface to fit diverse combinations of interpretable models, feature representations, and extraction strategies, leveraging variability across transparent models to identify robust biological signals and richer predictive relationships than feature importance alone can provide. The system also supports data transformation and cleaning, model fitting, hyperparameter tuning, reliability analysis, and synthesis of results into a contextualized written report. By providing these capabilities through a no-code workflow, SeqMaestro is designed to make interpretable sequence analysis accessible to researchers without requiring extensive programming or machine learning expertise. SeqMaestro thereby provides an accessible route from nucleotide sequences to biological hypotheses.

## Introduction

Nucleotide sequences encode the information required for life and underlie diverse biological processes in health and disease, including gene regulation and RNA processing, replication, genome maintenance, and mutation. Thus, they can be a great source of novel biological hypotheses and discoveries after a careful computational analysis. Indeed, bioinformatics analyses routinely involve attempts to extract biologically meaningful features from nucleotide sequences, a process that often relies on labour-intensive trial and error. Traditionally, such analysis relied on specialized bioinformatics methods that capture biologically meaningful sequence structure, common approaches including: homology-based search with tools such as BLAST [1], motif discovery with methods such as MEME [2], probabilistic sequence models such as hidden Markov models [3], explicit sequence representations based on k-mer composition [4], and supervised predictive methods built on related representations, such as gkm-SVM [5]. These approaches are often highly efective, but they typically require the researcher to specify in advance the form in which a relevant sequence signal is expected to appear. They define a particular vocabulary in which discoveries can be made, such that biologically relevant properties outside that vocabulary may be overlooked.

Deep learning has substantially relaxed this requirement by learning predictive sequence representations directly from nucleotide sequences. Models such as DeepSEA [6] learn regulatory sequence features using convolutional neural networks, whereas Enformer [7] integrates long-range sequence context using transformer-based architectures. More recent genomic foundation models, such as Nucleotide Transformer [8], extend this paradigm by learning general-purpose sequence representations through large-scale pretraining. These approaches have substantially expanded the range and complexity of biological phenotypes and molecular events that can be predicted from sequence. Yet prediction alone is often insuficient for scientific discovery: a highly accurate black-box model may reveal the existence of a signal without explaining which sequence properties drive it or how those properties interact. Post-hoc explanation methods can highlight sequence positions or patterns associated with model predictions, but they do not generally yield a predictive model whose logic is itself expressed in explicit rules. Moreover, post-hoc explanations may not be reliable and can be misleading [9]. Some explanation methods may not faithfully reflect the information learned by the underlying model [10]; explanations can be unstable under small input perturbations [11], and diferent methods can produce inconsistent explanations of the same prediction [12]. Rather than explaining a complex predictive model after training, we therefore ask whether the predictive model itself can serve as an explicit biological hypothesis.

Intrinsically interpretable (or transparent) machine learning constructs models whose predictive mechanisms can be inspected directly. That includes linear models, decision trees, generalized additive models (GAMs), and rule-based methods. However, this methodological ecosystem remains largely disconnected from nucleotide sequence analysis. Most interpretable learning algorithms operate on well-defined prediction tasks with specified targets and structured tabular features rather than raw sequences.

To address this challenge, we construct an interface between nucleotide sequences and transparent machine learning that maps raw sequences into a broad, searchable space of human-readable hypotheses, allowing us to utilize the existing ecosystem of interpretable machine learning models for sequencebased prediction and discovery. The interface has two layers. First, we introduce a taxonomy of interpretable sequence representations that separates three levels of feature expressivity: positional features, pattern-based features such as k-mers and motifs, and programmatic features defined by explicit, human-readable computations over a sequence. These representations form increasingly expressive languages for describing sequence-level hypotheses. Because intrinsically interpretable models can incorporate only a limited number of features and interactions while remaining understandable, richer biological structure must often be captured within the features themselves. Programmatic features provide a particularly flexible representation, allowing composition, positional relationships, motif organization, and higher-order logical structure to be expressed while remaining directly inspectable. This expands the space of interpretable hypotheses that can be explored without requiring the researcher to specify in advance the particular sequence properties that are likely to be relevant.

The second layer is the actual mechanism of extracting these features. In particular, it concerns three complementary strategies for programmatic feature generation. These strategies difer in how tightly hypothesis generation is coupled to empirical feedback: from one-shot proposal, through iterative propose-fit-refine cycles, to joint procedures in which feature construction becomes part of model fitting itself. This interface allows us to fit diferent combinations of transparent ML models, feature types, and even extraction strategies.

Here we introduce this interface within a framework called SeqMaestro. SeqMaestro is an end-to-end framework for searching nucleotide sequences for explicit, predictive, and testable biological hypotheses. It formalizes biological questions as supervised sequence-based prediction tasks and then leverages the Rashomon Efect by fitting multiple combinations of transparent model pipelines as defined by the interface. Diferent transparent model classes expose diferent forms of biological structure. Linear models identify additive associations, GAMs reveal nonlinear response relationships, and decision trees expose thresholds and conditional dependencies between sequence properties. SeqMaestro therefore treats the model class itself as part of the hypothesis search, fitting diverse, transparent pipelines, aggregating their results, and assessing the robustness of the resulting discoveries.

Crucially, SeqMaestro does not restrict interpretation to individual feature importance. Because the fitted models are themselves transparent, the resulting hypothesis may be an individual feature, a nonlinear response curve, a threshold, or a conditional interaction between sequence properties. This allows us to move from identifying isolated signals (e.g., that CpG frequency is predictive) to recovering richer predictive logic. A decision tree, for instance, may reveal that a particular sequence property is informative only when another condition is satisfied (i.e. only in a particular sequence context), whereas a GAM can show how the prediction changes continuously across the values of a feature, such as GC content. In this sense, the predictive model itself becomes part of the biological hypothesis.

![](images/896e09079f80ce1fad58cbc58ed056c537a0d17b555ebadaa44d11dd8be61672.jpg)  
Figure 1 In this paper, we build a bridge between nucleotide sequences and interpretable machine learning.

SeqMaestro also performs the practical steps required for end-to-end analysis, including data transformation and cleaning, model fitting, hyperparameter tuning, reliability analysis, and synthesis of results into a contextualized written report. This automation makes it practical to explore a hypothesis space that would otherwise require substantial manual feature engineering, model comparison, and interpretation. By providing these capabilities through a no-code workflow, SeqMaestro is designed to make interpretable sequence analysis accessible to researchers without requiring extensive programming or machine learning expertise. By jointly treating feature discovery, model fitting, and biological interpretation as components of a single workflow, SeqMaestro provides a general framework for moving from nucleotide sequences to explicit, predictive, and testable biological hypotheses.

## Results

## SeqMaestro methodology

SeqMaestro framework has three stages: (1) dataset preparation, (2) model fitting, and (3) analysis and synthesis. They are depicted in Figure 1.

## Dataset preparation

The goal of the first stage is to prepare a curated dataset that can be passed to our interface in the second stage. In particular, it transforms the raw sequences into a form suitable for nucleotide sequence-based prediction, in which each observation consists of a nucleotide sequence x and a binary label y<sub>i</sub>. Our current implementation focuses on binary classification, while regression and multiclass or multilabel prediction represent natural extensions.

Many biological questions can be formulated in this way. For example, in RNA polymerase II pausing, fixed-length sequence windows around genomic positions can be labeled according to whether strong pausing occurs; similarly, in somatic hypermutation, local sequence windows can be labeled according to whether the central nucleotide is mutated, enabling the discovery of sequence features associated with increased mutation probability. This stage also includes quality controls and splitting the sequences into train and test (holdout) samples. Holdout samples are not used for model fitting and are only used fo evaluation.

## Model fitting - SeqMaestro interface

SeqMaestro fits multiple intrinsically interpretable models, including logistic regression, additive models, and decision trees, across the diferent feature representations and extraction mechanisms described below. We refer to each such pipeline as a separate arm and to the whole collection of arms as a battery. Together, these approaches allow us to compare not only diferent interpretable model classes and feature representations, but also diferent strategies for searching the space of interpretable programmatic features.

Nucleotide sequences are naturally represented as strings over a small alphabet, typically A, C, G, and T for DNA or A, C, G, and U for RNA. Although this raw representation is biologically natural, most conventional interpretable machine learning methods are not designed to operate directly on nucleotide strings. Applying such models therefore requires transforming each sequence into a structured feature space.

We consider three broad families of feature representations: positional features, pattern-based features, and programmatic features.

Positional features. The most direct representation treats each sequence position as an individual feature. For a sequence of length L, this produces L categorical variables, where the feature at position i corresponds to the nucleotide observed at that position.

To accommodate sequences of varying lengths, we introduce two encoding variants: landmark-based and boundary-based. Both of them operate with a predefined window (e.g., 50 nucleotides). The landmarkbased system indexes positions from a particular point of a sequence in both directions. Boundary-based, on the other hand, focuses only on nucleotides at the beginning and end of the sequences. The choice of encoding strategy can be specified by the user or suggested by SeqMaestro itself, based on background knowledge of the problem.

This representation preserves a part of the positional structure of the original sequence while expressing it in a form that can be consumed by conventional machine learning algorithms. Models that do not natively support categorical variables require an additional encoding step. For example, one-hot encoding can represent each nucleotide at each position with binary indicator variables, enabling models such as logistic regression to use it.

Pattern-based features. A second family of representations describes sequences through the occurrence or strength of local sequence patterns. This category includes both exact subsequences, such as k-mers, and more flexible patterns, such as motifs.

For k-mer representations, features can describe the presence, count, or frequency of specific nucleotide subsequences of length k. For example, a feature may indicate whether a particular 6-mer occurs in a sequence or how many times it appears. Multiple values of k can be considered simultaneously, allowing patterns at diferent local scales to be represented.

Motif-based representations generalize this idea by allowing patterns to tolerate variation across positions. A motif may be represented, for example, by a consensus sequence or a position weight matrix. The motif representation itself is not necessarily passed directly to the predictive model. Instead, it defines a matching procedure from which conventional scalar features can be derived, such as the maximum motif-match score within a sequence, the presence of a match above a threshold, or the number of motif occurrences. The motifs themselves can be suggested by standard motif discovery algorithms.

Thus, both k-mers and motifs act as pattern detectors that transform raw sequences into Boolean, integer, or continuous variables suitable for standard interpretable machine learning methods. The principal distinction is that k-mers generally define exact sequence matches, whereas motifs describe generalized or probabilistic local patterns.

Programmatic features. A substantially more expressive feature space can be obtained by allowing features to be defined as explicit, interpretable computations over the sequence. Such features may represent global compositional properties; positional conditions; counts and distances between sequence elements; motif or k-mer relationships; palindromic or symmetric structure; or logical combinations of multiple conditions. For example, a feature might indicate whether a compositional statistic exceeds a threshold while a particular nucleotide occurs at a specified position, or whether two motifs occur within a given distance of one another.

Programmatic features therefore subsume many simpler feature types as possible building blocks. Positions, k-mers, motifs, sequence-composition statistics, and other primitive operations can be composed into higher-level, human-readable hypotheses about sequence structure.

The hypothesis space of possible programmatic features is extremely large, making exhaustive enumeration impractical. In our framework, candidate features are instead proposed by a feature-generation procedure powered by large language models. We consider three distinct strategies for feature generation. In the one-shot setting, a language model proposes a set of candidate features once, which are subsequently evaluated by the interpretable model. In the iterative setting, features are proposed and evaluated over multiple rounds: after each model fit, information about the resulting predictive performance and selected features is returned to the language model, which uses this feedback to propose improved features for the next iteration. Finally, in the joint setting, feature generation is embedded directly into the model-fitting procedure, so that new programmatic features are constructed adaptively as the model is learned. In the current implementation, this joint strategy is available for decision trees through an existing method [13].

Conceptually, the three representations form a progression from direct sequence encoding to increasingly abstract descriptions of sequence properties: positional features describe individual locations, patternbased features describe recurring local sequence patterns, and programmatic features represent general interpretable functions of the sequence. The main drawback of lower-complexity features is their limited expressivity within a fixed interpretability budget. For instance, if we want to use a decision tree with depth 3, sole dependence on positional features will make it harder to discover complex patterns. However, as our case studies below show, all feature types have their advantages. Positional features are best suited for capturing short, local patterns around a specific site or its boundaries. K-mers and motifs excel at identifying short, widespread, or recurring patterns. Programmatic features are ideal for situations involving global patterns, where we need to perform complex computations or rely on prior knowledge of the language model (LLM). These features also allow for optional scientist guidance regarding the preferable form of the discovered features. For instance, if the model should focus on global patterns, symmetries or counts. Appendix G follows features of every kind, whether derived from the data alone or proposed by the language model under each of these strategies, through the rest of the framework.

## Analysis and synthesis

The third stage is divided into two sub-stages. First, the models are reduced to the 10 most important features to ensure that they are understandable. After the features are selected, the reduced models are refitted to the data. The second sub-stage is the reliability panel, which uses various techniques (grouped into three tiers) to assess the robustness of the discovered features. Tier 1 checks the stability of the discovered features by testing whether the same features would appear under a random sub-sampling of the dataset. Tier 2 assesses the association between the target and each feature separately. Finally, Tier 3 assesses necessity. It checks whether the model loses accuracy when a particular feature is removed. After the reliability panel is completed, SeqMaestro prepares a written report on the entire analysis.

## Experimental Findings

SeqMaestro aims to give a biologist two things from a set of labeled sequences: interpretable model(s) small enough that their internal logic can be understood, and a tractable list of sequence properties that are dependable enough to follow up as hypotheses in subsequent experiments or analyses. To see whether it accomplishes these goals, we applied SeqMaestro to four biological questions, two about DNA and two on RNA, chosen to cover a range of diferent informative signals that the sequences contain.

Each question is discussed in turn below and presented in individual figures. The four figures share a common layout for ease of comparison: the task and its window (panel a), how every configuration fares when it is cut down to ten features (panel b), one interpretable model presented in full (panels c and d), and the concepts that several configurations converged on, with the evidence behind each (panel e). Where available, we show independent or orthogonal support for novel features identified in SeqMaestro analysis, in panel f. Gradient-boosted trees serve primarily as a black-box comparison and do not meaningfully surpass interpretable models on any task. We use AUROC as the primary performance metric throughout the report and figures.

## Polycomb nucleation

Polycomb domains are stretches of chromatin carrying the repressive histone mark H3K27me3. Replication halves this mark, and it is restored over the following cell cycle, first at a small number of nucleation sites and then by spreading outward from them. Veronezi and Ramachandran [14] followed that recovery across 1,362 domains in mouse embryonic stem cells, dividing each into segments and clustering the segments by their recovery. We took the fastest-recovering segments, their cluster 7, as one class and the slowest, their cluster 1, as the other, and gave every model 200 nt of genomic DNA centered on the segment (Fig. 2a). Because both classes come from the same domains, the comparison is between segments that share the same chromatin context, and it asks what distinguishes the places where the mark returns first versus last.

SeqMaestro identifies one concept as especially important for this problem. Six of the nine interpretable

![](images/4ae235a595750e4533bd6e5363adaf8e63529f7486710fbac28a93b29d247d3c.jpg)

![](images/3046f743d651d8fb5ab9864f61bcf8498c32cab8bcba0f68dc95018195619f8d.jpg)

## Polycomb nucleationa

Mouse embryonic stem cells, mm10. 5,789 segments of 1,362 Polycomb domains, labelled by how fast H3K27me3 returns after replication.

cluster 7 the mark returns to the segment first cluster 1 it returns last

Training / validation / hold-out: 4,053 / 578 / 1,158 segments, every domain kept on one side.   
Cluster 7 is 32% of segments, as the experiment found them.

Programmatic features, written by a language model Positional and pattern-based features, enumerated in advance

Gradient-boosted trees, a reference only

## c The decision tree with programmatic features, AUROC 0.839 on the hold-out set

Features written at each split. Each box gives the training segments reaching it and the share of cluster 7 among them, which also sets the shading. The outlined end point is the rule quoted in the text.

## b Every configuration, cut to ten features

![](images/e6fc3ce460684b4ccf7d7ccf94220e50ba8d3b91227c02b88c3b300b71399eb9.jpg)

![](images/058a4e3b29c33eeeea18b0d50d9dfee865cf3a9ef2806ee504e2db33567a2574.jpg)  
Every per-position indicator the additive model could choose from, beside the programmatic feature that leads the same task.

## d No single position carries it

![](images/f6b02b282663a066d440cc48e7f06428612cb04333082abcd6d9147ed5104295.jpg)

## e What several configurations reached, and how each holds up

A link joins a configuration to each concept its ten-feature model contained.

<table><tr><td colspan="2">Concept</td><td>Argues for</td><td>Reached by</td><td>Returned again in half-samples</td><td></td><td>Separation alone, hold-out set</td><td></td><td>q</td></tr><tr><td colspan="2">Tree · positional</td><td></td><td></td><td>0 0.5</td><td>1</td><td>0.2 0.5</td><td>0.8</td><td></td></tr><tr><td>Additive · positional</td><td>CpG density</td><td>cluster 7</td><td>6</td><td></td><td>1.00</td><td></td><td>0.84</td><td>&lt; 0.001</td></tr><tr><td>Linear · positional</td><td>G+C fraction and runs of G or C</td><td>cluster 7</td><td>5</td><td></td><td>1.00</td><td></td><td>0.84</td><td>&lt; 0.001</td></tr><tr><td>Linear · positional + k-mer 6</td><td>Adjacent pairs made only of C and G</td><td>cluster 7</td><td>3</td><td></td><td>1.00</td><td></td><td>0.84</td><td>&lt; 0.001</td></tr><tr><td>Tree · programmatic, joint</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>CCG and CGG trinucleotides</td><td>cluster 7</td><td>3</td><td></td><td>1.00</td><td></td><td>0.84</td><td>&lt; 0.001</td></tr><tr><td>Additive · programmatic, one-shot</td><td>Longest run of C and G</td><td>cluster 7</td><td>2</td><td></td><td>1.00</td><td></td><td>0.83</td><td>&lt; 0.001</td></tr><tr><td>Additive - programmatic, iterative</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Linear - programmatic, one-shot</td><td>Stretches holding no CpG</td><td>cluster 1</td><td>3</td><td></td><td>1.00</td><td></td><td>0.17</td><td>&lt; 0.001</td></tr><tr><td>Linear - programmatic, iterative</td><td>The best single position, T at 129</td><td></td><td>2</td><td></td><td>0.12</td><td></td><td>0.44</td><td>&lt; 0.001</td></tr></table>

![](images/5533fa9283fa1ddd4a60c171e38460aee65d8aca1e5acc4fc0c7c10d822e266c.jpg)  
Figure 2 Polycomb nucleation. a, The task. Segments of mouse Polycomb domains are labeled by how fast H3K27me3 returns after replication, the fastest-recovering cluster of segments (cluster 7) against the slowest (cluster 1), and each model sees 200 nt of genomic DNA centered on the segment. b, Every interpretable configuration reduced from all the features it selected down to ten. Curves are validation AUROC recorded during the reduction. Dots at the right are hold-out AUROC of the ten-feature model, and the open diamond is the best gradient-boosted configuration at ten features. c, The decision tree with programmatic features written at each split. Boxes give the training segments reaching them and the share of cluster 7 among those segments, also reflected in the shading. d, Every per-position indicator the additive model could choose from, laid along the window: the proportion of 100 half-samples that returned it (measure of robustness), and how far it separates the hold-out segments as a score on its own, beside both numbers for CpG density. e, Concepts reached by more than one configuration. A link joins each configuration to the concepts among its ten features. Each row gives how many configurations reached the concept, the proportion of half-samples that returned (measure of robustness), how far it separates the hold-out segments on its own, and the q-value of that separation after correction for the whole pool. A grey row fails at least one tier.

configurations placed a measure of CpG density among their ten features: the decision tree with joint programmatic features, both additive and both sparse linear models with programmatic features, and the sparse linear model with positional and k-mer features. Only the three configurations restricted to per-position indicators, which cannot express a count, did not. Between them the six wrote the same quantity in eight diferent forms, from a plain CpG count and a CpG density to the observedover-expected CpG ratio and the number of distinct CpG-containing 6-mers, which the reliability panel grouped into one concept because they compute near-identical values on the same sequences, and every one of those forms proved reliable in our robustness analysis. Moreover, CpG density also separates the hold-out segments at 0.84 AUROC on its own (Fig. 2e). SeqMaestro builds on this and identifies (using a decision tree) a simple condition that enables us to locate cluster 7 segments with very high precision. If there are more than seven CpG dinucleotides in the 200 bases and the fraction of G and C together in the central 60 bases is above 49% then 97% of training segments belong to cluster 7 (Fig. 2c). The reduction to ten features then allows us to check whether the signal is spread out over the 200-base window or localized at a specific position. After the reduction, programmatic features hold their score while models depending only on positional features fall from about 0.81 to about 0.64 (Fig. 2b), and none of the per-position indicators is returned reliably in our robustness analysis (Fig. 2d). Thus, what marks a cluster 7 segment within these 200 bases is spread across the window, with no particular coordinate highly influencing it.

The association between Polycomb recruitment and unmethylated CpG islands is among the best established in chromatin biology. KDM2B binds unmethylated CpG and recruits a variant PRC1 complex, whose H2A ubiquitylation in turn recruits PRC2 [15, 16], and GC-rich elements are suficient to nucleate H3K27me3 where they are inserted [17]. Recovering that from sequence alone, with no chromatin measurement of any kind, is the starting point, and it is achieved by SeqMaestro. Two aspects of SeqMaestro analysis go beyond this. The comparison here is within domains, so the usual statement that Polycomb targets CpG islands is already true of every segment, and the result says that CpG density also decides where inside a domain the mark is re-established first after replication [18, 19]. That connects the recruitment which establishes a domain to the way the domain is maintained through cell division, and it can be checked directly against maps of nucleation sites defined by PRC2 activity in a follow-up investigation. The second is the shape of the efect. The tree’s rule describes a threshold, and the curve the additive model fits to CpG density rises steeply and then flattens (Appendix K). This leads to the hypothesis that Polycomb nucleation responds to CpG density as a switch rather than a gradient. This could be tested by inserting a series of synthetic elements with CpG density spanning the fitted threshold and measuring whether re-establishment turns on sharply or increases gradually.

## RNA polymerase II pausing

RNA polymerase II does not transcribe at a constant rate. It pauses at particular bases, and mNET-seq [20] detects those pauses as peaks of polymerase density. We took 499,672 sites in expressed human genes, half of them pauses and half drawn from pause-free stretches of the same genes, and gave every model 101 nt of genomic DNA read in the direction of transcription, with the pause base at position −1 and the base after it at +1 (Fig. 3a).

The SeqMaestro feature-model battery shows that, in this task, several informative signals lie at fixed distances from the pause. Every interpretable configuration with ten features achieves AUROC between 0.92 and 0.96 (Fig. 3b). The additive model (GAM) with positional features yielded the clearest insights. Initially, it had 404 per-position indicators and no description of the task. After reduction, all ten features were within positions −10 to +1 (Fig. 3c). In particular, G at the pause base argues for a pause, A at +1 argues strongly against, and G at each of five positions upstream, −2, −3, −7, −8 and −10, argues for a pause.

Several of these features agree with the consensus motifs reported for human pause sites [21, 22] [23]. Most notably, the G at −10 is conserved in the consensus pause element of E. coli [24, 25]. The −10 position corresponds to the first base of the transcription bubble, and translocation requires breaking the rNdN base pair at this position. rGdC is a particularly stable base pair, whose disruption hampers translocation when it is situated at −10. The identification of G at −10 by SeqMaestro (Fig. 3c) as a significant feature of RNA polymerase II pause sites strongly suggests that the same mechanism

## RNA polymerase II pausinga

![](images/6083fc798df95906f30026f930d60361821cfb7ea4f825754d2294934dad6eda.jpg)  
Training / validation / hold-out: 349,769 / 49,968 / 99,935 sites  
Half the sites are paused, by design.

## b Every configuration, cut to ten features

![](images/28293a9a76c8c5ebf8d3f96a7495b1f6a747ac8bdb1276e670e7ec8ac9328001.jpg)

Programmatic features, written by a language model Positional and pattern-based features, enumerated in advance

Gradient-boosted trees, a reference only

## c Where the signal is

Top: how far each base at each position separates the classes on the hold-out sites, alone, for all 404 per-position indicators. Below: the ten of them an additive model kept, and the published motif on the same coordinate.

![](images/b2117d048622d1c48558c7001c42b7abc427cc36ede4d15633ed7bb35c657672.jpg)  
The ten columns the additive model chose from 404, with no description of the task

![](images/01ba781282f2cb1496361a53a832a7ab15cc5eaf0b3d31c27f61221df36b77c0.jpg)

## e What several configurations reached, and how each holds up

A link joins a configuration to each concept its ten-feature model contained.

## d The same rule as a tree, AUROC 0.954

The decision tree with programmatic features written at each split, cut to ten features and drawn to three levels. Each branch carries its condition, and h b h i i i hi i d h h d

![](images/009afbb45f66199ba202d94e47067597df8e8c2ed61535fa2d13e2e4ad420612.jpg)

![](images/0d11328fd7ffbef6f8dbe0d95c96ed6975c5a9ad54dab6020a6a09e1cdff8e7c.jpg)

Figure 3 RNA polymerase II pausing. a, The task. Sites in expressed human genes are labeled by whether mNET-seq detected a pause at the central base, and each model sees a 101 nt read in the direction of transcription, with the pause base, the $3 ^ { \prime }$ end of the nascent RNA, at position −1 and the base after it at +1, as the published pause motifs are numbered. The window runs from −51 to +50. b, Every interpretable configuration reduced to ten features, drawn as in Fig. 2b. c, Top: how far each base at each position separates the classes on the hold-out sites as a score on its own, for all 404 per-position indicators. Middle: the ten indicators the additive model kept, at the positions they name, with the log-odds the model adds when that base is present. Bottom: the pause motif reported for human polymerase, on the same coordinate, which the model matches at all three positions. d, The decision tree with programmatic features written at each split, cut to ten features and drawn to three levels of questions. Each branch carries the condition that leads down it. Each box gives the training sites reaching it and the share of them that are paused, which also sets its colour, and a chevron marks a branch that continues below the drawn depth. e, Concepts reached by more than one configuration, drawn as in Fig. 2e.

of inhibiting translocation contributes to pausing by both the human and the E. coli enzymes. The possible functional significance of G at −2, −3, −7 and −8 in promoting pausing (Fig. 3c) may merit future mechanistic investigation.

The decision tree with programmatic features (Fig. 3d) asks first whether the pause base is G and whether the base at +1 is a pyrimidine, and then recovers a similar enrichment of G over the ten bases upstream of the pause, within the transcription bubble, as a feature of pause sites. In addition, it identified G+C content at positions −26 to −17 as a further predictor of pausing. These positions correspond to the first bases of the RNA to emerge from the exit channel of the polymerase, and one hypothesis is that secondary structure in the emerging transcript contributes to pausing [26]. Every one of these properties passed our robustness checks and holds on the hold-out sites, and those reached by more than one configuration are listed in Fig. 3e.

## eIF4E dependence

The translation initiation factor eIF4E, the cap-binding component of the eIF4F complex, plays a central role in cap-dependent translation and is regulated by nutrient-sensing pathways. In previous work, ribosome profiling was used to identify mRNAs that remain eficiently translated when eIF4E is inhibited (eIF4E-resistant) versus those whose translation is blocked (eIF4E-sensitive) [27]. From this study, we generated two lists of the most eIF4E-resistant and eIF4E-sensitive transcripts, totaling 1,195 transcripts each. Because transcript lengths difer between these groups, and mRNA regions (5<sup>′</sup> UTR, CDS, 3<sup>′</sup> UTR) have diferent regulatory functions [28], evolutionary constraints [29], and relative length and nucleotide composition [30], we focused on three local windows: the thirty nucleotides before the start codon, the first 150 nucleotides of the coding region, and the last 100 nucleotides of the 3<sup>′</sup> UTR (Fig. 4a). The LLM-driven feature generation was not told which factor was involved, precluding any recall of literature on eIF4E.

The Kozak sequence, which comprises the ten nucleotides flanking the AUG start codon, is a wellestablished determinant of translation initiation eficiency, and thus an expected discriminating feature. The canonical Kozak consensus (GCCRCCAUGG) is GC-rich and promotes strong cap-dependent translation. Transcripts with strong Kozak sequences are known to be more resistant to eIF4E inhibition [31]. SeqMaestro identified elevated G and C content in positions 20 to 29 (immediately upstream of the start codon) as a discriminating feature, recovering this expected relationship and validating that the approach identifies functionally relevant sequence determinants.

The most striking finding is that features in the coding sequence, historically associated with translation elongation, emerged as the strongest predictors of eIF4E sensitivity (Fig. 4b). eIF4E-resistant mRNAs were enriched in codons with G or C at the third position (GC3, which Fig. 4e lists in its complementary form, A or T at the third codon base). GC3 content separated hold-out transcripts at AUROC 0.82. GC3 defines optimal codon usage in human cells and is associated with enhanced mRNA stability and translation eficiency [32]. Recent work showed that mRNAs bearing optimal codons preferentially bind the initiation factors eIF4E and eIF4G1 [33]. If codon optimality determines eIF4E resistance through enhanced initiation factor recruitment, we would predict that eIF4E-resistant mRNAs should preferentially associate with eIF4E at baseline. Reanalysis of eIF4E RNA-immunoprecipitation data [27] demonstrated that eIF4E-resistant mRNAs exhibited significantly higher eIF4E binding than sensitive mRNAs and the transcriptome as a whole (Fig. 4f). That SeqMaestro identified GC3 codons as the primary discriminator provides independent support for a mechanism coupling translation elongation kinetics to initiation factor engagement, suggesting that the sequence features that regulate elongation can reciprocally influence how the ribosome recruitment machinery associates with mRNA.

## PTBP1 exon regulation

A cassette exon is one that is spliced into the mature messenger RNA in some transcripts and skipped in others, thereby usually modulating the abundance or sequence of the resulting protein. The balance of transcript isoforms is regulated by RNA-binding proteins (RBPs), which can act to enhance or repress splicing based on the positional relationships between their binding sites and exons [34]. PTBP1 is an important factor in stem cell self-renewal and cell fate determination, where it is well-characterized as a

Split by gene. The three windows are cut from the same transcripts.

## eIF4E dependence

Human transcripts, 1,195 whose translation falls when eIF4E activity is reduced and 1,195 whose translation does not. The components that write features were never told which factor.

![](images/d338606c2243a1e760405f3837de0b42807796370273c7c5e7b6ee91d6fc61c3.jpg)

Programmatic features, written by a language model Positional and pattern-based features, enumerated in advance

## b Three windows, one battery

Gradient-boosted trees, a reference only

![](images/a7d1b27a957bdcc69c2deb507a6c83c93a67b9764595912e52870de87bb3a096.jpg)

## c The coding-window model states the difference in codon terms

![](images/527210cd3883aa2299d057d33ac62d4611deec1664d8f0b9c08d771750aa5d8f.jpg)

Three of the ten curves of the additive model with iterative programmatic features, AUROC 0.872 on the hold-out set. Each curve is the effect of one feature on the log-odds of factor dependence, above where the fitted transcripts fall.

e What several configurations reached in the coding window, and how each holds up A link joins a configuration to each concept its ten-feature model contained.

## d Agreement against support, all three windows

Every concept more than one configuration reached. The mos widely agreed idea in the 5′ UTR window is the one the panel declines to support.

![](images/0a0fcaa95dce58eff74a60207cfdd3c0581ad528e4ea194de006b5c680478fc8.jpg)  
5′ UTR coding 3′ UTR open: fails a reliability tier f Resistant transcripts bind more eIF4E

![](images/7a444bbdfcc150d4a514e1bb0cf673526e64c54915b85b026e41a79f56192fdf.jpg)  
Each transcript's eIF4E pull-down over its input, in vehicle-treated cells. Boxes span the middle half of each group. Wilcoxon test between the two lists.

![](images/23a5f8cbc44096b5ce6b9cffe90fa8183041350482a4b6dd822a1231531239a2.jpg)  
Figure 4 eIF4E dependence, three windows of one transcript set. a, The task. Two lists of human transcripts, one whose translation falls when eIF4E activity is reduced and one whose translation does not, windowed three ways. b, Every interpretable configuration at ten features, one column per window. Marks as in Fig. 2b. c, Three of the ten curves of the additive model fitted to the coding window. Each is the efect of one feature on the log-odds of factor dependence, above a strip showing where the fitted transcripts fall. d, Every concept that more than one configuration reached, in all three windows, placed by how many configurations reached it and how far it separates the hold-out transcripts on its own. Open marks fail at least one tier of the reliability panel. e, Concepts reached by more than one configuration in the coding window, drawn as in Fig. 2e. f, How much eIF4E each transcript binds, measured as its enrichment in an eIF4E immunoprecipitation over the input in vehicle-treated cells, for the eIF4E-sensitive list (the factor-dependent class of a to e), the eIF4E-resistant list (factor-independent) and all other genes, coloured as the classes in a. Boxes span the middle half of each group, with the median across them. The two lists difer by a Wilcoxon test, $p < 2 \times 1 0 ^ { - 1 6 }$ . Reanalysed from the immunoprecipitation data of Roiuk et al. [27].

## PTBP1 exon regulation

Human K562 cells, GRCh38. 883 cassette exons that PTBP1 regulates, labelled by the direction in which PTBP1 acts on them.

![](images/4478f40df427504d44770900483d7c1d9a375975470f60ddb60ea615f19fb358.jpg)

enhanced PTBP1 promotes the exon: inclusion falls when PTBP1 is removed

## b Every configuration, cut to ten features

![](images/38879751b0ef279e067ffa075deb4409e4c4256819827fa9800dfbefe8ae12f6.jpg)

Training / validation / hold-out: 618 / 87 / 178 exons, every gene kept on one side. 47% silenced, as the experiment found them.

Programmatic features, written by a language model Positional and pattern-based features, enumerated in advance

Gradient-boosted trees, a reference only

## c Where the additive model looks, AUROC 0.875 on the hold-out set

Its ten programmatic features, each drawn over the stretch of the window it measures and coloured by the class it argues for. Grey features are used in both directions. Below, two of its curves.

![](images/43185335d946ada03882f943f34f5bbf0d0c028904f073dbe1cd2f2889a2b5ad.jpg)

![](images/19b2814cc9e94940f6f601d29ef4e8bd2f127bc96618b7fd9efff8e7042c033a.jpg)

![](images/8fbad0b4eb6162088521d18f8620a18f9197fc95603af1067a88000cd1d7c3c1.jpg)

## d The asymmetry as one threshold, AUROC 0.725

The decision tree with programmatic features written at each split. Boxes give the training exons reaching them and the share silenced.

![](images/9c322410c8d48d47da6fe157a4cdfee314473267c00fe0a7be3016150a0c7436.jpg)

![](images/89d947f204b9706df750fb10cebb88fb70f82a1361f1d5001d58b95cc8afb8a1.jpg)

![](images/8c65af7f45dd78822701ea582f78a87716d772b3de9d12ddde8be0db4a8be20f.jpg)  
Figure 5 PTBP1 exon regulation. a, The task. Cassette exons that PTBP1 regulates are labeled by the direction of its efect, and each model sees 100 nt of intron on either side of the exon, joined, with the exon itself removed. b, Every interpretable configuration reduced to ten features, drawn as in Fig. 2b. c, The ten programmatic features of the additive model, each drawn over the stretch of the window it measures and coloured by the class it argues for, with two of its curves beneath. d, The decision tree with programmatic features written at each split, to two levels. e, Concepts reached by more than one configuration, drawn as in Fig. 2e.

master regulator of neuronal development [35]. It is also broadly expressed across many adult tissues, with roles in cancer, metabolism and immune responses. PTBP1 is known to repress splicing by binding to pyrimidine-rich sequence elements upstream of the 3<sup>′</sup> splice site (3<sup>′</sup>SS) and enhance exon inclusion when binding such elements downstream of the 5<sup>′</sup> splice site (5<sup>′</sup>SS) [36–38]. As the combination of motifs and position encodes the regulation, we were curious if SeqMaestro could learn this regulation de novo. 883 cassette exons were annotated as either silenced by PTBP1 (420), or enhanced by PTBP1 (463) [39]. SeqMaestro was challenged to classify exons as either enhanced or silenced by PTBP1 based on 100 nt upstream and downstream flanking intron sequence alone (Fig. 5a).

This is the task with the widest gap between the two kinds of feature sources (positional and patternbased versus programmatic). The best ten-feature model on programmatic features reaches AUROC of 0.875, and the best on positional and pattern-based features 0.656 (Fig. 5b). The signal is regional pyrimidine content weighted toward the exon, which is challenging to express with ten positional features and short motifs. However, programmatic features can capture it. The additive model lays its ten features out on the two introns as a map (see Fig. 5c). Pyrimidine content upstream of the exon, measured at five nested scales (the pyrimidine fraction of the whole 100 nt, the CU dinucleotide fraction of the last 80 nt, the count of the PTBP1-like trimers CUU/UCU/CUC/UCC in the last 70 nt, CU-rich hexamers in the last 60 nt, and the U fraction of the last 30 nt), argues in every case that PTBP1 silences the exon, and G content in the first 25 nt downstream argues that it enhances it. The decision tree reduces that map to one question: whether there are more than four more CU dinucleotides upstream than downstream, which on its own takes the silenced ratio from 48% to 79% (Fig. 5d). This captures what we know about PTBP1 CU-motif binding being repressive at the 3<sup>′</sup>SS, whilst promoting inclusion at the 5<sup>′</sup>SS, but in a simple, quantitative way that could be applied to sequence design. Three properties clear every tier of the reliability panel (Fig. 5e): (i) the count of the PTBP1-like pyrimidine trimers CUU, UCU, CUC and UCC in the 70 nt before the exon, reached by five configurations across all three model families; (ii) the diference between CU content upstream and downstream of the exon, reached by four; and (iii) the count of CU-rich hexamers in the 60 nt before the exon, reached by three. They also separate the hold-out exons at 0.81 to 0.85 on their own.

In conclusion, SeqMaestro is able to condense our knowledge of PTBP1 binding principles into a simple series of quantitative rules, derived from raw sequence de novo. Further, the model extracts several novel features that warrant further investigation. Firstly, that G-richness in the first 25 nt downstream of the 5<sup>′</sup>SS is predictive of PTBP1 enhanced exons. One hypothesis is that this feature could represent stabilization of weak 5<sup>′</sup>SS by binding of RBPs such as hnRNPH/F in the presence of PTBP1. It has been previously shown that PTBP1 can recruit hnRNPH to G-rich sequences in 3<sup>′</sup> UTRs to regulate polyadenylation [40]. Secondly, that CU upstream minus UG downstream is predictive of silencing, suggesting that RBPs binding to downstream UG repeats could negate the impact of PTBP1 binding at upstream CU-rich stretches.

Taken together, these four tasks demonstrate SeqMaestro’s ability to deliver interpretable models and to discover robust sequence properties that predict the target. On each, it delivered an interpretable model with at most ten features and a short list of properties that emerged from several independent configurations and hold on independent sequences. The lists contain known biology, recovered without human prompting: e.g., from simple positional features for pausing, and from LLM-driven feature generators that were not told the factor’s name for eIF4E. Beyond the known biology, every task also yielded at least one lead of its own: a threshold to construct for nucleation, a translocation mechanism shared with E. coli for pausing, a coupling of codon optimality to eIF4E recruitment for eIF4E, and two candidate co-regulators binding the downstream intron for PTBP1. Those are the starting points for the next experiment, computational or at the bench, and they were produced from sequence and labels alone.

## Discussion

SeqMaestro provides a framework for turning nucleotide sequence prediction problems into a search over explicit, human-readable biological hypotheses. Rather than committing in advance to a single representation of sequence or explaining a complex predictive model after it has been fitted, SeqMaestro searches across multiple interpretable representations, feature-generation strategies, and transparent model classes. Across the four biological questions considered here, this search produced compact predictive models and a small set of sequence properties that could be inspected directly and evaluated for robustness. The central contribution is therefore not simply an additional sequence classifier but an interface through which the broader ecosystem of interpretable machine learning can be applied to nucleotide sequence analysis. In this formulation, both the sequence property being measured and the predictive relationship in which it participates can form part of the resulting hypothesis.

The results also illustrate why no single interpretable representation is likely to be suficient across sequence analysis problems. Positional representations are well suited to signals tied to specific locations relative to a landmark, whereas pattern-based representations provide a compact vocabulary for recurring local sequence elements. Programmatic features extend this vocabulary to quantities that combine regions, positions, counts, composition, or other sequence properties within a single interpretable computation. Importantly, the more expressive representation was not uniformly preferable. In some tasks, simple positional features were already suficient to capture most of the predictive signal, whereas in others, programmatic features retained substantially more predictive information when models were reduced to a small number of features. This suggests that the choice of representation should itself be treated as part of the hypothesis search rather than as a preprocessing decision made before analysis.

The same argument applies to the transparent model class. Diferent model families expose diferent aspects of the relationship between a sequence property and the target. Sparse linear models provide additive efects with an explicit direction and magnitude; additive models can reveal gradients, thresholds, saturation, or reversals across the range of a feature; and decision trees can express conditional rules in which a sequence property matters only when another condition is satisfied. SeqMaestro therefore goes beyond identifying which features are important. A biological hypothesis may instead concern the shape of an efect, a threshold, or an interaction between interpretable sequence properties. This distinction is important because a ranked feature list can identify a predictive signal while leaving much of its predictive logic unspecified. By keeping the fitted model itself interpretable, SeqMaestro allows that logic to remain part of the scientific output.

Large language models play a deliberately restricted role in this process. They are used to propose programmatic features and, in some configurations, to refine those proposals using feedback from fitted models. They do not determine whether a feature is retained, whether it predicts the outcome, or whether it passes the reliability analysis. Those decisions are made using the observed data and fixed statistical procedures. The framework also records which information was exposed to the feature generator, enabling the withholding of identifying biological context when the goal is to distinguish discovery from the reproduction of prior knowledge.

SeqMaestro is intended to generate predictive and testable hypotheses, not to establish causal mechanisms directly. A sequence property that is reproducibly associated with a label may itself be causal, may serve as a proxy for another sequence property, or may reflect a confounding aspect of how the dataset was constructed. None of the reliability tiers can distinguish these possibilities in general. The same limitation applies to any supervised analysis of observational sequence data and places particular importance on the construction of the prediction task. Several additional limitations define the current scope of the framework. The implementation considered here addresses binary classification, although regression, multiclass, and multilabel settings are natural extensions. Stability under resampling of one dataset is weaker evidence than replication across independent datasets, conditions, species, or laboratories. Finally, the experiments in this study are intended to evaluate interpretable hypothesis search rather than to benchmark SeqMaestro against the full range of modern sequence foundation models or highly optimized deep sequence predictors. The gradient-boosted models provide a reference within the feature spaces considered here, but they do not constitute such a comparison.

These limitations also suggest several directions for extension. Additional transparent model families could broaden the forms of predictive relationships that can be expressed. Reliability analysis could incorporate replication across independent datasets as a further level of evidence. More broadly, the same architecture could support iterative experimental workflows in which hypotheses generated from one dataset are tested„ and the resulting measurements serve as the starting point for the next search.

SeqMaestro thus provides a route from sequence prediction toward a more explicit form of computational hypothesis generation, in which the output of machine learning is not only a prediction or an attribution score, but a compact, computable statement about nucleotide sequence that can be inspected, challenged, and tested.

## References

[1] Stephen F Altschul, Warren Gish, Webb Miller, Eugene W Myers, and David J Lipman. Basic local alignment search tool. Journal of Molecular Biology, 215(3):403–410, 1990.

[2] Timothy L Bailey, Nadya Williams, Chris Misleh, and Wilfred W Li. MEME: Discovering and analyzing DNA and protein sequence motifs. Nucleic Acids Research, 34(suppl\_2):W369–W373, 2006.

[3] Chris Burge and Samuel Karlin. Prediction of complete gene structures in human genomic DNA. Journal of Molecular Biology, 268(1):78–94, 1997.

[4] Camille Moeckel, Manvita Mareboina, Maxwell A Konnaris, Candace S Y Chan, Ioannis Mouratidis, Austin Montgomery, Nikol Chantzi, Georgios A Pavlopoulos, and Ilias Georgakopoulos-Soares. A survey of k-mer methods and applications in bioinformatics. Computational and Structural Biotechnology Journal, 23:2289–2303, 2024.

[5] Mahmoud Ghandi, Dongwon Lee, Morteza Mohammad-Noori, and Michael A Beer. Enhanced regulatory sequence prediction using gapped k-mer features. PLoS Computational Biology, 10(7): e1003711, 2014.

[6] Jian Zhou and Olga G Troyanskaya. Predicting efects of noncoding variants with deep learning– based sequence model. Nature Methods, 12(10):931–934, 2015.

[7] Žiga Avsec, Vikram Agarwal, Daniel Visentin, Joseph R Ledsam, Agnieszka Grabska-Barwinska, Kyle R Taylor, Yannis Assael, John Jumper, Pushmeet Kohli, and David R Kelley. Efective gene expression prediction from sequence by integrating long-range interactions. Nature Methods, 18 (10):1196–1203, 2021. doi: https://doi.org/10.1038/s41592-021-01252-x.

[8] Hugo Dalla-Torre, Liam Gonzalez, Javier Mendoza-Revilla, Nicolas Lopez Carranza, Adam Henryk Grzywaczewski, Francesco Oteri, Christian Dallago, Evan Trop, Bernardo P De Almeida, Hassan Sirelkhatim, et al. Nucleotide Transformer: Building and evaluating robust foundation models for human genomics. Nature Methods, 22(2):287–297, 2025. doi: https://doi.org/10.1038/s41592-024-0 2523-z.

[9] Cynthia Rudin. Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead. Nature Machine Intelligence, 1(5):206–215, 2019. doi: 10.1038/s42256-019-0048-x.

[10] Julius Adebayo, Justin Gilmer, Michael Muelly, Ian Goodfellow, Moritz Hardt, and Been Kim. Sanity checks for saliency maps. Advances in Neural Information Processing Systems, 31, 2018.

[11] Amirata Ghorbani, Abubakar Abid, and James Y. Zou. Interpretation of neural networks is fragile. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 33, pages 3681–3688. AAAI Press, 2019. doi: 10.1609/aaai.v33i01.33013681.

[12] Valerie Chen, Muyu Yang, Wenbo Cui, Joon Sik Kim, Ameet Talwalkar, and Jian Ma. Applying interpretable machine learning in computational biology—pitfalls, recommendations and opportunities for new developments. Nature Methods, 21(8):1454–1461, 2024.

[13] Nicolas Huynh, Krzysztof Kacprzyk, Ryan Sheridan, David Bentley, and Mihaela van der Schaar. Interpretable DNA sequence classification via dynamic feature generation in decision trees. arXiv, 2604.12060, 2026. doi: 10.48550/arXiv.2604.12060. Proceedings of the 29th International Conference on Artificial Intelligence and Statistics (AISTATS).

[14] Giovana M. B. Veronezi and Srinivas Ramachandran. Nucleation and spreading maintain Polycomb domains every cell cycle. Cell Reports, 43(4):114090, 2024. doi: 10.1016/j.celrep.2024.114090.

[15] Anca M Farcas, Neil P Blackledge, Ian Sudbery, Hannah K Long, Joanna F McGouran, Nathan R Rose, Sheena Lee, David Sims, Andrea Cerase, Thomas W Sheahan, Haruhiko Koseki, Neil Brockdorf, Chris P Ponting, Benedikt M Kessler, and Robert J Klose. KDM2B links the Polycomb repressive complex 1 (PRC1) to recognition of CpG islands. eLife, 1:e00205, 2012. doi: 10.7554/el ife.00205.

[16] Neil P. Blackledge, Anca M. Farcas, Takashi Kondo, Hamish W. King, Joanna F. McGouran, Lars L.P. Hanssen, Shinsuke Ito, Sarah Cooper, Kaori Kondo, Yoko Koseki, Tomoyuki Ishikura, Hannah K. Long, Thomas W. Sheahan, Neil Brockdorf, Benedikt M. Kessler, Haruhiko Koseki, and Robert J. Klose. Variant PRC1 complex-dependent H2A ubiquitylation drives PRC2 recruitment and Polycomb domain formation. Cell, 157(6):1445–1459, 2014. doi: 10.1016/j.cell.2014.05.004.

[17] Eric M. Mendenhall, Richard P. Koche, Thanh Truong, Vicky W. Zhou, Biju Issac, Andrew S. Chi, Manching Ku, and Bradley E. Bernstein. GC-rich sequence elements recruit PRC2 in mammalian ES cells. PLoS Genetics, 6(12):e1001244, 2010. doi: 10.1371/journal.pgen.1001244.

[18] Nazaret Reverón-Gómez, Cristina González-Aguilera, Kathleen R. Stewart-Morgan, Nataliya Petryk, Valentin Flury, Simona Graziano, Jens Vilstrup Johansen, Janus Schou Jakobsen, Constance Alabert, and Anja Groth. Accurate recycling of parental histones reproduces the histone modification landscape during DNA replication. Molecular Cell, 72(2):239–249.e5, 2018. doi: 10.1016/j.molcel.2 018.08.010.

[19] Ozgur Oksuz, Varun Narendra, Chul-Hwan Lee, Nicolas Descostes, Gary LeRoy, Ramya Raviram, Lili Blumenberg, Kelly Karch, Pedro P. Rocha, Benjamin A. Garcia, Jane A. Skok, and Danny Reinberg. Capturing the onset of PRC2-mediated repressive domain formation. Molecular Cell, 70 (6):1149–1162.e5, 2018. doi: 10.1016/j.molcel.2018.05.023.

[20] Takayuki Nojima, Tomás Gomes, Ana Rita Fialho Grosso, Hiroshi Kimura, Michael J. Dye, Somdutta Dhir, Maria Carmo-Fonseca, and Nicholas J. Proudfoot. Mammalian NET-seq reveals genome-wide nascent transcription coupled to RNA processing. Cell, 161(3):526–540, 2015. doi: 10.1016/j.cell.2015.03.027.

[21] Martyna Gajos, Olga Jasnovidova, Alena van Bömmel, Susanne Freier, Martin Vingron, and Andreas Mayer. Conserved DNA sequence features underlie pervasive RNA polymerase pausing. Nucleic Acids Research, 49(8):4402–4420, 2021. doi: 10.1093/nar/gkab208.

[22] Ryan M. Sheridan, Nova Fong, Angelo D’Alessandro, and David L. Bentley. Widespread backtracking by RNA Pol II is a major efector of gene activation, 5’ pause release, termination, and transcription elongation rate. Molecular Cell, 73(1):107–118.e4, 2019. doi: 10.1016/j.molcel.2018.10.031.

[23] Nova Fong, Ryan M Sheridan, Srinivas Ramachandran, and David L Bentley. The pausing zone and control of RNA polymerase II elongation by Spt5: Implications for the pause-release model. Molecular Cell, 82(19):3632–3645.e4, 2022.

[24] Matthew H. Larson, Rachel A. Mooney, Jason M. Peters, Tricia Windgassen, Dhananjaya Nayak, Carol A. Gross, Steven M. Block, William J. Greenleaf, Robert Landick, and Jonathan S. Weissman. A pause sequence enriched at translation start sites drives transcription dynamics in vivo. Science, 344(6187):1042–1047, 2014. doi: 10.1126/science.1251871.

[25] Irina O. Vvedenskaya, Hanif Vahedian-Movahed, Jeremy G. Bird, Jared G. Knoblauch, Seth R. Goldman, Yu Zhang, Richard H. Ebright, and Bryce E. Nickels. Interactions between RNA polymerase and the “core recognition element” counteract pausing. Science, 344(6189):1285–1289, 2014. doi: 10.1126/science.1253458.

[26] Jin Young Kang, Tatiana V Mishanina, Michael J Bellecourt, Rachel Anne Mooney, Seth A Darst, and Robert Landick. RNA polymerase accommodates a pause RNA hairpin by global conformational rearrangements that prolong pausing. Molecular Cell, 69(5):802–815.e5, 2018.

[27] Mykola Roiuk, Marilena Nef, and Aurelio A Teleman. eIF4E-independent translation is largely eIF3d-dependent. Nature Communications, 15(1):6692, 2024.

[28] Flavio Mignone, Carmela Gissi, Sabino Liuni, and Graziano Pesole. Untranslated regions of mRNAs. Genome Biology, 3(3):REVIEWS0004, 2002. doi: 10.1186/gb-2002-3-3-reviews0004.

[29] Svetlana A Shabalina, Aleksey Y Ogurtsov, Igor B Rogozin, Eugene V Koonin, and David J Lipman. Comparative analysis of orthologous eukaryotic mRNAs: Potential hidden functional signals. Nucleic Acids Research, 32(5):1774–1782, 2004.

[30] G Pesole, F Mignone, C Gissi, G Grillo, F Licciulli, and S Liuni. Structural and functional features of eukaryotic mRNA untranslated regions. Gene, 276(1-2):73–81, 2001.

[31] Julieta M Acevedo, Bernhard Hoermann, Tilo Schlimbach, and Aurelio A Teleman. Changes in global translation elongation or initiation rates shape the proteome via the Kozak sequence. Scientific Reports, 8(1):4018, 2018.

[32] Fabian Hia, Sheng Fan Yang, Yuichi Shichino, Masanori Yoshinaga, Yasuhiro Murakawa, Alexis Vandenbon, Akira Fukao, Toshinobu Fujiwara, Markus Landthaler, Tohru Natsume, Shungo Adachi, Shintaro Iwasaki, and Osamu Takeuchi. Codon bias confers stability to human mRNAs. EMBO Reports, 20(11):e48220, 2019. doi: 10.15252/embr.201948220.

[33] Chloe L Barrington, Gabriel Galindo, Amanda L Koch, Emma R Horton, Evan J Morrison, Samantha Tisa, Timothy J Stasevich, and Olivia S Rissland. Synonymous codon usage regulates translation initiation. Cell Reports, 42(12):113413, 2023. doi: 10.1016/j.celrep.2023.113413.

[34] Jernej Ule and Benjamin J Blencowe. Alternative splicing regulatory networks: Functions, mechanisms, and evolution. Molecular Cell, 76(2):329–345, 2019.

[35] Jing Hu, Hao Qian, Yuanchao Xue, and Xiang-Dong Fu. PTB/nPTB: Master regulators of neuronal fate in mammals. Biophysics Reports, 4(4):204–214, 2018.

[36] Fursham M Hamid and Eugene V Makeyev. A mechanism underlying position-specific regulation of alternative splicing. Nucleic Acids Research, 45(21):12455–12468, 2017.

[37] Miriam Llorian, Schraga Schwartz, Tyson A Clark, Dror Hollander, Lit-Yeen Tan, Rachel Spellman, Adele Gordon, Anthony C Schweitzer, Pierre de la Grange, Gil Ast, and Christopher W J Smith. Position-dependent alternative splicing activity revealed by global profiling of alternative splicing events regulated by PTB. Nature Structural & Molecular Biology, 17(9):1114–1123, 2010. doi: 10.1038/nsmb.1881.

[38] Yuanchao Xue, Yu Zhou, Tongbin Wu, Tuo Zhu, Xiong Ji, Young-Soo Kwon, Chao Zhang, Gene Yeo, Douglas L. Black, Hui Sun, Xiang-Dong Fu, and Yi Zhang. Genome-wide analysis of PTB-RNA interactions reveals a strategy used by the general splicing repressor to modulate exon inclusion or skipping. Molecular Cell, 36(6):996–1006, 2009. doi: 10.1016/j.molcel.2009.12.003.

[39] Serge Gueroussov, Thomas Gonatopoulos-Pournatzis, Manuel Irimia, Bushra Raj, Zhen-Yuan Lin, Anne-Claude Gingras, and Benjamin J. Blencowe. An alternative splicing event amplifies evolutionary diferences between vertebrates. Science, 349(6250):868–873, 2015. doi: 10.1126/scie nce.aaa8381.

[40] Stefania Millevoi, Adrien Decorsière, Clarisse Loulergue, Jason Iacovoni, Sandra Bernat, Michael Antoniou, and Stéphan Vagner. A physical and functional link between splicing factors promotes pre-mRNA 3’ end processing. Nucleic Acids Research, 37(14):4672–4683, 2009.

[41] Fabian Pedregosa, Gaël Varoquaux, Alexandre Gramfort, Vincent Michel, Bertrand Thirion, Olivier Grisel, Mathieu Blondel, Peter Prettenhofer, Ron Weiss, Vincent Dubourg, Jake Vanderplas, Alexandre Passos, David Cournapeau, Matthieu Brucher, Matthieu Perrot, and Édouard Duchesnay. Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12:2825–2830, 2011.

[42] Yin Lou, Rich Caruana, and Johannes Gehrke. Intelligible models for classification and regression. In Proceedings of the 18th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pages 150–158. ACM, 2012. doi: 10.1145/2339530.2339556.

[43] Yin Lou, Rich Caruana, Johannes Gehrke, and Giles Hooker. Accurate intelligible models with pairwise interactions. In Proceedings of the 19th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pages 623–631. ACM, 2013. doi: 10.1145/2487575.2487579.

[44] Harsha Nori, Samuel Jenkins, Paul Koch, and Rich Caruana. InterpretML: A unified framework for machine learning interpretability. arXiv, 1909.09223, 2019. doi: 10.48550/arXiv.1909.09223.

[45] Tianqi Chen and Carlos Guestrin. XGBoost: A scalable tree boosting system. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pages 785–794. ACM, 2016. doi: 10.1145/2939672.2939785.

[46] Gregory Faletto and Jacob Bien. Cluster stability selection. arXiv, 2201.00494, 2022. doi: 10.48550/arXiv.2201.00494.

[47] Rajen D. Shah and Richard J. Samworth. Variable selection with error control: Another look at stability selection. Journal of the Royal Statistical Society Series B: Statistical Methodology, 75(1): 55–80, 2013. doi: 10.1111/j.1467-9868.2011.01034.x.

[48] H. B. Mann and D. R. Whitney. On a test of whether one of two random variables is stochastically larger than the other. The Annals of Mathematical Statistics, 18(1):50–60, 1947. doi: 10.1214/ao ms/1177730491.

[49] Belinda Phipson and Gordon K. Smyth. Permutation P-values should never be zero: Calculating exact P-values when permutations are randomly drawn. Statistical Applications in Genetics and Molecular Biology, 9(1):Article 39, 2010. doi: 10.2202/1544-6115.1585.

[50] Yoav Benjamini and Yosef Hochberg. Controlling the false discovery rate: A practical and powerful approach to multiple testing. Journal of the Royal Statistical Society Series B: Statistical Methodology, 57(1):289–300, 1995. doi: 10.1111/j.2517-6161.1995.tb02031.x.

[51] Yoav Benjamini and Daniel Yekutieli. The control of the false discovery rate in multiple testing under dependency. The Annals of Statistics, 29(4):1165–1188, 2001. doi: 10.1214/aos/1013699998.

[52] Peter H. Westfall and S. Stanley Young. Resampling-based multiple testing: Examples and methods for P-value adjustment. Wiley, New York, 1993.

[53] John D. Storey and Robert Tibshirani. Statistical significance for genomewide studies. Proceedings of the National Academy of Sciences, 100(16):9440–9445, 2003. doi: 10.1073/pnas.1530509100.

[54] Jing Lei, Max G’Sell, Alessandro Rinaldo, Ryan J. Tibshirani, and Larry Wasserman. Distributionfree predictive inference for regression. Journal of the American Statistical Association, 113(523): 1094–1111, 2018. doi: 10.1080/01621459.2017.1307116.

[55] Richard Bourgon, Robert Gentleman, and Wolfgang Huber. Independent filtering increases detection power for high-throughput experiments. Proceedings of the National Academy of Sciences, 107(21): 9546–9551, 2010. doi: 10.1073/pnas.0914005107.

[56] James A. Hanley and Barbara J. McNeil. The meaning and use of the area under a receiver operating characteristic (ROC) curve. Radiology, 143(1):29–36, 1982. doi: 10.1148/radiology.143.1.7063747.

[57] Sture Holm. A simple sequentially rejective multiple test procedure. Scandinavian Journal of Statistics, 6(2):65–70, 1979.

## Methods

## Overview

SeqMaestro is given a set of nucleotide sequences, each labelled with one of two classes, together with a short description of what those classes mean and how they were measured. It fits a range of interpretable models to the sequences, records which properties of a sequence each model relies on, and then measures how well each of those properties is supported. The result is a report: every model that was fitted and how it scored, the small interpretable models presented feature by feature, and a panel of reliability analysis of features and concepts found.

Since somebody who wants to score new sequences needs the most accurate small model, yet somebody designing an experiment or follow-up investigation is interested in the properties that several independent methods converged on reliably, SeqMaestro’s report covers both of these parallel goals.

The work divides into three stages, shown in Figure 6. Figure 7 in Appendix G builds on it by following individual features through the same stages, whether they were derived from the data alone or proposed by the language model, up to the point where they reach the report.

1. Stage 1 turns the user-provided data into a prediction problem. It settles what is being predicted, from which part(s) of sequence, and how the sequences are divided between the training, validation, and hold-out sets.

2. Stage 2 pairs each of the five sequence feature representations with every compatible model family, and fits all the combinations, forming a representation-model “battery” of configurations. This is executed and systematically analysed to gather evidence and identify the signal carried by the sequences.

3. Stage 3 narrows the results to a tractable report. Stage 3A reduces each fitted model down to a small number of features (10, by default) and presents the best-performing small, interpretable models. Stage 3B, the reliability panel, then investigates whether every candidate feature would be found again if the analysis was repeated (tier 1), whether it separates the classes on sequences that weren’t used to fit the models (tier 2), and, finally, whether the small model independently needs the feature.

Three principles apply at every stage.

Sequences are held out until the end. A test set portion of the sequences is set aside at the start and is unavailable to everything that follows. When an agent drives the run, the withholding is enforced by what is made available to it (e.g. the folder it reads after Stage 1 contains no test set or scores computed on it). The set is unsealed once, after the models of Stage 3A and the stability tier of Stage 3B have been fixed. A single final pass then computes every hold-out quantity in the report: the scores of the models, and Tiers 2 and 3 of the panel that require measuring how a feature fares on unseen sequences.

A shufled-label control is applied throughout. Alongside each real fit we run a matched shufledlabel “twin”, which difers in that the training labels have been permuted. This control sets the minimum bar for any real data-driven result. It matters also for the programmatic features written by language models, as such a model may propose a plausible set of features regardless of the labels. Thus, it is important to evaluate how much better the model performs once the actual labels are used.

Each stage ends at a checkpoint. A checkpoint is a list of conditions that are verified automatically, together with a short written summary for the user. The first checkpoint asks for the user’s approval, as the choices made in Stage 1 afect what every later stage can possibly find, and hence corrections and direction by a domain expert user are especially important at this stage. The remaining three stages run without interruption, unless interrupted by the user.

Table 1, in Appendix A, contains further details that supplement this section.

![](images/05b8da38d91eb0c0b00621ccc7368d126505ecdc35d55b40103b3167e1ecd4e6.jpg)  
Figure 6 A SeqMaestro run. (a) The three stages are driven either by a tool-using AI agent, given written instructions for each stage, the SeqMaestro library, and a workspace folder, or run as a single command with sensible, documented defaults. Both routes follow these stages. (b) Stage 1 turns delivered data into a prediction problem, based on user interaction: the choice of sequence windows, and where they are aligned; provenance and composition of the data is checked; every filtered or discarded sequence is accounted for; and sequence stratification and grouping are defined. A simple composition-only model is recorded as a baseline. Finally, the user is asked to check and sign of this work at the checkpoint. (c) Stage 2 fits every pairing of a sequence feature representation with a compatible model, including a shufled-label control. Key properties of discovered features are recorded in a standardized format. (d) Stage 3A reduces each configuration to a fixed number of features and records the best performing interpretable model of each kind. (e) Stage 3B evaluates every candidate feature whether the selection would return it on another half of the data, whether it separates the classes on unseen sequences, and whether the model loses accuracy when the feature is individually discarded. The hold-out set is opened the first of those three questions is settled. The other two (and every hold-out score in the report) are evaluated at the end. (f) A run produces a written report, which discusses both the best interpretable models, and the robust concepts/hypotheses discovered.

## Stage 1: building the prediction task

Three parts of SeqMaestro have to explain the data to a language model. They are the component that proposes features, the decision tree that proposes features while it grows, and the rating step noted below. All three are handed a short description of the dataset, which describes what is being predicted, what the sequences are, and optional additional points of note supplied by the user. The user may also choose to not reveal the nature of the label (e.g. to withhold the protein name) from the LLM feature generating components of SeqMaestro. Stage 1 ensures that the description is settled with the user’s input.

Each sequence in the analysis requires a window of fixed length, set at a fixed distance from a landmark chosen for biological reasons (this might be a splice site, a start codon, the position at which some event was measured, or simply a midpoint of an interval when the sequence has no orientation). Anchoring the window this way makes a given position correspond to the same kind of location in every sequence, as otherwise “position 43”, for instance, would be ambiguous. Where the window cannot be derived from every user-provided sequence, SeqMaestro counts what was lost and reports it separately for each class. A window of fixed size applied to sequences of unequal length may afect the class balance – length is seldom unrelated to the label, so a step that removes a third of one class and a twentieth of the other afects the task setup – and it is crucial that this information is tracked and presented to the user.

SeqMaestro then checks the sequences themselves, before anything at all is fitted. A failing check here halts the run, and reverts to the user for extra data or clarification. The first check looks a sample of windows up in the reference genome (where available), at the coordinates of the sequences, and verifies the strand convention, so the genome build and the strand orientation are established. The second check covers the alphabet (A, T, G, C, U, and so on), the length, and whether any window appears to have erroneous duplicate copies. Another check concerns what was filtered: every step that removes sequences has to record what it removed and how much of each class was lost.

Sequences are then assigned to the training, validation and hold-out (test) sets in groups. A group is whatever makes two sequences dependent on one another, which might be a gene, a domain, a clonal family, etc., depending on the research problem. This ensures fair comparison across the two classes. The variable used for grouping is recorded and this grouping is programmatically enforced in the data splits throughout the SeqMaestro run.

Stage 1 also sets out a simple compositional baseline. On the sequences and splits the models will use, we compute the fraction of each base, the fraction of G or C, the fraction of C or T, and the counts of sixteen adjacent base pairs, fitting each family with an unpenalised logistic regression, and pick the strongest of these.

The summary the user approves at the first checkpoint states all of the key decision points discussed above with a note on what would change depending on the user’s answer. In agentic mode, the user may repeatedly provide additional data, or make clarifications, before the stage is re-run and a final study design settled.

## Stage 2: fitting the representation-model battery

We call one feature source paired with one model family a configuration. The three feature families introduced earlier appear in five variants, and pairing those with the model families that can consume them gives all the possible configurations listed in Table 1a. Positional and pattern-based features are defined (enumerated) in advance from the training sequences. Programmatic features are written by a language model, in the three ways our framework supports. It can propose them all at once, refine them over several rounds, or generate them while a model is being fitted.

For the first two of those, the model sees the description of the task and a sample of labelled training sequences balanced between the classes. It returns candidates as Python functions, each with a name, a sentence describing what it computes and a sentence arguing why it might matter. When features are refined over rounds, each round shows the model what the fitted model made of the features proposed so far, such that improved proposals can be made. When they are generated during fitting, the tree asks for candidates at each node as it grows, so a feature can be tailored to the sequences that reach that point in the tree. All generated code is parsed and checked against an allow-list before it runs, as Appendix B describes.

The battery uses four model families. Three of them are intrinsically interpretable, and the fourth (XGBoost) is present as a point of black-box comparison.

• Sparse linear. L -penalised logistic regression on standardised columns [41]. The magnitude of a coeficient says how much the model leans on a feature, while its sign says which class it argues for.

• Additive. An explainable boosting machine [42–44], which fits one curve per feature by boosting over the columns in rotation. The shape of a curve shows how the efect changes across the range of values the feature takes.

• Decision tree. A binary tree with a limit on depth, a cap on the number of leaves, and a requirement that every leaf hold at least one per cent of the training rows. Where features are generated during fitting, this is the adaptive tree of Huynh et al. [13], which asks a language model for candidates at each node.

• Gradient-boosted trees [45], included to show what an accurate black-box model achieves on the same dataset.

All four are fitted with balanced class weights, which reduces to the ratio scikit-learn defines (an already balanced dataset is unafected). Each configuration is fitted at a configurable number (default five) random seeds under the real labels, and again with permuted labels.

Every configuration’s features, as used by its model, are recorded in a common format to facilitate the subsequent analysis stages. Each contains the details of what it computes, how heavily the fitted model leaned on it (feature importance), which class the model uses it to argue for (direction), and how far it separates the classes on its own (marginal efect).

Every reported feature is also shown to a language model on its own and rated for how much it would matter if it were true, with features from the permuted fits mixed into the same panel to give that rating a baseline to be measured against, providing a supplementary LLM rating of each feature.

Appendix B gives the fields of the feature record and the matching procedure in full detail.

## Stage 3A: reducing to a small model

Each configuration is then cut down to a model of at most N features, with N chosen in advance and set to ten in all the work reported here. The reduction repeatedly drops whichever features the fitted model leans on least and refits on what remains. It begins from the features that model actually used, and applies the same feature reduction algorithm everywhere. We record how validation performance changes at every step, so a reader can see the efect of reducing the feature count. Appendix B elaborates on the details of this procedure.

For one configuration from each of the three interpretable families, the reduced model is then stored to be presented to the user in full. The tree appears with its features, its thresholds, the number of sequences reaching each node and the proportion of the positive class there. The additive model appears as one curve per feature above a histogram showing where the sequences actually fall. The sparse linear model appears as signed weights per standard deviation, with a mark on any feature the model uses in the opposite direction compared to the direction the feature points to in isolation.

As part of a checkpoint, each presented model is refitted from the run’s records, and has to reproduce the score already recorded for it, to ensure nothing was mangled during this stage.

## Stage 3B: the reliability panel

While Stage 3A focuses on best-performing yet small, interpretable models, Stage 3B goes beyond measuring performance, but rather aims to establish the robustness of the proposed concepts. That is, for each concept, is this a property of the biology that another study of the same kind would find again?

The reliability panel supplies analysis and does not fit any additional models. It leaves the models of Stage 3A untouched, and gives its results as curves across a range of thresholds, so a reader can see the whole trade-of across cut-ofs. It covers every feature used by the interpretable model configurations. Table 1b summarises this, and Appendix C covers the methodology in more detail. The first of its three questions (Tier 1) is answered on the training and validation sequences alone. The hold-out set is opened for the second and third (Tiers 2 and 3), and that one “opening” also supplies every hold-out score in the report.

Features that barely vary are removed first, and those that remain are grouped by how strongly they correlate with one another. This clustering is valuable because a feature and a near-duplicate of it can “divide the attention” of any selection procedure between them, so each of the three kinds of evidence below is reported for the individual feature and again for the cluster it belongs to [46].

Tier 1: Would this feature turn up again? We re-run the procedure that produced the reported model on 100 half-samples of the training and validation sequences, drawn as 50 complementary pairs, and record the proportion of those half-samples in which each feature appears [47]. For any threshold above one half, the expected number of features clearing it whose true selection probability is no better than chance is bounded, and the bound depends on how many features were available and how many a typical fit selects.

Tier 2: Does the feature separate the classes where nothing was fitted? For each candidate we compute how far its area under the curve departs from one half, using that feature alone as a score. This is the rank-sum statistic of Mann and Whitney [48], which requires no assumption about the shape of the feature and applies equally to indicators, counts and continuous quantities. It is computed on the hold-out set and calibrated against 10<sup>5</sup> reshuflings of their labels [49], then reported under three corrections for multiple testing [50–52], alongside a direct estimate of how many of the calls made at each threshold would have been made with no signal present at all [53].

Tier 3: Does the model need the feature? We refit the model without one of its features at a time, using the procedure and the sequences that produced it, and compare the two by the change in log loss on the hold-out set [54]. Where a correlation cluster contributes more than one feature to a model, the whole cluster is also removed at once. A feature that costs little when removed suggests that it is substitutable (rather than uninformative), and the cluster it belongs to reveals what may be standing in for it.

The panel also runs two sense-checks, on setups whose answer is known in advance, to verify that its results can be trusted. Firstly, if one replaces the labels with a random permutation, the panel should claim nothing at any threshold. And secondly, if one simulates labels from five known feature columns and those five should stand at the top of the selection frequencies and be found on the hold-out sequences.

## The report

The report is assembled from stored results as the final step of a SeqMaestro run. Numbers arrive through tables generated from the run results. Explanations that always apply, such as how to look at a curve from an additive model or what a q-value means, are included unchanged in every report.

The report is also designed to keep the two following kinds of statement apart. Anything outside a marked block is either a number or a description of what was done, and can be traced to a table or a figure. Anything that is written by the AI agent as advice or interpretation (rather than description of fact) is placed inside a marked block. The closing recommendations are one such block, for example.

## Two ways to run SeqMaestro

An AI agent with access to a shell and a filesystem can work through the SeqMaestro stages in order. It is given three things:

• The SeqMaestro library, which is everything described above, packaged.

• A set of written instructions, one file per stage, loaded when that stage begins.

• The third is the working folder that it may write inside.

It is not given the hold-out sequences until the point in Stage 3B at which they are required, after every model and the stability tier have been fixed. Within those limits it writes code of its own, where needed, typically in Stage 1, as that is where one has to make sense of an unfamiliar file format, or make dynamic decisions like noticing that a harmless-looking filter removes a large number of sequences of one class. The design therefore leaves open how the agent gets there, as long as the final user-mediated checkpoint is executed. Appendix E describes the instructions and the checkpoints, with an example.

SeqMaestro can also run under a fixed sequence of steps with documented defaults. The user supplies the sequences, the description of the task, the variable to group on, and connection details for a language model, which, here, is called only at the programmatic feature configurations, at a number of predefined checkpoints, and for the final report-writing stage. This route produces the same fitted models, records and checkpoints. However, it cannot “improvise”, as it is a naturally less flexible approach.

## Implementation

SeqMaestro is written in Python. Sparse logistic regression, decision trees, the code that makes the splits and the code that computes the scores come from scikit-learn [41], the additive model from InterpretML [44], the comparison model from XGBoost [45], and the adaptive feature-generating tree from the implementation released with Huynh et al. [13].

The results reported here came from the agent-driven route. The agent had shell and filesystem access and was driven by GPT-5.6 Sol. The two feature proposers, the proposer the adaptive tree calls at each node, and the rating step all used GPT-5.4 through Azure OpenAI.

Appendix H gives the computational requirements of a run, in wall-clock time and in tokens.

Scores are reported with percentile bootstrap intervals from 200 resamples of the evaluation set, the same number for every dataset so that intervals can be compared between them. Any threshold needed for an accuracy-style measure is chosen on the validation set and applied unchanged to the hold-out set. Each run records the command it was started with, the arguments after defaults were resolved, the version of the code, and a full description of the software environment, among other provenance metadata.

## Appendices

Appendix A The system at a glance 25   
Appendix B The battery and the reduction, in detai 26   
Appendix C The discovery reliability panel in full 28   
Appendix D Comparison with related works 31   
Appendix E Running SeqMaestro under an agent 31   
Appendix F The prediction tasks used in this study 32   
Appendix G Features and extraction mechanisms 34   
Appendix H Computational requirements 34   
Appendix I A design agnostic to the language model and the agent harness 37   
Appendix J Understanding the interpretable models 38   
Appendix K The full analysis reports 42

## Appendix A The system at a glance

Three inventories, collected here so the Methods can point at them and carry on. Panel (a) lists the configurations the battery fits. Panel (b) lists the kinds of evidence the reliability panel produces. Panel (c) lists the written instructions an agent driving a run is given, beside what each checkpoint verifies before the run continues.

Table 1 What a SeqMaestro run is made of. (a) The battery. Thirteen configurations, each a pairing of a feature source with a model class, every one fitted at five seeds under real labels and again under shufled training labels. (b) The discovery reliability panel. Every quantity is reported for the individual feature and again for the group of correlated features it belongs to, and every threshold is reported as a curve and not as a single verdict. Appendix C gives the statistics in full. (c) What an agent driving a run is given, and what each checkpoint verifies before the run continues. The agent writes code of its own inside the folder for the run, so a checkpoint settles what the run must contain rather than the route taken to it.

(a) The representation and model battery
<table><tr><td>Feature source</td><td>One feature is</td><td>How the candidates are obtained</td><td>Model classes fitted</td></tr><tr><td>Positional</td><td>An indicator for one base at one position</td><td>Listed in advance, four per position of the window</td><td>Sparse linear, additive, tree, boosted</td></tr><tr><td>Pattern-based</td><td>The number of times a short subsequence</td><td>Listed in advance, every k-mer from k = 1 to Sparse linear, boosted 6 seen in the training sequences, alongside the positional features</td><td></td></tr><tr><td>Programmatic, one-shot</td><td>occurs A short Python function of the sequence</td><td>Proposed once by a language model shown the task description and a class-balanced sample of labelled training rows</td><td>Sparse linear, additive, boosted</td></tr><tr><td>Programmatic, iterative</td><td>As above</td><td>Proposed over five rounds, the first blind and Sparse linear, additive, each later one shown what the fitted model made of the features proposed so far</td><td>boosted</td></tr><tr><td>Programmatic, joint</td><td>As above</td><td>Proposed at each node while the tree is grown, so a feature can be conditioned on the rows that reach that node</td><td>Decision tree</td></tr></table>

The additive model is not fitted on pattern-based features, because it boosts over columns in rotation, and on a k-mer vocabulary that costs far more than the comparison is worth. Boosted trees are present as a reference for what an accurate model nobody can inspect achieves on the same columns, and their features are recorded and never interpreted.

(b) The discovery reliability panel
<table><tr><td>Step</td><td colspan="2">The question it answers</td><td>How it is computed</td><td>What clearing a threshold says</td></tr><tr><td>Before filter and group</td><td>Which candidates stand in for one another</td><td>Absolute Spearman correlation between feature columns, complete linkage cut at 0.8, on the training and validation sequences. No label is consulted</td><td></td><td>The features in a group carry the same information on these sequences</td></tr><tr><td>Tier 1 stability</td><td>Would the selection return this feature on another half of the same data</td><td colspan="2">over a range of thresholds beside a bound on how many features could clear each one by chance</td><td>The share of 100 half-samples that select it, The model did not depend drawn as 50 complementary pairs, reported on which half of the data it happened to see</td></tr><tr><td>Tier 2 association</td><td colspan="2">Does the feature separate How far |AUC - 1/2| departs from zero for The association is present the classes where nothing that feature used alone as a score, on the was fitted</td><td colspan="2">hold-out set, against 10⁵ reshufflings of their labels. Reported under three multiplicity corrections and a direct</td></tr><tr><td>Tier 3 necessity</td><td colspan="2">lose accuracy without this feature</td><td colspan="2">estimate of the false discovery proportion Does the reported model The paired difference in log loss against the Nothing else in the model same model refitted without it, on the substitutes for what it hold-out sequences, one-sided with a carries sign-flip check and corrected across the ten</td></tr><tr><td colspan="5">features (c) The instructions an agent is given, and the checkpoint that closes each stage</td></tr><tr><td colspan="5">Instructions Stage What it must decide and record</td></tr><tr><td>task- construction.md</td><td colspan="3">1 The window and its landmark, the label rule, every step that removed rows and what it removed from each</td><td colspan="2">Rows discarded with no record of what was removed, a group appearing on more than</td></tr><tr><td>battery.md</td><td colspan="3">questions only the data owner can settle 2 Which configurations were run, the</td><td colspan="2">this one off A configuration with no shuffled-label twin, a</td></tr><tr><td>few-feature- models.md</td><td colspan="3">was made under 3A The budget, the order features are dropped in, and the rule for breaking</td><td colspan="2">disagree with the features it reported A smaller model whose refitted score does not reproduce the one already recorded, or</td></tr><tr><td>reliability- panel.md</td><td colspan="3">ties 3B The thresholds, the number of half-samples and of permutations, and</td><td colspan="2">any setting tuned during the reduction A panel that calls something when the labels are shuffled, a panel that fails to recover a</td></tr><tr><td>reporting.md</td><td colspan="3">selection departs from the original final Which configurations lead the</td><td colspan="2">stability tier were fixed A number in the prose that appears in no</td></tr></table>

Two further files are read at every stage rather than at one. task-description.md covers the single description every language model in the run sees, and what is kept from it. run-record.md covers the log of decisions, the software environment and the version of the code that together make a run repeatable.

## Appendix B The battery and the reduction, in detail

## Executing code that a language model wrote

The method works by having a language model write feature functions and then running them, and the prompts contain rows of the user’s data, so a maliciously constructed input file would otherwise be a route from data to code execution. Generated code passes three checks before it runs. Its syntax tree is inspected first: imports must come from a short list, attribute names beginning with a double underscore are refused, references to open, eval, exec, compile and getattr are refused, and nothing may appear at the top level of the file except imports and function definitions. The code is then executed against a minimal set of built-in functions with a restricted import mechanism. Finally, the syntax tree that was inspected is compiled directly, and the text is never parsed a second time, so what runs is what was checked. A function that fails any of these is discarded and counted, so a prompt that has started producing unusable code shows up as a shrinking number of candidates instead of passing unnoticed.

Together these measures make accidental or casual misuse dificult. An adversary with a Python interpreter and enough patience is stopped only by isolating the process, which is a separate piece of engineering that we have not attempted here.

## What is recorded about every feature

Each configuration reports the features its fitted model actually used, and all of them report in the same format. Every record carries a name, a description, an indication of what kind of feature it is, either its recipe or its source code, the importance the fitted model assigns it, a second importance computed by shufling that column and seeing how far accuracy falls, the class the model uses it to argue for, and how far it separates the classes on its own with no model involved. That last quantity is 2 × AUC − 1 over the training rows against the real labels, SeqMaestro computes it and none of the models does, so it means one thing across the entire battery.

The two importances look similar and are not interchangeable. A coeficient, a boosting gain and an impurity reduction are diferent quantities in diferent units, so importances reported by diferent model families can be compared only in rank order. The shufling-based importance requires nothing except a fitted model, some rows and a metric, so it is computed the same way everywhere, including for the models that run in a separate software environment.

Every configuration also writes out the values of its reported features on one common set of sequences, the validation set. Storing the values, and not the means of recomputing them, keeps every later comparison to a matter of loading two columns of numbers, and it means that code written by a language model is never executed outside the run that produced it.

## Matching features between configurations

Diferent configurations have no vocabulary in common. One writes pos\_50\_A for an indicator that position 50 holds an adenine, another writes pos\_50\_is\_A for it, and a language model asked to describe the middle of a window might call it central\_base\_is\_a. Nothing in SeqMaestro compares such names. Two features are considered the same when the absolute Spearman rank correlation between the values they produce on the common sequences reaches 0.8, with an exact match between normalised definitions taking precedence where one exists. Rank correlation is the right comparison because a count and a fraction of the same thing are related by a monotone transformation and should match perfectly, and the absolute value is taken because a feature and its negation carry identical information.

Features from two configurations are paired by the assignment that maximises total similarity, rather than by repeatedly taking the closest remaining pair, so the result does not depend on the order in which features happen to be listed. A concept is then a connected group in the network whose links are these matches. Because the grouping is transitive, a chain of borderline matches can join two properties that a reader would want to keep apart. The clearest symptom of that is a concept containing two features from the same configuration, and the report says when it has happened.

## Rating how much a feature would matter

Every reported feature is shown to a language model on its own, with no indication of which configuration produced it, and rated from one to five on a single question: how biologically significant or interesting would this be, if it were true. The conditional does the important work, because a model cannot check whether a description is true and would give a confident answer anyway. Features from the permuted-label fits enter the same panel and are indistinguishable within it, and are separated out only when the ratings are summarised, so the rating scheme carries its own comparison against noise. What it measures is closer to how a feature is phrased than to what produced it, and no result in this paper depends on it.

## Reducing a configuration to a fixed number of features

Algorithm 1: Reduction to a fixed number of features   
Input: F, the features the configuration reported; the training rows; the budget N   
fit the configuration’s model on $F$   
while $| F | > N$ do   
rank F by the importance the fitted model reports, in absolute value   
if some features have zero importance then   
drop all of them together   
else   
if $| F | > 3 0$ then   
drop the weakest quarter   
else   
drop the weakest one   
end   
end   
refit on what remains   
end   
Output: the model fitted on the surviving N features

Algorithm 1 leaves three things open, and we settle each of them once, for every configuration and every dataset.

Removing a quarter of the surviving features at a time while more than thirty remain keeps the number of refits proportional to the logarithm of the number of features it starts with rather than to that number itself, and still leaves the last thirty steps, which are the ones anyone examines, at full resolution.

Features with zero importance are removed together rather than a quarter at a time, because a zero means the model ignored the feature entirely, and choosing an arbitrary quarter of the features a model ignores would present an arbitrary choice as though it were a ranking.

Ties are broken by the most recent step at which a feature carried any importance, then by how much it carried at that point, and finally by the order in which the configuration listed it, which makes the result deterministic without any part of the rule consulting a label.

## Appendix C The discovery reliability panel in full

This appendix gives the statistics behind Stage 3B. The summary in the Methods and in Table 1b is enough to follow the results. This appendix gives what somebody would need in order to reimplement the panel, or to argue with it.

## Notation

One configuration is profiled at a time. Write $\ b { X } \in \mathbb { R } ^ { n \times p }$ for its candidate feature matrix and $y \in \{ 0 , 1 \} ^ { n }$ for the labels, with $^ { n _ { + } }$ positives and n<sub>−</sub> negatives. Write $\hat { S } ( \cdot )$ for the selection procedure that produced the configuration’s reported model, returning a set of K feature names, with K = 10 throughout this work. Write $R \subset \{ 1 , \ldots , p \}$ for the K reported features. Tier 1 uses the training and validation rows. Tiers 2 and 3 require the use of the hold-out rows once, after Stage 3A and Tier 1 are complete, which is the point in a run where the hold-out sequences are used.

## Filtering and grouping, before any label is consulted

Candidates with zero variance, or with a non-zero prevalence below 1% or above 99%, are removed and listed. That filter is applied again to the hold-out rows, so the candidate list is identical on both sides.

Neither the filter nor the grouping below consults a label, and that is why they can precede the tests without spending any of the error budget those tests have to control [55].

The survivors are clustered on $1 - | \rho |$ , where $\rho$ is the Spearman rank correlation between two feature columns, by hierarchical clustering with complete linkage, cut so that every pair within a cluster satisfies $| \rho | \geq 0 . 8$ . Complete linkage makes that a guarantee about every pair rather than only about the chain of links that joined them. The four indicators of one position are additionally grouped as a position group, since they are mutually exclusive and correlation clustering will not merge them. A cluster’s representative is its member with the highest Tier 1 selection frequency.

We call these groups clusters for the rest of this appendix. They matter because near-duplicate features split the vote of any selection procedure between them, so a feature can be reported by a fitted model and still appear unstable. Every tier is therefore reported twice, once for the feature and once for the cluster containing it [46].

## Tier 1: would the selection return this feature again

The procedure is complementary pairs stability selection [47]. Let D be the training and validation rows, $n = | D |$ . For $b = 1 , \dots , B$ with $B = 5 0$ , draw a random pair of disjoint subsets $A _ { b } , \bar { A } _ { b } \subset D$ , each of size $\lfloor n / 2 \rfloor$ , and run $\hat { S }$ on each. The selection frequency of feature $j$ is

$$
\hat { \pi } _ { j } ~ = ~ \frac { 1 } { 2 B } \sum _ { b = 1 } ^ { B } \Big [ { \bf 1 } \{ j \in \hat { S } ( A _ { b } ) \} + { \bf 1 } \{ j \in \hat { S } ( \bar { A } _ { b } ) \} \Big ] ,
$$

so 100 half-samples in total. For a threshold $\tau \in ( 1 / 2 , 1 ]$ the selected set is $S _ { \tau } = \{ j : \hat { \pi } _ { j } \geq \tau \}$ . Let $\hat { q }$ be the mean size of a selected set across the 2B fits, and let N be the set of features whose true probability of being selected on a half-sample is no greater than ${ \hat { q } } / p ,$ which is the level a feature would reach if the procedure were choosing at random. Then

$$
\mathbb { E } | S _ { \tau } \cap N | \ \leq \ \frac { \hat { q } ^ { 2 } } { ( 2 \tau - 1 ) p } .
$$

The bound holds for any B and needs no assumption about the selection procedure. Its usefulness depends on how many candidates there are. With p in the hundreds it is informative. With $p$ of order ten it can exceed the number of candidates altogether and says nothing, and the report states which of the two situations applies, so a reader is never left to work it out. The panel reports $| S _ { \tau } |$ and the bound on the grid $\tau \in \{ 0 . 5 5 , 0 . 6 0 , \ldots , 1 . 0 0 \}$ , together with the smallest τ at which the bound falls below one and below one half.

The same quantities are computed at the level of clusters, where a cluster counts as selected in a half-sample if any member of it is, with p the number of clusters and $\hat { q }$ the mean number of clusters selected.

$\hat { S }$ mirrors the procedure that produced the reported model, with the hyper-parameters fixed to those of that model, and must return approximately K features on every call. For the sparse linear model the penalty is chosen per call so that about K coeficients are non-zero. For the additive model and the tree, the top K features by the importance the model reports are taken. Any point at which this re-run departs from the original procedure is recorded with the results.

## Tier 2: association on the hold-out set

The statistic for feature $j$ is

$$
T _ { j } \ = \ \Big | \mathrm { A U C } _ { j } - \frac { 1 } { 2 } \Big | , \qquad \mathrm { A U C } _ { j } \ = \ \frac { U _ { j } } { n _ { + } n _ { - } } ,
$$

where $U _ { j }$ is the rank-sum statistic of Mann and Whitney [48] computed from that feature alone used as a score, with average ranks for ties. Written this way the statistic is the area under the receiver operating characteristic curve for that feature [56]. One statistic serves binary, count and continuous features, it makes no assumption about the shape of a feature, and it detects monotone association only, which is the main thing it cannot see.

The null is $B _ { 2 } = 1 0 ^ { 5 }$ random permutations of the evaluation labels. Computing it naively would require $p B _ { 2 }$ rank operations. Instead each column is ranked once into $R \in \mathbb { R } ^ { n \times p }$ , the permuted label vectors are stacked into $Y \in \{ 0 , 1 \} ^ { n \times B _ { 2 } }$ , and every rank sum for every feature and every permutation is the single matrix product $R ^ { \top } Y$ , which is converted to an area under the curve using the constant positive count. The permutation p-value uses the correction that keeps it away from zero [49],

$$
p _ { j } \ = \ \frac { 1 + \# \{ b : T _ { j } ^ { ( b ) } \geq T _ { j } \} } { B _ { 2 } + 1 } ,
$$

so the smallest reachable value is $1 / ( B _ { 2 } + 1 ) \approx 1 0 ^ { - 5 }$ $B _ { 2 }$ is raised if that floor is not comfortably below the smallest targeted false discovery threshold.

Three corrections are reported side by side, because they answer diferent questions and a reader should be able to see how much of a result depends on which is chosen.

• Benjamini and Hochberg [50]: controls the expected share of false calls among the calls, under independence or positive dependence.

• Benjamini and Yekutieli [51]: the same target under arbitrary dependence between features, at the cost of a factor $\textstyle \sum _ { i = 1 } ^ { p } 1 / i$ . Feature columns here are strongly dependent by construction, so this is the conservative choice here.

• Westfall and Young min-P [52]: controls the chance of even one false call, and takes the dependence structure from the permutations themselves instead of bounding it. Each feature’s row of the null matrix is converted to per-permutation p-values, the minimum over features is taken within each permutation, and $p _ { j }$ is compared against that distribution.

Alongside them the panel reports a direct plug-in estimate of the false discovery proportion at each raw cut-of α [53],

$$
{ \widehat { \mathrm { F D P } } } ( \alpha ) = { \frac { B _ { 2 } ^ { - 1 } \sum _ { b } \# \{ j : p _ { j } ^ { ( b ) } \leq \alpha \} } { \operatorname* { m a x } \left( 1 , \ \# \{ j : p _ { j } \leq \alpha \} \right) } } ,
$$

which is the number of calls a null run makes at that cut-of divided by the number the real run makes. It is the form of the answer a reader without a background in multiple testing can act on: at this cut-of, this many calls, of which about this many would have appeared with no signal present.

We use $T _ { j }$ here and never a model’s own importance in its place. Permuting the labels calibrates a multivariate importance only under the global null, so per-feature p-values derived from importances lose their validity as soon as any real signal is present.

## Tier 3: does the model need the feature

For each $k \in R$ , the model is refitted on $R \backslash \{ k \}$ using the hyper-parameters, the rows and the procedure that produced the reported model. On the evaluation rows, write $\ell _ { i }$ for the per-row log loss and

$$
d _ { i } = \ell _ { i } ( \hat { f } _ { - k } ) - \ell _ { i } ( \hat { f } ) ,
$$

the amount the reduced model costs on row i. The null hypothesis is $\mathbb { E } [ d ] \leq 0$ , tested one-sided and paired, with a sign-flip permutation test as a check [54]. The panel reports the mean diference with a 95% interval, the change in area under the curve, and the p-value, corrected across the K tests by Holm’s step-down procedure [57]. Where a cluster contains two or more reported features, the whole cluster is also removed as a block and tested the same way.

This tier needs care, and every report says so. A feature strongly correlated with another feature in the same model will cost almost nothing when removed, whatever its association with the outcome, because what it carried is still present. The correct label for that is substitutable rather than uninformative, and the cluster membership says what is substituting for it. The tier is therefore most useful as an ordering, and it is decisive only where a model with ten features loses measurably by dropping one.

Table 2 Comparison of SeqMaestro with major classes of methods for interpretable nucleotide-sequence analysis. ✓ indicates that a capability is supported, ✗ that it is not, and N/A that the criterion is not applicable. <sup>∗</sup> denotes that it depends on the specific model class.
<table><tr><td>Approach</td><td>Human-readable hypotheses</td><td>Open-ended hypotheses</td><td>Intrinsic interpretability</td><td>Interpretable functional relationships</td><td>Feature space</td></tr><tr><td>Motif discovery</td><td>√</td><td>√</td><td>N/A</td><td>x</td><td>Sequence motifs</td></tr><tr><td>k-mer predictive models</td><td>√</td><td>x</td><td>√/x*</td><td>√/x*</td><td>k-mer presence / frequency</td></tr><tr><td>Deep sequence models</td><td>x</td><td>x</td><td>x</td><td>x</td><td>Latent representations</td></tr><tr><td>Deep learning + post-hoc interpretation</td><td>√</td><td>√</td><td>x</td><td>x</td><td>Attributions / extracted motifs</td></tr><tr><td>Concept-based interpretation</td><td>√</td><td>x</td><td>x</td><td>x</td><td>Predefined concepts</td></tr><tr><td>SeqMaestro</td><td>√</td><td>√</td><td>√</td><td>√</td><td>Open-ended interpretable features</td></tr></table>

## Calibration checks

Two runs accompany every profile.

Global null. The labels are replaced by a random permutation, on the training and validation rows for Tier 1 and on the evaluation rows for Tier 2, and the tiers are re-run. The expected outcome is that nothing is called at any correction level, that the permutation p-values are close to uniform, and that selection frequencies are low and difuse. A procedure that returns K features will still return K under the null, so the informative quantity in Tier 1 is the agreement between half-samples rather than the size of the selected set.

Planted signal. Labels are simulated from a logistic model on five real feature columns with moderate coeficients, and the tiers are re-run. The expected outcome is that the five are recovered at the top of the Tier 1 ranking and are called by Tier 2 after correction for the number of candidates. This checks recovery rather than error control, since neighbours of the planted five are legitimately associated with the simulated labels. Its purpose is to give a negative result its meaning: an empty tier is worth reporting when a planted efect of comparable size would have been found, and is worth much less when it would not.

## The limits of the panel

The panel describes and does not select. It leaves the model Stage 3A produced exactly as it was, and acting on what it says would require fresh hold-out data, since the hold-out set has by then been used Tier 2 detects monotone marginal association and is blind to a feature that matters only in combination with another. Tier 1 measures the stability of a selection procedure on resamples of one dataset, which is a weaker statement than stability across datasets. And no tier can detect a confound that predicts the label, because such a confound survives every resampling and every permutation in the same way a real signal does. That question belongs to the design of the study, and is settled at the first checkpoint.

## Appendix D Comparison with related works

We provide a comparison of SeqMaestro with other approaches in Table 2.

## Appendix E Running SeqMaestro under an agent

An agent with access to a shell and a filesystem is given three things.

1. A library. Everything the Methods describes is a Python package with a test suite. The agent imports it rather than writing its own version, and the checkpoints verify that it did. If a second piece of code recomputes a feature column, there are two answers where there should be one, and nothing to say which of them is right.

2. Instructions, one file per stage. Each file states the purpose of the stage, the parts of the library that already do the work, the decisions the agent has to make and write down, what the stage must leave behind, and the conditions its checkpoint will verify. Table 1c lists the files, and Listing 1 shows the opening of the one for Stage 1.

3. A folder for the run. The agent may create anything inside it and nothing outside it. Every decision it takes is appended to a log recording the stage, the alternatives it considered and what the decision afects.

What it is denied is the hold-out set. Until the models of Stage 3A and the stability tier are fixed, the folder it can read contains none of those sequences, none of their labels and no score computed on them.

name: task-construction   
stage: 1   
checkpoint: checkpoints/task\_construction.py   
# Stage 1. Turn the delivered data into a prediction task   
## What this stage must produce   
One table of (sequence, label, group) rows, a written description of the   
task, and a split whose hold-out part is sealed.   
## Use these rather than writing your own   
seqmaestro.qc.orientation strand convention and genome build, checked   
against a reference   
seqmaestro.grouping a group-aware split   
seqmaestro.harness.power what this split can resolve, before any fit   
## Decide, and write down why   
1. The window, and the landmark it is cut from. Say what a wider window   
would cost in rows, class by class.   
2. The grouping variable. Say what makes two rows dependent here.   
3. What the models are told, and what is kept from them.   
## Leave behind   
README.md manifest.json splits/ description.json open\_questions.md   
## The checkpoint will not pass if   
- rows were discarded with no entry in the manifest   
a group appears on more than one side of the split   
description.json states a fact listed as withheld  
Listing 1 Opening of the Stage 1 instructions, abridged.

## Appendix F The prediction tasks used in this study

Four biological questions, six windows. The eIF4E transcripts were windowed three ways so that the same battery could measure how much of the answer each region gives away before a model is fitted. Every number below comes from the record the corresponding run wrote.

<table><tr><td>Window</td><td>Sequences, and what the label measures</td><td>Training / validation/ hold-out</td><td>Positive class, and its share</td><td>Split kept together on</td></tr><tr><td>Polycomb nucleation 200 nt from the centre of a</td><td>Mouse embryonic stem cells,</td><td>4,053 / 578 /</td><td>cluster 7, 32% Polycomb</td><td></td></tr><tr><td>domain segment</td><td>mm10. H3K27me3 recovery after replication, from a ChIP time course</td><td>1,158</td><td></td><td>domain</td></tr><tr><td>RNA polymerase II pausing 101 nt, read in the direction Human, GRCh38. mNET-seq of transcription, the site at</td><td></td><td>349,769 / 49,968 / paused, 50% 99,935</td><td></td><td>none; a stratified</td></tr><tr><td>position 50 eIF4E dependence</td><td></td><td></td><td></td><td>split by site</td></tr><tr><td>Last 30 nt of the  $5 ^ { \prime }$  untranslated region</td><td>Human transcripts. Translation 1,498 / 214 / 427 factor- measured under reduced eIF4E</td><td></td><td>dependent,</td><td>gene, and identical</td></tr><tr><td>First 150 nt of the coding region</td><td>activity</td><td>1,671 / 239 / 476</td><td>48% factor- dependent,</td><td>window gene, and identical</td></tr><tr><td>Last 100 nt of the  $3 ^ { \prime }$ </td><td></td><td>1,589 / 227 / 452</td><td>50% factor-</td><td>window gene, and</td></tr><tr><td>untranslated region</td><td></td><td></td><td>dependent, 50%</td><td>identical window</td></tr><tr><td>PTBP1 exon regulation 100 nt of intron on each side Human, GRCh38, K562 cells.</td><td>PTBP1 and PTBP2 knockdown,</td><td>618 / 87 / 178</td><td>silenced, 47%</td><td>gene, and identical</td></tr></table>

Table 3 What each dataset is. A window is the stretch of sequence every model sees, cut at a fixed distance from a landmark. The last column names the variable whose members were kept on one side of the split, so that no model is scored on a sequence whose relative it was fitted on.

<table><tr><td>Window</td><td>Strongest baseline of base composition features</td><td>Best of nine, all</td><td>Best of nine, ten features</td><td></td><td>Ref. Floor</td></tr><tr><td colspan="6">Polycomb nucleation 200 nt from the centre of</td></tr><tr><td>a domain segment</td><td>Sixteen dinucleotide counts, 0.867 (within ten numbers: four base iterative, 0.869 fractions, 0.822)</td><td>Additive, programmatic,</td><td>Additive, programmatic, one-shot, 0.867</td><td></td><td>0.858 0.536</td></tr><tr><td colspan="6">RNA polymerase II pausing 101 nt, read in the</td></tr><tr><td>direction of transcription, site, 0.875 the site at position 50 eIF4E dependence</td><td>Two indicators at the</td><td>Sparse linear, positional + k-mer, 0.978</td><td>Additive, programmatic, iterative, 0.960</td><td></td><td>0.981 0.504</td></tr><tr><td>Last 30 nt of the 5′ untranslated region</td><td>Sixteen dinucleotide counts, 0.648 (within ten numbers: four base iterative, 0.671 fractions, 0.607)</td><td>Additive, programmatic,</td><td>Additive, programmatic, one-shot, 0.688</td><td></td><td>0.640 0.555</td></tr><tr><td>First 150 nt of the coding region</td><td>Sixteen dinucleotide counts, 0.876 (within ten numbers: four base iterative, 0.877 fractions, 0.817)</td><td>Additive, programmatic,</td><td>Additive, programmatic, iterative, 0.872</td><td>0.868</td><td>0.552</td></tr><tr><td>Last 100 nt of the 3′ untranslated region</td><td>Four base fractions, 0.754</td><td>Additive, programmatic, iterative, 0.745</td><td>Additive, programmatic, iterative, 0.745</td><td></td><td>0.726 0.553</td></tr><tr><td colspan="6">PTBP1 exon regulation</td></tr><tr><td>100 nt of intron on each side of a cassette exon,</td><td>Pyrimidine fraction, 0.743</td><td>Additive, programmatic, iterative, 0.864</td><td>Additive, programmatic, one-shot, 0.875</td><td></td><td>0.857 0.584</td></tr></table>

Table 4 What each window supports. Every score is AUROC on the hold-out set. The baseline of base composition is the strongest of the Stage 1 baselines, fitted with nothing beyond a logistic regression, and where that baseline uses more numbers than a ten-feature model is allowed, the strongest one within that allowance is given as well. The two model columns cover the nine configurations whose models can be interpreted as statements about features. Ref. is the best of the four gradient-boosted configurations, present only as a point of comparison. Floor is the smallest separation the hold-out set can tell apart from chance.

## Appendix G Features and extraction mechanisms

Figure 6 presents a run stage by stage. Figure 7 complements it by following individual features through the same stages, from the point at which they are generated to the report. Positional and pattern-based features are derived from the sequences without any language model, whereas programmatic features are proposed by a language model, either from the task description alone or, in the iterative and joint settings, with the fitted models feeding back on which proposals should be kept. The figure traces how features from each source enter the models of Stage 2, how the reduction of Stage 3A and the reliability panel of Stage 3B narrow them down and assess their robustness, and how the panel groups computationally similar features from diferent configurations into concepts. It ends at the two outputs of the report that a feature can reach, the best small models and the list of reliable properties. The feature names are borrowed from the PTBP1 task but the figure is illustrative and does not correspond to a particular run.

## Appendix H Computational requirements

A run costs time in two ways that behave quite diferently. Fitting models on the local machine takes longer as the sequences and the candidate features multiply. Calls to a language model depend on how the Stage 2 battery, the grid of feature representations paired with model classes, is configured, so a dataset several hundred times larger asks for roughly the same number of them. The six runs behind this study each ran one stage after another on a single ten-core workstation with no GPU, and the numbers below come from the records they left behind.

![](images/ca739e0016125e8940cd84bc3daf4a234478990782ea5cfd91803c7ea0420595.jpg)  
Figure 7 The journey of features through a SeqMaestro run. A companion to Figure 6: its stages as grey bands, and thirteen features followed through them as coloured traces, named in the spirit of the PTBP1 task with every number illustrative. Positional and pattern-based features are enumerated from the sequences, programmatic features are written by a language model, and a diamond marks where the data filters the proposals, which happens at a diferent point in each of the three programmatic lanes. In Stage 3A every configuration is reduced to ten features by the same algorithm, and in Stage 3B computationally similar features are grouped into concepts, each analysed for stability, association and necessity in turn. The best small models reach the report with all their features, whereas only the concepts that clear every tier reach the list of properties to pursue. 25

Stage 2 to the end of the reliability panel, one stage after another on a single ten-core workstation. Waiting is time idle on API calls.

![](images/5d7dbdb71430f5c8389641a6db7777a474a0bd69f92e9af5c366e85a0e381668.jpg)

## d Token cost per unit of work

Subsamples of one task setup, a 101 nt window with 404 per-position columns, one seed, run alone. A selection fits the whole pool, then eliminates to ten features. Open markers extrapolate the fitted slope.

![](images/56d050dd3e4975773b8dd2bc611d1f51de176b9e7b447904d19e74e037730ea1.jpg)  
The adaptive tree requests 10 candidates over 6 refinement rounds at each of about 35 nodes, which is roughly 2,080 candidates at 2,565 tokens each.  
e API cost of one run

![](images/fc99f7f968318650fda85416d9d6d843d9fff035f82faaf77c96dedde745c2ec.jpg)  
List prices, September 2026, with the repeated part of every prompt served from cache. Thirteen configurations at one seed, or at five seeds with shuffled-label controls.

![](images/5a03f509ea7e28bcd75f889ff6d3aafce0e67c10eaaa8374899b6046be15d207.jpg)  
Figure 8 Computational requirements. a, Wall-clock time of each run from Stage 2 to the end of the reliability panel, split into computation and waiting on API responses. b, Time for a single fit against the number of sequences, on subsamples of one task setup, a 101 nt window with 404 per-position columns. Open markers extrapolate the fitted slope to a million sequences, well beyond any dataset here. The arrow marks the same model and rows with the pool widened to 5,864 columns. c, The same for one selection of the reliability panel: a fit on the whole pool, then elimination to ten features. d, Tokens consumed by each kind of request, averaged over the six runs. The dashed line on the adaptive tree marks the same fit at depth 3 with a single round of refinement. e, API cost at list prices in September 2026, with prompt caching, for the thirteen configurations at one seed, at one seed without joint generation, and at five seeds with shufled-label controls.

## Wall-clock time

Five of the six task setups ran unattended and finished in eight to twelve hours. The RNA polymerase II pausing task, with 499,672 sequences, took two days. Most of that time went on the Stage 2 battery in every case. The reliability panel is the stage most sensitive to the size of the dataset: under two hours on the five smaller setups, thirteen on the largest, where it ran with seven configurations and twenty half-samples, the random halves of the data on which the panel repeats its feature selection.

On the smaller setups, roughly three quarters of the elapsed time is spent waiting for the language model to respond (Fig. 8a), so the pace of a run is set by the latency of the API rather than by the processor. The largest share of the waiting belongs to the LLM iterative feature proposer, whose thirty fits ran one after another at five to eight minutes each. The ten fits of the adaptive tree took over an hour each but are set up to be executed concurrently, and finished in about eighty-five minutes together.

## How fitting and the panel scale

Figure 8b keeps the sequence window and the candidate pool fixed and varies only the number of sequences. Fitting time rises roughly in step with the sequences for the additive model, more slowly for gradient-boosted trees, and faster for the sparse linear model, with an exponent near 1.5. Extending the fitted slopes to a million sequences, twice the largest dataset in this study, puts a single additive fit at about seven minutes.

The reliability panel’s unit of work is a selection: a fit on the whole candidate pool followed by elimination to ten features, about twenty-five fits on a shrinking pool. Measured the same way (Fig. 8c), a selection costs between four and seven times a single fit and grows with the sequences at much the same rate. With a hundred selections per configuration, as in this study, the additive configuration on a million sequences would need about 90 hours of selection.

The number of candidate features matters as much as the number of sequences. With three thousand sequences held fixed, an additive fit on the 404 per-position columns of a 101 nt window takes four seconds. Adding k-mer counts brings the pool to 5,864 columns and the same fit to three minutes.

## Tokens

Taken one at a time, the requests are modest (Fig. 8d). Rating a reported feature takes about 2,000 tokens. A one-shot proposal takes about 13,000, which then serve all three models fitted on it. A fit with iterative refinement takes about 67,000 across twelve calls.

Joint generation is the exception, because the tree asks for candidates at every node it expands. Its cost is the product of three settings: the number of nodes, the rounds of refinement at each node, and the candidates requested per round. The runs here used a depth-5 tree, five rounds and ten candidates, which works out at roughly 2,000 candidates and 5.3 million tokens for one fit. A depth-3 tree with a single round of refinement comes to about a seventh of that.

## Cost

Fitting the thirteen configurations once, at a single seed, uses about 6 million tokens. At list prices in September 2026 that costs between \$9 and \$56 depending on the model (Fig. 8e). Without joint generation it is between \$1 and \$8. Five seeds, each fitted again on shufled labels as in this study, use about ten times as many tokens. These prices assume prompt caching, which serves the repeated part of each prompt at a tenth of the input price. It saves only about a fifth, because most of the bill is completions, which no cache serves.

The agent that prepares the data and drives the run adds a few tens of dollars on top, nearly all of it for context read back from the cache at each turn.

## Appendix I A design agnostic to the language model and the agent harness

SeqMaestro uses language models in two separate roles, and it is tied to a particular vendor in neither. The first role is the agent that carries a run through its stages. The second is the model that the library itself consults as it works, when it asks for programmatic features, grows the adaptive decision tree, and rates the features that were found. This appendix explains why neither role commits the method to a vendor, and shows how a run would be set up under three agent environments in wide use at the time of writing: OpenHands, Codex and Claude Science (Figure 9).

What the method requires of the agent is set out in Appendix E. The agent is given the SeqMaestro library, a set of written instructions with one file per stage, and a folder in which to work. At the end of each stage, a checkpoint verifies what the agent has left in that folder, and the first of these checkpoints also waits for the user’s approval. For the environment that hosts the agent, these requirements reduce to three ordinary capabilities: a shell that runs in an isolated working directory, a mechanism for loading instructions, and a means of pausing for approval. Since every current agent environment provides all three, the choice among them is a matter of preference and infrastructure rather than of method.

The instruction files themselves need no adaptation, because they already follow a convention that these environments share. Each file is a Markdown document headed by a short block of metadata, a format published as the Agent Skills specification and read by all three. OpenHands and Codex look for such files under .agents/skills/ within the repository, whereas Claude looks under .claude/skills/, so the same files serve every environment once they have been placed in the expected location. The metadata block that opens each file, shown in Listing 1, names the stage and the checkpoint script that closes it.

The two roles described above are filled independently of one another. The model that drives the agent comes with the environment: OpenHands can reach any provider through LiteLLM, Codex uses OpenAI’s models, and Claude Science uses Claude. The calls made by the library, on the other hand, go to whichever endpoint is named in the settings of the run, so the model that proposes and rates features can be chosen separately from the one that drives the agent and can be replaced without changing anything else.

## Appendix J Understanding the interpretable models

This appendix is for a reader who works with sequences and does not build models. The three model families at the centre of SeqMaestro are interpretable in the plain sense that every step from a sequence to a prediction can be written down and checked. They difer in what they can express, however, so the figure of one family has to be interpreted diferently from the figure of another. Each part below takes one family and answers the same three questions: how it turns a sequence into a prediction, how to interpret the figure the report draws of it, and what it can say about the biology that a ranked list of features cannot. The figures use the ten-feature models fitted to the PTBP1 exon task of the Results (Fig. 5), with one training exon followed through all three, but nothing below is specific to that task.

## What every model is given

No model works on the sequence itself. Each is given features, which are numbers computed from the sequence, to learn from. Figure 10a shows five features measured on one exon. The simplest is an indicator, one or zero, for a base at a position. The other four each summarise a stretch of the sequence in a diferent way: the composition of a region, the number of short words in a region, the length of a run, and a contrast between two regions. These four are programmatic features in the sense of the Methods, each a few lines of code written by a language model. Most of the expressive power of the framework lies in them, since such a feature can compare two regions, count a motif only within a set distance of a landmark, measure a run or a spacing, or make one measurement conditional on another. Composition is the simplest case. A feature such as “CU in the last 80 nt upstream minus UG in the first 80 nt downstream” already states a hypothesis about position and asymmetry.

Table 5 sets the three families side by side. They difer in what a feature becomes inside the model, which decides both what the model can express and how its figure is interpreted.

![](images/cba21693e8d86f4e920fb9fc10df78e5520b2911844d058495e61759688623ec.jpg)  
Figure 9 The method under three harnesses. Each row is one of the things the method requires of an agent, as listed in Appendix E, and each column shows how one harness provides it. The stage instructions are the same files in every column. The features named are those documented for each product in September 2026.

## What a model is given: five measurements on one exon

Each feature is a number computed from the sequence. The grey band is the stretch the feature measures and the coloured cells are the bases it counts. The values are what the study's own features return on this training exon.

![](images/93b72c8a0e8db8e61b35598747cc267cf9e191d8c333ba2ab5d9c70b9d72eddb.jpg)

## b The sparse linear model: a weight per feature

Ten features, ten weights. The score of a sequence is the sum of each standardised feature times its weight, in log-odds. Rows are ordered by the size of the weight.

![](images/4f359a21047929e41831257d87383c53c342accea67db550d4172785a7e488cb.jpg)  
An average training exon scores +0.06 log-odds. This exon scores +3.64, so the model calls it silenced with probability 97%.

Figure 10 What a model is given, and the sparse linear model. a, Five features measured on one training exon of the PTBP1 task. The window is the two introns flanking the exon, joined. For each feature the grey band is the stretch it measures, the coloured cells are the bases it counts, and its value is given at the right. b, The ten-feature sparse linear model of the same task. Left, the weight of each feature, in log-odds per standard deviation of that feature. Right, the exon of panel a: its value on each feature, and what that adds to its score relative to an average training exon. Red argues silenced and blue enhanced, as in Fig. 5.
<table><tr><td></td><td>Sparse linear</td><td>Additive</td><td>Decision tree</td></tr><tr><td>A feature becomes One weight</td><td></td><td>One curve</td><td>One question with a threshold, possibly asked more than once</td></tr><tr><td></td><td>The prediction is The sum of weight times value The sum of the heights looked</td><td>up on the curves</td><td>The end point of a path of questions</td></tr><tr><td></td><td>The figure showsOne bar per feature</td><td>One curve per feature, over a strip of where the sequences fall</td><td>The questions, the paths, and the class balance at every point</td></tr><tr><td>It can say</td><td>Which features matter, which How the effect changes across way, and by how much</td><td>the range of a feature: a gradient, a threshold, a saturation, a reversal</td><td>Which conditions combine, and which matter only when others hold</td></tr><tr><td>It leaves to the others</td><td>Thresholds, reversals, and features acting together</td><td>Features acting together, as fitted here with one curve per renders as steps feature</td><td>Gradual effects, which it</td></tr></table>

Table 5 The three model families side by side. Every model in the battery is given the same kind of input, a set of features computed from the sequence. The families difer in what they do with it.

## The sparse linear model

The score of a sequence is the sum of its features, each multiplied by a weight (Fig. 10b). Because the features are standardised before the fit, every weight is in the same units, the change in score for a change of one standard deviation in the feature. The score is in log-odds: zero is even odds between the two classes, about +0.7 doubles the odds of the positive class, +1.4 quadruples them, and the same numbers negative favour the other class in the same proportion. The $L _ { 1 }$ penalty holds most weights at zero, so the fitted model uses a handful of features and reports the rest as unused. That property is why the family is in the battery.

• Each row is a feature and the bar is its weight. The colour is the class the feature argues for and the length is how much of the score the feature can account for. A weight argues the same way across the whole range of the feature.

What matters is being above about 1. A ratio of 10 says no more than 3.

A plateau

Six of the ten curves of the additive model of the PTBP1 task. Each curve is what the model adds to the score, in log-odds, at each value of the feature: above the line argues silenced, below it enhanced. The grey strip is where the training exons fall. The dot is the training exon followed through this appendix.

![](images/b5013e664858a34fe348de67e8c2fb49338fd0a5d0768fcf82bc503623ac06ed.jpg)  
The more CU, the more the model argues silenced, without a break.

![](images/5d0d84c0498543c73a29f3c7bcccc0dba86f4734f4634c40ae307389fa0dde11.jpg)  
Flat below about 0.64, then a jump. The model found a threshold.

![](images/e71bfbef6ad5d0ea7a9beffc169435111f9efc78b242d2a6de834431bc629fbd.jpg)  
Above about 0.25 G, the curve drops: the feature argues enhanced.

![](images/9afb787479753ee1547b2e0755104770025e73ee013684225f6d806a2adf2000.jpg)

![](images/6edc2538a45c484d1d918c3600aef2b0c0017a3039a76d2a82ca9d1a1b13fabb.jpg)  
A turn, on few exons

![](images/7e9069ac38ef7a6442b4f1f64fb28421c204da6ea975271b30d583ba4aae06a0.jpg)

![](images/46424e689664687df2a584edc9e74231a35c63c7771894917413231a23f78ae5.jpg)  
Figure 11 The additive model. a, Six of the ten curves of the additive model of the PTBP1 task, chosen for the variety of shapes they show. Each curve is what the model adds to the score at each value of the feature, in log-odds. The grey strip beneath is how many training exons take each value. The dot marks the exon of Fig. 10a. b, The same exon’s ten heights, which add up to its score.

• The right-hand column assembles the score of one sequence. Beside each feature’s value is what it adds relative to an average training sequence. These parts sum to the score, which the model converts to a probability.

The model says which features matter, in which direction and by how much, in one number per feature.   
A threshold, a reversal, or two features acting together are what the next two families add.

## The additive model

The additive model, an explainable boosting machine, replaces each weight with a curve (Fig. 11). To score a sequence it looks up the value of each feature on that feature’s curve, takes the height there, and adds the heights. As fitted here, no term combines two features, so the ten curves are the whole model and a figure of them hides nothing.

• The horizontal axis is the value of the feature and the curve is what the model adds at that value. Above the line argues for one class, below it for the other. A curve may therefore argue both ways along its length.

• The strip underneath shows where the sequences are. A steep or extreme part of a curve above an empty stretch of strip rests on very few sequences. The last panel of Fig. 11a is an example, since the turn beyond a run of four comes from a handful of exons.

The shapes are where an additive model says more than “higher is more”. A curve can take any shape,

The ten-feature tree of the PTBP1 task, to three levels. Each box asks one question about one feature. A box gives the training exons that reach it and the share of them that are silenced, which also sets its colour. The heavy path is the route of the training exon followed through this appendix, with its values on the edges, ending in the outlined box.

![](images/09e6013abbb8ba20bafbc0f971b36378e69bd5a8c6f8a85c797c142603ea25b9.jpg)  
Figure 12 The decision tree. The ten-feature tree of the PTBP1 task with programmatic features written at each split, drawn to three levels. Each box gives the training exons that reach it and the share of them that are silenced, which also sets its colour, red for mostly silenced and blue for mostly enhanced. Chevrons mark branches that continue below the drawn depth. The heavy path is the route of the exon of Fig. 10a, with its values written on the edges.

since every step of it is fitted separately, and four kinds recur often enough to be worth naming. A gradient means the efect grows steadily across the range, which is what a weight would also have described. A switch, flat and then a jump, means the model found a threshold, whose value is testable in itself: an element built to cross it would show whether the efect is a switch or a gradient. A saturation or a plateau means the efect grows to a point and then stops. That point is again a number the biology can be asked about, such as a run beyond which extra length earns nothing. A turn means the feature argues one way at low values and the other way at high values, which is a diferent claim from either direction alone and one that no single number per feature could carry. Whatever the shape, it is interpreted the same way, as the height the model adds at each value.

## The decision tree

A decision tree is a sequence of yes-or-no questions (Fig. 12). Each box asks whether one feature is at or below a threshold. A sequence follows one path from the top box to one end box, taking the yes or the no branch at each question. The end box gives the prediction. The model chose each threshold from the training sequences, as the cut that best separated the two classes among the sequences reaching that box, so a threshold is to be treated as a hypothesis about a biological cut-of.

• Each box names the feature and the threshold it tests, together with how many training sequences reached it and what share of them belong to the positive class. Colour follows that share, so a branch that separates the classes deepens in colour as it descends, whereas a box near fifty per cent, drawn almost white, has separated nothing yet.

• Chevrons mark branches that continue below the drawn depth, so nothing is hidden about how much of the tree is of the page.

• One sequence follows one path, which can be written out as a rule.

A path is a rule made of several conditions. This is what a tree expresses most directly: a feature that matters only once other conditions hold. Table 6 writes out four paths of the tree in Fig. 12. Its second and third rows are the clearest case. A single base 51 nt before the exon separates 48% silenced from 94%, but only among exons that already carry more CU upstream than downstream and few UCU in the middle of the upstream intron. On its own, that base separates the classes hardly at all. When the same feature is tested at more than one threshold on diferent branches, as the count of CU and UC in the upstream intron is in the full tree, the model is describing a graded efect in steps, or a threshold that shifts with context.

<table><tr><td>The conditions along the path</td><td>Training exons</td><td>Silenced</td></tr><tr><td>More than 4 more CU upstream than downstream, and more than 2 UCU 83 between 21 and 60 nt before the exon</td><td></td><td>94%</td></tr><tr><td>More than 4 more CU upstream than downstream, at most 2 UCU there, 33 and U at 51 nt before the exon</td><td></td><td>94%</td></tr><tr><td>The same conditions, with any other base at that position At most 4 more CU upstream than downstream, more than 1 UCU there, 27</td><td>56</td><td>48%</td></tr><tr><td>and more than 12 pyrimidines in the first 20 nt after the exon</td><td></td><td>93%</td></tr><tr><td>At most 4 more CU upstream than downstream, at most 1 UCU there, and 163 a base other than U 83 nt after the exon</td><td></td><td>15%</td></tr></table>

Table 6 Four rules written out from the tree of Fig. 12. Each row is one path from the top box to a box at the third level, given as the conditions a sequence must meet to reach it. The second and third rows show a base that matters only in one context.

Because each question is chosen among the sequences that reach it, a drawn tree is one of several that would fit the same sequences about equally well. The first tier of the reliability panel measures this directly, by asking of every feature whether the selection would return it on another half of the data (Methods, Stage 3B).

## Appendix K The full analysis reports

Each run produces a self-contained document covering all of its stages: the preparation decisions and what each one cost, every configuration in the battery with its shufled-label control, the few-feature models drawn in full with the table of features behind each figure, the whole reliability panel including the calibration checks, and the per-feature evidence for every profiled configuration. Measured fact and interpretation are marked apart throughout. The six documents are available at the links below.

• RNA polymerase II pausing.

https://www.dropbox.com/scl/fi/7v3qvhepir5n2tdhrhnkj/REPORT\_polymerase\_full.pdf?rlkey=ufobb wan6ojekx7kt33fv4txt&dl=0

• Polycomb nucleation.

https://www.dropbox.com/scl/fi/zcssccvbsmcc7ipc2nr6m/REPORT\_nucleation\_full.pdf?rlkey=dsxu0   
sex6rctfp8nejdgs4vnx&dl=0

• eIF4E dependence, first fifty codons of the coding region. https://www.dropbox.com/scl/fi/eremnynkpsaojpy1i5e14/REPORT\_mrna\_factor\_cds\_full.pdf?rlkey= udvz1e5s7j88p4xu9nw9tlyya&dl=0

• eIF4E dependence, the last thirty bases of the 5<sup>′</sup> untranslated region. https://www.dropbox.com/scl/fi/pgjc6t4n0okbhj0c46xd7/REPORT\_mrna\_factor\_utr5\_full.pdf?rlkey =2rh05lnpfo0wmz61n7gnmya8y&dl=0

• eIF4E dependence, the last hundred bases of the 3<sup>′</sup> untranslated region. https://www.dropbox.com/scl/fi/vllmqz0vabvtgw02elwh1/REPORT\_mrna\_factor\_utr3\_full.pdf?rlkey =nguq2terr7dzg28vucv79dh8o&dl=0

• PTBP1 exon regulation.

https://www.dropbox.com/scl/fi/ar1hjl96umy4kmjjpz8s0/REPORT\_ptbp1\_exon\_regulation\_full.pdf? rlkey=445u76ji0i28cl2u85bb4yoou&dl=0