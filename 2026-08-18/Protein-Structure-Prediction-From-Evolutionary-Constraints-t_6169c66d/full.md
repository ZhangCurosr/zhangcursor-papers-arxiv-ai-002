# Protein Structure Prediction: From Evolutionary Constraints to Generative Modeling

Wengan He<sup>a,b,∗,1</sup>, Yongsheng Luo<sup>a,1</sup>, Lihong Jiang<sup>b</sup>, Wenhui Xu<sup>b</sup> and Yu Li<sup>a,∗</sup>

<sup>a</sup>Zhuhai College of Science and Technology, Zhuhai, China

<sup>b</sup>Zhuhai UNO Technology Co., Ltd., Zhuhai, China

## A R T I C L E I N F O

Keywords:   
protein structure prediction   
structural bioinformatics   
deep learning   
difusion models   
protein design

## A BS T RA C T

Accurate protein structure prediction is fundamental to structural biology because protein structure underlies molecular function and provides a basis for mechanistic interpretation. Recent advances in deep learning have transformed the field from multiple sequence alignment (MSA)-driven monomer folding into broader frameworks capable of modeling protein complexes and increasingly heterogeneous molecular systems. Existing reviews have summarized this progress from the perspectives of representative models, application domains, and protein design. Building on these eforts, this review focuses on the methodological evolution of the field itself. It examines recent developments through three closely related dimensions: representations and data, architectures and learning strategies, and confidence and evaluation. Within this perspective, the field is organized into four methodological phases and three cross-cutting transitions: from explicit evolutionary coupling features and early contact prediction to learned sequence representations in AlphaFold2, RoseTTAFold, and ESMFold; from protein-only monomer folding to increasingly integrated modeling of heterogeneous molecular systems in AlphaFold-Multimer, RoseTTAFoldNA, and AlphaFold3; and, more recently, from prediction-oriented structure inference to design-oriented generative modeling in RFdifusion and related frameworks. This framework provides a clearer understanding of how methodological shifts have shaped the capabilities, limitations, and practical roles of recent models.

## 1. Introduction

Protein structure is central to structural biology because it constrains molecular function and enables mechanistic interpretation [6, 18]. The Protein Data Bank provides the principal global archive of experimentally determined macromolecular structures [17]. Traditional computational approaches, including homology modeling, fragment assembly, and physics-based refinement, achieved important progress but remained constrained by template availability, conformational sampling complexity, and the accuracy of energy functions [95, 70, 18].

The emergence of deep learning fundamentally redefined the methodological foundations of structure prediction [66, 59]. Within a short period, the field progressed from co-evolution-assisted contact inference [82] to end-to-end coordinate prediction [7, 47] with near-experimental accuracy for many monomeric proteins [40]. Subsequent developments expanded modeling capabilities to protein–protein complexes [23], protein–nucleic acid assemblies [8], and ligand-inclusive heterogeneous molecular systems [42, 2, 50, 75, 85]. More recently, difusion- and flow-based generative frameworks [83, 27] extended structural modeling toward controllable protein design.

Existing reviews have summarized this progress from perspectives such as representative model families, application scope, and protein design [59, 63]. The present review instead emphasizes the methodological evolution of protein structure prediction itself. Specifically, it examines recent developments through three closely related dimensions: (1) representations and data, (2) architectures and learning strategies, and (3) confidence and evaluation. As summarized in Figure 1, the field is organized here into four methodological phases spanning explicit evolutionary modeling, end-to-end deep folding, relatively unified complex structure modeling, and generative structure modeling and design. Figure 1 further highlights three associated transitions: from explicit evolutionary coupling features and early contact prediction [82] to learned sequence representations in models such as AlphaFold2 [40], RoseTTAFold [7], and ESMFold [47]; from monomer-focused modeling in the first two phases to increasingly integrated treatment of heterogeneous molecular systems in AlphaFold-Multimer [23], RoseTTAFoldNA [8], RoseTTAFold All-Atom [42], and AlphaFold3 [2]; and, more recently, from prediction-oriented structure inference to design-oriented generative modeling in RFdifusion [83] and related frameworks [27]. Rather than replacing predictive approaches, generative protein design is treated here as a complementary paradigm that extends structural modeling toward controllable creation and exploration of novel sequence–structure solutions. This phaseand-transition perspective provides a clearer understanding of how methodological shifts have shaped the capabilities, limitations, and practical roles of recent models.

Phases in this review are assigned primarily according to modeling objective and system scope rather than architecture alone. Thus, AlphaFold3 is placed in Phase III because its principal task is structure inference for specified heterogeneous biomolecular assemblies, even though its structure module uses difusion-based denoising [2].

![](images/7b87f7ac7b04f7a7364245c24ae58335477cdc89ae71d3a1b52f0e13c552f272.jpg)  
Transitions are shown as layered arc connections, indicating scope expansion rather than strict temporal order.  
Figure 1: Four methodological phases in the evolution of deep learning for protein structure prediction.

## 2. Explicit Evolutionary Modeling

The earliest methodological phase of protein structure prediction with machine learning is characterized by the explicit modeling of evolutionary couplings derived from multiple sequence alignments (MSAs). As illustrated in Figure 2, this paradigm follows a structured and modular pipeline consisting of (A) MSA construction, (B) co-evolutionary feature extraction, (C) contact or distance prediction, and (D) downstream structure reconstruction . The underlying data regime is dominated by curated structural datasets such as the Protein Data Bank (PDB) [17], together with homologous sequence collections obtained through largescale alignment tools, including ClustalW [45], HMMER [26, 20], HHblits [62], and MMseqs2 [74].

At the core of this phase lies a key assumption: evolutionary constraints encoded in homologous sequences reflect three-dimensional structural relationships. According to the thermodynamic hypothesis of protein folding [6] and subsequent theoretical developments [18, 12], amino acid sequences determine native structures, and residues that are spatially proximal tend to co-evolve to maintain structural stability and function. Statistical approaches such as direct coupling analysis (DCA) quantify these dependencies by estimating residue–residue coupling strengths from MSAs [53, 54, 31]. These coupling signals provide an indirect but powerful proxy for structural proximity and serve as the primary source of structural information in this paradigm.

## 2.1. From Statistical Couplings to Deep Learning-Based Contact Prediction

Building on these statistical foundations, deep learning models introduced more expressive approaches to contact prediction. Convolutional neural networks and residual architectures trained on MSA-derived features were used to predict residue–residue contact maps or distance distributions [82]. In these models, co-evolutionary features—such as covariance matrices, coupling scores, and position-specific profiles—are explicitly computed and used as input representations.

Representative systems such as RaptorX and related frameworks substantially improved long-range contact prediction compared with purely statistical methods [82, 59]. However, despite these improvements, the overall methodological logic remained largely unchanged. Evolutionary signals were still engineered first, predictive constraints were generated second, and full three-dimensional structures were reconstructed only in a downstream stage. In this sense, deep learning enhances predictive accuracy without fundamentally altering the pipeline structure.

## 2.2. Modular Pipeline and Structural Decoupling

As shown in Figure 2, this phase exhibits a distinctly modular architecture. Contact or distance prediction is performed independently of coordinate reconstruction, and predicted constraints are subsequently passed to external folding or reconstruction procedures based on fragment assembly, distance geometry, or knowledge-based potentials [70, 72]. Although some implementations more tightly couple constraint prediction and reconstruction, the dominant early pipeline remained fragmented: gradients did not propagate from final structural outputs through all preceding stages, and individual components were optimized separately.

![](images/2a12311112ac4bfaeddbf30ad1153c9faa36e55d47f0dd5e16edc128bc9c3607.jpg)  
Figure 2: Co-evolutionary modeling pipeline in early protein structure prediction. The schematic illustrates (A) MSA construction from sequence databases, (B) extraction of co-evolutionary features (e.g., covariance or coupling matrices), (C) deep learning–based contact or distance prediction, and (D) downstream structure reconstruction using external folding engines.

This design introduces several limitations. Errors in MSA construction, coupling estimation, or contact prediction can propagate into downstream folding without endto-end correction. Moreover, global geometric consistency is not directly optimized at the contact-prediction stage, so locally plausible constraints may still be incompatible with a physically coherent three-dimensional structure. Consequently, gains in intermediate contact or distance accuracy do not necessarily yield proportional improvements in final structural quality [44, 5].

## 2.3. Methodological Characteristics and System-Level Limitations

These characteristics can be summarized from a systemlevel perspective, as shown in Table 1. The phase combines curated structural supervision with MSA-dependent homologous sequence information, relies on explicit coevolutionary features as its principal representation, and formulates prediction in terms of contacts or distances rather than direct coordinate generation.

From a representation standpoint, the strengths and weaknesses of this paradigm are tightly coupled. MSAs [45] encode evolutionary variation through covariance statistics and coupling scores [53, 54, 31], which can provide highly informative signals for structurally conserved protein families. However, predictive performance depends strongly on the depth, diversity, and alignment quality of available homologs [59]; when coverage is limited, co-evolutionary signals become sparse or noisy [82].

This limitation is further reinforced by the data regime. Although sequence databases have expanded rapidly, usable target-specific information remains constrained by the ability to identify and align informative homologs using tools such as HMMER [26, 20], HHblits [62], and MMseqs2 [74]. Proteins with shallow or biased evolutionary coverage therefore remain challenging even when structural supervision is available from repositories such as the PDB [17].

The learning objective introduces an additional limitation. Contact maps and distance matrices are intermediate structural representations rather than final atomic coordinates, and the inverse mapping from a set of predicted constraints to a unique, physically valid three-dimensional structure is generally underdetermined. Reconstruction quality therefore depends on the downstream optimization procedure as well as on the accuracy and consistency of the predicted restraints [46, 78].

## 2.4. Transition Toward End-to-End Modeling

Despite these limitations, explicit evolutionary modeling represents a critical foundation for subsequent developments. It established the importance of pairwise residue relationships, demonstrated that sequence variation can be systematically exploited for structural inference, and provided the conceptual basis for learning structure from sequencederived constraints.

More importantly, the limitations of this phase—particularly its dependence on explicit feature engineering and its modular separation between prediction and reconstruction—directly motivate the transition to end-to-end deep folding models. Later approaches seek to integrate representation learning, geometric reasoning, and coordinate prediction within a unified optimization framework, thereby addressing the

Table 1  
Methodological characteristics and limitations of explicit evolutionary modeling in early protein structure prediction.
<table><tr><td>Aspect</td><td>Description</td><td>Limitation</td></tr><tr><td>Data regime</td><td>Curated structures + homologous sequences (MSA- dependent)</td><td>Limited by MSA depth and coverage</td></tr><tr><td>Representation</td><td>Explicit co-evolutionary features (covariance, cou- pling)</td><td>Requires high-quality alignments</td></tr><tr><td>Objective</td><td>Contact/distance prediction</td><td>Indirect relation to final 3D structure</td></tr><tr><td>Pipeline</td><td>Modular (prediction + folding)</td><td>Error propagation, no end-to-end optimization</td></tr></table>

fragmentation and ineficiencies inherent in early pipelines [66, 40].

## 3. From Explicit Evolutionary Features to Learned Sequence Representations

A central methodological transition in protein structure prediction is the shift from explicit evolutionary feature engineering toward learned sequence representations (Figure 1). Rather than relying exclusively on query-specific coevolutionary statistics derived from MSAs, some modern predictors learn latent representations from large-scale sequence corpora through self-supervised pretraining [32, 22, 47]

## 3.1. MSA-Based Pipelines and Their Underlying Assumptions

In earlier MSA-based pipelines, structural information is obtained through a multi-stage process (Figure 3). Starting from a query sequence, homologous sequences are retrieved from large databases to construct an MSA, which serves as the basis for extracting co-evolutionary features such as covariance matrices and coupling scores. These features are then used as inputs to neural networks for predicting residue–residue contacts or distance distributions, which are subsequently converted into three-dimensional structures through external folding engines or constraint-based reconstruction.

This pipeline reflects the core assumption of explicit evolutionary modeling: structural constraints can be inferred from correlated mutations across homologous sequences. When suficiently deep and diverse MSAs are available, such co-evolutionary signals provide strong indicators of spatial proximity and functional constraint [53, 54, 31]. Their reliability nevertheless varies with MSA depth, sequence diversity, and alignment quality. Query-specific MSA construction also introduces database-search and preprocessing costs, particularly at proteome scale [74, 56].

## 3.2. Emergence of Learned Sequence Representations

In explicitly MSA-free PLM-based predictors such as ESMFold and OmegaFold, the query sequence is encoded directly by a pretrained protein language model rather than being accompanied by a query-specific MSA. Self-supervised pretraining on large sequence corpora produces contextual embeddings that can encode information relevant to structure and function [47, 87]. The precise structural content and transferability of these embeddings remain model- and task-dependent.

These learned embeddings are then passed to downstream folding or coordinate-prediction modules. In MSAfree implementations, removing query-specific homology search can simplify inference and improve throughput, although the magnitude of the speed and accuracy trade-of varies across models, target classes, and evaluation settings [47, 87]. This paradigm also shifts much of the computational burden from online alignment construction to ofline large-scale pretraining.

## 3.3. Pipeline-Level Comparison Between MSA and PLM Paradigms

The contrast between these two paradigms is illustrated in Figure 3. In MSA-based pipelines (Figure 3), structural information is derived from explicitly computed evolutionary features, and the workflow is multi-stage and alignmentdependent. In PLM-based pipelines (Figure 3), structural information is encoded in learned sequence representations, and prediction becomes more integrated and less dependent on external preprocessing. This transition reflects a broader methodological shift: from explicit feature engineering to representation learning. In the former, structural signals are manually derived from sequence variation; in the latter, they are implicitly captured within learned embedding spaces. As a result, the locus of structural information moves from external preprocessing steps to the internal representations of neural networks.

## 3.4. Methodological Diferences and Scaling Behavior

From a methodological perspective, the diferences between these paradigms can be summarized along several dimensions, as shown in Table 2. MSA-based methods obtain target-specific evolutionary information from homologous sequences, whereas PLM-based methods encode statistical regularities acquired during pretraining. MSA-dependent performance is therefore strongly influenced by homolog availability, while the capabilities of PLM-based systems depend on the architecture, pretraining data, model scale, and downstream folding module. These are tendencies rather than universal properties of every model in either category [15, 24, 47].

![](images/c6e3961ab1999ba4b6fe27c97a03c901c47ecfada5ff5d96cb5f5860b8308c04.jpg)  
Figure 3: Comparison between MSA-based and PLM-based structure prediction pipelines. The schematic contrast (A) traditional pipelines that rely on MSA construction and explicit co-evolutionary feature extraction with (B) PLM-based pipelines that directly encode sequence information into contextual embeddings. Key diferences include information source, computational cost, and integration with downstream structure modules.

Table 2  
Comparison between explicit evolutionary modeling and learned sequence representations
<table><tr><td>Dimension</td><td>Explicit evolutionary modeling</td><td>Learned sequence representations</td></tr><tr><td>Data regime</td><td>MSA-dependent homologous sequences</td><td>Large-scale unlabeled sequence corpora</td></tr><tr><td>Representation</td><td>Explicit co-evolutionary features (covariance, cou- pling)</td><td>Implicit contextual embeddings (PLMs)</td></tr><tr><td>Information</td><td>Directly computed evolutionary couplings</td><td>Learned statistical patterns from sequences</td></tr><tr><td>Scalability</td><td>Limited by MSA depth and coverage</td><td>Can improve with pretraining data and model scale, depending on architecture</td></tr><tr><td>Generalization</td><td>Strong when informative homologous coverage is available</td><td>Some MSA-free PLM models can be advantageous for orphan or low-homology targets</td></tr></table>

These diferences lead to distinct but overlapping generalization profiles. MSA-based approaches remain highly efective when informative homologous coverage is available. Some MSA-free PLM predictors, notably ESMFold and OmegaFold, can be useful for orphan or low-homology targets because they do not require a target-specific alignment, but this advantage should not be generalized to all PLM architectures or all dificult proteins [47, 87]. The two information sources are therefore better viewed as complementary than mutually exclusive.

## 3.5. Limitations and Hybrid Strategies

PLM-based approaches introduce their own challenges. Learned embeddings are distributed representations whose biological and geometric content is often less directly interpretable than explicit coupling matrices. In addition, the apparent reduction in inference-time preprocessing can obscure the substantial computational and data requirements of PLM pretraining [21, 16].

Whether learned sequence representations consistently recover the fine-grained residue-pair constraints supplied by deep MSAs remains model- and target-dependent. Hybrid systems that combine PLM embeddings with MSA-derived or pairwise geometric information may therefore be advantageous in some settings [15].

![](images/7f027e877ce766c7162da84c67fa2459bac1c0e62cef665ce639d276686396aa.jpg)  
Figure 4: Evolution from monomer folding to relatively unified heterogeneous modeling. The schematic illustrates (A) monomer folding with residue-level representation, (B) multi-chain protein-complex modeling, (C) heterogeneous biomolecular systems including nucleic acids and ligands, and (D) mixed token- and atom-level modeling with iterative or difusion-based refinement.

## 3.6. Transition Toward End-to-End and Unified Modeling

Importantly, this transition should not be interpreted as a simple replacement of MSA-based methods. Rather, it represents a broader shift in how structural information is conceptualized and learned. Evolutionary signals are not discarded but transformed—from explicitly computed statistics to latent representations embedded within model parameters.

More broadly, the move toward learned sequence representations lays the foundation for subsequent methodological developments. By reducing dependence on handcrafted features and enabling more integrated architectures, it facilitates the emergence of end-to-end folding models and, later, unified frameworks for modeling heterogeneous molecular systems. In this sense, this transition serves as a critical bridge between early co-evolution-based approaches and modern deep learning architectures.

## 4. From Monomer Folding to Unified Modeling of Heterogeneous Molecular Assemblies

A major development in contemporary protein structure prediction is the expansion of modeling scope from singlechain folding to the joint treatment of increasingly heterogeneous molecular systems [23, 8]. This transition involves more than an increase in target complexity: it changes the entities represented by the model [42], the geometric and chemical relationships that must be learned, and the criteria by which predictions are evaluated [2]. As illustrated in Figure 4, the evolution of protein structure prediction can be understood as a progressive expansion of both system complexity and representational richness, moving from isolated protein folding toward relatively unified modeling of heterogeneous molecular interactions.

## 4.1. Monomer Folding as the Foundational Paradigm

In the monomer-folding setting, the predictive task can be formulated as mapping a single amino acid sequence to a dominant three-dimensional structure. Such predictions are often assessed using global and local structuralsimilarity metrics, including TM-score, RMSD, and lDDT; these metrics are not exclusive to monomer prediction, although complex and ligand-containing systems require additional interface- and chemistry-aware measures [92, 52, 10]. This formulation underlies landmark end-to-end models such as AlphaFold2 [40].

Monomer folding nevertheless captures only a subset of biologically relevant structural phenomena. Protein function is frequently determined by binding, multimerization, nucleic-acid recognition, ligand association, and higherorder assembly, motivating models that represent intermolecular interfaces and chemically distinct entities [4, 64].

## 4.2. From Multi-Chain Complexes to Heterogeneous Systems

The first step beyond monomer folding is the extension to multi-chain protein complexes, as exemplified by AlphaFold-Multimer [23]. In this setting, the model must reason about intra-chain folding and inter-chain geometry, including interface arrangement, chain identity, and stoichiometry. Accurate individual-chain structures do not guarantee a correct assembly, so complex prediction requires interface-aware representations, training objectives, and evaluation criteria [93, 13].

The subsequent expansion proceeds through distinct steps. RoseTTAFoldNA extends the three-track framework specifically to protein–DNA and protein–RNA complexes [8], whereas RoseTTAFold All-Atom broadens the modeling space to generalized biomolecular systems that can include ligands and other non-polymer entities [42]. AlphaFold3 further supports joint structure inference for proteins, nucleic acids, ligands, ions, and modified residues within a common predictive system [2]. These models should therefore not be treated as having identical molecular scope.

## 4.3. Toward Relatively Unified Modeling Frameworks

As these capabilities converge, the field moves toward what may be described as relatively unified modeling frameworks. The qualifier “relatively” is important: current systems integrate several previously separate tasks within shared architectures [42, 2], but their accuracy, training coverage, and chemical treatment remain uneven across molecular classes and interaction types [36, 57].

This transition entails a corresponding change in representation. Earlier models emphasized residue-level sequence, pair, and frame representations, whereas heterogeneous systems require mixed token- and atom-level descriptions that encode chemical identity, connectivity, and crossentity geometry. The predictive objective correspondingly broadens from isolated-chain folding to joint structure inference across interacting molecular components [42, 2].

## 4.4. AlphaFold3 and the Redefinition of the Prediction Problem

Within this phase, AlphaFold3 represents a key methodological turning point [2]. Its significance lies not only in predictive performance, but also in reformulating structure inference around heterogeneous biomolecular interactions.

AlphaFold3 expands the modeled entity space by supporting proteins, nucleic acids, ligands, ions, and modified residues within a shared predictive framework. This is methodologically important because many biological interactions depend on precise geometry between chemically distinct components rather than on protein folds considered in isolation [2].

Its representation and output formulation also move toward joint atom-level modeling of heterogeneous systems, enabling the network to reason about local geometry, chemical identity, and interaction-specific contacts within a common structure-prediction pipeline [2]. The term “all-atom” should not be interpreted as implying perfect stereochemical validity in every prediction.

AlphaFold3 replaces the AlphaFold2 structure module with a difusion module that denoises atom coordinates conditioned on processed input representations. Repeated random seeds can yield multiple candidate predictions, but the model remains prediction-oriented because the identities of the molecular entities are specified and the primary objective is to infer their joint structure. The presence of a difusion architecture therefore does not, by itself, place a model in the design-oriented generative phase [2].

AlphaFold3 also increases the importance of evaluating local chemical validity. The difusion formulation accommodates diverse chemical components and reduces reliance on molecule-specific output modules, yet residual stereochemical violations and clashes can still occur. Structural accuracy should therefore be assessed using both global or interfacelevel similarity and local chemistry-aware validation [2, 14].

## 4.5. Methodological Implications and Remaining Challenges

The transition from monomer folding to heterogeneous modeling introduces systematic changes across multiple aspects of the problem. These changes extend beyond representation and encompass the definition of prediction targets, the nature of available data, and the criteria used for evaluation.

To clarify these diferences, a structured comparison between monomer folding and heterogeneous molecular system modeling is provided in Table 3.

These changes introduce several challenges. High-quality training and benchmark data are unevenly distributed across complex types; representations must balance chemical detail with computational tractability; and evaluation must distinguish global assembly accuracy from interface fidelity, ligand pose quality, and local stereochemical validity [51, 19, 89].

Furthermore, broader modeling scope does not eliminate the value of specialized inductive biases. Antibody modeling, particularly in highly variable CDR loops, remains a prominent example in which domain-specific systems such as IgFold can ofer practical advantages [1, 65].

## 4.6. Transition Toward Generative Modeling

The progression summarized in Figure 4 also highlights a broader methodological trajectory. As models become capable of representing heterogeneous molecular entities within shared all-atom spaces and performing iterative refinement, the conceptual boundary between prediction and generation begins to diminish.

In this sense, the expansion toward heterogeneous modeling serves as a bridge to subsequent developments in generative structural modeling. By redefining the prediction target from isolated protein folds to interaction-rich molecular systems, this phase lays the foundation for approaches that treat structure not as a single deterministic outcome, but as part of a broader distribution of plausible configurations.

Table 3  
Comparison between monomer folding and heterogeneous molecular system modeling. The table contrasts the two phases in terms of system scope, representation, objective, data regime, and evaluation, highlighting the movement from single-chain residue-level prediction toward interaction-aware modeling of multi-molecular assemblies.
<table><tr><td>Dimension</td><td>Monomer folding (Phase II)</td><td>Heterogeneous modeling (Phase III)</td></tr><tr><td>System scope</td><td>Single protein chain</td><td>Multi-chain and multimolecular assemblies</td></tr><tr><td>Representation</td><td>Residue-level sequence-pair and frame representa- tions</td><td>Mixed token and atom-level heterogeneous represen- tations</td></tr><tr><td>Objective</td><td>Structure inference for a specified single-chain pro- tein</td><td>Joint structure inference for specified multimolecular assemblies</td></tr><tr><td>Data regime</td><td>Relatively richer monomer-focused structural data</td><td>More heterogeneous and unevenly sampled complex data</td></tr><tr><td>Evaluation</td><td>Global and local structural-similarity metrics</td><td>Global, interface-specific, and chemistry-aware met- rics</td></tr></table>

## 5. From Prediction-Oriented Structure Inference to Generative Protein Design

As illustrated in the revised Figure 4, the key distinction is not whether a model produces one or multiple samples, nor whether it uses a difusion module. It is whether the primary objective is inference for specified molecular entities or generation of novel design candidates under constraints.

## 5.1. Prediction-Oriented Structure Inference

Prediction-oriented structure models estimate conformations for specified molecular inputs. AlphaFold2 largely behaves as a point-estimation system that returns a ranked structural hypothesis with confidence metrics [40]. AlphaFold3, by contrast, uses difusion-based coordinate denoising and can generate diferent candidate predictions across seeds, while still remaining prediction-oriented because the sequence and molecular composition are given [2].

Confidence measures such as pLDDT [77] and predicted aligned error [2] quantify expected reliability of predicted regions or relative placements, but they should not be equated with a complete conformational ensemble [3, 84]. Likewise, variability across random seeds may reflect model sampling behavior [2] without necessarily reproducing the biologically populated distribution of states [84].

Prediction-oriented inference is highly efective when the scientific question concerns the likely structure of a specified target under conditions represented by the training data. Its limitations become more apparent when the target occupies multiple biologically relevant states or when context-dependent dynamics are central to function.

## 5.2. Structural Heterogeneity and Distributional Prediction

Many proteins and complexes occupy conformational ensembles shaped by thermodynamic fluctuations, molecular partners, and environmental conditions. A single highconfidence structural hypothesis may therefore be insufficient for intrinsically disordered regions, fold-switching proteins, flexible loops, or ligand-induced rearrangements [11, 58, 76].

This problem motivates a distinct line of work on distributional structure prediction, in which a model samples alternative conformations for a specified sequence. Such approaches should be distinguished from de novo protein design: both can use generative machinery, but one aims to approximate a target’s conformational distribution, whereas the other aims to create new molecular solutions. EigenFold is one representative example of sequence-conditioned distributional structure prediction [39].

Accordingly, neither point prediction nor repeated stochastic sampling should automatically be interpreted as a faithful physical ensemble. Validation against experimental ensemble data, molecular simulations, or state-specific benchmarks remains necessary [37, 68].

## 5.3. Generative Modeling for Protein Design

Generative protein-design models are defined primarily by their ability to create novel molecular candidates rather than merely by their ability to sample multiple structures. Depending on the task, they may generate protein backbones, amino acid sequences, side-chain arrangements, or joint sequence–structure representations under design constraints.

Difusion models [30] have become an influential design paradigm because conditional denoising can be guided by geometric motifs, target surfaces, symmetries, or other structural constraints [83]. Flow-matching [48] and autoregressive approaches [81] provide alternative routes to the same broad objective: learning a generative process over molecular design space.

RFdifusion generates novel protein backbones conditioned on structural or functional constraints and has been applied to motif scafolding, symmetric assemblies, and binder design [83]. Its primary contribution is therefore controllable de novo design rather than conformational sampling of a fixed natural sequence.

La-Proteina extends this direction through partially latent flow matching for atomistic protein generation and joint treatment of sequence–structure information [27]. RFdifusion and La-Proteina difer in representation and generation strategy, but both are better characterized as design-oriented generative systems than as generic uncertainty models.

Table 4  
Comparison between prediction-oriented structure inference and generative protein design. The table distinguishes the paradigms by conditioning, primary objective, output, and modeling mechanism; it deliberately avoids equating prediction with a single output or generation with multiple samples.
<table><tr><td>Dimension</td><td>Prediction-oriented structure inference</td><td>Generative protein design</td></tr><tr><td>Conditioning</td><td>Specified sequence and/or molecular com- position</td><td>Motifs, target geometry, interfaces, func- tion, and optional sequence constraints</td></tr><tr><td>Primary objective</td><td>Infer structures compatible with specified molecules</td><td>Create novel structures or sequence—structure solutions</td></tr><tr><td>Output</td><td>One or more candidate structures for the same specified target</td><td>Diverse de novo backbones, sequences, or joint designs</td></tr><tr><td>Inference/generation mechanism</td><td>Deterministic regression or diffusion-based sampling</td><td>Diffusion, flow matching, autoregressive, or other generative processes</td></tr><tr><td>Representative models</td><td>AlphaFold2, AlphaFold3</td><td>RFdiffusion, La-Proteina</td></tr></table>

## 5.4. Conceptual Diferences Between Prediction and Generation

The distinction between prediction-oriented inference and generative design therefore concerns conditioning and objective. Prediction starts from specified molecular identities and asks what structure or interaction geometry is plausible. Generative design starts from a desired motif, target interface, geometry, or functional objective and searches for novel molecular realizations. Either paradigm may produce multiple candidates, and either may use difusion.

This distinction has direct implications for evaluation. Predictive models are assessed primarily by correspondence to experimentally determined structures and by confidence calibration, whereas design models must additionally be evaluated for novelty, validity, diversity, designability, and experimental function [43, 49, 90].

To clarify these diferences, a structured comparison is provided in Table 4.

## 5.5. Complementarity and Emerging Unified Perspective

Prediction-oriented inference and generative design are complementary. Predictive models can evaluate whether a designed sequence is compatible with a target fold [40, 83] or interaction geometry [2], while generative models can propose candidates that satisfy specified constraints [83]. In practical design loops, prediction, generation, ranking, and experimental validation may therefore be iterated rather than treated as isolated stages [61, 71].

This complementarity contributes to a broader convergence between structural inference and protein engineering. Nevertheless, prediction of a specified structure and generation of a novel design remain diferent tasks, with diferent notions of success and diferent validation requirements [83, 27].

## 5.6. Challenges and Future Directions

Generative protein design introduces several challenges. Generated candidates must satisfy stereochemical, energetic, and folding constraints; computational metrics may not predict expression, stability, binding, or biological function; and diversity must be distinguished from invalid or redundant sampling. Evaluation therefore requires a combination of structural validation, sequence analysis, physical modeling, and experimental testing. [90, 9] Computational cost and sampling eficiency also remain important, especially for iterative difusion or flow-based systems [34, 91].

Despite these challenges, generative modeling represents a fundamental expansion of computational structural biology. It extends the field from inference about specified molecules toward the controlled creation and exploration of new molecular solutions. This design-oriented phase complements, rather than supersedes, predictive structure models.

## 6. Synthesis, Challenges, and Future Directions

## 6.1. Synthesis of methodological evolution

Protein structure prediction has undergone a profound methodological transformation over the past decades, evolving from physics-based and knowledge-based modeling to deep learning–driven inference and, more recently, to increasingly integrated and generative frameworks. As organized in this review, this development is best understood not simply as a sequence of progressively stronger models, but as a broader reconfiguration of how structural information is represented, learned, and utilized. The four methodological phases and three cross-cutting transitions discussed throughout the review provide a structured way to interpret this evolution beyond a purely chronological narrative.

At a fundamental level, the history of protein structure prediction reflects a continuing redefinition of what counts as the relevant source of structural information. Early approaches relied on explicit physical principles, statistical potentials, and optimization over energy landscapes [18, 6,

12, 72]. The subsequent rise of co-evolutionary analysis introduced a new form of sequence-derived structural signal, enabling residue-level constraints to be inferred from variation across homologous sequences [53, 54, 31]. In this regime, structural reasoning was still mediated by explicitly constructed features and modular pipelines, but the field had already shifted toward data-informed inference.

The integration of deep learning marked a decisive methodological turning point. Models such as AlphaFold2 and RoseTTAFold [40, 7] demonstrated that structure prediction could be reformulated as an end-to-end learning problem in which representation learning, geometric reasoning, and coordinate prediction are tightly coupled. In parallel, protein language models introduced a complementary route in which structural information was no longer extracted primarily through explicit evolutionary features, but learned implicitly from large-scale sequence corpora [47]. Together, these developments changed both the data regime and the representational logic of the field, shifting the emphasis from handcrafted evolutionary descriptors to learned internal representations.

A further transformation occurred when the scope of prediction expanded beyond isolated monomer folding. Models such as AlphaFold-Multimer [23], RoseTTAFoldNA [8], RoseTTAFold All-Atom [42], and AlphaFold3 [2] moved the field toward increasingly integrated modeling of complexes and heterogeneous molecular systems. In this setting, the target of prediction is no longer just a single protein fold, but the structural organization of multiple interacting entities, often including nucleic acids, ligands, and ions. This shift is particularly important because it aligns structure prediction more closely with biological reality, where function is frequently determined by interfaces, assemblies, and molecular context rather than by isolated folds alone.

More recently, design-oriented generative modeling has introduced another conceptual change. RFdifusion and La-Proteina use difusion or flow-matching mechanisms to generate novel backbones or sequence–structure solutions under design constraints [83, 27]. This direction extends structural modeling from inference toward generation, optimization, and molecular engineering.

Taken together, these developments suggest that the recent history of protein structure prediction is not merely a story of accuracy improvement. It is also a story of methodological expansion: from explicit evolutionary modeling to learned representations, from single-structure prediction to uncertainty-aware generation, and from isolated proteins to heterogeneous biomolecular systems. The phase-andtransition framework adopted in this review is intended to make these deeper shifts visible and to clarify how they shape the difering capabilities and practical roles of contemporary models.

## 6.2. Current challenges and unresolved issues

Despite these advances, several important challenges remain. The first concerns the availability and distribution of data. Monomer-focused structure prediction benefits from relatively rich structural and sequence resources, whereas high-quality examples of heterogeneous complexes, alternative conformations, and chemically diverse interactions are more unevenly distributed. This imbalance can bias training and evaluation toward well-represented subsets of biomolecular space [41, 67].

A second challenge concerns representation and interpretability. Explicit co-evolutionary features can be related to residue-pair constraints, whereas latent representations in PLMs, all-atom predictors, and generative models are harder to map onto biological mechanisms. Improved mechanistic interpretation would support model validation, failure analysis, and appropriate scientific use [33, 69, 79].

A third unresolved issue concerns evaluation. Metrics such as TM-score, RMSD, and lDDT remain useful across monomer and complex tasks, but they are insuficient on their own as modeling scope broadens. Complex prediction requires interface-aware assessment [55]; ligand-containing systems require pose and chemistry validation [14]; distributional prediction requires ensemble-aware benchmarks [38, 73]; and generative design requires measures of validity, novelty, diversity, designability, and experimental function [43, 49, 90, 9].

A fourth challenge concerns computational cost and accessibility. MSA-free inference reduces query-time database search in some systems, but large PLMs incur substantial pretraining and memory costs. Heterogeneous all-atom prediction and iterative generative sampling can also increase inference expense. Broader deployment therefore requires transparent reporting of hardware, preprocessing, sampling count, and runtime [16, 94].

Finally, an important conceptual challenge concerns the relationship between generality and specialization. Broader architectures do not necessarily dominate every specialized setting. Antibody modeling, especially for variable CDR regions, illustrates how domain-focused training and inductive biases may remain valuable [60, 65, 86]. Determining when a general model is suficient and when specialized modeling is justified will remain an important practical question.

## 6.3. Future directions and emerging opportunities

Several directions are likely to shape the next phase of development. One is tighter integration of prediction and design in iterative computational–experimental workflows, in which generative models propose candidates and predictive or experimental modules evaluate structural and functional constraints [28, 83].

Another important direction is the incorporation of dynamics, ensembles, and context dependence. Most current predictors emphasize static structural hypotheses, whereas many proteins function through conformational transitions or partner-dependent states. Progress will require ensembleaware training data, experimentally grounded benchmarks, and integration with statistical mechanics or molecular simulation [38, 73].

A third opportunity lies in multimodal integration. Structural predictions can potentially be constrained by cryo-EM density, cross-linking, mutational scans, biochemical assays, ligand-binding measurements, and other experimental evidence [4, 64, 88, 25].

A fourth likely direction is the development of hybrid frameworks that combine prediction distributional inference, and generative design without conflating their objectives. Predictive models may evaluate target compatibility [40, 2], distributional models may explore alternative states [39, 37], and design models may propose novel molecular solutions [83, 35].

Finally, future models will need greater emphasis on efficiency, robustness, reproducibility, and accessibility. Useful reporting standards should include training-data provenance, inference settings, sampling counts, uncertainty calibration, and hardware requirements [29, 80].

## 6.4. Closing perspective

Protein structure prediction is no longer a narrowly defined problem of mapping one sequence to one static fold. It has become a broader modeling discipline that spans learned representation, interaction-aware structural reasoning, and increasingly generative approaches to design and exploration. The methodological evolution reviewed here shows that recent progress is best understood not only in terms of benchmark performance, but also in terms of changing assumptions about data, representation, objective, and scope.

Seen from this perspective, the field is moving from a protein-centric prediction problem toward a more general framework for modeling biomolecular organization. At the same time, its future progress will depend on how efectively it addresses the challenges of data imbalance, interpretability, evaluation, and computational accessibility. The boundary between prediction and design is likely to become increasingly blurred, and the most influential future models may be those that integrate deterministic inference, heterogeneous system modeling, and generative sampling within coherent and practically usable frameworks. In that sense, the recent history of protein structure prediction is not only a record of rapid technical progress, but also a guide to the conceptual directions that are likely to shape its next phase.

## Acknowledgements

The authors are grateful to the support of the Guangdong Key Disciplines Project (2024ZDJS137).

## Declaration of competing interest

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## CRediT authorship contribution statement

Wengan He: Conceptualization, Methodology, Investigation, Data curation, Visualization, Writing – original draft . Yongsheng Luo: Conceptualization, Methodology, Investigation, Writing – original draft . Lihong Jiang: Validation, Writing – review & editing . Wenhui Xu: Validation, Writing – review & editing . Yu Li: Conceptualization, Supervision, Writing – review & editing .

## Data availability statement

No new datasets were generated or analyzed in this review. All information discussed in the article was obtained from publicly available publications, databases, and resources cited in the manuscript.

## Declaration of generative AI and AI-assisted technologies in the manuscript preparation process

During the preparation of this work, the authors used ChatGPT, provided by OpenAI, to assist with language refinement, structural organization, and the drafting of selected manuscript and submission-related text. After using this tool, the authors reviewed and edited the content as needed and take full responsibility for the content of the article.

## References

[1] Abanades, B., Wong, W.K., Boyles, F., Georges, G., Bujotzek, A., Deane, C.M., 2023. Immunebuilder: Deep-learning models for predicting the structures of immune proteins. Communications Biology 6, 575. URL: https://doi.org/10.1038/s42003-023-04927-7, doi:10. 1038/s42003-023-04927-7.

[2] Abramson, J., Adler, J., Dunger, J., Evans, R., Green, T., Pritzel, A., Ronneberger, O., Willmore, L., Ballard, A.J., Bambrick, J., Bodenstein, S.W., Evans, D.A., Hung, C.C., O’Neill, M., Reiman, D., Tunyasuvunakool, K., Wu, Z., Žemgulyte, A., Arvaniti, E., Beattie, C.,˙ Bertolli, O., Bridgland, A., Cherepanov, A., Congreve, M., Cowen-Rivers, A.I., Cowie, A., Figurnov, M., Fuchs, F.B., Gladman, H., Jain, R., Khan, Y.A., Low, C.M.R., Perlin, K., Potapenko, A., Savy, P., Singh, S., Stecula, A., Thillaisundaram, A., Tong, C., Yakneen, S., Zhong, E.D., Zielinski, M., Žídek, A., Bapst, V., Kohli, P., Jaderberg, M., Hassabis, D., Jumper, J.M., 2024. Accurate structure prediction of biomolecular interactions with alphafold 3. Nature 630, 493– 500. URL: https://doi.org/10.1038/s41586-024-07487-w, doi:10. 1038/s41586-024-07487-w.

[3] del Alamo, D., Sala, D., Mchaourab, H.S., Meiler, J., 2022. Sampling alternative conformational states of transporters and receptors with alphafold2. eLife 11, e75751. URL: https://doi.org/10.7554/eLife. 75751, doi:10.7554/eLife.75751.

[4] Alber, F., Förster, F., Korkin, D., Topf, M., Sali, A., 2008. Integrating diverse data for structure determination of macromolecular assemblies. Annual Review of Biochemistry 77, 443–477. URL: https://www.annualreviews.org/content/journals/ 10.1146/annurev.biochem.77.060407.135530, doi:https://doi.org/10. 1146/annurev.biochem.77.060407.135530.

[5] AlQuraishi, M., 2019. End-to-end diferentiable learning of protein structure. Cell Systems 8, 292–301.e3. URL: https: //www.sciencedirect.com/science/article/pii/S2405471219300766, doi:https://doi.org/10.1016/j.cels.2019.03.006.

[6] Anfinsen, C.B., 1973. Principles that govern the folding of protein chains. Science 181, 223– 230. URL: https://www.science.org/doi/abs/10.1126/ science.181.4096.223, doi:10.1126/science.181.4096.223, arXiv:https://www.science.org/doi/pdf/10.1126/science.181.4096.223

[7] Baek, M., DiMaio, F., Anishchenko, I., Dauparas, J., Ovchinnikov, S., Lee, G.R., Wang, J., Cong, Q., Kinch, L.N., Schaefer, R.D., Millán, C., Park, H., Adams, C., Glassman, C.R., DeGiovanni, A., Pereira, J.H., Rodrigues, A.V., van Dijk, A.A., Ebrecht, A.C., Opperman, D.J., Sagmeister, T., Buhlheller, C., Pavkov-Keller, T., Rathinaswamy, M.K., Dalwadi, U., Yip, C.K., Burke, J.E., Garcia, K.C., Grishin, N.V., Adams, P.D., Read, R.J., Baker, D., 2021. Accurate prediction of protein structures and interactions using a three-track neural network. Science 373, 871–876. URL: https://www.science. org/doi/abs/10.1126/science.abj8754, doi:10.1126/science.abj8754, arXiv:https://www.science.org/doi/pdf/10.1126/science.abj8754.

[8] Baek, M., McHugh, R., Anishchenko, I., Jiang, H., Baker, D., DiMaio, F., 2024. Accurate prediction of protein–nucleic acid complexes using rosettafoldna. Nature Methods 21, 117–121. URL: https://doi.org/ 10.1038/s41592-023-02086-5, doi:10.1038/s41592-023-02086-5.

[9] Barnett, A.J., KC, R., Pandey, P., Somasiri, P., Fairfax, K.A., Hung, S., Hewitt, A.W., 2026. Benchmarking generative ai protein models reveals diferences between structural and sequence-based approaches. Genomics, Proteomics & Bioinformatics , qzag014URL: https:// doi.org/10.1093/gpbjnl/qzag014, doi:10.1093/gpbjnl/qzag014.

[10] Basu, S., Wallner, B., 2016. Dockq: A quality measure for proteinprotein docking models. PLOS ONE 11, 1–9. URL: https://doi.org/ 10.1371/journal.pone.0161879, doi:10.1371/journal.pone.0161879.

[11] Boehr, D.D., Nussinov, R., Wright, P.E., 2009. The role of dynamic conformational ensembles in biomolecular recognition. Nature Chemical Biology 5, 789–796. URL: https://doi.org/10.1038/ nchembio.232, doi:10.1038/nchembio.232.

[12] Brini, E., Simmerling, C., Dill, K., 2020. Protein storytelling through physics. Science 370, eaaz3041. URL: https://www.science. org/doi/abs/10.1126/science.aaz3041, doi:10.1126/science.aaz3041, arXiv:https://www.science.org/doi/pdf/10.1126/science.aaz3041.

[13] Bryant, P., Pozzati, G., Elofsson, A., 2022. Improved prediction of protein-protein interactions using alphafold2. Nature Communications 13, 1265. URL: https://doi.org/10.1038/s41467-022-28865-w, doi:10.1038/s41467-022-28865-w.

[14] Buttenschoen, M., Morris, G.M., Deane, C.M., 2024. Posebusters: Ai-based docking methods fail to generate physically valid poses or generalise to novel sequences††electronic supplementary information (esi) available. see doi: https://doi.org/10.1039/d3sc04185a. Chemical Science 15, 3130–3139. URL: https://www.sciencedirect. com/science/article/pii/S2041652024002402, doi:https://doi.org/10. 1039/d3sc04185a.

[15] Cao, H., Zhou, X., Gao, Z., Wang, C., Gao, X., Zhang, Z., de la Fuente-Nunez, C., Gu, C., Liu, G., Heng, P.A., 2025. Lightweight msa design advances protein folding from evolutionary embeddings. URL: https://arxiv.org/abs/2507.07032, arXiv:2507.07032.

[16] Cheng, X., Chen, B., Li, P., Gong, J., Tang, J., Song, L., 2024. Training compute-optimal protein language models, in: Globerson, A., Mackey, L., Belgrave, D., Fan, A., Paquet, U., Tomczak, J., Zhang, C. (Eds.), Advances in Neural Information Processing Systems, Curran Associates, Inc.. pp. 69386–69418. URL: https://proceedings.neurips.cc/paper\_files/paper/2024/ file/8066ae1446b2bbccb5159587cc3b3bcc-Paper-Conference.pdf, doi:10.52202/079017-2216.

[17] wwPDB consortium, 2019. Protein data bank: the single global archive for 3d macromolecular structure data. Nucleic Acids Research 47, D520–D528. URL: https://doi.org/10.1093/nar/gky949, doi:10. 1093/nar/gky949.

[18] Dill, K.A., MacCallum, J.L., 2012. The protein-folding problem, 50 years on. Science 338, 1042–1046. URL: https://www.science. org/doi/abs/10.1126/science.1219021, doi:10.1126/science.1219021, arXiv:https://www.science.org/doi/pdf/10.1126/science.1219021.

[19] Durairaj, J., Adeshina, Y., Cao, Z., Zhang, X., Oleinikovas, V., Duignan, T., McClure, Z., Robin, X., Kovtun, D., Rossi, E., Zhou, G., Veccham, S., Isert, C., Peng, Y., Sundareson, P., Akdel, M., Corso, G., Stärk, H., Carpenter, Z., Bronstein, M., Kucukbenli, E., Schwede, T., Naef, L., 2024. Plinder: The protein-ligand interactions dataset and evaluation resource. bioRxiv URL: https://www.biorxiv.

org/content/early/2024/07/17/2024.07.17.603955, doi:10.1101/2024. 07.17.603955.

[20] Eddy, S.R., 2011. Accelerated profile hmm searches. PLOS Computational Biology 7, e1002195. URL: https: //app.dimensions.ai/details/publication/pub.1001034143, doi:10.1371/journal.pcbi.1002195.

[21] Elnaggar, A., Heinzinger, M., Dallago, C., Rehawi, G., Wang, Y., Jones, L., Gibbs, T., Feher, T., Angerer, C., Steinegger, M., Bhowmik, D., Rost, B., 2022. Prottrans: Toward understanding the language of life through self-supervised learning. IEEE Transactions on Pattern Analysis and Machine Intelligence 44, 7112–7127. doi:10.1109/ TPAMI.2021.3095381.

[22] Elofsson, A., 2023. Progress at protein structure prediction, as seen in casp15. Current Opinion in Structural Biology 80, 102594. URL: https://www.sciencedirect.com/science/article/pii/ S0959440X23000684, doi:https://doi.org/10.1016/j.sbi.2023.102594.

[23] Evans, R., O’Neill, M., Pritzel, A., Antropova, N., Senior, A., Green, T., Žídek, A., Bates, R., Blackwell, S., Yim, J., Ronneberger, O., Bodenstein, S., Zielinski, M., Bridgland, A., Potapenko, A., Cowie, A., Tunyasuvunakool, K., Jain, R., Clancy, E., Kohli, P., Jumper, J., Hassabis, D., 2021. Protein complex prediction with alphafoldmultimer. bioRxiv URL: https://www.biorxiv.org/content/early/ 2021/10/04/2021.10.04.463034, doi:10.1101/2021.10.04.463034.

[24] Fang, X., Wang, F., Liu, L., He, J., Lin, D., Xiang, Y., Zhu, K., Zhang, X., Wu, H., Li, H., Song, L., 2023. A method for multiplesequence-alignment-free protein structure prediction using a protein language model. Nature Machine Intelligence 5, 1087–1096. URL: http://dx.doi.org/10.1038/s42256-023-00721-6, doi:10.1038/ s42256-023-00721-6.

[25] Feng, S., Chen, Z., Zhang, C., Xie, Y., Ovchinnikov, S., Gao, Y.Q., Liu, S., 2024. Integrated structure prediction of protein– protein docking with experimental restraints using colabdock. Nature Machine Intelligence 6, 924–935. URL: https://doi.org/10.1038/ s42256-024-00873-z, doi:10.1038/s42256-024-00873-z.

[26] Finn, R.D., Clements, J., Eddy, S.R., 2011. Hmmer web server: interactive sequence similarity searching. Nucleic Acids Research 39, W29–W37. URL: https://doi.org/10.1093/nar/gkr367, doi:10.1093/ nar/gkr367.

[27] Gefner, T., Didi, K., Cao, Z., Reidenbach, D., Zhang, Z., Dallago, C., Kucukbenli, E., Kreis, K., Vahdat, A., 2026. La-proteina: Atomistic protein generation via partially latent flow matching. URL: https: //arxiv.org/abs/2507.09466, arXiv:2507.09466.

[28] He, B., Qin, C., Zhao, Y., Huang, L.K., Wu, Z., Wang, F., Wu, F., Yang, F., Yao, J., 2026. Functional protein design and enhancement with ontology reinforcement iteration. Nature Communications 17, 4158. URL: https://doi.org/10.1038/s41467-026-69855-6, doi:10. 1038/s41467-026-69855-6

[29] Heil, B.J., Hofman, M.M., Markowetz, F., Lee, S.I., Greene, C.S., Hicks, S.C., 2021. Reproducibility standards for machine learning in the life sciences. Nature Methods 18, 1132–1135. URL: https://doi. org/10.1038/s41592-021-01256-7, doi:10.1038/s41592-021-01256-7.

[30] Ho, J., Jain, A., Abbeel, P., 2020. Denoising difusion probabilistic models, in: Larochelle, H., Ranzato, M., Hadsell, R., Balcan, M., Lin, H. (Eds.), Advances in Neural Information Processing Systems, Curran Associates, Inc.. pp. 6840–6851. URL: https://proceedings.neurips.cc/paper\_files/paper/2020/file 4c5bcfec8584af0d967f1ab10179ca4b-Paper.pdf.

[31] Hopf, T.A., Schärfe, C.P.I., Rodrigues, J.P.G.L.M., Green, A.G., Kohlbacher, O., Sander, C., Bonvin, A.M.J.J., Marks, D.S., 2014. Sequence co-evolution gives 3d contacts and structures of protein complexes. eLife 3, e03430. URL: https://doi.org/10.7554/eLife. 03430, doi:10.7554/eLife.03430.

[32] Hu, B., Xia, J., Zheng, J., Tan, C., Huang, Y., Xu, Y., Li, S.Z., 2022. Protein language models and structure prediction: Connection and progression. URL: https://arxiv.org/abs/2211.16742, arXiv:2211.16742.

[33] Hunklinger, A., Ferruz, N., 2026. Towards the explainability of protein language models. Nature Machine Intelligence 8, 649– 662. URL: https://doi.org/10.1038/s42256-026-01232-w, doi:10. 1038/s42256-026-01232-w.

[34] Jendrusch, M., Korbel, J.O., 2025. Eficient protein structure generation with sparse denoising models. Nature Machine Intelligence 7, 1429–1445. URL: https://doi.org/10.1038/s42256-025-01100-z, doi:10.1038/s42256-025-01100-z.

[35] Jendrusch, M.A., Yang, A.L.J., Cacace, E., Bobonis, J., Voogdt, C.G.P., Kaspar, S., Schweimer, K., Perez-Borrajero, C., Lapouge, K., Scheurich, J., Remans, K., Hennig, J., Typas, A., Korbel, J.O., Sadiq, S.K., 2025. Alphadesign: a de novo protein design framework based on alphafold. Molecular Systems Biology 21, 1166– 1189. URL: https://doi.org/10.1038/s44320-025-00119-z, doi:10. 1038/s44320-025-00119-z.

[36] Jiang, Y., Li, X., Zhang, Y., Han, J., Xu, Y., Pandit, A., ZHANG, Z., Wang, M., Wang, M., Liu, C., Yang, G., Choi, Y., Lu, Y., Li, W.J., Fu, T., Wu, F., Liu, J., 2026. Posex: AI defeats physicsbased methods on protein ligand cross-docking, in: The Fourteenth International Conference on Learning Representations. URL: https: //openreview.net/forum?id=qqzxKudD4T.

[37] Jing, B., Berger, B., Jaakkola, T., 2024. Alphafold meets flow matching for generating protein ensembles, in: Proceedings of the 41st International Conference on Machine Learning, JMLR.org.

[38] Jing, B., Berger, B., Jaakkola, T., 2026. Ai-based methods for simulating, sampling, and predicting protein ensembles. Current Opinion in Structural Biology 98, 103251. URL: https://www.sciencedirect. com/science/article/pii/S0959440X26000333, doi:https://doi.org/10. 1016/j.sbi.2026.103251.

[39] Jing, B., Erives, E., Pao-Huang, P., Corso, G., Berger, B., Jaakkola, T., 2023. Eigenfold: Generative protein structure prediction with difusion models. URL: https://arxiv.org/abs/2304.02198, arXiv:2304.02198.

[40] Jumper, J., Evans, R., Pritzel, A., Green, T., Figurnov, M., Ronneberger, O., Tunyasuvunakool, K., Bates, R., Žídek, A., Potapenko, A., Bridgland, A., Meyer, C., Kohl, S.A.A., Ballard, A.J., Cowie, A., Romera-Paredes, B., Nikolov, S., Jain, R., Adler, J., Back, T., Petersen, S., Reiman, D., Clancy, E., Zielinski, M., Steinegger, M., Pacholska, M., Berghammer, T., Bodenstein, S., Silver, D., Vinyals, O., Senior, A.W., Kavukcuoglu, K., Kohli, P., Hassabis, D., 2021. Highly accurate protein structure prediction with alphafold. Nature 596, 583–589. URL: https://doi.org/10.1038/s41586-021-03819-2, doi:10.1038/s41586-021-03819-2.

[41] Kosoglu, K., Aydin, Z., Tuncbag, N., Gursoy, A., Keskin, O., 2024. Structural coverage of the human interactome. Briefings in Bioinformatics 25, bbad496. URL: https://doi.org/10.1093/bib/bbad496, doi:10.1093/bib/bbad496.

[42] Krishna, R., Wang, J., Ahern, W., Sturmfels, P., Venkatesh, P., Kalvet, I., Lee, G.R., Morey-Burrows, F.S., Anishchenko, I., Humphreys, I.R., McHugh, R., Vafeados, D., Li, X., Sutherland, G.A., Hitchcock, A., Hunter, C.N., Kang, A., Brackenbrough, E., Bera, A.K., Baek, M., DiMaio, F., Baker, D., 2024. Generalized biomolecular modeling and design with rosettafold allatom. Science 384, eadl2528. URL: https://www.science. org/doi/abs/10.1126/science.adl2528, doi:10.1126/science.adl2528, arXiv:https://www.science.org/doi/pdf/10.1126/science.adl2528.

[43] Kuang, J., Liu, N., Wang, J., Sun, C., Ji, T., Wu, Y., 2026. Pdfbench: A benchmark for de novo protein design from function. URL: https: //arxiv.org/abs/2505.20346, arXiv:2505.20346.

[44] Laine, E., Eismann, S., Elofsson, A., Grudinin, S., 2021. Protein sequence-to-structure learning: Is this the end(-to-end revolution)? URL: https://arxiv.org/abs/2105.07407, arXiv:2105.07407.

[45] Larkin, M., Blackshields, G., Brown, N., Chenna, R., McGettigan, P., McWilliam, H., Valentin, F., Wallace, I., Wilm, A., Lopez, R., Thompson, J., Gibson, T., Higgins, D., 2007. Clustal w and clustal x version 2.0. Bioinformatics 23, 2947–2948. URL: https://doi.org/ 10.1093/bioinformatics/btm404, doi:10.1093/bioinformatics/btm404.

[46] Liberti, L., Lavor, C., Maculan, N., Mucherino, A., 2014. Euclidean distance geometry and applications. SIAM Review 56, 3– 69. URL: https://doi.org/10.1137/120875909, doi:10.1137/120875909, arXiv:https://doi.org/10.1137/120875909.

[47] Lin, Z., Akin, H., Rao, R., Hie, B., Zhu, Z., Lu, W., Smetanin, N., Verkuil, R., Kabeli, O., Shmueli, Y., dos Santos Costa, A., Fazel-Zarandi, M., Sercu, T., Candido, S., Rives, A., 2023. Evolutionaryscale prediction of atomic-level protein structure with a language model. Science 379, 1123–1130. URL: https://www.science. org/doi/abs/10.1126/science.ade2574, doi:10.1126/science.ade2574, arXiv:https://www.science.org/doi/pdf/10.1126/science.ade2574.

[48] Lipman, Y., Chen, R.T.Q., Ben-Hamu, H., Nickel, M., Le, M., 2023. Flow matching for generative modeling. URL: https://arxiv.org/ abs/2210.02747, arXiv:2210.02747.

[49] Liu, C., Ren, M., Guan, J., Gong, C., Sun, J., Chen, X., Xiao, W., 2026. Protdbench: A unified benchmark of protein binder design and evaluation. URL: https://arxiv.org/abs/2605.04118, arXiv:2605.04118.

[50] Liu, L., Zhang, S., Xue, Y., Ye, X., Zhu, K., Li, Y., Liu, Y., Gao, J., Zhao, W., Yu, H., Wu, Z., Zhang, X., Fang, X., 2024. Technical report of helixfold3 for biomolecular structure prediction. URL: https://arxiv.org/abs/2408.16975, arXiv:2408.16975.

[51] Liu, Z., Li, Y., Han, L., Li, J., Liu, J., Zhao, Z., Nie, W., Liu, Y., Wang, R., 2015. Pdb-wide collection of binding data: current status of the pdbbind database. Bioinformatics 31, 405–412. URL: https:// doi.org/10.1093/bioinformatics/btu626, doi:10.1093/bioinformatics/ btu626.

[52] Mariani, V., Biasini, M., Barbato, A., Schwede, T., 2013. lddt: a local superposition-free score for comparing protein structures and models using distance diference tests. Bioinformatics 29, 2722– 2728. URL: https://doi.org/10.1093/bioinformatics/btt473, doi:10. 1093/bioinformatics/btt473.

[53] Marks, D.S., Colwell, L.J., Sheridan, R., Hopf, T.A., Pagnani, A., Zecchina, R., Sander, C., 2011. Protein 3d structure computed from evolutionary sequence variation. PloS one 6, e28766. URL: https://europepmc.org/articles/PMC3233603, doi:10. 1371/journal.pone.0028766.

[54] Marks, D.S., Hopf, T.A., Sander, C., 2012. Protein structure prediction from sequence variation. Nature Biotechnology 30, 1072–1080. URL: https://doi.org/10.1038/nbt.2419, doi:10.1038/nbt.2419.

[55] Mirabello, C., Wallner, B., 2024. Dockq v2: improved automatic quality measure for protein multimers, nucleic acids, and small molecules. Bioinformatics 40, btae586. URL: https://doi.org/10. 1093/bioinformatics/btae586, doi:10.1093/bioinformatics/btae586.

[56] Mirdita, M., Schütze, K., Moriwaki, Y., Heo, L., Ovchinnikov, S., Steinegger, M., 2022. Colabfold: making protein folding accessible to all. Nature Methods 19, 679–682. URL: https://doi.org/10.1038/ s41592-022-01488-1, doi:10.1038/s41592-022-01488-1.

[57] Morehead, A., Giri, N., Liu, J., Neupane, P., Cheng, J., 2026. Assessing the potential of deep learning for protein–ligand docking. Nature Machine Intelligence 8, 32–41. URL: https://doi.org/10. 1038/s42256-025-01160-1, doi:10.1038/s42256-025-01160-1.

[58] Motlagh, H.N., Wrabl, J.O., Li, J., Hilser, V.J., 2014. The ensemble nature of allostery. Nature 508, 331–339. URL: https://doi.org/10. 1038/nature13001, doi:10.1038/nature13001.

[59] Pearce, R., Zhang, Y., 2021. Deep learning techniques have significantly impacted protein structure prediction and protein design. Current Opinion in Structural Biology 68, 194– 207. URL: https://www.sciencedirect.com/science/article/pii/ S0959440X21000142, doi:https://doi.org/10.1016/j.sbi.2021.01.007. protein-Carbohydrate Complexes and Glycosylation Sequences and Topology.

[60] Polonsky, K., Pupko, T., Freund, N.T., 2023. Evaluation of the ability of alphafold to predict the three-dimensional structures of antibodies and epitopes. The Journal of Immunology 211, 1578– 1588. URL: https://doi.org/10.4049/jimmunol.2300150, doi:10.4049/ jimmunol.2300150.

[61] Rapp, J.T., Bremer, B.J., Romero, P.A., 2024. Self-driving laboratories to autonomously navigate the protein fitness landscape. Nature

Chemical Engineering 1, 97–107. URL: https://doi.org/10.1038/ s44286-023-00002-4, doi:10.1038/s44286-023-00002-4.

[62] Remmert, M., Biegert, A., Hauser, A., Söding, J., 2012. Hhblits: lightning-fast iterative protein sequence searching by hmm-hmm alignment. Nature Methods 9, 173–175. URL: https://doi.org/10. 1038/nmeth.1818, doi:10.1038/nmeth.1818.

[63] Rennie, M.L., Oliver, M.R., 2025. Emerging frontiers in protein structure prediction following the alphafold revolution. Journal of The Royal Society Interface 22, 20240886. URL: https://doi.org/ 10.1098/rsif.2024.0886, doi:10.1098/rsif.2024.0886.

[64] Rout, M.P., Sali, A., 2019. Principles for integrative structural biology studies. Cell 177, 1384–1403. URL: https://www.sciencedirect. com/science/article/pii/S0092867419305148, doi:https://doi.org/10. 1016/j.cell.2019.05.016.

[65] Rufolo, J.A., Chu, L.S., Mahajan, S.P., Gray, J.J., 2023. Fast, accurate antibody structure prediction from deep learning on massive set of natural antibodies. Nature Communications 14, 2389. URL: https://doi.org/10.1038/s41467-023-38063-x, doi:10. 1038/s41467-023-38063-x.

[66] Senior, A.W., Evans, R., Jumper, J., Kirkpatrick, J., Sifre, L., Green, T., Qin, C., Žídek, A., Nelson, A.W.R., Bridgland, A., Penedones, H., Petersen, S., Simonyan, K., Crossan, S., Kohli, P., Jones, D.T., Silver, D., Kavukcuoglu, K., Hassabis, D., 2020. Improved protein structure prediction using potentials from deep learning. Nature 577, 706–710. URL: https://doi.org/10.1038/s41586-019-1923-7, doi:10.1038/s41586-019-1923-7.

[67] Shinada, N.K., Schmidtke, P., de Brevern, A.G., 2020. Accurate representation of protein-ligand structural diversity in the protein data bank (pdb). International Journal of Molecular Sciences 21. URL: https: //www.mdpi.com/1422-0067/21/6/2243, doi:10.3390/ijms21062243.

[68] Monteiro da Silva, G., Cui, J.Y., Dalgarno, D.C., Lisi, G.P., Rubenstein, B.M., 2024. High-throughput prediction of protein conformational distributions with subsampled alphafold2. Nature Communications 15, 2464. URL: https://doi.org/10.1038/s41467-024-46715-9, doi:10.1038/s41467-024-46715-9.

[69] Simon, E., Zou, J., 2025. Interplm: discovering interpretable features in protein language models via sparse autoencoders. Nature Methods 22, 2107–2117. URL: https://doi.org/10.1038/s41592-025-02836-7, doi:10.1038/s41592-025-02836-7.

[70] Simons, K.T., Kooperberg, C., Huang, E., Baker, D., 1997. Assembly of protein tertiary structures from fragments with similar local sequences using simulated annealing and bayesian scoring functions. Journal of Molecular Biology 268, 209– 225. URL: https://www.sciencedirect.com/science/article/pii/ S0022283697909591, doi:https://doi.org/10.1006/jmbi.1997.0959.

[71] Singh, N., Lane, S., Yu, T., Lu, J., Ramos, A., Cui, H., Zhao, H., 2025. A generalized platform for artificial intelligence-powered autonomous enzyme engineering. Nature Communications 16, 5648. URL: https://doi.org/10.1038/s41467-025-61209-y, doi:10. 1038/s41467-025-61209-y.

[72] Sippl, M.J., 1990. Calculation of conformational ensembles from potentials of mean force: An approach to the knowledge-based prediction of local structures in globular proteins. Journal of Molecular Biology 213, 859–883. URL: https://www.sciencedirect. com/science/article/pii/S0022283605802694, doi:https://doi.org/10. 1016/S0022-2836(05)80269-4.

[73] Sledzieski, S., Hanson, S.M., 2026. The landscape of machine learning approaches for modeling protein conformational ensembles. Current Opinion in Structural Biology 98, 103253. URL: https: //www.sciencedirect.com/science/article/pii/S0959440X26000357, doi:https://doi.org/10.1016/j.sbi.2026.103253.

[74] Steinegger, M., Söding, J., 2017. Mmseqs2 enables sensitive protein sequence searching for the analysis of massive data sets. Nature Biotechnology 35, 1026–1028. URL: https://doi.org/10.1038/nbt. 3988, doi:10.1038/nbt.3988.

[75] team, C.D., Boitreaud, J., Dent, J., McPartlon, M., Meier, J., Reis, V., Rogozhonikov, A., Wu, K., 2024. Chai-1: Decoding the molecular interactions of life. bioRxiv URL: https://www.biorxiv.org/content/

early/2024/10/11/2024.10.10.615955, doi:10.1101/2024.10.10.615955.

[76] Thomasen, F.E., Lindorf-Larsen, K., 2022. Conformational ensembles of intrinsically disordered proteins and flexible multidomain proteins. Biochemical Society Transactions 50, 541– 554. URL: https://www.sciencedirect.com/science/article/pii/ S1470875222001143, doi:https://doi.org/10.1042/BST20210499.

[77] Tunyasuvunakool, K., Adler, J., Wu, Z., Green, T., Zielinski, M., Žídek, A., Bridgland, A., Cowie, A., Meyer, C., Laydon, A., Velankar, S., Kleywegt, G.J., Bateman, A., Evans, R., Pritzel, A., Figurnov, M., Ronneberger, O., Bates, R., Kohl, S.A.A., Potapenko, A., Ballard, A.J., Romera-Paredes, B., Nikolov, S., Jain, R., Clancy, E., Reiman, D., Petersen, S., Senior, A.W., Kavukcuoglu, K., Birney, E., Kohli, P., Jumper, J., Hassabis, D., 2021. Highly accurate protein structure prediction for the human proteome. Nature 596, 590– 596. URL: https://doi.org/10.1038/s41586-021-03828-1, doi:10. 1038/s41586-021-03828-1.

[78] Vendruscolo, M., Kussell, E., Domany, E., 1997. Recovery of protein structure from contact maps. Folding and Design 2, 295–306. URL: https://www.sciencedirect.com/science/article/ pii/S1359027897000412, doi:https://doi.org/10.1016/S1359-0278(97) 00041-2.

[79] Vig, J., Madani, A., Varshney, L.R., Xiong, C., Socher, R., Rajani, N., 2021. BERTology Meets Biology: Interpreting Attention in Protein Language Models, in: International Conference on Learning Representations. URL: https://mlanthology.org/iclr/ 2021/vig2021iclr-bertology/.

[80] Walsh, I., Fishman, D., Garcia-Gasulla, D., Titma, T., Pollastri, G., Capriotti, E., Casadio, R., Capella-Gutierrez, S., Cirillo, D., Del Conte, A., Dimopoulos, A.C., Del Angel, V.D., Dopazo, J., Fariselli, P., Fernández, J.M., Huber, F., Kreshuk, A., Lenaerts, T., Martelli, P.L., Navarro, A., Broin, P.Ó., Piñero, J., Piovesan, D., Reczko, M., Ronzano, F., Satagopam, V., Savojardo, C., Spiwok, V., Tangaro, M.A., Tartari, G., Salgado, D., Valencia, A., Zambelli, F., Harrow, J., Psomopoulos, F.E., Tosatto, S.C.E., Group, E.M.L.F., 2021. Dome: recommendations for supervised machine learning validation in biology. Nature Methods 18, 1122– 1127. URL: https://doi.org/10.1038/s41592-021-01205-4, doi:10. 1038/s41592-021-01205-4.

[81] Wang, C., Alamdari, S., Domingo-Enrich, C., Amini, A.P., Yang, K.K., 2025. Toward deep learning sequence–structure co-generation for protein design. Current Opinion in Structural Biology 91, 103018. URL: https://www.sciencedirect.com/science/article/pii/ S0959440X25000363, doi:https://doi.org/10.1016/j.sbi.2025.103018.

[82] Wang, S., Sun, S., Li, Z., Zhang, R., Xu, J., 2017. Accurate de novo prediction of protein contact map by ultra-deep learning model. PLOS Computational Biology 13, e1005324. URL: https:// app.dimensions.ai/details/publication/pub.1010491660, doi:10.1371/ journal.pcbi.1005324. https://doi.org/10.1371/journal.pcbi.1005324.

[83] Watson, J.L., Juergens, D., Bennett, N.R., Trippe, B.L., Yim, J., Eisenach, H.E., Ahern, W., Borst, A.J., Ragotte, R.J., Milles, L.F., Wicky, B.I.M., Hanikel, N., Pellock, S.J., Courbet, A., Shefler, W., Wang, J., Venkatesh, P., Sappington, I., Torres, S.V., Lauko, A., De Bortoli, V., Mathieu, E., Ovchinnikov, S., Barzilay, R., Jaakkola, T.S., DiMaio, F., Baek, M., Baker, D., 2023. De novo design of protein structure and function with rfdifusion. Nature 620, 1089– 1100. URL: https://doi.org/10.1038/s41586-023-06415-8, doi:10. 1038/s41586-023-06415-8.

[84] Wayment-Steele, H.K., Ojoawo, A., Otten, R., Apitz, J.M., Pitsawong, W., Hömberger, M., Ovchinnikov, S., Colwell, L., Kern, D., 2024. Predicting multiple conformations via sequence clustering and alphafold2. Nature 625, 832–839. URL: https://doi.org/10.1038/ s41586-023-06832-9, doi:10.1038/s41586-023-06832-9.

[85] Wohlwend, J., Corso, G., Passaro, S., Reveiz, M., Leidal, K., Swiderski, W., Portnoi, T., Chinn, I., Silterra, J., Jaakkola, T., Barzilay, R., 2024. Boltz-1 democratizing biomolecular interaction modeling. bioRxiv URL: https://www.biorxiv.org/content/early/2024/11/ 20/2024.11.19.624167, doi:10.1101/2024.11.19.624167.

[86] Woo, H., Kim, Y., Seok, C., 2024. Protein loop structure prediction by community-based deep learning and its application to antibody cdr h3 loop modeling. PLOS Computational Biology 20, 1–17. URL: https://doi.org/10.1371/journal.pcbi.1012239, doi:10.1371/journal. pcbi.1012239.

[87] Wu, R., Ding, F., Wang, R., Shen, R., Zhang, X., Luo, S., Su, C., Wu, Z., Xie, Q., Berger, B., Ma, J., Peng, J., 2022. Highresolution de novo structure prediction from primary sequence. bioRxiv URL: https://www.biorxiv.org/content/early/2022/07/22/ 2022.07.21.500999, doi:10.1101/2022.07.21.500999.

[88] Xie, Y., Zhang, C., Li, S., Du, X., Lu, Y., Wang, M., Hu, Y., Chen, Z., Liu, S., Gao, Y.Q., 2025. Integrating diverse experimental information to assist protein complex structure prediction by grasp. Nature Methods 22, 2362–2374. URL: https://doi.org/10.1038/ s41592-025-02820-1, doi:10.1038/s41592-025-02820-1.

[89] Xu, S., Feng, Q., Qiao, L., Wu, H., Shen, T., Cheng, Y., Zheng, S., Sun, S., 2025. Benchmarking all-atom biomolecular structure prediction with foldbench. Nature Communications 17, 442. URL: https://doi. org/10.1038/s41467-025-67127-3, doi:10.1038/s41467-025-67127-3.

[90] Ye, F., Zheng, Z., Xue, D., Shen, Y., Wang, L., Ma, Y., Wang, Y., Wang, X., Zhou, X., Gu, Q., 2024. Proteinbench: A holistic evaluation of protein foundation models. URL: https://arxiv.org/abs/2409. 06744, arXiv:2409.06744.

[91] Yue, A., Wang, Z., Xu, H., 2025. Reqflow: Rectified quaternion flow for eficient and high-quality protein backbone generation. URL: https://arxiv.org/abs/2502.14637, arXiv:2502.14637.

[92] Zhang, Y., Skolnick, J., 2004. Scoring function for automated assessment of protein structure template quality. Proteins: Structure, Function, and Bioinformatics 57, 702– 710. URL: https://onlinelibrary.wiley.com/doi/abs/10. 1002/prot.20264, doi:https://doi.org/10.1002/prot.20264, arXiv:https://onlinelibrary.wiley.com/doi/pdf/10.1002/prot.20264.

[93] Zhu, W., Shenoy, A., Kundrotas, P., Elofsson, A., 2023. Evaluation of alphafold-multimer prediction on multi-chain protein complexes. Bioinformatics 39, btad424. URL: https://doi.org/10.1093/ bioinformatics/btad424, doi:10.1093/bioinformatics/btad424.

[94] Çelik, M.H., Xie, X., 2025. Eficient inference, training, and fine-tuning of protein language models. iScience 28, 113495. URL: https://www.sciencedirect.com/science/article/pii/ S2589004225017560, doi:https://doi.org/10.1016/j.isci.2025.113495.

[95] Šali, A., Blundell, T.L., 1993. Comparative protein modelling by satisfaction of spatial restraints. Journal of Molecular Biology 234, 779–815. URL: https://www.sciencedirect.com/science/article/ pii/S0022283683716268, doi:https://doi.org/10.1006/jmbi.1993.1626.