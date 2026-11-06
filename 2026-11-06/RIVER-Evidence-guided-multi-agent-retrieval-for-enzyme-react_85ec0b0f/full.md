# RIVER: Evidence-guided multi-agent retrieval for enzyme–reaction discovery

Yifan Chang<sup>1,2,3</sup>, Yunze Li<sup>2</sup>, Feifei Jiang<sup>2,6</sup>, Juntao Wu<sup>2</sup>, Xin Chen<sup>2</sup>, Han Hu<sup>2,7</sup>,

Yucheng Wu<sup>2,7</sup>, Wei Tang<sup>2</sup>, Mingqian Ma<sup>2,8</sup>, Wanli Ouyang<sup>2,9</sup>,

Kaipeng Zhang<sup>1</sup>, Lei Bai<sup>2</sup>, Hongting Tang<sup>5</sup>, Feiran Li<sup>4</sup>, Yuchen Ren<sup>2,\*</sup>

<sup>1</sup>Shanghai Innovation Institute <sup>2</sup>Shanghai AI Lab <sup>3</sup>University of Science and Technology of China <sup>4</sup>Tsinghua University <sup>5</sup>Sun Yat-sen University <sup>6</sup>Shanghai Jiao Tong University <sup>7</sup>Fudan University <sup>8</sup>Carnegie Mellon University <sup>9</sup>The Chinese University of Hong Kong Corresponding author: Yuchen Ren (renyuchen@pjlab.org.cn)

## Abstract

Enzyme function annotation and the identification of candidate enzymes for a given reaction are essential steps in connecting genomic information to biochemical reaction networks. However, existing methods typically rely on sequence or structural similarity or on training with known enzyme–reaction pairs, making it dificult to integrate existing biochemical evidence across large candidate spaces; they also struggle with retrieval tasks involving enzyme families whose members are highly similar. Here, we introduce RIVER (Reaction–enzyme Inference via Evidence-guided Retrieval), a multi-agent framework for bidirectional enzyme–reaction retrieval that identifies potentially catalyzed reactions for a given enzyme and ranks candidate enzymes for a given reaction. RIVER coordinates large language models, specialized biochemical computational tools, and external knowledge resources to assemble evidence at multiple levels, including reaction chemistry, protein sequences and domains, catalytic mechanisms, and database annotations. Dedicated validation modules further examine evidence gaps and potential conflicts and rerank candidates using traceable evidence records. On public benchmarks, RIVER achieves the highest success rates in both enzyme retrieval and enzyme annotation. In family-specific enzyme retrieval, RIVER outperforms the evaluated baselines on the selected datasets and metrics. In Escherichia coli enzyme retrieval tasks constructed from recent literature, RIVER achieves the best retrieval results with both closed-source and open-source model configurations, substantially outperforming closed-source generalpurpose and biology-specialized multi-agent systems, including Claude Code, Codex, and Phylo Max. These results suggest that evidence-guided multi-agent retrieval can prioritize biochemically supported enzyme–reaction relationships, providing a basis for subsequent experimental validation.

## 1 Introduction

Enzyme–reaction relationships provide mechanistic links between genomic information and biochemical transformations. They underpin protein function annotation, genome-scale metabolic reconstruction, pathway analysis, and biocatalyst discovery for chemical and industrial applications [1, 2]. However, the increasing abundance of protein sequences and metabolomics data has not eliminated the functional annotation bottleneck: many proteins remain insuficiently characterized, and standard annotations may not adequately capture the breadth of catalytic promiscuity or substrate-specific activities [3–7]. Experimental metabolomics screens have begun to reveal previously hidden enzyme activities at scale, while also highlighting the challenge of prioritizing informative candidates for mechanistic follow-up studies [8, 9].

Computational approaches to enzyme function encompass sequence homology and annotation transfer, sequence-based enzyme classification, structure-informed prediction, and reaction-conditioned retrieval. Similarity searches and pathway-oriented enzyme selection remain valuable for locating relevant catalysts [10, 11]; DeepEC and CLEAN infer Enzyme Commission (EC) functions from sequences [12, 13], whereas graph-based and geometric approaches use structural information to predict function or catalytic classes [14, 15]. Methods that explicitly represent substrates or reactions, including ESP, CLIPZyme, and EnzymeMap, address chemical compatibility more directly [16–18]. Recent work such as EnzymeCAGE and EnzGFM has advanced enzyme retrieval and enzyme-oriented foundation model research, whereas RXNRECer and ReactZyme emphasize fine-grained enzyme–reaction annotation and reaction-level evaluation [19–22]. These methods address overlapping but distinct tasks, and existing approaches should not be uniformly reduced to similarity matching. Nevertheless, a predicted label or ranking score does not necessarily distinguish directly documented catalytic activity from mechanistic analogy or unresolved substrate specificity, particularly when diferent methods use diferent candidate pools or evaluation protocols. This distinction matters when computational predictions are used to prioritize candidates for biochemical research.

Evidence relevant to enzyme–reaction hypotheses is distributed across resources with diferent scopes. UniProtKB records protein annotations, Rhea curates chemically well-defined biochemical transformations, and BRENDA compiles enzyme properties [23–25]; protein language models and structural resources provide complementary information, especially for poorly annotated proteins [26–30]. However, the availability and specificity of these records vary. The absence of annotation does not imply the absence of activity, just as a chemically plausible mechanism does not establish that a particular protein actually catalyzes the queried reaction. Retrieval-augmented language models and tool-using agents ofer ways to integrate dispersed information [31, 32]; chemistry-oriented agents and scientific multi-agent systems have demonstrated tool use, hypothesis generation, and critical review in other settings [33–37]. In enzyme research, EnzGFM-Agent provides a relevant precedent for agent-based approaches [20]. Thus, an important methodological opportunity is not simply to combine multiple agents, but to organize evidence acquisition, biochemical interpretation, review of missing or potentially conflicting support, and candidate reranking into a unified and inspectable retrieval workflow. Such a workflow should make the reasons for candidate prioritization traceable, while avoiding the conflation of supporting evidence with experimental confirmation of catalytic activity.

Here, we present RIVER (Reaction–enzyme Inference via Evidence-guided Retrieval), a training-free multi-agent framework that integrates bidirectional enzyme–reaction retrieval through a unified evidence guided approach. The two directions address complementary scientific needs: enzyme-to-reaction retrieval supports protein function annotation and metabolic network reconstruction, whereas reaction-to-enzyme retrieval supports the identification of candidate biocatalysts for specific chemical transformations (Fig. 1a). RIVER coordinates large language models (LLMs), external knowledge resources, and biochemical computational tools to assemble heterogeneous evidence spanning reaction chemistry, protein sequences and domains, catalytic mechanisms, and database annotations. Dedicated review modules examine evidence gaps and potential conflicts and use the resulting assessments to inform candidate reranking, ultimately producing ranked candidates with traceable, candidate-specific rationales (Fig. 1b). In E. coli enzyme retrieval tasks constructed using data and findings from recent literature, our multi-agent system accurately retrieves enzymes corresponding to biochemical reactions, outperforming general-purpose and biochemistry-specialized multiagent systems, including Claude Code, Codex, and Phylo Max, while demonstrating robustness to the choice of foundation model (Fig. 1c). On public benchmarks, we substantially outperform existing baselines in both enzyme retrieval and enzyme annotation; our multi-agent system also demonstrates a clear advantage in the challenging setting of family-specific enzyme retrieval.

## 2 Results

## 2.1 RIVER in experimentally validated E. coli enzyme discovery reported in the literature

Discovering enzyme functions requires more than inferring broad catalytic classes from protein sequences; it also requires identifying enzymes capable of performing specific biochemical transformations among numerous candidates with similar functions. To first evaluate the practical value of RIVER in this research setting, we used the Escherichia coli in vitro metabolomics study reported by Gruber et al. [9] in Communications Biology (2026) as a retrospective case study. Through experimental screening of 667 successfully purified metabolic enzymes from the ASKA collection [44], that study identified multiple previously undercharacterized catalytic activities and further validated selected reactions experimentally. We selected four experimentally supported enzyme–reaction associations: CobC-catalyzed dephosphorylation of FMN to riboflavin, NanK-catalyzed phosphorylation of mannose using ITP, DeoA-catalyzed uridine phosphorolysis, and RihB-catalyzed thymidine hydrolysis. Rather than comparing only a small number of homologous proteins, we ranked the same 667 candidate proteins for each reaction, yielding a total of 2,668 reaction–enzyme candidate associations (Fig. 2a–f). This setting directly tests whether the system can prioritize the experimentally confirmed extensions of substrate or reaction scope reported in the literature using existing biochemica knowledge and incomplete protein annotations.

The 667 candidate proteins span all seven primary EC classes, including 226 proteins in the transferase class and 121 each in the oxidoreductase and hydrolase classes; another 19 proteins have no recorded EC number. Counts across EC classes are not mutually exclusive: among the 648 proteins with EC annotations, 25 belong to more than one primary class (Fig. 2a). Candidate sequence lengths range from 76 to 1,320 amino acid residues, with a median of 336, and the proportion of hydrophobic residues also varies (Fig. 2b). Thus, the candidate pool includes both proteins with related functions identifiable from reaction classes and candidates requiring finer distinctions based on substrate structures, cosubstrate selection, and chemical bond changes.

We compared RIVER against direct LLM calls, combinations involving general-purpose coding agents, and existing specialized retrieval methods (Table 1). With Qwen3.6-35B-A3B [51] as the foundation model, RIVER ranked CobC, NanK, and RihB first and DeoA third, achieving a mean reciprocal rank (MRR) of 0.833 across the four cases. With GPT-5.6 Luna [52], the final ranks of the four target enzymes were 1, 1, 2, and 2, respectively, all within the top two candidates, with an MRR of 0.750 (Fig. 2c–f). The two configurations achieved 3/4 and 2/4 Top-1 hits, respectively, and both achieved 4/4 Top-3 hits. Although their rankings difered for individual substrates, both efectively narrowed the range of candidates meriting priority investigation.

a  
![](images/4dc675919b2c4b2c32222ced983124080093aaad94359cce93489dc678d6a11f.jpg)  
b

![](images/107fe21ccaa72c73cfac2183c909f8009696aadcd19fd26dda726fac833e0fc5.jpg)

c  
![](images/97193af35c745cae8b4e0c8f759e6bf3cfdeba7e1dd17fc6e539c598120e8a65.jpg)  
Fig. 1. RIVER: an evidence-guided multi-agent framework for bidirectional enzyme–reaction retrieval. a, Complementary retrieval settings. Given a query reaction, RIVER ranks candidate enzymes; given a query enzyme, it ranks candidate reactions. b, Two-stage workflow. During evidence-guided prescreening, the ReactionAnalysis and EnzymeAnnotate agents compile reaction-level and protein-level information with database support, while the Filter Agent narrows the candidate pool. During reflective reranking, the InitRank Agent produces an initial order, the Reconsider and Contradiction agents reassess supporting and conflicting evidence, and BioChem Expert provides computational evidence. The EvidenceCheck Agent reviews the resulting evidence, after which the FinalRank Agent generates the final ranking. Dashed connections indicate interactions involving computational expertise. c, Comparison with multi-agent systems. We constructed E. coli enzyme retrieval tasks from recent literature and compared RIVER with diferent categories of state-of-the-art multi-agent systems, obtaining a clear performance advantage.

Table 1. Retrieval performance ofRIVER, general-purpose coding agents, language models, and existing enzyme retrieval methods on four experimentally validated reactions. Each column corresponds to a query reaction and its literature reference enzyme. Values represent the rank of the reference enzyme in the candidate pool; lower is better. MRR is the mean reciprocal rank across the four queries; higher is better. For configurations with variation across three repeated runs, mean ± standard deviation is reported; the absence of a reported standard deviation indicates identical results across all three runs.
<table><tr><td>System</td><td>FMN/ CobC</td><td>Mannose- ITP/ NanK</td><td>Uridine / DeoA</td><td>Thymidine /RihB</td><td>MRR↑</td></tr><tr><td>RIVER (Qwen3.6-35B-A3B)</td><td>1</td><td>1</td><td>3</td><td>1</td><td>0.833</td></tr><tr><td>RIVER (GPT-5.6 Luna)</td><td>1</td><td>1</td><td>2</td><td>2</td><td>0.750</td></tr><tr><td>Phylo Max</td><td>1</td><td>2</td><td>3</td><td>1</td><td>0.708</td></tr><tr><td>GPT-5.6 Luna + Codex</td><td>1</td><td>5</td><td>2</td><td>1</td><td>0.675</td></tr><tr><td>Claude Haiku 5.5 + Claude Code</td><td> $1 . 7 \pm 1 . 2$ </td><td> $2 . 3 \pm 0 . 6$ </td><td> $2 . 7 \pm 0 . 6$ </td><td> $1 . 0 \pm 0 . 0$ </td><td> $0 . 6 5 3 \pm 0 . 1 0 5$ </td></tr><tr><td>GPT-5.6 Luna</td><td> $3 . 3 \pm 1 . 5$ </td><td> $2 . 7 \pm 1 . 2$ </td><td> $2 . 0 \pm 0 . 0$ </td><td> $1 . 7 \pm 0 . 6$ </td><td> $0 . 4 8 2 \pm 0 . 1 1 2$ </td></tr><tr><td>Claude Haiku 5.5</td><td> $5 . 3 \pm 1 . 2$ </td><td> $2 . 3 \pm 1 . 2$ </td><td> $2 . 3 \pm 0 . 6$ </td><td> $2 . 0 \pm 1 . 0$ </td><td> $0 . 4 5 1 \pm 0 . 1 0 3$ </td></tr><tr><td>Phylo Standard</td><td>14</td><td>3</td><td>2</td><td>3</td><td>0.310</td></tr><tr><td>Enz-GFM</td><td>22</td><td>105</td><td>339</td><td>20</td><td>0.027</td></tr></table>

In this comparison, both RIVER configurations achieved higher MRRs than all control settings listed in the table. Their advantage was evident not only over stand-alone language models but also over generalpurpose agents capable of planning, code execution, and external tool calls. Claude Haiku 5.5 [53] combined with Claude Code [54] achieved an MRR of $\mathbf { 0 . 6 5 3 \pm 0 . 1 0 5 } ,$ GPT-5.6 Luna combined with Codex [55] achieved 0.675, and Phylo Max [56] achieved 0.708; all were below both RIVER (GPT-5.6 Luna), at 0.750, and the Qwen-based version, at 0.833. Under the comparison using the same GPT-5.6 Luna foundation model, RIVER exceeded Luna + Codex by 0.075 MRR and the mean from repeated stand-alone Luna API experiments $( \mathbf { 0 . 4 8 2 \pm 0 . 1 1 2 } )$ by 0.268. Thus, for this retrieval task defined by specific catalytic reactions, the programming and tool-use capabilities of general-purpose agents do not necessarily translate into optimal enzyme–reaction matching; a dedicated workflow for biochemical evidence analysis, review, and ranking provides additional value.

Diferences from existing enzyme retrieval reference methods were similarly pronounced. Phylo Standard [56] achieved an MRR of 0.310 across the four tasks; the literature-based comparison result for Enz-GFM [20] was 0.027, with NanK and DeoA ranked 105th and 339th, respectively. For reactions that extend beyond conventional functional annotations to varying degrees, the retrieval challenge is not merely to identify a broad enzyme class but to assess whether a candidate may accept a noncanonical substrate or cosubstrate and perform the required chemical transformation at the target site.

## Diferent substrates and cosubstrates pose specificity challenges for candidate enzymes

The four experimental cases ofer complementary perspectives on the basis of RIVER’s rankings (Fig. 2c– f). For CobC, known activities related to adenosylcobalamin phosphate and α-ribazole phosphate, together with histidine phosphatase family information, provide an analogical basis for its potential dephosphorylation of FMN, another vitamin-derived phosphate ester. The key inference here is an extrapolation from shared phosphate-ester hydrolysis chemistry to a molecule with a substantially diferent substrate scafold, rather than a simple match to an identical substrate name. For NanK, the canonical function is the phosphorylation of N-acetylmannosamine. The system combines information about the C6 reaction position of hexoses with related sugar kinase evidence to support the possibility that NanK accepts mannose, although the retrieved records do not directly establish that ITP can replace the usual phosphate donor. The follow-up experimental validation in the original metabolomics study and the annotation evidence available during RIVER retrieval represent distinct levels of information and should be distinguished.

![](images/c595c7146bdc7af9b23aa94a767ffaf7ea7ce04b21e6dc2db4b3b2dc1caeb6be.jpg)

![](images/ea15813b7f226a8585e1220cb6c5620e06b257083b1ded56a90374c3452effde.jpg)

![](images/5f0e786688663dd31d7405070abad4aee9e2543e396cc9b4ad640506b71ceae3.jpg)  
Fig. 2. RIVER prioritizes experimentally validated enzyme–reaction associations among 667 E. coli proteins. a, Counts of the 667 candidate proteins by primary EC class; classes are not mutually exclusive, and 19 proteins lack recorded EC annotations. b, Candidate protein sequence lengths and proportions of hydrophobic residues (A, V, I, L, M, F, W, Y); colored points indicate the four literature reference enzymes. c–f, Reaction structures, reference protein structures, and top-five candidates for FMN dephosphorylation (CobC, P52086), mannose–ITP phosphorylation (NanK, P45425), uridine phosphorolysis (DeoA, P07650), and thymidine hydrolysis (RihB, P33022). Colored rows indicate the literature reference enzymes; other candidates are not presumed to lack catalytic activity. Protein structures are shown as post hoc visualization context and were not inputs to structure-pocket tools during ranking. Panels c–f correspond to the GPT-5.6 Luna configuration.

For DeoA, its pyrimidine nucleoside phosphorolysis chemistry is compatible with the phosphatedependent cleavage of the glycosidic bond and formation of ribose-1-phosphate in the uridine reaction. RIVER (GPT-5.6 Luna) ranked DeoA second, while the known uridine phosphorylase Udp ranked first. This outcome shows that the system retained the substrate-scope extension reported in the literature without excluding other enzymes already capable of catalyzing the reaction. For RihB, its pyrimidine nucleoside hydrolase function provides important evidence at the reaction-type level, but a substrate-specificity diference remains to be explained between previously documented ribonucleoside activity and the 2′-deoxyribose structure of the target substrate, thymidine. Together, these cases show that the practical value of the system lies both in placing the experimental reference enzymes within a high-priority range for validation and in preserving specific evidence gaps concerning substrates, cosubstrates, and structural selectivity.

## 2.2 Competitive performance of RIVER on public benchmarks

## Benchmark datasets

To systematically evaluate the generalization of RIVER in bidirectional enzyme–reaction retrieval and across diferent enzyme families, we used five public benchmark datasets provided by EnzymeCAGE [19]: two internal test sets constructed from the Rhea database (Enzyme-405 and Orphan-335) and three external test sets targeting specific enzyme families (P450, Terpene, and Phosphatase).

Enzyme-405 primarily evaluates functional prediction for unseen enzymes. It contains 405 enzymes and 295 biochemical reactions, forming 961 positive enzyme–reaction pairs and 14,960 negative pairs. Orphan-335 contains 335 reactions absent from the training data but subsequently assigned enzymatic annotations in later database versions, and evaluates the ability to retrieve known candidate enzymes for a target reaction. Both datasets were constructed from annotations added to Rhea between July 2023 and February 2025, with reaction-similarity filtering applied to reduce overlap with the training data.

The three external test sets further examine retrieval performance in enzyme families with diferent biochemical characteristics. The P450 dataset originates from the manually curated P450Rdb and covers oxidative reactions catalyzed by cytochrome P450 enzymes from multiple species, emphasizing diferences in substrate structures, oxidation sites, and regioselectivity. The Terpene dataset contains terpene synthases and their catalyzed reactions, involving cyclization and scafold rearrangement of diferent isoprenoid precursors, and tests the ability to distinguish similar substrates and diferent product scafolds. The Phosphatase dataset comes from a high-throughput enzyme activity screening study; the original data include 218 phosphatases and 165 reactions, totaling 35,970 enzyme–reaction combinations, and focus on hydrolytic specificity toward diferent substrates within the same family.

Together, these five datasets cover functional prediction for unseen enzymes, candidate enzyme retrieval for unseen reactions, and the identification of substrate and reaction specificity within diferent catalytic families. They provide complementary test settings for assessing RIVER’s bidirectional retrieval performance and biochemical generalization.

To test whether the advantages observed in the case studies extend to standardized retrieval tasks, we next examined both enzyme-to-reaction (E2R) and reaction-to-enzyme (R2E) retrieval and compared RIVER against baselines including MMseqs2, Selenzyme, ESP, CLIPzyme, and EnzymeCAGE (Fig. 3a–b). On Enzyme-405, RIVER ranked candidates for 402 evaluable queries out of 405, achieving Top-1, Top-3, Top-5, and Top-10 query success rates of 82.3%, 85.6%, 86.8%, and 88.3%, respectively. Its Top-1 success rate exceeded the 17.3% shown for EnzymeCAGE in the figure (Fig. 3a). On this task, MRR was 0.846, MAP was 0.831, nDCG@10 was 0.841, and the enrichment factor within the top 1% of candidates was 84.1. These results indicate that RIVER can not only rank candidate proteins for a given reaction but also prioritize the recovery of corresponding biochemical reactions from enzyme information.

On Orphan-335 reaction-to-enzyme retrieval, RIVER achieved Top-1, Top-3, Top-5, and Top-10 query success rates of 72.8%, 93.1%, 95.5%, and 97.9%, respectively (Fig. 3a). The corresponding EnzymeCAGE Top-1 value was 16.9%. As with Enzyme-405, the high Top-1 success rate indicates that the system can place known associations first among numerous reference candidates, while the further improvement at Top-10 shows that reference positives can still be recovered within relatively high ranks for queries without a first-place hit.

We further evaluated success rates within the top 1%, 3%, and 5% of candidate pools on three familyspecific benchmarks: P450, terpene synthases, and phosphatases (Fig. 3b). RIVER achieved corresponding success rates of 27.7%, 38.2%, and 41.9% on P450; 10.9%, 23.9%, and 28.3% on Terpene; and 75.5%, 75.5%, and 79.5% on Phosphatase. At the stringent top-1% threshold, these results represent gains of 19.8, 4.1, and 61.5 percentage points, respectively, over the strongest comparator shown for each dataset. The considerable diferences in absolute success rates and relative gains among the three families suggest that substrate classes, family annotation coverage, and candidate distributions afect the dificulty of reactionconditioned retrieval.

## 2.3 Evidence integration mechanisms in RIVER

RIVER’s performance depends not only on the final rankings, but also on whether prescreening retains the correct candidates and whether subsequent reasoning can extract discriminative reaction information. On P450, prescreening retained 51.1% of all candidate pairs while preserving 97.2% of labeled positive pairs; 96.9% of queries retained at least one positive. On Phosphatase, 72.3% of candidate pairs and 79.6% of positive pairs were retained, and every query retained at least one labeled positive. These results show that a lower candidate retention rate can coexist with high query-level positive coverage.

Efective online information retrieval is particularly important to the overall multi-agent system. Figure 3c compares the performance of a multi-agent system composed solely of LLMs with that of our multiagent system. To exclude possible online-access capabilities of closed-source models, we used the locally deployed GLM-5.2 as the foundation model for both the LLM and our multi-agent system in this experiment. The results show that retrieval performance decreases markedly without online data and knowledge as prerequisites. This demonstrates that the biochemical knowledge and reasoning capabilities inherent in an open-source LLM alone are insuficient for this task, and that knowledge augmentation using existing database resources is needed.

![](images/51024dd88bdba4c81b0162a2c6b7b3c5b8d1f1fe7fd0cba523c80b4b816f74f4.jpg)

![](images/7bef7e88ce48c7340307faab327ff32e5ee5c52fab7f89401feca456bc284fd2.jpg)

![](images/63f76e015ac03eb5d0310a3712f6f24fda160be2775dac02705a3f8527c75255.jpg)

![](images/3dc206d9a48d3e87174926d687c00982dd000d48224f0d0d941de0fea3f5251a.jpg)

![](images/a390e9c7cea1d7c1816ab54afce6dd160568512e4280f982444c59ca7ab86661.jpg)

![](images/cf10de526b2ec4ea6d0a8c4c4a9b21294d2cf5700e864d91d4e37148fbe43053.jpg)

![](images/76f6775450f4ad70fde6da16a2ba11e98ae1ad66aaf1e1fbe64e4f7fdd4054c0.jpg)

![](images/a5ea71dfd7c8f3877d54a3b1ade37a8483695d6dd071f233c75d8007791ab1ff.jpg)

![](images/2f0cafbc227a8b231a9467e2b27c8607e078bbace74a9049f6ed65bcbb7f43f5.jpg)

![](images/800411c739c11c8136b4fdf28d49a29c020c2d13f3308a955f9e46bbf66dd508.jpg)

![](images/5c9947b99c8aea868beff3de682651e2c37741c6a54dab92151bae8512675b2e.jpg)  
Fig. 3. RIVER’s bidirectional enzyme–reaction retrieval performance, module ablations, and biochemical evidence analysis. $\mathbf { a } ,$ E2R retrieval on Enzyme-405 and R2E retrieval on Orphan-335; the curves show the fraction of queries with at least one reference-positive candidate within the Top-k ranks. b, Query success rates for reference positives entering the top 1%, 3%, and 5% of candidate pools in the P450, Terpene, and Phosphatase datasets. $\mathbf { c } ,$ Comparison between the multi-agent system using a locally deployed LLM (GLM-5.2 as its base model) and ranking using GLM-5.2 alone without online search. ${ \bf d } ,$ Decreases in MRR and Top-1, Top-5, and Top-10 success rates relative to the full system following the removal of reaction reasoning, candidate summarization, reflection review, CPU experts, and family priors. Gray indicates that a module is not applicable; MRR diferences are plotted on a percentage scale. $\mathbf { e } ,$ Reaction-side requirements (upper row in each group) and coverage of supporting evidence among the top ten candidates (lower row) for ten biochemical attributes in each family. f, Changes in net discrimination between reference positives and hard negatives across biochemical evidence categories before and after reassessment; positive values indicate increased discrimination and negative values indicate decreased discrimination. ${ \bf g } ,$ EC annotation coverage for reference positives (open circles) and hard negatives (filled circles). h, Proportions of reference-positive candidates entering the top 1% of candidate pools, grouped by the number of supporting evidence categories.

To determine the relative contributions of diferent components, we separately removed reaction reasoning, candidate summarization, reflection review, and CPU biochemical computational experts, and examined changes in the final metrics relative to the full system (Fig. 3d). On Orphan-335, removing reaction reasoning reduced MRR by 0.068, while Top-1, Top-5, and Top-10 success rates decreased by 6.0, 8.4, and 7.8 percentage points, respectively. For orphan reactions lacking known direct catalytic representatives, functional names or sequence similarities alone often cannot reliably connect candidates to the target chem ical transformation. Active interpretation of reactants, products, and transformation types is therefore an important source of information for prioritizing candidates.

By contrast, the largest MRR loss on P450 occurred when candidate summarization was removed: MRR decreased by 0.0268, while Top-1 and Top-10 success rates declined by 3.7 and 4.7 percentage points, respectively (Fig. 3d). P450 candidates share much of their enzyme-family background, and generic oxidase labels cannot readily distinguish particular substrate scafolds and oxidation sites. Integrating candidate annotations, substrate ranges, reaction positions, and existing functional evidence is therefore particularly important. Reflection review and CPU experts also alter some rankings.

## 2.4 Relationships among biochemical evidence gaps, selectivity, and candidate ranking

We further investigated how diferent types of biochemical evidence contribute to candidate ranking by comparing the constraints proposed on the reaction side with supporting evidence among the top ten candidates (Fig. 3e). Coverage difered substantially across the ten categories of attributes. For example, approximately 97% of P450 reaction analyses involved stereo- or regioselectivity, whereas only approximately 8% of the top ten candidates had corresponding supporting information. This indicates that the attributes relevant to the reactions and the supporting evidence commonly available for candidates have diferent distributions. In conjunction with the reassessment analysis, evidence concerning stereo- or regioselectivity in P450 showed greater discrimination between positives and hard negatives after reassessment (Fig. 3f), indicating that even evidence with relatively limited coverage may have discriminative value. Meanwhile, hard negatives had higher EC annotation coverage than true-positive enzymes (Fig. 3g), showing that relatively extensive functional annotations may also occur for mismatched candidates. These findings support assessing both evidence coverage and its ability to distinguish candidates with respect to the target reaction, with particular attention to information on substrates, reaction sites, and stereoselectivity that can diferentiate similar candidates.

Further analysis of the net discriminatory ability of diferent attributes between reference positives and hard negatives before and after reassessment revealed nonuniform changes in attribute weighting (Fig. 3f). In P450, net discrimination associated with substrate specificity increased by approximately 0.335, whereas that associated with cofactor evidence decreased by approximately 0.335. This does not mean that cofactor information is unimportant for biochemical reactions. Rather, within the candidate set and comparison scheme considered, it was less capable of distinguishing known positives from hard negatives than the substratespecificity information emphasized after revision. Accordingly, a more valuable role of reflection may be to reconsider whether evidence genuinely distinguishes competing candidates, rather than merely accumulating additional seemingly relevant facts.

A comparison of EC annotation coverage further shows that more extensive annotation does not necessarily indicate a better match to the target reaction (Fig. 3g). In all three families, hard negatives had higher EC annotation coverage than reference positives; for P450, the corresponding proportions were 93.2% and 54.5%, respectively. When ranking candidates, the system must go beyond asking whether a protein has an

EC annotation to asking whether that annotation supports the target substrate, chemical bond changes, and reaction site. Correspondingly, the relationship between evidence breadth and retrieval success is family dependent (Fig. 3h). Among P450 reference positives with five categories of supporting evidence, 55.7% entered the top 1% of the candidate pool; among those with six or more evidence categories, the proportion increased to 72.7%. However, the proportions across evidence-count groups were not monotonically increasing for terpene synthases, and between-group diferences were smaller for phosphatases. This suggests that the specificity and discriminative power of evidence deserve greater attention than its simple count.

## 2.5 Six cases across reaction types reveal traceable evidence-guided reranking

Beyond aggregate statistics, we selected six cases in which reference-positive candidates improved in rank across three retrieval settings—Orphan-335, P450, and Phosphatase—to examine how the system reorganizes its assessment of evidence according to diferent substrates and reaction requirements (Fig. 4a–f). These cases involve acyl transfer, alkane hydroxylation, triterpene C28 oxidation, and sugar-phosphate hydrolysis. They therefore allow us to assess whether RIVER relies only on broad functional labels or can distinguish competing candidates at the level of specific chemical transformations.

For the two acyl-transfer reactions in Orphan-335, the central task of evidence reassessment was to distinguish general functional classes from specific acyl acceptors and modes of reaction (Fig. 4a,b). The initial assessment of MGAT1 had identified its monoacylglycerol acyltransferase-related function, but the sn-1/sn-2 position and substrate scope remained unclear. After further consideration of recorded substrateactivity annotations and acyl-CoA dependence, the reference enzyme moved from rank 4 to rank 1. For Q675A5, the initial annotation was more strongly associated with the hydrolase class; subsequent analysis identified its acyltransferase-related function and prioritized mechanistic compatibility with acyl transfe from phospholipids to alcohol acceptors, moving it from rank 3 to rank 1.

The two P450 cases further demonstrate that substrate scafolds and specific oxidation positions can provide discriminative evidence among enzymes of the same family (Fig. 4c,d). The initial information for alkane hydroxylase Q65A64 supported general alkane C–H hydroxylation but did not adequately establish the range of accepted branched alkanes. Reassessment strengthened the connection among reaction class, oxidase family, and EC annotation, moving it from rank 6 to rank 1. The analysis of CYP716A1 linked the triterpene scafold, the C28 position, and the carboxyl-containing product, raising its rank from 4 to 1.

In the phosphatase cases, ranking revisions relied more heavily on transferring evidence from related sugar-phosphate substrates and catalytic requirements (Fig. 4e,f). HxpB was recorded as having sugar- or polyol-phosphatase-related activity; when other sugar-phosphate reactions and divalent metal requirements were considered, it moved from rank 8 to rank 2. Based on hexose-phosphate-related activity and the compatibility of the phosphate position with the target reaction, YbiV moved from rank 7 to rank 2. Both cases show that, in the absence of direct functional annotations for an exact substrate, existing reaction information for related substrates can narrow the candidate range, while unvalidated aspects concerning the target deoxy sugar or amino sugar must remain explicit.

These cases complement the aggregate ablations and evidence profiles: ablation analyses measure the efect of each module on overall metrics, whereas the six specific reactions show which substrates, sites, and functional conflicts are documented in the reasoning traces. RIVER’s interpretability thus takes the form of inspectable evidence chains with clearly defined information boundaries.

![](images/ec6d26c0bb6891caef87bcf7242b3ce68c22ecdd55aea3b9bb09aeb1877f4678.jpg)  
Fig. 4. Evidence-guided reassessment improves the priority of reference-positive candidates across three enzyme retrieval datasets. $\mathbf { a } , \mathbf { b } ,$ MGAT1 (Q28C88; rank 4 to rank 1) and a lysosomal acyltransferase (Q675A5; rank 3 to rank 1) from Orphan-335. Subsequent assessment strengthened biochemical information concerning monoacylglycerol acceptor-related activity, acyl-CoA dependence, and acyl transfer from phospholipids to alcohol acceptors, respectively. $\mathbf { c } , \mathbf { d } ,$ An alkane hydroxylase (Q65A64; rank 6 to rank 1) and CYP716A1 (Q9LVY7; rank 4 to rank 1) from P450. Reranking emphasized compatibility between reaction class and P450/EC evidence in the former case and consistency among the triterpene scafold, C28 oxidation site, and carboxyl-containing product in the latter. Previously documented C28 oxidation activity does not, by itself, establish every oxidation step shown in panel d. $\mathbf { e , f , }$ The phosphatases HxpB (P77247; rank 8 to rank 2) and YbiV (P75792; rank 7 to rank 2). Activities involving related sugar-phosphate substrates, metal dependence, and compatibility of phosphate positions supported the ranking changes. Molecular structures were drawn from benchmark SMILES, with highlighted regions indicating the displayed chemical changes. The acyl donors and products in panels a and b are simplified; R denotes an unspecified substituent and does not indicate atom mapping. Pi denotes inorganic phosphate.

## 2.6 Summary of results

Starting from retrospective retrieval of experimentally validated reactions, RIVER prioritized all four target enzymes among 667 E. coli candidate proteins and achieved higher MRR than the compared generalpurpose coding-agent combinations and existing domain methods. Further results on public bidirectional retrieval benchmarks and enzyme-family retrieval indicate that this advantage is not restricted to a single class of enzymatic reaction. Module ablations show that reaction reasoning and candidate summarization play diferent important roles across tasks; evidence statistics and representative cases demonstrate that the system can examine compatibility among substrates, reaction sites, functional annotations, and pathway context while recording missing evidence and potential conflicts. Compared with outputs containing candidate rankings alone, this traceable organization of biochemical evidence makes RIVER better suited to proposing testable enzyme–reaction hypotheses and connecting computational prioritization with subsequent experimental work.

## 3 Methods

## 3.1 RIVER: an evidence-guided framework for bidirectional enzyme–reaction retrieval

RIVER (Reaction–enzyme Inference via Evidence-guided Retrieval) is a training-free multi-agent framework for prioritizing enzyme–reaction associations. It combines biochemical reasoning, retrieved annotations, and computational analyses in a two-stage workflow comprising evidence-guided prescreening and reflective reranking (Fig. 1). Here, training-free means that RIVER does not train or fine-tune model parameters on the evaluated retrieval datasets.

The functions of several RIVER agents are as follows. The InitRank Agent establishes the initial candidate order. The Reconsider Agent revisits supporting rationales and identifies evidence gaps or alternative biochemical explanations, while the Contradiction Agent checks for potentially inconsistent evidence. BioChem Expert provides computational evidence, including family, domain, and catalytic-site information where available, to support reassessment. The EvidenceCheck Agent reviews the consistency of the collected evidence and remaining gaps, and the FinalRank Agent generates the final ranking. Together, these components allow candidates to be reassessed as new support or conflicts emerge.

We formulate the two retrieval directions using a query q and the corresponding candidate pool $C ( q )$ :

$$
C ( q ) = \left\{ \begin{array} { l l } { E , } & { q = r \quad \mathrm { ( r e a c t i o n \mathrm { - } t o \mathrm { - } e n z y m e \ r e t r i e v a l ) } , } \\ { R , } & { q = e \quad \mathrm { ( e n z y m e \mathrm { - } t o \mathrm { - } r e a c t i o n \ r e t r i e v a l ) } . } \end{array} \right.\tag{1}
$$

Here, E denotes the specified set of candidate enzymes, and R denotes the specified set of candidate reactions. In reaction-to-enzyme retrieval, a reaction is evaluated against candidate proteins; in enzyme-toreaction retrieval, a protein is evaluated against candidate reactions. Both directions assess the biochemical compatibility of enzyme–reaction pairs, difering in which member of the pair is held fixed.

RIVER generates an ordered candidate list accompanied by candidate-specific evidence and unresolved questions. The order represents relative priority within the supplied candidate pool, rather than a calibrated probability of catalytic activity. Multiple candidates may be compatible with a query; candidates lacking positive benchmark labels are not assumed to be experimentally inactive.

## 3.2 Evidence-guided prescreening

Prescreening assembles reaction-level and protein-level information and reduces the candidate pool before detailed reassessment. For each enzyme–reaction pair $( e , r )$ , the framework constructs structured textual records:

$$
z _ { r } = F _ { \mathrm { r e a c t i o n } } ( r ) , \qquad z _ { e } = F _ { \mathrm { a n n o t a t i o n } } ( e ) .\tag{2}
$$

Here, $z _ { r }$ describes the queried transformation, while $z _ { e }$ summarizes available protein evidence. These records are textual descriptions rather than learned numerical embeddings.

ReactionAnalysis Agent. The ReactionAnalysis Agent interprets the supplied reaction description, including the identities and chemical structures of substrates and products where available. It summarizes the transformation type, substrate features, relevant bond changes, and potential cofactor or cosubstrate require ments. Mechanistic interpretations are treated as hypotheses that guide candidate assessment; requirements that cannot be determined from the available information are retained as unresolved questions.

EnzymeAnnotate Agent. The EnzymeAnnotate Agent organizes available protein information, including UniProtKB annotations, enzyme-family assignments, domain descriptions, and previously documented catalytic activities. Retrieved reaction records, including Rhea entries where available, provide additional chemical context. Candidate summaries preserve distinctions among documented activities, broader family level compatibility, and inferred substrate scope. Missing annotations are treated as evidence gaps rather than evidence that an enzyme cannot catalyze the reaction.

Filter Agent. The Filter Agent jointly evaluates reaction descriptions and candidate summaries to retain candidates for subsequent ranking:

$$
C _ { 1 } ( q ) = F _ { \mathrm { f i l t e r } } ( q , C ( q ) , \{ z _ { r } , z _ { e } \} ) , \qquad C _ { 1 } ( q ) \subseteq C ( q ) .\tag{3}
$$

Filtering considers biochemical compatibility, including reaction classes, substrate features, and catalytic requirements. Its aim is to reduce the number of candidates requiring detailed review while retaining plausible associations. Retention is a computational prioritization decision and does not establish catalytic activity; likewise, exclusion is not an experimentally validated negative assignment.

## 3.3 Evidence records and initial ranking

For each retained candidate $c \in C _ { 1 } ( q )$ , RIVER assembles a query-conditioned evidence dossier $D _ { q , c }$ . The dossier organizes available reaction descriptions, protein annotations, retrieved records, computational outputs, and agent assessments. It distinguishes supporting observations, biochemical interpretations, potential conflicts, and requests for further validation. Evidence availability may difer across candidates and experimental settings.

The InitRank Agent compares retained candidates using evidence available before reflective review and generates an initial order:

$$
\pi _ { q } ^ { ( 0 ) } = F _ { \mathrm { i n i t } } \Big ( q , C _ { 1 } ( q ) , \{ D _ { q , c } ^ { ( 0 ) } \} _ { c \in C _ { 1 } ( q ) } \Big ) .\tag{4}
$$

Here, $\pi _ { q } ^ { ( 0 ) }$ is a ranked list, and $D _ { q , c } ^ { ( 0 ) }$ denotes the evidence available at initial ranking. Candidates are assessed jointly in the context of the query, rather than receiving independent, calibrated probabilities of activity.

## 3.4 Reflective review and computational evidence

Reflective review examines whether the initial ordering is supported by reaction-specific evidence and whether alternative interpretations or unresolved conflicts warrant reassessment. Review may change how existing evidence is interpreted; a revised rationale does not necessarily imply that new external evidence has been retrieved.

Reconsider Agent. The Reconsider Agent reexamines the initial candidate ordering and supporting rationales. It considers missing support, alternative biochemical interpretations, and uncertainties in substrate or product specificity. In particular, it focuses on distinguishing broad reaction-class compatibility from ev idence for the specific transformation under consideration. For example, documented turnover of the query substrate is distinguished from preference for another substrate, and formation of an intermediate is distinguished from subsequent conversion.

Contradiction Agent. The Contradiction Agent checks evidence that may challenge a candidate assignment, including incompatible substrate scope, alternative reaction chemistry, cofactor requirements, and inconsistencies among retrieved descriptions. Explicit conflicting observations are distinguished from missing information. In particular, the absence of a documented activity or catalytic feature is not automatically interpreted as a contradiction.

BioChem Expert. When tools and suitable inputs are available, the BioChem Expert module supplements language-based assessments with computational evidence. Such evidence may include sequence-family and domain assignments from HMMER/Pfam [40, 41] and mechanism or catalytic-site information from M-CSA [43]. These results provide additional constraints on enzyme–reaction compatibility, but family membership or an isolated computational match does not establish activity toward the queried substrate.

Tool-derived support depends on the analyses actually executed and the outputs returned in each experiment. Missing, unavailable, or failed tool outputs do not provide positive biochemical support. Protein structures displayed solely for post hoc visualization are not treated as evidence used during ranking.

EvidenceCheck Agent. The EvidenceCheck Agent reviews the consistency, specificity, and remaining uncertainties of the assembled records. It checks whether rationales are supported by documented evidence, whether apparent conflicts remain unresolved, and whether claims about the exact queried activity extend beyond the available annotations or computational results. This review evaluates the internal support for candidate prioritization; it does not independently validate catalytic activity.

## 3.5 Final ranking and traceable outputs

The FinalRank Agent compares candidates using evidence available after review and produces the final ordering:

$$
\pi _ { q } ^ { \mathrm { ( f i n a l ) } } = F _ { \mathrm { f i n a l } } \Big ( q , C _ { 1 } ( q ) , \{ D _ { q , c } ^ { \mathrm { ( r e v i e w e d ) } } \} _ { c \in C _ { 1 } ( q ) } \Big ) .\tag{5}
$$

The agent integrates reaction-specific support, computational observations, and unresolved conflicts through evidence-conditioned reasoning, without imposing a fixed weighted sum across evidence categories.

For each reported candidate, the output includes its rank and corresponding evidence summary:

$$
\begin{array} { r } { O _ { q } = \left[ \left( c , \mathrm { r a n k } _ { \pi _ { q } ^ { \mathrm { ( f i n a l ) } } } ( c ) , D _ { q , c } ^ { \mathrm { ( r e v i e w e d ) } } \right) \right] _ { c \in \pi _ { q } ^ { \mathrm { ( f i n a l ) } } } . } \end{array}\tag{6}
$$

Candidate summaries describe supporting rationales, evidence sources, and potential conflicts or evidence gaps. The recorded traces make the basis for prioritization inspectable while preserving distinctions among retrieved observations, computational predictions, and language-model interpretations.

The same evidence-guided formulation applies to both retrieval directions: reaction-to-enzyme retrieval prioritizes candidate catalysts for a specified transformation, while enzyme-to-reaction retrieval prioritizes possible reactions for a specified protein. The resulting rankings support experimental candidate selection and review of functional annotations, but do not, by themselves, establish catalytic activity or physiological relevance.

## 4 Discussion

RIVER addresses a practical problem in biochemical discovery: determining which enzyme–reaction hypotheses merit experimental attention when annotations are incomplete and evidence is distributed across diferent resources. Its central contribution is not merely the use of multiple agents, but the organization of heterogeneous evidence acquisition, biochemical interpretation, review, and candidate ranking into a bidirectional workflow.

The next step is prospective biochemical validation. RIVER can help prioritize candidates for orphan reactions, targeted promiscuity assays, analysis of missing steps in metabolic reconstruction, and enzyme engineering projects. Follow-up experiments should measure substrate specificity, cofactor acceptance, product identity, catalytic rates, and appropriate negative controls; a high rank should not be interpreted as evidence of physiological relevance. Complete query-level outputs, frozen database snapshots, and transparent evaluation inventories would further improve reproducibility [38, 50]. By presenting candidate rankings alongside traceable evidence and unresolved questions, RIVER provides a framework for connecting computational enzyme retrieval to hypothesis-driven experimental research.

## References

[1] A. D. Hanson, A. Pribat, J. C. Waller, and V. de Crécy-Lagard. ““Unknown” proteins and “orphan” enzymes: the missing half of the engineering parts list—and how to find it”. In: Biochemical Journal 425 (2010), pp. 1–11. DOI: 10.1042/BJ20091328.

[2] J. A. Gerlt et al. “The Enzyme Function Initiative”. In: Biochemistry 50 (2011), pp. 9950–9962. DOI: 10.1021/bi201312u.

[3] O. Khersonsky and D. S. Tawfik. “Enzyme promiscuity: a mechanistic and evolutionary perspective”. In: Annual Review of Biochemistry 79 (2010), pp. 471–505. DOI: 10.1146/annurev- biochem-030409-143718.

[4] R. A. Notebaart et al. “Network-level architecture and the evolutionary potential of underground metabolism”. In: Proceedings of the National Academy of Sciences 111 (2014), pp. 11762–11767. DOI: 10.1073/pnas.1406102111.

[5] G. I. Guzmán et al. “Enzyme promiscuity shapes adaptation to novel growth substrates”. In: Molecular Systems Biology 15 (2019), e8462. DOI: 10.15252/msb.20188462.

[6] U. T. Bornscheuer et al. “Engineering the third wave of biocatalysis”. In: Nature 485 (2012), pp. 185– 194. DOI: 10.1038/nature11117.

[7] F. H. Arnold. “Directed evolution: bringing new chemistry to life”. In: Angewandte Chemie International Edition 57 (2018), pp. 4143–4148. DOI: 10.1002/anie.201708408.

[8] D. C. Sévin, T. Fuhrer, N. Zamboni, and U. Sauer. “Nontargeted in vitro metabolomics for highthroughput identification of novel enzymes in Escherichia coli”. In: Nature Methods 14 (2017), pp. 187–194. DOI: 10.1038/nmeth.4103.

[9] C. H. Gruber, D. C. Sévin, E. Noor, V. R. T. Zanotelli, J. Schmitz, et al. “Systematic discovery of enzyme promiscuity in Escherichia coli using in vitro metabolomics”. In: Communications Biology 9 (2026), p. 1004. DOI: 10.1038/s42003-026-10099-x.

[10] M. Steinegger and J. Söding. “MMseqs2 enables sensitive protein sequence searching for the analysis of massive data sets”. In: Nature Biotechnology 35 (2017), pp. 1026–1028. DOI: 10.1038/nbt.3988.

[11] P. Carbonell et al. “Selenzyme: enzyme selection tool for pathway design”. In: Bioinformatics 34 (2018), pp. 2153–2154. DOI: 10.1093/bioinformatics/bty065.

[12] J. Y. Ryu, H. U. Kim, and S. Y. Lee. “Deep learning enables high-quality and high-throughput prediction of enzyme commission numbers”. In: Proceedings of the National Academy of Sciences 116 (2019), pp. 13996–14001. DOI: 10.1073/pnas.1821905116.

[13] T. Yu et al. “Enzyme function prediction using contrastive learning”. In: Science 379 (2023), pp. 1358– 1363. DOI: 10.1126/science.adf2465.

[14] V. Gligorijević et al. “Structure-based protein function prediction using graph convolutional networks”. In: Nature Communications 12 (2021), p. 3168. DOI: 10.1038/s41467-021-23303-9.

[15] Y. Song et al. “Accurate prediction of enzyme function through geometric graph learning on ESMFoldpredicted structures”. In: Nature Communications 15 (2024), p. 8180. DOI: 10.1038/s41467-024- 52533-w.

[16] A. Kroll, S. Ranjan, M. K. M. Engqvist, and M. J. Lercher. “A general model to predict small molecule substrates of enzymes based on machine and deep learning”. In: Nature Communications 14 (2023), p. 2787. DOI: 10.1038/s41467-023-38347-2.

[17] P. Mikhael, I. Chinn, and R. Barzilay. “CLIPZyme: reaction-conditioned virtual screening of enzymes”. In: Proceedings of Machine Learning Research. Vol. 235. 2024, pp. 35647–35663.

[18] E. Heid, D. Probst, W. H. Green, and G. K. H. Madsen. “EnzymeMap: curation, validation and datadriven prediction of enzymatic reactions”. In: Chemical Science 14 (2023), pp. 14229–14242. DOI: 10.1039/D3SC02048G.

[19] Y. Liu et al. “A geometric foundation model for enzyme retrieval with evolutionary insights”. In: Nature Catalysis 9 (2026), pp. 148–160. DOI: 10.1038/s41929-026-01478-y.

[20] C. Wang et al. “An enzyme-specific protein language model for catalytic property prediction”. In: Nature Communications 17 (2026), p. 8760. DOI: 10.1038/s41467-026-75283-3.

[21] Z. Shi, J. Zhu, D. Wang, B. Chen, Q. Yuan, Z. Mao, F. Wei, W. Wu, X. Liao, and H. Ma. RXNRECer Enables Fine-grained Enzymatic Function Annotation through Active Learning and Protein Language Models. arXiv preprint, posted 13 March 2026. 2026. DOI: 10.48550/arXiv.2603.12694. arXiv: 2603.12694 [cs.LG].

[22] C. Hua et al. “ReactZyme: a benchmark for enzyme–reaction prediction”. In: Advances in Neural Information Processing Systems. 2024. DOI: 10.48550/arXiv.2408.13659.

[23] The UniProt Consortium. “UniProt: the Universal Protein Knowledgebase in 2025”. In: Nucleic Acids Research 53 (2025), pp. D609–D617. DOI: 10.1093/nar/gkae1010.

[24] P. Bansal et al. “Rhea, the reaction knowledgebase in 2022”. In: Nucleic Acids Research 50 (2022), pp. D693–D700. DOI: 10.1093/nar/gkab1016.

[25] J. Hauenstein et al. “BRENDA in 2026: a global core biodata resource for functional enzyme and metabolic data at DSMZ Digital Diversity”. In: Nucleic Acids Research 54 (2026), pp. D527–D534. DOI: 10.1093/nar/gkaf1113.

[26] A. Rives et al. “Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences”. In: Proceedings of the National Academy of Sciences 118 (2021), e2016239118. DOI: 10.1073/pnas.2016239118.

[27] Z. Lin et al. “Evolutionary-scale prediction of atomic-level protein structure with a language model”. In: Science 379 (2023), pp. 1123–1130. DOI: 10.1126/science.ade2574.

[28] J. Jumper et al. “Highly accurate protein structure prediction with AlphaFold”. In: Nature 596 (2021), pp. 583–589. DOI: 10.1038/s41586-021-03819-2.

[29] M. Varadi et al. “AlphaFold Protein Structure Database in 2024: providing structure coverage for over 214 million protein sequences”. In: Nucleic Acids Research 52 (2024), pp. D368–D375. DOI: 10.1093/nar/gkad1011.

[30] M. van Kempen et al. “Fast and accurate protein structure search with Foldseek”. In: Nature Biotechnology 42 (2024), pp. 243–246. DOI: 10.1038/s41587-023-01773-0.

[31] P. Lewis et al. “Retrieval-augmented generation for knowledge-intensive NLP tasks”. In: Advances in Neural Information Processing Systems. Vol. 33. 2020, pp. 9459–9474. DOI: 10.48550/arXiv. 2005.11401.

[32] S. Yao, J. Zhao, D. Yu, N. Du, I. Shafran, K. R. Narasimhan, and Y. Cao. “ReAct: Synergizing Reasoning and Acting in Language Models”. In: International Conference on Learning Representations (ICLR). OpenReview.net, 2023. DOI: 10.48550/arXiv.2210.03629. arXiv: 2210.03629.

[33] A. M. Bran et al. “Augmenting large language models with chemistry tools”. In: Nature Machine Intelligence 6 (2024), pp. 525–535. DOI: 10.1038/s42256-024-00832-8.

[34] D. A. Boiko, R. MacKnight, B. Kline, and G. Gomes. “Autonomous chemical research with large language models”. In: Nature 624 (2023), pp. 570–578. DOI: 10.1038/s41586-023-06792-0.

[35] K. Swanson, W. Wu, N. L. Bulaong, J. E. Pak, and J. Zou. “The AI scientist virtual lab designs novel SARS-CoV-2 nanobodies”. In: Nature 646 (2025), pp. 716–723. DOI: 10.1038/s41586-025-09442- 9.

[36] A. E. Ghareeb et al. “A multi-agent system for automating scientific discovery”. In: Nature 655 (2026), pp. 497–505. DOI: 10.1038/s41586-026-10652-y.

[37] J. Gottweis et al. “Accelerating scientific discovery with Co-Scientist”. In: Nature 655 (2026), pp. 487– 496. DOI: 10.1038/s41586-026-10644-y.

[38] S. Kapoor and A. Narayanan. “Leakage and the reproducibility crisis in machine-learning-based science”. In: Patterns 4 (2023), p. 100804. DOI: 10.1016/j.patter.2023.100804.

[39] N. Shinn, F. Cassano, A. Gopinath, K. Narasimhan, and S. Yao. “Reflexion: language agents with verbal reinforcement learning”. In: Advances in Neural Information Processing Systems. Vol. 36. 2023, pp. 8634–8652. DOI: 10.48550/arXiv.2303.11366.

[40] S. R. Eddy. “Accelerated profile HMM searches”. In: PLoS Computational Biology 7 (2011), e1002195. DOI: 10.1371/journal.pcbi.1002195.

[41] J. Mistry et al. “Pfam: the protein families database in 2021”. In: Nucleic Acids Research 49 (2021), pp. D412–D419. DOI: 10.1093/nar/gkaa913.

[42] M. Blum et al. “InterPro: the protein sequence classification resource in 2025”. In: Nucleic Acids Research 53 (2025), pp. D444–D456. DOI: 10.1093/nar/gkae1082.

[43] A. J. M. Ribeiro et al. “Mechanism and Catalytic Site Atlas (M-CSA): a database of enzyme reaction mechanisms and active sites”. In: Nucleic Acids Research 46 (2018), pp. D618–D623. DOI: 10.1093/ nar/gkx1012.

[44] M. Kitagawa et al. “Complete set of ORF clones of Escherichia coli ASKA library (a complete archive of E. coli K-12 ORFs): unique resources for biological research”. In: DNA Research 12 (2005), pp. 291– 299. DOI: 10.1093/dnares/dsi012.

[45] C. L. Zayas and J. C. Escalante-Semerena. “Reassessment of the late steps of coenzyme B12 synthesis in Salmonella enterica: evidence that dephosphorylation of adenosylcobalamin-5′-phosphate by the CobC phosphatase is the last step of the pathway”. In: Journal ofBacteriology 189 (2007), pp. 2210– 2218. DOI: 10.1128/JB.01665-06.

[46] J. M. Monk et al. “iML1515, a knowledgebase that computes Escherichia coli traits”. In: Nature Biotechnology 35 (2017), pp. 904–908. DOI: 10.1038/nbt.3956.

[47] L. Heirendt et al. “Creation and analysis of biochemical constraint-based models using the COBRA Toolbox v.3.0”. In: Nature Protocols 14 (2019), pp. 639–702. DOI: 10.1038/s41596-018-0098-2.

[48] F. Li et al. “Deep learning-based k prediction improves enzyme-constrained model reconstruction”. In: Nature Catalysis 5 (2022), pp. 662–672. DOI: 10.1038/s41929-022-00798-z.

[49] H. Yu, H. Deng, J. He, J. D. Keasling, and X. Luo. “UniKP: a unified framework for predicting enzyme kinetic parameters”. In: Nature Communications 14 (2023), p. 8211. DOI: 10.1038/s41467-023- 44113-1.

[50] M. D. Wilkinson et al. “The FAIR guiding principles for scientific data management and stewardship”. In: Scientific Data 3 (2016), p. 160018. DOI: 10.1038/sdata.2016.18.

[51] Qwen Team. Qwen3.6-35B-A3B: Agentic Coding Power, Now Open to All. Apr. 2026. URL: https: //qwen.ai/blog?id=qwen3.6-35b-a3b (visited on 11/05/2026).

[52] OpenAI. GPT-5.6: Frontier intelligence that scales with your ambition. July 2026. URL: https:// openai.com/index/gpt-5-6/ (visited on 11/05/2026).

[53] Anthropic. Claude Haiku 5.5. Oct. 2026. URL: https://www.anthropic.com/claude-haiku-5- 5 (visited on 11/05/2026).

[54] Anthropic. Claude Code: Agentic coding tool. 2026. URL: https://github.com/anthropics/ claude-code (visited on 11/05/2026).

[55] OpenAI. Introducing Codex. May 2025. URL: https : / / openai . com / index / introducing - codex/ (visited on 11/05/2026).

[56] Phylo, Inc. Phylo: AI agents for biomedical research. 2026. URL: https://phylo.bio/ (visited on 11/05/2026).