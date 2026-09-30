# IS MANUAL SOFTWARE OPTIMIZATION A THING OF THE PAST?

Pavlin G. Policarˇ <sup>∗</sup>

Martin Špendl

Tomaž Hocevarˇ

Faculty of Computer and Information Science University of Ljubljana Vecna pot 113, 1000 Ljubljana, Sloveniaˇ

## ABSTRACT

Context: Scientific software is increasingly required to process larger datasets while maintaining acceptable execution times. Software optimization traditionally requires substantial expertise in programming, algorithms, and numerical methods. Recent advances in large language models (LLMs) offer the possibility of automating much of this process.

Objectives: We investigate whether LLM-based agents can autonomously achieve substantial performance improvements in scientific software, including mature implementations that have already been extensively optimized by human developers.

Methods: We tasked an LLM-based agent with optimizing software for three computational problems: t-SNE, single-sample gene set enrichment analysis (ssGSEA), and graphlet counting. Humans defined the scope, correctness criteria, and a verification mechanism, after which the agent worked autonomously, in some cases for several hours. Code maintainers reviewed each resulting implementation and verified its correctness.

Results: The optimized implementations were faster in all tested configurations, by up to two orders of magnitude over the fastest existing tools. The improvements included low-level code optimizations, mathematical reformulations, and an entirely new algorithm for graphlet counting.

Conclusion: Software optimization can increasingly be delegated to autonomous agents, with the human role shifting from implementing optimizations to deciding which software to optimize, defining objectives, providing verification mechanisms, and ensuring the correctness of the final software. For well-scoped, verifiable problems, we argue that manual software optimization may be a thing of the past.

Keywords Software optimization, Agentic optimization, Algorithmic complexity, Large language models .

## 1 Introduction

Scientific research increasingly relies on software for data analysis, simulation, and other computational tasks [2]. As datasets grow, the efficiency of this software increasingly determines which analyses are feasible. Yet optimizing non-trivial scientific software requires expertise in software engineering, numerical computing, and algorithm design, as well as considerable development effort.

Recent advances in large language models (LLMs) have enabled agents that can generate, modify, test, and benchmark software. Performance can be improved at different levels, from low-level code optimizations, such as better memory access patterns, vectorization, or parallelization, to reformulating the computation mathematically or replacing the algorithm altogether. Whether LLM agents can autonomously find improvements beyond the low-level ones, even for well-scoped problems with verifiable results, remains unclear, as existing evaluations report mostly surface-level gains [9].

We investigate this question on three computational problems: t-SNE, single-sample gene set enrichment analysis (ssGSEA), and graphlet counting. In each case, humans defined the scope, correctness criteria, and a verification mechanism, after which an LLM agent worked without further intervention. The resulting implementations were faster in every case study, by up to two orders of magnitude over the fastest existing tools, with improvements spanning all three levels.

![](images/cf6f21358c6639a6732cd8a72510005baf16f516c83a728141a457180462fdce.jpg)

![](images/f903c823d80240f7385cdc8a21f503e4b582f84af76253e62667178534fee23f.jpg)

![](images/a82e1f0020dff144ac10a799b03bc74e6ccd3fd2235095db311eb5feb66d9ef2.jpg)  
Figure 1: Runtime of existing and agent-optimized implementations on a consumer-grade M1 MacBook Pro. (a) t-SNE optimization phase (8 threads) on subsamples of a single-cell data set. (b) ssGSEA implementations on 1,000 samples with increasing gene sets. (c) Graphlet counting implementations on random (Erdos–Rényi) graphs with 10k nodes and˝ an increasing number of edges.

## 2 Related Work

Early LLM-based optimizers wrapped the model in fixed pipelines. SysLLMatic [7] profiles a program to locate hotspots and rewrites them using a curated catalog of optimization patterns, while PIE [12] fine-tunes models on pairs of slower and faster C++ programs. Evolutionary systems that pair LLMs with automated evaluators have gone further, discovering new algorithms [6]. More recently, general-purpose coding agents have been applied to real scientific software, with the largest gains coming from rewriting code in faster languages or for GPUs [4].

Nevertheless, the effectiveness of general-purpose LLM-based agents remains uncertain. On AlgoTune [9], current models achieved only modest speedups over mature numerical libraries and failed to discover algorithmic improvements. On benchmarks built from real optimization commits in scientific Python libraries, agents reach less than a quarter of the expert speedup [5], and other evaluations report negligible gains [11]. In contrast, we show that a general-purpose agent can reduce the expected asymptotic cost of a computation, outperform an expert-designed algorithm, and speed up mature, hand-optimized software several-fold in place.

## 3 Case studies

To demonstrate that agentic software optimization extends beyond low-level code changes, we tasked Claude Opus 5, running in Claude Code, with either improving an existing implementation, producing a new, faster implementation of an existing algorithm, or producing a novel algorithm altogether for three real-world computational problems. Each implementation was reviewed by domain experts, including the original developers of openTSNE and Orca, and verified to be correct.

## 3.1 openTSNE

t-distributed stochastic neighbor embedding (t-SNE) [13] is widely used for visualizing high-dimensional data. Because its exact computation scales quadratically with the number of data points, practical applications rely on approximations. The two most widely used approximations are Barnes–Hut, which reduces the asymptotic complexity to O(N log N), and FIt-SNE, which further reduces it to O(N).

openTSNE [8] implements both approximations and is a mature, highly optimized CPU implementation. We tasked an agent to optimize its optimization phase, which typically accounts for 80–90% of its total runtime, while preserving the public API, parameter defaults, optimization procedure, and accuracy of both approximations. It ran the project’s 174 unit tests after each modification and retained only changes that improved the measured runtime.

After an overnight run, the optimized implementation was faster in every tested configuration, with speedups of 4.4–7.7× at 500,000 points. Single-threaded runtime decreased from 62 to 8 minutes for Barnes–Hut and from 9 to 2 minutes for FIt-SNE. On eight threads, the corresponding runtimes decreased from 12 to under 2 minutes and from 140 to 28 seconds (Figure 1.a).

The agent implemented and evaluated approximately 60 optimization ideas, retaining 34. The improvements included low-level optimizations such as specialized code paths, improved memory layout, and additional parallelism. The agent also derived a mathematical simplification of one stage of the FIt-SNE approximation, reducing the runtime of that stage by about a quarter. None changed the overall asymptotic complexity. This demonstrates that substantial gains can still be obtained through constant-factor optimization of mature software implementations.

## 3.2 Single-sample GSEA

Gene set enrichment analysis (GSEA) tests whether predefined groups of genes, such as biological pathways, are concentrated at the top or bottom of a list of genes ranked by differential expression. Single-sample GSEA (ssGSEA) [1] applies this idea to individual samples, scoring every gene set in every sample and thereby converting a gene expression matrix into a matrix of gene set activities. ssGSEA has become a standard step in gene expression analysis, routinely used to estimate cell-type composition and to stratify samples into molecular subtypes.

Existing implementations focus on accelerating the scoring of individual sample–gene set pairs. GSEApy implements a fast kernel but still requires a pass over all ranked genes of the sample. GSVA uses a closed-form expression that needs only the ranks of the set’s own genes, reducing asymptotic complexity in theory, but not realizing this gain in its implementation. We tasked an agent with producing a new implementation of ssGSEA, requiring its scores to exactly match those of existing libraries.

Building on GSVA’s closed form, the agent restructured the computation as a sparse matrix multiplication and eliminated inefficiencies, realizing this asymptotic gain. For 2,000 gene sets scored across 1,000 samples, the runtime decreased from 5 minutes with GSEApy, the fastest existing implementation, to 2.2 seconds, a speedup of 136×, with the gap widening with increasing numbers of gene sets (Figure 1.b). This makes it practical to score entire gene set collections rather than a preselected subset of pathways, removing a trade-off that analysts routinely make for computational reasons.

## 3.3 Graphlet counting

Graphlets [10] are small connected graphs that occur as induced subgraphs of a network and can characterize its local topology. Counting the different roles that nodes can occupy within graphlets, known as orbits, provides a finer-grained description of node topology. Graphlet and orbit counts have been widely used in network analysis, including for node classification, anomaly detection, and recommendation. Because the number of subgraphs around a node grows rapidly with its degree, brute-force enumeration quickly becomes infeasible for large or dense networks.

Orca [3], a widely used orbit-counting method, avoids brute-force enumeration by deriving the counts of larger graphlets from those of smaller ones through combinatorial relations. For graphlets with up to five nodes, this reduces the expected time complexity from $\mathcal { O } \big ( n d ^ { 4 } \big )$ to $\mathcal { O } ( n d ^ { 3 } )$ , where n is the number of nodes and d the maximum node degree. We tasked an agent with producing a new implementation with improved scaling of orbit counting, requiring its counts to match those of Orca and brute-force enumeration exactly

After running autonomously for several hours, the agent produced an entirely new algorithm that was faster than Orca on every tested network. On a human protein–protein interaction network from $\mathbf { B i o S N A P } _ { \mathrm { , } } ^ { 2 }$ with approximately 22k nodes and 340k edges, the new implementation counted all orbits in 13 seconds, compared with 35 minutes for Orca, a speedup of about 160×. Brute-force enumeration would require an estimated 3.5 days, over 20,000 times longer than the new implementation. On random graphs with 10k nodes, the gap widened with network density, and the runtime grew roughly quartically with the number of edges for brute-force enumeration, cubically for Orca, and quadratically for the agent’s implementation (Figure 1.c).

## 4 Conclusion

Our three case studies demonstrate that autonomous LLM-based agents can substantially speed up scientific software, starting from mature and naive implementations alike. The improvements were not limited to surface-level optimizations; agents applied low-level code optimizations, derived mathematical reformulations, and developed an entirely new algorithm for graphlet counting. The resulting implementations were faster in every case study, in some cases by orders of magnitude.

These findings suggest that software optimization can increasingly be delegated to autonomous agents. Improvements of this kind have typically required specialists in numerical computing or algorithm design, whom most scientific projects cannot afford. Autonomous agents put them within reach of individual researchers and small research groups. The human role is therefore shifting from implementing optimizations to deciding which software to optimize, defining scope and objectives, providing verification mechanisms, and ensuring that the final software is correct. Although problems with open-ended objectives or hard-to-verify correctness remain untested, we argue that for well-scoped, verifiable problems, manual software optimization may indeed be a thing of the past.

## Declaration of competing interest

The authors declare no competing interests.

## Reproducibility

We provide reusable optimization prompts, the new implementations, and replication materials at https://github.   
com/pavlin-policar/llm-software-optimization.

## Funding

This work was supported by the research programme P2-0209 of the Slovenian Research and Innovation Agency and the Project Grant L7-70273, and Young Research Grant 57111.

## References

[1] D. A. Barbie, P. Tamayo, J. S. Boehm, S. Y. Kim, S. E. Moody, I. F. Dunn, A. C. Schinzel, P. Sandy, E. Meylan, C. Scholl, et al. Systematic RNA interference reveals that oncogenic KRAS-driven cancers require TBK1. Nature, 462(7269):108–112, 2009.

[2] D. Ezer and K. Whitaker. Point of view: Data science for the scientific life cycle. eLife, 8:e43979, mar 2019.

[3] T. Hocevar and J. Demšar. A combinatorial approach to graphlet counting.ˇ Bioinformatics, 30(4):559–565, 2014.

[4] J. Li, A. Rubinsteyn, S. Feldman, T. O’Donnell, J. M. Ferguson, R. Patro, I. Driver, P. A. Ewels, F. Krueger, P. Angerer, I. Gold, J. Manning, L. Heumos, M. Ahangari, V. Goyal, H. Masoudi, B. Pedersen, A. Bai, H. Li, S. Shringarpure, and A. Ho. Scientific computing in the age of agentic AI: an exploratory field report. bioRxiv, 2026.

[5] J. J. Ma, M. Hashemi, A. Yazdanbakhsh, K. Swersky, O. Press, E. Li, V. J. Reddi, and P. Ranganathan. SWEfficiency: Can language models optimize real-world repositories on real workloads? In Proceedings ofthe 43rd International Conference on Machine Learning, 2026.

[6] A. Novikov, N. Vu, M. Eisenberger, E. Dupont, P.-S. Huang, A. Z. Wagner, S. Shirobokov, B. Kozlovskii, F. J. R.˜ Ruiz, A. Mehrabian, M. P. Kumar, A. See, S. Chaudhuri, G. Holland, A. Davies, S. Nowozin, P. Kohli, and M. Balog. AlphaEvolve: A coding agent for scientific and algorithmic discovery, 2025.

[7] H. Peng, A. Gupte, R. Hasler, N. J. Eliopoulos, C.-C. Ho, R. Mantri, L. Deng, K. Läufer, G. K. Thiruvathukal, and J. C. Davis. SysLLMatic: Large language models are software system optimizers. Journal of Systems and Software, 240:112929, 2026.

[8] P. G. Policar, M. Stražar, and B. Zupan. openTSNE: A modular python library for t-SNE dimensionality reductionˇ and embedding. Journal ofStatistical Software, 109(3):1–30, 2024.

[9] O. Press, B. Amos, H. Zhao, Y. Wu, S. Ainsworth, D. Krupke, P. Kidger, T. Sajed, B. Stellato, J. Park, N. Bosch, E. Meril, A. Steppi, A. Zharmagambetov, F. Zhang, D. Pérez-Piñeiro, A. Mercurio, N. Zhan, T. Abramovich,

K. Lieret, H. Zhang, S. Huang, M. Bethge, and O. Press. AlgoTune: Can language models speed up generalpurpose numerical programs? In D. Belgrave, C. Zhang, H. Lin, R. Pascanu, P. Koniusz, M. Ghassemi, and N. Chen, editors, Advances in Neural Information Processing Systems, volume 38, Main Conference. Curran Associates, Inc., 2025.

[10] N. Pržulj, D. G. Corneil, and I. Jurisica. Modeling interactome: scale-free or geometric? Bioinformatics, 20(18):3508–3515, 2004.

[11] E. Sarıkayak, W. Gu, H. Ghonim, and C. Chen. Evaluating LLMs on real-world software performance optimization. arXiv preprint arXiv:2606.25530, 2026.

[12] A. G. Shypula, A. Madaan, Y. Zeng, U. Alon, J. R. Gardner, Y. Yang, M. Hashemi, G. Neubig, P. Ranganathan, O. Bastani, and A. Yazdanbakhsh. Learning performance-improving code edits. In The Twelfth International Conference on Learning Representations, 2024.

[13] L. Van der Maaten and G. Hinton. Visualizing data using t-SNE. Journal of Machine Learning Research, 9(86):2579–2605, 2008.