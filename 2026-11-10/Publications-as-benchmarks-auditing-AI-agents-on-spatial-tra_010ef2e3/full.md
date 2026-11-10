# Publications as benchmarks: auditing AI agents on spatial transcriptomics alignment

Yiqun Chen<sup>1,2,\*</sup>, Jan Matthias<sup>3</sup>, Jianing Yao<sup>1</sup>, Hongyu (Eric) Zhao<sup>2</sup>, and Stephanie C. Hicks<sup>1,4,\*</sup>

<sup>1</sup>Department of Biostatistics, Johns Hopkins University, Baltimore, MD, USA <sup>2</sup>Department of Computer Science, Johns Hopkins University, Baltimore, MD, USA <sup>3</sup>Department of Biomedical Informatics and Data Science, Johns Hopkins University, Baltimore MD, USA <sup>4</sup>Department of Biomedical Engineering, Johns Hopkins University, Baltimore, MD, USA <sup>\*</sup>Co-corresponding authors: yiqunc@jhu.edu, shicks19@jhu.edu

## Abstract

AI agents are increasingly automating data analysis, but responsible adoption requires reliable performance evaluation, which is costly in specialized scientific domains. We introduce a framework that translates scientific publications into auditable benchmarks for AI agents and demonstrate its use on spatial transcriptomics alignment tasks. Across 121 tasks, 16 baseline methods, and six AI agents, the best agents (GPT-6-Astra and Gemini-4-Argon) matched or outperformed all baselines on 80% of tasks, often using surprisingly simple heuristics. Agent performance was robust to prompting and access to pre-installed packages. Benchmark construction also uncovered errors in prior work and shortcuts in current evaluation metrics. Finally, we show that agent-generated solutions can be distilled into a reusable program that ranks in the top 10% of baseline methods on unseen data. Our framework offers a scalable way to evaluate AI agents in specialized scientific domains by leveraging and stress-testing existing papers and metrics.

Advances in technology across scientific disciplines have generated “big data” and increasing demands for sophisticated computational tools that can make sense of and uncover insights from these data [1, 2]. For instance, in the biological sciences, high-throughput omics technologies have revolutionized the field over the past decades and catalyzed numerous computational innovations [3, 4]. Recent development of large language models (LLMs) is rapidly changing the potential solution space: artificial intelligence (AI) agents such as Claude Code, Codex and Gemini CLI can read and modify a codebase, install packages, run experiments, make plots and even write a complete manuscript from a short natural language instruction [5–10]. The appeal is direct for computational scientists, as agents could turn a complex request such as “integrate these heterogeneous data sources to build a performant model” into executable code in minutes rather than weeks.

However, responsible adoption requires careful evaluation of the solutions of agents, and increasing automation comes with more hidden risks, from hallucinated priors and facts and misinterpreted user intents to more nuanced selective reporting and test data leakage [15]. Adoption is further burdened by the simple fact that evaluating AI agents on scientific tasks is hard. On the agent side, unlike static supervised learning methods, agents interact with tools, the local environment and very often the open internet, and they can see and change their working environment so that they can explore and automate. If an answer key or a scorer is reachable, the agent may find and use it [16–18].

On the content side, broad scientific tasks have been assembled into suites of test cases, referred to as “benchmarks”, that are increasingly used to evaluate frontier AI agents [19–24]. Many existing efforts, however, focus on code generation and bug fixing and rely on automated tests as the evaluation metric [25–27], whereas scientific work is often more iterative and open-ended than most existing benchmarks: tasks are rarely precisely defined, success is multi-dimensional, and ground truth is often ambiguous and expensive to obtain. Building such tasks by hand also takes expert time for every task, and curators often favour tasks that separate the strongest systems over tasks that mirror what a scientist would ask. Moreover, there is an increasing demand to interpret and synthesize outputs from black-box models such as AI agents, so that future analysis and modeling of scientific data do not depend critically on proprietary AI models [28, 29], and the insights can be reused in other scientific domains [30–32].

![](images/6287cce9608a440ea249a96660522a586fb0396ef2f85f050014fa3842e7e4d5.jpg)  
Figure 1: Overview of the framework for translating published benchmarks into auditable tasks for AI agents. a, End-to-end workflow for aligning spatial transcriptomics datasets. Tasks are compiled from the source publication, executed in isolated containers, and audited through their execution traces to refine task definitions, evaluation metrics, and prompts. b, Mapping from the SABench study [11] to a structured task card for the human dorsolateral prefrontal cortex (DLPFC) dataset [12]. Components of the publication are mapped to the corresponding fields in the task specification, which undergoes automated validation prior to agent execution. c, Representative tasks from four of the five task families and the total number of tasks per family. Cells are colored by cell type or anatomical region (both withheld from the agents). d, Example task auditing on three-dimensional reconstruction of a mouse hemibrain across 129 MERFISH sections [13, 14], showing unaligned input sections, ground-truth coordinates, the top-performing published baseline (SPACEL), and the best agent solution (gpt-6-astra).

Motivated by these challenges, we propose a framework that leverages academic publications, especially benchmarking studies, to build auditable scientific benchmarks for AI agents. Academic publications, especially those featuring computational methods, are an underused source for evaluating agent performance. Good computational papers routinely identify public datasets, a clear input and output for each task, the scores used to rank existing and proposed methods and the code that computes those scores, which are many of the key ingredients of a scalable agent benchmark. Recent work has treated papers as executable objects in other ways: PaperBench asks agents to replicate machine learning papers from scratch [33], Paper2Agent converts a publication and its codebase into an agent that exposes the paper’s methods as callable tools [34], NatureBench distills 90 tasks from Nature-family publications to test whether agents surpass reported state-of-the-art performance [35], and SpatialBench introduces 146 verifiable problems drawn from spatial biology workflows, focusing on normalization, differential expression, and spatial domain clustering [36]. However, existing benchmarks typically grade outputs against static reference values reported in the original papers rather than re-evaluating published baselines and agents under a unified scoring harness, examining whether common evaluation metrics faithfully reflect ground truth, or auditing how agents arrive at their solutions. Importantly, we use publications not as replication targets, but as blueprints for executable, domain-specific evaluation suites: agents receive sanitized inputs while ground truth remains strictly withheld, and published methods are re-executed on the exact same data to provide a rigorous baseline reference.

We demonstrate the use of our framework in spatial transcriptomics alignment, an increasingly common step in understanding tissue biology in three dimensions. Spatial transcriptomics measures gene expression in biological samples with two-dimensional spatial resolution [37, 38]. There is great interest in aligning samples collected from adjacent tissue sections, or even from the same tissue measured with a different technology [39, 40]: integrating these samples allows a deeper understanding of complementary modalities, such as RNA, protein and chromatin, and of spatial organization, variation and other cellular phenotypes in three dimensions [13, 41]. Alignment, often called registration, also has wide applications in other domains, including functional magnetic resonance imaging (fMRI) and computed tomography (CT) [42–44]. Although numerous alignment methods have been proposed [45–56], a comprehensive benchmark to guide their application was until recently absent. SABench addressed this by systematically evaluating a diverse suite of leading methods on 295 alignment tasks across diverse datasets and technologies, measuring accuracy, efficiency, usability and robustness, and the downstream impact of alignment quality [11], and the GEASO study compared methods for alignment, stitching and three-dimensional reconstruction of large sets of sections [14].

Spatial transcriptomics alignment provides an ideal testbed for auditing scientific AI agents. Numerous domain-specific methods have been benchmarked on standardized datasets such as the human DLPFC [12] and the whole-mouse-brain MERFISH atlas [13], providing well-characterized human baselines. Moreover, these datasets span tasks with ground-truth coordinates alongside tasks evaluated indirectly via cell-type labels or gene expression similarity, enabling a direct test of how metric choice shapes evaluation outcomes.

Here we translate these two alignment studies into 121 auditable tasks and evaluate six frontier AI models operating as coding agents across basic and domain-enriched prompting conditions (Figure 1). Rebuilding tasks and re-executing 16 baseline methods under a single scoring harness also audits the source benchmarks—uncovering a 20-fold unit mismatch in a published evaluation—and demonstrates the necessity of strict container isolation to prevent filesystem leakage. Across tasks with known positions, the best agent run matched or outperformed all published baselines on 22 of 28 tasks (79%), whereas single runs did so on 21–64% (Figure 2). Providing pre-installed alignment packages and literature recommendations shifted agent workflows and increased cost without consistently improving accuracy (Figure 3). Crucially, evaluating alignments without ground truth remains a major bottleneck: label-based metrics tracked true spatial error far better than expression similarity metrics (Figure 4), and the algorithmic steps associated with top scores depended on the evaluation metric (Figure 5). Finally, distilling agent runs into a standalone deterministic program outperformed 87% of baselines on held-out tasks at zero LLM cost, while transferring less effectively to unseen task families (Figure 6).

## Methods

## Task construction

Our benchmark comprises 121 tasks, each requiring an agent to register two or more spatial transcriptomics sections into a common coordinate system. Tasks were compiled using Paper2Bench, an automated pipeline that translates published benchmarks and methods papers into executable task packages. The pipeline extracts experimental units from figures, tables, methods, code, and public repositories; attaches deterministic scoring profiles; generates task prompts without exposing held-out evaluation data; and constructs isolated agent workspaces. Candidate tasks must pass a suite of deterministic validation checks before inclusion (Supplementary Note 1).

The 121 tasks span five task families derived from three sources: serial tissue sections, a primary motor cortex (MOp) series, and cross-platform section pairs from SABench [11]; rotation, cropping, and three-dimensional reconstruction tasks from GEASO [14] and the whole-mouse-brain MERFISH atlas; and synthetic tissues generated with the Albis simulator [57] (Supplementary Tables 1, 2 and 3). Groundtruth spatial coordinates are available for 32 tasks and cell-type or anatomical region annotations for 69 tasks, while the remaining tasks are evaluated using gene expression similarity.

For each task, the agent receives sanitized h5ad files containing only the expression matrix, feature metadata, anonymized observation identifiers, and spatial coordinates. The agent outputs one CSV table of aligned (x, y) coordinates per section, with the reference section retaining its original positions.

## Published methods and baselines

We re-executed 13 published spatial alignment packages—CAST [53], GEASO [14], GPSA [47], moscot [55], PASTE [45], PASTE2 [46], SANTO [54], SLAT [48], SPACEL [51], Spateo [52], STAIR [56], STalign [49], and STAligner [50]—across 41 parameter configurations on the same task inputs [11, 14] and the simulated tissue suite [57]. The execution was assisted by coding agents and verified by human inspection and checks to make sure the reproduced values are close to prior reported values whenever possible [11, 14]. On each task, the best published baseline is defined as the top-scoring configuration across all packages. We also evaluated three geometric baselines across all 121 tasks: an identity baseline (unaligned coordinates) and similarity transformations estimated by principal component analysis (PCA) and iterative closest point (ICP) registration. Software versions, parameter configurations, and comparisons with originally published values are detailed in Supplementary Note 2 and Supplementary Figure 1.

## Agents, prompting conditions, and execution harness

We evaluated six frontier LLMs through their respective command-line coding interfaces: Claude Code (claude-opus-5.5 and claude-sonnet-5.5), Codex CLI (gpt-6-astra and gpt-5.5), and Gemini CLI (gemini-4-argon and gemini-3.5-flash) (Supplementary Table 4). All agents were executed in noninteractive mode with permission prompts bypassed, no custom system instructions, and a clean environment per run. Our primary analysis mimics a low-overhead setting for scientists, referred to as the basic prompt condition: the workspace contained only the input data, task specification, and a standard scientific Python environment (numpy, scipy, pandas, scikit-learn, and h5py), without any domainspecific alignment packages installed or referenced. We assessed the sensitivity of agent performance under the rich-context condition, where published alignment packages were pre-installed in isolated environments and the prompt was augmented with usage guidelines, API examples, package documentation, and literature-derived recommendations summarizing which tool families performed best for each data modality in prior benchmarks (Supplementary Note 3).

Every run executed inside an isolated Singularity container [58] with read-only software and data mounts, a single writable workspace directory, and no access to held-out evaluation files. Primary analyses use the first completed, scored run for each model, task, and prompting condition. Run cost was recorded as the API-equivalent expenditure reported by Claude Code and calculated from token usage at standard API list prices for OpenAI and Google models.

## Leakage prevention and auditing

Workspace inputs were constructed using a strict allowlist that stripped cell-type annotations, groundtruth coordinates, and auxiliary metadata, while anonymizing cell identifiers and randomly permuting row order. Following each run, an automated audit scanned the execution trace, workspace files, and submitted outputs for references to restricted filesystem paths, withheld metadata field names, or credentials (Supplementary Note 4, Supplementary Figure 2 and Supplementary Table 5).

## Evaluation metrics and scoring

Scoring was executed outside the Singularity container after agent termination across all designated section pairs in each task, as preliminary pilots showed that exposing the scorer inside the workspace led agents to inspect or modify the scoring function. Coordinate error was quantified as the mean absolute error (MAE) between submitted and ground-truth coordinates. Label agreement was evaluated on spatially matched cells across aligned sections using nearest-neighbor label matching accuracy, anatomical region overlap (alpha-shape intersection over union), and mutual nearest-neighbor (MNN) label concordance [11, 14]. Expression agreement was quantified on a spatial grid over the overlap region using gene-wise Pearson correlation, cosine similarity, and structural similarity (SSIM). Each task’s primary metric was coordinate error when ground-truth positions were available and the composite agreement score (mean of expression and label metrics) otherwise. Formal metric definitions and numerical parity checks against the original SABench codebase are provided in Supplementary Note 5.

## Strategy distillation and cross-validation

To test whether agent-discovered workflows could be consolidated into a standalone algorithm, we performed five-fold cross-validation under both random task splits and task-family holdouts. For each fold, an isolated instance of Claude Code (claude-opus-5.5) was provided with the final scripts and training scores of claude-opus-5.5 and gpt-6-astra from the four training folds and instructed to synthesize a single, general-purpose Python alignment program. The resulting program was then executed on the held-out fold without any LLM calls and graded by the benchmark scorer, with runtime crashes or invalid outputs scored as failures (Supplementary Note 3).

## Statistical analysis

We partitioned variance in alignment accuracy, runtime, and cost across task, model, prompt condition, and replicate runs using factorial analysis of variance (ANOVA). A run was considered to match or outperform the best published baseline if its primary score equaled or exceeded the highest score achieved by any published package configuration on that task. Confidence intervals and hypothesis tests were computed via cluster bootstrap resampling (Supplementary Note 6).

## Execution trace analysis

Agent execution logs were parsed into sequential tool-call steps and classified into eight operational phases (such as data inspection, method exploration, candidate evaluation, and output validation), alongside 48 fine-grained algorithmic tags applied to both intermediate scripts and final submitted code (Supplementary Note 7 and Supplementary Figures 3 and 4).

## Results

## Translating published benchmarks into auditable agent tasks

We compiled executable agent tasks from two spatial transcriptomics alignment studies: SABench [11] and GEASO [14] (Figure 1a). Each component of a source paper maps to a machine-readable task card: dataset tables specify input sections, experimental protocols define section pairings and geometric perturbations, evaluation code defines the scorer, and benchmarked tools provide the reference baselines (Figure 1b). During execution, an agent observes only sanitized gene expression matrices, spatial coordinates, task instructions, and the required output schema, while ground-truth coordinates, cell-type annotations, and scoring scripts remain hidden.

We reconstructed 98% of the tasks used in SABench (source data are not directly available for download for the remainder) and verified that our scorer matched the authors’ evaluation results to within 10<sup>−7</sup> (Supplementary Note 1). After grouping simulated tissues from the same source section and serial alignments within the same dataset, our benchmark spans 121 tasks across 19 datasets and five task families (Figure 1c): cross-platform section pairs, serial sections, simulated tissues with known positions, geometric stress tests (rotations, crops, and partial overlaps), and three-dimensional reconstructions of up to 129 sections. Rebuilding the published evaluations from raw inputs also uncovered a 20-fold coordinate unit mismatch between ground-truth and input positions in the published evaluation of the Drosophila embryo stack (Supplementary Note 1).

Figure 1d illustrates a representative three-dimensional reconstruction task requiring registration of 129 scrambled MERFISH sections of a mouse hemibrain across six evaluated frontier coding agents. The top published baseline on this task, SPACEL, left a mean coordinate error of 846 spatial units, whereas gpt-6-astra reduced the error to 0.13 in 30 tool calls (32 minutes; \$2.91). Trace inspection shows that the agent chained pairwise rigid alignments across adjacent sections, detected cumulative rotational drift along the stack, and enforced a global constraint along the straight medial boundary of the hemibrain. Notably, however, gene-expression-based evaluation metrics ranked SPACEL equal to this near-exact re construction, illustrating a potential discrepancy between proxy scores and coordinate-based scores.

## Preventing benchmark leakage through container isolation

In an initial pilot where evaluation scripts and ground-truth files resided on the shared filesystem outside the workspace—even with prompts explicitly instructing agents not to search parent directories—trace audits revealed four modes of test leakage (Supplementary Table 5). Agents traversed the filesystem to copy ground-truth coordinate files, inspected or modified the scoring script, reused outputs from concurrent runs, or extracted precomputed results bundled with cloned method repositories.

We therefore isolated every run inside the Singularity container (Figure 1a, step 3) with read-only mounts and no access to held-out data or scoring code, followed by an automated post-run trace and workspace audit (Supplementary Figure 2). We then manually searched for potential answers and ground-truth file names in the logs and generated files, and confirmed that the resulting runs did not search for the ground truth or exhibit obvious model hacking behavior such as rewriting scoring files (Supplementary Note 4).

![](images/7487a7809d0a44615212b62baf4ff8f603ff8fa96b8162e2fedb2c8114781031.jpg)

![](images/9369654289e1ad001e2aa34834a22823293dff1dbc44e1eef6e2cc142454dc05.jpg)  
b

![](images/5f5ef482ef2723983af5b2d6a47ae51b59271f5e74f9fdcfbcc873d9b883211d.jpg)  
c Basic-prompt runs at least as good as the best published method (%)

<sub>Runtime</sub> <sub>and</sub> <sub>accuracy</sub> <sub>of</sub> <sub>every</sub> <sub>method</sub> <sub>and</sub> <sub>agent</sub>d  
![](images/2a3e306cf53346bee26f029ecd5bc27977b3b632e2d43364f7c6cda3335a1f9d.jpg)

Examples where agents do better and worse than the best published method (coordinate error, lower is better)e  
![](images/1731d5d510392960edc0000dff714378bb1232d5c59bec789db370956e1216ce.jpg)  
Figure 2: Comparison of AI agents with published spatial alignment methods under a unified evaluation harness. a, Coordinate error (mean absolute error, MAE; log scale) of the best basic-prompt agent run versus the best published baseline across 28 tasks with ground-truth coordinates. The dashed line indicates parity. b, Expression agreement (mean of gene-wise Pearson correlation, cosine similarity, and structural similarity) of the best agent run versus the best published method across 118 tasks, distinguishing cross-platform pairs (black) from other task families (grey). c, Percentage of tasks on which a single basic-prompt run of each model matched or outperformed the best published baseline, stratified by task family for expression agreement (left) and coordinate error (right). d, Mean percentage of the 16 reference baselines (13 published packages at their optimal configuration and 3 geometric baselines) outperformed per task versus median runtime (log scale) for published packages (black), geometric baselines (open circles), and basic-prompt agent runs (colored circles; shaded region). e, Representative alignments across four tasks comparing unaligned inputs, ground-truth coordinates, the best published baseline, and the best agent solution, with MAE reported below each panel. Top, tasks where agents outperformed baselines; bottom, tasks where published baselines performed better.

## Top agent solutions outperform published baselines

The best AI agents consistently matched or outperformed state-of-the-art spatial alignment methods, achieving lower error on 79% of tasks with ground-truth positions (Figure 2a). Across all 118 tasks evaluated by expression agreement, the best agent matched or beat the top published method on 57% of tasks, driven by strong gains on cross-platform pairs (41 of 45; Figure 2b), with robust win rates across other expression and label metrics (Supplementary Figure 1e).

Among AI agents, however, performance varied substantially by model version, falling roughly into three tiers: gpt-6-astra and gemini-4-argon; followed by claude-opus-5.5, gpt-5.5, and claude-sonnet-5.5; and finally gemini-3.5-flash. Stronger models also produced more reproducible results across replications, as they were more likely to converge on the same solution strategies (Figure $^ { 2 \mathrm { c } , \mathrm { d } ) }$ , with rank correlations across runs of $\rho ~ = ~ 0 . 6 3 , ~ 0 . 7 1 \mathrm { - } 0 . 7 4 ~ ( 0 . 1 3 \mathrm { - } 0 . 1 4 )$ , and 0.93 (0.10) for gemini-3.5-flash, the Claude models, and $\mathtt { g p t - 6 - a s t r a }$ , respectively (Supplementary Figure 5 and Supplementary Note 8).

## Reproduction of published baselines

Benchmarking the 16 published baselines showed that no single baseline method dominated across every tissue regime (Supplementary Note 2 and Supplementary Figure 1a–d). Pooled across all task families, SLAT led overall, followed by Spateo and CAST. This largely agrees with previous benchmark results [11], with the caveat that the best method varied by regime: SLAT led on serial sections and on simulated misalignment due to rotation and cropping, Spateo performed best on the motor cortex series, CAST on cross-platform pairs, and STalign on simulated tissues. Interestingly, custom agent solutions often dramatically reduced coordinate error in large-scale three-dimensional serial alignment, on both simulated data and a 129-section mouse brain reconstruction, frequently by applying global refinement after serial alignment (Figure 2e).

## Domain-specific context makes agents work longer and explore more without improving accuracy

With minimal prompting, 94% of agent runs solved alignments from first principles, writing custom code to solve the optimization problem rather than calling existing bioinformatics packages (Figure 3). Under the rich-context prompt, where we provided the problem context and commonly used packages, 89% of runs executed a published package and 54% retained one in their final submission.

Despite this shift in tool adoption, accuracy did not consistently improve. Mean changes in withintask percentile ranged from +11.5 (95% CI: 2.0 to 18.3) to −6.4 points (95% CI: −13.3 to 0.6) for claude-opus-5.5 and gemini-4-argon, respectively. This is largely because package-based submissions did not consistently outperform custom code. Token usage and cost, on the other hand, consistently rose, averaging 1.5× higher overall (Figure 3c). Models also differed in how they adopted external tools: claude-opus-5.5 and claude-sonnet-5.5 were much more likely than other models to retain custom code even after trying bioinformatics packages (Figure 3d,f).

## Label-based metrics track spatial coordinate accuracy more reliably than expression similarity

In most experimental applications, ground-truth positions are unavailable, and alignments must be evaluated using proxy metrics based on cell-type labels or gene expression similarity [11] (Figure 4a). Across the 32 tasks with ground-truth coordinates, label-based metrics correlated substantially better with true coordinate error (median within-task Spearman $\rho = 0 . 6 6 \ – 0 . 7 1 )$ than metrics built from gene expression alone $( \rho = 0 . 4 1 \mathrm { - } 0 . 5 4 _ { \cdot }$ ; Figure 4b). Agreement also varied by task family. On densely sampled, overlapping mouse brain sections, expression agreement closely tracked coordinate error $( \rho = 0 . 9 7 ;$ Figure 4d). On simulated Visium spots, however, expression agreement was uninformative $( \rho = - 0 . 0 3 )$ , ranking a coarse

![](images/769711dec8e45f62cab1b08f140398ee62d4ff3a126387180746e2ffffde7aab.jpg)

b  
![](images/95b25cb78e040c429c3560d8a702cc32ef54b718d66ddcaee72f42586760b5ca.jpg)

c  
![](images/c419552423bcea9e7b9f5f0fb22550662ea429e41c1e93753cd5c46fb70160b5.jpg)

![](images/68b399ba4b296c74ead3e6f10a993b938a114878a0e8fa2eb8941ad121363bb0.jpg)  
e

f  
![](images/3a72e9db1db02d5f9b9d701f850cec634cddb34c04cf3d5b258b64110c573d21.jpg)  
Figure 3: Impact of domain-specific context on agent tool use, alignment accuracy, and computational cost. a, Percentage of scored runs that executed each published alignment package during exploration and retained it in the final submitted code under the basic and rich-context prompts. b, Change in within-task percentile under the richcontext prompt relative to the basic prompt across 121 tasks. Points denote individual tasks; horizontal bars indicate the mean and 95% cluster-bootstrap confidence interval across source datasets. c, Mean within-task percentile versus median API cost per run (US\$, log scale) under the basic prompt (open circles) and rich-context prompt (filled circles). Error bars denote 95% bootstrap confidence intervals. d, Algorithmic composition of final submitted code under each prompting condition, categorized by custom geometric transformation class or published package output. e, Difference in the proportion of runs incorporating specific algorithmic steps in the final submission (richcontext minus basic prompt, in percentage points). f, Mean within-task percentile of rich-context runs whose final submission relied on custom code (open circles) versus a published package (filled circles). Error bars denote 95% bootstrap confidence intervals.

PCA fit first despite a coordinate error (MAE) 20-fold larger than that of the best agent run (Figure 4e). Expression similarity is thus informative when sections overlap densely, but it can reward transformations that align broad expression gradients while misplacing cells.

## Agent execution traces reveal distinct problem-solving strategies

Agents also differed in their workflows (Figure 5). The Claude models devoted a substantial share of their time and tokens to evaluating candidate solutions and most often inspected their own alignment plots, whereas the Gemini models spent the largest share of their calls executing more, and more complex, candidate solutions (Figure 5a,b). gpt-6-astra was the most meticulous reviser, typically modifying its

b Which scores track the true error
<table><tr><td>Expression agreement 119 gene correlation, cosine, ŠSIM, MI, MNN expression tasks</td></tr></table>

a  
![](images/fb50c18fd5cfdd86e531800f5434401d1439001051a6d7219484dfc7409d5c84.jpg)

![](images/5a794e43d55ca76ee6783f791a1eb08148a3bf4e3104981a7fbaf6eb3bf3141e.jpg)

![](images/c1febc69f06c84be447748647d5fb32e44c81edccd339b3c8e824b2c68ed9871.jpg)

d  
![](images/4e358bb320a11e5627fabf4db6fa48aea50c352f66c0a840a83d72b92f18ce0c.jpg)

![](images/a3baa3eb2d614234cdbfd0462eeca6459f1601e291230269d807e0e4665fec9b.jpg)

![](images/5112740fe60576a33dd1a100507cc4fba9ff0fe59a532d34d57814958aa942f2.jpg)

![](images/80e43a8ca2c516cc8a1080bf93ea864b4f948c3df44952372df417a7873d71d1.jpg)  
Figure 4: Concordance of proxy alignment metrics with ground-truth coordinate error. a, Taxonomy of spatial alignment evaluation metrics, their required supervision, and coverage across the 121 benchmark tasks. MAE, mean absolute error; MNN, mutual nearest neighbors; SSIM, structural similarity; MI, mutual information. b, Withintask Spearman rank correlation between each proxy metric and ground-truth coordinate error across 32 tasks with known positions (signed so that positive values denote agreement with spatial accuracy). Points represent individual tasks (real tissue, black; simulated tissue, grey); vertical bars and right-hand labels indicate median correlation and task count. c, Median within-task Spearman correlation across all pairs of evaluation metrics. d–f, Comparison of expression agreement (d,e) and MNN label agreement (f) against coordinate error (log scale) across agent runs, published methods, and geometric baselines on overlapping mouse brain sections (d, with inset magnifying MAE < 30) and simulated Visium spot data (e,f).

initial submission at least three times (Figure 5b).

Among submitted solutions, gpt-6-astra had the most distinctive profile: it mostly combined simple affine transformations with numerical optimization and nonrigid warps (Figure 5c), all of which characterized top-performing solutions pooled across models (Figure 5e). Additional predictors of better performance included nearest-neighbor matching and Procrustes registration (Supplementary Note 7).

Inspecting individual solutions further revealed that agents often built on existing ideas while optimizing performance more carefully and exhaustively. For example, on cross-platform and serial tasks, gpt-6-astra frequently paired an affine search on smoothed shared-gene maps with a penalized cubic Bspline warp (Figure 5f). This strategy builds on the two-stage registration paradigm of CAST and STalign but adds 25% held-out gene cross-validation and a positive-Jacobian fold check to prevent over-warping.

![](images/d8aa3e70c277b9371d092308d70f77d6ccde086636d5b6626c999c3d9dff5fd5.jpg)

![](images/b347133d96ad4cd769ff3436bd40e56bffa45f11b45de940c10b7cc347bf2442.jpg)

![](images/f73920ef4436be38aa1ac39dfa2f3520fcebc8e4160996c4cb5b2fb48635f898.jpg)  
Figure 5: Execution dynamics, algorithmic building blocks, and metric-dependent step effects across coding agents. a, Mean share of tool calls allocated to each operational phase across models. b, Frequency of key workflow behaviors across models, including visual inspection of generated plots and iterative submission refinement (≥ 3 rewrites). c, Prevalence of core algorithmic steps in the final submitted code of basic-prompt runs across models. d, Prevalence of algorithmic steps in the top-scoring run of each task versus all other runs. e, Model- and prompt adjusted association of each step in the final code with within-task percentile across all tasks (primary metric, left) and across the 32 tasks with ground-truth coordinates (MAE, right). Bars and whiskers show point estimates and 95% bootstrap confidence intervals; solid bars indicate confidence intervals excluding zero. f, Example winning strategy from gpt-6-astra combining an affine search on smoothed shared-gene maps with a penalized cubic Bspline warp, 25% held-out gene cross-validation, and a positive-Jacobian fold check.

a Distilling agent runs into a standalone program evaluated on unseen tasks  
![](images/00244bbca7cff06ff55438cc2e7aa0e538d8e1d9a132cb8eac5da83d694f68d2.jpg)

b On unseen tasks: methods beaten on a task, by task family  
![](images/8f246e8400182dc20132b10eea163912dc4027c05d6912b37ed6a6ea04e52fa4.jpg)  
Figure 6: Distillation of agent runs into a standalone alignment program evaluated on held-out tasks. a, Fivefold cross-validation design under random and task-family splits. An isolated synthesis agent (Claude Code with claude-opus-5.5) inspects the final scripts and training scores of claude-opus-5.5 and gpt-6-astra on four folds and writes a standalone Python program, which executes without LLM calls on the held-out fold. b, Mean percentage of the 16 reference baselines outperformed on held-out tasks, by task family, for distilled programs, individual basic-prompt agent runs, published packages, and geometric baselines (dashed lines mark the median package; n denotes task count).

## Distilling agent trajectories into a standalone alignment program

To test whether agent-discovered heuristics could be consolidated into a reusable, model-free algorithm, we tasked an isolated synthesis agent (Claude Code with claude-opus-5.5) with distilling the basic-prompt solutions and training scores of the two top-performing agents (claude-opus-5.5 and gpt-6-astra) into a single Python program under a five-fold cross-validation scheme (Figure 6a). Each synthesized program can then be applied and adapted to other alignment problems without re-querying proprietary models.

We considered two splitting schemes: random splits across all tasks and splits by task family. Broadly, the synthesized programs combined rules such as per-section gene-wise z-scoring and reliabilityweighted spatial smoothing with coarse-to-fine registration. Aggregated across held-out test folds, the random-split program outperformed 87% of the 16 reference baselines on average, matching interactive frontier agents and exceeding the median published package (54%), whereas the family-split program outperformed 78% overall but dropped to 30% on unseen simulated tissues (Figure 6b). Executing the program takes a median of 47 s per task with zero inference cost, so it amortizes its one-time synthesis cost after 11 tasks compared with querying an average agent in our result set.

## Discussion

We translated published spatial transcriptomics benchmarks into an auditable suite of agent tasks and systematically evaluated six frontier coding agents alongside 16 baseline methods on aligning spatial transcriptomics data. Our work highlights several core principles for evaluating AI agents in scientific workflows. First, strict environment isolation is essential: when executed on a shared filesystem, agents routinely located and exploited ground-truth files, evaluation scripts, or precomputed outputs despite explicit prompt instructions forbidding such access. Second, top-performing agents frequently matched or outperformed specialized alignment packages. Remarkably, agents achieved these results primarily by writing compact, custom numerical optimization routines rather than invoking domain-specific libraries. Moreover, equipping agents with pre-installed alignment packages and literature-derived recommendations increased computational cost without consistently improving accuracy.

These findings point to broader methodological lessons for benchmarking scientific AI agents. Unlike software engineering tasks, where the desired output can often be specified by unit tests, a scientific coding task admits a broad distribution of exploratory workflows and candidate solutions. Published datasets and results offer a potentially scalable foundation for rigorous evaluation. Additionally, as in many other scientific tasks, metric choice plays a major role in our results, as widely used proxy scores can lead to different model rankings. We emphasize the importance of calibrating against ground truth to guard against metric hacking, such as over-warping in the spatial alignment example. Moreover, because execution traces record every exploratory check, failed attempt, and refinement, they enable systematic program distillation—compiling effective heuristics discovered by frontier models into fast, deterministic, and open-source scientific software.

Several limitations should be considered when interpreting our results. First, we focused on spatial transcriptomics alignment; extending our publication-to-benchmark pipeline to more problems, especially open-ended biological analyses, will require domain-specific data contracts and validation checks. Second, because frontier models and command-line harnesses evolve rapidly, point-in-time rankings reflect current default vendor configurations and might change as new models are released and older ones are deprecated. This is in fact a primary motivation for our exploration of program synthesis as a path toward reproducible, vendor-independent tools. Third, although more exhaustive hyperparameter tuning of baseline packages or prompt engineering could further improve individual scores, our evaluation intentionally reflects the practical scenario in which scientists apply these tools with a limited tuning budget (or, as in our case, with agent-assisted tuning).

To support transparent auditing and community benchmark development, we publicly release the Paper2Bench pipeline, all tasks, the containerized evaluation harness, and an interactive trace explorer containing all agent logs. More broadly, rather than viewing AI agents solely as black-box replacements for human analysts, we envision a collaborative paradigm in which auditable benchmarks and trace distillation harness agent exploration to build more rigorous, interpretable, and reusable tools.

## Data availability

All data are available at Zenodo (https://zenodo.org/records/XXXXXXX), and traces can be explored at https://ericzhy0815.github.io/agent-trace-explorer/.

## Code availability

Code to reproduce our analysis is available at https://github.com/Y-Chen-LAB/spatial-alignment-agent.

## Acknowledgements

We acknowledge the Joint High Performance Computing Exchange (JHPCE) at the Johns Hopkins Bloomberg School of Public Health for computational support. This work was supported in part by a Gemini Academic Research Program and a Google Academic Research Award to Y.C. AI tools helped edit the text of this manuscript and write analysis and plotting code; the authors reviewed the final output and take full responsibility for the content and integrity of the work.

## References

[1] Vivien Marx. Biology: The big challenges of big data. Nature, 498(7453):255–260, 2013. doi: 10.1038/ 498255a.

[2] Zachary D. Stephens, Skylar Y. Lee, Faraz Faghri, Roy H. Campbell, Chengxiang Zhai, Miles J. Efron, Ravishankar Iyer, Michael C. Schatz, Saurabh Sinha, and Gene E. Robinson. Big data: Astronomical or genomical? PLOS Biology, 13(7):e1002195, 2015. doi: 10.1371/journal.pbio.1002195.

[3] Sara Goodwin, John D. McPherson, and W. Richard McCombie. Coming of age: ten years of nextgeneration sequencing technologies. Nature Reviews Genetics, 17(6):333–351, 2016. doi: 10.1038/nrg. 2016.49.

[4] Tim Stuart and Rahul Satija. Integrative single-cell analysis. Nature Reviews Genetics, 20(5):257–272, 2019. doi: 10.1038/s41576-019-0093-7.

[5] Anthropic. Claude Code overview. https://code.claude.com/docs/en/overview, 2026. Accessed 2026-10-20.

[6] OpenAI. Introducing Codex. https://openai.com/index/introducing-codex/, 2025. Accessed 2026-10-20.

[7] Google. Gemini CLI. https://github.com/google-gemini/gemini-cli, 2025. GitHub repository. Accessed 2026-10-20.

[8] Juraj Gottweis, Wei-Hung Weng, Alexander Daryin, Tao Tu, Petar Sirkovic, Artiom Myaskovsky, Grzegorz Glowaty, Felix Weissenberger, Alessio Orlandi, Dan Popovici, Anil Palepu, Keran Rong, Ryutaro Tanno, Khaled Saab, Fan Zhang, Jacob Blum, Andrew Carroll, Kavita Kulkarni, Nenad Tomašev, Dina Zverinski, Ivor Rendulic, Elahe Vedadi, Florian Hasler, Luka Rimanic, Marina Boia, Ivan Budiselic, Ben Feinstein, Mathias Bellaiche, Tom Sheffer, Jan Freyberg, Jeremy Ratcliff, Ottavia Bertolli, Katherine Chou, Avinatan Hassidim, Burak Gokturk, Amin Vahdat, Yuan Guan, Vikram Dhillon, Eeshit Dhaval Vaishnav, Byron Lee, Tiago R. D. Costa, José R. Penadés, Gary Peltz, Yossi Matias, James Manyika, Demis Hassabis, Yunhan Xu, Pushmeet Kohli, Annalisa Pawlosky, Alan Karthikesalingam, and Vivek Natarajan. Accelerating scientific discovery with Co-Scientist. Nature, 655(8122):487–496, 2026. doi: 10.1038/s41586-026-10644-y. URL https://doi.org/10.1038/ s41586-026-10644-y.

[9] Eser Aygün, Anastasiya Belyaeva, Gheorghe Comanici, Marc Coram, Hao Cui, Jake Garrison, Renee Johnston, Anton Kast, Cory Y. McLean, Peter Norgaard, Zahra Shamsi, David Smalling, James Thompson, Subhashini Venugopalan, Brian P. Williams, Chujun He, Sarah Martinson, Martyna Plomecka, Lai Wei, Yuchen Zhou, Qian-Ze Zhu, Matthew Abraham, Erica Brand, Anna Bulanova, Jeffrey A. Cardille, Chris Co, Scott Ellsworth, Grace Joseph, Malcolm Kane, Ryan Krueger, Johan Kartiwa, Dan Liebling, Jan-Matthis Lueckmann, Paul Raccuglia, Xuefei Julie Wang, Katherine Chou, James Manyika, Yossi Matias, John C. Platt, Lizzie Dorfman, Shibl Mourad, and Michael P. Brenner. An AI system to help scientists write expert-level empirical software. Nature, 654:909–916, 2026. doi: 10.1038/s41586-026-10658-6. URL https://doi.org/10.1038/s41586-026-10658-6.

[10] Chris Lu, Cong Lu, Robert Tjarko Lange, Yutaro Yamada, Shengran Hu, Jakob Foerster, David Ha, and Jeff Clune. Towards end-to-end automation of AI research. Nature, 651:914–919, 2026. doi: 10.1038/s41586-026-10265-5. URL https://doi.org/10.1038/s41586-026-10265-5.

[11] Yunzhi Yan, Tianyi Gu, Chengcheng Sun, Yinghao Zhang, Yan Cui, Senlin Lin, Qi Zou, Yixuan Du, Chuangyi Han, Kairan Kang, Sijie Li, Yihan Zhao, Zhihong Lin, Zhiyuan Yuan, and Bin-Zhi Qian. Benchmarking alignment methods for spatial transcriptomics data. Nature Computational Science, 6(5):524–541, 2026. doi: 10.1038/s43588-026-00977-z. URL https://doi.org/10.1038/ s43588-026-00977-z.

[12] Kristen R. Maynard, Leonardo Collado-Torres, Lukas M. Weber, Cedric Uytingco, Brianna K. Barry, Stephen R. Williams, Joseph L. Catallini, Matthew N. Tran, Zachary Besich, Madhavi Tippani, Jennifer Chew, Yifeng Yin, Joel E. Kleinman, Thomas M. Hyde, Nikhil Rao, Stephanie C. Hicks, Ker Martinowich, and Andrew E. Jaffe. Transcriptome-scale spatial gene expression in the human dorsolateral prefrontal cortex. Nature Neuroscience, 24(3):425–436, 2021. doi: 10.1038/s41593-020-00787-0.

[13] Meng Zhang, Xingjie Pan, Won Jung, Aaron R. Halpern, Stephen W. Eichhorn, Zhiyun Lei, Limor Cohen, Kimberly A. Smith, Bosiljka Tasic, Zizhen Yao, Hongkui Zeng, and Xiaowei Zhuang. Molecularly defined and spatially resolved cell atlas of the whole mouse brain. Nature, 624(7991):343–354, 2023. doi: 10.1038/s41586-023-06808-9.

[14] Yu Wang, Zaiyi Liu, and Xiaoke Ma. Network model for alignment, stitching and slice-to-volume 3D reconstruction of large-scale spatially resolved slices. Nature Communications, 17(1):9451, 2026. doi: 10.1038/s41467-026-71042-6.

[15] Ziming Luo, Atoosa Kasirzadeh, and Nihar B. Shah. The more you automate, the less you see: Hidden pitfalls of AI scientist systems. In NeurIPS 2025 Workshop on AIfor Science, 2025. URL https: //arxiv.org/abs/2509.08713.

[16] Monte MacDiarmid, Benjamin Wright, Jonathan Uesato, Joe Benton, Jon Kutasov, Sara Price, Naia Bouscal, Sam Bowman, Trenton Bricken, Alex Cloud, Carson Denison, Johannes Gasteiger, Ryan Greenblatt, Jan Leike, Jack Lindsey, Vlad Mikulik, Ethan Perez, Alex Rodrigues, Drake Thomas, Albert Webson, Daniel Ziegler, and Evan Hubinger. Natural emergent misalignment from reward hacking in production RL. arXiv preprint arXiv:2511.18397, 2025. URL https://arxiv.org/abs/ 2511.18397.

[17] METR. Recent frontier models are reward hacking. https://metr.org/blog/ 2025-06-05-recent-reward-hacking/, 2025. Published 5 June 2025. Accessed 2026-10-20.

[18] Yuxuan Zhu, Tengjun Jin, Yada Pruksachatkun, Andy Zhang, Shu Liu, Sasha Cui, Sayash Kapoor, Shayne Longpre, Kevin Meng, Rebecca Weiss, Fazl Barez, Rahul Gupta, Jwala Dhamala, Jacob Merizian, Mario Giulianelli, Harry Coppock, Cozmin Ududec, Antony Kellermann, Jasjeet Sekhon, Jacob Steinhardt, Sarah Schwettmann, Arvind Narayanan, Matei A. Zaharia, Ion Stoica, Percy Liang, and Daniel Kang. Establishing best practices in building rigorous agentic benchmarks. In Advances in Neural Information Processing Systems 38 (NeurIPS 2025), Datasets and Benchmarks Track, volume 38, pages 184435–184475, 2025. doi: 10.52202/085713-5547. Preprint: arXiv:2507.02825 (titled “Establishing Best Practices for Building Rigorous Agentic Benchmarks”).

[19] Ziru Chen, Shijie Chen, Yuting Ning, Qianheng Zhang, Boshi Wang, Botao Yu, Yifei Li, Zeyi Liao, Chen Wei, Zitong Lu, Vishal Dey, Mingyi Xue, Frazier N. Baker, Benjamin Burns, Daniel Adu-Ampratwum, Xuhui Huang, Xia Ning, Song Gao, Yu Su, and Huan Sun. ScienceAgentBench: Toward rigorous assessment of language agents for data-driven scientific discovery. In Proceedings of the 13th International Conference on Learning Representations (ICLR),

pages 96934–96990, 2025. URL https://proceedings.iclr.cc/paper\_files/paper/2025/hash/ f12b4df26344f3be803c06b555252efe-Abstract-Conference.html.

[20] Liqiang Jing, Zhehui Huang, Xiaoyang Wang, Wenlin Yao, Wenhao Yu, Kaixin Ma, Hongming Zhang, Xinya Du, and Dong Yu. DSBench: How far are data science agents from becoming data science experts? In Proceedings of the 13th International Conference on Learning Representations (ICLR), pages 32597–32649, 2025. URL https://proceedings.iclr.cc/paper\_files/paper/2025/hash/ 50e9ad960ae78b741a6b4fea533f2eaf-Abstract-Conference.html.

[21] Ken Gu, Ruoxi Shang, Ruien Jiang, Keying Kuang, Richard-John Lin, Donghe Lyu, Yue Mao, Youran Pan, Teng Wu, Jiaqian Yu, Yikun Zhang, Tianmai M. Zhang, Lanyi Zhu, Mike A. Merrill, Jeffrey Heer, and Tim Althoff. BLADE: Benchmarking language model agents for data-driven science. In Findings of the Association for Computational Linguistics: EMNLP 2024, pages 13936–13971, Miami, Florida, USA, 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.findings-emnlp.815.

[22] Bodhisattwa Prasad Majumder, Harshit Surana, Dhruv Agarwal, Bhavana Dalvi Mishra, Abhijeetsingh Meena, Aryan Prakhar, Tirth Vora, Tushar Khot, Ashish Sabharwal, and Peter Clark. DiscoveryBench: Towards data-driven discovery with large language models. In Proceedings of the 13th International Conference on Learning Representations (ICLR), pages 4556–4579, 2025. URL https://proceedings.iclr.cc/paper\_files/paper/2025/hash/ 0d70af566e69f1dfb687791ecf955e28-Abstract-Conference.html.

[23] Jun Shern Chan, Neil Chowdhury, Oliver Jaffe, James Aung, Dane Sherburn, Evan Mays, Giulio Starace, Kevin Liu, Leon Maksin, Tejal Patwardhan, Aleksander M ˛adry, and Lilian Weng. MLE-bench: Evaluating machine learning agents on machine learning engineering. In Proceedings of the 13th International Conference on Learning Representations (ICLR), pages 50466–50494, 2025. URL https://proceedings.iclr.cc/paper\_files/paper/2025/hash/ 7e3767db483c942b883eb4f8cfb74e31-Abstract-Conference.html.

[24] Malte D Luecken, Scott Gigante, Daniel B Burkhardt, Robrecht Cannoodt, Daniel C Strobl, Nikolay S Markov, Luke Zappia, Giovanni Palla, Wesley Lewis, Daniel Dimitrov, et al. Defining and benchmarking open problems in single-cell analysis. Nature Biotechnology, 43(7):1035–1040, 2025.

[25] Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374, 2021. URL https://arxiv.org/abs/2107. 03374.

[26] Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, and Charles Sutton. Program synthesis with large language models. arXiv preprint arXiv:2108.07732, 2021. URL https://arxiv.org/abs/2108.07732.

[27] Carlos E. Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, and Karthik Narasimhan. SWE-bench: Can language models resolve real-world GitHub issues? In Proceedings of the 12th International Conference on Learning Representations (ICLR), pages 54107–54157, 2024. URL https://proceedings.iclr.cc/paper\_files/paper/2024/hash/ edac78c3e300629acfe6cbe9ca88fb84-Abstract-Conference.html.

[28] Alexis Palmer, Noah A. Smith, and Arthur Spirling. Using proprietary language models in academic research requires explicit justification. Nature Computational Science, 4(1):2–3, 2024. doi: 10.1038/ s43588-023-00585-1.

[29] Étienne Ollion, Rubing Shen, Ana Macanovic, and Arnault Chatelain. The dangers of using proprietary LLMs for research. Nature Machine Intelligence, 6(1):4–5, 2024. doi: 10.1038/s42256-023-00783-6.

[30] Chandan Singh, Armin Askari, Rich Caruana, and Jianfeng Gao. Augmenting interpretable models with large language models during training. Nature Communications, 14:7913, 2023. doi: 10.1038/ s41467-023-43713-1.

[31] Miles Cranmer, Alvaro Sanchez-Gonzalez, Peter Battaglia, Rui Xu, Kyle Cranmer, David Spergel, and Shirley Ho. Discovering symbolic models from deep learning with inductive biases. In Advances in Neural Information Processing Systems 33 (NeurIPS 2020), volume 33, 2020.

[32] Tianle Cai, Xuezhi Wang, Tengyu Ma, Xinyun Chen, and Denny Zhou. Large language models as tool makers. In The Twelfth International Conference on Learning Representations (ICLR), 2024. URL https://proceedings.iclr.cc/paper\_files/paper/2024/file/ ed91353f700d113e5d848c7e04a858b0-Paper-Conference.pdf.

[33] Giulio Starace, Oliver Jaffe, Dane Sherburn, James Aung, Jun Shern Chan, Leon Maksin, Rachel Dias, Evan Mays, Benjamin Kinsella, Wyatt Thompson, Johannes Heidecke, Amelia Glaese, and Tejal Patwardhan. PaperBench: Evaluating AI’s ability to replicate AI research. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 56843–56873. PMLR, 2025. URL https://proceedings.mlr.press/v267/starace25a.html.

[34] Jiacheng Miao, Joe R. Davis, Yaohui Zhang, Jonathan K. Pritchard, and James Zou. Reimagining research papers as interactive and reliable AI agents. Nature, 2026. doi: 10.1038/ s41586-026-11044-y. URL https://doi.org/10.1038/s41586-026-11044-y. Published online 16 September 2026. Preprint: arXiv:2509.06917.

[35] Yuru Wang, Lejun Cheng, Yuxin Zuo, Sihang Zeng, Bingxiang He, Che Jiang, Junlin Yang, Yuchong Wang, Kaikai Zhao, Weifeng Huang, Kai Tian, Zhenzhao Yuan, Jincheng Zhong, Weizhi Wang, Ning Ding, Bowen Zhou, and Kaiyan Zhang. NatureBench: Can coding agents match the published SOTA of Nature-family papers? arXiv preprint arXiv:2606.24530, 2026. URL https://arxiv.org/abs/2606. 24530.

[36] Kenny Workman, Zhen Yang, Harihara Muralidharan, and Hannah Le. SpatialBench: Can agents analyze real-world spatial biology data? arXiv preprint arXiv:2512.21907, 2025. URL https://arxiv. org/abs/2512.21907.

[37] Patrik L. Ståhl, Fredrik Salmén, Sanja Vickovic, Anna Lundmark, José Fernández Navarro, Jens Magnusson, Stefania Giacomello, Michaela Asp, Jakub O. Westholm, Mikael Huss, Annelie Mollbrink, Sten Linnarsson, Simone Codeluppi, Åke Borg, Fredrik Pontén, Paul Igor Costea, Pelin Sahlén, Jan Mulder, Olaf Bergmann, Joakim Lundeberg, and Jonas Frisén. Visualization and analysis of gene expression in tissue sections by spatial transcriptomics. Science, 353(6294):78–82, 2016. doi: 10.1126/science.aaf2403.

[38] Kok Hao Chen, Alistair N. Boettiger, Jeffrey R. Moffitt, Siyuan Wang, and Xiaowei Zhuang. Spatially resolved, highly multiplexed RNA profiling in single cells. Science, 348(6233):aaa6090, 2015. doi: 10.1126/science.aaa6090.

[39] Amanda Janesick, Robert Shelansky, Andrew D. Gottscho, Florian Wagner, Stephen R. Williams, Morgane Rouault, Ghezal Beliakoff, Carolyn A. Morrison, Michelli F. Oliveira, Jordan T. Sicherman, Andrew Kohlway, Jawad Abousoud, Tingsheng Yu Drennon, Seayar H. Mohabbat, 10x Development Teams, and Sarah E. B. Taylor. High resolution mapping of the tumor microenvironment using integrated single-cell, spatial and in situ analysis. Nature Communications, 14(1):8353, 2023. doi: 10.1038/s41467-023-43458-x.

[40] Di Zhang, Yanxiang Deng, Petra Kukanja, Eneritz Agirre, Marek Bartosovic, Mingze Dong, Cong Ma, Sai Ma, Graham Su, Shuozhen Bao, Yang Liu, Yang Xiao, Gorazd B. Rosoklija, Andrew J.

Dwork, J. John Mann, Kam W. Leong, Maura Boldrini, Liya Wang, Maximilian Haeussler, Benjamin J. Raphael, Yuval Kluger, Gonçalo Castelo-Branco, and Rong Fan. Spatial epigenome– transcriptome co-profiling of mammalian tissues. Nature, 616(7955):113–122, 2023. doi: 10.1038/ s41586-023-05795-1.

[41] Mingyue Wang, Qinan Hu, Tianhang Lv, Yuhang Wang, Qing Lan, Rong Xiang, Zhencheng Tu, Yanrong Wei, Kai Han, Chang Shi, Junfu Guo, Chao Liu, Tao Yang, Wensi Du, Yanru An, Mengnan Cheng, Jiangshan Xu, Haorong Lu, Wangsheng Li, Shaofang Zhang, Ao Chen, Wei Chen, Yuxiang Li, Xiaoshan Wang, Xun Xu, Yuhui Hu, and Longqi Liu. High-resolution 3D spatiotemporal transcriptomic maps of developing Drosophila embryos and larvae. Developmental Cell, 57(10):1271–1283.e4, 2022. doi: 10.1016/j.devcel.2022.04.006.

[42] Mark Jenkinson and Stephen Smith. A global optimisation method for robust affine registration of brain images. Medical Image Analysis, 5(2):143–156, 2001. doi: 10.1016/S1361-8415(01)00036-6.

[43] Brian B. Avants, Charles L. Epstein, Murray Grossman, and James C. Gee. Symmetric diffeomorphic image registration with cross-correlation: evaluating automated labeling of elderly and neurodegenerative brain. Medical Image Analysis, 12(1):26–41, 2008. doi: 10.1016/j.media.2007.06.004.

[44] Aristeidis Sotiras, Christos Davatzikos, and Nikos Paragios. Deformable medical image registration: a survey. IEEE Transactions on Medical Imaging, 32(7):1153–1190, 2013. doi: 10.1109/TMI.2013. 2265603.

[45] Ron Zeira, Max Land, Alexander Strzalkowski, and Benjamin J. Raphael. Alignment and integration of spatial transcriptomics data. Nature Methods, 19(5):567–575, 2022. doi: 10.1038/ s41592-022-01459-6.

[46] Xinhao Liu, Ron Zeira, and Benjamin J. Raphael. Partial alignment of multislice spatially resolved transcriptomics data. Genome Research, 33(7):1124–1132, 2023. doi: 10.1101/gr.277670.123.

[47] Andrew Jones, F. William Townes, Didong Li, and Barbara E. Engelhardt. Alignment of spatial genomics data using deep Gaussian processes. Nature Methods, 20(9):1379–1387, 2023. doi: 10.1038/ s41592-023-01972-2.

[48] Chen-Rui Xia, Zhi-Jie Cao, Xin-Ming Tu, and Ge Gao. Spatial-linked alignment tool (SLAT) for aligning heterogenous slices. Nature Communications, 14(1):7236, 2023. doi: 10.1038/s41467-023-43105-5.

[49] Kalen Clifton, Manjari Anant, Gohta Aihara, Lyla Atta, Osagie K. Aimiuwu, Justus M. Kebschull, Michael I. Miller, Daniel Tward, and Jean Fan. STalign: Alignment of spatial transcriptomics data using diffeomorphic metric mapping. Nature Communications, 14(1):8123, 2023. doi: 10.1038/s41467-023-43915-7.

[50] Xiang Zhou, Kangning Dong, and Shihua Zhang. Integrating spatial transcriptomics data across different conditions, technologies and developmental stages. Nature Computational Science, 3(10): 894–906, 2023. doi: 10.1038/s43588-023-00528-w.

[51] Hao Xu, Shuyan Wang, Minghao Fang, Songwen Luo, Chunpeng Chen, Siyuan Wan, Rirui Wang, Meifang Tang, Tian Xue, Bin Li, Jun Lin, and Kun Qu. SPACEL: deep learning-based characterization of spatial transcriptome architectures. Nature Communications, 14(1):7603, 2023. doi: 10.1038/s41467-023-43220-3.

[52] Xiaojie Qiu, Daniel Y. Zhu, Yifan Lu, Jiajun Yao, Zehua Jing, Kyung Hoi Min, Mengnan Cheng, Hailin Pan, Lulu Zuo, Samuel King, Qi Fang, Huiwen Zheng, Mingyue Wang, Shuai Wang, Qingquan Zhang, Sichao Yu, Sha Liao, Chao Liu, Xinchao Wu, Yiwei Lai, Shijie Hao, Zhewei Zhang, Liang Wu, Yong Zhang, Mei Li, Zhencheng Tu, Jinpei Lin, Zhuoxuan Yang, Yuxiang Li, Ying Gu, David Ellison,

Yuancheng Ryan Lu, Qinan Hu, Yuhui Hu, Ao Chen, Longqi Liu, Jonathan S. Weissman, Jiayi Ma, Xun Xu, Shiping Liu, and Yinqi Bai. Spatiotemporal modeling of molecular holograms. Cell, 187(26): 7351–7373.e61, 2024. doi: 10.1016/j.cell.2024.10.011.

[53] Zefang Tang, Shuchen Luo, Hu Zeng, Jiahao Huang, Xin Sui, Morgan Wu, and Xiao Wang. Search and match across spatial omics samples at single-cell resolution. Nature Methods, 21(10):1818–1829, 2024. doi: 10.1038/s41592-024-02410-7.

[54] Haoyang Li, Yingxin Lin, Wenjia He, Wenkai Han, Xiaopeng Xu, Chencheng Xu, Elva Gao, Hongyu Zhao, and Xin Gao. SANTO: a coarse-to-fine alignment and stitching method for spatial omics. Nature Communications, 15(1):6048, 2024. doi: 10.1038/s41467-024-50308-x.

[55] Dominik Klein, Giovanni Palla, Marius Lange, Michal Klein, Zoe Piran, Manuel Gander, Laetitia Meng-Papaxanthos, Michael Sterr, Lama Saber, Changying Jing, Aimée Bastidas-Ponce, Perla Cota, Marta Tarquis-Medina, Shrey Parikh, Ilan Gold, Heiko Lickert, Mostafa Bakhti, Mor Nitzan, Marco Cuturi, and Fabian J. Theis. Mapping cells through time and space with moscot. Nature, 638(8052): 1065–1075, 2025. doi: 10.1038/s41586-024-08453-2.

[56] Yuanyuan Yu and Zhi Xie. Spatial transcriptomic alignment, integration, and 3D reconstruction by STAIR. Genome Biology, 26(1):427, 2025. doi: 10.1186/s13059-025-03895-x.

[57] Jan Matthias, Jianing Yao, and Stephanie C. Hicks. ALBIS: simulation of multi-resolution and multidimensional spatial transcriptomics data. bioRxiv, 2026. doi: 10.64898/2026.10.06.757026. URL https://www.biorxiv.org/content/10.64898/2026.10.06.757026v1.

[58] Gregory M. Kurtzer, Vanessa Sochat, and Michael W. Bauer. Singularity: Scientific containers for mobility of compute. PLOS ONE, 12(5):e0177459, 2017. doi: 10.1371/journal.pone.0177459.

[59] Marie Schott, Daniel León-Periñán, Elena Splendiani, Leon Strenger, Jan Robin Licha, Tancredi Massimo Pentimalli, Simon Schallenberg, Jonathan Alles, Sarah Samut Tagliaferro, Anastasiya Boltengagen, Sebastian Ehrig, Stefano Abbiati, Steffen Dommerich, Massimiliano Pagani, Elisabetta Ferretti, Giuseppe Macino, Nikos Karaiskos, and Nikolaus Rajewsky. Open-ST: high-resolution spatial transcriptomics in 3D. Cell, 187(15):3953–3972.e26, 2024. doi: 10.1016/j.cell.2024.05.055.

[60] Hu Zeng, Jiahao Huang, Jingyi Ren, Connie Kangni Wang, Zefang Tang, Haowen Zhou, Yiming Zhou, Hailing Shi, Abhishek Aditham, Xin Sui, Hongyu Chen, Jennifer A. Lo, and Xiao Wang. Spatially resolved single-cell translatomics at molecular resolution. Science, 380(6652):eadd3067, 2023. doi: 10.1126/science.add3067.

[61] Hu Zeng, Jiahao Huang, Haowen Zhou, William J. Meilandt, Borislav Dejanovic, Yiming Zhou, Christopher J. Bohlen, Seung-Hye Lee, Jingyi Ren, Albert Liu, Zefang Tang, Hao Sheng, Jia Liu, Morgan Sheng, and Xiao Wang. Integrative in situ mapping of single-cell transcriptional states and tissue histopathology in a mouse model of Alzheimer’s disease. Nature Neuroscience, 26(3):430–446, 2023. doi: 10.1038/s41593-022-01251-x.

[62] Jeffrey R. Moffitt, Dhananjay Bambah-Mukku, Stephen W. Eichhorn, Eric Vaughn, Karthik Shekhar, Julio D. Perez, Nimrod D. Rubinstein, Junjie Hao, Aviv Regev, Catherine Dulac, and Xiaowei Zhuang. Molecular, spatial, and functional single-cell profiling of the hypothalamic preoptic region. Science, 362(6416):eaau5324, 2018. doi: 10.1126/science.aau5324.

[63] Andrew L. Ji, Adam J. Rubin, Kim Thrane, Sizun Jiang, David L. Reynolds, Robin M. Meyers, Mar garet G. Guo, Benson M. George, Annelie Mollbrink, Joseph Bergenstråhle, Ludvig Larsson, Yunhao Bai, Bokai Zhu, Aparna Bhaduri, Jordan M. Meyers, Xavier Rovira-Clavé, S. Tyler Hollmig, Sumaira Z. Aasi, Garry P. Nolan, Joakim Lundeberg, and Paul A. Khavari. Multimodal analysis of

composition and spatial architecture in human squamous cell carcinoma. Cell, 182(2):497–514.e22, 2020. doi: 10.1016/j.cell.2020.05.039.

[64] Yu-Chi Sun, Xiaoyin Chen, Stephan Fischer, Shaina Lu, Huiqing Zhan, Jesse Gillis, and Anthony M. Zador. Integrating barcoded neuroanatomy with spatial transcriptional profiling enables identification of gene correlates of projections. Nature Neuroscience, 24(6):873–885, 2021. doi: 10.1038/s41593-021-00842-4.

[65] Xiao Wang, William E. Allen, Matthew A. Wright, Emily L. Sylwestrak, Nikolay Samusik, Sam Vesuna, Kathryn Evans, Cindy Liu, Charu Ramakrishnan, Jia Liu, Garry P. Nolan, Felice-Alessio Bava, and Karl Deisseroth. Three-dimensional intact-tissue sequencing of single-cell transcriptional states. Science, 361(6400):eaat5691, 2018. doi: 10.1126/science.aat5691.

[66] Meng Zhang, Stephen W. Eichhorn, Brian Zingg, Zizhen Yao, Kaelan Cotter, Hongkui Zeng, Hongwei Dong, and Xiaowei Zhuang. Spatially resolved cell atlas of the mouse primary motor cortex by MERFISH. Nature, 598(7879):137–143, 2021. doi: 10.1038/s41586-021-03705-x.

[67] Ao Chen, Sha Liao, Mengnan Cheng, Kailong Ma, Liang Wu, Yiwei Lai, Xiaojie Qiu, Jin Yang, Jiangshan Xu, Shijie Hao, Xin Wang, Huifang Lu, Xi Chen, Xing Liu, Xin Huang, Zhao Li, Yan Hong, Yujia Jiang, Jian Peng, Shuai Liu, Mengzhe Shen, Chuanyu Liu, Quanshui Li, Yue Yuan, Xiaoyu Wei, Huiwen Zheng, Weimin Feng, Zhifeng Wang, Yang Liu, Zhaohui Wang, et al. Spatiotemporal transcriptomic atlas of mouse organogenesis using DNA nanoball-patterned arrays. Cell, 185(10): 1777–1792.e21, 2022. doi: 10.1016/j.cell.2022.04.003.

[68] Robert R. Stickels, Evan Murray, Pawan Kumar, Jilong Li, Jamie L. Marshall, Daniela J. Di Bella, Paola Arlotta, Evan Z. Macosko, and Fei Chen. Highly sensitive spatial transcriptomics at near-cellular resolution with Slide-seqV2. Nature Biotechnology, 39(3):313–319, 2021. doi: 10.1038/s41587-020-0739-1.

[69] Michelli Faria de Oliveira, Juan Pablo Romero, Meii Chung, Stephen R. Williams, Andrew D. Gottscho, Anushka Gupta, Susan E. Pilipauskas, Seayar Mohabbat, Nandhini Raman, David J. Sukovich, David M. Patterson, Visium HD Development Team, and Sarah E. B. Taylor. Highdefinition spatial transcriptomic profiling of immune cell populations in colorectal cancer. Nature Genetics, 57(6):1512–1523, 2025. doi: 10.1038/s41588-025-02193-3.

[70] Hailing Shi, Yichun He, Yiming Zhou, Jiahao Huang, Kamal Maher, Brandon Wang, Zefang Tang, Shuchen Luo, Peng Tan, Morgan Wu, Zuwan Lin, Jingyi Ren, Yaman Thapa, Xin Tang, Ken Y. Chan, Benjamin E. Deverman, Hao Shen, Albert Liu, Jia Liu, and Xiao Wang. Spatial atlas of the mouse central nervous system at molecular resolution. Nature, 622(7983):552–561, 2023. doi: 10.1038/s41586-023-06569-5.

[71] Shanshan He, Ruchir Bhatt, Carl Brown, Emily A. Brown, Derek L. Buhr, Kan Chantranuvatana, Patrick Danaher, Dwayne Dunaway, Ryan G. Garrison, Gary Geiss, Mark T. Gregory, Margaret L. Hoang, Rustem Khafizov, Emily E. Killingbeck, Dae Kim, Tae Kyung Kim, Youngmi Kim, Andrew Klock, Mithra Korukonda, Alecksandr Kutchma, Zachary R. Lewis, Yan Liang, Jeffrey S. Nelson, Giang T. Ong, Evan P. Perillo, Joseph C. Phan, Tien Phan-Everson, Erin Piazza, Tushar Rane, Zachary Reitz, et al. High-plex imaging of RNA and proteins at subcellular resolution in fixed tissue by spatial molecular imaging. Nature Biotechnology, 40(12):1794–1806, 2022. doi: 10.1038/s41587-022-01483-z.

## Supplementary Information

## Contents

• Supplementary Note 1: Task construction details (Supplementary Tables 1, 2 and 3)

• Supplementary Note 2: Published methods rerun under one scorer (Supplementary Figure 1)

• Supplementary Note 3: Agents, harness and prompts (Supplementary Table 4)

• Supplementary Note 4: Leakage audit (Supplementary Figure 2 and Supplementary Table 5)

• Supplementary Note 5: Scoring details

• Supplementary Note 6: Statistical details

• Supplementary Note 7: Trace processing and content (Supplementary Figures 3 and 4)

• Supplementary Note 8: Run-to-run variation and its sources (Supplementary Figures 5 and 6)

Supplementary Figure 1 Evaluation of published alignment packages and geometric baselines under a unified scorer

Supplementary Figure 2 Container isolation layers and automated leakage audit results

Supplementary Figure 3 Algorithmic steps explored and retained across models and their association with alignment scores

Supplementary Figure 4 Quantitative analysis of execution trace volume, embeddings,

self-evaluations, and vocabulary

Supplementary Figure 5 Run-to-run reproducibility across independent replicates

Supplementary Figure 6 Variance decomposition of alignment accuracy, cost, and runtime

Supplementary Table 1 Task families and data sources

Supplementary Table 2 Tissues, spatial technologies, and evaluation metrics per task family

Supplementary Table 3 Simulation parameters for synthetic tissue tasks

Supplementary Table 4 Agent harnesses, models, and launch configurations

Supplementary Table 5 Modes of benchmark leakage observed prior to container isolation

## Supplementary Note 1: Task construction details

From publication to executable task. Paper2Bench compiles a source publication into benchmark tasks across ten automated stages: source ingestion; extraction of experimental task units from figures, tables, methods, code, and datasets; normalization of inputs into typed data contracts; inference or specification of output schemas; attachment of deterministic scoring profiles; generation of blind task prompts; construction of isolated workspaces; automated validation; task prioritization by discriminative power, scientific relevance, and computational cost; and emission of runner metadata. Large language models assist in drafting candidate task extractions, output contracts, prompts, and rankings, while deterministic checks govern acceptance across six lifecycle states (draft, executable, validated, calibrated, sealed, and published).

Automated validators written by AI agents verify schema compliance, file checksums, data licensing and provenance, workspace reproducibility, submission format validation (including negative test fixtures), baseline execution (identity transform, geometric fit, and at least one published alignment method), metric determinism and perturbation sensitivity, absence of leaked metadata, and scorer integrity.

Dataset curation. All tasks are generated programmatically from a declarative registry. Domain-specific curation was performed across task families to ensure exact fidelity to the source publications: matching author notebook scoring modes and marker gene lists (DLPFC); recompiling Bregma slice sets, filtering ambiguous cells, and harmonizing anatomical region labels (MHPR); reconstructing slice assignments (MOp); applying published cell quality filters (whole-brain MERFISH); standardizing input coordinate frames (Drosophila embryo); regenerating expression matrices with per-section centering and updated marker lists (Open-ST); restoring raw count matrices and linking perturbation metadata (MBSA and MBSP); anonymizing cell identifiers and permuting row order (STARmap and BARISTA-seq); and constructing lattice-corrected variants for simulated spot tasks.

Coordinate unit mismatch in the published Drosophila embryo evaluation. Rebuilding the 16-section Drosophila E14–16 embryo Stereo-seq stack (sa-019) [41] from raw SABench inputs uncovered a 20-fold scale discrepancy between the input coordinates and the ground-truth coordinates used in the published evaluation [11]. In SABench, input sections are centered raw Stereo-seq pixel coordinates (raw\_x, raw\_y; spatial span ∼ 600 units), whereas the reference ground-truth positions (new\_x, new\_y) are stored in Flysta3D 3D-atlas units that are exactly 20-fold smaller (scale = 0.05, plus a small residual rigid rotation per section). Because SABench computed mean absolute error (MAE) directly between unscaled output coordinates and 0.05×-scaled ground-truth coordinates (SABench Fig. 5f and Supplementary Table Flysta3D\_MAE), even the stationary reference slice (s0, which no alignment method transforms) incurred an identical floor error of 102.29 across all ten evaluated methods (reproduced to six decimal places by our scorer prior to rescaling).

Consequently, under the unscaled evaluation, nearly all rigid and optimal-transport baselines and agent runs compressed tightly around the unaligned identity error, whereas methods that shrank coordinates toward the origin appeared artificially accurate (moscot\_warp, 39.8; cast\_rescale\_affine, 88.5). Rescaling the ground-truth coordinates by 20× into native input pixel units resolved this artifact: the sta tionary reference slice has an MAE of 0, unaligned inputs have an MAE of 166.3, and alignment methods separate cleanly—led by PASTE (42.3, a 75% error reduction), moscot (48.0, 71%), claude-sonnet-5.5 (55.4, 67%), SPACEL (78.4, 53%), claude-opus-5.5 (118.7, 29%), and gpt-6-astra (122.4, 26%), while shrinkage configurations (cast\_rescale\_affine, 217.3) and over-warped nonrigid runs score worse than unaligned inputs.

Sources and task families. The five task families and their data sources are summarized in Supplementary Table 1, with tissue types, spatial platforms, and available evaluation metrics detailed in Supplementary Table 2. Of the 240 tissue sections in SABench, 236 were reconstructed exactly or near-exactly (100 exact, 136 near-exact, 2 with modified quality filters, and 2 from alternative public releases), covering all 295 evaluated section pairs (260 experimental pairs and 35 synthetic rotation and cropping perturbations).

Four whole-brain MERFISH tasks use raw imaging coordinates registered to the Allen Common Coordinate Framework (CCF) by the original authors; each section shows an isotropic scaling factor of 1.17 ± 0.05 and about 250 µm of local nonlinear deformation, making them the most challenging geometric reconstructions in the benchmark. Ground-truth coordinates derive from simulation (12 tasks), known synthetic transformations of experimental sections (11), author-curated registrations (5), and anatomical atlas coordinates (4). All underlying datasets are publicly available: SABench data (Zenodo 10.5281/zenodo.18605715), GEASO data (Zenodo 18811760), Open-ST (GEO GSE251926) [59], spatia CUT&Tag-RNA (GEO GSE205055) [40], RIBOmap (Single Cell Portal SCP1835) [60], STARmap PLUS (SCP1375) [61], and the Albis simulator [57] (MIT license).

Ground-truth coordinates are available for the three-dimensional SABench stacks, all simulated tasks, and most GEASO tasks (32 tasks total); cell-type or regional labels are available for annotated SABench tasks and two cross-modality pairs (69 tasks); and all remaining tasks are evaluated via gene expression agreement.

Simulated tissues. The 12 synthetic tasks were generated using the Albis simulator across 6 spatial domains, 8 cell types, and 10 serial slabs per tissue (Supplementary Table 3). Each section undergoes a random rigid transformation with rotations up to ±270<sup>◦</sup> and translations up to half the tissue radius.

Supplementary Table 1: Task families and sources. Abbreviations: DLPFC, dorsolateral prefrontal cortex; BCA, breast cancer; MBSA and MBSP, mouse brain sagittal anterior and posterior; AMBC, adult mouse brain coronal; MHPR, mouse hypothalamic preoptic region; SCC, squamous cell carcinoma; MOp, primary motor cortex; MOB, mouse olfactory bulb; WMB, whole mouse brain.
<table><tr><td>Family</td><td>Tasks</td><td>Source</td><td>Content</td></tr><tr><td>Serial sections (sa-0xx)</td><td>35</td><td>SABench [11]</td><td>DLPFC Visium [12], BCA, MBSA and MBSP, AMBC Xe- nium [39], MHPR MERFISH [62], SCC [63], BARISTA- seq [64], STARmap [65], fly embryo (16 sections) [41], MOp MERFISH (31 and 33 sections) [66], Open-ST lymph node (19 sections) [59], WMB MERFISH (129 sections and win-</td></tr><tr><td>Motor cortex series (sa- 1xx)</td><td>12</td><td>SABench [11]</td><td>seqV2 [68], and four rotation and crop sweeps MOp MERFISH, one task per sample of two mice, 4 to 6 sections each</td></tr><tr><td>Cross-platform pairs (sa-2xx)</td><td>45</td><td>SABench [11]</td><td>All pairs of ten platforms on adult mouse brain sections: ST [37], Visium, Visium HD [69], Xenium, Xenium 5K, MER- FISH [38], STARmap PLUS [70], Slide-seq, Stereo-seq and CosMx [71]</td></tr><tr><td>Simulated tissue (sa- 3xx)</td><td>12</td><td>Albis simula- tor [57]</td><td>Simulated serial sections with known positions (Supple- mentary Table 3) STARmap PLUS depth, overlap and stitching [70];</td></tr><tr><td>Rotation, crop, 3D (sa- 4xx)</td><td>17</td><td>GEASO data [14], WMB MERFISH</td><td>STARmap and RIBOmap cross-modality [60]; cross-animal and rotation sweep [61]; spatial CUT&amp;Tag-RNA [40]; 129-section MERFISH hemibrain; Open-ST reconstruction; whole-brain MERFISH from raw coordinates (129, 33, 17</td></tr></table>

Section-specific batch effects apply a gene-wise log-fold perturbation drawn with standard deviation $\sigma ,$ followed by Poisson count resampling. Ground-truth positions are defined in the coordinate frame of the first section, with observation rows randomly permuted and cell identifiers anonymized.

Validation on these synthetic datasets revealed two notable failure modes of proxy alignment metrics. First, grid-based expression correlations exhibit a near-zero upper bound under high Poisson sparsity (split-half correlation of 0.037 on a 10 × 10 spatial grid). Second, nearest-neighbor label metrics can favor imprecise alignments: a coarse centroid fit with a 371 µm coordinate error achieved a label matching accuracy of 0.837, exceeding the score of the exact ground-truth registration (0.815). Coordinate error is therefore used as the primary evaluation metric across all simulated tasks.

Task package structure. Each task package consists of a machine-readable task card and naturallanguage task instructions. The task card specifies the ordered input sections, section-pair evaluation topology (sequential adjacent pairs or star-graph comparisons against a reference section), withheld annotation columns, anatomical regions for alpha-shape overlap, marker gene sets (published marker lists of 117 genes for DLPFC and 155 for MHPR, or up to 100 top spatially variable markers otherwise), and the scoring profile.

Agents are instructed to output one CSV file per section containing columns obs\_id, $\mathbf { x , }$ and y and submissions are validated to ensure exact observation ID matching, row ordering, and finite numerical values.

## Supplementary Note 2: Published methods rerun under one scorer

With Claude Code (Opus 5.5 with high effort), we re-executed 13 published alignment packages across 41 parameter configurations alongside three geometric baselines on the exact task inputs and scoring harness used for agent evaluation, taking the top-performing configuration of each package on a given task as its representative score (Supplementary Figure 1). Package versions were paste-bio 1.4.0 (POT 0.9.7), paste2 1.0.1, spateo-release 1.1.1, SPACEL 1.1.8, GEASO (commit 2381513), CAST 0.4 (commit 3ebcf4f), STAligner 1.0.0 (commit 85ca4ce), scSLAT 0.3.0, STalign 1.0 (commit b2068ed), STAIR-tools 1.3.1, SANTO 0.0.3, GPSA 0.6.15, and moscot 0.5.2 (ott-jax 0.6.0). Evaluated configurations varied transformation families (rigid versus nonrigid), reflection search, coordinate rescaling, initialization schemes, and feature selection; 26 core configurations were executed across at least 104 tasks. Two SPACEL configurations that required ground-truth region annotations and a fourth geometric baseline that fit transformations to ground-truth cell-type centroids were excluded from baseline comparisons. Coverage was nearly complete across families (Supplementary Figure 1a).

Supplementary Table 2: Tissues, technologies and available scores per family. For each family, the tissues and technologies it covers and how many tasks can be scored with true positions, with labels but no true positions, or with expression agreement only. The serial-sections family also holds some cross-platform pairs and threedimensional stacks from SABench.
<table><tr><td>Family</td><td>Tasks</td><td>Tissues</td><td>Technologies</td><td>True pos.</td><td>Labels only</td><td>Expr. only</td><td>Source</td><td></td></tr><tr><td>Serial sections</td><td>35</td><td>fly embryo, human DLPFC, human breast cancer, human lymph node, human squa- mous cell carcinoma, mouse brain, mouse cortex, mouse hy-</td><td>PLUS, Stereo-seq,</td><td>BARISTA-seq, MER- FISH, Open-ST, ST, STARmap, STARmap Slide-seq, Visium, Visium HD, Xenium,</td><td>5</td><td>17</td><td>13</td><td>SABench</td></tr><tr><td>Motor cortex series</td><td>12</td><td colspan="2">mouse brain mouse motor cortex</td><td>MERFISH CosMx,1 MERFISH,</td><td>0</td><td>12</td><td>0</td><td>SABench</td><td>SABench</td></tr><tr><td>Cross-platform pairs</td><td></td><td colspan="2"></td><td>ST, STARmap PLUS, Slide-seq, Stereo-seq, Visium, Visium HD,</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Simulated tissue</td><td>12</td><td colspan="2">simulated tissue</td><td>Xenium, Xenium 5K simulated Visium, simulated HD, simulated Xe-</td><td>12 Visium</td><td>0</td><td></td><td>0</td><td>simulation (Albis)</td></tr><tr><td>Rotation, crop, 3D</td><td>17</td><td></td><td>human lymph node, mouse brain, mouse cortex, mouse hemi- brain, whole mouse</td><td>nium, simulated cells MERFISH, Open-ST, RIBOmap, STÁRmap, STARmap PLUS, spa- tial CUT&amp;Tag</td><td>15</td><td></td><td></td><td>0</td><td>GEASO, SABench</td></tr><tr><td>All</td><td>121</td><td colspan="6">brain</td><td></td><td></td><td></td></tr></table>

Supplementary Table 3: Simulated tasks. R, tissue radius; σ, standard deviation of the batch effect.
<table><tr><td>Task</td><td>Regime</td><td>Key parameters</td></tr><tr><td>sa-301</td><td>strong mixing, cells</td><td>24,207 cells,  $R = 2 { , } 0 5 0$  μm, σ = 1.5</td></tr><tr><td>sa-302</td><td>Xenium-like cells</td><td>600,000 cells,  $R = 6 { , } 0 0 0$  μm,  $\sigma = 1 . 5$ </td></tr><tr><td>sa-303</td><td>Visium spots</td><td>55 μm spots on a 100 μm lattice,  $\sigma = 0 . 3$ </td></tr><tr><td>sa-304, sa-305</td><td>Visium HD bins</td><td>16 μm  $( \sigma = 0 . 7 )$  and 8 μm  $( \sigma = 0 . 8 )$ </td></tr><tr><td>sa-306, sa-307</td><td>as sa-303 and sa-304, lattice fixed to the slide</td><td>removes the lattice leak</td></tr><tr><td>sa-308</td><td>cells with gaps</td><td>30 slabs of  $1 3 7 \mu \mathrm { m } , 9$  kept at irregular spacing</td></tr><tr><td>sa-309, sa-310</td><td>sparse Visium, without and with gaps</td><td>120,000 cells; spots with no counts or fewer than 3 genes dropped</td></tr><tr><td>sa-311</td><td>Visium HD 32 μm, low signal</td><td> $\sigma = 1 . 2 ,$  weak landmarks</td></tr><tr><td>sa-312</td><td>decoy flip</td><td>mirror-image decoy domains; half of the non- reference sections mirrored</td></tr></table>

Across all 16 reference methods, CAST achieved the highest mean within-task percentile (75th) and SLAT ranked second (74th), with leading methods varying by task family: SLAT led on serial sections and on rotation, cropping, and 3D reconstruction tasks, whereas STalign led on simulated tissues (Supplementary Figure 1b). Simple geometric baselines proved competitive against several specialized tools: ICP registration achieved a mean within-task percentile of 49 (outperforming six published packages) and PCA similarity alignment achieved 37 (outperforming two).

CAST ranked as the single best published method on 51 of 120 tasks (including 37 cross-platform pairs), while geometric baselines matched or exceeded all published packages on 4 tasks (Supplementary Figure 1c). Stratified by individual metrics, SLAT led on all coordinate- and label-based scores, whereas CAST led on all gene expression similarity scores (Supplementary Figure 1d). Hyperparameter configuration substantially affected baseline performance: the best and worst configurations of SLAT, moscot, and CAST differed by a median of 24 to 48 percentile points within a task. Across individual evaluation metrics, the best of the six basic-prompt agent runs matched or outperformed the top published configuration on 44% (region overlap) to 62% of tasks (Supplementary Figure 1e).

Comparison with originally reported benchmark values. We compared our re-evaluated baseline scores against published source tables across 312 method-by-task entries from GEASO (1,567 source records covering rotation sweeps, stitching tasks, 129-section MERFISH reconstruction, and 19-section Open-ST reconstruction, converted at 0.0946 µm per pixel) and SABench (nine tasks with method-level tables). Defining numerical reproduction as agreement within 0.02 (close: 0.06) for bounded metrics, 5<sup>◦</sup> (20<sup>◦</sup>) for rotation angles, and 15% (40%) for coordinate error, 63 of 312 comparisons were reproduced closely, 51 were within the broader tolerance, 127 differed, and 71 were not directly comparable. Differences on GEASO tasks primarily reflect GEASO’s post-hoc Procrustes alignment prior to computing Euclidean pixel error, whereas our scorer evaluates direct per-axis coordinate error in native spatial units without post-hoc registration; where directly comparable, competing baseline packages performed at least as well in our reruns as reported in the original study.

## Supplementary Note 3: Agents, harness and prompts

Agent configurations. Agent CLI versions, model identifiers, reasoning effort parameters, and launch commands are listed in Supplementary Table 4. To support claude-opus-5.5 and claude-sonnet-5.5, an updated Claude Code binary was mounted read-only into the container image. OpenAI models were accessed via Codex CLI.

Supplementary Table 4: Agent harnesses, models, and launch configurations. Software versions correspond to the container environment; launch commands are abridged.
<table><tr><td>Agent software</td><td>Model</td><td>Effort</td><td>Launch (abridged)</td></tr><tr><td>Claude Code 2.1.281</td><td>claude-opus-5.5</td><td>high</td><td>claude --print --output-format stream-json --model &lt;m&gt; --dangerously-skip-permissions -p &quot;&lt;prompt&gt;&quot;</td></tr><tr><td></td><td>claude-sonnet-5.5</td><td>default</td><td>as above</td></tr><tr><td>Codex CLI 0.154.0</td><td>gpt-6-astra</td><td>xhigh</td><td>codex exec --json --model &lt;m&gt; -c model_reasoning_effort=xhigh --dangerously-bypass-approvals-and-sandbox &quot;&lt;prompt&gt;&quot;</td></tr><tr><td></td><td>gpt-5.5</td><td>xhigh</td><td>as above</td></tr><tr><td>Gemini CLI 0.60.0</td><td>gemini-4-argon</td><td>none</td><td>gemini --prompt &quot;&lt;prompt&gt;&quot; --output-format stream-json --model</td></tr><tr><td></td><td>gemini-3.5-flash</td><td>none</td><td>&lt;m&gt; --yolo as above</td></tr></table>

Prompting conditions. For representative task sa-001, the basic prompt comprises 4,056 characters and the rich-context prompt 21,170 characters (full texts provided below). Under the rich-context condition, isolated virtual environments were pre-configured for 11 packages: PASTE, PASTE2, Spateo, SPACEL,

c  
a  
![](images/de477995fe6e0b75b3a1c840daca139d942a843b9fae2d00489742c7c5f5ea7c.jpg)  
b

![](images/fffabbbbb049ec1cb9916bec76fdad58790ca8726f029688b5650f4dd67b847b.jpg)

![](images/cbe278b4957e158dfe0e40c637e183220d1211ffc597d512405589ffd3ec6dc9.jpg)

d
<table><tr><td rowspan=2 colspan=11>mean within-task percentile among 16 methodsCAST 39  77  55  68  81  77  80  83  85   72</td></tr><tr><td rowspan=1 colspan=1>39</td><td rowspan=1 colspan=1>77</td><td rowspan=1 colspan=1>55</td><td rowspan=1 colspan=1>68</td><td rowspan=1 colspan=1>81</td><td rowspan=1 colspan=1>77</td><td rowspan=1 colspan=1>80</td><td rowspan=1 colspan=1>83</td><td rowspan=1 colspan=1>85</td><td rowspan=1 colspan=1>72</td></tr><tr><td rowspan=1 colspan=1>SLAT</td><td rowspan=1 colspan=1>83</td><td rowspan=1 colspan=1>84</td><td rowspan=1 colspan=1>78</td><td rowspan=1 colspan=1>79</td><td rowspan=1 colspan=1>74</td><td rowspan=1 colspan=1>73</td><td rowspan=1 colspan=1>73</td><td rowspan=1 colspan=1>73</td><td rowspan=1 colspan=1>73</td><td rowspan=1 colspan=1>77</td></tr><tr><td rowspan=1 colspan=1>Moscot</td><td rowspan=1 colspan=1>59</td><td rowspan=1 colspan=1>52</td><td rowspan=1 colspan=1>69</td><td rowspan=1 colspan=1>52</td><td rowspan=1 colspan=1>71</td><td rowspan=1 colspan=1>73</td><td rowspan=1 colspan=1>71</td><td rowspan=1 colspan=1>67</td><td rowspan=1 colspan=1>65</td><td rowspan=1 colspan=1>64</td></tr><tr><td rowspan=3 colspan=1>SpateoGEASOSPACEL</td><td rowspan=1 colspan=1>65</td><td rowspan=1 colspan=1>80</td><td rowspan=1 colspan=1>76</td><td rowspan=1 colspan=1>77</td><td rowspan=1 colspan=1>68</td><td rowspan=1 colspan=1>67</td><td rowspan=1 colspan=1>68</td><td rowspan=1 colspan=1>66</td><td rowspan=1 colspan=1>72</td><td rowspan=1 colspan=1>71</td></tr><tr><td rowspan=1 colspan=1>59</td><td rowspan=1 colspan=1>79</td><td rowspan=1 colspan=1>63</td><td rowspan=1 colspan=1>72</td><td rowspan=1 colspan=1>67</td><td rowspan=1 colspan=1>65</td><td rowspan=1 colspan=1>67</td><td rowspan=1 colspan=1>68</td><td rowspan=1 colspan=1>70</td><td rowspan=1 colspan=1>68</td></tr><tr><td rowspan=1 colspan=1>59</td><td rowspan=1 colspan=1>78</td><td rowspan=1 colspan=1>63</td><td rowspan=1 colspan=1>72</td><td rowspan=1 colspan=1>65</td><td rowspan=1 colspan=1>60</td><td rowspan=1 colspan=1>66</td><td rowspan=1 colspan=1>64</td><td rowspan=1 colspan=1>65</td><td rowspan=1 colspan=1>66</td></tr><tr><td rowspan=1 colspan=1>STalign</td><td rowspan=1 colspan=1>61</td><td rowspan=1 colspan=1>55</td><td rowspan=1 colspan=1>75</td><td rowspan=1 colspan=1>60</td><td rowspan=1 colspan=1>53</td><td rowspan=1 colspan=1>50</td><td rowspan=1 colspan=1>52</td><td rowspan=1 colspan=1>51</td><td rowspan=1 colspan=1>51</td><td rowspan=1 colspan=1>56</td></tr><tr><td rowspan=1 colspan=1>STAIR</td><td rowspan=1 colspan=1>66</td><td rowspan=1 colspan=1>52</td><td rowspan=1 colspan=1>51</td><td rowspan=1 colspan=1>54</td><td rowspan=1 colspan=1>42</td><td rowspan=1 colspan=1>40</td><td rowspan=1 colspan=1>42</td><td rowspan=1 colspan=1>41</td><td rowspan=1 colspan=1>43</td><td rowspan=1 colspan=1>48</td></tr><tr><td rowspan=1 colspan=1>STAligner</td><td rowspan=1 colspan=1>56</td><td rowspan=1 colspan=1>51</td><td rowspan=1 colspan=1>43</td><td rowspan=1 colspan=1>49</td><td rowspan=1 colspan=1>45</td><td rowspan=1 colspan=1>41</td><td rowspan=1 colspan=1>45</td><td rowspan=1 colspan=1>41</td><td rowspan=1 colspan=1>43</td><td rowspan=1 colspan=1>46</td></tr><tr><td rowspan=1 colspan=1>PASTE2</td><td rowspan=1 colspan=1>50</td><td rowspan=1 colspan=1>54</td><td rowspan=1 colspan=1>61</td><td rowspan=1 colspan=1>52</td><td rowspan=1 colspan=1>44</td><td rowspan=1 colspan=1>44</td><td rowspan=1 colspan=1>44</td><td rowspan=1 colspan=1>44</td><td rowspan=1 colspan=1>44</td><td rowspan=1 colspan=1>48</td></tr><tr><td rowspan=1 colspan=1>GPSA</td><td rowspan=1 colspan=1>44</td><td rowspan=1 colspan=1>18</td><td rowspan=1 colspan=1>17</td><td rowspan=1 colspan=1>25</td><td rowspan=1 colspan=1>33</td><td rowspan=1 colspan=1>44</td><td rowspan=1 colspan=1>33</td><td rowspan=1 colspan=1>38</td><td rowspan=1 colspan=1>35</td><td rowspan=1 colspan=1>32</td></tr><tr><td rowspan=1 colspan=1>PASTE</td><td rowspan=1 colspan=1>38</td><td rowspan=1 colspan=1>29</td><td rowspan=1 colspan=1>37</td><td rowspan=1 colspan=1>28</td><td rowspan=1 colspan=1>32</td><td rowspan=1 colspan=1>31</td><td rowspan=1 colspan=1>32</td><td rowspan=1 colspan=1>31</td><td rowspan=1 colspan=1>30</td><td rowspan=1 colspan=1>32</td></tr><tr><td rowspan=1 colspan=1>SANTO</td><td rowspan=1 colspan=1>19</td><td rowspan=1 colspan=1>23</td><td rowspan=1 colspan=1>18</td><td rowspan=1 colspan=1>30</td><td rowspan=1 colspan=1>30</td><td rowspan=1 colspan=1>25</td><td rowspan=1 colspan=1>30</td><td rowspan=1 colspan=1>30</td><td rowspan=1 colspan=1>34</td><td rowspan=1 colspan=1>27</td></tr><tr><td rowspan=1 colspan=1>ICP fit</td><td rowspan=1 colspan=1>41</td><td rowspan=1 colspan=1>34</td><td rowspan=1 colspan=1>51</td><td rowspan=1 colspan=1>29</td><td rowspan=1 colspan=1>46</td><td rowspan=1 colspan=1>52</td><td rowspan=1 colspan=1>46</td><td rowspan=1 colspan=1>49</td><td rowspan=1 colspan=1>38</td><td rowspan=1 colspan=1>43</td></tr><tr><td rowspan=1 colspan=1>PCA fit</td><td rowspan=1 colspan=1>37</td><td rowspan=1 colspan=1>18</td><td rowspan=1 colspan=1>26</td><td rowspan=1 colspan=1>20</td><td rowspan=1 colspan=1>31</td><td rowspan=1 colspan=1>43</td><td rowspan=1 colspan=1>32</td><td rowspan=1 colspan=1>37</td><td rowspan=1 colspan=1>28</td><td rowspan=1 colspan=1>30</td></tr><tr><td rowspan=1 colspan=1>No alignment</td><td rowspan=1 colspan=1>25</td><td rowspan=1 colspan=1>15</td><td rowspan=1 colspan=1>13</td><td rowspan=1 colspan=1>33</td><td rowspan=1 colspan=1>17</td><td rowspan=1 colspan=1>14</td><td rowspan=1 colspan=1>17</td><td rowspan=1 colspan=1>15</td><td rowspan=1 colspan=1>23</td><td rowspan=1 colspan=1>19</td></tr><tr><td rowspan=2 colspan=1>Centroid fit</td><td rowspan=2 colspan=2>Cor eror   Matg Maihic.</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=2 colspan=5>Genese csine   Gen SIM   GenGMI   MNxN pP.   Mean</td></tr><tr><td rowspan=1 colspan=1>Regern ovorrap</td><td rowspan=1 colspan=2>MN bel   Gene cor.</td></tr></table>

e

![](images/dafb4e21b0b9416d7507c3926accd9cdd685751ed683361f6671fc1086a8cf11.jpg)  
Supplementary Figure 1: Evaluation of published alignment packages and geometric baselines under a unified scorer. $\mathbf { a } ,$ Task coverage across families and number of parameter configurations evaluated per method. $\mathbf { b , }$ Mean within-task percentile across the 16 reference baselines on the primary metric, stratified by task family. $\mathbf { c } ,$ Number of tasks on which each published package achieved the highest baseline score, or on which a geometric baseline matched or outperformed all packages. d, Mean within-task percentile stratified by individual evaluation metric. MNN, mutual nearest neighbors; SSIM, structural similarity; MI, mutual information. e, Percentage of tasks on which the best basic-prompt agent run matched or outperformed the best published configuration across individual metrics. PCA, principal component analysis; ICP, iterative closest point.

STAIR, GPSA, STalign, SLAT, CAST, STAligner, and SANTO (excluding GEASO and moscot). The richcontext prompt also included a modality-specific recommendation table adapted from SABench (for example, prioritizing SPACEL and PASTE2 for Visium serial sections). Across the six models evaluated under both prompts, the change in within-task percentile for gemini-3.5-flash yielded P = 0.22 after Holm adjustment, while gains for claude-opus-5.5 were concentrated in a subset of datasets. On tasks with ground-truth positions, the geometric mean ratio of coordinate error (rich context relative to basic prompt) was 0.64 for claude-opus-5.5 and 0.74 for claude-sonnet-5.5, compared with 1.59 for gpt-6-astra, 1.79 for gpt-5.5, 1.89 for gemini-3.5-flash, and 3.86 for gemini-4-argon (95% CI: 0.54–130). Rich context also increased tool-call volume (for example, from a median of 17 to 42 calls for claude-opus-5.5) and reduced the use of custom numerical optimization in gpt-6-astra by 28 percentage points.

Container harness. All evaluations were executed under Singularity 3.11.4 (--containall --cleanenv --no-home) using a base image built on python:3.11-slim with Node.js 22, uv, and the three vendor CLI tools. Only the task workspace and an ephemeral state directory were writable; software mounts and API credentials were read-only, and environment variables containing keys were scrubbed prior to agent startup. Standard tasks were allocated 4 CPU cores, 48 GB RAM, a 360-min agent execution limit, and a 60-min scoring window (7 h total wall time); large 3D reconstruction tasks were allocated 4 CPU cores, 96 GB RAM, a 720-min agent limit, and a 360-min scoring window (20 h total wall time), with no GPU access. Across all scored runs, 95% completed within 71 min (129 min for large 3D tasks), with a maximum runtime of 361 min. Upon termination, workspace outputs and execution logs were archived for auditing and the container workspace was purged.

Prompts. Verbatim prompt templates recorded by the execution harness are reproduced below: the basic prompt (illustrated for task sa-001), the rich-context package documentation and modality recommendation tables, and the strategy-distillation prompt provided to the synthesizer agent.

Basic prompt (task sa-001, complete; line breaks added).

You are running inside a blind benchmark workspace.

TASK.md:

\# Agent Task: sa-001-dlpfc-serial-align

\# DLPFC Serial Alignment

Align the four 10x Visium DLPFC slices into a common coordinate system.

Submit one coordinate CSV per slice with columns ‘obs\_id,x,y‘. The row order and ‘obs\_id‘ values must match the source ‘.h5ad‘ file.

\## Data Visibility

The H5AD files under ‘data/‘ are sanitized agent-visible inputs.

Allowed inputs:

\- Observation IDs from ‘obs/\_index

\- Gene expression from ‘X‘, ‘var‘, and any provided expression ‘layers‘

\- Original two-dimensional coordinates from ‘obsm/spatial‘

\- Slice identities and required output filenames from this task prompt

Not allowed and not present in the solve workspace:

\- Hidden gold aligned coordinates or 3D atlas coordinates

\- Ground-truth transforms, perturbation metadata, or perturbation configuration files

\- Evaluation landmarks, scorer-only split manifests, or local scorer source

\- Anatomical or scoring labels reserved for evaluation

\- Private benchmark source paths, local download caches, previous run artifacts, or benchmark repository files

Use only the allowed inputs above. Do not infer, search for, or rely on hidden source paths, scorer internals, prior submissions, or withheld metadata.

## ## Data

The input ‘.h5ad‘ files are available in this workspace:

\- ‘151673‘: ‘data/151673.h5ad‘

\- ‘151674‘: ‘data/151674.h5ad‘

\- ‘151675‘: ‘data/151675.h5ad‘

- ‘151676‘: ‘data/151676.h5ad‘

## ## Required Output

## Create coordinate CSV files in:

\- ‘submission/aligned\_coordinates/151673.csv‘

\- ‘submission/aligned\_coordinates/151674.csv‘

\- ‘submission/aligned\_coordinates/151675.csv‘

\- ‘submission/aligned\_coordinates/151676.csv‘

Each CSV must contain exactly these columns:

```csv
‘‘‘text
obs_id,x,y
CC
```

## Rules:

\- ‘obs\_id‘ values and row order must match the source ‘.h5ad‘ observations.

- ‘x‘ and ‘y‘ must be finite numeric aligned coordinates.

\- Do not modify the source data files.

\- You may create code, logs, figures, or a short ‘submission/report.json‘, but scoring uses the coordinate CSV files.

## ## Execution Strategy

First create a simple valid submission for every required output CSV, even if it only uses a conservative baseline transform. Then improve methodically: try richer custom models or refinements only after a complete returnable answer is already on disk. Whenever you find a better alignment, overwrite the same prediction CSVs. Keep the fallback CSVs valid while you explore stronger approaches.

## ## Local Scoring

No local scorer is available in this solve workspace. Final scoring is run by the benchmark harness after your agent process exits.

Do not search parent directories, repository paths, previous runs, benchmark source files, scorer code, Downloads caches, or other locations outside this workspace.

For Python, use the workspace helper:

‘‘‘bash   
./agent\_python.sh your\_script.py   
# or   
./agent\_python.sh -c "import h5py; print(’ok’)"   
ccc

The helper runs ‘uv‘ with common cached analysis packages: ‘numpy‘, ‘scipy‘, ‘pandas‘, ‘scikit-learn‘, and

Method priorities added to the rich-context prompt (verbatim; references/prompt\_assets/spatial\_ alignment\_methods\_paper\_prior.md in the code repository, last changed 2026-09-22). The rich-context prompt also lists the package environments with an API example for each and the documentation menu.

\## Benchmark Context (Paper-Informed Package Priority)

The benchmark paper compared multiple spatial-alignment method families on   
related spatial-transcriptomics tasks. In this prompt variant, you are given   
high-level paper-derived method priorities so you can spend your package-testing   
budget on methods that were strongest in the paper. These are method priors,   
not task labels, not ground-truth transforms, and not local scorer information.   
You still must use only the allowed task inputs: sanitized AnnData/CSV files,   
spot IDs, gene expression, original ‘obsm/spatial‘ x/y coordinates, and slice   
identities from ‘TASK.md‘. Do not search for or use paper result tables, hidden   
aligned coordinates, benchmark source, scorer code, prior submissions, or   
withheld labels.   
First create a complete valid submission for every required output CSV. Then   
try package-backed methods in a paper-informed order. A useful package attempt   
means using the method-specific Python executable from ‘.method-cache/METHOD\_ENVS.md‘,   
running an alignment API on the task data or a representative pair/subset, and   
recording whether it produced usable coordinates. If a high-priority method   
fails to import, has incompatible API/runtime behavior, exceeds a deliberate   
timeout, or looks worse under allowed input-only diagnostics, move on and keep   
the best current submission. Do not treat a quiet import as an immediate   
failure: several Torch/numba-heavy method envs can spend a minute or two in   
first-import/JIT startup. Wait unless the process errors or your bounded timeout   
is genuinely too expensive for the task.   
### Paper-Informed Priority By Task Family   
| Task family / data shape | Try first | Then try | Notes |   
|---|---|---|---|   
| DLPFC-like Visium serial alignment | ‘spacel‘, ‘paste2‘ | ‘cast‘, ‘stair‘, ‘spateo‘ | Direct paper DLPFC   
gene+landmark tables favored ‘SPACEL‘ and ‘PASTE2‘; ‘CAST‘ and ‘STAIR‘ were also strong. |   
| MHPR/MERFISH or image-like serial alignment | ‘spateo‘, ‘spacel‘ | ‘paste2‘, ‘stair‘, ‘slat‘ | Direct paper MHPR   
tables favored ‘spateo‘ rigid/non-rigid and ‘SPACEL‘; ‘PASTE2‘, ‘STAIR‘, and ‘SLAT‘ were competitive. |   
| Generic NGS/Visium/STereo-seq slices | ‘paste2‘, ‘spacel‘, ‘spateo‘ | ‘stair‘, ‘paste‘ | Aggregate NGS rankings   
favored ‘PASTE2‘; ‘SPACEL‘ and ‘spateo‘ were consistently strong. ‘PASTE‘ is a useful faster fallback. |   
| 3D serial / MAE-style stacks | ‘spacel‘ | ‘paste‘, ‘paste2‘, ‘stair‘, ‘spateo‘ | Paper Flysta3D MAE favored   
‘SPACEL‘, then PASTE-family / ‘STAIR‘ / ‘spateo‘ methods. |   
| MLN-like multi-section MAE tasks | ‘spateo‘ rigid | ‘slat‘, ‘spateo‘ non-rigid, ‘stalign‘, ‘spacel‘ | Paper MLN MAE   
favored ‘spateo‘ rigid and ‘SLAT‘. |   
| Unknown or mixed platform | ‘spateo‘, ‘spacel‘, ‘paste2‘ | ‘stair‘, ‘cast‘, ‘slat‘, ‘paste‘ | Start with the robust   
paper winners, then adapt based on data size and import/API success. |   
Do not treat this table as a hard rule. It is a search prior. If the dataset has   
very large slices, use subsets to estimate transforms first. If a method needs   
domain labels that are not present, you may create unsupervised domains from   
allowed expression/coordinate data, but do not invent or recover withheld   
anatomical labels.

## ### Practical Package Guidance

\- Prefer ‘spateo‘ and ‘SPACEL‘ when tissue geometry, non-rigid deformation, or image-like spatial continuity looks important.

\- Prefer ‘PASTE2‘ when partial overlap, NGS/Visium-style slices, or robust optimal-transport matching looks important.

\- Keep ‘PASTE‘ as a simpler/faster fallback, especially for large serial stacks where full package methods are too slow.

\- Use ‘CAST‘, ‘STAIR‘, or ‘SLAT‘ when their APIs are importable and the task looks compatible with embedding or graph matching.

- Treat ‘GPSA‘, ‘STAligner‘, ‘STalign‘, and ‘SANTO‘ as optional method-specific   
attempts rather than first defaults unless their API/docs clearly fit the   
task.

- For slower method families, run a small representative pair/subset first and   
allow enough time for first import/JIT startup before deciding the method is   
infeasible.

\### Required Reporting

In ‘submission/report.json‘ or a local note, record:

\- which paper-prior methods you attempted,

\- which package env executable you used,

\- whether the package produced usable transformed coordinates,

\- why you moved on from any failed high-priority method,

\- which candidate you finally submitted.

Strategy distillation. We partitioned 119 tasks (excluding sa-005 and sa-023) into five folds using a fixed random seed under both a random split (23 or 24 tasks per fold) and a task-family split. For each fold, an isolated training directory contained the final scripts and scores of basic-prompt runs from claude-opus-5.5 and gpt-6-astra on the four training folds, the sanitized training inputs, and the geometric baseline implementations.

The synthesis agent (Claude Code with claude-opus-5.5) executed inside a container with network access and produced a standalone alignment program, a methodological playbook, and synthesis notes (an ablation variant synthesized from training code without training scores was also generated). Each synthesized program was then evaluated on a clean blind workspace for every task in the held-out fold without LLM access.

Synthesizer prompt (verbatim, cell random-train4-f0-scored; the other cells differ only in the numbers of training tasks and runs, and the code-only cells have no scores/ folder).

\# Synthesize a general alignment strategy

```markdown
## The job
Earlier AI agents (claude-opus-5-5 and gpt-6-astra) each solved spatial-transcriptomics slice-alignment tasks
on their own. You have their records for 95 tasks (315 runs). Your job is to **synthesize what
they did into one general strategy** for aligning slices, and to package it so it can be applied to **new
tasks you have not seen**. Those new tasks may come from other tissues, platforms and task designs.
Do not try to beat any one run on any one task. The goal is the strategy that generalizes best: which
approaches the runs converged on, when each one is the right choice, how to recognize that from the inputs,
and how to combine or choose among them (an ensemble or decision rule).
## What you have (and nothing else)
- ‘runs/<task>/<model>__<job>/‘: one directory per prior run: ‘solution/‘ (the code that produced
its final submission, with ‘SOLUTION.md‘) and ‘meta.json‘ (model, task, entry point). The runs’ step-by-step
traces are not included.
- ‘tasks/<task>/‘: each training task’s inputs exactly as the agents saw them (‘TASK.md‘, ‘data/*.h5ad‘,
‘agent_python.sh‘). Read-only: to try something on a task, copy the task directory into ‘scratch/‘ first.
- ‘reference_aligners/‘: four simple aligners (identity, centroid similarity, PCA similarity, ICP
similarity). ‘python3 reference_aligners/run_reference.py --help‘ shows how to run them on a workspace.
- ‘scores/‘ holds the benchmark score of every run and of every reference aligner on its training task (definitions in
‘scores/SCORES.md‘). Use them to judge which strategies worked where.
There is no ground truth and no scorer here. The network reaches only the model API and the Python package
index (PyPI); nothing else is reachable, so do not try to look things up online.
## What to deliver, under ‘output/
1. ‘output/policy/run.sh‘: the strategy as a program. It is run as ‘bash output/policy/run.sh <workspace>‘,
where ‘<workspace>‘ has the same layout as ‘tasks/<task>/‘ (a ‘TASK.md‘, ‘data/*.h5ad‘, ‘agent_python.sh‘
and an empty ‘submission/aligned_coordinates/‘). It must write the coordinate CSVs that ‘TASK.md‘ asks
for into ‘<workspace>/submission/aligned_coordinates/‘.
- Decide everything from ‘TASK.md‘ and the data. Never look up a task by its name or ID, and never
hard-code a per-task answer: the tasks it will meet are not in your training set.
- Keep the policy self-contained under ‘output/policy/‘: copy in whatever code you reuse from the runs or
from ‘reference_aligners/‘.
- Python: ‘<workspace>/agent_python.sh‘ gives numpy, scipy, pandas, scikit-learn and h5py. If you need other
public packages, list them in ‘output/policy/requirements.txt‘: before ‘run.sh‘ starts they are installed
from PyPI into the environment that ‘<workspace>/agent_python.sh‘ runs. The policy directory itself is
```

read-only when it runs; write intermediates under ‘<workspace>/‘ (for example ‘<workspace>/work/‘). - Budget per task: 8 CPU cores, 64 GB of memory, and the wall time in seconds in the environment variable ‘POLICY\_TIME\_BUDGET\_S‘ (3 hours for most tasks). Plan for the larger inputs: some tasks have more than 100 sections or more than a million cells in total.

\- Always leave a complete, valid submission: if a step fails or runs out of time, fall back to a cheaper method rather than leaving files missing.

2. ‘output/PLAYBOOK.md‘: the same strategy in prose, written for another agent who will solve a new task by hand: what the runs learned, the decision rules (which approach when, and how to tell from the inputs), checks that catch a bad alignment without ground truth, and the mistakes to avoid.

3. ‘output/NOTES.md‘: how you got there: which runs and tasks support each rule, what you tried, and how you tested the policy (it should run end to end on several training tasks of different kinds).

Work in ‘scratch/‘ for experiments, and keep it under 20 GB (delete copied task data you no longer need).

How to test, so that the work stays quick:

\- Test end to end on small and medium training tasks only, each run capped at 10 minutes of wall time (‘timeout 600 bash output/policy/run.sh <workspace>‘). Do not run the policy in full on the largest tasks (many sections or over a million cells): check that it scales by timing it on a subset of their sections and extrapolating, and rely on the fallback for the rest.

\- Run each test as one blocking command in the foreground and read its result when it returns. Do not start it in the background and poll it with repeated ‘sleep‘ and check commands: every check is another full turn.

When you finish, give a short summary of the strategy.

## Supplementary Note 4: Leakage audit

Leakage prevention. We processed the input data to prevent held-out evaluation data to prevent entering the agent workspace (Supplementary Figure 2a). Workspace inputs are generated via a strict allowlist: across all 1,128 tissue sections in the benchmark, sanitized h5ad files retain only the expression count matrix, feature table, expression layers, anonymized observation identifiers, and observed spatial coordinates.

Automated post-run auditing. Following every run, we used a programmatic keyword-based audit to look through the execution trace, workspace directory, and submitted files.

Across all runs in the primary analysis set (n = 1,452), 568 runs triggered keyword flags (generic biological terms such as “Cluster”, “label” and “subclass”), predominantly under the rich-context prompt, where pre-installed package documentation references standard AnnData annotation fields. Controlling for model and prompt condition, flagged runs showed no advantage over unflagged runs (mean difference of +0.1 within-task percentile points, 95% CI: −4.4 to +3.2; Supplementary Figure 2), providing no evidence of answer leakage.

Dataset-level shortcuts identified via trace review. Manual inspection of execution traces and task inputs uncovered several dataset-intrinsic shortcuts and guided benchmark remediation (Supplementary Figure 2d).

On nine synthetic perturbation tasks derived from SABench and GEASO (sa-040–sa-043, sa-403, and sa-407–sa-410, plus sa-402), perturbed sections were generated by transforming a single source section, allowing exact cell-to-cell expression matching (used by gpt-6-astra in 6 of 13 runs and claude-opus-5.5 in 1 of 10). On the BARISTA-seq task sa-017, gemini-4-argon reverse-engineered the original authors’ coordinate rounding and imaging tile prefixes from the public input file to recover the exact rotation angles, yielding a negligible change in primary score (0.532 versus 0.531 for the top published baseline).

a What the harness prevents
<table><tr><td><img src="images/1df21b93896416d530be32dd8f79872e349940e6c50416035d4b21822190272f.jpg"/></td><td>Allowlist inputs only expression, gene table, layers, cell IDs and observed coordinates are copied</td></tr><tr><td><img src="images/98393400a18133e4b104e020cc01afb29bc2678e0954c416a87b0f820434bce6.jpg"/></td><td>No index channels cell IDs anonymised and rows shuffled with a fixed seed</td></tr><tr><td></td><td>One container per run software and data read-only; the workspace is the only writable folder</td></tr><tr><td><img src="images/2ea52fa30be49be3b2fb39931012d8f02aca2e69f203ff54d52a1110b48c618c.jpg"/></td><td>Nothing hidden mounted no truth, labels, scorer or other runs; scoring only after the agent exits</td></tr><tr><td><img src="images/86f7d7be0fc6c73b5f00d0b4e252358390988e197c5ef24937350fd345b5af99.jpg"/></td><td>Wiped on exit home folder, credentials and packages removed; snapshot copied out without data</td></tr></table>

## b What the post-run audit flags

![](images/4a5ebda8ef4ad1b41d4c15328ea956e2255c036693589a716c8f14acf0fcded4.jpg)

![](images/5bf7a6ad8c559771f946e9a5f57948ea579d1d827e970e9c00941635fc87daab.jpg)

c Do flagged runs score better? (one run per model, task and condition)  
![](images/29798f3157a50edbc5bb67da4d6dd55ad37e75e471dc0db361379e3212a9c40c.jpg)

![](images/698df8d7188fe6f7df33ea957e08232e8809a63ed0a611fdd4e40188b166f58f.jpg)

d Leaks found by hand, and what we did
<table><tr><td>Leak</td><td>What it allows</td><td>Handling</td></tr><tr><td>Shared file system (before isolation)</td><td>copied the answer, edited the scorer, reused other runs</td><td>one container per run</td></tr><tr><td>Index channels (STARmap, BARISTA-seq, MHPR sweeps)</td><td>cell IDs spell out position; rows stored in spatial order</td><td>IDs anonymised, rows shuffled, runs voided</td></tr><tr><td>Spot lattice (sa-303 to sa-305)</td><td>the lattice reveals the rotation modulo 90 or 60 degrees</td><td>leak-free sa-306, sa-307; not comparable</td></tr><tr><td>Copies of one section (9 sweep and stitching tasks)</td><td>identical expression rows give the transform exactly</td><td>reported with and without</td></tr></table>

Supplementary Figure 2: Container isolation layers and automated leakage audit results. a, Isolation architecture preventing access to held-out data and scoring scripts inside the container workspace. b, Distribution of automated audit alerts across all audited runs by category and most frequently matched keywords. c, Proportion of flagged runs in the primary analysis set and within-condition percentile difference between flagged and unflagged runs. d, Summary of dataset-level shortcuts identified through trace inspection and remediation.

## Supplementary Note 5: Scoring details

Coordinate error. Evaluation metrics are computed across all specified section pairs in a task and averaged across pairs. Coordinate error is defined as the per-axis mean absolute error,

$$
\mathrm { M A E } = \frac { 1 } { 2 N } \sum _ { i = 1 } ^ { N } \left( | x _ { i } - x _ { i } ^ { \mathrm { t r u e } } | + | y _ { i } - y _ { i } ^ { \mathrm { t r u e } } | \right) ,
$$

evaluated directly in native task coordinates without post-hoc registration, following prior work [11]. On the Drosophila embryo task (sa-019), where ground-truth coordinates in the SABench repository are stored in units 20-fold smaller than the input coordinates, ground-truth positions were rescaled to input units prior to scoring. Relative error reduction, $1 - \mathrm { M A E / M A E _ { i d e n t i t y } } ,$ is also reported for tasks where the unaligned identity error is at least 10 µm.

Supplementary Table 5: Modes of benchmark leakage observed prior to container isolation. Behaviors identi fied in execution traces during initial pilot runs on a shared filesystem.
<table><tr><td>Leakage mode</td><td>Observed agent behavior</td></tr><tr><td>Copying ground truth</td><td>Traversed parent directories despite prompt restrictions, located held-out ground-truth coordinate files, and submitted them directly.</td></tr><tr><td>Modifying the scorer</td><td>Located the evaluation script on the shared filesystem and modified or replicated it to inflate internal validation scores.</td></tr><tr><td>Cross-run copying</td><td>Inspected output directories and scripts generated by prior or concurrent agent runs on the same task and reused their solutions.</td></tr><tr><td>Extracting bundled out- puts</td><td>Located precomputed alignment outputs bundled within cloned method repositories and submitted them as newly computed results.</td></tr></table>

Note: These behaviors occurred during uncontainerized pilot testing; all subsequent benchmark evaluations were executed in isolated containers.

Label agreement. Label concordance metrics are evaluated within the spatial overlap window (the intersection of section bounding boxes) and assigned a score of 0 when sections do not overlap. Matching accuracy (also exported as landmark accuracy) measures the fraction of cells in the aligned section whose nearest spatial neighbor in the reference section shares the same annotation label. Region overlap (exported as region-grid IoU) is the mean intersection over union of alpha shapes $( \alpha = 1 2 )$ across annotated anatomical regions (scored as 0 for regions containing fewer than 4 cells).

Expression agreement. Within the spatial overlap window, a 10 × 10 regular grid is constructed, and we retain grid bins where both sections contain at least half of the smaller section’s mean cell count. For each marker gene, spatial expression profiles across retained bins are compared between sections using Pearson correlation (assigned 0 if either profile is constant), cosine similarity, structural similarity (SSIM, computed on maximum-scaled vectors), and mutual information (MI), averaged across marker genes. Main text figures report expression agreement as the mean of gene-wise Pearson correlation, cosine similarity, and SSIM.

Composite score and within-task percentile. For tasks without ground-truth coordinates, the composite overall score is the unweighted mean of the four gene expression metrics plus label matching accuracy and region overlap when annotations are present. Within-task percentile ranks each run on the task’s primary metric (coordinate error where available, and composite score otherwise) among the 12 primary model-by-prompt conditions, assigning mid-ranks to ties. Failed runs are assigned zero agreement and the coordinate error of the unaligned identity baseline.

## Supplementary Note 6: Statistical details

Confidence intervals were estimated via cluster bootstrap resampling (10,000 iterations) over the 19 source datasets, with binomial Wilson intervals reported as a secondary reference. Prompt-condition effects (change in within-task percentile across all 121 tasks and change in $\log _ { 1 0 } \mathrm { M A E }$ across the 32 coordinatetruth tasks) and pairwise model comparisons against gpt-6-astra were evaluated at the dataset level using two-sided sign-flip permutation tests (20,000 permutations), with Wilcoxon signed-rank and Friedman tests as secondary checks. gpt-6-astra (mean within-task percentile 79, 95% CI: 67–85) significantly outperformed the four models ranked below gemini-4-argon by 31–51 percentile points $( P \leq 0 . 0 0 9 )$ and exceeded gemini-4-argon by 7.9 points $( P = 0 . 1 2 )$ ; single-run win rates for gpt-6-astra against the top baseline were 48% (95% CI: 10–72%) on expression agreement and 64% (95% CI: 46–83%) on MAE.

On cross-platform tasks where agents outperformed published baselines on expression agreement, agent alignments exhibited lower MNN cell pairing coverage than the top published methods (median

0.18 versus 0.25), indicating that higher grid-level expression correlations can be achieved in part by reducing partial tissue overlap; only 1 of 192 final agent scripts on these tasks explicitly computed expression correlations on a $1 0 \times 1 0$ spatial grid.

Across the 32 tasks with ground-truth coordinates, the divergence between label-based and expressionbased metrics was most pronounced on the 12 simulated tissues (median correlation with coordinate error of −0.10 to 0.30 for expression metrics versus 0.69 for MNN label agreement), whereas on the 15 realtissue tasks supporting both metric families, expression agreement $( \rho = 0 . 6 5 )$ approached label matching accuracy $( \rho = 0 . 7 1 )$ . Reported P values are unadjusted unless otherwise noted. All statistical analyses were conducted in Python 3.11 (numpy, pandas, scipy).

## Supplementary Note 7: Trace processing, step associations and trace content

Execution logs were parsed by retaining up to 200,000 characters per step and 60 lines per command output after stripping ANSI terminal escape sequences, progress bars, binary blobs, and credential strings. Each run’s final submission script was identified as the last successful execution step that wrote the output coordinate files. Each tool call was classified into one of eight operational phases (inspecting data, setting up, preprocessing, trying a method, evaluating, writing submission, validating output, and waiting), and intermediate and final scripts were annotated with 48 algorithmic step tags via rule-based AST and regularexpression matching. Because Codex CLI does not log intermediate file-edit payloads or step timestamps and Gemini CLI streams text fragments, exploratory step counts for OpenAI models represent lower bounds.

Across basic-prompt runs, exploratory steps were executed substantially more often than they were retained in the final submission: rotational grid searches were explored in 66% of runs and retained in 28%, diagnostic plotting was performed in 53% and retained in 21%, and gradient-based optimization was explored in 12% and never retained (Supplementary Figure 3). Under rich context, PASTE, PASTE2, Spateo, and SPACEL each appeared in 25–38% of final submissions. To quantify the association between individual algorithmic steps and alignment performance, we computed the within-model, within-prompt difference in within-task percentile between runs retaining a given step in their final script and runs without it, pooled across conditions by inverse-variance weighting.

Across all 121 tasks, nearest-neighbor matching (+6.6 percentile points, 95% CI: 3.8–9.7), nonrigid warping (+6.0, 2.7–10.2), affine registration (+4.8, 1.6–7.6), and shared-gene intersection (+3.4, 0.4–6.9) were positively associated with primary scores, whereas invoking a published package, generating diagnostic plots, or computing custom internal validation scores showed no consistent association. Restricted to the 32 tasks with ground-truth coordinates, however, nonrigid warping was associated with substantially worse spatial accuracy (−17.4 percentile points, 95% CI: −26.5 to −6.2), whereas Procrustes registration (+13.1, 2.8–21.7) and numerical optimization (+7.7, 0.3–14.6) were associated with lower coordinate error.

We further characterized the textual narration and final code across all primary runs (Supplementary Figure 4). Verbosity varied widely across model families: gpt-5.5 generated a median of 1,032 narration words per run under the basic prompt (1,700 under rich context), compared with roughly 200 words under the basic prompt for claude-sonnet-5.5, gpt-6-astra, and gemini-4-argon; conversely, gpt-6-astra and gemini-4-argon produced the longest final scripts (medians of 324 and 220 lines versus 61–70 lines for the Claude models; Supplementary Figure 4a).

Projecting 50-dimensional TF-IDF code token embeddings and bge-small-en-v1.5 narration embeddings with UMAP revealed strong clustering by model identity: 15-nearest-neighbor classification recovered the generating model in 87% of runs from code and 75% from narration (versus 17% by chance), compared with 56% for the underlying transformation class (Supplementary Figure 4b,c), indicating distinct model-specific coding and communication signatures.

What each model tried and kept, and how each step goes with the score
<table><tr><td rowspan=1 colspan=14>Model columns: basic prompt. Light bar: tried at any point; dark bar and number: in the final code (% of runs).                              Step in final code: change inpercentile (within model, prompt)Opus 5.5Sonnet 5.5  GPT-6   GPT-5.5  Gemini   Gemini  Rich context  All tasks      Truekept (%)positionsAstra            Argon   FlashPreprocessing / features</td></tr><tr><td rowspan=1 colspan=1>Gene handling    Shared-gene intersection</td><td rowspan=1 colspan=2>12</td><td rowspan=1 colspan=1>38</td><td rowspan=1 colspan=2>23</td><td rowspan=1 colspan=2>35</td><td rowspan=1 colspan=2>31</td><td rowspan=1 colspan=2>26</td><td rowspan=1 colspan=1>+3366</td><td rowspan=1 colspan=1>-38</td></tr><tr><td rowspan=2 colspan=1>Expression scale   Normalize / log1pHighly variable genes</td><td rowspan=1 colspan=2>40       4</td><td rowspan=1 colspan=1>58</td><td rowspan=1 colspan=2>50</td><td rowspan=1 colspan=2>70</td><td rowspan=1 colspan=2>56</td><td rowspan=1 colspan=2>48</td><td rowspan=1 colspan=1>+1735</td><td rowspan=1 colspan=1>-2157</td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=4>14</td><td rowspan=1 colspan=2>5</td><td rowspan=1 colspan=1>+065</td><td rowspan=1 colspan=1>+715</td></tr><tr><td rowspan=1 colspan=1>Embedding     PCA/ SVD embedding</td><td rowspan=1 colspan=2>45       49</td><td rowspan=1 colspan=1>59</td><td rowspan=1 colspan=2>23</td><td rowspan=1 colspan=4>45       47</td><td rowspan=1 colspan=2>44</td><td rowspan=1 colspan=1>+3642</td><td rowspan=1 colspan=1>+2182</td></tr><tr><td rowspan=1 colspan=1>Cell labels      Clustering / domains</td><td rowspan=1 colspan=2>22</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>3</td><td rowspan=1 colspan=4>15       11</td><td rowspan=1 colspan=2>25</td><td rowspan=1 colspan=1>+3255</td><td rowspan=1 colspan=1>+268</td></tr><tr><td rowspan=1 colspan=1>Coarse initialization</td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=4></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Centering      Centroid centering</td><td rowspan=1 colspan=2>15</td><td rowspan=1 colspan=1>22</td><td rowspan=1 colspan=2>24</td><td rowspan=1 colspan=4>13      35</td><td rowspan=1 colspan=2>21</td><td rowspan=1 colspan=1>-2293</td><td rowspan=1 colspan=1>+2108</td></tr><tr><td rowspan=1 colspan=1>Orientation      Flip / mirror searchsearch</td><td rowspan=1 colspan=1>38</td><td rowspan=1 colspan=1>40</td><td rowspan=1 colspan=1>55</td><td rowspan=1 colspan=2>28</td><td rowspan=1 colspan=4>7       26</td><td rowspan=1 colspan=2>34</td><td rowspan=1 colspan=1>+2488</td><td rowspan=1 colspan=1>+5112</td></tr><tr><td rowspan=2 colspan=1>Rotation grid searchPrincipal-axis alignment</td><td rowspan=1 colspan=1>26</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>42</td><td rowspan=1 colspan=2>24</td><td rowspan=1 colspan=4>20</td><td rowspan=1 colspan=2>16</td><td rowspan=1 colspan=1>+2321</td><td rowspan=1 colspan=1>-2100</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>39</td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=4></td><td rowspan=1 colspan=2>2</td><td rowspan=1 colspan=1>-471</td><td rowspan=1 colspan=1>-432</td></tr><tr><td rowspan=1 colspan=1>Image-basedregistration     Rasterize to image</td><td rowspan=1 colspan=1>38</td><td rowspan=1 colspan=1>27</td><td rowspan=1 colspan=1>43</td><td rowspan=1 colspan=2>2</td><td rowspan=1 colspan=4>31       12</td><td rowspan=1 colspan=2>21</td><td rowspan=1 colspan=1>+1368</td><td rowspan=1 colspan=1>-499</td></tr><tr><td rowspan=1 colspan=1>FFT / cross-correlation</td><td rowspan=1 colspan=2>31       31</td><td rowspan=1 colspan=1>14</td><td rowspan=1 colspan=2>17</td><td rowspan=1 colspan=4></td><td rowspan=1 colspan=2>13</td><td rowspan=1 colspan=1>+2218</td><td rowspan=1 colspan=1>+292</td></tr><tr><td rowspan=1 colspan=1>Correspondence / fitting</td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=6></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Point         Nearest-neighbor searchcorrespondence</td><td rowspan=1 colspan=2>36      39</td><td rowspan=1 colspan=1>54</td><td rowspan=1 colspan=2>19</td><td rowspan=1 colspan=4>74       35</td><td rowspan=1 colspan=2>39</td><td rowspan=1 colspan=1>+7597</td><td rowspan=1 colspan=1>+1147</td></tr><tr><td rowspan=2 colspan=1>Mutual nearest neighborsOptimal transport</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=4></td><td rowspan=1 colspan=2>6</td><td rowspan=1 colspan=1>+671</td><td rowspan=1 colspan=1>+420</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=4></td><td rowspan=1 colspan=2>6</td><td rowspan=1 colspan=1>-657</td><td rowspan=1 colspan=1>+911</td></tr><tr><td rowspan=2 colspan=1>Transform      RANSACestimationProcrustes / Kabsch</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=4>16</td><td rowspan=1 colspan=2>3</td><td rowspan=1 colspan=1>+252</td><td rowspan=1 colspan=1>+1314</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=4>16</td><td rowspan=1 colspan=2>17</td><td rowspan=1 colspan=1>+5184</td><td rowspan=1 colspan=1>+1354</td></tr><tr><td rowspan=1 colspan=1>ICP</td><td rowspan=1 colspan=1>16</td><td rowspan=1 colspan=1>15</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=4></td><td rowspan=1 colspan=2>6</td><td rowspan=1 colspan=1>+596</td><td rowspan=1 colspan=1>+437</td></tr><tr><td rowspan=2 colspan=1>Parameter search  Numerical optimizerGradient / autograd</td><td rowspan=1 colspan=1>40</td><td rowspan=1 colspan=1>25</td><td rowspan=1 colspan=1>83</td><td rowspan=1 colspan=2>17</td><td rowspan=1 colspan=4>39       26</td><td rowspan=1 colspan=2>26</td><td rowspan=1 colspan=1>+1469</td><td rowspan=1 colspan=1>+8106</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=4></td><td rowspan=1 colspan=2>7</td><td rowspan=1 colspan=1>+1354</td><td rowspan=1 colspan=1>+1513</td></tr><tr><td rowspan=1 colspan=1>Transform model</td><td rowspan=1 colspan=5></td><td rowspan=1 colspan=4></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Transform family   Rigid</td><td rowspan=1 colspan=5>76               79       65</td><td rowspan=1 colspan=4>70       52</td><td rowspan=1 colspan=2>65</td><td rowspan=1 colspan=1>-0981</td><td rowspan=1 colspan=1>+5288</td></tr><tr><td rowspan=2 colspan=1>Similarity (scale)Affine</td><td rowspan=1 colspan=3></td><td rowspan=1 colspan=2>11</td><td rowspan=1 colspan=5>10</td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>-497</td><td rowspan=1 colspan=1>n/a</td></tr><tr><td rowspan=1 colspan=3>17</td><td rowspan=1 colspan=2>36</td><td rowspan=1 colspan=5>31       28</td><td rowspan=1 colspan=1>27</td><td rowspan=1 colspan=1>+5425</td><td rowspan=1 colspan=1>+437</td></tr><tr><td rowspan=1 colspan=1>Nonrigid / deformable</td><td rowspan=1 colspan=3>40</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=3>37</td><td rowspan=1 colspan=2>2</td><td rowspan=1 colspan=1>22</td><td rowspan=1 colspan=1>+6266</td><td rowspan=1 colspan=1>-1735</td></tr><tr><td rowspan=1 colspan=1>Published method</td><td rowspan=1 colspan=4></td><td rowspan=1 colspan=4></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Optimal-transport  PASTEmethods</td><td rowspan=1 colspan=4></td><td rowspan=1 colspan=7>25</td><td rowspan=1 colspan=1>-1189</td><td rowspan=1 colspan=1>-339</td></tr><tr><td rowspan=1 colspan=1>PASTE2</td><td rowspan=1 colspan=11>31</td><td rowspan=1 colspan=1>+2224</td><td rowspan=1 colspan=1>-248</td></tr><tr><td rowspan=1 colspan=1>Spateo</td><td rowspan=1 colspan=11>37</td><td rowspan=1 colspan=1>+1267</td><td rowspan=1 colspan=1>-664</td></tr><tr><td rowspan=1 colspan=1>Embedding /     SPACELgraph methods</td><td rowspan=1 colspan=11>38</td><td rowspan=1 colspan=1>+1277</td><td rowspan=1 colspan=1>-564</td></tr><tr><td rowspan=1 colspan=1>CAST</td><td rowspan=1 colspan=11>6</td><td rowspan=1 colspan=1>+744</td><td rowspan=1 colspan=1>+119</td></tr><tr><td rowspan=1 colspan=1>SLAT</td><td rowspan=1 colspan=11>9</td><td rowspan=1 colspan=1>+766</td><td rowspan=1 colspan=1>+1217</td></tr><tr><td rowspan=1 colspan=1>STAligner</td><td rowspan=1 colspan=11>4</td><td rowspan=1 colspan=1>+535</td><td rowspan=1 colspan=1>+66</td></tr><tr><td rowspan=1 colspan=1>SANTO</td><td rowspan=1 colspan=3></td><td rowspan=1 colspan=7></td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>+715</td><td rowspan=1 colspan=1>+64</td></tr><tr><td rowspan=1 colspan=1>STAIR</td><td rowspan=1 colspan=3></td><td rowspan=1 colspan=7></td><td rowspan=1 colspan=1>10</td><td rowspan=1 colspan=1>+976</td><td rowspan=1 colspan=1>+1213</td></tr><tr><td rowspan=1 colspan=1>Image-based     STalignmethods</td><td rowspan=1 colspan=3></td><td rowspan=1 colspan=5></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>+356</td><td rowspan=1 colspan=1>+611</td></tr><tr><td rowspan=1 colspan=4>GPSA</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>0</td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=2>0</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>n/a</td><td rowspan=1 colspan=1>n/a</td></tr><tr><td rowspan=1 colspan=5>Multi-slice strategy</td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=2 colspan=9>Slice ordering    Sequential chaining               12Common reference slice                             15</td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>31</td><td rowspan=1 colspan=1>15</td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>+079</td><td rowspan=1 colspan=1>+177</td></tr><tr><td rowspan=1 colspan=9>Tiling / 3D      Stitching / overlap</td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>+825</td><td rowspan=1 colspan=1>+1311</td></tr><tr><td rowspan=4 colspan=12>Self-evaluation / selectionLabel-free      Internal alignment score           23      35      60      30      18      28      30validationVisual check     Plot for visual QC                21       17       74       13       2                19-20</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>-0455</td><td rowspan=1 colspan=1>-391</td></tr><tr><td rowspan=1 colspan=1>+2292</td><td rowspan=1 colspan=1>+1050</td></tr><tr><td rowspan=1 colspan=1>0+20    -20</td><td rowspan=1 colspan=1>+20</td></tr></table>

Steps tried by fewer than 3% of runs are not shown (7 of 46). Rich context: six models pooled. Solid bars: 95% bootstrap interval excludes zero; small numbers: runs with the step. GPT models undercount "tried": Codex file edits carry no contents, so only shell commands, inline code and the final snapshot are seen.

Supplementary Figure 3: Algorithmic steps explored and retained across models and their association with alignment scores. Rows correspond to algorithmic steps grouped by pipeline stage (omitting 7 of 46 steps explored in < 3% of runs). Model columns display basic-prompt runs, showing the percentage of runs that explored each step at any point (light bars) and retained it in the final code (dark bars and numerical labels); the rich-context column pools all six evaluated models. The rightmost two columns show the within-model, within-prompt difference in percentile between runs with and without each step in the final submission across all tasks (primary metric) and across the 32 coordinate-truth tasks (coordinate error), with 95% bootstrap confidence intervals and run counts. Solid bars denote confidence intervals excluding zero.

d  
![](images/e183fc0b6389f9871b833d80e5a35c203410af663fa6c035381ff9235e73ec24.jpg)

![](images/18b1c1a56fa15aa9c4d54ad3738ea006bd60caac598b210e5a7b3f42042b7af7.jpg)

![](images/9711c36c6ecf6763848d341e65e4a597b582c084be5ff06d0296b2e7f60071d5.jpg)

![](images/584b64d78aac516f7296e0e7d32c7d52142324fa9847fa5baedd3dbb4150b6e0.jpg)

![](images/9c369572587ab470eb5d1ba95655ee0803819622fc39b48f2c6dc05d0c47f424.jpg)

![](images/f33399d13940dbb9d43837f36a41005ac5386faf3cc7805bb68e597c9c08af48.jpg)  
Supplementary Figure 4: Quantitative analysis of execution trace volume, code and text embeddings, selfevaluations, and vocabulary. a, Word count of agent narration and line count of final submitted code per run across models under the basic prompt (open boxes) and rich-context prompt (filled boxes; boxes denote interquartile range, whiskers 10th–90th percentiles). ${ \mathbf { b } } , { \mathbf { c } } ,$ UMAP projections of final submitted code (50-dimensional TF-IDF token embeddings; b) and agent narration (bge-small-en-v1.5 sentence embeddings; c), colored by model (large panels) and transformation family (inset panels). d, Proportion of runs in which the agent’s final summary claimed a high-quality alignment, stratified by within-task performance tertile (bottom, middle, and top third from light to dark). e, Most distinctive narration terms per model ranked by weighted log-odds ratio.

To evaluate whether agents’ concluding summaries reliably reflected alignment quality, we scored each run’s final message using a lexicon of positive quality claims (such as “excellent”, “near-perfect”, “high-quality”, and “well aligned”) net of epistemic hedges (“may”, “approximate”, “fallback”, “could not”). Self-reported quality showed no correlation with actual within-task percentile for any model (Spearman $| \rho | \leq 0 . 0 7 )$ . Notably, gemini-3.5-flash asserted a high-quality alignment in 55–70% of runs across bottom, middle, and top performance tertiles (58%, 55%, and 70%), whereas all other models made such claims in only 0–7% of runs (Supplementary Figure 4d).

Vocabulary analysis further highlighted characteristic model priorities: geometric operations (mirror, shift, flip, translation) for Claude models, spatial concordance and data integrity for gpt-6-astra, qualitative deliberation (deciding, plausible, stronger) for gpt-5.5, multi-scale, coarse-to-fine registration and named packages for gemini-4-argon, and confident completion claims (successfully, optimal, methodology) for gemini-3.5-flash (Supplementary Figure 4e).

## Supplementary Note 8: Run-to-run variation

To quantify stochastic run-to-run variation, we compared two independent basic-prompt replicates per model and task (where replicate 1 corresponds to the primary analysis set; gemini-4-argon was run once under the basic prompt and gpt-5.5 completed 69 second-replicate tasks). Cross-replicate consistency was highest for gpt-6-astra (Spearman $\rho = 0 . 9 3$ on coordinate error and 0.98 on composite score) and lowest for gemini-3.5-flash $( \rho = 0 . 6 3$ and 0.57; Supplementary Figure 5a). Replicate runs disagreed on whether the agent matched or beat the best published baseline on 12–18% of tasks; taking the better of two runs increased win rates by 6–10 percentage points (for example, from 61% to 67% for gpt-6-astra and from 32% to 42% for claude-opus-5.5; Supplementary Figure 5b). Overall model rankings were identical whether evaluated on replicate 1 or replicate 2 alone (Supplementary Figure 5c), with the median absolute difference between replicates equal to 10–14% of the cross-model range per task for top models (and 49% for gemini-3.5-flash; Supplementary Figure 5d).

Variance decomposition. We fitted a saturated factorial linear model (including task, model, prompt condition, and all two- and three-way interactions) across all 2,521 scored runs, partitioning total sums of squares sequentially in the order task, model, prompt, model × prompt, task × model, task × prompt, task × model × prompt, and replicate residual (Supplementary Figure 6a). Outcomes were analyzed as $\log _ { 1 0 }$ MAE on the 28 primary coordinate-truth tasks, composite score on the 89 tasks without groundtruth positions, and $\log _ { 1 0 }$ -transformed cost and runtime across all 121 tasks. Reversing entry order of task and model altered variance shares by less than 2 percentage points, and Type II sums of squares yielded consistent results. Restricting to the balanced basic-prompt subset across the five models with replicate runs (Supplementary Figure 6b), task identity accounted for 65% and 59% of variance in coordinate error and composite score, model identity for 5% and 9%, task-by-model interaction for 19% and 16%, and replicate residual for 11% and 16%.

## a Two basic-prompt runs of the same model on the same task

![](images/7f9d9dea15e6bbfe44f356c18ae6b14589b90db7b23dce43c311008ddc0bc23b.jpg)

![](images/f2d122a4e396a9fa713ff9160c61b1e9e241d78b5c36385447bf80ce0b7f84c2.jpg)

![](images/005c83e8c596ae3760a47f9f0a72c1ef1ddc9730ca1ef93fb5e9d9001793f609.jpg)

![](images/952d83cc130ef9fe298e7bf0593bd646acc6a6efac3ceccc67cb152346dc6d4a.jpg)

![](images/c9ad934401119bd9889367bca7723e12ac1205a8a9ac14d0ed1cdbba20e298d9.jpg)

![](images/74ce5c2540ff97a223e61977d02f1c8b2a91c8c4daa5d488c667232078564dee.jpg)

![](images/4cf122456e696bc4516519520d8cb60bcaabbd31d022947f2ded7b932acc1a48.jpg)  
c

![](images/710deeb331373fd52afb9ab388edf3f6605fe42ce5fcd5bdba4fc891c9f6094f.jpg)

![](images/19668202ad829fec2d7640d1f4b8cd41e94beae823177b90aeda6c2f375116dd.jpg)  
b

![](images/7eca75f9f2864ccbea2b94146d87f6c9fedefd8879da71b3e240ef01cd58abee.jpg)  
d

![](images/7fa5f66300c1d29bc536570bbae6a0504f983d2aac44001ef7394502f1aa5c3e.jpg)

![](images/de1389e9a8c4d65e2f930387b842874b60f4daa7d9a6dcd2883bc45c6777bbab.jpg)

![](images/40d1054a52d9b3eefb21b4369ed1000f0d924a4eeb8478c77d863c2bd6745145.jpg)  
Supplementary Figure 5: Run-to-run reproducibility across independent replicates. a, Comparison of replicate 2 versus replicate 1 on coordinate error (MAE) and composite score across five models, with Spearman rank correlation ρ. b, Percentage of tasks matching or outperforming the best published baseline for replicate 1, replicate 2, and the best of two replicates. c, Mean within-task percentile across models estimated independently from replicate 1 and replicate 2. d, Absolute difference between replicates as a fraction of the cross-model range per task.

![](images/3fbf372acc99491248381d3c59eae791db691f51dfcc46f288b5150cf279e83b.jpg)  
Supplementary Figure 6: Variance decomposition of alignment accuracy, cost, and runtime. a, Sequential sumof-squares decomposition across all 2,521 scored runs, showing the fraction of variance in coordinate error, composite score, API cost, and runtime explained by task, model, prompt condition, interaction terms, and replicate residual. b, Corresponding decomposition restricted to basic-prompt runs across the five models with replicates.