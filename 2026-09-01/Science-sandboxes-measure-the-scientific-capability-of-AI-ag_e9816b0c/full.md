# Science sandboxes measure the scientific capability of AI agents

Arya S. Rao<sup>1,2\*</sup>, Rodrigo I. Castro<sup>3</sup>, Sager J. Gosai<sup>4</sup>, Kenneth B. Hsu<sup>1,10</sup>, Yasha Ektefaie<sup>1</sup>, Shantanu Singh<sup>1</sup>, Sangeeta N. Bhatia<sup>1,5,6,7,8</sup>, Steven K. Reilly<sup>9</sup>, Ryan Tewhey<sup>3</sup>, Eric S. Lander<sup>1,10,11\*</sup>, Pardis C. Sabeti<sup>1,6,12,13\*</sup>

<sup>1\*</sup>The Broad Institute of MIT and Harvard; Cambridge, MA 02142, USA.

<sup>2</sup>Department of Biomedical Informatics, Harvard Medical School; Boston, MA 02115, USA.

The Jackson Laboratory; Bar Harbor, ME 04609, USA.

<sup>4</sup>Sutter Hill Ventures; Palo Alto, CA 94304, USA.

<sup>5</sup>David H. Koch Institute for Integrative Cancer Research, Massachusetts Institute of Technology, Cambridge, MA 02139, USA.

<sup>6</sup>Howard Hughes Medical Institute, Chevy Chase, MD 20815, USA.

<sup>7</sup>The Wyss Institute for Biologically Inspired Engineering at Harvard University, Boston, MA 02115, USA.

<sup>8</sup>Harvard-MIT Program in Health Sciences and Technology, Institute for Medical Engineering and Science, Massachusetts Institute of Technology, Cambridge, MA 02139, USA.

Department of Genetics, Yale School of Medicine; New Haven, CT, USA.

<sup>10</sup>Department of Systems Biology, Harvard Medical School, Boston, MA, USA.

<sup>11</sup>Department of Biology, Massachusetts Institute of Technology, Cambridge, MA, USA.

<sup>12</sup>Department of Immunology and Infectious Diseases, Harvard T.H. Chan School of Public Health, Boston, MA 02115, USA.

<sup>13</sup>Department of Organismic and Evolutionary Biology, Harvard University, Cambridge, MA 02138, USA.

\*Corresponding authors: arao@broadinstitute.org, eric@broadinstitute.org, pardis@broadinstitute.org

## Abstract

Scientific progress depends not only on finding solutions, but on learning the rules that explain why they work and using that understanding to design better experiments. We introduce science sandboxes, a framework for studying this capability in AI agents through repeated cycles of experimentation, feedback, and hypothesis revision. Science sandboxes invite an agent to query the natural world in diferent ways, ranging from “wet” physical experiments, to “damp” predictive models trained on empirical data, to “dry” invented rules. By establishing a common experimental loop and a protocol for evaluating agents within it, science sandboxes allow assessment of both quantitative performance on specific metrics and qualitative scientific reasoning, across a spectrum of empirical verifiability. Here, we instantiate this framework in two biological settings, models of regulatory genomics and protein fitness prediction, and examine the capabilities of frontier agents. Across these settings, we could see when agents successfully optimized a quantitative metric without understanding the rules underlying the system. In particular, their scientific reasoning deteriorated when they encountered systems whose rules fell outside familiar biological priors. By highlighting such failure modes, science sandboxes make the frontier of scientific capability measurable and provide a controlled setting in which to study and ultimately expand it.

## Introduction

Experimental science advances by asking questions of nature to learn its rules. Consider the problem of learning how DNA sequences control gene expression. An investigator might select a set of sequences, measure their activity in the laboratory, infer which features of the sequences matter, and use that hypothesis to design a more informative set. In the course of this work, the aim is not merely to find sequences that drive gene expression, but to discover and understand regularities that explain why they work and guide the next experiment. A true measure of scientific capability should therefore assess whether experimental evidence leads to better hypotheses and more informative experiments, reflecting growing understanding rather than optimization alone. If artificial intelligence (AI) is to become a useful partner in scientific discovery, it must demonstrate this ability, motivating growing eforts to develop and evaluate increasingly capable scientific AI systems.<sup>1–8</sup>

We introduce the science sandbox: a controlled testbed for measuring scientific capability. A science sandbox recreates the essential structure of science by allowing an autonomous investigator to conduct successive experiments, interpret the results, and use what it learns to refine its hypotheses and choose the next experiments. Because the source of experimental feedback is hidden, the investigator must infer the rules of the system from the evidence alone. This source may be a physical experiment, a computational model of one, or an invented system whose rules are known only to the designers of the sandbox. By preserving the investigator’s evolving hypotheses and experimental choices, a sandbox reveals whether improved performance simply reflects mathematical optimization or is based on a deeper understanding of the system. Traditional benchmarks cannot make this distinction: they reward any strategy that raises the score.

Here, we describe science sandboxes for two archetypal problems in biology, evaluating the scientific capabilities of frontier agentic AI systems. The first sandbox, MPRAbox, instantiates the sequence-selection problem posed above: each agent is tasked with selecting maximally informative libraries of regulatory DNA sequences for training downstream predictive models, a task of choosing a large experimental set from a combinatorially vast sequence space. The second sandbox, CodonBox, tests agents’ ability to conduct de novo rule discovery, asking, as early molecular geneticists once did, how an unfamiliar genetic system translates sequence into function.<sup>9,10</sup> Together, these sandboxes test AI as a scientific investigator, not merely as a solver of scientific tasks. In the process, they open a path towards understanding and scaling scientific capability itself.

## Results

## Science sandboxes measure how agents learn from experiments

A science sandbox is a controlled environment in which AI agents can conduct experiments to learn the rules governing a phenomenon (Figure 1a). In each round, the agent chooses what to test based on its initial hypotheses, receives feedback from a sealed oracle, revises its hypotheses based on the feedback, and selects the next experiment. The aim is not only to determine whether the agent can improve a score, but whether it can behave like an experimental scientist: proposing informative tests, interpreting evidence, and converging toward coherent rules.

Every science sandbox has three core parts: specimens, assays, and an oracle. By specimens, we broadly mean any entity that can be submitted for testing, such as a DNA sequence, protein, chemical, cell, designed construct, or abstract string. By assays, we mean methods for ascertaining specific properties of those specimens. By an oracle, we mean a hidden mechanism that applies the assays to each specimen, analyzes the results, and returns a report to the agent, which may contain the full results or just a limited summary. In general, sandboxes can contain multiple kinds of specimens and multiple assays, and the oracle’s reports can range from a single scalar value to rich outputs such as images, time series, or molecular structures. For simplicity, we focus here on sandboxes with one type of specimen, one assay, and reports consisting ofa handful ofnumbers.

Oracles can vary in how they apply assays to generate feedback on chosen specimens. Wet oracles obtain results from actual physical experiments. Damp oracles use computational models trained on empirical data to approximate experimental results. Dry oracles apply invented rules specified by the sandbox designer, which may have no relationship to the natural world. Each type serves a diferent purpose. Wet oracles provide realism, damp oracles make repeated experimentation scalable, and dry oracles can test how well agents can infer hidden rules in arbitrary settings.

A sandbox may have a known physical meaning, a concealed physical meaning, or no physical meaning at all. In the first case, the sandbox corresponds to real biological or chemical systems and the agents are informed of this fact, allowing them to draw on published scientific knowledge in deciding what to test. In the second case, the sandbox again corresponds to real biological or chemical systems, but the information is made abstract, for example, by replacing DNA sequences (strings of A, C, G, T) with arbitrary character strings (e.g., from {0, 1, 2, 3} or $\{ \# , \$ \Phi , \%, \& \} )$ , with the goal of preventing agents from relying on familiar biological priors. This distinction is useful because it allows a sandbox to ask not only whether an agent can exploit known science, but whether it can infer unfamiliar rules from evidence alone.

![](images/27069624c59e6720b33c556551ff960d2d7cf7ab5bd323dbac660598a40c1d47.jpg)

![](images/7189fcf727297b36aef489367bdc059a89540d91877c2898fd7f1e7c2e0f005e.jpg)  
Fig. 1 Science sandboxes for closed-loop evaluation of AI experimentalists. (a) A science sandbox couples an autonomous agent to a sealed oracle through a closed loop of action, feedback, and hypothesis revision. At each design round, the agent proposes a batch of actions, receives feedback from the oracle, records its interpretation, and selects its next actions. The oracle can be wet (a physical experiment), damp (a model trained on experimental data), or dry (an invented rule created by the experimenter). (b) MPRAbox instantiates this framework in regulatory genomics. The agent selects a finite massively parallel reporter assay (MPRA) library from the efectively unbounded space of possible 200 bp regulatory DNA sequences. The library is evaluated by a sealed in silico MPRA oracle that acts as a simulated wet lab, and aggregate performance metrics from diverse hidden evaluation sets are returned to guide subsequent design rounds.

The central object of evaluation is the agent’s reasoning, not simply its final score. A science sandbox asks agents to conduct a series of experiments and to record their reasoning in a lab notebook. These notebooks allow humans, or independent AI judges, to evaluate whether the agent merely found higherscoring specimens through local search or instead formed, tested, and revised hypotheses that explain the system. In this sense, a science sandbox is not just a benchmark of performance. It is a controlled setting for measuring scientific exploration itself.

This framework is intentionally broad. Some biological examples include:

(1) Cell Growth Assay, with (i) the specimens being a collection of chemicals and (ii) the assay being applying a specific concentration of a molecule to a specific number of cells and determining percentage change in the number of live cells present five days later. The report might consist of the percentage change observed for each chemical tested.

(2) Protein Structure, with (i) the specimens being amino acid sequences and (ii) the assay being a determination of the structure of the corresponding protein. The report might contain only the proportion of amino-acid residues in an alpha-helical conformation.

(3) Massively Parallel Reporter Assay (MPRA), with (i) the specimens being a DNA sequence of length 200 (“candidate enhancer”) and (ii) the assay being inserting a candidate enhancer upstream of the promoter of a particular gene in a cell type and measuring the proportional change in transcription level (amount of RNA produced). The report might consist of the change observed for each of the candidate enhancers submitted for testing or, alternatively, just the mean change across all these candidate enhancers.

We will begin by studying this last example.

## MPRAbox is a science sandbox for regulatory sequence design

Regulatory genomics — specifically, the example of MPRA noted above — provides a natural setting for a science sandbox, because it combines (i) a combinatorially vast space of DNA sequences, providing ample specimens, with (ii) an assay that can be experimentally performed at large scale, efectively approximated by computational models at even greater scale, or scored by a rich array of arbitrary rules (Figure 1b). The fundamental goal ofMPRA is to understand how regulatory DNA sequences drive gene expression.<sup>11–13</sup>

In the simplest form of the MPRA sandbox, an oracle would receive from an agent a “library” ofN candidate enhancers and return to the agent a report containing the results (proportional change in expression) for each of the N candidate enhancers. Those results could come directly from a laboratory experiment or, for faster iteration, from a computational model that approximates the assay. For example, one could use Malinois, a published model trained on more than 700,000 experimental MPRA measurements widely used across academia and industry to approximate wet-lab results.<sup>14–21</sup> Based on the report, the agent could select a new set of specimens that it believes will have higher scores.

A more interesting and useful challenge is to ask agents not simply to find high-scoring enhancers, but to design the most informative library of size N for training a predictive sequence-to-activity model. Specifically, researchers use MPRA data to train predictive sequence-to-activity models and, because the amount of data they can collect is limited, they want to use the most informative set of candidate enhancers for training the model — that is, the library of N specimens that will most improve training of a predictive sequence-to-activity model. We wanted to see how agents perform on this task, allowing them to submit distinct libraries of N candidate enhancers across one or more rounds (Figure 1b).

In this version of the MPRA sandbox, the oracle returns to the agent only a short report, consisting of a handful ofscalar values that summarize the value ofthe library in training a model (Figure 1b). Specifically, we use a damp oracle that performs the following steps: (1) for each of the N candidate enhancers, calculate the results of the MPRA assay by using Malinois; (2) based on the (candidate enhancer, result) pairs from Malinois, train a new sequence-to-activity model from scratch; (3) evaluate the performance ofthe new model on 14 hidden test sets ofcandidate enhancers, consisting of5 sets for which the activity scores are based on empirical laboratory measurements and 9 sets for which the activity scores are based on the Malinois model (Table 1); and (4) return to the agent a summary report consisting of only (i) the Pearson correlation between the new model’s prediction and the “ground truth” for each of the 14 sets and (ii) an overall performance score, consisting of the mean of these 14 numbers. (The agents receive no other information, including about the nature or contents ofthe 14 sets.)

On each round, agents received an instructions.md file (Supplementary Information) directing them to design an MPRA library for training a generalizable sequence-to-activity model and to maintain an ongoing “lab notebook” documenting their reasoning. The task was not only to design a high-performing library, but also to develop a theory of what makes a library informative and use successive rounds to test and revise that theory.

We set N = 50,000, because this value is typical for MPRA libraries, is large enough to train meaningful predictive models, and is small enough that the constraint on library size is a major determinant of performance (Supplementary Figure 1).

As noted above, our goal is not only to assess the agent’s quantitative performance (correlation score), but also to probe the agent's reasoning to discover rules that explain why some libraries are more informative than others.

Table 1 Hidden test sets used in MPRAbox. We evaluated models across nine sequence collections representing experimentally measured regulatory sequences, human genetic variants, annotated regulatory elements, genomic background, and synthetic DNA. For five collections, we evaluated predictions against both wet-lab MPRA measurements and Malinois-predicted activity, yielding 14 labeled evaluation sets in total. All genomic sequences came from chromosomes excluded from Malinois training. Full construction and filtering criteria are described in Methods.
<table><tr><td>Evaluation set</td><td>Sequence source</td><td>Labels</td><td>n</td></tr><tr><td>MPRA holdout, chr7/13</td><td>Held-out sequences from the Gosai et al. episomal MPRA</td><td>Experimental + Malinois</td><td>60,055</td></tr><tr><td>MPRA holdout, chr19/21/X</td><td>Independent held-out sequences from the Gosai et al. episomal MPRA</td><td>Experimental + Malinois</td><td>56,340</td></tr><tr><td>UKBB/GTEx fine-mapped</td><td>Fine-mapped UK Biobank and GTEx cis-eQTL variants</td><td>Experimental + Malinois</td><td>59,084</td></tr><tr><td>UKBB/GTEx, both alleles</td><td>UK Biobank and GTEx variants with reference and alternate alleles retained</td><td>Experimental + Malinois</td><td>62,966</td></tr><tr><td>UKBB/GTEx, one allele</td><td>UK Biobank and GTEx variants</td><td>Experimental +</td><td>30,505</td></tr><tr><td>Sei classes</td><td>represented by one allele per locus Genomic regions spanning Sei sequence</td><td>Malinois Malinois</td><td>20,000</td></tr><tr><td>DHS index</td><td>classes DNase I hypersensitive sites</td><td>Malinois</td><td>20,000</td></tr><tr><td>Genomic windows</td><td>Random genomic windows</td><td>Malinois</td><td>20,000</td></tr><tr><td>Synthetic DNA</td><td>Uniformly random 2oo-bp sequences</td><td>Malinois</td><td>20,000</td></tr></table>

Table 2: Human-selected baseline strategies. Strategy names correspond directly to the labels in Fig. 2. DHS denotes a DNase I hypersensitive site, a genomic region of accessible chromatin. NMF denotes non-negative matrix factorization, which we used to summarize variation in DHS activity across cell types into 16 recurring regulatory patterns, referred to here as topics. Sei is a sequence-based regulatory model that assigns genomic sequences to regulatory classes. Oracle labels are activity values predicted by Malinois; real MPRA labels are experimentally measured activities. The mixed strategies combine sequences selected by these individual approaches in the proportions indicated.
<table><tr><td>Strategy</td><td>Description</td></tr><tr><td>DHS topic-weighted</td><td>Sequences were drawn from DNase I hypersensitive sites (DHSs), genomic regions associated with accessible chromatin. We used non-negative matrix factorization (NMF) to represent patterns of DHS activity across cell types as 16</td></tr><tr><td>DHS random</td><td>loading on these NMF topics. Sequences were drawn uniformly at random from the same collection of DHSs, without using the NMF topics to determine sampling frequency.</td></tr><tr><td>DHS component-stratified</td><td>DHSs were grouped according to the 16 NMF regulatory topics, and the library allocated an equal number of sequences to each topic rather than sampling according to their natural abundance.</td></tr><tr><td>SEI class-balanced</td><td>Sequences were drawn from genomic regions assigned to regulatory classes by Sei, a sequence-based model that classifies regulatory activity. Sampling was based on the Sei regulatory classes rather than directly on DHS activity.</td></tr><tr><td>SEI random</td><td>Sequences were drawn uniformly at random from the same Sei-annotated genomic regions, without balancing or weighting by regulatory class.</td></tr><tr><td>Random synthetic + oracle labels</td><td>Each 2oo-bp sequence was generated synthetically by choosing A, C, G, or T independently with equal probability at every position. Activity labels were then assigned by the Malinois oracle rather than measured experimentally.</td></tr><tr><td>Malinois training sequences + oracle labels</td><td>Sequences were sampled uniformly from the original Malinois training set, and their activity labels were generated by the Malinois oracle.</td></tr><tr><td>Malinois training sequences + real MPRA labels</td><td>The same type of sequences were sampled from the original Malinois training set, but the labels were the experimentally measured MPRA activities used to train Malinois rather than Malinois predictions.</td></tr><tr><td>½ DHS + ½ SEI</td><td>Half of the library was drawn using the DHS topic-weighted strategy and half using the Sei class-based strategy.</td></tr><tr><td>½ DHS + ½ synth</td><td>Half of the library was drawn using the DHS topic-weighted strategy and half consisted of uniformly random synthetic DNA sequences.</td></tr><tr><td>DHS stratified + ½ SEI</td><td>The library combined DHSs sampled to represent the NMF topics evenly with sequences sampled across Sei regulatory classes.</td></tr><tr><td>½ SEI + ½ synth</td><td>Half of the library consisted of sequences sampled across Sei regulatory classes and half consisted of uniformly random synthetic DNA sequences.</td></tr><tr><td>1-3 DHS + SEI + synthetic</td><td>The library contained equal contributions from DHS topic-weighted sequences, Sei class-based sequences, and uniformly random synthetic DNA.</td></tr><tr><td>DHS stratified + SEI + 1-3 synthetic</td><td>The library combined DHSs sampled to represent the NMF topics evenly with Sei class-based sequences and uniformly random synthetic DNA.</td></tr></table>

## Performance of frontier agents in a single-round test with no prior knowledge

We invited several frontier agents, Claude Opus 4.7 in Claude Code <sup>22</sup> (Claude), GPT-5.5 in Codex <sup>23</sup> (GPT), and Gemini 3.5 Flash in the Gemini Command Line Interface (CLI) <sup>24</sup> (Gemini), to play in MPRAbox, initially with just a single round of exploration $( M = 1 )$ . In this setting, the agents were told the physical meaning of the sandbox (see Supplementary Information for instructions to agents); they were welcome to search the scientific literature for ideas, write code to analyze data, and otherwise operate autonomously, but they could only submit a single library of $N = 5 0 , 0 0 0$ specimens for evaluation. For each frontier agent, we performed five independent replicates (each starting from scratch with no knowledge of the results from prior replicates). Our goal was to evaluate how and how well agents designed MPRA libraries without iterative feedback.

To provide a baseline for performance, we evaluated MPRA libraries with N = 50,000 based on 14 diferent human-chosen strategies, with five independent replicates for each strategy (Table 2, Figure 2a). The strategies reflected diferent expert views of what might make a library informative for training a model. Some strategies sampled naturally occurring regulatory DNA, either broadly or with deliberate weighting across known regulatory programs and sequence classes; others reused sequences from prior MPRA experiments, generated fully random synthetic DNA, or combined genomic and synthetic sources in the same library.<sup>14,25,26</sup> These libraries showed large diferences in performance, indicating a rich and complex design landscape (Figure 2a). Notably, libraries drawn from actual genomic DNA generally performed better than libraries consisting of fully synthetic sequences. In the initial single round experiments, the agents were not provided with prior knowledge of the results from human-generated MPRA libraries.

In this setting, Claude produced the best libraries, with a median performance of $r = 0 . 7 7 4$ across its five replicates. Each of Claude’s five libraries met or exceeded the mean performance of the best humanselected strategy $( r = 0 . 7 6 3$ across five replicate libraries). In contrast, Gemini and GPT had median performance of $r = 0 . 6 8 0$ and $r = 0 . 6 5 5$ , respectively; neither generated a library that exceeded the best human reference.

When we examined the agents’ lab notebooks, we observed a striking diference in what diferent agents believed would make an informative library. In all five replicates, Claude chose actual regulatory DNA taken from the human genome, reasoning that these sequences would preserve the natural surrounding context in which regulatory elements normally function. It sampled broadly across known classes of regulatory sequences based on ENCODE candidate cis-regulatory element (cCRE) annotations, <sup>27</sup> taking care to ensure “calibration” by giving additional weight to rarer classes rather than concentrating the library on the most common or obviously “interesting” elements:

(a)  
![](images/c59f699e0f4f867fdd57ba87485de415f0bb8cf08db7b1e4d2720a1ccc6b501e.jpg)  
(b)

![](images/2f1d4df04c5fa1e90b666739906d5ce440259b329a22aa06ede68e03d66aa39a.jpg)

  
![](images/e6b1db1bf7f09bd572bba58b248d66b75f36415b632cae29d616187b5faec9ae.jpg)  
Fig. 2 Human-designed reference panel and one-shot agent performance in MPRAbox. (a) Human-designed reference panel of massively parallel reporter assay (MPRA) library design strategies. $( L e f t )$ Performance of human-designed strategies, with five sampling seeds per strategy; points show perseed mean Pearson r across the held-out evaluation suite, ordered by mean performance. $( R i g b t )$ Per-set performance of the same strategies (rows) across the evaluation suite, with the mean shown in the rightmost column. Color indicates Pearson r. (b) One-shot library performance by model and condition, with the human-designed reference panel shown for comparison. Dashed lines, maximum, mean, and minimum performance of the human-designed reference panel. Each point represents one independently generated library; $\varkappa = 5$ runs per model and condition. (c) Sequence sources used by each one-shot run. $( L e f t )$ Each point represents one independently generated 50,000-sequence library; its x-position indicates the fraction of sequences in that library drawn from genomic sources. $( R i g b t )$ Specific sequence sources used to construct each library; filled cells indicate that the run used sequences from the indicated source. Rows are grouped by model and condition.

“A library of only 'interesting' elements teaches the model that everything is active. Negatives are critical for calibration.” – Claude Opus 4.7

GPT took the opposite approach in all five replicates. It built fully synthetic sequences that systematically varied short regulatory motifs, their spacing, and their frequency, reasoning that controlled perturbations would yield a cleaner training signal:

“Synthetic sequences help separate causalfeatures that are confounded in the genome: a motifcan be varied against many backgrounds, motifpairs can be swept across distance and orientation, and null backgrounds can establish what the model should ignore.” – GPT-5.5

While this was a reasonable strategy often used in regulatory genomics, in this instance it produced less informative libraries, likely because the synthetic perturbation series lacked the native genomic context captured by actual human DNA. After receiving the report from the sandbox, GPT recognized this limitation:

“If I had another shot, I would prioritize adding true genome-derived sequence contexts from broad cCRE/FANTOM/promoter annotations, especially matched positive/negative genomic neighborhoods, while retaining the synthetic motif perturbation series.” – GPT-5.5

Gemini, in contrast, used diferent approaches across its five replicates: genomic sequences, synthetic sequences, or a mixture of both. Accordingly, its performance was intermediate. Its underlying rationale leaned heavily on information theory:

“Unlike genomic libraries, which are often highly redundant andfilled with inactive or repetitive regions, a synthetic library can be engineered to maximize the Shannon entropy of regulatory grammar. Every sequence must be a designed experiment that teaches the model a specific rule.” – Gemini 3.5 Flash

## Performance in single-round testing with prior knowledge

We next repeated the single-round experiment, but gave agents some prior knowledge about the results from the human-selected MPRA libraries. Specifically, in addition to the standard task instructions, agents received a strategies.md file describing the 14 human-selected strategies used to construct the

libraries and the summary of the performance (correlation scores) for the libraries (as described above);   
the underlying library sequences were not provided.

Knowledge of how the various human-selected strategies performed changed how the agents designed their libraries, particularly for GPT and Gemini (Figure 2b,c; Supplementary Figure 2). Without prior knowledge, both had relied mostly on synthetic sequence designs; after seeing the human-selected strategies, they shifted strongly toward genomic sequence. Claude had already favored genomic regulatory DNA and therefore changed less.

Performance improved for all agents, but more for the agents that had performed less well in the absence of prior knowledge. Claude’s median performance increased from $r = 0 . 7 7 4$ without prior knowledge to $r = 0 . 7 8 1$ with prior knowledge; GPT increased from $r = 0 . 6 5 5$ to $r = 0 . 7 6 0 ;$ ; and Gemini increased from $r = 0 . 6 8 0 \mathrm { \ t o \ } r = 0 . 7 5 1$ . For reference, the best-performing human-selected strategy had a mean $r = 0 . 7 6 3$ across its five replicate libraries.

Notably, despite access to the same prior information, agents did not converge on a single strategy, and agents did not simply copy the best human-designed strategy. Their notebooks showed that they tried to infer why certain human strategies worked and then designed modifications around those conclusions:

“Given a single shot and no probing, I want a strategy that is strictly better than dhs\_topic on average, not a flashy bet.” – Claude Opus 4.7

![](images/9e9063d65e5f1220782d4b30fa669547e2b509906b7615a3ec3d881a41d17a67.jpg)

Fig. 3 Long-horizon agents use experimental feedback to refine library-design hypotheses. (a) Performance across 30 sequential MPRAbox experiments in two runs without prior knowledge of human-designed strategies (top) and two runs with this information (bottom). Annotations highlight key experiments, interpretations and resulting changes in each agent’s working theory of informative library design; boxes summarize the principal hypothesis developed during each run. Stars indicate the highest-performing library in each trajectory. Horizontal lines show the maximum, mean and minimum performance of the human-designed reference panel. (b) Performance of the best library from each 30- round run across the 14 held-out evaluation sets, grouped into empirical and oracle-labeled sets. Purple points indicate the four agent runs; gray intervals show the range ofthe human-designed reference panel for each evaluation set, with horizontal bars indicating the human mean.

## Agents refine hypotheses about MPRA library design in multi-round testing

We next invited Claude, the best performer in the single-round experiments, to participate in longhorizon play in MPRAbox, with M = 30 rounds. We ran four experiments: two in which the agent began without prior knowledge and two in which it was provided with prior knowledge of the humandesigned MPRA library strategies. The agent was instructed, before each round, to record its rationale for designing a given library in its lab notebook and, after each round, to interpret the report that it received about its performance before turning to the next round.

We evaluated each trajectory in two ways. Quantitatively, we asked whether the performance scores improved across rounds. Qualitatively, we examined whether the results led Claude to revise its hypotheses about library design and to design subsequent experiments that could distinguish among those hypotheses.

The four 30-round agent runs produced somewhat higher performance scores than single-round testing, and all four runs exceeded the strongest human-selected strategy (Figure 3a,b; Supplementary Figures 4 and 5). But the quantitative gains over the single-round tests were modest. The purpose of multi-round play was not primarily to characterize hill-climbing performance toward a higher score, but to let us observe how an agent’s scientific reasoning changed in light of iterative feedback over time, including how it made strategic use of a known, finite experimental horizon.

In the early rounds, agents often explored broadly, comparing random DNA, genomic DNA (including previously annotated regulatory sequences), and synthetic designs. Some of these experiments produced results that directly contradicted their expectations. In one run, for example, the agent expected a library of random DNA to perform near zero, but instead observed a surprisingly high performance score. It wrote:

“This is a strong contradiction to my initial prediction. A ‘library with no biology’ was supposed to score near zero... my updated theory is: a substantial fraction of the eval signal is composition-driven and ‘free,’ and the library-design problem is to provide informative sequences that go beyond composition to teach the model true regulatory grammar.

This result led the agent to revise its working hypothesis about what information a useful library needed to contain and to design subsequent experiments around the contribution of synthetic sequences versus regulatory sequence content.

As the runs progressed, the agent generally moved from broad comparisons toward more controlled tests in which it changed one aspect of the library at a time. This strategy was not specified in the instructions. In one run, the agent began exploring whether sequences from other species, including chicken, could improve the library. It later recognized that one comparison had changed the human regulatory sequence component at the same time as it included the chicken sequences, making the result dificult to interpret:

“Operational lesson. Always isolate one variable at a time when the result will be interpreted as ‘X works/doesn’t work’... [The earlier experiment] moved cCRE AND chicken, leading to the wrong conclusion.”

The four runs arrived at diferent high-performing libraries rather than converging on a single design. Multi-round play therefore allowed us to observe not only whether agents found better libraries, but how they used experiments to build, test, and revise a theory of what makes a library informative. Across the four runs, the agents developed diferent hypotheses about what makes a training library informative. The first agent, which began without prior knowledge, compared synthetic motif-injected sequences with naturally occurring regulatory DNA and concluded that motifs were most informative in their native genomic context. The second agent, also without prior knowledge, varied the representation of common and rare regulatory classes and found that enriching rare classes could improve performance, but only while maintaining broad coverage of all classes of sequences. The third agent, with prior knowledge of the human-selected strategies, began from a genomic regulatory library and explored cross-species augmentation; adding chicken sequences improved performance, whereas more distant species did not produce larger gains. The fourth agent, also with prior knowledge, started from the strongest DHSbased human strategy and tested modifications to it, ultimately finding that stratifying sequences by GC content further improved performance.

Table 3: Invented rules used in the dry MPRAbox oracles. Each dry oracle replaced the biological Malinois oracle with an invented sequence-to-activity rule hidden from the agent. As in MPRAbox, the agent submitted a training library and received a single aggregate performance score reporting how well a model trained on that library generalized to held-out sequences. Full definitions of the hidden functions are provided in Methods.
<table><tr><td>Oracle</td><td>Hidden rule</td></tr><tr><td>GC balance</td><td>Rewards particular patterns of overall nucleotide composition, such as so% GC content or equal representation of all four nucleotides.</td></tr><tr><td>Alternating purine</td><td>Rewards sequences in which purines (A/G) and pyrimidines (C/T) alternate between successive positions.</td></tr><tr><td>English words</td><td>Converts pairs of nucleotides into letters using a hidden cipher and rewards sequences whose decoded text contains English words.</td></tr><tr><td>Compression</td><td>Rewards sequences according to how compressible they are, distinguishing highly repetitive, highly irregular, and intermediate sequence structure.</td></tr><tr><td>Prime counts</td><td>Rewards sequences in which the count of a specified nucleotide is a prime number.</td></tr><tr><td>Fibonacci positions</td><td>Rewards particular nucleotides at positions whose indices follow the Fibonacci sequence.</td></tr><tr><td>Game of Life</td><td>Converts the sequence into a two-dimensional grid, evolves it according to Conway&#x27;s Game of Life, and rewards particular properties of the resulting dynamics.</td></tr><tr><td>Modular cross</td><td>Rewards sequences whose nucleotide counts satisfy a specified modular-arithmetic relationship</td></tr><tr><td>Collatz</td><td>Derives an integer from a property of the sequence, applies the Collatz iteration, and rewards sequences according to the resulting stopping behavior.</td></tr><tr><td>RLE / sine / XOR</td><td>Rewards sequences according to hidden mathematical functions based on run-length encoding, position-specific sine calculations, or bitwise XOR operations.</td></tr><tr><td>Substring</td><td>Rewards occurrences of a specified short nucleotide sequence.</td></tr><tr><td>Position 1oo</td><td>Rewards particular nucleotides at one specific position in the sequence.</td></tr><tr><td>Exact nucleotide count</td><td>Rewards sequences whose count of a specified nucleotide is close to a particular target value.</td></tr><tr><td>Parity</td><td>Rewards sequences according to whether the count of a specified nucleotide is even or odd.</td></tr></table>

## Dry oracles with invented rules test the limits of agents’ logical inference

The MPRAbox experiments above ask agents to reason in a domain where they can draw on extensive prior knowledge of regulatory biology. We next asked whether Claude could discover rules it had no reason to expect (Figure 4a). We kept the basic structure of MPRAbox, again asking the agent to design an optimal library for training a predictive sequence-to-activity model, but replaced its damp oracle with 14 diferent dry oracles whose scoring rules were invented by us and hidden from the agents (Figure 4a; Table 3). Because we knew the true rule but the agents did not, we could distinguish an agent that simply learned how to improve the score from one that actually inferred the rule producing it.

The invented rules ranged from familiar sequence properties to deliberately unexpected mathematical and computational constructions. For example, one rule rewarded sequences containing exactly 50 cytosines, so changing overall base composition could move the score in the right direction even before the exact target was discovered. A second rule ignored most of the sequence and rewarded particular bases only at Fibonacci-numbered positions. A third rule used a hidden cipher in which pairs of DNA bases encoded ordinary letters, with higher scores for sequences whose decoded text contained English words. Across the full set of 14 rules, the rules were designed to require very diferent kinds of hypotheses, from ordinary sequence statistics to positional and symbolic structure (Table 3).

We also varied how the task was presented to the agent. In one framing, the agent was told that it was designing regulatory DNA for an MPRA library. In another framing, the agent still designed sequences over $\{ A , C , G , T \}$ , but was told only that it was optimizing against a black-box scoring system, with no reference to MPRA or gene regulation. Finally, in a third framing, we replaced {A, C, G, T} with {0, 1, 2, 3}, removing even the suggestion that specimens represent DNA. The hidden rules themselves were unchanged across these framings, allowing us to ask how the presentation shaped the agents’ hypotheses.

Across the 14 rules, the framing of the task changed both how the agent investigated the problem and, in some cases, how much quantitative progress it made (Figure 4b). When told that it was designing regulatory DNA, the agent immediately tested hypotheses inspired by biology, including known regulatory elements and transcription-factor motifs. When that context was removed but the alphabet remained $\{ A , C , G , T \}$ , the agent focused more on nucleotide frequencies and short sequence patterns. When the DNA alphabet was replaced by {0, 1, 2, 3}, it explored still simpler properties of the strings, such as just the frequency of each symbol. These diferent starting points sometimes mattered for performance. In some cases, biological priors could direct the agent toward sequence properties that happened to correlate with an invented rule, giving it a useful proxy even when it did not understand why the proxy worked.

  
![](images/df94cbfb1c86a061a63c4d68ef0d25981db401830d6ae33297ade82664c64bdd.jpg)

![](images/1ab30c2d0674f504b8c88746f88c130defe912b4bffb151c5ca38fd2a4b40415.jpg)  
Fig. 4 Dry oracle experiments test rule discovery. (a) Schematic of the dry oracle experiments. Agents iteratively design sequence libraries and receive performance feedback from one of $\dot { \mathbf { \Omega } } _ { \mathbf { I } 4 }$ hidden, invented rules that serve as the oracle in this version of MPRAbox. (b) Performance across the $_ { \textrm I 4 }$ rules under MPRA-framed, unframed, and symbolic task presentations. Gray bars, random-library baseline. Colored symbols, agent performance. Matrix at right, whether the agent correctly inferred the hidden rule in each framing.

Framing shaped exploration, but it did not reliably produce rule discovery. We assessed rule discovery from the agent’s stated hypotheses rather than from whether it reached a particular quantitative perfor mance threshold. For the hidden English-word cipher, for example, none of the three framings led the agent to consider that pairs of characters might encode letters or that the decoded sequence might contain language. For the Fibonacci rule, the agent instead concluded that “The model isn’t learning motif syntax. It’s learning dinucleotide composition statistics.” That explanation captured enough structure in the feedback to guide some quantitative improvement, but it was not the rule generating the observa tions. Thus, these experiments surface the distinction that motivates the sandbox framework: learning how to improve a quantitative score is not necessarily the same as learning how a system works.

## CodonBox: a science sandbox for invented biological rule discovery

We next created CodonBox to ask whether an agent could infer the rules of an unfamiliar genetic system from experiment alone. CodonBox uses a simplified model of how DNA instructions in genes give rise to three-dimensionally folded proteins. We sought to explore how much an agent, again Claude, could infer about hidden rules linking DNA sequence to protein folding.

In standard biology, (i) a stretch of genomic instructions (in the 4-letter alphabet of DNA nucleotides) is directly transcribed into a matching RNA sequence (in an equivalent 4-letter alphabet of RNA nucleotides); (ii) the RNA sequence is then translated into a protein sequence, with consecutive nonoverlapping groups of 3 nucleotides, called codons, specifying one of 20 possible amino acids, according to a look-up table (called the genetic code); and (iii) the protein sequence folds into a specific threedimensional structure, based on the order and chemical properties of the amino acids.

CodonBox turns this familiar biological pipeline into a family of invented genetic worlds. In CodonBox, we generalized and modified the process as follows: (i) genetic instructions are written as a nucleotide sequence in an alphabet with j letters; (ii) the instructions are translated into a protein sequence, with consecutive non-overlapping codons of k nucleotides being converted into one ofm possible amino acids, according to a look-up table; and (iii) the protein sequence folds into the conformation that maximizes the number of ‘favorable’ interactions between non-consecutive amino acids.

We explored nucleotide alphabets of size j = 4, 6, or 8, and codon lengths of size $k = 2 , 3 ,$ or 4. The oracle used a hidden codon table to translate the codons into m amino acids. For simplicity, we used Dill’s classic two-dimensional model, in which there are only m = 2 amino acids, called hydrophobic (H) or polar (P), and the chain of amino acids is laid out along the edges of a square lattice, with amino acids at the vertices. <sup>28</sup> In Dill’s model, a protein folds into the conformation that has the largest number of “favorable” interactions, which occur whenever two H’s that are not consecutive in the protein sequence are adjacent in the square lattice. Given a submitted specimen, the oracle in CodonBox determines the protein sequence and finds the most favorable fold. The agent sees none of these intermediate steps: the oracle simply reports a performance score consisting of the number of favorable interactions in the best fold.

CodonBox: a science sandbox for biological rule discovery  
  
![](images/3db6560908c4e2006c02081c710939dcc2b2c18b88a0ac4b2be0d9f083dd48bd.jpg)

![](images/7d0a7385e260082d050b0d0122acbd99c9216c7225c04125f3b5c3ca5ff385fa.jpg)

![](images/8224f8ea63a5e6ef5a47f3466d5124d0489dbd24cbe8407dc14214437229104b.jpg)  
  <sub> </sub>

![](images/da95598aeab1e01878607a63cff76ffc84082514a95952ce4b6d8c911245f8d4.jpg)  


![](images/498361e3df36512bc4877695f9128ba7431968951892f1630be1f544098d9829.jpg)  
<sup></sup> <sub></sub> 

  
![](images/9fc2e264bc90852baf3d631514411d92c6f28cc77dc5d8b91ba2b31825ff2664.jpg)  
Fig. 5 CodonBox tests rule discovery in invented biological worlds. (a) Schematic of the Codon-Box oracle. A submitted DNA-like sequence is parsed into codons and translated by a hidden worldspecific rule into a 16-residue hydrophobic/polar (H/P) chain. The H/P chain is then folded on a two-dimensional lattice using a fixed HP folding model. Fitness is defined as the number of favorable non-consecutive H-H contacts in the optimal fold. The hidden translation rule varies across Codon-Box worlds, whereas the folding model is held fixed. (b) CodonBox instantiated as a science sandbox. In each round, the agent nominates a candidate sequence as its action. The sealed oracle applies the hidden translation rule and fixed folding model, then returns only the resulting fitness score as feedback. Across rounds, the agent uses this sequence-to-fitness feedback to revise its hypotheses about the hidden genetic rules.

The agent engaged in multi-round play, submitting one nucleotide sequence of its choice on each of 500 rounds. We did not tell the agent that the sequence contained codons, how those codons might be translated into amino acids, or what rules governed protein folding. The oracle simply returned the single quantitative performance score for the final folded product. The agent had to infer the hidden rules by changing the sequence and observing how that final score changed.

The structure of this model makes quantitative maximization relatively easy even when the underlying rules remain unknown. Consider a sequence made from a single repeated character, such as $A A A A A = \tan \angle C$ Whatever the hidden codon length $k ,$ the oracle reads that sequence as repetitions of the same codon. If that codon maps to $H ,$ the translated sequence consists entirely of H residues. For the sequence lengths used here, such an all-H chain can achieve the maximum possible number of favorable interactions. Because the nucleotide alphabet contains only a few possible characters, the agent can test repeated character sequences and quickly stumble onto one whose repeated codon maps to H. Indeed, agents reached the maximum performance score within the first 10 rounds in every run. Once it had found a sequence with the maximum score, it could use the remaining experiments to ask a harder question: what rules caused diferent sequences to produce diferent scores?

We first tested whether the agent could infer the structure ofthe hidden world as the size ofthe underlying codon table increased. We varied codon length k while holding the nucleotide alphabet fixed $\mathfrak { a t j } = 4$ This gave codon tables with 16, $6 4 \div$ and 256 possible entries for $k = 2 , 3 ,$ , and 4, respectively. The agent correctly inferred the codon length in all three cases. When $k = 2$ or 3, a simple strategy worked: the agent tested individual codons, recorded their efects, and gradually filled in the table. With $k = 4 ,$ , however, the agent identified the four-nucleotide codon structure and learned the behavior of many individual codons, but it did not recover the complete table. Thus, the agent could infer that sequences were parsed into codons, but exhaustive codon-by-codon mapping became less efective as the table grew.

We next tested whether codon-structure inference would hold as the alphabet expanded and the table grew further. Holding codon length fixed at $k \ = \ 3$ , we expanded the nucleotide alphabet from $j ~ =$ 4 to $j = 6$ or $\begin{array} { r } { 8 , } \end{array}$ , which increased the number of possible codons from $6 _ { 4 }$ to 216 or 512. As the table grew, testing codons one at a time became less useful. With $j = 8 ,$ , the agent still discovered that the oracle used three-nucleotide codons and mapped a substantial fraction of the table. With $\displaystyle { j = 6 } ,$ , however, it never discovered the codon structure at all. Instead, it built an increasingly elaborate theory around individual nucleotides, repeated runs, and short sequence patterns. Many of these patterns genuinely predicted changes in the performance score, but they did not reveal the simpler codon structure that generated them.

The most revealing results came when the agent stopped treating each codon as a separate case and instead asked how the codon table was organized. We returned to j = 4 and k = 3 and changed the table so that only the middle nucleotide of each codon determined H or P, with the two outer positions having no efect. The agent repeatedly changed those outer positions, observed that the score stayed the same, and correctly inferred that they were silent. It then identified which middle nucleotides mapped to H and which mapped to P. In this setting, a small number of controlled perturbations revealed a general rule that applied across the codon table.

We then tested whether the agent could infer rules involving interactions between codon positions. We again used j = 4 and k = 3, but now the middle position was silent and the two outer positions jointly determined H or P. The efect of one outer nucleotide therefore depended on the other. The agent initially tried to assign simple efects to individual nucleotides. One experiment broke that theory: “ACDACD... = 0! Wow. So ACDACD repeated gives 0, but AC alone gives 9, AD alone gives 9, CD alone gives 9... This breaks my ‘B is poison’ theory. Something more complex going on.” The agent changed its strategy. It began testing combinations of positions and ultimately recovered the interaction rule. In this case, collecting more individual examples did not solve the problem; asking a better experimental question did.

Finally, we tested whether the agent could recover a general rule when multiple sources of complexity were combined. We combined several challenges in one system with j = 6 and k = 4, giving 1,296 possible codons. We made one codon position silent and made the remaining positions interact to determine H or P. The agent eventually inferred that the oracle parsed the sequence into groups of four, but it never identified the silent position or recovered the general mapping rule. As the run continued, it increasingly returned to cataloging individual codons that had worked before and reusing them. The hardest Codon-Box world therefore sharpened the distinction between finding high-scoring sequences and discovering the rules that make them high-scoring.

## Discussion

Science advances not merely by finding shortcuts to predict an outcome. While ancient astronomers could use epicycles to fit planetary motion, their models revealed nothing about the underlying cause. True scientific progress comes from inferring the fundamental rules, like gravity, that explain why a system behaves as it does. Science sandboxes allow us to measure whether AI agents are merely building modern epicycles to optimize a benchmark score, or are genuinely capable of discovering the governing rules of nature.

Our empirical findings across regulatory genomics and invented biological systems reveal a sharp divergence between agents’ ability to optimize a target metric and their ability to deeply understand the underlying system. In MPRAbox, agents achieved high quantitative performance, matching or exceeding human-designed reference strategies by relying on familiar biological priors present in their pretraining data. However, this apparent capability proved brittle when even the best-performing agent in the standard MPRAbox experiments, Claude, encountered synthetic MPRAbox rules and non-canonical CodonBox translation systems that fell outside familiar biological priors. While agents continued to optimize experimental scores, their ability to infer the underlying rules collapsed whenever those rules required positional, mathematical, or combinatorial logic outside standard biological intuition. Current AI systems excel at searching within established conceptual frameworks, but struggle with genuine scientific induction when forced to deduce rules they have no prior reason to anticipate. This distinction is dificult to capture with fixed benchmarks that report only a final score.

Evaluating whether an agent has inferred a system's governing rules requires looking beyond its quantitative yield to inspect its qualitative reasoning traces. Even when quantitative performance has saturated, much can still be learned from how an agent approaches a problem. <sup>29</sup> In a science sandbox, this process is preserved in the agent’s lab notebook. In our experiments, notebooks revealed a recurring weakness in experimental strategy: agents often exhausted their budgets through brute-force search, while stronger trajectories used structured exploration followed by targeted tests. Characterizing this failure mode should make it possible to design harnesses that steer future agents toward more efective search strategies. Further, while we directly inspected the agents’ notebooks, this evaluation step can be automated by using independent AI judges trained to score hypotheses against the ground-truth rules of the oracle. Automating qualitative evaluation removes the human bottleneck, enabling the rapid, high-throughput benchmarking of hundreds of agent architectures across an imaginably infinite range of potential science sandboxes.

There are important caveats to the framework presented here, particularly with respect to the choice of oracle within a sandbox. “Wet” physical experiments are, of course, often expensive, time-consuming, and noisy; “damp” predictive models provide feedback at larger scale, but inherit the data biases and assumptions of their underlying models; and “dry” invented rules ofer an exact ground truth at the expense of real-world physical complexity. In our experiments, we sought to make the evaluation as faithful and robust as possible, for example by testing MPRAbox libraries across hidden empirical and diverse oraclelabeled evaluation sets. More broadly, the spectrum of wet, damp, and dry oracles enables a wide range of evaluations of scientific capability, from settings where the ground truth is known to those where genuine novel scientific discovery is possible.

A strength of the science sandbox is its conceptual simplicity: it requires only an agent (which selects specimens to be assayed) and an oracle (which returns a report about the results of the assay). This minimal architecture allows investigators to instantiate sandboxes across a spectrum of settings, spanning physical wet labs, predictive surrogates, and synthetic rule systems. Via BroadBox, we intend to release new sandboxes to support community-wide evaluation, and we encourage researchers across disciplines to construct and release sandboxes in their own fields. Such evaluations will enable the study of science itself at scale, providing a path towards understanding and ultimately expanding scientific capability.

## Methods

## Definition of a science sandbox

A science sandbox is a closed-loop setting for evaluating how an agent learns through experiment. The sandbox specifies a class of specimens, one or more assays that can be applied to those specimens, and a sealed oracle that performs the assays and returns a report. On each round M, the agent selects one or more specimens to test; the oracle applies the assay and returns a report; and the agent may use that report to choose its next experiment. The sandbox may run for one round or many rounds, allowing both final performance and the agent’s sequence of hypotheses, experiments, interpretations, and revisions to be evaluated.

Oracles difer in how they generate assay results. Wet oracles perform physical experiments, damp oracles use computational models trained on empirical data, and dry oracles apply invented rules specified by the sandbox designer. The oracle provides a mechanism for verification: unlike software or mathematics, where candidate solutions can be checked instantly through unit tests or formal proofs, empirical sciences like biology and chemistry generally lack scalable, automated verifiers. This separation allows the same experimental loop to be instantiated with diferent sources of feedback while keeping the underlying rules hidden from the agent.

## The MPRAbox task and sandbox

MPRAbox frames MPRA library design as a sequential decision problem. In each round $t ,$ an autonomous agent produces a library $L _ { t }$ of 50,000 DNA sequences, each 200 bp long over the alphabet $\{ A , C , G , T \}$ . The objective is to compose libraries that generate informative training data for sequenceto-activity models that generalize across regulatory sequence distributions and cell types, including contexts not directly measured.

The agent submits candidate libraries through a filesystem-based interface and is instructed to treat prepare.py as a wet lab collaborator: a sealed process that accepts a proposed library and returns aggregate results. For each round, the agent writes a program, generate.py, that constructs the library and saves its sequences. The agent then invokes prepare.py, a sealed script which checks that the library contains exactly 50,000 valid 200 bp DNA sequences, sends those sequences to the in silico MPRA oracle, and returns evaluation metrics. Each one-shot submission consists of a single call to prepare.py; each long-horizon agent run consists of successive calls to prepare.py.

The Malinois scoring and downstream model-training pipeline are treated as sealed.<sup>14</sup> The scoring pipeline and oracle weights are present in the working environment, but the agent is instructed not to inspect or reverse-engineer them; analysis of all session logs reveals that the agents studied here successfully obeyed these instructions.

After prepare.py completes, it returns an anonymized report containing the Pearson correlation for each of the 14 evaluation sets and an overall performance score equal to their mean. Evaluation-set identities, sequences, labels, genomic coordinates, and individual sequence-level predictions are not returned to the agent.

The two information conditions difered only in what the agent knew about the human-selected reference strategies. Without prior knowledge, the agent received only the task specification and submission interface. With prior knowledge, it additionally received descriptions of the 14 human-selected strategies and their performance scores; it did not receive the underlying library sequences.

In the one-shot regime, each agent generated and submitted a single library of 50,000 sequences and received one report. We ran five independent replicates for each model under each information condition, yielding 30 one-shot runs in total: three models × two conditions × five replicates. The two conditions difered only in whether the agent received the results of the human-selected reference strategies before designing its library.

In the long-horizon regime, the agent could submit 30 successive libraries, receiving the sandbox report after each round before designing the next. We used Claude Opus 4.7, the strongest model in the oneshot experiments, and ran four independent 30-round runs: two without prior knowledge of the humanselected strategies and two with that prior knowledge. We limited long-horizon experiments to one model because of computational cost and throughput.

Agents maintained two forms of state: an append-only notebook.md, a free-form running record of hypotheses, observations, and plans, and a directory of agent-written skill files documenting reusable procedures. The long-horizon agents reread their notebooks and skills at the start of each round.

## In silico MPRA oracle

Malinois, a deep convolutional neural network trained on a corpus of 776,474 200 bp sequences assayed in K562, HepG2 and SK-N-SH cells (achieving a reported Pearson r ≈ 0.88 across the three cell types), was used as an in silico oracle of MPRA measurement.<sup>14</sup> For each submitted 200 bp sequence, Malinois returns predicted activity in the three cell types. The published checkpoint was downloaded directly from the original publication.<sup>14</sup> As described in the original publication, each 200 bp candidate sequence was embedded in the fixed 600 bp reporter-vector context required by the model, and predictions were averaged across the forward and reverse-complement orientations.

## Downstream model training and scoring

Each submitted library was used to train a fresh sequence-to-activity model from random initialization. The model used the same architecture as the Malinois oracle, but no weights were shared with the oracle. Malinois supplied activity labels for the submitted library; the downstream model was trained only on those 50,000 submitted sequences and their Malinois-generated labels.

The model takes a one-hot encoded 600 bp input, formed by embedding the 200 bp candidate sequence in the fixed reporter-vector context used by Malinois. The architecture consists of a convolutional sequence encoder followed by a shared fully connected layer and cell-type-specific output branches. The convolutional trunk contains three one-dimensional convolutional layers with 300, 200 and 200 filters, with kernel widths of $^ { 1 9 , }$ 11 and $^ { 7 , }$ respectively, and batch normalization in the convolutional network. The shared representation is passed through a fully connected layer with 1,000 hidden units and dropout. The model then branches into separate output heads for $\mathrm { K } \varsigma 6 2 .$ , HepG2 and SK-N-SH; each branch contains three fully connected layers with $_ { \mathrm { I 4 O } }$ hidden units and returns predicted MPRA $\log _ { 2 }$ fold-change for one cell type.

For each submitted library, the architecture, optimization procedure and evaluation pipeline were fixed. Training used the Malinois objective inherited from the released implementation, combining an $\mathrm { L _ { 1 } }$ activity loss with a KL-divergence term. Optimization used Adam with learning rate $3 . 2 7 \times 1 0 ^ { - 3 } , \beta \ =$ $( 0 . 8 6 6 , 0 . 8 7 9 ) , \varepsilon \ = \ 1 0 ^ { - 8 }$ , weight decay $3 . 4 4 \times 1 0 ^ { - 4 }$ and AMSGrad. The learning-rate schedule was CosineAnnealingWarmRestarts with $T _ { \mathrm { 0 } } = 4 0 9 6 , T _ { \mathrm { m u l t } } = 1 \mathrm { a n d } \eta _ { \mathrm { m i n } } = 0$ . Models were trained with batch size 512 for up to 200 epochs, with early stopping on a 10% random validation split.

Forward/reverse-complement averaging was applied only when Malinois generated labels, including labels for submitted libraries and oracle-labeled evaluation sets. Downstream models trained inside MPRAbox were trained on single-strand inputs without reverse-complement augmentation and evaluated using a single forward pass. All oracle inference and downstream model training were run on an NVIDIA DGX Spark.

## Evaluation suite

Trained models were evaluated on $_ { \textrm I 4 }$ held-out evaluation sets derived from nine source distributions (Table 1). Five sets used empirical MPRA measurements as labels and nine used Malinois-generated labels. The empirical-label sets provide direct evaluation against measured regulatory activity, while the oracle labeled sets extend the suite to sequence distributions not represented in the experimental data.

The evaluation suite included held-out reporter sequences from the Gosai et al. MPRA experiment,<sup>14</sup> UK Biobank and GTEx variant sequences,<sup>13</sup> DNase I hypersensitive sites from the Meuleman et al. DHS index,  regulatory sequence classes assigned by the deep learning model Sei, random genomic windows and fully synthetic random sequences. Chromosome-level holdout was enforced during construction. Evaluation sets drawn from the genome used chromosomes excluded from Malinois training, including chromosomes 7, 13, 19, 21 and X. UK Biobank and GTEx variant sets were constructed as 200 bp windows centered on candidate causal variants. Depending on the evaluation set, either a single allele per locus or both reference and alternate alleles were retained. Evaluation set identities were hidden from the agent. Performance was computed as Pearson correlation between predicted and labeled MPRA log fold-change across the included sequences.

## Human-curated baseline strategies

We constructed 14 human-curated library-design strategies as a reproducible reference panel. Each strategy defined a sampling rule rather than a fixed library; for each rule, we generated five independent libraries using distinct sampling seeds, so that replicates difered only in the specific sequences drawn under the same design procedure. The strategies sampled from four sequence sources: accessible chromatin regions from the DHS index, regulatory sequence classes from Sei, fully synthetic random DNA, and previously assayed MPRA sequences from the Malinois training set.

DHS-based strategies sampled from the DNase I hypersensitive site index, a genome-wide catalog of accessible chromatin regions.  The DHS index includes non-negative matrix factorization (NMF) topic annotations that summarize accessibility patterns across biosamples into 16 regulatory programs. We used these annotations to define two structured sampling schemes. In topic-weighted sampling, DHS elements were sampled with probability proportional to their NMF topic loadings. In topic-stratified sampling, each element was assigned to its maximum-loading topic, and equal numbers of sequences were sampled from each of the 16 topics. Uniform DHS sampling treated all DHS elements as equally likely.

Sei-based strategies sampled from genomic regions annotated by Sei regulatory sequence class.<sup>30</sup> Sei regions were sampled either in proportion to class frequency or with equal allocation across regulatory classes. Synthetic random libraries were generated by drawing each nucleotide independently from a uniform distribution over $\{ A , C , G , T \}$ . Prior MPRA libraries were sampled uniformly from the Malinois training MPRA sequence set and evaluated either with Malinois-generated labels or with the original empirical measurements. The full panel also included two- and three-source mixtures combining DHS, Sei and synthetic sequences. The 14 human-curated design strategies are listed in Table 2.

Strategies were evaluated at library sizes of 10,000, 25,000, 50,000, 100,000, 150,000, 200,000 and 300,000 sequences. The primary comparison point was 50,000 sequences, matching the library size used for agent submissions.

## Agent configuration and regimes

We instantiated three model-harness systems using each provider’s native coding-agent environment: Claude Opus $4 { \cdot } 7$ in Claude Code, ${ \mathrm { G P T } } { \cdot } { 5 } { . } { 5 }$ in Codex CLI, and Gemini $3 { \cdot } 5$ Flash in Gemini CLI. All three operated in a sandboxed Linux environment with access to a shell, Python, the filesystem, and the internet, with permissions configured to allow autonomous tool use; otherwise, we used the default parameters of the respective provider-native harnesses. We introduced no third-party agent scafold.

## Dry oracles with invented rules in MPRAbox

We constructed $_ { \textrm I 4 }$ invented rules for use as dry oracles in MPRAbox. Each dry oracle replaced Malinois as the function that assigns activity labels to sequences while preserving the same basic library-design task. Given a submitted library of N 200-bp sequences, the dry oracle assigned three synthetic activity values to each sequence, mirroring the three-cell-type output format of Malinois. These sequence-level values were hidden from the agent. The $_ { \textrm I 4 }$ rules spanned nucleotide composition, short sequence patterns, positional rules, compressibility, number-theoretic relationships, cellular-automaton dynamics, and a hidden cipher that converted nucleotide pairs into English letters (Table 3).

To score a submitted library, we represented each sequence by its normalized 6-mer frequencies and trained a three-output Ridge regression model $( \alpha = 1 . 0 )$ on the submitted sequences and their oraclegenerated labels. We then evaluated this model on the same $_ { \textrm I 4 }$ held-out sequence sets used in MPRAbox, after relabeling those sequences with the same dry oracle. For each evaluation set, we calculated the Pearson correlation between the model predictions and the hidden oracle labels and averaged across the three outputs; we then averaged across the $_ { \textrm I 4 }$ evaluation sets. The agent received only this final aggregate performance score, not the sequence-level oracle labels, the individual output values, or the hidden scoring rule.

For each evaluation set, performance is measured as the Pearson correlation between predicted and oraclegenerated labels, averaged across the three output dimensions. For each library, we report the mean across the $_ { \textrm I 4 }$ MPRAbox evaluation sets.

We ran Claude Opus $4 { \cdot } 7$ on each of the $_ { \textrm I 4 }$ dry oracles under three task framings. In the MPRA-framed condition, the agent was told that it was designing regulatory DNA for an MPRA library. In the unframed condition, the agent still designed sequences over $\{ A , C , G , T \}$ , but was told only that it was optimizing against a black-box scoring system, with no reference to MPRA or gene regulation. In the symbolic condition, we replaced the DNA alphabet with {0, 1, 2, 3}. The hidden scoring rule was identical across the three framings. We ran each oracle-framing combination independently for 30 rounds.

## CodonBox implementation

CodonBox places the agent in invented biological worlds, each governed by a concealed genetic system whose rules are unknown to the agent but yield a measurable fitness readout. On every round the agent (Claude Opus $4 { \cdot } 7$ in a minimal harness, see https://github.com/asr2210/science-sandbox) submits one candidate sequence. A sealed oracle translates that sequence under a world-specific hidden rule, folds the resulting product with a fixed physical model, and returns a single fitness number and nothing more. Neither the translation rule nor the folding step is visible to the agent, so it can only probe each world through its input–output behavior and must deduce the mapping from sequence to fitness by experiment alone. Since the true rule of each world is set in advance by us but withheld from the agent, the setup separates agents that actually reconstruct the governing rule from those that only accumulate a list of high-scoring sequences.

The physical substrate is Dill's HP model of protein folding, <sup>28</sup> which reduces a protein to a string of hydrophobic (H) and polar (P) residues. Every translated product in CodonBox is a 16-residue $H / P$ string that folds on a two-dimensional square lattice, and its fitness is the count of favorable nonconsecutive H–H contacts in the lowest-energy fold. We enumerated this value for all $2 ^ { 1 6 } \ = \ 6 5 { , } 5 3 6$ possible 16-residue H/P strings ahead of time, which fixes the folding physics while letting the genetic layer difer from world to world. On this lattice no 16-residue chain can exceed 9 such contacts, so the same fitness ceiling of 9 holds in every world.

Each hidden rule defines how an input string is segmented into codons and how each codon becomes an H or P residue. Varying these rules gave us eight worlds that difer along four axes: codon length (2, 3, or 4 characters), alphabet size (4, 6, or 8 characters), whether a codon contains silent positions that leave its residue unchanged, and whether the informative positions act additively or interact. A control world mirrors familiar genetics, with a four-character alphabet, three-character codons, and independent contributions from all three positions. Two worlds are built so that changing one character at a time reveals little: in one, only a codon's central position sets the residue and the two outer positions are silent; in the other, the two outer positions jointly determine the residue through a non-additive interaction while the center is silent. The hardest world layers several of these features together, pairing a six-character alphabet with four-character codons, a silent position, and a three-position joint dependence. Each world's codon-to-residue table was built deterministically from its parameters under a fixed seed and balanced so that exactly half of the possible codons map to H and half to P. For each world, we ran Claude Opus 4.7 once, for 500 design rounds.

## Data and code availability

All code used to produce the sandboxes presented here is available at https://github.com/asr2210/science-sandbox.

## Funding

This work was made possible by funding from the Ladders to Cures Scientific Accelerator of the Broad Institute of MIT and Harvard and the Howard Hughes Medical Institute.

## Competing Interests

P.C.S. holds several patents related to diagnostic technologies and is a cofounder and equity holder in Delve Biosciences and Lyra Labs, a board member and equity holder in Polaris Genomics, and an equity holder of NextGenJane. P.C.S. was formerly a co-founder of Sherlock Biosciences and a board member of Danaher Corporation, until December 2024. S.N.B. reports interests in Amplifyer Bio, Catalio Capital, Danaher, Earli Inc., Impilo Therapeutics, Matrisome Bio, Ochre Bio, Pictet, Port Therapeutics, Ropirio Therapeutics, Satellite Bio, Sunbird Bio, Vertex Pharmaceuticals, and Xilio Therapeutics. S.N.B.’s interests are reviewed and managed under MIT’s policies for potential conflicts of interest. S.S. serves on the scientific advisory board of Waypoint Bio, Deepcell, Novo Nordisk, and is an advisor for Dewpoint Therapeutics.

## Acknowledgements

We thank Debora Marks and Robert Langer for insightful discussions that helped shape this study.

## References

1. Gao, S. et al. Empowering biomedical discovery with AI agents. arXiv [cs.AI] (2024). doi:10.48550/arXiv.2404.02831.

2. Gottweis, J. et al. Accelerating scientific discovery with co-scientist. Nature (2026). doi:10.1038/s41586-026-10644-y.

3. Swanson, K., Wu, W., Bulaong, N. L., Pak, J. E. & Zou, J. The virtual lab of AI agents designs new SARS-CoV-2 nanobodies. Nature 646, 716–723 (2025).

4. Ghareeb, A. E. et al. A multi-agent system for automating scientific discovery. Nature (2026). doi:10.1038/s41586-026-10652-y.

5. Majumder, B. P. et al. DiscoveryBench: Towards data-driven discovery with large language models. arXiv [cs.CL] (2024). doi:10.48550/arXiv.2407.01725.

6. Laurent, J. M. et al. LAB-bench: Measuring capabilities of language models for biology research. arXiv [cs.AI] (2024). doi:10.48550/arXiv.2407.10362.

7. Starace, G. et al. PaperBench: Evaluating AI’s ability to replicate AI research. arXiv [cs.AI] (2025). doi:10.48550/arXiv.2504.01848.

8. Mitchener, L. et al. BixBench: A comprehensive benchmark for LLM-based agents in computational biology. arXiv [q-bio.QM] (2025). doi:10.48550/arXiv.2503.00096.

9. Nirenberg, M. W. & Matthaei, J. H. The dependence of cell-free protein synthesis in E. coli upon naturally occurring or synthetic polyribonucleotides. Proc. Natl. Acad. Sci. U. S. A. 47, 1588–1602 (1961).

10. Crick, F. H., Barnett, L., Brenner, S. & Watts-Tobin, R. J. General nature of the genetic code for proteins. Nature 192, 1227–1232 (1961).

11. Tewhey, R. et al. Direct identification of hundreds of expression-modulating variants using a multiplexed reporter assay. Cell 165, 1519–1529 (2016).

12. Kircher, M. et al. Saturation mutagenesis of twenty disease-associated regulatory elements at single base-pair resolution. Nat Commun 10, 3583 (2019).

13. Siraj, L. et al. Functional dissection of complex trait variants at single-nucleotide resolution. Nature (2026). doi:10.1038/s41586-026-10121-6.

14. Gosai, S. J. et al. Machine-guided design of cell-type-targeting cis-regulatory elements. Nature 634, 1211–1220 (2024).

15. Exonic – AI for precision gene therapy. https://www.exonic.ai/. Accessed: 2026-8-27.

16. 10,000 AI-designed regulatory DNA sequences, open for research. https://origin.bio/blogs/switch/. Accessed: 2026-6-8.

17. Butts, J. C. et al. Identifying non-coding variant efects at scale via machine learning models of cis-regulatory reporter assays. bioRxiv 2025.04.16.648420 (2025). doi:10.1101/2025.04.16.648420.

18. Ghari, P. M., Sciabola, S. & Wang, Y. Iterative foundation model fine-tuning on multiple rewards. arXiv [cs.LG] (2025). doi:10.48550/arXiv.2511.00220.

19. Ma, M. et al. Reconstructing sequence-grammar trajectories enables interpretable and tunable cisregulatory element design (2026). doi:10.2139/ssrn.7096326.

20. Weykopf, G., Bickmore, W. A., Biddie, S. C. & Friman, E. T. Identifying severe COVID-19 risk variants modulating enhancer reporter activity in lung cells. PLoS Genet. 22, e1012222 (2026).

21. Uehara, M. et al. Reward-guided iterative refinement in difusion models at test-time with applications to protein and DNA design. arXiv [cs.LG] (2025). doi:10.48550/arXiv.2502.14944.

22. Introducing claude opus 4.7. https://www.anthropic.com/news/claude-opus-4-7. Accessed: 2026-5-31.

23. Website. https://openai.com/index/introducing-gpt-5-5/.

24. Kavukcuoglu, K. Gemini 3.5: frontier intelligence with action. https://blog.google/innovation-a nd-ai/models-and-research/gemini-models/gemini-3-5/ (2026). Accessed: 2026-5-31.

25. Meuleman, W. et al. Index and biological spectrum of human DNase I hypersensitive sites. Nature 584, 244–251 (2020).

26. de Boer, C. G. & Taipale, J. Hold out the genome: a roadmap to solving the cis-regulatory code. Nature 625, 41–50 (2024).

27. Moore, J. E. et al. An expanded registry of candidate cis-regulatory elements. Nature 1–10 (2026).

28. Dill, K. A. Theory for the folding and stability of globular proteins. Biochemistry 24, 1501–1509 (1985).

29. Nadgir, N. et al. Life after benchmark saturation: A case study of CORE-bench. arXiv [cs.AI] (2026). doi:10.48550/arXiv.2606.26158.

30. Chen, K. M., Wong, A. K., Troyanskaya, O. G. & Zhou, J. A sequence-based global map of regulatory activity for deciphering human genetics. Nat. Genet. 54, 940–949 (2022).

## Supplementary Figures and Tables

Library size vs. surrogate model performance across full evaluation suite

![](images/cb9c9c51319b7c920c6587895f03eb17cb850b8e977113d5901bb7f93f55faa8.jpg)  
Fig. S1 Performance versus training library size. Performance against library size $\left( \mathrm { I O } , \mathrm { O O O } - 3 0 0 , \mathrm { O O O } \right)$ sequences) for each of the 14 human-curated strategies (one panel each). Line, mean over five seeds; band, ± 1 SD.

![](images/c67f714d49e6773f5b7f94e0bbdb79fafbb1c3cba047597f38271992e376dd08.jpg)  
Fig. S2 Per-set performance of one-shot libraries by model across each held-out evaluation set, which includes empirical MPRA measurements and oracle-labeled regulatory, variant, genomic, and synthetic sequence sets.

![](images/c703d0e36648c6c68a2dbf4ee97af82bd527f8f4d2dbf944c81546ff931943be.jpg)  
Fig. S3 Agent activity versus one-shot performance. One-shot performance against eight activity metrics (points colored by model). Each panel shows the Pearson r and two-sided p across the 30 runs.

![](images/6794dfa9092b926d2c557b49638b85f093ecf3e1dd05ea7d95d570d661f53164.jpg)  
Fig. $\pmb { \mathscr { s } _ { 4 } }$ One-shot versus long-horizon performance. One-shot performance (points, by model) and the best long-horizon design (star) in the blind and informed conditions. Dashed line, human best; dotted line, random floor.

![](images/998690172310665c0d92121ad35dc8ddbc959fffcca5e3d5ac15214964e57152.jpg)  
Fig. S5 Performance versus token cost. Performance against total tokens used (input + output, log scale) for one-shot agents (points, by model) and long-horizon runs (stars, best of 30 libraries per agent).

# Supplementary Agent Instructions and Harness

# All agent-facing files in both sandboxes are provided directly below.

# MPRAbox one-shot instructions: without prior knowledge

mprabox/instructions/oneshot\_without\_prior\_knowledge.md

\# MPRA Library Design — One Shot

## ## Objective

You are an autonomous, independent researcher designing a 50,000-sequence library for a massively parallel reporter assay (MPRA), a high-throughput experiment that measures how DNA sequences drive gene expression. The purpose of this library is to train the best possible model of gene regulatory activity. You can measure activity in K562, HepG2, and SK-N-SH, but the goal is a model that captures regulatory grammar across ALL cell types — not just these three. Design for general regulatory grammar, not for these specific cell lines.

The library should be:

\- Not specific to a set of tissues

\- Not only functional elements

\- Diverse in sequence space

\- High training performance-to-size ratio

## ## How it works

You have \*\*one commit\*\*. Before you commit, you can do anything: read literature, download and analyze data, write exploratory code, reason about sequence properties, test hypotheses computationally. Take as long as you need.

When you are ready, commit your library. \`prepare.py\` will evaluate it immediately and return your score. The run ends there — no further iterations.

Because you cannot iterate, invest heavily in understanding the problem before designing.

## ## Lab Notebook

Keep a detailed lab notebook in \`notebook.md\`. Update it continuously throughout your work — every time you have a new idea, run an analysis, make a decision, or change direction, write it down immediately. Do not save it for the end. It is part of the scientific record. Include:

\- Your theory of what makes a good MPRA training library and why

\- What sources of data or sequence types you considered and why you included or excluded each

\- The specific design decisions you made and the reasoning behind them

Any analyses you ran and what they told you   
What you would try next if you had another shot   
Write this as if explaining your reasoning to a colleague who will read it   
after the run. It should be detailed enough that someone could reconstruct   
your thinking from scratch.   
## What to produce   
1. Write \`notebook.md\` documenting your reasoning and design decisions   
2. Create a \`library/\` directory   
3. Write \`generate.py\` that produces exactly 50,000 200bp sequences and saves   
them to \`library/sequences.txt\` (one per line, {A,C,G,T} only)   
4. Run \`python prepare.py library/sequences.txt   
5. \`prepare.py\` writes \`library/result.json\` with your scores   
6. \`git add -A && git commit -m "oneshot: mean\_r=X.XXX" && git push   
## Rules   
\`library/sequences.txt\` must contain \*\*exactly 50,000 lines\*\*, each exactly   
200 characters from {A, C, G, T}   
You \*\*cannot\*\* modify \`prepare.py\` or any other existing files   
You \*\*can\*\* search the web, download data, install packages, write any code   
Store downloaded data and reference files in \`data/\`   
\`prepare.py\` is a black box — treat it like a wet lab assay. You hand it   
sequences; it returns measurements. Do not read or inspect it.   
## Evaluation   
\`prepare.py\` evaluates your library against 14 anonymous test sets (eval\_01   
through eval\_14). You do not know what these sets contain. Each returns   
mean\_r, k562\_r, hepg2\_r, sknsh\_r. Optimize across all of them — there is no   
single primary metric.   
The result is written to \`library/result.json\`.

# MPRAbox one-shot instructions: with prior knowledge

mprabox/instructions/oneshot\_with\_prior\_knowledge.md

## # MPRA Library Design — One Shot (Informed)

## ## Objective

You are an autonomous, independent researcher designing a 50,000-sequence library for a massively parallel reporter assay (MPRA), a high-throughput experiment that measures how DNA sequences drive gene expression. The purpose of this library is to train the best possible model of gene regulatory activity. You can measure activity in K562, HepG2, and SK-N-SH, but the goal is a model that captures regulatory grammar across ALL cell types — not just these three. Design for general regulatory grammar, not for these specific cell lines.

The library should be:

\- Not specific to a set of tissues

\- Not only functional elements

\- Diverse in sequence space

\- High training performance-to-size ratio

## ## How it works

You have \*\*one commit\*\*. Before you commit, you can do anything: read literature, download and analyze data, write exploratory code, reason about sequence properties, test hypotheses computationally. Take as long as you need.

When you are ready, commit your library. \`prepare.py\` will evaluate it immediately and return your score. The run ends there — no further iterations.

Because you cannot iterate, invest heavily in understanding the problem before designing.

## ## Lab Notebook

Keep a detailed lab notebook in \`notebook.md\`. Update it continuously throughout your work — every time you have a new idea, run an analysis, make a decision, or change direction, write it down immediately. Do not save it for the end. It is part of the scientific record. Include:

\- Your theory of what makes a good MPRA training library and why

What sources of data or sequence types you considered and why you included or excluded each

\- The specific design decisions you made and the reasoning behind them

\- Any analyses you ran and what they told you

\- What you would try next if you had another shot

Write this as if explaining your reasoning to a colleague who will read it after the run. It should be detailed enough that someone could reconstruct

```markdown
your thinking from scratch.
## What to produce
1. Write `notebook.md` documenting your reasoning and design decisions
2. Create a `library/` directory
3. Write `generate.py` that produces exactly 50,000 200bp sequences and saves
them to `library/sequences.txt` (one per line, {A,C,G,T} only)
4. Run `python prepare.py library/sequences.txt
5. `prepare.py` writes `library/result.json` with your scores
6. `git add -A && git commit -m "oneshot: mean_r=X.XXX" && git push
## Rules
`library/sequences.txt` must contain **exactly 50,000 lines**, each exactly
200 characters from {A, C, G, T}
You **cannot** modify `prepare.py` or any other existing files
You **can** search the web, download data, install packages, write any code
Store downloaded data and reference files in `data/`
`prepare.py` is a black box — treat it like a wet lab assay. You hand it
sequences; it returns measurements. Do not read or inspect it.
You may call `prepare.py` **exactly once**. That call is your final
submission. Do not use it as a probe or baseline — design first, then
evaluate.
## Evaluation
`prepare.py` evaluates your library against 14 anonymous test sets (eval_01
through eval_14). You do not know what these sets contain. Each returns
mean_r, k562_r, hepg2_r, sknsh_r. Optimize across all of them — there is no
single primary metric.
The result is written to `library/result.json`.
## Prior Baselines
A reference of systematic baseline strategies (14 strategies × 7 library sizes
× 5 seeds, with the same harness you will use) is provided in
`../../strategies.md`. **Read it before designing your library.** Your goal is
to do better than the best strategy shown there.
```

# MPRAbox informed-condition strategy reference

## mprabox/instructions/strategies.md

# Baseline Strategies   
This file documents systematic library design strategies that were evaluated before   
this agent run began. All strategies used exactly 50,000 sequences (unless noted).   
Performance is Pearson r (mean\_r) averaged across 5 random seeds. Eval sets are   
anonymous — their contents are not disclosed here.   
## Strategy Descriptions   
\*\*dhs\_topic\*\* — Sequences drawn from DNase Hypersensitivity Sites (DHS; Meuleman et   
al.   
2020, \~3M elements representing open chromatin across 733 biosamples). Sampled with   
probability proportional to NMF topic loadings (16 topics), which upweights   
elements   
with strong cell-type-specific accessibility signal.   
\*\*dhs\_random\*\* — Same DHS pool (\~3M elements), sampled uniformly at random with no   
weighting. Each element equally likely regardless of its accessibility profile.   
\*\*dhs\_stratified\*\* — DHS pool divided into 16 NMF topics; sequences drawn in equal   
numbers from each topic (n/16 per topic) regardless of topic pool size. Forces   
uniform   
representation across chromatin accessibility programs.   
\*\*dhs\_sei\*\* — 50% DHS (topic-weighted) + 50% sequences from SEI chromatin state   
regions   
(Chen et al. 2022, \~3M regions covering 40 chromatin state classes), sampled   
proportional to class frequency.   
\*\*dhs\_synth\*\* — 50% DHS (topic-weighted) + 50% fully random sequences (i.i.d.   
uniform   
draw from {A, C, G, T}). Adds sequence diversity at the cost of biological   
relevance.   
\*\*dhs\_sei\_synth\*\* — 1/3 DHS (topic-weighted) + 1/3 SEI (class-proportional) + 1/3   
random synthetic. Three-way mixture of open chromatin, chromatin states, and noise.   
\*\*dhs\_stratified\_sei\*\* — 50% DHS (NMF-stratified, equal topics) + 50% SEI   
(class-balanced, equal classes). Both components are diversity-maximized within   
their   
respective annotation systems.   
\*\*dhs\_stratified\_sei\_synth\*\* — 1/3 DHS (NMF-stratified) + 1/3 SEI (class-balanced)   
+

1/3 random synthetic. Diversity-maximized genomic components plus random sequences.   
\*\*sei\_class\*\* — Sequences from SEI chromatin state regions only, sampled   
proportional to   
class size (biased toward common chromatin states).   
\*\*sei\_random\*\* — SEI regions sampled uniformly at random (no class weighting).   
\*\*sei\_synth\*\* — 50% SEI (class-balanced across 40 classes) + 50% random synthetic   
sequences.   
\*\*synth\_oracle\*\* — Fully random sequences (i.i.d. uniform {A, C, G, T}),   
oracle-labeled.   
No biological structure; serves as a coverage/diversity floor.   
\*\*mpra\_oracle\*\* — Sequences drawn from an existing published MPRA dataset (\~798K   
sequences), sampled randomly and oracle-labeled. Biologically curated but   
constrained   
to the distribution of a prior experiment.   
\*\*mpra\_real\*\* — Same sequences as mpra\_oracle but trained using the actual   
experimental   
MPRA measurements as labels rather than oracle predictions. Tests whether empirical   
labels (noisier but real) help or hurt surrogate training.   
## Table 1 — 50k Performance Across All Eval Sets   
Mean Pearson r across 5 seeds. Strategies ordered by eval\_01 (primary metric).   
| strategy | eval\_01 | eval\_02 | eval\_03 | eval\_04 | eval\_05 |   
eval\_06 | eval\_07 | eval\_08 | eval\_09 | eval\_10 | eval\_11 | eval\_12 | eval\_13 |   
eval\_14 |   
| dhs\_topic | 0.7232 | 0.8138 | 0.7933 | 0.7904 | 0.7230 |   
0.8136 | 0.7398 | 0.7011 | 0.8601 | 0.7904 | 0.7098 | 0.6822 | 0.7271   
0.8144 |   
| dhs\_sei | 0.7201 | 0.8121 | 0.7944 0.7754 | 0.7204   
0.8117 | 0.7640 | 0.6526 | 0.8413 | 0.7688 | 0.7072 0.6826 | 0.7578   
0.8121 |   
| dhs\_synth | 0.7174 | 0.8084 | 0.7869 | 0.7800 | 0.7169   
0.8082 | 0.7277 | 0.7523 | 0.8469 | 0.7829 | 0.7040 | 0.6767 | 0.7102   
0.8091 |   
| dhs\_random | 0.7089 | 0.8023 | 0.7902 | 0.7429 | 0.7088 |   
0.8027 | 0.7615 | 0.6673 | 0.8051 | 0.7742 | 0.6970 | 0.6783 | 0.7639   
0.8021 |

<table><tr><td></td><td>I dhs_stratified_sei_synth | 0.7094 0.8015 | 0.7553 | 0.6956</td><td></td><td>0.8012</td><td>0.8013 0.7873 0.7592 0.6975</td><td>0.7395 0.6778</td><td></td><td>0.7098 1 0.7570</td></tr><tr><td>0.8006 1</td><td></td><td></td><td></td><td></td><td></td><td></td><td>1</td></tr><tr><td></td><td>dhs_stratified 0.7509</td><td>0.6596</td><td>0.7055 0.8030</td><td>0.7978 I 0.7847</td><td>0.7424</td><td>0.7055</td><td>I</td></tr><tr><td>0.7983 0.7977</td><td></td><td></td><td></td><td>0.7708 0.6939</td><td>0.6740</td><td>0.7583</td><td></td></tr><tr><td></td><td>dhs_sei_synth</td><td></td><td>0.6975</td><td>0.7876 1 0.7685</td><td>1 0.7511</td><td>0.6978</td><td></td></tr><tr><td>0.7874</td><td></td><td>0.72550.6746</td><td>0.8131</td><td>0.7435 0.6853</td><td>0.6608</td><td>0.7145</td><td></td></tr><tr><td>0.7875</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>I synth_oracle</td><td></td><td>0.6840 1</td><td>0.7719 0.7459</td><td>0.7401</td><td>0.6836</td><td>1</td></tr><tr><td></td><td></td><td>0.7724|0.6483| 0.7696</td><td>0.8012</td><td>0.7433 0.6719</td><td>0.6419</td><td>0.6410</td><td></td></tr><tr><td>0.7723</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>I dhs_stratified_sei</td><td></td><td>0.6818</td><td>0.7705 0.7595</td><td>0.7156</td><td>0.6823</td><td>I</td></tr><tr><td></td><td>0.7709 | 0.7394 | 0.5997</td><td></td><td>0.7731</td><td>0.7295 0.6708</td><td>0.6523</td><td>0.7445</td><td></td></tr><tr><td>0.7700</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>I sei_synth</td><td></td><td>0.6682</td><td>0.7558 0.7418</td><td>1 0.7019</td><td>0.6684</td><td>I</td></tr><tr><td>0.7560</td><td></td><td>0.7107|0.6580</td><td>0.7593</td><td>0.7076 0.6569</td><td>0.6386</td><td>0.7146</td><td></td></tr><tr><td>0.7556</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>I mpra_oracle</td><td></td><td>0.6643</td><td>0.7505 」 0.7361</td><td>1 0.7107</td><td>0.6651</td><td>I</td></tr><tr><td></td><td></td><td>0.7509| 0.6879 | 0.5407</td><td>0.7665</td><td>0.6805 0.6534</td><td>0.6322</td><td>0.7050</td><td></td></tr><tr><td>0.7497</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>| sei_class</td><td></td><td></td><td>0.6593</td><td>0.7445 0.7362</td><td>0.6961</td><td>0.6596</td><td>I</td></tr><tr><td>0.7452</td><td></td><td>0.7303|0.5510</td><td>0.7504</td><td>0.6963 0.6490</td><td>0.6333</td><td>0.7354</td><td></td></tr><tr><td>0.7439</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>I sei_random</td><td></td><td>0.6454</td><td>0.7286 1 0.7211</td><td>1 0.6762</td><td>0.6459</td><td></td></tr><tr><td>0.7292</td><td></td><td>0.7146|0.5322</td><td>0.7299</td><td>0.6758 0.6353</td><td>0.6208</td><td>0.7268</td><td></td></tr><tr><td>0.7281</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td> mpra_real</td><td></td><td></td><td>0.6026 1</td><td>0.6781 0.6595</td><td>0.6560</td><td>0.6034</td><td>I</td></tr><tr><td>0.6791</td><td></td><td>|0.5952| 0.4387| 0.7020</td><td></td><td>0.5890 0.5927</td><td>0.5683</td><td>0.6106</td><td>1</td></tr><tr><td>0.6774</td><td>1</td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Mean eval\_01 Pearson r across 5 seeds, at each library size.

<table><tr><td>strategy 300k 1</td><td> 10k | 25k | 50k | 100k | 150k | 200k |</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>---------|-----------------------------------------</td><td></td><td></td><td></td><td></td></tr><tr><td>I dhs_topic 0.8448|</td><td></td><td> 0.4462 | 0.5318 | 0.7232 | 0.7688 | 0.8157 | 0.8356 |</td><td></td><td></td><td></td></tr><tr><td>| dhs_sei 0.8528</td><td></td><td> 0.4621 | 0.5490 | 0.7201 | 0.7809 | 0.8198 | 0.8446 |</td><td></td><td></td><td></td></tr><tr><td>I dhs_synth</td><td></td><td></td><td> 0.4069 | 0.5059 | 0.7174 | 0.7576 | 0.8106 | 0.8255 |</td><td></td><td></td></tr><tr><td>0.8376 I dhs_random 0.8462 </td><td></td><td></td><td> 0.3928 | 0.5342 | 0.7089 | 0.7584 | 0.8075 | 0.8328 |</td><td></td><td></td></tr></table>

<table><tr><td>dhs_stratified_sei_synth | 0.4293 | 0.5168 | 0.7094 | 0.7585 | 0.8081 | 0.8316 | 0.8430</td></tr><tr><td>I dhs_stratified 0.3929 | 0.5577 | 0.7055 | 0.7593 | 0.8105 | 0.8332 |</td></tr><tr><td>0.8477</td></tr><tr><td>I dhs_sei_synth 0.4538 | 0.5210 | 0.6975 | 0.7721 | 0.8100 | 0.8416 | 0.8530</td></tr><tr><td>I synth_oracle 0.3817 | 0.4673 | 0.6840 | 0.7246 | 0.7471 | 0.7725 | 0.7841|</td></tr><tr><td>I dhs_stratified_sei 0.4231 | 0.5401 | 0.6818 | 0.7490 | 0.7956 | 0.8265 |</td></tr><tr><td>0.8467|</td></tr><tr><td>| sei_synth 0.4033 | 0.5028 | 0.6682 | 0.7613 | 0.7939 | 0.8210 | 0.8404</td></tr><tr><td>I mpra_oracle 0.4683 | 0.5416 | 0.6643 | 0.7376 | 0.7609 | 0.8152 |</td></tr><tr><td>0.8372 </td></tr><tr><td>sei_class 0.4438 | 0.5162 | 0.6593 | 0.7361 | 0.7728 | 0.8087 | 0.8371|</td></tr><tr><td>I sei_random 0.4155 | 0.5385 | 0.6454 | 0.7254 | 0.7741 | 0.8075 |</td></tr><tr><td>0.8317</td></tr><tr><td>I mpra_real 0.4259 | 0.5178 | 0.6026 | 0.6899 | 0.7318 | 0.7893 |</td></tr><tr><td>0.8200|</td></tr></table>

\## Table 3 — Mean Performance Across All Strategies, by Eval Set and Size

Mean Pearson r averaged across all 14 strategies. Shows how hard each eval set is and how performance scales with library size.

<table><tr><td rowspan=1 colspan=1>eval</td><td rowspan=1 colspan=1>10k</td><td rowspan=1 colspan=1>25k</td><td rowspan=1 colspan=1>50k</td><td rowspan=1 colspan=1>100k</td><td rowspan=1 colspan=1>150k</td><td rowspan=1 colspan=1>200k</td><td rowspan=1 colspan=1>300k</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>1 一-   一 1</td><td rowspan=1 colspan=1>一-       1</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>1- -1</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>1–</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>eval_01</td><td rowspan=1 colspan=1>0.4247</td><td rowspan=1 colspan=1>0.5243</td><td rowspan=1 colspan=1>0.6849</td><td rowspan=1 colspan=1>0.7485</td><td rowspan=1 colspan=1>0.7899</td><td rowspan=1 colspan=1>0.8204</td><td rowspan=1 colspan=1>0.83731</td></tr><tr><td rowspan=1 colspan=1>eval_02</td><td rowspan=1 colspan=1>0.4791</td><td rowspan=1 colspan=1>0.5904</td><td rowspan=1 colspan=1>0.7731</td><td rowspan=1 colspan=1>0.8433</td><td rowspan=1 colspan=1>0.8871</td><td rowspan=1 colspan=1>0.9184</td><td rowspan=1 colspan=1>0.9358 I</td></tr><tr><td rowspan=1 colspan=1>eval_03</td><td rowspan=1 colspan=1>0.4501</td><td rowspan=1 colspan=1>0.5647</td><td rowspan=1 colspan=1>0.7575</td><td rowspan=1 colspan=1>0.8324</td><td rowspan=1 colspan=1>0.8790</td><td rowspan=1 colspan=1>0.9120</td><td rowspan=1 colspan=1>0.9302 1</td></tr><tr><td rowspan=1 colspan=1>eval_04</td><td rowspan=1 colspan=1>0.4813</td><td rowspan=1 colspan=1>0.5774</td><td rowspan=1 colspan=1>0.7299</td><td rowspan=1 colspan=1>0.7841</td><td rowspan=1 colspan=1>0.8193</td><td rowspan=1 colspan=1>0.8498</td><td rowspan=1 colspan=1>0.8659 1</td></tr><tr><td rowspan=1 colspan=1>eval_05</td><td rowspan=1 colspan=1>0.4251</td><td rowspan=1 colspan=1>0.5247</td><td rowspan=1 colspan=1>0.6850</td><td rowspan=1 colspan=1>0.7486</td><td rowspan=1 colspan=1>0.7900</td><td rowspan=1 colspan=1>0.8205</td><td rowspan=1 colspan=1>0.8374 </td></tr><tr><td rowspan=1 colspan=1>eval_06</td><td rowspan=1 colspan=1>0.4802</td><td rowspan=1 colspan=1>0.5909</td><td rowspan=1 colspan=1>0.7733</td><td rowspan=1 colspan=1>0.8436</td><td rowspan=1 colspan=1>0.8874</td><td rowspan=1 colspan=1>0.9186</td><td rowspan=1 colspan=1>0.93601</td></tr><tr><td rowspan=1 colspan=1>eval_07</td><td rowspan=1 colspan=1>0.3631</td><td rowspan=1 colspan=1>0.5002</td><td rowspan=1 colspan=1>0.7179</td><td rowspan=1 colspan=1>0.7998</td><td rowspan=1 colspan=1>0.8545</td><td rowspan=1 colspan=1>0.8940</td><td rowspan=1 colspan=1>0.9144</td></tr><tr><td rowspan=1 colspan=1>eval_08</td><td rowspan=1 colspan=1>0.2744</td><td rowspan=1 colspan=1>0.3914</td><td rowspan=1 colspan=1>0.6352</td><td rowspan=1 colspan=1>0.7468</td><td rowspan=1 colspan=1>0.8188</td><td rowspan=1 colspan=1>0.8701</td><td rowspan=1 colspan=1>0.8985 1</td></tr><tr><td rowspan=1 colspan=1>eval_09</td><td rowspan=1 colspan=1>0.51061</td><td rowspan=1 colspan=1>0.6174</td><td rowspan=1 colspan=1>0.7895</td><td rowspan=1 colspan=1>0.8500</td><td rowspan=1 colspan=1>0.8891</td><td rowspan=1 colspan=1>0.9244</td><td rowspan=1 colspan=1>0.9424</td></tr><tr><td rowspan=1 colspan=1>eval_10</td><td rowspan=1 colspan=1>0.4088</td><td rowspan=1 colspan=1>0.5213</td><td rowspan=1 colspan=1>0.7294</td><td rowspan=1 colspan=1>0.8102</td><td rowspan=1 colspan=1>0.8616</td><td rowspan=1 colspan=1>0.9004</td><td rowspan=1 colspan=1>0.9224</td></tr><tr><td rowspan=1 colspan=1>eval_11</td><td rowspan=1 colspan=1>0.4176</td><td rowspan=1 colspan=1>0.5152</td><td rowspan=1 colspan=1>0.6732</td><td rowspan=1 colspan=1>0.7360</td><td rowspan=1 colspan=1>0.7766</td><td rowspan=1 colspan=1>0.8063</td><td rowspan=1 colspan=1>0.82281</td></tr><tr><td rowspan=1 colspan=1>eval_12</td><td rowspan=1 colspan=1>0.3871</td><td rowspan=1 colspan=1>0.4863</td><td rowspan=1 colspan=1>0.6514</td><td rowspan=1 colspan=1>0.7172</td><td rowspan=1 colspan=1>0.7594</td><td rowspan=1 colspan=1>0.7901</td><td rowspan=1 colspan=1>0.80691</td></tr><tr><td rowspan=1 colspan=1>eval_13</td><td rowspan=1 colspan=1>0.3624</td><td rowspan=1 colspan=1>0.4970</td><td rowspan=1 colspan=1>0.7190</td><td rowspan=1 colspan=1>0.8026</td><td rowspan=1 colspan=1>0.8567</td><td rowspan=1 colspan=1>0.8934</td><td rowspan=1 colspan=1>0.91321</td></tr><tr><td rowspan=1 colspan=1>eval_14</td><td rowspan=1 colspan=1>0.4791</td><td rowspan=1 colspan=1>0.5904</td><td rowspan=1 colspan=1>0.7729</td><td rowspan=1 colspan=1>0.8431</td><td rowspan=1 colspan=1>0.8870</td><td rowspan=1 colspan=1>0.91851</td><td rowspan=1 colspan=1>0.9359</td></tr></table>

# MPRAbox long-horizon instructions: without prior knowledge

# mprabox/instructions/long\_horizon\_without\_prior\_knowledge.md

# MPRA Library Design   
## Objective   
You are an autonomous, independent researcher designing a 50,000-sequence library   
for a massively parallel reporter assay (MPRA), a high-throughput experiment that   
measures how DNA sequences drive gene expression.   
The purpose of this library is to serve as training data for a sequence-to-activity   
model. You can measure activity in K562, HepG2, and SK-N-SH — but these three cell   
types are a measurement constraint, not the goal. The goal is a library that would   
be equally informative for training a model on a completely different set of cell   
types that you have never measured. Before proposing each experiment, ask yourself:   
if someone trained a model on this library but evaluated it in cell types we have   
no data on, would the library still have been worth designing this way? Justify   
your answer explicitly.   
The goal is not just to find a good library but to understand what makes a library   
good. What properties of a sequence make it informationally valuable for training a   
model that will be evaluated in conditions beyond its labeling context? Build and   
refine a theory through experimentation.   
## How it works   
1. Create a new directory under \`libraries/\` for each experiment   
2. Write a \`generate.py\` that produces exactly 50,000 200bp sequences   
\*\*three times\*\* with different random seeds, saving them to   
\`sequences\_0.txt\`, \`sequences\_1.txt\`, and \`sequences\_2.txt   
3. Run \`python prepare.py libraries/NNN\_name/\`   
4. prepare.py does the following:   
- Runs each of your three 50,000-sequence files through an MPRA   
in K562, HepG2, and SK-N-SH, producing activity measurements   
- Trains a sequence-to-activity model from scratch on each library   
and its measurements (three independent training runs in parallel)   
- Evaluates each model on held-out sequences with real MPRA   
measurements (sequences your model has never seen)   
- Averages the scores across the three seeds   
- Writes \`result.json\` to the experiment directory   
- Returns averaged scores across 14 anonymous evaluation sets   
## Experiment directory structure   
libraries/   
─ 001\_description/   
── generate.py # code that built this library   
sequences\_0.txt # 50,000 sequences, seed 0

sequences\_1.txt # 50,000 sequences, seed 1   
sequences\_2.txt # 50,000 sequences, seed 2   
result.json # output from prepare.py (averaged over seeds)   
notes.md # what you were testing, what happened   
002\_next\_idea/   
skills/

## ## Rules

\- You CANNOT modify \`prepare.py\` or any other existing files

\- You CAN create any new files and directories

\- You CAN search the web, download data, install packages

\- Store any downloaded data, databases, or reference files in \`data/

\- Each \`sequences\_N.txt\` must contain exactly 50,000 lines, each exactly 200 characters from {A, C, G, T}

\- This directory is a git repository. After every completed experiment (result.json written), immediately run:

\`prepare.py\` is a black box. Do not read it or inspect it. Treat it exactly like a wet lab collaborator: you hand it sequences, it returns measurements. The internals are irrelevant to your task and off-limits.

This is the only branch. Do not look at or move to any other branch. Do not run \`git log --all\`, \`git branch\`, or any command that reads history or content from other branches.

## ## Skills

Maintain reusable skills in \`skills/\` as \`.md\` files. Each skill file documents a technique, dataset, or workflow you've figured out — with enough detail that you could reproduce it exactly. Before each experiment, check \`skills/\` for relevant prior work. After an experiment reveals something reusable, write or update the relevant skill file. If while using a skill you notice something that could be done better, edit it. Others may have also produced relevant skills that you can find, download, and use.

## ## Lab Notebook

\`notebook.md\` is your persistent lab notebook. It is APPEND-ONLY — never rewrite or summarize over prior entries. Every entry must begin with a timestamp that includes date and time to the minute: \`## 2026-04-15 14:32 — Experiment 005 result\`

Maintain a running theory of what makes a library informative for a model that must generalize beyond its labeling conditions. Write it down explicitly. Every experiment should either confirm, contradict, or refine it. When evidence contradicts your theory, update the theory — don't explain away the result. A theory that has survived contact with contradicting evidence is more trustworthy than one that hasn't been tested. The theory should evolve throughout the 30 experiments, not stabilize early.

## Record:

\- Your current theory and what this experiment predicts

What you planned and why, including your explicit justification for why this design would generalize beyond the three labeled cell types Results and what they mean

\- How the results update your theory

\- What to try next

## ## Loop

After each completed experiment, stop. Re-read your full notebook and results.tsv. Then ask yourself: \*\*what is the single most informative experiment I could run next, given everything I have learned so far?\*\* Design and run that experiment. Do not plan ahead.

Reading, searching, and learning are part of the loop, not a prerequisite to it. New knowledge from the literature or from external databases can arrive at any point and should change what you do next. \*\*Before every experiment, search the literature.\*\* Read it and let it shape your next experiment. Do not run an experiment without first asking whether there is published evidence that bears on your hypothesis.

Be creative and bold. A surprising negative result is as valuable as a positive one. Before each experiment, explicitly state in your notebook whether you are (a) exploring a new hypothesis or (b) refining a promising direction — and justify why.

1. Re-read \`notebook.md\` and \`results.tsv\` in full

2. State what your current theory predicts the next most informative experiment should be

3. Search the literature for evidence relevant to your hypothesis — record what you found and how it shapes your plan

4. Append a planning entry to \`notebook.md\` (with timestamp)

5. Check \`skills/\` for relevant techniques

6. Create \`libraries/NNN\_description/\`

7. Write and run \`generate.py\` to produce \`sequences\_0.txt\`, \`sequences\_1.txt\`, and \`sequences\_2.txt\` (same strategy, different random seeds)

8. Run \`python prepare.py libraries/NNN\_description/

9. Write \`notes.md\` in the experiment directory

10. Append result entry to \`notebook.md\` (with timestamp), including how the result updates your theory

11. Update \`results.tsv

12. Update any relevant skill files in \`skills/\`

13. \`git add -A && git commit -m "NNN\_description: mean\_r=X.XXX" && git push\` If push fails due to SSH/auth, commit locally and continue — do not retry push repeatedly.

14. Stop after 30 experiments total. Write a final summary entry in   
\`notebook.md\` covering: your final theory, what worked, what didn't,   
your best library, and recommendations for the next round.   
## Evaluation   
prepare.py evaluates your library against 14 anonymous evaluation sets   
(eval\_01 through eval\_14). You do not know what these sets contain.   
Each returns mean\_r, k562\_r, hepg2\_r, sknsh\_r averaged across your   
three sequence files. \*\*eval\_01 is the primary metric.\*\* Aim for high   
performance across all of them.   
## results.tsv format   
Tab-separated, one row per experiment:   
\`experiment eval\_01 eval\_02 eval\_14 time\_s description\`   
Record the mean\_r for each eval set.

# MPRAbox long-horizon instructions: with prior knowledge

# mprabox/instructions/long\_horizon\_with\_prior\_knowledge.md

# MPRA Library Design   
## Objective   
You are an autonomous, independent researcher designing a 50,000-sequence library   
for a massively parallel reporter assay (MPRA), a high-throughput experiment that   
measures how DNA sequences drive gene expression.   
The purpose of this library is to serve as training data for a sequence-to-activity   
model. You can measure activity in K562, HepG2, and SK-N-SH — but these three cell   
types are a measurement constraint, not the goal. The goal is a library that would   
be equally informative for training a model on a completely different set of cell   
types that you have never measured. Before proposing each experiment, ask yourself:   
if someone trained a model on this library but evaluated it in cell types we have   
no data on, would the library still have been worth designing this way? Justify   
your answer explicitly.   
The goal is not just to find a good library but to understand what makes a library   
good. What properties of a sequence make it informationally valuable for training a   
model that will be evaluated in conditions beyond its labeling context? Build and   
refine a theory through experimentation.   
## How it works   
1. Create a new directory under \`libraries/\` for each experiment   
2. Write a \`generate.py\` that produces exactly 50,000 200bp sequences   
\*\*three times\*\* with different random seeds, saving them to   
\`sequences\_0.txt\`, \`sequences\_1.txt\`, and \`sequences\_2.txt   
3. Run \`python prepare.py libraries/NNN\_name/\`   
4. prepare.py does the following:   
- Runs each of your three 50,000-sequence files through an MPRA   
in K562, HepG2, and SK-N-SH, producing activity measurements   
- Trains a sequence-to-activity model from scratch on each library   
and its measurements (three independent training runs)   
- Evaluates each model on held-out sequences with real MPRA   
measurements (sequences your model has never seen)   
- Averages the scores across the three seeds   
- Writes \`result.json\` to the experiment directory   
- Returns averaged scores across 14 anonymous evaluation sets   
## Experiment directory structure   
libraries/   
─ 001\_description/   
── generate.py # code that built this library   
sequences\_0.txt # 50,000 sequences, seed 0

sequences\_1.txt # 50,000 sequences, seed 1   
sequences\_2.txt # 50,000 sequences, seed 2   
result.json # output from prepare.py (averaged over seeds)   
notes.md # what you were testing, what happened   
002\_next\_idea/   
skills/

## ## Rules

\- You CANNOT modify \`prepare.py\` or any other existing files

\- You CAN create any new files and directories

\- You CAN search the web, download data, install packages

\- Store any downloaded data, databases, or reference files in \`data/

\- Each \`sequences\_N.txt\` must contain exactly 50,000 lines, each exactly 200 characters from {A, C, G, T}

\- This directory is a git repository. After every completed experiment (result.json written), immediately run:

\`prepare.py\` is a black box. Do not read it or inspect it. Treat it exactly like a wet lab collaborator: you hand it sequences, it returns measurements. The internals are irrelevant to your task and off-limits.

This is the only branch. Do not look at or move to any other branch. Do not run \`git log --all\`, \`git branch\`, or any command that reads history or content from other branches.

## ## Skills

Maintain reusable skills in \`skills/\` as \`.md\` files. Each skill file documents a technique, dataset, or workflow you've figured out — with enough detail that you could reproduce it exactly. Before each experiment, check \`skills/\` for relevant prior work. After an experiment reveals something reusable, write or update the relevant skill file. If while using a skill you notice something that could be done better, edit it. Others may have also produced relevant skills that you can find, download, and use.

## ## Lab Notebook

\`notebook.md\` is your persistent lab notebook. It is APPEND-ONLY — never rewrite or summarize over prior entries. Every entry must begin with a timestamp that includes date and time to the minute: \`## 2026-04-15 14:32 — Experiment 005 result\`

Maintain a running theory of what makes a library informative for a model that must generalize beyond its labeling conditions. Write it down explicitly. Every experiment should either confirm, contradict, or refine it. When evidence contradicts your theory, update the theory — don't explain away the result. A theory that has survived contact with contradicting evidence is more trustworthy than one that hasn't been tested. The theory should evolve throughout the 30 experiments, not stabilize early.

## Record:

\- Your current theory and what this experiment predicts

What you planned and why, including your explicit justification for why this design would generalize beyond the three labeled cell types Results and what they mean

\- How the results update your theory

\- What to try next

## ## Loop

After each completed experiment, stop. Re-read your full notebook and results.tsv. Then ask yourself: \*\*what is the single most informative experiment I could run next, given everything I have learned so far?\*\* Design and run that experiment. Do not plan ahead.

Reading, searching, and learning are part of the loop, not a prerequisite to it. New knowledge from the literature or from external databases can arrive at any point and should change what you do next. \*\*Before every experiment, search the literature.\*\* Read it and let it shape your next experiment. Do not run an experiment without first asking whether there is published evidence that bears on your hypothesis.

Be creative and bold. A surprising negative result is as valuable as a positive one. Before each experiment, explicitly state in your notebook whether you are (a) exploring a new hypothesis or (b) refining a promising direction — and justify why.

1. Re-read \`notebook.md\` and \`results.tsv\` in full

2. State what your current theory predicts the next most informative experiment should be

3. Search the literature for evidence relevant to your hypothesis — record what you found and how it shapes your plan

4. Append a planning entry to \`notebook.md\` (with timestamp)

5. Check \`skills/\` for relevant techniques

6. Create \`libraries/NNN\_description/\`

7. Write and run \`generate.py\` to produce \`sequences\_0.txt\`, \`sequences\_1.txt\`, and \`sequences\_2.txt\` (same strategy, different random seeds)

8. Run \`python prepare.py libraries/NNN\_description/

9. Write \`notes.md\` in the experiment directory

10. Append result entry to \`notebook.md\` (with timestamp), including how the result updates your theory

11. Update \`results.tsv

12. Update any relevant skill files in \`skills/\`

13. \`git add -A && git commit -m "NNN\_description: mean\_r=X.XXX" && git push\` If push fails due to SSH/auth, commit locally and continue — do not retry push repeatedly.

14. Stop after 30 experiments total. Write a final summary entry in   
\`notebook.md\` covering: your final theory, what worked, what didn't,   
your best library, and recommendations for the next round.   
## Evaluation   
prepare.py evaluates your library against 14 anonymous evaluation sets   
(eval\_01 through eval\_14). You do not know what these sets contain.   
Each returns mean\_r, k562\_r, hepg2\_r, sknsh\_r averaged across your   
three sequence files. \*\*eval\_01 is the primary metric.\*\* Aim for high   
performance across all of them.   
## results.tsv format   
Tab-separated, one row per experiment:   
\`experiment eval\_01 eval\_02 eval\_14 time\_s description\`   
Record the mean\_r for each eval set.

# MPRAbox dry-oracle instructions: MPRA-framed

mprabox/dry\_oracles/instructions/mpra\_framed.md

```markdown
# MPRA Library Design
## Objective
You are an autonomous, independent researcher designing a 50,000-sequence library
for a massively parallel reporter assay (MPRA), a high-throughput experiment that
measures how DNA sequences drive gene expression.
The purpose of this library is to serve as training data for a sequence-to-activity
model. You can measure activity in K562, HepG2, and SK-N-SH — but these three cell
types are a measurement constraint, not the goal. The goal is a library that would
be equally informative for training a model on a completely different set of cell
types that you have never measured. Before proposing each experiment, ask yourself:
if someone trained a model on this library but evaluated it in cell types we have
no data on, would the library still have been worth designing this way? Justify
your answer explicitly.
The goal is not just to find a good library but to understand what makes a library
good. What properties of a sequence make it informationally valuable for training a
model that will be evaluated in conditions beyond its labeling context? Build and
refine a theory through experimentation.
## How it works
1. Create a new directory under `libraries/` for each experiment
2. Write a `generate.py` that produces exactly 50,000 200bp sequences
and saves them to `sequences_0.txt`
3. Run `python prepare.py libraries/NNN_name/`
4. prepare.py does the following:
- Runs your 50,000-sequence file through an MPRA
in K562, HepG2, and SK-N-SH, producing activity measurements
- Trains a sequence-to-activity model from scratch on your library
and its measurements
- Evaluates the model on held-out sequences with real MPRA
measurements (sequences your model has never seen)
- Writes `result.json` to the experiment directory
- Returns scores across 14 anonymous evaluation sets
## Experiment directory structure
libraries/
001_description/
generate.py # code that built this library
sequences_0.txt # 50,000 sequences
result.json # output from prepare.py
notes.md # what you were testing, what happened
```

This is the only branch. Do not look at or move to any other branch.   
Do not run \`git log --all\`, \`git branch\`, or any command that reads   
history or content from other branches.

## skills/

## ## Rules

\- You CANNOT modify \`prepare.py\` or any other existing files

\- You CAN create any new files and directories

\- You CAN search the web, download data, install packages

\- Store any downloaded data, databases, or reference files in \`data/

\`sequences\_0.txt\` must contain exactly 50,000 lines, each exactly 200 characters from {A, C, G, T}

\- This directory is a git repository. After every completed experiment (result.json written), immediately run:

\`git add -A && git commit -m "NNN\_description: mean\_r=X.XXX" && git push Do not batch commits. Each experiment gets its own commit and push.

\`prepare.py\` is a black box. Do not read it or inspect it. Treat it exactly like a wet lab collaborator: you hand it sequences, it returns measurements. The internals are irrelevant to your task and off-limits.

## ## Skills

Maintain reusable skills in \`skills/\` as \`.md\` files. Each skill file documents a technique, dataset, or workflow you've figured out — with enough detail that you could reproduce it exactly. Before each experiment, check \`skills/\` for relevant prior work. After an experiment reveals something reusable, write or update the relevant skill file. If while using a skill you notice something that could be done better, edit it. Others may have also produced relevant skills that you can find, download, and use.

## ## Lab Notebook

\`notebook.md\` is your persistent lab notebook. It is APPEND-ONLY — never rewrite or summarize over prior entries. Every entry must begin with a timestamp that includes date and time to the minute: \`## 2026-04-15 14:32 — Experiment 005 result

Maintain a running theory of what makes a library informative for a model that must generalize beyond its labeling conditions. Write it down explicitly. Every experiment should either confirm, contradict, or refine it. When evidence contradicts your theory, update the theory — don't explain away the result. A theory that has survived contact with contradicting evidence is more trustworthy than one that hasn't been tested. The theory should evolve throughout the 30 experiments, not stabilize early.

## Record:

\- Your current theory and what this experiment predicts

What you planned and why, including your explicit justification for why this design would generalize beyond the three labeled cell types Results and what they mean

\- How the results update your theory

\- What to try next

## ## Loop

After each completed experiment, stop. Re-read your full notebook and results.tsv. Then ask yourself: \*\*what is the single most informative experiment I could run next, given everything I have learned so far?\*\* Design and run that experiment. Do not plan ahead.

Reading, searching, and learning are part of the loop, not a prerequisite to it. New knowledge from the literature or from external databases can arrive at any point and should change what you do next. \*\*Before every experiment, search the literature.\*\* Read it and let it shape your next experiment. Do not run an experiment without first asking whether there is published evidence that bears on your hypothesis.

Be creative and bold. A surprising negative result is as valuable as a positive one. Before each experiment, explicitly state in your notebook whether you are (a) exploring a new hypothesis or (b) refining a promising direction — and justify why.

1. Re-read \`notebook.md\` and \`results.tsv\` in full

2. State what your current theory predicts the next most informative experiment should be

3. Search the literature for evidence relevant to your hypothesis — record what you found and how it shapes your plan

4. Append a planning entry to \`notebook.md\` (with timestamp)

5. Check \`skills/\` for relevant techniques

6. Create \`libraries/NNN\_description/

7. Write and run \`generate.py\` to produce \`sequences\_0.txt\` (same strategy, one random seed)

8. Run \`python prepare.py libraries/NNN\_description/

9. Write \`notes.md\` in the experiment directory

10. Append result entry to \`notebook.md\` (with timestamp), including how the result updates your theory

11. Update \`results.tsv\`

12. Update any relevant skill files in \`skills/

13. \`git add -A && git commit -m "NNN\_description: mean\_r=X.XXX" && git push\` If push fails due to SSH/auth, commit locally and continue — do not retry push repeatedly.

14. Stop after 30 experiments total. Write a final summary entry in \`notebook.md\` covering: your final theory, what worked, what didn't, your best library, and recommendations for the next round.

## ## Evaluation

prepare.py evaluates your library against 14 anonymous evaluation sets   
(eval\_01 through eval\_14). You do not know what these sets contain.   
Each returns mean\_r, k562\_r, hepg2\_r, sknsh\_r. \*\*eval\_01 is the   
primary metric.\*\* Aim for high performance across all of them.   
## results.tsv format   
Tab-separated, one row per experiment:   
\`experiment eval\_01 eval\_02 eval\_14 time\_s description\`   
Record the mean\_r for each eval set.

# MPRAbox dry-oracle instructions: unframed

# mprabox/dry\_oracles/instructions/unframed.md

```markdown
# Sequence Optimization
## Objective
You are an autonomous researcher. You have access to a black-box
scoring function that takes 50,000 DNA sequences (200bp each) and
returns scores. Your goal is to maximize those scores.
You have 30 experiments. Use them wisely.
## How it works
1. Create a new directory under `libraries/` for each experiment
2. Write a `generate.py` that produces exactly 50,000 200bp sequences
and saves them to `sequences_0.txt
3. Run `python prepare.py libraries/NNN_name/`
4. prepare.py scores your library and writes `result.json
## Experiment directory structure
libraries/
001_description/
generate.py # code that built this library
sequences_0.txt # 50,000 sequences
result.json # scores from prepare.py
notes.md # what you were testing, what happened
002_next_idea/
skills/
## Rules
- You CANNOT modify `prepare.py` or any other existing files
You CAN create any new files and directories
You CAN search the web, install packages, write any code
Store any downloaded data or reference files in `data/`
`sequences_0.txt` must contain exactly 50,000 lines, each exactly
200 characters from {A, C, G, T}
This directory is a git repository. After every completed experiment
(result.json written), immediately run:
`git add -A && git commit -m "NNN_description: mean_r=X.XXX" && git push
Do not batch commits. Each experiment gets its own commit and push.
`prepare.py` is a black box. Do not read it or inspect it. Treat it
like an API endpoint you cannot see inside.
This is the only branch. Do not look at or move to any other branch.
```

\## Skills Maintain reusable skills in \`skills/\` as \`.md\` files. Document techniques you figure out with enough detail to reproduce them.

```markdown
## Lab Notebook
`notebook.md` is your persistent lab notebook. It is APPEND-ONLY.
Every entry must begin with a timestamp:
`## 2026-04-15 14:32 — Experiment 005 result
```

Maintain a running theory of what the scoring function rewards. Write it down explicitly. Every experiment should either confirm, contradict, or refine it. When evidence contradicts your theory, update the theory — don't explain away the result. A theory that has survived contact with contradicting evidence is more trustworthy than one that hasn't been tested. The theory should evolve throughout the 30 experiments, not stabilize early.

## Record:

\- Your current theory and what this experiment predicts

\- What you planned and why

\- Results and what they mean

\- How the results update your theory

\- What to try next

```markdown
## Loop
1. Re-read `notebook.md` and `results.tsv` in full
2. Plan your next experiment based on everything you've learned
3. Append a planning entry to `notebook.md` (with timestamp)
4. Check `skills/` for relevant techniques
5. Create `libraries/NNN_description/
6. Write and run `generate.py` to produce `sequences_0.txt
7. Run `python prepare.py libraries/NNN_description/`
8. Write `notes.md` in the experiment directory
9. Append result entry to `notebook.md` (with timestamp)
10. Update `results.tsv`
11. Update any relevant skill files in `skills/`
12. `git add -A && git commit -m "NNN_description: mean_r=X.XXX" && git push`
If push fails, commit locally and continue.
13. Stop after 30 experiments total. Write a final summary in
`notebook.md`.
## Evaluation
prepare.py evaluates your library across 14 anonymous test sets
(eval_01 through eval_14). Each returns mean_r, k562_r, hepg2_r,
sknsh_r. **eval_01 is the primary metric.**
## results.tsv format
Tab-separated, one row per experiment:
`experiment eval_01 eval_02 eval_14 time_s description`
```

# MPRAbox dry-oracle instructions: symbolic

mprabox/dry\_oracles/instructions/symbolic.md

```markdown
# String Optimization
## Objective
You have access to a black-box scoring function that takes 50,000
strings of length 200 over the alphabet {0, 1, 2, 3} and returns
scores. Your goal is to maximize those scores.
You have 30 submissions. Use them wisely.
## How it works
1. Create a new directory under `libraries/` for each experiment
2. Write a `generate.py` that produces exactly 50,000 strings
and saves them to `sequences_0.txt
3. Run `python prepare.py libraries/NNN_name/`
4. prepare.py scores your submission and writes `result.json
## Format
`sequences_0.txt` must contain exactly 50,000 lines, each exactly
200 characters from {0, 1, 2, 3}. Example line:
`01230123012301230123...` (200 characters)
## Experiment directory structure
libraries/
001_description/
generate.py
sequences_0.txt
result.json
notes.md
002_next_idea/
skills/
## Rules
- You CANNOT modify `prepare.py` or any other existing files
You CAN create any new files and directories
You CAN install packages, write any code
`prepare.py` is a black box. Do not read it or inspect it.
This directory is a git repository. After every completed experiment
(result.json written), immediately run:
`git add -A && git commit -m "NNN_description: mean_r=X.XXX" && git pu
This is the only branch. Do not look at or move to any other branch.
```

```markdown
## Skills
Maintain reusable skills in `skills/` as `.md` files. Document
techniques you figure out with enough detail to reproduce them.
```

```markdown
## Lab Notebook
`notebook.md` is your persistent lab notebook. It is APPEND-ONLY.
Every entry must begin with a timestamp:
`## 2026-04-15 14:32 — Experiment 005 result
```

Maintain a running theory of what the scoring function rewards. Write it down explicitly. Every experiment should either confirm, contradict, or refine it. When evidence contradicts your theory, update the theory — don't explain away the result. A theory that has survived contact with contradicting evidence is more trustworthy than one that hasn't been tested. The theory should evolve throughout the 30 experiments, not stabilize early.

## Record:

\- Your current theory and what this experiment predicts

\- What you planned and why

\- Results and what they mean

\- How the results update your theory

\- What to try next

```markdown
## Loop
1. Re-read `notebook.md` and `results.tsv` in full
2. Plan your next experiment based on everything you've learned
3. Append a planning entry to `notebook.md` (with timestamp)
4. Check `skills/` for relevant techniques
5. Create `libraries/NNN_description/`
6. Write and run `generate.py` to produce `sequences_0.txt`
7. Run `python prepare.py libraries/NNN_description/`
8. Write `notes.md` in the experiment directory
9. Append result entry to `notebook.md` (with timestamp)
10. Update `results.tsv`
11. Update any relevant skill files in `skills/`
12. `git add -A && git commit -m "NNN_description: mean_r=X.XXX" && git push`
If push fails, commit locally and continue.
13. Stop after 30 experiments total. Write a final summary in
`notebook.md`.
## Evaluation
prepare.py evaluates your submission across 14 anonymous test sets
(eval_01 through eval_14). Each returns mean_r, k562_r, hepg2_r,
sknsh_r. **eval_01 is the primary metric.** The column names are
arbitrary labels.
## results.tsv format
Tab-separated, one row per experiment:
```

<table><tr><td>`experiment</td><td>t eval_01 eval_02</td><td></td><td></td><td></td><td>eval_14 time_s description`</td></tr><tr><td colspan="6">Record the mean_r for each eval set.</td></tr></table>

## CodonBox instructions

## codonbox/instructions.md

<table><tr><td># The Biology of a New World You are studying life that arose on a world of its own. Its biology evolved</td></tr><tr><td>independently, and its rules are not assumed to match any you already know. You are here to discover those rules. You can submit a sequence and measure a single number: how well the product that sequence specifies functions in this world. A higher number is a better-functioning</td></tr><tr><td>product. Your aim is to design sequences that function well and to work out what governs how well a sequence functions.</td></tr><tr><td>You will be told which characters a sequence may use and how many characters each</td></tr><tr><td>sequence must contain. Sequences in this world are exactly {n chars} characters long. Everything else is for you to determine by experiment, and you have a fixed number of experiments. You must use all of them - there is no early stopping. After</td></tr><tr><td></td></tr><tr><td>you have a working theory, use the remaining experiments to test edge cases, probe predictions, and refine your understanding.</td></tr></table>

## CodonBox harness

codonbox/harness.py

```python
import argparse
import json
import os
import time
from typing import List, Dict
import anthropic # pip install anthropic
from world import build_catalogue, code_summary
from oracle import Oracle
def load_instructions(path="instructions.md"):
with open(path) as f:
return f.read()
def tool_specs(alphabet: str) -> List[dict]:
# Deliberately minimal. No mention of codons, residues, length, or folding.
return [
{
"name": "query",
"description": (
"Submit one sequence to the organism and receive a number "
"(the outcome). Higher is better. The sequence may use only "
f"these characters: {', '.join(alphabet)}. "
"Only ONE query is performed per turn — if you issue more "
"than one query in a single message, only the first will be "
"executed and the rest will be refused. After each query you "
"may write to your notebook and then issue the next query."
),
"input_schema": {
"type": "object",
"properties": {
"sequence": {"type": "string",
"description": f"A string over {{{',
'.join(alphabet)}}}."},
"rationale": {"type": "string",
"description": "What hypothesis this experiment
tests."},
},
"required": ["sequence", "rationale"],
},
},<sub>{</sub>
"name": "write_notebook",
"description": (
```

```python
"Append an entry to your lab notebook. APPEND-ONLY: never "
"rewrite prior entries. Record your current theory, what your "
"last experiments showed, and what you will test next and why."
),
"input_schema": {
"type": "object",
"properties": {"entry": {"type": "string"}},
"required": ["entry"],
},
},
]
def _mark_caching(messages):
"""Set ephemeral cache_control on the trailing block of the LAST user
message; strip it from any earlier user messages.
Why this pattern: Anthropic prompt caching uses a marker on the last block
that should be cached. Everything before & including that marker becomes
the cache prefix for the *next* request. Each turn we move the marker
forward, so each call reads the full prior conversation from cache and
only processes the newest delta. The API caps at 4 breakpoints, so we
keep exactly one rolling user-message breakpoint (plus the system-prompt
breakpoint set elsewhere)."""
last_user_idx = None
for i, m in enumerate(messages):
if m["role"] != "user":
continue
# Normalize bare-string content into list form so we can attach a
# cache_control field to the last block.
if isinstance(m["content"], str):
m["content"] = [{"type": "text", "text": m["content"]}]
# Strip cache_control from earlier user messages.
for blk in m["content"]:
if isinstance(blk, dict) and "cache_control" in blk:
del blk["cache_control"]
last_user_idx = i
if last_user_idx is not None:
last_blocks = messages[last_user_idx]["content"]
if isinstance(last_blocks, list) and last_blocks and
isinstance(last_blocks[-1], dict):
last_blocks[-1]["cache_control"] = {"type": "ephemeral"}
def call_model(client, model, system, messages, tools, max_tokens=4096):
# System prompt is cacheable too — wrap as a list block so we can mark it.
if isinstance(system, str):
system = [{"type": "text", "text": system,
"cache_control": {"type": "ephemeral"}}]
_mark_caching(messages)
return client.messages.create(
model=model, max_tokens=max_tokens, system=system,
tools=tools, messages=messages,
1
```

```python
def run(
world_name: str,
n_residues: int = 16,
budget: int = 300,
model: str = "claude-opus-4-7",
out_dir: str = "runs",
table_path: str = "table_16.bin",
contacts_path: str = "contacts_16.bin",
instructions_path: str = "instructions.md",
seed_tag: str = "",
):
cat = build_catalogue()
if world_name not in cat:
raise SystemExit(f"unknown world {world_name}; choices: {list(cat)}")
world = cat[world_name]
label = world_name + (f"_{seed_tag}" if seed_tag else "")
run_dir = os.path.join(out_dir, label)
os.makedirs(run_dir, exist_ok=True)
notebook_path = os.path.join(run_dir, "notebook.md")
transcript_path = os.path.join(run_dir, "transcript.jsonl")
query_log = os.path.join(run_dir, "queries.jsonl")
state_path = os.path.join(run_dir, "state.json")
# Auto-resume if a prior checkpoint exists; otherwise fresh start.
resuming = os.path.isfile(state_path)
if not resuming:
open(notebook_path, "w").close()
open(transcript_path, "w").close()
# Ground truth for the human's post-hoc read (NEVER shown to the agent).
with open(os.path.join(run_dir, "GROUND_TRUTH.json"), "w") as f:
json.dump(code_summary(world), f, indent=2)
oracle = Oracle(world, n_residues=n_residues, table_path=table_path,
contacts_path=contacts_path, log_path=query_log,
resume=resuming)
# Load instructions and substitute per-world placeholders. Each world
# exposes its own sequence-length (codon_length * n_residues) so the
# agent isn't forced to discover the magic length cold.
system = load_instructions(instructions_path)
n_chars = world.codon_length * n_residues
system = system.replace("{n_chars}", str(n_chars))
# Vertex (sabeti-ai, global endpoint) — same auth path as MPRAgent / Fable
# runs in this project. Pull project/region from env so they can be
# overridden per-run without code edits.
client = anthropic.AnthropicVertex(
project_id=os.environ.get("ANTHROPIC_VERTEX_PROJECT_ID", "sabeti-ai"),
region=os.environ.get("CLOUD_ML_REGION", "global"),
)
tools = tool_specs(world.alphabet)
```

```python
# The intro reveals ONLY the alphabet and the budget. Nothing structural.
intro = (
f"You may use the characters {{{', '.join(world.alphabet)}}}. "
f"You have {budget} experiments. Begin by writing an initial notebook "
f"entry with your starting assumptions and first experiment, then start."
)
if resuming:
# Restore prior conversation + counters. Serialized form is plain dicts
# (Anthropic API accepts dict content blocks for replay).
with open(state_path) as f:
state = json.load(f)
messages: List[Dict] = state["messages"]
oracle.n_queries = state["n_queries"]
oracle.best_fitness = state["best_fitness"]
oracle.best_dna = state["best_dna"]
print(f"[resume] picking up at query {oracle.n_queries}/{budget} "
f"(best fitness so far: {oracle.best_fitness})", flush=True)
else:
messages: List[Dict] = [{"role": "user", "content": intro}]
def log_transcript(obj):
with open(transcript_path, "a") as f:
f.write(json.dumps(obj) + "\n")
def _serialize_content(content):
# content can be a string, list of dicts (user/tool_result), or list of
# SDK content blocks (assistant). Normalize to plain JSON-able dicts.
if isinstance(content, str):
return content
out = []
for c in content:
if isinstance(c, dict):
out.append(c)
else:
out.append(c.model_dump())
return out
def save_state():
# Atomic write so an interrupt mid-write can't leave a half-file.
snap = {
"messages": [{"role": m["role"],
"content": _serialize_content(m["content"])}
for m in messages],
"n_queries": oracle.n_queries,
"best_fitness": oracle.best_fitness,
"best_dna": oracle.best_dna,
}
tmp = state_path + ".tmp"
with open(tmp, "w") as f:
json.dump(snap, f)
os.replace(tmp, state_path)
def append_notebook(entry, tag=None):
```

```python
stamp = time.strftime("%Y-%m-%d %H:%M")
header = f"FINAL" if tag == "final" else f"query {oracle.n_queries}"
with open(notebook_path, "a") as f:
f.write(f"\n## {stamp} — {header}\n\n{entry}\n")
while oracle.n_queries < budget:
resp = call_model(client, model, system, messages, tools)
u = resp.usage
# Track cache hit rate so it's visible in the run log.
cr = getattr(u, "cache_read_input_tokens", 0) or 0
cw = getattr(u, "cache_creation_input_tokens", 0) or 0
log_transcript({"role": "assistant",
"content": [b.model_dump() for b in resp.content],
"usage": {"input": u.input_tokens, "output":
u.output_tokens,
"cache_read": cr, "cache_write": cw}})
messages.append({"role": "assistant", "content": resp.content})
# Process ANY tool_use blocks in the response, regardless of
# stop_reason. Even when stop_reason is "max_tokens" or "refusal", the
# response may still contain a complete tool_use block that the API
# requires us to answer with a matching tool_result; skipping that here
# leaves an orphaned tool_use that 400s on the next call.
has_tool_use = any(b.type == "tool_use" for b in resp.content)
if not has_tool_use:
remaining = budget - oracle.n_queries
messages.append({"role": "user", "content":
f"You did not issue any tool call. You have "
f"{remaining} experiments remaining. You must use "
f"all of them — there is no early stopping. Even "
f"after you have a working theory, use remaining
f"experiments to test edge cases, probe your "
f"predictions, and refine your understanding. "
f"Issue your next query now."})
save_state()
continue
tool_results = []
query_done_this_turn = False # enforce one-probe-per-turn
for block in resp.content:
if block.type != "tool_use":
continue
if block.name == "query":
if query_done_this_turn:
# Refuse extra queries in the same turn; the API still
# requires a tool_result for every tool_use block.
tool_results.append({"type": "tool_result",
"tool_use_id": block.id,
"content": json.dumps({
"ok": False,
"refused": True,
"reason": ("Only one query per turn is
permitted. "
```

```python
"This experiment was NOT
performed and "
"no budget was consumed for
it. "
"Re-issue it next turn if
you still want to."),
"experiments_used": oracle.n_queries,
"experiments_left": budget -
oracle.n_queries,
})})
continue
seq = block.input.get("sequence", "")
result = oracle.query(seq)
payload = {"ok": result["ok"], "fitness": result["fitness"],
"experiments_used": oracle.n_queries,
"experiments_left": budget - oracle.n_queries}
tool_results.append({"type": "tool_result",
"tool_use_id": block.id,
"content": json.dumps(payload)})
query_done_this_turn = True
elif block.name == "write_notebook":
append_notebook(block.input.get("entry", ""))
tool_results.append({"type": "tool_result",
"tool_use_id": block.id,
"content": "notebook entry appended."})
else:
# Unknown / hallucinated tool name — still must answer it.
tool_results.append({"type": "tool_result",
"tool_use_id": block.id,
"content": json.dumps({
"ok": False,
"error": f"unknown tool {block.name!r}; "
f"available tools are: query,
write_notebook",
}),
"is_error": True})
# If the turn used tools (e.g. write_notebook) but did NOT run a query,
# push back the same way as a no-tool turn: budget must be exhausted.
if not query_done_this_turn:
remaining = budget - oracle.n_queries
tool_results.append({"type": "text",
"text": (f"You did not run an experiment this turn. You have "
f"{remaining} experiments remaining. You must use all "
f"of them — there is no early stopping. Even after you "
f"have a working theory, use remaining experiments to "
f"test edge cases, probe your predictions, and refine
f"your understanding. Issue your next query now.")})
messages.append({"role": "user", "content": tool_results})
save_state()
if oracle.n_queries >= budget:
messages.append({"role": "user", "content":
"Your experiment budget is exhausted. Write your "
"final notebook entry: your best account of how "
```

```python
"this organism works, your best sequence, and what "
"you would test next."})
final = call_model(client, model, system, messages, tools)
log_transcript({"role": "assistant_final",
"content": [b.model_dump() for b in final.content]})
for block in final.content:
if block.type == "tool_use" and block.name == "write_notebook":
append_notebook(block.input.get("entry", ""), tag="final")
break
summary = {
"world": world_name, "n_residues": n_residues, "budget": budget,
"queries_used": oracle.n_queries,
"best_fitness": oracle.best_fitness, "best_sequence": oracle.best_dna,
}
if oracle.table and oracle.table.loaded:
_, best_fit = oracle.table.true_optimum()
summary["true_optimum_fitness"] = best_fit
summary["fraction_of_optimum"] = (
round(oracle.best_fitness / best_fit, 4)
if (oracle.best_fitness and best_fit) else None)
with open(os.path.join(run_dir, "summary.json"), "w") as f:
json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))
print(f"\nRun complete. Read {notebook_path} against "
f"{os.path.join(run_dir, 'GROUND_TRUTH.json')} for the qualitative
read.")
if __name__ == "__main__":
ap = argparse.ArgumentParser()
ap.add_argument("world")
ap.add_argument("--budget", type=int, default=300)
ap.add_argument("--n", type=int, default=16, dest="n_residues")
ap.add_argument("--model", default="claude-opus-4-7")
ap.add_argument("--table", default="table_16.bin")
ap.add_argument("--contacts", default="contacts_16.bin")
ap.add_argument("--out", default="runs")
ap.add_argument("--instructions", default="instructions.md")
ap.add_argument("--seed-tag", default="", dest="seed_tag")
args = ap.parse_args()
run(args.world, n_residues=args.n_residues, budget=args.budget,
model=args.model, out_dir=args.out, table_path=args.table,
contacts_path=args.contacts, instructions_path=args.instructions,
seed_tag=args.seed_tag)
```