# Reimagining computational quantum research with AI scientists

Yuxuan Du,<sup>1,</sup> <sup>2,</sup> <sup>∗</sup> Hanqing Zhao,<sup>1,</sup> <sup>∗</sup> Xinbiao Wang,<sup>1</sup> Yiming Lu,<sup>1</sup> Kaining Zhang,<sup>1</sup> and Dacheng Tao<sup>1,</sup> <sup>†</sup>

<sup>1</sup>College of Computing and Data Science, Nanyang Technological University, Singapore 639798, Singapore <sup>2</sup>School of Physical and Mathematical Sciences, Nanyang Technological University, Singapore 639798, Singapore

AI scientists are expanding automated research across scientific domains, yet their ability to turn the scientific record into a foundation for autonomous discovery remains elusive. Here we uncover a fundamental barrier to this transition and demonstrate a route to overcome it in com putational quantum science, where advances shape both what we can learn from complex quantum systems and what quantum computing can ultimately deliver. To identify this barrier, we introduce OmniQ, an expert-audited virtual research environment that converts published studies into executable reproduction tasks. Across 59 papers spanning classical simulation, quantum algorithms and quantum-property characterization, we systematically evaluate five agentic systems powered by large language models and six third-party agent harnesses. Our findings reveal that current AI scientists often fail to reproduce published results, exposing a fundamental mismatch between what scientific papers communicate and what autonomous agents require. By contrast, human expert intervention can turn these failures into successful reproductions, showing that the missing information can be identified and converted into reusable research experience. Guided by this observation, we develop Residual Agents (RA), a quantum-tailored self-improving AI scientist that distils externally verified workflow repairs into such experience. RA substantially improves research performance under fixed base models and limited adaptation budgets. For example, RA raises DeepSeek V4.1 Flash performance from 68% to 85%, matching Claude Code with Opus 5 at a third of its cost despite using a weaker base model. We integrate all reproduced results into an extensible platform built on the model context protocol, making verified research workflows directly accessible to AI scientists and allowing new scientific papers to be continuously incorporated. These results establish a new paradigm for computational quantum research, where published results become reusable scientific capabilities that support autonomous discovery.

<sup>-</sup> Project page: https://omniq-qid-lab.github.io/

## INTRODUCTION

Computational quantum science develops algorithms to understand and exploit complex quantum systems [1–3], spanning classical simulation [4–6], quantum algorithms [7–9], and property characterization of quantum systems [10– 12]. Despite this promise, algorithm design across all three directions is becoming increasingly dificult as quantum systems and hardware continue to scale. For classical simulation, pushing the limits of classical tractability while establishing reliable baselines for practical utility increasingly relies on sophisticated methods such as tensor networks [13–15] and Pauli-propagation techniques [16–20]. For both near-term and fault-tolerant quantum algorithms, achieving computational advantage remains challenging under realistic resource constraints, requiring careful co-design of algorithms and hardware capabilities [21–24]. For quantum property characterization, extracting informative properties from limited measurements becomes dificult as the Hilbert space grows exponentially, motivating eficient measurement and inference protocols [25–30]. As a result, these challenges create an urgent need for more autonomous approaches to computational quantum science.

Recent breakthroughs in AI scientists ofer a potential route towards autonomous computational quantum sci ence [31–33]. By integrating frontier large language models (LLMs) with planning, tool use and iterative feedback [34–37], these systems have begun to automate algorithm discovery in mathematics [38–40] and software engineer ing [41–44]. This trend is also extending to other disciplines, such as biology [45–47], chemistry [48–50] and materials science [51]. All of these advances point to a shift from AI-assisted research towards autonomous algorithm discovery. Computational quantum science is a natural next arena because progress in this field may reshape both what we can learn from quantum systems and what quantum computing can ultimately deliver. In this context, early eforts have demonstrated AI-driven discovery of quantum error-correcting codes [52–54], automated design of quantum circuits [55–57] and algorithmic search for quantum-chemistry applications [58–61]. However, the progress remains fragmented and largely confined to narrowly defined settings, such as a single subproblem, a human-tailored workflow or a search space strongly constrained by prior knowledge. This raises a central question:

![](images/4328c336fab09e4df9a870179075f1fa350a136311f33755cb4beab4e56ff69d.jpg)  
FIG. 1. Paper-to-result reproduction in computational quantum science. a, Scientific papers from three research directions in computational quantum science are converted into reproduction tasks centred on a target figure within OmniQ, an expert-audited virtual research environment. In OmniQ, a plan agent and an execute agent reproduce the source code and the target figure, and a judge agent compares the reproduced figure with the published one. $\mathbf { b } ,$ The mechanism of the self-improving AI scientist RA for computational quantum science. The planner decomposes a research plan into sub-plans, each executed by an executor and evaluated by a critic. Failed steps are revised and re-executed, with the planner replanning when necessary. A judge evaluates the assembled result. Upon failure, expert feedback is processed by the residual agent to refine executor guidance, critic criteria and planner strategies.

Can AI scientists drive the next wave of discovery in computational quantum science?

In this study, we address this question by starting from paper-to-result reproduction, where verifiable outcomes allow AI scientists to accumulate scientific experience before moving to open-ended research. To this end, we introduce OmniQ, an expert-audited virtual research environment that transforms published studies in computational quantum science into executable reproduction tasks. More precisely, given a full paper and a target result, an AI scientist reconstructs the underlying method, implements and executes it under a comparable computational budget, and attempts to reproduce the reported outcome. The workflow is evaluated for consistency with the reported method and outcome, with expert intervention reserved for ambiguous or inconsistent cases. In doing so, OmniQ turns paperto-result reproduction into a controlled and diagnosable research setting.

Using OmniQ, we systematically evaluate five LLM-powered AI scientists and six third-party agent harnesses across 60 papers spanning computational quantum science. Diferent from recent paper-to-agent approaches [62] that transform papers, code and data into executable agentic systems, here we consider a more direct reproduction setting in computational quantum science. That is, the agent must reconstruct the workflow underlying a reported result. Experimental results show that current AI scientists frequently fail at this task. Moreover, human expert review reveals a deeper problem. That is, in 18 studies, the paper description, released code and reported results are inconsistent. Such cases obscure the boundary between agent failure and limitations of the scientific record, posing a barrier to reliable autonomous discovery in computational quantum science.

After separating source-side inconsistencies from genuine agent failures, we find that human expert intervention can often turn failed attempts into successful reproductions, revealing domain knowledge that can be learned and reused. Building on this observation, we develop Residual Agents (RA), an eficient self-improving AI scientist that distils externally verified workflow repairs into reusable quantum research experience. Unlike generic evolution mechanisms, RA accumulates domain-specific lessons from successful repairs and applies them to subsequent research. Across base LLMs, RA substantially improves performance under a limited adaptation budget, with gains that generalize to unseen tasks and transfer across models. We further integrate all reproduced results into an extensible platform built on the model context protocol, enabling AI scientists and the broader research community to directly access, reuse, and extend verified research workflows. Collectively, these advances recast paper-to-result reproduction from a one-of translation problem into a foundation for reusable scientific capabilities and continual learning, opening a path towards autonomous discovery in computational quantum science and potentially beyond.

## RESULTS

## Failures of paper-to-result reproduction

Autonomous computational quantum science envisions AI scientists that can independently understand and develop computational methods for quantum systems. Towards this goal, a crucial step is reproduction, which can accelerate the validation and reuse of existing methods while reducing the expert efort required to reconstruct complex computational workflows. We therefore ask whether an AI scientist can reconstruct the underlying workflow from the scientific record and faithfully reproduce a reported result. We refer to this setting as paper-to-result reproduction.

We study paper-to-result reproduction using OmniQ, a virtual research environment that turns published computational quantum studies into executable and diagnosable reproduction tasks. As illustrated in Fig. 1a, OmniQ is employed to curate 60 studies spanning classical simulation, quantum algorithms and quantum-property characterization. The studies are selected to balance scientific impact, publication in leading venues, methodological diversity, and reproducibility under realistic computational budgets, with approximately even coverage across the three research directions. More details are provided in Supplementary Information (SI) A and B.

To faithfully assess reproduction quality, we classify reproduction quality into three levels according to how well the central scientific claim is preserved through reconstruction. Specifically, Level I captures missing methodological details that can be recovered from the released code or expert interpretation. Level II captures localized errors in the paper that can be resolved from the code, figures or broader scientific context, but otherwise lead to reproduction discrepancies. Level III captures fundamental inconsistencies among the context, figures and released code that cannot be resolved through expert inspection, leaving the target result unreproducible from the available scientific record.

The reproduction experiment in OmniQ assigns eight agent configurations to each study, each tasked with independently reproducing a designated result under a computational budget comparable to that of the original work. Across representative frontier models and agentic systems, the reconstructed workflows and reported results are cross-checked across agents, compared against available source code, and independently verified by domain experts (See SI B). As shown in Fig. 2a, this procedure identifies reproducibility issues in 18 of the 60 studies. Below, we examine three representative studies from each reproduction level to illustrate the distinct failure modes.

![](images/61f564b0c3ca9cb506026aa38dc08fc2901e31fe646f85e251c9a05274f71615.jpg)

b  
Level 1 Paper ID #46  
![](images/fb2c9bc4fce451492988cebc903eb2c4ade54b4f9b0cf4091b9b0f5b3f14b68b.jpg)

One feature is considered for the best split

## Contradiction found by AI scientist

With one feature, the forest loses its edge in extrapolation while the authors' code uses all features

## Edited by experts

all features at every split (max\_features = 1.0)

![](images/31a00b8dd21a2c53d465331ce3e3ab21a9121545a0480c3f8c35c8b6a4c401ae.jpg)

![](images/437ff62c15a2fecdc0956087560df52716374f5160decdf5e7bfcd241c4a7eec.jpg)

d  
c  
Level 2 Paper ID #42  
![](images/751e8f481c7171ea2b4a91bd1684b4c3a386d6fd0692d2d32fd3c0ac0a8e00e9.jpg)

Silhouette of both kernels vs circuit depth

## Contradiction found by AI scientist

Several readings of the text; The values need two steps the text never states

## Edited by experts

One reading fixed in the task book; judged on the stated orderings, not the values

Level 3 Paper ID #60  
![](images/010d0f6b9c78e4d05a7e893cb0f21a88d9e0acf7f549d47f6c45c504fcacae81.jpg)

![](images/305f9f2b0161b573f738b5154528bb1346104b75579b30c57751210feaa570a0.jpg)

![](images/a662f2c06ad1c6b616f6a3e8aa75efd926086f6bf14df0389db75a31fa8821ec.jpg)

## Fig. 6b

## Paper's claim

kernel variance Var[K]

Contradiction found by AI scientist (1) Plots the variance of C in K=exp(C); n=2: 0.49 versus 0.25 (2) Source code at n=12: 13x and 57 x below the paper at L=2 and 4

## Edited by experts

Task removed from the benchmark

![](images/2af83fc6746400e2bac50cca5058e5021039e445aee5102e93ed0dd64001da6f.jpg)

![](images/66c67850c79595ae696494450e4a83b82001ae41b05f42f774f617c440d955ea.jpg)  
FIG. 2. AI-assisted reproduction reveals inconsistencies in the scientific record. a, Reproduction assessment across three directions of computational quantum science. Grey denotes studies with no identified problem, whereas increasingly dark red indicates Level I–III issues according to their severity and resolvability. b–d, Representative Level I–III cases. From top to bottom, each case shows the scientific problem, the identified reproduction issue—including the paper’s claim, the contradiction found by the AI scientist and expert feedback—and the reproduced results before and after expert correction, each compared with the original result. For the panel b, a machine-learning configuration discrepancy resolved using the released code. For the panel $\mathbf { c } ,$ an ambiguous experimental procedure requiring expert resolution. For the panel d, a fundamental inconsistency between the reported quantity, released code and theoretical constraints that cannot be resolved from the available scientific record. Dashed black and solid red curves denote the reported and reproduced results, respectively.

Level I is exemplified by Paper #46 [63] on quantum error mitigation [64], exploring whether learning-based mitigation can recover accurate observable expectation values beyond the training regime, enabling extrapolation to deeper quantum circuits without the full overhead of fault-tolerant error correction [65]. Its Figure 3 directly probes this claim by tracking prediction errors across Trotter steps within and beyond the training region for Hamiltonian simulation tasks. The reproduction with OmniQ reveals a localized but consequential inconsistency. That is, the paper states that only one feature is considered when selecting the best split of the machine-learning model, whereas the released code uses all available features at each split (i.e., max features=1.0). As shown in the third panel of Fig. 2b, following the written description causes the reproduced model to lose its reported advantage in the extrapolation regime. In contrast, as shown in the bottom panel, following the released code resolves this discrepancy and recovers the reported extrapolation advantage.

Level II is exemplified by Paper #42 [66] on quantum-property characterization, where eficient learning methods are developed to identify topological order in many-body quantum systems without prior phase labels. Its Figures 3b and 3d quantify the robustness of the proposed fidelity and entanglement kernels under increasingly deep two-qubit random circuits using the silhouette coeficient, a standard metric of clustering quality [67]. Reproduction with OmniQ fails to recover the reported quantitative values. As shown in the third panel of Fig. 2c, the silhouette coeficient, which measures clustering quality from −1 (poor) to 1 (well separated), decreases from approximately 0.59 in the paper to 0.29 in the reproduction at circuit depth four. Expert inspection reveals that the experimental procedure is incompletely specified, as recovering the reported value requires an additional two-step procedure that is not described in the paper. Moreover, as shown in the last panel, applying the procedure described in the paper to the authors own data yields a silhouette coeficient of only 0.24. After expert editing focuses the task on the reported scientific conclusion instead of unrecoverable numerical values, RA reproduces the reported degradation of the fidelity kerne and the consistent advantage of the entanglement kernel.

Level III is represented by Paper #60 [68] on quantum machine learning, which studies whether quantum kernels remain eficiently distinguishable as the number of qubits increases. This question is closely tied to whether quantum kernels can ultimately support quantum advantage in learning [69]. The scientific paper claims that highly expressive data encodings can cause quantum kernels to concentrate exponentially, making their estimation increasingly costly and ultimately limiting their usefulness for learning. Its Figure 6b provides numerical evidence for this claim by plotting the variance of the projected quantum kernel ${ \mathsf { K } } ^ { \mathrm { P Q } }$ as a function of qubit number n and circuit depth L. However, reproduction with OmniQ reveals a fundamental inconsistency: although the figure is described as reporting $\mathrm { V a r } [ \mathsf { K } ^ { \mathrm { P Q } } ]$ , the released code instead evaluates the variance of the distance C entering the kernel ${ \sf K } ^ { \mathrm { P Q } } = \exp ( - \gamma C )$ As shown in the third panel of Fig. 2d, the discrepancy is already evident at $n = 2 .$ , where the reported value reaches 0.49, exceeding the theoretical upper bound of 0.25 on the variance of a kernel taking values in [0, 1]. A separate discrepancy concerns the generation of the data underlying the figure. Even when Var[C] was used as the target, rerunning the released code fails to recover the published values. For the case of n = 12 qubits, the published variances at both L = 2 and L = 4 are more than an order of magnitude larger than the recalculated values, as shown in the last panel. Expert inspection identified the statistical error but did not establish the configuration used to generate the archived data. Unlike Levels I and II, the discrepancy between the released code and the published figure remains unresolved, preventing faithful reproduction of the target result from the available scientific record.

## Performance across frontier AI scientists

We next explore how existing AI scientists perform on a cleaned computational quantum science dataset. Following expert review, resolvable Level I and II issues are corrected, whereas the unresolved Level III case (i.e., Paper #60) is excluded, yielding a consistent dataset of 59 papers for agent evaluation. Using this cleaned dataset and OmniQ, we systematically evaluate leading base models from the GPT, Claude, DeepSeek, Kimi and Grok families [70–74]. We also consider first-party agentic systems, including Codex, Claude Code, Kimi Code, Grok Build and DeepSeek Harness (DSH), together with six widely used third-party agent harnesses, i.e., InternAgent, DeepCode, mini-sweagent, live-swe-agent, Huxley–G¨odel Machine (HGM) and Meta-Harness [75–80]. A reproduction task is counted as successful only when the reconstructed workflow satisfies the strict reproduction criterion defined above.

Figure 3 summarizes reproduction success together with inference cost per task and wall-clock time across diferent agents and harness mechanisms. A first observation is that performance varies markedly across frontier agentic systems. In particular, Claude Code with Opus 5.5 achieves the highest reproduction rate, recovering approximately 93% of the computational quantum science dataset. Besides, Codex paired with several frontier base models exceeds 70% reproduction success. At the other end of the spectrum, Codex with GPT-6-luna reaches only about 42%, and Meta-Harness with DeepSeek performs worst at 19%. These results indicate that once the scientific record is self-consistent, frontier AI scientists already possess the capability for autonomous computational quantum science.

The achieved results further reveal two unexpected findings. First, reproduction success does not increase monotonically with inference cost. As shown in Fig. 3, Claude Code with Sonnet 5.5, Kimi Code with K3, and severa Codex-based systems use larger inference budgets than Claude Code with Opus 5.5 yet perform worse. Meanwhile, DSH with DeepSeek V4.1 Flash achieves approximately 68% success at the lowest cost, outperforming every other system below \$7 per task and several AI scientists costing an order of magnitude more. This observation suggests that higher inference spending often translates into more deliberation but not more execution, a “thinking more but doing less” behavior that echoes recent observations of overthinking in agentic reasoning [81]. Collectively, this decoupling identifies the allocation of inference budgets between reasoning and scientific execution as a central determinant of

![](images/644f98172c5b8cc605bd98541dd3362e2dab55435095d5f195a361c5ec133f0f.jpg)  
FIG. 3. Performance of RA for computational quantum science. Reproduction success across 59 reproduction tasks is plotted against model-inference cost per task on a logarithmic scale. Marker shape denotes the agent system or harness type, marker color denotes the underlying model family or access mode, and marker size indicates wall-clock time per task; semitransparent markers denote earlier model generations. Labels identify the corresponding model or harness. RA, the proposed self-improving AI scientist, runs DSH as its executor and is drawn as the DSH marker inside a gear; the dashed line indicates the Pareto frontier between reproduction success and inference cost. A brown ring marks GPT models accessed through an application programming interface (API).

autonomous computational science.

A second unexpected finding is that existing agent harnesses do not necessarily improve reproduction and can even degrade performance. As illustrated in Fig. 3, third-party harnesses span a wide performance range, from about 19% for Meta-Harness with DeepSeek to 48% for mini-swe-agent with DeepSeek, all substantially below the strongest first-party systems. Even on the same base LLM model, none of the six exceeds the vendor’s own harness. With DeepSeek V4 Pro, their reproduction success ranges from 19% to 48%, compared with 49% for DSH. Failure analysis reveals common behaviors, i.e., agents may declare success without validating the reproduced result against the paper, replace the required computation with a cheaper surrogate, or silently reduce the problem size. Therefore, harness mechanisms tuned on other tasks thus do not transfer to computational quantum science [82, 83], which calls for scientific experience grounded in the computational quantum science domain.

## RA: Self-improving AI scientists

The performance gap between proprietary and open-source AI scientists in Fig. 3 underscores a dilemma. On the one hand, leading proprietary systems achieve strong reproduction performance, but their closed implementations limit the transparency and adaptability of scientific workflows. Conversely, existing open-source alternatives struggle to reproduce published results reliably, limiting their utility for computational quantum science.

To bridge this gap, we develop Residual Agents (RA), an open-source, self-improving AI scientist for autonomous computational quantum science that learns from verified failures to improve future reproductions. As shown in Fig. 1b, RA splits a reproduction task into sub-plans and coordinates three roles powered by the same base LLM. Specifically, a planner develops a resource-bounded plan, executors carry out each sub-plan, and a critic validates the results against the paper. When a step fails, it is repaired or replanned based on the critic’s assessment. Beyond resolving individual failures, RA identifies the residual between failed executions and their verified corrections, distilling these diferences into generalizable failure signatures stored in a residual memory alongside the evolving memories of the planner and critic. As such, adaptation is confined to the experience layer and driven only by verified evidence. To ensure reliable evolution, a human expert reviews each proposed update before it is incorporated, while the underlying model, executors, and orchestration remain fixed. Further details and its associated research platform are provided in SI C and SI D, respectively.

We next evaluate RA across all 59 reproduction tasks. During development, RA uses only the five most challenging tasks, whose verified failures form its experience. The remaining 54 tasks are unseen during development and evaluate how well this experience generalizes. For evaluation, RA uses DeepSeek V4.1 Flash with DSH as the executor and is given 24 hours per task. We compare it directly with DSH using the same base model and computational budget, with all systems evaluated under the same criteria as in the preceding reproduction experiments.

Figure 3 compares RA with existing frontier AI scientists, with further details provided in SI E. In particular, RA improves the reproduction success of DeepSeek V4.1 Flash from 68% to approximately 85%, at an inference cost of only \$5.6 per task. This performance matches Claude Code with Opus 5 and Codex with GPT-5.6-sol API, which achieve approximately 85% success at \$16 and \$35 per task, respectively. Although the best-performing system stil achieves a higher success rate, RA reaches competitive performance at substantially lower inference cost.

The internal trajectories further reveal how RA improves reproduction. Across the 59 tasks, the critic rejects more than half of intermediate results, triggering repair or replanning. Notably, 90% of successful tasks require at least one correction, including all 13 tasks recovered beyond DSH. These results show that RA gains its advantage primarily by identifying and correcting failures. More experimental results are deferred to SI E.

## DISCUSSION

Our study establishes paper-to-result reproduction as a foundation for developing autonomous AI scientists. Supported by OmniQ, we show that reliable reproduction in computational quantum science depends jointly on the capabilities of AI scientists and the consistency of the scientific record. We further develop RA, an open-source AI scientist that converts verified reproduction failures into reusable scientific experience, enabling improvement without modifying the underlying model. By making verified reproduction workflows accessible through an extensible research platform, we further lay the groundwork for a community-driven ecosystem of reusable scientific capabilities.

Three important directions remain for future research. The first is to extend OmniQ towards fully autonomous paperto-result reproduction across scientific disciplines. Recent studies have explored AI-driven scientific reproduction in diverse domains, including Paper2Agent [62] for computational biology, AutoExperiment [84] for machine learning, and SocSci-Repro-Bench [85] for social sciences. However, our study reveals a fundamental challenge. That is, inconsistencies in the scientific record can make faithful reproduction itself ambiguous, requiring expert intervention to resolve. In this regard, integrating OmniQ and RA with these domain-specific approaches could enable more reliable and autonomous scientific reproduction across disciplines.

A second direction is to further improve the performance of paper-to-result reproduction in computational quantum science. Although RA outperforms several more expensive models at lower cost, achieving fully reliable scientific reproduction with greater eficiency remains an open challenge. Toward this goal, there are three potential solutions. First, adaptive allocation of test-time computation could improve reproduction performance by directing greater computational resources toward challenging tasks [86, 87]. Second, integrating specialized scientific tools and reusable implementations of previously reproduced algorithms could improve the accuracy and reliability of complex scientific reproductions [88]. Third, optimizing the performance–cost trade-of through adaptive model selection and eficient agent orchestration could further reduce the cost of scientific reproduction without compromising reliability [89].

The last direction is to extend RA beyond scientific reproduction towards autonomous algorithm discovery in computational quantum science. Recent studies have demonstrated the potential of AI scientists to discover more eficient quantum circuits through iterative search and independent verification [90–92]. Building on these advances, RA could strengthen autonomous discovery by first establishing reliable computational baselines and then learning from the diferences between unsuccessful candidates and verified improvements. By accumulating these insights as reusable experience, RA could guide subsequent algorithm design and optimization beyond individual search tasks. Extending its self-improvement mechanism from verified reproduction repairs to verified algorithmic advances ofers a promising route towards more reliable and eficient autonomous quantum algorithm discovery.

[1] Iulia M Georgescu, Sahel Ashhab, and Franco Nori. Quantum simulation. Reviews of Modern Physics, 86(1):153–185, 2014.

[2] Sam McArdle, Suguru Endo, Al´an Aspuru-Guzik, Simon C Benjamin, and Xiao Yuan. Quantum computational chemistry. Reviews of Modern Physics, 92(1):015003, 2020.

[3] Yuxuan Du, Yan Zhu, Yuan-Hang Zhang, Min-Hsiu Hsieh, Patrick Rebentrost, Weibo Gao, Yi-Zhuang You, Jens Eisert, Giulio Chiribella, Dacheng Tao, et al. Artificial intelligence for representing and characterizing quantum systems. Nature Reviews Physics, pages 1–18, 2026.

[4] Ulrich Schollw¨ock. The density-matrix renormalization group. Reviews of modern physics, 77(1):259–315, 2005.

[5] Rom´an Or´us. Tensor networks for complex quantum systems. Nature Reviews Physics, 1(9):538–550, 2019.

[6] Aleksandr Berezutskii, Minzhao Liu, Atithi Acharya, Roman Ellerbrock, Johnnie Gray, Reza Haghshenas, Zichang He, Abid Khan, Viacheslav Kuzmin, Dmitry Lyakh, et al. Tensor networks for quantum computing. Nature Reviews Physics, 7(10):581–593, 2025.

[7] Andrew M Childs and Wim Van Dam. Quantum algorithms for algebraic problems. Reviews of Modern Physics, 82(1):1– 52, 2010.

[8] Kishor Bharti, Alba Cervera-Lierta, Thi Ha Kyaw, Tobias Haug, Sumner Alperin-Lea, Abhinav Anand, Matthias Deg roote, Hermanni Heimonen, Jakob S Kottmann, Tim Menke, et al. Noisy intermediate-scale quantum algorithms. Reviews of Modern Physics, 94(1):015004, 2022.

[9] Mauro ES Morales, Lirand¨e Pira, Philipp Schleich, Kelvin Koor, Pedro CS Costa, Dong An, Al´an Aspuru-Guzik, Lin Lin, Patrick Rebentrost, and Dominic W Berry. Quantum linear system solvers: A survey of algorithms and applications. Reviews of Modern Physics, 98(2):025005, 2026.

[10] Giuseppe Carleo, Ignacio Cirac, Kyle Cranmer, Laurent Daudet, Maria Schuld, Naftali Tishby, Leslie Vogt-Maranto, and Lenka Zdeborov´a. Machine learning and the physical sciences. Reviews of Modern Physics, 91(4):045002, 2019.

[11] Jens Eisert, Dominik Hangleiter, Nathan Walk, Ingo Roth, Damian Markham, Rhea Parekh, Ulysse Chabaud, and Elham Kashefi. Quantum certification and benchmarking. Nature Reviews Physics, 2(7):382–390, 2020.

[12] Valentin Gebhart, Rafaele Santagati, Antonio Andrea Gentile, Erik M Gauger, David Craig, Natalia Ares, Leonardo Banchi, Florian Marquardt, Luca Pezze, and Cristian Bonato. Learning quantum systems. Nature Reviews Physics, 5(3):141–156, 2023.

[13] J Ignacio Cirac, David Perez-Garcia, Norbert Schuch, and Frank Verstraete. Matrix product states and projected entan gled pair states: Concepts, symmetries, theorems. Reviews of Modern Physics, 93(4):045003, 2021.

[14] Joseph Tindall, Matthew Fishman, E Miles Stoudenmire, and Dries Sels. Eficient tensor network simulation of ibm’s eagle kicked ising experiment. Prx quantum, 5(1):010308, 2024.

[15] Joseph Tindall, Antonio Francesco Mello, Matthew Fishman, E Miles Stoudenmire, and Dries Sels. Dynamics of disordered quantum systems with two-and three-dimensional tensor networks. Science, 392(6800):868–872, 2026.

[16] Yuguo Shao, Fuchuan Wei, Song Cheng, and Zhengwei Liu. Simulating noisy variational quantum algorithms: A polynomial approach. Physical Review Letters, 133(12):120603, 2024.

[17] Tomislav Beguˇsi´c, Johnnie Gray, and Garnet Kin-Lic Chan. Fast and converged classical simulations of evidence for the utility of quantum computing before fault tolerance. Science Advances, 10(3):eadk4321, 2024.

[18] Tomislav Beguˇsi´c and Garnet Kin-Lic Chan. Real-time operator evolution in two and three dimensions via sparse pauli dynamics. PRX quantum, 6(2):020302, 2025.

[19] Manuel S Rudolph, Tyson Jones, Yanting Teng, Armando Angrisani, and Zo¨e Holmes. Pauli propagation: A computational

framework for simulating quantum systems. PRX Quantum, 7(3):032001, 2026.

[20] Kaining Zhang, Xinbiao Wang, Kunsheng Li, Qixin Zhang, Yuxuan Du, Min-Hsiu Hsieh, and Dacheng Tao. Stochastic pauli-path simulator for large-scale quantum optimization. arXiv preprint arXiv:2607.17804, 2026.

[21] Youngseok Kim, Andrew Eddins, Sajant Anand, Ken Xuan Wei, Ewout Van Den Berg, Sami Rosenblatt, Hasan Nayfeh, Yantao Wu, Michael Zaletel, Kristan Temme, et al. Evidence for the utility of quantum computing before fault tolerance. Nature, 618(7965):500–505, 2023.

[22] Minzhao Liu, Ruslan Shaydulin, Pradeep Niroula, Matthew DeCross, Shih-Han Hung, Wen Yu Kon, Enrique Cervero-Mart´ın, Kaushik Chakraborty, Omar Amer, Scott Aaronson, et al. Certified randomness using a trapped-ion quantum processor. Nature, 640(8058):343–348, 2025.

[23] Dmitry A. Abanin, Rajeev Acharya, Laleh Aghababaie-Beni, Georg Aigeldinger, Ashok Ajoy, Ross Alcaraz, Igor Aleiner, Trond I. Andersen, Markus Ansmann, Frank Arute, Kunal Arya, Abraham T. Asfaw, Nikita Astrakhantsev, Juan Atalaya, Ryan Babbush, Dave Bacon, Brian Ballard, Joseph C. Bardin, Christian Bengs, Andreas Bengtsson, Alexander Bilmes, Sergio Boixo, Gina Bortoli, Alexandre Bourassa, Jenna Bovaird, Dylan Bowers, Leon Brill, Mick Broughton, David A. Browne, Brett Buchea, Bob B. Buckley, David A. Buell, Tim Burger, Brian Burkett, Nicholas Bushnell, Anthony Cabrera, J. Richard Campero, Hung-Shen Chang, Yu Chen, Zi-Jun Chen, Benjamin Chiaro, Liang-Ying Chih, Desmond Chik, Charina Chou, Jahan Claes, Agnetta Y. Cleland, J. Zachery Cogan, Saul Cohen, Roberto Collins, P. N. Conner, William Courtney, Alexander L. Crook, Ben Curtin, Sayan Das, Laura de Lorenzo, Dripto M. Debroy, Sean Demura, Michel H. Devoret, Agustin Di Paolo, Paul Donohoe, Ilya K. Drozdov, Andrew Dunsworth, Clint Earle, Alec Eickbusch, Aviv Moshe Elbag, Mahmoud Elzouka, Catherine Erickson, Lara Faoro, Edward Farhi, Vinicius S. Ferreira, Leslie Flores Burgos, Ebrahim Forati, Austin G. Fowler, Brooks Foxen, Suhas Ganjam, Gonzalo Garc´ıa, Robert Gasca, Elie Genois, William<sup>´</sup> Giang, Craig Gidney, Dar Gilboa, Raja Gosula, A. Grajales Dau, Dietrich Graumann, Alex Greene, Jonathan A. Gross, Hanfeng Gu, Steve Habegger, John Hall, Ikko Hamamura, Michael C. Hamilton, Monica Hansen, Matthew P. Harrigan, Sean D. Harrington, Stephen Heslin, Paula Heu, Oscar Higgott, Gordon Hill, Jeremy P. Hilton, Sabrina Hong, Hsin-Yuan Huang, Ashley N. Huf, William J. Huggins, Lev B. Iofe, Sergei V. Isakov, Justin Iveland, Evan Jefrey, Zhang Jiang, Xiaoxuan Jin, Cody Jones, Stephen P. Jordan, Chaitali Joshi, Pavol Juh´as, Andreas Kabel, Dvir Kafri, Hui Kang, Amir H. Karamlou, Kostyantyn Kechedzhi, Julian Kelly, Trupti Khaire, Tanuj Khattar, Mostafa Khezri, Seon Kim, Robbie King, Paul V. Klimov, Andrey R. Klots, Bryce Kobrin, A. N. Korotkov, Fedor Kostritsa, Robin Kothari, John Mark Kreikebaum, Vladislav D. Kurilovich, Elica Kyoseva, David Landhuis, Tiano Lange-Dei, Brandon W. Langley, Pavel Laptev, Kim Ming Lau, Lo¨ıck Le Guevel, Justin Ledford, Joonho Lee, Kenny Lee, Yuri D. Lensky, Shannon Leon, Brian J. Lester, Wing Li, Alexander T. Lill, Wayne Liu, William P. Livingston, A. Locharla, Erik Lucero, Daniel Lundahl, Aaron Lunt, Sid Madhuk, Fionn D. Malone, Ashley Maloney, Salvatore Mandr´a, James Manyika, Leigh S. Martin, Orion Martin, Steven Martin, Yossi Matias, Cameron Maxfield, Jarrod R. McClean, Matthew J. McEwen, Seneca Meeks, Anthony Edward Megrant, Xiao Mi, Kevin C. Miao, Amanda Mieszala, Zlatko K. Minev, Reza Molavi, Sebastian Molina, Shirin Montazeri, Alexis Morvan, Ramis Movassagh, Wojciech Mruczkiewicz, Ofer Naaman, Matthew Neeley, Charles J. Neill, Ani Nersisyan, Hartmut Neven, Michael Newman, Jiun How Ng, Anthony Nguyen, Murray L. Nguyen, Chia-Hung Ni, Murphy Yuezhen Niu, Logan Oas, Thomas E. O’Brien, William D. Oliver, Alexander Opremcak, Kristofer Ottosson, Andre Petukhov, Alex Pizzuto, John C. Platt, Rebecca Potter, Orion Pritchard, Leonid P. Pryadko, Chris Quintana, Ganesh Ramachandran, Chandrasekhar Ramanathan, Matthew J. Reagor, John Redding, David M. Rhodes, Gabrielle Roberts, Eliott Rosenberg, Emma Rosenfeld, Pedram Roushan, Nicholas C. Rubin, Negar Saei, Daniel Thomas Sank, Kannan Sankaragomathi, Kevin J. Satzinger, Alexander Schmidhuber, Henry F. Schurkus, C. Schuster, Thomas Schuster, Michael J. Shearn, Aaron Shorter, Noah Shutty, Vladimir Shvarts, Volodymyr Sivak, Jindra Skruzny, Spencer Small, Vadim Smelyanskiy, W. Clarke Smith, Rolando D. Somma, Sofia Springer, George Sterling, Doug Strain, Jordan Suchard, Philippe Suchsland, Aaron Szasz, Alexander Sztein, Douglas Thor, Eifu Tomita, Alfredo Torres, Mustafa Mert Torunbalci, Abeer Vaishnav, Justin Vargas, Sergey N. Vdovichev, G. Vidal, Benjamin Villalonga, Catherine Vollgraf Heidweiller, Steven Waltman, Shannon Wang, Brayden Ware, Kate Weber, Travis Weidel, Tom Westerhout, Theodore White, Kristi Wong, Bryan W. K. Woo, Chen Xing, Z. Jamie Yao, Ping Yeh, B. Ying, Juhwan Yoo, Noureldin Yosri, Grayson Young, Adam Zalcman, Chongwe Zhang, Yaxing Zhang, N. Zhu, and Nicholas Zobrist. Observation of constructive interference at the edge of quantum ergodicity. Nature, 646:825 – 830, 2025.

[24] Wei-You Liao, Ge Yan, Yujin Song, Tian-Ci Tian, Wei-Ming Zhu, De-Tao Jiang, Yuxuan Du, and He-Liang Huang. Sample-eficient quantum error mitigation via classical learning surrogates. Communications Physics, 2026.

[25] Xinbiao Wang, Yuxuan Du, Zhuozhuo Tu, Yong Luo, Xiao Yuan, and Dacheng Tao. Transition role of entangled data in quantum machine learning. Nature Communications, 15(1):3716, 2024.

[26] Yuxuan Du, Min-Hsiu Hsieh, and Dacheng Tao. Eficient learning for linear properties of bounded-gate quantum circuits. Nature Communications, 16(1):3790, 2025.

[27] Hong-Ye Hu, Andi Gu, Swarnadeep Majumder, Hang Ren, Yipei Zhang, Derek S Wang, Yi-Zhuang You, Zlatko Minev, Susanne F Yelin, and Alireza Seif. Demonstration of robust and eficient quantum property learning with shallow shadows. Nature Communications, 16(1):2943, 2025.

[28] Ya-Dong Wu, Yan Zhu, Yuexuan Wang, and Giulio Chiribella. Learning quantum properties from short-range correlations using multi-task networks. Nature Communications, 15(1):8796, 2024.

[29] Wei-You Liao, Yuxuan Du, Xinbiao Wang, Tian-Ci Tian, Yong Luo, Bo Du, Dacheng Tao, and He-Liang Huang. Demon stration of eficient predictive surrogates for large-scale quantum processors. Nature Communications, 17(1):4731, 2026.

[30] Yuxuan Du, Yibo Yang, Tongliang Liu, Zhouchen Lin, David A Clifton, Bernard Ghanem, and Dacheng Tao. Shadownet for data-centric quantum system learning. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2026.

[31] Chris Lu, Cong Lu, Robert Tjarko Lange, Yutaro Yamada, Shengran Hu, Jakob Foerster, David Ha, and Jef Clune.

Towards end-to-end automation of ai research. Nature, 651(8107):914–919, 2026.

[32] Juraj Gottweis, Wei-Hung Weng, Alexander Daryin, Tao Tu, Petar Sirkovic, Artiom Myaskovsky, Grzegorz Glowaty, Felix Weissenberger, Alessio Orlandi, Dan Popovici, et al. Accelerating scientific discovery with co-scientist. Nature, pages 1–3, 2026.

[33] Ali Essam Ghareeb, Benjamin Chang, Ludovico Mitchener, Angela Yiu, Caralyn J Szostkiewicz, Dmytro Shved, Gavin J Gyimesi, Jon M Laurent, Samantha M Wright, Muhammed T Razzak, et al. A multi-agent system for automating scientific discovery. Nature, pages 1–3, 2026.

[34] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.

[35] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020.

[36] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.

[37] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, et al. Deepseek-r1 incentivizes reasoning in llms through reinforcement learning. Nature, 645(8081):633–638, 2025.

[38] Bernardino Romera-Paredes, Mohammadamin Barekatain, Alexander Novikov, Matej Balog, M Pawan Kumar, Emilien Dupont, Francisco JR Ruiz, Jordan S Ellenberg, Pengming Wang, Omar Fawzi, et al. Mathematical discoveries from program search with large language models. Nature, 625(7995):468–475, 2024.

[39] Trieu H Trinh, Yuhuai Wu, Quoc V Le, He He, and Thang Luong. Solving olympiad geometry without human demonstrations. Nature, 625(7995):476–482, 2024.

[40] Thomas Hubert, Rishi Mehta, Laurent Sartran, Mikl´os Z Horv´ath, Goran Zuˇzi´c, Eric Wieser, Aja Huang, Julian Schrit-<sup>ˇ</sup> twieser, Yannick Schroecker, Hussain Masoom, et al. Olympiad-level formal mathematical reasoning with reinforcement learning. Nature, 651(8106):607–613, 2026.

[41] Alexander Novikov, Ngˆan V˜u, Marvin Eisenberger, Emilien Dupont, Po-Sen Huang, Adam Zsolt Wagner, Sergey Shirobokov, Borislav Kozlovskii, Francisco JR Ruiz, Abbas Mehrabian, et al. Alphaevolve: A coding agent for scientific and algorithmic discovery. arXiv preprint arXiv:2506.13131, 2025.

[42] Carlos E Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, and Karthik Narasimhan. Swebench: Can language models resolve real-world github issues? In International Conference on Learning Representations, volume 2024, pages 54107–54157, 2024.

[43] John Yang, Carlos Jimenez, Alexander Wettig, Kilian Lieret, Shunyu Yao, Karthik Narasimhan, and Ofir Press. Sweagent: Agent-computer interfaces enable automated software engineering. Advances in Neural Information Processing Systems, 37:50528–50652, 2024.

[44] Xingyao Wang, Boxuan Li, Yufan Song, Frank F Xu, Xiangru Tang, Mingchen Zhuge, Jiayi Pan, Yueqi Song, Bowen Li, Jaskirat Singh, et al. Openhands: An open platform for ai software developers as generalist agents. In International Conference on Learning Representations, volume 2025, pages 65882–65919, 2025.

[45] Shanghua Gao, Ada Fang, Yepeng Huang, Valentina Giunchiglia, Ayush Noori, Jonathan Richard Schwarz, Yasha Ektefaie, Jovana Kondic, and Marinka Zitnik. Empowering biomedical discovery with ai agents. Cell, 187(22):6125–6151, 2024.

[46] Kyle Swanson, Wesley Wu, Nash L Bulaong, John E Pak, and James Zou. The virtual lab of ai agents designs new sars-cov-2 nanobodies. Nature, 646(8085):716–723, 2025.

[47] Kexin Huang, Serena Zhang, Hanchen Wang, Yuanhao Qu, Yingzhou Lu, Ryan Li, Yusuf Roohani, Lin Qiu, Shiyi Cao, Gavin Li, et al. Autonomous biomedical research with an artificial intelligence agent. Science, page eadz4351, 2026.

[48] Daniil A Boiko, Robert MacKnight, Ben Kline, and Gabe Gomes. Autonomous chemical research with large language models. Nature, 624(7992):570–578, 2023.

[49] Andres M. Bran, Sam Cox, Oliver Schilter, Carlo Baldassari, Andrew D White, and Philippe Schwaller. Augmenting large language models with chemistry tools. Nature machine intelligence, 6(5):525–535, 2024.

[50] Yixiang Ruan, Chenyin Lu, Ning Xu, Yuchen He, Yixin Chen, Jian Zhang, Jun Xuan, Jianzhang Pan, Qun Fang, Hanyu Gao, et al. An automatic end-to-end chemical synthesis development platform powered by large language models. Nature communications, 15(1):10160, 2024.

[51] Ziyang Huang, Yi Cao, Ali K Shargh, Jing Luo, Ruidong Mei, Mohd Zaki, Zhan Liu, Wyatt Bunstine, William Jurayj, Somdatta Goswami, et al. Can coding agents reproduce findings in computational materials science? arXiv preprint arXiv:2605.00803, 2026.

[52] Juan Cruz-Benito, Andrew W Cross, David Kremer, and Ismael Faro. Evolutionary discovery of bivariate bicycle codes with llm-guided search. arXiv preprint arXiv:2606.02418, 2026.

[53] Zidu Liu and Florian Marquardt. Large-language-model discovery of quantum ldpc codes through structured concept evolution. arXiv preprint arXiv:2606.24808, 2026.

[54] Ge Yan, Shanchuan Li, Pengyue Ma, Qixin Zhang, Pingchuan Ma, Jianping Wang, Min-Hsiu Hsieh, and Yuxuan Du. Omniqec: discovering practical quantum error-correcting codes by an ai scientist. arXiv preprint arXiv:2607.25865, 2026.

[55] S¨oren Arlt, Haonan Duan, Felix Li, Sang Michael Xie, Yuhuai Wu, and Mario Krenn. Meta-designing quantum experiments with language models. Nature Machine Intelligence, 8(2):148–157, 2026.

[56] Marco Knipfer, Alexander Roman, Konstantin T Matchev, Katia Matcheva, and Sergei Gleyzer. Ai agents for variational

quantum circuit design. arXiv preprint arXiv:2602.19387, 2026.

[57] Kenya Sakka, Wataru Mizukami, and Kosuke Mitarai. An llm system for autonomous variational quantum circuit design. arXiv preprint arXiv:2606.13380, 2026.

[58] Yunheng Zou, Austin H Cheng, Abdulrahman Aldossary, Jiaru Bai, Shi Xuan Leong, Jorge Arturo Campos-Gonzalez-Angulo, Changhyeok Choi, Cher Tian Ser, Gary Tom, Andrew Wang, et al. El agente: An autonomous agent for quantum chemistry. Matter, 8(7), 2025.

[59] Zijian Zhang, Aiwei Yin, Amaan Baweja, Jiaru Bai, Ignacio Gustin, Varinia Bernales, and Al´an Aspuru-Guzik. El agente forjador: Task-driven agent generation for quantum simulation. arXiv preprint arXiv:2604.14609, 2026.

[60] Weitang Li, Jiajun Ren, Lixue Cheng, and Cunxi Gong. Autonomous quantum simulation through large language mode agents. arXiv preprint arXiv:2601.10194, 2026.

[61] Fabian Finger, Frederic Rapp, Pranav Kalidindi, Kerry He, Kante Yin, Alexander Koziell-Pipe, David Zsolt Manrique, Gabriel Greene-Diniz, Stephen Clark, Hamza Fawzi, et al. Automated near-term quantum algorithm discovery for molecular ground states. arXiv preprint arXiv:2603.26359, 2026.

[62] Jiacheng Miao, Joe R Davis, Yaohui Zhang, Jonathan K Pritchard, and James Zou. Reimagining research papers as interactive and reliable ai agents. Nature, pages 1–9, 2026.

[63] Haoran Liao, Derek S Wang, Iskandar Sitdikov, Ciro Salcedo, Alireza Seif, and Zlatko K Minev. Machine learning for practical quantum error mitigation. Nature Machine Intelligence, 6(12):1478–1486, 2024.

[64] Zhenyu Cai, Ryan Babbush, Simon C Benjamin, Suguru Endo, William J Huggins, Ying Li, Jarrod R McClean, and Thomas E O’Brien. Quantum error mitigation. Reviews of Modern Physics, 95(4):045005, 2023.

[65] Joschka Rofe. Quantum error correction: an introductory guide. Contemporary Physics, 60(3):226–245, 2019.

[66] Yanming Che, Clemens Gneiting, Xiaoguang Wang, and Franco Nori. Quantum circuit complexity and unsupervised machine learning of topological order. Nature Communications, 2026.

[67] Peter J Rousseeuw. Silhouettes: a graphical aid to the interpretation and validation of cluster analysis. Journal of computational and applied mathematics, 20:53–65, 1987.

[68] Supanut Thanasilp, Samson Wang, Marco Cerezo, and Zo¨e Holmes. Exponential concentration in quantum kernel methods. Nature communications, 15(1):5200, 2024.

[69] Hsin-Yuan Huang, Michael Broughton, Masoud Mohseni, Ryan Babbush, Sergio Boixo, Hartmut Neven, and Jarrod R McClean. Power of data in quantum machine learning. Nature communications, 12(1):2631, 2021.

[70] OpenAI. GPT-5.6 system card, July 2026.

[71] Anthropic. System card: Claude opus 5.5, September 2026.

[72] Anyi Xu, Bangcai Lin, Bing Xue, Bingxuan Wang, Bingzheng Xu, Bochao Wu, Bowei Zhang, Chaofan Lin, Chen Dong, Chenchen Ling, et al. Deepseek-v4: Towards highly eficient million-token context intelligence. arXiv preprint arXiv:2606.19348, 2026.

[73] Kimi Team, Tongtong Bai, Yifan Bai, Yiping Bao, Jianfeng Cai, Xinyuan Cai, Peizhou Cao, Yuxuan Cao, Ziwei Chai, Y Charles, et al. Kimi k3: Open frontier intelligence. arXiv preprint arXiv:2607.24653, 2026.

[74] xAI. Model card: Grok 4.6, August 2026.

[75] InternAgent Team, Bo Zhang, Shiyang Feng, Xiangchao Yan, Jiakang Yuan, Runmin Ma, Yusong Hu, Zhiyin Yu, Xiaohan He, Songtao Huang, et al. Internagent: When agent becomes the scientist–building closed-loop system from hypothesis to verification. arXiv preprint arXiv:2505.16938, 2025.

[76] Zongwei Li, Zhonghang Li, Zirui Guo, Xubin Ren, and Chao Huang. Deepcode: Open agentic coding. arXiv preprint arXiv:2512.07921, 2025.

[77] SWE-agent Team. mini-swe-agent: The 100 line ai agent that solves github issues, 2024.

[78] Chunqiu Steven Xia, Zhe Wang, Yan Yang, Yuxiang Wei, and Lingming Zhang. Live-swe-agent: Can software engineering agents self-evolve on the fly? arXiv preprint arXiv:2511.13646, 2025.

[79] Wenyi Wang, Piotr Piakos, Li Nanbo, Firas Laakom, Yimeng Chen, Mateusz Ostaszewski, Mingchen Zhuge, and J¨urgen Schmidhuber. Huxley-g\” odel machine: Human-level coding agent development by an approximation of the optimal self-improving machine. In International Conference on Learning Representations, volume 2026, pages 80356–80386, 2026.

[80] Yoonho Lee, Roshen Nair, Qizheng Zhang, Kangwook Lee, Omar Khattab, and Chelsea Finn. Meta-harness: End-to-end optimization of model harnesses. arXiv preprint arXiv:2603.28052, 2026.

[81] Xingyu Chen, Jiahao Xu, Tian Liang, Zhiwei He, Jianhui Pang, Dian Yu, Linfeng Song, Qiuzhi Liu, Mengfei Zhou, Zhuosheng Zhang, et al. Do not think that much for 2+ 3=? on the overthinking of long reasoning models. In Forty second International Conference on Machine Learning, 2025.

[82] Yilun Yao, Xinyu Tan, Chao-Hsuan Liu, Yaoming Li, Zhengyang Wang, Wenhan Yu, Zhewen Tan, Yuxuan Tian, Guangxiang Zhao, Lin Sun, et al. Harness-bench: Measuring harness efects across models in realistic agent workflows. arXiv preprint arXiv:2605.27922, 2026.

[83] Haichuan Hu, Quanjun Zhang, Shengcheng Yu, Zhifei Chen, Tianyu Luo, Chunrong Fang, Zhenyu Chen, and Liang Xiao. Beyond the model: Demystifying harness efects in software engineering agents. arXiv preprint arXiv:2609.32459, 2026.

[84] Gyeongwon J Kim, Alex Wilf, Louis-Philippe Morency, and Daniel Fried. From reproduction to replication: Evaluating research agents with progressive code masking. In International Conference on Learning Representations, volume 2026, pages 123997–124024, 2026.

[85] Meysam Alizadeh, Mohsen Mosleh, Fabrizio Gilardi, Atoosa Kasirzadeh, and Joshua Tucker. Ai coding agents can reproduce social science findings. arXiv preprint arXiv:2606.11447, 2026.

[86] Charlie Victor Snell, Jaehoon Lee, Kelvin Xu, and Aviral Kumar. Scaling LLM test-time compute optimally can be more

efective than scaling parameters for reasoning. In The Thirteenth International Conference on Learning Representations, 2025.

[87] Wenkai Yang, Shuming Ma, Yankai Lin, and Furu Wei. Towards thinking-optimal scaling of test-time compute for LLM reasoning. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025.

[88] Keyan Ding, Jing Yu, Junjie Huang, Yuchen Yang, Qiang Zhang, and Huajun Chen. Scitoolagent: a knowledge-graph driven scientific agent for multitool integration. Nature Computational Science, 5(10):962–972, 2025.

[89] Isaac Ong, Amjad Almahairi, Vincent Wu, Wei-Lin Chiang, Tianhao Wu, Joseph E. Gonzalez, M Waleed Kadous, and Ion Stoica. RouteLLM: Learning to route LLMs from preference data. In The Thirteenth International Conference on Learning Representations, 2025.

[90] Andi Gu, Alek H Kemeny, Susanne F Yelin, and Mikhail D Lukin. Discovering state-of-the-art quantum circuits with large language models. arXiv preprint arXiv:2610.22367, 2026.

[91] Tanuj Khattar, Nicholas C Rubin, Craig Gidney, Michael Broughton, Guang Hao Low, and Ryan Babbush. Leveraging verifiable structure in quantum circuits to find more eficient compilations with agentic ai. arXiv preprint arXiv:2610.23832, 2026.

[92] Marcos Crichigno and Steve Flammia. Automating quantum circuit optimization with ai agents. arXiv preprint arXiv:2610.31352, 2026.

[93] Steven R White. Density matrix formulation for quantum renormalization groups. Physical review letters, 69(19):2863, 1992.

[94] Guifr´e Vidal. Eficient simulation of one-dimensional quantum many-body systems. Physical review letters, 93(4):040502, 2004.

[95] Ulrich Schollw¨ock. The density-matrix renormalization group in the age of matrix product states. Annals of physics, 326(1):96–192, 2011.

[96] Giuseppe Carleo and Matthias Troyer. Solving the quantum many-body problem with artificial neural networks. Science, 355(6325):602–606, 2017.

[97] Markus Schmitt and Markus Heyl. Quantum many-body dynamics in two dimensions with artificial neural networks. Physical Review Letters, 125(10):100503, 2020.

[98] Andrey A Bagrov, Askar A Iliasov, and Tom Westerhout. Kinetic samplers for neural quantum states. Physical Review B, 104(10):104407, 2021.

[99] Alessandro Sinibaldi, Clemens Giuliani, Giuseppe Carleo, and Filippo Vicentini. Unbiasing time-dependent variationa monte carlo by projected quantum evolution. Quantum, 7:1131, 2023.

[100] Patrick Rall, Daniel Liang, Jeremy Cook, and William Kretschmer. Simulation of qubit quantum circuits via pauli propagation. Physical Review A, 99(6):062337, 2019.

[101] Tomislav Beguˇsi´c, Johnnie Gray, and Garnet Kin-Lic Chan. Fast and converged classical simulations of evidence for the utility of quantum computing before fault tolerance. Science Advances, 10(3):eadk4321, 2024.

[102] Armando Angrisani, Alexander Schmidhuber, Manuel S Rudolph, Marco Cerezo, Zo¨e Holmes, and Hsin-Yuan Huang. Classically estimating observables of noiseless quantum circuits. Physical review letters, 135(17):170602, 2025.

[103] Richard Blankenbecler, DJ Scalapino, and RL Sugar. Monte carlo calculations of coupled boson-fermion systems. i. Physical Review D, 24(8):2278, 1981.

[104] David M Ceperley. Path integrals in the theory of condensed helium. Reviews of Modern Physics, 67(2):279, 1995.

[105] Matthias Troyer and Uwe-Jens Wiese. Computational complexity and fundamental limitations to fermionic quantum monte carlo simulations. Physical review letters, 94(17):170201, 2005.

[106] Guy Cohen, Emanuel Gull, David R Reichman, and Andrew J Millis. Taming the dynamical sign problem in real-time evolution of quantum many-body problems. Physical review letters, 115(26):266802, 2015.

[107] Goran Lindblad. On the generators of quantum dynamical semigroups. Communications in mathematical physics, 48(2):119–130, 1976.

[108] Jean Dalibard, Yvan Castin, and Klaus Mølmer. Wave-function approach to dissipative processes in quantum optics. Physical review letters, 68(5):580, 1992.

[109] Klaus Mølmer, Yvan Castin, and Jean Dalibard. Monte carlo wave-function method in quantum optics. Journal of the Optical Society of America B, 10(3):524–538, 1993.

[110] Barbara M Terhal and David P DiVincenzo. Classical simulation of noninteracting-fermion quantum circuits. Physical Review A, 65(3):032325, 2002.

[111] Leslie G Valiant. Quantum circuits that can be simulated classically in polynomial time. SIAM Journal on Computing, 31(4):1229–1254, 2002.

[112] Gian-Carlo Wick. The evaluation of the collision matrix. Physical review, 80(2):268, 1950.

[113] Benedikt Fauseweh. Quantum many-body simulations on digital quantum computers: State-of-the-art and future challenges. Nature Communications, 15(1):2123, 2024.

[114] Andrew M Childs, Yuan Su, Minh C Tran, Nathan Wiebe, and Shuchen Zhu. Theory of trotter error with commutator scaling. Physical Review X, 11(1):011020, 2021.

[115] Masuo Suzuki. General theory of higher-order decomposition of exponential operators and symplectic integrators. Physics Letters A, 165(5-6):387–395, 1992.

[116] Jan Lukas Bosse, Andrew M Childs, Charles Derby, Filippo Maria Gambetta, Ashley Montanaro, and Raul A Santos. Efficient and practical hamiltonian simulation from time-dependent product formulas. Nature Communications, 16(1):2673, 2025.

[117] Mohsen Bagherimehrab, Luis Mantilla Calder´on, Dominic W Berry, Philipp Schleich, Mohammad Ghazi Vakili, Abdul-

rahman Aldossary, Jorge A Campos Gonzalez Angulo, Christoph Gorgulla, and Al´an Aspuru-Guzik. Faster algorithmic quantum and classical simulations by corrected product formulas. PRX Quantum, 7(3):033018, 2026.

[118] David Poulin, Angie Qarry, Rolando Somma, and Frank Verstraete. Quantum simulation of time-dependent hamiltonians and the convenient illusion of hilbert space. Physical review letters, 106(17):170501, 2011.

[119] Yangyu Lu, Yifei Huang, Dong An, Qi Zhao, Dingshun Lv, and Xiao Yuan. Digital adiabatic evolution is universally accurate. Nature Communications, 2026.

[120] Tameem Albash and Daniel A Lidar. Adiabatic quantum computation. Reviews of Modern Physics, 90(1):015002, 2018.

[121] Mario Motta, Chong Sun, Adrian TK Tan, Matthew J O’Rourke, Erika Ye, Austin J Minnich, Fernando GSL Brandao, and Garnet Kin-Lic Chan. Determining eigenstates and thermal states on a quantum computer using quantum imaginary time evolution. Nature Physics, 16(2):205–210, 2020.

[122] Erenay Karacan, Yanbin Chen, and Christian B Mendl. Enhancing scalability of quantum eigenvalue transformation of unitary matrices for ground state preparation through adaptive finer filtering. Quantum, 9:1624, 2025.

[123] Richard Cleve, Artur Ekert, Chiara Macchiavello, and Michele Mosca. Quantum algorithms revisited. Proceedings: Mathematical, Physical and Engineering Sciences, pages 339–354, 1998.

[124] Miroslav Dobˇs´ıˇcek, G¨oran Johansson, Vitaly Shumeiko, and G¨oran Wendin. Arbitrary accuracy iterative quantum phase estimation algorithm using a single ancillary qubit: A two-qubit benchmark. Physical Review A—Atomic, Molecular, and Optical Physics, 76(3):030306, 2007.

[125] Nathan Wiebe and Chris Granade. Eficient bayesian phase estimation. Physical review letters, 117(1):010503, 2016.

[126] Hongkang Ni, Haoya Li, and Lexing Ying. On low-depth algorithms for quantum phase estimation. Quantum, 7:1165, 2023.

[127] Gilles Brassard, Peter Hoyer, Michele Mosca, and Alain Tapp. Quantum amplitude amplification and estimation. arXiv preprint quant-ph/0005055, 2000.

[128] Tudor Giurgica-Tiron, Iordanis Kerenidis, Farrokh Labib, Anupam Prakash, and William Zeng. Low depth algorithms for quantum amplitude estimation. Quantum, 6:745, 2022.

[129] Alexandra Ramˆoa and Luis Paulo Santos. Bayesian quantum amplitude estimation. Quantum, 9:1856, 2025.

[130] Marco Cerezo, Andrew Arrasmith, Ryan Babbush, Simon C Benjamin, Suguru Endo, Keisuke Fujii, Jarrod R McClean, Kosuke Mitarai, Xiao Yuan, Lukasz Cincio, et al. Variational quantum algorithms. Nature Reviews Physics, 3(9):625–644, 2021.

[131] Alberto Peruzzo, Jarrod McClean, Peter Shadbolt, Man-Hong Yung, Xiao-Qi Zhou, Peter J Love, Al´an Aspuru-Guzik, and Jeremy L O’brien. A variational eigenvalue solver on a photonic quantum processor. Nature communications, 5(1):4213, 2014.

[132] Stefano Barison, Filippo Vicentini, and Giuseppe Carleo. An eficient quantum algorithm for the time evolution of parameterized circuits. Quantum, 5:512, 2021.

[133] Maria Schuld, Ville Bergholm, Christian Gogolin, Josh Izaac, and Nathan Killoran. Evaluating analytic gradients on quantum hardware. Physical Review A, 99(3):032331, 2019.

[134] Jarrod R McClean, Jonathan Romero, Ryan Babbush, and Al´an Aspuru-Guzik. The theory of variational hybrid quantumclassical algorithms. New Journal of Physics, 18(2):023023, 2016.

[135] Harper R Grimsley, Sophia E Economou, Edwin Barnes, and Nicholas J Mayhall. An adaptive variational algorithm for exact molecular simulations on a quantum computer. Nature communications, 10(1):3007, 2019.

[136] Manuel S Rudolph, Jacob Miller, Danial Motlagh, Jing Chen, Atithi Acharya, and Alejandro Perdomo-Ortiz. Synergistic pretraining of parametrized quantum circuits via tensor networks. Nature Communications, 14(1):8367, 2023.

[137] Edward Farhi, Jefrey Goldstone, and Sam Gutmann. A quantum approximate optimization algorithm. arXiv preprint arXiv:1411.4028, 2014.

[138] Edward Farhi, Jefrey Goldstone, Sam Gutmann, and Leo Zhou. The quantum approximate optimization algorithm and the sherrington-kirkpatrick model at infinite size. Quantum, 6:759, 2022.

[139] Gian Gentinetta, Friederike Metz, and Giuseppe Carleo. Overhead-constrained circuit knitting for variational quantum dynamics. Quantum, 8:1296, 2024.

[140] Cristina Cirstoiu, Zoe Holmes, Joseph Iosue, Lukasz Cincio, Patrick J Coles, and Andrew Sornborger. Variational fast forwarding for quantum simulation beyond the coherence time. npj Quantum Information, 6(1):82, 2020.

[141] Kaoru Mizuta, Yuya O Nakagawa, Kosuke Mitarai, and Keisuke Fujii. Local variational quantum compilation of large-scale hamiltonian dynamics. PRX Quantum, 3(4):040302, 2022.

[142] Yuxuan Du, Xinbiao Wang, Naixu Guo, Zhan Yu, Yang Qian, Kaining Zhang, Min-Hsiu Hsieh, Patrick Rebentrost, and Dacheng Tao. A Gentle Introduction to Quantum Machine Learning. Springer Nature Singapore, Singapore, 2025.

[143] Maria Schuld, Alex Bocharov, Krysta M Svore, and Nathan Wiebe. Circuit-centric quantum classifiers. Physical Review A, 101(3):032308, 2020.

[144] Maria Schuld and Nathan Killoran. Quantum machine learning in feature hilbert spaces. Physical review letters, 122(4):040504, 2019.

[145] Seth Lloyd and Christian Weedbrook. Quantum generative adversarial learning. Physical review letters, 121(4):040502, 2018.

[146] Jinkai Tian, Xiaoyu Sun, Yuxuan Du, Shanshan Zhao, Qing Liu, Kaining Zhang, Wei Yi, Wanrong Huang, Chaoyue Wang, Xingyao Wu, et al. Recent advances for quantum neural networks in generative learning. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(10):12321–12340, 2023.

[147] Samuel Yen-Chi Chen, Chao-Han Huck Yang, Jun Qi, Pin-Yu Chen, Xiaoli Ma, and Hsi-Sheng Goan. Variational quantum circuits for deep reinforcement learning. IEEE access, 8:141007–141024, 2020.

[148] Yuxuan Du, Zhuozhuo Tu, Xiao Yuan, and Dacheng Tao. Eficient measure for the expressivity of variational quantum algorithms. Physical Review Letters, 128(8):080506, 2022.

[149] Jarrod R McClean, Sergio Boixo, Vadim N Smelyanskiy, Ryan Babbush, and Hartmut Neven. Barren plateaus in quantum neural network training landscapes. Nature communications, 9(1):4812, 2018.

[150] Yuxuan Du, Yibo Yang, Dacheng Tao, and Min-Hsiu Hsieh. Problem-dependent power of quantum neural networks on multiclass classification. Physical Review Letters, 131:140601, 2023.

[151] Vojtˇech Havl´ıˇcek, Antonio D C´orcoles, Kristan Temme, Aram W Harrow, Abhinav Kandala, Jerry M Chow, and Jay M Gambetta. Supervised learning with quantum-enhanced feature spaces. Nature, 567(7747):209–212, 2019.

[152] Craig Saunders, Alexander Gammerman, and Vladimir Vovk. Ridge regression learning algorithm in dual variables. In International Conference on Machine Learning, 1998.

[153] Thomas Hubregtsen, David Wierichs, Elies Gil-Fuster, Peter-Jan HS Derks, Paul K Faehrmann, and Johannes Jakob Meyer. Training quantum embedding kernels on near-term quantum computers. Physical Review A, 106(4):042431, 2022.

[154] Cong Lei, Yuxuan Du, Peng Mi, Jun Yu, and Tongliang Liu. Neural auto-designer for enhanced quantum kernels. In International Conference on Learning Representations, volume 2024, pages 17817–17841, 2024.

[155] Jennifer R Glick, Tanvi P Gujarati, Antonio D Corcoles, Youngseok Kim, Abhinav Kandala, Jay M Gambetta, and Kristan Temme. Covariant quantum kernels for data with group structure. Nature Physics, 20(3):479–483, 2024.

[156] David Layden, Guglielmo Mazzola, Ryan V Mishmash, Mario Motta, Pawel Wocjan, Jin-Sung Kim, and Sarah Sheldon. Quantum-enhanced markov chain monte carlo. Nature, 619(7969):282–287, 2023.

[157] Yuichiro Nakano, Hideaki Hakoshima, Kosuke Mitarai, and Keisuke Fujii. Markov-chain monte carlo method enhanced by a quantum alternating operator ansatz. Physical Review Research, 6(3):033105, 2024.

[158] Wen Ting Hsieh, Alev Orfi, and Dries Sels. Adiabatic dressing of quantum enhanced markov chains. arXiv preprint arXiv:2603.28076, 2026.

[159] Stuart Ferguson and Petros Wallden. Quantum-enhanced markov chain monte carlo for systems larger than a quantum computer. Physical Review Research, 7(1):013231, 2025.

[160] Kate V Marshall, Daniel J Egger, Michael Garn, Francesca Schiavello, Sebastian Brandhofer, Christa Zoufal, and Stefan Woerner. Quantum-enhanced markov chain monte carlo for combinatorial optimization. arXiv preprint arXiv:2602.06171, 2026.

[161] Manas Sajjan, Vinit Singh, and Sabre Kais. Polynomially eficient quantum enabled variational monte carlo for training neural-network quantum states for physico-chemical applications. npj Quantum Information, 2026.

[162] Hsin-Yuan Huang, Richard Kueng, and John Preskill. Predicting many properties of a quantum system from very few measurements. Nature Physics, 16:1050–1057, 2020.

[163] Hsin-Yuan Huang, Richard Kueng, and John Preskill. Eficient Estimation of Pauli Observables by Derandomization. Physical Review Letters, 127:030503, 2021.

[164] Alexander Gresch and Martin Kliesch. Guaranteed eficient energy estimation of quantum many-body Hamiltonians using ShadowGrouping. Nature Communications, 16:689, 2025.

[165] Xinbiao Wang, Yuxuan Du, and Dacheng Tao. Learnable yet not simulable: a quantum resource theory of learning models. arXiv preprint arXiv:2608.02325, 2026.

[166] Yusheng Zhao, Chi Zhang, and Yuxuan Du. Rethink the role of deep learning towards large-scale quantum systems. In Forty-second International Conference on Machine Learning, 2025.

[167] Xiaoting Gao, Mingsheng Tian, Feng-Xiao Sun, Ya-Dong Wu, Yu Xiang, and Qiongyi He. Classifying multipartite continuous-variable entanglement structures through data-augmented neural networks. Nature Machine Intelligence, pages 1–10, 2026.

[168] A. Elben, B. Vermersch, C. F. Roos, and P. Zoller. Statistical correlations between locally randomized measurements: A toolbox for probing entanglement in many-body quantum states. Physical Review A, 99(5), May 2019.

[169] Tif Brydges, Andreas Elben, Petar Jurcevic, Benoˆıt Vermersch, Christine Maier, Ben P. Lanyon, Peter Zoller, Rainer Blatt, and Christian F. Roos. Probing r´enyi entanglement entropy via randomized measurements. Science, 364(6437):260–263, April 2019.

[170] Qing Liu, Zihao Li, Xiao Yuan, Huangjun Zhu, and You Zhou. Auxiliary-Free Replica Shadows: Eficient Estimation of Multiple Nonlinear Quantum Properties. Physical Review Letters, 136:100602, 2026.

[171] Jianwei Wang, Stefano Paesani, Rafaele Santagati, Sebastian Knauer, Antonio A Gentile, Nathan Wiebe, Maurangelo Petruzzella, Jeremy L O’brien, John G Rarity, Anthony Laing, et al. Experimental quantum hamiltonian learning. Nature Physics, 13(6):551–555, 2017.

[172] Andi Gu, Lukasz Cincio, and Patrick J. Coles. Practical Hamiltonian learning with unitary dynamics and Gibbs states. Nature Communications, 15:312, 2024.

[173] Tudor Giurgica-Tiron, Yousef Hindy, Ryan LaRose, Andrea Mari, and William J Zeng. Digital zero noise extrapolation for quantum error mitigation. In 2020 IEEE international conference on quantum computing and engineering (QCE), pages 306–316. IEEE, 2020.

[174] Kristan Temme, Sergey Bravyi, and Jay M Gambetta. Error mitigation for short-depth quantum circuits. Physical review letters, 119(18):180509, 2017.

[175] X. Bonet-Monroig, R. Sagastizabal, M. Singh, and T. E. O’Brien. Low-cost error mitigation by symmetry verification. Physical Review A, 98(6), December 2018.

[176] William J. Huggins, Sam McArdle, Thomas E. O’Brien, Joonho Lee, Nicholas C. Rubin, Sergio Boixo, K. Birgitta Whaley, Ryan Babbush, and Jarrod R. McClean. Virtual distillation for quantum error mitigation. Physical Review X, 11(4), November 2021.

[177] Giacomo Torlai, Christopher J. Wood, Atithi Acharya, Giuseppe Carleo, Juan Carrasquilla, and Leandro Aolita. Quantum process tomography with unsupervised learning and tensor networks. Nature Communications, 14:2858, 2023.

[178] Hsin-Yuan Huang, Sitan Chen, and John Preskill. Learning to Predict Arbitrary Quantum Processes. PRX Quantum, 4:040337, 2023.

[179] Giulio Starace, Oliver Jafe, Dane Sherburn, James Aung, Jun Shern Chan, Leon Maksin, Rachel Dias, Evan Mays, Benjamin Kinsella, Wyatt Thompson, et al. Paperbench: evaluating ai’s ability to replicate ai research. arXiv preprint arXiv:2504.01848, 2025.

[180] Shi Qiu, Junyi Deng, Yiwei Deng, Haoran Dong, Jieyu Fu, Mao Li, Zeyu Li, Zhaolong Zhang, Huiwen Zheng, Leidong Bao, et al. Prbench: End-to-end paper reproduction in physics research. arXiv preprint arXiv:2603.27646, 2026.

[181] Zachary S. Siegel, Sayash Kapoor, Nitya Nadgir, Benedikt Stroebl, and Arvind Narayanan. CORE-Bench: Foster ing the credibility of published research through a computational reproducibility agent benchmark. arXiv preprint arXiv:2409.11363, 2024.

[182] Shengran Hu, Cong Lu, and Jef Clune. Automated design of agentic systems. International Conference on Learning Representations, 2025.

[183] Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. Reflexion: Language agents with verbal reinforcement learning. Advances in Neural Information Processing Systems, 2023.

[184] Andrew Zhao, Daniel Huang, Quentin Xu, Matthieu Lin, Yong-Jin Liu, and Gao Huang. ExpeL: LLM agents are experiential learners. AAAI Conference on Artificial Intelligence, 2024.

[185] Zora Zhiruo Wang, Jiayuan Mao, Daniel Fried, and Graham Neubig. Agent workflow memory. arXiv preprint arXiv:2409.07429, 2024.

[186] Lakshya A. Agrawal, Shangyin Tan, Dilara Soylu, Noah Ziems, Rishi Khare, Krista Opsahl-Ong, Arnav Singhvi, Herumb Shandilya, Michael J. Ryan, Meng Jiang, et al. GEPA: Reflective prompt evolution can outperform reinforcement learning. International Conference on Learning Representations, 2026.

[187] Qing-Hong Cao, Zong-Yue Hou, Ying-Ying Li, Xiaohui Liu, Zhuo-Yang Song, Liang-Qi Zhang, Shutao Zhang, and Ke Zhao. Scalable quantum state preparation via large-language-model-driven discovery. arXiv preprint arXiv:2505.06347, 2025.

[188] C Zhang, RG Cortinas, AH Karamlou, N Noll, J Provazza, J Bausch, S Shirobokov, A White, M Claassen, SH Kang, et al. Quantum computation of molecular geometry via many-body nuclear spin echoes. arXiv preprint arXiv:2510.19550, 2025.

[189] Tingjia Miao, Wenkai Jin, Jinxin Tan, Muhua Zhang, Xianghe Pang, Zexi Liu, Yuwen Du, Tian Jin, Tu Guo, Zhengliang Zhang, et al. Physmaster: Building an autonomous ai physicist for theoretical and computational physics research. arXiv preprint arXiv:2512.19799, 2025.

[190] Maximilian N¨agele and Florian Marquardt. Agentic exploration of physics models. Physical Review X, 16(3):031002, 2026.

[191] Hongjian Zhou, Xianfan Nie, Sean Wu, Tarun Patel, Jinge Wu, Andrew Liu, Adam Wei Tsen, and David A Clifton. The ai theorist reveals excitonic structure in α-rucl . arXiv preprint arXiv:2610.02417, 2026.

[192] Tingjia Miao, Wenkai Jin, Muhua Zhang, Jinxin Tan, Yuelin Hu, Tu Guo, Jiejun Zhang, Yuhan Wang, Wenbo Li, Yinuo Gao, et al. Prl-bench: A comprehensive benchmark evaluating llms’ capabilities in frontier physics research. arXiv preprint arXiv:2604.15411, 2026.

[193] Naixu Guo, Changhao Li, Siyu Cheng, Qicheng Tang, Binzhao Luo, Bikun Li, Yuxuan Du, Shihao Ru, and Jiaqi Cai. Evaluating verified autonomy in quantum engineering. arXiv preprint arXiv:2609.17439, 2026.

[194] Ken Deng, Xiangfei Wang, Guijing Duan, Chen Mo, Junkun Huang, Runqing Zhang, Ling Qian, Zhiguo Huang, Jize Han, and Di Luo. Towards verifiable and self-correcting ai physicists for quantum many-body simulations. arXiv preprint arXiv:2604.00149, 2026.

[195] Sanjay Vishwakarma, Francis Harkins, Siddharth Golecha, Vishal Sharathchandra Bajpe, Nicolas Dupuis, Luca Buratti, David Kremer, Ismael Faro, Ruchir Puri, and Juan Cruz-Benito. Qiskit humaneval: An evaluation benchmark for quantum code generative models. In 2024 IEEE International Conference on Quantum Computing and Engineering (QCE), volume 1, pages 1169–1176. IEEE, 2024.

[196] Xiaoyu Guo, Minggu Wang, and Jianjun Zhao. Quanbench: Benchmarking quantum code generation with large language models. In 2025 40th IEEE/ACM International Conference on Automated Software Engineering (ASE), pages 2657–2669. IEEE, 2025.

[197] Rui Yang, Ziruo Wang, Yuntian Gu, Yitao Liang, and Tongyang Li. Qcircuitbench: A large-scale dataset for benchmarking quantum algorithm design. Advances in Neural Information Processing Systems, 38, 2026.

[198] Ali Slim, Haydar Hamieh, Jawad Kotaich, Yehya Ghosn, Mahdi Chehimi, Ammar Mohanna, Hasan Abed Al Kader Hammoud, and Bernard Ghanem. Quanbench+: A unified multi-framework benchmark for llm-based quantum code generation. arXiv preprint arXiv:2604.08570, 2026.

[199] Jheng-Wei Li and Xavier Waintal. Matrix product states and first quantization. Physical Review Letters, 136(11):116503, 2026.

[200] Hirone Ishida, Natsuki Okada, Shintaro Hoshino, and Hiroshi Shinaoka. Low-rank quantics tensor train representations of feynman diagrams for multiorbital electron-phonon models. Physical Review Letters, 135(4):046502, 2025.

[201] C Mc Keever and MH Szyma´nska. Stable ipepo tensor-network algorithm for dynamics of two-dimensional open quantum lattice models. Physical Review X, 11(2):021035, 2021.

[202] Dominik S Kufel, Jack Kemp, DinhDuy Vu, Simon M Linsel, Chris R Laumann, and Norman Y Yao. Approximately symmetric neural networks for quantum spin liquids. Physical Review Letters, 135(5):056702, 2025.

[203] Di Luo, Zhuo Chen, Juan Carrasquilla, and Bryan K Clark. Autoregressive neural network for simulating open quantum systems via a probabilistic formulation. Physical review letters, 128(9):090501, 2022.

[204] Linda Mauron, Zakari Denis, Jannes Nys, and Giuseppe Carleo. Predicting topological entanglement entropy in a rydberg analogue simulator. Nature physics, 21(8):1332–1337, 2025.

[205] Sergey Bravyi, Dan Browne, Padraic Calpin, Earl Campbell, David Gosset, and Mark Howard. Simulation of quantum circuits by low-rank stabilizer decompositions. Quantum, 3:181, 2019.

[206] Xiangjian Qian, Jiale Huang, and Mingpu Qin. Cliford circuits augmented time-dependent variational principle. Physical Review Letters, 134(15):150404, 2025.

[207] Igor Poboiko, Paul P¨opperl, Igor V Gornyi, and Alexander D Mirlin. Theory of free fermions under random projective measurements. Physical Review X, 13(4):041046, 2023.

[208] Mario Collura, Jacopo De Nardis, Vincenzo Alba, and Guglielmo Lami. The non-stabilizerness of fermionic gaussian states. Quantum, 10:2036, 2026.

[209] Tong Shen and Daniel A Lidar. Real-time sign-problem-suppressed quantum monte carlo algorithm for noisy quantum circuit simulations. Physical Review Letters, 136(23):230601, 2026.

[210] Zhe Wang, Zhiyan Wang, Yi-Ming Ding, Bin-Bin Mao, and Zheng Yan. Bipartite reweight-annealing algorithm of quantum monte carlo to extract large-scale data of entanglement entropy and its derivative. Nature Communications, 16(1):5880, 2025.

[211] Aaron Sander, Maximilian Fr¨ohlich, Martin Eigel, Jens Eisert, Patrick Gelß, Michael Hinterm¨uller, Richard M Milbradt, Robert Wille, and Christian B Mendl. Large-scale stochastic simulation of open quantum systems. Nature Communications, 16(1):11074, 2025.

[212] Dario Tamascelli, Andrea Smirne, James Lim, Susana F Huelga, and Martin B Plenio. Eficient simulation of finitetemperature open quantum systems. Physical review letters, 123(9):090402, 2019.

[213] Adil A Gangat, Te I, and Ying-Jer Kao. Steady states of infinite-size dissipative quantum chains via imaginary time evolution. Physical review letters, 119(1):010501, 2017.

[214] Andrew Pocklington and Aashish A Clerk. Eficient simulation of nontrivial dissipative spin chains via stochastic unraveling. PRX Quantum, 6(3):030349, 2025.

[215] Martin Larocca, Nathan Ju, Diego Garc´ıa-Mart´ın, Patrick J Coles, and Marco Cerezo. Theory of overparametrization in quantum neural networks. Nature Computational Science, 3(6):542–551, 2023.

[216] Marco Cerezo, Akira Sone, Tyler Volkof, Lukasz Cincio, and Patrick J Coles. Cost function dependent barren plateaus in shallow parametrized quantum circuits. Nature communications, 12(1):1791, 2021.

[217] Zo¨e Holmes, Andrew Arrasmith, Bin Yan, Patrick J Coles, Andreas Albrecht, and Andrew T Sornborger. Barren plateaus preclude learning scramblers. Physical Review Letters, 126(19):190501, 2021.

[218] Franz J Schreiber, Jens Eisert, and Johannes Jakob Meyer. Classical surrogates for quantum learning models. Physical Review Letters, 131(10):100803, 2023.

[219] Christian Bertoni, Jonas Haferkamp, Marcel Hinsche, Marios Ioannou, Jens Eisert, and Hakop Pashayan. Shallow Shad ows: Expectation Estimation Using Low-Depth Random Cliford Circuits. Physical Review Letters, 133:020602, 2024.

[220] O. Veltheim and E. Keski-Vakkuri. Optimizing Quantum Measurements by Partitioning Multisets of Observables. Physical Review Letters, 134:030801, 2025.

[221] Minh C. Tran, Daniel K. Mark, Wen Wei Ho, and Soonwon Choi. Measuring Arbitrary Physical Properties in Analog Quantum Simulation. Physical Review X, 13:011049, 2023.

[222] Hsin-Yuan Huang, Richard Kueng, and John Preskill. Information-theoretic bounds on quantum advantage in machine learning. Physical Review Letters, 126(19):190505, 2021.

[223] Laura Lewis, Hsin-Yuan Huang, Viet T. Tran, Sebastian Lehner, Richard Kueng, and John Preskill. Improved machine learning algorithm for predicting ground state properties. Nature Communications, 15:895, 2024.

[224] C. Huerta Alderete, Max Hunter Gordon, Frederic Sauvage, Akira Sone, Andrew T. Sornborger, Patrick J. Coles, and M. Cerezo. Inference-Based Quantum Sensing. Physical Review Letters, 129:190501, 2022.

[225] Aniket Rath, Rick van Bijnen, Andreas Elben, Peter Zoller, and Benoˆıt Vermersch. Importance Sampling of Randomized Measurements for Probing Entanglement. Physical Review Letters, 127:200503, 2021.

[226] Zhenhuan Liu, Yifan Tang, Hao Dai, Pengyu Liu, Shu Chen, and Xiongfeng Ma. Detecting Entanglement in Quantum Many-Body Systems via Permutation Moments. Physical Review Letters, 129:260501, 2022.

[227] Benoˆıt Vermersch, Marko Ljubotina, J. Ignacio Cirac, Peter Zoller, Maksym Serbyn, and Lorenzo Piroli. Many-Body Entropies and Entanglement from Polynomially Many Local Measurements. Physical Review X, 14:031035, 2024.

[228] Hong-Ye Hu, Muzhou Ma, Weiyuan Gong, Qi Ye, Yu Tong, Steven T. Flammia, and Susanne F. Yelin. Ansatz-Free Hamiltonian Learning with Heisenberg-Limited Scaling. PRX Quantum, 6:040315, 2025.

[229] Hamza Jnane, Jonathan Steinberg, Zhenyu Cai, H. Chau Nguyen, and B´alint Koczor. Quantum Error Mitigated Classica Shadows. PRX Quantum, 5:010324, 2024.

[230] J. Helsen, M. Ioannou, J. Kitzinger, E. Onorati, A. H. Werner, J. Eisert, and I. Roth. Shadow estimation of gate-set properties from random sequences. Nature Communications, 14:5039, 2023.

[231] Yongtao Zhan, Andreas Elben, Hsin-Yuan Huang, and Yu Tong. Learning Conservation Laws in Unknown Quantum Dynamics. PRX Quantum, 5:010350, 2024.

[232] Sebastian Diehl, Andrea Micheli, Adrian Kantian, B Kraus, Hans Peter B¨uchler, and Peter Zoller. Quantum states and phases in driven open quantum systems with cold atoms. Nature Physics, 4(11):878–883, 2008.

[233] Lukas M Sieberer, Sebastian D Huber, Ehud Altman, and S Diehl. Dynamical critical phenomena in driven-dissipative systems. Physical review letters, 110(19):195301, 2013.

[234] Mattias Fitzpatrick, Neereja M Sundaresan, Andy CY Li, Jens Koch, and Andrew A Houck. Observation of a dissipative phase transition in a one-dimensional circuit qed lattice. Physical Review X, 7(1):011016, 2017.

[235] Jonathan Romero, Jonathan P Olson, and Alan Aspuru-Guzik. Quantum autoencoders for eficient compression of quantum data. Quantum Science and Technology, 2(4):045001, 2017.

[236] Supanut Thanasilp, Samson Wang, Nhat Anh Nghiem, Patrick Coles, and Marco Cerezo. Subtleties in the trainability of quantum machine learning models. Quantum Machine Intelligence, 5(1):21, 2023.

## CONTENTS

Introduction 1   
Results 3   
Failures of paper-to-result reproduction 3   
Performance across frontier AI scientists 5   
RA: Self-improving AI scientists 7   
Discussion 7   
References 8   
A. Preliminary and literature review 18   
1. Classical simulation of quantum systems 19   
2. Quantum algorithms 21   
3. Property characterization of quantum systems 24   
4. AI agents and harnesses 27   
5. Literature review 28   
B. Implementation details of OmniQ 29   
1. Workflow of OmniQ for selecting and organizing research papers 29   
2. Construction of computational quantum science dataset 31   
3. Analysis of paper-to-result reproduction issues 34   
a. Level I (Paper #46): Hyperparameter inconsistency in quantum error mitigation 34   
b. Level I (Paper #14): Unclear Hamiltonian specification in dynamics simulation 37   
c. Level I (Paper #38): Inconsistent measurement-cost criteria in the training of variational quantum   
algorithms 40   
d. Level II (Paper #42): Underspecified implementation details in topological phase characterization 42   
e. Level III (Paper #60): Unresolved inconsistencies in the reproduction of quantum kernel concentration 45   
C. Implementation of RA 48   
1. Overview 48   
2. Planner 48   
3. Interactive mechanism of critic and executor 49   
4. Judge agent 51   
5. Experience distiller 51   
D. An autonomous platform for quantum computational science 52   
E. More experimental results 53   
1. Additional results of AI scientists on the established computational quantum science dataset 53   
2. Case studies of RA 54

## SI A: Preliminary and literature review

For completeness, this section provides background on computational quantum science and AI scientists. We first introduce three primary areas of computational quantum science: classical simulation of quantum systems in SI. A 1, quantum algorithms in SI. A 2, and quantum property characterization in SI. A 3. For each area, we review repre sentative methods and computational procedures relevant to our reproduction tasks. Next, we recap representative AI agents and harnesses in SI. A 4, followed by a review of related work on AI scientists in computational quantum science in SI. A 5.

## 1. Classical simulation of quantum systems

Classical simulation uses classical resources to predict quantum properties and assess potential quantum computational advantages. Here we explain how to estimate observable expectation values, which can be performed either by explicitly simulating quantum states or through methods that directly estimate the target quantities. Mathematically, given classical descriptions of an N-qubit initial state $\rho _ { 0 }$ , a quantum evolution channel $u ,$ and an observable O, the goal is to estimate the expectation value

$$
f _ { O } ( \rho ) = \mathrm { T r } \left[ \mathcal { U } ( \rho _ { 0 } ) O \right] ,\tag{A1}
$$

where $\rho = \mathcal { U } ( \rho _ { 0 } )$ denotes the evolved quantum state. For unitary dynamics described by a unitary operator $U _ { ; }$ , the corresponding channel takes the form $\bar { \mathcal { U } } ( \rho _ { 0 } ) = U \rho _ { 0 } U ^ { \dag }$

A central challenge in classical simulation is the exponential growth of the Hilbert space with the number of qubits $N .$ , making general quantum-state simulation computationally expensive. To address this challenge, various classical simulation algorithms have been developed. In what follows, we introduce six representative classes of classica simulation methods: tensor networks, neural quantum states, Pauli propagation, quantum Monte Carlo methods, Lindbladian simulation for open quantum systems, and free-fermion simulation. For each class, we summarize its formulation, computational principles, and conditions for eficient simulation.

Tensor networks. Tensor-network methods simulate quantum dynamics by representing the evolving quantum state as a network of interconnected tensors [93–95]. Specifically, consider a decomposition of the evolution channel into elementary channels $\mathcal { U } = \mathcal { U } _ { L } \circ \cdot \cdot \cdot \circ \mathcal { U } _ { 1 }$ . At each stage of the evolution, the N-qubit state, described by a density matrix $\begin{array} { r } { \rho = \sum _ { i , j } \rho _ { i j } \vert i \rangle \langle j \vert } \end{array}$ , is approximated by a tensor network on a graph $G = ( V , E )$ with $| V | = N , { \mathrm { i . e . } }$

$$
\rho _ { i j } \approx \sum _ { \alpha _ { 1 } = 1 } ^ { \chi _ { 1 } } \cdot \cdot \cdot \sum _ { \alpha _ { \left| E \right| } = 1 } ^ { \chi _ { \left| E \right| } } \prod _ { v \in V } A _ { i _ { v } , j _ { v } , \left\{ \alpha _ { e } : e \ni v \right\} } ^ { [ v ] } ,
$$

where $\pmb { i } = ( i _ { 1 } , \dots , i _ { N } )$ and ${ \pmb j } = ( j _ { 1 } , \dots , j _ { N } )$ are computational-basis indices, $A ^ { [ v ] }$ denotes the local tensor at vertex v, and each edge $e \in E$ carries a bond index $\alpha _ { e }$ of dimension $\chi _ { e }$ . The state at the next stage, following the application of an m-qubit channel $\mathcal { U } _ { \ell }$ to the qubit set $S _ { \ell } ,$ can then be approximated by applying $\mathcal { U } _ { \ell }$ to the contracted tensor $\mathcal { T } _ { \ell }$ of the corresponding subgraph and decomposing it back into the original network structure with truncated bond dimensions, i.e.,

$$
\{ A ^ { [ v ] } \} _ { v \in S _ { \ell } } \xrightarrow { \mathrm { c o n t r a c t i o n } } \mathcal { T } _ { \ell } \xrightarrow { \mathcal { U } _ { \ell } } \mathcal { T } _ { \ell } ^ { \prime } \xrightarrow { \mathrm { d e c o m p o s i t i o n } + \mathrm { t r u n c a t i o n } } \{ \widetilde { A } ^ { [ v ] } \} _ { v \in S _ { \ell } } .
$$

The eficiency of tensor-network simulation depends on both the entanglement structure of the quantum state and the geometry of the underlying network $G .$ States with area-law entanglement often admit compact representations with small bond dimensions $\chi _ { e }$ , whereas volume-law entanglement can require exponentially large bond dimensions. Consequently, tensor-network methods are particularly efective for weakly entangled systems, enabling eficient exact simulation in certain structured cases and accurate approximate simulation more broadly through tensor contraction. Representative methods include matrix product states, matrix product operators, and projected entangled-pair states.

Neural quantum states. According to Eq. (A1), a neural quantum state (NQS) simulates quantum dynamics by approximating the evolving quantum state $\mathcal { U } ( \rho _ { 0 } )$ . Diferent from tensor networks, NQS methods represent the wavefunction amplitudes using a parameterized neural network [96, 97]. For pure-state dynamics with $\rho _ { 0 } = | \psi _ { 0 } \rangle \langle \psi _ { 0 } |$ and $\mathcal { U } ( \rho _ { 0 } ) = \vert \psi \rangle \langle \psi \vert$ , the goal is to approximate |ψ⟩ with high fidelity using an $\mathrm { N Q S } \ | \psi _ { \pmb { \theta } } \rangle$ of the form

$$
| \psi _ { \pmb \theta } \rangle : = \frac { 1 } { \sqrt { \sum _ { \pmb i } | g _ { \pmb \theta } ( \pmb i ) | ^ { 2 } } } \sum _ { i } g _ { \pmb \theta } ( i ) | \pmb i \rangle ,
$$

where $g _ { \boldsymbol { \theta } } ( \cdot )$ is a complex-valued neural network that outputs an unnormalized amplitude for each configuration i. A common strategy for simulating quantum dynamics with neural quantum states (NQS) is to sequentially optimize the network parameters after each elementary unitary evolution. Starting from an initial $\mathrm { N Q S } \ | \psi _ { \pmb { \theta } _ { 0 } } \rangle = | \bar { \psi } _ { 0 } \rangle$ , the parameters $\pmb { \theta } _ { \ell }$ at step ℓ are obtained by minimizing the infidelity between the normalized variational state $| \psi _ { \pmb { \theta } } \rangle$ and the target state $U _ { \ell } | \hat { \psi _ { \pmb { \theta } _ { \ell - 1 } } } \rangle , \mathrm { i . e . , } \mathcal { T } _ { \ell } ( \pmb { \theta } ) = \overset { \sim } { 1 } - | \langle \psi _ { \pmb { \theta } } | U _ { \ell } | \overset { \sim } { \psi _ { \pmb { \theta } _ { \ell - 1 } } } \rangle | ^ { 2 }$ with ${ \pmb \theta } _ { \ell } \in \mathrm { a r g } \operatorname* { m i n } _ { { \pmb \theta } } { \pmb { \mathcal { T } } } _ { \ell } ( { \pmb \theta } )$ . To enable eficient optimization, the infidelity gradient can be expressed in terms of expectations over the Born distribution $p _ { \pmb { \theta } } ( i ) = | \langle i | \psi _ { \pmb { \theta } } \rangle | ^ { 2 }$ as

$$
\frac { \partial \mathcal { T } _ { k } } { \partial \theta _ { a } } = - 2 \operatorname { R e } \left[ \mathbb { E } _ { i \sim p _ { \theta } } \left[ \frac { \langle i | U _ { \ell } | \psi _ { \theta _ { \ell - 1 } } \rangle } { \langle i | \psi _ { \theta } \rangle } \right] ^ { * } \mathbb { E } _ { i \sim p _ { \theta } } \left[ \frac { \langle i | U _ { \ell } | \psi _ { \theta _ { \ell - 1 } } \rangle } { \langle i | \psi _ { \theta } \rangle } \left( O _ { a } ^ { * } ( i ) - \mathbb { E } _ { p _ { \theta } } [ O _ { a } ^ { * } ] \right) \right] \right] ,\tag{A2}
$$

where $O _ { a } ( i ) = \partial _ { \pmb { \theta } _ { } }$ <sub>a</sub> log $g _ { \pmb { \theta } } ( \pmb { i } )$ is the logarithmic derivative of the neural-network amplitude.

The infidelity gradient in Eq. (A2) can be estimated using Monte Carlo sampling, allowing NQS parameters to be optimized without explicitly representing the full quantum state. However, gradient estimation can sufer from numerical instability and sampling ineficiency, making optimization challenging [98]. Improved neural-network architectures, sampling strategies, and gradient estimators have been developed to address these limitations [99], enabling NQS to simulate increasingly complex quantum states and dynamics.

Pauli propagation. Pauli propagation simulates quantum dynamics by evolving observables in the Heisenberg picture and is particularly eficient for circuits with limited non-Cliford resources [100, 101]. Following Eq. (A1), for an N-qubit channel U and an observable $O ,$ the expectation value is given by $f _ { O } \dot { ( } \rho _ { 0 } \big ) = \dot { \mathrm { T r } } [ O \mathcal { U } ( \rho _ { 0 } ) ] \stackrel { - } { = } \mathrm { T r } [ \mathcal { U } ^ { \dag } ( O ) \rho _ { 0 } ]$ where $\mathcal { U } ^ { \dagger } = \mathcal { U } _ { 1 } ^ { \dagger } \circ \cdots \circ \mathcal { U } _ { L } ^ { \dagger }$ is the adjoint channel. The evolved observable is represented as a Pauli expansion, i.e.,

$$
\mathcal { U } ^ { \dagger } ( O ) = \sum _ { { \pmb a } \in A } c _ { \pmb { a } } P _ { \pmb { a } } ,
$$

where $A \subseteq \mathbb { F } _ { 2 } ^ { 2 N }$ indexes the nonzero terms, $P _ { \pmb { a } } \in \{ \mathbb { I } , X , Y , Z \} ^ { \otimes N }$ denotes a Pauli string, and $c _ { a }$ is its coeficient. The expectation value is then evaluated as $\begin{array} { r } { f _ { O } ( \rho _ { 0 } ) = \sum _ { { \pmb a } \in A } c _ { \pmb { a } } \mathrm { T r } ( P _ { { \pmb a } } \rho _ { 0 } ) } \end{array}$ . As a result, the computational eficiency depends on the growth of the Pauli expansion during backward propagation.

To reduce this computational cost, approximate Pauli-propagation algorithms truncate the Pauli expansion during the propagation process by retaining nontrivial Pauli strings [101]. Denote the retained Pauli strings by $A ^ { \prime } \subseteq \mathbb { F } _ { 2 } ^ { 2 N }$ The resulting approximation of the evolved observable takes the form $\begin{array} { r } { \hat { \mathcal { U } } ^ { \dagger } ( O ) = \sum _ { { \pmb { a } } \in A ^ { \prime } } \hat { \bf c } _ { { \pmb { a } } } P _ { { \pmb { a } } } } \end{array}$ , and the corresponding estimate of $f _ { O } ( \rho )$ is given by

$$
\hat { f } _ { O } ( \rho ) = \mathrm { T r } [ \hat { \mathcal { U } } ^ { \dag } ( O ) \rho _ { 0 } ] = \sum _ { { \bf a } \in A ^ { \prime } } \hat { c } _ { a } \mathrm { T r } ( P _ { a } \rho _ { 0 } ) ,
$$

where $\hat { c } _ { a }$ denotes the coeficient obtained from the truncated propagation procedure. An efective Pauli-propagation algorithm should satisfy $| f _ { O } ( \rho ) - \hat { f } _ { O } ( \rho ) | \le \epsilon$ for a prescribed accuracy $\epsilon ,$ while keeping the number of retained Pauli terms and the computational cost manageable. Typical approaches in this regime include Pauli-weight truncation, coeficient-based pruning, and Monte Carlo sampling of Pauli paths [100–102].

Quantum Monte Carlo methods. Quantum Monte Carlo methods directly estimate observable expectations $\mathrm { T r } [ O \mathcal { U } ( \rho _ { 0 } ) ]$ through stochastic sampling of suitable representations of quantum systems [103–106]. Depending on the formulation, these methods can address equilibrium properties, quantum dynamics, and quantum-circuit simulations, while their eficiency may be limited by sign or phase problems. As a representative example, consider the estimation of thermal expectation values for a Hamiltonian H at inverse temperature $\beta ,$ given by

$$
\mathrm { T r } [ { \cal O } \mathcal { U } _ { \beta } ( \rho _ { 0 } ) ] = \frac { \mathrm { T r } ( { \cal O } e ^ { - \beta H } ) } { \mathrm { T r } ( e ^ { - \beta H } ) } = \sum _ { i } \frac { \langle i | e ^ { - \beta H } | i \rangle } { \mathrm { T r } ( e ^ { - \beta H } ) } \frac { \langle i | { \cal O } e ^ { - \beta H } | i \rangle } { \langle i | e ^ { - \beta H } | i \rangle } ,
$$

where $\mathcal { U } _ { \beta } ( \rho _ { 0 } ) = \mathrm { T r } ( \rho _ { 0 } ) e ^ { - \beta H } / \mathrm { T r } ( e ^ { - \beta H } )$ refers to a thermal-state preparation channel.

To exploit Hamiltonian locality, imaginary-time QMC decomposes $e ^ { - \beta H } = ( e ^ { - \Delta \tau H } ) ^ { L }$ , where $\Delta \tau = \beta / L$ , and inserts computational-basis resolutions of the identity between successive time slices. This yields a path-integral representation of the thermal expectation value $\begin{array} { r } { \langle O \rangle _ { \beta } = \sum _ { \mathcal { C } } p ( \mathcal { C } ) O _ { \mathrm { l o c } } ( \mathcal { C } ) } \end{array}$ , where $\mathcal { C } = ( i _ { 1 } , \dots , i _ { L } )$ is a configuration path with $\pmb { i } _ { L + 1 } = \pmb { i } _ { 1 }$ , and

$$
p ( \mathcal { C } ) = \frac { w ( \mathcal { C } ) } { \sum _ { \mathcal { C } ^ { \prime } } w ( \mathcal { C } ^ { \prime } ) } , \quad w ( \mathcal { C } ) = \prod _ { \ell = 1 } ^ { L } \langle i _ { \ell + 1 } | e ^ { - \Delta \tau H } | i _ { \ell } \rangle , \quad O _ { \mathrm { l o c } } ( \mathcal { C } ) = \frac { \langle i _ { 2 } | e ^ { - \Delta \tau H } O | i _ { 1 } \rangle } { \langle i _ { 2 } | e ^ { - \Delta \tau H } | i _ { 1 } \rangle } .
$$

When $w ( \mathcal { C } ) \geq 0$ , the normalized weights define a sampling distribution, allowing the expectation value to be estimated without explicitly summing over exponentially many paths, i.e., $\begin{array} { r } { \langle O \rangle _ { \beta } \approx \frac { 1 } { M } \sum _ { m = 1 } ^ { M } O _ { \mathrm { l o c } } ( \mathcal { C } ^ { ( m ) } ) } \end{array}$ with ${ \mathcal { C } } ^ { ( m ) } \sim p$

Lindbladian simulation for open quantum systems. Lindbladian simulation provides a framework for simulating the dynamics of an open quantum system interacting with an environment without explicitly representing the environmental degrees of freedom [107]. Under the assumption of time-homogeneous Markovian dynamics, the evolution channel over time $T$ takes the Lindblad form $\boldsymbol { \mathcal { U } } = e ^ { T \dot { \boldsymbol { \mathcal { L } } } }$ , where the Lindblad generator $\mathcal { L }$ is defined by

$$
\mathcal { L } ( \rho _ { t } ) = - \imath [ H , \rho _ { t } ] + \sum _ { \mu } \left( L _ { \mu } \rho _ { t } L _ { \mu } ^ { \dagger } - \frac { 1 } { 2 } \{ L _ { \mu } ^ { \dagger } L _ { \mu } , \rho _ { t } \} \right) ,\tag{A3}
$$

where H is the system Hamiltonian, $\rho _ { t }$ is the system density matrix at time $t \leq T , L _ { \mu }$ are the jump operators describing dissipative processes, and $[ A , B ] = { \dot { A } } B - B A$ and $\{ A , B \} = A B + B A$ denote the commutator and anticommutator, respectively.

A standard approach to evaluating $f _ { O } ( \rho )$ in Eq. (A1) under Lindblad dynamics is the quantum-trajectory method [108, 109], which unravels the mixed-state evolution into an ensemble of stochastic pure-state trajectories, i.e., $\rho _ { t } = \mathbb { E } _ { \xi \sim p } [ | \psi _ { t } ^ { ( \xi ) } \rangle \langle \psi _ { t } ^ { ( \xi ) } | ]$ . Each trajectory $| \psi _ { t } ^ { ( \xi ) } \rangle$ undergoes non-unitary evolution governed by the efective non-Hermitian Hamiltonian $\begin{array} { r } { \dot { H _ { \mathrm { e f f } } } = H - \frac { \ i } { 2 } \sum _ { \mu } L _ { \mu } ^ { \dagger } L _ { \mu } ^ { ' } } \end{array}$ , derived from the Lindblad generator in Eq. (A3), interrupted by stochastic quantum jumps associated with the operators $L _ { \mu }$ . The observable expectation is then estimated by averaging over M independently sampled trajectories:

$$
\mathrm { T r } ( { \cal O } \rho _ { T } ) \approx \frac { 1 } { M } \sum _ { m = 1 } ^ { M } \langle \psi _ { T } ^ { ( m ) } | { \cal O } | \psi _ { T } ^ { ( m ) } \rangle .
$$

Free-fermion simulation. Free-fermion methods enable eficient classical simulation of fermionic Gaussian states evolving under quadratic Hamiltonians, without explicitly representing the exponentially large quantum state [110, 111]. To evaluate the observable expectation $f _ { O } ( \rho )$ in $\mathrm { E q . ~ ( A 1 ) }$ , we first introduce suitable representations of the initial state, its evolution, and the target observable.

For an N-mode fermionic system, define the Majorana operators $\gamma _ { 2 j - 1 } = { \pmb a } _ { j } + { \pmb a } _ { i } ^ { \dagger }$ and $\gamma _ { 2 j } = - \imath ( { \bf a } _ { j } - { \bf a } _ { j } ^ { \dagger } )$ , where $a _ { j } ^ { \dagger }$ and ${ \pmb a } _ { j }$ denote fermionic creation and annihilation operators, respectively. A full-rank fermionic Gaussian state can be expressed as $\rho _ { 0 } = e ^ { - K _ { 0 } } / \mathrm { T r } ( e ^ { - K _ { 0 } } )$ , where $K _ { 0 } = ( \iota / 4 ) \gamma ^ { \top } B _ { 0 } \gamma$ and $B _ { 0 }$ is a real antisymmetric $2 N \times 2 N$ matrix. Consider the evolution $\mathcal { U } = \mathcal { U } _ { L } \circ \cdot \cdot \cdot \circ \mathcal { U } _ { 1 }$ , where each unitary channel is generated by a quadratic Hamiltonian $H _ { \ell } = ( \imath / 4 ) \gamma ^ { \top } M _ { \ell } \gamma .$ with $M _ { \ell }$ real and antisymmetric. The evolved state remains Gaussian, and its coeficient matrix is updated as

$$
B _ { \ell } = Q _ { \ell } B _ { \ell - 1 } Q _ { \ell } ^ { \top } \qquad \mathrm { w i t h } \ Q _ { \ell } = e ^ { M _ { \ell } x _ { \ell } } ,
$$

where $x _ { \ell }$ denotes the evolution time at step ℓ. The final state $\rho = \mathcal { U } ( \rho _ { 0 } )$ is thus characterized by $B _ { L }$ , from which its covariance matrix $\begin{array} { r } { \Gamma _ { a b } = \frac { \imath } { 2 } \mathrm { T r } ( \rho [ \gamma _ { a } , \gamma _ { b } ] ) } \end{array}$ can be obtained as $\Gamma = - \tan ( B _ { L } / 2 )$ . In addition, we consider Hermitian observables of the form $O = \imath ^ { k } \gamma _ { a _ { 1 } } \cdot \cdot \cdot \gamma _ { a _ { 2 k } }$ , where $a _ { 1 } , \ldots , a _ { 2 k }$ denote distinct Majorana indices.

With the above notations, the expectation value $f _ { O } ( \rho )$ under Free-fermion system can then be eficiently evaluated using Wick’s theorem [112]. Mathematically, we have

$$
f _ { O } ( \rho ) = \mathsf { P f } ( \Gamma _ { A } ) ,
$$

where Pf(·) denotes the Pfafian of an antisymmetric matrix, and $\Gamma _ { A }$ is the $2 k \times 2 k$ submatrix of Γ obtained by selecting the rows and columns indexed by $A = \left( a _ { 1 } , \ldots , a _ { 2 k } \right)$ , as specified by the Majorana operators appearing in O.

## 2. Quantum algorithms

Quantum algorithms address computational problems by processing information encoded in quantum states and extracting results through measurement. We focus on five representative categories spanning both fault-tolerant and near-term quantum computing: non-variational quantum simulation, quantum phase estimation, variational quantum simulation, and quantum machine learning, quantum-enhanced Markov chain Monte Carlo. The diversity of computational approaches and implementation strategies within these categories further complicates autonomous reproduction, requiring AI scientists to reconstruct the specific methods adopted in individual studies. For each category, we introduce the problem formulation and basic workflows of representative algorithms.

Non-variational quantum simulation. Quantum simulation aims to understand the properties and dynamics of quantum systems, whose exponentially growing state spaces often make classical computation intractable. Nonvariational approaches address this challenge by directly constructing quantum operations from the system Hamiltonian to simulate time evolution or prepare ground states [113]. Mathematically, for a Hamiltonian $H ( t )$ and an initial state |ψ(0)⟩, the evolving state obeys the Schr¨odinger equation

$$
\partial _ { t } | \psi ( t ) \rangle = - \imath H ( t ) | \psi ( t ) \rangle .\tag{A4}
$$

Below, we discuss this approach for time-independent and time-dependent Hamiltonians, respectively.

For time-independent H in $\mathrm { E q . \ ( A 4 ) }$ , its solution is $| \psi ( t ) \rangle = e ^ { - \imath H t } | \psi ( 0 ) \rangle$ . Trotterization approximates this evolution by composing operations generated by individual Hamiltonian terms [114]. Let the explored Hamiltonian be $H =$

$\sum _ { \ell = 1 } ^ { m } h _ { \ell }$ , where each Hermitian term $h _ { \ell }$ acts on a small number of qubits, and dividing the total time $t > 0$ into L steps of duration $\Delta t = t / L$ gives the first-order approximation

$$
U _ { \mathrm { T r o t } } ( t ) = \left( e ^ { - \imath h _ { m } \Delta t } \cdot \cdot \cdot e ^ { - \imath h _ { 1 } \Delta t } \right) ^ { L } \approx e ^ { - \imath H t } .
$$

Each local unitary can be decomposed into elementary gates, and applying the full sequence prepares an approximation to $| \psi ( t ) \rangle$ . For fixed $H _ { ; }$ , the error satisfies $\lVert U _ { \mathrm { T r o t } } ( t ) - \mathrm { e x p } ( - \iota H t ) \rVert = \mathcal { O } ( t ^ { 2 } / L )$ [114]. Increasing L reduces the error bound at the cost of greater circuit depth. Higher-order Trotter formulas [115] improve this tradeof by canceling lower-order errors through local evolutions with suitably chosen ordering and durations. Further refinements exploit separated energy scales [116] or introduce auxiliary correctors [117] to reduce product-formula errors.

For time-dependent Hamiltonians $H ( t )$ in Eq. (A4), |ψ(t)⟩ composes short evolutions with the Hamiltonian held fixed within each step [118], enabling digital adiabatic ground-state preparation [119]. More precisely, starting from an easily prepared ground state of $H _ { \mathrm { i n i t } }$ , one varies $\bar { H ( s ) } = ( 1 - s ) \bar { H } _ { \mathrm { i n i t } } + s H$ from $s = 0$ to $s = 1$ . For a smooth path with a nondegenerate ground state and a nonzero spectral gap, suficiently slow evolution follows the instantaneous ground state toward the target ground state [120]. Beyond adiabatic preparation, quantum imaginary-time evolution [121] and adaptive spectral filtering [122] can prepare approximate ground states by suppressing excitedstate components through unitary updates inferred from measurements and postselected filters built from controlled evolution, respectively.

Quantum phase estimation. Quantum phase estimation (QPE) [123] extracts the eigenphases of a unitary operator $U _ { \textrm { \scriptsize i } }$ , defined by $U | \phi _ { j } \rangle = e ^ { \imath \varphi _ { j } } | \phi _ { j } \rangle$ with $\varphi _ { j } \in [ 0 , 2 \pi )$ . QPE ofers rigorous precision guarantees, achieving additive phaseestimation error ε with $\mathcal { O } ( 1 / \varepsilon )$ controlled applications of $U$ under standard assumptions. As a fundamental primitive underlying quantum algorithms such as Shor’s factoring algorithm and Hamiltonian eigenvalue estimation, $\mathrm { Q P E }$ plays a central role in the pursuit of quantum computational advantages.

The mechanism of QPE is as follows. Given an initial state $\begin{array} { r } { | \psi _ { \mathrm { i n i t } } \rangle = \sum _ { j } c _ { j } | \phi _ { j } \rangle } \end{array}$ and m ancillary qubits prepared in $| + \rangle ^ { \otimes m }$ , the circuit performs

$$
| + \rangle ^ { \otimes m } | \psi _ { \mathrm { i n i t } } \rangle \xrightarrow { \prod _ { r = 0 } ^ { m - 1 } \mathbb { C } _ { r } { U ^ { 2 } } ^ { r } } \frac { 1 } { 2 ^ { m / 2 } } \sum _ { j , s } c _ { j } e ^ { \imath s \varphi _ { j } } | s \rangle | \phi _ { j } \rangle \xrightarrow { \mathrm { i n v e r s e ~ Q F T } } \sum _ { j , y } c _ { j } a _ { j y } | y \rangle | \phi _ { j } \rangle .
$$

Here, $\mathrm { C } _ { r } U ^ { 2 ^ { r } }$ denotes $U ^ { 2 ^ { r } }$ controlled by ancillary qubit $r ,$ and the inverse quantum Fourier Transform (QFT) acts on the ancillary register. The indices $s , y ~ \in ~ \{ 0 , \ldots , 2 ^ { m } ~ - ~ 1 \}$ label its computational-basis states, with $a _ { j y } =$ $2 ^ { - m } \sum _ { s = 0 } ^ { 2 ^ { m } - 1 }$ ex $\mathrm { p } ( \imath s ( \varphi _ { j } - 2 \pi y / 2 ^ { m } ) )$ ). Measuring this register yields $y$ with probability $\begin{array} { r } { p ( y ) = \sum _ { j } | c _ { j } | ^ { 2 } | a _ { j y } | ^ { 2 } } \end{array}$ , giving the estimate $\hat { \varphi } = 2 \pi y / 2 ^ { m }$ . For each eigenstate component, $| a _ { j y } | ^ { 2 }$ is concentrated near its eigenphase modulo 2π, with resolution ${ \mathcal { O } } ( 2 ^ { - m } )$ . Improving the resolution requires higher controlled powers of $U .$ , up to $U ^ { 2 ^ { m - 1 } }$

Alternative implementations reduce ancillary-qubit requirements through iterative measurements [124], guide measurements through Bayesian updates [125], or accommodate inexact initial states with suficient target-state overlap [126]. Beyond extracting spectral information, QPE also serves as a subroutine in quantum algorithms such as quantum amplitude estimation [127], which estimates the probability a of a target measurement outcome for a state prepared by a unitary W. Combining a reflection about W|0⟩ with a phase flip on the target outcomes defines an amplification operator $Q$ whose relevant eigenphases ±2θ satisfy $a = \sin ^ { 2 } \theta$ . Applying QPE to $Q$ gives an estimate $\hat { \theta }$ of θ and hence $\hat { a } = \sin ^ { 2 } \hat { \theta }$ , achieving $| \hat { a } - a | \le \epsilon$ with $\mathcal { O } ( \epsilon ^ { - 1 } )$ calls to W and $W ^ { \dagger }$ , compared with $\mathcal { O } ( \epsilon ^ { - 2 } )$ for direct sampling. To reduce circuit depth, classical inference can replace QPE by using measurements at diferent amplification powers [128], with Bayesian schemes adapting these powers while accounting for device noise [129].

Variational quantum simulation. Complementing non-variational approaches, variational quantum simulation aims to investigate quantum states and dynamics using variational quantum circuits (VQCs) tailored to the limited coherence times and circuit depths of near-term quantum devices [130]. It approximates quantum states or time evolution through hybrid quantum-classical optimization, ofering a hardware-eficient alternative to direct Hamiltonianbased simulation. Specifically, the variational quantum eigensolver (VQE) [131] targets ground-state preparation through energy minimization, while variational quantum Hamiltonian evolution [132] simulates dynamics by fitting evolved states or time-evolution operators. In what follows, we present these two standard approaches separately.

Variational quantum eigensolver. VQE approximates the ground state within the circuit family $| \psi ( \pmb \theta ) \rangle = U ( \pmb \theta ) | \psi _ { \mathrm { i n i t } } \rangle$ where $| \psi _ { \mathrm { i n i t } } \rangle$ is an easily prepared reference state and θ denotes the parameters of $U ( \pmb \theta ) , \mathrm { i . e . }$ , the employed VQC with a tailored circuit architecture. For the target Hamiltonian H, the objective is to minimize

$$
E ( \pmb \theta ) = \langle \psi ( \pmb \theta ) | H | \psi ( \pmb \theta ) \rangle \geq E _ { 0 } ,\tag{A5}
$$

where $E _ { 0 }$ is the ground-state energy. A classical optimizer updates θ to minimize Eq. (A5). Gradients used in the optimization can be estimated by evaluating the energy value $E ( \pmb \theta )$ at shifted circuit parameters [133]. The optimized $\operatorname { V Q C } \ ( { \mathrm { i . e . , ~ } } U ( \theta ^ { * } ) )$ prepares a ground-state approximation, whose accuracy depends on the circuit family and the quality of the optimization [134]. Further developments address circuit construction through adaptive operator selection based on energy gradients [135] and parameter initialization through tensor-network pretraining [136].

Beyond ground-state preparation, the VQE framework can be extended to combinatorial optimization with a cost function C(s) over binary configurations $\pmb { s } \in \{ 0 , 1 \} ^ { N }$ , as in the quantum approximate optimization algorithm (QAOA) [137, 138]. Here, the VQC alternates evolution under

$$
H _ { C } = \sum _ { s } C ( s ) | s \rangle \langle s | \quad { \mathrm { a n d } } \quad H _ { B } = \sum _ { j = 1 } ^ { N } X _ { j } .
$$

The evolution times are optimized to minimize Eq. (A5) with $H = H _ { C }$ , after which low-cost solutions are selected from configurations sampled in the computational basis.

Variational quantum Hamiltonian evolution. Variational approaches simulate Hamiltonian dynamics by optimizing $U ( \pmb \theta )$ that prepare approximate evolved states $\lvert \psi ( \pmb \theta ) \rangle = \bar { U } ( \pmb \theta ) \lvert \psi _ { \mathrm { i n i t } } \rangle$ . For a time-independent Hamiltonian H, one approach starts from $\bar { \vert \psi ( { \pmb \theta } ( 0 ) ) \rangle } \approx \vert \psi ( 0 ) \rangle$ ⟩ and fits each short-time evolved state $e ^ { - \imath H \Delta t } | \bar { \psi } ( \pmb { \theta } ( j ) ) \rangle$ with $| \psi ( \pmb \theta ) \rangle$ ⟩ by minimizing their infidelity

$$
\pmb \theta ( j + 1 ) = \operatorname * { a r g m i n } _ { \pmb \theta } \left[ 1 - \left| \langle \psi ( \pmb \theta ) | e ^ { - \imath H \Delta t } | \psi ( \pmb \theta ( j ) ) \rangle \right| ^ { 2 } \right] ,\tag{A6}
$$

where $\pmb \theta ( j )$ denotes the parameters at time $j \Delta t$ [132]. The short-time evolution $\exp ( - \imath H \Delta t )$ can be approximated by Trotterization. Repeating the update approximates the state $| \psi ( t ) \rangle$ at successive times.

Subsequent studies have explored complementary strategies to improve the eficiency and scalability of variational quantum Hamiltonian evolution. To reduce the required quantum resources, circuit knitting estimates the fidelity in Eq. (A6) using smaller subcircuits while constraining the sampling overhead [139]. Other approaches approximate the time-evolution operator $\exp ( - \imath H \Delta t )$ directly by learning a diagonal factorization to extend simulation times [140] or fitting its action on subsystems to reduce the number of qubits required for training [141].

Quantum machine learning. Quantum machine learning (QML) aims to leverage quantum computing to learn patterns from classical or quantum data, with the potential to enhance the expressivity and computational eficiency of learning models [142]. Two representative formulations are quantum neural networks (QNNs) [143], which optimize circuits to learn predictive models, and quantum kernel methods [144], which use quantum circuits to estimate kernel values for classical model fitting.

Quantum neural networks. A typical QNN encodes an input x as $| \phi ( { \pmb x } ) \rangle = V ( { \pmb x } ) | 0 \rangle$ and processes it with a VQC $U ( \pmb \theta )$ . The expectation value of an observable O defines the model output, i.e.,

$$
f _ { \pmb \theta } ( \pmb x ) = \langle \phi ( \pmb x ) | U ^ { \dagger } ( \pmb \theta ) O U ( \pmb \theta ) | \phi ( \pmb x ) \rangle .
$$

For supervised learning, consider a training set $\mathcal { D } = \{ ( \boldsymbol { { \mathbf { x } } } ^ { ( i ) } , y ^ { ( i ) } ) \} _ { i = 1 } ^ { M }$ , where $\boldsymbol y ^ { ( i ) }$ is the target associated with input $\pmb { x } ^ { ( i ) }$ . The parameters in $f _ { \pmb \theta } ( \pmb x )$ are optimized to minimize the empirical loss $\begin{array} { r } { \mathcal { L } ( \pmb { \theta } ) = M ^ { - 1 } \sum _ { i = 1 } ^ { M } \ell ( f _ { \pmb { \theta } } ( \pmb { x } ^ { ( i ) } ) , y ^ { ( i ) } ) } \end{array}$ , where $\ell ( z , y )$ measures the discrepancy between a model output z and its target y. For regression, one can use the squarederror loss $\ell ( z , y ) = ( z - y ) ^ { 2 }$ , and for binary classification with $y \in \{ - 1 , 1 \}$ , the logistic loss $\ell ( z , y ) = \log ( 1 + \exp ( - y z ) )$ ). Once trained, the circuit predicts outputs for new inputs.

Subsequent studies have explored both the applications and theoretical properties of QNNs. Beyond predictive learning, QNNs have been extended to generative modeling of probability distributions [145, 146] and reinforcement learning of decision policies [147]. Meanwhile, theoretical studies have investigated the expressivity, trainability, and generalization of QNNs. Expressivity analyses characterize the functions and quantum states representable by parameterized circuits [148]. Regarding trainability, global cost functions and scrambling dynamics can cause gradients to vanish exponentially [149]. Generalization analyses further investigate how model complexity, training data, and quantum measurements afect predictive performance on unseen inputs [150].

Quantum kernel methods. Quantum kernel methods construct predictors from a kernel $\kappa ( \pmb { x } , \pmb { x } ^ { \prime } )$ , defined as an inner product between input representations in a feature space [144, 151]. For the training set ${ \mathcal { D } } ,$ the predictor takes the form $\begin{array} { r } { f ( \pmb { x } ) = \sum _ { i = 1 } ^ { M } \pmb { \alpha } _ { i } \kappa ( \pmb { x } , \pmb { x } ^ { ( i ) } ) } \end{array}$ , where $\pmb { \alpha } \in \mathbb { R } ^ { M }$ contains the learned coeficients. A typical example of quantum kernels is the fidelity kernel $\kappa ( \pmb { x } , \pmb { x } ^ { \prime } ) = \mathrm { T r } [ \rho ( \pmb { x } ) \rho ( \pmb { x } ^ { \prime } ) ]$ , where $\rho ( { \pmb x } ) = | \phi ( { \pmb x } ) \rangle \langle \phi ( { \pmb x } ) |$ |. The value of this kernel equals the probability of the all-zero outcome after applying $\bar { V ^ { \dag } } ( { \pmb x } ) V ( { \pmb x } ^ { \prime } )$ to |0⟩ and measuring in the computational basis. The coeficients $\pmb { \alpha } = ( \pmb { \alpha } _ { 1 } , \cdots , \pmb { \alpha } _ { M } ) ^ { \top }$ can then be fitted classically by minimizing the regularized least-squares objective of kernel ridge regression,

$$
\pmb { \alpha } ^ { * } = \arg \operatorname* { m i n } _ { \pmb { \alpha } \in \mathbb { R } ^ { M } } \left\{ \| K \pmb { \alpha } - \pmb { y } \| _ { 2 } ^ { 2 } + \lambda \pmb { \alpha } ^ { \top } K \pmb { \alpha } \right\} , i . e . ,
$$

where $K$ is the kernel matrix with entries $K _ { i j } = \kappa ( \pmb { x } ^ { ( i ) } , \pmb { x } ^ { ( j ) } ) , \pmb { y } = ( y ^ { ( 1 ) } , \dots , y ^ { ( M ) } ) ^ { \top }$ , and $\lambda > 0$ is a regularization parameter [152]. Solving $\begin{array} { r } { ( K + \lambda \mathbb { I } _ { M } ) \pmb { \alpha } ^ { * } = \pmb { y } } \end{array}$ gives the fitted coeficients $\ b { \alpha } ^ { * }$ . The prediction for a new input x is then computed using the fitted coeficients $\ b { \alpha } ^ { * }$ and the kernel values $\kappa ( \pmb { x } , \pmb { x } ^ { ( i ) } )$ between this input and the training samples. Following this framework, projected kernels select observable features to improve generalization [69], kernel alignment adapts data encodings to the learning task [153, 154], and covariant kernels exploit group structure in the data [155].

Quantum-enhanced Markov chain Monte Carlo. Quantum-enhanced Markov chain Monte Carlo (MCMC) [156] combines quantum proposals with classical acceptance rules to sample equilibrium distributions and estimate thermal averages. $\mathrm { B y }$ potentially accelerating the exploration of complex energy landscapes, it ofers a promising approach to studying many-body systems and addressing sampling bottlenecks in statistical physics and computational chemistry.

A representative setting of Quantum-enhanced MCMC is a classical Ising model with configurations $s \in \{ - 1 , 1 \} ^ { \bar { N } }$ and energy $E ( s )$ . The target is the Boltzmann distribution $\pi ( \pmb { \mathscr { s } } ) = \exp ( - \beta E ( \pmb { \mathscr { s } } ) ) / \mathcal { Z }$ , where $\beta$ is the inverse temperature and $\begin{array} { r } { \mathcal { Z } = \sum _ { s } \mathrm { e x p } ( - \beta E ( \pmb { s } ) ) } \end{array}$ is the partition function. At each step of the Markov chain, the current configuration is encoded in a computational basis state $| s \rangle$ and evolved under the Hamiltonian $\begin{array} { r } { H _ { \mathrm { p r o p } } = \sum _ { s } E ( s ) | s \rangle \langle s | + \eta \sum _ { j = 1 } ^ { N } X _ { j } , } \end{array}$ where $\eta \in \mathbb { R }$ controls the spin-flip strength. After evolution for a time t, a computational-basis measurement yields a candidate $s ^ { \prime }$ with probability $p ( \tilde { \mathbf { \rho } } ^ { \prime } | s ) = \breve { | } \langle s ^ { \prime } | \exp ( - \imath H _ { \mathrm { p r o p } } t ) | s \rangle | ^ { 2 }$ . Since $H _ { \mathrm { p r o p } }$ is real and symmetric, $p ( \pmb { \mathscr { s } } ^ { \prime } | \pmb { \mathscr { s } } ) = p ( \pmb { \mathscr { s } } | \pmb { \mathscr { s } } ^ { \prime } )$ so the Metropolis–Hastings rule used to update the Markov chain accepts the proposed configuration $s ^ { \prime }$ with a probability that can be computed classically from the energy diference, $\mathrm { i . e . , }$

$$
\operatorname* { P r } _ { \operatorname { a c c } } ( { \pmb s } ^ { \prime } | { \pmb s } ) = \operatorname* { m i n } \left\{ 1 , \frac { \pi ( { \pmb s } ^ { \prime } ) p ( { \pmb s } | { \pmb s } ^ { \prime } ) } { \pi ( { \pmb s } ) p ( { \pmb s } ^ { \prime } | { \pmb s } ) } \right\} = \operatorname* { m i n } \left\{ 1 , e ^ { - \beta \left[ E ( { \pmb s } ^ { \prime } ) - E ( { \pmb s } ) \right] } \right\} .
$$

This rule preserves π as the stationary distribution, so averages along an ergodic chain converge to the corresponding thermal expectations. The spin-flip strength η and evolution time t can be chosen to favor transitions between configurations that difer in many spins but have similar energies, which can accelerate equilibration compared with single-spin updates.

Subsequent studies have further explored improved quantum proposal strategies [157], faster mixing [158], and reduced quantum resource requirements [159], extending its applicability to larger and more complex systems [160, 161].

## 3. Property characterization of quantum systems

Quantum property characterization protocols aim to infer physical properties of quantum systems from finite measurement results. Following the framework in Ref. [3], we review five representative directions: linear property estimation, nonlinear property estimation, Hamiltonian learning, quantum error mitigation, and quantum process characterization. For each direction, we outline representative algorithms, focusing on their problem formulations, key assumptions, and computational procedures.

Linear properties of quantum states. The linear property of an N-qubit quantum state $\rho$ refers to the expectation value of a fixed observable $O , \mathrm { i . e . , } f _ { O } ( \rho )$ in Eq. (A1). Applying diferent observables $O$ can characterize various physical quantities of quantum systems, such as magnetization, energy, and correlation functions. Two complementary approaches to estimating $f _ { O } ( \rho )$ are classical shadows [162] and classical surrogates [26]. That is, classical shadows estimate multiple observable expectation values from measurements of a fixed quantum state, whereas classical surrogates learn to predict these values across a parametrized family of quantum states. Their mechanisms are summarized below.

Classical shadows. Classical shadows enable the prediction of multiple quantum properties from shared measurement data without additional quantum measurements. Its mechanism is as follows. For an unknown N-qubit state $\rho ,$ the classical-shadow protocol performs $T$ independent measurement rounds. In each round, a unitary U is sampled from a prescribed ensemble U and applied to a fresh copy of $\rho ,$ followed by a computational-basis measurement. The outcome is an N-bit string ${ \pmb b } \in \{ 0 , 1 \} ^ { \hat { N } }$ , obtained with probability $\mathbb { P } ( \pmb { b } | U ) = \langle \pmb { b } | U \hat { \rho } U ^ { \dag } | \pmb { b } \rangle$ . This procedure defines a linear map $\mathcal { M }$ , known as the measurement channel, i.e., $\begin{array} { r } { \mathcal { M } ( \rho ) = \mathbb { E } _ { U \sim \mathcal { U } } \mathbb { E } _ { b \sim \mathbb { P } ( b ) } \dot { U } ^ { \dagger } \left. \dot { b } \right. \left. b \right. \dot { U } = \sum _ { b } \bar { \int } \mathsf { d } U U ^ { \dagger } \left. b \right. \left. b \right. U \left. b \right. U \rho U ^ { \dagger } \left. b \right. } \end{array}$

When the measurement channel $\mathcal { M }$ is invertible, each measurement outcome yields a classical-shadow snapshot $\hat { \rho } _ { t } = \mathcal { M } ^ { - 1 } ( U _ { t } ^ { \dagger } | b _ { t } \rangle \langle b _ { t } | U _ { t } )$ . For randomized single-qubit Pauli measurements, where $U _ { t } \sim \mathrm { C l } ( 2 ) ^ { \otimes N }$ , the snapshot takes

the explicit form

$$
\hat { \rho } _ { t } = \bigotimes _ { j = 1 } ^ { N } \left( { 3 U _ { j , t } ^ { \dag } | b _ { j , t } \rangle \langle b _ { j , t } | U _ { j , t } - \mathbb { I } _ { 2 } } \right) .
$$

Averaging $T$ independent snapshots gives ${ \hat { \rho } } _ { T } = T ^ { - 1 } \sum _ { t = 1 } ^ { T } { \hat { \rho } } _ { t }$ , which provides an unbiased estimator of $\rho .$ The same snapshots can then estimate multiple observables $\{ O _ { i } \} _ { i = 1 } ^ { M }$ through $\begin{array} { r } { \hat { f } _ { O _ { i } } ( \rho ) = T ^ { - 1 } \sum _ { t = 1 } ^ { T } \mathrm { T r } ( O _ { i } \hat { \rho } _ { t } ) } \end{array}$

Subsequent advances, including derandomized measurements [163] and measurement grouping [164], further reduce measurement costs by exploiting the structure of target observables.

Classical surrogates. Unlike classical shadows for a fixed quantum sta $^ \mathrm { { , e , } }$ classical surrogates aim to learn how a linear property $f _ { O } ( \rho ( { \pmb x } ) ) = \mathrm { T r } ( \rho ( { \pmb x } ) O )$ varies across a parametrized family of quantum states $\rho ( { \pmb x } )$ , where $\pmb { x } \in \mathbb { R } ^ { d }$ specifies circuit parameters or Hamiltonian couplings. The goal of a classical surrogate is to learn a predictive model $h ( { \pmb x } )$ that approximates the observable expectation value $f _ { O } ( { \pmb x } )$ across a parametrized family of quantum states, with prediction error $\begin{array} { r } { \mathbb { E } _ { \pmb { x } \sim \mathbb { D } } | f _ { O } ( \pmb { x } ) - h ( \pmb { x } ) | ^ { 2 } \leq \epsilon . } \end{array}$

A common machine-learning-based approach represents the target function using a prescribed family of basis functions $\{ \Phi _ { \omega } \} _ { \omega \in \Lambda }$ . The construction involves two steps. First, a training dataset $\mathcal T = \{ ( \hat { \mathbf { x } ^ { ( i ) } } , y ^ { ( i ) } ) \} _ { i = 1 } ^ { n }$ is collected, where each label $\boldsymbol y ^ { ( i ) }$ estimates $f _ { O } ( \pmb { x } ^ { ( i ) } )$ through quantum measurements. Second, a subset of basis functions $\Lambda ^ { \prime } \subseteq \Lambda$ is selected to construct the surrogate

$$
h ( \pmb { x } ) = \sum _ { \pmb { \omega } \in \Lambda ^ { \prime } } \hat { w } _ { \pmb { \omega } } \Phi _ { \pmb { \omega } } ( \pmb { x } ) ,
$$

where the coeficients $\hat { w } _ { \omega }$ are learned from $\tau$ through regression. Once trained, the surrogate predicts observable expectation values for unseen inputs using only classical computation, amortizing the quantum measurement cost across subsequent predictions. Under suitable structural assumptions, classical surrogates admit provable guarantees on sample complexity and prediction accuracy, potentially reducing the computational cost of repeated quantum evaluations [26, 29, 165].

Beyond these structured machine-learning approaches, recent studies have explored deep learning and large language models to develop more flexible surrogates for both qubit-based and continuous-variable quantum systems [166, 167].

Nonlinear properties of quantum states. Beyond linear properties, many physically relevant quantities depend nonlinearly on the quantum state $\rho .$ A representative class consists of degree-k polynomial functions of $\rho ,$ such as

$$
f _ { O } ( \rho ^ { k } ) = \mathrm { T r } \left[ O \rho ^ { k } \right] ,
$$

where O is a fixed observable and k is a positive integer. Such nonlinear functions arise naturally in the characterization of mixed states, with important examples including purity [168] and R´enyi entropies [169].

A key approach to estimating nonlinear properties is to reformulate them as linear-property estimation problems on multiple copies of the quantum state. Specifically, a homogeneous polynomial of degree k in the matrix elements of $\rho$ can be represented as a linear functional of the replicated state $\rho ^ { \otimes k }$ . This gives $f _ { O } ( \rho ^ { k } ) = \mathrm { T r } \left. A _ { k } ( O ) \rho ^ { \otimes k } \right.$ , where $\boldsymbol { \mathcal { A } } _ { k } ( \boldsymbol { O } )$ is an operator acting on $k$ copies of the original Hilbert space. For example, the purity of a quantum state is defined as $\operatorname { T r } ( { \bar { \rho } } ^ { 2 } )$ , which can be expressed as $\mathrm { T r } ( \rho ^ { 2 } ) = \mathrm { T r } ( \mathbb { S } \rho ^ { \otimes 2 } )$ , where $O = \mathbb { I }$ and $ { \mathcal { A } } _ { 2 } ( O )$ is the swap operator S. By writing $f _ { O } ( \rho ^ { k } )$ as a linear property in terms of $\dot { \rho } ^ { \otimes \dot { k } }$ , it could be characterized using classical shadow methods as

$$
\hat { f } _ { O } ( \rho ^ { k } ) = { \frac { ( T - k ) ! } { T ! } } \sum _ { \substack { t _ { 1 } \neq t _ { 2 } \cdots \neq t _ { k } \in [ T ] } } \operatorname { T r } \left[ { \cal A } ( O ) \hat { \rho } _ { t _ { 1 } } \otimes \cdot \cdot \cdot \otimes \hat { \rho } _ { t _ { k } } \right] .
$$

where $\{ \hat { \rho } _ { t _ { i } } \} _ { i = 1 } ^ { T }$ refers to $T$ independent snapshots of $\rho ,$ and $\mathbb { E } [ \hat { f } _ { O } ( \boldsymbol { \rho } ^ { k } ) ] = f _ { O } ( \boldsymbol { \rho } ^ { k } )$ . Beyond this basic approach, subsequent developments have introduced more sophisticated measurement protocols to improve the statistical eficiency of estimating $f _ { O } ( \rho ^ { k } )$ and reduce the measurement resources required for higher-order quantities [170].

Hamiltonian Learning. Hamiltonian learning aims to identify the interactions governing a quantum system from experimental observations, such as measurements of its time evolution [171]. Consider an N-qubit Hamiltonian $\begin{array} { r } { H = \sum _ { l = 1 } ^ { L } h _ { l } P _ { l } } \end{array}$ , where $P _ { l } \in \{ \mathbb { I } , X , Y , Z \} ^ { \otimes N } \setminus \{ \mathbb { I } ^ { \otimes N } \}$ are non-identity Pauli strings and $\pmb { h } = ( h _ { 1 } , \dots , h _ { L } ) \in \mathbb { R } ^ { L }$ denotes the unknown interaction coeficients. When the candidate Pauli terms $\{ P _ { l } \} _ { l = 1 } ^ { \breve { L } }$ are known, Hamiltonian learning reduces to estimating h from measurement data.

A common approach to Hamiltonian learning involves two stages. First, collecting time-dependent measurements for a set of initial-state and observable pairs $\big \{ ( \rho _ { j } , O _ { j } ) \big \} _ { j = 1 } ^ { M }$ . Second, inferring the Hamiltonian coeficients h from these measurements. For each pair, the initial state evolves as $\rho _ { j } ( t ) = e ^ { - \imath H t } \rho _ { j } e ^ { \imath H t }$ , and measurements of $O _ { j }$ yield estimates of $f _ { O _ { j } } ( \rho _ { j } ( t ) ) = \operatorname { T r } [ O _ { j } \bar { \rho } _ { j } ( t ) ]$ . The key insight is that the initial time derivative of $f _ { O _ { j } } ( \rho _ { j } ( t ) )$ depends linearly on the unknown coeficients h, enabling Hamiltonian learning through linear regression. Mathematically, we have

$$
d _ { j } : = \left. \frac { \mathrm { d } f _ { O _ { j } } ( \rho _ { j } ( t ) ) } { d t } \right| _ { t = 0 } = \imath \cdot \mathrm { T r } ( \rho _ { j } [ H , O _ { j } ] ) = \sum _ { l = 1 } ^ { L } K _ { j l } h _ { l } ,
$$

where $K _ { j l } = \iota \cdot \mathrm { T r } ( \rho _ { j } [ P _ { l } , O _ { j } ] )$ defines the entries of a matrix $\pmb { K } \in \mathbb { R } ^ { M \times L }$ determined by the known input states, observables, and candidate Pauli strings. This relation reduces Hamiltonian learning to a linear inverse problem [172]. Specifically, by measuring $f _ { O _ { j } } ( \rho _ { j } ( t ) )$ at suficiently short evolution times and fitting its time dependence, one can estimate the initial derivatives $\hat { \pmb { d } } = ( \hat { d } _ { 1 } , \dots , \hat { d } _ { M } )$ , which approximately satisfy $\hat { d } \approx d = K h$ . The Hamiltonian coeficients can then be inferred by solving a regression problem, such as least-squares estimation, i.e.,

$$
\hat { \pmb { h } } = \arg \operatorname* { m i n } _ { { \pmb { h } } \in \mathbb { R } ^ { L } } \| { \pmb { K } } { \pmb { h } } - \hat { { \pmb { d } } } \| _ { 2 } ^ { 2 } .
$$

Quantum error mitigation. The objective of quantum error mitigation (QEM) is to estimate ideal observable expectation values $f _ { O } ( \rho )$ in Eq. (A1) using measurement data collected from noisy quantum devices [64]. In practice, hardware noise leads to a noisy implementation $\tilde { \mathcal { U } } ,$ yielding $f _ { O } ( \tilde { \rho } ) = \mathrm { T r } [ O \tilde { \mathcal { U } } ( \rho _ { 0 } ) ]$ , which generally deviates from $f _ { O } ( \rho )$ QEM seeks to reduce this discrepancy through suitably designed noisy experiments and classical post-processing.

A representative QEM approach is zero-noise extrapolation (ZNE) [173], which estimates $f _ { O } ( \rho )$ by measuring a quantum circuit at multiple noise levels and extrapolating to the zero-noise limit. Let $\lambda \geq 0$ denote the efective noise strength and $f _ { O } ( \tilde { \rho } _ { \lambda } ) = \hat { \mathrm { T r } } [ O \tilde { \rho } _ { \lambda } ]$ , where $\tilde { \rho } _ { \lambda } = \tilde { \mathcal { U } } _ { \lambda } ( \rho _ { 0 } )$ and $\tilde { \mathcal { U } } _ { 0 } = \mathcal { U }$ corresponds to the ideal circuit. ZNE first collects estimates $\{ \hat { f } _ { O } ( \tilde { \rho } _ { \lambda _ { j } } ) \} _ { j = 1 } ^ { M }$ at diferent noise levels, typically generated using techniques such as gate folding or pulse stretching. It then extrapolates these estimates toward $\lambda = 0$ . In particular, Richardson extrapolation constructs the estimator as

$$
\hat { f } _ { O } ^ { \mathrm { Z N E } } ( \tilde { \rho } _ { 0 } ) = \sum _ { j = 1 } ^ { M } \pmb { c } _ { j } \cdot \hat { f } _ { O } ( \tilde { \rho } _ { \lambda _ { j } } ) ,
$$

where the extrapolation coeficients $\{ \boldsymbol { c } _ { j } \} _ { j = 1 } ^ { M }$ satisfy $\textstyle \sum _ { j = 1 } ^ { M } c _ { j } = 1$ and $\begin{array} { r } { \sum _ { j = 1 } ^ { M } \pmb { c } _ { j } \lambda _ { j } ^ { k } = 0 } \end{array}$ for $k \in [ M - 1 ]$ . These constraints cancel the leading M−1 orders of noise contributions, reducing the extrapolation bias at the potential cost of amplified statistical errors. Beyond ZNE, various other QEM approaches have been developed, including probabilistic error cancellation [174], symmetry verification [175], and virtual distillation [176], each employing diferent strategies to suppress or compensate for the efects of quantum noise.

Quantum process characterization. This task aims to infer the properties of an unknown quantum process from measurements of its input-output behavior. Mathematically, a quantum process is described by a completely positive and trace-preserving $\left( \mathrm { C P T P } \right)$ map $\mathcal { E } ,$ which transforms an input quantum state $\rho$ into an output state $\mathcal { E } ( \rho )$ Depending on the characterization objective, one may seek to reconstruct the entire quantum channel or predict selected properties of its output states.

Quantum process tomography. Quantum process tomography aims to reconstruct an unknown quantum channel E from measurements of its outputs for a set of known input states. A convenient representation of the channel is its Choi matrix, defined as

$$
J _ { \mathcal { E } } = \sum _ { a , b = 1 } ^ { d } \left| a \right. \left. b \right| \otimes \mathcal { E } ( \left| a \right. \left. b \right| ) ,
$$

where d denotes the dimension of the input Hilbert space. For a CPTP map, the Choi matrix satisfies the physica constraints $J _ { \mathcal { E } } ~ \succeq ~ \mathbf { 0 }$ and $\operatorname { T r } _ { \mathrm { o u t } } ( J _ { \mathcal { E } } ) = \mathbb { I } _ { d } ^ { \top }$ These properties allow quantum process tomography to be formulated as a constrained estimation problem. Specifically, for a known input state $\rho _ { s }$ and an output measurement efect $M _ { b } .$ , the probability of obtaining measurement outcome b is given by $\mathbb { P } ( b | s ) \stackrel { } { = } \operatorname { T r } [ M _ { b } \pmb { \mathcal { E } } ( \rho _ { s } ) ] \stackrel { } { = } \operatorname { T r } [ ( \rho _ { s } ^ { \top } \otimes M _ { b } ) J _ { \pmb { \mathcal { E } } } ]$ . By preparing diferent input states and performing measurements on their corresponding outputs, one obtains a collection of measurement statistics that constrain the unknown Choi matrix. The channel can then be reconstructed by fitting a physical Choi matrix to the observed data, for example through maximum-likelihood estimation [177].

Prediction of output properties. Diferent from full process tomography, output-property prediction focuses on estimating selected observable expectation values for previously unseen input states. In particular, for an unknown quantum channel E and a target observable O, the prediction task is to learn the function

$$
f _ { \varepsilon , O } ( \rho ) = \mathrm { T r } [ O \mathcal { E } ( \rho ) ] = \mathrm { T r } [ \mathcal { E } ^ { \dagger } ( O ) \rho ] ,
$$

where $\mathcal { E } ^ { \dagger }$ denotes the adjoint channel. This relation indicates that predicting the expectation value of O at the output is equivalent to estimating the expectation value of the efective observable ${ \bar { \mathcal { E } } } ^ { \dagger } ( O )$ at the input. Therefore, instead of reconstructing the entire channel, one can construct a classical predictive model by learning an approximation of this efective observable.

A representative approach collects training data consisting of known input states and estimates of the corresponding output expectation values [178]. The efective observable can then be approximated using a selected operator basis $\{ P _ { j } \}$ as $\begin{array} { r } { \hat { O } _ { \mathrm { e f f } } = \sum _ { j } { w _ { j } P _ { j } } } \end{array}$ , where the coeficients $\{ w _ { j } \}$ are learned from the training data. For a new input state $\rho ,$ the resulting predictor takes the form

$$
\hat { f } _ { \mathcal { E , O } } ( \rho ) = \mathrm { T r } ( \hat { O } _ { \mathrm { e f f } } \rho ) = \sum _ { j } { \pmb { w } } _ { j } \mathrm { T r } ( P _ { j } \rho ) .
$$

Once the coeficients are learned, the predictor can estimate the target observable expectation value without additional access to the unknown channel, provided that the required input-state features are available.

## 4. AI agents and harnesses

Large language models (LLMs) have evolved from conversational assistants into autonomous agents capable of executing complex tasks, ranging from software engineering to scientific research. Here, we introduce three foundations of AI scientists relevant to our work, including LLM agents for software engineering, scientific-reproduction agents, and automated harness design and agent self-improvement.

LLM agents for software engineering. Unlike conventional LLMs that generate responses to individual prompts, LLM agents can execute complex tasks through iterative reasoning and tool interactions. Software engineering has become a prominent application and evaluation setting for such agents. Specifically, an LLM agent operates within a software framework, known as a harness, that manages interactions between the LLM and its environment. At each step, the harness provides the model with task instructions and execution history, receives a proposed action, executes it through available tools, and returns the resulting feedback. This process continues until the task is completed or a predefined resource limit is reached, with inference costs determined by the tokens consumed across model calls.

A representative benchmark for LLM agents in software engineering is SWE-bench [42], which evaluates agents on resolving real-world software issues, with correctness assessed through repository tests. Building on this setting, SWE-agent [43] demonstrated that specialized tools for code navigation, search, and editing can substantially improve agent performance without changing the underlying LLM. These findings underpin the importance of harness design alongside model capabilities.

Subsequent research has explored diferent design philosophies, ranging from minimal harnesses built around a single shell tool to adaptive agents that construct their own tools during execution [77, 78]. In practice, harnesses can be developed by model providers for their own models, such as DeepSeek Harness (DSH), or by third parties to support diferent models through application programming interfaces (APIs).

Scientific-reproduction agents. Beyond software engineering, LLM agents are increasingly being developed as AI scientists to automate scientific research. One important application is scientific reproduction, which aims to reconstruct computational procedures and reproduce results reported in published papers, thereby facilitating the verification and reuse of scientific findings. Previous research has advanced along two complementary directions, namely developing benchmarks to evaluate reproduction capabilities and designing agents to automate reproduction workflows.

Existing benchmarks difer in their task objectives and the amount of prior information provided to agents. Typically, benchmark development involves defining reproduction objectives, constructing tasks and reference materials, and establishing evaluation criteria to assess the performance of diferent agents. For example, PaperBench [179] evaluates the reproduction of machine-learning research from published papers, supplemented by author-provided clarifications, using expert-developed grading rubrics. PRBench [180] extends paper-based reproduction to physics, providing expert-curated task instructions that specify target results and implementation requirements. In contrast, CORE-Bench [181] evaluates agents on executing and interpreting released research code and data, while AutoExperiment [84] examines their ability to reconstruct missing components of existing implementations. Other benchmarks investigate computational reproduction in materials science and the social sciences, using domain-specific research materials and evaluation protocols [51, 85].

Complementing these benchmarks, research on reproduction agents focuses on developing computational workflows for recovering published results. Depending on the available research materials and task specifications, agents may be required to implement methods from scientific papers, execute existing research code, or reconstruct incomplete implementations. A common design is to coordinate multiple stages of scientific reproduction, including interpreting research materials, planning computational procedures, implementing or adapting code, executing experiments, and validating the resulting outputs. For example, DeepCode [76] proposes an agentic framework for generating executable code from research papers and technical specifications, whereas InternAgent [75] develops an agent-based workflow intended to support scientific understanding, implementation, and experimental execution. Related eforts, such as Paper2Agent [62], transform scientific papers and associated research artifacts into interactive agents that expose reusable computational capabilities.

Automated harness design and agent self-improvement. The performance of LLM agents depends not only on the underlying language model but also on how the agent is organized and how it learns from previous executions. This has motivated research on two complementary approaches to automatically improving agent capabilities: optimizing the harness that governs agent execution and accumulating reusable experience from completed tasks.

The first approach, automated harness design, aims to improve agent performance by optimizing its prompts, tools, and control flow. Existing methods employ LLM-based search to generate candidate harnesses, evaluate their performance on benchmark tasks, and iteratively refine their designs [79, 80, 182]. The resulting harness can then be deployed to execute subsequent tasks. This approach primarily improves the agent’s execution framework, with the optimized design typically remaining fixed during deployment.

The second approach focuses on improving agents through experience accumulated during task execution. Instead of modifying model parameters or harness code, experience-based methods extract useful information from previous runs, such as reflections on failures, successful problem-solving strategies, reusable workflows, and optimized instructions [183–186]. This information is stored as external memory and retrieved to guide subsequent executions, enabling agents to adapt their behavior without retraining the underlying model. In this work, we adopt this experience-based perspective of self-improvement, emphasizing the accumulation and reuse of validated knowledge across scientific reproduction tasks.

## 5. Literature review

AI scientists are increasingly being explored to automate scientific reasoning, computational implementation, and validation. Three research directions are particularly relevant to our work: autonomous quantum algorithm design, autonomous investigation of quantum systems, and quantum research reproduction. These directions pursue distinct scientific objectives: designing improved computational methods, discovering or explaining physical phenomena, and independently verifying published findings, respectively. Below, we review representative advances in each direction and highlight their diferences from our study.

Autonomous quantum computing design. Autonomous quantum computing design aims to discover and optimize quantum algorithms, circuit architectures, and error-correcting codes under computational and hardware constraints. A central challenge is the vast design space of quantum circuits, programs, and error-correcting codes. Existing approaches leverage AI models to guide candidate generation with such domain-specific reasoning, while computational evaluation provides feedback for iterative refinement.

For algorithm construction, LLMs have been trained to generate programs representing families of quantum circuits [55], while iterative search has been applied to quantum machine learning architectures [56, 57], quantum state preparation [187], and molecular ground-state energy estimation [61]. Quantum resource optimization instead improves existing computational procedures by improving Hamiltonian simulation accuracy [188] and reducing gate and qubit counts in quantum arithmetic subroutines [90]. In quantum error correction, AI-guided search has been used to discover quantum error-correcting codes [52–54].

Autonomous investigation of quantum systems. Autonomous investigation of quantum systems aims to advance scientific understanding through two complementary tasks: predicting physical behaviour from specified models and inferring explanations for observed phenomena. For the former, AI agents integrate established numerical methods with computational validation to investigate quantum phase transitions [189] and dynamical evolution [60]. For the latter, agents propose candidate physical models and computationally assess their consistency with observations. This capability has been explored through controlled rediscovery of known physical models [190] and applied to interpreting experimental optical measurements in quantum materials [191], where human scientists guided the investigation and collected measurements, while AI agents developed and tested candidate explanations.

Benchmark for quantum science. As AI scientists become increasingly capable of conducting computational research, systematic benchmarks are needed to assess their scientific reasoning, implementation, and validation capabilities. Existing benchmarks evaluate AI agents on tailored research questions derived from scientific papers [192], simulated quantum engineering tasks [193], and the reproduction of computational results reported in physics [180] and materials science [51]. Automated scientific checks have also been introduced to assess computational correctness and provide feedback for code refinement [194]. Quantum programming benchmarks further assess whether LLMs can translate task descriptions into functionally correct programs across diferent software packages [195–198]. These eforts establish structured evaluation environments for measuring the capabilities of AI scientists.

Our work difers from the three prior research directions by advancing autonomous paper-to-result reproduction in computational quantum science. Although PRBench [180] benchmarks end-to-end paper reproduction across broader areas of physics, its reliance on expert-curated task instructions specifying implementation requirements and expected outputs limits its ability to assess whether AI scientists can independently reconstruct research procedures from original papers. Our objective is to develop an AI scientist capable of reproducing published results directly from scientific papers and target figures with minimal human intervention. Importantly, our investigation reveals that ambiguities and even errors in published research can substantially hinder autonomous reproduction. Beyond benchmarking, the proposed RA systematically investigates the causes of reproduction failures and distils validated corrections into reusable knowledge, enabling continual improvement across research tasks.

## SI B: Implementation details of OmniQ

This section provides additional implementation details of OmniQ and an analysis of the challenges encountered during scientific reproduction. In SI B 1, we first describe the workflow for selecting and organizing research papers. Next, in SI B 2, we present the construction of the computational quantum science dataset for evaluating paper-to-result reproduction. Finally, in SI B 3, we examine representative reproduction issues identified during our experiments.

## 1. Workflow of OmniQ for selecting and organizing research papers

To ensure the reliability and consistency of the reproduction benchmark, OmniQ employs a seven-step workflow that integrates multi-agent cross-validation with expert verification, as illustrated in Fig. SI 1. The workflow begins with task construction in Steps 1–3, where a domain expert selects a research paper and its target figure, an LLM agent drafts the task specification and evaluation criteria, and the expert subsequently reviews and refines both. Next, Step 4 involves initial benchmarking, in which multiple agent configurations independently execute each reproduction task, with each configuration evaluated once. In Step 5, their evaluation results are cross-checked by an auditing agent to identify potentially disputed judgments and assess their consistency with the original paper, task specification, and evaluation criteria. Based on this audit, domain experts review the flagged papers in Step 6, classify confirmed reproduction issues into three levels (i.e., Levels I–III, as defined in the main text) according to their severity, and determine necessary revisions to the task specification or evaluation criteria. Finally, in Step 7, the afected tasks are re-evaluated or re-executed using the revised specifications and criteria, yielding a more reliable benchmark for assessing scientific reproduction capabilities.

![](images/74f7c2e6297f463d2ecdbdef9e80e74a3a2c2da828cbdc64c8be2630820c8269.jpg)  
FIG. SI 1. Workflow of OmniQ. The workflow consists of seven steps involving domain experts, agents, and benchmark evaluation, distinguished by diferent colors. Arrows indicate the interactions and progression between steps.

Beyond the overall workflow, three key mechanisms underpin the construction and reliability of the OmniQ benchmark, i.e., task specification and evaluation criteria, automated judging, and cross-agent auditing with expert verification. We detail these mechanisms below.

Task specification and evaluation criteria. Steps 1–3 of OmniQ establish each reproduction task through agentassisted construction and expert review. As illustrated in Fig. 1, each task comprises two Markdown documents with distinct purposes: a preliminary document that guides the AI scientist and an evaluation criteria document reserved for the automated judge. This separation allows the reproduction objectives to be specified explicitly while keeping the expected results inaccessible to the AI scientist.

The preliminary document specifies the target figure and provides both general guidelines and paper-specific instructions. The general guidelines, shared across all tasks, require the AI scientist to follow the methods and settings reported in the original paper, prohibit fabricated or externally obtained results, and require all approximations to be documented. The paper-specific instructions define the target problem, experimental scope, and relevant conventions, while leaving the reconstruction of computational procedures and numerical results to the AI scientist.

The evaluation criteria document defines the requirements for assessing reproduction correctness. These include the structural properties of the target figure, quantitative checkpoints with reference values and tolerances, and verification of the experimental parameters and essential computational procedures. Both documents are initially drafted by an LLM agent and subsequently reviewed by domain experts to ensure consistency with the original paper. In particular, expert review addresses potentially misleading criteria, such as purely visual assessments, numerical tolerances that are overly restrictive relative to computational variability, or reference values incorrectly extracted from published figures. Importantly, the evaluation criteria are made available to the automated judge only after the reproduction is completed, ensuring that the AI scientist cannot access the reference values during execution.

Automated judging. To assess scientific reproduction beyond visual similarity, Step 4 of OmniQ employs an automated judge to verify both the reproduced results and their underlying computational procedures. Implemented using Claude Code with Opus 4.8, the judge examines the complete execution workspace, including source code, execution logs, numerical data, and experimental reports, against the predefined evaluation criteria. This evidencebased assessment examines not only whether the reproduced figure matches the published results but also whether those results are supported by the prescribed computational methods and experimental settings.

The judge classifies identified issues into two categories according to their impact on scientific correctness. In particular, critical issues include quantitative results outside the specified tolerances, deviations from essential methodologica steps, inconsistent experimental parameters, and numerical values that cannot be traced to the executed computations. Every quantitative checkpoint is treated as a mandatory requirement, and documenting an approximation does not exempt the reproduction from satisfying the corresponding criteria. In contrast, non-critical issues are limited to visual presentation and do not afect scientific validity.

A reproduction is considered successful only when no critical issues are identified. For tasks involving multiple target figures, each figure is evaluated independently, and the task is considered successfully reproduced only if all target figures satisfy their respective criteria. This protocol prioritizes the fidelity of scientific results and computational procedures over superficial agreement with published figures.

Cross-agent verification and expert review. To further establish the reliability of scientific reproduction assessments, Steps 5–7 of OmniQ introduce a cross-agent verification mechanism to systematically examine potential inconsistencies in the original papers, task specifications, and evaluation criteria. Specifically, this mechanism leverages evaluation results from multiple independent agent configurations to identify potential issues in the benchmark itself.

Across the 60 selected research papers, a verification agent examines 3,376 judging outcomes collected from mul tiple agent configurations and target figures, focusing on systematic failures where diferent configurations, including the strongest-performing ones, fail to satisfy the same evaluation criteria. This process identifies 19 papers for further investigation. For each flagged paper, the verification agent systematically compares the original publication, task specification, evaluation criteria, and computational results across configurations. Where possible, it also independently checks key numerical values using analytical calculations, authors’ released code, or data extracted from published figures.

Domain experts subsequently review these findings, classify identified issues into three levels according to their severity, and determine whether revisions to the task specifications or evaluation criteria are warranted. Expert review confirms issues in 18 of the 19 flagged papers. Depending on their underlying causes, revisions may align task specifications with settings recovered from the authors’ released code or data, restrict computational requirements to feasible resource budgets, or adjust evaluation checkpoints that depend on unstated or idealized conventions. Note that tasks whose evaluation outcomes are unafected by the identified issues are retained without substantive changes. Finally, afected tasks are re-evaluated when the criteria are revised or re-executed when the task specifications are modified.

## 2. Construction of computational quantum science dataset

To systematically evaluate the capabilities of AI scientists in computational quantum science, we construct a dataset comprising 60 research papers and their corresponding figure-level reproduction tasks. The selected papers span three major research areas with relatively balanced representation: classical simulation of quantum systems (20 papers), quantum algorithms (19 papers), and quantum property characterization (21 papers). This selection captures diverse computational paradigms and methodological challenges in computational quantum science. We prioritize influential studies published in leading journals while requiring that their numerical results be reproducible within a computational budget of approximately one day per task. This constraint largely excludes computationally intensive deep learning-based approaches. An overview of the dataset is provided in Table SI 1.

For each paper, we select representative figures that capture its central scientific findings or methodological contributions. Reproducing these figures requires translating scientific descriptions into executable workflows, implementing the underlying methods, and validating the numerical results within OmniQ. By requiring AI scientists to reconstruct computational results from the underlying scientific principles, this design provides a systematic assessment of their scientific reasoning and implementation capabilities. In the following, we describe the dataset composition and representative reproduction tasks across the three research areas.

Classical simulation of quantum systems. To evaluate the agent’s ability to reproduce research-level results in classical quantum simulation, we select papers based on two complementary criteria. The first prioritizes state-of-theart numerical methods that introduce novel algorithmic ideas, require technically demanding implementations, or play an important role in assessing claims of quantum utility. The second emphasizes physics-informed methods, where successful reproduction requires understanding the underlying quantum systems and how their physical properties, such as symmetries, locality, and entanglement structure, are exploited to enable eficient numerical simulation. These criteria enable us to systematically assess the AI scientist’s ability to understand and faithfully reproduce sophisticated classical simulation methods.

The selected papers provide balanced coverage of representative classical quantum simulation methods discussed in SI A 1, spanning tensor networks, neural quantum states, Cliford simulation and Pauli propagation, free-fermion simulation, quantum Monte Carlo, and open-system simulation. More specifically, tensor-network methods cover diverse representations and computational tasks, ranging from quantum many-body dynamics and open-system evo lution to Feynman-diagram calculations [14, 199–201]. Neural quantum states explore learning-based representations for simulating quantum systems and characterizing their physical properties [202–204]. Cliford simulation and Paul propagation encompass complementary strategies for reducing the computational cost of quantum-circuit and manybody simulations [100–102, 205, 206]. Free-fermion methods leverage fermionic structure to investigate measurementinduced dynamics and non-stabilizerness [207, 208]. Quantum Monte Carlo covers stochastic methods for entanglement characterization and noisy quantum-circuit simulation [209, 210], while open-system simulation addresses dissipative dynamics, finite-temperature efects, and nonequilibrium steady states [211–214].

Quantum algorithms. Our selection in this category covers both advances in quantum algorithm design and meth ods for reducing their quantum resource requirements. Together, these studies help assess the practical utility of quantum computing on both near-term and fault-tolerant devices. In this context, AI scientists capable of independently reproducing these results could accelerate the discovery of novel quantum algorithms and advance their practical applications.

The selected papers cover the five categories of quantum algorithms introduced in SI A 2. Concretely, non-variationa quantum simulation includes product formulas for Hamiltonian evolution, along with adiabatic evolution, imaginarytime evolution and spectral filtering for ground-state preparation [116, 117, 119, 121, 122]. Quantum phase estimation papers study the circuit depth and query cost required to estimate phases and amplitudes to a given precision [126, 128, 129]. For variational quantum simulation, the papers cover ground-state preparation, combinatorial optimization and quantum dynamics, including adaptive ansatz construction, tensor-network initialization and circuit knitting [135, 136, 138, 139]. Work on quantum machine learning examines the trainability and predictive performance of quantum neura networks and kernel methods, including comparisons with classical models and the efects of data and measurement resources [68, 69, 215–218]. The quantum-enhanced arkov chain Monte Carlo study compares Markov chain Monte Carlo convergence under quantum and classical proposal methods [156].

Property characterization of quantum systems. The selection of papers on quantum property characterization is guided by two complementary criteria. The first emphasizes measurement-eficient protocols for characterizing individual quantum systems, enabling the estimation of important physical properties with favorable scaling in measurement cost under suitable assumptions. The second focuses on learning unknown quantum systems and processes, including methods that extract predictive models from experimental data and generalize across diferent states, dynamics, or system configurations. These capabilities are essential for advancing quantum hardware development, evaluating quantum algorithms, and deepening our understanding of quantum physics. Accordingly, reproducing these results provides a meaningful test of the AI scientist’s ability to implement and validate computational methods with direct relevance to quantum research.

The selected papers provide balanced coverage of five representative research topics discussed in SI A 3. In particular, linear-property estimation spans classical-shadow protocols, measurement optimization, and learningbased prediction of quantum observables, covering both theoretical foundations and practical estimation strategies [26, 162, 164, 219–224]. Nonlinear-property estimation focuses on measurement-eficient characterization of entanglement and related quantities using randomized measurements, replica-based estimators, and permutation moments [170, 225–227]. Hamiltonian learning addresses the inference of unknown Hamiltonian parameters from quantum dynamics or thermal-state information [172, 228]. Quantum error mitigation investigates computational strategies for recovering ideal observables from noisy quantum measurements [63, 229]. Quantum process characterization encompasses the estimation of gate-set properties, identification of conservation laws, and prediction of quantum-process observables [178, 230, 231].

TABLE SI 1: Collection of representative papers in computational quantum science. Journal names are abbreviated for conciseness. Public code availability is classified as Y (Yes), a publicly available implementation; P (Partial), an implementation covering only a subset of the reported numerical targets or computational workflow; and N (No), no public implementation identified, including code available only upon request. For papers with publicly accessible implementations, the code availability entries provide direct hyperlinks to the corresponding repositories or software archives.
<table><tr><td colspan="6">Classical simulation of quantum systems</td></tr><tr><td>Subcategory</td><td>Paper ID</td><td>Paper title</td><td>Journal</td><td>Fig.</td><td>Code</td></tr><tr><td>Tensor network</td><td>#2</td><td>Efficient tensor network simulation of IBM's Eagle kicked Ising experiment [14]</td><td>PRX Quantum</td><td>3</td><td>N</td></tr><tr><td></td><td>#7</td><td>Matrix product states and first quantization [199]</td><td>PRL</td><td>4</td><td>N</td></tr><tr><td></td><td>#8</td><td>Low-rank quantics tensor train representations of Feynman diagrams for multiorbital electron-phonon models [200]</td><td>PRL</td><td>1</td><td>N</td></tr><tr><td></td><td>#14</td><td>Stable iPEPO tensor-network algorithm for dynamics of two-dimensional open quantum lattice models [201]</td><td>PRX</td><td>5</td><td>N</td></tr><tr><td>Neural quantum state</td><td>#16</td><td>Predicting topological entanglement entropy in a Rydberg analog simulator [204]</td><td>Nat. Phys.</td><td>6</td><td>Y</td></tr><tr><td></td><td>#17</td><td>Approximately-symmetric neural networks for quantum spin liquids [202]</td><td>PRL</td><td>3</td><td>Y</td></tr><tr><td></td><td>#18</td><td>Autoregressive Transformer Neural Network for Simulating Open Quantum Systems via a Probabilistic Formulation [203]</td><td>PRL</td><td>3</td><td>Y</td></tr><tr><td>Pauli propagation</td><td>#1</td><td>Simulation of quantum circuits by low-rank stabilizer decompositions [205]</td><td>Quantum</td><td>4</td><td>N</td></tr><tr><td></td><td>#4</td><td>Simulation of Qubit Quantum Circuits via Pauli Propagation [100] PRA</td><td></td><td>6</td><td>P</td></tr><tr><td></td><td>#5</td><td>Classically estimating observables of noiseless quantum circuits [102]</td><td>PRL</td><td>2</td><td>P</td></tr><tr><td></td><td>#6</td><td>Fast and converged classical simulations of evidence for the utility of quantum computing before fault tolerance [101</td><td>Sci. Adv.</td><td>5A</td><td>N</td></tr><tr><td></td><td>#11</td><td>Clifford Circuits Augmented Time-Dependent Variational Principle [206]</td><td>PRL</td><td>3</td><td>N</td></tr><tr><td>Free fermion simulation</td><td>#3</td><td>Theory of free fermions under random projective measurements [207]</td><td>PRX</td><td>3</td><td>N</td></tr><tr><td></td><td>#20</td><td>The non-stabilizerness of fermionic Gaussian states [208]</td><td>Quantum</td><td>4</td><td>N</td></tr><tr><td>Quantum Monte Carlo</td><td>#9</td><td>Real-time Sign-Problem-Suppressed Quantum Monte Carlo Algorithm For Noisy Quantum Circuit Simulations [209] Bipartite reweight-annealing algorithm of quantum Monte Carlo</td><td>PRL</td><td>2</td><td>Y</td></tr><tr><td></td><td>#10</td><td>to extract large-scale data of entanglement entropy and its derivative [210]</td><td>Nat. Commun.</td><td>3</td><td>N</td></tr><tr><td>Simulation for open quantum system</td><td>#12</td><td>Large-scale stochastic simulation of open quantum systems [211] Efficient simulation of finite-temperature open quantum</td><td>Nat. Commun.</td><td>7</td><td>Y</td></tr><tr><td></td><td>#13</td><td>systems [212] Steady States of Infinite-Size Dissipative Quantum Chains via</td><td>PRL</td><td>2</td><td>P</td></tr><tr><td></td><td>#15</td><td>Imaginary Time Evolution [213]</td><td>PRL</td><td>1</td><td>N</td></tr><tr><td></td><td>#19</td><td>Efficient simulation of non-trivial dissipative spin chains via stochastic unraveling [214]</td><td>PRX Quantum</td><td>6</td><td>N</td></tr><tr><td colspan="6">Quantum algorithms</td></tr><tr><td>Subcategory</td><td>Paper ID</td><td>Paper title</td><td>Journal</td><td>Fig.</td><td>Code</td></tr><tr><td>Non-variational quantum</td><td>#23</td><td>Efficient and practical Hamiltonian simulation from time-dependent product formulas [116]</td><td>Nat. Commun.</td><td></td><td></td></tr><tr><td>simulation</td><td>#31</td><td>Faster Algorithmic Quantum and Classical Simulations by Corrected Product Formulas [117]</td><td>PRX Quantum</td><td>2, 3</td><td>Y</td></tr><tr><td></td><td>#34</td><td>Digital adiabatic evolution is universally accurate [119]</td><td>Nat. Commun.</td><td>3</td><td>Y</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>#30</td><td>Determining eigenstates and thermal states on a quantum computer using quantum imaginary time evolution [121] Enhancing Scalability of Quantum Eigenvalue Transformation of</td><td>Nat. Phys.</td><td>3</td><td>Y</td></tr><tr><td></td><td>#33</td><td>Unitary Matrices for Ground State Preparation through Adaptive Finer Filtering [122]</td><td>Quantum</td><td>6</td><td>P</td></tr><tr><td>Quantum phase estimation</td><td>#32</td><td>On low-depth algorithms for quantum phase estimation [126]</td><td>Quantum</td><td>5,6</td><td>N</td></tr><tr><td></td><td>#35</td><td>Low depth algorithms for quantum amplitude estimation [128]</td><td>Quantum</td><td>5, 6</td><td>N</td></tr><tr><td></td><td>#36</td><td>Bayesian Quantum Amplitude Estimation [129]</td><td>Quantum</td><td>4, 5, 6</td><td>Y</td></tr><tr><td>Variational quantum simulation</td><td>#28</td><td>An adaptive variational algorithm for exact molecular simulations on a quantum computer [135]</td><td>Nat. Commun.</td><td>2</td><td>P</td></tr><tr><td></td><td>#29</td><td>Synergistic pretraining of parametrized quantum circuits via tensor networks [136]</td><td>Nat. Commun.</td><td>2</td><td>P</td></tr><tr><td></td><td>#22</td><td>The Quantum Approximate Optimization Algorithm and the Sherrington-Kirkpatrick Model at Infinite Size [138]</td><td>Quantum</td><td>1</td><td>P</td></tr><tr><td></td><td>#24</td><td>Overhead-constrained circuit knitting for variational quantum dynamics [139]</td><td>Quantum</td><td>4</td><td>Y</td></tr><tr><td>Quantum machine learning</td><td>#25</td><td>Theory of overparametrization in quantum neural networks [215]</td><td>Nat. Comput. Sci.</td><td>3,5</td><td>Y</td></tr><tr><td></td><td>#38</td><td>Cost Function Dependent Barren Plateaus in Shallow</td><td>Nat. Commun.</td><td>6</td><td>N</td></tr><tr><td></td><td>#27</td><td>Parametrized Quantum Circuits [216] Barren plateaus preclude learning scramblers [217]</td><td>PRL</td><td>2</td><td>N</td></tr><tr><td></td><td>#26</td><td>Classical surrogates for quantum learning models [218]</td><td>PRL</td><td>4</td><td>N</td></tr><tr><td></td><td>#21</td><td>Power of data in quantum machine learning [69]</td><td>Nat. Commun.</td><td>4</td><td>P</td></tr><tr><td></td><td>#60</td><td>Exponential concentration in quantum kernel methods [68]</td><td>Nat. Commun.</td><td>6</td><td>P</td></tr><tr><td>Quantum-enhanced Markov chain</td><td>#37</td><td>Quantum-enhanced Markov chain Monte Carlo [156]</td><td></td><td></td><td>Y</td></tr><tr><td colspan="6">Quantum property characterization</td></tr><tr><td>Subcategory</td><td>Paper ID</td><td>Paper title</td><td>Journal</td><td>Fig.</td><td>Code</td></tr><tr><td>Linear state properties</td><td>#39</td><td>Information-theoretic bounds on quantum advantage in machine learning [222]</td><td>PRL</td><td>3</td><td>N</td></tr><tr><td></td><td></td><td>Predicting Many Properties of a Quantum System from Very Few Measurements [162]</td><td>Nat. Phys.</td><td>3</td><td>Y</td></tr><tr><td></td><td>#40 #41</td><td>Inference-Based Quantum Sensing [224]</td><td>PRL</td><td>2,3</td><td>N</td></tr><tr><td></td><td>#44</td><td>Shallow shadows: Expectation estimation using low-depth random Clifford circuits [219]</td><td>PRL</td><td>3</td><td>Y</td></tr><tr><td></td><td></td><td>Efficient Learning for Linear Properties of Bounded-Gate</td><td>Nat. Commun.</td><td></td><td>P</td></tr><tr><td></td><td>#47</td><td>Quantum Circuits [26] Improved machine learning algorithm for predicting ground state</td><td></td><td>2,3</td><td></td></tr><tr><td></td><td>#48</td><td>properties [223] Guaranteed efficient energy estimation of quantum many-body</td><td>Nat. Commun.</td><td>2</td><td>Y</td></tr><tr><td></td><td>#49</td><td>Hamiltonians using ShadowGrouping [164] Optimizing quantum measurements by partitioning multisets of</td><td>Nat. Commun.</td><td>3</td><td>Y</td></tr><tr><td></td><td>#50</td><td>observables [220] Measuring Arbitrary Physical Properties in Analog Quantum</td><td>PRL</td><td>3</td><td>N</td></tr><tr><td></td><td>#55</td><td>Simulation [221]</td><td>PRX</td><td>6</td><td>N</td></tr><tr><td>Nonlinear state properties</td><td>#51</td><td>Importance sampling of randomized measurements for probing entanglement [225] Auxiliary-Free Replica Shadows: Efficient Estimation of Multiple</td><td>PRL</td><td>2</td><td>Y</td></tr><tr><td></td><td>#52</td><td>Nonlinear Quantities [170] Detecting entanglement in quantum many-body systems via</td><td>PRL</td><td>2</td><td>N</td></tr><tr><td></td><td>#53</td><td>permutation moments [226] Many-body entropies and entanglement from polynomially-many</td><td>PRL</td><td>3</td><td>N</td></tr><tr><td></td><td>#54</td><td>local measurements [227]</td><td>PRX</td><td>4</td><td>N</td></tr><tr><td>Hamiltonian learning</td><td>#43</td><td>Practical Hamiltonian learning with unitary dynamics and Gibbs states [172]</td><td>Nat. Commun.</td><td>3,4,5</td><td>P</td></tr><tr><td></td><td>#56</td><td>Ansatz-free Hamiltonian learning with Heisenberg-limited scaling [228]</td><td>PRX Quantum</td><td>2</td><td>N</td></tr><tr><td>Quantum error mitigation</td><td>#46</td><td>Machine learning for practical quantum error mitigation [63]</td><td>Nat. Mach. Intell.</td><td>3</td><td>Y</td></tr><tr><td></td><td>#59</td><td>Quantum Error Mitigated Classical Shadows [229]</td><td>PRX Quantum</td><td>3</td><td>N</td></tr><tr><td>Quantum processes characterization</td><td>#42</td><td>Quantum circuit complexity and unsupervised machine learning of topological order [66]</td><td>Nat. Commun.</td><td>2,3</td><td>Y</td></tr><tr><td></td><td>#45</td><td>Shadow estimation of gate-set properties from random</td><td>Nat. Commun.</td><td>2</td><td>N</td></tr><tr><td></td><td>#57</td><td>sequences [230] Learning conservation laws in unknown quantum dynamics [231]</td><td>PRX Quantum</td><td>2</td><td>N</td></tr><tr><td></td><td>#58</td><td>Learning to predict arbitrary quantum processes [178]</td><td>PRX Quantum</td><td>2,3</td><td>Y</td></tr></table>

## 3. Analysis of paper-to-result reproduction issues

In this subsection, we provide a detailed analysis of the reproduction issues identified through AI-assisted cross-agent verification and independently confirmed by domain experts. Specifically, our examination of the initial reproduction results revealed issues afecting 18 of the 60 selected papers, including inconsistencies between published methodolog ical descriptions and released implementations, incomplete experimental specifications, and discrepancies between reported numerical results and theoretical constraints. These issues span all three areas of computational quantum science considered in this study and are classified into three levels according to their severity and resolvability, as introduced in the main text, i.e.,

• Level I (16 papers): Recoverable methodological details or discrepancies in task specifications and evaluation criteria that can be resolved using the available scientific record or expert interpretation.

• Level II (1 paper): Localized inconsistencies in the published study that prevent faithful reproduction of the reported results but can be resolved through expert inspection of the available evidence.

• Level III (1 paper): Fundamental inconsistencies between the reported results, methodological descriptions, and released code that remain unresolved even after expert inspection, preventing faithful reproduction from the available scientific record.

A summary of the major issues of these 18 research papers is provided in Table SI 2.

In the following, we examine representative studies at each level to illustrate the corresponding failure modes. For Level I, we select three representative papers, one from each of the three research areas, while for Levels II and III, we examine the sole paper in each category. For each paper, we first introduce the scientific background and then follow the steps of the curation workflow described in SI. B 1 to illustrate how AI scientists and domain experts collaborate to identify, verify, and resolve potential reproduction issues.

## a. Level I (Paper #46): Hyperparameter inconsistency in quantum error mitigation

Quantum error mitigation (QEM) plays an important role in improving the accuracy of quantum computations on noisy quantum devices, yet achieving accurate mitigation with limited experimental overhead remains challenging [64]. Paper #46 [63], published in Nature Machine Intelligence in 2024, investigated machine-learning-based QEM (ML QEM) to improve mitigation accuracy and computational eficiency, with particular emphasis on generalization beyond the training regime. Here, we examine the reproduction of its random-forest (RF) results, where an inconsistency between the hyperparameter settings described in the paper and those used in the released implementation led to substantial reproduction discrepancies.

Algorithmic background. Recall from SI. A 3 that QEM aims to recover the ideal expectation value $f _ { O } ( \rho ) =$ $\mathrm { T r } ( O U \rho _ { 0 } U ^ { \dagger } )$ from noisy measurements, which yield $f _ { O } ( { \tilde { \rho } } ) = \mathrm { T r } [ O { \widetilde { \mathcal { U } } } ( \rho _ { 0 } ) ]$ with $\widetilde { \mathcal { U } }$ representing the corresponding noisy quantum channel. ML-QEM approaches this task by learning a classical predictive model from training data $\mathcal { T } = \{ \bar { ( } z ^ { ( i ) } , y ^ { ( i ) } ) \} _ { i = 1 } ^ { n }$ , where $z ^ { ( i ) } \in \mathbb { R } ^ { d }$ encodes noisy measurement outcomes and circuit-related features, while $\boldsymbol y ^ { ( i ) }$ denotes the corresponding ideal expectation value. As shown in Fig. SI 2a, given the dataset T, RF constructs an ensemble of B decision trees and averages their predictions to obtain the mitigated expectation value,

$$
\hat { f } _ { O } ^ { \mathrm { R F } } ( z ) = \frac { 1 } { B } \sum _ { b = 1 } ^ { B } T _ { b } ( z ) ,
$$

where $T _ { b }$ denotes the prediction of the b-th decision tree. Each tree recursively partitions the feature space by selecting a feature and a splitting threshold that minimize the prediction error. At each split, only a subset of candidate features $S \subseteq [ d ]$ is considered. The hyperparameter max features controls the number of these candidate features |S|, thereby influencing the construction of the decision trees and their predictive performance.

Task construction with OmniQ. The domain expert selected Fig. 3 of the original paper as the reproduction target. Fig. SI 2c shows its upper-left panel, which compares the mitigation errors of random forest (RF), digital zero-noise extrapolation (ZNE), and unmitigated measurements for four-qubit Trotterized Ising dynamics under incoherent noise, with the number of Trotter steps spanning both interpolation and extrapolation regimes. The target reproduction figure was selected for two reasons. First, it requires reproducing distinct computational procedures, including noisycircuit simulation, RF model training, and noise-scaled circuit simulations for ZNE. Second, it directly examines the paper’s central claim that ML-QEM can achieve superior mitigation accuracy and generalize to circuits deeper than those used for training.

TABLE SI 2. Papers with an issue confirmed in the curation audit. Paper numbers are those of Table SI 1, and levels follow the three-level scale of SI B 3. The last column shows how we handled each issue. Re-judge means that the criteria were revised and the afected runs re-judged. Re-run means that the task specification was corrected and the afected tasks re-run. Keep means that no verdict depended on the issue, so that neither a re-judgement nor a re-run was needed. Removed means that the paper was excluded from the benchmark.
<table><tr><td>Level</td><td>Paper</td><td>Issue</td><td>Handling</td></tr><tr><td>I</td><td>#39 [222]</td><td>The reported results reflect an idealized setting rather than the actual execution of the algorithm described in the paper</td><td>Re-judge</td></tr><tr><td></td><td>#40 [162]</td><td>Underspecified experimental details in the paper and overly restrictive figure-based evaluation criteria</td><td>Re-judge</td></tr><tr><td></td><td>#26 [218]</td><td>The initial criteria require the classical model to have at least ten times lower error than the quantum models, a requirement not explicitly stated in the paper, and incorrectly marks quantum-model results with errors below the prescribed range as failures.</td><td>Re-judge</td></tr><tr><td></td><td>#9 [209]</td><td>The excited-state convention is reversed relative to the standard qubit convention</td><td>Re-judge</td></tr><tr><td></td><td>#14 [201]</td><td>Hamiltonian bond-counting convention differs from the standard interpretation</td><td>Re-judge</td></tr><tr><td></td><td>#51 [225]</td><td>Underspecified experimental details in the paper and overly restrictive figure-based evaluation criteria</td><td>Re-judge</td></tr><tr><td></td><td>#52 [170]</td><td>Unstated experimental setting about the number of measurements</td><td>Re-judge</td></tr><tr><td></td><td>#28 [135]</td><td>The initial criteria set an incorrect energy target for BeH2.</td><td>Re-judge</td></tr><tr><td></td><td>#38 [216]</td><td>The paper leaves details of the optimization procedure unspecified, but the initial criteria require the resulting measurement counts to fall within fixed ranges and mark results that correctly reproduce the growth trend as failures.</td><td>Re-judge</td></tr><tr><td></td><td>#34 [119]</td><td>The paper specifies the truncation order as  $K = \lfloor 4 \log T \rfloor$  without defining the logarithm base, while the released code uses  $K = 4 \lfloor \log _ { 1 0 } T \rfloor$  , giving different truncation orders.</td><td>Re-judge</td></tr><tr><td></td><td>#21 [69]</td><td>The paper omits the input rescaling needed to reproduce the reported quantum kernel classification performance and gives conflicting rules for generating class labels.</td><td>Re-run</td></tr><tr><td></td><td>#10 [210]</td><td>Incorrect lattice dimensions in the preliminary task specification</td><td>Re-run</td></tr><tr><td></td><td>#33 [122]</td><td>The paper confuses the ground-state overlap with its square, and the initial task specification recommends a random state that does not reliably reproduce the reported speedup.</td><td>Re-run</td></tr><tr><td></td><td>#46 [63]</td><td>Inconsistency of hyperparameter settings between the paper&#x27;s claim and the released code</td><td>Re-run</td></tr><tr><td></td><td>#17 [202]</td><td>Incorrect entropy units in the preliminary task specification</td><td>Keep</td></tr><tr><td></td><td>#50 [220]</td><td>Unspecified detailed about the random number generator</td><td>Keep</td></tr><tr><td>II</td><td>#42 [66]</td><td>Analysis admits several readings, and the published values need two unstated steps</td><td>Re-run</td></tr><tr><td>III</td><td>#60 [68]</td><td>One figure plots the variance of the distance term C in the kernel  $\kappa ^ { \mathrm { P Q } } = e ^ { - \gamma C }$  but incorrectly labels it as kernel variance. The released code and documented settings are insufficient to regenerate the underlying data.</td><td>Removed</td></tr></table>

Based on the methodology described in the paper, an agent drafted the initial task specification (Preliminary.md) and evaluation criteria (Criteria.md), as illustrated in Fig. SI 2b. The former specified the RF hyperparameter max features=1, consistent with the paper’s statement that only one feature is considered when selecting the best split. The latter specified quantitative evaluation criteria requiring RF to outperform ZNE in both interpolation and extrapolation regimes. However, some experimental details, such as the initial quantum state $\rho _ { 0 }$ , were not fully specified in the original paper. To make the task executable, Preliminary.md required a product state but allowed the reproducing agent to choose the specific state and document it in report.md. Since the mitigation errors depend on the choice of initial state, the evaluation criteria in Criteria.md focused on reproducing the relative performance of the mitigation methods instead of requiring strict numerical agreement with the published error values.

Initial benchmark. Fig. SI 3a presents the initial reproduction results of three representative AI scientists selected from the 23 agents evaluated on this task, which are DeepCode with DeepSeek V4 Pro, DSH with DeepSeek V4 Pro, and Claude Code with Opus 5.5. Both DeepCode and DSH followed the single-feature instruction by setting max features=1. Their reproduced results deviated from the published findings in the extrapolation regime. That is, DeepCode largely lost the reported RF advantage over ZNE at larger circuit depths, as its mitigation error approaches the unmitigated baseline, while the RF and ZNE errors obtained by DSH became nearly indistinguishable at the largest displayed depths.

![](images/2e7262e2fdcfe598549c753a34b29726022d200e84c87e852344f5880cd8a1f5.jpg)  
FIG. SI 2. Overview of the reproduction task in Paper #46 for ML-based quantum error mitigation. a. Schematic of random-forest regression for quantum error mitigation, where the predictions of multiple decision trees are averaged to esti mate ideal observable expectation values from noisy measurement features. At each split of a decision tree, the hyperparameter max features controls the number of candidate features considered when selecting the optimal split. b. Description of the max features hyperparameter settings in the original paper, initial task specification (Preliminary.md), and released implementation, together with the evaluation criteria (Criteria.md) for successful reproduction. c. The upper-left panel of original Figure 3 of Paper #46 [63], showing four-qubit Trotterized Ising dynamics with incoherent noise and no readout error. The red, orange, and blue curves represent the errors obtained using random-forest (RF) mitigation, zero-noise extrapolation (ZNE), and unmitigated measurements, respectively. The vertical dashed line at Trotter step 14 separates interpolation from extrapolation regimes.

Diferent from DeepCode with DeepSeek V4 Pro and DSH with DeepSeek V4 Pro, Claude Code with Opus 5.5 independently recognized the distinction between the integer 1 and floating-point 1.0 settings, tested the prescribed single-feature configuration separately, and used max features=1.0 in its main reproduction. Consequently, its results verified the RF advantage in the extrapolation regime. The initial judge marked this run as successful because it classified the documented deviation from the prescribed setting as non-critical. These contrasting outcomes revealed a potential inconsistency in the task specification and motivated a closer examination of the published methodology and released implementation.

Cross-agent verification of reproduction inconsistencies. To investigate the discrepancy, the verification AI scientist cross-checked the original paper, task specification, evaluation criteria, reproduction records, and authors released RF implementation. As illustrated in Fig. SI 2b, this examination revealed an inconsistency in the RF hyperparameter max features. Specifically, the paper and initial task specification required a single candidate feature at each split, whereas the released implementation used the scikit-learn default, which considers all available features. In scikit-learn, the integer setting max features=1 corresponds to |S| = 1, while the floating-point setting max features=1.0 corresponds to |S| = d. The released RF notebook instantiated the regressor without explicitl specifying max features, thereby adopting the all-feature configuration.

Domain experts independently reviewed this evidence and confirmed that the task specification should follow the released implementation by explicitly requiring max features=1.0. They further determined that the discrepancy did not invalidate the paper’s central scientific claim, since the underlying RF method remained unchanged and the reported extrapolation advantage could be recovered using the authors’ implementation. The issue was therefore classified as Level I, as the available source code provided suficient evidence for a reliable correction.

Final benchmark. Following expert verification, the task specification was revised to require max features=1.0, and the afected reproduction tasks were rerun and evaluated under the corrected specification. Fig. SI 3b presents the resulting reproduction outcomes. All three agents recovered the qualitative ordering reported in the original paper, with RF outperforming ZNE throughout the displayed extrapolation regime. In particular, DeepCode recovered the RF advantage that it had failed to reproduce in its initial run, DSH showed a clearer separation between RF and ZNE at larger circuit depths, and Claude Code with Opus 5.5 maintained the performance ordering observed in its initial reproduction. These results demonstrate that correcting the task specification enables recovery of the reported qualitative mitigation advantage of ML methods.

![](images/559221fe68f0811759481d3d3f227bef4f7b4930ef90f6af43edf2138018807a.jpg)  
FIG. SI 3. Reproduction results for Paper #46 on ML-based quantum error mitigation. a. Initial benchmark results, with one candidate feature per RF split prescribed in the preliminary (max features=1). b. Final benchmark results after expert revision of the preliminary to specify all features (max features=1.0). Columns show DeepCode+DeepSeek V4 Pro, DSH+DeepSeek V4 Pro, and Claude Opus 5.5, from left to right. Each plot compares RF, ZNE, and unmitigated measurements under incoherent noise without readout error, corresponding to the original Figure 3 of Paper #46 (as shown in Fig. SI 2c). Vertical dashed lines at Trotter step 14 separate interpolation from extrapolation regimes. The curve colors representing diferent methods in each panel may vary across agents, as each agent independently selects its plotting colors The corresponding methods are identified in the legend of each panel.

Beyond recovering the qualitative advantage of RF, the DeepCode rerun with DeepSeek V4 Pro also produced mitigation errors whose magnitudes more closely matched those reported in the original figure. The domain experts further examined the report.md and found that it employed a diferent initial quantum state from that used in its initial run, despite the initial-state specification in Preliminary.md remaining unchanged. This observation indicates that the improved numerical agreement cannot be attributed solely to the correction of max features. More broadly, it highlights how diferent implementations of underspecified experimental settings can afect numerical reproducibility, even when the central scientific claim is successfully recovered.

## b. Level I (Paper #14): Unclear Hamiltonian specification in dynamics simulation

Understanding the dynamics of interacting open quantum systems in the thermodynamic limit is crucial for exploring nonequilibrium phases and dissipative critical phenomena [232–234], yet accurately simulating such dynamics remains challenging, particularly in higher dimensions. Paper #14 [201], published in Physical Review X in 2021, developed an infinite projected entangled pair operator (iPEPO) algorithm that incorporates optimal bond truncation techniques, enabling accurate simulations of correlated open-system dynamics in regimes beyond the reach of existing methods. Here, we examine the reproduction of its Figure 5 benchmark, where an unclear specification of the system Hamiltonian led to substantial discrepancies between the reproduced and reported results.

![](images/aa34996c85dede580132f341be2e3f39be49d3c86686b6b23847aa88f421ef1e.jpg)  
FIG. SI 4. Overview of the reproduction task in Paper #14 for iPEPO 2D-dynamics simulation. a. Schematic of the dissipative transverse-field Ising model on an infinite two-dimensional square lattice, with nearest-neighbor Ising interactions $V / z ,$ a transverse magnetic field $h _ { x }$ , and local Lindblad jump operators $L _ { j }$ describing dissipation. b. Parameter settings reported in the original paper and specified in the initial reproduction instructions (Preliminary.md), together with the evaluation criteria (Criteria.md). c. Original Figure 5 of Paper #14, comparing the simple update (SU) and full-environment truncation with weighted trace gauge (FET+WTG) algorithms at diferent iPEPO bond dimensions D. The upper panel shows the trace distance $T _ { 2 } ( t )$ between the simulated and exact nearest-neighbor reduced density matrices, with the inset showing its dependence on D at $t \gamma = 1 0$ . The lower panels show the time evolution of the local magnetization $m ^ { x } ( t )$ and nearest-neighbor correlation $S _ { 1 2 } ^ { x x } ( t )$ , respectively, with black dashed curves indicating the exact solutions.

Physical model background. As reviewed in SI. A 1, the evolution of a Markovian open quantum system is governed by the Lindblad master equation $\begin{array} { r } { \frac { \partial \rho _ { t } } { \partial t } = \mathcal { L } ( \rho _ { t } ) = - \imath [ H , \rho _ { t } ] + \sum _ { j } ( L _ { j } \rho _ { t } L _ { j } ^ { \dag } - \frac { 1 } { 2 } \{ L _ { j } ^ { \dag } L _ { j } , \rho _ { t } \} ) } \end{array}$ , where H governs the coherent dynamics and $L _ { j }$ are jump operators describing the coupling to the environment. As Fig. SI 4a shows, Paper #14 considers a dissipative transverse-field Ising model on an infinite two-dimensional square lattice. Its Hamiltonian and local jump operators are given by

$$
H = \frac { V } { z } \sum _ { \langle j , l \rangle } \sigma _ { j } ^ { z } \sigma _ { l } ^ { z } + \frac { h _ { x } } { 2 } \sum _ { j } \sigma _ { j } ^ { x } , \quad w i t h \ L _ { j } = \frac { \sqrt { \gamma } } { 2 } ( \sigma _ { j } ^ { y } - \imath \sigma _ { j } ^ { z } ) .
$$

Here, $\boldsymbol { \sigma } _ { j } ^ { x , y , z }$ are Pauli operators, $\langle j , l \rangle$ denotes nearest-neighbor pairs, and $z = 4$ is the lattice coordination number. The parameters $V , h _ { x }$ , and γ control the interaction strength, transverse field, and dissipation rate, respectively. In particular, the regime $h _ { x } = 0$ admits an exact solution for local observables, providing a benchmark for evaluating the accuracy of the iPEPO simulations.

Task construction with OmniQ. The domain expert selected Fig. 5 of the original paper as the reproduction target. As shown in Fig. SI 4c, this reproduction task requires the agent to correctly implement iPEPO evolution using diferent update strategies, including full-environment truncation with weighted trace gauge (FET+WTG) and simple update (SU), and benchmark their predictions against exact solutions for the trace distance $\langle T _ { 2 } ( t ) \rangle$ magnetization $\langle m ^ { x } ( t ) \rangle$ ⟩, and nearest-neighbor spin correlations $\left. S _ { 1 2 } ^ { x x } ( t ) \right.$ . This task captures the paper’s core technical contributions and enables a direct and rigorous assessment of reproduction accuracy against exact solutions.

When submitted this task to OmniQ, the agent generates the initial task specification (Preliminary.md) and evaluation criteria (Criteria.md), as shown in Fig. SI 4b. In particular, following the description in the original paper, the agent specified the model parameters as $z = 4 , V / \gamma = 1 . 2$ , and $h _ { x } = 0$ in Preliminary.md to ensure that the reproduction was conducted in the same moderate-damping regime. The evaluation criteria in Criteria.md also required these parameter settings to be matched exactly for the reproduction to be considered successful.

Initial benchmark. Fig. SI 5i and Fig. SI 5ii present the initial reproduction results of InternAgent with DeepSeek V4 Pro and Grok Build Harness with Grok 4.6 as representative examples among the 23 agents evaluated on this task, where all reproduced curves deviate from the published results at late times $( \gamma t = 1 0 )$ . Notably, both InternAgent and

t°  
m<sup>x</sup>(b) (t)  
t°  
![](images/d9991dc222fc47ccd5a8c69d7c708d0ce1f999b79163019d95dbcf58a6cc809d.jpg)

(initial) Grok Build Harness+Grok 4.6  
![](images/ff9d4d030fe8bc4354234950aaf1142ee5c67cb95df6693ed84afb73affce278.jpg)

(Final) Claude Code+Opus 5.5  
![](images/a7db5bf56175a58370ba379a97f3316b8418ccf713efa22fdf1058ed273e930d.jpg)

(b) $m ^ { x } ( t )$  
![](images/ba024142d777cba28944ba218903aeb0d26434f222bc3d106490178c743643a6.jpg)  
t°

(c) S<sup>xx</sup><sub>12</sub> (t)  
![](images/5a002c7a56d7bfa162f5fcc1e8be71e8dbc847eb01f5fcefcc9b971bc494f7aa.jpg)  
t°

![](images/d3acc5fbe11dd6fe7a0f55404f391f0adc181847dfd30512a70b26d449f5bf62.jpg)  
t°

(c) S<sup>xx</sup><sub>12</sub> (t)  
![](images/0f3597731cd1943fe44c6251984bb8bdf28ea2c57e6813031fd6c0b525b75cc8.jpg)

m<sup>x</sup>(b) (t)  
(c) S<sup>xx</sup><sub>12</sub> (t)  
![](images/4000c46350ae2bf766c23c31d14b15fd4fb2d7d3fb0f72169c50f5db763e1239.jpg)

![](images/421fa582834c23651aa41b41894a3fdd41a047515bb7424f5958b447d13c960d.jpg)  
t°  
FIG. SI 5. Initial and corrected reproduction results for Paper #14 on iPEPO 2D-dynamics simulation. i. Initial reproduction results of InterAgent obtained by following the reported Hamiltonian parameters and counting each undirected nearest-neighbor bond once. ii. Initial reproduction results of Grok Build Harness obtained by following the reported Hamiltonian parameters and counting each undirected nearest-neighbor bond once. iii. Final reproduction results of Claude Code after expert revision to count each nearest-neighbor bond twice. In each group, panel (a) shows the trace distance $T _ { 2 } ( t )$ between the iPEPO and exact reduced density matrices, comparing SU (dotted lines) and FET+WTG (solid lines) for bond dimensions $D = 3 , \ldots , 9$ . The inset shows $T _ { 2 } ( 1 0 )$ as a function of D. Panels (b) and (c) show the time evolution of the local magnetization $m ^ { x } ( t )$ and nearest-neighbor correlation $S _ { 1 2 } ^ { x x } ( t )$ , respectively, with black dotted curves indicating the exact trajectories. The gray horizontal lines in panel (b) indicate the analytically determined steady-state magnetizations, approximately −0.549 and −0.347 for the initial and corrected reproductions, respectively. After correcting the bond-counting convention, the exact trajectories and the performance of the iPEPO algorithms become consistent with the results reported in the original paper.

Grok Build Harness obtained nearly identical values of approximately −0.549 for the exact magnetization $\langle m ^ { x } ( t ) \rangle$ , as shown in panels i(b) and ii(b). Both agents independently obtained nearly identical exact trajectories that difered substantially from the published results. This agreement suggests that their exact simulations were correctly implemented, motivating the domain expert to investigate whether ambiguities in the definition of the model Hamiltonian could explain the discrepancy.

Cross-agent verification of reproduction inconsistencies. The domain expert independently derived the exact dynamics of the magnetization $m ^ { x } ( t )$ and compared the resulting trajectory with the reproduced and original results to investigate the discrepancy. Following the Heisenberg Lindblad evolution $\partial _ { t } \langle O \rangle = \langle \mathcal { L } ^ { \dagger } ( O ) \rangle$ , the domain expert found that the evolution of magnetization $m ^ { x } ( t ) = \langle \sigma _ { j } ^ { \bar { x } } \rangle$ can be fully described in the subspace span by $\{ \langle \sigma _ { j } ^ { y } \sigma _ { a } ^ { z } \rangle , \langle \sigma _ { j } ^ { x } \sigma _ { a } ^ { z } \sigma _ { b } ^ { z } \rangle , \langle \sigma _ { j } ^ { y } \sigma _ { a } ^ { z } \sigma _ { b } ^ { z } \sigma _ { c } ^ { z } \rangle , \langle \sigma _ { j } ^ { x } \sigma _ { a } ^ { z } \sigma _ { b } ^ { z } \sigma _ { c } ^ { z } \sigma _ { d } ^ { z } \rangle \}$ where $a , b , c , d$ are denoted as the four distinct nearest neighbors of lattice site $j .$ . In particular, for the initial product state $\rho _ { 0 } = | \uparrow \rangle \langle \uparrow | ^ { \otimes N }$ along z-axis and counting each undirected nearest-neighbor bond once, their dynamics obeys

$$
\partial _ { t } \langle \sigma _ { j } ^ { x } \rangle = - \gamma ( 1 + \langle \sigma _ { j } ^ { x } \rangle ) - \frac { 8 V } { z } \langle \sigma _ { j } ^ { y } \sigma _ { a } ^ { z } \rangle ,
$$

$$
\partial _ { t } \langle \sigma _ { j } ^ { y } \sigma _ { a } ^ { z } \rangle = - \gamma \langle \sigma _ { j } ^ { y } \sigma _ { a } ^ { z } \rangle + \frac { 2 V } { z } \big ( \langle \sigma _ { j } ^ { x } \rangle + 3 \langle \sigma _ { j } ^ { x } \sigma _ { a } ^ { z } \sigma _ { b } ^ { z } \rangle \big ) ,
$$

$$
\partial _ { t } \langle \sigma _ { j } ^ { x } \sigma _ { a } ^ { z } \sigma _ { b } ^ { z } \rangle = - 2 \gamma \langle \sigma _ { j } ^ { x } \sigma _ { a } ^ { z } \sigma _ { b } ^ { z } \rangle - \gamma e ^ { - \gamma t } - \frac { 4 V } { z } \big ( \langle \sigma _ { j } ^ { y } \sigma _ { a } ^ { z } \rangle + \langle \sigma _ { j } ^ { y } \sigma _ { a } ^ { z } \sigma _ { b } ^ { z } \sigma _ { c } ^ { z } \rangle \big ) ,
$$

$$
\partial _ { t } \langle \sigma _ { j } ^ { y } \sigma _ { a } ^ { z } \sigma _ { b } ^ { z } \sigma _ { c } ^ { z } \rangle = - 2 \gamma \langle \sigma _ { j } ^ { y } \sigma _ { a } ^ { z } \sigma _ { b } ^ { z } \sigma _ { c } ^ { z } \rangle \frac { 2 V } { z } \big ( 3 \langle \sigma _ { j } ^ { x } \sigma _ { a } ^ { z } \sigma _ { b } ^ { z } \rangle + \langle \sigma _ { j } ^ { x } \sigma _ { a } ^ { z } \sigma _ { b } ^ { z } \sigma _ { c } ^ { z } \sigma _ { d } ^ { z } \rangle \big ) ,
$$

$$
\partial _ { t } \langle \sigma _ { j } ^ { x } \sigma _ { a } ^ { z } \sigma _ { b } ^ { z } \sigma _ { c } ^ { z } \sigma _ { d } ^ { z } \rangle = - 3 \gamma \langle \sigma _ { j } ^ { x } \sigma _ { a } ^ { z } \sigma _ { b } ^ { z } \sigma _ { c } ^ { z } \sigma _ { d } ^ { z } \rangle - \frac { 8 V } { z } \langle \sigma _ { j } ^ { y } \sigma _ { a } ^ { z } \sigma _ { b } ^ { z } \sigma _ { c } ^ { z } \rangle - \gamma e ^ { - 2 \gamma t } .
$$

Starting at all zero expectation values when $t = 0 , m ^ { x } ( t )$ achieves the steady-state limit when $t \to \infty$ and all time derivatives equal to zero, which yields the exact analytical expression

$$
m ^ { x } ( \infty ) = - \frac { 3 + 6 2 \left( \frac { V } { z \gamma } \right) ^ { 2 } + 9 6 \left( \frac { V } { z \gamma } \right) ^ { 4 } } { 3 + 1 1 0 \left( \frac { V } { z \gamma } \right) ^ { 2 } + 5 1 2 \left( \frac { V } { z \gamma } \right) ^ { 4 } } .
$$

Following the convention above, this formula gives $m ^ { x } ( \infty ) = - 0 . 5 4 8 9 2 3$ when the parameters are $V / \gamma = 1 . 2$ and $z = 4$ as reported in Paper #14. In contrast, counting each bond twice with efectively $V / \gamma = 2 . 4$ yields $m ^ { x } ( \infty ) =$ −0.346579. As shown in Fig. SI 4c(b), Fig. SI 5i(b) and Fig. SI 5ii(b), the original and initially reproduced trajectories approach approximately −0.347 and −0.549, respectively. These values agree with the analytical predictions under the double- and single-counting conventions, respectively. This suggests that the original simulations efectively counted each nearest-neighbor bond twice, revealing a factor-of-two inconsistency between the reported Hamiltonian under its conventional single-counting interpretation and the numerical results. This issue can therefore be corrected by specifying the Hamiltonian parameters from an exact calculation and is classified as Level I.

Final benchmark. To resolve this ambiguity, the domain expert revised the task specification to explicitly require each nearest-neighbor bond to be counted twice in the Hamiltonian. As shown in Fig. SI 5iii-(b), the corrected exact trajectories implemented by Claude Code with Opus 5.5 agree with the independent analytical predictions, and the iPEPO simulations from all three reproduction runs closely match the results reported in the original paper. This confirms that the initial reproduction failures were mainly caused by ambiguity in the Hamiltonian bond-counting convention rather than errors in the implementation of the iPEPO algorithms.

## c. Level $I ~ ( P a p e r \# 3 8 ) .$ : Inconsistent measurement-cost criteria in the training of variational quantum algorithms

Variational quantum algorithms (VQAs) can sufer from barren plateaus, where gradient variance decreases exponentially with the number of qubits [149]. Paper #38 [216], published in Nature Communications in 2021, studied when barren plateaus arise for global and local cost functions at diferent circuit depths. The authors also simulated quantum autoencoder training [235], claiming that increasing circuit depth slows training and raises the number of measurement shots needed to reach a given cost threshold under a local cost function. Here, we leverage OmniQ to reproduce this experiment, revealing an inconsistency between the acceptance criteria based on absolute shot counts and the optimization choices permitted by the task specification.

Algorithmic background. A quantum autoencoder uses a variational circuit $V ( \pmb \theta )$ to compress an ensemble of quantum states $\{ ( p _ { \mu } , | \psi _ { \mu } \rangle ) \} _ { \mu } ,$ where $p _ { \mu }$ is the probability of state $| \psi _ { \mu } \rangle$ and $\begin{array} { r } { \sum _ { \mu } p _ { \mu } = \overset { \cdot } { 1 } } \end{array}$ . As shown in ${ \mathrm { F i g } } .$ SI 6a, the circuit retains the encoded information in subsystem A while bringing subsystem B close to a fixed reference state so that B can be discarded. Training minimizes a cost that measures how closely B approaches this reference.

In Paper #38, the reference state is $| 0 \rangle ^ { \otimes n _ { B } }$ , and the local cost is

$$
C _ { L } ^ { \prime } ( \pmb { \theta } ) = 1 - \frac { 1 } { n _ { B } } \sum _ { j = 1 } ^ { n _ { B } } \mathrm { T r } \big [ \big ( | 0 \rangle \langle 0 | _ { j } \otimes \mathbb { I } _ { \bar { j } } \big ) \mathrm { T r } _ { A } [ V ( \pmb { \theta } ) \rho V ( \pmb { \theta } ) ^ { \dag } ] \big ] ,\tag{B1}
$$

where $\begin{array} { r } { \rho = \sum _ { \mu } p _ { \mu } | \psi _ { \mu } \rangle \langle \psi _ { \mu } | } \end{array}$ is the ensemble-average input state, and $\mathbb { I } _ { \bar { j } }$ is the identity on all qubits in B except j. The simulations use two input states, $| \psi _ { 1 } \rangle = | 0 \rangle _ { A } | 0 \rangle _ { B } ^ { \otimes n _ { B } }$ and $| \psi _ { 2 } \rangle = | 1 \rangle _ { A } | 1 1 0 \cdot \cdot \cdot 0 \rangle _ { B }$ , with probabilities $p _ { 1 } = 2 / 3$ and $p _ { 2 } = 1 / 3$ . The authors fix $n _ { A } = 1$ and $n _ { B } = 1 0$ and vary the number of circuit layers over $L = 2 , 4 , \dots , 2 0$ using a brickwork ansatz of $R _ { y }$ rotations and nearest-neighbor controlled-Z gates as shown in Fig. SI 6a. The circuit parameters θ are optimized using gradient-free random-subspace search. Mathematically, at iteration t, a random isometry $A ( t ) \in \mathbb { R } ^ { \hat { P } \times d }$ defines a d = 10-dimensional search subspace of the P circuit parameters. An inner optimizer then solves the following d-dimensional optimization problem

$$
\pmb { s } ^ { * } = \arg \operatorname* { m i n } _ { \pmb { s } \in \mathbb { R } ^ { d } } C _ { L } ^ { \prime } \Big ( \pmb { \theta } ( t ) + A ( t ) \pmb { s } \Big ) .\tag{B2}
$$

The parameters are then updated as $\pmb \theta ( t + 1 ) = \pmb \theta ( t ) + A ( t ) \pmb s ^ { * }$ . The number of shots used for each evaluation of the cost function starts at 10 and increases by a factor of $3 / 2$ whenever the optimization reaches a plateau, up to a maximum of $1 0 ^ { 5 }$

Task construction with OmniQ. The domain expert selected Fig. 6 of Paper $\# 3 8$ as the reproduction target to examine how circuit depth afects training convergence and measurement cost. As shown in Fig. SI 6c, the main panel of the original figure presents cost trajectories during optimization for diferent circuit depths, while the inset plots the cumulative number of measurement shots required to reach a target cost of 0.02 or 0.05 as a function of the circuit depth L. The reproduction task aims to recover the slower convergence and approximately exponential growth in measurement cost as circuit depth L increases. Reproducing the inset also requires accounting for all shots used during the inner optimizations in Eq. (B2).

a  
![](images/9f353759b0b9cc1c4ce1fbd105d9854c8a582f956403d073dce389a847069d5f.jpg)

![](images/4969b2ad6ca99c55c3273b31ecd0a5b5c6f1c409c109c0778e82213a338b4b71.jpg)  
FIG. SI 6. Overview of the reproduction task in Paper #38 for quantum autoencoder training. a. Schematic of the quantum autoencoder, which compresses the input quantum information into subsystem A while bringing subsystem B close to the reference state |0⟩<sup>⊗nB</sup> . The expanded circuit shows the ansatz used in the numerical simulation, with initial RY rotations followed by L layers of alternating nearest-neighbor controlled-Z gates and RY rotations. b. Optimization settings described in the original paper and task specification (Preliminary.md), together with the initial and revised evaluation criteria (Criteria.md). c. Original Figure 6 of Paper #38 [216]. The main panel shows the local cost during training for diferent circuit depths. The inset shows the cumulative number of shots needed to reach target costs 0.02 and 0.05 as a function of circuit depth L.

With OmniQ, an agent prepared Preliminary.md based on the published input ensemble, circuit architecture, cost function, random-subspace search, and shot schedule, and drafted the corresponding Criteria.md, as summarized in Fig. SI 6b. Since the original paper did not specify the solver for the inner optimization problem in Eq. (B2), the task allowed reproducing agents to choose the solver, convergence tolerances, and evaluation budget, provided that these choices were fixed and documented. The plateau-detection rule and shot-allocation convention were also left to the agents, but the shots used in every sampled cost evaluation had to be included in the cumulative count.

The initial Criteria.md defined numerical checkpoints for the training curves and their expected ordering across circuit depths. For the inset, each shot-count curve was required to fall within at least three of four absolute acceptance windows at L ≈ 2, 10, 15, 20, for example $[ 1 0 ^ { 9 } , 2 \times 1 0 ^ { 1 0 } ]$ for the target cost 0.02 at L = 20. The criteria further required $S _ { 0 . 0 2 } ( L ) \geq S _ { 0 . 0 5 } ( L )$ and at least a hundredfold increase in each series from L = 2 to L = 20.

Initial benchmark. The three representative runs reproduced the reported depth-dependent training behavior but failed the absolute shot-count criteria. As shown in Figs. SI 7a–c, Claude Code with Opus 5, DeepCode with DeepSeek V4 Pro, and HGM with DeepSeek V4 Pro all obtained decreasing local costs, with convergence slowing as circuit depth increased. All three satisfied the requirements for threshold ordering and growth in cumulative shots with depth. However, their shot counts fell below the acceptance windows specified in the initial Criteria.md at al four depth checkpoints, as indicated by the pale bars in the lower panels. At $L = 2 0$ , all three runs required fewer than $5 \times 1 0 ^ { 8 }$ cumulative shots to reach ${ \cal C } _ { L } ^ { \prime } \le 0 . 0 2 .$ compared with the required range $[ 1 0 ^ { 9 } , 2 \times 1 0 ^ { 1 0 } ]$ . The initial judge classified these discrepancies as critical, resulting in three failed reproduction assessments. This pattern suggested that the absolute shot-count requirements might be incompatible with the optimization choices permitted by the task specification.

Cross-agent verification of reproduction inconsistencies. The verification AI scientist examined whether the absolute shot-count requirements were justified by the published method and task specification. The paper’s Methods section described the random-subspace search and shot schedule but did not specify the inner solver, convergence tolerances, or evaluation budget. As shown in Fig. SI 6b, the task permitted reproducing agents to select and document these settings. The reproduction records showed that all three agents used the Nelder–Mead method for the inner optimization and included the shots from every sampled cost evaluation in the cumulative count. Claude Code set a limit of 400 cost evaluations per random-subspace optimization step, compared with 200 for DeepCode and HGM. Both limits were permitted by the task specification. Since each cost evaluation consumes measurement shots, these choices can afect the cumulative shot count even under the same shot schedule.

![](images/6e3e5a01c6e0b3852f05ae2f10d14120c2e3ca78ed77397807e96af1b9e34e11.jpg)  
FIG. SI 7. Reproduction results for Paper #38 on quantum autoencoder training. Panels a-c show results from Claude Code with Opus 5, DeepCode with DeepSeek V4 Pro, and HGM with DeepSeek V4 Pro, respectively, corresponding to the original Figure 6 in Fig. SI 6c. The upper panels show the local cost during training for $L = 2 , 4 , \dots , 2 0 .$ . The lower panels show cumulative shots needed to reach target costs 0.02 (red open circles, solid lines) and 0.05 (black open squares, dashed lines). Pale red and gray bars mark the initial acceptance windows for targets 0.02 and 0.05, respectively, at $L \approx 2 , 1 0 , 1 5 , 2 0$ The annotations give the slopes of $\log _ { 1 0 } S _ { \tau }$ versus L and the revised acceptance interval [0.09, 0.22].

Expert review confirmed that the published method did not uniquely determine the absolute shot counts imposed by the initial criteria. As such, the expert approved replacing the absolute acceptance windows with checks on shotcount growth with circuit depth and the relative measurement costs at the two thresholds. The inconsistency was classified as Level I because it could be identified and resolved using the original paper and reproduction records.

Final benchmark. Reassessment of the existing runs under the revised evaluation criteria changed all three outcomes from failure to success. The revised Criteria.md in Fig. SI 6b requires the least-squares slope of $\log _ { 1 0 } S _ { \tau } ( L )$ against L to fall within [0.09, 0.22] for each target over $2 \leq L \leq 2 0$ . It also requires the ratio $S _ { 0 . 0 2 } ( L ) / S _ { 0 . 0 5 } ( \bar { L } )$ to lie within [1.2, 15]. A broader absolute shot-count range of $[ 1 0 ^ { 7 } , 1 0 ^ { 1 1 } ]$ at L = 20 was included as a non-critical check, while the threshold ordering and hundredfold-growth requirement remained unchanged. For targets 0.02 and 0.05, respectively, the fitted slopes in Figs. SI 7a–c were (0.156, 0.151) for Claude Code, (0.159, 0.149) for DeepCode, and (0.149, 0.142) for HGM. All six slopes fall within the revised acceptance interval.

## d. Level II (Paper #42): Underspecified implementation details in topological phase characterization

Paper #42 [66], published in Nature Communications in 2026, investigates the unsupervised identification of topological order, an important task in quantum many-body physics. The paper proposes two quantum kernels based on reduced-state fidelities and entanglement entropies to distinguish topologically ordered states from topologically trivial states. In particular, they examined the robustness of these kernels under random two-qubit unitary circuits of increasing depth. However, the initial reproduction experiments failed to recover the reported clustering performance despite following the methodological description in the paper. Subsequent AI-assisted auditing identified several consequential diferences between the published description and the released implementation, which required domain-expert intervention to resolve.

![](images/48d9fa258cc95de1acff44e67521b274a29e90bb4173364d7be521bbd61fda82.jpg)

![](images/aaf1070c3be3c006ad74b2be46d788601ec924c3753e3f5c3df57d437eea8232.jpg)  
FIG. SI 8. Overview of the reproduction task in Paper #42 for topological phase characterization. a. Schematic of quantum-kernel-based unsupervised learning for topological phase characterization. Pairwise similarities between quantum states are used to construct a kernel matrix, followed by kernel principal component analysis (kernel PCA) to obtain lowdimensional representations for clustering states belonging to distinct quantum phases. b. Figure 3(a)&(b) from Paper #42 [66] The left panel illustrates the separation of Kitaev’s toric code (TC) ground states and random product states (RPS) using the fidelity-based quantum kernel under random unitary perturbations, while the right panel quantifies the degradation in clustering performance with increasing circuit depth using the silhouette coeficient. c. Comparison of the methodological descriptions in the original paper, the task specification (Preliminary.md), the evaluation criteria (Criteria.md), and the authors’ released implementation.

Algorithmic background. Topological phases are characterized by nonlocal properties, such as long-range entanglement, that cannot generally be distinguished using conventional local order parameters. Unsupervised topologica phase characterization aims to identify diferent quantum phases by clustering quantum states without using their phase labels during learning. As shown in Fig. SI 8a, given a dataset of N quantum states $\{ \rho _ { i } \} _ { i = 1 } ^ { N } ,$ kernel-based meth ods construct a similarity matrix K with entries $K _ { i j } = \kappa ( \rho _ { i } , \rho _ { j } )$ , where κ denotes a quantum kernel. Kernel principal component analysis (PCA) then maps these states into a low-dimensional representation that can reveal their underlying clustering structure. To evaluate robustness against random unitary perturbations, let $\rho _ { i } ( d )$ denote the i-th state after applying a random circuit of depth $d ,$ and let $K ( d )$ be the corresponding kernel matrix. Under a standard centered kernel-PCA procedure, the kernel matrix is first centered as $K _ { c } ( d ) = H K ( d ) H .$ , where $H = \mathbb { I } - \mathbf { 1 1 } ^ { \top } / N$ . The leading eigenvector ${ \pmb v } _ { 1 } ( d )$ of $K _ { c } ( d )$ is then standardized to obtain a low-dimensional representation $z ( d ) \in \dot { \mathbb { R } } ^ { N }$ . Its clustering performance can be evaluated using the silhouette coeficient [67], i.e.,

$$
s ( d ) = \mathrm { S i l h o u e t t e } \left( z ( d ) , \pmb { y } ( d ) \right) ,\tag{B3}
$$

where $\pmb { y } ( d ) = [ \pmb { y } _ { 1 } ( d ) , \pmb { \cdot } \cdot \pmb { \cdot } , \pmb { y } _ { N } ( d ) ]$ contains the phase labels at depth d used for evaluation. A larger silhouette coeficient indicates more compact and better-separated clusters.

Task construction with OmniQ. Figure 3 of Paper $\# 4 2$ is selected as the reproduction target to evaluate the AI scientist’s ability to reproduce the reported clustering performance and its robustness against random circuit perturbations. Two panels from this figure are shown in Fig. SI 8b. That is, the left panel, i.e., Fig. 3(a) of Paper #42, illustrates the ability of the fidelity-based quantum kernel to distinguish the ground states of Kitaev’s toric code (TC) from random product states (RPS) under random unitary perturbations. The right panel, i.e., Fig. 3(b) of Paper #42, quantifies the degradation of clustering performance using the silhouette coeficient, which remains above 0.55 at depth d = 4 in the published results. These panels support the paper’s claim that entanglement-based kernels exhibit greater robustness to random unitary perturbations than fidelity-based kernels.

![](images/77dbe4c026836d84c46eef1d1928c34ea56dc26b9721c57f7d3165a16f567285.jpg)  
FIG. SI 9. Reproduction results for Paper #42 on unsupervised learning for topological phase characterization. a. Initial benchmark results under the depth-wise protocol and the original [0.5, 1] axis requirement. b. Final benchmark results under the revised specification, which retains independent analysis at each depth and expands the permitted y-axis range to [0, 1]. Columns show DSH with DeepSeek V4 Pro, Grok2, and Claude Code with Opus 5.5, from left to right. Each main curve shows the fidelity-kernel silhouette coeficient computed from 10 TC and 10 RPS samples at each circuit depth corresponding to the target quantity in original Figure 3(b). Initial panels retain the agents’ plotting styles and y-axis ranges: [0, 1], [0.5, 1], and [0, 1.02], respectively. Consequently, Grok2’s values at depths 2–4 fall outside its displayed window. Grey shading in the DSH panel and the dash-dotted line in the Opus panel mark the original paper’s display window and its lower limit, respectively. The Opus initial panel additionally shows the mean and population standard deviation over five independent realizations, and the main curves show single realizations.

Following the methodological description in the paper, as illustrated in Fig. SI 8c, the task specification in Preliminary.md requires standard kernel PCA and a dataset of N = 20 quantum states. Although the paper did not explicitly specify whether clustering evaluations at diferent circuit depths were performed independently or jointly, Fig. 3(a) of Paper #42 presents separate one-dimensional representations for each depth, while Fig. 3(b) is described as quantifying their clustering performance. Accordingly, the initial task specification adopted a depth-wise evaluation protocol, consistent with the interpretation introduced above. As illustrated in Fig. SI 8c, the evaluation criteria also required the reproduced silhouette plot to retain the original vertical-axis range of [0.5, 1]. Although no strict numerical tolerance was imposed on the silhouette coeficients at every depth, this plotting requirement implicitly assumed that the reproduced values would remain within the displayed range.

Initial benchmark. As shown in Fig. SI 9a, we present the initial reproduction results of three representative AI scientists: DSH with DeepSeek V4 Pro, a Grok-based agent, and Claude Code with Opus 5.5. All three followed the prescribed depth-wise protocol in Preliminary.md, independently constructing the fidelity-kernel representation and calculating the silhouette coeficient at each circuit depth. However, their results exhibited a substantially stronger degradation in clustering performance than reported in the original paper. At depth $d = 4 ,$ the reproduced silhouette coeficients were approximately 0.173, 0.283, and 0.267, respectively, all falling outside the displayed y-axis range of [0.5, 1] in the original figure.

The agents adopted in OmniQ responded diferently to this discrepancy. The Grok-based agent keeped the prescribed vertical-axis range of $[ 0 . 5 , 1 ] .$ , causing the silhouette values at depths 2–4 to fall outside the visible plotting window. In contrast, DSH and Claude Code with Opus 5.5 expanded the axis range to display their computed values. These diferent treatments also led to inconsistent evaluation outcomes: clipped or missing curves were reported as critical issues in some runs, whereas axis extensions were accepted as non-critical deviations in others. Notably, the initial judge accepted the Claude Code reproduction, despite its lower silhouette values and modified axis range. These outcomes exposed a discrepancy between faithfully executing the specified computational procedure and reproducing the appearance of the published figure, motivating further examination of the original implementation and evaluation criteria.

Cross-agent verification of reproduction inconsistencies. To identify the source of these discrepancies, the auditing AI scientist in OmniQ examined how the authors’ released implementation evaluated clustering performance across diferent circuit depths. As illustrated in Fig. SI 8c, the released implementation combines kernel-PCA representations from all depths up to $d ,$ contrary to the requirement in Preliminary.md that each depth be evaluated independently. Specifically, instead of computing the silhouette coeficient from the depth-specific representation $z ( d )$ as in Eq. (B3), the released implementation constructs an accumulated feature matrix $\bar { X = [ \mathfrak { z } ( 0 ) , \cdot \cdot \cdot , \mathfrak { z } ( d ) ] }$ to calculate

$$
s ( d ) = \mathrm { S i l h o u e t t e } \left( X , \pmb { y } \right) ,
$$

where $\pmb { y } = [ \pmb { y } ( 1 ) , \cdots , \pmb { y } ( d ) ]$ denotes the cluster labels obtained by applying K-means to the accumulated feature matrix $X .$ . This procedure preserves information from earlier depths when evaluating clustering performance at later depths, potentially obscuring the degradation caused by increasingly deep random circuits.

The AI scientist further identified a diference in the kernel-PCA implementation. Specifically, the released code of Paper #42 diagonalizes the uncentered kernel matrix and subsequently standardizes its leading eigenvector, instead of centering the kernel before eigendecomposition. Considering that these operations are not equivalent, this additional diference can also afect the resulting clustering representations and silhouette coeficients. Together, these implementation choices explain why following the initial task specification does not necessarily reproduce the published numerical results.

Following the AI-assisted identification of these discrepancies, domain experts determined that the independentdepth protocol should be retained to preserve the stated correspondence between Fig. 3(a) and Fig. 3(b) of Paper #42. Accordingly, the revised Preliminary.md explicitly requires centered kernel PCA and independent silhouette evaluation for the 20 quantum states at each depth. Meanwhile, the evaluation criteria in Criteria.md were revised to allow a y-axis range of [0, 1] and assess the relative robustness of the two kernels without requiring the published fidelity-kernel silhouette magnitudes.

Accordingly, this issue was classified as Level II because the discrepancy between the published description and released implementation afected the experimental methodology and could not be resolved through straightforward corrections, requiring domain experts to determine the appropriate reproduction protocol.

Final benchmark. With the clarified depth-wise evaluation protocol and revised criteria, the final benchmark assesses whether AI scientists can recover the reported clustering behavior under a consistent experimental setting via OmniQ.

As shown in Fig. SI 9b, all three AI scientists reproduced the degradation of fidelity-kernel clustering performance with increasing circuit depth. Specifically, the silhouette coeficients decreased from approximately 0.789 at depth $d = 0$ to 0.330, 0.274, and 0.201 at $d = 4$ for DSH with DeepSeek V4 Pro, the Grok-based agent, and Claude Code with Opus 5.5, respectively. Meanwhile, the corresponding entanglement-kernel results remained above 0.98 at nonzero depths, recovering the reported qualitative advantage of the entanglement-based kernel under random circuit perturbations. All three agents passed the revised evaluation for Fig. 3 of Paper #42, although their fidelitykernel silhouette coeficients remained substantially below the values reported in the original paper. These results demonstrate that the expert-revised benchmark enables faithful reproduction of the central scientific conclusion under a consistent depth-wise evaluation protocol, rather than exact recovery of the published numerical results.

## e. Level III (Paper #60): Unresolved inconsistencies in the reproduction of quantum kernel concentration

Quantum kernel methods measure similarities between classical inputs using quantum states prepared by a dataencoding circuit. Paper #60 [68], published in Nature Communications in 2024, identified several mechanisms that cause kernel values to concentrate exponentially with the number of qubits, increasing the measurement cost of resolving their diferences. The authors also investigated how encoding expressivity afects kernel concentration through numerical simulations with diferent qubit counts and circuit depths. Here, we examine the reproduction o this experiment, which identified an inconsistency in the plotted statistic and a separate discrepancy in the underlying data that remained unresolved.

![](images/49189c85870e19d92c46549e130cbefe36329b632d5848c664cb892b6075b14d.jpg)  
c

![](images/8bace1b11fac4471541aee0a2bfe75d145b1000b1e178df1ffa5f6b278c77ce0.jpg)  
FIG. SI 10. Reproduction task for Paper #60 on quantum kernel concentration. a. Schematic of the projected quantum kernel. Two encoding circuits prepare n-qubit states for inputs x and $\mathbf { x } ^ { \prime }$ . The corresponding quantum projected kernel follows from Eq. (B4). The lower surface illustrates kernel values over input pairs. b. Excerpts from the figure caption in the original paper (Paper’s original claim), task specification (Preliminary.md), and evaluation criteria (Criteria.md). The magnitude windows apply to n = 12 qubits. c. Original Figure 6 of Paper #60 [68], showing the reported variance curves for the fidelity and projected quantum kernels versus qubit count. The experiment uses 40 MNIST images of digits 0 and 1, reduced by PCA to $n = 2 , 4 , 6 , 8 .$ , 10, 12 features, with L = 2, 4, 8, 16, 25, 50, 75, 100 hardware-eficient encoding layers.

Algorithmic background. For a classical input x, the encoding circuit $U ( { \pmb x } )$ prepares $\rho ( \pmb { x } ) = U ( \pmb { x } ) \rho _ { 0 } U ( \pmb { x } ) ^ { \dagger }$ from an n-qubit initial state $\rho _ { 0 }$ . Given a dataset $\{ \mathbf { \boldsymbol { x } } ^ { ( i ) } \}$ , the kernel Gram matrix has entries $K _ { i j } = \kappa ( \pmb { x } ^ { ( i ) } , \pmb { x } ^ { ( j ) } )$ . The fidelity quantum kernel measures the overlap between the full encoded states, $\kappa ^ { F Q } ( \pmb { x } ^ { ( i ) } , \pmb { x } ^ { ( j ) } ) = \mathrm { T r } [ \rho ( \pmb { x } ^ { ( i ) } ) \rho ( \pmb { x } ^ { ( j ) } ) ]$ The projected quantum kernel compares the single-qubit reduced states $\rho _ { k } ( \pmb { x } ) = \mathrm { T r } _ { \neq k } \rho ( \pmb { x } )$ , as illustrated in Fig. SI 10a:

$$
C _ { i j } = \sum _ { k = 1 } ^ { n } \bigl | \bigl | \rho _ { k } ( \pmb { x } ^ { ( i ) } ) - \rho _ { k } ( \pmb { x } ^ { ( j ) } ) \bigr | \bigr | _ { F } ^ { 2 } , \qquad \kappa _ { i j } ^ { P Q } = \exp ( - \gamma C _ { i j } ) .\tag{B4}
$$

Here, $\gamma > 0$ is a constant and $\| \cdot \| _ { F }$ denotes the Frobenius norm. The encoding circuit used in the Paper #60 consists of L hardware-eficient encoding (HEE) layers [236], each applying data-dependent RX rotations followed by nearest-neighbor entangling gates. For the $B = 4 0$ inputs used in the experiment, let $\mathcal { P } = \{ ( i , j ) : 1 \leq i < j \leq B \}$ denote the $M = | \mathcal { P } | = 7 8 0$ unordered pairs of distinct inputs. The empirical variance for either $Z = C$ or $Z = \kappa ^ { P Q }$ is defined as

$$
\mathrm { V a r } _ { \mathcal { P } } [ Z ] = \frac { 1 } { M } \sum _ { ( i , j ) \in \mathcal { P } } ( Z _ { i j } - \bar { Z } ) ^ { 2 } , \qquad \bar { Z } = \frac { 1 } { M } \sum _ { ( i , j ) \in \mathcal { P } } Z _ { i j } ,\tag{B5}
$$

where the figure axes abbreviate the variance as $\mathrm { V a r } [ Z ]$

Task construction with OmniQ. The reproduction task targets the complete Figure 6 of Paper #60, as shown in Fig. SI 10c. Reproducing it requires classical simulation of the fidelity and projected quantum kernels across qubit counts $n = 2 , 4 , 6 , 8 , 1 0 ,$ 12 and circuit depths $L = 2 , 4 , 8 , 1 6 , 2 5 , 5 0 , 7 5 , 1 0 0$ , followed by variance calculations over distinct input pairs. The task tests both the implementation of the encoding and reduced-state calculations and the reported dependence of kernel concentration on qubit count and circuit depth.

The task specification required agents to generate the kernel values independently and document the preprocessing, input selection, random seeds, entangling gates, and kernel parameter. For settings not specified in the paper, it recommended selecting 20 images of each digit (0 and 1) as the input subset, using a CNOT ring as the entangling pattern, and setting the kernel parameter to $\gamma = 1$ . The Criteria.md, summarized in Fig. SI 10b, specified absolute acceptance windows for the quantity labeled as projected-kernel variance at $n = 1 2 \colon \mathrm { [ 1 0 ^ { - 2 } , 1 0 ^ { - 1 } ] }$ at $L = 2$ and $[ 5 \times \mathrm { \bar { 1 0 ^ { - 7 } } } , 1 0 ^ { - 5 } ]$ at $L = 1 0 0$ . It also required a reduction of at least $1 0 ^ { 4 }$ between these depths. The absolute windows applied only under the recommended settings. When agents documented alternative choices for parameters unspecified in the paper, these windows were treated as advisory, but the depth-reduction requirement remained critical.

a  
![](images/9a6c7df69ac25098ed2f44ad5e9963eca9f73ab8f6066041a957c20b8b5ac761.jpg)

b  
![](images/825a2af2bfcd37b10cd365f934d29625de2cf06c64fe37eae18302678d72cb69.jpg)

c  
![](images/fd1296fc9106f87e9a923393032b4ab33c6f0e9703577c09212178417a03f7bd.jpg)  
FIG. SI 11. Initial benchmark and numerical checks of the projected-kernel result in Paper #60. a. Projected kernel variance at $n = 1 2$ across circuit depths for Codex with GPT-5.6-sol (blue circles), Claude Code with Opus 5.5 (orange triangles), and the values reported in original Figure 6b (black squares). Gray bands indicate the acceptance windows at $L = 2$ and $L = 1 0 0$ . b. Variance computed from the authors’ released data for Figure 6b of Paper #60 at $n = 2 .$ . The red dashed line marks the upper bound $\mathrm { V a r } [ \kappa ^ { \hat { P } Q } ] \leq 1 / 4$ for projected-kernel variance, as given in Eq. (B6). c. Normalized distance variances at $n = 1 2$ from the two agent runs and 3,000 balanced subsets of 20 images per digit. Each normalized value, denoted by $R ( L )$ on the axis, is the ratio of the corresponding distance variance to the reference variance computed from the authors’ released data at the same depth. The subsets were drawn from an audit pool of 1,500 images per digit. Gray diamonds indicate the subset medians, and $\mathrm { g r a y }$ bars show the empirical 5th–95th percentile ranges. The dashed line marks $R = 1$

Initial benchmark. At $n = 1 2$ and $L = 2$ , Codex with GPT-5.6-sol and Claude Code with Opus 5.5 each obtained projected-kernel variances below $3 \times 1 0 ^ { - 3 }$ , as shown in Fig. SI 11a. Both values were more than an order of magnitude smaller than the value reported in the original paper and outside the acceptance window $[ 1 0 ^ { - 2 } , 1 0 ^ { - 1 } ]$ specified in Criteria.md. At $L = 1 0 0$ , both results fell within the acceptance window. However, the variance decreased by less than a factor of $1 0 ^ { 3 }$ between these depths in each run, falling short of the required $1 0 ^ { 4 }$ reduction. Both runs therefore failed the initial numerical criteria. We then examined the statistic plotted in the original figure and the data used to generate it.

Cross-agent verification of reproduction inconsistencies. Equation (B4) confines every projected-kernel value to [0, 1]. Writing $\kappa = \kappa ^ { P Q }$ and taking averages over ${ \mathcal { P } } ,$ we have

$$
\operatorname { V a r } _ { \mathcal { P } } [ \kappa ] = \mathbb { E } _ { \mathcal { P } } \left[ \kappa ^ { 2 } \right] - \left( \mathbb { E } _ { \mathcal { P } } \left[ \kappa \right] \right) ^ { 2 } \leq \mathbb { E } _ { \mathcal { P } } \left[ \kappa \right] - \left( \mathbb { E } _ { \mathcal { P } } \left[ \kappa \right] \right) ^ { 2 } = \mathbb { E } _ { \mathcal { P } } \left[ \kappa \right] \left( 1 - \mathbb { E } _ { \mathcal { P } } \left[ \kappa \right] \right) \leq \frac { 1 } { 4 } .\tag{B6}
$$

This bound holds for any finite collection of input pairs, regardless of the distribution P. At $n = 2 .$ , however, the variance computed from the authors’ released data for the original Figure 6b is approximately 0.5 at both $L = 2 5$ and 75, as shown in Fig. SI 11b. Both values exceed the upper bound of $1 / 4$ and should not represent the projected-kernel variance labeled in the original figure. The released scripts explain this inconsistency. The function proj kernel in the authors’ code returns the distance sum $C _ { i j } ,$ while the plotting routine applies np.var directly to these distances and labels the result as projected-kernel variance.

We compared the distance variances Var[C] obtained by Codex with GPT-5.6-sol and Claude Code with Opus 5.5 with those calculated from the data released by the authors at $n = 1 2$ , as shown in Fig. SI 11c. The ratio $R ( L )$ is defined as the variance from each run divided by the reference variance at the same depth, with values close to 1.0 indicating agreement with the reference results. For both agents, $R ( L ) < 1 0 ^ { - 1 }$ at $L = 2$ and 4, and approaches 1.0 for $L  \geq 1 6$ . To examine whether input selection could explain these diferences, we calculated distance variances for 3,000 balanced subsets of 20 images per digit and normalized them by the reference variance at each depth. At both $L = 2$ and 4, the reference variance exceeds the 95th percentile of the sampled subsets. These results suggest that random input selection is unlikely to explain the shallow-depth discrepancies under the settings examined.

Final benchmark. Expert review classified the case as Level III because the plotted-statistic mismatch could be explained from the released scripts, but no reproducible procedure was identified for generating the reference variances.

Consequently, we exclude this task from the cleaned computational quantum science dataset used to evaluate the reproduction capabilities of diferent AI scientists.

## SI C: Implementation of RA

In this section, we provide the implementation details of RA, whose overall workflow is illustrated in Fig. 1b in the main text. We begin with an overview of the framework of RA in SI. C 1, followed by a detailed description of the implementation of its components in SI. C 2–C 5.

## 1. Overview

RA consists of five functional components: a planner, an executor, a critic, a judge agent, and an experience distiller. These components collaborate to reconstruct computational workflows from scientific papers, validate reproduction results, resolve implementation issues, and accumulate reusable experience. Specifically, the planner, executor, and critic form the core reproduction workflow, while the judge agent independently evaluates the completed reproduction and the experience distiller extracts generalizable lessons from verified feedback.

Given a scientific paper, a target figure, and the corresponding task specification, the planner decomposes the reproduction task into a set of sub-plans with explicit dependencies, represented by a dependency graph. Each node corresponds to a computational sub-plan, while each edge specifies a dependency between sub-plans. The planner also establishes shared methodological interpretations and assigns completion requirements to individual sub-plans to maintain consistency throughout execution.

Following this plan, the executor and critic interact iteratively to implement and validate the sub-plans. The executor performs the required computations, while the critic independently checks the resulting implementations and outputs against the prescribed methodology and task requirements. Based on its review, the critic determines whether a sub-plan can be accepted, requires local repair, or needs a new computational approach. This iterative process continues until the required computations are completed or the available computational budget is exhausted.

After execution, the judge agent independently evaluates the assembled reproduction results against the prescribed evaluation criteria, which are not accessible to the executor or critic. The resulting feedback, together with execution records and verified failures, provides the evidence from which the experience distiller extracts reusable lessons. During development, these lessons are reviewed by domain experts before being incorporated into the experience layer, which guides subsequent reproduction tasks without modifying the underlying base LLM or orchestration code. Through this process, RA combines within-task issue resolution with experience-based self-improvement across tasks.

## 2. Planner

The planner is responsible for translating a paper-to-result reproduction task into a structured computational plan, which will be passed to the executor and critic for implementation and verification. Given the scientific paper, target figure, and task specification, it first identifies the computational requirements needed to reproduce the reported results and decomposes them into a set of sub-plans. Each requirement, such as a numerical value, parameter-dependent curve, or figure panel, is then assigned to exactly one sub-plan to ensure that all prescribed outputs are covered.

For each sub-plan, the planner sets a computational resource budget and defines completion criteria based on two sources. First, the planner establishes criteria based on explicit requirements in the task specification and derives additional criteria from the scientific claims to be reproduced when necessary. Moreover, these criteria can be further specified using additional information extracted from published figures to support self-validation without treating the plotted values as fitting targets.

The dependencies among these sub-plans are organized into a dependency graph, which determines their execution order and enables intermediate results to be shared across related computations. When the paper admits multiple methodological interpretations, the planner establishes a consistent interpretation shared by the relevant sub-plans, while preserving experimental settings explicitly fixed by the task specification. These shared interpretations may be revised when supported by new evidence, with the afected sub-plans subsequently revalidated.

![](images/353349ccf55af3657650c48972b71bf0a85aeba4cb04229605f4f553ec4b6cc9.jpg)  
FIG. SI 12. Schematic of the planner, executor, and critic workflow. The planner decomposes the reproduction task into interdependent sub-plans organized in a dependency graph, while the executor and critic collaborate to execute, review, and correct each sub-plan. a. Local repair mechanism, where the critic returns defective results to the executor for correction. b. Schematic of the dependency graph constructed by the planner to organize sub-plans and their execution dependencies. Each blue circle represents a sub-plan. c. Experiment-driven repair mechanism, where three independent probe agents investigate failures through small experiments, followed by a majority vote among three critics to select a new approach for execution.

## 3. Interactive mechanism of critic and executor

The executor and critic form the core iterative mechanism of RA for implementing, validating, and correcting computational sub-plans. As illustrated in Fig. SI 12, this mechanism operates on the dependency graph generated by the planner, where each node represents a computational sub-plan and each edge specifies a dependency between sub-plans. During execution, the critic evaluates the results of individual nodes and determines whether they can be accepted, require local repair, or need a new computational approach. These decisions guide the progression of the dependency graph, with additional diagnostic branches introduced when necessary. In this way, the reproduction process can be viewed as an adaptive execution graph that progressively resolves computational sub-plans through validation and issue resolution.

Sub-plan execution and validation. Following the dependency graph, the executor implements each sub-plan once its prerequisite nodes have been accepted. Each executor operates in an independent session and receives the corresponding task requirements, shared methodological interpretations, and outputs from preceding sub-plans. Before performing full-scale computations, the executor validates its implementation on the smallest exactly solvable instance and estimates the computational cost by timing a representative step at the largest required system size. These checks help identify implementation errors and assess computational feasibility before committing substantial resources. For computationally demanding tasks, long-running calculations are launched as background jobs tracked by the orchestrator, allowing the executor session to terminate without interrupting the computation. The orchestrator subsequently resumes the executor when the job completes or stalls, without requiring continuous calls to the base LLM. To ensure reproducibility, all reported numerical results must be traceable to the corresponding code and data

Critic review and decision routing. The critic independently reviews the computational results generated by each executor before accepting the corresponding sub-plan. Unlike the judge agent, which evaluates the completed reproduction against the benchmark criteria, the critic only accesses the scientific paper, task specification, and execution records. Its review focuses on three aspects: (i) result provenance, verifying that reported numerical values are supported by the generated code and data; (ii) methodological correctness, checking whether the implementation follows the prescribed algorithms and experimental settings; and (iii) task completion, assessing whether all requirements of the sub-plan have been satisfied. These checks are primarily conducted through inspection, supplemented by small verification experiments when necessary. The critic can also inspect ongoing computations and identify potential failures before they finish.

Based on these checks, the critic assigns each sub-plan to one of three outcomes: (i) Pass, when all requirements are satisfied and no verified defects remain, allowing dependent sub-plans to proceed; (ii) Local repair, when an identified defect can be corrected within the remaining computational budget; and (iii) New approach, when the defect cannot be reliably resolved through local correction or requires a diferent computational strategy. A passing node is marked as completed, making its validated outputs available to dependent nodes in the graph. In contrast, the latter two outcomes trigger diferent corrective branches, as described below.

Local repair. When a defect has a known correction, the critic returns the corresponding sub-plan to the executor with specific feedback identifying the error and the required changes. The executor then modifies the existing implementation and resubmits its results for another critic review. This procedure allows the same node to be corrected without changing its dependencies or restarting the entire computational workflow. The repair process continues until the sub-plan passes the review or further local correction is deemed inefective. Importantly, results containing fabricated or untraceable numerical values are not eligible for ordinary repair, since their computational provenance cannot be established.

Experiment-driven diagnosis and new approach. When local repair is insuficient, either because the source of the defect remains unclear or because repeated corrections have failed, the critic initiates an experiment-driven diagnosis. This process introduces three temporary diagnostic branches associated with the failed node, each handled by an independent probe agent. These agents examine the failed sub-plan and its residual, consisting of verified defects and reusable computational artifacts. Each probe agent independently proposes a possible cause of the failure and tests its hypothesis through a small numerical experiment.

The resulting diagnoses are evaluated by a critic panel consisting of three independent critic sessions. Before selecting a solution, the panel excludes proposals that violate fixed task requirements, substitute prescribed methodological components, or rely solely on improved agreement with the published figure without independent validation. The remaining diagnoses are ranked by majority vote, and the highest-ranked diagnosis forms the basis of a new computational approach, potentially incorporating compatible elements from other proposals. A fresh executor then implements this approach using the archived code and residual, with its results subjected to another round of critic review. These diagnostic branches support the resolution of the original failed node without changing the scientific requirements or dependency structure of the reproduction task.

Dependency-aware revalidation. The three-way decision mechanism allows RA to progressively complete the reproduction task specified by the dependency graph while correcting issues at individual nodes. Beyond these local decisions, changes to shared methodological interpretations may afect multiple previously accepted sub-plans. When such a change is supported by new evidence, the orchestrator reopens the sub-plans that used the shared information, and the critic reassesses their validity. Results that remain correct are retained, while afected computations undergo local repair or a new approach. This dependency-aware revalidation maintains consistency across the computational workflow without unnecessarily repeating completed work.

In exceptional cases where the prescribed requirements of a sub-plan are contradictory or cannot be satisfied, the critic may retire the afected node. If this prevents subsequent sub-plans from proceeding, the planner may revise the dependency graph to determine whether the remaining requirements can still be completed without violating the task specification. Throughout execution, all components operate under a shared computational budget, with part of the available time reserved for saving results and preparing the final report. Once all required sub-plans have passed the critic’s review, their outputs are assembled into the final reproduction artifacts for independent evaluation by the judge agent. Together, these procedures enable RA to combine independent validation, iterative issue resolution, and coordinated revisions within a reliable computational reproduction workflow.

![](images/a076b9d05b1d437970c063da04e9ffb0e792bba4ab0ccfaf1ed9dedbbc1e54e3.jpg)  
FIG. SI 13. The scheme of the experience distiller. The distiller identifies common patterns across development runs and transforms them into reusable experience, which is incorporated into the corresponding agent memories after human expert approval. Left, three representative patterns identified across multiple runs, with stacked copies indicating that similar patterns appear in other runs. Middle, the distiller evaluates each pattern and determines which agent role can benefit from it. Right, the workflow of RA, where purple highlights components improved by accumulated experience. In particular, residual memory guides the executor in repairing failed steps. All runs are illustrated schematically.

## 4. Judge agent

The Judge agent is used to evaluate the final reproduction results generated by RA. Once the reproduction run is complete, it receives the task specification, evaluation criteria, and execution workspace, including the source code, logs, numerical data, figures, and final report. Only the Judge has access to the evaluation criteria, while the Planner, Executor, and Critic do not receive it during task execution.

For each target figure, the Judge first checks that all required panels and curves are present over the specified parameter range. It then traces the plotted values to the saved data and code to verify how they were calculated. The implementation is checked against the task specification, including the definitions of observables, experimental parameters, and required computational steps. Any permitted approximations or choices of convention are checked against the description in the report. The Judge then compares the saved numerical results with the prescribed checkpoints and assesses whether the curves reproduce the expected performance rankings and trends.

The Judge records any issues found in this assessment together with evidence from the code, data, or execution logs. Each issue is classified as critical or non-critical under the OmniQ protocol. Critical issues afect scientific correctness or leave a required part of the reproduction incomplete, such as an incorrect computational procedure, an unsupported numerical value, a missing required curve, or a result outside a mandatory tolerance. Non-critical issues concern presentation diferences, such as colors or legend placement, that leave the scientific content unchanged. A figure passes when no critical issue is found. A task is counted as successful when all its target figures pass.

## 5. Experience distiller

The experience distiller enables RA to learn from previously encountered failures by converting verified issue resolutions into reusable guidance for subsequent reproduction tasks. It operates outside individual reproduction runs and is invoked after each round of development experiments. Importantly, the distiller updates only the experience layer, while the base LLM, agent roles, and orchestration code remain unchanged. To extract useful experience, it examines the judge agent’s feedback and the complete execution records, including the computational plans, critic reviews, and implementations of individual sub-plans.

The distilled experience is organized into three types of memory: planner memory, critic memory, and residual memory, as illustrated in Fig. SI 13. The planner memory provides guidance for constructing reliable computational plans, such as scheduling validation on small, exactly solvable instances before full-scale computation and prioritizing tasks essential to the paper’s central claims. The critic memory improves the identification of implementation issues by emphasizing checks on numerical provenance, methodological correctness, and consistency with the reported results. The residual memory stores generalizable failure signatures, each consisting of three elements: a description of the failure, a diagnostic check for identifying it, and a corresponding remedy. These checks are derived from how the critic or judge agent identified the issue, while the remedies are extracted from verified corrections that subsequently led to successful reproduction.

During task execution, relevant entries in the residual memory are retrieved to guide issue resolution. For example, when the critic requests a local repair, it identifies the entries corresponding to the verified defect, which are then provided to the executor to guide the correction. Similar guidance is supplied to a fresh executor when a new computational approach is required. This mechanism allows experience accumulated from previous failures to inform subsequent decisions without modifying the underlying execution framework.

To maintain the reliability and generalizability of the accumulated experience, the distiller excludes lessons specific to individual papers and merges those that overlap with existing entries. It can propose additions, modifications, or deletions across all three memories, but every update must be reviewed and approved by a human expert before being incorporated into the experience layer. In our implementation, the memories were developed exclusively from five development tasks, resulting in ten entries in the residual memory. The resulting experience layer was then fixed throughout evaluation on the remaining 54 unseen tasks, allowing us to assess whether the distilled experience generalizes beyond the tasks used for its development.

## SI D: An autonomous platform for quantum computational science

To make computational quantum science more accessible and reusable, we develop an autonomous research platform that integrates OmniQ in SI B, RA in SI C, and the Model Context Protocol (MCP). The platform enables researchers and AI agents to reproduce published results and build upon previously verified computational methods. Specifically, it provides two core functionalities:

1. Reproducing new research. Automatically reproducing computational results from user-submitted scientific papers.

2. Reusing existing algorithms. Discovering and applying previously reproduced algorithms through simple MCP queries.

Below, we briefly describe how these two functionalities are implemented and accessed. More details are provided in https://omniq-qid-lab.github.io/.

Reproducing new research. Building on the OmniQ framework, the platform integrates task specification, AIdriven computation, independent verification, and result delivery into a unified workflow. In particular, a user begins by submitting a scientific paper and defining a research objective, such as reproducing a published figure or investigating the same method under new experimental parameters. An analysis agent translates this request into two documents: a task specification that guides the AI scientist and evaluation criteria that define the requirements for successful reproduction. The user reviews and confirms both documents before execution, while the evaluation criteria remain inaccessible to the AI scientist.

Once the task is confirmed, the user selects an AI scientist, such as RA or a first-party coding agent, together with the underlying LLM and computational budget. The platform executes the task in an isolated computational environment and independently evaluates the resulting code, data, and figures against the predefined criteria.

Upon completion, the user receives an evaluation report and can download the source code, generated data, re produced figures, and a runnable container. This workflow enables researchers to inspect, reproduce, and extend computational results without reconstructing the experimental environment.

Reusing existing algorithms. Beyond reproducing individual papers, the platform organizes successfully reproduced algorithms into a shared collection of verified computational workflows. To establish their reliability, successfu reproductions can undergo additional verification using a fixed judge configuration, with optional review by domain experts.

The platform is designed to issue digitally signed certificates linking each verified workflow to its source paper, evaluation criteria, AI scientist configuration, and executable results. These certificates establish the reproducibility of the specified computational results without making claims about the scientific validity of the original paper.

Researchers and external AI agents can access these workflows through a web interface or MCP. In particular, MCP enables an AI agent to search for relevant algorithms using natural-language queries, retrieve their implementations and experimental settings, and incorporate them into new research tasks. As such, researchers can directly build upon existing, verified implementations, reducing redundant computational efort and facilitating subsequent scientific investigations.

Extensibility. The platform adopts a modular architecture that supports integration with diferent AI scientists, underlying LLMs, computational environments, and scientific evaluation protocols.

Although initially developed for computational quantum science, the framework can be extended to other computational disciplines by adapting their task specifications, evaluation criteria, and computational resources. The 59 reproductions from this study provide the initial collection of workflows, which can expand as new papers are reproduced and verified. We have implemented a prototype web interface and MCP tool definitions, with deployment of the full public service planned.

## SI E: More experimental results

This section provides additional experimental results to further evaluate the capabilities and limitations of AI scientists in computational quantum science. In SI E 1, we present additional results of diferent AI scientists on the established computational quantum science dataset. In SI E 2, we examine representative case studies to understand the mechanism of RA.

## 1. Additional results of AI scientists on the established computational quantum science dataset

To investigate how harness design influences the scientific reproduction capabilities of AI scientists, here we conduct a model-controlled comparison in which the underlying LLM is held fixed while the agent harness varies. In doing so, this comparison allows us to assess how alternative agent architectures and execution strategies afect reproduction performance on the cleaned computational quantum science dataset.

The experimental settings are as follows. We use DeepSeek V4 Pro, a text-only model, to compare six third party harnesses across three categories introduced in SI A, i.e., general software harnesses with mini-swe-agent and live-swe-agent, scientific-reproduction harnesses with InternAgent and DeepCode, and evolved harnesses with HGM and Meta-Harness. We additionally include DeepSeek Harness (DSH), the model provider’s first-party agent, as a reference. All third-party harnesses use their publicly released implementations and default configurations, except for the modifications described below. DSH operates with its full toolset, including shell commands, file operations, sub-agents, and background jobs.

The experimental settings are as follows. We use DeepSeek V4 Pro, a text-only model, to compare six third-party harnesses across three categories introduced in SI A 4: general software harnesses (i.e., mini-swe-agent [77] and liveswe-agent [78]), scientific-reproduction harnesses (i.e., InternAgent [75] and DeepCode [76]), and evolved harnesses (i.e., HGM [79] and Meta-Harness [80]). We additionally include DeepSeek Harness (DSH), the model provider’s firstparty agent, as a reference. Specifically, DSH operates with its full toolset, including shell commands, file operations, sub-agents, and background jobs. Note that all third-party harnesses use their publicly released implementations and default configurations, except for the modifications described below.

• For general software harnesses, mini-swe-agent uses a single bash tool, with a limit of 250 steps and 1800 s per command. live-swe-agent additionally supports tools constructed during execution, with 300 steps and 3600 s per command.

• For scientific-reproduction harnesses, InternAgent operates in paper-reproduction mode with idea generation disabled. Its coding sessions use the OpenAI-compatible tool-calling interface, with initial limits of 70 steps and 45 min per session and 65 min per experiment run. For some tasks, the session limit is increased to 150 steps, with the time cap extended to the task-level limit. DeepCode follows its paper-to-code workflow, followed by execution of the generated code without its web interface.

• For evolved harnesses, HGM uses its released best agent with a single attempt instead of the default three and a 120 s limit per bash call. Meta-Harness uses its bash, mark-complete, and image-read tools, with a 600 s limit per command block.

Table SI 3 summarizes the reproduction performance and failure patterns of diferent AI scientists on the cleaned computational quantum science dataset. To characterize these failures, we used Claude Opus 5.5 to classify 954 critical findings across 266 failed runs into predefined categories, without revealing harness identities. A repeated classification of 60 randomly sampled findings yielded 98.3% agreement (59/60).

An interesting observation is that despite sharing the same underlying LLM, the seven harnesses exhibit substantial diferences in reproduction performance. More concretely, DSH achieves the highest success rate, reproducing 29 of 59 tasks, closely followed by mini-swe-agent with 28 successful reproductions at approximately one-third of DSH’s cost. In contrast, live-swe-agent reproduces only 18 tasks despite incurring roughly twice the cost of mini-swe-agent. Among the scientific-reproduction harnesses, DeepCode outperforms InternAgent, while HGM reproduces 24 tasks, exceeding both scientific-reproduction harnesses. Meta-Harness achieves the lowest performance, reproducing only 11 tasks.

TABLE SI 3. Failure categories of AI scientists using DeepSeek V4 Pro. The first row reports successful reproductions out of 59 tasks. Failure categories are non-exclusive, with each run counted once per applicable category; runs without reports are counted only under “No report”. “Reduced problem” denotes reductions in problem size, sampling, evaluation points, or numerical precision, while “Mismatch with the paper” indicates discrepancies in reported values, rankings, or trends. The final two rows characterize the consistency of failed-run reports with their underlying results.
<table><tr><td></td><td>DSH</td><td>mini-swe- agent</td><td>live-swe- agent</td><td>Intern- Agent</td><td>Deep- Code</td><td>HGM</td><td>Meta- Harness</td></tr><tr><td>Reproduced tasks</td><td>29</td><td>28</td><td>18</td><td>16</td><td>21</td><td>24</td><td>11</td></tr><tr><td>Failed runs</td><td>30</td><td>31</td><td>41</td><td>43</td><td>38</td><td>35</td><td>48</td></tr><tr><td>No report</td><td>6</td><td>0</td><td>1</td><td>13</td><td>0</td><td>1</td><td>1</td></tr><tr><td>Substitution or fabrication</td><td>4</td><td>8</td><td>14</td><td>7</td><td>4</td><td>9</td><td>13</td></tr><tr><td>Reduced problem</td><td>6</td><td>6</td><td>13</td><td>7</td><td>15</td><td>7</td><td>18</td></tr><tr><td>Implementation error</td><td>9</td><td>13</td><td>22</td><td>13</td><td>12</td><td>10</td><td>11</td></tr><tr><td>Missing content</td><td>2</td><td>3</td><td>7</td><td>4</td><td>10</td><td>6</td><td>6</td></tr><tr><td>Rendering error</td><td>2</td><td>10</td><td>6</td><td>3</td><td>7</td><td>9</td><td>9</td></tr><tr><td>Mismatch with the paper</td><td>24</td><td>25</td><td>37</td><td>30</td><td>34</td><td>28</td><td>42</td></tr><tr><td>Report contradicts its data</td><td>3</td><td>1</td><td>5</td><td>7</td><td>4</td><td>2</td><td>3</td></tr><tr><td>Report admits the gap</td><td>16</td><td>18</td><td>21</td><td>12</td><td>21</td><td>23</td><td>31</td></tr></table>

To further characterize these diferences, we analyze the critical issues identified in unsuccessful runs. As shown in Tab. SI 3, mismatches with the original paper are the most prevalent failure category across all harnesses, afecting 24 − 42 runs per configuration. However, the failure patterns vary considerably. InternAgent fails to produce a report in 13 of its 43 unsuccessful runs, suggesting dificulties in completing the reproduction workflow. In contrast, liveswe-agent exhibits implementation errors in 22 of its 41 failed runs, while Meta-Harness reduces the original problem settings in 18 of its 48 failed runs. Notably, these failure categories are not mutually exclusive, as a single run may exhibit multiple issues. Collectively, these results highlight the importance of harness design in scientific reproduction and suggest that greater architectural complexity does not necessarily translate into higher reproduction performance.

## 2. Case studies of RA

In this subsection, we examine how RA identifies and corrects reproduction failures through three representative case studies from Paper #2, Paper #35, and Paper #29. On these tasks, DSH and most third-party harnesses evaluated with DeepSeek V4 Pro failed to reproduce the reported results, whereas RA succeeded. Notably, none of the three papers had encountered reproduction issues in the curation audit in Table SI 2, allowing us to focus on failures in agent execution rather than known inconsistencies in the scientific record. In each case, RA identified discrepancies between intermediate results and the methodological or quantitative requirements of the original paper, subsequently diagnosing and correcting the underlying errors without altering the prescribed task settings. These cases illustrate three complementary mechanisms of RA, namely sub-plan decomposition, critic-guided issue resolution, and experiment-driven diagnosis, which we examine separately below.

Case study 1 (Paper #2): Sub-plan decomposition. This case illustrates how RA uses sub-plan decomposition to overcome a reproduction challenge that other AI scientists failed to resolve, by ensuring consistent methodologica interpretations and independent validation across computational sub-plans.

Reproduction challenge and baseline failure. Paper #2 [14] investigated the classical simulation of IBM’s 127-qubit kicked Ising experiment using tensor networks contracted through belief propagation, with simulation errors evaluated against an exact light-cone benchmark. As shown in Fig. SI 14a, the circuit consists of repeated Trotter steps combining fixed Ising interactions with single-qubit X rotations parameterized by an angle $\theta _ { h } .$ , where $\theta _ { h } \ = \ \pi / 2$ corresponds to a Cliford circuit that provides an exact reference point. A valid comparison requires the tensornetwork estimate $\langle O \rangle _ { \mathrm { T N } }$ and the exact benchmark $\langle O \rangle _ { \mathrm { e x a c t } }$ to evaluate the same physical observable O. In particular, the weight-17 Pauli observable $O _ { 1 7 }$ in Figure 3(c) of the original paper is constructed by propagating a single-site Z operator backward through several Cliford steps, requiring the correct qubit and number of steps to be identified. However, DSH using the same base LLM failed to maintain this consistency. It initially simulated Figure $3 ( \mathrm { c } )$ using a qubit selected during an early search, but subsequently discovered that this choice generated a weight-23 observable $O _ { 2 3 }$ . Although DSH then computed the exact benchmark for the correct observable, the corresponding simulation was not completed before the run ended. Consequently, its final figure incorrectly compared $\langle O _ { 2 3 } \rangle _ { \mathrm { T N } }$ with $\langle O _ { 1 7 } \rangle _ { \mathrm { e x a c t } } .$ resulting in substantially larger errors than those reported in the paper.

a  
![](images/8dc74f1eff9a5845044af2c73500042ba4654fed5c6f44b1c8fa3fd35c60eaa7.jpg)

b  
![](images/6be73087fc696921367a5ada69523b762c28a10bdd91d476717396bfc9092b6c.jpg)  
FIG. SI 14. Simulation workflow and reproduction results for Paper #2. a. Schematic of the 127-qubit kicked Ising circuit (upper panel) and the corresponding tensor-network simulation based on belief propagation (lower panels). b. Comparison of the reproduction results obtained by diferent AI scientists with the original results from Fig. 3(c) of Paper #2 [14]. The horizontal axis denotes the single-qubit rotation angle $\theta _ { h }$ , while the vertical axis shows the error between the tensor-network estimate and the exact light-cone benchmark on a logarithmic scale. The labels ‘RA-Initial’ and ‘RA-Final’ denote the results obtained by RA before and after its internal corrections, respectively, while ‘DSH’ denotes the baseline reproduction using DeepSeek Harness. The label ‘Paper’ denotes the reference error values reported in the original paper.

Sub-plan decomposition and reproduction results. RA identified the correct target observable before execution and decomposed the task into five independently validated sub-plans, using shared methodological interpretations to maintain consistency across their computations. In particular, the exact benchmark was implemented independently of the simulation engine, while both evaluated the same observable. When the full-scale simulation revealed a separate error in the belief-propagation implementation, RA coordinated corrections across the afected sub-plans while retaining the validated exact benchmark.

The reproduction results are presented in Fig. SI 14b. DSH reported errors ranging from approximately $1 0 ^ { - 3 }$ to $1 0 ^ { - 1 }$ , several orders of magnitude larger than the published reference values at many rotation angles. As discussed above, these discrepancies resulted from comparing diferent observables rather than directly reflecting the accuracy of the intended tensor-network simulation. By contrast, the initial RA reproduction already achieved substantially smaller errors, although noticeable deviations from the published results remained at intermediate rotation angles, particularly around $\theta _ { h } = 3 \pi / 8$ . Following the coordinated corrections, the final RA results more closely reproduced both the reported error magnitudes and their dependence on $\theta _ { h }$ , including the peak near $\theta _ { h } = 3 \pi / 8$ and the sharp decrease towards the Cliford point $\theta _ { h } = \pi / 2$ . Quantitatively, the final results agreed with the published error values to within approximately 2% at 15 of the 23 evaluated points. These results illustrate how sub-plan decomposition and coordinated correction enabled RA to recover the reported simulation behavior that DSH failed to reproduce.

Case study 2 (Paper #35). Critic-guided issue resolution. In this case, RA corrected an underestimated oraclecall cost through critic-guided local repair: the critic identified an invalid circuit schedule, the executor corrected the implementation, and the critic reviewed the revised results before accepting the afected sub-plan.

Background and baseline failure. Paper #35 [128] develops low-depth algorithms for estimating the angle θ encoded by a state-preparation oracle U, i.e., $U | 0 \rangle ^ { \otimes n } \stackrel { \cdot } { = } \cos \theta | \psi _ { 0 } \rangle \bar { | } 0 \rangle + \sin \theta | \psi _ { 1 } \rangle | \bar { 1 } \rangle$ , where $0 \leq \theta \leq \pi / 2$ , and $\{ | \bar { \psi } _ { 0 } \rangle , | \psi _ { 1 } \rangle \}$ are normalized states of the remaining register. The task is to estimate θ within additive error ϵ using the procedure illustrated in Fig. SI 15a. Specifically, each amplification step applies $Q = U R _ { 0 } U ^ { \dagger } S _ { 0 }$ , where $R _ { 0 } = 2 | \bar { 0 } \rangle ^ { \otimes n } \langle \bar { 0 } | ^ { \otimes n } - \mathbb { I } _ { 2 ^ { n } }$ and $S _ { 0 } = \mathbb { I } _ { 2 ^ { n - 1 } } \otimes Z$ . Each application of $Q$ uses two oracle calls of $U .$ . In this amplitude estimation procedure, the number of amplitude amplification steps $m _ { k }$ increases across measurement rounds $k = 1 , 2 , \cdots$ according to a power law $m _ { k } = \hat { \left. \right.} k ^ { \left( 1 - \beta \right) / \left( 2 \beta \right) } $ with a tunable exponent, which allows a tradeof between circuit depth and total oracle calls. The reproduction targets are the relationships between the number of oracle calls N and the estimation error ϵ shown in Figures 5 and 6 of the original paper, for noiseless circuits and circuits with depolarizing noise at diferent decay rates $\gamma _ { \mathrm { { i } } }$ respectively. With schedules adapted to the target error and noise level, estimation at $\epsilon \gg \gamma$ retains approximately the quantum scaling $N \propto \epsilon ^ { - 1 }$ . At finer precision with $\epsilon \ll \gamma$ , the measurement contrast is suppressed by $e ^ { - \gamma q }$ after q oracle calls. More measurements are then needed, and the cost approaches the classical scaling $N \propto \bar { \epsilon } ^ { - 2 }$ . The crossover error $\epsilon _ { C }$ marks the transition between these scaling regimes and is expected to be of order $\gamma .$ . DSH reproduced the noiseless relation but obtained a noisy crossover at $\epsilon _ { C } \approx 0 . 0 5 \gamma$ , well below this scale, as shown in Fig. SI 15c. Although it documented this discrepancy, DSH incorrectly declared the noisy reproduction successful. The judge agent marked the noisy target figure as failed because the crossover fell outside the prescribed evaluation interval $[ 0 . 3 \gamma , 3 \gamma ]$

a  
![](images/3b6c493a67238a5736e826e2ded28fad3aaf0caeff31c2ca64cce57dbf0329a8.jpg)

![](images/253e0a5e5e5304e09136b134ad796c71c62fff0b0cfdaf7e926e26f4a2b1ac25.jpg)

c  
![](images/a3c5b7ff50b2b3779e96f39f7e13b13a8caa636e1db304a434227218682732a6.jpg)  
FIG. SI 15. Noisy amplitude estimation and critic-guided issue resolution for Paper #35. a. Amplitude estimation with state preparation U and $m _ { k } = \vert k ^ { \left( 1 - \beta \right) / \left( 2 \beta \right) } \vert$ ⌋ amplification steps at round k. Each $Q$ uses two oracle calls. In this equivalent circuit representation, noise is grouped into $\mathcal { D } _ { \gamma }$ after the initial $\bar { U }$ and $\mathcal { D } _ { 2 \gamma }$ after each $Q .$ . Here $\mathcal { D } _ { t }$ denotes global depolarizing noise with the factor $e ^ { - t } .$ b. Reported oracle calls N versus error ϵ at $\gamma = 1 0 ^ { - 3 }$ for the original paper, DSH and RA before and after correction. c. Crossover diagnostics versus noise strength. For both RA stages, $\epsilon _ { C }$ is the geometric midpoint between adjacent target errors where the chosen circuit family changes between power-law and exponential schedules. Vertical bars delimit these sampled intervals. The DSH values are crossover estimates extracted from its numerically evaluated theoretica cost curves. The dashed line marks $\epsilon _ { C } = \gamma$ , and the shaded region is the evaluation interval $[ 0 . 3 \gamma , 3 \gamma ]$

Critic-guided issue resolution. At the initial stage, RA also underestimated the cost in the noise-dominated regime. The executor attributed the discrepancy to the published curve, but the critic checked the recorded circuit depths against the original method. In the power-law algorithm, these depths depend on $\beta .$ . For a requested error $\epsilon < \gamma$ , the requirement $\beta \ge 1 - \log \gamma / \log ( \epsilon / c )$ holds, where c is a constant. In the initial simulation, the executor selected the circuit sequence requiring the fewest oracle calls among candidates designed for other error values, without enforcing this bound at the requested precision. The recorded parameters confirmed violations of the bound, and exponentia schedules were retained below the expected crossover. The critic therefore identified the circuit-selection rule as a methodological defect and returned the afected sub-plan to the executor for local repair.

Following this review, the executor rebuilt each noisy-regime schedule with $\beta$ at the bound for the requested $( \epsilon , \gamma )$ keeping c = 1 throughout, and recomputed the afected results. The corrected schedule rule was recorded as a revised shared methodological interpretation, and the critic reviewed the recomputed results before accepting the afected sub-plan. As shown in Fig. SI 15b, the correction increased the estimated cost by roughly an order of magnitude near $\epsilon \sim 1 0 ^ { - 4 }$ , while the costs at large target errors remained unchanged. Across all three noise strengths, the schedule switches in Fig. SI 15c moved from below the evaluation interval to within it. The judge agent evaluated the completed reproduction and marked both target figures as passed.

Case study 3 (Paper #29): Experiment-driven diagnosis. This case illustrates how critic corrects failures in individual subplans by first analyzing their objectives and intermediate results to formulate diagnostic hypotheses, then deploying probing agents to test these hypotheses through small-scale experiments and guide targeted corrections. Background and baseline failure. Paper #29 [136] aims to improve the performance of parameterized quantum circuits (PQCs) by classically pretraining a matrix product state (MPS) and converting it into a quantum circuit as the initia ansatz for subsequent optimization. More concretely, the proposed method first uses MPS pretraining to obtain an MPS that closely approximates the target state and then performs MPS-to-circuit conversion to transfer the pretrained state into a PQC with minimal fidelity loss. As Fig. SI 16a shows, both DSH with DeepSeek V4.1 Flash and RA with the same base LLM failed at MPS pretraining and MPS-to-circuit conversion in their initial attempts. Specifically, the KL divergence between the pretrained MPS and the target state for both methods remained above 1 after 50 training sweeps, far exceeding the target range of [0.015, 0.03]. Similarly, the MPS-to-circuit conversion achieved transfer fidelities of only 0.9641 for DSH and 0.9777 for RA, both below the required 0.990. These discrepancies ultimately prevented successful reproduction of the PQC training results reported in the original paper.

![](images/2a801cf432e9bf9cd5c84041c474c384745188535ce2f5b6226d701315ac0107.jpg)

a  
![](images/67e717225b4f3f0ef5916d9e00815116b4b3673f2e7517a39d5620f831b31829.jpg)  
FIG. SI 16. Experiment-driven diagnosis and correction for Paper #29. a. Initial reproduction failures of DSH and RA before correction. (i), KL divergence during MPS pretraining with target accuracy. (ii), MPS-to-circuit transfer fidelities of DSH (0.9641) and RA (0.9777) with target accuracy. b. Workflow of experiment-driven diagnosis in Paper #29. Knowledge retrieved from residual memory guides the identification of potential implementation errors and insuficient optimization. Three probing agents conduct small-scale experiments to test these hypotheses and inform subsequent corrections by major voting. c. Comparison between the results reported in the original paper (left) and the corrected reproduction by RA (right) for diferent tasks. The corrected reproduction successfully recovers the improved training performance.

Experiment-driven correction. By retrieving relevant knowledge from its residual memory, the critic hypothesized that the failures in both MPS pretraining and MPS-to-circuit conversion could arise from implementation errors or insuficient variational optimization. To investigate these possibilities, the critic deployed three probing agents for each subplan to test each hypothesis from small-scale experiments, as shown in Fig. SI 16b. For MPS pretraining, two probing agents independently identified errors in the tensor contractions used for gradient computation by comparing independently computed reference gradients with those produced by the original code. For MPS-to-circuit conversion, one probing agent found that the original implementation failed to correctly follow the truncation-and-disentangling procedure for constructing circuit layers prescribed in the paper. Based on majority voting on these findings, RA corrected the errors in both subplans and finally reproduced the performance curves reported in the original paper correctly, as illustrated in Fig. SI 16c.