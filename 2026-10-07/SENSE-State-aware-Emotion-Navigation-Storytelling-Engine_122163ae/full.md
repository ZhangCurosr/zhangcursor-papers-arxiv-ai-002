# SENSE: State-aware Emotion Navigation Storytelling Engine

Yi Xia<sup>1,\*</sup>, Pablo Carrasco Velo<sup>1</sup>, Mudit Paliwal<sup>1</sup>, Ibrahim Khan<sup>2</sup>, Yifan Geng<sup>1</sup>, Mustafa Can Gursesli<sup>3</sup>, Juho Hamari<sup>3</sup>, and Ruck Thawonmas<sup>2</sup>

<sup>1</sup>Graduate School of Information Science and Engineering, Ritsumeikan University, Ibaraki, Japan,

yi.xia@ice.ci.ritsumei.ac.jp, {gr0797xi, gr0798hh, gr0797rx}@ed.ritsumei.ac.jp

<sup>2</sup>College of Information Science and Engineering, Ritsumeikan University, Ibaraki, Japan,

khan@fc.ritsumei.ac.jp, ruck@is.ritsumei.ac.jp

<sup>3</sup>Faculty of Information Technology and Communication Sciences, Tampere University, Tampere, Finland, {can.gursesli, juho.hamari}@tuni.fi

Corresponding author: yi.xia@ice.ci.ritsumei.ac.jp

Abstract—This paper presents SENSE, a state-aware framework for generating playable branching visual novels with multitrack emotional navigation. Integrating a state-based narrative architecture called MIND, a structure analyzer, and a path-aware context management module, SENSE produces narratives that are both structurally coherent and emotionally rich. From minimal high-level inputs, it generates multiple intersecting routes while preserving character consistency and narrative causality. Evaluations using LLM judges, affective metrics, and visual assessments indicate SENSE outperforms baselines in narrative diversity and robust asset integration, while preliminary human trials show directional improvements in emotional fidelity alongside comparable enjoyment.

Index Terms—Multi-Modal Content Generation, Multi-Agent Systems, Visual Novel, Narrative-Driven Games, Large Language Models

## I. INTRODUCTION

Visual novels (VNs) provide deeply interactive and immersive experiences by integrating narrative text, visual presentation, and branching choices. In such narrative-driven game design, emotional arcs are considered the logical core guiding plot progression and fostering player resonance [1], [2]. Although hierarchical “Outline-to-Detail” (O2D) paradigms have streamlined scriptwriting efficiency, large language models (LLMs) struggle with complex networked topologies [3]. The core challenge lies in the systematic graph-level failures LLMs encounter during long-term navigation, including missing passages, trap-like cycles, and causal asymmetries. Following that, a recent study by Puchalski et al., [4] attempted to rectify these defects through post-hoc, name-based repair loops. However, such approaches primarily focus on the physical path coherence and do not address the guidance of emotional intent. Without emotional orchestration, even a physically intact path is prone to emotional drift [2], causing the player’s experience to deviate from the intended emotional trajectory.

In order to address these challenges, we introduce the Stateaware Emotion Navigation Storytelling Engine (SENSE), a framework for scalable, emotionally salient, and visually aligned VN generation. SENSE decouples creative narrative logic from deterministic code, translating LLM-generated scripts into executable Ren’Py code while isolating pathdependent histories to prevent cross-branch interference. Its Memory and Intelligent Navigation Director (MIND) incorporates the Structural Thematic and Adaptive Router (STAR) to maintain thematic coherence while allowing branching divergence. State-based narrative mediation integrates historical path discrepancies into coherent psychological narratives, transforming abrupt emotional shifts into logical, emotionally resonant progressions. This state-aware, topology-validated architecture is designed to support structurally sound branching while mitigating emotional drift across routes.

The main contributions of this study are as follows:

• SENSE: a state-aware emotion navigation storytelling engine. We propose SENSE, an end-to-end engine that generates playable VNs from minimal inputs by explicitly modeling narrative state, branching history, player choices, and emotional trajectories.

• Emotion-guided branching synthesis via MIND. The MIND state-based architecture integrates macro-level emotional planning with path-aware script generation, guiding emotional progression while supporting causal consistency across branching narratives.

• Deterministic topology validation and context-aware script synthesis. SENSE supports structurally sound branching narratives through a mechanism called Structure Analyzer, which corrects node errors, cycles, and orphans. It further generates route-specific scripts via topology-driven, path-aware synthesis, improving context consistency across shared nodes.

• Executable integration with context-aware assets. We compile validated scripts and visuals into playable VNs, enhancing semantic fidelity and interactive coherence.

## II. RELATED WORK

## A. Emotional Arcs and Narrative Structure and Generation

Stories often follow predictable emotional trajectories, a concept rooted in early narrative theory and validated by sentiment analysis. Reagan et al. [1] showed that these arcs cluster into six fundamental shapes recurring across cultures and genres. Maintaining emotional trajectories is critical in interactive media, where player choices introduce variability. Frameworks such as PACE [5] use automated planning to keep players on target emotional curves, while systems like the Virtual Storyteller [6] generate story plots influenced by autonomous character actions, enabling the narrative affect to align with user preferences. While preliminary work, such as Wen et al. [2], applies basic emotional patterns to simplified branching games, integrating emotional trajectories into LLMbased generation of narrative-driven VNs with long texts and intersecting branches remains underexplored.

Hierarchical narrative generation, separating high-level plot planning from surface text realization, is a well-established paradigm in computational storytelling. Early symbolic systems, such as TAIL-SPIN [7], demonstrated the feasibility of top-down narrative construction. Modern neural approaches decompose generation into event-to-event and eventto-sentence levels [8], while Fan et al. [9] generate narrative sketches and expand them into full stories, improving coherence. Recent methods incorporate dynamic outlining to adjust high-level plans based on intermediate content [10], addressing the rigidity of static hierarchical approaches. PlotMachines [11] treats story outlines as sets of guiding phrases for generation. Together, these approaches illustrate the evolution of hierarchical strategies in automated storytelling. These hierarchical approaches mirror human writing and help mitigate long-range dependency issues, providing conceptual support for our SENSE pipeline that separates narrative planning, state-aware script synthesis, and executable integration.

## B. LLM-Based Branching Narrative Generation

LLMs have catalyzed new approaches to interactive narrative generation, producing branching stories via few-shot prompting and in-context learning. Recent systems such as “what-if” demonstrate that LLMs can generate and visualize branching narratives, enabling authors to explore “what-if” scenarios through mixed-initiative interfaces [12]. GENEVA [13] generates complete branching narratives from high-level descriptions and renders them as interactive graphs that authors can refine. TaleFrame [14] deploys locally-hosted LLMs fine-tuned on preference datasets to achieve fine-grained control over story generation. Despite these advances, LLMbased approaches still struggle with global consistency across branching paths [4], [15], producing structural errors such as edge omission, branch pruning, and hallucinated connections, which motivates the need for topology-aware validation and state-based narrative management.

## C. LLM-driven Multi-modal Narrative Generation

Recent studies show that LLMs can mimic convincing human social behaviors [16], inspiring multi-agent systems like AutoGen [17], CAMEL [18], and MetaGPT [19] that provide fundamental role-playing and coordination capabilities. Building on this, multi-agent debate protocols [20], [21] and collaborative narrative frameworks like IBSEN [22], Agents’ Room [23], and PatternTeller [24] leverage structured interactions. Despite structural gains, current hierarchical frameworks, such as [3], still have room for improvement in text-image alignment. Inspired by multi-agent paradigms, we adopt a role-specialized, agent-based workflow, in which distinct generative and evaluative components are orchestrated through an iterative multi-modal selection and validation loop to improve visual–textual alignment.

## III. METHODOLOGY

## A. System Overview

To fulfill the dual requirements of scalable narrative expansion and emotional expressiveness, SENSE is designed as an integrated end-to-end framework driven by a structural emotional blueprint and hierarchical narrative-to-code decoupling. As illustrated in Figure 1, the system autonomously generates playable VNs from minimal high-level inputs (theme, genre, and emotional arc type). It achieves this by anchoring the LLM’s creative synthesis to dynamic affective trajectories, while strictly separating the narrative prose from deterministic logical routing and code compilation. Audio generation is explicitly excluded from this study to limit modality scope, while retaining textual and visual asset generation.

The system follows seven decoupled phases, integrating MIND with deterministic structure analysis and path-aware context management:

a) The system expands the minimal inputs into a comprehensive world setting.

b) MIND utilizes its internal STAR Protocol to generate a macroscopic story outline embedded with emotional trajectory tags.

c) A deterministic safeguard mechanism called the Structure Analyzer processes this outline to detect and resolve logical deadlocks and output a valid routing topology.

d) The system derives granular chapter outlines from the story structure processed by the Structure Analyzer.

e) MIND internalizes path-dependent histories through state-based narrative mediation to generate engineagnostic, tag-annotated intermediate chapter scripts.

f) The pipeline produces context-aware image assets derived from the world setting and chapter scripts.

g) A deterministic parser integrates these visual assets with the tagged scripts to synthesize executable engine code, ensuring robust syntactic validity.

## B. The MIND state-based Architecture

The MIND module serves as the inference-time orchestration layer and state-based narrative controller of SENSE, bridging the gap between minimal input and granular narrative execution. Rather than modifying the underlying LLM architecture, MIND orchestrates inference through structured prompt-based constraints and dynamic context injection. Following that, to support emotional trajectory guidance and logical consistency, MIND operates through a dual-layered governance strategy: macroscopic planning via the internal STAR Protocol and microscopic synthesis through Path-Aware Narrative Synthesis.

![](images/41fa39daf0605f13aeda94d6424505dd6ffa5989e318f1496f6862db2593ce00.jpg)  
Fig. 1: Overall architecture of the SENSE system.

1) Macroscopic Governance: The STAR Protocol: In the initial phase, MIND employs the STAR Protocol to establish a foundation for thematically coherent narrative expansion. This protocol is operationalized through prompt-based constraint injection, wherein the LLM is furnished with a strict logical framework embedded within its system instructions. The protocol implements a dual-governance strategy to regulate the global emotional trajectory. The Spine, or primary narrative path, is governed by hard constraints: MIND retrospectively derives narrative logic to conform to the given emotional arc. In contrast, the Limbs, or branching paths, are governed by autonomous reasoning, wherein emotional trajectories are dynamically generated through the Reaction instruction specified in the system prompt. This instruction orchestrates the causal interplay between internal character factors (e.g., psychological states and value systems) and external contextual factors (e.g., environmental pressures and resource scarcity). Based on these emotional trajectories, explicit emotional tags (i.e., Rise or Fall) are assigned to each narrative node, thereby enabling the STAR Protocol to construct a structured emotional blueprint that encodes predetermined outcome constraints. This mechanism supports global emotional variation across narrative branches, while mitigating the risk of model-induced affective monotony.

2) Microscopic Logic Safety: The Schrodinger and Col-¨ lision Protocols: To mitigate logical hallucinations during branching synthesis, MIND incorporates two specialized execution modes defined within the script generation prompts. The activation of these modes is strictly controlled by the system’s underlying code, which dynamically constructs the user prompt based on narrative routing conditions.

When the code’s routing logic detects structural ambiguity (specifically, when multiple diverging player choices route to the same subsequent chapter), the system conceptualizes this convergence as a narrative superposition. To handle this state where the actual player choice remains unobserved, it dynamically injects “POSSIBLE PLAYER CHOICES” into the user prompt, triggering Mode A (The Schrodinger Rule) ¨ . Named after the quantum mechanical thought experiment, this rule dictates that the generated narrative must remain valid across all potential player histories without collapsing into a logical contradiction. To maintain this consistency before writing the merged scene, this triggered rule enforces a protocol across three dimensions:

• Physical Check: If physical states (e.g., item possession or health) are in conflict, the system must not describe the specific item or body part, focusing instead on the environment.

• Relational Check: If past interactions conflict, NPC reactions must be described as “Unreadable,” “Complex,” or “Silent” to ensure sentiment validity.

• Informational Check: If the player’s knowledge state is ambiguous, the narrative focus shifts to the overarching mystery or goal rather than specific solutions.

If any of these dimensions remain in conflict, the prompt makes the LLM shift to an internal monologue about general consequences or introduces a sudden external event to drive the story forward without logical friction.

Conversely, Mode B (The Collision Rule) is activated when the routing algorithm identifies a unique branch or ending. This acts as the narrative observation that collapses the superposition. The code dynamically constructs a user prompt that explicitly states “The player DEFINITELY chose: [Action]” or contains a “CRITICAL INSTRUCTION”. This mode completely ignores ambiguity and enforces Specific Causality. The system must explicitly reference the specific choice (e.g., describing blood on a weapon if the player chose to attack, or shame if they chose to run). By colliding the chapter outline with the specific choice’s tone and outcome, MIND creates a unique divergence that is logically robust and responsive to player choices.

## C. Structure Analyzer

During the macroscopic narrative planning phase, specifically the outline generation, LLMs often struggle with strict graph-theoretic consistency. We find that even when adopting a baseline-inspired two-step generation pipeline (first drafting the raw narrative events in natural language and subsequently converting them into a structured JSON format), the LLM still frequently hallucinates node identifiers (IDs) or generates illogical structural loops. To address this limitation, SENSE introduces a deterministic safeguard mechanism called the Structure Analyzer, which consolidates the story graph before chapters and scripts synthesis.

This mechanism operates through three core deterministic processes:

• Heuristic Priority-Based Node Resolution: LLMs frequently mismatch numerical chapter IDs during the JSON conversion step while accurately retaining semantic titles. To rectify broken transition links, the analyzer employs a multi-dimensional target resolution algorithm. Instead of relying solely on exact ID matching, it evaluates potential target nodes through a strict priority sequence: Full Title Match, Explicit ID Match, Cleaned Title Match (filtering out ID and emotion tags), and finally, a composite Branch and Keyword overlap. This heuristic approach supports robust edge reconstruction even when the LLM’s numerical indexing fails.

• Cycle Detection and Heuristic Breaking: Non-linear narrative generation is highly susceptible to structural cycles (e.g., self-loops or complex backward loops) that can cause infinite rendering errors. The system utilizes a Depth-First Search (DFS) algorithm to detect all topological cycles. Rather than randomly severing connections, it breaks cycles intelligently by computing a Connection Quality Score for each edge involved in a loop. This heuristic scoring system incorporates narrative domain knowledge, rewarding logical transitions, such as natural index progression or connections within the same branch designation, while heavily penalizing structural regressions. By evaluating these scores, the algorithm deterministically isolates and breaks the weakest link, thereby preserving the most logical narrative flow.

• Generalized Orphan Resolution and Topological Preservation: In the final stage, the algorithm systematically distinguishes between true orphans (nodes completely lacking incoming routes) and falsely flagged orphans (disconnected sub-graphs retaining internal parentchild links). To rescue the latter, the system executes a global reverse-validation of all narrative pointers. By evaluating implicit connections and employing type-agnostic identifier comparisons to mitigate data type hallucinations (e.g., integer-string confusion), the algorithm dynamically reconstructs missing edges and updates the traversal maps. Ultimately, only confirmed true orphans are pruned. The system subsequently utilizes this fully validated graph structure to autonomously rectify the original narrative outline, yielding a structurally validated foundation for subsequent script synthesis.

## D. Topology-Driven Context Injection and Script Synthesis

SENSE programmatically activates microscopic logic protocols by evaluating the structural properties of target nodes and dynamically constructing prompt payloads. This mechanism is designed to provide the generative model with specific contextual triggers based on the graph’s topological identity.

• Schrodinger State Maintenance (Convergence Logic):¨ For chapters with a topological in-degree ≥ 2 representing narrative convergence, the system aggregates all potential player choices, including their labels and underlying router logic, into the prompt payload under the flag POSSIBLE PLAYER CHOICES (SCHR<sup>¨</sup>ODINGER STATE). This serves as the deterministic trigger for Mode A (The Schrodinger Rule)¨ , established in Section III-B2.

• Collision and Ending Divergence (Divergence Logic): For terminal nodes identified in the global Identity Map, the system supports ending divergence by generating unique variant files (e.g., \_End\_xx). By declaring a definitive player action in the prompt, the system triggers Mode B (The Collision Rule). This collapses the narrative superposition, requiring the model to detail the specific consequences associated with that singular choice.

• Path-Aware Context and Sequential Synthesis: SENSE synthesizes content sequentially along individual routes, creating unique context signatures for shared chapters conditioned on their ancestor sequences. The system builds a Path-Aware Context by retrieving the full textual transcripts of the current path. This allows shared topological nodes to produce distinct script variants, isolating narrative causality and avoiding inter-route interference.

## E. VN Image Assets Generation

We propose an automated, identity-preserving generation pipeline (see Fig. S1 in Section S7 of the supplementary file in the project repository; see Section IV-F for access information) designed to synthesize emotionally aligned visual assets while mitigating character drift. In contrast to the baseline approach, where the LLM generates novel prompts without contextual awareness of prior iterations [3], which often exacerbates identity instability, our method relies on the deterministic extraction of structured visual attributes from the world-setting configuration to establish a strict Base Identity. For narrative backgrounds, transient visual cues parsed from chapter scripts are systematically merged with the parent location’s metadata to maintain contextual grounding.

To generate dynamic character expressions while minimizing identity degradation, the pipeline employs an image-toimage approach. An LLM maps abstract narrative tags (e.g., [anger]) to the closest match within a 34-emotion dictionary derived from the Ekman and GO frameworks [25], [26] to extract corresponding facial descriptors for prompt construction. Crucially, an IP-Adapter [27] enforces strong semantic conditioning directly from the Base Identity. In parallel, an automated object detection algorithm (YOLOv8) [28] isolates and inpaints solely the facial region. This architecture ensures that emotional variations are applied exclusively to facial features without altering the character’s core structure or attire.

In order to regulate output quality and prevent character drift, the pipeline utilizes a vision-language model (VLM) within an iterative generative-discriminative loop. During each iteration, the generative model outputs a batch of candidate images $( N = 3 )$ . The VLM evaluates these candidates against predefined quality criteria, which specifically include verifying strict consistency with the Base Identity alongside general visual fidelity. If the best-scoring image fails to satisfy the criteria, the VLM generates textual correction feedback. This feedback is appended to the prompt for the subsequent iteration. The cycle repeats until the image passes validation or reaches the maximum iteration limit $( T _ { m a x } = 2 )$

## F. Script Integration and Backend Generation

This module bridges the gap between the intermediate text-based VNs script and executable runtime artifacts by generating engine-specific outputs for different backends. It operates on a shared script representation and applies a unified script-level integration strategy, enabling the same narrative content to be deployed across multiple execution environments. The current implementation is demonstrated using a Ren’Py integration. The intermediate representation is reusable, and similar conversion logic can be applied when adapting the system to other platforms, such as Unity, with platform-specific integration stages.

## G. Game script conversion and asset integration

This stage transforms the intermediate text-based VNs script into an executable Ren’Py script, binding all pre-generated visual assets to their corresponding narrative elements. For this stage, we developed a series of parsers utilizing regular expressions to ensure that the resulting script can be executed directly without manual intervention, achieving high accuracy in the text-to-game conversion process.

In this stage, two distinct parsers operate sequentially to achieve this conversion. The first parser integrates previously generated images into the chapter structure. To facilitate asset integration, it identifies labels in the script and assigns asset identifiers, creating the links required for further conversion into a game script. The second parser processes the script generated by the first parser into a playable Ren’Py game.

This script begins by initializing all assets within the Ren’Py “script” file, a necessary step enabling Ren’Py to display images based on asset names. Additionally, this script implements core game mechanics, including interactions and decision points where the narrative branches into different routes. It utilizes the background labels provided by the processed script to display the correct background image and matches character names from dialogues to their corresponding asset images for character display. To prevent character overlap, a queuing mechanism limits the display to a maximum of three characters at any given time, positioned on the left, right, and center of the screen. When a fourth character with dialogue is encountered, the script replaces the oldest character image on screen, implementing a queue-based replacement strategy.

## IV. EXPERIMENTS

## A. Experimental Setup

1) Frameworks and Generation Setup: We evaluate the proposed SENSE framework against two frameworks: (1) Baseline: the recent hierarchical VN generation framework by Zhang et al., which lacks explicit affective modeling [3]; and (2) WithoutMIND: an ablated SENSE pipeline stripped of the MIND architecture. We selected “Climate Change” as the central theme for all 54 generated VNs, spanning three distinct genres with 6 VNs per framework. These three genres were specifically selected because they are distantly related and heterogeneous, ensuring a rigorous test of the framework’s generalization capability across diverse narrative spaces. To standardize evaluation across frameworks, we instructed the LLM during outline generation to target a maximum of 6 chapters per narrative route. Despite LLMs’ inherent variance in strictly following numerical limits, this soft constraint effectively bounded the narrative scale, ensuring comparable lengths for fair cross-modal assessment. To verify SENSE’s affective saliency, we assigned one of six distinct emotional trajectories to each of the 6 VNs within every genre for SENSE frameworks, whereas the baseline operated without explicit constraints.

The choice of climate change as the main theme is strategically grounded in its dual nature as both a complex and multifaceted global issue requiring exceptional narrative complexity and a representative application for Serious VNs [29]. By demanding high logical coherence and precise emotional pacing to effectively engage players and foster social awareness, this theme provides a rigorous testbed for our MIND mechanism’s ability to balance structural constraints with prose rendering. Furthermore, utilizing this theme ensures our results provide a robust foundation for future extensions into the narrative experience and efficacy of AI-generated stories in serious game contexts.

2) Comparative Integrity and Route Extraction Protocol: To ensure a controlled comparison, all frameworks utilized GPT-5-mini as the underlying LLM for text and script generation. Aside from this model alignment, the baseline was executed according to its original default settings. During preliminary testing on the baseline, we observed instances of technical limitations, such as outline inaccuracies and syntax bottlenecks. Although its script validation mechanism is designed to perform error correction using Ren’Py lint feedback, the authors acknowledge it resolves approximately 76% of syntax errors rather than eliminating them entirely [3]. Following the original implementation, we limited this automated correction process to three iterations.

We developed a dedicated parsing script grounded in Ren’Py syntax to isolate storytelling quality from code execution failures. This script traverses the branching logic of all methods to reconstruct player-facing text into pure-text narrative routes. This protocol acts as a lenient extraction for the baseline by bypassing non-fatal execution errors to salvage structurally readable content. This approach evaluates the baseline at its peak narrative performance rather than penalizing its architectural limitations.

## B. Primary Evaluation: LLM-as-a-Judge Protocol

Conventional metrics like BLEU and ROUGE rely primarily on surface-level lexical overlap. However, such metrics often fail to capture the structural and creative dimensions crucial for evaluating open-ended storytelling. Consequently, we adopt the LLM-as-a-Judge paradigm as a validated and scalable proxy for narrative assessment. Following the evaluative framework established by Chhun et al. [30] and subsequently adopted in [31], we assess the generated results across six orthogonal criteria: Relevance (RE), Coherence (CH), Empa thy (EM), Surprise (SU), Engagement (EG), and Complexity (CX). To ensure evaluative consistency, we conduct three independent evaluation trials for each narrative route at a zerotemperature setting (T = 0) using distinct random seeds. The resulting scores, measured on a 5-point Likert scale, are subsequently aggregated into an Overall (OV) quality metric to mitigate the impact of stochastic variability. This holistic methodology serves to quantify the comparative performance of the SENSE framework against the baselines across diverse narrative dimensions.

Given that each VNs case yields multiple narrative routes, we conduct route-level evaluations using two distinct sampling pools: AllMean, which incorporates the complete distribution of successfully generated routes to reflect system-wide expected quality, and Top-3 (or all, if $N < 3 )$ , which isolates the highest-scoring routes per case to assess upper-bound creative potential. Evaluating this Top-3 subset is statistically imperative for an equitable comparison. Baseline methods frequently produce malformed scripts, resulting in severely restricted route yields. Consequently, relying solely on the AllMean pool disproportionately penalizes high-yield generative frameworks. A baseline may achieve a deceptively high mean from isolated serendipitous successes, whereas a robust system’s comprehensive distribution inherently includes average-quality routes that dilute the aggregate score. By adopting the Top-3 (“Best-of-N”) alignment, we explicitly control for these sample size discrepancies, mitigating statistical dilution and enabling a direct comparison of the frameworks’ upper-bound creative capabilities.

## C. Quantitative Analysis of Affective Salience

Complementing the primary evaluation of story-level narrative quality, we conduct a fine-grained affective analysis to investigate the framework’s capacity to enhance emotional intensity and affective richness. This evaluation utilizes the DeBERTa-based model of Christ et al. (2024), which is specifically optimized for modeling continuous valence (V) and arousal (A) trajectories in narratives using a context-aware sliding window with a radius of four sentences to integrate surrounding emotional context [32]. The model demonstrates strong empirical reliability, with reported Concordance Correlation Coefficients (CCC) of 0.8221 for valence and 0.7125 for arousal.

Following the conceptualization of affective salience in [33], which defines it as the valence distance from neutral in either direction, we adopt this conceptualization to quantify emotional depth through a two-stage analytical pipeline. For each sentence $i ( i = 1 , . . . , n )$ , we compute its affective deviation $| E _ { i } - 0 . 5 |$ and identify its salience status based on a predefined non-salient range of [0.35, 0.65], where $E \in V , A$ denotes the affective dimensions of valence and arousal, respectively. These sentence-level values are subsequently aggregated into two route-level metrics:

• Affective Salience (AS): quantifies the average intensity of emotional expression by computing the mean absolute deviation of all sentences in a route from the neutral baseline of 0.5, effectively measuring the narrative’s overall emotional intensity:

$$
A S = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \left| E _ { i } - 0 . 5 \right|\tag{1}
$$

• High-Salience Proportion (HSP): identifies affective richness by calculating the percentage of sentences whose valence or arousal values fall outside the non-salient range:

$$
H S P = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \mathbb { 1 } ( E _ { i } > 0 . 6 5 \vee E _ { i } < 0 . 3 5 ) \times 1 0 0 \%\tag{2}
$$

where $\mathbb { 1 } ( \cdot )$ is the indicator function that outputs 1 if the affective value $E _ { i }$ (where $E \in \{ V , A \} )$ is salient and 0 otherwise.

D. Visual Evaluation: Cross-Modal Alignment and Character Identity Preservation

To evaluate the quality of the generated visual assets, we assess both cross-modal text-image alignment and character identity preservation. Following Lin et al. [34], we adopt VQAScore via the CLIP-FlanT5-XXL model as one of our metrics to quantify prompt alignment, computed as the probability of answering “Yes” to “Does this figure show prompt?”. To directly measure cross-expression character persistence, we adopt two image-to-image similarity metrics from the evaluation protocol introduced in [35]: DINO and CLIP-I. Specifically, we compute the cosine similarity between the embeddings of each generated expression variant and its corresponding base character reference image, using self-supervised ViT-S/16 DINO and CLIP visual encoders, respectively. Together with VQAScore, these metrics evaluate whether SENSE preserves robust character identity across emotional variations while faithfully capturing prompt semantics.

## E. Statistical Methodology and Significance Testing

To evaluate statistical significance across textual and visual dimensions, we implement an adaptive pipeline determined by the Shapiro-Wilk normality test. For overall evaluation across all experimental groups, we employ a one-way ANOVA or the Kruskal-Wallis H test as an omnibus measure to identify overall differences. To isolate the contribution of specific modules, we conduct pairwise comparisons using Welch’s ttests or Mann-Whitney U tests. Given our ablation framework, these pairwise evaluations are treated as a priori planned contrasts [36]; consequently, we report unadjusted p-values to mitigate the Type II error inflation inherent in conservative post-hoc corrections like Bonferroni.

TABLE I: Comparison of Effective Narrative Route Yield by Genre.
<table><tr><td>Narrative Genre</td><td>Baseline</td><td>WithoutMIND</td><td>Full SENSE</td></tr><tr><td>Historical</td><td>12</td><td>30</td><td>75</td></tr><tr><td>Romance</td><td>48</td><td>39</td><td>71</td></tr><tr><td>Sci-Fi</td><td>39</td><td>25</td><td>80</td></tr><tr><td>Total Yield</td><td>99</td><td>94</td><td>226</td></tr></table>

## F. Implementation Details

Narrative generation for all evaluated frameworks utilized gpt-5-mini-2025-08-07. Textual evaluation via the LLM-asa-Judge protocol used three open-weight models (Gemma 3 12B, Mistral 7B v0.3, and Qwen 3 14B). For visual assets, images were generated using the waiIllustriousSDXL checkpoint in ComfyUI, while the VLM-guided generation loop and visual alignment evaluation utilized qwen3-vl:32b-instruct. Generated scripts and multimedia assets were compiled using the Ren’Py engine. All additional materials, including the supplementary file, source code, data, prompts, and results, are available in the project GitHub repository.<sup>1</sup>

## V. RESULTS

## A. Framework Robustness and Complexity: Narrative Route Yield

Evaluating narrative generation frameworks requires ensuring the executability and structural integrity of multi-branching VNs. Beyond robustness, generating complex, intersecting narrative routes reflects each framework’s structural richness and narrative complexity, quantified using the parsing protocol described in Section IV-A2. We define Narrative Route Yield as the total number of distinct narrative routes produced by each framework. Table I summarizes the results, showing substantial differences in both robustness and narrative complexity among the tested frameworks: Full SENSE, its ablated variant withoutMIND, and an external Baseline. For complete transparency, the exhaustive route statistics, along with the raw generated Ren’Py script files, are provided in the project repository.

The probabilistic Baseline framework produced only 99 narrative routes, due to inherent limitations in its generation process combined with errors in the LLM-generated Ren’Py scripts. The ablated WithoutMIND framework yielded slightly fewer routes, generating only 94. This reduction stems from two main factors. First, the ablation method removes certain cycles and isolated nodes, which could have led to logically inconsistent routes during subsequent generation. Second, assigning a single emotional trajectory inherently limits route diversity. In contrast, Full SENSE framework generated 226 narrative routes, representing a more than 2.2-fold increase over Baseline. This improvement primarily reflects its ability to handle multiple emotional trajectories, together with the integrated Path-Aware Context mechanism and Schrodinger¨ Rule for managing narrative convergences.

TABLE II: Top-3 Narrative Quality Comparison. Higher values indicate better performance.
<table><tr><td>Method</td><td>RE</td><td>CH</td><td>EM</td><td>SU</td><td>EG</td><td>CX</td><td>OV</td></tr><tr><td>(1) Baseline</td><td>4.81</td><td>4.67</td><td>4.43</td><td>3.45</td><td>4.68</td><td>4.742,3†</td><td>4.47</td></tr><tr><td>(2) WithoutMIND</td><td>4.89</td><td>4.72</td><td>4.48</td><td>3.31</td><td>4.75</td><td>4.27</td><td>4.42</td></tr><tr><td>(3) Full SENSE</td><td>4.85</td><td>4.75</td><td>4.581,2</td><td>3.562</td><td>4.76</td><td>4.532</td><td>4.522</td></tr></table>

Note: Values denote the macro-average scores across Gemma-3 (12B), Mistral (7B), and Qwen-3 (14B) evaluators. To maintain readability, standard deviations, 95% Confidence Intervals (CIs), and effect sizes (Cohen’s d) are exhaustively detailed in the supplementary file in the project repository.  
Statistical Significance: Superscripts indicate a statistically significant improvement $( p < 0 . 0 5 )$ over the corresponding method number. Significance was confirmed by a majority consensus (at least 2 of 3 judges) using independent Mann-Whitney U tests, strictly protected by a prior significant Kruskal-Wallis omnibus test to guard against FWER inflation.  
<sup>†</sup> The significantly higher CX score in Baseline is an artifact of unconstrained logical hallucinations rather than coherent narrative depth (see Sec. VI).

## B. Narrative Quality

Using the Top-3 protocol described in Section IV-B, both WithoutMIND and Full SENSE frameworks generated all 54 evaluable routes, whereas the probabilistic Baseline produced only 43 due to generation and script errors, reflecting its structural and script limitations. Table II reports the aggregated Top-3 scores, with comprehensive statistical details (including 95% CIs, standard deviations, and individual judge p-values) provided in the supplementary file in the project repository.

Crucially, Full SENSE achieved the highest Emotional Fidelity (EM: M = 4.58, 95% CI [4.53, 4.63]<sup>1,2</sup>) and Overall Quality (OV: M = 4.52, 95% CI [4.48, 4.56]<sup>2</sup>), significantly outperforming both Baseline and WithoutMIND configurations. RE and CH differences were not statistically significant, though WithoutMIND and Full SENSE consistently exceeded Baseline in CH. The Baseline reached the highest per-route Complexity (CX: $M = 4 . 7 4 ^ { 2 , 3 \dagger }$ , 95% CI [4.64, 4.84]) but suffered from limited route coverage. WithoutMIND dropped to M = 4.27 (95% CI [4.17, 4.37]) due to structural restrictions and single-emotion assignment, whereas Full SENSE successfully restored Complexity $( M = 4 . 5 3 ^ { 2 }$ , 95% CI [4.41, 4.65]) and Surprise $( M ~ = ~ 3 . 5 6 ^ { 2 }$ 95% CI [3.43, 3.69]) through its multi-branch, multi-emotion handling mechanism. This observation is further supported by the AllMean robustness analysis (supplementary file in the project repository, Section S1.3), which evaluates the complete set of generated routes for each visual novel rather than only its Top-3 ranked routes. Despite the substantially larger evaluation set (226 routes), Full SENSE maintained high average narrative quality across all generated routes.

## C. Affective Salience and Narrative Pacing Results

Using the sentence-level valence (V ) and arousal (A) predictions from the DeBERTa-based model described in Section IV-C, we computed route-level Affective Salience (AS) and High-Salience Proportion (HSP). These metrics measure how frequently and strongly narrative sentences escape the neutral range [0.35, 0.65], providing an objective assessment of emotional dynamics and narrative pacing (shown in Table III).

TABLE III: Route-level Affective Metrics (Mean and 95% CI). Higher values indicate stronger affective salience.
<table><tr><td rowspan="2">Method</td><td colspan="2">Valence (Polarity)</td><td colspan="2">Arousal (Intensity)</td></tr><tr><td>AS</td><td>HSP (%)</td><td>AS</td><td>HSP (%)</td></tr><tr><td>(1) Baseline</td><td>0.084 [0.082, 0.086]</td><td>15.3 [14.3, 16.3]</td><td> $0 . 2 4 6 ^ { 2 }$  [0.244, 0.248] [86.1, 87.6]</td><td> $8 6 . 9 ^ { 2 , 3 }$ </td></tr><tr><td>(2) WithoutMIND</td><td> $0 . 0 9 6 ^ { 1 }$  [0.093, 0.099] [19.8, 22.1] [0.239, 0.244]</td><td> $2 1 . 0 ^ { 1 }$ </td><td>0.241</td><td>84.6 [83.9, 85.3]</td></tr><tr><td>(3) Full SENSE</td><td> $\mathbf { 0 . 1 0 3 ^ { 1 , 2 } }$   $[ 0 . 1 0 1 , 0 . 1 0 5 ]$ </td><td> $2 3 . 6 ^ { 1 , 2 }$   $[ 2 2 . 8 , 2 4 . 4 ]$ </td><td>0.244 [0.242, 0.245] [84.6, 85.5]</td><td>85.0</td></tr></table>

TABLE IV: Visual Evaluation Results Across Frameworks (Mean). Higher values indicate better performance.
<table><tr><td>Metric / Category (1) Baseline (2) WithoutMIND</td><td></td><td></td><td>(3) Full SENSE</td></tr><tr><td colspan="4">Semantic Alignment (VQAScore ↑):</td></tr><tr><td>Backgrounds</td><td>0.590</td><td> $0 . 7 5 0 ^ { 1 , 3 }$ </td><td> $0 . 6 7 2 ^ { 1 }$ </td></tr><tr><td>Characters</td><td>0.601</td><td> $0 . 8 6 2 ^ { 1 }$ </td><td> $0 . 8 6 2 ^ { 1 }$ </td></tr><tr><td>Total Mean Score</td><td>0.595</td><td> $\mathbf { 0 . 8 1 5 ^ { 1 , 3 } }$ </td><td> $0 . 8 0 3 ^ { 1 }$ </td></tr><tr><td colspan="4">Character Identity Preservation ↑:</td></tr><tr><td>CLIP-I</td><td>0.824</td><td> $0 . 9 4 2 ^ { 1 }$ </td><td>0.9581</td></tr><tr><td>DINO</td><td>0.611</td><td> $0 . 9 3 3 ^ { 1 }$ </td><td> ${ \bf 0 . 9 7 2 ^ { 1 , 2 } }$ </td></tr></table>

Note: Values in bold green indicate the highest mean score in each row. Superscripts Note: Brackets denote 95% CI. Values in bold green indicate the highest meanindicate statistically significant improvement $( p \ < \ 0 . 0 5 )$ over the corresponding score in each column. Superscripts indicate statistically significant improvementmethod number via planned ablation contrasts (Mann-Whitney U tests). Detailed $( p \ < \ 0 . 0 5 )$ over the corresponding method number via Mann-Whitney U tests.test statistics and 95% confidence intervals (CIs) are provided in the supplementary Detailed test statistics and Cohen’s d effect sizes are provided in the supplementaryfile in the project repository. file in the project repository.

Valence Salience: Full SENSE framework achieved the highest Valence AS (M = 0.103, 95% CI [0.101, 0.105]) and HSP (M = 23.6%, 95% CI [22.8, 24.4]), significantly exceeding both Baseline and WithoutMIND $( p < 0 . 0 0 1 )$ . In contrast, Baseline narratives exhibited low emotional variation (Valence HSP 15.3%). The increase from WithoutMIND to Full SENSE indicates that enabling multiple intersecting trajectories allows higher affective salience without reducing structural diversity. Note that the 95% CIs between Baseline and Full SENSE do not overlap, providing robust visual confirmation of statistical significance.

Arousal Salience: In the arousal dimension, the scores remained consistent with the base model’s inherent adjustments and were not significantly affected by the proposed framework. No statistically significant difference in Arousal AS was observed between Full SENSE and Baseline $( p = 0 . 1 0 5 )$ . While Baseline exhibited a slightly higher Arousal HSP (86.9%) than Full SENSE (85.0%, $p < 0 . 0 0 1 )$ .

## D. Visual Evaluation Results

Following Section IV-D, we assessed semantic alignment (VQAScore) across 23,291 total generated images (12,686 for Full SENSE; 7,405 for WithoutMIND; 3,200 for Baseline). As Table IV shows, Full SENSE achieved a character VQAScore of 0.862, showing no significant difference from WithoutMIND $( p \ : = \ : 0 . 5 4 5 )$ but significantly outperforming Baseline $( p < 0 . 0 0 1 )$ , whereas WithoutMIND scored higher on backgrounds $( p ~ < ~ 0 . 0 0 1 )$ . One possible speculation is that MIND incorporates additional affective guidance, shifting the focus of image-generation prompts from concrete object descriptions to abstract atmospheric elements. Additionally, we evaluated character identity preservation (CLIP-I, DINO) specifically on the character asset subsets (7,981 for Full SENSE; 3,855 for WithoutMIND; 1,306 for Baseline). Full SENSE achieved the highest CLIP-I and DINO. It significantly outperformed Baseline $( p \textless 0 . 0 0 1 )$ . Compared to Without-MIND, Full SENSE demonstrated significant improvement in identity retention as measured by DINO $( p < 0 . 0 0 1 )$ while performing comparably in CLIP-I $( p = 0 . 4 1 4 )$ . This confirms SENSE effectively retains character consistency even across diverse emotional states.

![](images/e70798f71a0757188a380bf1a3e2564df5151278c4ae9c06c94a9b0232150a87.jpg)  
Fig. 2: Gameplay captures of the SENSE-generated VNs. The subfigures illustrate (a) integration of assets within a sci-fi setting; (b) interactive choice menus; (c) multi-asset integration in a romance narrative; (d) the same character sprite in a different background.

## E. End-to-End System Deployment

We deployed the SENSE framework as a fully playable VNs using the Ren’Py engine. Figure 2 shows representative gameplay captures. The system executes the full pipeline without manual intervention. It demonstrates robust integration of narrative, visual, and interactive components, while preserving semantic alignment between characters, backgrounds, and story events.

## F. User Study

To complement the automatic evaluation, we conducted a user study using a representative Romance scenario, specifically a median-performing case (ranked third among the six generated Romance narratives based on LLM-as-Judge scores), given the genre’s widespread adoption in visual novels. As the baseline-generated Ren’Py project was not directly executable, we reimplemented the Ren’Py project and conducted script modifications required for interactive evaluation while preserving the original generated narrative and branching structure. Prior to the study, informed consent was obtained from all participants. A total of 20 participants completed the study. Two participants were excluded because their reading speeds were identified as extreme outliers, suggesting that the visual novel content was unlikely to have been read attentively, leaving a final sample of 18 participants for analysis (mean age = 23.2 years, $\mathrm { S D } = 2 . 9 )$ . To establish baseline familiarity with the medium, participants also selfreported their prior visual novel experience on a 7-point scale.

TABLE V: Pairwise Bootstrap Resampling Results $( M _ { d i f f }$ and 95% CI)
<table><tr><td>Metric</td><td>Overall (N = 18)</td><td>Exp. (n = 8)</td></tr><tr><td>Enjoyment</td><td> $0 . 5 6 \ [ - 0 . 2 2 , 1 . 3 9 ]$ </td><td> $0 . 3 8 \ [ - 0 . 5 0 , 1 . 5 0 ]$ </td></tr><tr><td>Boredom  $\mathrm { ( R e v . ) }$ </td><td> $- 0 . 1 1 \dot { { \left[ - 0 . 6 7 , 0 . 4 \dot { 4 } \right] } }$ </td><td> $- 0 . 2 5 \dot { [ - 1 . 1 3 , 0 . 6 \dot { 3 } ] }$ </td></tr><tr><td>Theme Relevance</td><td>0.50 [−0.44, 1.39]</td><td> $\mathbf { 1 . 1 3 \ [ 0 . 2 5 , 2 . 2 5 ] ^ { * * ^ { \circ } } }$ </td></tr><tr><td>Genre Relevance</td><td> $\mathbf { 0 . 6 7 \ [ 0 . 1 1 , 1 . 3 9 ] ^ { * } }$ </td><td> $0 . 3 8 \bar { [ - 0 . 1 3 , 0 . 8 \bar { 8 } ] } ^ { \dagger }$ </td></tr><tr><td>Image Consistency</td><td> $0 . 7 2 \dot { [ - 0 . 0 6 , 1 . 5 6 ] } ^ { \dagger }$ </td><td> $0 . 6 3 \ [ - 0 . 2 5 , 1 . 6 3 ]$ </td></tr><tr><td>Narrative Coherence</td><td>0.56 [−0.11, 1.28]†</td><td> $0 . 2 5 \ [ - 0 . 3 8 , 1 . 0 0 ]$ </td></tr></table>

Note: $\overline { { M _ { d i f f } } }$ is the mean difference. ‘Exp.’ denotes players with VN experience ≥ 5. ${ \dag } _ { p } < 0 . 1 0 ,$ ${ ^ { * } p } < 0 . 0 5 ,$ $^ { * * * } p < 0 . 0 0 1$ (unadjusted).

Participants experienced both versions in randomized order. After each session, they completed a post-session questionnaire assessing enjoyment, boredom, theme relevance, genre relevance, image consistency, and narrative coherence on a 7-point Likert scale, with the boredom item reverse-scored during statistical analysis. After experiencing both versions, participants additionally identified their preferred version, the version that they perceived as having a more perceptible emotional arc, and provided qualitative feedback. Additional implementation and study details are provided in the supplementary file in the project repository.

To robustly handle discrete ties without relying on rankbased tests, we analyzed responses using pairwise bootstrap resampling (n = 10, 000) across two groups: the overall sample $( N \ = \ 1 8 )$ and a subgroup of experienced players (experience $\mathrm { s c o r e } \geq 5 , \ N \ = \ 8 )$ . Detailed mean differences and 95% confidence intervals are provided in Table V. Unadjusted p-values were used not only because these are planned comparisons, but also because strict corrections would severely inflate Type II errors in a small sample size. Overall, participants rated our method higher in Genre Relevance $( M _ { d i f f } = 0 . 6 7 , p = 0 . 0 1 3 )$ , with marginal trends in Image Consistency $( M _ { d i f f } ~ = ~ 0 . 7 2 , p ~ = ~ 0 . 0 5 3 )$ and Narrative Coherence $( M _ { d i f f } = 0 . 5 6 , p = 0 . 0 7 8 )$ . Crucially, the experienced subgroup reported significantly higher Theme Relevance $( M _ { d i f f } = 1 . 1 3 , p < 0 . 0 0 1 )$ , suggesting that greater player experience heightens sensitivity to thematic alignment. Furthermore, in a forced-choice assessment, 66.7% (12/18) of participants identified our method as having a clearer emotional arc, while preferences for overall enjoyment were evenly split (9 vs. 9).

## G. Generalizability to Open-Weight Models

To evaluate framework generalizability, we replicated our primary experiments using the open-weight qwen3.6:35b-a3bbf16 model [37], deferring comprehensive statistical details to the supplementary file in the project repository. Full

TABLE VI: Summary of Qwen-3.6 Case Study Performance. Higher values indicate better performance.
<table><tr><td>Metric</td><td>(1) Baseline</td><td>(2) WithoutMIND</td><td>(3) Full SENSE</td></tr><tr><td>Total Route Yield</td><td>37</td><td>53</td><td>111</td></tr><tr><td>Overall Quality (OV)</td><td>4.50</td><td>4.58</td><td>4.61</td></tr><tr><td>Valence AS</td><td>0.130</td><td>0.1581</td><td> $0 . 1 6 2 ^ { 1 }$ </td></tr><tr><td>Valence HSP (%)</td><td>34.2</td><td> $4 6 . 5 ^ { 1 }$ </td><td> $4 7 . 0 ^ { 1 }$ </td></tr><tr><td>VQAScore (Global)</td><td>0.778</td><td> $_ { 0 . 8 2 5 ^ { 1 , 3 } }$ </td><td> $0 . 8 0 5 ^ { 1 }$ </td></tr><tr><td>CLIP-I</td><td>0.846</td><td> $0 . 9 3 7 ^ { 1 }$ </td><td> $0 . 9 5 1 ^ { 1 , 2 }$ </td></tr><tr><td>DINO</td><td>0.438</td><td> $0 . 9 2 4 ^ { 1 }$ </td><td> $0 . 9 4 2 ^ { 1 , 2 }$ </td></tr></table>

Note: Values denote absolute means, with the best performance in each row highlighted in bold. Superscripts indicate a statistically significant improvement $( p < 0 . 0 5 )$ over the corresponding method number. For Overall Quality (OV), differences did not reach strict statistical consensus across evaluators; thus, no significance superscripts are applied. Detailed statistics are available in the supplementary file in the project repository.

SENSE generated 111 distinct narrative routes, substantially outperforming both WithoutMIND $( N \ : = \ : 5 3 )$ and Baseline $( N = 3 7 )$ , suggesting enhanced generation for models with strong instruction-following capabilities. While Full SENSE and WithoutMIND achieved higher absolute OV scores (Table VI), these improvements did not reach strict statistical consensus.

In terms of affective and visual performance, both Full SENSE configurations achieved significantly higher Valence HSP compared to Baseline $( p \textless 0 . 0 0 1 )$ . For visual evaluation, while WithoutMIND scored highest in global text-image alignment, Full SENSE achieved the highest character identity preservation (CLIP-I, DINO), aligning with our primary results and demonstrating the image generation pipeline’s resilience when utilizing sufficiently capable models.

## VI. DISCUSSION

## A. Valence Guidance and Multi-Track Emotional Support

The Top-3 LLM evaluation results (Section V-B) show that Full SENSE framework achieves the highest Empathy (EM: 4.58<sup>1,2</sup>) and Overall Quality $\mathrm { { ( O V \colon 4 . 5 2 ^ { 2 } ) } }$ among all tested methods, with key comparisons reaching statistical significance (as indicated by superscripts). Compared with Baseline and WithoutMIND, Full SENSE achieves higher emotional engagement and expressive depth according to the LLM judges. The use of a single emotional arc in WithoutMIND constrains the diversity of emotional trajectories across routes and may, to a certain extent, limit the perceived emotional richness of the generated VNs. In contrast, MIND enables multiple simultaneous emotional tracks, supporting the generation of narratives in which emotional arcs are both structurally coherent and affectively varied.

Sentence-level analysis of Valence Affective Salience (AS) and High-Salience Proportion (HSP) (Section V-C) provides complementary quantitative support. Full SENSE achieves the highest Valence AS (0.103) and HSP (23.6%), significantly exceeding both Baseline (15.3%) and WithoutMIND (21.0%). In contrast, differences in Arousal AS and HSP are minor across frameworks (Full SENSE: 0.244, 85.0%; Baseline: 0.246, 86.9%). This pattern aligns with classic computational narrative analysis [1], where “emotional arcs” are primarily quantified by shifts in narrative polarity (valence). Since LLMs internalize these structural conventions during pretraining, they inherently interpret pure “Rise” and “Fall” directives as valence shifts. As a result, while SENSE successfully modulates valence across branching narratives, overall arousal levels remain largely driven by the baseline language model’s inherent generative tendencies.

B. The Complexity Paradox: Global vs. Per-Route Narrative Complexity

Our results reveal a complexity paradox inherent to branching VNs. Full SENSE generates a large number of routes (226), which exponentially increases the challenge of maintaining structural integrity and emotional coherence per path. In contrast, Baseline framework produces significantly fewer routes (99); this allows it to attain a relatively high per-route complexity (CX), but at the severe expense of overall narrative coverage, branching richness, and emotional diversity. This trade-off demonstrates that global narrative complexity imposes strict constraints on local coherence. However, by leveraging SENSE’s MIND cognitive module, structure analyzer, and path-aware context management, our framework successfully overcomes this paradox. It effectively balances macro-level structural richness with micro-level coherence, maintaining a robust level of logical and affective integrity across a vastly expanded narrative space.

## C. User Study Findings

Although overall enjoyment was evenly split (9 vs. 9), the broader pattern of results points toward an advantage for our framework. Participants rated our framework higher in Genre Relevance, with marginal trends in Image Consistency and Narrative Coherence pointing in the same direction. Theme Relevance showed the largest mean difference among all metrics, reaching significance within the experienced subgroup $( n = 8 , p < 0 . 0 0 1 )$ though not in the overall sample, indicating that participants with greater prior VN experience tended to rate our framework as more thematically relevant than the baseline. Consistent with these results, 66.7% of participants identified our version as having a clearer emotional arc, echoing comments describing a “more emotional” story with a “better representation of the given theme.” The baseline, meanwhile, was criticized for visual anomalies described as “inconsistent AI slop,” including sprites “stacked on top of another image,” aligning with its lower ratings on Image Consistency.

This raises the question of why enjoyment itself did not follow the same pattern, a tension partly clarified by the qualitative feedback: some players found the baseline’s “chaotic” and “haphazard” transitions entertaining because of their unpredictability, whereas the visual stability of our framework was, for some, described as “tedious” and lacking variation by comparison. This suggests that enjoyment in visual novels may be shaped not only by narrative or thematic strength, but also by visual variety and unpredictability, an interaction that merits further investigation in future work. Dedicated players also noted prose and character-writing limitations shared across both versions, including “unnatural” text lacking “proper character introduction,” and characters described as feeling “empty and bland” without clear motivation or background. These observations point to text quality and character depth as an area for improvement in future iterations.

## D. Limitations and Future Work

Despite SENSE’s robust performance, several limitations remain. First, while the framework successfully manages complex branching structures across different models (yielding highly competitive results even on open-weight models like Qwen-3.6), there remains a noticeable gap in literary quality between LLM-generated text and authentic, human-authored visual novels. Current models often struggle to produce the nuanced prose and deep character motivations expected in commercial VNs, meaning the system’s ultimate narrative depth is inherently bounded by the LLM’s creative limitations. Second, although the revised pipeline successfully incorporates facial expressions while preserving character identity, modeling expressive body poses and gestures remains an orthogonal challenge, as current image models still struggle to preserve character identity consistency across diverse poses. Third, while our framework effectively modulates narrative valence, achieving fine-grained guidance over narrative arousal remains limited by the LLM’s inherent tendencies. Finally, our user study (N = 18) provides suggestive trends that warrant validation in broader contexts.

Future work will explore pose-constrained image generation and context-aware inpainting to improve cross-prompt visual consistency for dynamic body gestures. Additionally, developing more effective arousal guidance mechanisms, enhancing the literary depth of generated prose (e.g., via specialized finetuning), and conducting larger participatory user studies with multimodal physiological metrics [38] are critical next steps to better evaluate player emotional engagement.

## E. Ethical Considerations

The use of generative AI in creative industries raises ethical concerns, particularly regarding potential displacement of human artists. SENSE is designed not to replace creators, but to serve as a “Human-AI Co-creation” tool. By automating laborious branching logic and repetitive asset generation, it lowers the barrier for indie developers and small studios. The SENSE generates an initial, emotionally rich, playable VN that serves as a baseline reference for creators. This AI-constructed “affective draft” acts as a catalyst, allowing human creators to further refine the emotional nuances and expand the narrative.

Additionally, because SENSE’s emotion navigation module can generate routes involving negative emotions or extreme conflict, there is an inherent risk of producing harmful, biased, or inappropriate content. For industrial deployment, it is essential to implement stricter safety alignment and robust content filtering on top of SENSE, ensuring that automatically generated interactive narratives and multimedia assets adhere to ethical standards and age-appropriate guidelines.

## VII. CONCLUSION

We presented SENSE, a state-aware framework for generating playable, branching VNs with multi-track emotional navigation from minimal user input. By integrating the MIND cognitive module, a structure analyzer, and path-aware context management, SENSE produces narratives that are both structurally coherent and emotionally resonant. Extensive evaluations demonstrate significant improvements in emotional richness, narrative diversity, and the visual-textual alignment of generated assets, all while maintaining a robust end-to-end integration within a deterministic game environment.

Overall, SENSE demonstrates the potential of AI-assisted tools to accelerate Human-AI co-creation, allowing creators to focus on high-level narrative design, artistic refinement, and emotionally engaging storytelling in playable interactive experiences.

## REFERENCES

[1] A. J. Reagan, L. Mitchell, D. Kiley, C. M. Danforth, and P. S. Dodds, “The emotional arcs of stories are dominated by six basic shapes,” EPJ data science, vol. 5, no. 1, p. 31, 2016.

[2] Y. Wen, C. Huang, H. Zhou, Z. Zeng, C. M. L. Po, J. Togelius, T. Merino, and S. Earle, “All stories are one story: Emotional arc guided procedural game level generation,” arXiv preprint arXiv:2508.02132, 2025.

[3] Y. Zhang, Y. Wei, Z. Zhang, J. Fan, H. Zhang, and S. Yan, “From outline to detail: An hierarchical end-to-end framework for coherent and consistent visual novel generation and assembly,” in Proceedings of the 33rd ACM International Conference on Multimedia, ser. MM ’25. New York, NY, USA: Association for Computing Machinery, 2025, p. 8506–8516. [Online]. Available: https://doi.org/10.1145/3746027.3755541

[4] M. Puchalski and B. Wozna-Szcze´ sniak, “Symmetry-aware llm-driven´ generation and repair of interactive fiction graphs in twine/twee,” Symmetry, vol. 18, no. 1, 2026. [Online]. Available: https://www.mdpi. com/2073-8994/18/1/113

[5] S. P. Hernandez, V. Bulitko, and M. Spetch, “Keeping the player on an emotional trajectory in interactive storytelling,” in Proceedings of the AAAI conference on artificial intelligence and interactive digital entertainment, vol. 11, no. 1, 2015, pp. 65–71.

[6] M. Theune, S. Faas, D. K. Heylen, and A. Nijholt, “The virtual storyteller: Story creation by intelligent agents,” in 1st International Conference on Technologies for Interactive Digital Storytelling and Entertainment 2003. Fraunhofer IRB Verlag, 2003, pp. 204–215.

[7] J. R. Meehan, “Tale-spin, an interactive program that writes stories.” in Ijcai, vol. 77, 1977, pp. 91–98.

[8] L. Martin, P. Ammanabrolu, X. Wang, W. Hancock, S. Singh, B. Harrison, and M. Riedl, “Event representations for automated story generation with deep neural nets,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 32, no. 1, 2018.

[9] A. Fan, M. Lewis, and Y. Dauphin, “Hierarchical neural story generation,” in Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 2018, pp. 889–898.

[10] Q. Wang, J. Hu, Z. Li, Y. Wang, D. Li, Y. Hu, and M. Tan, “Generating long-form story using dynamic hierarchical outlining with memoryenhancement,” in Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), 2025, pp. 1352–1391.

[11] H. Rashkin, A. Celikyilmaz, Y. Choi, and J. Gao, “Plotmachines: Outline-conditioned generation with dynamic plot state tracking,” in Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), 2020, pp. 4274–4295.

[12] R. Huang, L. J. Martin, C. Callison-Burch et al., “What-if: Exploring branching narratives by meta-prompting large language models,” arXiv preprint arXiv:2412.10582, 2024.

[13] J. Leandro, S. Rao, M. Xu, W. Xu, N. Jojic, C. Brockett, and B. Dolan, “Geneva: Generating and visualizing branching narratives using llms,” in 2024 IEEE Conference on Games (CoG). IEEE, 2024, pp. 1–5.

[14] Y. Wang, G. Sun, Z. Fu, Z. Liu, K. Du, H. Gao, and R. Liang, “Taleframe: An interactive story generation system with fine-grained control and large language models,” arXiv preprint arXiv:2512.02402, 2025.

[15] R. Rameshkumar, J. Huang, Y. Sun, F. Xia, and A. Saparov, “Reasoning models reason well, until they don’t,” in Proceedings of the 14th International Joint Conference on Natural Language Processing and the 4th Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics, 2025, pp. 936–956.

[16] J. S. Park, J. C. O’Brien, C. J. Cai, M. R. Morris, P. Liang, and M. S. Bernstein, “Generative agents: Interactive simulacra of human behavior,” in Proceedings of the 36th Annual ACM Symposium on User Interface Software and Technology, 2023.

[17] Q. Wu, G. Bansal, J. Zhang, Y. Wu, B. Li, E. Zhu, L. Jiang, X. Zhang, S. Zhang, J. Liu, A. H. Awadallah, R. W. White, D. Burger, and C. Wang, “Autogen: Enabling next-gen LLM applications via multi-agent conversation,” in ICLR 2024 Workshop on Large Language Model (LLM) Agents, 2024. [Online]. Available: https://openreview.net/forum?id=uAjxFFing2

[18] G. Li, H. A. A. K. Hammoud, H. Itani, D. Khizbullin, and B. Ghanem, “Camel: Communicative agents for “mind” exploration of large language model society,” in Advances in Neural Information Processing Systems, vol. 36, 2023.

[19] S. Hong, M. Zhuge, J. Chen, X. Zheng, Y. Cheng, J. Wang, C. Zhang, Z. Wang, S. K. S. Yau, Z. Lin, L. Zhou, C. Ran, L. Xiao, C. Wu, and J. Schmidhuber, “Metagpt: Meta programming for a multi-agent collaborative framework,” in International Conference on Learning Representations, 2024.

[20] Y. Du, S. Li, A. Torralba, J. B. Tenenbaum, and I. Mordatch, “Improving factuality and reasoning in language models through multiagent debate,” in Proceedings of the 41st International Conference on Machine Learning, 2024.

[21] T. Liang, Z. He, W. Jiao, X. Wang, Y. Wang, R. Wang, Y. Yang, Z. Tu, and S. Shi, “Encouraging divergent thinking in large language models through multi-agent debate,” in Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, 2024.

[22] S. Han, L. Chen, L. Lin, Z. Xu, and K. Yu, “Ibsen: Director-actor agent collaboration for controllable and interactive drama script generation,” in Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics, 2024, pp. 1607–1619.

[23] F. Huot, R. K. Amplayo, J. Palomaki, A. G. Jakobovits, E. Clark, and M. Lapata, “Agents’ room: Narrative generation through multi-step collaboration,” in The Thirteenth International Conference on Learning Representations, 2025.

[24] E. S. d. Lima, M. M. Neggers, M. A. Casanova, B. Feijo, and A. L.´ Furtado, “A pattern-oriented ai-powered approach to story composition,” in International Conference on Entertainment Computing. Springer, 2024, pp. 135–150.

[25] P. Ekman, “An argument for basic emotions,” Cognition & emotion, vol. 6, no. 3-4, pp. 169–200, 1992.

[26] A. S. Cowen and D. Keltner, “Self-report captures 27 distinct categories of emotion bridged by continuous gradients,” Proceedings ofthe national academy of sciences, vol. 114, no. 38, pp. E7900–E7909, 2017.

[27] H. Ye, J. Zhang, S. Liu, X. Han, and W. Yang, “Ip-adapter: Text compatible image prompt adapter for text-to-image diffusion models,” arXiv preprint arxiv:2308.06721, 2023.

[28] G. Jocher, A. Chaurasia, and J. Qiu, “Ultralytics yolov8,” 2023. [Online]. Available: https://github.com/ultralytics/ultralytics

[29] M. C. Gursesli, S. Chen, M. F. Dewantoro, X. You, E. Anbar, P. Taveekitworachai, F. Abdullah, P. Tarchi, M. Duradoni, A. Lanata, A. Guazzini, and R. Thawonmas, “The role of large language model-generated stories in the narrative experience of serious visual novel games,” International Journal of Human–Computer Interaction, vol. 42, no. 4, pp. 2489–2508, 2026. [Online]. Available: https://doi.org/10.1080/10447318.2025.2528993

[30] C. Chhun, F. M. Suchanek, and C. Clavel, “Do language models enjoy their own stories? prompting large language models for automatic story evaluation,” Transactions of the Association for Computational Linguistics, vol. 12, pp. 1122–1142, 09 2024. [Online]. Available: https://doi.org/10.1162/tacl a 00689

[31] M. Zhang, Z. Ye, P. Suntichaikul, S. Goto, O. Nakamura, and R. Thawonmas, “Ap-based llm story generation: Envision technology development through science fiction,” in 2025 IEEE 14th Global Conference on Consumer Electronics (GCCE), 2025, pp. 631–634.

[32] L. Christ, S. Amiriparian, M. Milling, I. Aslan, and B. Schuller, “Modeling emotional trajectories in written stories utilizing transformers and weakly-supervised learning,” in Findings of the Association for

Computational Linguistics: ACL 2024, L.-W. Ku, A. Martins, and V. Srikumar, Eds. Bangkok, Thailand: Association for Computational Linguistics, Aug. 2024, pp. 7144–7159. [Online]. Available: https: //aclanthology.org/2024.findings-acl.426/

[33] H. Revers, J. J. Stekelenburg, J. Vroomen, K. Van Deun, and M. Bastiaansen, “Dissociating affective salience and valence in (very) longlatency erps,” Psychophysiology, vol. 62, no. 3, p. e70030, 2025.

[34] Z. Lin, D. Pathak, B. Li, J. Li, X. Xia, G. Neubig, P. Zhang, and D. Ramanan, “Evaluating text-to-visual generation with image-to-text generation,” in Computer Vision – ECCV 2024, A. Leonardis, E. Ricci, S. Roth, O. Russakovsky, T. Sattler, and G. Varol, Eds. Cham: Springer Nature Switzerland, 2025, pp. 366–384.

[35] N. Ruiz, Y. Li, V. Jampani, Y. Pritch, M. Rubinstein, and K. Aberman, “Dreambooth: Fine tuning text-to-image diffusion models for subjectdriven generation,” in 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2023, pp. 22 500–22 510.

[36] T. D. Wickens and G. Keppel, Design and analysis: A researcher’s handbook. Pearson Prentice-Hall Upper Saddle River, NJ, 2004, vol. 860.

[37] Qwen Team, “Qwen3.6-35B-A3B: Agentic coding power, now open to all,” April 2026. [Online]. Available: https://qwen.ai/blog?id=qwen3. 6-35b-a3b

[38] M. C. Gursesli, P. Tarchi, F. Cala, L. Frassineti, A. Guazzini, M. Du-´ radoni, K. Park, R. Thawonmas, X. You, and A. Lanata, “Multimodal analysis of emotions in gaming: Understanding cultural influences,” IEEE Transactions on Affective Computing, 2026.