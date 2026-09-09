# Vision: Data-Centric Anchoring for Robust and Interpretable Agentic AI

Arun Vignesh Malarkkan, Xinyuan Wang, and Yanjie Fu

School of Computing and Augmented Intelligence, Arizona State University, USA

{arun.malarkkan, xwang735, yanjie.fu}@asu.edu

## Abstract

Agentic AI systems built on Large Language Models (LLMs) exhibit two persistent failure modes that scaling does not resolve: 1. brittleness under distribution shift, and 2. opacity in decision-making. We argue that both failures are co-symptoms of a single structural deficiency in the data lifecycle governing how agents are trained, evaluated, and deployed. Observational interaction logs of these agentic systems often encode spurious correlations without controlled variation. As a result, the data lacks the counterfactual structure needed to distinguish causal signals from coincidental patterns or to validate explanations. No model-centric method can recover invariances that the data does not contain. This paper presents Data-Centric Anchoring, a vision in which robustness and interpretability are engineered into the data environment rather than recovered from models after training. Our primary contribution is the Data-Centric Agentic Loop, a four-stage framework: Curate, Augment, Constrain, and Attribute. The ordering of these stages is not arbitrary, but structurally motivated. Curation must precede augmentation, because generative models tend to amplify the biases present in the data they are trained on. Augmentation must precede constraint enforcement, because invariance objectives require suficient variation across environments to be meaningful. Finally, attribution closes the loop by converting observed failures into targeted data interventions for the next iteration. This iterative structure makes the framework self-correcting: each stage produces the conditions required for the next. To ground this framework, we introduce a failure-driven taxonomy that links four core failure modes in the data lifecycle: spurious feature reliance, distribution shift fragility, uncertainty miscalibration, and explanation unfaithfulness. We conclude by identifying the key limitations of this approach and the open challenges that must be resolved to make robust and interpretable agentic AI practical at scale.

## 1. Introduction

Agentic AI systems built on Large Language Models (LLMs) are increasingly deployed in multi-step, interactive settings. Yet their behavior remains unreliable in ways that are dificult to predict, diagnose, or rectify. They fail under distributional shifts, accumulate errors over long trajectories, and produce explanations that appear coherent but do not reflect the true basis of their decisions. These failures are not rare edge cases but systematic.

At a high level, these issues appear to fall into separate categories. Robustness concerns performance under distribution shift, while interpretability concerns the ability to explain model behavior. But in practice, the same systems that fail to generalize are also the ones that fail to explain themselves in a verifiable way. This suggests that brittleness and opacity are not distinct failure modes, but symptoms of a shared underlying limitation. If this limitation were primarily architectural, recent progress in model scaling and training would have mitigated it. Instead, increasingly capable models exhibit the same patterns of failure, often with greater fluency. Despite advances in scaling, alignment, and prompting (e.g., RLHF, chain-of-thought and counterfactual reasoning, and toolaugmented agents) [29, 44, 22, 47], these methods do not address the structure of the data that determines what can be learned.

Models can only learn invariances that are present in their training distribution, and they can only produce faithful explanations if the data distinguishes causal relationships from coincidental correlations. Observational interaction logs, which dominate agent training, rarely provide such structure. They encode correlations without controlled variation, lack counterfactual contrast, and entangle causal and spurious signals. No model-centric intervention can recover invariances or causal annotations that the data does not contain. This observation reframes the problem. Robustness and interpretability are not properties that can be added to a trained model; they must be enabled by the data that shapes it. In agentic systems, this data is not static. It evolves across a lifecycle that includes pretraining corpora, synthetic augmentations, interaction logs, and evaluation scenarios. Each stage introduces its own failure modes: spurious correlations during curation, insuficient coverage during augmentation, miscalibration during training, and unfaithful explanations during evaluation.

With this perspective, the reliability gap in agentic AI can be seen as a set of recurring patterns. Agents rely on features that do not generalize beyond the training distribution, fail under shifts in environment or context, express confidence that does not reflect correctness, and produce explanations that cannot be validated under perturbation. We refer to these as spurious feature reliance, distribution shift fragility, uncertainty miscalibration, and explanation unfaithfulness. In agentic settings, these manifest concretely as contextual confounding, trajectory instability, confidence misalignment, and reasoning–decision decoupling. Crucially, each of these failures can be mapped to a deficiency in the data lifecycle. If the problem is rooted in the data lifecycle, then the solution must be as well. This data-centric perspective motivates our central claim:

Robustness and interpretability in agentic AI are achievable only when the data lifecycle encodes suficient variation to identify invariant relationships and validate decisions under perturbation; we formalize this requirement through a data-centric loop that structures how such data is constructed, expanded, and evaluated.

This claim has several implications. First, robustness and interpretability must be treated as joint objectives. An agent that generalizes without explainability cannot be audited, while an agent that explains without robustness produces unreliable justifications. Second, data-centric and model-centric methods are complementary: data defines what invariances are available to learn, while models determine how those invariances are represented and exploited. Third, robustness and interpretability cannot be retrofitted after training. They must be enabled during data construction, through suficient variation to support generalization and validation under perturbation.

![](images/df9c7dba213f9b70b2d23c5a2bb341eaa3699241a811532416ec5115e91adf67.jpg)  
Figure 1: Robustness and interpretability failures in agentic AI are not isolated issues, but cosymptoms of a shared structural deficiency in the data lifecycle. Weakly structured observational data lacks the variation needed to support invariant learning and counterfactual validation, motivating a data-centric anchoring perspective.

As illustrated in Figure 1, brittleness under distribution shift and explanation unfaithfulness can be understood as two surface manifestations of the same underlying limitation in the data lifecycle.

To operationalize this perspective, we introduce the Data-Centric Agentic Loop, a four-stage framework: Curate, Augment, Constrain, and Attribute. Curation reduces spurious correlations, augmentation expands coverage across environments, constraint enforcement stabilizes behavior and calibration, and attribution ensures that explanations remain valid under perturbation.

Importantly, each stage can be evaluated through measurable criteria, providing signals for diagnosing failures and guiding subsequent interventions. The stages are not independent: each produces the conditions required for the next, forming an iterative, self-correcting system in which observed failures inform future data refinement. This perspective reframes reliability in agentic AI as a data lifecycle problem, motivating the framework envisioned in the following sections.

## 2. A Failure-Driven Taxonomy of Agentic AI

The reliability gap in agentic AI systems does not arise from a single deficiency, but from a sequence of structural failures that compound across the data lifecycle. These failures are often treated independently. However, they are tightly coupled, and addressing one exposes the next. We organize them into four primary modes: spurious feature reliance, distribution shift fragility, uncertainty miscalibration, and explanation unfaithfulness, which together characterize the limits of current systems.

Spurious feature reliance: Models trained on observational data learn correlations that are predictive within the training distribution but do not reflect invariant structure. In agentic settings, this appears as contextual confounding, where decisions depend on superficial textual cues such as formatting patterns, phrasing conventions, or latent contextual signals that fail to generalize. Formally, if $\boldsymbol { X } = ( X _ { c } , X _ { s } )$ denotes causal and spurious features, the model may learn $\textstyle P ( Y \mid X _ { s } )$ even when $X _ { s }$ has no efect under intervention. This behavior is dificult to avoid through model-centric methods: increasing capacity or refining objectives simply strengthens the model’s ability to exploit the strongest available signals in the data. It can be detected through performance variation across subpopulations, where it appears as slice leakage: strong aggregate performance masking systematic errors on specific slices. Mitigation therefore requires modifying the data, like rebalancing slices, introducing counterexamples, or identifying proxy variables, often via automated slice discovery or representation-based clustering.

Distribution shift fragility: Correcting spurious feature reliance, however, is not suficient. Even if a model focuses on more stable features, it may still fail when the input distribution shifts. In agentic systems, this problem is amplified by the sequential nature of interaction, producing cascading failures. This phenomenon, which we refer to as trajectory instability, reflects the fact that models are evaluated on distributions that difer from those seen during training. This gap can be quantified through an invariance gap, measuring performance variation across environments or perturbations. Model-centric approaches alone cannot resolve this issue, as invariance cannot be learned without exposure to diverse conditions. Addressing it requires expanding the support of the data distribution through augmentation, adversarial generation, and environment diversification, closely related to domain randomization and sim-to-real transfer.

Uncertainty miscalibration: Robust models may produce predictions with high confidence in regions where they lack suficient data support. In LLM agents, this appears as confidence misalignment: fluent outputs that do not reflect underlying uncertainty. Standard training objectives optimize likelihood, not calibration, and post-hoc methods only partially correct aggregate discrepancies. Calibration can be evaluated using metrics such as expected calibration error (ECE), but these metrics degrade under distribution shift. The root cause is again data-related: regions of low density lead to unstable confidence estimates. Improving calibration requires augmenting underrepresented regions and tracking data provenance to understand where uncertainty arises.

Explanation unfaithfulness: Finally, even a robust and calibrated system may fail to provide meaningful explanations. In agentic systems, chain-of-thought (CoT) reasoning often appears plausible, but does not correspond to the true decision mechanism, resulting in reasoning–decision decoupling. Existing interpretability methods typically explain correlations within the model rather than causal dependencies. As a result, explanations may remain unchanged even when the underlying decision would difer under intervention. This failure can be detected through counterfactual faithfulness: whether modifying explanatory features changes the output. Achieving faithful explanations requires data with suficient interventional variation to support such tests. In practice, this is approximated through counterfactual data generation and attribution methods, though these approaches remain imperfect at scale.

These failures reflect breakdowns at diferent points in the data lifecycle: curation, coverage, data density, and interventional structure. As a result, robustness and interpretability cannot be achieved through model-centric improvements alone, but require systematic intervention in how data is constructed, expanded, and evaluated. This observation motivates our vision for a data-centric design principle.

<table><tr><td>Failure Mode</td><td>Detection Signal</td><td>Data-Centric Interven- tions</td><td>Limitations</td></tr><tr><td>Spurious Feature Re- liance</td><td>Slice leakage (per- formance variation across subgroups)</td><td>Dataset rebalancing, slice discovery, counterexample generation, proxy variable identification</td><td>Requires reliable slice discovery; latent con- founders may remain unobserved</td></tr><tr><td>Distribution Shift Fragility</td><td>Invariance gap across environments or per- turbations</td><td>Synthetic augmentation, ad- versarial data generation, environment diversification, domain randomization</td><td>Synthetic data may in- troduce artifacts; cover- age of real-world varia- tion is incomplete</td></tr><tr><td>Uncertainty Miscali- bration</td><td>Expected Calibration Error (ECE), confi- dence-accuracy mis- match</td><td>Augmentation of low- density regions, uncertainty estimation, calibration methods (e.g., temperature scaling)</td><td>Post-hoc calibration does not fix epistemic uncertainty; sensitive to distribution shift</td></tr><tr><td>Explanation Unfaith- fulness</td><td>Counterfactual faith- fulness (output sensi- tivity to feature per- turbations)</td><td>Data attribution (influence functions, TracIn), prove- nance tracking, counterfac- tual testing</td><td>Attribution methods are approximate; coun- terfactuals depend on valid perturbations</td></tr></table>

Table 1: Mapping from failure modes to detection signals, data-centric interventions, and limitations.

## 3. A Data Lifecycle Perspective on Robustness and Interpretability

The failure modes identified in Section 2 share a common origin: they arise when the data available to the model does not expose the structure required for robust generalization and reliable explanation. This suggests that reliability is not solely a property of models, but of the data–model system as a whole. A standard learning formulation treats the dataset as fixed and optimizes model parameters:

$$
\theta ^ { * } = \arg \operatorname* { m i n } _ { \theta } \mathcal { L } ( f _ { \theta } , \mathcal { D } )\tag{1}
$$

This formulation assumes that D contains suficient variation to identify invariant relationships and support evaluation. In agentic settings, this assumption is systematically violated. So, we view learning as a joint optimization problem:

$$
( D ^ { * } , \theta ^ { * } ) = \arg \operatorname* { m i n } _ { { D } , \theta } \mathcal { L } ( f _ { \theta } , { D } )\tag{2}
$$

where the dataset itself is treated as a design variable. The objective is not only to fit a model, but to construct a data distribution that makes robustness and interpretability attainable.

This reframing shifts the focus from model optimization to system design. Data determines what variation is available, training objectives determine how it is used, and models determine how it is represented. Reliability therefore depends on aligning these components so that the variation exposed by the data supports the invariances the model is expected to learn. Achieving this requires coordinated data-centric and model-centric interventions.

We distinguish four types of interventions:

• Data interventions: modifying the dataset (e.g., curation, augmentation, reweighting),

• Training interventions: enforcing objectives (e.g., invariance, calibration),

• Model interventions: changing representation or capacity,

• Evaluation interventions: diagnosing failures through targeted metrics.

These interventions are interdependent: data exposes variation, training enforces consistency across it, and models represent the resulting structure. Their efectiveness depends on alignment. Without suficient variation, invariance cannot be learned; without coverage, calibration fails; and without perturbations, explanations cannot be validated. This alignment naturally leads to an iterative process, in which failures observed during evaluation guide subsequent data refinement.

We formalize this process as the Data-Centric Agentic Loop.

## 4. The Data-Centric Agentic Loop

![](images/60249666a190d4d8e0072f7ab544b58d325be1b6c4438d5ce682010457f2cb03.jpg)  
Figure 2: The Data-Centric Agentic Loop iteratively refines both the dataset and the agent through four coupled stages: Curate, Augment, Constrain, and Attribute. Each stage addresses a distinct failure signal and produces the conditions required for the next, forming a failure-driven closed loop for robust and interpretable agentic AI.

We formalize the Data-Centric Agentic Loop as an iterative process that jointly updates the data distribution and the agent in order to reduce the failure modes identified in Section 2. At iteration t, the system consists of a dataset $\mathcal { D } _ { t }$ and an agent $f _ { \theta _ { t } }$ , and evolves through updates of the form

$$
( \mathcal { D } _ { t + 1 } , \theta _ { t + 1 } ) = \mathcal { A } ( \mathcal { D } _ { t } , f _ { \theta _ { t } } ) ,\tag{3}
$$

where A decomposes into four interdependent stages: Curate, Augment, Constrain, and Attribute.

Figure 2 summarizes the proposed Data-Centric Agentic Loop, in which failure signals identified through attribution inform subsequent rounds of curation, augmentation, and constraint enforcement.

## 4.1. Curate: Exposing Reliable Signal

The loop begins with Curate, where the objective is to reduce spurious feature reliance by restructuring the data distribution. In observational datasets, predictive performance is often driven by features that correlate with outcomes but do not reflect invariant structure. This manifests empirically as variation in performance across subpopulations, which we denote as slice leakage:

$$
\Delta _ { \mathrm { s l i c e } } = \operatorname* { m a x } _ { s \in \mathcal { S } } \mathcal { L } _ { s } - \operatorname* { m i n } _ { s \in \mathcal { S } } \mathcal { L } _ { s } ,\tag{4}
$$

where $\boldsymbol { \mathcal { S } }$ denotes a set of discovered slices.

Curation seeks to reduce $\Delta _ { \mathrm { s l i c e } }$ by identifying and rebalancing these subpopulations. In practice, slices are discovered through representation-based clustering, feature probing, or heuristic search over covariates, followed by targeted data collection or reweighting. This process weakens the dominance of spurious correlations and exposes signals that are more stable across contexts.

However, curation is fundamentally limited by observability. Latent confounders that cannot be detected or approximated through proxies remain embedded in the data, and errors at this stage propagate forward. The role of curation is therefore not to eliminate bias completely, but to make the residual structure more amenable to downstream correction.

## 4.2. Augment: Expanding Support

Even after curation, the dataset remains confined to the support of the observed distribution. The Augment stage addresses this by introducing additional variation through synthetic data, adversarial perturbations, and environment diversification.

The efectiveness of augmentation can be quantified through an invariance ${ \mathit { g a p } } ,$ measuring performance variation across environments:

$$
\Delta _ { \mathrm { i n v } } = \operatorname* { m a x } _ { e \in \mathcal { E } } \mathcal { L } ( f , \mathcal { D } _ { e } ) - \operatorname* { m i n } _ { e \in \mathcal { E } } \mathcal { L } ( f , \mathcal { D } _ { e } ) .\tag{5}
$$

Reducing $\Delta _ { \mathrm { i n v } }$ requires exposing the model to suficient variation so that invariant features can be identified.

In practice, augmentation introduces new samples that simulate counterfactual or adversarial conditions. These may be generated through learned generative models, rule-based transformations, or simulated environments. The goal is not merely to increase data volume, but to expand the range of conditions under which the model is trained.

This stage, however, inherits the limitations of the previous one. Residual bias in $\mathcal { D } _ { t }$ is propagated and often amplified by generation, and synthetic environments may introduce artifacts that distort the data distribution. Augmentation therefore improves coverage, but does not guarantee correctness.

## 4.3. Constrain: Enforcing Consistency

Given suficient variation, the Constrain stage enforces consistency across the expanded data distribution. Here, the goal is to learn representations and decision rules that are stable across environments and aligned with uncertainty.

This can be formalized through a worst-case risk objective:

$$
\operatorname* { m i n } _ { \theta } \operatorname* { m a x } _ { e \in \mathcal { E } } \mathcal { L } ( f _ { \theta } , D _ { e } ) ,\tag{6}
$$

as well as calibration constraints that align predicted confidence with empirical accuracy. In practice, this is implemented through invariant training objectives, distributionally robust optimization, and calibration techniques such as temperature scaling or ensembling.

The key dependency is on variation introduced by Augment. Without suficient diversity in $\mathcal { D } _ { t }$ , invariance cannot be identified, and constraint-based methods either fail or enforce incorrect structure. Thus, constraint enforcement does not create invariance—it filters and stabilizes what the data already exposes.

## 4.4. Attribute: Validating Decisions

The final stage, Attribute, evaluates whether model decisions can be explained in terms of the data that produced them. This is where interpretability is tested, rather than assumed.

We define counterfactual faithfulness as the sensitivity of the model’s output to perturbations of explanatory variables:

$$
\Delta _ { \mathrm { c f } } = \mathbb { E } _ { x } \left[ \lVert f ( x ) - f ( x ^ { \mathrm { c f } } ) \rVert \right] ,\tag{7}
$$

where $x ^ { \mathrm { c f } }$ denotes a counterfactual modification of x along explanatory dimensions.

Attribution methods identify candidate explanatory features or training examples, while counterfactual testing verifies whether these explanations are causally relevant. When $\Delta _ { \mathrm { c f } }$ is small despite large changes in explanatory variables, the explanation is unfaithful.

In practice, attribution relies on approximate methods such as influence functions or gradient tracing, and counterfactual generation depends on valid perturbations. As a result, this stage provides noisy but highly informative signals about model behavior.

## 4.5. Iterative Refinement

The four stages form a coupled system in which each stage transforms the problem faced by the next. Curation reduces dominant spurious correlations, enabling more efective augmentation; augmentation expands variation, enabling meaningful constraint enforcement; constraints stabilize

behavior, enabling reliable attribution; and attribution identifies failure modes that guide subsequent data refinement.

This induces a feedback process:

$$
\mathrm { A T T R I B U T E }  \mathrm { C U R A T E }  \mathrm { A U G M E N T }  \mathrm { C O N S T R A I N }  \mathrm { A T T R I B U T E } .\tag{8}
$$

While global convergence is not guaranteed, each stage is designed to reduce a measurable failure signal, and their composition yields a system that improves through iteration. Rather than treating robustness and interpretability as properties imposed on a fixed model, the loop treats them as outcomes of a continuously refined data–model interaction.

## 4.6. Stopping criteria and resource-aware deployment.

At iteration k, let $m _ { k } = ( \Delta _ { \mathrm { s l i c e } } ^ { ( k ) } , \Delta _ { \mathrm { i n v } } ^ { ( k ) } , \mathrm { E C E } ^ { ( k ) } , \Delta _ { \mathrm { c f } } ^ { ( k ) } )$ denote the current diagnostic vector. We stop iterating when all target metrics fall below user-specified thresholds, or when the marginal improvement per unit cost drops below a budget parameter λ for r consecutive rounds:

$$
\frac { m _ { k - 1 } - m _ { k } } { \mathrm { c o s t } _ { k } } < \lambda \mathrm { f o r } r \mathrm { r o u n d s } .\tag{9}
$$

This makes the loop budget-aware and prevents overfitting the diagnostics themselves.

## 4.7. Prototype: Operational Instantiation for Language Agents

For language-agent deployments, the loop can be instantiated over a set of trajectories $\{ \tau _ { i } \} _ { i = 1 } ^ { n }$ and a candidate environment set $\mathcal { E } _ { t }$ discovered from those trajectories. We define environments by clustering trajectories along task identity, tool behavior, interface version, and noise profile:

$$
\mathcal { E } _ { t } = \mathrm { C l u s t e r } ( \{ \tau _ { i } \} _ { i = 1 } ^ { n } ; \phi _ { \mathrm { t a s k } } , \phi _ { \mathrm { t o o l } } , \phi _ { \mathrm { i n t e r f a c e } } , \phi _ { \mathrm { n o i s e } } ) .\tag{10}
$$

A valid counterfactual $x ^ { c f }$ is not an arbitrary edit, but a minimal perturbation that preserves task semantics while remaining execution-valid:

$$
x ^ { c f } \in \arg \operatorname* { m i n } _ { x ^ { \prime } \in \mathcal { P } ( x ) } d ( x , x ^ { \prime } ) \quad \mathrm { s . t . } \quad \mathrm { S e m } ( x ^ { \prime } ) = \mathrm { S e m } ( x ) , \ \mathrm { E x e c } ( x ^ { \prime } ) \in \mathcal { V } ,\tag{11}
$$

where ${ \mathcal { P } } ( x )$ denotes a perturbation set, Sem(·) denotes semantic equivalence, and V denotes the set of valid executions.

In practice, $\mathcal { E } _ { t }$ can be approximated using minimal interface rewrites in the spirit of PIPE [5] and controlled noise injection as in AgentNoiseBench [43]. A candidate counterfactual is retained only if it passes executor-based validation and a semantic-preservation filter.

## 5. Limitations

The Data-Centric Agentic Loop assumes that reliability can be improved through iterative refinement of the data lifecycle. Its limitations arise when this assumption breaks: when the structure required is not accessible, not preservable, or not stable.

The first breakdown is observability. The Curate stage relies on identifying spurious structure through slice discovery or proxy variables, yet in agentic settings much of the relevant variation is latent. Inputs encode implicit user intent, platform conventions, and interaction history that are only partially observable. Curation therefore operates on an incomplete view of the data-generating process: it removes detectable biases while leaving latent confounding untouched. What cannot be observed cannot be corrected, and this residual structure propagates through all subsequent stages.

This incompleteness is amplified, not corrected, by augmentation. The Augment stage expands the support of the data distribution, but does so by re-sampling from the structure exposed during curation. Any residual bias is therefore inherited and often strengthened. In long-horizon settings, this leads to a deeper failure: synthetic trajectories remain locally plausible while becoming globally inconsistent. Augmentation increases variation, but cannot guarantee that the variation is structurally valid.

Even when the data is expanded, grounding model behavior remains fragile. The Attribute stage is intended to link decisions to the data that produced them, but attribution is least reliable precisely where it matters most. In low-density regions and at large model scales, influence estimates degrade, producing explanations that appear grounded but are incorrect. In such cases, interpretability does not merely fail, but it becomes misleading, providing confidence without justification.

Causal reasoning introduces a more subtle failure. Unlike empirical methods, which degrade visibly, causal methods fail silently under misspecification. An incorrect structural assumption does not simply reduce performance; it propagates through the loop, shaping augmentation, and validating explanations against an incorrect criterion. Because these errors do not necessarily surface in standard metrics, they can remain undetected while reinforcing incorrect structure.

These limitations converge in the most fundamental constraint: the loop assumes a stable target distribution. In agentic systems, this assumption does not hold. The data distribution is shaped by the agent itself,

$$
\mathcal { D } _ { t + 1 } = \mathcal { F } ( \mathcal { D } _ { t } , f _ { \theta _ { t } } ) ,\tag{12}
$$

so that each iteration modifies the environment it is meant to model. This performative nonstationarity breaks the premise of iterative refinement. The system is not converging toward a fixed distribution; it is continually reshaping it.

Taken together, these limitations reveal a deeper constraint: the efectiveness of the loop is bounded not by any single stage, but by the extent to which the data lifecycle itself can be observed, controlled, and stabilized.

## 6. Future Directions

The limitations above define a corresponding research agenda. Each direction addresses a specific structural constraint in the framework, rather than extending it in isolation.

Addressing observability requires scalable methods for discovering and representing latent structure in high-dimensional data. This includes advances in automated slice discovery, representation probing, and methods that incorporate uncertainty over unobserved variables rather than relying on point estimates. Progress in this direction would expand the efectiveness of the Curate stage beyond observable correlations.

Improving augmentation requires moving from distributional coverage to structural validity. Future work must ensure that synthetic data preserves the causal and temporal consistency of the underlying process, particularly for long-horizon trajectories. This includes developing metrics that detect structural inconsistencies and methods that generate data under multiple plausible environments rather than a single learned distribution.

Reliable attribution requires new paradigms for grounding decisions in data. Scalable alternatives to gradient-based influence estimation, as well as methods for compositional attribution across multi-step trajectories, are necessary. Equally important is the development of evaluation protocols that detect when explanations are misleading rather than merely incomplete.

Robust integration of causal reasoning requires methods that tolerate uncertain or partially incorrect structural assumptions. This includes incorporating sensitivity analysis, uncertainty over causal graphs, and hybrid approaches that combine causal reasoning with empirical validation rather than treating causal structure as fixed.

Finally, addressing non-stationarity requires rethinking the loop as a dynamic system rather than a static optimization process. Future work must account for the fact that the agent co-produces its data distribution, requiring mechanisms that track the efects of past interventions, update structural assumptions over time, and anticipate future distribution shifts rather than reacting to them.

Together, these directions point toward a broader shift: from improving individual components of the learning pipeline to designing systems that explicitly account for the interaction between data, models, and the environments they shape.

Evaluation Protocol. A natural next step is a small-scale synthetic evaluation environment with known environment labels, controllable interface perturbations, and known counterfactual ground truth. Such a protocol would support ablations that remove Curate, Augment, Constrain, or Attribute in isolation, allowing the loop to be evaluated using $\Delta _ { \mathrm { s l i c e } } , \Delta _ { \mathrm { i n v } }$ , ECE, and $\Delta _ { \mathrm { c f } }$ without claiming a full empirical validation.

## 7. Related Work

## 7.1. Data-Centric AI and the Data Lifecycle

A central premise of this paper is that robustness and interpretability are not solely model properties, but emerge from the structure of the data lifecycle. This view builds on data-centric AI, which shifts attention from model design to training-data quality, inference-data quality, and ongoing data maintenance [51, 52, 50, 19]. More broadly, this direction is consistent with earlier work on data valuation, dataset debugging, and data attribution, which seeks to quantify the efect of individual examples or subsets on downstream model behavior [3, 8, 10, 48, 34, 6]. Recent tutorials and surveys have further clarified that data-centric methods span curation, augmentation, maintenance, and evaluation, rather than any single intervention point [51, 7]. Our framing difers from these works in that we treat the entire data lifecycle as a coupled system whose failures are organized around robustness and interpretability rather than data quality in the abstract.

## 7.2. Robustness, Domain Generalization, and Invariance

The robustness side of our framework is closely related to domain generalization, distributionally robust optimization, and invariance learning. Domain generalization surveys summarize a broad family of approaches that seek out-of-distribution generalization under domain shift, including data augmentation, meta-learning, ensemble methods, and representation learning [54, 9, 35, 53]. Within this literature, IRM formalized the goal of learning representations whose optimal classifier is stable across environments, and group DRO and related formulations optimize worst-group or worst-environment risk [1, 38, 12, 36, 24]. These ideas are also connected to sim-to-real transfer and domain randomization, where the training distribution is deliberately broadened to expose invariances that survive environmental change [39, 33]. Our perspective difers in two ways. First, we treat invariance not as a standalone objective but as one stage in a larger data-centric loop. Second, we explicitly connect invariance to interpretability, since the same data variations that expose stable structure are also needed to validate counterfactual explanations.

## 7.3. Interpretability, Data Attribution, and Faithful Explanations

Classic influence-function methods trace model predictions back to training examples, and representerpoint methods provide a scalable alternative for linking predictions to influential data points [10, 48]. Recent work on training-data influence analysis and data attribution has expanded this line of research, including scalable approximations and broader taxonomies of attribution goals [34, 6, 37]. At the explanation level, however, a major challenge is faithfulness. Chain-of-thought prompts can improve performance, but multiple studies show that reasoning traces are not necessarily faithful explanations of the underlying computation [15, 14, 32]. This motivates our emphasis on counterfactual validation rather than surface-level explanation. Our goal is not merely to produce plausible rationales, but to check whether changing the factors named in an explanation actually changes the decision. That criterion is closer to causal validity than to standard feature attribution, and it is the reason the Attribute stage in our loop is defined around perturbation and provenance.

## 7.4. Agentic Evaluation, Benchmarks, and Causal Grounding

Recent surveys show that agent evaluation has matured into a distinct research area, with benchmarks spanning planning, tool use, memory, robustness, and cost-eficiency [49]. Representative benchmarks and evaluation frameworks include AgentBoard, AgentQuest, GTA, Mobile-Bench, SmartPlay, and MixEval, each of which highlights gaps between toy task success and realistic multi-step behavior [16, 4, 41, 2, 45, 26]. These eforts motivate our claim that agentic systems should be evaluated through failure signals that are visible across stages, not only by final task success. A related but distinct line of work uses causal reasoning for discovery, counterfactual generation, and explanation. Surveys on causal inference with LLMs, causal generative modeling, and causality more broadly show growing interest in using causal language and interventional reasoning to support learning and explanation [18, 11, 13, 21, 23, 20]. We draw on this literature selectively: causality is useful when the assumptions behind interventional reasoning are approximately satisfied, but our framework does not require every stage to be causal in the formal sense. Instead, causality is one tool within a broader data-centric roadmap that also includes curation, augmentation, calibration, and attribution.

## 7.5. Agentic Robustness Benchmarks and Data Debugging

Recent benchmark work suggests that agent failures should be diagnosed rather than only scored. PIPE [5] exposes interface shortcutting by minimally rewriting environment interfaces while preserving semantics and execution behavior; AgentNoiseBench [43] injects controllable user-noise and tool-noise into existing benchmarks; and PALADIN [40] trains on recovery-annotated trajectories constructed via systematic failure injection and expert demonstrations. These works align naturally with the Augment and Attribute stages of our loop. On the data side, Snorkel [25] operationalizes weak supervision and data programming, while Cleanlab [28] implements confident learning for noisy-label detection. For scalable attribution, TRAK [31] and Daunce [30] provide practical alternatives to classical influence estimation. Existing agent evaluation suites such as AgentBoard [17], GTA [42], SmartPlay [46], and MixEval [27] already cover general multi-turn and tool-use behavior, but they do not directly expose slice leakage, invariance gaps, or counterfactua faithfulness.

Across these literatures, existing work addresses important pieces of the problem: data quality, invariance, calibration, attribution, and agent evaluation. Our contribution is to connect them through a single failure-driven loop that treats the data lifecycle as the primary object of design for robust and interpretable agentic AI.

## 8. Conclusion

Agentic AI systems fail not because models are insuficiently powerful, but because the data they are trained on does not encode the variation required for reliable generalization and verifiable explanation. Brittleness under distribution shift and opacity in reasoning are therefore not separate problems, but consequences of the same constraint: invariant structure cannot be learned or tested without appropriate variation in the data. This perspective shifts the focus of progress. Improving model architectures or training objectives cannot compensate for missing structure in the data. Reliability becomes a question of whether the data lifecycle exposes the right variation, at the right granularity, and under the right conditions. Viewed in this light, robustness and interpretability are not properties to be added to trained systems, but properties that must be enabled by design. The Data-Centric Agentic Loop provides one abstraction for this process, but its significance lies not in its specific stages, but in the shift it represents: from optimizing models over fixed data to engineering the data itself as part of the learning system. The path forward is therefore not only to build more capable models, but to construct data environments in which reliable and verifiable behavior is possible.

<table><tr><td>Loop stage</td><td>marks / tools</td><td>Relevant bench- What they diagnose</td><td>Suggested metric</td></tr><tr><td>Curate</td><td>Cleanlab, PIPE</td><td>Snorkel, Noisy labels, weak super- vision, interface short- cutting, slice leakage</td><td> $\Delta _ { \mathrm { s l i c e } }$ </td></tr><tr><td>Augment</td><td>AgentNoiseBench, counterfactual augmentation, PAL- ADIN</td><td>Robustness under user- noise, tool-noise, and failure injection</td><td> $\Delta _ { \mathrm { i n v } }$ </td></tr><tr><td>Constrain</td><td>tion methods</td><td>Group DRO, IRM, Worst-group instability, Worst-group risk, ASGDRO, calibra- invariance failure, mis- ECE calibration</td><td></td></tr><tr><td>Attribute</td><td>TRAK, Daunce, in- Training-data influence fluence functions, rep- and explanation faithful- resenter points</td><td>ness</td><td> $\Delta _ { \mathrm { c f } }$ </td></tr></table>

Table 2: A concrete mapping from loop stages to benchmarks, debugging tools, and diagnostics.

## References

[1] Martin Arjovsky, Léon Bottou, Ishaan Gulrajani, and David Lopez-Paz. Invariant risk minimization. arXiv preprint arXiv:1907.02893, 2019.

[2] Shihan Deng, Weikai Xu, Hongda Sun, Wei Liu, Tao Tan, Liujianfeng Liujianfeng, Ang Li, Jian Luan, Bin Wang, Rui Yan, et al. Mobile-bench: An evaluation benchmark for llm-based mobile agents. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 8813–8831, 2024.

[3] Amirata Ghorbani and James Zou. Data shapley: Equitable valuation of data for machine learning. In International conference on machine learning, pages 2242–2251. PMLR, 2019.

[4] Luca Gioacchini, Giuseppe Siracusano, Davide Sanvito, Kiril Gashteovski, David Friede, Roberto Bifulco, and Carolin Lawrence. Agentquest: A modular benchmark framework to measure progress and improve llm agents. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 3: System Demonstrations), pages 185–193, 2024.

[5] Weizheng Gu, Chengze Li, Zhuohao Yu, Mengyuan Sun, Zhibang Yang, Wei Wang, Hongrui Jia, Shikun Zhang, and Wei Ye. What do agents learn from trajectory-sft: Semantics or interfaces?, 2026. URL https://arxiv.org/abs/2602.01611.

[6] Zayd Hammoudeh and Daniel Lowd. Training data influence analysis and estimation: A survey. Machine Learning, 113(5):2351–2403, 2024.

[7] Yuzheng Hu, Fan Wu, Haotian Ye, David Forsyth, James Zou, Nan Jiang, Jiaqi W Ma, and Han Zhao. A snapshot of influence: A local data attribution framework for online reinforcement learning. arXiv preprint arXiv:2505.19281, 2025.

[8] Ruoxi Jia, David Dao, Boxin Wang, Frances Ann Hubis, Nick Hynes, Nezihe Merve Gürel, Bo Li, Ce Zhang, Dawn Song, and Costas J Spanos. Towards eficient data valuation based on the shapley value. In The 22nd international conference on artificial intelligence and statistics, pages 1167–1176. PMLR, 2019.

[9] Arsham Gholamzadeh Khoee, Yinan Yu, and Robert Feldt. Domain generalization through meta-learning: a survey. Artificial Intelligence Review, 57(10):285, 2024.

[10] Pang Wei Koh and Percy Liang. Understanding black-box predictions via influence functions. In International conference on machine learning, pages 1885–1894. PMLR, 2017.

[11] Aneesh Komanduri, Xintao Wu, Yongkai Wu, and Feng Chen. From identifiable causal representations to controllable counterfactual generation: A survey on causal generative modeling. arXiv preprint arXiv:2310.11011, 2023.

[12] David Krueger, Ethan Caballero, Joern-Henrik Jacobsen, Amy Zhang, Jonathan Binas, Dinghuai Zhang, Remi Le Priol, and Aaron Courville. Out-of-distribution generalization via risk extrapolation (rex). In International conference on machine learning, pages 5815–5826. PMLR, 2021.

[13] Asmae Lamsaf, Rui Carrilho, João C Neves, and Hugo Proença. Causality, machine learning, and feature selection: a survey. Sensors, 25(8):2373, 2025.

[14] Tamera Lanham, Anna Chen, Ansh Radhakrishnan, Benoit Steiner, Carson Denison, Danny Hernandez, Dustin Li, Esin Durmus, Evan Hubinger, Jackson Kernion, et al. Measuring faithfulness in chain-of-thought reasoning. arXiv preprint arXiv:2307.13702, 2023.

[15] Qing Lyu, Shreya Havaldar, Adam Stein, Li Zhang, Delip Rao, Eric Wong, Marianna Apidianaki, and Chris Callison-Burch. Faithful chain-of-thought reasoning. In Proceedings of the 13th International Joint Conference on Natural Language Processing and the 3rd Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics (Volume 1: Long Papers), pages 305–329, 2023.

[16] Chang Ma, Junlei Zhang, Zhihao Zhu, Cheng Yang, Yujiu Yang, Yaohui Jin, Zhenzhong Lan, Lingpeng Kong, and Junxian He. Agentboard: An analytical evaluation board of multi-turn llm agents. Advances in neural information processing systems, 37:74325–74362, 2024.

[17] Chang Ma, Junlei Zhang, Zhihao Zhu, Cheng Yang, Yujiu Yang, Yaohui Jin, Zhenzhong Lan, Lingpeng Kong, and Junxian He. Agentboard: An analytical evaluation board of multi-turn llm agents, 2024. URL https://arxiv.org/abs/2401.13178.

[18] Jing Ma. Causal inference with large language model: A survey. Findings of the Association for Computational Linguistics: NAACL 2025, pages 5886–5898, 2025.

[19] Arun Vignesh Malarkkan, Haoyue Bai, Anjali Kaushik, and Yanjie Fu. Delta: Variational disentangled learning for privacy-preserving data reprogramming. arXiv preprint arXiv:2509.00693, 2025.

[20] Arun Vignesh Malarkkan, Haoyue Bai, Xinyuan Wang, Anjali Kaushik, Dongjie Wang, and Yanjie Fu. Rethinking spatio-temporal anomaly detection: A vision for causality-driven cybersecurity, 2025. URL https://arxiv.org/abs/2507.08177.

[21] Arun Vignesh Malarkkan, Dongjie Wang, Haoyue Bai, and Yanjie Fu. Incremental causal graph learning for online cyberattack detection in cyber-physical infrastructures. IEEE Transactions on Big Data, pages 1–12, 2025. doi: 10.1109/TBDATA.2025.3648314.

[22] Arun Vignesh Malarkkan, Manan Roy Choudhury, Guangwei Zhang, Vivek Gupta, Qingyun Wang, Yanjie Fu, and Denghui Zhang. Finrule-bench: A benchmark for joint reasoning over financial tables and principles. arXiv preprint arXiv:2603.11339, 2026.

[23] Arun Vignesh Malarkkan, Wangyang Ying, and Yanjie Fu. Causally-guided automated feature engineering with multi-agent reinforcement learning, 2026. URL https://arxiv.org/abs/ 2602.16435.

[24] Ronak Mehta, Vincent Roulet, Krishna Pillutla, and Zaid Harchaoui. Distributionally robust optimization with bias and variance reduction. arXiv preprint arXiv:2310.13863, 2023.

[25] Mike A. Merrill, Alexander G. Shaw, Nicholas Carlini, Boxuan Li, and Harsh Raj et al. Terminal-bench: Benchmarking agents on hard, realistic tasks in command line interfaces, 2026. URL https://arxiv.org/abs/2601.11868.

[26] Jinjie Ni, Fuzhao Xue, Xiang Yue, Yuntian Deng, Mahir Shah, Kabir Jain, Graham Neubig, and Yang You. Mixeval: Deriving wisdom of the crowd from llm benchmark mixtures. Advances in Neural Information Processing Systems, 37:98180–98212, 2024.

[27] Jinjie Ni, Fuzhao Xue, Xiang Yue, Yuntian Deng, Mahir Shah, Kabir Jain, Graham Neubig, and Yang You. Mixeval: Deriving wisdom of the crowd from llm benchmark mixtures, 2024. URL https://arxiv.org/abs/2406.06565.

[28] Curtis G. Northcutt, Lu Jiang, and Isaac L. Chuang. Confident learning: Estimating uncertainty in dataset labels, 2022. URL https://arxiv.org/abs/1911.00068.

[29] Long Ouyang, Jefrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730–27744, 2022.

[30] Xingyuan Pan, Chenlu Ye, Joseph Melkonian, Jiaqi W. Ma, and Tong Zhang. Daunce: Data attribution through uncertainty estimation, 2025. URL https://arxiv.org/abs/2505.23223.

[31] Sung Min Park, Kristian Georgiev, Andrew Ilyas, Guillaume Leclerc, and Aleksander Madry. Trak: Attributing model behavior at scale, 2023. URL https://arxiv.org/abs/2303.14186.

[32] Debjit Paul, Robert West, Antoine Bosselut, and Boi Faltings. Making reasoning matter: Measuring and improving faithfulness of chain-of-thought reasoning. In Findings of the Association for Computational Linguistics: EMNLP 2024, pages 15012–15032, 2024.

[33] Xue Bin Peng, Marcin Andrychowicz, Wojciech Zaremba, and Pieter Abbeel. Sim-to-real transfer of robotic control with dynamics randomization. In 2018 IEEE international conference on robotics and automation (ICRA), pages 3803–3810. IEEE, 2018.

[34] Garima Pruthi, Frederick Liu, Satyen Kale, and Mukund Sundararajan. Estimating training data influence by tracing gradient descent. Advances in Neural Information Processing Systems, 33:19920–19930, 2020.

[35] Taki Hasan Rafi, Ratul Mahjabin, Emon Ghosh, Young-Woong Ko, and Jeong-Gun Lee. Domain generalization for semantic segmentation: a survey. Artificial Intelligence Review, 57(9):247, 2024.

[36] Hamed Rahimian and Sanjay Mehrotra. Distributionally robust optimization: A review. arXiv preprint arXiv:1908.05659, 2019.

[37] Qudrat E Alahy Ratul, Edoardo Serra, and Alfredo Cuzzocrea. Evaluating attribution methods in machine learning interpretability. In 2021 IEEE International Conference on Big Data (Big Data), pages 5239–5245. IEEE, 2021.

[38] Shiori Sagawa, Pang Wei Koh, Tatsunori B Hashimoto, and Percy Liang. Distributionally robust neural networks for group shifts: On the importance of regularization for worst-case generalization. arXiv preprint arXiv:1911.08731, 2019.

[39] Josh Tobin, Rachel Fong, Alex Ray, Jonas Schneider, Wojciech Zaremba, and Pieter Abbeel. Domain randomization for transferring deep neural networks from simulation to the real world. In 2017 IEEE/RSJ international conference on intelligent robots and systems (IROS), pages 23–30. IEEE, 2017.

[40] Sri Vatsa Vuddanti, Aarav Shah, Satwik Kumar Chittiprolu, Tony Song, Sunishchal Dev, Kevin Zhu, and Maheep Chaudhary. Paladin: Self-correcting language model agents to cure tool-failure cases, 2025. URL https://arxiv.org/abs/2509.25238.

[41] Jize Wang, Zerun Ma, Yining Li, Songyang Zhang, Cailian Chen, Kai Chen, and Xinyi Le. Gta: a benchmark for general tool agents. Advances in Neural Information Processing Systems, 37:75749–75790, 2024.

[42] Jize Wang, Zerun Ma, Yining Li, Songyang Zhang, Cailian Chen, Kai Chen, and Xinyi Le. Gta: A benchmark for general tool agents, 2024. URL https://arxiv.org/abs/2407.08713.

[43] Ruipeng Wang, Yuxin Chen, Yukai Wang, Chang Wu, Junfeng Fang, Xiaodong Cai, Qi Gu, Hui Su, An Zhang, Xiang Wang, Xunliang Cai, and Tat-Seng Chua. Agentnoisebench: Benchmarking robustness of tool-using llm agents under noisy condition, 2026. URL https://arxiv.org/ abs/2602.11348.

[44] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022.

[45] Yue Wu, Xuan Tang, Tom M Mitchell, and Yuanzhi Li. Smartplay: A benchmark for llms as intelligent agents. arXiv preprint arXiv:2310.01557, 2023.

[46] Yue Wu, Xuan Tang, Tom M. Mitchell, and Yuanzhi Li. Smartplay: A benchmark for llms as intelligent agents, 2024. URL https://arxiv.org/abs/2310.01557.

[47] Shunyu Yao, Jefrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik R Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. In The eleventh international conference on learning representations, 2022.

[48] Chih-Kuan Yeh, Joon Kim, Ian En-Hsu Yen, and Pradeep K Ravikumar. Representer point selection for explaining deep neural networks. Advances in neural information processing systems, 31, 2018.

[49] Asaf Yehudai, Lilach Eden, Alan Li, Guy Uziel, Yilun Zhao, Roy Bar-Haim, Arman Cohan, and Michal Shmueli-Scheuer. Survey on evaluation of llm-based agents. arXiv preprint arXiv:2503.16416, 2025.

[50] Wangyang Ying, Cong Wei, Nanxu Gong, Xinyuan Wang, Haoyue Bai, Arun Vignesh Malarkkan, Sixun Dong, Dongjie Wang, Denghui Zhang, and Yanjie Fu. A survey on data-centric ai: Tabular learning from reinforcement learning and generative ai perspective. arXiv preprint arXiv:2502.08828, 2025.

[51] Daochen Zha, Zaid Pervaiz Bhat, Kwei-Herng Lai, Fan Yang, and Xia Hu. Data-centric ai: Perspectives and challenges. In Proceedings of the 2023 SIAM international conference on data mining (SDM), pages 945–948. SIAM, 2023.

[52] Daochen Zha, Zaid Pervaiz Bhat, Kwei-Herng Lai, Fan Yang, Zhimeng Jiang, Shaochen Zhong, and Xia Hu. Data-centric artificial intelligence: A survey. ACM Computing Surveys, 57(5): 1–42, 2025.

[53] Yingyi Zhong, Wen’an Zhou, and Zhixian Wang. A survey of data augmentation in domain generalization. Neural Processing Letters, 57(2):34, 2025.

[54] Kaiyang Zhou, Ziwei Liu, Yu Qiao, Tao Xiang, and Chen Change Loy. Domain generalization: A survey. IEEE transactions on pattern analysis and machine intelligence, 45(4):4396–4415, 2022.