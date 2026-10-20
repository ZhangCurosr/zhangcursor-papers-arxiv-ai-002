# Toward Structured Financial Foundation Models

Natraj Raman<sup>\*</sup>, Saumitra Mishra, Vamsi Potluru and Sumitra Ganesh

JPMorgan AI Research

## Abstract

Foundation models promise reusable capabilities and step-change performance gains across tasks. In finance, large-scale pretraining of domain-specialized foundation models can advance both goals. However, prevailing architectures and objectives do not by themselves address the combined demands of financial systems: partial observability, interacting entities, asynchronous events, changing regimes, action-dependent outcomes, and rare states with disproportionate consequences. Moreover, independently developed specialists may learn incompatible representations, while multi-domain monolithic alternatives couple data access, model updates, and governance. We therefore propose a structured-datafirst research agenda centered on modular, domain-specialized financial foundation models connected by a representation fabric for reuse and composition across domains and tasks. These models should maintain persistent financial state across aggregation levels and time horizons, capture its dynamics, and simulate action-conditioned futures, while remaining calibrated across regimes and for tail outcomes. They must also learn eficiently under data and access constraints and provide faithful, traceable explanations. We identify research challenges, directions, and capability-oriented evaluation criteria.

Keywords: structured data, domain models, representation interoperability, dynamics, simulation

## 1. Introduction

Can the paradigm of foundation models (FMs) [1] extend to the structured data that underpin financial activity—transactions, customers, accounts, firms, securities, orders, positions, and portfolios? Financial institutions repeatedly address diverse prediction and decision problems involving interconnected entities and events [2], yet their models and representations remain fragmented across datasets and tasks. The challenge is to make them reusable while substantially improving performance.

One promising approach is large-scale pretraining of modality-specific architectures—such as tabular [3], time-series [4], or graph [5] models—on financial data, followed by adaptation to downstream tasks. Another approach adapts language-model architectures to structured financial records through text serialization or tokenization [6–8]. These approaches can improve task performance and support reuse across tasks [9, 10], but do not by themselves address the wider demands of financial systems.

Financial activity unfolds across complex networks of interacting entities, under changing regimes, and with rare, consequential outcomes [11]. Computationally, this setting is asynchronous and partially observable, with outcomes shaped by agents’ actions [12]. These properties require capabilities beyond those induced by prevailing architectures, representations, and objectives. Reuse also presents a systemlevel challenge: independently developed domain models may learn incompatible representations, while multi-domain monolithic models [13] couple data access, model updates, and governance.

To address these gaps, we advance a structured-data-first agenda built around domain-specialized financial FMs and learned interoperability among them. Each specialist FM focuses on a coherent area of financial activity, such as transactions, markets, credit, or portfolios, so it can evolve independently within domain boundaries. Records, events, entities, and relationships form its modelling substrate, supplemented where useful by unstructured information. State representations describe the domain’s financial objects and processes, such as accounts and repayment behaviour in credit, or securities and order flow in markets, while a downstream task may draw on multiple specialists.

![](images/2ca5c2b97c8e324c8894dff5a75b4dd85118a7677ee9126c7e473f0fc108d68b.jpg)  
Figure 1: From isolated models to a structured financial foundation-model paradigm. Top: Diverse financial domains and modalities connect through shared abstractions and support a broad range of tasks. Left: Task-centric models learn specialized mappings for individual financial problems. Middle: Modality-specific FMs share pretrained representations across downstream tasks. Right: Specialist financial FMs produce interoperable, multi-resolution representations that a representation fabric can compose across model boundaries, enabling task heads to combine state from multiple FMs.

Our position. A structured-data-first financial FM ecosystem should build on domain-specialized pretraining through specialists that construct explicit, persistent state representations for events, entities, and relationships across varying hierarchical levels and temporal resolutions, treating these representations as the primary transferable objects. Each specialist should learn regime-conditioned and action-conditioned state transitions, select evidence under data and access constraints, represent uncertainty and tails, and ground outputs in traceable evidence and state changes. A learnedfinancial representation fabric should enable reuse across independently trained specialists through discovery, translation, routing, and compatibility assessment, rather than requiring a single monolithic FM spanning all financial domains and use cases.

Table 1: Structural and institutional mismatches and the capabilities they motivate.
<table><tr><td>Financial property</td><td>Assumption challenged</td><td>Required capability</td></tr><tr><td>Interacting entities</td><td>Independent observations</td><td>Relational state representation</td></tr><tr><td>Multi-scale event streams</td><td>Regular sequences</td><td>Event-aware, multi-scale representation</td></tr><tr><td>Regime dependence</td><td>Stable distributions</td><td>Adaptive dynamics</td></tr><tr><td>Actions affect outcomes</td><td>Passive prediction</td><td>Action-conditioned simulation</td></tr><tr><td>Consequential tails</td><td>Average-case objectives</td><td>Tail-aware learning</td></tr><tr><td>Partial observability</td><td>Complete context</td><td>Uncertainty-aware state inference</td></tr><tr><td>Vast data; sparse evidence</td><td>Abundant pooled data</td><td>Selective learning; synthetic coverage</td></tr><tr><td>High-consequence decisions</td><td>Accuracy is sufficient</td><td>Explainable, auditable outputs</td></tr></table>

The goal is not to claim a complete financial world model [14], but to frame a research agenda around testable hypotheses. We identify mismatches between financial-system structure and prevailing foundation-model assumptions (§ 2), derive required capabilities (§ 3), describe the obstacles (§ 4), outline research directions (§ 5), and propose evaluation principles (§ 6). Figure 1 summarizes the progression from isolated task models toward reusable, interoperable specialist systems.

## 2. Why Finance Is Diferent

Conventional approaches apply an existing FM paradigm to financial data in its native modality [15–17], combining generic self-supervision with downstream adaptation. Matching modalities and scaling pretraining can yield substantial task-level gains. For a broader financial FM paradigm, however, an additional challenge is structural: financial observations arise from evolving systems of persistent entities, relationships, decisions, and constraints. Table 1 summarizes eight mismatches. Several also arise in other domains, but their co-occurrence—sharpened by accounting identities, strategic and adversarial participants, mandated information barriers, and direct economic consequences— distinguishes the financial setting.

## 2.1. Financial systems are relational, event-driven, and multi-scale

Customers, firms, counterparties, accounts, securities, and portfolios persist through time. Events can alter several entities, and their meaning depends on identity and relationships. Independent rows discard this continuity, while homogeneous token streams—including text serializations of records— obscure which objects persist and which relationships change [18]. The useful modelling unit is the state of an interacting system that observations only partially reveal.

Observations also arrive asynchronously across milliseconds to years. A fixed sequence index obscures elapsed time and the distinction between transient activity and persistent state, while exhaustive attention over every historical event is neither computationally plausible [19] nor useful.

## 2.2. The data-generating process changes

Market structure, institutional constraints, and customer or participant behaviour can cause concept drift or structural breaks [20]. Models must separate persistent relationships and constraints from changing dynamics and detect when adaptation is warranted.

Credit limits, portfolio changes, prices, and controls alter subsequent outcomes, so historical data reflect both the system and its decision rules. When deployed policies induce participants to respond or

adapt, relationships learned under one policy may fail under another [21]. Predicting what followed an observed state does not establish what would follow a diferent action [22].

## 2.3. Average performance and point predictions hide consequential failure

Rare outcomes such as defaults, liquidity shocks, operational failures, and market dislocations contribute little to average likelihood yet can dominate financial value and risk [23]. Scaling may improve common-state modelling without correcting tail miscalibration. Moreover, point predictions can conceal uncertainty about inferred state and regime when evidence is incomplete or conflicting [24], materially afecting decisions about credit limits, capital allocation, or hedging.

Financial state is also partially observed: records omit intentions, of-platform activity, delayed reports, and external relationships. Models must infer latent state without becoming certain about what they cannot observe [25].

## 2.4. Data access and model use are constrained

Prevailing FM recipes rely either on large pooled corpora [26] or on synthetic distributions broad enough to transfer to real datasets [27, 28]. Neither assumption is straightforward in finance: records may be vast, but sensitivity, permissions, fragmentation, and sheer volume prevent indiscriminate pooling or use, even as evidence for task-relevant states and outcomes remains sparse. Realistic synthetic priors must also capture interacting entities, changing regimes, actions, and consequential tails.

Financial models may influence credit, fraud, trading, risk, and operational decisions subject to formal governance review. Such review requires understanding the basis and limitations of a model-assisted decision, not just its predictive accuracy [29].

Core implication. Together, these properties challenge the idea that the financial foundation-model problem is simply to apply a larger generic model to more financial data. They broaden the research question beyond how much data and capacity are required? to also ask what should be represented, how should financial state evolve, and which capabilities should pretraining support?

## 3. Desiderata for Structured Financial Foundation Models

Addressing the mismatches in Section 2 across the full paradigm requires more than scale or average downstream accuracy. We group the requirements for structured financial FMs into seven capabilities. Figure 2 illustrates how these capabilities interact within transaction, market, and credit FMs and across model boundaries.

## Desiderata at a glance

<table><tr><td>Financial state</td><td>How should entities, events, and relations be represented across resolutions?</td></tr><tr><td>Dynamics</td><td>How should state evolve and adapt across regimes?</td></tr><tr><td>Actions</td><td>How should decisions alter possible futures?</td></tr><tr><td>Uncertainty and tails</td><td>What is unknown, and how should the model behave under stress?</td></tr><tr><td>Reuse and specialization</td><td>What should transfer across modular financial FMs?</td></tr><tr><td>Data and access</td><td>How should relevant data be selected, shared, or supplemented?</td></tr><tr><td>Traceability</td><td>What observations and state changes support an output?</td></tr></table>

Represent meaningful financial state. A financial FM should represent persistent entities, their relationships, events, and partially observed latent state. It must support irregular timing, multiple temporal scales, and multiple entity and relational granularities, relating local activity to broader geographic, organizational, or portfolio-level state.

Model dynamics and adapt across regimes. State-transition modelling should distinguish stable financial structure from regime-dependent behaviour. The FM should detect changes in dynamics, localize updates to the afected behaviour, and retain knowledge that remains valid.

Represent actions and possible futures. Financial FMs should model how actions alter possible future states while preserving accounting, relational, and temporal consistency. They should distinguish conditional scenarios from causal, counterfactual, or strategic-response scenarios.

Treat uncertainty and tails as first-class objects. Uncertainty about inferred state, the active regime, and future outcomes should remain distinguishable under limited data, partial observability, conflicting signals, and distribution shift. Calibration should support reliable decisions for low-frequency, high-impact outcomes without degrading common-state performance.

## Transfer through a representation fabric and selective specialization.

Financial representation fabric. A paradigm for governed reuse of state representations across independently trained specialist FMs. Given a consumer’s request for an entity, event, or relationship— specified by as-of time, temporal granularity, hierarchical level, and intended use—the fabric returns a view from permitted producers, adapted to the consumer’s representation space and accompanied by provenance, uncertainty, and a compatibility estimate, or declines the request. It is not a single architecture, shared schema, representation store, or orchestration layer; its distinctive AI claim is that learned interoperability can generalize beyond bespoke fixed-pair integrations to newly paired producer–consumer combinations and model upgrades with limited adaptation.

State reuse through the fabric should enable downstream adaptation with limited labelled data, parameter updates, and deployment overhead while preserving specialist capabilities.

Learn eficiently from constrained data. Pretraining should prioritize observations by state change, and inference should retrieve only relevant, permitted history and relationships, so costs scale with useful information. Training may combine synthetic, privacy-transformed, and real data to ease access constraints while preserving capability. Model checkpoints should retain source provenance and permitted uses so access and release reflect training lineage.

Ground outputs in evidence and state changes. Predictions, simulations, and recommendations should be traceable to the observations, retrieved context, representations, and state transitions that materially influenced them. Explanations should distinguish evidence from inference, attribute contributions across specialists, and be faithful, robust, auditable, concise, and intelligible.

## 4. Open Challenges

The desiderata clarify the target but not how to realize it. Representation choices constrain learnable dynamics, available evidence limits claims about interventions and tails, and institutional boundaries limit reuse. The central challenges concern abstractions, evidence, and governance rather than a checklist of model components.

![](images/ef21064707569514ce7bb10026e3b139dc2fcdacff27a7e8eed455175e7ebf71.jpg)  
Figure 2: Illustrative (hypothetical) capability view for structured financial foundation models. Specialist FMs independently construct financial state representations, model their regime-conditioned dynamics, and simulate action-conditioned futures. In the cross-model example, the Credit FM augments its native borrower state with sector state from the Market FM and customer state from the Transaction FM, exchanged through the representation fabric. Task heads consume representations either directly from specialists (Fraud Detection) or after adaptation and composition through the fabric (Credit Limit Decisioning). Uncertainty is preserved and propagated throughout, while a provenance-aware trace links a decision to its evidence, state changes, and faithfulness checks.

Defining the object of representation. Tabular and sequential FMs [30] learn patterns from records or event histories but do not necessarily maintain persistent, addressable entity and relationship state that can be updated and reused across tasks. Graph models [5, 31] provide explicit entity–relation structure, while temporal graph models [32] capture evolving interactions and can maintain entity memory. However, FM-scale pretraining and reuse raise challenges in scaling relational aggregation, accommodating heterogeneous schemas and entity types, and maintaining useful state over large, continually evolving graphs.

Open question. Which identities, event semantics, relationships, continuous quantities, missingness, and delayed observations should be represented, and which should remain specialist-specific? A universal vocabulary may erase task-relevant detail, while dataset-specific encodings reproduce fragmentation; the boundary between persistent state, retrieved context, and specialist computation remains open.

<table><tr><td colspan="2">Research gaps at a glance</td></tr><tr><td>Representation</td><td>No common abstraction for heterogeneous entities, events, and state</td></tr><tr><td>Objectives</td><td>Reconstruction alone may not induce transferable financial capabilities</td></tr><tr><td>Dynamics</td><td>Persistent structure and regime-specific behaviour are entangled</td></tr><tr><td>Intervention</td><td>Observational prediction does not identify action effects</td></tr><tr><td>Uncertainty and tails</td><td>Evidence is incomplete and consequential extremes are scarce</td></tr><tr><td>Interoperability</td><td>Reuse must preserve specialization, control, and governance</td></tr><tr><td>Data and coverage</td><td>Data are vast, restricted, and sparse for plausible scenarios</td></tr><tr><td>Explanation</td><td>Outputs must remain faithful to evidence and composed state changes</td></tr></table>

Learning useful objectives beyond reconstruction. Masked reconstruction and next-token prediction provide scalable self-supervision for structured and event data [33–35]. JEPA-style objectives predict target representations rather than reconstruct observations [36]. Neither objective family necessarily yields transferable representations of persistent financial state. Direct supervision is also limited because these states are rarely observed in full.

Open question. Which pretraining signals recover persistent state, relational structure, and multi-horizon dynamics that transfer beyond the source task? Objectives must balance fidelity and reuse: retaining too much source detail inhibits transfer, while abstracting too aggressively undermines specialist performance.

Separating persistent structure from changing dynamics. Drift-detection methods [37, 38] can flag distributional changes, while continual-learning-based adaptation [39] can accommodate them. Neither establishes whether observed drift reflects noise, a temporary regime, or durable structural change [40], particularly when financial subsystems evolve at diferent rates.

Open question. What should remain invariant, what should adapt, and when does the evidence warrant an update? Regimes are rarely labelled and change points may appear only retrospectively; rapid adaptation risks forgetting stable relationships, while conservative updates preserve obsolete dynamics.

Distinguishing prediction from intervention. Causal inference and of-policy evaluation reason about interventions from observational or experimental evidence [22, 41], but depend on identification assumptions, overlap, or controlled variation. Financial histories show outcomes under actions shaped by policies and information that may be absent from training data, while participants may adapt to deployed policies or strategically alter observed signals.

Open question. When do learned dynamics support forecasting, conditional simulation, policy evaluation, strategic-response modelling, or causal intervention? Action conditioning can reproduce historical associations without identifying action efects, and fixed-response simulations can miss feedback, gaming, or equilibrium change; stronger claims require corresponding identification and validation evidence.

Separating event likelihood, severity, and uncertainty. Reweighting, robust objectives, extremevalue methods, and synthetic generation target low-frequency outcomes, tail severity, or uncertainty under shift and weak support [42–44], but these are distinct problems. Crashes, defaults, fraud, liquidity shocks, and operational failures exhibit diferent combinations of rarity, severity, and shift: pooling them obscures distinct mechanisms, while separating them leaves little data.

Open question. How can models disentangle rare-event likelihood, tail severity, and evidential support when extremes are scarce and mechanisms difer? Historical extremes cover only a narrow range of future stress, while synthetic cases can reproduce blind spots; models must communicate when predictions exceed evidential support.

Sharing knowledge without erasing boundaries. Multi-task learning, representation alignment, adapters, and modular routing share knowledge across models or domains [45, 46], but commonly assume joint training, a shared representation space, or mappings fixed to known models, tasks, or pairs. Operational interfaces can standardize access and metadata without establishing semantic compatibility between independently learned representations. New consumers or model versions may change representation semantics, and sharing can cause negative transfer.

Open question. Can learned interoperability discover and translate relevant producer state for an independently trained consumer, detect conflicts among producer views, and predict incompatibility or information loss without an engineered mapping for every pair or version? The AI challenge is to preserve specialist performance across heterogeneous, evolving representation spaces within privacy, lineage, latency, cost, and model-risk constraints.

Selecting relevant data and expanding coverage. Adaptive sampling, retrieval, and long-context methods reduce learning costs [47]. Privacy-preserving methods can enable learning from restricted data [48], while synthetic generation can reduce direct data use and expand scenario coverage [49, 50]. In finance, however, transformations may remove entity, relational, temporal, regime, or tail information needed to infer state, while synthetic-first training may encode generator assumptions that later realdata adaptation cannot repair. Synthetic data may also leak source information or violate financial structure.

Open question. How should observed, transformed, and synthetic evidence be selected and combined during pretraining and specialization without discarding material signals or replacing evidence with plausible synthesis? Provenance must capture timing, entitlements, generators, transformations, adaptation stages, and inherited restrictions.

Explaining composed financial state. Feature attribution, example-based explanation, conceptbased methods, and provenance tracing provide partial accounts of model behaviour [51, 52], but they commonly explain one model or prediction at a time. A composed financial output may instead depend on retrieved evidence, latent state updates, representation translation, routing, and several independently trained specialists.

Open question. How can an explanation identify the permitted evidence, inferred state changes, specialist contributions, and fabric operations that produced a composed output? Plausible narratives are insuficient: explanations must faithfully disclose the conditions and limitations afecting interpretation.

Core implication. Existing methods provide starting points, but the required capabilities do not follow automatically from their use. These gaps shift the research question from which methods address each requirement? to what must be learned at scale so that these capabilities transfer beyond their original settings?

![](images/4f7932dba66f0b0ee4cf061e960fbbdb4cc84791a50f8f082cf1b3f85bae75c3.jpg)  
Figure 3: Seven interconnected research bets. Each card contrasts prevailing practice with the proposed shift. Top pane traces the operational flow. Bottom pane shows capabilities that act across it.

## 5. Research Directions

The challenges above motivate seven research bets. Each states a claim, plausible mechanisms, tradeofs, and testable implications without prescribing an architecture. The section follows conceptual dependencies; Figure 3 shows the operational flow.

## 5.1. Reusable state and interoperability

## 5.1.1. Persistent financial state as the unit of transfer

Research bet. The primary transferable object should be explicit, multi-resolution entity, event, and relationship state, rather than state that remains implicit within a model’s event sequence, parameters, or context window. Explicit state is addressable outside the producing FM, is updated as new observations arrive, carries uncertainty, and can be selectively consumed by new tasks and specialists. Pretraining should recover these states and their transitions rather than optimize event reconstruction alone.

Core claim Persistent, multi-resolution state enables reuse beyond its originating task and model.   
Approaches Typed encoding; entity-centric attention; hierarchical reconciliation; selective exports.   
Central tension Abstraction can erase detail; exhaustive retention limits compression and raises cost.   
Empirical test Improved transfer to unseen entities, schemas, tasks, and consumers.

Example. After observing a transaction, a sequence model typically encodes it only to predict the next event. A state-oriented model may instead retain a representation of the transaction, update a transaction-trail summary, and update the states of the card, customer, merchant, their relationships, and relevant aggregates. These states remain available beyond the current context window and need not all be exposed to every consumer.

Candidate mechanisms. Observation encoding, state update, and selective state exposure provide an architecture-independent functional decomposition. To construct state representations, typed event encoders can preserve type-specific semantics [53], while point-process components can preserve irregular timing [54]. Memory-augmented entity-centric attention can route observations to entity and relationship states [55]. To integrate new observations with prior state, event-driven memory mechanisms from temporal graphs [32] could be adapted with type-aware learned update operators [56]. To support multiple resolutions, multi-scale encoders can connect state across short and long temporal horizons [57]. Hierarchical aggregation with cross-level reconciliation constraints [58] can relate state across entity, customer, business, and portfolio levels while enforcing relevant relational and accounting identities. Nested embeddings ofer coarse-to-fine views of the same state representation [59]. These state representations need not be wholly latent: they may combine observed attributes, derived measures, learned features, and references to supporting evidence. Objectives that predict future state representations or transitions can encourage the current state to retain information needed for future dynamics, while cross-view alignment can promote agreement across sources and resolutions [60, 61]. FMs can expose selective typed views designed to preserve transfer utility across tasks and latent spaces [62]; semantic-stability and disclosure constraints govern what those views may contain.

Key trade-of. Explicit state can improve continuity and transfer but requires choices about what to persist, update, address, and expose. Greater abstraction may suppress numerical or source-specific detail, while retaining every event reproduces fragmentation and raises cost.

Testable implications. Explicit state should improve transfer to unseen entities, schemas, tasks, and consumers over implicit sequence representations, task-specific temporal graph models, and static entity embeddings at matched resource and adaptation budgets, not merely improve reconstruction or next-event prediction. Efective transfer from representations that persist no state beyond their context would weaken the bet.

## 5.1.2. Learned representation interoperability

Research bet. Given specialist financial FMs exposing semantically meaningful, multi-resolution state, a representation fabric should enable reuse by heterogeneous, evolving consumers through learned interoperability. Queryable contracts can declare a representation’s intended meaning and constrain its use; learned components must determine which permitted state is useful, how to translate and compose it, and when to reject transfer. The bet is that these capabilities can support eficient consumer adaptation and independent model evolution without shared weights, a universal latent space, or separately engineered integration for every producer–consumer pair.

Core claim Learned interoperability composes useful state and rejects unsupported transfer.   
Approaches State discovery; meta-learned adapters; compatibility assessment; routing and fusion.   
Central tension Independent producer optimization versus reliable consumer reuse.   
Empirical test Unseen pairings and upgrades need less adaptation than bespoke links.

Example. To reassess a corporate borrower, a credit-risk FM may request permitted transaction-flow state and liquidity state for the borrower’s issued securities. The fabric could discover entity- and time-aligned views from transaction and market FMs, translate and compose them with the credit FM’s native borrower state, and reject mappings whose entity correspondence, temporal aggregation, or semantics are insuficiently supported.

Candidate mechanisms. Discovery can identify applicable state views and feasible interoperability paths using representation contracts, semantic descriptors, and prior transfer evidence; it may occur ofline when establishing reusable paths or at query time for a particular consumer and task. Directional translators can map discovered state into a consumer’s representation space, trained from shared entities, aligned periods, cross-view objectives, or downstream supervision [62]. Shared translator families or meta-learned adapters, trained across existing producer–consumer pairs, may generalize these mappings to new pairs and versions from a small entity-aligned sample rather than a dedicated translator [63, 64]. Methods for backward-compatible alignment and studies of semantic convergence provide starting points for compatibility models that estimate semantic mismatch, information loss, and out-of-support transfer [65, 66]. Adaptive routing can use these estimates to choose among state views, resolutions, versions, translators, or composition paths according to expected task utility, confidence, cost, and latency [67]; gates can accept, downweight, or reject the resulting contributions. Consumerside import adapters or fusion modules can compose accepted producer state with native state or with translated state from other specialists [45]. Depending on available supervision and adaptation budgets, consumers may use imported state through in-context learning [28], parameter-eficient adaptation [68], or task-specific post-training [69], while producer cores remain fixed.

Key trade-of. Producer autonomy fosters domain-specific optimization but shifts much of the inter operability burden to the fabric. Strong alignment may improve composition while erasing specialist information, increasing cost, or exposing restricted attributes. Translator training also needs entityaligned data, and permitted overlap may be limited. Fabric-side selection and aggregation can form task-relevant or coarser views when semantics and permissions allow, but cannot reliably reconstruct finer state from lossy summaries. Generalization across unseen models and versions may sacrifice fi delity relative to a bespoke pairwise adapter, while conservative compatibility gates may reject transfers that would have been useful.

Testable implications. Learned interoperability should enable useful transfer to previously unpaired consumers and across model upgrades with less adaptation and integration efort than bespoke pairwise integration. Compatibility estimates should identify unsupported or harmful transfer; failure on either condition would weaken the bet.

## 5.2. Dynamics over financial state

## 5.2.1. Regime-conditioned state transitions

Research bet. Separating durable state semantics from regime-dependent transition dynamics should improve prediction across recurring regimes. A stable state interface should represent the current financial condition, while an inferred regime belief determines how that state is likely to evolve. When learned dynamics recur, models should reactivate them without redefining state or retraining the full model; when dynamics are unsupported, adaptation should remain localized where possible.

<table><tr><td>Core claim Approaches</td><td>Separating state semantics from dynamics enables regime reuse and localized adaptation.</td></tr><tr><td>Central tension</td><td>Regime inference; transition experts; persistence priors; support and shift detection.</td></tr><tr><td></td><td>Learned regimes may reflect state differences rather than distinct dynamics.</td></tr><tr><td>Empirical test</td><td>Recurring dynamics are reused without retraining or redefining state.</td></tr></table>

Example. In an order book, instrument identity and trading rules provide durable descriptors, while current spread, depth, and order imbalance form part of the evolving state. Under stress, the same state may evolve diferently as cancellation intensity rises, depth replenishment slows, and price impact increases. A regime-aware FM would infer greater support for stressed transition dynamics and apply them to the same order-book state representation. It could forecast wider spreads, lower liquidity, and greater price impact, then return to ordinary dynamics as conditions normalize.

Candidate mechanisms. Regimes may take the form of discrete latent components, soft mixtures, or continuous embeddings rather than predefined labels [70]. A jointly trained regime-assignment component can estimate regime weights or embeddings from recent state transitions, prediction residuals, and change-point evidence, while transition experts, adapters, or regime embeddings parameterize the corresponding dynamics over a shared state representation. Persistence-aware priors can encourage temporally coherent assignments rather than rapid switching in response to transient noise [71]. Multi-horizon objectives and latent state-space or generative transition models can learn these representations and dynamics across timescales [72–74]. Sharing the state interface across regime-conditioned transitions and imposing cross-regime consistency can encourage state semantics to remain stable as transition behaviour changes. Support estimation and shift detection can distinguish recurring from unsupported dynamics [75]; the latter may require a localized transition component, while failures shared across regimes may instead call for revising the state representation.

Key trade-of. Too many regime components can absorb noise, while too few can conflate distinct dynamics or miss structural change. Detection that is too sensitive produces false switching, whereas conservative detection delays adaptation. The factorization is also not automatically identifiable: state may absorb regime information, regime variables may encode current condition rather than dynamics, and transition components may specialize by entity instead of behaviour. Strong invariance constraints may incorrectly treat changing relationships as durable structure.

Testable implications. When previously learned regimes recur, a regime-conditioned model should match periodically retrained regime-blind models without requiring parameter updates, while preserving the meaning and downstream usability of its state across regimes. For genuinely unsupported dynamics, updating or adding a localized transition component should be more efective and cause less forgetting than retraining the full model. Failure on either comparison, or evidence that state must be redefined across regimes, would weaken the bet.

## 5.2.2. Action-conditioned state transitions

Research bet. The credible path toward financial world models starts with controlled, actionconditioned transition models, not end-to-end simulators trained only on observational histories. Such transitions should represent both an action’s direct efects and relevant behavioural responses, since deploying a policy can change the process being modelled [21]. Claims should advance through an evidence ladder: from observational forecasting $p ( s _ { t + 1 } \mid s _ { t } )$ , to conditional simulation $p { \big ( } s _ { t + 1 } \mid s _ { t } , a _ { t } { \big ) }$ then to of-policy evaluation, and ultimately to causal intervention $p { \big ( } s _ { t + 1 } \mid s _ { t } , \mathbf { d o } ( a _ { t } ) { \big ) }$ when identification is defensible.

Core claim Action-conditioned transitions enable simulation and a graduated path to causal claims.   
Approaches Action-conditioned models; rollout objectives; of-policy evaluation; response models.   
Central tension Plausible simulations can mask wrong dynamics; conditioning alone is not causal evidence.   
Empirical test Improved prediction within support; stronger claims require defensible identification.

Example. A conventional portfolio model may forecast returns or risk for current holdings, or perturb those holdings and recompute its prediction. An action-conditioned FM would instead condition explicitly on a proposed rebalance or hedge action. It would update the portfolio state—including holdings, exposures, and concentration—as implied by that action, then simulate possible returns, execution costs, and liquidity as markets respond. These are conditional scenarios, not automatically causal estimates.

Candidate mechanisms. History-based transition models and input-perturbation analyses provide forecasting baselines but do not distinguish controllable actions from observed features. Actionconditioned transition and world models can encode a proposed action separately from current state and generate conditional trajectories [76, 77]. Training can combine one-step conditional prediction with multi-step rollout objectives, constraints on action-implied state changes and accounting consistency, and support-aware regularization against confident extrapolation. Behaviour-policy estimation and support diagnostics can identify where logged data permit stronger evaluation; within that support, of-policy evaluation methods can estimate outcomes under alternative policies [78, 79]. Randomized or quasi-experimental variation, structural assumptions, and validated simulators can support causal intervention claims. Separately, performative-prediction frameworks characterize how deploying a policy can change the data-generating process [80]; explicit participant-response models, gametheoretic or multi-agent simulators, and adversarial or best-response training can represent such adaptive behaviour. Uncertainty about support, transition dynamics, and participant response should remain explicit.

Key trade-of. More expressive simulators can generate plausible scenarios while concealing misspecified dynamics. Action conditioning alone does not identify causal efects, and modelling behavioural responses introduces assumptions about participants’ objectives and information. Stronger claims narrow the conditions for which adequate support exists.

Testable implications. Action conditioning should improve prediction for held-out action–state combinations within the available support. Models of participant response should improve on actionconditioned models that assume historical responses remain fixed when repeated deployment changes how participants react. Of-policy or causal claims require corresponding identification assumptions and evidence; plausible trajectories alone do not sufice.

## 5.3. Cross-cutting research programs

## 5.3.1. Learningfrom constrainedfinancial data

Research bet. Financial FMs should jointly optimize which evidence they consume and how evidence from diferent sources is combined under constraints on access, volume, context, and historical coverage. During pretraining and inference, selection should prioritize permitted evidence, including informative non-events, expected to alter or disambiguate financial state rather than rely only on similarity, recency, or salience. Across training stages, curricula should combine observed, privacy-transformed, distributed, and synthetic evidence to balance empirical grounding, structural coverage, and permitted use.

Core claim State-informed selection and curricula preserve capability under data constraints.   
Approaches Adaptive sampling; temporal aggregation; state-aware retrieval; synthetic–real curricula.   
Central tension Selection can miss subtle signals; synthetic data inherit generator assumptions.   
Empirical test Maintained task quality with less context; valid, privacy-preserving synthetic coverage.

Example. A lender reassessing credit risk after a repayment-policy change may have billions of repetitive account events, restricted income and transaction records, and few historical outcomes under the new policy. A data-constrained FM could prioritize permitted observations that reveal changes in repayment capacity, aggregate low-information balance records, and use synthetic histories to supplement policy conditions poorly covered by observed data. At inference, it would retrieve only the events and relationships expected to clarify the customer’s current state.

Candidate mechanisms. During pretraining, adaptive sampling, temporal aggregation, state-change detection, meta-learning, and importance weighting can prioritize informative observations and compress repetitive evidence [81–83]. During inference, hierarchical memory, graph retrieval, and temporal routing can select permitted entities, relationships, events, and periods according to their expected efect on state or uncertainty [84]. A separate design choice is the composition and ordering of training evidence. Candidate curricula [85] include synthetic structural pretraining, synthetic pretraining followed by grounding on restricted or privacy-transformed real evidence, joint synthetic–real training, learning from unpooled data, and targeted augmentation of sparsely observed transitions [48, 86]. Provenance-aware, use-scoped checkpoints can keep observed, transformed, and generated evidence distinguishable while retaining source classes, transformations, generator ancestry, approvals, and adaptation history.

Key trade-of. Aggressive selection reduces cost and noise but can discard weak or distant signals; broad context preserves optionality at potentially prohibitive cost. Synthetic supplementation trades broader coverage against dependence on generator assumptions and weaker grounding in observed evidence.

Testable implications. State-informed selection should maintain downstream quality while using less data and context than generic retrieval or fixed-window baselines, and removing selected evidence should alter relevant state or uncertainty. Synthetic supplementation should improve coverage of sparsely observed transitions without sacrificing privacy or structural validity.

## 5.3.2. Uncertainty and tail-aware state

Research bet. Financial FMs should learn state representations that retain information about infrequent states, extreme observed quantities, and unusual transitions that may matter during specialization, while explicitly representing uncertainty and evidential support. Uncertainty about inferred state, regime, translation, and future outcomes should remain distinguishable and propagate through composition.

Core claim Separating uncertainties and modelling extremes supports reliable downstream use.   
Approaches Distributional state; shift and support detection; tail objectives; uncertainty propagation.   
Central tension Tail-focused learning can distort calibration; conservative propagation can reduce utility.   
Empirical test Improved tail estimates without ordinary-state loss; calibration through composition.

Example. A conventional risk model may return one default score even for a newly observed counterparty. An uncertainty- and tail-aware FM would separately represent how weakly the current counterparty state is supported, uncertainty about the prevailing regime, translation loss in imported exposure state, the likelihood of default and the severity of resulting loss or contagion trajectories. A consumer could then widen its loss distribution, downweight imported exposure state, or decline to issue a confident estimate when evidence or translation support is inadequate.

Candidate mechanisms. To represent uncertainty where it arises, producers can maintain uncertaintyaware latent states and attach distributional representations, ensembles, or prediction sets to exported state [42, 87]. Shift and support detectors can identify states, regimes, or inputs that are weakly covered by training evidence [88]. To improve tail behaviour, distributionally robust objectives can focus learning on specified adverse regions [89]; extreme-value components can model tail severity, while quantile and expected-shortfall objectives can target the distribution of extreme losses [44, 90]. When state crosses model boundaries, translators should preserve producer uncertainty while separately estimating semantic mismatch and translation loss [91]. Composition should retain the source and type of uncertainty rather than collapse heterogeneous qualifications into a single confidence score. Consumers can marginalize over, downweight, reject, or escalate uncertain imported state and carry the resulting uncertainty into task outputs. Calibration-aware post-training can balance downstream decision quality with calibrated confidence during specialization [92].

Key trade-of. Greater tail sensitivity can improve detection of consequential extremes but increase false alarms or distort ordinary-state calibration. Conservative uncertainty propagation may make composed outputs uninformative, while aggressive aggregation may hide unsupported specialist or

translated state.

Testable implications. Uncertainty-aware state and tail-sensitive specialization should improve calibration, unsupported-state detection, rare-event detection, severity ranking, and stressed-trajectory coherence relative to average-likelihood training alone, without unacceptable degradation under ordi nary conditions. Uncertainty should remain calibrated through composition: it should increase under missing evidence, novel regimes, and lossy or conflicting transfer, but may decrease when independently informative state is successfully combined.

## 5.3.3. Evidence-grounded explanation of state composition

Research bet. Explanations grounded in interpretable state, provenance-aware evidence, and computational state traces will be more faithful and auditable than post-hoc narratives generated from an output alone. A trace should connect selected evidence, inferred state updates, translations, routing decisions, assumptions, uncertainty propagation, and specialist contributions that materially produced a composed output.

<table><tr><td>Core claim</td><td>Material state-to-output traces make explanations faithful and auditable.</td></tr><tr><td>Approaches</td><td>Concept-aligned state; provenance; intervention tests; entitlement-safe explanations.</td></tr><tr><td>Central tension Empirical test</td><td>Richer traces improve auditability but raise disclosure, compute, and cognitive costs.</td></tr></table>

Example. A post-hoc explanation may identify influential features or semantic groups behind a creditlimit recommendation without tracing the composed computation. A trace-grounded explanation would connect permitted payment and utilization events to inferred customer-state changes, translated macroeconomic state, information lost in translation, and recommendation uncertainty, while verifying materiality by showing how the proposed limit changes when late-payment updates or the imported state are removed.

Candidate mechanisms. Concept bottlenecks can expose concept-aligned components of specialist state [93]. Pairing such state with task heads whose dependencies are transparent can make parts of the state-to-output pathway interpretable by design [94, 95]. Influence attribution, component ablation, and counterfactual intervention can test which state changes and computational pathways materially afected an output [96, 97]. Provenance-aware traces [98] can connect these pathways to timestamped evidence and uncertainty, establishing lineage without by themselves demonstrating materiality. For composed outputs, such traces should additionally identify state and translator versions, translation and routing decisions, and imported specialist contributions. An explanation layer can then selectively expose an entitlement-safe subset of supported observations, inferences, contributions, and limitations, associate its claims with trace or intervention evidence, and communicate relevant uncertainty [99].

Key trade-of. Interpretable-by-design constraints may limit representation capacity. Richer traces can improve validation and contestability but may expose restricted information, increase computation, or overwhelm users with immaterial detail; selective explanations risk omitting weak but consequential contributions.

Testable implications. Trace-grounded explanations should change consistently under material interventions, remain stable under immaterial perturbations, and preserve provenance, temporal validity, and entitlements through composition. Failure to track changes in state, translation, or specialist contribution would weaken the claim of greater faithfulness.

## 5.4. Scope and alternatives

Research horizon. Near-term work should establish whether persistent state and learned interoperability deliver reusable capability under leakage-resistant evaluation, with uncertainty and traceability tested throughout. Eficient downstream specialization and continual adaptation are enabling capabilities evaluated within these bets rather than separate research directions. Regime- and actionconditioned simulation can build on this foundation; general financial world models, strategic multi agent simulation, and autonomous decision making remain longer-horizon ambitions.

Agentic orchestration. Agentic orchestration is a downstream layer rather than part of the fabric: the fabric makes heterogeneous state discoverable, compatible, and governable, whereas an orchestrator plans sequential, multi-step workflows and selects the next specialist, tool, or action based on intermediate results. This extension raises distinct questions about bounded authority, permission-aware routing, failure recovery, and traces connecting orchestration decisions to specialist outputs and evidence [98].

Feedback-driven improvement. As specialists update from new observations over time, they must retain valid prior knowledge [39] while accounting for selective and delayed feedback shaped by which outcomes their decisions make observable. More autonomous forms of self-improvement, including models that generate their own feedback [100], are a distinct, longer-horizon program beyond the scope of the representation fabric.

Inference constraints. Latency, throughput, and state freshness matter especially in real-time fraud detection and trading. Tight response budgets may favour context selection, asynchronously updated state, precomputed interoperability paths, and lightweight task heads, with expensive composition and simulation of the critical path. Serving architectures are beyond this agenda.

Design choices. Domain-specialized FMs that keep state implicit and treat actions as observed inputs ofer a simpler alternative, but do not expose reusable state for cross-model transfer or explicit transition models for intervention analysis. For interoperability and reuse across financial models, alternative designs make diferent trade-ofs. A monolithic FM avoids translation but couples data, updates, and deployment; disconnected specialists preserve independence but reuse only outputs; pairwise adapters provide tailored transfer but multiply across pairs and versions; and agentic orchestration may sufice when output-level coordination is enough. The representation fabric instead targets state-level reuse across independently evolving specialists, accepting additional interoperability complexity to discover, translate, compose, and assess heterogeneous state. Its value should be assessed along the performance–reuse–governance frontier.

## 6. How Should Progress Be Evaluated?

Evaluation should test both generalization beyond entities and periods seen during development and reuse across new tasks and model combinations with limited adaptation. Random train–test splits and average scores can obscure these capabilities by mixing future and past information, repeating entities across splits, and underweighting consequential regimes and outcomes [101, 102]. Tests should therefore mirror each claimed capability and report task quality separately from reuse value: progress may mean meeting stringent quality requirements or meeting practical requirements with substantially lower data, adaptation, integration, or governance cost. Table 2 summarizes the evaluation principles; constructing credible tasks, datasets, and metrics remains part of the agenda.

Table 2: Capability-oriented evaluation for structured financial foundation models.
<table><tr><td>Capability</td><td>Test design</td><td>Evidence of progress</td></tr><tr><td>Task quality</td><td>task-specific protocol</td><td>Compare models under the same Meets task-specific quality and risk tol- erances</td></tr><tr><td>Transfer and adaptation</td><td>Hold out entities, tasks, schemas, or regimes</td><td>Required quality with less data or re- training</td></tr><tr><td>Dynamics, actions &amp; tails</td><td>Test shifts, interventions, and stress chronologically</td><td>Retained regimes; warranted interven- tions; calibrated tails</td></tr><tr><td>Information efficiency</td><td>selection and data curricula</td><td>Limit resources; compare evidence Quality under constraints; valid, privacy-preserving synthesis</td></tr><tr><td>Interoperability</td><td>Hold out consumers and versions; inject incompatible state</td><td>Useful cross-model state use; reliable compatibility scores</td></tr><tr><td>Trustworthiness</td><td>Perturb evidence and components; verify traces and access controls</td><td>Faithful and robust explanations; pre- served validity and entitlements</td></tr></table>

Task performance and transfer must be evaluated together. Task-specific models, modality-specific FMs, and structured financial FMs should each be evaluated using fixed protocols, information, and adaptation budgets, with prespecified quality, calibration, and consequential-error criteria. Unseen entities, tasks, products, and regimes test distinct forms of transfer and should not collapse into one score. Results should identify what transfers and report a performance–reuse frontier: neither task performance without reuse nor low-cost sharing with unacceptable degradation validates the fabric.

Representation interoperability must generalize beyond fixed model pairs. Evaluation should hold out consumers, tasks, and model versions and assess learned interoperability alongside native-state consumption, pairwise adapters, universal latent spaces, score exchange, and disconnected specialists. Compatibility estimates should predict degradation under semantic mismatch, missing resolution, distribution shift, and producer replacement, enabling rejection or downweighting. Report these AI tests separately from permissions, lineage, and version enforcement.

Chronology and regime change are first-class splits. Evaluations must respect when information became available, including revisions and delayed reports. Training, adaptation, validation, and testing should be chronological, with entity overlap and relationship paths documented to expose leakage. Cross-regime tests [103, 104] should distinguish drift from structural change, report both adaptation and retention, and remain robust across defensible regime definitions. For unseen dynamics, localized expansion should be compared with full-model retraining.

Intervention claims require stronger evidence. Evaluation suites should state whether evidence supports forecasting, conditional simulation, of-policy generalization, strategic response, or causal intervention [41, 105]. Strategic-response tests should vary deployed policies and include adaptive or adversarial participant behaviour; counterfactual claims require experimental variation, validated simulators, known processes, or stated identification assumptions. Synthetic environments and simulators can provide controlled test beds with known interventions and regimes outside training, enabling tests

of generalization.

Tails require conditional and joint reporting. Report ordinary, rare, stressed, and shifted conditions jointly [106], including calibration, unsupported inputs, severity ranking, scenario coherence, and ordinary–tail trade-ofs. Tests should also verify that uncertainty responds appropriately to missing evidence, novel regimes, translation loss, and composition. Stress transformations and synthetic environments may supplement historical backtests only with explicit assumptions and preserved financial relationships and accounting identities.

Explanations must be tested for faithfulness and robustness. Perturbation, intervention, and trace-based tests should verify appropriate sensitivity and invariance as evidence, state updates, assumptions, uncertainty, and specialist contributions are varied [107, 108]. Explanations should distinguish observed data from inferred state and generated scenarios, remain coherent when specialists are composed or replaced, and preserve provenance, temporal validity, and entitlements.

Systems must be evaluated under realistic information constraints. End-to-end evaluation should constrain pretraining data, context, latency, compute, monetary cost, and access. Real-only, syntheticfirst, transformed-real, and mixed training should be compared under matched budgets and on held-out real data, with synthetic supplementation assessed for privacy and structural validity. Performance– budget curves and ablations should measure eficiency and the contribution of selected evidence. Every suite should specify the task, transfer unit, chronological boundary, reference systems, metrics, and failure criterion.

The complete paradigm must improve on alternative system designs. End-to-end evaluation should compare the proposed paradigm with monolithic and disconnected specialists, pairwise adapters, and agent-coordinated specialists under matched task, information, adaptation, integration, and governance budgets, including existing retrieval, simulation, and downstream controls. Explicit state, localized dynamics, and learned interoperability justify their added complexity only if they deliver greater reuse under practical performance and governance constraints.

## 7. Conclusion

Structured financial foundation models should not be defined by scale, a single architecture, or the breadth of pretraining. Their value lies in whether they recover reusable state from heterogeneous, asynchronous, and permissioned data, and make it useful across tasks, regimes, and independently evolving specialists. We argue for domain-specialized FMs that expose meaningful financial state, connected by a learned representation fabric that discovers, translates, and assesses state across model boundaries without bespoke integration for every pair. Operational controls govern these exchanges; generalizable representation interoperability is the central AI claim.

This proposal is a testable research agenda, not a deployment blueprint. Progress requires accepted task quality and eficient transfer under chronological, cross-regime, tail, and action-conditioned evaluation; composition that preserves semantics, uncertainty, provenance, and entitlements; and explanations faithful to the evidence and state changes behind an output. The broader structured-financial-FM paradigm is validated when reusable capability survives these constraints; scale and domain-specific pretraining provide important foundations but do not by themselves establish the full paradigm.

## Acknowledgments

This paper was prepared for informational purposes by the Artificial Intelligence Research group of JPMorgan Chase & Co. and its afiliates ("JP Morgan") and is not a product of the Research Department of JP Morgan. JP Morgan makes no representation and warranty whatsoever and disclaims all liability, for the completeness, accuracy or reliability of the information contained herein. This document is not intended as investment research or investment advice, or a recommendation, ofer or solicitation for the purchase or sale of any security, financial instrument, financial product or service, or to be used in any way for evaluating the merits of participating in any transaction, and shall not constitute a solicitation under any jurisdiction or to any person, if such solicitation under such jurisdiction or to such person would be unlawful. © 2026 JP Morgan Chase & Co. All rights reserved.

## References

[1] C. Zhou, Q. Li, C. Li, J. Yu, Y. Liu, G. Wang, K. Zhang, C. Ji, Q. Yan, L. He et al., “A comprehensive survey on pretrained foundation models: A history from BERT to ChatGPT,” International Journal of Machine Learning and Cybernetics, vol. 16, no. 12, pp. 9851–9915, 2025.

[2] S. Biswas, B. Carson, V. Chung, S. Singh, and R. Thomas, “Ai-bank of the future: Can banks meet the ai challenge,” New York: McKinsey & Company, vol. 28, 2020.

[3] J.-P. Jiang, S.-Y. Liu, H.-R. Cai, Q.-L. Zhou, and H.-J. Ye, “Representation learning for tabular data: A comprehensive survey,” IEEE Transactions on Pattern Analysis and Machine Intelligence, 2026.

[4] Y. Wang, H. Wu, J. Dong, Y. Liu, C. Wang, M. Long, and J. Wang, “Deep time series models: A comprehensive survey and benchmark,” IEEE Transactions on Pattern Analysis and Machine Intelligence, 2026.

[5] J. Liu, C. Yang, Z. Lu, J. Chen, Y. Li, M. Zhang, T. Bai, Y. Fang, L. Sun, P. S. Yu et al., “Graph foundation models: Concepts, opportunities and challenges,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 47, no. 6, pp. 5023–5044, 2025.

[6] N. Gruver, M. Finzi, S. Qiu, and A. G. Wilson, “Large language models are zero-shot time series forecasters,” Advances in neural information processing systems, vol. 36, pp. 19 622–19 635, 2023.

[7] A. Kim, M. Muhn, and V. Nikolaev, “Financial statement analysis with large language models,” arXiv preprint arXiv:2407.17866, 2024.

[8] Y. Shi, Z. Fu, S. Chen, B. Zhao, W. Xu, C. Zhang, and J. Li, “Kronos: A foundation model for the language of financial markets,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 40, no. 30, 2026, pp. 25 366–25 373.

[9] Z. Zhu, H. Chen, Q. Qu, and V. Chung, “Fincast: A foundation model for financial time-series forecasting,” in Proceedings of the 34th ACM International Conference on Information and Knowledge Management, 2025, pp. 4539–4549.

[10] J. Li, Y. Liu, W. Liu, S. Fang, L. Wang, C. Xu, and J. Bian, “Mars: a financial market simulation engine powered by generative foundation model,” in International Conference on Learning Representations, vol. 2025, 2025, pp. 39 490–39 524.

[11] G. Hayman and R. Ka-Kay Pang, “Reframing financial markets as complex systems: Tools for systemic risk analysis, portfolio management, and system-level investing,” CFA Institute Research

and Policy Center, 2025.

[12] L. Ardon, N. Vadori, T. Spooner, M. Xu, J. Vann, and S. Ganesh, “Towards a fully rl-based market simulator,” in Proceedings of the Second ACM International Conference on AI in Finance, 2021, pp. 1–9.

[13] S. Wu, O. Irsoy, S. Lu, V. Dabravolski, M. Dredze, S. Gehrmann, P. Kambadur, D. Rosenberg, and G. Mann, “BloombergGPT: A large language model for finance,” arXiv preprint arXiv:2303.17564, 2023.

[14] X. Chen, H. Guo, S. Guo, B. Jiang, C. Shen, X. Shen, T. Xue, Y. Xue, M. Yu, W. Zhang et al., “A definition and roadmap for world models,” arXiv preprint arXiv:2607.06401, 2026.

[15] L. Chen, S. Liu, J. Yan, X. Wang, H. Liu, C. Li, K. Jiao, J. Ying, Y. V. Liu, Q. Yang et al., “Advancing financial engineering with foundation models: progress, applications, and challenges,” Engineering, 2025.

[16] J. Lu, “Time-series foundation models in finance: Pretraining corpora, architectures, financial benchmarks, and risk-aware evaluation,” in Proceedings of the 2025 2nd International Conference on Digital Economy and Computer Science, 2025, pp. 144–151.

[17] X.-Y. L. Yanglet, Y. Cao, and L. Deng, “Multimodal financial foundation models (mfms): Progress, prospects, and challenges,” arXiv preprint arXiv:2506.01973, 2025.

[18] M. Bardoscia, P. Barucca, S. Battiston, F. Caccioli, G. Cimini, D. Garlaschelli, F. Saracco, T. Squartini, and G. Caldarelli, “The physics of financial networks,” Nature Reviews Physics, vol. 3, no. 7, pp. 490–507, 2021.

[19] A. Katharopoulos, A. Vyas, N. Pappas, and F. Fleuret, “Transformers are rnns: Fast autoregressive transformers with linear attention,” in International conference on machine learning. PMLR, 2020, pp. 5156–5165.

[20] J. Gama, I. Žliobait˙e, A. Bifet, M. Pechenizkiy, and A. Bouchachia, “A survey on concept drift adaptation,” ACM computing surveys (CSUR), vol. 46, no. 4, pp. 1–37, 2014.

[21] R. E. Lucas Jr, “Econometric policy evaluation: A critique,” in Carnegie-Rochester conference series on public policy, vol. 1. North-Holland, 1976, pp. 19–46.

[22] D. Thumm, A. Reuter, J. Robertson, S. B. Hoo, A. Weller, F. Hutter, Y. Chen, and B. Schölkopf, “Causal foundation models for time series based on prior-data fitted networks,” in 2nd ICML Workshop on Foundation Modelsfor Structured Data, 2026.

[23] N. N. Taleb, “Statistical consequences of fat tails: Real world preasymptotics, epistemology, and applications,” arXiv preprint arXiv:2001.10488, 2020.

[24] A. Saly-Kaufmann, K. Wood, J. Peter-Calliess, and S. Zohren, “Deep learning for financial time series: A large-scale benchmark of risk-adjusted performance,” arXiv preprint arXiv:2603.01820, 2026.

[25] J. Yang, Y. Hu, Y. Li, K. Zhang, K. Ding, and P. S. Yu, “From observations to states: Latent time series forecasting,” arXiv preprint arXiv:2602.00297, 2026.

[26] Y. Liu, G. Qin, Z. Shi, Z. Chen, C. Yang, X. Huang, J. Wang, and M. Long, “Sundial: A family of highly capable time series foundation models,” arXiv preprint arXiv:2502.00816, 2025.

[27] N. Hollmann, S. Müller, L. Purucker, A. Krishnakumar, M. Körfer, S. B. Hoo, R. T. Schirrmeister, and F. Hutter, “Accurate predictions on small data with a tabular foundation model,” Nature, vol. 637, no. 8045, pp. 319–326, 2025.

[28] J. Qu, D. Holzmüller, G. Varoquaux, and M. L. Morvan, “TabICLv2: A better, faster, scalable, and open tabular foundation model,” arXiv preprint arXiv:2602.11139, 2026.

[29] J. Černevičien˙e and A. Kabašinskas, “Explainable artificial intelligence (xai) in finance: a systematic literature review,” Artificial Intelligence Review, vol. 57, no. 8, p. 216, 2024.

[30] Q. Sun, H. Yuan, Y. Huang, Z. Zhang, X. Fu, R. Wang, H. Zhou, J. Li, J. Wu, and P. S. Yu, “A survey on foundation models for structured data: Tabular, time series, and graphs,” IEEE Transactions on Knowledge and Data Engineering, 2026.

[31] M. Fey, W. Hu, K. Huang, J. E. Lenssen, R. Ranjan, J. Robinson, R. Ying, J. You, and J. Leskovec, “Position: Relational deep learning-graph representation learning on relational databases,” in Forty-first International Conference on Machine Learning, 2024.

[32] E. Rossi, B. Chamberlain, F. Frasca, D. Eynard, F. Monti, and M. Bronstein, “Temporal graph networks for deep learning on dynamic graphs,” arXiv preprint arXiv:2006.10637, 2020.

[33] N. Raman, S. Ganesh, and M. Veloso, “Scalable representation learning for multimodal tabular transactions,” arXiv preprint arXiv:2410.07851, 2024.

[34] M. Ostroukhov, R. Mikhailov, V. Iashin, A. Sokolov, A. Akshonov, V. Protasov, A. Goncharov, D. Beloborodov, V. Mullin, R. Y. Enzmann et al., “PRAGMA: Revolut foundation model,” arXiv preprint arXiv:2604.08649, 2026.

[35] M. Kawawa-Beaudan, S. Sood, K. Papasotiriou, D. Borrajo, and M. Veloso, “TradeFM: A generative foundation model for trade-flow and market microstructure,” arXiv preprint arXiv:2602.23784, 2026.

[36] M. Assran, Q. Duval, I. Misra, P. Bojanowski, P. Vincent, M. Rabbat, Y. LeCun, and N. Ballas, “Self-supervised learning from images with a joint-embedding predictive architecture,” in 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2023, pp. 15 619–15 629.

[37] D. Lukats, O. Zielinski, A. Hahn, and F. Stahl, “A benchmark and survey of fully unsupervised concept drift detectors on real-world data streams,” International Journal of Data Science and Analytics, vol. 19, no. 1, pp. 1–31, 2025.

[38] S. I. Amoukou, T. Bewley, S. Mishra, F. Lecue, D. Magazzeni, and M. Veloso, “Sequential harmful shift detection without labels,” Advances in Neural Information Processing Systems, vol. 37, pp. 129 279–129 302, 2024.

[39] H. Guo, F. Zeng, F. Zhu, J. Wang, X. Wang, J. Zhou, H. Zhao, W. Liu, S. Ma, X.-Y. Zhang et al., “A comprehensive survey on continual learning in generative models,” arXiv e-prints, 2025.

[40] Y. Wang, X. Sun, L. Kong, W. Gao, Y. Li, Z. Fan, and Z. Zhuang, “Do time series foundation model benchmarks hide regime-dependent failures? evidence from trafic speed forecasting,” arXiv preprint arXiv:2606.18367, 2026.

[41] J. Pearl, M. Glymour, and N. P. Jewell, Causal inference in statistics: A primer. John Wiley & Sons, 2016.

[42] M.-L. Zhang and D.-B. Wang, “Uncertainty calibration in deep learning: Methods, emerging challenges, and llm frontiers,” Journal of Computer Science and Technology, pp. 1–23, 2026.

[43] J. Gu, X. Zhang, and G. Wang, “Beyond the norm: A survey of synthetic data generation for rare events,” arXiv preprint arXiv:2506.06380, 2025.

[44] Y. Zhang, K. Zhu, and Z. Zhu, “Resga: A large tail risk model for learning value-at-risk and expected shortfall,” arXiv preprint arXiv:2606.04576, 2026.

[45] J. Pfeifer, S. Ruder, I. Vulić, and E. M. Ponti, “Modular deep learning,” arXiv preprint arXiv:2302.11529, 2023.

[46] J. Yu, Y. Dai, X. Liu, J. Huang, Y. Shen, K. Zhang, R. Zhou, E. Adhikarla, W. Ye, Y. Liu et al., “Unleashing the power of multi-task learning: A comprehensive survey spanning traditional, deep, and pretrained foundation model eras,” arXiv preprint arXiv:2404.18961, 2024.

[47] C. Taguchi, S. Maekawa, and N. Bhutani, “Eficient context selection for long-context qa: No tuning, no iteration, just adaptive-k,” in Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, 2025, pp. 20 116–20 141.

[48] A. M. Elsheikh, J. Rokne, and R. Alhajj, “Privacy-preserving in foundation models: a systematic review of techniques, threats, and trade-ofs,” Artificial Intelligence Review, 2026.

[49] V. K. Potluru, D. Borrajo, A. Coletta, N. Dalmasso, Y. El-Laham, E. Fons, M. Ghassemi, S. Gopalakrishnan, V. Gosai, E. Kreačić et al., “Synthetic data applications in finance,” arXiv preprint arXiv:2401.00081, 2023.

[50] M. Bühler, L. Purucker, and F. Hutter, “Towards synthetic data for fine-tuning tabular foundation models,” in 1st ICML Workshop on Foundation Models for Structured Data, 2025.

[51] M. Biloš, J. T. Wilson, A. Schneider, and Y. Nevmyvaka, “A mechanistic study of tabular foundation models,” arXiv preprint arXiv:2605.21288, 2026.

[52] A. R. Balef, M. Koshil, and K. Eggensperger, “Is one layer enough? understanding inference dynamics in tabular foundation models,” arXiv preprint arXiv:2605.06510, 2026.

[53] N. Rusakov, V. Meshkov, K. Zorin, G. Zaripov, A. Uglov, A. Vasilev, and A. Klenitskiy, “A foundation model for multimodal event sequences in financial applications,” in Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 2, 2026, pp. 7890–7898.

[54] D. Berghaus, P. Seifner, K. Cvejoski, C. A. Ojeda Marin, and R. Sanchez, “In-context learning of temporal point processes with foundation inference models,” in International Conference on Learning Representations, vol. 2026, 2026, pp. 29 610–29 641.

[55] D. Cheng, Y. Zou, S. Xiang, and C. Jiang, “Graph neural networks for financial fraud detection: a review,” Frontiers of Computer Science, vol. 19, no. 9, p. 199609, 2025.

[56] S. Li, Y. Wu, Y. Xiao, P. Huang, P. Li, R. Liu, Y. Wen, and T. Sun, “Evolving beyond snapshots: Harmonizing structure and sequence via entity state tuning for temporal knowledge graph forecasting,” in Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 2026, pp. 38 336–38 350.

[57] Y. Zhang, L. Ma, S. Pal, Y. Zhang, and M. Coates, “Multi-resolution time-series transformer for long-term forecasting,” in International conference on artificial intelligence and statistics. PMLR,

2024, pp. 4222–4230.

[58] S. L. Wickramasuriya, “Probabilistic forecast reconciliation under the gaussian framework,” Journal of Business & Economic Statistics, vol. 42, no. 1, pp. 272–285, 2024.

[59] A. Kusupati, G. Bhatt, A. Rege, M. Wallingford, A. Sinha, V. Ramanujan, W. Howard-Snyder, K. Chen, S. Kakade, P. Jain et al., “Matryoshka representation learning,” Advances in Neural Information Processing Systems, vol. 35, pp. 30 233–30 249, 2022.

[60] X. Zou, L. Li, J. Wang, J. Yang, M. Ding, J. Wei, Z. Yang, F. Li, H. Zhang, S. Liu et al., “Interfacing foundation models’ embeddings,” Advances in Neural Information Processing Systems, vol. 37, pp. 40 164–40 183, 2024.

[61] J. Lu, H. Wang, Y. Xu, Y. Wang, K. Yang, and Y. Fu, “Representation potentials of foundation models for multimodal alignment: A survey,” in Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, 2025, pp. 16 680–16 695.

[62] L. Moschella, V. Maiorca, M. Fumero, A. Norelli, F. Locatello, and E. Rodolà, “Relative rep resentations enable zero-shot latent space communication,” arXiv preprint arXiv:2209.15430, 2022.

[63] A. Chen, J. Merullo, A. Stolfo, and E. Pavlick, “Transferring linear features across language models with model stitching,” Advances in Neural Information Processing Systems, vol. 38, pp. 48 531–48 563, 2026.

[64] R. Jha, C. Zhang, V. Shmatikov, and J. Morris, “Harnessing the universal geometry of embeddings,” Advances in Neural Information Processing Systems, vol. 38, pp. 45 963–45 987, 2026.

[65] S. Ricci, N. Biondi, F. Pernici, and A. Del Bimbo, “Backward-compatible aligned representations via an orthogonal transformation layer,” in European Conference on Computer Vision. Springer, 2024, pp. 451–464.

[66] D. Son, S. Rathore, A. Rufail, A. Simon, D. Zhang, S. Dave, C. Blondin, K. Zhu, and S. O’Brien, “Semantic convergence: Investigating shared representations across scaled llms,” arXiv preprint arXiv:2507.22918, 2025.

[67] Z. Li, Z. Li, and T. Zhou, “Routing manifold alignment improves generalization of mixtureof-experts llms,” in International Conference on Learning Representations, vol. 2026, 2026, pp. 21 576–21 595.

[68] Z. Han, C. Gao, J. Liu, J. Zhang, and S. Q. Zhang, “Parameter-eficient fine-tuning for large models: A comprehensive survey,” arXiv preprint arXiv:2403.14608, 2024.

[69] Z. Liu, B. Li, H. Huang, Y. Sun, Y. Wang, M. Wu, and Q. Ma, “From pre-training to post-training: A survey on time series foundation models,” TechRxiv, 2026.

[70] P. Becker-Ehmck, J. Peters, and P. Van Der Smagt, “Switching linear dynamics for variational bayes filtering,” in International conference on machine learning. PMLR, 2019, pp. 553–562.

[71] E. B. Fox, E. B. Sudderth, M. I. Jordan, and A. S. Willsky, “A sticky hdp-hmm with application to speaker diarization,” The Annals of Applied Statistics, pp. 1020–1056, 2011.

[72] X. Xu, H. Peng, and Y. Chen, “Deep switching state space model for nonlinear time series forecasting with regime switching,” International Journal of Forecasting, 2025.

[73] A. Gu and T. Dao, “Mamba: Linear-time sequence modeling with selective state spaces,” arXiv preprint arXiv:2312.00752, 2023.

[74] R. Dey, A. Matin, N. Orwick, Y. Zhang, S. Pallickara, and S. L. Pallickara, “When to trust, how to distill: Multi-foundation model guidance for lightweight, robust scientific time series forecasting,” in Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 2, 2026, pp. 10 807–10 818.

[75] C. Truong, L. Oudre, and N. Vayatis, “Selective review of ofline change point detection methods,” Signal processing, vol. 167, p. 107299, 2020.

[76] N. Hansen, H. Su, and X. Wang, “Td-mpc2: Scalable, robust world models for continuous control,” in International Conference on Learning Representations, vol. 2024, 2024, pp. 47 376–47 405.

[77] J. Xiang, Y. Gu, Z. Liu, Z. Feng, Q. Gao, Y. Hu, B. Huang, G. Liu, Y. Yang, K. Zhou et al., “Pan: A world model for general, interactable, and long-horizon world simulation,” arXiv preprint arXiv:2511.09057, 2025.

[78] S. Yu, S. Fang, R. Peng, Z. Qi, F. Zhou, and C. Shi, “Two-way deconfounder for of-policy evaluation in causal reinforcement learning,” Advances in Neural Information Processing Systems, vol. 37, pp. 78 169–78 200, 2024.

[79] M. G. Marmarelis, A. Hasan, K. Azizzadenesheli, R. M. Alvarez, and A. Anandkumar, “Of-policy predictive control with causal sensitivity analysis.” in UAI, 2025, pp. 2958–2972.

[80] M. Hardt and C. Mendler-Dünner, “Performative prediction: Past and future,” Statistical Science, vol. 40, no. 3, pp. 417–436, 2025.

[81] Y. Gu, L. Dong, H. Wang, Y. Hao, Q. Dong, F. Wei, and M. Huang, “Data selection via optimal control for language models,” in International Conference on Learning Representations, vol. 2025, 2025, pp. 61 932–61 959.

[82] Z. Yu, F. Peng, J. Lei, A. Overwijk, S. Yih, and C. Xiong, “Group-level data selection for eficient pretraining,” Advances in Neural Information Processing Systems, vol. 38, pp. 154 744–154 770, 2026.

[83] D. A. Calian, G. Farquhar, I. Kemaev, L. Zintgraf, M. Hessel, J. Shar, J. Oh, A. György, T. Schaul, J. Dean et al., “Datarater: Meta-learned dataset curation,” Advances in Neural Information Processing Systems, vol. 38, pp. 20 060–20 092, 2026.

[84] J. Deng, Y. Shen, Z. Pei, Y. Chen, and L. Huang, “Influence guided context selection for efective retrieval-augmented generation,” Advances in Neural Information Processing Systems, vol. 38, pp. 29 225–29 252, 2026.

[85] A. Garg, M. Ali, N. Hollmann, L. Purucker, S. Müller, and F. Hutter, “Real-tabpfn: Improving tabular foundation models via continued pre-training with real-world data,” arXiv preprint arXiv:2507.03971, 2025.

[86] V. Schlegel, A. A. Bharath, Z. Zhao, and K. Yee, “Generating synthetic data with formal privacy guarantees: State of the art and the road ahead,” arXiv preprint arXiv:2503.20846, 2025.

[87] X. Zhou, B. Chen, Y. Gui, and L. Cheng, “Conformal prediction: A data perspective,” ACM computing surveys, vol. 58, no. 2, pp. 1–37, 2025.

[88] H. Liu, L. Cao, and Y. Li, “Rag or learning? understanding the limits of llm adaptation under continuous knowledge drift in the real world,” in Findings of the Association for Computational Linguistics: ACL 2026, 2026, pp. 11 234–11 252.

[89] L. Shi and Y. Chi, “Distributionally robust model-based ofline reinforcement learning with near-optimal sample complexity,” Journal of Machine Learning Research, vol. 25, no. 200, pp. 1–91, 2024.

[90] K. S. NS, Y. Wang, M. Schram, J. Drgona, M. Halappanavar, F. Liu, and P. Li, “Extreme risk mitigation in reinforcement learning using extreme value theory,” arXiv preprint arXiv:2308.13011, 2023.

[91] Y. Ling, J. Li, L. Li, and S. Liang, “Bayesian domain adaptation with gaussian mixture domainindexing,” Advances in Neural Information Processing Systems, vol. 37, pp. 87 226–87 254, 2024.

[92] D. N. Yaldiz, E. Spiliopoulou, Z. Qi, S. Varia, S. Doss, and N. Pappas, “Balancing classification and calibration performance in decision-making llms via calibration aware reinforcement learning,” in Findings of the Associationfor Computational Linguistics: ACL 2026, 2026, pp. 12 537–12 553.

[93] P. W. Koh, T. Nguyen, Y. S. Tang, S. Mussmann, E. Pierson, B. Kim, and P. Liang, “Concept bottleneck models,” in International conference on machine learning. Pmlr, 2020, pp. 5338–5348.

[94] A. Chattopadhyay, S. Slocum, B. D. Haefele, R. Vidal, and D. Geman, “Interpretable by design: Learning predictors by composing interpretable queries,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 45, no. 6, pp. 7430–7443, 2022.

[95] Y. Yordanov, M. Forasassi, B. Menzat, R. Wang, C. Qi, M. Kaltenberger, A. M’Charrak, T. Salvatori, and T. Lukasiewicz, “Prototype transformer: Towards language model architectures interpretable by design,” arXiv preprint arXiv:2602.11852, 2026.

[96] O. Hill, M. Espinosa Zarlenga, and M. Jamnik, “Hierarchical concept-based interpretable models,” in International Conference on Learning Representations, vol. 2026, 2026, pp. 11 581–11 609.

[97] Z. Zhu, Z. Jin, J. Zhang, B. Sobieski, P. Biecek, F. Chen, and J. Zhou, “Faithfulness under the distribution: A new look at attribution evaluation,” in International Conference on Learning Representations, vol. 2026, 2026, pp. 38 289–38 312.

[98] Y. Wang, J. Zhang, Z. Wu, T. Cai, Z. Liu, Z. Sun, M. Dong, M. Zheng, Y. Duan, X. Yin et al., “From agent traces to trust: A survey of evidence tracing and execution provenance in llm agents,” arXiv preprint arXiv:2606.04990, 2026.

[99] N. Raman, D. Magazzeni, and S. Shah, “Bayesian hierarchical models for counterfactual estimation,” in International Conference on Artificial Intelligence and Statistics. PMLR, 2023, pp. 1115–1128.

[100] Y. Qu, T. Zhang, N. Garg, and A. Kumar, “Recursive introspection: Teaching language model agents how to self-improve,” Advances in Neural Information Processing Systems, vol. 37, pp. 55 249–55 285, 2024.

[101] M. L. De Prado, Advances in financial machine learning. John Wiley & Sons, 2018.

[102] J. Tan, J. Yang, S. Wu, G. Chen, and J. Zhao, “A critical look at the current train/test split in machine learning,” arXiv preprint arXiv:2106.04525, 2021.

[103] R. D. Arnott, C. R. Harvey, and H. Markowitz, “A backtesting protocol in the era of machine learning,” Available at SSRN 3275654, 2018.

[104] D. M. Cabral, A. M. Lima, G. H. Oliveira, and A. L. Oliveira, “Non-stationarity in financial time series: A taxonomy-based survey of drift detection, adaptation, and evaluation,” Neurocomputing, p. 134647, 2026.

[105] C. Gische, S. G. West, and M. C. Voelkle, “Forecasting causal efects of interventions versus predicting future outcomes,” Structural Equation Modeling: A Multidisciplinary Journal, vol. 28, no. 3, pp. 475–492, 2021.

[106] D. Aikman, R. Angotti, and K. Budnik, Stress testing with multiple scenarios: a tale on tails and reverse stress scenarios. ECB Working Paper, 2024, no. 2941.

[107] C. S. Chan, H. Kong, and L. Guanqing, “A comparative study of faithfulness metrics for model interpretability methods,” in Proceedings of the 60th annual meeting of the association for computational linguistics (volume 1: Long papers), 2022, pp. 5029–5038.

[108] C. Agarwal, N. Johnson, M. Pawelczyk, S. Krishna, E. Saxena, M. Zitnik, and H. Lakkaraju, “Rethinking stability for attribution-based explanations,” arXiv preprint arXiv:2203.06877, 2022.