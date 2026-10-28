# Proactive Memory for Event-Driven LLM Agents

Jinyang Li<sup>1</sup> Zichuan Fu<sup>1</sup> Xian Wu<sup>2</sup>

Wenlin Zhang<sup>1</sup> Yimin Deng<sup>1</sup> Kaifeng Guo<sup>1</sup>

Guojing Li<sup>1</sup> Junjia QI<sup>1</sup> Xiangyu Zhao<sup>1</sup>\*

<sup>1</sup>City University of Hong Kong <sup>2</sup>Tencent {jinyali2-c,zc.fu}@my.cityu.edu.hk xianzhao@cityu.edu.hk

## Abstract

Proactive LLM agents anticipate service opportunities from ongoing observations and accumulated history. Existing approaches emphasize recent context or retrieval of related memories, providing limited support for validating recurring behavioral regularities and assessing their applicability to the current event. We propose Proactive Memory (ProactMem), which organizes history into recent events, temporal trajectories, and behavioral regularities. Trajectory facts are checked against source events, and regularities are assessed through historical replay before being matched to current conditions. We also construct ProactiveStream, a longitudinal benchmark covering three application scopes, where agents process isolated event streams chronologically under episode-aware opportunity labels. On ProactiveStream, ProactMem achieves an overall effectiveness (F1) of 52.6%, a 21.4% relative improvement over the strongest of five external baselines. Code is available at: https://anonymous.4open.science/r/ProactMem-5351/.

## 1 Introduction

LLM agents are extending task execution toward proactive assistance that identifies service opportunities from ongoing observations and accumulated history. In activity support, smart homes, and software collaboration, useful assistance depends on recognizing new needs, selecting relevant historical evidence, and tracking prior assistance.

Existing research addresses proactive assistance through service decisions and memory construction. ProactiveAgent learns task proposals from activity contexts (Lu et al., 2025); ContextAgent combines sensed context with historical personas (Yang et al., 2025). Mem0 consolidates and retrieves information (Chhikara et al., 2025), while A-MEM links historical notes (Xu et al., 2025). ProactiveBench and ContextAgentBench evaluate task proposals and contextual service; LoCoMo and LongMemEval assess long-term conversational memory (Maharana et al., 2024; Wu et al., 2025). Continuous assistance also requires assessing memory accumulation, current applicability, and changing opportunities within one event stream. This raises a central question: how can an agent turn accumulated history into reliable evidencefor deciding when to offer assistance?

![](images/749a9e72bf9f7b50afde89bf6a3001d4f0f62243419bf07e4d91c3d135217ddf.jpg)  
Figure 1: Illustrative Sports scenario: ProactMem uses a supported evening-walk regularity beyond recent events.

To study this question, we construct ProactiveStream, a longitudinal event-stream benchmark spanning Sports, Home, and Code. Agents build memory chronologically within isolated participant, home, and repository streams, assessing each opportunity’s first actionable window and subsequent changes. With a shared backbone and intervention control policy, existing methods show different trade-offs between opportunity recovery and false interventions (Table 2). Figure 1 links earlier evening walks to current assistance.

We propose Proactive Memory (ProactMem; Figure 2), linking recent observations, temporal activities, and recurring behavior. Event memory preserves recent evidence; temporal trajectory memory abstracts source-grounded activities; behavioral regularity memory captures supported conditional behavior.

LLMs construct cited trajectories from completed events, followed by source validation of typed facts. Historical replay assesses regularities inferred from these trajectories. At decision time, contextual assessment combines the current event, recent history, trajectories, and applicable regularities to judge service value. Intervention control filters repeated proposals; the finalized decision precedes event commitment and memory updates.

Our contributions are:

• We introduce ProactiveStream, a longitudinal benchmark for proactive assistance across Sports, Home, and Code, preserving causal history, entity isolation, and episode-aware opportunity labels.

• We propose ProactMem, a hierarchical memory framework that integrates recent events, sourcevalidated temporal trajectories, and replayassessed behavioral regularities, matching historical evidence to current conditions.

• We find that validated hierarchical memory recovers more service opportunities with fewer false interventions than recent-event memory alone across three scopes.

## 2 Problem formulation

Each participant, household, or repository u has an isolated public event stream

$$
E ^ { u } = ( e _ { 1 } ^ { u } , \ldots , e _ { T _ { u } } ^ { u } ) , \qquad e _ { t } ^ { u } = ( \tau _ { t } ^ { u } , x _ { t } ^ { u } ) ,\tag{1}
$$

where $T _ { u }$ is the stream length and $\ v x _ { t } ^ { u }$ contains public fields available at time $\tau _ { t } ^ { u }$ . Events follow chronological order with deterministic tie-breaking; we omit u when clear. State $Z _ { t } ^ { u }$ comprises prior observations and finalized outputs for the same entity, together with the resulting memory and intervention control states. The policy produces

$$
( \hat { y } _ { t } , \hat { m } _ { t } ) = \pi ( e _ { t } ^ { u } , Z _ { t } ^ { u } ) , \qquad \hat { y } _ { t } \in \{ 0 , 1 \} ,\tag{2}
$$

where $\hat { y } _ { t } = 1$ denotes intervene, $\hat { y } _ { t } = 0$ denotes abstain, and $\hat { m } _ { t }$ is an optional evidence-grounded service message. After finalization, update function U commits the event and output:

$$
Z _ { t + 1 } ^ { u } = U ( Z _ { t } ^ { u } , e _ { t } ^ { u } , \hat { y } _ { t } , \hat { m } _ { t } ) .\tag{3}
$$

The objective is to detect timely, actionable service opportunities supported by current or prior observations of the same entity. Evaluation uses fixed

reference decisions $y _ { t } \in \{ 0 , 1 \}$ under Section 4.2; labels and annotation metadata are reserved for evaluation.

## 3 Proactive Memory

ProactMem uses three complementary memory layers (Figure 2): event memory for recent public events, temporal trajectory memory for sourcechecked activity summaries, and behavioral regularity memory for replay-assessed conditional regularities. Construction checks source fidelity and historical support. The decision stage combines current observations with applicable historical evidence through contextual assessment and intervention control.

## 3.1 Memory construction and update

Recent events capture immediate changes; recurring behavior emerges across days. ProactMem organizes these observations into three memory layers.

Bounded event memory. Evidence utilization can degrade in long contexts (Liu et al., 2024), motivating a bounded event layer that preserves recent observations at their original granularity. For event $e _ { t } ,$ event memory retains prior events in their original public representation, intersecting a temporal horizon $\Delta$ with capacity $K \colon$ :

$$
S _ { t } = \mathrm { T a i l } _ { K - 1 } \left( ( e _ { i } ) _ { i < t , 0 \leq \tau _ { t } - \tau _ { i } \leq \Delta } \right) .\tag{4}
$$

Here Tail $K - 1$ selects the most recent $K - 1$ events in the filtered sequence. We use $\Delta = 7 2$ hours and $K = 1 0 0 !$ : at most 99 prior events accompany the separately supplied current event. A separate buffer retains all committed events from the open local day for trajectory construction.

Fact-validated temporal trajectory memory. At the end of each local day, an LLM summarizes that day’s events into trajectory segments with intervals, typed facts, and source citations. Let $F ( \cdot )$ denote normalized checkable facts obtained from typed fields. For segment d and its cited events $\mathcal E ( d )$ , source consistency requires

$$
\begin{array} { l } { F ( d ) \subseteq \bigcup _ { e \in \mathcal { E } ( d ) } F ( e ) , } \\ { \mathcal { E } ( d ) \subseteq E _ { \mathrm { c o m p l e t e d } } ^ { u } \mathrm { d a y } } \end{array}\tag{5}
$$

Here $E _ { \mathrm { c o m p l e t e d d a y } } ^ { u }$ contains the same entity’s events from that completed day. The validator checks entity, interval, citations, and canonicalized scalar values at compatible field paths. Acceptance requires source-supported typed facts and resolved citations. Source citations preserve the link between an accepted trajectory and the observations it summarizes. An unsupported detail could otherwise propagate into a proposed regularity as daily records are consolidated. Validation therefore constrains the factual basis on which regularities are proposed, while retaining the source links needed to inspect their support. Accepted temporal trajectories form $W _ { t } .$ , retaining historical context beyond event memory. Appendix B details field handling and buffering.

![](images/c8e20d923a371544465f7ca1561a11a16b81b46550b05a726c7124ab76cd0586.jpg)  
Figure 2: ProactMem architecture. (A) Source validation and replay build memory. (B) Retrieved memory and current conditions inform assessment and intervention control. Events are stored after the decision (dashed arrows).

## Replay-assessed behavioral regularity memory.

Historical replay tests whether a proposed behavioral regularity recurs in the recorded observations. Weekly consolidation groups seven completed daily records per entity in time order, with each record assigned to one cycle. A cycle may span more than seven calendar days. An LLM proposes a trigger $^ { g , }$ expected behavior $p ,$ time window, optional weekday restrictions, and source daily identifiers. Replay checks these conditions against validated daily facts: an observable triggered day is fulfilled when $p$ occurs at or after the trigger within the window and violated otherwise. Missing observations are censored, inapplicable days are excluded, and each date contributes at most once. Let $n _ { f }$ count fulfilled (supporting) days and $n _ { v }$ count violated days:

$$
\begin{array} { r } { \mathrm { c o n f } _ { \mathrm { r e p l a y } } = \displaystyle \frac { n _ { f } } { n _ { f } + n _ { v } } , } \\ { \displaystyle \operatorname* { l i f t } = \frac { \mathrm { c o n f } _ { \mathrm { r e p l a y } } } { \mathrm { b a s e } ( p ) } . } \end{array}\tag{6}
$$

The base rate is the frequency of p over observable days under the same time-window and weekday restrictions across both triggered and other observable days. A zero denominator gives a zero score. Confidence measures consistency across observable triggered days, whereas lift accounts for how common the expected behavior is under comparable temporal restrictions. A frequently observed activity can yield high confidence even when it also occurs without the trigger. Comparing the two frequencies makes this distinction explicit when assessing a proposed regularity. The support requirement further ensures that activation rests on repeated observations across distinct days.

A well-formed proposal remains a candidate until $n _ { f } \geq 3 ,$ confidence $\ge 0 . 5 .$ , and $\operatorname { l i f t } \geq 1 . 0 ;$ it then becomes active and available for retrieval. Malformed proposals are rejected. Consolidation resets the online trajectory view, retaining records for audit. Later observations update regularity confidence and lifecycle: repeated violations can lower retrieval priority or suspend eligibility. Appendix B.2 specifies these transitions.

## 3.2 Memory retrieval and selection

The event memory supplies the bounded sequence and $W _ { t }$ supplies all completed temporal trajectories in the current cycle. Regularity retrieval checks trigger context for the same entity, weekdays, and temporal applicability with a 30-minute lead and grace interval. Matching considers current and previously recorded causal trigger occurrences. An occurrence is fulfilled when its expected behavior has been observed; censored occurrences have insufficient observations for assessment. Selection excludes both states. For regularity store $L _ { t } ^ { u }$ , lifecycle eligibility $\mathrm { E l i g i b l e } _ { t } .$ , trigger and time matching $\mathrm { M a t c h } _ { t } .$ , and occurrence state $s _ { t }$

$$
\begin{array} { r l } & { R _ { t } = \mathrm { T o p 3 }  \ell \in L _ { t } ^ { u } :  } \\ & { \qquad \mathrm { E l i g i b l e } _ { t } ( \ell ) \wedge \mathrm { M a t c h } _ { t } ( \ell ) , } \\ & { \qquad s _ { t } ( \ell ) \not \in  \mathrm { f u l f i l l e d , c e n s o r e d }   . } \end{array}\tag{7}
$$

The top-three selection ranks eligible regularities by occurrence state, lifecycle, confidence, and temporal distance (Appendix B.2). Consider the evening-walk example in Figure 1. Repeated walks under similar conditions may support a regularity even after their source events leave recent memory. When a later event satisfies its trigger near the usual walking time, the regularity contributes evidence from those earlier days. A walk already completed during the applicable occurrence removes it from selection, as does an occurrence with insufficient observations.

## 3.3 Proactive decision

Contextual assessment and intervention control. For each event, the decision context combines the current event, recent events, temporal trajectories, selected regularities, and observable episode state:

$$
C _ { t } = ( e _ { t } , S _ { t } , W _ { t } , R _ { t } , Q _ { t } ) ,\tag{8}
$$

where $Q _ { t }$ describes the service opportunity’s observable state; the controller separately records previously allowed interventions to track service coverage. The LLM produces $( \tilde { y } _ { t } , \tilde { m } _ { t } ) = \pi _ { \theta } ( C _ { t } )$ ， proposing intervention when the current event or historical context supports concrete, timely, feasible, low-risk assistance, and abstention otherwise. For the walking example, a retrieved regularity supplies the historical reason to consider a reminder. The assessment then uses recent activity and the current situation to determine whether such assistance is timely and feasible.

Within each scope, all methods apply the same intervention controller policy to their own context and intervention history. The controller compares proposed services with previously allowed interventions and converts duplicates to abstain. Scopespecific changes in opportunity state, service identity, contextual regularities, or local day permit renewed assessment; each intervention originates from an LLM proposal. The controller finalizes the proposal as $( \hat { y } _ { t } , \hat { m } _ { t } )$ , the output defined in Section 2. Appendix B.3 specifies matching fields, phase transitions, and re-entry rules.

Inference stages $e _ { t } .$ retrieves prior memory and matches current conditions, assembles $C _ { t }$ for assessment, applies intervention control, and commits $e _ { t }$ after finalization. Each complete public event enters the event buffer and open-day buffer, advancing trajectory construction, regularity re-$\mathrm { p l a y } ,$ and lifecycle updates. Abstained events can still record completed activities, changed circumstances, or violations of an existing regularity. Storing them preserves the observations needed to revise memory as behavior changes. Maintenance incorporating $e _ { t }$ can affect only later decisions. The current event is therefore assessed against the memory available on arrival, together with its own observed content. Reference labels are reserved for offline evaluation.

## 4 ProactiveStream benchmark

ProactiveStream evaluates service-opportunity detection in longitudinal streams as agents accumulate history. Table 1 gives its composition.

## 4.1 Scopes and public events

Sports uses participant activity and availability contexts for optional walking and everyday physical-activity assistance.

Home uses time-bounded home-sensor observations for gentle, low-risk check-ins. Each sensor segment becomes available at interval closure.

Code uses repository activity at GitHub’s native event granularity for low-risk triage, review, and maintenance, with one decision per event.

Each participant, home, or repository defines an isolated chronological stream, allowing earlier observations to inform later service opportunities. Public inputs contain contemporaneously available observations, using the same scope-specific schema for both label classes. Agents construct their own memory from these events; annotation and scoring metadata are reserved for evaluation. Appendix A details source datasets, fields, and preprocessing.

Table 1: Composition and provenance of ProactiveStream. Reference interventions mark service opportunities.
<table><tr><td>Scope</td><td>Primary Source</td><td>Entity</td><td>Streams</td><td>Input Events</td><td>Temporal Coverage</td><td>Reference Interventions</td></tr><tr><td>Sports</td><td>HeartSteps</td><td>participant</td><td>9</td><td>1,456</td><td>367 observed participant-days</td><td>365</td></tr><tr><td>Home</td><td>CASAS</td><td>home</td><td>4</td><td>1,624</td><td>14 consecutive days/home</td><td>190</td></tr><tr><td>Code</td><td>GH Archive</td><td>repository</td><td>4</td><td>1,459</td><td>longitudinal repository histories</td><td>174</td></tr><tr><td>Total</td><td>three scopes</td><td>isolated stream</td><td>17</td><td>4,539</td><td>three longitudinal settings</td><td>729</td></tr></table>

## 4.2 Reference labels and causal replay

Annotators assess $e _ { t }$ and its complete prior history $\mathcal { H } _ { < t }$ within the same stream. Let A indicate a concrete, timely, feasible, low-risk opportunity supported by $e _ { t }$ , B an opportunity at $e _ { t }$ supported by historical observations, and $N _ { t } ^ { \mathrm { r e f } }$ the novelty of the reference episode, shared across methods:

$$
\begin{array} { r } { y _ { t } = 1 \iff \left( A ( e _ { t } ) = 1 \right. } \\ { \left. \vee B ( e _ { t } , \mathcal { H } _ { < t } ) = 1 \right) } \\ { \wedge N _ { t } ^ { \mathrm { r e f } } = 1 . } \end{array}\tag{9}
$$

An event can provide enough evidence for immediate assistance, as represented by A. For B, its significance depends on earlier observations: a recurring activity or an earlier unresolved need can make a seemingly routine event actionable. The complete prior history available to annotators allows such opportunities to be recognized even when the supporting events fall outside a method’s bounded context. A and B may both hold for the same event; each requires a concrete service opportunity at the current time.

The reference protocol labels the first reliable actionable window intervene and covered windows abstain. This assigns a detection point to an opportunity that may span several events. Later events can reopen assessment when support strengthens, the service changes, or an episode reopens. Eventlevel labels therefore capture when useful assistance becomes warranted, together with the persistence and renewal of the underlying opportunity.

LLM-assisted triage prioritizes candidate opportunities; three team members then independently label every event, followed by cross-review and adjudication. Appendix D details a separate-team diagnostic of reference sensitivity and episode behavior. Main comparisons fix labels, prompts, memory bounds, and validation settings before evaluation.

Each stream is replayed chronologically from empty method state, using the current event and strictly prior context from that stream. Each decision is finalized before the current event is committed to memory; reference labels are joined after inference for scoring. Scoring includes the initial events and continues as history accumulates.

## 5 Experimental evaluation

We examine overall effectiveness (RQ1), model choice (RQ2), and memory organization and validation (RQ3), with supplementary checks of reference sensitivity and episode behavior.

## 5.1 Experimental settings

Baselines. We evaluate five external methods:

• ProactiveAgent uses its released prompt and proposal-or-null output contract, with causal history of events and decisions capped at 1,000 events (Lu et al., 2025; THUNLP, 2026).

• ContextAgent adapts proactive-service judgment to public events (Yang et al., 2025).

• Mem0 consolidates and retrieves memories for the same entity (Chhikara et al., 2025; Mem0, 2026).

• A-MEM updates linked notes and associations to retrieve relevant notes (Xu et al., 2025).

• Graphiti stores temporal event relations and retrieves facts from the same stream $( Z { \mathrm { e p } } ,$ 2026).

Metrics. Each event receives one decision finalized by intervention control; immediate and scheduled services count as intervene at the decision event. We report relevance (precision), the share of interventions matching reference opportunities; coverage (recall), the share of opportunities recovered; and effectiveness (F1), their harmonic mean. Overall effectiveness pools confusion counts across all streams and scopes. The intrusion score (false positive rate) measures the share of reference negatives receiving interventions; balance (balanced accuracy) averages coverage and the correctabstention rate on reference negatives. Higher relevance, coverage, effectiveness, and balance and lower intrusion are preferred. The frequency metric measures interventions per reference-positive event: $( \mathrm { T P + F P } ) / ( \mathrm { T P + F N } )$ , with counts pooled across streams within each scope. Appendix B.4 gives metric formulas and frequency interpretation.

Table 2: Full-scope comparison (GPT-5.6 Luna, %; frequency in ×). Bold/underline: best/second-best per scope.
<table><tr><td>Scope</td><td>Method</td><td>Frequency (×)</td><td>Relevance ↑</td><td>Coverage ↑</td><td>Effectiveness ↑</td><td>Intrusion ↓</td><td>Balance ↑</td></tr><tr><td rowspan="6">Sports</td><td>ProactiveAgent</td><td>2.75</td><td>30.0</td><td>82.5</td><td>43.9</td><td>64.5</td><td>59.0</td></tr><tr><td>ContextAgent</td><td>2.16</td><td>31.0</td><td>67.1</td><td>42.4</td><td>50.0</td><td>58.6</td></tr><tr><td>Mem0</td><td>3.40</td><td>28.8</td><td>97.8</td><td>44.5</td><td>81.0</td><td>58.4</td></tr><tr><td>A-MEM</td><td>2.41</td><td>30.6</td><td>74.0</td><td>43.3</td><td>56.0</td><td>59.0</td></tr><tr><td>Graphiti</td><td>2.20</td><td>29.3</td><td>64.4</td><td>40.3</td><td>52.0</td><td>56.2</td></tr><tr><td>ProactMem</td><td>2.15</td><td>35.8</td><td>76.7</td><td>48.8</td><td>46.1</td><td>65.3</td></tr><tr><td rowspan="6">Home</td><td>ProactiveAgent</td><td>0.78</td><td>34.2</td><td>26.8</td><td>30.1</td><td>6.8</td><td>60.0</td></tr><tr><td>ContextAgent</td><td>1.38</td><td>35.9</td><td>49.5</td><td>41.6</td><td>11.7</td><td>68.9</td></tr><tr><td>Mem0</td><td>0.73</td><td>23.9</td><td>17.4</td><td>20.1</td><td>7.3</td><td>55.0</td></tr><tr><td>A-MEM</td><td>1.63</td><td>34.6</td><td>56.3</td><td>42.9</td><td>14.1</td><td>71.1</td></tr><tr><td>Graphiti</td><td>1.53</td><td>33.4</td><td>51.1</td><td>40.4</td><td>13.5</td><td>68.8</td></tr><tr><td>ProactMem</td><td>1.52</td><td>48.6</td><td>73.7</td><td>58.6</td><td>10.3</td><td>81.7</td></tr><tr><td rowspan="6">Code</td><td>ProactiveAgent</td><td>4.45</td><td>17.5</td><td>78.2</td><td>28.7</td><td>49.7</td><td>64.2</td></tr><tr><td>ContextAgent</td><td>2.68</td><td>22.5</td><td>60.3</td><td>32.8</td><td>28.2</td><td>66.1</td></tr><tr><td>Mem0</td><td>0.65</td><td>39.8</td><td>25.9</td><td>31.4</td><td>5.3</td><td>60.3</td></tr><tr><td>A-MEM</td><td>2.55</td><td>30.4</td><td>77.6</td><td>43.7</td><td>24.0</td><td>76.8</td></tr><tr><td>Graphiti</td><td>2.36</td><td>26.8</td><td>63.2</td><td>37.7</td><td>23.3</td><td>69.9</td></tr><tr><td>ProactMem</td><td>2.24</td><td>40.1</td><td>89.7</td><td>55.4</td><td>18.1</td><td>85.8</td></tr></table>

Table 3: Within-series model scaling (%). Bold and underline mark the best and second-best scores per column.
<table><tr><td rowspan="2">Model</td><td colspan="2">Sports</td><td colspan="2">Home</td><td colspan="2">Code</td></tr><tr><td>Effectiveness ↑</td><td>Intrusion ↓</td><td>Effectiveness ↑ Intrusion↓</td><td></td><td>Effectiveness ↑</td><td>Intrusion ↓</td></tr><tr><td>GPT-5.6 Luna</td><td>48.8</td><td>46.1</td><td>58.6</td><td>10.3</td><td>55.4</td><td>18.1</td></tr><tr><td>GPT-5.6 Terra</td><td>54.1</td><td>39.4</td><td>64.4</td><td>9.1</td><td>60.3</td><td>14.9</td></tr><tr><td>GPT-5.6 Sol</td><td>58.6</td><td>34.4</td><td>69.3</td><td>7.5</td><td>66.4</td><td>11.8</td></tr></table>

![](images/01d50b5b37345ed5e88bea7066909f969ec310035a506328be9159600e7ea827.jpg)  
Figure 3: Event-only comparison of four backbones on selected streams (0–100%). Specificity = 1 − intrusion.

Implementation details. The primary model is gpt-5.6-luna, with temperature zero and thinking disabled; ProactMem uses the official OpenAI API. Methods share task semantics and the intervention controller policy, with interface-specific prompts and budgets. The event-only, event and trajectory memory, and ProactMem variants share their decision prompt, schema, and context bound of 72 hours and at most 100 events, including the current event. A-MEM and Graphiti augment that bound with their representations. For ProactMem, each event undergoes contextual assessment with prepared memory and current-condition matches, followed by intervention control. Appendix B.4 lists detailed configurations.

![](images/c187fbc73cfaf24af2bbf748c5092d5cdd045cd49da92eaad82e39fd8128d106.jpg)  
Figure 4: Memory-layer ablation (GPT-5.6 Luna) under a shared decision prompt and intervention control policy.

Table 4: Validation and Code intervention control (GPT-5.6 Luna, %). Each w/o row drops one check; before suppression scores recorded proposals with later states fixed. Bold and underline mark the best and second-best.
<table><tr><td></td><td colspan="2">Sports</td><td colspan="2">Home</td><td colspan="2">Code</td></tr><tr><td>Method</td><td>Relevance</td><td>Effectiveness</td><td>Relevance</td><td>Effectiveness</td><td>Relevance</td><td>Effectiveness</td></tr><tr><td>w/o replay gate</td><td>31.1</td><td>42.7</td><td>43.6</td><td>54.4</td><td>33.8</td><td>48.1</td></tr><tr><td>w/o trajectory validation</td><td>34.4</td><td>47.5</td><td>45.9</td><td>56.7</td><td>36.1</td><td>51.5</td></tr><tr><td>before suppression</td><td></td><td></td><td></td><td></td><td>36.7</td><td>53.6</td></tr><tr><td>ProactMem</td><td>35.8</td><td>48.8</td><td>48.6</td><td>58.6</td><td>40.1</td><td>55.4</td></tr></table>

## 5.2 Main results (RQ1)

We compare ProactMem with five external methods on ProactiveStream.

Comparisons with external methods. Across all three scopes, ProactMem achieves higher effectiveness and balance than the external baselines (Table 2, RQ1). The comparison reflects different trade-offs across scopes. On Sports, Mem0 attains greater coverage alongside more false interventions; ProactMem is more selective, with higher relevance and lower intrusion. On Home, ProactMem recovers more opportunities than every external baseline, with fewer false interventions than the higher-frequency A-MEM and Graphiti. On Code, it likewise obtains the highest coverage, whereas Mem0 achieves lower intrusion with substantially less opportunity recovery. The gains therefore come with different changes in operating behavior: more selective intervention on Sports and broader opportunity recovery on Home and Code.

Appendix D examines reference sensitivity and episode behavior on a separate diagnostic sample. Rescoring fixed diagnostic outputs against separate-team references gives similar effectiveness on Sports and lower effectiveness on Home and Code, indicating that absolute performance depends on how actionable opportunities are labeled.

## 5.3 Model comparison (RQ2)

Within-series model scaling. With all other settings fixed, construction and decision LLMs change together (Table 3). Effectiveness increases and intrusion decreases from GPT-5.6 Luna through GPT-5.6 Terra to GPT-5.6 Sol across scopes. Stronger backbones can improve both the construction of conditional regularities and their use in decisions about the current event. Controlled ablations retain GPT-5.6 Luna.

Cross-family event-only comparison. Using the shared event-only interface, we compare GPT-5.6 Luna, Gemini 3.8 Flash, DeepSeek V4 Flash, and GLM 5.3 Flash on two selected complete streams per scope (Figure 3; Appendix B.5). On Home and Code, Luna recovers more opportunities than DeepSeek and GLM, while the latter achieve higher specificity. Sports provides a contrast: Luna achieves both higher coverage and higher specificity than DeepSeek.

## 5.4 Ablation and memory analysis (RQ3)

Proactive agent ablation. Figure 4 compares event-only memory, event and trajectory memory, and the full hierarchy using Luna under shared decision settings. Trajectories extend activity context beyond recent events, while regularities organize recurring evidence around its conditions of applicability. Adding trajectory memory raises effectiveness in every scope, with the largest improvement on Code. Intrusion remains unchanged on Sports and Home and rises slightly on Code, so the gain at this stage lies mainly in opportunity recovery.

The full hierarchy improves effectiveness further and lowers intrusion across all three scopes. It retains recurring behavior together with its triggers and time windows, allowing earlier observations to inform assessment when the corresponding conditions recur. The improvement on both measures is consistent with a useful role for this more selective historical context. These configurations also use representation-dependent intervention keys, so the comparison captures memory retrieval and intervention control operating together.

Memory validation. Source validation checks trajectory fidelity; replay assesses historical support for conditional regularities. Table 4 compares both ablations with the same model and decision settings. w/o replay gate retains current matching and bypasses activation thresholds for well-formed, source-linked candidates reconstructed online. w/o trajectory validation retains schema-valid extractions for the same entity, weekly construction, and replay, omitting fact checks. Both reduce relevance and effectiveness; removing replay yields the larger effectiveness decline. On Code, trajectory validation is associated with fewer false interventions at unchanged coverage (RQ3).

Intervention control. The Code analysis reveals a trade-off between coverage and selectivity. On recorded proposals, suppression removes more false positives than true positives, improving relevance and effectiveness while reducing coverage. Later states and decisions stay fixed (Appendix C).

## 6 Related work

Agent memory. Generative Agents, Memory-Bank, and MemGPT organize experience through reflection, memory updates, or tiered context management (Park et al., 2023; Zhong et al., 2024; Packer et al., 2023). For retrieval, Mem0 uses extracted memories, A-MEM links records, and HippoRAG relies on graphs (Chhikara et al., 2025; Xu et al., 2025; Gutiérrez et al., 2024). Graphiti also tracks when event-derived relations are valid and where they originate (Zep, 2026).

Proactive agents and memory. ProactiveAgent studies whether to propose a task or remain silent given annotated events (Lu et al., 2025). To anticipate service needs, ContextAgent combines sensory contexts with historical personas (Yang et al., 2025), and PASK couples streaming demand detection with hybrid long-term memory (Xie et al., 2026). MemCog explores memory for dialogue reasoning (Li et al., 2026); Remember When It Matters uses a separate memory agent to supply selected reminders for long-horizon action (Wu et al., 2026). ProactMem checks trajectory facts against source events and tests regularities through replay before matching them to current conditions.

Evaluation of proactive service. LoCoMo and LongMemEval assess long-term conversational memory and reasoning (Maharana et al., 2024; Wu et al., 2025). For proactive agents, benchmarks examine task proposals, latent-need detection, or memory triggering (Lu et al., 2025; Xie et al., 2026; Li et al., 2026). ProactiveStream follows successive intervene or abstain decisions as memory accumulates, using isolated, causally replayed streams and episode-aware references for intervention timing. Interruption costs vary with task phase and user receptivity (Adamczyk and Bailey, 2004; Iqbal and Bailey, 2007, 2008; Mehrotra et al., 2016). We therefore evaluate both opportunity recovery and intrusion.

## 7 Conclusion

We introduced ProactiveStream to evaluate proactive assistance over event streams as history accumulates. ProactMem combines recent events, source-checked temporal trajectories, and replayassessed behavioral regularities, bringing evidence from earlier observations into current decisions. Across Sports, Home, and Code, it achieves higher effectiveness and balance than the evaluated baselines. The memory ablations further show improved opportunity recovery with trajectories and fewer false interventions with the full hierarchy. These results support maintaining historical evidence at multiple temporal scales for event-driven proactive assistance.

## Limitations

ProactiveStream covers low-risk service settings in Sports, Home, and Code under a protocol fixed after joint method and benchmark development. Independent held-out evaluation and cross-family transfer of constructed memory remain future work. Reference labels represent service opportunities under the annotation protocol; the 450-target review assesses three selected streams. The Code suppression diagnostic holds later states fixed; controller-free behavior requires a sequential comparison. Historical replay assesses observed support; prospective validity under changing behavior requires further study. Our offline metrics evaluate opportunity detection. Recipient benefit, interruption burden, end-to-end cost, and deployment safety require prospective measurements. The reported results support comparisons within the evaluated protocols, with broader generalization depending on additional populations, observation conditions, and model families.

## Ethical considerations

Our evaluation is limited to optional activity support, gentle check-ins, and low-risk repository assistance. Medical and high-risk safety decisions require separate domain-specific validation and safeguards. Inference is isolated by stream, and reference labels, episode annotations, and scoring metadata are excluded from model inputs.

Longitudinal events and derived memories can reveal sensitive routines even when identifiers are pseudonymized. Event content, prediction traces, and inferred trajectories require access and retention protections; external-model use should limit data transmission and respect source permissions.

Deployment should require informed opt-in and allow inspection, correction, deletion, retention control, and disabling proactive assistance. An intervention prediction represents a service proposal; consequential action requires separate authorization. These deployment safeguards require separate implementation and evaluation.

## References

Piotr D. Adamczyk and Brian P. Bailey. 2004. If not now, when?: The effects of interruption at different moments within task execution. In Proceedings of the SIGCHI Conference on Human Factors in Computing Systems, pages 271–278.

Prateek Chhikara, Dev Khant, Saket Aryan, Taranjeet Singh, and Deshraj Yadav. 2025. Mem0: Building production-ready AI agents with scalable long-term memory. In ECAI 2025, volume 413 of Frontiers in Artificial Intelligence and Applications, pages 2993–3000. IOS Press.

Diane J. Cook, Aaron S. Crandall, Brian L. Thomas, and Narayanan C. Krishnan. 2013. CASAS: A smart home in a box. Computer, 46(7):62–69.

Ilya Grigorik. 2026. GH Archive: A project to record the public GitHub timeline. Project website. Accessed August 21, 2026.

Bernal Jiménez Gutiérrez, Yiheng Shu, Yu Gu, Michihiro Yasunaga, and Yu Su. 2024. HippoRAG: Neurobiologically inspired long-term memory for large language models. In Advances in Neural Information Processing Systems, volume 37, pages 59532–59569.

Shamsi T. Iqbal and Brian P. Bailey. 2007. Understanding and developing models for detecting and differentiating breakpoints during interactive tasks. In Proceedings ofthe SIGCHI Conference on Human Factors in Computing Systems, pages 697–706.

Shamsi T. Iqbal and Brian P. Bailey. 2008. Effects of intelligent notification management on users and their tasks. In Proceedings of the SIGCHI Conference on Human Factors in Computing Systems, pages 93–102.

Predrag Klasnja, Shawna Smith, Nicholas J. Seewald, Andy Lee, Kelly Hall, Brook Luers, Eric B. Hekler, and Susan A. Murphy. 2019. Efficacy of contextually tailored suggestions for physical activity: A microrandomized optimization trial of HeartSteps. Annals ofBehavioral Medicine, 53(6):573–582.

Zihan Li, Xingyu Fan, Feifei Li, and Wenhui Que. 2026. MemCog: From memory-as-tool to memory-as-cognition in conversational agents. arXiv preprint arXiv:2605.28046.

Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. 2024. Lost in the middle: How language models use long contexts. Transactions ofthe Association for Computational Linguistics, 12:157–173.

Yaxi Lu, Shenzhi Yang, Cheng Qian, Guirong Chen, Qinyu Luo, Yesai Wu, Huadong Wang, Xin Cong, Zhong Zhang, Yankai Lin, Weiwen Liu, Yasheng Wang, Zhiyuan Liu, Fangming Liu, and Maosong Sun. 2025. Proactive agent: Shifting LLM agents from reactive responses to active assistance. In The Thirteenth International Conference on Learning Representations.

Adyasha Maharana, Dong-Ho Lee, Sergey Tulyakov, Mohit Bansal, Francesco Barbieri, and Yuwei Fang. 2024. Evaluating very long-term conversational memory of LLM agents. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 13851–13870.

Abhinav Mehrotra, Veljko Pejovic, Jo Vermeulen, Robert J. Hendley, and Mirco Musolesi. 2016. My phone and me: Understanding people’s receptivity to mobile notifications. In Proceedings ofthe 2016 CHI Conference on Human Factors in Computing Systems, pages 1021–1032.

Mem0. 2026. Mem0 v2.0.12. Software release. Commit 42cf18c4e6adb448e981aa1c7b55c1602b0cb670; released July 13, 2026; accessed July 18, 2026.

Charles Packer, Sarah Wooders, Kevin Lin, Vivian Fang, Shishir G. Patil, Ion Stoica, and Joseph E. Gonzalez. 2023. MemGPT: Towards LLMs as operating systems. arXiv preprint arXiv:2310.08560.

Joon Sung Park, Joseph C. O’Brien, Carrie J. Cai, Meredith Ringel Morris, Percy Liang, and Michael S. Bernstein. 2023. Generative agents: Interactive simulacra of human behavior. In Proceedings of the 36th Annual ACM Symposium on User Interface Software and Technology.

THUNLP. 2026. ProactiveAgent: Official repository. GitHub repository. Accessed July 18, 2026; commit 3fcf9beebe256b86871659fbb12541c41c9381b9.

Di Wu, Hongwei Wang, Wenhao Yu, Yuwei Zhang, Kai-Wei Chang, and Dong Yu. 2025. Long-MemEval: Benchmarking chat assistants on long-term interactive memory. In International Conference on Learning Representations.

Yifan Wu, Lizhu Zhang, Yuhang Zhou, Mingyi Wang, Bo Peng, Serena Li, Xiangjun Fan, and Zhuokai Zhao. 2026. Remember when it matters: Proactive memory agent for long-horizon agents. arXiv preprint arXiv:2607.08716.

Zhifei Xie, Zongzheng Hu, Fangda Ye, Xin Zhang, Haobo Chai, Zihang Liu, Pengcheng Wu, Guibin Zhang, Yue Liao, Xiaobin Hu, Deheng Ye, Chunyan Miao, and Shuicheng Yan. 2026. PASK: Toward intent-aware proactive agents with long-term memory. arXiv preprint arXiv:2604.08000. Technical report; work in progress.

Wujiang Xu, Zujie Liang, Kai Mei, Hang Gao, Juntao Tan, and Yongfeng Zhang. 2025. A-Mem: Agentic memory for LLM agents. In Advances in Neural Information Processing Systems, volume 38, pages 17577–17604.

Bufang Yang, Lilin Xu, Liekang Zeng, Kaiwei Liu, Siyang Jiang, Wenrui Lu, Hongkai Chen, Xiaofan Jiang, Guoliang Xing, and Zhenyu Yan. 2025. ContextAgent: Context-aware proactive LLM agents with open-world sensory perceptions. In Advances in Neural Information Processing Systems, volume 38.

Zep. 2026. Graphiti: A framework for building temporal knowledge graphs. Open-source software. Accessed September 14, 2026.

Wanjun Zhong, Lianghong Guo, Qiqi Gao, He Ye, and Yanlin Wang. 2024. MemoryBank: Enhancing large language models with long-term memory. Proceedings of the AAAI Conference on Artificial Intelligence, 38:19724–19731.

## A Benchmark and annotation protocol

## A.1 Public events and availability

Sports. HeartSteps (Klasnja et al., 2019) supplies 1,456 mobile activity meta-events from nine complete participant streams over 367 observed participant-days. Public fields include decisiontime availability, recognized activity, recent steps, location category, weather, and local decision slot. Services concern optional activity support. Inputs omit future outcomes and randomized treatments.

Home. CASAS (Cook et al., 2013) supplies ambient-sensor traces from four homes over 14 consecutive days each. Per local day, ten-minute source blocks are assigned exactly once to 28 regular 50-minute segments and one 40-minute tail, yielding 406 events per home. A segment becomes available only after interval closure and contains within-interval motion counts, room sequences, door actions, and compact numeric summaries. Gentle check-ins exclude disease, fall, travel, danger, and unsupported preference inference.

Code. Public repository activity, including GH Archive records (Grigorik, 2026), supplies 1,459 events across four repositories with 174 reference interventions. Pushes, issue updates, pull-request actions, and reviews retain GitHub’s native event granularity, with one decision per event. Records are ordered by timestamp and then event ID. Each repository owns isolated memory and intervention state even when contributors overlap. Services cover low-risk triage, review, and maintenance; native PushEvent records enter the stream.

Public-input contract. Both label classes share the same scope-specific schema: event and stream identifiers, timestamps, and contemporaneously observable fields. Reference labels, candidate flags, episode annotations, memory-level markers, and scoring masks are reserved for evaluation. Evaluation includes early events before temporal trajectories or behavioral regularities are available.

## A.2 Reference labels and causal replay

Annotators apply the opportunity and novelty semantics in Section 4.2 to each current event and its complete prior history within the same stream, independently of a tested method’s memory. LLMassisted triage prioritizes the review queue; three team members independently assign labels while blinded to decisions from the evaluated systems, followed by cross-review and adjudication of every public event.

The Home aggregation rule precedes event construction. Each evaluation version fixes the event inventory, labels, intervention prompt, event memory bound, and validation settings before main runs on the jointly developed benchmark.

Every stream starts from empty method state and is replayed chronologically under the causal execution order in Section 3.3. Current events are committed only after the output is finalized; maintenance incorporating an event affects later decisions. Labels are joined offline; reference episode states and scoring metadata are reserved for evaluation.

## A.3 Example events and reference labels

Figure 5 shows one public event record from each scope of ProactiveStream. Identifiers use event\_id; values and nesting are retained, with some fields omitted. Events are processed chronologically within each stream. The ground-truth (GT) reference label is shown above each excerpt: y<sub>t</sub> = 1 denotes intervene and y<sub>t</sub> = 0 denotes abstain. These labels are stored separately for evaluation and are excluded from agent inputs.

## B Implementation and experimental configuration

## B.1 Trajectory evidence and replay

The trajectory construction step uses an independent buffer of all committed events from the open local day, cleared after processing. Activity segments contain a local interval, typed observed, context-before, and outcome fields, plus source event identifiers. The checks in Equation 5 canonicalize scalar values, accept exact or schemapreserving suffix field paths, and require consistent entities, intervals, and resolved citations. Validation and replay operate on typed evidence. Accepted trajectories remain in the current construction cycle after their events leave event memory.

Empty or unobservable records also advance the seven-record construction cycle. Replay follows Equation 6: a triggered, observable day is fulfilled when the expected behavior occurs at or after the trigger within the window; otherwise it is violated. Missing observations are censored, each date counts at most once, and the base rate includes triggered and other observable days under the same time and weekday restrictions. Confidence and lift have zero-denominator scores of zero.

Home.   
GT: y<sub>t</sub> = 0 (abstain).

GT: y<sub>t</sub> 0 (abstain).   
{   
"event\_id": "6379c89c927a28f79bc620c3",   
"timestamp": "2015-07-22T21:31:48Z",   
"stream\_id": "heartsteps\_participant\_01",   
"observable\_payload": {   
"source\_local\_decision\_time":   
"2015-07-22T17:31:48",   
"source\_utc\_offset\_seconds": -14400,   
"available": true,   
"recognized\_activity": "STILL",   
"google\_fit\_steps\_pre30": 0,   
"google\_fit\_steps\_pre60": 73,   
"jawbone\_steps\_pre30": 261,   
"jawbone\_steps\_pre60": 456,   
"location\_category": "home",   
"weather\_condition": "Mostly Cloudy",   
"temperature\_c": 25.8   
}   
}

An activity-context record at 17:31:48 local time (UTC−04:00), with availability, location, weather, and step counts from Google Fit and Jawbone over the preceding 30 and 60 minutes.

```jsonl
{
"event_id": "casas50_d_000000",
"stream_id": "casas_home_0001",
"window_start": "2012-07-21T00:00:00",
"window_end": "2012-07-21T00:50:00",
"decision_time": "2012-07-21T00:50:00",
"duration_minutes": 50,
"motion": {
"raw_event_count": 84,
"activation_count": 43,
"deactivation_count": 41,
"location_counts": {
"Bathroom": 13,
"Bedroom": 17,
"LivingRoom": 11,
"OutsideDoor": 2
},
"location_sequence": [
"Bedroom", "LivingRoom", "OutsideDoor",
"Bedroom", "Bathroom", "Bedroom"
]
},
"door_actions": [],
"numeric_sensors": [],
"other_actions": []
}
```

This event summarizes sensor observations from 00:00 to 00:50 in source-local time and becomes available when the interval closes at 00:50. Location counts record sensor activations; the location sequence collapses consecutive repeated location labels.

```jsonl
{
"event_id": "ghe_51480602428",
"event_time": "2025-06-30T00:22:02.000Z",
"stream_id": "github_repo_169435703",
"repository": {"full_name": "callstack/react-native-pager-view"},
"event": {"action": "created", "github_type": "IssueCommentEvent"},
"object": {"number": 1005, "type": "issue"},
"content": {
"title": "java.lang.IllegalArgumentException: Scrapped or attached views may not be recycled",
"comment_body": "In my case, I was able to easily reproduce this by starting to scroll to view
the next screen and then going back before the scroll finished."
},
"attributes": {"state": "open"}
}
```

Prior context. On June 23, a collaborator requested reproduction details for this crash, and the reporter remained unable to reproduce it. Intervention opportunity. The June 30 comment supplies a concrete reproduction sequence, making targeted validation of the unresolved crash actionable.

Figure 5: Example input events and GT reference labels from Sports, Home, and Code. Excerpts preserve the original values and nesting; labels are shown separately from the inputs.

## B.2 Lifecycle and current matching

After admission, confidence combines replay and online counts in a support ratio smoothed with one initial success and failure; censored observations preserve both counts. Two consecutive violations move an active regularity to weakening, lowering retrieval priority; four move it to dormant. A fulfilled observation restores a weakening regularity to active at confidence $\geq 0 . 5$ . Dormant regularities return to candidate for replay readmission.

Current matching. Current matching is deterministic. Retrieval admits upcoming, pending, or deviated occurrences of active or weakening regularities. Occurrence priority is deviated, followed by pending and upcoming.

## B.3 Intervention control

In Sports and Home, the intervention key combines scope-specific episode state, proposed service action and target, and sorted regularity identifiers available in context. Plain summaries or graph facts supply an empty list of regularity identifiers. Sports state includes availability and activity; Home state includes activity and room and onset, confirmed, and sustained phases at one, two, and three or more consecutive matching observations. Without an explicit episode key, scope-specific time or target fields identify the opportunity, with unchanged evidence additionally required for duplicate suppression. State changes, sequence restarts, changed service or regularity identifiers, and local-day transitions can generate new keys.

Code uses the identity of the repository and object, along with the most recent allowed intervention’s mode, recipient, content, and firing condition. An exact match converts a repeated create proposal to abstain; changed signatures and other episode operations pass this check. Representationdependent key evidence can yield different suppression decisions across memory interfaces.

## B.4 Model and method interfaces

Decision and memory-construction calls use the model, API, and decoding settings in Section 5.1, with separate output budgets. Decision limits are interface-specific (256 or 512 tokens); construction limits are configured by call role. Table 5 consolidates the external-method interfaces. Each method retains its prompts, retrieval, and budgets.

Graphiti supplies relation facts directly, using the task policy and output contract with memoryspecific guidance; its ingestion episodes correspond to source records.

Scoring. Using final-decision confusion counts, relevance is $\frac { \mathrm { T } \breve { \mathrm { P } } } { \mathrm { T } \mathrm { P } + \mathrm { F P } }$ , coverage is $\scriptstyle { \frac { \mathrm { T P } } { \mathrm { T P + F N } } }$ , effectiveness is $\frac { 2 \mathrm { T P } } { \mathrm { 2 T P + F P + F N } }$ , and intrusion is $\frac { \mathrm { F P } } { \mathrm { F P + T N } }$ . Balance is $\frac { 1 } { 2 } [ \frac { \mathrm { \bar { T P } } } { \mathrm { T P + F N } } + \frac { \mathrm { T N } } { \mathrm { T N } + \mathrm { F P } } ]$ . For frequency, 1× indicates equal counts of interventions and referencepositive events; larger and smaller values indicate more and fewer interventions, respectively. Frequency is interpreted jointly with relevance, coverage, and intrusion.

## B.5 Cross-family event-only configuration

Table 6 lists the two complete streams selected within each scope. Streams are ranked by SHA256 of seed|domain|public\_stream\_id. The fixed seed is proactivestream-event-onlyni $\mathsf { n e { - } s t r e a m s { - } } 2 { \theta } 2 6 { \theta } 9 2 5 { \mathsf { - } } \mathsf { v } 1$ ; the two lowest hashes in each scope determine selection. The selection uses public stream identifiers.

Each stream is replayed chronologically from empty state. Models share the scope-specific task prompt, intervention control policy, and a causal context containing the current event and up to 99 prior events from the same stream within 72 hours. Requests use temperature zero and request disabled thinking, with decision output limits of 256 tokens for Sports and Home and 512 for Code. Outputformat and transport-recovery settings follow each backend’s interface. Final decisions are scored offline against the common reference labels.

## C Additional ablation and intervention control results

Table 7 supplements the relevance and effectiveness scores in Table 4 with frequency and coverage. On Code, suppression removes 66 falsepositive and 17 true-positive proposals; coverage falls from 99.4% to 89.7%. The diagnostic scores recorded proposals while holding later states fixed. In Figure 4, the event and trajectory memory variant records 4, 15, and 64 more true positives than event-only on Sports, Home, and Code, respectively; ProactMem records a further 57, 33, and 22.

## D Reference and episode diagnostics

## D.1 Reference sensitivity

The diagnostic selects 150 consecutive targets from one stream per scope using a fixed SHA256 ranking of eligible streams and contiguous starts based only on public metadata, with at least seven days of preceding history. The streams are Sports participant\_35, Home home\_0004, and Code repository callstack/react-native-pager-view. Their target periods are December 16, 2015–January 18, 2016; June 23–28, 2011; and June 21–November 4, 2025. All 37, 203, and 152 preceding events are processed chronologically, respectively: 842 public events supply context for 450 scored targets.

Table 5: External-method memory interfaces and implementation settings; all methods share the backbone.
<table><tr><td>Method</td><td>Interface and Settings</td></tr><tr><td>ProactiveAgent</td><td>Released prompt and proposal/null contract on the study backbone; causal event– decision history capped at 1,000 events per request including the current event (Lu et al., 2025; THUNLP, 2026).</td></tr><tr><td>ContextAgent Mem0</td><td>In-context proactive-service judgment on public events (Yang et al., 2025). Open-source mem0ai==2.0.12 extraction, consolidation, and retrieval; local</td></tr><tr><td></td><td>al1-MiniLM-L6-v2 embeddings; retrieval of up to eight memories for the same entity with search threshold 0.1 (Chhikara et al., 2025; Mem0, 2026).</td></tr><tr><td>A-MEM Graphiti</td><td>Linked notes with association updates augment bounded context (Xu et al., 2025). Graphiti 0.30.2 with Neo4j and local 384-dimensional a11-MiniLM-L6-v2 embed-</td></tr><tr><td></td><td>dings; committed events populate separate graphs for each stream. Queries from the current event use native hybrid edge search with reciprocal-rank fusion, retrieving up to ten relation facts with validity fields and source IDs (Zep, 2026).</td></tr></table>

Table 6: Streams in the cross-family comparison.
<table><tr><td>Scope</td><td>Source Stream</td><td>Events</td></tr><tr><td rowspan="2">Sports</td><td>HeartSteps participant 05</td><td>184</td></tr><tr><td>HeartSteps participant 35</td><td>191</td></tr><tr><td rowspan="2">Home</td><td>CASAS home 0004</td><td>406</td></tr><tr><td>CASAS home 0003</td><td>406</td></tr><tr><td rowspan="2">Code</td><td>Unitech/pm2</td><td>365</td></tr><tr><td>thephpleague/commonmark</td><td>347</td></tr><tr><td>Total</td><td>six complete streams</td><td>1,899</td></tr></table>

A separate team of three reviewers independently labels each target using prior history from the same stream, blinded to outputs from the evaluated systems, then discusses judgments to obtain consensus references. Earlier references use original Sports and Home labels and 122 original plus 28 additional Code annotations. Relative to these references, Sports and Home each have 13 changes from intervene to abstain, with 4 and 0 changes in the reverse direction, respectively; Code has 6 removals and 16 additions to positive labels.

Table 8 scores identical recorded ProactMem outputs against these two sets of references. The predictions come from separate diagnostic runs on the selected targets. The diagnostic uses Luna with temperature zero, thinking disabled, decision limits of 256 tokens for Sports and Home and 512 for Code, and the same context bound of 72 hours and at most 100 events, trajectory validation, replay gating, and intervention control policy. Immediate and scheduled services both count as intervene.

Table 7: Additional validation and Code intervention control scores. Frequency uses multiples (×); coverage uses percentages. Bold and underline mark the best and second-best coverage per scope, including ties.
<table><tr><td rowspan="2">Scope</td><td rowspan="2">Method</td><td rowspan="2">Frequency</td><td rowspan="2">Coverage ↑</td></tr><tr><td>(×)</td></tr><tr><td>Sports</td><td>w/o replay gate w/o trajectory validation ProactMem</td><td>2.19 2.23 2.15</td><td>68.2 76.7 76.7</td></tr><tr><td>Home</td><td>w/o replay gate w/o trajectory validation ProactMem</td><td>1.65 1.62 1.52</td><td>72.1 74.2</td></tr><tr><td>Code</td><td>w/o replay gate w/o trajectory validation before suppression</td><td>2.47 2.48 2.71</td><td>73.7 83.3 89.7 99.4</td></tr></table>

Rescoring against the separate-team references yields similar effectiveness on Sports and lower effectiveness on Home and Code relative to the earlier references. Absolute scores are label-sensitive, especially on Home and Code.

Table 8: Reference sensitivity on 150 targets per scope. Effectiveness (%) scores identical outputs under earlier and separate-team references. Positives and confusion counts use the latter; I counts final interventions.
<table><tr><td colspan="8">Earlier Diagnostic</td><td rowspan="2">Separate-Team Effectiveness</td></tr><tr><td>Scope</td><td>Positives</td><td>I</td><td>TP</td><td>FP</td><td>FN</td><td>TN</td><td>Effectiveness</td></tr><tr><td>Sports</td><td>27</td><td>86</td><td>24</td><td>62</td><td>3</td><td>61</td><td>42.6</td><td>42.5</td></tr><tr><td>Home</td><td>15</td><td>39</td><td>15</td><td>24</td><td>0</td><td>111</td><td>68.7</td><td>55.6</td></tr><tr><td>Code</td><td>34</td><td>42</td><td>18</td><td>24</td><td>16</td><td>92</td><td>57.6</td><td>47.4</td></tr></table>

Table 9: ProactMem episode diagnostic on 150 targets per scope. Coverage, SR, and RIR use percentages. Unresolved denotes ambiguous semantic or condition matches; uncertain follow-ups (U) are zero in all scopes.
<table><tr><td>Scope</td><td>Covered/New (Coverage)</td><td>I</td><td>F</td><td>V</td><td>R</td><td>Unmatched</td><td>Late</td><td>Unresolved</td><td>SR</td><td>RIR</td></tr><tr><td>Sports</td><td>23/26 (88.5)</td><td>86</td><td>23</td><td>1</td><td>1</td><td>60</td><td>0</td><td>1</td><td>27.9</td><td>1.2</td></tr><tr><td>Home</td><td>9/9 (100.0)</td><td>39</td><td>10</td><td>6</td><td>17</td><td>6</td><td>0</td><td>0</td><td>41.0</td><td>43.6</td></tr><tr><td>Code</td><td>14/31 (45.2)</td><td>42</td><td>14</td><td>3</td><td>4</td><td>16</td><td>4</td><td>1</td><td>40.5</td><td>9.5</td></tr></table>

## D.2 Episode references and matching

A model-assisted post-hoc analysis on these targets reconstructs episodes defined by service goals and objects from positive event anchors. Each reference specifies a service goal and object, causal evidence, onset, explicitly listed valid observations, and any deadline or closure evidence. Semantic matching uses these fields to associate interventions with reference episodes.

Matching and follow-up. Episode coverage follows ProactMem’s own interventions, initialized from recorded prefix outputs. Only target interventions are scored. An intervention must match the goal and object at a listed valid observation. Its proposed service time is the decision time plus its specified delay (zero for immediate service), and must meet any explicit episode deadline. The first timely match covers the episode. Each output receives at most one credit: the first uncovered compatible episode in the fixed reference order, otherwise the first already covered one.

Subsequent matches are compared with the method’s preceding assistance for that episode. A valuable follow-up requires new causal evidence that changes the actionable service or next step, and a message addressing that change. Rephrasing or continued status alone is redundant; value requiring unobserved delivery, recipient response, or completion remains uncertain. Each reference episode receives coverage credit once.

Metrics and boundaries. Let I count all target interventions, F first valid interventions across all reference episodes, V valuable follow-ups, R redundant follow-ups, and U uncertain follow-ups. New reference episodes have onsets within the target interval. Episode coverage divides covered new episodes by all new reference episodes; supported relevance is $\mathrm { S R } = ( F + V ) / I$ , and redundant intervention rate is $\mathrm { R I R } = R / I$ . Uncertain, unmatched, late, and unresolved outcomes contribute only to the intervention count I. Table 9 combines the rates and outcome counts.

Episode coverage is highest on Home and lowest on Code; the redundant-intervention share is highest on Home and lowest on Sports.

The exploratory scores depend on reference construction and semantic matching.