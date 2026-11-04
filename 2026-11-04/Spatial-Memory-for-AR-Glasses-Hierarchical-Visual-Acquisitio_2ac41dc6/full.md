# Spatial Memory for AR Glasses: Hierarchical Visual Acquisition and Source-Linked Recall

Tianxing Zhu<sup>\*</sup>, Dan Huang <sup>\*</sup>Independent Researcher, Shanghai, China

## Abstract

We present a spatial-memory system that enables optical seethrough AR glasses to recall earlier physical encounters. The dual-camera prototype combines hierarchical visual acquisition with source-linked storage and staged retrieval. Quality screening, global and regional appearance novelty, and temporal support select monochrome observationsfor detailed RGB capture. Semantic observations retain links to their source images, allowing retrieval to answer indexed historical questions directly or inspect visual details omitted from scene descriptions. The architecture separates capture acceptance, interpretation outcome, and support for aparticular question. This distinction motivates two core control mechanisms: bounded reassessment of familiar views when earlierprocessing leaves an evidence gap, and retention guided by source dependencies. In a 131.35 s egocentric replay, the implemented acquisition policy selected six RGB viewsfrom 648 monitoring observations, whilefeature extraction ran on 97.7% of observations. These results demonstrate selective RGB acquisition and identify frequent local inference as a remaining cost. Exposure duration and cross-view timing further constrain the evidence preserved from selected encounters. The system provides a practical acquisition-to-recall architecturefor persistent AR assistance, with explicit boundaries between captured evidence, indexed interpretations, and supported historical claims.

## Author Keywords

optical see-through AR; spatial memory; hierarchical visual acquisition; source-linked recall.

## 1. Introduction

Persistent memory lets an AI assistant carry information across interactions. Hierarchical context management, editable memory blocks, and durable task artifacts support this continuity in digital environments [1, 2, 3]. Physical assistance also needs a record of objects, places, and activities encountered outside those interactions. A mug left on a desk or a detail briefly visible on a shelf may never enter a message or document. Answering a later question about that encounter depends on acquiring and retaining its evidence in advance.

Optical see-through AR glasses combine a first-person camera viewpoint with a display available in the physical environment. Our prototype uses RayNeo X3 Pro glasses, with monochrome and RGB cameras and a binocular full-color MicroLED difractivewaveguide display [4]. The wearer can retrieve an earlier encounter after the object or scene has left the current view. This requires sufficient visual detail during acquisition and a readable answer within the limited near-eye viewing field.

Acquisition decisions determine what the assistant can remember. Frequent recording accumulates similar views; aggressive selection risks losing brief encounters. Compact scene descriptions make retrieval eficient but can omit details needed by a later question. Interpretation can also fail after an image has been saved. If the capture gate treats that view as adequately remembered, it may suppress a later observation that could fill the gap.

We developed a spatial-memory system that selects monochrome views for detailed RGB capture and stores object-place descriptions with their source images and acquisition times. Retrieval answers from an indexed historical observation when it matches the question, then escalates to image-backed reasoning when further detail is needed. The memory represents physical encounters through these observations and semantic relations.

The contributions are:

1. A hierarchical dual-camera acquisition method that combines global and regional appearance novelty with temporal support to select detailed RGB evidence from monochrome monitoring.

2. A source-linked memory and staged-retrieval architecture that uses searchable interpretations to access original visual evidence.

3. An outcome-aware control design that uses evidence deficits to guide bounded reassessment and source retention.

The prototype implements this complete acquisition-to-recall pipeline. Causal replay, exposure metadata, and memoryprocessing outcomes characterize how selection and evidence availability depend on camera geometry and acquisition timing.

## 2. Related Work

Egocentric episodic memory and historical retrieval: Ego4D includes episodic-memory queries among its egocentric perception tasks, alongside present-activity understanding and futureevent forecasting [5]. Such queries recover earlier observations of objects or events that may have left the current view. A wearable assistant must preserve the relevant encounter through acquisition and storage decisions made before the question arrives.

Bärmann and Waibel formulate episodic-memory question answering with a constant-size video representation and introduce QAEgo4D [6]. This makes query-independent encoding and memory capacity explicit. In a wearable system, information can be removed before detailed capture, during semantic interpretation, or through later retention. The resulting losses difer: discarding a caption can leave its source image available, whereas rejecting an encounter before detailed capture removes its fine visual content.

ExoMem combines an AR headset, visual processing, and localization to build an external object-location memory [7]. ESOM constructs compact egocentric object memory for online visual-query localization [8]. Both address recall from physical encounters and the construction of memory over time. Our system uses language queries over semantic relations and source images, with acquisition decisions determining the evidence available for recall.

EgoLife studies daily-life assistance using wearable multimodal observations and long-context question answering [9]. It introduces EgoLifeQA and a retrieval-based assistant architecture. Sustained assistance requires memories that answer questions whose required detail was unknown at acquisition. Our work addresses the capture and evidence management needed for that use.

Persistent agent memory and source representations: MemGPT manages memory tiers to extend efective languagemodel context [1]. Letta provides persistent editable memory blocks, and long-running agent harnesses preserve task state in progress artifacts [2, 3]. Their access and persistence mechanisms support continuity across interactions and inform the organization of wearable memory.

Visual memories also need a record of their origin. A scene description interprets an image captured at a particular time and viewpoint. It indexes that image but may omit readable text, a spatial relation, or a small object requested later. Keeping the source image lets retrieval choose between the semantic index and direct inspection at the required level of detail.

Descriptions, indexes, and derived claims can change while source identity and acquisition time stay fixed. Processing updates therefore preserve the history of physical encounters.

Dynamic scene memory and optical see-through AR assistance: Embodied VideoAgent builds persistent scene memory from egocentric video and embodied sensing, including depth and pose, and updates it with perceived activities [10]. R4DSG indexes object-centric entries by time, place, persistent identity, and anchorrelative change [11]. Both represent evolving scenes. Their treatment of identity and temporal change calls for associations across observations as well as descriptions of individual images.

Our memory stores source observations and semantic relations. Class-level sightings support historical recall; identity associations across sources and their observation intervals can support later temporal models.

Memento uses spatiotemporal and activity context to recall remembered interactions and provide proactive AR responses [12]. Presentation through an optical see-through head-mounted display brings optical constraints. Grubert et al. survey eye-to-display calibration for aligning virtual content with physical objects [13]. Gabbard et al. show how background texture, natural illumination, and drawing style afect outdoor AR text legibility [14]. Erickson et al. study interface color schemes under additive display optics, diferent backgrounds, and lighting conditions [15]. Gabbard, Mehra, and Swan report performance costs and visual fatigue during repeated context and focal-distance switching [16].

Source-linked semantic relations and timestamps allow concise historical recall across presentation formats. World-registered cues can use localization and calibration, while the underlying memory retains the source and time of the encounter.

Adaptive acquisition, cascaded sensing, and the capture-memory boundary: Continuous vision consumes resources in image acquisition, processing, and inference. RedEye moves early convolutional processing into the sensor’s analog domain before quantization, reducing readout and downstream processing costs [17]. Buckler et al. study configurable photography and vision modes with reduced sensor precision, resolution, and ISP processing [18]. Glimpse combines local tracking and scenechange triggers with selective server ofloading and cached frames to handle recognition latency [19]. AdaFrame adapts frame selection for eficient recognition [20], while JOCA jointly optimizes camera hardware and adaptive control for downstream tasks [21].

We apply staged selection to an open-ended memory task. Lowresolution monochrome monitoring selects detailed RGB acquisitions for later interpretation. Because the question often arrives after the encounter, selection must preserve evidence for future queries while limiting acquisition and interpretation demand.

Visual novelty and memory value can diverge. A novel view may contribute little to a later task, a familiar scene may contain an important small change, and a captured image may fail interpretation. Our outcome-aware controller uses interpretation and retrieval outcomes to reconsider capture decisions when the acquired view has left an evidence gap.

## 3. System Architecture and Hierarchical Acquisition

Optical platform and acquisition-to-recall framework: The prototype runs on RayNeo X3 Pro (ARGF20), whose published display specifications are a binocular difractive waveguide, full-color MicroLED illumination, 640 480 resolution, and a 30° FOV [4]. Camera metadata reports a 640 480 monochrome sensor and a 4032 3024 native RGB array. The reference recording uses 640 480 monochrome and 1280 720 RGB streams. Monochrome monitoring selects encounters, RGB acquisition preserves detail, and the near-eye display presents recalled information.

Acquisition selects detailed images from the available observation history before a future question is known. Interpretation converts these images into source-linked semantic records, and retention determines which source images remain available. When a question arrives, retrieval selects relevant evidence from the retained sources to support an answer.

The architecture distinguishes three states:

• Acquisition history: visual signatures of views accepted for detailed acquisition.

• Memory evidence state: retained source images, their processing outcomes, and committed interpretations and quality attributes.

• Query support: retained evidence suficient for the claim requested by a particular question.

Separating these states supports asynchronous acquisition, interpretation, and retrieval. An image stays available after interpretation fails, and retrieval can inspect it when a description omits a needed detail.

Camera FOV estimates follow the rectilinear relation $\theta \quad =$ 2 arctan(d/(2f)), with reported sensor dimension d and focal length f [22]. These full-array estimates use sensor axes. Stream cropping, lens distortion, and the RGB stream’s 90° presentation rotation afect the displayed camera view; the 30° display FOV belongs to a separate optical path.

Feature computation, detailed acquisition, interpretation requests, transmitted image data, and retained bytes incur separate costs. Selective RGB capture can coexist with frequent local inference.

Escalation from monitoring to detailed evidence: The acquisition hierarchy uses successive tests to decide which observations receive more detailed processing (Figure 1). Quality and visual-change checks operate on the low-resolution monochrome stream. Appearance analysis compares eligible observations with accepted views, and temporal support confirms candidates for RGB acquisition and scene interpretation. Each stage can reject a candidate before further work is incurred.

Table 1. Optical channels and their roles in the memory system.
<table><tr><td rowspan=1 colspan=1>Channel</td><td rowspan=1 colspan=1>Optical or sensor characteristics</td><td rowspan=1 colspan=1>Role and design consequence</td></tr><tr><td rowspan=1 colspan=1>Near-eye display</td><td rowspan=1 colspan=1>Manufacturer-specified binocular MicroLED/diffractivewaveguide; 640 × 480; 30° FOV [4]</td><td rowspan=1 colspan=1>Concise recall within a limited display field</td></tr><tr><td rowspan=1 colspan=1>Monochrome camera</td><td rowspan=1 colspan=1>Device-reported 640 × 480; 1.92 × 1.44 mm sensor; 0.71mm focal length; f/2.0; nominal sensor-axis FOV $1 0 7 . 0 ^ { \circ } \times$ 90.8°</td><td rowspan=1 colspan=1>Broad monitoring view for global and regionalselection</td></tr><tr><td rowspan=1 colspan=1>RGB camera</td><td rowspan=1 colspan=1>Sony IMX681 (manufacturer [4]); device-reported 4032 ×3024 native array; 4.032 × 3.024 mm sensor; 1.90 mmfocal length; f/2.25; nominal sensor-axis FOV $9 3 . 4 ^ { \circ } \times$ 77.0°</td><td rowspan=1 colspan=1>Detailed source evidence with a different viewinggeometry and exposure state</td></tr></table>

![](images/5902b740a9a6667e8bcc06af52cda2bf34ebd280917e41dd0354f9e99b1ea782.jpg)  
Figure 1. Hierarchical capture and evidence feedback.

The first stage checks brightness and sharpness, then compensates for exposure variation using mean and bounded-gain correction before computing luminance residuals. Whole-view residuals detect broad changes; overlapping patch residuals detect changes confined to smaller regions and can wake feature analysis. Pending candidates continue to receive feature checks after adjacent frames become similar so that temporal support can accumulate.

Global and regional novelty: A mobile image embedder derived from MobileNetV3 and accessed through MediaPipe [23, 24] produces a normalized full-frame vector $f _ { t } ^ { \overline { { 0 } } }$ and four overlapping regional vectors $\underline { { f } } _ { t } ^ { r }$ . For normalized vectors, the distance is $d ( \mathbf { u } , \mathbf { v } ) = \mathrm { c l i p } ( 1 - \mathbf { u } ^ { \mathsf { T } } \mathbf { v } , 0 , 2 )$

For the accepted-view bank $B _ { t }$ , global novelty is the nearest fullframe distance,

$$
G _ { t } = \operatorname* { m i n } _ { j \in \mathcal { B } _ { t } } d ( f _ { t } ^ { 0 } , f _ { j } ^ { 0 } ) .\tag{1}
$$

Regional comparison uses the h globally nearest reference views, $\mathcal { N } _ { h } ( t )$ . Each current region searches their regional vectors. The largest of the resulting nearest-region distances is

$$
R _ { t } = \operatorname* { m a x } _ { r = 1 , \ldots , 4 } \operatorname* { m i n } _ { j \in \mathcal { N } _ { h } ( t ) } \operatorname* { m i n } _ { s = 1 , \ldots , 4 } d ( f _ { t } ^ { r } , f _ { j } ^ { s } ) .\tag{2}
$$

A candidate satisfies either the global or regional novelty condition:

$$
n _ { t } = { \bf 1 } [ G _ { t } \geq \tau _ { g } ~ \lor ~ ( R _ { t } \geq \tau _ { r } \land G _ { t } \geq \tau _ { \operatorname* { m i n } } ) ] .\tag{3}
$$

The global branch admits broad scene transitions. The regional branch responds to localized diferences with some full-view support. Searching across reference regions accommodates appearances that shift between image corners. Taking the largest unmatched regional distance preserves sensitivity to a local diference even when the full view has a close reference.

An empty bank initiates coverage. Accepted views then form a bounded acquisition history. Viewpoint, illumination, and object changes all afect appearance distances; the retained evidence determines their value for later questions.

Temporal support and asynchronous acquisition: A candidate needs a minimum supporting observation count $n _ { \mathrm { m i n } }$ and elapsed duration $T _ { \mathrm { m i n } }$ . The count suppresses isolated responses, while duration prevents a dense burst of samples from qualifying as a sustained encounter. An appearance anchor limits feature drift to the threshold δ. Excessive drift starts a new candidate; familiar observations and failed quality checks clear the pending candidate.

Longer temporal support rejects transient triggers but can delay or lose a brief view. In the tested still-capture path, the application closes monochrome monitoring, opens the RGB session, runs a 1 s preview warm-up, acquires a JPEG, and resumes monitoring. Camera closure, session configuration, and exposure timing also contribute to the handof.

A feature selected in the monochrome view must stay visible and suficiently resolved in the RGB image. FOV diferences, aspectratio cropping, camera separation, and head rotation can shift it or move it outside the detailed view. A local pinhole approximation gives

$$
\begin{array} { r l } & { \Delta u _ { \mathrm { r o t a t i o n } } \simeq f _ { \mathrm { p x } } | \omega | \Delta t , } \\ & { ~ \Delta u _ { \mathrm { p a r a l l a x } } \simeq f _ { \mathrm { p x } } \frac { | b _ { \perp } | } { Z } , } \\ & { ~ \ell _ { \mathrm { b l u r } } \simeq f _ { \mathrm { p x } } | \omega | T _ { e } . } \end{array}\tag{4}
$$

where $f _ { \mathrm { p x } }$ is the local focal scale in pixels, ω is angular velocity in radians per second, $\Delta t$ is the inter-view acquisition ofset, $b _ { \perp }$ is the baseline component transverse to the viewing direction, Z is scene depth, and $T _ { e }$ is exposure time. Rotation and parallax change framing, while exposure integrates motion into blur.

Temporal persistence checks the monochrome candidate before promotion. The RGB quality stage then assesses framing and blur in the detailed source image.

Acquisition waits for downstream processing capacity. Monitoring can resume while interpretation of earlier images is still pending. Candidate persistence follows observation time independently of interpretation completion.

Table 2. Representational distinctions preserved by the memory hierarchy.
<table><tr><td rowspan=1 colspan=1>Distinction</td><td rowspan=1 colspan=1>Representation</td><td rowspan=1 colspan=1>Consequence for recall</td></tr><tr><td rowspan=1 colspan=1>Observation versus interpretation</td><td rowspan=1 colspan=1>Stable source identity with a linkeddescription</td><td rowspan=1 colspan=1>A revised description remains tied to theoriginal encounter</td></tr><tr><td rowspan=1 colspan=1>Acquisition time versus processing time</td><td rowspan=1 colspan=1>Capture-time ordering independent ofcompletion</td><td rowspan=1 colspan=1>Each response retains the time of its sourceencounter</td></tr><tr><td rowspan=1 colspan=1>Class sighting versus persistent identity</td><td rowspan=1 colspan=1>Entity labels and appearance tied to a source</td><td rowspan=1 colspan=1>Similar objects can remain identity-ambiguous</td></tr><tr><td rowspan=1 colspan=1>Historical versus current state</td><td rowspan=1 colspan=1>Time-qualified relations</td><td rowspan=1 colspan=1>Reports the observed location and time</td></tr><tr><td rowspan=1 colspan=1>Indexed fact versus original detail</td><td rowspan=1 colspan=1>Search structure linked to its image</td><td rowspan=1 colspan=1>Retrieval can escalate when a question exceedsthe description</td></tr><tr><td rowspan=1 colspan=1>Processing outcome versus query support</td><td rowspan=1 colspan=1>Outcome and quality attributes assessed inquery context</td><td rowspan=1 colspan=1>A completed record may still be insufficient fora claim</td></tr></table>

## Algorithm 1. Causal acquisition with separate capture and semantic commits.

For each observation x at acquisition time t:

if image quality is inadequate: clear candidate; continue if no wake condition and no candidate needs verification: continue

compute full-frame and regional appearance features evaluate global novelty G and regional novelty R if the novelty condition fails: clear candidate; continue initialize or extend a candidate consistent with its appearance anchor

if temporal support or downstream capacity is insuficient: continue

acquire detailed RGB evidence

if image persistence and queue admission fail: continue add the triggering signature to acquisition history interpret the saved image asynchronously commit a valid source-linked observation or preserve the failure outcome

Capture acceptance and semantic commitment occur at diferent times. A saved view can therefore become familiar to the gate before interpretation succeeds.

## 4. Source-Linked Memory and Outcome-Aware Coordination

Logical memory hierarchy and invariants: The memory hierarchy has four roles (Figure 2): working acquisition context (M0), indexed observations (M1), durable source evidence (M2), and derived temporal knowledge (M3).

The observation is the durable unit of evidence, represented as $o _ { i } = \left( \mathrm { i d } _ { i } , t _ { i } , \eta _ { i } , I _ { i } , D _ { i } , E _ { i } , \mu _ { i } \right)$ , where $t _ { i }$ is acquisition time, η<sub>i</sub> records time quality, $I _ { i }$ is the source image, $D _ { i }$ is the scene descrip tion, $E _ { i }$ contains entity-level descriptions and spatial relations, and $\mu _ { i }$ identifies the interpretation model. The source identity remains stable across retrieval and later interpretation.

Scene descriptions provide semantic access to source images. An entity entry records an object class, place, and time; questions about readable text or a disputed relation can return to the image. The semantic index thus provides eficient access while the source preserves further visual detail.

Reprocessing preserves encounter identity, acquisition time orders observations independently of response arrival, and derived claims retain their supporting source interpretations. These update rules preserve encounter history during asynchronous processing.

Memory retains positive observations and uncertain views. A later description that omits an object leaves its earlier sightings intact.

A temporal extension can associate sightings of the same object at diferent locations at t<sub>1</sub> and t<sub>2</sub> to represent a change interval [t<sub>1</sub>, t<sub>2</sub>]. The derived record retains those sources and identity assumptions, allowing revision when a supporting source is corrected or removed.

Staged evidence access and query support: Retrieval chooses an access path for the requested claim. Historical location questions first use the index. An eligible class-level sighting returns its recorded place, relation, acquisition time, and source. Ambiguous identity, unsuitable source quality, or a question beyond that indexed relation triggers image inspection.

General questions use a catalog of at most K observations selected for recency and lexical relevance. A selection stage chooses at most L sources for image-backed answering, giving catalog search and visual inspection separate budgets. Descriptions supply temporal and semantic context; images supply details omitted during memory construction.

Retrieval can fail during source selection or image interpretation. A relevant observation may be excluded from the catalog or selected set, or the answer stage may misread a supplied image. Originalimage access can recover omitted detail when selection retrieves that source.

Source IDs returned with an answer are checked against the supplied evidence to validate provenance. The sources remain available for inspection. When their content does not support the requested claim, the system can return an insuficient-evidence response.

Support depends on the question. A source may establish “the mug was last seen on the desk” while leaving “the mug is on the desk now” unresolved. A description that lists object classes may omit a serial number, ownership, or exact identity. An image marked globally uncertain may still contain a useful local detail.

Processing outcome records whether a step completed, interpretation quality describes the view, and query support relates a claim to its sources. Keeping these attributes separate allows an imperfect observation to remain useful for questions it can answer.

Table 3. Outcome-dependent control actions.
<table><tr><td rowspan=1 colspan=1>Available evidence state</td><td rowspan=1 colspan=1>Control action</td><td rowspan=1 colspan=1>Reason</td></tr><tr><td rowspan=1 colspan=1>Interpretation pending</td><td rowspan=1 colspan=1>Wait and suppress duplicate work</td><td rowspan=1 colspan=1>The acquisition has not yet resolved into asemantic outcome</td></tr><tr><td rowspan=1 colspan=1>Transient interpretation failure</td><td rowspan=1 colspan=1>Consider bounded reuse of the saved image</td><td rowspan=1 colspan=1>A suitable saved image can support anotherinterpretation attempt</td></tr><tr><td rowspan=1 colspan=1>Visual evidence inadequate and a clearerview recurs</td><td rowspan=1 colspan=1>Admit a bounded repeat acquisition</td><td rowspan=1 colspan=1>New detail may repair an existing evidencedeficit</td></tr><tr><td rowspan=1 colspan=1>Committed evidence supports therelevant claim</td><td rowspan=1 colspan=1>Preserve the source and suppress redundantwork</td><td rowspan=1 colspan=1>Additional acquisition has low immediateevidential value</td></tr><tr><td rowspan=1 colspan=1>Relevant source excluded by search</td><td rowspan=1 colspan=1>Expand or rerank retrieval within its budget</td><td rowspan=1 colspan=1>The relevant visual evidence already exists inmemory</td></tr><tr><td rowspan=1 colspan=1>Past encounter never acquired</td><td rowspan=1 colspan=1>Preserve the historical gap</td><td rowspan=1 colspan=1>Historical claims require evidence from therelevant encounter</td></tr></table>

![](images/3cb0807d535f9d8f5776d00caf024227c829168051335c8679f2692eeafe1ca7.jpg)  
Figure 2. Memory organization and staged recall. Indexed observations support local historical lookup; general questions use catalog selection and image-backed answering.

Outcome-aware reassessment: The outcome-aware controller feeds memory outcomes back to acquisition (Figures 1 and 2). It connects each accepted visual signature to its processing outcome and unresolved evidence needs. A familiar view can then be reconsidered when an earlier capture failed to provide adequate memory, while redundant captures remain suppressed.

A control record stores source identity and appearance, processing outcome, evidence attributes, unresolved needs, and the remaining work budget. An outcome becomes eligible for control when processing completes, preserving causal order under variable interpretation delay.

Reassessment chooses between reinterpreting a saved image and acquiring a new one. A suitable image can be reused after processing fails. A new view is useful when it can improve the visual evidence. Retry bounds and termination conditions limit repeated work after systematic failures.

An evidence need records the target, unsupported claim, temporal scope, expiry, and remaining budget. Current-location questions can use fresh observations; historical questions need sources from the relevant interval. Appearance-based place associations identify candidate revisits.

Feature checks can be spaced farther apart during stable, familiar views and brought closer together for a pending candidate or unresolved evidence need. A maximum refresh interval provides checks for local changes missed by the inexpensive wake stage.

Source-preserving retention: Retention determines which sources remain available to future questions. For total memory budget $B _ { M }$ , the retained set S must satisfy

$$
\begin{array} { r l r } {  { \sum _ { i \in S } \mathrm { b y t e s } ( I _ { i } , D _ { i } , E _ { i } ) } } \\ & { } & { + \mathrm { b y t e s } ( \operatorname { a c c e s s \ s t r u c t u r e s \ a n d \ d e r i v e d \ r e c o r d s } ) } \\ & { } & { \leq B _ { M } . } \end{array}\tag{5}
$$

Source selection weighs unique support, encounter coverage, improved detail, and unresolved needs. Similar-looking images can preserve diferent text or object states; diferently worded descriptions can refer to nearly identical visual evidence.

Retention tracks the evidence afected by each operation. A rebuildable access vector can be discarded, image compression can remove small-object or text detail, and deleting a source can remove the sole support for a claim. Dependent indexes and summaries must then be updated or marked unsupported. Summaries retain selected facts; source images preserve pixel-level detail for later inspection.

Evidence dependencies guide source selection. FIFO, uniform sampling, and similarity-based selection provide reference policies at equal retained bytes for assessing their efect on supported recall.

## 5. Prototype Characterization and Discussion

Causal replay and selection behavior: We replayed paired monochrome VGA and RGB reference video from an egocentric walk over a common interval of 131.35 s. VGA observations were processed in source-time order, with each request paired to the first subsequent RGB acquisition within the allowed pairing interval.

In the pouring encounter (Figure 3), the ofline description records the activity but omits the mug-handle orientation visible in the source crop. The example question uses that detail to illustrate source inspection.

The policy selected six RGB observations from 648 analyzed VGA observations, or 2.74 selections per minute. Raising the capture ceiling left the count at six, with no budget blocking. The matchedcount periodic reference also retained six observations.

Feature extraction ran on 633 observations (97.7%); the luminance

(c) Description and source detail Existing offline description (excerpt "hands pouring amber liquid from a glass pitcher into a metal mug on a countertop' Example question On which side is the mug handle visible (a) Monochrome monitoring

![](images/2a769e46ec93d418ac2a8563c71b3f5222a7c1169beae1f1044402a9c17e19b9.jpg)

(b) Paired RGB evidence  
![](images/80f40ee4d695c11de15eb37871c098a4812cb4bf44e8100b11272e51d0fabe6e.jpg)

![](images/aa7fe5be6b6378d7e0a1853e4a5c1c13f02f9ac4ebb605d7e91a3568eca522f8.jpg)

Figure 3. Recorded encounter and source-detail walkthrough. (a) The VGA observation with the four overlapping regional encoder crops. Both novelty scores exceed their thresholds, and the global branch admits this candidate. (b) A countertop crop of the paired RGB reference frame, acquired 29.4 ms after the VGA frame and 18.4 ms after the replay request. The cameras provide distinct fields of view. (c) An excerpt from the existing ofline audit and an enlarged source detail: the mug handle is visible on the image-left side, a relation absent from the excerpt. The question illustrates source inspection using the recorded reference pair.  
Table 4. Imaging conditions of the six selected RGB reference acquisitions.
<table><tr><td rowspan=1 colspan=1>RGB time (s)</td><td rowspan=1 colspan=1>VGA-to-RGB offset (ms)</td><td rowspan=1 colspan=1>Request-to-RGB offset (ms)</td><td rowspan=1 colspan=1>RGB exposure (ms)</td></tr><tr><td rowspan=1 colspan=1>6.95</td><td rowspan=1 colspan=1>11.13</td><td rowspan=1 colspan=1>0.48</td><td rowspan=1 colspan=1>10.00</td></tr><tr><td rowspan=1 colspan=1>14.35</td><td rowspan=1 colspan=1>22.49</td><td rowspan=1 colspan=1>11.48</td><td rowspan=1 colspan=1>10.00</td></tr><tr><td rowspan=1 colspan=1>27.33</td><td rowspan=1 colspan=1>29.39</td><td rowspan=1 colspan=1>18.41</td><td rowspan=1 colspan=1>19.99</td></tr><tr><td rowspan=1 colspan=1>56.53</td><td rowspan=1 colspan=1>67.33</td><td rowspan=1 colspan=1>56.90</td><td rowspan=1 colspan=1>19.99</td></tr><tr><td rowspan=1 colspan=1>91.06</td><td rowspan=1 colspan=1>45.64</td><td rowspan=1 colspan=1>35.39</td><td rowspan=1 colspan=1>19.99</td></tr><tr><td rowspan=1 colspan=1>119.40</td><td rowspan=1 colspan=1>25.88</td><td rowspan=1 colspan=1>15.19</td><td rowspan=1 colspan=1>10.00</td></tr></table>

stage bypassed 15 checks (2.3%). RGB acquisition was selective, but appearance inference still ran on most observations. The adaptive feature scheduler targets this remaining local computation.

The low bypass rate is consistent with continuous head motion during the walk: ego-motion and background parallax shift image content, so pixel-wise luminance residuals can exceed fixed thresholds even when the semantic scene is unchanged. Mean and boundedgain exposure correction do not remove these geometric changes. Pixel-diference gating is therefore less selective here than in a fixed-camera view. Regional embedding comparison provides an appearance-level test for localized diferences, although viewpoint changes also afect embedding distances.

The first RGB observation was accepted at 6.95 s. Initial coverage uses the same persistence rule as later candidates, trading an early capture for stable evidence. The second and sixth acquisitions used the regional branch with global scores below the global threshold (Figure 4). Acquisition also required candidate consistency, temporal support, and queue admission. Scores fell sharply after accepted views entered the acquisition bank.

The continuous reference streams share a sensor clock (Table 4). VGA-to-RGB ofsets range from 11.13 to 67.33 ms, with a median of 27.63 ms. Request-to-next-RGB ofsets range from 0.48 to 56.90 ms, with a median of 16.80 ms.

At an illustrative head-rotation speed of 60°/s, a 10 to 20 ms exposure integrates 0.6 to 1.2° of angular motion. The median and maximum inter-view ofsets correspond to 1.66° and 4.04° of rotation at that speed. A stable monochrome candidate can thus produce a shifted or blurred RGB image, which the post-acquisition quality stage assesses.

Interpretation outcomes: Memory inspection found committed observations marked uncertain and a separate scene-schema failure. The uncertain sources were retained but were ineligible for the local clear-view shortcut. The schema failure retained its image and processing outcome without creating a semantic scene. Saved evidence and indexed memory therefore had distinct processing states.

The prototype uses Xiaomi MiMo-V2.6-Flash for scene descriptions, source selection, and image-backed answers. The glasses call the cloud service over HTTPS; image-quality checks and feature embedding run locally, and source images and memory records remain in app-private storage. No VLM weights are deployed on the glasses. Provider-side precision and quantization are not configured by the client. Eligible historical lookups use the local index without a model call.

The acquisition bank commits when capture is accepted; semantic records depend on later interpretation. The outcome-feedback path uses these processing outcomes to choose between reinterpreting a saved source and reassessing a recurring view.

![](images/e14c9cc6555646c9501661f4bc5b397b3da69d65aaa25d51c42edad71d32929c.jpg)  
Figure 4. Temporal trace of the 131.35 s causal replay. (a) Logged global and regional novelty with the configured thresholds. Gaps mark observations without feature extraction. The shaded initial-coverage interval uses the empty-bank initialization $G _ { t } = R _ { t } = 1$ . (b) Feature checks and luminance bypasses. (c) Six paired RGB acquisitions, colored by the admitting branch.

Implications for optical see-through assistance and useful memory: Recall appears within $\iota 3 0 ^ { \circ }$ near-eye field while the physical scene stays visible. For an additive optical see-through path, the first-order luminance contrast of a bright symbol is

$$
C _ { W } \simeq \frac { L _ { d } } { \tau L _ { b } + L _ { v } } ,\tag{6}
$$

where $L _ { d }$ is display luminance added at the eye, $L _ { b }$ is scene luminance, τ is optical transmittance, and $L _ { v }$ is veiling light. A brighter scene raises the background against which symbols must be read. Sparse bright text and low visual density follow from this contrast constraint and the findings on OST legibility and color schemes [14, 15]. RayNeo advertises 3,500 nit average and 6,000 nit peak brightness [4]; transmittance and background structure also afect scene-dependent contrast.

Display FOV bounds the visible content, and eye position within the usable eyebox afects visibility. Concise cards place recalled information near the center of the view. World-registered overlays also need eye-to-display calibration and current pose alignment [13]. A historical object-place relation and timestamp can communicate an earlier encounter directly. The prototype uses concise binocular text; auditory answers could provide a complementary format during visually demanding activity. Real/virtual context changes and focal-distance switching also afect presentation [16].

A remembered object-place relation can guide the wearer toward a previously observed location, with its timestamp dating the encounter. Source-linked retrieval and question-specific support preserve this historical meaning in the displayed answer.

Information can be lost when acquisition rejects a brief encounter, interpretation omits a detail, retrieval excludes a source, or retention removes it. Each stage needs a corresponding remedy. Regional analysis addresses localized visual changes, original-image access recovers descriptive omissions, retrieval expansion addresses search misses, and outcome feedback responds to processing or evidence deficits.

Useful memory is assessed through supported recall across the intended questions. Coverage, correctness among answered queries, unsupported present-state assertions, acquisition demand, and retained bytes relate that utility to its acquisition and storage costs.

## 6. Conclusion

We developed an AR-glasses memory system that combines hierarchical visual acquisition, source-linked storage, and staged recall. Capture acceptance, interpretation outcome, and support for a later question have distinct roles in this system. These sourcelinked paths preserve visual encounters for historical access. Replay shows selective RGB acquisition with frequent local inference; exposure and cross-view timing describe the imaging conditions of the selected evidence. The outcome-aware controller and sourcepreserving retention policy direct further acquisition and processing toward unresolved evidence needs.

Code availability. The source code for the ARglasses application is publicly available on GitHub: https://github.com/TZHU64/spatial-memory-ar.

## 7. References

1. Packer C, Wooders S, Lin K, Fang V, Patil SG, Stoica I, et al. MemGPT: towards LLMs as operating systems. arXiv [Preprint]. 2023 [revised 2024 Feb 12; cited 2026 Oct 23]. Available from: https://arxiv.org/abs/2310.08560.

2. Letta. Memory blocks (core memory) [Internet]. Letta; [date unknown] [cited 2026 Oct 23]. Available from: https://docs.letta.com/v1-sdk/memory/memory-blocks.

3. Young J. Efective harnesses for long-running agents [Internet]. Anthropic; 2025 Nov 26 [cited 2026 Oct 23]. Available from: https://www.anthropic.com/research/ effective-harnesses-for-long-running-agents.

4. RayNeo. RayNeo X3 Pro: Technical specifications [Internet]. RayNeo; [date unknown] [cited 2026 Oct 23]. Available from: https://www.rayneo.com/products/x3-pro-ai-display-glasses.

5. Grauman K, Westbury A, Byrne E, Chavis Z, Furnari A, Girdhar R, et al. Ego4D: around the world in 3,000 hours of egocentric video. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2022); 2022 Jun 19-24; New Orleans, LA, USA. Los Alamitos (CA): IEEE Computer Society; 2022. p. 18995-9012.

6. Bärmann L, Waibel A. Where did I leave my keys? Episodic-memory-based question answering on egocentric videos. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW 2022); 2022 Jun 19-24; New Orleans, LA, USA. Los Alamitos (CA): IEEE Computer Society; 2022. p. 1560-8.

7. Makhataeva Z, Akhmetov T, Varol HA. Augmented-reality-based human memory enhancement using artificial intelligence. IEEE Trans Hum Mach Syst. 2023;53(6):1048-60. doi:10.1109/THMS.2023.3307397.

8. Manigrasso Z, Dunnhofer M, Furnari A, Nottebaum M, Finocchiaro A, Marana D, et al. Online episodic memory visual query localization with egocentric streaming object memory. In: Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV 2026); 2026 Mar 6-10; Tucson, AZ, USA. Los Alamitos (CA): IEEE Computer Society; 2026. p. 3951-60.

9. Yang J, Liu S, Guo H, Dong Y, Zhang X, Zhang S, et al. EgoLife: towards egocentric life assistant. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2025); 2025 Jun 11-15; Nashville, TN, USA. Los Alamitos (CA): IEEE Computer Society; 2025. p. 28885-900.

10. Fan Y, Ma X, Su R, Guo J, Wu R, Chen X, et al. Embodied VideoAgent: persistent memory from egocentric videos and embodied sensors enables dynamic scene understanding. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV 2025); 2025 Oct 19-23; Honolulu, HI, USA. Los Alamitos (CA): IEEE Computer Society; 2025. p. 6342-52.

11. Ma K, Mao Y, Li W, Tan S, Zhong Y, Chen H, et al. R4DSG: relative 4D scene graph memory for object-centric question answering in long egocentric video. arXiv [Preprint]. 2026 [cited 2026 Oct 23]. Available from: https://arxiv.org/abs/2608.11017.

12. Kim Y, Yang Y, Kaufman AE. Memento: towards proactive visualization of everyday memories with personal wearable AR assistant. In: 2026 IEEE Conference on Virtual Reality and 3D User Interfaces Abstracts and Workshops (VRW); 2026 Mar 21-25; Daegu, Republic of Korea. Piscataway (NJ): IEEE; 2026. p. 993-1000. doi:10.1109/VRW70859.2026.00180.

13. Grubert J, Itoh Y, Moser K, Swan JE II. A survey of calibration methods for optical see-through head-mounted displays. IEEE Trans Vis Comput Graph. 2018;24(9):2649-62. doi:10.1109/TVCG.2017.2754257.

14. Gabbard JL, Swan JE II, Hix D. The efects of text drawing styles, background textures, and natural lighting on text legibility in outdoor augmented reality. Presence. 2006;15(1):16-32.

15. Erickson A, Kim K, Lambert A, Bruder G, Browne MP, Welch GF. An extended analysis on the benefits of dark mode user interfaces in optical see-through head-mounted displays. ACM Trans Appl Percept. 2021;18(3):12:1-22. doi:10.1145/3456874.

16. Gabbard JL, Mehra DG, Swan JE II. Efects of AR display context switching and focal distance switching on human performance. IEEE Trans Vis Comput Graph. 2019;25(6):2228-41. doi:10.1109/TVCG.2018.2832633.

17. LiKamWa R, Hou Y, Gao J, Polansky M, Zhong L. RedEye: analog ConvNet image sensor architecture for continuous mobile vision. In: Proceedings of the 43rd International Symposium on Computer Architecture (ISCA 2016); 2016 Jun 18-22; Seoul, Republic of Korea. New York (NY): Association for Computing Machinery; 2016. p. 255-66.

18. Buckler M, Jayasuriya S, Sampson A. Reconfiguring the imaging pipeline for computer vision. In: Proceedings of the IEEE International Conference on Computer Vision (ICCV 2017); 2017 Oct 22-29; Venice, Italy. Los Alamitos (CA): IEEE Computer Society; 2017. p. 975-84.

19. Chen TYH, Ravindranath L, Deng S, Bahl P, Balakrishnan H. Glimpse: continuous, real-time object recognition on mobile devices. In: Proceedings of the 13th ACM Conference on Embedded Networked Sensor Systems (SenSys 2015); 2015 Nov 1-4; Seoul, Republic of Korea. New York (NY): Association for Computing Machinery; 2015. p. 155-68.

20. Wu Z, Xiong C, Ma CY, Socher R, Davis LS. AdaFrame: adaptive frame selection for fast video recognition. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2019); 2019 Jun 15-20; Long Beach, CA, USA. Los Alamitos (CA): IEEE Computer Society; 2019. p. 1278-87.

21. Yan C, Bryson M, Dansereau DG. JOCA: task-driven joint optimisation of camera hardware and adaptive camera control algorithms. In: Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV 2026); 2026 Mar 6-10; Tucson, AZ, USA. Los Alamitos (CA): IEEE Computer Society; 2026. p. 3928-38.

22. Google. CameraCharacteristics [Internet]. Google; [date unknown] [cited 2026 Oct 23]. Available from: https://developer.android.com/reference/android/hardware/ camera2/CameraCharacteristics.

23. Howard A, Sandler M, Chu G, Chen LC, Chen B, Tan M, et al. Searching for MobileNetV3. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV 2019); 2019 Oct 27-Nov 2; Seoul, Republic of Korea. Los Alamitos (CA): IEEE Computer Society; 2019. p. 1314-24.

24. Google. Image embedding task guide [Internet]. Google; [date unknown] [cited 2026 Oct 23]. Available from: https://developers.google.com/edge/mediapipe/solutions/ vision/image\_embedder.