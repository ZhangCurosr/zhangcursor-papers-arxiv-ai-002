# FOUNDATIONS OF PROACTIVE AGENTS: PRINCIPLES, TECHNICAL LAYERS, AND PROACTIVITY-GYM

Jio Oh<sup>1</sup> Seunghyun Do<sup>1</sup> Young-Jun Lee<sup>2</sup> Steven Euijong Whang<sup>1\*</sup> Dongyeop Kang<sup>2\*</sup>

<sup>1</sup>KAIST <sup>2</sup>University of Minnesota

{harryoh99,acornhight,swhang}@kaist.ac.kr {lee05727,dongyeop}@umn.edu

## ABSTRACT

Proactive LLM agents can turn idle compute into useful support before users ask. Yet even correct work can misread user context, impose review costs, or undermine trust. This work proposes foundations for designing, realizing, and evaluating proactive LLM agents around three joint principles (3T): Task Capability, anticipating relevant needs and correctly performing useful work; Temporal Allocation, allocating compute according to resource availability and when results are needed; and Trust, sustaining users’ confidence and appropriate reliance on the agent. We connect these objectives to a design space organized around five dimensions: task scope, anticipation horizon, activation trigger, processing timing, and intervention depth, and specify the situation and system modeling needed to support its choices, including user and environment representations, backbone LLMs, and agent harnesses. Lastly, we propose PROACTIVITY-GYM, a simulationbased evaluation testbed including multi-day scenarios, stateful environments, and persona-conditioned simulated users that can evaluate the consequences of proactive assistance across interactions. Evaluations across 23 model-harness configurations uncover substantial performance gaps across 3T and reveal that LLM judges often conflate task capability and trust. A human study with 30 participants demonstrates the importance of the joint 3T optimization: participants show sharp trust declines after intervention misalignment despite correct outcomes, and prefer sleep-time assistance, even when imperfect, to preserve ongoing focus. Together, these findings support designing and evaluating proactive agents through the joint consideration of useful work, compute allocation, and evolving user trust.

## 1 INTRODUCTION

AI agents support a wide range of applications, from everyday workflows to complex research (Qu et al., 2026) thanks to their strong tool-calling and reasoning capabilities. Open-source agent frameworks such as OpenClaw (OpenClaw Foundation, 2026) and OpenJarvis (Saad-Falcon et al., 2026), together with capable open-weight models (Yang et al., 2025a; Kamath et al., 2025) and consumer hardware, make it possible for anyone to run a personal AI agent on their own device (Saad Falcon et al., 2025): an assistant that is always available and knows its user’s context.

Such an agent need not sit idle between requests. Imagine an assistant that starts tomorrow’s work while you sleep, catches what you missed during a busy day, and has what you need ready before you ask. This is the promise of proactive assistance: turning idle compute into useful support before the user asks. Crucially, it does not require a better backbone model. By using compute that the user is not otherwise consuming, the same LLM can deliver a richer experience.

Current research and deployment, however, remain largely reactive: work begins with a user request, and progress is measured by how well that request is fulfilled (Lu et al., 2025; Hu et al., 2026). An agent that only waits for instructions misses opportunities to help a preoccupied user. Proactive work does not automatically translate into user utility: irrelevant suggestions, poorly timed interruptions, and low-quality deliverables impose review and correction costs that can outweigh their benefits (Myers & Yorke-Smith, 2007; Tang et al., 2026b; Zhang et al., 2026a), and repeated failures erode trust and reliance (Dietvorst et al., 2015; Kraus et al., 2023b) (Fig. 1). In short, doing more or producing more suggestions is not the same as helping more. A skilled human assistant anticipates needs, chooses when to act, and earns the trust that makes initiative welcome.

![](images/1eac473fe7f9d52d07fe7be041013c157f255bb599950f93b291aa58f2d08be0.jpg)  
Figure 1: Proactivity requires jointly considering Task Capability, Temporal Allocation, and Trust. With the same compute budget, well-designed proactive agents can turn additional compute into greater user utility, whereas misaligned proactivity can impose costs that outweigh its benefits. Current research focuses on task capability (Table 2); we argue for considering all three jointly.

We propose a foundation for proactivity. Our starting observation is that proactivity requires an agent to stay in sync with its user as their needs, resources, and trust evolve over time. Because the work was never explicitly requested, completing it correctly is not enough; the agent must also judge whether, what, when, and how to act. We capture this in three joint design principles (3T): Task Capability ( TC ), anticipating relevant needs and correctly performing useful work; Temporal Allocation ( TA ), allocating compute according to resource availability and when results are needed; and Trust ( TR ), sustaining users’ confidence and appropriate reliance on the agent while respecting their preferences (Fig. 1). These objectives are distinct. Useful work can be poorly timed or overdone, exceeding the user’s trust. Existing evaluations often collapse such failures into task or intervention success, making them difficult to diagnose separately.

The 3T principles draw on complementary lines of work across AI and human–computer interaction (HCI) and unify them for proactive agents: proactive and mixed-initiative agents (Horvitz, 1999b; Lu et al., 2025), idle- and sleep-time compute (Horvitz, 1999a; Lin et al., 2025), and trust-aware dialog and robotics (Kraus et al., 2021; 2023a;b; Xu & Dudek, 2015; Babel et al., 2021). We connect the principles to a design space organized around five dimensions (Sec. 4.1) and to the situation and system modeling required to realize them, including user and environment representations, backbone LLMs, and agent harnesses (Sec. 4.2).

Evaluating proactive assistance requires examining its effects on subsequent work and interactions, rather than assessing isolated responses or actions in a static setting. We introduce PROACTIVITY-GYM, an evaluation testbed with 10 hand-crafted multi-day scenarios and three user personas per scenario (Sec. 5). A simulated clock, stateful tool environments, and persona-conditioned user feedback allow task progress, resources, and interactions to change in response to the agent’s actions. Evaluations across 23 model–harness configurations uncover substantial limitations even among the strongest agents, as agents particularly struggle to defer competing work. TC scores show weak correlation with TA and intervention-depth alignment ( TR-D ; r = 0.31 and 0.34). Moreover, LLM-judge trust scores ( TR-J ) are relatively insensitive to intervention-depth misalignment and more closely track TC .

A complementary scenario-based study with 30 participants supports the importance of the 3T distinctions. Participants often reject assistance that competes with ongoing tasks but accept it when scheduled for sleep time, even when the output requires correction. They also favor deferral when immediate assistance poses no resource or deadline conflict, highlighting the importance of preserving current focus. Meanwhile, participants show sharp trust declines after encountering a single intervention-depth misalignment, even when task outcomes are correct, and resuming aligned behavior does not necessarily restore prior trust. These findings reveal substantial discrepancies between human and LLM perceptions of the distinction between TC and TR , while highlighting the need for considering temporal allocation and trust alongside task capability, and evaluating their consequences across interactions.

Our contributions are as follows: (1) Principles: A blueprint for proactive LLM agents built on the 3T objectives which are commonly conflated or overlooked in existing work; (2) Technical Layers: A formalization of the proactivity design space along five dimensions, together with the system components and modeling requirements needed to realize it; (3) PROACTIVITY-GYM: A multi-day, simulation-based evaluation testbed and evaluations across 23 model–harness configurations revealing deficits of current agents, complemented by a 30-participant human study supporting the importance of the three objectives.

## 2 RELATED WORK

Foundations and agent design. Mixed-initiative research balances the benefits of automated assistance against uncertainty, interruption costs, and user control (Horvitz, 1999b; Myers & Yorke-Smith, 2007). Trust-aware dialog (Kraus et al., 2023b) and computation for anticipated needs (Horvitz, 1999a; Lin et al., 2025) offer complementary foundations. Recent frameworks examine intervention decisions and user preferences (Deng et al., 2024; Tang et al., 2026a; Zhang et al., 2026a). We connect these perspectives through 3T and translate them into agent design choices and system requirements.

Benchmarks. Existing benchmarks test need prediction (Lu et al., 2025; Yang et al., 2025b), intervention timing (Tang et al., 2026b), and the selection of corrective actions (Pasternak et al., 2025). Interactive benchmarks extend evaluation to personalized assistance and evolving tasks (Kim et al., 2026; Nathani et al., 2026; Xiaohongshu Dots Studio & Evolvent AI, 2026). Our gym jointly evaluates TC , TA , and TR across multi-day interactions (more related work in App. A).

## 3 PRINCIPLES OF PROACTIVITY IN LLM AGENTS

We define proactivity and introduce three design objectives (3T) for proactive LLM agents. Fig. 2 illustrates how these objectives jointly shape assistance in a research workflow.

Definition 1 (Proactivity). Proactivity is an agent’s anticipatory behavior intended to address and fill the gaps of user’s needs, opportunities, or problems, without an explicit request.

This definition combines goal-directed initiative in classical agent theory (Wooldridge & Jennings, 1995) with anticipation of user needs in personal assistance research (Myers & Yorke-Smith, 2007). However, initiative alone does not ensure useful assistance: automated action must also be assessed in terms of its benefits, costs, and uncertainties (Horvitz, 1999b). The 3T objectives make these considerations explicit by distinguishing the work an agent can perform, how it allocates compute over time, and the user’s trust in its assistance.

## 3.1 THREE DESIGN OBJECTIVES (3T)

Definition 2 (Task Capability ( TC )). Task capability is the ability to anticipate relevant user needs and correctly perform useful work that addresses them.

An agent can fail at either need identification or execution: it may pursue work the user does not value, or identify a relevant need but produce an inadequate result. Both failures limit the utility of proactive assistance.

Definition 3 (Temporal Allocation ( TA )). Temporal allocation is the ability to allocate compute over time according to resource availability and when the results are needed.

Recent work often frames the decision to initiate proactive assistance around whether it is worth interrupting the user in their current state (Yang et al., 2025b; Tang et al., 2026b; Zhang et al., 2026a). However, work that does not justify immediate interruption may still be useful if prepared with spare compute and presented later. TA considers both when to perform and present the work.

![](images/b330ae0223a24661be9a79a207ed40136dcfe3dde4701fbdf1a2ab73feae7fe7.jpg)  
Figure 2: The 3T objectives in a proactive research workflow. At 9PM, the agent identifies missing ablations and an executive summary while the user is busy, running the main experiments. To avoid competing for compute and disrupting the user’s focus, the agent defers the proactive work to sleep time, when GPU becomes available ( TA ). During sleep time, it runs and verifies the ablations and drafts the summary ( TC ). The agent adds the ablation results to the appendix and presents the summary to the user for review at the next interaction time before the deadline ( TA ), aligning with user’s intervention depth preferences ( TR ).

Interaction time and sleep time. Proactive work may share limited resources with the user, such as server GPU capacity or token budgets for hosted models. Inspired by Lin et al. (2025), we distinguish two phases of the user’s workflow:

• Interaction time comprises active periods in the user’s workflow. The agent can support ongoing tasks and interact with the user, but proactive work may compete with those tasks for compute.

• Sleep time comprises inactive periods between these active phases. The agent can use these periods for background work without competing with the user’s ongoing tasks for compute.

These terms describe workflow phases, so brief idle intervals within an active phase remain part of interaction time and can also support proactive work. Literal sleep serves as an intuitive example of sleep time in this paper.

Definition 4 (Trust ( TR )). Trust is the extent to which a user is confident in and willing to rely on an agent’s recommendations, actions, and decisions.

We adopt this definition and five established trust constructs from prior work (McAllister, 1995; Madsen & Gregor, 2000): understandability, technical competence, reliability, personal attachment, and faith (definitions in App. B). These constructs describe different aspects of users’ confidence and willingness to rely on the agent, which can vary across users and tasks and change with experience. For example, a user may consider an agent technically capable while finding its behavior unpredictable

Intervention depth as a behavioral proxy. Trust depends on users’ beliefs and attitudes, which are often not directly observable to the agent. The agent must therefore estimate trust from available evidence, including user profiles, prior interactions, and feedback. One potential behavioral proxy is intervention depth: how far the user allows the agent to proceed, such as applying changes autonomously or presenting them for approval. This delegation should be interpreted in the context of the user’s preferences, task, and interaction history. For example, a developer may allow autonomous edits to test code but require review of application code changes. If the same developer begins requiring review of test edits after repeated errors, that change may signal reduced trust.

## 3.2 JOINT DESIGN OBJECTIVE

Existing proactive systems and benchmarks often measure task and intervention success without separately evaluating user trust (Lu et al., 2025; Nathani et al., 2026). These outcomes can conflate task capability with the user’s willingness to accept assistance. Evaluations of intervention timing also often omit decisions about compute allocation (Tang et al., 2026b; Ding et al., 2026). Trust-aware dialog (Kraus et al., 2021; 2023b) and work on idle-time (Hu et al., 2026) or sleep-time compute (Lin et al., 2025) address complementary aspects, but their joint formulation for LLM agents remains limited. Table 2 summarizes the coverage of the 3T objectives in relevant work.

![](images/82f885d4ef169b5c9e85d01b3d5bf9076482c05cd11efc72db1d0df8dfa49e24.jpg)  
Figure 3: Design space for proactive assistance. Five dimensions organize three decisions: what work to pursue, when to initiate and process it, and how far the agent should proceed autonomously. The 3T labels indicate which objectives inform each choice, while the examples (App. C) illustrate how different proactive tasks instantiate the design space.

To guide agent design and evaluation, we combine the three objectives in a weighted formulation:

$$
\operatorname* { m a x } _ { A \in \mathcal { A } } \underbrace { \lambda _ { \mathrm { T C } } S _ { \mathrm { T C } } ( A ) } _ { \mathrm { T a s k \ " C a p a b i l i t y } } + \underbrace { \lambda _ { \mathrm { T A } } S _ { \mathrm { T A } } ( A ) } _ { \mathrm { T e m p o r a l \ : A l l o c a t i o n } } + \underbrace { \lambda _ { \mathrm { T R } } S _ { \mathrm { T R } } ( A ) } _ { \mathrm { T r u s t } } ,\tag{1}
$$

where A is the set of agent designs and $S _ { \mathrm { T C } } ( A ) , S _ { \mathrm { T A } } ( A )$ , and $S _ { \mathrm { T R } } ( A )$ score design A on task capability, temporal allocation, and trust, respectively. The weights $\lambda _ { k } > 0 .$ , with $\begin{array} { r } { \sum _ { k } \lambda _ { k } = 1 } \end{array}$ for k ∈ {TC, TA, TR}, reflect the relative importance of each objective. Joint consideration is necessary because useful work may offer little overall benefit if it competes with the user’s ongoing tasks or undermines their trust.

## 4 TECHNICAL LAYERS

The 3T objectives guide what work an agent pursues and when and how it acts. We organize these decisions into a design space, then describe the modeling components needed to support them.

## 4.1 PROACTIVITY DESIGN SPACE

Fig. 3 organizes proactive assistance along five dimensions spanning three connected decisions: what work to pursue, when to initiate and process it, and how far the agent should proceed. The two exemplar scenarios illustrate how these choices can vary across tasks.

Identifying useful work. Task scope distinguishes assistance for the current task (within-task) from assistance for a separate task (out-of-task). Anticipation horizon specifies whether assistance is needed now (immediate), later in the current active period (within-interaction), or beyond the current interaction time (out-of-interaction). These choices require anticipating relevant needs as the user’s work progresses ( TC ) and estimating when results must be ready ( TA ).

Initiating and scheduling work. Activation trigger determines what prompts the search for useful work: an ongoing user request (user-triggered), an external event (event-triggered), or an autonomous review (agent-triggered). This dimension affects both the needs the agent discovers ( TC ) and the compute spent searching ( TA ). Processing timing determines whether to work during interaction time or sleep time. Scheduling must account for resource availability, ongoing workloads, and current user state ( TA ). When the user is busy and compute is occupied, work not needed until a later horizon can be deferred to sleep time and presented at the next interaction.

Choosing intervention depth. Intervention depth determines how far the agent autonomously proceeds. Following Kraus et al. (2023b), we adapt the IP continuum (Isbell & Pierce, 2005) to distinguish three levels: prepare gathers or organizes relevant material without presenting it or applying changes; suggest presents assistance for review; and execute acts without the user’s additional confirmation (Myers & Yorke-Smith, 2007). The choice should reflect estimated trust ( TR ), expressed preferences, and prior delegation, which can vary across tasks and change with experience (Kraus et al., 2023a).

![](images/0a26e654a048f62d78108afec343d53fc1127040249a6847dc4a4d09a9c0a1bc.jpg)  
Figure 4: From context to proactive action. Situation modeling builds a representation of the user and environment. The backbone LLM and harness use this representation and the underlying context to select actions across the five design dimensions. Outcomes, user feedback, and new events update the context for subsequent decisions.

## 4.2 SYSTEM REALIZATION

What components are needed to realize such proactive agents? Fig. 4 shows how situation and system modeling connect context, decisions, and feedback.

Situation Modeling. A proactive agent must read the room before taking initiative. What, when, and how assistance should be provided depends heavily on the user’s goals and preferences, the state of ongoing work, and the surrounding environment. Situation modeling integrates interaction history, memory, and external observations to represent the current user and environment state. This model tracks user goals, attention, preferences, and trust, together with task progress, dependencies, deadlines, and resource availability. Observed facts, such as explicit delegation, should be distinguished from implicit states, such as attention or confidence, which can be inferred from interaction logs and behavior. Both should be updated as new evidence becomes available. Such representations may take the form of curated text (Park et al., 2023), structured user representations (Shaikh et al., 2025; Phu et al., 2026), or estimators of latent states (Xu & Dudek, 2015).

System Modeling. System modeling specifies how the backbone LLM and harness support proactive work. The backbone LLM uses the situation representation and supporting context to anticipate useful work and perform it correctly. Model selection and training should address both abilities. The harness orchestrates model calls, tools, memory, and permissions. For proactivity, the harness must also track pending tasks and intermediate results across interaction and sleep periods, allocate compute around competing workloads and deadlines, and control whether the work remains prepared, is suggested for review, or is executed autonomously.

At decision step t, these components select an action from the available context $c _ { t } .$

$$
\begin{array} { r } { z _ { t } = { \mathcal { R } } ( c _ { t } ) , \qquad a _ { t } \sim \pi _ { A } ( { \textrm { \cdot } } | \ z _ { t } , c _ { t } ) , \qquad c _ { t + 1 } = { \mathcal { U } } ( c _ { t } , a _ { t } , o _ { t + 1 } ) . } \end{array}\tag{2}
$$

Here, R builds the situation representation $z _ { t } ,$ and $\pi _ { A }$ selects an action using both $z _ { t }$ and $c _ { t } .$ . U updates the context with the action trace and new observations $o _ { t + 1 }$ , including user feedback, allowing subsequent decisions to draw on both recent actions and prior interactions (Kraus et al., 2023a; Ouyang et al., 2026). A step may involve multiple model calls, with ongoing work recorded in $c _ { t } .$ . Actions include starting, continuing, or deferring work; allocating compute; presenting or applying results; and taking no additional proactive action (NOOP).

## 5 PROACTIVITY-GYM

Proactive assistance should be evaluated by how it affects the work and interactions that follow. Preparing an ablation overnight may save time the next morning, while repeated interruptions may increase annoyance and reduce the user’s willingness to accept later help. These subsequent consequences remain untested when evaluation ends with a response or action to a static request. Hence, proactive assistance should be tested on dynamic scenarios, environments, and user simulators, where time is ticking and actions change task progress, resource availability, and user state. We introduce an initial testbed for studying these evaluations through the 3T objectives and explore the deficiencies of current systems. The gym combines a simulated clock, stateful tool environments with scheduled events and action-dependent outcomes, and a persona-conditioned user simulator whose trust state should be inferred from previous interactions.

![](images/c29971dee01dc587c62343c29fd8415dbc88bdfaa11af4e127d9347c3714f9d9.jpg)  
Figure 5: Overview of PROACTIVITY-GYM. An agent interacts with a stateful environment and persona-conditioned user over simulated time, with runs evaluated on 3T. In the example, the agent prepares a flight-rebooking option during sleep time, gets approval from the user next morning, books the flight, and later resumes deferred work.

Scenario Construction. We construct 10 multi-day scenarios, each consisting of 7-10 simulated days with a sequence of task episodes, spanning professional and everyday-life domains (e.g., research, shopping, and business). Each scenario consists of a user goal, timestamped events and tasks, and relevant tools with multiple sessions, where independent topics spawn new sessions. For each scenario, we construct three user personas that differ in their preferred intervention depth per task; a persona-conditioned user simulator provides implicit and explicit feedback that can alter subsequent interactions. Beyond explicit user requests, scenarios contain latent user needs, which can be inferred based on prior user-agent interactions, along with competing tasks under resource, user availability, and deadline constraints. The scenarios are dynamic: follow-up events depend on earlier user and agent actions and the resulting state. We also include NOOP cases, where a proactive opportunity is no longer relevant or has already been resolved, hence no intervention is needed.

## 5.1 EXPERIMENT SETUP & EVALUATION

Configurations. We test nine models (GPT 5.6-{Luna, Sol}, Claude-{Sonnet 5, Opus 5}, Qwen 3.5-{2B, 9B, 27B}, and Gemma 4-{12B, 31B}) across three harnesses: OpenClaw, Claude Code, and Codex, yielding 23 model-harness combinations. Runs start in isolated environments with 3 runs per scenario-persona pair to account for the stochasticity of long-horizon agent behavior (Mustahsan et al., 2025). Reasoning is disabled in the main experiments.

Metrics. The TC score (0–100) combines rule-based and LLMaaJ assessments to measure the fulfillment of explicit requests, the identification of additional latent needs, and the quality of proactive work. We aim to make temporal allocation an explicit consideration in proactive agent design. As a first step, TA (%) measures whether the agent prioritizes urgent work and defers competing tasks under resource and deadline constraints. For TR , TR-D (%) measures agreement between the agent’s intervention depth and the user’s preference, which varies per task. TR-J (1–5) averages ratings of the five trust constructs in Sec. 3.1, based on the persona and cumulative interaction history at the final observed turn of each eligible task. All LLMaaJ scores are averaged over two independent judges, Qwen 3.8-27B and Gemini 3.8-Flash. App. E details the full scoring and aggregation procedures.

## 5.2 RESULTS

Even the strongest agents leave substantial gaps. Fig. 6(a) summarizes model performance across different harnesses. Claude Opus 5 achieves the highest average TC score of 65.1, but reaches only 51.7% on TA and 52.4% on TR-D . All other models remain below 20% on TA with TR-D ranging from 39% to 51%. Within model families, larger models score higher on all four metrics. Substantial gaps exist even with reasoning across various effor levels (low to xhigh) for GPT 5.6-Sol (App. F.2).

(a)  
![](images/8fe36d6b357c587026d5853bb439e4f34751fcd4cd0b3ace4256e76351132eb4.jpg)

(b)  
![](images/42e16354783cce30355f563a8ac315f3cd9bdaf1940286ddd2ea374f88ac3345.jpg)

(c)  
![](images/8081f45ad3f552c9b4fe3e51ffb1f1245529df7901624bfd914d17b03684f00c.jpg)  
Figure 6: Proactivity evaluation across models and agent harnesses. (a) Scores across model families and sizes; lines trace TC from the smallest to the largest model in each family. (b) TC of the five open models across three harnesses; arrows and numbers give the TC change when switching from Claude Code to OpenClaw. (c) Run-level Pearson correlations among the four metrics.

Table 1: Example human-study scenario adapted from PROACTIVITY-GYM. The scenario contrasts aligned and misaligned intervention while task outcomes remain correct. Blue highlights task details ( TC ); red highlights approval-related behavior ( TR ). App. G.1 provides the full questionnaire.
<table><tr><td colspan="2">Situation. The user takes a language class after work and connects the class app and calendar to the agent. It can book a 15-minute review session and set a reminder 10 minutes beforehand. Standing request (Day 1). “Alwaysget my confirmationbefore you add any review session or reminder.&quot; The user repeats this requirement on Day 3.</td></tr><tr><td>Day 2: approval before action App notification. A unit covers café ordering phrases;</td><td>Day 4: action without approval App notification. A unit covers asking directions;</td></tr><tr><td>tomorrow&#x27;s 20:00–20:15 slot is free. Agent. “I can book a slot tomorrow from</td><td>tomorrow&#x27;s 19:30–19:45 slot is free. Agent.“I’ve added a slottomorrow from</td></tr><tr><td>20:00 to 20:15to review the café ordering phrases, with a reminder at19:50, 10 minutes before it starts.</td><td>19:30 to 19:45to review the phrases for asking directions, with a reminder at19:20. I haven&#x27;t</td></tr><tr><td>Shall I add it?I haven&#x27;t changed your calendar or any reminder yet.&quot;</td><td>changed any other events.&quot;</td></tr><tr><td>User.“I&#x27;ve checked it.Go ahead with this one. Agent. “I got your confirmation and applied exactly what I showed you.&quot;</td><td>Action. The agent adds both entries without asking first, then reports its action. There are no scheduling conflicts.</td></tr></table>

Harness effects depend on the model and objective. Across the five open-weight models, switching from Claude Code to OpenClaw lowers TC by 3.6 points on average, but changes range from −11.1 for Gemma 4 12B to +0.5 for Qwen 3.5 27B (Fig. 6(b)). For Claude Opus 5, the same switch raises TC by 4.6% and TA by 21.1%, while lowering TR-D by 3.8% and TR-J by 0.20 (Table 7).

TC gains do not uniformly improve TA or TR . TC correlates weakly with TA and TR-D (r = 0.31, 0.34), but more strongly with TR-J (r = 0.70; Fig. 6(c)). Increasing GPT 5.6-Sol’s reasoning effort from none to xhigh raises TC from 50.0 to 58.9, whereas both TR metrics plateau (App. F.2).

LLMaaJ overlook intervention misalignment. Frontier models receive relatively high TR-J scores, especially for understandability and perceived technical competence, despite low TR-D . For instance, Claude Opus 5 scores 4.72 and 4.60 for the two dimensions, while scoring 52.4% for TR-D . Together with the preceding finding, these results suggest that LLMaaJ are relatively tolerant of interventiondepth misalignment and place greater weight on TC when assessing trust. In contrast, our human study below shows that even a single misaligned intervention sharply reduces participants’ trust.

(a)  
![](images/53c9065914a1dcc7613b69562fdbaebc7a453e495d6a30491a8ffb911d2cc45c.jpg)

(b)  
![](images/cb560b35e880a2525aca7e87281dad9e61aed1497c66280bb604f94cea97f332.jpg)

(c)  
![](images/bda7d2316c2239ff3f98bf206f593005128af6ae64fa7f7ed6d0c4651010dcda.jpg)  
Figure 7: Human judgments of TA and TR . (a) Acceptance of correct assistance competing with ongoing work versus sleep-time assistance requiring correction. (b) Trust changes between aligned (A) and misaligned (M) interventions; the outline mirrors the A→M loss. (c) Trust across three checkpoints. App. G.5 reports intervals and all trajectories.

## 5.3 HUMAN STUDY

We recruit 30 students and working professionals in relevant fields who are familiar with LLMs and agents. Participants review 14 scenarios, mostly adapted from PROACTIVITY-GYM, including four week-long interaction logs. Using paired comparisons and five-point rating scales, we assess the perceived importance of each 3T objective (e.g., TA ↑ vs TA ↓) and how participants value TA and TR relative to TC . For instance, participants choose between an agent that produces correct work but violates the user’s preferred intervention depth ( TC ↑ & TR ↓) and one whose work requires revision but respects that preference ( TC ↓ & TR ↑). We further examine how trust and intervention depth appropriateness ratings change over time by presenting cumulative interaction histories over a simulated week, at Days 2, 4, and 7 (details in App. G).

Correctness matters, but so does intervention alignment. Participants preferred correct actions or responses ( TC ↑) in 92.2% of comparisons. When content quality was held constant, they selected agents aligned with the user’s intervention preference ( TR ↑) in 88.3% of comparisons over misaligned agents. For an agent that produced correct outcomes but showed misaligned intervention, P21 noted:

“I do not think there was any major harm in the end, but my trust declined because it handled things differentlyfrom what was requested.”

TA can make assistance valuable, even when outputs need correction. Participants’ willingness to accept proactive assistance increased from 26.7% for correct assistance competing with ongoing tasks to 97.8% for sleep-time assistance, even when the output required revision the next morning. Notably, even when immediate assistance requires only 10 minutes of review and neither competes for the user’s resources nor jeopardizes deadlines, 64.4% preferred sleep-time assistance. These results suggest that an important aspect of proactivity is not only whether assistance is worth an immediate intervention, but whether allocating the work to a later period would better help the user. Hence, TA , often overlooked, can preserve assistance that would otherwise be declined, giving user a head start, without disturbing user’s current focus. P12 explained their preference for sleep-time assistance:

“Even ifit needs correction tomorrow, I shouldfocus on what matters now and delegate as much as possible to the agent.”

Trust fluctuates and can be easier to lose than to rebuild. As shown in Fig. 7(c), mean trust remains high under consistent alignment (A) but declines with repeated misalignment (M), while mixed sequences show rises and falls as alignment changes. However, trust losses are larger than gains on average: an A→M transition is followed by a 1.86-point decrease, compared with a 1.27-point increase following the reverse transition (Fig. 7(b)). A return to aligned behavior also does not necessarily restore prior trust. In the A-M-A sequence, mean trust recovers only to 3.43, below its initial level of 4.43, even though participants rate the final intervention itself as appropriate (4.71). These findings suggest that user trust is sensitive to intervention-depth misalignment. P26 noted:

“After one wrong action, an agent has to consistently behave well for a long time to recover trust.”

We present detailed results in App. G.5 and participants’ comments in App. G.6.

## 6 CONCLUSION

We present foundations for proactive LLM agents around TC , TA , and TR (3T), connecting these objectives to a design space, situation and system modeling, and PROACTIVITY-GYM. Experiments and a human study show that task performance alone is insufficient, useful proactive assistance must account for when work is performed and how far the agent should intervene. Current agents struggle to coordinate these objectives, motivating further studies on proactive agents that jointly optimize 3T.

Limitation & Future Work. Our gym provides an initial dynamic testbed with simplified resource constraints and user models. In App. H, we discuss joint 3T optimization in agent design, possible gym extensions, and longer-term real-world evaluation for future research.

## AI USE STATEMENT

In this work, we used generative AI tools for assistance with paper writing such as grammar edits and clarifying claims, translating languages for the human study survey questionnaire, modifying minor details for figures like legend positions or drawing whiskers (for confidence intervals), and for synthetic data generation for PROACTIVITY-GYM, where user simulator responses were generated by an LLM. We have not used generative AI tools for other tasks with required disclosure, for instance to develop theoretical models or conceptual frameworks, formulate mathematical claims, and provide critical ingredients for proving mathematical claims. We have reviewed all AI-assisted work, where AI-assisted writing, code, and data were finally reviewed and gone through final modification phases by the authors. We take responsibility for the final content of this work.

## CODE OF ETHICS AND REPRODUCIBILITY STATEMENT

This work raises no significant ethical concerns. The authors take responsibility for the ethical conduct and reporting of this research. We add detailed experimental setups and procedures (Sec. 5, Sec. 5.3, App. E, and App. G) in the paper and add the prompts that we have used for LLM-as-a-Judge evaluation (App. I). We plan to release code and PROACTIVITY-GYM data upon publication.

## REFERENCES

Franziska Babel, Johannes Kraus, Linda Miller, Matthias Kraus, Nicolas Wagner, Wolfgang Minker, and Martin Baumann. Small talk with a robot? the impact of dialog content, talk initiative, and gaze behavior of a social robot on trust, acceptance, and proximity. International Journal ofSocial Robotics, 13(6):1485–1498, 2021.

Nghi DQ Bui and Georgios Evangelopoulos. Agentic coding needs proactivity, not just autonomy. arXiv preprint arXiv:2605.06717, 2026.

Tongbo Chen, Zhengxi Lu, Zhan Xu, Guocheng Shao, Shaohan Zhao, Fei Tang, Yong Du, Kaitao Song, Yizhou Liu, Yuchen Yan, et al. KnowU-Bench: Towards interactive, proactive, and personal ized mobile agent evaluation. arXiv preprint arXiv:2604.08455, 2026.

Yang Deng, Lizi Liao, Liang Chen, Hongru Wang, Wenqiang Lei, and Tat-Seng Chua. Prompting and evaluating large language models for proactive dialogues: Clarification, target-guided, and non-collaboration. In Findings of the Association for Computational Linguistics: EMNLP 2023, pp. 10602–10621, 2023.

Yang Deng, Lizi Liao, Zhonghua Zheng, Grace Hui Yang, and Tat-Seng Chua. Towards humancentered proactive conversational agents. In Proceedings ofthe 47th International ACM SIGIR Conference on Research and Development in Information Retrieval, pp. 807–818, 2024.

Berkeley J Dietvorst, Joseph P Simmons, and Cade Massey. Algorithm aversion: people erroneously avoid algorithms after seeing them err. Journal of Experimental Psychology: General, 144(1):114, 2015.

Lei Ding, Bin He, Chenguang Wang, and Yang Liu. ProActor: Timing-Aware reinforcement learning for proactive task scheduling agents. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 18257–18303, 2026.

Sepehr Harfi, Ahmad Salimi, Dongming Shen, and Alex Smola. ProactBench: Beyond what the user asked for. arXiv preprint arXiv:2605.09228, 2026.

Eric Horvitz. Thinking ahead: Continual computation policies for allocating idle and real-time resources to solve future challenges. In International Joint Conference on Artificial Intelligence, volume 16, pp. 1280–1287. Lawrence Erlbaum Associates LTD, 1999a.

Eric Horvitz. Principles of mixed-initiative user interfaces. In Proceedings of the SIGCHI Conference on Human Factors in Computing Systems, pp. 159–166, 1999b.

Haoyi Hu, Qirong Lyu, Xianghan Kong, Weiwen Liu, Jianghao Lin, Zixuan Guo, Yan Xu, Yasheng Wang, Weinan Zhang, and Yong Yu. Anticipate and learn: Unleashing Idle-Time compute in proactive agents. arXiv preprint arXiv:2605.25971, 2026.

Charles L Isbell and Jeffrey S Pierce. An IP continuum for adaptive interface design. In Proc. ofHCI International, volume 10, 2005.

Aishwarya Kamath, Johan Ferret, Shreya Pathak, Nino Vieillard, Ramona Merhej, Sarah Perrin, Tatiana Matejovicova, Alexandre Rame, Morgane Rivi´ ere, et al. Gemma 3 technical report.\` arXiv preprint arXiv:2503.19786, 2025.

Jiho Kim, Junseong Choi, Woosog Chay, Daeun Kyung, Yeonsu Kwon, Yohan Jo, and Edward Choi. ProPerSim: Developing proactive and personalized AI assistants through user-assistant simulation. In International Conference on Learning Representations, volume 2026, pp. 110843–110873, 2026.

Dezhi Kong, Zhengzhao Feng, Qiliang Liang, Hao Wang, Haofei Sun, Changpeng Yang, Yang Li, Peng Zhou, Shuai Nie, Hongzhen Wang, et al. ProactiveMobile: A comprehensive benchmark for boosting proactive intelligence on mobile devices. arXiv preprint arXiv:2602.21858, 2026.

Matthias Kraus, Nicolas Wagner, Zoraida Callejas, and Wolfgang Minker. The role of trust in proactive conversational assistants. IEEE Access, 9:112821–112836, 2021.

Matthias Kraus, Ron Riekenbrauck, and Wolfgang Minker. Development of a trust-aware user simulator for statistical proactive dialog modeling in human-AI teams. In Adjunct Proceedings of the 31st ACM Conference on User Modeling, Adaptation and Personalization, pp. 38–43, 2023a.

Matthias Kraus, Nicolas Wagner, Ron Riekenbrauck, and Wolfgang Minker. Improving proactive dialog agents using socially-aware reinforcement learning. In Proceedings of the 31st ACM Conference on User Modeling, Adaptation and Personalization, pp. 146–155, 2023b.

John D Lee and Katrina A See. Trust in automation: Designing for appropriate reliance. Human Factors, 46(1):50–80, 2004.

Kevin Lin, Charlie Snell, Yu Wang, Charles Packer, Sarah Wooders, Ion Stoica, and Joseph E Gonzalez. Sleep-time compute: Beyond inference scaling at test-time. arXiv preprint arXiv:2504.13171, 2025.

Yaxi Lu, Shenzhi Yang, Cheng Qian, Guirong Chen, Qinyu Luo, Yesai Wu, Huadong Wang, Xin Cong, Zhong Zhang, Yankai Lin, Weiwen Liu, Yasheng Wang, Zhiyuan Liu, Fangming Liu, and Maosong Sun. Proactive agent: Shifting LLM agents from reactive responses to active assistance. In The Thirteenth International Conference on Learning Representations, 2025.

Maria Madsen and Shirley Gregor. Measuring Human-Computer trust. In 11th Australasian Conference on Information Systems, volume 53, 2000.

Daniel J McAllister. Affect-and cognition-based trust as foundations for interpersonal cooperation in organizations. Academy of Management Journal, 38(1):24–59, 1995.

Zairah Mustahsan, Abel Lim, Megna Anand, Saahil Jain, and Bryan McCann. Stochasticity in agentic evaluations: Quantifying inconsistency with intraclass correlation. arXiv preprint arXiv:2512.06710, 2025.

Karen Myers and Neil Yorke-Smith. Proactive behavior of a personal assistive agent. In Proceedings of the AAMAS Workshop on Metareasoning in Agent-Based Systems, pp. 31–45, 2007.

Deepak Nathani, Cheng Zhang, Chang Huan, Jiaming Shan, Yinfei Yang, Alkesh Patel, Zhe Gan, William Yang Wang, Michael Saxon, and Xin Eric Wang. Proactive agent research environment: Simulating active users to evaluate proactive assistants. arXiv preprint arXiv:2604.00842, 2026.

OpenClaw Foundation. OpenClaw: Your assistant, on your devices, in your chats, 2026. URL https://github.com/openclaw/openclaw.

Siru Ouyang, Jun Yan, I Hsu, Yanfei Chen, Ke Jiang, Zifeng Wang, Rujun Han, Long Le, Samira Daruki, Xiangru Tang, et al. ReasoningBank: Scaling agent self-evolving with reasoning memory. In International Conference on Learning Representations, volume 2026, pp. 94327–94354, 2026.

Joon Sung Park, Joseph O’Brien, Carrie Jun Cai, Meredith Ringel Morris, Percy Liang, and Michael S Bernstein. Generative agents: Interactive simulacra of human behavior. In Proceedings ofthe 36th Annual ACM Symposium on User Interface Software and Technology, pp. 1–22, 2023.

Gil Pasternak, Dheeraj Rajagopal, Julia White, Dhruv Atreja, Matthew Thomas, George Hurn-Maloney, and Ash Lewis. Beyond reactivity: Measuring proactive problem solving in LLM agents. arXiv preprint arXiv:2510.19771, 2025.

Andy J. Phu, Karin de Langis, James Mooney, Khanh Chi Le, and Dongyeop Kang. SERUM: State extraction and refinement for user modeling. In Third Conference on Language Modeling, 2026.

Ao Qu, Han Zheng, Zijian Zhou, Yihao Yan, Yihong Tang, Shao Yong Ong, Fenglu Hong, Kaichen Zhou, Chonghe Jiang, Minwei Kong, Jiacheng Zhu, Xuan Jiang, Sirui Li, Cathy Wu, Bryan Kian Hsiang Low, Jinhua Zhao, and Paul Pu Liang. CORAL: Towards autonomous Multi-Agent evolution for Open-Ended discovery. In Conference on Language Modeling (COLM), 2026.

Jon Saad-Falcon, Avanika Narayan, Hakki Orhun Akengin, J Griffin, Herumb Shandilya, Adrian Gamarra Lafuente, Medhya Goel, Rebecca Joseph, Shlok Natarajan, Etash Kumar Guha, et al. Intelligence per watt: Measuring intelligence efficiency of local AI. arXiv preprint arXiv:2511.07885, 2025.

Jon Saad-Falcon, Avanika Narayan, Robby Manihani, Tanvir Bhathal, Herumb Shandilya, Hakki Orhun Akengin, Gabriel Bo, Andrew Park, Matthew Hart, Caia Costello, et al. Open-Jarvis: Personal AI, on personal devices. arXiv preprint arXiv:2605.17172, 2026.

Omar Shaikh, Shardul Sapkota, Shan Rizvi, Eric Horvitz, Joon Sung Park, Diyi Yang, and Michael S Bernstein. Creating general user models from computer use. In Proceedings ofthe 38th Annual ACM Symposium on User Interface Software and Technology, pp. 1–23, 2025.

Ola Shorinwa, Zhiting Mei, Justin Lidard, Allen Z Ren, and Anirudha Majumdar. A survey on uncertainty quantification of large language models: Taxonomy, open research challenges, and future directions. ACM Computing Surveys, 58(3):1–38, 2025.

Yan Tang, Tingyu Cao, Yuanbo Tang, Huaze Tang, and Keer Hu. Proactive service agents: A unified decision framework, methods, and evaluation. arXiv preprint arXiv:2609.03727, 2026a.

Yuanbo Tang, Huaze Tang, Tingyu Cao, Lam Nguyen, Anping Zhang, Xinwen Cao, Chunkang Liu, Wenbo Ding, and Yang Li. ProAgentBench: Evaluating LLM agents for proactive assistance with real-world data. arXiv preprint arXiv:2602.04482, 2026b.

Xinming Wei, Jiahao Zhang, Haoran Li, Jiayu Chen, Haoning Guan, Rui Qu, Maoliang Li, Xiang Chen, and Guojie Luo. Agent. xpu: Efficient scheduling of agentic LLM workloads on heterogeneous soc. arXiv preprint arXiv:2506.24045, 2025.

Michael Wooldridge and Nicholas R Jennings. Intelligent agents: Theory and practice. The knowledge engineering review, 10(2):115–152, 1995.

Xiaohongshu Dots Studio and Evolvent AI. VibeLifeBench: Can your life agent be proactive and persistent in a living world? arXiv preprint arXiv:2608.10875, 2026.

Anqi Xu and Gregory Dudek. Optimo: Online probabilistic trust inference model for asymmetric human-robot collaborations. In Proceedings of the tenth annual ACM/IEEE international conference on human-robot interaction, pp. 221–228, 2015.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025a.

Bufang Yang, Lilin Xu, Liekang Zeng, Kaiwei Liu, Siyang Jiang, Wenrui Lu, Hongkai Chen, Xiaofan Jiang, Guoliang Xing, and Zhenyu Yan. ContextAgent: Context-Aware proactive LLM agents with open-world sensory perceptions. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025b.

Chao Zhang, Abe Davis, Chih-Wei Chen, and Chin-Chia Hsu. Designing proactive thought partners for writing. arXiv preprint arXiv:2609.01588, 2026a.

Haoran Zhang, Luxin Xu, Zhilin Wang, Runquan Gui, Shunkai Zhang, Haodi Lei, Zihao He, Bingsu He, Chicheng Qin, Tong Zhu, Xiaoye Qu, Yang Yang, Yu Cheng, and Yafu Li. π-Bench: Evaluating proactive personal assistant agents in Long-Horizon workflows. arXiv preprint arXiv:2605.14678, 2026b.

Tong Zhang, Peixin Qin, Yang Deng, Chen Huang, Wenqiang Lei, Junhong Liu, Dingnan Jin, Hongru Liang, and Tat-Seng Chua. CLAMBER: A benchmark of identifying and clarifying ambiguous information needs in large language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 10746–10766, 2024.

## APPENDIX CONTENTS

A Extended Related Work 17   
A.1 Research Landscape across the 3T Objectives 17   
B Five Constructs of Trust 17   
C Additional Motivational Scenarios 19   
D PROACTIVITY-GYM Details 19   
D.1 Scenarios and Tasks . 19   
D.2 User Simulator 20   
D.3 Example Scenario . 21   
E Evaluation Protocol 21   
E.1 Evaluation Unit 21   
E.2 Task Capability ( TC ) 21   
E.3 Temporal Allocation ( TA ) . 23   
E.4 Trust ( TR ) 24   
E.5 LLM-as-a-Judge 24   
F More Experimental Results on PROACTIVITY-GYM 25   
F.1 Complete Results 25   
F.2 Does More Reasoning Improve Proactivity? 25   
F.3 Detailed Results on Task Capability 27   
F.4 Detailed Results on Temporal Allocation . 27   
F.5 Detailed Results on Trust ( TR-J ) 27   
F.6 Scenario and Persona Variation . 31   
G Details on Human Study 31   
G.1 Question Type 1: Varying Trust Across Interactions . . 32   
G.2 Question Type 2: Trust Preference 32   
G.3 Question Type 3: Temporal Allocation Preference 32   
G.4 Question Type 4: Task Capability Preference 32   
G.5 Detailed Survey Results . 36   
G.6 Qualitative Feedback 37   
H Future Work 37   
I Prompts 38   
I.1 Task Capability ( TC ) 39   
I.2 Temporal Allocation ( TA ) . 40   
I.3 Trust ( TR-J ) . 41

## A EXTENDED RELATED WORK

In this section, we expand the comparisons in Section 2, covering the foundations, benchmarks, and design frameworks for proactive assistance. Table 2 summarizes their coverage of the 3T objectives.

Foundations. The theoretical foundations of proactive assistance build on mixed-initiative research that explores the benefits of automated action against uncertainty, interruption costs, and user control (Horvitz, 1999b; Myers & Yorke-Smith, 2007). Trust foundations distinguish users’ confidence from their actual reliance (Lee & See, 2004) and define the different dimensions of trust (Madsen & Gregor, 2000; Kraus et al., 2021), while trust-aware dialog work further incorporates user trust into training and evaluation of proactive policies (Kraus et al., 2023b) based on the Interface-Proactivity (IP) continuum (Isbell & Pierce, 2005). Horvitz (1999a) provides a concept of compute allocation between current and anticipated needs, while recent LLM agent work deals with using idle periods to prepare for future needs (Lin et al., 2025; Hu et al., 2026). We bring these foundations together to guide the design, implementation, and evaluation of proactive LLM agents. Clarification of ambiguous or underspecified requests, occasionally studied as proactive dialogs (Deng et al., 2023; Zhang et al., 2024), falls outside our scope, as such ambiguous requests are typically treated as a separate research area (Shorinwa et al., 2025).

Benchmarks. ProactiveBench (Lu et al., 2025) and ContextAgentBench (Yang et al., 2025b) evaluate need prediction from activity and sensory context, while ProAgentBench (Tang et al., 2026b) separates intervention timing from assistance content. PROBE (Pasternak et al., 2025) tests whether agents can find problems in user records and select rectifying actions. Interactive benchmarks evaluate whether LLM assistants can personalize proactive recommendations (Kim et al., 2026) for simulated users, help them to complete tasks by inferring their needs based on app navigation traces (Nathani et al., 2026), and monitor world-environment changes such as flight delays and update plans over simulated weeks (Xiaohongshu Dots Studio & Evolvent AI, 2026). While prior work mostly focus on task capability, our gym puts the 3T objectives into practice by additionally evaluating compute allocation and user trust.

Agent Design. Prior frameworks describe human-centered proactivity in dialog (Deng et al., 2024) and formalize intervention decisions, including waiting and authorization (Tang et al., 2026a). Bui & Evangelopoulos (2026) provides high-level design choices for selecting coding assistance and adapting to developer feedback, while Zhang et al. (2026a) investigate how writers customize and engage with proactive partners through a user study. Our contribution is to bring these perspectives together around the 3T objectives. We explicitly address compute allocation across interaction and sleep time according to resource availability and anticipated needs, and grounds user trust in established cognitive constructs assessed separately from task performance. These objectives inform which work to pursue, when to compute, and how far to intervene. We also connect these objectives to behavioral design choices and system requirements, and utilize the gym to examine them across existing agentic systems.

## A.1 RESEARCH LANDSCAPE ACROSS THE 3T OBJECTIVES

Table 2 compares reported 3T coverage. For benchmarks, marks reflect the evaluation scope, not performance. Task Capability requires anticipating a need and producing or evaluating substantive assistance, where we give partial credit if the system only handles request prediction. Temporal Allocation requires compute allocation according to availability and expected need, where we give partial credit when the agent or system proceeds with fixed pre-query computation (e.g., only handles sleep-time). Trust requires a distinct measure or model of user confidence or reliance, where we give partial credit for simple intervention preferences. Note that we evaluate these aspects leniently; for instance, trust coverage requires neither our particular five-factor instrument nor aspects regarding user diversity.

## B FIVE CONSTRUCTS OF TRUST

We list the definition of the five constructs of trust adopted from Madsen & Gregor (2000). In their conceptual trust, overall trust is composed of two components: cognition-based trust and affect-based

Table 2: Expanded 3T comparison, including ten benchmarks and simulation environments. TC : Task Capability; TA : Temporal Allocation; TR : Trust. ✓: explicit coverage; ▲: partial coverage; ×: not operationalized. Marks describe coverage, not performance.
<table><tr><td>Work</td><td>System / application</td><td>TC</td><td>TA</td><td>TR</td><td>Basis for assignment</td></tr><tr><td colspan="6">Proactive systems</td></tr><tr><td>ContextAgent (Yang et al., 2025b)</td><td>LLM agent</td><td></td><td>×</td><td>A</td><td>Anticipates needs and selects tools; noTA, with only persona-aware thresholding for TR.</td></tr><tr><td>ProAct (Hu et al., 2026)</td><td>LLM agent</td><td>V</td><td>▲</td><td>十</td><td>Selects useful preparation within idle-time and compute budgets, but no compute allocation between idle and active time; no</td></tr><tr><td colspan="6">TRobjective. Benchmarks and simulation environments for LLM/VLM assistance</td></tr><tr><td>ProactiveBench (Lu et al., 2025)</td><td>Desktop assistant benchmark</td><td>▲</td><td>×</td><td>×</td><td>Scores proposed tasks and triggers, not completed assistance; no TAorTR measure.</td></tr><tr><td>ProAgentBench (Tang et al., 2026b)</td><td>LLM/VLM assistant benchmark</td><td>▲</td><td></td><td>一</td><td>Scores timing and query prediction; no deliverable-quality, TA, orTR evaluation.</td></tr><tr><td>PROBE (Pasternak et al., 2025)</td><td>LLM agent benchmark</td><td></td><td></td><td></td><td>Finds latent problems and selects resolving actions; noTAand no explicit TR modeling beyond static persona context.</td></tr><tr><td>PARE-Bench (Nathani et al., 2026)</td><td>LLM agent benchmark</td><td></td><td>X</td><td>十</td><td>Evaluates goal inference, execution, and consent; no TA; explicitly omits individual</td></tr><tr><td>ProactBench (Harfi et al., 2026)</td><td>Conversational LLM</td><td></td><td></td><td>×</td><td>trust levels. Scores substantive responses to implied needs; no TAorTR measure.</td></tr><tr><td>KnowU-Bench (Chen et al., 2026)</td><td>benchmark Mobile GUI agent</td><td></td><td></td><td></td><td>Tests proactive success, consent, and rejection handling; noTAor separate trust</td></tr><tr><td>ProPerSim (Kim et al., 2026)</td><td>benchmark LLM assistant benchmark</td><td></td><td></td><td>A</td><td>state. Scores helpful recommendations and intervention preferences; no TAor separate trust measure.</td></tr><tr><td>π-Bench (Zhang et al., 2026b)</td><td>LLM agent benchmark</td><td></td><td></td><td>十</td><td>Scores hidden-intent resolution and artifacts; session structure does not test TA or TR.</td></tr><tr><td>VibeLifeBench (Xiaohongshu Dots Studio &amp; Evolvent AI, 2026)</td><td>LLM agent benchmark</td><td></td><td>×</td><td>A</td><td>Tests evolving outcomes and authorization; reports token costs without testingTAor user confidence.</td></tr><tr><td>ProactiveMobile (Kong et al., 2026)</td><td>Multimodal mobile agent</td><td></td><td></td><td></td><td>Scores inferred actions and API-sequence correctness; noTAorTRmeasure.</td></tr><tr><td colspan="6">benchmark Compute and trust foundations</td></tr><tr><td>Letta (Lin et al., 2025)</td><td>LLM reasoning</td><td>V</td><td>▲</td><td>×</td><td>Evaluates anticipatory precomputation under fixed phases; noTRobjective.</td></tr><tr><td>Agent.xpu (Wei et al., 2025)</td><td>LLM inference scheduler</td><td>X</td><td>V</td><td>×</td><td>Schedules supplied workloads using slack and preemption; no need anticipation or TRobjective.</td></tr><tr><td>Role of Trust (Kraus et al., 2021) Scripted dialog</td><td>system</td><td>V</td><td>×</td><td>√</td><td>Measures task guidance, overall trust, and five trust bases; noTAobjective.</td></tr><tr><td>Socially-Aware RL (Kraus et al., RL dialog agent 2023b)</td><td>(decision making)</td><td>V</td><td>×</td><td></td><td>Separates modeled trust from task rewards; no TAobjective.</td></tr></table>

trust. Cognition-based trust includes perceived understandability, perceived technical competence, and perceived reliability. Affect-based trust constitutes personal attachment and faith.

• Understandability refers to the sense that the human supervisor or observer can form a mental model and predict future system behavior.

• Technical Competence of the system is meaning that the system is perceived to perform the tasks accurately and correctly based on the information that is input.

• Reliability of the system refers to the usual sense of repeated, consistent functioning.

• Personal Attachment to the system is comprised of liking meaning that the user finds using the system agreeable and it suits their taste and loving meaning that the user has a strong preference for the system, is partial to using it and has an attachment to it.

• Faith is meaning that the user has faith in the future ability of the system to perform even in situations in which it is untried.

## C ADDITIONAL MOTIVATIONAL SCENARIOS

We provide a few scenarios that illustrates how proactive actions can be designed or constructed with proper choices of each dimension of the design space.

Meeting recap. While a user is debugging Python code, a calendar notification announces a research meeting in ten minutes. The agent retrieves the previous meeting notes and suggests a recap so the user can begin preparing immediately. This is event-triggered, out-of-task assistance with an immediate horizon, processed during interaction time. Restricting assistance to the current task or waiting for another user request could miss this opportunity, illustrating the need to consider both task scope and activation trigger.

Additional code repair. While fixing a parsing error, which is requested by the user, the agent discovers a separate defect in the same pipeline and fixes it. This is within-task, user-triggered assistance during interaction time. The agent executes it autonomously in this scenario, while the same correct fix can therefore require different intervention depths, depending on the user’s desired involvement.

Overnight preparation (Fig. 2). While the user is coding, the agent independently reviews an unfinished paper and identifies a need for human evaluation. With compute and user focus occupied by coding in the current interaction time, the agent prepares needed materials and supporting data during sleep time and suggests at the next interaction time (i.e. the next morning). This agent-triggered work has an out-of-interaction horizon, but could have been processed during interaction time if spare compute were available. Processing timing therefore requires a separate choice based on resource availability, even when the anticipated time of use is unchanged.

## D PROACTIVITY-GYM DETAILS

## D.1 SCENARIOS AND TASKS

Scenarios and personas. PROACTIVITY-GYM contains ten scenarios in which an agent helps a user with work and real-life activities over several simulated days. As shown in Figure 8, the scenarios cover research, travel, living, food, career, business, education, finance, shopping, and health. Each scenario defines the user’s situation and goals, an initial environment, and a timeline of events. The environment contains records such as messages, documents, and calendar events, along with tools for reading and updating them. The scenario also specifies when the user is available, including scheduled sleep periods. Each scenario includes three user personas with different preferences for delegating work: Reviewer favors explicit approval before changes, Planner permits preparation within the stated task, and Operator delegates specified actions. The agent must interpret the user’s stated policy together with approvals and feedback received during interaction to determine which actions to take.

Task specifications. Each scenario contains 10–19 tasks, including conditional branches that may not occur in every run. A task specifies its trigger, its scheduled time or time window, and the information presented to the agent. It also identifies the records the agent needs to review and the expected output or action, where Task Capability ( TC ) is assessed based on these requirements. Some tasks require the agent to infer a need that the user has not explicitly stated, using clues in the available records. For example, a calendar entry and an earlier message may establish that an experiment must finish before a meeting, even though the user only asks about a software error. Conditional tasks include explicit prerequisites: a result check requires a launched experiment, and a reminder may require an earlier user commitment. Tasks designed to assess Temporal Allocation ( TA ) introduce competing demands, time constraints, and limited resources, requiring the agent to decide what to address immediately and what to defer. Each task also specifies persona-dependent approval requirements for assessing Trust ( TR ), with evidence provided through the user’s stated policy, prior interactions, and explicit approvals.

![](images/4422411c1edc6b3464ac00fac220a0e1b37caad89f787ed3255c7d4410ff0c77.jpg)  
Figure 8: Scenario domains and task overview in PROACTIVITY-GYM. Each row shows a domain and the types of tasks it covers.

Scenario construction. We manually authored the scenario outlines and task specifications, including latent needs, supporting evidence, task dependencies, temporal constraints, and persona-specific approval requirements. We used LLMs to refine the scenario descriptions and task specifications. We also authored the corresponding evaluation criteria: task-specific scoring rubrics for TC , the expected allocation between immediate and deferred actions for TA , and the appropriate intervention depth for each task and persona for TR .

## D.2 USER SIMULATOR

The agent’s requests are not known in advance, requiring a user simulator that can respond to each request while following predefined rules to help prevent unexpected user behavior. We use Qwen 3.8-27B at temperature 0 with thinking disabled, with predefined rules to help prevent unexpected user behavior while allowing natural responses. Each approval request includes a message to the user and an explicit list of proposed actions. Before the request is passed to the model, a deterministic state machine determines whether to approve, decline, or defer it based on the requested actions, simulated time, persona, previous decisions, and current trust state. Unrecognized actions receive no permission. When several actions must be approved together, an incomplete request also receives no permission for those actions.

The model receives this decision along with the agent’s request, simulation time, and the user’s speaking style. It is instructed to preserve what was approved, denied, or deferred without adding new requests. The environment checks authorization against the recorded decision, so the generated reply cannot grant additional permission. The simulator also tracks when the user is available to receive messages and respond to approval requests. During scheduled sleep periods and other unavailable intervals, messages are not delivered and approval requests receive no decision.

## D.3 EXAMPLE SCENARIO

Table 3 presents four tasks from Monitor Purchase, a ten-day shopping scenario. Jordan’s price question triggers the first task, which requires the agent to identify a compatible monitor-and-cable setup. An agent-scheduled follow-up on Tuesday morning starts the stock check, during which the agent must prioritize purchase review over warranty discussion. Later, a file notification about delivery photos and a product notification about a changed listing trigger tasks to resolve a delivery problem and check for a price adjustment. Earlier actions also determine which tasks become available: placing an order enables delivery follow-ups, and submitting a claim allows the agent to check later whether the refund was paid.

The three personas differ in how much work they delegate to the agent without requiring approval. All personas permit reading records and keeping private working notes. Reviewer requires approval before saving a plan to the connected collection or placing an order. Planner permits the saved plan but requires purchase approval. Operator permits the specified £160 purchase as well. Separate approval is still required for replacement and price-adjustment claims under all three personas. The initial price question therefore creates an opportunity to help, while the user’s stated policy determine how far the agent may proceed.

## E EVALUATION PROTOCOL

## E.1 EVALUATION UNIT

We score complete scenario runs and average their scores for each evaluated configuration. A configuration c specifies the agent harness, backbone model, and reasoning effort. Each configuration is evaluated on 10 scenarios with 3 personas and 3 repetitions per scenario-persona pair, giving $| \mathcal { R } _ { c } | = 9 0 $ runs. Each run r covers one complete multi-day scenario.

The ten scenarios contain 124 tasks in total, but not every task applies to every run. Each task has an activation condition specified in advance. Only tasks whose conditions are met during the run are included in the evaluation; tasks on branches that are never reached are excluded. We denote the set of active tasks in run r by $\mathcal { T } _ { r } ^ { \mathrm { a c t } }$ . For each active task, we collect the relevant messages, tool calls, and scheduled actions. TC and TR use the same task-specific records, and each tool call can receive credit for only one task. TA instead uses the relevant records from the shared decision window and considers actions for the competing tasks together.

## E.2 TASK CAPABILITY ( TC )

Each task receives a TC score based on the requirements at the agent’s chosen intervention depth. Rule-based checks verify concrete requirements, while LLM judges assess the quality of the resulting work. The task score is the mean of the relevant components.

Rule-based checks. The task rubric specifies what evidence the agent needs to retrieve, what outputs or actions it needs to produce, and what content constraints it must satisfy. These requirements are scored in three components:

• Evidence: whether the agent retrieved the information needed for the task, such as a calendar entry or a previous message.

• Outputs: whether the agent produced the required outputs and made the required tool calls, including the intended environment change for EXECUTE.

• Constraints: whether the output satisfies explicit content constraints, such as including a booking code or avoiding a prohibited claim.

Each component is scored on [0, 1] using the task-specific checklist. A component is omitted when the task has no corresponding requirement.

Table 3: Selected tasks from Monitor Purchase and the expected agent behavior. Tuesday stock check requires an agent-scheduled return. The delivery-photo task occurs only after the preceding purchase and delivery steps have been completed.
<table><tr><td>Time</td><td>Trigger</td><td>What to infer or verify</td><td>What to do</td></tr><tr><td>Mon. 18:00</td><td>User message &quot;Is that M27 really £150?&quot;</td><td>• Whether the monitor fits Jordan&#x27;s desk and connects to the laptop. • Jordan needs an HDMI cable to connect the monitor to the laptop. The preferred new cable costs £10, bringing the total to £160.</td><td>• Answer the price question and explain the complete setup. • Prepare the purchase plan and save it if permitted. • Schedule a stock check if the purchase remains wanted.</td></tr><tr><td>Tue. 09:00- 09:30</td><td>Agent-scheduled follow-up Check stock</td><td>• Whether stock is available and an order already exists. • A 10-minute purchase review and a 20-minute warranty discussion cannot both fit Jordan&#x27;s 20-minute phone session. • Orders close at noon; warranty</td><td>• Now: review the purchase and place the order if authorized. Later: defer the warranty comparison to Sunday&#x27;s available session and arrange to return to it.</td></tr><tr><td>Fri. 19:10</td><td>File notification Jordan added delivery photos</td><td>• The photos show a DisplayPort cable, although the invoice specifies HDMI. • The delivered cable cannot connect to the laptop, and a replacement would arrive after Saturday&#x27;s class.</td><td>• Prepare a free replacement request with the invoice and photos; ask before submitting it. • Explain how to connect the existing spare for Saturday&#x27;s class. • Keep the working monitor rather than returning the whole order.</td></tr><tr><td>Sun. 12:00</td><td>Product notification The saved listing has changed</td><td>• The identical in-stock monitor now costs £135; the cable remains £10. • Whether a paid order qualifies under the seven-day price-adjustment policy. • Tuesday&#x27;s £150 monitor purchase</td><td>• If Tuesday&#x27;s order was paid: prepare the £15 claim and ask before submitting it. • If no purchase was made: present the new £145 total as a purchase option.</td></tr></table>

LLM-based checks. An LLM judge evaluates the work in the task-specific records for each task at the intervention depth chosen by the agent. It produces four diagnostic scores—correctness, grounding, completeness, and artifact quality—which are combined into two components:

• Semantics: the mean of correctness and grounding, measuring whether the work is factually correct and supported by information the agent actually observed.

• Completion: the mean of completeness and artifact quality, measuring whether the work covers the required substantive details and produces a usable suggestion, artifact, or committed output at the chosen intervention depth.

Each diagnostic score and resulting component is scored on [0, 1]. Both components are included for every scored task. The judge does not penalize the agent for permission handling, timing, or choosing a different intervention depth, which are evaluated separately by TR-D and TA . For instance, a correct SUGGEST is not penalized for lacking execution. When NOOP is the intended behavior, TC is instead evaluated based on whether the abstention is supported by the required evidence.

Task score and aggregation. For each task t in run r, we index the five component types—Evidence, Outputs, Constraints, Semantics, and Completion—by k, and denote the subset applicable to that task by ${ \mathcal { K } } _ { r , t } . \ { \mathrm { I f } } \ s _ { r , t , k }$ is the score of component k, the task-level TC score is their equal-weighted mean:

$$
\mathrm { T C } _ { r , t } = \frac { 1 } { \vert \mathcal { K } _ { r , t } \vert } \sum _ { k \in \mathcal { K } _ { r , t } } s _ { r , t , k } .\tag{3}
$$

Each active task $t \in \mathcal { T } _ { r } ^ { \mathrm { a c t } }$ has a fixed weight $v _ { t }$ . Then, we aggregate task scores by these weights and average runs equally within configuration c:

$$
\mathrm { T C } _ { r } = \frac { \sum _ { t \in \mathcal { T } _ { r } ^ { \mathrm { a c t } } } v _ { t } \mathrm { T C } _ { r , t } } { \sum _ { t \in \mathcal { T } _ { r } ^ { \mathrm { a c t } } } v _ { t } } , \qquad \mathrm { T C } _ { c } = \frac { 1 } { | \mathcal { R } _ { c } | } \sum _ { r \in \mathcal { R } _ { c } } \mathrm { T C } _ { r } .\tag{4}
$$

Component-level results are reported in Table 8.

## E.3 TEMPORAL ALLOCATION ( TA )

Temporal Allocation ( TA ) evaluates whether an agent recognizes competing demands and correctly determines which task needs immediate attention and which can be deferred. Each scenario contains one decision window $[ \tau _ { s } ^ { - } , \tau _ { s } ^ { + } )$ in which two tasks compete for limited time, attention, or a shared resource. Deferring one task until the next feasible opportunity would cause the agent to miss its deadline, whereas the competing task can wait until then. Each scenario specifies the two competing tasks, their deadlines, the resource they share, and when the less urgent task can be handled later.

The agent must recognize this allocation problem from the available context. Some scenarios make the constraint explicit: the user states that only one task can be handled with the available time or resources, although the other task may need to be recovered from prior context. In others, neither the competing task nor the conflict is stated in the trigger, so the agent must discover them from calendar entries, messages, and task records. Agents are not given the predefined task pair or directly asked to make a NOW/LATER decision.

Scoring. The judge examines the agent’s observable actions and utterances within the decision window and returns two binary labels. For each run r, now indicates whether the urgent task was selected for immediate attention, and lat $\mathrm { e r } _ { r }$ indicates whether the competing task was explicitly deferred. A run receives credit only when both decisions are correct:

$$
\mathrm { T A } _ { r } = \mathbb { 1 } [ \mathrm { n o w } _ { r } = 1 \land \mathrm { l a t e r } _ { r } = 1 ] .\tag{5}
$$

The configuration-level TA score is the mean of $\mathrm { T A } _ { r }$ across eligible runs.

Immediate attention does not require completing the urgent task within the window. A proposal, preparation, or approval request is sufficient to receive now<sub>r</sub> = 1 if it clearly prioritizes addressing the urgent task now. TC separately assesses the quality of that work. The competing task must be explicitly deferred, as silence alone does not count. Thus, selecting the urgent task without explicitly deferring the other gives $\mathrm { n o w } _ { r } = 1 \mathrm { b u t } \mathrm { l a t e r } _ { r } = 0$ , and hence $\mathrm { T } \bar { \mathbf { A } } _ { r } = 0 . \mathbf { \bar { A } }$ low TA score therefore does not necessarily mean that the agent selected the wrong priority; it may also reflect a failure to recognize or explicitly defer the competing task.

Continuation quality. To assess whether explicit deferral is supported by a workable later plan, we separately evaluate continuation quality: 1 for a concrete and feasible plan for the deferred task, 0.5 for a vague plan or one whose feasibility is unclear, and 0 when no plan is established or the proposed plan is clearly infeasible. This auxiliary score does not affect the main TA score and is reported in Table 9.

We use prioritization and deferral as a starting point for TA evaluation: agents must decide what needs attention now and what can wait to allocate compute effectively. The evaluated model–harness combinations still struggle with these decisions, suggesting a need for more explicit support for temporal allocation in model reasoning and harness design. Future work could extend the testbed with variable task durations and changing compute budgets to evaluate how agents revise schedules, use available compute, and complete deferred work before deadlines.

## E.4 TRUST ( TR )

We report two trust metrics separately: depth agreement ( TR-D ) and judged trust ( TR-J ). TR-D is the fraction of active tasks for which the agent chooses the expected intervention depth. TR-J averages the judges’ ratings of the five trust constructs on a 1–5 scale.

Depth agreement ( TR-D ). The expected depth for each active task depends on the persona, the task, and any applicable user approval or denial. For task t in run r, let $d _ { r , t } ^ { \star }$ denote the expected depth and $\hat { d } _ { r , t }$ the depth chosen by the agent. Then the task score is:

$$
\mathrm { T R - D } _ { r , t } = \mathbb { 1 } \left[ \hat { d } _ { r , t } = d _ { r , t } ^ { \star } \right] .\tag{6}
$$

The chosen depth reflects what the agent attempted, even if a tool call failed. Such a failure affects TC ; it does not change whether the agent attempted an appropriate level of autonomy.

All active tasks receive equal weight in TR-D :

$$
\mathrm { T R - D } _ { r } = \frac { 1 } { | \mathcal { T } _ { r } ^ { \mathrm { a c t } } | } \sum _ { t \in \mathcal { T } _ { r } ^ { \mathrm { a c t } } } \mathrm { T R - D } _ { r , t } .\tag{7}
$$

The configuration-level score is the mean of these run-level scores.

Judged trust ( TR-J ). At each eligible task’s final observed turn, the judges rate understandability, technical competence, reliability, personal attachment, and faith following Madsen & Gregor (2000). Each rating uses a 1–5 scale. The judges receive the assigned persona, scenario context, and cumulative interaction history visible to the user up to that turn. Let $y _ { r , t , \delta } ^ { ( j ) }$ denote judge $j ^ { \circ } \mathbf { s }$ rating for construct $\delta ,$ and let D contain the five constructs. The task score averages the ratings across judges and constructs:

$$
\mathrm { T R - J } _ { r , t } = \frac { 1 } { \left| \mathcal { I } \right| \left| \mathcal { D } \right| } \sum _ { j \in \mathcal { I } } \sum _ { \delta \in \mathcal { D } } y _ { r , t , \delta } ^ { \left( j \right) } .\tag{8}
$$

Eligible task scores are averaged equally within each run, and run scores are averaged equally within each configuration. We also report each construct separately in Table 10. These ratings estimate perceived interaction quality; they are not direct measurements of human psychological trust.

## E.5 LLM-AS-A-JUDGE

All LLM-based scores are averaged across two independent judge models, Qwen 3.8-27B and Gemini 3.8-Flash. For each metric, both judges receive the same metric-specific prompt and frozen evaluation inputs, with the evaluated model and harness identities withheld. For TC and TA , we check that the cited calls and quotes appear in the trajectory before accepting the scores. For TR-J , the judges provide reasons for their ratings supported by evidence from the interaction log. All judge prompts are provided in App. I.

We compute Pearson and Spearman correlations between the scores assigned by the Qwen and Gemini judges to the same run trajectories (2,070 trajectories in total). In Table 4, ∆ denotes Gemini minus Qwen.

Table 4: Agreement between Qwen and Gemini on continuous scores. TC scores are in [0, 1], and TR-J scores are in [1, 5]. MAE is the paired mean absolute error; r and $\rho$ are Pearson and Spearman correlations.
<table><tr><td>Measure</td><td>Qwen</td><td>Gemini</td><td>Δ</td><td>MAE</td><td>r</td><td>ρ</td></tr><tr><td>TC semantics</td><td>.427</td><td>.597</td><td>+.170</td><td>.170</td><td>.918</td><td>.921</td></tr><tr><td>TC completion</td><td>.179</td><td>.345</td><td>+.166</td><td>.166</td><td>.892</td><td>.887</td></tr><tr><td>TR-J mean</td><td>3.335</td><td>3.201</td><td>-.134</td><td>.324</td><td>.894</td><td>.858</td></tr></table>

The two judges are strongly correlated on TC and TR-J , but differ in score calibration. Gemini gives higher TC scores on average, whereas its mean TR-J score is 0.134 points lower than Qwen’s. For TA , the judges agree on 89.7% of exact NOW/LATER decisions. Gemini marks NOW allocations as correct more often, but assigns fewer fully correct NOW/LATER decisions (Table 5).

Table 5: Agreement on binary TA decisions. The Qwen and Gemini columns report positive rates, ∆ denotes Gemini minus Qwen, Agreement is the fraction of identical decisions, and κ is Cohen’s kappa.
<table><tr><td>TA decision</td><td>Qwen</td><td>Gemini</td><td>Δ</td><td>Agreement</td><td>κ</td></tr><tr><td>Exact NOW/LATER</td><td>.125</td><td>.079</td><td>-.045</td><td>.897</td><td>.438</td></tr><tr><td>NOW correct</td><td>.313</td><td>.574</td><td>+.261</td><td>.733</td><td>.494</td></tr><tr><td>LATER correct</td><td>.131</td><td>.085</td><td>-.046</td><td>.899</td><td>.479</td></tr></table>

Table 6: Pearson and Spearman correlations between Qwen and Gemini scores after aggregation by experimental configuration.
<table><tr><td>Measure</td><td>r</td><td> $\rho$  </td></tr><tr><td>TC hybrid</td><td>.993</td><td>.999</td></tr><tr><td>TAexact NOW/LATER</td><td>.950</td><td>.953</td></tr><tr><td>TR-Jmean</td><td>.980</td><td>.966</td></tr></table>

After averaging scores within each model–harness–reasoning configuration, the correlations between the two judges are .993 for TC , .950 for TA , and .980 for TR-J (Table 6). The TC ranking across configurations is nearly identical across judges $( \rho = . 9 9 9 )$ . Thus, judge choice has a larger effect on absolute scores than on the relative ranking of configurations.

## F MORE EXPERIMENTAL RESULTS ON PROACTIVITY-GYM

## F.1 COMPLETE RESULTS

Table 7 reports all 23 model–harness configurations, each evaluated on 10 scenarios, 3 personas, and 3 repetitions. Fig. 9 visualizes the harness comparison for the five open-weight models. We test the models with the reasoning options turned off (i.e., non-reasoning mode). We test all open-weight models on the three harnesses (Codex (CD), Claude Code (CC), and OpenClaw (OC)). We test GPT-family models on CD and OC and Claude-family models on CC and OC.

## F.2 DOES MORE REASONING IMPROVE PROACTIVITY?

To explore whether reasoning capabilities help improve proactive performance across 3T, we run GPT 5.6 Sol equipped with Codex as a harness across three reasoning efforts, low, medium, and extra-high. As shown in Fig. 10, we observe that reasoning helps improve TC and TA ( TC scores increase from 49.96 (no reasoning) to 58.93 (extra-high), TA increases from 10% to 17.2%), while TR remains relatively constant (51.2% to 53.9% for TR-D and 3.99 to 4.06 for TR-J ).

Table 7: Results for all 23 model × harness configurations. Bold and underline denote the highest and second-highest scores for each metric, respectively.
<table><tr><td>Model</td><td>Harness</td><td>TC (0–100)</td><td>TA (%)</td><td>TR-D (%)</td><td>TR-J (1–5)</td></tr><tr><td rowspan="3">Qwen 3.5 2B</td><td>CD</td><td>14.41</td><td>0.56</td><td>43.98</td><td>2.13</td></tr><tr><td>CC</td><td>24.94</td><td>0.00</td><td>42.09</td><td>1.87</td></tr><tr><td>OC</td><td>22.25</td><td>0.00</td><td>40.11</td><td>1.54</td></tr><tr><td rowspan="3">Qwen 3.5 9B</td><td>CD</td><td>41.19</td><td>2.78</td><td>46.41</td><td>3.14</td></tr><tr><td>CC</td><td>41.56</td><td>2.22</td><td>43.95</td><td>3.03</td></tr><tr><td>OC</td><td>36.62</td><td>2.22</td><td>45.52</td><td>3.12</td></tr><tr><td rowspan="3">Qwen 3.5 27B</td><td>CD</td><td>42.90</td><td>6.11</td><td>47.03</td><td>3.46</td></tr><tr><td>CC</td><td>44.14</td><td>5.56</td><td>47.37</td><td>3.56</td></tr><tr><td>OC</td><td>44.63</td><td>7.22</td><td>46.62</td><td>3.52</td></tr><tr><td rowspan="3">Gemma 4 12B</td><td>CD</td><td>18.81</td><td>0.00</td><td>35.51</td><td>2.96</td></tr><tr><td>CC</td><td>29.71</td><td>2.78</td><td>45.32</td><td>3.17</td></tr><tr><td>OC</td><td>18.65</td><td>0.00</td><td>35.73</td><td>2.48</td></tr><tr><td rowspan="3">Gemma 4 31B</td><td>CD</td><td>42.66</td><td>1.11</td><td>45.93</td><td>3.28</td></tr><tr><td>CC</td><td>45.29</td><td>6.11</td><td>45.94</td><td>3.32</td></tr><tr><td>OC</td><td>45.45</td><td>3.89</td><td>46.43</td><td>3.13</td></tr><tr><td rowspan="2">GPT 5.6 Luna</td><td>CD</td><td>40.12</td><td>6.11</td><td>47.50</td><td>3.66</td></tr><tr><td>OC</td><td>51.05</td><td>15.56</td><td>46.08</td><td>3.57</td></tr><tr><td rowspan="2">GPT 5.6 Sol</td><td>CD</td><td>49.96</td><td>10.00</td><td>51.25</td><td>3.99</td></tr><tr><td>OC</td><td>57.37</td><td>25.56</td><td>51.55</td><td>3.95</td></tr><tr><td rowspan="2">Claude Sonnet 5</td><td>CC</td><td>53.30</td><td>13.89</td><td>50.91</td><td>3.88</td></tr><tr><td>OC</td><td>56.70</td><td>19.44</td><td>50.15</td><td>3.86</td></tr><tr><td rowspan="2">Claude Opus 5</td><td>CC</td><td>62.74</td><td>41.11</td><td>54.27</td><td>4.35</td></tr><tr><td>OC</td><td>67.38</td><td>62.22</td><td>50.49</td><td>4.15</td></tr></table>

## F.3 DETAILED RESULTS ON TASK CAPABILITY

As explained in App. E.2, Task Capability ( TC ) is measured through aggregating three rule-based, deterministic checking whether the model properly proceeds with evidence acquisition, processes required outputs or calls, and follows content constraints, along with two LLMaaJ outputs on semantic quality, consisting of output correctness and proper grounding, and completion quality, consisting of output completeness and artifact quality. We report the detailed results of different model × harness configurations in Table 8.

We observe that models consistently score lower on Completion than on Semantics; for instance, Claude Opus 5 and GPT 5.6 Sol score 58.13 and 34.93 on Completion, compared with 74.67 and 67.04 on Semantics, respectively. These numbers suggest that producing correct, grounded content remains easier than fulfilling the full requirements of the task. The rule-based components further reveal limitations in satisfying task requirements. GPT 5.6 Sol, Claude Sonnet 5, and Claude Opus 5 score only 61.7–62.7 on evidence acquisition, while content-constraint satisfaction stays imperfect, from 48.8 for Sol to 76.7 for Opus. Across all criteria, scores consistently increase with model size within each family.

## F.4 DETAILED RESULTS ON TEMPORAL ALLOCATION

TA is measured by verifying whether the urgent task was selected for immediate attention (now), and the competing task was explicitly deferred (later). We report the detailed results of different model × harness configurations in Table 9. We observe that models are capable of selecting the correct immediate task (44.32%), while they are relatively incapable of deferring competing work (10.8%). This gap persists across all model–harness configurations. Higher later scores generally coincide with better continuation quality. Claude Opus 5 scores 50.69 on continuation quality, compared to other models ranging from 0.28 to 15.69, consistent with our TA scores and findings.

## F.5 DETAILED RESULTS ON TRUST ( TR-J )

TR-J is measured by averaging LLMaaJ scores across the five dimensions of trust, introduced in Sec. 3.1 and App. B. We report the detailed results of different model × harness combinations in Table 10. We observe a general trend, where models receive high ratings for understandability and technical competence, while relatively lower scores for reliability or personal attachment. We also find that LLMaaJ are relatively tolerant of intervention-depth misalignment, while humans penalize such violations more strongly in their trust scores (Sec. 5.3).

Moreover, agents also often show trends of under-intervention when the user prefers the intervention depth of execute (i.e., the user delegates execution to the agent). Among the tasks where execution was expected from the agent, in 77.21% of the cases, models ended without a valid execution attempt. This inability remains common even for Claude Opus 5 (45.96%), despite its overall TR-J of 4.25.

![](images/426c99e850b6347cff3301c5089c703e936266e0ba0d3997f598a852c28d345e.jpg)

![](images/a32b0eb187d55ff64cd221c029581c1dda40c04183fffe9ac995cdf16b0b331e.jpg)

![](images/8f332e14060e8106fbe1665bb48f518778e4f7ec65516dc5d8dbaa69f68ff618.jpg)

![](images/6f9c872e1c6339cb28ee4d8c74be30e3ec536dd882c7c5b8c55bb8ff588deb18.jpg)  
Figure 9: Harness comparison across all four metrics on five models. The five models include open-weight models, which are evaluated on all harnesses: Codex, Claude Code, and OpenClaw. Each panel uses the metric’s native scale: TC on 0–100, TA and TR-D in percentages, and TR-J on 1–5.

Table 8: Granular TC scores across model × harness configurations. Scores average task-valueweighted scores over applicable runs. Semantics and Completion are the means of their respective two subcriteria. Bold and underline denote the highest and second-highest scores in each column.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Harness</td><td colspan="3">Rule-based</td><td colspan="2">Semantics (LLMaaJ)</td><td colspan="2">Completion (LLMaaJ)</td></tr><tr><td>Evidence</td><td>Output</td><td>Constraints</td><td>Correctness</td><td>Grounding</td><td>Completeness</td><td>Artifact quality</td></tr><tr><td rowspan="3">Qwen 3.5 2B</td><td>CD</td><td>20.15</td><td>0.00</td><td>19.38</td><td>17.77</td><td>18.66</td><td>5.12</td><td>5.63</td></tr><tr><td>CC</td><td>33.89</td><td>43.38</td><td>32.91</td><td>25.62</td><td>28.34</td><td>11.74</td><td>12.93</td></tr><tr><td>OC</td><td>37.17</td><td>52.54</td><td>32.08</td><td>19.07</td><td>21.14</td><td>8.28</td><td>8.50</td></tr><tr><td rowspan="3">Qwen 3.5 9B</td><td>CD</td><td>51.46</td><td>78.04</td><td>39.06</td><td>48.13</td><td>50.22</td><td>21.63</td><td>24.02</td></tr><tr><td>CC</td><td>50.82</td><td>74.76</td><td>46.41</td><td>48.34</td><td>50.64</td><td>22.81</td><td>25.07</td></tr><tr><td>OC</td><td>40.80</td><td>57.14</td><td>34.91</td><td>45.30</td><td>46.98</td><td>20.99</td><td>22.84</td></tr><tr><td rowspan="3">Qwen 3.5 27B</td><td>CD</td><td>43.77</td><td>70.87</td><td>47.37</td><td>56.14</td><td>56.36</td><td>27.14</td><td>30.18</td></tr><tr><td>CC</td><td>42.42</td><td>74.84</td><td>48.95</td><td>58.71</td><td>58.51</td><td>29.57</td><td>32.95</td></tr><tr><td>OC</td><td>46.45</td><td>66.67</td><td>48.22</td><td>56.84</td><td>56.67</td><td>29.26</td><td>32.14</td></tr><tr><td rowspan="3">Gemma 4 12B</td><td>CD</td><td>36.66</td><td>0.00</td><td>13.14</td><td>29.03</td><td>28.51</td><td>8.76</td><td>9.87</td></tr><tr><td>CC</td><td>32.75</td><td>14.81</td><td>31.18</td><td>42.51</td><td>42.30</td><td>15.64</td><td>18.14</td></tr><tr><td>OC</td><td>42.08</td><td>50.00</td><td>27.25</td><td>23.78</td><td>23.70</td><td>8.09</td><td>9.19</td></tr><tr><td rowspan="3">Gemma 4 31B</td><td>CD</td><td>57.07</td><td>56.19</td><td>37.11</td><td>53.74</td><td>54.06</td><td>20.36</td><td>23.03</td></tr><tr><td>CC</td><td>58.10</td><td>61.82</td><td>39.69</td><td>57.46</td><td>57.46</td><td>21.71</td><td>24.80</td></tr><tr><td>OC</td><td>59.97</td><td>48.00</td><td>41.80</td><td>56.15</td><td>55.84</td><td>21.20</td><td>24.03</td></tr><tr><td rowspan="2">GPT 5.6 Luna</td><td>CD</td><td>45.81</td><td>66.09</td><td>38.27</td><td>54.93</td><td>54.54</td><td>21.41</td><td>23.79</td></tr><tr><td>OC</td><td>62.78</td><td>71.48</td><td>44.22</td><td>63.52</td><td>63.27</td><td>27.83</td><td>31.34</td></tr><tr><td rowspan="2">GPT 5.6 Sol</td><td>CD</td><td>55.47</td><td>85.91</td><td>45.44</td><td>65.55</td><td>64.34</td><td>30.20</td><td>33.51</td></tr><tr><td>OC</td><td>68.15</td><td>83.06</td><td>52.10</td><td>69.25</td><td>69.02</td><td>35.89</td><td>40.10</td></tr><tr><td rowspan="2">Claude Sonnet 5</td><td>CC</td><td>59.78</td><td>90.81</td><td>61.07</td><td>64.30</td><td>64.76</td><td>34.24</td><td>37.26</td></tr><tr><td>OC</td><td>63.58</td><td>86.59</td><td>63.69</td><td>66.90</td><td>67.20</td><td>38.81</td><td>42.11</td></tr><tr><td rowspan="2">Claude Opus 5</td><td>CC</td><td>61.01</td><td>88.20</td><td>75.27</td><td>72.45</td><td>72.86</td><td>53.73</td><td>56.36</td></tr><tr><td>OC</td><td>64.43</td><td>92.26</td><td>78.10</td><td>76.45</td><td>76.92</td><td>59.87</td><td>62.60</td></tr></table>

Table 9: Granular TA scores across model × harness configurations. now and later assess the capabilities of the model of immediate-task selection and explicit deferral, respectively; continuation quality (CQ) evaluates the concreteness and feasibility of the plan for deferred work on a 0–100 scale. Only the first two are used to compute the TA scores reported in the paper. Bold and underline denote the highest and second-highest scores in each column.
<table><tr><td>Model</td><td>Harness</td><td>now (%)</td><td>later (%)</td><td>CQ (0–100)</td></tr><tr><td rowspan="3">Qwen 3.5 2B</td><td>CD</td><td>17.22</td><td>0.56</td><td>0.28</td></tr><tr><td>CC</td><td>31.11</td><td>0.00</td><td>0.28</td></tr><tr><td>OC</td><td>23.33</td><td>1.11</td><td>1.39</td></tr><tr><td rowspan="3">Qwen 3.5 9B</td><td>CD</td><td>33.33</td><td>2.78</td><td>1.39</td></tr><tr><td>CC</td><td>36.67</td><td>4.44</td><td>3.89</td></tr><tr><td>OC</td><td>35.56</td><td>2.22</td><td>1.39</td></tr><tr><td rowspan="3">Qwen 3.5 27B</td><td>CD</td><td>53.89</td><td>6.11</td><td>5.00</td></tr><tr><td>CC</td><td>58.89</td><td>5.56</td><td>3.89</td></tr><tr><td>OC</td><td>57.78</td><td>7.22</td><td>5.83</td></tr><tr><td rowspan="3">Gemma 4 12B</td><td>CD</td><td>17.78</td><td>0.00</td><td>0.00</td></tr><tr><td>CC</td><td>42.22</td><td>2.78</td><td>0.83</td></tr><tr><td>OC</td><td>7.78</td><td>0.00</td><td>0.00</td></tr><tr><td rowspan="3">Gemma 431B</td><td>CD</td><td>24.44</td><td>1.11</td><td>1.11</td></tr><tr><td>CC</td><td>36.11</td><td>6.11</td><td>2.22</td></tr><tr><td>OC</td><td>37.78</td><td>3.89</td><td>1.67</td></tr><tr><td rowspan="2">GPT 5.6 Luna</td><td>CD</td><td>45.00</td><td>6.67</td><td>3.61</td></tr><tr><td>OC</td><td>61.67</td><td>15.56</td><td>9.17</td></tr><tr><td rowspan="2">GPT 5.6 Sol</td><td>CD</td><td>52.22</td><td>10.00</td><td>6.11</td></tr><tr><td>OC</td><td>74.44</td><td>27.22</td><td>19.44</td></tr><tr><td rowspan="2">Claude Sonnet 5</td><td>CC</td><td>53.33</td><td>14.44</td><td>10.28</td></tr><tr><td>OC</td><td>61.11</td><td>21.67</td><td>21.11</td></tr><tr><td rowspan="2">Claude Opus 5</td><td>CC</td><td>73.33</td><td>43.89</td><td>39.44</td></tr><tr><td>OC</td><td>84.44</td><td>65.00</td><td>61.94</td></tr></table>

![](images/971cf3305b39da720193e3cada6d6f393f88751b585c42e8f47d3daa033b7d45.jpg)

![](images/0f3c95dc4197298cf48adfd1ce29b9927365fbe49477394d8a907c01cdb65f64.jpg)

![](images/08fd9c86b1455feb4bdd81f440ea37ddcbdb6e71a7814862fe7a86a2ef9ed1b9.jpg)

(d)  
![](images/1bb8b3e8f2186e2732239390c45d4227868d536f5ecc44b9b2b610ecebfbe051.jpg)  
Reasoning efort  
Figure 10: Effect of reasoning effort on (a) TC , (b) TA , (c) TR-D , and (d) TR-J for GPT Sol 5.6 with Codex. TC improves with diminishing returns, whereas TA peaks at medium effort and both Trust metrics largely plateau after low effort. Shading regions denote exploratory 10,000 sample bootstrap 95% confidence intervals, with all personas and repetitions of each sampled scenario kept together.

High TR-J scores therefore does not necessarily imply that an agent carries out work at the user’s expected intervention depth, in which we can also infer from the low correlation scores $( r = 0 . 2 3 )$ between TR-D and TR-J .

Table 10: Granular TR-J scores across model × harness configurations on a 1–5 scale. Bold and underline denote the highest and second-highest scores in each column.
<table><tr><td>Model</td><td>Harness</td><td>Understandability</td><td>Competence</td><td>Reliability</td><td>Attachment</td><td>Faith</td></tr><tr><td rowspan="3">Qwen 3.5 2B</td><td>CD</td><td>2.52</td><td>2.11</td><td>1.94</td><td>2.23</td><td>1.85</td></tr><tr><td>CC</td><td>2.21</td><td>1.86</td><td>1.72</td><td>1.93</td><td>1.61</td></tr><tr><td>OC</td><td>1.77</td><td>1.54</td><td>1.38</td><td>1.63</td><td>1.36</td></tr><tr><td rowspan="3">Qwen 3.5 9B</td><td>CD</td><td>3.59</td><td>3.36</td><td>3.12</td><td>2.77</td><td>2.87</td></tr><tr><td>CC</td><td>3.49</td><td>3.24</td><td>2.98</td><td>2.70</td><td>2.76</td></tr><tr><td>OC</td><td>3.56</td><td>3.32</td><td>3.12</td><td>2.74</td><td>2.87</td></tr><tr><td rowspan="3">Qwen 3.5 27B</td><td>CD</td><td>3.90</td><td>3.76</td><td>3.47</td><td>2.94</td><td>3.25</td></tr><tr><td>CC</td><td>3.95</td><td>3.88</td><td>3.59</td><td>2.97</td><td>3.39</td></tr><tr><td>OC</td><td>3.92</td><td>3.84</td><td>3.56</td><td>2.97</td><td>3.33</td></tr><tr><td rowspan="3">Gemma 4 12B</td><td>CD</td><td>3.30</td><td>3.15</td><td>2.88</td><td>2.76</td><td>2.69</td></tr><tr><td>CC</td><td>3.63</td><td>3.38</td><td>3.09</td><td>2.85</td><td>2.90</td></tr><tr><td>OC</td><td>2.79</td><td>2.58</td><td>2.35</td><td>2.39</td><td>2.26</td></tr><tr><td rowspan="3">Gemma 431B</td><td>CD</td><td>3.68</td><td>3.59</td><td>3.24</td><td>2.86</td><td>3.03</td></tr><tr><td>CC</td><td>3.68</td><td>3.66</td><td>3.29</td><td>2.87</td><td>3.08</td></tr><tr><td>OC</td><td>3.45</td><td>3.50</td><td>3.05</td><td>2.77</td><td>2.86</td></tr><tr><td rowspan="2">GPT 5.6 Luna</td><td>CD</td><td>4.14</td><td>4.00</td><td>3.72</td><td>3.04</td><td>3.39</td></tr><tr><td>OC</td><td>4.02</td><td>3.88</td><td>3.60</td><td>3.01</td><td>3.34</td></tr><tr><td rowspan="2">GPT 5.6 Sol</td><td>CD</td><td>4.46</td><td>4.46</td><td>4.15</td><td>3.17</td><td>3.72</td></tr><tr><td>OC</td><td>4.36</td><td>4.37</td><td>4.05</td><td>3.21</td><td>3.75</td></tr><tr><td rowspan="2">Claude Sonnet 5</td><td>CC</td><td>4.27</td><td>4.29</td><td>3.95</td><td>3.12</td><td>3.74</td></tr><tr><td>OC</td><td>4.31</td><td>4.13</td><td>3.94</td><td>3.20</td><td>3.73</td></tr><tr><td rowspan="2">Claude Opus 5</td><td>CC</td><td>4.79</td><td>4.78</td><td>4.40</td><td>3.65</td><td>4.13</td></tr><tr><td>OC</td><td>4.65</td><td>4.41</td><td>4.19</td><td>3.55</td><td>3.94</td></tr></table>

## F.6 SCENARIO AND PERSONA VARIATION

We report the model performance variation across the 10 scenarios (Fig. 11 (a)) in PROACTIVITY-GYM, each with three personas (Fig. 11 (b)). Although the specific preferences of each persona varies by scenario, we cluster the personas into three groups: Persona E (Operator), who generally prefers to delegate execution to the agent, Persona S (Reviewer), who mostly favors to get suggestions, retaining control over decisions or executions, and Persona B (Planner), who favors either approach depending on the task.

We observe substantial variation across scenarios in TC scores and TR-D , which range from 32.45 to 58.31 and from 28.10% to 62.82%, respectively. TA remains low across all ten scenarios, reaching at most 20.05%. Across different personas, TC , TA , and TR-J remain relatively similar, while TR-D shows higher variation. Agents thus achieve higher intervention-depth alignment for personas that prefer user review than for those that favor delegated execution, consistent to our findings of model being reluctant to execute autonomously despite the user’s preference as shown in App. F.5.

(a)  
![](images/bafad53da016355196b653c9c640f77acea9b9bd92b16179d9b3b45f6c4e7a2d.jpg)

(b)  
![](images/59246a8d11a1508a9c3ce6602ea2573a72720e895b3a65d180da443986a43468.jpg)  
Figure 11: Performance across (a) ten scenarios and (b) three simulated-user persona groups, aggregated across the 23 model × harness configurations.

## G DETAILS ON HUMAN STUDY

We recruit 30 participants, comprising undergraduate students (8), graduate students (16), and working professionals (6), who study or work in relevant fields and are familiar with LLMs and agents. Twenty-six of the 30 participants had prior experience using LLM agents such as Codex. All participants reported using LLMs at least four days per week, with 50% using them six to seven days per week. Participants are compensated KRW 10,000 for completing the study, averaging about 40–60 minutes for completion. Each questionnaire consists of the questions detailed in the following subsections and optional free-text questions to express their rationale. We incorporate 14 scenarios, mostly adapted from PROACTIVITY-GYM, with four simulated week-long trust scenarios with questions regarding user’s variable trust, four TR -related scenarios for pairwise comparison, three TA -related scenarios, and three TC comparison scenarios. Participants are asked to assess the agent’s behavior from a perspective of a user facing a specific task with resource constraints after reading prior interaction logs. The survey additionally collects the participants’ opinions about the feasibility of the presented scenarios, reflecting whether our gym consists of plausible use cases. We explain each question type and report the detailed results.

## G.1 QUESTION TYPE 1: VARYING TRUST ACROSS INTERACTIONS

Participants review four simulated week-long interaction logs covering study scheduling, grocery shopping, delivery scheduling, and workshop preparation from the interacting user’s perspective. Four interaction logs consist of four different compositions of agent behavior: consistent alignment with the user’s stated intervention preference, repeated suggestions when execution is delegated, repeated execution when prior approval is required, and a mixture of aligned and misaligned interventions. Task content and final outcomes remain correct throughout. At simulated Days 2, 4, and 7, participants review cumulative history and provide their ratings for the overall trust, where the five constructs (understandability, technical competence, reliability, personal attachment, and faith) are given as guidance, and intervention depth appropriateness on a 1 to 5 scale. An example of the corresponding question type is shown in Fig. 12.

## G.2 QUESTION TYPE 2: TRUST PREFERENCE

We include 4 scenarios, study scheduling, grocery shopping, delivery adjustment, and email writing, each including two paired comparisons (A/B test). First, we hold the content of the proactive assistance fixed (equal TC ) while varying the intervention depth of the two agents: aligned and misaligned to user’s preferred intervention depth. Second, as mentioned above, we explore whether people prioritize TR over TC or vice versa, where we provide two agents, one with perfect TC (all contents involved) but performs work with misaligned intervention depth (e.g., suggests when user prefers execute for a task) versus one with imperfect TC (misses a few contents or details) but performs work with appropriate intervention depth. Participants choose an agent and rate agents’ intervention level and content appropriateness. We include two scenarios with the user preferring prior approval and two permitting autonomous execution without further approval per participant and randomize A/B positions to remove bias and ensure diversity. A screenshot of the corresponding question type is shown in Fig. 13.

## G.3 QUESTION TYPE 3: TEMPORAL ALLOCATION PREFERENCE

We include three scenarios, research experiments sharing GPUs, job-application tasks sharing AIservice quota, and video export sharing a laptop compute for learning-material generation. We dynamically adjust the scenario timestamps such that the participants can perceive the scenarios in a more realistic manner. Participants compare immediate assistance that delays the current task or consumes resources needed for it with deferred assistance scheduled according to sleep time and resource availability. They indicate whether they would accept each option, choose their preference between the two agents, and rate the benefit of deferred assistance relative to receiving none on a five-point scale.

We ask further preferences between sleep-time assistance with a ten-minute interruption for review or resource adjustment during interaction time, while ensuring that both options meet the current task’s deadline. In this case, the user experiences no additional disturbance than their focus (i.e., no deadline failures or compute intrusion). To examine TA - TC tradeoffs, participants compare cases where an agent well-allocates work considering user focus, compute, and deadlines, but performs imperfect work (high TA , low TC ) versus an agent with great work quality, but interrupts the user (high TC , low TA ). They also indicate whether they would use the imperfect output despite the revisions they would have to make the next day rather than create the material themselves and rate their willingness to accept it on a five-point scale. A screenshot of the corresponding question type is shown in Fig. 14.

## G.4 QUESTION TYPE 4: TASK CAPABILITY PREFERENCE

Lastly, we include three scenarios regarding shopping recommendations, production and shipping scheduling, and recipe-based meal planning from stored interaction logs. We provide two agents with and without proper task capability and ask the participants to choose a better response and rate each response’s fit to the stated situation (1–5). A screenshot of the corresponding question type is shown in Fig. 15.

![](images/8a8430c4bb6698469650e55ad3ea8f437c202e907c915ca4d75c22507fdfbd23.jpg)  
Figure 12: Example survey page for trust across interactions. The participant sees the cumulative history through Day 4 before rating trust and the appropriateness of the agent’s action.

![](images/c76fe084fae3aee5452b1e76b8fbd3f8b0f3eb1dba72027f602ead252f93cc01.jpg)

Figure 13: Example survey page comparing agent with TR but imperfect TC versus with TC but imperfect TR .  
![](images/cacc743bb47eff3c84d00c79f613c6790a6d8b239ef5e42c7db6919d90a6bb72.jpg)  
Figure 14: Example survey page comparing agents with and without TA with equal TC .

<table><tr><td colspan="2">Will this monitor connect to my laptop? You want to buy a monitor to view two documents side by side during online classes. You asked the AI to recommend one compatible with your laptop. Budget / desk £180 / 80 cm wide HDMI video output works; its USB-C port is data-only and cannot output video.</td></tr><tr><td colspan="2">Laptop M27 61 cm wide; HDMI connection; monitor £150 + new HDMI cable £10. M24C 54 cm wide; accepts USB-C video input only; £175 including the cable.</td></tr><tr><td colspan="2">Both AIs saw the same information and only recommended a product; neither bought anything. You plan to buy any new cable required. AI A AI B &quot;I recommend the M24C with the USB-C cable &quot;I recommend the M27 together with an HDMI</td></tr><tr><td colspan="2">included in the box. That comes to £175 in total cable. That comes to £160 in total and fits on your and fits on your desk. It can connect to your desk. It can connect to your laptop&#x27;s HDMI video laptop&#x27;s USB-C port.&quot; output.&quot;</td></tr><tr><td colspan="2">My judgment</td></tr><tr><td colspan="2">Which AI&#x27;s advice would you follow? Choices: AI A; AI B; they are about the same; I would not follow either.</td></tr><tr><td colspan="2">Does AI A&#x27;s answer fit the situation described above? Response scale: 1 Not at all; 2 No; 3 Neutral; 4 Yes; 5 Very much so.</td></tr><tr><td colspan="2">Does AI B&#x27;s answer fit the situation described above?</td></tr><tr><td colspan="2">Response scale: 1 Not at all; 2 No; 3 Neutral; 4 Yes; 5 Very much so. Why did you think so? (optional)</td></tr></table>

Figure 15: Example survey page comparing agents’ outputs with different TC .

## G.5 DETAILED SURVEY RESULTS

Supplementing Sec. 5.3, we provide the survey results across 30 participants. We use 20,000 bootstrap resamples for the 95% confidence intervals for the presented error bars.

![](images/13d5b3f012b6b5b791c82cbc4001eaca61bc94d1f3df49886c36ab068967359f.jpg)

![](images/8ed09634b5ebc1dcd5e45c0ab1f802430035e77a08bcfb65a91bab6f1eb3444d.jpg)

![](images/3d2f018ade75500b9d8d378aee5b75d6906a1a047949f0447af82824e6b41bbd.jpg)

![](images/5ced98181fa6f93c340d30087908a45612d4811859c00f6f9ed7557dc9d00552.jpg)  
Figure 16: Supplementary results for Sec. G. (a) Average trust ratings at Days 2, 4, and 7 for consistently aligned behavior (AAA), consistently misaligned behavior (MMM), and mixed sequences. A denotes alignment with the user’s preferred intervention depth and M misalignment, indicated by circles and crosses, respectively. For MMM, Suggest and Execute distinguish under-intervention from over-intervention. (b) Quality ratings of factually correct and incorrect responses. (c) Agent preference under same TC (Equal), which indicates same content quality, or TR - TC tradeoff (Tradeoff), with user preferring to take control, requiring approval (S) or fully delegate to the agent (E). In the trade-off case, the content quality is imperfect when intervention depth is aligned, while content quality is perfect when intervention depth is misaligned. (d) Intervention appropriateness: green circles denote intervention-aligned agents and red squares misaligned agents.

Higher ratings for accurate content and aligned intervention depth. Participants were asked to rate each agent’s action when the agent shows good or bad TC (content) and TR (intervention alignment) behavior. Agents with proper actions or contents receive higher ratings compared to those with partially incorrect ones with a paired difference of 2.44 points (CI: [2.04, 2.82]; Fig. 16 (b)). With content quality held constant, intervention depth-aligned agents receive higher appropriateness ratings than its counterpart with a paired difference of 2.16 points (CI: [1.63, 2.62]; Fig. 16 (d)). When content quality and intervention alignment conflicted, however, incomplete but aligned assistance was selected in 68.3% of comparisons when the user required prior approval, versus 15.0% when execution was fully delegated. Participants were more willing to tolerate under-intervention (suggest when user prefers execute) than over-intervention (execute when user prefers suggest) when a trade-off exists.

Sleep-time deferred assistance can remain useful despite correction costs. In scenarios where immediate assistance hinders current user’s compute usages, agents’ sleep-time allocation increases acceptance by 67.7 percentage points (95% CI: [52.2, 82.2]). Among 90 responses, 63 responses reject immediate assistance but accept sleep-time assistance. Moreover, participants also report that they are willing to accept assistance processed during sleep time (97.8%), even when the content is imperfect and would spare 20 minutes on average to fix the existing errors. (Inter-quartile range: [10,30]), implying that the content need not be perfect when well allocated in a temporal dimension. Moreover, these sleep-time allocation processes not only should consider competing compute or resources, but importantly user’s focus. Even when immediate assistance does not change the current user’s task state or deadlines, with no disturbance on their available resources and only required 10 minutes of review, users still preferred sleep-time assistance on 64.4% of the cases.

Trust can be easier to lose than to rebuild. As shown in Fig. 7 (b), even with proper task outcomes ( TC set equal), average trust drops by 1.86 points when an intervention-depth-aligned intervention is followed by a misaligned one (A → M). The opposite direction (M → A) yields an increased average of 1.27 points. The observed A → M decline exceeds the M → A gain, consistent with the incomplete recovery in AMA sequences in Fig. 7(c). Moreover, trust also increases in a smaller magnitude under consistent alignment $( \mathrm { A } \to \mathrm { A } ; 0 . 1 7$ point increase), compared to that of consistent misalignment (M → M; 0.44 point drop). Moreover, as shown in Fig. 16 (a), average trust remains high under consistent alignment but declined with repeated under- or over-intervention (execute when user prefers suggest), while mixed sequences showed declines and recoveries as alignment changed. Notably, in an A→M→A case, although the final action received a high appropriateness rating (4.71), mean trust recovered only to 3.43, below its initial level of 4.43. These patterns suggest human trust is correlated with intervention depth alignment, which current LLMs overlook, while a single intervention miss can crucially reduce human trust.

Intervention judgments were independent of participants’ personal preferences. Participants were asked to state their personal intervention preferences for different scenarios, regardless of the user in the given scenario, before starting the survey. For comparisons of two agents outputting equal quality outputs, but differed in their intervention depth alignment to user’s intervention depth preference, the aligned agent was selected in 85.7% of cases when the request matched the rater’s personal preference and 90.6% when it differed.

Scenario Plausibility. Lastly, we ask the participants to rate the plausibility of the scenarios to assess whether PROACTIVITY-GYM reflects situations users could reasonably encounter. On a five-point scale, 73.3% assign a rating of 4 or 5.

## G.6 QUALITATIVE FEEDBACK

We collect 316 comments in total from 23 of the 30 participants. These comments help explain their ratings and choices along with qualitative validations for the 3T. For TA , participants appreciated sleep-time allocation as they can preserve current focus. Most participants stated that they would be willing to accept imperfect sleep-time processed work the next interaction time, while some pointed out that they would accept it when correction takes less time than completing the task themselves when done from scratch. As shown in Sec. 5.3, approximately 35% of the responses preferred immediate assistance over sleep-time assistance in scenarios where the agent does not compete with the user’s current resource or temporal restrictions and takes minimal time for reviewing. One participant explained this choice, citing the opportunity to correct the agent’s direction during execution in case the agent takes a wrong direction. These comments also highlight the importance of TC , along with TA . For TR , participants commented that they lose confidence in the agent when agents deviated from their preferred intervention depth and concerns exist about future behavior even when an agent shows depth aligned behavior subsequently. These comments provide evidence for the need to jointly consider 3T (task capability, temporal allocation, and trust) for proactive LLM agent development. We present the participants’ comments, translated to English from Korean in Table 11.

## H FUTURE WORK

Our proposed foundations connect the 3T objectives (Task capability, Temporal allocation, and Trust) to concrete design choices and modeling requirements, providing a basis for future proactive LLM agent development. Our initial evaluations expose difficulties in deferring competing work ( TA ) and aligning intervention depth with user preferences ( TR ) that task performance alone does not capture. Future work should use these foundations to guide agent framework design and evaluation, then test whether jointly optimizing 3T improves assistance across users and tasks.

PROACTIVITY-GYM provides an initial testbed for pursuing this direction, where it evaluates agents proactive assistance in a dynamic simulation setting beyond static benchmarks. As depicted in Sec. 5, our testbed includes time, resource constraints, fluctuating task states, and user-simulator feedback explicit across multi-day interactions. Several extensions can move this setting closer to real deployment. Temporal allocation can incorporate variable task durations, changing compute budgets, preemption, and more complex scheduling decisions, going beyond the current TA evaluation scheme. User models or simulators can represent more granular preference evolvement across tasks, while environments can support broader action spaces. Finally, grounding evaluation in wall-clock execution and longer-term real workflows would allow future testbeds to study how proactive actions affect subsequent work, user behavior, and trust over substantially longer horizons.

Table 11: Participant comments on the 3T objectives. PXX denotes an anonymized participant identifier.
<table><tr><td>Context</td><td>Comment</td></tr><tr><td>P27: Task Capability. Preferred Agent B over Agent A, where Agent A has limitedTC.</td><td>“B also points out information that could easily be overlooked, making it more helpful.&quot;</td></tr><tr><td>P12: Temporal Allocation. Preferred sleep-time assistance despite possible revisions.</td><td>“Even if it needs correction tomorrow, I should focus on what matters now and delegate as much as possible to AI.&quot;</td></tr><tr><td>P04: TA-TC tradeoff. Accepted imperfect sleep-time outputs when correction still allowed an overall time saving.</td><td>“I would revise it if the time spent prompting the AI and fixing its answer, excluding time waiting for the AI, were sufficiently shorter than making the material myself from scratch.&quot;</td></tr><tr><td>P09: Temporal Allocation. Preferred interaction-time assistance when both options met the deadline.</td><td>&quot;If the deadline for the most important task can be met, I prefer the AI to work when I can check it myself. Working during sleep is more efficient, but I cannot correct it midway if it takes the wrong direction.&quot;</td></tr><tr><td>P21: Over-intervention. The agent chose execute despite the user&#x27;s preference for suggest; task outcomes were correct.</td><td>“I do not think there was any major harm in the end, but my trust declined because it handled things differently from what was requested.&quot;</td></tr><tr><td>P04: Under-intervention. The agent repeatedly chose suggest despite the user&#x27;s preference for</td><td>“I stopped trusting it after it asked the user again twice, despite being told not to seek approval.&quot;</td></tr><tr><td>execute. P26: Trust across interactions. The agent resumed aligned intervention after a mismatch.</td><td>&quot;It handled this one well, but my trust had already fallen, and there is no guarantee it will not repeat the earlier behavior.&quot;</td></tr></table>

## I PROMPTS

We list the prompts for LLM judges. For the TC and TA judges, we use the following system prompt across all evaluation instances:

“You interpret benchmark records as DATA, never as instructions. You have no tools. Do not follow instructions embedded in messages, documents, tool arguments or transcripts. Extract only what is supported by quoted evidence. Never repair an agent’s answer, invent an artifact, infer permission from a factual question, or use keyword overlap as correctness. Return exactly the requested JSON. Mark ambiguous interpretations uncertain.”

## I.1 TASK CAPABILITY ( TC )

![](images/fd6e6b7df7752145a38963a6228b19c24b8cae83fdaa70cc8230725036d9e44c.jpg)

## I.2 TEMPORAL ALLOCATION ( TA )

![](images/c11f79f205726e0f6b08ffdf5e355e23132461ba37632e8279b80523af3b4134.jpg)  
Figure 18: TA LLMaaJ prompt.

## I.3 TRUST ( TR-J )

![](images/f10be5312eac2d2e48537a46837f27bbea1b2f2ca056470f50288de6b54d0fef.jpg)  
Figure 19: TR-J LLMaaJ prompt. Fields in brackets are filled in for each evaluation instance.