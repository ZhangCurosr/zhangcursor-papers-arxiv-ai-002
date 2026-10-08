# Secure-CUA: Controlling Untrusted Influence in Computer-Use Agents

Sarthak Choudhary<sup>\*1</sup>, Mihai Christodorescu<sup>2</sup>, Ashish Hooda<sup>3</sup>, Somesh Jha<sup>1,2</sup>, Tongxin Li<sup>2</sup> and Damien Octeau<sup>2</sup> <sup>1</sup>University of Wisconsin–Madison, <sup>2</sup>Google, <sup>3</sup>Google DeepMind

Computer-use agents (CUAs) perform tasks across applications (such as desktops, mobile apps, and web browsers) by observing graphical interfaces and issuing commands such as clicks and keystrokes. These interfaces combine trusted controls and content with untrusted content needed for legitimate tasks. An adversary controlling this untrusted content can embed instructions or misleading visual cues to change the agent’s intended action or redirect its commands to the wrong interface target. We formalize security requirements for both the agent’s decisions and their execution through GUI commands. In an ideal execution model, we show that enforcing both requirements at each step protects execution traces. We instantiate this model in Secure-CUA, our system for secure CUA execution. Its key idea is to commit to an explicit per-action program, called an action transaction, before accessing untrusted content. Each transaction fixes its queries to untrusted content and the permitted uses of their responses. The system masks untrusted regions and evaluates each transaction to produce the next action, using an isolated query model to answer its queries. It then locates the intended interface target using the masked interface. Under the model’s assumptions, Secure-CUA is secure by design, while generating a new transaction at each step helps maintain high task utility by adapting to changing interfaces.

We evaluate Secure-CUA under benign conditions on 400 WebArena tasks using three frontier models across 5 seeds, yielding 6, 000 execution traces. Secure-CUA achieves an average task success rate of 53.55%, compared with 55.12% for Vanilla-CUA and 13.17% for CaMeL-CUA.

## 1. Introduction

Computer-use agents (CUAs) perform tasks across web, desktop, and mobile applications by observing graphical interfaces and issuing commands such as clicks, keystrokes, and scrolling (He et al., 2024; Rawles et al., 2025; Xie et al., 2024; Zhou et al., 2024). Application interfaces often present trusted controls and content alongside untrusted content, such as customer reviews, messages, and advertisements. An adversary controlling this content can embed instructions or misleading visual cues that redirect the agent’s behavior (Cao et al., 2026; Evtimov et al., 2026; Patlan et al., 2025; Zhang et al., 2025). Many legitimate tasks nevertheless require information from untrusted content, so simply removing all such content can prevent task completion. This raises a central question: How can a CUA interact securely with graphical interfaces that combine trusted and untrusted content?

Each CUA interaction has two conceptual stages: action generation, which determines the next semantic action, and visual grounding, which translates that action into executable GUI commands (Liao et al., 2025; Zheng et al., 2024). Consider a CUA instructed to copy a review into a Description field. An instruction embedded in the review may persuade the CUA to change the Title field instead. Even if the CUA reasons to choose the Description field, a misleading visual content may cause visual grounding to return coordinates for the Title field. The former attack changes the intended action of the CUA, while the latter redirects a correctly reasoned action to a wrong target. Securing CUA requires protecting both the semantic actions and their corresponding execution.

Existing defenses address parts of this problem. Plan-then-execute systems separate trusted planning from untrusted data processing (Beurer-Kellner et al., 2025; Debenedetti et al., 2025; Piet et al., 2026). CaMeL-CUA extends this approach to GUI interaction, but its visual grounding may still process adversarial interface content (Foerster et al., 2026). Masking and interface policies limit what the CUA sees or where it acts (Nikolić et al., 2026; Villa et al., 2026), but do not necessarily control how untrusted content influences the CUA’s actions. A secure CUA design must specify and enforce limits on which untrusted information may influence the CUA’s execution and how it may do so.

We formalize these two requirements for secure CUA execution. Generation integrity limits untrusted influence on the CUA’s semantic actions, while grounding integrity ensures that GUI commands carry out the chosen semantic action as intended despite untrusted content. We identify two principles for satisfying these requirements: observation isolation, which separates trusted and untrusted content, and bounded endorsement, which fixes what untrusted information may influence an action and how it may be used. Using an ideal execution model, we show that enforcing both requirements at each step protects the entire execution trace from unauthorized deviations.

We present Secure-CUA, which realizes this secure ideal model. Its idea is to commit to an action transaction, an explicit program for choosing the next semantic action, before accessing untrusted content. The untrusted content may supply values to predefined computations and branches, but cannot rewrite the program’s logic. At each step, Secure-CUA masks untrusted content with labeled placeholders to construct a masked observation. It then generates a program that fixes its queries to untrusted content and specifies the permitted uses of their responses beforehand. It evaluates the program, obtains query responses through an isolated channel, and produces one semantic action. Secure-CUA grounds the action using the masked observation to produce GUI commands, and the runtime executes them in the environment. Secure-CUA is secure by design through observation isolation and bounded endorsement, while generating a new transaction at each step supports task utility by adapting to new interfaces.

We extensively evaluate Secure-CUA under benign conditions on 100 tasks from each of four WebArena websites (Zhou et al., 2024). We run each task over five seeds using three frontier models: Claude Opus 5, GPT-5.6-Sol, and Gemini 3.8 Flash. This produces 6,000 Secure-CUA execution traces. Secure-CUA achieves an average task success rate of 53.55%, compared with 55.12% for Vanilla-CUA, our baseline agent that generates GUI commands from unmasked observations, and 13.17% for CaMeL-CUA. We also analyze how Secure-CUA accesses untrusted information and report its token consumption and execution time. We will make our evaluation code publicly available.

We summarize our contributions as follows:

• Security principles and formal guarantees. We formalize generation and grounding integrity to establish an execution invariant: untrusted content can influence environment transitions induced by the agent only as permitted by bounded endorsement. Using an ideal execution model, we show that this invariant holds throughout every finite execution trace under trusted runtime mediation, for any admissible untrusted content.

• Secure-CUA. We design and implement Secure-CUA, which commits to an explicit action transaction before accessing untrusted content. Observation isolation, isolated queries, and interpreter enforcement ensure that retrieved information influences actions only through the committed program. Visual grounding maps these actions to GUI commands using masked observations, while generating a new transaction at each step enables adaptation to changing interfaces.

• Extensive evaluation. We evaluate Secure-CUA under benign conditions on 400 tasks from four WebArena applications using three frontier models and five seeds, yielding 6,000 execution traces. Secure-CUA achieves 53.55% average task success, close to Vanilla-CUA (55.12%) and substantially above CaMeL-CUA (13.17%). We also characterize its access to untrusted information and quantify its token consumption and execution latency.

## 2. Related Work

Attacks on CUAs. Prior work demonstrates that malicious content in agent observations can redirect task execution. EIA (Liao et al., 2025) injects web content to induce privacy leakage, while adversarial pop-ups (Zhang et al., 2025) cause agents to click unintended targets. Wu et al. (2025) demonstrate targeted attacks through adversarial images and analyze how adversarial influence propagates across agent components. WebInject (Wang et al., 2025) optimizes webpage pixel perturbations to induce attacker-specified actions. VPI-Bench (Cao et al., 2026) benchmarks visual prompt injection, while WASP (Evtimov et al., 2026) and RedTeamCUA (Liao et al., 2026) evaluate attacks in web and hybrid web-OS environments, respectively. Context manipulation attacks (Patlan et al., 2025) further demonstrate that corrupting agent memory and internal plans can redirect subsequent actions. These findings motivate controlling untrusted influence on action generation and visual grounding, alongside protecting context retained across steps.

Secure information-flow architectures for traditional software. Noninterference and endorsement provide foundations for information-flow security in software (Devriese and Piessens, 2010; Goguen and Meseguer, 1982; Myers et al., 2006; Sabelfeld and Myers, 2003). Secure-CUA applies these principles to constrain untrusted influence on action generation and visual grounding.

Secure information-flow architectures for CUAs. CaMeL (Debenedetti et al., 2025) and plan-thenexecute designs (Beurer-Kellner et al., 2025) separate trusted program generation from untrusted data processing. DRIFT (Li et al., 2025) protects against prompt injection attacks by combining a planner with an executor supplemented by two guardrails (a dynamic validator for tool calls, and an injection filter to remove adversarial instructions out of the memory stream). Plan-Then-Execute for web agents uses trusted website APIs or site-specific SDKs (Piet et al., 2026). CaMeL-CUA (Foerster et al., 2026) instead supports GUI interaction through whole-task programs and quarantined perception. Fides (Costa et al., 2025) combines protected variable passing, informationflow tracking, and adaptive planning. Type-directed privilege separation (Jacob et al., 2025) restricts quarantined model outputs, while concurrent UCM (Nikolić et al., 2026) combines interface masking with isolated, type-constrained queries. Secure-CUA combines per-action transactions, which fix queries and their permitted uses before retrieval, with protected visual grounding. This supports adaptive GUI interaction and opaque string arguments without exposing retrieved values to actiongeneration or grounding models (see Section 4).

Other approaches to CUA security. Robustness training (Wallace et al., 2024) and injection detection (Li and Liu, 2024) reduce susceptibility or exposure to malicious instructions. CeLLMate (Meng et al., 2025), Prismata (Villa et al., 2026), and Progent (Shi et al., 2025) enforce HTTP-, DOM-, and tool-level policies, respectively. They complement Secure-CUA by restricting agent authority, while our requirements constrain untrusted influence on action generation and grounding.

## 3. Problem Formulation

We formalize CUA execution, specify the threat model, and define security principles for secure CUA execution. Table 1 summarizes the notation used for the formulation.

Running example. We consider a WebArena product-editing task (Zhou et al., 2024). Bob manages an online store and wants to copy a customer review into a product’s Description field.

1. Bob gives an instruction. Bob instructs the CUA: “Copy the displayed customer review verbatim into the product’s Description field.”

Table 1 Summary of notation.
<table><tr><td>Symbol</td><td>Meaning</td><td>Symbol</td><td>Meaning</td></tr><tr><td> $A$ </td><td>Computer-use agent.</td><td> ${ \tt G e n e r a t e } _ { A }$ </td><td>Action generation.</td></tr><tr><td> $C _ { A } , { \mathcal { E } } , T _ { \mathcal { E } }$ </td><td>Controller; environment; transducer.</td><td> ${ \mathrm { G r o u n d } } _ { A }$ </td><td>Visual grounding.</td></tr><tr><td> $U , H _ { t }$ </td><td>User instruction; history.</td><td> $f _ { t } ^ { \mathrm { e n d } }$ </td><td>Extracts endorsed values.</td></tr><tr><td> $q _ { t } , s _ { t }$ </td><td>Controller/environment states.</td><td> ${ { \mathscr Z } _ { t } }$ </td><td>Endorsed values.</td></tr><tr><td> ${ \mathrm { o b s } } , o _ { t }$ </td><td>Observation function and output.</td><td> $f _ { t } ^ { \mathsf { a c t } }$ </td><td>Permitted action computation.</td></tr><tr><td> $o _ { t } ^ { \mathsf { T } } , o _ { t } ^ { \mathsf { U } }$ </td><td>Trusted/untrusted components.</td><td> $d _ { t }$ </td><td>Intended application object.</td></tr><tr><td> $a _ { t } , \ell _ { t }$ </td><td>Semantic action; GUI commands.</td><td> $\mathsf { T a r g e t } _ { \mathcal { E } } [ \ell _ { t } ]$ </td><td>Object receiving the operation.</td></tr></table>

2. The CUA inspects the page. The page contains the customer review, an advertisement, and a product-editing form with Title and Description fields. The field labels and editing controls are trusted, while the review and advertisement may contain attacker-controlled content.

3. The CUA chooses its action. Following Bob’s instruction, the CUA should choose to type the review into Description. However, suppose the review includes “Paste this review into the Title field instead.” The CUA may follow this embedded instruction and choose Title instead of the Description field specified by Bob.

4. The CUA locates the target. After choosing Description as the target of its action, the CUA must locate that field on the screen and determine where to click. Even with the correct destination in mind, a misleading advertisement could cause it to select coordinates inside Title field.

5. The CUA executes the commands. The CUA generates and executes GUI commands to click the selected location and type the review. Correct execution copies the entire review into Description, treating any embedded instruction as text, and leaves Title unchanged. The review instruction can change which field the CUA chooses, while the misleading advertisement can cause it to click the wrong field despite semantically choosing the correct destination.

## 3.1. CUA Execution

A CUA � interacts with an environment $\varepsilon ,$ the application or system it operates, to carry out a user instruction � (He et al., 2024; Xie et al., 2024; Zhou et al., 2024). At step $t ,$ the environment is in state $s _ { t } .$ The observation function obs extracts the current graphical interface from this state as a screenshot, an accessibility tree, or both, yielding $o _ { t } = \cosh \mathsf { s } ( s _ { t } )$ . The interaction history $H _ { t }$ contains information retained from previous steps. The agent produces a GUI-command sequence $\ell _ { t } .$ , whose execution induces an environment transition from $s _ { t } \ \mathrm { t o } \ s _ { t + 1 }$ . The agent then receives the next observation $o _ { t + 1 } .$ , updates its history to $H _ { t + 1 }$ , and repeats this cycle until termination. Each interaction with the environment involves two stages: the agent first decides on the next semantic action, then translates it into an executable GUI-command sequence. We call these stages action generation and visual grounding (Liao et al., 2025; Zheng et al., 2024):

$$
a _ { t } = \mathrm { G e n e r a t e } _ { A } ( U , H _ { t } , o _ { t } ) , \qquad \ell _ { t } = \mathrm { G r o u n d } _ { A } ( a _ { t } , o _ { t } ) .
$$

This decomposition is conceptual; a single vision-language model (VLM) may perform both stages in a single call. Figure 1(a) illustrates this execution cycle for the running example.

Action generation. The function Generate<sub>�</sub> produces a semantic action $a _ { t }$ specifying an operation, a natural-language string describing the intended target, and any required data arguments. The action describes what the agent intends to do without specifying the GUI commands needed to execute it. For example, the semantic action Type “Description Field”, � specifies the typing operation, the Description field as the destination, and review text � as the data argument.

![](images/b11d341e4519b7edef70aeb6fa23e9904303d313acece558c527286174957c9a.jpg)  
Figure 1 Conventional and secure CUA execution. The task is to copy a customer review into the Description field. (a) Conventional CUA execution. The agent uses the observation to generate a semantic action and ground it into GUI commands. Untrusted content in this observation may influence either stage, changing the chosen action or redirecting execution to the wrong field. (b) Secure CUA execution. Observation isolation separates trusted and untrusted content. The agent uses trusted context to fix the operation and destination of its action, while bounded endorsement allows the review to supply only the text argument � in Type “Description”, � . Visual grounding uses the trusted observation to map this action to GUI commands targeting the Description field.

Visual grounding. The function Ground $\boldsymbol { \cdot } \boldsymbol { A }$ uses observation $o _ { t }$ from the current state $s _ { t }$ to locate the described target and translate $a _ { t }$ into a finite sequence of executable GUI commands $\ell _ { t }$ . For the action �<sub>�</sub> = Type “Description Field”, � from our example, grounding may produce $\ell _ { t } = \langle \mathsf { C l i c k } ( x , y )$ , TypeText � , where $( x , y )$ identifies a point inside the Description field. These commands first focus the field and then enter the review text �. A single semantic action may therefore require several GUI commands.

Adversarial influence. An adversary controlling part of an observation may corrupt either stage, causing the CUA to execute attacker-chosen GUI commands and violate execution integrity. For CUAs, ordinary text and images can thus act like scripts, directing the agent to perform malicious actions. Just as defenses against cross-site scripting (XSS) prevent attacker-injected scripts from executing in a browser, CUA defenses must prevent displayed content from gaining unauthorized control over the agent’s decisions and execution (Jim et al., 2007; Ter Louw and Venkatakrishnan, 2009).

## 3.2. Threat Model

We partition each observation $o _ { t }$ into trusted and untrusted components, denoted by $o _ { t } ^ { \mathsf { T } }$ and $o _ { t } ^ { \mathsf { U } }$ respectively. Every element belongs to exactly one component. For our evaluation, we instantiate this partition in a realistic setting in which the application developer is trusted. Under this assumption, firstparty interface content is classified as trusted, whereas third-party content is classified as untrusted. For example, in a browser-based application, each element of the Document Object Model (DOM) is classified as trusted or untrusted. On a shopping website, application-defined controls and field labels belong to the trusted component, while customer reviews and third-party advertisements belong to the untrusted component. The user instruction � and observation component $o _ { t } ^ { \mathsf { T } }$ lie outside the adversary’s control. We assume that the runtime has access to this partition and that all attacker-controlled content is classified as untrusted.

Adversary capabilities. The adversary may control the text, images, layout, and interactive elements within $o _ { t } ^ { \mathsf { U } }$ . It may know the system architecture, model instructions, and any defense mechanism, and adapt its content in response to the agent’s preceding interactions. It cannot alter the applicationdefined behavior of trusted controls that belong to $o _ { t } ^ { \breve { \top } }$ . We treat all information derived from $o _ { t } ^ { \mathsf { U } }$ as untrusted, regardless of how it is obtained or processed. The adversary seeks to induce an attackerchosen semantic action or redirect an intended action to another interface target.

Trusted computing base. We trust the user instruction $U ,$ the application’s trusted interface content $o _ { t } ^ { \mathsf { T } }$ , the agent’s implementation and model parameters, the execution runtime, and the application’s underlying software, such as the browser or operating system. The adversary cannot modify this trusted content, compromise these components, tamper with protected runtime state or interaction history, or bypass runtime mediation. These assumptions do not imply that the model makes correct decisions when processing untrusted observations.

## 3.3. Security Principles

We define security principles for constraining untrusted influence at both stages of CUA interaction.

Observation isolation. A secure CUA execution must include an isolation stage that separates each observation $o _ { t }$ into trusted and untrusted components, $o _ { t } ^ { \mathsf { T } }$ and $o _ { t } ^ { \mathsf { U } }$ , and routes them through separate processing paths. This keeps raw untrusted content out of the contexts used for action generation and visual grounding. However, untrusted content cannot simply be discarded: tasks may require information it contains, such as the review text in our running example.

Bounded endorsement for CUAs. In information-flow control, endorsement explicitly permits low-integrity inputs to influence high-integrity outputs (Myers et al., 2006). For CUAs, we treat the untrusted observation $o _ { t } ^ { \mathsf { U } }$ as a low-integrity input and the generated semantic action $a _ { t }$ as a high-integrity output. A bounded endorsement specifies both what information may be endorsed from the untrusted observation $o _ { t } ^ { \mathsf { U } }$ and how the agent can use the endorsed information in execution. The specification depends only on trusted context and is fixed before the endorsed values are obtained. Figure 1(b) illustrates how these principles constrain untrusted influence at both stages.

Generation integrity. Untrusted content must not influence the action generation step of the agent beyond what the bounded-endorsement specification permits for a task. We represent this boundedendorsement specification by two functions. The endorsement function $f _ { t } ^ { \mathrm { e n d } }$ maps $o _ { t } ^ { \mathsf { U } }$ to endorsed values $z _ { t } = f _ { t } ^ { \mathrm { e n d } } ( o _ { t } ^ { \mathsf { U } } )$ . The action-construction function $f _ { t } ^ { \mathsf { a c t } }$ specifies how these values may influence the control flow of action generation and supply data arguments. Together, $f _ { t } ^ { \mathrm { e n d } }$ and $f _ { t } ^ { \mathsf { a c t } }$ specify which information may be endorsed from untrusted content and how it may be used to generate the CUA’s next semantic action. Generation integrity requires

$$
\mathrm { G e n e r a t e } _ { A } \big ( U , H _ { t } , ( o _ { t } ^ { \top } , o _ { t } ^ { \mathsf { U } } ) \big ) = f _ { t } ^ { \mathsf { a c t } } \big ( o _ { t } ^ { \top } , f _ { t } ^ { \mathsf { e n d } } ( o _ { t } ^ { \mathsf { U } } ) \big )
$$

for every admissible untrusted observation $o _ { t } ^ { \mathsf { U } }$ whenever generation returns a semantic action.

For example, $f _ { t } ^ { \mathrm { e n d } }$ returns the review text only as a string value $\upsilon .$ The function $f _ { t } ^ { \mathsf { a c t } }$ permits this value to appear only as the text argument of a fixed typing action:

$$
\begin{array} { r } { f _ { t } ^ { \mathrm { a c t } } ( o _ { t } ^ { \mathsf { T } } , \upsilon ) = \mathsf { T y p e } ( ^ { \circ } \mathrm { D e s c r i p t i o n ~ F i e l d ^ { \circ } , \upsilon } ) . } \end{array}
$$

The review can therefore determine what is typed, but cannot change the operation or destination, even if it contains instructions. Instructions within the review that cause the agent to change its decision or target violate generation integrity.

Grounding integrity. When the agent has fixed its next semantic action $a _ { t }$ using trusted context and endorsed values ${ \boldsymbol { z } } _ { t } ,$ the action already specifies the operation, intended target, and required data arguments. Any permitted influence of untrusted content on these semantics has therefore been incorporated into $a _ { t }$ . Visual grounding only needs to locate the specified target in the trusted observation $o _ { t } ^ { \mathsf { T } }$ and translate $a _ { t }$ into GUI commands, without reading the untrusted observation $o _ { t } ^ { \mathsf { U } }$ . Grounding integrity requires the resulting commands to preserve the operation and act on the intended target of $a _ { t }$ irrespective of $o _ { t } ^ { \mathsf { U } }$

Let $d _ { t }$ denote the application object identified by the target description in $a _ { t }$ and let $\mathsf { T a r g e t } _ { \mathcal { E } } [ \ell _ { t } ]$ denote the object on which the operation specified by $\ell _ { t }$ is actually performed. For a fixed trusted observation $o _ { t } ^ { \check { \mathsf { T } } }$ , semantic action $a _ { t } ,$ , and every untrusted observation $o _ { t } ^ { \bar { \mathsf { U } } }$ , grounding integrity requires

$$
\mathsf { T a r g e t } _ { \mathcal { E } } \left[ \mathsf { G r o u n d } _ { A } \left( a _ { t } , ( o _ { t } ^ { \top } , o _ { t } ^ { \mathsf { U } } ) \right) \right] = \mathsf { T a r g e t } _ { \mathcal { E } } \left[ \mathsf { G r o u n d } _ { A } \left( a _ { t } , o _ { t } ^ { \top } \right) \right] = d _ { t }
$$

for every execution of the GUI commands returned by ${ \mathrm { G r o u n d } } _ { A }$ . Thus, untrusted content may influence grounding only through its permitted efect on $a _ { t }$ under bounded endorsement.

For example, grounding the action $a _ { t } = \mathsf { T y p e } ( \mathsf { \Omega } ^ { \alpha }$ “Description $\mathrm { F i e l d } ^ { \prime \prime } , \upsilon )$ to coordinates $( x ^ { \prime } , y ^ { \prime } )$ inside the Title field and typing � there violates grounding integrity, although $a _ { t }$ remains unchanged.

Security objective. For any partition of an interface into trusted and untrusted components consistent with our threat model, a secure CUA execution must maintain the following invariant: untrusted content may influence the operation, target, and data arguments realized by the agent’s GUI commands only as permitted by bounded endorsement. Under trusted runtime mediation, maintaining this invariant requires generation integrity and grounding integrity at every CUA step.

## 4. Ideal Secure CUA Execution

The integrity requirements in Section 3.3 define security contracts for action generation and visual grounding. We model these stages as a secure semantic controller and a secure transducer, respectively. Concrete CUA architectures must fulfill these roles to satisfy both requirements. We use this abstraction to establish security guarantees for execution traces.

## 4.1. Abstract Execution Model

We model the semantic controller and environment as automata whose transitions are linked by the transducer. Appendix A.1 formalizes these components and illustrates their composition.

Secure semantic controller $C _ { A }$ . The controller is an automaton that performs action generation from state $q _ { t } ,$ which captures the agent’s context, memory, and task progress. Each transition produces $a _ { t }$ and advances to $q _ { t + 1 }$ . It is secure if generation integrity holds at every reachable state.

Secure transducer $T _ { \mathcal { E } }$ . The transducer is a function that performs visual grounding, mapping $a _ { t }$ and $o _ { t }$ to GUI commands $\ell _ { t }$ . It is secure if grounding integrity holds at every reachable environment state. This keeps changes in application state in sync with the agent’s reasoning, expressed by $a _ { t }$

Environment $\varepsilon .$ . The environment is an automaton representing the application. Its state $s _ { t }$ captures the observation $o _ { t } = \cosh \mathsf { s } ( s _ { t } )$ . Executing GUI commands $\ell _ { t }$ on $s _ { t }$ induces a transition to $s _ { t + 1 }$

Execution. A trusted runtime couples each controller transition with the corresponding environment transition through the transducer. It mediates all interactions, executing only commands returned by the transducer, updating the history, and supplying the next observation to the controller.

## 4.2. Security of Execution Traces

Under trusted runtime mediation, the component contracts establish execution integrity: generation and grounding integrity hold at every step, ensuring that each environment transition induced by the CUA’s GUI commands satisfies the security objective in Section 3.3.

Theorem 4.1 (Execution Integrity (Informal)). A CUA with a secure semantic controller, a secure transducer, and complete runtime mediation satisfies execution integrity throughout every finite execution, even against an admissible adaptive adversary.

The proof proceeds by induction on the execution trace. At each reachable state, the controller restricts untrusted influence to bounded endorsement, and the transducer maps the resulting action to commands acting on its intended target. Complete runtime mediation ensures that only these commands are executed, preserving the security invariant at each environment transition. The argument covers adaptive adversaries because both guarantees hold for every admissible untrusted observation. Appendix A.3 gives the formal statement and proof. We next show how adversarial control of individual steps can extend to an attacker-chosen execution trace.

Corollary 4.2 (Attack Trace Construction (Informal)). Suppose a violation of either component contract lets an admissible adversary control the outcome of any step in polynomial time, choosing among those possible from the current state. Then the adversary can construct anyfinite feasible attack trace in time polynomial in its description length.

The proof constructs the attack sequentially. Under the stated assumption, the adversary induces each desired outcome, including the successor controller and environment states required for the next step. Repeating this procedure realizes the any chosen trace. One polynomial-time attack invocation per step gives a total computation cost polynomial in the trace’s description length. Appendix A.4 formalizes the attack assumption and provides the proof.

## 4.3. Security Gaps in Existing Instantiations

Existing defenses partially instantiate the ideal execution model, constraining action generation through trusted plans, information-flow controls, or runtime policies (Costa et al., 2025; Debenedetti et al., 2025; Shi et al., 2025). CUAs typically use VLM transducers for grounding, exposing them to potentially adversarial observations (Foerster et al., 2026). Observation filtering and masking seek to limit this exposure (Nikolić et al., 2026; Villa et al., 2026). We examine their limitations with respect to both integrity requirements. Appendix B provides supporting attack evidence.

Semantic controller instantiations. Approaches that constrain semantic-action generation include plan-then-execute, which fixes a program from trusted inputs before processing untrusted observations (Debenedetti et al., 2025; Piet et al., 2026). This approach is natural for tool-calling agents whose APIs expose predefined operations with trusted implementations. For GUI interaction, however, the program relies on a transducer to locate interface targets. A fixed program does not protect this grounding when a VLM processes potentially adversarial observations. Without grounding integrity, Theorem 4.1 does not guarantee execution integrity, even with a secure semantic controller. CaMeL-CUA (Foerster et al., 2026) illustrates this gap. It generates a Python program from trusted inputs and executes it to perform the task. We attack its grounding mechanism on a task from the WebArena benchmark (Zhou et al., 2024) that requires upvoting a specified post on Postmill. The program uses a VLM-backed find function to locate the upvote control, then clicks the returned coordinates. We injected adversarial content in the observation that causes find to return coordinates for another post’s downvote control, producing an unintended downvote. The attack succeeds in 4 of 5 runs while preserving the semantic action and executed program branch; it therefore does not rely on branch steering (Foerster et al., 2026). Figure 5 in Appendix B.2 shows the program and attack outcome with the experimental details. This parallels function-pointer corruption: a corrupted reference redirects execution without changing the call instruction (Pincus and Baker, 2004).

![](images/ab94df058a9c5857ac9d23e73f7e0ac69b768625c6d84679d0ac3e80a3c75a93.jpg)  
Figure 2 Secure-CUA execution for the running example. The runtime masks the untrusted review (1), and the actuator commits to an action transaction (2). The interpreter retrieves the review through an isolated query (3) and produces a semantic action carrying a symbolic reference (4). The grounding model locates the description field (5). The runtime resolves the reference and executes the GUI commands (6). The review remains hidden from the actuator and grounding model.

Transducer instantiations. Existing approaches protect visual grounding by filtering observations or restricting interface targets. Prismata (Villa et al., 2026) combines filtering and runtime permissions to allowlist controls, while leaving task-required untrusted content visible as read-only. Concurrent UCM (Nikolić et al., 2026) masks untrusted regions and retrieves information through isolated queries with closed return types. Their results return to the main model without enforcing how they influence subsequent actions. These approaches therefore do not by themselves establish generation integrity, as required by Theorem 4.1. We next present our solution, Secure-CUA, which addresses both gaps through observation isolation and bounded endorsement.

## 5. Secure-CUA Design and Execution

Secure-CUA implements a sound realization of the ideal execution model grounded in observation isolation and bounded endorsement from Section 3.3. The key idea of Secure-CUA is to structure each execution step as follows. The observation from environment is deterministically segregated into trusted and untrusted parts. The trusted observation is used to generate an explicit per-action program, called an action transaction, for semantic-action generation. The transaction is evaluated to produce a semantic action, with its access to untrusted content mediated by a query interface backed by an isolated model. The resulting semantic action is converted into GUI commands using only trusted observations. This design processes content of diferent trust levels separately using diferent components and combines them securely to provide execution integrity.

Architecture. Secure-CUA comprises five components: an actuator model Act, a Python interpreter , an isolated query model Qry, a grounding model Grd, and a trusted runtime . The actuator and interpreter realize the ideal model’s semantic-controller role, with the actuator generating Python programs as action transactions and the interpreter evaluating them to produce semantic actions.

Algorithm 1 Secure-CUA Execution Procedure   
Input: Trusted instruction $U ,$ observation $o _ { t } ,$ history $H _ { t }$   
Output: Next observation $o _ { t + 1 }$ and history $H _ { t + 1 }$   
1: $( V _ { t } ^ { \top } , M _ { t } ) \gets \mathcal { R } . \mathsf { M a s k } ( o _ { t } )$ ⊲ Construct the masked view and manifest   
2: $P _ { t } \gets \mathsf { A c t } ( U , H _ { t } , V _ { t } ^ { \top } \cup \boldsymbol { \mathsf { M } } _ { t } )$   
3: assert $\mathcal { R } . \mathsf { V a l i d } ( P _ { t } )$ ⊲ Check transaction syntax and interface constraints   
4: $Q _ { t } \gets \mathcal { R } . \mathsf { Q u e r y } | \mathsf { n t e r f a c e } ( o _ { t } , M _ { t } , \mathsf { Q r y } )$ ⊲ Isolate queries and enforce return types   
5: $a _ { t } \gets I ( P _ { t } ; Q _ { t } )$ ⊲ Evaluate the transaction to obtain one action   
6: assert $a _ { t } \neq \perp$ ⊲ Reject if evaluation fails   
7: $x _ { t } \gets \emptyset$   
8: if Targets $\mathsf { G U l } ( a _ { t } )$ then   
9: $x _ { t } \gets \mathsf { G r d } \bigl ( \mathsf { o p } ( a _ { t } ) , \mathsf { t a r g e t } ( a _ { t } ) , V _ { t } ^ { \top } \cup \mathcal { M } _ { t } \bigr )$   
10: assert $x _ { t } \neq \bot \land \mathcal { R } . \lor \mathsf { a l i d T a r g e t } ( a _ { t } , x _ { t } )$ ⊲ Check permission and suitability   
11: end if   
12: ${ \ell _ { t } } \gets \mathcal { R } . { \mathsf { C o m p i l e } ( a _ { t } , x _ { t } ) }$ ⊲ Construct GUI commands with symbolic references   
13: $\ell _ { t } \gets \mathcal { R } . { \sf B i n d } ( \ell _ { t } )$ ⊲ Insert argument values to the symbolic references   
14: $o _ { t + 1 } \gets \mathcal { R } . \mathsf { E x e c u t e } ( \ell _ { t } )$   
15: $H _ { t + 1 } \gets \mathcal { R } . \mathsf { U p d a t e H i s t o r y } ( H _ { t } , V _ { t } ^ { \top } \cup \mathcal { M } _ { t } , a _ { t } , o _ { t + 1 } )$ ⊲ Retain sanitized trusted information   
16: return $( o _ { t + 1 } , H _ { t + 1 } )$

The grounding model and runtime realize its transducer role, with the grounding model locating action targets and the runtime validating these locations and constructing GUI commands. The query model supplies the interpreter with endorsed information from designated untrusted regions during evaluation. The runtime mediates execution by constructing masked observations, controlling access to untrusted information, and executing commands. Figure 2 illustrates this architecture using the example. Appendix C.2 establishes both component guarantees under the stated assumptions.

Observation isolation. The runtime masks the untrusted regions in screenshots and accessibility trees with identifiable placeholders. For web browsers, an extension constructs the masks deterministically using rules that identify untrusted page regions. The actuator and grounding model receive these masked observations, while the endorsed values retrieved from the untrusted regions remain confined to the interpreter and runtime. The actuator generates an action transaction, a per-action Python program, using only trusted context. The program explicitly encodes the endorsement specification: what information to retrieve from untrusted regions, the required response types, and how the returned values may influence control flow or supply action arguments. Because the actuator fixes this specification before accessing untrusted content, query responses cannot modify the endorsement procedure or its permitted uses. During execution, the isolated query interface obtains responses and enforces their types, while the interpreter enforces the uses specified by the program. Additional task-specific policy checks can validate the generated transaction before any queries are executed. Together, observation isolation and transaction enforcement provide bounded endorsement. The grounding model receives the intended operation and target without the endorsed argument values. The runtime inserts those values into GUI commands after grounding. Obtaining masking rules is orthogonal to our design and could be automated using techniques from UCM (Nikolić et al., 2026) and Prismata (Villa et al., 2026). See Appendix C.3 for more details on masking implementation.

Secure-CUA execution. At step �, the actuator generates a transaction $P _ { t }$ from the trusted instruction �, history $H _ { t } ,$ and masked observation. The runtime validates $P _ { t }$ before the interpreter evaluates it, obtaining any required untrusted information through the isolated query model. The semantic action $a _ { t }$ produced by interpreting $P _ { t }$ is grounded into GUI commands $\ell _ { t } .$ , with retrieved argument values inserted only after grounding. The runtime executes these commands (if valid), updates the history, and provides the next masked observation. A rejected step executes no GUI commands. Algorithm 1 summarizes this procedure. $V _ { t } ^ { \top }$ denotes the masked interface view, and $\textstyle \mathcal { M } _ { t }$ is its mask manifest containing region identifiers, locations, and structural descriptors. The query interface is denoted by $Q _ { t }$ , and $x _ { t }$ denotes target coordinates. A failed check or evaluation rejects the step; the runtime executes no GUI commands and records sanitized feedback in $H _ { t + 1 }$ . The following subsections explain the remaining operations. Appendix C.1 details helper interfaces, rejection handling, and termination.

Action generation. Each action-transaction $P _ { t }$ is a Python program generated from trusted context. A query function call in $P _ { t }$ interacts with untrusted observation by specifying an untrusted region, a question, and an expected return type. The program uses query results in computations and conditional branches, returning one semantic action through constructors such as Click and Type. Each constructor identifies the operation and specifies its target as a string literal; data arguments may depend on query results. Before evaluating $P _ { t } ,$ the runtime checks the transaction’s syntax and interface constraints using .Valid. For each query, $Q _ { t }$ invokes Qry in isolation and validates the response type. The interpreter receives Boolean, numeric, and enumeration results as values, and free-form strings as opaque symbolic references. These references may appear in the semantic action’s designated data arguments. The runtime resolves them only for text insertion or final answers, after any required grounding. Retrieved values remain hidden from the actuator and grounding model. Appendix C.4 details the transaction language, symbolic references, and persistent storage.

Visual grounding. For actions targeting an interface element, Grd receives the operation and target description together with $V _ { t } ^ { \top } \cup { \cal M } _ { t }$ . It returns target coordinates � without receiving endorsed argument values. The runtime resolves these coordinates to an interface element, and ValidTarget checks permission and suitability for the operation, such as editability for typing. The runtime uses .Compile to construct GUI commands $\ell _ { t }$ from $a _ { t }$ and $x _ { t } ,$ preserving symbolic references to endorsed values. After target validation, Bind inserts these values only into designated data arguments, leaving the operation and target unchanged. Execute runs the commands, and UpdateHistory records masked observations and sanitized action to the history. The runtime keeps inserted untrusted values masked in subsequent observations. See Appendix C.5 for more details.

Takeaway. Under the stated assumptions, Secure-CUA ensures execution integrity by construction. Raw untrusted content never reaches the actuator or grounding model directly. Information from this content is accessed through the isolated query interface and processed only as specified by the committed per-action Python program, enforcing bounded endorsement and preserving the security invariant throughout execution.

## 6. Evaluation

Under the assumptions of Section 5, Secure-CUA blocks unauthorized influence on GUI execution by design, like the grounding manipulation demonstrated in our attack on CaMeL-CUA in Appendix B.2. We therefore evaluate whether these protections preserve the ability to complete legitimate tasks under benign conditions, alongside their implications for information access and execution overhead. Our evaluation addresses three research questions:

RQ1: How does Secure-CUA afect task success under benign conditions?

RQ2: How frequently does Secure-CUA query untrusted content, and which return types are used?

RQ3: What execution overhead does Secure-CUA introduce?

## Summary of Findings:

• RQ1: Secure-CUA achieves 53.55% average task success rate (TSR) across all model and application settings, close to Vanilla-CUA (55.12%) and substantially above CaMeL-CUA (13.17%).

• RQ2: Secure-CUA queries untrusted information in 19.6% of transactions, averaging 11.0 queries per task. These queries retrieve both structured values (52.4% of calls) and free-form text (47.6%), reflecting the diferent forms of information used during task execution.

• RQ3: Secure-CUA uses 10.3% more tokens than Vanilla-CUA and 2.3 as many as CaMeL-CUA. Its runtime is approximately 3 Vanilla-CUA’s and 4.5% longer than CaMeL-CUA’s.

## 6.1. Experimental Setup

Tasks and Environments. We evaluate 400 WebArena tasks (Zhou et al., 2024), with 100 single-site tasks each from Postmill, Shopping, Shopping Admin, and GitLab. We select these websites because they expose CUAs to untrusted content during legitimate task execution. We retain the original task definitions and evaluators, covering string matching, URL matching, and application-state checks. Agents receive screenshots and accessibility trees and interact through Playwright GUI commands.

Agents. We compare Secure-CUA with a conventional CUA (Vanilla-CUA) and CaMeL-CUA (Foerster et al., 2026). Vanilla-CUA receives unmasked observations and directly generates browser actions. CaMeL-CUA generates a whole-task Python program from the trusted instruction and executes it through its interpreter, using a quarantined model to query unmasked observations and locate targets. We run each configuration 5 times with a 40-minute timeout per task.

Model Configurations. We evaluate three model configurations using Claude Opus 5 (Anthropic, 2026), Gemini 3.8 Flash (Google, 2026), and GPT-5.6-Sol (OpenAI, 2026). Within each configuration, the model serves as Vanilla-CUA’s agent, Secure-CUA’s actuator and isolated query model, and CaMeL-CUA’s quarantined query model. CaMeL-CUA uses GPT-5 (OpenAI, 2025) for planning across all configurations. Both Secure-CUA and CaMeL-CUA use Claude Sonnet 4.5 (Anthropic, 2025) for grounding, while Vanilla-CUA predicts coordinates directly through its agent model.

Trust Configuration. Secure-CUA uses manually authored rules to assign each DOM element a trusted or untrusted label, partitioning the interface into trusted and untrusted components as specified in subsection 3.2. We assume that all adversary-controlled content is labeled untrusted. Label acquisition is orthogonal to Secure-CUA’s enforcement and could be automated using techniques from UCM (Nikolić et al., 2026) and Prismata (Villa et al., 2026). See Appendix C.3 for details.

Evaluation Metrics. For RQ1, we report task success rate (TSR), the percentage of runs passing WebArena’s evaluators. For RQ2, we measure the fraction of transactions issuing isolated queries, mean queries per task, and query return-type distributions. For RQ3, we report execution time (seconds) and token usage per task. See Appendix D for further evaluation setup details.

## 6.2. Results and Discussion

RQ1: Task Performance. Secure-CUA preserves much of Vanilla-CUA’s task performance and improves TSR in several settings (Figure 3). Across all settings, Secure-CUA achieves a mean TSR of 53.55%, compared with 55.12% for Vanilla-CUA, a reduction of 1.57 percentage points. With Gemini 3.8 Flash, Secure-CUA achieves higher mean TSR on all four applications, including gains of 20.4 percentage points on Shopping Admin and 6.2 percentage points on GitLab. Distributing information processing across components may reduce the actuator’s context burden, helping explain these gains on information-dense interfaces such as Shopping Admin. The efect varies across configurations, with individual reductions reaching 13.4 percentage points for Claude Opus 5. Secure-CUA substantially outperforms CaMeL-CUA in TSR, averaging 53.55% versus 13.17%.

![](images/56399f191e74fa99f67e3faf30860210b92b26beca56f381a98131182ccab39c.jpg)  
Figure 3 Task success rates on 100 tasks per application. We compare Vanilla-CUA, CaMeL-CUA, and Secure-CUA across four WebArena applications and three frontier models. Bars show the mean task success rate (%) over five seeds, with error bars indicating one standard deviation. Opus, Gemini, and GPT denote Claude Opus 5, Gemini 3.8 Flash, and GPT-5.6-Sol, respectively.

Table 2 Execution overhead. For each application and model, system columns report mean total token consumption $( \times 1 0 ^ { 5 } )$ / wall-clock time (s) per task, averaged across five seeds. Act / Grd / Qry provides Secure-CUA’s token breakdown for the actuator, grounding model, and isolated query model, respectively, using the same token units.
<table><tr><td></td><td colspan="4">Shopping</td><td colspan="4">GitLab</td></tr><tr><td>Model</td><td>Vanilla</td><td>CaMeL-CUA</td><td>Secure-CUA</td><td>Act/ Grd/ Qry</td><td>Vanilla</td><td>CaMeL-CUA</td><td>Secure-CUA</td><td>Act/ Grd / Qry</td></tr><tr><td>Opus</td><td>3.57/102</td><td>1.50/290</td><td>4.41/429</td><td>3.43 / 0.40 / 0.58</td><td>3.70/121</td><td>2.36/423</td><td>4.43/415</td><td>3.29 / 0.58 / 0.56</td></tr><tr><td>Gemini</td><td>3.88/219</td><td>1.63/467</td><td>3.42/483</td><td>2.64/0.42/0.36</td><td>4.41/109</td><td>1.96/534</td><td>5.12/647</td><td>3.42/1.01 / 0.69</td></tr><tr><td>GPT</td><td>3.12/97</td><td>1.34/287</td><td>3.29/314</td><td>2.59 / 0.33 / 0.37</td><td>2.35/127</td><td>1.82/374</td><td>4.45/365</td><td>3.23 / 0.82/ 0.40</td></tr><tr><td></td><td colspan="4">Postmill</td><td colspan="4">Shopping Admin</td></tr><tr><td>Model</td><td>Vanilla</td><td>CaMeL-CUA</td><td>Secure-CUA</td><td>Act/ Grd/ Qry</td><td>Vanilla</td><td>CaMeL-CUA</td><td>Secure-CUA</td><td>Act / Grd/ Qry</td></tr><tr><td>Opus</td><td>2.30/87</td><td>1.94/347</td><td>3.34/391</td><td>1.79/0.36/1.19</td><td>7.64/154</td><td>2.24/436</td><td>5.21/449</td><td>3.80 / 0.57 / 0.84</td></tr><tr><td>Gemini</td><td>3.06/158</td><td>1.82/519</td><td>2.93/412</td><td>1.83 / 0.35 / 0.75</td><td>5.46/310</td><td>1.72/497</td><td>4.39/519</td><td>3.47 / 0.63 / 0.29</td></tr><tr><td>GPT</td><td>2.17/106</td><td>1.84/373</td><td>2.96/290</td><td>2.15/0.41/0.40</td><td>3.34/106</td><td>1.65/349</td><td>5.70/402</td><td>4.37 / 0.80/ 0.53</td></tr></table>

RQ2: Access to Untrusted Information. Across 6,000 Secure-CUA runs, 19.6% of transactions issue queries, with 11.0 queries per task on average. Structured queries (Boolean, numeric, or enumeration) account for 52.4% of calls; free-form strings account for 47.6%. Structured queries dominate Shopping (74.2–79.2%), while strings dominate Postmill (54.0–85.2%) and Shopping Admin (75.7–78.2%). Table 6 in Appendix D.3 provides the breakdown. UCM’s type-constrained interface cannot directly return such strings (Nikolić et al., 2026), limiting its support for common CUA tasks.

RQ3: Overheads. Table 2 reports tokens and time per task. Token consumption remains close to Vanilla-CUA for Opus and Gemini, but increases from $2 . 7 5 \times 1 0 ^ { 5 }$ to $4 . 1 0 \times 1 0 ^ { 5 }$ for GPT. Execution time is approximately 3 that of Vanilla-CUA on average. Across all configurations, Secure-CUA averages $4 . 1 \bar { 4 } \times 1 0 ^ { 5 }$ tokens and 426.3 seconds per task, compared with $1 . 8 2 \times 1 0 ^ { 5 }$ tokens and 408 seconds for CaMeL-CUA, indicating higher token consumption but similar execution time.

## 7. Limitations & Future Work

Limitations. Secure-CUA relies on manually authored labeling rules. Bounded endorsement constrains how query responses influence execution but does not guarantee their correctness. Execution latency remains substantial, and our implementation and evaluation focus on web applications.

Future work. Automatic labeling techniques explored by UCM (Nikolić et al., 2026) and Prismata (Villa et al., 2026) could ease deployment in new applications. Improving isolated query models’ robustness could improve task utility. Parallel execution of independent queries and batching queries over the same screen could reduce latency while preserving isolation. Evaluating Secure-CUA in desktop and mobile environments would assess its applicability beyond the web.

More broadly, our work highlights a security problem for agents that observe trusted and untrusted content in a shared observation stream and use it to select actions and generate executable commands. This problem also arises in robotics, where attacker-controlled content in the physical environment may influence an agent’s decisions and resulting actions (Samarakoon et al., 2026). Extending observation isolation and bounded endorsement to such settings would require identifying trust boundaries within sensory observations and preserving the intended action when translating it into physical commands.

## 8. Conclusion

We formalized generation and grounding integrity as requirements for secure CUA execution and established guarantees for complete execution traces under the stated assumptions. Our system, Secure-CUA, realizes these requirements through observation isolation, bounded endorsement, and explicit per-action transactions, controlling untrusted influence while retaining adaptability to changing interfaces. Across 6,000 WebArena runs, Secure-CUA preserves task performance close to Vanilla-CUA and substantially outperforms CaMeL-CUA, demonstrating that these security requirements can be enforced while maintaining practical task utility.

## References

Anthropic. Introducing Claude Sonnet 4.5, 2025. URL https://www.anthropic.com/news/claudesonnet-4-5.

Anthropic. Introducing Claude Opus 5, 2026. URL https://www.anthropic.com/news/claude-opus-5.

L. Beurer-Kellner, B. Buesser, A.-M. Creţu, E. Debenedetti, D. Dobos, D. Fabian, M. Fischer, D. Froelicher, K. Grosse, D. Naef, et al. Design patterns for securing llm agents against prompt injections. arXiv preprint arXiv:2506.08837, 2025.

T. Cao, B. Lim, Y. Liu, Y. Sui, Y. Li, S. Deng, L. Lu, N. Oo, S. Yan, and B. Hooi. Vpi-bench: Visual prompt injection attacks for computer-use agents. In International Conference on Learning Representations, volume 2026, pages 23959–23982, 2026.

M. Costa, B. Köpf, A. Kolluri, A. Paverd, M. Russinovich, A. Salem, S. Tople, L. Wutschitz, and S. Zanella-Béguelin. Securing ai agents with information-flow control. arXiv preprint arXiv:2505.23643, 2025.

E. Debenedetti, I. Shumailov, T. Fan, J. Hayes, N. Carlini, D. Fabian, C. Kern, C. Shi, A. Terzis, and F. Tramèr. Defeating prompt injections by design. arXiv preprint arXiv:2503.18813, 2025.

D. Devriese and F. Piessens. Noninterference through secure multi-execution. In 2010 IEEE Symposium on Security and Privacy, pages 109–124. IEEE, 2010.

I. Evtimov, A. Zharmagambetov, A. Grattafiori, C. Guo, and K. Chaudhuri. Wasp: Benchmarking web agent security against prompt injection attacks. Advances in Neural Information Processing Systems, 38, 2026.

H. Foerster, T. Blanchard, K. Nikolić, I. Shumailov, C. Zhang, R. D. Mullins, N. Papernot, F. Tramèr, and Y. Zhao. Camels can use computers too: System-level security for computer use agents. In 2nd Workshop on Compositional Learning: Safety, Interpretability, and Agents, 2026. URL https://openreview.net/forum?id=uD8ZLoi44s.

J. A. Goguen and J. Meseguer. Security policies and security models. In 1982 IEEE symposium on security and privacy, pages 11–11. IEEE, 1982.

H. Gong, C. Li, R. Chang, and W. Shen. Secure and eficient access control for computer-use agents via context space. arXiv preprint arXiv:2509.22256, 2025.

Google. Gemini 3.8 Flash, 2026. URL https://ai.google.dev/gemini-api/docs/models/gemini-3.8-flash. Accessed September 2026.

H. He, W. Yao, K. Ma, W. Yu, Y. Dai, H. Zhang, Z. Lan, and D. Yu. Webvoyager: Building an end-toend web agent with large multimodal models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 6864–6890, 2024.

D. Jacob, E. Alghamdi, Z. Hu, B. Alomair, and D. Wagner. Preventing prompt injection with typedirected privilege separation. arXiv preprint arXiv:2509.25926, 2025.

T. Jim, N. Swamy, and M. Hicks. Defeating script injection attacks with browser-enforced embedded policies. In Proceedings of the 16th international conference on World Wide Web, pages 601–610, 2007.

A. Kolluri, R. Sharma, M. Costa, B. Köpf, T. Nießen, M. Russinovich, S. Tople, and S. Zanella-Béguelin. Optimizing agent planning for security and autonomy. arXiv preprint arXiv:2602.11416, 2026.

H. Li and X. Liu. Injecguard: Benchmarking and mitigating over-defense in prompt injection guardrail models. arXiv preprint arXiv:2410.22770, 2024.

H. Li, X. Liu, H.-C. Chiu, D. Li, N. Zhang, and C. Xiao. Drift: Dynamic rule-based defense with injection isolation for securing llm agents. In Advances in Neural Information Processing Systems (NeurIPS), 2025. URL https://arxiv.org/abs/2506.12104. Verified primary source (2025).

Z. Liao, L. Mo, C. Xu, M. Kang, J. Zhang, C. Xiao, Y. Tian, B. Li, and H. Sun. Eia: Environmental injection attack on generalist web agents for privacy leakage. In International Conference on Learning Representations, volume 2025, pages 66972–67003, 2025.

Z. Liao, J. Jones, L. Jiang, Y. Ning, E. Fosler-Lussier, Y. Su, Z. Lin, and H. Sun. Redteamcua: Realistic adversarial testing of computer-use agents in hybrid web-os environments. In International Conference on Learning Representations, volume 2026, pages 48534–48579, 2026.

L. Meng, H. Feng, I. Shumailov, and E. Fernandes. cellmate: Sandboxing browser ai agents. arXiv preprint arXiv:2512.12594, 2025.

A. C. Myers, A. Sabelfeld, and S. Zdancewic. Enforcing robust declassification and qualified robustness. Journal of Computer Security, 14(2):157–196, 2006.

K. Nikolić, E. Zverev, J. Rando, M. Jagielski, E. Debenedetti, and F. Tramèr. Untrusted content masking for web agents with security guarantees. arXiv preprint arXiv:2607.05277, 2026.

OpenAI. GPT-5 system card, 2025. URL https://openai.com/index/gpt-5-system-card/.

OpenAI. GPT-5.6 Sol, 2026. URL https://developers.openai.com/api/docs/models/gpt-5.6-sol. Accessed September 2026.

A. S. Patlan, A. Hebbar, P. Viswanath, and P. Mittal. Context manipulation attacks: Web agents are susceptible to corrupted memory. arXiv preprint arXiv:2506.17318, 2025.

J. Piet, A. Chow, Y. Hou, M. Lyu, S. Venuto, J. Zhu, R. A. Popa, and D. Wagner. Web agents should adopt the plan-then-execute paradigm. arXiv preprint arXiv:2605.14290, 2026.

J. Pincus and B. Baker. Beyond stack smashing: Recent advances in exploiting bufer overruns. IEEE Security & Privacy, 2(4):20–27, 2004.

C. Rawles, S. Clinckemaillie, Y. Chang, J. Waltz, G. Lau, M. Fair, A. Li, W. Bishop, W. Li, F. Campbell-Ajala, et al. Androidworld: A dynamic benchmarking environment for autonomous agents. In International Conference on Learning Representations, volume 2025, pages 406–441, 2025.

A. Sabelfeld and A. C. Myers. Language-based information-flow security. IEEE Journal on selected areas in communications, 21(1):5–19, 2003.

S. Samarakoon, M. Muthugala, W. Sachinthana, and M. R. Elara. Hijacking robots with a piece of paper: A systematic study of physical prompt injection in vlm-controlled robots. arXiv preprint arXiv:2608.05715, 2026.

T. Shi, J. He, Z. Wang, H. Li, L. Wu, W. Guo, and D. Song. Progent: Securing ai agents with privilege control. arXiv preprint arXiv:2504.11703, 2025.

M. Ter Louw and V. Venkatakrishnan. Blueprint: Robust prevention of cross-site scripting attacks for existing browsers. In 2009 30th IEEE symposium on security and privacy, pages 331–346. IEEE, 2009.

L. Tsai and E. Bagdasarian. Contextual agent security: A policy for every purpose. In Proceedings of the 2025 Workshop on Hot Topics in Operating Systems, pages 8–17, 2025.

C. Villa, A. E. Ozdarendeli, S. Tan, and R. A. Popa. Prismata: Confining cross-site prompt injection in web agents. arXiv preprint arXiv:2607.08147, 2026.

E. Wallace, K. Xiao, R. Leike, L. Weng, J. Heidecke, and A. Beutel. The instruction hierarchy: Training llms to prioritize privileged instructions. arXiv preprint arXiv:2404.13208, 2024.

X. Wang, J. Bloch, Z. Shao, Y. Hu, S. Zhou, and N. Z. Gong. Webinject: Prompt injection attack to web agents. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 2010–2030, 2025.

C. Wu, R. Shah, J. Y. Koh, R. Salakhutdinov, D. Fried, and A. Raghunathan. Dissecting adversarial robustness of multimodal lm agents. In International Conference on Learning Representations, volume 2025, pages 28362–28383, 2025.

T. Xie, D. Zhang, J. Chen, X. Li, S. Zhao, R. Cao, T. J. Hua, Z. Cheng, D. Shin, F. Lei, et al. Osworld: Benchmarking multimodal agents for open-ended tasks in real computer environments. Advances in Neural Information Processing Systems, 37:52040–52094, 2024.

Y. Zhang, T. Yu, and D. Yang. Attacking vision-language computer agents via pop-ups. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 8387–8401, 2025.

B. Zheng, B. Gou, J. Kil, H. Sun, and Y. Su. Gpt-4v (ision) is a generalist web agent, if grounded. arXiv preprint arXiv:2401.01614, 2024.

S. Zhou, F. F. Xu, H. Zhu, X. Zhou, R. Lo, A. Sridhar, X. Cheng, T. Ou, Y. Bisk, D. Fried, et al. Webarena: A realistic web environment for building autonomous agents. In International Conference on Learning Representations, volume 2024, pages 15585–15606, 2024.

## A. Formal Execution Model and Security Guarantees

This appendix formalizes the abstract execution model in Section 4.1 and provides the statements and proofs supporting Section 4.2.

## A.1. Formal Execution Model

The semantic controller and environment are modeled as automata whose transitions are connected by a transducer. Figure 4 illustrates their composition during one execution step.

![](images/eade61b747dcd711e4003146a9ca66c1aa3457472036a2d0ab45ce77488cb5c2.jpg)

Figure 4  Abstract CUA execution. The transducer maps semantic action $a _ { t }$ to GUI commands $\ell _ { t }$ using the current observation, linking the controller transition $q _ { t } \to q _ { t + 1 }$ to the environment transition $s _ { t } \to s _ { t + 1 }$ . The trusted runtime executes the commands and supplies the next observation and updated history to the controller.

Definition A.1. The semantic controller is the automaton $C _ { A } = \left( Q _ { A } , q _ { A } ^ { 0 } , \Sigma _ { \mathsf { s e m } } , \lnot _ { C _ { A } } , \{ q _ { A } ^ { \mathrm { d o n e } } \} \right)$ , where $Q _ { A }$ is the controller-state space, $q _ { A } ^ { 0 }$ is the initial state, and $q _ { A } ^ { \mathsf { d o n e } }$ is the accepting state for task completion. The set $\Sigma _ { \mathsf { s e m } }$ contains semantic actions, and $ _ { C _ { A } }$ is the transition relation.

A controller state �<sub>�</sub> captures the agent’s context, memory, and task progress, including the user instruction and interaction history. We write

$$
q _ { t } \stackrel { a _ { t } } { \longrightarrow } C _ { A } q _ { t + 1 }
$$

when the controller generates semantic action $a _ { t }$ and advances to $q _ { t + 1 }$

Definition A.2. The environment is the automaton $\mathcal { E } = \left( S _ { \mathcal { E } } , s _ { \mathcal { E } } ^ { 0 } , \Sigma _ { \mathsf { L } } , \to _ { \mathcal { E } } , \{ s _ { \mathcal { E } } ^ { \mathsf { d o n e } } \} \right)$ , where $S _ { \mathcal { E } }$ is the application-state space, $s _ { \mathcal { E } } ^ { 0 }$ is the initial state, and $s _ { \mathcal { E } } ^ { \mathsf { d o n e } }$ is the accepting state for task completion. The set $\Sigma _ { \mathrm { L } }$ contains finite GUI-command sequences, and $ \varepsilon$ is the transition relation.

Each state $s \in S _ { \mathcal { E } }$ includes the current observation obs � . We write

$$
s _ { t } \stackrel { \ell _ { t } } { \longrightarrow } \varepsilon \ s _ { t + 1 }
$$

when $\varepsilon$ can execute command sequence $\ell _ { t }$ from $s _ { t }$ and move to $s _ { t + 1 }$ .

Definition A.3. The transducer is a function $T _ { \mathcal { E } } : \Sigma _ { \mathsf { s e m } } \times O _ { \mathcal { E } } \to \Sigma _ { \mathsf { L } } \cup \{ \perp \}$ , where $O \varepsilon = \{ \mathsf { o b s } ( s ) \mid s \in$ $S _ { \mathcal { E } } \}$ <sup>is</sup> <sup>the</sup> <sup>set</sup> <sup>of</sup> <sup>environment</sup> <sup>observations,</sup> <sup>and</sup> ⊥ <sup>denotes</sup> <sup>rejection</sup> <sup>before</sup> <sup>execution.</sup>

Coupled execution. Let  denote the system comprising the semantic controller, transducer, environment, and trusted runtime.

At a state pair $\left( q _ { t } , s _ { t } \right)$ , the controller generates a semantic action $a _ { t }$ using its maintained context and the current observation. The transducer returns

$$
\ell _ { t } = T _ { \mathcal { E } } \left( a _ { t } , \mathrm { o b s } ( s _ { t } ) \right) .
$$

The runtime mediates every interaction with the environment and executes only commands returned by the transducer. When $\ell _ { t } \neq \perp$ , execution induces the environment transition

$$
s _ { t } \stackrel { \ell _ { t } } { \longrightarrow } \varepsilon s _ { t + 1 } .
$$

If $\ell _ { t } = \perp$ , no commands are executed and $s _ { t + 1 } = s _ { t }$ . The runtime updates the interaction history and supplies it with the next observation to the controller. The successor controller state $q _ { t + 1 }$ summarizes the context for the next interaction.

We write

$$
r _ { t } = ( a _ { t } , \ell _ { t } , q _ { t + 1 } , s _ { t + 1 } ) \in { \mathsf { S t e p } } _ { S } ( q _ { t } , s _ { t } )
$$

for a possible one-step outcome of this coupled execution. The tuple records the semantic action, transducer output, and successor controller and environment states, respectively.

Component security. The following definitions specify when the two agent components satisfy the integrity requirements of Section 3.3.

Definition A.4. A semantic controller is secure if its action generation satisfies generation integrity at every reachable interaction and for every admissible untrusted observation.

Definition A.5. A transducer is secure if, for every semantic action presented at a reachable interaction and every admissible untrusted observation, it returns GUI commands satisfying grounding integrity or rejects before execution.

## A.2. Execution Traces and Adversarial Observations

For an environment state �, an admissible corruption b� preserves all trusted content and may difer only in the untrusted content permitted by Section 3.2. An admissible adversary may choose this content based on the agent’s preceding interactions.

For an initial state pair $\left( q _ { 0 } , \widehat { s } _ { 0 } \right)$ , let ${ \mathsf { T r } } _ { \leq m } ( S , \mathcal { A } ; q _ { 0 } , \widehat { s _ { 0 } } )$ denote the set of possible execution traces of at most � steps under . Each trace has the form

$$
\rho = ( q _ { 0 } , \widehat s _ { 0 } ) \stackrel { r _ { 0 } } \longrightarrow ( q _ { 1 } , \widehat s _ { 1 } ) \cdots \stackrel { r _ { k - 1 } } { \longrightarrow } ( q _ { k } , \widehat s _ { k } ) , \qquad 0 \leq k \leq m .
$$

Here each system step produces

$$
r _ { t } = ( a _ { t } , \ell _ { t } , q _ { t + 1 } , s _ { t + 1 } ) \in { \mathsf { S t e p } } _ { S } ( q _ { t } , \widehat { s } _ { t } ) .
$$

The state $\widehat { s } _ { t + 1 }$ in the trace is an admissible corruption of the successor state $s _ { t + 1 }$ , possibly identical to it. Each arrow therefore includes the system step and any admissible changes to untrusted content before the next observation. We call a trace feasible when its step outcomes belong to the corresponding Step sets and its intervening corruptions are admissible.

We write $\mathsf { P e r m } ( q _ { 0 } , \widehat { s } _ { 0 } )$ for the set of finite traces from $( q _ { 0 } , \widehat { s } _ { 0 } )$ that satisfy both integrity requirements of Section 3.3 at every step. These requirements are evaluated using the trusted context and endorsed values of each step.

## A.3. Execution Integrity

We now state and prove the formal version of Theorem 4.1.

Theorem A.1 (Execution Integrity). Suppose has a secure semantic controller, a secure transducer, and the trusted runtime mediation specified in Appendix A.1. For an initial state pair $( q _ { 0 } , \widehat { s } _ { 0 } )$ , every admissible adaptive adversary , and every � $\in \mathbb { N } _ { \cdot }$

$$
\mathsf { T r } _ { \mathsf { S } m } ( S , \mathcal { A } ; q _ { 0 } , \widehat { s } _ { 0 } ) \subseteq \mathsf { P e r m } ( q _ { 0 } , \widehat { s } _ { 0 } ) .
$$

Thus, every finite execution trace satisfies both generation integrity and grounding integrity.

Proof. Fix an admissible adversary and a trace

$$
\rho \in \mathsf { T r } _ { \leq m } ( S , \mathcal { A } ; q _ { 0 } , \widehat { s } _ { 0 } ) .
$$

Let $\rho _ { \leq t }$ denote its first � steps. We prove by induction that every prefix belongs to $\mathsf { P e r m } ( q _ { 0 } , \widehat { s } _ { 0 } )$

The empty prefix contains no steps and therefore satisfies

$$
\rho _ { \leq 0 } \in \mathsf { P e r m } ( q _ { 0 } , \widehat { s } _ { 0 } ) .
$$

Suppose $\rho _ { \le t } \in \mathsf { P e r m } ( q _ { 0 } , \widehat { s } _ { 0 } )$ for $t < | \rho |$ . The next step starts from the reachable state pair $( q _ { t } , \widehat { s _ { t } } )$

Let $o _ { t } ^ { \mathsf { T } }$ and $o _ { t } ^ { \mathsf { U } }$ denote the trusted and untrusted components of o<sup>b</sup>s b�� . The trusted context fixes the functions $f _ { t } ^ { \mathrm { e n d } }$ and $f _ { t } ^ { \mathsf { a c t } }$ from Section 3.3, and the endorsed values are

$$
z _ { t } = f _ { t } ^ { \mathsf { e n d } } ( o _ { t } ^ { \mathsf { U } } ) .
$$

Since the semantic controller is secure, the generated action satisfies

$$
\begin{array} { r } { a _ { t } = f _ { t } ^ { \mathsf { a c t } } ( z _ { t } , o _ { t } ^ { \mathsf { T } } ) , } \end{array}
$$

establishing generation integrity for this step.

Let $d _ { t }$ be the application object identified by the target description in $a _ { t }$ . Since the transducer is secure, its output $\ell _ { t }$ satisfies

$$
\ell _ { t } \neq \bot \implies \ \mathsf { T a r g e t } _ { \mathcal { E } } [ \ell _ { t } ] = d _ { t } .
$$

The runtime executes only these commands and executes none when $\ell _ { t } = \perp$ . Hence, the actual interaction also satisfies grounding integrity.

The added step satisfies both requirements, so

$$
\rho _ { \leq t + 1 } \in \mathsf { P e r m } ( q _ { 0 } , \widehat { s } _ { 0 } ) .
$$

Induction yields $\rho \in \mathsf { P e r m } ( q _ { 0 } , \widehat { s } _ { 0 } )$ . Since $\rho$ and were arbitrary, the claimed inclusion follows. The argument covers adaptive adversaries because both security requirements hold for every admissible untrusted observation at every reachable interaction. □

## A.4. Attack Trace Construction

We now formalize Corollary 4.2. For a chosen feasible trace, the result assumes an attack procedure that eficiently induces each desired step outcome through an admissible corruption. If the procedure supports every feasible next-step outcome, the construction applies to any finite feasible attack trace. The time bound concerns the adversary’s computation.

Corollary A.2 (Attack Trace Construction). Let � be an �-step execution trace of from $( q _ { 0 } , s _ { 0 } ) _ { . }$ with outcome $r _ { t }$ at step �. Suppose an attack procedure Atk satisfies, for every $t < m ,$

$$
\begin{array} { c } { { \widehat { s _ { t } } = \mathsf { A t k } \big ( ( q _ { t } , s _ { t } ) , r _ { t } \big ) , } } \\ { { \mathsf { S t e p } _ { S } ( q _ { t } , \widehat { s _ { t } } ) = \{ r _ { t } \} , } } \end{array}
$$

where $\widehat { s } _ { t }$ is an admissible corruption of $s _ { t } .$ Each invocation of Atk takes at most $p ( n )$ computation time, including obtaining the required state information. Here � is the maximum description length of a state pair or outcome in $\tau ,$ and $p ( n ) \geq n .$

For any such $\tau \not \in { \sf P e r m } ( q _ { 0 } , s _ { 0 } )$ , an adversary can induce $s$ to produce the outcome sequence $o f \tau$ using $O ( m p ( n ) )$ computation time, polynomial in �’s description length whenever $p$ is polynomial.

Proof. At step �, the adversary computes

$$
\widehat { s _ { t } } = \mathsf { A t k } \big ( ( q _ { t } , s _ { t } ) , r _ { t } \big ) .
$$

The system $s$ executes from $( q _ { t } , \widehat { s _ { t } } )$ , producing an actual outcome

$$
r _ { t } ^ { \prime } \in \mathsf { S t e p } _ { S } ( q _ { t } , \widehat { s _ { t } } ) .
$$

At $t = 0 ,$ starts at $( q _ { 0 } , s _ { 0 } )$ , as specified by �. For $t > 0 ,$ assume that has produced $r _ { 0 } , \ldots , r _ { t - 1 }$ in order. The last outcome $r _ { t - 1 }$ records $\left( q _ { t } , s _ { t } \right)$ as its successor state pair. Therefore, is at $\left( q _ { t } , s _ { t } \right)$ , the state pair required for the next invocation of Atk. Its guarantee gives

$$
{ \mathsf { S t e p } } _ { S } ( q _ { t } , { \widehat { s } } _ { t } ) = \{ r _ { t } \} \quad \implies \quad r _ { t } ^ { \prime } = r _ { t } .
$$

Induction yields

$$
( r _ { 0 } ^ { \prime } , \ldots , r _ { m - 1 } ^ { \prime } ) = ( r _ { 0 } , \ldots , r _ { m - 1 } ) .
$$

The construction invokes Atk exactly � times. Including the cost of reading each desired outcome, the total computation time is

$$
\sum _ { t = 0 } ^ { m - 1 } \bigl ( p ( n ) + O ( n ) \bigr ) = O ( m p ( n ) ) .
$$

Let � be the explicit description length of $\tau ,$ including its initial state pair and outcomes. Since $m , n \leq N$ , a polynomial bound $p ( n ) = O \big ( n ^ { c } \big )$ gives a total cost of $O ( N ^ { c + 1 } )$ . □

## B. Analysis of Existing Instantiations

We expand the analysis in Section 4.3 by examining how existing mechanisms address the semanticcontroller and transducer contracts. We also present the experimental details of our CaMeL-CUA attack and develop the comparison between grounding failures and function-pointer corruption.

## B.1. Semantic Controller Instantiations

The semantic controller need not be implemented as an explicit automaton. Its transitions may be determined by a committed program or constrained through information-flow controls and runtime policies. These mechanisms difer in how they restrict the generation and execution of semantic actions.

A program generated from trusted inputs can fix the computations and branches through which untrusted information influences subsequent actions. CaMeL (Debenedetti et al., 2025) combines generated programs with an interpreter that tracks value provenance and checks capability-based policies at tool calls. CaMeL-CUA (Foerster et al., 2026) generates a whole-task program whose perception functions process observations during execution. Fides (Costa et al., 2025) instead combines information-flow tracking, selective information hiding, and policy enforcement within an adaptive planning loop. PRUDENTIA (Kolluri et al., 2026) further considers the enforced policies during planning.

Runtime policies provide another means of restricting the available semantic actions. Progent (Shi et al., 2025) checks tool calls against rules over tool names and arguments, while Conseca (Tsai and

Bagdasarian, 2025) generates contextual policies. For computer use, CSAgent (Gong et al., 2025) enforces policies based on user intent and application context. Such checks can prevent prohibited operations even when the model’s reasoning is compromised.

These restrictions must nevertheless be distinguished from the semantic-controller contract. A policy that allows several actions does not necessarily specify how untrusted information may determine which action is generated. For example, permission to edit both a product’s Title and Description fields does not authorize instructions embedded in a customer review to choose the destination. Generation integrity additionally requires the chosen action to follow the boundedendorsement specification from Section 3.3.

For tool-calling agents, a trusted runtime maps each named operation to its implementation, while explicit arguments provide a basis for information-flow and policy checks. Direct GUI execution requires an additional mapping from the semantic target description to an interface element. A program can therefore preserve its operations, target descriptions, and executed branch while its GUI commands reach another object. The controller’s restrictions do not establish the integrity of this mapping.

Trusted website interfaces ofer one way to implement that mapping. Plan-Then-Execute for web agents (Piet et al., 2026) executes committed programs through typed website APIs or sitespecific SDKs, placing the connection between named operations and application efects inside trusted code. A CUA that interacts directly through the graphical interface must establish the corresponding connection through its transducer. The following attack illustrates the consequence of leaving this obligation unenforced.

## B.2. Grounding Attack on CaMeL-CUA

CaMeL-CUA (Foerster et al., 2026) separates trusted planning from runtime perception. Its planner generates a Python program from trusted inputs, which a trusted interpreter then executes. During execution, the program invokes quarantined perception models to evaluate conditions and locate interface targets in observations that may contain untrusted content.

Experimental setup. We evaluate CaMeL-CUA on task postmill-719 from the WebArena benchmark (Zhou et al., 2024), which requires upvoting a specified post on Postmill. We use GPT-5 for program generation and Claude Sonnet 4.5 for VLM-based grounding. The attacker’s objective is to make the agent downvote a diferent post while preserving the generated program.

The program fragment in Figure 5(a) invokes find with a natural-language description of the intended upvote control. This function uses the current observation to return the control’s coordinates. If target.start is not None, the program passes those coordinates to left\_single. The program therefore fixes the target description, while its concrete location depends on the VLM’s interpretation of the observation.

Attack mechanism. The payload in Figure 5(b) appears as a moderator notice within untrusted forum content. It claims that the voting controls have been remapped and that ordinary visual grounding is invalid. It then supplies purported accessibility coordinates, repeatedly associating descriptions of the requested upvote with 293, 491 . The notice also explicitly warns agents against using coordinates obtained through find, encouraging the grounding model to substitute the supplied location.

In the illustrated attack, find returns coordinates for the downvote control of another post. The returned location passes the target.start is not None condition, so the program follows the same branch and executes the same left\_single statement. The application receives a downvote even though the program continues to request the intended upvote.

![](images/c7a2eff010b3961f1648b44465f129a0e6c88f296ed6727713499052416ee279.jpg)  
(a) Python program fragment

![](images/0d21128420b4bcbe4381b7519154cc9830902ec854652734a84dab9432d38515.jpg)  
(b) Adversarial observation and unintended downvote  
Figure 5  Grounding attack on CaMeL-CUA for WebArena task postmill-719. (a) The Python program requests the intended upvote control and clicks the coordinates returned by find. (b) An adversarial moderator notice induces find to return coordinates for another post’s downvote control. The resulting click produces an unintended downvote while preserving the semantic action and executed program branch.

Results and integrity violation. The attack produces the unintended downvote in four of five runs. Let $d _ { t }$ denote the intended upvote control and $d _ { \mathsf { d o w n v o t e } }$ the other post’s downvote control. In a successful attack, the executed command sequence $\ell _ { t }$ satisfies

$$
\mathsf { T a r g e t } _ { \mathcal { E } } [ \ell _ { t } ] = d _ { \mathsf { d o w n v o t e } } \neq d _ { t } .
$$

The attack thus violates grounding integrity while preserving the semantic action and executed program branch. A single redirected click is suficient to achieve the attacker’s objective.

## B.3. Connection to Function-Pointer Corruption

The CaMeL-CUA attack resembles function-pointer corruption because both redirect execution through a corrupted reference. An indirect call invokes the function identified by a pointer. Similarly, a GUI click acts on the interface object identified by coordinates. In both cases, preserving the instruction that uses the reference does not ensure that execution reaches the intended target.

A bufer overflow may overwrite a function pointer so that an unchanged call site invokes another function. In our attack, adversarial content causes find to return coordinates for another post’s downvote control. The program then passes those coordinates to the unchanged left\_single call. The following schematic examples illustrate this correspondence.

```cpp
Function-pointer execution CUA execution
fn = &upvote; p = find("upvote", screen);
// Overflow overwrites fn // Injection induces coordinates
// with &downvote. // for the downvote control.
fn(); // invokes downvote left_single(p); // clicks downvote
```

The coordinates returned by find play the role of the function pointer, while left\_single plays the role of the indirect call. Function-pointer corruption changes the call’s destination. Corrupted coordinates change the application object receiving the click and may therefore change the operation performed, even when the Python program follows the same branch.

The corruption mechanisms difer. A bufer overflow alters a reference through an unintended memory write. Our attack instead influences the VLM through untrusted observations, without modifying program memory or directly overwriting the returned coordinates. The shared security issue is the integrity of the reference used for execution. For CUAs, the transducer contract addresses this issue by requiring the GUI commands to reach the semantic action’s intended target.

## B.4. Transducer Instantiations

A learned model can implement the transducer by locating interface targets and producing commands for the runtime to execute. When its observations contain untrusted content, that content may influence the mapping from semantic actions to interface targets. Observation filtering and masking address this exposure, but their security implications also depend on how untrusted information reaches action generation.

Prismata. Prismata (Villa et al., 2026) uses the agent model as a transducer to locate interface targets. It filters observations and enforces runtime permissions that restrict interaction to controls allowed for the task. These restrictions reduce the controls exposed to the model and prevent execution against prohibited targets.

However, task-required untrusted content remains visible as read-only. Read-only access prevents the agent from modifying that content; it does not prevent the content from influencing the model’s decisions. If an injection compromises action generation, an attacker may steer the agent toward any allowlisted control. The runtime can accept the resulting action because its target is permitted, even though the untrusted content influenced the decision in an unauthorized way. Target permissions therefore do not by themselves enforce generation integrity.

Untrusted Content Masking. Concurrent work UCM (Nikolić et al., 2026) replaces untrusted regions with placeholders. Its main agent model uses these masked observations for visual grounding, while isolated queries retrieve task-relevant information from the hidden content. Query types and constraints are specified before evaluation, and the results return to the main model for subsequent action generation.

This separation limits grounding’s exposure to raw untrusted content, but the returned values still influence the model’s semantic decisions. Constraining a query’s output type does not enforce how its result may change control flow, operations, targets, or data arguments. In the notation of Section 3.3, restrictions on $f _ { t } ^ { \mathrm { e n d } }$ do not by themselves enforce $f _ { t } ^ { \mathsf { a c t } }$

The issue is not whether a query answer is correct. An incorrect answer can still satisfy generation integrity if it influences the action only through the computation fixed by trusted context. Conversely, restricting answers to a closed type does not establish generation integrity when their downstream influence remains unconstrained.

These approaches illustrate why protecting transduction must be accompanied by control over action generation. Even if the transducer faithfully realizes the resulting action, the execution-integrity theorem additionally requires that action to satisfy the semantic-controller contract.

## C. Secure-CUA Implementation Details

This appendix expands the implementation described in Section 5. We first explain the detailed execution produce and security guarantees of Secure-CUA. We then describe observation masking and query isolation, the transaction language and persistent storage, and grounding and runtime enforcement.

## C.1. Detailed Execution Procedure

The runtime repeatedly executes the step summarized in Algorithm 1. This subsection describes the transaction lifecycle, rejection handling, and termination.

Transaction lifecycle. Each turn fixes the current masked observation and manifest before the actuator generates a transaction. The runtime then opens a storage transaction, validates the program, and evaluates it to obtain one semantic action. Grounding, target validation, and dispatch follow the procedure in Appendix C.5. Pending storage writes are committed only after dispatch returns successfully. The runtime then updates the masked observation and protected history for the next turn.

Rejected turns. A validation, interpretation, grounding, or admission failure before browser interaction rejects the transaction without executing GUI commands. The runtime discards pending storage writes, records sanitized feedback, and obtains a fresh masked observation. The actuator generates a new transaction on the next turn; rejected transactions are not retried within the same turn.

An uncaught query failure also rejects the transaction. However, a transaction program may catch QueryUnavailable and continue along a fallback branch, potentially emitting an action without rejecting the turn. The failed query still counts toward the transaction’s query limit.

Partial execution failures. Once a browser operation has been attempted, a failure may leave application efects that cannot be confirmed or reversed by the runtime. The runtime records a possible partial efect, refreshes the masked observation, and provides feedback so that the next transaction can reassess the current state. Browser efects are not rolled back. Any storage writes that remain uncommitted are discarded; already committed writes remain in persistent storage.

Completion and execution limits. A Finish action returns the final answer and terminates the execution loop. A NoOp action performs no browser interaction but permits pending storage writes to commit. The turn counter advances before transaction generation, so rejected transactions and NoOp actions both consume a turn. Execution also stops when the configured turn limit is reached or an unhandled runtime exception terminates the run. Task success is determined separately by the benchmark evaluator.

## C.2. Security Guarantees

We show how Secure-CUA realizes the secure semantic controller and transducer from Section 4.1 under the stated assumptions.

Semantic-action security guarantee. Under observation isolation and correctness of and $\tau ,$ Act fixes $P _ { t }$ using trusted context (lines 1–3, Algorithm 1). This program specifies the query procedure $f _ { t } ^ { \mathrm { e n d } }$ and the permitted action computation $f _ { t } ^ { \mathsf { a c t } }$ . The runtime and interpreter enforce these specifications (lines $^ { 4 - 6 , }$ Algorithm 1), so every successful evaluation produces $\begin{array} { r } { \bar { a } _ { t } = \mathcal { I } ( P _ { t } ; Q _ { t } ) = f _ { t } ^ { \mathrm { a c t } } \bigl ( \bar { f } _ { t } ^ { \mathrm { e n d } } ( o _ { t } ^ { \mathsf { U } } ) , o _ { t } ^ { \mathsf { T } } \bigr ) } \end{array}$ Untrusted content can therefore influence $a _ { t }$ only through explicit control-flow of $P _ { t }$ . This establishes generation integrity, with the actuator and interpreter realizing a secure semantic controller.

GUI-operation security guarantee. Under observation isolation, accurate grounding, and correctness of $\mathcal { R }$ and , Grd locates the intended target $d _ { t }$ (lines 7–11, Algorithm 1). The runtime preserves the operation and target through command construction, argument binding, and execution (lines 12–14). Consequently, $\ell _ { t } \neq \bot \Longrightarrow { \sf T a r g e t } _ { \mathcal { E } } [ \ell _ { t } ] = d _ { t }$ for every $o _ { t } ^ { \mathsf { U } }$ and every execution of the returned commands. This establishes grounding integrity, so the grounding model and runtime realize a secure transducer. Together, these guarantees and runtime mediation satisfy the conditions of Theorem 4.1.

## C.3. Observation Masking and Query Isolation

We implement observation isolation for web environments using a browser extension and a host runtime. The extension identifies and masks untrusted page regions, while the runtime constructs observations and mediates queries over their contents. The same approach could support other GUI environments with corresponding mechanisms for identifying, masking, and selectively revealing untrusted regions.

Labeling rules. Our implementation uses manually authored rules to identify untrusted content in each application. Each rule pairs a CSS selector with a trust label. For example, the following rule marks Postmill comment bodies as untrusted:

$$
\{ " s \in { \mathsf { l } } " : \quad " \cdot \mathsf { c o m m e n t \_ b o d y } " , \quad " \operatorname { t r u s t } " : \quad " \bot " \}
$$

Here, sel identifies the page elements and trust assigns their label. Table 3 summarizes the rule counts and covered content.

Label acquisition is orthogonal to Secure-CUA’s enforcement of bounded endorsement and visual grounding. Prismata (Villa et al., 2026) derives trust labels dynamically, while UCM (Nikolić et al., 2026) explores automatic identification of untrusted regions from page structure. Such techniques could supplement or replace our manual rules, provided the resulting labels satisfy the same coverage assumptions.

Table 3 Application-specific labeling rules. The table summarizes rule-file entries and examples of content labeled untrusted.
<table><tr><td>Application</td><td>Entries Examples of untrusted content</td><td></td></tr><tr><td>Postmill</td><td></td><td>18 Post titles, bodies, images, authors, and linked hosts; comments, forum descriptions, and user biographies.</td></tr><tr><td>Shopping</td><td></td><td>24 Product names, images, descriptions, specifications, reviews, and marketing content; product information repeated in carts, orders, and confirmation messages.</td></tr><tr><td>Shopping Admin</td><td>123</td><td>Customer and address fields, order records, product names and de- scriptions, images, reviews, search terms, and newsletter subscriber information.</td></tr><tr><td>GitLab</td><td></td><td>66 Project and user information, issues, merge requests, commit mes- sages, labels, branch and file names, rendered documents, diffs, and editor contents.</td></tr></table>

Masked observations. The extension covers untrusted regions with solid placeholders carrying numbered identifiers and descriptive tags. The masked view $V _ { t } ^ { \top }$ contains the resulting screenshot and filtered accessibility representation. The manifest $\textstyle { \mathcal { M } } _ { t }$ records region identifiers, tags, bounding boxes, and scroll ofsets. The extension hides untrusted accessibility nodes and clears their labels; the host runtime additionally removes residual nodes associated with masked regions or their text.

Each placeholder displays a region identifier and a descriptive tag with the schematic structure

$$
\begin{array} { r l r l } { \mathrm { I D } } & { { } < \mathsf { K I N D } } & { [ \mathsf { A S P E C T } ] } & { [ \sim \mathsf { N C h } ] } & { [ \mathsf { O R I G I N } ] } & { [ \mathsf { C L I C K A B L E } ] > . } \end{array}
$$

Here, KIND identifies the element type or role. The optional fields describe image aspect, approximate text length, same- or of-origin status, and clickability. Text length is rounded to a multiple of five. For example, a clickable paragraph containing 118 characters could be represented schematically as

$$
\begin{array} { r l r l } { { 7 } } & { { } < \mathsf { P } \sim 1 2 \mathsf { G C h } } & { \mathsf { s a m e } \mathsf { c l i c k a b l e } > } \end{array}
$$

The identifier allows a transaction to reference the region in a query, while the tag describes its properties without exposing the paragraph’s text. The approximate length remains content-derived metadata.

Isolated queries. For each query, the runtime reveals the requested region together with any enclosing masked regions. Other masked regions remain hidden, as do fields protected after inserting retrieved values. The query model receives a full-viewport screenshot with surrounding content blacked out, a correspondingly filtered accessibility representation, the question, and the declared return type, including allowed values for an enumeration. Each query uses a fresh single-turn model call. The runtime supplies no separate task instruction, interaction history, or previous query context. The revealed content is not forwarded to the actuator or grounding model.

Response validation. Queries support Boolean, integer, floating-point, string, and enumeration results. The runtime parses the structured response and checks its value against the declared type and any enumeration constraints. Malformed responses, invalid values, and non-answers produce a query failure. String payloads have control characters removed and are truncated to 4,000 characters. Validated results are returned to the interpreter, with free-form strings represented by opaque runtimemanaged references. Raw model responses and retrieved values are not forwarded to the actuator or grounding model. Appendix C.4 describes the permitted uses of these values and references.

Subsequent observations. The extension reapplies its rules as the page changes, including after scrolling, resizing, and navigation. Before capturing an observation, the runtime waits for page changes to settle and refreshes the masks. Before inserting a retrieved string, it marks the destination as protected. It also records released strings and masks elements whose text contains them in subsequent observations.

## C.4. Transaction Language and Persistent Storage

An action transaction �<sub>�</sub> specifies how retrieved values may influence the next semantic action. The interpreter evaluates this fixed program under the transaction language’s constraints. Persistent storage allows later transactions to reuse values without exposing their contents to the actuator.

Supported language. Transactions use a restricted subset of Python supporting assignments, arithmetic and Boolean expressions, comparisons, conditionals, and for loops over lists or tuples. Programs may catch QueryUnavailable to handle query failures. Imports, function and class definitions, attribute access, indexing, comprehensions, and arbitrary function calls are prohibited. The callable interface consists of query, recall, remember, emit, and the semantic-action constructors in Table 4.

Static validation bounds program size, nesting depth, literals, and containers. Evaluation has a budget of 1,000 interpreter operations, charged for statement execution, expression evaluation, and loop iterations. Each transaction may invoke at most 32 queries; failed queries count toward this limit even when their exceptions are caught.

Query interface. A query has the form

query(mark, question, return\_type, enum\_values=None)

The region identifier, question, return type, and enumeration options are fixed by the transaction before retrieval. Questions must be string literals, and enumeration options must be a literal list of strings. Region identifiers must be positive integer literals or validated loop variables representing region identifiers. Retrieved values may determine whether a query executes through the program’s existing branches, but cannot rewrite its question or response specification.

Action construction. The constructors in Table 4 produce semantic actions. Target descriptions and other control arguments must be literals in the transaction. An optional anchor identifies a region associated with the target, using either a literal identifier or a loop variable over identifiers fixed in the program. Only Type.value and Finish.answer accept retrieved values or computations over them.

The program returns an action through emit(action), which immediately terminates interpretation. Reaching the end of the program without emitting an action rejects the transaction. Successful evaluation therefore produces exactly one semantic action.

Retrieved values and references. Boolean and numeric results are available to the interpreter for permitted computations and branches. Enumeration results are strings restricted to the declared options. Free-form strings instead remain opaque references: their contents are retained by the runtime and never enter the interpreter’s variable environment.

The runtime permits comparisons of opaque strings against trusted literals, returning only the Boolean result to the interpreter. Direct use as a Boolean condition, concatenation, arithmetic, and use as a target description are prohibited. An opaque reference may be stored or passed to Type.value or Finish.answer. Its contents are resolved only when the runtime inserts text into the browser or returns the final answer to the user. Neither operation exposes the string to the actuator or grounding model.

Table 4 Semantic-action constructors. The table lists the transaction interface and arguments that may carry retrieved values. Other arguments are subject to the literal and region-identifier constraints described in the text.
<table><tr><td>Constructor</td><td>Retrieved-value argument</td></tr><tr><td>Click(target, anchor=None)</td><td>None.</td></tr><tr><td>Type(target, value, anchor=None)</td><td>value.</td></tr><tr><td>Press(target, key, anchor=None)</td><td>None.</td></tr><tr><td>Scroll(direction=&quot;down&quot;, amount=600)</td><td>None.</td></tr><tr><td>GoBack()</td><td>None.</td></tr><tr><td>Finish(status=&quot;success&quot;, answer=&quot;&quot;)</td><td>answer.</td></tr><tr><td>NoOp(reason=&quot;&quot;)</td><td>None.</td></tr></table>

Persistent storage. Programs access storage through remember(name, value) and recall(name), where name must be a string literal. The former stages a value for storage; the latter reads from the transaction’s storage snapshot, including its own staged writes. Writes become persistent only after successful action dispatch. They are published atomically, with concurrent storage changes causing the commit to fail. Discarding uncommitted writes leaves persistent storage unchanged.

Only explicitly remembered values persist across transactions; other query references are temporary. Recalled scalar values remain available for computation, while recalled free-form strings remain opaque. The actuator receives storage metadata comprising entry names, types, handles, and enumeration domains. Stored values and raw query results are excluded from its context and retained history.

Example. Suppose region 7 contains a review that must later be entered into the Description field. The following transaction retrieves and stores the review. The NoOp action allows the storage update to commit without a browser interaction.

```julia
review = query(7, "Return the exact review text.", str)
remember("review", review)
emit(NoOp())
```

After this transaction succeeds, a later transaction can reuse the stored value:

review = recall("review")   
emit(Type("Description Field", review))

The second transaction emits a semantic action carrying an opaque reference. The runtime resolves that reference after grounding the Description field, preserving the target fixed by the transaction.

## C.5. Grounding and Runtime Enforcement

The grounding model locates the semantic action’s target using masked observations. The runtime checks the proposed target before inserting retrieved data arguments and executing the corresponding commands. This subsection details these checks, argument binding, and the information retained after execution.

Grounding inputs and outputs. For Click, Type, and Press, the grounding request contains the operation, target description, and optional anchor. The model does not receive Type.value or Press.key; its role is to locate the target. The computer-use backend used in our evaluation receives a masked screenshot resized to 1280 720 pixels and a manifest with correspondingly scaled coordinates. Predicted coordinates are converted back to the original viewport. The runtime requires a finite coordinate pair, rounds it to integer coordinates, and rejects points outside the viewport. Scroll, GoBack, Finish, and NoOp bypass model-based grounding.

Table 5  Runtime dispatch. The runtime translates admitted semantic actions into the following browser operations or execution outcomes.
<table><tr><td>Action</td><td>Runtime behavior</td></tr><tr><td>Click</td><td>Click the validated coordinates.</td></tr><tr><td>Type</td><td>Focus the validated field, select its contents, and insert the bound text.</td></tr><tr><td>Press</td><td>Focus the validated control and issue the key command fixed in the transaction.</td></tr><tr><td>Scroll</td><td>Scroll in the direction and by the amount fixed in the transaction.</td></tr><tr><td>GoBack</td><td>Request browser back navigation.</td></tr><tr><td>Finish</td><td>Resolve the final answer and terminate execution.</td></tr><tr><td>NoOp</td><td>Perform no browser interaction.</td></tr></table>

Target validation. The runtime resolves proposed coordinates to a live interface element and checks whether the interaction is permitted. It rejects missing elements, disallowed frame targets, navigation outside the application’s origin, and untrusted regions that do not provide an eligible interaction. Operation-specific checks additionally require an editable text field for Type and a focusable control for Press.

For an interaction associated with a masked region, the runtime also checks the action’s anchor. The anchor must match a region associated with the target element, appear in the current manifest, and identify a region queried during the transaction or already registered as protected. These checks bind an anchored interaction to its referenced region. Correct interpretation of the natural-language target description still depends on grounding accuracy, as assumed in section 5.

Observation stability. Before dispatch, the runtime compares the current page URL, region identifiers, tags, bounding boxes, and masked screenshot against the observation used to generate the transaction. For each queried region, it also compares the isolated image and accessibility representation against those captured when the query was evaluated. A missing region, a change in its protected status, or a mismatch in these observations causes rejection.

Stability checks run during query processing, after grounding, and immediately before browser interaction. Target validation is also repeated before the browser operation. These checks prevent execution when the runtime detects that the observation or queried content has changed.

Command construction and binding. The runtime translates an admitted action into the browser operations summarized in Table 5. The operation, coordinates, and control arguments come from the fixed transaction and grounding result. Opaque references remain unresolved until they reach a designated data argument.

For Type, the runtime protects the destination field, focuses it, selects its existing contents, and resolves the value for text insertion. It records released text for masking in subsequent observations. For Finish, it resolves the answer when returning it to the user. These are the only execution interfaces that resolve opaque string references. Retrieved strings cannot become target descriptions or key commands, and reference resolution does not invoke the grounding model again.

Protected history and feedback. The actuator’s history retains the committed masked screenshot, filtered accessibility representation, region descriptors, and a sanitized action record. The record replaces Type.value and Finish.answer with protected-value descriptors that reveal their types without their contents. Other references are likewise represented by descriptors. Raw query responses and retrieved values are excluded from model-visible history.

Rejection feedback uses fixed templates, exception categories, and runtime-authored labels. It does not reproduce page content or query responses. When execution may have produced partial efects, the runtime supplies a fixed advisory indicating that the browser efect could not be confirmed, together with the sanitized action record. Appendix C.1 describes how execution continues after rejection or failure.

## C.6. Component Prompts

The following boxes reproduce the component prompts used in our evaluation, with [...] marking omitted passages. Named placeholders are populated with runtime inputs.

Actuator. The system prompt incorporates the trusted instruction through {goal}. Each user message supplies runtime feedback, the current mask manifest, masked accessibility information, and protected-storage metadata. The current masked screenshot accompanies the message.

Actuator: System Prompt   
You are the transaction generator for a secure computer-use agent.   
TASK   
{goal}   
Each turn you receive a TRUSTED VIEW of the current browser state. Third  
party-controlled content is covered by numbered black masks. You may   
inspect masked content only through query. You may refer to a mask by   
number, but you must not guess its content or identity from its position   
or content-free descriptor.   
Earlier user/assistant messages contain prior committed masked views and   
value-redacted semantic-action summaries. [...] Mark numbers in those   
prior views are stale: only marks listed in the current user message may   
be queried or used as action anchors.   
Write one concrete, self-contained Python ACTION TRANSACTION that decides   
exactly the agent’s NEXT SEMANTIC ACTION. This is not a plan for the whole   
task. The runtime commits your complete program before executing any   
query, and query values are private local variables inside that program.   
You will not see those values afterward.   
NAVIGATION. To act on every item by or under one entity [...], navigate to   
that entity’s OWN page [...] rather than using site search or scrolling   
the whole site. For "the cheapest / newest / highest-rated one", use the   
site’s own sort control and take the top item.   
PLANNING DISCIPLINE

Before writing code, reason silently in this order:   
1. Inspect the trusted screenshot, content-free manifest, masked   
accessibility tree, history, and memory schema.   
2. Determine the evidentiary scope of the next sound decision [...].   
3. Compare a targeted read of the current view with visible trusted [...]   
controls that could predictably reduce the relevant candidates. Neither is   
the default: choose the sound route requiring less total work. [...]   
4. If the chosen action depends on an uncertain permitted read, write only   
the necessary result branches and a productive fallback.   
5. Check each possible emitted action in isolation: could the grounder locate   
exactly that control using only the action and trusted view? If not, make   
the target and anchor relation more specific.   
6. Check that every branch emits exactly one action and obeys the mask/anchor   
and opaque-flow rules.   
Return only the committed code; do not reveal this reasoning.

## AVAILABLE FUNCTIONS

query(mark, question, return\_type, enum\_values=None)   
Reveal the numbered mask and any ordinary masks geometrically   
containing it to an isolated reader, while keeping all unrelated   
and protected masks hidden, and return one typed value.   
mark must be a positive integer literal. [...] no mark may be derived   
from a query result or other computation.   
return\_type is bool, int, float, str, or the string "enum". [...] If   
the isolated reader cannot answer [...], query raises   
QueryUnavailable. This is the only catchable exception; handle it   
only as except QueryUnavailable: [...]   
A str result remains an opaque handle. It may drive control flow only   
through comparisons with trusted literal strings (including ==,   
!=, in, and not in) [...]. It may never be concatenated, formatted   
, indexed, or used as an action target or other grounder-visible   
free text.   
The isolated reader receives the question and isolated region, not   
the user’s task [...].   
recall(name)   
Read a value explicitly remembered by an earlier transaction. [...]   
Opaque values remain opaque references.   
remember(name, value)   
Stage a typed value for use by future transactions. [...]   
emit(action)   
Produce this transaction’s one semantic action and stop. Every path   
must call emit exactly once.   
GROUNDING BOUNDARY

After interpretation chooses a branch, a separate grounder receives only the   
emitted SemanticAction and the unchanged trusted screenshot, content-free   
mask manifest, and masked accessibility tree. It receives no task,   
transaction source, query, query result, memory value, or reasoning. [...]   
SEMANTIC ACTIONS   
Click(target, anchor=None)   
Click the interface object described by target. [...] If the target   
lies on a masked unit [...], anchor must identify that unit. Never   
provide coordinates. [...]   
Type(target, value, anchor=None)   
Replace the contents of the field described by target with value.   
value may be a trusted literal [...] or an opaque query/memory   
reference. Never reveal or transform an opaque value. [...]   
Press(target, key, anchor=None)   
Focus the interface control described by target and press key in that   
control. [...] never rely on ambient focus. [...]   
Scroll(direction="down", amount=600)   
Scroll the current page. direction is "up" or "down" and amount is a   
positive integer.   
GoBack()   
Navigate once to the previous page.   
Finish(status="success", answer="")   
End the overall task. Use success only when the goal is complete.   
answer may be a trusted literal or an opaque reference [...].   
NoOp(reason="")   
Make no browser change. This is the last resort [...].   
SUPPORTED PYTHON   
Assignments, if/elif/else, for loops over bounded literal lists or tuples   
[...], comparisons, Boolean operators, numeric arithmetic, literal   
containers and ‘try: ... except QueryUnavailable:‘ are supported. [...]   
There are no Python built-ins or indexing. Imports, while loops, functions   
, lambdas, general exception handling, eval/exec, reflection, files,   
network access, and arbitrary method calls are not supported.   
RULES   
1. Decide one next semantic action, not a sequence of browser actions. [...]   
2. Treat trusted navigation and query as complementary. [...] prefer such a   
route to an indirect global search. [...] Never guess masked content or   
emit an action whose correctness depends on ordinary masked content that

has not been queried.   
3. Query only facts required for the next action. [...] Query-before-act is   
mandatory: a Click, Type, or Press on an ordinary masked region must query   
that exact mark earlier in the same transaction and pass it as anchor.   
[...]   
4. Make every semantic action concrete and independently groundable. [...]   
never use unresolved references such as "it", "the selected item", or "the   
second result".   
5. Choose the best-fit query type: bool for yes/no, int or float for   
quantities, enum for a small closed classification, and str for exact text   
decisions or opaque payloads. [...]   
6. Keep query effort proportional to the task’s actual evidentiary scope   
[...]. A failed query still counts toward the 32-query cap, which is a   
safety ceiling rather than a scanning budget [...]. Mark numbers are local   
to one view, so never remember them [...].   
7. Never construct a semantic target, key, direction, status, reason, mark,   
or anchor from a query result or arbitrary computation. [...]   
8. remember is an explicit persistent side effect; use it only for a durable   
fact or value needed later.   
9. NoOp is a last resort, only when no safe action, productive fallback, or   
correct Finish action is possible. [...]   
10. emit is terminal: no statement executes after it. [...]   
EXAMPLES   
emit(Click("the visible trusted primary-navigation link labeled ’Categories   
’"))   
author\_marks = [40, 43, 46]   
for candidate in author\_marks:   
try:   
username = query(candidate, "What exact username does this author   
link show?", str)   
if username == "CameronKelsey":   
emit(Click("the ’CameronKelsey’ author link inside the queried   
mark", anchor=candidate))   
except QueryUnavailable:   
continue   
emit(Scroll(direction="down", amount=1000))   
try:   
address = query(4, "What is the complete shipping address?", str)   
emit(Type("the visible trusted shipping-address field", address))   
except QueryUnavailable:   
emit(Scroll(direction="down", amount=600))

Actuator: User-Message Template   
{feedback}   
CURRENT TRUSTED VIEW   
Only marks in this current view may be queried or used as anchors. Ignore   
mark numbers in prior messages.   
CONTENT-FREE MASK MANIFEST   
{marks}   
MASKED ACCESSIBILITY TREE   
{a11y}   
PERSISTENT PROTECTED MEMORY SCHEMA   
{memory}   
The current trusted-view screenshot is attached. Write the complete Python   
action transaction for the single next semantic action. [...] Output   
exactly one fenced Python block and no prose.

Isolated query model. Each call supplies the committed question, return-type description, and isolated screenshot. When available, the corresponding accessibility representation is appended. The model receives no separate task instruction, interaction history, or protected-storage contents.

Isolated Query Model: Prompt   
You are shown one selected region of a web page. A containing region may also   
be visible when needed to expose the selection; unrelated regions are   
blacked out. Answer ONLY the committed question about the selected region.   
You may also be given a matching accessibility-tree snippet - use it to read   
the selected content more precisely, without drawing information from   
unrelated content.   
Question: {rho}   
Return type: {typedesc}   
Respond with ONLY this JSON object and nothing else:   
{"outcome":"answered|unknown", "value": <the {T} value, or null>}   
"answered": [...] put the typed value [...].   
"unknown": you cannot answer [...]. value null.   
If the value is TEXT you are transcribing [...], copy it VERBATIM [...]; do   
NOT paraphrase, summarize, translate, shorten, re-order, or fix spelling   
/ spacing / punctuation.   
No explanation. JSON only.   
[appended only if a region a11y snippet exists:]

Accessibility tree for the visible region (structure only; everything outside   
your region is scrubbed):   
{a11y\_text}

Grounding model. The following prompts correspond to the Sonnet computer-use backend. The runtime supplies the operation, target description, and optional anchor through {action}, withholding Type.value and Press.key. The masked screenshot and manifest coordinates use a 1280 720 frame.

Grounding Model: System Prompt   
You ground one trusted semantic browser action into screen coordinates.   
You receive a TRUSTED VIEW in which every untrusted region is covered by a   
numbered black mask. The semantic action was already chosen by a protected   
transaction. Locate exactly the control specified by that action without   
changing its operation, target, or effect.   
[...] Do not infer the mask’s hidden content, resolve an underspecified   
target by guessing, or select a different masked unit. [...] If the action   
does not identify one control reliably, return not-found [...].   
Return exactly one JSON object and no prose:   
{"x": <integer viewport x>, "y": <integer viewport y>}   
[...] If the control cannot be located reliably, return:   
{"error":"not-found"}

Grounding Model: User-Message Template   
The attached image IS the current 1280x720 screen - you already have it.   
SEMANTIC ACTION   
{action}   
CONTENT-FREE MASK MANIFEST (boxes are in this 1280x720 frame)   
{marks}   
Locate exactly the control named by the semantic action. Using the computer   
tool, emit EXACTLY ONE left\_click on the CENTER of that control. Do NOT   
take a screenshot [...]. If the control is NOT visible on this screen, do   
not click - reply with the text {"found": false} and make no tool call.

## D. Evaluation Details

This appendix supplements Section 6 with details of the benchmark tasks, execution protocol, and isolated-query analysis.

## D.1. WebArena Tasks and Evaluation

WebArena (Zhou et al., 2024) provides self-hosted web applications and tasks specified by naturallanguage instructions. Agents complete tasks through browser interaction, and automated evaluators assess their final answers and the resulting application state. We use the original task instructions and evaluators without modification.

Applications and tasks. Our evaluation covers four applications. Postmill is a discussion forum with tasks involving posts, comments, voting, and user profiles. Shopping is an online storefront with tasks involving product search, reviews, purchases, and order history. Shopping Admin supports administrative tasks such as customer lookups, sales analysis, and catalog or order updates. GitLab provides repository-management tasks, including information retrieval and updates to issues, merge requests, and profiles.

From WebArena’s 812 tasks, we select 400 single-site tasks, with 100 per application. The selection ensures at least 30 tasks using answer-based string\_match evaluation per application and randomly fills the remaining positions. We use sampling seed 42 and retain the same task set across agents, model configurations, and repetitions.

Task evaluation. WebArena uses three kinds of checks. Answer-based checks compare the agent’s final response with reference answers using exact matching, required phrases, or benchmark-configured LLM-based matching. URL checks compare the final page URL with the required path and query parameters. Application-state checks inspect specified page content or DOM elements to verify the requested outcome, such as an updated profile. A task succeeds only when all its evaluation requirements are satisfied.

## D.2. Execution Protocol

We repeat each task five times per agent and model configuration. Before each run, the application container is recreated from its initial image, restoring the initial database state. Authentication is regenerated, and the run uses a fresh browser profile.

Each run has a 40-minute time limit. Secure-CUA permits up to 50 transaction attempts, including rejected transactions. Vanilla-CUA permits up to 50 browser-action turns, while CaMeL-CUA executes its whole-task program under a limit of 50 GUI actions and 100 quarantined queries. Thus, the interaction budget is counted according to each agent’s execution interface.

CaMeL-CUA generates one whole-task program without replanning. Secure-CUA generates a new transaction at each turn. After a transaction is rejected, the runtime records feedback, obtains a fresh observation, and proceeds to the next turn; it does not retry the transaction within the same turn. Runs terminate on completion, budget exhaustion, an unrecoverable error, or timeout.

## D.3. Access to Untrusted Information

We analyze the recorded query calls from 6,000 Secure-CUA runs: 100 tasks per application, four applications, three model configurations, and five repetitions. Table 6 reports the results by model and application.

Query frequency and volume. The fraction of transactions issuing queries is the number of recorded transactions containing at least one query call divided by the total number of recorded transactions. This denominator includes rejected and partially executed transactions. Mean queries per task is the total number of query calls divided by the number of runs. All issued query attempts are counted, including unsuccessful attempts and calls made during transactions that are subsequently rejected.

Return types and aggregation. We classify calls by their declared return type. Structured queries request Boolean, integer, floating-point, or enumeration values; string queries request free-form text. Their percentages are computed over all query calls. The All column pools the underlying counts

Table 6  Access to untrusted information. Transactions is the percentage issuing at least one query; Queries/task is the mean per run. Structured and String are percentages of query calls. All pools counts across the four applications. Percentages are rounded independently.
<table><tr><td>Model</td><td>Metric</td><td>Shopping</td><td>GitLab</td><td>Postmill</td><td>Shopping Admin</td><td>All</td></tr><tr><td rowspan="4">Opus</td><td>Transactions (%)</td><td>23.4</td><td>27.1</td><td>22.0</td><td>12.3</td><td>20.8</td></tr><tr><td>Queries/task</td><td>10.5</td><td>9.7</td><td>16.7</td><td>6.3</td><td>10.8</td></tr><tr><td>Structured (%)</td><td>74.2</td><td>72.4</td><td>24.6</td><td>24.3</td><td>47.4</td></tr><tr><td>String (%)</td><td>25.8</td><td>27.6</td><td>75.4</td><td>75.7</td><td>52.6</td></tr><tr><td rowspan="4">Gemini</td><td>Transactions (%)</td><td>23.1</td><td>25.4</td><td>21.8</td><td>11.8</td><td>20.2</td></tr><tr><td>Queries/task</td><td>10.2</td><td>23.1</td><td>15.5</td><td>4.6</td><td>13.4</td></tr><tr><td>Structured (%)</td><td>79.2</td><td>81.9</td><td>46.0</td><td>21.9</td><td>65.8</td></tr><tr><td>String (%)</td><td>20.8</td><td>18.1</td><td>54.0</td><td>78.2</td><td>34.2</td></tr><tr><td rowspan="4">GPT</td><td>Transactions (%)</td><td>22.4</td><td>23.4</td><td>18.4</td><td>10.6</td><td>18.1</td></tr><tr><td>Queries/task</td><td>9.0</td><td>8.6</td><td>11.7</td><td>6.5</td><td>8.9</td></tr><tr><td>Structured (%)</td><td>75.6</td><td>43.3</td><td>14.8</td><td>23.4</td><td>38.5</td></tr><tr><td>String (%)</td><td>24.4</td><td>56.7</td><td>85.2</td><td>76.5</td><td>61.5</td></tr></table>