# The Watchtower Imperative: Lifespan-Calibrated Evaluation for Frontier AI

Casey M. Williams University of Kansas casey.williams@ku.edu

Jennifer Victoria Scurrell<sup>∗</sup> University of Zurich jennifer.scurrell@ipz.uzh.ch

## Abstract

The gap between AI capability deployment and safety assurance is structural and cannot be closed through internal evaluations alone. Analyzing documented harms (2017– 2026) across the human lifespan reveals that disparate incidents — from voice assistants endangering children to AI companions implicated in adolescent suicide and automated care denials for the elderly — reflect a single failure: evaluating systems against an implicit adult user model. We introduce five developmental moderators structured into a Lifespan Vulnerability Matrix across six life stages. The matrix reveals a nonmonotonic risk distribution: vulnerability peaks at both ends of the lifespan, governance coverage is asymmetrical, and authority to convert model errors into real-world impact is scrutinized least. We then propose a set of lifespan design principles generalizing existing child-protective instruments, and an institution-agnostic, three-tier external evaluation framework — comprising automated simulation, expert red teaming, and post-deployment monitoring — indexed directly to matrix risk cells.

Keywords: AI safety, external red teaming, continuous monitoring, structured escalation, developmental psychology, lifespan development, cognitive aging, AI governance

Content note: this paper discusses suicide, child safety, CSAM, and elder exploitation in the context ofdocumented AI safety incidents.

## 1 Introduction

In December 2021, a voice assistant instructed a ten-year-old to touch a coin to a live plug [8]. In February 2024, a fourteen-year-old died by suicide after an AI companion cultivated dependency and ignored suicidal ideation [28]. In November 2023, an insurer was alleged to have used predictive algorithms to systematically deny post-acute care to elderly enrollees [23, 54]. In February 2024, a finance employee transferred \$25 million following a deepfake video conference with synthetic colleagues [17].

Though assigned to disparate literatures — child safety, adolescent mental health, healthcare accountability, and enterprise fraud — these incidents stem from a single structural error: general-purpose AI systems are evaluated against an implicit model of a cognitively typical adult and deployed to a population where that user is a minority. What varies is not system failure, but human developmental state.

Consequently, we argue that independent external AI safety evaluation must be organized around developmental state, which determines how system behaviors convert into injury. Even a rigorous, adversarial evaluation regime will miss most harms if calibrated solely to a modal adult.

Recent events highlight the urgency of this gap. In mid-2026, frontier models demonstrated autonomous capability to identify large numbers of zero-day vulnerabilities [3, 5]. The rapid deployment, suspension, and redeployment cycle that followed was governed by capability disclosures and trade policy [4], without independent evaluation bodies positioned to contest deployment decisions.

This paper proceeds as follows: Section 2 surveys the empirical record of AI harms (2017–2026) by developmental stage. Section 3 details the Lifespan Vulnerability Matrix, its five moderators, and a proposed set of lifespan design principles. Section 4 diagnoses why internal evaluations fail and examines cross-industry precedents. Section 5 presents a three-tier external governance framework. Sections 6–9 outline governance recommendations, address limitations, and conclude.

## 2 The Empirical Record

This section reviews the documented evidence of AI safety failures, tracing both the quantitative surge in reported incidents and their real-world impact. While safety analyses conventionally categorize failures by model behavior, evaluating harm requires examining human vulnerability. We first present macro-level metrics on incident growth before reorganizing the empirical record across human developmental stages.

## 2.1 The Acceleration of Documented Harm

Documented AI harm incidents have more than doubled since 2023. The Stanford AI Index recorded a 56.4% year-over-year increase for 2023–2024 (149 to 233 incidents) [58], followed by a further rise to 362 incidents in 2025 [59]. Over the same period CrowdStrike observed an 89% year-over-year increase in attacks conducted by AI-enabled adversaries [20].

These metrics represent severe underestimates due to commercial disincentives and high statutory reporting thresholds (e.g., the EU AI Act [25]; California SB-53 [13]). Rather than total volume, rising incident curves signal a widening gap between deployment pace and assurance capacity. Table 1 summarizes key indicators.

## 2.2 Harm Incidents by Developmental Stage

Conventional taxonomies categorize harms by failure mechanism (e.g., hallucination, bias). We reorganize incidents by human developmental stage, which dictates severity and risk exposure.

## 2.2.1 Early and Middle Childhood (0–12)

Children in this range cannot reliably evaluate conversational AI as a fallible, commercially motivated artifact. First-order theory-of-mind consolidates by roughly age five [69], but the capacity to model institutional and commercial intent behind a persuasive interface develops considerably later; existing child-protection instruments accordingly treat age-appropriate design as a producer obligation rather than a comprehension assumption [33, 35, 63]. Documented incidents include voice assistants instructing children to perform dangerous acts [8], insecure smart toys permitting unauthorized surveillance and communication [12], and chatbots lacking age-verification safeguards [27]. These failures are consistently detected reactively, after internal developer reviews fail to account for child developmental limitations.

Table 1: Indicators of the AI safety monitoring gap.
<table><tr><td>Metric</td><td>Finding</td><td>Source</td></tr><tr><td>Incident growth 2023→2024</td><td>+56.4% YOY</td><td>Stanford AI Index 2025 [58]</td></tr><tr><td>Incident growth 2024→2025</td><td>233 → 362</td><td>Stanford AI Index 2026 [59]</td></tr><tr><td>Attacks by AI-enabled adversaries</td><td>+89% YOY</td><td>CrowdStrike 2026 [20]</td></tr><tr><td>Vulnerabilities exploited before public disclosure</td><td>42%</td><td>CrowdStrike 2026 [20]</td></tr><tr><td>FDA-registered AI health tools with a</td><td>9%</td><td>Dolin et al. (2026) [22]</td></tr><tr><td>post-deployment surveillance plan Organizations actively monitoring production AI</td><td>&lt;50% (9% of small</td><td>Gradient Flow (2025) [30]</td></tr><tr><td>Teens (13–17) who have used AI companions</td><td>firms) 72%</td><td>Common Sense Media (2025)</td></tr><tr><td>Companion users spending at least as much time with AI as with friends</td><td>≈20%ª</td><td>[19] Common Sense Media (2025)</td></tr><tr><td>Adolescent depression per daily social media hour</td><td>+13% odds</td><td>[19] Liu et al. (2022) [36]</td></tr><tr><td>US ad revenue from users under 18, major social platforms (2022)</td><td>$11 billion</td><td>Raffoul et al. (2023) [50]</td></tr></table>

<sup>a</sup> Derived: 80% of companion users report spending more time with friends and 6% report the reverse; the residual reports parity. Denominator is companion users, not all teens.

## 2.2.2 Adolescence (13–17)

Adolescents face an asymmetry between peak reward sensitivity and maturing regulatory control [46, 60]. Engagement-optimized companion AIs have been alleged to cultivate dependency, discourage help-seeking, and supply suicidal ideation and method details, leading to fatalities and litigation [18, 28]. Further harms include chatbots advising minors on concealing alcohol and sexual activity [16], an eating-disorder support chatbot offering harmful weight-loss guidance [44], and recommendation algorithms rapidly delivering self-harm content [15, 31].

## 2.2.3 Emerging Adulthood (18–25)

Despite ongoing prefrontal maturation and skill formation, regulatory oversight terminates at age 18. Documented risks include chatbot-mediated radicalization preceding violent crime [9] and false academic integrity accusations from detection tools [68]. A third risk — erosion of professional judgment through premature reliance on automated outputs during early-career skill acquisition — follows from the cognitive offloading literature [53] but has not been systematically documented in this population, a gap we return to in Section 7.

## 2.2.4 Established Adulthood (26–64)

Adults possess authority to translate model errors into real-world consequences, exacerbated by automation bias [47] and Bainbridge’s irony [7]. Documented failures include legal briefs citing hallucinated precedents [29, 37], binding corporate liability for chatbot misrepresentations [38], false criminal allegations generated about a named individual [43], wrongful arrests via facial recognition errors [70, 71], systemic administrative fraud misdeterminations [55], batch adjudication of insurance claims without individual review [49], and large-scale financial fraud via deepfake video calls [17].

## 2.2.5 Later Adulthood (65+)

Older adults experience cognitive decline [56] and elevated risk of social isolation [32] while retaining financial and medical authority. Principal documented harms involve alleged algorithmic denial of post-acute care via predictive insurance software [23, 54, 67] and AI-enabled voice cloning and extortion fraud [26, 66], which exploit susceptibility patterns established in the pre-AI elder-fraud literature [21]. Conversational companion robotics have been deployed at state scale as a social isolation intervention [41]; whether such systems supplement or displace human contact is, in our view, an open empirical question that current deployment evaluations are not designed to answer.

## 2.2.6 Stage-Independent Harm Domains

Three harm domains affect all populations:

(1) Physical Safety: Navigation and autonomous driving failures causing fatalities and traffic disruption [1, 45].

(2) Adversarial Exploitation: Systematic prompt vulnerabilities, red-teaming failures, and automated espionage campaigns [6, 61, 62].

(3) Democratic Integrity: Deepfakes and AI disinformation targeting elections [1, 45].

Systemic market concentration [10, 11] amplifies these risks: model miscalibrations propagate simultaneously across all developmental stages.

## 3 The Developmental Basis

This section provides the theoretical foundation for developmental AI safety evaluation. Rather than viewing harm through isolated system failures, we establish a framework centered on human cognitive and social vulnerability. We first define five stage-varying moderators, map them across six lifespan stages in a Lifespan Vulnerability Matrix, and then propose a set of lifespan design principles generalizing existing child-protection instruments.

## 3.1 Five Developmental Moderators

Converting the empirical record into an evaluation specification requires constructs that are stageindependent in definition but stage-varying in value. We propose five moderators that determine how system behaviors convert into injury and how severely:

M1. Attribution capacity. The user’s ability to accurately model the system’s nature, limits, incentives, and epistemic status. Absent before theory-of-mind consolidates [69], near ceiling in healthy adulthood, and declining with cognitive aging. Low attribution converts fluency into misplaced trust. Candidate measures: calibration of user confidence against system accuracy; elicited beliefs about agency and memory.

M2. Self-regulatory capacity. The ability to disengage from reinforcing interactions. Suppressed in adolescence by reward–control asymmetry [60], highest in established adulthood, and declining in later life alongside broader executive change [56]. Low capacity converts engagement optimization into compulsive use. Candidate measures: session lengths, disengagement latency, return frequency.

M3. Skill-formation exposure. Whether an offloaded capability is under active acquisition, consolidated, or under maintenance. Offloading consolidated skills improves efficiency. We hypothesize that offloading skills under active formation produces developmental displacement: the offloading lit erature establishes the underlying mechanism in adult populations [53, 57], but has not been extended to users who are still acquiring the offloaded capability. We treat this as a measurement priority rather than an established finding (Section 8). Relevant during foundational learning, early-career skill acquisition, and cognitive maintenance in aging populations.

M4. Consequence authority. The real-world stakes a user can commit through an AI-mediated decision. Minimal in childhood, peaking in established adulthood, and remaining high for financial and medical decisions in later life. Consequence authority acts as a multiplier on factual error: identical hallucination rates produce vastly different harms depending on user authorization scope.

Table 2: The Lifespan Vulnerability Matrix. Listed moderators are those elevated at that stage; unlisted moderators are present but not governing.
<table><tr><td>Stage</td><td>Elevated moderators</td><td>Dominant harm vector</td><td>Detection requirement</td></tr><tr><td>Early childhood (0-7)</td><td>M1, M3</td><td>Misattributed agency; unsafe instruction acceptance; displacement of adult-mediated learning</td><td>Elicited beliefs on system nature; instruction-following under unsafe prompts; substitution metrics</td></tr><tr><td>Middle childhood (8-12)</td><td>M1, M3, M5</td><td>Cognitive offloading during foundational learning; algorithmic amplification of comparison content</td><td>Offloading events indexed to learning stage; exposure composition; comparison-content escalation</td></tr><tr><td>Adolescence (13-17)</td><td>M2, M3, M5</td><td>Engagement optimization against reward sensitivity; parasocial dependency; crisis mishandling</td><td>Disengagement latency; AI-to-human interaction ratio; behavior under distress; escalation reliability</td></tr><tr><td>Emerging adulthood (18–25)</td><td>M3, M4</td><td>Displacement of professional judgment; epistemic dependency prior uncertainty; trajectory of unaided task to independent competence</td><td>Verification behavior under model performance</td></tr><tr><td>Established adulthood (26–64)</td><td>M4</td><td>Automation bias at maximum decision authority; hallucination harm multiplied by authorization scope</td><td>Override/challenge rates on erroneous outputs; confidence calibration; unaided performance tracking</td></tr><tr><td>Later adulthood (65+)</td><td>M1, M2, M4, M5</td><td>Declining attribution capacity with high decision authority; synthetic-media susceptibility; social substitution</td><td>Calibration drift; response to persuasive/authority prompts; substitution ratios; capacity-sensitive consent</td></tr></table>

M5. Social substitution risk. The likelihood that AI interactions displace human relationships rather than supplementing them. U-shaped across the lifespan: elevated in adolescence during identity formation [19, 46] and in later adulthood due to social network contraction [14, 32]. Candidate measures: longitudinal AI-to-human interaction ratios; displacement of reported contacts.

## 3.2 The Lifespan Vulnerability Matrix

Table 2 maps these five moderators across six developmental stages, linking elevated moderators to dominant harm vectors and evaluation requirements.

Two distribution features carry major governance implications:

Non-monotonic risk distribution. Four of five moderators are elevated in later adulthood — more than any other stage. However, current governance lacks elder-specific protective instruments, high-risk classifications, or test batteries calibrated to declining attribution capacity.

Inverse correlation of authority and scrutiny. Established adulthood carries the highest decision authority yet receives the least developmental scrutiny. Its dominant harms stem from automation bias and authorization scope rather than individually unsafe outputs, rendering them invisible to content-based safety rubrics.

## 3.3 Toward Lifespan Design Principles

Substantive protective obligations already exist in binding and quasi-binding form, but almost exclusively for children. The UK Age Appropriate Design Code [35], IEEE 2089-2021 [33], General Comment No. 25 to the Convention on the Rights of the Child [63], and UNICEF’s guidance on AI and children [64] converge on a common set of commitments: design calibrated to developmental state, non-exploitation of immature reward systems, protection of identity formation, and meaningful supported control.

Table 3: Proposed Developmental Harm (DH) principles. Each generalizes an existing child-protective obligation into a lifespan form indexed to a moderator or governance criterion. The set is initial and extensible.
<table><tr><td></td><td>Proposed lifespan principle</td><td>Basis and index</td></tr><tr><td>DH1</td><td>Design calibrated to developmental state across the entire lifespan</td><td>Generalizes age-appropriate design [33, 35]; indexes all moderators</td></tr><tr><td>DH2</td><td>Do not exploit affective or reward systems where self-regulation is reduced</td><td>Generalizes neurodevelopmental protection [63]; indexes M2</td></tr><tr><td>DH3</td><td>Do not offload capabilities that the user is actively acquiring or maintaining</td><td>Extends educational protection duties [64]; indexes M3</td></tr><tr><td>DH4</td><td>Scale verification friction to the consequence authority being exercised</td><td>No child-specific antecedent; indexes M4</td></tr><tr><td>DH5</td><td>Protect identity and social cohesion where substitution risk is elevated</td><td>Generalizes identity-formation protection [63]; indexes M5</td></tr><tr><td>DH6</td><td>Verify supported decision-making where attribution capacity is reduced</td><td>Generalizes caregiver control [35, 63]; indexes M1</td></tr><tr><td>DH7</td><td>Provide third-party verifiable safety attestation</td><td>Governance criterion; motivated by liability for deployed outputs [38]</td></tr><tr><td>DH8</td><td>Mitigate continuously post-deployment rather than at episodic pre-deployment checkpoints</td><td>Governance criterion; see Section 5.4</td></tr><tr><td>DH9+</td><td>(open)</td><td>To be specified as moderators and agentic harm classes are identified</td></tr></table>

Our claim is that these commitments are not properly child-specific. They are lifespan principles that were codified first for children because that is where the evidentiary case was clearest and the political constituency strongest. Each maps onto a moderator that varies continuously across the lifespan rather than switching off at a statutory age boundary.

We therefore propose the Developmental Harm (DH) principles set out in Table 3: a generalization of these instruments into stage-agnostic, moderator-indexed obligations. Three properties are worth stating explicitly.

The set is a proposal, not a codification. We are not restating an existing enumerated standard. Table 3 is our synthesis, offered as a starting point for standards development rather than as a finished instrument.

The set is deliberately non-exhaustive and open to extension. Each principle is anchored to a moderator or to a governance criterion we can currently defend. Moderators we have not identified, or harm classes that emerge from agentic deployment (Section 8), should generate additional principles. We expect DH9 onward to be written by others, and we regard extensibility as a design requirement rather than an admission of incompleteness.

The set requires no new statutory categories. Because each principle generalizes an obligation already enacted somewhere for children, adoption is a question of extending scope, not of establishing an unprecedented duty — a materially lower regulatory bar.

DH4 warrants comment: it has no child-specific antecedent, precisely because consequence authority is minimal in childhood. It is visible only once the matrix is read across the full lifespan — which is the argument for constructing the matrix in the first place.

## 4 Why Internal Evaluation Cannot Close the Gap

Self-evaluation by developer organizations is structurally incapable of ensuring safety. Commercial incentives, evaluator bias, and episodic testing create systematic blind spots that internal programs cannot overcome. This section diagnoses these three structural failures and examines precedents from

high-stakes industries — aviation, pharmaceuticals, and financial auditing — that demonstrate why independent external oversight is mandatory.

## 4.1 Three Structural Failures

Systemic safety deficits stem directly from organizational and incentive architectures rather than individual incompetence.

Incentive misalignment. Organizations evaluating safety hold commercial incentives to deploy. Evaluations are scoped to confirm launch readiness rather than maximize failure discovery, compressing adversarial timelines under competitive pressure [10, 11]. Developmental harms are deferred most under this architecture because they surface slowly and attribute poorly — a dynamic recurring in automotive, pharmaceutical, and financial sectors.

The familiarity blind spot. Internal red teams, composed predominantly of adult technical professionals, test for failures conceivable from their own developmental position. However, an adversarial prompt recognized as manipulation by a security researcher may appear as benign conversation to a child, a negotiable boundary to an adolescent, or incomprehensible to an impaired older adult [2]. Internal teams operating under deployment pressure lack structural motivation to integrate clinical, developmental, or geriatric expertise.

The coverage gap. Pre-deployment testing is episodic, whereas deployed models interact with evolving populations, environments, and adversarial tactics. CrowdStrike reports that adversaries injected malicious prompts into production generative AI tools at more than ninety organizations during 2025, and that 42% of exploited vulnerabilities were weaponized before public disclosure [20]: both are post-deployment phenomena invisible to pre-release testing. Monitoring practice has not kept pace. Fewer than half of organizations actively monitor production AI systems for accuracy, drift, and misuse, falling to 9% among small firms [30]; in the more heavily regulated medical device context, only 9% of FDA-registered AI-based tools include a post-deployment surveillance plan [22]. For developmental harms — parasocial dependency, skill displacement, calibration drift — harm accumulates across months of interaction and is undetectable in principle via static pre-release content rubrics.

## 4.2 Cross-Industry Governance Precedent

Mandating independent external oversight for safety-critical systems subject to commercial deployment pressure is standard practice across high-risk sectors:

• Aviation rejects manufacturer self-attestation, requiring independent type certification and continuing airworthiness management across the operational lifecycle [34].

• Pharmaceutical post-market surveillance mandates ongoing pharmacovigilance to track longterm adverse effects across diverse populations that pre-approval trials miss [24] — the closest analogue to developmental harm tracking.

• Financial auditing mandates independent attestation because internal reporting incentives conflict with market transparency needs [48, 65].

Anthropic’s Project Glasswing [3] demonstrates the feasibility of continuous external adversarial evaluation and coordinated disclosure. However, its published scope remains limited to a small number of security-focused organizations [3], excluding clinical and developmental expertise. It establishes proof-of-concept for external oversight, but not broad ecosystem coverage.

## 5 A Three-Tier Framework, Indexed by Developmental Stage

This section specifies an external evaluation architecture designed to close the developmental safety gap. We establish core design criteria and an indexing principle that anchors evaluation to the Lifespan Vulnerability Matrix rather than testing methods alone. We then detail a three-tier operational framework — automated simulation, expert red teaming, and post-deployment monitoring — followed by structured escalation pathways and independence safeguards.

## 5.1 Design Criteria and the Indexing Principle

The evaluation framework satisfies five design criteria: institutional independence, operational continuity, breadth and depth, developmental specificity, and structured escalation.

Prior work has proposed layered technical and procedural designs for auditing large language models [39] and end-to-end frameworks for internal algorithmic audit [51]. Our contribution is orthogonal to both: we specify what such audits must cover, indexed to the developmental state of the user, and we place the audit function outside the deploying organization.

Crucially, evaluation instruments are indexed to matrix risk cells rather than organized by method alone. Method-based tiering allows regimes to execute tests while leaving entire developmental profiles unassessed. Under our model, coverage is defined as the proportion of matrix cells with an assigned instrument, a defined baseline, and an escalation threshold — providing an auditable metric usable for regulatory oversight, procurement, or underwriting.

## 5.2 Tier 1: Automated Adversarial Simulation

Synthetic agent populations continuously probe deployed systems across an adversarial matrix. Personas are specified by moderator profiles (e.g., low attribution capacity coupled with high consequence authority) rather than raw demographic tags. By including child, adolescent, and older-adult profiles as primary test cases, Tier 1 provides population-scale statistical coverage to detect low-frequency failure modes and generate performance baselines for Tier 3 tracking.

## 5.3 Tier 2: Expert Human Red Teaming

Human evaluation addresses nuanced harms outside predefined algorithmic rubrics. To resolve the familiarity blind spot, red team panel composition is directly informed by matrix requirements: child and developmental psychologists (Stages 1–3), human-factors specialists (Stages 4–5), and geriatric cognition experts (Stage 6), alongside offensive security analysts across all stages. Operating via triage, Tier 2 experts investigate and contextualize candidate signals flagged by Tier 1.

## 5.4 Tier 3: Continuous Post-Deployment Monitoring

Adapting the pharmacovigilance model [24], Tier 3 monitors production behavior using developmental cohort-stratified baselines. Aggregate baselines obscure population-specific anomalies (e.g., session-length spikes indicating healthy engagement in adults vs. parasocial dependency in adolescents). Continuous monitoring is uniquely capable of observing accumulating longitudinal harms — including dependency, social substitution, skill displacement, and calibration drift.

## 5.5 Structured Escalation

Findings escalate based on moderator-weighted severity rather than raw incident counts. A single high-severity incident (e.g., low attribution capacity paired with high consequence authority) triggers escalation where high volumes of low-moderator events would not. Findings route to deploying organizations, sector regulators, and national cybersecurity bodies, bridging the gap between monitoring and regulatory enforcement.

## 5.6 Institutional Form and Independence Safeguards

While compatible with various structures — such as independent standards nonprofits, accredited auditing firms, or government evaluation bodies — the framework strictly mandates financial and organizational independence. Evaluators financially dependent on deploying entities replicate internal conflicts of interest. Credibility requires formal accreditation, conflict prohibitions, and enforceable reporting mandates established prior to deployment.

## 6 Governance Recommendations

Translating the Lifespan Vulnerability Matrix, the DH principles, and the three-tier framework into policy requires action across the AI governance ecosystem. This section outlines concrete recommendations for regulatory bodies, standards organizations, frontier AI developers, and institutional investors.

Regulatory bodies — including the EU AI Office, whose high-risk system obligations take effect in August 2026 [25], the US AI Safety Institute, CISA, and sector regulators — should replace binary age gating with developmental risk classification, classifying elder- and impairment-facing AI as high-risk by default. Incident reporting thresholds must adopt epidemiological models that aggregate low-severity signals to detect accumulating developmental harms before catastrophic events occur. Frontier models with autonomous exploitation capabilities warrant mandatory coordinated disclosure (analogous to the CVE process), while accreditation standards for external evaluators must mandate clinical and developmental expertise.

Standards bodies (e.g., ISO, NIST, IEEE) must develop validated, age-stratified test instruments for the constructs in Table 2, such as attribution capacity, dependency, and skill displacement. The DH principles proposed in Table 3 are offered as a starting point for that work and are explicitly open to extension; IEEE 2089 [33] provides an existing procedural template whose scope could be widened beyond children. Extending frameworks like the NIST AI RMF [42] into operational, developmentally stratified test protocols represents the highest-leverage standards contribution available.

AI developers should adopt matrix coverage as an operational deployment requirement rather than a compliance posture, implementing the DH principles through external testing rather than internal checklists. Initiatives like Project Glasswing should be expanded in scope and multidisciplinary composition.

Boards, investors, and insurers should require audited evidence of independent developmental coverage as a core governance control. Emerging litigation regarding addictive design [40], corporate liability for AI outputs [38], and automated care denials [23, 67] demonstrates that lifespan AI liability is a material risk. Matrix coverage provides an auditable, comparable metric directly usable for ESG disclosures, due diligence, and insurance underwriting.

## 7 Discussion and Limitations

This section outlines methodological constraints and scope boundaries of the proposed framework. While independent evaluation shifts the failure distribution toward lower-severity events, it does not eliminate risk. Below, we address central limitations across causal attribution, construct validity, evaluator incentives, alignment scope, and empirical selection bias.

Causal attribution and cohort confounding. Isolating AI-attributable harm from broader environmental, socio-economic, and educational influences remains a methodological challenge. Furthermore, longitudinal monitoring risks confounding developmental age effects with generational cohort effects (e.g., varying lifetime technology exposure across generations), requiring designs that systematically decouple them.

Construct validity. The five moderators and six stages are grounded in developmental science but represent hypothesis structuring instruments rather than fully validated parameterizations. Their factor structure, independence, and measurement validity in human-AI interaction require ongoing empirical testing. The same caution applies to the DH principles: they are a proposed synthesis, and the mapping from principle to moderator is an argument we advance rather than a validated correspondence.

Evaluator incentive risks. External evaluation bodies remain vulnerable to commercial capture. Mitigating conflicts of interest requires institutional safeguards adapted from financial auditing, such as mandatory evaluator rotation, whistleblower protections, and independent quality oversight [65]. We note the countervailing caution that formal audit regimes can decay into ritualized verification that produces assurance documents without producing assurance [48]; an evaluation regime indexed to auditable coverage metrics is intended in part to resist that failure mode.

Scope boundaries. The framework evaluates safety at the behavioral and output level; it serves as a necessary complement to, rather than a substitute for, foundational alignment research.

Empirical reporting biases. The incident record exhibits strong selection bias — overrepresenting high-severity cases from large organizations in high-income, Western contexts. Stages with sparse incident documentation (e.g., emerging adulthood) reflect an absence of systematic monitoring rather than an absence of risk. The global distribution of AI harm is broader and more concentrated in vulnerable populations than currently documented.

## 8 Future Work

Future research must validate the developmental evaluation framework, formalize measurement standards, and address emerging technical challenges. We identify four priority avenues for empirical, methodological, policy, and cross-cultural investigation.

Empirical validation. Research must verify whether each tier reliably surfaces distinct failure classes and quantify detection rate differentials between internal-only and hybrid external evaluation regimes. The M3 displacement hypothesis requires direct test in populations acquiring the offloaded capability, rather than extrapolation from adult offloading studies.

Measurement standardization. Interdisciplinary efforts across clinical psychology, developmental science, gerontology, and HCI are needed to establish validated operational metrics for the five moderators, and to extend the DH principles beyond the initial set proposed here.

Political economy of certification. Investigating the market and regulatory conditions — such as liability law, procurement mandates, insurance underwriting, and network dynamics — that drive adoption of independent safety evaluation.

Cross-cultural generalizability and agentic AI. Testing the validity of the matrix’s developmental constructs across non-Western populations and diverse regulatory jurisdictions. Crucially, extending the framework to autonomous agentic systems represents the most urgent technical challenge; agentic deployment fundamentally transfers consequence authority (M4) from the user directly to the system, and is the most likely source of DH principles beyond those proposed here.

## 9 Conclusion

This paper has argued that safety deficits in frontier AI are structural rather than procedural. Closing the assurance gap requires shifting from developer self-assessment to independent, external evaluation indexed directly to human developmental vulnerability.

The empirical record demonstrates that isolated product harms across the lifespan reflect a single underlying failure: systems are evaluated against an implicit model of a cognitively typical adult, ignoring stage-varying human vulnerability. Consequently, vulnerability peaks at both ends of the lifespan while oversight remains asymmetrical, leaving high-authority adult decisions largely unscrutinized and vulnerable populations exposed.

The operational tools — automated simulation, expert red teaming, and continuous post-deployment monitoring — already exist in prototype. What remains missing is their systematic integration into an independent governance architecture indexed to the developmental state of the user. Independent external evaluation is a structural necessity; the imperative now is how rapidly institutions can build the capacity to execute it.

## Acknowledgments

The authors were members of OpenAI’s External Red Teaming Network and want to thank OpenAI for foundational insights into adversarial evaluation methodology. The authors have no financial interests to disclose that bear on the governance arguments presented here.

## Ethics and Privacy Statement

This paper discusses documented, publicly reported AI harm incidents, including cases involving suicide, child safety, CSAM, and elder exploitation, solely for the purpose of empirical governance analysis; no new or previously undisclosed harmful content is introduced. Individuals in cited incidents are identified only where their involvement is already a matter of public legal or journalistic record, and minors are not named. Allegations in pending or concluded litigation are described as allegations. The authors acknowledge that the empirical record carries documented selection and reporting biases, discussed in Section 7, and that any governance framework derived from it should be applied with attention to populations underrepresented in that record.

## References

[1] AI Incident Database. 2025. Incident Database. Responsible AI Collaborative. https:// incidentdatabase.ai

[2] American Psychological Association. 2025. Health advisory: artificial intelligence and adolescent wellbeing. (June 2025).

[3] Anthropic. 2026. Project Glasswing: securing critical software for the AI era. https://www. anthropic.com/glasswing

[4] Anthropic. 2026. Statement on Fable and Mythos access. https://www.anthropic.com/news/ fable-mythos-access

[5] Anthropic Frontier Red Team. 2026. Claude Mythos. https://red.anthropic.com/2026/ mythos-preview

[6] Sabin, Sam. The Age of AI-Powered Cyberattacks is here. Axios, 2025.

[7] L. Bainbridge. 1983. Ironies of automation. Automatica 19, 6 (1983), 775–779.

[8] BBC News. 2021. Alexa tells 10-year-old girl to touch live plug with penny. (December 28, 2021).

[9] BBC News. 2023. Windsor Crossbow Intruder Sent Sex Texts to Chatbot. (October 5, 2023).

[10] Y. Bengio et al. 2025. International AI Safety Report 2025. Commissioned by 30 nations.

[11] Y. Bengio et al. 2026. International AI Safety Report 2026. (February 3, 2026).

[12] Bundesnetzagentur. 2017. Prohibition of the connected doll “My Friend Cayla” as an unlawful transmitting device. (February 2017).

[13] California State Legislature. 2025. Senate Bill 53: Transparency in Frontier Artificial Intelligence Act.

[14] L. L. Carstensen. 2006. The influence of a sense of time on human development. Science 312, 5782 (2006), 1913–1915.

[15] Center for Countering Digital Hate. 2022. Deadly by Design. (December 2022).

[16] Center for Countering Digital Hate. 2023. Assessment of Snapchat My AI responses to simulated minor accounts.

[17] CNN Business. 2024. Finance worker pays out \$25 million after video call with deepfake “chief financial officer.” (February 2024).

[18] CNN Investigates. 2025. ‘You’re not rushing. You’re just ready:’ Parents say ChatGPT encouraged son to kill himself. (November 20, 2025).

[19] Common Sense Media. 2025. Talk, Trust, and Trade-Offs: How and Why Teens Use AI Companions. Survey of 1,060 US teens aged 13–17 conducted by NORC, April–May 2025.

[20] CrowdStrike. 2026. Global Threat Report 2026. (February 24, 2026).

[21] M. DeLiema. 2018. Elder fraud and financial exploitation: application of routine activity theory. The Gerontologist 58, 4 (2018), 706–718.

[22] P. Dolin, W. Li, G. Dasarathy, and V. Berisha. 2026. Statistically valid post-deployment monitoring should be standard for AI-based digital health. Advances in Neural Information Processing Systems 38.

[23] Estate of Lokken v. UnitedHealth Group. 2023. Complaint filed D. Minn., November 2023.

[24] European Medicines Agency. 2017. Guideline on Good Pharmacovigilance Practices (GVP), Module IX: Signal Management (Rev. 1).

[25] European Parliament. 2024. Regulation (EU) 2024/1689: The AI Act. Official Journal of the European Union.

[26] Federal Bureau of Investigation, Internet Crime Complaint Center. 2024. Elder Fraud Report.

[27] Garante per la protezione dei dati personali. 2023. Emergency order restricting processing by Replika. (February 2, 2023).

[28] Garcia v. Character Technologies, Inc. 2024. Complaint filed M.D. Fla., October 2024.

[29] Gauthier v. Goodyear Tire & Rubber Co. 2024. E.D. Tex., November 2024.

[30] Gradient Flow. 2025. 2025 AI Governance Survey.

[31] HM Coroner, North London. 2022. Inquest into the death of Molly Russell: prevention of future deaths report. (October 2022).

[32] J. Holt-Lunstad et al. 2015. Loneliness and social isolation as risk factors for mortality: a meta-analytic review. Perspectives on Psychological Science 10, 2 (2015), 227–237.

[33] IEEE. 2021. IEEE 2089-2021: Standard for an Age Appropriate Digital Services Framework Based on the Rights Principles for Children. IEEE Standards Association.

[34] International Civil Aviation Organization. 2018. Annex 8 to the Convention on International Civil Aviation: Airworthiness of Aircraft (12th ed.).

[35] Information Commissioner’s Office. 2020. Age Appropriate Design: A Code of Practice for Online Services. United Kingdom.

[36] M. Liu et al. 2022. Time spent on social media and risk of depression in adolescents: a dose-response meta-analysis. Int. J. Environ. Res. Public Health 19 (2022), 5164.

[37] Mata v. Avianca, Inc. 2023. S.D.N.Y., sanctions order, June 2023.

[38] Moffatt v. Air Canada. 2024. British Columbia Civil Resolution Tribunal, February 2024.

[39] J. Mökander, J. Schuett, H. R. Kirk, and L. Floridi. 2023. Auditing large language models: a three-layered approach. AI and Ethics 3 (2023), 1315–1339.

[40] Multistate attorneys general coalition. 2023. Complaint against Meta Platforms regarding design features directed at minors. (October 2023).

[41] New York State Office for the Aging. 2022. Deployment of conversational companion technology as a social isolation intervention.

[42] NIST. 2023. AI Risk Management Framework (AI RMF 1.0).

[43] noyb. 2025. AI hallucinations: ChatGPT created a fake child murderer. (March 2025).

[44] NPR. 2023. Eating disorder helpline takes down chatbot after it gave harmful advice. (June 2023).

[45] OECD. 2025. AI Incidents Monitor. https://oecd.ai/en/incidents

[46] A. Orben et al. 2022. Windows of developmental sensitivity to social media. Nature Communications 13, 1 (2022), 1649.

[47] R. Parasuraman and D. H. Manzey. 2010. Complacency and bias in human use of automation: an attentional integration. Human Factors 52, 3 (2010), 381–410.

[48] M. Power. 1997. The Audit Society: Rituals ofVerification. Oxford University Press.

[49] M. Miller, D. Armstrong, and P. Rucker. 2023. How Cigna saves millions by having its doctors reject claims without reading them. ProPublica / The Capitol Forum. (March 25, 2023).

[50] A. Raffoul, Z. J. Ward, M. Santoso, J. R. Kavanaugh, and S. B. Austin. 2023. Social media platforms

generate billions of dollars in revenue from US youth: Findings from a simulated revenue model. PLOS ONE 18, 12 (2023), e0295337.

[51] I. D. Raji et al. 2020. Closing the AI accountability gap: defining an end-to-end framework for internal algorithmic auditing. In ACM FAccT.

[52] A. Remmel. 2026. What the numbers show about AI’s harms. Time Magazine. (January 19, 2026).

[53] E. F. Risko and S. J. Gilbert. 2016. Cognitive offloading. Trends in Cognitive Sciences 20, 9 (2016), 676–688.

[54] C. Ross and B. Herman. 2023. Denied by AI: how Medicare Advantage plans use algorithms to cut off care for seniors in need. STAT News. (March 2023).

[55] Royal Commission into the Robodebt Scheme. 2023. Final Report. Commonwealth of Australia. (July 2023).

[56] T. A. Salthouse. 2009. When does age-related cognitive decline begin? Neurobiology ofAging 30, 4 (2009), 507–514.

[57] B. Sparrow, J. Liu, and D. M. Wegner. 2011. Google effects on memory: cognitive consequences of having information at our fingertips. Science 333, 6043 (2011), 776–778.

[58] Stanford HAI. 2025. Artificial Intelligence Index Report 2025. Stanford University.

[59] Stanford HAI. 2026. Artificial Intelligence Index Report 2026. Stanford University.

[60] L. Steinberg. 2008. A social neuroscience perspective on adolescent risk-taking. Developmental Review 28, 1 (2008), 78–106.

[61] TechRadar. 2025. Researcher tricks ChatGPT into revealing security keys – by saying “I give up”. (July 11, 2025).

[62] UK AI Safety Institute. 2024. Agent Red-Teaming Challenge Report. Gray Swan Arena Platform.

[63] UN Committee on the Rights of the Child. 2021. General Comment No. 25 (2021) on Children’s Rights in Relation to the Digital Environment. CRC/C/GC/25.

[64] UNICEF Innocenti. 2025. Policy Guidance on AI and Children, version 3.0.

[65] US Congress. 2002. Sarbanes-Oxley Act of 2002. Pub. L. No. 107-204, 116 Stat. 745.

[66] US Senate Committee on the Judiciary. 2023. Testimony on AI voice cloning and extortion. (June 2023).

[67] US Senate Permanent Subcommittee on Investigations. 2024. Report on algorithmic denial practices in Medicare Advantage. (October 2024).

[68] The Washington Post. 2023. We tested a new ChatGPT-detector for teachers. It flagged an innocent student. (2023).

[69] H. M. Wellman, D. Cross, and J. Watson. 2001. Meta-analysis of theory-of-mind development: the truth about false belief. Child Development 72, 3 (2001), 655–684.

[70] Williams v. City of Detroit. Wrongful arrest following facial recognition misidentification, January 2020.

[71] Woodruff v. City of Detroit. Wrongful arrest following facial recognition misidentification, February 2023.