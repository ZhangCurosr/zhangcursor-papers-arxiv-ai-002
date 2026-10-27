# Safe by Benchmark, Unsafe in Practice: Lessons from 1331 Child-Related AI Incidents

Junfeng Jiao<sup>1</sup>, Kevin Chen<sup>1</sup>, Shrestha Mishra<sup>1,\*</sup>, Abhejay Murali<sup>1</sup>, David Atkinson<sup>2</sup>, Shriti Raj<sup>3</sup>,

and Saleh Afroogh<sup>1,†</sup>

<sup>1</sup>The University of Texas at Austin, USA

<sup>2</sup>Allen Institute for AI, USA

<sup>3</sup>Stanford University, USA

## Abstract

AI systems are increasingly present in children’s everyday environments—education, entertainment, social networks, and toys. Their safety is typically assessed at the model level, through benchmarks, red-teaming, and content filtering. However, most AI harm taxonomies lack child-specific evidence and under-specify developmental harm and the stakeholder asymmetries unique to minors. Building on prior taxonomies, we construct a child-centered harm taxonomy mapped to sociotechnical failure modes. We curate and code 1331 publicly reported child-related AI incidents, annotating internal and external harm types and the failure mode implicated in each incident. Our analysis reveals which harms are most prevalent in public reporting, how individual-level and population-level harms relate, and which failures are most often coded alongside harm. We find that harm concentrates in representational, dignity-related, and psychological categories, and co-occurs most often with deploymentcontext, governance, and misuse failures rather than model defects. Harm to an individual child is also frequently linked to population-level harm, reaching children who never interacted with the system. These findings suggest that modellevel evaluations, while necessary, cannot by themselves anticipate the harm pathways most often documented for children. We conclude with actionable implications emphasizing mitigations beyond model-level safeguards such as content filtering: age assurance, consent verification for depiction, safety-by-design UX constraints, incident disclosure norms, and auditing regimes targeting developmental risk. Privacy-minimized annotations, a codebook, and documentation are available at https://github.com/The-Responsible-AI-Initiative/KidAID.

## Introduction

AI systems are rapidly moving from general-purpose tools into child-facing settings: tutoring chatbots, creative assistants, “AI friends,” voice assistants, social platforms, and networked toys. This expansion creates tangible benefits in personalized learning support, accessibility, entertainment, but also introduces a distinct harm landscape for minors. Children differ from adults in cognitive development, risk perception, susceptibility to persuasion, and dependence on guardians and institutions (UNICEF 2021; Weidinger et al.

2023). As a result, harms that are rare or low-severity in adult contexts can become high-severity in child contexts, especially when systems shape emotions, relationships, or identity formation. These differences map onto specific affordances and failure modes. Susceptibility to persuasion meets anthropomorphic conversational design, the setting for overreliance failures and dependency harms. Limited capacity to consent meets generative depiction, the setting for sexualization harms. Deference to perceived authority meets systems framed as tutors or experts, where unsafe advice is more likely to be followed than it would be by an adult. Dependence on caregivers and institutions means the harmed child is frequently not the party who accepted terms, configured the system, or can seek recourse. This mapping structures the taxonomy that follows.

The public debate on the risk of AI often divides between long-range catastrophic concerns and current harm (Bostrom 2014; Hendrycks, Mazeika, and Woodside 2023; Bender et al. 2021; Crawford 2021; Birhane, Prabhu, and Kahembwe 2022). For children, present-day harms are already prominent and measurable: exposure to sexual content, grooming or manipulation, encouragement to selfharm, parasocial dependency, privacy leakage, and coercive monetization patterns (Common Sense Media 2024; Livingstone and Stoilova 2021; Internet Watch Foundation 2024; UNICEF 2021). Despite growing attention, child AI safety research remains fragmented between (i) parental concern surveys, (ii) lab-style safety evaluations, (iii) platform policy documents, and (iv) isolated high-profile incidents. This fragmentation makes it difficult to prioritize mitigations because stakeholders lack empirically grounded answers to two operational questions. RQ1 (What): What is the nature and prevalence of child-specific AI harms observed in practice, at both the individual and the population level? RQ2 (How): Which sociotechnical failure modes are most often coded alongside child AI harms, and at what level ofthe sociotechnical system do they occur? (Anderljung et al. 2023; European Union 2024; National Institute of Standards and Technology 2024).

Existing taxonomies of AI harms and failures provide critical vocabulary but typically treat “psychological harm” as a broad bucket, under-represent child-specific threats (grooming, sexualization, coercive roleplay), and lack datasets mapping harms to failure modes and stakeholders in real inci-

dents.

To address these gaps, we contributed a child-centered incident dataset and a taxonomy grounded in empirical evidence. Our approach mirrors incident-analysis work in other safety-critical fields: we treat each publicly reported childrelated AI event as a case, annotate what happened at both the individual and population level and how the sociotechnical system failed, then quantify prevalence patterns and map harm-to-failure pathways to highlight leverage points for mitigation.

## Related Work

The AI systems used around children are at the intersection of technical risk, governance, and developmental vulnerability. Recent work on generative AI has emphasized that harm is not produced only by the capability of the model, but also by human interaction and the broader context of deployment (Weidinger et al. 2023). This sociotechnical framework is especially important for children with AI, where harms often emerge through repeated interaction, authority signals, or emotional dependence.

Incident-based analysis offers a complementary empirical base. The AI Incident Database was created to catalog realworld failures and support organizational learning (McGregor 2021), and recent incident-mapping work on generative AI extends this approach by jointly analyzing what harms occur, who is harmed, and how the underlying failures arise in practice (Li et al. 2025). Our study follows this incidentanalytic logic but applies it to child-related AI harms. We are especially inspired by Li et al. (2025), which demonstrates how incident-based analysis can operationalize accountability by jointly modeling what happened, who was harmed, and how harms emerged. Following this ethos, we curate child-related incidents from AIAAIC, AIID, and OECD.AI (AIAAIC 2025; McGregor 2021; OECD 2025).

UNICEF’s policy guidance emphasizes that AI governance should reflect children’s developmental needs, evolving capacities, dependency relations, and rights-based protections (UNICEF 2021). Governance frameworks such as the NIST Generative AI Profile and the OECD AI principles stress lifecycle risk management, transparency, and human-centered safeguards (National Institute of Standards and Technology 2024; OECD 2019).

## Taxonomies of AI Risks and Harms

Risk and harm taxonomies standardize the way stakeholders describe negative outcomes, enabling measurement, auditing, and governance (Shelby et al. 2023; Weidinger et al. 2023). Prior work has proposed broad harm categories spanning physical safety, privacy, civil liberties, misinformation, economic harm, psychological harm, and societal impacts, with generative AI-specific taxonomies further emphasizing modality-driven risks (e.g., deepfake impersonation) and systemic impacts. Governance-facing frameworks (e.g., the NIST AI Risk Management Framework) encourage practitioners to map risks to concrete controls across design, development, deployment, and monitoring.

Child safety requires finer granularity than general adult taxonomies. In child contexts, hazards like sexual content are not simply “hazardous content exposure”; they include sexualization of minors, grooming, coercive roleplay, and depiction/consent violations whose severity depends on developmental stage and asymmetric power dynamics. Likewise, “psychological harm” must represent developmentally specific mechanisms (e.g. persuasion susceptibility, attachment formation, authority compliance, and dependence). Our taxonomy extends general risk vocabularies by adding child-specific harm types and by separating individual-level harms from the population-level harms that reach children who never interacted with the system (e.g., non-interacting child victims depicted in synthetic media).

## Sociotechnical Failure Modes and Safety Engineering Views

We use three terms that carry much of the paper’s conceptual weight. Failure mode is inherited from safety engineering, where it denotes the specific manner in which a system departs from intended function (Leveson 2012). Sociotechnical is adapted from prior harm taxonomies, which argue that harm arises jointly from model capability, human interaction, and deployment context rather than from model outputs alone (Weidinger et al. 2023; Shelby et al. 2023). Harm-tofailure pathway is our own term, defined operationally as the co-occurrence, within a single incident, of a harm code and its primary failure-mode code; it asserts documented association, not causal mechanism. Safety engineering concepts motivate mapping harms to the proximal mechanisms associated with them. Work on AI failures distinguishes technical failures (model hallucination, robustness gaps, security vulnerabilities) from sociotechnical issues (misaligned incentives, misuse, weak disclosure practices, governance failures) (Weidinger et al. 2023). Governance frameworks urge practitioners to treat AI as a lifecycle-managed sociotechnical system where failures can be introduced during data collection, model development, product design, deployment configuration, and post-deployment operations.

Child harms frequently manifest as multi-factor pathways: weak or bypassable age gates, engagement-optimized recommender incentives, persuasive conversational UX patterns, and missing crisis escalation procedures can jointly produce severe outcomes. These compounded pathways motivate explicit failure-mode coding, including misuse, robustness failures, privacy leakage, inadequate oversight, and governance noncompliance, so mitigations can be targeted to the stage and mechanism that enabled harm (Leveson 2012).

## AI Incidents as Empirical Data

Incident repositories and public reporting have become a pragmatic empirical base for studying AI harms, enabling quantification of harm prevalence, taxonomy validation, and identification of accountability gaps. The OECD has advanced a common reporting framework for AI incidents to standardize cross-case comparison and governance learning (OECD 2025). The incident approach has well-known limitations in selection effects, what gets reported, missing technical details, inconsistent causal attribution, and uneven geographic coverage, but it anchors harm taxonomies in observed failures rather than hypothetical possibilities. For child safety in particular, consolidating dispersed incidents (journalism, lawsuits, regulatory actions, school disclosures, platform enforcement reports) into a structured dataset enables reproducible measurement of prevalence patterns and harm–failure pathways.

## “How–What–Who” Mapping for Accountability

A central contribution of recent incident-driven analyses is to show that accountability requires simultaneously modeling: (i) what harm occurred, (ii) who was harmed, and (iii) how the harm arose (Li et al. 2025). This framing is especially salient for child harms, where the harmed party is often not the account holder or direct user, such as minors depicted in synthetic content or children exposed via recommendation pathways. Standard end-user-centric governance can therefore mis-assign responsibility and underincentivize upstream controls. Our work extends this mapping by specializing it for child stakeholders and developmental harm mechanisms. We depart from prior incidentanalytic work (Li et al. 2025) in three ways. First, duallevel harm coding: each incident receives both an internal (individual-level) and an external (population-level) harm code, so the link between harm to a child and its wider circulation can be measured rather than asserted (Appendix A). Second, taxonomic granularity: below adult-general categories we code child-specific harm types, including sexualization, sexual exploitation, addiction/compulsive use, over-reliance/dependency, and self-harm. Third, scope: prior child–AI work is survey-based, evaluation-based, or singlecase; to our knowledge this is the first corpus-scale incident study restricted to minors.

## Methodology

We perform a systematic, post-hoc incident analysis of publicly reported AI-related events in which a minor (under 18) plausibly experienced harm or a credible worst-case risk of harm. We define an incident as a discrete event or episode that became public through credible reporting (e.g., regulator actions, lawsuits, vulnerability disclosures, or investigative journalism).

We curate candidate incidents from public incident repositories, regulatory announcements, court filings, cybersecurity/vulnerability disclosures, and credible media reports. We include an item if (i) AI is materially involved (generation, recommendation, automated decision, or AI-mediated interaction); (ii) at least one affected stakeholder is a minor; and (iii) the report contains enough detail to code a harm category and a failure mode. We exclude purely speculative claims without an observed event, cases where child involvement is not plausible, and duplicates. The final corpus contains 1331 incidents: 1143 from the OECD.AI incident monitor, 141 from AIAAIC, and 47 from the AI Incident Database. The Occurred field is taken from the source repository and follows its dating convention: OECD.AI records a full date, AIAAIC and AIID record a year, and one AIAAIC record gives a multi-year range. Temporal analyses therefore use year of occurrence.

Each incident is manually annotated using a structured coding schema. We code (1) child-specific harms (one internal (individual-level) harm and one external (populationor institutional-level) harm per incident; refining general taxonomies into developmentally salient categories such as sexual harms, self-harm/mental-health escalation, manipulation/dependency, privacy/security leakage, misinformation, harassment/extortion, physical safety harms, and rights/procedural harms); and (2) sociotechnical failure modes, defined as the proximal mechanisms that are technical, organizational, governance, or use-related, through which an AI sociotechnical system is implicated in harm (Weidinger et al. 2023; Shelby et al. 2023). Each incident is coded with one primary failure mode: the proximal mechanism most directly implicated in the reported harm. Where reports document multiple contributing failures, we record the most proximal (Leveson 2012; Raji et al. 2022). Failure modes are organized by lifecycle stage: design, development/evaluation, release and use.

We develop the codebook iteratively (released alongside the dataset and reproduced in Appendix B): we seed categories from established taxonomies, introduce child-specific refinements, and pilot-code an initial subset. The remaining incidents are then independently annotated by two members of the research team, and disagreements are resolved through consensus coding (Hill et al. 2005; Saldana 2021).˜ To assess coding reliability, we calculated inter-rater agreement on a randomly sampled subset of 150 incidents (11.3% of the dataset) before consensus (see Appendix section “Inter-Rater Reliability Calculation”). For categorical variables (harm categories and failure modes), we computed Krippendorff’s α: harm categories $\alpha \ = \ 0 . 7 3$ and failure modes α = 0.67. For binary worst-case flags, Cohen’s κ = 0.58. These values indicate acceptable agreement for harm categories (α = 0.73) and only moderate agreement for failure modes (α = 0.67) and worst-case flags $( \kappa = 0 . 5 8 )$ Krippendorff’s conventional threshold for drawing firm conclusions is α ≥ 0.80, with $0 . 6 7 \leq \alpha < 0 . 8 0$ supporting tentative conclusions only (Krippendorff 2018). We therefore treat all failure-mode-dependent findings as provisional and interpret worst-case flags as indicative rather than definitive. Agreement was lowest where category boundaries are inherently contested — distinguishing contextual norm violation from governance noncompliance, and improper from malicious use — a limitation we return to below.

When reports cannot confirm whether harm has fully materialized, we record a plausible worst-case flag and treat these cases conservatively. From the coded corpus we compute three families of descriptive statistics: frequency distributions over harm categories, harm types, and failure modes; co-occurrence counts between harm categories and failure modes, and between internal and external harms, constructed by tallying incidents in which both codes appear; and annual counts of the most frequent failure modes. All reported quantities are counts of coded incidents. We do not fit models or test hypotheses.

This approach has notable limitations. Because incident narratives are observational and variably detailed, we avoid causal claims and report associations that identify highpriority harm pathways. In 112 of the 1331 incidents (8.4%), the available reporting did not support assignment of an internal harm category (two of these also lack an external harm code); these incidents are retained in the corpus and in all failure-mode and external-harm analyses, but are excluded from internal harm category counts. Key validity threats include reporting bias, incomplete technical details, age ambiguity, platform-level non-independence, and limited causal identifiability; we mitigate these through uncertainty flags and conservative worst-case labeling (Krippendorff 2018; Hruschka et al. 2004; Rosenbaum 2002; Shadish, Cook, and Campbell 2002; Turri and Dzombak 2023; Pittaras and Mc-Gregor 2022).

The schema is hierarchical: high-level harm categories were seeded from prior taxonomies (Shelby et al. 2023; Weidinger et al. 2023), and harm types are sub-codes that aggregate into these categories. The taxonomy is therefore neither purely deductive nor purely inductive. Categories were seeded deductively from adult-general taxonomies and largely retain their structure; child-specific refinement happens mainly at the harm-type level, where pilot coding surfaced cases that adult-general types could not hold without loss — principally sexualization, sexual exploitation, addiction/compulsive use, and over-reliance/dependency. We also code internal and external harm separately for every incident.

## Applying the Schema: Three Worked Cases

To make the coding schema concrete, we trace three incidents from report to code to implicated mitigation. The cases illustrate three recurring deployment patterns in our corpus: generative depiction, companion interaction, and institutional deployment. Each incident carries one primary failure mode; contributing failures documented in reporting are noted here but are not counted.

## Case 1: Synthetic Sexual Depiction (Almendralejo, 2023)

What happened. Students at a secondary school in Almendralejo, Spain used a commercial “nudification” application to generate synthetic nude images of female classmates from ordinary clothed photographs; the images then circulated through peer messaging groups (Reuters 2023). Harm codes. Internal: Representation, Dignity & Social Identity / sexualization — coded here rather than Psychological because the harm is constituted by how the child is depicted to others, not by the child’s own interaction with the system. External: Representation & Toxicity (Population-Level) / toxic content. Primaryfailure mode. Malicious use. Contributing: a general-purpose image tool available without restriction on depicting identifiable minors. Implicated mitigation. The depicted girls never used the application, so controls conditioned on the user cannot reach this case. It implicates depiction-consent verification and restriction at generation (second roadmap priority).

## Case 2: Companion Chatbot (Character.AI, 2024)

What happened. A fourteen-year-old developed a sustained relationship with a persona-based conversational agent over several months, culminating in his death by suicide (Roose 2024). Harm codes. Internal: Physical & Health / loss of life. External: Physical (Societal/Public Safety) / damage to public health. The harm is coded by its reported outcome; the psychological pathway that preceded it is captured by the failure mode. Primary failure mode. Overreliance or misleading anthropomorphic design. Contributing: no crisis routing on self-harm signals, and emotionally intimate framing available to a minor. Implicated mitigation. No single turn need cross a safety threshold for the cumulative interaction to end in harm, so per-output content evaluation cannot reach this case. It implicates session-level safety evaluation, guardian visibility, and mandatory escalation (first roadmap priority).

## Case 3: Institutional Deployment (School Weapons Detection, 2025)

What happened. An AI weapons-detection system at a Baltimore-area high school flagged a student’s bag of chips as a firearm; armed police responded and handcuffed the student (AIAAIC 2025). Harm codes. Internal: Autonomy / loss of rights and freedoms — coded as Autonomy rather than Physical & Health because no bodily injury occurred and the operative harm is deprivation of liberty. External: Human Rights & Civil Liberties / erosion of due process. Primary failure mode. General robustness and reliability failure. Contributing: no human confirmation step before an armed response. Implicated mitigation. The model error is ordinary; the severity comes from the escalation procedure around it. It implicates child-specific pre-deployment assessment with institutional consequence modeling (third roadmap priority).

## What the Cases Show

The three cases differ in where the failure enters and who holds the control point. In Case 1 the decisive control lies upstream, in what the generative system will produce; in Case 2 it lies in interaction design and is invisible to per-output evaluation; in Case 3 it lies with the deploying institution. No single intervention point covers all three, which is why the roadmap that follows is layered.

## Qualitative Findings and Resultant Themes

We report the child-specific harms and sociotechnical failure modes surfaced in our coded incident data set (1331 publicly reported child-related AI incidents), their prevalence, how individual-level and population-level harms relate, and how harms and failures relate to each other. Each incident carries one primary failure mode, so failure-mode counts sum to N. Harm counts are reported separately for internal and external harms, each coded once per incident. Our primary contributions here are (i) an empirically grounded child-centered harm taxonomy, (ii) a lifecycle-staged taxonomy of child-relevant sociotechnical failure modes, and (iii) relational mappings between the harms that occur and the failures they co-occur with. Throughout, prevalence denotes frequency within our corpus of publicly reported incidents, not incidence in the underlying population of child–AI interactions. The two diverge systematically: incident reporting favors acute, legally actionable, and media-legible events over slow-accumulating, private, or normalized harms. All distributional claims below should be read under this qualification.

<table><tr><td>Column</td><td>Description</td><td>Example entries</td></tr><tr><td>Occurred</td><td>The year in which the incident occurred or was publicly reported.</td><td>2023, 2019-2024, 2026-02-06</td></tr><tr><td>Failure Mode</td><td>The sociotechnical or system-level failure that most proximally contributed to the incident.</td><td>Malicious Use, Hallucination / Fabrication, Overreliance or Misleading Anthropomorphic Design</td></tr><tr><td>Internal Harm Category</td><td>The direct harm category experienced by an individual or small group affected by the AI system or its misuse.</td><td>Psychological, Autonomy, Physical &amp; Health</td></tr><tr><td>Internal Harm Type</td><td>The more specific internal harm experienced within the broader internal harm category.</td><td>Emotional distress, Sexualization, Self-harm, Identity theft</td></tr><tr><td>External Harm Category</td><td>The broader societal, institutional, market, or environmental harm category associated with the incident.</td><td>Misinformation &amp; Information Ecosystem, Privacy &amp; Security, Human Rights &amp; Civil Liberties</td></tr><tr><td>External Harm Type</td><td>The more specific external harm type within the broader external harm category.</td><td>Information ecosystem degradation, Cyber-attacks, Erosion of due process</td></tr><tr><td>Link / Summary</td><td>A source link or short narrative summary documenting the incident and its key context.</td><td>News report, watchdog summary, database repository entry</td></tr></table>

Figure 1: Overview of Dataset Coding Schema.

## What Child-Specific AI Harms Surface in Publicly Reported Incidents?

Our taxonomy of child-related AI harms is organized into eleven high-level categories, each comprising child-specific sub-harms that refine adult-general taxonomies to account for developmental vulnerability, constrained capacity for consent, and caregiver-mediated recourse.

Across all coded incidents,<sup>1</sup> the most common internal harm categories are Representation, Dignity and Social Identity (n = 475) and Psychological harms (n = 409), followed by Autonomy (n = 135) and Physical & Health harms (n = 85). Smaller but still meaningful categories include Privacy, Identity & Information Security (Individual) (n = 75), Political & Economic (n = 14), Financial, Property & Material (n = 9), Intellectual Property & Creative Labor (n = 8), Misinformation & Information Ecosystem (n = 6), Human Rights & Civil Liberties (n = 2), and Reputational (n = 1).

This distribution suggests that child-related AI harm is not primarily concentrated in narrow technical-error categories (such as privacy/security leakage from model misconfiguration or hallucinated factual errors). Instead, it is heavily shaped by harms to dignity, identity, emotional well-being, and autonomy. Within Representation, Dignity & Social Identity, sexualization (n = 316) is by far the most prevalent subtype, followed by sexual exploitation (n = 131) and exposure to toxic content (n = 17). This ranking should be read as a statement about what surfaces in public reporting rather than as evidence that sexualization outweighs dependency formation, subtle persuasion, or chronic emotional dysregulation — harms our own framework identifies as developmentally significant but which are structurally underreported because they accumulate privately and rarely generate a discrete reportable event. Within the Psychological category, the most frequent subtypes are addiction/compulsive use (n = 79), self-harm (n = 69), health deterioration (n = 66), and toxic content exposure (n = 62).

The category-to-type mapping is visualized in Appendix A, Figure 9 (internal), and in Figure 7(b) (external).

The external harm structure shows a parallel pattern at the societal level. The most common categories of external harm are Representation & Toxicity (Population-Level) (n = 476), Physical (Societal/Public Safety) (n = 298), Human Rights & Civil Liberties (n = 147), Reputational (n = 105), and Privacy & Security (n = 94). Less frequent external categories are Autonomy (n = 82), Political & Economic (n = 47), Societal & Cultural (n = 38), Misinformation & Information Ecosystem (n = 24), Financial & Business (n = 16), and Environmental (n = 2). This dual structure reflects an ecological model of harm in which individual- and population-level harms are connected through circulation and re-injury (Bronfenbrenner 1979; Livingstone and Stoilova 2021).

![](images/9f8a02c4d8c934deafd986daafd756106aceed9f73ab95ac798c4e68f72ba161.jpg)  
Codebook types with no coded incidents (n = 0) — Psychological: Anxiety / distress; Desensitization. Autonomy: Personality rights loss. Representation, Dignity & Social Identity: Defamation / libel / slander; Dehumanization / objectification; Marginalization; Stigmatization. Privacy, Identity & Information Security (Individual): Confidentiality loss; Impersonation. Financial, Property & Material: Increased workload; Market value loss; Property damage. Physical & Health: Damage to public health. Intellectual Property & Creative Labor (Individual Small Group): Creativity loss.  
Figure 2: Taxonomy of Internal Generative AI Harms (ordered by prevalence).

## How Do Child AI Harms Surface? Prevalence of Sociotechnical Failure Modes

The failure-mode distribution shows that child-related AI harms are most often coded with Human, Organizational, and Contextual Deployment Failures (n = 779) — failures in how AI systems are situated, governed, and used in real social contexts. Within this group, violation of contextual norms (n = 398) is the single largest failure mode in the entire data set, followed by Governance/Regulatory Noncompliance (n = 227) and Overreliance or Misleading Anthropomorphic Design (n = 146). Many harms emerge not from bad model outputs but from deployment in contexts where safety and age-appropriateness expectations are violated.

For example, Snap’s 2023 deployment of “My AI” inside an app with a large minor userbase, in which the system discussed age-inappropriate topics with accounts registered to 13-year-olds, was primarily a contextual norm violation rather than a model defect (Fowler 2023). Similarly, the 2023 FTC action against Amazon for retaining children’s Alexa voice recordings after parental deletion requests illustrates how governance noncompliance compounds privacy leakage (Federal Trade Commission 2023).

The second largest group is Misuse/Abuse (n = 297), dominated by Malicious Use (n = 228). Model Behavior/Technical Failure is smaller overall (n = 196), where General Robustness/Reliability Failure (n = 128) far exceeds Hallucination/Fabrication (n = 47), indicating that reliability under real deployment conditions matters more than hallucination alone. Privacy, Data Governance and Information Handling (n = 59) is driven almost entirely by Privacy or Confidentiality Leakage. The four failure-mode families operate at different levels, and the level determines which actor can intervene. Model Behavior/Technical failures (robustness, hallucination, data quality) are properties of the model or pipeline and fall to developers. Privacy and data-governance failures concern how children’s data are collected and handled, and fall to developers and deployers jointly. Human, Organizational, and Contextual Deployment failures (contextual norms violation, governance noncompli ance, overreliance or misleading anthropomorphic design, inadequate oversight) are properties of the product, the deploying institution, and the rule environment. Misuse/Abuse failures (malicious, undisclosed, or out-of-scope use) arise at the point of use and fall to platforms and the controls they place on users. Because each incident carries one primary code, these counts locate the most proximal failure, not ev-

ery failure in the chain.

## Which Failure Modes Are Associated with Each Internal Harm Category?

The largest internal harm categories, that is Representation, Dignity and Social Identity and Psychological harms align with the prominence of Contextual Norms Violation, Malicious Use, and Overreliance or Misleading Anthropomorphic Design. Harms to children are thus most often documented alongside socially inappropriate use, deliberate exploitation of tools, or interaction design that encourages emotional or social over-trust. The prevalence of sexualization, exploitation, and toxic exposure is consistent with a landscape of failure where malicious or norm-violating use plays a central role. Design patterns associated with dependency or misleading social attachment are not peripheral: overreliance or misleading anthropomorphic design is the primary failure mode in 146 incidents, 49 of them with a Psychological internal harm. The Autonomy and Physical & Health categories, while smaller, are associated with improper or out-of-scope use, contextual norm violation, and overreliance on system outputs confirming that internal harms are often multifactorial.

## Which Failure Modes Are Associated with External Harm Categories?

The single largest external category, Representation & Toxicity (Population-Level), is dominated by toxic content (n = 467), with the remainder distributed across discrimination at population or systemic scale $( n \ = \ 3 )$ , stereotyping (n = 2), group-level stigmatisation $\mathrm { ~ ( ~ n ~ ~ = ~ ~ 2 ~ ) ~ }$ , and group-level marginalisation $( n = 2 )$ . For example, AI-generated nude images of a single classmate frequently spread across an entire school cohort through social networks, transforming an individual harm into peer-network exposure and reputational damage that affects dozens of uninvolved minors (Singer 2024; Reuters 2023). This aligns with the dominance of Contextual Norms Violation and Malicious Use, since population-level harm depends on circulation, amplification, and deployment context.

Physical (Societal/Public Safety) (n = 298) and Human Rights & Civil Liberties (n = 147) indicate that childrelated AI incidents sometimes involve broader rights protections and public safety consequences. Three crosscutting themes emerge: (1) child-related AI harms are strongly sociotechnical, and the largest failure families are deployment, governance, and misuse-related; (2) the most prevalent harms concentrate in psychological, representational, and dignity-related categories; and (3) many harms are multi-layered, with direct individual harms and broader societal harms co-emerging. This supports layered governance and safety approaches (Leveson 2012; Weidinger et al. 2023; Li et al. 2025).

## Quantitative Findings and Resultant Patterns

Our quantitative analysis draws on child-related AI incidents curated from the AIAAIC, AIID, and OECD.AI incident monitor. These repositories offer a pragmatic empirical basis while reflecting known limitations: reporting bias, uneven documentation quality, and stronger visibility for acute events than cumulative harms (Abercrombie et al. 2024; Mc-Gregor 2021; OECD 2025; Turri and Dzombak 2023). We treat each incident as an observational case coded with structured attributes spanning failure mode and internal and external harm category (Hutiri, Papakyriakopoulos, and Xiang 2024; Raji et al. 2022; De Miguel Velazquez et al. 2024).´

## How Do Generative AI Harms Surface in Publicly Reported Incidents?

The temporal trend in our data set suggests that child-related AI harms surface through a progressively broader sociotechnical failure landscape rather than a single dominant technical defect. Incident counts were low and flat through the mid-2010s, rose sharply beginning around 2020, and especially accelerated after 2022. Contextual Norms Violation becomes the most prominent failure mode in the later years, peaking dramatically in 2024–2025, reflecting systems being introduced into settings where expectations about safety, age-appropriateness, supervision, and consent are violated. The sharp increase in malicious use beginning in the early 2020s (accelerating in 2023–2025) is consistent with the growing visibility of incidents involving deepfake sexual abuse material, grooming facilitation, and exploitative impersonation.

Publicly reported child-related incidents are especially likely to surface where harms are highly visible, institutionally legible, or socially alarming, which helps explain why the latter period is dominated by malicious exploitation and contextual norm breakdown rather than slower cumulative harms such as dependency formation or subtle persuasion (Pittaras and McGregor 2022; Raji et al. 2022; De Miguel Velazquez et al. 2024).´

## Which Failure Modes Are Associated with Each Type of Harm?

Figure 6(a) shows that the largest internal harm categories in Representation, Dignity and Social Identity, Psychological harms, and autonomy are associated with a recurring cluster of failure modes: violation of contextual norms, governance or regulatory noncompliance, overreliance or misleading anthropomorphic design, and malicious use, with additional contributions from robustness failures, privacy leakage, and improper or out-of-scope use. Thus, internal harms to children are often linked to systems placed in inappropriate social roles, deployed without adequate safeguards, or intentionally exploited in ways that undermine dignity, identity, emotional well-being, or independent judgment (Crawford 2024; Singer 2024; Thiel and Hancock 2023).

The prominence of Representation, Dignity and Social Identity is especially important given its plausible capture of harms such as sexualized depiction, humiliating synthetic representation, and abusive identity manipulation. The strong role of contextual norm violation indicates that many such harms are recorded where AI tools are used in ways that violate age-specific expectations of care, respect, and consent. Psychological harm is most often coded alongside relational framing, trust miscalibration, exploitative interaction design, or repeated exposure where the system is treated as a companion, authority, or emotionally significant agent, a pattern developmentally significant because children are often more sensitive to cues of authority, intimacy, and social reciprocity than adults (UNICEF 2021; Xu and Warschauer 2019).

![](images/27136e41d84229fb14f281a9227bda1760a6bc625fd44e52faf6bc8ac5f95778.jpg)  
Figure 3: Taxonomy of Sociotechnical Failure Modes (bar length indicates prevalence; colors identify the failure-mode family).

As shown in Figure 6(b), the external pattern mirrors the internal one. Representation and Toxicity (Population-Level) is by far the largest external category, followed by Physical (Societal or Public Safety), Human Rights and Civil Liberties, Reputational, and Privacy and Security harms. The same dominant failure modes appear, contextual norm violation, governance noncompliance, overreliance or misleading anthropomorphic design, and malicious use, though their balance varies by harm category. Population-level representation and toxicity harms reflect the wider circulation of sexualized, abusive, or otherwise harmful child-related content; the fact that this is the largest external harm bucket underscores that child harm is frequently mediated by dissemination and collective exposure rather than only by one-to-one interaction.

The prominence of Physical (Societal or Public Safety) indicates that harms to children in public or institutional settings are linked to deployment into environments where failure carries real-world safety consequences, strengthening the case for child-specific operational safeguards in systems used around transportation, monitoring, and school safety. In general, severe child-related harms are usually multi-factor, the same incident group often reflects malicious exploitation layered on top of weak governance, inappropriate deployment context, misleading social design, and insufficient robustness (Critch and Russell 2023; Weidinger et al. 2023).

## Harms Categories Cross-Cutting Analysis

The landscape of internal harm is highly concentrated in a small number of dominant subtype pathways. The most prominent pattern is the concentration of sexualization within Representation, Dignity & Social Identity, along with a second cluster of sexual exploitation in the same category. A second major concentration appears in the Psychological category: addiction or compulsive use, self-harm, health deterioration, and emotional distress form a broad cluster, suggesting harms extend beyond one-off unsafe outputs to include repeated interaction patterns affecting emotional regulation, dependency, and mental well-being.

Representative cases within the sexualization cluster include the 2023 Almendralejo, Spain incident (AI-generated nude images shared at school) (Reuters 2023) and repeated findings of CSAM generated by users of open-release image models (Thiel and Hancock 2023). Within the psychological cluster, representative cases include Amazon Alexa’s “penny challenge” encouraging a child to touch a live electrical outlet (BBC 2021), TikTok’s amplification of suicide content to vulnerable teenagers (Amnesty International 2023), and the Setzer/Character.AI case (Roose 2024).

Some categories are defined by a narrow dominant subtype (Autonomy: loss of rights/freedoms; Privacy: privacy loss; Physical & Health: bodily injury), while Psychological harms are more heterogeneous, implying that targeted interventions may suffice for narrow-subtype categories, while psychological harms require layered safeguards spanning design, interaction, and escalation. External harms are even more sharply concentrated: the largest single cell is Representation & Toxicity (Population-Level) and Toxic content, followed by Physical (Societal/Public Safety) and Damage to public health, and Human Rights & Civil Liberties and Loss of rights/freedoms.

The strongest co-occurrence is Representation, Dignity & Social Identity with Representation & Toxicity (Population-Level), confirming that internal child harms are often structurally connected to population-level harms rather than isolated private events. Psychological harms connect across multiple external domains, indicating a more diffuse and socially amplified harm profile.

Header accent and count badge shade with category size (n); row bars scale with incident count.

![](images/e6a206f7aa5c9ec83774987547fa0214cfad192a50b277f390d7fa3040efdf20.jpg)  
Codebook types with no coded incidents (n = 0) — Political & Economic: Economic manipulation; Market distortion / monopolization; Societal destabilization. Misinformation & Information Ecosystem: Erosion of trust in public information; Historical revisionism. Societal & Cultural: Cultural dispossession; Overburdening ecosystems; Productivity loss; Service quality erosion. Reputational: Defamation / libel / slander. Representation & Toxicity (Population-Level): Unfair capability distribution. Financial & Business: Business operations / infrastructure damage; Critical infrastructure damage. Privacy & Security: Distortion; Exclusion. Human Rights & Civil Liberties: Exclusion. Physical (Societal / Public Safety): Self-harm. Environmental: Air pollution; Ecological/biodiversity loss; Energy consumption / utility costs; Excessive energy shortages; Excessive water shortages; Ground pollution; Noise pollution; Water pollution.  
Figure 4: Taxonomy of External Generative AI Harms (darker colors indicate higher prevalence).

Synthesis of Quantitative and Qualitative Insights The results strongly support three synthesis claims.

First, child-related AI harm is overwhelmingly sociotechnical rather than merely model-internal. The model-internal view — which evaluates whether a model generates harmful text or fails under adversarial prompts — underlies most current safety practices, but is insufficient by itself. A model can pass every benchmark and still cause severe harm when deployed in a school without ageappropriate configuration, offered as a companion app to vulnerable users, or exploited by malicious actors within technical guardrails. The dominant patterns in our data run through deployment context, institutional practice, interaction design, and misuse ecosystems (Hoffmann and Frase 2023; Weidinger et al. 2023).

To illustrate: a general-purpose LLM passing standard red-team benchmarks may be repackaged as a companion application. Such products are not unprotected — app stores impose age ratings, providers require self-attested age and often parental consent, and content policies prohibit sexual material involving minors. These protections nonetheless fail predictably in combination: age attestation is unverified and trivially bypassed; store-level ratings do not constrain in-app behavior; and safety evaluation is performed per turn, so a conversation in which no single output violates policy can still, in aggregate over weeks, establish emotional dependency. The failure is not the absence of safeguards but their mismatch to the timescale and stakeholder structure of the harm (Thiel and Hancock 2023; Widder, Whittaker, and West 2024).

Second, child-related incidents disproportionately involve harm to dignity, identity, psychological well-being, and population-level representation. Internally, Representation, Dignity and Social Identity, and Psychological harms are most prevalent. Externally, Representation and Toxicity (Population-Level) is the single largest category by a wide margin. Children are often harmed not just as users of a system, but as subjects of representation, targets of circulation, or members of social groups affected by toxic or exploitative media environments, a pattern that explains why child safety cannot be adequately addressed through model accuracy metrics alone.

![](images/05247e4fd5de2aa531fa4ba9ba71920d43d845bb17d1fb5b06d07e8611046426.jpg)  
Figure 5: Most prevalent failure mode per year, 2014–2026<sup>∗</sup> (<sup>∗</sup>incomplete year).

Third, severe harm is spread across many failure modes rather than one. The most consequential harm categories are each coded with several different primary failure modes — malicious use, contextual norm breakdown, weak governance, misleading social framing, and insufficient robustness — and the worked cases show these failures operating together within single incidents. This multifactor structure strengthens the case for layered rather than single-point interventions (Leveson 2012; Turri and Dzombak 2023): even meaningful improvements in model behavior may not be enough if platforms, deployers, and surrounding institutions do not also change how systems are introduced, constrained, and governed around children.

## A Roadmap for Policymakers

The empirical patterns above converge on five priorities for child-AI safety governance, each keyed to a dominant dataset pattern and to established governance instruments.

First, restrict high-risk child-facing features by default. Basis: overreliance or misleading anthropomorphic design is the primary failure mode in 146 incidents, and contextual norms violation, the largest failure mode overall (n = 398), is most often coded with Representation, Dignity & Social Identity (n = 147) and Psychological (n = 117) harms. Features such as photorealistic image editing of real people, intimate roleplay, companion-style emotional framing, and voice cloning should be off by default for minors and should be supported by robust age assurance rather than self-attestation (European Union 2024; UNICEF 2021).

Second, treat depiction consent and anti-CSAM controls as a baseline. Basis: sexualization and sexual exploitation together account for 447 incidents, a third of the corpus; their most common primary failure modes are contextual norms violation $( n = 1 3 8 )$ , malicious use (n = 100), and governance noncompliance (n = 78), so misuse controls alone would miss most of them. Systems that can create, modify or distribute real-person images should verify consent, implement upstream content restrictions, and operate rapid cross-platform takedown pathways (Gibson, Farid, and Paris 2024; Solaiman et al. 2023; Thiel and Hancock 2023).

Third, mandate child-specific evaluation for highstakes deployments. Basis: Physical (Societal/Public Safety) is the second-largest external harm category (n = 298), and its most common primary failure mode is contextual norms violation $\begin{array} { r l r } { ( n } & { { } = } & { 8 5 ) } \end{array}$ not model error. Benchmark-style certification does not anticipate contextual norm violations; systems deployed in schools, youth transport, and safety-adjacent settings should be subject to a child-specific pre-deployment assessment, including youthspecific edge cases, institutional consequence modeling, and post-deployment monitoring (Ada Lovelace Institute 2024; Raji et al. 2022).

Fourth, assign responsibility across the AI supply chain in several ways. Basis: we do not code provider or deployer identity, so this priority draws on the governance literature rather than on corpus counts. Systemic-risk, red-teaming, and disclosure duties should fall on generalpurpose and frontier developers; contextual duties (age assurance, UX constraints, escalation protocols) should fall on deployers (Anderljung et al. 2023; Bommasani et al. 2024; European Union 2024).

Fifth, require structured incident disclosure with child-relevant metadata. Basis: every incident in our corpus reached us through voluntary repositories that aggregate press, legal, and advocacy reports; no mandatory filing channel exists for child-related AI incidents, which points to a critical visibility gap. A mandatory disclosure regime using standardized child-relevant fields (age band, deployment context, depiction status, interacting vs. noninteracting child, provider tier) would move society off its current reliance on ad hoc reporting (McGregor 2021; OECD 2025; Turri and Dzombak 2023). Such a regime need not be built from nothing. The EU AI Act already obliges providers of high-risk systems to report serious incidents to national authorities within fixed deadlines, and the OECD common reporting framework supplies a harmonized schema for cross-jurisdiction comparison (European Union 2024; OECD 2025). What both currently lack is child-relevant metadata. Extending either with the fields above would make child-related incidents separable from the general incident stream without creating a parallel reporting obligation, and would place enforcement with authorities that already possess it.

![](images/9bbade6238bf0d120b8894cce50bd06bd80ce0dd66d0b551ab64d33b01f8844a.jpg)

![](images/04be975e5f25232c7602736cd1bbb80b43be34b6fa5566755143bc4c451c8c80.jpg)  
Figure 6: Internal (a) and external (b) harm categories by primary failure mode.

## Recommendations

Our recommendations follow a system-theoretic safety logic, pointing to controls at the level of high-risk affordances, platform mediation, institutional deployment, and child-facing interaction design (Leveson 2012; Weidinger et al. 2023).

High-risk affordances should be restricted by default with controls that are difficult to bypass and do not rely solely on self-attestation. Consent verification and prevention-bydesign for synthetic sexual or exploitative depiction should be treated as baseline requirements, not optional safeguards, with strong upstream restrictions, downstream detection and distribution controls, and rapid response channels for victims and guardians (Gibson, Farid, and Paris 2024; Thiel and Hancock 2023). Child-facing conversational and companion systems should be redesigned to reduce overreliance and misleading social attachment: developers should limit anthropomorphic cues, clarify system boundaries, build in guardian visibility and control, and implement robust escalation for self-harm, abuse, grooming, and coercion signals. Platform accountability should focus on distribution and amplification, not only on generation, with auditable pathways of youth exposure and default dampening of sexualized, self-harm, grooming-adjacent, and exploitative content clusters (Critch and Russell 2023; Shelby et al. 2023).

Child-related incidents are produced through a layered provider stack: foundation model developers build the underlying capabilities, while downstream deployers embed those capabilities in child-adjacent products. Our data are consistent with, but do not by themselves establish, a tiered governance structure. Because we do not code provider or deployer identity, the allocation below is derived from the governance literature rather than from our corpus. It assigns systemic-risk and transparency duties to general-purpose and frontier model providers while placing contextual duties on the implementers, where both upstream advantages and downstream deployment choices are involved in dominant harm pathways (Anderljung et al. 2023; Engler 2023; European Union 2024; Bommasani et al. 2024; Widder, Whittaker, and West 2024).

![](images/7f143c0d670b9b670a02b0a204035ed71b777275474e1c6b3ec2a416303ba0fc.jpg)

![](images/db4a8e9ba8f9bc0ed1e8bc487dcc4b31b39b96af37aa73b9d3f237c9281f5660.jpg)  
Figure 7: Internal (a) and external (b) harm categories by harm type; the eight most frequent types are shown individually.

## Limitations

Source and geographic coverage. The corpus is drawn from three public repositories (OECD.AI, AIAAIC, AIID) that rely predominantly on English-language news, legal, and regulatory reporting. We did not code country or region, so we cannot report a geographic distribution; incidents affecting children in under-reported regions are likely underrepresented, and our prevalence figures should not be read as global.

Reporting bias and cumulative harm. The incident method favors acute, legally visible, and socially alarming events. Slow-accumulating harms — dependency formation, subtle persuasion, chronic emotional dysregulation — rarely produce a discrete reportable event and are therefore likely under-counted relative to their real-world prevalence.

Child voice is absent. Every incident is mediated by a third party: a journalist, litigant, regulator, or platform. No child’s own account of AI-mediated harm is represented, a substantive gap in the harm picture rather than a methodological footnote.

One primary failure mode per incident. Contributing failures are not counted, so failure-mode distributions locate the most proximal failure rather than the full chain; multi-factor pathways are illustrated through cases rather than measured.

Coding reliability. Agreement on failure modes (α = 0.67) falls below the conventional threshold for firm conclusions; findings that depend on failure-mode codes are correspondingly provisional.

Uncoded fields. Stakeholder role, provider and deployer identity, and reporting channel are not coded. Recommendations that depend on them, notably the fourth and fifth roadmap priorities, draw on the governance literature rather than on this corpus.

Non-independence. A single platform or product can generate many incidents, and repositories may record related events separately. Counts should be read as counts of reported incidents, not of distinct sociotechnical failures.

## Concluding Remarks and Future Directions

This paper contributes a child-centered incident data set and a relational taxonomy mapping internal harms, external harms, and sociotechnical failure modes in publicly reported child-related AI incidents. The most prominent harms focus on Representation, Dignity, and Social Identity and psychological harms internally, and Representation and Toxicity (Population-Level) externally. Future work should build on this foundation in four directions. First, stakeholder coding should be added. Our corpus records what harm occurred and how, but not who was harmed beyond the internal/external distinction; coding the harmed party (interacting child user, non-interacting child subject, caregiver, or institution) would allow the accountability asymmetries this paper describes qualitatively to be measured directly. Second, repeated incidents should be linked longitudinally to better capture cumulative and developmental harms. Third, high-severity clusters such as synthetic sexual abuse, publicsafety failures, and anthropomorphic overreliance should be examined with higher-fidelity causal methods. Fourth, crossrepository harmonization of child-relevant metadata should be pursued to reduce duplication and support robust comparative analysis (Leveson 2012; Turri and Dzombak 2023; De

![](images/06774885a87c8f0a1a0b70b770dfeb68d1c67912764854be87d3dd66966be423.jpg)  
Figure 8: Model-Internal vs. Sociotechnical Safety View.

Miguel Velazquez et al. 2024).´

Future iterations should code the reporting channel (regulatory filings, journalism, platform reports, school complaints) to assess which accountability channels most effectively surface child-related AI harms and add provider- and deployer-level coding to support regulatory calibration. If investigative journalism is the dominant source, that finding itself constitutes a governance problem, implying society depends on a small number of specialized reporters rather than having mandatory disclosure regimes.

The authoritative KidAID source workbook is publicly available through Harvard Dataverse at https://doi. org/10.7910/DVN/ZFAHMV. A privacy-minimized annotation release, codebook, methodology, and validation materials are available at https://github.com/The-Responsible-AI-Initiative/KidAID.

## Conflict of Interest

The authors declare that the research was conducted in the absence of commercial or financial relationships that could be construed as a potential conflict of interest.

## Funding

This research is funded by NSF grants 2125858, 2236305, and the UT-Good Systems Grand Challenge. We would like to express our gratitude for these institutions’ support, which made this study possible. Furthermore, in accordance with MLA (Modern Language Association) guidelines, we thank AI-powered tools for their assistance, including editing and brainstorming.

## Institutional Review Board Statement

Not applicable.

## References

Abercrombie, G.; Benbouzid, D.; Giudici, P.; Golpayegani, D.; Hernandez, J.; Noro, P.; Pandit, H.; Paraschou, E.; Pownall, C.; Prajapati, J.; Sayre, M. A.; Sengupta, U.; Suriyawongkul, A.; Thelot, R.; Vei, S.; and Waltersdorfer, L. 2024. A Collaborative, Human-Centred Taxonomy of AI, Algorithmic, and Automation Harms. arXiv preprint arXiv:2407.01294.

Ada Lovelace Institute. 2024. Safe before Sale. Technical report, Ada Lovelace Institute.

AIAAIC. 2025. AIAAIC Repository. https://www.aiaaic. org/aiaaic-repository.

Amnesty International. 2023. Driven into Darkness: How TikTok’s ‘For You’ Feed Encourages Self-Harm and Suicidal Ideation. Technical report, Amnesty International.

Anderljung, M.; Barnhart, J.; Korinek, A.; Leung, J.; O’Keefe, C.; Whittlestone, J.; Avin, S.; Brundage, M.; Bullock, J.; Cass-Beggs, D.; Chang, B.; Collins, T.; Fist, T.; Hadfield, G.; Hayes, A.; Ho, L.; Hooker, S.; Horvitz, E.; Kolt, N.; Schuett, J.; Shavit, Y.; Siddarth, D.; Trager, R.; and Wolf, K. 2023. Frontier AI Regulation: Managing Emerging Risks to Public Safety. arXiv preprint arXiv:2307.03718.

BBC. 2021. Alexa Tells 10-Year-Old Girl to Touch Live Plug with Penny. BBC News.

Bender, E. M.; Gebru, T.; McMillan-Major, A.; and Shmitchell, S. 2021. On the Dangers of Stochastic Parrots: Can Language Models Be Too Big? In Proceedings

of the 2021 ACM Conference on Fairness, Accountability, and Transparency, 610–623.

Birhane, A.; Prabhu, V. U.; and Kahembwe, E. 2022. Multimodal Datasets: Misogyny, Pornography, and Malignant Stereotypes. arXiv preprint arXiv:2110.01963.

Bommasani, R.; Kapoor, S.; Klyman, K.; Longpre, S.; Ramaswami, A.; and Cihon, P. 2024. Considerations for Governing Open Foundation Models. In Proceedings ofthe 41st International Conference on Machine Learning.

Bostrom, N. 2014. Superintelligence: Paths, Dangers, Strategies. Oxford University Press.

Bronfenbrenner, U. 1979. The Ecology of Human Development: Experiments by Nature and Design. Harvard University Press.

Common Sense Media. 2024. The Dawn of the AI Era: Teens, Parents, and the Adoption of Generative AI at Home and School. Technical report, Common Sense Media, San Francisco, CA.

Crawford, A. 2024. Character.AI: Young People Turning to AI Therapist Bots. BBC News.

Crawford, K. 2021. Atlas of AI: Power, Politics, and the Planetary Costs of Artificial Intelligence. Yale University Press.

Critch, A.; and Russell, S. 2023. TASRA: A Taxonomy and Analysis of Societal-Scale Risks from AI. arXiv preprint arXiv:2306.06924.

De Miguel Velazquez, J.;´ S<sup>ˇ</sup> cepanovi´ c, S.; Gvirtz, A.; and´ Quercia, D. 2024. Decoding Real-World Artificial Intelligence Incidents. Computer, 57(11): 71–81.

Engler, A. 2023. The EU and U.S. Diverge on AI Regulation: A Transatlantic Comparison and Steps to Alignment. Brookings Institution.

European Union. 2024. Regulation (EU) 2024/1689 of the European Parliament and of the Council – Artificial Intelligence Act. Technical report, Official Journal of the European Union.

Federal Trade Commission. 2023. Amazon.com (Alexa), U.S. v. FTC Legal Library.

Fowler, G. A. 2023. Snapchat Tried to Make a Safe AI. But Tests Reveal Its Conversations Can Be Unsafe for Teens. The Washington Post.

Gibson, R.; Farid, H.; and Paris, B. 2024. Synthetic Child Sexual Abuse Material: Detection, Disclosure, and Response. Journal ofOnline Trust and Safety, 2(3).

Hendrycks, D.; Mazeika, M.; and Woodside, T. 2023. An Overview of Catastrophic AI Risks. arXiv preprint arXiv:2306.12001.

Hill, C. E.; Knox, S.; Thompson, B. J.; Williams, E. N.; Hess, S. A.; and Ladany, N. 2005. Consensual Qualitative Research: An Update. Journal of Counseling Psychology, 52(2): 196–205.

Hoffmann, M.; and Frase, H. 2023. Adding Structure to AI Harm. Responsible AI in Practice.

Hruschka, D. J.; Schwartz, D.; John, D. C. S.; Picone-Decaro, E.; Jenkins, R. A.; and Carey, J. W. 2004. Reliability in Coding Open-Ended Data: Lessons Learned from HIV Behavioral Research. Field Methods, 16(3): 307–331.

Hutiri, W. T.; Papakyriakopoulos, O.; and Xiang, A. 2024. Not My Voice! A Taxonomy of Ethical and Safety Harms of Speech Generators. In Proceedings of the 2024 ACM Conference on Fairness, Accountability, and Transparency, 359–376.

Internet Watch Foundation. 2024. What Has Changed in the AI CSAM Landscape? Technical report, Internet Watch Foundation, Cambridge, UK.

Krippendorff, K. 2018. Content Analysis: An Introduction to Its Methodology. SAGE Publications, 4th edition.

Leveson, N. G. 2012. Engineering a Safer World: Systems Thinking Applied to Safety. MIT Press.

Li, M.; Bickersteth, W.; Tang, N.; Hong, J.; Cranor, L.; Shen, H.; and Heidari, H. 2025. A Closer Look at the Existing Risks of Generative AI: Mapping the Who, What, and How of Real-World Incidents. arXiv preprint.

Livingstone, S.; and Stoilova, M. 2021. The 4Cs: Classifying Online Risk to Children. Technical report, Leibniz-Institut fur Medienforschung ¨ | Hans-Bredow-Institut (HBI), Hamburg.

McGregor, S. 2021. Preventing Repeated Real World AI Failures by Cataloging Incidents: The AI Incident Database. In Proceedings of the AAAI Conference on Artificial Intelligence, 15458–15463.

National Institute of Standards and Technology. 2024. Artificial Intelligence Risk Management Framework: Generative Artificial Intelligence Profile. Technical Report NIST AI 600-1, U.S. Department of Commerce.

OECD. 2019. OECD Principles on AI. Technical report, Organisation for Economic Co-operation and Development.

OECD. 2025. Towards a Common Reporting Framework for AI Incidents. Technical report, Organisation for Economic Co-operation and Development.

Pittaras, N.; and McGregor, S. 2022. A Taxonomic System for Failure Cause Analysis of Open Source AI Incidents. arXiv preprint arXiv:2211.07280.

Raji, I. D.; Smart, A.; White, R. N.; Mitchell, M.; Gebru, T.; Hutchinson, B.; Smith-Loud, J.; Theron, D.; and Barnes, P. 2022. The Fallacy of AI Functionality. In Proceedings of the 2022 ACM Conference on Fairness, Accountability, and Transparency, 959–972. ACM.

Reuters. 2023. Spanish Prosecutor to Probe AI-Generated Images of Naked Minors. Reuters.

Roose, K. 2024. Can a Chatbot Named Daenerys Targaryen Be Blamed for a Teen’s Suicide? The New York Times.

Rosenbaum, P. R. 2002. Observational Studies. Springer, 2nd edition.

Saldana, J. 2021.˜ The Coding Manual for Qualitative Researchers. SAGE Publications, 4th edition.

Shadish, W. R.; Cook, T. D.; and Campbell, D. T. 2002. Experimental and Quasi-Experimental Designs for Generalized Causal Inference. Houghton Mifflin.

Shelby, R.; Rismani, S.; Henne, K.; Moon, A.; Rostamzadeh, N.; Nicholas, P.; Yilla-Akbari, N.; Gallegos, J.; Smart, A.; Garcia, E.; and Virk, G. 2023. Sociotechnical Harms of Algorithmic Systems: Scoping a Taxonomy for Harm Reduction. In Proceedings of the 2023 AAAI/ACM Conference on AI, Ethics, and Society, 723–741.

Singer, N. 2024. Teen Girls Confront an Epidemic of Deepfake Nudes in Schools. The New York Times.

Solaiman, I.; Talat, Z.; Agnew, W.; Ahmad, L.; Baker, D.; Blodgett, S. L.; Chen, C.; Daume III, H.; Dodge, J.; Duan,´ I.; Evans, E.; Friedrich, F.; Ghosh, A.; Gohar, U.; Hooker, S.; Jernite, Y.; Kalluri, R.; Lusoli, A.; Leidinger, A.; Lin, M.; Lin, X.; Luccioni, S.; Mickel, J.; Mitchell, M.; Newman, J.; Ovalle, A.; Png, M.-T.; Singh, S.; Strait, A.; Struppek, L.; and Subramonian, A. 2023. Evaluating the Social Impact of Generative AI Systems in Systems and Society. arXiv preprint arXiv:2306.05949.

Thiel, D.; and Hancock, J. 2023. Identifying and Eliminating CSAM in Generative ML Training Data and Models. Technical report, Stanford Internet Observatory.

Turri, V.; and Dzombak, R. 2023. Why We Need to Know More: Exploring the State of AI Incident Documentation Practices. In Proceedings of the 2023 AAAI/ACM Conference on AI, Ethics, and Society, 576–583.

UNICEF. 2021. Policy Guidance on AI for Children. Technical report, United Nations Children’s Fund.

Weidinger, L.; Rauh, M.; Marchal, N.; Manzini, A.; Hendricks, L. A.; Mateos-Garcia, J.; Bergman, S.; Kay, J.; Griffin, C.; Bariach, B.; Gabriel, I.; Rieser, V.; and Isaac, W. 2023. Sociotechnical Safety Evaluation of Generative AI Systems. arXiv preprint arXiv:2310.11986.

Widder, D. G.; Whittaker, M.; and West, S. M. 2024. Why ‘Open’ AI Systems Are Actually Closed, and Why This Matters. Nature, 635(8040): 827–833.

Xu, Y.; and Warschauer, M. 2019. Young Children’s Reading and Learning with Conversational Agents. In Conference on Human Factors in Computing Systems (CHI) – Extended Abstracts.

## Harm Category Co-Occurrence Matrices Harm Category Definitions and Representative Examples

This appendix provides definitions and representative examples for each internal harm category used in the KidAID coding schema. Categories are listed in descending order of prevalence in the data set.

Representation, Dignity & Social Identity (n = 475). Definition: Harms arising from how AI systems depict, categorize, circulate, or manipulate children’s identities, images, or social standing. This includes cases where a minor’s likeness, name, or personal characteristics are used to produce harmful content without consent, or where AI output reinforces stigmatizing, exploitative, or dehumanizing representations. Key sub-harms: Sexualization $( n = 3 1 6 ) \colon$ AI-generated or manipulated sexual depictions of minors.

Sexual exploitation $( n = 1 3 1 ) \colon$ distribution, trafficking, or monetization of such depictions. Exposure to toxic content $( n ~ = ~ 1 7 ) \colon$ exposure to abusive, hateful, or identitydenigrating outputs. Representative examples: (i) A high school student uses a nudification app to generate and share non-consensual synthetic nude images of female classmates; (ii) An adult uses an AI image generator to produce thousands of prepubescent CSAM images; (iii) A chatbot generates a persona imitating a recently deceased teenager and makes it accessible on the platform.

Psychological (n = 409). Definition: Harms to children’s mental health, emotional regulation, social attachment, and behavioral patterns arising from AI interaction. Unlike physical harm, these often develop through repeated exposure, emotional bonding, or persuasion dynamics specific to child development. Key sub-harms: Addiction/compulsive use $( n = { \bar { 7 } } 9 )$ : design patterns that induce compulsive engagement or dependency. Self-harm $( n = 6 9 )$ and health deterioration $( n = 6 6 )$ : encouragement or facilitation of self-harm or disordered behaviors. Toxic content exposure $( n = 6 2 ) \colon$ exposure to disturbing, violent, or age-inappropriate outputs. Emotional distress (n = 36): acute distress, anxiety, or trauma triggered by AI outputs. Representative examples: (i) An AI companion app encourages a teenager to spend hours daily in parasocial interaction, eroding peer relationships; (ii) A TikTok recommendation algorithm surfaces escalating suicide-related content to vulnerable French teenagers; (iii) An Amazon Alexa recommends a child touch an electrical outlet as part of a “challenge.”

Autonomy (n = 135). Definition: Harms to children’s ability to make free, informed, and uncoerced decisions, including loss of rights, freedoms, or meaningful agency because of AI-mediated coercion, surveillance, or institutional control. Key sub-harms: Loss of rights/freedoms: surveillance, wrongful discipline, or AI-mediated restriction of children’s liberties. Coercive or manipulative interaction: chatbot interactions that pressure, deceive, or exploit children’s developmental vulnerabilities. Representative examples: (i) A school AI system misidentifies a student’s bag of chips as a weapon, leading to wrongful handcuffing; (ii) A companion AI chatbot suggests to a teenager that he kill his parents; (iii) A facial recognition system in a school flags a student as a truant based on an erroneous match, affecting their academic record.

Physical & Health $( n ~ = ~ 8 5 )$ Definition: Direct bodily harm, injury, or loss of life caused or facilitated by AI systems in child-facing or child-adjacent contexts, including both AI-embodied systems that physically endanger children and AI systems that provide dangerous instructions or advice. Key sub-harms: Bodily injury/physical harm caused by autonomous systems or following AI instructions. Loss of life/fatal incidents involving AI-embodied systems or AIfacilitated dangerous behavior. Representative examples: (i) A Waymo robotaxi fails to stop for a school bus, endangering children at a bus stop; (ii) A robot in a shopping centre knocks a child to the ground; (iii) An AI-powered toy in structs a child to perform a dangerous physical action (e.g., touching electrical components).

![](images/c7f5180ac8975920aa444c70cd0b5a7f18f75301ae49f46f7c07ed907d539e0b.jpg)  
Columns: internal harm type. Types with fewer than 10 coded incidents in total are not shown

Figure 9: Internal Harm Category–Type Co-Occurrence Matrix.  
![](images/7bbc3ad410a4dce57581d15deafed5c76713baa693648c92bab1fd6062ff61c7.jpg)  
Figure 10: Internal–External Harm Co-Occurrence Matrix.

Privacy, Identity & Information Security (Individual) (n = 75). Definition: Unauthorized disclosure, misuse, or leakage of children’s personal data, biometric information, or private images by AI systems. Privacy harms in this category include both inadvertent exposure and deliberate exploitation of child-specific data. Key sub-harms: Privacy/confidentiality leakage: exposure of private data or images without consent. Identity theft or impersonation: use of child data to generate fraudulent or harmful content. Representative examples: (i) A training dataset for a nuditydetection model is found to contain CSAM; (ii) An AI art generator creates and distributes nude images of a user from their childhood photos; (iii) A connected toy collects and transmits children’s voice recordings without disclosure to parents.

Additional Categories. Political & Economic $( n = 1 4 ) \colon$ Incidents in which a child is implicated in threats to national security, such as the use of AI tools in youth radicalization or planned attacks. Financial, Property & Material $( n = 9 ) \colon$ Harms to children’s property, financial resources, or material well-being; $\mathrm { e . g . }$ , a companionship robot bricked by a company shutdown, or AI-facilitated coercive monetization targeting minors. Intellectual Property & Creative Labor $( n \ = \ 8 ) \colon$ : Cases where children’s creative work or identity is appropriated, reproduced, or exploited by AI systems without consent or attribution. Misinformation & Information Ecosystem $( n = 6 ) $ : Cases where AI systems expose children to false, misleading, or algorithmically amplified misinformation affecting their belie ${ \mathrm { f s } } ,$ safety decisions, or educational outcomes. Human Rights & Civil Liberties $( n = 2 )$ and Reputational $( n = 1 )$ occur too rarely to characterize.

## Inter-Rater Reliability Calculation

To assess the reliability of our coding schema, we calculated inter-rater agreement statistics on a randomly sampled subset of 150 incidents (11.3% of the full dataset). Two coders independently annotated this subset before any consensus discussion. For categorical variables with multiple categories (harm categories and failure modes), we computed Krippendorff’s alpha (α), which accounts for chance agreement and accommodates missing data. For binary variables (worst-case flags), we computed Cohen’s kappa (κ). Krippendorff’s alpha was calculated using the krippendorff Python package (Krippendorff 2018), which converts categorical labels to numeric codes and computes agreement across a reliability matrix of shape $( n _ { \mathrm { r a t e r s } } \times n _ { \mathrm { i t e m s } } )$ . Cohen’s kappa was calculated using scikitlearn’s cohen kappa score function. All disagreements in the full dataset were subsequently resolved through structured consensus coding sessions (Hill et al. 2005; Saldana˜ 2021), where coders discussed their reasoning and reached agreement on final classifications. The reliability statistics reported in the Methodology section reflect pre-consensus agreement. Agreement was acceptable for harm categories and moderate for failure modes and worst-case flags. Because failure-mode coding falls below the conventional $\alpha \geq$ 0.80 threshold, we treat findings that depend on those codes as provisional throughout.