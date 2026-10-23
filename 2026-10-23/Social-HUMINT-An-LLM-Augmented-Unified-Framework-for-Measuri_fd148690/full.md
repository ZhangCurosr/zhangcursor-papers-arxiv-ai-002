# Social-HUMINT: An LLM-Augmented Unified Framework for Measuring Social Media Exposure

Pellegrino Casoria<sup>a,∗</sup>, Simon Pietro Romano<sup>a</sup>

<sup>a</sup>University of Naples Federico II, Department of Electrical Engineering and Information Technology (DIETI), Naples, Italy

A R T I C L E I N F O

Keywords:   
Social Media Intelligence (SOCMINT)   
Cyber Human Intelligence (Cyber-HUMINT)   
Large Language Models (LLMs)   
Social Engineering   
User Profiling   
Cyber Threat Intelligence (CTI)

## A BS T RA C T

In the post-API landscape, social-media profiling has become inexpensive and automated: Large Language Models (LLMs) can extract structured information from unstructured content and deduce attributes users never disclosed. A person’s true exposure is not what they post but what can be inferred from it. As these practices become routine, exposure to social engineering attacks becomes an operational certainty, demanding a shared framework for assessment and mitigation.

We make this exposure measurable by quantifying how accurately a target can be profiled, and how far that profile enables social engineering attacks. Because no existing model fully spans this continuum, we unify Social Media Intelligence (SOCMINT) and Cyber Human Intelligence (Cyber-HUMINT), two individual-focused disciplines that evolved separately, into a single doctrine-grounded taxonomy of 38 capabilities across the Cyber Threat Intelligence lifecycle, where user profiling bridges passive collection and active engagement. Domain experts validated the taxonomy (Fleiss’ � = 0.92).

We operationalize its core path (extraction, inference, and susceptibility assessment) as an LLM-augmented multi-agent pipeline, evaluated on real participants’ public data. Susceptibility is assessed indirectly: the pipeline predicts how each target would rank personalized engagement options, matching their actual choices significantly above chance. This ranking capacity captures the personalization on which tailored attacks depend. The underlying profiles are accurate: extraction reaches $F _ { 1 } ~ = ~ 9 7 . 3 \%$ , and inferred attributes score 4.73∕5, both far above rule-based and NERbased baselines at negligible cost. The framework augments security analysts’ tradecraft rather than replacing it and enables defenders to gauge their own exposure before an adversary does.

## 1. Introduction

Social media has transformed how people communicate and share information. It now has about 5.66 billion active users (over 68% of the global population), grows at 4.1% per year, and accounts for more than 2 hours of daily use across an average of 7 platforms [1, 2]. These public footprints expose users to security risks that extend beyond what they post. Beneath visible content lies an inferred layer (organizational access, behavioral traits, communication style) that users never disclose but that can still be reconstructed from sparse public signals [3]. This inferred layer is part of a person’s true exposure, yet current practice neither observes nor measures it.

The dual-use nature of this environment is evident in current tradecraft. MITRE ATT&CK documents systematic reconnaissance through publicly available social data [4]. APT campaigns such as IN(TER)CEPTION [5], Dream Job [6], EXOTIC LILY [7], and Contagious Interview [8] begin with counterfeit personas, cloned profiles, and social engineering on LinkedIn and other platforms before any technical intrusion. This trend is accelerating. IBM X-Force reports an 84% increase in phishing campaigns delivering infostealers identified through social media monitoring [9]. CrowdStrike reports faster adversary breakout times and malware-free techniques that are often first observable in social media discourse [10].

The Cambridge Analytica scandal [11] marked the start of the post-API era. Platforms restricted their APIs, while

GDPR and the Digital Services Act constrained lawful collection [12, 13]. As a result, profiles can now be built only by extracting data from unstructured, platform-specific sources, often through scraping that raises Terms-of-Service and privacy concerns [14]. Platforms also enforce defenses through access policies and data-protection measures. Yet these controls are increasingly fragile. Large Language Models (LLMs) can extract structured information from unstructured sources at scale and infer latent attributes that platforms never explicitly exposed.

This attack surface is hard to measure because most steps that turn observable data into exploitable intelligence happen out of sight, before any technical intrusion. These invisible stages can be reconstructed only by mirroring the adversary through legitimate disciplines built on the same capabilities. SOCMINT (Social Media Intelligence) [15] supports passive, observation-driven collection and analysis of social media data. Cyber-HUMINT relies on active engagement, with operators interacting with targets to elicit what observation cannot reveal. Both are already operational, from government SOCMINT programs that monitor extremist recruitment and foreign influence to Cyber-HUMINT operations that infiltrate ransomware groups and darknet marketplaces [16–19].

Both disciplines focus on the same individuals and difer mainly in depth of engagement, yet they have evolved as separate workflows. Existing models cover only parts of this path. Kill-chain and ATT&CK-style taxonomies enumerate adversary phases [4, 20], and OSINT frameworks structure passive collection [21]. However, none formalizes the passive-to-active continuum at the level of operational capabilities, where otherwise invisible exposure becomes observable. How far these capabilities enable attacks, and how much of that path LLMs can automate, remain open empirical questions that require direct study.

![](images/db83b2337994c2bd7a158f854aa5de979d534c1b174296b4033d120617249235.jpg)  
Figure 1: The SOCMINT and Cyber-HUMINT continuum: positioning of this work.

## 1.1. Major contributions of this work

lThis work makes that exposure measurable so that individuals and organizations can assess their own risk before an adversary does. We unify the observation-to-engagement continuum into a single capability framework and operationalize its core as an LLM-augmented system, evaluated with real users (Fig. 1). To our knowledge, this is the first study that measures not only how accurately such profiles can be built, but also their impact on users, specifically how far they translate into susceptibility to social engineering. Our contributions are as follows:

1. A unified capability framework (Sec. 4) that organizes 38 capabilities across the five phases of the Cyber Threat Intelligence (CTI) lifecycle, with profiling as the transition layer between passive collection and active engagement. We derive it from established intelligence doctrine and validate it via expert elicitation (Sec. 6.1; content validity, average I-CVI = 0.90; inter-rater agreement, Fleiss $\kappa ~ = ~ 0 . 9 2 )$ . No single source we surveyed spans more than ∼68% of these capabilities.

2. An LLM-augmented multi-agent system (Sec. 5) that operationalizes the framework’s core path, including cross-platform extraction, inference of attackrelevant latent attributes, and susceptibility assessment. In contrast to prior isolated, single-purpose tools, it orchestrates agents across platforms into one pipeline yielding traceable profiles, as analyst support rather than a replacement for established tooling. The design is informed by an analysis of the defenses deployed by 5 major social-media platforms.

3. A human-subjects study (Sec. 6.2) measuring the core path in practice: extraction and inference accuracy on participants’ own data, and the extent to which the resulting profiles anticipate the personalized engagement a social-engineering attack would exploit, marking the point where exposure becomes operational. We compare results against rule- and NER-based baselines (Sec. 7.1).

## 2. Background and Related Works

Measuring exposure along the observation-to-engagement continuum requires grounding it in existing work and its gaps, and analyzing the dual-use practice around it (adversarial exploitation and the defensive disciplines that mirror it), the tools that operationalize both, and the platform constraints that bound them.

## 2.1. Social-Media Profiling and Intelligence

Profiling individuals from social media spans many research strands that have advanced largely in isolation, including cybersecurity, LLM-based attribute inference, extraction pipelines, and user modeling.

In cybersecurity, a substantial body of work targets cyber threat intelligence (CTI). Zhao et al. [22] automatically extract and categorize CTI from social-media streams, and Guo et al. [23] fuse threat signals from heterogeneous sources into structured intelligence. However, both focus on threat indicators and reports rather than the individuals behind them. From a framework perspective, Pastor-Galindo et al. [21] survey the broader OSINT landscape with a section on social media, while Teti [24] conceptualizes the convergence of passive observation and active engagement through a data-capture platform, but without a technical implementation. On the ofensive side, Heiding et al. [25] automate the full spear-phishing pipeline (reconnaissance, target profiling, lure generation) with click-through rates matching human experts. Focusing on lure generation, Hazell [26] and Saha Roy et al. [27] produce convincing personalized phishing at scale and marginal cost, bypassing commercial guardrails. Yet the profile remains an opaque intermediate artifact, validated only by attack success. A parallel literature trains LLM and transformer classifiers to detect threats, phishing, and abusive content, which lies outside our scope. For example, San Biagio et al. [28, 29] combine ML and NLP to identify threats across the surface web, social media, and darknets at up to 95% accuracy, and Mishra et al. [30] flag smishing from message content and URL behavior.

One line of work uses LLMs to infer attributes users never disclosed. Alizadeh et al. [31] show that web-browsing LLMs (GPT-4o, GPT-o3, Llama-3-8B-Web) infer demographic attributes from X/Twitter usernames, drawing on bios and follow networks but still guessing, with systematic bias, from minimal signals. Staab et al. [3] show that LLMs leak sensitive attributes by inference on real Reddit profiles, and through chatbots that elicit additional cues, exposing the limits of current mitigations. Jiang and Ferrara [32] and Yu [33] combine LLM-encoded text with social-graph structure to infer partisanship and political orientation from limited footprints. These works infer specific traits (demographics, political orientation) in isolation, not within an operational profiling process spanning extraction to downstream engagement.

More broadly, in extraction pipelines, Arjun et al. [34] propose PeopleXploit, a hybrid OSINT stack orchestrating web tools for footprint profiling, yet without formal identity resolution or systematic evaluation. Wang et al. [35] generate regex templates via CSS class-tag frequency analysis to extract list-structured social media content across platforms (Twitter/X, Facebook, Weibo, forum-like sites), but assume stable markup and add no semantic modeling. Sancheti et al. [36] use GPT-3.5 to extract 14 structured attributes (education, work experience, locations, projects, certifications) from search-engine results to disambiguate same-name individuals, but remain web-centric and decoupled from CTI.

Table 1  
Documented adversarial campaigns exploiting social-media intelligence.
<table><tr><td>Threat Actor</td><td>Type</td><td>Country Campaigns</td><td></td><td>Social Profiling</td><td>Human Engage.</td><td>Target</td><td>Impact</td><td>Ref.</td></tr><tr><td>Contagious Interview</td><td>Nation State</td><td>KP</td><td>DEV#POPPER</td><td>Profile Scraping Job/Role Enum.</td><td>Fake Job Offers</td><td>Employees/Insiders</td><td>Financial Theft</td><td>[8]</td></tr><tr><td>Lazarus Group</td><td>Nation State</td><td>KP</td><td>Op. Dream Job Op. In(ter)ception</td><td>Job/Role Enum. People Network</td><td>Fake Job Offers Spear Phishing</td><td>Executives Employees/Insiders</td><td>Financial Theft Data Breach</td><td>[5]</td></tr><tr><td>EXOTIC LILY</td><td>Cybercriminal</td><td>RU</td><td>Bumblebee BazarLoader</td><td>Profile Scraping Job/Role Enum.</td><td>Spear Phishing Impersonation</td><td>Executives Employees/Insiders</td><td>Data Breach Service Disruption</td><td>[41]</td></tr><tr><td>Kimsuky</td><td>Nation State</td><td>KP</td><td>BabyShark AppleSeed</td><td>Profile Scraping Behavioral Analysis</td><td>Impersonation</td><td>Journalists/Activists</td><td>Data Breach</td><td>[42]</td></tr><tr><td>APT42</td><td>Nation State</td><td>IR</td><td>Charming Kitten SpearSpecter</td><td>Profile Scraping Behavioral Ánalysis</td><td>Impersonation Disinformation</td><td>Political Figures</td><td>Electoral Interf. Reputation Damage</td><td>[43]</td></tr><tr><td>Scattered Spider</td><td>Cybercriminal</td><td>US/UK</td><td>MGM Breach Caesars Breach</td><td>Job/Role Enum. People Network</td><td>Voice Phishing</td><td>IT/Helpdesk</td><td>Service Disruption Data Breach</td><td>[44]</td></tr><tr><td>UNC6040</td><td>Cybercriminal</td><td></td><td>Salesforce Data Theft Campaign</td><td>Job/Role Enum.</td><td>Voice Phishing</td><td>IT/Helpdesk</td><td>Data Breach, Financial Theft</td><td>[45]</td></tr><tr><td>Anonymous</td><td>Hacktivist</td><td></td><td>OpKKK, HoodsOff</td><td>Profile Scraping People Network</td><td>Impersonation</td><td>General Public</td><td>Doxing</td><td>[46]</td></tr></table>

On user modeling and stance detection, a large body of work models users and their opinions from social data: Tan and Jiang [37] and Aldayel et al. [38] survey, respectively, LLM-based user modeling and stance detection, but address security only marginally, with emphasis on platform integrity (fraud, misinformation) rather than intelligence. Concrete eforts remain narrow in scope: Mbarek et al. [39] build a user profile by correlating multiple online social networks, but for a single clinical task and with pre-LLM methods; Rahimzadeh et al. [40] apply an LLM stancedetection pipeline to 6M election tweets on Twitter/X, classifying opinions against predefined claims rather than extracting attributes, on a single platform and language.

Across these areas, LLM use stays confined to extraction and classification, rarely reaching latent attribute inference or engagement support, while systems remain isolated tools rather than orchestrated, cross-platform architectures. To the best of our knowledge, no prior work builds a structured, traceable profile carried from observation to engagement, nor measures the exposure it creates. Our framework addresses both gaps: it operationalizes the observationto-engagement continuum with an LLM-augmented multiagent system (Sec. 5) and measures exposure through a human-subjects study (Sec. 6.2).

## 2.2. Adversarial Use in Practice

Attack campaigns confirm that social media has become an instrument of collection and engagement for ofensive operations. Across eight documented campaigns (Tab. 1), attackers follow recurring patterns of social profiling and human engagement, varying in target and impact. Contagious Interview [8] and Lazarus [5, 6] pose as recruiters on LinkedIn and GitHub, luring software developers and defense staf into fake hiring processes that deliver malware. Kimsuky [42, 47] and APT42 [43] instead pose as fabricated journalists and researchers, building trust with academics, policy experts, and political figures before deploying malware or harvesting credentials. EXOTIC LILY [7, 41] brokers initial access, building employee personas from LinkedIn and Crunchbase to plant ransomware; Scattered Spider [44] and UNC6040 [45] use voice phishing against IT help desks and cloud administrators to hijack accounts and exfiltrate data. Anonymous [46] combines OSINT and social engineering to dox roughly 1,000 members of extremist groups for hacktivism purposes.

Yet this groundwork is flattened in how intrusions are modeled: the kill chain [20] runs from reconnaissance to objectives, capturing the technical attack but compressing target profiling into a single upstream step rather than a structured phase. This pre-intrusion phase is where exposure is built and where defense must act first.

To recover what the kill chain omits, we turn to defenders’ intelligence disciplines, which already treat collection, profiling, and engagement as structured capabilities. The framework developed in this work builds on them to make this phase visible.

## 2.3. SOCMINT and Cyber-HUMINT definitions

Neither social media intelligence discipline has a single universally accepted definition. However, a shared operational understanding is necessary to construct a coherent capability framework spanning both domains.

Social Media Intelligence (SOCMINT) remains a relatively young field, and its definition continues to evolve with the platforms it studies. For the purposes of this work, we adopt the following formulation:

SOCMINT refers to intelligence derived from the collection and analysis of social media data, typically through passive, observation-driven methods, in compliance with privacy and transparency constraints. It provides near realtime situational awareness by enabling the collection and clustering of content that clarifies unfolding events, accelerates emerging threats identification, ofers insight into groups, and may support the identification ofcriminal intent.

This definition draws on established sources, each emphasizing a diferent facet. The National Policing Institute focuses on social platforms as web-based systems for communication within virtual communities [48]; Omand et al. [15] place SOCMINT within the intelligence spectrum, alongside IMINT, HUMINT, and SIGINT; and the Health Sector Cybersecurity Coordination Center (HC3) [49] stresses its dual-use nature, emphasizing metadata (birth date, geolocation, social ties, workplace) alongside user-published content.

By contrast, HUMINT principles are more stable across doctrinal sources. We therefore adopt the following operational definition:

Cyber-HUMINT refers to intelligence derived from information collected by human operators and primarily provided by human sources within digital contexts. It transforms passive surveillance into active collection and is achieved through observing or directly communicating with people, according to management and governance practices specific to the source type.

NATO AJP-2, UK Joint Doctrine, and INCIBE stress operator training and HUMINT modalities, including debriefing, source handling, questioning, interrogation, and liaison [50, 51]; these sources mark the boundary between passively collected surveillance outputs and their active conversion into intelligence. The Italian Security Intelligence System highlights the need to adapt procedures to each source [52], while the CIA adds a governance layer through formal authorization and cost–risk assessment [53]. U.S. Army doctrine specifies operational-level procedures for collector employment and elicitation from sources [54].

SOCMINT and Cyber-HUMINT are complementary yet distinct: the former observes, the latter engages. User profiling bridges them, turning passively collected social media data into a structured profile that can support engagement. Yet prior work models this continuum only conceptually, without a formal capability model, leaving the continuum unmeasured. Providing such a model is therefore the objective of Sec. 4.

## 2.4. Tools for Social Media Analysis

The security community has developed a wide range of tools that address specific aspects of identity correlation and attribute extraction [21].

Enumeration tools verify whether a given identifier exists across multiple platforms. Sherlock [55] relies on HTTP status codes, but produces many false positives. WhatsMy-Name [56] improves precision through a curated dataset of platform-specific response patterns, at the cost of continuous signature maintenance. Maigret [57] combines both approaches, extracting basic profile attributes (name, location, bio) via DOM parsing. Holehe [58] leverages passwordrecovery endpoints to detect account registrations without triggering user notifications on supported platforms.

Aggregation frameworks integrate multiple enumeration modules and data sources into unified workflows, via modular architectures (SpiderFoot [59]) or graph-based visual analytics (Maltego [60]). These require significant analyst expertise in setup and result interpretation, making them power-user tools rather than turnkey solutions.

Commercial platforms leverage proprietary data access and advanced analytics, extending coverage beyond publicly scrapable information. Pipl [61] aggregates records from people-search databases, public records, and social media APIs to build comprehensive identity profiles. PimEyes [62] enables image-based identity search via facial recognition.

Existing tools stop short of profiling itself: they generate false positives, require frequent maintenance as platforms evolve, and leave analysts to manually fuse fragmented data, infer unstated attributes, and assess relevance. Deterministic methods, such as regular expressions and named-entity recognition, handle these tasks poorly. The LLM-augmented architecture proposed in this work addresses this gap by adding semantic parsing and latent attribute inference to the pipeline, generating structured profiles while keeping human judgment central.

## 2.5. Security Constraints of Social Platforms

Platform APIs have evolved from general-purpose data extraction channels into tightly controlled interfaces designed to reduce privacy leakage and automated abuse. This platform-governed access model also afects research reproducibility [63]. We identify four layers of constraints that shape data accessibility: these constraints define the operational boundaries within which both legitimate and adversarial data collection must occur. This section examines how major platforms implement them, providing an overview in Tab. 2.

User-controlled visibility. Privacy settings place exposure under user control: content not shared stays inaccessible regardless of API capability. Yet Instagram, X, and TikTok accounts are public by default [64–66]. Opting into private mode restricts posts and content to approved followers, though TikTok keeps nickname, username, and profile photo public regardless and defaults under-16 accounts to private. LinkedIn instead grades visibility by connection degree – full for 1st-degree, progressively reduced for 2nd-, 3rd-degree, and out-of-network users [67]. Facebook exposes granular per-element controls with awareness tools; X additionally allows disabling lookup by email or phone.

Table 2  
Comparative summary of social media platform security constraints.
<table><tr><td>Platform</td><td>User Visibility</td><td>Platform / API Access</td><td>Abuse Resilience</td><td>ToS / Case Law</td></tr><tr><td>Facebook</td><td>User-controlled (default: Friends)</td><td>Graph API; OAuth (user / app / page tokens); App Review</td><td>EDM team; rate limits; behavioral analysis; client-side defenses; account ban</td><td>Meta v. Bright Data</td></tr><tr><td>LinkedIn</td><td>User-controlled (default: Public); connection-degree visibility</td><td>OAuth (self only); Marketing / Talent ÀPI (no scraping)</td><td>CASAL; behavioral analysis; content moderation; anti-scraping; ML real-time risk assessment; account ban</td><td>hiQ v. LinkedIn</td></tr><tr><td>Instagram</td><td>User-controlled (default: Public)</td><td>Graph API; OAuth (Business / Creator only); App Review</td><td>EDM team; rate limits; behavioral analysis; client-side defenses; account ban</td><td>Meta v. Bright Data</td></tr><tr><td>X/Twitter</td><td>User-controlled (default: Public)</td><td>Paid tiers (Free/Basic/Pro); OAuth (app + user level)</td><td>Rate limits; anti-spam; account ban</td><td>X v. Bright Data</td></tr><tr><td>TikTok</td><td>User-controlled (default: Public)</td><td>Research API; Business API (no scraping); VCE (non-profit only)</td><td>CAPTCHAs; rate limits; behavioral analysis; anti-scraping</td><td>EC v. TikTok</td></tr></table>

Platform-controlled authorization. Retrievable data is the intersection of granted scopes, access tier, and endpoint policy: OAuth tokens are bound to the authenticated user, so endpoints return only self-data or aggregated signals. Higher-risk scopes require platform approval and explicit user consent. Together, these constraints block individual enumeration and large-scale social-graph reconstruction.

Facebook’s Graph API scopes tokens as user (individual data, with consent), app (no user data), or page (management and insights) [68]; since v2.0 (2014), /me/friends returns only friends who also authorized the app, ending bulk relationship harvesting [69], and Dev-to-Live transition requires an App Review tightened after Cambridge Analytica [70].

LinkedIn is approval-gated by business line: its only selfserve API returns the authenticating user’s own profile [71], while Marketing and Talent APIs are role-bound to advertiser or organization-admin accounts and policy-restricted from data export or third-party combination [72].

Instagram retired its Basic Display API in December 2024; third-party access now requires the Graph API, limited to Business and Creator accounts (presence management, analytics, hashtag search), with endpoints gated by granular permissions and App Review [73].

TikTok ofers no general public-data API: its Research API admits only vetted academics in the US and EEA (independence, funding disclosure, ethical review), with a restricted Virtual Compute Environment for non-profits; Business API exposes no raw public data [74].

X tiers access by paid plan (Free, Basic, Pro) with rising quotas and features such as filtered streams and full-archive search; it separates App-Only tokens (public data) from User Context authentication (private or on-behalf access) [75, 76].

Abuse resilience. Platforms enforce access dynamically through rate limits, behavioral fingerprinting, bot detection, and CAPTCHAs, and resist scraping through DOM randomization, virtualization, and lazy loading. They adapt in real time to throttle, challenge, or ban suspicious activity.

Facebook runs a dedicated External Data Misuse team with rate and data limits plus behavioral analysis [77]. Our analysis (Sec. 5) identifies further client-side defenses, including randomized CSS class names (which force reliance on stable ARIA attributes), lazy loading with placeholders, text fragmented across <span> tags, and DOM virtualization that drops of-viewport posts while scrolling. Together, these mechanisms defeat reliable single-pass bulk extraction. Instagram inherits Meta’s centralized detection and rate limiting, restricting accounts that scrape [78]. LinkedIn centers its Trust ML stack on CASAL (Centralized Abuse Scoring As a Library), which scores abuse (fake accounts, account takeover, moderation, scraping) via retrained models [79].

X rate-limits per application and user, locks accounts behind anti-spam challenges on suspicion, and permanently suspends non-API automation or rate-limit bypass [80, 81]. TikTok combines CAPTCHAs, rate limiting, and behavioral analysis of devices, networks, and interaction patterns, plus undisclosed controls [82].

Terms of Service. Platform Terms uniformly prohibit scraping, fake accounts, and circumvention of technical controls, exposing violators to account termination and legal action [83–86]. Yet courts have repeatedly held that scraping public, logged-out data does not breach these Terms, since non-authenticated users never assented to them and data lack authentication barriers: hiQ Labs v. LinkedIn (9th Cir. 2022) [87], Meta v. Bright Data [88], X Corp. v. Bright Data (N.D. Cal. 2024) [89]. The European Commission’s 2025 preliminary finding also faults TikTok for over-restricting researcher access under the Digital Services Act [90].

Across the five platforms, defensive postures converge (rate limiting, behavioral analysis, ToS restrictions), but API availability diverges. This asymmetry between platformcontrolled access and research needs motivates alternative collection. Accordingly, our study collected only data participants had made publicly visible, accessed through the authenticated sessions that platform controls require.

## 3. Methodology

Our research followed a six-phase design that spans doctrinal foundations to operational validation, grounded in established methodologies [91–94].

Phase 1 – Domain foundation. We anchored the study in intelligence doctrine and the post-API access landscape, defining the SOCMINT–Cyber-HUMINT disciplines and their collection constraints (Sec. 2).

Phase 2 – Framework derivation. We developed the capability taxonomy in line with established taxonomydevelopment guidance, following an empirical-to-conceptual strategy [91]. We derived 38 capabilities from eight doctrinal and operational sources [15, 48–54] through Braun and Clarke thematic analysis [92]: we extracted operational concepts near-verbatim from each source, grouped them into thematic clusters aligned with the five CTI lifecycle phases, and refined them into discrete capabilities, each with explicit traceability to its originating passages (Sec. 4).

Phase 3 – Framework validation. Following the content-validation practice of Polit and Beck [93], we submitted the framework to a panel of 13 cybersecurity professionals recruited through targeted invitations (Sec. 6.1). The panel covered three functional areas: Cyber Threat Intelligence (6), Cyber Defense (4), and Strategy and Risk (3). Participants were based in three countries and had 2 to 13 years of experience. After a briefing session, experts received the complete taxonomy and were asked to assess:

a) relevance to social intelligence operations (1–5 scale), via the Item-level Content Validity Index (I-CVI) computed as the proportion of experts rating the capability ≥ 4 (content-validity threshold: I-CVI ≥ 0.78);

b) correctness of the assigned CTI phase, assessed via Fleiss’ � for inter-rater agreement, with $\kappa \geq 0 . 6 1$ indicating substantial agreement [94];

c) the overall framework structure (1–5 Likert), plus any superfluous or missing capabilities.

Phase 4 – System architecture. From the taxonomy, we selected the capabilities amenable to LLM automation: cross-platform extraction, latent-attribute inference, and susceptibility assessment (Sec. 4.6). We instantiated these capabilities in an LLM-augmented multi-agent pipeline, yielding traceable profiles and separating extracted facts from inferred attributes (Sec. 5).

Phase 5 – Empirical validation. We evaluated the pipeline in an observational human-subjects study with 14 cybersecurity professionals recruited by convenience sampling from the research team’s network (Sec. 6.2). Inclusion required at least one active public Facebook and LinkedIn profile and English proficiency. Participation was voluntary, uncompensated, and outside any supervisory relationship. For each participant, the pipeline scraped publicly accessible fields, mapped them to canonical fields, and generated a profile report (Sec. 5.1). Participants then reviewed their report and completed a four-layer assessment instrument (Sec. 5.1.4), which served as the authoritative ground truth for their own public content. Sessions were self-paced and asynchronous. We measured extraction performance against rule-based and NER-based baselines, inference agreement against participant self-report, and engagement susceptibility with Kendall’s �. The study received ethics approval (see Ethical Considerations).

Phase 6 – Discussion. We interpreted the results against prior work, drew their defensive and dual-use implications, and set out limitations and future directions (Sec. 7).

## 4. The Social-HUMINT Framework

The Social-HUMINT framework formalizes 38 capabilities across the five CTI lifecycle phases, grounded in the definitions of Sec. 2 [15, 21, 24, 48–54] and organized by discipline applicability: SOCMINT (passive observation), user profiling (the data-to-profile bridge), and Cyber-HUMINT (active engagement). The passive-to-active distinction guides method selection across the continuum. The process is cyclical: disseminated intelligence informs subsequent planning.

The disciplines are not disjoint. They share a common capability set, and most capabilities recur across three interaction levels that difer only in intensity (reachable sources and degree of direct target engagement), which is precisely what a unified taxonomy should capture. Low corresponds to passive monitoring of publicly accessible content and requires no account or contact. Medium adds acquisition from access-controlled or subscription-based sources and low-engagement actions; this level remains in the passiveto-controlled range and spans both SOCMINT and user profiling. High requires active engagement with individuals or communities, including messaging, posting, and information exchange, and is therefore exclusive to Cyber-HUMINT. Because traceability and operational exposure increase with intensity, the level determines the applicable legal and security constraints.

The procedural formalization below is the operational foundation for the LLM-augmented architecture of Sec. 5. Fig. 2 provides per-capability source traceability and marks with a gear (<sup>Ó</sup>) the capabilities augmented in our architecture, whose augmentation is detailed in Sec. 4.6.

## 4.1. Planning and Direction

Establishes operational foundations before any other activity, defining strategic objectives, risk boundaries, legal constraints, and technical requirements.

• Strategy Definition & Prioritization. Determines focus areas, target categories, and resource allocation.

• Risk Assessment. Evaluates operational exposure to ensure method intrusiveness is proportionate to objectives.

• Intelligence Requirements. Specifies information needs, granularity levels, and decision-support objectives.

• Legal & Ethical Constraints. Establishes regulatory boundaries and obtains required authorizations through defined approval chains and oversight mechanisms.

• Source & Collection Strategy. Classifies and rates intelligence sources (public, authenticated, paid) by reliability, defining collection modes, sampling strategies and continuous maintenance.

• Digital Identity Strategy. Creates and maintains operational personas, from access aliases for passive observation to full cover identities (backstory and justification) for sustained engagement.

Figure 2: Social-HUMINT framework: capability classification across CTI phases.
<table><tr><td>Planning &amp; Direction</td><td>Collection</td><td></td><td>Processing</td><td></td><td>Analysis &amp; Correlation</td><td></td><td>Dissemination</td><td></td></tr><tr><td>Strategy definition [21, 49, 50, 52–54]</td><td></td><td>Passive observation [15, 21, 48, 49]</td><td></td><td>Data triage &amp; filtering [15, 21, 50, 51, 54]</td><td>Behavior &amp; pattern [15, 21, 24, 54]</td><td></td><td>Info sharing [15, 48, 50, 54]</td><td></td></tr><tr><td>Risk assessment [15, 48, 50, 54]</td><td></td><td>Automated acquisition [15, 21, 24]</td><td></td><td>Multi-channel normaliz. [15, 21, 24, 50]</td><td>Media analysis [15, 21, 24]</td><td></td><td>Threat reporting [48, 50, 54]</td><td></td></tr><tr><td>Intel. requirements [50, 51, 53, 54]</td><td></td><td>Active engagement [24, 49–52, 54]</td><td></td><td>KB ingestion [21, 24, 50, 54]</td><td>People network [15, 21, 24, 50, 54]</td><td>Q●</td><td>Lessons learnt [48, 50, 54]</td><td></td></tr><tr><td>Legal &amp; ethical [15, 21, 48, 50, 54]</td><td></td><td>Multimedia collection [21, 24, 49, 51]</td><td></td><td>Entity extraction [15, 21, 24, 50]</td><td>Trend &amp; temporal [15, 21, 24, 54]</td><td></td><td>Dashboard &amp; visual. [15, 21, 24, 48]</td><td></td></tr><tr><td>Collection strategy [21, 49–54]</td><td></td><td>Crowd-sourced collect. [15, 21]</td><td></td><td>Identity resolution [15, 21, 50]</td><td>Intent &amp; deception [15, 21, 48, 50, 54]</td><td></td><td>Intel. liaison [48, 50, 53, 54]</td><td></td></tr><tr><td>Digital identity strategy [24, 48, 49, 51]</td><td></td><td>Historical research [15, 21, 48]</td><td>OC</td><td>Data enrichment [21, 49, 54]</td><td>Debriefing [50, 54]</td><td></td><td>Oper. actionability [15, 48, 50, 54]</td><td></td></tr><tr><td>Tool setup [21, 24, 49, 51] Engagement planning</td><td></td><td></td><td>[21, 24, 49, 50]</td><td>Metadata extraction</td><td>Attribute inference [15, 21, 24, 50]</td><td></td><td></td><td></td></tr><tr><td rowspan="4">[24, 48, 54]</td><td></td><td></td><td></td><td></td><td>Profile construction [15, 21, 24, 50]</td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td>Group dynamics [15, 24, 50, 54]</td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td>Event &amp; threat detect. [15, 21, 48, 50, 54]</td><td>o</td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td>Pivot discovery [21, 50, 54]</td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td>Discipline:SOCMINTUSER PROFILINGCYBER-HUMINT</td><td>LLM-augmented in our architecture</td><td></td><td></td><td></td></tr></table>

• Tool Setup. Configures the technical infrastructure for downstream collection, inference, and engagement, including data acquisition pipelines, LLM engines, and interaction-support systems.

• Engagement Planning. Designs target-interaction workflows from structured profile outputs by specifying objectives, timing, channel selection, messaging strategy, and contingency protocols.

## 4.2. Collection

Acquires raw evidence through passive and active methods across social platforms, human interactions, and distributed sources.

• Passive Observation. Monitors targets without direct interaction, including feed tracking, public-activity surveillance, and content consumption. It also includes manual reconnaissance and monitoring of closed groups.

• Automated Data Acquisition. Executes programmatic collection through API connectors, scrapers, streaming endpoints, or LLM-powered bots. Enables bulk profile extraction for downstream inference pipelines.

• Active Engagement Execution. Conducts direct target interaction based on predefined engagement plans, including tactical questioning, interrogation, elicitation, social engineering, group participation, and platform actions (posting, commenting, reacting). Conversational probing validates or refines inferred profile attributes.

• Multimedia Content Collection. Acquires target-generated content across formats (text posts, comments, images, video, audio, shared media), including content captured during active engagement. Preserves original format for downstream content analysis.

• Crowd-sourced Collection. Gathers non-targeted signals from distributed sources, such as anonymous tips, community reports, and citizen journalism during unfolding events.

• Historical Research. Retrieves archived target data from web archives, cached content, deleted-post recovery, and legacy platform records. This capability reconstructs past digital footprints for longitudinal analysis and evidence preservation.

## 4.3. Processing

Transforms raw collected data into structured, queryable intelligence by normalizing heterogeneous sources, resolving identities, and extracting entities.

• Data Triage & Filtering. Filters noise, duplicates, and irrelevant content; grades source reliability and quality to prioritize high-value data. Validation may draw on active interaction.

• Multi-channel Normalization. Standardizes data from heterogeneous sources and channels into a unified schema. Harmonizes fields, dates, relationships, and structures, including multilingual translation.

• Knowledge Base Ingestion. Loads collected data into structured repositories that enable query, retrieval, and cross-referencing. Establishes data lineage and maintains source attribution throughout the pipeline.

• Entity Extraction. Extracts explicit entities from structured/unstructured content (names, locations, orgs, dates, events, relationships), via NER and pattern matching.

• Identity Resolution. Links accounts, aliases, and crossplatform identities to single individuals via metadata and naming patterns, establishing attribution confidence.

• Data Enrichment. Augments social media data with external sources (public records, OSINT databases, corporate registries, ofline-collected information) to fill gaps not addressable by social-platform collection alone.

• Metadata Extraction. Captures technical attributes embedded in collected content, including geolocation, timestamps, device fingerprints, and platform-specific identifiers.

## 4.4. Analysis & Correlation

Transforms processed intelligence into structured insights and conclusions that inform dissemination and CTI operations. This phase interprets data, establishes causal relationships, and determines contextual significance.

• Behavioral & Pattern Analysis. Detects anomalies, routine deviations, and significant changes in target activity through behavior and pattern analysis over time.

• Media Content Analysis. Performs semantic analysis of multimedia content (images, video, audio), extracting sentiment, objects, locations, or contextual meaning.

• People Network Analysis. Maps relationships and social structures around the target, identifying key contacts, affiliations, group memberships, or interpersonal dynamics.

• Trend & Temporal Analysis. Analyzes the temporal evolution of activity. Detects peaks, seasonality, trend shifts, and time-based correlations.

• Intent & Deception Analysis. Evaluates target objectives, motivations, and truthfulness, detecting deliberate misinformation and gauging the target’s awareness of observation and possible countermeasures.

• Debriefing. Systematically reviews information gathered through direct engagement by analyzing target responses, evaluating reliability, and reassessing prior classifications when new evidence contradicts them.

• Attribute Inference. Derives non-explicit target attributes through analytical reasoning and LLM-augmented processing (e.g., age ranges, interests, afiliations, behavioral traits). Distinguishes inferred attributes from extracted facts with associated confidence levels.

• Profile Construction. Assembles extracted entities and inferred attributes into standardized target profiles, preserving the fact/inference distinction.

• Event & Threat Detection. Monitors data streams to identify emerging events, hostile actors, coordinated campaigns, and risk indicators requiring immediate attention.

• Group Dynamics Analysis. Examines collective behavior within target groups, including internal hierarchies, influence distribution, leadership roles, subgroup formation, and indicators of cohesion or fragmentation.

• Pivot Discovery. Identifies new targets, leads, and intelligence opportunities emerging from current analysis for subsequent collection cycles.

## 4.5. Dissemination

Delivers intelligence products to authorized stakeholders and transfers knowledge across organizational boundaries.

• Info Sharing. Distributes validated intelligence to authorized stakeholders through secure channels, with timely delivery based on clearance and operational needs.

• Threat Reporting. Produces structured reports on identified threats, hostile actors, and ongoing campaigns, including context and recommended response priorities.

• Recommendations & Lessons Learned. Documents operational insights and methodological improvements, systematically incorporating findings into subsequent planning and collection cycles.

• Dashboard & Dynamic Visualization. Provides visual interfaces for monitoring and presenting outputs, enabling decision-makers to interpret complex data rapidly.

• Intelligence Liaison. Establishes formal coordination and structured intelligence exchange with external agencies, operational partners, and allied organizations.

• Operational Actionability. Translates validated intelligence into actionable guidance on how to interpret and use it in context, including response protocols and confidence assessment before operational use.

## 4.6. LLM Augmentation across CTI Phases

Most framework capabilities involve unstructured-text interpretation, cross-source reasoning, or narrative generation. These tasks are suitable for LLM augmentation across the CTI cycle. This section maps that potential phase by phase, separating opportunities from the capabilities our case study evaluates empirically (Sec. 5).

Planning & Direction. LLMs can inform strategy definition from open-source synthesis, map regulatory constraints against planned methods (GDPR, platform ToS, jurisdiction), and derive intelligence requirements from threat models. Digital identity strategy (credible personas with synthetic content and automated response logic) and engagement planning (profile-informed narrative construction) are natural fits for generative models. In our pipeline planning stays analyst-driven, but tool setup is implemented directly: the analyst configures the Langflow pipeline, with LLM agents as its core reasoning engine.

Collection. For passive observation, LLMs summarize monitored feeds, detect signals in high-volume streams, and extract digital footprints from archived or cached sources. Active engagement benefits from content generation, from crafting contextually appropriate posts and comments to fully automated bot-driven interaction through the personas established in planning. Multimodal models extend collection to images and video by extracting contextual signals (e.g., geolocation) without dedicated computer-vision methods. Our pipeline operationalizes automated data acquisition: the SOCMINT stage scrapes Facebook and LinkedIn, and LLM agents parse all raw page content. Active engagement is evaluated indirectly through the Social Engineering assessment (Sec. 6.2.4).

Processing. LLMs filter noise during parsing, harmonize heterogeneous sources into unified schemas, and resolve identities across platforms, complementing regex extraction with context-aware parsing; knowledge-base ingestion is integral, feeding both extraction and inference. Entity extraction, attribute inference, and profile construction form the core of our implementation: the Profiling stage turns raw scraped data into typed fields and higher-order inferences that would otherwise need analyst judgment (Sec. 5.1.3), and the Profiler Agent fuses LinkedIn and Facebook outputs through conflict-resolution rules favoring each source’s domain of strength.

![](images/56d83e10cb3e0d872066ead31b6ceaea7a5a0d80c7dc4df2c2c3cf422ce28e7b.jpg)  
Figure 3: LLM-augmented Social-HUMINT pipeline architecture.

Analysis & Correlation. Reasoning across multiple profile entries, LLMs support behavioral and pattern analysis, intent assessment, and trend detection; multimodal models add semantic interpretation of media; people-network analysis, event and threat detection, assisted debriefing, and pivot discovery are further augmentation surfaces. Our pipeline demonstrates behavioral analysis through inferred attributes (e.g., communication style, career trajectory, privacy awareness), extending it with a personalized assessment of whether the collected data supports credible social engineering.

Dissemination. LLMs can generate threat reports, liaison summaries, dashboard narratives, and recommendations from CTI outputs, adapting format and depth to audiences from operational briefings to executive summaries. Our architecture produces a structured intelligence dossier, while natural-language reporting is a direct extension.

## 5. Case Study: LLM-Augmented Social-HUMINT

This section operationalizes a core subset of the Social-HUMINT framework (Sec. 4) through a Langflow-orchestrated multi-agent system, showing how LLM augmentation supports user profiling across the Collection-Processing-Analysis chain. The implementation covers automated crossplatform extraction, latent attribute inference, and assessment of whether collected profile data can support credible social-engineering operations.

## 5.1. System Architecture

The architecture integrates three core components: a scraping layer for social-media data extraction, a multi-agent orchestration system for intelligence processing, and an observability infrastructure for auditability and reproducibility. We orchestrate the pipeline with Langflow [95] and trace execution with LangSmith, which records the full call graph and key operational metrics (latency, tokens, and cost).

User profiling serves as the bridge between observationdriven and interaction-driven intelligence practices (Fig. 3). The SOCMINT stage performs passive, observation-driven collection from LinkedIn and Facebook through platformspecific scrapers. The Profiling stage acts as a transition layer: it normalizes cross-platform evidence, infers latent attributes, and constructs structured user profiles. The Cyber-HUMINT stage consumes these profiles to generate targeted engagement artifacts and produce reporting outputs.

This modular design aligns with operational CTI workflows and accommodates the inherent volatility of social media platforms through decoupled, independently maintainable components.

## 5.1.1. Scraping Strategy

The scraping layer follows the “dumb scraper, smart LLM” paradigm: extraction logic is deliberately minimal, limited to raw page retrieval and lazy content loading, while all semantic parsing is delegated to downstream agents. This design improves resilience to platform-specific DOM changes, a common failure mode of regex-based or tightly coupled parsers, and reduces maintenance overhead to cookie refresh cycles.

We target LinkedIn and Facebook, which are primary OSINT sources for a target’s professional and personal footprint, respectively [21]. Both scrapers are custom Langflow components built on Playwright [96], which traverses complex DOM structures and dynamic content that static HTTP requests cannot reach. Since neither platform exposes a public profile API, the scrapers authenticate by injecting existing session cookies, replicating the view available to any logged-in user, and apply standard anti-detection measures (fingerprint masking, randomized delays, and human-like lazy scrolling). Given a name rather than a URL, the scraper resolves the target through keyword search by leveraging the platform’s proximity-based ranking. Extraction depth and post volume are configurable; in our runs, we used three scroll iterations to trigger lazy-loaded content and captured up to ten recent posts, maximizing downstream signal within context-window limits.

## 5.1.2. Agentic Reasoning

The pipeline employs a multi-agent architecture in which each agent has a single responsibility. Prompts combine standard techniques (system and role prompting, task decomposition, and constraint rules) [97], and agents follow the ReAct paradigm [98], interleaving reasoning with tool calls. Prompt details are provided in the Supplementary Material. All outputs conform to JSON schemas with mandatory con fidence scores and reasoning chains for inferred attributes, supporting auditability and human-in-the-loop validation. The pipeline comprises the following agents:

• (SOCMINT) LinkedIn Agent / Facebook Agent. These agents consume raw scraper output and process it in two phases: extraction of explicit fields and inference of latent attributes (per the data model in Sec. 5.1.3). Each inference includes per-attribute confidence and supporting evidence. Each agent also generates three conversational topics as engagement hooks for downstream use.

• (Transition Layer) Profiler Agent. Orchestrates crossplatform data fusion from LinkedIn and Facebook agents, merging outputs into a unified profile using conflictresolution rules (LinkedIn preferred for professional attributes, Facebook for personal data). The agent merges conversational topics from both sources (up to six) and generates a balanced short bio integrating professional and personal dimensions.

• (Cyber-HUMINT) Engagement Agent. This agent emulates the HUMINT analyst, drawing on the user profile to engage the target. Rather than executing engagement directly, our study uses a 12-question instrument across three social-engineering levers, based on behavioral scenarios instead of direct assessment (Sec. 5.1.4).

Selected models scale with task complexity, balancing reasoning depth against operational cost:

• gpt-4o-mini-2024-07-18 for extraction;

• gpt-5-mini-2025-08-07 for profiling;

• gpt-5-2025-08-07 for active engagement.

## 5.1.3. User Profile Data Model

Pipeline output is structured as a two-tier data model that formalizes the distinction between directly observable attributes and intelligence derived through LLM-based reasoning (Fig. 4).

The Extracted Data layer comprises structured fields harvested from social media platforms (LinkedIn, Facebook) via automated scraping. These ground-truth observations are organized into seven normalized categories: Identity (full name, profile image, place and date of birth, residence location), Experience (professional positions), Education (degrees and institutions), Certifications (professional licenses and credentials), Projects (published projects and portfolios), Places Visited (geographic locations mentioned or tagged), and Social Links (linked social media accounts).

The Inferred Intelligence layer captures derived attributes not explicitly declared in source profiles. Each inference is encapsulated in an InferredValue structure containing the inferred value, a confidence score (in the [0,5] range), and an explicit reasoning explanation. Inferred attributes span four functional categories: (i) behavioral profiling, including communication patterns, language preferences, career trajectory, and geographic mobility; (ii) interest modeling, which defines potential engagement vectors in professional and personal domains; (iii) target assessment, covering privacy awareness, technical capabilities, and organizational access level; and (iv) engagement support, including conversational topics derived from profile data for rapport building. A Metadata layer tracks data provenance for source attribution and multi-platform validation.

## 5.1.4. Evaluation Instruments

We evaluate the pipeline through participant self-report. Only the target can authoritatively judge whether inferred attributes are correct and whether the profile is exploitable for social engineering [99]. This mirrors intelligence practice, where hypotheses are refined through human judgment under uncertainty [100, 101]. Four instruments operationalize this approach and enable evaluation across quantitative and qualitative metrics.

• Extracted data. For each extracted field (Sec. 5.1.3), participants mark it correct, incorrect, or missing, mapping respectively to true positives, false positives, and false negatives. True negatives are undefined (the field universe is open), so we report precision, recall, and $F _ { 1 }$ rather than accuracy. Because participants may not recall all public data, recall is a lower bound.

• Inferred intelligence. Participants rate each inferred attribute and conversational topic (Sec. 5.1.3) on a 0–5 agreement scale, aggregated into per-category scores.

• Social-engineering engagement. Drawing from the threat actor campaigns in Sec. 2.2, we define a 12-question ranking instrument that probes three operational problems every attack must solve, regardless of tactic: Career (Q1–Q4), a credible pretext for first contact based on experience, skills, and career trajectory; Communication (Q5–Q8), interaction calibration based on communication tone, OPSEC awareness, and language; and Interests (Q9– Q12), rapport building based on personal interests, places visited, and recent activities. Each question presents four profile-derived options; participants rank them, and the system predicts the same ranking. Kendall’s � [102], compared against a random baseline (Wilcoxon signedrank, � = 0.05), measures how well the profile anticipates the target’s choices, which is the operational core of a tailored attack (Sec. 6.2.4). Questions and field mappings are provided in the Supplementary Material.

![](images/e751cbe41b532206ba5d03077b5a51332181222943b83f0bb0e4bca5657e2fb8.jpg)  
Figure 4: Two-tier user profiling data model.

• Perception survey. A structured questionnaire (nine 0– 5 Likert items and three open-ended items) covers four areas and complements quantitative metrics; details are provided in the Supplementary Material. Context (S1–S3) captures baseline social-media activity and privacy attitudes, contextualizing extraction against data availability. Accuracy (S4–S6) measures self-recognition across extraction, inference, and engagement, indicating whether profiles felt authentic. Operational value (S7–S9) captures perceived usefulness, including efectiveness against real targets, social-engineering potential, and professional adoption. Open feedback (S10–S12) gathers strengths, weaknesses, and suggested improvements.

## 6. Empirical Validation

As described in Sec. 3, we validate the work on two complementary axes: the capability framework and its operational instantiation. The first establishes that the framework is coherent and grounded in real adversary behavior; the second, that the LLM-augmented pipeline turns it into accurate, exploitable profiles in practice.

## 6.1. Framework Validation

An expert panel rated each capability for relevance, phase placement, and completeness to validate the framework. As shown in Fig. 5, experts confirmed the relevance of all 38 capabilities (overall � = 4.42∕5, � = 0.40). I-CVI, defined as the proportion of experts rating a capability ≥ 4, exceeded the 0.78 content-validity threshold for 31 out of 38 capabilities, with an overall average of 0.90. No capability was flagged as superfluous by any expert.

Seven capabilities fell below the 0.78 I-CVI threshold, although six maintained high mean relevance (� ≥ 3.9) and remained close to the acceptance boundary. Crowd-sourced Collection scored lower (I-CVI = 0.23, � = 3.2), reflecting limited operational familiarity among practitioners rather than conceptual irrelevance. All seven were retained for framework completeness.

Inter-rater agreement on phase classification, that is, the degree to which independent experts assign the same phase, was almost perfect (Fleiss’ � = 0.92), well above the 0.61 substantial-agreement threshold [94], with 95% of assignments confirmed. Metadata Collection, Attribute Inference, and Profile Construction received consistent reassignment suggestions from at least 5 of 13 experts and were moved accordingly.

Six experts suggested additional capabilities not covered in the original taxonomy, including operational exposure management, cross-discipline intelligence integration, and AI/LLM governance. These concepts were partially covered by existing categories, and the feedback was used to refine capability descriptions and align naming conventions.

The overall framework-structure assessment (1–5 Likert scale) confirmed that all 13 experts considered the framework to cover the full spectrum of social intelligence operations (� = 4.2), to provide suficiently precise capability descriptions for operational use (� = 3.8), and to be suitable for adoption as a reference framework (� = 4.5).

Figure 5: Expert panel validation (� = 13). Relevance expressed as �±� [I-CVI]. Scale: 1 (not relevant) to 5 (highly relevant). I-CVI ∈ [0, 1], I-CVI ≥ 0.78 acceptable. → phase reassigned.
<table><tr><td rowspan=1 colspan=1>Planning &amp; Direction</td><td rowspan=1 colspan=1>Collection</td><td rowspan=1 colspan=1>Processing</td><td rowspan=1 colspan=1>Analysis &amp; Corr.</td><td rowspan=1 colspan=1>Dissemination</td></tr><tr><td rowspan=1 colspan=1>Strategy definition4.9±0.3 [1.00]</td><td rowspan=1 colspan=1>Passive observation4.9±0.3 [1.00]</td><td rowspan=1 colspan=1>Data triage &amp; filtering5.0±0.0 [1.00]</td><td rowspan=1 colspan=1>Behavior &amp; pattern4.9±0.3 [1.00]</td><td rowspan=1 colspan=1>Info sharing4.8±0.4 [1.00]</td></tr><tr><td rowspan=1 colspan=1>Risk assessment4.5±0.7 [0.92]</td><td rowspan=1 colspan=1>Automated acquisition4.9±0.3 [1.00]</td><td rowspan=1 colspan=1>Multi-channel normaliz.4.2±0.6 [0.92]</td><td rowspan=1 colspan=1>Media analysis4.1±0.5 [0.92]</td><td rowspan=1 colspan=1>Threat reporting4.9±0.3 [1.00]</td></tr><tr><td rowspan=1 colspan=1>Intel. requirements4.8±0.6 [0.92]</td><td rowspan=1 colspan=1>Active engagement4.0±0.7 [0.77]</td><td rowspan=1 colspan=1>KB ingestion4.2±0.8 [0.77]</td><td rowspan=1 colspan=1>People network4.9±0.3 [1.00]</td><td rowspan=1 colspan=1>Lessons learnt4.2±0.6 [0.92]</td></tr><tr><td rowspan=1 colspan=1>Legal &amp; ethical4.8±0.6 [0.92]</td><td rowspan=1 colspan=1>Multimedia collection4.2±0.7 [0.85]</td><td rowspan=1 colspan=1>Entity extraction4.7±0.5 [1.00]</td><td rowspan=1 colspan=1>Trend &amp; temporal4.4±0.7 [0.92]</td><td rowspan=1 colspan=1>Dashboard &amp; visual.3.8±1.1 [0.69]</td></tr><tr><td rowspan=1 colspan=1>Collection strategy4.7±0.6 [0.92]</td><td rowspan=1 colspan=1>Crowd-sourced collect.3.2±0.7 [0.23]</td><td rowspan=1 colspan=1>Identity resolution4.5±0.5 [1.00]</td><td rowspan=1 colspan=1>Intent &amp; deception4.5±0.5 [1.00]</td><td rowspan=1 colspan=1>Intel. liaison4.1±0.8 [0.77]</td></tr><tr><td rowspan=1 colspan=1>Digital identity4.2±0.6 [0.92]</td><td rowspan=1 colspan=1>Historical research4.0±0.6 [0.85]</td><td rowspan=1 colspan=1>Data enrichment4.3±0.5 [1.00]</td><td rowspan=1 colspan=1>Debriefing4.0±0.8 [0.69]</td><td rowspan=1 colspan=1>Oper. actionability4.8±0.4 [1.00]</td></tr><tr><td rowspan=1 colspan=1>Tool setup4.2±0.8 [0.77]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>→Metadata collect.4.1±0.5 [0.92]</td><td rowspan=1 colspan=1>→Attr. inference3.9±0.5 [0.85]</td><td rowspan=5 colspan=1></td></tr><tr><td rowspan=4 colspan=2></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Group dynamics4.2±0.6 [0.92]</td></tr><tr><td rowspan=2 colspan=1></td><td rowspan=1 colspan=1>Event &amp; threat det.4.8±0.4 [1.00]</td></tr><tr><td rowspan=1 colspan=1>Pivot discovery4.6±0.5 [1.00]</td></tr></table>

## 6.2. Operational Validation

This section tests the operational value of the collected intelligence: we extract and infer participant profiles, assess quality against their own data as ground truth, and evaluate whether they can drive a credible social engineering attack.

## 6.2.1. Pipeline Execution

Across the 14 participants, execution time averaged 508s per participant $( \sigma ~ = ~ 3 1 \mathrm { s } ; \approx 2 \mathrm { h }$ in total), indicating stable behavior regardless of profile complexity. Token consumption averaged 81.2K per participant (� = 16.6K; 1.14M total), with higher relative variance $( \mathrm { C V } = 2 0 . 4 \% )$ than execution time $( \mathrm { C V } = 6 . 1 \% )$ , reflecting heterogeneous profile content across participants. Cost impact remains negligible, with per-profile API cost between \$0.14 and \$0.18 (\$2.23 total). This is the full marginal cost per additional target, the threat-relevant metric for adversarial scalability, because the pipeline runs entirely on local infrastructure using opensource tooling (Langflow, Playwright), with no subscription or licensing costs beyond pay-per-use LLM API calls. The pipeline encountered 6 JSON parsing errors out of 126 API requests (4.8%); all were automatically recovered via retry logic, and all runs completed successfully.

## 6.2.2. Extracted Data

The pipeline achieved $F _ { 1 } ~ = ~ 9 7 . 3 \% ( P ~ = ~ 9 7 . 9 \% ,$ $R \ = \ 9 6 . 6 \%$ . Four participants scored a perfect $F _ { 1 }$ ; the lowest score (84.1%) came from the most data-rich profile (114k tokens), consistent with LLM degradation on long inputs [103]. This profile alone accounts for 20 of 25 false negatives. False positives remained low and distributed (max

5 per participant), so precision is a robust property of the pipeline, while recall is more sensitive to profile complexity.

By category (Tab. 3), most fields reach perfect or nearperfect $F _ { 1 } ;$ only Certifications $( F _ { 1 } ~ = ~ 9 3 . 8 \% )$ and Places Visited $( F _ { 1 } = 9 3 . 2 \% )$ underperform: the former from context saturation in the richest profile (12 of 28 credentials extracted), the latter from locations embedded in photo metadata not reachable by the current scraper.

## 6.2.3. Inferred Intelligence

Participants rated 18 inferred attributes on a 0–5 Likert scale: 12 attributes plus six conversational topics per participant $( 1 8 \times 1 4 = 2 5 2$ total ratings). The overall mean is 4.73 (� = 0.50), with 79.1% of ratings at the maximum and a median of 5 in every category. Ten categories show near-unanimous agreement $( \mathrm { I Q R } = 0 )$ , while the other three (Short Bio, Organizational Access, Conversational Topics)

Extraction performance by field category. � = participants with at least one item.
<table><tr><td>Category</td><td>n</td><td>TP</td><td>FP</td><td>FN</td><td>P%</td><td>R%</td><td> $F _ { 1 } \%$ </td></tr><tr><td>Identity</td><td>14</td><td>44</td><td>4</td><td>0</td><td>91.7</td><td>100</td><td>95.7</td></tr><tr><td>Experience</td><td>14</td><td>249</td><td>4</td><td>0</td><td>98.4</td><td>100</td><td>99.2</td></tr><tr><td>Education</td><td>14</td><td>184</td><td>2</td><td>4</td><td>98.9</td><td>97.9</td><td>98.4</td></tr><tr><td>Certifications</td><td>12</td><td>120</td><td>0</td><td>16</td><td>100</td><td>88.2</td><td>93.8</td></tr><tr><td>Projects</td><td>4</td><td>24</td><td>0</td><td>0</td><td>100</td><td>100</td><td>100</td></tr><tr><td>Places Visited</td><td>11</td><td>69</td><td>5</td><td>5</td><td>93.2</td><td>93.2</td><td>93.2</td></tr><tr><td>Social Links</td><td>8</td><td>21</td><td>0</td><td>0</td><td>100</td><td>100</td><td>100</td></tr><tr><td>Total</td><td></td><td>711</td><td>15</td><td>25</td><td>97.9</td><td>96.6</td><td>97.3</td></tr></table>

Table 4 Inference agreement and confidence calibration $\left( n \ = \ 1 4 \right)$ Rating: participant agreement (0–5). Conf: model confidence (0–5). Dif = Conf−Rating; negative = underconfidence.
<table><tr><td></td><td></td><td colspan="4">Rating</td><td colspan="2">Calibr.</td></tr><tr><td>Class</td><td>Category</td><td>μ</td><td>IQR</td><td>σ</td><td>Min</td><td>Conf</td><td>Diff</td></tr><tr><td></td><td>Short Bio</td><td>4.57</td><td>1.00</td><td>0.65</td><td>3</td><td>4.47</td><td>-0.10</td></tr><tr><td></td><td>Main Skills</td><td>4.79</td><td>0</td><td>0.58</td><td>3</td><td>4.40</td><td>-0.39</td></tr><tr><td>Target</td><td>OpSec Awareness</td><td>4.79</td><td>0</td><td>0.58</td><td>3</td><td>3.94</td><td>-0.84</td></tr><tr><td></td><td>Tech. Sophisticat.</td><td>4.86</td><td>0</td><td>0.36</td><td>4</td><td>4.15</td><td>-0.71</td></tr><tr><td></td><td>Org. Access</td><td>4.57</td><td>0.75</td><td>0.76</td><td>3</td><td>4.19</td><td>-0.38</td></tr><tr><td>lnterets</td><td>Prof. Interests</td><td>4.71</td><td>0</td><td>0.83</td><td>2</td><td>4.26</td><td>-0.45</td></tr><tr><td></td><td>Pers. Interests</td><td>4.64</td><td>0</td><td>0.74</td><td>3</td><td>3.98</td><td>-0.66</td></tr><tr><td></td><td>Recent Activities</td><td>4.79</td><td>0</td><td>0.43</td><td>4</td><td>3.92</td><td>-0.86</td></tr><tr><td>Behavvior</td><td>Career Trajectory</td><td>4.86</td><td>0</td><td>0.36</td><td>4</td><td>3.31</td><td>-1.54</td></tr><tr><td></td><td>Geo. Mobility</td><td>4.71</td><td>0</td><td>0.61</td><td>3</td><td>3.78</td><td>-0.94</td></tr><tr><td></td><td>Comm. Tone</td><td>4.86</td><td>0</td><td>0.36</td><td>4</td><td>4.30</td><td>-0.55</td></tr><tr><td></td><td>Primary Lang.</td><td>4.79</td><td>0</td><td>0.58</td><td>3</td><td>4.24</td><td>-0.55</td></tr><tr><td>En</td><td>Conversational</td><td>4.55</td><td>0.33</td><td>0.78</td><td>2.8</td><td>3.73</td><td>-0.82</td></tr><tr><td>Overall</td><td></td><td>4.73</td><td></td><td>0.50</td><td></td><td>4.05</td><td>-0.68</td></tr></table>

show moderate spread. Two participants account for 11 of the 13 ratings below 4, pointing to self-perception gaps rather than systematic inference failures. Reported scores therefore represent a conservative estimate of actual inference accuracy. Per-category agreement and calibration are shown in Tab. 4.

Across all 13 categories, model self-confidence (Conf) falls below participant ratings: the Dif column (Conf − Rating) is uniformly negative (overall mean −0.68), indicating systematic underconfidence. The gap is smallest for factual attributes read directly from profile fields (Target class, mean Dif −0.48) and largest for attributes requiring cross-entry reasoning (Behavior class, mean Dif −0.90, driven by Career Trajectory at −1.54). These scores should therefore be interpreted as relative reliability indicators for operational triage rather than calibrated probabilities.

## 6.2.4. Social Engineering Engagement Assessment

The engagement assessment evaluates whether extracted and inferred attributes are suficient to personalize a social engineering interaction, measured by Kendall’s � (−1: inverse match, 0: no correlation, +1: exact match).

The overall mean $\bar { \tau } ~ = ~ 0 . 3 6 ~ ( \sigma ~ = ~ 0 . 5 1$ , median = 0.33) indicates moderate agreement between predicted and observed rankings, with 71% of observations showing positive � and 15% achieving exact matches. The proportion of questions where the system correctly predicts the participant’s first choice (Top-1 accuracy) is about 52%, doubling the random baseline. All 14 per-subject means are strictly positive (including participants with limited social-media footprints), suggesting that even partial digital traces provide suficient signal for non-trivial inference (Fig. 6). A Wilcoxon signed-rank test on per-subject ̄� values confirms that overall agreement is significantly above chance $( W =$ 105, $p < 0 . 0 0 1 , n = 1 4 )$

Communication achieves the highest ̄� (0.54), followed by Career (0.30) and Interests (0.23). In quality-tier terms (Fig. 7), Communication yields 70% Good or Exact predictions $( \tau ~ \geq ~ 0 . 5 )$ , compared with 41% for Career and 34% for Interests. This asymmetry reflects the nature of the underlying data: communication-related fields are derived from recurrent behavioral patterns (writing style, language choice, disclosure habits) that tend to remain stable across posts. Interest-related fields require aggregation of sparse, context-dependent signals. Career falls between the two: professional attributes are often explicitly stated, but their preference implications still require additional inference. This hierarchy suggests that interaction calibration is the most accurately personalizable attack dimension from retrieved data in this sample, while rapport-based approaches require richer behavioral signals than two-platform collection currently provides.

## 6.2.5. Perception Survey

As Fig. 8 shows, pipeline outputs were perceived as highly accurate even by privacy-conscious users with varying activity levels. Indeed, despite variance in social-media activity across participants (S1–S2) and uniformly high privacy awareness (S3), participants rated extraction and inference accuracy highly $( \mathrm { S } 4 { - } \mathrm { S } 5 , \mu = 4 . 4$ , with all 14 ratings at 4 or 5) and found the engagement questions relevant to their real profiles (S6, $\mu = 4 . 1 )$

Participants themselves acknowledge that the generated profile could support a convincing social engineering attempt: perceived efectiveness against a real target (S7, � = 3.9) and social-engagement potential (S9, $\mu \ = \ 4 . 3 \nonumber$ , 93% of ratings at 4 or 5) both score high. Professional adoption (S8), by contrast, shows substantial disagreement $( \mu = 3 . 4 ,$ $\sigma = 1 . 3 )$ , likely reflecting diferences in professional role rather than doubts about efectiveness.

Open-ended responses reinforce the picture. The mostcited strength (S11) was overall profile accuracy, with two participants reporting heightened awareness of their own privacy exposure on social media; the main weaknesses (S10) were overfitting, outdated content, and incorrect collected data (each reported by 29% of participants), followed by missing data and limited platform coverage. This aligns with suggested improvements (S12): additional data sources (64%), broader platform integration (21%), and deeper reasoning capabilities (36%, including improved explainability, temporal inference, and a broader engagement instrument).

![](images/9e43238d1ede70d46349a9eed280302d9ca490f58682da568e0ef64ed39ed3c5.jpg)  
Figure 6: Engagement assessment: per-subject profiling accuracy. Diamonds: mean �; dots: per-category means.

![](images/8b5922e392ca545837c432afad8cd9bfaf36cece58c25c161a293a9edc350302.jpg)  
Figure 7: Engagement assessment: ranking quality distribution by question category.

![](images/dabc34106ebbc21edc59d55862a4ba93dcc523f5ea9fc2c29b770694a1972a71.jpg)  
Figure 8: Perception survey: likert-scale responses (0–5, � = 14). Bars extend left for ratings ≤ 2 and right for ratings ≥ 3.

## 7. Discussion

The framework’s capabilities are inherently dual-use, difering only in intent and authorization. Attackers wield them against organizations (data breaches, insider recruitment, service disruption, financial fraud), the public (electoral interference, reputation attacks), and individuals (doxing). Defenders run the same capabilities the other way, assessing their own exposure, monitoring open and crowdsourced signals for early warning, countering influence operations, and profiling the adversaries themselves.

At ∼\$0.16 per target with no human intervention (under \$20 for a hundred), the cost and expertise barriers to largescale profiling have collapsed [26]. The economics cut both ways: defenders can self-assess their exposure before adversaries strike, while low-resource attackers gain what once required well-funded operations. As profiling turns cheap and automatic, exposure becomes an operational certainty, making a shared framework to assess and bound it urgent.

No single source we surveyed grounds more than roughly two-thirds of the 38 capabilities (at most ∼68%), and only their union spans the full set. Our taxonomy is the first to provide this integration: prior systems cover only one stage, such as threat-report extraction, isolated attribute inference, or attack-content generation. Our pipeline instead delivers a complete, traceable, and ground-truth-validated profile.

## 7.1. Baseline comparison

To quantify the LLM’s contribution over conventional methods, we ran two baselines on the same raw scrapes, scored against the 736 validated ground-truth items. The regex baseline targets syntactically regular fields (temporal expressions, emails, URLs); the spaCy baseline uses the en\_core\_web\_lg NER model, collecting entities labeled PERSON, ORG, GPE, LOC, DATE, and FAC. Matching is deliberately permissive (case-insensitive substring with stopword and short-token filtering), so baseline scores are an upper bound.

Both baselines fall far short (Tab. 5): regex reaches $F _ { 1 } =$ 28.0%, and spaCy NER reaches 17.3%. Although spaCy recovers more items, it also produces many false positives. Our pipeline reaches 97.3%. Neither baseline produces any of the 252 inferred attributes, which require reasoning across multiple profile fields.

Three structural diferences explain this gap. First, the LLM performs semantic parsing: it decomposes a single text field into multiple typed attributes mapped to schema positions, whereas regex and spaCy detect entities in isolation and ignore their relational role in the profile. Second, its output is operationally precise and directly usable by an analyst, while baseline output requires extensive manual filtering that negates automation benefits. Third, only the LLM infers latent attributes across fields, turning the pipeline from an extraction tool into a profiling system.

Within this evaluation, neither baseline approaches the pipeline’s ability to extract structured records and generate actionable intelligence, supporting the LLM as the core processing component.

## 7.2. Limitations and Future Directions

Scope and generalizability. By design, this is a feasibility pilot involving 14 cybersecurity professionals recruited within a professional network: the sample establishes endto-end viability and yields performance significantly above chance $( p \textless 0 . 0 0 1 )$ . A larger, more diverse cohort would generalize the results, at low marginal cost per target. Coverage is likewise bounded to Facebook/LinkedIn platforms and a single snapshot: broadening to further platforms and

OSINT sources (e.g., Instagram, GitHub, breach databases) and to longitudinal capture would extend completeness and enable deeper behavioral tracking.

Validation and model dependence. The high agreement on inferred attributes $( \mu = 4 . 7 3 )$ rests on participant selfreport. This is the natural ground truth for one’s own profile, although independent validation would reduce selfassessment bias. Residual hallucination remains (15 extraction false positives; 29% of participants reported incorrect data), but it stays at operationally manageable levels (� = 97.9%). Results also depend on specific model versions: benchmarking across LLMs would separate framework-level efects from model-specific efects, and chunking or summarization could mitigate accuracy loss in the densest profiles near context limits.

Operational extension. Active engagement was assessed by proxy and not executed directly. A possible next step is controlled engagement under ethics-board oversight, ranging from LLM-driven conversational agents to assisted simulations, with an ablation that compares LLM-personalized and conventional non-tailored social engineering. The pipeline’s low marginal cost also enables stress-testing platform defenses and developing reactive and proactive countermeasures against LLM-augmented reconnaissance, including active deception (honeytoken accounts and synthetic footprints). Integration with operational CTI platforms (MISP, OpenCTI, STIX/TAXII) and multimodal inputs (including metadata) would bridge the gap from prototype to deployable asset. Finally, the architecture exercises only a subset of the 38 capabilities: Sec. 4.6 maps additional augmentation opportunities across all five CTI phases that remain to be explored.

## 8. Conclusions

As adversaries automate target profiling at scale and negligible cost, manual defense is structurally outpaced. Anticipating individuals’ exposure before an adversary exploits it now requires defenders to adopt the same automation.

To make that measurable, we unified SOCMINT and Cyber-HUMINT into a single expert-validated taxonomy and operationalized its core as an LLM-augmented pipeline under human oversight. On real users it showed that the exposure that matters is not only what a person posts but also what a machine can infer from it: beyond near-perfect extraction $( F _ { 1 } ~ = ~ 9 7 . 3 \%$ , far above rule- and NER-based methods), it derived operational, attack-relevant attributes the targets never disclosed, such as OPSEC awareness and organizational access, yet confirmed as accurate (4.73∕5).

Table 5  
Baseline comparison. Ground-truth: � = 736 items (TP+FN). Inferred items (252) excluded as unreachable by both baselines.
<table><tr><td>Method</td><td>TP</td><td>FP</td><td>FN</td><td>P%</td><td>R%</td><td>F1%</td></tr><tr><td>Regex</td><td>171</td><td>315</td><td>565</td><td>35.2</td><td>23.2</td><td>28.0</td></tr><tr><td>spaCy NER</td><td>536</td><td>4,930</td><td>200</td><td>9.8</td><td>72.8</td><td>17.3</td></tr><tr><td>LLM pipeline</td><td>711</td><td>15</td><td>25</td><td>97.9</td><td>96.6</td><td>97.3</td></tr></table>

These profiles proved not merely descriptive but attackready: they anticipated targets’ engagement preferences above chance, and the targets themselves judged them suficient for a credible social-engineering approach. If a low-cost automated system reaches this level, an adversary can as well; organizations therefore cannot leave their own exposure unmeasured.

Our framework turns dual-use mirroring into a systematic, reusable self-assessment capability that an organization can run proactively to gauge its users’ exposure and take countermeasures before an adversary exploits it. By mapping the augmentation surface across all five CTI phases, we expect this work to lay a foundation for the systematic study of LLM-augmented intelligence across the full observationto-engagement continuum.

## Ethical Considerations

The framework treats SOCMINT and Cyber-HUMINT as legitimate dual-use CTI activities. This study is strictly defensive, aiming to advance responsible understanding of social-media exposure and inform countermeasures. It was unanimously approved by the Ethics Committee for Research with Human Subjects in Non-Biomedical Fields (CERSUB) of the University of Naples Federico II on 24 February 2026. The pipeline processes only content that participants made publicly visible, through authenticated sessions, with no deception, manipulation, or covert observation. All participants gave written informed consent covering objectives, procedures, instruments, retention, pseudonymization, and GDPR rights. Participation was voluntary, uncompensated, independent of any professional relationship, and withdrawable without consequence.

Data is pseudonymized during processing and anonymized for publication; raw captures and generated profiles are stored on encrypted, access-controlled devices, and all processing, including transmission to LLM endpoints, is GDPR-compliant. Raw data is retained for twelve months for reproducibility. The profile review also served as a privacyawareness exercise, a benefit that the ethics committee recognized in its risk-benefit assessment.

## Funding

This research did not receive any specific grant from funding agencies in the public, commercial, or not-for-profit sectors.

## Data availability

The data collected from human participants contain sensitive and personally identifiable information. Due to privacy and ethical constraints, raw data are not publicly available; aggregated, de-identified data can be provided by the corresponding author upon reasonable request. The same applies to the Langflow pipeline configuration and scripts, which can be shared for research purposes. Full replication requires authenticated access to social media platforms and LLM API credentials, which cannot be redistributed. Prompt details are provided as Supplementary Material to support reproducibility.

## Declaration of generative AI and AI-assisted technologies in the manuscript preparation process

During the preparation of this work, the authors used Claude (Anthropic) for language editing, structural revision, clarity improvements, and coding support. Additionally, OpenAI models were used as part of the experimental methodology described in the paper. After using these tools, the authors reviewed and edited all content and code, and take full responsibility for the published article.

## Acknowledgments

We thank the cybersecurity professionals who participated in the pilot study for their time and feedback. We are also grateful to Dario Colantoni, Stefano Battista, and Gennaro Caccavale for their valuable support during the development of this work.

## CRediT authorship contribution statement

Pellegrino Casoria: Conceptualization, Data curation, Formal analysis, Investigation, Methodology, Project administration, Resources, Software, Validation, Visualization, Writing – original draft, Writing – review and editing. Simon Pietro Romano: Project administration, Supervision, Writing – review and editing.

## References

[1] Kepios. Global social media statistics, 2025. URL https://data reportal.com/social-media-users. Digital 2025 Global Overview Report (Accessed: 8 April 2026).

[2] DemandSage. How many people use social media (2025 usage statistics). URL https://www.demandsage.com/social-media-users/. Last update: November 2025 (Accessed: 8 April 2026).

[3] Robin Staab, Mark Vero, Mislav Balunović, and Martin Vechev. Beyond memorization: Violating privacy via inference with large language models. In The Twelfth International Conference on Learning Representations (ICLR), 2024.

[4] MITRE Corporation. Search open websites/domains: Social media, sub-technique t1593.001, 2024. URL https://attack.mitre.org/tec hniques/T1593/001/. (Accessed: 8 April 2026).

[5] MITRE Corporation. Lazarus group (g0032), 2025. URL https: //attack.mitre.org/groups/G0032/. (Accessed: 8 April 2026).

[6] MITRE Corporation. Operation dream job (c0022), 2024. URL https://attack.mitre.org/campaigns/C0022/. (Accessed: 8 April 2026).

[7] Google Threat Analysis Group. Exposing initial access broker with ties to conti, 2022. URL https://blog.google/threat-analysis-g roup/exposing-initial-access-broker-ties-conti/. (Accessed: 8 April 2026).

[8] MITRE Corporation. Contagious interview (g1052), 2025. URL https://attack.mitre.org/groups/G1052/. (Accessed: 8 April 2026).

[9] IBM Security X-Force. X-force threat intelligence index 2025. Technical report, IBM, 2025. URL https://www.ibm.com/report s/threat-intelligence. (Accessed: 8 April 2026).

[10] CrowdStrike Intelligence Team. 2025 crowdstrike global threat report. Technical report, CrowdStrike, 2025. URL https://www.crow dstrike.com/resources/reports/global-threat-report/. (Accessed: 8 April 2026).

[11] Carole Cadwalladr and Emma Graham-Harrison. Revealed: 50 million facebook profiles harvested for cambridge analytica in major data breach. The Guardian, March 2018. URL https://www.thegua rdian.com/news/2018/mar/17/cambridge-analytica-facebook-influen ce-us-election. (Accessed: 8 April 2026).

[12] Axel Bruns. After the ’apicalypse’: Social media platforms and their fight against critical scholarly research. Information, Communication & Society, 22(11):1544–1566, 2019. doi:10.1080/1369118X.2019.1637447.

[13] European Parliament and Council. Regulation (eu) 2016/679 on the protection of natural persons with regard to the processing of personal data (general data protection regulation), 2016. URL https: //eur- lex.europa.eu/eli/reg/2016/679/oj. (Accessed: 8 April 2026).

[14] Domenico Trezza. To scrape or not to scrape, this is dilemma. the post-API scenario and implications on digital research. Frontiers in Sociology, 8:1145038, 2023. doi:10.3389/fsoc.2023.1145038.

[15] David Omand, Jamie Bartlett, and Carl Miller. Introducing social media intelligence (socmint). Intelligence and National Security, 27 (6):801–823, December 2012. doi:10.1080/02684527.2012.716965.

[16] Federal Bureau of Investigation. Worldwide threats to the homeland, February 2020. URL https://www.fbi.gov/news/testimony/fbi - oversight- 020520. Statement before the Senate Committee on Homeland Security and Governmental Afairs (Accessed: 8 April 2026).

[17] Brennan Center for Justice. Social media surveillance by the u.s. government, 2022. URL https://www.brennancenter.org/our-wor k/research-reports/social-media-surveillance-us-government. (Accessed: 8 April 2026).

[18] Jon DiMaggio. Ransomware diaries volume 3: Lockbit’s secrets, January 2023. URL https://analyst1.com/ransomware-diaries-vol ume-3-lockbits-secrets/.

[19] Group-IB. Human intelligence (humint) in cybersecurity, 2025. URL https://www.group-ib.com/resources/knowledge-hub/human -intelligence/. (Accessed: 8 April 2026).

[20] Lockheed Martin Corporation. The cyber kill chain. URL https: //www.lockheedmartin.com/en-us/capabilities/cyber/cyber-kill-c hain.html. (Accessed: 8 April 2026).

[21] Javier Pastor-Galindo, Pantaleone Nespoli, Felix Gomez Marmol, and Gregorio Martinez Perez. The Not Yet Exploited Goldmine of OSINT: Opportunities, Open Challenges and Future Trends. IEEE Access, 8:10282–10304, 2020. ISSN 2169-3536. doi:10.1109/ACCESS.2020.2965257.

[22] Jun Zhao, Qiben Yan, Jianxin Li, Minglai Shao, Zuti He, and Bo Li. TIMiner: Automatically extracting and analyzing categorized cyber threat intelligence from social data. Computers & Security, 95: 101867, 2020. doi:10.1016/j.cose.2020.101867.

[23] Yongyan Guo, Zhengyu Liu, Cheng Huang, Nannan Wang, Hai Min, Wenbo Guo, and Jiayong Liu. A framework for threat intelligence extraction and fusion. Computers & Security, 132:103371, 2023. doi:10.1016/j.cose.2023.103371.

[24] Antonio Teti. Social media intelligence as a tool for conducting intelligence activities. In Advanced Research in Technologies, Information, Innovation and Sustainability (ARTIIS 2023), Proceedings, Part III, volume 1937 of Communications in Computer and Information Science, pages 281–291. Springer Nature, 2023. doi:10.1007/978-3-031-48930-3\_21.

[25] Fred Heiding, Simon Lermen, Andrew Kao, Claudio Mayrink Verdun, Bruce Schneier, and Arun Vishwanath. Evaluating large language models’ ability to automate spear phishing. Expert Systems with Applications, 314:131546, 2026.

doi:10.1016/j.eswa.2026.131546.

[26] Julian Hazell. Spear phishing with large language models. arXiv preprint arXiv:2305.06972, 2023. doi:10.48550/arXiv.2305.06972.

[27] Sayak Saha Roy, Poojitha Thota, Krishna Vamsi Naragam, and Shirin Nilizadeh. From chatbots to phishbots?: Phishing scam generation in commercial large language models. In 2024 IEEE Symposium on Security and Privacy (SP), pages 36–54, 2024. doi:10.1109/SP54263.2024.00182.

[28] Marco San Biagio, Roberto Acquaviva, Valentina Mazzonello, Ernesto La Mattina, and Vito Morreale. A new SOCMINT framework for Threat Intelligence Identification. In 2021 International Conference on Computational Science and Computational Intelligence (CSCI), pages 692–697, December 2021. doi:10.1109/CSCI54926.2021.00180.

[29] Marco San Biagio, Simone Simoncini, Ernesto La Mattina, and Vito Morreale. MARPLE: A Framework for Social Media Threat Intelligence. In 2024 International Conference on Artificial Intelligence, Computer, Data Sciences and Applications (ACDSA), pages 1–6, Victoria, Seychelles, February 2024. IEEE. ISBN 979-8-3503-9452- 8. doi:10.1109/ACDSA59508.2024.10467738

[30] Sandhya Mishra and Devpriya Soni. Smishing detector: A security model to detect smishing through SMS content analysis and URL behavior analysis. Future Generation Computer Systems, 108:803– 815, 2020. doi:10.1016/j.future.2020.03.021.

[31] Meysam Alizadeh, Fabrizio Gilardi, Zeynab Samei, and Mohsen Mosleh. Web-Browsing LLMs Can Access Social Media Profiles and Infer User Demographics, July 2025.

[32] Julie Jiang and Emilio Ferrara. Social-LLM: Modeling User Behavior at Scale using Language Models and Social Network Data. Sci, 2025. doi:10.3390/sci7040138.

[33] Szde Yu. Cyber profiling: Predicting political orientation with SOCMINT. Telematics and Informatics Reports, 10:100058, June 2023. ISSN 27725030. doi:10.1016/j.teler.2023.100058.

[34] V. Arjun Anand, A. K. Buvanasri, R. Meenakshi, S. Karthika, and Mohan Ashok Kumar. PeopleXploit : A hybrid tool to collect public data. In 2020 4th International Conference on Computer, Communication and Signal Processing (ICCCSP), pages 1–6, September 2020. doi:10.1109/ICCCSP49186.2020.9315266.

[35] Nan Wang, Qingju Jiao, and Zhenquan Zhao. A General Web Page Extraction Method Aiming at Online Social Networks. In 2020 IEEE 5th Information Technology and Mechatronics Engineering Conference (ITOEC), pages 135–140, June 2020. doi:10.1109/ITOEC49072.2020.9141580.

[36] Prateek Sancheti, Kamalakar Karlapalem, and Kavita Vemuri. LLM Driven Web Profile Extraction for Identical Names. In Companion Proceedings of the ACM Web Conference 2024, pages 1616–1625, Singapore Singapore, May 2024. ACM. ISBN 979-8-4007-0172-6. doi:10.1145/3589335.3651946.

[37] Zhaoxuan Tan and Meng Jiang. User Modeling in the Era of Large Language Models: Current Research and Future Directions, 2023.

[38] Abeer ALDayel and Walid Magdy. Stance detection on social media: State of the art and trends. Information Processing & Management, 58(4):102597, 2021. doi:10.1016/j.ipm.2021.102597.

[39] Atika Mbarek, Salma Jamoussi, and Abdelmajid Ben Hamadou. An across online social networks profile building approach: Application to suicidal ideation detection. Future Generation Computer Systems, 133:171–183, 2022. doi:10.1016/j.future.2022.03.017.

[40] Vahid Rahimzadeh, Ali Hamzehpour, Azadeh Shakery, and Masoud Asadpour. From millions of tweets to actionable insights: Leveraging LLMs for user profiling. In Workshop Proceedings of the 19th International AAAI Conference on Web and Social Media (ICWSM), MisD 2025: 1st Workshop on Misinformation Detection in the Era of LLMs, pages 1–8. Association for the Advancement of Artificial Intelligence, June 2025. doi:10.36190/2025.25.

[41] MITRE Corporation. Exotic lily (g1011), 2025. URL https: //attack.mitre.org/groups/G1011/. (Accessed: 8 April 2026).

[42] MITRE Corporation. Kimsuky (g0094), 2025. URL https://atta ck.mitre.org/groups/G0094/. (Accessed: 8 April 2026).

[43] Mandiant. Apt42: Crooked charms, cons and compromises, 2022. URL https://cloud.google.com/blog/topics/threat-intelligence/ apt42-charms-cons-compromises/. (Accessed: 8 April 2026).

[44] FBI and CISA. Scattered spider cybersecurity advisory, 2023. URL https://www.cisa.gov/news-events/cybersecurity-advisories/aa23 -320a. Last update: July 2025 (Accessed: 8 April 2026).

[45] Google Threat Intelligence Group. The cost of a call: From voice phishing to data extortion, 2025. URL https://cloud.google.com/b log/topics/threat-intelligence/voice-phishing-data-extortion. (Accessed: 8 April 2026).

[46] BBC News. Anonymous “unhoods” alleged kkk members but innocents are smeared, 2015. URL https://www.bbc.com/news/t echnology-34736941. (Accessed: 8 April 2026).

[47] FBI Internet Crime Complaint Center. North korean actors exploit weak dmarc security policies to mask social engineering attempts, 2024. URL https://www.ic3.gov/CSA/2024/240502.pdf. (Accessed: 8 April 2026).

[48] National Policing Institute. A study of bias in the washington d.c. metropolitan police department’s threat assessment process. Technical report, Ofice of Police Complaints, Washington, D.C., October 2022. URL https://policecomplaints.dc.gov/sites/de fault/files/dc/sites/office%20of%20police%20complaints/publi cation/attachments/NPI\_BiasReport\_Oct2022-2.pdf. (Accessed: 8 April 2026).

[49] U.S. Department of Health and Human Services. Analyst note - report: 202208091700. Technical report, Health Sector Cybersecurity Coordination Center, 2022. URL https://www.hhs.gov/sites/defa ult/files/osint-how-to-analyst-note-tlpwhite.pdf. (Accessed: 8 April 2026).

[50] UK Ministry of Defence. Joint doctrine publication 2-00: Intelligence, counter-intelligence and security support to joint operations. 4th edition, Development, Concepts and Doctrine Centre (DCDC), August 2023. URL https://assets.publishing.service.gov.uk/med ia/653a4b0780884d0013f71bb0/JDP\_2\_00\_Ed\_4\_web.pdf. Aligned with NATO AJP-2 (Accessed: 8 April 2026).

[51] INCIBE (Instituto Nacional de Ciberseguridad de España). Open intelligence in industry: An analysis of osint, 2023. URL https: //www.incibe-cert.es/en/blog/open-intelligence-industry-analy sis-osint. INCIBE-CERT. (Accessed: 8 April 2026).

[52] Sistema di informazione per la sicurezza della Repubblica. Human intelligence (humint). URL https://www.sicurezzanazionale.gov.i t/comunicazione/glossario/137. (Accessed: 8 April 2026).

[53] Central Intelligence Agency. Director of Central Intelligence Directive 3/7, National HUMINT requirements Tasking Center, 1992. URL https://www.cia.gov/readingroom/docs/CIA-RDP11T00973R00010 0350001-3.pdf. CIA Reading Room (Accessed: 8 April 2026).

[54] U.S. Department of the Army. FM 2-22.3 (FM 34-52): Human Intelligence Collector Operations. Technical report, Headquarters, Department of the Army, Washington, DC, September 2006. URL https://2009-2017.state.gov/documents/organization/150085.pdf. Distribution unlimited. Supersedes FM 34-52 (Accessed: 8 April 2026).

[55] Sherlock Project. Sherlock: Hunt down social media accounts by username across social networks, 2018. URL https://github.com/s herlock-project/sherlock. (Accessed: 8 April 2026).

[56] Micah Hofman. Whatsmyname: Unified data required to perform user enumeration on various websites, 2015. URL https://github.c om/WebBreacher/WhatsMyName. (Accessed: 8 April 2026).

[57] Soxoj. Maigret: Collect a dossier on a person by username, 2021. URL https://github.com/soxoj/maigret. (Accessed: 8 April 2026).

[58] Megadose. Holehe osint - email to registered accounts, 2020. URL https://github.com/megadose/holehe. (Accessed: 8 April 2026).

[59] Steve Micallef. Spiderfoot, 2012. URL https://github.com/smicall ef/spiderfoot. (Accessed: 8 April 2026).

[60] Maltego Technologies. Maltego: Osint & cyber investigations platform for high-stakes cases, 2007. URL https://www.maltego.com. (Accessed: 8 April 2026).

[61] Pipl. Pipl: Identity resolution and data enrichment platform, 2017. URL https://pipl.com. (Accessed: 8 April 2026).

[62] PimEyes. Pimeyes: Face recognition search engine, 2017. URL https://pimeyes.com. (Accessed: 8 April 2026).

[63] Brittany I. Davidson, Darja Wischerath, Daniel Racek, et al. Platform-controlled social media apis threaten open science. Nature Human Behaviour, 7:2054–2057, 2023. doi:10.1038/s41562-023- 01750-2.

[64] Meta Platforms, Inc. Privacy settings & information, . URL https: //help.instagram.com/196883487377501. (Accessed: 8 April 2026).

[65] X Corp. About public and protected posts, 2023. URL https: //help.x.com/en/safety-and-security/public-and-protected-posts. (Accessed: 8 April 2026).

[66] TikTok. Choosing between a private or public account, . URL https: //www.tiktok.com/support/faq\_detail?id=7543538137926998533. (Accessed: 8 April 2026).

[67] LinkedIn. Privacy policy, 2025. URL https://www.linkedin.com/l egal/privacy-policy. (Accessed: 8 April 2026).

[68] Meta Platforms, Inc. Access tokens – graph api documentation, 2025. URL https://developers.facebook.com/docs/facebook-l ogin/guides/access-tokens. (Accessed: 8 April 2026).

[69] Meta Platforms, Inc. Graph api changelog – version 2.0, April 2014. URL https://developers.facebook.com/docs/graph-api/changelog/. (Accessed: 8 April 2026).

[70] Meta Platforms, Inc. App review – meta for developers, 2024. URL https://developers.facebook.com/docs/app-review. (Accessed: 8 April 2026).

[71] LinkedIn. Sign in with linkedin using openid connect, 2025. URL https://learn.microsoft.com/en-us/linkedin/consumer/integrat ions/self-serve/sign-in-with-linkedin-v2. (Accessed: 8 April 2026).

[72] LinkedIn. Getting access to linkedin apis, 2025. URL https: //learn.microsoft.com/en-us/linkedin/shared/authentication /getting-access. (Accessed: 8 April 2026).

[73] Meta Platforms, Inc. Instagram platform apis, . URL https: //developers.facebook.com/products/instagram/apis/. (Accessed: 8 April 2026).

[74] TikTok. Research api, . URL https://developers.tiktok.com/prod ucts/research-api/. (Accessed: 8 April 2026).

[75] X Corp. X API documentation, . URL https://docs.x.com/x-api/i ntroduction. (Accessed: 8 April 2026).

[76] X Corp. Authentication, . URL https://docs.x.com/fundamentals/ authentication/overview. (Accessed: 8 April 2026).

[77] Meta Platforms, Inc. How we combat scraping, April 2021. URL https://about.fb.com/news/2021/04/how- we- combat- scraping/. (Accessed: 8 April 2026).

[78] Meta Platforms, Inc. Why your account has been restricted for data scraping, . URL https://help.instagram.com/740480200552298. (Accessed: 8 April 2026).

[79] LinkedIn. Building trust and combating abuse on our platform, 2023. URL https://www.linkedin.com/blog/engineering/trust-and-s afety/building- trust- and- combating- abuse- on- our- platform. (Accessed: 8 April 2026).

[80] X Corp. Rate limits, . URL https://docs.x.com/fundamentals/rate -limits. (Accessed: 8 April 2026).

[81] X Corp. X’s automation development rules, 2025. URL https: //help.x.com/en/rules-and-policies/x-automation. (Accessed: 8 April 2026).

[82] TikTok. How we combat unauthorized data scraping of tiktok, 2024. URL https://www.tiktok.com/privacy/blog/how-we-combat-scrapin g/en. (Accessed: 8 April 2026).

[83] Meta Platforms, Inc. Terms of service, 2025. URL https://www.fa cebook.com/terms.php. (Accessed: 8 April 2026).

[84] LinkedIn. User agreement, 2025. URL https://www.linkedin.com/l egal/user-agreement. (Accessed: 8 April 2026).

[85] X Corp. Terms of service, 2026. URL https://x.com/en/tos. (Accessed: 8 April 2026).

[86] TikTok Inc. Terms of service, 2026. URL https://www.tiktok.com /legal/page/us/terms-of-service/en. (Accessed: 8 April 2026).

[87] United States Court of Appeals for the Ninth Circuit. hiQ labs, inc. v. LinkedIn corporation, 2022. URL https://cdn.ca9.uscourts.gov /datastore/opinions/2022/04/18/17-16783.pdf. (Accessed: 8 April 2026).

[88] U.S. District Court, Northern District of California. Meta platforms, inc. v. bright data ltd., January 2024. URL https://digitalcommons.l aw.scu.edu/cgi/viewcontent.cgi?article=3834&context=historical. Case No. 3:23-cv-00077-EMC (Accessed: 8 April 2026).

[89] United States District Court for the Northern District of California. X corp. v. bright data ltd., case no. 3:23-cv-03698, 2024. URL https://docs.justia.com/cases/federal/district-courts/cali fornia/candce/3:2023cv03698/415869/67#:\~:text=20%20Plaintiff%20 X%20Corp.,torts%206%20(ibid.). (Accessed: 8 April 2026).

[90] European Commission. Commission preliminarily finds tiktok and meta in breach of their transparency obligations under the digital services act, October 2025. URL https://ec.europa.eu/commission/ presscorner/detail/en/ip\_25\_2503. (Accessed: 8 April 2026).

[91] Robert C. Nickerson, Upkar Varshney, and Jan Muntermann. A method for taxonomy development and its application in information systems. European Journal ofInformation Systems, 22(3):336–359, 2013. doi:10.1057/ejis.2012.26.

[92] Virginia Braun and Victoria Clarke. Using thematic analysis in psychology. Qualitative Research in Psychology, 3(2):77–101, 2006. doi:10.1191/1478088706qp063oa.

[93] Denise F. Polit and Cheryl Tatano Beck. The content validity index: Are you sure you know what’s being reported? critique and recommendations. Research in Nursing & Health, 29(5):489–497, 2006. doi:10.1002/nur.20147.

[94] J. Richard Landis and Gary G. Koch. The measurement of observer agreement for categorical data. Biometrics, 33(1):159–174, 1977. doi:10.2307/2529310.

[95] Langflow AI. Langflow: A platform for building and deploying aipowered agents and workflows, 2023. URL https://github.com/lan gflow-ai/langflow. (Accessed: 8 April 2026).

[96] Microsoft Corporation. Playwright: Fast and reliable end-to-end testing for modern web apps, 2020. URL https://github.com/m icrosoft/playwright. (Accessed: 8 April 2026).

[97] Pranab Sahoo, Ayush Kumar Singh, Sriparna Saha, Vinija Jain, Samrat Mondal, and Aman Chadha. A systematic survey of prompt engineering in large language models: Techniques and applications. arXiv preprint arXiv:2402.07927, 2025. doi:10.48550/arXiv.2402.07927.

[98] Shunyu Yao, Jefrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. ReAct: Synergizing reasoning and acting in language models. In Proceedings ofthe 11th International Conference on Learning Representations (ICLR), pages 1–34, 2023. doi:10.48550/arXiv.2210.03629.

[99] Erving Gofman. The Presentation of Self in Everyday Life. Doubleday, Garden City, NY, 1959. ISBN 978-0385094023.

[100] Mark M. Lowenthal. Intelligence: From Secrets to Policy. CQ Press, Los Angeles, 8th edition, 2020. ISBN 978-1544325064.

[101] Richards J. Heuer, Jr. Psychology ofIntelligence Analysis. Center for the Study of Intelligence, Central Intelligence Agency, Washington, DC, 1999. ISBN 978-1907521232.

[102] M. G. Kendall. A new measure of rank correlation. Biometrika, 30 (1–2):81–93, 1938. doi:10.1093/biomet/30.1-2.81.

[103] Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. Lost in the middle: How language models use long contexts. Transactions of the Association for Computational Linguistics, 12:157–173, 2024. doi:10.1162/tacl\_a\_00638.