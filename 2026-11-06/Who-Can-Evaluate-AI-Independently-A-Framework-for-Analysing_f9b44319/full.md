# Who Can Evaluate AI Independently? A Framework for Analysing Evaluation Infrastructure

Sumaya Nur Adan<sup>1,2</sup>, Joanna Wiaterek<sup>2</sup>, Nourhan Abdelrhim<sup>3</sup>, Sydney Reis<sup>1</sup>, Robert F. Trager<sup>1</sup>, Benjamin Rosman<sup>3</sup>, Marina Jirotka

<sup>1</sup>University of Oxford

<sup>2</sup>Centre for AI Security and Access <sup>3</sup>University of the Witwatersrand

## Abstract

AI evaluation is increasingly central to AI safety and governance, yet taking part in an evaluation does not by itself confer authority over its design, its execution or the resources it depends on. This paper examines how infrastructure arrangements distribute that authority, and so shape who can evaluate AI systems independently. Drawing on Science and Technology Studies, political economy and applied evaluation standards, we introduce PAAI (Power, Access, Adequacy, Independence), a framework for examining who controls evaluation infrastructure, the terms on which it is available, whether it is adequate to the evaluation’s purpose and what autonomy it supports. We apply PAAI to twelve African AI evaluation projects (2022–2025) and to the documented roles of all ten member institutes in the three published joint testing exercises of the international network of AI Safety Institutes. We identify three recurring configurations: extractive collaboration, dependent innovation and constrained independence. African actors often contribute data, language expertise and task design while external institutions retain authority over infrastructure and evaluation frameworks; design authority without infrastructure authority never yielded more than partial independence. The joint testing exercises show a similar division among mostly high-income members: the institutes that led each exercise also ran the models, while Kenya, the network’s only African member, moved from annotation to task design with no documented model execution. Participation in evaluation does not by itself produce evaluative authority. For evaluation to serve as a credible governance mechanism, access must come with control over priorities, methods, publication and continuity.

## 1 Introduction

When Kenya’s AI Safety Institute participated in a 2025 international joint testing exercise on AI agents, Kenyan researchers constructed new tasks and tools on fraud, phishing and impersonation, translated and annotated test data in Kiswahili, and contributed to the cybersecurity analysis (International Network of AI Safety Institutes 2025b). Kenyan staf are credited co-authors of the resulting report (Seah et al. 2026). Singapore and the United Kingdom led the exercise and its principal testing and analysis activities, while no model execution by Kenya is documented. This division of work does not establish whether Kenya could have repeated the evaluation independently. It does, however, reveal a distinction between contributing substantive evaluation content and holding responsibility for the infrastructure and procedures through which it is tested. That distinction raises the question addressed in this paper: what forms of authority and access must accompany participation for evaluators to conduct and sustain assessments on their own terms?

This dynamic is not unique to Kenya. It reflects a structural tension at the heart of contemporary AI governance. By AI evaluation, we mean the empirical assessment of AI models and systems to establish their safety properties, capabilities and risks, through methods such as red-teaming, capability benchmarking and domain-specific risk assessment (Ganguli et al. 2022; Liang et al. 2023; Weidinger et al. 2023; Shevlane et al. 2023). Independent evaluation of this kind is increasingly recognised as essential to AI governance (Bengio et al. 2025), yet the infrastructure that makes rigorous evaluation possible is controlled largely by the entities being evaluated or by institutions in a small number of well-resourced countries (Sastry et al. 2024; Crawford 2021). By evaluation infrastructure we mean the material and institutional resources an evaluation depends on: the compute needed to run tests; access to the models being tested, whether only to their inputs and outputs or also to their internal workings (Casper et al. 2024); test datasets and benchmarks; tools for running and scoring tests; and the funding, agreements and institutions that keep these available over time (Ojewale et al. 2025). The result is not only a capacity gap but a control gap. Actors may contribute to evaluation processes while lacking authority over evaluation priorities, methods, publication and continuity.

Building on standards for independent third-party evaluation (Stosz et al. 2025) and work on external auditing of AI systems (Raji et al. 2022; Costanza-Chock, Raji, and Buolamwini 2022), we define independent evaluation as evaluation in which evaluators can exercise methodological and interpretive judgement and communicate their findings without undue influence from the developers whose systems are assessed or the organisations that provide the infrastructure for assessing them. This autonomy depends on two further conditions: the operational capability to execute and repeat tests, and the institutional continuity to sustain assessment over time. The three may support one another, but they are not interchangeable: an evaluator may publish independently while relying on precarious infrastructure, or possess substantial technical resources while facing restrictions on its methods or findings. We treat all three as necessary for independent evaluation.

Recent initiatives appear to address these gaps. Frontier AI developers ofer API credits for safety research (Anthropic 2024); cloud providers run academic programmes available in Africa (Google Cloud 2024; Microsoft Azure 2024); and international collaborations promise distributed evaluation capacity (International Network of AI Safety Institutes 2025d). Existing evaluation standards identify operating conditions that support the independence and transparency of particular assessments (Stosz et al. 2025), while research on AI infrastructure examines the distribution and governance of the resources on which AI development depends (Sastry et al. 2024). Our concern is how these questions intersect: how do infrastructure arrangements distribute the authority, practical capabilities and continuity required for independent evaluation? We address this question through a framework that connects control over evaluation design and resources to the conditions under which evaluators can use them, and apply it to evaluation in Africa and to international joint testing.

Statement of Contributions. We make four contributions. First, we introduce the PAAI framework (Power, Access, Adequacy, Independence), a framework designed specifically to characterise power arrangements in AI evaluation infrastructure, drawing on infrastructure studies, applied evaluation standards and the political economy of AI (Section 3). Second, we apply PAAI through structured thematic analysis of twelve African evaluation projects, producing a cross-case mapping of evaluation infrastructure in Africa across three evaluation types (Section 5). Third, we test whether these patterns reflect regional scarcity or the way shared evaluation is organised, by coding the role of every member institute in the three published joint testing exercises of the international network of AI Safety Institutes (Section 6). Fourth, we derive three governance propositions and a set of design conditions that explain why expanding evaluation access can widen participation while entrenching dependency, and what would need to change (Section 7).

## 2 Background and Related Work AI Evaluation as a Governance Mechanism

Beyond the methods noted above, AI evaluation includes bias auditing (Raji et al. 2020) and ethical risk assessment (Weidinger et al. 2021). A growing literature argues that such evidence is credible only if external evaluators can scrutinise models independently of developers (Raji et al. 2022; Anderljung et al. 2023; Mökander et al. 2024). Yet Raji et al. (2022) demonstrate that the conditions enabling credible third-party oversight are unevenly distributed: mode access, legal protections for auditors and institutional legitimacy all shape who can conduct consequential assessments. Developers decide who receives access to their models and in what form (Shevlane 2022); black-box access is insufficient for many rigorous audits (Casper et al. 2024); and researchers who probe models risk losing access or facing legal threats, prompting calls for a safe harbour for evaluators (Longpre et al. 2024). Field scans find that auditors lack shared standards, tools and protections, in an ecosystem that remains fragmented and concentrated in well-resourced contexts (Costanza-Chock, Raji, and Buolamwini 2022; Ojewale et al. 2025; Birhane et al. 2024). Reuel et al. (2025) map systematic gaps in who conducts AI social impact evaluations, finding that actors from the Global South are consistently underrepresented both as evaluators and as subjects of evaluation frameworks.

## Power Asymmetries in African AI Development

Scholarship on AI in the Global South documents systematic asymmetries in development capacity and governance participation. Abebe et al. (2021) demonstrate that Western control over Africa’s data and digital infrastructure shapes which AI applications are feasible, while Birhane (2020) frames this as “algorithmic colonization”, whereby systems developed elsewhere are deployed without addressing local needs. Mohamed, Png, and Isaac (2020) use decolonial theory to show how such patterns extend historical relations of power, and Okolo (2023) documents how global inequities in AI development capacity disadvantage actors in the Global South across the full AI pipeline. Fairness and evaluation concepts developed in Western settings often fail to transfer to other contexts (Sambasivan et al. 2021). Birhane et al. (2022) examine how participatory AI initiatives reproduce similar asymmetries: formal inclusion does not translate into substantive power over design, deployment or evaluation decisions. Png (2022) identifies a “paradox of participation”, whereby Global South stakeholders are formally included in governance initiatives yet excluded from resource allocation and priority-setting. Much AI labour, including annotation, is performed in the Global South with little control over its use (Gray and Suri 2019), while African-led communities such as Masakhane have shown what participatory, locally controlled research can achieve (Nekoto et al. 2020). We show that the paradox of participation is reproduced specifically within evaluation infrastructure arrangements.

## Material Barriers to Evaluation Infrastructure

Literature on compute access documents severe constraints in African contexts: only 5% of Africa’s AI talent has adequate computational resources (United Nations Development Programme 2024). Solatorio et al. (2024) document how speakers of low-resource languages incur higher financial and environmental costs per unit of useful output, compounded by currency volatility against provider pricing in US dollars, and most of the world’s languages remain poorly served by the data and tools that evaluation depends on (Joshi et al. 2020). Raji et al. (2021) document how benchmark availability shapes what can be evaluated. Lehdonvirta, Wu, and Hawkins (2024) map a structural “Compute North” versus “Compute South” divide, in which countries without local AI compute face systematic disadvantages not only in AI development but in the regulatory and evaluative capacity that compute enables. Even systems described as open depend on infrastructure controlled by a small number of firms (Widder, Whittaker, and West 2024). Most of this work concerns infrastructure for development rather than evaluation and treats barriers one at a time; PAAI examines them together, as conditions of evaluation.

## Infrastructure as Sociotechnical Power

Science and Technology Studies provides the theoretical foundation for understanding evaluation infrastructure as a power arrangement. Star (1999) argues that infrastructure is relational and political: invisible to those it serves well, and visible only when it breaks down or excludes. Bowker and Star (1999) demonstrate how standards and classification systems embody power relations, encoding particular assumptions about the world while marginalising others, and Lampland and Star (2009) show how quantifying and formalising practices shape what can be known and by whom. Applied to AI evaluation, it shows that compute programmes, API terms and benchmark frameworks are not neutral conduits: they shape who can produce credible knowledge about AI systems and whose knowledge counts. Crawford (2021) and Sastry et al. (2024) extend this to the political economy of AI infrastructure, showing that control over compute functions as a governance lever, and Luitse (2024) traces how cloud platforms have become central to that political economy. Whittaker (2021) shows that researchers who depend on industry infrastructure face structural pressures that compromise the independence of their findings even without explicit censorship.

## 3 The PAAI Framework

## The Gap This Framework Addresses

Existing tools address adjacent but distinct questions. AEF-1 asks whether a specific evaluation met minimum conditions of access, resources, analytic autonomy and methodological transparency (Stosz et al. 2025); research on political economy characterises who controls AI development infrastructure (Crawford 2021; Sastry et al. 2024); and work on participation and structural capture explains how inclusion can coexist with unequal authority (Png 2022; Whittaker 2021). AEF-1 presupposes an evaluator with suficient access to be assessed. PAAI asks the prior question: do the infrastructure arrangements available to a given evaluator make adequate access achievable, on its own terms and over time? The answer determines whether capacity gaps are contingent, and so addressable through more resources, or structural, requiring changes to the terms and governance of infrastructure arrangements themselves.

## Theoretical Derivation of Each Dimension

PAAI synthesises four scholarly traditions because evaluation infrastructure sits at their intersection and no single tradition covers it (Figure 1a).

Power draws on Star (1999) and Bowker and Star (1999), whose work establishes that control over the material and institutional conditions enabling a practice translates into authority over what that practice can know and produce. We identify two sub-dimensions of Power. Design authority concerns who defines evaluation frameworks, selects targets and tasks, sets priorities and authors findings. It operates at two levels: framework authority, which sets the rules of an evaluation, such as what counts as harm, which metrics and automated scorers are used and how results are analysed; and content authority, which contributes tasks, data or languages within a framework set by others. Infrastructure authority concerns who controls the compute, data, funding and model access on which evaluation depends, and the terms under which they are provided. We code infrastructure authority as held by the evaluator where it controls the resources for the funded period and they cannot be withdrawn at a provider’s discretion; philanthropic grants are therefore coded as evaluator-held and provider credits as partial. These sub-dimensions can be, and in African evaluation contexts frequently are, decoupled. Casper et al. (2024) demonstrate that infrastructure authority shapes what evaluation is possible regardless of design authority: an evaluator who holds full intellectual control over evaluation design but depends on black-box API access is structurally prevented from executing rigorous evaluation of many safety-relevant properties.

Access draws on the political economy of AI infrastructure (Crawford 2021; Luitse 2024), which documents how access terms are not neutral technical arrangements but instruments through which providers shape what evaluation can occur. Access asks: under what conditions can actors use evaluation infrastructure? This includes eligibility criteria, geographic restrictions, time limits, renewal, usage constraints and publication requirements; whether the evaluator runs models itself or works on outputs produced by others; and whether access is fragmented across several short-term sources.

Adequacy operationalises minimum standards from applied evaluation frameworks. AEF-1 holds that credible independent evaluation requires suficient technical access (at minimum open-ended query access, ideally access to model internals), adequate compute for iterative and exploratory testing, suficient time for thorough assessment, and analytic autonomy over evaluation scope, methods and the presentation of results (Stosz et al. 2025). We distinguish technical adequacy (rigour and scale) from contextual validity (whether an evaluation fits the deployment contexts it claims to inform) (Hutchinson et al. 2022). Adequacy asks whether an evaluator’s arrangements meet these conditions (National Institute of Standards and Technology 2023).

Independence operationalises two complementary accounts of how infrastructure dependency compromises evaluator autonomy. Png’s (2022) paradox of participation identifies the dynamic whereby Global South actors are formally included in governance processes while remaining excluded from resource allocation and priority-setting. Whittaker’s (2021) account of structural capture shows how researchers who depend on industry infrastructure face implicit pressure to avoid findings that might jeopardise future access. Independence, as defined in the introduction, asks: can evaluators select evaluation targets without provider approval, publish findings without restriction, sustain evaluation programmes across model releases, and conduct adversarial assessment without risk to their infrastructure access? We assess Independence as this capacity and record recognition (co-authorship, credited sections) separately, because credit for contributing to an evaluation is not the same as the ability to conduct one.

![](images/4192645f9160e8bfad6e39a0e6b4aa306688d2b8f5f2e89967c4b1b66b48456f.jpg)  
Figure 1: The PAAI framework. (a) Power has two parts. Design authority decides what is tested and how, and shapes whether <sup>1</sup>the evaluation fits its setting. Infrastructure authority controls the compute, data, funding and model access an evaluation runs on, and sets the terms of Access. Together they determine Independence. Each box gives the question PAAI asks. (b) Who holds each part of Power gives four configurations, each shown with an example from our data.

## What the Four Dimensions Together Enable

Applied alone, each dimension gives an incomplete picture: moderate Access with no design authority is a control problem, not a resource problem; African-led design with terminable access is a sustainability problem, not a design problem (Figure 1a).

Crossing the two sub-dimensions of Power gives four configurations that look similar on the surface but have diferent root causes and require diferent remedies (Figure 1b). Three emerge from the twelve African cases:

• Extractive collaboration: external or partial design authority with external infrastructure authority. African actors contribute labour, data or limited design inputs; Independence is minimal or absent.

• Dependent innovation: African-led design authority with external or absent infrastructure authority. African actors lead evaluation design but cannot execute or sustain it without infrastructure they do not control.

• Constrained independence: African-led design and African-led or partial infrastructure authority, limited by resources to scale or sustain evaluation.

A fourth configuration, execution without design, in which an actor runs tests that others designed, appears only in the joint testing data. Without all four dimensions, these configurations collapse into undiferentiated “resource gaps” with no clear remedy.

## Scope and Limitations of the Framework

PAAI is designed specifically for evaluation infrastructure, not AI infrastructure generally. It applies whether an evaluator assesses its own models or others’: in either case the question is whether the evaluator can choose, run, publish and continue its assessments without depending on those who supply its infrastructure. Independence from the developer of the assessed system matters additionally where evaluator and developer difer. Legitimacy, transparency and legal protections for auditors, identified by Raji et al. (2022) as prerequisites for consequential assessment, are not captured by PAAI and are directions for future extension. PAAI is an initial operationalisation, ofered for refinement through engagement with practitioners conducting evaluation in resourceconstrained settings.

## 4 Methodology

## Project Identification

We conducted structured documentary analysis of African AI evaluation projects carried out between November 2022 (the public release of ChatGPT) and December 2025. Sources searched included academic databases (ACM Digital Library, arXiv, ACL Anthology), organisational and policy reports, and public GitHub repositories. Search queries combined evaluation types with geographic and infrastructural terms (e.g. [evaluation type] AND “Africa”; [country name] AND (“compute” OR “API” OR “benchmark”)). Of over 200 candidate sources, 47 were assessed for eligibility and 12 projects met all inclusion criteria.

Projects were included if they evaluated an identifiable object (model behaviour, benchmark performance or an evaluation framework), concerned African languages, institutions, users or deployment contexts, and were documented well enough to characterise at least two PAAI dimensions (Appendix B). Ten projects report empirical evaluation of models or benchmarks. Two, a critique of a medical benchmark (Mutisya et al. 2025) and a systematic review of AI in African higher education (Sangwa et al. 2025), are meta-evaluative: they assess evaluation frameworks or deployment evidence rather than running models. We retain them as boundary cases: they are among the few documented cases of African actors exercising full framework authority, and their independence is assessed for the work they undertook, which did not require running models. Excluding them leaves all three configurations populated. Lead institutions are in Kenya, Uganda, South Africa and Rwanda, alongside pan-African networks; four projects are led from Singapore and the UK, the United States, or Spain. The sample is a structured mapping of visible evaluation infrastructure arrangements, not a census.

## Joint Testing Exercises

To test whether the patterns found in Africa reflect regional scarcity or the way shared evaluation is organised, we coded the role of every member institute in the three joint testing exercises published by the international network of AI Safety Institutes: a November 2024 pilot (U.S. AI Safety Institute, UK AI Safety Institute, and Singapore AI Safety Institute 2024), a February 2025 multilingual and cybersecurity exercise (International Network of AI Safety Institutes 2025c,d; Vij et al. 2026), and a July 2025 agentic exercise (International Network of AI Safety Institutes 2025a,b; Seah et al. 2026). The network was renamed the International Network for Advanced AI Measurement, Evaluation and Science in late 2025 (UK AI Security Institute 2026); we found no later published exercise. This yields 29 institute-by-exercise records (the United States did not take part in the third exercise). For each record we coded design authority, infrastructure authority (whether and how the institute ran models), access mode, adequacy issues for its language, and recognition; Appendix E gives the categories and full coding.

## Thematic Analysis and Case Characterisation

We applied thematic analysis (Braun and Clarke 2006) using the PAAI dimensions as predefined themes, following a template-driven approach in which theoretically derived categories are applied systematically across cases. For each project, we extracted metadata, evaluation characteristics, infrastructure inputs, resource provision terms, scope and stated constraints. Where possible, information was triangulated across publications, project documentation, repositories and public statements. We also engaged informally with participants from several projects to check our reading of published documentation; these exchanges were not used as data. In parallel, we analysed the published eligibility, duration, renewal and publication terms of the access programmes the projects relied on (Appendix D). Rather than assigning numerical scores, which would imply a precision inconsistent with qualitative evidence, we characterise each project with descriptive categories grounded in documentary evidence (Table 1). Three researchers coded every case independently against a shared codebook and resolved disagreements through discussion until consensus, refining code definitions where needed (McDonald, Schoenebeck, and Forte 2019). Codebook, justifications and evidence for both datasets are given in Appendices A–E.

## 5 Findings: Evaluation Infrastructure in Africa

## Multilingual Safety Testing

Multilingual safety testing requires native-speaker expertise, culturally appropriate test scenarios and model access for behavioural assessment across languages (Bayes et al. 2024; International Network of AI Safety Institutes 2025d). Three projects reveal a consistent pattern: African participation is high in linguistic labour and low in infrastructural control.

Power and control. In the AISI multilingual exercise, Singapore led the exercise, ran all tests for ten languages on its own platform, developed the annotation guide and metrics calculator, and led the analysis. Harm categories were discussed among participating institutes, but the instruments that put them into practice were Singapore’s; Kenya annotated and analysed Kiswahili (International Network of AI Safety Institutes 2025d). Kenya’s larger role in the agentic exercise is examined in Section 6. Masakhane’s Uhura benchmark was designed by an African research community but evaluated OpenAI and Anthropic models through commercial APIs, with free credits from OpenAI and several coauthors afiliated with a model provider (Bayes et al. 2024). This dependency confines evaluation to provider-exposed behaviour and excludes white-box analysis.

Access and adequacy. Neither the Kenyan exercises nor Uhura documents renewable access that would support longitudinal safety monitoring: Kenya’s access came through the exercise’s coordination, with no documented right to renew it, and Uhura’s API access rested on provider credits whose renewal terms are not established. Black-box access is already insuficient for rigorous evaluation of many safetycritical properties (Casper et al. 2024); whether this limits a given evaluation depends on the properties it sets out to assess.

Independence and evaluation priorities. Only one of the three projects introduced risks defined from African contexts: Kenya’s fraud and phishing tasks in the agentic exercise, which entered an evaluation whose harm categories, scoring and analysis were set elsewhere. Kiswahili-specific failures in the multilingual exercise, such as models giving partial harmful instructions despite disclaimers, emerged as findings within an externally defined taxonomy rather than as part of a Kenyan agenda (International Network of AI Safety Institutes 2025d). Uhura translated English benchmarks, risking the importation of Western evaluation assumptions (Hutchinson et al. 2022). Without control over evaluation priorities, African deployment risks remain underrepresented in safety testing.

## Capability Benchmarking

Capability benchmarking requires substantial compute, large-scale benchmark datasets and evaluation harnesses (Ojo et al. 2025; Tonja et al. 2024). Five projects reveal diferent infrastructure positions within a common structural constraint.

<table><tr><td>Project</td><td>Type</td><td>Power (D/I)</td><td>Access</td><td>Adequacy</td><td>Independence</td><td>Configuration</td></tr><tr><td>AISI Multilingual Joint Testing</td><td>Ml. safety</td><td>Ext / Ext</td><td>Mediated, time-limited</td><td>Externally defined</td><td>None</td><td>Extractive</td></tr><tr><td>Kenya AISI Agentic Testing</td><td>Ml. safety</td><td>Part / Ext</td><td>Coordinated; renewal not</td><td>Limited relevance</td><td>Minimal</td><td>Extractive</td></tr><tr><td>Masakhane Uhura</td><td>Ml. safety</td><td>AL / Ext</td><td>established Provider credits; renewal not</td><td>Translated benchmarks</td><td>Partial</td><td>Dependent</td></tr><tr><td>Sunbird AI ASR</td><td>Benchmarking</td><td>AL / AL</td><td>established Grant-funded</td><td>Contextually apt</td><td>High,</td><td>Constrained</td></tr><tr><td>Lelapa InkubaLM</td><td>Benchmarking</td><td>AL / Part</td><td>Provider credits</td><td>Rigorous</td><td>cost-limited Partial</td><td>Constrained</td></tr><tr><td>Wits/UJ SAfriSenti</td><td>Benchmarking</td><td>AL /AL</td><td>Own hardware +</td><td>coverage Sufficient for</td><td>High, volunteer-</td><td>Constrained</td></tr><tr><td>Masakhane AfroBench</td><td>Benchmarking</td><td>AL / Ext</td><td>cloud Multiple credits</td><td>scope Wide,</td><td>limited Partial</td><td>Dependent</td></tr><tr><td>IrokoBench</td><td>Benchmarking</td><td>AL / Ext</td><td>Five short-term</td><td>English-adapted Human</td><td>Partial</td><td>Dependent</td></tr><tr><td>Georgia Tech Medical LLM</td><td>Domain Risk</td><td>Ext / Ext</td><td>sources Data source only</td><td>translation Rigorous,</td><td>Not assessed‡</td><td>Extractive</td></tr><tr><td>Barcelona Fetal Ultrasound</td><td>Domain Risk</td><td>Ext / Ext</td><td>Data source only</td><td>external Rigorous,</td><td>Not assessed‡</td><td>Extractive</td></tr><tr><td>Qhala HealthBench critique†</td><td>Domain Risk</td><td>AL / None</td><td>Public documents</td><td>external Proposal-stage</td><td>Of critique only</td><td>Dependent</td></tr><tr><td>Rwanda Higher-Ed review†</td><td>Domain Risk</td><td>AL / None</td><td>Literature only</td><td>Policy-relevant</td><td>Of analysis only</td><td>Dependent</td></tr></table>

Table 1: Twelve African AI evaluation projects characterised through PAAI. Power = design authority / infrastructure authority (AL = African-led; Ext = external; Part = partial). Configuration: extractive collaboration, dependent innovation or constrained independence. <sup>†</sup>Meta-evaluative case; configuration provisional, since no models were run. <sup>‡</sup>No African author in a design or analysis role. Justifications are in Appendix C.

Compute access. Sunbird AI (Uganda) and Lelapa AI (South Africa) show the highest infrastructure capacity in the sample. Sunbird used H100/A100-class GPUs funded by a Hewlett Foundation grant, totalling about 20 GPUhours for its data-scaling study (Akera et al. 2025); Lelapa trained and benchmarked InkubaLM on eight A100 GPUs provided through free Microsoft Azure credits (Tonja et al. 2024). SAfriSenti used institutional hardware supplemented by Google Colab Pro+ and AWS EC2 instances, with no external funding (Mabokela, Primus, and Celik 2025), showing that constrained independence is achievable at modest cost, although reliance on volunteer annotators limits scale. AfroBench relied on two providers’ credits and IrokoBench on five, each time-limited and separately applied for (Ojo et al. 2025; Adelani et al. 2025). None of the five documents compute access secured beyond the current project, which would be needed to keep benchmarking as new models are released.

Benchmark control. Sunbird, Lelapa and the SAfriSenti team retained control over benchmark design, data and publication, the clearest examples of what African-led evaluation can achieve under present constraints. AfroBench achieved unprecedented coverage of 64 African languages but adapted many tasks from English datasets, raising questions about whether measured performance gaps reflect model failures or benchmark transfer artefacts (Peppin et al. 2025); IrokoBench’s human translation reduces but does not remove this concern.

Benchmarking after release. None of the projects documents access to pre-deployment versions or to the internals of proprietary systems. The evidence they produce therefore arrives after release decisions have been made.

## Domain-Specific Risk Assessment

Domain-specific risk assessment requires partnerships with domain institutions, contextually relevant data, and expertise bridging AI and specialised fields (Mutisya et al. 2025; Sendra-Balcells et al. 2023). Four projects, two of them extractive collaborations, show both the starkest form of that pattern and its limits.

Data provision without evaluation capacity. Hospitals in five African countries (Malawi, Egypt, Uganda, Ghana and Algeria) contributed fetal ultrasound scans to the Barcelona study, and ten clinicians and researchers from African institutions are credited co-authors for image collection (Sendra-Balcells et al. 2023). For the Georgia Tech and Google Research evaluation, knowledge of herbal remedies from ten

African countries was drawn from published literature and WHO guidelines through desk research, and no authors were afiliated with African institutions (Nimo et al. 2025). In neither case did African institutions design the evaluation or conduct the analysis. The Barcelona study ran on a single consumer-grade GPU, so the barrier was not hardware but the combination of expertise, institutional capacity and evaluation authority that the African sites were not positioned to supply. This is what Couldry and Mejias (2019) theorise as data colonialism, applied to AI knowledge production: knowledge leaves African contexts to inform evaluation conducted elsewhere, without a reciprocal transfer of evaluative capacity.

Qhala and Rwanda as partial exceptions. Qhala’s critique of HealthBench (Mutisya et al. 2025) and the Rwandan higher-education review (Sangwa et al. 2025) show that African-led assessment of evaluation frameworks is possible without external infrastructure, but both remain at the leve of critique and policy analysis rather than empirical evaluation. We classify both provisionally as dependent innovation: their autonomy is intellectual rather than operational, since the alternatives they propose cannot be implemented without infrastructure their authors do not control.

## Cross-Cutting Patterns

Independence follows infrastructure authority. Figure 2 sets out who held each kind of authority in every project. Where both were external, African participants in the AISI multilingual exercise had no documented independence; the Georgia Tech study identifies no African-afiliated author, and Barcelona’s African co-authors are credited with image collection rather than design or analysis. Where Kenya shared in the design (the agentic exercise), independence was minimal. Where African actors held design but not infrastructure authority (Uhura, AfroBench, IrokoBench), or only partial infrastructure authority (Lelapa), independence was partial. Only Sunbird and SAfriSenti, which held both, reached high independence. Because the ability to run and repeat tests is part of what we mean by independence, its link to infrastructure authority is partly built into the definitions. What the cases add is that design authority alone never raised independence above partial. With this few cases, it is a documented pattern, not a statistical result.

Participation without power. African actors contribute essential labour: task construction and language evaluation (Kenya), clinical data (Barcelona), domain knowledge through published literature (Georgia Tech), and annotation (SAfriSenti, AfroBench). Yet they frequently do not control evaluation frameworks, compute or research agendas. Contributing inputs is not equivalent to exercising authority over what is evaluated or how. This is Png’s (2022) paradox of participation reproduced within evaluation infrastructure.

Time-limited access as structural constraint. Every project relying on external infrastructure faced temporal limits: API credits expire, cloud grants end and coordination windows close. This precarity prevents longitudinal monitoring and constrains adversarial testing, which requires independence from provider goodwill (Whittaker 2021).

![](images/88b7e0af5b7af4b5c16a0f82e8035be937ddc25161a14b3b229ffba84124844b.jpg)  
Figure 2: Who held design and infrastructure authority in 1each African project, and the independence of African participants, grouped by configuration. Darker cells indicate more African authority or independence. Design authority alone never raised independence above partial. <sup>†</sup> Meta-evaluative case (no models run); configuration provisional. <sup>‡</sup> No African author in a design or analysis role (Georgia Tech: no Africanafiliated authors; Barcelona: African co-authors credited for image collection only).

Extractive knowledge flows and the AEF-1 floor. Data and domain expertise flow from Africa to external evaluators, while evaluation capability remains concentrated elsewhere. Measured against AEF-1, the closest available standard (Stosz et al. 2025), none of the eight evaluations with an African evaluator documents all of its minimum operating conditions for African participants. Few academic evaluations anywhere would; what matters is which conditions fail and why. The closest, Sunbird and SAfriSenti, are limited by funding rather than by providers. The Georgia Tech and Barcelona studies, in which no African author held a design or analysis role, and the two meta-evaluative cases cannot be assessed against it. Extractive collaboration is incompatible with analytic autonomy, and time-limited access can sufice for one evaluation while leaving continuity unsecured.

## 6 Findings: International Joint Testing

Joint testing is currently the most visible route by which states share evaluation capacity, and its participants are mostly high-income. If PAAI captured only scarcity, well-resourced members would hold design and infrastructure authority together. Figure 3 shows that several do not.

Leads run the models. In every exercise, the lead institutes also executed the evaluation and built its analysis tools (International Network of AI Safety Institutes 2025d,b). Framework authority and infrastructure authority coincide at the top of the arrangement.

Participants divide on execution. Below the leads, institutes separate by whether they ran models. Australia ran independent tests in both later exercises; France, Korea and the EU AI Ofice ran tests or variations in the agentic exercise. Two members have no documented model execution in any exercise: Kenya, a lower-middle-income country, and Canada, a high-income country. Canada translated, annotated and wrote the analyses for Cantonese and Farsi and contributed to the cybersecurity analysis, a position close to Kenya’s. The divide is therefore not reducible to national income. It reflects how joint testing allocates execution, although documents cannot tell us whether Canada’s role was chosen or constrained.

Kenya: design authority grows, infrastructure authority does not. In the multilingual exercise, Kenya annotated Kiswahili responses and authored the Kiswahili analysis. In the agentic exercise, it constructed new tasks and tools on fraud, phishing and impersonation, one of only two nonlead members to contribute new tasks, and also translated, annotated and contributed to the cybersecurity analysis (International Network of AI Safety Institutes 2025b). Kenyan staf are credited co-authors of both reports (Vij et al. 2026; Seah et al. 2026). This expanded Kenya’s content authority and brought regional risks into the evaluation. Yet the reports attribute the execution of Kenya’s tasks to the lead, and scoring and analysis used the lead’s instruments and framework. Of the four non-lead institutes that made substantive design contributions to the agentic exercise, Kenya is the only one that did not run models. The records show design authority without documented infrastructure authority, inside an arrangement intended to share capacity; they cannot show whether this division was chosen or constrained.

Agentic scoring was least reliable for participants’ languages. Automated judging in the agentic exercise was least reliable for Telugu, Hindi and Kiswahili (International Network of AI Safety Institutes 2025b). Telugu and Hindi were evaluated with no Indian member institution, and the pilot reported results for Swahili and Yoruba with no documented African role in that analysis (U.S. AI Safety Institute, UK AI Safety Institute, and Singapore AI Safety Institute 2024). The instruments built by the institutes holding infrastructure authority performed worst on the languages of participants who held least of it, consistent with gaps documented in safety datasets for African languages (Onuoha, Sunu, and Sikiru 2026). The records cannot show that the distribution of authority caused these failures, but they show that those best placed to detect them had least control over the instruments.

## 7 Governance Propositions

The configurations generate three governance propositions, each with an implication for how evaluation infrastructure is designed, funded and governed.

## Proposition 1: Access Without Control Can Reproduce Dependency

The evaluations in which resources sat most fully with external institutions, the AISI joint testing exercises and the Barcelona and Georgia Tech studies, are those in which African actors held the weakest Power and Independence, because the institutions holding the resources also controlled framework design and priority-setting. Kenya’s design role grew between exercises, yet the records give it no means to decide independently what counts as a safety concern in Kenyan contexts or to run that evaluation itself. This is a control gap that more resources, provided under the same terms, would not close.

Access that comes without authority over priorities, framework design, execution and publication reproduces participation without power, whatever its stated intentions. The condition for independence is not the volume of access but its governance: who decides what gets evaluated, by what methods, and who runs the evaluation.

Implication for joint testing coordinators and research partners: participants should receive direct model access and lead at least one strand, with analytic autonomy and transparent publication rights as baseline conditions.

## Proposition 2: Project-Based Funding Is Poorly Suited to Independent Evaluation

AEF-1 calls for suficient time for thorough assessment, and credible oversight of deployed systems requires the institutional continuity to monitor them over time. Project-based funding can support high-quality individual evaluations, but it is poorly suited to sustaining the standing infrastructure that independent evaluation requires. All twelve African projects ran on time-limited resources: research grants, provider credits and coordination windows close when projects end. None documents standing evaluation infrastructure that combines dedicated compute, a permanent evaluation team and a multiyear mandate to assess AI systems deployed in African contexts; Kenya AISI has the latter two but, on the public record, not the first.

This is a structural mismatch, not simply a funding shortfall. Sunbird and Lelapa show that individual projects can achieve high-quality evaluation, but project funding does not by itself produce the capacities independent evaluation requires: monitoring models across updates, building expertise over time and accumulating knowledge of African deployment contexts. Unless something outlasts the project, capability ends with it. For frontier AI developers and cloud providers, the corresponding condition is multi-year, renewable access, including access to model internals where an evaluation requires it.

![](images/4bb43dbb6f3fdf6a079b27931261bda73af3b012b0b02600d68d8cc5d393c5d3.jpg)  
Figure 3: Roles of the ten member institutes across the three published joint testing exercises. (a) Participation tier derived from 1coded design roles and documented model execution (n/p: not a participant; Undoc.: not documented, because the pilot report names only its leads). (b) Exercises with a lead or substantive design role against exercises in which the institute ran models; the dashed line marks equal counts.

Implicationfor international bodies and governments: the first wave of AI Safety Institutes concentrated standing evaluation capacity in a small number of high-income countries (Araujo, Fort, and Guest 2024). International bodies should support institutional, not merely project-based, evaluation infrastructure in Africa: multi-year funding, dedicated compute and authority over evaluation priorities.

## Proposition 3: Evaluation Framework Co-Design Is a Governance Problem, Not Only a Methodological One

When evaluation frameworks are defined externally, by AI Safety Institutes in other regions, by Western academic institutions or by AI providers, African evaluators cannot determine what counts as harm, risk or adequate performance in their deployment contexts. Uhura translated English benchmarks rather than developing benchmarks from African deployment realities; Kenya’s own tasks were scored with instruments designed elsewhere; AfroBench adapted tasks from English datasets. These are not only translation problems; they reflect a governance arrangement in which authority over evaluation design sits outside the contexts where AI systems are deployed.

Models produce more harmful content in low-resource languages (Yong, Menghini, and Bach 2024; Peppin et al. 2025), yet standard safety evaluations rarely test for these disparities (Deng et al. 2024), and in the agentic joint testing exercise the automated scorers were least reliable on exactly these languages. Such failures are predictable when frameworks do not include African practitioners in prioritysetting. Framework co-design is therefore a prerequisite for the legitimacy of evaluation evidence in AI governance, not a procedural nicety.

Implication for standard-setters and evaluation designers: co-design with resource-constrained evaluators should be treated as a central condition of AI accountability. Minimum standards for independent evaluation, including AEF-1, have so far been drafted primarily by organisations that already have the infrastructure to meet them, and risk encoding assumptions that do not hold in resource-constrained settings.

Table 2 turns each PAAI question into a design condition that can be checked against the published record of any exercise or programme, and applies it to Kenya in the agentic exercise.

## 8 Discussion

## Situating the Findings

The patterns identified here reproduce, in evaluation, dynamics documented across the AI pipeline (Birhane 2020; Couldry and Mejias 2019; Kwet 2019; Png 2022). PAAI specifies how they operate in evaluation infrastructure. The same structure appears in joint testing among mostly highincome members, which supports reading independence as a property of infrastructure arrangements rather than of evaluators alone. The documented gaps concern the conditions for sustained independent scrutiny, not only equity.

<table><tr><td>Dimension</td><td>Diagnostic question</td><td>Design condition for joint testing and access programmes</td><td>Applied: Kenya, agentic exercise</td></tr><tr><td>Power: design authority</td><td>Who defines the harms, tasks and metrics, and how results are scored?</td><td>Each participant leads at least one harm category or task family, including its metrics and scoring.</td><td>Partly met: contributed new fraud and phishing tasks, but did not lead a strand or set scoring.</td></tr><tr><td>Power: infrastructure authority</td><td>Who controls the compute, data, funding and model access?</td><td>Each participant receives model access and compute directly, not only through the lead&#x27;s platform.</td><td>Not met: tests were run on the lead&#x27;s platform; no record of Kenya running tests itself.</td></tr><tr><td>Access</td><td>On what terms, and for how long?</td><td>Access terms, duration and renewal are written down, shared with participants and multi-year.</td><td>Not demonstrated in the public record: access was mediated through the exercise; duration and renewal terms are not reported.</td></tr><tr><td>Adequacy</td><td>Is the evaluation rigorous and fit for the contexts it informs?</td><td>Tasks reflect participants&#x27; deployment risks, and automated judges are validated per language.</td><td>Partly met: regional tasks included, but automated judges were least reliable for Kiswahili.</td></tr><tr><td>Independence</td><td>Can participants publish, replicate and continue the evaluation?</td><td>Participants may publish separately and keep using the datasets and tools they helped build.</td><td>Partly met: credited co-authorship; no documented right to rerun or reuse.</td></tr></table>

Table 2: PAAI as a tool for diagnosis and design. The last column applies the questions to Kenya’s role in the agentic joint testing exercise.

## Limitations

First, both datasets are documentary: the twelve projects are those identifiable from public documentation that met our criteria (Appendix B), not a representative sample, and because reports may describe leads’ roles more fully than participants’, “not documented” means unreported, not absent. Second, documents cannot distinguish chosen from constrained non-execution: Canada, and possibly Kenya, may have specialised by agreement. Third, proprietary contract terms are undisclosed, so Access and Independence rest on observable indicators. Fourth, AEF-1 was designed for thirdparty evaluations of developers’ systems and fits some of our cases imperfectly. Fifth, PAAI characterisations are qualitative judgements, reached by consensus among three coders (Section 4) rather than through inter-rater statistics; the evidence for each is in Appendices C and E. Sixth, two of the three joint testing exercises also appear among the African cases through Kenya’s participation, so the comparison extends the African analysis rather than replicating it independently.

## Future Work

We plan to extend this study through fieldwork and interviews with practitioners in Africa and with joint testing participants, to establish how roles are negotiated and to refine PAAI with those who would use it. We also plan to apply PAAI to third-party evaluators in high-income countries and in other resource-constrained regions.

## 9 Conclusion

This paper asked not whether African actors participate in AI evaluation but whether the infrastructure arrangements they depend on provide the forms of control required for independent evaluation. Across twelve African projects in multilingual safety testing, capability benchmarking and domainspecific risk assessment, and across three international joint testing exercises, the answer is consistent: existing arrangements expand participation more readily than they transfer evaluative authority.

Independent evaluation capacity requires institutional infrastructure rather than access programmes, co-designed frameworks, and sustained, evaluator-controlled compute and model access rather than time-limited credits.

## Ethical Statement

This study relies on public documentation. Informal exchanges with practitioners were used only to check our reading; no personal data were collected and no interview data are reported. Our characterisations concern infrastructure arrangements, not the quality of anyone’s work. The main risk of adverse impact is that the critique is read as criticising collaboration or particular institutes; we argue that collaboration is necessary and should build independent capacity, not substitute for it.

## References

Abebe, R.; Aruleba, K.; Birhane, A.; Kingsley, S.; Obaido, G.; Remy, S. L.; and Sadagopan, S. 2021. Narratives and Counternarratives on Data Sharing in Africa. In Proceedings of the 2021 ACM Conference on Fairness, Accountability, and Transparency, 329–341.

Adelani, D. I.; Ojo, J.; Azime, I. A.; Zhuang, J. Y.; Alabi, J. O.; He, X.; et al. 2025. IrokoBench: A New Benchmark for African Languages in the Age of Large Language Models. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics (NAACL). ArXiv:2406.03368.

Akera, B.; Nafula, E.; Walukagga, P.; Yiga, G.; Quinn, J.; and Mwebaze, E. 2025. How Much Speech Data Is Necessary for ASR in African Languages? An Evaluation of Data Scaling in Kinyarwanda and Kikuyu. ArXiv:2510.07221.

Anderljung, M.; Smith, E. T.; O’Brien, J.; et al. 2023. Towards Publicly Accountable Frontier LLMs: Building an External Scrutiny Ecosystem under the ASPIRE Framework. ArXiv:2311.14711.

Anthropic. 2024. What is the External Researcher Access Program? https://support.claude.com/en/articles/9125743- what-is-the-external-researcher-access-program. Accessed: 2025-12-01.

Araujo, R.; Fort, K.; and Guest, O. 2024. Understanding the First Wave of AI Safety Institutes: Characteristics, Functions, and Challenges. https://www.iaps.ai/research/ understanding-aisis. Accessed: 2025-12-01.

Bayes, E.; Azime, I. A.; Alabi, J. O.; Kgomo, J.; Eloundou, T.; Proehl, E.; Chen, K.; Khadir, I.; Etori, N. A.; Muhammad, S. H.; Mpanza, C.; Thete, I. P.; Klakow, D.; and Adelani, D. I. 2024. Uhura: A Benchmark for Evaluating Scientific Question Answering and Truthfulness in Low-Resource African Languages. ArXiv:2412.00948.

Bengio, Y.; Mindermann, S.; Privitera, D.; et al. 2025. International AI Safety Report. https: //internationalaisafetyreport.org/publication/internationalai-safety-report-2025. Accessed: 2025-12-01.

Birhane, A. 2020. Algorithmic Colonization of Africa. SCRIPTed, 17(2): 389–409.

Birhane, A.; Isaac, W.; Prabhakaran, V.; Díaz, M.; Elish, M. C.; Gabriel, I.; and Mohamed, S. 2022. Power to the People? Opportunities and Challenges for Participatory AI. In Equity and Access in Algorithms, Mechanisms, and Optimization, 1–8.

Birhane, A.; Steed, R.; Ojewale, V.; Vecchione, B.; and Raji, I. D. 2024. AI Auditing: The Broken Bus on the Road to AI Accountability. In 2024 IEEE Conference on Secure and Trustworthy Machine Learning (SaTML).

Bowker, G. C.; and Star, S. L. 1999. Sorting Things Out: Classification and Its Consequences. Cambridge, MA: MIT Press.

Braun, V.; and Clarke, V. 2006. Using Thematic Analysis in Psychology. Qualitative Research in Psychology, 3(2): 77–101.

Casper, S.; Ezell, C.; Siegmann, C.; Kolt, N.; et al. 2024. Black-Box Access is Insuficient for Rigorous AI Audits. In Proceedings of the 2024 ACM Conference on Fairness, Accountability, and Transparency, 2254–2272.

Costanza-Chock, S.; Raji, I. D.; and Buolamwini, J. 2022. Who Audits the Auditors? Recommendations from a Field Scan of the Algorithmic Auditing Ecosystem. In Proceedings of the 2022 ACM Conference on Fairness, Accountability, and Transparency, 1571–1583.

Couldry, N.; and Mejias, U. A. 2019. Data Colonialism: Rethinking Big Data’s Relation to the Contemporary Subject. Television & New Media, 20(4): 336–349.

Crawford, K. 2021. Atlas of AI: Power, Politics, and the Planetary Costs of Artificial Intelligence. Yale University Press.

Deng, Y.; Zhang, W.; Pan, S. J.; and Bing, L. 2024. Multilingual Jailbreak Challenges in Large Language Models. ArXiv:2310.06474.

Ganguli, D.; Lovitt, L.; Kernion, J.; et al. 2022. Red Teaming Language Models to Reduce Harms: Methods, Scaling Behaviors, and Lessons Learned. ArXiv:2209.07858.

Google Cloud. 2024. Google Career Launchpad Program. https://cloud.google.com/edu/faculty/career-launchpad. Accessed: 2025-12-01.

Gray, M. L.; and Suri, S. 2019. Ghost Work: How to Stop Silicon Valleyfrom Building a New Global Underclass. Houghton Miflin Harcourt.

Hutchinson, B.; Rostamzadeh, N.; Greer, C.; Heller, K.; and Prabhakaran, V. 2022. Evaluation Gaps in Machine Learning Practice. In Proceedings of the 2022 ACM Conference on Fairness, Accountability, and Transparency, 1859–1876.

International Network of AI Safety Institutes. 2025a. International Joint Testing Exercise: Agentic Testing. https://www.aisi.gov.uk/blog/international-joint-testingexercise-agentic-testing. Accessed: 2025-12-01.

International Network of AI Safety Institutes. 2025b. International Joint Testing Exercise: Agentic Testing. Advancing Methodologies for Agentic Evaluations Across Domains. https://www.peren.gouv.fr/rapports/2025- July\_Advancing-Methodologies-for-Agentic-Evaluations-Across-Domains\_Report.pdf. Accessed: 2026-10-04.

International Network of AI Safety Institutes. 2025c. Joint Testing Exercise: Improving Methodologies for AI Model Evaluations Across Global Languages. https: //www.sgdsn.gouv.fr/files/files/Publications/International-Network-of-AI-Safety-Institutes-Joint-Testing-Exercise-Improving-Methodologies-for-AI-Model-Evaluations-Across-Global-Languages.pdf. Accessed: 2026-10-04.

International Network of AI Safety Institutes. 2025d. Multilingual Joint Testing Exercise: Improving Methodologies for LLM Evaluations Across Global Languages. https://www.peren.gouv.fr/rapports/2025-June\_Improving-Methodologies-for-LLM-Evaluations-Across-Global-

Languages-Evaluation-Report-1.pdf. Accessed: 2025-12- 01.

Joshi, P.; Santy, S.; Budhiraja, A.; Bali, K.; and Choudhury, M. 2020. The State and Fate of Linguistic Diversity and Inclusion in the NLP World. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, 6282–6293.

Kwet, M. 2019. Digital Colonialism: US Empire and the New Imperialism in the Global South. Race & Class, 60(4): 3–26.

Lampland, M.; and Star, S. L., eds. 2009. Standards and Their Stories: How Quantifying, Classifying, and Formalizing Practices Shape Everyday Life. Ithaca, NY: Cornell University Press.

Lehdonvirta, V.; Wu, B.; and Hawkins, Z. 2024. Compute North vs. Compute South: The Uneven Possibilities of Compute-based AI Governance Around the Globe. In Proceedings of the AAAI/ACM Conference on AI, Ethics, and Society, volume 7, 828–838.

Liang, P.; Bommasani, R.; Lee, T.; et al. 2023. Holistic Evaluation of Language Models. Transactions on Machine Learning Research.

Longpre, S.; Kapoor, S.; Klyman, K.; Ramaswami, A.; Bommasani, R.; et al. 2024. A Safe Harbor for AI Evaluation and Red Teaming. ArXiv:2403.04893.

Luitse, D. 2024. Platform Power in AI: The Evolution of Cloud Infrastructures in the Political Economy of Artificial Intelligence. Internet Policy Review, 13(2).

Mabokela, K. R.; Primus, M.; and Celik, T. 2025. Advancing Sentiment Analysis for Low-Resourced African Languages Using Pre-Trained Language Models. PLOS ONE, 20(6): e0325102.

McDonald, N.; Schoenebeck, S.; and Forte, A. 2019. Reliability and Inter-rater Reliability in Qualitative Research: Norms and Guidelines for CSCW and HCI Practice. Proceedings of the ACM on Human-Computer Interaction, 3(CSCW): 72:1–72:23.

Microsoft Azure. 2024. Azure for Students. https://azure. microsoft.com/en-us/free/students. Accessed: 2025-12-01.

Mohamed, S.; Png, M.-T.; and Isaac, W. 2020. Decolonial AI: Decolonial Theory as Sociotechnical Foresight in Artificial Intelligence. Philosophy & Technology, 33(4): 659–684.

Mökander, J.; Schuett, J.; Kirk, H. R.; and Floridi, L. 2024. Auditing Large Language Models: A Three-Layered Approach. AI and Ethics, 4.

Mutisya, F.; Gitau, S.; Ongoma, N.; Mbae, K.; and Wamicha, E. 2025. Rethinking Evidence Hierarchies in Medical Language Benchmarks: A Critical Evaluation of HealthBench. ArXiv:2508.00081.

National Institute of Standards and Technology. 2023. AI Risk Management Framework. https://www.nist.gov/itl/airisk-management-framework. Accessed: 2025-12-01.

Nekoto, W.; Marivate, V.; Matsila, T.; et al. 2020. Participatory Research for Low-resourced Machine Translation: A Case Study in African Languages. In Findings of the Association for Computational Linguistics: EMNLP 2020, 2144–2160.

Nimo, C.; Liu, S.; Essa, I.; and Best, M. L. 2025. Africa Health Check: Probing Cultural Bias in Medical LLMs. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, 32219–32232.

Ojewale, V.; Steed, R.; Vecchione, B.; Birhane, A.; and Raji, I. D. 2025. Towards AI Accountability Infrastructure: Gaps and Opportunities in AI Audit Tooling. In Proceedings of the 2025 CHI Conference on Human Factors in Computing Systems.

Ojo, J.; Ogundepo, O.; Oladipo, A.; Ogueji, K.; Lin, J.; Stenetorp, P.; and Adelani, D. I. 2025. AfroBench: How Good are Large Language Models on African Languages? ArXiv:2311.07978.

Okolo, C. T. 2023. Addressing Global Inequity in AI Development. In Handbook of Critical Studies of Artificial Intelligence. Edward Elgar Publishing.

Onuoha, C. P.-M.; Sunu, B. E.; and Sikiru, R. 2026. Language-Specific Gaps in AI Safety Training Datasets. ArXiv:2608.13695.

Peppin, A.; Kreutzer, J.; Sebag, A. S.; Marchisio, K.; Ermis, B.; Dang, J.; Cahyawijaya, S.; Singh, S.; Goldfarb-Tarrant,

S.; Aryabumi, V.; Ko, W.-Y.; Üstün, A.; Gallé, M.; Fadaee, M.; and Hooker, S. 2025. The Multilingual Divide and Its Impact on Global AI Safety. ArXiv:2505.21344.

Png, M.-T. 2022. At the Tensions of South and North: Critical Roles of Global South Stakeholders in AI Governance. In Proceedings of the 2022 ACM Conference on Fairness, Accountability, and Transparency, 1434–1445.

Raji, I. D.; Bender, E. M.; Paullada, A.; Denton, E.; and Hanna, A. 2021. AI and the Everything in the Whole Wide World Benchmark. ArXiv:2111.15366.

Raji, I. D.; Smart, A.; White, R. N.; Mitchell, M.; Gebru, T.; Hutchinson, B.; Smith-Loud, J.; Theron, D.; and Barnes, P. 2020. Closing the AI Accountability Gap: Defining an Endto-End Framework for Internal Algorithmic Auditing. In Proceedings of the 2020 Conference on Fairness, Accountability, and Transparency, 33–44.

Raji, I. D.; Xu, P.; Honigsberg, C.; and Ho, D. 2022. Outsider Oversight: Designing a Third Party Audit Ecosystem for AI Governance. In Proceedings of the 2022 AAAI/ACM Conference on AI, Ethics, and Society, 557–571.

Reuel, A.; Ghosh, A.; Chim, J.; et al. 2025. Who Evaluates AI’s Social Impacts? Mapping Coverage and Gaps in First and Third Party Evaluations. ArXiv:2511.05613.

Sambasivan, N.; Arnesen, E.; Hutchinson, B.; Doshi, T.; and Prabhakaran, V. 2021. Re-imagining Algorithmic Fairness in India and Beyond. In Proceedings ofthe 2021 ACM Conference on Fairness, Accountability, and Transparency, 315– 328.

Sangwa, S.; Nsabiyumva, S.; Ngobi, D.; and Mutabazi, P. 2025. Ethical AI Integration in African Higher Education: Enhancing Research Supervision, Grant Discovery, and Proposal Writing Without Compromising Academic Integrity. SSRN 5331595.

Sastry, G.; Heim, L.; Belfield, H.; Anderljung, M.; et al. 2024. Computing Power and the Governance of Artificial Intelligence. ArXiv:2402.08797.

Seah, E. W.; Zheng, Y.; Nikshith, N.; et al. 2026. Improving Methodologies for Agentic Evaluations Across Domains: Leakage of Sensitive Information, Fraud and Cybersecurity Threats. ArXiv:2601.15679.

Sendra-Balcells, C.; Campello, V. M.; Bijnens, B.; Piella, G.; Gratacós, E.; Crispi, F.; and Lekadir, K. 2023. Generalisability of Fetal Ultrasound Deep Learning Models to Low-Resource Imaging Settings in Five African Countries. Scientific Reports, 13.

Shevlane, T. 2022. Structured Access: An Emerging Paradigm for Safe AI Deployment. ArXiv:2201.05159.

Shevlane, T.; Farquhar, S.; Garfinkel, B.; et al. 2023. Model Evaluation for Extreme Risks. ArXiv:2305.15324.

Solatorio, A. V.; Vicente, G. S.; Krambeck, H.; and Dupriez, O. 2024. Double Jeopardy and Climate Impact in the Use of Large Language Models: Socio-economic Disparities and Reduced Utility for Non-English Speakers. ArXiv:2410.10665.

Star, S. L. 1999. The Ethnography of Infrastructure. American Behavioral Scientist, 43(3): 377–391.

Stosz, C.; Elmgren, K.; Foster, C.; et al. 2025. AEF-1: Minimum Operating Conditions for Independent Third Party AI Evaluations. Technical report, AI Evaluator Forum. Version 1, updated 4 December 2025.

Tonja, A. L.; Dossou, B. F. P.; Ojo, J.; et al. 2024. InkubaLM: A Small Language Model for Low-Resource African Languages. ArXiv:2408.17024.

UK AI Security Institute. 2026. International Consensus and Open Questions in AI Evaluations. https://www.aisi.gov.uk/blog/international-ai-networkconsensus-and-open-questions. Accessed: 2026-10-04.

United Nations Development Programme. 2024. Only Five Percent of Africa’s AI Talent Has the Compute Power It Needs. https://www.undp.org/digital/blog/only-fivepercent-africas-ai-talent-has-compute-power-it-needs. Accessed: 2025-12-01.

U.S. AI Safety Institute; UK AI Safety Institute; and Singapore AI Safety Institute. 2024. Improving International Testing of Foundation Models: A Pilot Testing Exercise from the International Network of AI Safety Institutes. Technical report, International Network of AI Safety Institutes. https://www.nist.gov/system/files/ documents/2024/11/21/Improving\%20International\ %20Testing\%20of\%20Foundation\%20Models-\%20\%20\%20A\%20Pilot\%20Testing\%20Exercise\ %20from\%20the\%20International\%20Network\%20of\ %20AI\%20Safety\%20Institutes.pdf.

Vij, A.; Chua, B.; Ramiah, D.; et al. 2026. Improving Methodologies for LLM Evaluations Across Global Languages. ArXiv:2601.15706.

Weidinger, L.; Mellor, J.; Rauh, M.; et al. 2021. Ethical and Social Risks of Harm from Language Models. ArXiv:2112.04359.

Weidinger, L.; Rauh, M.; Marchal, N.; et al. 2023. Sociotechnical Safety Evaluation of Generative AI Systems. ArXiv:2310.11986.

Whittaker, M. 2021. The Steep Cost of Capture. Interactions, 28(6): 50–55.

Widder, D. G.; Whittaker, M.; and West, S. M. 2024. Why “Open” AI Systems Are Actually Closed, and Why This Matters. Nature, 635: 827–833.

Yong, Z.-X.; Menghini, C.; and Bach, S. H. 2024. Low-Resource Languages Jailbreak GPT-4. ArXiv:2310.02446.

## Appendices: Data and Coding

These appendices document the two datasets used in the paper. Appendix A explains what we looked for under each PAAI question. Appendix B explains how we chose the cases. Appendix C gives the evidence behind each entry for the twelve African projects. Appendix D summarises the access programmes those projects relied on. Appendix E records each institute’s role in the joint testing exercises. The full dataset, a spreadsheet with six sheets, will be released publicly if the paper is accepted. These appendices contain the same evidence and rules, so the paper can be assessed without it. All sources are public documents, and no personal data are included.

## A What We Looked For

Design authority. We asked who defined the harms, measures and scoring; who wrote the annotation guides; who chose what to test; and who led the analysis and wrote up the findings. Framework authority means setting these rules. Content authority means contributing tasks, data or languages within rules set by others. We drew on author afiliations, method and contribution sections, and acknowledgements.

Infrastructure authority. We asked who controlled the computing power and model access; whether the team used models directly or through someone else; and whether a provider or lead institution could withdraw access. We drew on acknowledgements of computing support, provider terms, and descriptions of where and by whom models were run.

Access. We asked whether the team ran models itself or worked with results produced by others; who was eligible and where; how long access lasted and whether it could be renewed; how many separate sources the team relied on; and whether it depended on a go-between. We drew on provider programme pages and acknowledgements.

Adequacy. For technical adequacy, we asked whether the computing power, data and model access matched what the evaluation claimed, and whether the team could rerun and refine its tests. For contextual validity, we asked whether the tasks, languages and measures reflected the settings the evaluation claimed to describe, whether tasks were built locally or translated, and how reliable automated scoring was in each language. We drew on descriptions of scope, benchmark documentation, stated limitations and reported results.

Independence. We asked whether the team could choose what to evaluate, design the methods, repeat the evaluation, publish and keep going over time without needing approval or goodwill from providers or lead institutions. Being credited as an author or having a named section was recorded separately and did not by itself raise the level. For the ten African projects that ran evaluations, we used four levels: none (no documented control over targets, execution or publication), minimal (contributes content but cannot run or continue the evaluation), partial (controls design and publication but depends on infrastructure that can be withdrawn) and high (controls design, execution and publication, limited only by resources). We drew on publication practices, funding arrangements, statements about provider dependence, and evidence that work was repeated or continued.

Configurations. We placed each case in one of four configurations according to who held the two sub-dimensions of Power. A team holding both design and infrastructure authority is in constrained independence. A team holding design but not infrastructure authority is in dependent innovation. A team that runs tests others designed is in execution without design, which appears only in joint testing. A team holding neither is in extractive collaboration.

## B How We Chose the Cases

Search. Academic databases (ACM Digital Library, arXiv, ACL Anthology), organisational and policy reports, and public repositories, November 2022 to December 2025. Queries combined evaluation types with geographic and infrastructural terms, e.g. [evaluation type] AND “Africa”; [country name] AND (“compute” OR “API” OR “benchmark”). The searches returned more than 200 sources. After removing duplicates and screening, we assessed 47 in detail and included 12 projects.

Inclusion. A project was included if (1) it evaluated something identifiable, such as a model’s behaviour, performance on a benchmark, or an evaluation method itself; (2) it concerned African languages, institutions, users or settings; (3) its public documents were detailed enough to answer at least two PAAI questions; and (4) it took place within the date range.

Exclusion. We excluded (1) evaluations of Western settings with no African focus, (2) proposals that did not evaluate anything identifiable, and (3) studies of under-resourced languages that were not African.

Special cases. Two projects (C.9 and C.10) assessed evaluations or published evidence without running models. InkubaLM (C.5) mainly built a model and is included because it also tested that model on six tasks. Removing C.9 and C.10 still leaves cases in all three African configurations.

Joint testing. We included all three joint testing exercises the network had published by October 2026, and recorded the role of every member institute named in each report. The pilot report names only its three leads.

## C Evidence for Each African Project

Each subsection matches a row of Table 1. The line under each heading gives the type of evaluation and the configuration. The “AEF-1” notes compare each case with the closest available standard (Stosz et al. 2025), which was written for outside evaluations of companies’ systems and fits several cases imperfectly.

## C.1 AISI Multilingual Joint Testing

## Multilingual safety · Extractive collaboration

Power (External / External). Singapore AISI led the exercise, tested two open-weight models (Mistral Large and Gemma 2 27B) on its Moonshot platform for all ten languages, developed the annotation guide and automated metrics calculator, and led the analysis. Harm categories were discussed among participating institutes, but Singapore built the tools that applied them. Kenya annotated Kiswahili responses and authored the Kiswahili analysis section. Access (Through the lead, time-limited). Kenya worked on results produced on Singapore’s platform during the exercise; no direct model access is recorded. Adequacy (Rules set externally). Rigorous methods: over 6,000 translated prompts, scored by an AI model and checked by human annotators.

Kiswahili showed up to 31 percentage points of variation between the two tested models, and models sometimes gave partial harmful instructions despite disclaimers; these emerged as findings within a list of harms defined by others. Independence (None documented). Kenya did not choose the models, define the measures or repeat the evaluation, and the records show no right to do so. Recognition: named Kiswahili section; Kenyan staf are credited co-authors of the arXiv version. AEF-1: freedom of analysis and independent access not met. Sources: International Network of AI Safety Institutes (2025d,c); Vij et al. (2026).

## C.2 Kenya AISI Agentic Testing

## Multilingual safety · Extractive collaboration

Power (Partial / External). Kenya constructed a new set of ten tasks and tools on fraud, phishing and impersonation, making it one of two non-lead institutes to contribute new tasks. It also translated the dataset into Kiswahili, annotated results and contributed to analysing the cybersecurity test records. Singapore led the data leakage and fraud part, ran the tests on Moonshot, supplied the annotation guides and led the analysis; the UK led the cybersecurity part. Access (Coordinated; renewal not established). The lead ran Kenya’s tasks on its own platform; no Kenyan model runs or computing resources are recorded. Adequacy (Limited fit to context). Kenya’s tasks brought in regional risks, but scoring and analysis followed the lead’s framework, and the automated scorers disagreed most with human reviewers on Telugu, Hindi and Kiswahili. Independence (Minimal). Kenya contributed substantive tasks and analysis, while the lead institutes retained responsibility for the overarching framework and execution. The records do not establish Kenya’s authority to initiate further runs or continue the evaluation independently. Recognition: Kenyan staf are credited co-authors of the arXiv version. AEF-1: independent access not met; the rules were set by others. Sources: International Network of AI Safety Institutes (2025a,b); Seah et al. (2026).

## C.3 Georgia Tech / Google Research Medical LLM Audit

Domain-specific risk · Extractive collaboration

Power (External / External). Georgia Tech and Google Research designed the Cultural Bias Score and Cultural Bias Attribution metrics, ran all the analysis, including analysis of the models’ internal workings, and wrote the paper. Knowledge of herbal remedies from ten African countries came from PubMed (2020–2024) and WHO guidelines. Access (Data source only). No African institution used the computing resources, models or tools. Adequacy (Rigorous, external). New measures and detailed model analysis on computing resources the paper does not describe. Independence (not assessed). No authors afiliated with African institutions are identified, so the case informs who held evaluative authority rather than the independence of an African evaluator. AEF-1: not applicable. Sources: Nimo et al. (2025).

## C.4 Sunbird AI ASR Data Scaling

Capability benchmarking · Constrained independence

Power (African-led / African-led). Sunbird AI (Uganda) designed the study, curated data from a 1,400-hour Kinyarwanda corpus and Kikuyu data, trained and evaluated models, published and released models openly. The funder provided resources without design authority. Access (Grantfunded). High-end graphics processors paid for by a Hewlett Foundation grant, used for about 20 hours in total. Adequacy (Good fit to context). 270 hours ofcurated training audio and 6,910 test samples, with analysis of errors under real conditions. Independence (High, cost-limited). No provider limits on publication; continuing depends on grant renewal rather than on a provider. AEF-1: closest to the standard; constrained by funding. Sources: Akera et al. (2025).

## C.5 Lelapa AI InkubaLM

Capability benchmarking · Constrained independence

Power (African-led / Partial). Lelapa AI (South Africa) defined the problem, curated Inkuba-Mono (2.4B tokens, five languages) and instruction data, trained the 0.4B-parameter model, tested it on six tasks and kept ownership of the model; international partners provided research support. Access (Provider credits). Eight high-end graphics processors through free Microsoft Azure credits, which Microsoft could withdraw; venture funding ofers some protection. Adequacy (Rigorous coverage). Performs comparably to larger models on the reported tasks. Independence (Partial). Published independently and kept ownership, but depends on provider computing. AEF-1: partly limited by dependence on provider computing. Sources: Tonja et al. (2024).

## C.6 Barcelona Fetal Ultrasound Generalisation

Domain-specific risk · Extractive collaboration

Power (External / External). The University ofBarcelona and BCNatal trained an image model on Spanish data, checked it on Danish data, developed the method for adapting it to new hospitals and tested it on African data; all computation and analysis took place in Spain. Access (Data source only). Hospitals in Egypt, Uganda, Ghana, Malawi and Algeria collected ultrasound data (120 patients in total) and did not take part in model design, training or analysis. Adequacy (Rigorous, external). Trained on a single standard graphics card; after adaptation using a few local patients, the model was highly accurate on African data (AUC above 0.98). Independence (not assessed). Development and analysis were located in Spain. Ten co-authors from African institutions are credited with image collection; the paper documents no design, training or analysis role at the African sites. Recognition: co-authorship without evaluative authority. AEF-1: not applicable to African participants. Sources: Sendra-Balcells et al. (2023).

## C.7 Wits/UJ SAfriSenti

Capability benchmarking · Constrained independence

Power (African-led / African-led). South African universities designed the study, curated over 115,000 tweets in five Southern African languages, designed a partly automated labelling process with native-speaker volunteers, adapted existing multilingual models and released the collection. Access (Own hardware + cloud). University computers plus paid cloud services (Google Colab Pro+ and AWS); no specific funding. Adequacy (Suficient for scope). Labelling method can be repeated; scale limited by reliance on volunteers. Independence (High, volunteer-limited). No funder restrictions; collection publicly released. AEF-1: closest to the standard; limited by resources rather than by providers. Sources: Mabokela, Primus, and Celik (2025).

## C.8 Masakhane Uhura

Multilingual safety · Dependent innovation

Power (African-led / External). The Masakhane community designed the benchmark and human-translated ARC-Easy and TruthfulQA into six African languages. Models were tested through the companies’ online services: GPT-4o and o1-preview (OpenAI) and Claude (Anthropic), with free credits from OpenAI; several co-authors worked for OpenAI. This is the clearest case in the sample of a team controlling the design but not the infrastructure. Access (Free online access, can be withdrawn). The team saw only the models’ inputs and outputs, and the providers could end access. Adequacy (Translated benchmarks). Rigorous, but built on English originals. Independence (Partial). Published independently, but seeing only inputs and outputs limits how hard the team could probe the models, and losing access would stop the work. AEF-1: sustained access not established. Sources: Bayes et al. (2024).

## C.9 Qhala HealthBench Critique<sup>†</sup>

Domain-specific risk · Dependent innovation (provisional)

Power (African-led / None). Kenyan authors identified flaws in OpenAI’s HealthBench (reliance on expert opinion over stronger forms of evidence, neglect of tropical diseases, and Western clinical assumptions) and proposed an alternative based on clinical guidelines. Access (Public documents). Published HealthBench documentation and Kenyan, South African and WHO guidelines; no model or compute access. Adequacy (Proposal only). No evaluation was run. Independence (of critique only). Published without funder constraints; the documents do not show whether the authors had or sought resources to implement the alternative. AEF-1: not assessable. Sources: Mutisya et al. (2025).

## C.10 Rwanda Higher-Education Review<sup>†</sup>

Domain-specific risk · Dependent innovation (provisional)

Power (African-led / None). Led from the African Leadership University (Rwanda) with a US co-author institution; a systematic review of published studies on AI in African universities, ending in a plan for how universities should govern AI. Access (Literature only). Adequacy (Policy-relevant). Quantified efects are those reported in the reviewed studies; no technical evaluation. Independence (of analysis only). The authors controlled their analysis; implementing the plan would need resources the documents do not describe. AEF-1: not assessable. Sources: Sangwa et al. (2025).

## C.11 Masakhane AfroBench

Capability benchmarking · Dependent innovation

Power (African-led / External). Masakhane aggregated 22 datasets across 64 African languages and 15 tasks and designed how they would be tested; international partners provided infrastructure for running the tests. Computing came from Google Cloud and OpenAI credits. Access (Multiple credits). Time-limited credits; a smaller AfroBench-Lite was created so that others could run it more cheaply. Adequacy (Wide, English-adapted). Wider coverage than any earlier benchmark for African languages; many tasks adapted from English sources. Independence (Partial). Published independently with a public leaderboard; continuing depends on credits. AEF-1: sustained access not established. Sources: Ojo et al. (2025).

## C.12 IrokoBench

Capability benchmarking · Dependent innovation

Power (African-led / External). Masakhane-led human translation of three tasks (AfriXNLI, AfriMGSM, AfriMMLU) into 17 African languages, with many African partner institutions. Access (Five short-term sources). Lacuna Fund, Oracle compute credits, OpenAI Researcher Access Program, Google’s Gemma academic programme and a Cohere For AI grant; the most scattered set of sources in the sample. Adequacy (Human translation). Highest translation quality among the benchmarking cases. Independence (Partial). Published independently; no long-term computing arrangement. AEF-1: sustained access not established. Sources: Adelani et al. (2025).

## D Provider Access Programmes

Table 3 summarises the published terms of the programmes the African projects used. Private contract terms were not available. None of the programmes ofers long-term access that renews automatically. All are tied to a project or a fixed amount of credit, and researchers must apply again. None of the commercial programmes publishes its rules on what researchers may publish.

## E Joint Testing Records

What each record contains. There is one record for each member institute in each exercise (29 records), combining the separate parts of an exercise. Design is one of: Lead (led the exercise or one of its parts, or wrote its core method, annotation guides or measures); New parts (contributed new tasks, tools, benchmarks, measures, scoring prompts or its own test variations); Language/analysis (translation, checking, annotation or analysis only); None recorded. Running models is one of: Main tests (ran the main tests); Own tests (ran its own tests, repeats or variations); None recorded. Access is Direct (ran models) or Via lead (worked with results produced by another institute). Role follows from these: Lead if the institute led; otherwise Executing if it ran models; otherwise Contributing if any role is recorded; otherwise Undocumented. Evidence is High if the report states the role, Medium if it comes from a summary of the report, and Low if the institute is not named.

First exercise (pilot, November 2024). The report names three leads, the United States, United Kingdom and Singapore, which tested Llama-3.1 405B on three benchmarks; the United States ran the 14-language MMMLU evaluation, which found English substantially outperforming Yoruba and Swahili. Other members contributed “input” but are not named, so their roles are recorded as undocumented (U.S. AI Safety Institute, UK AI Safety Institute, and Singapore AI Safety Institute 2024).

Second and third exercises. Tables 5 and 6 give the records. Sources: International Network of AI Safety Institutes (2025c,d); Vij et al. (2026) for the second exercise; International Network of AI Safety Institutes (2025a,b); Seah et al. (2026) for the third.

Totals and scoring quality. Table 4 gives each institute’s totals, which are the values plotted in Figure 3b. In the second exercise, the automated scorer disagreed with human reviewers more than 5% of the time in six of ten languages, most often for Japanese (18.9%) and Telugu (15.0%) (International Network of AI Safety Institutes 2025d). In the third, disagreement between automated scorers and humans ranged from 15% to 41%, depending on the scoring model, and was highest for Telugu, Hindi and Kiswahili in most risk categories (International Network of AI Safety Institutes 2025b).

Limits of these records. Reports may describe the leads work more fully than the participants’. “None recorded” means the report does not mention a role, not that there was none. We took roles from the published reports and summaries. Author counts are estimated from the order of names on the arXiv versions, which list the lead institutes first and then other members alphabetically by country.

<table><tr><td>Provider</td><td>Programme</td><td>Duration and renewal</td><td>Publication terms</td><td>Projects</td></tr><tr><td>OpenAI</td><td>Researcher Access Program</td><td>Project-based; duration not disclosed; reapplication at provider discretion</td><td>Not disclosed; usage policy applies</td><td>C.8, C.11, C.12</td></tr><tr><td>Anthropic</td><td>External Researcher Access Program</td><td>Project-based; reapplication required</td><td>Not disclosed; usage policy applies</td><td>C.8</td></tr><tr><td>Google</td><td>Cloud research credits; Gemma academic programme; Lacuna Fund</td><td>Credits expire per cycle; competitive reapplication</td><td>Not disclosed</td><td>C.11, C.12</td></tr><tr><td>Microsoft</td><td>co-funder Azure credits (direct provision)</td><td>Duration of credit agreement; reapplication required</td><td>Not disclosed</td><td>C.5</td></tr><tr><td>AWS</td><td>Research credits</td><td>12-month cycles; competitive</td><td>Not disclosed</td><td>None (C.7 used</td></tr><tr><td>Oracle</td><td>Oracle for Research credits</td><td>reapplication Project-based; application-based</td><td>Not disclosed</td><td>paid EC2) C.12</td></tr><tr><td>Cohere For AI</td><td>Research grant</td><td>Grant cycle; competitive</td><td>Not disclosed</td><td>C.12</td></tr><tr><td>Lacuna Fund</td><td>Dataset grants (Rockefeller Foundation,</td><td>Grant cycle; competitive</td><td>Open data and public results required</td><td>C.12</td></tr><tr><td>Hewlett</td><td>Google.org, IDRC) Programme grant</td><td>Grant cycle; multi-year possible</td><td>Generally permissive for</td><td>C.4</td></tr><tr><td>Foundation Cassava Technologies</td><td>No documented AI research access</td><td></td><td>academic publication</td><td>None</td></tr></table>

Table 3: Infrastructure access programmes used by the African projects (public documentation, accessed December 2025). Numbers refer to Appendix C.

<table><tr><td>Institute</td><td>Roles</td><td>Led</td><td>Ran models</td><td>Design</td></tr><tr><td>Singapore</td><td>3</td><td>3</td><td>3</td><td>3</td></tr><tr><td>United Kingdom</td><td>3</td><td>3</td><td>3</td><td>3</td></tr><tr><td>United States</td><td>2</td><td>1</td><td>1</td><td>1</td></tr><tr><td>Japan</td><td>2</td><td>1</td><td>1</td><td>1</td></tr><tr><td>Australia</td><td>2</td><td>0</td><td>2</td><td>2</td></tr><tr><td>France</td><td>2</td><td>0</td><td>1</td><td>2</td></tr><tr><td>Korea</td><td>2</td><td>0</td><td>1</td><td>1</td></tr><tr><td>EU AI Office</td><td>2</td><td>0</td><td>1</td><td>0</td></tr><tr><td>Canada</td><td>2</td><td>0</td><td>0</td><td>0</td></tr><tr><td>Kenya</td><td>2</td><td>0</td><td>0</td><td>1</td></tr></table>

Roles: exercises with a recorded role. Design: exercises in which the institute led or designed new parts.

Table 4: Institute totals across the three exercises.

<table><tr><td>Institute</td><td>Documented role</td><td>Design</td><td>Ran models</td><td>Access</td><td>Role</td><td>Credit</td><td>Evidence</td></tr><tr><td>Singapore</td><td>Exercise lead (multilingual); ran all tests; wrote annotation guide and metrics calculator; led analysis</td><td>Lead</td><td>Main tests</td><td>Direct</td><td>Lead</td><td>12 authors; First-listed author block</td><td>High</td></tr><tr><td>Japan</td><td>Co-lead (multilingual); translated dataset into ten languages; ran independent tests; Japanese annotation and language analysis;</td><td>Lead</td><td>Own tests</td><td>Direct</td><td>Lead</td><td>5 authors; Japanese language analysis</td><td>High</td></tr><tr><td>UK</td><td>translated cyber benchmarks Lead of cybersecurity strand; ran cyber evaluations</td><td>Lead</td><td>Main tests</td><td>Direct</td><td>Lead</td><td>4 authors</td><td>High</td></tr><tr><td>Australia</td><td>Ran independent tests in Mandarin Chinese; input on cyber analysis</td><td>New parts</td><td>Own tests</td><td>Direct</td><td>Executing</td><td>7 authors</td><td>High</td></tr><tr><td>France</td><td>Contributed internal cyber benchmark; French annotation and language analysis</td><td>New parts</td><td>None recorded</td><td>Via lead</td><td>Contributing</td><td>3 authors; French language analysis</td><td>High</td></tr><tr><td>Korea</td><td>Korean annotation and language analysis; Korean translation of cyber benchmarks</td><td>Language/ analysis</td><td>None recorded</td><td>Via lead</td><td>Contributing</td><td>7 authors; Korean language analysis</td><td>High</td></tr><tr><td>Canada</td><td>Translated/validated Cantonese and Farsi; annotation; two language language analyses</td><td>Language/ analysis</td><td>None recorded</td><td>Via lead</td><td>Contributing</td><td>4 authors; Cantonese and Farsi language</td><td>Medium</td></tr><tr><td>Kenya</td><td>Kiswahili annotation and language analysis</td><td>Language/ analysis</td><td>None recorded</td><td>Via lead</td><td>Contributing</td><td>analyses 4 authors; Kiswahili</td><td>High</td></tr><tr><td>EU AI</td><td>Listed as joint participant; input on</td><td>Language/ analysis</td><td>None</td><td>Not</td><td>Contributing</td><td>language analysis None</td><td>Low</td></tr><tr><td>Office US</td><td>discussions and cyber analysis Input on cyber analysis</td><td>Language/ analysis</td><td>recorded None recorded</td><td>recorded Not recorded</td><td>Contributing None</td><td></td><td>Low</td></tr><tr><td>Singapore</td><td>Lead of leakage/fraud strand; ran tests; annotation guides; led</td><td>Lead</td><td>Main tests</td><td>Direct</td><td>Lead</td><td>12 authors; First-listed author block</td><td>High</td></tr><tr><td>UK</td><td>analysis Lead of cybersecurity strand; ran evaluations; statistical framework</td><td>Lead</td><td>Main tests</td><td>Direct</td><td>Lead</td><td>11 authors; Second-listed author block</td><td>High</td></tr><tr><td>France</td><td>New tasks and tools; ran variant tests; alternate judge prompt; extra metrics; French translation and</td><td>New parts</td><td>Own tests</td><td>Direct</td><td>Executing</td><td>6 authors; French analysis</td><td>High</td></tr><tr><td>Korea</td><td>annotation Validated annotations; ran temperature/model-size variations; translation, annotation; analysis of</td><td>New parts</td><td>Own tests</td><td>Direct</td><td>Executing</td><td>7 authors; Korean High analysis</td><td></td></tr><tr><td>Australia</td><td>cyber test records Validated annotations; added metrics; ran cyber evaluations; translation, annotation; analysis of test records</td><td>New parts</td><td>Own tests</td><td>Direct</td><td>Executing</td><td>14 authors</td><td>High</td></tr><tr><td>EU AI Office</td><td>Ran cyber evaluations; analysis of test records</td><td>Language/ analysis</td><td>Own tests</td><td>Direct</td><td>Executing</td><td>4 authors</td><td>High</td></tr><tr><td>Japan</td><td>Validated annotations; Japanese translation, annotation and language analysis</td><td>Language/ analysis</td><td>None recorded</td><td>Via lead</td><td>Contributing 3 authors;</td><td>Japanese analysis</td><td>High</td></tr><tr><td>Canada</td><td>Farsi translation, annotation and language analysis; analysis of cyber analysis test records</td><td>Language/</td><td>None recorded</td><td>Via lead</td><td></td><td>Contributing 9 authors; Farsi analysis</td><td>High</td></tr><tr><td>Kenya</td><td>New fraud/phishing tasks and tools New parts (10 tasks); Kiswahili translation, annotation and language analysis; analysis of cyber test records</td><td></td><td>None recorded</td><td>Via lead</td><td>Contributing</td><td>4 authors; Kiswahili analysis</td><td>High</td></tr></table>

Table 5: Second exercise (multilingual and cybersecurity, February 2025). Column meanings are given in Appendix E. Author counts are estimated from the order of names on the arXiv version.

Table 6: Third exercise (agentic, July 2025). Column meanings are given in Appendix E. Author counts are estimated from the order of names on the arXiv version.