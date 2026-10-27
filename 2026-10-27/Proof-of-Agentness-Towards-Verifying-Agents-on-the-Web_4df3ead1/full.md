# Proof-of-Agentness: Towards Verifying Agents on the Web

Zuyao Xu

xuzuyao@mail.nankai.edu.cn

Nankai University

Tianjin, China

Xiang Li<sup>∗</sup>   
lixiang@nankai.edu.cn   
Nankai University   
Tianjin, China

## Ab<sub>s</sub>t<sub>rac</sub>t

CAPTCHAs and bot detection have helped protect Web services from automated abuse for decades by distinguishing humans from bots. However, the emergence of LLM-powered agents introduces a new setting: Web services increasingly accommodate agents as legitimate users. Without a way to distinguish agents from other automated clients such as simple bots and scripts, these services risk large-scale automated junk submissions that waste service resources and consume users’ attention or tokens. In this paper, we investigate a novel question for today’s Web: “How can a Web service obtain evidence that a client is an agent?” We introduce the concept of “Proof-of-Agentness” (PoA), a mechanism for distinguishing agents from other automated clients, motivated by the parallel between human attention and an agent’s inference resources. We outline two approaches: active verification through challenge–response and passive analysis of service-visible signals. Eight language models achieve an overall accuracy of 96.5% on our challenges, with an average overhead of 1,647 tokens. For passive analysis, we conduct an IRB-approved user study comparing the Web interactions of twelve human participants and representative Bot and Agent implementations, achieving an average classification accuracy of 95.8% across four classifiers, motivating further study.

## CCS Conce<sub>p</sub>ts

• Security and privacy → Access control; Authentication.

## Ke<sub>y</sub>words

Proof-of-Agentness, Web agents, admission control, resource abuse, behavioral verification

## 1 I<sub>n</sub>t<sub>ro</sub>d<sub>uc</sub>ti<sub>on</sub>

CAPTCHAs and other bot detection mechanisms have helped protect human-facing Web services from automated abuse at scale by requiring evidence of human participation before granting access [50, 58]. Each operation performed by a human consumes time and attention, while scripts can cheaply repeat requests and submissions, overwhelming services. These mechanisms aim to make access depend on evidence of human participation that bots cannot cheaply reproduce.

Meanwhile, Web services are increasingly being designed for agents as well as humans. Cloudflare and Vercel serve agent-friendly Markdown pages [19, 41]. Many online services provide agent access through the Model Context Protocol (MCP), covering search (Brave, Exa, and Tavily) [7, 25, 57]; data and knowledge queries (Hugging Face, Kaggle, Wikidata) [32, 36, 64]; payments (Stripe and PayPal) [51, 56]; shopping (Shopify) [55]; and development and workplace collaboration (GitHub, Notion, and Atlassian) [6, 29, 48].

However, these interfaces can also be used by ordinary programs. When agents rely on large language models (LLMs) to interpret tasks and choose actions, they consume inference compute and tokens [53]. A traditional automated program can avoid such resource costs and repeatedly submit requests directly, occupying serverside queues and consuming storage and computing resources. Junk submissions produced in this way can also require users or agents to spend attention or tokens processing their contents.

The resource analogy between human attention and model inference motivates our central research question: Can a Web service obtain evidence that a client is an agent and protect service capacity for useful agent work, just as CAPTCHAs and bot detection protect human-facing services for genuine human interaction?

Agent verification is already entering Web infrastructure. Cloudflare’s Signed agents and Verified bots programs recognize agent trafic, and Web Bot Auth checks signed HTTP requests against registered public keys [15, 16, 38]. However, these approaches authenticate identities asserted by agent operators or infrastructure providers without establishing actual agent participation. Recent studies also explore browser and behavioral fingerprints to distinguish Web agents from human users and other automated clients [26, 61]. These eforts advance agent recognition, but the positioning of agent verification on the Web remains unclear, and the concept itself is not yet fully developed.

Our Study In this paper, we introduce Proof-of-Agentness (PoA), which asks for evidence of language model participation in an interaction. We outline two design approaches: active verification that issues challenges to clients and checks their responses, and passive analysis using Web trafic and interaction characteristics. We then introduce a unified AgentScore that combines evidence for admission decisions. Such mechanisms could empower Web services to restrict access to agents acting on users’ behalf and regulate automated trafic more selectively, limiting large-scale scripted abuse of service resources.

We evaluate active verification with eight language models and two rule-based scripts on 100 challenges, achieving an overall accuracy of 96.5% with an average overhead of 1,647 tokens. For passive analysis, we conduct an IRB-approved study of 720 Web visits across three shopping tasks, involving twelve human participants alongside six Bot and three Agent configurations. Using 40 service-visible features, we achieve an average classification accuracy of 95.8% across four classifiers on held-out tasks. Figure 1 connects the CAPTCHA analogy to agent verification and scripted abuse at a shared Web service.

Contributions. We make three contributions:

• Concept positioning. We introduce Proof-of-Agentness (PoA) as a Web verification objective centered on language model participation, analyzing its potential benefits and trade-ofs for Web services.

![](images/8cd6b2fcfe5ff89029d93fa0c1b0c8fbdf6d24d47db5227d466f98269098307a.jpg)  
Figure 1: Human users, AI agents, and scripted abuse on the Web. (a) CAPTCHA seeks human-participation evidence to protect human-facing services from automated abuse. (b) Agents and LLM workflows spend tokens interpreting tasks and feedback; scri ts can re la junk without such costs while consumin shared service resources. Motivated b reservin ca acit for useful agent work, Proof-of-Agentness asks for evidence of language model participation.

• Verification design. We outline two approaches to verifying agent participation: active verification through a challenge–response mechanism and passive analysis of Web trafic and interaction characteristics.

• Empirical study. We evaluate active verification with eight language models, examining challenge performance and overhead. To explore passive recognition, we collect a Web trafic and interaction dataset from twelve human participants and representative Bot and Agent implementations.

## 2 B<sub>ac</sub>k<sub>groun</sub>d <sub>an</sub>d R<sub>e</sub>l<sub>a</sub>t<sub>e</sub>d W<sub>or</sub>k

## 2<sub>.</sub>1 B<sub>o</sub>t D<sub>e</sub>t<sub>ec</sub>ti<sub>on an</sub>d Ad<sub>m</sub>i<sub>ss</sub>i<sub>on</sub> C<sub>on</sub>t<sub>ro</sub>l

Bot detection is a mechanism used by Web services to protect against large-scale automated abuse and attacks. Such systems examine request headers, browser characteristics, and behavior within sessions for signs of automation [12, 50]. Cloudflare, for example, combines known fingerprints, JavaScript checks, and machinelearning models to assign a bot score indicating how likely a request is to be automated [13]. A service may also use an explicit challenge to block automated trafic. CAPTCHAs follow this approach with tasks designed to be easy for humans and dificult for contemporary programs [58]. The service checks the submitted response for admission control, admitting clients judged to be human.

Admission control can also make access conditional on computational efort. Proof-of-work (PoW) raises the cost of large-scale abuse, such as spam, by requiring clients to solve computational problems that are costly to solve but inexpensive to verify [23]. Client puzzles apply this asymmetry to defend against connectiondepletion attacks: the server issues a puzzle and allocates connection resources only after verifying a valid solution [35]. Requiring a solution before allocating resources raises the cost of repeated requests, helping protect service capacity from large-scale abuse.

## 2.2 LLM-Powered Web A<sub>g</sub>ents

LLM-powered Web agents are becoming increasingly common on the Internet [17]. In an LLM-powered Web agent, a language model interprets user instructions and selects actions based on observations from the environment [30, 31]. Chain-of-thought (CoT) prompting guides models to reason step by step [62], while Re-Act combines reasoning, actions, and observations [67]. WebAgent brings reasoning to browser interaction by combining planning, HTML understanding, and program generation [30]. Moreover, multimodal agents such as WebVoyager and SeeAct incorporate visual observations to interpret webpages and select actions [31, 68].

A Web agent can access Web services through tools such as web\_search and web\_fetch [4, 5], MCP interfaces [44], browser extensions [3], or browser-control and computer-use frameworks [2, 8]. For visual interaction, multimodal agents can observe screenshots of the browser or desktop environment and select actions based on visual cues [2, 31]. To evaluate agents’ performance on Web tasks, researchers have developed a variety of Web-agent benchmarks. These include Mind2Web, WebArena, and WorkArena for Web tasks [21, 22, 69], VisualWebArena for visually grounded Web tasks [37], and OSWorld for broader computer-use tasks involving Web applications [65], showcasing the capabilities of LLMpowered agents across various domains and tasks.

## 2.3 A<sub>g</sub>ent Identification and Authentication

Web platforms are introducing mechanisms to identify agents and authenticate their claimed identities. Cloudflare, for example, recog nizes registered bots and agents through its Verified bots program, while Signed agents supports cryptographically authenticated agent trafic [15, 38]. Web Bot Auth checks HTTP request signatures against registered public keys, allowing services to authenticate the sender associated with a claimed agent identity [16].

Academic research also explores identity and authentication mechanisms for AI agents. For example, Garzon et al. propose a framework for agents to authenticate one another across organizational boundaries [28]. AgentDID uses decentralized identifiers and verifiable credentials to authenticate agent identities [66]. Beyond identity authentication, several studies explore the use of fingerprints to identify AI browsing agents [26, 40, 61], reflecting research interest in this emerging area. However, identifying an agent or authenticating its identity does not by itself establish whether a language model participates in the current interaction. On the human-facing Web, account authentication verifies control of an account, but valid credentials can also be used by automated programs. CAPTCHAs and other bot detection mechanisms therefore seek evidence of human participation to limit automated abuse [13, 58]. Similarly, an authenticated agent identity can also be used by a scripted program. This motivates a complementary mechanism for verifying model participation in agent-facing Web services.

## 3 Proof-of-A<sub>g</sub>entness: Definition and Desi<sub>g</sub>n

## 3<sub>.</sub>1 P<sub>ro</sub>bl<sub>em</sub> F<sub>ormu</sub>l<sub>a</sub>ti<sub>on</sub>

Drawing inspiration from CAPTCHAs and bot detection, we propose Proof-of-Agentness (PoA) based on the resource analogy between human attention and language model inference. Human participation consumes time and attention, while language model participation consumes inference compute and tokens. Simple scripts can repeatedly submit requests at low marginal cost while occupying server-side queues and consuming storage and computing resources. PoA is designed to extend the principle of admitting humans while keeping bots out to agent-facing Web services. The goal is to admit agents while keeping automated bots out, empowering Web services to regulate automated trafic more selectively.

## Definition: Proof-of-A<sub>g</sub>entness

Proof-of-Agentness (PoA) is a mechanism for verifying language model participation in an interaction. Participation means that model output is used to interpret information, select an action, or produce a response in that interaction.

Requiring evidence of model participation could help agentfacing services prevent automated bots from gaining access by posing as agents, limiting opportunities for large-scale automated abuse at no cost. Such a mechanism can be deployed across Web interfaces, APIs, and MCP tools, particularly for operations that change service state, allocate limited resources, or directly afect other users. Services can specify which interactions or operations require evidence of model participation, making participation an explicit part of their access policies.

## 3.2 Design Objectives

Beyond verifying agent participation, PoA should also remain practical for services and agents. This requires considering both service overhead and agent efort during ordinary interactions. We identify three practical design goals for deployment.

• Eficiency. Verification should impose limited computation, storage, and communication costs on the service, allowing it to scale with incoming trafic.

• Validity. Verification results should be bound to the client and permitted operations, with explicit expiry and reuse limits. The service should enforce these limits so that results cannot authorize access outside their intended scope.

• Usability. Verification requirements and formats should be compatible with supported agents and their access interfaces. The additional latency, inference cost, and interaction efort should remain acceptable for useful agent work.

In pursuing these objectives, PoA faces a familiar constraint: a service generally observes a client’s interactions without direct access to the computation behind them. A similar challenge is faced by CAPTCHAs and bot detection mechanisms, whose design principles and deployment experience ofer a foundation for extending participation verification to language models.

## 3<sub>.</sub>3 V<sub>er</sub>ifi<sub>ca</sub>ti<sub>on</sub> F<sub>ramewor</sub>k

Building on these principles and prior work, we organize PoA around two approaches to obtaining participation evidence: active verification and passive analysis. Figure 2 presents our framework, linking evidence to access decisions.

Active verification. The service issues a public challenge � and retains private verification state �. After receiving the client’s answer �b, it computes $a = \mathrm { V e r i f y } ( \sigma , { \widehat { a } } )$ , where $a \in \{ 0 , 1 \}$ indicates failure or success. The challenge is designed under the assumption that it is easy for language models to solve but dificult for simple rule-based programs. Under this assumption, a correct response provides evidence of agent participation, and the strength of this evidence depends on the soundness of the challenge design.

Passive analysis. During ordinary use, the service collects a record ℎ of observable interactions. A feature extractor maps the available observations to $x = \Phi ( h )$ , capturing patterns in requests and browser interactions. This approach relies on the assumption that agents can produce observable behavioral diferences from other automated clients. A model trained on labeled interactions uses these features to estimate $p _ { A } ( x )$ , the probability that the client is an Agent. The strength of this evidence depends on the quality of observed signals and the reliability of behavioral analysis.

Hybrid policy. In practice, a service could use a hybrid policy to combine multiple sources of evidence, such as active challenge results, passive interaction signals, and verified agent declarations [16]. The policy could also incorporate existing access controls to govern access to protected operations. To support such hybrid policies, we introduce AgentScore, a unified score expressing the service’s confidence in recognizing a client as an Agent under its policy. If the score is too low to support the requested access, the service may defer its decision and request additional verification through an active challenge or an interaction probe. The resulting evidence updates AgentScore, allowing the service to reassess the request. Moreover, the permission granted should be bound to the authorized client and permitted operations, with expiry and usage limits. We detail the proposed scoring and access policy in Section 4.3.

## 4 Im<sub>p</sub>lementation

In this section, we present our implementation of the active and passive approaches and detail the proposed AgentScore policy.

![](images/239af0362b4d83ae984eb3647bc1637e5041808fd9f40f66a62333a02cd8a51f.jpg)  
Fi<sub>g</sub>ure 2: A<sub>g</sub>ent verification framework. The <sub>p</sub>ro<sub>p</sub>osed framework combines active and <sub>p</sub>assive evidence with verified a<sub>g</sub>ent declarations throu<sub>g</sub>h a unified AgentScore to <sub>g</sub>uide access to <sub>p</sub>rotected o<sub>p</sub>erations.

## 4<sub>.</sub>1 A<sub>c</sub>ti<sub>ve</sub> V<sub>er</sub>ifi<sub>ca</sub>ti<sub>on</sub>

We implement active verification through challenge–response based on natural language understanding (NLU), a basic capability of language models. NLU tasks assess the interpretation of text and its implications [47, 60], and carefully designed questions can expose the limitations of shallow rules and pattern matching [39, 42]. Structured simulations and templates also support the systematic generation of natural-language questions and test cases [52, 63].

Building on these studies, we consider NLU a plausible basis for challenges intended to be easy for agents yet dificult for bots. We also believe that other forms of challenge such as mathematical reasoning and visual understanding have potential for agent verification. Nonetheless, we use NLU as a starting point and develop a methodology for dynamically constructing such challenges.

This methodology begins with crafting narrative templates and their accompanying questions. Each template is a long passage with slots for entity names, attribute names, and values, and the questions contain corresponding slots. Each question has a naturallanguage form for presentation and a corresponding executable function � for computing its answer once the slots are instantiated. For each challenge, a verifier uses a random seed to sample entities and attribute values from predefined sets, forming a world state � that records the sampled entities and values. It uses this state to fill a selected narrative template and its question, producing a natural-language report � and question �. The verifier applies the function � associated with � to � to compute the answer directly. The responding prover receives only the natural-language report and question. It is expected to reconstruct the relevant facts from the report and infer the question’s intended operation to derive its answer. Figure 3 illustrates this construction. We describe the protocol between verifier � and prover � as a tuple of four procedures, Π = (Setup, Gen, Interact, Verify):

(1) Setup() → L. First, prepare a resource library L containing narrative templates, paired questions, typed entities, and permitted attribute values. The verifier should keep this library L private because knowledge of its contents could allow bots to solve the challenges using fixed rules.

(2) Gen( $\mathcal { L } ; s ) \to ( c , \sigma )$ . Then, using a fresh private seed �, the verifier selects a template and samples entities and attribute values to form a world state �. It fills the template and its associated question with these facts, producing report � and question �. The function � associated with question � computes the answer $a ^ { \star } = f ( W )$ , which the verifier uses to construct an accepted-answer set � for verification. The procedure outputs the public challenge � = (�, �) and retains private verification state $\sigma = ( s , W , f , A )$

(3) Interact(�) ↔ �. The prover receives $c \ = \ ( R , q )$ and returns its answer �bin the required format. It is expected to recover the relevant facts and intended operation by interpreting the report and question.

(4) Verify(�, �b) <sup>→</sup> {0, 1}. The verifier extracts and normalizes the submitted answer and compares it against the retained set �, returning 1 on a match and 0 otherwise.

Security assumptions. The construction relies on the assumption that language models can solve the generated challenges reliably, while simple rule-based programs cannot achieve comparable success. If this assumption holds, a correct response provides evidence of agent participation. Security also depends on keeping each random seed � and the resource library L private. Knowledge of both could allow an adversary to reproduce the world state and compute the answer directly. Repeated observations of public challenges may also reveal the library’s templates and question functions. The verifier should therefore periodically update or replace L to make reconstruction from accumulated observations more dificult.

We present the evaluation results of this approach in Section 5.1 and provide further implementation details and a worked challenge example in Appendix A.

## 4<sub>.</sub>2 P<sub>ass</sub>i<sub>ve</sub> A<sub>na</sub>l<sub>ys</sub>i<sub>s</sub>

Existing bot detection studies characterize human and bot trafic through request characteristics, browser fingerprints, and interaction behavior [33, 34, 50]. A prominent industrial deployment is Cloudflare Bot Management, which combines several detection engines to assign bot scores [13]. Its Precursor system further analyzes interaction signals throughout a session [24].

However, as LLM-powered agents become participants in Web trafic, it is still unclear whether they exhibit observable patterns distinct from those of humans and traditional bots. We started by reviewing Web access approaches available to agents in terms of their supported operations and service-visible signals, as summarized in Table 1. Agents can use nearly all the access approaches listed, sharing interfaces with both humans and traditional bots. Their observable patterns can vary across interfaces and execution tools, overlapping with those of other client types. Automated scripts may also use agent-oriented interfaces and frameworks, such as MCP and Browser Use [9, 44], to pose as agents. Agent verification on the Web fundamentally involves making a finer distinction between agents and bots within automated trafic. Reliable passive analysis generally relies on large, diverse, and representative datasets, which may give providers with broad visibility into Web trafic an advantage in developing and maintaining such models. We illustrate the passive analysis pipeline in Figure 4.

![](images/97394815f716bb508f4550db8e5dcd7df7799666907a16c4db4264825c4272aa.jpg)  
Figure 3: Structured generation and semantic challenge solving. Pre<sub>p</sub>ared tem<sub>p</sub>lates and <sub>q</sub>uestion <sub>p</sub>airs (<sub>�</sub>, �) let the verifier <sub>ren</sub>d<sub>er a repor</sub>t f<sub>rom samp</sub>l<sub>e</sub>d <sub>s</sub>t<sub>a</sub>t<sub>e</sub> � <sub>an</sub>d <sub>compu</sub>t<sub>e</sub> $a ^ { \star } = f ( W )$ di<sub>rec</sub>tl<sub>y.</sub> Th<sub>e prover uses</sub> th<sub>e pu</sub>bli<sub>c c</sub>h<sub>a</sub>ll<sub>enge</sub> $c = ( R , q )$ to <sub>recons</sub>t<sub>ruc</sub>t <sub>re</sub>l<sub>evan</sub>t <sub>wor</sub>ld<sub>-s</sub>t<sub>a</sub>t<sub>e</sub> f<sub>ac</sub>t<sub>s an</sub>d i<sub>n</sub>f<sub>er</sub> th<sub>e answer</sub> f<sub>unc</sub>ti<sub>on</sub>’<sub>s seman</sub>ti<sub>cs,</sub> th<sub>en re</sub>t<sub>urns an answer</sub> f<sub>or ver</sub>ifi<sub>ca</sub>ti<sub>on.</sub>

T<sub>a</sub>bl<sub>e</sub> 1<sub>:</sub> W<sub>e</sub>b <sub>access across</sub> h<sub>umans,</sub> b<sub>o</sub>t<sub>s, an</sub>d <sub>agen</sub>t<sub>s.</sub> Th<sub>e</sub> t<sub>a-</sub> ble summarizes re<sub>p</sub>resentative interfaces and tools t<sub>yp</sub>icall<sub>y</sub> <sub>use</sub>d b<sub>y</sub> h<sub>umans,</sub> b<sub>o</sub>t<sub>s,</sub> <sub>an</sub>d LLM<sub>-powere</sub>d <sub>agen</sub>t<sub>s,</sub> th<sub>e</sub>i<sub>r</sub> <sub>poss</sub>i<sub>-</sub> bl<sub>e</sub> <sub>opera</sub>ti<sub>ons,</sub> <sub>an</sub>d th<sub>e</sub> <sub>s</sub>i<sub>gna</sub>l<sub>s</sub> <sub>o</sub>b<sub>serva</sub>bl<sub>e</sub> b<sub>y</sub> W<sub>e</sub>b <sub>serv</sub>i<sub>ces.</sub>
<table><tr><td>Interface / examples</td><td>Usersa Ops.b Signalsc</td><td></td><td></td></tr><tr><td>GUI interfaces: Browser GUI; WebView [1, 59]</td><td>8张</td><td>R/W</td><td>H/B/E</td></tr><tr><td>HTTP / API clients: curl; SDKs [20] Browser automation: Playwright;</td><td>血张</td><td>R/W</td><td>H</td></tr><tr><td>Selenium [43, 54]</td><td>由 生</td><td>R/W</td><td>H/B/E</td></tr><tr><td>Browser extensions: Content scripts; Claude in Chrome [3, 11]</td><td>血然</td><td>R/W</td><td>H/B/E</td></tr><tr><td>Search / fetch: web_search; web_fetch [4, 5]</td><td>张</td><td>R</td><td>Var.d</td></tr><tr><td>Agent browser frameworks: Browser Use [9]</td><td>张</td><td>R/W</td><td>H/B/E</td></tr><tr><td>Agent desktop tools: Computer Use [2]</td><td>张</td><td>R/W</td><td>H/B/E</td></tr><tr><td>MCP tools: Brave; Stripe; GitHub [7, 29, 56]</td><td>张</td><td>R/W</td><td>Var.d</td></tr></table>

<sup>a</sup> Typical user types: Human; Bot; Agent. Icons indicate typical users; users can use other interfaces too; for example, humans can also use curl.  
<sup>b</sup> Possible operations: R: read; W: write or change service state. Final available operations depend on the interface and permissions.  
<sup>c</sup> Observable signals: H: HTTP; B: browser environment; E: page events. B/E require page execution and data collection from the browser.  
<sup>d</sup> Var.: Observable signals depend on the underlying tool implementation.

Here, we created a minimal dataset containing Web interactions from human participants, traditional bots, and LLM-powered agents, together with associated client fingerprints. Then, we conduct a preliminary evaluation using supervised classification methods to explore the feasibility of this approach. We present the experimental details and results in Section 5.

## 4.3 H<sub>y</sub>brid Scorin<sub>g</sub> Polic<sub>y</sub>

We detail how a service could integrate the available evidence into an AgentScore. Similar to Cloudflare’s Bot Score [13], AgentScore denotes the service’s confidence in recognizing a client as an agent and can be used alongside existing access controls to govern protected operations.

Unified score. We express the available evidence as an AgentScore on a common scale in [0, 100]. Let $r \in \{ 0 , 1 \}$ indicate whether a client passes an active challenge or presents a verified agent declaration [16], and let $u \in \{ 0 , 1 \}$ indicate whether suficient observations for passive analysis are available. The proposed score is

$$
S ( x , r , u ) = \left\{ \begin{array} { l l } { 1 0 0 , } & { r = 1 , } \\ { 1 0 0 \ p _ { A } ( x ) , } & { r = 0 \ \mathrm { a n d } u = 1 , } \\ { \perp , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.\tag{1}
$$

The rule-assigned score of 100 expresses the service’s decision to accept the verified evidence under its policy. For the passive branch, $p _ { A } ( x )$ from Section 4.2 is scaled to $1 0 0 p _ { A } ( x )$ . Here, ⊥ means that neither verified rule evidence nor a usable passive estimate is available, leaving further handling to the service’s policy.

Access decision. AgentScore could be used alongside identification and authorization to inform access decisions. For each protected operation �, let $\tau _ { o } \in [ 0 , 1 0 0 ]$ be the required score and $C _ { \pi } ( i , o ) \in \{ 0 , 1 \}$ indicate whether client � satisfies these controls under policy �. For score $s = S ( x , r , u )$ , the final access decision � can be expressed as

$$
d = \mathrm { a l l o w } \iff s \in [ \tau _ { o } , 1 0 0 ] \land C _ { \pi } ( i , o ) = 1 .\tag{2}
$$

Otherwise, the policy determines whether to deny or defer the request. If $s = \bot \mathrm { o r } s < \tau _ { o }$ , the service may request an active challenge or an interaction probe, such as a Turnstile-like embedded webpage component [14], to update AgentScore and reassess access.

![](images/dbf2e6b58810f5f5629d550779a3e6270de11b95f5fef669868f8ed15fd81d91.jpg)  
Fi<sub>g</sub>ure 4: Passive anal<sub>y</sub>sis <sub>p</sub>i<sub>p</sub>eline. Ofline trainin<sub>g</sub> and <sub>p</sub>ro<sub>p</sub>osed calibration <sub>p</sub>re<sub>p</sub>are a fixed <sub>p</sub>redictor for online A<sub>g</sub>ent-<sub>p</sub>robabilit<sub>y</sub> estimation<sub>;</sub> insuficient observations defer estimation.

Granted permissions should also be bound to the authorized client and permitted operations, with expiry and usage limits. Together, these mechanisms could give Web services finer control over automated access: allowing AI agents to act on behalf of users while limiting large-scale scripted abuse of service resources.

## 5 E<sub>va</sub>l<sub>ua</sub>ti<sub>on</sub>

## 5.1 Challen<sub>g</sub>e–Res<sub>p</sub>onse Evaluation

Experimental setup. We evaluated the proposed NLU challenge design using eight language models and two rule-based scripts for text processing on the same 100 questions. Each challenge contains a textual report of 591–1,090 words, with a median of 819.5 words. Appendix A provides the implementation settings.

We implement two rule-based scripts, Nearest Text and Shallow Relation. Nearest Text selects answer spans using keyword overlap and text proximity. Shallow Relation uses predefined patterns to extract entity–attribute–value relations, then retrieves attribute values or compares numbers to answer the question. For comparison, the models receive the same report, question, and structured-answer instruction. After extracting the first explicit answer, the verifier normalizes type prefixes and checks the result against the accepted-answer set.

Challenge performance. Table 2 shows that the eight models achieve 85–100% accuracy, compared with 21% for Nearest Text and 4% for Shallow Relation. Overall, the models answered 772 of the 800 assigned questions correctly; the remaining 28 responses (3.5%) include incorrect answers and malformed output. The scripts’ successes show that the generation process can yield instances vulnerable to simple solvers. A verifier could introduce a screening procedure to filter out weak instances before use, similar to candidate screening during key generation [46].

Verification overhead. Next we examined the overhead introduced by this verification process. On the client side, Figure 5 shows the time and tokens spent answering challenges, with median response times of 1.4–6.1 seconds across the eight models. Verification will surely incur additional token costs and processing time for agents, just as CAPTCHAs cost humans time and attention.

T<sub>a</sub>bl<sub>e</sub> 2<sub>:</sub> NLU <sub>c</sub>h<sub>a</sub>ll<sub>enge per</sub>f<sub>ormance an</sub>d <sub>resource use.</sub> E<sub>ac</sub>h client is assi<sub>g</sub>ned the same 100 <sub>q</sub>uestions.
<table><tr><td>Model</td><td>Acc. (%)</td><td>Time (s)ª</td><td>Inputb</td><td>Outputb</td></tr><tr><td>DeepSeek-V4.1-Flash</td><td>100</td><td>2.918</td><td>1380.9</td><td>200.4</td></tr><tr><td>Qwen3.8-Flash</td><td>100</td><td>6.115</td><td>1413.2</td><td>238.9</td></tr><tr><td>GPT-5.6-Luna</td><td>99</td><td>2.062</td><td>1332.5</td><td>42.5</td></tr><tr><td>∅Grok-4.3</td><td>99</td><td>3.784</td><td>1509.4</td><td>307.9</td></tr><tr><td>Z GLM-5.3-Flash</td><td>98</td><td>2.309</td><td>1335.0</td><td>201.6</td></tr><tr><td>Claude-Haiku-5.5</td><td>96</td><td>2.317</td><td>2287.4</td><td>65.7</td></tr><tr><td>Gemini-3.5-Flash-Lite</td><td>95</td><td>1.625</td><td>1390.8</td><td>13.4</td></tr><tr><td>MiniMax-M3.1-Flash</td><td>85</td><td>1.400</td><td>1463.7</td><td>8.9</td></tr><tr><td>&gt;_ Nearest Text</td><td>21</td><td>0.00297</td><td>一</td><td>一</td></tr><tr><td>Shallow Relation</td><td>4</td><td>0.00189</td><td>一</td><td>一</td></tr></table>

<sup>a</sup> Time: Median observed solving time (model response or script execution time).  
<sup>b</sup> Tokens: Mean input and output tokens per challenge response.

T<sub>a</sub>bl<sub>e</sub> 3<sub>:</sub> S<sub>erver ver</sub>ifi<sub>ca</sub>ti<sub>on over</sub>h<sub>ea</sub>d<sub>.</sub> M<sub>e</sub>di<sub>an an</sub>d P95 <sub>per-ca</sub>ll <sub>cos</sub>t<sub>s, w</sub>ith <sub>me</sub>di<sub>an ser</sub>i<sub>a</sub>li<sub>ze</sub>d <sub>s</sub>i<sub>zes o</sub>f <sub>c</sub>h<sub>a</sub>ll<sub>enges an</sub>d <sub>pr</sub>i<sub>va</sub>t<sub>e</sub> <sub>ver</sub>ifi<sub>ca</sub>ti<sub>on s</sub>t<sub>a</sub>t<sub>es.</sub>
<table><tr><td rowspan="2">Phase</td><td colspan="2">Wall (µs)</td><td colspan="2">CPU (µs)</td><td colspan="2">Instr. (106)</td></tr><tr><td>Median</td><td>P95</td><td>Median</td><td>P95</td><td>Median</td><td>P95</td></tr><tr><td>Assembly</td><td>324.3</td><td>373.2</td><td>323.2</td><td>371.1</td><td>6.546</td><td>7.946</td></tr><tr><td>Verification</td><td>3.48</td><td>3.91</td><td>3.45</td><td>3.89</td><td>0.0788</td><td>0.0875</td></tr><tr><td>Total</td><td>327.6</td><td>376.6</td><td>326.5</td><td>374.6</td><td>6.626</td><td>8.024</td></tr></table>

Challenge / private state: 6.21 / 10.06 kB<sup>a</sup>  
<sup>a</sup> Sizes: Compact UTF-8 JSON; private state includes its template.

On the server side, Table 3 reports a combined median of 327.6 microseconds for challenge assembly and response verification. With templates prepared ofline, the service only needs to assemble each challenge and check the response at runtime, keeping the overhead low for concurrent verification at scale.

![](images/e95c0308a49565dbc71c60f616943061b526e38c60d3b79def07258ebd406959.jpg)  
(a) Response time.  
(b) Token consumption.  
Fi<sub>g</sub>ure 5: Model verification overhead. Res<sub>p</sub>onse time uses a lo<sub>g</sub>arithmic axis and tokens use a linear axis.

## 5<sub>.</sub>2 P<sub>ass</sub>i<sub>ve</sub> A<sub>na</sub>l<sub>ys</sub>i<sub>s</sub>

Experimental setup. To collect behavioral data, we used the WebArena shopping website [69], with three custom tasks: Wi-Fi adapters, USB drive, and wireless mice. The Wi-Fi task uses fixed instructions, while the USB and mouse tasks allow openended product selection within attribute and budget constraints. We recruited twelve participants from our institute with institutional review board (IRB) approval and selected three Bot and three Agent implementations for the main comparison, as summarized in Table 4. Human participants completed the tasks manually using browsers on their own devices. Bots followed predefined rules, while Agents used language models to interpret observations and select actions. The adversarial settings include bots posing as humans using native mouse and keyboard input, and bots posing as agents using Browser Use or Playwright MCP.

During browsing, we collected HTTP metadata, browser characteristics, and page events, with pointer movement and scrolling recorded. Finally, we retained a dataset of720 visits: 72 Human visits (12 participants × 3 tasks × 2 rounds) and 648 automatic visits (432 Bot and 216 Agent, with 24 visits per task for each configuration). Appendix B provides detailed experimental settings.

HTTP behavior and client fingerprints. Table 5 summarizes service-visible HTTP behavior and client fingerprints. To characterize the networking software, we also examine JA4 and $\mathrm { J } \mathrm { A } 4 \mathrm { H } _ { a b }$ from the JA4+ suite, which provide TLS and HTTP fingerprints, respectively [27]. From the HTTP perspective, Bot visits have a shorter median request span (4.37 s) than Human and Agent visits (39.42 and 43.55 s), while Agent visits have a longer median request gap (3220.34 ms) than Human and Bot visits (586.47 and 395.15 ms). However, we believe these timing patterns are unreliable indicators because they are easy to alter. From the fingerprinting perspective, Bot and Agent configurations using the same tools share identical JA4 and $\mathrm { J } \mathrm { A } 4 \mathrm { H } _ { a b }$ values, while ordinary Chrome also matches several automatic configurations, motivating us to further examine client interaction behavior for distinction.

Observed interaction patterns. Figure 6 compares page interactions across Humans and interaction-capable Bot and Agent implementations. From the results, we see that Human visits generally contain more pointer movements and scrolls. On the same tasks, even though clients may share the same interfaces, they exhibit diferent patterns of interaction. Across repeated visits, Humans show the most diverse event-count combinations, while Bots are generally more repetitive, and Agents often fall between them. Features and model comparison. We extract 40 service-visible features from each visit ℎ:

T<sub>a</sub>bl<sub>e</sub> 4<sub>:</sub> Cli<sub>en</sub>t <sub>con</sub>fi<sub>gura</sub>ti<sub>ons.</sub> I<sub>mp</sub>l<sub>emen</sub>t<sub>a</sub>ti<sub>ons an</sub>d <sub>o</sub>b<sub>serva-</sub> tion in<sub>p</sub>uts for the main com<sub>p</sub>arison and adversarial settin<sub>g</sub>s.
<table><tr><td>User</td><td>Implementation</td><td>Inputª</td></tr><tr><td>8 Human</td><td>Participant browser</td><td>Rendered pages</td></tr><tr><td rowspan="3">Bot</td><td>Python Requests</td><td>HTML / forms</td></tr><tr><td>Playwright Python</td><td>DOM</td></tr><tr><td>Selenium WebDriver</td><td>DOM</td></tr><tr><td rowspan="3">Agent</td><td>Browser Use</td><td>DOM</td></tr><tr><td>Playwright MCP</td><td>AX tree</td></tr><tr><td>Codex Computer Use</td><td>Screenshots</td></tr><tr><td>Adversarial settings</td><td></td><td></td></tr><tr><td>poses as 8</td><td>Native input</td><td>DOM</td></tr><tr><td rowspan="2">白poses as</td><td>Browser Use</td><td>DOM</td></tr><tr><td>Playwright MCP</td><td>DOM + AX tree</td></tr></table>

<sup>a</sup> AX tree: Accessibility tree, a textual description of page elements.

Table 5: HTTP behavior and client fin<sub>g</sub>er<sub>p</sub>rints. Median HTTP measurements and fin<sub>g</sub>er<sub>p</sub>rint <sub>g</sub>rou<sub>p</sub>s across clients.
<table><tr><td>Attribute</td><td>Human (72)</td><td>Bot (432)</td><td>Agent (216)</td></tr><tr><td>HTTP requests</td><td>14</td><td>9</td><td>10.5</td></tr><tr><td>Peak concurrency</td><td>1</td><td>2</td><td>1</td></tr><tr><td>Request span (s)a</td><td>39.42</td><td>4.37</td><td>43.55</td></tr><tr><td>Request gap (ms)b</td><td>586.47</td><td>395.15</td><td>3220.34</td></tr><tr><td>JA4</td><td>T2, T3</td><td>T1, T2</td><td>T2</td></tr><tr><td> $\mathrm { J A 4 H } _ { a b }$ </td><td>H2, H4</td><td>H1, H2, H3</td><td>H2, H3</td></tr></table>

<sup>a</sup> Span: Time between the first and last request starts within a visit.  
<sup>b</sup> Gap: Median interval between consecutive request starts within a visit.

$$
x = \Phi ( h ) = ( x _ { \mathrm { B } } , x _ { \mathrm { A } } , x _ { \mathrm { I } } , x _ { \mathrm { S } } , x _ { \mathrm { T } } ) ,
$$

where B/A/I/S/T denote Browser, Activity, Input, Sequence, and Timing, containing 12, 5, 9, 7, and 7 features, respectively. We specify their components in Table 8 (Appendix B).

We compare logistic regression (LR), random forest (RF), RBF-SVM [18], and XGBoost [10] using nested cross-validation. Each outer fold trains on two tasks and tests on the remaining task. Human data are split by participant, with P01–P08 used for training and P09–P12 for testing; both rounds of each participant follow this assignment. Within each outer training set, two inner folds select parameters separately for each model and feature setting. Each task contributes 200 test visits, giving 600 test visits across the three tasks: 24 Human, 360 Bot, and 216 Agent visits.

Table 6 shows that RF achieves the highest macro-F1 (1.0000), followed by XGBoost (0.9913), correctly labeling 600 and 593 of the 600 test visits, respectively. All four models correctly identify the 24 Human visits, with errors confined to Bot–Agent confusions. LR and RBF-SVM label 47 and 46 Bot visits as Agents, respectively, whereas XGBoost labels 7 Agent visits as Bots. Figure 7 shows the efects of cumulatively adding feature groups. Activity and Timing provide the largest gains across all four models (15.7–23.5 and 6.2– 8.1 pp, respectively), while Input has mixed efects and Sequence adds smaller improvements. These results suggest the feasibility of distinguishing Humans, Bots, and Agents using passive analysis of service-visible interaction patterns. However, given the limited scale of our dataset, we believe a larger and more comprehensive study would be a valuable direction for future work.

![](images/a605131cbb1b973969ec16e21d3d211e8a30465604e702569310fc29c4118e5c.jpg)  
(a) Clicks.  
(b) Pointer moves.  
(c) Scrolls.  
Fi<sub>g</sub>ure 6: Interaction <sub>p</sub>atterns on three sho<sub>pp</sub>in<sub>g</sub> tasks. Bars and ticks mark inter<sub>q</sub>uartile ran<sub>g</sub>es and medians. Axes use log(1 + �) scaling; points may overlap.

T<sub>a</sub>bl<sub>e</sub> 6<sub>:</sub> P<sub>ass</sub>i<sub>ve c</sub>li<sub>en</sub>t <sub>c</sub>l<sub>ass</sub>ifi<sub>ca</sub>ti<sub>on.</sub> R<sub>ows</sub> i<sub>n</sub>di<sub>ca</sub>t<sub>e</sub> t<sub>rue</sub> <sub>c</sub>l<sub>asses</sub><sup>a</sup><sub>; co</sub>l<sub>umns</sub> i<sub>n</sub>di<sub>ca</sub>t<sub>e pre</sub>di<sub>c</sub>ti<sub>ons, an</sub>d <sub>re</sub>d <sub>ce</sub>ll<sub>s mar</sub>k <sub>er-</sub> <sub>rors.</sub> T<sub>as</sub>k<sub>-</sub>l<sub>eve</sub>l <sub>macro-</sub>F1 <sub>scores</sub><sup>b</sup> <sub>appear</sub> b<sub>e</sub>l<sub>ow eac</sub>h <sub>ma</sub>t<sub>r</sub>i<sub>x.</sub>
<table><tr><td rowspan="2">LR True\Pred.</td><td colspan="4">Macro-F1: 0.9440 RF</td><td colspan="3">Macro-F1: 1.0000</td></tr><tr><td>H</td><td>B</td><td>A</td><td>True\Pred.</td><td>H</td><td>B</td><td>A</td></tr><tr><td>H (24)</td><td>24</td><td>0</td><td>0</td><td>H (24)</td><td>24</td><td>0</td><td>0</td></tr><tr><td>B (360)</td><td>0</td><td>313</td><td>47</td><td>B (360)</td><td>0</td><td>360</td><td>0</td></tr><tr><td>A (216)</td><td>0</td><td>0</td><td>216</td><td>A (216)</td><td>0</td><td>0</td><td>216</td></tr><tr><td>Macro-F1: W 1.000 U 1.000 M 0.837</td><td></td><td></td><td></td><td>Macro-F1: W 1.000 U 1.000 M 1.000</td><td></td><td></td><td></td></tr><tr><td>RBF-SVM</td><td>Macro-F1: 0.9452</td><td></td><td></td><td>XGBoost</td><td></td><td>Macro-F1: 0.9913</td><td></td></tr><tr><td>True\Pred.</td><td>H</td><td>B</td><td>A</td><td>True\Pred.</td><td>H</td><td>B</td><td>A</td></tr><tr><td>H (24)</td><td>24</td><td>0</td><td>0</td><td>H (24)</td><td>24</td><td>0</td><td>0</td></tr><tr><td>B (360)</td><td>0</td><td>314</td><td>46</td><td>B (360)</td><td>0</td><td>360</td><td>0</td></tr><tr><td>A (216)</td><td>0</td><td>0</td><td>216</td><td>A (216)</td><td>0</td><td>7</td><td>209</td></tr><tr><td>Macro-F1: W 1.000 U 1.000 M 0.840 Macro-F1: W 1.000 U 0.974 M 1.000</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

<sup>a</sup> Classes: H = Human (24), B = Bot (360), A = Agent (216).  
<sup>b</sup> Tasks: W = Wi-Fi adapters, U = USB drive, M = wireless mice.

## 6 Di<sub>scuss</sub>i<sub>o</sub>n

Abuse Cost. An attacker may use a malicious agent or language model to solve a challenge while pursuing abusive goals. This is analogous to malicious humans passing CAPTCHAs themselves or outsourcing them to CAPTCHA-solving services [45]. Such model use meets the participation requirement and incurs inference costs. Under the challenge assumptions, active verification could therefore raise the cost of access for clients that would otherwise rely entirely on scripts. Its practical value depends on whether this added cost changes the economics of large-scale abuse without imposing excessive overhead on agents performing normal tasks.

![](images/e62ff24fe1a4eda8f9d5d56910d63a9e2f6220389e9b8f25a6b62022b05ea8d3.jpg)  
(a) LR.

![](images/3d7c4a44ce94ed701ab8e2d52475a2987e4e09746f83df6238276b3e4ba64554.jpg)

![](images/b06c9496799216ef560b1d1078d7f7b3959ddadf4cf1b622720c9d31771f3692.jpg)  
(c) RBF-SVM.

(b) RF.  
![](images/b1c2442ec139c37f9a786361c93970f34680a409eeba7d9055ed6b4ec20aebbc.jpg)  
(d) XGBoost.  
Fi<sub>gu</sub>r<sub>e</sub> 7: C<sub>u</sub>m<sub>u</sub>l<sub>a</sub>ti<sub>ve</sub> f<sub>ea</sub>t<sub>u</sub>r<sub>e a</sub>bl<sub>a</sub>ti<sub>o</sub>n<sub>.</sub> B/A/I/S/T d<sub>e</sub>n<sub>o</sub>t<sub>e</sub> Br<sub>owse</sub>r<sub>,</sub> A<sub>c</sub>ti<sub>v</sub>it<sub>y,</sub> In<sub>pu</sub>t<sub>,</sub> S<sub>eque</sub>n<sub>ce,</sub> <sub>a</sub>nd Timin<sub>g.</sub> Fl<sub>oa</sub>tin<sub>g</sub>- bar labels show macro-F1 changes (pp).

Verification Scope. From this economic perspective, verification scope afects the cost of sustained access. A client may use a model to pass a challenge and then perform subsequent operations through scripts. The inference cost is then spread across those operations, reducing the average cost per operation. Verification frequency and permission scope therefore need to be considered together. Frequent challenges add overhead to normal agent activity, whereas broad permissions allow more scripted operations after a single check.

Practical Deployment. Passive analysis uses evidence from ordinary interactions and can reduce verification overhead during normal agent activity. However, a script that is able to reproduce agent-like signals could also receive a high score without using a model. A practical policy should consider when passive evidence is suficient and when an active challenge is warranted. Stricter and more frequent verification could help maintain agent participation during continued access, but would also increase the burden on ordinary agents. The acceptability of this overhead could be explored in future deployments. Like CAPTCHAs and bot detection mechanisms, Proof-of-Agentness faces a usability challenge: the added verification requirements may not be welcomed by all users.

## 7 C<sub>o</sub>n<sub>c</sub>l<sub>us</sub>i<sub>o</sub>n

We introduced Proof-of-Agentness (PoA) as a mechanism for verifying language model participation in an interaction and outlined two approaches to obtaining evidence: active verification and passive analysis. These mechanisms could help Web services regulate automated trafic more selectively, admitting agents acting on users’ behalf while limiting large-scale scripted abuse.

## R<sub>e</sub>f<sub>erences</sub>

[1] Android Developers. n.d.. Build Web Apps in WebView. https://developer. android.com/develop/ui/views/layout/webapps/webview Oficial documentation accessed 2026-10-23.

[2] Anthropic. n.d.. Computer Use Tool. https://platform.claude.com/docs/en/ agents-and-tools/tool-use/computer-use-tool Oficial documentation accessed 2026-10-21.

[3] Anthropic. n.d.. Get Started with Claude in Chrome. https://support.claude.com/ en/articles/12012173-get-started-with-claude-in-chrome Oficial documentation accessed 2026-10-21.

[4] Anthropic. n.d.. Web Fetch Tool. https://platform.claude.com/docs/en/agentsand-tools/tool-use/web-fetch-tool Oficial documentation accessed 2026-10-21.

[5] Anthropic. n.d.. Web Search Tool. https://platform.claude.com/docs/en/agentsand-tools/tool-use/web-search-tool Oficial documentation accessed 2026-10-21.

[6] Atlassian. n.d.. Atlassian MCP Server. https://atlassian.github.io/atlassian-mcpserver/ Documentation accessed 2026-10-21.

[7] Brave. n.d.. Brave Search MCP Server. https://github.com/brave/brave-searchmcp-server Oficial repository accessed 2026-10-21.

[8] Browser Use. n.d.. Browser Actor. https://github.com/browser-use/browseruse/blob/main/browser\_use/actor/README.md Oficial documentation accessed 2026-10-21.

[9] Browser Use. n.d.. Browser Use: Open-Source Browser Agents. https://github. com/browser-use/browser-use Oficial repository accessed 2026-10-23.

[10] Tianqi Chen and Carlos Guestrin. 2016. XGBoost: A Scalable Tree Boosting System. In Proceedings ofthe 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. 785–794. doi:10.1145/2939672.2939785

[11] Chrome for Developers. n.d.. Content Scripts. https://developer.chrome.com/ docs/extensions/develop/concepts/content-scripts Oficial documentation ac cessed 2026-10-23.

[12] Cloudflare. 2026. Bot detection engines. https://developers.cloudflare.com/bots/ concepts/bot-detection-engines/ Documentation accessed 2026-10-14.

[13] Cloudflare. 2026. Bot scores. https://developers.cloudflare.com/bots/concepts/ bot-score/ Documentation accessed 2026-10-14.

[14] Cloudflare. 2026. Turnstile. https://developers.cloudflare.com/turnstile/ Documentation accessed 2026-10-24.

[15] Cloudflare. 2026. Verified bots. https://developers.cloudflare.com/bots/concepts/ bot/verified-bots/ Updated October 8, 2026; accessed 2026-10-14.

[16] Cloudflare. 2026. Web Bot Auth. https://developers.cloudflare.com/bots/ reference/bot-verification/web-bot-auth/ Updated October 8, 2026; accessed 2026-10-14.

[17] Matthew Conroy. 2026. The Internet has a second audience. Cloudflare Blog. https://blog.cloudflare.com/agentic-web/ Published September 30, 2026; accessed 2026-10-21.

[18] Corinna Cortes and Vladimir Vapnik. 1995. Support-Vector Networks. Machine Learning 20, 3 (1995), 273–297. doi:10.1007/BF00994018

[19] Zach Cowan and Mitul Shah. 2026. Making agent-friendly pages with content negotiation. Vercel Blog. https://vercel.com/blog/making-agent-friendly-pageswith-content-negotiation Published February 3, 2026; accessed 2026-10-15.

[20] curl. n.d.. curl—How To Use. https://curl.se/docs/manpage.html Oficial documentation accessed 2026-10-23.

[21] Xiang Deng, Yu Gu, Boyuan Zheng, Shijie Chen, Sam Stevens, Boshi Wang, Huan Sun, and Yu Su. 2023. Mind2Web: Towards a Generalist Agent for the Web. In Advances in Neural Information Processing Systems, Vol. 36. Curran Associates, Inc., 28091–28114. doi:10.52202/075280-1220

[22] Alexandre Drouin, Maxime Gasse, Massimo Caccia, Issam H. Laradji, Manuel Del Verme, Tom Marty, David Vazquez, Nicolas Chapados, and Alexandre Lacoste. 2024. WorkArena: How Capable are Web Agents at Solving Common Knowledge Work Tasks?. In Proceedings of the 41st International Conference on Machine Learning (Proceedings ofMachine Learning Research, Vol. 235). PMLR, 11642– 11662. https://proceedings.mlr.press/v235/drouin24a.html

[23] Cynthia Dwork and Moni Naor. 1993. Pricing via Processing or Combatting Junk Mail. In Advances in Cryptology—CRYPTO ’92 (Lecture Notes in Computer Science, Vol. 740). Springer, Berlin, Heidelberg, 139–147. doi:10.1007/3-540-48071-4\_10

[24] Marina Elmore and Benedikt Wolters. 2026. Introducing Precursor: Detecting Agentic Behavior with Continuous Client-Side Signals. Cloudflare Blog. https: //blog.cloudflare.com/introducing-precursor/ Published July 13, 2026; accessed 2026-10-23.

[25] Exa. n.d.. Exa MCP. https://exa.ai/docs/get-started/exa-mcp Documentation accessed 2026-10-21.

[26] Iliana Fayolle, Sihem Bouhenniche, Samuel Pélissier, Pierre Laperdrix, Clémentine Maurice, and Walter Rudametkin. 2026. On the Internet, Nobody Knows You’re an LLM Bot: Unmasking Web Agents with Multi-Layer Fingerprinting. arXiv:2606.30119v1. doi:10.48550/arXiv.2606.30119

[27] FoxIO. n.d.. JA4+ Network Fingerprinting. https://github.com/FoxIO-LLC/ja4/ tree/c6469a7b0dbf8fe963810162429fbfd9d1acc4d5 Reference implementation accessed 2026-10-25.

[28] Sandro Rodriguez Garzon, Awid Vaziry, Enis Mert Kuzu, Dennis Enrique Gehrmann, Buse Varkan, Alexander Gaballa, and Axel Küpper. 2026. AI Agents with Decentralized Identifiers and Verifiable Credentials. In Proceedings ofthe 18th International Conference on Agents and Artificial Intelligence - Volume 1: ICAART. SciTePress, 252–259. doi:10.5220/0014234400004052

[29] GitHub. n.d.. GitHub MCP Server. https://github.com/github/github-mcp-server Oficial repository accessed 2026-10-21.

[30] Izzeddin Gur, Hiroki Furuta, Austin Huang, Mustafa Safdari, Yutaka Matsuo, Douglas Eck, and Aleksandra Faust. 2024. A Real-World WebAgent with Planning, Long Context Understanding, and Program Synthesis. In The Twelfth International Conference on Learning Representations. 52690–52717. https://proceedings.iclr.cc/paper\_files/paper/2024/hash/ e91bf7dfba0477554994c6d64833e9d8-Abstract-Conference.html

[31] Hongliang He, Wenlin Yao, Kaixin Ma, Wenhao Yu, Yong Dai, Hongming Zhang, Zhenzhong Lan, and Dong Yu. 2024. WebVoyager: Building an End-to-End Web Agent with Large Multimodal Models. In Proceedings ofthe 62nd Annual Meeting ofthe Association for Computational Linguistics (Volume 1: Long Papers). Association for Computational Linguistics, 6864–6890. doi:10.18653/v1/2024.acllong.371

[32] Hugging Face. n.d.. Hugging Face MCP Server. https://huggingface.co/docs/ hub/en/agents-mcp Documentation accessed 2026-10-21.

[33] Christos Iliou, Theodoros Kostoulas, Theodora Tsikrika, Vasilis Katos, Stefanos Vrochidis, and Ioannis Kompatsiaris. 2021. Detection of Advanced Web Bots by Combining Web Logs with Mouse Behavioural Biometrics. Digital Threats: Research and Practice 2, 3 (2021), 1–26. doi:10.1145/3447815

[34] Hugo Jonker, Benjamin Krumnow, and Gabry Vlot. 2019. Fingerprint Surface-Based Detection of Web Bot Detectors. In Computer Security—ESORICS 2019 (Lecture Notes in Computer Science, Vol. 11736). Springer International Publishing, Cham, 586–605. doi:10.1007/978-3-030-29962-0\_28

[35] Ari Juels and John Brainard. 1999. Client Puzzles: A Cryptographic Countermeasure Against Connection Depletion Attacks. In NDSS. Internet Society, 15 pages. https://www.ndss-symposium.org/ndss1999/cryptographic-defenseagainst-connection-depletion-attacks/

[36] Kaggle. n.d.. MCP Server Documentation. https://www.kaggle.com/docs/mcp Documentation accessed 2026-10-21.

[37] Jing Yu Koh, Robert Lo, Lawrence Jang, Vikram Duvvur, Ming Lim, Po-Yu Huang, Graham Neubig, Shuyan Zhou, Russ Salakhutdinov, and Daniel Fried. 2024. VisualWebArena: Evaluating Multimodal Agents on Realistic Visual Web Tasks. In Proceedings ofthe 62nd Annual Meeting ofthe Association for Computational Linguistics (Volume 1: Long Papers). Association for Computational Linguistics, 881–905. doi:10.18653/v1/2024.acl-long.50

[38] Jin-Hee Lee. 2025. The age of agents: cryptographically recognizing agent trafic. Cloudflare Blog. https://blog.cloudflare.com/signed-agents/ Published August 28, 2025; accessed 2026-10-14.

[39] Hector J. Levesque, Ernest Davis, and Leora Morgenstern. 2012. The Winograd Schema Challenge. In Proceedings ofthe Thirteenth International Conference on Principles ofKnowledge Representation and Reasoning. AAAI Press, Menlo Park, CA, USA, 552–561. https://cdn.aaai.org/ocs/4492/4492-21843-1-PB.pdf

[40] William Lugoloobi, Samuelle Marro, Jabez Magomere, Joss Wright, and Chris Russell. 2026. Known By Their Actions: Fingerprinting LLM Browser Agents via UI Traces. arXiv:2605.14786v1. doi:10.48550/arXiv.2605.14786 Accepted at NeurIPS 2026. Author implementation: https://github.com/KabakaWilliam/ known\_actions.

[41] Celso Martinho and Will Allen. 2026. Introducing Markdown for Agents. Cloudflare Blog. https://blog.cloudflare.com/markdown-for-agents/ Published February 12, 2026; accessed 2026-10-15.

[42] R. Thomas McCoy, Ellie Pavlick, and Tal Linzen. 2019. Right for the Wrong Reasons: Diagnosing Syntactic Heuristics in Natural Language Inference. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics. Association for Computational Linguistics, Florence, Italy, 3428–3448. doi:10.18653/v1/P19-1334

[43] Microsoft. n.d.. BrowserType: Connecting to an Existing Browser. https: //playwright.dev/python/docs/api/class-browsertype Oficial documentation accessed 2026-10-23.

[44] Model Context Protocol. 2026. MCP Specification: Tools. https:// modelcontextprotocol.io/specification/2026-07-28/server/tools Protocol revision 2026-07-28; accessed 2026-10-14.

[45] Marti Motoyama, Kirill Levchenko, Chris Kanich, Damon McCoy, Geof frey M. Voelker, and Stefan Savage. 2010. Re: CAPTCHAs—Understanding CAPTCHA-Solving Services in an Economic Context. In 19th USENIX Security Symposium (USENIX Security 10). USENIX Association, Washington, DC. https://www.usenix.org/conference/usenixsecurity10/re-captchas%E2% 80%94understanding-captcha-solving-services-economic-context

[46] National Institute of Standards and Technology. 2023. Digital Signature Standard (DSS). Federal Information Processing Standards Publication 186-5. National Institute of Standards and Technology, Gaithersburg, MD, USA. doi:10.6028/ NIST.FIPS.186-5

[47] Yixin Nie, Adina Williams, Emily Dinan, Mohit Bansal, Jason Weston, and Douwe Kiela. 2020. Adversarial NLI: A New Benchmark for Natural Language Un derstanding. In Proceedings ofthe 58th Annual Meeting ofthe Association for Computational Linguistics. Association for Computational Linguistics, Online, 4885–4901. doi:10.18653/v1/2020.acl-main.441

[48] Notion. n.d.. Notion MCP. https://www.notion.com/help/notion-mcp Documentation accessed 2026-10-21.

[49] OpenRouter. n.d.. OpenRouter Quickstart Guide. https://openrouter.ai/docs/ quickstart Oficial documentation accessed 2026-10-26.

[50] KyoungSoo Park, Vivek S. Pai, Kang-Won Lee, and Seraphin Calo. 2006. Securing Web Service by Automatic Robot Detection. In 2006 USENIX Annual Technical Conference (USENIX ATC 06). USENIX Association, 255– 260. https://www.usenix.org/conference/2006-usenix-annual-technicalconference/securing-web-service-automatic-robot-detection

[51] PayPal. 2026. MCP server quickstart guide. https://developer.paypal.com/aitools/mcp-server Updated June 30, 2026; accessed 2026-10-21.

[52] Marco Tulio Ribeiro, Tongshuang Wu, Carlos Guestrin, and Sameer Singh. 2020. Beyond Accuracy: Behavioral Testing of NLP Models with CheckList. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics. Association for Computational Linguistics, Online, 4902–4912. doi:10.18653/v1/2020.acl-main.442

[53] Nikhil Sardana, Jacob Portes, Sasha Doubov, and Jonathan Frankle. 2024. Be yond Chinchilla-Optimal: Accounting for Inference in Language Model Scal ing Laws. In Proceedings ofthe 41st International Conference on Machine Learning (Proceedings ofMachine Learning Research, Vol. 235). PMLR, 43445–43460. https://proceedings.mlr.press/v235/sardana24a.htm

[54] Selenium. n.d.. Selenium Overview. https://www.selenium.dev/documentation/ overview/ Oficial documentation accessed 2026-10-23.

[55] Shopify. n.d.. Build commerce agents with UCP. https://shopify.dev/docs/agents Documentation accessed 2026-10-21.

[56] Stripe. 2026. Model Context Protocol (MCP). https://docs.stripe.com/mcp Documentation accessed 2026-10-21.

[57] Tavily. n.d.. Tavily MCP Server. https://docs.tavily.com/documentation/mcp Documentation accessed 2026-10-21.

[58] Luis von Ahn, Manuel Blum, Nicholas J. Hopper, and John Langford. 2003. CAPTCHA: Using Hard AI Problems for Security. In Advances in Cryptology— EUROCRYPT 2003 (Lecture Notes in Computer Science, Vol. 2656). Springer, Berlin, Heidelberg, 294–311. doi:10.1007/3-540-39200-9\_18

[59] W3C Web Accessibility Initiative. n.d.. Tools and Techniques. https://www.w3. org/WAI/people-use-web/tools-techniques/ Oficial documentation accessed 2026-10-23.

[60] Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel R. Bowman. 2019. GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding. In The Seventh International Conference on Learning Representations. https://openreview.net/forum?id=rJ4km2R5t7

[61] Ethan Wang, Zubair Shafiq, and Yash Vekaria. 2026. Poster: FP-Agent: Towards Fingerprinting AI Browsing Agents. In Proceedings ofthe 2026 ACM Internet Measurement Conference (IMC ’26). ACM, 1227–1228. doi:10.1145/3777912.3847600

[62] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc V. Le, and Denny Zhou. 2022. Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. In Advances in Neural Information Processing Systems, Vol. 35. Curran Associates, Inc., 24824–24837. doi:10.52202/ 068431-1800

[63] Jason Weston, Antoine Bordes, Sumit Chopra, Alexander M. Rush, Bart van Merriënboer, Armand Joulin, and Tomas Mikolov. 2016. Towards AI-Complete Question Answering: A Set of Prerequisite Toy Tasks. In The Fourth International Conference on Learning Representations. https://arxiv.org/abs/1502.05698

[64] Wikidata contributors. n.d.. Wikidata:MCP. https://www.wikidata.org/wiki/ Wikidata:MCP Project documentation accessed 2026-10-21.

[65] Tianbao Xie, Danyang Zhang, Jixuan Chen, Xiaochuan Li, Siheng Zhao, Ruisheng Cao, TohJing Hua, Zhoujun Cheng, Dongchan Shin, Fangyu Lei, Yitao Liu, Yiheng Xu, Shuyan Zhou, Silvio Savarese, Caiming Xiong, Victor Zhong, and Tao Yu. 2024. OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments. In Advances in Neural Information Processing Systems, Vol. 37. Curran Associates, Inc., 52040–52094. doi:10.52202/079017-1650

[66] Minghui Xu, Xiaoyu Liu, Yihao Guo, Chunchi Liu, Yue Zhang, and Xiuzhen Cheng. 2026. AgentDID: Trustless Identity Authentication for AI Agents. In 2026 IEEE 46th International Conference on Distributed Computing Systems (ICDCS). IEEE, 1180–1190. doi:10.1109/2575-8411.2026.00116

[67] Shunyu Yao, Jefrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. 2023. ReAct: Synergizing Reasoning and Acting in Language Models. In The Eleventh International Conference on Learning Representations. https://openreview.net/forum?id=WE\_vluYUL-X

[68] Boyuan Zheng, Boyu Gou, Jihyung Kil, Huan Sun, and Yu Su. 2024. GPT-4V(ision) is a Generalist Web Agent, if Grounded. In Proceedings ofthe 41st International Conference on Machine Learning (Proceedings of Machine Learning Research, Vol. 235). PMLR, 61349–61385. https://proceedings.mlr.press/v235/zheng24e.html

[69] Shuyan Zhou, Frank F. Xu, Hao Zhu, Xuhui Zhou, Robert Lo, Abishek Srid har, Xianyi Cheng, Tianyue Ou, Yonatan Bisk, Daniel Fried, Uri Alon, and Graham Neubig. 2024. WebArena: A Realistic Web Environment for Building Autonomous Agents. In The Twelfth International Conference on Learning Representations. 15585–15606. https://proceedings.iclr.cc/paper\_files/paper/2024/ hash/4410c0711e9154a7a2d26f9b3816d1ef-Abstract-Conference.htm

## A A<sub>c</sub>ti<sub>ve</sub> V<sub>er</sub>ifi<sub>ca</sub>ti<sub>on</sub> D<sub>e</sub>t<sub>a</sub>il<sub>s</sub>

## A.1 Im<sub>p</sub>lementation and Ex<sub>p</sub>erimental Settin<sub>g</sub>s

Ofline library construction. We prepare the library L ofline using LLM-generated materials from 5 domains: biochemistry, clinical trials, cybersecurity, financial markets, and food safety. Domainspecific prompts guide the LLM to generate entity names grouped by type and narrative templates with placeholders for entities, attributes, and values. For each template, the LLM also supplies possible attribute values and questions paired with functions that compute their answers from the world state. We implement rule-based checks to ensure consistent entity types and placeholders and verify that each question refers to facts included in its report. The resulting library contains 208 entities, 40 narrative templates, and 291 questions. At runtime, the verifier constructs each challenge at minimal cost by sampling from this library and instantiating the selected template.

Challenge evaluation. We evaluate the models on 100 questions generated from 35 of the library’s 40 templates, with 20 questions from each domain. Of these questions, 60 ask for an attribute value, 20 compare two entities, and 20 ask for the entity with the highest or lowest value. All model queries go through OpenRouter [49] with the same structured-answer requirement and no model-specific sampling or reasoning settings. The verifier extracts the first submitted answer and normalizes Unicode, case, and whitespace without converting numbers or units. We report accuracy and latency over all 100 assigned questions, counting incorrect, empty, and filtered responses as failures. Each question contributes one selected response to the latency and token statistics. During evaluation, Claude did not answer 2 food-safety questions because of safety filtering. We count both as incorrect and retain their latency, while calculating Claude’s mean token usage from the 98 responses with reported usage.

Server overhead. We measure challenge assembly and answer checking on the same 100 questions using Python 3.9.6 on an Apple M5 running macOS arm64. The saved correct answers serve as inputs to the verifier. Each of the 3 independent Python processes loads resources and completes a warmup before timing. Batching calls reduces the overhead of clock and hardware-counter readings. For each question, we use the median per-call cost from the 3 runs; Table 3 summarizes these 100 estimates. To obtain combined overhead, we add each question’s assembly and verification costs within the same run before taking medians. The measurements exclude ofline library preparation, resource loading, serialization, and HTTP serving. Instruction counts include the Python interpreter and operating-system work. Challenge and private-state sizes are measured using compact UTF-8 JSON serialization.

## A<sub>.</sub>2 W<sub>or</sub>k<sub>e</sub>d Ch<sub>a</sub>ll<sub>enge</sub> E<sub>xamp</sub>l<sub>e</sub>

We illustrate the construction with a biochemistry challenge from the evaluation set. It asks which of two receptors has a higher IC50 value in a 624-word report. The box below reproduces the complete report, question, and answer instruction received by the client. Table 7 shows the relevant values and how the verifier obtains and checks the answer.

## P<sub>u</sub>bli<sub>c</sub> <sub>c</sub>h<sub>a</sub>ll<sub>enge</sub>

## <sup>R</sup>e<sub>p</sub>ort.

## ASSAY SCREENING PANEL SUMMARY

Panel reference: BIO-SCR-6; plate map revision 4.2; acquisition window 18 min per plate. The following narrative integrates inhibition curves, receptor displacement traces, and enzyme turnover checks for 6 entries. Bufer was 50 mM HEPES with 150 mM NaCl, pH 7.4, and the initial robotic dispense log also recorded fixed controls of 0.5% DMSO, 2.0 mM MgCl2, and 37 C incubation. Compound Erlotinib was logged with replicate CV 16.3% in the first potency table, while compound Metformin was entered nearby with replicate CV 12.8%. No preference language is used here; these are simply panel entries.

The preliminary readout appeared to place the receptor curves in a narrow band. Receptor GABA-A showed Hill slope 1.58, and receptor CXCR4 showed Hill slope 0.74, with a fixed reference tracer displacement of 68% at 100 nM. On that basis, the run captain initially wrote that receptor behavior looked essentially interchangeable. That early sentence was later softened after the kinetic annotations and replicate summaries were reconciled across the document, because slope, dispersion, and assigned mode did not remain parallel.

For the enzyme block, enzyme LDH-� had IC50 0.72 uM after blank subtraction, and enzyme PFK-1 had IC50 64.5 uM. Their turnover checks used 12 substrate points, with a fixed calibration standard at 40.0 nmol/min/mg and a carryover flag below 0.3%. Enzyme LDH-� was separately assigned binding mode noncompetitive; enzyme PFK-1 was assigned binding mode uncompetitive. These annotations refer to the fitted inhibition pattern, not to suitability or priority.

Compound records were revisited after the enzyme plates were thawed a second time. Compound Erlotinib then had Hill slope 1.03 and IC50 0.31 uM; compound Metformin had Hill slope 1.76 and IC50 4.90 uM. A fixed distractor value of 91% activity recovery appeared in the washout column for a pooled control and should not be confused with any sampled attribute. The mode calls were kept in a separate column: Erlotinib carried binding mode covalent reversible, while Metformin carried binding mode partial agonist.

Receptor annotations were then cross-checked against the same reporting vocabulary used for the enzyme section. Receptor GABA-A carried replicate CV 9.4% and IC50 0.14 uM. Receptor CXCR4 carried replicate CV 25.5% and IC50 8.35 uM. The fixed scintillation background was 145 cpm, and the ligand concentration was held at 7.5 nM. Receptor GABA-A was annotated as competitive for binding mode; receptor CXCR4 was annotated as inverse agonist for the same label.

The earlier impression of receptor interchangeability was not retained in the final notes. The diference was not attributed

## Public challenge (continued)

to a single number, because the final read used the full set of replicate CV, slope, dispersion, and categorical mode calls. In the same reconciliation pass, enzyme LDH-� was also listed with replicate CV 5.6% and Hill slope 0.96; enzyme PFK-1 was listed with replicate CV 1.4% and Hill slope 0.62. A fixed note of 24 h storage at 4 C applied to all plates and is not an entity-specific measurement.

A final pass scattered the compound and receptor entries back into the same spreadsheet view so that no row looked privileged. Compound Erlotinib was mentioned again beside a pooled Zprime of 0.72, and compound Metformin beside a pooled Zprime of 0.69; neither number is the sampled IC50. Receptor GABA-A and receptor CXCR4 were also repeated in the reviewer margin to remind readers that categorical calls and quantitative values both contributed to the readout.

The closing assessment states only that all 6 entries were reviewed after curve fitting, replicate inspection, and nomenclature harmonization. Enzyme LDH-�, enzyme PFK-1, compound Erlotinib, compound Metformin, receptor GABA-A, and receptor CXCR4 remain neutral records in the panel. The preliminary receptor sentence should be treated as provisional, while the scattered final values above are the values to use for lookup, ranking, and comparison.

## Question.

Between GABA-A and CXCR4, which had a higher IC50?

## A<sub>nswer</sub> i<sub>ns</sub>t<sub>ruc</sub>ti<sub>on.</sub>

For extrema, compare only entities of the requested type for which this report explicitly gives the requested attribute; missing values are not zero. Return one JSON object with exactly one key, answer, whose value is a string. Put only the exact entity name for a which-question, or the reported value for a what-question, inside that string. A numeric answer may omit its unit, but must keep the number exactly as reported. Do not convert units, expand unit names, or rewrite the numeric notation. Do not add a type prefix, explanation, Markdown, or any other field.

T<sub>a</sub>bl<sub>e</sub> 7<sub>:</sub> E<sub>xamp</sub>l<sub>e ver</sub>ifi<sub>ca</sub>ti<sub>on.</sub> Th<sub>e ver</sub>ifi<sub>er compares</sub> t<sub>wo</sub> IC50 <sub>va</sub>l<sub>ues</sub> t<sub>o o</sub>bt<sub>a</sub>i<sub>n</sub> th<sub>e correc</sub>t <sub>answer an</sub>d <sub>c</sub>h<sub>ec</sub>k th<sub>e</sub> <sub>c</sub>li<sub>en</sub>t’<sub>s response.</sub>
<table><tr><td>Component</td><td>Content</td></tr><tr><td></td><td>Relevant values GABA-A: 0.14 uM; CXCR4: 8.35 uM</td></tr><tr><td>Answer rule</td><td>Select the receptor with the higher IC50 value.</td></tr><tr><td>Correct answer CXCR4</td><td></td></tr><tr><td>Example</td><td>{&quot;answer&quot;:&quot;CXCR4&quot;}</td></tr><tr><td>response</td><td></td></tr></table>

The correct answer is CXCR4, whose IC50 value is 8.35 uM, compared with 0.14 uM for GABA-A. The verifier reads these values directly from its private world state, whereas the client receives them only as part of the natural-language report.

Table 8: Service-visible feature <sub>g</sub>rou<sub>p</sub>s. We use 40 features from 5 <sub>g</sub>rou<sub>p</sub>s for <sub>p</sub>assive client classification.
<table><tr><td>Feature group</td><td>Count</td><td>Measurements</td><td>Available visitsª 8 Human (72)Bot (360) Agent (216)</td><td></td></tr><tr><td>Browser (xB)</td><td>12</td><td colspan="3">Inner/outer window dimensions; total/available screen dimensions; pixel ratio, color depth, maximum touch points, and webdriver flag.</td></tr><tr><td>Activity (xA)</td><td>5</td><td colspan="3">Application-request count and peak concurrency; click (including auxiliary click), input, and scroll counts.</td></tr><tr><td>Input (xI)</td><td>9</td><td colspan="3">Pointermove count, moves per click, and fraction of clicks preceded by movement; focus, paste, and composition counts; paste, composition, and deletion fractions among inputs with reported types.</td></tr><tr><td>Sequence (xs)</td><td>7</td><td colspan="3">Runs and adjacent-repeat fraction of click/input/scroll events; click→input, input→click, click→scroll, and</td></tr><tr><td>Timing (xT)</td><td>7</td><td colspan="3">scroll→click fractions; normalized transition entropy. Application-request span and median request gap; click/input/scroll span; within-document gap median, P95, and coefficient of variation; fraction of gaps above one second.</td></tr></table>

<sup>a</sup> Visits: Of the 720 study visits, 648 have browser observations; the remaining 72 are from the HTTP-only Bot.

Table 9: Sho<sub>pp</sub>in<sub>g</sub> tasks. All tasks start from the sho<sub>pp</sub>in<sub>g</sub> h<sub>omepage</sub> <sub>an</sub>d <sub>en</sub>d <sub>w</sub>ith <sub>car</sub>t <sub>rev</sub>i<sub>ew</sub> <sub>w</sub>ith<sub>ou</sub>t <sub>c</sub>h<sub>ec</sub>k<sub>ou</sub>t<sub>.</sub>
<table><tr><td>Task</td><td>Cart requirements</td></tr><tr><td>Wi-Fi adapters</td><td>One specified Foktech AC1200 adapter and one specified EDUP dual-band adapter with an extender antenna.</td></tr><tr><td>8 USB drive</td><td>One USB flash drive with at least 32 GB capacity and a price at most $22.</td></tr><tr><td>Wireless mice</td><td>Two different wireless mice, one of each, costing at most $20 each and $30 combined; at least one must support Bluetooth.</td></tr></table>

## B P<sub>ass</sub>i<sub>ve</sub> A<sub>na</sub>l<sub>ys</sub>i<sub>s</sub> D<sub>e</sub>t<sub>a</sub>il<sub>s</sub>

## B.1 Ex<sub>p</sub>erimental Settin<sub>g</sub>s

We use the 3 shopping tasks summarized in Table 9. The Wi-Fi adapter task specifies the exact products to add to the cart, while the other 2 allow clients to select products satisfying the stated constraints.

Browser Use and MCP Agents call openai/gpt-6.1-sol via OpenRouter [49], with the MCP Agent controlled through OpenAI Agents SDK. The Computer Use Agent accesses the same model family through Codex with a ChatGPT subscription and operates through screenshots and GUI actions. For each Computer Use visit, we use a separate conversation and Chrome profile. The native-input Bot reads public DOM information for positioning and executes macOS mouse and keyboard events.

Each visit uses an independent guest shopping session. Automatic clients use new browser profiles or Cookie jars, while Human participants use their own browsers and may retain resource caches. For HTTP summaries, we retain application-page and API requests, including repetitions and failures, and exclude static resources, telemetry, and grading. The grader checks the final cart after observation closes. The analyzed dataset retains visits when clients do not complete the shopping tasks or their carts do not satisfy the task requirements. For 5 of the 72 Human visits, event recording was unexpectedly incomplete; we use the available observations and treat unavailable fields as missing. We exclude the HTTP-only Requests Bot from page-based classification.

To construct features, we apply log(1 + �) to HTTP measurements, event counts, spans, and gap quantiles, keeping other features on their original scale. Timing and sequence measurements use click, input, and scroll events, with adjacency reset at document boundaries. The collector records pointer-move and scroll events at most once every 100 ms each. Within each training split, we estimate preprocessing parameters and class weights, filling missing fields with the observed means without adding missingness indicators. LR and RBF-SVM also use standardization fitted on that split. The predictors exclude identifiers and proxy-side connection counts, along with model-side logs and task grades.

Within each outer training set, we alternate the 2 tasks between inner training and validation, using P01–P04 for Human training visits and P05–P08 for validation visits. We select among 6 prespecified candidates per classifier and feature setting by mean innervalidation macro-F1, breaking ties by candidate order, and then refit the selected classifier on all outer training data. To supplement the interaction records, we measure TLS/HTTP fingerprints before proxy rewriting using 33 separate HTTPS probes: 3 per automatic configuration and 3 each for ordinary Chrome and Safari. In Table 5, these reference browsers supply the Human fingerprints, while the 72 participant visits provide the Human HTTP statistics.

## B<sub>.</sub>2 Ethi<sub>ca</sub>l C<sub>ons</sub>id<sub>era</sub>ti<sub>ons</sub>

We obtained institutional review board (IRB) approval for the user study and informed all participants about the study and its data collection. We collected only task-related HTTP metadata, browser characteristics, and page events within the experimental shopping website and did not collect browsing activity outside the site or any personally identifiable information.