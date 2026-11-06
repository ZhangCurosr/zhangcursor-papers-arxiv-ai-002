# Responsible GenAI Use During Active Armed Conflict: Evidence from Sudan’s Emergency Response Rooms

DALIA ALI<sup>∗</sup>, Technical University of Munich, Germany

YEARIN LEE<sup>∗</sup>, Technical University of Munich, Germany

CHENG YU, Technical University of Munich, Germany

RO ENCARNACIÓN, University of Pennsylvania, USA

ALSANOSI ADAM, Emergency Response Rooms, Sudan

DANAÉ METAXA, University of Pennsylvania, USA

INGMAR WEBER, Saarland University, Germany

ORESTIS PAPAKYRIAKOPOULOS, Technical University of Munich, Germany

Armed conflict can increase the value of Generative AI (GenAI) while undermining the conditions required for its responsible use. Yet the perspectives of community-based humanitarian responders working under these conditions remain underrepresented in responsible AI research. We examine GenAI use in Sudan’s Emergency Response Rooms (ERRs), a community-led humanitarian network, through an exploratory survey of 89 respondents and 15 semi-structured interviews with GenAI users and non-users. War reconfigures GenAI’s role, making it most useful where its outputs are hardest to verify. ERR responders turned to GenAI when conflict made human expertise dificult or dangerous to reach. However, participants cautioned that GenAI could not independently establish rapidly changing local conditions, and that relying on its outputs could lead to harmful decisions when human verification was dificult. Risks arose before GenAI was used. For some volunteers, carrying a phone or accessing the internet could expose them to danger. Others were concerned that sharing sensitive beneficiary or operational information with GenAI could put people at risk Participants feared that AI-assisted reports could undermine their credibility with donors, whose funding sustains the food and medical aid ERRs provide. Access to GenAI was uneven, and volunteers using it sometimes relied on colleagues with relevant expertise or local knowledge to verify its results. Our findings show that responsible use of GenAI during conflict depends not only on model performance or individual adoption, but also on whether responders can safely access these tools, verify their outputs against local conditions, and recognize when limiting or avoiding GenAI is the more responsible choice.

CCS Concepts: • Human-centered computing → Empirical studies in HCI; Collaborative and social computing; • Social and professional topics → Computing / technology policy; • Computing methodologies → Artificial intelligence.

Additional Key Words and Phrases: Generative AI, Armed Conflict, Responsible AI, Community-Based Humanitarianism

## 1 Introduction

Armed conflict afects civilians by causing harm, death, displacement, and limiting access to basic needs such as food, water, and health care. Its efects are widespread; in 2025, 117.8 million people were forcibly displaced, including 68.7 million displaced within their countries [24, 43]. In Sudan, since fighting started in April 2023, 9.3 million people have been displaced within the country, and 4.3 million have crossed into neighboring countries. In 2026, 33.7 million people there require urgent humanitarian assistance, and 37% of health facilities are non-functional [48]. International humanitarian organizations have faced access and capacity constraints, increasing the importance of locally led mutual aid like Emergency Response Rooms (ERRs), a community-led volunteer network that provides humanitarian assistance during the war in Sudan [15]. The term “Rooms” refers to the WhatsApp group chats through which volunteers initially organized their emergency response activities [31].

Under such conditions, Generative AI (GenAI), particularly large language models (LLMs), may provide a readily accessible source of information and assistance when human expertise or organizational support is dificult to reach. Previous research has documented the use of AI for crisis forecasting and resource allocation in humanitarian settings [10]. Similarly, practitioners in humanitarian, development, and environmental organizations report using LLM-based tools, such as ChatGPT and Copilot, for drafting and summarization [4]. LLMs have also been used for humanitarian situation reports, which summarise the scope, impact, immediate needs, and ongoing relief eforts during crises, by synthesizing humanitarian documents into structured reports [16].

The use of digital technologies in humanitarian work can expand operational capabilities while introducing new operational, ethical, and political challenges [35]. Increased access to information does not automatically mean a commensurate ability to process and make use of it [34]. In addition to these challenges, collecting and using digital data can itself create risks for populations afected by violence and insecurity [36]. These tensions are particularly acute during armed conflict, where restrictions on humanitarian access can increase reliance on local responders who mus operate in dificult and unsafe conditions [7].

Recent work has examined AI adoption, organizational readiness, and governance challenges in humanitarian contexts [6, 10], as well as AI use in humanitarian interventions [37] and LLM-based situation reporting [16]. However, much of this work focuses on formal humanitarian organizations, humanitarian professionals and experts, or systems introduced for specific organizational purposes. There is far less empirical knowledge about community-based respon ders’ use of, limitations on, or avoidance of GenAI in their daily work during active armed conflicts. This gap matters because self-organized networks like Sudan’s ERRs may lack the institutional protection and support available to larger humanitarian organizations. ERR volunteers also face disrupted infrastructure, limited access to expertise, and direct threats to their safety and the people they assist [5, 22, 31]. For the purposes of this study, we define responsible GenAI use as that which considers risks to responders and afected communities, protects sensitive information, and ensures appropriate human verification before outputs inform consequential decisions. We therefore ask:

## RQ1: How do community-based humanitarian responders use and/or avoid GenAI during active conflict? RQ2: How do wartime conditions shape GenAI accessibility, usefulness, and responsible use?

To answer these questions, we study Sudan’s ERRs, which rely on locally organized volunteers who work under varying infrastructural and security conditions, often without the institutional support available to formal humanitarian organizations. This network includes more than 700 ERRs across Sudan and over 26,000 responders [1]. We first conducted an exploratory survey with 89 ERR respondents to identify patterns of GenAI use and non-use, followed by 15 semi-structured interviews with ERR responders to examine these patterns in greater depth.

Our findings show that GenAI use in Sudan’s ERRs is not a straightforward case of adoption or non-adoption. Some ERR responders turned to GenAI when conflict made human assistance dificult to reach, but they also recognized its limitations in interpreting rapidly changing conditions on the ground. Safety concerns shaped both whether GenAI could be used and what information could be shared with it, while GenAI-assisted humanitarian communication introduced questions about accuracy, representation, and credibility. Moreover, access to GenAI and responsibility for reviewing its outputs were distributed across responders operating under unequal infrastructural and security conditions. In total, we contribute the following:

(1) Empirical Evidence on GenAI Use During Active Armed Conflict. We provide new empirical evidence on the ways community-based humanitarian responders use, limit, and avoid GenAI during active conflict, bringing the perspectives of volunteers delivering essential humanitarian assistance under direct security threats, disrupted infrastructure, and limited access to expertise into responsible AI research.

(2) Conflict-Specific Conditions of Responsible GenAI Use. We show that active conflict can simultaneously increase the value of GenAI and undermine the conditions required for its responsible use, a tension starkest in life-or-death situations. For example, a responder unable to safely access medical care might find great value in GenAI assistance, but simultaneously be put at risk by carrying the phone needed to access it. Responsible GenAI use therefore depends not only on the model reliability and individual use, but also on situated human knowledge, task-specific boundaries, and how responders coordinate access to GenAI and review its outputs.

(3) Conflict-Sensitive Design and Governance of GenAI. We derive implications for humanitarian organizations, donors, and technology providers that address safety risks before and during GenAI use, preserve human judgment and the legitimacy of non-use, and safeguard the credibility and accountability of AI-assisted humanitarian work.

## 2 Related Work

This section reviews current applications of AI and GenAI in humanitarian response. It also addresses challenges associated with GenAI use in high-stakes contexts, identifying gaps in existing work that our study aims to address.

## 2.1 Use of AI and GenAI in Humanitarian Response

Armed conflicts, natural disasters, and political pressures disrupt the delivery of essential humanitarian services to afected communities [3, 11, 38]. Under such conditions, AI has demonstrated its utility across diferent phases of crisis response [27]. Before crises occur or escalate, AI can support early warning and anticipatory planning. AI can predict some crises using satellite imagery, household surveys, mobile phone records, and social media data [20]. For instance, HungerMap LIVE uses machine learning (ML) to monitor food insecurity in near-real time across more than 50 countries and provide actionable insights into where food insecurity is most severe [44, 47].

During crises, AI has also been used to support situational awareness, emergency coordination, and damage assessment in afected areas. For example, the Qatar Computing Research Institute developed AIDR and tested it during the 2013 earthquake in Pakistan, classifying disaster-related social media messages using ML and providing actionable information in real time [23]. Similarly, World Food Programme’s SKAI tool, developed with Google Research, uses ML to detect damaged infrastructure from satellite imagery, with reported uses during Hurricane Ian and the Pakistan flood in 2022, and the Türkiye earthquake in 2023 [19, 25]. Recent studies have explored the potential of LLMs to analyze and synthesize disaster-related information and generate reports from the data for emergency response planning and coordination [49]. AI can also contribute to post-crisis recovery eforts by facilitating rapid damage assessments and identifying priority areas for recovery. The United Nations Development Programme’s RAPIDA tool has been applied for this purpose in response to the 2026 earthquakes in Venezuela and cyclones in Madagascar [33].

However, these applications are often dependent on technical infrastructure, expertise, and organizational resources that may be challenging for decentralized local humanitarian organizations to secure. Commercially available GenAI tools like ChatGPT could provide an alternative source of necessary support, since humanitarian responders can more easily access them, without developing dedicated AI systems. Such tools may, however, pose significant risks, as they are not explicitly designed for conflict environments. Limited research has examined how local humanitarian responders use, limit, or avoid GenAI in their work, particularly during active conflict, or documented these practices in their own voices. We examine these practices among ERR volunteers in Sudan, providing empirical information from community-based responders whose experiences are underrepresented in responsible AI research.

## 2.2 Sociotechnical Risks of GenAI in High-Stakes Contexts

The use of GenAI in humanitarian response requires considerations beyond practical challenges, as its limitations can lead to real-world harm. One concern is bias, which can emerge from unrepresentative training data and be reinforced during model inference, potentially disadvantaging certain populations [2]. Many language technologies have also been shown to perform unevenly across languages or fail to represent linguistic and cultural specificities [9]. Privacy presents another challenge, as LLMs can memorize and reproduce personal information from training data [14], which can pose considerable security risks in sensitive contexts. LLMs can also infer sensitive personal attributes such as location and income from contextual details, even when explicit identifiers have been removed [39].

Another major concern is hallucination, which occurs when LLMs generate factually conflicting or unverifiable information or produce responses that are inconsistent with user instructions, provided context, or their own reasoning [21]. For example, news outlet CNN reported that an AI-assisted US military intelligence report falsely identified a Chinese vessel as transporting nuclear weapons components, prompting preparations for a military interception before the error was detected [26]. GenAI can also be misused to produce misinformation and disinformation, particularly through deepfakes [42]. For example, during Russia’s invasion of Ukraine in 2022, a deepfake video falsely depicted Ukrainian President Volodymyr Zelensky calling on Ukrainians to surrender to Russia [45]

Dificulties in accurately assessing conditions on the ground can further complicate efective humanitarian response. For example, a review of AI-based food insecurity prediction identifies the lack of ground-truth data in conflict settings as a barrier to accurately assessing food insecurity [20]. Another study found substantial diferences in how human annotators labeled damage to the same buildings when using drone versus satellite imagery and raised concerns about potential risks of AI damage assessment systems trained on such imagery-derived labels [28]. Related reliability challenges have also been observed in GenAI. Research has shown that LLMs often perform poorly in some low-resource languages used in crisis settings and that an LLM assigned markedly diferent urgency levels to the same simulated crisis scenarios presented in diferent languages [41]. While these studies discuss important limitations in data quality and model performance, less attention has been paid to how humanitarian responders encounter and manage them in practice. We address this gap by examining how ERR volunteers in Sudan perceive GenAI’s risks, verify its outputs, and decide when to limit or avoid its use during active conflict.

## 3 Method

The study comprised a two-stage design. We first conducted an exploratory survey to develop an initial understanding of GenAI use and non-use among ERR responders and to identify issues for deeper investigation. We then conducted semi-structured interviews to examine these issues and participants’ experiences in greater depth.

## 3.1 Ethics and Participant Safety

Given the sensitivity of conducting research with humanitarian responders during active armed conflict, approaching this work with extreme care and appropriate positionality was crucial. Two members of the research team are part of the ERR network and are familiar with the Sudanese humanitarian context. Their background knowledge and existing relationships within the network informed the study design, participant recruitment, and data interpretation.

We obtained Institutional Review Board (IRB) approval and took additional steps to address participant safety, confidentiality, and barriers to participation in the context of the ongoing war. Specifically, participants were encouraged to avoid providing names, precise locations, or operationally sensitive information, and we removed identifying details during data analysis and reporting. To reduce barriers to participation, we ofered to cover participants’ internet costs.

After conducting the interviews and before processing the data, we assessed whether combinations of location, role, and experience could identify participants within the ERR network or expose them to harm, and generalized details where necessary. Interview recordings, transcripts, and survey data were stored privately, with access limited to researchers involved in the analysis. For the LLM-assisted translation and coding, which participants were informed about verbally before recording, we used GPT-6 through a ChatGPT Pro account provided by the research group [32] (details in Appendix B). We only provided de-identified documents to the model, after removing or generalizing information that could directly or indirectly identify participants; we disabled model-improvement data sharing before submitting any content to ChatGPT

Participants could choose not to answer particular questions or stop the interview at any time, as well as request withdrawal from the study after the interviews were conducted. Moreover, because interviews might include discussions of events experienced in the context of conflict that could cause emotional distress, the study protocol also included access to mental-health support or referral information if participants requested it (although none did)

## 3.2 Stage 1: Exploratory Survey

We started by conducting an exploratory survey with ERR responders to identify issues for deeper investigation in the interviews. The survey was designed and translated into Arabic by a native Arabic-speaking researcher and was open to ERR responders aged 18 years or older. Survey respondents were recruited through network-based and participant-referral sampling. A research team member who is part of the ERR network initially distributed the survey to existing ERR contacts, who subsequently shared it with other volunteers. In some cases, responders with internet access also reached those without connectivity and provided access to the survey so that they could complete it in person.

The survey focused on the roles and humanitarian activities of the participants, their experience with the ERRs, their use and non-use of GenAI, the tools they use, the tasks they perform, the languages and devices used, the reasons why they do not use GenAI, and how they verify AI outputs. Moreover, there were 23 Likert-scale questions on technological, organizational, and environmental aspects, including access to devices and the internet, organizational support and training, privacy and security issues, and the necessity of human verification.

The final survey sample consisted of 89 unique respondents after removing duplicate submissions with identical response information. The survey was exploratory and was not intended to provide representative estimates of GenAI use across ERRs. We used the survey responses descriptively to characterize the sample and identify initial patterns that informed the design of the semi-structured interview guide, including questions about output verification, sensitive information, and internet access. The full questionnaire and further descriptive statistics are provided in Appendix A.

## 3.3 Stage 2: Semi-Structured Interviews

Following the exploratory survey, we invited some respondents who had agreed to be contacted for follow-up interviews, which were conducted between August and September 2026. Among respondents who agreed to be contacted and provided contact information, we purposively selected both GenAI users and non-users. We also sought variation in geographic areas and roles within the ERRs, including participants working in areas under the control of diferent armed actors. We conducted 15 semi-structured online interviews with ERR volunteers, each lasting approximately 45–60 minutes. The final sample included nine participants who had used GenAI in their ERR work and six who had not. We allowed participants to choose the language they were most comfortable with. We therefore conducted 14 interviews in Arabic and one in English.

For GenAI users, we asked about their experiences with GenAI, including the circumstances that led them to use it, the data entered, how outputs were used and checked, and situations where participants chose not to use GenAI or limited its use. For non-users, we explored indirect exposure to GenAI, reasons for non-use, existing alternatives, and the conditions under which they might consider using it. Both paths examined how the ongoing war afected access, safety, and decisions about appropriate use.

The guide included a sector-specific question based on each participant’s work, such as health, food distribution, protection, logistics, information management, or donor relations. These questions examined whether GenAI was used in potentially high-risk activities, what participants considered appropriate or inappropriate, and where human decision-making was important. With participants’ permission, interviews were recorded for transcription. Participants using GenAI could share their screen to demonstrate their workflow, but were instructed not to display identifiable, confidential, or operationally sensitive information. Interview guides are provided in Appendix B.

## 3.4 Data Analysis

We used the survey responses descriptively to characterize the sample and identify initial patterns to inform the subsequent interviews. A native Arabic-speaking author transcribed the 14 Arabic interviews and translated them into English with AI assistance, manually reviewing the machine translations against the Arabic originals.

We analyzed the interview data using a codebook approach to thematic analysis [12, 13]. Our coding was primarily inductive, with codes developed from participants’ responses instead of from a predefined theoretical codebook. We first read the transcripts and coded relevant segments in QCAmap [17], a qualitative data analysis tool; then we used GPT-6 to generate an additional set of candidate codes and possible patterns for researcher review. We did not treat the LLM as an independent coder, and we checked all suggested codes and patterns against the interview transcripts and our own judgment. As coding progressed, we compared both sets of codes with the transcripts, compared similar codes, revised their definitions, and removed unsupported or redundant codes.

We then conducted a collaborative analysis workshop involving four members of the research team to map related codes and develop higher-level themes. During the workshop, we compared GenAI users and non-users and examined patterns across diferent forms of humanitarian work. We grouped related codes, discussed diferences and recurring patterns, and returned to the transcripts to check emerging themes against the interview data. More details on coding and thematic analysis are in Appendix B.

## 4 Results

Of the 89 respondents, 49 (55.1%) reported using GenAI in their ERR work, 30 (33.7%) reported not using it, and 10 (11.2%) were unsure. Among respondents, 65.2% agreed that GenAI could be useful in fast-paced ERR work, while 70.8% agreed that its outputs usually required verification, and only 41.6% agreed that GenAI-generated information was usually accurate. Half of respondents (50.6%) expressed concern about exposing sensitive information and only 44.9% reported suficiently stable internet access. These patterns suggested that perceived usefulness of GenAI coexisted with concerns about its reliability, information protection, and accessibility. Additional survey results are in Appendix A.

Table 1. Overview of semi-structured interview participants $( N = 1 5 )$
<table><tr><td>ID</td><td>Location</td><td>Working area</td><td>Role / main focus</td></tr><tr><td colspan="4">GenAI users  $( n = 9 )$ </td></tr><tr><td>P01</td><td>National</td><td>External communications / advocacy</td><td>External liaison and advocacy</td></tr><tr><td>P02</td><td>Red Sea</td><td>Finance</td><td>Financial reporting, transfers, and suppliers</td></tr><tr><td>P03</td><td>Khartoum State</td><td>Monitoring and evaluation</td><td>Implementation and quality monitoring</td></tr><tr><td>P04</td><td>Khartoum State</td><td>External relations</td><td>Fundraising, partnerships, and advocacy</td></tr><tr><td>P05</td><td>Gezira / National</td><td>Programs coordination</td><td>Multi-level program coordination</td></tr><tr><td>P06</td><td>Khartoum / National</td><td>Protection / gender</td><td>Women&#x27;s programming and safe spaces</td></tr><tr><td>P07</td><td>Khartoum</td><td>Logistics / food assistance</td><td>Community kitchens, procurement, and supplies</td></tr><tr><td>P08</td><td>Khartoum</td><td>Health / medical coordination</td><td>Medical coordination and health projects</td></tr><tr><td>P09</td><td>South Darfur</td><td>Programs / project management</td><td>Needs assessment and project implementation</td></tr><tr><td colspan="4">GenAI non-users (n = 6)</td></tr><tr><td>P10</td><td>West Darfur</td><td>Programs / project management</td><td>Planning, implementation, and reporting</td></tr><tr><td>P11</td><td>National</td><td>Communications / advocacy</td><td>Field information, donor liaison, and advocacy</td></tr><tr><td>P12</td><td>Khartoum</td><td>Emergency / medical relief</td><td>Hospital support and emergency medical relief</td></tr><tr><td>P13</td><td>River Nile</td><td>Community relief</td><td>Kitchens, distributions, and basic services</td></tr><tr><td>P14</td><td>South Kordofan</td><td>Communications / field coordination</td><td>Donor, volunteer, and distribution coordination</td></tr><tr><td>P15</td><td>North Darfur</td><td>Programs / needs assessment</td><td>Project planning and community coordination</td></tr></table>

Note. “National” refers to ERR work spanning multiple states. Combined labels such as “Khartoum/National” indicate work based in the named state with additional national-level responsibilities. Location and role descriptors were reviewed by the research team, including a researcher who is part of the ERR network, and by an additional person familiar with the ERR network, to assess whether combinations could identify participants or expose them to harm.

## 4.1 Turning to GenAI When War Makes Human Support Hard to Reach

GenAI became useful for some ERR responders because active fighting had made ordinary routes to human assistance unsafe, slow, or unavailable. Participants described scenarios in which consulting a doctor, reaching a specialist, or returning to a site with the right equipment was no longer straightforward. P07 illustrated this with a hypothetical situation in which health emergencies could arise among people sheltering during the fighting:

“One of the girls might have a seizure. You do not have a doctor at hand, and taking her to a doctor would itself put her life at risk. You have to turn to [GenAI]. [...] If no one around you can help solve the problem, the quickest thing is to use [GenAI].” (P07)

Beyond food, housing, and medical care, P07 described psychological needs among people sheltering during the fighting, like some people becoming so frightened that they refused to eat or drink, and children being exposed to “gunfire or the sound of gunfire.” P07 used GenAI chatbots to generate suggestions for things they could do in such moments, such as activities for children and ways to move people away from places where the sounds of fighting were most distressing.

A similar pattern appeared in a technical assignment; P03 recounted how Gemini helped him during the analysis of a solar energy system installed at a dialysis center, including assessing the risk of shadows on the solar panels based on photographs and the position of the sun. In the moment, he was faced with an unforeseen challenge, lacked the proper equipment, and was uncertain about his ability to revisit the site later due to the active war: “First, I did not have a measuring tape. Second, there was nobody with me, and I did not think I would be able to go back again, so I used Gemini straight away.” (P03)

P03 treated Gemini’s output as an initial assessment and discussed it with the engineer responsible for the installation, who said he had already accounted for the issue and would address it. In this case, GenAI provided provisional assistance rather than replacing human expertise. While P07 described why immediate GenAI assistance could be valuable when reaching professional help was delayed or impossible, P03’s experience demonstrates how the need for additional information could prompt its use even in the presence of a human expert. These accounts show how wartime conditions could move GenAI into situations where responders would otherwise seek human support.

## 4.2 GenAI Could Not Independently Establish Wartime Ground Truth

Participants distinguished between information that they were confident GenAI could process and that which it could not, including the current circumstances needed to make humanitarian decisions. During active war, circumstances could quickly change after information had been collected or entered into the system. Prices, currency exchange rates, weather, population migration, and resources could change in days. Such fluctuations were highlighted by P04 when talking about using GenAI in project planning:

“Today, you might design a project that costs 600, 800, or 1,000 dollars. Within a week, by the time

you implement it, things have changed. The market is not stable; the exchange rate is not stable [...]

[GenAI] only gives you indicators that you can build on. But in the end, the final decision should be

made by the human who is on the ground and sees how things are actually going.” (P04)

The need for up-to-date knowledge of the locality also influenced how willing the participants were to use GenAI. P08 utilized GenAI for organizing, but set limits to its usage in initiating projects or determining what the community needs. Project development instead began with information obtained directly from volunteers and afected communities When asked why she would not use AI to develop the core idea for a project, she was direct: “AI does not know the actual situation. Instead, we turn to the volunteers themselves.” (P08) For her, GenAI could help organize what responders already knew, but it could not replace the local knowledge used to decide what should be done.

P11, a non-user, made a similar distinction between information available to a model and knowledge attained from living and working in the afected environment. He described local problem-solving as an iterative process in which responders tried an approach and adjusted it as conditions like weather, materials, or community needs changed. He argued that online information could not capture people’s daily struggles or the repeated adjustments involved in responding in times of war:

“[GenAI tools] don’t know about the struggle, the daily life of the people. What do you know about what the need is and how to respond to it? The people know, right? The people on the ground. It’s not a one-question, one-answer situation.” (P11)

These cases show diferent limits to relying on GenAI-generated information during the war. P04 emphasized that GenAI-generated cost estimates could become outdated as market prices and exchange rates changed, afecting project planning and implementation. P08 and P11 pointed to a diferent limitation, as identifying community needs and adapting humanitarian responses required knowledge developed through direct engagement with afected people and changing circumstances. Notably, this concern appeared in both P08’s selective use of GenAI and P11’s non-use. For these participants, evaluating whether GenAI-generated information could inform humanitarian decisions required more than assessing the output itself; it also depended on current knowledge of the situation on the ground. Consistent with this, only 49.4% of survey respondents agreed that GenAI understands local words, places, and context in Sudan.

## 4.3 Wartime Safety Concerns Shape GenAI Use and Information Disclosure

For some responders, the safety concerns related to technology use impacted whether GenAI could be used at all. P12 described the period during which he was operating on the front lines, got arrested, and often either hid or discarded his phone: “I always left my phone behind at the hospital or hid it somewhere. Even when I was arrested, thank God, they did not find my phone with me. At that time, I was not thinking about AI.” (P12) Later, P12 described that in this period “having a phone was itself a crime”. Armed actors could search phones and interpret messages, search histories, or social media activity as support for the opposing side, potentially exposing their owners to detention or death.

Among participants who did use GenAI, the same safety issues determined what kind of information they permitted it to access. P06 engaged with data related to women who have accessed help services or participated in events. She was comfortable storing such information digitally, but drew a clear boundary around giving the same data to GenAI. She also gave the example of handwritten beneficiary lists that could be converted into a digital document using AI:

“Those are not something we would upload [to an AI tool]. We put them in Drive [...] I do not think I

would ever take one of those lists, for example, a handwritten one that I want converted into a PDF or a

document, and put it into an AI tool and say, ‘Convert this for me.’ Those really are sensitive data.” (P06)

P06 saw a distinction between digital storage and disclosure to GenAI. She described GenAI services as “open websites” where she could not be confident that sensitive information would remain securely stored. When discussing the greatest risks, she specifically referred to account numbers, beneficiary lists, and volunteer incentive payments as information that should not be uploaded.

P09 described a diferent way of setting this boundary while using GenAI for project and reporting work. Rather than avoiding the technology, he separated the parts of a task he was willing to give to the system from the information he kept outside it. He added names, budgets, and other details manually after receiving output text from GenAI. P09 explained that he avoided providing personal and operational information because he worried that it could be used for purposes beyond those he intended or become exposed.

“Preparing budgets, numbers, and those things, I never enter them into AI. I only ask it to write an overview of the project, the project’s objectives, and so on. [...] I do not tell AI that this speaker was so-and-so from such-and-such a place.” (P09)

However, three other participants who stated similar boundaries also described workflows that appeared to involve uploading names, numbers, or original records, although their accounts did not establish whether this information had been de-identified.

In these cases, the wartime safety concerns influenced the use of GenAI at diferent points. For P12, the danger associated with digital participation led to the non-use of GenAI during the period when he was in danger. For P06 and P09, the question instead was about what information could be disclosed to GenAI while still using the technology P06 was concerned about losing control over sensitive beneficiary and financial information, while P09 described possible consequences for other volunteers and the wider ERR if information became exposed. These show that wartime safety concerns shaped both whether responders could engage with GenAI and how they limited its involvement in humanitarian work.

## 4.4 GenAI Complicates the Credibility of Humanitarian Communications

Participants expressed concerns about using GenAI to communicate humanitarian eforts to donors and afected communities. Credibility in such cases required not only the clarity of the generated text and the ability to present it in a grammatically correct way, but also that reports and public communication correspond to what had actually happened, use language that people could understand, and remain recognizable as the work of the people producing it. If donor doubts about GenAI-assisted reporting afected funding, ERRs could struggle to provide food and medical assistance, with potentially life-threatening consequences for the communities they serve.

P10 was particularly concerned about GenAI-generated reporting becoming detached from activities on the ground. He described reporting as an account of what had actually been implemented, including its efects, challenges, and feedback from beneficiaries:

“It is a matter of credibility; you need to convey things as they happened, particularly in the program work concerning the impacts of the activities. The lessons learned, the challenges. The impact on the beneficiaries. ChatGPT might give you information diferent from what you want, and it might give you information you would not expect to have been there on the ground. So it becomes a matter of the credibility of the work.” (P10)

Participants also raised questions about whether fluent AI-generated text was appropriate for its intended audience. P15, a non-user, explained that humanitarian communication needs to include local cultures, customs, and languages, including communities for whom Arabic might not be readily understood. His account identifies a requirement against which GenAI-generated communication would need to be assessed. In this case, appropriate communication could no be judged from the fluency of the generated text alone; it required knowledge of the people receiving it.

“There also has to be consideration for people’s cultures, their customs and traditions, and the language communicated to people should also match what they can understand [...] The communities now in the northwest, bordering Chad, most of them -even if you speak Arabic-, a large proportion do not understand what you are saying at all, unless it is explained to them in their own local language.” (P15)

Credibility could also be afected even when participants believed that AI had only played a limited role. P05 described a project proposal that his team had written themselves and then given to AI to organize and turn parts of the material into tables. A donor subsequently questioned whether the entire proposal had been generated by AI:

“We wrote it entirely ourselves and gave it to [AI] only to organize it, nothing more. It made tables for us. Then the donors spoke to us and said that they thought the whole project had been written by AI [...] But we had written it all ourselves. We had only used AI to arrange or organize it.” (P05)

Although P05’s example does not prove that the donor detected the use of AI correctly or that the proposal was rejected, it does show that even limited GenAI assistance could create uncertainty about authorship and the legitimacy of a humanitarian document.

All these examples demonstrate that the concern was not simply whether GenAI produced readable text. Humanitarian communication needs to remain connected to the activities being reported, relevant to the audience it is delivered to, and credible to external partners. GenAI could help with writing, translating, organizing, and presenting texts, but its use also entered existing relationships between responders, afected communities, and donors. Participants therefore viewed human judgment as necessary not only for reviewing what GenAI produced, but also for determining whethe the resulting communication appropriately represented their work and the people involved.

## 4.5 Unequal Access and Collective Use of GenAI During Wartime

The use of GenAI by ERR responders depended not only on personal willingness, but also on diferences in security, connectivity, access to devices, paid services, and educational opportunities. At the same time, GenAI use was shaped by the distribution of work among volunteers, including who had access to these tools, who held relevant local knowledge, and who could review their outputs.

P13, a non-user, believed that there was no urgent need for GenAI since reporting and data work were already handled by another person in his emergency response room: “We do not need [GenAI tools]. In our room someone specializes in data and works on it, communicates with the rooms, prepares the reports and submits them.” (P13)

P14 described a diferent reason for non-use. Her work focused on organizing volunteers and distributions through direct interaction with people and was largely field-based. She had never used GenAI and also reported limited access to smartphones and reliable connectivity, although she expressed interest in using it if appropriate devices and training became available. For both P13 and P14, non-use did not necessarily indicate rejection of GenAI.

P15 anticipated that responders in safer locations would have better opportunities for learning and using GenAI than responders in less secure areas, describing a concern about a possible future diference in GenAI experience rather than an observed diference in proficiency:

“I think people in safe places will get ahead of those in unsafe places in how to use artificial intelligence and benefit from it. They will keep advancing if the war does not stop [...] If we had the same space that they have, we would use artificial intelligence.” (P15)

Even among GenAI users, access was unstable. P02 described electricity cuts and disconnected networks that postponed tasks and created backlogs. Poor and expensive connectivity additionally limited GenAI use. Other participants described that access to paid GenAI was available through certain projects or ofices, user accounts were shared among colleagues, and better-connected people were taking care of the tasks that were not available to others. GenAI access in such a way depended on the resources within the ERR rather than solely on resources held by an individual responder

Access to the tool did not remove dependence on human review. P02, who used GenAI in financial work, argued that checking should reflect the consequences of the task:

“We use [GenAI] to help us, but the output needs to be reviewed. There needs to be a human element going through it and reviewing it [...] With financial matters, you need to check even the fraction, tiniest detail [...] The person generates something without going and verifying it and checking its accuracy; that is entirely their responsibility. We cannot leave it at ‘It was the AI’s fault.’ ” (P02)

P01 described a similar process in which GenAI outputs could be reviewed by the individual user, the wider team, or another responsible person before being acted upon. He also argued that responders should develop the skills needed to perform the task independently before relying on GenAI, warning that otherwise “instead of it being your assistant, you become its assistant.” (P01)

One responder may hold the field knowledge, while another may have better access to the device or network needed to use GenAI, and another may be responsible for checking and reviewing the output before it is shared or acted upon. In this sense, GenAI engagement cannot be understood only at the level of the individual user, since its use also depends on task roles, access to infrastructure, and human review.

## 5 Discussion

Our findings show that active conflict can increase the value of GenAI for community-based responders while undermining the conditions required for its responsible use. We first discuss how this tension shapes GenAI use, then derive implications for humanitarian organizations, donors, and technology providers.

## 5.1 War Reconfigures the Value and Limits of GenAI

Previous work has shown that AI can support humanitarian eforts while simultaneously raising concerns about inaccurate outputs, data privacy, governance, and organizational readiness [8, 10]. Our findings extend this work by showing how active conflict changes both the circumstances that make GenAI assistance valuable and the conditions under which humanitarian responders can access and use it. ERR responders turned to GenAI when it was dificult or dangerous to contact doctors, technical experts, and colleagues. However, war conditions made electricity, the internet, and paid services less accessible, restricted the availability of devices, and decreased the opportunities to learn. When expert human assistance was dificult to reach, GenAI could ofer temporary support, but the same lack of access could delay the qualified human review needed to assess that support. This creates a tension for responsible GenAI use during wartime, insofar as the circumstances in which GenAI becomes most useful can also be those in which appropriate human verification becomes more dificult

The observation that GenAI is not able to independently verify ground truth in wartime adds another limitation. Studies on conflict forecasting show that recent external information and retrieval can improve LLM performance [30], while work on humanitarian reporting shows that retrieval from verified sources and source-linked citations can ground generated outputs [16]. Our participants pointed out changes in prices, needs, population movement, available resources, and security conditions throughout the humanitarian response process. As a result, relevant information may not yet exist in any documents available to be retrieved by a model, particularly when conditions can change faster than they can be documented. This is not a problem of information recency alone. Part of the knowledge needed for humanitarian action is created through interaction with afected communities, local volunteers, and changing conditions on the ground [18, 46]. While GenAI can analyze the available information, it is incapable of replacing local human activities through which some of that knowledge is established and updated

The context of active conflict also reshapes the consequences of GenAI’s limitations. Research on crisis scenarios has shown that LLMs can assign diferent levels of urgency to the same situation depending on the language used [41]. Similarly, a benchmark of LLMs for coding humanitarian data found that aggregate performance measures can obscure weaknesses in identifying concerns related to physical safety and discrimination [29]. Our participants treated fluent output as insuficient when it did not match the local language, the current situation, or the experience of the community Our findings further show that these safety concerns can arise before any information is entered into GenAI. In some situations, having the device or going online was risky. Responsible GenAI use in active conflict therefore cannot be assessed only through model outputs or the data submitted to the system; it also depends on the conditions under which people access the technology and the wider consequences that may follow from its use.

War may also shape the opportunities people have for developing GenAI competence over time. P15 anticipated that responders in a safer environment would have more opportunities for learning and becoming familiar with GenAI, while responders facing higher levels of insecurity might have fewer opportunities. This corresponds to the literature, which associates humanitarian AI adoption with the resource constraints of organizations [10], as well as to literature that discusses the role of power inequalities in participation in humanitarian AI [40]. Yet even where responders had access to GenAI, they did not necessarily have the expertise needed to verify its outputs. Within ERRs, the responder using GenAI was not always the person with current local knowledge or responsibility for reviewing its outputs. Responsible use therefore depended on both access to the technology and coordination with responders who could assess the information it produced.

## 5.2 Implications for Conflict-Sensitive GenAI and Humanitarian Practice

5.2.1 Safety before and beyond GenAI use. Our findings indicate that in an active conflict setting, the risks associated with GenAI could emerge even before any information is fed into the model. The use of GenAI might involve carrying a device, accessing internet connectivity, or digitizing information at a point where such digital activities raise safety concerns. Therefore, it is important for humanitarian organizations and ERR coordinators to establish a protocol fo assessing whether to use GenAI for specific tasks given particular security circumstances. The questions to be asked include whether the use of technology puts a responder in additional danger, whether performing the task require the sharing of sensitive humanitarian information, and whether such disclosure could create risks for other people involved in the humanitarian response.

If such concerns apply, responders might have to limit the use of GenAI or perform tasks without it. Where GenAI use is considered appropriate, responders could remove identifying or sensitive information before submission, although this may not eliminate all disclosure risks. The urgency of the situation does not exempt consideration of information sensitivity. Input from people familiar with the situation on the ground is important for such decisions, with additional specialist consultation where feasible, because general rules do not take into account possible changes in local security. Alternatives to the use of GenAI require a similar evaluation, because transferring the task to another responder may shift the risk rather than remove it.

Technology providers should enable such decision-making by clearly indicating whether submitted information is being stored, for how long, and for what purposes. Users should have practical ways to limit the retention of sensitive humanitarian information, including zero-data-retention options where feasible. All users deserve these abilities, but heightened risks make them especially high-stakes in active conflict settings.

5.2.2 Verification when current information and expertise are unavailable. Our findings indicate that GenAI can be especially useful in responding to war situations when reaching doctors, engineers, or other experts becomes challenging, though these conditions also make its outputs harder to verify. Information about prices, resource availability, needs, or local conditions can change quickly, become out of date, or be hard to verify safely. Humanitarian organizations and ERR coordinators should set task-specific boundaries on GenAI usage. Such boundaries should distinguish cases where GenAI can provide provisional assistance for tasks where errors have limited consequences from cases where its output requires qualified human verification before action is taken.

Where possible and necessary, ERR coordinators could establish peer-review procedures involving responders with the expertise and current local knowledge required to verify the output, while considering whether sharing information for verification creates additional risks. In the absence of verification and when immediate action is necessary, GenAI may still provide provisional information, but its unverified output should not become the sole basis for consequential decisions. Organizations should retain humanitarian procedures or local knowledge as alternative sources for decision-making when expert review is unavailable

Technology providers should facilitate assessment of whether the information obtained by a user is current by providing clear model knowledge cutof dates and, where information is retrieved, verifiable sources and dates of retrieval. These elements would help establish the origin and timeliness of information, even if they cannot guarantee that it corresponds to the situation in the area.

5.2.3 Unequal access and the legitimacy of non-use. Our findings suggest that opportunities to access GenAI tools and acquire relevant skills may difer depending on the region and security circumstances of responders; those in safer and better-connected areas may have greater opportunities to access paid tools or knowledge about GenAI. At the same time, insecurity, low connectivity, few devices, and urgent humanitarian work may limit such opportunities. Moreover, non-use can be deliberate, as some volunteers prefer not to use GenAI for certain tasks because of security concerns or a preference for human judgment. Non-use can also occur when GenAI is not relevant to a responder’s responsibilities or when another person already performs the relevant task.

Therefore, humanitarian organizations and ERR coordinators should diferentiate between non-use caused by access constraints and cases where GenAI is unnecessary or deliberately avoided. Training should be voluntary and related to responders’ responsibilities, while those who choose or need to use GenAI should be supported. Where funders and organizational leadership support GenAI use, they should allocate dedicated funds where possible and avoid initiatives that divert resources from emergency humanitarian operations, since responders reported prioritizing humanitarian needs over GenAI spending. Shared accounts and devices may improve access to GenAI but can expose sensitive information or make responsibility for submitted material unclear. Technology providers could consider ofering lower-cost humanitarian access, low-bandwidth interfaces, and lightweight models suited to unreliable connectivity.

5.2.4 Credibility and representation in humanitarian communication. Participants highlighted concerns related to the possibility that a report prepared with GenAI or a summary of a discussion could misrepresent the situation on the ground. At the same time, volunteers use GenAI to write English-language documents for mainly English-speaking funders. This creates a tension, since the same technology that can improve communication can also distort the message or cast doubt on document authorship.

Humanitarian organizations and ERR coordinators should connect GenAI-assisted reports with evidence of the activities and check translations and documents with the involvement of people who know the local context. Where possible, this evidence should support the claims made in the document without requiring the disclosure of sensitive operational information. Donors should distinguish between the use of GenAI for translation, editing, or organizing documents and the fabrication of claims, and assess documents based on evidence supporting their claims rather than assumptions about AI authorship alone, particularly when volunteers write in a non-native language. They should also consider the time-sensitive and risky conditions under which volunteers work, and the need for rapid assistance from GenAI when human assistance is dificult to obtain. Technology providers could provide optional ways of distinguishing user-written information, information added by GenAI, and human changes while keeping sensitive operational information undisclosed.

## 5.3 Limitations

There are limitations to this research. First, the sample of the exploratory survey and interview is not representative of the diversity of ERRs in Sudan. Darfur and Kordofan were less represented among interviewees than in the survey, and participation largely required connectivity, devices, and relative safety, likely underrepresenting the most constrained responders. In addition, recruitment through existing ERR contacts and participant-referral sampling could have influenced the composition of the sample and how participants reported their experiences. The geographic distribution of users and non-users also difered, limiting our ability to distinguish use-related patterns from regional conditions.

The interviews mostly depend on participants’ reports on the usage of GenAI. We invited participants to share their screens, but all but one declined to do so due to security and privacy concerns. Thus, we were unable to directly observe the usage of GenAI and verify its efectiveness. Medical, technical, and humanitarian cases thus need to be interpreted as participants’ reported experiences, rather than evidence of efective GenAI performance. We encourage future work examining the applicability of these findings to other conflict situations and other community-based humanitarian response eforts, to understand how they difer depending on local security conditions, access to infrastructure, and the way humanitarian response is organized. Longitudinal studies could follow the same responders over time to examine whether access to GenAI, confidence in using it, and local rules for its use change as the conflict, the technology, and their expertise evolve.

## 6 Conclusion

This work examines how community-based humanitarian responders in Sudan use, limit, and avoid GenAI during active armed conflict. The findings demonstrate that war can both increase the value of GenAI when human assistance is dificult to reach, while also weakening the conditions required for responsible use. In such cases, GenAI may provide provisional assistance, but its use still depends on current local knowledge, appropriate human verification, and whether accessing the technology is safe. Responsible use of GenAI in the context of ongoing conflict cannot be considered solely in terms of the performance of models or individual adoption, but must also account for the conditions under which humanitarian responders work. Supporting responsible use means supporting the judgment of responders who decide, often under threat, when these tools help and when they put people at risk.

## 7 Generative AI Usage Statement

We used GPT-6 to assist the native Arabic-speaking researcher with translating the Arabic-language interviews into English. The researcher manually reviewed the translations against the original Arabic to ensure accuracy. We also used GPT-6 alongside QCAmap to generate candidate codes, identify patterns for researcher review, and refine the grammar and wording of the manuscript. We reviewed all translations, coding decisions, and interpretations, and the final text was reviewed and approved by the authors.

## 8 Ethical Considerations and Potential Adverse Impacts

Research during active conflict involved risks of distress and indirect identification. We mitigated these through informed consent, de-identification, restricted data access, and review of potentially identifying contextual details. We also considered the possibility that publishing participants’ accounts could expose sensitive humanitarian activities, so we generalized such details where necessary and had location and role descriptors reviewed by a research team member in the ERR network and an additional person familiar with it.

## References

[1] Alsanosi Adam. 2026. Emergency Response Rooms Work in Sudan – We Could Change Aid Models in Other Warzones. The World Today (15 June 2026). Retrieved November 2026 from https://www.chathamhouse.org/publications/the-world-today/2026-06/emergency-response-rooms-work sudan-we-could-change-aid-models

[2] Juveria Afreen, Mahsa Mohaghegh, and Maryam Doborjeh. 2025. Systematic Literature Review on Bias Mitigation in Generative AI. AI and Ethics 5 (2025), 4789–4841. doi:10.1007/s43681-025-00721-9

[3] Emmanuel Oghenekome Akpoghelie, Emmanuella Obiajulu Chiadika, Great Iruoghene Edo, Asmaa Yahya Al-Baitai, Khalid Zainulabdeen, Sydney Clever Keremah, Irene Ebosereme Ainyanbhor, Patrick Othuke Akpoghelie, Joseph Oghenewogaga Owheruo, Priscillia Nkem Onyibe, Maureen Marris Dinzei, Helen Avuokerie Ekokotu, Ufuoma Ugbune, Ephraim Evi Alex Oghroro, and Lauretta Dohwodakpo Ekpekpo. 2024. Malnutrition

and Food Insecurity in Northern Nigeria: An Insight into the United Nations World Food Program (WFP) in Nigeria. Discover Food 4, Article 165 (2024). doi:10.1007/s44187-024-00249-7

[4] Dalia Ali, Muneeb Ahmed, Hailan Wang, Arfa Khan, Naira Paola Arnez Jordan, Sunnie SY Kim, Meet Dilip Muchhala, Anne Kathrin Merkle, and Orestis Papakyriakopoulos. 2025. AI Adoption Across Mission-Driven Organizations. arXiv preprint arXiv:2510.03868.

[5] Amnesty International. 2024. Sudan: Internet Shutdown Threatens Delivery of Humanitarian and Emergency Services. https://www.amnesty.org/ en/latest/news/2024/03/sudan-internet-shutdown-threatens-delivery-of-humanitarian-and-emergency-services

[6] Yasmine Attia, Diana Maddah, Al-Anoud Al-Kaabi, Ghalia Seifo, Mohammed Bader Al-Sada, Jesha Mohammed Ali Mundodan, Abdelaziz Khatir Jadeed, Niveen ME Abu-Rmeileh, and Kwalombota Kwalombota. 2026. Responsible AI readiness in a humanitarian and development ecosystem: an exploratory case study. Frontiers in Public Health 14 (2026), 1937518. doi:10.3389/fpubh.2026.1937518

[7] Dustin Barter and Gun Mai Sumlut. 2023. The ‘conflict paradox’: humanitarian access, localisation, and (dis) empowerment in Myanmar, Somalia, and Somaliland. Disasters 47, 4 (2023), 849–869.

[8] Ana Beduschi. 2022. Harnessing the potential of artificial intelligence for humanitarian action: Opportunities and risks. International Review ofthe Red Cross 104, 919 (2022), 1149–1169.

[9] Gábor Bella, Paula Helm, Gertraud Koch, and Fausto Giunchiglia. 2024. Tackling language modelling bias in support of linguistic diversity. In Proceedings ofthe 2024 ACM Conference on Fairness, Accountability, and Transparency. Association for Computing Machinery, New York, NY, USA, 562–572.

[10] Tigmanshu Bhatnagar, Maarya Omar, Davor Orlic, James Smith, Catherine Holloway, and Maria Kett. 2025. Bridging AI and humanitarianism: An HCI-informed framework for responsible AI adoption. In Proceedings of the 2025 CHI Conference on Human Factors in Computing Systems. Association for Computing Machinery, New York, NY, USA, 1–17.

[11] Miriam Bradley. 2023. The Politics and Everyday Practice of International Humanitarianism. Oxford University Press. doi:10.1093/oso/9780198840695. 001.0001

[12] Virginia Braun and Victoria Clarke. 2006. Using thematic analysis in psychology. Qualitative Research in Psychology 3, 2 (2006), 77–101.

[13] Virginia Braun and Victoria Clarke. 2021. One size fits all? What counts as quality practice in (reflexive) thematic analysis? Qualitative research in psychology 18, 3 (2021), 328–352.

[14] Nicholas Carlini, Florian Tramèr, Eric Wallace, Matthew Jagielski, Ariel Herbert-Voss, Katherine Lee, Adam Roberts, Tom Brown, Dawn Song, Úlfar Erlingsson, Alina Oprea, and Colin Rafel. 2021. Extracting Training Data from Large Language Models. In 30th USENIX Security Symposium (USENIX Security 21). USENIX Association, 2633–2650. https://www.usenix.org/conference/usenixsecurity21/presentation/carlini-extracting

[15] Becky Carter and Hassan-Alattar Satti. 2026. Supporting conflict-sensitive, locally-led humanitarianism in Sudan: rebalancing donors’ approach to risk. Disasters 50, 1 (2026), e70034

[16] Ivan Decostanzi, Yelena Mejova, and Kyriaki Kalimeri. 2026. A large-language-model framework for automated humanitarian situation reporting. Scientific Reports (2026). doi:10.1038/s41598-026-68010-x

[17] Thomas Fenzl and Philipp Mayring. 2017. QCAmap: eine interaktive Webapplikation für qualitative inhaltsanalyse. Zeitschriftfür Soziologie der Erziehung und Sozialisation 37 (08 2017), 333–339.

[18] Megan Finn and Elisa Oreglia. 2016. A fundamentally confused document: Situation reports and the work of producing humanitarian information In Proceedings ofthe 19th ACM Conference on Computer-Supported Cooperative Work & Social Computing. Association for Computing Machinery, New York, NY, USA, 1349–1362.

[19] Yu. A. Fomina, M. N. Guseva, and A. M. M. Izz Eldin. 2026. Artificial Intelligence in Humanitarian Projects of the World Food Program. Scientific and Technical Information Processing 53 (2026), 155–161. doi:10.3103/S0147688226700140

[20] Shan He, Shabnam Mehboob, Naila Afghan, and Nguyen Manh Hung. 2026. Artificial Intelligence and Machine Learning in Predicting Food Hunger and Severity: A Comprehensive Review. Journal of Food Composition and Analysis 158, Article 109474 (2026).

[21] Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong, Zhangyin Feng, Haotian Wang, Qianglong Chen, Weihua Peng, Xiaocheng Feng, Bing Qin, and Ting Liu. 2025. A Survey on Hallucination in Large Language Models: Principles, Taxonomy, Challenges, and Open Questions. ACM transactions on information systems 43, 2, Article 42 (2025), 55 pages. doi:10.1145/3703155

[22] Human Rights Watch. 2024. Khartoum Is Not Safe for Women!: Sexual Violence against Women and Girls in Sudan’s Capital. https://www.hrw org/report/2024/07/28/khartoum-not-safe-women/sexual-violence-against-women-and-girls-sudans-capita

[23] Muhammad Imran, Carlos Castillo, Ji Lucas, Patrick Meier, and Sarah Vieweg. 2014. AIDR: Artificial Intelligence for Disaster Response. In Proceedings ofthe 23rd International Conference on World Wide Web (WWW’14 Companion). Association for Computing Machinery, New York, NY, USA, 159–162 doi:10.1145/2567948.2577034

[24] International Committee of the Red Cross. 2026. ICRC Annual Report 2025. https://www.icrc.org/en/report/icrc-annual-report-2025

[25] International Telecommunication Union. 2021. United nations activities on artificial intelligence (AI). https://www.itu.int/pub/S-GEN-UNACT-202

[26] Katie Bo Lillis and Zachary Cohen. 2026. Exclusive: US military had close call after using AIfor false intelligence report, sources say. Retrieved 2026-10-22 from https://edition.cnn.com/2026/09/18/politics/us-military-ai-false-intelligence-china-ship

[27] Sophie Lythreatis, Fulya Acikgoz, and Noura Yassine. 2026. Artificial Intelligence in Humanitarian Aid: A Review and Future Research Agenda. Technovation 151, Article 103415 (2026). doi:10.1016/j.technovation.2025.103415

[28] Thomas Manzini, Priyankari Perali, Jayesh Tripathi, and Robin R Murphy. 2025. Now you see it, now you don’t: Damage label agreement in drone & satellite post-disaster imagery. In Proceedings of the 2025 ACM Conference on Fairness, Accountability, and Transparency. Association for Computing

Machinery, New York, NY, USA, 1998–2008.

[29] Jerome Marston, Tino Kreutzer, Salomé Garnier, Ella Boone, Phuong N Pham, and Patrick Vinck. 2026. Can Large Language Models Reliably Code Qualitative Humanitarian Data? A Benchmark Study Against Human Expert Adjudication. arXiv preprint arXiv:2606.26541.

[30] Apollinaire Poli Nemkova, Sarath Chandra Lingareddy, Sagnik Ray Choudhury, and Mark V Albert. 2025. Do large language models know conflict? Investigating parametric vs. Non-parametric knowledge of LLMs for conflict forecasting. arXiv preprint arXiv:2505.09852.

[31] Samantha K. Olson, Maysoon Dahab, and Melissa Parker. 2024. Key Considerations: Mutual Aid Lessons and Experiences from Emergency Response Rooms in Sudan. Technical Report. Social Science in Humanitarian Action (SSHAP). doi:10.19088/SSHAP.2024.056

[32] OpenAI. 2026. GPT-6. Large language model.

[33] Anila Qehaja. 2026. Better Crisis Data, Faster Recovery. United Nations Development Programme (UNDP). Retrieved 2026-10-21 from https: //www.undp.org/blog/better-crisis-data-faster-recovery

[34] Róisín Read, Bertrand Taithe, and Roger Mac Ginty. 2016. Data hubris? Humanitarian information systems and the mirage of technology. Third World Quarterly 37, 8 (2016), 1314–1331.

[35] Kristin Bergtora Sandvik, Maria Gabrielsen Jumbert, John Karlsrud, and Mareile Kaufmann. 2014. Humanitarian technology: a critical research agenda. International review ofthe Red Cross 96, 893 (2014), 219–242

[36] Kristin Bergtora Sandvik and Nathaniel A. Raymond. 2017. Beyond the Protective Efect: Towards a Theory of Harm for Information Communication Technologies in Mass Atrocity Response. Genocide Studies and Prevention: An International Journal 11, 1 (2017), 9–24. doi:10.5038/1911-9933.11.1.1454

[37] Veronica Scuotto, Kingsley Obi Omeihe, Valentina Cillo, and Del Giudice Manlio. 2025. Tackling the empowerment of artificial intelligence for humanitarian intervention from save the children. Technological Forecasting and Social Change 221 (2025), 124330.

[38] Meltem Şenol Balaban, Canay Doğulu, Nil Akdede, Haldun Akoğlu, Onur Karakayalı, Sarper Yılmaz, Serkan Yılmaz, Tolulope Ajobiewe, Selin Güzel, Gözde İkizer, Muge Akin, Yeşim Ünal, and Ayşe Nuray Karancı. 2025. Emergency Response, and Community Impact after February 6 2023 Kahramanmaraş Pazarcık and Elbistan Earthquakes: Reconnaissance Findings and Observations on Afected Region in Türkiye. Bulletin of Earthquake Engineering 23 (2025), 1053–1081. doi:10.1007/s10518-024-01867-3

[39] Robin Staab, Mark Vero, Mislav Balunovic, and Martin Vechev. 2024. Beyond Memorization: Violating Privacy via Inference with Large Language Models. In International Conference on Learning Representations.

[40] Stella Suge, Sarah W Spencer, Nyalleng Moorosi, Helen McElhinney, Geof Loane, and Sue Black. 2026. From experimentation to engagement: on the paradox of participatory AI and power in contexts of forced displacement and humanitarian crises. arXiv preprint arXiv:2604.06219

[41] Belu Ticona and Antonis Anastasopoulos. 2026. LLM-Powered Automatic Translation and Urgency in Crisis Scenarios. arXiv preprint arXiv:2602.13452.

[42] Chandrasekhar Uddagiri and Bala Venkateswarlu Isunuri. 2024. Ethical and Privacy Challenges of Generative AI. In Generative AI: Current Trends and Applications. Springer Nature, Singapore, 219–244. doi:10.1007/978-981-97-8460-8\_11

[43] United Nations High Commissioner for Refugees. 2026. Figures at a Glance. UNHCR. Data from Global Trends 2025, updated 11 June 2026. https://www.unhcr.org/about-unhcr/overview/figures-glanc

[44] Rob Vos, Arif Husain, Friederike Greb, Peter Läderach, and Brendan Rice. 2023. Food crisis risk monitoring: Early warning for early action. In Global Food Policy Report2023: Rethinking Food Crisis Responses. International Food Policy Research Institute, Chapter 2, 20–35. doi:10.2499/9780896294417\_02

[45] Jane Wakefield. 2022. Deepfake presidents used in Russia-Ukraine war. BBC News (18 March 2022). Retrieved 2026-10-22 from https://www.bbc. com/news/technology-60780142

[46] David Gray Widder. 2024. Epistemic power in AI ethics labor: Legitimizing located complaints. In Proceedings ofthe 2024 ACM Conference on Fairness, Accountability, and Transparency. Association for Computing Machinery, 1295–1304. doi:10.1145/3630106.3658973

[47] World Food Programme. 2026. WFP Releases HungerMap Live; a Modernized Intelligence Platform that Turns Data on Global Hunger into Early Action. Retrieved 2026-10-21 from https://www.wfp.org/news/wfp-releases-hungermap-live-modernized-intelligence-platform-turns-data-global hunger-early

[48] World Health Organization. 2026. Health Emergency Appeal 2026: Sudan. https://www.who.int/publications/m/item/sudan--who-healthemergency-appeal-2026

[49] Fengyi Xu, Jun Ma, Nan Li, and Jack C.P. Cheng. 2025. Large Language Model Applications in Disaster Management: An Interdisciplinary Review. International Journal ofDisaster Risk Reduction 127, Article 105642 (2025). doi:10.1016/j.ijdrr.2025.105642

## A Exploratory Survey

## A.1 Exploratory Survey: Descriptive Results

Before undertaking the survey, we provided the respondents with information about the research and a consent form. The term "GenAI" was explained as referring to AI generating content, and the examples mentioned included ChatGPT, Gemini, Copilot, and Claude. The resulting analytical sample consisted of 89 respondents. In this sample, 49 respondents (55.1%) used GenAI in their ERR activities, 30 respondents (33.7%) did not use it, and 10 respondents (11.2%) were unsure. Among those who reported the use of GenAI, the most common activities included translation (22/49, 44.9%), data analysis (21/49, 42.9%), and creation of social media content (20/49, 40.8%). Among those who reported no use of GenAI, the top three answers were lack of knowledge about how to use GenAI (8/30, 26.7%), distrust of the output (7/30, 23.3%), and unreliable internet connection (5/30, 16.7%). These statistics reflect the participants recruited into the survey, and it is not safe to generalize them for all of the ERR members. The full questionnaire employed in the exploratory survey is provided below.

Table 2. Characteristics of exploratory survey respondents $\left( N = 8 9 \right)$ . Counts are based on each respondent’s selected primary role, tenure and geographic area primarily supported. Regions are aggregated for confidentiality.
<table><tr><td>Characteristic</td><td>n</td><td>%</td><td></td><td></td></tr><tr><td>Primary role</td><td>34.8</td><td></td><td>n</td><td>%</td></tr><tr><td>Coordinator/leader Field volunteer</td><td>31</td><td></td><td>Geographic area supported</td><td></td></tr><tr><td></td><td>17</td><td>19.1</td><td>Kordofan states</td><td>34.8</td></tr><tr><td>Communications/media</td><td>11</td><td>12.4</td><td>Darfur states</td><td>34.8</td></tr><tr><td>Data/reporting</td><td>10</td><td>11.2</td><td>Khartoum</td><td>13.5</td></tr><tr><td>Finance/administration</td><td>7</td><td>7.9</td><td>Other states</td><td>13.5</td></tr><tr><td>Other roles</td><td>13</td><td>14.6</td><td>Prefer not to answer</td><td>3.4</td></tr><tr><td>Time working with ERRs Less than 3 months</td><td></td><td></td><td>GenAI use in ERR work</td><td></td></tr><tr><td></td><td>2</td><td>2.2</td><td>Yes</td><td>55.1</td></tr><tr><td>3-6 months 7-12 months</td><td>13</td><td>14.6</td><td>No</td><td>33.7</td></tr><tr><td></td><td>5</td><td>5.6</td><td>Not sure</td><td>11.2</td></tr><tr><td>More than 1 year Prefer not to answer</td><td>67 2</td><td>75.3 2.2</td><td></td><td></td></tr></table>

Interpretation note. Some respondents who reported not using GenAI in their ERR work also answered questions about GenAI tools and verification. We therefore report tasks, tools, and devices for the 49 self-reported users, and reasons for non-use for the 30 self-reported non-users. Likert results include all 89 respondents and measure reported agreement rather than actual behavior. Among these respondents, 16.9% disagreed that GenAI outputs usually require verification, 21.3% disagreed that human review is needed in urgent situations, and 29.2% disagreed that inaccurate outputs could harm beneficiaries. We report descriptive results only, without inferential comparisons.

Table 3. Reported tools and working conditions among respondents who indicated that they used GenAI in ERR work $( n = 4 9 )$ Multiple tools and devices could be selected.
<table><tr><td>Characteristic</td><td>n</td><td>%</td></tr><tr><td>Tools</td><td></td><td></td></tr><tr><td>ChatGPT</td><td>32</td><td>65.3</td></tr><tr><td>Gemini</td><td>27</td><td>55.1</td></tr><tr><td>Claude</td><td>12</td><td>24.5</td></tr><tr><td>Copilot</td><td>4</td><td>8.2</td></tr><tr><td>Name unknown</td><td>6</td><td>12.2</td></tr><tr><td>Other</td><td>3</td><td>6.1</td></tr><tr><td>Primary language used</td><td></td><td></td></tr><tr><td>Arabic</td><td>26</td><td>53.1</td></tr><tr><td>English</td><td>1</td><td>2.0</td></tr><tr><td>Both</td><td>21</td><td>42.9</td></tr><tr><td>Other</td><td>1</td><td>2.0</td></tr><tr><td>Devices</td><td></td><td></td></tr><tr><td>Personal phone</td><td>43</td><td>87.8</td></tr><tr><td>Shared phone</td><td>2</td><td>4.1</td></tr><tr><td>Personal computer/laptop</td><td>16</td><td>32.7</td></tr><tr><td>Shared computer/laptop</td><td>4</td><td>8.2</td></tr></table>

## A. Reported GenAI tasks

![](images/d7251b9c2dc00508cc010e16a848a745afb12e630a096464a2b925b4a1515bae.jpg)

B. Reasons for non-use  
![](images/5187c2c53c2c5ce342631daf8d2f1617b7cf0be93cc51916949d22538df24b71.jpg)  
Fig. 1. Selected tasks among respondents reporting GenAI use (� = 49; panel A), and selected reasons for non-use among respondents reporting no GenAI use (� = 30; panel B). Percentages are calculated using the respective group denominators (49 users and 30 non-users).

Table 4. Response distributions for all 23 Likert statements (� = 89 per item). Values are percentages of respondents. Disagree combines strongly disagree and disagree; agree combines agree and strongly agree. Each item is stated as a claim or concern, so agreement is not uniformly favourable towards GenAI.
<table><tr><td>Item</td><td>Statement (shortened)</td><td>Disagree</td><td>Neutral</td><td>Agree</td></tr><tr><td colspan="3">Technological factors</td><td></td><td></td></tr><tr><td>T1</td><td>GenAI improves work quality or speed</td><td>28.1</td><td>6.7 9.0</td><td>65.2</td></tr><tr><td>T2</td><td>Useful in fast-paced ERR work</td><td>25.8</td><td></td><td>65.2</td></tr><tr><td>T3</td><td>Can use GenAI with little help</td><td>31.5</td><td>15.7</td><td>52.8</td></tr><tr><td>T4</td><td>Supports different work tasks</td><td>23.6</td><td>6.7</td><td>69.7</td></tr><tr><td>T5</td><td>Understands Sudanese local context</td><td>33.7</td><td>16.9</td><td>49.4</td></tr><tr><td>T6</td><td>Reliable access to a device</td><td>27.0</td><td>18.0</td><td>55.1</td></tr><tr><td>T7</td><td>Internet connection sufficiently stable</td><td>38.2</td><td>16.9</td><td>44.9</td></tr><tr><td>T8</td><td>Can charge devices when needed</td><td>39.3</td><td>13.5</td><td>47.2</td></tr><tr><td>T9</td><td>GenAI information is usually accurate</td><td>34.8</td><td>23.6</td><td>41.6</td></tr><tr><td>T10</td><td>Arabic/local dialect outputs worse than English</td><td>36.0</td><td>14.6</td><td>49.4</td></tr><tr><td colspan="3">Organisational factors</td><td></td><td></td></tr><tr><td>O1</td><td>ERR leaders support GenAI use</td><td>29.2</td><td>13.5</td><td>57.3</td></tr><tr><td>O2</td><td>ERR members open to new digital tools</td><td>25.8</td><td>12.4 13.5</td><td>61.8</td></tr><tr><td>03 04</td><td>ERRs can afford GenAI-related costs</td><td>47.2</td><td>16.9</td><td>39.3</td></tr><tr><td>05</td><td>Peers can help with GenAI use Access to training or support</td><td>25.8 29.2</td><td></td><td>57.3</td></tr><tr><td></td><td></td><td></td><td>22.5</td><td>48.3</td></tr><tr><td colspan="3">Environmental factors</td><td></td><td></td></tr><tr><td>E1</td><td>Concern about exposing sensitive information</td><td>28.1</td><td>21.3</td><td>50.6</td></tr><tr><td>E2</td><td>More funding would enable greater use</td><td>22.5</td><td>10.1</td><td>67.4</td></tr><tr><td>E3</td><td>GenAI may weaken community trust</td><td>37.1</td><td>31.5</td><td>31.5</td></tr><tr><td>E4</td><td>Government rules for GenAI are clear</td><td>43.8</td><td>28.1</td><td>28.1</td></tr><tr><td>E5</td><td>Concern that inputs may be monitored</td><td>30.3</td><td>20.2</td><td>49.4</td></tr><tr><td>E6 E7</td><td>Usually need to verify GenAI outputs</td><td>16.9 29.2</td><td>12.4 20.2</td><td>70.8 50.6</td></tr><tr><td>E8</td><td>Inaccurate outputs could harm beneficiaries</td><td>21.3</td><td></td><td></td></tr><tr><td></td><td>Human review needed in urgent situations</td><td></td><td>15.7</td><td>62.9</td></tr></table>

![](images/20c0a2fd924013623eaed803884f2718b5094ef6eb9202b529647cd5c9a43ff4.jpg)  
Fig. 2. Distribution of all 23 Likert responses (� = 89). Labels $\mathsf { T } 1 \mathrm { - } \mathsf { T } 1 0 , \mathsf { O } 1 \mathrm { - } \mathsf { O } 5 ,$ and E1–E8 correspond to the technological, organizational, and environmental statements in Table 4. Items are not combined into composite scores.

## 01 Background and eligibility

1

I have read the information above and agree to participate in this study

O No

2

Are you over 18 years old?

3

What is your main role in the Emergency Response Rooms?

Coordinator/leader

Data/reporting

Field volunteer

Medical/health

Communications/media

Other

Finance/administration

## 4 Which project or activity areas of the Emergency Response Rooms are you currently involved in or familiar with?

Community kitchens / food distribution

Logistics and supply distribution

Health and medical services

Information management and needs assessment

Medical referral or transport

Volunteer coordination

Shelter and support for displaced people

Education support

Water, sanitation, and hygiene

External relations and partnerships

Protection and support for vulnerable groups

□ Other

## 02 GenAl use and non-use

8

Do you use Generative Al tools in your work with the Emergency Response Rooms?

Not sure

ONo

9

## How often do you use Generative Al tools in the work of the Emergency Response Rooms?

Always

Rarely

Often

Never

Sometimes

Not sure / Not applicable

Seasonally

Which Generative Al tools do you use?

10

ChatGPT

□Gemini

Claude

I do not know the name

☐ Copilot

Other

11

If you selected “Other” in the previous question, please specify which Generative Al tools you use.

12

## For which tasks do you use Generative Al tools?

Translation

Data analysis

Writing or editing reports or messages

Planning and coordination

Social media posts

□ Other

Summarizing information

Cash assistance fundraising

## 5 If you selected “Other" in the previous question, please specify which project or activity areas of the Emergency Response Rooms you are currently involved in or familiar with.

## 6 How long have you worked with the Emergency Response Rooms?

If you selected “Other” in the previous question, please specify the tasks for which you use Generative Al tools

Less than 3 months

3 to 6 months

More than 1 year

13

Prefer not to answer

7 to 12 months

Both

Which language do you mostly use with Generative Al tools?

Other

Northern State

## 7 In which geographic area do you mainly support the work of the Emergency Response Rooms?

Personal phone

Shared phone

North Kordofan

On which devices do vou use Generative Al tools?

South Kordofan

Shared computer or laptop □ Other

West Kordofan

AI Jazirah

Khartoum

River Nile

North Darfur

South Darfur

English

15

Arabic

14

Personal computer or laptop

East Darfur

Red Sea

Kassala

Central Darfur

Sennar

West Darfur

Gedaref

Outside Sudan

White Nile

Prefer not to answer

Blue Nile

16

Please describe examples of how Generative Al tools are used in the work of the Emergency Response Rooms.

17

Why have you not used Generative Al tools?

I do not know how to use them

They do not support Arabic well

I do not trust their outputs

They are not useful for my work

The internet connection is unstable

I do not have suitable devices

No one in my team uses them

I do not have time to learn

I have privacy or security concerns

Other

18

If you selected “Other" in the previous question, please specify the reasons why you have not used them.

Fig. 3. Exploratory survey questionnaire (Part 1).

<table><tr><td>Verification of GenAl outputs</td><td></td><td></td><td></td></tr><tr><td colspan="4">19 When Generative Al tools produce an answer, I check the answer before using it.</td></tr><tr><td>20 If you check the answer, how do you usually do so?</td><td>Always Often Sometimes Seasonally Rarely</td><td>Never</td><td>Not sure / Not applicable</td></tr></table>

<table><tr><td colspan="6">Five-point response scale for Q21-Q43: SD Strongly disagree D Disagree N Neither agree nor disagree A Agree SA Strongly agree</td></tr><tr><td colspan="2">Technological factors</td><td></td><td>SD D</td><td>N</td><td></td><td>A</td><td>SA</td></tr><tr><td>21 before.</td><td>Using Generative Al tools helps me do my work in the Emergency Response Rooms better or faster than</td><td></td><td>O</td><td>O</td><td>○ O</td><td></td><td>O</td></tr><tr><td>22</td><td>Generative Al tools are useful in the fast-paced environment of Emergency Response Rooms.</td><td></td><td>O</td><td>O</td><td>O</td><td>O</td><td>O</td></tr><tr><td>23</td><td>I can use Generative Al tools properly without needing much help. Generative Al tools can support different tasks I do, such as writing, translation, planning, reporting, or</td><td>○○</td><td></td><td></td><td>O</td><td>O</td><td>O</td></tr><tr><td>24</td><td>communication.</td><td></td><td>O</td><td>O</td><td>O</td><td>O</td><td>O</td></tr><tr><td>25</td><td>Generative Al tools understand local words, places, and context in Sudan.</td><td></td><td>O</td><td>O</td><td>O</td><td>O</td><td>O</td></tr><tr><td>26</td><td>I have a working phone or computer available whenever I need to use Generative Al tools.</td><td></td><td>O</td><td>O</td><td>O</td><td>O</td><td>O</td></tr><tr><td>27</td><td>The internet or mobile-data connection is stable enough to use Generative Al tools.</td><td>O</td><td></td><td>O</td><td>O</td><td>O</td><td>O</td></tr><tr><td>28</td><td>I can charge my phone or computer whenever I need to use Generative Al tools for Emergency Response Room work.</td><td>O</td><td></td><td>O</td><td>○</td><td>O</td><td>O</td></tr><tr><td>29</td><td>The information that Generative Al tools provide to us is usually accurate.</td><td>O</td><td></td><td>O</td><td>o</td><td>O</td><td>O</td></tr><tr><td>30</td><td>The quality of Generative Al outputs is lower when using Arabic or local dialects compared with English.</td><td>O</td><td></td><td>O</td><td>O</td><td>O</td><td>O</td></tr><tr><td colspan="2">Organizational factors</td><td>SD</td><td></td><td>D</td><td>N</td><td>A</td><td>SA</td></tr><tr><td>31</td><td>The leaders of our Emergency Response Rooms support the use of Generative Al tools in our work.</td><td>O</td><td></td><td>O</td><td>O</td><td>O</td><td>O</td></tr><tr><td>32</td><td>People in the Emergency Response Rooms are open to trying new digital tools and are willing to learn from each other about tools such as Generative Al.</td><td>O</td><td></td><td>O</td><td>O</td><td>O</td><td>O</td></tr><tr><td>33</td><td>In the Emergency Response Rooms, we can afford the additional costs of using Generative Al tools, such as devices, internet, or paid versions.</td><td>O</td><td></td><td>O</td><td>O</td><td>O</td><td>O</td></tr><tr><td>34</td><td>We have people in the Emergency Response Rooms who can help others use Generative Al tools properly.</td><td>O</td><td></td><td>O</td><td>O</td><td>O</td><td>O</td></tr><tr><td>35</td><td>We have access to training or support when we want to learn how to use Generative Al tools.</td><td>○○</td><td></td><td></td><td>○</td><td>O</td><td>O</td></tr><tr><td colspan="2">Environmental factors</td><td>SD</td><td></td><td>D</td><td>N</td><td>A</td><td>SA</td></tr><tr><td>36</td><td>I am concerned that sensitive information about the people we help could be exposed through Generative Al tools.</td><td></td><td>O</td><td>O</td><td>O</td><td>O</td><td>O</td></tr><tr><td>37</td><td>I could use Generative Al tools more if there were enough funding for them from the Emergency Response Rooms.</td><td>O</td><td></td><td>O</td><td>O</td><td>O</td><td>O</td></tr><tr><td>38</td><td>Using Generative Al tools may weaken trust between us and the people we help.</td><td></td><td>O</td><td>O</td><td>O</td><td>O</td><td>O</td></tr><tr><td>39</td><td>The government has clear rules or laws about how organizations such as Emergency Response Rooms can use Generative Al tools.</td><td></td><td>O</td><td>O</td><td>O</td><td>O</td><td>O</td></tr><tr><td>40</td><td>I am concerned that someone could see or monitor the information I enter into Generative Al tools.</td><td></td><td>O</td><td>O</td><td>O</td><td>O</td><td>O</td></tr><tr><td>41</td><td>I usually need to verify the outputs of Generative Al tools before using them in my work.</td><td></td><td>O</td><td>O</td><td>O</td><td>O</td><td>O</td></tr><tr><td>42</td><td>Relying on inaccurate information from Generative Al tools could cause harm to the people we help.</td><td>○○</td><td></td><td></td><td>O</td><td>O</td><td>O</td></tr><tr><td>43</td><td>In urgent situations, I cannot rely on Generative Al tools without human review.</td><td>O</td><td></td><td>O</td><td>O</td><td>O</td><td>O</td></tr><tr><td colspan="8">Follow-up and additional comments</td></tr><tr><td>44</td><td>Are there any other benefits or challenges of using Generative Al tools in your work that you would like to share?</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>45</td><td colspan="8">May we contact you for a short follow-up interview?</td></tr><tr><td>46</td><td colspan="8">OYes O No Maybe Please share one way we can contact you, such as an email address or WhatsApp number. Your contact details will be used only to arrange the interview, will be kept separate from your survey answers, and will not be shared with anyone outside the research team. You can change your mind at any time.</td></tr></table>

Fig. 4. Exploratory survey questionnaire (Part 2).

## B Interview Instrument, Coding and Thematic Analysis

## B.1 Interview Instrument

Interview recruitment sought variation across humanitarian roles and regions, although wartime insecurity and disrupted connectivity constrained participation from some areas. Figure 5 summarises the semi-structured interview guide, core questions for all participants, separate follow-up questions for GenAI users and non-users, and shared closing questions.

![](images/e8875eed53e3d02fd1c6f625b3d9d0fda4b43e94aa5744713a8d11a427cdb72d.jpg)  
Fig. 5. Overview of the semi-structured interview instrument (� = 15).

## B.2 Data Protection and Researcher Responsibility

We conducted the study with the approval of the institutional ethics board. Before conducting the interviews, the interviewer informed each participant that anonymized transcripts would be translated and analyzed with the help of an AI model. The interviewees gave their verbal consent before continuing the process.

We only provided anonymized interview transcripts to GPT-6 using a paid ChatGPT Pro service with data sharing for model improvement turned of. Because LLMs can infer attributes such as place of residence from contextual details even after explicit identifiers are removed, we also generalized locations, roles, and other contextual details before submission.

We also note a tension between this practice and participants’ own concerns about disclosing information to Generative AI (Section 4.3). We therefore limited model input to de-identified text and kept final analytical decisions with the researchers. The native Arabic-speaking researcher manually reviewed all GPT-6-assisted translations against the Arabic originals. Translation and coding nonetheless remain subject to interpretive limitations.

## B.3 Qualitative Coding Framework

The interview coding workbook contains 83 operational codes assigned to 217 selected meaning units from 15 interviews. The code IDs P01–P14 refer to operational codes, not to interview participants (P01–P15) in Section 4. The research team reviewed the codebook and confirmed the relationship between the coded material and the five final themes presented in Section 4. Table B1 presents the thematic groupings used during the analysis and their associated operational codes, while Table B2 provides the full code labels and coded-unit counts. These counts refer to coded passages, not the number of participants or the prevalence of an experience.

Table B1. Thematic groupings and associated operational code IDs used during the analysis.
<table><tr><td>Working thematic group</td><td>Associated code IDs</td></tr><tr><td>Practical benefits and value</td><td></td></tr><tr><td>Fallback when human support is disrupted</td><td>N01, N07, P06</td></tr><tr><td>Sustaining care under constrained conditions</td><td>N02, N03, S11</td></tr><tr><td>Augmentation while retaining human authority</td><td>P01, P07, P12</td></tr><tr><td>Organisational translation and knowledge infrastructure</td><td>P03, P04, P14</td></tr><tr><td>Administrative capacity under resource constraints</td><td>P06, P08, P12</td></tr><tr><td>Adoption and access conditions</td><td></td></tr><tr><td>Fit with existing tasks and work gaps</td><td>P10, L01, T04, P02, P11, T01</td></tr><tr><td>Learning through practical experience and peers</td><td>L02, L09, N04, T02</td></tr><tr><td>Wartime digital access, costs and infrastructure</td><td>A01, A02, A05, A06, A08, A09, P05</td></tr><tr><td>Unequal geographic and security conditions</td><td>A07, T05, T09, S03</td></tr><tr><td>Emergency pressure and limits to reliance</td><td>N05, N06, N08, N09</td></tr><tr><td>Organisational support and distributed knowledge</td><td>A04, A10, L10, P13</td></tr><tr><td>Training and foundational capacity</td><td>L01, L03, L04, L05, L11</td></tr><tr><td>Professional judgement and selective adoption</td><td>T01, T04, T07, S12</td></tr><tr><td>Risks</td><td></td></tr><tr><td>Exposing people through humanitarian data</td><td>S01, S02, S07, S08, V08</td></tr><tr><td>Physical exposure through digital traces</td><td>S03, T09</td></tr><tr><td>Decisions made without reliable ground knowledge</td><td>E03, E08, V07</td></tr><tr><td>Loss of local meaning in AI-mediated work</td><td>E02, E04, E05, E09</td></tr><tr><td>Displacement of community knowledge and judgment</td><td>E01, E06</td></tr><tr><td>Threats to dignity and legitimacy</td><td>S09, S10, E07</td></tr><tr><td>Costs shifted onto frontline volunteers</td><td>A03, T08</td></tr><tr><td>Network-wide security and governance gaps</td><td>S02, L06, L08</td></tr><tr><td>Blurred responsibility for AI-assisted work</td><td>V04, V05</td></tr><tr><td>Safeguards and human oversight</td><td></td></tr><tr><td>Keep sensitive information outside AI systems</td><td>S04, S05, S06</td></tr><tr><td>Check outputs with evidence and human review</td><td>V01, V02, V03, V06, L12</td></tr><tr><td>Co-design guidance with ERR volunteers</td><td>L07</td></tr></table>

Coding note. Some codes recur across working themes. The complete codebook below follows the workbook’s analytic role categories rather than assigning each code to a single final theme.

Table B2. Operational codebook for the ERR interview analysis. Codes and passage counts are transcribed from the coding workbook.

<table><tr><td>Code</td><td>Operational code label</td><td>Units</td></tr><tr><td colspan="3">Adoption and use processes (48 codes)</td></tr><tr><td>A01</td><td>Losing digital time savings to unreliable power and connectivity</td><td>3</td></tr><tr><td>A02</td><td>Prioritizing subsistence and aid over digital expenditure</td><td>5</td></tr><tr><td>A04</td><td>Maintaining work through connected intermediaries and redundancy</td><td>3</td></tr><tr><td>A05</td><td>Obtaining paid capacity through selective support arrangements</td><td>6</td></tr><tr><td>A06</td><td>Finding that access barriers obstruct training itself</td><td>3</td></tr><tr><td>A07</td><td>Anticipating conflict-driven geographic inequality in adoption</td><td>3</td></tr><tr><td>A08</td><td>Facing payment, currency and platform restrictions</td><td>1</td></tr><tr><td>A09</td><td>Requiring usable devices and infrastructure for AI participation</td><td>5</td></tr><tr><td>A10</td><td>Negotiating institutional restrictions on humanitarian partnerships</td><td>1</td></tr><tr><td>L01</td><td>Needing a practical entry point before usefulness becomes imaginable</td><td>3</td></tr><tr><td>L02</td><td>Learning through experimentation and visible peer practice</td><td>4</td></tr><tr><td>L03</td><td>Seeking training tailored to tasks and the conflict context</td><td>5</td></tr><tr><td>L04</td><td>Combining sustained access with introductory learning</td><td>1</td></tr><tr><td>L05</td><td>Prioritizing foundational capabilities before advanced AI use</td><td>2</td></tr><tr><td>L09</td><td>Developing interest or proposed action through the interview</td><td>2</td></tr><tr><td>L10</td><td>Recognizing limits of knowledge about other volunteers&#x27; use</td><td>2</td></tr><tr><td>L11</td><td>Seeking basic understanding of AI benefits, risks and operation</td><td>5</td></tr><tr><td>N01</td><td>Turning to AI when danger makes other assistance inaccessible</td><td>3</td></tr><tr><td>N02</td><td>Seeking partial action and hope rather than a complete solution</td><td>1</td></tr><tr><td>N03</td><td>Extending humanitarian care beyond food through AI suggestions</td><td>1</td></tr><tr><td>N04</td><td>Building practical trust through reported successful experience</td><td>1</td></tr><tr><td>N05</td><td>Allowing necessity exceptions without endorsing routine reliance</td><td>1</td></tr><tr><td>N06</td><td>Experiencing wartime urgency as pressure to adopt</td><td>2</td></tr><tr><td>N07</td><td>Seeking AI support to interpret conflict-related threats</td><td>1</td></tr><tr><td>N08</td><td>Requiring speed, flexibility and quality together in emergencies</td><td>1</td></tr><tr><td>N09</td><td>Lacking time to use AI despite recognizing its value</td><td>1</td></tr><tr><td>P01</td><td>Organizing human-supplied material while retaining authorship</td><td>4</td></tr><tr><td>P02</td><td>Expanding AI from wording support into digital production</td><td>4</td></tr><tr><td>P03</td><td>Turning meeting records into reusable organizational documentation</td><td>1</td></tr><tr><td>P04</td><td>Translating across donor and volunteer language boundaries</td><td>2</td></tr><tr><td>P05</td><td>Rationing limited AI capacity across tools or accounts</td><td>4</td></tr><tr><td>P06</td><td>Replacing unavailable colleague time with on-demand assistance</td><td>2</td></tr><tr><td>P07</td><td>Extending technical work while retaining specialist authority</td><td>2</td></tr><tr><td>P08</td><td>Proposed AI-assisted digitization of beneficiary records</td><td>2</td></tr><tr><td></td><td></td><td></td></tr></table>

Continued on next page

Table B2 continued

Code Operational code label Units   
P09 Using standard AI structures as scafolding for local content 1   
P10 Leaving digital work with colleagues rather than personally adopting 3   
P11 Letting work demands and role changes shape AI use 3   
P12 Automating document conversion and comparison under human control   
P13 Coordinating practical work through existing tools and human expertise   
P14 Using AI-supported presentation in fundraising work   
S11 Defining wartime AI value through protection and basic needs   
S12 Anticipating continuing responsibilities beyond emergency response 1   
T01 Defining GenAI use by purpose rather than by tool name   
T02 Reading familiarity and interaction style as trust cues 3   
T04 Maintaining selective refusal despite professional capability 3   
T05 Changing adoption boundaries after a change in exposure 1   
T06 Questioning whom connected-user evidence represents 1   
T07 Judging AI by practical humanitarian value rather than output volume 2   
Risks (21 codes)   
A03 Risk — Digital coordination costs fall unevenly on frontline volunteers 2   
E01 Risk — AI-generated solutions can displace situated community knowledge 8   
E02 Risk — AI mediation can distort collectively agreed meanings and decisions 3   
E03 Risk — Generated reporting can detach claims from what actually happened on the ground 4   
E04 Risk — Local language and context gaps shift interpretive repair work onto volunteers 7   
E05 Risk — Formulaic AI language can flatten human voice and lived specificity 2   
E06 Risk — Habitual AI dependence can erode skills, judgment, and authorship 7   
E07 Risk — Donor suspicion of AI authorship can undermine document legitimacy 1   
E08 Risk — Rapidly changing field conditions can invalidate AI-supported judgments 1   
E09 Risk — Polished AI outputs can misfit local language, culture, and material realities 3   
S01 Risk — Uploading identifiers can expose beneficiaries and volunteers to financial or security harm 6   
S02 Risk — One person’s data exposure can compromise teams and interrupt aid delivery 2   
S03 Risk — Devices, connectivity, and digital traces can create physical exposure in conflict 5   
S07 Risk — Shared digital accounts concentrate sensitive data and access dependencies 3   
S08 Risk — Uncertain data exposure and unknown downstream misuse 6   
S09 Risk — AI-assisted documentation and synthetic media can violate dignity, consent, or identity 3   
T03 Risk — Benign intent and model refusals can create a false sense of safety 2   
T08 Risk — Provider incentives and resource costs can conflict with humanitarian priorities 1   
T09 Risk — Location-dependent access and context can make AI behavior uneven across settings 1   
V07 Risk — AI errors can create rework, delay, and service disruption 3   
V08 Risk — Stated data-protection boundaries can coexist with potentially conflicting upload practices 3   
Safeguards (8 codes)   
L12 Safeguard — AI outputs remain drafts until explicit human review and approval 4   
S04 Safeguard — Use paper when digital collection threatens safety or community trust   
S05 Safeguard — Separate organizational visibility from individual identifiability 1   
S06 Safeguard — Keep sensitive operational records outside AI-assisted drafting 1   
V01 Safeguard — Verify AI outputs against known inputs and independent evidence 4   
V02 Safeguard — Increase verification as technical and humanitarian stakes rise 2   
V03 Safeguard — Route AI-assisted outputs through collective human review workflows 2   
V06 Safeguard — Cross-model checking still requires human verification 2   
Accountability (2 codes)   
V04 Accountability — The submitter or designated role remains responsible for AI-assisted work 3   
V05 Accountability — Collective review distributes responsibility before release 4   
Governance risks (1 code)   
L06 Governance risk — Informal AI use across an open volunteer network can outpace safeguards 3

Continued on next page

<table><tr><td>Code</td><td>Operational code label</td><td>Units</td></tr><tr><td colspan="3">Governance safeguards (1 code)</td></tr><tr><td>L07</td><td>Governance safeguard — Co-design AI guidance with volunteers to fit field realities</td><td>2</td></tr><tr><td colspan="3">Governance gaps (1 code)</td></tr><tr><td>L08</td><td>Governance gap — Existing policies can leave AI-specific risks unaddressed</td><td>2</td></tr><tr><td colspan="3">Boundaries of acceptable use (1 code)</td></tr><tr><td>S10</td><td>Boundary — Humanitarian protection and dignity constrain acceptable AI use</td><td>3</td></tr></table>

Note. Units refer to coded passages, not unique participants. Code frequencies were not used to determine thematic importance or select final themes. The code IDs P01–P14 refer to operational codes, not to interview participants (P01–P15) in Section 4.

## B.4 Development of Final Themes

The 83 operational codes were organized into broader thematic groupings during the collaborative analysis workshop. These groupings were not mapped one-to-one to the five final themes. Instead, the research team examined relationships and overlaps between codes, compared experiences across participants, and revisited the interview transcripts to develop the five themes presented in Section 4. Twenty codes were not mapped to a final theme; they remain in the codebook but did not form part of the five themes reported in Section 4.

Table B3. Mapping of operational codes to the five final themes.
<table><tr><td>Final theme</td><td>Relevant operational codes</td></tr><tr><td>4.1 Human support hard to reach</td><td>N01, N02, N03, N04, N05, N06, N07, N08, P06, P07, S11</td></tr><tr><td>4.2 Wartime ground truth</td><td>E01, E04, E06, E08, E09, P01, P07, T07, V01, V02, V06</td></tr><tr><td>4.3 Protection concerns</td><td>S01, S02, S03, S04, S05, S06, S07, S08, S10, T03, V08</td></tr><tr><td>4.4 Humanitarian communication credibility</td><td>P01, P04, P14, E02, E03, E04, E05, E07, E09, S09</td></tr><tr><td>4.5 Unequal access and collective use</td><td>A01, A02, A03, A04, A05, A06, A07, A08, A09, L01, L02, L03, L05, P05, P10, P11, P13, S07, T04, T05, T06, T09, V03, V04, V05</td></tr></table>