# Redacting Personal Data: A Speed and Accuracy Comparison of Regex, Presidio and Hybrid Designs

Sai Yashwant AI Product & Platform Head ManpowerGroup Services India Pvt. Ltd. sai.yashwant@manpowergroup.com

Arjun Sharma AIOps Engineer ManpowerGroup Services India Pvt. Ltd. arjun.sharma@manpowergroup.com

Manoj Kumar Gondimalla AI Operations Lead   
ManpowerGroup Services India Pvt. Ltd.   
manojkumar.gondimalla@manpowergroup.com

Shubham Data Scientist ManpowerGroup Services India Pvt. Ltd. shubham.kumar@manpowergroup.com

## ABSTRACT

Many enterprise projects now pass free text to large language models (LLMs), and that text often contains personally identifiable information (PII) such as names, phone numbers, email addresses, home addresses and dates of birth. Once a prompt is sent, this data is copied into provider systems, application logs and observability traces. Redacting PII before the call is an obvious control, yet teams seldom have numbers on what it costs in latency or what it lets through. This paper compares four redaction designs that any LLM pipeline can adopt: a pure regular-expression (regex) redactor, Microsoft Presidio running on a spaCy model, a hybrid in which Presidio finds only person names while regex handles structured identifiers, and a fast hybrid that calls spaCy’s named-entity component directly. All four share a known-entity rule, which removes the person the system already knows it is handling, and a consistent placeholder scheme, so the LLM can still tell one person from another after redaction. We evaluate the designs on a sample workflow, which we call ProfileAssist, that turns resumes into structured profiles, using 100 synthetic resumes (523 characters on average, with 900 PII values and 600 business terms that must survive). Regex ran at a 0.20 ms median but caught only 88.9% of PII, because it cannot find names that are not known in advance, such as the references listed in a resume. Presidio, hybrid and fast hybrid each caught 100% of PII and kept 100% of business terms, at median latencies of 37.59 ms, 30.23 ms and 13.48 ms. Measured against a 9,120 ms median LLM call, even the slowest option adds only 0.41% to the request. We therefore recommend the fast hybrid as a default for LLM pipelines that handle personal data, with regex kept as a safety net rather than as the only control.

Keywords PII redaction, Microsoft Presidio, regular expressions, named entity recognition, spaCy, LLM privacy, LLMOps, data minimization

## 1 Introduction

Large language models are now part of many enterprise projects. Teams use them to summarize documents, classify requests, extract fields from free text and draft replies. In a large share of these projects the input contains personally identifiable information (PII): a support ticket names the customer and gives a phone number, an HR document lists an employee’s address, and a resume holds a person’s contact details along with the names of their references. The value of the LLM is real, but so is the risk, because every prompt that carries PII spreads it further.

That data does not stay in one place once the call is made. It travels to the model provider, it is written into application logs, and in a well-run LLMOps setup it is also captured by an observability platform, where prompts and responses are stored as traces for debugging and evaluation. Each copy is one more place where personal data must be protected, retained and finally deleted under rules such as the GDPR [1], the CCPA [2] and India’s Digital Personal Data Protection Act [3]. Research on training-data extraction and PII leakage shows that what goes into a language model, or into a dataset later used for tuning, can come back out [4, 5]. For this reason the OWASP Top 10 for LLM applications lists sensitive information disclosure among the leading risks of LLM systems [6].

The simplest defense is to redact PII before it ever reaches the model. Here, however, engineering teams face a familiar trade-of. Regular expressions are fast and predictable, and they handle structured identifiers such as emails and phone numbers well, but they cannot recognize a name they have never seen. Machine-learning tools such as Microsoft Presidio [7], which relies on spaCy named entity recognition (NER) [8], can find unseen names, yet they cost more CPU time and take seconds to load. In a request path that already waits on an LLM, it is not obvious whether that extra cost matters, and so teams often decide on instinct rather than on measurement.

This paper replaces that instinct with numbers. We compare four redaction designs that can be placed in front of any LLM call: regex alone, Presidio, a hybrid of Presidio names plus regex, and a fast hybrid that uses only spaCy’s NER component plus regex. All four reuse one redaction core with a known-entity rule and consistent placeholders. We report accuracy both as PII recall and as business-term retention, so that over-redaction is measured as well as leakage, and we set load time and latency against the median latency of an LLM call. We then turn the results into practical guidance for any team that runs LLM pipelines over personal data.

To keep the measurements concrete, we ground them in a sample workflow that we call ProfileAssist. In ProfileAssist, a resume is passed to an LLM that returns a structured profile with skills, job titles and a short summary. We chose resumes because they are among the densest sources of PII a company handles, and because they show a pattern that recurs in many other projects. The text is about one person the system already knows, yet it also mentions other people, such as references, whom the system does not know. The same pattern appears in support tickets that mention agents, in case notes that mention relatives, and in email threads that copy colleagues. The example, in other words, uses resumes, but the problem is general.

ProfileAssist and all names in this paper are illustrative. All benchmark text is synthetic, and the only measured external figure is the median LLM latency, taken from traces of a representative LLM workload.

## 2 Related Work

Automatic removal of personal identifiers from free text has a long history in clinical informatics. Sweeney’s Scrub system combined pattern templates with lists of known names to clean medical records [9], and later rule-based tools, such as the de-identification software of Neamatullah et al., relied on regular expressions, dictionaries and context rules [10]. These early systems already showed that patterns work well for formatted identifiers but struggle with names, which vary too much to list. Shared tasks such as the 2014 i2b2/UTHealth challenge then pushed the field toward machine learning, and hybrid systems that mixed rules with statistical models were often near the top of the rankings [11], and recurrent networks later reached high recall on patient notes without hand-crafted features [12].

Outside healthcare, most of the progress came from general-purpose NER. The BiLSTM-CRF model of Lample et al. [13] and the transformer models that followed made person detection reliable enough to package in open libraries. spaCy ofers fast, production-ready pipelines in which NER is one component next to a tagger, a parser and a lemmatizer [8]. Presidio builds on such pipelines and adds pattern recognizers, checksum validation and context words, together with an anonymizer that replaces the detected spans [7]. Lison et al. reviewed text anonymization more broadly and argued that detection quality must always be weighed against how much useful content survives [14]. The Text Anonymization Benchmark (TAB) of Pilán et al. made the same point with an evaluation framework that scores privacy protection and utility side by side [15], and Yermilov et al. showed that pseudonymization, rather than blanket masking, helps downstream language tasks keep working [16].

Our work is narrower than these studies, but more operational. We do not propose a new detector. Instead, we take tools that teams actually deploy, run them on text shaped like a real LLM prompt, and measure what matters in a live request path: load time, tail latency, leakage and over-redaction. Few published studies set the cost of redaction against the cost of the LLM call it protects. Guidance such as NIST SP 800-122 [17] and the NIST AI Risk Management Framework [18] asks organizations to keep PII exposure to a minimum, and this paper ofers one measured way of doing so that is not tied to a single domain.

## 3 Problem Setting

## 3.1 Redaction in a Generic LLM Pipeline

In a typical LLM pipeline, a service gathers some text, wraps it in a prompt and sends the prompt to an LLM. Often the service already knows whose data it is handling, for example the customer behind a ticket or the employee behind a record, because that identity comes from its own systems. We call this person the known entity. The redactor sits between prompt assembly and the LLM client, as Figure 1 shows. Because redaction happens before the call, the model provider, the application logs and the trace store all receive the same placeholder text. The LLM sees tokens such as [PERSON\_1] or [PHONE\_NUMBER\_2] in place of real values, and the service can map them back afterwards if its output needs them.

![](images/51a25ce3e0d80da891aa1cdb95749ab0c0e265f96e97fe2fba60f62ddabd97a2.jpg)  
Figure 1: How redaction works in a generic LLM pipeline, shown with values from the ProfileAssist example. Three detectors find PII spans: the known-entity rule, regex patterns, and spaCy NER for names that are not known in advance (the regex-only design skips this step). The spans are merged and replaced with consistent placeholders, so only placeholder text crosses the service boundary to the LLM, the logs and the traces.

## 3.2 Running Example: ProfileAssist

ProfileAssist is a sample service that takes a person’s resume and asks an LLM for a structured profile with skills, job titles, a short summary and a suggested job family. The known entity is the resume owner, and the prompt names them on a line of the form KNOWN\_SUBJECT: <name> before the resume, which sits inside <subject\_data> tags. As the latency yardstick we use a median (P50) LLM call time of 9,120 ms, observed in traces of a representative LLM workload. Whatever redaction adds should be small next to this figure.

## 3.3 What Must Be Removed and What Must Stay

Two kinds of error matter in any redaction setting. A leak happens when a PII value reaches the model untouched. Over-redaction happens when business content is masked by mistake, for example when a skill such as “pallet jack” is tagged as a person. A leak is a privacy failure, while over-redaction is a quality failure, because the LLM cannot use content it cannot see. Which terms count as business content depends on the project, and Table 1 lists the PII types and business terms for our example. City and state stay in the text on purpose: they are useful for the profile and, on their own, they do not identify a person. The last column shows which items are counted in the accuracy scores of Section 5.

Table 1: PII types removed and business terms kept in the ProfileAssist example, with values from the synthetic corpus.
<table><tr><td>Category</td><td>Item</td><td>Example</td><td>Scored</td></tr><tr><td>PII</td><td>Subject name (full name and parts)</td><td>Andre Kowalczyk, ANDRE KOWALCZYK, Andre</td><td>Yes</td></tr><tr><td>PII</td><td>Email address and phone number</td><td>andre.kowalczyk0@gmail.com, (901) 555-4821</td><td>Yes</td></tr><tr><td>PII</td><td>Street address and ZIP code</td><td>4521 Maple Ave, 38103</td><td>Yes</td></tr><tr><td>PII</td><td>Date of birth</td><td>04/17/1991</td><td>Yes</td></tr><tr><td>PII</td><td>Reference name and phone number</td><td>Greta Fontaine, 412-555-7310</td><td>Yes</td></tr><tr><td>PII</td><td>Profile URL and US SSN</td><td>linkedin.com/in/andrekowalczyk, 123-45-6789</td><td>No</td></tr><tr><td>Keep</td><td>City, state and employer</td><td>Memphis, TN, Northwind Logistics</td><td>Yes</td></tr><tr><td>Keep</td><td>Skills and certifications</td><td>forklift operation, OSHA 10, pallet jack</td><td>Yes</td></tr></table>

## 4 Redaction Approaches

All four designs share one core and difer only in how they find spans, so any diference in the results comes from detection and not from formatting.

## 4.1 Shared Redaction Core

The core has three parts. The first is a set of compiled regular expressions, listed in Table 2 [19]. Some patterns use a named group called pii so that only part of a match is replaced. The ZIP pattern, for instance, matches “TN 38103” to make sure the number follows a state code, yet it replaces only “38103” and leaves the state readable. The date-of-birth pattern works in the same way and replaces a date only when it follows “DOB” or “Date of birth”, so that employment years such as 2021 are left alone. Because our example data is US-based, the patterns target US formats; a project in another region would swap in its own, for instance Indian phone numbers or PAN and Aadhaar numbers.

Table 2: Regular-expression patterns in the shared core and the designs that use them.
<table><tr><td>Entity</td><td>What the pattern matches</td><td>Used by</td></tr><tr><td>EMAIL ADDRESS</td><td>local part, @, domain and a top-level domain of two or more letters</td><td>Regex, Hybrid, Fast hybrid</td></tr><tr><td>PROFILE_URL</td><td>linkedin.com/in/. .. or github.com/. . . , with optional scheme and www</td><td>Regex, Hybrid, Fast hybrid</td></tr><tr><td>US_SSN</td><td>three, two and four digits joined by hyphens</td><td>Regex, Hybrid, Fast hybrid</td></tr><tr><td>PHONE_NUMBER</td><td>ten-digit US number with optional +1, brackets, dots, dashes or spaces</td><td>Regex, Hybrid, Fast hybrid</td></tr><tr><td>STREET_ADDRESS</td><td>house number, optional direction, one to three capitalized words, street suffix, optional unit</td><td>All four</td></tr><tr><td>US_ZIP</td><td>five- or nine-digit ZIP after a two-letter state code (only the ZIP is replaced)</td><td>All four</td></tr><tr><td>DATE OF BIRTH</td><td>date after “DOB&quot; or “Date of birth&quot; (only the date is replaced)</td><td>All four</td></tr></table>

The second part is the known-entity rule. The redactor reads the name of the known entity, which in our example comes from the KNOWN\_SUBJECT line, and searches the whole text, ignoring case, for the full name and for each part of it that is at least two characters long, starting with the longest terms. In the example this one rule catches the uppercase name at the top of a resume and the first name used later in the summary. It is cheap and exact, and it covers the person who appears most often in the text.

The third part merges spans and writes placeholders. Spans from all detectors are sorted by start position, longer spans first, and any span that overlaps one already taken is skipped. Each distinct value, compared in lower case, gets its own numbered token for its entity type, so “ANDRE KOWALCZYK” and “Andre Kowalczyk” both become [PERSON\_1], while “Andre” on its own becomes [PERSON\_2]. This consistency matters to the LLM, which can still see that the same person appears twice and that the reference is somebody else. Plain masking with a single [REDACTED] token would hide exactly that.

## 4.2 Regex Redactor

The regex redactor uses the known-entity rule and all seven patterns, and nothing else. It loads in under a millisecond and never needs a model file. Its weakness, however, is built into the design: a name that does not belong to the known entity, such as a reference in a resume or a colleague copied on an email, matches no pattern and passes straight through to the model.

## 4.3 Presidio Redactor

The Presidio redactor runs Presidio’s analyzer engine on the spaCy en\_core\_web\_lg model and asks for six entity types (PERSON, EMAIL\_ADDRESS, PHONE\_NUMBER, US\_SSN, CREDIT\_CARD and URL) with a score threshold of 0.4. Presidio has no built-in recognizers for US street addresses, ZIP codes or dates of birth in the form we need, so the three shared patterns are added on top, along with the known-entity rule.

Two small fixes turned out to be necessary. First, spaCy sometimes lets a person entity run across a line break, for example from the name header into the address below it, so each PERSON span is cut at the first newline. Second, spaCy tags some domain words as people, in our example section headings or “jack” in “pallet jack”, so a short stop-list drops any PERSON span that contains one of them. Each project needs its own stop-list for the same reason.

## 4.4 Hybrid Redactor

The hybrid redactor keeps Presidio but asks it for PERSON only. Emails, phone numbers, URLs, SSNs and addresses come from the regex set, which already handles them well. The idea is simple: use the model only where a model is really needed, and save the cost of Presidio’s other recognizers.

## 4.5 Fast Hybrid Redactor

The fast hybrid takes this idea one step further and drops Presidio altogether. It loads the same spaCy model but disables every component except NER, namely the shared token-to-vector layer, the tagger, the parser, the attribute ruler, the lemmatizer and the sentence segmenter. Person entities are read straight from the spaCy document, with the same newline cut and stop-list as before, and everything else comes from regex. Since the parser and the tagger do a lot of work that PERSON detection never uses, removing them should cut most of the cost per document.

## 5 Experimental Setup

## 5.1 Synthetic Corpus

For the ProfileAssist example we used synthetic data, so that the ground truth is exact and no real personal data is involved. A generator with a fixed random seed produced 100 resumes for a warehouse and logistics role, a common entry-level profile. Each resume combines a first and a last name, each drawn from ten options, one of five reference names and one of three US city, state and ZIP combinations. The names are deliberately varied in origin (Kowalczyk, Haddad, Okafor and Lindqvist, for example), because NER models often do worse on names that are rare in their training data. Every document follows the layout shown later in Example 1, with a KNOWN\_SUBJECT line, contact details, a summary that repeats the first name, experience, skills and one reference. The average length is 523 characters, which gives 900 scored PII values (nine per document) and 600 business terms that must survive (six per document). The same script can also redact the input column of an exported trace file, but that mode has no ground truth, so we used it only for spot checks.

## 5.2 Metrics

We measure both accuracy and speed. For accuracy, a PII value counts as leaked if it can still be found in the redacted text as a whole word, ignoring case. With L leaked values out of P scored PII values, recall, which we call “PII caught”, is

$$
\mathrm { P I I } \mathrm { c a u g h t } = 1 - { \frac { L } { P } }\tag{1}
$$

Retention measures the opposite failure. With K business terms still present, exactly as written, out of B terms in total, the share kept is

$$
\mathrm { K e p t } = { \frac { K } { B } }\tag{2}
$$

For speed, we time each call to the redactor with Python’s high-resolution perf\_counter and summarize the 300 samples as the median (P50) and the 95th percentile (P95). Load time is the time taken to build the redactor, including any model loading. Finally, to judge whether the cost matters, we divide the redactor’s median by the median latency of the LLM call:

$$
\mathrm { L L M } \mathrm { s h a r e } = \frac { P 5 0 _ { \mathrm { r e d a c t o r } } } { P 5 0 _ { \mathrm { L L M } } }\tag{3}
$$

## 5.3 Procedure

All four designs ran one after another in a single Python process on one developer workstation, in the order regex, Presidio, hybrid and fast hybrid, using the CPU only. For each design the script first timed the constructor, then redacted all 100 documents three times while timing each call, and finally ran one more pass to score recall and retention, giving 300 timed calls per design.

## 6 Results

## 6.1 Accuracy

Table 3 shows the main results. The three designs that use a model caught every one of the 900 PII values, while regex caught 800 of them, or 88.9%. All four designs kept all 600 business terms, so none of them over-redacted on this corpus. That last point is worth noting, because the model-based designs would have masked “pallet jack” without the stop-list described in Section 4.3.

Table 3: Load time, latency and accuracy of the four designs (100 documents, 300 timed calls each).
<table><tr><td>Design</td><td>Load (ms)</td><td>P50 (ms)</td><td>P95 (ms)</td><td>PII caught</td><td>Kept</td><td>LLM share</td></tr><tr><td>Regex</td><td>0.8</td><td>0.20</td><td>0.49</td><td>88.9%</td><td>100.0%</td><td>0.002%</td></tr><tr><td>Presidio</td><td>5,021.2</td><td>37.59</td><td>49.96</td><td>100.0%</td><td>100.0%</td><td>0.41%</td></tr><tr><td>Hybrid</td><td>1,416.7</td><td>30.23</td><td>43.42</td><td>100.0%</td><td>100.0%</td><td>0.33%</td></tr><tr><td>Fast hybrid</td><td>1,698.6</td><td>13.48</td><td>17.10</td><td>100.0%</td><td>100.0%</td><td>0.15%</td></tr></table>

The 100 values that regex missed are easy to explain once the result is broken down by PII type. Regex caught all 100 subject first names, last names, emails, phone numbers, addresses, ZIP codes, dates of birth and reference phone numbers, and it missed all 100 reference names. The subject is covered by the known-entity rule and the structured fields are covered by patterns, but a reference such as “Greta Fontaine” is neither known in advance nor shaped like anything a pattern can match. The reference’s phone number was still caught, which makes the leak a little less harmful, but a name together with a job title and an employer is often enough to find a person.

## 6.2 Latency

Figure 2 plots P50 and P95 latency on a log scale. Regex was about two orders of magnitude faster than any model-based option, at 0.20 ms median and 0.49 ms P95. Presidio was the slowest, at 37.59 ms and 49.96 ms. Limiting Presidio to PERSON cut its median by about 20%, to 30.23 ms, which shows that Presidio’s other recognizers do cost something, although most of the time is still spent inside the full spaCy pipeline. The fast hybrid, which runs only the NER component, reached 13.48 ms median and 17.10 ms P95. That is 2.8 times faster than Presidio and 2.2 times faster than the hybrid, with the same recall. Its tail was also tight, since the gap between P50 and P95 was 3.6 ms against 12.4 ms for Presidio.

![](images/6506dacbc3d91f95921fa4fd7208c1cddac549e5d63c414c787530a27c0fe3bf.jpg)  
Figure 2: Median (P50) and 95th-percentile (P95) redaction latency per document. Note the log scale.

Seen as throughput on a single core, the fast hybrid can process about 74 documents of this size per second, against about 27 for Presidio, a gap that matters more in batch jobs than in a single request.

## 6.3 Cost Relative to the LLM Call

For a design decision, the last column of Table 3 matters most. Even Presidio, the slowest option, adds only 0.41% to a request whose LLM step takes 9,120 ms at the median. The fast hybrid adds 0.15%, and regex adds about 0.002%, which is efectively zero. Reading the accuracy and latency columns together, regex is fast but leaky, while the three model-based designs all reach 100%, with the fast hybrid the cheapest of them. Put plainly, the step from 88.9% to 100% recall costs about 13 ms on a request that already takes more than nine seconds.

## 6.4 Load Time

Load time tells a diferent story. Regex compiled its patterns in 0.8 ms, whereas Presidio took 5,021 ms to start, the hybrid 1,417 ms and the fast hybrid 1,699 ms. These costs are paid once per process rather than once per request, so in a long-running service they hardly matter. They do matter for serverless functions with frequent cold starts, where a five-second start-up would be felt by users.

Example 1: Fast hybrid output for one synthetic resume (hybrid and Presidio outputs are the same apart from the URL label).

KNOWN\_SUBJECT: [PERSON\_1]   
<subject\_data> === RESUME ===   
[PERSON\_1]   
[STREET\_ADDRESS\_1], Memphis, TN [US\_ZIP\_1]   
[EMAIL\_ADDRESS\_1] | [PHONE\_NUMBER\_1] | [PROFILE\_URL\_1]   
DOB: [DATE\_OF\_BIRTH\_1]   
SUMMARY   
Warehouse associate in Memphis. [PERSON\_2] prefers first shift.   
EXPERIENCE   
Forklift Operator - Northwind Logistics (2021 - present)   
- Operated pallet jack and reach truck; cycle counting in WMS.   
SKILLS   
forklift operation, OSHA 10, pallet jack   
REFERENCES   
[PERSON\_3], Shift Supervisor - [PHONE\_NUMBER\_2]   
</subject\_data>

## 6.5 A Worked Example

Example 1 shows the fast hybrid output for one synthetic resume. The subject’s full name appears twice and in two diferent cases, yet both map to [PERSON\_1], and the first name on its own maps to [PERSON\_2]. The reference becomes [PERSON\_3], clearly a diferent person, while city, state, employer and skills all stay readable. The regex output for the same document is identical except for the last line, which reads “Greta Fontaine, Shift Supervisor - [PHONE\_NUMBER\_2]”. The Presidio output difers in one further detail: the LinkedIn path is labeled [URL\_1] by Presidio’s generic URL recognizer instead of [PROFILE\_URL\_1].

## 7 Discussion

## 7.1 Why Regex Alone Is Not Enough

At first sight regex looks attractive. It is fast, it needs no model and it is easy to audit. Our results, however, show its limit clearly. In most real text the known entity is only one of the people mentioned. A resume names references and former managers, a support ticket names agents and family members, and an email thread names everyone in copy. None of these names are known in advance, and none of them follow a pattern, so a regex-only design will leak them every time. Making the regex cleverer, for instance by matching any two capitalized words, would quickly start to mask job titles, company names and place names, which is precisely the over-redaction that the retention metric is there to catch.

## 7.2 Why the Fast Hybrid

The fast hybrid gives the model only the job a model is good at, which is finding names, and leaves structured identifiers to patterns that are exact and cheap. Because it calls spaCy’s NER directly, it avoids both Presidio’s extra recognizers and the parts of the spaCy pipeline that PERSON detection does not use. The result is full

recall at roughly a third of Presidio’s cost, with fewer moving parts: one model and a list of patterns instead of an analyzer framework with its own registry, context logic and dependencies.

Presidio still has its place. It ofers checksum validation, context-aware scoring, more languages and a ready-made anonymizer, and teams that need those features may well accept its higher cost, which in ou setting was still under half a percent of the LLM call.

## 7.3 Practical Recommendations

Based on these results, we suggest six practices for teams that run LLM pipelines over personal data. First, redact before the LLM call and not only before logging, so that the provider, the logs and the traces all see the same safe text. Second, use a known-entity rule whenever the system already knows whose data it is handling, since it is the cheapest high-recall control available. Third, use NER for person names only and leave structured identifiers to regex. Fourth, load the model once at process start and keep it warm, so that the load cost is never paid per request. Fifth, measure both leakage and retention on a labeled set before every change to patterns or models, and track redaction latency next to LLM latency in the same trace. Sixth, keep regex as a fallback: if the NER model fails to load, a service that falls back to regex-only redaction leaks some names, but a service that skips redaction leaks everything.

## 7.4 Applying the Approach to Other Projects

Nothing in the four designs is specific to resumes. To reuse them in another project, a team changes three things. First, the source of the known entity, such as the customer behind a ticket, the employee behind an HR request or the patient behind a clinical note. Second, the pattern set, which should match the identifier formats of the project’s region and domain. Third, the stop-list and the list of business terms that must survive, which are what the retention metric checks. The evaluation method stays the same: build a small labeled set that looks like the project’s prompts, measure PII caught and terms kept, and compare redaction latency with the LLM latency from the project’s own traces.

## 8 Threats to Validity

Several limits apply to these findings. First, the results come from one sample workflow, and the corpus is synthetic, short and uniform. Real text is longer, often several thousand characters, and varies widely in layout, language and the kinds of PII it holds, so the 100% recall of the model-based designs should be read as a result on this corpus and not as a general claim; other domains need their own labeled set before the numbers can be carried over. On messy real text we expect some misses from all of the designs. Second, the name pool is small, so the test does not probe how NER copes with rare names, initials or names that are also common words. Third, profile URLs and SSNs are redacted but not scored. Fourth, all timings came from one workstation and one process, with the designs run in a fixed order. Presidio was loaded first, so the later designs benefited from modules already imported and model files already cached, which probably flatters their load times. Finally, the 9,120 ms LLM median comes from one workload over one period. A faster model would raise the relative cost of redaction; even so, if spaCy’s cost grows roughly in line with text length, the fast hybrid on a document ten times longer would take around 135 ms, which is still about 1.5% of the current LLM median.

## 9 Conclusion and Future Work

This paper compared four ways to remove PII from text before it reaches an LLM, using ProfileAssist, a sample resume-to-profile workflow, as a worked example. Regex alone was by far the fastest, but it missed every name that was not known in advance and caught 88.9% of PII. Presidio, a Presidio-plus-regex hybrid and a fast hybrid built on spaCy’s NER component all caught 100% of PII and kept 100% of business terms. Among these, the fast hybrid was the quickest at a 13.48 ms median, 2.8 times faster than Presidio, and it added only 0.15% to a 9,120 ms LLM call. The broader lesson, which holds beyond this example, is that the cost of redaction should be judged against the call it protects; seen that way, a few extra milliseconds are a small price for not sending personal data to a third party. Future work will apply the same benchmark to other LLM workloads that handle personal data, run it on de-identified trace exports with human-labeled ground truth, extend it to longer and non-US text, add transformer-based NER models to the comparison, and test whether placeholder tokens change the quality of the LLM’s output.

## References

[1] European Parliament and Council of the European Union, “Regulation (EU) 2016/679 (General Data Protection Regulation).” Oficial Journal of the European Union, L 119, pp. 1–88, 2016.

[2] State of California, “California Consumer Privacy Act of 2018.” Cal. Civ. Code § 1798.100 et seq., 2018.

[3] Government of India, “The Digital Personal Data Protection Act, 2023.” Act No. 22 of 2023, Ministry of Law and Justice, New Delhi, 2023.

[4] N. Carlini et al., “Extracting training data from large language models,” in Proc. 30th USENIX Security Symposium, pp. 2633–2650, 2021.

[5] N. Lukas, A. Salem, R. Sim, S. Tople, L. Wutschitz, and S. Zanella-Béguelin, “Analyzing leakage of personally identifiable information in language models,” in Proc. IEEE Symp. Security and Privacy (SP), pp. 346–363, 2023.

[6] OWASP Foundation, “OWASP Top 10 for Large Language Model Applications.” version 2025, OWASP GenAI Security Project, 2025.

[7] Microsoft, “Presidio: Data protection and de-identification SDK.” GitHub repository. [Online]. Available: https://github.com/microsoft/presidio.

[8] M. Honnibal, I. Montani, S. Van Landeghem, and A. Boyd, “spaCy: Industrial-strength natural language processing in Python.” Zenodo, doi: 10.5281/zenodo.1212303, 2020.

[9] L. Sweeney, “Replacing personally-identifying information in medical records, the Scrub system,” in Proc. AMIA Annual Fall Symposium, pp. 333–337, 1996.

[10] I. Neamatullah et al., “Automated de-identification of free-text medical records,” BMC Medical Informatics and Decision Making, vol. 8, 2008. Art. no. 32.

[11] A. Stubbs, C. Kotfila, and Ö. Uzuner, “Automated systems for the de-identification of longitudinal clinical narratives: Overview of 2014 i2b2/UTHealth shared task Track 1,” Journal of Biomedical Informatics, vol. 58, pp. S11–S19, 2015. suppl.

[12] F. Dernoncourt, J. Y. Lee, Ö. Uzuner, and P. Szolovits, “De-identification of patient notes with recurrent neural networks,” Journal oftheAmerican Medical InformaticsAssociation, vol. 24, no. 3, pp. 596–606, 2017.

[13] G. Lample, M. Ballesteros, S. Subramanian, K. Kawakami, and C. Dyer, “Neural architectures for named entity recognition,” in Proc. NAACL-HLT, pp. 260–270, 2016.

[14] P. Lison, I. Pilán, D. Sánchez, M. Batet, and L. Øvrelid, “Anonymisation models for text data: State of the art, challenges and future directions,” in Proc. 59th Annu. Meeting of the Association for Computational Linguistics (ACL-IJCNLP), pp. 4188–4203, 2021.

[15] I. Pilán, P. Lison, L. Øvrelid, A. Papadopoulou, D. Sánchez, and M. Batet, “The text anonymization benchmark (TAB): A dedicated corpus and evaluation framework for text anonymization,” Computational Linguistics, vol. 48, no. 4, pp. 1053–1101, 2022.

[16] O. Yermilov, V. Raheja, and A. Chernodub, “Privacy- and utility-preserving NLP with anonymized data: A case study of pseudonymization,” in Proc. 3rd Workshop on Trustworthy Natural Language Processing (TrustNLP), 2023.

[17] E. McCallister, T. Grance, and K. Scarfone, “Guide to protecting the confidentiality of personally identifiable information (PII),” NIST Special Publication 800-122, National Institute of Standards and Technology, Gaithersburg, MD, USA, 2010.

[18] National Institute of Standards and Technology, “Artificial intelligence risk management framework (AI RMF 1.0),” NIST AI 100-1, National Institute of Standards and Technology, Gaithersburg, MD, USA, Jan. 2023.

[19] J. E. F. Friedl, Mastering Regular Expressions. Sebastopol, CA, USA: O’Reilly Media, 3rd ed., 2006.