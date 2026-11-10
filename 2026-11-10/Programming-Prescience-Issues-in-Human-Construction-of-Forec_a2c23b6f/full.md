# Programming Prescience: Issues in Human Construction of Forecasting Bots

Priya Dhawka   
pdhawka@uw.edu   
University of Washington   
Seattle, Washington, USA

Benjamin Wilson ben@metaculus.com Metaculus Santa Monica, California, USA

Andrew Head head@engineering.upenn.edu University of Pennsylvania Philadelphia, Pennsylvannia, USA

## Abstract

AI models’ growing capacity for complex reasoning has made them attractive tools for forecasting, or the systematic prediction of the future based on past and present data. However, should AI forecasting systems be adopted, human builders will shoulder the work of building and evaluating them. In this paper, we explore what this work looks like with a qualitative interview study with 15 builders of forecasting bots, sampled from among successful participants in a recent AI forecasting competition. Our analysis surfaces the challenges around (1) tuning a bot to acquire factual research, (2) implementing coherent reasoning, and (3) understanding how well one’s bot actually does. During their work, builders frequently introduced complexity that did not pay of, and tested in ways that were opportunistic and piecemeal. Our detailed account of their challenges helps clarify what problems should be targeted in up coming innovations in process, tools, and human-AI teaming for forecasting.

## CCS Concepts

• Human-centered computing → Empirical studies in HCI.

## Keywords

Forecasting, Reasoning agents, Evaluation, Development

## ACM Reference Format:

Priya Dhawka, Benjamin Wilson, and Andrew Head. 2027. Programming Prescience: Issues in Human Construction ofForecasting Bots. In Proceedings ofMake sure to enter the correct conference title from your rights confirmation email (Conference acronym ’XX). ACM, New York, NY, USA, 11 pages. https: //doi.org/XXXXXXX.XXXXXXX

## 1 Introduction

Forecasting is an essential method for predicting and communi cating uncertain outcomes about the future. It is widely used to motivate decisions across myriad domains like election forecasting, public policy planning, business investments and financial trading among others [5, 8, 9]. For example, within the US, major election cycles often include political polling and probabilistic election forecasts by expert human forecasters who make predictions about election outcomes to support public sensemaking around voting [8]. Human forecasting is also a costly and challenging efort involving stages of information gathering, reasoning about collected information and context, and producing a final forecast while reducing uncertainty at various stages [5, 30, 31].

Recently, there have been indications that AI may become a viable tool to support forecasting. Halawi et al. [11] recently demonstrated language models achieving competitive performance with a human crowd on forecasting questions. Schoenegger et al. [26] observed that when people were given access to an AI during forecasting tasks, they achieved higher performance than when completing them alone. Models appear to be improving in their forecasting ability. The Forecasting Research Institute recently published ForecastBench as a benchmark for AI ability on the task [16], and its public leaderboard posts continual improvements among the leading LLMs [1].

If AI is incorporated into forecasting, then the task of constructing and evaluating forecasting bots may become a significant human task. Aspects of this work are likely to diverge significantly from conventional human forecasting. For instance, a bot builder will have to figure out how to help a bot do background research over pools of sources that are more restricted to the bot than to themselves. Achieving good performance in forecasting may involve examining lengthy system outputs and translating them into ideas for productive changes.

The human-computer interaction research community has developed better approaches to predicting uncertain futures by refining and describing techniques for visualizing uncertainty [6, 10, 12, 14, 17, 20, 38] and providing tools for extrapolating and making decisions about the future with time series data [6, 18, 29]. We seek to complement this work with a deeper understanding of the task of building AI machinery that makes forecasts.

Recent community research has begun to describe this kind of work. For instance, one AI forecasting competition has made public surveys of bot builders’ experiences. Among other things, it has revealed the efort involved in creating competitive bots (anywhere from hours to months) and identified features of bots that correlate with better performance (like use of web scraping) [36]. We seek to provide researchers, tool-builders, and practitioners with greater color and depth around the human process of building forecasting AIs, and where the current ceilings are in this kind of work. What specifically are people trying to build? Where do they get their ideas? What bottlenecks are they hitting, both in regard to the underlying technology and their ability to make it do what they want? What workarounds are they devising?

To paint this picture, we conducted semi-structured interviews with 15 builders of forecasting bots. Builders were sampled from competitors in a recent AI forecasting tournament. They were all to some extent successful builders by nature of being prizewinners in the competition. They attempted a variety of implementation techniques, invested varying amounts of time and money, and ulti mately achieved diferent levels of success. Our qualitative analysis provides a rich account of the current human work in building forecasting bots across the tasks of research, reasoning, and eval uation. Motivated by our observations, we discuss opportunities to improve process, tools, and human-AI coordination to improve AI-assisted forecasting.

## 2 Related Work

We describe relevant concepts and prior research at the intersection of human forecasting, AI-assisted forecasting, general barriers to AI agent use, and the challenges of communicating uncertainty in predictions.

## 2.1 Human Forecasting

Forecasting is the systematic process of making predictions informed by historical and present data. The forecasting literature describes best practices on how forecasters should produce reliable predictions [30]. Notably, work from the Good Judgment Project identifies the reasoning practices that separate accurate human forecasters from the rest, such as decomposing questions, seeking contradicting evidence, and updating incrementally on new infor mation [19, 30, 31], whereas the forecasting methodology literature more broadly catalogs principles around prioritizing simplicity in methods, combining forecasts, and structuring judgment for producing and evaluating reliable forecasts [5]. Bot builders in our sample attempt to embed several of these best practices in their bots, albeit to varying levels of success.

## 2.2 Forecasting with AI

Forecasting with AI comes with unique challenges that are understudied in current human-computer interaction and human-AI interaction research. Existing work by Schoenegger et al. [26] investigated the potential of using AI to augment human forecasts, finding that AI-augmented forecasts match and may even exceed performance by human forecasters. More recently, Paleka et al. [21] identified two categories of pitfalls in evaluating AI forecasters relevant to our study: (1) users struggle to trust evaluation results due to temporal leakage, where models may access information published after an event’s resolution date rather than genuinely predicting the outcome, and (2) users struggle to extrapolate from test results to real-world forecasting scenarios. Participants in our study also raised the temporal leakage concern, and we further identify the extrapolation problem as a limitation specific to tournament-style forecasting.

Prior work in this space also includes benchmarking resources for AI forecasting systems. Karger et al. [16] proposed a benchmarking dataset of forecast questions to evaluate the accuracy of AI-generated predictions while other benchmarks for AI-assisted, tournament-style forecasting relevant for our study include Halawi et al.’s work approaching human-level forecasting accuracy with language models [11], ForecastQA’s temporal event-forecasting question-answering challenge [13], work examining whether language models can use forecasting strategies [23], Autocast++ that consists of retrieval-augmented approaches to the Autocast benchmark [37], and the original Autocast dataset of real-world forecasting questions [39]. Considering the nature of forecasting where there is a time delay between forecasts being generated and questions being resolved in real time, forecasters often require access to testing benchmarks that consist of historical questions that have already concluded as well as future questions that have yet to resolve. Thus, the main limitation of this existing benchmarking literature is that these datasets often have fixed date cutofs, which is an issue that our participants also raised as forecasting questions often resolved after the training cutof for most frontier models.

A separate line of research also compares the performance of AI forecasters with human forecasters, with recent studies tracking AI’s growing forecasting capabilities [1, 4, 32, 33]. For instance, Schoenegger and Park [25] and Abolghasemi et al. [3] both evaluated large language models’ predictive accuracy against human judgmental forecasting, while Schoenegger et al. [27] compared the performance of an ensemble of LLMs against the aggregated accuracy of human forecasting crowds, finding that aggregated LLM predictions can at least rival the accuracy from human crowd forecasting tournaments. Additionally, ongoing research by Metaculus, a popular forecasting platform, has explored the problem of constructing an AI forecasting bot by surveying bot builders on their platform about the technical challenges of the process, documenting the forecasting strategies that resulted in the best performing bots. Metaculus’ findings are publicly available as survey results and an accompanying guide for bot builders [2]. In comparison, our study builds on this prior work by more closely investigating the strategies bot builders use to design successful forecasting bots and the workarounds they implemented to address the challenges they encountered in the process.

## 2.3 Barriers to AI Agent Use

The literature on obstacles to AI agent adoption is large, spanning HCI, software engineering, and AI systems research, but we summarize a small number of papers within this space relevant to our study. At the systems level, Sapkota et al. [24] proposed a broader systems view, distinguishing “AI agents” from more autonomous “agentic AI” and documenting recurring technical obstacles (hallucination, brittleness under distribution shift, coordination failures across tool calls, lack of persistent memory) alongside proposed solutions, some of which our participants also encountered, although in the context of forecasting where these challenges have unique impacts on the final forecast. Bansal et al. [7], however, located the problem one level up, at human-agent communication rather than the underlying model. In particular, Bansal et al. [7] catalogued recurring breakdowns where an agent’s actions, explanations, or requests for user input failed to align with what a user expects or needs, arguing that many failures attributed to model capability are in fact better understood as interaction-design failures. Separately, work by Shome et al. [28] surfaced a gap between industry aspirations for AI agents and the realities users experienced when actually deploying and relying on them, echoing challenges around understanding agent’s behaviors, such as conservativeness during specific tasks and users expressing uncertainty about agentic behaviors, which we also observed among bot builders in our sample.

## 2.4 AI Support for Communicating Uncertainty

Communicating uncertainty is a challenge in the forecasting process, where a forecast represents the probability of an event outcome, which also leaves it open to the decision-maker to decide how much confidence to place in and how to interpret the forecast. Prior HCI and visualization research has addressed the challenges and strategies of visually communicating uncertainty in forecasts, specifically highlighting the challenges of representing uncertainty in visualizations, that often result from the human inability to fully grasp the uncertainty associated with predictions [12, 14, 17, 20]. This body of work largely centers around the representation of uncertainty, focusing on how uncertainty is visualized and how audiences interpret uncertainty visualizations [6, 6, 10, 18, 29, 38]. In our study, however, we found that participants raised other uncertainty concerns related to their bot’s forecasts, including understanding the potential impacts of brittle changes and unnecessary complexity in their systems, and the absence of test datasets for questions among others.

## 3 Methods

To understand the human issues in constructing AI forecasting bots, we conducted a qualitative interview study with 15 bot builders. We sought to answer the following research question in our study:

• RQ1: What challenges and bottlenecks do capable bot builders encounter when designing an AI forecasting bot?

## 3.1 Research Setting

We recruited participants in collaboration with a platform, [Anonymous], that hosts regular human and AI forecasting tournaments for hobbyist and professional forecasters. With the help of one of the authors, A2 who also works at said platform, we recruited participants who had built successful AI forecasting systems for the last two tournaments during Fall 2025 and Spring 2026. In this AI forecasting competition held every quarter, [Anonymous], humans build AI forecasting bots, but given a question, their bots have to come up with the forecast autonomously. The forecasting platform, [Anonymous] ofers monetary prizes, split from a shared pool of \$50k, where top performers in the tournament can make upwards of \$5k.

The forecasting platform provided a default bot template that included basic human forecasting principles and best practices that the human bot builders could customize at will. Customizations could incur at personal financial cost during the competition as participants needed to pay out of pocket for AI and other tools used outside of the small free usage tier provided by the forecasting platform.

The AI forecasting tournament starts with a set of questions written by human experts, with specifications for which resolution criteria are included or excluded for each question outcome. In the competition, the bot is responsible for providing predictions for questions at a steady cadence. Questions were scheduled to resolve throughout the tournament. Submitted bots are given a question and a time limit to make a prediction, which is then compared against a resolution reflecting reality. Questions were of three kinds:

• Binary questions: simple yes/no forecasts where the bot submits a single probability of the event’s occurrence.

• Numeric questions: the bot submits a full probability distribution over a continuous range of values instead of single point estimate.

• Multiple choice questions: the bot submits a probability distribution across a discrete set of mutually exclusive options, which sum up to 100%.

Participants then build and test their AI systems (which we refer to as “bots” throughout this paper). At the end of each tournament, participants are ranked according to the accuracy of the forecasts produced by their AI bots, resulting in a leader board of bot builders, between whom a pool of prize money of \$50k, is distributed. Participant bots post their forecasts, with a rationale for their reasoning as a comment on the discussion threads for each question, making peer bot forecasts accessible to anyone participating in the forecasting tournament. The forecasting platform also runs shorter tournaments with a faster resolution turnaround every two weeks, that several participants mentioned using as an opportunity to test their bot performance. In terms of fostering engagement between participants, the platform ofers a corresponding Discord channel for bot builders for tournament participants to discuss bot building and forecasting strategies.

## 3.2 Participants

To be eligible to participate in the study, participants were required to have been a recent prize winner in a recent AI forecasting tournament (prizes are awarded algorithmically based on the bot’s rank and performance in coming up with timely and accurate predictions), be over 18 years old, and be willing to discuss their AI system. The second author identified prize winners in their most recent AI forecasting tournament (Fall 2025 and Spring 2026), to whom we then reached out via email to advertise the study. We chose to recruit prize winners as they were more likely to have successfully deployed a forecasting bot and implemented substantial forecasting strategies that (1) yielded positive outcomes and (2) may have required a substantial technical lift.

Our final sample consisted of 15 participants, all identifying as male, within the age ranges of 20 to 60 years old, with the majority of our participants geographically located in the US and 6 participants residing outside of the US. Our study was approved by our university institutional review board. Participants had a variety of programming and forecasting expertise, which we summarize in Table 1, with the majority of participants working in professional software development positions ranging from interns to software engineers and sometimes AI/ML experts. In terms of coding expertise, 5 participants described having 6 months of programming experience whereas 2 participants were the most experienced builders in our sample with over 2 years of coding experience. Out of the 15 study participants, each participant managed a single bot set up for both forecasting tournaments, except for one pair of participants (P11a/b) who joined the tournament as a team, and who we interviewed as a team.

Conference acronym ’XX, May 10–14, 2027, Pitsburgh, PA
<table><tr><td>Participant</td><td>Rank bucket</td><td>Prior Coding Experience</td><td>Prior Forecasting Experience</td></tr><tr><td>P1</td><td>Top 25</td><td>No</td><td>Yes</td></tr><tr><td>P2</td><td>Top 15</td><td>Yes</td><td>Yes</td></tr><tr><td>P3</td><td>Top 50</td><td>Yes</td><td>Yes</td></tr><tr><td>P4</td><td>Top 5</td><td>Yes</td><td>N/A</td></tr><tr><td>P5</td><td>Top 30</td><td>No</td><td>None</td></tr><tr><td>P6</td><td>Top 10</td><td>Yes</td><td>None</td></tr><tr><td>P7</td><td>Top 10</td><td>Yes</td><td>None</td></tr><tr><td>P8</td><td>Top 15</td><td>Yes</td><td>Yes</td></tr><tr><td>P9</td><td>Top 20</td><td>Yes</td><td>No</td></tr><tr><td>P10</td><td>Top 20</td><td>Yes</td><td>Yes</td></tr><tr><td>P11a</td><td>Top 25</td><td>Yes</td><td>Yes</td></tr><tr><td>P11b</td><td>Top 25</td><td>Yes</td><td>No</td></tr><tr><td>P12</td><td>Top 20</td><td>Yes</td><td>Yes</td></tr><tr><td>P13</td><td>Top 5</td><td>N/A</td><td>Yes</td></tr><tr><td>P14</td><td>Top 15</td><td>Yes</td><td>Yes</td></tr></table>

Table 1: A table showing the details of participants who chose to share additional information about their tournament rank, prior coding and forecasting experience. We show bucketed participant ranks ([1–5], [6–10], [11–20], etc) to obfuscate identity.

## 3.3 Procedure

We used a semi-structured interview protocol with guiding questions to probe into our participants’ process of building AI systems for forecasting. The first and senior authors, A1 and A3, conducted co-interviews with participants for up to an hour remotely over Zoom. We conducted all interviews as a pair except for two interviews. During the first half of the interview, we asked participants to walk us through their AI system’s process of prediction for a single question from a recent forecasting tournament, asking them to describe the research and prediction set up of their forecasting bot and reasoning patterns or strategies they had successfully implemented or experimented with. During the second half of the interview, we probed participants for bottlenecks or challenges they experienced, including questions on how they tested their bots and reviewed output. We also probed participants for areas where they saw opportunities for future tools or resources that would help them improve their forecasting bot. We excluded from analysis any information that participants told us they did not want to share, such as private information related to their bot set up including anything they asked to keep out of the research manuscript. At the end of the interview, we ofered participants a \$100 gift card as compensation for their time. Lastly, transcripts were generated by Zoom, which we then used a commercial LLM to fix errors before analysis. A sample semi-structured interview protocol is attached in Appendix A.

## 3.4 Analysis

We performed thematic analysis to extract themes describing related to participants’ approaches for (1) implementing reasoning in their AI systems, (2) acquiring accurate and reliable research, (3)

evaluating their system performance, and (4) tooling or research that could address the bottlenecks they encountered when building AI forecasting systems. The first author generated codes during the open coding stage until a saturation point was met, meeting regularly with the senior author to discuss coverage of relevant phenomena. Then they clustered them into higher-level themes based on their frequency in the data and similarities or diferences in the transcripts. Our co-interview practice allowed us to debrief and reflect on participant strategies and challenges after the interviews [35], contributing to a richer understanding of participant experiences. Throughout the analysis and writing, we referred to a human-AI co-authored summary spreadsheet of the participants’ detailed bot set ups and to the transcripts for fidelity and depth.

## 4 Findings

Our study focused on understanding the experiences of individuals building and deploying AI systems for forecasting. We discuss themes related to participants’ experiences developing the research and reasoning abilities of their bots and evaluating their bots. Throughout, we call out factors that put a ceiling on the success of forecasting attempts.

## 4.1 Research

Bots invariably began their predictions by looking for context on which to base them. The default template bot included set up instructions for the bot to gather research using the AskNews API<sup>1</sup> and to adopt the persona of a “professional forecaster” when coming up with a forecast. This template also uses ensembling or aggregating the results of multiple forecasts from various AI models into a final forecast, either as a mean or median of the multiple AI predictions. Participants could customize the template with their LLM of choice. The following challenges came up in getting them to do research well.

4.1.1 Thoroughness. Quality predictions depended on quality research. As P4, builder of one of the best-performing bots, told us, “the areas where you will beat the market by like 10 times are the ones where your research is really, really good.” They described their bot as doing a “whole load of research.” P4 invested a great deal of efort in helping their bot find unusually useful information, exemplified by a particular question they described:

“My bot put 90% on 29 cities. . . First of, it loads up the website that it’s about, and it requests an image of the website and analyzes that image. . . it used that to manually count that there were 29 dots. . . [and then it found that the page] was last written in 2022. . . and it looked at the author. And then it had loaded up LinkedIn. . . But it worked out he’d changed jobs from the organization that was recording this to a diferent job around when the blog stopped being updated, around when the last Freedom City was added. And it deduced from this that the chance of them updating it recently were very, very low.” (P4)

P4 told us that “that question went very well” for their bot. P4 described their approach as instilling “curious agency” in their bot, because “if you can make an LLM that’s curious about the things that matter, you can find stuf that others don’t.” To nudge their system to be curious, they prompted the bot to use the above example as a case study for future iterations. Several participants reported eagerness to push forward agentic approaches that granted more flexibility to the bot in seeking out background information (P4,P6, P13, P14). And several were implicitly working with agents, by invoking models that had deep research capabilities (P2, P4,P6, P11, P14). However, this flexibility came at a cost—P4 described adding switches to their bot so that it would adaptively shut of subsystems as it approached the per-question time budget, given that more flexible research had the potential to run for a long time.

Thoroughness involves not just accumulating the right background knowledge, but also checking one’s work. P6 described a problem they perceived of models “skipping cheap verification” opportunities. P14 described their “worst failure” as involving a prediction that relied on a misreading of a foreign language website and how they recently added a fact checking module to the bot. When confronted with conflicting candidate predictions, P14’s bot would use tools like web scrapers to search for “facts that would sway the final forecast one way or the other.”

Like in P14’s case, bots could rely on inappropriate information. P2 described their bot giving too much weight to overly biased sources. Broadly speaking, builders described the importance of seeking out a “diversity” of sources (P10 and P14), and many did, for instance by querying multiple sources (P3, P4, P6, P10, P13, P14) or implicitly querying specialized datasets (like social media posts) through calls to external models (like Grok) (P7, P10, P14). Builders also described conventional model issues afecting their research, including hallucinations (P7) and losing nuance from sources as their information is processed over multiple stages (P1, P3, P12).

4.1.2 Access. Bots hit impasses in getting access to the best background information. As P3 put it,

“Whenever I went to look for the best data myself personally — even for the data that LLMs mentioned, ‘oh, it’d be nice if I had this’ — they just didn’t get it” (P3).

Why do LLMs fail to get the “best data”? Builders described two reasons. First, sometimes the data is locked in a format that is inaccessible to the bot:

“a lot of data is. . . often in, like, an interactive dashboard that, you actually need to have the page load. You might even need to click around in a few diferent places to put the right filters in, etc. There is not necessarily a canonical URL that shows you exactly what the source that you need, and that data is often not indexed, so the search tools are not going to find it” (P10).

Second, access to the information could be restricted (P3 and P14). P3 provided the example of forecasting the price of plywood. They described the process of how one would seek out relevant information about the current cost, ruling out several options and arriving at Home Depot (an American hardware store) as a reasonable source of information. However, P3 shared how “Home Depot’s API is really hard to access these days—you have to go through a lot.

And you don’t want to just get banned because scraping their data is kind of illegal.” P3 pointed out that in this situation one could actually get access to the Home Depot GraphQL, but because it is poorly-documented, it would require quite a bit of time for bots to research before it could be used to query useful information.

P14 described wanting “tooling to pull any. . . arbitrary piece of information” that could provide “robust” and “precise” information about claims. In the absence of such tooling, builders took steps towards curating such a source by themselves. To overcome the issue of inaccessible data formats, they gave AIs control of the browser, for instance to take screenshots of pages that served reference material for forecasting questions (P10 and P13). Several builders programmed their bots to query APIs (P3, P6, P10). P3 built up their own knowledge base, reverse engineering data APIs and fetching their data into a local database that could be queried by their bot.

4.1.3 Base Rates. One practice of efective forecasters is to anchor predictions in initial best-guess base rates and update them as new evidence is encountered [30, 31]. For our participants, one of the roles of research is inform those base rates. Builders sometimes anchored their predictions in human predictions they found elsewhere because “a lot of information is already kind of included in [them]” (P13), presumably the information that others used justify placing a bet on the question. P13 explicitly encoded this approach in their bot:

“If we have a community prior, then I’m telling my bot that it should start from this. How strong is your prior? If it’s strong, then don’t let the next step influence you so much. Then use the news output from the assistants and update it depending on how relevant and trustworthy it is. Then state it again.” (P13).

Builders sought out these base rates from multiple sources. For a time, they were able to anchor on the human “community prediction” from [Anonymous], and several did (P2, P13, P1, P11, P8) ; however, this was disabled, in part so that bot predictions would reflect situations with no available consensus [34]. Some participants described anchoring on questions from prediction markets (P3, P6, P11, P13) such as Kalshi and Polymarket [15, 22]. However, anchoring could be dificult to do well in this situation as “you don’t have the exact same question, the exact same resolution criteria, and so on” for many of the questions posed (P13).

## 4.2 Reasoning

We interviewed builders when their bot forecasting performance was improving just by nature of improvement of the models they depended on (P3, P4, P6, P7, P8, P14). Here we focus not on that aspect—which was out oftheir control, but rather the design choices in their bots that they anticipated (though did not always) would improve their forecasts.

4.2.1 Classic Forecasting Strategies. Builders were often themselves forecasters, whether in a professional or amateur capacity (P1, P2, P3, P8, P10, P11a, P12, P13, P14). Many attempted to translate patterns of forecasting from their own work into the bots. They described trying to instill the right mindset in their bots, like using “structured thinking,” “open-mindedness,” using the “wisdom of the crowd” (P1), “thinking hard,” “using math,” (P2) “superforecasters” and “Tetlock research”<sup>2</sup> (P10). Some of these goals were written explicitly into prompts. As described above, builders implemented techniques for finding reliable base rates (Section 4.1.3) and seeking out diverse sources (Section 4.1.1). They experimented with patterns like incorporating Monte Carlo simulations (P4) and evaluating their bot’s performance using Brier scores (P3).

Builders spoke at length about developing their bot to form predictions that aggregate over multiple individual predictions. This resembled the Delphi protocol from human forecasting [5, 31] (P8 named the method explicitly), where multiple forecasters make independent predictions, review each other’s predictions, and submit revisions to be aggregated into a final prediction. Many bots had some form of aggregation approximating Delphi, as the default bot template had a basic form of the method built in. However, there was considerable variation—with participants aggregating over 2, 5, 8, and 54 predictions (P1), aggregating via medians or weighted averages (P3,P6, P7, P10), and aggregating over predictions from distinct models (P5, P6, P10, P14). Use of aggregation was common but seemingly not essential, with P2 achieving decent performance with a single model.

Some builders took efort to reduce undesirable interference between reasoning of various models used. An important aspect of procedures like Delphi is blindness of the individual predictors. P8 described enforcing blindness by architecting their bot so that individual predictions were not biased by each other. P7 described a variant in reasoning where a forecaster and critic evolved a prediction through iterative dialog. The critic was re-initialized on each turn to allow it to target diferent parts of the forecast, while the forecaster maintained context so that it could have a “consistent sense of self” and avoid vacillating in its forecasting approach.

4.2.2 Optimizations. Some of the work of building a bot involved local optimizations that more closely resembled patching over reasoning. Bots sometimes made obvious mistakes that were costly. P2 provided an example, where a question like “will Google stock price go above X dollars?” was predicted to be almost certainly true certain because the bot found it had previously been true, although the intent of the question was to ask about future price. P8 saw their model mistaking what the current date was at time of prediction leading to incorrect interpretation of prediction questions, where a date like “May 25” was seen as already having occurred in a previous year, rather than in the future. Builders described frequently fixing up these issues (P2). Others described adjusting the prediction distributions they output to the forecasting platform, calling tools to generate distributions (P3 and P6) or tweaking it, like P13, who added spikes to the distribution when the bot had high certainty in a specific prediction.

4.2.3 Contending With Complexity. Some builders described very elaborate bots. P5, for instance, described an involved approach to aggregating over multiple predictions. Their bot aggregated over 8 predictions, each made with a confidence score. The final pre diction was formed by using an average weighted by confidence score. However, P5 believed the aggregation sufered because models tended to produce confidence scores in a flattened range of 50–80. They therefore subtracted a constant from all scores until the minimum confidence was 15. Another example came from P1, whose bot produced output distributions on the basis of 54 individual predictions. 6 bots were prompted to predict 3 outcomes based on what they called “low,” “mid,” and “high world” scenarios, and then to refine those predictions into further low/mid/high variants (6 × 3 × 3 = 54). Their output distribution was formed by interpolating over those 54 predictions. These bot builders had invested considerable efort in their bot’s reasoning strategies. P5 based their choices on what empirically seemed to work best; P1 had custom monitoring dashboards, and based their approach on ideas from when they had done forecasting professionally. However, these were among the less competitive bots in our sample, suggesting complexity does not always pay of.

P4, one of the better performers, provoked that “complexity is unintuitively bad.” They elaborated that “generally I find that kind of you shouldn’t overcomplicate your predictors. Every time I try and do something fancy, it’s worse than just letting the base model do its thing.” Participants experimented with many techniques, and many reported them not leading to substantial enough improvements to maintain. These techniques included tool calls to mathematical and statistical tools (P2), use of multiple predictions (P2), Monte Carlo simulations (P4), a post-prediction synthesis step (P4), prompt optimizations (P6), and use of probabilistic models like mixture models and-distributions (P6). Modest improvements could be less useful if accompanied by high token costs:

“I tried diferent like weighted averages where a review bot would weigh diferent forecasts. . . and I tried also using diferent models. . . it didn’t have very good or I would say much improved results over the exploding token usage. So I think I used double or triple the tokens, but I didn’t achieve scores double or triple” (P12).

This is not to say that the best bots are totally minimal. As described above, some of the best bots (such as P3, P13, and P14) involved instantiations of aggregation, fact-checking, agentic search, and other local optimizations. However, it does mean that many attempts to add useful techniques, including those grounded in participants’ personal forecasting experience, do not substantially pan out. P11 conjectured that bots need a certain degree of “flexibility” to perform well; many of the adjustments described have the potential to limit this flexibility.

## 4.3 Evaluation

Builders needed input on how well their bot predicted, but getting this input remains a tricky problem.

4.3.1 Britleness and Bugs. Bots could fail in substantial ways that were not easy to predict. P2 and P8 described two examples that were triggered by only specific formats of questions that we covered in Section 4.2.2. Nor were they necessarily easy to detect. P8 described an additional example of a major issue that was not readily obvious:

“there’s this weird thing about how I’m doing some of the interpolation, where sometimes you accidentally do it in a slightly wrong way, and. . . it just, like, destroys the template and just dumps a diferent thing there that shouldn’t be there. . . [the LLMs] still will give you an answer that probably is much worse than it otherwise would be, but. . . it’s not worse in a way that immediately is apparent that something is broken” (P8).

Furthermore, changes made to fix one problem could introduce others. P8 described a case where they made changes to a prompt for deriving consensus from their predictors that “caused it [to] try do a net new forecast” without the research data and “was much worse than the individual forecasts that were feeding into it.” Given the potential for a change to degrade unanticipated predictions, P14 shared that “you need a lot of data to be able to justify doing some kind of adjustment, or you’ll cause more harm than it was worth in the first place.”

4.3.2 Testing. Builders described at times how they would have found it useful to get more feedback on how well their bot was doing, especially after making changes to it (P6 and P8). One setup involved running the bot on a set of historical questions and computing the accuracy of its predictions, an approach called backtesting. P6 described their approach:

“I log my research and go back and feed it to typically a slightly smaller version of my ensemble to save money, and actually compute the peer score that we’re evaluated on in the tournament, and use that for evals. But again, that’s really noisy. I do compute statistics, of course, to try to be rigorous. And it typically requires at least 30 or 50, maybe 80 or 100 plus questions” (P6).

However, this approach was not common. The only other builder who described backtesting was P3, though they did not share the particulars of their approach. P8 designed their bot to cache research so that they could selectively evaluate their research stage, but did not integrate it into a backtesting flow. Backtesting is dificult because when a bot is tested on historical questions it is often “taking in information from the future” in its predictions (P13), finding clues or answers to how an event resolved from published research. In this way, a bot can outperform its actual ability for forecasting future questions. One potential solution involved restricting a bot to only use historical sources, though P3 described that information could still be leaked:

“Retrodiction/backtesting for forecasting is notoriously leaky. . . you look into the actual logs [and see that] there’s a little bit of leakage from—because they changed the article, and our SERP or search provider didn’t really recognize that this had been changed, so the time-archived copy actually leaked information a little bit, just enough so the LLM didn’t explicitly use it, but was tainted anyways” (P3).

Complicating the picture, testing is costly. If a testing suite in volves 100 questions and a typical question costs on the order of \$1, as P6’s did, “if I want to do rigorous evals, call it \$100.” Other bots cost more (P13 estimated 3 euros per question). For these builders, their bot would net a few thousand dollars in the best case if they performed well in the competition. Tests of P6’s scale could only be run sparingly if a bot was to break even. P4 told us they would need “way more money” to do more backtests. P5 wished for historical source datasets they could use to backtest.

Monitoring. Some participants made custom dashboards for monitoring their bots (P1, P3, P14). P2 and P11 set up their bot to send them email alerts on failure. However, these alerts were not silver bullets for catching failures: P11 described that sometimes “it doesn’t report an error, but it [also] doesn’t predict.”

4.3.3 Inspection. The most common approach to evaluating one’s bot was to review its outputs—its predictions, summaries of rationale, and other logs produced during execution. All participants reviewed their bots during except for P1, P5, P9, and P11a/b who reviewed after prediction. Review would often spike early during the bots development(P5, P10, P11a/b, P13). P5 described when developing the bot that initially “I just sat there, I just, like, I’d run it, look at [the output], run it, look at it, run it, like, all day.” The amount of time spent looking at the outputs could be significant. P11a/b estimated spending “maybe half, or even over half” of their total development time looking at logs, where most of that was spent on reviewing forecasting outputs.

Later, when the bot was implemented and deployed, time reviewing logs decreased. Builders described spending ten minutes daily (P3), a couple of minutes for a handful of questions (P6) or a couple hours a month (P4), or even less time(P10 and P13) reviewing them. Review was often triggered by anomalies:

“I’m not checking every single answer. I’m mostly checking the ones where my bot says something very diferent from the community prediction. And then I check my bot’s answers, and think about “Is my bot being very smart or very dumb?” (P13)

P2 similarly described looking at reasoning output only when they saw a “crazy prediction.”

Builders could also review outputs in a batch during retrospectives. P8 analyzed the logs from their bot when a competition season concluded. This approach allows review of predictions across many questions when the resolution is known. P8 used AI to summarize strengths and opportunities for improvement for their bot based on the logs.

While reviewing outputs was never described as a significant bottleneck, it also did not seem trivial. P8 shared that “knowing that something’s awry is quite challenging.” P5 described that “it’s really hard to, like, read through pages and pages of, you know, AI slop to determine some sort of signal.” P10 put it that “some of them might just be relatively long, and it just takes a little bit more time and and mental overhead” to review them. Logs could indeed be quite long—P4 described generating 10MB of logs per prediction.

Review of outputs was described as involving “intuition” (P4) or was “vibes-based” (P10). Those who were experienced forecasters would sometimes perform forecasts of their own and compare the bots’ output to them (P10 and P11). Other times, builders compared their logs to those of competitors. This could help them determine ifa diference in their prediction arose from a “reasonable deviation”

such as finding a piece of information other bots did not (P2). P6 and P8 described reviewing comments from other bots, with P6 drawing inspiration from competitor bot outputs. P13 described an intriguing variant where they attempted to anticipate the efect of a change to their bot by downloading earlier predictions and asking what would have been diferent if their change was in place.

To look for opportunities to improve, builders sometimes consulted reference prediction values. P3 described comparing their bot’s predictions to live market predictions on Kalshi [15]. Some of the bot builders had previously compared to aggregated community predictions made by human participants on [Anonymous] (P6).<sup>3</sup> Builders observed that human predictions may become less valuable as models continue to improve (P6). As P14 put it, such scores could be “limiting, unless you really don’t trust that you’re gonna exceed community prediction ever.” P1 pointed out how AIgenerated forecasts impacted community predictions which were no longer reliable for testing:

‘So I think it’s mostly that if you ask the same LLM the same question or a similar question in a diferent way, the responses are more likely to be similar than if you’re asking a diferent LLM” (P1).

4.3.4 Samples of Questions. When participants evaluated their bots, they needed to do so on specific questions. As described above, one strategy was to review questions from the competition that led to anomalous outputs. While developing and evolving their bots, however, builders needed to select sets of questions to test on.

Builders described selecting questions on the basis of:

• familiarity. For instance, P9 tested their bot on random fac tual questions to which they could find or easily derive an answer, like who is the president of a given country, music trivia, or basic equations and formulas

• personal insight. P6 described selecting questions “where I have a strong opinion, and where I’ve seen bots—at least based on the community benchmark—get them wrong in the past.”

• diversity. P5 and P11 described selecting samples that covered a diversity of question types and subject matter, perhaps because their bots would be evaluated against a diversity of questions.

• half-life. P8 described testing against a recommended practice question of whether there will be human extinction by 2100. The benefit of this question was that “the news changes. . . but not that much to actually impact that most of the time. So it’s relatively steady, constant thing to run predictions against.”

• substance. P6 chose questions that involved some “proactive thought. . . connecting the dots, thinking beyond the exact single chain of thought and not drawing any conclusions.”

Some participants evaluated their bot’s performance by entering it in a small-stakes biweekly tournament also ofered by [Anonymous] (P7, P8, P14). For one participant, P1, this smaller tournament provided an opportunity to clarify what makes a useful question, which this smaller tournament did not seem to reliably provide <sup>4</sup>. The questions either “resolve very quickly,” which presumably means they require less ambitious reasoning; or they have “high random content,” which presumably means that changes to a bot are unlikely to lead to reliably better performance on them. P1 and P10 both pointed out the value of having more useful questions. They pointed to an efort by Mantic AI<sup>5</sup> to collect more useful forecasting questions, which were defined as those that maximally diferentiate between forecasters.

Regardless of how questions were selected, one issue is that a small sample of questions could only be so informative. As P1 told us, “it’s easy to over-interpret the results of your questions because you have a relatively few questions of each type.” P5, who evaluated against a set of 17 questions meant to cover diferent question types, described results oftheir tests as necessarily “incredibly noisy.” Even P6 described test results as being “really noisy” when they ran their backtest benchmark of 30–100 questions.

## 5 Discussion

## 5.1 Takeaways

Our observations of builders lead us to make the following provocations for how to improve the process, tooling, and human-AI teaming involved in forecasting.

5.1.1 Process. For future builders of bots, we ofer the following ideas:

Approach Complexity with Caution. More complex approaches to designing AI forecasting bots do not necessarily correlate with capability or better performing bots in the tournament. Complex or conceptually intriguing reasoning approaches, including those inspired by real-world forecasting techniques, often did not pay of. Bots could succeed even when omitting patterns common to most others, as in the case of P2 whose bot made use of a single model without aggregation. Builders should not avoid complexity entirely—top performers like P4 and P13 integrated specialized research and fact checking components, for instance–but it may be appropriate to adopt a default skeptical stance towards changes one is considering making to a bot.

Opportunistic Review and Beyond. We documented patterns of review that may be instructive to those getting started with bot building. Builders in our study reviewed bot outputs diferently at three stages: (1) heavily and regularly during initial development (2) upon anomaly during deployment, and (3) systematically after a large number of predictions had been made. The first two stages were more common than the last. Builders largely did not review outputs after deployment unless an anomaly occurred. This does not necessarily mean that there is no value to be had in the remaining outputs. One builder, P8, described attempting to derive insight from bot outputs by processing them with AI. One opportunity for improved process might be to log and review predictions in ways that make it easier to learn how to optimize predictions that were not clearly bad but could nevertheless be improved.

Before questions were resolved, builders compared predictions to forecasts they made themselves and to community predictions. After resolution, builders compared their bot’s reasoning to others’. Builders iterated on their bots by reviewing their performance on sample questions. Sample questions could be more or less useful based on their familiarity, alignment with their personal knowledge, diversity, half-life, and substance. We posit that future work might better establish how to build a maximally useful roster of questions under a fixed budget.

Testing and Iteration. While builders mostly did not backtest their bots, we provide proof of possibility of doing so. P6 described a backtesting setup where they tested a reduced-complexity reasoning model against 30–100 questions using archived sources at an estimated cost of around \$1 per question. P8 described a related setup where they cached partial results from the pipeline, allowing for selective re-running of parts of the bot.

5.1.2 Tooling. Our analysis revealed several areas where builders could benefit from better tooling.

Intrepid Research Tools. Some builders believed that better research methodology had the potential to greatly improve their bot. Two current bottlenecks in research was access to sources and the format of those sources. Builders extended access to additional information by querying external APIs and prediction markets and scraping web content. One builder created personal datasets that could be queried by their bot. These builders’ bots could have benefited from deep research agents that went deeper and services that made specialized information discoverable and easily parseable.

Robustness to Bugs. Small issues were capable of raising errors or steering a prediction in a wrong direction. P11 described a complication that issues in prediction can fly under the radar of standard error notification tools. Builders might benefit from better ways of detecting errors, both during deployment and before. One interesting opportunity may be to develop approaches to generating test questions that will maximally exercise the machinery of a bot pre-deployment and detect unusual output predictions.

Maximally Useful Questions. Two participants called out Mantic AI’s initiative to develop a test set of questions that can maximally diferentiate the capabilities of diferent forecasters. P1 described wanting a similar discriminative set of questions for regular testing of their bot so that they could more rapidly converge on which parts of their bot would be the highest leverage to change.

5.1.3 Human-AI Teaming. Our study also helped to point out how the work of forecasting might be best organized within a human-AI team.

Research. Research was a perceived bottleneck for some of the participants. They described sources that were easy for a person to access and parse, but dificult for their bot. This helps to point out the role that people will likely have to play in accessing and reading sources in order to make the very best forecasts.

Making Sense. We speculate that there may be a significant role for people to play in sensemaking in various parts of the forecasting task. One participant described the dificulty of using a potentially useful piece of information—community predictions on prediction markets—because of diferences in particulars of a prediction question between how it was posed to the bot and how it was written on the prediction market platform. We speculate that people might be able to make judgments about what kind of information might be derived and transferable from predictions on alternative but related questions. Furthermore, we also speculate that humans and AI may have to work together to make sense of suggestions that AI makes for how to improve a bot based on past performance. Understanding what to take action on likely requires taking into account a person’s appetite for changing their bot and understanding of the kinds of situations in which it will be later deployed.

## 5.2 Limitations & Future Work

This work has several limitations. Our sample skewed towards certain demographics, with the majority of our participants having prior coding experience, which potentially influenced the diversity of bot builder experiences in our study. Future work could investigate how the practices of users with less forecasting and coding experiences difer from our findings. We also acknowledge the limitations of using AI in the context of a forecasting competition, which include (1) questions are often written in a way for bots to try to come up with a prediction, and do not necessarily reflect real life forecasting conditions, (2) participants were not allowed to intervene in their bot’s forecasting process due to the rules of the competition, which likely influenced the strategies they could implement. Future work could investigate the design of AI forecasting systems in real life, high-stakes contexts such as financial trading and election forecasting, where individuals and organizations are more likely to invest significant time and resources in building AI-assisted forecasting systems.

Building on our participant insights, future work should also investigate how the structure of tournament-style forecasting and financial incentives influence the design of AI forecasting bots. For instance, future studies in this space should explore whether participant reasoning strategies generalize beyond competitions to other forecasting contexts like prediction markets, and the impacts and challenges of deploying competitive AI forecasting bots on live markets. Lastly, as AI-assisted forecasts become more common, we foresee opportunities to study how audiences and end users engage with these types of forecasts, particularly when considering downstream trust and accountability when using these types of forecasts in high-stakes settings where fact-checking is essential, such as in journalism or election forecasting.

## 6 Conclusion

We investigated the experiences of 15 bot builders and the challenges they encountered when building AI forecasting bots. Our analysis covers three tasks where bot builders hit ceilings in building their bots: research, reasoning, and evaluation. Among our observations are that complexity in AI forecasting bots often fails to translate into better forecasts, bot builders desire more resources to quickly and eficiently test their bots, and testing tended to occur opportunistically and piecemeal. Our work ofers provocations to inform those entering bot building, those building tools to support it, and those looking to refine the human-AI relationship in forecasting.

## References

[1] [n. d.]. https://www.forecastbench.org/

[2] [n. d.]. https://www.metaculus.com/notebooks/43363/ai-forecasting-in-2026/

[3] Mahdi Abolghasemi, Odkhishig Ganbold, and Kristian Rotaru. 2025. Humans vs. large language models: Judgmental forecasting in an era of advanced AI. International Journal ofForecasting 41, 2 (2025), 631–648.

[4] Rohan Alur, Bradly C. Stadie, Daniel Kang, Ryan Chen, Matt McManus, Michael Rickert, Tyler Lee, Michael Federici, Richard Zhu, Dennis Fogerty, Hayley Williamson, Nina Lozinski, Aaron Linsky, and Jasjeet S. Sekhon. 2025. AIA Forecaster: Technical Report. arXiv:2511.07678 (Nov. 2025). arXiv:2511.07678 [cs.AI]. doi:10.48550/arXiv.2511.07678

[5] Jon Scott Armstrong et al. 2001. Principles offorecasting: a handbookforresearchers and practitioners. Vol. 30. Springer.

[6] Sriram Karthik Badam, Jieqiong Zhao, Shivalik Sen, Niklas Elmqvist, and David Ebert. 2016. TimeFork: Interactive Prediction of Time Series. In Proceedings ofthe 2016 CHI Conference on Human Factors in Computing Systems (San Jose, California, USA) (CHI ’16). Association for Computing Machinery, New York, NY, USA, 5409–5420. doi:10.1145/2858036.2858150

[7] Gagan Bansal, Jennifer Wortman Vaughan, Saleema Amershi, Eric Horvitz, Adam Fourney, Hussein Mozannar, Victor Dibia, and Daniel S. Weld. 2024. Challenges in Human-Agent Communication. arXiv:2412.10380 [cs.HC] https://arxiv.org abs/2412.10380

[8] Mandi Cai and Matthew Kay. 2024. Watching the Election Sausage Get Made: How Data Journalists Visualize the Vote Counting Process in U.S. Elections. In Proceedings ofthe 2024 CHI Conference on Human Factors in Computing Systems (Honolulu, HI, USA) (CHI ’24). Association for Computing Machinery, New York, NY, USA, Article 391, 19 pages. doi:10.1145/3613904.3642329

[9] Davis Giangiulio, Ananya Chetia. 2026. Looking at just the odds isn’t enough. How traders gain an edge on prediction markets. https://www.cnbc.com/2026/08/01/traders-go-full-time-on-predictionmarkets-using-ai-bots-and-antennas.html

[10] Shunan Guo, Fan Du, Sana Malik, Eunyee Koh, Sungchul Kim, Zhicheng Liu, Donghyun Kim, Hongyuan Zha, and Nan Cao. 2019. Visualizing Uncertainty and Alternatives in Event Sequence Predictions. In Proceedings ofthe 2019 CHI Conference on Human Factors in Computing Systems (Glasgow, Scotland Uk) (CHI ’19). Association for Computing Machinery, New York, NY, USA, 1–12. doi:10.1145/3290605.3300803

[11] Danny Halawi, Fred Zhang, Chen Yueh-Han, and Jacob Steinhardt. 2024. Approaching human-level forecasting with language models. Advances in Neural Information Processing Systems 37 (2024), 50426–50468.

[12] Jessica Hullman, Xiaoli Qiao, Michael Correll, Alex Kale, and Matthew Kay. 2019. In Pursuit of Error: A Survey of Uncertainty Visualization Evaluation. IEEE Transactions on Visualization and Computer Graphics 25, 1 (Jan. 2019), 903–913. doi:10.1109/TVCG.2018.2864889

[13] Woojeong Jin, Rahul Khanna, Suji Kim, Dong-Ho Lee, Fred Morstatter, Aram Galstyan, and Xiang Ren. 2021. Forecastqa: A question answering challenge for event forecasting with temporal text data. In Proceedings ofthe 59th Annual Meeting ofthe Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers). 4636– 4650.

[14] Alex Kale, Matthew Kay, and Jessica Hullman. 2021. Visual Reasoning Strategies for Efect Size Judgments and Decisions. IEEE Transactions on Visualization and Computer Graphics 27, 2 (Feb. 2021), 272–282. doi:10.1109/TVCG.2020.3030335

[15] Kalshi Inc. 2026. Kalshi: A Regulated Exchange and Prediction Market. https: //kalshi.com. Accessed 2026-09-07.

[16] Ezra Karger, Houtan Bastani, Chen Yueh-Han, Zachary Jacobs, Danny Halawi, Fred Zhang, and Philip E. Tetlock. 2025. ForecastBench: A Dynamic Benchmark of AI Forecasting Capabilities. International Conference on Learning Representations 2025 (May 2025), 93943–93980. https://proceedings.iclr.cc/paper\_files/paper/ 2025/hash/ea74e45a229dac70b5b63b28d8934db6-Abstract-Conference.html

[17] Matthew Kay, Tara Kola, Jessica R. Hullman, and Sean A. Munson. 2016. When (ish) is My Bus?: User-centered Visualizations of Uncertainty in Everyday, Mobile Predictive Systems. In Proceedings ofthe 2016 CHI Conference on Human Factors in Computing Systems. ACM, San Jose California USA, 5092–5103. doi:10.1145/ 2858036.2858558

[18] Morgane Koval and Yvonne Jansen. 2022. Do You See What You Mean? Using Predictive Visualizations to Reduce Optimism in Duration Estimates. In Proceedings ofthe 2022 CHI Conference on Human Factors in Computing Systems (New Orleans, LA, USA) (CHI ’22). Association for Computing Machinery, New York, NY, USA, Article 30, 19 pages. doi:10.1145/3491102.3502010

[19] Barbara Mellers, Eric Stone, Pavel Atanasov, Nick Rohrbaugh, S. Emlen Metz, Lyle Ungar, Michael M. Bishop, Michael Horowitz, Ed Merkle, and Philip Tetlock. 2015. The psychology of intelligence analysis: Drivers of prediction accuracy in world politics. Journal ofExperimental Psychology: Applied 21, 1 (2015), 1–14. doi:10.1037/xap0000040

[20] Lace M. K. Padilla, Maia Powell, Matthew Kay, and Jessica Hullman. 2021. Uncer tain About Uncertainty: How Qualitative Expressions of Forecaster Confidence

Impact Decision-Making With Uncertainty Visualizations. Frontiers in Psychology 11 (Jan. 2021). doi:10.3389/fpsyg.2020.579267

[21] Daniel Paleka, Shashwat Goel, Jonas Geiping, and Florian Tramèr. 2025. Pitfalls in Evaluating Language Model Forecasters. arXiv:2506.00723 (May 2025). arXiv:2506.00723 [cs.LG]. doi:10.48550/arXiv.2506.00723

[22] Polymarket. 2026. Polymarket: The World’s Largest Prediction Market. https: //polymarket.com. Accessed 2026-09-07.

[23] Sarah Pratt, Seth Blumberg, Pietro Kreitlon Carolino, and Meredith Ringel Morris. 2024. Can language models use forecasting strategies? arXiv preprint arXiv:2406.04446 (2024).

[24] Ranjan Sapkota, Konstantinos I. Roumeliotis, and Manoj Karkee. 2026. AI Agents vs. Agentic AI: A Conceptual taxonomy, applications and challenges. Information Fusion 126 (2026), 103599. doi:10.1016/j.infus.2025.103599

[25] Philipp Schoenegger and Peter S Park. 2023. Large language model prediction capabilities: Evidence from a real-world forecasting tournament. arXiv preprint arXiv:2310.13014 (2023).

[26] Philipp Schoenegger, Peter S. Park, Ezra Karger, Sean Trott, and Philip E. Tetlock. 2025. AI-Augmented Predictions: LLM Assistants Improve Human Forecasting Accuracy. ACM Transactions on Interactive Intelligent Systems 15, 1 (Feb. 2025), 4:1–4:25. doi:10.1145/3707649

[27] Philipp Schoenegger, Indre Tuminauskaite, Peter S Park, Rafael Valdece Sousa Bastos, and Philip E Tetlock. 2024. Wisdom of the silicon crowd: LLM ensemble prediction capabilities rival human crowd accuracy. Science Advances 10, 45 (2024), eadp1528.

[28] Pradyumna Shome, Sashreek Krishnan, and Sauvik Das. 2026. Why Johnny Can’t Use Agents: Industry Aspirations vs. User Realities with AI Agents. In Proceedings ofthe ACM Conference on AI and Agentic Systems. ACM, San Jose CA USA, 597–619. doi:10.1145/3786335.3813140

[29] Dong Sun, Zezheng Feng, Yuanzhe Chen, Yong Wang, Jia Zeng, Mingxuan Yuan, Ting-Chuen Pong, and Huamin Qu. 2020. DFSeer: A Visual Analytics Approach to Facilitate Model Selection for Demand Forecasting. In Proceedings ofthe 2020 CHI Conference on Human Factors in Computing Systems (Honolulu, HI, USA) (CHI ’20). Association for Computing Machinery, New York, NY, USA, 1–13. doi:10.1145/3313831.3376866

[30] Philip E Tetlock. 2009. Expert political judgment: How good is it? How can we know? Princeton University Press.

[31] Philip Eyrikson Tetlock, Dan Gardner, and Dan Gardner. 2015. Superforecasting: The art and science ofprediction. Vol. 221. Crown Publishers New York.

[32] Benjamin Turtel, Danny Franklin, and Philipp Schoenegger. 2025. LLMs Can Teach Themselves to Better Predict the Future. arXiv:2502.05253 (Feb. 2025). arXiv:2502.05253 [cs.CL]. doi:10.48550/arXiv.2502.05253

[33] Benjamin Turtel, Paul Wilczewski, Danny Franklin, and Kris Skothiem. 2026. Future-as-Label: Scalable Supervision from Real-World Outcomes. arXiv:2601.06336 (Jan. 2026). arXiv:2601.06336 [cs.LG]. doi:10.48550/arXiv.2601. 06336

[34] Unnaamed. 2026. Changes to the Metaculus API. https://www.metaculus.com notebooks/42554/changes-to-the-metaculus-api/. Metaculus notebook; accessed 2026-09-07.

[35] Stefania Velardo and Sam Elliott. 2021. Co-Interviewing in Qualitative Social Research: Prospects, Merits and Considerations. International Journal ofQualitative Methods 20 (Jan. 2021), 16094069211054920. doi:10.1177/16094069211054920

[36] Ben Wilson. 2026. https://www.metaculus.com/notebooks/45382/bot-surveyspring-2026/

[37] Qi Yan, Raihan Seraj, Jiawei He, Lili Meng, and Tristan Sylvain. 2024. Autocast++: Enhancing world event prediction with zero-shot ranking-based context retrieval. In International Conference on Learning Representations, Vol. 2024. 34523–34539.

[38] Fumeng Yang, Chloe Rose Mortenson, Erik Nisbet, Nicholas Diakopoulos, and Matthew Kay. 2024. In Dice We Trust: Uncertainty Displays for Maintaining Trust in Election Forecasts Over Time. In Proceedings ofthe 2024 CHI Conference on Human Factors in Computing Systems (Honolulu, HI, USA) (CHI ’24). Association for Computing Machinery, New York, NY, USA, Article 389, 24 pages. doi:10. 1145/3613904.3642371

[39] Andy Zou, Tristan Xiao, Ryan Jia, Joe Kwon, Mantas Mazeika, Richard Li, Dawn Song, Jacob Steinhardt, Owain Evans, and Dan Hendrycks. 2022. Forecasting future world events with neural networks. Advances in Neural Information Processing Systems 35 (2022), 27293–27305.

## A Semi-Structured Interview Protocol

(1) Can you walk me through your bot’s process of prediction for a single question from a recent tournament?

(2) Follow-up as needed: What does your bot do?

(3) Can you show me any outputs or traces from the bot?

(4) Can you describe the reasoning patterns that you implemented in your bot?

(5) Did it work? If not, what broke and how did you resolve these issues?

(6) What other reasoning patterns would you implement if time and cost were not an issue?

(7) Optional: What are some other approaches that you have tried and failed?

(8) How well do you think your bot does? How do you know?

(9) What would you build if time and cost were not an issue?

(10) Optional if time permits Tell me about a question that your bot has failed on in an interesting way.

(11) What were the bottlenecks to success that you noticed?

(12) For each bottleneck: How did you get (or try to get) around this bottleneck?

(13) How did you test your bot?

(15) What outputs or traces do you review? For example, do you review the final prediction, comments, Github action logs or other print outputs?

(14) How did you iterate on your research pipeline?

(16) How much time do you spend manually reviewing your code, in one session or overall?

(17) How do you design your logs or outputs to support testing and review? What is your process for making sense of your logs?

(18) If not already covered Did you learn or implement any strategies from other participants or the forecasting community?

(19) How do you (and your bot) handle questions outside of your area of expertise?

(20) If not already covered: What does your current tech stack lack?

(21) Optional if time permits: Did you encounter other bottlenecks?

(23) If there was one aspect of bot building or AI for forecasting that you would want researchers or tool-builders to make easier, what would it be?

(22) What would make resolving these issues easier?