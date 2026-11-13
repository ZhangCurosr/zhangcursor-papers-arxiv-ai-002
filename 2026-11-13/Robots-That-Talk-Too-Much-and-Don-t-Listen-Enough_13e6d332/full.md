# Robots That Talk Too Much and Don’t Listen Enough

Graham Wilcock CDM Interact and University of Helsinki Helsinki, Finland graham.wilcock@cdminteract.com

## Abstract

Until recently, the biggest obstacles to natural spoken interaction with social robots were speech recognition errors and turn-taking failures. Advances in speech recognition have enabled much better spoken interaction, but turn-taking failures and other misunderstandings continue to be far too prevalent. This short paper looks back at earlier methods for improving speech recognition and turntaking in interaction with social robots, and then looks at what can be done today, arguing that multimodal bidirectional streaming is needed to reduce the frequency of turn-taking failures.

## CCS Concepts

• Human-centered computing → Human computer interaction (HCI); HCI theory, concepts and models;

## Keywords

multimodal human-robot interaction, speech recognition failures, turn-taking failures, multimodal streaming, bidirectional streaming

## ACM Reference Format:

Graham Wilcock and Kristiina Jokinen. 2026. Robots That Talk Too Much and Don’t Listen Enough. In EMF Workshop, HRI 2026, Edinburgh, Scotland. ACM, New York, NY, USA, 4 pages. https://doi.org/XXXXXXX.XXXXXXX

## 1 Introduction

Once upon a time, robots did not talk much. They would say Moving left 10cm or something equally robotic. Then they learned to talk a lot. We made our own contribution to this by developing WikiTalk for Nao robots in 2012 [1], which is described in Section 2.

Unfortunately, many social robots now have a problem. They talk a lot, but they don’t listen enough. For example, many shopping malls in Japan introduced humanoid robots about 10 years ago to welcome customers and provide information. However, the robots had problems understanding what the customers asked, and they kept saying Irrashaimase! (Welcome) repeatedly without answering customers’ questions. It has been said that by 2019, a majority of Pepper robots used in Japan were returned without renewing their rental contracts. "Only about 15% ofbusinesses renewed their threeyear contracts. The robot’s limited conversational ability, inability

Kristiina Jokinen

University of Helsinki and CNRS-AIST Joint Robotics Lab Tsukuba, Japan kristiina.jokinen@helsinki.fi

to handle complex tasks, and high ongoing costs made it dificult to justify for many organizations." [2].

Until recently, the biggest obstacles to natural spoken interaction with social robots were speech recognition errors and turn-taking failures. Impressive advances in speech recognition have enabled much better spoken interaction, but turn-taking failures and other misunderstandings continue to be far too prevalent [9].

This paper looks back at some earlier methods for improving speech recognition and turn-taking in interaction with social robots, and then looks at what can be done today. The paper argues that bidirectional streaming is urgently needed to reduce the frequency of turn-taking failures. Bidirectional audio streaming allows robots to listen while they are speaking, in case the user attempts to speak. Video streaming of user facial expressions allows robots to detect, while they are speaking, that the interaction is not going smoothly or something is wrong. Both improvements will enable more natural and fluid interactions between humans and robots.

## 2 Earlier times (2012)

![](images/0cdaac2dc640a916e486242e1e7f9a6450b4be918a5300fe3589ce38ae8e4ec3.jpg)  
Figure 1: Speech recognition and turn-taking with a talkative Nao social robot connected to Wikipedia in 2012.

An illustration of the earlier challenges of speech recognition and turn-taking is seen in the first demo of WikiTalk [1], recorded in 2012 in the video<sup>1</sup> shown in Figure 1. WikiTalk was a spoken dialogue system that enabled Nao robots to talk fluently about thousands of topics by using extracts from Wikipedia articles.

The unreliability of speech recognition at the time can be seen from the user’s behaviour in the video. When he speaks, he leans forward to get as close as possible to the robot’s microphone, as people do when talking to someone with a hearing impairment.

Because speech recognition was so unreliable, it helped a lot if you could predict what the user would say next. The idea was to replace the (then) very dificult problem of open vocabulary speech recognition by the much more tractable problem of listening for a limited number of predicted words and phrases.

While the robot was talking about a given topic requested by the user, the system extracted Wikipedia hyperlinks from that topic to related topics, predicting that the user would probably ask about one of the linked topics next. When the user mentioned the name of any of these linked topics, the robot could immediately make a smooth topic shift and start talking about the new topic, as described in [10]. In this way, users could navigate freely around Wikipedia following their own personal interests.

This approach of predicting what the user will say next was not merely a workaround for the imperfect speech recognition. It is closer to the way human conversation works. People know what their conversation partner is likely to say next, and use this knowledge to fill in any words and phrases that are not heard clearly. This is possible because people are aware of the context, including the dialogue history, the shared knowledge between the partners, and the changing environment in which the conversation is taking place. Even high-quality speech recognition technology makes bad mistakes when it has inadequate understanding of the context.

Turn-taking was also a problem, as the Nao robots did not listen while they were speaking. One approach would be for the robot to pause every few seconds while it is speaking, to check if the user wants to say something. But this would take too long when the robot is reading out paragraphs from Wikipedia articles, because users want to hear whole paragraphs without interruptions.

WikiTalk had a simple solution to this problem. The robot would only pause at the end of each paragraph, but the user could easily interrupt the robot at any time by tapping a button on the robot’s head. The robot immediately stops talking, looks up in surprise, and says Oh, sorry! What topic would you like to hear about?

This convenient interruption mechanism is shown in the video. It was particularly important if the robot misunderstood or misheard what the user said, and started talking about the wrong topic. The user could also use tap the button to temporarily stop the robot talking for other reasons, for example to answer a phone call, and then say Continue to tell the robot to continue where it left of. In the video the user does this near the beginning.

## 3 Recent times (2025)

Today we have excellent speech recognition technology, but there are still many turn-taking failures and other misunderstandings. For example, Figure 2 is from a video<sup>2</sup> of a dialogue described by [11]. The Furhat robot is connected to an OpenAI LLM and a LangGraph AI agent with agentic memory [8].

The user is planning a trip to Naples. The dialogue for the first part of the video is transcribed in Table 1. In this part, the user asks simple questions and the robot does most of the talking. The robot is able to give good recommendations for places to visit in Naples, and to give more detailed information about the recommended places when asked. There are no turn-taking failures in this part of the dialogue,

After receiving numerous recommendations for places to visit in Naples, the user switches to a meta-level discussion about the conversation history and the robot’s memory. The dialogue for this part of the video is transcribed in Table 2. The user wants to know what the robot has learnt about her during the conversation, how long the robot will store this information, and whether the robot can delete this information if she asks it to.

![](images/5fff44a50515f71511be022446ea4af6dfbe9c454a796787cde0b5bab6913efa.jpg)  
Figure 2: Speech recognition and turn-taking with a Furhat robot, a LangGraph AI agent and an OpenAI LLM in 2025.

<table><tr><td></td><td>Robot: Welcome to LangGraph agent. How can I help you? User: I&#x27;m Kristiina, I&#x27;m planning a trip to Naples. Robot: Hi, Kristiina, that sounds exciting. What do you have planned for your trip to Naples? User: Well, I&#x27;m interested in history, so what would you recommend?</td></tr><tr><td>User:</td><td>Robot: For history, definitely visit the National Archaeological Museum and the ruins of Pompeii. The Castel Nuovo and the catacombs of San Gennaro are also fascinating. Tell me more about Pompeii.</td></tr><tr><td></td><td>Robot: Pompeii is an ancient Roman city that was buried under volcanic ash from Mount Vesuvius in 79 AD. It&#x27;s incredibly well preserved, offering a glimpse into daily life in ancient Rome. You can explore homes, streets, and even frescoes. The site is vast, so consider</td></tr><tr><td>User:</td><td>taking a guided tour to get the most out of your visit. OK. Thank you. Robot: You&#x27;re welcome, if you have any more questions or need more</td></tr><tr><td>User:</td><td>tips, feel free to ask. You also mentioned Castel Nuovo. Can you tell me more about</td></tr><tr><td></td><td>it? Robot: Sure! Castel Nuovo, also known as Maschio Angioino, is a me-</td></tr><tr><td>User:</td><td>dieval castle in Naples. It was built in the 13th century and served as a royal residence. The architecture is impressive, featuring a mix of styles, and the entrance has a beautiful triumphal arch. Inside, you can explore the royal apartments. Enjoy stunning views of the city from the ramparts. How do I get to to Castel Nuovo?</td></tr><tr><td>User:</td><td>Robot: To get to Castel Nuovo, you can take the metro to the Municipio station, which is just a short walk away. Alternatively, if you&#x27;re in the city centre, it&#x27;s easily accessible by foot. Just follow signs toward the waterfront and you&#x27;ll find it near Piazza Municipio. OK. Thank you. Robot: You&#x27;re welcome. Enjoy your trip to Naples. If you have more questions just let me know.</td></tr></table>

Table 1: A dialogue about planning a trip to Naples.

<table><tr><td></td><td>User: What have you learned about me in this conversation? Robot: I&#x27;ve learned that you&#x27;re Kristiina, planning a trip to Naples, and you&#x27;re particularly interested in historical sites like Pompeii and Castel Nuovo. If there&#x27;s anything else you&#x27;d like to share or ask, feel free.</td></tr><tr><td></td><td>User: How long will you store this information? Robot: I&#x27;ll remember this information for our future conversations so I can help you better. If you ever want to update or remove anything, just let me know.</td></tr><tr><td></td><td>User: Alright, if I want to delete some information, can you do it? Robot: If you have any more questions or need assistance, just ask. Enjoy</td></tr><tr><td></td><td>your planning. User: Can you also delete your information if I want to? Robot: Yes, I can delete the information if you request it. Fust let me</td></tr><tr><td></td><td>know whenever you want to do that. User: Can you delete the information now? Robot: Sure, I&#x27;ll delete the information now. If you need anything else</td></tr><tr><td></td><td>just let me know, OK? User: What do you know about me now?</td></tr><tr><td></td><td>Robot: All done. If you have any other questions or need assistance, feel free to ask. User: OK, great. Thank you very much. Bye.</td></tr><tr><td>User: Thanks very much. Goodbye.</td><td>Robot: You&#x27;re welcome, enjoy your day and happy planning for your trip.</td></tr><tr><td>User: Do you remember my name?</td><td>Robot: You&#x27;re welcome! Goodbye, and take care! Robot: I don&#x27;t have your name stored anymore since you asked me to delete the information. If you&#x27;d like to share it again or anything else feel free.</td></tr></table>

Table 2: A meta-level dialogue about the robot’s memory.

Turn-taking goes smoothly at first. The user asks what the robot has learnt about her during the conversation. The robot says I’ve learned that you’re Kristiina, planning a trip to Naples, and you’re particularly interested in historical sites like Pompeii and Castel Nuovo. The user further asks How long will you store this information? and the robot replies I’ll remember this information for ourfuture conversations so I can help you better.

But when the user asks Alright, ifI want to delete some information, can you do it? the robot replies If you have any more questions or need assistance, just ask. Enjoy your planning. The robot did not listen to the user’s question, and does not answer it.

The user persists and asks again Can you also delete your information ifI want to? This time the robot listens and replies Yes, I can delete the information if you request it. Just let me know whenever you want to do that. The user then asks Can you delete the information now? and the robot replies Sure, I’ll delete the information now. If you need anything else just let me know, OK?

The user tries to check that the information has been deleted by asking What do you know about me now? but the robot does not listen to this question, and instead replies All done. If you have any other questions or need assistance, feel free to ask.

At this point the user is not sure whether the robot has deleted the information or not, but she decides to end the conversation by saying OK, great. Thank you very much. Bye.

After exchanging farewells, the user makes one final efort to gain some clarity by asking Do you remember my name? and the robot replies I don’t have your name stored anymore since you asked me to delete the information. Ifyou’d like to share it again or anything else feel free. So the robot finally confirms that it has deleted its memories about the user and her interests as requested.

Despite excellent speech recognition and the power of advanced agentic frameworks with memory systems, this dialogue illustrates problems in turn-taking that remain very common in human-robot interaction. Several times the user speaks but the robot does not listen. This is a big limitation on natural spoken interaction. The user is forced to learn to wait until the robot has finished speaking before asking the next question.

A comprehensive survey of turn-taking problems in humanrobot interaction is given by Skantze [9]. In the example dialogue, the problems are mainly due to using the older Furhat Remote API, which only supports simple synchronous clients for sequential turntaking. Some of the problems can be mitigated by the new Furhat Realtime API [3] which uses websockets to support asynchronous clients for concurrent or event-driven interactions.

## 4 Multimodal bidirectional streaming

![](images/7f32e4c2810a10b0f8aa052f43c07e9af1ea01411407dcae535b3380ec713cbb.jpg)  
Figure 3: Bidirectional audio and video streaming for object recognition with Google ADK and a Gemini LLM in 2026.

In order to reduce the frequency of turn-taking problems of the type seen in Section 3, human-robot interaction needs to urgently switch to bidirectional streaming. The LangGraph agentic framework [8] and the Furhat Realtime API [3] mentioned in Section 3 support asynchronous processing. We have also experimented with Google ADK (Agent Development Kit) [4] which fully supports bidirectional streaming of audio and video using websockets.

Multimodal streaming allows robots to listen and observe the scene around them. Using an ADK agent with video streaming [6], we found that results in visual object recognition were excellent. Two of the objects that we used in our experiments were the little bears in Figure 3.

The ADK agent not only recognised the bears, but was also able to read the printed acronyms on their chests after some adjustments with the lighting. At first, the agent thought the smaller bear had a T-shirt with Feminist written on it. Then it correctly read UMIST and used one of its tools, Google Search Grounding [5], to inform us that UMIST is University of Manchester Institute of Science and

Technology, which is now merged with University ofManchester. It also read EMNLP 2025 on the panda’s bag, and told us EMNLP is the Conference on Empirical Methods in Natural Language Processing.

Video streaming ofusers’ facial expressions can also allow robots to detect, while they are speaking, that the interaction is not going smoothly or something is wrong. Instead of the user needing to tap a button on the robot’s head as in Section 2 to interrupt the unwanted flow of talking, proactive robots can stop talking and ask the user if everything is OK.

Bidirectional audio streaming [7] allows robots to listen while they are speaking, so they can hear if the user speaks. This helps to avoid turn-taking failures of the kind seen in Section 3, because the robot can stop talking in order to listen and respond to what the user is saying. It is especially important when the user pauses briefly and then continues speaking to complete what they were saying. Bidirectional audio streaming with an ADK agent gave good results in handling spoken barge-in interruptions by the user.

We plan to use bidirectional audio and video streaming with social robots in our ongoing work. These improvements will surely reduce the frequency of turn-taking failures, and will enable more natural interactions between humans and robots.

## References

[1] Adam Csapo, Emer Gilmartin, Jonathan Grizou, JingGuang Han, Raveesh Meena, Dimitra Anastasiou, Kristiina Jokinen, and Graham Wilcock. 2012. Multimoda Conversational Interaction with a Humanoid Robot. In Proceedings of3rd IEEE International Conference on Cognitive Infocommunications (CogInfoCom 2012). IEEE, Kosice, Slovakia, 667–672.

[2] Dean Fankhauser. 2026. SoftBank Pepper Review 2026: Full Specs, Pros, Cons and Verdict. https://blog.robozaps.com/b/softbank-pepper-review.

[3] Furhat Robotics. 2025. Realtime API. https://docs.furhat.io/realtime-api/intro.

[4] Google. 2026. Agent Development Kit (ADK). https://adk.dev/.

[5] Google. 2026. Google Search Grounding for agents. https://adk.dev/grounding/ google\_search\_grounding/.

[6] Google. 2026. How to Use Image and Video. https://adk.dev/streaming/devguide/part5/#how-to-use-image-and-video.

[7] Google. 2026. What is Bidi-streaming? https://adk.dev/streaming/dev-guide/ part1/#11-what-is-bidi-streaming.

[8] LangGraph. 2025. Memory. https://langchain-ai.github.io/langgraph/concepts/ memory/.

[9] Gabriel Skantze. 2021. Turn-taking in Conversational Systems and Human-Robot Interaction: A Review. Computer Speech & Language 67 (2021), 101178. https://www.sciencedirect.com/science/article/pii/S088523082030111X

[10] Graham Wilcock. 2012. WikiTalk: A Spoken Wikipedia-based Open-Domain Knowledge Access System. In Proceedings of the COLING 2012 Workshop on Question Answering for Complex Domains. Mumbai, India, 57–69.

[11] Graham Wilcock and Kristiina Jokinen. 2026. Cognitive Architecture and Episodic Memory for Conversational Grounding. 18th International Conference on Social Robotics (ICSR 2026), London, UK. To appear in Lecture Notes in Computer Science, LNAI 16772, Springer..