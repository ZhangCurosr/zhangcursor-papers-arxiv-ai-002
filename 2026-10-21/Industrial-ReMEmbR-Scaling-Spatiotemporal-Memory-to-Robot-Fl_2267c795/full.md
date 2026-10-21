# Industrial ReMEmbR: Scaling Spatiotemporal Memory to Robot Fleets

Jad Yammine<sup>1</sup>, Malek Wahidi<sup>1</sup>, Abrar Anwar<sup>2</sup>, Yan Chang<sup>2</sup>, Mihir Acharya<sup>2</sup> and Anthony Rizk<sup>1</sup> <sup>1</sup>Idealworks, <sup>2</sup>NVIDIA

## Abstract

Warehouse robot fleets continuously observe operational conditions, but preserving these observations as reusable semantic memory incurs substantial computation, communication, and storage costs. For example, explaining recurring aisle blockages can require reviewing video from multiple robots across several shifts. We present Industrial ReMEmbR, a shared spatiotemporal memory system that extends NVIDIA ReMEmbR to robot fleets and reduces server-side computation by generating scene descriptions only on demand. On each robot, an NVIDIA Jetson encodes video clips into retrieval embeddings with Cosmos-Embed1, vision tokens with Cosmos3-Edge, and reconstruction latents with the Wan2.2 VAE, without uploading raw camera frames. The encoding models are re-used to detect people in image patches, zeroing them out to support PII abstraction for GDPR-compliant deployments. We split Cosmos3-Edge at its vision–language boundary, placing the vision encoder on the robot and the language decoder on an on-premises server. A Nemotron agent retrieves memories by content, position, and time and queries the decoder using cached vision tokens. This design eliminates caption generation at write time, while cached-token decoding reproduces the unsplit model’s output in all 450 tested decodes. On the NaVQA benchmark, the updated model stack achieves comparable retrieval recall while reducing unparsable answers from 4.5% to 1.9% with an agent half the parameter size. We further demonstrate a live semantic trafic map that classifies aisle conditions directly from retrieval embeddings. Across 914 simulated warehouse frames, this method exceeds the best caption-based alternative by 6.0 percentage points. By decoupling memory encoding from description, Industrial ReMEmbR transforms a fleet’s collective experience into an extensible foundation for spatial and temporal reasoning, including faithful scene reconstruction, context-guided elaboration, and semantic mapping of dynamic warehouse conditions as they unfold.

## 1. Introduction

An embodied memory lets a robot recall what it has seen, where, and when, long after the moment has passed. NVIDIA’s ReMEmbR (Anwar et al., 2025), a spatiotemporal memory system designed for long-horizon visual question answering for robot navigation, showed that a single robot can build such a memory from its own camera stream. On a warehouse floor, however, that robot is only one of many. A fleet management system, or fleet manager, continuously tracks dozens of autonomous mobile robots (AMRs) and other integrated devices in a modern warehouse, assigning them orders and planning their routes over VDA5050, the standard interface between mobile robots and fleet management systems. Obstacles outside the tracking system are left to each robot’s intelligent navigation system to recognize and avoid. What a passing robot sees is rarely kept, because continuously uploading and storing raw video from every robot is expensive, so the rich semantic context of each scene never reaches the rest of the fleet.

To illustrate, consider the case of a fleet management system that keeps routing AMRs through an aisle partially blocked by a clutter of untracked obstacles. Each robot independently maneuvers around the clutter, losing time before carrying on with its scheduled tasks. A few days later, a human operator reviewing the daily logs notices that task completion times slipped that day, but has no record of the scene from which to trace its root cause. Neither the fleet manager nor the operator has the semantic context that could have prevented the congestion, by adaptively routing later robots away from the aisle, or its recurrence, by revealing what caused it.

Closing this contextual blind spot requires scaling embodied memory from a single robot to the entire fleet, with every connected robot acting as a live reporter from the warehouse floor that continuously contributes its observations, tagged with location and time, to a shared fleet memory. In the scenario above, the first robot to encounter the clutter would report it on a live semantic map, letting the fleet manager route subsequent robots around the blocked aisle. Days later, the operator could ask the fleet’s shared semantic memory what caused the reroutes and receive an answer grounded in direct visual observation: when and where the clutter appeared, and what it consisted of. The same record can supply the world state of a centralized planner such as RobotFleet (Gupta et al., 2025), which schedules heterogeneous robots from declarative statements about the site and replans when a robot reports a discovery.

To this end, we build on ReMEmbR, which captions each short video clip, stores the caption along with the robot’s time and location and a corresponding caption embedding, and lets an agentic model perform semantic retrieval over the vector database to answer arbitrary questions. Scaling this design from one robot to a fleet breaks three of its assumptions, and each break motivates one principle of our design. First, ReMEmbR generates a caption for every clip as it is written, so its cost grows with every robot added and every hour it operates, and continuously uploading raw video from every robot is costly to begin with. We therefore encode once, compressing each clip on the robot’s own edge compute into compact representations that every later step reuses. Second, ReMEmbR answers occasional questions, whereas the fleet manager also needs a continuously updated view of the site. We therefore interpret continuously, reading that state directly from each memory as it arrives, without generating any text. Third, the questions a memory must answer are not known when it is first encoded. We therefore elaborate on demand, deferring any description of a scene until a question about it arrives. In the blocked aisle example, the clutter is encoded once by the first robot that passes it, flagged on the live map as soon as its memory arrives, and described in detail only when the operator asks about it days later.

We present Industrial ReMEmbR (Fig. 1), a shared memory system built around these three principles. It encodes clips on edge compute across the fleet, continuously aggregates and interprets them on server compute for live semantic mapping, and elaborates on any of them on demand. Each robot’s onboard Jetson edgecompute device runs iw.encode, the edge component for eficient multi-purpose encoding. This deployed service continuously generates video-text embedding vectors for memory retrieval derived from Cosmos-Embed1 (NVIDIA, 2025), vision tokens for optional scene description derived from the detached vision encoder of Cosmos3-Edge (NVIDIA, 2026a), and compressed latent frames for optional scene inspection generated by the Wan2.2 VAE (Team Wan, 2025a,b) encoder. Only the aforementioned multi-purpose scene representations, robot pose and memory timestamp are uploaded into a shared fleet memory, whose vector index is accelerated with NVIDIA cuVS (RAPIDS Team, 2024).

To reduce server-side computation, we generate scene descriptions and answers on demand for memories retrieved by the agent, rather than captioning every encoded clip. We enable this by splitting Cosmos3-Edge at its vision–language boundary, where its server-side language decoder answers questions from cached vision tokens without accessing or re-encoding the original clips. We deploy a Nemotron-3-Nano-4B (NVIDIA, 2026b; NVIDIA et al., 2025) agent through an NVIDIA NIM inference microservice to retrieve memories by content, position, and time, and optionally query the decoder with a text prompt and the retrieved vision tokens. Operators can also reconstruct scenes from stored latent frames for visual inspection. On NaVQA, Upgraded ReMEmbR, our version of ReMEmbR updated with the NVIDIA models and a video-based index, achieves comparable retrieval recall while reducing unparseable answers from 4.5% to 1.9% with an agent half the parameter size.

We demonstrate the system through a live semantic trafic map that communicates observed aisle conditions to the fleet manager. We classify each robot’s latest observation by comparing its retrieval embedding with paired text prompts describing occupied and clear aisles. A map service periodically queries the agent for these classifications and updates predefined semantic zones using robot poses already reported to the fleet manager over VDA5050. In a simulated warehouse, this approach achieves 71.3% balanced accuracy, exceeding the best caption-based alternative evaluated with the same models by 6.0 percentage points. We deploy each model as a modular service to support future extensions for station-readiness checks, landmark-based navigation guidance for deployed policy models, and curation of diverse preprocessed training data (Section 7).

![](images/b6e57b989df32e06bf6b3d61e367ffa06d469661f3c6d8db60c259a49e540c78.jpg)  
Figure 1: Industrial ReMEmbR moves encoding onto the robots and defers text generation to the server, until a question arrives. Each robot encodes its camera clips on board, and only the resulting embedding, vision tokens and video latent are uploaded to the fleet memory. On the server, incoming embeddings are scored continuously to keep a live semantic trafic map up to date for the fleet manager. When a question arrives, the agent searches the memory and has the retrieved clips described on demand, and an operator can reconstruct any scene for inspection.

## Our contributions are:

• Video-native agentic memory. We rebuild ReMEmbR on an eficient NVIDIA model stack and move its index key from the caption to the video itself. Retrieval is no longer bottlenecked by what a captioner chose to mention, and a smaller, tool-calling agent returns more reliable structured answers (Section 3).

• Encode once, distributed across the fleet. Every robot acts as a distributed sensor, and a modular set of parallel encoder services on its Jetson turns each clip, in a single pass, into multi-purpose representations for memory retrieval, description and reconstruction. Per-clip compute is ofloaded to the edge and scales out with the fleet, and raw camera frames never leave the robot (Section 4.1).

• Disaggregated vision-language inference for describe-on-demand. Cosmos3-Edge is split at its vision– language boundary, with the vision encoder on the robot and the language decoder on the server. Cached vision tokens are encoded once and queried many times, so description cost scales with the questions asked rather than with the video recorded (Sections 4.2 and 4.3).

• A zero-shot, multi-purpose live semantic map. The retrieval embedding the robot already computes doubles as a training-free semantic classifier, in this case reused for trafic classification, interpreted continuously and fed to the fleet manager over its existing VDA5050 interface. An agent sits between the map and the memory as a natural-language interface, so a new site-level insight requires only a new question, with no new model to train (Section 5).

• A modular, scale-out service architecture. Encoders, a GPU-accelerated vector index, a tiered object store, NIM-served agents and on-demand decoders run as independent, versioned services that can each be scaled or upgraded in isolation. We profile the edge encoders on Jetson and model how server compute, bandwidth and storage scale from a single robot to a full fleet (Sections 4.3 and 6.3).

## 2. Preliminaries

Industrial ReMEmbR builds on three main lines of research: large language model (LLM) agents that support versatile tool use, retrieval-augmented generation from semantic embeddings, and the ReMEmbR system, which combines the two to encode and query a robot’s memories over long horizons.

Agents. In this context, an agent is an LLM that runs in an interactive loop with access to predefined tools. At each step in the loop, the model reads the task and its own running history, and then either calls any of the available tools or provides its final answer. ReAct (Yao et al., 2023) features the model writing out its intermediate reasoning between tool calls, so that the result of one call guides the next. Practically, every tool call must be well-formatted to allow for deterministic parsing and automated execution. The loop also needs a limit on the allowed number of steps, so that it always ends with a concrete answer. Industrial ReMEmbR follows this loop, and adds a tool that lets the agent ask a vision–language decoder its own question about any memory it retrieves.

Retrieval-augmented generation. Retrieval-augmented generation (RAG) (Lewis et al., 2020) allows a model to derive answers from a corpus even if it is too large to directly fit in context. The corpus is chunked and indexed in advance, and for each question the � most relevant chunks are retrieved and incorporated in the model’s context window. Retrieval is usually done with a bi-encoder, which embeds queries and documents separately so that search involves a fast vector comparison (Karpukhin et al., 2020). The trade-of is that every document vector is computed before any query is known. A cross-encoder, by contrast, reads a query and a document together, which tends to be more accurate but often too slow to apply to a whole corpus, so it normally only re-ranks a shortlist already retrieved (Nogueira and Cho, 2019) by the bi-encoder. Agentic RAG (Singh et al., 2025) exposes retrieval as a tool for the agent. The agent writes its own retrieval queries, retries upon search misses, and decides when it has suficient evidence to return with a contextually grounded response. Industrial ReMEmbR uses agentic RAG over video instead of text, with a video–text bi-encoder as the index and a decoder that reads the question and the retrieved memory together.

ReMEmbR. ReMEmbR (Anwar et al., 2025) lets a user ask a mobile robot where something is, when something happened, or what it saw over hours of operation, and returns answers the robot can act on, such as an $( x , y , z )$ position to navigate toward. Because a question rarely needs the whole history �, ReMEmbR answers from a small retrieved subset $R \subseteq H$ , approximating $p ( A \mid H , Q )$ with $p ( A \mid R , Q )$ for a question � and answer �.

It works in two phases. In the memory-building phase, VILA (Lin et al., 2024) captions each 3 s video segment, MxBai (Lee et al., 2024) embeds the caption, and Milvus (Wang et al., 2021) stores the embedding with the segment’s mean position and time.<sup>1</sup> In the querying phase, an LLM agent searches this memory by text, position or time for up to three rounds, and then returns a structured JSON answer that holds text, a yes/no value, a time, a duration or a goal position. The accompanying NaVQA benchmark contains 210 questions over seven CODa sequences (Zhang et al., 2024) of 15 to 30 minutes each. A position answer counts as correct within 15 m, and a time answer within 2 min.

The memory is built before any question is asked, so it has to be general enough to serve whichever question comes later. However, ReMEmbR captions every clip when it is recorded. The memory can answer only what the captioner chose to mention, and generating a caption for every clip is the main cost of writing a memory.

Upgraded ReMEmbR keeps its pipeline and updates its models (Section 3). Industrial ReMEmbR keeps the two-phase design but stores no caption, and describes a memory only once a question asks about it (Section 4).

## 3. Upgraded ReMEmbR: Updated Models and Video-Based Retrieval

In ReMEmbR, a caption does two jobs. It serves as the index key, since its text embedding decides which questions can find the clip, and it serves as the context, since it is all the agent reads once the memory is retrieved. Both jobs depend on what the captioner chose to mention when first encoding. “A forklift in a warehouse” may be accurate but still leave out whether the forklift blocks the aisle, in which case a question about blocked aisles becomes less likely to align with the caption’s embedding. Captioning is therefore an information bottleneck for retrieval, since a scene detail has to survive both the captioner’s limited attention and the embedding step to play any role in the search. To build toward Industrial ReMEmbR, we first introduce Upgraded ReMEmbR. It keeps ReMEmbR’s pipeline, updates its underlying models, and moves the index key from the caption to the video itself, circumventing the retrieval bottleneck.

Upgraded ReMEmbR retains ReMEmbR’s three-second segments, its position and time metadata, and its search by content, position and time, so that any change on NaVQA can be attributed to the models. Two of the three model updates are direct replacements. Cosmos3-Edge (NVIDIA, 2026a) takes over captioning from VILA-1.5-13B (Lin et al., 2024). It is roughly one-third the size, and its emphasis on physical-world and spatial understanding suits questions about how a scene is arranged. Nemotron-3-Nano-4B (NVIDIA, 2026b; NVIDIA et al., 2025) takes over the agent role from Llama-3.1-8B (Grattafiori et al., 2024), ReMEmbR’s locally hosted agent. It is a hybrid Mamba-Transformer model with half the parameters, post-trained for tool use and structured output, and it has access to the same search tools as before.

The third update changes what the index is built from. ReMEmbR embeds the caption text with MxBai (Lee et al., 2024). Upgraded ReMEmbR embeds the clip itself with Cosmos-Embed1 (NVIDIA, 2025), which maps video and text into a shared semantic space, so a natural-language query is matched against the content of the video (Fig. 2). The caption still accompanies each retrieved memory as context, but it no longer limits what the search can find.

Upgraded ReMEmbR leaves the caption in place as context, so generating it remains the bulk of the cost of writing a memory. Section 6 evaluates each of the three replacements on NaVQA, changing one component at a time.

Original ReMEmbR  
![](images/350165983acd5a5d9edf14e8d3ff6ec227f2d79336ac790b7d0638dc2c5c2a3d.jpg)

Upgraded ReMEmbR  
![](images/9af99fd0067aaacb8578a57223211d46bc85bbc18b343ef9d2b1956b4a349c45.jpg)  
Figure 2: Upgraded ReMEmbR indexes video directly instead of caption text, so retrieval is no longer limited to what the captioner chose to say. Original ReMEmbR is shown on top and Upgraded ReMEmbR at the bottom, with the replaced components in green. In the original, the index is built from caption text, so retrieval can find only what the captioner chose to say. Upgraded ReMEmbR sends each clip down two independent paths: Cosmos-Embed1 produces the video embedding that serves as the index key, and Cosmos3-Edge produces the caption the agent reads once an observation is retrieved. Both workflows still caption every segment in advance. Bidirectional arrows denote agent tool calls and returned evidence.

## 4. Industrial ReMEmbR: Encode Once, Interpret Continuously, Elaborate on Demand

Upgraded ReMEmbR showed that a memory can be found without its caption, yet it still generates one for every clip before any question is known. Industrial ReMEmbR goes further and eliminates that step. Each robot stores the inputs that a server-side model would need to describe a memory, and the description itself is generated only when a question calls for one (Fig. 3).

Three principles guide the design. The first is to encode once. Each clip passes through the robot’s encoders a single time and comes out as the representations that later steps read: an embedding for retrieval, vision tokens for description, and a video latent for inspection. No later step has to return to the camera frames, which never need to leave the robot. Per-clip work therefore runs on the robot, and its cost grows with the length of the footage. The second is to interpret continuously. The server scores each incoming embedding as it arrives, which keeps the live semantic map current without generating any text. The third is to elaborate on demand. Text is generated once a question is known, so it can address that question, and the same memory can be described again for the next one. Per-question work therefore runs on the server, and its cost grows with the number of questions asked rather than with the amount of video recorded. The following subsections follow these principles in turn: encode once on the robot (Section 4.1), interpret continuously on the memory the robots feed (Section 4.2), and elaborate on demand on the server (Section 4.3).

![](images/166ec28bb63bece4ef9cec82d048ea601d21a0f33d90073ba13afad40a2c0a69.jpg)  
Figure 3: Industrial ReMEmbR defers text generation until a question arrives, whereas Upgraded ReMEmbR (top) fixes the text an answer can draw on when the clip is recorded. Upgraded ReMEmbR runs the full Cosmos3-Edge captioner on every clip; Industrial ReMEmbR (bottom) runs only encoders at that point and keeps their outputs in fleet memory. The language half of Cosmos3-Edge runs once a question has arrived, on the memories the agent retrieves, and the Wan2.2 decoder turns a stored latent back into video when someone wants to see the scene. The dark tag marks the step that generates text. Green boxes are models, blue boxes are storage.

## 4.1. On the Robot: One Pass, Three Representations

Each robot runs iw.encode, a set of three encoder services on its Jetson AGX Xavier, on every three-second segment. Each encoder produces the input for a later application, and none of them needs the camera frames again.

Cosmos-Embed1 maps the segment to a single 768-dimensional vector in a shared video–text space, which takes 3 kB. This vector is the memory’s index key, and the agent’s content search reads it (Section 4.3). As in Upgraded ReMEmbR, it replaces ReMEmbR’s MxBai embedding of the caption, so retrieval no longer depends on what a caption mentions. Text prompts map into the same space, so the server can score the same vector for the trafic verdict of Section 5 and can support other prompt-based checks, each at the cost of a dot product.

The vision encoder of Cosmos3-Edge produces 60 projected vision tokens of width 2048, which the model’s language decoder would normally receive from it directly. The robot stores these tokens with the metadata the decoder needs to read them, 247 kB in total. The language decoder on the server reads them when the agent asks about a retrieved memory (Section 4.3). They replace the caption that ReMEmbR generates for every clip, so the robot never runs the expensive step of generating text on edge.

The Wan2.2 VAE (Team Wan, 2025a,b) encoder compresses the segment into a 48 × 2 × 30 × 52 video latent of 300 kB, one-twentieth of the size of its uncompressed frames. The Wan2.2 decoder on the server reads this latent when an operator wants to revisit a memory, and reconstructs the scene for inspection. ReMEmbR has no counterpart to it, since its memory keeps the caption’s embedding and not a compression of the scene itself.

The three encoders run in parallel as independent services, and their outputs are the only data that leave the robot (Fig. 1). Each memory covers three seconds in our configuration, and how often a robot writes one is a deployment setting. NaVQA uses contiguous segments; fleet resource projections assume one write every 12 s, while the live map polls every 10 s. Encoder timing and memory on Xavier are reported in Tab. 7, Tab. 8 and Appendix D.1.

## 4.2. Fleet Memory

Fleet memory has two parts, matched to how each kind of data is used. Every segment adds one row to a shared Milvus collection (Wang et al., 2021), containing its Cosmos-Embed1 vector, the robot’s pose, the capture time and any verdict the server computes as the row arrives, which is what lets the server interpret continuously without reading the larger data. Indexes over the vector, the position and the time support search by content, place and time, and NVIDIA cuVS (RAPIDS Team, 2024) builds and searches the vector index on the GPU. The vision tokens and the latent are much larger and are read only when a memory is retrieved, so they are kept in an object store, and the row holds a reference to them.

ReMEmbR’s reference deployment runs Milvus on the robot (Anwar et al., 2025). Here, we centralize it on the on-premises server to ofload database and index-serving work from robots, avoid duplicated fleet indexes and per-robot reconciliation, and provide the fleet a shared live database of memories to independently contribute to and query from.

## 4.3. On the Server: Answering Questions

Questions are answered on an on-premises server by a Nemotron-3-Nano-4B agent. As in ReMEmbR, the agent searches the index by content, position and time, and the entries it retrieves accumulate in its working memory. Each retrieved entry now holds a reference to its cached vision tokens in place of a caption, and a fourth tool lets the agent put a question of its own to the Cosmos3-Edge language decoder about any entry. The agent therefore decides both which moments to look at and what to ask about them.

This is where the server elaborates on demand. The decoder answers from the cached tokens alone (Fig. 4). It builds its usual prompt from the chat template and the question, and it places the stored vision tokens in the slot where freshly encoded video would normally go. This split turns an intermediate model output into a lasting memory. Edge–server partitioning such as Neurosurgeon (Kang et al., 2017) chooses a split layer per request to reduce its latency or energy, and the intermediate output is discarded once that request is served. Here, the boundary sits at the model’s vision–language interface, and the robot’s tokens persist in fleet memory long after they are encoded. One set of tokens can answer any number of questions.

![](images/8ff766f258923c851f7cbff58e0bc8b5349f0fd5252792200adbe98d69104bd3.jpg)  
Figure 4: Cached vision tokens let the server describe a scene on demand, without the source frames. The robot runs only the vision half of Cosmos3-Edge and stores its vision tokens (green). When a question arrives, the server’s language model receives the usual chat template and the embedded question (gray), with the cached vision tokens spliced into the slot where encoded video would normally go, after checking that the token counts match. The same cached tokens can answer any number of diferent questions.

After one or more rounds of search, the agent returns a structured answer with text, yes/no, position, time and duration fields that downstream systems can consume directly. An operator can query the agent interactively, and a service can call it on a schedule, as the live map does in Section 5. When an operator wants to see a scene, the Wan2.2 decoder reconstructs it from the stored latent. New typed tools can expose site-specific tasks to the agent without any change to how memories are encoded.

Every component runs as a versioned service on the local network. Our reference setup places fleet memory, cuVS, both decoders and the NIM-served agent on one on-premises server with NVIDIA RTX PRO 6000 Blackwell Server Edition GPUs. Larger sites can distribute the agent and the decoders across dedicated GPUs and provision Milvus, the object store and the map service separately, and each modular service can be scaled or upgraded in isolation (Appendix A.2).

## 5. A First Application: The Live Semantic Trafic Map

We return to the blocked aisle example from the introduction. The first robot to pass the clutter stores a memory of it like any other clip, and whether the aisle ahead is occupied is already implicit in that memory’s embedding. When the live semantic map next refreshes, it finds that this embedding vector aligns more closely with those of descriptions of an occupied aisle than those of a clear one, and the aisle’s zone turns orange. The fleet manager can then take the blocked aisle into account when planning routes in that area. Further, an operator who wants to know what exactly is in the way can click the robot and read a description of what it saw, generated on the spot from the memory’s cached vision tokens. None of this changes the robots, which run the same iw.encode pipeline as in Section 4.1. The server adds two cached text prototypes and a map service that queries the agent on a fixed clock, and the rest is Industrial ReMEmbR (Fig. 5).

The report adds no robot-side computation, as it reuses the Cosmos-Embed1 vector v generated for retrieval as part of the deployed iw.encode pipeline. Cosmos-Embed1 projects video and text onto a shared embedding space, so a written description can be directly compared with a recorded video. We describe each of the two outcomes with a handful of concise text prompts, such as “a warehouse aisle blocked by a pallet” for an occupied aisle and “an empty warehouse aisle” for a clear one (Appendix F). Averaging the embeddings of each set and rescaling the result to unit length gives one prototype per outcome, $\mathbf { p } _ { \mathrm { o c c } }$ and $\mathbf { p } _ { \mathrm { c l r } }$ , following the prompt ensembling that CLIP introduced for zero-shot classification (Radford et al., 2021). The server caches both prototypes, so the prompt ensembles can be maintained and updated. Unlike CLIP’s evaluation, where each image is embedded solely to be classified, the verdict reuses the vector the robot already uploads as the memory’s index key, so it requires no separate classification pass. As each vector arrives, the server’s verdict on the clip is trafic when the clip is closer to the occupied prototype,

![](images/e3e7a23092487b4ec4a98fc21599558c54a52c6cc9e208016b214879079ef9fe.jpg)  
Figure 5: Robots send only a vector, pose and time, and the server turns them into a live trafic map. (1) Each robot embeds its camera clip with Cosmos-Embed1 and sends only the vector, its pose and the time. (2) The vector is compared with cached prompt vectors for an occupied and a clear aisle, and the closer one becomes the verdict stored with the memory. (3) Every 10 s, the map service asks the agent whether there is trafic ahead of each robot. (4) Each answer colors the robot’s zone, and the pose of the memory the agent retrieved is drawn beside the robot. Green boxes are NVIDIA models, blue boxes are storage.

$$
\mathbf { v } ^ { \top } \mathbf { p } _ { \mathrm { o c c } } > \mathbf { v } ^ { \top } \mathbf { p } _ { \mathrm { c l r } } ,\tag{1}
$$

and clear otherwise. The trafic verdict is thus derived on the server straight from the embedding the robot already computes at encoding time, at the cost of two dot products, without a detector, captioner or any other trained model dedicated to trafic classification.

A single similarity score is a weak signal. A clip’s embedding describes the whole scene, including its background, colors and lighting, so its similarity to any short prompt is low, and the scores for the two outcomes end up close together. This score attenuation is a direct consequence of the average pooling applied to the Q-Former tokens in the Cosmos-Embed1 architecture. Because this pooling operation collapses all visual tokens into a single global vector, v encodes the entire scene thereby diluting the specific signals. By contrasting v against the two opposing, generalized prototypes, we isolate the relevant semantic axis and bypass the dilution caused by the global visual pooling. A fixed threshold on either score would be fragile, so the verdict compares the two directly. Their diference, $\mathbf { v } ^ { \top } ( \mathbf { p } _ { \mathrm { o c c } } - \mathbf { p } _ { \mathrm { c l r } } )$ , cancels whatever the prototypes have in common, such as the aisle itself, and keeps only how much more the clip looks like an occupied aisle than a clear one.

The server stores the verdict in the same memory row as the embedding, its pose, its timestamp and the reference to the clip’s cached vision tokens. Next to the verdict, the row holds a short sentence such as “In the aisle straight ahead there is trafic,” filled in from a fixed template. The agent can read this sentence directly, and writing the memory still generates no text.

The map combines this memory with what the fleet manager already knows. Under VDA5050, each AMR regularly sends the fleet manager a state message whose agvPosition field reports the robot’s pose, its position (�, �) and heading �. Fleet memory adds what each robot sees. The map service divides the floor plan into zones, polygons that each cover an aisle segment or a crossing, and keeps one trafic state per zone.

![](images/4f52dc4626953f1f8b0c1748ceb697e2f3cfc6239b597190c5f8b3ff9e1c9fcb.jpg)  
Figure 6: Zone colors encode the agent’s latest answer per zone, and an operator can request a description of the memory behind it. Each zone takes the color of the agent’s latest answer for the robot inside it, and each robot is drawn at its VDA5050 pose beside the memory the agent retrieved for it. Selecting a robot shows its answer and lets the operator request a description, which the Cosmos3-Edge decoder writes from the vision tokens already in memory. The zone layout and robots are illustrative.

To minimize redundancy, the map service does not react to every memory a robot writes. It polls on a fixed period of ∆� seconds (10 s by default). The request rate follows the robot count and the poll interval. The compute each request takes also depends on the agent’s work and the length of its answer. On each tick it takes the latest pose of every robot that reported since the last one and looks up which zone the robot is in. Robots outside all zones are skipped. For the rest, it asks the agent a question that carries the robot’s time and position:

“The time now is HH:MM:SS. You are a robot at position [�, �, �] in a warehouse. . . . Is there any other

vehicle or agent in the zone directly ahead of you right now? Answer with only one word yes or no.”

## (full text in Appendix F.2)

The agent answers it the way it answers an operator, by searching memory with its text, position and time tools. Each query runs with a fresh agent state, so no memories carry over from the previous robot and the tool-call budget is reset. A yes labels the zone as trafic and a no relabels it as clear. Any other output triggers a re-query (Fig. 6).

While updating the map does not strictly need the agent in this narrow case, since the verdict is already available with each memory, we maintain the agent as an adaptive natural language interface. It decides how to answer, relying on a robot’s latest memory or checking it against what other robots recorded in the same aisle shortly before. The map service only sends it a question, and a diferent question entails no further changes across the pipeline. We see trafic as the first layer of a multi-purpose semantic map of industrial sites, whose zones could eventually record whether a charging station is free, whether a dock is filling up, or whether a recognizable landmark is still where a navigation policy expects it. Building a dedicated classifier for each of these would mean new models, new memory fields and new robot software every time a unique need arose. With an agent exposing a natural language interface between the semantic map and the shared memory, each new need becomes a new question posed to the same interface.

Each update is pushed to a web page as soon as it is computed. Robot markers follow the reported poses, and zone colors follow the latest answers. Beside each robot, the map also marks where the memory behind its answer was recorded, so an operator can see what the answer was based on. Clicking a robot asks for more detail about its latest memory (Fig. 6, right; Fig. 7). The Cosmos3-Edge language decoder reads that memory’s cached vision tokens together with a general request to describe the scene, and the operator receives a conditionally generated description.

![](images/f2092a5bf77727814cd528bede735203580f9e035e8c8a4e471e942dc2df465e.jpg)  
Figure 7: Operators can request an on-demand description from the Industrial ReMEmbR frontend. The popover window allows users to generate on-demand descriptions using vision tokens.

## 6. Results

Sections 3–5 make three changes to ReMEmbR. They replace its models and index the video itself, they store encoder outputs in place of captions, and they read a live trafic map from those outputs. Our experiments test whether each change does what it was meant to do, and whether the resulting system remains afordable across a fleet. They are organized around four questions:

• Do the new models retrieve and answer better? (Section 6.1)

• Can one encoding answer any later question? (Section 6.2)

• Does the cost scale to a fleet? (Section 6.3)

• Can the memory keep a live trafic map? (Section 6.4)

Tab. 1 summarizes the main results.

## 6.1. Do the New Models Retrieve and Answer Better?

Upgraded ReMEmbR replaces all three models of ReMEmbR. Nemotron-3-Nano-4B takes over from Llama-3.1- 8B as the agent, Cosmos3-Edge from VILA-1.5-13B as the captioner, and a Cosmos-Embed1 video index from the MxBai caption-text index. The pipeline around these models is unchanged, so any diference on NaVQA comes from the models, and swapping in one model at a time shows which model causes which diference. NaVQA (Anwar et al., 2025) tests long-horizon memory over all seven CODa campus sequences, where each memory holds one three-second clip with its capture pose and time. We run three trials per configuration, each answering 67 yes/no questions, for 201 responses (Appendix B). Retrieval recall is the fraction of annotated evidence segments that appear in any of the agent’s tool results, so it depends on both the index and the queries the agent writes. Reliability is measured byfirst-attempt parse failures, answers the evaluator cannot parse on the first try, and by unscorable answers, which stay unparsable after retries. Position errors are medians over parsed answers.

<table><tr><td>Result</td><td>Compared</td><td>Baseline</td><td>This work</td><td>Section</td></tr><tr><td>First-attempt parse failures (NaVQA)</td><td>Original vs. Upgraded ReMEmbR</td><td>20.7%</td><td>13.8%</td><td>6.1</td></tr><tr><td>Unscorable answers (NaVQA)</td><td>Original vs. Upgraded ReMEmbR</td><td>4.5%</td><td>1.9%</td><td>6.1</td></tr><tr><td>Retrieval recall (NaVQA)</td><td>Caption-text vs. video index</td><td>20.8%</td><td>24.6%</td><td>6.1</td></tr><tr><td>Description GPU equiv. (100 robots)</td><td>Every clip vs. 10% described</td><td>36.25</td><td>1.04</td><td>6.3</td></tr><tr><td>Traffic balanced accuracy</td><td>Best caption vs. embedding</td><td>65.3%</td><td>71.3%</td><td>6.4</td></tr></table>

Table 1: Main results and their baselines.

<table><tr><td>System</td><td>Swapped in</td><td>First-try fail. (%)↓</td><td>Unscorable (%) ↓</td><td>Recall (%)↑</td><td>Position err. (m) ↓</td></tr><tr><td>Original ReMEmbR</td><td>一</td><td>20.7</td><td>4.5</td><td>22.9</td><td>69.1</td></tr><tr><td rowspan="3">One swap</td><td>Agent: Nemotron</td><td>16.2</td><td>0.9</td><td>18.7</td><td>71.0</td></tr><tr><td>Captioner: Cosmos3-Edge</td><td>22.4</td><td>3.6</td><td>26.7</td><td>74.6</td></tr><tr><td>Index: Cosmos-Embed1</td><td>23.5</td><td>5.5</td><td>29.7</td><td>79.4</td></tr><tr><td>Upgraded ReMEmbR</td><td>All three</td><td>13.8</td><td>1.9</td><td>24.6</td><td>66.6</td></tr></table>

Table 2: Swapping one component of original ReMEmbR at a time on full NaVQA, three trials per configuration. Original ReMEmbR uses Llama-3.1-8B, VILA-1.5-13B and MxBai. Each one-swap row replaces one of them and keeps the other two. Bold marks the best single swap in each column where it beats original ReMEmbR. Tab. 5 reports all eight configurations.

Each new model improves a diferent measure (Tab. 2). Nemotron is the one that makes answers reliable. Swapping it in alone cuts first-attempt parse failures from 20.7% to 16.2% and unscorable answers from 4.5% to 0.9%, whereas swapping in the captioner or the index instead raises first-attempt failures to 22.4% and 23.5%. The efect does not depend on what the agent reads. Across all four captioner–index pairs of Tab. 5, Nemotron stays within 12.2–16.2% first-attempt failures against 20.7–23.5% for Llama, and within 0.9–1.9% unscorable answers against 3.6–5.5%. Nemotron has half the parameters of Llama, so this gain likely reflects its post-training for tool use and structured output rather than its size.

Recall, in contrast, follows the index. Replacing the caption-text index with Cosmos-Embed1 alone raises recall from 22.9% to 29.7%, the largest gain of any single swap, and the video index matches or improves recall for every agent–captioner pair. With Nemotron as the agent, it adds 8.5 points over caption-text indexing with VILA captions and 3.8 points with Cosmos3-Edge captions. This is the bottleneck of Section 3. A caption-text index can find only what the captioner chose to mention, while the video index matches a question against the clip itself, and so it needs neither a caption nor a caption-text embedding stage. Industrial ReMEmbR relies on this property when it stops writing captions altogether.

Combined, the three models improve on original ReMEmbR in both reliability and recall. First-attempt parse failures fall from 20.7% to 13.8% and unscorable answers from 4.5% to 1.9%, relative reductions of about 33% and 58%, and median position error falls from 69.1 to 66.6 m. Recall rises from 22.9% to 24.6%, less than the index swap alone achieves. Because recall also depends on the agent’s queries, Nemotron lowers it by 2.3–5.9 points relative to Llama in every pair, and the full stack keeps only part of the index’s gain. In exchange, Nemotron lowers median position error by 7.7–11.9 m relative to Llama whenever it is paired with either new component.

Upgraded ReMEmbR therefore finds more evidence and answers more reliably, yet it still captions every clip as it is written. The next question is whether Industrial ReMEmbR can drop that step without losing what the caption provided.

## 6.2. Can One Encoding Answer Any Later Question?

Industrial ReMEmbR stores Cosmos3-Edge vision tokens in place of a caption and splits the model at its vision– language boundary (Section 4). The split is only worthwhile if decoding from stored tokens gives the same answer as running the full model on the frame, and if elaborating a memory is fast enough to fit inside an

agent’s answer. We test both on a separate run of 150 trafic frames (Appendix C). The vision encoder encodes each frame once, and the language decoder answers three questions from the cached tokens. The same frames and questions also run through the unsplit model for comparison. All timings use one NVIDIA RTX PRO 6000 Blackwell Server Edition GPU.
<table><tr><td>Operation</td><td>Reads from memory</td><td>Output</td><td>Tokens</td><td>Time (s)</td></tr><tr><td>Describe</td><td>Vision tokens, 247 kB</td><td>One-word answer</td><td>2</td><td>0.03</td></tr><tr><td></td><td></td><td>List of obstacles</td><td>43</td><td>0.38</td></tr><tr><td></td><td></td><td>Scene description</td><td>146</td><td>1.25</td></tr><tr><td>Inspect scene</td><td>Latent, 300 kB</td><td>Five-frame scene</td><td></td><td>1.76</td></tr></table>

Table 3: Time to elaborate one memory on the server. Descriptions use the Cosmos3-Edge language decoder and scene inspection the Wan2.2 decoder. Description times and token counts are means over the 150 frames of that run; the inspection time is a median.

The split model reproduces the full model exactly. Under greedy decoding, decoding from cached tokens produced the same text as the unsplit Cosmos3-Edge for all three questions on every frame, 450 of 450 cases, and tokens written to the object store and read back later decoded through the same path. Splitting the model therefore leaves its outputs unchanged, and a later question can be answered from memory as if the frame were still at hand.

Encoding is paid once per memory and decoding once per question. Encoding a frame’s vision tokens takes 0.056 s, after which the same tokens answer a one-word check, a list of obstacles or a full scene description (Fig. 8). Under these fixed input and decoding settings, a 146-token description takes 1.25 s (Tab. 3); generation time is approximately proportional to output length for this workload.

Reconstructing a scene for human inspection works the same way. A five-frame 480×832 clip is stored as a 48 × 2 × 30 × 52 BF16 latent of 300 kB, one-twentieth of its uncompressed size; casting the FP32 encoder output to BF16 halves storage and changes the reconstruction metrics below by under 0.1% (Tab. 6), and the Wan2.2 decoder has a median reconstruction time of 1.76 s (Tab. 3). Reconstructions of the 150 frames reach 39.7 dB PSNR, 0.974 SSIM (Wang et al., 2004) and 0.0052 LPIPS (Zhang et al., 2018), with the remaining diferences mostly in fine textures and thin edges (Fig. 9). An operator who inspects a memory therefore sees a close reconstruction of the frames the robot recorded.

![](images/44e8503d5ffa841f278d4db57903b7856c5d7aabaf8df86e000306df4fd4b7da.jpg)  
Figure 8: One encoding answers many questions: the cached vision tokens serve every description without the frame. (1) A frame is encoded once into vision tokens, assigned to the robot in deployment. (2) The server decodes the logged answers on the right from the same cached tokens, without receiving the frame. The scene description is excerpted; its token count and time refer to the full response.

Forklift ahead  
Robot ahead  
![](images/40efd4a2a5f11eaae2bd4dd1c8d1da26f9151a56dd1b657b752a375d734034e2.jpg)  
Clear aisle  
Figure 9: A stored latent is enough to reproduce a scene on the server for inspection. Shown are three observations of the trafic run. Top: reference observations used in the fidelity evaluation. Middle: the scene reconstructed on the server from each memory’s stored latent. Bottom: absolute pixel diferences, with full brightness marking an error of at least 10% of the intensity range. The examples cover a forklift ahead, a robot ahead and a clear aisle and were selected for scene content variety.

## 6.3. Does the Cost Scale to a Fleet?

Industrial ReMEmbR moves encoding onto the robots and generation onto the questions, so its cost has an edge side and a server side, which we measure separately. Edge timings use warmed TensorRT 8.5.2 engines on a Jetson AGX Xavier with the inputs of Tab. 4, and they include transfers between host and device (Appendix D.1). Server measurements use one RTX PRO 6000 Blackwell Server Edition GPU unless stated otherwise. Where we extrapolate to a fleet, we assume one memory per robot every 12 s over an eight-hour day and mark the result as modeled (Appendix D.3). The live map’s 10 s poll interval is a separate setting (Section 5).

On the robot. Robots run only encoders, and the language decoder stays on the server. The Cosmos3-Edge vision encoder takes 196 ms per frame, the Wan2.2 VAE 6.46 s for five frames in FP32, and Cosmos-Embed1 11.76 s for six frames (Tab. 7). Cosmos-Embed1 dominates encoding time and is the first target for engine optimization (Section 7). GPU memory is less of a constraint. The three encoder services together hold 6.55 GiB, about a quarter of the 26.3 GiB peak of the original VILA captioner (Fig. 10).

![](images/cdec63d75ea126c1619a66a5bb1013dc51a0dfd75af81145860c3bc595884ce7.jpg)  
Figure 10: The three iw.encode services fit in 6.55 GiB of resident GPU memory, against a 26.3 GiB sampled process peak for the original server-side captioner. The figure shows GPU-memory profiles for both (Tab. 8); Cosmos3-Edge loads only its vision encoder.

On the server. Original ReMEmbR’s write-time captioning takes a mean of 4.35 s per memory, whereas a description in Industrial ReMEmbR costs 1.25 s from cached tokens and is generated only when requested. We measure server load in GPU equivalents, the GPU time requested per second, so one equivalent keeps one GPU fully busy. At the modeled write interval of 12 s, captioning every memory represents 36.25 GPU equivalents for 100 robots. If 10% of memories are each described once, the serialized demand is 1.04 GPU equivalents; at 1%, it is 0.10. At a fixed write interval and description fraction, both loads scale linearly with fleet size. Describing 10% of memories once yields a 34.8-fold reduction. This combines the move from VILA captioning to Cosmos3-Edge decoding with deferral. Deferral costs less than captioning unless memories are described more than 3.48 times each on average. These are serialized single-GPU service-demand estimates before provisioning headroom, modeled from measured per-memory costs (Fig. 11). Appendix D.3 adds the live map’s agent load, sized below, and scene inspection as a symbolic term.

The agent must also keep up as the map polls more robots at once. Original ReMEmbR serves its agent through Ollama, whose throughput in our deployment barely grows with concurrency. At an 8k context, Ollama with Q8\_0 grows from 203 to 235 output tokens/s as concurrent requests rise from one to eight, while NIM in FP8 grows from 236 to 1,217 tokens/s, about five times Ollama’s throughput at eight requests (Fig. 13). Ollama’s scheduler serves the Nemotron architecture families, including Nemotron-3-Nano-4B, one request at a time (Morgan, 2026). We measure how many robots the agent can serve under load. With one GPU serving the agent, 95% of map updates finish within 20 s for 24 robots polling every 10 s, and two GPUs serve at least 48 (Appendix D.2). The map therefore costs about 0.042 GPU equivalents per robot, or 4.2 for 100 robots. With 10% of memories described, the server’s generation load for 100 robots is about 5.2 GPU equivalents, still below the 36.25 of captioning alone, although captioning is costed unbatched here and batching it would narrow the gap. Because most of this load comes from map polling, the poll interval outweighs the description fraction as the main lever on server cost at fleet scale.

Captioning GPU demand: always-on vs. describe-on-demand  
![](images/a0a66dbf8285a63062f4e6687c136f07a0e2d7fc97b536486dee2067148b4eb0.jpg)  
Figure 11: Measured GPU time for captioning, extrapolated at one memory per robot every 12 s. Always-on captions every memory as written (4.35 s). Describe-on-demand captions a memory only when asked (1.25 s), for 10% and 1% of memories. Squares mark 100 robots.

Each memory adds 0.55 MB of objects, almost all of it the 247 kB token cache and the 300 kB latent, plus a 3.2 kB index row (Appendix A.1). At the modeled 12 s write interval, each robot uploads 1.3 GB per eight-hour day, an average of 0.37 Mbit/s; a fleet of 100 robots adds 131 GB of objects and 0.768 GB of index per day, or 3.9 TB and 23 GB per month (Tab. 12). Search reads the index and runs on the GPU. Within Milvus (Wang et al., 2021), the cuVS GPU exact index (RAPIDS Team, 2024) serves 910 exact queries per second at 32 concurrent queries, compared with 47 for CPU exact search, with identical results (Fig. 14). Its advantage over CPU exact search grows from parity at 10,000 rows to 19-fold at one million rows; exact-search work still grows with retained rows (Fig. 14).

## 6.4. Can the Memory Keep a Live Trafic Map?

The live map of Section 5 reads a trafic verdict directly from each memory’s Cosmos-Embed1 vector, without a detector or a captioner. The map is only useful if this verdict is accurate and if the agent passes it on unchanged, so we compare it with caption-based memories under the same agent. The test set is a deterministic, contiguous warehouse simulation of 914 frames, each stored as a single-frame memory with its pose and time and paired with one yes/no question about trafic ahead. Labels come from the simulator’s record of other agents in the zone ahead of the robot, giving 607 trafic frames and 307 clear ones. Trafic is the majority class, so always answering “trafic” already scores 66.4% accuracy. We therefore report balanced accuracy, the mean of trafic recall and clear recall, on which that baseline scores 50.0%.

The three memories difer only in what each frame writes. The embedding verdict scores the frame’s Cosmos-Embed1 vector against the occupied and clear prototypes of Section 5, with no tuned threshold. For detection, Cosmos3-Edge lists the obstacles in the forward path as labeled bounding boxes, and for description it writes a free-text description of the scene, both with reasoning enabled. Before answering, the agent must search memory by the frame’s position, so it reads every representation from the same memory row. A frame counts toward a representation’s coverage when that representation produced usable content for it. Prompts and prototypes are listed in Appendix F, and protocols in Appendix E.

The embedding verdict reaches 71.3% balanced accuracy, 6.0 points above detection, the best caption-based memory, and it is the only representation that covers all 914 frames, against 844 for detection and 902 for description (Fig. 12). The caption arms are scored only on the frames they cover, so counting dropped frames as errors would only widen this margin. The two caption-based memories fail in opposite directions. Detection finds trafic in 88.4% of trafic frames but recognizes only 42.3% of clear frames, so its 72.7% accuracy, slightly above the verdict’s 72.1%, mostly reflects the majority class. Description errs the other way, with 91.1% clear recall and only 27.1% trafic recall. The verdict balances the two classes, with 73.8% trafic recall and 68.7% clear recall (Tab. 13).

Routing the verdict through the agent barely changes its classification. Nemotron agrees with the raw embedding verdict on 913 of 914 frames, scoring 659/914 against the direct classifier’s 660/914. It retrieves the target memory in 99.6% of cases and produces no unparseable answers. The map can keep the agent as its interface, which lets it ask new questions of the same memory.

## 7. Road Ahead

The trafic map is the first application built on fleet memory. The directions below reuse the same encoding and storage, and several of them require only new tools on the server.

New uses of the same memory. The agent can already locate any landmark that any robot has seen, yet it uses that knowledge only to answer questions. Recast as semantic subgoals, such as “pass the blue loading bay, then turn left at the marked crossing,” that knowledge could guide a robot along routes it has never driven, drawing on what other robots have already observed. The live map’s zone verdicts can play the same role for fleet scheduling. RobotFleet (Gupta et al., 2025) allocates subtasks across a heterogeneous fleet from a shared world state of declarative statements and replans when a robot reports something new. An occupied zone can update that state as soon as its memory arrives, and the agent can be asked what is blocking the zone before the next allocation.

Fleet memory as training data. Each robot already encodes its clips into the vision tokens and video latents that compatible Cosmos3 models consume (NVIDIA, 2026a), so routine fleet operation doubles as distributed preprocessing for training. This costly visual encoding is ofloaded to edge devices across the fleet, paid once per memory, and amortized over every epoch and every compatible model trained from the cached representations, without transferring raw video or re-running the encoders (Appendix G). A world model trained on these latents could become a learned simulator of the site, replaying recorded scenes under alternative robot behaviors to show what would have happened had a robot taken another aisle, therefore translating failed rollouts into successful ones and distilling back to deployed policies and closing the flywheel. Fine-tuning the language decoder on the cached vision tokens could teach it the site’s own objects and terminology. Fine-tuning Cosmos-Embed1 with location-aware captioning objectives, as in LocCa (Wan et al., 2024) and SigLIP 2 (Tschannen et al., 2025), could also strengthen the spatial awareness that physical AI tasks require, thereby improving the balanced accuracy on trafic verdicts.

![](images/4d4156cce00f7db0c2e48a56e8ff270f7d609e33738111b9583f615e5c8cbf00.jpg)  
Figure 12: The caption-free embedding verdict (green) has the highest balanced accuracy among the reported representations. Results are on simulated warehouse trafic over 914 frames, by what each frame writes to memory; caption arms use reasoning. Scores exclude frames without usable context; coverage is listed at the right. The dashed line marks chance, where always answering “trafic” sits.

A cheaper memory at every stage. On the robot, the general-purpose video–text embedder answers the trafic question with two dot products but dominates encoding time. An embedder specialized for industrial scenes and distilled into a compact backbone such as SigLIP 2 (Tschannen et al., 2025) could encode faster on the edge and sharpen every prompt-based check on factory and warehouse vocabulary. On the server, the edge deployment expertise behind our encoders could yield a leaner agent, pruned, quantized and distilled from open models, that answers map polls with fewer calls and tokens. In storage, every memory is kept in full, so the store grows linearly, reaching 3.9 TB per month for 100 robots at the modeled write interval (Tab. 12). Much like human episodic memory, fleet memory could consolidate routine observations of the same place while keeping surprises in full, so that storage grows with the site’s rate of change rather than with fleet operating hours.

The physical world as an attack surface. Text that the decoder reads from a scene enters the agent’s context like any other content, so a sign or a screen in view of a camera becomes an input the agent might obey, a physical form of prompt injection. As the agent gains authority over the fleet, perception must be treated as untrusted data. iw.shield, a protection layer under development, is our first step toward agents that can read the world without taking malicious orders from it.

## 8. Conclusion

We present Industrial ReMEmbR, a shared fleet memory that decouples observation encoding from on-demand interpretation. Each robot encodes a clip once into a retrieval embedding, vision tokens, and a video latent while retaining raw camera frames onboard. These representations support semantic retrieval, live trafic mapping, scene description, and reconstruction. We derive trafic classifications directly from retrieval embeddings at encoding time and generate scene descriptions and reconstructions only when needed. Encoding, storage, mapping, and interpretation operate as independent services, allowing multiple applications to reuse each observation and individual models to be replaced independently.

Our evaluation demonstrates the benefits of deferred description and validates key components of the architecture. Deferring caption generation removes text generation from the memory-write path. Video-based indexing matches or improves retrieval recall in every matched comparison, and cached-token decoding reproduces the unsplit Cosmos3-Edge output in all 450 tests. On full NaVQA, the updated model stack reduces first-attempt parse failures from 20.7% to 13.8% and unparsable answers from 4.5% to 1.9% with an agent half the parameter size. At a modeled 12 s write interval, description demand for 100 robots is 1.04 GPU equivalents when 10% of memories are described, compared with 36.25 GPU equivalents for captioning every memory. Across 914 simulated warehouse frames, embedding-based trafic classification achieves 71.3% balanced accuracy, exceeding the best reasoning-enabled caption-based alternative by 6.0 percentage points.

Further engineering is required for production deployment. Cosmos-Embed1, the slowest encoder on Jetson AGX Xavier, is our first target for TensorRT optimization and fine-tuning, while agent provisioning must account for the map-update workload and polling cadence. We plan to extend the system to congestion-aware fleet planning, semantic guidance for navigation policies, and continuous curation of deployment data for model finetuning. We will also investigate deploying agents on robot edge devices, reorganizing encoding components on the robot to obfuscate people from memories directly on the robots’ volatile memory, and developing iw.shield to protect the system against cyberattacks. The broader contribution is a reusable memory architecture through which a fleet’s collective experience can support an expanding range of applications as models, tasks, and warehouse conditions evolve.

## References

Abrar Anwar, John Welsh, Joydeep Biswas, Soha Pouya, and Yan Chang. ReMEmbR: Building and reasoning over long-horizon spatio-temporal memory for robot navigation. In 2025 IEEE International Conference on Robotics and Automation (ICRA), pages 2838–2845, 2025. doi: 10.1109/ICRA55743.2025.11127706. 1, 4, 8, 12

Aaron Grattafiori et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024. 5

Rohan Gupta, Trevor Asbery, Zain Merchant, Abrar Anwar, and Jesse Thomason. RobotFleet: An opensource framework for centralized multi-robot task planning. arXiv preprint arXiv:2510.10379, 2025. doi: 10.48550/arXiv.2510.10379. URL https://arxiv.org/abs/2510.10379. 2, 18

Yiping Kang, Johann Hauswald, Cao Gao, Austin Rovinski, Trevor Mudge, Jason Mars, and Lingjia Tang. Neurosurgeon: Collaborative intelligence between the cloud and mobile edge. In Proceedings of the 22nd International Conference on Architectural Supportfor Programming Languages and Operating Systems (ASPLOS), pages 615–629, 2017. doi: 10.1145/3037697.3037698. 8

Vladimir Karpukhin, Barlas Oğuz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. Dense passage retrieval for open-domain question answering. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 6769–6781, 2020. doi: 10.18653/v1/2020.emnlp-main.550. 4

Sean Lee, Aamir Shakir, Darius Koenig, and Julius Lipp. Open source strikes bread: New flufy embeddings model. Mixedbread blog, 2024. URL https://www.mixedbread.ai/blog/mxbai-embed-large-v1. 4, 5

Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. Retrieval-augmented generation for knowledge-intensive NLP tasks. In Advances in Neural Information Processing Systems (NeurIPS), volume 33, pages 9459–9474, 2020. 4

Ji Lin, Hongxu Yin, Wei Ping, Pavlo Molchanov, Mohammad Shoeybi, and Song Han. VILA: On pre-training for visual language models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 26689–26699, 2024. 4, 5

Jefrey Morgan. models: add nemotronh architecture support (#14356). Ollama commit 0ade920, https: //github.com/ollama/ollama/commit/0ade9205cce88006245dc54ea8884607822b103b, 2026. 16

Rodrigo Nogueira and Kyunghyun Cho. Passage re-ranking with BERT. arXiv preprint arXiv:1901.04085, 2019. 4

NVIDIA. Cosmos-Embed1: A joint video-text embedder for physical AI. Hugging Face model card, nvidia/Cosmos-Embed1-448p, 2025. URL https://huggingface.co/nvidia/Cosmos-Embed1-448p. 2, 5

NVIDIA. Cosmos 3: Omnimodal world models for physical AI. arXiv preprint arXiv:2606.02800, 2026a. 2, 5, 18, 29

NVIDIA. NVIDIA Nemotron 3 Nano 4B. Hugging Face model card, nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16, 2026b. URL https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16. 2, 5

NVIDIA, Aaron Blakeman, Aaron Grattafiori, et al. NVIDIA Nemotron 3: Eficient and open intelligence. arXiv preprint arXiv:2512.20856, 2025. 2, 5

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In Proceedings of the 38th International Conference on Machine Learning (ICML), volume 139 of Proceedings of Machine Learning Research, pages 8748–8763, 2021. 10

RAPIDS Team. cuVS: Vector search and clustering on the GPU. GitHub repository, rapidsai/cuvs, 2024. URL https://github.com/rapidsai/cuvs. 2, 8, 17, 22, 25

Aditi Singh, Abul Ehtesham, Saket Kumar, and Tala Talaei Khoei. Agentic retrieval-augmented generation: A survey on agentic RAG. arXiv preprint arXiv:2501.09136, 2025. 4

Team Wan. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025a. 2, 7

Team Wan. Wan2.2. GitHub repository, Wan-Video/Wan2.2, 2025b. URL https://github.com/Wan-Video/ Wan2.2. 2, 7

Michael Tschannen, Alexey Gritsenko, Xiao Wang, et al. Siglip 2: Multilingual vision-language encoders with improved semantic understanding, localization, and dense features, 2025. URL https://arxiv.org/abs/ 2502.14786. 18, 19

Bo Wan, Michael Tschannen, Yongqin Xian, Filip Pavetic, Ibrahim Alabdulmohsin, Xiao Wang, André Susano Pinto, Andreas Steiner, Lucas Beyer, and Xiaohua Zhai. LocCa: Visual pretraining with location-aware captioners. In Advances in Neural Information Processing Systems (NeurIPS), 2024. 18

Jianguo Wang, Xiaomeng Yi, Rentong Guo, Hai Jin, Peng Xu, Shengjun Li, Xiangyu Wang, Xiangzhou Guo, Chengming Li, Xiaohai Xu, Kun Yu, Yuxing Yuan, Yinghao Zou, Jiquan Long, Yudong Cai, Zhenxiang Li, Zhifeng Zhang, Yihua Mo, Jun Gu, Ruiyi Jiang, Yi Wei, and Charles Xie. Milvus: A purpose-built vector data management system. In Proceedings of the 2021 International Conference on Management of Data (SIGMOD), pages 2614–2627, 2021. doi: 10.1145/3448016.3457550. 4, 8, 17, 22

Zhou Wang, Alan C. Bovik, Hamid R. Sheikh, and Eero P. Simoncelli. Image quality assessment: From error visibility to structural similarity. IEEE Transactions on Image Processing, 13(4):600–612, 2004. doi: 10.1109/TIP.2003.819861. 14

Shunyu Yao, Jefrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. ReAct: Synergizing reasoning and acting in language models. In International Conference on Learning Representations (ICLR), 2023. 4

Arthur Zhang, Chaitanya Eranki, Christina Zhang, Ji-Hwan Park, Raymond Hong, Pranav Kalyani, Lochana Kalyanaraman, Arsh Gamare, Arnav Bagad, Maria Esteva, and Joydeep Biswas. Towards robust robot 3D perception in urban environments: The UT campus object dataset. IEEE Transactions on Robotics, 40: 3322–3340, 2024. 4

Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, and Oliver Wang. The unreasonable efectiveness of deep features as a perceptual metric. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 586–595, 2018. 14

## Appendix

Appendix A describes the memory record and the agent. Appendices B–E give the protocols and full results behind Sections 6.1–6.4, Appendix F lists the prompts, and Appendix G outlines training on the fleet cache (Section 7).

## A. System Reference

## A.1. Memory Record

A memory covers a three-second clip sampled at 2 FPS. NaVQA uses contiguous clips, and the fleet model of Appendix D.3 assumes one clip every 12 s. Tab. 4 lists the stored representations. The Wan2.2 VAE runs in FP32, and its latent is cast to BF16 for storage, halving it from 599 kB to 300 kB.
<table><tr><td>Use</td><td>Encoder</td><td>Input  $( T \times H \times W )$ </td><td>Stored tensor</td><td>Size (kB)</td></tr><tr><td>Retrieval</td><td>Cosmos-Embed1</td><td> $6 \times 4 4 8 \times 4 4 8$ </td><td>768, FP32</td><td>3.1</td></tr><tr><td>Description</td><td>Cosmos3-Edge vision</td><td> $1 \times 1 9 2 \times 3 2 0$ </td><td> $6 0 \times 2 0 4 8 , \mathrm { B F } 1 6$ </td><td>246.6</td></tr><tr><td>Inspection</td><td>Wan2.2 VAE</td><td> $5 \times 4 8 0 \times 8 3 2$ </td><td> $4 8 \times 2 \times 3 0 \times 5 2 , \mathrm { B F } 1 6$ </td><td>299.5</td></tr></table>

Table 4: Memory record for one three-second clip. Each memory retains 546.1 kB of objects and uploads 549.2 kB with the search vector; its index row is 3.2 kB. The vision-token size includes the token-grid and prompt-template metadata the decoder reads.

Storage and index. Each memory is one Milvus row holding the search vector, the position $( x , y , z )$ , the heading, the capture time and the trafic verdict. Vision tokens and latents sit in an S3-compatible object store (SeaweedFS) under a per-memory prefix that the row references. All search is exact: vectors use the embedder’s similarity metric, and position and time use Euclidean distance. The accuracy runs use CPU search, and Appendix D.2 compares it with the cuVS (RAPIDS Team, 2024) GPU\_BRUTE\_FORCE index in Milvus (Wang et al., 2021).

## A.2. Agent, Tools and Services

Agent loop. Nemotron-3-Nano-4B may call tools for up to three rounds, with several calls per round, before a final step writes a structured JSON answer with text, yes/no, position, time and duration fields. Each search returns the � nearest memories (� = 5 in the NaVQA runs). An answer that fails to parse is regenerated, up to three attempts in total. In deployment, NVIDIA NIM serves the agent in FP8 with a 16k-token context and a 4,096-token output limit, and removes the reasoning trace before tool calls are parsed. The describe tool decodes greedily from the cached tokens with at most 512 new tokens. Tool signatures and the answer fields are listed in Appendix F.3.

Services. Each model service holds one loaded instance and returns HTTP 503 when the model stays busy for more than one second or runs out of GPU memory, rather than queueing without bound.

Live semantic map. Zones are directed floor-plan polygons at 0.05 m per pixel. Every 10 s, the map asks the agent, once for each robot it has seen, whether there is trafic at that robot’s position and colors the zone as trafic or clear. The agent runs with reasoning enabled.

## B. NaVQA Protocol and Full Results

Protocol. The ablation covers all seven CODa sequences (0, 3, 4, 6, 16, 21 and 22) with three trials per configuration; the yes/no column counts 67 binary questions per trial, 201 responses per configuration. The indicated captioner writes the descriptions; Cosmos-Embed1 indexes the video, whereas MxBai indexes the caption text. Agents run with reasoning disabled, and malformed yes/no responses count as incorrect.

All configurations. Tab. 5 reports every combination of agent, captioner and index. Across matched swaps, which change one component and hold the other two fixed, the video index never lowers recall (+0.2 to +8.5 points). The Nemotron agent lowers first-attempt failures in all four settings (by 4.5 to 10.5 points) and unparsable answers by 2.4 to 4.4 points, at a cost of 2.3 to 5.9 points of recall. The captioner swap has no consistent direction.
<table><tr><td>Agent</td><td>Captioner</td><td>Index</td><td>First-try fail. (%)↓</td><td>Unparsable Recall (%) ↓</td><td>(%) ↑</td><td>Position err. (m) ↓</td><td>Time err. (min) ↓</td><td>Yes/no ok↑</td><td>System</td></tr><tr><td>Llama</td><td>VILA</td><td>MxBai</td><td>20.7</td><td>4.5</td><td>22.9</td><td>69.1</td><td>1.64</td><td>87/201</td><td>Original</td></tr><tr><td>Llama</td><td>VILA</td><td>Cosmos-Embed1</td><td>23.5</td><td>5.5</td><td>29.7</td><td>79.4</td><td>2.17</td><td>94/201</td><td></td></tr><tr><td>Llama</td><td>Cosmos3-Edge</td><td>MxBai</td><td>22.4</td><td>3.6</td><td>26.7</td><td>74.6</td><td>3.38</td><td>109/201</td><td></td></tr><tr><td>Llama</td><td>Cosmos3-Edge</td><td>Cosmos-Embed1</td><td>22.0</td><td>4.3</td><td>26.9</td><td>76.2</td><td>2.19</td><td>106/201</td><td></td></tr><tr><td>Nemotron</td><td>VILA</td><td>MxBai</td><td>16.2</td><td>0.9</td><td>18.7</td><td>71.0</td><td>2.21</td><td>81/201</td><td></td></tr><tr><td>Nemotron</td><td>VILA</td><td>Cosmos-Embed1</td><td>13.0</td><td>1.1</td><td>27.2</td><td>67.5</td><td>1.87</td><td>74/201</td><td></td></tr><tr><td>Nemotron</td><td>Cosmos3-Edge</td><td>MxBai</td><td>12.2</td><td>0.9</td><td>20.8</td><td>66.9</td><td>1.66</td><td>81/201</td><td></td></tr><tr><td>Nemotron</td><td>Cosmos3-Edge</td><td>Cosmos-Embed1</td><td>13.8</td><td>1.9</td><td>24.6</td><td>66.6</td><td>2.37</td><td>81/201</td><td>Upgraded</td></tr></table>

Table 5: Full seven-sequence NaVQA component ablation: three trials, 201 yes/no responses per configuration. Agents are Llama-3.1-8B and Nemotron-3-Nano-4B; VILA is VILA-1.5-13B. “First-try fail.” is the first-attempt parse-failure rate. Position and time errors are medians over parsed answers. “Original” is original ReMEmbR and “Upgraded” is Upgraded ReMEmbR.

## C. Describe-on-Demand and Reconstruction

Describe-on-demand run. In a separate 150-frame run, each frame is encoded once by the Cosmos3-Edge vision encoder, and its cached tokens are decoded with three prompts: the detection prompt of Appendix F.1 (at most 256 new tokens), a general scene-description request (512) and a one-word yes/no question (16). Decoding is greedy with reasoning disabled, and each prompt gets a single attempt. The same frame and prompt are also run through the unsplit model. Two outputs match when the decoded strings are equal after trimming surrounding whitespace; all 450 frame–prompt pairs match. Times are wall-clock after one warm-up and include tokenization and image preprocessing.

Reconstruction run. Each of the 150 trafic frames is repeated five times into a $5 \times 4 8 0 \times 8 3 2$ input, encoded, and reconstructed from its stored BF16 latent; reconstruction time is a median over frames.

Latent precision. Decoding the BF16 latent in place of the FP32 one changes the reconstruction metrics of Section 6.2 by less than 0.1% of their value (Tab. 6).
<table><tr><td>Metric</td><td>BF16 latent</td><td>Change vs. FP32</td><td>Relative change</td></tr><tr><td>PSNR (dB) ↑</td><td>39.7</td><td>-0.003</td><td>0.008%</td></tr><tr><td>SSIM↑</td><td>0.974</td><td> $- 1 . 3 \times 1 0 ^ { - 5 }$ </td><td>0.001%</td></tr><tr><td>LPIPS ↓</td><td>0.0052</td><td> $+ 4 \times 1 0 ^ { - 6 }$ </td><td>0.08%</td></tr></table>

Table 6: Reconstruction fidelity of the stored BF16 latent over 150 trafic frames, each repeated five times, and its change from decoding the FP32 latent.

## D. Robot, Server and Fleet Costs

## D.1. Jetson AGX Xavier Engine Profiling

Tab. 7 gives the encoder latencies behind Section 6.3.

<table><tr><td>Encoder</td><td>Input  $( T \times H \times W )$ </td><td>Mean latency (s)</td><td>P95 latency (s)</td><td>Runs</td></tr><tr><td>Cosmos-Embed1</td><td> $6 \times 4 4 8 \times 4 4 8$ </td><td>11.76</td><td>11.78</td><td>10</td></tr><tr><td>Cosmos3-Edge vision</td><td> $1 \times 1 9 2 \times 3 2 0$ </td><td>0.196</td><td>0.254</td><td>16</td></tr><tr><td>Wan2.2 VAE (FP32)</td><td> $5 \times 4 8 0 \times 8 3 2$ </td><td>6.46</td><td>6.51</td><td>10</td></tr></table>

Table 7: Mean and P95 latency of warmed TensorRT 8.5.2 engines on a Jetson AGX Xavier with random inputs, including host–device transfers. Cosmos-Embed1 uses custom TensorRT plugins.

## D.2. Server Memory, Serving and Search

GPU memory. Tab. 8 lists the GPU memory of each process in Fig. 10.

<table><tr><td>Process GPU memory (GiB)</td></tr><tr><td>Robot (iw.encode)</td></tr><tr><td>Cosmos-Embed1 2.96</td></tr><tr><td>Cosmos3-Edge vision encoder 1.61 Wan2.2 VAE encoder 1.98</td></tr><tr><td>All three loaded 6.55</td></tr><tr><td>Server</td></tr><tr><td>Cosmos3-Edge language decoder 5.4</td></tr><tr><td>Wan2.2 VAE decoder 10.3</td></tr><tr><td>Nemotron-3-Nano-4B (NIM, FP8) 5.9</td></tr><tr><td>Original ReMEmbR</td></tr><tr><td>VILA-1.5-13B captioner 26.3</td></tr></table>

Table 8: GPU memory by process. Encoder values are resident memory with the services loaded; server and captioner values are peaks during inference; the NIM value excludes the key-value cache that NIM reserves.

Agent serving. Fig. 13 compares NIM (FP8) and Ollama (Q8\_0) serving Nemotron-3-Nano-4B on synthetic 8k-context requests at concurrency 1 to 8. A separate sweep (Tab. 9) uses a larger NIM memory allocation and extends concurrency to 256. Throughput peaks at 2,844 output tokens/s at 128 concurrent requests; beyond that, time to first token rises sharply. All requests completed without errors.

![](images/8240283f2f41ade33f82e1024d15c094b6140cbbab6d54d3edd3c0bcb7fbf0cd.jpg)  
Figure 13: NIM (FP8) scales to 1217 tokens/s at 8 concurrent requests, while Ollama (Q8\_0) stays near 235. Throughput is aggregate output for Nemotron-3-Nano-4B at 8k context on one RTX PRO 6000 Blackwell Server Edition.

<table><tr><td>Concurrency</td><td>Output tok/s</td><td>Mean TPOT (ms)</td><td>Mean TTFT (s)</td><td>P99 TTFT (s)</td></tr><tr><td>1</td><td>155.8</td><td>5.96</td><td>0.097</td><td>0.120</td></tr><tr><td>2</td><td>297.3</td><td>6.11</td><td>0.149</td><td>0.167</td></tr><tr><td>4</td><td>570.2</td><td>6.42</td><td>0.272</td><td>0.307</td></tr><tr><td>8</td><td>953.3</td><td>7.21</td><td>0.447</td><td>0.541</td></tr><tr><td>16</td><td>1574.7</td><td>8.71</td><td>0.715</td><td>1.032</td></tr><tr><td>32</td><td>2026.9</td><td>12.52</td><td>1.002</td><td>1.674</td></tr><tr><td>48</td><td>2357.4</td><td>16.59</td><td>1.131</td><td>1.735</td></tr><tr><td>64</td><td>2670.4</td><td>20.62</td><td>1.079</td><td>1.412</td></tr><tr><td>96</td><td>2726.6</td><td>29.19</td><td>1.085</td><td>1.314</td></tr><tr><td>128</td><td>2844.1</td><td>37.78</td><td>1.139</td><td>1.220</td></tr><tr><td>192</td><td>2252.1</td><td>57.59</td><td>1.115</td><td>4.660</td></tr><tr><td>256</td><td>1909.9</td><td>64.91</td><td>14.173</td><td>28.339</td></tr></table>

Table 9: NIM FP8 synthetic-request sweep. TTFT is time to first token; TPOT is time per output token.

Live-map capacity. In the load test, each emulated robot requests one map update every 10 s. Each update makes two to three agent calls, with input and output lengths similar to those of the live map. A setup serves a fleet when 95% of its updates finish within 20 s (Tab. 10). In the live map itself, 17 robots on one GPU see a mean update time of 4.5 s.

<table><tr><td>Robots</td><td>1 GPU</td><td>2 GPUs</td></tr><tr><td>8</td><td>8.1</td><td>7.1</td></tr><tr><td>16</td><td>11.2</td><td>8.2</td></tr><tr><td>20</td><td>13.3</td><td>8.9</td></tr><tr><td>24</td><td>17.0</td><td>9.5</td></tr><tr><td>28</td><td>23.6</td><td>11.4</td></tr><tr><td>32</td><td>37.2</td><td>11.6</td></tr><tr><td>40</td><td>76.7</td><td>13.8</td></tr><tr><td>48</td><td>116.2</td><td>17.6</td></tr><tr><td>Robots served</td><td>24</td><td>48</td></tr></table>

Table 10: P95 map-update time (s) for robots polling every 10 s, with one NIM instance per GPU. A setup serves a fleet when 95% of updates finish within 20 s; 48 robots is the largest fleet tested.

Vector search. At one million rows, a single query takes 10.6 ms with the cuVS GPU index and 22.8 ms on the CPU (Fig. 14).

![](images/02fc259924d3ba6d28c91a90af8843787d7a782f4622d32f787e3d39eaaa759b.jpg)  
Figure 14: GPU exact search’s throughput advantage over CPU exact search grows across the tested index sizes, from parity at $1 0 ^ { 4 }$ rows to 3.3× at $1 0 ^ { 5 }$ and $1 9 \times \mathrm { a t ~ 1 0 ^ { 6 } }$ . Results are for 32 concurrent queries in one Milvus instance over synthetic 768-dimensional vectors; the GPU index uses NVIDIA cuVS (RAPIDS Team, 2024). Both searches return identical neighbours. The axis is logarithmic, so the length of each green bar is the relative speedup.

## D.3. Fleet Model

Notation. Let � be the number of robots and $r = 1 / 1 2 s ^ { - 1 }$ the write rate of one robot, one memory every 12 s. An operating day is $h _ { \mathrm { o p } } = 2 8 , 8 0 0 \ s \ ( 8 \mathrm { h } )$ , so a robot writes $2 , 4 0 0$ memories per day. Service times are $t _ { \mathrm { c a p } } = 4 . 3 5$ s for original captioning and $t _ { \mathrm { d e s c } } = 1 . 2 5 \ : \{ $ s for describing a memory from cached tokens, both means, and $t _ { \mathrm { i n s p } }$ for scene inspection, kept symbolic. Per new memory, $p$ is the mean number of descriptions and � the mean number of inspections; $p = 0 . 1$ and $p = 0 . 0 1$ mean one description for 10% and 1% of memories. The live map polls every $\tau \ : s \ : ( 1 0 \ : s$ by default), and $u _ { a }$ is the rate of map updates one GPU completes while answering 95% of them within two polling periods, 20 s. One GPU serves at least 24 robots within this limit (Tab. 10), so $u _ { a } \approx 2 . 4$ updates/s. Unlike the serialized description terms, $u _ { a }$ is a batched serving rate. A memory keeps $b = 5 4 6 . 1$ kB of objects and a 3.2 kB index row, and � is the retention in days. A GPU equivalent is the serialized service time ofered per second, so one equivalent is one fully busy GPU.

Per-robot demand. Tab. 11 gives the demand of one robot; all quantities scale linearly with �.
<table><tr><td>Event</td><td>Input</td><td>Rate</td><td>Per robot</td></tr><tr><td>Caption a memory (original)</td><td> $t _ { \mathrm { c a p } } = 4 . 3 5 s$ </td><td>r</td><td> $0 . 3 6 2 5 \ : { \mathrm { G P U } } { \mathrm { e q } } .$ </td></tr><tr><td>Describe a memory</td><td> $t _ { \mathrm { d e s c } } = 1 . 2 5 s$ </td><td> $p r$ </td><td> $0 . 1 0 4 p \mathrm { G P U } \mathsf { e q } .$ </td></tr><tr><td>Inspect a scene</td><td> $t _ { \mathrm { i n s p } }$ </td><td> $q r$ </td><td> $q t _ { \mathrm { i n s p } } / \mathrm { 1 2 G P U e q . }$ </td></tr><tr><td>Update the map</td><td>ua ≈ 2.4 updates/s/GPU</td><td> $1 / \tau$ </td><td> $0 . 0 4 2 \ : \mathrm { G P U } \ : \mathrm { e q . }$ </td></tr><tr><td>Retain objects</td><td> $b = 5 4 6 . 1 \mathrm { k B }$ </td><td>r</td><td>1.31 D GB</td></tr><tr><td>Index rows</td><td>3.2 kB</td><td>r</td><td>7.68 D MB</td></tr></table>

Table 11: Per-robot demand and storage. Sources are Tab. 3, Tab. 4 and Tab. 10.  
Cost model. Eq. 2 gives the monthly server GPU and object-storage cost of one robot for a retention of � days,

$$
C _ { \mathrm { r o b o t } } = C _ { g } \left[ \frac { 1 } { \tau u _ { a } } + r \left( p t _ { \mathrm { d e s c } } + q t _ { \mathrm { i n s p } } \right) \right] + C _ { s } \frac { r h _ { \mathrm { o p } } D b } { 1 0 ^ { 6 } } ,\tag{2}
$$

where $C _ { g }$ is the monthly cost of one reserved GPU, $C _ { s }$ the storage cost per GB-month and � is in kB. Original write-time captioning alone costs $0 . 3 6 2 5 C _ { g }$ per robot. $\mathtt { A t } \tau = 1 0$ s and $p = 0 . 1 $ , the map and description terms together cost $0 . 0 5 2 C _ { g }$ per robot, so even with the live map, which original ReMEmbR lacks, server generation demand stays below that of captioning alone.

Break-even and the 34.8× factor. Describing from cached tokens ofers less generation demand than writetime captioning whenever $p t _ { \mathrm { d e s c } } < t _ { \mathrm { c a p } } ,$ that is, below $p < t _ { \mathrm { c a p } } / t _ { \mathrm { d e s c } } = 4 . 3 5 / 1 . 2 5 = 3 .$ 48 descriptions per memory. $\mathrm { A t } \ p = 0 . 1$ and $p = 0 . 0 1$ the demand falls by $t _ { \mathrm { c a p } } / ( p t _ { \mathrm { d e s c } } ) = 3 4 . 8$ and 348 times at any fleet size. These factors combine two changes: replacing VILA captioning with Cosmos3-Edge decoding, and deferring description to the memories that are asked about.

Fleet projection and retention. Tab. 12 projects upload and retention for one robot and a 100-robot fleet.   
One robot uploads 549 kB every 12 s, an average of 0.37 Mbit/s while writing.

<table><tr><td>Scope</td><td>Period</td><td>Memories</td><td>Uploaded</td><td>Objects</td><td>Index</td></tr><tr><td>1 robot</td><td>1 day</td><td>2,400</td><td>1.32 GB</td><td>1.31 GB</td><td>7.68 MB</td></tr><tr><td>100 robots</td><td>1 day</td><td>240,000</td><td>131.81 GB</td><td>131.07 GB</td><td>0.768 GB</td></tr><tr><td>100 robots</td><td>30 days</td><td>7.2 million</td><td>3.95 TB</td><td>3.93 TB</td><td>23.04 GB</td></tr><tr><td>100 robots</td><td>365 days</td><td>87.6 million</td><td>48.11 TB</td><td>47.84 TB</td><td>280.32 GB</td></tr></table>

Table 12: Totals over each period at one memory every 12 s over an 8-hour day.

## E. Trafic Evaluation

Dataset. The evaluation set is 914 consecutive RGB frames of one warehouse simulation run from a single ego robot. Of the 607 trafic frames, 289 are low severity and 318 are high; 307 frames are clear.

Embedding verdict. Each class has an ensemble of short text prompts (Appendix F.1). Their Cosmos-Embed1 text embeddings are mean-pooled and renormalized into one vector per class, and a clip is labeled by whichever class vector its video embedding is closer to. The verdict is written to memory as one sentence.

Caption arms. The Cosmos3-Edge detection and description arms run at temperature 0 with reasoning enabled and 2,048 output tokens each. Detection returns one JSON array of {label, bbox\_2d} objects in normalized 0 to 1000 coordinates. A response that does not parse, or has the wrong keys, a generic label or an out-of-range or misordered box, is re-asked with the shorter correction prompt, for up to three attempts in total, and the frame is dropped if all three fail. Severity follows from the number of validated boxes: none is clear, one is low and two or more is high. Description asks for one sentence of at most 40 words. A description whose reasoning was cut of, or that exceeds the length limit, is dropped. The prompts are in Appendix F.1.

Full results. Tab. 13 gives coverage, accuracy, balanced accuracy and per-class recall, split by severity, for every representation in Fig. 12. Detection has the highest recall at both trafic severities, while the embedding verdict retains more balanced trafic and clear recall.
<table><tr><td rowspan="2">Memory per frame</td><td rowspan="2">Coverage</td><td rowspan="2">Accuracy</td><td rowspan="2">Balanced acc.</td><td colspan="3">Traffic recall</td><td rowspan="2">Clear recall</td></tr><tr><td>All</td><td>Low</td><td>High</td></tr><tr><td>Embedding verdict</td><td>914/914</td><td>72.1</td><td>71.3</td><td>73.8</td><td>68.5</td><td>78.6</td><td>68.7</td></tr><tr><td>Detection + reasoning</td><td>844/914</td><td>72.7</td><td>65.3</td><td>88.4</td><td>87.0</td><td>89.6</td><td>42.3</td></tr><tr><td>Description + reasoning</td><td>902/914</td><td>48.7</td><td>59.1</td><td>27.1</td><td>25.1</td><td>28.9</td><td>91.1</td></tr><tr><td>Always “traffic&quot;</td><td>914/914</td><td>66.4</td><td>50.0</td><td>100.0</td><td>100.0</td><td>100.0</td><td>0.0</td></tr></table>

Table 13: Simulated warehouse trafic over 914 frames (%). The embedding verdict is read from the Cosmos-Embed1 vector; the detection and description arms use Cosmos3-Edge with reasoning enabled. Coverage counts frames with usable context; the other columns exclude dropped frames. Bold marks the best of the three representations in each column. Rows are ordered by balanced accuracy.

## F. Prompt Catalog

Curly-brace placeholders are filled per request.

## F.1. Trafic Evaluation

Embedding verdict: trafic prototype prompts (Cosmos-Embed1 text encoder)

a warehouse aisle blocked by a forklift   
a warehouse aisle blocked by a pallet   
a warehouse aisle blocked by stacked boxes   
a person walking in a warehouse aisle   
a mobile robot blocking a warehouse aisle   
a warehouse aisle blocked by a pallet jack

Embedding verdict: clear prototype prompts (Cosmos-Embed1 text encoder)

an empty warehouse aisle   
a clear, open warehouse aisle   
a warehouse aisle with an unobstructed path

Embedding verdict: sentence written to memory

In the aisle straight ahead there is traffic.   
In the aisle straight ahead there is no traffic.

Detection arm: detection prompt (Cosmos3-Edge)

Return only one JSON array of objects. Each object must have exactly {"label": string, "bbox\_2d": [x\_min,   
y\_min, x\_max, y\_max]} using normalized 0-1000 coordinates. Include distinct whole vehicles or agents   
physically occupying or narrowing the navigable path directly ahead; use one box per object, never object   
parts or duplicate boxes. Label each object by its category, such as forklift, vehicle, or person; never   
use a generic label such as obstacle. Exclude the ego vehicle, railings, floor markings, fixtures, fixed   
infrastructure, and objects beside the path. Do not include markdown or any text outside the JSON array.

Detection arm: correction prompt for the second and third attempts (Cosmos3-Edge)

Return only one JSON array. Each object must have exactly {"label": string, "bbox\_2d": [x\_min, y\_min,   
x\_max, y\_max]} with normalized 0-1000 coordinates. Include only distinct whole vehicles or agents that   
block or narrow the path directly ahead; exclude object parts, duplicate boxes, railings, floor markings,   
fixtures, fixed infrastructure, the ego vehicle, and beside-path objects. Label each detection by its   
object category, such as forklift, vehicle, or person, never obstacle. Do not include markdown or   
explanatory text.

Description arm: description prompt (Cosmos3-Edge)

This is a single still image for robot navigation. In one concise sentence of at most 40 words, describe   
the relevant objects, obstacle layout, free space, and navigable paths. Do not infer motion, speed, or   
events over time.

Agent question, one per frame (Nemotron-3-Nano-4B)

It is {HH:MM:SS} and I am a robot at position {[x, y, z]} in the warehouse. What is the traffic information   
in this area? Specifically, is there any other vehicle or agent in the zone directly ahead of me right now?   
Answer with only one word yes or no.

## F.2. Live Semantic Map

Map query, one per robot per poll (Nemotron-3-Nano-4B)

The time now is {HH:MM:SS}. You are a robot at position {[x, y, z]} in a warehouse. We are looking to know   
the traffic situation in the position where you are in right now. Is there any other vehicle or agent in   
the zone directly ahead of you right now? Answer with only one word yes or no.

Operator description request on robot selection (Cosmos3-Edge language decoder)

Provide two sentences on the traffic status of the aisle in front of you. Mention other vehicles or agents   
in the scene.

## F.3. Agent Interfaces

Search tools

```python
retrieve_from_text(x: string)
retrieve_from_position(x: (x, y, z))
retrieve_from_time(x: "HH:MM:SS")
```

Describe tool

```erlang
describe(memory_id, question) -> text
```

Final structured answer

fields: text, binary (yes or no), position, time, duration   
position is null or three numbers

## G. Training on Cached Representations

Fleet operation already performs the preprocessing that Cosmos3 training needs. Each memory stores the Cosmos3-Edge vision tokens $\mathbf { H } = f _ { \phi } ( \mathbf { x } )$ and the Wan2.2 video latents ${ \bf z } _ { 0 } = \mathrm { V A E } ( { \bf x } )$ (NVIDIA, 2026a), so training reads both from the object store without moving or re-encoding video (Fig. 15). Cosmos-Embed1 queries select relevant events and diverse scenes for the training set. Over � epochs on � memories, the cache saves �� forward passes of each encoder, and training compute goes only to the components being adapted: the language decoder, difusion components or action heads.

Conditions.

• Frozen cache boundary. The learnable parameters of everything upstream of a cached tensor remain frozen through finetuning.

• Pinned encoders. Each artifact records its encoder version, preprocessing and tensor layout, and the consumer must match the Cosmos3 variant and vision interface that produced it.

• Shared latent space. The world model trains in the Wan2.2 VAE latent space. Another tokenizer needs re-encoding or a learned latent adapter.

• Targets. Recorded latents are self-supervised targets for the world model. Language training uses operator-validated answers attached to selected memories, and action adaptation needs synchronized, embodiment-compatible action targets.

Objectives. The language decoder (parameters $\omega ,$ or a LoRA adapter) learns to write a validated answer y to a prompt q with H in the video slot. The world model $v _ { \psi }$ is trained by flow matching on a clean latent $\mathbf { z } _ { 0 } ,$ , with noise $\mathbf { \epsilon } \epsilon \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$ , time � ∼ Uniform[0, 1] and conditioning c (template text, pose, time and preceding latents):

$$
\mathcal { L } _ { \mathrm { A R } } ( \omega ) = - \mathbb { E } _ { ( \mathbf { H } , \mathbf { q } , \mathbf { y } ) } \sum _ { j } \log p _ { \omega } ( y _ { j } \mid y _ { < j } , \mathbf { q } , \mathbf { H } ) ,\tag{3}
$$

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { D M } } ( \psi ) = \mathbb { E } _ { ( \mathbf { z } _ { 0 } , \mathbf { c } ) , t , \epsilon } \left\| v _ { \psi } ( \mathbf { z } _ { t } , t , \mathbf { c } ) - ( \epsilon - \mathbf { z } _ { 0 } ) \right\| _ { 2 } ^ { 2 } , \qquad \mathbf { z } _ { t } = ( 1 - t ) \mathbf { z } _ { 0 } + t \epsilon . } \end{array}\tag{4}
$$

![](images/6a31b4acf9b034a84f97d664f07ec518efefc43f488a286e74f76d31d61af8be.jpg)  
Figure 15: Fleet memory supplies Cosmos3 fine-tuning without rerunning the encoders. Cached vision tokens H feed the vision-language reasoner, and cached Wan2.2 latents z<sub>0</sub>, noised to level �, feed the generator. Semantic filtering selects high-priority memories, and validated text labels and aligned action targets supply supervision. With everything upstream of the cache boundary frozen, the language, difusion or action components can be adapted (Appendix G).