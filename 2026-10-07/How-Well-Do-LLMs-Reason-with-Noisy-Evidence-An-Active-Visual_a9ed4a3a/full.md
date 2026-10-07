# How Well Do LLMs Reason with Noisy Evidence? An Active Visual Reasoning Benchmark

Bach Nguyen Zhaonan Li Mau Son Nguyen Sanika Chavan Nilay Kumar Hong Anh Nguyen Khoa Vo Ben Zhou Arizona State University

## Abstract

Real-world reasoning rarely reduces to static question answering: agents must actively gather information from tools and sensors that are often noisy and unreliable. Yet most existing active reasoning benchmarks assume that environmental feedback is trustworthy, or introduce noise without exposing an explicit, calibrated uncertainty signal, leaving open how LLMs should reason when the evidence itself is uncertain. We introduce VISUALNOISEQA<sup>1</sup>, a novel benchmark for active reasoning under noisy visual feedback. A text-only LLM must solve VQA problems by iteratively querying a fixed, off-the-shelf VLM treated as a stochastic visual sensor. For each query, we draw multiple samples and expose an empirical uncertainty signal via self-consistency, enabling the reasoner to probe from different angles and decide what to ask next and when to stop. Our construction is automatic and scalable: starting from diverse VQA sources and two noisy VLMs, we retain only questions where the sensor is inconsistent yet human-solvable. We evaluate multiple LLM reasoners on 1,000 instances spanning perception, chart understanding, and knowledge-intensive reasoning. VISU-ALNOISEQA thus provides a controlled playground to study how different LLMs exploit uncertainty signals for robust reasoning.

## 1 Introduction

Humans typically solve problems by actively gathering information from tools and sensors that are noisy and sometimes wrong: we form expectations about the world, compare new observations against these beliefs, and cross-check conflicting evidence before committing to a decision. Recent work on LLM agents and active reasoning begins to move in this direction, letting models plan tool calls, ask follow-up questions, or navigate simulators instead of answering in a single shot (Shridhar et al., 2020; Nakano et al., 2021; Yao et al., 2022; Abdulhai et al., 2023; Li et al., 2024; Hu et al., 2024; Zhou et al., 2025). However, these settings usually assume that the environment behaves as an oracle, or they provide no explicit and reliable signal about the uncertainty of tool outputs, leaving open how LLMs should act when feedback itself is unreliable. People, in contrast, rarely trust a single observation at face value, especially when they are explicitly asked to double-check or when the stakes are high. For instance, when a navigation app suggests an exit that conflicts with road signs, we slow down and cross-check. A similar pattern arises in vision: consider the image in Figure 1 and the question “Where is the woman standing with respect to the man?” A visual-language model repeatedly answers that the woman is behind the man, yet also describes them as walking arm-in-arm. Without seeing the image, a human reasoner can detect the inconsistency, realize that walking arm-in-arm while one person is behind the other is unlikely, and infer that they are most plausibly standing next to each other, while a weaker reasoner may instead re-ask a confirmatory question and commit prematurely to the wrong answer (Figure 1). This illustrates that arriving at the correct conclusion depends not only on the confidence with which each observation is reported, but also on critically assessing the reliability of each piece of evidence—particularly when responses are contradictory. This form of reliability-aware, beliefdriven cross-validation is largely absent from current LLM benchmarks.

In this paper, we study how text-only LLMs perform active reasoning when their only access to an image is uncertain feedback from a fixed VLM sensor. Rather than pushing the frontier of visual recognition, we treat an off-the-shelf VLM as a black-box noisy sensor and use its self-consistency across samples as a cheap, observable, scalable, and externally verifiable uncertainty signal that the reasoner can act on directly. Building on prior “blind” VQA setups (Li et al., 2025), the LLM never sees the image: it must solve each instance by deciding what perception questions to ask, how to aggregate the answers, and when to trust or override them. For each sensor query we draw multiple VLM samples and expose a coarse consistency score, enabling the reasoner to probe from different angles and recover from misleading feedback. To study this systematically, we introduce VISUALNOISEQA, a benchmark and analysis suite for active reasoning under noisy visual feedback. Crucially, VISUALNOISEQA is designed to evaluate text-only LLM reasoners, not vision models, isolating the ability to reason under uncertainty from visual perception itself. VISUALNOISEQA is constructed automatically from diverse VQA sources and two modern VLMs with distinct failure modes, selecting instances where the sensors are self-inconsistent, yet “blind” human annotators can still solve the task.

![](images/064288970bfd97efd2afac3405a933bd73498cda919336e374a97e715c6d246a.jpg)  
Figure 1: Active reasoning under noisy visual feedback. From the 3/5 claim “the woman is behind the man,” a strong reasoner diversifies its query, detects the “arm-in-arm” contradiction, and defers commitment, while a weak reasoner asks a confirmatory question and commits to a highly consistent (4/5) but wrong answer. VISUALNOISEQA tests whether LLMs can probe, detect contradictions, and reason under unreliable evidence.

We use VISUALNOISEQA to evaluate a range of LLM reasoners and interaction strategies, finding that all models still lag behind humans, but strong reasoners improve over direct VLM predictions and leverage consistency signals to reduce miscalibration, especially on perception tasks. These results highlight both the promise and current limitations of uncertainty-aware reasoning with noisy sensors. Beyond accuracy, our analysis surfaces failure modes that separate strong from weak reasoners and provide targets for models that reason reliably under noisy evidence. While other benchmarks and datasets focus on gathering missing information and treating received information as essentially true, our main contribution is to evaluate how well LLMs can navigate through unreliable and untrustworthy signals, which represent realworld scenarios.

## 2 Related Work

Our work connects to modular visual reasoning frameworks that separate perception from reasoning, to LLM-based agents for active information seeking and tool use, and to RAG systems that use external sources for question answering.

Modular visual reasoning. Classical modular VQA approaches separate perception from reasoning by composing neural modules or executing programs over scene graphs and other structured visual abstractions, as in neural-symbolic VQA systems (Yi et al., 2018). More recent work such as ViperGPT and VisProg uses LLMs to synthesize code or programs that call visual tools while keeping a clean interface between text-based reasoning and perception (Surís et al., 2023; Gupta and Kembhavi, 2023). Closest to our setup, recent “blind” VQA frameworks (e.g., Li et al., 2025) let a text-only reasoner query a VLM sensor through restricted perception primitives; our benchmark follows this separation but focuses explicitly on noisy VLM feedback, shifting the challenge from information gathering to reasoning under unreliable evidence.

LLM agents and tool use. A growing line of work studies LLM agents that interact with external tools or environments, such as web browsers, simulators, or APIs. Early systems like WebGPT train models to browse and cite web resources for question answering (Nakano et al., 2021). ReAct interleaves chain-of-thought with actions, enabling LLMs to plan and execute multi-step tool calls in textual environments (Yao et al., 2023), while Toolformer learns to call APIs or modular experts when needed (Schick et al., 2023). More recent agent frameworks (e.g., Voyager and other longhorizon agents) extend this paradigm to embodied and open-ended environments with persistent state and skills (Wang et al., 2023a). In most of these settings, tools are treated as approximately reliable oracles (e.g., a browser or API returning accurate results), whereas in our benchmark the only tool for accessing the image is an explicitly noisy and stateless VLM sensor, which provides explicit consistency information associated with each query.

Retrieval-augmented generation (RAG). RAG systems augment LLMs with retrieved evidence that can itself be incomplete or conflicting (Lewis et al., 2020). Nevertheless, most RAG pipelines condition on retrieved passages as if they were reliable. VISUALNOISEQA can be seen as a perceptual analogue that makes this unreliability explicit: the reasoner queries a VLM for multiple, possibly inconsistent descriptions and must aggregate them using an exposed self-consistency signal, rather than trusting any single retrieval.

Active reasoning and information seeking under uncertainty. Active reasoning works study models that gather additional information before committing to an answer, often by interacting with tools or environments over multiple turns. LMRL Gym introduces benchmarks for multi-turn reinforcement learning with LLM-based agents, highlighting challenges of long-horizon control and partial observability (Abdulhai et al., 2023). In highstakes domains, MedIQ evaluates question-asking LLMs that propose targeted follow-up queries to reduce diagnostic uncertainty in clinical reasoning (Li et al., 2024). Zhou et al. move from passive to active reasoning and ask whether LLMs can pose the right questions under incomplete information, proposing benchmarks and methods for query selection when key evidence is missing (Zhou et al., 2025). VISUALNOISEQA instantiates a similar active reasoning loop in the visual domain: a text-only reasoner must decide which perception queries to issue to a noisy VLM sensor, interpret self-consistency as a coarse uncertainty signal, and determine when the accumulated evidence is sufficient to commit to an answer. Unlike standard self-reflection methods (Renze and Guven, 2024), which focus on an LLM revising its own internal textual logic, VISUALNOISEQA forces the agent to handle explicitly stochastic, external sensory feedback. To the best of our knowledge, we are among the first to explicitly model uncertainty in the environmental feedback itself within an active reasoning benchmark; moreover, our construction is fully automatic and inexpensive, and does not require hand-crafted text simulators or environment-specific infrastructure.

## 3 VISUALNOISEQA Construction

## 3.1 Problem Setup: Active Reasoning with a Noisy Visual Sensor

We formulate VQA as an active reasoning problem for a text-only LLM reasoner R interacting with a noisy visual sensor S. Each instance consists of an image x, an original VQA question $q ^ { ( 0 ) }$ , and a ground-truth answer y. The sensor S is stateless: every call $S ( x , q )$ depends only on $( x , q )$ . At interaction step $t ,$ the reasoner observes the interaction history

$$
\begin{array} { c } { { h ^ { ( t ) } = \big ( q ^ { ( 0 ) } , ( q ^ { ( 1 ) } , a ^ { ( 1 ) } , u ^ { ( 1 ) } ) , \dots , } } \\ { { \big ( q ^ { ( t - 1 ) } , a ^ { ( t - 1 ) } , u ^ { ( t - 1 ) } \big ) \big ) , } } \end{array}\tag{1}
$$

and chooses either to conclude with an answer or to issue a self-contained query $\boldsymbol { q } ^ { ( t ) }$ to the sensor. To model noisy perception and expose uncertainty, we sample the sensor K times with the same prompt:

$$
\{ a _ { 1 } ^ { ( t ) } , \ldots , a _ { K } ^ { ( t ) } \} \sim S ( x , q ^ { ( t ) } ) .\tag{2}
$$

We then (i) sample one reply $a ^ { ( t ) }$ uniformly from $\{ a _ { i } ^ { ( t ) } \} _ { i = 1 } ^ { K }$ as the realized sensory feedback, and (ii) estimate an uncertainty score by computing the empirical consistency of this answer:

$$
\boldsymbol { u } ^ { ( t ) } = \sum _ { i = 1 } ^ { K } \mathbf { 1 } \big [ \mathrm { s e m \_ e q } ( a _ { i } ^ { ( t ) } , a ^ { ( t ) } ) \big ] ,\tag{3}
$$

<table><tr><td></td><td></td><td colspan="6">InternVL3.5-8B</td><td colspan="6">Qwen2.5-VL-7B</td></tr><tr><td>Reasoner</td><td>Strategy</td><td>Perception Acc ↑ ÈCE↓</td><td></td><td>Charts</td><td></td><td>Knowledge Acc ↑ ECE ↓ Acc ↑ ECE ↓</td><td></td><td>Perception Acc ↑ ECE↓ Acc ↑ ECE↓</td><td></td><td></td><td>Charts</td><td>Acc ↑ ECE↓</td><td>Knowledge</td></tr><tr><td>Human</td><td>-</td><td>72.0</td><td>一</td><td>86.0</td><td>1</td><td>69.3</td><td></td><td>70.0</td><td></td><td>92.0</td><td>–</td><td>74.0</td><td></td></tr><tr><td rowspan="3">GPT-5 Mini</td><td>BP</td><td>49.0</td><td></td><td>32.0</td><td></td><td>60.7</td><td></td><td>42.0</td><td></td><td>29.0</td><td></td><td>61.3</td><td></td></tr><tr><td>UoT</td><td>46.0</td><td></td><td>36.0</td><td></td><td>50.0</td><td></td><td>39.0</td><td></td><td>30.0</td><td></td><td>58.0</td><td></td></tr><tr><td>ReAct</td><td>36.0</td><td></td><td>46.0</td><td></td><td>56.7</td><td></td><td>42.0</td><td></td><td>35.0</td><td></td><td>63.0</td><td></td></tr><tr><td rowspan="3">Qwen3.5-122B-A10B</td><td>BP</td><td>48.6</td><td>0.27</td><td>59.8</td><td>0.14</td><td>61.0</td><td>0.12</td><td>50.8</td><td>0.22</td><td>60.6</td><td>0.18</td><td>67.7</td><td>0.09</td></tr><tr><td>3 UoT</td><td>40.6</td><td>0.14</td><td>29.0</td><td>0.55</td><td>54.3</td><td>0.14</td><td>45.2</td><td>0.22</td><td>35.4</td><td>0.47</td><td>59.9</td><td>0.13</td></tr><tr><td>ReAct</td><td>46.0</td><td>0.32</td><td>46.2</td><td>0.31</td><td>60.7</td><td>0.14</td><td>53.6</td><td>0.20</td><td>45.0</td><td>0.39</td><td>65.9</td><td>0.12</td></tr><tr><td rowspan="3">GPT-oss-20B</td><td>BP</td><td>43.0</td><td>0.29</td><td>36.2</td><td>0.49</td><td>49.3</td><td>0.18</td><td>49.6</td><td>0.19</td><td>35.2</td><td>0.53</td><td>54.1</td><td>0.17</td></tr><tr><td>UoT</td><td>36.4</td><td>0.19</td><td>30.2</td><td>0.57</td><td>42.7</td><td>0.13</td><td>41.8</td><td>0.20</td><td>28.4</td><td>0.59</td><td>45.8</td><td>0.15</td></tr><tr><td>ReAct</td><td>39.0</td><td>0.32</td><td>33.4</td><td>0.52</td><td>50.6</td><td>0.19</td><td>47.6</td><td>0.28</td><td>42.8</td><td>0.38</td><td>52.0</td><td>0.21</td></tr><tr><td rowspan="3">Qwen2.5-72B</td><td>BP</td><td>43.6</td><td>0.31</td><td>43.4</td><td>0.41</td><td>42.7</td><td>0.22</td><td>49.4</td><td>0.16</td><td>45.0</td><td>0.41</td><td>48.1</td><td>0.20</td></tr><tr><td>UoT</td><td>39.2</td><td>0.27</td><td>28.4</td><td>0.52</td><td>38.9</td><td>0.23</td><td>41.8</td><td>0.29</td><td>31.2</td><td>0.51</td><td>45.0</td><td>0.18</td></tr><tr><td>ReAct</td><td>44.0</td><td>0.33</td><td>38.4</td><td>0.45</td><td>46.7</td><td>0.21</td><td>46.6</td><td>0.34</td><td>47.0</td><td>0.32</td><td>48.9</td><td>0.21</td></tr><tr><td rowspan="3">Qwen3-32B</td><td>BP</td><td>39.8</td><td>0.23</td><td>38.2</td><td>0.34</td><td>43.2</td><td>0.16</td><td>43.8</td><td>0.14</td><td>40.8</td><td>0.30</td><td>45.0</td><td>0.20</td></tr><tr><td>UoT</td><td>29.2</td><td>0.16</td><td>30.8</td><td>0.35</td><td>29.9</td><td>0.11</td><td>26.0</td><td>0.15</td><td>34.8</td><td>0.24</td><td>31.3</td><td>0.18</td></tr><tr><td>ReAct</td><td>37.6</td><td>0.31</td><td>39.2</td><td>0.41</td><td>44.7</td><td>0.13</td><td>45.4</td><td>0.21</td><td>48.0</td><td>0.26</td><td>48.7</td><td>0.17</td></tr><tr><td rowspan="3">Qwen3-8B</td><td>BP</td><td>38.8</td><td>0.22</td><td>28.6</td><td>0.40</td><td>32.1</td><td>0.18</td><td>38.4</td><td>0.23</td><td>30.8</td><td>0.37</td><td>38.5</td><td>0.18</td></tr><tr><td>UoT</td><td>27.0</td><td>0.20</td><td>19.6</td><td>0.42</td><td>24.3</td><td>0.15</td><td>27.4</td><td>0.23</td><td>20.6</td><td>0.45</td><td>26.3</td><td>0.18</td></tr><tr><td>ReAct</td><td>38.2</td><td>0.31</td><td>25.8</td><td>0.60</td><td>36.5</td><td>0.22</td><td>46.0</td><td>0.19</td><td>29.4</td><td>0.57</td><td>39.5</td><td>0.24</td></tr><tr><td rowspan="3">Qwen2.5-7B</td><td>BP</td><td>38.0</td><td>0.32</td><td>34.8</td><td>0.25</td><td>32.9</td><td>0.28</td><td>42.8</td><td>0.19</td><td>29.2</td><td>0.28</td><td>35.7</td><td>0.21</td></tr><tr><td>UoT</td><td>32.4</td><td>0.24</td><td>25.8</td><td>0.38</td><td>31.1</td><td>0.24</td><td>35.0</td><td>0.26</td><td>26.8</td><td>0.42</td><td>31.9</td><td>0.23</td></tr><tr><td>ReAct</td><td>37.0</td><td>0.31</td><td>32.8</td><td>0.55</td><td>34.9</td><td>0.25</td><td>45.0</td><td>0.26</td><td>27.8</td><td>0.59</td><td>39.5</td><td>0.27</td></tr><tr><td rowspan="3">LLaMa3.1-8B</td><td>BP</td><td>33.8</td><td>0.28</td><td>42.2</td><td>0.27</td><td>32.7</td><td>0.20</td><td>36.8</td><td>0.29</td><td>40.8</td><td>0.25</td><td>37.3</td><td>0.16</td></tr><tr><td>UoT</td><td>32.2</td><td>0.25</td><td>32.6</td><td>0.47</td><td>32.9</td><td>0.16</td><td>34.8</td><td>0.20</td><td>32.4</td><td>0.45</td><td>30.1</td><td>0.23</td></tr><tr><td>ReAct</td><td>38.0</td><td>0.28</td><td>51.4</td><td>0.17</td><td>34.3</td><td>0.18</td><td>46.4</td><td>0.18</td><td>58.2</td><td>0.16</td><td>36.3</td><td>0.22</td></tr><tr><td>VLM</td><td></td><td>27.0</td><td>0.44</td><td>35.7</td><td>0.50</td><td>25.2</td><td>0.43</td><td>20.3</td><td>0.49</td><td>33.7</td><td>0.43</td><td>25.3</td><td>0.33</td></tr></table>

Table 1: Accuracy and calibration on three task categories: Perception (SeedBench), Charts (FigureQA), and Knowledge (macro-average over MMStar, CVQA, and ReasonVQA). Full table with standard deviations across trials is reported in App. E

where sem\_ $\mathsf { e q } ( \cdot , \cdot )$ is implemented by an LLMbased aggregator that judges whether two responses are semantically equivalent. The pair $( a ^ { ( t ) } , \bar { u } ^ { ( t ) } )$ is appended to the history to form $\hat { h } ^ { ( t + 1 ) }$ . The interaction loop continues until R outputs a parsable answer or the step budget T is reached.

## 3.2 Dataset Creation

Dataset Sources. We construct our benchmark from multiple-choice VQA datasets that collectively cover diverse visual reasoning abilities. Concretely, we include three broad categories: perception on natural images, using SeedBench (Li et al., 2023) for everyday scene understanding; chart understanding, using FigureQA (Kahou et al., 2017); and knowledge-based VQA, using MMStar (Chen et al., 2024), CVQA (Romero et al., 2024), and ReasonVQA (Tran et al., 2025) to capture reasoning settings that require discipline-specific, cultural, and encyclopedic knowledge.

VLM Sensors. We deliberately utilize two modern VLM sensors, Qwen2.5-VL-7B and InternVL3.5- 8B, chosen because preliminary experiments indicate that they exhibit complementary behaviors and failure modes. We report the VLM behavior results in App. J and App. K. Since medium-sized VLMs are not perfectly calibrated, they may yield inconsistent outputs, serving as ideal noisy sensors. Specifically, as our goal is to assess LLMs’ capability, rather than VLMs’ perceptual accuracy, these VLMs better reveal LLMs’ weaknesses, where reasoners cannot rely solely on VLM confidence and reasoning abilities to make decisions.

Two-stage Filtering. To ensure that the reasoning tasks are both challenging and solvable, we adopt a two-stage filtering strategy. First, we apply a consistency-based filter to isolate questions on which the sensors themselves struggle: for each question and sensor, we sample 11 independent responses, compute a consistency score c as the number of times the ground-truth option is predicted, and retain only questions with $1 \le c \le 5$ We use self-consistency by design: it is an observable, reproducible, and externally verifiable signal that a text-only reasoner can directly perceive and act on, and it is a well-established, practical proxy for predictive uncertainty in the literature (Wang et al., 2023b; Kuhn et al., 2023; Xia et al., 2025). This guarantees that the VLM is occasionally correct but overall unreliable, so the reasoner cannot succeed by simply forwarding the original question and must engage in additional information seeking. Second, human annotators assume the role of a blind LLM reasoner: they interact with the VLM sensor via text-only queries (without seeing the image), and we keep only questions that are correctly solved by at least one annotator. Finally, we obtain 100 high-quality questions per (sensor, dataset) pair, yielding 1,000 challenging reasoning instances spanning a diverse range of VQA skills and VLM failure modes. This two-stage filter serves as a quality-control criterion: it discards instances where the VLM is either reliable or unsolvable even in theory, retaining only instances where the noisy evidence is logically resolvable by careful reasoning.

![](images/993b338016b5180e3d849c330d5ff5607da33e72cd46ea21fe813fc88f3e876c.jpg)  
Figure 2: Model performance using BP vs. VLM consistency level. The x-axis shows binned levels of how often the VLM gives the correct answer (out of 11 samples), with 1 being the lowest consistency and 5 being the highest. LLM reasoners improve over the raw VLM baseline (gray), with humans (dashed black) maintaining the highest performance across all consistency levels.

## 4 Experiments and Analysis

## 4.1 Baselines and Metrics

We evaluate a range of LLM reasoners spanning proprietary and open-source families: GPT-5 Mini, GPT-oss-20B, Qwen3.5-122B-A10B, Qwen3 (8B, 32B), Qwen2.5 (7B, 72B), and LLaMa3.1-8B. For each model, we consider three reasoning strategies.

(i) ReAct (Yao et al., 2023), where the model alternates between textual thoughts and actions, and actions correspond to querying the VLM sensor. (ii) Uncertainty of Thoughts (UoT) (Hu et al., 2024), adapted from the original information-seeking setting to VQA by modifying the prompt to describe our setup instead of medical diagnosis. (iii) Beliefstate prompting (BP), a prompting recipe we design around the failure modes we hypothesize models exhibit under noisy evidence, which we later substantiate in Sections 4.4 and 4.5. BP instructs the model to track its internal belief over answer options during the interaction and to commit to a final choice only when sufficiently certain. While we do not claim BP as a core contribution, it serves as a strong task-specific prompting ceiling to ensure underperformance is not caused by inadequate instructions.

We report accuracy and Expected Calibration Error (ECE), computed from cross-run prediction consistency, as our primary metrics (Kostumov et al., 2024). For non-proprietary models, we draw 5 trajectories per instance at temperature 1.0 and compute ECE using the agreement of the predicted option across runs. For GPT-5 Mini, due to cost constraints, we obtain a single response per instance.

In addition, we collect a human baseline by randomly sampling 50 questions for each setting and asking a different human annotator to solve them, using the same interaction protocol. In Perception (SeedBench) and Charts (FigureQA), human annotators had no access to external tools, yet the human–model gap persists, suggesting it reflects reasoning and uncertainty-handling deficits rather than information access. For knowledge-heavy datasets (MMStar, CVQA, ReasonVQA), annotators were allowed limited use of search engines or AI tools only to clarify unfamiliar terms or background knowledge required to understand the question, but not to carry out the main reasoning or come up with the final answer. This keeps the comparison fair: the benchmark targets uncertainty handling rather than penalizing annotators for lacking high-level domain knowledge in novel datasets. See App. F for detailed per-dataset results.

![](images/1a29bc148c240b753c9c0fe5277f7c89ef69979968735baf1181c23a21ea2aa4.jpg)  
Figure 3: Ablation study evaluating the impact of the consistency signal on humans and models under the Belief-State Prompting (BP) strategy.

LLM use disclosure. We use LLM-based judges for semantic grouping in Eq. (3) and for query-type annotation in Section 4.5; details and validation are provided in the appendix.

## 4.2 Analysis and Observations

Tasks are genuinely hard even for strong LLMs. The strongest models—Qwen3.5-122B-A10B, GPT-5 Mini, and Qwen2.5-72B—achieve only 40–60% accuracy in most category-level settings (at best 67.7%), while human performance reaches roughly 70–90%, underscoring that robustly reasoning under noisy visual evidence remains an open challenge for current LLMs. Among LLM reasoners, Qwen3.5-122B-A10B leads overall, followed by GPT-5 Mini; within each family, larger models generally outperform their smaller counterparts, confirming that scale partially mitigates—but does not close—the gap to human performance. Results are qualitatively consistent across the two VLM sensors, though performance is commonly higher with Qwen2.5-VL-7B.

Among strategies, BP and ReAct are the two strongest, while UoT trails in accuracy. BP, which we treat as a task-specialized baseline rather than a methodological contribution, helps most for the strongest reasoner: Qwen3.5-122B-A10B. It increases accuracy over ReAct in most settings and achieves among the lowest ECE. For weaker models, ReAct is usually on par or better, consistent with belief tracking paying off mainly when the reasoner follows it faithfully (Sec. 4.4).

<table><tr><td></td><td>SeedBench†</td><td>FigureQA CVQA MMStar ReasonVQA</td><td></td><td></td><td></td></tr><tr><td colspan="6">InternVL3.5-8B sensor</td></tr><tr><td>Qwen3-8B</td><td>38.8</td><td>28.6</td><td>26.0</td><td>32.0</td><td>38.2</td></tr><tr><td>+SFT</td><td>36.0</td><td>30.6</td><td>27.6</td><td>34.0</td><td>26.8</td></tr><tr><td colspan="6">Qwen2.5-VL-7B sensor</td></tr><tr><td>Qwen3-8B</td><td>38.4</td><td>30.8</td><td>29.0</td><td>38.4</td><td>48.2</td></tr><tr><td>+ SFT</td><td>45.8</td><td>32.6</td><td>20.6</td><td>32.0</td><td>46.8</td></tr></table>

Table 2: BP accuracy (%) before and after self-distillation on 1,800 successful SeedBench trajectories. <sup>†</sup>In-distribution.

LLM reasoners improve over raw VLM baselines. On this explicitly filtered pool, inferenceonly LLM reasoners generally outperform direct VLM predictions, yielding higher accuracy and lower ECE. For instance, with the Qwen2.5-VL-7B sensor, direct VLM achieves a highly miscalibrated ECE of 0.49 on Perception tasks, while querying it via Qwen3.5-122B-A10B (BP) reduces the ECE to 0.22 and more than doubles the accuracy. This suggests that strong LLMs can act as “cognitive errorcorrectors” for noisy perceptual systems, providing the largest lift over raw VLM baseline precisely when the VLM’s internal consistency is lowest.

Effect of VLM consistency. Figure 2 shows the accuracy of humans, VLM-direct, and LLM-based reasoners as a function of the VLM self-consistency level. Across both sensors, LLM performance increases with consistency; however, the gap between VLM-direct and LLM-based systems narrows in the easiest bins, and only the largest reasoners achieve substantial gains over the VLM baseline there. In contrast, human blind performance degrades only mildly in the lowest-consistency bins, indicating a large performance gap in the most difficult settings. In addition, reasoners paired with the better-calibrated Qwen2.5-VL benefit more from increasing consistency than those using InternVL3.5, while human accuracy remains relatively stable even with miscalibrated signals.

Ablation study of the consistency signal. Figure 3 compares BP performance of Qwen2.5-7B/72B with the true VLM answer-consistency metadata, without the metadata, and with a randomly injected signal. We observe sizable gains on most perception and diagram datasets when using the true signal, indicating that reasoners can exploit this uncertainty cue rather than just the presence of numerical metadata. In contrast, the signal can sometimes hurt on the knowledge-based benchmark (CVQA): the VLM can sometimes be confidently wrong, making true consistency a noisy or even adversarial proxy for reliability (see App. B for a qualitative example). By contrast, human accuracy drops sharply under a random signal on SeedBench and CVQA, suggesting that humans rely on the signal more than LLMs do. This presents a challenge as a reasoning task and suggests a direction for future work where reasoners learn domain-specific strategies.

## 4.3 Can VISUALNOISEQA be solved with basic fine-tuning?

As a first step, we explore whether VISUAL-NOISEQA adaptive behavior can be learned through standard supervised fine-tuning (SFT), instead of relying only on inference-time prompting. We test a self-distillation approach, fine-tuning a representative model, Qwen3-8B, on its own successful belief-state prompting (BP) traces.

Setup. We collect 1,800 successful BP trajectories from a held-out SeedBench split, balanced across InternVL3.5-8B and Qwen2.5-VL-7B. We apply LoRA fine-tuning (rank 16, $\alpha = 3 2 )$ for 7 epochs, optimizing cross-entropy on the assistant’s turns, and select the checkpoint by best validation loss.

Results. As shown in Table 2, self-distillation does not yield consistent gains: accuracy improves on some combinations (in-distribution Seed-Bench with Qwen2.5-VL-7B, FigureQA with both sensors) but degrades on others (CVQA with Qwen2.5-VL-7B, ReasonVQA with InternVL3.5- 8B), with improvements remaining sensor- and dataset-specific. This suggests that the model memorizes sensor-specific patterns instead of the adaptive strategy VISUALNOISEQA requires, formulating queries based on prior evidence and switching tactics when direct inference fails. This aligns with recent findings that supervised fine-tuning tends to fit surface patterns (Chu et al., 2025; Guo et al., 2025). We thus read this as a preliminary probe: within our setup, cloning successful trajectories does not internalize active reasoning, and the finetuned model remains far below humans (Table 1), underscoring the difficulty of VISUALNOISEQA and leaving it as an open training challenge.

## 4.4 Retrospective Belief-State Analysis

To understand how models make decisions (whether they terminate prematurely or update their belief states faithfully) and to explain the performance gap between Qwen3.5-122B-A10B and

![](images/0918bdad90f57f9a236b84a40ac7bde562fb2395a87590132ce90f5d1b5839fd.jpg)  
Figure 4: Mean $p _ { \mathrm { c o r r e c t } } ( t )$ over the belief trajectory $( b _ { 0 }  b _ { T } )$ . The figure also shows p<sub>chosen</sub> (confidence in the chosen answer) at termination.

Qwen2.5-7B, we retrospectively elicit their implicit belief states during reasoning. While this reflects the model’s self-reported interpretation rather than guaranteed internal states, the elicitation reveals consistent differences between the two reasoners.

Methodology. For each step t of a BP conversation, we replay the trajectory (2000 trajectories in total over 4 settings) and prompt the original LLM to output its implicit belief state as a probability distribution $b _ { t } \in \Delta ^ { | \mathcal { O } | }$ over the answer options (details of our setting in App. I). We track belief convergence via $p _ { \mathrm { c o r r e c t } } ( t ) = b _ { t }$ [ground-truth option] and final confidence via $p _ { \mathrm { c h o s e n } }$ (the probability assigned to the ultimately selected option).

Reported Confidence vs. Belief Convergence. Tracking $p _ { \mathrm { { c o r r e c t } } }$ over time (Figure 4) reveals differences in how the two models update beliefs. Qwen3.5-122B-A10B’s trajectories are consistent with evidence-driven convergence: its $p _ { \mathrm { { c o r r e c t } } }$ rises steadily from the uniform prior to 0.44 (Seed-Bench) and 0.57 (CVQA) at termination, while Qwen2.5-7B’s p<sub>correct</sub> rises only marginally (from 0.25 to about 0.32). Despite this limited convergence, Qwen2.5-7B shows what we call performative confidence at termination, reporting a higher mean $p _ { \mathrm { c h o s e n } }$ than the 122B model. Moreover, Qwen2.5-7B’s wrong answers carry even higher confidence (0.93) than its correct ones (0.89), suggesting its stated confidence follows the prompt’s 90% threshold more closely than the strength of the gathered evidence.

Premature Commitment and Early Anchoring. One failure mode we observe for the smaller model is deciding too early. Qwen2.5-7B frequently anchors prematurely: 46% of its SeedBench trajectories already exceed 0.7 confidence after the first evidence step, compared to 13% for Qwen3.5-122B-A10B. As shown in Table 7 (App. C), Qwen2.5-7B reaches high confidence roughly 1.6× faster than the 122B model. Notably, its incorrect trajectories commit even faster than the 122B model’s correct ones (1.9 vs. 3.1 steps on SeedBench; 2.6 vs. 3.6 on CVQA), suggesting that the smaller model’s speed reflects early anchoring rather than stronger evidence. In contrast, the 122B model gathers more evidence before committing, consistent with its steadier rise in $p _ { \mathrm { c o r r e c t } }$

<table><tr><td></td><td colspan="2">Round 1</td><td colspan="3">Round 2+</td></tr><tr><td></td><td>Descriptive</td><td>Targeted</td><td>Descriptive</td><td>Targeted</td><td>Confirmatory</td></tr><tr><td>Human</td><td>97%</td><td>3%</td><td>26%</td><td>56%</td><td>13%</td></tr><tr><td>Qwen3.5-122B-A10B</td><td>75%</td><td>25%</td><td>8%</td><td>68%</td><td>15%</td></tr><tr><td>Qwen2.5-7B</td><td>38%</td><td>57%</td><td>2%</td><td>60%</td><td>23%</td></tr></table>

Table 3: Distribution of the three most common query types by round (averaged over SeedBench and CVQA).

Within our two-model comparison, these results suggest that faithful belief tracking under uncertainty depends on reasoner capability rather than being induced by the prompt alone.

## 4.5 Querying Strategy

A key question is how human annotators achieve their large accuracy advantage. We classify each of 1,937 queries from human and model performances on SeedBench and CVQA into five types— Descriptive, Targeted, Contrastive, Confirmatory, and Comparative—using an LLM judge (the categories are defined in App. H). Table 3 reports the distribution of the three most frequent query types across models and humans split by round.

Opening strategy. Humans begin with broad scene-overview queries in virtually every conversation (97% descriptive in Round 1), building a global mental model before narrowing to specific evidence. Qwen3.5-122B-A10B follows a similar but weaker pattern (75%), while Qwen2.5-7B skips the initial scene survey in 62% of conversations, jumping directly to narrow targeted questions without first establishing global context.

Confirmatory bias in weaker models. In subsequent rounds, the sharpest divergence is the confirmatory rate: Qwen2.5-7B seeks VLM agreement (“Is it true that. . . ”, “So the answer is (B)?”) 23% of the time, nearly double the human rate (13%). Rather than stress-testing evidence by challenging the VLM’s earlier answer, the weaker model tends to solicit validation of its initial hypothesis. This overconfident acceptance of early evidence is consistent with the premature commitment pattern described in Section 4.4. The gradient in confirmatory rate (Qwen2.5-7B > Qwen3.5-122B-A10B > Human) tracks model capability inversely, suggesting that stronger reasoners more often adopt a deliberate explore-then-verify strategy.

Overall, the annotators often outperform the models through a more deliberate querying strategy. They first request a broad scene description to build an initial mental model of the image, then narrow to question-relevant details. Importantly, rather than trusting a single answer, they also issue contrastive follow-up questions to verify whether the VLM’s responses remain consistent under rephrasing.

## 4.6 Additional Analysis and Error Attribution

We validate our automated semantic equivalence aggregator against expert human annotators in App. A. The LLM judge demonstrates high agreement, confirming its reliability as a scalable proxy for determining empirical consistency. Furthermore, we explore the trade-off between computational cost and the consistency signal by increasing the VLM sampling budget from K = 5 to K = 11, which slightly lowers ECE and Standard Deviation, while the raw accuracy remains similar.

To contextualize the performance gaps and querying behaviors, App. B details two prominent failure modes. First, in knowledge-intensive domains (e.g., CVQA), the VLM sensor can be confidently wrong. In these cases, empirical consistency becomes a trap, and the reasoner must rely heavily on reasoning over descriptive features rather than VLM assertions. Second, when confronted with self-contradictory evidence, models frequently exhibit premature commitment. Unlike annotators who dynamically adjust their strategies based on the provided signal, LLM reasoners often stubbornly accept the highest-confidence response.

## 5 Conclusion

Real-world reasoning depends not just on gathering more evidence, but on deciding what to trust when the evidence itself is noisy. VISUALNOISEQA makes this challenge measurable by casting VQA as active reasoning through unreliable visual sensors: a reproducible 1,000-instance benchmark of cases where the sensor is inconsistent yet the problem remains solvable by humans, isolating uncertainty handling from raw perception and providing complementary failure modes. Even strong LLMs benefit from consistency signals but remain well below human performance, calling for methods that can actively probe, reconcile, and search over conflicting evidence.

## Limitations

Our study focuses on a controlled source of unreliable evidence: the stochastic perceptual noise from a visual sensor. This scope is our intentional design choice, as it lets us study reasoning under uncertainty separately from perception itself. However, we do not claim to characterize or measure the full space of noise and unreliability that reasoners face in practice, and it can arise from other sources, such as retrieval and web search, interactions with external tools, or modalities beyond vision. We therefore view our setting as one representative instance of a broader phenomenon, and we hope that a similar style of analysis can be extended to these additional sources of unreliability in future work.

## References

Marwa Abdulhai, Isadora White, Charlie Snell, Charles Sun, Joey Hong, Yuexiang Zhai, Kelvin Xu, and Sergey Levine. 2023. Lmrl gym: Benchmarks for multi-turn reinforcement learning with language models. arXiv preprint arXiv:2311.18232.

Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Jiaqi Wang, Yu Qiao, Dahua Lin, et al. 2024. Are we on the right way for evaluating large vision-language models? Advances in Neural Information Processing Systems, 37:27056–27087.

Tianzhe Chu, Yuexiang Zhai, Jihan Yang, Shengbang Tong, Saining Xie, Dale Schuurmans, Quoc V. Le, Sergey Levine, and Yi Ma. 2025. SFT memorizes, RL generalizes: A comparative study of foundation model post-training. arXiv preprint arXiv:2501.17161.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. 2025. DeepSeek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning. arXiv preprint arXiv:2501.12948.

Tanmay Gupta and Aniruddha Kembhavi. 2023. Visual programming: Compositional visual reasoning without training. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 14953–14962.

Zhiyuan Hu, Chumin Liu, Xidong Feng, Yilun Zhao, See-Kiong Ng, Anh Tuan Luu, Junxian He, Pang Wei Koh, and Bryan Hooi. 2024. Uncertainty of thoughts: Uncertainty-aware planning enhances information seeking in large language models. arXiv preprint arXiv:2402.03271.

Samira Ebrahimi Kahou, Vincent Michalski, Adam Atkinson, Ákos Kádár, Adam Trischler, and Yoshua

Bengio. 2017. Figureqa: An annotated figure dataset for visual reasoning. arXiv preprint arXiv:1710.07300.

Vasily Kostumov, Bulat Nutfullin, Oleg Pilipenko, and Eugene Ilyushin. 2024. Uncertainty-aware evaluation for vision-language models. arXiv preprint arXiv:2402.14418.

Lorenz Kuhn, Yarin Gal, and Sebastian Farquhar. 2023. Semantic uncertainty: Linguistic invariances for uncertainty estimation in natural language generation. In The Eleventh International Conference on Learning Representations.

Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al. 2020. Retrieval-augmented generation for knowledge-intensive nlp tasks. Advances in neural information processing systems, 33:9459–9474.

Bohao Li, Rui Wang, Guangzhi Wang, Yuying Ge, Yixiao Ge, and Ying Shan. 2023. Seed-bench: Benchmarking multimodal llms with generative comprehension. arXiv preprint arXiv:2307.16125.

Stella Li, Vidhisha Balachandran, Shangbin Feng, Jonathan Ilgen, Emma Pierson, Pang Wei W Koh, and Yulia Tsvetkov. 2024. Mediq: Question-asking llms and a benchmark for reliable interactive clinical reasoning. Advances in Neural Information Processing Systems, 37:28858–28888.

Zhaonan Li, Shijie Lu, Fei Wang, Jacob Dineen, Xiao Ye, Zhikun Xu, Siyi Liu, Young Min Cho, Bangzheng Li, Daniel Chang, et al. 2025. Unbiased visual reasoning with controlled visual inputs. arXiv preprint arXiv:2512.22183.

Reiichiro Nakano, Jacob Hilton, Suchir Balaji, Jeff Wu, Long Ouyang, Christina Kim, Christopher Hesse, Shantanu Jain, Vineet Kosaraju, William Saunders, et al. 2021. Webgpt: Browser-assisted questionanswering with human feedback. arXiv preprint arXiv:2112.09332.

Matthew Renze and Erhan Guven. 2024. Self-reflection in LLM agents: Effects on problem-solving performance. arXiv preprint arXiv:2405.06682.

David Romero, Chenyang Lyu, Haryo Akbarianto Wibowo, Teresa Lynn, Injy Hamed, Aditya Nanda Kishore, Aishik Mandal, Alina Dragonetti, Artem Abzaliev, Atnafu Lambebo Tonja, et al. 2024. Cvqa: Culturally-diverse multilingual visual question answering benchmark. arXiv preprint arXiv:2406.05967.

Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Eric Hambro, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. 2023. Toolformer: Language models can teach themselves to use tools. Advances in Neural Information Processing Systems, 36:68539–68551.

Mohit Shridhar, Xingdi Yuan, Marc-Alexandre Côté, Yonatan Bisk, Adam Trischler, and Matthew Hausknecht. 2020. Alfworld: Aligning text and embodied environments for interactive learning. arXiv preprint arXiv:2010.03768.

Dídac Surís, Sachit Menon, and Carl Vondrick. 2023. Vipergpt: Visual inference via python execution for reasoning. In Proceedings ofthe IEEE/CVF international conference on computer vision, pages 11888– 11898.

Duong T Tran, Trung-Kien Tran, Manfred Hauswirth, and Danh Le Phuoc. 2025. Reasonvqa: A multi-hop reasoning benchmark with structural knowledge for visual question answering. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 18793–18803.

Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan, and Anima Anandkumar. 2023a. Voyager: An open-ended embodied agent with large language models. arXiv preprint arXiv:2305.16291.

Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed H. Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. 2023b. Self-consistency improves chain of thought reasoning in language models. In The Eleventh International Conference on Learning Representations.

Zhiqiu Xia, Jinxuan Xu, Yuqian Zhang, and Hang Liu. 2025. A survey of uncertainty estimation methods on large language models. In Findings of the Associationfor Computational Linguistics: ACL 2025.

Shunyu Yao, Howard Chen, John Yang, and Karthik Narasimhan. 2022. Webshop: Towards scalable realworld web interaction with grounded language agents. Advances in Neural Information Processing Systems, 35:20744–20757.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik R Narasimhan, and Yuan Cao. 2023. React: Synergizing reasoning and acting in language models. In The eleventh international conference on learning representations.

Kexin Yi, Jiajun Wu, Chuang Gan, Antonio Torralba, Pushmeet Kohli, and Josh Tenenbaum. 2018. Neuralsymbolic vqa: Disentangling reasoning from vision and language understanding. Advances in neural information processing systems, 31.

Zhanke Zhou, Xiao Feng, Zhaocheng Zhu, Jiangchao Yao, Sanmi Koyejo, and Bo Han. 2025. From passive to active reasoning: Can large language models ask the right questions under incomplete information? arXiv preprint arXiv:2506.08295.

## A Additional Analysis

## A.1 Frontier VLM Sensor Pilot

To probe how the benchmark behaves when the visual sensor is a frontier model, we ran a small pilot study using Gemini 3.1 Flash Lite as the sensor with a Qwen3.5-122B-A10B reasoner on 100 InternVLuncertain instances per dataset (and Gemini 2.5 Flash on FigureQA), following the same interaction protocol as the main table. Although a rigorous two-stage filtering would extract frontier-VLM-uncertain instances separately, which is too costly and resource-intensive, we treat this as an insight into the frontiersensor regime. As shown in Table 4, when Gemini is generally reliable—as in most settings—there is no substantial difference between the reasoner under the interactive setting and VLM Direct; indeed, using an LLM reasoner can sometimes hurt the VLM Direct baseline. When Gemini 3.1 Flash Lite is genuinely uncertain (FigureQA: 37.8%), Qwen3.5-122B-A10B improves the VLM Direct baseline by around 30.8 percentage points, even exceeding the InternVL-based gain. Even though this pilot uses InternVL-uncertain instances, frontier VLMs are less suitable as sensors in our setting: as the sensor becomes reliable, the task reduces to trusting it and the reasoning challenge for the LLM degrades, whereas medium-scale VLMs expose the noisy-evidence regime the benchmark targets.

<table><tr><td>Dataset</td><td>Gemini VLM Direct</td><td>BP (Gemini sensor)</td><td>Paper BP (InternVL sensor)</td></tr><tr><td>CVQA</td><td>85.2%</td><td>81.0%</td><td>61.2%</td></tr><tr><td>FigureQA</td><td>37.8%</td><td>68.6%</td><td>59.8%</td></tr><tr><td>MMStar</td><td>62.1%</td><td>67.0%</td><td>50.6%</td></tr><tr><td>ReasonVQA</td><td>76.1%</td><td>74.6%</td><td>71.2%</td></tr><tr><td>SeedBench</td><td>49.1%</td><td>50.4%</td><td>48.6%</td></tr></table>

Table 4: Frontier VLM sensor pilot: BP accuracy with a Gemini 3.1 Flash Lite sensor (Qwen3.5-122B-A10B reasoner, 100 InternVL-uncertain instances per dataset) compared against the Gemini VLM Direct baseline and the paper’s BP with the InternVL3.5-8B sensor. Observing that Gemini 3.1 Flash Lite did not perform well on FigureQA, we tested another evaluation with Gemini 2.5 Flash as a sensor: VLM Direct 61.6%, BP 67.0%.

## A.2 LLM Judge Validation

To validate the automated semantic equivalence judge, two expert annotators independently evaluated 100 sampled response lists (balanced between those judged by Qwen2.5-7B and 72B). For each list of five VLM responses, annotators assigned a confidence score based on semantic grouping. Initial inter-annotator discrepancies were resolved through collaborative discussion to establish a consensus ground truth, which we then used to benchmark the LLM-based judge’s performance. We calculate both the agreement score and Cohen’s Kappa Metric to compare human judgments against the automated LLM judges, which are shown in Table 5. We observe high within-1-point agreement and moderate exact agreement. We also calculated a Quadratic Weighted Cohen’s Kappa and a Pearson correlation. These high values demonstrate that our LLM judge is a robust proxy for human semantic grouping. Specifically, both Qwen2.5-7B and 72B have high Pearson correlations hovering around 0.80, suggesting that the LLM judge is consistently calibrated across different model sizes.

<table><tr><td>Evaluation Metric</td><td>Value</td></tr><tr><td>Exact Agreement</td><td>66.0%</td></tr><tr><td>Within-1 Agreement</td><td>91.0%</td></tr><tr><td>Mean Absolute Error (MAE)</td><td>0.49</td></tr><tr><td>Cohen&#x27;s Kappa (Quadratic)</td><td>0.79</td></tr><tr><td>Pearson Correlation</td><td>0.80</td></tr></table>

Table 5: Validation of the LLM semantic equivalence judge against human annotators.

## A.3 Trade-off between computation cost and consistency signal

<table><tr><td></td><td></td><td colspan="2">SeedBench</td><td colspan="2">FigureQA</td><td colspan="2">CVQA</td></tr><tr><td>Reasoner</td><td>VLM Samples (K)</td><td> $\mathbf { A c c } \pm \mathbf { S t d }$ </td><td>ECE</td><td> $\mathbf { A c c } \pm \mathbf { S t d }$ </td><td>ECE</td><td> $\mathbf { A c c } \pm \mathbf { S t d }$ </td><td>ECE</td></tr><tr><td>Qwen2.5-72B</td><td> $K = 5$ </td><td> $4 3 . 6 \pm 3 . 5 1$ </td><td>0.31</td><td> $4 3 . 4 \pm 4 . 2 8$ </td><td>0.41</td><td> $4 3 . 2 \pm 2 . 6 8$ </td><td>0.19</td></tr><tr><td></td><td> $K = 1 1$ </td><td> $4 4 . 4 \pm 1 . 8 2 $ </td><td>0.27</td><td> $4 2 . 6 \pm 4 . 7 2$ </td><td>0.40</td><td> ${ \bf 4 3 . 4 \pm 2 . 3 0 }$ </td><td>0.20</td></tr><tr><td>Qwen2.5-7B</td><td> $K = 5$ </td><td> $3 8 . 0 \pm 4 . 0 0$ </td><td>0.32</td><td> $3 4 . 8 \pm 6 . 4 6$ </td><td>0.25</td><td> $2 9 . 2 \pm 6 . 1 4$ </td><td>0.32</td></tr><tr><td></td><td> $K = 1 1$ </td><td> $3 7 . 4 \pm 2 . 1 9$ </td><td>0.27</td><td> $3 1 . 8 \pm 1 . 3 0$ </td><td>0.39</td><td> $3 2 . 8 \pm 2 . 9 5$ </td><td>0.23</td></tr></table>

Table 6: Ablation study showing trade-offs between computation cost (number of VLM samples K) and the consistency signal on SeedBench, FigureQA, and CVQA.  
We expanded the sampling budget for InternVL3.5-8B to $K = 1 1$ , instead of $K = 5 ,$ on 300 SeedBench, FigureQA, and CVQA instances (100 samples for each dataset) for Qwen2.5-7B and 72B. While raw accuracy remains similar, the Expected Calibration Error (ECE) and Standard Deviation slightly decrease. Table 6, which outlines the BP performance and calibration differences, shows a modest trade-off: increasing computational cost (more sensor samples) stabilizes the uncertainty signal.

## B Qualitative Error Analysis

We present two failure cases that illustrate qualitatively distinct error modes: one in which the VLM sensor is confidently wrong, rendering the consistency signal uninformative, and one in which the VLM provides self-contradictory evidence that a human reconciles but models do not. Together, these cases concretize the quantitative gaps reported in the main text.

## B.1 VLM Confidently Wrong on Domain Knowledge

Question: What is the name of this building in Mongolia?

(A) Hunnu Mall (B) Department Store (C) Ulaanbaatar Department Store (D) Sport Center GT: (C)

This CVQA (knowledge-intensive) instance asks a reasoner to identify a Mongolian landmark (see the full interaction trace in App. L). We highlight the key contradiction. When prompted to confirm the building’s identity, the VLM answers:

R2 Q: Can you confirm whether the building is the Hunnu Mall, Department Store, Ulaanbaatar Department Store, or Sport Center?

VLM: The building in the image appears to be the Ulaanbaatar Department Store. (4/5)

R4 <sup>Q:</sup> <sup>Are</sup> <sup>there</sup> <sup>any</sup> <sup>signs,</sup> <sup>labels,</sup> <sup>or</sup> <sup>other</sup> <sup>indicators</sup> <sup>in</sup> <sup>the</sup> <sup>image</sup> <sup>that</sup> <sup>specifically</sup> <sup>identify</sup> <sup>the</sup> <sup>building</sup> <sup>as</sup> <sup>the</sup> R2 Ulaanbaatar Department Store?

VLM: No, the image does not show any visible signage or labels that clearly identify the building as the Ulaanbaatar Department Store. (4/5)

Reasoner: “The VLM’s response introduces more uncertainty . . . ” → [revises belief downward]

and R4 are in tension: if the building is the Ulaanbaatar Department Store, some perceptual basis must underlie that claim—yet R4 explicitly denies that any such basis is visible. Crucially, both answers carry identical 4/5 confidence. The consistency signal therefore provides no discriminative information: the VLM is equally self-assured when asserting and when retracting the same fact. The reasoner is forced into genuine uncertainty, where it has to place more trust in the descriptive information to resolve the problem.

## B.2 Human Inconsistency Detection vs. Model Acceptance

Question (SeedBench): What is the relative position of the woman’s boots and jeans in the image? (A) The jeans are overlapping the boots. (B) The boots are overlapping the jeans. (C) The boots and jeans are next to each other. (D) The boots andjeans are not visible in the sameframe. GT: (C)

This SeedBench instance demonstrates the contrasting strategies of a human annotator and two LLM reasoners when confronted with self-contradictory VLM feedback.

Human annotator (9 rounds, correct). The annotator begins with a broad scene description (u=2/5), then isolates each item individually before asking the VLM directly about overlap:

R5 Q: What is the relative position between the jeans and the boots?

VLM: The jeans are worn over the boots. (4/5)

R6 Q: What is the relative position between the boots and the jeans?

VLM: The boots are worn over the jeans. (5/5)

The VLM gives logically contradictory answers in consecutive rounds: “jeans over boots” (4/5) followed by “boots over jeans” (5/5). Rather than accepting the higher-confidence answer, the human changes strategy entirely, probing what objects are near each item:

R7 Q: What can be found nearby the boots?

VLM: . . . a black handbag and a person wearing . . . denim shorts. . . (3/5)

R8 Q: What can be found nearby the jeans?

VLM: . . . a black beret, a gray knitted sweater, black boots. . . (3/5)

→ (C) Next to each other ✓

By triangulating from a different angle, the annotator infers that the items sit side-by-side rather than overlapping—a conclusion that neither direct-overlap answer supported.

Qwen2.5-7B (4 rounds, wrong). The model receives the same contradictory signal—round 1 says “boots on top” (2/5) while round 3 says “jeans overlap boots” (5/5). Crucially, the model acknowledges the contradiction in its own chain-of-thought yet still commits to (A) based on the higher-confidence response.

Average Number of Queries - Qwen2.5-VL-7B

Qwen3.5-122B-A10B (5 rounds, wrong). The stronger model fares no better. It detects that rounds 1 and 2 disagree (3/5 vs. 1/5), asks two additional clarifying questions, and follows the trend of later, higher-confidence answers toward (B), which is the opposite wrong direction from the 7B model.

Takeaway. The human’s advantage is not simply asking more questions but switching inference strategy when evidence is self-contradictory—probing for indirect evidence instead of repeatedly asking the same direct question. Both models detect the contradiction but default to recency or confidence weighting rather than seeking a fundamentally different line of evidence. This qualitative gap aligns with the quantitative querying-strategy analysis (Section 4.5): models issue confirmatory follow-ups where humans pivot to novel angles.

## C Number of Queries per Setting

Number of Queries and Reasoning Strategy. Figure 5 reveals distinct trade-offs between reasoning strategies across datasets. Uncertainty-aware prompting (BP and UoT) encourages active information seeking compared to ReAct, which consistently terminates early across all models. However, UoT query counts scale aggressively and inefficiently with model capability, reaching up to 9.8 queries for GPT-5-mini on SeedBench. In contrast, Belief-State Prompting (BP) strikes a calibrated balance. BP consistently drives deeper exploration than ReAct while remaining significantly more efficient than UoT, averaging between 3.6 and 5.3 queries across datasets—a range that most closely mirrors the human strategy (4.8 to 6.4 queries). This targeted querying behavior correlates with the accuracy gains observed in the main paper, consistent with deliberate uncertainty tracking supporting effective interaction.

![](images/a7e754ac9a14732b97d3b8620c3c300aedeca8e912c6466b24bb960f1b8206da.jpg)  
Figure 5: Average Number of Rounds for 3 Datasets Under 3 Different LLM Baselines and Human Performance.

In our evaluation pipeline, the active reasoning loop terminates under one of three conditions: the model reaches sufficient confidence to select a final answer, exhausts its predefined query budget, or exceeds its context window. In reality, we set a very high maximum round limit T of 100 for both humans and LLMs across all settings, and this does not limit the LLM and human behavior. We have kept track of the conversations and found that there is no question where LLMs failed to conclude due to cutoffs by reaching the maximum round or their context windows. To make sure that humans did not brute-force the VLM sensors, we also report that the average number of queries across all 10 settings is 5.73.

<table><tr><td rowspan="2"></td><td rowspan="2"></td><td colspan="2">Steps to max p ≥ 0.7</td><td colspan="2">Reach rate</td></tr><tr><td>Correct</td><td>Wrong</td><td>Correct</td><td>Wrong</td></tr><tr><td rowspan="2">SeedBench</td><td>Qwen3.5-122B-A10B</td><td>3.1</td><td>3.0</td><td>90%</td><td>94%</td></tr><tr><td>Qwen2.5-7B</td><td>1.8</td><td>1.9</td><td>94%</td><td>98%</td></tr><tr><td rowspan="2">CVQA</td><td>Qwen3.5-122B-A10B</td><td>3.6</td><td>4.1</td><td>92%</td><td>88%</td></tr><tr><td>Qwen2.5-7B</td><td>2.1</td><td>2.6</td><td>97%</td><td>92%</td></tr></table>

Table 7: Convergence speed: mean steps to reach max<sub>i</sub> $b _ { t } [ i ] \geq 0 . 7 ,$ split by final-answer correctness. Qwen2.5-7B reaches high confidence about 1.6× faster than the 122B model on both datasets, and even its incorrect trajectories commit earlier than the 122B model’s correct ones.

## D Compute and Latency Cost

Wall-clock latency. We report the end-to-end latency of the three interaction strategies. We measure the average latency (seconds per instance) of VLM Direct, ReAct, and BP for both visual sensors on 100 SeedBench instances (batch size 16, single trial), where the text reasoner (Qwen2.5-7B) and both sensors (InternVL3.5-8B and Qwen2.5-VL-7B) are served locally with vLLM on a single H200 GPU. A single VLM query averages 0.19 s for InternVL3.5-8B and 0.25 s for Qwen2.5-VL-7B. As reported in Table 8, the interactive strategies add a modest reasoning overhead on top of the sensor: ReAct averages 1.05–1.07 s per instance and BP averages 2.42–2.48 s per instance, against 0.19–0.25 s for VLM Direct.

<table><tr><td>Method</td><td>InternVL sensor</td><td>Qwen2.5-VL sensor</td></tr><tr><td>VLM Direct (no text model)</td><td>0.19 s</td><td>0.25 s</td></tr><tr><td>ReAct</td><td>1.05 s</td><td>1.07 s</td></tr><tr><td>BP</td><td>2.42 s</td><td>2.48 s</td></tr></table>

Table 8: Average wall-clock latency (seconds per instance) of VLM Direct, ReAct, and BP for both visual sensors, measured on 100 SeedBench instances. The text reasoner (Qwen2.5-7B) and both sensors are served locally with vLLM on a single H200 GPU.

Number of VLM calls. The number of VLM inference calls scales linearly with the number of interaction rounds: a reasoner issuing r rounds makes $K \times ( r - 1 )$ VLM calls, since each perception query draws $K = 5$ samples and the final round is an answer commit that issues no query, while VLM Direct makes a single call per question. Table 9 reports the average per dataset. BP is the most query-intensive among the strategies in Table 9 (14.7–26.8 calls per question) and ReAct the least among the interactive strategies (7.0–14.5), reflecting BP’s deeper information seeking. Aggregated over a full run, the benchmark remains inexpensive to reproduce: solving 1,000 instances with the most verbose reasoner (BP with Qwen3.5-122B-A10B) issues on average $^ { 3 , 9 6 1 }$ perception queries, or roughly 19,800 VLM inference calls at $K = 5 ;$ at approximately 0.2 s per call, the total VLM inference time is about 66 minutes on a single H200 GPU with vLLM.

<table><tr><td>Dataset</td><td>VLM Direct</td><td>BP (InternVL)</td><td>BP (Qwen2.5-VL)</td><td>ReAct (InternVL)</td><td>ReAct (Qwen2.5-VL)</td></tr><tr><td>SeedBench</td><td>1</td><td>18.4</td><td>18.4</td><td>7.0</td><td>7.2</td></tr><tr><td>FigureQA</td><td>1</td><td>16.2</td><td>14.7</td><td>7.4</td><td>7.7</td></tr><tr><td>CVQA</td><td>1</td><td>21.4</td><td>21.6</td><td>9.9</td><td>10.5</td></tr><tr><td>ReasonVQA</td><td>1</td><td>26.8</td><td>21.8</td><td>14.5</td><td>11.6</td></tr><tr><td>MMStar</td><td>1</td><td>18.8</td><td>18.8</td><td>7.2</td><td>8.1</td></tr></table>

Table 9: Average number of VLM inference calls per question, computed as $K \times ( \mathrm { a v g r o u n d s - 1 } )$ with $K = 5$ (the final round is an answer commit with no VLM call); VLM Direct issues a single call per question. Reasoner: Qwen3.5-122B-A10B.

E Full Results with Standard Deviations
<table><tr><td rowspan="3"></td><td rowspan="3"></td><td colspan="6">InternVL3.5-8B</td><td colspan="6">Qwen2.5-VL-7B</td></tr><tr><td colspan="2">Perception</td><td colspan="2">Charts</td><td colspan="2">Knowledge</td><td colspan="2">Perception</td><td colspan="2">Charts</td><td colspan="2">Knowledge</td></tr><tr><td>Strategy</td><td>Acc ↑ ECE↓</td><td>Acc ↑</td><td>ECE↓</td><td>Acc ↑</td><td>ECE↓</td><td>Acc ↑</td><td>ECE↓</td><td>Acc ↑</td><td>ECE↓</td><td>Acc ↑</td><td>ECE↓</td></tr><tr><td>Human</td><td></td><td>72.0</td><td></td><td>86.0</td><td></td><td>69.3</td><td>-</td><td>70.0</td><td></td><td>92.0</td><td></td><td>74.0</td><td></td></tr><tr><td rowspan="3">GPT-5 Mini</td><td>BP</td><td>49.0</td><td></td><td>32.0</td><td></td><td>60.7</td><td>1</td><td>42.0</td><td>-</td><td>29.0</td><td></td><td>61.3</td><td>-</td></tr><tr><td>UoT</td><td>46.0</td><td>-</td><td>36.0</td><td>1</td><td>50.0</td><td>-</td><td>39.0</td><td>-</td><td>30.0</td><td></td><td>58.0</td><td>-</td></tr><tr><td>ReAct</td><td>36.0</td><td></td><td>46.0</td><td>-</td><td>56.7</td><td>=</td><td>42.0</td><td>1</td><td>35.0</td><td>1</td><td>63.0</td><td>=</td></tr><tr><td rowspan="2">Qwen3.5-122B-A10B UoT</td><td>BP</td><td> $4 8 . 6 \pm 1 . 5 2 $   $4 0 . 6 \pm 4 . 3 9$ </td><td>0.27 0.14</td><td> ${ \pm \mathbf { 9 . 8 \ : \pm 8 . 7 6 } }$ </td><td>0.14</td><td> ${ \bf 6 1 . 0 \pm } 1 . 8 0$ </td><td>0.12</td><td> $5 0 . 8 \pm 3 . 0 3$ </td><td>0.22</td><td> ${ \bf 6 0 . 6 \pm 6 . 1 5 }$ </td><td>0.18</td><td> ${ \bf 6 7 . 7 \pm } 2 . 0 7$ </td><td>0.09 0.13</td></tr><tr><td>ReAct</td><td> $4 6 . 0 \pm 3 . 6 7$ </td><td>0.32</td><td> $2 9 . 0 \pm 4 . 3 0$   $4 6 . 2 \pm 8 . 4 1$ </td><td>0.55 0.31</td><td> $5 4 . 3 \pm 0 . 8 9$   $6 0 . 7 \pm 2 . 7 9$ </td><td>0.14 0.14</td><td> $4 5 . 2 \pm 5 . 4 0$   ${ \pm 3 . 6 \pm 6 . 1 9 }$ </td><td>0.22 0.20</td><td> $3 5 . 4 \pm 2 . 1 9$   $4 5 . 0 \pm 4 . 1 8$ </td><td>0.47 0.39</td><td> $5 9 . 9 \pm 1 . 5 6 $   $6 5 . 9 \pm 2 . 0 6$ </td><td>0.12</td></tr><tr><td rowspan="2">GPT-oss-20B</td><td>BP</td><td> $4 3 . 0 \pm 6 . 6 7$ </td><td>0.29</td><td> $3 6 . 2 \pm 4 . 6 6$ </td><td>0.49</td><td> $4 9 . 3 \pm 1 . 8 6$ </td><td>0.18</td><td> $4 9 . 6 { \pm } 4 . 3 4 $ </td><td>0.19</td><td> $3 5 . 2 \pm 2 . 7 7$ </td><td>0.53</td><td> $5 4 . 1 \pm 2 . 4 0 $ </td><td>0.17</td></tr><tr><td>UoT ReAct</td><td> $3 6 . 4 \pm 4 . 7 2$   $3 9 . 0 \pm 5 . 5 7$ </td><td>0.19 0.32</td><td> $3 0 . 2 \pm 5 . 2 2$   $3 3 . 4 \pm 4 . 6 2$ </td><td>0.57 0.52</td><td> $4 2 . 7 \pm 3 . 0 2$   $5 0 . 6 \pm 2 . 0 0$ </td><td>0.13 0.19</td><td> $4 1 . 8 \pm 4 . 6 0$   $4 7 . 6 \pm 4 . 9 8$ </td><td>0.20 0.28</td><td> $2 8 . 4 \pm 2 . 6 1$   $4 2 . 8 \pm 3 . 5 6$ </td><td>0.59 0.38</td><td> $4 5 . 8 \pm 3 . 2 1$   $5 2 . 0 \pm 2 . 0 2$ </td><td>0.15 0.21</td></tr><tr><td rowspan="2">Qwen2.5-72B</td><td>BP</td><td> $4 3 . 6 \pm 3 . 5 1$ </td><td>0.31</td><td> $4 3 . 4 \pm 4 . 2 8$ </td><td>0.41</td><td> $4 2 . 7 \pm 2 . 5 4$ </td><td>0.22</td><td> $4 9 . 4 \pm 5 . 9 4$ </td><td>0.16</td><td> $4 5 . 0 \pm 3 . 1 6$ </td><td>0.41</td><td> $4 8 . 1 \pm 2 . 3 7$ </td><td>0.20</td></tr><tr><td>UoT</td><td> $3 9 . 2 \pm 2 . 9 5$ </td><td>0.27</td><td> $2 8 . 4 \pm 5 . 5 5$ </td><td>0.52</td><td> $3 8 . 9 \pm 3 . 2 8 $ </td><td>0.23</td><td> $4 1 . 8 \pm 5 . 3 1$ </td><td>0.29</td><td> $3 1 . 2 \pm 2 . 8 6$ </td><td>0.51</td><td> $4 5 . 0 \pm 1 . 8 7$ </td><td>0.18</td></tr><tr><td rowspan="2"></td><td>ReAct</td><td> $4 4 . 0 \pm 4 . 1 8$ </td><td>0.33</td><td> $3 8 . 4 \pm 6 . 6 9$ </td><td>0.45</td><td> $4 6 . 7 \pm 2 . 0 3$ </td><td>0.21</td><td> $4 6 . 6 \pm 2 . 6 1$ </td><td>0.34</td><td> $4 7 . 0 \pm 6 . 4 4$ </td><td>0.32</td><td> $4 8 . 9 \pm 2 . 0 5$ </td><td>0.21</td></tr><tr><td>BP</td><td> $3 9 . 8 \pm 5 . 0 7$ </td><td>0.23</td><td> $3 8 . 2 \pm 4 . 9 7$ </td><td>0.34</td><td> $4 3 . 2 \pm 2 . 1 8$ </td><td>0.16</td><td> $4 3 . 8 \pm 5 . 3 6$ </td><td>0.14</td><td> $4 0 . 8 \pm 3 . 0 3$ </td><td>0.30</td><td> $4 5 . 0 \pm 1 . 9 4$ </td><td>0.20</td></tr><tr><td rowspan="2">Qwen3-32B</td><td>UoT</td><td> $2 9 . 2 \pm 3 . 7 0 $ </td><td>0.16</td><td> $3 0 . 8 \pm 3 . 7 7$ </td><td>0.35</td><td> $2 9 . 9 \pm 2 . 2 8$ </td><td>0.11</td><td> $2 6 . 0 \pm 4 . 3 0$ </td><td>0.15</td><td> $3 4 . 8 \pm 5 . 7 6$ </td><td>0.24</td><td> $3 1 . 3 \pm 2 . 6 7$ </td><td>0.18</td></tr><tr><td>ReAct</td><td> $3 7 . 6 \pm 6 . 1 1$ </td><td>0.31</td><td> $3 9 . 2 \pm 5 . 2 6 $ </td><td>0.41</td><td> $4 4 . 7 \pm 2 . 6 1$ </td><td>0.13</td><td> $4 5 . 4 \pm 2 . 1 9$ </td><td>0.21</td><td> $4 8 . 0 \pm 2 . 7 4$ </td><td>0.26</td><td> $4 8 . 7 \pm 2 . 3 3 $ </td><td>0.17</td></tr><tr><td rowspan="2">Qwen3-8B</td><td>BP</td><td> $3 8 . 8 \pm 2 . 1 7 $ </td><td>0.22</td><td> $2 8 . 6 \pm 4 . 1 6$ </td><td>0.40</td><td> $3 2 . 1 \pm 2 . 0 9$ </td><td>0.18</td><td> $3 8 . 4 \pm 3 . 3 6 $ </td><td>0.23</td><td> $3 0 . 8 \pm 4 . 6 6$ </td><td>0.37</td><td> $3 8 . 5 \pm 2 . 1 3 $ </td><td>0.18</td></tr><tr><td>UoT</td><td> $2 7 . 0 \pm 2 . 5 5$ </td><td>0.20</td><td> $1 9 . 6 \pm 3 . 4 4$ </td><td>0.42</td><td> $2 4 . 3 \pm 3 . 2 6$ </td><td>0.15</td><td> $2 7 . 4 \pm 3 . 7 1$ </td><td>0.23</td><td> $2 0 . 6 \pm 2 . 7 0$ </td><td>0.45</td><td> $2 6 . 3 \pm 1 . 6 8$ </td><td>0.18</td></tr><tr><td rowspan="2"></td><td>ReAct</td><td> $3 8 . 2 \pm 1 . 3 0 $ </td><td>0.31</td><td> $2 5 . 8 \pm 3 . 6 3$ </td><td>0.60</td><td> $3 6 . 5 \pm 1 . 5 4$ </td><td>0.22</td><td> $4 6 . 0 \pm 3 . 5 4$ </td><td>0.19</td><td> $2 9 . 4 \pm 2 . 0 7$ </td><td>0.57</td><td> $3 9 . 5 \pm 2 . 1 9$ </td><td>0.24</td></tr><tr><td>BP</td><td> $3 8 . 0 \pm 4 . 0 0 $ </td><td>0.32</td><td> $3 4 . 8 \pm 6 . 4 6$ </td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.21</td></tr><tr><td rowspan="2"> $\mathrm { Q w e n } 2 . 5 – 7 \mathrm { B }$ </td><td>UoT</td><td> $3 2 . 4 \pm 5 . 1 3$ </td><td>0.24</td><td> $2 5 . 8 \pm 6 . 5 0$ </td><td>0.25 0.38</td><td> $3 2 . 9 \pm 3 . 0 8$ </td><td>0.28 0.24</td><td> $4 2 . 8 \pm 3 . 7 0$   $3 5 . 0 \pm 1 . 5 8$ </td><td>0.19 0.26</td><td> $2 9 . 2 \pm 3 . 1 9$   $2 6 . 8 \pm 3 . 4 2$ </td><td>0.28 0.42</td><td> $3 5 . 7 \pm 2 . 4 4$ </td><td>0.23</td></tr><tr><td>ReAct</td><td> $3 7 . 0 \pm 2 . 1 2$ </td><td>0.31</td><td> $3 2 . 8 \pm 2 . 1 7$ </td><td>0.55</td><td> $3 1 . 1 \pm 2 . 2 4$   $3 4 . 9 \pm 2 . 2 1$ </td><td>0.25</td><td> $4 5 . 0 \pm 6 . 7 1$ </td><td>0.26</td><td> $2 7 . 8 \pm 3 . 3 5$ </td><td>0.59</td><td> $3 1 . 9 \pm 2 . 3 5$   $3 9 . 5 \pm 2 . 9 5$ </td><td>0.27</td></tr><tr><td rowspan="2">LLaMa3.1-8B</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>BP</td><td> $3 3 . 8 \pm 7 . 1 6$ </td><td>0.28 0.25</td><td> $4 2 . 2 \pm 5 . 7 2$ </td><td>0.27</td><td> $3 2 . 7 \pm 3 . 1 6$ </td><td>0.20</td><td> $3 6 . 8 \pm 2 . 2 8$ </td><td>0.29</td><td> $4 0 . 8 \pm 3 . 8 3$ </td><td>0.25</td><td> $3 7 . 3 \pm 2 . 1 8$ </td><td>0.16 0.23</td></tr><tr><td rowspan="2"></td><td>UoT ReAct</td><td> $3 2 . 2 \pm 3 . 6 3$ </td><td>0.28</td><td> $3 2 . 6 \pm 5 . 4 1$   $5 1 . 4 \pm 4 . 6 2$ </td><td>0.47</td><td> $3 2 . 9 \pm 2 . 4 0$ </td><td>0.16</td><td> $3 4 . 8 \pm 1 . 9 2$ </td><td>0.20</td><td> $3 2 . 4 \pm 4 . 3 4$ </td><td>0.45 0.16</td><td> $3 0 . 1 \pm 3 . 7 3$ </td><td>0.22</td></tr><tr><td></td><td> $3 8 . 0 \pm 2 . 9 2 $ </td><td></td><td></td><td>0.17</td><td> $3 4 . 3 \pm 2 . 9 2$ </td><td>0.18</td><td> $4 6 . 4 \pm 2 . 4 1$ </td><td>0.18</td><td> $5 8 . 2 \pm 1 . 7 9$ </td><td></td><td> $3 6 . 3 \pm 2 . 2 5$ </td><td></td></tr><tr><td>VLM</td><td>=</td><td> $2 7 . 0 \pm 3 . 3 8$ </td><td>0.44</td><td> $3 5 . 7 \pm 3 . 7 4$ </td><td>0.50</td><td> $2 5 . 2 \pm 2 . 0 0$ </td><td>0.43</td><td>20.3 ±2.61</td><td>0.49</td><td> $3 3 . 7 \pm 5 . 2 9$ </td><td>0.43 25.3 ±2.32</td><td></td><td>0.33</td></tr></table>

Table 10: Full results with standard deviations across trials. Accuracy and calibration on three task categories: Perception (SeedBench), Charts (FigureQA), and Knowledge (macro-average over MMStar, CVQA, and ReasonVQA).

F Knowledge Category Breakdown
<table><tr><td rowspan="3"></td><td rowspan="3"></td><td colspan="6">InternVL3.5-8B</td><td colspan="6">Qwen2.5-VL-7B</td></tr><tr><td colspan="2">MMStar</td><td colspan="2">CVQA</td><td colspan="2">ReasonVQA</td><td colspan="2">MMStar</td><td colspan="2">CVQA</td><td colspan="2">ReasonVQA</td></tr><tr><td>Strategy Acc ↑</td><td>ECE↓</td><td>Acc ↑</td><td>ECE↓</td><td>Acc ↑</td><td>ECE↓</td><td>Acc ↑</td><td>ECE↓</td><td>Acc ↑</td><td>ECE↓</td><td>Acc ↑</td><td>ECE↓</td></tr><tr><td>Human</td><td>-</td><td>60.00</td><td>-</td><td>78.00</td><td>-</td><td>70.00</td><td>-</td><td>68.00</td><td>-</td><td>78.00</td><td>-</td><td>76.00</td><td>1</td></tr><tr><td rowspan="2">GPT-5 Mini</td><td>BP UoT</td><td>52.00</td><td></td><td>61.00</td><td>-</td><td>69.00</td><td></td><td>51.00</td><td></td><td>53.00</td><td></td><td>80.00</td><td></td></tr><tr><td>ReAct</td><td>39.00 49.00</td><td>-</td><td>63.00 56.00</td><td></td><td>48.00 65.00</td><td>- -</td><td>43.00 53.00</td><td>-</td><td>59.00 58.00</td><td>-</td><td>72.00 78.00</td><td>-</td></tr><tr><td rowspan="2">Qwen3.5-122B-A10B UoT</td><td>BP</td><td> $5 0 . 6 \pm 2 . 7 0 $ </td><td>0.20</td><td> $6 1 . 2 \pm 2 . 8 6$ </td><td>0.09</td><td> $7 1 . 2 \pm 1 . 9 2$ </td><td>0.05</td><td> $6 1 . 0 \pm 3 . 5 4$ </td><td>0.13</td><td> $6 1 . 0 \pm 4 . 6 4$ </td><td>0.10</td><td> $8 1 . 2 \pm 3 . 2 7$ </td><td>0.05</td></tr><tr><td></td><td> $4 4 . 4 \pm 3 . 2 1$ </td><td>0.17</td><td> $5 3 . 2 \pm { 3 . 8 3 }$ </td><td>0.18</td><td> $6 5 . 4 \pm 3 . 2 9$ </td><td>0.07</td><td> $5 2 . 2 \pm 4 . 7 6$ </td><td>0.16</td><td> $5 3 . 6 \pm 5 . 1 3$ </td><td>0.12</td><td> $7 3 . 8 \pm 2 . 7 7$ </td><td>0.10</td></tr><tr><td rowspan="2">GPT-oss-20B</td><td>ReAct BP</td><td> $5 0 . 8 \pm 5 . 8 9$ </td><td>0.18</td><td> $6 2 . 2 \pm 4 . 4 9$ </td><td>0.14</td><td> $6 9 . 0 \pm 3 . 3 2 $ </td><td>0.09</td><td> $5 7 . 8 \pm 3 . 8 3$ </td><td>0.15</td><td> $6 0 . 0 \pm 3 . 8 1$ </td><td>0.13</td><td> $8 0 . 0 \pm 4 . 9 0 $ </td><td>0.07</td></tr><tr><td></td><td> $4 7 . 6 0 \pm 4 . 2 8$ </td><td>0.18</td><td> $4 0 . 0 0 \pm 2 . 5 5$ </td><td>0.26</td><td> $6 0 . 2 0 \pm 2 . 4 9$ </td><td>0.09</td><td> $5 1 . 4 0 \pm 3 . 3 6$ </td><td>0.17</td><td>40.80 ±4.32</td><td>0.25</td><td> $7 0 . 0 0 \pm 4 . 6 9$ </td><td>0.09</td></tr><tr><td rowspan="2"></td><td>UoT ReAct</td><td> $3 7 . 0 0 \pm 5 . 1 0$ </td><td>0.19 0.25</td><td> $4 1 . 4 0 \pm 1 . 6 7$ </td><td>0.09</td><td> $4 9 . 8 0 \pm 7 . 2 9$ </td><td>0.10</td><td> $4 1 . 6 0 \pm 4 . 9 3$ </td><td>0.14</td><td> $3 5 . 8 0 \pm 5 . 7 6$ </td><td>0.20</td><td> $6 0 . 0 0 \pm 5 . 9 6$ </td><td>0.11</td></tr><tr><td></td><td>48.00 ±4.47</td><td></td><td> $4 5 . 2 0 \pm 2 . 2 8$ </td><td>0.19</td><td> $5 8 . 6 0 \pm 3 . 2 9$ </td><td>0.13</td><td> $4 8 . 6 0 \pm 5 . 4 1$ </td><td>0.23</td><td>38.60 ±2.30</td><td>0.30</td><td> $6 8 . 8 0 \pm 1 . 4 8 $ </td><td>0.09</td></tr><tr><td rowspan="2">Qwen2.5-72B</td><td>BP UoT</td><td>42.00 ±6.08</td><td>0.24</td><td> $4 3 . 2 0 \pm 2 . 6 8$ </td><td>0.19</td><td>42.80 ±3.70</td><td>0.22</td><td> $4 2 . 6 0 \pm 3 . 1 3$ </td><td>0.22</td><td>40.40 ±4.04</td><td>0.29</td><td>61.40 ±4.93</td><td>0.09</td></tr><tr><td></td><td>37.60 ±6.35</td><td>0.20</td><td>39.00 ±5.10</td><td>0.28</td><td>40.20 ±5.54</td><td>0.21</td><td> $4 1 . 6 0 \pm 2 . 3 0$ </td><td>0.21</td><td>41.00 ±3.32</td><td>0.22</td><td>52.40 ±3.91</td><td>0.11</td></tr><tr><td rowspan="2">Qwen3-32B</td><td>ReAct BP</td><td> $4 5 . 4 0 \pm 2 . 5 1$ </td><td>0.25</td><td>48.80 ±3.27</td><td>0.18</td><td>46.00 ±4.47</td><td>0.22</td><td> $4 5 . 0 0 \pm 3 . 3 2$ </td><td>0.24</td><td>37.60 ±2.07</td><td>0.31</td><td> $6 4 . 2 0 \pm 4 . 7 6$ </td><td>0.09</td></tr><tr><td></td><td> $4 1 . 4 0 \pm 4 . 1 6$ </td><td>0.19</td><td> $4 1 . 6 0 \pm 3 . 9 1$ </td><td>0.17</td><td>46.60 ±3.21</td><td>0.13</td><td> $4 2 . 0 0 \pm 2 . 5 5$ </td><td>0.19</td><td> $3 4 . 8 0 \pm 2 . 5 9$ </td><td>0.24</td><td> $5 8 . 2 0 \pm 4 . 5 5$ </td><td>0.16</td></tr><tr><td rowspan="2"></td><td>UoT ReAct</td><td> $2 6 . 2 0 \pm 3 . 1 1$ </td><td>0.13</td><td>35.40 ±4.28</td><td>0.06</td><td>28.20 ±4.32</td><td>0.14</td><td> $2 8 . 6 0 \pm 4 . 1 6$ </td><td>0.15</td><td> $2 8 . 4 0 \pm 5 . 4 1$ </td><td>0.20</td><td> $3 6 . 8 0 \pm 4 . 2 1$ </td><td>0.18</td></tr><tr><td></td><td> $4 3 . 0 0 \pm 4 . 0 0$ </td><td>0.16</td><td>41.20 ±6.14</td><td>0.17</td><td>49.80 ±2.77</td><td>0.07</td><td> $4 7 . 2 0 \pm 4 . 7 6$ </td><td>0.15</td><td> $3 7 . 2 0 \pm 3 . 5 6 $ </td><td>0.23</td><td> $6 1 . 6 0 \pm 3 . 6 5$ </td><td>0.12</td></tr><tr><td rowspan="2">Qwen3-8B</td><td>BP UoT</td><td> $3 2 . 0 0 \pm 4 . 0 0$ </td><td>0.19</td><td>26.00 ±3.39</td><td>0.25</td><td>38.20 ±3.42</td><td>0.11</td><td> $3 8 . 4 0 \pm { 3 . 7 8 }$ </td><td>0.14</td><td>29.00 ±4.18</td><td>0.27</td><td>48.20 ±3.03</td><td>0.12</td></tr><tr><td></td><td> $2 3 . 0 0 \pm 6 . 6 7$ </td><td>0.19</td><td> $2 5 . 2 0 \pm 5 . 4 0$ </td><td>0.11</td><td> $2 4 . 8 0 \pm 4 . 6 6$ </td><td>0.16</td><td> $2 7 . 6 0 \pm 1 . 9 5$ </td><td>0.12</td><td> $2 2 . 8 0 \pm { 3 . 6 3 }$ </td><td>0.21</td><td> $2 8 . 6 0 \pm 2 . 8 8$ </td><td>0.22</td></tr><tr><td rowspan="2">Qwen2.5-7B</td><td>ReAct</td><td> $4 0 . 0 0 \pm 3 . 4 6$ </td><td>0.20</td><td> $2 9 . 4 0 \pm 1 . 5 2$ </td><td>0.31</td><td> $4 0 . 2 0 \pm 2 . 6 8$ </td><td>0.16</td><td> $3 6 . 8 0 \pm 2 . 1 7$ </td><td>0.32</td><td> $3 3 . 6 0 \pm 4 . 9 3$ </td><td>0.27</td><td> $4 8 . 2 0 \pm 3 . 7 7$ </td><td>0.12</td></tr><tr><td>BP</td><td> $3 2 . 8 0 \pm 6 . 5 3 $ </td><td>0.33</td><td> $2 9 . 2 0 \pm 6 . 1 4$ </td><td>0.32</td><td> $3 6 . 8 0 \pm 2 . 2 8$ </td><td>0.20</td><td> $3 2 . 8 0 \pm 4 . 0 9$ </td><td>0.21</td><td>30.60 ±4.72</td><td>0.32</td><td> $4 3 . 8 0 \pm 3 . 8 3$ </td><td>0.09</td></tr><tr><td rowspan="2"></td><td>UoT</td><td> $3 3 . 8 0 \pm 4 . 0 9$ </td><td>0.19</td><td> $2 9 . 4 0 \pm 2 . 9 7$ </td><td>0.28</td><td> $3 0 . 2 0 \pm 4 . 4 4$ </td><td>0.26</td><td> $2 9 . 4 0 \pm 5 . 1 3$ </td><td>0.25</td><td>29.20 ±2.86</td><td>0.30</td><td> $3 7 . 2 0 \pm 3 . 9 0$ </td><td>0.15</td></tr><tr><td>ReAct</td><td> $3 8 . 2 0 \pm 3 . 5 6 $ </td><td>0.20</td><td> $3 1 . 2 0 \pm 3 . 1 1$ </td><td>0.31</td><td> $3 5 . 4 0 \pm 4 . 6 7$ </td><td>0.25</td><td> $3 5 . 6 0 \pm 6 . 4 3$ </td><td>0.34</td><td>33.00 ±3.08</td><td>0.37</td><td> $5 0 . 0 0 \pm 5 . 2 4$ </td><td>0.11</td></tr><tr><td rowspan="2">LLaMa3.1-8B</td><td>BP</td><td> $2 9 . 8 0 \pm 5 . 8 1$ </td><td>0.20</td><td> $3 7 . 8 0 \pm 4 . 5 5$ </td><td>0.14</td><td> $3 0 . 4 0 \pm 5 . 9 4$ </td><td>0.27</td><td> $3 2 . 8 0 \pm 3 . 9 6$ </td><td>0.17</td><td>36.00 ±2.45</td><td>0.21</td><td> $4 3 . 2 0 \pm 4 . 6 0$ </td><td>0.11</td></tr><tr><td>UoT</td><td> $2 9 . 0 0 \pm 4 . 7 4$ </td><td>0.16</td><td> $3 7 . 8 0 \pm 3 . 9 0$ </td><td>0.12</td><td> $3 1 . 8 0 \pm { 3 . 7 7 }$ </td><td>0.22</td><td> $2 5 . 6 0 \pm 7 . 0 9$ </td><td>0.24</td><td>27.60 ±3.78</td><td>0.29</td><td> $3 7 . 0 0 \pm 7 . 7 8$ </td><td>0.17</td></tr><tr><td>VLM</td><td>ReAct</td><td> $3 0 . 8 0 \pm 1 . 7 9$  27.73 ±3.32</td><td>0.18</td><td> $3 9 . 6 0 \pm 5 . 4 1$ </td><td>0.12</td><td> $3 2 . 4 0 \pm 6 . 6 6$ </td><td>0.24</td><td> $3 2 . 0 0 \pm 5 . 0 0$ </td><td>0.22</td><td>32.20 ±3.96</td><td>0.30</td><td> $4 4 . 6 0 \pm 2 . 1 9$ </td><td>0.15</td></tr></table>

Table 11: Knowledge-based category breakdown showing accuracy and calibration on: MMStar, CVQA, and ReasonVQA.

## G Baseline Prompt

## Belief-State Prompt Used to Evaluate BP method

## Belief Tracking

• Initialize: If 2 options → 50% each; If 4 options → 25% each

• Update beliefs after each question based on response patterns and the VLM’s confidence

• Aim for ≥90% confidence before finalizing your answer

• Only provide final answer when truly confident you’ve eliminated alternatives

• Responses with low confidence score will need to be further cross-validated because the VLM is providing conflicting information.

Output Format At every step, you must include the following and correct format:

• Thought: Before every question or final answer, explicitly state your thought process by outputting ‘Thought: <complete description of your rationales>’.

• Action: Then output exactly one of:

– ‘My question is: <fully self-contained question>’

– ‘The answer is: (A)’ or ‘(B)’, ‘(C)’, ‘(D)’

## Critical Format Rules

• Phrase “My question is: ” must appear exactly when asking a question

• Phrase “The answer is: ” must appear exactly when providing final answer

• Output only ONE thought & ONE action per turn

• Each “My question” must include all necessary context (e.g., “about the largest red shape”, “regarding the texture of the object on the right”).

• You may ask multiple questions (up to the high interaction budget), so you should not rely too heavily on early questions.

Decision Points Ask another question when:

• You’re uncertain which option is correct

• You got conflicting answers that need clarification (The responses with low confidence score, such as 1/5 or 2/5)

• You want to verify your leading hypothesis

• You haven’t yet tested all the key differences between options

Provide your final answer when:

• One option is clearly supported by multiple reliable observations

• You’ve tested the main alternatives and ruled them out

• You are ≥90% confident about the final option

• You’ve asked enough questions to feel justified in your conclusion

## KEY REMINDERS

• Think about what you actually learned and what you’re still uncertain about

• Ask follow-ups when you notice contradictions

• Make decisions when you feel reasonably certain

• Each question should be fully self-contained with all necessary context

Begin now with your first question. Think about what distinguishing features matter for this problem, and ask an exploratory question that will help you understand the scene.

## UoT: Prompt for Generating Candidate Questions In Visual Setting

You are helping a blind reasoner answer an MCQ about an image. You will generate {n\_candidates} diverse visual   
questions that could help distinguish between the MCQ options.   
MCQ Question: {mcq\_question}   
Options: {options}   
Round: {round\_num}   
Previous questions and answers:   
{history}   
Generate exactly {n\_candidates} diverse visual questions that could help distinguish between the MCQ options. Each   
question should:   
1. Be self-contained (include all necessary context)   
2. Target a specific visual feature that differs between options   
3. Help eliminate at least one MCQ option   
Output a JSON array of exactly {n\_candidates} question strings:   
["Question 1 about specific visual feature",   
"Question 2 about another feature", ...]   
Output ONLY the JSON array, no other text.

## UoT: Prompt for Information Gain Estimation In Visual Setting

You are estimating how informative a visual question would be for solving an MCQ.   
MCQ Question: {mcq\_question}   
Options: {options}   
Current belief: {belief\_str}   
Candidate visual question: “{candidate\_question}”   
Imagine a vision model sees the image and answers this question. Predict 2-3 most likely responses and how each would   
change your beliefs over the MCQ options. Output ONLY the JSON object in the exact format below (no other text):   
{   
"outcomes": [   
{   
"response": "likely answer 1",   
"probability": 0.5,   
"new\_belief": {example\_belief}   
},   
{   
"response": "likely answer 2",   
"probability": 0.3,   
"new\_belief": {example\_belief2}   
},   
{   
"response": "uncertain/unclear",   
"probability": 0.2,   
"new\_belief": {current\_belief}   
}   
]   
}   
IMPORTANT:   
• Probabilities must sum to 1.0   
• Each new\_belief must have exactly {n\_options} values summing to 1.0   
• Output only valid JSON

![](images/c8e5c36022d79d1e8b8a4e97c0c13b7a36dac46e1b3bad1935a6d081344f9925.jpg)

Begin now. Think about what visual information would help distinguish between the answer options, then ask your first question.

## H Query Type Classification

Query Categories. We classify each reasoner query into exactly one of five types:

• Descriptive — Broad, open-ended request for a scene or image overview. The reasoner is building an initial mental model without targeting a specific hypothesis (e.g., “Describe the image”, “What do you see?”).

• Targeted — Narrow question about a specific visual attribute, object, or detail relevant to the MCQ options. The reasoner already has a direction and seeks focused evidence (e.g., “What color is the shirt?”, “How many people are visible?”).

• Contrastive — Rephrases or stress-tests a prior VLM answer to probe its stability. Often uses phrasing such as “or”, “actually”, “are you sure”, or asks the same thing in a different way (e.g., “You said red—could it be orange?”).

• Confirmatory — Directly asks the VLM to verify or agree with a specific claim or option. Unlike Contrastive, it seeks confirmation rather than challenge (e.g., “Can you confirm this is a department store?”, “So the answer is (B), correct?”).

• Comparative — Asks the VLM to rank or evaluate multiple MCQ options against the image simultaneously (e.g., “Between a mall and a department store, which fits better?”).

Pipeline. We classify 1,937 queries drawn from one trial each of human, Qwen3.5-122B-A10B (BP), and Qwen2.5-7B (BP) runs on SeedBench and CVQA (262, 225, 379, 469, 270, and 332 queries respectively). For each query, we provide the LLM judge with the MCQ question, the query itself, the round number, and the list of prior queries in that conversation. The judge outputs a one-sentence rationale followed by a structured JSON response with the assigned type and a confidence score.

Validation. Following the same annotation proxy used for the semantic equivalence judge, three annotators independently labeled 200 stratified instances (100 per model: 40 per query type from Qwen3.5-122B-A10B and Qwen2.5-7B). Disagreements were resolved through discussion to form a consensus label, and an item is counted as correct if at least one annotator agrees with the judge. The judge achieves a union match rate of 82.5% (165/200) and a Cohen’s κ of 0.781, confirming it as a reliable proxy for query-type annotation.

## I Belief Probe Assessment

Detailed Setup. For each stored BP conversation of length T rounds, we replay the trajectory using an assessment prompt, then ask the same LLM to output a probability distribution $b _ { t } \in \Delta ^ { | \mathcal { O } | }$ over the MCQ options based only on the evidence available up to step t.

Concretely, let $h _ { 1 }$ denote the MCQ question message, $h _ { 2 t }$ the VLM response (with consistency score) at round t, and $h _ { 2 t + 1 }$ the model’s subsequent chain-of-thought. The context fed to the model for probe step $t \left( 1 \leq t \leq T - 1 \right)$ is:

$$
[ p _ { \mathrm { a s s e s s } } , h _ { 1 } , h _ { 2 } , h _ { 3 } , . . . , h _ { 2 t } , h _ { 2 t + 1 } ] ,\tag{4}
$$

ending with the model’s own chain-of-thought after processing the t-th VLM response. The initial belief $b _ { 0 }$ is set to the uniform prior, which is hardcoded.

We ran this procedure over 2,000 trajectories across four settings (Qwen3.5-122B-A10B and Qwen2.5-7B on SeedBench and CVQA). The assessment prompt $p _ { \mathrm { a s s e s s } }$ , shown below, replaces the original BP system prompt:

## Assessment Prompt for Belief Probing

You are re-examining your own past reasoning on a Visual Question Answering (VQA) task. Imagine yourself as the reasoner at this exact point in the conversation—you have seen only the evidence gathered so far, and you have not yet seen what comes next. Based on the reasoning you expressed—your thoughts, interpretations, stated confidence or doubt—trace back what your implicit belief state was and express it as a probability distribution over the MCQ options. Your response MUST end with a JSON object on its own line. Example: {“A”: 0.25, “B”: 0.50, “C”: 0.15, “D”: 0.10}

Rules: Include ALL options; probabilities must sum to 1.0; be faithful to what the reasoning actually expressed—do not inject new analysis.

## J Accuracy and Consistency Correlation

![](images/e7140d978c57c7bcc36d220c88240712e3007ea729e86c394376c2f9f4739cef.jpg)  
Figure 6: Accuracy vs. Consistency across different datasets for InternVL3.5-8B and Qwen2.5-VL-7B. Solid lines represent the base VLM performance, while dashed lines indicate reasoning performance. The x-axis represents the consistency level (1-11), showing that higher sensor consistency generally correlates with better downstream accuracy.

## K Expected Calibration Error (ECE)

Expected Calibration Error (ECE)
<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>SeedBench</td><td rowspan=1 colspan=1>MMMU-Pro</td><td rowspan=1 colspan=1>CVQA</td><td rowspan=1 colspan=1>ChartQA-Pro</td><td rowspan=1 colspan=1>ReasonVQA</td><td rowspan=1 colspan=1>FigureQA</td><td rowspan=1 colspan=1>Average</td></tr><tr><td rowspan=1 colspan=1>InternVL3.5-8B</td><td rowspan=1 colspan=1>0.11</td><td rowspan=1 colspan=1>0.13</td><td rowspan=1 colspan=1>0.16</td><td rowspan=1 colspan=1>0.12</td><td rowspan=1 colspan=1>0.08</td><td rowspan=1 colspan=1>0.07</td><td rowspan=1 colspan=1>0.11</td></tr><tr><td rowspan=1 colspan=1>Llama3.2-11B-Vision</td><td rowspan=1 colspan=1>0.10</td><td rowspan=1 colspan=1>0.12</td><td rowspan=1 colspan=1>0.10</td><td rowspan=1 colspan=1>0.09</td><td rowspan=1 colspan=1>0.04</td><td rowspan=1 colspan=1>0.14</td><td rowspan=1 colspan=1>0.10</td></tr><tr><td rowspan=1 colspan=1>Qwen2.5-VL-7B</td><td rowspan=1 colspan=1>0.09</td><td rowspan=1 colspan=1>0.11</td><td rowspan=1 colspan=1>0.08</td><td rowspan=1 colspan=1>0.09</td><td rowspan=1 colspan=1>0.04</td><td rowspan=1 colspan=1>0.11</td><td rowspan=1 colspan=1>0.09</td></tr></table>

Figure 7: Preliminary effort to select the two most distinct VLMs (InternVL3.5-8B and Qwen2.5-VL-7B) based on the largest gap in Expected Calibration Error (ECE) across diverse datasets.

![](images/c726d973ceb511e7b7891289996e243dfc18f63cf2d98169c42f22c3f4c0b994.jpg)  
Figure 8: CVQA instance: a large neoclassical building in Ulaanbaatar, Mongolia, photographed in winter. Ground truth: (C) Ulaanbaatar Department Store. (see App. B).

## Step 1

To determine the correct answer, I need to gather information about the building’s characteristics, such as its name, location, and purpose. I’ll start by asking about the building’s name to see if the VLM can provide a direct answer.

My question is: “What is the name of the building in the image?”

The building in the image appears to be a theater, characterized by its large entrance and classical architecture. Confidence: 3/5

## Step 2

The VLM provided a response that the building appears to be a theater, but this does not match any of the given options. The confidence level is moderate, so I need to ask more specific questions to clarify the building’s identity. I will focus on the building’s name and purpose to narrow down the options.

My question is: “Can you confirm the name of the building in the image, specifically if it is called Hunnu Mall, Department Store, Ulaanbaatar Department Store, or Sport Center?”

The building in the image appears to be the Ulaanbaatar Department Store. Confidence: 4/5

## Step 3

The VLM has provided a more specific and relevant answer, indicating that the building is the Ulaanbaatar Department Store with a confidence level of 4/5. This is a strong indication, but I want to ensure there are no contradictions or ambiguities. I will ask a follow-up question to confirm the building’s purpose to further validate this answer.

My question is: “Can you confirm if the building in the image is used as a department store?”

The building appears to have an ornate facade with large arches, which is characteristic of historic architecture from certain European countries. It’s less common for modern department stores to adopt such traditional architectural styles, suggesting it might serve another purpose or be a historical building, potentially not currently used as a department store. However, without more specific labels or signs visible on the building, it is difficult to confirm the exact function. Confidence: 3/5

## Step 4

The VLM’s response introduces some uncertainty about the building’s current use, despite the previous indication that it is the Ulaanbaatar Department Store. Given the moderate confidence level, I need to ask a more direct question to clarify the building’s current function and name.

My question is: “Are there any signs, labels, or other indicators in the image that specifically identify the building as the Ulaanbaatar Department Store?”

No, the image does not show any visible signage or labels that clearly identify the building as the Ulaanbaatar Department Store. Confidence: 4/5

## Step 5

The VLM’s response indicates that there are no visible signs or labels identifying the building as the Ulaanbaatar Department Store, which introduces more uncertainty. However, the previous response with a 4/5 confidence level suggested it is the Ulaanbaatar Department Store. I will ask a final clarifying question to see if there are any other distinguishing features or details that can help confirm the building’s identity.

My question is: “Can you describe any distinctive features or details of the building that might help identify it as one of the

options: Hunnu Mall, Department Store, Ulaanbaatar Department Store, or Sport Center?”

The building has a modern design with large windows, a flat roof, and decorative arches above the entryway. These features align more closely with those typically found in a shopping center like a mall or department store rather than a traditional sport center or single-use building. The presence of what appears to be advertisement screens on its facade further supports this conclusion. However, without more specific signage or interior clues, it’s challenging to definitively classify the building as one of the given options. Confidence: 4/5

## Step 6

The VLM’s response provides more details about the building’s modern design and features, which align more closely with a shopping center like a mall or department store. The presence of advertisement screens further supports this. Given the consistent indication that the building is likely a department store and the lack of evidence pointing to the other options, I am now reasonably confident that the building is the Ulaanbaatar Department Store. The correct answer is (C) Ulaanbaatar Department Store

The answer is: (C)

## M Use of AI Tools

We used AI-assisted tools (e.g., large language models) for grammatical and stylistic editing, and to help search for and organize relevant background information and literature. All technical content, experimental designs, analyses, and conclusions were created and verified by the authors, who remain responsible for any remaining errors.

## N Annotation Instruction

We recruited five expert annotators with prior experience in vision-and-language tasks. Annotators were instructed to carefully read each VISUALNOISEQA question and its answer options, and to use common sense and domain knowledge to select the single best answer. We first familiarized annotators with the task via a small pilot set of examples, discussed ambiguous cases, and provided feedback and clarifications on the guidelines before they proceeded to the final annotations. In knowledge-heavy domains, annotators were able to utilize external search engines and AI tools. In contrast, they were not allowed to access these tools in Perception and Chart domains.

Annotation protocol and inter-annotator reliability. To validate annotation quality before the full annotation effort, two annotators independently completed the first 50 questions of this benchmark spanning all five datasets. Any disagreements were discussed to clarify the guidelines, and each instance was successfully done by at least one of the two annotators. Following this validation, the remaining questions were distributed so that each question was handled by a single annotator, a design chosen deliberately for efficiency given the consistent pilot behavior. A question is retained if the annotator answered it correctly under the blind VLM interaction protocol.

O Annotation Interface  
![](images/ce0e272ff74e61c9a3053146df61ae4c3944f9d67ac001ccd57107d8f1a1a935.jpg)  
Figure 9: Annotation Interface for Human Annotators