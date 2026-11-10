# Let the Fixer Decide: Training a Code Localizer from the Fixer’s Likelihood

Khalil Hennara, Muhammad Hreden, Zeina Aldallal, Sara Chrouf, and Safwan AlModhayan

MISRAJAI

Khobar, Saudi Arabia {hennara,hreden,aldallal,chrouf,safwan}@misraj.ai

## Abstract

Code localizers are usually trained to predict the files a reference patch edits, but the code a fixer needs to read is not the same as the code it changes. We propose Fixer-Guided Localization (FGL): the localizer proposes several localizations, a frozen fixer scores each by how likely it makes a verified patch, and the localizer is trained on these scores with weighted supervised fine-tuning, without running tests and without reinforcement learning. We build the full pipeline on Gemma 4 E4B for the Kaggle Gemma-4 Developer Agent competition <sup>\*</sup>, from 1,629 test-verified Kimi K3 trajectories. The FGL target helps the fixer find the bug as often as the teacher’s localization while using 27.5% less code. On the development set, one EM round moves the agent in the expected direction compared with a matched control: more tasks reach a patch (53 vs. 46), fewer tasks get stuck reading files that do not exist (43 vs. 54), and more tasks are solved (8 vs. 6). Training also changes how the agent works: trained agents explore about three times more than the base model and write far fewer wrong patches, and the single agent solves up to 11 of 78 tasks, against 8 for the base model. An analysis of 13 evaluation runs identifies what limits a 4B agent under an 8-minute budget (time, repetition loops and edit quality) and gives practical lessons for training small coding agents. All data, code and adapters are publicly available.

## 1 Introduction

Language-model agents now fix many real GitHub issues on benchmarks such as SWE-bench [Jimenez et al., 2024; OpenAI, 2024]. Some explore and edit the repository freely with shell tools [Yang et al., 2024; Wang et al., 2025]; others first find the relevant code and then write a patch [Xia et al., 2024]. When open models are trained for the second design, the localizer learns to predict the locations that the reference patch edits [Xie et al., 2025; Tang et al., 2025].

That label is a weak stand-in for what the fixer needs. A fix often depends on code it does not change, such as a caller, a base class or a test, and a localizer with about 80% file accuracy can still give only 38% resolved issues [Al Mamun & Uddin, 2026]. In retrieval-augmented language models the answer is well known: train the retriever on what helps the reader [Lewis et al., 2020; Sachan et al., 2021; Izacard et al., 2023; Shi et al., 2024]. For issue resolution, learning from outcomes has meant reinforcement learning with test execution [Wei et al., 2025; Ma et al., 2025], which is expensive. We ask whether a localizer can learn from the fixer with supervised fine-tuning alone.

Fixer-Guided Localization (FGL) treats the localization as a hidden variable. The localizer proposes candidates, a frozen fixer scores each by the log-probability it gives to a verified patch, and the localizer is fine-tuned on the candidates weighted by these scores. This is expectation-maximization (EM) [Dempster et al., 1977]; every update is ordinary weighted fine-tuning, and the only “environment” is a forward pass of the fixer.

We built the method for the Kaggle Gemma 4 Developer Agent competition [Markowitz et al., 2026], which fixes the model, the serving stack and the tools. With a limited budget we trained only the smallest model, Gemma 4 E4B. The result is mixed, and we report it as it is: the method works ofline and moves the agent’s behaviour in the right direction, but on 78 development tasks no trained system is significantly better than the untrained model. We study 13 evaluation runs to explain why.

## Contributions.

• Method. A localizer trained from the fixer’s likelihood of a verified patch, with weighted supervised fine-tuning only <sup>†</sup>(Section 2).

• Data. A pipeline that generates and verifies teacher trajectories inside the competition’s own agent loop and sandbox <sup>‡</sup>, a pipeline that converts them into two-agent training data without inventing any tool output, and the released set of 1,629 verified trajectories (Section 3).

• Evaluation. Eight systems on Gemma 4 E4B, compared with paired tests on the same tasks, including a matched control for the EM round (Section 5).

• Analysis. What limits a small agent under a strict time budget, with practical lessons for others (Sections 5.3 and 6).

## 2 Method

## 2.1 Two agents

One frozen base model carries two LoRA adapters [Hu et al., 2022] (Figure 1). The fixer is the main agent. It calls the localizer as a tool; the localizer explores the repository in a fresh session and answers with a list of at most 8 code locations (path, symbol, line range, and whether the location is to be edited or only read). The fixer reads these locations, may read more code (a request), then edits, tests and submits.

## 2.2 Localization as a hidden variable

For a task we have the issue x, the repository R and the teacher’s patch $y ^ { \star }$ , which passed the task’s tests. The localizer gives $p _ { \phi } ( z \mid x , R )$ over localizations z, and the fixer gives $p _ { \theta } ( y \mid x , \operatorname { c t x } ( z ) )$ , where ctx(z) is the code that z points to. We never see the “right” $z ,$ so we maximize the likelihood of the verified patch over all of them:

$$
\log p ( y ^ { \star } \mid x , R ) = \log \sum _ { z } p _ { \phi } ( z \mid x , R ) p _ { \theta } \bigl ( y ^ { \star } \mid x , \operatorname { c t x } ( z ) \bigr ) .\tag{1}
$$

![](images/78e9f0b4bfd125d9020c32738fcb628f8323140f9e9b6f8575f2b8668a34f596.jpg)  
Figure 1: The two-agent structure. One frozen base model carries two LoRA adapters. The fixer calls the localizer as a tool, receives a list of locations $z ,$ reads them, may request more code, then edits, tests and submits. Both agents share the task’s tool-call budget.

A localization is judged only by how much it helps the fixer write the verified patch. The gradient for the localizer is an average under the posterior

$$
q ( z ) \propto p _ { \phi } ( z \mid x , R ) p _ { \theta } \bigl ( y ^ { \star } \mid x , \operatorname { c t x } ( z ) \bigr ) ,\tag{2}
$$

so training should raise the probability of localizations that make the patch likely and lower the others.

## 2.3 Scores and weights

We draw K candidates $z _ { 1 } , \dots , z _ { K }$ from the localizer itself. With the localizer as the proposal, the posterior weights become a softmax over the fixer’s scores. We score each candidate by

$$
s _ { k } = \frac { 1 } { | y _ { + } ^ { \star } | } \sum _ { t \in y _ { + } ^ { \star } } \log p _ { \theta } \big ( y _ { t } ^ { \star } \mid y _ { < t } ^ { \star } , x , \mathrm { c t x } ( z _ { k } ) \big ) - \lambda _ { \mathrm { c t x } } \mathrm { t o k } ( z _ { k } ) - \lambda _ { \mathrm { r e q } } | a _ { k } | ,\tag{3}
$$

and set $w _ { k } = \mathrm { s o f t m a x } _ { k } ( s _ { k } / \tau )$ . Three choices make this work in practice:

• Only the new lines. $y _ { + } ^ { \star }$ is the set of tokens on the patch’s added lines. Removed and context lines can be copied from $\operatorname { c t x } ( z )$ , so scoring them mostly rewards “the edited file is shown”.

• A cost for size and requests. Without a cost, the best localization is “show everything”. tok $\mathrm { \Psi } ( z _ { k } )$ is the size of the localized code and $\left| a _ { k } \right|$ the number of extra reads the fixer needs.

• A target efective sample size. Raw score gaps make the weights one-hot. We choose τ per task so that the efective sample size is $n ^ { \star } \colon n ^ { \star } = 2 \ ( s o f t \ \mathrm { E M } )$ keeps about two candidates; $n ^ { \star } = 1$ (hard EM) keeps the best one.

Both penalties are set to 0.25 σ, where $\sigma = 0 . 0 6 8$ nats per token is the median spread of scores within a task.

![](images/48232f2a13f572040507613838729ecd9402c69ab81fabbf18a91d61aa0b294e.jpg)  
Figure 2: One round of FGL. The localizer proposes K candidate localizations; a frozen fixer scores each by its likelihood of the verified patch $y ^ { \star }$ ; the normalized scores weight ordinary fine-tuning of the localizer. No tests run and no reinforcement learning is used.

## 2.4 Training

The localizer loss is a weighted negative log-likelihood of the candidates plus an anchor on the teacher’s own localization $\bar { z } ^ { \mathrm { t e a c h e r } }$ with weight $\beta = 0 . 5$

$$
{ \mathcal L } _ { \mathrm { l o c } } ( \phi ) = - \sum _ { i } \sum _ { k } w _ { i k } \log p _ { \phi } ( z _ { i k } \mid x _ { i } , R _ { i } ) ~ - ~ \beta \sum _ { i } \log p _ { \phi } ( z _ { i } ^ { \mathrm { t e a c h e r } } \mid x _ { i } , R _ { i } ) .\tag{4}
$$

## 1. Two folds.

We split the training tasks into two halves, fold A and fold B, and train one localizer and one fixer on each half. These fold models are used only to propose and score candidates; they are never deployed. The reason is memorization. A fixer that was trained on a task’s patch can reproduce that patch from almost any context, so every candidate would get a high score and the weights would carry no information. In the same way, a localizer trained on a task mostly repeats the teacher’s answer, so there would be little to choose from. Using the other fold’s models avoids both problems: every task is handled only by models that never saw it.

## 2. Candidates.

For each task, the localizer of the other fold proposes K = 8 localizations (temperature 0.8). We add the teacher’s localization $z ^ { \mathrm { t e a c h e r } }$ and a few simple variants (only the edited code, only the viewed code, the teacher’s answer minus one item). After removing duplicates, a task has 7.1 candidates on average.

## 3. Scores.

The fixer of the other fold reads each candidate and gives the score $s _ { k }$ of eq. (3): the average logprobability of the patch’s new lines, minus the costs for size and requests. This is one forward pass per candidate, with no gradient and no test run.

![](images/e290e3379e9435e947751a12a57a56655b81719a618cca90ce4ee7c4b6730601.jpg)  
Figure 3: Where the training signal comes from. Top: reinforcement learning runs every sampled patch against tests in a sandbox. Bottom: FGL scores each localization with one forward pass of the frozen fixer over the verified patch and trains with weighted fine-tuning.

## 4. Weights.

The scores become weights $w _ { k }$ with the per-task temperature of the previous section. For each task we keep the three candidates with the highest weight and add the teacher’s localization as an anchor with weight $\beta = 0 . 5$ , so the localizer does not drift too far from it. The weights of a task sum to one.

## 5. Fine-tuning.

The deployed localizer, trained on all tasks, is fine-tuned on these weighted candidates with the loss of eq. (4), starting from its previous version at half the learning rate. This is ordinary weighted supervised fine-tuning: there are no online rollouts and no test runs (Figure 3).

The full method also retrains the fixer on the new localizations in alternation with the localizer. In this paper we run one round and update only the localizer; every two-agent system uses the same fixer.

## 3 Data

Data was the most expensive part of the project, (Table 1).

## Tasks.

We collected about 10K tasks from SWE-bench [Jimenez et al., 2024], SWE-Gym [Pan et al., 2025], R2E- $\mathrm { G y m }$ [Jain et al., 2025] and SWE-smith [Yang et al., 2025]. Every patch was checked to apply at its base commit, and a run without any model checked that the target tests fail before the reference fix and pass after it: 3,894 of the first 4,500 ranked tasks passed. Tasks from the four repositories of the competition’s development set are excluded, so there is no overlap with the evaluation. Figure 4 and table 9 in appendix D give every count.

Table 1: Cost of the project. All training and Stage 3 work (candidate generation, scoring, signal test) ran on A100 80 GB GPUs; evaluation ran on Kaggle.
<table><tr><td>Item</td><td>Cost</td></tr><tr><td>Teacher trajectories (Kimi K3, 3,068 attempts) Recall cleaning (two passes)</td><td>$775 ≈$100</td></tr><tr><td>Data, total</td><td>≈$875</td></tr><tr><td>GPU time: all training, candidate generation, scoring and signal test ≈130 A100 hours</td><td></td></tr></table>

![](images/400bf3611778af11ad7d4eac4e2c9aed5d63594be1e69669e11b2cd368b51bf1.jpg)  
Figure 4: From collected tasks to training records (not to scale; full counts in table 9).

## Teacher trajectories.

The teacher, Kimi K3 [Moonshot AI, 2026], ran inside the competition’s own agent loop on each task’s container, with 8 minutes per attempt and no network. Git history that could reveal the fix was removed first. A fresh container then ran the task’s tests on the teacher’s patch; a trajectory counts as solved only if they pass. The code of generation and validation is released as codeagent-validate<sup>†</sup>. Of 3,068 attempts on 2,959 tasks, 1,629 tasks were solved (55%): 564 from SWE-Gym, 492 from SWE-bench, 307 from SWE-smith and 266 from R2E-Gym. 1,577 of them were kept under per-source and per-repository caps. Only verified trajectories are used for training. A solved trajectory has a median of 15 tool calls (90th percentile 33) and 13.8K tokens.

## Recall cleaning.

The teacher often “remembered” the real upstream fix: it cited pull requests and commit hashes, some invented, and sometimes wrote a fix before reading the code. A student cannot do this at test time, and training on such text teaches it to state facts it never saw. We asked the teacher not to do this, and then ran a cleaning pass that found 3,233 recall passages in 1,116 trajectories and rewrote only those reasoning passages. Tool calls, tool outputs and patches were never changed.

![](images/f25492e526214799bc708380895373200e9e10fb7530e052945a6bbce7356a29.jpg)

Figure 5: One teacher trajectory becomes two records. The trajectory is cut at the first edit. The exploration becomes the localizer record ending in z; the fixer record reads z, may request missing code using only outputs the teacher saw, then continues with the teacher’s edits unchanged.

Table 2: Training records (Gemma 4 tokenizer). “Trained” is the share of tokens with a loss.
<table><tr><td>Records</td><td>Role</td><td>Records</td><td>Tasks</td><td>Tokens p50 p90</td><td>max</td><td>Total tokens</td><td>Trained</td></tr><tr><td>s1_agent</td><td>single agent</td><td>1,386</td><td>1,384</td><td>13K</td><td>22K 32K</td><td>19.2M</td><td>23.0%</td></tr><tr><td>s2_fix_gold</td><td>fixer, teacher&#x27;s z</td><td>1,294</td><td>1,292</td><td>10K</td><td>17K 31K</td><td>15.0M</td><td>20.4%</td></tr><tr><td>s2_fix_noisy</td><td>fixer, imperfect z + requests</td><td>2,729</td><td>1,205</td><td>11K</td><td>19K 32K</td><td>33.3M</td><td>20.8%</td></tr><tr><td>s2_loc_edit</td><td>localizer, edited code</td><td>1,286</td><td>1,286</td><td>5K 11K</td><td>20K</td><td>8.0M</td><td>14.6%</td></tr><tr><td>s2_loc_view</td><td>localizer, viewed code</td><td>1,175</td><td>1,175</td><td>6K 11K</td><td>20K</td><td>7.5M</td><td>16.9%</td></tr><tr><td>s3_loc_r0</td><td>localizer, edited + viewed</td><td>1,303</td><td>1,303</td><td>6K</td><td>11K 20K</td><td>8.4M</td><td>17.1%</td></tr></table>

## From one agent to two.

The teacher works as a single agent. We cut each trajectory at its first edit (Figure 5). The exploration before the cut becomes a localizer record that ends with an answer z. The fixer record calls the localizer, reads every location of z, may request code that z misses, and then continues with the teacher’s own edits and submission, unchanged. Every requested read is answered with code the teacher actually saw, checked byte for byte; no tool output is invented. The few steps we add (calling the localizer, reading z, asking for missing code) use short fixed sentences, for example “I also need X, which the code I have read so far refers to. Let me read it.” We used fixed text on purpose: a language model writing these lines could bring recalled knowledge back.

## Exact training text.

A trajectory does not contain the exact request the model receives: tool schemas, system-prompt additions and message structure come from the agent framework. We therefore replay every trajectory through the competition’s own agent stack with a scripted model that returns the teacher’s turns, and keep the requests it sends. Our token ids match vLLM’s own rendering on all 141 test prompts. Table 2 lists the resulting 9,173 records (about 90M tokens).

## 4 Experimental Setup

## Model and training.

All adapters <sup>¶</sup> are trained on Gemma 4 E4B [Gemma Team, Google DeepMind, 2026]. The competition serves an int4 checkpoint, so we train bf16 LoRA (rank 64) on a dequantized copy and serve the adapter unmerged on the int4 model, as the scorer does; log-probabilities difer by 0.03 nats per token between training and serving. Each model is trained for about 250 optimizer steps of 16 sequences of up to 32,768 tokens, at learning rate $2 \times 1 0 ^ { - 4 }$ (half for warm starts). Settings are in appendix B.

## Systems.

• Base: the untrained model as a single agent.

• S1: one adapter trained on the full teacher trajectories.

• S2a / S2a-view: localizer and fixer trained separately; the localizer learns the edited code (S2a) or the viewed code (S2a-view).

• S2b: as S2a, with the fixer also trained on imperfect localizations and requests. This is the fixer for every Stage 3 system.

• S3r0: localizer trained on the teacher’s localization (edited and viewed code).

• S3r1-soft / S3r1-hard: S3r0 plus one EM round with $n ^ { \star } = 2 ~ \mathrm { o r } ~ 1$

• S3r1-ctrl: S3r0 plus the same extra training on the teacher’s localization only. Soft against ctrl isolates the efect of the EM targets.

## Stage 3 as run.

Stage 3 uses three kinds of localizer and fixer. (1) Fold models: we split the training tasks into two halves, A and B, and train one localizer and one fixer on each half (the fixers with the S2b recipe). (2) S3r0: a localizer trained on all tasks with the teacher’s localization (s3\_loc\_r0). (3) The S2b fixer, trained on all tasks. For every task, the localizer of the other half proposes K = 8 answers (temperature 0.8); we add the teacher’s localization and a few simple variants, which gives 7.1 diferent candidates per task on average. The fixer of the other half then scores every candidate. One of the four scoring jobs stopped early, so 1,040 of the tasks have scores; the other tasks keep only the teacher’s localization in Round 1. S3r1-soft and S3r1-hard are S3r0 fine-tuned on these weighted candidates; S3r1-ctrl is S3r0 fine-tuned on the teacher’s localization again. The fold models only propose and score; at evaluation, every S3 localizer works with the S2b fixer.

## Evaluation.

We use the 78 development tasks whose reference fix passes its tests in our Kaggle environment (of 129). Each run uses the competition harness on 4×L4 GPUs (vLLM, tensor parallel 4), 8 minutes per task, 36 tool calls for one agent and 44 for two, and 4 tasks in parallel. A task is solved when its hidden tests pass. Runs used two sampling settings: temperature 0.2-0.3 with an 8,192-token output limit, and temperature 0.7 with 6,656. We compare systems only within a setting, with the exact McNemar test [McNemar, 1947] on the same tasks.

## 5 Results

## 5.1 Ofline: the EM target helps the fixer as much, with less code

Before training on the EM targets we ran an ofline check on 1,016 training tasks. For each task, the other fold’s fixer ran live from two inputs: the teacher’s localization (T) or the highest-weight candidate (S). We then compared where it edited with the verified patch (Table 3). The two inputs are equally useful: the first edit lands at a patched location in 43.1% of attempts with S and 42.4% with T $( p = 0 . 7 4 )$ . But S is 27.5% smaller and causes 44% fewer context overflows. The fixer also prefers a candidate other than the teacher’s on 83% of tasks, and on 86% even without the size penalty; most often it prefers the localizer’s own proposals (43%) or the edited functions alone (26%). Writing the exact verified patch is too rare at this model size (1%) to compare.

## 5.2 End to end on the development set

Table 4 shows all 13 runs. Three results stand out.

The EM targets move the agent in the right direction, but the gain in solved tasks is not significant.

Under identical settings (temperature 0.7, 44 calls), S3r1-soft solves 8 tasks and the matched control S3r1-ctrl solves 6; soft solves 4 tasks that ctrl does not and ctrl 2 that soft does not $( p = 0 . 6 9 )$ . The process measures agree with the ofline test: soft reaches a patch on 53 tasks against 46, ends with no patch on 9 against 16, and has repeated “No such file” errors on 43 tasks against 54. Hard EM is not better than soft.

Table 3: Ofline signal test on 1,016 training tasks: the other fold’s fixer starts from the teacher’s localization (T) or the highest-weight candidate (S). Only-T / only-S count tasks where exactly one attempt succeeds; p is the exact McNemar test.
<table><tr><td>Measure</td><td>T</td><td>S</td><td>Only T</td><td>Only S</td><td>p</td></tr><tr><td rowspan="4">Every hunk of y* reproduced At least one hunk reproduced First edit at a patched location Any edit</td><td>1.0%</td><td>1.3%</td><td>3</td><td>6</td><td>0.51</td></tr><tr><td>5.3%</td><td>4.9%</td><td>24</td><td>20</td><td>0.65</td></tr><tr><td>42.4%</td><td>43.1%</td><td>155</td><td>162</td><td>0.74</td></tr><tr><td>55.3%</td><td>57.0%</td><td>168</td><td>185</td><td>0.39</td></tr><tr><td rowspan="2">Localization size (tokens) Context overflows</td><td>2,731</td><td>1,980</td><td>-27.5%</td><td></td><td></td></tr><tr><td>146</td><td>82</td><td>-44%</td><td></td><td></td></tr></table>

Table 4: All runs on the 78 development tasks (Gemma 4 E4B, 4×L4, 8 minutes per task). Out of budget: timeouts plus tasks that used all tool calls. Loops: tasks with at least one output that reached the output limit. src/: tasks where the agent used a /workspace/src/ path that does not exist. No file: tasks with five or more “No such file” errors. vs. base: tasks solved only by the base / only by the system, with the McNemar p.
<table><tr><td>System</td><td>Temp.</td><td>Output</td><td>Calls</td><td>Solved</td><td>Out of budget</td><td>Wrong patch</td><td>Median calls</td><td>Loops</td><td>src/</td><td>No file</td><td>vs. base (p)</td></tr><tr><td>Base</td><td>0.2</td><td>8,192</td><td>36</td><td>8</td><td>2</td><td>61</td><td>11</td><td></td><td>一</td><td></td><td></td></tr><tr><td>S1</td><td>0.2</td><td>8,192</td><td>36</td><td>10</td><td>54</td><td>14</td><td>30.5</td><td>13</td><td>6</td><td>4</td><td>2 / 4 (0.69)</td></tr><tr><td>S1</td><td>0.2</td><td>6,656</td><td>36</td><td>11</td><td>56</td><td>9</td><td>31.5</td><td>18</td><td>6</td><td>7</td><td>2 / 5 (0.45)</td></tr><tr><td>S2b</td><td>0.3</td><td>8,192</td><td>44</td><td>6</td><td>54</td><td>14</td><td>32.5</td><td>28</td><td>34</td><td>27</td><td>4 / 2 (0.69)</td></tr><tr><td>S3r1-soft</td><td>0.2</td><td>8,192</td><td>44</td><td>8</td><td>56</td><td>10</td><td>38.5</td><td>35</td><td>36</td><td>39</td><td>4 / 2 (0.69)</td></tr><tr><td>S3r1-hard</td><td>0.2</td><td>8,192</td><td>44</td><td>4</td><td>60</td><td>10</td><td>42.5</td><td>29</td><td>34</td><td>33</td><td>5 / 1 (0.22)</td></tr><tr><td>S1</td><td>0.6</td><td>6,656</td><td>32</td><td>8</td><td>24</td><td>40</td><td>32</td><td>10</td><td>6</td><td>2</td><td>5 / 5 (1.00)</td></tr><tr><td>S2a</td><td>0.7</td><td>6,656</td><td>44</td><td>7</td><td>38</td><td>27</td><td>44</td><td>12</td><td>41</td><td>38</td><td>3 / 2 (1.00)</td></tr><tr><td>S2a-view</td><td>0.7</td><td>6,656</td><td>44</td><td>4</td><td>37</td><td>25</td><td>44</td><td>13</td><td>43</td><td>44</td><td>7 / 3 (0.34)</td></tr><tr><td>S3r0</td><td>0.7</td><td>6,656</td><td>44</td><td>5</td><td>34</td><td>31</td><td>44</td><td>12</td><td>24</td><td>32</td><td>5/2 (0.45)</td></tr><tr><td>S3r1-ctrl</td><td>0.7</td><td>6,656</td><td>44</td><td>6</td><td>32</td><td>26</td><td>44</td><td>16</td><td>28</td><td>54</td><td>6 2 (0.29)</td></tr><tr><td>S3r1-soft</td><td>0.7</td><td>6,656</td><td>44</td><td>6</td><td>34</td><td>28</td><td>44</td><td>14</td><td>35</td><td>43</td><td>4 / 2 (0.69)</td></tr><tr><td>S3r1-soft</td><td>0.7</td><td>6,656</td><td>32</td><td>4</td><td>34</td><td>31</td><td>32</td><td>15</td><td>34</td><td>44</td><td>5 / 1 (0.22)</td></tr></table>

The single agent is the most robust system.

S1 solves 8, 10 and 11 tasks in three runs. Under the base model’s settings (temperature 0.2, 36 calls) it solves 10 and 11 against 8 for the base (p = 0.69 and 0.45). Every two-agent system solves 4-8 tasks.

## Diferences of a few tasks are noise.

The same S1 model solves 8 to 11 tasks across runs; even its two runs at temperature 0.2 each solve 2-3 tasks that the other does not. Across all 13 runs, only 18 diferent tasks were ever solved. None of the comparisons in table 4 reaches $p < 0 . 0 5$

## 5.3 What limits a 4B agent here

The trace of every run lets us see where the 8 minutes go. Four problems explain most failures.

## 1. Trained agents explore, then run out of time.

The base model makes a median of 11 tool calls, finishes in about 2.5 minutes and submits a guess: 61 of 78 tasks end with a patch that fails the tests. Every trained system makes about three times as many calls (30-44) and makes far fewer wrong patches (9-14 at low temperature). But 54-60 of 78 tasks then run out of budget. The teacher needed a median of 15 calls within the same 8 minutes, on a fast API. On 4×L4, with 4 tasks in parallel, the student generates about 50 tokens per second per task, so it imitates a pace it cannot keep. The scorer runs one task at a time, so our timeout counts are pessimistic, but the gap is large.

## 2. Low temperature causes repetition loops.

At temperature 0.2-0.3, 13–35 tasks per run contain an output that repeats one line until the output limit; each such output costs 130-150 seconds, more than a quarter of the budget. Two-agent systems loop about twice as often as S1 (28-35 tasks against 13-18). At temperature 0.6-0.7, loops drop to 10-16 tasks for every system. Only 16 of 222 looping outputs in the two-agent runs contain one of our fixed sentences, so the fixed text is not the main cause.

## 3. Temperature trades timeouts for wrong patches.

Moving S1 from 0.2 to 0.6 cuts the out-of-budget tasks from 54-56 to 24, but wrong patches rise from 9-14 to 40, and solved tasks do not rise. The same holds for the two-agent systems (out of budget 54-60 → 32-38; wrong patches 10–14 → 25–31). Once the agent has time, the next limit is the quality of its edits.

## 4. The trained localizers assume a code layout.

Every two-agent system uses paths under /workspace/src/ that do not exist in 24-43 tasks, for example in fastapi and rich, and gets five or more “No such file” errors in 27-54 tasks. S1 does this in 6 and 2-7 tasks. The localizers keep the assumption even after listing the repository. The S3 localizers do it less than the S2 ones at the same settings (24-35 against 41-43 tasks), which fits the ofline finding that the EM target is closer to what the fixer needs.

## 6 Discussion

Why the ofline gain does not become more solved tasks.

The signal test and the EM scores measure whether the fixer is looking at the right code. At E4B scale, most tasks fail later: the agent runs out of time, loops, or writes a wrong edit (Section 5.3). A better localizer cannot fix these. The EM round did improve what it can afect, reaching a patch and wasting fewer reads, which suggests the method is sound and the bottleneck is elsewhere for a 4B model.

## Lessons for others.

• Match the teacher’s pace to the student’s hardware. A trajectory that fits in 8 minutes on a fast API does not fit on 4×L4. Filter or shorten trajectories by the student’s expected time, not the teacher’s.

• Leave room in the context window. The harness asks for up to 16,384 output tokens with a 32K window, so any prompt above 16K tokens is rejected; in our first run this ended 4 of 5 tasks. An output limit of 6-8K avoids it. On this serving path the thinking budget was not enforced: outputs reached the full limit with a 1,536-token budget.

• Run small evaluations more than once. On 78 tasks the same model varies by three solved tasks; report paired tests and the set of solved tasks, not only the rate.

## 7 Limitations

We trained one model size (E4B) and ran one EM round that updated only the localizer. Each setting was run once per system, and the two groups of runs use diferent sampling settings, so we compare only within a group. We evaluated 4 tasks in parallel, which is slower per task than the scorer. The signal test measures where the fixer edits, not whether its patch passes. Candidate localizations were sampled after the teacher’s exploration, not after the localizer’s own.

## 8 Conclusion

We proposed Fixer-Guided Localization, which trains a code localizer on what makes a verified patch likely under the fixer, using only weighted supervised fine-tuning. Ofline, its target helps the fixer as much as the teacher’s localization with 27.5% less code; end to end on Gemma 4 E4B it moves the agent the right way (more patches, fewer wasted reads) but the gain in solved tasks is not significant. Our analysis of 13 runs shows that, for a 4B model under an 8-minute budget on 4×L4, time, repetition loops and edit quality limit the agent more than localization does. We release the 1,629 verified trajectories, the data and training code, and all trained adapters. Larger models, further EM rounds and fixer updates are left to a follow-up paper.

## References

Md Afif Al Mamun and Gias Uddin. BLAgent: Agentic RAG for file-level bug localization. ACM Transactions on Software Engineering and Methodology (TOSEM), to appear. arXiv:2605.17965, 2026.

Arthur P. Dempster, Nan M. Laird, and Donald B. Rubin. Maximum likelihood from incomplete data via the EM algorithm. Journal of the Royal Statistical Society: Series B, 39(1):1–38, 1977.

Gemma Team, Google DeepMind. Gemma 4 technical report. arXiv preprint arXiv:2607.02770, 2026.

Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations (ICLR), 2022.

Gautier Izacard, Patrick Lewis, Maria Lomeli, Lucas Hosseini, Fabio Petroni, Timo Schick, Jane Dwivedi-Yu, Armand Joulin, Sebastian Riedel, and Edouard Grave. Atlas: Few-shot learning with retrieval augmented language models. Journal of Machine Learning Research, 24(251):1–43, 2023.

Naman Jain, Jaskirat Singh, Manish Shetty, Liang Zheng, Koushik Sen, and Ion Stoica. R2E-Gym: Procedural environments and hybrid verifiers for scaling open-weights SWE agents. arXiv preprint arXiv:2504.07164, 2025.

Carlos E. Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, and Karthik Narasimhan. SWE-bench: Can language models resolve real-world GitHub issues? In International Conference on Learning Representations (ICLR), 2024. URL https://arxiv.org/abs/23 10.06770.

Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. Retrieval-augmented generation for knowledge-intensive NLP tasks. In Advances in Neural Information Processing Systems (NeurIPS), 2020.

Zexiong Ma, Chao Peng, Pengfei Gao, Xiangxin Meng, Yanzhen Zou, and Bing Xie. SoRFT: Issue resolving with subtask-oriented reinforced fine-tuning. arXiv preprint arXiv:2502.20127, 2025.

Elan Markowitz, Bryan Perozzi, Benedek Rózemberczki, Glenn Cameron, Hadi Hemmati, Yuchen Li, Michael Galkin, Majid Farhadi, Ryan Holbrook, and Ashley Oldacre. Google - the gemma 4 developer agent paper track. https://www.kaggle.com/competitions/gemma-4-developer-a gent-paper, 2026. Kaggle.

Quinn McNemar. Note on the sampling error of the diference between correlated proportions or percentages. Psychometrika, 12(2):153–157, 1947.

Moonshot AI. Kimi K3. https://github.com/MoonshotAI/Kimi-K3, 2026.

OpenAI. Introducing SWE-bench verified. https://openai.com/index/introducing-swe-bench -verified/, 2024.

Jiayi Pan, Xingyao Wang, Graham Neubig, Navdeep Jaitly, Heng Ji, Alane Suhr, and Yizhe Zhang. Training software engineering agents and verifiers with SWE-Gym. In International Conference on Machine Learning (ICML), 2025. URL https://arxiv.org/abs/2412.21139.

Devendra Singh Sachan, Siva Reddy, William Hamilton, Chris Dyer, and Dani Yogatama. Endto-end training of multi-document reader and retriever for open-domain question answering. In Advances in Neural Information Processing Systems (NeurIPS), 2021.

Weijia Shi, Sewon Min, Michihiro Yasunaga, Minjoon Seo, Rich James, Mike Lewis, Luke Zettlemoyer, and Wen-tau Yih. REPLUG: Retrieval-augmented black-box language models. In Conference of the North American Chapter of the Association for Computational Linguistics (NAACL), 2024.

Yuheng Tang, Hongwei Li, Kaijie Zhu, Michael Yang, Yangruibo Ding, and Wenbo Guo. Co-PatcheR: Collaborative software patching with component-specific small reasoning models. In Advances in Neural Information Processing Systems (NeurIPS), 2025. URL https://arxiv.or g/abs/2505.18955.

Xingyao Wang et al. OpenHands: An open platform for AI software developers as generalist agents. In International Conference on Learning Representations (ICLR), 2025. URL https: //arxiv.org/abs/2407.16741.

Yuxiang Wei, Olivier Duchenne, Jade Copet, Quentin Carbonneaux, Lingming Zhang, Daniel Fried, Gabriel Synnaeve, Rishabh Singh, and Sida I. Wang. SWE-RL: Advancing LLM reasoning via reinforcement learning on open software evolution. arXiv preprint arXiv:2502.18449, 2025.

Chunqiu Steven Xia, Yinlin Deng, Soren Dunn, and Lingming Zhang. Agentless: Demystifying LLM-based software engineering agents. arXiv preprint arXiv:2407.01489, 2024.

Chengxing Xie, Bowen Li, Chang Gao, He Du, Wai Lam, Difan Zou, and Kai Chen. SWE-Fixer: Training open-source LLMs for efective and eficient GitHub issue resolution. arXiv preprint arXiv:2501.05040, 2025.

John Yang, Carlos E. Jimenez, Alexander Wettig, Kilian Lieret, Shunyu Yao, Karthik Narasimhan, and Ofir Press. SWE-agent: Agent-computer interfaces enable automated software engineering. In Advances in Neural Information Processing Systems (NeurIPS), 2024. URL https://arxiv. org/abs/2405.15793.

John Yang, Kilian Lieret, Carlos E. Jimenez, Alexander Wettig, Kabir Khandpur, Yanzhe Zhang, Binyuan Hui, Ofir Press, Ludwig Schmidt, and Diyi Yang. SWE-smith: Scaling data for software engineering agents. arXiv preprint arXiv:2504.21798, 2025.

## A Harness Facts that Shape Training

Table 5: Behaviour of the competition harness (swegemma 0.2.11, adk\_submission 0.2.13, adk\_eval\_core 0.1.0, google-adk 1.39.1), verified from the wheels and on Kaggle, and its consequence for training.
<table><tr><td rowspan=1 colspan=2>Fact                                                       Consequence for training</td></tr><tr><td rowspan=1 colspan=1>The whole trajectory renders as one model turn;tool calls and results are special-token spans in-side it.</td><td rowspan=1 colspan=1>Training text is built from the exact messagesthe client sends, rendered with the model&#x27;s ownchat template.</td></tr><tr><td rowspan=1 colspan=1>Thinking is shown only for assistant messageswith tool calls after the last user message.</td><td rowspan=1 colspan=1>Records are cut into segments at every new usermessage.</td></tr><tr><td rowspan=1 colspan=1>Visible text on a tool-call message renders afterthe tool responses.</td><td rowspan=1 colspan=1>Teacher narration on tool-call turns is folded intothe thinking channel.</td></tr><tr><td rowspan=1 colspan=1>ADK appends &quot;You are an agent. Your internalname is ...&quot; to the system prompt.</td><td rowspan=1 colspan=1>Replay routes each request to the right agent&#x27;sscript by this line.</td></tr><tr><td rowspan=1 colspan=1>Counted tools: run_command, read_file,edit_file, write_file;    not counted:get_status, submit_patch.</td><td rowspan=1 colspan=1>Budget messages are rewritten per counted call.</td></tr><tr><td rowspan=1 colspan=1>A budget warning is appended to a tool responsewhen 10 or fewer calls remain (budget at least20).</td><td rowspan=1 colspan=1>Re-generated with the harness&#x27;s own function forthe training budget.</td></tr><tr><td rowspan=1 colspan=1>The time budget is a hard timeout; afterwardsthe current diff is graded.</td><td rowspan=1 colspan=1>A cut-off run can still score if it edited early.</td></tr><tr><td rowspan=1 colspan=1>Compaction: interval 15, overlap 2, threshold14,336 tokens, retention 5; the root agent writesthe summary.</td><td rowspan=1 colspan=1>Contexts above about 14K tokens are summa-rized at inference; no compaction inside the lo-calizer.</td></tr><tr><td rowspan=1 colspan=1>read_file: at most 150 lines and 10,000 char-acters, cut mid-line at the character cap whileend_line reports the line cap.run_command: stdout and stderr each cut at5,000 characters; 300 s timeout.</td><td rowspan=1 colspan=1>Synthetic reads are chunked so every output iscomplete.Used when rebuilding code from traces.</td></tr><tr><td rowspan=1 colspan=1>A sub-agent tool returns {&quot;result&quot;: &quot;&lt;finalnon-thought text&gt;&quot;}.</td><td rowspan=1 colspan=1>Fixer records contain exactly this wrapper.</td></tr><tr><td rowspan=1 colspan=1>skip_summarization: true ends the fixer&#x27;sturn after the localizer call and triggers a nudge.</td><td rowspan=1 colspan=1>The localizer tool uses skip_summarization:false.</td></tr><tr><td rowspan=1 colspan=1>LoRA modules: 258 on E4B (KV-shared layershave no k/v projections).</td><td rowspan=1 colspan=1>Adapters cover only the projections that exist.</td></tr><tr><td rowspan=1 colspan=1>E4B on 4×L4: 298,464 tokens of KV cache.</td><td rowspan=1 colspan=1>Several full 32K sessions fit at once.</td></tr></table>

## B Training Settings and Budget Sweeps

Table 6: Choices of budget and length, and the share of trajectories (and tokens) each option keeps.
<table><tr><td>Setting</td><td>Options measured</td><td>Chosen</td><td>Reason</td></tr><tr><td>agent</td><td>Tool calls, single 24 / 28 / 30 / 32 / 36 / 40 calls keep 78.0 / 85.0 / 86.8 / 89.2 / 92.5 / 94.8%</td><td>36</td><td>keeps the hard tasks</td><td></td></tr><tr><td>Tool calls, two agents</td><td></td><td>44</td><td>+8 for re-reading z</td><td></td></tr><tr><td>Time budget</td><td>5 or 8 minutes</td><td>8</td><td>the teacher&#x27;s own limit</td><td></td></tr><tr><td>Localizer ration</td><td>explo- 12 calls keep 86.8%, 15 keep 92.1%</td><td>15</td><td>exploration p90 is 14</td><td></td></tr><tr><td>Maximum length</td><td>trajectories / tokens kept: 24K 86.2 / 70.9%, 28K 90.8 / 78.5%, 32K 93.8 /</td><td>32,768</td><td>24K drops 29% of tokens</td><td></td></tr><tr><td>Compaction records</td><td>84.0% ries</td><td>in replay with compaction vs. full trajecto- full</td><td></td><td>chosen; see limita- tions</td></tr></table>

Table 7: Settings in use.
<table><tr><td>Setting</td><td>Value</td></tr><tr><td>Budget in records</td><td>single agent 36 calls / 8 min; two agents 44 calls / 8 min; 300 turns</td></tr><tr><td>Maximum sequence length</td><td>32,768</td></tr><tr><td>Localizer calls / locations / con- 15 / 8 / 6,000 tokens text budget</td><td></td></tr><tr><td>Noisy fixer records without re- 40%</td><td></td></tr><tr><td>quests</td><td>Candidates per task in Stage 3 8 samples at temperature 0.8 from the other fold&#x27;s localizer, plus zteacher and perturbations</td></tr><tr><td>(K) Target ESS n*</td><td>2 (soft), 1 (hard)</td></tr><tr><td> $\lambda _ { \mathrm { c t x } } , \lambda _ { \mathrm { r e q } }$ </td><td>0.25σ per median-size z and per request: 0.0097 per 1K tokens, 0.017 per request (σ = 0.068)</td></tr><tr><td>Round-1 records</td><td>top M = 3 candidates (weight ≥ 0.01) + anchor  $\beta = 0 . 5 ;$  total weight 1 per task</td></tr><tr><td>LoRA</td><td>rank 64, α 32, dropout 0; q, k, v, o, gate, up, down  $2 \times 1 0 ^ { - 4 }$  (no sweep); 0.5× for warm starts (Round 1 and</td></tr><tr><td>Peak learning rate</td><td>control)</td></tr><tr><td>Optimizer and schedule</td><td>AdamW (0.9, 0.999), no weight decay, clip 1.0; 10 warm-up steps, cosine to 10% 16 rows per optimizer step (4 GPUs × 1 row × 4 accu-</td></tr><tr><td></td><td>mulation steps), no packing, length grouping; 3 epochs for localizers and S1, 1 epoch for fixers trained on gold + noisy records and for Round 1</td></tr><tr><td>Hardware</td><td>4×A100 80 GB (training, Stage 3 generation and scoring); Kaggle for evaluation</td></tr><tr><td>Model</td><td>Gemma 4 E4B only</td></tr><tr><td>Validation split</td><td>3% of tasks</td></tr><tr><td>Seed</td><td>1234</td></tr></table>

## C Record Filters and Drop Counts

Table 8: Filters applied when building records.
<table><tr><td>Filter</td><td>Applies to</td><td>Meaning</td></tr><tr><td>over budget</td><td>both</td><td>more counted calls than the budget (two agents: lo- calizer and fixer together)</td></tr><tr><td>too long</td><td>both</td><td>single agent: any record above 32,768 tokens drops the trajectory; two agents: that record alone</td></tr><tr><td>localizer over calls</td><td>two agents</td><td>exploration longer than 15 calls</td></tr><tr><td>impossible input</td><td>two agents</td><td>z misses an edited location and no request recovers it</td></tr><tr><td>unseen code</td><td>two agents</td><td>code rebuilt from the trace, and a read needs a line the teacher never printed</td></tr><tr><td>missing code dev repository, not in solved both</td><td>two agents</td><td>no code for an edited file</td></tr><tr><td>set, replay error</td><td></td><td>development-set repository; dropped by the cleaning audit; trace cannot be loaded or replayed</td></tr></table>

Two-agent drops, counted per variant (a task can appear several times): localizer exploration over 15 calls, 51-54 per variant; over the 44-call shared budget, 15-35 per variant; view-only z missing an edited location with no way to recover it, 80; code a read needs but the teacher never printed, 1-2 per variant; fixer records over 32K tokens, 11 gold and 11 noisy. Tasks without code for an edited file: 3; without any edit: 3; with the first edit mixed into an exploration turn: 1.

## D Data Funnel

Table 9: From collected tasks to training records.
<table><tr><td>Step</td><td>Count</td></tr><tr><td>Tasks collected from seven datasets</td><td>≈10K</td></tr><tr><td>Passed ingest checks (patches apply at the base commit)</td><td>9,966</td></tr><tr><td>Ranked candidates under source and repository caps</td><td>7,772</td></tr><tr><td>Valid among the first 4,500 ranked (tests fail before, pass after the gold patch)</td><td>3,894</td></tr><tr><td>Tasks attempted by the teacher (attempts)</td><td>2,959 (3,068)</td></tr><tr><td>Tasks solved</td><td>1,629 (55%)</td></tr><tr><td>Solved tasks accepted under the caps</td><td>1,577</td></tr><tr><td>Trajectories with single-agent records</td><td>1,384</td></tr><tr><td>Tasks with two-agent records</td><td>≈1,300</td></tr><tr><td>Unsolved attempts kept for Stage 4 (with code)</td><td>1,338 (1,310)</td></tr></table>

## E Issues Found During Development

Table 10: Problems found while building the pipeline, their efect if missed, and the fix.

<table><tr><td>Problem</td><td>Effect if missed</td><td>Fix</td></tr><tr><td>records keep up to 15</td><td>Localizer prompt said 12 calls; training on examples that break their own instruction</td><td>prompt says 15; a test ties prompt numbers to the configuration; records rebuilt</td></tr><tr><td>solved tasks</td><td>No base-commit code for 622 of 1,384 tasks lost from two-agent data</td><td>fetch needed files with plain git, checked against the trace</td></tr><tr><td>SWE-smith commits not served by GitHub (300 tasks)</td><td>no code for SWE-smith tasks</td><td>instance branch of the SWE-smith mirror</td></tr><tr><td>LoRA for Gemma 4&#x27;s model from the scorer class</td><td>Public vLLM 0.19.1 lacks parity or evaluation differs</td><td>install the competition wheel; build check</td></tr><tr><td>position; rotary buffer order with wrong positions depends on the hash seed</td><td></td><td>DDP broadcasts buffers by all GPUs but the first train sort buffers by name before wrap- ping; regression test</td></tr><tr><td>replay with a server error</td><td>cause hidden</td><td>40 single-agent traces failed about 3% of Stage 1 data lost; replay re-raises the underlying error; cause open</td></tr><tr><td>12 audit-dropped traces still produced records</td><td>training on rejected trajecto- ries</td><td>builder skips traces not in the solved set</td></tr><tr><td>5-minute budget replayed on the teacher&#x27;s 8-minute clock</td><td>teaches the model to ignore</td><td>time scale = budget / teacher min-</td></tr><tr><td>skip_summarization: true on the localizer tool</td><td>the clock broken two-agent loop at in-</td><td>utes (1.0 at 8 minutes) set to false; a test checks one user</td></tr><tr><td>read_file cuts mid-line at 10,000 characters</td><td>ference synthetic reads that differ</td><td>message per fixer record reads chunked to 150 lines and 10,000</td></tr><tr><td>End-of-epoch validation as- sumed one validation file</td><td>from the harness</td><td>characters runs with two (S2b and fold per-file sources in validation; the fixers) stopped after saving saved adapter exported afterwards</td></tr><tr><td>memory for prompt log- long scoring sequence</td><td>the adapter, before export</td><td>vLLM does not reserve out of memory on the first scoring at 55% GPU memory utiliza- tion</td></tr><tr><td>probabilities Offline rollouts without com- paction</td><td>server rejects prompts above such attempts counted as stopped,</td><td></td></tr><tr><td>One scoring shard stopped</td><td>about 16K tokens; more often not excluded for larger z 1,040 instead of about 1,240</td><td>unscored tasks keep zteacher in</td></tr><tr><td>early Templatedrops earlier</td><td>tasks with weights training text differs from infer-</td><td>Round 1 segments at user messages; narration</td></tr><tr><td>thoughts after a user message Output limit of 16,384 tokens</td><td>ence text every prompt above 16K to-</td><td>folded into thinking output limit of 6,656–8,192 tokens</td></tr><tr><td>with a 32,768-token window</td><td>kens is rejected; 4 of the first 5 development tasks ended this way</td><td></td></tr><tr><td>Low sampling temperature (0.2–0.3)</td><td>repetition loops in 13–35 of 78 tasks, each costing 130–150 s</td><td>temperature 0.6–0.7</td></tr></table>

## F Reproduction

Every step runs through one command-line tool; each command takes a configuration file and overrides, and writes its resolved configuration, git commit and package versions next to its output. The code is split into two repositories:

• Data (https://github.com/khalil-Hennara/codeagent-validate): task ingest and validation, teacher runs and grading, recall cleaning, trace statistics and budget sweeps, base-commit code fetch, and single-agent and two-agent record building in parallel shards, with merge, schema check and the task-level validation split.

• Training (https://github.com/misraj-ai/locfix): dequantization once per model; a GPU smoke test (dequantize, parity, overfit, export, parity, generation); a dry-run training plan; training; and a parity check of the exported adapter before any evaluation. Stage 3 adds the fold split (em.fold-split), candidate generation (em.candidates), scoring sequences (score.prepare), scoring (score.run), weights (score.weights), the ofline signal test (em.signal-test) and the Round-1 records (em.r1-records); submit.package builds and checks the submissions.

Versions: vLLM 0.19.1 (competition build), transformers 5.13.1, peft 0.21.2, accelerate 1.15.0, compressed-tensors 0.15.0.1, google-adk 1.39.1, litellm 1.106.3, Python 3.12 (training).