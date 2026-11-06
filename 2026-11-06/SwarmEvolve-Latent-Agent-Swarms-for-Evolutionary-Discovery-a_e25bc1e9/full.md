# SwarmEvolve: Latent Agent Swarms for Evolutionary Discovery and Optimization

Qile Jiang<sup>1</sup> George Em Karniadakis<sup>1</sup>

<sup>1</sup>DIVISION OF APPLIED MATHEMATICS, BROWN UNIVERSITY

## Abstract

LLM-guided evolutionary search uses AI agents to improve programs for mathematical and computational problems by iteratively refining solutions and combining high-scoring methods. However, the process is often slow because the LLM generates a long textual chain of thought for each round and must repeat this generation many times to maintain population diversity. We present SwarmEvolve, a framework in which hundreds of low-cost agents per round work in swarms, reason in latent space without decoding to text, and share information by passing KV caches. Each swarm studies the problem from a different angle. A novelty filter keeps the most distinct agent ideas by directly measuring the distances between the agents’ hidden state vectors. Crossover of ideas happens naturally in latent space, as a summarizer agent receives the combined KV caches of all swarm members. On 21 problems in mathematical discovery and scientific computing, SwarmEvolve gets the highest average score and needs a median of only 3 to 8 minutes to reach the best scores that other state-of-the-art frameworks reach in 60 minutes. On several problems, SwarmEvolve with Qwen3 8B improves on the solutions that AlphaEvolve first found with frontier LLMs. Our work shows that agent swarms that communicate in latent space offer a fast, low-cost approach to program search for scientific problems.

<sup>§</sup> github.com/Qile-J/SwarmEvolve  
Mean best score at 15 minutes  
![](images/e7250eb5206db8c46a1e15e476936ce1c203cf0ff7770edf787e098afc153cd6.jpg)  
Figure 1: SwarmEvolve (ours) compared with OpenEvolve, ShinkaEvolve, and CodeEvolve on 21 problems, with Qwen3 8B on one GPU for 60 minutes per run. Left: total number of programs generated. Middle: mean best score at 15 minutes. Right: median minutes to reach OpenEvolve’s 60-minute score. SwarmEvolve has a higher throughput and better scores.

## 1 Introduction

Many problems in mathematics and scientific computing can be framed as a search for a good program. Examples include symbolic regression, optimizing constants in inequalities, and building robust solvers for differential equations. This search can be carried out through evolutionary computation, a method to iteratively improve candidate programs through variation, crossover, and selection [32, 39].

Recently, LLMs have been used as the mutation and crossover operators in such evolutionary processes [43, 49, 73]. These LLM-guided frameworks have made strong progress on many open problems. In particular, FunSearch found new cap set constructions [54], and AlphaEvolve found a more efficient algorithm for multiplying 4 × 4 complex matrices [50]. Later work applied this approach to solve many more problems in mathematics and computer systems [24, 20, 12]. These frameworks differ mainly in how they explore. FunSearch evolves programs on separate islands, and AlphaEvolve and its open-source variants add a MAP-Elites archive to keep diverse programs [59, 1, 37]. To save LLM calls, some methods reject near duplicate proposals, adaptively allocate the search budget, or evolve the search strategy itself [42, 10, 46]. Others guide exploration with better context, such as how past changes affected the score, related literature, simulated peer review, or a solver split into separate steps [34, 45, 78, 40].

Recent work also scales up the search. Some methods run many parallel trajectories [77] or train the LLM on each problem with reinforcement learning [79, 64, 70]. Others use more agents with different roles, such as an ensemble of frontier LLMs in AlphaEvolve, multi-agent debate, and coding agents that coordinate through shared memory or an orchestrator [35, 52, 69]. When agents build on one another’s findings, they have produced new results on open problems [9, 7]. These works suggest that diverse proposals, crosspollination of ideas, and iterative refinement are key to new discoveries on open scientific problems.

Yet in most of these frameworks, the search is slow and expensive. Each agent writes a long chain of thought in text. The growing solution archive, evaluator feedback, and text messages between agents make the LLM context very long. Scaling to many agents, larger models, or test-time training multiplies this cost. As a result, the search cannot run at the scale that many open problems need. We therefore ask:

##  <sub>Can</sub> <sub>swarms</sub> <sub>of</sub> <sub>low-cost</sub> <sub>agents</sub> <sub>make</sub> <sub>evolutionary</sub> <sub>search</sub> <sub>faster</sub> <sub>and</sub> <sub>better?</sub>

We present SwarmEvolve, an evolutionary search framework in which swarms of low-cost latent agents reason in latent space and communicate by passing KV caches. It builds on recent work on latent reasoning, where an LLM reasons with continuous vectors instead of sampled tokens [29, 80, 60, 74, 82], and on latent communication, where LLMs exchange token embeddings [51], activations [53], hidden states [81, 19, 84], or KV caches [61, 23] instead of text messages. In particular, LatentMAS combines both without training. Its agents reason in latent space and pass their KV caches to a summarizer that writes the answer [85]. We use this mechanism as the basis for large-scale, low-cost evolutionary search

Because latent reasoning and communication are cheap, SwarmEvolve runs hundreds of agents in each round of evolution. The agents form distinct swarms, and each swarm has its own focus, such as implementing a construction from the literature or using a textbook method. Inside a swarm, each agent reasons from its unique perspective for 40 latent steps. A novelty filter then keeps the agents in each swarm whose latent reasoning trajectories are farthest apart in cosine distance. Finally, each swarm’s summarizer agent reads the KV caches of the kept agents and writes one new program.

Compared with existing frameworks, SwarmEvolve has three advantages:

1. Greater diversity. While existing methods generate one response at a time or run a small team of agents, SwarmEvolve runs 450 latent agents per round, each with a different specialization, parent program, and way of thinking, and so samples far more diverse proposals.

2. Built-in crossover. SwarmEvolve recombines ideas directly in latent space. A summarizer agent reads the KV caches of agents that studied different parent programs and writes one program that combines their ideas.

3. Higher throughput. Since agents do not decode or encode text between steps, SwarmEvolve evaluates 5 to 13 times more programs per hour than existing SOTA frameworks and reaches their 60- minute scores in a median of 3 to 8 minutes.

We compare SwarmEvolve with OpenEvolve, ShinkaEvolve, and CodeEvolve on 21 problems from mathematical discovery and scientific computing. Each framework runs with Qwen3 8B, 14B, and 32B-FP8 [75] for 60 minutes on one GPU (Figure 1). SwarmEvolve gets the highest average score with all three models. With Qwen3 8B, it scores higher than all three baselines with Qwen3 32B-FP8. After only 15 minutes, it already scores higher than the 60-minute averages of all baselines with every model. On three problems, SwarmEvolve with a frozen Qwen3 8B also improves on the constructions that AlphaEvolve originally found with frontier Gemini models. These results suggest that swarms of low-cost latent agents are a fast and scalable way to use AI for open mathematical and scientific discovery.

## 2 Preliminaries

In this section, we introduce the preliminary setup and notations for our method.

## 2.1 LLMs and the KV cache

We use decoder-only transformer LLMs [68], which embed the input token at position t as a vector $\mathbf { e } _ { t } \in \mathbb { R } ^ { d } .$ where d is the hidden dimension, and pass the sequence through L transformer layers. The output of the last layer at position t is the last hidden state h $\mathbf { \epsilon } \in \bar { \mathbb { R } ^ { d } }$ . The LLM computes the distribution of the next token from $\mathbf { h } _ { t }$

Inside each layer, the query $\mathbf { q } _ { t }$ of position t attends to the keys $\mathbf { k } _ { j }$ and values $\mathbf { v } _ { j }$ of all positions $j \leq t .$ . Because the keys and values of a position depend only on the tokens up to that position, the LLM stores them in a KV cache $\{ ( \mathbf { K } ^ { ( \ell ) } , \mathbf { V } ^ { ( \ell ) } ) \} _ { \ell = 1 } ^ { L ^ { - } } ,$ where $\bar { \mathbf { K } } ^ { ( \ell ) }$ and $\mathbf { V } ^ { ( \ell ) }$ stack the keys and values of all positions in layer ℓ. Each new token then costs one forward pass of a single position, which attends to the KV cache and appends its own keys and values to it. The KV cache of a shared prefix, such as a problem statement, is usually computed once and reused by every sequence that continues it for efficiency.

The Qwen3 models that we use encode positions with rotary position embeddings (RoPE) [63, 75]. RoPE multiplies each query and key at position i by a rotation matrix $\mathbf { R } _ { i }$ whose angles grow linearly with i, so the attention score depends only on the offset between the two positions:

$$
\left( \mathbf { R } _ { i } \mathbf { q } \right) ^ { \top } ( \mathbf { R } _ { j } \mathbf { k } ) = \mathbf { q } ^ { \top } \mathbf { R } _ { j - i } \mathbf { k } .\tag{1}
$$

Moving a block of the KV cache from positions 0, 1, 2, . . . to positions $\delta , \delta + 1 , \delta + 2 , . .$ . therefore requires multiplying each cached key by $\mathbf { R } _ { \delta } ,$ since the values carry no positional information.

## 2.2 Latent reasoning

LLMs usually reason by decoding a chain of thought in text $\left[ 7 1 , 3 8 \right]$ . In latent reasoning, the LLM instead feeds the last hidden state $\mathbf { h } _ { t }$ back as the next input embedding $\mathbf { e } _ { t + 1 }$ , so it reasons in its continuous embed ding space without sampling tokens [29, 80, 83].

LatentMAS applies latent reasoning to LLM agents without training [85]. After an agent reads its input, each latent reasoning step maps the last hidden state h<sub>t</sub> directly to the next input embedding $\mathbf { e } _ { t + 1 }$ . This map has two parts. First, an alignment matrix ${ \bf W } _ { \sf a }$ maps $\mathbf { h } _ { t }$ into the input embedding space. Second, the result is rescaled to norm $\gamma ,$ where $\gamma$ is the mean norm of the input token embeddings over the vocabulary. The matrix ${ \bf W } _ { \mathrm { a } }$ is the ridge regression solution, with regularization weight $\rho ,$ that maps the output embedding of each token to its input embedding:

$$
\mathbf { e } _ { t + 1 } = \gamma \frac { \mathbf { W } _ { \mathrm { a } } \mathbf { h } _ { t } } { \left\| \mathbf { W } _ { \mathrm { a } } \mathbf { h } _ { t } \right\| } , \qquad \mathbf { W } _ { \mathrm { a } } = \mathbf { W } _ { \mathrm { i n } } ^ { \top } \mathbf { W } _ { \mathrm { o u t } } \big ( \mathbf { W } _ { \mathrm { o u t } } ^ { \top } \mathbf { W } _ { \mathrm { o u t } } + \rho \mathbf { I } \big ) ^ { - 1 } ,\tag{2}
$$

where $\boldsymbol { W } _ { \mathrm { i n } } , \boldsymbol { W } _ { \mathrm { o u t } } \in \mathbb { R } ^ { | \mathcal { V } | \times d }$ are respectively the input and output embedding matrices and $| \nu |$ is the vocabulary size. The agent takes a fixed number of latent reasoning steps given by Equation 2 and decodes no text. Each latent reasoning step adds one position to the KV cache.

![](images/322db5b79c96ba6620fae43bdbe441f0e3e6065ca254959fba48e71beaf697b9.jpg)

Figure 2: Workflow of SwarmEvolve. In each round, SwarmEvolve launches 10 swarms of 45 latent agents. Each swarm draws parent programs from the archive and gets a focus prompt to refine, hybridize, explore, or start fresh. In each swarm, the agents take 40 latent reasoning steps, a novelty filter removes the 15 most similar agents (hollow circles), and the summarizer reads the KV caches of the rest to write one new program. The new programs are evaluated and added to the archive. A parameter tuning agent tunes the hyperparameters of the best program.  
![](images/2394ee5171283c0d36e26f0d006d6044f455ea064f0fe25b83c8de114052eb42.jpg)  
Figure 3: Latent reasoning and communication in a swarm. Each latent agent reasons recursively in the embedding space without decoding to text. Its latent reasoning steps are stored as KV cache. The summarizer then reads the concatenated KV caches from all agents and writes a new program in text.

## 2.3 Latent communication

LatentMAS agents communicate by passing KV caches instead of text [85]. In the standard hierarchical setting, J agents, usually with different specializations, each take their own latent reasoning steps. A summarizer agent then writes the final answer. Let ${ \bf K } _ { j } ^ { ( \ell ) }$ and $\mathbf { V } _ { j } ^ { ( \ell ) }$ be the keys and values of agent j in layer $\ell ,$ for $j = 1 , \dots , J $ . Let ${ \bf K } _ { \mathrm { s u m } } ^ { ( \ell ) }$ and $\mathbf { V } _ { \mathrm { s u m } } ^ { ( \ell ) }$ be the keys and values of the summarizer’s own prompt in layer ℓ. The KV cache of the summarizer concatenates them along the sequence dimension:

$$
{ \bf K } ^ { ( \ell ) } = [ { \bf K } _ { 1 } ^ { ( \ell ) } ; \ldots ; { \bf K } _ { J } ^ { ( \ell ) } ; { \bf K } _ { \mathrm { s u m } } ^ { ( \ell ) } ] , \qquad { \bf V } ^ { ( \ell ) } = [ { \bf V } _ { 1 } ^ { ( \ell ) } ; \ldots ; { \bf V } _ { J } ^ { ( \ell ) } ; { \bf V } _ { \mathrm { s u m } } ^ { ( \ell ) } ] \qquad \mathrm { f o r e a c h l a y e r } \ \ell .\tag{3}
$$

The summarizer then decodes the answer as text, attending to both its own inputs and the latent reasoning steps of all J agents. When the agents run in parallel, the KV cache of each agent starts at position 0. We therefore use the rotation in Equation 1 to move each cache to its offset in the concatenation.

## 3 SwarmEvolve

SwarmEvolve is an evolutionary search framework that runs latent-reasoning agent swarms to improve programs over many rounds. It keeps an archive of all candidate programs, together with their scores and evaluation feedback. In each round, it selects parent programs from the archive for refinement and crossover. The search requires an initial program, which is usually a simple baseline solution. As illustrated in Figure 2, each round consists of the following steps.

Form swarms. To start a round, SwarmEvolve forms 10 swarms of 45 latent agents each. All agents in a swarm share one main focus. Each agent receives a distinct prompt, which combines one of five phrasings of this focus with one of nine ways of thinking, such as reasoning from first principles or reconsidering assumptions (Supplementary Table 4). Each swarm also receives a set of parent programs, which are assigned to its agents in turn so that each agent reads exactly one program. The 10 swarms are of four types:

• Four refinement swarms each receive one of the two best programs in the archive, with two swarms per program. Each swarm takes the main focus used least often so far for its program, such as its mathematical formulation or its use of the time budget. Their goal is to improve the best programs.

• Three hybridization swarms each receive five programs drawn at random from the ten best, so nine agents read each program. Their goal is to combine the strongest ideas into one new program.

• Two exploration swarms each receive four programs drawn at random from the entire archive. Nine agents read each program, and the other nine agents read only the problem statement with no parent program. Their goal is to explore less considered directions.

• One fresh-start swarm receives only the initial program. Its goal is to find a new approach that differs from existing ones.

Reason in latent space. Each latent agent reads the problem statement, its focus prompt, and its assigned parent program with the program’s evaluation results. It then takes 40 latent reasoning steps. All 450 agents run in one batch. Agents that read the same program attend to one shared copy of its KV cache through paged attention [41].

Filter for novelty. In each swarm, the novelty filter keeps the 30 latent agents whose reasoning differs the most and discards the other 15. Let $\mathbf { h } _ { j , t }$ be the last hidden state of agent j at latent reasoning step t. Each agent is represented by its mean hidden state over the 40 steps,

$$
\bar { \mathbf { h } } _ { j } = \frac { 1 } { 4 0 } \sum _ { t = 1 } ^ { 4 0 } \mathbf { h } _ { j , t } , \qquad j = 1 , \ldots , 4 5 .\tag{4}
$$

Then, each agent’s mean hidden state $\bar { \mathbf { h } } _ { j }$ is centered on the swarm mean µ and normalized to a unit vector $\mathbf z _ { j } = ( \bar { \mathbf h } _ { j } - \pmb { \mu } ) / \| \bar { \mathbf h } _ { j } - \pmb { \mu } \|$ . The kept set K starts with the agent closest to µ. Greedy farthest-point selection [26] in cosine distance then adds one agent at a time, selecting

$$
j ^ { \star } = \underset { j \notin { \cal K } } { \arg \operatorname* { m a x } } \underset { i \in { \cal K } } { \operatorname* { m i n } } \left( 1 - { \mathbf { z } } _ { j } ^ { \top } { \mathbf { z } } _ { i } \right) ,\tag{5}
$$

until K holds 30 agents.

Transfer KV caches. Each summarizer reads the reasoning of its 30 kept agents through one KV cache (Figure 3). In each layer ℓ, this KV cache is the concatenation

$$
\begin{array} { r } { { \bf K } ^ { ( \ell ) } = [ { \bf K } _ { \mathrm { p r o b } } ^ { ( \ell ) } ; { \bf K } _ { 1 } ^ { ( \ell ) } ; \ldots ; { \bf K } _ { 3 0 } ^ { ( \ell ) } ; { \bf K } _ { p _ { 1 } } ^ { ( \ell ) } ; \ldots ; { \bf K } _ { p _ { Q } } ^ { ( \ell ) } ; { \bf K } _ { \mathrm { i n s } } ^ { ( \ell ) } ] , } \\ { { \bf V } ^ { ( \ell ) } = [ { \bf V } _ { \mathrm { p r o b } } ^ { ( \ell ) } ; { \bf V } _ { 1 } ^ { ( \ell ) } ; \ldots ; { \bf V } _ { 3 0 } ^ { ( \ell ) } ; { \bf V } _ { p _ { 1 } } ^ { ( \ell ) } ; \ldots ; { \bf V } _ { p _ { Q } } ^ { ( \ell ) } ; { \bf V } _ { \mathrm { i n s } } ^ { ( \ell ) } ] , } \end{array}\tag{6}
$$

where $\mathbf { K } _ { \mathrm { p r o b } } ^ { ( \ell ) }$ and $\mathbf { V } _ { \mathrm { p r o b } } ^ { ( \ell ) }$ are the keys and values of the problem statement. For $j = 1 , \ldots , 3 0 , \mathbf { K } _ { j } ^ { ( \ell ) }$ and $\mathbf { V } _ { j } ^ { ( \ell ) }$ are the keys and values of the focus prompt and 40 latent reasoning steps of the j-th kept agent. For $\dot { q } = 1 , \dots , Q , \mathbf { K } _ { p _ { q } } ^ { ( \ell ) }$ and $\mathbf { V } _ { p _ { q } } ^ { ( \ell ) }$ are the keys and values of the $q \mathrm { . }$ th parent program $p _ { q }$ of the swarm. ${ \bf K } _ { \mathrm { i n s } } ^ { ( \ell ) }$ and $\mathbf { V } _ { \mathrm { i n s } } ^ { ( \ell ) }$ are the keys and values of the summarizer instruction. Before the concatenation, each latent agent’s key k is rotated to $\mathbf { R } _ { \delta } \mathbf { k }$ (Equation 1), where δ is the position of its block in $\mathbf { K } ^ { \left( \ell \right) }$ . The values carry no positional information and are concatenated unchanged.

Write programs. After receiving the KV cache, the summarizer synthesizes and decodes the latent reasoning traces of its agents. The summarizer outputs a short description of its direction and one complete program required to solve the problem.

Tune parameters. Every round, a parameter tuning agent reads the best program, its evaluation results, and a list of all numeric parameters of the program. It then proposes a single new value for one of the parameters to improve the program. SwarmEvolve evaluates up to 8 such one-number edits per round.

Evaluate. After new programs are generated, the framework discards programs that duplicate an earlier one, checks the syntax, and automatically adds missing imports of standard package names. The deterministic evaluator then scores each new program on a CPU core. When the evaluator finishes, the scored programs enter the archive with their scores, logs, and error messages.

SwarmEvolve iterates these steps for the time budget of the run. The best-scoring program in the archive at the end of the run is returned as the final solution.

## 4 Experiments

We compare SwarmEvolve with three state-of-the-art frameworks: OpenEvolve [59], ShinkaEvolve [42], and CodeEvolve [1] on 21 problems from five domains (Appendix D). Sixteen of these problems, in packing, geometry, analysis, and combinatorics, come from Appendix B of AlphaEvolve [50]. For these problems, we use the same problem statements, initial programs, and evaluators from OpenEvolve. The score of each problem is its objective value divided by that of the construction published by AlphaEvolve, so a score above 1 improves on AlphaEvolve.

We also design five scientific computing problems on sparse linear systems, eigenvalues of non-normal matrices, and the Burgers, convection-diffusion, and Helmholtz equations, whose score is the number of correct digits relative to a high-fidelity ground truth, averaged over challenging test cases with various parameters or matrices. Problem details are given in Appendix D.

Each framework is given 60 minutes on one GPU with 8 CPU cores that evaluate 8 programs in parallel. The three baselines keep their default configurations, let the model reason in text before it answers, and serve it with vLLM [41] for fast inference. Every framework runs three seeds on every problem with each of Qwen3 8B, 14B, and 32B-FP8.

## 4.1 Results

As shown in Table 1, SwarmEvolve achieves the highest average score with all three LLMs. It also achieves the best score on most individual problems. SwarmEvolve’s average score with Qwen3 8B outperforms all three baselines running Qwen3 32B-FP8. With Qwen3 32B-FP8, SwarmEvolve solves all five sparse linear systems to full accuracy, which no baseline run ever achieves. In addition to higher scores, SwarmEvolve is also superior in the following ways:

Table 1: Best score within 60 minutes, averaged over three seeds (scores defined in Section 4 and Appendix D). For each model, bold marks the best score and underline marks the second best. SwarmEvolve (ours) has the highest average score with every model.
<table><tr><td></td><td></td><td colspan="4">Qwen3 8B @ 60 min 1× L40S GPU, 8 CPU cores</td><td colspan="4">Qwen3 14B @ 60 min 1× RTX PRO 6000 GPU, 8 CPU cores</td><td colspan="4">Qwen3 32B-FP8 @ 60 min 1× H100 GPU, 8 CPU cores</td></tr><tr><td>Group</td><td>Problem</td><td>Ours</td><td>Open- Evolve</td><td>Shinka- Evolve</td><td>Code- Evolve</td><td>Ours</td><td>Open- Evolve</td><td>Shinka- Evolve</td><td>Code- Evolve</td><td>Ours</td><td>Open- Evolve</td><td>Shinka- Evolve</td><td>Code- Evolve</td></tr><tr><td rowspan="4">Packing</td><td>Circles in a square</td><td>0.998</td><td>0.877</td><td>0.876</td><td>0.990</td><td>0.999</td><td>0.868</td><td>0.996</td><td>0.991</td><td>0.997</td><td>0.915</td><td>0.987</td><td>0.992</td></tr><tr><td>Circles in a rectangle</td><td>0.888</td><td>0.969</td><td>0.958</td><td>0.987</td><td>0.976</td><td>0.956</td><td>0.954</td><td>0.992</td><td>1.000</td><td>0.960</td><td>0.954</td><td>0.984</td></tr><tr><td>Hexagons, n = 11</td><td>0.627</td><td>0.546</td><td>0.491</td><td>0.491</td><td>0.771</td><td>0.688</td><td>0.545</td><td>0.540</td><td>0.874</td><td>0.780</td><td>0.697</td><td>0.491</td></tr><tr><td>Hexagons, n = 12</td><td>0.707</td><td>0.593</td><td>0.546</td><td>0.493</td><td>0.829</td><td>0.721</td><td>0.648</td><td>0.595</td><td>0.650</td><td>0.786</td><td>0.521</td><td>0.493</td></tr><tr><td rowspan="5">Geometry</td><td>Heilbronn, triangle</td><td>0.558</td><td>0.050</td><td>0.418</td><td>0.514</td><td>0.710</td><td>0.152</td><td>0.172</td><td>0.591</td><td>0.831</td><td>0.058</td><td>0.473</td><td>0.513</td></tr><tr><td>Heilbronn, convex, n = 13</td><td>0.724</td><td>0.601</td><td>0.656</td><td>0.610</td><td>0.739</td><td>0.617</td><td>0.660</td><td>0.641</td><td>0.878</td><td>0.570</td><td>0.704</td><td>0.679</td></tr><tr><td>Heilbronn, convex, n = 14</td><td>0.807</td><td>0.421</td><td>0.342</td><td>0.487</td><td>0.769</td><td>0.638</td><td>0.659</td><td>0.555</td><td>0.780</td><td>0.654</td><td>0.692</td><td>0.638</td></tr><tr><td>Distance ratio, plane</td><td>0.994</td><td>0.716</td><td>0.876</td><td>0.887</td><td>1.000</td><td>0.716</td><td>0.907</td><td>0.842</td><td>0.999</td><td>0.716</td><td>0.848</td><td>0.735</td></tr><tr><td>Distance ratio, space</td><td>0.994</td><td>0.817</td><td>0.890</td><td>0.913</td><td>0.999</td><td>0.880</td><td>0.903</td><td>0.924</td><td>0.999</td><td>0.897</td><td>0.915</td><td>0.887</td></tr><tr><td rowspan="4">Analysis</td><td>Autocorrelation I</td><td>0.995</td><td>0.993</td><td>0.992</td><td>0.992</td><td>0.994</td><td>0.993</td><td>0.993</td><td>0.992</td><td>0.995</td><td>0.993</td><td>0.993</td><td>0.992</td></tr><tr><td>Autocorrelation II</td><td>1.021</td><td>1.006</td><td>0.994</td><td>1.003</td><td>1.032</td><td>1.013</td><td>1.004</td><td>0.994</td><td>1.016</td><td>0.985</td><td>0.992</td><td>0.985</td></tr><tr><td>Autocorrelation III</td><td>1.005</td><td>0.779</td><td>0.979</td><td>0.991</td><td>1.005</td><td>0.988</td><td>0.998</td><td>0.988</td><td>1.005</td><td>0.998</td><td>0.997</td><td>0.991</td></tr><tr><td>Uncertainty</td><td>0.911</td><td>0.899</td><td>0.000</td><td>0.780</td><td>0.912</td><td>0.899</td><td>0.000</td><td>0.889</td><td>0.906</td><td>0.911</td><td>0.000</td><td>0.895</td></tr><tr><td rowspan="3"></td><td>Erdős overlap</td><td>0.999</td><td>0.998</td><td>0.997</td><td>0.961</td><td>1.000</td><td>1.000</td><td>0.999</td><td>0.997</td><td>0.999</td><td>0.999</td><td>0.999</td><td>0.998</td></tr><tr><td>Combinatorics Sums and differences</td><td>0.957</td><td>0.939</td><td>0.000</td><td>0.923</td><td>0.958</td><td>0.929</td><td>0.299</td><td>0.914</td><td>0.964</td><td>0.932</td><td>0.580</td><td>0.945</td></tr><tr><td>Kissing number, d = 11</td><td>0.309</td><td>0.481</td><td>0.508</td><td>0.371</td><td>0.570</td><td>0.782</td><td>0.617</td><td>0.371</td><td>0.709</td><td>0.753</td><td>0.615</td><td>0.319</td></tr><tr><td rowspan="5">Scientific computing</td><td>Sparse linear systems</td><td>0.933</td><td>0.648</td><td>0.695</td><td>0.420</td><td>0.867</td><td>0.881</td><td>0.594</td><td>0.664</td><td>1.000</td><td>0.814</td><td>0.903</td><td>0.800</td></tr><tr><td>Eigenvalues</td><td>0.700</td><td>0.723</td><td>0.723</td><td>0.723</td><td>0.723</td><td>0.723</td><td>0.723</td><td>0.723</td><td>0.776</td><td>0.849</td><td>0.804</td><td>0.729</td></tr><tr><td>Burgers equation</td><td>0.741</td><td>0.661</td><td>0.682</td><td>0.568</td><td>0.715</td><td>0.716</td><td>0.724</td><td>0.641</td><td>0.833</td><td>0.750</td><td>0.792</td><td>0.662</td></tr><tr><td>Convection-diffusion</td><td>0.494</td><td>0.479</td><td>0.420</td><td>0.269</td><td>0.502</td><td>0.542</td><td>0.514</td><td>0.287</td><td>0.520</td><td>0.500</td><td>0.447</td><td>0.277</td></tr><tr><td>Helmholtz equation</td><td>0.529</td><td>0.346</td><td>0.447</td><td>0.323</td><td>0.517</td><td>0.496</td><td>0.486</td><td>0.397</td><td>0.515</td><td>0.430</td><td>0.487</td><td>0.348</td></tr><tr><td>Average</td><td></td><td>0.804</td><td>0.693</td><td>0.642</td><td>0.700</td><td>0.837</td><td>0.771</td><td>0.686</td><td>0.739</td><td>0.869</td><td>0.774</td><td>0.733</td><td>0.731</td></tr></table>

More programs per hour. SwarmEvolve evaluates 5 to 13 times more programs per hour than the base lines (Figure 1, left). This difference comes from how the frameworks spend GPU time. The baselines spend most of it on text reasoning, which takes 20 to 35 seconds per program. In contrast, the 450 latent agents of one round take only a few seconds in total. The novelty filter of SwarmEvolve also saves time, because it uses the hidden states of the agents directly. ShinkaEvolve instead needs an external embedding mode and an LLM judge to reject similar proposals. Supplementary Table 2 gives a detailed time breakdown.

Faster to the same score. SwarmEvolve reaches the 60-minute scores of the baselines early in its run (Figure 4). The median time is 4 to 8 minutes with Qwen3 8B and 3 to 6 minutes with the larger LLMs. Each round of SwarmEvolve takes about one minute and explores several parent programs at once. As a result, the first few rounds already cover many directions, so the best program is discovered sooner.

Beyond AlphaEvolve with an 8B model. With Qwen3 8B, SwarmEvolve surpasses AlphaEvolve’s published constructions on three problems, compared with one for each baseline (Supplementary Table 5). Its packing of 26 circles in a square reaches the best known sum of radii, 2.635983 [62], and its step functions improve AlphaEvolve’s bounds for the second and third autocorrelation inequalities. While AlphaEvolve relies on frontier LLMs, SwarmEvolve achieves these results with swarms of small open-weight LLMs.

Together, these results show that SwarmEvolve is a faster and more efficient framework for evolutionary search, which finds better programs than text-based frameworks even with a small open-weight LLM.

Ours Ours, target not reached in 60 min  
![](images/a856b36ba8207c7afe150ab929125bd152301b9ab713329c6b0e4fd8f027b6c3.jpg)  
Figure 4: Minutes that SwarmEvolve needs to reach the best score that OpenEvolve (left), ShinkaEvolve (middle), and CodeEvolve (right) reach in 60 minutes. All frameworks use Qwen3 8B and run for three seeds. Hollow circles mark problems where SwarmEvolve does not reach the target within 60 minutes. On most problems, SwarmEvolve reaches in a few minutes what the baselines reach in one hour.

## 4.2 Latent space analysis

Latent agents make the reasoning of a swarm directly measurable and fast to filter numerically. In this section, we analyze how the reasoning trajectories of the swarms are spread across latent space.

Figure 5 shows one round of a circle packing problem with Qwen3 8B. It projects the centered and normalized mean hidden states of the latent agents onto their first two principal components.

On the left panel, agents of the same swarm lie close together and apart from other swarms, so each focus clearly moves its swarm to its own region of latent space. In this example, the first principal component roughly orders the swarms by role, from the refinement swarms to the exploration and fresh-start swarms.

On the right panel, within a swarm, the agents form clusters by the parent program they read. The novelty filter keeps agents from every cluster and covers the swarm roughly evenly. The focus prompts and the novelty filter therefore give each round proposals that differ both across swarms and within each swarm, leading to diversity in the search process.

![](images/94e860efe32848bf37ef65719b8f1294c59e53b46078479247084750ba13d668.jpg)  
Figure 5: Mean hidden states of the latent agents for a circle packing problem with Qwen3 8B, projected onto their first two principal components: the 450 latent agents of all 10 swarms with the swarm centers (left), and the 45 latent agents of one exploration-focused swarm, each reading different parent programs (right). The swarms occupy distinct regions, and the novelty filter keeps latent agents from every cluster.

## 5 Discussion

SwarmEvolve shows that evolutionary search can be fast, cheap, and strong at the same time. Latent agents skip the slow step of writing text, so one GPU can run hundreds of agents in each round. With the same model and the same time budget, SwarmEvolve evaluates many more programs than text-based frameworks, reaches their final scores early in the run, and ends with higher scores. Even with a small openweight LLM, it improves on several constructions that AlphaEvolve found with frontier LLMs.

Latent reasoning is a key part of this gain. When we replace the latent reasoning steps with text reasoning, the scores drop (Appendix A.1), even when the agents are given a higher text token budget than the latent token budget. Longer text makes each round slower, so the search evaluates fewer programs in the same time and ends with lower scores. Latent reasoning gives each agent enough room to think while keeping each round short.

Scaling the number of agents in a swarm also helps to improve the framework performance early in the search (Appendix A.2). Since a larger swarm studies each parent program from more perspectives, each round explores more of the design space initially. This early advantage matters most when each evaluation is expensive, such as a large simulation in science or engineering, where only a few programs can be evaluated. Later in the search, swarms of different sizes reach similar scores, while scaling to larger, more fiintelligent LLMs still raises the final score consistently.

More broadly, SwarmEvolve makes large-scale search for new programs affordable on a single GPU. This opens the method to many problems in mathematics and scientific computing. In mathematics, swarms can search for new constructions and proof strategies. In computing, they can improve algorithms for speed and accuracy. In scientific computing, they can write better PDE solvers, discover formulas through symbolic regression, and design more efficient numerical methods. Many of these problems have clear evaluators or desired outcomes, such as a higher accuracy or faster speed, which is what evolutionary search needs.

SwarmEvolve has three limitations. First, all latent agents and summarizers share one LLM, because a summarizer can read the KV caches of its agents directly only when they come from the same model.

Learned translations between the KV caches of different models [23] could extend SwarmEvolve to swarms whose agents run different models. Second, our implementation requires LLMs with standard attention. Linear-attention, state-space, and hybrid models compress the past into a fixed-size state [36, 27, 76], so latent communication between them needs a way to merge the states of several agents. Third, the memory for the KV caches of a round grows linearly with the number of agents, the number of latent reasoning steps, and the number of layers, so the largest LLM we test needs a large GPU. Compressed latent messages [11] and rounds spread over several GPUs could let SwarmEvolve scale to larger models and longer runs.

## 6 Summary

We introduce SwarmEvolve, an evolutionary search framework in which swarms of low-cost agents reason in latent space and communicate by passing KV caches. This design lets one GPU run hundreds of agents in each round, so the search explores much more of the design space in the same time. On problems from mathematical discovery and scientific computing, SwarmEvolve finds better programs faster than textbased frameworks, and with a small open-weight LLM it improves on several results that AlphaEvolve found with frontier LLMs. These results show that large-scale search for new mathematical constructions, algorithms, and scientific computing methods does not necessarily need frontier models or large compute budgets. Swarms of small latent agents can make this kind of AI-driven scientific discovery fast, cheap, capable, and widely available.

## Acknowledgments

This work was supported by the Defense Advanced Research Projects Agency (DARPA) MATHBAC grant HR0011-26-3-E152.

## References

[1] Henrique Assumpção, Diego Ferreira, Leandro Campos, and Fabricio Murai. “CodeEvolve: an open source evolutionary coding agent for algorithmic discovery and optimization”. In: arXiv preprint arXiv:2510.14150 (2025).

[2] Ivo M. Babuška and Stefan A. Sauter. “Is the pollution effect of the FEM avoidable for the Helmholtz equation considering high wave numbers?” In: SIAM Journal on Numerical Analysis 34.6 (1997), pp. 2392–2423.

[3] C. Bradford Barber, David P. Dobkin, and Hannu Huhdanpaa. “The quickhull algorithm for convex hulls”. In: ACM Transactions on Mathematical Software 22.4 (1996), pp. 469–483.

[4] Richard H. Bartels and G. W. Stewart. “Solution of the matrix equation $A X + X B = C ^ { \prime \prime }$ . In: Communications of the ACM 15.9 (1972), pp. 820–826.

[5] Paul Bateman and Paul Erd˝os. “Geometrical extrema suggested by a lemma of Besicovitch”. In: The American Mathematical Monthly 58.5 (1951), pp. 306–314.

[6] Timo Berthold, Dominik Kamp, Gioni Mexi, Sebastian Pokutta, and Imre Pólik. “Global Optimization for Combinatorial Geometry Problems Revisited in the Era of LLMs”. In: arXiv preprint arXiv:2601.05943 (2026).

[7] Federico Bianchi, Yongchan Kwon, Aneesh Pappu, and James Zou. “Harnessing the Collective Intelligence of AI Agents in the Wild for New Discoveries”. In: arXiv preprint arXiv:2606.10402 (2026).

[8] Jean Bourgain, Laurent Clozel, and Jean-Pierre Kahane. “Principe d’Heisenberg et fonctions positives”. In: Annales de l’Institut Fourier 60.4 (2010), pp. 1215–1232.

[9] Yang Cai, Vineet Gupta, Yanchen Jiang, Christopher Liaw, Aranyak Mehta, Grigoris Velegkas, and Di Wang. “Cogentic: Multi-Agent Orchestration for Automated Proof Discovery”. In: arXiv preprint arXiv:2609.40324 (2026).

[10] Mert Cemri, Shubham Agrawal, Akshat Gupta, Shu Liu, Audrey Cheng, Qiuyang Mang, Ashwin Naren, Lutfi Eren Erdogan, Koushik Sen, Matei Zaharia, Alex Dimakis, and Ion Stoica. “AdaEvolve: Adaptive LLM Driven Zeroth-Order Optimization”. In: arXiv preprint arXiv:2602.20133 (2026).

[11] Xiaoyu Chen, Fengge Wu, Junsuo Zhao, and Yun Fan. “CondenseFlow: Scalable Latent Space Collaboration via Semantic Compression for Multi-Agent Systems”. In: Findings of the Association for Computational Linguistics: ACL 2026. 2026, pp. 13694–13712.

[12] Audrey Cheng, Shu Liu, Melissa Pan, Zhifei Li, Bowen Wang, Alexander Krentsel, Tian Xia, Mert Cemri, Jongseok Park, Shuo Yang, Jeff Chen, Lakshya Agrawal, Aditya Desai, Jiarong Xing, Koushik Sen, Matei Zaharia, and Ion Stoica. “Barbarians at the Gate: How AI is Upending Systems Research”. In: arXiv preprint arXiv:2510.06189 (2025).

[13] Stephen Chung, Wenyu Du, and William J. Wesley. “Autonomous Mathematical Discovery in an Open-World Multi-Agent Environment”. In: arXiv preprint arXiv:2608.23691 (2026).

[14] Alexander Cloninger and Stefan Steinerberger. “On suprema of autoconvolutions with an application to Sidon sets”. In: Proceedings of the American Mathematical Society 145.8 (2017), pp. 3191–3200.

[15] Julian D. Cole. “On a quasi-linear parabolic equation occurring in aerodynamics”. In: Quarterly of Applied Mathematics 9.3 (1951), pp. 225–236.

[16] Francesc Comellas and J. Luis A. Yebra. “New lower bounds for Heilbronn numbers”. In: The Electronic Journal of Combinatorics 9 (2002), R6.

[17] John H. Conway and Neil J. A. Sloane. Sphere Packings, Lattices and Groups. 3rd ed. Springer, 1999.

[18] Tri Dao. “FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning”. In: International Conference on Learning Representations. 2024.

[19] Zhuoyun Du, Runze Wang, Huiyu Bai, Zouying Cao, Xiaoyong Zhu, Yu Cheng, Bo Zheng, Wei Chen, and Haochao Ying. “Enabling Agents to Communicate Entirely in Latent Space”. In: Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2026, pp. 27106–27129.

[20] Emilien Dupont, Marvin Eisenberger, Borislav Kozlovskii, Abbas Mehrabian, Francisco J. R. Ruiz, Abigail See, Renfei Zhou, Josh Alman, Virginia Vassilevska Williams, and Matej Balog. “Improving the matrix multiplication exponent with modern optimization and AlphaEvolve”. In: arXiv preprint arXiv:2608.16884 (2026).

[21] EinsteinArena. Leaderboard, accessed November 5, 2026. 2026. URL: https://einsteinarena.com.

[22] Erich Friedman. Hexagons in hexagons. Erich’s Packing Center, accessed November 5, 2026. 2026. URL: https://erich-friedman.github.io/packing/hexinhex/.

[23] Tianyu Fu, Zihan Min, Hanling Zhang, Jichao Yan, Guohao Dai, Wanli Ouyang, and Yu Wang. “Cache-to-Cache: Direct Semantic Communication Between Large Language Models”. In: International Conference on Learning Representations. 2026.

[24] Bogdan Georgiev, Javier Gómez-Serrano, Terence Tao, and Adam Zsolt Wagner. “Mathematical discovery and exploration can be done at scale”. In: Proceedings of the National Academy of Sciences 123.40 (2026), e2536158123.

[25] Felipe Gonçalves, Diogo Oliveira e Silva, and Stefan Steinerberger. “Hermite polynomials, linear flows on the torus, and an uncertainty principle for roots”. In: Journal of Mathematical Analysis and Applications 451.2 (2017), pp. 678–711.

[26] Teofilo F. Gonzalez. “Clustering to minimize the maximum intercluster distance”. In: Theoretical Computer Science 38 (1985), pp. 293–306.

[27] Albert Gu and Tri Dao. “Mamba: Linear-Time Sequence Modeling with Selective State Spaces”. In: Conference on Language Modeling. 2024.

[28] Katalin Gyarmati, François Hennecart, and Imre Z. Ruzsa. “Sums and differences of finite sets”. In: Functiones et Approximatio Commentarii Mathematici 37.1 (2007), pp. 175–186.

[29] Shibo Hao, Sainbayar Sukhbaatar, DiJia Su, Xian Li, Zhiting Hu, Jason Weston, and Yuandong Tian. “Training Large Language Models to Reason in a Continuous Latent Space”. In: Conference on Language Modeling. 2025.

[30] Jan Kristian Haugland. “The minimum overlap problem revisited”. In: arXiv preprint arXiv:1609.08000 (2016).

[31] Heilbronn problem records. Accessed November 5, 2026. 2026. URL: https://math.tejstead.com/ heilbronn/.

[32] John H. Holland. Adaptation in Natural and Artificial Systems. University of Michigan Press, 1975.

[33] Eberhard Hopf. “The partial differential equation $u _ { t } + u u _ { x } = \mu u _ { x x } ^ { \prime \prime }$ . In: Communications on Pure and Applied Mathematics 3.3 (1950), pp. 201–230.

[34] Jiachen Jiang, Tianyu Ding, and Zhihui Zhu. “DeltaEvolve: Accelerating Scientific Discovery through Momentum-Driven Evolution”. In: International Conference on Machine Learning. 2026.

[35] Qile Jiang and George Karniadakis. “AgenticSciML: Collaborative Multi-Agent Systems for Emergent Discovery in Scientific Machine Learning”. In: arXiv preprint arXiv:2511.07262 (2025).

[36] Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, and François Fleuret. “Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention”. In: International Conference on Machine Learning. 2020, pp. 5156–5165.

[37] Valentin Khrulkov, Andrey V. Galichin, Denis Bashkirov, Dmitry Vinichenko, Oleg Travkin, Roman Alferov, Andrey Kuznetsov, and Ivan Oseledets. “GigaEvo: An Open Source Optimization Framework Powered By LLMs And Evolution Algorithms”. In: arXiv preprint arXiv:2511.17592 (2025).

[38] Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. “Large Language Models are Zero-Shot Reasoners”. In: Advances in Neural Information Processing Systems. 2022.

[39] John R. Koza. Genetic Programming: On the Programming of Computers by Means of Natural Selection. MIT Press, 1992.

[40] Alexey Kravatskiy, Valentin Khrulkov, and Ivan Oseledets. “ImprovEvolve: Basin-Hopping Meets LLM-Guided Evolutionary Search”. In: arXiv preprint arXiv:2602.10233 (2026).

[41] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. “Efficient Memory Management for Large Language Model Serving with PagedAttention”. In: Proceedings of the 29th Symposium on Operating Systems Principles. 2023, pp. 611–626.

[42] Robert Tjarko Lange, Yuki Imajuku, and Edoardo Cetin. “ShinkaEvolve: Towards Open-Ended and Sample-Efficient Program Evolution”. In: International Conference on Learning Representations. 2026.

[43] Joel Lehman, Jonathan Gordon, Shawn Jain, Kamal Ndousse, Cathy Yeh, and Kenneth O. Stanley. “Evolution Through Large Models”. In: Handbook of Evolutionary Machine Learning. Springer, 2024, pp. 331–366.

[44] Yaniv Leviathan, Matan Kalman, and Yossi Matias. “Fast Inference from Transformers via Speculative Decoding”. In: International Conference on Machine Learning. 2023, pp. 19274–19286.

[45] Gang Liu, Yihan Zhu, Jie Chen, and Meng Jiang. “Scientific Algorithm Discovery by Augmenting AlphaEvolve with Deep Research”. In: arXiv preprint arXiv:2510.06056 (2025).

[46] Shu Liu, Shubham Agarwal, Monishwaran Maheswaran, Mert Cemri, Zhifei Li, Qiuyang Mang, Ashwin Naren, Ethan Boneh, Audrey Cheng, Melissa Z. Pan, Alexander Du, Kurt Keutzer, Alvin Cheung, Alexandros G. Dimakis, Koushik Sen, Matei Zaharia, and Ion Stoica. “EvoX: Meta-Evolution for Automated Discovery”. In: arXiv preprint arXiv:2602.23413 (2026).

[47] Greg Martin and Kevin O’Bryant. “The supremum of autoconvolutions, with applications to additive number theory”. In: Illinois Journal of Mathematics 53.1 (2009), pp. 219–235.

[48] Máté Matolcsi and Carlos Vinuesa. “Improved bounds on the supremum of autoconvolutions”. In: Journal of Mathematical Analysis and Applications 372.2 (2010), pp. 439–447.

[49] Elliot Meyerson, Mark J. Nelson, Herbie Bradley, Adam Gaier, Arash Moradi, Amy K. Hoover, and Joel Lehman. “Language Model Crossover: Variation through Few-Shot Prompting”. In: ACM Transactions on Evolutionary Learning and Optimization 4.4 (2024), pp. 1–40.

[50] Alexander Novikov, Ngân Vu, Marvin Eisenberger, Emilien Dupont, Po-Sen Huang, Adam Zsolt ˜ Wagner, Sergey Shirobokov, Borislav Kozlovskii, Francisco J. R. Ruiz, Abbas Mehrabian, M. Pawan Kumar, Abigail See, Swarat Chaudhuri, George Holland, Alex Davies, Sebastian Nowozin, Pushmeet Kohli, and Matej Balog. “AlphaEvolve: A coding agent for scientific and algorithmic discovery”. In: arXiv preprint arXiv:2506.13131 (2025).

[51] Chau Pham, Boyi Liu, Yingxiang Yang, Zhengyu Chen, Tianyi Liu, Jianbo Yuan, Bryan A. Plummer, Zhaoran Wang, and Hongxia Yang. “Let Models Speak Ciphers: Multiagent Debate through Embeddings”. In: International Conference on Learning Representations. 2024.

[52] Ao Qu, Han Zheng, Zijian Zhou, Yihao Yan, Yihong Tang, Shao Yong Ong, Fenglu Hong, Kaichen Zhou, Chonghe Jiang, Minwei Kong, Jiacheng Zhu, Xuan Jiang, Sirui Li, Cathy Wu, Bryan Kian Hsiang Low, Jinhua Zhao, and Paul Pu Liang. “CORAL: Towards Autonomous Multi-Agent Evolution for Open-Ended Discovery”. In: Conference on Language Modeling. 2026.

[53] Vignav Ramesh and Kenneth Li. “Communicating Activations Between Language Model Agents”. In: International Conference on Machine Learning. 2025.

[54] Bernardino Romera-Paredes, Mohammadamin Barekatain, Alexander Novikov, Matej Balog, M. Pawan Kumar, Emilien Dupont, Francisco J. R. Ruiz, Jordan S. Ellenberg, Pengming Wang, Omar Fawzi, Pushmeet Kohli, and Alhussein Fawzi. “Mathematical discoveries from program search with large language models”. In: Nature 625.7995 (2024), pp. 468–475.

[55] Klaus F. Roth. “On a problem of Heilbronn”. In: Journal of the London Mathematical Society 26.3 (1951), pp. 198–204.

[56] Yousef Saad. Iterative Methods for Sparse Linear Systems. 2nd ed. SIAM, 2003.

[57] Apoorv Saxena. Prompt Lookup Decoding. https : / / github . com / apoorvumang / prompt - lookup - decoding. 2023.

[58] Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, and Tri Dao. “Flash-Attention-3: Fast and Accurate Attention with Asynchrony and Low-precision”. In: Advances in Neural Information Processing Systems. 2024.

[59] Asankhaya Sharma. OpenEvolve: an open-source evolutionary coding agent. https : / / github . com / codelion/openevolve. 2025.

[60] Zhenyi Shen, Hanqi Yan, Linhai Zhang, Zhanghao Hu, Yali Du, and Yulan He. “CODI: Compressing Chain-of-Thought into Continuous Space via Self-Distillation”. In: Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing. 2025, pp. 677–693.

[61] Xiangyu Shi, Marco Chiesa, Gerald Q. Maguire Jr., and Dejan Kosti´c. “KVComm: Enabling Efficient LLM Communication through Selective KV Sharing”. In: International Conference on Learning Representations. 2026.

[62] Eckard Specht. Packomania: circles in a square. Accessed November 5, 2026. 2026. URL: https://www. packomania.com/csqv/csqv.html.

[63] Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. “RoFormer: Enhanced Transformer with Rotary Position Embedding”. In: Neurocomputing 568 (2024), p. 127063.

[64] Anja Šurina, Amin Mansouri, Lars C. P. M. Quaedvlieg, Amal Seddas, Maryna Viazovska, Emmanuel Abbe, and Caglar Gulcehre. “Algorithm Discovery With LLMs: Evolutionary Search Meets Reinforcement Learning”. In: Conference on Language Modeling. 2025.

[65] Terence Tao et al. Crowdsourced repository of optimization constants. Accessed November 5, 2026. 2026. URL: https://github.com/teorth/optimizationproblems.

[66] Lloyd N. Trefethen. Spectral Methods in MATLAB. SIAM, 2000.

[67] Lloyd N. Trefethen and Mark Embree. Spectra and Pseudospectra: The Behavior of Nonnormal Matrices and Operators. Princeton University Press, 2005.

[68] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin. “Attention Is All You Need”. In: Advances in Neural Information Processing Systems. 2017.

[69] Yuvraj Virk, Zack Edds, Chunqiu Steven Xia, and Lingming Zhang. “SwarmResearch: Orchestrating Coding Agents for Open-Ended Discovery”. In: arXiv preprint arXiv:2607.02807 (2026).

[70] Yiping Wang, Shao-Rong Su, Zhiyuan Zeng, Eva Xu, Liliang Ren, Xinyu Yang, Zeyi Huang, Xuehai He, Luyao Ma, Baolin Peng, Hao Cheng, Pengcheng He, Weizhu Chen, Shuohang Wang, Simon Shaolei Du, and Yelong Shen. “ThetaEvolve: Test-time Learning on Open Problems”. In: International Conference on Machine Learning. 2026.

[71] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le, and Denny Zhou. “Chain-of-Thought Prompting Elicits Reasoning in Large Language Models”. In: Advances in Neural Information Processing Systems. 2022.

[72] Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, et al. “Transformers: State-of-the-Art Natural Language Processing”. In: Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations. 2020, pp. 38–45.

[73] Xingyu Wu, Sheng-Hao Wu, Jibin Wu, Liang Feng, and Kay Chen Tan. “Evolutionary Computation in the Era of Large Language Model: Survey and Roadmap”. In: IEEE Transactions on Evolutionary Computation 29.2 (2025), pp. 534–554.

[74] Yige Xu, Xu Guo, Zhiwei Zeng, and Chunyan Miao. “SoftCoT: Soft Chain-of-Thought for Efficient Reasoning with LLMs”. In: Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2025, pp. 23336–23351.

[75] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. “Qwen3 Technical Report”. In: arXiv preprint arXiv:2505.09388 (2025).

[76] Songlin Yang, Jan Kautz, and Ali Hatamizadeh. “Gated Delta Networks: Improving Mamba2 with Delta Rule”. In: International Conference on Learning Representations. 2025.

[77] Haotian Ye, Haowei Lin, Jingyi Tang, Yizhen Luo, et al. “Structured Scaling of AI Discovery Across Diverse Scientific Domains”. In: arXiv preprint arXiv:2604.19341 (2026).

[78] Zhaojian Yu, Kaiyue Feng, Yilun Zhao, Shilin He, Xiao-Ping Zhang, and Arman Cohan. “AlphaResearch: Accelerating New Algorithm Discovery with Language Models”. In: arXiv preprint arXiv:2511.08522 (2025).

[79] Mert Yuksekgonul, Daniel Koceja, Xinhao Li, Federico Bianchi, Jed McCaleb, Xiaolong Wang, Jan Kautz, Yejin Choi, et al. “Learning to Discover at Test Time”. In: arXiv preprint arXiv:2601.16175 (2026).

[80] Zhen Zhang, Xuehai He, Weixiang Yan, Ao Shen, Chenyang Zhao, and Xin Eric Wang. “Soft Thinking: Unlocking the Reasoning Potential of LLMs in Continuous Concept Space”. In: Advances in Neural Information Processing Systems. 2025.

[81] Yujia Zheng, Zhuokai Zhao, Zijian Li, Yaqi Xie, Mingze Gao, Lizhu Zhang, and Kun Zhang. “Thought Communication in Multiagent Collaboration”. In: Advances in Neural Information Processing Systems. 2025.

[82] Hanlin Zhu, Shibo Hao, Zhiting Hu, Jiantao Jiao, Stuart Russell, and Yuandong Tian. “Reasoning by Superposition: A Theoretical Perspective on Chain of Continuous Thought”. In: Advances in Neural Information Processing Systems. 2025.

[83] Rui-Jie Zhu, Tianhao Peng, Tianhao Cheng, Xingwei Qu, Jinfa Huang, Dawei Zhu, Hao Wang, Kaiwen Xue, Xuanliang Zhang, Yong Shan, Tianle Cai, Taylor Kergan, Assel Kembay, Andrew Smith, Chenghua Lin, Binh Nguyen, Yuqi Pan, Yuhong Chou, Zefan Cai, Zhenhe Wu, Yongchi Zhao, Tianyu Liu, Jian Yang, Wangchunshu Zhou, Chujie Zheng, Chongxuan Li, Yuyin Zhou, Zhoujun Li, Zhaoxiang Zhang, Jiaheng Liu, Ge Zhang, Wenhao Huang, and Jason Eshraghian. “A Survey on Latent Reasoning”. In: arXiv preprint arXiv:2507.06203 (2025).

[84] Jiaru Zou, Rui Pan, Ruizhong Qiu, Pan Lu, Shizhe Diao, Jindong Jiang, Hanghang Tong, Tong Zhang, Markus J. Buehler, Jingrui He, and James Zou. “Recursive Multi-Agent Systems”. In: Advances in Neural Information Processing Systems. 2026.

[85] Jiaru Zou, Ruizhong Qiu, Gaotang Li, Xiyuan Yang, Katherine Tieu, Pan Lu, Ke Shen, Hanghang Tong, Yejin Choi, Jingrui He, James Zou, Mengdi Wang, and Ling Yang. “Latent Collaboration in Multi-Agent Systems”. In: International Conference on Machine Learning. 2026.

## A Ablation and Scaling Studies

## A.1 Replacing Latent Reasoning with Text Reasoning

Using Qwen3 8B, we replace each agent’s 40 latent reasoning steps with text reasoning of up to 40 tokens, which matches the latent budget, or up to 256 tokens, which is enough for a short paragraph. The summarizer receives this text in its prompt. Latent reasoning gives the highest average score after 60 minutes, 0.804, against 0.796 and 0.767 for 40 and 256 tokens of text reasoning (Supplementary Table 1).

Text reasoning is much slower. With 256 tokens, the agents of a round take about 32 seconds to write their text reasoning, against only about 4 seconds for 40 latent reasoning steps. The summarizers then encode the text of their agents, which adds 8 seconds per round with 40 tokens and 15 seconds with 256 tokens. In 60 minutes, SwarmEvolve therefore evaluates 734 programs with latent reasoning, more than the 663 and 402 with 40 and 256 tokens of text reasoning.

Specifically, text reasoning with 40 tokens scores higher than text reasoning with 256 tokens. This is potentially due to longer reasoning messages overwhelming the 32,768-token context window of the summarizer. This suggests that agent swarms can efficiently communicate with short messages, and latent reasoning and communication are more efficient than text.

Supplementary Table 1: Best score within 60 minutes, averaged over three runs. Each agent takes 40 latent reasoning steps (Latent, the default) or writes up to 40 or 256 tokens of text reasoning, which the summarizer reads as text in its prompt. Bold: best; underlined: second best.

<table><tr><td colspan="2">Qwen3 8B × 45 agents</td><td colspan="3">Text reasoning</td></tr><tr><td>Group</td><td>Problem</td><td>Latent (40 steps)</td><td>40 tokens</td><td>256 tokens</td></tr><tr><td rowspan="4">Packing</td><td>Circles in a square</td><td>0.998</td><td>0.997</td><td>0.996</td></tr><tr><td>Circles in a rectangle</td><td>0.888</td><td>0.868</td><td>0.887</td></tr><tr><td>Hexagons, n = 11</td><td>0.627</td><td>0.509</td><td>0.544</td></tr><tr><td>Hexagons, n = 12</td><td>0.707</td><td>0.738</td><td>0.600</td></tr><tr><td rowspan="5">Geometry</td><td>Heilbronn, triangle</td><td>0.558</td><td>0.796</td><td>0.596</td></tr><tr><td>Heilbronn, convex, n = 13</td><td>0.724</td><td>0.622</td><td>0.688</td></tr><tr><td>Heilbronn, convex, n = 14</td><td>0.807</td><td>0.609</td><td>0.536</td></tr><tr><td>Distance ratio, plane</td><td>0.994</td><td>0.863</td><td>0.828</td></tr><tr><td>Distance ratio, space</td><td>0.994</td><td>0.970</td><td>0.927</td></tr><tr><td rowspan="4">Analysis</td><td>Autocorrelation I</td><td>0.995</td><td>0.996</td><td>0.995</td></tr><tr><td>Autocorrelation II</td><td>1.021</td><td>1.016</td><td>1.021</td></tr><tr><td>Autocorrelation III</td><td>1.005</td><td>1.005</td><td>1.004</td></tr><tr><td>Uncertainty</td><td>0.911</td><td>0.906</td><td>0.911</td></tr><tr><td rowspan="4">Combinatorics</td><td>Erdős overlap</td><td>1.000</td><td>0.999</td><td>0.999</td></tr><tr><td>Sums and differences</td><td>0.957</td><td>0.960</td><td>0.955</td></tr><tr><td>Kissing number, d = 11</td><td>0.309</td><td>0.320</td><td>0.400</td></tr><tr><td>Sparse linear systems</td><td>0.933</td><td></td><td></td></tr><tr><td rowspan="5">Scientific computing</td><td></td><td></td><td>1.000</td><td>0.775</td></tr><tr><td>Eigenvalues</td><td>0.700</td><td>0.731</td><td>0.723</td></tr><tr><td>Burgers equation</td><td>0.741</td><td>0.730</td><td>0.694</td></tr><tr><td>Convection-diffusion</td><td>0.494</td><td>0.551</td><td>0.500</td></tr><tr><td>Helmholtz equation</td><td>0.529</td><td>0.528</td><td>0.526</td></tr><tr><td colspan="2">Average</td><td>0.804</td><td>0.796</td><td>0.767</td></tr><tr><td colspan="2">Programs evaluated per hour</td><td>734</td><td>663</td><td>402</td></tr></table>

## A.2 Swarm Size Scaling

We vary the number of latent agents in each swarm with Qwen3 8B: 0, 15, 45 (the default), and 75. The novelty filter keeps two thirds of them, so each summarizer reads 0, 10, 30, or 50 latent agents. In the case of 0 latent agents, the summarizer reads the problem statement and all programs of its swarm, the union of what the latent agents would read. Each setting runs three seeds with the setup of Section 4.

Larger latent teams raise the score most early in the search (Supplementary Figures 1 and 2). Over the first 150 evaluated programs, the mean best score increases with the swarm size. From the 55th program on, SwarmEvolve with 75 agents has the highest mean best score of all settings and baselines. The three baselines evaluate 80 to 160 programs in 60 minutes, and their final scores lie below the final score of every swarm size. After about 150 programs, the gains fall within the bootstrap intervals. Smaller swarms run faster, so they evaluate more programs in the 60-minute budget and their scores keep rising for longer. All four settings end the hour at similar scores.

The early gain suggests that a larger swarm examines each program from more perspectives, so its summarizer draws on more diverse ideas. The later convergence likely reflects the limits of the LLM itself. Since all latent agents share their knowledge, additional agents can’t find the blind spots that earlier agents overlook, and the diversity gain saturates. A larger model does raise the final score as shown in Table 1.

![](images/67b67237c0b10fa2aefc4e03e893c72b5ae8aa08fe8f282e20eca2a7f47aedaa.jpg)  
Supplementary Figure 1: Mean best score against the number of evaluated programs with Qwen3 8B, for 0 to 75 latent agents per swarm and the three baselines, over the first 150 programs (left) and the full 60- minute runs (right). More latent agents raise the score early in the search, and the settings converge later.

![](images/740d30c4650691e8664ef81a208968ce42db6b0662337217c27390cfcb76b315.jpg)

Supplementary Figure 2: Paired gain in mean best score relative to 0 latent agents per swarm, plotted against the number of evaluated programs with Qwen3 8B, with 95% bootstrap intervals. The gains are largest in the first 100 programs and shrink within their intervals later in the search.

## B Time Breakdowns

Supplementary Table 2 divides the 60 minutes of each run among the stages of GPU work, averaged over the runs of each framework with Qwen3 8B. These times exclude the CPU time of the evaluator, which runs in parallel and never holds up the GPU. Besides proposing programs, ShinkaEvolve asks the LLM to summarize earlier programs, and CodeEvolve asks it to rewrite its system prompt.

Supplementary Table 2: Minutes (and shares) of a 60-minute run that each framework spends on each stage of its GPU work, averaged over its runs on the 21 problems with Qwen3 8B and three seeds.
<table><tr><td>Stage</td><td>Ours</td><td>OpenEvolve</td><td>ShinkaEvolve</td><td>CodeEvolve</td></tr><tr><td>Reading prompts</td><td>7.1 min (12%)</td><td>1.0 min (2%)</td><td>0.5 min (0.9%)</td><td>0.1 min (0.2%)</td></tr><tr><td>Text reasoning</td><td></td><td>52.5 min (87%)</td><td>34.4 min (57%)</td><td>46.8 min (78%)</td></tr><tr><td>Latent reasoning</td><td>4.1 min (7%)</td><td></td><td></td><td></td></tr><tr><td>Latent communication</td><td>2.6 min (4%)</td><td></td><td></td><td></td></tr><tr><td>Writing programs</td><td>39.4 min (66%)</td><td>6.4 min (11%)</td><td>5.7 min (9%)</td><td>4.7 min (8%)</td></tr><tr><td>Novelty checks</td><td>0.05 min (0.1%)</td><td></td><td>4.5 min (7%)</td><td></td></tr><tr><td>Summarizing earlier programs</td><td></td><td></td><td>14.7 min (24%)</td><td></td></tr><tr><td>Rewriting the system prompt</td><td></td><td></td><td></td><td>7.7 min (13%)</td></tr><tr><td>Parameter tuning agent</td><td>1.9 min (3%)</td><td></td><td></td><td></td></tr><tr><td>Other</td><td>4.7 min (8%)</td><td>0.1 min (0.2%)</td><td>0.2 min (0.4%)</td><td>0.7 min (1%)</td></tr><tr><td>Total runtime</td><td>60 min (100%)</td><td>60 min (100%)</td><td>60 min (100%)</td><td>60 min (100%)</td></tr><tr><td>Total number of programs evaluated</td><td>734</td><td>158</td><td>99</td><td>81</td></tr></table>

Compared with the three baselines, SwarmEvolve achieves a higher throughput mainly in the following two ways:

1. Latent reasoning instead of text reasoning. Text reasoning takes most of the GPU time of all three baselines, 20 to 35 seconds and several thousand tokens per evaluated program. SwarmEvolve replaces it with latent reasoning and latent communication, which together take about a tenth of its GPU time: the latent agents of a round reason for a few seconds in total. SwarmEvolve therefore spends most of its GPU time writing programs, and this shift explains its higher throughput.

2. Novelty check in latent space. To check novelty, ShinkaEvolve embeds the code of each new program with a separate embedding model and, when its cosine similarity to an archived program exceeds 0.99, asks the LLM whether the two programs differ. These extra LLM calls take several minutes of each run. The novelty filter of SwarmEvolve instead compares the mean hidden states of the latent agents, which are already in GPU memory, and takes seconds per run.

## C Hyperparameters and Implementation

Throughout this work, our framework uses the hyperparameters in Supplementary Table 3. The summarizers of the refinement-focused swarms, which edit one of the best programs, sample at the lowest temperature for stability, while those of the exploration-focused and fresh-start swarms, which write new approaches, sample at the highest temperature for diversity. Supplementary Table 4 lists the focuses of the swarms.

Supplementary Table 3: Hyperparameters of SwarmEvolve used throughout all experiments in this work.
<table><tr><td colspan="2">Agent</td><td>Temperature</td><td>Top-p</td><td>Top-k</td><td>Maximum new tokens</td></tr><tr><td rowspan="4">Summarizer</td><td>Refinement-focused swarm</td><td>0.3</td><td>0.95</td><td>20</td><td>4096</td></tr><tr><td>Hybridization-focused swarm</td><td>0.5</td><td>0.95</td><td>20</td><td>4096</td></tr><tr><td>Exploration-focused swarm</td><td>0.8</td><td>0.95</td><td>20</td><td>4096</td></tr><tr><td>Fresh-start swarm</td><td>0.8</td><td>0.95</td><td>20</td><td>4096</td></tr><tr><td>Parameter tuning agent Hyperparameter</td><td></td><td>1.0</td><td>0.95</td><td>20</td><td>24</td></tr><tr><td colspan="2"></td><td colspan="4">Value</td></tr><tr><td colspan="2">Latent agents per swarm, run and kept</td><td colspan="4">45,30</td></tr><tr><td colspan="2">Latent reasoning steps per latent agent</td><td colspan="4">40</td></tr><tr><td colspan="2">Maximum refinement steps of one program</td><td colspan="4">20</td></tr><tr><td colspan="2">One-number parameter tuning edits per round</td><td colspan="4">up to 8</td></tr></table>

Supplementary Table 4: Focuses of the swarms. A refinement- or hybridization-focused swarm takes one of the 14 “aspects”, and an exploration-focused or fresh-start swarm takes one of the 14 “perspectives”. Each latent agent receives the focus in one of five phrasings, such as “Pay particular attention to” an aspect or “Consider” a perspective, followed by one of nine ways of thinking, such as “Reason from first principles” or “Question one assumption the program makes” for sample diversity.

<table><tr><td>Aspects (refinement and hybridization)</td><td>Perspectives (exploration and fresh start)</td></tr><tr><td>The starting point of the solution</td><td>The standard textbook method</td></tr><tr><td>The optimization or solution algorithm</td><td>A fundamentally different algorithm</td></tr><tr><td>Global exploration, restarts, and annealing</td><td>A known construction from the literature</td></tr><tr><td>Symmetries and structure of good solutions</td><td>A physics-inspired method</td></tr><tr><td>Feasibility, tolerances, and a final repair step</td><td>A parametrization that satisfies the constraints</td></tr><tr><td>The use of the time limit</td><td>Many cheap starts of one fast local optimizer</td></tr><tr><td>The largest source of the remaining error</td><td>A symmetry that forces structure</td></tr><tr><td>A smoother surrogate or a penalty formulation</td><td>A coarse-to-fine or incremental construction</td></tr><tr><td>The weakest part of the current solution</td><td>A greedy construction with local improvement</td></tr><tr><td>What the best programs in the field do differently</td><td>Linear or convex programming for convex parts</td></tr><tr><td>Bugs, wasted computation, and silent failures</td><td>An exact or analytical treatment</td></tr><tr><td>Simpler and more robust methods</td><td>Sparsity, separability, or low rank</td></tr><tr><td>The numbers in the program</td><td>A higher-order or adaptive method</td></tr><tr><td>The formulation and its free variables</td><td>An idea that nobody would try first</td></tr></table>

Implementation. We implement SwarmEvolve in PyTorch with the Qwen3 models of Hugging Face Transformers [72]. The latent agents of a round run as one batch, and latent agents that read the same program attend to one shared copy of its KV cache through the paged FlashAttention kernel of vLLM [18, 41]. The summarizers of a round decode as one batch with paged KV caches and prompt-lookup speculative sampling, which drafts tokens by copying them from the programs in the context [57] and verifies them with the acceptance rule of speculative sampling, so the samples follow the distribution of the model exactly [44]. With Qwen3 14B and 32B-FP8, SwarmEvolve also captures its decoding steps in CUDA graphs and uses the fused kernels of vLLM, including FlashAttention-3 on the H100 GPU [58]. The three baselines serve the model with vLLM, which batches requests continuously, caches shared prefixes, and runs CUDA graphs, so their decoding runs at the speed of a production serving engine. CodeEvolve runs its configuration for Qwen models with eight islands, one per CPU core. All four frameworks run Qwen3 32B-FP8, which uses the FP8 weights released by Qwen.

## D Benchmark Problems

The 16 mathematical problems in Sections D.1–D.16 are those of Appendix B of AlphaEvolve [50], with the problem statements, initial programs, and evaluators of OpenEvolve [59]; the problem statement of circle packing in the unit square comes from ShinkaEvolve [42]. All feasibility checks use the numerical tolerances of the OpenEvolve evaluators. A construction with objective value v scores $v / v _ { \mathrm { A E } }$ , or $v _ { \mathrm { A E } } / v$ for minimization, where $v _ { \mathrm { A E } }$ is the exact value of AlphaEvolve’s published construction,<sup>1</sup> so a score above 1 improves on AlphaEvolve. Where an evaluator divides by a rounded $v _ { \mathrm { A E } } .$ , we rescale its scores to the exact value.

We also design the 5 scientific computing problems in Sections D.17–D.21. Each problem consists of several cases, and a program solves one case at a time. A case scores min $( d _ { \mathrm { c a p } } , - \log _ { 1 0 } e ) / d _ { \mathrm { c a p } }$ , where e is the relative error with respect to a high-accuracy reference solution, and the score of a program is the mean over the cases.

## D.1 Circle Packing in the Unit Square

Problem. Circle packing asks how to place disjoint circles in a container, and this variant lets every circle have its own radius. Pack $n = 2 6$ disjoint circles, with centers $c _ { i } = ( x _ { i } , y _ { i } )$ and radii $r _ { i } ,$ inside the unit square so that the sum of their radii is as large as possible (Supplementary Figure 3):

$$
\begin{array} { r l } { \underset { c _ { i } , r _ { i } } { \operatorname* { m a x } } } & { \displaystyle \sum _ { i = 1 } ^ { n } r _ { i } } \\ { \mathrm { s u b j e c t \ t o } } & { r _ { i } \leq x _ { i } \leq 1 - r _ { i } , \quad r _ { i } \leq y _ { i } \leq 1 - r _ { i } , } \\ & { r _ { i } \geq 0 , \quad \| c _ { i } - c _ { j } \| \geq r _ { i } + r _ { j } \quad \mathrm { f o r } i \neq j . } \end{array}
$$

![](images/1c095c757e2c0454cf5b71094926a85228556a1eae247a2a0fcee3e4645498a9.jpg)

(7)

Supplementary Figure 3: Disjoint circles in the unit square. The objective is the sum of the radii $r _ { i }$

Evaluator. The entry function run\_packing() returns the centers as a $2 6 \times 2$ array, the radii as an array of length 26, and their sum. The evaluator rejects negative radii and checks both constraints with the tolerance $\epsilon = 1 0 ^ { - 6 } \mathrm { : }$ every circle satisfies $r _ { i } - \epsilon \le x _ { i } \le 1 - r _ { i } + \epsilon$ and $r _ { i } - \epsilon \le y _ { i } \le 1 - r _ { i } + \epsilon ,$ and every pair satisfies $\| \boldsymbol { c } _ { i } - \boldsymbol { c } _ { j } \| \ge r _ { i } + r _ { j } - \epsilon$ . It recomputes the objective value $\boldsymbol { v } = \sum _ { i } \boldsymbol { r } _ { i }$ from the returned radii.

## D.2 Circle Packing in a Rectangle

Problem. This circle packing problem fixes the perimeter of a rectangular container and lets the construction choose its shape together with the circles. Pack $n = 2 1$ disjoint circles, with centers $c _ { i } = \left( x _ { i } , y _ { i } \right)$ and radii $\boldsymbol { r } _ { i } ,$ inside a rectangle of perimeter 4 so that the sum of their radii is as large as possible. The width w of the rectangle is free, and its height is $2 - w$ (Supplementary Figure 4):

$$
\begin{array} { r l } { \underset { w , c _ { i } , r _ { i } } { \operatorname* { m a x } } } & { \displaystyle \sum _ { i = 1 } ^ { n } r _ { i } } \\ { \mathrm { s u b j e c t \ t o } } & { r _ { i } \leq x _ { i } \leq w - r _ { i } , \quad r _ { i } \leq y _ { i } \leq 2 - w - r _ { i } , } \\ & { r _ { i } \geq 0 , \quad \lVert c _ { i } - c _ { j } \rVert \geq r _ { i } + r _ { j } \quad \mathrm { f o r } i \neq j . } \end{array}\tag{8}
$$

![](images/ee9695d9525968d438932c17e6d53783de313bac34089aff2c5c197917e7cd39.jpg)  
Supplementary Figure 4: Disjoint circles in a rectangle of width w and height $2 - w .$ . The objective is the sum of the radii.

Evaluator. The entry function circle\_packing21() returns a $2 1 \times 3$ array with rows $\left( x _ { i } , y _ { i } , r _ { i } \right)$ . The evaluator rejects negative radii and computes the width m $\mathrm { a x } _ { i } ( x _ { i } + r _ { i } ) - \mathrm { m i n } _ { i } ( x _ { i } - r _ { i } )$ and the height ma $\mathsf x _ { i } ( y _ { i } +$ $r _ { i } ) - \mathrm { m i n } _ { i } ( y _ { i } - r _ { i } )$ of the smallest axis-aligned rectangle that contains the circles. It requires the sum of this width and height to be at most $2 + \epsilon$ and every pair of circles to satisfy $\begin{array} { r } { \| \boldsymbol { c } _ { i } - \boldsymbol { c } _ { j } \| \geq r _ { i } + r _ { j } - \epsilon . } \end{array}$ , with $\epsilon = 1 0 ^ { - 6 }$ The objective value is $\boldsymbol { v } = \sum _ { i } \boldsymbol { r } _ { i }$

## D.3 Hexagon Packing with 11 Hexagons

Problem. This problem packs congruent regular hexagons into the smallest regular hexagon that contains them, where each hexagon may be translated and rotated. Let $\mathcal { H } ( x , y , \theta , a )$ denote the closed regular hexagon with center $\left( x , y \right)$ , side length a, and vertices at the angles $\theta + 6 0 ^ { \circ } k$ for $k = 0 , \ldots , 5$ . Pack $n = 1 1$ regular hexagons of side length 1, $\mathcal { H } _ { i } = \mathcal { H } ( x _ { i } , y _ { i } , \theta _ { i } , 1 )$ , with disjoint interiors inside an outer regular hexagon $\dot { \mathcal { H } } _ { 0 } = \mathcal { H } ( x _ { 0 } , y _ { 0 } , \theta _ { 0 } , R )$ of the smallest possible side length R (Supplementary Figure 5):

![](images/8b4b39a7ab62e3bf52fabef806723b56a0d336b326887ea591e9505ee7b4ad6a.jpg)

$$
\begin{array} { r l r } { \underset { R , x _ { 0 } , y _ { 0 } , \theta _ { 0 } , x _ { i } , y _ { i } , \theta _ { i } } { \operatorname* { m i n } } } & { R } & \\ { \mathrm { s u b j e c t ~ t o } } & { \mathcal { H } _ { i } \subseteq \mathcal { H } _ { 0 } , } & { \mathrm { i n t } \mathcal { H } _ { i } \cap \mathrm { i n t } \mathcal { H } _ { j } = \emptyset \mathrm { f o r } i \not = j , } \end{array}
$$

where int denotes the interior.

(9)

Supplementary Figure 5: Eleven disjoint unit hexagons inside a regular hexagon of side length $R ,$ which is minimized.

Evaluator. The entry function hexagon\_packing\_11() returns an $1 1 \times 3$ array with rows $\left( x _ { i } , y _ { i } , \theta _ { i } \right)$ , with angles in degrees, the triple $\left( x _ { 0 } , y _ { 0 } , \theta _ { 0 } \right)$ of the outer hexagon, and R. The evaluator computes the six vertices of every hexagon. For every pair of inner hexagons, it projects both hexagons onto the edge normals of the two hexagons and requires a gap larger than ϵ along at least one normal, which is the separating axis test for convex polygons. It requires every vertex of every inner hexagon to lie inside the outer hexagon, with the tolerance ϵ on its distance to each edge, where $\epsilon \doteq 1 0 ^ { - 6 }$ . The objective value is $v = R$

## D.4 Hexagon Packing with 12 Hexagons

Problem. This problem packs congruent regular hexagons into the smallest regular hexagon that contains them, where each hexagon may be translated and rotated. Let $\mathcal { H } ( x , y , \theta , a )$ denote the closed regular hexagon with center $( x , y )$ , side length $a ,$ and vertices at the angles $\bar { \theta { + } } 6 0 ^ { \circ } k$ for $k = 0 , \ldots , 5 .$ Pack $n = 1 2$ regular hexagons of side length $1 , \mathcal { H } _ { i } = \mathcal { H } ( x _ { i } , y _ { i } , \theta _ { i } , 1 )$ , with disjoint interiors inside an outer regular hexagon $\mathcal { H } _ { 0 } = \mathcal { H } ( \boldsymbol { x } _ { 0 } , \boldsymbol { y } _ { 0 } , \boldsymbol { \theta } _ { 0 } , R )$ of the smallest possible side length R (Supplementary Figure 6):

![](images/24a1d26837e1071995ee83ae22d17e21809385089f4b9e4057db175e576168fc.jpg)

$$
\begin{array} { r l r } { \underset { R , x _ { 0 } , y _ { 0 } , \theta _ { 0 } , x _ { i } , y _ { i } , \theta _ { i } } { \operatorname* { m i n } } } & { R } & \\ { \mathrm { s u b j e c t ~ t o } } & { \mathcal { H } _ { i } \subseteq \mathcal { H } _ { 0 } , } & { \mathrm { i n t } \mathcal { H } _ { i } \cap \mathrm { i n t } \mathcal { H } _ { j } = \emptyset \mathrm { f o r } i \not = j , } \end{array}\tag{10}
$$

where int denotes the interior.

Supplementary Figure ${ \bf 6 : }$ Twelve disjoint unit hexagons inside a regular hexagon of side length $R ,$ which is minimized.

Evaluator. The entry function hexagon\_packing\_12() returns a $1 2 \times 3$ array with rows $\left( x _ { i } , y _ { i } , \theta _ { i } \right)$ , with angles in degrees, the triple $\left( x _ { 0 } , y _ { 0 } , \theta _ { 0 } \right)$ of the outer hexagon, and R. The evaluator computes the six vertices of every hexagon. For every pair of inner hexagons, it projects both hexagons onto the edge normals of the two hexagons and requires a gap larger than ϵ along at least one normal, which is the separating axis test for convex polygons. It requires every vertex of every inner hexagon to lie inside the outer hexagon, with the tolerance ϵ on its distance to each edge, where $\epsilon = 1 0 ^ { - 6 }$ . The objective value is $v = R$

## D.5 Heilbronn Problem in a Triangle

Problem. The Heilbronn problem asks for n points in a region that maximize the smallest area of a triangle formed by three of the points [55, 16]. Place $n = 1 1$ points in the equilateral triangle $T$ with vertices $( 0 , 0 ) , ( 1 , 0 ) \quad$ , and $\textstyle { \bigl ( } { \frac { 1 } { 2 } } , { \frac { \sqrt { 3 } } { 2 } } { \bigr ) }$ to maximize the smallest area of a triangle formed by three of the points (Supplementary Figure 7):

$$
\operatorname* { m a x } _ { p _ { 1 } , \ldots , p _ { n } \in T } \frac { \operatorname* { m i n } _ { i < j < k } \operatorname { a r e a } ( p _ { i } , p _ { j } , p _ { k } ) } { \operatorname { a r e a } ( T ) } ,\tag{11}
$$

![](images/c991fe8d2f07f02b17f7a024f8c1faf519756c058c55d67fa997762579d8d444.jpg)  
Supplementary Figure 7: Eleven points in the equilateral triangle. The shaded triangle has the smallest area among all triangles formed by the points.

where area $\left( { p _ { i } , p _ { j } , p _ { k } } \right)$ is the area of the triangle with vertices $p _ { i } , p _ { j } ,$ and $p _ { k } .$ , and area $( T ) = { \sqrt { 3 } } / 4$

Evaluator. The entry function heilbronn\_triangle11() returns an $1 1 \times 2$ array of points. The evaluator checks that every point $( x , y )$ satisfies the three edge inequalities of $T , y \ge - \epsilon , \sqrt { 3 } x \le \sqrt { 3 } - y + \epsilon ,$ , and $y \leq \sqrt { 3 } x + \epsilon ,$ , with $\epsilon = 1 0 ^ { - 6 }$ . It computes the areas $\begin{array} { r } { \frac { 1 } { 2 } | \mathrm { d e t } ( p _ { j } - p _ { i } , p _ { k } - p _ { i } ) | } \end{array}$ of all 165 triangles and divides the smallest area by $\sqrt { 3 } / 4$ to obtain v.

## D.6 Heilbronn Problem in a Convex Region with 13 Points

Problem. The Heilbronn problem asks for n points in a region that maximize the smallest area of a triangle formed by three of the points [55, 16]. In this variant the region is any convex region of unit area, so the relevant region is the convex hull of the points, and the ratio below is invariant under invertible affine maps of the plane. Place $n = 1 3$ points in the plane to maximize the smallest area of a triangle formed by three of the points, relative to the area of their convex hull (Supplementary Figure 8):

$$
\operatorname* { m a x } _ { p _ { 1 } , \ldots , p _ { n } \in \mathbb { R } ^ { 2 } } \frac { \operatorname* { m i n } _ { i < j < k } \mathrm { a r e a } ( p _ { i } , p _ { j } , p _ { k } ) } { \mathrm { a r e a } ( \mathrm { c o n v } \{ p _ { 1 } , \ldots , p _ { n } \} ) } ,\tag{12}
$$

![](images/a036fac1c574f41b2f52532e249fb28c63300cc02f615912b2e902df831b7b3e.jpg)  
Supplementary Figure 8: Thirteen points with their convex hull (dashed). The shaded triangle has the smallest area.

where area $\left( { { p } _ { i } } , { { p } _ { j } } , { { p } _ { k } } \right)$ is the area of the triangle with vertices $p _ { i } , p _ { j } .$ and $p _ { k } ,$ , and conv denotes the convex hull.

Evaluator. The entry function heilbronn\_convex13() returns a $1 3 \times 2$ array of points anywhere in the plane. The evaluator computes the areas of all 286 triangles and the area of the convex hull of the points with the Quickhull algorithm [3] in SciPy. It divides the smallest triangle area by the hull area to obtain v.

## D.7 Heilbronn Problem in a Convex Region with 14 Points

Problem. The Heilbronn problem asks for n points in a region that maximize the smallest area of a triangle formed by three of the points [55, 16]. In this variant the region is any convex region of unit area, so the relevant region is the convex hull of the points, and the ratio below is invariant under invertible affine maps of the plane. Place $n = 1 4$ points in the plane to maximize the smallest area of a triangle formed by three of the points, relative to the area of their convex hull (Supplementary Figure 9):

![](images/7ca6d78f5be325e23e89f696db43f841738f6b67991eb20563008f9235aae072.jpg)

$$
\operatorname* { m a x } _ { p _ { 1 } , \ldots , p _ { n } \in \mathbb { R } ^ { 2 } } \frac { \operatorname* { m i n } _ { i < j < k } \mathrm { a r e a } ( p _ { i } , p _ { j } , p _ { k } ) } { \mathrm { a r e a } ( \mathrm { c o n v } \{ p _ { 1 } , \ldots , p _ { n } \} ) } ,\tag{13}
$$

where area $\left( { p } _ { i } , { p } _ { j } , { p } _ { k } \right)$ is the area of the triangle with vertices ${ } p _ { i } , p _ { j } ,$ and $p _ { k } .$ , and conv denotes the convex hull.

Supplementary Figure 9: Fourteen points with their convex hull (dashed). The shaded triangle has the smallest area.

Evaluator. The entry function heilbronn\_convex14() returns a $1 4 \times 2$ array of points anywhere in the plane. The evaluator computes the areas of all 364 triangles and the area of the convex hull of the points with the Quickhull algorithm [3] in SciPy. It divides the smallest triangle area by the hull area to obtain v.

## D.8 Distance Ratio in the Plane

Problem. This problem asks for evenly spread point sets, whose smallest pairwise distance is large relative to the diameter of the set [5]. Place $n = 1 6$ points in the plane to maximize the ratio of their smallest pairwise distance $d _ { \mathrm { m i n } } ^ { \mathrm { ~ - ~ } } = \mathrm { \ m i n } _ { i \neq j } \| p _ { i } - p _ { j } \|$ to their largest pairwise distance $d _ { \operatorname* { m a x } } = \operatorname* { m a x } _ { i \neq j } \lVert p _ { i } - \dot { p _ { j } } \rVert$ (Supplementary Figure 10):

$$
\operatorname* { m a x } _ { p _ { 1 } , \ldots , p _ { n } \in \mathbb { R } ^ { 2 } } \left( \frac { d _ { \operatorname* { m i n } } } { d _ { \operatorname* { m a x } } } \right) ^ { 2 } .\tag{14}
$$

![](images/3d2df42232e7c81c4eb3f3b426fe3efd0a2f15a6f63e52670280979b82a5c113.jpg)  
Supplementary Figure 10: Sixteen points in the plane. The thin lines join the pairs at the smallest distance $d _ { \operatorname* { m i n } } ,$ and the dashed line marks the largest distance $d _ { \mathrm { m a x } }$

The ratio is invariant under translations, rotations, and scalings of the point set.

Evaluator. The entry function min\_max\_dist\_dim2\_16() returns a $1 6 \times 2$ array of points anywhere in the plane. The evaluator computes the 120 pairwise distances with SciPy and forms $v = ( d _ { \operatorname* { m i n } } / \dot { d _ { \operatorname* { m a x } } } ) ^ { 2 }$

## D.9 Distance Ratio in Space

Problem. This problem asks for evenly spread point sets, whose smallest pairwise distance is large relative to the diameter of the set [5]. Place $n = 1 4$ points in three-dimensional space to maximize the ratio of their smallest pairwise distance $d _ { \operatorname* { m i n } } \stackrel { - } { = } \operatorname* { m i n } _ { i \neq j } \lVert p _ { i } - p _ { j } \rVert$ to their largest pairwise distance $d _ { \operatorname* { m a x } } = \operatorname* { m a x } _ { i \neq j } \lVert p _ { i } - p _ { j } ^ { * } \rVert \mathrm { ( S u p p l e - }$ mentary Figure 11):

$$
\operatorname* { m a x } _ { p _ { 1 } , \dots , p _ { n } \in \mathbb { R } ^ { 3 } } \left( \frac { d _ { \operatorname* { m i n } } } { d _ { \operatorname* { m a x } } } \right) ^ { 2 } .\tag{15}
$$

![](images/20c5aee0e817985892b1374ca5b1e08d57c0930a1313c006c7092c3e99895487.jpg)  
Supplementary Figure 11: Fourteen points in space, shaded by depth. The thin lines join the pairs at the smallest distance $d _ { \operatorname* { m i n } } ,$ and the dashed line marks the largest distance $d _ { \mathrm { m a x } }$

The ratio is invariant under translations, rotations, and scalings of the point set.

Evaluator. The entry function min\_max\_dist\_dim3\_14() returns a $1 4 \times 3$ array of points anywhere in space. The evaluator computes the 91 pairwise distances with SciPy and forms $v = ( d _ { \operatorname* { m i n } } ^ { \bullet } / d _ { \operatorname* { m a x } } ) ^ { \sharp }$

## D.10 First Autocorrelation Inequality

Problem. For every non-negative function $f$ supported on $[ - \frac { 1 } { 4 } , \frac { 1 } { 4 } ]$ , the first autocorrelation inequality states

$$
\operatorname* { m a x } _ { - 1 / 2 \leq t \leq 1 / 2 } ( f * f ) ( t ) \geq C _ { 1 } \left( \int _ { - 1 / 4 } ^ { 1 / 4 } f ( x ) d x \right) ^ { 2 } ,\tag{16}
$$

where $\textstyle ( f * f ) ( t ) = \int f ( t - x ) f ( x )$ dx is the autoconvolution of $f .$ The constant $C _ { 1 }$ bounds the size of Sidon sets in additive combinatorics, and the bounds $1 . 2 8 \leq C _ { 1 } \leq 1$ .5098 come from Cloninger and Steinerberger

[14] and Matolcsi and Vinuesa [48]. Every such $f$ with a positive integral gives an upper bound on $C _ { 1 }$ . Find a non-negative function $f$ supported on $\left[ - { \frac { 1 } { 4 } } , { \frac { 1 } { 4 } } \right]$ with a positive integral that minimizes this bound,

$$
\operatorname* { m i n } _ { f \geq 0 } { \frac { \operatorname* { m a x } _ { t } ( f * f ) ( t ) } { \left( \int f \right) ^ { 2 } } } .\tag{17}
$$

The program represents $f$ by its values $f _ { 1 } , \ldots , f _ { N }$ on N equal intervals of $[ - \frac { 1 } { 4 } , \frac { 1 } { 4 } ]$ and chooses N.

Evaluator. The entry function run() returns $f _ { 1 } , \ldots , f _ { N } .$ , the bound ${ \tilde { v } } ,$ a loss value, and N. The evaluator requires every $f _ { i }$ to be non-negative and, with the interval width $\Delta = 1 / \left( 2 N \right)$ , computes

$$
v = { \frac { \operatorname* { m a x } _ { k } \Delta \sum _ { i } f _ { i } f _ { k - i } } { \left( \Delta \sum _ { i } f _ { i } \right) ^ { 2 } } } .\tag{18}
$$

The autoconvolution of a step function is piecewise linear with breakpoints at multiples of $\Delta ,$ so v is the exact bound of the step function with values $f _ { i } .$ . The evaluator requires $( \dot { \Delta } \sum _ { i } f _ { i } ) ^ { 2 } \geq 1 0 ^ { - 8 }$ and $\left| v - \tilde { v } \right| \leq 1 0 ^ { - 6 }$ and the score uses v˜.

## D.11 Second Autocorrelation Inequality

Problem. For every non-negative function $f ,$ the second autocorrelation inequality states

$$
\| f \ast f \| _ { 2 } ^ { 2 } \leq C _ { 2 } \| f \ast f \| _ { 1 } \| f \ast f \| _ { \infty } ,\tag{19}
$$

where $\textstyle ( f * f ) ( t ) = \int f ( t - x ) f ( x )$ dx is the autoconvolution of $f$ and $\textstyle \| f * f \| _ { 1 } = ( \int f ) ^ { 2 } [ 4 7 ]$ . The inequality $\| g \| _ { 2 } ^ { 2 } \leq \| g \| _ { 1 } \| g \| _ { \infty }$ gives $C _ { 2 } \leq 1$ , and Matolcsi and Vinuesa [48] prove $C _ { 2 } \geq 0 . 8 8 9 2 2$ with an explicit function. Every such $f$ with a positive integral gives a lower bound on $C _ { 2 }$ . Find a non-negative function $f$ with a positive integral that maximizes this bound,

$$
\operatorname* { m a x } _ { f \geq 0 } { \frac { \| f * f \| _ { 2 } ^ { 2 } } { \| f * f \| _ { 1 } \| f * f \| _ { \infty } } } .\tag{20}
$$

The program represents f by its values $f _ { 1 } , \ldots , f _ { N }$ on N equal intervals and chooses $N .$

Evaluator. The entry function run() returns $f _ { 1 } , \ldots , f _ { N } ,$ the bound ${ \tilde { v } } ,$ a loss value, and N. The evaluator requires $f _ { i } \geq - 1 0 ^ { - 6 }$ , sets negative values to zero, and forms the discrete autoconvolution $\begin{array} { r } { g _ { k } = \sum _ { i } f _ { i } f _ { k - i } } \end{array}$ for $k \bar { = } 1 , \ldots , 2 N - 1$ . It places these values on the interior nodes of a uniform grid with spacing $\Delta = 1 / ( 2 N )$ on $[ - \frac { 1 } { 2 } , \frac { 1 } { 2 } ]$ , sets $g _ { 0 } \ = \ g _ { 2 N } \ = \ 0$ at the two end nodes, and computes the norms of the piecewise linear interpolant g of these values exactly,

$$
\| g \| _ { 2 } ^ { 2 } = \sum _ { k = 0 } ^ { 2 N - 1 } \frac { \Delta } { 3 } \left( g _ { k } ^ { 2 } + g _ { k } g _ { k + 1 } + g _ { k + 1 } ^ { 2 } \right) , \qquad \| g \| _ { 1 } = \Delta \sum _ { k } | g _ { k } | , \qquad \| g \| _ { \infty } = \operatorname* { m a x } _ { k } | g _ { k } | .\tag{21}
$$

The ratio is invariant under scaling of $f$ and of its domain, so $v = \| g \| _ { 2 } ^ { 2 } / ( \| g \| _ { 1 } \| g \| _ { \infty } )$ is the exact bound of the step function with values $f _ { i }$ . The score uses this recomputed v.

## D.12 Third Autocorrelation Inequality

Problem. The third autocorrelation inequality extends the first one to functions that take values of either sign: for every function $f$ supported on $[ - \frac { 1 } { 4 } , \frac { 1 } { 4 } ]$

$$
\operatorname* { m a x } _ { - 1 / 2 \leq t \leq 1 / 2 } | ( f * f ) ( t ) | \geq C _ { 3 } \left( \int _ { - 1 / 4 } ^ { 1 / 4 } f ( x ) d x \right) ^ { 2 } ,\tag{22}
$$

where $\textstyle ( f * f ) ( t ) = \int f ( t - x ) f ( x )$ dx is the autoconvolution of $f .$ Every such $f$ with a nonzero integral gives an upper bound on $C _ { 3 }$ . Find a function $f$ supported on $[ - \frac { 1 } { 4 } , \frac { 1 } { 4 } ]$ ], with values of either sign and a nonzero integral, that minimizes this bound,

$$
\operatorname* { m i n } _ { f } { \frac { \operatorname* { m a x } _ { t } { \big | } ( f * f ) ( t ) { \big | } } { { \big ( } \int f { \big ) } ^ { 2 } } } .\tag{23}
$$

The program represents $f$ by its values $f _ { 1 } , \ldots , f _ { N }$ on N equal intervals of $[ - \frac { 1 } { 4 } , \frac { 1 } { 4 } ]$ and chooses N.

Evaluator. The entry function run() returns $f _ { 1 } , \ldots , f _ { N }$ , the bound ${ \tilde { v } } ,$ a loss value, and N. With the interval width $\Delta = 1 / \left( 2 N \right)$ , the evaluator computes

$$
v = { \frac { \operatorname* { m a x } _ { k } \Delta \left| \sum _ { i } f _ { i } f _ { k - i } \right| } { \left( \Delta \sum _ { i } f _ { i } \right) ^ { 2 } } } .\tag{24}
$$

The autoconvolution of a step function is piecewise linear with breakpoints at multiples of $\Delta ,$ so v is the exact bound of the step function with values $f _ { i }$ . The evaluator requires $( \dot { \Delta } \sum _ { i } f _ { i } ) ^ { 2 } \geq 1 0 ^ { - 9 } \mathrm { a n d } | v - \tilde { v } | \leq 1 0 ^ { - 3 }$ and the score uses v˜. The evaluator divides by 1.4557, AlphaEvolve’s bound on |max $( f * f ) ( t ) | / ( \bar { f } f ) ^ { 2 } ;$ for v as defined here, AlphaEvolve’s value is $v _ { \mathrm { A E } } = 1 . 4 6 8 8$

## D.13 Uncertainty Inequality

Problem. The uncertainty principle of Bourgain, Clozel, and Kahane [8] bounds from below the product of the radii beyond which an even function and its Fourier transform are non-negative. For a real even function f with Fourier transform $\begin{array} { r } { \hat { f } ( \xi ) = \int f ( x ) e ^ { - 2 \pi i x \xi } d x , f ( 0 ) \le 0 , } \end{array}$ , and ${ \hat { f } } ( 0 ) \leq 0 .$ , let

$$
A ( f ) = \operatorname* { i n f } \{ r > 0 : f ( x ) \geq 0 { \mathrm { ~ f o r ~ a l l ~ } } | x | \geq r \}\tag{25}
$$

be the radius of the last sign change of $f .$ The inequality $A ( f ) A ( { \hat { f } } ) \geq C _ { 4 }$ holds for a positive constant $C _ { 4 } ,$ so every such $f$ gives an upper bound on $C _ { 4 }$ . Following Gonçalves, Silva, and Steinerberger [25], the search runs over $f ( x ) = P ( { \sqrt { 2 \pi } } x ) e ^ { - \pi x ^ { 2 } }$ with $\begin{array} { r } { P = \sum _ { k = 0 } ^ { m } a _ { k } H _ { 4 k } } \end{array}$ and $P ( 0 ) = 0 ~ $ , where $H _ { 4 k }$ is the Hermite polynomial of degree 4k. Each such $f$ equals its own Fourier transform and satisfies $f ( 0 ) = { \hat { f } } ( 0 ) = 0 .$ , so $\bar { A } ( \bar { f } ) A ( \hat { f } ) = r _ { \operatorname* { m a x } } ^ { 2 } / ( 2 \pi )$ , where $r _ { \mathrm { m a x } }$ is the largest positive root at which P changes sign. Find coefficients $a _ { 0 } , \ldots , a _ { m }$ that minimize this bound,

$$
\operatorname* { m i n } _ { a _ { 0 } , \ldots , a _ { m } } \ { \frac { r _ { \operatorname* { m a x } } ^ { 2 } } { 2 \pi } } .\tag{26}
$$

Evaluator. The entry function run() returns $a _ { 0 } , \ldots , a _ { m - 1 } .$ , the bound ${ \tilde { v } } ,$ and the root $\tilde { r } _ { \mathrm { m a x } }$ . The evaluator builds $P$ in exact rational arithmetic with SymPy: it converts every $a _ { k }$ to a rational number, chooses $a _ { m } =$ $\begin{array} { r } { - \sum _ { k < m } a _ { k } H _ { 4 k } \bigl ( 0 \bigr ) / H _ { 4 m } \bigl ( 0 \bigr ) } \end{array}$ so that $P ( 0 ) = 0$ , and multiplies P by −1 when $P ( x )  - \infty \mathrm { a s } x  \infty$ . It divides $P \log x ^ { 2 }$ , isolates the real roots of the quotient, evaluates every root to 200 digits, and keeps the roots at which the quotient changes sign. The largest of these roots is $r _ { \mathrm { m a x } } ,$ and the objective value is $\dot { v } = r _ { \mathrm { m a x } } ^ { 2 } / ( 2 \pi )$ . The evaluator requires v˜ and $\tilde { r } _ { \mathrm { m a x } }$ to agree with v and $r _ { \mathrm { m a x } }$ to within a relative and an absolute tolerance of $1 0 ^ { - 9 }$ and the score uses the recomputed v. $v _ { \mathrm { A E } } = 0 . 3 2$ 16 comes from Laguerre polynomials, outside this family; AlphaEvolve’s Hermite construction gives 0.3521.

## D.14 Erd˝os Minimum Overlap Problem

Problem. For a partition of $\left\{ 1 , \ldots , 2 n \right\}$ into two sets X and Y of n elements each, let $M _ { k }$ be the number of pairs $( x , y ) \in { \mathsf { \bar { X } } } \times Y$ with $x - y = k .$ . The minimum overlap problem of Erd˝os asks for the smallest possible value of $\mathbf { m a x } _ { k } M _ { k }$ , and this minimum divided by n converges to a constant $C _ { 5 }$ as $n  \infty [ 3 0 ]$ . In

(3, −10)

the continuous formulation, every function $h : [ 0 , 2 ]  [ 0 , 1 ]$ with $\begin{array} { r } { \int _ { 0 } ^ { 2 } h ( x ) d x = 1 } \end{array}$ gives an upper bound on $C _ { 5 }$ . Find such a function h that minimizes this bound,

$$
\operatorname* { m i n } _ { h } \operatorname* { m a x } _ { k } \int h ( x ) \bar { h } ( x + k ) d x , \qquad \bar { h } = \left\{ \begin{array} { l l } { 1 - h } & { \mathrm { o n } [ 0 , 2 ] , } \\ { 0 } & { \mathrm { e l s e w h e r e } . } \end{array} \right.\tag{27}
$$

The program represents h by its values $h _ { 1 } , \ldots , h _ { N }$ on N equal intervals of [0, 2] and chooses N.

Evaluator. The entry function run() returns $h _ { 1 } , \ldots , h _ { N }$ , the bound v˜, and N. The evaluator requires every $h _ { i }$ to lie in [0, 1] and, with the interval width $\Delta = 2 / N$ , requires $| \Delta \sum _ { i } h _ { i } - 1 | \leq 1 0 ^ { - 3 }$ . It computes

$$
v = \operatorname* { m a x } _ { k } \Delta \sum _ { i } h _ { i } \left( 1 - h _ { i + k } \right) ,\tag{28}
$$

where k ranges over all shifts and the sum runs over the indices with $1 \leq i + k \leq N$ . It requires $| v - \tilde { v } | \leq$ $1 0 ^ { - 4 }$ , and the score uses v˜. Our reported scores instead correct h to integrate exactly to 1 and recompute v.

## D.15 Sums and Differences of Finite Sets

Problem. For a finite set U of integers, Gyarmati, Hennecart, and Ruzsa [28] study how large the difference set $U - U = \{ u - u ^ { \prime } : u , u ^ { \prime } \in \breve { U } \}$ can be relative to the sum set $U + U = \{ u + { u ^ { \prime } } : u , { u ^ { \prime } } \in U \}$ . For a finite set U of non-negative integers with $0 \in U$ , the quantity

$$
1 + \frac { \log { \left( | U - U | / | U + U | \right) } } { \log { \left( 2 \operatorname* { m a x } { U } + 1 \right) } }\tag{29}
$$

is a lower bound on the constant $C _ { 6 }$ of this problem. Find a finite set U of non-negative integers with $0 \in U$ that maximizes this bound.

Evaluator. The entry function run() returns U as a one-dimensional array and the bound v˜. The evaluator requires $0 \in U$ and every element of U to be a non-negative integer. It computes $U + U$ and $U - U$ from all pairwise sums and differences, recomputes the objective value $v ,$ requires v and $\tilde { v }$ to agree to within a relative tolerance of $1 0 ^ { - 5 }$ and an absolute tolerance of $\mathrm { 1 0 ^ { - 8 } }$ , and the score uses ${ \tilde { v } } .$

## D.16 Kissing Number in Dimension 11

Problem. The kissing number $\tau _ { 1 1 }$ is the largest number of unit balls in $\mathbb { R } ^ { 1 1 }$ with disjoint interiors that touch a central unit ball [17]. A set $S \subset \mathbb { Z } ^ { 1 1 } \setminus \{ 0 \}$ whose largest norm is at most its smallest pairwise distance satisfies $\langle p , q \rangle \leq { \frac { 1 } { 2 } } \| p \| \| q \|$ for every pair, so the unit balls centered at the points $2 p / \bar { \| } p \|$ touch the central unit ball and have disjoint interiors, which proves $\tau _ { 1 1 } \geq | S | ;$ Supplementary Figure 12 shows the two-dimensional analogue. Find the largest such set:

$$
\begin{array} { r l } { \underset { S \subset \mathbb { Z } ^ { 1 1 } \backslash \{ 0 \} } { \operatorname* { m a x } } } & { | S | } \\ { \mathrm { s u b j e c t ~ t o } } & { \underset { p \in S } { \operatorname* { m a x } } \lVert p \rVert \leq \underset { p , q \in S , p \neq q } { \operatorname* { m i n } } \lVert p - q \rVert . } \end{array}
$$

![](images/938267e358210c651416b684127eeaef8bcb6fecac1300eca50418dc42ae02b1.jpg)

(30)

Supplementary Figure 12: The two-dimensional analogue: five points $p$ of $\mathbb { Z } ^ { 2 }$ that satisfy the constraint, and the unit circles centered at $2 p / \| p \| .$ , which touch the central unit circle.

Evaluator. The entry function kissing\_number11() returns an array with 11 columns. The evaluator rounds every coordinate to the nearest integer and computes the squared norms and the squared pairwise distances in exact integer arithmetic. It requires the smallest squared norm to be positive and the smallest squared pairwise distance to be at least the largest squared norm, and the objective value is $v = | S |$

## D.17 Sparse Linear Systems

Problem. Solve $A x = b$ for five sparse matrices A from finite-difference discretizations of partial differential equations on the unit square or cube with Dirichlet boundary conditions [56]:

1. the two-dimensional Poisson equation with the 5-point stencil on a $2 5 6 \times 2 5 6 ~ \mathrm { g r i d } ~ ( n ~ = ~ 6 5 , 5 3 6$ unknowns), symmetric positive definite;

2. the three-dimensional diffusion equation $- \nabla \cdot ( c \nabla u ) = f$ with the 7-point stencil on a $6 4 ^ { 3 }$ grid $( n =$ $^ { 2 6 2 , 1 4 4 ) }$ , where c is constant on the blocks of an $8 \times 8 \times 8$ partition with values $1 0 ^ { s } , s$ uniform in $[ 0 , 6 ] .$ and harmonic means on the cell faces, symmetric positive definite and ill-conditioned;

3. the two-dimensional Helmholtz operator $- \Delta - k ^ { 2 }$ with $k = 4 0 . 3$ and the 5-point stencil on a $3 0 0 \times 3 0 0$ grid $( n = 9 0 , 0 0 0 )$ , symmetric indefinite;

4. the two-dimensional convection-diffusion operator $- \varepsilon \Delta u + w \cdot \nabla u$ with the rotating flow $w = ( 2 y -$ $1 , 1 - 2 x ) , \varepsilon = 1 0 ^ { - 3 }$ , and first-order upwind differences on a $3 0 0 \times 3 0 0$ grid $( n = 9 0 , 0 0 0 )$ , nonsymmetric;

5. the three-dimensional convection-diffusion operator with the flow $w = ( 2 y - 1 , 1 - 2 x , 0 . 5 ) , \varepsilon = 1 0 ^ { - 2 } .$ and first-order upwind differences on a $4 8 ^ { 3 }$ grid $( n = 1 1 0 , 5 9 2 )$ , nonsymmetric.

Evaluator. The entry function solve $( \mathtt { A } , \mathtt { b } )$ receives A as a $\operatorname { S c i P y }$ sparse matrix in compressed sparse row format and returns x. Each right-hand side is $b = A x ^ { \star }$ for a solution $x ^ { \star }$ with independent standard normal entries, and the relative error is $e = \| x - x ^ { \star } \| _ { 2 } / \| x ^ { \star } \| _ { 2 }$ . The cap is $d _ { \tt C a p } = 8$ digits, and the time limit is 9 seconds per case. The initial program uses the conjugate gradient method with a Jacobi preconditioner and a relative tolerance of $1 0 ^ { - 1 0 }$

## D.18 Eigenvalues of Non-Normal Matrices

Problem. Compute all eigenvalues of six square matrices with small integer entries to high relative accuracy. Backward-stable eigenvalue algorithms return the eigenvalues of a nearby matrix, and the eigenvalues of a non-normal matrix can be far from those of a nearby matrix, so these algorithms lose digits on non-normal matrices [67]. The six matrices are

1. a dense circulant matrix of size $n = 2 0 0 .$ , which is normal;

2. a tridiagonal Toeplitz matrix with diagonal 9, sub-diagonal 1, and super-diagonal 4, of size $n = 2 0 0 ;$

3. the Kronecker sum of two nonsymmetric tridiagonal Toeplitz matrices of sizes 24 and 22, an upwind discretization of a two-dimensional convection-diffusion operator, of size $n = 5 2 8 ;$

4. a nonsymmetric tridiagonal matrix with random diagonal entries in $\{ - 1 2 , \ldots , 1 2 \}$ , sub-diagonal entries in $\{ 1 , \dot { 2 } \}$ , and super-diagonal entries in $\{ 4 , \ldots , 8 \}$ , with rows and columns permuted by the same random permutation, of size $n = 2 0 0 ;$

5. a block upper-triangular matrix with $2 \times 2$ diagonal blocks and sparse random entries above them, with rows and columns permuted by the same random permutation, of size $n = 2 0 0$ ;

6. a dense matrix $U B U ^ { - 1 }$ , where B is block diagonal with $2 \times 2$ blocks and U is a product of elementary integer matrices with determinant 1, of size $n = 1 8 0$

No eigenvalue is zero, and the problem statement names the structure of each matrix.

Evaluator. The entry function eigenvalues(A) receives an integer array and returns the n eigenvalues as a complex array in any order. The evaluator knows every eigenvalue in closed form or from a well-conditioned computation: the discrete Fourier transform of the first row of the circulant matrix, $a + 2 \sqrt { b c } \cos ( j \pi / ( m + 1 ) )$ for a tridiagonal Toeplitz matrix of size m with diagonal a and off-diagonals b and $c ,$ sums of these values for the Kronecker sum, the eigenvalues of the similar symmetric tridiagonal matrix with off-diagonals $\sqrt { b _ { i } c _ { i } }$ for the fourth matrix, and the eigenvalues of the $2 \times 2$ blocks for the last two matrices. It pairs the returned eigenvalues one-to-one with the true eigenvalues $\lambda _ { j }$ by the assignment that minimizes the total relative error, and the error e of a matrix is the largest relative error $| \mu _ { j } - \bar { \lambda } _ { j } | / | \lambda _ { j } |$ over the pairs. The cap is $d _ { \tt C a p } = 1 0$ digits, and the time limit is 8 seconds per case. The initial program calls the LAPACK eigenvalue routine of NumPy.

## D.19 Viscous Burgers Equation

Problem. Solve the one-dimensional viscous Burgers equation

$$
u _ { t } + u u _ { x } = \nu u _ { x x } , \quad x \in [ - 1 , 1 ] , \quad u ( x , 0 ) = - \sin ( \pi x ) ,\tag{31}
$$

with periodic boundary conditions up to the final time $T =$ $^ { 1 , }$ for the five viscosities $\nu \in \{ 0 . 1 , \bar { 0 . 0 3 } , 0 . 0 1 , 0 . 0 0 3 , 0 . 0 0 1 \}$ From $t = 1 / \pi \mathrm { o n } ,$ , a shock of width proportional to ν forms at $x = 0$ (Supplementary Figure 13).

![](images/cf071decdc1bc7d0a2c46176e7198550344ef3a2304045f97fd8cbda3083ac87.jpg)  
Supplementary Figure 13: Reference solution $u ( x , 1 )$ for three viscosities. The shock at $x = 0$ steepens as ν decreases.

Evaluator. The entry function solve(nu, T, x) receives 1000 points x drawn uniformly from $[ - 1 , 1 ]$ and returns $u ( x , T )$ at these points. The reference solution is the Cole–Hopf solution [33, 15],

$$
u ( x , t ) = - \frac { \int \sin ( \pi ( x - y ) ) F ( x - y ) G ( y ) d y } { \int F ( x - y ) G ( y ) d y } , F ( y ) = e ^ { - \cos ( \pi y ) / ( 2 \pi \nu ) } , G ( y ) = e ^ { - y ^ { 2 } / ( 4 \nu t ) } ,\tag{32}
$$

which the evaluator integrates with the trapezoidal rule on 8001 nodes in scaled logarithmic form; four times as many nodes and arbitrary-precision quadrature give the same values. The error e is the relative 2-norm error over the 1000 points, the cap is $\begin{array} { r } { d _ { \mathsf { c a p } } = 6 \mathrm { d i g i t s } , } \end{array}$ , and the time limit is 9 seconds per case. The initial program uses explicit central finite differences on 256 grid points with forward Euler time steps.

## D.20 Convection-Diffusion Equation

Problem. Solve the steady convection-diffusion equation

$$
- \varepsilon ( u _ { x x } + u _ { y y } ) + b _ { 1 } u _ { x } + b _ { 2 } u _ { y } = 1 \quad \mathrm { i n ~ } ( 0 , 1 ) ^ { 2 } ,
$$

$$
u = 0 \quad \mathrm { o n t h e b o u n d a r y } ,\tag{33}
$$

(34)

with the constant flow $b = ( 1 , 0 . 5 )$ , for the five diffusion coefficients $\varepsilon \in \{ 1 0 ^ { - 1 } , 1 0 ^ { - 2 } , 3 \cdot 1 0 ^ { - 3 } , 1 0 ^ { - 3 } , 3 \cdot 1 0 ^ { - 4 } \}$ . The solution has boundary layers of width proportional to ε along the outflow sides $x = 1$ and $y = 1$ (Supplementary $\mathrm { F i g - }$ ure 14).

![](images/ab1c0de8165a2379e0b0766c58a85d236a9dc29bde2cf445bcad77a58fefe25d.jpg)  
Supplementary Figure 14: Reference solution for $\varepsilon \ = \ 1 0 ^ { - 2 }$ , with boundary layers along $x = 1$ and $y = 1$

Evaluator. The entry function solve(eps, b, x, y) receives 2000 points drawn uniformly from the unit square and returns u at these points. The reference solution comes from Chebyshev collocation [66]: with constant coefficients, the collocation equations on the interior points form a Sylvester equation, which the evaluator solves with the Bartels–Stewart algorithm [4] and interpolates to the points with barycentric interpolation. Each reference uses two resolutions, between 60 and 1200 Chebyshev points per direction depending on ε, that agree to $1 0 ^ { - 8 }$ . The error e is the relative 2-norm error over the 2000 points, the cap is $d _ { \tt C a p } = 6$ digits, and the time limit is 9 seconds per case. The initial program uses central finite differences on a $6 4 \times 6 4$ grid with a sparse direct solver and bilinear interpolation.

## D.21 Helmholtz Equation

Problem. Solve the Helmholtz equation with the impedance boundary condition

$$
- ( u _ { x x } + u _ { y y } ) - k ^ { 2 } u = f \quad \mathrm { i n } ( 0 , 1 ) ^ { 2 } ,
$$

$$
\partial _ { n } u - i k u = 0 \quad \mathrm { o n t h e b o u n d a r y } ,\tag{35}
$$

(36)

where $\partial _ { n }$ is the outward normal derivative, for the five wave numbers $k \in \{ 1 0 , 2 0 , 4 0 , 6 0 , 8 0 \}$ . The source consists of two Gaussians whose width is proportional to the wavelength,

$$
\begin{array} { c } { { f ( x , y ) = e ^ { - \frac { k ^ { 2 } } { 2 } \left( ( x - 0 . 3 ) ^ { 2 } + ( y - 0 . 6 ) ^ { 2 } \right) } } } \\ { { - 0 . 6 e ^ { - \frac { k ^ { 2 } } { 2 } \left( ( x - 0 . 7 ) ^ { 2 } + ( y - 0 . 2 5 ) ^ { 2 } \right) } , } } \end{array}\tag{37}
$$

![](images/9f71faf8b0934ad7c0b65a3e093c0dfb7bcd7e76a3f436f677ca2714a09d790d.jpg)  
Supplementary Figure 15: Real part of the reference solution for $k = 4 0 ,$ , with waves radiating from the two sources.

and the solution u is complex (Supplementary Figure 15). At a fixed number of grid points per wavelength, the error of low-order discretizations grows with k [2].

Evaluator. The entry function solve $( \Bbbk , \ \textup { \bf x } , \ \textup { \bf y } )$ receives 2000 points drawn uniformly from the unit square and returns the complex values of u at these points. The reference solution comes from Chebyshev collocation with the boundary values eliminated through the impedance condition in each direction, which again gives a Sylvester equation $[ 6 6 , 4 ]$ . Each reference uses two resolutions, between 60 and 320 Chebyshev points per direction depending on k, that agree to 10<sup>−8</sup>, and second-order finite differences converge to it at the expected rate. The error e is the relative 2-norm error over the 2000 points, the cap is $d _ { \mathrm { c a p } } = 6$ digits, and the time limit is 9 seconds per case. The initial program uses second-order finite differences on a $1 0 1 \times 1 0 1$ grid, with ghost points for the impedance condition, and a sparse direct solver.

## E Best Known Results

Supplementary Table 5 lists the objective values behind the scores of Table 1: the value of AlphaEvolve’s published construction, the best value of the three runs of each framework, and the best known value. We rerun the best program of every run and recompute its value under AlphaEvolve’s exact constraints. For the programs that reproduce their logged score, the exact constraints lower the score by at most $3 . 3 \times 1 0 ^ { - 4 }$ . The remaining programs draw unseeded random numbers, so we rerun each of them 33 times (once without a seed and with seeds 0 to 31) and report the best value under the exact constraints among the reruns that the evaluator scores no higher than the logged score of the run. We call a best known value verified if its construction or certificate has been checked, and claimed otherwise.

With Qwen3 8B and 32B-FP8, SwarmEvolve improves on AlphaEvolve on the same three problems: it matches the best known packing of 26 circles in the unit square [62], and its step functions improve AlphaEvolve’s bounds on the second and third autocorrelation inequalities. The three baselines improve on AlphaEvolve only on the autocorrelation inequalities.

Two problems need care in the comparison with AlphaEvolve. On the third autocorrelation inequality, the evaluator of OpenEvolve takes the absolute value of the autoconvolution inside the maximum, while AlphaEvolve’s published bound of 1.4557 takes it outside the maximum. Under the definition of the evaluator, AlphaEvolve’s construction gives 1.4688, the value that we compare against (Appendix D.12). On the uncertainty inequality, the evaluator accepts only functions built from Hermite polynomials. Within this family, AlphaEvolve’s first construction gives 0.3521, while its best bound of 0.3216 uses Laguerre polynomials, which lie outside the family. SwarmEvolve reaches 0.3521 with Qwen3 32B-FP8, the value of AlphaEvolve’s construction within the same family.

With Qwen3 32B-FP8, SwarmEvolve comes close to AlphaEvolve on four more problems: within one part per million on circles in a rectangle, and within 0.02% on the two distance ratios and the minimum overlap problem. For all four frameworks, the largest gaps to AlphaEvolve include the hexagon packings, the Heilbronn problems, and the kissing number in dimension 11.

Supplementary Table 5: Best values on the 16 AlphaEvolve problems: the value of AlphaEvolve, the best of three 60-minute runs of each framework with Qwen3 8B, 14B, and 32B-FP8, and the best known value to the best of our knowledge. Values are given to seven significant digits. Bold: best of all values of a problem; underlined: second best; <sup>†</sup>: better than AlphaEvolve; –: no valid program within 60 minutes.
<table><tr><td>Problem</td><td>Objective</td><td>AlphaEvolve Qwen3</td><td></td><td>Ours</td><td>OpenEvolve ShinkaEvolve CodeEvolve</td><td></td><td></td><td>Best known (Nov. 5, 2026)</td><td>Status</td></tr><tr><td rowspan="3">Circles in a square</td><td rowspan="3">∑iri↑</td><td rowspan="3">2.635863</td><td>8B 14B</td><td>2.635983† 2.634292</td><td>2.615077 2.346719</td><td>2.626003</td><td>2.613583 2.616260</td><td rowspan="3">2.635983</td><td rowspan="3">verified [62]</td></tr><tr><td>32B-FP8</td><td>2.635983†</td><td></td><td>2.630328</td><td>2.620510</td></tr><tr><td></td><td></td><td>2.613827</td><td>2.629864</td><td></td></tr><tr><td rowspan="3">Circles in a rectangle</td><td rowspan="3">∑r↑</td><td rowspan="3">2.365832</td><td>8B 14B</td><td>2.100000 2.357262</td><td>2.332669 2.268246</td><td>2.357264</td><td>2.342198 2.357264</td><td rowspan="3">2.365832</td><td rowspan="3">verified [6]</td></tr><tr><td>32B-FP8</td><td>2.365830</td><td>2.313769</td><td>2.295059 2.272727</td><td>2.346834</td></tr><tr><td></td><td>4.505917</td><td>6.002853</td><td></td><td>8.000000</td></tr><tr><td rowspan="3">Hexagons, n = 11</td><td rowspan="3">R↓</td><td rowspan="3">3.930092</td><td>8B 14B</td><td>4.677289</td><td>5.404701</td><td>8.000000 6.031025</td><td>6.333333</td><td rowspan="3">3.924501</td><td rowspan="3">verified [22]</td></tr><tr><td>32B-FP8</td><td>4.472263</td><td>5.004000</td><td>5.418582</td><td>8.000000</td></tr><tr><td>8B</td><td>4.924979</td><td>6.031025</td><td>6.031025</td><td>8.000000</td></tr><tr><td rowspan="3">Hexagons, n = 12</td><td rowspan="3">R↓</td><td rowspan="3">3.941912</td><td>14B</td><td>4.537633</td><td>5.080000</td><td>6.031025</td><td>5.595754</td><td rowspan="3">3.94164</td><td rowspan="3">claimed [22]</td></tr><tr><td>32B-FP8</td><td>5.029604</td><td>5.000400</td><td>6.829775</td><td>8.000000</td></tr><tr><td>8B</td><td>0.02227742</td><td>0.001599633</td><td>0.02003859</td><td>0.02217071</td></tr><tr><td rowspan="3">Heilbronn, triangle</td><td rowspan="3">Amin/A↑</td><td rowspan="3">0.03652989</td><td>14B</td><td>0.02600437</td><td>0.009468544</td><td>0.01510912</td><td>0.02590121</td><td rowspan="3">0.03652989</td><td rowspan="3">verified [50]</td></tr><tr><td>32B-FP8</td><td>0.03064911</td><td>0.003508768</td><td>0.02927480</td><td>0.02144022</td></tr><tr><td>8B</td><td>0.02302219</td><td>0.02052219</td><td>0.02195328</td><td>0.02002288</td></tr><tr><td rowspan="3"></td><td rowspan="3"></td><td rowspan="3">0.03093689</td><td>14B</td><td>0.02341767</td><td>0.02202470</td><td>0.02224608</td><td>0.02009201</td><td rowspan="3">0.03093721</td><td rowspan="3">verified [31]</td></tr><tr><td>32B-FP8</td><td>0.02769717 0.02261399</td><td>0.01762215</td><td>0.02230236</td><td>0.02145342</td></tr><tr><td>8B</td><td></td><td>0.01414730</td><td>0.01414730</td><td>0.01414730</td></tr><tr><td rowspan="3"></td><td rowspan="3"></td><td rowspan="3">0.02783557</td><td>14B</td><td>0.02229207</td><td>0.01807289</td><td>0.01980031</td><td>0.01762215</td><td rowspan="3"></td><td rowspan="3">verified [31]</td></tr><tr><td>32B-FP8 8B</td><td>0.02232631 12.94885</td><td>0.01893828 18.00000</td><td>0.01980032 13.20266</td><td>0.01807289 13.15947</td></tr><tr><td></td><td>12.89897</td><td>18.00000</td><td></td><td>14.20428</td></tr><tr><td rowspan="3"></td><td rowspan="3"></td><td rowspan="3"> $( d _ { \mathrm { { m a x } } } / d _ { \mathrm { { m i n } } } ) ^ { 2 } \downarrow$  12.88927 4.165850</td><td>14B</td><td>12.88956</td><td>18.00000</td><td>13.01783 13.27883</td><td>17.01391</td><td rowspan="3"></td><td rowspan="3">verified [21]</td></tr><tr><td>32B-FP8 8B</td><td>4.177907</td><td>4.732051</td><td>4.546903</td><td>4.474247</td></tr><tr><td>14B</td><td>4.167784</td><td>4.732051</td><td>4.586633</td><td>4.352043</td></tr><tr><td rowspan="3"></td><td rowspan="3"></td><td rowspan="3">1.505294</td><td>32B-FP8</td><td>4.166486</td><td>4.540188</td><td>4.370018</td><td>4.630033</td><td rowspan="3"></td><td rowspan="3"></td></tr><tr><td>8B</td><td>1.512329</td><td>1.515894</td><td>1.516525</td><td>1.516662</td></tr><tr><td>14B</td><td>1.512125</td><td>1.515684</td><td></td><td>1.517188</td></tr><tr><td rowspan="3">Autocorrelation II</td><td rowspan="3">C2↑</td><td rowspan="3">0.8962799</td><td>32B-FP8</td><td>1.512209</td><td>1.516507</td><td>1.515220 1.510459</td><td>1.516020</td><td rowspan="3"></td><td rowspan="3">verified [65]</td></tr><tr><td>8B</td><td>0.9213998†</td><td>0.9034590†</td><td>0.8995052†</td><td>0.9032550†</td></tr><tr><td></td><td>0.9314238†</td><td>0.9198818†</td><td>0.9064627†</td><td></td></tr><tr><td rowspan="3">Autocorrelation III</td><td rowspan="3">C3↓</td><td rowspan="3">1.468762</td><td>14B 32B-FP8</td><td>0.9289826†</td><td>0.8879468</td><td>0.9042086†</td><td>0.9001606† 0.8880995</td><td rowspan="3">0.9635881</td><td rowspan="3">verified [21]</td></tr><tr><td>8B</td><td>1.460021†</td><td>1.513955</td><td>1.478389</td><td>1.477685</td></tr><tr><td>14B</td><td>1.460180†+</td><td>1.473397</td><td>1.468036†</td><td>1.476167</td></tr><tr><td rowspan="3">Uncertainty</td><td rowspan="3">C₄↓</td><td rowspan="3">0.3215872</td><td>32B-FP8</td><td>1.459198†</td><td>1.469457</td><td>1.468273†</td><td>1.478609</td><td rowspan="3"></td><td rowspan="3">verified [77]</td></tr><tr><td>8B 14B</td><td>0.3524471 0.3525058</td><td>0.3566070 0.3560237</td><td>一</td><td>0.3615746 0.3531502</td></tr><tr></table>

## F Best Solutions Found by Each Framework

We compare the best program that each framework found with Qwen3 8B for one packing problem, one geometry problem, and the three PDE problems. For each framework, the highest-scoring program is selected. For the PDE problems, we plot the pointwise absolute error $| u - u _ { \mathrm { r e f } } | ,$ , where u is each framework’s solution and $u _ { \mathrm { r e f } }$ is the reference solution, on the same logarithmic scale for all three problems.

## F.1 Circle Packing in the Unit Square

Supplementary Figure 16 shows the best packing of each framework.

Our method. The program runs ten restarts of one SLSQP optimization over all centers and radii, each from 16 perturbed lattice positions and 10 random positions, and keeps the best valid packing.

OpenEvolve. The program places one circle at the center, eight on an inner ring, four in the corners, and thirteen on an outer ring, and optimizes all centers and radii with one SLSQP run.

ShinkaEvolve. The program places one circle at the center, eight on an inner ring, thirteen on an outer ring clipped to the square, and four near the corners, and optimizes all centers and radii with one SLSQP run.

CodeEvolve. The program places 25 circles on a 5 × 5 lattice and one at the center, and optimizes all centers and radii with one SLSQP run.

Ours  
![](images/d0b1ed420300df991439faa733af2b9ff9801dc1f9dc39990ff70621ec64783b.jpg)

OpenEvolve  
![](images/46502301498053c5316d8e883fee115eb715942bd7d9a39f9f177086012df8db.jpg)

ShinkaEvolve  
![](images/cd9d556c0911d1932c596463d12693802245fdcf9dd7945cce46cf1571c58150.jpg)

CodeEvolve  
![](images/9b954d6267fd246843252e1b777c69abaf297e7b9292c102ce6f9e05139d6580.jpg)  
Supplementary Figure 16: Best packings of 26 circles in the unit square. The sums of the radii are 2.636 (our method), 2.615 (OpenEvolve), 2.626 (ShinkaEvolve), and 2.614 (CodeEvolve).

## F.2 Heilbronn Problem in a Triangle

Supplementary Figure 17 shows the best placement of each framework.

Our method. The program runs ten restarts of simulated annealing from a symmetric lattice of points, moving one point at a time and accepting a worse smallest area with a probability that decreases as the temperature drops.

OpenEvolve. The program draws 11 random points in the triangle, redrawing a point whenever it is collinear with two earlier points.

ShinkaEvolve. The program draws random configurations from 50 fixed seeds, keeps the best ones, and refines them with SLSQP on the smallest triangle area under the three edge constraints of the triangle.

CodeEvolve. The program draws 11 random points in the triangle and runs one simulated annealing of 50,000 steps that moves one point at a time.

![](images/28e4e47689b410b7a57e6deb2b9e3bc0d265158fc8c94f4622c710635ab63058.jpg)

Supplementary Figure 17: Best placements of 11 points in the equilateral triangle. The shaded triangle has the smallest area among all triangles formed by the points. The smallest areas, divided by the area of the triangle, are 0.0223 (our method), 0.0016 (OpenEvolve), 0.0200 (ShinkaEvolve), and 0.0222 (CodeEvolve).

## F.3 Viscous Burgers Equation

Supplementary Figure 18 shows the solution of each framework for the smallest viscosity and its error.

Our method. The program uses central finite differences on 8192 grid points for $\nu < 0 . 0 1$ and 2048 points otherwise, with a two-stage Runge–Kutta method and a time step limited by both the advection and the diffusion stability conditions.

OpenEvolve. The program uses central finite differences of the flux $u ^ { 2 } / 2$ on 2048 grid points with forward Euler time steps.

ShinkaEvolve. The program splits each time step into a two-stage Runge–Kutta step for the advection on 1024 grid points and an implicit diffusion step solved exactly in Fourier space.

CodeEvolve. The program uses central finite differences of the flux $u ^ { 2 } / 2$ on 1024 grid points with the explicit midpoint method, a two-stage Runge–Kutta method, and a time step limited by both the advection and the diffusion stability conditions.

![](images/1965bed877926bc31648847bab1d700af699402dc766820034d645a30dede700.jpg)

![](images/f069f472951d658b3fded34ab167f4b35a9b6707db175c3ec955e8e6e809431f.jpg)  
Supplementary Figure 18: Solutions $u ( x , 1 )$ of the Burgers equation for $\nu = 0 . 0 0 1$ (top) and their absolute errors (bottom). Our solution is the most accurate away from the shock at $x = 0$

## F.4 Convection-Diffusion Equation

Supplementary Figure 19 shows the solution of each framework for $\varepsilon = 1 0 ^ { - 3 }$ and its error.

Our method. The program uses central finite differences on a $7 6 8 \times 7 6 8$ grid with a sparse direct solver and bilinear interpolation to the evaluation points.

OpenEvolve. The program uses central finite differences on a $5 1 2 \times 5 1 2$ grid with a sparse direct solver and bilinear interpolation.

ShinkaEvolve. The program uses central finite differences on a 512 × 512 grid with a sparse direct solver and bilinear interpolation.

CodeEvolve. The program uses central finite differences on a 128 × 128 grid with a sparse direct solver and bilinear interpolation.

![](images/c339e29d0a5c7b9989c9cdb9ab7219dbbcbc47faf4c31737902e137af1f7cfd3.jpg)  
Supplementary Figure 19: Solutions of the convection-diffusion equation for $\varepsilon = 1 0 ^ { - 3 }$ (top) and their absolute errors (bottom). The errors concentrate in the boundary layers along x = 1 and y = 1 and along the characteristic line from the corner (0, 0). The coarser grid of CodeEvolve spreads the error near $x = 1$ over a wider region.

## F.5 Helmholtz Equation

Supplementary Figure 20 shows the solution of each framework for k = 40 and its error.

Our method. The program uses second-order finite differences with ghost points for the impedance condition on an $8 1 8 \times { \bar { 8 1 8 } } { \bar { \mathrm { g r i d } } }$ , a sparse direct solver, and bilinear interpolation.

OpenEvolve. The program uses the same discretization on a 301 × 301 grid.

ShinkaEvolve. The program uses the same discretization on a $6 0 1 \times 6 0 1 \mathrm { g r i d }$

CodeEvolve. The program uses the same discretization on a $4 0 1 \times 4 0 1$ grid and smooths the real and imaginary parts of the solution with a Gaussian filter.

![](images/1c9bf6a606cc8030242683a79f2c54e71e9259000d4db698d7ee1f2eed521a7b.jpg)  
Supplementary Figure 20: Real parts of the solutions of the Helmholtz equation for k = 40 (top) and their absolute errors (bottom). The error decreases with the number of grid points per wavelength.

## G Evolution Trajectories

Supplementary Figure 21 plots the best score so far against wall-clock minutes for each problem and frame work with Qwen3 8B, and Supplementary Figures 22 and 23 show the same with Qwen3 14B and 32B-FP8. Each line averages the three runs of a framework, and each band spans its lowest and highest run. With every model, the curves of SwarmEvolve rise most steeply in the first minutes, when its rounds explore many focuses at once, and end the hour highest on most problems. Its bands often lie entirely above those of the baselines, so its lead holds for every seed.

![](images/c6b1d15731a6eed0c25a52c4b316bc9f42d7602f442f676ea28705adff6f0256.jpg)  
Supplementary Figure 21: Best score so far against wall-clock minutes for each problem with Qwen3 8B. Each line is the best score so far averaged over three runs, and each band spans the lowest and highest run. SwarmEvolve ends the hour with the highest score on 18 of the 21 problems, and its curves rise most steeply in the first few minutes.

![](images/ed6c2fc0befa244975b8acc460e9afb482f1b99910d80aedfe84fdf782ca1182.jpg)  
Supplementary Figure 22: Best score so far against wall-clock minutes for each problem with Qwen3 14B. Each line is the best score so far averaged over three runs, and each band spans the lowest and highest run. SwarmEvolve ends the hour with the highest or tied-highest score on 16 of the 21 problems.

![](images/86bef710a7aef0e7510037bc40162d4f668d58145299200d3a372a68036c6068.jpg)  
Supplementary Figure 23: Best score so far against wall-clock minutes for each problem with Qwen3 32B-FP8. Each line is the best score so far averaged over three runs, and each band spans the lowest and highest run. SwarmEvolve ends the hour with the highest or tied-highest score on 17 of the 21 problems.