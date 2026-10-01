# MULTI-LLM COLLABORATIVE ALIGNMENT VIA STACKELBERG GAMES

Christina Hahn<sup>1,</sup>∗, Shangbin Feng<sup>1,</sup>∗, Dean Light<sup>1</sup>, Swastik Roy<sup>2</sup>, Hila Gonen<sup>3</sup>, Yulia Tsvetkov<sup>1</sup>   
<sup>1</sup>University of Washington <sup>2</sup>Amazon <sup>3</sup>University of British Columbia   
chahn317@uw.edu, shangbin@cs.washington.edu

## ABSTRACT

A pool of language models can collaborate and improve collectively by learning from one another’s responses. These interactions depend on the instructions used during training. Existing methods typically sample instructions uniformly, even though their usefulness may change as the models improve: an instruction on which models’ responses once differed in quality may later be answered equally well, while a previously difficult instruction may begin to provide a useful learning signal. We propose STACKELBERG ALIGNMENT, a game-theory-inspired leader– follower framework that turns instruction selection into an adaptive curriculum. An EXP3 bandit acts as the leader, allocating a fixed sampling budget across instructions and updating its sampling distribution using a reward that combines instruction difficulty and response discriminability. The language models act as followers: they respond to the selected instructions, evaluate one another’s responses, and learn from the resulting preference signals through DPO or GRPO. The framework uses Elo-style reputation-weighted peer judgment and reputationbased opponent matching to support reliable and competitive model interactions. Experiments across three heterogeneous model pools and 12 benchmarks spanning scientific discovery, reasoning, code, instruction following, and knowledge show that STACKELBERG ALIGNMENT achieves the highest macro-average across three diverse model pools, outperforming the strongest training-time baseline by up to 7.4% and the best static inference baseline by 12–25%. Analysis confirms that the adaptive leader concentrates duels on the most informative instructions, and ablations show that both reputation-weighted judgment and reputation-based matching improve the effectiveness of multi-LLM evolution.

## 1 INTRODUCTION

Model collaboration allows language models to benefit from one another’s complementary capabilities, and through inference-time interactions produce outputs better than any individual model (Du & Kaelbling, 2024; Feng et al., 2026a). Training-time collaboration offers a more fundamental opportunity: models can improve by interacting with an environment of other models, generating training signals that neither could produce alone. Recent methods often achieve this through comparison and competition (Luo et al., 2024; Subramaniam et al., 2025). Sparta Alignment (Jiang et al., 2025), for example, has pairs of models respond to the same instruction, the remaining models judge the responses, and the winning pair is used for preference learning, reliably lifting all models in the pool across reasoning, code, and open-ended tasks. It further maintains reputation scores that weight peer judgments and guide opponent matching; we build on top of this framework. A key ingredient to all of these methods is, however, left unexplored: which instructions should models compete on. Existing approaches sample instructions uniformly throughout training with fixed schedule, rather than treating instruction selection as an adaptive component of the training loop.

Uniform instruction sampling is inefficient because instruction informativeness is non-stationary and heterogeneous. Instructions where all models already agree yield no discriminative preference signal; instructions that all models fail on produce only tied scores with no learning gradient; the most valuable instructions are those at thefrontier ofdisagreement, where duels and pairwise comparisons generate learnable preference pairs. This frontier shifts as models improve across iterations: an instruction that drives rich learning early in training may become trivial later, while harder instructions become tractable only after the pool has strengthened. A fixed sampling distribution cannot track these shifts, wasting a large fraction of compute on uninformative instructions. Adaptive instruction selection, informed by the evolving capabilities of the model pool, is therefore central to efficient collaborative alignment and training-time collaboration in general.

![](images/f47d796ffc033098042e2cc7847a96e7e41dc0a67fff9158dbac8d4437337646.jpg)  
Figure 1: Overview of STACKELBERG ALIGNMENT for one iteration. The EXP3 leader maintains learned weights $\mathbf { w } ^ { t }$ over the instruction pool and selects instruction $x _ { k }$ non-uniformly. Two follower models $( M _ { 3 } , M _ { 4 }$ in the figure) are matched by reputation and duel on $x _ { k } ,$ generating responses y<sub>3</sub> and $y _ { 4 }$ . Remaining models act as peer judges, scoring both responses; scores are aggregated weighted by judge reputation, determining the winner. The Update step performs three actions: (1) reputation scores are adjusted, (2) leader weights are updated via EXP3 based on duel informativeness, and (3) the winning preference pair is added to the dataset P for DPO training, or the selected instruction and online judge scores are used directly as the reward for GRPO training. Dashed arrows show the resulting feedback loop: the updated leader weights reshape the instruction distribution, and the updated reputations and retrained models feed back into the follower pool for the next iteration.

We frame this observation as a bilevel Stackelberg game (von Stackelberg & Peacock, 1952; Bas¸ar & Olsder, 1998) between two classes of players. A leader controls a distribution over the instruction pool and commits to it before any model generates a response; its objective is to concentrate compute on instructions that currently lie at the frontier of model disagreement. The followers are the LLMs in the pool: they observe the selected instruction, duel, and are trained on the resulting preference pairs. The leader implements this via EXP3 (Auer et al., 2002), a no-regret bandit algorithm well suited to non-stationary reward sequences, with a composite reward that captures both instruction difficulty and duel discriminability. Each model also maintains a reputation score reflecting its track record in past duels; models not part of a duel act as peer judges whose votes are weighted by their own reputations, and opponents are matched by reputation proximity so that duels remain competitive as skill levels diverge across training. The resulting method, STACKELBERG ALIGNMENT, is the first instantiation of collaborative LLM alignment with an adaptive leader that learns the instruction distribution jointly with the follower models, yielding a dynamic curriculum that concentrates compute on instructions where the pool still disagrees and shifts automatically as models strengthen.

We evaluate STACKELBERG ALIGNMENT on three pools of heterogeneous LLMs spanning specialized expert models, diverse academic research models, and general-purpose models, across 12 benchmarks covering scientific discovery, reasoning, code, instruction following, and knowledge. STACKELBERG ALIGNMENT achieves the highest macro-average in all three pools, outperforming other training-time algorithms by up to 7.4% and the best static inference baseline by 12–25%. Our main contributions are: (1) a Stackelberg game formulation of training-time multi-LLM collaborative alignment; (2) an EXP3 bandit leader with a composite duel-informativeness reward that tracks the shifting capability frontier of the model pool; and (3) an empirical demonstration of consistent gains across three diverse model pools and twelve benchmarks, with analysis tracing the gains to the adaptive curriculum.

## 2 METHOD

Figure 1 illustrates one complete training iteration of STACKELBERG ALIGNMENT.

Setup. Let $\mathcal { M } ^ { 0 } = \{ M _ { 1 } ^ { 0 } , . . . , M _ { m } ^ { 0 } \}$ be a pool of m language models and $\mathcal { X } = \{ x _ { 1 } , . . . , x _ { K } \}$ a dataset of instructions. Our goal is to produce an improved pool $\mathcal { M } ^ { T }$ by having models interact, evaluate, and learn from each other across $T$ iterations. Models engage in pairwise duels on instructions: two models generate responses, the remaining models act as peer judges and score each response on a $[ s _ { \mathrm { m i n } } , s _ { \mathrm { m a x } } ]$ scale, and these scores are aggregated into preference pairs for RL finetuning. Each iteration runs D such duels, with the instruction for each duel drawn independently (with replacement) from the leader’s distribution $\pi _ { L }$ rather than sweeping the instruction pool exhaustively; a given instruction may therefore be drawn multiple times, once, or not at all within a single iteration, and only duels that are actually drawn contribute preference pairs or reward signal to that iteration’s training. This is an intentional design choice rather than an incidental side effect: concentrating duels on the currently most informative instructions, instead of exhausting the pool every iteration, is exactly the adaptive curriculum this framework is designed to realize (§2.1). Because the leader’s sampling distribution keeps a nonzero floor probability for every instruction (Eq. 1), no instruction is permanently excluded, though there is no guarantee that every instruction in the pool is drawn within any single iteration or even over the full training run. Following Jiang et al. (2025), each model $M _ { i }$ carries a reputation score $R _ { i } \in \mathbb { R }$ and a reputation deviation $\sigma _ { i }$ , initialized uniformly. Reputation encodes each model’s perceived reliability, as judged by its peers over training, and serves two roles: it governs opponent selection, and weights peer judgments during score aggregation.

A Stackelberg view of training-time model collaboration. Existing model collaboration approaches (Du et al., 2024; Subramaniam et al., 2025; Wang et al., 2025a; Jiang et al., 2025) treat instruction selection as fixed, sampling uniformly from the training pool regardless of what the current model pool still needs to learn. However, instructions vary widely in difficulty and informativeness: some are already mastered by all models and yield no useful gradient, others are beyond reach and produce no meaningful preference signal, and the most valuable ones lie at the frontier of disagreement where duels generate genuine learning signal. Critically, as models collaborate and improve across iterations, this frontier shifts, making the optimal instruction distribution non-stationary. An adaptive curriculum that tracks these shifts is therefore essential for efficient collaborative alignment. We frame the duel-based training loop above as a bilevel Stackelberg game (von Stackelberg & Peacock, 1952) between two classes of players.

The leader L controls a distribution $\pi _ { L } \in \Delta ( { \mathcal { X } } )$ over the instruction pool and commits to it before any model generates a response. Its strategic objective is to select the instructions that produce the most informative learning signal. The followers $\{ M _ { i } \} _ { i = 1 } ^ { m }$ observe the selected instruction $x \sim \pi _ { L }$ and respond by generating outputs, which are then evaluated and used for training. This sequential commitment structure (leader first, followers second) defines a Stackelberg game (Bas¸ar & Olsder, 1998), and the leader’s optimal strategy anticipates the followers’ most informative responses.

This framework subsumes several prior alignment paradigms as special cases. When $\pi _ { L }$ is fixed to uniform and $m = 1 .$ , it recovers single-model self-alignment methods such as Self-Rewarding (Yuan et al., 2025) and SPIN (Chen et al., 2024). When $\pi _ { L }$ is uniform but $m > 1$ with pairwise competition, it recovers SPARTA ALIGNMENT (Jiang et al., 2025): we adopt its reputation-weighted peer judgment and Elo-style reputation updates directly (§2.3), and extend its reputation-based opponent matching with a time-varying gap schedule (§2.2). STACKELBERG ALIGNMENT is the first instantiation of the full framework with an adaptive leader that learns $\pi _ { L }$ jointly with the followers.

Algorithm 1 summarizes the full procedure; the subsections below detail each component.

## 2.1 STACKELBERG LEADER: ADAPTIVE INSTRUCTION SELECTION

Instruction sampling distribution. At each training iteration $t \in \{ 1 , \ldots , T \}$ , the leader must pick which of the K candidate instructions $\mathcal { X } = \{ x _ { k } \} _ { k = 1 } ^ { K ^ { - } }$ to use for that iteration’s duels; $p _ { k } ^ { t }$ denotes the probability of sampling instruction $x _ { k }$ at iteration $t ,$ and instructions that recently produced informative duels should be sampled more often. To this end, the leader maintains a weight vector $\mathbf { w } ^ { t } = ( w _ { 1 } ^ { t } , \ldots , w _ { K } ^ { t } ) \in \mathbb { R } _ { > 0 } ^ { K }$ and induces the sampling distribution $p _ { k } ^ { t }$ via EXP3 (Auer et al., 2002):

$$
p _ { k } ^ { t } = ( 1 - \gamma ) \frac { w _ { k } ^ { t } } { \Vert \mathbf { w } ^ { t } \Vert _ { 1 } } + \frac { \gamma } { K } ,\tag{1}
$$

where $\gamma \in [ 0 , 1 ]$ is the exploration rate. The $( 1 - \gamma )$ term exploits high-weight instructions while the $\gamma / K$ term encourages that all instructions are visited. EXP3 is fitting here because instruction rewards are non-stationary, meaning an instruction’s informativeness is not fixed but drifts over the course of training: as models improve across iterations, the same instruction may yield rich signal early and none later. Many bandit algorithms assume each instruction’s reward is drawn from a fixed distribution and can perform poorly once that assumption breaks; EXP3 instead offers a no-regret guarantee, i.e., its cumulative reward stays close to that of the best fixed instruction in hindsight, even against such adversarially shifting reward sequences.

Algorithm 1 STACKELBERG ALIGNMENT   
Require: Model pool $\mathcal { M } ^ { 0 }$ , instruction set $\mathcal { X } = \{ x _ { k } \} _ { k = 1 } ^ { K } ,$ , iterations $T ,$ duels per iteration D   
1: Initialize reputation $R _ { i }  R _ { 0 } , \sigma _ { i }  \sigma _ { 0 }$ for all i; leader weights $\mathbf { w } ^ { 0 }  \mathbf { 1 } _ { K } ^ { \cdot }$   
2: for $t = 1 , \cdot \cdot . , T$ do   
3: Judged duels ${ \mathcal { I } }  \emptyset ;$ preference pairs $\mathcal { P }  \emptyset$   
4: for each duel $d = 1 , \hdots , D$ do   
5: $x \sim \pi _ { L } ( \mathbf { w } ^ { t - 1 } )$ ▷ Leader commits to instruction   
6: Select active model $M _ { i } ^ { t } ;$ draw opponent $M _ { i ^ { \prime } . } ^ { t } \sim p _ { \mathrm { { m a t c h } } } ( \cdot \mid M _ { i } ^ { t } , \{ R _ { j } \} , t )$ ▷ Eq. 7   
7: Combat: generate $y _ { i } \stackrel { . } {  } M _ { i } ^ { t } ( x ) , \stackrel { . } { y } _ { i ^ { \prime } }  \bar { M } _ { i ^ { \prime } } ^ { t } ( x )$   
8: Judge: each non-combatant $M _ { k }$ scores responses: $s _ { i } ^ { ( k ) } , s _ { i ^ { \prime } } ^ { ( k ) } \in [ s _ { \operatorname* { m i n } } , s _ { \operatorname* { m a x } } ]$ ▷ §2.3   
9: ${ \bar { s } } _ { i } , { \bar { s } } _ { i ^ { \prime } } \gets \mathrm { A g g r e g a t e } \big ( \{ s _ { i } ^ { ( k ) } , s _ { i ^ { \prime } } ^ { ( k ) } \} _ { k \notin \{ i , i ^ { \prime } \} } , \{ R _ { k } \} \big )$ ▷ Eq. 8   
10: Add $( x , y _ { i } , y _ { i ^ { \prime } } , \bar { s } _ { i } , \bar { s } _ { i ^ { \prime } } )$ to $\mathcal { T } ;$ update $\mathrm { \dot { \cal R } } _ { i } , \mathrm { \dot { \cal R } } _ { i ^ { \prime } } , \sigma _ { i } , \mathrm { \dot { \sigma } } _ { i ^ { \prime } }$ ▷ Eq. 9; see Appx. C   
11: If $\bar { s } _ { i } \neq \bar { s } _ { i ^ { \prime } } :$ add preference pair to   
12: end for   
13: Leader update: $\mathbf { w } ^ { t }  \mathrm { E X P 3 } \ –$ -Update $( \mathbf { w } ^ { t - 1 } , \mathcal { I } )$ ▷ uses all judged duels   
14: Train (DPO): $M _ { i } ^ { t } \gets \mathrm { D P O } ( M _ { i } ^ { t - 1 } , \mathcal { P } )$ for all i   
15: or Train (GRPO): $M _ { i } ^ { t } \gets \dot { \mathrm { G R P O } } ( M _ { i } ^ { t - 1 } , \{ ( x , y _ { i } , y _ { i ^ { \prime } } ) \} \in \mathcal { I } , \{ R _ { k } \} )$ for all i ▷ online peer rewards   
16: end for   
17: return Improved model pool $\mathcal { M } ^ { T }$

Leader reward. After a duel on instruction $x _ { k }$ with peer-aggregated scores $\bar { s } _ { i } , \bar { s } _ { i ^ { \prime } } \in [ s _ { \operatorname* { m i n } } , s _ { \operatorname* { m a x } } ]$ (Eq. 8), the leader receives a composite reward $r _ { k } \in [ 0 , 1 ]$ , which is fed into the EXP3 weight update below to raise or lower how often instruction $x _ { k }$ is sampled in future iterations:

(i) Difficulty reward. Among instructions that are learnable (score above a floor threshold), the leader prefers harder ones: instructions where models score perfectly are already mastered and offer no room for improvement, while those just above the threshold are most informative. For a model’s aggregate score s this is:

$$
r _ { \mathrm { d i f f } } ( s ) = \left\{ \begin{array} { l l } { \displaystyle \frac { s _ { \mathrm { m a x } } - s } { s _ { \mathrm { m a x } } - \tau _ { \mathrm { t h } } } } & { s \geq \tau _ { \mathrm { t h } } } \\ { 0 } & { s < \tau _ { \mathrm { t h } } } \end{array} \right.\tag{2}
$$

where $\tau _ { \mathrm { t h } }$ is a minimum score threshold below which the instruction is too hard to yield useful signal.

(ii) Preference quality reward. For preference learning, the gap between chosen and rejected responses matters: deviations from the scheduled target in either direction are penalized: a gap of zero provides no learning signal, and a gap far above the target indicates the comparison is already trivially decided (Yao et al., 2024). We schedule a target gap $g _ { t } ^ { * }$ that starts large (clear preference distinctions early) and decays (finer distinctions later):

$$
g _ { t } ^ { * } = g _ { \mathrm { e n d } } + \left( g _ { \mathrm { s t a r t } } - g _ { \mathrm { e n d } } \right) ( 1 - \tau ) , \qquad \tau = \frac { t } { T - 1 } ,\tag{3}
$$

and reward proximity to this target via a Gaussian kernel:

$$
r _ { \mathrm { p r e f } } \big ( \bar { s } _ { i } , \bar { s } _ { i ^ { \prime } } \big ) = \mathrm { e x p } \left( { - \frac { \left( \left| \bar { s } _ { i } - \bar { s } _ { i ^ { \prime } } \right| / \left( s _ { \mathrm { m a x } } - s _ { \mathrm { m i n } } \right) - g _ { t } ^ { * } \right) ^ { 2 } } { 2 \sigma _ { r } ^ { 2 } } } \right) .\tag{4}
$$

where $\sigma _ { r }$ is the kernel bandwidth. The instruction reward is computed per combatant and averaged: $\begin{array} { r } { r _ { k } = \frac { 1 } { 2 } \dot { \sum } _ { s \in \{ \bar { s } _ { i } , \bar { s } _ { i ^ { \prime } } \} } \left( \lambda _ { 1 } r _ { \mathrm { d i f f } } ( s ) + \lambda _ { 2 } r _ { \mathrm { p r e f } } ( \bar { s } _ { i } , \bar { s } _ { i ^ { \prime } } ) \right) } \end{array}$ , with $\lambda _ { 1 } + \lambda _ { 2 } = 1$

EXP3 weight update. To correct for the bias introduced by non-uniform sampling, we apply the standard importance-weighted multiplicative update:

$$
\log w _ { k } ^ { t + 1 } = \log w _ { k } ^ { t } + \frac { 1 } { K } \cdot \frac { r _ { k } \cdot \mathbf { 1 } [ x _ { k } \sim \pi _ { L } ^ { t } ] } { p _ { k } ^ { t } } ,\tag{5}
$$

followed by re-normalization, where $p _ { k } ^ { t }$ is from Eq. 1 and $1 [ x _ { k } \sim \pi _ { L } ^ { t } ]$ indicates whether instruction $x _ { k }$ was sampled in the current duel. Instructions that yield high reward relative to their sampling probability receive increased weight, concentrating future sampling on informative prompts.

## 2.2 SCHEDULED FOLLOWER MATCHING

For each duel $d ,$ the active model $M _ { i } ^ { t }$ is selected in round-robin order. Sparta Alignment (Jiang et al., 2025) draws its opponent from $\mathcal { M } ^ { t } \setminus \{ M _ { i } ^ { t } \}$ by reputation proximity; we extend this with a scheduled soft-matching policy that targets a time-varying reputation gap, annealing from mismatched to near-peer opponents over training.

Define the normalized reputation gap between $M _ { i } ^ { t }$ and any candidate $M _ { j }$ as

$$
\hat { g } _ { i j } ^ { t } = \frac { | R _ { i } ^ { t } - R _ { j } ^ { t } | } { \operatorname* { m a x } _ { k \neq i } | R _ { i } ^ { t } - R _ { k } ^ { t } | } \in [ 0 , 1 ] .\tag{6}
$$

We select the opponent by sampling from a distribution peaked at a target gap $g _ { t } ^ { * }$

$$
p ( M _ { i ^ { \prime } } ^ { t } = M _ { j } ) \propto \exp \left( - \frac { ( \hat { g } _ { i j } ^ { t } - g _ { t } ^ { * } ) ^ { 2 } } { 2 \sigma _ { g } ^ { 2 } } \right) , \quad j \neq i ,\tag{7}
$$

where $\sigma _ { g }$ is the selection bandwidth and $g _ { t } ^ { * }$ follows Eq. 3 with $g _ { \mathrm { s t a r t } } = 1 , g _ { \mathrm { e n d } } = 0$ , giving $g _ { t } ^ { * } = 1 - \tau$ which linearly decreases from 1 (early: mismatched opponents) to 0 (late: near-peer opponents).

This schedule has a curriculum interpretation. As reputations diverge over training, the early target of $g ^ { * } \approx 1$ steers the system toward mismatched opponents: the stronger model nearly always wins, producing clear preference pairs that provide clean supervision for improvement (Sun et al., 2024). Later, as the target shifts toward $g ^ { * } \approx 0$ , near-peer duels produce finer-grained preference distinctions that drive precise refinement, analogous to competitive matchmaking in human rating systems (Ebtekar & Liu, 2021).

## 2.3 PEER JUDGMENT AND REPUTATION SYSTEM

Following Jiang et al. (2025), all non-competing models evaluate both combat responses. Each judge $M _ { k } ^ { t } \in \mathcal { M } ^ { t } \setminus \{ M _ { i } ^ { t } , M _ { i ^ { \prime } } ^ { t } \}$ assigns scores $s _ { i } ^ { ( k ) } , s _ { i ^ { \prime } } ^ { ( k ) } \in [ s _ { \operatorname* { m i n } } , s _ { \operatorname* { m a x } } ]$ (a 1–10 scale, i.e., $s _ { \operatorname* { m i n } } = 1$ $s _ { \operatorname* { m a x } } = 1 0 )$ via an LLM-as-a-judge prompt (Zheng et al., 2023). The peer-aggregate score is a reputation-weighted mean:

$$
\begin{array} { r } { \bar { s } _ { i } = \left( \sum _ { k \notin \{ i , i ^ { \prime } \} } R _ { k } ^ { t } s _ { i } ^ { ( k ) } \right) \Big / \left( \sum _ { k \notin \{ i , i ^ { \prime } \} } R _ { k } ^ { t } \right) . } \end{array}\tag{8}
$$

Weighting by $R _ { k } ^ { t }$ discounts the influence of low-reputation judges, so that models with stronger track records contribute more to each score aggregation (Wang et al., 2025b).

Reputation update. Reputations are updated after each duel via an adapted Elo rule (Ebtekar & Liu, 2021), as in Sparta Alignment (Jiang et al., 2025), that incorporates three factors:

$$
R _ { i }  R _ { i } + ( K _ { t } / \lambda ) \cdot \underbrace { ( \bar { s } _ { i } - \bar { s } _ { i ^ { \prime } } ) } _ { \mathrm { s c o r e ~ g a p } } \cdot \underbrace { \operatorname { t a n h } ( \sigma _ { i } ) } _ { \mathrm { s t a b i l i t y } } \cdot \underbrace { \operatorname { m a x } \Bigl ( | \Phi ( z _ { i } ) - \Phi ( z _ { i ^ { \prime } } ) | , \epsilon \Bigr ) } _ { \mathrm { g a p ~ f a c t o r } } ,\tag{9}
$$

where $z _ { i } = ( R _ { i } - R _ { i ^ { \prime } } ) / \sqrt { \sigma _ { i } ^ { 2 } + \sigma _ { i ^ { \prime } } ^ { 2 } }$ , Φ is the standard normal $\mathrm { C D F } , \epsilon > 0$ is a floor preventing stagnation in near-tied duels, and λ is a scaling constant. Reputations are floored at a fixed minimum to prevent degenerate negative drift. The K factor decays exponentially over update count to cool large swings as reputations stabilize; $\sigma _ { i }$ is the standard deviation of recent reputation deltas over a sliding window, measuring volatility. Full formulas are in Appendix C.

## 2.4 FOLLOWER TRAINING

STACKELBERG ALIGNMENT supports two training settings for collaborative improvement. For DPO, non-tied preference pairs from all duels are collected and used for offline fine-tuning. For GRPO, the duel assignment structure determines each model’s training prompts, with peer judges scoring completions online during training.

DPO. Non-tied preference pairs $( x , y _ { w } \succ y _ { l } )$ collected across all duels are used to fine-tune each model via Direct Preference Optimization (Rafailov et al., 2024), with the previous iteration’s checkpoint as the reference policy.

GRPO. As an online alternative, the duel structure determines each model’s training prompts. Each model generates G completions per assigned prompt; peer judges score each completion online, and the normalized peer score serves as the per-completion reward in the GRPO objective (Shao et al., 2024). Reputation updates are derived from the logged rewards after training completes.

Connection to the Stackelberg equilibrium. At convergence, the leader’s instruction distribution $\pi _ { L } ^ { * }$ maximizes expected instructional informativeness under the followers’ best-response policies $\{ \pi _ { i } ^ { * } \}$ ; the followers in turn are optimal for the instruction distribution $\pi _ { L } ^ { * }$ generates. This joint fixed point corresponds to a Stackelberg equilibrium (Bas¸ar & Olsder, 1998), where neither the leader can improve its choice of instructions given the followers’ behavior, nor can the followers improve their policies given the leader’s instruction distribution. In practice we approximate this equilibrium via alternating updates: EXP3 for the leader and DPO/GRPO for the followers.

## 3 EXPERIMENT SETTINGS

Models. We employ three model pools of varying scale and architecture for evaluation.

Pool 1 contains 9 heterogeneous models with five Qwen2.5-7B variants (the base instruction-tuned model plus four domain fine-tunes on OASST1, ShareGPT, WizardLM, and biomedical corpora, in Jiang et al. (2025)), alongside AgentFlow-Planner-7B (Li et al., 2026), Llama-3.1-8B-Instruct (Dubey et al., 2024), Aya-Expanse-8B (Dang et al., 2024), and Pangea-7B (Yue et al., 2025). This pool spans diverse model families, training corpora, and areas of expertise.

Pool 2 contains 8 models trained in diverse academic research projects, where researchers contribute specialized models as modular components to a collaborative system. We use the top-8 models sourced in Feng et al. (2026b) (see Appendix D for model identities).

Pool 3 contains 4 general models: Qwen3.5-9B (Qwen Team, 2026), Gemma-4-12B-Instruct (Team et al., 2026), Nemotron-3-Nano-4B (Blakeman et al., 2025), and Phi-4 (Abdin et al., 2024).

Baselines. We compare STACKELBERG ALIGNMENT in two training configurations, Stackelberg (DPO) and Stackelberg (GRPO), against eight baselines spanning static inference and training-based collaboration.

Static inference: Majority Vote, ensemble plurality voting (applicable to objective tasks); MoA (Wang et al., 2025a), iterative synthesis and aggregation of multiple models’ responses; Multiagent Debate (Du et al., 2024), iterative response refinement at inference time; Heterogeneous Swarms (Feng et al., 2025a), which optimizes directed acylic graphs of model interactions for collaboration. Training-based: Sparta Alignment (Jiang et al., 2025), the direct predecessor with a uniform instruction leader and DPO-only training; Multiagent Fine-tuning (Subramaniam et al., 2025), joint fine-tuning via debate-based data augmentation; Trained Router (Ong et al., 2025), which learns to route queries to the best model via supervised fine-tuning on validation-set oracle labels; AggLM (Zhao et al., 2025), a GRPO-trained aggregator with verifiable rewards.

Hyperparameters. For Stackelberg (DPO) and Stackelberg (GRPO), we run $T = 8$ iterations with $D = 6 4$ duels per iteration. For the EXP3 leader we set exploration rate $\gamma = 0 . 2$ The difficulty reward uses score threshold $\tau _ { \mathrm { t h } } = 3 . 0$ , and the preference quality reward uses gap schedule $g _ { \mathrm { s t a r t } } = 0 . 6 , g _ { \mathrm { e n d } } = 0 . 1 5 , \sigma _ { r } = 0 . 1 5 ;$ the combined reward weights are $\lambda _ { 1 } = 0 . 3$ (difficulty) and $\lambda _ { 2 } = 0 . 7$ (preference quality). For opponent matching we use $\sigma _ { g } = 0 . 1 5$ and random override probability $p _ { \mathrm { r a n d } } = 0 . 2 .$ , with a candidate pool of top-3 by reputation proximity. The reputation system uses $K _ { 0 } = 1 0 , K _ { \mathrm { m i n } } = 5 ,$ decay rate $\alpha = 0 . 9$ , decay step size $n _ { s } = 1 0$ , scaling factor $\lambda = 2 0$ , and Elo floor $\epsilon = 0 . 0 1$ . For DPO training, we use learning rate $1 \times 1 0 ^ { - 6 }$ , 1 epoch per iteration, and effective batch size 16 with LoRA. For GRPO training, we use the same learning rate and epoch schedule with $G = 4$ rollouts per prompt; all experiments run on NVIDIA H100 GPUs.

Table 1: Performance across three model pools and 12 datasets grouped by domain. Bold: best per column within each pool; underline: second best; –: not applicable. Shaded rows ( ) are STACKELBERG ALIGNMENT variants. The Avg column is the macro-average over all available datasets, with AlpacaEval min-max normalized to [0, 1].
<table><tr><td></td><td colspan="3">Scientific</td><td colspan="2">Reasoning GPA-Dia</td><td colspan="2">Code</td><td colspan="2">Instruction Adyval</td><td colspan="2">Knowledge TruuldpA CulturalBench</td><td></td></tr><tr><td>Method</td><td>Bixnch</td><td>Laqbch</td><td>SMDD</td><td>AssayBench</td><td></td><td>ATH</td><td>HumanEval</td><td>MBPP</td><td></td><td>IEVvl</td><td></td><td>AVS</td></tr><tr><td colspan="9">Pool 1: Specialized Expert LLMs</td><td></td><td></td><td></td><td></td></tr><tr><td>Sparta Alignment</td><td>0.301</td><td>0.308</td><td>0.168</td><td>0.017</td><td>0.313</td><td>0.820 0.719</td><td>0.634</td><td>7.528</td><td>0.707</td><td>0.686</td><td>0.656</td><td>0.524</td></tr><tr><td>Majority Vote</td><td>0.184</td><td>0.272</td><td></td><td></td><td>0.263</td><td>0.778</td><td></td><td></td><td></td><td>0.616</td><td>0.670</td><td>0.464</td></tr><tr><td>AggLM</td><td>0.291</td><td>0.317</td><td>0.007</td><td>0.029</td><td>0.273</td><td>0.817</td><td>0.649 0.565</td><td>-2.647</td><td>0.469</td><td>0.605</td><td>0.591</td><td>0.407</td></tr><tr><td>Trained Router</td><td>0.233</td><td>0.263</td><td>0.011</td><td>0.015</td><td>0.313</td><td>0.735</td><td>0.623 0.511</td><td>1.718</td><td>0.631</td><td>0.605</td><td>0.633</td><td>0.428</td></tr><tr><td>Multiagent FT</td><td>0.223</td><td>0.316</td><td>0.187</td><td>0.019</td><td>0.273</td><td>0.729</td><td>0.640 0.579</td><td>5.785</td><td>0.625</td><td>0.585</td><td>0.682</td><td>0.475</td></tr><tr><td>Multiagent Debate</td><td>0.319</td><td>0.292</td><td>0.011</td><td>0.015</td><td>0.353</td><td>0.789</td><td>0.711 0.608</td><td>-6.553</td><td>0.567</td><td>0.571</td><td>0.732</td><td>0.414</td></tr><tr><td>Het. Swarms</td><td>0.194</td><td>0.264</td><td>0.000</td><td>0.033</td><td>0.364</td><td>0.789</td><td>0.597 0.495</td><td>-3.633</td><td>0.740</td><td>0.622</td><td>0.746</td><td>0.420</td></tr><tr><td>MoA</td><td>0.223</td><td>0.287</td><td>0.095</td><td>0.016</td><td>0.353</td><td>0.833</td><td>0.816 0.589</td><td>-3.848</td><td>0.707</td><td>0.582</td><td>0.729</td><td>0.451</td></tr><tr><td>Stackelberg (DPO)</td><td>0.233</td><td>0.311</td><td>0.204</td><td>0.024</td><td>0.364</td><td>0.857</td><td>0.763 0.643</td><td>8.081</td><td>0.749</td><td>0.682</td><td>0.747</td><td>0.548</td></tr><tr><td>Stackelberg (GRPO)</td><td>0.349</td><td>0.322</td><td>0.237</td><td>0.023</td><td>0.353</td><td>0.877</td><td>0.816 0.653</td><td>7.819</td><td>0.730</td><td>0.684</td><td>0.724</td><td>0.563</td></tr><tr><td colspan="9">Pool 2: LLMs from Diverse Academic Research</td><td></td><td></td><td></td></tr><tr><td>Sparta Alignment</td><td>0.272</td><td>0.330 0.287</td><td>0.216</td><td>0.019</td><td>0.303 0.263</td><td>0.853 0.825</td><td>0.789 0.628</td><td>6.273</td><td>0.672</td><td>0.632 0.650</td><td>0.680 0.657</td><td>0.533 0.471</td></tr><tr><td>Majority Vote AggLM</td><td>0.146 0.282</td><td>0.282</td><td>0.011</td><td>0.015</td><td></td><td></td><td></td><td>-1.835</td><td>0.416</td><td>0.551</td><td>0.582</td><td>0.341</td></tr><tr><td>Trained Router</td><td>0.214</td><td>0.261</td><td>0.170</td><td>0.015</td><td>0.283</td><td>0.727 0.439</td><td>0.382</td><td>3.066</td><td>0.649</td><td>0.614</td><td>0.654</td><td>0.476</td></tr><tr><td>Multiagent FT</td><td>0.233</td><td>0.339</td><td>0.156</td><td></td><td>0.343</td><td>0.826 0.702</td><td>0.616</td><td>3.564</td><td>0.623</td><td>0.624</td><td>0.683</td><td></td></tr><tr><td>Multiagent Debate</td><td></td><td></td><td></td><td>0.015</td><td>0.303</td><td>0.837 0.754</td><td>0.604</td><td></td><td></td><td></td><td></td><td>0.490</td></tr><tr><td></td><td>0.280</td><td>0.311</td><td>0.096</td><td>0.014</td><td>0.313</td><td>0.805</td><td>0.693 0.622</td><td>-2.921</td><td>0.583</td><td>0.574</td><td>0.353</td><td>0.387</td></tr><tr><td>Het. Swarms</td><td>0.311</td><td>0.303</td><td>0.182</td><td>0.028</td><td>0.374</td><td>0.884</td><td>0.693 0.598</td><td>0.440</td><td>0.721</td><td>0.638</td><td>0.689</td><td>0.482</td></tr><tr><td>MoA</td><td>0.252</td><td>0.310</td><td>0.164</td><td>0.016</td><td>0.414</td><td>0.826</td><td>0.737 0.616</td><td>-0.524</td><td>0.681</td><td>0.530</td><td>0.686</td><td>0.458</td></tr><tr><td>Stackelberg (DPO) Stackelberg (GRPO)</td><td>0.311 0.291</td><td>0.334 0.355</td><td>0.196 0.234</td><td>0.026 0.026</td><td>0.393 0.423</td><td>0.889 0.816</td><td>0.632 0.634</td><td>4.986 5.207</td><td>0.676 0.680</td><td>0.656 0.626</td><td>0.696 0.680</td><td>0.540 0.544</td></tr><tr><td colspan="9">0.884 0.816 Pool 3: General-Purpose LLMs</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Sparta Alignment Majority Vote</td><td>0.349 0.214</td><td>0.218 0.171</td><td>0.294</td><td>0.034</td><td>0.384</td><td>0.854</td><td>0.868 0.661</td><td>7.109</td><td>0.775</td><td>0.770</td><td>0.749</td><td>0.579 0.192</td></tr><tr><td>AggLM</td><td>0.330</td><td>0.346</td><td>0.120</td><td>0.031</td><td>0.313</td><td>0.857</td><td>0.535 0.550</td><td>-2.975</td><td>0.569</td><td>0.789</td><td>0.790</td><td>0.436</td></tr><tr><td>Trained Router</td><td>0.243</td><td>0.173</td><td>0.073</td><td>0.016</td><td>0.292</td><td>0.825 0.781</td><td>0.606</td><td>5.888</td><td>0.589</td><td>0.681</td><td>0.667</td><td>0.485</td></tr><tr><td>Multiagent FT</td><td>0.340</td><td>0.263</td><td>0.139</td><td>0.027</td><td>0.208</td><td>0.494 0.447</td><td>0.413</td><td>3.603</td><td>0.655</td><td>0.682</td><td>0.683</td><td>0.417</td></tr><tr><td>Multiagent Debate</td><td>0.307</td><td>0.152</td><td>0.348</td><td>0.027</td><td>0.125</td><td>0.890 0.614</td><td>0.626</td><td>-1.965</td><td>0.765</td><td>0.707</td><td>0.830</td><td>0.458</td></tr><tr><td>Het. Swarms</td><td>0.312</td><td>0.181</td><td>0.265</td><td>0.030</td><td>0.208</td><td>0.853 0.860</td><td>0.691</td><td>-2.623</td><td>0.736</td><td>0.357</td><td>0.803</td><td>0.444</td></tr><tr><td>MoA</td><td>0.291</td><td>0.257</td><td>0.303</td><td>0.024</td><td>0.375</td><td>0.906</td><td>0.500 0.626</td><td>-1.233</td><td>0.769</td><td>0.770</td><td>0.853</td><td>0.487</td></tr><tr><td>Stackelberg (DPO)</td><td>0.322</td><td>0.309</td><td>0.390</td><td>0.030</td><td>0.394</td><td>0.882</td><td>0.860 0.663</td><td>4.106</td><td>0.778</td><td>0.770</td><td>0.810</td><td>0.576</td></tr><tr><td>Stackelberg (GRPO)</td><td>0.340</td><td>0.330</td><td>0.353</td><td>0.028</td><td>0.424</td><td>0.869</td><td>0.868 0.684</td><td>7.138</td><td>0.791</td><td>0.780</td><td>0.841</td><td>0.609</td></table>

Datasets. We evaluate across 12 datasets in five domains. Scientific (4): BixBench (Mitchener et al., 2025), LabBench (Laurent et al., 2024), SMDD (Han et al., 2026), and AssayBench (De Brouwer et al., 2026), spanning biology, bioinformatics, and drug discovery. Reasoning (2): GPQA-Diamond (Rein et al., 2024) (graduate-level science) and MATH (Hendrycks et al., 2021). Code (2): HumanEval (Chen et al., 2021) and MBPP (Austin et al., 2021). Instructionfollowing (2): AlpacaEval (Dubois et al., 2024) (score from a reward model, Skywork-Reward-Llama-3.1-8B-v0.2 (Liu et al., 2024)) and IFEval (Zhou et al., 2023) (verifiable instruction-following accuracy). Knowledge and truthfulness (2): TruthfulQA (Lin et al., 2022) and CulturalBench-Hard (Chiu et al., 2024).

## 4 RESULTS

Table 1 reports performance of all methods across the three model pools and 12 datasets.

Stackelberg consistently outperforms all baselines on average. STACKELBERG ALIGNMENT achieves the highest macro-average across all three pools (Table 1). Stackelberg (GRPO) ranks first with Avg scores of 0.563, 0.544, and 0.609 in Pools 1–3, while Stackelberg (DPO) ranks second in Pools 1 and 2 (0.548 and 0.540). Relative to the strongest training-based baseline, Sparta Alignment, Stackelberg (GRPO) improves macro-average by 7.4%, 2.1%, and 5.2% in the three pools. The gap over the best static inference baseline (MoA or Heterogeneous Swarms) is substantially larger, exceeding 20% in Pools 1 and 3 and 12% in Pool 2, confirming that (1) training-based collaboration methods are generally stronger and (2) STACKELBERG ALIGNMENT further improves upon the state of the art by adaptively concentrating training on informative instructions. Appendix B.1 reports per-task 95% confidence intervals for Stackelberg (GRPO) and (DPO), confirming that many of these individual gains are statistically significant.

Consistent gains on open-ended tasks. The most consistent column-wise advantage appears on SMDD (Small Molecule Drug Discovery), an open-ended scientific task: Stackelberg (GRPO) and (DPO) rank first and second on SMDD across all three pools. On AlpacaEval, Stackelberg achieves the highest reward-model score in Pools 1 and 3 (8.081 and 7.138). IFEval follows similarly, with Stackelberg (DPO) first in Pool 1 and Stackelberg (GRPO) first in Pool 3. A common thread is that these tasks are open-ended and lack a single correct answer, making the quality of the peer-judgment signal central to training progress; the EXP3 curriculum concentrates duels on instructions where the model pool still disagrees, producing richer preference pairs exactly where they matter most.

![](images/a4c646a7f7de07a6630376292bb0c43362e8f687158804f6c208dbe1512757e1.jpg)

![](images/d0c7d0763e4dd9dc0ddf1ca0b6cb1da54d18acec30323da28e3d328a6446f135.jpg)  
Figure 3: Reputation score trajectories over training iterations for TruthfulQA and HumanEval (Pool 2, Stackelberg GRPO). Reputation scores diverge from the same initialization and stabilize by iteration 4–5. The leading model differs across tasks, confirming genuine task-specific competitive structure and collaborative learning landscape.

Gains transfer broadly across reasoning, code, and knowledge domains. The improvement extends well beyond the tasks most directly linked to preference learning. On GPQA-Diamond, Stackelberg (GRPO) leads in Pools 2 and 3 (0.423 and 0.424) and Stackelberg (DPO) ties for first in Pool 1 alongside Heterogeneous Swarms. On MATH, Stackelberg (GRPO) achieves the best score in Pool 1 (0.877) and Stackelberg (DPO) leads in Pool 2 (0.889); MATH instructions span a wide difficulty range, and we hypothesize this is exactly the setting where an adaptive curriculum that concentrates duels at the frontier of the pool’s current ability (§2.1) has the most room to help relative to fixed uniform sampling. On HumanEval, Stackelberg (GRPO) ties for first in all three pools. Exceptions occur on narrow scientific tasks: LabBench in Pool 3 is led by AggLM, and AssayBench in Pools 1–2 is led by Heterogeneous Swarms. This suggests that highly specialized benchmarks like scientific discovery may benefit from more high-quality data in the instruction pool.

GRPO and DPO show complementary strengths. Within our two training variants, Stackelberg (GRPO) outperforms Stackelberg (DPO) on Avg in all pools and on most scientific and reasoning columns, while Stackelberg (DPO) is more competitive on instruction-following and cultural knowl edge tasks (IFEval and CulturalBench in Pool 1; TruthfulQA and CulturalBench in Pool 2). Notably, Stackelberg (DPO) falls slightly below Sparta in Pool 3 (0.576 vs. 0.579), the one exception to Stackelberg’s consistent advantage, suggesting that the GRPO objective provides a signal that DPO alone does not reliably deliver in this setting.

Margins vary with pool heterogeneity. Among non-Stackelberg training baselines, Sparta Alignment is consistently the strongest, validating the duel-based preference learning framework; the additional gains from the adaptive leader over Sparta range from 2.1% to 7.4%. The narrowest margin appears in Pool 2, where the heterogeneous academic training backgrounds of the models make it harder to find a single instruction distribution that benefits all models simultaneously. Pool 1 (specialized expert LLMs) shows the largest GRPO margin over Sparta (7.4%), while Pool 3 (four general-purpose LLMs) shows the largest gap over static inference (25.1% above MoA).

## 5 ANALYSIS

Pool heterogeneity and competitive stratification. To achieve competitive pool training in STACKELBERG ALIGNMENT, models must differ by capabilities. Figure 2 (Pool 2, GRPO) confirms this: no model dominates all tasks, and the specialist structure is clear: M4 leads on 5/10 tasks but loses to M0 on reasoning and coding and to M1 on TruthfulQA. This heterogeneity is captured by the reputation system: Figure 3 shows reputation trajectories for TruthfulQA and HumanEval. Reputation scores gradually diverge from equal starting values, stabilize by iteration 4–5, and the winning model differs across tasks, providing evidence that task-specific competitive structure is resolved early and maintained throughout training.

![](images/707bedff460e2e8a6cfeeae17af57c1d29e0c6242ef2f9a6ae41930aca6b3c24.jpg)  
Figure 2: Per-model eval scores at best iteration (Pool 2, Stackelberg GRPO). Row-normalized; green = best, red = worst per task.

![](images/651caad6c4a03d3c3abacc68943dbb82569d05992f0401d03dc05ff99006e49f.jpg)

![](images/e371270c7ee10d00f850fa0775ad5e71d6fbf523655c55fcc96d9c44a0c5bd02.jpg)

Figure 4: Normalized opponent weight assigned by the Stackelberg leader over iterations (Pool 2, Stackelberg GRPO). Dashed line = uniform selection baseline. The leader shields the emerging dominant model from being excessively used as an opponent and adapts its strategy per task.  
![](images/1fea3154f54f91d700a7f13d83a3c86020966022430bf67099026224a87613af.jpg)

![](images/00713a474b8bb1de938da661c39d835253ffaef28cb81ca019edb54f534170e8.jpg)  
Figure 5: Leader instruction weights over training iterations (the MATH dataset, model pool 1). Left: heatmap of normalized weight deviation from uniform for the top-32 most dynamic instructions. Right: trajectory of top-5 and bottom-5 instructions by final-iteration weight; dashed line = uniform. The leader converges to an implicit difficulty curriculum by iteration 5.

Adaptive opponent selection. Given this heterogeneity, the competitive landscape the leader must navigate is non-trivial: pairwise win rates on MATH span 0.1–0.6 with no model dominating (Appendix B.3, Figure 6), and the leader exploits this structure through adaptive opponent selection. Figure 4 traces the leader’s normalized opponent weights over iterations for math and TruthfulQA. Early choices are already task-specific: math initially concentrates on M3; TruthfulQA peaks on M5 at iteration 2. Notably, the model with the highest final eval score is actively shielded as an opponent at peak training time, with its weight dropping to < 0.5× uniform at iterations 4–5, consistent with the Stackelberg objective of not wasting training budget on unwinnable duels. The strategy is genuinely task-adaptive: the same model (M7) is the most-selected opponent on math (weight 0.176 at iteration 5) but nearly irrelevant on TruthfulQA (weight 0.062 at iteration 7).

Emergent difficulty curriculum over instructions. Figure 5 shows the leader’s per-instruction weights over training iterations (math task, Pool 1). Iterations 0–4 are near-uniform: the leader explores all instructions before committing. By iteration 5, the distribution sharpens: the top-5 instructions rise to ∼1.6× uniform by iteration 7, while the bottom-5 fall to ∼0.75× uniform, and the divergence exists and is accelerating. This constitutes an emergent difficulty curriculum: the leader identifies instructions where model disagreement persists, precisely where preference signal is most informative, and concentrates duels there without any explicit difficulty label. The behavior mirrors the leader’s opponent-selection strategy (Fig. 4): both start near-uniform and progressively concentrate attention as the competitive hierarchy becomes clearer.

## 6 CONCLUSION

We presented STACKELBERG ALIGNMENT, a collaborative alignment framework for adaptive instruction selection and curriculum learning in multi-LLM evolution. By casting the training loop as a Stackelberg game solved via EXP3, STACKELBERG ALIGNMENT concentrates pairwise duels on the instructions where the model pool still disagrees. Across three heterogeneous model pools and 12 benchmarks, STACKELBERG ALIGNMENT with GRPO learning achieves the highest macro-average in every pool, improving over training-based baselines by up to 7.4% and over the best static inference baseline by 12–25%, with ablations confirming that the adaptive curriculum, reputation-weighted judgment, and reputation-based matching each contribute to these gains.

## AI USE STATEMENT

We used generative AI tools to help with language polishing. The AI tools were used solely for language editing and did not contribute to the intellectual content of the work. We have reviewed all AI-assisted edits and take responsibility for the final content of this work.

## REPRODUCIBILITY STATEMENT

All hyperparameters are reported in Appendix C (Implementation Details), including EXP3 learning rate and reward coefficients, LoRA configuration, DPO and GRPO training settings, and generation parameters. Dataset sizes and splits are provided in Appendix E (Dataset Statistics). Baseline configurations are listed in Appendix F (Baseline Configurations), and the judge prompt is presented in Appendix G (Judge Prompt). Statistical significance for our main results is reported in Appendix B.1. Our method builds on publicly available pretrained models and standard training libraries; no proprietary data or infrastructure is required beyond the compute described in Appendix C.

## ETHICS STATEMENT

STACKELBERG ALIGNMENT is designed to improve language model alignment through collaborative training: models improve each other via structured competition and peer judgment, which is broadly beneficial. However, competitive training pools also introduce potential misuse scenarios. A malicious actor who controls one model in the pool could manipulate duel outcomes to inflate its own reputation score, skew peer judgment, and inject adversarial preference pairs into the training data, degrading the alignment of all other models in the pool (Yang et al., 2026). The reputation system provides partial robustness by down-weighting low-reputation judges, but does not make it a formal defense against adversarial participants. We see adversarial robustness of collaborative training as an important open problem, particularly as decentralized or federated model collaboration systems are deployed at scale.

## REFERENCES

Marah Abdin, Jyoti Aneja, Harkirat Behl, Sebastien Bubeck, Ronen Eldan, Suriya Gunasekar,´ Michael Harrison, Russell J Hewett, Mojan Javaheripi, Piero Kauffmann, et al. Phi-4 technical report. arXiv preprint arXiv:2412.08905, 2024.

Peter Auer, Nicolo Cesa-Bianchi, Yoav Freund, and Robert E Schapire. The nonstochastic multiarmed bandit problem. SIAM Journal on Computing, 32(1):48–77, 2002.

Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, et al. Program synthesis with large language models. arXiv preprint arXiv:2108.07732, 2021.

Tamer Bas¸ar and Geert Jan Olsder. Dynamic Noncooperative Game Theory. SIAM, 1998.

Aaron Blakeman, Aaron Grattafiori, Aarti Basant, Abhibha Gupta, Abhinav Khattar, Adi Renduchintala, Aditya Vavre, Akanksha Shukla, Akhiad Bercovich, Aleksander Ficek, et al. Nvidia nemotron 3: Efficient and open intelligence. arXiv preprint arXiv:2512.20856, 2025.

Lingjiao Chen, Matei Zaharia, and James Zou. Frugalgpt: How to use large language models while reducing cost and improving performance. Transactions on Machine Learning Research, 2023.

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, et al. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374, 2021.

Xiaoyin Chen, Jiarui Lu, Minsu Kim, Dinghuai Zhang, Jian Tang, Alexandre Piche, Nicolas Gontier,´ Yoshua Bengio, and Ehsan Kamalloo. Self-evolving curriculum for llm reasoning. arXiv preprint arXiv:2505.14970, 2025.

Zixiang Chen, Yihe Deng, Huizhuo Yuan, Kaixuan Ji, and Quanquan Gu. Self-play fine-tuning converts weak language models to strong language models. arXiv preprint arXiv:2401.01335, 2024.

Yu Ying Chiu, Liwei Jiang, Bill Yuchen Lin, Chan Young Park, Shuyue Stella Li, Sahithya Ravi, Mehar Bhatia, Maria Antoniak, Yulia Tsvetkov, Vered Shwartz, et al. CulturalBench: A robust, diverse, and challenging cultural benchmark by human-ai culturalteaming. arXiv preprint arXiv:2410.02677, 2024.

John Dang, Shivalika Singh, Daniel D’souza, Arash Ahmadian, Alejandro Salamanca, Madeline Smith, Aidan Peppin, Sungjin Hong, Manoj Govindassamy, Terrence Zhao, et al. Aya expanse: Combining research breakthroughs for a new multilingual frontier. arXiv preprint arXiv:2412.04261, 2024.

Edward De Brouwer, Carl Edwards, Alexander Wu, Jenna Collier, Graham Heimberg, Xiner Li, Meena Subramaniam, Ehsan Hajiramezanali, David Richmond, Jan-Christian Hutter, et al. Assay-¨ Bench: An assay-level virtual cell benchmark for llms and agents. arXiv preprint arXiv:2605.10876, 2026.

Yilun Du and Leslie Pack Kaelbling. Position: Compositional generative modeling: A single model is not all you need. In Forty-first International Conference on Machine Learning, 2024.

Yilun Du, Shuang Li, Antonio Torralba, Joshua B. Tenenbaum, and Igor Mordatch. Improving factuality and reasoning in language models through multiagent debate. In Forty-first International Conference on Machine Learning, 2024.

Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models, 2024.

Yann Dubois, Chen Xuechen Li, Rohan Taori, Tianyi Zhang, Ishaan Gulrajani, Jimmy Ba, Carlos Guestrin, Percy S Liang, and Tatsunori B Hashimoto. Alpacafarm: A simulation framework for methods that learn from human feedback. Advances in Neural Information Processing Systems, 36, 2024.

Aram Ebtekar and Paul Liu. Elo-mmr: A rating system for massive multiplayer competitions. In Proceedings ofthe Web Conference 2021, pp. 1772–1784, 2021.

Shangbin Feng, Taylor Sorensen, Yuhan Liu, Jillian Fisher, Chan Young Park, Yejin Choi, and Yulia Tsvetkov. Modular pluralism: Pluralistic alignment via multi-llm collaboration. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pp. 4151–4171, 2024.

Shangbin Feng, Zifeng Wang, Palash Goyal, Yike Wang, Weijia Shi, Huang Xia, Hamid Palangi, Luke Zettlemoyer, Yulia Tsvetkov, Chen-Yu Lee, et al. Heterogeneous swarms: Jointly optimizing model roles and weights for multi-llm systems. arXiv preprint arXiv:2502.04510, 2025a.

Shangbin Feng, Wenxuan Ding, Alisa Liu, Zifeng Wang, Weijia Shi, Yike Wang, Shannon Zejiang Shen, Xiaochuang Han, Hunter Lang, Chen-Yu Lee, et al. When one LLM drools, multi-LLM collaboration rules. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 17048–17063, 2026a.

Shangbin Feng, Yike Wang, Weijia Shi, Luke Zettlemoyer, Yejin Choi, and Yulia Tsvetkov. Scaling participation in modular ai systems. arXiv preprint arXiv:2606.07812, 2026b.

Tao Feng, Yanzhen Shen, and Jiaxuan You. Graphrouter: A graph-based router for llm selections. In The Thirteenth International Conference on Learning Representations, 2025b.

Kevin Han, Renfei Zhang, Kathy Wei, Hamed Mahdavi, Niloofar Mireshghallah, and Amir Barati Farimani. SMDD-Bench: Can llms solve real-world small molecule drug design tasks? arXiv preprint arXiv:2605.21740, 2026.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the MATH dataset. In The Thirty-fifth Annual Conference on Neural Information Processing Systems, 2021.

Yuru Jiang, Wenxuan Ding, Shangbin Feng, Greg Durrett, and Yulia Tsvetkov. Sparta alignment: Collectively aligning multiple language models through combat. Advances in Neural Information Processing Systems, 38:132970–133009, 2025.

Jon M Laurent, Joseph D Janizek, Michael Ruzo, Michaela M Hinks, Michael J Hammerling, Siddharth Narayanan, Manvitha Ponnapati, Andrew D White, and Samuel G Rodriques. LAB-Bench: Measuring capabilities of language models for biology research. arXiv preprint arXiv:2407.10362, 2024.

Zhuofeng Li, Haoxiang Zhang, Seungju Han, Sheng Liu, Jianwen Xie, Yu Zhang, Yejin Choi, James Y Zou, and Pan Lu. In-the-flow agentic system optimization for effective planning and tool use. In International Conference on Learning Representations, volume 2026, pp. 50524–50570, 2026.

Stephanie Lin, Jacob Hilton, and Owain Evans. TruthfulQA: Measuring how models mimic human falsehoods. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 3214–3252, 2022.

Bo Liu, Simon Yu, Yiding Jiang, Ao Qu, Andrew Zhao, Zichen Liu, Junsu Kim, Zijian Zhou, Seungone Kim, Tongzheng Ren, et al. Spade: Self-play in adaptive synthetic executable environments. arXiv preprint arXiv:2608.19197, 2026a.

Bo Liu, Simon Yu, Zichen Liu, Leon Guertler, Penghui Qi, Daniel Balcells, Mickel Liu, Cheston Tan, Weiyan Shi, Min Lin, et al. Spiral: Self-play on zero-sum games incentivizes reasoning via multiagent multi-turn reinforcement learning. In International Conference on Learning Representations, volume 2026, pp. 35407–35434, 2026b.

Chris Yuhao Liu, Liang Zeng, Jiacai Liu, Rui Yan, Jujie He, Chaojie Wang, Shuicheng Yan, Yang Liu, and Yahui Zhou. Skywork-reward: Bag of tricks for reward modeling in llms. arXiv preprint arXiv:2410.18451, 2024.

Haipeng Luo, Qingfeng Sun, Can Xu, Pu Zhao, Qingwei Lin, Jianguang Lou, Shifeng Chen, Yansong Tang, and Weizhu Chen. Arena learning: Build data flywheel for llms post-training via simulated chatbot arena. arXiv preprint arXiv:2407.10627, 2024.

Ludovico Mitchener, Jon M Laurent, Alex Andonian, Benjamin Tenmann, Siddharth Narayanan, Geemi P Wellawatte, Andrew White, Lorenzo Sani, and Samuel G Rodriques. BixBench: A comprehensive benchmark for llm-based agents in computational biology. arXiv preprint arXiv:2503.00096, 2025.

Remi Munos, Michal Valko, Daniele Calandriello, Mohammad Gheshlaghi Azar, Mark Rowland, Zhaohan Daniel Guo, Yunhao Tang, Matthieu Geist, Thomas Mesnard, Come Fiegel, Andrea Michi,ˆ Marco Selvi, Sertan Girgin, Nikola Momchev, Olivier Bachem, Daniel J Mankowitz, Doina Precup, and Bilal Piot. Nash learning from human feedback. In Proceedings of the 41st International Conference on Machine Learning, pp. 36743–36768, 2024.

Isaac Ong, Amjad Almahairi, Vincent Wu, Wei-Lin Chiang, Tianhao Wu, Joseph E Gonzalez, M Waleed Kadous, and Ion Stoica. Routellm: Learning to route llms from preference data. In The Thirteenth International Conference on Learning Representations, 2025.

Qwen Team. Qwen3.5: Towards native multimodal agents, February 2026. URL https://qwen. ai/blog?id=qwen3.5.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. Advances in Neural Information Processing Systems, 36, 2024.

David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R Bowman. GPQA: A graduate-level google-proof q&a benchmark. In First Conference on Language Modeling, 2024.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Vighnesh Subramaniam, Yilun Du, Joshua B Tenenbaum, Antonio Torralba, Shuang Li, and Igor Mordatch. Multiagent finetuning: Self improvement with diverse reasoning chains. arXiv preprint arXiv:2501.05707, 2025.

Zhiqing Sun, Longhui Yu, Yikang Shen, Weiyang Liu, Yiming Yang, Sean Welleck, and Chuang Gan. Easy-to-hard generalization: Scalable alignment beyond human supervision. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024.

Gemma Team, Sherif El Abd, Vaibhav Aggarwal, Robin Algayres, Alek Andreev, Olivier Bachem, Ian Ballantyne, Cormac Brick, Victor Carbune, Michelle Casbon, et al. Gemma 4 technical report.˘ arXiv preprint arXiv:2607.02770, 2026.

Vinzenz Thoma, Barna Pasztor, Andreas Krause, Giorgia Ramponi, and Yifan Hu. Contextual ´ bilevel reinforcement learning for incentive alignment. Advances in Neural Information Processing Systems, 37:127369–127435, 2024.

H.F. von Stackelberg and A.T. Peacock. The Theory ofthe Market Economy. Hodge, 1952.

Junlin Wang, Jue Wang, Ben Athiwaratkun, Ce Zhang, and James Y Zou. Mixture-of-agents enhances large language model capabilities. In International Conference on Learning Representations, volume 2025, pp. 33944–33963, 2025a.

Zhaoyang Wang, Weilei He, Zhiyuan Liang, Xuchao Zhang, Chetan Bansal, Ying Wei, Weitong Zhang, and Huaxiu Yao. Cream: Consistency regularized self-rewarding language models, 2025b.

Chu Xu, Zhixin Zhang, Tianyu Jia, and Yujie Jin. Stackelberg self-annotation: A robust approach to data-efficient llm alignment. Advances in Neural Information Processing Systems, 38:62912– 62949, 2025.

Ziyuan Yang, Wenxuan Ding, Shangbin Feng, and Yulia Tsvetkov. Among us: Measuring and mitigating malicious contributions in model collaboration systems. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 15969–15988, 2026.

Jihan Yao, Wenxuan Ding, Shangbin Feng, Lucy Lu Wang, and Yulia Tsvetkov. Varying shades of wrong: Aligning llms with wrong answers only, 2024.

Weizhe Yuan, Richard Yuanzhe Pang, Kyunghyun Cho, Xian Li, Sainbayar Sukhbaatar, Jing Xu, and Jason Weston. Self-rewarding language models, 2025.

Xiang Yue, Yueqi Song, Akari Asai, Seungone Kim, JEAN NYANDWI, Simran Khanuja, Anjali Kantharuban, Lintang Sutawika, Sathyanarayanan Ramamoorthy, and Graham Neubig. Pangea: A fully open multilingual multimodal llm for 39 languages. In International Conference on Learning Representations, volume 2025, pp. 47758–47811, 2025.

Yongcheng Zeng, Zexu Sun, Bokai Ji, Erxue Min, Hengyi Cai, Shuaiqiang Wang, Dawei Yin, Haifeng Zhang, Xu Chen, and Jun Wang. Cures: From gradient analysis to efficient curriculum learning for reasoning llms. In International Conference on Learning Representations, volume 2026, pp. 1531–1556, 2026.

Yuheng Zhang, Dian Yu, Baolin Peng, Linfeng Song, Ye Tian, Mingyue Huo, Nan Jiang, Haitao Mi, and Dong Yu. Iterative nash policy optimization: Aligning llms with general preferences via no-regret learning. In International Conference on Learning Representations, volume 2025, pp. 31833–31849, 2025.

Wenting Zhao, Pranjal Aggarwal, Swarnadeep Saha, Asli Celikyilmaz, Jason Weston, and Ilia Kulikov. The majority is not always right: Rl training for solution aggregation. arXiv preprint arXiv:2509.06870, 2025.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, et al. Judging llm-as-a-judge with mt-bench and chatbot arena. Advances in Neural Information Processing Systems, 36:46595–46623, 2023.

Jeffrey Zhou, Tianjian Lu, Swaroop Mishra, Siddhartha Brahma, Sujoy Basu, Yi Luan, Denny Zhou, and Le Hou. Instruction-following evaluation for large language models. arXiv preprint arXiv:2311.07911, 2023.

## A RELATED WORK

Model collaboration. The landscape of multi-LLM collaboration can be organized by the level of information exchange (Feng et al., 2026b). API-level methods route or cascade queries to the most suitable model in the pool (Ong et al., 2025; Feng et al., 2025b; Chen et al., 2023). Text-level methods have models exchange generated text: through debate and multi-round discussion (Du et al., 2024), response synthesis across model layers (Wang et al., 2025a), structured interaction graphs (Feng et al., 2025a), and modular value alignment (Feng et al., 2024). Weight-level methods operate directly in parameter space via model merging. Among training-based approaches, Multiagent Fine-tuning (Subramaniam et al., 2025) uses debate-generated data for supervised fine-tuning; AggLM (Zhao et al., 2025) trains an RL aggregator over model outputs; Arena Learning (Luo et al., 2024) simulates pairwise ELO-rated battles to generate post-training data; and Sparta Alignment (Jiang et al., 2025) trains models through competitive duels with peer judgment. STACKELBERG ALIGNMENT extends this line by introducing an adaptive leader that determines which instructions drive collaborative training at each step, turning the instruction distribution itself into a learned, non-stationary curriculum.

Self-play and curriculum learning. A related line of work trains LLMs through self-play and iterative self-improvement. SPIN (Chen et al., 2024) and Self-Rewarding LMs (Yuan et al., 2025) improve a single model by having it compete against or judge its own prior outputs. SPIRAL (Liu et al., 2026b) extends self-play to zero-sum multi-agent games, generating an automatic opponent curriculum; SPADE (Liu et al., 2026a) trains a single model as both environment designer and reasoning agent, adaptively targeting its own capability frontier. A parallel thread develops curriculum learning for LLM fine-tuning: easy-to-hard generalization (Sun et al., 2024) shows that ordering training examples by difficulty improves reasoning; SEC (Chen et al., 2025) frames curriculum selection as a non-stationary bandit over instruction categories; CurES (Zeng et al., 2026) uses gradient signals to identify the most informative training examples at each stage. STACKELBERG ALIGNMENT unifies these threads: it applies bandit-driven instruction selection within a competitive multi-LLM training loop, where the difficulty signal emerges from inter-model disagreement rather than from predefined categories or gradient heuristics.

Game-theoretic alignment. Our work is grounded in game-theoretic alignment, where the interaction between alignment objectives and model responses is formalized as a strategic game between rational agents rather than a fixed optimization target. Nash Learning from Human Feedback (Munos et al., 2024) formalizes preference optimization as a two-player constant-sum game with the Nash policy as solution; INPO (Zhang et al., 2025) iteratively approximates the Nash policy via no-regret online learning. These Nash-based methods model alignment as a symmetric simultaneous game. STACKELBERG ALIGNMENT instead adopts a Stackelberg formulation (von Stackelberg & Peacock, 1952; Bas¸ar & Olsder, 1998): the leader (instruction selector) commits to a distribution before the followers (LLMs) respond and train on such data, capturing the natural asymmetry between curriculum design and model adaptation. SSAPO (Xu et al., 2025) also applies Stackelberg games to LLM alignment, but as a two-policy robustness problem against noisy preference labels; STACKELBERG ALIGNMENT applies the framing to a multi-follower competitive pool where the leader maximizes instructional informativeness. Bilevel RL for incentive alignment (Thoma et al., 2024) provides further theoretical grounding for leader-follower RL under best-response constraints.

## B ADDITIONAL RESULTS

## B.1 STATISTICAL SIGNIFICANCE

Table 2 reports per-task 95% confidence intervals for Stackelberg (GRPO) and (DPO) across all three pools, computed via 10,000 Monte Carlo draws (Bernoulli for binary tasks; Gaussian for AlpacaEval with $\sigma = \bar { 5 } . 0 / \sqrt { n } ) . \mathrm { ~ A ~ } ^ { * }$ mark indicates the point estimate exceeds every baseline on that task; ∗∗ indicates the CI lower bound does. Stackelberg (GRPO) earns at least one ∗∗ mark in every pool, and at least one ∗ mark on 7–8 of the 10 reported tasks. SMDD and AssayBench are excluded: their small test sets (n ≤ 334) inflate Gaussian CIs to ±0.55, making interval comparisons uninformative.

Table 2: Per-task scores and 95% confidence intervals for Stackelberg (GRPO) and Stackelberg (DPO). CIs computed via 10,000 Monte Carlo draws (Bernoulli for binary tasks; Gaussian for AlpacaEval). ∗: point estimate exceeds every baseline. ∗∗: CI lower bound exceeds every baseline. SMDD and AssayBench omitted (CI ≈ ±0.55).
<table><tr><td rowspan="2">Task</td><td colspan="2">Stackelberg (GRPO)</td><td colspan="2">Stackelberg (DPO)</td></tr><tr><td>Score</td><td>95% CI</td><td>Score</td><td>95% CI</td></tr><tr><td colspan="5">Pool 1: Specialized Expert LLMs</td></tr><tr><td>BixBench LabBench</td><td>0.350* 0.322*</td><td>[0.262, 0.447] [0.284, 0.360]</td><td>0.233 0.311</td><td>[0.155, 0.320] [0.273, 0.348]</td></tr><tr><td>GPQA-Dia MATH HumanEval</td><td>0.354 0.877** 0.816</td><td>[0.263, 0.444] [0.856, 0.898] [0.746, 0.886]</td><td>0.364 0.857** 0.763</td><td>[0.273, 0.455] [0.834, 0.879] [0.684, 0.842]</td></tr><tr><td>MBPP</td><td>0.653*</td><td>[0.610, 0.696]</td><td>0.643</td><td>[0.602, 0.684]</td></tr><tr><td>AlpacaEval IFEval</td><td>7.819* 0.730</td><td>[7.512, 8.127] [0.676, 0.784]</td><td>8.081** 0.749*</td><td>[7.771, 8.397]</td></tr><tr><td>TruthfulQA</td><td></td><td>[0.647, 0.720]</td><td>0.682</td><td>[0.692, 0.800]</td></tr><tr><td>CulturalBench</td><td>0.684</td><td></td><td></td><td>[0.645, 0.718]</td></tr><tr><td></td><td>0.724</td><td>[0.687, 0.759]</td><td>0.747*</td><td>[0.711, 0.781]</td></tr><tr><td>Pool 2: Diverse Academic Research</td><td></td><td></td><td></td><td></td></tr><tr><td>BixBench</td><td>0.291</td><td>[0.204, 0.379]</td><td>0.311</td><td>[0.223, 0.398]</td></tr><tr><td>LabBench</td><td>0.355*</td><td>[0.317,0.395]</td><td>0.334</td><td>[0.296, 0.374]</td></tr><tr><td>GPQA-Dia</td><td>0.423*</td><td>[0.323, 0.515]</td><td>0.393</td><td>[0.293, 0.495]</td></tr><tr><td>MATH</td><td>0.884</td><td>[0.863, 0.904]</td><td>0.889*</td><td>[0.868, 0.909]</td></tr><tr><td>HumanEval</td><td>0.816*</td><td>[0.746, 0.886]</td><td>0.816*</td><td>[0.746, 0.886]</td></tr><tr><td>MBPP</td><td>0.634</td><td>[0.591, 0.678]</td><td>0.632</td><td></td></tr><tr><td></td><td>5.207</td><td></td><td></td><td>[0.589, 0.674]</td></tr><tr><td>AlpacaEval</td><td></td><td>[4.897, 5.514]</td><td>4.986</td><td>[4.676, 5.303]</td></tr><tr><td>IFEval</td><td>0.680</td><td>[0.620, 0.736]</td><td>0.676</td><td>[0.620, 0.732]</td></tr><tr><td>TruthfulQA</td><td>0.626</td><td>[0.588, 0.663]</td><td>0.656</td><td>[0.618, 0.692]</td></tr><tr><td>CulturalBench</td><td>0.680</td><td>[0.643, 0.718]</td><td>0.696*</td><td>[0.659, 0.731]</td></tr><tr><td>Pool 3: General-Purpose LLMs</td><td></td><td></td><td></td><td></td></tr><tr><td>BixBench</td><td>0.340</td><td>[0.252, 0.437]</td><td>0.322</td><td>[0.233, 0.418]</td></tr><tr><td>LabBench</td><td>0.330</td><td>[0.291, 0.370]</td><td>0.309</td><td>[0.272, 0.346]</td></tr><tr><td>GPQA-Dia</td><td>0.424</td><td>[0.323, 0.525]</td><td>0.394</td><td>[0.303, 0.495]</td></tr><tr><td>MATH</td><td>0.869</td><td>[0.848, 0.890]</td><td>0.882</td><td>[0.861, 0.902]</td></tr><tr><td>HumanEval</td><td>0.868</td><td>[0.807, 0.930]</td><td>0.860</td><td>[0.790, 0.921]</td></tr><tr><td>MBPP</td><td>0.684</td><td>[0.641, 0.725]</td><td>0.663</td><td>[0.620, 0.706]</td></tr><tr><td>AlpacaEval</td><td>7.138</td><td>[6.830, 7.449]</td><td></td><td></td></tr><tr><td></td><td></td><td></td><td>4.106</td><td>[3.796, 4.417]</td></tr><tr><td>IFEval</td><td>0.791*</td><td>[0.740, 0.840]</td><td>0.778*</td><td>[0.724, 0.828]</td></tr><tr><td>TruthfulQA</td><td>0.780</td><td>[0.746, 0.812]</td><td>0.770</td><td>[0.736, 0.802]</td></tr><tr><td>CulturalBench</td><td>0.841</td><td>[0.811, 0.870]</td><td>0.810</td><td>[0.778, 0.842]</td></tr></table>

## B.2 CROSS-TASK TRANSFER

Table 3 tests whether competitive pool training on one task transfers to others. We take the best Stackelberg (GRPO) checkpoint trained exclusively on MATH-type instructions (Pool 2, parti 4 full, iteration 6) and evaluate it on HumanEval and MBPP; symmetrically, we take the best checkpoint trained on HumanEval instructions (parti 0 full, iteration 7) and evaluate it on MATH and MBPP. The baseline is the best untrained single model in the pool per task.

![](images/0a2f78418ccf5629d270fe9ce313cbdeceb602d5139482b18f4a6f2c00b91b7e.jpg)  
Figure 6: Pairwise win-rate matrix for the MATH dataset (Pool 2, Stackelberg GRPO). No model dominates; win rates span 0.1–0.6; ∼41% of duels are draws given the objective nature of the task.

Table 3: Cross-task transfer in Pool 2. Each trained column shows the best single-model checkpoint from competitive pool training on that source task, evaluated on the target. Baseline: best untrained model in the pool per task. “No gain” indicates same-task training matched or fell below the untrained baseline.
<table><tr><td colspan="2"></td><td colspan="3">Checkpoint source</td></tr><tr><td>Eval Task</td><td>Best Untrained</td><td>Same-Task</td><td>Math-Trained</td><td>HumanEval-Trained</td></tr><tr><td>MATH</td><td>0.863</td><td>0.865</td><td></td><td>0.866</td></tr><tr><td>HumanEval</td><td>0.790</td><td>0.790 (no gain)</td><td>0.974 (+18pp)</td><td></td></tr><tr><td>MBPP</td><td>0.639</td><td>0.632 (no gain)</td><td>0.963 (+33pp)</td><td>0.961 (+32pp)</td></tr></table>

Training on MATH transfers dramatically to coding (+18 pp on HumanEval, +33 pp on MBPP), while same-task pool training on coding yields no improvement over the best untrained model. HumanEval pool training likewise transfers to MBPP (+32 pp). These results suggest that competitive training on mathematical reasoning produces broadly transferable gains rather than task-specific memorization.

## B.3 ABLATION: OPPONENT SELECTION STRATEGY

Figure 6 shows the pairwise win-rate matrix for MATH (Pool 2, Stackelberg GRPO) referenced in §5: no model dominates, win rates span 0.1–0.6, and ∼41% of duels are draws given the objective nature of the task, motivating the adaptive opponent selection studied below.

Figure 7 compares four opponent-selection strategies on the MATH task (Pool 2). All four beat the single-model baseline (0.857), confirming that competitive training helps regardless of opponent schedule. schedule increasing (0.878) outperforms the default schedule decreasing (0.865), suggesting that gradually concentrating on harder opponents is beneficial. lowest diff (0.875) also outperforms the default, while highest diff (0.861) performs worst; overexposure to very hard opponents appears noisy. Note that the three ablation variants ran for 2 iterations vs. 8 for the default; their final scores are not directly comparable but are reported for reference.

## B.4 ABLATION: POOL SIZE

Figure 8 compares pool sizes of 4, 6, and 8 models on the MATH task. All three beat the singlemodel baseline. Pool 6 (0.880) slightly outperforms Pool 4 (0.875) and Pool 8 (0.865), suggesting a moderate pool size is preferable; however, CIs overlap between Pool 4 and Pool 6. Larger pools

![](images/8df75fac5966a53d0155de773edc94c2d0349055f2b47845399873da0efd213e.jpg)

![](images/1c7c645b8592ec27e5140fbf4c89bb2e0eda66266c46bc3666c3f74fb2473576.jpg)  
Figure 8: Pool size ablation on MATH (Pool 2). All sizes outperform the single-model baseline (0.857); Pool 6 is best.

Figure 7: Opponent selection ablation on MATH (Pool 2). All strategies beat the single-model baseline (0.857); schedule increasing is best.  
![](images/60b494915bbca270f0dca6b482fe90f958c3370a0d4afc18370c040254ff7d83.jpg)

![](images/24a570f7fc410ce61a7d18ef043cec05ad2eec69cc3b304a977d87e0e7d1caf3.jpg)  
Figure 9: Response properties over training iterations (MATH, Pool 2, 8 models). Left: type-token ratio (lexical diversity). Right: mean response length. Bold line = pool average. No model collapses toward repetitive or length-degenerate outputs; pool diversity is maintained throughout.

do not monotonically help; Pool 8 underperforms both smaller configurations, consistent with the intuition that very large pools dilute the competitive signal each model receives.

## B.5 RESPONSE DIVERSITY DURING TRAINING

Figure 9 shows lexical diversity (type-token ratio, TTR) and mean response length per model over training iterations (MATH task, Pool 2). TTR oscillates stably across all 8 iterations with no downward trend, and response length remains in the 100–135 word range throughout. Between-model variation dominates within-model trends, indicating that stylistic properties of the pre-trained models persist despite competitive training. This confirms that the pool maintains behavioral heterogeneity throughout training, a prerequisite for the leader’s adaptive selection strategy to remain meaningful.

## B.6 ABLATION: LLM LEADER VS. PROBABILISTIC LEADER

Table 4 compares two leader strategies on Pool 1 trained with DPO and GRPO across four shared tasks. The LLM leader uses Gemini to select instructions based on model embeddings and current reputation scores (pilot experiment on the same Pool 1 models). The probabilistic leader uses EXP3, an adversarial bandit algorithm that maintains a sampling distribution over instructions updated by follower win-rates; results are taken from Table 1.

The probabilistic leader substantially outperforms the Gemini leader on AlpacaEval (8.08 vs. 1.65) and GPQA (+10.1% absolute), with consistent gains on MBPP (+6.2% for DPO, +7.2% for GRPO)

Table 4: Leader strategy comparison on Pool 1 across four tasks. Best score per task is bolded. AlpacaEval is scored by a reward model (higher = better); all other tasks report accuracy.
<table><tr><td>Method</td><td>AlpacaEval</td><td> $\mathrm { G P Q A }$ </td><td>MBPP</td><td>TruthfulQA</td></tr><tr><td>Gemini Leader + DPO</td><td>1.65</td><td>0.263</td><td>0.581</td><td>0.682</td></tr><tr><td>Prob.  $_ \mathrm { L e a d e r + D P O }$ </td><td>8.081</td><td>0.364</td><td>0.643</td><td>0.682</td></tr><tr><td>Prob.  $_ \mathrm { L e a d e r + G R P O }$ </td><td>7.819</td><td>0.353</td><td>0.653</td><td>0.684</td></tr></table>

and essentially matched performance on TruthfulQA. These results suggest that the expressivity of an LLM leader does not translate to better instruction selection in practice: the EXP3 bandit, which adapts directly to observed win-rates without requiring generalization from model embeddings, yields more effective curricula. The probabilistic leader also avoids the latency and cost of LLM inference at each training round. Between the two probabilistic conditions, GRPO edges out DPO on MBPP and TruthfulQA while DPO leads on $\mathrm { G P Q A }$ , consistent with the complementary strengths in Table 1.

## C IMPLEMENTATION DETAILS

EXP3 leader. We run $T { = } 8$ training iterations. The leader maintains a weight vector $\mathbf { w } ^ { t } \in \mathbb { R } _ { > 0 } ^ { K }$ over an instruction pool of size $K { = } 6 4 .$ . The sampling distribution mixes exploitation and exploration with rate $\gamma { = } 0 . 2 ~ ( \mathrm { E q . } ~ 1 )$ The composite reward is $r _ { k } = 0 . 3 r _ { \mathrm { d i f f } } + 0 . 7 r _ { \mathrm { p r e f } }$ , where $r _ { \mathrm { d i f f } }$ is the difficulty reward (Eq. 2) with floor threshold $\tau _ { \mathrm { t h } } { = } 3 . 0 ~ ( \mathrm { o n }$ the $_ { 1 - 1 0 }$ judge scale), and $r _ { \mathrm { p r e f } }$ is the preference-quality reward (Eq. 4) with Gaussian bandwidth $\sigma _ { r } { = } 0 . 1 5$ on the normalized score gap. The ideal gap target is annealed from 0.6 to 0.15 across training. All scores are normalized to [0, 1] by dividing by $s _ { \mathrm { m a x } } - s _ { \mathrm { m i n } } = 9$

Reputation system. The K factor decays exponentially with the total number of reputation updates n for each model:

$$
K _ { t } = \operatorname* { m a x } \ ( K _ { \operatorname* { m i n } } , \ K _ { 0 } \cdot \alpha ^ { \lfloor n / n _ { s } \rfloor } ) ,\tag{10}
$$

with $K _ { 0 } { = } 1 0 , K _ { \mathrm { m i n } } { = } 5$ , decay rate $\alpha { = } 0 . 9 ,$ and decay step $n _ { s } = 1 0$ . This cools the magnitude of reputation swings as models accumulate more duels and their standings stabilize. The reputation deviation $\sigma _ { i }$ is the sample standard deviation of the last $W { = } 2 0$ reputation deltas:

$$
\sigma _ { i } = \sqrt { \frac { 1 } { | W _ { i } | - 1 } \sum _ { \Delta \in W _ { i } } ( \Delta - \bar { \Delta } ) ^ { 2 } } ,\tag{11}
$$

where $W _ { i }$ is the sliding window of the most recent mi $\scriptstyle 1 ( n , 2 0 )$ updates for model $M _ { i }$ and $\bar { \Delta }$ is their mean. $\sigma _ { i }$ measures per-model rating volatility and enters the opponent-matching score via $z _ { i } \left( \operatorname { E q . 9 } \right)$ .

Opponent matching. We use schedule decreasing matching: the cosine-scheduled target reputation gap decreases from wide (random-like) to tight (near-peer) as training progresses. A fraction $p _ { \mathrm { r a n d } } { = } 0 . 2$ of duels use uniformly random matching to maintain diversity; each combatant draws from $n _ { \mathrm { o p p } } { = } 3$ candidate opponents.

DPO training. Each model is fine-tuned with LoRA $( r { = } 6 4 , \alpha { = } 1 6$ , dropout 0.1) applied to the query, key, value, and output projections. Optimizer: AdamW, learning rate $1 0 ^ { - 6 }$ , cosine schedule, effective batch size 16 (per-device batch 1, gradient accumulation 16), 1 epoch per iteration. $\beta { = } 0 . 1$ max sequence length 2048, max prompt length 1536.

GRPO training. Same LoRA configuration as DPO. Learning rate $1 0 ^ { - 6 }$ , cosine schedule, perdevice batch size 8, 1 epoch, 4 rollout generations per prompt, max completion length 256 tokens. KL coefficient $\beta { = } 0 . 0 ;$ clip ratio $\epsilon { = } 0 . 2$ . Judge rewards are normalized to $[ 0 , 1 ]$ via min-max scaling before being passed to the GRPO objective.

Generation and evaluation. All models generate responses with temperature 0.7, top-p 0.9. Peer judges generate scores greedily (temperature $1 0 ^ { - 5 }$ , max 64 tokens). Evaluation uses task-specific formats: multiple-choice benchmarks are scored by exact match; AlpacaEval uses a reward model;

open-ended scientific tasks (SMDD, AssayBench) use domain-specific metrics as described in Section 3. All baselines use identical generation and evaluation settings.

## D MODEL POOL CONFIGURATIONS

Pool 1 and Pool 3 model identities are described in Section 3. Table 5 lists the 8 models comprising Pool 2 (LLMs from Diverse Academic Research), sourced from Feng et al. (2026b).

Table 5: Pool 2 model configurations (LLMs from Diverse Academic Research).
<table><tr><td>ID</td><td>HuggingFace identifier</td><td>Specialty</td></tr><tr><td>M0</td><td>chtmp223/Qwen2.5-7B-CLIPPER</td><td>Reasoning/alignment</td></tr><tr><td>M1</td><td>chengq9/ToolRL-Qwen2.5-3B</td><td>Tool use / RL</td></tr><tr><td>M2</td><td>AgentFlow/agentflowplanner-7b</td><td>Agent planning</td></tr><tr><td>M3</td><td>nanami/ladder-last16L-1lama3.1-8b-instruct-sft4k</td><td>Instruction following</td></tr><tr><td>M4</td><td>viswavi/qwen2.5-rlcf</td><td>RL from feedback</td></tr><tr><td>M5</td><td>milli19/promptmii-1lama3.1-8b-instruct</td><td>Prompt optimization</td></tr><tr><td>M6</td><td>Zhengping/conditionalprobability-regression</td><td>Calibration</td></tr><tr><td>M7</td><td>yale-nlp/MDCure-Qwen2-7B-Instruct</td><td>Medical</td></tr></table>

## E DATASET STATISTICS

Table 6 summarizes the instruction pool and test split sizes for each benchmark. The instruction pool (Dev) is drawn from the development set of each benchmark; 80% is used as the pool of candidate instructions available to the EXP3 leader and DPO/GRPO training, and the remaining 20% serves as a held-out validation set for checkpoint selection. The test set is used exclusively for final evaluation.

Table 6: Dataset sizes. Dev: full development set; 80% forms the instruction pool for training and 20% is held out for validation. Test: held-out evaluation set.
<table><tr><td>Benchmark</td><td>Dev</td><td>Test</td></tr><tr><td>BixBench</td><td>102</td><td>103</td></tr><tr><td>LabBench</td><td>578</td><td>578</td></tr><tr><td>SMDD</td><td>135</td><td>137</td></tr><tr><td>AssayBench</td><td>218</td><td>334</td></tr><tr><td>GPQA-Diamond</td><td>99</td><td>99</td></tr><tr><td>MATH</td><td>956</td><td>956</td></tr><tr><td>HumanEval</td><td>50</td><td>114</td></tr><tr><td>MBPP</td><td>487</td><td>487</td></tr><tr><td>AlpacaEval</td><td>2000</td><td>1000</td></tr><tr><td>IFEval</td><td>250</td><td>250</td></tr><tr><td>TruthfulQA</td><td>200</td><td>617</td></tr><tr><td>CulturalBench</td><td>613</td><td>613</td></tr></table>

## F BASELINE CONFIGURATIONS

All baselines use the same model pools, generation hyperparameters (temperature 0.7, top-p 0.9), and evaluation protocols as STACKELBERG ALIGNMENT. Training-based baselines (Sparta Alignment, Multiagent FT, AggLM, Trained Router) are given equal gradient updates to control for compute.

Sparta Alignment uses the same duel-based preference learning loop as STACKELBERG ALIGNMENT but samples instructions uniformly from the same instruction pool, with the same opponent matching schedule and reputation system. Multiagent FT collects debate-style responses from all models and fine-tunes each with supervised cross-model distillation using LoRA $( \bar { 1 } \mathrm { r } 1 0 ^ { - 5 }$ , 3 epochs, same LoRA config as our DPO). AggLM trains an RL aggregator to select among model outputs; rollout batch size 4. Trained Router learns to route each query to the best model via supervised training (1 epoch, cross-entropy on dev-set oracle labels). Mixture of Agents (MoA) performs one round of response synthesis (proposer → aggregator), inference only. Heterogeneous Swarms runs 3 rounds of iterative refinement across the pool with population size 3. Multiagent Debate uses 1 round of cross-model critique and revision. Majority Vote uses plurality voting across all models at inference.

## G JUDGE PROMPT

All peer judges receive the following prompt, filled with the instruction and the response being evaluated. Scores are parsed from JSON output and fall back to regex extraction if needed; the default score on parse failure is 5.0.

Please judge the following response based on the question   
and the response to be evaluated.   
Question: {instruction}   
Response to be evaluated: {response}   
Operation: Output ONLY a JSON object with one score in this   
exact format. Score must be in the range of 1 to 10.   
Your output should be like this:   
{"score": score}

The same prompt is used for both offline DPO judging and online GRPO reward generation. Judge scores are aggregated as a reputation-weighted mean: $\begin{array} { r } { \bar { s } = \sum _ { k } R _ { k } s _ { k } / \sum _ { k } R _ { k } } \end{array}$ , where $R _ { k }$ is the current reputation score of judge $M _ { k }$

## H COMPUTATIONAL COMPLEXITY ANALYSIS

Table 7 summarizes the per-iteration theoretical complexity of STACKELBERG ALIGNMENT and key baselines in terms of model forward passes F, training steps S, pool size m, duels per iteration D, instruction pool size K, and dataset size N.

Table 7: Per-iteration theoretical complexity. m: pool size; D: duels per iteration; N: dataset size; K: instruction pool size. “Inference only” methods have S = 0.
<table><tr><td>Method</td><td>Forward passes</td><td>Training steps</td></tr><tr><td>STACKELBERG ALIGNMENT</td><td>O(D · m)</td><td>O(m)</td></tr><tr><td>Sparta Alignment</td><td>O(D · m)</td><td>O(m)</td></tr><tr><td>Multiagent FT</td><td>O(N · m)</td><td>O(m)</td></tr><tr><td>AggLM</td><td>O(N · m)</td><td>O(1)</td></tr><tr><td>Trained Router</td><td>O(N · m)</td><td>O(1)</td></tr><tr><td>Mixture of Agents (MoA)</td><td>O(N · m)</td><td>0</td></tr><tr><td>Heterogeneous Swarms</td><td>O(N · m · r)</td><td>0</td></tr><tr><td>Multiagent Debate</td><td>O(N · m · r)</td><td>0</td></tr><tr><td>Majority Vote</td><td>O(N · m)</td><td>0</td></tr></table>

Discussion. STACKELBERG ALIGNMENT and Sparta Alignment share the same asymptotic complexity: each of the D duels requires two generation passes (the combatants) plus m−2 judge passes, giving O(D · m) forward passes per iteration, followed by m independent LoRA training steps. The EXP3 leader adds only an O(K) weight update per duel, which is negligible compared to model forward passes (K=64 instructions vs. billions of parameters per model pass). STACKELBERG ALIGNMENT therefore introduces zero additional model calls relative to Sparta Alignment; all gains come from reallocating the fixed duel budget toward more informative instructions, not extra compute.

Multiagent FT, AggLM, and Trained Router each require $O ( N \cdot m )$ forward passes because they process all N training instructions with all m models before each training update; with $N \gg D \left( \mathrm { e . g . } \right)$ N=956 vs. D=64 for MATH), this is substantially more expensive per iteration. Inference-only methods (MoA, Heterogeneous Swarms, Debate, Majority Vote) pay no training cost but incur repeated forward passes at inference time; Swarms and Debate multiply by the number of refinement rounds r. In our setting $( D { = } 6 4 , m { \leq } 9 )$ , the per-iteration forward-pass count for STACKELBERG ALIGNMENT is at most $6 4 \times 9 = 5 7 6$ model calls, vs. up to $9 5 6 \times 9 = 8 , 6 0 4$ for full-dataset sweeps.

## I LIMITATIONS

STACKELBERG ALIGNMENT requires running all models in the pool simultaneously during the combat and judgment phases, which increases memory and compute requirements relative to single-model fine-tuning; this is a shared characteristic of all training-time multi-LLM collaboration methods. The quality and diversity of the instruction pool X are important: a pool that does not span the capability frontier of the model pool limits the learning signal the EXP3 leader can identify. As models improve across iterations, productive disagreement naturally narrows, suggesting that combining STACKEL-BERG ALIGNMENT with mechanisms for dynamic instruction augmentation or pool expansion is a promising direction for future work. Finally, the reputation system initializes all models equally and may take several iterations to reliably differentiate model strengths on highly imbalanced pools; warm-starting reputations from pre-measured benchmark performance is a natural extension.