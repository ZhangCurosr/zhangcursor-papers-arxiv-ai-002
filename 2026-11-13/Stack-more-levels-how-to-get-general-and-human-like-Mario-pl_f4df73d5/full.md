# Stack more levels: how to get general and human-like Mario playing

Shinichi Arimura, Yuchen Li, and Julian Togelius

New York University

Brooklyn, New York, USA

sa9090@nyu.edu, yl6394@nyu.edu, julian@togelius.com

Abstract—Super Mario Bros. is widely used in game AI and procedural content generation (PCG), yet it remains unclear whether a model-free agent can reliably win unseen levels. To address this question, we train Proximal Policy Optimization (PPO) agents on increasingly large sets of PCG-generated Mario levels and evaluate their generalization on 1000 held-out test levels. We then shape competent PPO policies into three gameplay archetypes—runner, killer, and collector. Finally, we fine-tune these PPO policies using Diffusion-Reward Adversarial Imitation Learning (DRAIL) with human gameplay demonstrations. We find that increasing the number of distinct PCG-generated training levels generally improves unseen-level performance, and that fine-tuning PPO policies with DRAIL increases action-level human-likeness while preserving much of the persona induced by PPO pretraining.

Index Terms—game AI, reinforcement learning, Super Mario Bros., human-like agents, playstyle

## I. INTRODUCTION

Perhaps surprisingly, it remains unclear whether any agent can reliably play Super Mario Bros in a real-time model-free setting and win unseen levels. There are strong planning agents (such as the ubiquitous A\* agent by Baumgarten [1]), but those require access to a fast forward model, something that most games do not give you. There are also agents that can play individual levels well, i.e., agents that overfit [2].

Super Mario Bros (SMB), in the form of the Mario AI framework [1], is a widely used benchmark, and a competent player agent underpins much PCG and player-modeling research [3], [4], [5]. As an archetypical side-scrolling platformer, SMB also exposes a question that remains open: what it takes to play even a “simple” game well across unseen levels [6], [7]. This concern is not specific to SMB; much reinforcement learning research, including work on the Atari ALE framework [8], evaluates agents that overfit to a small set of levels [9], [10], [11].

Here, we start by asking the simple question if we can train a model-free reinforcement learning agent to play Mario well on unseen levels, and if so, how many levels it must be trained on. After establishing that this is possible (mostly by adding more levels to the training set), we investigate if we can get them to play in different human-like ways. Finally, we investigate whether we can make these policies more human-like by finetuning them with imitation learning [12]. In this work, we use

”human-like” in a narrow, action-level sense: a policy is more human-like if, on held-out human states, its action choices agree more often with the recorded human action and its marginal action distribution is closer to that of human players. We do not claim human-likeness at the level of strategy, intent, or subjective perception. Our contributions are: (i) trainingpool size strongly affects unseen-level performance for modelfree Mario agents, and (ii) DRAIL fine-tuning increases actionlevel human-likeness while preserving the persona learned during PPO pretraining.

## II. RELATED WORK

Mario agents and model-free generalization. The Mario AI benchmark has a long history of strong search- and planning-based agents [13], [14]. However, such agents often rely on a fast forward model, whereas our agents are modelfree policies that act directly from observations. We use Proximal Policy Optimization (PPO), a widely used policy-gradient algorithm [15], placing our work closer to RL generalization studies in benchmarks such as ALE, where overfitting to narrow environment distributions is a central concern [8].

Human-like and playstyle agents. Recent work has explored controllable or human-like game agents through interactive reinforcement learning, behavior cloning, adversarial imitation, and macro-action abstraction [16], [17], [18], [19], [20]. Behavior cloning predicts human actions from demonstrations, but can suffer from rollout distribution shift. Adversarial imitation methods such as Generative Adversarial Imitation Learning (GAIL) instead match expert and agent stateaction distributions through interaction [21]. Diffusion-Reward Adversarial Imitation Learning (DRAIL) extends this idea with a diffusion-based discriminator that provides smoother imitation rewards, making it suitable as a post-training shaping signal for competent PPO personas [22].

## III. METHOD

Our method has three stages: PPO training for level generalization, segment-reward shaping for playstyles, and DRAIL fine-tuning with human demonstrations.

## A. Training Levels and Evaluation Split

All levels are generated using Experience-Driven PCG via Reinforcement Learning [7] and split into disjoint training and test sets. We filter the test set to levels with more than 10 coins and more than 10 enemies, yielding 1000 held-out levels with enough opportunities for collection and enemy interaction.

To study level quantity, we create four fixed training pools with N ∈ {100, 1000, 10000, 50000} distinct PCG-generated levels. PPO episodes are sampled using inverse-visit weighting, so less-visited levels are sampled more often. All agents share the same architecture and PPO configuration; only pool size and playstyle reward differ.

## B. Segment-Based Playstyle Rewards

We train three playstyle archetypes: runner, killer, and collector. The runner is encouraged to complete levels directly, the killer to engage enemies, and the collector to gather coins and power-ups. Instead of dense horizontal progress, we use 16-tile segment checkpoints, preserving completion while allowing local detours for collection or enemy engagement. All agents share the same base competence reward:

$$
r _ { t } ^ { \mathrm { b a s e } } = w _ { \mathrm { s e g } } \Delta S _ { t } + w _ { \mathrm { w i n } } \mathbb { 1 } [ \mathrm { W I N } _ { t } ] .\tag{1}
$$

Here, $\Delta S _ { t }$ is the number of newly reached 16-tile segment checkpoints at time t, and $\mathbb { 1 } [ \mathrm { W I N } _ { t } ]$ indicates whether the episode ends in a win. Each segment gives reward only the first time it is reached.

The final reward is the base and a style-specific bonus:

$$
r _ { t } ^ { \mathrm { r u n n e r } } = r _ { t } ^ { \mathrm { b a s e } } ,\tag{2}
$$

$$
r _ { t } ^ { \mathrm { c o l l e c t o r } } = r _ { t } ^ { \mathrm { b a s e } } + w _ { c } \Delta C _ { t } + w _ { m } \Delta M _ { t } + w _ { f } \Delta F _ { t } ,\tag{3}
$$

$$
r _ { t } ^ { \mathrm { k i l l e r } } = r _ { t } ^ { \mathrm { b a s e } } + w _ { k } \Delta K _ { t } ^ { * } .\tag{4}
$$

Here, $\Delta C _ { t } , \ \Delta M _ { t }$ , and $\Delta F _ { t }$ denote newly collected coins, mushrooms, and fire flowers, and $\Delta K _ { t } ^ { * }$ denotes newly credited enemy kills, capped per segment to prevent farming. We set $w _ { \mathrm { s e g } } = 1 . 0 , w _ { \mathrm { w i n } } = 5 . 0 $ , and $w _ { c } = w _ { m } = w _ { f } = w _ { k } = 2 . 5$

Thus, all agents share the same completion objective, but differ in their secondary incentives. The runner uses no extra time reward because, without secondary objectives, detours do not increase return.

## C. DRAIL Fine-Tuning

Reward-shaped PPO agents can be competent and stylistically distinct, but still non-human-like. We fine-tune each policy with DRAIL [22], whose discriminator Dφ $D _ { \phi }$ predicts whether $\left( o _ { t } , a _ { t } \right)$ comes from human demonstrations or the current policy. The imitation reward is

$$
r _ { t } ^ { \mathrm { D R A I L } } = \log D _ { \phi } \big ( o _ { t } , a _ { t } \big ) - \log \big ( 1 - D _ { \phi } \big ( o _ { t } , a _ { t } \big ) \big ) .\tag{5}
$$

Let $\boldsymbol { r } _ { t } ^ { \mathrm { e n v } }$ be the original PPO persona reward and let λ<sub>DRAIL</sub> weight the imitation reward. During PPO→DRAIL finetuning, we use

$$
r _ { t } ^ { \mathrm { F T } } = r _ { t } ^ { \mathrm { e n v } } + \lambda _ { \mathrm { D R A I L } } r _ { t } ^ { \mathrm { D R A I L } } .\tag{6}
$$

We set $\lambda _ { \mathrm { D R A I L } } ~ = ~ 1 . 2 , ~ 2 . 9$ , and 2.5 for runner, killer, and collector, selected independently via an Optuna sweep optimizing Action Agreement Ratio (AAR).

## D. Human Demonstration Split

The human demonstration pool is taken from the human gameplay study of [23] and contains 220 action-state trajectories collected under a five-life protocol. For DRAIL training, we select 50 style-filtered trajectories per archetype: short-run trajectories for runner, high-kill trajectories for killer, and highcoin trajectories for collector. These sets can overlap, since a trajectory may satisfy more than one criterion; the remaining 95 unique trajectories are held out for evaluation. We evaluate Direct DRAIL and PPO→DRAIL rollouts with five lives to match the demonstration collection protocol.

## IV. EXPERIMENTS AND RESULTS

We evaluate three questions: (i) does increasing the number of training levels improve unseen-level generalization, (ii) do segment rewards produce distinct playstyles, and (iii) does PPO→DRAIL fine-tuning improve human-likeness?

## A. Metrics and Statistical Tests

Competence is measured by win rate and completion. For PPO level-scaling, we use a one-life, best-of-10 protocol: each level is counted as won if any of 10 stochastic rollouts reaches the goal, and completion is the maximum progress across rollouts. This measures per-level capability under repeated attempts, not single-rollout success. DRAIL rollouts use five lives to match the human demonstrations.

Playstyle is measured by a target metric for each archetype: episode length for runner, kill ratio for killer, and coin ratio for collector. For runner, fewer steps indicate shorter episodes, interpreted together with win rate and completion to avoid rewarding fast failure. Kill ratio is killed enemies divided by the number of enemies initially present in the level, and coin ratio is collected coins divided by total coins in the level. Kill ratio can exceed 1 because enemies may respawn.

Human-likeness is measured by Action Agreement Ratio (AAR), following Holmgard et al.’s decision-level comparison˚ of agent and human play styles [24]. AAR is the fraction of held-out human states where the agent’s most likely action matches the human action; higher AAR indicates stronger action-level agreement. Avg. AAR and Avg. Action JS are the simple averages over the runner, killer, and collector policies.

For win rate, we fit a logistic mixed-effects model with training-pool size, playstyle, and their interaction as fixed effects, and random intercepts for level and seed. Completion and playstyle contrasts use paired bootstrap CIs over levels, with Holm-corrected Wilcoxon signed-rank tests for playstyle p values. AAR uses paired bootstrap CIs over the 95 held-out human trajectories.

## B. PPO Level Generalization and Playstyle

Table I reports the one-life PPO level-scaling evaluation. Increasing the number of training levels generally improves unseen-level win rate and completion. Runner improves from 9.1% to 64.8% and collector from 20.2% to 85.5%, the best final win rate, while killer peaks at 82.3% (10k) before dropping to 68.0% (50k).

TABLE I  
ONE-LIFE PPO EVALUATION ON 1000 HELD-OUT TEST LEVELS.
<table><tr><td>Agent</td><td>Levels</td><td>Win</td><td>Compl.</td><td>Kill</td><td>Coin</td></tr><tr><td rowspan="4">Runner</td><td>100</td><td>0.091</td><td>0.415</td><td>0.092</td><td>0.125</td></tr><tr><td>1k</td><td>0.462</td><td>0.709</td><td>0.135</td><td>0.238</td></tr><tr><td>10k</td><td>0.630</td><td>0.806</td><td>0.142</td><td>0.248</td></tr><tr><td>50k</td><td>0.648</td><td>0.817</td><td>0.151</td><td>0.253</td></tr><tr><td rowspan="4">Killer</td><td>100</td><td>0.222</td><td>0.561</td><td>0.384</td><td>0.158</td></tr><tr><td>1k</td><td>0.540</td><td>0.776</td><td>0.638</td><td>0.221</td></tr><tr><td>10k</td><td>0.823</td><td>0.926</td><td>0.846</td><td>0.307</td></tr><tr><td>50k</td><td>0.680</td><td>0.832</td><td>0.747</td><td>0.247</td></tr><tr><td rowspan="4">Collector</td><td>100</td><td>0.202</td><td>0.549</td><td>0.174</td><td>0.382</td></tr><tr><td>1k</td><td>0.535</td><td>0.786</td><td>0.267</td><td>0.653</td></tr><tr><td>10k</td><td>0.751</td><td>0.888</td><td>0.326</td><td>0.804</td></tr><tr><td>50k</td><td>0.855</td><td>0.929</td><td>0.321</td><td>0.845</td></tr></table>

TABLE II  
ONE-LIFE GENERALIZATION STATISTICS ON HELD-OUT LEVELS.
<table><tr><td>Agent</td><td> $\mathrm { O R } _ { 1 0 \times }$  Win</td><td>10k→50k Win</td><td>10k→50k Compl.</td></tr><tr><td>Runner</td><td>3.70 [3.65, 3.74] OR 1.12,</td><td> $p = . 0 9 9 + 0 . 0 1 1$ </td><td>[-0.014, 0.034]</td></tr><tr><td>Killer</td><td>2.83 [2.76, 2.90] OR 0.37,</td><td> $p < . 0 0 1$ </td><td>-0.094 [-0.112, -0.075]</td></tr><tr><td></td><td>Collector 4.34 [4.23, 4.46] OR 2.30,</td><td> $p < . 0 0 1$ </td><td>+0.041 [0.026, 0.057]</td></tr></table>

$\mathrm { O R } _ { 1 0 \times }$ is per tenfold level increase; 10k→50k is the final adjacent contrast.

The logistic mixed-effects model confirms a significant positive effect of training-level count on unseen-level win probability. Each tenfold increase multiplies the odds of winning by 3.70 for runner, 2.83 for killer, and 4.34 for collector. To avoid forcing a log-linear trend, we also fit a categorical-pool win model and test adjacent training-pool contrasts. The 10k to 50k contrast shows a playstyle-dependent pattern: runner plateaus, killer decreases, and collector continues improving. Completion shows the same qualitative pattern under paired bootstrap. We therefore interpret the result as positive scaling overall with no single universal saturation point. One possible explanation for the killer drop is that all pool sizes were trained for the same total number of timesteps; with 50k levels, each individual level is visited less often, and the riskier enemyengagement behavior may require more samples to stabilize than running or collecting.

Table III confirms that the final 50k agents express distinct playstyles. Runner has significantly shorter episodes than killer and collector, killer has significantly higher kill ratio than both runner and collector, and collector has significantly higher coin ratio than both runner and killer. All six planned Wilcoxon tests remain significant after Holm correction.

These results show that segment-based rewards induce distinct playstyles while preserving the shared goal of level completion. Runner shortens episodes, killer increases enemy engagement, and collector increases resource collection.

## C. DRAIL and Human-Likeness

Table IV reports human-likeness on the 95 held-out human trajectories. AAR measures exact agreement with human actions, while Action JS measures Jensen–Shannon divergence between the marginal action distributions of the agent and human data. Thus, higher is better for AAR, while lower is better for Action JS. Avg. AAR and Avg. Action JS are computed as the simple averages of the corresponding runner, killer, and collector scores.

TABLE III  
PLAYSTYLE DIFFERENTIATION AT THE FINAL 50K PPO CHECKPOINT.
<table><tr><td>Comparison</td><td>Mean Diff.</td><td>95% CI</td><td>Holm p</td></tr><tr><td>K − R steps</td><td>116.737</td><td>[96.796, 137.175]</td><td> $< 1 0 ^ { - 4 9 }$ </td></tr><tr><td> $\mathrm { ~ C ~ - ~ R ~ s t e p s }$ </td><td>187.274</td><td>[165.583, 210.795]</td><td> $< 1 0 ^ { - 9 0 }$ </td></tr><tr><td> $\mathrm { ~ K ~ - ~ R ~ k i l l }$ </td><td>0.598</td><td>[0.578, 0.619]</td><td> $< 1 0 ^ { - 1 5 0 }$ </td></tr><tr><td> $\mathrm { ~ K ~ - ~ C ~ k i l l }$ </td><td>0.425</td><td>[0.401, 0.448]</td><td> $< 1 0 ^ { - 1 2 3 }$ </td></tr><tr><td> $\mathbf { C } \mathrm { ~ - ~ } \mathbf { R }$  coin</td><td>0.593</td><td>[0.575, 0.612]</td><td> $< 1 0 ^ { - 1 5 8 }$ </td></tr><tr><td> $\mathbf { C } \mathrm { ~ - ~ } \mathbf { K }$  coin</td><td>0.598</td><td>[0.580, 0.616]</td><td> $< 1 0 ^ { - 1 5 9 }$ </td></tr></table>

R = Runner, K = Killer, C = Collector.

The original PPO agents have near-zero AAR, showing that strong reward-shaped policies are not action-level human-like. Direct DRAIL achieves the strongest pure imitation score, with Avg. AAR 0.397 and Avg. Action JS 0.103. PPO→DRAIL also substantially improves over PPO, reaching Avg. AAR 0.324 and reducing Avg. Action JS from 0.643 to 0.218. The paired trajectory-level bootstrap confirms that PPO→DRAIL significantly improves AAR over PPO for all three roles. The improvement is strongest for runner and collector, while the killer improvement is smaller.

Table V shows why imitation metrics are insufficient. Direct DRAIL achieves the best AAR and Action JS, but struggles to produce distinct rollout personas because the human demonstrations are not clearly separated by style. PPO→DRAIL starts from a reward-shaped PPO policy and better preserves target behaviors: runner remains faster, killer kill ratio improves from 0.853 to 1.109, and collector coin ratio improves from 0.565 to 0.915, providing a practical compromise between persona preservation and human-likeness.

## V. CONCLUSION AND FUTURE WORK

Our results suggest three lessons. First, training-level quantity is important for model-free Mario competence: more distinct PCG-generated levels improve generalization to 1000 held-out levels, although the effect is playstyle-dependent. Collector scales most strongly, while killer peaks at 10k levels, suggesting that enemy-focused behavior is more sensitive to the training-level distribution and total training budget.

Second, segment-based rewards induce distinct playstyles while preserving the shared completion objective. Rewarding progress only at 16-tile checkpoints avoids constantly pushing the policy rightward, leaving room for collector and killer behavior within each segment. The final PPO agents remain strongly separated across all six planned playstyle contrasts.

Third, human demonstrations are useful but noisy. Direct DRAIL achieves the strongest imitation metrics, but imitation alone does not recover task competence or persona behavior.

TABLE IV  
HUMAN-LIKENESS ON 95 HELD-OUT HUMAN TRAJECTORIES.
<table><tr><td>Method</td><td>Runner AAR</td><td>Killer AAR</td><td>Collector AAR</td><td>Avg. AAR</td><td>Avg. Action JS</td></tr><tr><td>PPO</td><td>0.016</td><td>0.007</td><td>0.021</td><td>0.015</td><td>0.643</td></tr><tr><td>Direct DRAIL</td><td>0.355 (+0.339 [0.322, 0.356])</td><td>0.416 (+0.409 [0.394, 0.424])</td><td>0.420 (+0.399 [0.379, 0.417])</td><td>0.397</td><td>0.103</td></tr><tr><td>PPO→DRAIL</td><td>0.387 (+0.371 [0.353, 0.388])</td><td>0.192 (+0.185 [0.172, 0.198])</td><td>0.393 (+0.372 [0.354, 0.388])</td><td>0.324</td><td>0.218</td></tr></table>

TABLE V  
FIVE-LIFE ROLLOUT EVALUATION FOR PERSONA PRESERVATION.
<table><tr><td rowspan="2">Method</td><td colspan="3">Runner</td><td colspan="3">Killer</td><td colspan="3">Collector</td></tr><tr><td>Win</td><td>Compl.</td><td>Steps</td><td>Win</td><td>Compl.</td><td>Kill</td><td>Win</td><td>Compl.</td><td>Coin</td></tr><tr><td>PPO</td><td>0.995</td><td>0.998</td><td>435.0</td><td>0.985</td><td>0.993</td><td>1.023</td><td>0.990</td><td>0.996</td><td>0.927</td></tr><tr><td>Direct DRAIL</td><td>0.922</td><td>0.968</td><td>1042.1</td><td>0.927</td><td>0.973</td><td>0.853</td><td>0.904</td><td>0.967</td><td>0.565</td></tr><tr><td>PPO→DRAIL</td><td>0.971</td><td>0.984</td><td>915.2</td><td>0.968</td><td>0.989</td><td>1.109</td><td>0.990</td><td>0.994</td><td>0.915</td></tr></table>

PPO→DRAIL starts from a competent reward-shaped policy and fine-tunes toward human-like actions while retaining more persona, suggesting a practical pipeline: train for competence, shape playstyle, then fine-tune for human-likeness.

The main limitations are that we vary training-level quantity rather than explicitly measuring level diversity, and that human-likeness evaluation uses a small demonstration pool and exact-action AAR, which may under-credit different but plausible human actions. Future work will use explicit leveldiversity measures, and larger human datasets.

## REFERENCES

[1] S. Karakovskiy and J. Togelius, “The Mario AI benchmark and competitions,” IEEE Transactions on Computational Intelligence and AI in Games, vol. 4, no. 1, pp. 55–67, 2012.

[2] J. Togelius, S. Karakovskiy, J. Koutn´ık, and J. Schmidhuber, “Super mario evolution,” in 2009 ieee symposium on computational intelligence and games. IEEE, 2009, pp. 156–161.

[3] N. Shaker, J. Togelius, G. N. Yannakakis, B. Weber, T. Shimizu, T. Hashiyama, N. Sorenson, P. Pasquier, P. Mawhorter, G. Takahashi et al., “The 2010 mario ai championship: Level generation track,” IEEE Transactions on Computational Intelligence and AI in Games, vol. 3, no. 4, pp. 332–347, 2011.

[4] A. Summerville, S. Snodgrass, M. Guzdial, C. Holmgard, A. K. Hoover,˚ A. Isaksen, A. Nealen, and J. Togelius, “Procedural content generation via machine learning (pcgml),” IEEE Transactions on Games, vol. 10, no. 3, pp. 257–270, 2018.

[5] C. Pedersen, J. Togelius, and G. N. Yannakakis, “Modeling player experience for content creation,” IEEE Transactions on Computational Intelligence and AI in Games, vol. 2, no. 1, pp. 54–67, 2010.

[6] V. Volz, J. Schrum, J. Liu, S. M. Lucas, A. Smith, and S. Risi, “Evolving Mario levels in the latent space of a deep convolutional generative adversarial network,” in Proceedings of the 2018 Genetic and Evolutionary Computation Conference, 2018, pp. 221–228.

[7] T. Shu, J. Liu, and G. N. Yannakakis, “Experience-driven PCG via reinforcement learning: A Super Mario Bros study,” in 2021 IEEE Conference on Games. IEEE, 2021, pp. 1–9.

[8] M. G. Bellemare, Y. Naddaf, J. Veness, and M. Bowling, “The arcade learning environment: An evaluation platform for general agents,” Journal of artificial intelligence research, vol. 47, pp. 253–279, 2013.

[9] M. C. Machado, M. G. Bellemare, E. Talvitie, J. Veness, M. Hausknecht, and M. Bowling, “Revisiting the arcade learning environment: Evaluation protocols and open problems for general agents,” Journal of Artificial Intelligence Research, vol. 61, pp. 523–562, 2018.

[10] K. Cobbe, O. Klimov, C. Hesse, T. Kim, and J. Schulman, “Quantifying generalization in reinforcement learning,” in International conference on machine learning. PMLR, 2019, pp. 1282–1289.

[11] K. Cobbe, C. Hesse, J. Hilton, and J. Schulman, “Leveraging procedural generation to benchmark reinforcement learning,” in International conference on machine learning. PMLR, 2020, pp. 2048–2056.

[12] A. Hussein, M. M. Gaber, E. Elyan, and C. Jayne, “Imitation learning: A survey of learning methods,” ACM Computing Surveys (CSUR), vol. 50, no. 2, pp. 1–35, 2017.

[13] J. Togelius, S. Karakovskiy, and R. Baumgarten, “The 2009 mario ai competition,” in IEEE Congress on Evolutionary Computation. IEEE, 2010, pp. 1–8.

[14] J. Togelius, N. Shaker, S. Karakovskiy, and G. N. Yannakakis, “The mario ai championship 2009-2012,” Ai magazine, vol. 34, no. 3, pp. 89–92, 2013.

[15] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, “Proximal policy optimization algorithms,” arXiv preprint arXiv:1707.06347, 2017.

[16] C. Arzate Cruz and T. Igarashi, “Mariomix: Creating aligned playstyles for bots with interactive reinforcement learning,” in Extended abstracts of the 2020 annual symposium on computer-human interaction in play, 2020, pp. 134–139.

[17] P. L. P. De Woillemont, R. Labory, and V. Corruble, “Automated play-testing through rl based human-like play-styles generation,” in Proceedings of the AAAI Conference on Artificial Intelligence and Interactive Digital Entertainment, vol. 18, no. 1, 2022, pp. 146–154.

[18] B. Ingram, C. Van Alten, R. Klein, and B. Rosman, “Creating diverse play-style-centric agents through behavioural cloning,” in Proceedings of the AAAI Conference on Artificial Intelligence and Interactive Digital Entertainment, vol. 19, no. 1, 2023, pp. 255–265.

[19] W. Ahlberg, A. Sestini, K. Tollmar, and L. Gisslen, “Generating personas´ for games with multimodal adversarial imitation learning,” in 2023 IEEE Conference on Games (CoG). IEEE, 2023, pp. 1–8.

[20] J.-T. Guo, Y.-C. Chen, P.-C. Hsieh, K.-H. Ho, P.-W. Huang, T.-R. Wu, I. Wu et al., “Learning human-like rl agents through trajectory optimization with action quantization,” Advances in Neural Information Processing Systems, vol. 38, pp. 83 534–83 565, 2025.

[21] J. Ho and S. Ermon, “Generative adversarial imitation learning,” Advances in neural information processing systems, vol. 29, 2016.

[22] C.-M. Lai, H.-C. Wang, P.-C. Hsieh, Y.-C. F. Wang, M.-H. Chen, and S.-H. Sun, “Diffusion-reward adversarial imitation learning,” Advances in Neural Information Processing Systems, vol. 37, pp. 95 456–95 487, 2024.

[23] Z. Wang, Y. Li, H. Du, J. Liu, and G. N. Yannakakis, “Fun as moderate divergence: Evaluating experience-driven pcg via rl,” IEEE Transactions on Games, vol. 17, no. 2, pp. 360–373, 2025.

[24] C. Holmgard, A. Liapis, J. Togelius, and G. N. Yannakakis, “Personas˚ versus clones for player decision modeling,” in Entertainment Computing – ICEC 2014. Springer, 2014, pp. 159–166.