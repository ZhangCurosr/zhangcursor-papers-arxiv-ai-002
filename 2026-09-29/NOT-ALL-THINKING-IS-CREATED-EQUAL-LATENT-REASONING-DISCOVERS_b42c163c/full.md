# NOT ALL THINKING IS CREATED EQUAL:LATENT REASONING DISCOVERS A RECURRENTSEARCH ALGORITHM FOR DEPTH GENERALIZATION

Huzi Cheng University of Minnesota hzcheng15@gmail.com

Zhewei Zhang Independent Researcher zhzhewei36@gmail.com

## ABSTRACT

Large Language Models can perform multi-step reasoning and improve task performance through different forms of intermediate computation, from token-based traces to computation carried out in latent space. However, a question remains open: do these different forms of thinking rely on the same underlying mechanism? To address this, we train and compare five variants of the same GPTNeoX backbone from scratch on an extended multi-hop reasoning task (ProsQA-Ext): a vanilla model, a Chain-of-Thought (CoT) model, a Pause Token model, and two latent-reasoning models that are optimized end-to-end without intermediate reasoning traces. We find that, strong in-distribution (ID) performance does not guarantee depth generalization. Vanilla, CoT, and Pause Token models solve ID problems well, but rely largely on local graph features and generalize poorly to out-of-distribution (OOD) problems with longer hops. In contrast, latent variants generalize better and show internal dynamics consistent with forward reachability propagation on the graph. Causal interventions and circuit analysis localize this computation to a sparse recurrent search circuit in the bottleneck latent model: an attention head retrieves graph relations, an MLP and the residual stream update the reachability state across recurrent steps, while multiple attention heads together then do the candidate matching. Together, these results show that different thinking mechanisms can learn distinct computational solutions, even at similar ID performance. In this setting, latent recurrence supports a reusable forward-search algorithm that generalizes beyond the training depth.

## 1 INTRODUCTION

Prepending “Let’s think step by step” to a prompt can improve pretrained language models’ performance on reasoning tasks (Kojima et al., 2022; Wei et al., 2022). Models that fail to answer directly can sometimes solve the same question by first generating intermediate steps. More recently, this approach has been successfully scaled by baking the reasoning traces into the training rather than prompting (Chung et al., 2024; Ho et al., 2023; Magister et al., 2023), which enables smaller models to solve problems where CoT prompting alone is ineffective.

However, whether a model actually follows the reasoning traces it generates remains debated. Part of the reasoning traces can be replaced or removed without hurting the final answer (Lanham et al., 2023; Zhao et al., 2026). Models can also benefit from intermediate steps using meaningless filler or pause tokens (Pfau et al., 2024; Goyal et al., 2024). Together, these findings suggest that useful intermediate computation need not be realized as a verbally meaningful reasoning trace. Coconut (Hao et al., 2025) and related works have demonstrated alternative ways to perform such computation by feeding high-dimensional vectors, instead of tokens, directly into the model (Wei et al., 2025). The mechanisms underlying this latent computation remain poorly understood.

Symbolic reasoning tasks were widely used to probe the circuits and computation inside language models (Wu et al., 2025; Brinkmann et al., 2024). Zhu et al. (2025) showed that latent thoughts theoretically can encode multiple search frontiers in superposition and enable parallel search. However, recent work finds that similar patterns also arise in models without recurrence and do not always causally affect the answer (Aswal et al., 2026; Rizvi-Martel et al., 2026), leaving its causal role contested. More broadly, it remains unclear how the learned computation differs across thinking interfaces, and what mechanisms support generalization beyond the training distribution.

To investigate this question, we focus on five model variants: a vanilla model, a CoT model, a Pause Token model, and two latent-reasoning models based on Coconut. One full-latent model retains access to all previous tokens, while the other bottleneck-latent model can only rely on the intermediate hidden representations when generating answers. We train these five models on an extended version of the well-established ProsQA task. Notably, the latent variants are trained without intermediate reasoning traces and RL, allowing us to examine whether latent reasoning can discover a generalizable reasoning mechanism without being shown how to solve the task step by step and without slow trial-and-error process.

By testing these models on out-of-distribution (OOD) problems, we find that strong performance within the training range does not guarantee depth generalization, with the latent variants performing best on OOD problems. The strong generalization, together with the lack of shortcut effects in the latent models, indicates that they learn to reason rather than use surface heuristics. Further causal interventions in the bottleneck-latent model show that intermediate states carry intermediate variables during forward search that are reused and transformed across recurrent steps. We localize this computation to a sparse search circuit in which an attention head retrieves graph relations and an MLP, together with the residual stream, update the state for subsequent steps, and finally multiple attention heads read this information for candidate matching. These findings show that different forms of thinking can learn very different computational solutions, even at similar performance. Importantly, latent recurrence supports better discovery of a reusable computation that generalizes beyond the training depth.

## 2 METHODS

## 2.1 TASK

A signature of a model that understands rules and can reason with them is that it learns from small scale datasets and generalizes to unseen, more complex problems. In natural language problems, multi-hop symbolic reasoning, such as extended syllogisms, is a good candidate for such datasets: the level of difficulty, i.e., the number of hops, can be controlled, and the symbols used can be permuted without changing the meaning. In this study, adapted from ProsQA by Hao et al. (2025), we construct such a task, ProsQA-Ext. As shown in Fig. 1A, each ProsQA-Ext sample $( x , y ) \in \mathcal { D }$ describes a directed acyclic graph (DAG) $G = ( V , E )$ and a question about G. The graph description $g$ is a token sequence of premises of the form A is B., each representing a directed edge from A to B. The query q gives a root node $r \in V$ and two candidate nodes $c _ { 0 } , c _ { 1 } \in V$ . Together, they form the complete input $x = g \| q$ . Exactly one candidate is reachable from $r ,$ at a shortest distance of $H ;$ the other is either an isolated node or lies on a chain whose root is not r. We denote the reachable candidate by $c ^ { * }$ and the reference answer-token sequence by $y ,$ , which states that $r$ is $c ^ { * }$ . At a fixed $H ,$ node labels, premise order, and candidate positions are randomly sampled, yielding varied graph structures and inputs $x ,$ while the underlying reachability operation $f$ remains unchanged. Unlike ProsQA, we carefully control the generation of $G$ so that no superficial features can be exploited to infer $c ^ { * }$ (≈ 50% accuracy), and we use two separate datasets: a training set with $H \in \{ 3 , \ldots , 6 \}$ and a validation set with $\mathbf { \bar { \mathit { H } } } \in \{ 7 , \dots , 1 2 \}$

## 2.2 MODELS

We train five model variants on the same ProsQA-Ext dataset with the same tokenizer to examine how different forms of thinking address the symbolic reasoning problem. All variants use the same GPTNeoX backbone (number of layers=4, hidden size=256, dimensionality of $\mathrm { F F N } { = } 7 6 8 )$ . Unlike Hao et al. (2025), all models are trained from random initialization, ensuring their knowledge of the task comes completely through training, rather than possibly inherited from pretraining. The variants differ in their intermediate computation and answer readout (Fig. 1B). For all models, we denote the residual state at token position i after block ℓ by $h _ { i } ^ { ( \ell ) }$ , and denote $\ell = 0$ as the input to the first block. During prompt encoding, the token embedding $E _ { \theta }$ supplies $h _ { i } ^ { ( 0 ) } = E _ { \theta } ( x _ { i } )$ . After the final block, a final LayerNorm $N _ { \theta }$ and an output projection map $h _ { i } ^ { ( L ) }$ to next-token logits.

After the input x, the Direct variant generates the answer directly. The Chain-of-Thought variant first generates a proof and then the answer. Its training is supervised by both the shortest proofs (a sequence of premises forming the shortest path from r to $c ^ { * } )$ and final answers. In the Pause-token variant, before answer decoding, the model “thinks” by inserting K=6 identical learnable embeddings, $z _ { t } = E _ { \theta } { \left( < { \mathrm { P A U S E } } > \right) }$ The two latent variants, Full-latent and Bottleneck-latent, in their “thinking” process, instead, feed the normalized output of one step directly into the next, instead of decoding it into a token. This process can be described with

$$
z _ { 1 } = N _ { \theta } \Big ( h _ { n } ^ { ( L ) } \Big ) , z _ { t + 1 } = N _ { \theta } \Big ( h _ { n + t } ^ { ( L ) } \Big ) , \qquad 1 \le t < K ,\tag{1}
$$

in which each vector enters the first block as $h _ { n + t } ^ { ( 0 ) } = z _ { t }$ , where $n = | x |$

These two latent variants differ in the readout phase. The Full-latent model processes $z _ { K }$ with the prompt and recurrent cache retained, keeping both available during answer generation. In this sense, it is similar to the Coconut model (Hao et al. (2025)), but without human guidance involved. However, Full-latent can still access the g and $q$ during answer decoding, which may allow the model to exploit statistical patterns to solve the problem. To isolate the effect of direct access, we build the Bottleneck-latent model, which cannot access the recurrent key/value cache and has to re-encode the $z _ { 1 : K }$ as the sole prefix for answer generation. This allows us to examine what is happening inside the latent state trajectories and to decompose the reasoning with memorization.

![](images/e23661ab9e0a8c72676b6229efcc0a72a980fb7f7975df41b71e14f3d432c61b.jpg)

C  
![](images/94f4b13cd56dba24345a98794e3191d6594a6a82586127eab6e5cecfde164c97.jpg)

![](images/60e87ae05d75325cd6f2433dbf81699453393fdc310bca2ede355070b2cbc457.jpg)  
Figure 1: ProsQA-Ext task and reasoning performance. A shows an example $( H = 8 )$ with the correct path highlighted in green and distractor edges in gray. B shows the five model variants. C and D show free-generation accuracy on ID and OOD problems, respectively (error bars are SEM).

## 3 RESULTS

## 3.1 ID AND OOD DATASET PERFORMANCE

All models are trained on the same set of problems with $H \in \{ 3 , \ldots , 6 \}$ , using the same budget and optimization settings (see A.2). On the ID validation set, Bottleneck-latent and Full-latent achieve nearly perfect accuracy, and Pause-token is closely behind. Chain-of-Thought also achieves high accuracy overall, although its performance declines as the H increases. In contrast, without additional computational slots, Direct is worse than others and is more sensitive to the seeds.

We then test their generalization capability with OOD dataset $( H \in \{ 7 , \ldots , 1 2 \} )$ ). Without further training, the variants start to show divergent behaviors. Full-latent and Bottleneck-latent retain the highest accuracy, with Bottleneck-latent slightly ahead of Full-latent, and both clearly outperforming Pause-token and Direct. Unexpectedly, Chain-of-Thought performs worst despite its strong ID performance. Thus, models that perform similarly on the training range can generalize very differently beyond it (Fig. 1C,D).

## 3.2 HOW DO DIFFERENT MODELS SOLVE THE PROSQA-EXT TASK?

## 3.2.1 ALIGNMENT WITH FORWARD PROPAGATION

The divergent OOD performance suggests that these variants may reach the same answer through different computations. Two natural heuristic strategies are forward propagation from the query root and backward tracing from the candidate answers. We first test whether their representations track forward propagation. Using representational similarity analysis (RSA), we compare pairwise dissimilarities between model representations and algorithmic states, without assuming a one-to-one correspondence between model steps and algorithmic updates (Appendix A.3).

A  
![](images/529a91a2e100c8e76acedb5267ceea5178162980cc42388fd474a52dbb8e953b.jpg)

C  
ID  
![](images/d6599fa324ac302ebe47deb663a12c57c739db84ea8d7ad8a25b69e909d1e6dd.jpg)  
Figure 2: Representational alignment with forward graph propagation. A and B show RSA heatmaps for five variants on ID (4-hop) and OOD (8-hop) problems, respectively, using the same examples across variants. Each entry shows the Spearman correlation between pairwise modelrepresentation dissimilarities and pairwise graph-frontier dissimilarities at depth d. For $G = ( V , E )$ with query root $r ,$ , the propagation frontiers are defined by ${ \cal F } _ { 0 } = \{ r \}$ and $\overset { \cdot } { F _ { d + 1 } } = \{ v \in V : \exists u \in$ $F _ { d } , \ ( u , v ) \in E \}$ . Gray hatched cells denote undefined correlations. Color scales are shared across rows within each variant. C shows diagonality across 3–12-hop problems, with colors indicating variants. The dashed line separates ID (3–6 hops) from OOD (7–12 hops).

Direct and Pause-token show nearly no sequential alignment with forward propagation (Fig. 2A,B). Nevertheless, Chain-of-Thought shows some alignment, consistent with its supervision on step-bystep proofs, despite very poor OOD performance. The clearest diagonal-like patterns appear in Fulllatent and Bottleneck-latent, suggesting that they learn a forward-search-like procedure without intermediate supervision. We quantify this progression intuition with diagonality, which measures whether alignment shifts toward later model positions as algorithm depth increases (Appendix A.4). A score approaching one indicates a consistent progression, without requiring one model step per graph hop. Bottleneck-latent has the highest and most consistent diagonality across depths, followed by Full-latent, with the clearest separation from the other variants in OOD problems (Fig. 2C). The corresponding analysis of parallel backward tracing shows no clear, consistent sequential alignment across task depths (Appendix A.3, Fig. A.2).

## 3.2.2 LOCAL GRAPH SHORTCUTS DRIVE PREDICTIONS IN NON-LATENT MODELS

Latent models represent search-related intermediate variables, but no comparable alignment is found in Direct, Chain-of-Thought, or Pause-token. However, these three variants remained highly accurate within the training distribution, which raises the question of what supports their answers. One possibility is that they exploit shortcuts from local graph features. Such shortcuts are actually available since the graph-generation procedure inherited from ProsQA induces systematic degree asymmetries: correct candidates tend to have lower in-degree than incorrect ones, and the immediate successors of the query root that lead to the correct candidate tend to have lower in-degree and higher out-degree than the alternatives. These correlations provide local predictive cues, so we test whether model variants rely on them by manipulating local graph features.

We first ask whether the models use the in-degree of candidates when generating answers. To test this, we construct a matched-pair dataset, in which we select one or two edges not on the proof path and redirect their destinations to the correct candidate or to the incorrect candidate (Fig. 3A). This matched pair differs only in the in-degrees of two candidates, keeping all others the same. Note that this matched-pair dataset is constructed to isolate individual graph features and differs in structure from the OOD evaluation set in Section 3.1. Evaluating the five variants on this dataset shows that candidate in-degree strongly affects accuracy in Direct and Pause-token across the problems with different reasoning depths, but has much smaller effects on the latent models and minimally affects Chain-of-Thought (Fig. 3B, C).

A  
B  
![](images/7eaaa6712b9e81eb0e689c1bf31e9150e295c2990e1b8df2fa99fe1d3594ce62.jpg)

![](images/711717485545da0227dfdfc65276b8afdea5cd5193d1360da8199884816c7b6b.jpg)  
C

![](images/9059ad4c27bf18ee5a10b29b177f1b2d8cec106be3cf70627ac34d29af240ea9.jpg)

D  
E  
![](images/3dabe743bbce602d3270a858de8d3ad89fa9d1d25c38de05f34356644304b6cd.jpg)

![](images/0bf1f29ec6bc862c0366fc9b34dfc7e41c3c995b0a903e1303db14155c66168d.jpg)  
F

![](images/c95017e43dd6c4b9e1136eaf92e9e8930a108c390fa18912ff03048b05863ff8.jpg)  
Figure 3: Effects of candidate and successor degree on model predictions. A shows matched pairs that reverse the candidates’ relative in-degree while preserving the proof path and correct answer. B and C show the final-answer accuracy on problems with short and long reasoning depths. Solid and hatched bars indicate that the correct candidate has lower and higher in-degree, respectively. D shows matched pairs that switch which immediate successor of the query root leads to the correct candidate while preserving all node degrees and the correct final answer. E and F show final-answer accuracy for all five models and first-successor accuracy for CoT on problems with short and long reasoning depths. Solid and hatched bars indicate that the correct successor has a lower or higher in-degree minus out-degree, respectively. In schematics, (r) denotes the query root, (+) and (-) denote the correct and incorrect candidates, and green arrows indicate the proof path.

Next, we ask whether the models use local degree cues at the query root’s immediate successors. Unlike other variants, Chain-of-Thought explicitly identifies a successor in its first generated statement before producing the final answer, making it more vulnerable to the degree at the query root’s successors. To test this, we construct a second matched-pair dataset in which the query root has two immediate successors, with one leading to the correct candidate. Within each pair, we change the in/out degrees of successors by reassigning the source endpoints of non-proof outgoing edges from one successor to the other and redirecting the destination endpoints of non-proof incoming edges between them (Fig. 3D). These manipulations strongly affect both the first-step successor choice and the final answer in Chain-of-Thought, which tends to prefer the successor with lower in-degree minus out-degree, while having little effect on all other variants (Fig. 3E, F).

![](images/e28a0e3a49ae1bc91d0ad4db57fb368ab79bdb8eece7551585f4c0a4bcdc93c9.jpg)

![](images/c7b797114c5e4ce4583297da88bcc0ada2a2ae22b42855b5e66c325360121805.jpg)

Together, these results indicate that Direct and Pause-token rely substantially on candidate in-degree when predicting the answer, whereas Chain-of-Thought uses the degree of the query root’s successors both when selecting which successor to follow and when predicting the answer. In contrast, the latent models are less affected by either type of local graph structure.

## 3.3 A RECURRENT SEARCH ALGORITHM IN LATENT REASONING MODELS

## 3.3.1 SWAPPED CHAINS AS A PROBE OF A “SOFT” FOR-LOOP

Analyses in Fig. 2 suggest that, unlike other variants, latent reasoning models, especially Bottlenecklatent, may implement some recurrent algorithms that support generalization to longer hop problems. To further show if this causally holds, we swap the graph connectivity at each depth and use the change as a probe.

![](images/365cd7d40f41136d4e70fa7d93d7945fc21d7783d97d20ffc0e420823c38fcaf.jpg)

B  
![](images/49955f44bbad58cb64b3161dbc61713183436b15e3cd8d981a6830daa0d3a6e9.jpg)

![](images/0698f820eb925a360a78b3b5b045b791da086ec127ee92a30d5fda2bb5fc8916.jpg)

![](images/d52afcba8654c56f7cf9b256829dd213cd84fa4fc40a18e23ea8a74408a99b2b.jpg)

C  
![](images/682c2dacc6eff8d9a120ee3642c5ff0d5671c8c812351254224c66cef1701ea1.jpg)

![](images/965fd0ec389ed0a9d43bdc5fc2457d17fe71088982e9b09d7719d3ccc15c2db4.jpg)

![](images/87b2f47c8ab82c4a0a0f990270a77539931f3f34d58147302d5a539ae8bf004b.jpg)

![](images/c9389bfd5c2b939dc3cdb175705b2672376847d4a889e3d10adc2b9efe57c9e6.jpg)  
Figure 4: Controlled interventions probe the recurrent computation in latent reasoning models. A shows how swapping premises shifts the query root $r { } _ { \mathrm { { s } } }$ correct candidate from A to B. In B, the top row shows the normalized cosine distance between paired latent states for 8-hop problems with connectivity swaps at depth $d ,$ and the bottom row shows the fraction of pairs for which transplanting the latent state at step t redirects the answer to $y ^ { \prime } .$ . C shows OOD accuracy with one fewer $( K = 5 )$ the trained number $( K = 6 )$ , or one additional $( K = 7 )$ thinking position. Shaded region indicates 95% CI.

For each sample $x = g \| q$ , we build a matched one $x ^ { \prime } = g ^ { \prime } \| q$ by swapping the destinations of two edges at depth d: one on the solution chain and one on a matched distractor chain. This preserves the query root, node labels, premise order, and node degrees, but switches the reachable candidates, flipping the answer from y to $y ^ { \prime }$ (Fig. 4A). Next, we run both inputs to obtain the latent trajectories $z _ { 1 : K } ( x )$ and $z _ { 1 : K } ( x ^ { \prime } )$ . At step t, we replace $z _ { t } ( x )$ with $z _ { t } ( x ^ { \prime } )$ , keep the original prompt cache and preceding computation unchanged, and recompute the remaining latent states before generating the answer. We apply this probe to Bottleneck-latent and Full-latent, with Pause-token as a control. We measure the state-level effect of the connectivity change as the cosine distance between $z _ { t } ( x )$ and $z _ { t } ( \boldsymbol { x } ^ { \prime } )$ , normalized by subtracting the distance at $t = 1 ~ ( \mathrm { F i g . 4 B ~ t o p } )$ . The causal influence is the fraction of pairs for which transplantation changes the answer from y to y<sup>′</sup> (Fig. 4B bottom).

If the recurrent computation propagates reachability step by step, shallow perturbations should become effective earlier, and deep ones should influence the output only at later steps. This upper triangular pattern clearly emerges in Bottleneck-latent (Fig. 4B,C), while it is less clear in Full-latent and absent in Pause-token. These results suggest that, among the three variants, only Bottlenecklatent strongly adopts a recurrent algorithm.

Nevertheless, such computation might be “soft” rather than a strict “hard” for-loop, since a single latent step handles more than one exact hop. This soft-iteration hypothesis makes a further prediction: small perturbations to K should barely affect Bottleneck-latent’s accuracy. Indeed, only Bottleneck-latent retains its performance under $K \ : = \ : 5 , 6 , 7$ , whereas Full-latent is sensitive to $K = 5$ and Pause-token to $\bar { K } = 7 ( \mathrm { F i g } . 4 \mathrm { C } )$ . Together, these results support that Bottleneck-latent may implement $\mathrm { { a \ ^ { 6 6 } s o f f } ^ { \prime 5 } }$ forward search through its recurrent circuit.

## 3.3.2 LOCALIZE THE SPARSE CIRCUIT INSIDE THE BOTTLENECK LATENT MODEL

Motivated by the RSA and intervention analyses above, we examine which components support the recurrent computation in Bottleneck-latent and how this circuit enables OOD generalization (Fig. 5) by circuit pruning to the recurrent updates from $z _ { 1 } \ \mathrm { t o } \ z _ { K }$ . We check all 20 physical components in Bottleneck-latent, including 4× attention heads and $1 \times \mathrm { M L P }$ per layer in GPTNeoX. A component is removed if the remaining circuit retains at least 90% output consistency with the full model and an $R ^ { 2 } > 0 . 8$ for the candidate logit margin. This procedure (see A.6) greedily continues until no further component can be removed (Fig. 5A), leading to an 8-component sparse circuit (Fig. 5B) that preserves 91.9% output consistency and an $R ^ { 2 }$ of 0.837.

We next run the pruned circuit on other 7 to 12-hop samples that are not involved in pruning. The circuit retains 90.9% output consistency with the full model, above the 52.7% when these eight components are removed and 55.8% when a random size-matched subset is retained instead (Fig. 5C).

A  
![](images/bbd4bb8b75ea549b9bd94645a7dcfb6dcdf032b7562e0f1607690c90c3cb14e1.jpg)

B  
![](images/f4dbf7fa20cfd479b48e39b3d3147845dec0c67727e7d4e69d2bca35230d1240.jpg)

C  
![](images/96e1b3453c0c4ca9ffa5880856f5bc813dee7d30fce2c79cef478d148b86c997.jpg)

D  
![](images/6c939618f436472d1c6d9c32d29b4c743c80ad8d4d99fa85d79ff630623188e9.jpg)  
Figure 5: Localization of the recurrent circuit in Bottleneck-latent. Replacement based pruning (A) retains eight of twenty recurrent components (B). The selected circuit largely preserves candidate choices (C) and causal state-transfer effects (D), whereas removing it or retaining random size-matched components does not. Error bars indicate pointwise 95% bootstrap confidence intervals over base graphs; the gray band shows the 10th–90th percentiles across twenty random circuits.

To test whether the selected circuit preserves the causal recurrent mechanism in Bottleneck-latent, we repeat the latent-state transplantation. On separate 8-hop pairs with a connectivity swap at depth 4, we measure the increase in counterfactual-answer choices relative to each condition’s own baseline. The selected circuit retains a similar step-dependent transfer profile of the full model, with a peak increase of 63.9% versus 69.9%. In contrast, this effect is largely absent when the circuit is replaced by a random size-matched subset (Fig. 5D). In addition, the results remain stable across 7-12 hops (Fig. A.3). Together, our selected circuit preserves not only the model’s output, but also the causal state-transfer mechanism identified above.

## 3.3.3 A RECURRENT SEARCH ALGORITHM INSIDE THE SPARSE CIRCUIT

With the pruned circuit narrowing the recurrent computation to a 8 components, we next examine the specific role of each during recurrent computation. Among them, two components are particularly interesting. Attention head 1 in layer 4 (L4H1) separates how premise sources and destinations are transmitted. The corresponding MLP (L4MLP) helps propagate the retrieved information into subsequent recurrent states.

For a premise A is B., A and B are referred to as the left-hand side (LHS) and the right-hand side (RHS), respectively. Using the same constructions above, we create the same candidate-switching interventions by either swapping the LHS or the RHS at the same premise location (Fig. 6A). By replacing the actual cache with the swapped one, this matched-pair swap isolates whether the influence propagates through the RHS K-cache or the V-cache. We first characterize how attention reads graph premises. We find that, in L4H1, LHS swaps affect the answer mainly through keys, whereas RHS swaps affect mainly through values (Fig. 6B). To check if this division persists across different steps, we measure the similarity between the attention distribution before and after transplant using Jensen-Shannon distance, and find that these distributions are highly consistent (≈ 1, Fig. 6C). These results suggest an ’address–content’ organization in L4H1: keys determine which nodes are linked, while values supply the destination node information.

A  
![](images/2547064cd9ae513dd1fa515dc0b65f77ceed08b4d36ddb5ff5ddb057b21844ea.jpg)

![](images/33171c02bf3853e4ee57ebff25462196c4914236e61f80925478391c769c7903.jpg)

![](images/bd9f529ceeccc2ac5e3293b0c3999b5b8f26d671aad05f5be6c9303fa5dd9a39.jpg)

![](images/83343e9f67a83a808516d9c52d8eb5bd276d2efadd37e190751445a1f6e6a0ae.jpg)

![](images/78de39b1beb3059def872368798ec705ae333c7d28cb652a6d1586bcb8041781.jpg)

![](images/39c512a0a57317d768cb9db166a6d936d30dffb7fd3224e8f80de44b11953b2b.jpg)

![](images/9853cb9d3aa6f7b08d6a43242e87cf69b51696d787a7941b7dee145d809258f0.jpg)

![](images/10017a969634bc42c19060e3dc25471b14f945d654d6ed4f7c725984307e2541.jpg)  
Figure 6: Functional analysis of the recurrent circuit in Bottleneck-latent. A illustrates the LHS and RHS swaps. Interventions within the pruned circuit distinguish L4H1’s key/value routing (B), query-dependent selection of reading depth (C), and component contributions to the next query and final answer (D). Fixing or transplanting the L4 MLP response (E) separates its contribution from the residual pathway. $\check { \mathbf { F } }$ tests candidate matching through Q/K interventions in L4H1/H2/H4. Error bars indicate pointwise 95% bootstrap confidence intervals.

Then, we examine how MLP layers contribute to the computation. Instead of changing the structure of G, we swap the query root node r in matched chains used by Fig. 4, and measure how each component shifts the next query and the final answer in each transition (Fig. 6D). The L4MLP along with L4H1 show persistent influence on both, suggesting that they work together and change the query direction to further influence the next recurrent step. To isolate the contribution of L4MLP, we select the recurrent update $z _ { 2 }  z _ { 3 }$ , use the swapped L4H1 and measure how L4MLP influences the downstream targets from next query to final answers under different interventions. With L4H1’s output changed, we find recomputing L4MLP shifts all downstream targets to the swapped direction, compared with a frozen L4MLP (Fig. 6E). However, this effect is not additive (single L4MLP change barely shifts the direction) and relies on the residual stream (shifts exist even the MLP is fixed, though the magnitude is much lower). This shows that, L4MLP, they works by the whole residual stream, act as a “filter”, to select information for the next round’s operation.

We next ask how the evolving $z _ { t }$ becomes evidence for answer candidate $c ^ { * }$ , given an $^ { r } \cdot$ Candidate positions in x carry graph conditioned representations that recurrent attention can read (during reasoning phase). Using the sparse pruned circuit above, we keep the original question intact and test two types of changes with L4H1/H2/H4 during $z _ { t }  z _ { t + 1 }$ . In the first change case, we replace the query with the $\mathrm { Q }$ from same step’s update of a separate run starting from the other chain’s root. For example, if the original graph contains $r  A$ and $s  B$ , we replace $Q _ { r , t }$ with $Q _ { s , t }$ , while the original question still asks about r. In the second condition, we use candidate keys obtained from a graph with exchanged candidate endpoints: $r \sim B$ and $s  A$ , while keeping the question and its candidate positions unchanged. Either change alone reduce the $c ^ { * }$ logit margin, while applying both changes together can restore it. This suggests the candidate evidence depends on a match between the current recurrent state and the candidates’ graph context. Among the retained L4 Hs, H1 shows the strongest recurrent Q/K matching effects, whereas H2 shows the largest accuracy loss unde candidate value exchange and the greatest logit margin recovery at final readout phase.

Together, these results reveal how a recurrent search algorithm is implemented in the Bottlenecklatent: the $z _ { t }$ maintains currently reachable node in G; attention head L4H1 works as a soft tracer and uses premise LHSs to retrieve their RHSs to expand the reachable set; and L4 MLP, together with the residual pathway, then incorporates the retrived information back to $z _ { t + 1 } ,$ , guiding the next round of seaching. The candidate reading pathways in L4H1/H2/H4 also connect this evolving state to answer evidence. The Q/K matching controls the candidate information written into the latent z<sub>t</sub>, from which the final answer is decoded. These processes can operate in parallel, which allows the Bottleneck-latent to build a faster reachability search than strict step-by-step traversal as we have seen in 7 to 12 hop problems.

## 4 CONCLUSION

We ask whether models under different forms of thinking develop mechanistically distinct solutions, or converge on the same solutions through different ways. We train five GPT-like variants with the same backbone on an extended ProsQA task, and compare the mechanisms they induce.

Similar in-distribution performance hides the mechanistic divergence among the models. Direct, Chain-of-Thought, and Pause-token models solve in-distribution problems well, but rely on shortcuts related to local graph structure and generalize poorly to out-of-distribution problems. Notably, Chain-of-Thought, which is explicitly supervised with step-by-step proofs, fails to generalize, suggesting that training on reasoning traces does not guarantee that a model will reason in the same way. In contrast, the latent models (Full-latent and Bottleneck-latent), despite receiving no intermediate reasoning trace or reward signal, develop a recurrent circuit that implements “soft” forward search algorithm, expanding the reachable set across recurrent steps and generalizing to problems with longer reasoning depths. Digging deeper into the circuit in the bottleneck model, we find that an attention head retrieves graph relations through an address–content organization, while an MLP, together with the residual stream, integrates the retrieved information into the next state and a multi-head reading mechanism performs the candidate matching.

Together, these results show that different thinking interfaces lead to distinct underlying mechanisms, even at similar performance. The latent-reasoning models that allow information to flow fully across steps develop genuine reasoning computation that matches the structure of the task.

## 5 DISCUSSION

We want to emphasize the importance of testing model behavior on OOD problems before turning to internal mechanisms. As the Stroop’s color–word interference task reveals how humans process language and visual information, carefully designed OOD problems can also reveal how a model solves a task and guide where further mechanistic analysis should go (Friedman et al., 2024).

Building on this behavioral comparison, our design also isolates the effect of the thinking interface itself. Most mechanistic studies analyze a single model or a single form of reasoning in isolation. Instead, we train five variants that share the same backbone, dataset, and budget and differ only in their thinking interface, separating the mechanistic differences attributed to the interface from others. One family that is commonly used but not included here is the looped transformer, which applies the same weight-tied block for multiple iterations (Dehghani et al., 2019; Giannou et al., 2023). This explicit recurrence formalizes the iterative computation that we discover in the latent models, but whether the latent models generate the same forward search is still left to future work.

Finally, our conclusions come from small models on a controlled symbolic reasoning task, which simplifies analysis and lets us narrow the computation down to an interpretable circuit with confidence. However, the use of small models and a synthetic task limits how far the current conclusions can extend. Thus, whether our results and conclusions hold for larger pretrained LLMs and more naturalistic problems remains an open question for future work.

## REFERENCES

Darpan Aswal, Thomas Palmeira Ferraz, Yongxin Zhou, and Maxime Peyrard. Observable patterns are not explanations: A causal-geometric analysis of latent reasoning models, June 2026.

Jannik Brinkmann, Abhay Sheshadri, Victor Levoso, Paul Swoboda, and Christian Bartelt. A mechanistic analysis of a transformer trained on a symbolic multi-step reasoning task. In

Findings of the Association for Computational Linguistics: ACL 2024, pp. 4082–4102. Association for Computational Linguistics, 2024. doi: 10.18653/v1/2024.findings-acl.242. URL https://aclanthology.org/2024.findings-acl.242/.

Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Yunxuan Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, Albert Webson, Shixiang Shane Gu, Zhuyun Dai, Mirac Suzgun, Xinyun Chen, Aakanksha Chowdhery, Alex Castro-Ros, Marie Pellat, Kevin Robinson, Dasha Valter, Sharan Narang, Gaurav Mishra, Adams Yu, Vincent Zhao, Yanping Huang, Andrew Dai, Hongkun Yu, Slav Petrov, Ed H. Chi, Jeff Dean, Jacob Devlin, Adam Roberts, Denny Zhou, Quoc V. Le, and Jason Wei. Scaling instruction finetuned language models. Journal of Machine Learning Research, 25(70):1–53, 2024. URL https://jmlr.org/papers/v25/23-0870.html.

Mostafa Dehghani, Stephan Gouws, Oriol Vinyals, Jakob Uszkoreit, and Łukasz Kaiser. Universal transformers. In International Conference on Learning Representations, 2019. URL https: //openreview.net/forum?id=HyzdRiR9Y7.

Dan Friedman, Andrew Kyle Lampinen, Lucas Dixon, Danqi Chen, and Asma Ghandeharioun. Interpretability illusions in the generalization of simplified models. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research. PMLR, 2024. URL https://proceedings.mlr.press/v235/ friedman24a.html.

Angeliki Giannou, Shashank Rajput, Jy-Yong Sohn, Kangwook Lee, Jason D. Lee, and Dimitris Papailiopoulos. Looped transformers as programmable computers. In Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pp. 11398–11442. PMLR, 2023. URL https://proceedings.mlr.press/ v202/giannou23a.html.

Sachin Goyal, Ziwei Ji, Ankit Singh Rawat, Aditya Krishna Menon, Sanjiv Kumar, and Vaishnavh Nagarajan. Think before you speak: Training language models with pause tokens, 2024.

Shibo Hao, Sainbayar Sukhbaatar, DiJia Su, Xian Li, Zhiting Hu, Jason Weston, and Yuandong Tian. Training Large Language Models to Reason in a Continuous Latent Space, November 2025.

Namgyu Ho, Laura Schmid, and Se-Young Yun. Large language models are reasoning teachers. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 14852–14882. Association for Computational Linguistics, 2023. doi: 10.18653/v1/2023.acl-long.830. URL https://aclanthology.org/2023. acl-long.830/.

Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. Large language models are zero-shot reasoners, 2022.

Tamera Lanham, Anna Chen, Ansh Radhakrishnan, Benoit Steiner, Carson Denison, Danny Hernandez, Dustin Li, Esin Durmus, Evan Hubinger, Jackson Kernion, Kamile Luko ˙ siˇ ut¯ e, Karina ˙ Nguyen, Newton Cheng, Nicholas Joseph, Nicholas Schiefer, Oliver Rausch, Robin Larson, Sam McCandlish, Sandipan Kundu, Saurav Kadavath, Shannon Yang, Thomas Henighan, Timothy Maxwell, Timothy Telleen-Lawton, Tristan Hume, Zac Hatfield-Dodds, Jared Kaplan, Jan Brauner, Samuel R. Bowman, and Ethan Perez. Measuring faithfulness in chain-of-thought reasoning, July 2023.

Lucie Charlotte Magister, Jonathan Mallinson, Jakub Adamek, Eric Malmi, and Aliaksei Severyn. Teaching small language models to reason, 2023. URL https://arxiv.org/abs/2212. 08410.

Jacob Pfau, William Merrill, and Samuel R. Bowman. Let’s think dot by dot: Hidden computation in transformer language models, 2024.

Michael Rizvi-Martel, Guillaume Rabusseau, and Marius Mosbach. The illusion of superposition? a principled analysis of latent thinking in language models, 2026.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc $\mathrm { L e } ,$ and Denny Zhou. Chain-of-thought prompting elicits reasoning in large language models. Advances in Neural Information Processing Systems, 35:24824–24837, 2022.

Xilin Wei, Xiaoran Liu, Yuhang Zang, Xiaoyi Dong, Yuhang Cao, Jiaqi Wang, Xipeng Qiu, and Dahua Lin. SIM-CoT: Supervised Implicit Chain-of-Thought, 2025.

Yiwei Wu, Atticus Geiger, and Raphael Milli ¨ ere. How do transformers learn variable binding in\` symbolic programs? In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pp. 67284–67299. PMLR, 2025. URL https://proceedings.mlr.press/v267/wu25j.html.

Jiachen Zhao, Yiyou Sun, Weiyan Shi, and Dawn Song. Can aha moments be fake? towards quantifying decorative and true thinking in chain-of-thought, 2026.

Hanlin Zhu, Shibo Hao, Zhiting Hu, Jiantao Jiao, Stuart Russell, and Yuandong Tian. Reasoning by superposition: A theoretical perspective on chain of continuous thought, 2025.

## A APPENDIX

## A.1 DATASET GENERATION ALGORITHM

We construct ProsQA-Ext from the graph-reachability task in ProsQA (Hao et al., 2025). In ProsQA, each sample consists of a series of premises describe a DAG and a question asks which of the two candaites is reachable from the specified root node. In ProsQA-Ext, we preserved this logic, while we changed the how each sample is constructed. Labels are assigned after graph construction, premises are shuffled, and the correct candidate appears equally often on either side of the question, so that no obvious superficial statistics are related to the answer.

For the extended 400k 3–6 hop training set, we sample new graphs with an empirical quota that keeps the empirical joint distribution of graph size, proof length, queried root, and binned number of shortest paths similar to ProsQA. For ID validation and test, we retain the released graphs and queries, but reassign node labels and randomize premise and candidate order.

For OOD 7–12 hop dataset for evaluation, we use algorithm 1 to generate it:

Algorithm 1 Controlled long-hop graph generation   
Require: Proof length $\overline { { H \in \{ 7 , \dots , 1 2 \} } }$ ; balanced candidate-side assignment b   
1: Construct two vertex-disjoint directed paths $P _ { 1 } , P _ { 2 } ,$ each of length $\dot { \boldsymbol { H } }$   
2: for $j \in \{ 1 , 2 \}$ do   
3: Add ${ \dot { 1 } } 3 - H$ off-path nodes $U _ { j }$   
4: Attach the first node of $U _ { j }$ to the root of $P _ { j }$   
5: Attach each remaining node of $U _ { j }$ to an internal node of $P _ { j }$ or an earlier node of $U _ { j }$   
6: end for   
7: Sample additional edges into off-path nodes, without duplicates or cycles, until $| E | = 3 8$   
8: Choose one of $P _ { 1 } , P _ { 2 } ^ { \bar { 2 } }$ as the queried component   
9: Set r to its root, $c ^ { + }$ to its endpoint, and $c ^ { - }$ to the endpoint of the other path   
10: Randomly assign node labels and permute the premise order   
11: Place $c ^ { + }$ on side b of the binary question

## A.2 TRAINING DETAILS

All five variants are trained from random initialization on the same 400k ProsQA-Ext problems with proof lengths of 3–6 hops, separately on two Nvidia 5090 and 4090 machines. Each model uses a four-layer GPTNeoX backbone with hidden size 256, four attention heads, feed-forward dimension 768, untied input and output embeddings, and attention and hidden dropout of 0.1. We use AdamW with a constant learning rate of $4 \times \mathrm { { \bar { 1 0 } ^ { - 4 } , ( \beta _ { 1 } , \beta _ { 2 } ) = ( 0 . 9 5 , 0 . 9 9 9 ) } }$ , weight decay of $1 0 ^ { - 4 }$ and a batch size of 256. Training used BF16 without learning-rate warmup or decay. The objective averages next-token cross-entropy over prompt and answer tokens with equal per-token weights; Text CoT additionally supervised the shortest-proof tokens. Pause and latent variants used $K = 6$ thinking positions. Latent models are optimized end-to-end through the full recurrence without discrete intermediate targets, curriculum training or RL.

## A.3 OBSERVED CORRELATION BETWEEN MODELS’ INTERNAL COMPUTATION AND THE ALGORITHM INTERMEDIATE VARIABLES

We compare model representations with propagated frontiers under forward search and successive ancestor sets of the candidates under backward tracing. For a graph $G = ( V , E )$ with query root r, the forward frontier is initialized at the root and updated by following outgoing edges:

$$
F _ { 0 } = \{ r \} , \qquad F _ { d + 1 } = \{ v \in V : \exists u \in F _ { d } , ( u , v ) \in E \} .
$$

Each update expands all nodes in the current frontier, allowing multiple branches to be followed in parallel. Thus, $F _ { d }$ contains nodes reachable from r by a directed path of exactly d edges. It differs from the cumulative reachable set $\textstyle \bigcup _ { j = 0 } ^ { d } F _ { j }$ , which retains nodes reached at earlier depths. A node can appear in multiple frontiers if paths of different lengths lead to it. For example, edges $r  a .$ $r  b ,$ and $a \to b$ give $F _ { 1 } = \{ a , b \}$ and $F _ { 2 } = \left\{ b \right\}$

For backward tracing, we start from both candidates simultaneously and propagate their joint frontier along incoming edges:

$$
U _ { 0 } = \{ c _ { 0 } , c _ { 1 } \} , \qquad U _ { d + 1 } = \{ u \in V : \exists v \in U _ { d } , ( u , v ) \in E \} .
$$

Here, $U _ { d }$ is the union of both candidates’ frontiers after exactly d reverse steps. For RSA, forward and backward frontiers are encoded as binary vectors over node labels, indicating membership in the set. The backward representation is invariant to the order of candidates in the query.

The analysis covers the final prompt position and the thinking phase, where applicable, on both ID and OOD problems. For latent models, we use the continuous states $z _ { t } ,$ , with $z _ { 1 }$ defined as the normalized last-layer state at the final prompt position. For Chain-of-Thought, we use the first 21 generated proof tokens, matching the shortest generated proof among the samples. For Pause-token, we analyze the normalized last-layer residual states $N _ { \theta } ( h _ { n + t } ^ { ( L ) } )$ ), since its input vectors $z _ { t }$ are identical across steps. Direct serves as a control without additional thinking steps, using normalized last-layer states at the final prompt position and during answer readout. Figure A.1 shows forward-propagation alignment across all evaluated depths.

![](images/e9fcd7eb84ed3c1f8a35d75f196f05cf8976ed9c380553e09f78e9403478ed37.jpg)  
Figure A.1: Representational alignment with forward graph propagation across task depths. RSA heatmaps for five model variants on 3 to 12-hop problems (rows), across ID (3–6 hops) and OOD (7–12 hops) conditions. Entries represent Spearman correlations between modelrepresentation and frontier dissimilarities, as shown in Fig. 2.

## A.4 DIAGONALITY

We develop Diagonality to quantify whether stronger alignment shifts toward later computation positions as algorithm depth increases. Let $R _ { d } ( t )$ denote the RSA value at depth d and position t. First, we convert each row to normalized ranks, $r _ { d } ( t ) = ( \mathrm { r a n k } ( R _ { d } ( t ) ) - 1 ) / ( n _ { d } - 1 )$ , where $n _ { d }$ counts finite entries. Undefined entries and rows with fewer than two finite entries are set to zero.

We the compute the best fixed position, independently selected row maxima, and the best nondecreasing path:

$$
S _ { \mathrm { s t a t i c } } = \operatorname* { m a x } _ { t } \sum _ { d } r _ { d } ( t ) , \qquad S _ { \mathrm { t o p } } = \sum _ { d } \operatorname* { m a x } _ { t } r _ { d } ( t ) , \qquad S _ { \mathrm { f w d } } = \operatorname* { m a x } _ { t _ { 0 } \leq \cdots \leq t _ { D } } \sum _ { d = 0 } ^ { D } r _ { d } ( t _ { d } ) .\tag{2}
$$

We define diagonality as

$$
\mathrm { D i a g o n a l i t y } = { \frac { S _ { \mathrm { f w d } } - S _ { \mathrm { s t a t i c } } } { S _ { \mathrm { t o p } } - S _ { \mathrm { s t a t i c } } } } .\tag{3}
$$

The score lies in $[ 0 , 1 ] \colon$ when the denominator is positive, one means that a nondecreasing path reaches every row’s maximum (perfect propagation), while zero means that allowing forward propagation gives no advantage over a fixed position. We assign zero when the denominator vanishes.

Note that Diagonality measures the ordering of alignment, not its absolute strength. It allows pauses and jumps between positions, without requiring one model step per algorithm update.

Across numer of hops, no variant shows a clear, consistent pattern of alignment with backward search (Fig. A.2).

![](images/a25f34030abc1c0cb4e72b0d7c13a8711daf146f308eb30e9b7eabd4b017cd14.jpg)  
Figure A.2: Diagonality of alignment with parallel backward search. The same diagonality measure is applied to RSA against the joint backward frontier for all five model variants. The dashed line separates ID from OOD problems.

## A.5 EFFECT OF HOP PERTURBATION ON PRUNED CIRCUIT IN BOTTLENECK-LATENT MODEL

Figure A.3 shows the effect of hop perturbation on state transfer in the pruned circuit across 7–12- hop problems.

![](images/255f5b5db7d1bedfbe0a0571894b711fed469cccb453d19e8201a2a142b9d44d.jpg)  
Figure A.3: Effect of hop perturbation on pruned circuit in Bottleneck-latent model.

## A.6 RECURRENT CIRCUIT PRUNING

We apply greedy mean-replacement pruning (Algorithm 2) to the 20 recurrent components of Bottleneck-latent (16 attention heads and four MLPs). Each component is pruned across all five recurrent transitions from $z _ { 1 } ~ \mathrm { t o } ~ z _ { 6 }$ . Replacement means are computed separately for each component and transition from both members of counterfactual pairs constructed from 512 independent 8-hop graphs. These means remain fixed throughout pruning. Input encoding and answer readout remain the same.

For a retained component set S, let $m _ { i } ( S )$ denote the candidate logit margin after mean-replacing all components outside $S ,$ and let $m _ { i }$ denote the full-model margin. We measure candidate-choice consistency $C ( S )$ and margin fidelity $R ^ { 2 } ( S )$ on the examples:

$$
C ( S ) = \frac { 1 } { N } \sum _ { i } { \bf 1 } [ { \bf 1 } [ m _ { i } ( S ) \geq 0 ] = { \bf 1 } [ m _ { i } \geq 0 ] ] , \qquad R ^ { 2 } ( S ) = 1 - \frac { \sum _ { i } ( m _ { i } ( S ) - m _ { i } ) ^ { 2 } } { \sum _ { i } ( m _ { i } - \bar { m } ) ^ { 2 } } .
$$

Algorithm 2 Greedy recurrent circuit pruning   
Require: Full component set C; fixed replacement means   
1: $\bar { S }  \mathcal { C }$   
2: while $S \neq \emptyset$ do   
3: for each $c \in S$ do   
4: Evaluate $C _ { c } = C ( S \setminus \{ c \} )$ and $R _ { c } ^ { 2 } = R ^ { 2 } ( S \setminus \{ c \} )$   
5: end for   
6: $\mathcal { F }  \{ c \in S : C _ { c } \geq 0 . 9 0 , \ R _ { c } ^ { 2 } \geq 0 . 8 0 \}$   
7: if $\mathcal { F } = \emptyset$ then   
8: break   
9: end if   
10: $c ^ { \star }  \arg \operatorname* { m i n } _ { c \in \mathcal { F } } \frac { 1 } { 2 } ( \frac { 1 - C _ { c } } { 0 . 1 0 } + \frac { 1 - R _ { c } ^ { 2 } } { 0 . 2 0 } )$   
11: $S \gets S \setminus \{ c ^ { \star } \}$   
12: end while   
13: return S

## A.7 HEAD CONTRIBUTIONS DURING RECURRENCE AND ANSWER READOUT

We compare L4’s H1, H2, and H4 using the same Bottleneck-latent recurrent circuit (Fig 5).

During recurrence, we apply the Q/K interventions in Fig. 6F to one head at a time across all five updates. We measure the decrease in the correct − incorrect candidate logit margin. We also test candidate values separately: we reverse the candidate order in a separate run and use its candidate position values in the original run, keeping the attention weights unchanged at each update. We measure the resulting accuracy loss in percentage points.

For final answer readout, we use graph pairs with opposite correct answers but the same task root, candidate positions, and answer prefix. We replace one head’s output at the answer prediction position with its output from the opposite-answer run. The original latent trajectory stays fixed. 43 of 96 graph pairs for which both answers are initially predicted correctly are used for analysis. Here, the margin is the right candidate’s logit - the remaining candidate’s logit. Recovery measures the shift toward the opposite answer margin: 0% means no change, and 100% means reaching that margin.

![](images/23ca8d9bd665a6a758a8f12fc2194a524e75f7b41d3783210220e1c9d6f764d2.jpg)

![](images/1702937b30815142dd109f85cef9dd3ff2924379d75422f39b994cf19868296d.jpg)

![](images/e68f71db00ea68726a9db83c4a8c964a97faf2f0707daac34e0a8a7054f0ef68.jpg)  
Figure A.4: Head contributions during recurrence and answer readout. Individual interventions compare recurrent Q/K matching (left), candidate-value exchange (middle), and final answer readout (right) for L4H1, L4H2, and L4H4. Error bars indicate pointwise 95% bootstrap confidence intervals.