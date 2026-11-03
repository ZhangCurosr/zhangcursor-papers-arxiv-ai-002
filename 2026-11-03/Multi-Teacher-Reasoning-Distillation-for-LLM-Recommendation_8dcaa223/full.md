# Multi-Teacher Reasoning Distillation for LLM Recommendation with Self-Exploration

Liangbo Ning<sup>1</sup>, Siyuan Liang<sup>2</sup>, Yi Zhou<sup>1</sup>, Yifeng Zhang<sup>1</sup>, Qing Li<sup>1</sup>, Wenqi Fan<sup>1</sup>, Dacheng Tao<sup>2</sup> <sup>1</sup> The Hong Kong Polytechnic University, Hong Kong SAR, China

<sup>2</sup> Nanyang Technological University, Singapore

{BigLemon1123,wenqifan03}@gmail.com, qing-prof.li@polyu.edu.hk, {echo-yi.zhou,yifeng.zhang}@connect.polyu.hk, {siyuan.liang,dacheng.tao}@ntu.edu.sg

## Abstract

Reasoning-augmented recommender systems (RecSys) enhance user preference modeling by generating reasoning trajectories over historical interactions before making recommendations. Knowledge distillation provides a practical paradigm for transferring such reasoning capabilities from powerful large language models to lightweight RecSys. However, existing methods typically rely on a single fixed teacher, remain constrained by the teacher knowledge ceiling, and indiscriminately distill teacher-generated trajectories without assessing their actual utility for downstream recommendation. To address these limitations, we propose MUSE (Multi-Teacher Utility-guided Selection with Self-Exploration), an adaptive reasoning distillation framework that jointly improves reasoning generation and reasoning utilization. MUSE first constructs a diverse reasoning pool using multiple teacher LLMs and progressively augments it via student self-exploration, enabling the student to leverage the strengths of different teachers and gradually explore reasoning strategies beyond teacher imitation. We then develop Rank-Gain Utility Assessment (RGUA), which evaluates each candidate according to the normalized full-catalog rank improvement of the ground-truth item and identifies the most and least recommendation-beneficial trajectories. Based on these utility-aware trajectory pairs, we introduce dual-level contrastive learning objectives that guide the student generator to prefer beneficial reasoning while training the recommendation encoder to more effectively exploit recommendation-relevant preference evidence. Extensive experiments on three real-world datasets demonstrate the effectiveness and robustness of the proposed framework.

## 1 Introduction

Recommender systems (RecSys) have become a core component of Web platforms, helping users navigate information overload and discover relevant content, products, and services [60, 1, 16, 5, 8, 9, 54, 61, 37, 50]. With the rapid advancement of large language models (LLMs) [43, 6, 28, 30, 18, 32], the recommendation paradigm has been fundamentally transformed, opening up a new era of LLMempowered recommendation [60, 35, 33, 34]. By representing users and items in natural language, LLM-empowered RecSys can leverage rich semantic knowledge and strong instruction-following capabilities to understand item descriptions and interpret user preferences, thereby delivering more accurate recommendations. More recently, the remarkable progress of reasoning-empowered LLMs (e.g., DeepSeek [42, 45]) has motivated a further transition from language-based recommendation to reasoning-augmented recommendation. Rather than directly predicting the next item from an interaction sequence, a reasoning-augmented recommender system first generates an explicit or latent reasoning trajectory that analyzes the user’s historical behaviors and infers the user’s underlying preferences. This trajectory is then used to produce more accurate recommendations. [55, 20, 22, 29]. Recent methods, such as R<sup>2</sup>ec [55] and ReRec [20], have preliminarily explored this direction by encouraging LLM-based recommenders to reason over user histories before making predictions. Such preference reasoning provides a promising mechanism for capturing nuanced user interests while offering explicit evidence to support the generated recommendations.

![](images/76947620e012e620069b1d4fac18e85715a4fd1d6f9119d0d6ff4575d9d35f7a.jpg)  
Figure 1: Illustration of the limitations of existing reasoning distillation methods and the proposed MUSE. Existing reasoning distillation methods rely on a single fixed general-purpose teacher, leading to heterogeneous teacher capability, teacher knowledge ceiling, and poor-quality reasoning supervision. MUSE instead adaptively selects recommendation-beneficial reasoning from multiple teachers and progressively incorporates superior student-generated reasoning through self-exploration, enabling more effective reasoning distillation for recommendation.

Technologically speaking, existing approaches for acquiring reasoning capabilities for recommendations can be broadly divided into two paradigms: reinforcement learning (RL) and knowledge distillation. Reinforcement learning-based methods encourage a recommender to autonomously explore reasoning trajectories and optimize them according to carefully designed recommendation rewards [20, 55, 14, 58, 47, 10]. In principle, this trial-and-error process allows the model to discover reasoning strategies that are directly aligned with downstream ranking quality, instead of merely imitating human-written rationales. However, directly training a relatively small language model to acquire reliable reasoning capabilities remains difficult [15, 59]. Owing to its limited model capacity, a small language model often struggles to generate sufficiently informative reasoning trajectories during the early stages of exploration, resulting in sparse or noisy reward signals, unstable optimization, and substantial computational overhead. Knowledge distillation-based methods employ a powerful teacher LLM to generate preference reasoning trajectories for each user, which are subsequently used as supervision to train a smaller reasoning-augmented recommender system through standard supervised fine-tuning [53, 51, 52, 24, 39]. Benefiting from the accurate supervision provided by the teacher model, knowledge distillation avoids relying solely on the student’s initially weak exploration capability and typically yields a more stable optimization process than reinforcement learning from scratch. It is also complementary to reinforcement learning: a distilled model can serve as a strong initialization for subsequent RL-based post-training [56, 19, 29, 7]. Therefore, the favorable optimization stability and compatibility with further post-training collectively make knowledge distillation a practical paradigm for equipping lightweight recommender systems with explicit reasoning capabilities.

Despite this potential, as illustrated in Figure 1, existing knowledge distillation-based methods still suffer from several limitations. (1) Heterogeneous Teacher Capability. Existing studies typically rely on a single fixed teacher to generate reasoning trajectories for subsequent distillation. However, due to differences in model architectures, pre-training corpora, and optimization objectives, LLMs often exhibit substantially different reasoning capabilities for recommendation tasks. A teacher that provides informative reasoning for one user, instance, or domain may generate inferior supervision for another. Consequently, reasoning trajectories produced by a single fixed teacher are unlikely to provide optimal supervision across diverse recommendation scenarios. (2) Teacher Knowledge Ceiling. General-purpose LLMs are primarily optimized for broad language understanding and generation rather than recommendation. As a result, they often lack sufficient recommendationspecific knowledge for accurately identifying users’ latent preferences. Nevertheless, existing methods typically rely exclusively on reasoning trajectories generated by a fixed teacher throughout the entire distillation process. Consequently, the student model is fundamentally limited by the recommendation capability of its teacher, as it merely learns to imitate the teacher’s reasoning behavior. (3) Poor-quality Reasoning Supervision. Existing methods usually treat all teacher-generated reasoning trajectories as equally reliable supervision without assessing their actual utility for downstream recommendation. A reasoning trajectory can be fluent, coherent, and seemingly persuasive while contributing little to recommendation performance, as it may emphasize preference signals that are irrelevant to retrieving the ground-truth item. Consequently, indiscriminately distilling such reasoning trajectories may fail to equip the downstream reasoning-augmented RecSys with reasoning capabilities that genuinely improve recommendation performance.

To address these limitations and enable recommenders to acquire more effective reasoning knowledge, we propose MUSE (Multi-Teacher Utility-guided Selection with Self-Exploration), an adaptive generation-then-selection framework that identifies and distills the reasoning trajectories most beneficial for recommendation. Specifically, MUSE first introduces multiple teacher LLMs to construct a diverse reasoning pool, allowing different teachers to provide supervision for different users and thereby alleviating the issue of heterogeneous teacher capability. To overcome the knowledge ceiling of general-purpose teachers, MUSE further incorporates a progressive self-exploration strategy, in which the student progressively explores alternative reasoning trajectories as its recommendation capability improves. These self-generated trajectories are also included in the reasoning pool and evaluated together with teacher-generated candidates, enabling the student to gradually evolve beyond merely imitating its teachers. Then, we develop a Rank-Gain Utility Assessment (RGUA) strategy that evaluates each candidate according to its contribution to downstream recommendation performance, enabling MUSE to select recommendation-beneficial reasoning while filtering out uninformative / poor-quality supervision. Finally, MUSE introduces the dual-level contrastive learning objectives that encourage the RecSys to both accurately generate and effectively leverage recommendation-beneficial reasoning, leading to more accurate recommendations. The main contributions of this work are summarized as follows:

• We propose MUSE, a multi-teacher reasoning distillation framework with progressive selfexploration, which alleviates heterogeneous teacher capability by using diverse teacher supervision and enables the student to move beyond the knowledge ceiling of general-purpose LLMs through progressive exploration.

• We develop Rank-Gain Utility Assessment (RGUA), which adaptively identifies recommendationbeneficial reasoning trajectories according to their downstream recommendation utility for knowledge distillation. Furthermore, we introduce a dual-level contrastive learning objective that jointly improves reasoning generation and reasoning utilization, enabling the recommender to generate recommendation-beneficial reasoning and more effectively exploit it for accurate recommendations.

• We conduct extensive experiments on three real-world recommendation datasets. The results demonstrate the effectiveness of the proposed framework.

## 2 Preliminaries

Reasoning-Augmented Recommendation. Let U and V denote the user set and the complete item catalog, respectively. We consider a next-item recommendation dataset: $\mathcal { D } = \{ ( u _ { i } , \mathcal { \bar { H } } _ { i } , y _ { i } ) \} _ { i = 1 } ^ { N }$ where $u _ { i } ~ \in { \mathrm { ~ } } \mathcal { U }$ is a user, ${ \mathcal { H } } _ { i } = [ ( v _ { i , 1 } , a _ { i , 1 } ) , \dotsc , ( v _ { i , m _ { i } } , a _ { i , m _ { i } } ) ]$ is the user’s interaction history, $v _ { i , j } \in \mathcal V$ and $a _ { i , j }$ denote the interacted item and its textual attributes $( \mathrm { e . g . }$ ., item titles, descriptions), and $y _ { i } \in \mathcal V$ is the ground-truth next item with attributes $a _ { y _ { i } }$ . Generally, the user’s interaction history $\mathcal { H } _ { i }$ is verbalized into a recommendation prompt $x _ { i }$ , which is then fed into the recommender system to generate the recommendation results. A reasoning-augmented RecSys usually consists of a reasoning trajectory generator $\mathcal { G } _ { \theta }$ and a recommendation encoder $\mathcal { E } _ { \phi }$ [55, 20, 58, 14, 53, 31]. Conditioned on $x _ { i } ,$ the generator first produces a textual / latent reasoning trajectory that summarizes the user’s preferences:

$$
r _ { i } = { \mathcal { G } } _ { \theta } ( x _ { i } ) .
$$

![](images/b597e98a97538a011822b422b7506ae07948d0a612befb53600995866e4bab99.jpg)  
Figure 2: Overall architecture of MUSE. (1) Reasoning Pool Generation constructs diverse reasoning candidates from multiple teachers and progressive student self-exploration; (2) Rank-Gain Utility Assessment identifies the most and least recommendation-beneficial reasoning trajectories according to their downstream ranking improvement; and (3) Utility-Aware Contrastive Training optimizes the reasoning generator and recommendation encoder using the dual-level contrastive learning objectives.

The recommendation encoder then maps the reasoning trajectory to a user representation and the textual attribute $a _ { j }$ of each item to the item representation, respectively:

$$
\begin{array} { r } { \mathbf { z } _ { i } ( r _ { i } ) = \mathcal { E } _ { \phi } ( [ x _ { i } ; \bar { r } _ { i } ] ) , \quad \mathbf { q } _ { j } = \mathcal { E } _ { \phi } ( [ v _ { j } ; a _ { j } ] ) . } \end{array}
$$

The relevance of item $v _ { j }$ to user $u _ { i }$ is measured by representation similarity (e.g., cosine similarity): $s _ { \phi } ( u _ { i } , v _ { j } | \boldsymbol { r } _ { i } ) = \sin ( \mathbf { z } _ { i } ( \boldsymbol { r } _ { i } ) , \mathbf { q } _ { j } )$

Items are sorted in descending order of $s _ { \phi } ( u _ { i } , v _ { j } | r _ { i } )$ , and the highest-ranked items are returned to the user as the final recommendation outcomes. It is worth noting that the recommendation module ${ \mathcal { E } } _ { \phi }$ can take different forms depending on the underlying architecture. In some reasoning-augmented RecSys, the reasoning generator itself is directly employed as the recommendation module [55, 20, 58], whereas other approaches adopt an additional lightweight encoder to encode the reasoning trajectories and candidate items for recommendation [14, 53, 24].

Reasoning Distillation. Recently, with the rapid advancement of LLM reasoning capabilities, increasing research attention has been devoted to transferring such reasoning abilities to recommender systems to enhance user preference modeling. Typically, reasoning distillation employs a powerful teacher LLM T to generate a trajectory for each training instance: $r _ { i } ^ { \mathcal { T } } = \mathcal { T } ( x _ { i } ) ^ { \ast }$ . Let $\boldsymbol { r } _ { i } ^ { \hat { T } } = [ r _ { i , 1 } ^ { \mathcal { T } } , \dots , r _ { i , L _ { i } } ^ { \mathcal { T } } ]$ denote the teacher-generated reasoning trajectory for user $u _ { i } .$ , where $L _ { i }$ is its token length. The reasoning trajectory generator of the student reasoning-augmented RecSys is optimized using the standard autoregressive language modeling objective through teacher-forced supervised fine-tuning (SFT):

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { S F T } } = - \sum _ { i = 1 } ^ { N } \sum _ { t = 1 } ^ { L _ { i } } \log \mathcal { G } _ { \theta } \left( r _ { i , t } ^ { \mathcal { T } } \mid x _ { i } , r _ { i , < t } ^ { \mathcal { T } } \right) , } \end{array}
$$

where $r _ { i , < t } ^ { \mathcal { T } }$ denotes the preceding reasoning tokens. After that, the student model learns to imitate the teacher-generated reasoning trajectories and produce preference reasoning conditioned on the user’s historical interactions.

## 3 Methodology

Overview. To effectively distill reasoning knowledge from LLMs while overcoming the limitations of existing methods caused by relying on static general-purpose teachers, we propose Multi-Teacher Utility-guided Selection with Self-Exploration (MUSE). As illustrated in Figure 2, MUSE consists of three stages: Reasoning Pool Generation, Rank-Gain Utility Assessment, and Utility-Aware Contrastive Training. In the Reasoning Pool Generation stage, we introduce multiple teacher LLMs to generate diverse preference reasoning trajectories for each user, thereby avoiding reliance on a globally fixed teacher. We further propose a Progressive Self-Exploration strategy, in which the student progressively explores alternative reasoning trajectories as its capability improves and augments the candidate pool with these self-generated trajectories. In the Rank-Gain Utility Assessment stage, each candidate reasoning trajectory is evaluated according to the improvement it brings to the full-catalog ranking of the ground-truth item relative to the reference recommendation without reasoning, allowing the most beneficial reasoning trajectory to be selected for supervised fine-tuning. Finally, during the Utility-Aware Contrastive Training stage, to more effectively exploit both highquality and low-quality reasoning trajectories, we design the dual-level contrastive learning objectives (i.e., Contrastive Trajectory Distillation and Contrastive Reasoning Utilization). The preferred and dispreferred reasoning provide comparative supervision for both the recommendation encoder and the student reasoning generator, encouraging the former to distinguish informative from misleading reasoning signals and the latter to generate recommendation-beneficial reasoning.

## 3.1 Multi-Teacher Distillation with Self-Exploration

## 3.1.1 Multi-teacher reasoning pool construction.

Different LLMs are trained with diverse architectures, optimization strategies, and pre-training corpora, resulting in substantially different reasoning capabilities for recommendation tasks. Arbitrarily selecting a single teacher largely overlooks whether it is genuinely well-suited for recommendationspecific reasoning, potentially limiting the effectiveness of the distilled student RecSys. To address this issue, we propose a multi-teacher distillation framework that leverages multiple teacher LLMs to generate diverse reasoning rationales for each user. By selecting the most appropriate reasoning trajectory for each user, the student recommender system can adaptively leverage the strengths of different teacher models, rather than being constrained by the capability of an arbitrarily selected single teacher. Mathematically, we employ M teacher LLMs, denoted by $\{ \mathcal { T } _ { m } \} _ { m = 1 } ^ { M }$ , and prompt each teacher with the same recommendation query to generate a preference reasoning trajectory for each user:

$$
r _ { i } ^ { ( m ) } = \mathcal { T } _ { m } ( x _ { i } ) , \qquad m = 1 , \ldots , M .\tag{1}
$$

The initial teacher candidate set for user $u _ { i }$ is:

$$
\mathcal { C } _ { i } ^ { \mathrm { t e a } } = \{ r _ { i } ^ { ( 1 ) } , \ldots , r _ { i } ^ { ( M ) } \} .\tag{2}
$$

## 3.1.2 Progressive student self-exploration.

Although we use multiple teachers, these teacher LLMs are primarily pretrained to acquire generalpurpose language understanding and generation capabilities, and therefore inevitably suffer from limited recommendation-specific domain knowledge. Consequently, even the best teacher-generated reasoning trajectories may not always be optimal for recommendation tasks. To overcome this limitation, we further introduce a self-exploration strategy. As training progresses and the reasoning capability of the student reasoning-augmented RecSys continuously improves, the student is encouraged not only to learn from the reasoning knowledge distilled from multiple teachers but also to autonomously explore alternative reasoning trajectories that may be more effective for recommendations. The self-generated reasoning trajectories are subsequently evaluated together with those produced by the teacher models, enabling the student to progressively discover reasoning strategies that are better aligned with recommendation objectives. In this way, the student evolves from merely imitating the teacher’s reasoning to continuously improving through self-exploration, thereby allevi ating the inherent limitation of teacher models with insufficient recommendation-specific knowledge. Mathematically, let $\mathcal { G } _ { \boldsymbol { \theta } _ { \sf { \epsilon } } }$ denote the current student model at epoch e. We then sample $S _ { e }$ reasoning trajectories based on its current parameters, defined as: $\tilde { r } _ { i , e } ^ { ( j ) } \sim \mathcal { G } _ { \theta _ { e } } ( r | x _ { i } ) , j = 1 , \dots , S _ { e }$ , where $S _ { e }$ denotes the sampling number. As the student model becomes increasingly capable during training, we progressively increase the number of reasoning trajectories sampled from the student to encourage more extensive self-exploration, formulated as:

$$
S _ { e } = \operatorname* { m i n } \left( S _ { \mathrm { m a x } } , [ S _ { \mathrm { m a x } } \cdot \frac { [ e - e _ { 0 } ] _ { + } } { E _ { \mathrm { s } } - e _ { 0 } } ] \right) ,\tag{3}
$$

where $[ e - e _ { 0 } ] _ { + } = \operatorname* { m a x } ( 0 , e - e _ { 0 } ) , e _ { 0 } = 2$ is the warm-up epoch, $S _ { \mathrm { m a x } }$ is the maximum number of student reasoning candidates, and $E _ { \mathrm { s } }$ denotes the epoch at which the number of sampled trajectories reaches its maximum value $S _ { \mathrm { m a x } }$ . Finally, the complete candidate reasoning pool $\mathcal { C } _ { i } ^ { ( e ) }$ becomes:

$$
\mathcal { C } _ { i } ^ { ( e ) } = \mathcal { C } _ { i } ^ { \mathrm { t e a } } \cup \{ \tilde { r } _ { i , e } ^ { ( 1 ) } , \dots , \tilde { r } _ { i , e } ^ { ( S _ { e } ) } \} .\tag{4}
$$

## 3.2 Rank-Gain Utility Assessment (RGUA)

After constructing a diverse candidate pool through multiple teachers and self-exploration, the remaining challenge is to identify high-quality reasoning trajectories without human annotations. We argue that useful reasoning should consistently improve recommendation quality by helping the model better identify the user’s true preference. Therefore, instead of evaluating reasoning in isolation, we can directly measure its contribution to the downstream recommendation objective through the ranking improvement of the ground-truth item. Based on this intuition, we propose a Rank-Gain Utility Assessment strategy. Specifically, for each training instance, we first encode the recommendation prompt without appending any reasoning trajectory and compute the rank of the ground-truth item $y _ { i }$ over the complete catalog:

$$
\rho _ { i } ( \emptyset ) = \sum _ { v _ { j } \in \mathcal { V } } \mathbb { I } [ s _ { \phi } ( u _ { i } , v _ { j } \mid \emptyset ) > s _ { \phi } ( u _ { i } , y _ { i } \mid \emptyset ) ] .\tag{5}
$$

For a candidate $r \in \mathcal { C } _ { i } ^ { ( e ) }$ , we append it to the same prompt and recompute the rank:

$$
\rho _ { i } ( r ) = \sum _ { v _ { j } \in \mathcal { V } } \mathbb { I } [ s _ { \phi } ( u _ { i } , v _ { j } \mid r ) > s _ { \phi } ( u _ { i } , y _ { i } \mid r ) ] .\tag{6}
$$

A lower value of $\rho _ { i } ( r )$ indicates that the candidate places the target closer to the top of the ranking list. Then, we define the trajectory utility score as the normalized rank improvement:

$$
A _ { i } ( \boldsymbol { r } ) = \frac { \rho _ { i } ( \boldsymbol { \emptyset } ) - \rho _ { i } ( \boldsymbol { r } ) } { | \mathcal { V } | } ,\tag{7}
$$

where $A _ { i } ( r ) > 0$ means that reasoning improves the target rank and $A _ { i } ( r ) < 0$ indicates that it worsens the target-item rank relative to the no-reasoning input. Normalization by the catalog size constrains the utility score to (−1, 1) and makes the scale comparable across datasets with different catalog sizes. Intuitively, trajectories that yield larger ranking improvements better capture the user’s latent preferences and are therefore considered more beneficial for the downstream recommendation task. Therefore, MUSE selects the trajectories with the highest and lowest assessed utility as the preferred and dispreferred candidates, respectively:

$$
\begin{array} { r l } & { r _ { i } ^ { + } = \arg \underset { r \in \mathcal { C } _ { i } ^ { ( e ) } } { \operatorname* { m a x } } A _ { i } ( r ) , \quad r _ { i } ^ { - } = \arg \underset { r \in \mathcal { C } _ { i } ^ { ( e ) } } { \operatorname* { m i n } } A _ { i } ( r ) , } \\ & { \qquad \quad \Delta _ { i } ^ { A } = A _ { i } ( r _ { i } ^ { + } ) - A _ { i } ( r _ { i } ^ { - } ) . } \end{array}\tag{8}
$$

The trajectory $r _ { i } ^ { + }$ serves as the primary distillation target, while the pair $( r _ { i } ^ { + } , r _ { i } ^ { - } )$ captures which reasoning should be preferred and how strongly it should be preferred.

## 3.3 Utility-Aware Contrastive Training of MUSE

MUSE consists of two trainable components: a reasoning generator that learns to generate recommendation-beneficial preference reasoning, and a recommendation encoder that learns to leverage the generated reasoning to produce the final recommendation. During training, to more effectively exploit reasoning trajectories of varying quality/utility, we introduce the dual-level contrastive learning objectives (i.e., Contrastive Trajectory Distillation and Contrastive Reasoning Utilization) for these two key components. Next, we present the training details for each component.

## 3.3.1 Student reasoning distillation.

The primary objective of this stage is to identify suitable reasoning trajectories that serve as supervision targets to guide the student’s distillation learning. Since Rank-Gain Utility Assessment already explicitly evaluates the recommendation utility of different reasoning trajectories, it naturally provides effective supervision signals for training the student reasoning generator. Specifically, for each training instance, we select the candidate trajectory $r _ { i } ^ { + }$ with the highest trajectory utility score as the distillation target, since it most accurately captures the user’s true preferences and thus contributes the most to the downstream recommendation objective. The student generator is then optimized using the standard autoregressive negative log-likelihood objective:

$$
\mathcal { L } _ { \mathrm { S F T } } = - \sum _ { i \in \mathcal { B } } \sum _ { t = 1 } ^ { L _ { i } ^ { + } } \log \mathcal { G } _ { \theta } \left( r _ { i , t } ^ { + } \mid x _ { i } , r _ { i , < t } ^ { + } \right) .\tag{9}
$$

However, this standard knowledge distillation only encourages the student to imitate the selected high-quality reasoning trajectory $r _ { i } ^ { + }$ , but provides no explicit supervision on which reasoning trajectories should be avoided. Fortunately, the proposed RGUA enables us to readily identify not only the most but also the least recommendation-beneficial reasoning trajectories according to their utility for downstream recommendation. Based on these reasoning pairs, we further introduce a Contrastive Trajectory Distillation (ConDistill) objective that encourages the student to assign a higher likelihood to the most beneficial reasoning trajectory than to the least beneficial one, formulated as:

$$
\mathcal { L } _ { \mathrm { C o n D i s t i l l } } = \frac { 1 } { | \mathcal { B } | } \sum _ { i \in \mathcal { B } } \left[ \gamma _ { i } - \left( \bar { \ell } _ { \theta } ( r _ { i } ^ { + } \ | \ x _ { i } ) - \bar { \ell } _ { \theta } ( r _ { i } ^ { - } \ | \ x _ { i } ) \right) \right] _ { + } ,\tag{10}
$$

where $\begin{array} { r } { \bar { \ell } _ { \theta } ( r | x _ { i } ) = \frac { 1 } { | r | } \sum _ { t = } ^ { | r | } } \end{array}$ <sub>1</sub> log $\mathcal { G } _ { \theta } ( r _ { t } | \boldsymbol { x } _ { i } , r _ { < t } )$ is the length-normalized log-likelihood of a trajectory and $\boldsymbol { B }$ is the current batch data. Length normalization prevents longer trajectories from being systematically penalized by accumulated token log-likelihood. $\gamma _ { i } = [ \Delta _ { i } ^ { \check { A } } ] _ { + }$ is the desired likelihood gap margin for instance $i , \mathrm { ~ A ~ }$ pair with a large trajectory utility gap imposes a stronger separation requirement, whereas similar-quality trajectories induce a smaller margin, thereby enabling adaptive margin adjustment. This contrastive optimization objective explicitly enlarges the likelihood gap between recommendation-beneficial and recommendation-detrimental reasoning trajectories, thereby enabling the student reasoning generator to better discriminate high-quality reasoning during training. Finally, combining the autoregressive SFT loss with the proposed ConDistill objective, the overall training objective of the student generator is defined as:

$$
{ \mathcal { L } } _ { \mathrm { g e n } } = { \mathcal { L } } _ { \mathrm { S F T } } + \lambda _ { \mathrm { C o n D i s t i l l } } { \mathcal { L } } _ { \mathrm { C o n D i s t i l l } } ,\tag{11}
$$

where $\lambda _ { \mathrm { C o n D i s t i l l } } = 0 . 5$ balances the contributions of different terms.

## 3.3.2 Recommendation encoder training.

While the preceding stage trains the student generator to produce recommendation-beneficial reason ing trajectories, accurate recommendations further require the recommendation encoder to effectively utilize the generated reasoning for user preference modeling. Therefore, the primary objective of this stage is to train the recommendation encoder to align the reasoning-enhanced user representation with the target item representation. At inference time, generating reasoning trajectories with multiple teacher LLMs is impractical due to the stringent latency requirement of online recommendation. Consequently, the recommendation encoder must consume reasoning generated solely by the student model during inference. Training the recommendation encoder exclusively with teacher-generated reasoning would therefore create a train–test discrepancy. To maintain consistency, the current student first generates a trajectory $r _ { i } ^ { \mathrm { c u r } }$ for each training user. Then, the user and the corresponding target item representations are denoted by: $\mathbf { z } _ { i } ^ { \mathrm { c u r } } = \mathcal { E } _ { \phi } ( [ x _ { i } ; r _ { i } ^ { \mathrm { c u r } } ] ) , \mathbf { q } _ { i } ^ { + } = \mathcal { E } _ { \phi } ( [ y _ { i } ; a _ { y _ { i } } ] )$ . For a mini-batch B, we first optimize the recommendation encoder using the typical in-batch softmax [55, 36]:

$$
\mathcal { L } _ { \mathrm { N C E } } = - \frac { 1 } { | \mathcal { B } | } \sum _ { i \in \mathcal { B } } \log \frac { \exp ( s _ { \phi } ( u _ { i } , y _ { i } | \boldsymbol { r } _ { i } ^ { \mathrm { c u r } } ) / \tau _ { r } ) } { \sum _ { j \in \mathcal { B } } \exp ( s _ { \phi } ( u _ { i } , y _ { j } | \boldsymbol { r } _ { i } ^ { \mathrm { c u r } } ) / \tau _ { r } ) } ,\tag{12}
$$

where $s _ { \phi } ( u _ { i } , y _ { i } | r _ { i } ^ { \mathrm { c u r } } ) = \sin ( \mathbf { z } _ { i } ^ { \mathrm { c u r } } , \mathbf { q } _ { i } ^ { + } ) , \tau _ { r } = 0 . 1$ is the temperature, and the target items of other users serve as negatives.

However, the standard recommendation objective provides only outcome-level supervision: it requires the resulting user representation to align with the target item, but does not constrain the encoder to achieve this alignment by exploiting recommendation-relevant reasoning signals. Since the interaction history itself can provide strong predictive cues, the encoder may reduce the recommendation loss by relying primarily on historical interactions, leaving the preference information conveyed by the reasoning trajectory insufficiently utilized. To provide additional supervision across alternative reasoning trajectories, we introduce Contrastive Reasoning Utilization (ConUtil), which exploits the preferred and dispreferred reasoning trajectories to provide fine-grained supervision for reasoningaware representation learning. Specifically, ConUtil contrasts the user representations induced by $r _ { i } ^ { + }$ and $r _ { i } ^ { - }$ while keeping the interaction history and target item unchanged. This controlled comparison complements the standard outcome-level supervision by explicitly revealing how reasoning quality should influence target-item matching. As a result, $r _ { i } ^ { + }$ is encouraged to induce a user representation that is more strongly aligned with the target item than $r _ { i } ^ { - }$ , enabling the recommendation encoder to emphasize recommendation-beneficial preference evidence while attenuating irrelevant or misleading reasoning signals. Formally, the user representations based on different reasonings are defined as: $\mathbf { z } _ { i } ^ { + } = \mathcal { \bar { E } } _ { \phi } \bar { ( } [ x _ { i } ; r _ { i } ^ { + } ] ) , \mathbf { z } _ { i } ^ { - } = \mathcal { \bar { E } } _ { \phi } ( [ x _ { i } ; r _ { i } ^ { - } ] )$ . ConUtil penalizes cases where the target item representation $\mathbf { q } _ { i } ^ { + } = \mathcal { E } _ { \phi } ( [ y _ { i } ; a _ { y _ { i } } ] )$ is more similar to the user representation induced by $r _ { i } ^ { - }$ than to that induced by $r _ { i } ^ { + }$

$$
\mathcal { L } _ { \mathrm { C o n U t i l } } = \frac { 1 } { | \mathcal { B } | } \sum _ { i \in \mathcal { B } } \left[ s _ { \phi } ( u _ { i } , y _ { i } | r ^ { - } ) - s _ { \phi } ( u _ { i } , y _ { i } | r ^ { + } ) \right] _ { + } ,\tag{13}
$$

where $s _ { \phi } ( u _ { i } , y _ { i } | r ^ { - } ) \ = \ \mathrm { s i m } ( \mathbf { z } _ { i } ^ { - } , \mathbf { q } _ { i } ^ { + } )$ and $s _ { \phi } ( u _ { i } , y _ { i } | r ^ { + } ) = \mathrm { s i m } ( \mathbf { z } _ { i } ^ { + } , \mathbf { q } _ { i } ^ { + } )$ . In this way, ConUtil encourages the recommendation encoder to explicitly exploit recommendation-relevant preference evidence. Finally, the overall objective for the recommendation encoder is:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { r e c } } = \mathcal { L } _ { \mathrm { N C E } } + \lambda _ { \mathrm { C o n U t i l } } \mathcal { L } _ { \mathrm { C o n U t i l } } , } \end{array}\tag{14}
$$

where $\lambda _ { \mathrm { C o n U t i l } } = 0 . 5$ controls the contribution of different terms.

## 3.4 Inference

MUSE requires neither teacher LLMs nor trajectory utility estimation during inference. Given a new user’s prompt $x _ { i } ,$ , the well-trained student generator first produces a preference reasoning trajectory: ${ \hat { r } } _ { i } = { \mathcal G } _ { \theta } ( x _ { i } )$ . The recommendation encoder then computes the user representation $\hat { \mathbf { z } } _ { i } = \mathbf { \mathcal { \bar { E } } } _ { \phi } ( \mathbf { \bar { [ } } x _ { i } ; \hat { r } _ { i } ] )$ and compares it with the representations of all catalog items using cosine similarity. The final recommendation list is defined as:

$$
\begin{array} { r } { \widehat { \mathcal { R } } _ { i } ^ { K } = \mathrm { T o p K } _ { v _ { j } \in \mathcal { V } } s _ { \phi } ( u _ { i } , v _ { j } | \hat { r } _ { i } ) = \mathrm { T o p K } _ { v _ { j } \in \mathcal { V } } \mathrm { s i m } ( \hat { \mathbf { z } } _ { i } , \mathbf { q } _ { j } ) . } \end{array}\tag{15}
$$

The resulting pipeline requires neither costly teacher calls nor candidate evaluation at test time, while retaining the recommendation-beneficial reasoning discovered from multiple teachers and the student’s own exploration. The complete training and inference procedures are summarized in Algorithm 1.

## 4 Experiments

## 4.1 Experimental Details

4.1.1 Datasets. Following prior studies [55, 3, 4], we conduct experiments on three publicly available real-world datasets from Amazon, namely CDs and Vinyl (CDs), Video Games (Games), and Mu sical Instruments (Instruments). We adopt the widely used leave-one-out evaluation protocol [40]. Detailed dataset statistics are summarized in Table 1. To evaluate recommendation performance, we employ two widely used ranking metrics, Hit Rate (H@K) and Normalized Discounted Cumulative Gain (N@K), with $K \in \{ \breve { 5 } , 1 0 , 2 0 \}$ . Following common practice, we adopt the full-ranking evaluation protocol, where each user’s ranking metrics are computed over the entire candidate item set.

Table 1: Dataset Statistics.
<table><tr><td>Dataset</td><td>Users</td><td>Items</td><td>Interactions</td></tr><tr><td>Video Games</td><td>29,230</td><td>10,144</td><td>63,502</td></tr><tr><td>CDs and Vinyl</td><td>7,701</td><td>12,024</td><td>13,435</td></tr><tr><td>Musical Instruments</td><td>15,656</td><td>10,320</td><td>34,373</td></tr></table>

4.1.2 Baselines. To comprehensively evaluate the effectiveness of the proposed method, we compare it against a diverse set of competitive baselines from different categories, including traditional sequential RecSys (GRU4Rec [17], Caser [46], SASRec [23]), LLM-based RecSys (TIGER [41], BigRec [4], and $\mathbf { \dot { D } } ^ { 3 } \left[ 3 \right] )$ , and reasoning-augmented RecSys (RecSAVER [51], LangPTune [14], and R<sup>2</sup>ec [55]). More detailed descriptions of these baselines are provided in Appendix B.1.

4.1.3 Implementation. All baselines and MUSE are implemented in PyTorch. We instantiate MUSE with two student reasoning generators, Gemma-2-2B [48] and Qwen2.5-3B [49], together with a fully fine-tuned recommendation encoder, GTE-large-en-v1.5 [57, 27], which produces 1024-dimensional CLS-pooled embeddings with a maximum input length of 2,048 tokens. For more implementation details of MUSE and the prompts used, please refer to Appendix B.2 and Appendix C.

## 4.2 Overall Performance

Table 2, Table 3, and Table 7 (Appendix B.3) report the overall recommendation performance of different methods. Several important observations can be drawn. (1) First, incorporating explicit preference reasoning consistently improves recommendation performance. Across almost all datasets and evaluation metrics, reasoning-augmented RecSys, including RecSAVER, LangPTune, $\mathrm { R ^ { 2 } e c } .$ , and the proposed MUSE, consistently outperform both traditional sequential recommenders (e.g., Caser and GRU4Rec) and LLM-based RecSys baselines without explicit reasoning $( \mathrm { e . g . , D ^ { 3 } }$ and BigRec). These results demonstrate that explicitly modeling users’ latent preferences through reasoning provides more informative semantic signals for user preference modeling, leading to substantially better recommendation performance. (2) Second, MUSE consistently outperforms RL-based methods, including LangPTune and $\scriptstyle \mathrm { \mathsf { R } } ^ { 2 } \mathrm { e c }$ , despite relying solely on reasoning distillation during training. This result suggests that carefully selecting high-quality reasoning trajectories can be more effective than directly optimizing reasoning through reinforcement learning. Moreover, reasoning distillation is naturally complementary to RL-based post-training, allowing the distilled student model to serve as a stronger initialization for subsequent reinforcement learning. Therefore, the proposed framework not only achieves superior recommendation performance by itself but also provides a more effective foundation for further reasoning optimization. (3) Third, the proposed MUSE consistently achieves the best overall performance among all reasoning-based recommendation methods. Compared with the distillation-based baseline RecSAVER, which relies on a single fixed teacher for distillation, MUSE constructs a diverse reasoning pool using multiple teachers and progressive self-exploration, and further selects the most recommendation-beneficial reasoning trajectories through the proposed Rank-Gain Utility Assessment. This dynamic reasoning distillation paradigm consistently yields more effective supervision, resulting in substantial improvements across almost all datasets and metrics. Due to space limitations, additional case studies are provided in Appendix D.

Table 2: The overall performance of baselines and MUSE (Dataset: CDs and Vinyl).
<table><tr><td>Datasets</td><td colspan="6">CDs and Vinyl</td></tr><tr><td>Methods</td><td>N@5</td><td>N@10</td><td>N@20</td><td>H@5</td><td>H@10</td><td>H@20</td></tr><tr><td colspan="7">Caser 0.0034±0.0003</td></tr><tr><td>GRU4Rec</td><td>0.0036±0.0013</td><td>0.0046±0.0008 0.0049±0.0016</td><td>0.0061±0.0005 0.0068±0.0021</td><td>0.0068±0.0008 0.0059±0.0022</td><td>0.0105±0.0021 0.0102±0.0033</td><td>0.0166±0.0011 0.0178±0.0049</td></tr><tr><td>SASRec</td><td>0.0051±0.0009</td><td>0.0060±0.0009</td><td>0.0070±0.0006</td><td>0.0084±0.0007</td><td>0.0114±0.0008</td><td>0.0154±0.0014</td></tr><tr><td>TIGER</td><td>0.0023±0.0009</td><td>0.0034±0.0009</td><td>0.0041±0.0006</td><td>0.0033±0.0011</td><td>0.0069±0.0011</td><td>0.0095±0.0003</td></tr><tr><td colspan="7">Qwen2.5-3B</td></tr><tr><td colspan="7"></td></tr><tr><td>D3 BigRec</td><td>0.0030±0.0024 0.0031±0.0009</td><td>0.0048±0.0028 0.0033±0.0011</td><td>0.0070±0.0033 0.0037±0.0011</td><td>0.0054±0.0041 0.0040±0.0012</td><td>0.0110±0.0053 0.0048±0.0017</td><td>0.0196±0.0073 0.0064±0.0020</td></tr><tr><td>RecSAVER</td><td>0.0138±0.0014</td><td>0.0175±0.0032</td><td>0.0215±0.0023</td><td>0.0201±0.0051</td><td>0.0315±0.0105</td><td>0.0475±0.0069</td></tr><tr><td>LangPTune</td><td>0.0231±0.0011</td><td>0.0289±0.0006</td><td>0.0338±0.0010</td><td>0.0349±0.0008</td><td>0.0531±0.0011</td><td>0.0729±0.0007</td></tr><tr><td>R2ec</td><td>0.0243±0.0006</td><td>0.0302±0.0002</td><td>0.0352±0.0005</td><td>0.0362±0.0011</td><td>0.0545±0.0008</td><td>0.0743±0.0012</td></tr><tr><td>MUSE</td><td>0.0295±0.0009</td><td>0.0348±0.0017</td><td>0.0404±0.0010</td><td>0.0421±0.0019</td><td>0.0584±0.0041</td><td>0.0804±0.0018</td></tr><tr><td colspan="7">Gemma-2-2B</td></tr><tr><td colspan="7"></td></tr><tr><td>D3 BigRec</td><td>0.0015±0.0011 0.0026±0.0004</td><td>0.0019±0.0007 0.0028±0.0003</td><td>0.0031±0.0008</td><td>0.0026±0.0015</td><td>0.0040±0.0011</td><td>0.0086±0.0004</td></tr><tr><td>RecSAVER</td><td>0.0154±0.0030</td><td></td><td>0.0039±0.0004</td><td>0.0039±0.0003</td><td>0.0058±0.0006</td><td>0.0093±0.0005</td></tr><tr><td>LangPTune</td><td>0.0204±0.0026</td><td>0.0196±0.0029 0.0254±0.0025</td><td>0.0243±0.0029</td><td>0.0212±0.0053</td><td>0.0349±0.0051</td><td>0.0533±0.0050</td></tr><tr><td>R2ec</td><td>0.0234±0.0020</td><td></td><td>0.0302±0.0028</td><td>0.0285±0.0039</td><td>0.0442±0.0037</td><td>0.0631±0.0046</td></tr><tr><td></td><td></td><td>0.0282±0.0010</td><td>0.0355±0.0008</td><td>0.0355±0.0039</td><td>0.0503±0.0016</td><td>0.0795±0.0011</td></tr><tr><td>MUSE</td><td>0.0273±0.0029</td><td>0.0332±0.0035</td><td>0.0394±0.0049</td><td>0.0379±0.0039</td><td>0.0561±0.0064</td><td>0.0806±0.0122</td></tr></table>

Red and Blue denote the best and second-best results, respectively.

## 4.3 Ablation Study

We first compare MUSE with three single-teacher variants, where the reasoning trajectories are generated solely by GPT, Gemini, or DeepSeek, respectively, to evaluate the effectiveness of the proposed multi-teacher distillation strategy. We further investigate the contribution of each proposed component. MUSE w/o SE disables the proposed progressive self-exploration strategy, leaving only teacher-generated reasoning trajectories for distillation. MUSE-Rand replaces the proposed RGUA by randomly selecting a reasoning trajectory from the teacher- and self-generated candidate pool for supervision. Finally, MUSE w/o ConDistill removes the proposed ConDistill objective for the reasoning generator, while MUSE w/o ConUtil removes the proposed ConUtil objective for the recommendation encoder. Table 4, Table 5, and Table 8 (Appendix B.4) report the experimental results. Several important observations can be made.

Table 3: The overall performance of baselines and MUSE (Dataset: Musical Instruments).
<table><tr><td>Datasets</td><td colspan="6">Musical Instruments</td></tr><tr><td>Methods</td><td>N@5</td><td>N@10</td><td>N@20</td><td>H@5</td><td>H@10</td><td>H@20</td></tr><tr><td>Caser</td><td>0.0087±0.0022</td><td>0.0134±0.0025</td><td>0.0191±0.0018</td><td>0.0169±0.0043</td><td>0.0317±0.0055</td><td>0.0543±0.0034</td></tr><tr><td>GRU4Rec</td><td>0.0090±0.0007</td><td>0.0117±0.0010</td><td>0.0152±0.0017</td><td>0.0143±0.0013</td><td>0.0225±0.0022</td><td>0.0367±0.0048</td></tr><tr><td>SASRec</td><td>0.0058±0.0010</td><td>0.0097±0.0004</td><td>0.0133±0.0012</td><td>0.0099±0.0013</td><td>0.0219±0.0018</td><td>0.0362±0.0058</td></tr><tr><td>TIGER</td><td>0.0139±0.0038</td><td>0.0173±0.0034</td><td>0.0218±0.0022</td><td>0.0208±0.0050</td><td>0.0313±0.0041</td><td>0.0491±0.0011</td></tr><tr><td colspan="7">Qwen2.5-3B</td></tr><tr><td>D3</td><td>0.0089±0.0024</td><td>0.0124±0.0036</td><td>0.0152±0.0051</td><td>0.0170±0.0044</td><td>0.0273±0.0081</td><td>0.0386±0.0141 0.0123±0.0015</td></tr><tr><td>BigRec RecSAVER</td><td>0.0038±0.0006 0.0105±0.0025</td><td>0.0049±0.0007 0.0144±0.0021</td><td>0.0059±0.0003 0.0189±0.0022</td><td>0.0061±0.0003 0.0156±0.0031</td><td>0.0087±0.0004 0.0279±0.0022</td><td>0.0456±0.0024</td></tr><tr><td>LangPTune</td><td>0.0159±0.0009</td><td>0.0209±0.0008</td><td>0.0258±0.0009</td><td>0.0241±0.0019</td><td>0.0397±0.0025</td><td>0.0590±0.0026</td></tr><tr><td>R2ec</td><td>0.0133±0.0005</td><td>0.0195±0.0012</td><td>0.0274±0.0006</td><td>0.0206±0.0009</td><td>0.0402±0.0029</td><td>0.0713±0.0006</td></tr><tr><td>MUSE</td><td>0.0194±0.0008</td><td>0.0255±0.0005</td><td>0.0323±0.0013</td><td>0.0290±0.0011</td><td>0.0480±0.0008</td><td>0.0755±0.0030</td></tr><tr><td colspan="7">Gemma-2-2B</td></tr><tr><td>D3</td><td>0.0090±0.0033</td><td>0.0141±0.0031</td><td>0.0201±0.0009</td><td>0.0148±0.0058</td><td>0.0306±0.0056</td><td>0.0547±0.0080</td></tr><tr><td>BigRec</td><td>0.0010±0.0005</td><td>0.0017±0.0005</td><td>0.0029±0.0006</td><td>0.0011±0.0005</td><td>0.0032±0.0005</td><td>0.0075±0.0010</td></tr><tr><td>RecSAVER</td><td>0.0103±0.0008</td><td>0.0138±0.0003</td><td>0.0186±0.0003</td><td>0.0157±0.0020</td><td>0.0270±0.0027</td><td>0.0458±0.0041</td></tr><tr><td>LangPTune</td><td>0.0125±0.0020</td><td>0.0177±0.0018</td><td>0.0221±0.0011</td><td>0.0190±0.0034</td><td>0.0353±0.0029</td><td>0.0529±0.0026</td></tr><tr><td>R2ec</td><td>0.0174±0.0003</td><td>0.0231±0.0006</td><td>0.0294±0.0016</td><td>0.0259±0.0014</td><td>0.0436±0.0021</td><td>0.0686±0.0063</td></tr><tr><td>MUSE</td><td>0.0195±0.0009</td><td>0.0240±0.0011</td><td>0.0317±0.0011</td><td>0.0295±0.0009</td><td>0.0448±0.0007</td><td>0.0742±0.0011</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Red and Blue denote the best and second-best results, respectively.

First, multi-teacher distillation consistently outperforms all single-teacher variants across all datasets. This demonstrates that no single teacher can consistently provide the most recommendation-beneficial reasoning for every user. By leveraging diverse reasoning trajectories from multiple teachers and selecting the most effective one through the proposed RGUA, MUSE provides higher-quality supervision than relying on any individual teacher alone. Second, each proposed component contributes positively to the overall performance. Disabling progressive self-exploration (MUSE w/o SE) results in inferior performance, suggesting that allowing the student to continuously discover recommendationbeneficial reasoning beyond the fixed teacher pool effectively alleviates the teacher knowledge ceiling. Replacing Rank-Gain Utility Assessment with random trajectory selection (MUSE-Rand) leads to a noticeable performance degradation, validating the importance of evaluating reasoning trajectories according to their contribution to downstream recommendation performance rather than treating all trajectories equally. Moreover, removing either the ConDistill (w/o ConDistill) or the ConUtil objective (w/o ConUtil) consistently degrades recommendation accuracy, indicating that jointly learning to distinguish both beneficial and detrimental reasoning is important for improving the reasoning generator and the recommendation encoder.

Table 4: Ablation study on key components (Backbone: Qwen2.5-3B; Datasets: CDs and Vinyl).
<table><tr><td>Datasets</td><td colspan="6">CDs and Vinyl</td></tr><tr><td>Methods</td><td>N@5</td><td>N@10</td><td>N@20</td><td>H@5</td><td>H@10</td><td>H@20</td></tr><tr><td>GPT</td><td>0.0265±0.0004</td><td>0.0323±0.0007</td><td>0.0376±0.0002</td><td>0.0372±0.0014</td><td>0.0552±0.0011</td><td>0.0764±0.0014</td></tr><tr><td>Gemini</td><td>0.0243±0.0011</td><td>0.0299±0.0010</td><td>0.0356±0.0012</td><td>0.0333±0.0010</td><td>0.0510±0.0009</td><td>0.0743±0.0013</td></tr><tr><td>DeepSeek</td><td>0.0258±0.0031</td><td>0.0316±0.0036</td><td>0.0372±0.0031</td><td>0.0388±0.0048</td><td>0.0563±0.0059</td><td>0.0783±0.0053</td></tr><tr><td>w/o SE</td><td>0.0258±0.0021</td><td>0.0305±0.0032</td><td>0.0380±0.0020</td><td>0.0363±0.0035</td><td>0.0510±0.0069</td><td>0.0803±0.0013</td></tr><tr><td>Rand</td><td>0.0269±0.0020</td><td>0.0327±0.0020</td><td>0.0373±0.0021</td><td>0.0384±0.0023</td><td>0.0559±0.0020</td><td>0.0743±0.0025</td></tr><tr><td>w/o ConDistill</td><td>0.0257±0.0031</td><td>0.0297±0.0039</td><td>0.0355±0.0035</td><td>0.0356±0.0045</td><td>0.0482±0.0069</td><td>0.0710±0.0060</td></tr><tr><td>w/o ConUtil</td><td>0.0262±0.0010</td><td>0.0322±0.0009</td><td>0.0395±0.0009</td><td>0.0381±0.0018</td><td>0.0568±0.0025</td><td>0.0825±0.0042</td></tr><tr><td>MUSE</td><td>0.0295±0.0009</td><td>0.0348±0.0017</td><td>0.0404±0.0010</td><td>0.0421±0.0019</td><td>0.0584±0.0041</td><td>0.0804±0.0018</td></tr></table>

## 4.4 Model Analysis

4.4.1 Parameter Analysis. To investigate the sensitivity of MUSE to different hyperparameters, we vary one hyperparameter at a time while keeping the others fixed and evaluate the resulting recommendation performance. The experiments are conducted on the Video Games dataset with Qwen2.5-3B as the backbone model. As shown in Figure 3, increasing the maximum number of student reasoning candidates $S _ { \mathrm { m a x } }$ from a small value generally improves recommendation performance, with strong results observed around $S _ { \mathrm { m a x } } = 6$ and $S _ { \mathrm { m a x } } \dot { = } 7$ . A small $S _ { \mathrm { m a x } }$ restricts self-exploration, preventing the student from sufficiently exploring potentially more recommendation-beneficial reasoning trajectories and thereby leaving it constrained by the knowledge ceiling of the teacher models. In practice, $S _ { \mathrm { m a x } }$ should therefore be selected according to the teacher’s capability to balance teacher-guided distillation and student self-exploration. Stronger teachers require a smaller $S _ { \mathrm { m a x } }$ for sufficient supervision, whereas weaker teachers benefit from a larger $S _ { \mathrm { m a x } }$ to promote self-exploration and reduce teacher dependence. For $\lambda _ { \mathrm { C o n U t i l } }$ and $\lambda _ { \mathrm { C o n D i s t i l l } } .$ , the recommendation performance exhibits only moderate fluctuations within the tested ranges, without substantial degradation under different settings. This demonstrates that MUSE is robust to the weighting of the two contrastive objectives and does not require excessively fine-grained hyperparameter tuning to achieve competitive performance.

Table 5: Ablation study on key components (Backbone: Qwen2.5-3B; Datasets: Musical Instruments).
<table><tr><td rowspan="2">Datasets</td><td colspan="6">Musical Instruments</td></tr><tr><td>N@5</td><td>N@10</td><td>N@20</td><td>H@5</td><td>H@10</td><td>H@20</td></tr><tr><td>Methods GPT</td><td>0.0155±0.0004</td><td>0.0200±0.0002</td><td>0.0264±0.0002</td><td>0.0250±0.0011</td><td>0.0388±0.0009</td><td>0.0645±0.0006</td></tr><tr><td>Gemini</td><td>0.0141±0.0004</td><td>0.0193±0.0012</td><td>0.0236±0.0014</td><td>0.0213±0.0006</td><td>0.0377±0.0032</td><td>0.0549±0.0039</td></tr><tr><td>DeepSeek</td><td>0.0146±0.0020</td><td>0.0188±0.0033</td><td>0.0238±0.0036</td><td>0.0221±0.0044</td><td>0.0353±0.0086</td><td>0.0554±0.0100</td></tr><tr><td>w/o SE</td><td></td><td></td><td>0.0318±0.0012</td><td>0.0282±0.0008</td><td></td><td></td></tr><tr><td>Rand</td><td>0.0195±0.0006 0.0165±0.0006</td><td>0.0231±0.0013 0.0218±0.0008</td><td>0.0279±0.0008</td><td>0.0243±0.0008</td><td>0.0456±0.0011 0.0404±0.0031</td><td>0.0719±0.0008 0.0645±0.0019</td></tr><tr><td>w/o ConDistill</td><td>0.0168±0.0019</td><td>0.0217±0.0023</td><td>0.0285±0.0020</td><td>0.0262±0.0030</td><td>0.0414±0.0039</td><td>0.0684±0.0029</td></tr><tr><td>w/o ConUtil</td><td>0.0139±0.0006</td><td>0.0199±0.0014</td><td>0.0258±0.0008</td><td>0.0214±0.0009</td><td>0.0400±0.0031</td><td>0.0639±0.0008</td></tr><tr><td>MUSE</td><td>0.0194±0.0008</td><td>0.0255±0.0005</td><td>0.0323±0.0013</td><td>0.0290±0.0011</td><td>0.0480±0.0008</td><td>0.0755±0.0030</td></tr></table>

![](images/b4e946641d83c8f6aa1dc8962dfd7598aae9f0d1c19c1e38e02aa6beb0e2d8b0.jpg)  
Figure 3: Effect of the hyper-parameters.

4.4.2 Training and Inference Efficiency. (1) Training Efficiency. Figure 4 demonstrates that MUSE does not introduce prohibitive training overhead despite employing multiple teacher LLMs and student self-exploration. The reason is that all teacher-generated reasoning trajectories are produced in advance and cached for reuse, avoiding repeated teacher inference during iterative training. Meanwhile, Rank-Gain Utility Assessment only uses the lightweight recommendation encoder to perform candidate ranking, without introducing an additional LLM-based evaluator. Consequently, MUSE is substantially more efficient than LangPTune and incurs comparable computational overhead to $\mathrm { R ^ { 2 } e c } .$ , while consistently achieving superior recommendation performance.

![](images/5106e9d8aaf3fdd243172b5411f6167a125a32bcdcf8563f260f3bbd47606d0c.jpg)  
Figure 4: Training time of various methods.

(2) Inference Efficiency. We further compare the inference efficiency of different reasoning-augmented recommender systems. All methods are deployed with the same vLLM inference engine [26] and evaluated on identical hardware. As shown in Figure 5, our method achieves competitive inference latency across all three datasets. While $\mathsf { R } ^ { 2 } \mathrm { e c }$ exhibits the lowest inference latency, our method consistently delivers substantially better recommendation performance, as demonstrated in Table 2, Table 3, and Table 7 (Appendix B.3). These results indicate that the proposed framework introduces little additional inference overhead, while providing a favorable trade-off between recommendation accuracy and inference efficiency.

![](images/2ffa33a4ecf56504cd98205d7ea86d2fa66c87ba887efadb83c3d26f0d169cf2.jpg)  
Figure 5: Recommendation time of various methods.

4.4.3 Effectiveness of Progressive Self-Exploration. Figure 6 illustrates the evolution of the proposed self-exploration strategy during training. The experiment is conducted on the Video Games dataset using Qwen2.5-3B as the student model. The red curve (CTR) reports the ratio of reasoning trajectories selected from the student model. At the beginning of training, only a small proportion of student-generated reasoning is selected because the student has not yet acquired sufficient recommendation-specific reasoning capability. As training progresses, the student continuously improves by learning from recommendation-beneficial reasoning distilled from multiple teachers, enabling it to generate increasingly informative reasoning trajectories. Consequently, RGUA gradually selects more student-generated reasoning instead of teacher-generated reasoning for supervision. Meanwhile, the validation performance consistently improves across all metrics, indicating that the progressively explored reasoning provides increasingly effective guidance for downstream recommendation.

![](images/81695d478066f1c54e5f840a431a0352a6050a5872ff430df0d42d6aabf81eab.jpg)

![](images/bb61fa74a2c84d3713da8c4ef75d17e143a412e586b8e336a185afd706563760.jpg)  
Figure 6: Validation performance and student reasoning selection ratio during training.

To further verify that the effectiveness of MUSE is not merely attributable to using more reasoning trajectories, we compare it with MTMR, where each teacher generates multiple trajectories without student self-exploration, while keeping the total number of candidate trajectories identical to that of MUSE. As shown in Table 6, MUSE outperforms MTMR on most metrics, with particularly clear improvements on CDs and Vinyl and Video Games. These results indicate that increasing the quantity of teacher-generated reasoning alone is insufficient. Instead, student self-exploration introduces complementary reasoning beyond teacher imitation, thereby helping the student overcome the teacher knowledge ceiling.

Table 6: The overall performance of MTMR and MUSE.
<table><tr><td rowspan=1 colspan=2>|Methods|   N@5</td><td rowspan=1 colspan=1>N@10</td><td rowspan=1 colspan=1>N@20</td><td rowspan=1 colspan=1>H@5</td><td rowspan=1 colspan=1>H@10</td><td rowspan=1 colspan=1>H@20</td></tr><tr><td rowspan=1 colspan=7>CDs and Vinyl</td></tr><tr><td rowspan=1 colspan=2>MTMR |0.0262±0.0010|0.0MUSE0.0295±0.00090</td><td rowspan=1 colspan=1>327±0.0026|0.0.0348±0.00170</td><td rowspan=1 colspan=1>394±0.0033|0.0404±0.0010|0</td><td rowspan=1 colspan=1>.0367±0.0004|0.0421±0.00190</td><td rowspan=1 colspan=1>.0566±0.0049|0.0584±0.00410</td><td rowspan=1 colspan=1>.0832±0.0079.0804±0.0018</td></tr><tr><td rowspan=1 colspan=7>Musical Instruments</td></tr><tr><td rowspan=1 colspan=2>MTMR |0.0192±0.0010|0.0MUSE0.0194±0.00080</td><td rowspan=1 colspan=1>253±0.0012|0.0.0255±0.00050</td><td rowspan=1 colspan=1>320±0.0019|0.0323±0.00130</td><td rowspan=1 colspan=1>.0295±0.0020|0.0290±0.00110</td><td rowspan=1 colspan=1>.0473±0.0028|.0480±0.00080</td><td rowspan=1 colspan=1>0.0735±0.0060.0755±0.0030</td></tr><tr><td rowspan=1 colspan=7>Video Games</td></tr><tr><td rowspan=1 colspan=1>MTMR |0.0MUSE</td><td rowspan=1 colspan=1>438±0.0032|0.00.0421±0.00050</td><td rowspan=1 colspan=1>525±0.0023|0.0.0536±0.0006|0</td><td rowspan=1 colspan=1>626±0.0025|0.0632±0.00090</td><td rowspan=1 colspan=1>.0638±0.0056|0.0626±0.00060</td><td rowspan=1 colspan=1>.0912±0.0028|.1020±0.00230</td><td rowspan=1 colspan=1>0.1313±0.0058.1399±0.0040</td></tr></table>

## 5 Related Work

This section reviews recent advances in reasoning-augmented recommendation. According to their training paradigms, existing approaches can be broadly divided into three categories: SFT distillation based methods, reinforcement learning (RL)-based methods, and hybrid approaches that combine both paradigms. Due to space limitations, the detailed and comprehensive review of the related work is provided in Appendix E.

## 6 Conclusion

In this paper, we presented MUSE, an adaptive reasoning distillation framework for reasoningaugmented recommendation. Unlike existing methods that directly imitate reasoning generated by a single fixed teacher, MUSE introduces multiple teachers and progressively expands the reasoning space through self-exploration, enabling the student to draw on the complementary strengths of different teachers and gradually overcome the knowledge limitation of general-purpose teachers. Furthermore, the proposed Rank-Gain Utility Assessment (RGUA) adaptively identifies recommendationbeneficial reasoning, and our dual-level contrastive learning objectives can jointly improve both reasoning generation and reasoning utilization by explicitly contrasting beneficial and detrimental reasoning trajectories for high-quality reasoning supervision. Extensive experiments on three real-world recommendation datasets demonstrate that MUSE consistently achieves superior recommendation performance, revealing the significant yet largely untapped potential of reasoning distillation for recommender systems.

## References

[1] M Mehdi Afsar, Trafford Crump, and Behrouz Far. Reinforcement learning based recommender systems: A survey. ACM Computing Surveys, 55(7):1–38, 2022.

[2] Xiang Ao, Yiran Qiao, Ningtao Wang, Xing Fu, Yang Liu, Thapana Boonchoo, Weiqiang Wang, Qing He, and Xueqi Cheng. Passing on wisdom: Llm-driven cascaded knowledge distillation for sequential recommendation. IEEE Transactions on Knowledge and Data Engineering, 2026.

[3] Keqin Bao, Jizhi Zhang, Yang Zhang, Xinyue Huo, Chong Chen, and Fuli Feng. Decoding matters: Addressing amplification bias and homogeneity issue in recommendations for large language models. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 10540–10552, 2024.

[4] Keqin Bao, Jizhi Zhang, Wenjie Wang, Yang Zhang, Zhengyi Yang, Yanchen Luo, Chong Chen, Fuli Feng, and Qi Tian. A bi-step grounding paradigm for large language models in recommendation systems. ACM Transactions on Recommender Systems, 3(4):1–27, 2025.

[5] Maximilian Boeker and Aleksandra Urman. An empirical investigation of personalization factors on tiktok. In Proceedings ofthe ACM web conference 2022, pages 2298–2309, 2022.

[6] Ruoyu Chen, Xiaoqing Guo, Kangwei Liu, Siyuan Liang, Shiming Liu, Qunli Zhang, Laiyuan Wang, Hua Zhang, and Xiaochun Cao. Where mllms attend and what they rely on: Explaining autoregressive token generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 17057–17066, 2026.

[7] Yiwen Chen, Fuwei Zhang, Zehao Chen, Deqing Wang, Hehan Li, Peizhi Xu, Hanmeng Liu, Shuanglong Li, Xin Pei, Fuzhen Zhuang, et al. Lasar: Latent adaptive semantic aligned reasoning for generative recommendation. arXiv preprint arXiv:2605.10207, 2026.

[8] Paul Covington, Jay Adams, and Emre Sargin. Deep neural networks for youtube recommendations. In Proceedings of the 10th ACM conference on recommender systems, pages 191–198, 2016.

[9] James Davidson, Benjamin Liebald, Junning Liu, Palash Nandy, Taylor Van Vleet, Ullas Gargi, Sujoy Gupta, Yu He, Mike Lambert, Blake Livingston, et al. The youtube video recommendation system. In Proceedings of the fourth ACM conference on Recommender systems, pages 293–296, 2010.

[10] Hongxun Ding, Keqin Bao, Jizhi Zhang, Yi Fang, Wenxin Xu, Fuli Feng, and Xiangnan He. Towards sample-efficient and stable reinforcement learning for llm-based recommendation. arXiv preprint arXiv:2602.00632, 2026.

[11] Ning Ding, Yujia Qin, Guang Yang, Fuchao Wei, Zonghan Yang, Yusheng Su, Shengding Hu, Yulin Chen, Chi-Min Chan, Weize Chen, et al. Parameter-efficient fine-tuning of large-scale pre-trained language models. Nature machine intelligence, 5(3):220–235, 2023.

[12] Shan Dong, Palakorn Achananuparp, Hieu Hien Mai, Lei Wang, Yao Lu, and Ee-Peng Lim. On reasoning behind next occupation recommendation. In PAKDD, pages 312–324. Springer, 2026.

[13] Jingtong Gao, Zeyu Song, Chi Lu, Xiaopeng Li, Derong Xu, Maolin Wang, Peng Jiang, Kun Gai, Qingpeng Cai, and Xiangyu Zhao. Reinforced preference optimization for reasoning-augmented recommendations. arXiv preprint arXiv:2605.21967, 2026.

[14] Zhaolin Gao, Joyce Zhou, Yijia Dai, and Thorsten Joachims. End-to-end training for recommendation with language-based user profiles. arXiv preprint arXiv:2410.18870, 2024.

[15] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

[16] Emrul Hasan, Mizanur Rahman, Chen Ding, Jimmy Xiangji Huang, and Shaina Raza. Reviewbased recommender systems: a survey of approaches, challenges and future perspectives. ACM Computing Surveys, 58(1):1–41, 2025.

[17] Balázs Hidasi, Alexandros Karatzoglou, Linas Baltrunas, and Domonkos Tikk. Session-based recommendations with recurrent neural networks. arXiv preprint arXiv:1511.06939, 2015.

[18] Zheng Yi Ho, Siyuan Liang, and Dacheng Tao. Understanding hallucinations in large visual and language models. ACM Computing Surveys, 58(13):1–36, 2026.

[19] Nan Hu, Han Li, Jimeng Sun, Lu Wang, Fangkai Yang, Bo Qiao, Pu Zhao, David Dai, Mengyu Liu, Yuefeng Zhan, et al. Adnanny: One reasoning llm for all offline ads recommendation tasks. arXiv preprint arXiv:2602.01563, 2026.

[20] Jiani Huang, Shijie Wang, Liang-Bo Ning, Wenqi Fan, and Qing Li. Rerec: Reasoningaugmented llm-based recommendation assistant via reinforcement fine-tuning. In Proceedings ofthe 64th Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 21040–21055, 2026.

[21] Jie Jiang, Yang Wu, Qian Li, Yuling Xiong, Hongbo Tang, Xun Liu, Haoze Wang, Jun Zhang, Huan Yu, and Hailong Shi. Scoter: Structured chain-of-thought transfer for enhanced recommendation. arXiv preprint arXiv:2511.19514, 2025.

[22] Jie Jiang, Yang Wu, Qian Li, Yuling Xiong, Yihang Su, Junbang Huo, Longfei Lu, Jun Zhang, and Huan Yu. Diffureason: Bridging latent reasoning and generative refinement for sequential recommendation. arXiv preprint arXiv:2602.09744, 2026.

[23] Wang-Cheng Kang and Julian McAuley. Self-attentive sequential recommendation. In 2018 IEEE international conference on data mining (ICDM), pages 197–206. IEEE, 2018.

[24] Jieyong Kim, Hyunseo Kim, Hyunjin Cho, SeongKu Kang, Buru Chang, Jinyoung Yeo, and Dongha Lee. Review-driven personalized preference reasoning with large language models for recommendation. In Proceedings ofthe 48th International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 1697–1706, 2025.

[25] Xiaoyu Kong, Junguang Jiang, Bin Liu, Ziru Xu, Han Zhu, Jian Xu, Bo Zheng, Jiancan Wu, and Xiang Wang. Think before recommendation: Autonomous reasoning-enhanced recommender. Advances in Neural Information Processing Systems, 38:141209–141232, 2026.

[26] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In Proceedings ofthe 29th symposium on operating systems principles, pages 611–626, 2023.

[27] Zehan Li, Xin Zhang, Yanzhao Zhang, Dingkun Long, Pengjun Xie, and Meishan Zhang. Towards general text embeddings with multi-stage contrastive learning. arXiv preprint arXiv:2308.03281, 2023.

[28] Jiawei Liang, Ruoyu Chen, Xianghao Jiao, Siyuan Liang, Shiming Liu, Qunli Zhang, Zheng Hu, and Xiaochun Cao. Explaining multimodal llms via intra-modal token interactions. arXiv preprint arXiv:2509.22415, 2025.

[29] Mingfu Liang, Yufei Li, Jay Xu, Kavosh Asadi, Xi Liu, Shuo Gu, Kaushik Rangadurai, Frank Shyu, Shuaiwen Wang, Song Yang, et al. Gr2: Generative reasoning re-ranker. arXiv preprint arXiv:2602.07774, 2026.

[30] Daizong Liu, Tianyao Luo, Shuwei Huang, Junyi Yang, Siyuan Liang, Xiaojun Jia, Chun Pong Lau, Jianfeng Dong, Yan Liu, Yang Bai, et al. A survey on ai agent security: Reasoning, acting, and self-evolving. AI Plus, 1(1):100004–1, 2026.

[31] Weihai Lu, Xiaoxi Cui, and Chenke Yin. Think, but don’t tell: Implicit reasoning for llmbased sequential recommendation via multi-teacher distillation. In Proceedings of the 49th International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 1244–1254, 2026.

[32] Kaiwen Luo, Zhenhong Zhou, Leyan Wang, Liang Lin, Tianyu Shao, Yuanhe Zhang, Yang Xiao, Yuxuan Li, Miao Yu, Kailin Lyu, et al. A survey of large audio language models: Generalization, trustworthiness, and outlook. arXiv preprint arXiv:2605.20266, 2026.

[33] Liangbo Ning, Wenqi Fan, and Qing Li. Exploring backdoor attack and defense for llmempowered recommendations. IEEE Transactions on Knowledge and Data Engineering, 2026.

[34] Liangbo Ning, Wenqi Fan, and Qing Li. Retrieval-augmented purifier for robust llm-empowered recommendation. ACM Transactions on Information Systems, 44(4):1–42, 2026.

[35] Liang-bo Ning, Shijie Wang, Wenqi Fan, Qing Li, Xin Xu, Hao Chen, and Feiran Huang. Cheatagent: Attacking llm-empowered recommender systems via llm agent. In Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pages 2284–2295, 2024.

[36] Aaron van den Oord, Yazhe Li, and Oriol Vinyals. Representation learning with contrastive predictive coding. arXiv preprint arXiv:1807.03748, 2018.

[37] Junwei Pan, Wei Xue, Ximei Wang, Haibin Yu, Xun Liu, Shijie Quan, Xueming Qiu, Dapeng Liu, Lei Xiao, and Jie Jiang. Ads recommendation in a collapsed and entangled world. In Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pages 5566–5577, 2024.

[38] Yijun Pan, Weikang Qiu, Qiyao Ma, Mingxuan Ju, Tong Zhao, Neil Shah, and Rex Ying. Flexrec: Adapting llm-based recommenders for flexible needs via reinforcement learning. arXiv preprint arXiv:2603.11901, 2026.

[39] Lianyong Qi, Jianye Xie, Chunhua Hu, Xiaolong Xu, Haolong Xiang, Haipeng Dai, Rong Gu, Xuyun Zhang, and Wanchun Dou. Knowledge-driven reasoning for compatible and interpretable api recommendation via teacher llm distillation. ACM Transactions on Information Systems, 44 (1):1–30, 2025.

[40] Haohao Qu, Wenqi Fan, Zihuai Zhao, and Qing Li. Tokenrec: learning to tokenize id for llm-based generative recommendation. arXiv preprint arXiv:2406.10450, 2024.

[41] Shashank Rajput, Nikhil Mehta, Anima Singh, Raghunandan Hulikal Keshavan, Trung Vu, Lukasz Heldt, Lichan Hong, Yi Tay, Vinh Tran, Jonah Samost, et al. Recommender systems with generative retrieval. Advances in Neural Information Processing Systems, 36:10299–10315, 2023.

[42] ZZ Ren, Zhihong Shao, Junxiao Song, Huajian Xin, Haocheng Wang, Wanjia Zhao, Liyue Zhang, Zhe Fu, Qihao Zhu, Dejian Yang, et al. Deepseek-prover-v2: Advancing formal mathematical reasoning via reinforcement learning for subgoal decomposition. arXiv preprint arXiv:2504.21801, 2025.

[43] Zhiyao Ren, Yibing Zhan, Siyuan Liang, Guozheng Ma, Baosheng Yu, and Dacheng Tao. Towards reliable medical llms: Benchmarking and enhancing confidence estimation of large language models in medical consultation. arXiv preprint arXiv:2601.15645, 2026.

[44] Haz Sameen Shahgir, Yufei Li, Frank Shyu, Luke Simon, Sandeep Pandey, Xi Liu, and Yue Dong. Scored: Student-aware cot optimization for recommendation distillation. arXiv preprint arXiv:2607.05734, 2026.

[45] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

[46] Jiaxi Tang and Ke Wang. Personalized top-n sequential recommendation via convolutional sequence embedding. In Proceedings of the eleventh ACM international conference on web search and data mining, pages 565–573, 2018.

[47] Jiakai Tang, Sunhao Dai, Teng Shi, Jun Xu, Xu Chen, Wen Chen, Jian Wu, and Yuning Jiang. Think before recommend: Unleashing the latent reasoning power for sequential recommendation. IEEE Transactions on Knowledge and Data Engineering, 2026.

[48] Gemma Team, Morgane Riviere, Shreya Pathak, Pier Giuseppe Sessa, Cassidy Hardin, Surya Bhupatiraju, Léonard Hussenot, Thomas Mesnard, Bobak Shahriari, Alexandre Ramé, et al. Gemma 2: Improving open language models at a practical size. arXiv preprint arXiv:2408.00118, 2024.

[49] Qwen Team. Qwen2.5: A party of foundation models, 2024.

[50] Georgios Theocharous, Philip S Thomas, and Mohammad Ghavamzadeh. Ad recommendation systems for life-time value optimization. In Proceedings ofthe 24th international conference on world wide web, pages 1305–1310, 2015.

[51] Alicia Tsai, Adam Kraft, Long Jin, Chenwei Cai, Anahita Hosseini, Taibai Xu, Zemin Zhang, Lichan Hong, Ed H Chi, and Xinyang Yi. Leveraging llm reasoning enhances personalized recommender systems. In Findings of the Association for Computational Linguistics: ACL 2024, pages 13176–13188, 2024.

[52] Xinfeng Wang, Jin Cui, Yoshimi Suzuki, and Fumiyo Fukumoto. Rdrec: Rationale distillation for llm-based recommendation. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), pages 65–74, 2024.

[53] Yuling Wang, Changxin Tian, Binbin Hu, Yanhua Yu, Ziqi Liu, Zhiqiang Zhang, Jun Zhou, Liang Pang, and Xiao Wang. Can small language models be good reasoners for sequential recommendation? In Proceedings ofthe ACM Web Conference 2024, pages 3876–3887, 2024.

[54] Fangzhao Wu, Ying Qiao, Jiun-Hung Chen, Chuhan Wu, Tao Qi, Jianxun Lian, Danyang Liu, Xing Xie, Jianfeng Gao, Winnie Wu, et al. Mind: A large-scale dataset for news recommendation. In Proceedings of the 58th annual meeting of the association for computational linguistics, pages 3597–3606, 2020.

[55] Runyang You, Yongqi Li, Xinyu Lin, Xin Zhang, Wenjie Wang, Wenjie Li, and Liqiang Nie. R<sup>2</sup>ec: Towards large recommender models with reasoning. Advances in Neural Information Processing Systems, 38:62376–62405, 2026.

[56] Junjie Zhang, Beichen Zhang, Wenqi Sun, Hongyu Lu, Wayne Xin Zhao, Yu Chen, and Ji-Rong Wen. Slow thinking for sequential recommendation. arXiv preprint arXiv:2504.09627, 2025.

[57] Xin Zhang, Yanzhao Zhang, Dingkun Long, Wen Xie, Ziqi Dai, Jialong Tang, Huan Lin, Baosong Yang, Pengjun Xie, Fei Huang, et al. mgte: Generalized long-context text representation and reranking models for multilingual text retrieval. arXiv preprint arXiv:2407.19669, 2024.

[58] Yang Zhang, Wenxin Xu, Xiaoyan Zhao, Wenjie Wang, Fuli Feng, Xiangnan He, and Tat-Seng Chua. Reinforced latent reasoning for llm-based recommendation. arXiv preprint arXiv:2505.19092, 2025.

[59] Siyan Zhao, Zhihui Xie, Mengchen Liu, Jing Huang, Guan Pang, Feiyu Chen, and Aditya Grover. Self-distilled reasoner: On-policy self-distillation for large language models. arXiv preprint arXiv:2601.18734, 2026.

[60] Zihuai Zhao, Wenqi Fan, Jiatong Li, Yunqing Liu, Xiaowei Mei, Yiqi Wang, Zhen Wen, Fei Wang, Xiangyu Zhao, Jiliang Tang, et al. Recommender systems in the era of large language models (llms). IEEE Transactions on Knowledge and Data Engineering, 36(11):6889–6907, 2024.

[61] Guanjie Zheng, Fuzheng Zhang, Zihan Zheng, Yang Xiang, Nicholas Jing Yuan, Xing Xie, and Zhenhui Li. Drn: A deep reinforcement learning framework for news recommendation. In Proceedings of the 2018 world wide web conference, pages 167–176, 2018.

[62] Zaiyi Zheng, Guanghui Min, Yaochen Zhu, Liang Wu, Liangjie Hong, Chen Chen, and Jundong Li. Sapo: Step-aligned policy optimization for reasoning-based generative recommendation. arXiv preprint arXiv:2605.17648, 2026.

[63] Mengdan Zhu, Yufan Zhao, Tao Di, Yulan Yan, and Liang Zhao. Learning user interests via reasoning and distillation for cross-domain news recommendation. arXiv preprint arXiv:2602.15005, 2026.

## A Whole Process of MUSE

Algorithm 1: Training and Inference Procedure of MUSE   
Input: Training dataset $\mathcal { D } ;$ teacher LLMs $\lbrace \mathcal { T } _ { m } \rbrace _ { m = 1 } ^ { M } ;$   
student reasoning generator ${ \mathcal { G } } _ { \theta } ;$ recommendation encoder $\mathcal { E } _ { \phi } .$   
Output: Trained reasoning generator $\mathcal { G } _ { \theta }$ and recommendation encoder $\mathcal { E } _ { \phi } .$   
1 // Training Phase;   
2 Generate teacher reasoning trajectories $\mathcal { C } ^ { \mathrm { t e a } }$ using all teacher LLMs;   
3 for epoch = 1, . . . , E do   
4 Generate $S _ { e }$ self-exploration reasoning trajectories using the current student according to   
Eq. (3);   
5 Construct the candidate reasoning pool $\mathcal { C } ^ { ( e ) } = \mathcal { C } ^ { \mathrm { t e a } } \cup \mathcal { C } ^ { \mathrm { s t u } }$   
6 Compute the trajectory utility score for all candidates and obtain $r ^ { + }$ and $r ^ { - }$ according to   
Eq. (8);   
7 Update the reasoning generator $\mathcal { G } _ { \theta }$ by minimizing $\mathcal { L } _ { \mathrm { g e n } }$ (Eq. (11));   
8 Update the recommendation encoder $\mathcal { E } _ { \phi }$ by minimizing $\mathcal { L } _ { \mathrm { r e c } }$ (Eq. (14));   
9 // Inference Phase;   
10 Generate the user preference reasoning ${ \hat { r } } = { \mathcal { G } } _ { \theta } ( x ) ;$   
11 Encode the user representation $\hat { \mathbf { z } } = \mathcal { E } _ { \phi } \big ( \big [ x ; \hat { r } \big ] \big ) ;$   
12 Rank all candidate items according to Eq. (15) and return the Top-K recommendation list;

## B Experiments

## B.1 Baselines

• Caser [46] models users’ sequential behaviors using convolutional neural networks (CNNs), capturing high-order sequential patterns for next-item recommendation.

• GRU4Rec [17] employs gated recurrent units (GRUs) to model users’ sequential interaction histories, making it one of the earliest RNN-based sequential recommendation methods.

• SASRec [23] adopts the Transformer self-attention mechanism to capture long-range dependencies within user interaction sequences for personalized recommendation.

• TIGER [41] reformulates recommendation as a generative retrieval task, directly generating item identifiers using a T5-based sequence-to-sequence framework.

• BigRec [4] introduces a bi-step grounding paradigm that first grounds user preferences and then performs recommendation with LLMs to improve recommendation accuracy.

$\mathbf { D } ^ { 3 }$ [3] mitigates amplification bias and recommendation homogeneity in LLM-based recommender systems through a decoding strategy that promotes recommendation diversity.

• RecSAVER [51] enhances recommendation by prompting LLMs to explicitly generate preference reasoning for knowledge distillation, improving interpretability and recommendation accuracy. We use DeepSeek-V3.2-Thinking as the fixed teacher to generate reasoning trajectories and GTE-largeen-v1.5 as the recommendation encoder for sequential recommendations.

• LangPTune [14] adopts Reinforcement Learning for System Optimization (RLSO), a representative reinforcement learning paradigm, to enhance the reasoning capability of LLM-based recommender systems.

• R<sup>2</sup>ec [55] incorporates the recently proposed Group Relative Policy Optimization (GRPO) [15] algorithm into LLM-empowered recommender systems to enhance user preference reasoning through reinforcement learning.

## B.2 Implementation

All experiments are conducted on 2 × NVIDIA H20 GPUs using the AdamW optimizer. We adopt component-specific learning rates of $3 \times 1 0 ^ { - 6 }$ for the recommendation encoder and $1 \times 1 0 ^ { - 5 }$ for the student reasoning generator. Training is performed for at most 50 epochs, with early stopping based on the validation performance with a patience of 6 epochs. For multi-teacher distillation, we employ three general-purpose teacher LLMs, namely DeepSeek-V3.2-Thinking, Gemini-3-Flash-Preview, and GPT-5-Mini, each of which generates one preference reasoning trajectory for every training user. All teacher trajectories are generated with a temperature of 0 and cached for reuse throughout training. The reasoning generator is fine-tuned using LoRA adapters [11] applied to all attention projections, feed-forward projections, and the language modeling head, while all parameters of the recommendation encoder are fully optimized. During training, the recommendation encoder is optimized using user representations constructed from the current student-generated reasoning trajectories, thereby avoiding the train-test discrepancy that would otherwise arise from using teacher generated reasoning during training but student-generated reasoning at inference time. The two components are trained in an alternating manner until convergence. The first two epochs serve as a warm-up stage, during which only the student reasoning generator is optimized via supervised fine-tuning on the best teacher trajectories, while the recommendation encoder remains frozen. After the warm-up stage, self-exploration is progressively activated. Specifically, we set $S _ { \mathrm { m a x } } = 6$ and $E _ { \mathrm { s } } = 1 0 $ , and increase the number of self-exploration trajectories $S _ { e }$ linearly from 0 to $S _ { \mathrm { m a x } }$ as training proceeds. For recommendation learning, we adopt a temperature of $\tau _ { r } = 0 .$ 1 and set the weight of the ConUtil to $\lambda _ { \mathrm { C o n U t i l } } = 0 . 5$ . For reasoning distillation, the weight of the ConDistill objective is set to $\lambda _ { \mathrm { C o n D i s t i l l } } = 0 . 5$ . We use a global batch size of 128, evenly partitioned across two GPUs for data-parallel training. All experiments are repeated three times, and the mean and standard deviation are reported to ensure the robustness and reliability of the results.

## B.3 Additional Experiments

Table 7 presents additional experimental results on the Video Games dataset. MUSE consistently achieves the best performance across all six metrics with both Qwen2.5-3B and Gemma-2-2B as the backbone. In particular, compared with the strongest baseline $\scriptstyle { \mathrm { R ^ { 2 } e c } }$ , MUSE improves H@10 from 0.0921 to 0.1020 under Qwen2.5-3B, corresponding to a relative improvement of 10.7%. It also improves H@20 from 0.1254 to 0.1313 under Gemma-2-2B. These results further confirm that the effectiveness of MUSE generalizes consistently across different datasets and student backbones.

Table 7: The overall performance of baselines and MUSE on the Video Games dataset.
<table><tr><td>Datasets</td><td colspan="6">Video Games</td></tr><tr><td>Methods</td><td>N@5</td><td>N@10</td><td>N@20</td><td>H@5</td><td>H@10</td><td>H@20</td></tr><tr><td colspan="7">Caser 0.0233±0.0006</td></tr><tr><td>GRU4Rec</td><td>0.0083±0.0028</td><td>0.0289±0.0002 0.0117±0.0041</td><td>0.0366±0.0003 0.0145±0.0048</td><td>0.0468±0.0014 0.0139±0.0055</td><td>0.0644±0.0010 0.0243±0.0092</td><td>0.0952±0.0009 0.0356±0.0121</td></tr><tr><td>SASRec</td><td>0.0171±0.0010</td><td>0.0229±0.0010</td><td>0.0289±0.0027</td><td>0.0280±0.0023</td><td>0.0462±0.0028</td><td>0.0702±0.0111</td></tr><tr><td>TIGER</td><td>0.0226±0.0007</td><td>0.0265±0.0005</td><td>0.0315±0.0007</td><td>0.0328±0.0027</td><td>0.0433±0.0016</td><td>0.0662±0.0031</td></tr><tr><td colspan="7">Qwen2.5-3B</td></tr><tr><td colspan="7"></td></tr><tr><td> $\mathrm { D ^ { 3 } }$  BigRec</td><td>0.0151±0.0006</td><td>0.0225±0.0013</td><td>0.0311±0.0059</td><td>0.0256±0.0023</td><td>0.0487±0.0067</td><td>0.0832±0.0246</td></tr><tr><td></td><td>0.0025±0.0010</td><td>0.0035±0.0005</td><td>0.0053±0.0005</td><td>0.0032±0.0005</td><td>0.0084±0.0010</td><td>0.0195±0.0050</td></tr><tr><td>RecSAVER</td><td>0.0305±0.0033</td><td>0.0382±0.0030</td><td>0.0457±0.0044</td><td>0.0465±0.0030</td><td>0.0705±0.0035</td><td>0.1005±0.0077</td></tr><tr><td>LangPTune</td><td>0.0271±0.0013</td><td>0.0360±0.0007</td><td>0.0448±0.0010</td><td>0.0422±0.0028</td><td>0.0699±0.0023</td><td>0.1050±0.0028</td></tr><tr><td>R2ec MUSE</td><td>0.0419±0.0002 0.0421±0.0005</td><td>0.0515±0.0017 0.0536±0.0006</td><td>0.0621±0.0013 0.0632±0.0009</td><td>0.0622±0.0023 0.0626±0.0006</td><td>0.0921±0.0023 0.1020±0.0023</td><td>0.1346±0.0019 0.1399±0.0040</td></tr><tr><td colspan="7">Gemma-2-2B</td></tr><tr><td colspan="7"></td></tr><tr><td>D3 BigRec</td><td>0.0286±0.0034</td><td>0.0383±0.0034</td><td>0.0513±0.0021</td><td>0.0416±0.0018</td><td>0.0718±0.0019</td><td>0.1229±0.0042</td></tr><tr><td></td><td>0.0131±0.0020</td><td>0.0155±0.0019</td><td>0.0189±0.0026</td><td>0.0160±0.0011</td><td>0.0241±0.0021</td><td>0.0354±0.0019</td></tr><tr><td>RecSAVER</td><td>0.0293±0.0031</td><td>0.0358±0.0031</td><td>0.0433±0.0019</td><td>0.0431±0.0014</td><td>0.0631±0.0014</td><td>0.0933±0.0076</td></tr><tr><td>LangPTune</td><td>0.0266±0.0020</td><td>0.0338±0.0021</td><td>0.0425±0.0016</td><td>0.0425±0.0009</td><td>0.0647±0.0019</td><td>0.0995±0.0028</td></tr><tr><td>R2ec</td><td>0.0380±0.0016</td><td>0.0483±0.0010</td><td>0.0574±0.0015</td><td>0.0557±0.0022</td><td>0.0890±0.0023</td><td>0.1254±0.0043</td></tr><tr><td>MUSE</td><td>0.0387±0.0010</td><td>0.0492±0.0015</td><td>0.0595±0.0027</td><td>0.0573±0.0016</td><td>0.0902±0.0019</td><td>0.1313±0.0074</td></tr></table>

## B.4 Additional Ablation Study

Table 8 reports the additional ablation results on the Video Games dataset. The complete MUSE consistently outperforms all single-teacher and component-removed variants across every evaluation metric. For example, replacing RGUA with random trajectory selection also decreases H@10 from 0.1020 to 0.0844, demonstrating the importance of selecting reasoning trajectories according to their downstream recommendation utility. Moreover, the performance degradation caused by removing self-exploration, ConDistill or ConUtil further confirms that all proposed components contribute positively to the final recommendation performance.

Table 8: Ablation study on key components (Backbone: Qwen2.5-3B; Dataset: Video Games).
<table><tr><td>Datasets</td><td colspan="6">Video Games</td></tr><tr><td>Methods</td><td>N@5</td><td>N@10</td><td>N@20</td><td>H@5</td><td>H@10</td><td>H@20</td></tr><tr><td colspan="7">GPT</td></tr><tr><td rowspan="3">Gemini DeepSeek</td><td>0.0395±0.0005</td><td>0.0482±0.0007</td><td>0.0589±0.0019</td><td>0.0598±0.0021</td><td>0.0866±0.0010</td><td>0.1294±0.0097</td></tr><tr><td>0.0400±0.0008</td><td>0.0491±0.0029</td><td>0.0564±0.0043</td><td>0.0548±0.0021</td><td>0.0832±0.0066</td><td>0.1137±0.0145</td></tr><tr><td>0.0309±0.0053</td><td>0.0390±0.0056</td><td>0.0493±0.0057</td><td>0.0481±0.0065</td><td>0.0736±0.0076</td><td>0.1146±0.0080</td></tr><tr><td colspan="7">w/o SE</td></tr><tr><td rowspan="3">Rand</td><td>0.0413±0.0006</td><td>0.0527±0.0006</td><td>0.0623±0.0007</td><td>0.0609±0.0008</td><td>0.0998±0.0021</td><td>0.1343±0.0030</td></tr><tr><td>0.0376±0.0013</td><td>0.0457±0.0016</td><td>0.0567±0.0013</td><td>0.0591±0.0024</td><td>0.0844±0.0038</td><td>0.1284±0.0024</td></tr><tr><td>0.0332±0.0007</td><td>0.0434±0.0016</td><td>0.0521±0.0015</td><td>0.0530±0.0019</td><td>0.0847±0.0033</td><td>0.1195±0.0044</td></tr><tr><td rowspan="2">w/o ConDistill w/o ConUtil</td><td></td><td></td><td>0.0602±0.0030</td><td>0.0596±0.0020</td><td></td><td></td></tr><tr><td>0.0390±0.0020 0.0421±0.0005</td><td>0.0496±0.0024 0.0536±0.0006</td><td>0.0632±0.0009</td><td>0.0626±0.0006</td><td>0.0915±0.0009 0.1020±0.0023</td><td>0.1334±0.0028 0.1399±0.0040</td></tr></table>

## C Used Prompts

The prompt used to guide the reasoning generator in producing user preference reasoning is shown in Prompt 1. Specifically, {category} denotes the product category (e.g., music, games, or musical instruments), while {items} contains the textual metadata of the items previously interacted with by the user, including the title, description, and categories. Here, description refers to a concise summary of each item. After the reasoning is generated, we append the generated reasoning to the prompt in the form of "User preference summary: {Reasoning}", and feed the resulting prompt into the recommendation encoder to generate the final recommendation.

Prompt 1: Instruction for the reasoning generator G<sub>θ</sub>

The user has rated (out of 5.0) the following {category} products arranged chronologically from the oldest (top) to the newest (bottom). Please provide a high-level summary of the user preference in detail. Here are the items the user interacted with and the rating from the user: {items}

## D Case Study

To qualitatively evaluate the effectiveness of MUSE, we present some representative examples in Example 1 and 2. As shown, MUSE consistently generates accurate and interpretable preference reasoning by identifying users’ high-level interests from historical interactions, rather than merely summarizing individual items. For example, it successfully captures preferences for premium gaming peripherals (Example 1) and the PlayStation 2 ecosystem and classic open-world games (Example 2). These reasoning trajectories effectively characterize the users’ underlying preferences and provide informative guidance for the recommendation encoder, thereby facilitating correct recommendations. These qualitative examples further demonstrate that MUSE learns accurate reasoning that is both interpretable and beneficial for downstream recommendation.

## E Related Work

SFT distillation-based methods. Motivated by the strong reasoning capabilities of LLMs, a growing body of research has explored transferring such capabilities to recommender systems through knowledge distillation for enhanced user preference modeling [53, 51, 52, 24, 39, 63, 21, 12, 2, 44, 31]. For example, SLIM [53] transfers the reasoning ability of a large LLM into a much smaller recommendation model through rationale distillation, so that the student can perform recommendations with explicit reasoning while remaining computationally efficient. Exp3rt [24] trains the student to sequentially perform preference extraction, user/item profile construction, and step-by-step reasoning for rating prediction, enabling the model to leverage rich semantic information from reviews while producing interpretable recommendation rationales. The most closely related study is IRMD [31], which synthesizes CoT trajectories and progressively distills them into the student’s hidden states to enable implicit reasoning without generating rationales at inference. Despite this connection, IRMD differs fundamentally from MUSE. It filters reasoning using teacher selfassessed correctness and coherence rather than its actual utility for downstream recommendation, and uses only teacher-generated reasoning as supervision without addressing the teacher knowledge

## Example 1

## Recommendation Query:

The user has rated the following Video Games products arranged chronologically from the oldest to the newest. Please provide a high-level summary of the user preference in detail. Here are the items the user interacted with and the rating from the user:

— 1. [EASARS Wireless Gaming Headset] ( Rating: 4.0 )

— 2. [Smart Duck Keyboard] ( Rating: 5.0 )

— 3. [AULA Mechanical Gaming Keyboard] ( Rating: 5.0 )

## Preference Reasoning:

The user strongly prefers premium gaming peripherals, particularly high-quality mechanical keyboards with RGB lighting, hot-swappable switches, and strong typing performance. They also value immersive gaming audio, but are more tolerant of headsets than keyboards, indicating higher expectations for keyboard quality and customization.

## Example 2

## Recommendation Query:

The user has rated the following Video Games products arranged chronologically from the oldest to the newest. Please provide a high-level summary of the user preference in detail. Here are the items the user interacted with and the rating from the user:

— 1. [Grand Theft Auto 3] ( Rating: 5.0 )

— 2. [Grand Theft Auto Vice City] ( Rating: 5.0 )

— 3. [PlayStation 2 Slim Console (Renewed)] ( Rating: 4.0 )

## Preference Reasoning:

The user shows a strong preference for PlayStation 2 titles, especially classic open-world action games. They enjoy immersive single-player experiences and demonstrate nostalgia toward the PS2 ecosystem, while also maintaining interest in PS2 hardware.

ceiling. Moreover, its implicit distillation aligns the teacher’s and student’s final hidden states, requiring white-box access that is typically unavailable for commercial LLMs. Finally, although termed multi-teacher, its implementation uses only Qwen2.5-14B under three temperatures, whereas MUSE leverages heterogeneous teacher LLMs and progressive student self-exploration.

Reinforcement learning (RL)-based methods. With the remarkable success of reinforcement learning in enhancing the reasoning capabilities of LLMs across a wide range of complex tasks, an increasing number of studies have begun to leverage reinforcement learning algorithms to improve the reasoning ability of LLM-based recommender systems [20, 55, 14, 58, 47, 10, 22, 38, 62, 13]. For example, LangPTune [14] proposes an end-to-end framework that optimizes LLM-generated languagebased user profiles for recommendation. Instead of relying on supervised reasoning distillation, it employs reinforcement learning to optimize the profile generator using downstream recommendation performance as the reward signal, while jointly training the ranking model through contrastive learning to improve recommendation accuracy and profile quality. R<sup>2</sup>ec [55] integrates reasoning generation and item prediction into a unified large recommender model, enabling end-to-end optimization of both tasks. It employs an RL-based optimization framework (RecPO) to jointly optimize reasoning and recommendation using a fused reward derived from recommendation supervision, enabling end-to-end training without human reasoning annotations. ReRec [20] introduces fine-grained reward shaping together with reasoning-aware advantage estimation to provide more informative supervision for both recommendation outcomes and intermediate reasoning steps during reinforcement fine-tuning.

Hybrid methods. Beyond these approaches, several studies adopt hybrid training paradigms that integrate supervised fine-tuning and reinforcement learning to further improve the reasoning capabilities of LLM-empowered recommender systems [56, 19, 29, 7]. RecOne [25] adopts a hybrid SFT+RL paradigm for reasoning-enhanced recommendation. It first initializes the LLM through supervised fine-tuning on a small set of high-quality reasoning samples, and subsequently improves its reasoning capabilities via reinforcement learning with recommendation-specific rewards. GR2 [29] further applies the hybrid training strategy to recommendation re-ranking, where the model is first supervised using high-quality hierarchical reasoning traces generated by a teacher LLM and then refined through a re-ranking-oriented reinforcement learning objective.