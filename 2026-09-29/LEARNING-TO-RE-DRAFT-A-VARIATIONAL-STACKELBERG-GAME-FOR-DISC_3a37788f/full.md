# LEARNING TO RE-DRAFT: A VARIATIONAL STACKELBERG GAME FOR DISCRETE DIFFUSION

Dmitrii Moor<sup>1</sup> Federico Tomasi<sup>1</sup> Paul N. Bennett<sup>2</sup> Alice Wang<sup>3</sup> Mounia Lalmas<sup>1</sup>

<sup>1</sup>Spotify, London, UK <sup>2</sup>Spotify, Boston, USA <sup>3</sup>Spotify, New York, USA {dmitriim,federicot,pbennett,alicew,mounial}@spotify.com

## ABSTRACT

Discrete diffusion models offer the ability to re-draft, revisiting and correcting earlier tokens throughout generation. This capability depends on the forward corruption process that defines what the denoiser learns to correct. Masked diffusion models fix tokens once they are unmasked, while uniform diffusion permits revisions but relies on uniformly random token substitutions. We instead learn which substitutions are most useful for training the denoiser to re-draft. We introduce Variational Stackelberg Discrete Diffusion (VSDD), a framework for learning a semantically aware corruption process. VSDD formulates training as a leaderfollower game: the leader defines a Markovian corruption process parameterized by the denoiser’s token embeddings, while the follower optimizes a variational denoising objective with the corruption process held fixed. The leader rewards corruptions based on how much the denoiser improves after learning from them, rather than on how easily the current denoiser can reconstruct them. We measure this improvement under a fixed reference corruption process, approximate the follower’s response with a one-step gradient update, and optimize the leader using a score-function estimator. We evaluate VSDD across molecular, text, and playlist generation. VSDD substantially improves molecular validity over uniform and masked diffusion, reduces text perplexity relative to uniform diffusion while remaining competitive with masked diffusion, and achieves sizable improvements in offline playlist recommendation metrics.

## 1 INTRODUCTION

Discrete diffusion models (Austin et al., 2021; Hoogeboom et al., 2021) have emerged as a powerful paradigm for generating sequences of discrete tokens. These models learn to reverse a stochastic corruption process that gradually transforms a clean sequence into noise. Unlike autoregressive models, which generate tokens sequentially from left to right and can suffer from error accumulation (Bengio et al., 2015; Ranzato et al., 2015), diffusion models operate over the entire generated sequence and can iteratively revise earlier predictions during denoising. This ability to re-draft is particularly valuable for sequences with strong structural dependencies, such as molecular sequences.

The performance of discrete diffusion models depends on two closely coupled components: the denoiser, which predicts the clean sequence from a noisy observation, and theforward noise process, which determines how clean tokens are perturbed during training. While significant research effort has focused on denoiser architectures (Sahoo et al., 2024; Lou et al., 2024; Shi et al., 2024), the forward process is typically fixed to a simple predefined distribution, such as uniform or absorbing noise (Austin et al., 2021). Yet the choice of the forward corruption process can substantially affect learning efficiency and generation quality.

For instance, Sahoo et al. (2024) showed that absorbing noise can substantially outperform the uniform diffusion (Austin et al., 2021) for text generation, suggesting that how tokens are corrupted can be as important as how they are reconstructed. Similarly, Nichol and Dhariwal (2021) showed that a non-linear noise schedule can further improve generation quality. These observations motivate a natural question: can we learn a forward noise process that adapts to the data distribution and the denoiser’s capabilities?

Learning the forward corruption process, however, introduces a fundamental optimization challenge. Recent work takes a step in this direction by jointly optimizing the forward process and the denoiser through a score function based surrogate loss (Bartosh et al. 2026). With such joint optimization, the forward process may improve the training objective by compensating an underperforming denoiser rather than providing corruptions that help the denoiser improve. In particular, it may favor trivial corruptions that are easy to reconstruct, reducing the KL-loss without improving the resulting generative model. The challenge, however, is to learn corruptions based on how much the denoiser benefits from training on them, rather than how well the current denoiser can reconstruct them.

To address this challenge, we introduce Variational Stackelberg Discrete Diffusion (VSDD), which formulates training as a variational Stackelberg game. Instead of jointly optimizing the forward process and the denoiser with a shared objective, the leader selects the corruption process while anticipating the follower’s response, and the follower optimizes the variational objective of the denoiser with the corruption process fixed. This separation discourages the forward process from reducing the training loss by adapting to the current denoiser. Instead, the leader is rewarded for selecting corruptions that improve the denoiser after the follower has learned from them.

The posterior of the learned forward process affects how the model resamples, or re-drafts, tokens at inference time. Effective re-drafting therefore requires capturing relationships between tokens. Uniform noise treats all substitutions equally, regardless of the structure learned by the denoiser. In contrast, re-drafting an almost complete sequence may require distinguishing between plausible, closely related alternatives. We therefore parameterize the forward process using token embeddings learned by the denoiser. These embeddings capture relationships between tokens, while the leader learns how those relationships should shape the noise process. Related tokens can thus receive higher or lower transition probabilities depending on which substitutions provide useful training signals for the denoiser. We refer to this as a semantically aware noise process.

To optimize the forward noise process, the leader must anticipate the follower’s response. Computing the best response would require fully optimizing the denoiser for each candidate noise process, which is prohibitively expensive. We therefore approximate it with a better response using a single virtual gradient update on samples from the proposed noise process. We measure how much this update improves the denoiser under a fixed reference noise process and use this improvement as the leader’s reward. The leader then optimizes the noise process using a score function estimator.

We evaluate our approach across three structurally different domains: molecular, text, and playlist generation. Across these domains, VSDD improves the primary generation quality metrics over a number of diffusion baselines with and without learnable noise.

Our main contributions are:

1. We formulate learning the forward noise process and the denoiser as a variational Stackelberg game, where corruptions are optimized based on how much the denoiser improves after learning from them rather than how well the current denoiser can reconstruct them.

2. We introduce a semantically aware noise process parameterized by token embeddings learned by the denoiser, allowing the forward process to adapt to the learned structure of the domain.

3. We design an efficient training algorithm that approximates the follower’s best response with a single virtual gradient step and optimizes the noise process via a score function estimator.

4. We show improved generation and re-drafting across molecular, text and playlist domains.

## 2 RELATED WORK

Discrete diffusion models. Sohl-Dickstein et al. (2015) introduced diffusion based generative models for continuous data, later developed through DDPM (Ho et al., 2020) and score based generative modeling (Song et al., 2020). Austin et al. (2021) extended diffusion to discrete state spaces with D3PM, where the forward process is defined through categorical transition matrices, including uniform and absorbing corruption. Hoogeboom et al. (2021) proposed multinomial diffusion with argmax flows. More recent work focused on masked diffusion (Sahoo et al., 2024; Shi et al., 2024) and score entropy based methods (Lou et al., 2024). In particular, Sahoo et al. (2024) showed that masked diffusion with absorbing noise can substantially outperform uniform corruption for text generation. These results demonstrate that the choice of corruption process is consequential, but existing approaches typically prescribe this process in advance. Our work instead learns the structure of the corruption process according to how useful different corruptions are for training the denoiser.

Learned noise schedules and forward processes. In continuous diffusion, several works learn the noise schedule. Nichol and Dhariwal (2021) learn a variance schedule through the variational bound, while Kingma et al. (2021) parameterize the signal to noise ratio via a neural network. Dieleman et al. (2022) show that the forward process in continuous diffusion is equivalent up to reparameterization, implying that learning the schedule affects the weighting of the training objective rather than the underlying generative model. In discrete diffusion, however, different noise processes change what the denoiser is trained to reconstruct. Learning this forward process in discrete diffusion is thus a modeling decision rather than only a training convenience.

For discrete diffusion, Bartosh et al. (2026) recently proposed learning the forward process by jointly optimizing the noise model and the denoiser via a score function based surrogate objective. This approach presents two challenges: the score function gradient can have high variance, and joint optimization may allow the forward process to reduce the training objective by adapting to the current denoiser rather than by providing useful training corruptions. Our work differs in how the forward process is optimized: VSDD evaluates corruptions based on how much the denoiser improves after learning from them, rather than how well they suit the current denoiser. To reduce the variance of the score function estimator, we evaluate multiple corruption samples and use a leave one out baseline.

Game-theoretic concepts in machine learning. Stackelberg games model sequential interactions where a leader chooses a strategy while anticipating a follower’s response (Conitzer and Sandholm, 2006). Stackelberg formulations have been studied in machine learning, including analyses of GANs (Goodfellow et al., 2014; Fiez et al., 2020). Our approach is related to MAML style meta learning (Finn et al., 2017), where outer optimization evaluates its objective after inner loop adaptation. Similarly, in VSDD the leader evaluates a corruption process via the denoiser’s response after adapting to corruptions sampled from that process. Unlike these settings, the leader in VSDD parameterizes the forward process itself, learning which corruptions provide useful training signals for the denoiser.

## 3 NOTATION AND PRELIMINARIES

We let V be the vocabulary size, and let $\bar { \alpha } _ { t } \in [ 0 , 1 ]$ be the cumulative retention schedule. Following standard notation in diffusion modeling (Austin et al., 2021), we let $\alpha _ { t } = \bar { \alpha } _ { t } / \bar { \alpha } _ { t - 1 }$ be the per-step retention rates and $\beta _ { t } = 1 - \alpha _ { t }$ be the respective noise rates. Let $x _ { t } = ( x _ { t } ^ { 1 } , \ldots \cdot , x _ { t } ^ { L } )$ be the sequence of L tokens at time $t ,$ where each $x _ { t } ^ { \ell } \in \bar { \{ 1 , \ldots , V \} }$ is represented as a one-hot vector. One token is a PAD token that is never corrupted. We let $x _ { 0 }$ be the clean (uncorrupted) sequence.

In standard discrete diffusion, the forward noise process is fixed in advance: one specifies the retention schedule ${ { \bar { \alpha } } _ { t } } .$ , constructs the per-step token transition matrices $Q _ { t } \in \mathbf { R } ^ { V \times V }$ , and only trains the denoiser $p _ { \theta } ( x _ { 0 } | x _ { t } , t )$ against this fixed noise process $Q _ { t }$ (Austin et al., 2021). To this end, we let $\begin{array} { r } { \bar { Q } _ { t } = \prod _ { \tau = 1 } ^ { t } Q _ { i } } \end{array}$ be the cumulative transition matrix after t steps. The forward marginal probability distribution used to sample a noisy sequence $x _ { t }$ is:

$$
q ( x _ { t } \mid x _ { 0 } ) = \mathrm { C a t } \big ( x _ { t } \mid \bar { Q } _ { t } ^ { \top } x _ { 0 } \big ) .\tag{1}
$$

In this setting, the denoiser $p _ { \theta } ( x _ { 0 } | x _ { t } , t )$ is parameterized by a transformer with learnable parameters θ. At inference, it takes a corrupted sequence $x _ { t } .$ , the timestep t and predicts the clean sequence ${ \hat { x } } _ { 0 }$

We let $E _ { \theta } \in \mathbb { R } ^ { V \times d }$ denote the token embedding matrix learned as part of the denoiser $p _ { \theta } ( x _ { 0 } | x _ { t } , t )$ with one d-dimensional embedding for each of the $V$ vocabulary tokens.<sup>1</sup> Finally, we use the standard ELBO loss $\mathcal { L }$ that includes the per-step KL terms $\mathcal { L } _ { \mathrm { K L } } ( t )$ (for $t > 1 )$ and the auxiliary reconstruction cross-entropy $\mathcal { L } _ { \mathrm { C E } }$ , over non-PAD positions:

$$
\mathcal { L } = \sum _ { t = 2 } ^ { T } \underbrace { D _ { \mathrm { K L } } \big ( q ( x _ { t - 1 } \mid x _ { t } , x _ { 0 } ) \mid \big | p _ { \theta } ( x _ { t - 1 } \mid x _ { t } ) \big ) } _ { \mathcal { L } _ { \mathrm { K L } } ( t ) } \underbrace { - \log p _ { \theta } ( x _ { 0 } \mid x _ { 1 } ) } _ { \mathcal { L } _ { \mathrm { C E } } } .\tag{2}
$$

Here, $q ( x _ { t - 1 } | x _ { t } , x _ { 0 } )$ is the true reverse posterior induced by the forward process, and $p _ { \theta } ( x _ { t - 1 } | x _ { t } )$ is the corresponding learned posterior. In Section 5, we show how to compute these posteriors.

## 4 SEMANTIC-AWARE NOISE PROCESS

As motivated in Section 1, effective re-drafting may need to distinguish between plausible, closely related tokens rather than treating all substitutions equally. We therefore construct a semantically aware noise whose transitions depend on the token representations learned by the denoiser. It provides the structure between tokens that we leverage to learn the forward process in Section 5.

To parameterize the forward noise process by the token representations we let

$$
Q _ { t , \phi } = \alpha _ { t } I + ( 1 - \alpha _ { t } ) M _ { t , \phi } , \quad t = 1 , . . . , T ,\tag{3}
$$

where I is the identity matrix of rank $V ,$ and $M _ { t , \phi }$ is the learnable transition matrix parameterized by $\phi .$ Thus, at each step t, a token is retained with probability $\alpha _ { t } .$ , while with probability $1 - \alpha _ { t }$ it is replaced according to $M _ { t , \phi }$

To construct $M _ { t , \phi } ,$ we define $s _ { t , \phi } ( i , j ) = e _ { i } A _ { t , \phi } e _ { j } ^ { T }$ as a learned similarity score between tokens i and $j .$ Here, $e _ { i } , e _ { j } \in \mathbf { R } ^ { d }$ are L2-normalized ith and jth rows from $E _ { \theta }$ (Section 3), and $A _ { t , \phi } \in \mathbb { R } ^ { d \times d }$ learns how relationships in the denoiser’s embedding space should influence corruption at time t. In other words, the learned embeddings $e _ { i } , e _ { j }$ provide the underlying token structure, while $A _ { t , \phi }$ learns which aspects of this structure should determine likely substitutions. For all non-pad tokens we define

$$
M _ { t , \phi } [ i , j ] = \mathbf { 1 } _ { \{ i \neq j \} } \frac { \exp { s _ { t , \phi } ( i , j ) } } { \sum _ { \ell \neq i } \exp { s _ { t , \phi } ( i , \ell ) } } ,\tag{4}
$$

which ensures that $M _ { t , \phi }$ and $Q _ { t , \phi }$ are row stochastic, meaning that every row defines a valid probability distribution over the next token. We let $\begin{array} { r } { \bar { Q } _ { t , \phi } = \prod _ { \tau = 1 } ^ { t } Q _ { t , \phi } } \end{array}$ denote the corresponding cumulative transition matrix. Consequently, we let the forward marginal be $q _ { \phi } ( x _ { t } | x _ { 0 } ) = \mathrm { C a t } \bigl ( x _ { t } ; \bar { Q } _ { t , \phi } ^ { T } x _ { 0 } \bigr )$ and its corresponding reverse posterior be $q _ { \phi } \big ( x _ { t - 1 } \big | x _ { t } , x _ { 0 } \big )$

A direct approach would be to add ϕ to the diffusion loss (Equation (2)) and to jointly optimize the forward process and the denoiser. However, this can lead to learning a degenerate noise process and a poor denoiser. To see this, observe that the KL term in Equation (2) can decrease in two ways: (1) by improving the denoiser $p _ { \theta }$ to better match the true posterior, or (2) by simplifying $q _ { \phi }$ so that the posterior becomes trivial to match. Joint optimization permits the latter path: the forward process can adapt its off-diagonal transitions to the current denoiser, increasing the probabilities of substitutions that are already easy for $p _ { \theta } .$ , simplifying the posterior target. The denoising loss may therefore decrease without an improvement in generation quality. In Section 5, we address this by optimizing the noise process based on how the denoiser improves after learning from its corruptions.

## 5 VARIATIONAL STACKELBERG GAME

To prevent the degenerate solutions discussed above, we formulate the optimization problem from a game theoretic perspective. In particular, we consider two players, a leader and afollower, interacting in a Stackelberg game fashion (Conitzer and Sandholm, 2006). The leader chooses an action ϕ that parameterizes the forward corruption process $Q _ { t , \phi } ,$ , while the follower observes this process and chooses an action $\theta$ corresponding to the parameters of the denoiser $p _ { \theta } ( x _ { 0 } | x _ { t } , t )$ . Thus, the leader determines how training sequences are corrupted, while the follower learns to denoise sequences generated by that process. We now elaborate on the objectives of the follower and the leader and show how their interaction is used to jointly learn ϕ and θ.

Follower. The follower optimizes the standard diffusion loss from Equation (2) under the noise process selected by the leader. Importantly, as is common in Stackelberg games we assume that the follower cannot directly affect the action of the leader (Conitzer and Sandholm, 2006). Therefore, when choosing its action θ the follower samples noisy sequences $x _ { t }$ from $q _ { \mathrm { s g } , \phi } ( x _ { t } | x _ { 0 } ) =$ Ca $\mathbf { \Psi } ( x _ { t } | \operatorname { s g } \{ \bar { Q } _ { t , \phi } \} ^ { T } x _ { 0 } )$ , where $\mathrm { s g } \{ . \}$ is the stop-gradient operator. This allows the follower to train on the corruption process selected by the leader while preventing gradients from the follower’s objective from propagating to the leader parameters $\phi .$ Consequently, the follower’s loss can be obtained from Equation (2) as follows:

$$
\begin{array} { r } { \mathcal { L } ^ { F } ( \theta | \phi ) = ( T - 1 ) D _ { K L } \big ( q _ { \mathrm { s g } , \phi } ( x _ { t - 1 } | x _ { t } , x _ { 0 } ) | | p _ { \theta } ( x _ { t - 1 } | x _ { t } ) \big ) - \log p _ { \theta } ( x _ { 0 } | x _ { 1 } ) , } \end{array}\tag{5}
$$

Here, $q _ { \mathrm { s g } , \phi } ( x _ { t - 1 } | x _ { t } , x _ { 0 } )$ is the reverse posterior corresponding to $q _ { \mathrm { s g } , \phi } ( x _ { t } | \boldsymbol { x } _ { 0 } )$ , and $p _ { \theta } ( x _ { t - 1 } | x _ { t } )$ is the denoiser’s posterior, computed as

$$
p _ { \theta } ( x _ { t - 1 } | x _ { t } ) = \sum _ { \hat { x } _ { 0 } } p _ { \theta } ( \hat { x } _ { 0 } | x _ { t } , t ) q _ { \mathrm { s g } , \phi } ( x _ { t - 1 } | x _ { t } , \hat { x } _ { 0 } ) ,\tag{6}
$$

Following Austin et al. (2021), this combines the denoiser’s prediction over the clean sequence $\scriptstyle { \hat { x } } _ { 0 }$ with the reverse posterior of the fixed corruption process to obtain the distribution over the previous diffusion state $x _ { t - 1 }$ . Thus, the follower optimizes the standard discrete diffusion loss with the corruption process $Q _ { t , \phi }$ held fixed.

Leader. The leader’s goal is to find a corruption process $Q _ { t , \phi }$ such that, after the follower optimizes the denoiser against this process (via Equation (5)), the quality of the generated sequences would be maximal. To this end, the leader needs to: (1) assess whether a candidate strategy ϕ improves the log-likelihood of cleaned data log $p ( \hat { x } _ { 0 } )$ , and (2) determine how to update $\phi$ while accounting for the follower’s subsequent response θ.

To address the first problem, we would ideally evaluate the resulting denoiser through the log likelihood of clean data, but optimizing this quantity directly is intractable. We therefore use an ELBO objective as a tractable surrogate. To ensure that different leader strategies are evaluated against the same corruption distribution, we introduce a fixed reference noise process $Q _ { \mathrm { r e f } }$ , and we let $q _ { \mathrm { r e f } } ( x _ { t } | x _ { 0 } )$ be its corresponding forward marginal distribution.<sup>2</sup> Using a fixed reference process provides a common evaluation distribution for comparing different leader strategies.

While the reference process determines the noise distribution used for evaluation, the current leader strategy ϕ determines the reverse posterior $q _ { \phi } ( x _ { t - 1 } | x _ { t } , x _ { 0 } )$ used to construct the reverse transition probabilities. As in the follower’s case, we assume that the leader cannot affect the follower’s action θ directly, and therefore, we freeze the gradient flow via the follower’s embeddings, i.e., sg{E<sub>θ</sub>} when computing $\operatorname { s g } _ { \theta } \{ q _ { \phi } \}$ . The leader’s reference ELBO objective results in

$$
\begin{array} { r } { \mathcal { L } _ { \mathtt { E L B O } } ^ { L } ( \phi | \theta ) = ( T - 1 ) D _ { K L } \big ( q _ { \mathrm { r e f } } ( x _ { t - 1 } | x _ { t } , x _ { 0 } ) | | p _ { \theta , \phi } ( x _ { t - 1 } | x _ { t } ) \big ) - \log p _ { \theta } ( x _ { 0 } | x _ { 1 } ) , } \end{array}\tag{7}
$$

where

$$
p _ { \theta , \phi } ( x _ { t - 1 } | x _ { t } ) = \sum _ { \hat { x } _ { 0 } } p _ { \theta } \big ( \hat { x } _ { 0 } | x _ { t } \big ) \mathrm { s g } _ { \theta } \big \{ q _ { \phi } ( x _ { t - 1 } | x _ { t } , \hat { x } _ { 0 } ) \big \} .\tag{8}
$$

The fixed reference process is important: the leader is evaluated against the same reference distribution as $\phi$ changes, rather than against an evaluation distribution that changes with its own strategy.

In a Stackelberg game, the leader anticipates the follower’s best response to its strategy (Conitzer and Sandholm, 2006). In our setting, computing the exact best response would require fully optimizing the denoiser for each candidate corruption process, which is prohibitively expensive. We therefore use a single gradient update as an approximate better response. To this end, we sample a validation minibatch $x _ { 0 }$ and inject K different noise realizations into it to create $K$ new minibatches $x _ { t } ^ { ( k ) }$ $k = 1 , \ldots , K$ . We let $d \theta _ { k } = - \gamma \nabla _ { \theta } \mathcal { L } ^ { F } ( \theta ; x _ { t } ^ { ( k ) } )$ denote the resulting virtual follower update for the kth validation minibatch, where $\gamma$ is a step size hyperparameter. Thus, $\theta + d \theta _ { k }$ approximates how the follower would change after learning from that particular corruption realization.

We then compute the reward for that noise realization as the relative improvement in the leader’s reference loss:

$$
R _ { k } = \frac { \mathcal { L } _ { \mathrm { E L B O } } ^ { L } ( \phi | \theta ) - \mathcal { L } _ { \mathrm { E L B O } } ^ { L } ( \phi | \theta + d \theta _ { k } ) } { \gamma | \mathcal { L } _ { \mathrm { E L B O } } ^ { L } ( \phi | \theta ) | } .\tag{9}
$$

Algorithm 1 Variational Stackelberg Training for Discrete Diffusion   
Require: Reference process $q _ { \mathrm { r e f } } ,$ block size N, number of probes K; learning rates $\eta _ { \phi } , \eta _ { \theta } ; \gamma , \lambda _ { T }$   
1: Initialize denoiser $p _ { \theta } ( x _ { 0 } | x _ { t } , t )$ and noise kernel $Q _ { \phi }$   
2: for epoch $= 1 , \hdots \bar { M }$ do   
3: block\_steps $ 0$   
4: for each minibatch $x _ { 0 }$ do   
5: if block\_steps = 0 then   
6: Build $Q _ { \phi } ^ { - } , \bar { Q } _ { \phi }$   
7: Follower step: $\theta  \theta - \eta _ { \theta } \nabla _ { \theta } \mathcal { L } ^ { F } ( \theta ; \operatorname { s g } \{ Q _ { \phi } \} )$ {Eq. (5)}   
8: block\_steps ← block\_steps + 1   
9: if block\_steps = N then   
10: Sample a validation batch $x _ { 0 } ^ { \mathrm { v a l . } }$ compute $\mathcal { L } _ { \mathrm { E L B O } } ^ { L } ( \phi | \theta )$ via Eq. (7)   
11: for $\dot { k } = 1 , \dots , K$ do   
12: $x _ { t } ^ { ( k ) } \gets \mathrm { C o r r u p t } x _ { 0 } ^ { \mathrm { v a l } }$ under $Q _ { \phi }$   
13: Follower’s better response: $\theta _ { b r }  \theta - \gamma \nabla _ { \theta } \mathcal { L } ^ { F } ( \theta ; x _ { t } ^ { ( k ) } )$   
14: Reward: $\begin{array} { r } { R _ { k }  \frac { \mathcal { L } _ { \mathrm { E L B O } } ^ { L } ( \phi \vert \theta ) - \mathcal { L } _ { \mathrm { E L B O } } ^ { L } ( \phi \vert \theta _ { b r } ) } { \gamma \vert \mathcal { L } _ { \mathrm { E L B O } } ^ { L } ( \phi \vert \theta ) \vert } } \end{array}$   
15: Compute normalized rewards $\tilde { R } _ { k }$ via Eq. 10   
16: $\mathcal { L } ^ { L } ( \dot { \phi } )  \mathcal { L } ^ { L } ( \phi ) \ + \ \lambda _ { T } D _ { K L } [ \bar { Q } _ { T , \phi } \vert \vert \dot { \mathcal { U } } ]$ {Eq. 11 + terminal regularization}   
17: Leader step: $\phi  \phi - \eta _ { \phi } \nabla _ { \phi } \mathcal { L } ^ { L } ( \phi )$ {Eq. (11)}   
18: block\_steps ← 0

A positive reward therefore indicates that training on the sampled noise improves the follower under the fixed reference objective. This evaluates a corruption by the improvement it induces in the denoiser, rather than by how easily the current denoiser can reconstruct it.

These rewards allow the leader to increase the probability of noise realizations that improve the follower and decrease the probability of those that do not, using a score function estimator (Bartosh et al., 2026). Because such estimators may have high variance, we normalize and clip the rewards:

$$
\tilde { R } _ { k } = \operatorname* { m i n } \Big \{ \operatorname* { m a x } \big \{ \frac { R _ { k } - B _ { k } } { \sqrt { v + \epsilon } } , - c \big \} , c \Big \} , \quad \mathrm { w h e r e } \ B _ { k } = \frac { 1 } { K - 1 } \sum _ { j \neq k } R _ { j } ,\tag{10}
$$

where $B _ { k }$ is a leave one out baseline, v is the running variance of the rewards, and c is the clipping threshold. The final leader’s objective is

$$
\mathcal { L } ^ { L } ( \phi ) = - \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \mathrm { s g } \{ \widetilde { R } _ { k } \} \log q _ { \phi } \left( x _ { t } ^ { ( k ) } \mid x _ { 0 } , t \right) .\tag{11}
$$

Minimizing this objective increases the probability of noise realizations with positive relative reward and decreases the probability of those with negative relative reward.

Stackelberg Game Dynamics. As mentioned above, computing the exact best responses is prohibitively expensive. We therefore organize training into blocks of N steps. At the start of each block, the leader uses a one-step virtual update to estimate the follower’s expected better response and commits to a forward noise process $Q _ { t , \phi } .$ . The follower then performs N gradient updates while the noise process remains unchanged.

The full algorithm is summarized in Algorithm 1. Here, we let $\eta _ { \theta }$ and $\eta _ { \phi }$ denote the learning rates of the follower and the leader, respectively. At the beginning of each training block, we fix the leader’s forward corruption process $Q _ { \phi }$ and the cumulative transition matrix ${ \bar { Q } } _ { \phi } ,$ , which remain unchanged during the N follower updates (lines 5-6). For each minibatch within the block, the follower performs a gradient update of θ by optimizing $\mathcal { L } ^ { F }$ via Equation (5) (line 7). At the end of the block, the leader samples a validation minibatch $x _ { 0 } ^ { \mathrm { v a l } }$ and computes its reference loss using Equation (7) (line 10). The leader then creates K corrupted versions $x _ { t } ^ { ( k ) }$ of the same validation minibatch, computes a one-step better response for each corruption realization, and evaluates the corresponding reward (lines 11-14). The rewards are then normalized using Equation 10 (line 15). In line 16, we compute the leader’s loss via Equation (11) and add a terminal KL-regularization that encourages $\bar { Q } _ { T , \phi }$ to match the prior (here, $\lambda _ { T }$ is a regularization term). Finally (line 17), the leader updates its action ϕ by minimizing its loss as in Equation (11). Importantly, this update is based on how much the sampled corruptions are predicted to improve the denoising process, rather than on how well they suit the current denoiser.

## 6 EVALUATION

We evaluate our approach on three domains that differ substantially in vocabulary size, structural constraints, and the nature of sequential dependencies: molecular, text, and playlist generation. We compare the following five forward process designs, all sharing the same denoiser architecture and comparable training settings. More details on the datasets and the setups are in Appendix B.

1. MDLM (Sahoo et al., 2024). The standard absorbing forward process where tokens are independently replaced with a designated [MASK] token.

2. D3PM-Uniform (Austin et al., 2021). A uniform noise process in which corrupted tokens are replaced by tokens sampled uniformly at random. The training objective combines a KL-term with a cross-entropy reconstruction loss following the standard discrete diffusion loss.

3. Forward-Learned Discrete Diffusion (FLDD) (Bartosh et al., 2026). A non-Markov forward noise process that uses a REINFORCE surrogate objective to learn the forward process.

4. D3PM-Reinforce. A learnable Markov forward process with the same parameterization as VSDD, in which ϕ and θ are jointly optimized using the score function surrogate objective of Bartosh et al. (2026).

5. Variational Stackelberg Discrete Diffusion (VSDD). Our model, which uses the same learnable Markov forward process as D3PM-Reinforce but optimizes it through the Stackelberg dynamics (Algorithm 1). The leader is updated every N = 100 follower minibatch steps.

## 6.1 RESULTS

Molecular generation. We use molecular data from a publicly available large-scale ZINC database (Irwin et al., 2020), represented as SMILES strings (Weininger, 2002). This domain imposes strict structural constraints: small syntactic errors can render an entire generated sequence invalid, making it particularly suitable for evaluating the re-drafting capabilities of our model. The small vocabulary of 64 tokens allows us to visually inspect the learned noise processes. We evaluate the chemical validity of the generated molecular sequences, i.e., the fraction of generated SMILES strings that parse into valid molecules under RDKit (Landrum et al., 2026), as well as their uniqueness, novelty, and diversity. All evaluations are computed over 15K generated samples per run.

Table 1 shows that VSDD achieves a validity rate of 84.3%, substantially outperforming both the absorbing MDLM baseline (58.7%) and D3PM-Uniform (48.1%). We further inspect the learned offdiagonal transitions and observe several chemically interpretable patterns (Figure 1b). At t = 25, substitutions among halogens, such as F, Cl, I, are more likely than under uniform corruption. These halogens commonly attach to the rest of a molecule through a single chemical bond, so exchanging them can preserve the local bonding pattern. The learned corruption also favors substitutions between the aromatic atom tokens c and n. These patterns suggest that the learned semantically aware noise process captures meaningful chemical relationships between tokens.

We further examine the relationship between the learned noise process and the denoiser’s token embeddings. The global Spearman rank correlation between token-embedding cosine similarity and transition probability is ρ = 0.70 (Figure 2a). Thus, tokens that are representationally similar under the denoiser’s learned embeddings $E _ { \theta }$ tend to be more likely to be substituted for one another. This correlation is absent by construction in D3PM-uniform and MDLM baselines.

Finally, to inspect the re-drafting abilities of VSDD, we partially corrupt clean molecules by running the forward process for 20%, 40% and 60% of the diffusion steps, then denoise them to reconstruct the originals. Since different noise processes reach different corruption levels after the same number of steps, we plot the reconstruction validity against the mean Tanimoto similarity (Bajusz et al., 2015) between the corrupted molecules and clean ones. This provides a model-agnostic measure of the trade-off between the injected noise and reconstruction quality. Figure 1a shows that VSDD achieves higher reconstruction validity than the baselines across all perturbation levels.

![](images/634347cac4b363b5113b8acba03009db9a2225bf3a593eca8b1d74723dd8cf36.jpg)  
(a) Validity/similarity after partial re-drafting.

![](images/db4d2ca2d2010207e7d04a8476e3e3ed71c3e3cfa442ce4a91c5f51bc9419a4e.jpg)  
(b) Learned transition matrices for the molecular domain. The color map is relative to the uniform noise.

Figure 1: Re-drafting and learned corruption in molecular generation.  
Table 1: Chemical validity, uniqueness, novelty and diversity of generated molecules.
<table><tr><td>Method</td><td>Validity (%) ↑</td><td>Uniqueness (%) ↑</td><td>Novelty (%) ↑</td><td>Diversity ↑</td></tr><tr><td>MDLM</td><td>58.7</td><td>100</td><td>100</td><td>0.8717</td></tr><tr><td>D3PM-Uniform</td><td>48.1</td><td>100</td><td>100</td><td>0.8674</td></tr><tr><td>D3PM-Reinforce</td><td>62.0</td><td>100</td><td>100</td><td>0.8727</td></tr><tr><td>FLDD</td><td>3.0</td><td>100</td><td>100</td><td>0.9761</td></tr><tr><td>VSDD (ours)</td><td>84.3</td><td>100</td><td>100</td><td>0.8679</td></tr></table>

Text Generation. We evaluate VSDD on the TinyStories dataset (Eldan and Li, 2023), which allows us to study the learned forward process on natural language sequences with less rigid syntax and longer range semantic dependencies. We preprocess data using a BPE tokenizer with a vocabulary of 2048 tokens and sequence length of 256 tokens. For computational efficiency, we train all models on a random subset of 100K samples from the dataset. We evaluate the generated stories using GPT-2 perplexity (PPL) and Distinct-N language diversity metric (Li et al., 2016).

VSDD more than halves the GPT-2 perplexity of D3PM-Uniform, reducing it from 236.33 to 106.71 (Table 2a). This approaches MDLM’s perplexity of 103.26 while achieving slightly higher Distinct-1 and Distinct-2 scores. Thus, VSDD substantially narrows the perplexity gap between uniform and masked diffusion while retaining the ability to revise previously generated tokens. These results show that the flexibility to re-draft need not incur the large perplexity penalty observed with unstructured uniform corruption.

In contrast, the jointly learned D3PM-Reinforce and FLDD baselines have significantly higher perplexity. As discussed in Section 4, jointly optimizing the forward process and the denoiser can favor corruptions that are more predictable rather than informative. Minimizing the denoising loss $\mathcal { L } _ { K L } ( t )$ can therefore yield nearly deterministic transitions (for example, $\mathrm { P r } ( ^ { \circ } \mathrm { w a s } ^ { , \prime \prime }  ^ { \circ } \mathrm { a n d } ^ { , \prime \prime } ) = 0 . 9 8$ , and $\operatorname* { P r } ( \mathbf { \ddot { \eta } } . \mathbf { \dot { \mu } } ) \mathbf { \ddot { \Psi } } ( \mathbf { \dot { \mu } } _ { 0 } , \mathbf { \dot { \mu } } ) = 0 . 9 9 \mathbf { \dot { \Psi } }$ ; see Appendix D). In contrast, VSDD accounts for the follower’s adaptation when updating the leader and favors transitions that improve generation quality under the reference noise process. We further analyze VSDD’s re-drafting capability in Appendix C.1.

Playlist generation. We now use a real world playlist recommendation dataset comprising approximately 44K listening histories. Each history is a variable length sequence of discrete semantic IDs of the tracks the user interacted with. Each semantic ID is represented by a triplet of tokens (see Appendix B.3). This domain allows us to evaluate our approach on sequences whose coherence is governed by behavioral and semantic relationships rather than an explicit formal grammar.

To evaluate our model, we hold out the final three tracks (9 tokens) of each validation playlist and perform conditional reverse diffusion. The prefix positions remain fixed to their ground-truth values throughout all reverse steps, and we only denoise the held-out suffix positions. A generated triplet is considered relevant if it exactly matches any of the three ground-truth held-out triplets. We report NDCG@3 and HitRate@3 on 500 playlists in Table 2b. With only three relevant tracks among approximately 228K unique triplets, exact match prediction is challenging: a uniformly random prediction would have a hit probability of approximately $1 . 3 \times 1 0 ^ { - 5 }$ . VSDD achieves NDCG@3 of 0.108 and HitRate@3 of 0.246, outperforming all four baselines on both metrics.

![](images/48ec503f386546da4e51f80b3abc25b1f621ac06485e50786563de0eb91ee1d2.jpg)  
(a) Molecular generation.

![](images/64248d022680348ae4df317db2fb55809be3059e9b1b35ea0dd7f4a7a2520a5e.jpg)  
(b) Text generation.

![](images/145e26c2b9caa68dfaa935385e43e8ed838427f59717e672fe88fd0548aa68de.jpg)  
(c) Playlist generation.  
Figure 2: Correlation between the similarity of token embeddings and the transition probabilities.

Table 2: Text and playlist generation results.
<table><tr><td>Model</td><td>PPL↓</td><td>Dist-1 ↑</td><td>Dist-2 ↑</td></tr><tr><td>MDLM</td><td>103.26</td><td>0.1535</td><td>0.5303</td></tr><tr><td>D3PM-Uniform</td><td>236.33</td><td>0.1362</td><td>0.5718</td></tr><tr><td>D3PM-Reinforce</td><td>351.83</td><td>0.2447</td><td>0.7022</td></tr><tr><td>FLDD</td><td>871.51</td><td>0.2441</td><td>0.7671</td></tr><tr><td>VSDD (ours)</td><td>106.71</td><td>0.1570</td><td>0.5426</td></tr></table>

(a) Perplexity and diversity of generated stories.

<table><tr><td>Model</td><td>NDCG@3↑</td><td>HitRate@3↑</td></tr><tr><td>MDLM</td><td>0.0882</td><td>0.2020</td></tr><tr><td>D3PM-Uniform</td><td>0.0575</td><td>0.1540</td></tr><tr><td>D3PM-Reinforce</td><td>0.0526</td><td>0.1340</td></tr><tr><td>FLDD</td><td>0.0418</td><td>0.1100</td></tr><tr><td>VSDD (ours)</td><td>0.1077</td><td>0.2460</td></tr></table>

(b) NDCG@3 and HitRate@3 of generated playlists.

In Appendix C.2, we further study the re-drafting properties of VSDD by analyzing the model’s ability to repair targeted structural corruptions in playlists.

## 6.2 DISCUSSION

Our evaluation highlights that both the choice of the corruption process and its optimization matter. VSDD outperforms the jointly learned D3PM-Reinforce and FLDD baselines on the primary metrics across all three domains. The comparison with D3PM-Reinforce is particularly informative because it uses the same noise parameterization but optimizes the forward process differently. These results support our central premise that useful corruptions should be selected by the improvement they induce after denoiser adaptation, rather than on how easily the current denoiser reconstructs them.

The comparison with fixed corruption processes varies across domains. For molecular generation, VSDD substantially improves validity over both uniform and masked diffusion, while for playlist generation it improves both NDCG@3 and HitRate@3. For text generation, VSDD substantially reduces the perplexity of uniform diffusion and is comparable to masked diffusion.

The learned noise and re-drafting analyses provide complementary evidence beyond generation quality. In molecules, the learned noise maps reveal pronounced, token-dependent structure, including chemically interpretable substitutions. Furthermore, it achieves higher reconstruction validity across the evaluated partial-corruption levels. We observed similar results for the text and playlist generation tasks. Together, these results show that the benefits of the learned corruption process extend beyond generation to re-drafting perturbed sequences across all three domains.

## 7 CONCLUSION

We introduced Variational Stackelberg Discrete Diffusion (VSDD), a framework for learning a semantically aware forward noise process through the denoiser’s response. The leader evaluates sam pled corruptions by the improvement they induce after a virtual follower update, while the follower learns to reverse the selected noise process. Experiments on molecular, text, and playlist generation show improvements over fixed and learnable forward noise process baselines. Our reconstruction and conditional-infilling experiments further show that these benefits extend to re-drafting, with VSDD effectively repairing controlled perturbations across all three domains. Together, these findings suggest that effective discrete diffusion benefits from learning not only how to reverse corruption, but also which corruptions help the denoiser learn to revise.

## REFERENCES

Jacob Austin, Daniel D. Johnson, Jonathan Ho, Daniel Tarlow, and Rianne van den Berg. 2021. Structured Denoising Diffusion Models in Discrete State-Spaces. In Advances in Neural Information Processing Systems, Marc’Aurelio Ranzato, Alina Beygelzimer, Yann Dauphin, Percy S. Liang, and Jennifer Wortman Vaughan (Eds.), Vol. 34. Curran Associates, Inc., 17981– 17993. https://proceedings.neurips.cc/paper\_files/paper/2021/file/ 958c530554f78bcd8e97125b70e6973d-Paper.pdf

Dávid Bajusz, Anita Rácz, and Károly Héberger. 2015. Why is Tanimoto index an appropriate choice for fingerprint-based similarity calculations? Journal of Cheminformatics 7 (2015). https: //api.semanticscholar.org/CorpusID:1221969

Grigory Bartosh, Teodora Pandeva, Sushrut Karmalkar, and Javier Zazo. 2026. Forward-Learned Discrete Diffusion: Learning How to Noise to Denoise Faster. In International Conference on Learning Representations. arXiv:2605.18204 [stat.ML] doi:10.48550/arXiv.2605. 18204

Samy Bengio, Oriol Vinyals, Navdeep Jaitly, and Noam Shazeer. 2015. Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks. CoRR abs/1506.03099 (2015). arXiv:1506.03099 http://arxiv.org/abs/1506.03099

Vincent Conitzer and Tuomas Sandholm. 2006. Computing the optimal strategy to commit to. In Proceedings of the 7th ACM Conference on Electronic Commerce (Ann Arbor, Michigan, USA) (EC ’06). Association for Computing Machinery, New York, NY, USA, 82–90. doi:10.1145/ 1134707.1134717

Sander Dieleman, Laurent Sartran, Arman Roshannai, Nikolay Savinov, Yaroslav Ganin, Pierre H. Richemond, Arnaud Doucet, Robin Strudel, Chris Dyer, Conor Durkan, Curtis Hawthorne, Rémi Leblond, Will Grathwohl, and Jonas Adler. 2022. Continuous diffusion for categorical data. arXiv:2211.15089 [cs.CL] https://arxiv.org/abs/2211.15089

Ronen Eldan and Yuanzhi Li. 2023. TinyStories: How Small Can Language Models Be and Still Speak Coherent English? arXiv:2305.07759 [cs.CL] https://arxiv.org/abs/2305. 07759

Tanner Fiez, Benjamin Chasnov, and Lillian Ratliff. 2020. Implicit Learning Dynamics in Stackelberg Games: Equilibria Characterization, Convergence Analysis, and Empirical Study. In Proceedings of the 37th International Conference on Machine Learning (Proceedings of Machine Learning Research, Vol. 119), Hal Daumé III and Aarti Singh (Eds.). PMLR, 3133–3144. https://proceedings.mlr.press/v119/fiez20a.html

Chelsea Finn, Pieter Abbeel, and Sergey Levine. 2017. Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks. In Proceedings of the 34th International Conference on Machine Learning (Proceedings of Machine Learning Research, Vol. 70), Doina Precup and Yee Whye Teh (Eds.). PMLR, 1126–1135. https://proceedings.mlr.press/v70/finn17a. html

Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. 2014. Generative adversarial nets. In Proceedings of the 28th International Conference on Neural Information Processing Systems - Volume 2 (Montreal, Canada) (NIPS’14). MIT Press, Cambridge, MA, USA, 2672–2680.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. 2020. Denoising diffusion probabilistic models. In Proceedings of the 34th International Conference on Neural Information Processing Systems (Vancouver, BC, Canada) (NIPS ’20). Curran Associates Inc., Red Hook, NY, USA, Article 574, 12 pages.

Emiel Hoogeboom, Didrik Nielsen, Priyank Jaini, Patrick Forré, and Max Welling. 2021. Argmax flows and multinomial diffusion: learning categorical distributions. In Proceedings ofthe 35th International Conference on Neural Information Processing Systems (NIPS ’21). Curran Associates Inc., Red Hook, NY, USA, Article 953, 12 pages.

John J. Irwin, Khanh G. Tang, Jennifer Young, Chinzorig Dandarchuluun, Benjamin R. Wong, Munkhzul Khurelbaatar, Yurii S. Moroz, John W. Mayfield, and Roger A. Sayle. 2020. ZINC20 – A Free Ultra Large-Scale Chemical Database for Ligand Discovery. Journal ofchemical information and modeling 60 (2020), 6065 – 6073. https://api.semanticscholar.org/ CorpusID:226059428

Diederik P. Kingma, Tim Salimans, Ben Poole, and Jonathan Ho. 2021. Variational diffusion models. In Proceedings of the 35th International Conference on Neural Information Processing Systems (NIPS ’21). Curran Associates Inc., Red Hook, NY, USA, Article 1660, 12 pages.

Greg Landrum, Paolo Tosco, Ricardo Rodriguez, Brian Kelley, David Cosgrove, Riccardo Vianello, sriniker, Peter Gedeck, Gareth Jones, Dan Nealschneider, Eisuke Kawashima, NadineSchneider, tadhurst cdd, Andrew Dalke, Niels Maeder, Matt Swain, Yakov Pechersky, Brian Cole, Kevin Boyd, Samo Turk, Aleksandr Savelev, Rachel Walker, Alain Vaucher, Maciej Wójcikowski, Hussein Faara, Ichiru Take, Vincent F. Scalfani, Steven Kearnes, Kazuya Ujihara, and Daniel Probst. 2026. rdkit/rdkit: 2026\_03\_6 (Q1 2026) Release. doi:10.5281/zenodo.22140358

Jiwei Li, Michel Galley, Chris Brockett, Jianfeng Gao, and Bill Dolan. 2016. A Diversity-Promoting Objective Function for Neural Conversation Models. In Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Kevin Knight, Ani Nenkova, and Owen Rambow (Eds.). Association for Computational Linguistics, San Diego, California, 110–119. doi:10.18653/v1/N16-1014

Aaron Lou, Chenlin Meng, and Stefano Ermon. 2024. Discrete diffusion modeling by estimating the ratios of the data distribution. In Proceedings of the 41st International Conference on Machine Learning (Vienna, Austria) (ICML’24). JMLR.org, Article 1333, 30 pages.

Dmitrii Moor, Yi Yuan, Rishabh Mehrotra, Zhenwen Dai, and Mounia Lalmas. 2023. Exploiting Sequential Music Preferences via Optimisation-Based Sequencing. In Proceedings of the 32nd ACM International Conference on Information and Knowledge Management (Birmingham, United Kingdom) (CIKM ’23). Association for Computing Machinery, New York, NY, USA, 4759–4765. doi:10.1145/3583780.3615476

Alexander Quinn Nichol and Prafulla Dhariwal. 2021. Improved Denoising Diffusion Probabilistic Models. In Proceedings ofthe 38th International Conference on Machine Learning (Proceedings of Machine Learning Research, Vol. 139), Marina Meila and Tong Zhang (Eds.). PMLR, 8162– 8171. https://proceedings.mlr.press/v139/nichol21a.html

Marc’Aurelio Ranzato, Sumit Chopra, Michael Auli, and Wojciech Zaremba. 2015. Sequence Level Training with Recurrent Neural Networks. CoRR abs/1511.06732 (2015). https://api. semanticscholar.org/CorpusID:7147309

Subham Sekhar Sahoo, Marianne Arriola, Yair Schiff, Aaron Gokaslan, Edgar Marroquin, Justin T Chiu, Alexander Rush, and Volodymyr Kuleshov. 2024. Simple and Effective Masked Diffusion Language Models. In Advances in Neural Information Processing Systems, Vol. 37. doi:10. 52202/079017-4135

Jiaxin Shi, Kehang Han, Zhe Wang, Arnaud Doucet, and Michalis K. Titsias. 2024. Simplified and generalized masked diffusion for discrete data. In Proceedings of the 38th International Conference on Neural Information Processing Systems (Vancouver, BC, Canada) (NIPS ’24). Curran Associates Inc., Red Hook, NY, USA, Article 3277, 37 pages.

Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. 2015. Deep Unsupervised Learning using Nonequilibrium Thermodynamics. In Proceedings of the 32nd International Conference on Machine Learning (Proceedings of Machine Learning Research, Vol. 37), Francis Bach and David Blei (Eds.). PMLR, Lille, France, 2256–2265. https: //proceedings.mlr.press/v37/sohl-dickstein15.html

Yang Song, Jascha Narain Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. 2020. Score-Based Generative Modeling through Stochastic Differential Equations. ArXiv abs/2011.13456 (2020). https://api.semanticscholar.org/ CorpusID:227209335

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Ł ukasz Kaiser, and Illia Polosukhin. 2017. Attention is All you Need. In Ad vances in Neural Information Processing Systems, I. Guyon, U. Von Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett (Eds.), Vol. 30. Curran Associates, Inc. https://proceedings.neurips.cc/paper/2017/file/ 3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf

David Weininger. 2002. SMILES, a chemical language and information system. 1. Introduction to methodology and encoding rules. Journal of Chemical Information and Computer Sciences 28, 1 (05 2002), 31–36. arXiv:https://pubs.acs.org/jcisd8/articlepdf/28/1/31/10787984/ci00057a005.pdf doi:10.1021/ci00057a005

## A APPENDIX

## B ADDITIONAL EXPERIMENTAL DETAILS

In all learnable Markov noise models in our experiments (i.e., D3PM-Reinforce and VSDD, see Section 6) we rely on the same architecture of the forward noise process $Q _ { t , \phi }$ and the learnable noise kernel. Specifically, the off-diagonal transition kernel $M _ { t , \phi }$ is parameterised by a timeconditioned bilinear scoring module that computes token-to-token affinities according to Equation (4), and $A ( t ) \in \mathbb { R } ^ { d \times d }$ is produced by a two-layer MLP with SiLU activations operating on sinusoidal time features. Rows of the score matrix are softmax-normalised, giving a valid row-stochastic off-diagonal kernel $M _ { t , \phi } .$ . The full single-step transition matrix $Q _ { t , \phi }$ is then computed according to Equation (3), and the cumulative product $\begin{array} { r } { \bar { Q } _ { t , \phi } = \prod _ { \tau = 1 } ^ { t } Q _ { \tau , \phi } } \end{array}$ is computed sequentially. A terminaluniform regularizer encourages $\hat { Q } _ { T }$ to converge to a uniform stationary distribution over valid tokens.

## B.1 MOLECULAR DOMAIN

Data & Vocabulary. In the molecular domain, we randomly sample 1M molecules and use a 90/10 train/validation split. We tokenize SMILES sequences provided in the dataset at the atom level using a custom tokenizer that handles bracket atoms (e.g., [C@@H], [NH+]), two-letter elements (e.g., Cl, Br), single-letter atoms, and structural characters (bonds, branches, ring closures). This yields a compact vocabulary of $| \nu | = 6 4$ tokens and a maximum sequence length of $L = 1 3 4$ . An [EOS] token terminates each molecule; sequences shorter than L are right-padded with [PAD].

Denoiser architecture. All baselines as well as our proposed model use the same denoising backbone: a bi-directional Transformer encoder (Vaswani et al., 2017) with the embedding dimensionality $d = 5 1 2$ , ten layers, 32 attention heads, and sinusoidal time embeddings injected additively into the input representation. The diffusion horizon is $T = 5 0$ steps with a linear noise schedule $\bar { \alpha } _ { t }$ decreasing from 1 to 0. Models are trained for 50 epochs with the Adam optimizer (learning rate $1 0 ^ { - 4 }$ , batch size 512) on two A100 GPUs.

## B.2 TEXT GENERATION DOMAIN

Data & Vocabulary. We evaluate on the TinyStories dataset (Eldan and Li, 2023), tokenised with a byte-pair encoding (BPE) tokeniser trained on the full training split, yielding a vocabulary of $V = 2 0 4 8$ tokens and a maximum sequence length of $L = 2 5 6$ . Three special tokens are reserved: [PAD] for right-padding sequences shorter than L, [EOS] for end-of-sequence, and [MASK] (used only by the masked-diffusion baselines). We train on 100k sequences randomly selected from the training split and validate on a held-out set of 1,000 sequences.

Denoiser architecture. All models share the same denoising network $p _ { \theta } ( x _ { 0 } | x _ { t } , t )$ : a bidirectional Transformer encoder with the embedding dimension $d = 5 1 2 ,$ ten layers, sixteen attention heads, and sinusoidal time embeddings added to the input representation. The model takes noisy tokens $x _ { t }$ and a time embedding as input and outputs logits over the vocabulary at every sequence position. We perform training on multiple A100 80 GB GPUs with a batch size of 256. The diffusion horizon has $T = 5 0$ steps with a linear noise schedule $\bar { \alpha } _ { t }$ decreasing from 1 to 0. The denoiser is trained with Adam optimizer with the learning rate $\eta _ { \theta } = 1 0 ^ { - 4 }$ , weight decay $1 0 ^ { - 4 }$ , and gradient clipping at 1.0. The kernel parameters ϕ are optimized with a separate Adam instance with the learning rate $\eta _ { \phi } = 2 \times 1 0 ^ { - 4 }$

## B.3 PLAYLIST DOMAIN

Data & Vocabulary. We evaluate our model on a proprietary playlist dataset comprising listening sequences derived from over one million unique music tracks, each represented by an 80- dimensional embedding vector. The dataset consists of 44,042 listening histories (36,765 training, 7,277 validation), where each history is a variable-length sequence of tracks. To obtain a discrete vocabulary suitable for our model, we map each track’s 80-dimensional embedding vector to a semantic ID (SID) via locality-sensitive hashing (LSH): 21 random hyperplanes are partitioned into 3 levels of 7 bits each, yielding $2 ^ { 7 } = 1 2 8$ buckets per level plus 3 special tokens (PAD, MASK, EOS) for a vocabulary of $V = 1 3 1$ . Each track is thus represented by a triplet $\left( { t _ { 0 } , t _ { 1 } , t _ { 2 } } \right)$ of semantic tokens, and the resulting corpus contains 228,132 unique triplets. Sequences are truncated or padded to a fixed length of $L = 1 5 0$ semantic tokens.

Denoiser architecture. The denoiser architecture for playlist generation is similar to the other two tasks. We let the dimensionality of the token embeddings be $d = 5 1 2$ , and we use ten layers with 16 attention heads, trained for 50 epochs with batch size 28 and the Adam optimizer (follower $\mathrm { l r } = 1 0 ^ { - 4 }$ , leader $\mathrm { l r } = 2 \times 1 0 ^ { - 4 } )$ . The diffusion process uses $T = 5 0$ steps.

## C RE-DRAFTING ANALYSIS

In this section, we expand our analysis of the re-drafting (conditional infilling) capabilities of VSDD for the text and playlist generation domains. Conceptually, such an analysis is similar to the qualitative analysis that we presented for the molecular generation domain (see Section 6.1, last paragraph). However, as both the playlist and the text generation tasks have significantly larger vocabularies, the visual analysis of the learned forward noise process (Figure 1b) is not straightforward in these domains. Instead, we construct a number of rule-based re-drafting tests, and we analyse the performance of our model on those tests.

## C.1 TEXT GENERATION

In the text generation domain, we evaluate three discrete diffusion models trained on the TinyStories dataset. Re-drafting tests whether a model, given a story with a localised corruption, can restore the original content by conditioning on the uncorrupted context. This tests the model’s ability to maintain a coherent narrative and factual agreement across the sequence.

To this end, we compare our VSDD model that learns the forward noise process against two baselines that rely on a fixed (pre-computed) forward noise process, namely, MDLM and D3PM-Uniform. We then apply repaint-style conditional infilling, i.e., given a sequence with a designated repainting mask, we use the model to regenerate only the masked positions while keeping the context positions fixed.

In the VSDD and D3PM-Uniform experiments, we initialise the positions marked with the repainting mask with uniform random noise over the valid tokens. The full reverse diffusion is then run for T steps. At each step, the model predicts $p _ { \theta } ( x _ { 0 } | x _ { t } , t )$ and computes the posterior sampling probabilities $p ( x _ { t - 1 } | x _ { t } , x _ { 0 } )$ . It then samples the new tokens from this posterior. Only the masked positions are updated; context positions are held fixed. For the MDLM model the edit positions are set to the absorbing mask token [MASK]. Only currently-masked non-pad positions are updated at each step while the context positions remain unchanged. All experiments use five independent trials per corruption per model.

Experiment 1: Entity Swap. For the first experiment, we construct five templated stories with repeated entities (names and objects). Each story contains 3–5 occurrences of a target entity. We corrupt exactly one occurrence by replacing its BPE token (see Appendix B.2) with a different entity token, creating a single-token inconsistency. All entity names (“Lily", “Tom", “Max", “Sam", “Ben") and objects (“ball", “dog", “car", “cat", “train") are verified to be single BPE tokens, ensuring the corruption is a clean one-token swap with no sequence-length change.

Observe that in this setting the absorbing-state mechanism of MDLM is particularly well-suited for single-token infilling. Indeed, MDLM is directly optimized for this specific task of unmasking a single masked token at the corrupted position with full surrounding context. Consequently, its training objective is precisely to predict the clean token at masked positions. This provides an ideal solution for entity swap, and we are interested in analysing how close the rest of the models can approach MDLM on this task.

The five corruption instances are in Table 3, and we provide an example of such a corruption in Table 5. For each story we run five corruption trials, and average the resulting metrics across the trials and stories. We measure the exact repair rate as a fraction of the corrupted tokens restored to the original value (averaged over all trials) and consistency as the fraction of all entity occurrences that agree with the most-common value after infilling. A score of 1.0 means all instances of the entity hold the same token. Table 4 contains the aggregated results.

Table 3: Synthetic story corruptions. For each story, an entity is replaced at a selected occurrence.
<table><tr><td>Story name</td><td>Corruption</td><td># occurrences</td><td>Corrupted occurrence</td></tr><tr><td>Lily and red ball</td><td>Lily → Tom</td><td>4</td><td>2</td></tr><tr><td>Tom and Max</td><td>Max → Sam</td><td>5</td><td>1</td></tr><tr><td>Sam and blue car</td><td>car → dog</td><td>4</td><td>3</td></tr><tr><td>Lily and cat</td><td>cat → dog</td><td>5</td><td>2</td></tr><tr><td>Ben and toy train</td><td>train → ball</td><td>5</td><td>3</td></tr></table>

Table 4: Aggregate repair performance. Results report the mean and the standard deviation across corruption types.
<table><tr><td>Model</td><td>Exact Repair ↑</td><td>Consistency ↑</td></tr><tr><td>VSDD</td><td> $0 . 8 8 0 \pm 0 . 1 6 0$ </td><td> $0 . 9 7 4 \pm 0 . 0 3 3$ </td></tr><tr><td>D3PM-Uniform</td><td> $0 . 3 6 0 \pm 0 . 2 6 5$ </td><td> $0 . 8 6 0 \pm 0 . 0 6 5$ </td></tr><tr><td>MDLM</td><td> $0 . 9 6 0 \pm 0 . 0 8 0$ </td><td> $0 . 9 9 2 \pm 0 . 0 1 6$ </td></tr></table>

From Table 4 we see that VSDD closely approaches the “ideal" performance of MDLM, leaving a substantial gap with D3PM-Uniform. These results indicate that even though VSDD was not trained to perform well on the entity swap task, it manages to efficiently re-draft the corrupted parts of the sequence. Despite the small performance gap with MDLM, below we illustrate that the infilled stories are still structurally and semantically consistent.

In our first example (Table 5), D3PM-Uniform never recovers "Lily" across all 5 trials (repair = 0.0); it consistently generates the pronoun "She", which is grammatically plausible but breaks the entity pattern. VSDD recovers "Lily" in 4 out of 5 trials (repair = 0.8).

The second example (Table 6) is the hardest corruption: "Max" appears 5 times but the corrupted position is the first occurrence, where the model must choose between two plausible names $( " \mathrm { { M a x } " }$ from later context vs. "Sam" or "Tom" from the immediate sentence). MDLM occasionally generates "Tom" (the human character’s name) instead of "Max", showing a consistency failure despite high overall repair rate.

With strong contextual cues (three other occurrences of "blue car" surrounding the corruption), all models perform well (Table 7). In this case, even D3PM-Uniform achieves 80% repair, demonstrating that sufficient redundancy can compensate for the lack of the structured noise process.

Overall, we see that using D3PM-Uniform with an unstructured noise process frequently generates plausible but incorrect tokens (e.g., the pronoun "She" instead of "Lily", or the article "a" instead of "cat"). In contrast, VSDD’s learned kernel provides intermediate token-level affinity structure, yielding substantially better repair than the uniform baseline.

Experiment 2: Subsequence Removal. In our second experiment, a contiguous middle subsequence (approximately 35%–55% of the story length, corresponding to 8–9 tokens) is removed and replaced with [MASK] tokens. The removed subsequence typically contains a narrative bridge connecting the story’s setup to its conclusion (see the example below). For all stories, we measure the token overlap (i.e., the fraction of in-filled tokens that exactly match the original removed subsequence). Table 11 shows the aggregated results.

From Table 11 we see that across the three stories VSDD achieves the highest average overlap (24.3%) and produces the most coherent bridges, followed by MDLM (18.4%) and D3PM-Uniform (16.3%). The low absolute values of the token overlaps are expected since there are many valid ways to bridge the narrative gap (so the metric provides a lower bound on generation quality).

<table><tr><td>Example 1: Lily → Tom (2nd occurrence)</td><td></td></tr></table>

Table 5: The 2nd occurrence of “Lily” is corrupted to “Tom”, each model attempts to recover the original token.

Original Once upon a time there was a little girl named Lily. Lily had a big red ball. She loved to play with the red ball every day. One day Lily took the red ball to the park. Lily was very happy.

Corrupted Once upon a time there was a little girl named Lily. Tom had a big red ball. She loved to play with the red ball every day. One day Lily took the red ball to the park. Lily was very happy.

<table><tr><td>Model</td><td>Infilled text</td><td>Repair?</td></tr><tr><td>VSDD</td><td>. . . named Lily. Lily had a big red ball. She loved ..</td><td>Yes</td></tr><tr><td>D3PM-Uniform</td><td>. . . named Lily. She had a big red ball. She loved . . . Generates &quot;She&quot; instead of &quot;Lily&quot;.</td><td>No</td></tr><tr><td>MDLM</td><td>. . . named Lily. Lily had a big red ball. She loved ...</td><td>Yes</td></tr></table>

Table 6: The 1st occurrence of “Max” is corrupted to “Sam”.
<table><tr><td>Example 2: Max → Sam (1st occurrence)</td><td></td></tr></table>

Original Tom had a small dog named Max. Tom and Max liked to play in the park. Every morning Tom took Max for a long walk. Max was a good dog and Tom loved Max very much.

Corrupted Tom had a small dog named Sam. Tom and Max liked to play in the park. Every morning Tom took Max for a long walk. Max was a good dog and Tom loved Max very much.

<table><tr><td>Model</td><td>Infilled text</td><td>Repair?</td></tr><tr><td>VSDD</td><td>. . . dog named Max. Tom and Max liked .. . Successful in 3/5 trials.</td><td>Yes</td></tr><tr><td>D3PM-Uniform</td><td>. . . dog named Max. Tom and Max liked . .. Successful in 1/5 trials.</td><td>Rarely</td></tr><tr><td>MDLM</td><td>. . . dog named Tom. Tom and Max liked . .. Repairs in 4/5 trials, but occasionally generates &quot;Tom&quot; instead of &quot;Max&quot;.</td><td>Partial</td></tr></table>

In Table 12, we provide an example from this experiment. From the example, we can see that VSDD preserves both the key entity ("red ball") and the narrative structure ("she loved ... She took her"), yielding the highest token overlap. D3PM-Uniform produces incoherent repetition ("The red liked red new toy"). MDLM generates a grammatical but nonsensical phrase ("ball with many hair"). Qualitatively, VSDD more consistently preserves key entities from the surrounding context and generates grammatical continuations.

Table 7: The third occurrence of $\mathbf { \dot { \omega } } _ { \mathbf { C a r } } \mathbf { \vec { \omega } } _ { \mathbf { \xi } }$ is corrupted to “dog”.
<table><tr><td>Example 3: car → dog</td><td>(3rd occurrence)</td></tr><tr><td colspan="2">Original There was a little boy named Sam. Sam had a blue car. Sam liked to play with his blue car. One day Sam lost his blue car. Sam was sad but then he found the blue car under the bed. Corrupted There was a little boy named Sam. Sam had a blue car. Sam liked to play with his blue car. One day Sam lost his blue dog. Sam was sad but then he found the blue car under the bed.</td></tr><tr><td>Model Infilled text</td><td>Repair?</td></tr><tr><td>VSDD</td><td>. . . lost his blue car. Sam was sad .. Yes</td></tr><tr><td>D3PM-Uniform</td><td>Successful in 5/5 trials. . . . lost his blue car. Sam was sad ... Mostly</td></tr><tr><td>MDLM</td><td>Successful in 4/5 trials. . . . lost his blue car. Sam was sad ..</td></tr></table>

Table 8: The second occurrence of “cat” is corrupted to “dog”.
<table><tr><td>Example 4: cat → dog (2nd occurrence)</td><td></td></tr><tr><td colspan="2">Original Lily had a pretty cat. The cat was soft and white. Lily liked to pet her cat every day. One day the cat found a little mouse. Lily and the cat played in the garden. Corrupted Lily had a pretty cat. The dog was soft and white. Lily liked to pet her cat every day. One day the cat found a little mouse. Lily and the cat played in the garden.</td></tr><tr><td>Model Infilled text</td><td>Repair?</td></tr><tr><td>VSDD</td><td>. .. The cat was soft and white ...</td></tr><tr><td>D3PM-Uniform</td><td>Successful in 5/5 trials. . . . The a was soft and white ...</td></tr><tr><td>MDLM</td><td>Occasionally generates the article “a&quot; instead of “cat&quot;. . .. The cat was soft and white ...</td></tr></table>

Table 9: Qualitative example for the train → ball corruption. The third occurrence of “train” is corrupted to “ball”, and each model attempts to recover the original token.
<table><tr><td>Example 5: train → ball</td><td>(3rd occurrence)</td></tr><tr><td colspan="2">Original Ben had a toy train. Ben loved his toy train. Every day Ben played with the toy train in his room. One day Ben took the toy train to show his friend. His friend liked the toy train too. Corrupted Ben had a toy train. Ben loved his toy train. Every day Ben played with the toy ball in his room.</td></tr><tr><td>One day Ben took the toy train to show his friend. His friend liked the toy train too. Model Infilled text</td><td>Repair?</td></tr><tr><td>VSDD</td><td>. . . the toy train in his room . .. Yes</td></tr><tr><td>D3PM-Uniform</td><td>Successful in 5/5 trials. . . . the toy train in his room . . </td></tr><tr><td>MDLM</td><td>Successful in 2/5 trials. . . . the toy train in his room . . .</td></tr></table>

Table 10: Removed subsequences used for re-drafting. Each example masks a contiguous subsequence of tokens from the original story.
<table><tr><td>Story</td><td>Removed subsequence</td><td>subsequence length</td></tr><tr><td>Lily &amp; red ball</td><td>“red ball. She loved to play with  $t h e ^ { , { \boldsymbol { \mathbf { \mathit { \varepsilon } } } } }$ </td><td>9</td></tr><tr><td>Tom &amp; Max</td><td>&quot;in the park. Every morning Tom  $t o o k '$ </td><td>8</td></tr><tr><td>Sam &amp; blue car</td><td>&quot;liked to play with his blue car.  $o n e '$ </td><td>9</td></tr></table>

Table 11: Span infilling performance. Token Overlap measures agreement between the generated and original removed spans.
<table><tr><td>Model</td><td>Token Overlap ↑</td></tr><tr><td>VSDD</td><td> $\mathbf { 0 . 2 4 3 \pm 0 . 1 2 7 }$ </td></tr><tr><td>D3PM-Uniform</td><td> $0 . 1 6 3 \pm 0 . 1 5 9$ </td></tr><tr><td>MDLM</td><td> $0 . 1 8 4 \pm 0 . 0 8 2$ </td></tr></table>

Table 12: Example: Subsequence Removal.
<table><tr><td>Example: Lily story</td><td>(9 tokens removed)</td></tr><tr><td>Original .. . Lily. Lily had a big red ball. She loved to play with the red ball every day . . .</td><td></td></tr><tr><td>Removed span &quot;red ball. She loved to play with</td><td> $t h e ^ { , { \boldsymbol { \mathbf { \mathit { \varepsilon } } } } }$  Infilled span</td></tr><tr><td>Model VSDD</td><td>“red ball that she loved. She took her&quot;</td></tr><tr><td>D3PM-Uniform</td><td>“red toy. The red liked red new toy&quot;</td></tr><tr><td>MDLM</td><td>ball with many hair. She played with the&quot;</td></tr><tr><td></td><td></td></tr></table>

## C.2 PLAYLIST GENERATION

Unlike in the molecular domain discussed in Section 6.1, analysing the re-drafting capabilities of VSDD by visually inspecting the learned forward noise matrices in the playlist generation domain is not feasible. This is because the different token substitutions are not based on the underlying world knowledge but on the subjective (unobservable) user preferences. In this section, we show that the model can successfully repair local rule-based corruptions that typically reduce playlist quality.

To evaluate whether the learned forward process captures meaningful structural properties of playlist sequences, we design a targeted repair experiment. To this end, we introduce controlled corruptions into held-out playlists, and we measure the ability of the trained model to restore the structural coherence of the playlists via conditional reverse diffusion.

In particular, we define three corruption operators, each modifying exactly one track (i.e., three semantic tokens) per sequence:

• Track duplication. A source track position s is selected uniformly at random, and a distinct destination position d $\neq s$ is selected uniformly at random. The SID triplet corresponding to the track at position d is overwritten with a copy of the triplet of the track at position s, simulating a repeated-track artifact.

• Transition disruption. We identify the consecutive track pair $( i , i { + } 1 )$ with the highest audio-feature<sup>3</sup> cosine similarity (i.e., the smoothest transition) across all adjacent pairs in the playlist. We then scan all remaining tracks $j \notin \{ i , i + 1 \}$ and select the one whose audiofeature vector has the lowest cosine similarity to track i. The SID triplet corresponding to the track at position i+1 is then overwritten with the triplet of this maximally dissimilar track, creating a jarring transition at the previously smoothest point, thereby reducing consumption (Moor et al., 2023). The corrupted region is $\{ 3 ( i { + } 1 ) , 3 ( i { + } 1 ) { + } 1 , 3 ( i { + } 1 ) { + } 2 \}$

• Energy misplacement. We identify the track with the highest energy value across all positions and swap it with the track at the very first position of the playlist (the playlist opening). If the highest-energy track is already at the first position, the lowest-energy track is swapped to this position instead. The corrupted region is {0, 1, 2} (the triplet corresponding to the opening track only). Note that while the swap modifies two track positions, only the track in the first position is designated for repair, testing whether the model can restore an appropriate opening track given one-sided (right-only) context.

For each corrupted sequence, we construct a binary generation mask $\mathbf { m } \in \{ 0 , 1 \} ^ { L }$ with $\begin{array} { r l } { m _ { \ell } } & { { } = } \end{array}$ 1 at the three corrupted token positions and $m _ { \ell } = 0$ elsewhere. The corrupted sequence x˜ and mask m are passed to the conditional reverse diffusion procedure: masked positions are initialized with tokens drawn uniformly from the valid SID vocabulary, while unmasked positions retain the corrupted values throughout. The model then performs $T = 5 0$ reverse diffusion steps using the learned transition matrices from the Stackelberg-trained forward process. At each step, reverse sampling probabilities are computed from the model’s predicted logits and the learned posterior, but token updates are applied only at the three masked positions. The surrounding 147 tokens serve as fixed conditioning context.

We report three quantities, each computed over 200 corrupted validation playlists:

• Repair rate. The fraction of corrupted token positions at which the model’s output differs from the corrupted value. A high repair rate indicates that the model recognizes the corruption as inconsistent with the surrounding context.

• Audio smoothness. The mean cosine similarity between the 16-dimensional audio-feature vectors of the consecutive tracks, computed over all adjacent pairs in the playlist. We report this for the original (O), corrupted (C), and repaired (R) sequences.

• Style diversity. The number of unique genre/style descriptor tags across all tracks in the sequence, obtained from per-track metadata (with approximately 50 weighted descriptors per track from the editorial taxonomy). Reported for $\mathrm { O , C , }$ and R.

Table 13 summarizes the repair outcomes.

Table 13: Repair results for different corruption types.
<table><tr><td></td><td></td><td colspan="3">Smooth</td><td colspan="3">Diversity</td></tr><tr><td>Corruption</td><td>Repair rate</td><td>0</td><td>C</td><td>R</td><td>0</td><td>C</td><td>R</td></tr><tr><td>Duplication</td><td>78.9%</td><td>0.9800</td><td>0.9802</td><td>0.9807</td><td>2477</td><td>2453</td><td>2476</td></tr><tr><td>Transition disruption</td><td>86.1%</td><td>0.9800</td><td>0.9767</td><td>0.9792</td><td>2510</td><td>2497</td><td>2520</td></tr><tr><td>Energy misplacement</td><td>85.3%</td><td>0.9800</td><td>0.9797</td><td>0.9799</td><td>2477</td><td>2477</td><td>2481</td></tr></table>

First, we see that VSDD actively repairs corrupted positions in 78–86% of cases, with the highest repair rate for transition disruption (86.1%), where the corrupted track is maximally inconsistent with its immediate neighbours. This suggests the model has learned local coherence constraints: It detects that a track with very different audio characteristics does not belong between its neighbours and proposes a more suitable replacement.

For transition disruption, the corruption reduces audio smoothness from 0.9800 to 0.9767. After repair, the smoothness recovers to 0.9792. The model does not fully recover the original smoothness because it is not constrained to reproduce the original track. Instead, it generates any track consistent with the learned distribution conditioned on the context.

Track duplication yields a slightly lower repair rate (78.9%), consistent with the observation that a duplicated track may be contextually plausible at its destination: Unlike transition disruption, duplication does not necessarily create a local inconsistency, especially when the track is highly relevant for the user.

Energy misplacement achieves an intermediate repair rate (85.3%) despite having access to only right-side context at the very first position of the playlist. The model must infer an appropriate opening track from the subsequent sequence alone, which is a more complex task as users typically interact with the playlist from left to right (so the prefix context is often more informative when making the track allocation decision than the suffix-context).

Audio smoothness and style diversity show minimal variation across conditions because each corruption modifies only one of approximately 49 tracks. The smoothness metric averages over 48 consecutive pairs, diluting the local effect of a single corrupted transition. Style diversity is similarly insensitive: each track contributes dozens of tags, and the union over 49 tracks saturates at approximately 2,500 unique descriptors regardless of single-track substitutions.

The repair experiment provides evidence that the Stackelberg-trained diffusion model captures local sequential structure in playlist data. The learned forward process, optimized to improve the denoiser’s generalization, produces a reverse model capable of context-sensitive infilling. The high repair rate for transition disruption (86.1%) demonstrates that the model encodes audio-feature continuity as an implicit constraint, even though audio features are not explicitly provided during training as the model operates solely on discrete SID tokens derived from track embeddings via locality-sensitive hashing. The correlation between embedding cosine similarity and learned tran sition probabilities (Spearman $\rho = 0 . 7 4$ for the learned noise vs. embedding similarity, see Figure 2c) provides a mechanistic explanation: The forward process preferentially corrupts tokens toward embedding-similar alternatives, and the reverse process inherits this inductive bias.

## D DEGENERATE NOISE

Table 14 illustrates the per-row entropy and the dominant transitions of $Q _ { 1 , \phi }$ at t=1 of the most frequent tokens learned by naive joint optimisation of ϕ and θ (D3PM-Reinforce, see Section 6.1). The uniform maximum entropy in this example is $\ln ( V - 1 ) \approx 7 . 6 2$ . From the table we can see that the top rows exhibit near-deterministic transitions, indicating degenerate noise process collapse targeting the high-frequency tokens.

Table 14: Naive joint optimisation of ϕ and θ learns trivial near-deterministic noise processes.
<table><tr><td>Token</td><td>Row Entropy</td><td>Top Target</td><td> $P ( i  j )$ </td></tr><tr><td>•</td><td>0.023</td><td>to</td><td>0.998</td></tr><tr><td>the</td><td>0.062</td><td>t</td><td>0.996</td></tr><tr><td>and</td><td>0.095</td><td>it</td><td>0.993</td></tr><tr><td>1</td><td>0.096</td><td>that</td><td>0.993</td></tr><tr><td>to</td><td>0.111</td><td></td><td>0.992</td></tr><tr><td>it</td><td>0.115</td><td>was</td><td>0.991</td></tr><tr><td>in</td><td>0.216</td><td>1</td><td>0.983</td></tr><tr><td>was</td><td>0.232</td><td>and</td><td>0.981</td></tr><tr><td>t</td><td>0.439</td><td>was</td><td>0.963</td></tr><tr><td>she</td><td>0.552</td><td>that</td><td>0.952</td></tr><tr><td>She</td><td>0.730</td><td>They</td><td>0.935</td></tr><tr><td>so</td><td>0.863</td><td>in</td><td>0.921</td></tr><tr><td>out</td><td>1.733</td><td>up</td><td>0.829</td></tr><tr><td>on</td><td>1.944</td><td>for</td><td>0.806</td></tr></table>