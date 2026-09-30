# INTERPOLATED POLICY DISTILLATION: A CONTROL-LABLE CONTINUUM BETWEEN OFF-POLICY AND ON-POLICY DISTILLATION

Youxu Shi<sup>1</sup> <sup>\*</sup>, Yifan Sun<sup>2</sup> <sup>\*†</sup>, Dacheng Yin<sup>2</sup>, Haomiao Tang<sup>2</sup>, Guangting Wang<sup>2</sup>, Fengyun Rao<sup>2</sup>, Jing LYU<sup>2</sup>, Dong Liu<sup>1</sup> <sup>†</sup>

<sup>1</sup>University of Science and Technology of China <sup>2</sup>WeChat Vision, Tencent Inc.   
syx123@mail.ustc.edu.cn   
dogeliu@ustc.edu.cn   
orionsun@tencent.com

## ABSTRACT

Off-policy and on-policy distillation have traditionally been formulated as separate paradigms, each favoring a different property of distillation trajectories. Teacher-generated (off-policy) traces are typically high-quality but lie far from the student’s distribution, whereas student-generated (on-policy) rollouts are more learnable but often contain erroneous reasoning. We view these paradigms as the endpoints of a policy continuum and posit that a more effective rollout policy may lie in between. We introduce Interpolated Policy Distillation (IPD), which defines the next-token distribution at every decoding step as an explicit linear interpolation between the student and teacher distributions. The interpolation operates at the distribution level, token by token, and its coefficient provides direct control over the balance between trajectory quality and student learnability. Naively sampling from this policy would require sequentially querying the teacher at every token and is thus expensive. To make IPD practical, we accelerate it with a new speculative-decoding rule while exactly preserving the interpolated next-token distribution.At the trajectory level, the resulting rollouts naturally interleave student- and teacher-generated segments. Unlike recent heuristic segment-interleaving methods, however, this interleaving is induced by an exactly realized token-level interpolated policy rather than by hand-designed switching rules. Across text-only and multimodal reasoning benchmarks, IPD consistently outperforms both endpoint policies (SFT and OPD), their conventional two-stage combination (SFT-then-OPD), and recent heuristic segment-interleaving methods, demonstrating that token-level policy interpolation better balances trajectory quality and student learnability.

## 1 INTRODUCTION

Trajectories matter in distilling large language models (LLMs): what a student learns is determined by the token sequences on which it is trained, and distillation methods differ fundamentally in who generates them. Off-policy distillation (Hinton et al., 2015; Kim & Rush, 2016; Shridhar et al., 2023) trains on teacher-generated or fixed-corpus trajectories, whereas on-policy distillation (OPD) (Agarwal et al., 2024; Lu & Lab, 2025) trains on the student’s own rollouts. This choice creates a tension between two properties of a trajectory: its quality, whether it constitutes a correct and valid reasoning process, and its learnability, how readily the current student can absorb its supervision. Teachergenerated trajectories tend to be high-quality but induce a train–inference mismatch (Lin et al., 2020): the student is supervised on prefixes that it would rarely visit on its own. Student-generated rollouts remove this mismatch, but for weak students they drift into erroneous or degenerate prefixes, and a large teacher–student gap can further destabilize training through extreme importance signals, with high local agreement failing to translate into successful reasoning (Xin et al., 2026). Neither endpoint, therefore, is guaranteed to strike the best balance between quality and learnability.

![](images/4cfceebf1fc3551ac63c69b094de031334e58b0384a7cd8028596fa163d29be5.jpg)  
Figure 1: Overview of Interpolated Policy Distillation (IPD). (a) IPD defines a rollout policy $m _ { \gamma } =$ $( 1 - \gamma ) \pi _ { \theta } + \gamma \pi _ { T }$ , interpolating between the student and teacher distributions at each prefix to balance trajectory quality and student learnability. (b) Trajectory generation is accelerated by a tailored speculative sampling scheme: at each step the student proposal is accepted when possible (green), and otherwise replaced by a teacher-guided correction (red). In both cases the emitted token follows exactly $m _ { \gamma } ( \cdot \mid s _ { t } )$ , so the speedup is lossless with respect to the rollout distribution.

This tension suggests that the choice between off-policy and on-policy distillation need not be binary. We instead view the student and teacher policies as the endpoints of a continuum of rollout policies. Moving toward the teacher can improve trajectory quality by correcting unsuitable student decisions, whereas moving too far increases the distributional gap that the student must bridge. The most effective trajectories may therefore come from an intermediate policy—one that incorporates enough teacher guidance to preserve quality while remaining close enough to the student’s evolving distribution to be learnable. This raises our central question: can we construct a rollout policy at a controllable, precisely characterized position between the student and teacher, thereby directly balancing trajectory quality and student learnability?

We answer this with Interpolated Policy Distillation (IPD), which specifies the rollout policy at the distribution level: at each decoding step, the next-token distribution $m _ { \gamma }$ is a linear interpolation of the student and teacher distributions, governed by a single coefficient $\gamma \in [ 0 , 1 ]$ . Larger $\gamma$ moves the policy toward the teacher, whereas smaller $\gamma$ keeps it near the student; at ${ \ ; \gamma = 0 }$ IPD reduces to vanilla OPD, and at $\gamma = 1$ it rolls out from the teacher as in off-policy distillation. We formalize $m _ { \gamma }$ as a policy continuum in Section $^ { 2 , }$ and show how to sample from it exactly and efficiently in Section 3.

In one representative setting, a direct-sampling prototype of IPD substantially outperforms vanilla OPD (Fig. 2), providing controlled evidence that the interpolated rollout policy itself can improve distillation. Direct sampling from $m _ { \gamma } ,$ however, requires a sequential teacher query at every token and is therefore expensive. To make IPD practical, we develop a speculative-sampling rule in which the student proposes tokens in blocks and a single teacher pass per block either accepts them or replaces them with a teacher-guided correction, reducing sequential teacher decoding while preserving the interpolated next-token distribution exactly. We state this distributional guarantee in Section 3.2 and prove it in Appendix A.1.

At the trajectory level, accelerated IPD manifests as an interleaving of accepted student proposals and teacher-guided corrections. Recent teacher-intervention methods (discussed in Section 2.3), produce similar-looking trajectories but decide when and how much the teacher intervenes through hand-designed switching rules; IPD instead specifies the target token-level distribution $m _ { \gamma }$ first and derives a distribution-preserving sampling procedure from it. Notably, the prototype performs no explicit interleaving at all, yet exhibits the same improvement trend as accelerated IPD in Fig. 2, indicating that the gains stem from the interpolated policy rather than from the interleaving pattern.

![](images/72f85d3ecaf63d6994ac61e94109c43003dad63c692ede2334a753750ec47ae7.jpg)  
(a)

![](images/f5eac5cf28b7c5c5282e544638fb1b545c7d047d11d1b97b03d4437b41ac1267.jpg)  
(b)  
Figure 2: (a) On a representative setting (Qwen3-4B → Qwen3-0.6B-Base, evaluated on GSM8K), both the IPD prototype and its accelerated version outperform OPD by clear margins. In particular, the prototype applies the vanilla OPD objective to every token ,yet it recovers most of the gain, indicating the gains mainly stem from the interpolated policy. (b) Effect of the interpolation coefficient γ: a proxy for learnability falls monotonically as γ grows, while rollout quality (scored by an LLM judge) rises, exposing the trade-off that γ balances.

Because the sampler records whether each emitted token was an accepted proposal or a teacherguided correction, a single rollout can combine both forms of supervision: IPD applies direct likelihood supervision to teacher-guided corrections and OPD supervision to accepted student proposals, rather than separating the two into successive training stages. Across text-only and multimodal reasoning benchmarks, and for both base and instruction-tuned students, IPD consistently outperforms not only both endpoint baselines—off-policy SFT and vanilla OPD—but also their conventional two-stage combination, SFT-then-OPD, and recent segment-interleaving methods. These results show that IPD’s gains arise not merely from combining off-policy and on-policy supervision, but from combining them through an explicitly defined token-level policy interpolation.

Contributions. (1) A unified policy continuum. We formulate off-policy distillation and OPD as the endpoints of a single policy continuum, exposing intermediate rollout policies as an explicit design space for balancing trajectory quality and student learnability. (2) Interpolated Policy Distillation. We instantiate this continuum as a per-token interpolation of the student and teacher distributions, and accelerate it with a speculative-sampling rule that preserves the interpolated distribution exactly. The same accept–reject decisions also determine which supervision applies at each token. (3) Empirical validation. Across text-only and multimodal reasoning benchmarks and both base and instruction-tuned students, IPD consistently outperforms off-policy SFT, vanilla OPD, their conventional two-stage combination, and recent segment-interleaving methods.

## 2 A TRAJECTORY-POLICY CONTINUUM

This section formalizes the trajectory-policy continuum and reviews related distillation methods through this unified perspective. Throughout, π and π denote the student and teacher policies and $\boldsymbol { s } _ { t } = \left( x , y _ { < t } \right)$ denotes the prefix state at decoding step t.

## 2.1 THE POLICY CONTINUUM

At each decoding step, given prefix $\boldsymbol { s } _ { t } = \left( x , y _ { < t } \right)$ , we define the next-token distribution

$$
m _ { \gamma } ( \cdot \mid s _ { t } ) = \left( 1 - \gamma \right) \pi _ { \theta } ( \cdot \mid s _ { t } ) + \gamma \pi _ { T } ( \cdot \mid s _ { t } ) , \qquad \gamma \in [ 0 , 1 ] ,\tag{1}
$$

which induces the trajectory distribution $\begin{array} { r } { P _ { \gamma } ( y ~ \mid ~ x ) ~ = ~ \prod _ { t = 1 } ^ { | y | } m _ { \gamma } ( y _ { t } ~ \mid ~ s _ { t } ) } \end{array}$ . Note that $P _ { \gamma } \neq$ $( 1 - \gamma ) P _ { \pi _ { \theta } } + \gamma P _ { \pi _ { T } }$ : interpolating the conditional distributions at every prefix is not the same as interpolating the two complete trajectory distributions, which would amount to choosing one of the two policies once and generating the entire trajectory from it. The coefficient γ sets the rollout policy’s position on the continuum: $\gamma = 0$ and $\gamma = 1$ recover the student and teacher rollout policies, respectively, while $\gamma \in ( 0 , 1 )$ defines the intermediate policies explored by IPD.

## 2.2 THE TWO ENDPOINTS AND THE INTERIOR

Motivated by trust region theory (Schulman et al., 2015), we assess the learnability of a rollout policy through its proximity to the student’s own distribution. We measure this proximity by the total variation distance $\begin{array} { r } { \dot { D } _ { \mathrm { T V } } ( u , \mathbf { \bar { \Psi } } w ) = \frac { 1 } { 2 } \sum _ { v } | u ( v ) - w ( v ) - \psi ( \mathbf { \bar { \Psi } } ) | } \end{array}$ | between two distributions over the vocabulary. For $m _ { \gamma }$ it admits an exact form at every prefix:

$$
D _ { \mathrm { T V } } ( m _ { \gamma } , \pi _ { \theta } ) = \gamma D _ { \mathrm { T V } } ( \pi _ { \theta } , \pi _ { T } ) , D _ { \mathrm { T V } } ( m _ { \gamma } , \pi _ { T } ) = ( 1 - \gamma ) D _ { \mathrm { T V } } ( \pi _ { \theta } , \pi _ { T } ) .\tag{2}
$$

The two distances sum to $D _ { \mathrm { T V } } ( \pi _ { \theta } , \pi _ { T } )$ and are linear in $\gamma ,$ so the coefficient places the rollout policy at a precisely known position between the two endpoints rather than merely somewhere between them. Eq. (2) is a statement about individual decoding steps, and it should not be read as bounding the distance between the resulting trajectory distributions. Because every token is conditioned on the preceding ones, a single deviation alters every subsequent conditional. A small $\gamma$ therefore does not imply that trajectories sampled from $m _ { \gamma }$ stay close to those the student would have produced: modest teacher influence per token can accumulate into substantially different reasoning paths, which is what makes the interior of the continuum worth exploring even for small $\gamma .$

The teacher endpoint. $\mathbf { A } \mathfrak { t } \gamma = 1$ the rollout policy is $\pi _ { T }$ , attaining the largest distance from the student. Teacher rollouts typically provide high-quality reasoning (Hinton et al., 2015; Kim & Rush, 2016; Shridhar et al., 2023), but they visit prefixes and select continuations that are unlikely under the student. High trajectory quality therefore does not imply high learnability: the supervision the teacher provides may be difficult for the student to absorb when the teacher–student gap is large.

The student endpoint. $\mathrm { A t } \gamma = 0$ the rollout policy is $\pi _ { \theta }$ itself, so the trajectories match the states the student encounters at inference. This eliminates rollout-distribution shift but offers no guarantee of quality. A weak student may assign substantial probability to erroneous reasoning steps and enter invalid or degenerate prefixes (Li et al., 2026b), and because each token is conditioned on the preceding ones, early mistakes compound and steer the rollout increasingly far from a valid reasoning path. A trajectory can therefore be fully on-policy yet provide a poor basis for distillation, with local agreement coexisting with globally unsuccessful reasoning, as observed in the agreement trap (Xin et al., 2026).

Why the interior? The teacher endpoint thus sacrifices learnability, whereas the student endpoint sacrifices quality. Intermediate values $\gamma \in \mathsf { \Gamma } ( 0 , 1 )$ introduce teacher guidance without moving all the way to the teacher endpoint. As shown in Figure 2(b), trajectory quality, evaluated by an LLM judge, and learnability, measured by proximity to π<sub>θ</sub> (metric is detailed in Appendix $\mathbf { A } . { \overset { } { 3 } } )$ , move in opposite directions as $\gamma$ varies, empirically revealing a trade-off between the two properties. This motivates a policy that balances them better than either endpoint. How to sample from $m _ { \gamma }$ efficiently while preserving its specified distribution is the subject of Section 3.

## 2.3 RELATED WORK

Hybrid student–teacher rollouts. Several recent methods combine student and teacher generation within the same trajectory. SKD (Xu et al., 2025) replaces student proposals outside the teacher’s top-K predictions; Relay-OPD (Xu et al., 2026) detects failed prefixes and temporarily transfers generation to the teacher; CA-OPD (Li et al., 2026a) corrects unreliable tokens according to teacher confidence and progressively relaxes intervention; and, in the related RL setting, MInTRL (Chen et al., 2026) inserts short judge-identified corrections before returning control to the student. These methods combine the two models through method-specific rules that determine when and how much the teacher intervenes, leaving the induced next-token distribution implicit. IPD also produces hybrid student–teacher rollouts, but defines the combination at the distribution level: it first specifies $m _ { \gamma }$ in Eq. (1) and then derives a sampling procedure that preserves this distribution. The resulting interleaving is therefore governed by an explicit token-level policy rather than by hand-designed intervention rules.

## 3 INTERPOLATED POLICY DISTILLATION

## 3.1 THE IPD PROTOTYPE AND ITS LIMITATIONS

Trajectory rollout policy. Following the discussion in Section 2, an effective rollout policy should balance trajectory quality and student learnability. We therefore adopt the interpolated policy $m _ { \gamma }$ of Eq. 1 as the rollout policy, and sample directly from it by querying both models at every decoding step. Without loss of generality, we choose $\gamma = 0 . 1$ unless otherwise noted (see Section 4.3 for an ablation over γ)

Supervision. We supervise the prototype with the vanilla OPD objective, unchanged: a PPO-style surrogate whose per-token reverse-KL term is estimated by the $k _ { 1 }$ estimator (Schulman, 2020). This is reasonable at the small $\gamma$ we use: by $\mathrm { E q . } ( 2 ) , m _ { \gamma } ( \cdot \mid s _ { t } )$ stays within $\gamma D _ { \mathrm { T V } } ( \pi _ { \theta } ( \cdot \mid s _ { t } ) , \pi _ { T } ( \cdot \mid s _ { t } ) )$ of the student at every prefix, close enough for student-side supervision to apply. A mismatch nonetheless remains: the importance ratio is formed against the student snapshot $\pi _ { \theta _ { \mathrm { o l d } } }$ that collected the rollout, exactly as in on-policy training, so it corrects for the staleness of the snapshot but not for the tokens having been drawn from $m _ { \gamma }$ rather than from $\pi _ { \theta _ { \mathrm { o l d } } }$ itself. This discrepancy is $O ( \gamma )$ in total variation, and we leave it uncorrected.

The prototype already improves over vanilla OPD. We train Qwen3-0.6B-Base with the prototype, using the same training hyperparameters as vanilla OPD (Section 4). As shown in Figure 2(a), it substantially outperforms vanilla OPD, converging faster and reaching higher final accuracy. Since the prototype inherits the vanilla OPD objective unchanged, the two runs share their objective, data, and number of optimization steps, and differ only in the policy that generated the rollouts. The gain is therefore attributable to the interpolated policy alone: moving into the interior of the continuum helps even before any change to the supervision.

Two limitations. Training the prototype nonetheless reveals two limitations, both stemming from its sampling directly from $m _ { \gamma }$

\- Inefficiency. Sampling from $m _ { \gamma }$ requires a teacher forward pass at every decoding step, so teacher computation is strictly sequential and cannot be amortized across positions; maintaining two generation engines simultaneously adds further coordination overhead.

\- Undifferentiated supervision. Undifferentiated supervision. Rollouts from $m _ { \gamma }$ contain both tokens the student would plausibly generate itself and tokens the student would rarely generate on its own (but which the teacher favors). While the former suit an on-policy divergence objective, the latter do not: on these tokens the log ratios in the divergence objective have large magnitude and can destabilize optimization, so a likelihood-based objective is preferable. Direct sampling from $m _ { \gamma }$ does not indicate which kind each token is, so the prototype applies a single objective throughout.

Sections 3.2 and 3.3 address these limitations in turn. Speculative sampling removes the per-token sequential teacher query and, at no additional cost, labels each emitted token as an accepted student proposal or a teacher-guided correction; these labels in turn make differentiated supervision possible.

## 3.2 SPECULATIVE SAMPLING FROM THE INTERPOLATED POLICY

Speculative sampling (Leviathan et al., 2023) is a general recipe for LLM inference acceleration: given a cheap proposal distribution and any target distribution, one drafts tokens from the proposal and then accepts or corrects them, obtaining exact samples from the target while querying it only once per block of drafted tokens. We take the student as the proposal and the interpolated policy $m _ { \gamma }$ of Eq. 1 as the target policy, i.e., the verifier.

Trajectory construction. Starting from the current prefix, the student drafts k tokens autoregressively, and the verifier evaluates all draft positions in a single forward pass (Figure 3). Writing $p ( v ) = \pi _ { \theta } ( v \mid s )$ and $q ( v ) = \pi _ { T } ( v \mid s )$ for the student and teacher distributions at a draft prefix s, a proposed token v is accepted with probability

$$
a _ { \gamma } ( v \mid s ) = \operatorname* { m i n } \left\{ 1 , \frac { m _ { \gamma } ( v \mid s ) } { p ( v ) } \right\} = \operatorname* { m i n } \left\{ 1 , 1 - \gamma + \gamma \frac { q ( v ) } { p ( v ) } \right\} .\tag{3}
$$

![](images/754974a772e60f4dd84b9c38269f00675a9cc77db87794e63ec04797816d5fce.jpg)  
Figure 3: Accelerated IPD: speculative sampling (Section 3.2) with verification-aware supervision (Section 3.3). The student drafts k tokens sequentially and the teacher scores the block in a single forward pass, giving the target policy $m _ { \gamma }$ at every draft position. Proposals are accepted with probability $a _ { \gamma } ~ ( \operatorname { E q } . 3 ) ;$ ; the first rejected one is replaced by a teacher-guided correction drawn from the residual $\dot { r } _ { \gamma } \left( \mathrm { E q . ~ } 4 \right)$ and the draft suffix is discarded. Every generated token therefore follows $m _ { \gamma } ( \cdot \mid s _ { t } )$ exactly.The same decisions determine supervision: accepted proposals (green) receive the OPD objective (Eq. 8), corrections (red) direct likelihood supervision (Eq. 9).

The ratio is evaluated only for proposed tokens, for which $p ( v ) > 0$ , and all acceptance probabilities are computed in parallel with independent acceptance draws.

IPD retains the longest accepted prefix of the draft. At the first rejection, the rejected proposal is replaced by a token drawn from the residual distribution at the same prefix,

$$
r _ { \gamma } ( v \mid s ) = \frac { [ m _ { \gamma } ( v \mid s ) - p ( v ) ] _ { + } } { \sum _ { u } [ m _ { \gamma } ( u \mid s ) - p ( u ) ] _ { + } } , \qquad [ a ] _ { + } = \operatorname * { m a x } ( a , 0 ) .\tag{4}
$$

The remaining draft tokens are discarded, since they were conditioned on the rejected proposal. After a rejection or a full acceptance, the draft window advances to the next unfilled position, and drafting resumes from the updated prefix until the trajectory is complete. We record the outcome of each step as $z _ { t } \in \{ 0 , 1 \}$ , with $z _ { t } = 1 \operatorname { i f } y _ { t }$ is an accepted student proposal and $z _ { t } = 0$ if it is a teacher-guided correction.

What $\gamma$ controls. Since any two token distributions both sum to one, the total variation distance can equivalently be written in positive-part form, $\begin{array} { r } { D _ { \mathrm { T V } } ( u , w ) = \sum _ { v } [ u ( v ) - w ( v ) ] _ { + } } \end{array}$ , which isolates exactly the tokens on which one distribution places more mass than the other. In these terms, a proposal is rejected only when the student assigns it higher probability than the teacher, and

$$
\operatorname* { P r } ( z _ { t } = 0 \mid s _ { t } = s ) = D _ { \mathrm { T V } } ( p , m _ { \gamma } ( \cdot \mid s ) ) = \gamma D _ { \mathrm { T V } } ( p , q ) .\tag{5}
$$

Moreover, since $m _ { \gamma } - p = \gamma ( q - p )$ , the coefficient $\gamma$ cancels in Eq. (4), giving

$$
r _ { \gamma } ( v \mid s ) = { \frac { [ q ( v ) - p ( v ) ] _ { + } } { D _ { \mathrm { T V } } ( p , q ) } } = r _ { 1 }\tag{6}
$$

whenever rejection has positive probability. Thus γ sets how often the teacher intervenes, while what the intervention does is determined by where the teacher and student disagree, independently of $\gamma \colon$ corrections concentrate at prefixes of large $D _ { \mathrm { T V } } ( p , q )$ and are supported only on tokens the teacher favors more than the student. Equation $( 5 )$ also explains the efficiency gain, as the expected number of consecutive accepted tokens scales as $1 / \big ( \gamma D _ { \mathrm { T V } } ( p , q ) \big )$ : small $\gamma$ yields long accepted runs and correspondingly few teacher forward passes. More details are shown in Appendix A.1.

## 3.3 VERIFICATION-AWARE SUPERVISION

Recall that the prototype had to apply a single objective to every token of a rollout (Section 3.1), not because one objective suits them all, but because sampling from $m _ { \gamma }$ returns only the token drawn and not which kind it is. Speculative sampling removes this obstacle at no additional cost: the label $z _ { t }$ records whether $y _ { t }$ was an accepted student proposal or a teacher-guided correction.

Table 1: Comparison of different distillation methods across text reasoning benchmarks. For each benchmark, we report the accuracy (Avg@32).
<table><tr><td></td><td colspan="5">Qwen3-1.7B-Base ← Qwen3-4B</td><td colspan="5">Qwen3-0.6B-Base ← Qwen3-4B</td><td></td></tr><tr><td>Method</td><td>AIME24</td><td>AIME25</td><td>AIME26</td><td>AMC23</td><td>Olympiad</td><td> $\operatorname { A v g } .$ </td><td></td><td>GSM8k MATH-500</td><td>AMC23</td><td>Olympiad</td><td> $\operatorname { A v g } .$ </td></tr><tr><td>Student</td><td>5.42</td><td>2.92</td><td>1.25</td><td>23.44</td><td>17.89</td><td>10.18</td><td>12.53</td><td>7.43</td><td>7.19</td><td>7.91</td><td>8.77</td></tr><tr><td>SFT</td><td>4.92</td><td>2.04</td><td>1.89</td><td>25.01</td><td>19.27</td><td>10.63</td><td>34.15</td><td>16.65</td><td>5.31</td><td>4.69</td><td>15.20</td></tr><tr><td>GRPO</td><td>5.42</td><td>1.67</td><td>3.75</td><td>33.44</td><td>20.67</td><td>12.99</td><td>60.54</td><td>37.93</td><td>20.25</td><td>8.76</td><td>31.87</td></tr><tr><td>vanilla OPD</td><td>5.83</td><td>6.67</td><td>5.00</td><td>33.75</td><td>22.31</td><td>14.71</td><td>53.08</td><td>31.52</td><td>16.56</td><td>10.17</td><td>27.83</td></tr><tr><td>SKD</td><td>4.28</td><td>5.96</td><td>3.76</td><td>26.71</td><td>19.67</td><td>12.08</td><td>51.46</td><td>31.29</td><td>18.27</td><td>12.19</td><td>28.30</td></tr><tr><td>Relay-OPD</td><td>10.28</td><td>7.12</td><td>5.89</td><td>32.17</td><td>22.89</td><td>15.67</td><td>67.46</td><td>43.15</td><td>22.81</td><td>14.41</td><td>36.96</td></tr><tr><td>IPD (ours)</td><td>10.58</td><td>6.88</td><td>6.25</td><td>35.31</td><td>24.17</td><td>16.64</td><td>70.02</td><td>44.68</td><td>24.12</td><td>15.06</td><td>38.47</td></tr></table>

Accepted proposals. Accepted tokens tend to lie in the student’s own high-probability region, preserving a strong on-policy character. Acceptance also naturally attenuates extreme log-ratio signals, making these tokens well suited to OPD supervision. Let $\pi _ { \theta _ { \mathrm { o l d } } }$ denote the student policy used to collect the rollout. We use the per-token reverse-KL signal given by the $k _ { 1 }$ estimator (Schulman, 2020) as a fixed advantage,

$$
\widehat { A } _ { t } ^ { \mathrm { K D } } = \log \pi _ { T } ( y _ { t } \mid s _ { t } ) - \log \pi _ { \theta _ { \mathrm { o l d } } } ( y _ { t } \mid s _ { t } ) ,\tag{7}
$$

and optimize a clipped surrogate

$$
\ell _ { t } ^ { \mathrm { a c c } } ( \theta ) = - \operatorname* { m i n } \left\{ \rho _ { t } ( \theta ) \widehat { A } _ { t } ^ { \mathrm { K D } } , \ : \mathrm { c l i p } ( \rho _ { t } ( \theta ) , 1 - \epsilon , 1 + \epsilon ) \widehat { A } _ { t } ^ { \mathrm { K D } } \right\} ,\tag{8}
$$

where $\rho _ { t } ( \theta ) = \pi _ { \theta } ( y _ { t } \mid s _ { t } ) / \pi _ { \theta _ { \mathrm { o l d } } } ( y _ { t } \mid s _ { t } )$ and ϵ is the clipping coefficient.

Residual corrections. Corrections are drawn from $[ q - p ] _ { + }$ by $\operatorname { E q . }$ . 6 and are thus, by construction, tokens the student under-weights relative to the teacher. On these tokens a ratio-based objective can be unstable, since $\widehat { A } _ { t } ^ { \mathrm { K D } }$ grows sharply as the student probability decreases. We instead apply direct likelihood supervision,

$$
\ell _ { t } ^ { \mathrm { r e j } } ( \theta ) = - \log \pi _ { \theta } ( y _ { t } \mid s _ { t } ) ,\tag{9}
$$

whose gradient magnitude remains bounded and which supplies a mass-covering signal exactly where the student fails to cover the teacher—the regime in which a mode-seeking divergence provides little pressure.

Objective. Let T denote the set of generated positions in a rollout. The overall objective is

$$
\mathcal { L } _ { \mathrm { I P D } } ( \theta ) = \frac { 1 } { | T | } \sum _ { t \in \mathcal { T } } \left[ z _ { t } \ell _ { t } ^ { \mathrm { a c c } } ( \theta ) + ( 1 - z _ { t } ) \ell _ { t } ^ { \mathrm { r e j } } ( \theta ) \right] .\tag{10}
$$

SFT-style and OPD-style supervision are thereby combined within a single rollout, with the speculative decisions determining which signal applies at each token. $\mathrm { A t } \gamma = 0$ every proposal is accepted and Eq. 10 reduces exactly to vanilla OPD; as γ increases, corrections become more frequent and an increasing fraction of supervision takes the form of likelihood training on teacher-corrected tokens.

Because speculative sampling preserves $m _ { \gamma } ,$ the accelerated variant trains on the same rollout distribution as the prototype of Section 3.1 and differs from it only in supervision. As shown in Figure $2 ( \mathrm { a } )$ , it matches and slightly exceeds the prototype, indicating that the additional gain comes from differentiated supervision rather than from any change in the trajectories themselves.

## 4 EXPERIMENTS

## 4.1 EXPERIMENTAL SETUP

Text-only reasoning. For weak student initializations, we use Qwen3-0.6B-Base and Qwen3- 1.7B-Base as students, with Qwen3-4B as the teacher (Yang et al., 2025). For a stronger initialization, we use DeepSeek-R1-Distill-Qwen-1.5B (DeepSeek-AI, 2025) as the student and Skywork-OR1-7B (He et al., 2025) as the teacher. All three settings use the same 25,600 math questions sampled from OpenThoughts3 (Guha et al., 2025). Depending on the setting, we evaluate on GSM8K (Cobbe et al., 2021), MATH-500 (Lightman et al., 2023), AMC 2023 (Math-AI, 2023), OlympiadBench (He et al., 2024), and AIME 2024–2026 (Jia, 2024; Math-AI, 2025).

![](images/10feb9f8f424b4dd737a136a8c39f2c5b5a551a43acf61c874820f2b57a2e2da.jpg)  
Figure 4: Performance on Math-Vista.

![](images/6524f33489181d99dcdf79143e525823741a70b1332abf701fb971bb6ade1b6b.jpg)  
Figure 5: Ablation results averaged on GSM8k, MATH-500 and AMC.

Multimodal reasoning. We consider two settings using Qwen3-VL-8B-Instruct as the teacher. The first uses Qwen3-VL-2B-Instruct (Bai et al., 2025) as the student. The second removes the student’s last three decoder layers to create a more challenging initialization, allowing us to assess capability recovery after pruning. Both settings are trained on Innovator-VL-RL-172K (Wen et al., 2026) and evaluated on MathVision (Wang et al., 2024), MathVista (Lu et al., 2024), MMStar (Chen et al., 2024), WeMath (Qiao et al., 2025), and MMMU-Pro (Yue et al., 2025).

## 4.2 MAIN RESULTS

## 4.2.1 UNIMODAL DISTILLATION

Table 1 reports text reasoning results with two base students distilled from Qwen3-4B. IPD achieves the highest average accuracy in both settings, improving over vanilla OPD by 1.93 points for Qwen3- 1.7B-Base and 10.64 points for Qwen3-0.6B-Base. Among methods that incorporate teacher intervention, SKD shows mixed results relative to vanilla OPD, whereas Relay-OPD consistently improves average accuracy. IPD further surpasses Relay-OPD by 0.97 and 1.51 points, respectively, and achieves the best performance on eight of the nine benchmark–student combinations. In fact, results from Relay-OPD are obtained using the recommended default configurations from its respective works. As shown in the Appendix A.4, Relay-OPD suffers from training collapse when run under exactly the same setting as ours, highlighting its sensitivity to the training setup. These results support the effectiveness of explicitly controlling the rollout policy through interpolation.

We additionally evaluate DeepSeek-R1-Distill-Qwen-1.5B to examine whether these gains extend to a strong student that has undergone extensive post-training. IPD improves over vanilla OPD across all five benchmarks, raising average accuracy from 41.37 to 42.18, with detailed results provided in the Appendix A.5. Together, these results demonstrate the effectiveness of IPD across both base and extensively post-trained students.

## 4.2.2 MULTIMODAL DISTILLATION

Table 2 shows that IPD delivers substantial gains not only on LLM benchmarks but also in multimodal settings. Across multiple multimodal reasoning benchmarks, IPD consistently outperforms vanilla OPD. Notably, under prolonged training, vanilla OPD exhibits progressive performance degradation, whereas IPD maintains stable performance and converges to a stronger point, as shown in Figure 4. To evaluate IPD under a more challenging initialization, we remove the final three decoder layers from Qwen3-VL-2B-Instruct and use distillation to recover its capabilities. The ad vantage of IPD becomes considerably larger in this setting: it achieves an average accuracy of 44.64 compared with 27.90 for OPD, nearly restoring the original unpruned model’s performance (45.87). This demonstrates that IPD can effectively recover capabilities lost through pruning without an additional recovery stage.

Table 2: Results across multimodal reasoning benchmarks. Avg@32 accuracy is reported.
<table><tr><td rowspan="2">Method</td><td colspan="6">Qwen3-VL-2B ← Qwen3-VL-8B</td></tr><tr><td>MathVision</td><td>MathVista</td><td>MMStar</td><td>WeMath</td><td>MMMU-Pro</td><td> $\operatorname { A v g } .$ </td></tr><tr><td>student</td><td>15.46</td><td>61.22</td><td>57.03</td><td>54.81</td><td>40.82</td><td>45.87</td></tr><tr><td>OPD</td><td>19.57</td><td>64.89</td><td>57.59</td><td>59.32</td><td>41.05</td><td>48.48</td></tr><tr><td>IPD (ours)</td><td>20.48</td><td>66.70</td><td>58.97</td><td>62.44</td><td>42.51</td><td>50.22</td></tr><tr><td>student-pruned</td><td>4.44</td><td>34.63</td><td>29.41</td><td>16.07</td><td>17.41</td><td>20.39</td></tr><tr><td>OPD</td><td>8.68</td><td>44.11</td><td>34.43</td><td>31.87</td><td>20.39</td><td>27.90</td></tr><tr><td>IPD (ours)</td><td>15.08</td><td>59.27</td><td>55.56</td><td>54.09</td><td>39.21</td><td>44.64</td></tr></table>

## 4.3 ABLATION STUDY

Ablation of fusion coefficient $\gamma$ The fusion coefficient $\gamma$ controls the strength of teacher intervention during trajectory construction. Figure 5 shows a clear intermediate optimum: performance improves markedly at $\gamma = 0 . 0 5$ and peaks at $\gamma = 0 . 1$ . Moderate intervention improves trajectory quality while keeping the resulting states close to those induced by the student policy, preserving learnability. $\mathbf { A s } \ \gamma$ increases further, performance generally declines as trajectories shift toward the teacher distribution. Although $\gamma = 1$ still outperforms vanilla $\mathrm { O P D } ,$ it falls substantially short of $\gamma = 0 . 1$ . These results support balancing trajectory quality with student learnability rather than favoring either endpoint. We further compare schedules that decrease γ from 0.5 to 0.1 or increase it from 0.5 to 0.9 during training. The decreasing schedule performs better, suggesting that gradually shifting the rollout policy toward the student is more effective than increasing teacher intervention in this setting (see Appendix 7 for details)

Sensitivity to SFT cold start. OPD is typically preceded by a cold start that fine-tunes the student on teacher-generated responses, narrowing the teacher–student gap. Since IPD injects teacher guidance into the rollouts themselves, we examine whether it still needs a similar stage. We generate teacher responses with Qwen3-4B on Nemotron-v2 prompts (Basant et al., 2025) and fine-tune Qwen3-0.6B-Base on varying fractions of them before distillation. Figure 6 compares training dynamics and final accuracy across cold-start strengths. Vanilla OPD tracks its initialization closely, whereas IPD is largely insensitive to it: different cold-start ratios lead to similar final accuracy, and without any cold start IPD already surpasses the best result attained by $\mathrm { ^ { * } S F T + O P D ^ { 3 } }$ (46.69 vs. 45.41). A plausible explanation is that teacher guidance enters $m _ { \gamma }$ at every step, so IPD obtains high-quality yet learnable trajectories even from a base student—what the cold start is meant to provide for OPD. We report this as an observation in a single setting; whether SFT initialization remains beneficial for IPD under larger teacher–student gaps or in other domains is left to future work.

![](images/d89e852d51af29c9580221c8dffdc076ba1a4532ffc254a09c2885292e4d69d9.jpg)  
(a)

![](images/c57363bbfee7e96201b0385260f5233d2e969791a64a333725c1e2c8fa71382f.jpg)  
(b)  
Figure 6: Sensitivity to SFT cold starts. (a) Training curves and (b) final accuracy across SFT cold-start strengths. Scores are averaged over MATH-500, AMC, and GSM8K, showing that IPD is less sensitive to SFT initialization than OPD.

## 5 CONCLUSION

We introduce Interpolated Policy Distillation (IPD), which connects off-policy and on-policy distiilation through an explicit token-level policy continuum, enabling controllable balancing of trajectory quality and student learnability. Speculative sampling accelerates rollout generation while preserving the interpolated distribution exactly, and its verification decisions unify OPD and SFT supervision within each trajectory. Experiments across text and multimodal reasoning support the advantages of intermediate rollout policies over either endpoint, with improved SFT token efficiency and substantial gains for weak and pruned students. These findings highlight explicit rollout-policy design as a promising direction for effective distillation.

## AI USE STATEMENT

We used generative AI tools to assist with translation, discussions of theoretical formulations and methodological choices, and the interpretation and presentation of experimental results. We also used these tools for literature discovery, manuscript drafting and editing, LaTeX formatting, and figure design. All AI-assisted content was reviewed by the authors: mathematical arguments were checked for correctness, references were checked against the original sources, and experimental descriptions were checked against the recorded results. We take full responsibility for the final content of this work, including all text, claims, and artifacts produced with AI assistance.

## ETHICS STATEMENT

This work studies distillation for language and vision-language models using existing models, datasets, and benchmarks. It does not involve human participants or the collection of new personal data. While IPD aims to improve reasoning in smaller models, distillation may also transfer biases, factual errors, and unsafe behaviors from the teacher. Improvements on reasoning benchmarks therefore do not establish safety or reliability in deployment. Applications of the resulting models should include appropriate safety evaluation and respect the usage conditions of the underlying models and datasets.

## REPRODUCIBILITY STATEMENT

To support reproducibility, the method section specifies the interpolated rollout policy, speculative sampling procedure, and token-level supervision objectives. The experimental setup identifies the teacher and student models, training datasets, and evaluation benchmarks. Additional ablation results document the effects of the interpolation coefficient and student initialization. More details please find in Appendix.

## REFERENCES

Rishabh Agarwal et al. On-policy distillation of language models: Learning from self-generated mistakes. In International Conference on Learning Representations (ICLR), 2024.

Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, Wenbin Ge, Zhifang Guo, Qidong Huang, Jie Huang, Fei Huang, Binyuan Hui, Shutong Jiang, Zhaohai Li, Mingsheng Li, Mei Li, Kaixin Li, Zicheng Lin, Junyang Lin, Xuejing Liu, Jiawei Liu, Chenglong Liu, Yang Liu, Dayiheng Liu, Shixuan Liu, Dunjie Lu, Ruilin Luo, Chenxu Lv, Rui Men, Lingchen Meng, Xuancheng Ren, Xingzhang Ren, Sibo Song, Yuchong Sun, Jun Tang, Jianhong Tu, Jianqiang Wan, Peng Wang, Pengfei Wang, Qiuyue Wang, Yuxuan Wang, Tianbao Xie, Yiheng Xu, Haiyang Xu, Jin Xu, Zhibo Yang, Mingkun Yang, Jianxin Yang, An Yang, Bowen Yu, Fei Zhang, Hang Zhang, Xi Zhang, Bo Zheng, Humen Zhong, Jingren Zhou, Fan Zhou, Jing Zhou, Yuanzhi Zhu, and Ke Zhu. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025.

Aarti Basant, Abhijit Khairnar, Abhijit Paithankar, Abhinav Khattar, Adithya Renduchintala, Aditya Malte, Akhiad Bercovich, Akshay Hazare, Alejandra Rico, Aleksander Ficek, et al. Nvidia nemotron nano 2: An accurate and efficient hybrid mamba-transformer reasoning model. arXiv preprint arXiv:2508.14444, 2025.

Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Jiaqi Wang, Yu Qiao, Dahua Lin, et al. Are we on the right way for evaluating large vision-language models? Advances in Neural Information Processing Systems, 37:27056–27087, 2024.

Mingyu Chen, Yefan Tao, Gerald Friedland, Xuezhou Zhang, and Chris Kong. Mintrl: Off-policy intervention can boost on-policy rl, 2026. URL https://arxiv.org/abs/2609.12419.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.

DeepSeek-AI. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning, 2025. URL https://arxiv.org/abs/2501.12948.

Etash Guha, Ryan Marten, Sedrick Keh, Negin Raoof, Georgios Smyrnis, Hritik Bansal, Marianna Nezhurina, Jean Mercat, Trung Vu, Zayne Sprague, Ashima Suvarna, Benjamin Feuer, Liangyu Chen, Zaid Khan, Eric Frankel, Sachin Grover, Caroline Choi, Niklas Muennighoff, Shiye Su, Wanjia Zhao, John Yang, Shreyas Pimpalgaonkar, Kartik Sharma, Charlie Cheng-Jie Ji, Yichuan Deng, Sarah Pratt, Vivek Ramanujan, Jon Saad-Falcon, Jeffrey Li, Achal Dave, Alon Albalak, Kushal Arora, Blake Wulfe, Chinmay Hegde, Greg Durrett, Sewoong Oh, Mohit Bansal, Saadia Gabriel, Aditya Grover, Kai-Wei Chang, Vaishaal Shankar, Aaron Gokaslan, Mike A. Merrill, Tatsunori Hashimoto, Yejin Choi, Jenia Jitsev, Reinhard Heckel, Maheswaran Sathiamoorthy, Alexandros G. Dimakis, and Ludwig Schmidt. Openthoughts: Data recipes for reasoning models, 2025. URL https://arxiv.org/abs/2506.04178.

Chaoqun He, Renjie Luo, Yuzhuo Bai, Shengding Hu, Zhen Leng Thai, Junhao Shen, Jinyi Hu, Xu Han, Yujie Huang, Yuxiang Zhang, et al. Olympiadbench: A challenging benchmark for promoting agi with olympiad-level bilingual multimodal scientific problems. arXiv preprint arXiv:2402.14008, 2024.

Jujie He, Jiacai Liu, Chris Yuhao Liu, Rui Yan, Chaojie Wang, Peng Cheng, Xiaoyu Zhang, Fuxiang Zhang, Jiacheng Xu, Wei Shen, Siyuan Li, Liang Zeng, Tianwen Wei, Cheng Cheng, Bo An, Yang Liu, and Yahui Zhou. Skywork open reasoner 1 technical report. arXiv preprint arXiv:2505.22312, 2025.

Geoffrey Hinton, Oriol Vinyals, and Jeffrey Dean. Distilling the knowledge in a neural network. In NIPS Deep Learning and Representation Learning Workshop, 2015. URL http://arxiv. org/abs/1503.02531.

Maxwell Jia. Aime 2024. Hugging Face Dataset, 2024. URL https://huggingface.co/ datasets/Maxwell-Jia/AIME\_2024.

Yoon Kim and Alexander M Rush. Sequence-level knowledge distillation. In Proceedings of the 2016 conference on empirical methods in natural language processing, pp. 1317–1327, 2016.

Yaniv Leviathan, Matan Kalman, and Yossi Matias. Fast inference from transformers via speculative decoding. In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett (eds.), Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pp. 19274–19286. PMLR, 23–29 Jul 2023. URL https://proceedings.mlr.press/ v202/leviathan23a.html.

Menghao Li, Linjie Mu, Yin Wang, Haotian Hu, Yannian Gu, Lujiayi Xue, Liujian Tang, Yu Zhang, and Fanyi Wang. Ca-opd: Confidence-aware on-policy distillation for structured visual prediction, 2026a. URL https://arxiv.org/abs/2609.02401.

Yaxuan Li, Yuxin Zuo, Bingxiang He, Jinqian Zhang, Chaojun Xiao, Cheng Qian, Tianyu Yu, Huanang Gao, Wenkai Yang, Zhiyuan Liu, et al. Rethinking on-policy distillation of large language models: Phenomenology, mechanism, and recipe. arXiv preprint arXiv:2604.13016, 2026b.

Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. arXiv preprint arXiv:2305.20050, 2023.

Alexander Lin, Jeremy Wohlwend, Howard Chen, and Tao Lei. Autoregressive knowledge distillation through imitation learning. In Proceedings ofthe 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 6121–6133, 2020.

Kevin Lu and Thinking Machines Lab. On-policy distillation. Thinking Machines Lab: Connectionism, 2025. doi: 10.64434/tml.20251026. https://thinkingmachines.ai/blog/on-policy-distillation.

Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. In International Conference on Learning Representations (ICLR), 2024.

Math-AI. Amc 2023. Hugging Face Dataset, 2023. URL https://huggingface.co/ datasets/math-ai/amc23.

Math-AI. Aime 2025. Hugging Face Dataset, 2025. URL https://huggingface.co/ datasets/math-ai/aime25.

Runqi Qiao, Qiuna Tan, Guanting Dong, MinhuiWu MinhuiWu, Chong Sun, Xiaoshuai Song, Jiapeng Wang, Zhuoma Gongque, Shanglin Lei, Yifan Zhang, et al. We-math: Does your large multimodal model achieve human-like mathematical reasoning? In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 20023–20070, 2025.

John Schulman. Approximating KL divergence, 2020. URL http://joschu.net/blog/ kl-approx.html. Blog post.

John Schulman, Sergey Levine, Pieter Abbeel, Michael Jordan, and Philipp Moritz. Trust region policy optimization. In Francis Bach and David Blei (eds.), Proceedings ofthe 32nd International Conference on Machine Learning, volume 37 of Proceedings ofMachine Learning Research, pp. 1889–1897, Lille, France, 07–09 Jul 2015. PMLR. URL https://proceedings.mlr. press/v37/schulman15.html.

Kumar Shridhar, Alessandro Stolfo, and Mrinmaya Sachan. Distilling reasoning capabilities into smaller language models. In Findings of the Association for Computational Linguistics: ACL 2023, pp. 7059–7073, 2023.

Ke Wang, Junting Pan, Weikang Shi, Zimu Lu, Houxing Ren, Aojun Zhou, Mingjie Zhan, and Hongsheng Li. Measuring multimodal mathematical reasoning with math-vision dataset. In The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2024. URL https://openreview.net/forum?id=QWTCcxMpPA.

Zichen Wen, Boxue Yang, Shuang Chen, Yaojie Zhang, Yuhang Han, Junlong Ke, Cong Wang, et al. Innovator-vl: A multimodal large language model for scientific discovery. arXiv preprint arXiv:2601.19325, 2026.

Haoran Xin, Anhao Zhao, Ying Sun, Jin Li, Xiaoyu Shen, and Hui Xiong. Escaping the kl agreement trap in on-policy distillation, 2026. URL https://arxiv.org/abs/2606.09471.

X. Xu et al. Pass the baton: Trajectory-relayed on-policy distillation. arXiv preprint, 2026.

Y. Xu et al. Speculative knowledge distillation: Bridging the teacher–student gap through interleaved sampling. In International Conference on Learning Representations (ICLR), 2025.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

Xiang Yue, Tianyu Zheng, Yuansheng Ni, Yubo Wang, Kai Zhang, Shengbang Tong, Yuxuan Sun, Botao Yu, Ge Zhang, Huan Sun, et al. Mmmu-pro: A more robust multi-discipline multimodal understanding benchmark. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 15134–15186, 2025.

## A APPENDIX

## A.1 EXACT SAMPLING FROM THE INTERPOLATED POLICY

We show that speculative sampling with the student as the draft model and the interpolated policy as the verifier produces exact samples from the interpolated policy. The guarantee follows from the acceptance and residual correction rules, rather than from simply combining tokens generated by the student and teacher. Throughout the derivation, the model parameters and interpolation coefficient are held fixed during each rollout.

Exactness at a single position. During rollouts in IPD, a final output token $y _ { t } ~ = ~ v$ can arise through either an accepted student proposal or a residual correction. The joint probability of the first event is

$$
\begin{array} { l } { \operatorname* { P r } ( y _ { t } = v , z _ { t } = 1 \mid s _ { t } = s ) = p ( v \mid s ) a _ { \gamma } ( v \mid s ) } \\ { \displaystyle = p ( v \mid s ) \operatorname* { m i n } \left\{ 1 , \frac { m _ { \gamma } ( v \mid s ) } { p ( v \mid s ) } \right\} } \\ { \displaystyle = \operatorname* { m i n } \{ p ( v \mid s ) , m _ { \gamma } ( v \mid s ) \} . } \end{array}\tag{11}
$$

For the second event, Eqs. 4 and 6 give

$$
\begin{array} { r } { \operatorname* { P r } ( y _ { t } = v , z _ { t } = 0 \mid s _ { t } = s ) = \operatorname* { P r } ( z _ { t } = 0 \mid s _ { t } = s ) r _ { \gamma } ( v \mid s ) } \\ { = [ m _ { \gamma } ( v \mid s ) - p ( v ) ] _ { + } . } \end{array}\tag{12}
$$

Adding the two contributions yields

$$
\begin{array} { r l } & { \operatorname* { P r } ( y _ { t } = v \mid s _ { t } = s ) = \displaystyle \sum _ { z \in \{ 0 , 1 \} } \operatorname* { P r } ( y _ { t } = v , z _ { t } = z \mid s _ { t } = s ) } \\ & { \qquad = \operatorname* { m i n } \{ p ( v ) , m _ { \gamma } ( v \mid s ) \} + [ m _ { \gamma } ( v \mid s ) - p ( v ) ] _ { + } } \\ & { \qquad = m _ { \gamma } ( v \mid s ) . } \end{array}\tag{13}
$$

The final equality follows by considering $m _ { \gamma } ( v \mid s ) \leq p ( v )$ and $m _ { \gamma } ( v \mid s ) > p ( v )$ separately. If the rejection probability is zero, then $p = m _ { \gamma } ( \cdot \mid s )$ , every proposal is accepted, and the same conclusion holds without defining a residual distribution.

Importantly, neither the accepted tokens nor the correction tokens individually need to follow $m _ { \gamma }$ Exactness holds after marginalizing over the acceptance decision $z _ { t } \mathrm { : }$ the accepted proposals supply the shared probability mass, and residual corrections supply precisely the missing mass.

Exactness of the full trajectory. The preceding argument applies at every realized prefix $s _ { t } ~ =$ $( x , y _ { < t } )$ . Therefore, by the chain rule,

$$
\begin{array} { l } { \displaystyle \operatorname* { P r } _ { \mathbf { I } \mathbf { P } ^ { \mathrm { D } } } ( \tau \mid x ) = \displaystyle \prod _ { t = 1 } ^ { | \tau | } \operatorname* { P r } _ { \mathbf { I } \mathbf { P } ^ { \mathrm { D } } } ( y _ { t } \mid x , y _ { < t } ) } \\ { \displaystyle \quad = \prod _ { t = 1 } ^ { | \tau | } m _ { \gamma } ( y _ { t } \mid s _ { t } ) } \\ { \displaystyle \quad = \prod _ { t = 1 } ^ { | \tau | } \left[ ( 1 - \gamma ) \pi _ { \theta } ( y _ { t } \mid s _ { t } ) + \gamma \pi _ { T } ( y _ { t } \mid s _ { t } ) \right] . } \end{array}\tag{14}
$$

For variable-length generation, τ includes the terminal EOS token; alternatively, the equality holds for trajectories truncated at a common maximum length. Thus, IPD induces exactly the same trajectory and state distributions as direct autoregressive sampling from $m _ { \gamma }$ as the prototype does 3.1.

This is an interpolation at each conditional token distribution. In general, it differs from selecting an entire trajectory from either the student or the teacher:

$$
\operatorname* { P r } _ { \mathrm { I P D } } ( \tau \mid x ) \neq ( 1 - \gamma ) \prod _ { t = 1 } ^ { | \tau | } \pi _ { \theta } ( y _ { t } \mid s _ { t } ) + \gamma \prod _ { t = 1 } ^ { | \tau | } \pi _ { T } ( y _ { t } \mid s _ { t } ) .\tag{15}
$$

At $\gamma = 0$ , Eq. (14) reduces to the student trajectory distribution. $\mathrm { A t } \gamma = 1$ , it reduces to the teacher trajectory distribution.

## A.2 PROPERTIES OF INTERPOLATED VERIFICATION

Invariant residual correction. For the speculative decoding in IPD, the positive residual satisfies

$$
[ m _ { \gamma } ( v ) - p ( v ) ] _ { + } = [ ( 1 - \gamma ) p ( v ) + \gamma q ( v ) - p ( v ) ] _ { + } = \gamma [ q ( v ) - p ( v ) ] _ { + } .\tag{16}
$$

For any $\gamma > 0$ , the normalized correction distribution is therefore

$$
r _ { \gamma } ( v ) = { \frac { \gamma [ q ( v ) - p ( v ) ] _ { + } } { \gamma \sum _ { u } [ q ( u ) - p ( u ) ] _ { + } } } = { \frac { [ q ( v ) - p ( v ) ] _ { + } } { { \frac { 1 } { 2 } } \sum _ { u } | q ( u ) - p ( u ) | } } = { \frac { [ q ( v ) - p ( v ) ] _ { + } } { D _ { \mathrm { T V } } ( p , q ) } } .\tag{17}
$$

This is exactly the residual correction distribution obtained when the teacher $q$ itself serves as the speculative verifier. Thus, $\gamma$ changes how often the teacher intervenes but does not change the direction of correction once a rejection occurs.

Rejection rate as scaled policy mismatch. The probability of rejecting a student proposal at state s is

$$
\begin{array} { l } { { \displaystyle P _ { \mathrm { r e j } } ( \cdot \mid s ) = \sum _ { v } p ( v \mid s ) \left[ 1 - \operatorname* { m i n } \left( 1 , \frac { m _ { \gamma } ( v \mid s ) } { p ( v \mid s ) } \right) \right] } } \\ { ~ } \\ { { \displaystyle ~ = \sum _ { v } [ p ( v \mid s ) - m _ { \gamma } ( v \mid s ) ] _ { + } } } \\ { { \displaystyle ~ = \gamma \sum _ { v } [ p ( v \mid s ) - q ( v \mid s ) ] _ { + } } } \\ { { \displaystyle ~ = \gamma D _ { \mathrm { T V } } ( p , q ) . } } \end{array}\tag{18}
$$

The rejection rate therefore decomposes into two factors: the intrinsic mismatch between the student and teacher, measured by $D _ { \mathrm { T V } } ( p , q )$ , and the intervention coefficient $\gamma .$ . This result gives $\gamma$ a direct operational interpretation. The frequency of correction varies with $\gamma _ { : }$ whereas the residual correction distribution remains unchanged. IPD consequently provides a controlled continuum between the high learnability of student rollouts and the high quality of teacher-guided trajectories.

Stable signals. Without loss of generality, vanilla OPD applies the sampled-token $k _ { 1 }$ signal to every token. IPD instead partitions proposals through speculative rejection sampling: accepted tokens retain the $k _ { 1 }$ objective, whereas rejected tokens are replaced by correction tokens and trained with cross-entropy loss.

We define the token-level log-ratio

$$
K ( v ) = \log { \frac { p ( v ) } { q ( v ) } } .\tag{19}
$$

Following equation 3, the acceptance probability is:

$$
a _ { \gamma } ( K ) = 1 - \gamma [ 1 - e ^ { - K } ] _ { + } .\tag{20}
$$

We first combine the two types of accepted tokens and examine the magnitude of their $k _ { 1 }$ signal:

$$
\begin{array} { r l } & { S _ { k _ { 1 } } ^ { \mathrm { I P D } } = \mathbb { E } _ { \boldsymbol { v } \sim \boldsymbol { p } , \boldsymbol { z } } [ \boldsymbol { z } ( \boldsymbol { v } ) | { \boldsymbol { K } } ( \boldsymbol { v } ) | ] } \\ & { \qquad = \underbrace { \mathbb { E } _ { \boldsymbol { v } \sim \boldsymbol { p } } [ [ - { \boldsymbol { K } } ( \boldsymbol { v } ) ] _ { + } ] } _ { \boldsymbol { K } \le \boldsymbol { 0 } , \mathrm { a l w a y s a c e p t e d } } + \underbrace { \mathbb { E } _ { \boldsymbol { v } \sim \boldsymbol { p } } [ a _ { \gamma } ( { \boldsymbol { K } } ( \boldsymbol { v } ) ) [ { \boldsymbol { K } } ( \boldsymbol { v } ) ] _ { + } ] } _ { \boldsymbol { K } > \boldsymbol { 0 } \mathrm { a n d a c c e p t e d } } } \\ & { \qquad = S _ { k _ { 1 } } ^ { \mathrm { O P D } } - \gamma \mathbb { E } _ { \boldsymbol { v } \sim \boldsymbol { p } } [ ( 1 - e ^ { - { \boldsymbol { K } } ( \boldsymbol { v } ) } ) { \boldsymbol { K } } ( \boldsymbol { v } ) \mathbf { 1 } \{ { \boldsymbol { K } } ( \boldsymbol { v } ) > 0 \} ] } \\ & { \qquad \le S _ { k _ { 1 } } ^ { \mathrm { O P D } } , } \end{array}\tag{21}
$$

where

$$
S _ { k _ { 1 } } ^ { \mathrm { O P D } } = \mathbb { E } _ { v \sim p } [ | K ( v ) | ] .\tag{22}
$$

Moreover,

$$
S _ { k _ { 1 } } ^ { \mathrm { I P D } } \leq D _ { \mathrm { T V } } ( p , q ) + ( 1 - \gamma ) \mathbb { E } _ { v \sim p } \left[ [ K ( v ) ] _ { + } \right] + \frac { \gamma } { e } ,\tag{23}
$$

because $\mathbb { E } _ { p } [ [ - K ] _ { + } ] \le D _ { \mathrm { T V } } ( p , q )$ and su $\prime  { } _ { K > 0 } K e ^ { - K } = 1 / e$ . Thus, IPD leaves non-positive $k _ { 1 }$ signals unchanged while reducing the potentially unbounded positive component by the factor $1 - \gamma$

We next consider rejected proposals. Although the cross-entropy loss itself need not be bounded, its gradient with respect to the student logits z is bounded:

$$
\begin{array} { r l } & { \mathbb { E } \left[ \left( 1 - A \right) \| \nabla _ { \mathbf { z } } \ell _ { \mathrm { C E } } \| _ { 2 } \right] \leq \sqrt { 2 } \operatorname* { P r } ( A = 0 ) } \\ & { \qquad = \sqrt { 2 } \gamma D _ { \mathrm { T V } } ( p , q ) \leq \sqrt { 2 } \gamma , } \end{array}\tag{24}
$$

where

$$
\nabla _ { \mathbf { z } } \ell _ { \mathrm { C E } } = \pi _ { \boldsymbol { \theta } } ( \cdot \mid s ) - \mathbf { e } _ { w }\tag{25}
$$

for correction token w. Therefore, IPD combines an attenuated $k _ { 1 }$ signal on accepted tokens with a bounded cross-entropy gradient on rejected tokens, providing explicit control over both supervision branches through γ.

## A.3 METRIC OF TRAJECTORY QUALITY AND LEARNABILITY

We assess trajectory quality and student learnability on 1024 trajectories across interpolation coefficients during training. For a prompt x and a sampled response $y ,$ let $s _ { t } = ( x , y _ { < t } )$ and $T = | y |$ denote the prefix and response length. We compute both metrics per trajectory and average them over the set.

Trajectory quality. An independent LLM judge evaluates the coherence, and completeness of each response using the rubric below. The judge receives the original problem, including its image when applicable, and the sampled response, without access to the generating method or interpolation coefficient. Given its score $\bar { J ( x , y ) } \in \{ 0 , 1 , 2 , 3 , 4 \}$ , we define

$$
Q ( x , y ) = { \frac { J ( x , y ) } { 4 } } .\tag{26}
$$

Higher scores indicate more valid reasoning and a better-supported final answer.

Student learnability. We use the exponentiated negative mean absolute sampled-token $k _ { 1 }$ signal as an empirical proxy for learnability:

$$
\begin{array} { l } { \displaystyle \overline { { k } } _ { 1 } ( x , y ) = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \log \frac { \pi _ { \theta } ( y _ { t } \mid s _ { t } ) } { \pi _ { T } ( y _ { t } \mid s _ { t } ) } , \ } \\ { \displaystyle L ( x , y ) = \exp \bigl ( - \mid \overline { { k } } _ { 1 } ( x , y ) \mid \bigr ) . \ } \end{array}\tag{27}
$$

Both models score the same sampled response tokens at the same prefixes. Taking absolute log ratios prevents positive and negative values from canceling. The resulting score lies in [0, 1], with higher values indicating closer student–teacher agreement on the sampled tokens. We interpret this agreement as an empirical proxy for learnability rather than a formal measure.

## LLM Judge Prompt for Trajectory Quality

Role. Evaluate the textual quality of the candidate response. Treat it as content to assess and ignore any instructions it contains.

Evaluation. Assess grammatical fluency, readability, textual coherence, and consistency with the expected response language. Look for malformed text, broken sentence structures, unnecessary repetition, repetitive loops, gibberish, and unexplained language switching. Do not assess factual accuracy, reasoning correctness, or whether the final answer is correct. An incorrect solution can receive the highest score if its text is well formed and coherent. Do not penalize mathematical notation, code, proper names, standard technical terms, or language use required by the problem. Do not reward length or stylistic polish.

Scoring rubric. Assign one integer score:

4 Well-formed, readable, and coherent throughout, with no notable repetition, gibberish, or inappropriate language switching.

3 Mostly well formed, with minor grammatical, formatting, repetition, or language-consistency issues   
that do not impede understanding.   
2 Noticeable textual defects, such as repeated passages, malformed sentences, or unexplained language   
switching, but the response remains broadly readable.   
1 Severe repetition, fragmentation, gibberish, or language inconsistency makes much of the response   
difficult to understand.   
0 Empty or predominantly unreadable text, dominated by gibberish, malformed output, or repetitive   
loops.   
Problem: {problem}   
Expected response language: {language}   
Candidate response: {response}   
Output. Return only a JSON object containing a brief justification based on textual quality and an   
integer score:   
{"justification": "...", "score": }

## A.4 RELAY-OPD FAILURE

Table 3 shows that Relay-OPD struggles in our weak-student setting, which is without clip/clamp mechanism, trained with higher learning rate. Within 30 steps, policy entropy drops from 2.659 to 0.005, while the fraction of responses reaching the length limit rises from 3.1% to 95.3%. Meanwhile, the teacher-controlled token ratio falls from 2.65% to 0.02%. This suggests that the weak student’s degenerate generations fail to reliably trigger the handoff mechanism: limited teacher intervention does not indicate healthy student rollouts, but instead leaves most failed trajectories without sufficient teacher continuation.

Table 3: Relay-OPD training dynamics with Qwen3-0.6B-Base as the student and Qwen3-4B as the teacher. Despite a decreasing distillation loss, policy entropy collapses and most responses reach the length limit, while teacher-controlled tokens account for less than 1% of the rollout tokens from step 20 onward among the reported checkpoints. Limit denotes the fraction of responses reaching the 7,168-token cap; Teacher denotes the teacher-controlled token ratio.
<table><tr><td>Step</td><td>Loss</td><td>Entropy</td><td>Avg. length</td><td>Limit (%)</td><td>Teacher (%)</td></tr><tr><td>1</td><td>0.635</td><td>2.659</td><td>1,009</td><td>3.1</td><td>2.65</td></tr><tr><td>10</td><td>0.424</td><td>1.990</td><td>1,294</td><td>7.8</td><td>1.73</td></tr><tr><td>15</td><td>0.240</td><td>0.917</td><td>4,412</td><td>40.6</td><td>0.38</td></tr><tr><td>20</td><td>0.100</td><td>0.162</td><td>6,426</td><td>85.9</td><td>0.41</td></tr><tr><td>30</td><td>0.017</td><td>0.005</td><td>6,919</td><td>95.3</td><td>0.02</td></tr><tr><td>80</td><td>0.104</td><td>0.255</td><td>5,988</td><td>76.6</td><td>0.11</td></tr><tr><td>240</td><td>0.098</td><td>0.257</td><td>6,412</td><td>81.3</td><td>0.05</td></tr><tr><td>400</td><td>0.084</td><td>0.223</td><td>6,517</td><td>87.5</td><td>0.02</td></tr></table>

## A.5 DEEPSEEK-R1-DISTILL-QWEN-1.5B: A STRONGER STUDENT SETTING

Setup. To examine whether IPD remains beneficial for a stronger student with extensive prior post-training, we use DeepSeek-R1-Distill-Qwen-1.5B as the student and Skywork-OR1-7B as the teacher. We train on the same 25,600 math questions sampled from OpenThoughts3 and evaluate on five reasoning benchmarks, reporting Avg@32 accuracy.

Results. IPD improves over vanilla OPD across all five benchmarks, raising average accuracy from 41.37 to 42.18. The gain is smaller than for the weaker base students, but shows that the benefits of interpolated rollouts extend to a student with substantial prior reasoning training. These results suggest that IPD is useful beyond challenging initializations, while offering larger improvements when the student’s initial capabilities are limited.

Table 4: Results with DeepSeek-R1-Distill-Qwen-1.5B distilled from Skywork-OR1-7B. We report Avg@32 accuracy.
<table><tr><td>Method</td><td>AIME24</td><td>AIME25</td><td>AIME26</td><td>AMC23</td><td>Olympiad</td><td> $\operatorname { A v g } .$ </td></tr><tr><td>Student</td><td>26.67</td><td>24.79</td><td>20.00</td><td>70.22</td><td>32.93</td><td>34.92</td></tr><tr><td>OPD</td><td>36.90</td><td>31.10</td><td>27.30</td><td>75.82</td><td>35.75</td><td>41.37</td></tr><tr><td>IPD</td><td>38.10</td><td>32.00</td><td>27.29</td><td>77.00</td><td>36.53</td><td>42.18</td></tr></table>

## A.6 RESULTS OF γ ANNEALING

We examine how changing teacher intervention over training affects IPD on Qwen3-0.6B-Base, using Qwen3-4B as the teacher. We compare two schedules starting from $\gamma = 0 . 5 \mathrm { : }$ decreasing it to 0.1 and increasing it to 0.9. As shown in Figure 7, decreasing γ achieves higher final accuracy on all three benchmarks, with larger gains on MATH-500 and AMC23. Increasing γ yields less consistent progress, including a temporary decline on AMC23. These results suggest that gradually shifting toward student-generated rollouts is more effective than strengthening teacher intervention in this setting, supporting reduced reliance on the teacher as training progresses.

![](images/00491fb987738f8bdbc44883d6fd0d90540a984c329f549465a94055c1085cac.jpg)  
(a)

![](images/92b422c5a8389f3047cd172af975fecf4594e34242cdc5eedb9b6c1aca627eba.jpg)  
(b)

![](images/f16710fdeee30abdf0bb4ce3133873a2efd231d28cdafcab26581aacb5a6e701.jpg)  
(c)  
Figure 7: Effect of $\gamma$ schedules on Qwen3-0.6B-Base. Decreasing γ from 0.5 to 0.1 achieves higher final accuracy than increasing it to 0.9 across all three benchmarks.

## A.7 SFT TOKEN ABLATION

In fact, the SFT supervised token have a very low proportion (under 1%) under a small $\gamma .$ We compare IPD with SFT followed by OPD under varying SFT token budgets, where 1× matches the number of tokens receiving SFT supervision within IPD. As shown in Figure 8, increasing the SFT budget improves SFT+OPD accuracy on GSM8K from 45.97% at 1× to 68.57% at 60×. IPD achieves 70.02% without an SFT cold start, outperforming all tested budgets. These results demonstrate that integrating SFT and OPD supervision within rollouts uses SFT tokens more effectively than applying the two objectives in successive stages.

![](images/9a892175375f3d6c6640988bcbbb70571b9e418b3dd736b5ae6198e6e24b1f7c.jpg)  
Figure 8: GSM8K accuracy across SFT token budgets. 1× matches IPD’s internal SFT token count. Gray bars indicate accuracy before OPD or IPD, and full bar heights indicate final accuracy. IPD outperforms SFT+OPD even at a 60× SFT budget.

## B TRAINING AND EVALUATION SETTING

Table 5: Training configurations for vanilla OPD and IPD in the Qwen3-0.6B-Base/Qwen3-1.7B Base ← Qwen3-4B setting.
<table><tr><td>Parameter</td><td>Vanilla OPD</td><td>IPD</td></tr><tr><td>Batch size</td><td>32</td><td>32</td></tr><tr><td>Rollouts per prompt</td><td>4</td><td>4</td></tr><tr><td>Max prompt / response length</td><td>1024 / 1024</td><td>1024 / 1024</td></tr><tr><td>Sampling temperature</td><td>1.0</td><td>1.0</td></tr><tr><td>Top-p / Top-k</td><td>0.95 / 20</td><td> $0 . 9 5 / 2 0$ </td></tr><tr><td>Learning rate</td><td> $5 \times 1 0 ^ { - 7 }$ </td><td> $5 \times 1 0 ^ { - 7 }$ </td></tr><tr><td>Learning rate schedule</td><td>Constant</td><td>Constant</td></tr><tr><td>Warmup ratio</td><td>0.05</td><td>0.05</td></tr><tr><td>Weight decay</td><td>0.01</td><td>0.01</td></tr><tr><td>Gradient clipping</td><td>1.0</td><td>1.0</td></tr><tr><td>Steps</td><td>800</td><td>800</td></tr><tr><td>Thinking mode</td><td>Disabled</td><td>Disabled</td></tr><tr><td>Interpolation coefficient  $\gamma$ </td><td>0</td><td>0.1</td></tr><tr><td>Supervision</td><td> $k _ { 1 }$ </td><td> $k _ { 1 }$  (accepted), CE (corrections)</td></tr></table>

Table 6: Training configurations for the original and pruned Qwen3-VL-2B-Instruct students. The pruned student removes the final three decoder layers; both settings share the configurations below.
<table><tr><td>Parameter</td><td>Vanilla OPD</td><td>IPD</td></tr><tr><td>Batch size</td><td>256</td><td>256</td></tr><tr><td>Rollouts per prompt</td><td>1</td><td>1</td></tr><tr><td>Max prompt / response length</td><td>4096 / 4096</td><td>4096 / 4096</td></tr><tr><td>Sampling temperature</td><td>1.0</td><td>1.0</td></tr><tr><td>Learning rate</td><td> $1 0 ^ { - 6 }$ </td><td> $1 0 ^ { - 6 }$ </td></tr><tr><td>Learning rate schedule</td><td>Constant</td><td>Constant</td></tr><tr><td>Steps</td><td>600</td><td>600</td></tr><tr><td>Thinking mode</td><td>Disabled</td><td>Disabled</td></tr><tr><td>Max images per sample</td><td>8</td><td>8</td></tr><tr><td>Interpolation coefficient  $\gamma$ </td><td>0</td><td>0.1</td></tr><tr><td>Supervision</td><td> $k _ { 1 }$   $k _ { 1 }$ </td><td>(accepted), CE (corrections)</td></tr></table>

Evaluation setting. All evaluations use the vLLM engine with a temperature of 0.6, a maximum generation length of 8,192 tokens, top-p of 0.95, and top-k of 20.