# Satisficing as competitive Bayesian weighting: a minimal model of goal-directed choice under an unattained target

Moto Shinriki<sup>a</sup>, Yu Kono<sup>a</sup>, Tatsuji Takahashi<sup>a,∗</sup>

<sup>a</sup>Graduate School of Advanced Science and Engineering, Tokyo Denki University, Hatoyama-machi, 350-0394, Saitama, Japan

## Abstract

Animals and humans often do not maximize; they act relative to goals (aspiration levels) that are frequently not yet met. How can a goal that is not (yet) realized organize choice? Deacon (2012) calls such absence-referenced organization ententional, and García-Valdecasas and Deacon (2024) argue that ends are best understood as constraints rather than as particular states. We examine this idea in a minimal formal setting: Risk-sensitive Satisficing (RS), a bandit learning rule that values each action by its share of past trials multiplied by its surplus or shortfall relative to an aspiration level. We show that RS can be rewritten as a Bayesian-like estimator, Attention Competitive Bayes (ACB), in which the aspiration level is the prior mean of every action and the finite budget of trials sets how strongly evidence can override it. Two consequences follow. First, actions that are no longer sampled are partially overridden by the goal rather than by past observations: their estimates relax toward the aspiration level without any data being discarded. This “pseudo forgetting” distinguishes ACB from models with an explicit forgetting rate. Second, when the aspiration is unattainable, exploration does not settle; in Bernoulli bandit simulations, adjusting the aspiration level from evidence reduced regret (expected loss). ACB ofers an analyzable model in which an unattained goal and a limited attentional resource jointly constrain behavior, and it yields testable predictions about how organisms revalue options they have abandoned.

Keywords: reinforcement learning, multi-armed bandit, decision making, aspiration level, goal-directedness, risk-sensitive foraging, bounded rationality

## 1. Introduction

Living systems rarely pursue a single best outcome. A foraging animal needs enough energy to survive the night, not the maximum obtainable. A person searching for an apartment stops searching when one is good enough. Simon called this satisficing: behavior organized with respect to an aspiration level rather than an optimum [1, 2]. Satisficing is not merely a cheaper approximation to optimization. In studies of risk-sensitive foraging, animals whose expected intake falls short of their requirement tend to prefer variable food sources, whereas those above it tend to prefer reliable ones [3, 4, 5]. The risk attitude of an organism can thus switch according to whether a goal is currently met.

This raises a basic question about biological organization: how can a goal that is not yet attained shape what an organism does next? Under maximization, the target is a single best option, and the fact that it is not yet attained plays no role in the valuation and choice itself. Under satisficing, the target is a range of acceptable outcomes, rather than a specific value (point), and whether that range has been reached changes behavior.

Deacon called phenomena that are organized with respect to something not (yet) present ententional [6]. His examples include functions that have conditions of satisfaction. García-Valdecasas and Deacon further argued that ends are represented by constraints rather than by particular physical states [7]. This view suggests a way to approach the above question formally, but doing so requires a minimal model that can be analyzed.

In this paper, we use Risk-sensitive Satisficing (RS), which we proposed in previous works, as such a minimal model [8, 9, 10, 11, 12, 13, 14]. RS is a simple value bandit learning rule that values each action by the product of its share of past trials and its surplus or shortfall relative to the aspiration level. Hereafter, we call this aspiration level the satisfaction level S. An RS agent optimistically explores rarely tried actions when no action meets the satisfaction level, and it conservatively exploits an action that meets the level when one exists. Kamiya and Takahashi showed that RS and Softsatisficing, its softmax version, can serve as models of risk-sensitive decision making such as animal foraging [15]. Tamatsukuri and Takahashi proved that the cumulative expected loss (hereafter, regret; see Section 2 for details) of RS is bounded by a finite value under specific conditions [16]. However, the theoretical structure behind its defining equation, namely, how a satisfaction level that is not yet attained enters the valuation of actions, has not been suficiently clarified.

In this paper, we address this issue of what it is to have a goal through three questions. First, how does the satisfaction level enter the valuation of actions? We reinterpret the defining equation of RS and show that it has a Bayesian-like structure in which the satisfaction level serves as the prior mean of every action. We formulate this structure as Attention Competitive Bayes (ACB). Second, what does this model predict about actions that are no longer chosen? In ACB, the estimates of such actions approach the satisfaction level without discarding data. We call this pseudo-forgetting and compare it with existing Bayesian inference models through simulations. Third, what happens when the satisfaction level is unattainable, and how can the level be adjusted? We show that exploration does not settle when the satisfaction level is too high, and we examine the efect of adjusting the level from observations, comparing RS with other satisficing methods.

This study is also related to reinforcement learning (RL). Traditionally, RL has been treated mainly as an optimization problem aimed at maximizing cumulative rewards [17]. This framework has a theoretical foundation inspired by biological behavior and cognition [18, 19, 20], and it has achieved remarkable results under certain levels of environmental complexity [21, 22]. RL methods that incorporate human cognitive biases have also been proposed [20, 23, 24, 25]. However, real-world applications of RL remain limited [26]. Ortner et al. point out that the cause lies in the gap between RL theory, which aims at optimization, and real-world problems [27]. Many real-world problems do not necessarily require finding the optimal solution; it is often more important to quickly find a suficiently good behavior that achieves the goal. With this motivation, Michel and Ortner proposed a framework of satisficing RL, in which the agent aims to find actions whose rewards exceed a satisfaction level, together with a method called SAT-UCB [28]. We also place RS in this context of satisficing RL and compare it with SAT-UCB<sup>+</sup> (a performance improved variant) and with Thompson Sampling, an optimization method with highest performance. Our main aim, however, is not to establish the superiority of a method but to answer the three questions above.

The rest of this paper is organized as follows. Section 2 describes the task settings and related work, and Section 3 describes the definition of RS and its exploration process. Section 4 reinterprets RS as ACB (the first question). Section 5 compares pseudo-forgetting with existing Bayesian inference models (the second question). Section $6$ examines the case in which the satisfaction level is unattainable (the third question). Section 7 discusses the results as a whole, and Section 8 concludes the paper.

## 2. Goal-Referenced Choice: Task Settings and Related Work

The task considered in this paper is a satisficing RL task, specifically the satisficing bandit task. In this section, we clarify the task setting considered in this paper. We also review related work to clarify the position of RS and its Bayesian interpretation, ACB.

## 2.1. Satisficing Bandit Tasks

In multi-armed bandit tasks, an agent chooses an action from the action set $[ [ 1 , K ] ] : = \{ 1 , 2 , \cdots , K \}$ . Hereafter, we call an action in a bandit task an arm. Each arm yields a reward according to a reward distribution that is unknown to the agent. At time step t, the agent chooses an arm $a _ { t } \in$ 1, K and receives the reward as the result of that choice. In standard (optimization) bandit tasks, the agent repeats this procedure a finite number of times and aims to maximize the cumulative sum of obtained rewards.

In contrast, in the satisficing bandit task, the agent’s target is not to maximize cumulative reward, but to choose an arm whose reward exceeds a given satisfaction level S, more precisely, whose sample mean reward exceeds S [28]. Among satisficing bandit tasks, this paper focuses on Bernoulli bandit tasks, in which each arm yields a reward $r \in \{ 0 , 1 \}$ and is characterized by the probability that $r = 1$ . Fig. 1 illustrates an overview of the task setting considered in this paper. In satisficing bandit task, how the satisfaction level $S$ is set strongly afects learning behavior and performance. One such setting is the “optimal satisfaction level” $S _ { \mathrm { o p t } }$ . In this paper, we define the optimal satisfaction level as the midpoint between the highest true expected reward $E [ r _ { \mathrm { 1 s t } } ]$ and the second-highest true expected reward $E [ r _ { \mathrm { 2 n d } } ]$ . Thus, we define $S _ { \mathrm { o p t } }$ by

$$
S _ { \mathrm { o p t } } : = \frac { E [ r _ { \mathrm { 1 s t } } ] + E [ r _ { \mathrm { 2 n d } } ] } { 2 } .\tag{1}
$$

![](images/63fc685219e22a85ca4e50e5406c8fc01d1709d5e39211bb6a271db3b58c1724.jpg)  
Figure 1: The Bernoulli bandit task with K arms. $E [ r _ { i } ]$ denotes the expected reward of an arm. In this paper, we consider tasks only under this setting.

With this satisfaction level, the only arm that satisfies the satisfaction level is the best arm, and satisficing becomes equivalent to optimization.

In this paper, we use regret as a performance metric for the satisficing bandit task. Regret is defined as the cumulative diference between the true expected reward of the best arm, $E [ r _ { \mathrm { 1 s t } } ]$ , and that of the actually chosen arm at time step t, $E [ r _ { \mathrm { c h o s e n } } ]$ . Thus, we define regret by

$$
\mathrm { r e g r e t } : = \sum _ { t = 1 } ^ { T } \left( E [ r _ { \mathrm { 1 s t } } ] - E [ r _ { \mathrm { c h o s e n } } ] \right) .\tag{2}
$$

Regret is a metric in optimizing bandit tasks that measures the gap from the optimal action sequence. We use regret for the following reasons in the satisficing bandit task. We adopt Thompson Sampling (TS) [29], a representative optimizing method [30, 31], as the evaluation baseline. For this reason, we need a metric that enables us to compare satisficing methods and optimizing methods on the same scale. It is also reasonable to adopt regret as a common metric for examining how satisficing-based methods behave when evaluated by an optimizing metric. Michel & Ortner proposed S-regret as a satisficing-specific metric [28]. S-regret is a metric that does not increase when the reward is greater than the satisfaction level S, and increases by the diference between S and the reward when the reward is less than S. For this reason, when the satisfaction level is too high, comparing methods using S-regret reduces to comparing the size of their slopes. This comparison is not substantially diferent from comparing where the normal regrets of multiple algorithms converge. For these reasons, we do not use S-regret as the primary metric, and instead we use regret, which enables comparison of both satisficing methods and optimizing methods by the same measure.

## 2.2. Related Work

Studies related to satisficing come from two lines: models of decision making in living systems and methods in RL. In the former line, Simon formulated satisficing as a theory of decision making under resource constraints [2]. Studies of risk-sensitive foraging have shown that the risk attitude of animals depends on the relation between their energy budget and their requirement [3, 4, 5]. Kamiya and Takahashi examined RS and Softsatisficing as models of such risk-sensitive foraging behavior [15]. Murakoshi et al. used a new RL method to conduct simulations of rat behavior [19]. Kim et al. proposed the TOW model, an amoeba-inspired framework for efficiently solving bandit problems [32], which implicitly performs satisficing. Uragami et al. proposed learning through social information sharing (social satisficing) [33]. We position RS and ACB in this paper as a minimal model in this line. Studies in the latter line, on satisficing RL, can be broadly classified into the following two categories: (i) RL methods that aim to achieve satisficing; and (ii) methods that define satisficing in a diferent way than the one considered in this paper.

RS considered in this paper also belongs to category (i) and is one of the satisficing RL methods proposed by Takahashi et al [8]. Oyo and Takahashi analyzed the behavior of RS in bandit problems [10]. Tamatsukuri and Takahashi showed that, when the optimal satisfaction level is given, the regret of RS converges in finite time [16]. Furthermore, Tsuboya, Wakabayashi, and Shinriki and others extended RS to contextual bandits [13], tabular RL [11, 12], and deep RL [14], demonstrating its efectiveness across a wide range of problem settings. Kamiya and Takahashi also leveraged properties of RS and proposed Softsatisficing as a method for adjusting the inverse temperature (for each arm) of the Softmax policy [15]. Section 3 describes the definition and properties of RS in detail. Michel et al. proposed SAT-UCB as a satisficing RL method [28]. In this paper, we categorize SAT-UCB as a method in category (i). SAT-UCB is based on the UCB method[34], that is proven to have an optimal property in bandit problems, and widely used in RL-related methods such as AlphaGo [35]. A SAT-UCB agent switches its action-selection behavior depending on whether the sample mean reward of arm $i , \bar { r } _ { i }$ , exceeds the satisfaction level S. If $\bar { r } _ { i } \geq S$ , the agent selects from arms that satisfy the satisfaction level. If $\bar { r } _ { i } < S$ , the agent performs exploration based on UCB. In this paper, we show the results of SAT-UCB<sup>+</sup>, a performance-improved variant of SAT-UCB [28], as a comparison method. Reverdy et al. also proposed the UCL algorithm as a Bayesian method for satisficing RL [36, 37]. We categorize UCL in (i). UCL assumes Gaussian rewards and estimates the expected reward of each arm by Bayesian inference. Reverdy et al. formulated satisficing in terms of two aspects: satisfaction and suficiency. In addition to a satisfaction level M, which corresponds to S in this paper, UCL has a suficiency parameter $\delta \in [ 0 , 1 ]$ . When $\delta > 0$ the agent becomes satisfied and stops exploring once the chosen arm meets the satisfaction level with probability at least $1 - \delta$ . In the experiments of Michel et al., UCL performed worse than $\mathrm { S A T - U C B ^ { + } \ [ 2 8 ] }$ . Therefore, we use $\mathrm { S A T - U C B ^ { + } }$ as a benchmark and do not compare our method with UCL.

There are studies whose objectives difer from those of the satisficing RL considered in this paper, namely, methods categorized as (ii). Locatelli et al., Mukherjee et al., and Kano et al. studied threshold bandit tasks [38, 39, 40]. Wu et al. studied conservative bandit tasks [41]. Merlis et al. studied lenient regret [42]. These studies have some aspects in common with ours, but their problem settings qualitatively difer. Russo and Van Roy also investigated satisficing under discounted rewards and provided a bound on the expected discounted regret for a satisficing extension of TS [43]. These algorithms are also incomparable and hence out of the scope of the current study.

## 3. A Minimal Model: Risk-sensitive Satisficing (RS)

RS is one of the satisficing RL methods that incorporates satisficing, a decision-making tendency, into RL. Unlike the other prior algorithms mentioned above, RS afects both the agent’s action selection and value updating. This paper tries to clarify the theoretical structure underlying this characteristic, in the light of Bayesian belief update. In this section, we describe the definition of RS and the exploration process specific to RS.

## 3.1. Definition of RS

Let us consider the multi-armed bandit tasks. Let $n _ { i } ^ { t }$ denote the number of times that the agent has chosen arm $i \in [ [ 1 , K ] ]$ up to time step t. We define the total number of trials by

$$
N ^ { t } : = \sum _ { i = 1 } ^ { K } { n _ { i } ^ { t } } .
$$

We define the trial proportion of arm i by

$$
\rho _ { i } ^ { t } : = \frac { n _ { i } ^ { t } } { N ^ { t } } .
$$

We define the sample mean reward observed from arm i up to time step t by

$$
\bar { r } _ { i } ^ { t } : = \frac { 1 } { n _ { i } ^ { t } } \sum _ { s = 1 } ^ { t } r _ { s } \mathbb { I } \{ a _ { s } = i \} \qquad ( n _ { i } ^ { t } > 0 ) .
$$

The symbol I denotes the indicator function. The RS agent selects arms based on the value function I defined as follows:

$$
I _ { i } ^ { t } : = \rho _ { i } ^ { t } \big ( \bar { r } _ { i } ^ { t } - S \big ) .\tag{3}
$$

At each time step t, the RS agent greedily selects an arm:

$$
a _ { t } = \arg \operatorname* { m a x } _ { i } I _ { i } ^ { t } .\tag{4}
$$

Tamatsukuri & Takahashi proved that the regret converges to a finite value for RS when it uses an optimal satisfaction level $S _ { \mathrm { o p t } }$ (Eq. 1) as the satisfaction level [16].

## 3.2. Exploration Process Specific to RS

RS agent behaves diferently in exploration and exploitation depending on whether the sample mean of rewards, $\bar { r } _ { i } ^ { t } .$ , exceeds the satisfaction level S. In this subsection, we explain the exploration process specific to RS. To describe the behavior of RS, we distinguish two states, the over-achieved condition and the under-achieved condition, based on the relative relationship between the satisfaction level and the sample means (Figs. 2a and 2b, resp.). The over-achieved condition is where there exists an arm i that satisfies $\bar { r } _ { i } ^ { t } \geq S$ . The under-achieved condition is where for all arm i, $\bar { r } _ { i } ^ { t } < S$ holds. RS switches its risk attitude depending on these two conditions. In the over-achieved condition, the agent keeps selecting arms that satisfy the satisfaction level. The agent also prefers more reliable arms among those that satisfy the satisfaction level. That is, arms with larger $\rho _ { i } ^ { t }$ tend to be preferred. From Eq. (3), for arms satisfying $\bar { r } _ { i } ^ { t } > S$ , a larger $\rho _ { i } ^ { t }$ yields a larger $I _ { i } ^ { t }$ , which promotes exploitative action selection by the agent. In contrast, in the under-achieved condition, $\bar { r } _ { i } ^ { t } < S$ holds for all arms. In this case, when the trial proportion $\rho _ { i } ^ { t }$ is smaller, the negative diference $\bar { r } _ { i } ^ { t } - S$ has a smaller efect, and $I _ { i } ^ { t }$ becomes larger. As a result, the agent tends to evaluate the potential of arms with fewer trials. Therefore, in the under-achieved condition, the agent tends to select arms with fewer trials more, and exploration is promoted. Such optimistic exploration of RS in the under-achieved condition is similar to the principle for the UCB methods, namely, “optimism in the face of uncertainty” [44], though the optimism in additive in UCB while multiplicative in under-achieved RS.

Surplus  
![](images/f2c616580c2efb19bbd92009e0581bea926afbc0eb7dc1ef8facc6e324716d6e.jpg)

![](images/5d63f60de4d04f038cb24ab3099be2c58d4315985293ebc4b064f94150439fc1.jpg)  
(a) Over-achieved condition  
(b) Under-achieved condition  
Figure 2: Illustration of the two states defined by the relative magnitude of the sample mean of rewards, $\bar { r } _ { i } ,$ with respect to the satisfaction level S. (a) The over-achieved condition occurs if there exists at least one arm i such that $S \leq { \bar { r } } _ { i }$ . (b) The under-achieved condition occurs if $S > \bar { r } _ { i }$ for all arms.

This state-dependent switching of risk attitudes enables eficient exploration and exploitation. Michel & Ortner defined the concept of the “realizable case,” which is similar to, but distinct from, the over-achieved condition, in their paper [28]. The realizable case refers to a situation in which for some i, $E [ r _ { i } ] \ge S$ holds, i.e., a situation in which the task itself is satisfiable. In contrast, being in the over-achieved condition does not necessarily imply that the task is realizable. Conversely, even if the task is in the realizable case, it does not necessarily imply that the agent is in the over-achieved condition, unless in a limit case.

This switching of risk attitude in RS is consistent with the findings on risk-sensitive foraging described in Section 1 [5, 15].

## 4. The Goal as a Kind of Prior: RS as Attention Competitive Bayes (ACB)

In this section, we analyze the internal structure of RS and clarify the competitive mechanism of attention resources inherent in RS. Based on this analysis, we propose Attention Competitive Bayes (ACB). ACB is a reinterpretation of RS from a Bayesian perspective. The purpose of this section is to deepen our understanding of the exploration behavior and satisficing process of RS by reinterpreting it as ACB.

## 4.1. Attention Mechanism of RS

As in Eq. (3), RS uses the trial proportion $\rho _ { i } ^ { t }$ to calculate the value function I. The diference between the sample mean and the satisfaction level, $\delta _ { i } ^ { t } = \bar { r } _ { i } ^ { t } - S$ , is weighted by the trial proportion. This structure can also be viewed as arms compete for the finite resource of trials. That is, the more trials are allocated to an arm, the more importance is assigned to $\delta _ { i } ^ { t }$ ; conversely, the fewer trials are allocated to an arm, the less importance is assigned to $\delta _ { i } ^ { t }$ . In this paper, we call this resource-allocation process, in which the trial proportion adjusts the weight of $\delta _ { i } ^ { t }$ , the Attention competitive mechanism inherent in RS. Here, attention does not refer to selective attention in psychology or to attention mechanisms in machine learning; it refers only to the allocation of a finite number of trials among arms. In the next subsection, we explicitly extract this mechanism and formalize it as a more general framework.

## 4.2. Bayesian-like Structure of RS

In this subsection, we clarify the mathematical structure of ACB by reinterpreting the value function of RS as an estimator of the expected reward. Eq. (4) shows that action selection depends only on the relative ordering of the value functions. Therefore, adding (or subtracting) the same constant S to the value function does not change the action selection result. Thus, we reinterpret the value function of RS plus $S$ as an estimator of the expected reward, denoted by $q _ { i } ,$ , and define

$$
q _ { i } = S + I _ { i } ^ { t } = S + \rho _ { i } ^ { t } ( \bar { r } _ { i } ^ { t } - S ) .\tag{5}
$$

Eq. (5) can be transformed into the following weighted-average form:

$$
q _ { i } = ( 1 - \rho _ { i } ^ { t } ) S + \rho _ { i } ^ { t } \bar { r } _ { i } ^ { t } .\tag{6}
$$

Since $\rho _ { i } ^ { t } = n _ { i } ^ { t } / N ^ { t }$

$$
q _ { i } = S + \frac { n _ { i } ^ { t } } { ( N ^ { t } - n _ { i } ^ { t } ) + n _ { i } ^ { t } } ( \bar { r } _ { i } ^ { t } - S )\tag{7}
$$

and it is expressed as above.

For comparison, we consider the standard Beta-Bernoulli model for a Bernoulli distribution. Let $\alpha _ { 0 }$ and $\beta _ { 0 }$ be the two parameters of the initial prior distribution. We define the expectation $\mu _ { 0 }$ and the confidence $m _ { 0 }$ of the initial prior distribution as

$$
\mu _ { 0 } : = \frac { \alpha _ { 0 } } { \alpha _ { 0 } + \beta _ { 0 } } ,\tag{8}
$$

$$
m _ { 0 } : = \alpha _ { 0 } + \beta _ { 0 } .\tag{9}
$$

Then, the expectation of the posterior distribution given observed data is written as

$$
q _ { i } = \frac { \alpha _ { 0 } + n _ { i } ^ { t } \bar { r } _ { i } ^ { t } } { \alpha _ { 0 } + \beta _ { 0 } + n _ { i } ^ { t } }\tag{10}
$$

$$
= \mu _ { 0 } + \frac { n _ { i } ^ { t } } { m _ { 0 } + n _ { i } ^ { t } } ( \bar { r } _ { i } ^ { t } - \mu _ { 0 } ) .\tag{11}
$$

(Note that $\alpha _ { 0 } / ( m _ { 0 } + n _ { i } ^ { t } ) = \mu _ { 0 } \cdot m _ { 0 } / ( m _ { 0 } + n _ { i } ^ { t } ) = \mu _ { 0 } ( m _ { 0 } + n _ { i } ^ { t } - n _ { i } ^ { t } ) / ( m _ { 0 } + n _ { i } ^ { t } ) =$ $\mu _ { 0 } ( 1 - n _ { i } ^ { t } / ( m _ { 0 } + n _ { i } ^ { t } ) . )$ By comparing Eq. (7) with Eq. (11), we find that they have the same mathematical structure under the following correspondence:

$$
\mu _ { 0 } = S ,
$$

$$
m _ { 0 } = N ^ { t } - n _ { i } ^ { t } .\tag{12}
$$

(13)

Similar structural correspondences can also be established for the Gamma-Poisson model and the Gaussian-Gaussian model. Table 1 summarizes the correspondence between the initial prior expectation $\mu _ { 0 }$ and the initial confidence $m _ { 0 }$ for each model.

Our analysis suggests that, although ACB is not a strict Bayesian inference model, it has a Bayesian-like inference structure that integrates observed data and the prior satisfaction level.

Table 1: Expected Value and Confidence of the Initial Prior for Each Model.
<table><tr><td></td><td></td><td></td><td>Beta-Bernoulli Gamma-Poisson Gaussian-Gaussian</td><td>ACB</td></tr><tr><td>Initial Prior Expectation  $\mu _ { 0 }$ </td><td> $\alpha _ { 0 } / ( \alpha _ { 0 } + \beta _ { 0 } )$ </td><td> $\overline { { k / \lambda } }$ </td><td> $\mu _ { 0 }$ </td><td>S</td></tr><tr><td>Initial Confidence  $m _ { 0 }$ </td><td> $\alpha _ { 0 } + \beta _ { 0 }$ </td><td>λ</td><td> $\sigma ^ { 2 } / \sigma _ { 0 } ^ { 2 }$ </td><td> $N ^ { t } - n _ { i } ^ { t }$ </td></tr></table>

Table 2: Comparison of Properties among Standard-Bayes, Forgetful-Bayes, ACB, and FQL (forgetful Q-learning).
<table><tr><td>Properties</td><td>Standard-Bayes</td><td>Forgetful-Bayes</td><td>ACB FQL</td></tr><tr><td>Memory Persistence</td><td>Yes</td><td>No</td><td>Yes No</td></tr><tr><td>Belief Relaxation</td><td>No</td><td>Yes</td><td>Yes Yes</td></tr><tr><td>Initial Belief Variability</td><td>No</td><td>No</td><td>Yes No</td></tr></table>

## 4.3. Comparison of ACB (RS) and Existing Bayesian Models

Eq. (6) shows that the estimate of an unselected arm (with smaller $\rho )$ approaches the expectation S of the initial prior distribution. Since this behavior appears as if the observed data for unselected arms are being weakened, but actually not lost, we call it pseudo-forgetting in this paper. A key feature of pseudo-forgetting in ACB is that it is realized by the relative allocation of attention, without explicit data deletion or introducing a forgetting rate. This property is similar to the sharpening and blurring properties of LSVR [9], a precursor model to RS [8]. Sharpening refers to the property that estimates for frequently tried arms become “sharper”. Blurring refers to the property that estimates for rarely tried arms remain around the initial values (the individual diferences in the values are “blurred”). Thus, the pseudo-forgetting behavior of ACB shares features with the sharpening and blurring properties of LSVR. Reinterpreting RS as ACB also makes it possible to identify its commonalities with LSVR.

Examples of methods with a forgetting property include Forgetful-Bayes (Dynamic Belief Model) proposed by Zhang et al. [45] and FQL (Forgetful Q-learning) proposed by Toyama et al. [46]. These methods explicitly attenuate the influence of past data by introducing a forgetting rate. In contrast, in ACB, the accumulated data are retained, and only their relative influence changes. We call a general Bayesian inference model Standard-Bayes and summarize the properties of each method in Table 2. In Table 2, Memory persistence means that the observed data are not lost. Belief relaxation is that the estimate of an unselected arm approaches the initial prior distribution. Initial belief variability refers to that the efective prior strength, expressed as a pseudo-count, changes over time. In ACB, the fixed satisfaction level S corresponds to the prior mean, whereas $m _ { 0 , i } ^ { t } : = N ^ { t } - n _ { i } ^ { t }$ is the efective prior strength for arm i. This property is distinct from variation in either the prior mean or the normalized mixture weight $1 - \rho _ { i } ^ { t }$ . Although these properties can be theoretically inferred from the defining equations, we will verify them through simulations in Sec. 5.

## 5. Experiment 1: Comparison of ACB and Existing Bayesian Methods

In this section, we address the second question: how the relatively unchosen arms are treated. We conduct simulation experiments on ACB, Standard-Bayes, Forgetful-Bayes, and Forgetful Q-learning (FQL). The purpose of these experiments is not to compare their performances, but to visualize the time evolution of the internal variables maintained by each method and compare their properties. Specifically, we aim to empirically confirm the three properties shown in Table 2: Memory persistence, belief relaxation, and initial belief variability. Based on these results, we also discuss how pseudo-forgetting arises in ACB, which provides a Bayesian-like interpretation of RS.

Here, we summarize the comparison methods. Forgetful-Bayes assumes that environmental statistics change over time and performs Bayesian inference accordingly. In this model, the posterior distribution of the reward probability $\theta _ { i } ^ { t }$ for each arm i is updated as a weighted mixture of the posterior distribution at time step $t - 1$ and the initial prior distribution, as follows:

$$
\operatorname* { P r } ( \theta _ { i } ^ { t } = \theta \mid S _ { i } ^ { t - 1 } , F _ { i } ^ { t - 1 } ) = \gamma q _ { i } ^ { t - 1 } ( \theta ) + ( 1 - \gamma ) q _ { 0 } ( \theta ) ,\tag{14}
$$

where the weight $\gamma \in [ 0 , 1 ]$ . In Eq. (14), S and F denote the numbers of successes and failures, respectively, and $q _ { 0 }$ denotes the initial prior distribution, which follows Beta $( \alpha , \beta )$ . This update rule enables Forgetful-Bayes to discount past data and adapt to a changing environment. See [45] for details.

FQL is a Q-learning algorithm with forgetting, proposed by Toyama et al [46]. FQL is a method that extends standard RL algorithms to reproduce memory decay over time. For an unselected action a˜, FQL updates its value as follows:

$$
Q ( \tilde { a } )  Q ( \tilde { a } ) + \alpha _ { F } ( \mu - Q ( \tilde { a } ) ) .\tag{15}
$$

Here, $\alpha _ { F }$ is the forgetting-rate parameter $( 0 \leq \alpha _ { F } \leq 1 )$ , and $\mu$ is the defaultvalue parameter $( 0 ~ \leq ~ \mu ~ \leq ~ 1 )$ to which forgetting converges. See [46] for details. In this section, we reinterpret the internal state of FQL in a Bayesianlike manner to enable comparison as a Bayesian inference model. However, this reinterpretation is a formal framework intended only for comparing internal behavior, and it does not claim that FQL performs strict Bayesian inference. For details of this reinterpretation, see Appendix A.

## 5.1. Experiment Settings

We conducted this experiment under the Bernoulli bandit problem setting shown in Fig. 1. We set the number of arms to $K = 2$ and call them arm 1 and arm 2, respectively. We set the reward probability of arm 1 to 0.2 and that of arm 2 to 0.8. We ran 10 simulations and plotted the mean result.

For ACB, we set the satisfaction level S to 0.6. For Standard-Bayes and Forgetful-Bayes, we set $\alpha _ { 0 } = 2 . 4$ and $\beta _ { 0 } = 1 . 6$ so that the mean of the initial prior distribution equals 0.6. For FQL, we set $\mu = 0 . 6$ and set $\alpha _ { 0 } = \mu / \alpha _ { F }$ and $\beta _ { 0 } = ( 1 - \mu ) / \alpha _ { F }$ . In ACB and FQL, we set the initial sample mean of all arms to 0.6. For the initial value of the action selection counts $n _ { i } ^ { 0 }$ we initialized it to $1 0 ^ { - 4 }$ as a very small positive value to avoid making the definition of $\rho _ { i } ^ { 0 }$ undefined. We set the learning rate of FQL to $\alpha = 0 . 1$ and the forgetting rate to $\alpha _ { F } = 0 . 8$ . Although FQL does not originally require action selection counts, we introduce them as internal variables in this experiment for Bayesian-like comparison. The update of the action selection counts $n _ { i } ^ { t }$ in FQL follows the next equation:

$$
n _ { i } ^ { t + 1 }  \{ { n _ { i } ^ { t } + \alpha } \begin{array} { l l } { { ( i = a _ { t } ) , } } \\ { { ( 1 - \alpha _ { F } ) n _ { i } ^ { t } } } & { { ( i \neq a _ { t } ) . } } \end{array}\tag{16}
$$

For the selected arm, we increment the count weighted by the learning rate α. For unselected arms, we decay their count by a factor of $( 1 - \alpha _ { F } )$ to match the decay of $\Delta _ { t }$ in Eq. (A.4) (Appendix A).

In this experiment, to ensure a fair comparison across all methods, we controlled the action selection so that arm 1 and arm 2 were chosen alternately for the first 100 steps. From step 101 onward, each method selected arms using a greedy policy (action selection by argmax operator) based on its own internal estimates. In FQL, the agent selects arms based on its forgetful Q-values.

## 5.2. Results

The results are shown in Fig. 3. In short, in ACB, the estimate of arm 1, which is abandoned after step 100, returns to the satisfaction level 0.6 without its internal trial count decreasing. Below, we describe the results for each property: Memory persistence, belief relaxation, and initial belief variability.

Memory persistence: Fig. 3a shows memory persistence, that is, whether the number of trials (or the corresponding internal variables) that each method maintains internally decays or not. For Standard-Bayes and Forgetful Bayes, we plotted $\alpha _ { t } + \beta _ { t } - 2$ , and for ACB and FQL, we plotted $n _ { i } ^ { t }$ as the internal variables on a log–log scale. Note that this figure does not show the actual action selection counts but visualizes the count variables maintained internally by each method.

From step 101 onward, although the selection frequency of arm 1 decreases in all methods, the way the internal variables change difers substantially across methods. In Standard-Bayes and ACB, the internal action counts do not decrease even after arm 1 is no longer selected. In contrast, in Forgetful-Bayes, the internal action counts decrease exponentially as arm 1 stops being selected. In FQL, due to the efect of the forgetting rate, the action counts fluctuate repeatedly from the early stage of learning. After the switch to greedy action selection, they hardly increase. These results confirm that Standard-Bayes and ACB have memory persistence. They also confirm that Forgetful-Bayes and FQL do not have this property.

Belief relaxation: Fig. 3b shows belief relaxation, that is, the behavior that the estimate of an unselected arm approaches the expectation of the initial prior distribution. The solid gray line indicates the mean of the initial prior distribution. The dashed gray line indicates the true expected reward of each arm.

For arm 2, after the switch to greedy action selection, the estimates concentrate around the true expected reward in all methods, indicating that the estimation is appropriate. For arm 1, a clear diference is observed across these methods. In Standard-Bayes, even after arm 1 is no longer selected, the posterior distribution does not change and the estimate remains fixed. In contrast, in ACB, Forgetful-Bayes, and FQL, the estimates gradually converge to 0.6, which is the mean of the initial prior distribution. This behavior indicates that, through true forgetting or pseudo-forgetting mechanisms, the belief for an arm that receives no new observations is relaxed toward the initial belief. From this fact, we confirm that ACB, Forgetful-Bayes, and FQL have the belief relaxation property.

Initial belief variability: Fig. 3c shows the time evolution of the prior mean and efective prior strength. We visualize $\alpha _ { 0 } , \beta _ { 0 }$ , and the prior mean $\alpha _ { 0 } / ( \alpha _ { 0 } + \beta _ { 0 } )$ for each method. The prior mean remains constant in all methods, whereas $\alpha _ { 0 }$ and $\beta _ { 0 }$ , and thus the efective prior strength $\alpha _ { 0 } + \beta _ { 0 }$ , vary in ACB. Under the ACB correspondence, $\alpha _ { 0 , i } ^ { t } = S ( N ^ { t } - n _ { i } ^ { t } )$ and $\beta _ { 0 , i } ^ { t } = ( 1 - S ) ( N ^ { t } - n _ { i } ^ { t } )$ . At each step when arm i is not selected, $N ^ { t } - n _ { i } ^ { t }$ increases by one. Thus, both parameters increase linearly while the arm remains unselected. When arm i is selected, both $N ^ { t }$ and $n _ { i } ^ { t }$ increase by one, leaving its efective prior strength unchanged. This explains why the parameters increase for arm 1 after it is no longer selected, but change little for arm 2 while it continues to be selected. These results demonstrate initial belief variability in ACB, rather than variation in the prior mean.

## 5.3. Discussion

In this subsection, we discuss the mechanism of pseudo-forgetting in ACB. As shown in the first-row graph in Fig. 3, in ACB, even after arm 1 is no longer selected from step 101 onward, the internal trial count associated with arm 1 does not decrease. This reflects the fact that ACB does not use an explicit forgetting rate. On the other hand, in the second row of Fig. 3, we see that the estimate of arm 1 eventually converges to 0.6, which is the mean of the initial prior distribution. This means that ACB exhibits pseudoforgetting, where the belief returns to the initial prior without discarding past observations. The reason lies in the change in the confidence of the initial belief. As shown in the third row of Fig. 3, for arm 1, which is not selected, the parameters $\alpha _ { 0 }$ and $\beta _ { 0 }$ of the initial prior distribution increase. This increasing trend means that, for arms that receive no new observations, the relative influence of the initial belief is strengthened. Eq. (6) shows that the weight of the initial belief in ACB is $1 - \rho _ { i } ^ { t }$ , which corresponds to the allocation of attention in terms of the trial proportion. Thus, ACB, and consequently RS, can be interpreted as implementing pseudo-forgetting by changing the weights assigned to data rather than forgetting the data themselves.

## 6. Experiment 2: Performance Comparison of Satisficing RL Methods

In this section, we address the third question: what happens when the satisfaction level is unattainable. When the satisfaction level is too high, all arms remain in the under-achieved condition, and exploration by RS does not settle. We therefore compare, in terms of regret, RS with a fixed satisfaction level and RS that adjusts the level from observations. Comparisons with other satisficing RL methods and an optimizing method are shown as a benchmark check. We also discuss the results in terms of RS-value dynamics and satisfaction-level adjustment.

We compare the behavior of RS with SAT-UCB<sup>+</sup> [28] and Softsatisficing [15]. We also show the results of Thompson Sampling (TS)[29], a standard optimization algorithm, as a baseline for comparing satisficing algorithms . Michel et al. proposed $\mathrm { S A T \mathrm { - } U C B ^ { + } }$ as an experimental algorithm extending the satisficing RL algorithm SAT-UCB [28]. The theoretical guarantee for $\mathrm { S A T - U C B ^ { + } }$ is not yet fully established. The SAT-UCB<sup>+</sup> agent uses the given satisfaction level S as the threshold and switches its policy depending on whether the sample mean reward $\bar { r } _ { i } ^ { t }$ exceeds S. If there are arms that satisfy $\bar { r } _ { i } ^ { t } \geq S$ , it selects an arm among them based on the following equation:

$$
\begin{array} { r } { \beta _ { i } ^ { t } = \sqrt { \cfrac { 2 \log ( f ^ { t } ) } { n _ { i } ^ { t - 1 } } } \quad \mathrm { w h e r e } \quad f ^ { t } = 1 + t \log ^ { 2 } ( t ) } \\ { \mathrm { U C B } _ { i } ^ { t } : = \bar { r } _ { i } ^ { t } + \beta _ { i } ^ { t } , \quad \mathrm { L C B } _ { i } ^ { t } : = \bar { r } _ { i } ^ { t } - \beta _ { i } ^ { t } \qquad } \\ { a _ { t } = \arg \operatorname* { m a x } \left\{ \cfrac { \mathrm { U C B } _ { i } ^ { t } - \operatorname* { m a x } \{ S , \mathrm { L C B } _ { i } ^ { t } \} } { \beta _ { i } ^ { t } } \right\} \qquad } \end{array}\tag{17}
$$

In Eq. (17), UCB and LCB are the upper and lower confidence bounds, respectively, and $\beta _ { i } ( t )$ is the confidence width. If no satisfactory arm exists, it selects arms in the same way as the standard UCB1. The SAT-UCB<sup>+</sup> agent selects each arm at least once before choosing arms according to Eq. (17). See [28] for details.

Softsatisficing, proposed by Kamiya and Takahashi, is a probabilistic transformation of RS [15]. While the RS agent selects arms deterministically, the Softsatisficing agent selects arms according to a certain Softmax probability distribution, which allows its exploration behavior to be represented. Softsatisficing defines the satisfaction degree of each arm as $\delta _ { i } = \bar { r } _ { i } ^ { t } - S$ and the satisficing value as $E _ { i } ^ { \mathrm { R S } } = - \log ( S - \bar { r } _ { i } ^ { t } )$ . Using this $E _ { i } ^ { \mathrm { R S } }$ , it defines the action selection probability as

$$
\mathrm { S o f t s a t i s f i c i n g } ( a _ { i } ; S ) = \frac { \exp ( E _ { i } ^ { \mathrm { R S } } ) } { \sum _ { j } \exp ( E _ { j } ^ { \mathrm { R S } } ) }\tag{18}
$$

and then selects arm according to this probability. See [15] for details.

## 6.1. Experiment Settings

We conducted this experiment under the Bernoulli bandit problem setting shown in Fig. 1. We set the number of arms to $K = 8$ . The reward probability of each arm is sampled uniformly from [0, 1]. We ran 1,000 simulations and plotted the mean regret. As the satisfaction level S, we used the following two values:

• the optimal satisfaction level $S _ { \mathrm { o p t } }$ (see Eq. 1),

• the upper bound on the task reward 1.0

We conducted the experiment with 1.0 to examine how each method behaves when the satisfaction level is set to a relatively high value.

For RS, we set the initial sample mean of all arms to 0.5. We initialized the initial action selection count $n _ { i } ^ { 0 }$ with a very small positive value 10<sup>−4</sup> to avoid making the definition of the trial proportion $\rho _ { i } ^ { 0 }$ undefined. In this experiment, we used both the standard RS and RS equipped with a mechanism for dynamically adjusting the satisfaction level, namely, dynamically adjusted RS. This dynamic satisfaction-level adjustment mechanism exploits properties of RS. The reasons for introducing this mechanism and its derivation are described in Appendix B. Algorithm 1 summarizes the adjustment procedure. With this mechanism, the satisfaction level of dynamically adjusted RS can change during a simulation. We denote the satisfaction level of standard RS by $S ,$ and the original satisfaction level of dynamically adjusted RS by $S _ { 0 }$ . Algorithm 1 assumes that the empirically best arm g is unique and that its sample mean falls short of the initial satisfaction level $( \bar { r } _ { g } < S _ { 0 } )$ . The input is the initial satisfaction level $S _ { 0 }$ , not the result of the previous adjustment. At each step t, the procedure uses the current counts $n _ { i } ^ { t }$ and sample means $\bar { r } _ { i } ^ { t }$ . RS then compares arms using the resulting adjusted satisfaction level $S ^ { \prime }$ . The algorithm omits the superscript t. $D _ { \mathrm { K L } } ( \cdot | | \cdot )$ denotes the KL divergence, and Ber $( p )$ denotes the Bernoulli distribution with mean $p .$ . For TS, we set the parameters of the initial Beta distribution to $\alpha _ { 0 } = 1 . 0$ and $\beta _ { 0 } = 1 . 0$

Algorithm 1 Upper-limit-based satisfaction-level adjustment   
Input: Initial satisfaction level $S _ { 0 } \in [ 0 , 1 ]$ , counts $n _ { i } > 0$ , and means $\bar { r } _ { i } \in$   
[0, 1] for $i = 1 , \ldots , K$ , with $K \geq 2$   
Output: Adjusted satisfaction level $S ^ { \prime }$   
1: $g  \mathrm { a r g }$ max $\bar { r _ { i } }$ ▷ unique by assumption   
i   
2: for each $j \in \{ 1 , \ldots , K \} \setminus \{ g \}$ do   
3: $D _ { j }  D _ { \mathrm { K L } } ( \mathrm { B e r } ( \bar { r } _ { j } ) \vert \vert$ Ber $\left( \hat { r } _ { g } \right) )$   
4: $\tilde { \tau } _ { j } \gets \exp ( - n _ { j } D _ { j } )$   
5: $S _ { j }  \bar { r } _ { g } + \frac { \tau _ { j } } { 1 - \tilde { \tau } _ { j } } ( \bar { r } _ { g } - \bar { r } _ { j } )$   
6: end for   
7: $S _ { \mathrm { U L } }  \operatorname* { m a x } _ { j \neq g } S _ { j }$   
8: $S ^ { \prime } \gets \operatorname* { m i n } \{ S _ { 0 } , S _ { \mathrm { U L } } \}$   
9: return $S ^ { \prime }$

## 6.2. Results

We show the results in Fig. 4a and Fig. 4b. Fig. 4a shows the time evolution of regret when the satisfaction level is set to $S _ { \mathrm { o p t } } \ ( \mathrm { o r } \ S _ { 0 } = S _ { \mathrm { o p t } } )$ . Under this setting, RS and Softsatisficing achieve lower regret than ${ \mathrm { S } } { \dot { \mathrm { A } } } { \mathrm { T } } { \mathrm { - } } { \mathrm { U C B } } ^ { + }$ Fig. 4b shows the results when the satisfaction level is set to 1.0. Even in this case, dynamically adjusted RS achieves lower regret than $\mathrm { S A T \mathrm { - } U C B ^ { + } }$ These results show that, in the settings we tested, satisfaction-level adjustment mitigates the efects of an excessively high initial satisfaction level in RS. By contrast, standard RS and Softsatisficing do not have an adjustment mechanism for the satisfaction level, and its regret linearly increases over time. The plots for standard RS and Softsatisficing overlap almost completely.

## 6.3. Discussion

In the current definition, the RS value is normalized by the trial proportion $\rho _ { i } = n _ { i } / N$ and is written as $I _ { i } = \rho _ { i } ( \bar { r } _ { i } - S )$ . In the original proposal of RS, the value was not normalized by the trial proportion and was given by $n _ { i } ( \bar { r } _ { i } - S )$ [8]. We now derive the value corresponding to this original definition using the current notation. Multiplying the current definition by

N gives

$$
\begin{array} { c } { { N I _ { i } = N \rho _ { i } ( \bar { r } _ { i } - S ) } } \\ { { \ } } \\ { { \ } } \\ { { \ } } \\ { { \displaystyle = N \frac { n _ { i } } { N } ( \bar { r } _ { i } - S ) } } \\ { { \ } } \\ { { \displaystyle = n _ { i } ( \bar { r } _ { i } - S ) . } } \end{array}\tag{19}
$$

Henceforth, we write this unnormalized value as $N I _ { i } \mathrm { t o }$ make clear that it is the product of N and $I _ { i }$ . In this discussion, we examine how this value changes under a fixed satisfaction level S. At any given time, $N > 0$ is common to all arms, so $N I _ { i }$ and $I _ { i }$ rank the arms identically and yield the same action choice. Let $I _ { i } ^ { + }$ denote the normalized RS value after selecting arm i once and receiving reward r. Since the total trial count also increases from N to $N + 1$ , the increment in the unnormalized value $N I _ { i }$ is

$$
\Delta ( N I _ { i } ) : = ( N + 1 ) I _ { i } ^ { + } - N I _ { i } = r - S .\tag{20}
$$

Taking the expectation over the reward, conditional on the state before selection and on selecting arm i, gives

$$
E [ \Delta ( N I _ { i } ) ] = E [ r _ { i } ] - S .\tag{21}
$$

Here, $\Delta ( N I _ { i } )$ denotes the change in the entire product and difers from $N \Delta I _ { i }$ Thus, the expected value of $N I _ { i }$ increases after selecting arm i if $E [ r _ { i } ] > S$ and decreases if $E [ r _ { i } ] < S$ . If the satisfaction level lies between $E [ r _ { \mathrm { 1 s t } } ]$ and $E [ r _ { \mathrm { 2 n d } } ]$ , namely, $E [ r _ { 2 \mathrm { n d } } ] < S < E [ r _ { 1 \mathrm { s t } } ]$ , selecting the best arm increases its $N I _ { i }$ on average. In contrast, the unnormalized values $N I _ { i }$ of suboptimal arms decrease on average as those arms continue to be selected. RS can autonomously escape from situations in which it continues to select subopti mal arms. Tamatsukuri & Takahashi showed that, under this condition, the regret of RS is bounded above by a finite value [16].

The lower regret of standard RS in Fig. 4a is consistent with the theoretical property established above for a fixed satisfaction level. Dynamically adjusted RS also achieved low regret in this figure, although this empirical result is separate from the guarantee for a fixed satisfaction level. In contrast, for standard RS in Fig. 4b, 1.0 is at least as large as $E [ r _ { \mathrm { 1 s t } } ]$ . Hence, $E [ r _ { i } ] - 1 . 0 \leq 0$ for all arms. In this under-achieved setting, even the unnormalized value $N I _ { i }$ of the best arm does not increase on average. Therefore, the condition for the finite-regret result described above is not satisfied. As shown in Fig. 4b, the regret of standard RS rapidly increased in a linear way, overlapping with that of Softsatisficing.

The dynamically adjusted RS balances exploration and exploitation by lowering an excessively high satisfaction level based on observed data. The adjusted satisfaction level need not lie between the true mean rewards of the best and second-best arms, and changing the satisfaction level also changes $N I _ { i }$ . Therefore, the expected-increment argument and finite-regret guarantee for a fixed satisfaction level do not directly apply to the dynamic adjustment mechanism. In Fig. 4b, dynamic adjustment yielded lower regret than keeping the satisfaction level fixed at 1.0. These results show empirically that, in the tested setting, satisfaction-level adjustment can curb excessive exploration and reduce regret.

## 7. Discussion

In this section, we summarize the answers to the three questions in Introduction and discuss their implications and limitations.

## 7.1. The Answers and Two Constraints on Choice

The reinterpretation as ACB shows that action selection in RS is determined by a combination of two constraints. One is the satisfaction level S, and the other is the finite allocation of trials, $\textstyle \sum _ { i } \rho _ { i } ^ { t } = 1$ . The fewer trials an arm has received, the more its estimate is described by the satisfaction level rather than by observations (Section 4.2). The three results of this paper fol low from this single structure. First, in the under-achieved condition, rarely tried arms are valued highly because of the satisfaction level, which promotes exploration (Section 3.2). Second, the estimates of arms that are no longer selected return to the satisfaction level without discarding data (Section 5). Third, when the satisfaction level is unattainable, exploration does not settle, and adjusting the level from observations reduced regret in the tested settings (Section 6). Maximization can be positioned as a special case of this framework. When the satisfaction level lies between the expected rewards of the best and second-best arms, only one arm satisfies it (Section 2.1).

## 7.2. Relation to Deacon’s Framework

As noted in Section 1, Deacon called phenomena organized with respect to something not yet present ententional, and García-Valdecasas and Deacon argued that ends are represented by constraints rather than by particular states [6, 7]. The satisfaction level S of RS specifies a range of acceptable outcomes rather than a single best arm. Moreover, in the under-achieved condition, S orders the arms even though no arm has reached it. In this sense, RS can be regarded as a minimal formal model of action selection referenced to a constraint that is not yet realized. However, S is given from outside the model. The model therefore captures the logical form of such organization, not its origin. In Deacon’s framework, the origin of goal-directedness requires a system that maintains the constraints on which its own persistence depends, and a bandit agent does not have this property.

## 7.3. Possible Biological Implications

Because we have not compared the model with behavioral data, we state the following as hypotheses. First, pseudo-forgetting yields a testable prediction about how animals revalue options that they have abandoned. In ACB, the estimate of an arm that is no longer selected approaches neither zero nor its last observed value, but the current satisfaction level. In contrast, Forgetful-Bayes approaches the mean of the initial prior, and FQL approaches the default value $\mu ;$ both values are independent of the satisfaction level [45, 46]. In Experiment 1, we set all of these values to 0.6, so this diference did not appear. A behavioral experiment that manipulates only the aspiration level, for example through deprivation or energy requirements as in studies of risk-sensitive foraging [5], might distinguish these models. Note, however, that relaxation in ACB advances with trials on other arms, not with the passage of time alone.

Second, weighting by the trial proportion $\rho _ { i } = n _ { i } / \sum _ { k } n _ { k }$ has the simplest form of divisive normalization, which is regarded as a canonical neural computation [47]. In value representation, value signals have also been reported to be divisively normalized by the values of the available options [48, 49]. In RS, however, what is normalized is experience (the number of trials), not value, and whether this formal similarity reflects a shared mechanism remains an open empirical question.

Third, the satisfaction-level adjustment in Appendix B modifies one constraint, the satisfaction level, by another rule based on observations. It can be seen as a simple example of regulating a constraint, although the rule itself is designed.

## 7.4. Limitations and Future Work

This study has several limitations. First, the satisfaction level S is given exogenously. Linking S to the agent’s own viability, for example to its energy budget, is one direction toward the self-maintaining case that Deacon considers. Second, the experiments and the adjustment in the appendix are limited to Bernoulli rewards and stationary environments. The correspondence of ACB also holds for the Gamma-Poisson and Gaussian-Gaussian models (Table 1), but we have not examined the behavior in those cases. Third, RS selects arms deterministically, and we have not fitted it to behavioral data. A stochastic version such as Softsatisficing is more suitable for such comparisons [15]. Lastly, the satisfaction level adjusment method in Algorithm 1, described in Appendix B, is conceptually described but not yet mathematically guaranteed to be optimal.

## 8. Conclusion

We asked how a goal that is not yet attained can organize choice, and we examined this question in a minimal model, Risk-sensitive Satisficing (RS). By reinterpreting RS as Attention Competitive Bayes (ACB), we showed that the satisfaction level enters the valuation of every arm as its prior mean, and that the finite allocation of trials determines how far observations override it. Two constraints, what counts as good enough and how many trials are available, thus jointly shape behavior. Three consequences follow from this structure. Rarely tried arms are explored because they are described by the unattained satisfaction level. The estimates of abandoned arms return to the satisfaction level without discarding data (pseudo-forgetting). When the satisfaction level is unattainable, exploration does not settle; in the tested Bernoulli bandit settings, adjusting the level from observations reduced regret. In the light of Deacon’s framework, RS can be regarded as a minimal formal model of action selection referenced to a constraint that is not yet realized. Because the satisfaction level is given from outside, however, the model only captures the form of such organization, and not its origin. Future work should derive the satisfaction level from the agent’s own viability, test the prediction of pseudo-forgetting in animal and human choice, and extend the analysis beyond Bernoulli rewards.

## 9. Declaration of generative AI and AI-assisted technologies in the manuscript preparation process

During the preparation of this work, M. Shinriki used Codex (OpenAI) and Claude Code (Anthropic) in order to translate the part of the manuscript he authored from Japanese into English. After using these tools, the authors thoroughly reviewed and edited the content as needed and take full responsibility for the content of the published article.

## 10. Acknowledgment

This work was partially supported by JSPS KAKENHI Grant Numbers 23H03469, 23K28159, and 20H04259, and Research Institute for Science and Technology of Tokyo Denki University Grant Number Q26D-03 / Japan.

## References

[1] H. A. Simon, A behavioral model of rational choice, The quarterly journal of economics (1955) 99–118.

[2] H. A. Simon, Rational choice and the structure of the environment., Psychological review 63 (2) (1956) 129.

[3] T. Caraco, S. Martindale, T. S. Whittam, An empirical demonstration of risk-sensitive foraging preferences, Animal Behaviour 28 (3) (1980) 820–830. doi:10.1016/S0003-3472(80)80142-4.

[4] D. W. Stephens, The logic of risk-sensitive foraging preferences, Animal Behaviour 29 (2) (1981) 628–629. doi:10.1016/S0003-3472(81)80128-5.

[5] M. Bateson, Recent advances in our understanding of risk-sensitive foraging preferences, Proceedings of the Nutrition Society 61 (4) (2002) 509–516.

[6] T. W. Deacon, Incomplete Nature: How Mind Emerged from Matter, W. W. Norton & Company, New York, 2012.

[7] M. García-Valdecasas, T. W. Deacon, Origins of biological teleology: how constraints represent ends, Synthese 204 (2) (2024) 75. doi:10.1007/s11229-024-04705-w.

[8] T. Takahashi, Y. Kono, D. Uragami, Cognitive satisficing: Bounded rationality in reinforcement learning, Transactions of the Japanese Society for Artificial Intelligence 31 (6) (2016) AI30–M\_1–11, (in Japanese).

[9] Y. Kohno, T. Takahashi, A cognitive satisficing strategy for bandit problems, International Journal of Parallel, Emergent and Distributed Systems 32 (2) (2017) 232–242.

[10] K. Oyo, T. Takahashi, Optimization through satisficing with prospects, in: AIP Conference Proceedings, Vol. 1863, AIP Publishing, 2017, p. 360013.

[11] H. Wakabayashi, T. Kamiya, T. Takahashi, Balancing policy improvement and evaluation in risk-sensitive satisficing algorithm, in: Advances in Artificial Intelligence: Selected Papers from the Annual Conference of Japanese Society of Artificial Intelligence (JSAI 2020), Springer, 2020, pp. 175–182.

[12] M. Shinriki, H. Wakabayashi, Y. Kono, T. Takahashi, Flexibility of emulation learning from pioneers in nonstationary environments, in: Annual Conference of the Japanese Society for Artificial Intelligence, Springer, 2019, pp. 90–101.

[13] A. Tsuboya, Y. Kono, T. Takahashi, Reinforcement learning with a focus on adjusting policies to reach targets, arXiv preprint arXiv:2412.17344 (2024).

[14] A. Tsuboya, Y. Kono, T. Takahashi, Target-oriented exploration guided by anticipated returns, Artificial Life and Robotics (in press) (2025).

[15] T. Kamiya, T. Takahashi, Softsatisficing: Risk-sensitive softmax action selection, Biosystems 213 (2022) 104633.

[16] A. Tamatsukuri, T. Takahashi, Guaranteed satisficing and finite regret: Analysis of a cognitive satisficing value function, Biosystems 180 (2019) 46–53.

[17] R. S. Sutton, A. G. Barto, Reinforcement Learning: An Introduction, MIT Press, Cambridge, MA, 1998.

[18] K. Murakoshi, J. Mizuno, A parameter control method in reinforcement learning to rapidly follow unexpected environmental changes, Biosystems 77 (1-3) (2004) 109–117.

[19] K. Murakoshi, T. Noguchi, Simulation of rat behavior by a reinforcement learning algorithm in consideration of appearance probabilities of reinforcement signals, BioSystems 80 (1) (2005) 83–90.

[20] D. Uragami, T. Takahashi, Y. Matsuo, Cognitively inspired reinforcement learning architecture and its application to giant-swing motion control, Biosystems 116 (2014) 1–9.

[21] D. Silver, T. Hubert, J. Schrittwieser, et el., Mastering chess and shogi by self-play with a general reinforcement learning algorithm (2017).

[22] P. R. Wurman, S. Barrett, K. Kawamoto, et el., Outracing champion gran turismo drivers with deep reinforcement learning, Nature 602 (7896) (2022) 223–228.

[23] D. Uragami, Y. Kohno, T. Takahashi, Robotic action acquisition with cognitive biases in coarse-grained state space, BioSystems 145 (2016) 41–52.

[24] K. Ochi, M. Kamiura, Overtaking method based on sand-sifter mechanism: Why do optimistic value functions find optimal solutions in multiarmed bandit problems?, Biosystems 135 (2015) 55–65.

[25] M. Kamiura, K. Sano, Optimism in the face of uncertainty supported by a statistically-designed multi-armed bandit algorithm, Biosystems 160 (2017) 25–32.

[26] D. Abel, M. K. Ho, A. Harutyunyan, Three dogmas of reinforcement learning, Reinforcement Learning Journal 2 (2024) 629–644.

[27] R. Ortner, Reinforcement Learning: Beyond optimality, https: //infotech.unileoben.ac.at/fileadmin/shares/unileoben/ lehrstuehle\_department\_institute/infotech/personal-sites/ ronald\_ortner/Proposal-TAI-590.pdf, [Online; accessed 30-Oct-2026].

[28] T. Michel, H. Hajiabolhassan, R. Ortner, Regret bounds for satisficing in multi-armed bandit problems, Transactions on Machine Learning Research 3 (2022).

[29] W. R. Thompson, On the likelihood that one unknown probability exceeds another in view of the evidence of two samples, Biometrika 25 (3-4) (1933) 285–294.

[30] E. Kaufmann, N. Korda, R. Munos, Thompson sampling: An asymptotically optimal finite-time analysis, in: International conference on algorithmic learning theory, Springer, 2012, pp. 199–213.

[31] S. Agrawal, N. Goyal, Analysis of thompson sampling for the multiarmed bandit problem, in: Conference on learning theory, JMLR Workshop and Conference Proceedings, 2012, pp. 39–1.

[32] S.-J. Kim, M. Aono, M. Hara, Tug-of-war model for the two-bandit problem: Nonlocally-correlated parallel exploration via resource conservation, BioSystems 101 (1) (2010) 29–36.

[33] D. Uragami, N. Sonota, T. Takahashi, Social satisficing: Multi-agent reinforcement learning with satisficing agents, BioSystems 243 (2024) 105276.

[34] P. Auer, N. Cesa-Bianchi, P. Fischer, Finite-time analysis of the multiarmed bandit problem (2002).

[35] D. Silver, A. Huang, C. J. Maddison, et al., Mastering the game of go with deep neural networks and tree search, nature 529 (7587) (2016) 484–489.

[36] P. B. Reverdy, V. Srivastava, N. E. Leonard, Modeling human decision making in generalized gaussian multiarmed bandits, Proceedings of the IEEE 102 (4) (2014) 544–571.

[37] P. Reverdy, V. Srivastava, N. E. Leonard, Satisficing in multi-armed bandit problems, IEEE Transactions on Automatic Control 62 (8) (2016) 3788–3803.

[38] A. Locatelli, M. Gutzeit, A. Carpentier, An optimal algorithm for the thresholding bandit problem, in: International Conference on Machine Learning, PMLR, 2016, pp. 1690–1698.

[39] S. Mukherjee, K. P. Naveen, N. Sudarsanam, B. Ravindran, Thresholding bandits with augmented ucb, arXiv preprint arXiv:1704.02281 (2017).

[40] H. Kano, J. Honda, K. Sakamaki, K. Matsuura, A. Nakamura, M. Sugiyama, Good arm identification via bandit feedback, Machine Learning 108 (2019) 721–745.

[41] Y. Wu, R. Sharif, T. Lattimore, C. Szepesvári, Conservative bandits, in: International Conference on Machine Learning, PMLR, 2016, pp. 1254–1262.

[42] N. Merlis, S. Mannor, Lenient regret for multi-armed bandits, in: Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 35, 2021, pp. 8950–8957.

[43] D. Russo, B. Van Roy, Satisficing in time-sensitive bandit learning, Mathematics of Operations Research 47 (4) (2022) 2815–2839.

[44] R. Munos, From bandits to monte-carlo tree search: The optimistic principle applied to optimization and planning, Foundations and Trends® in Finance 7 (1) (2014) 1–129.

[45] S. Zhang, A. J. Yu, Forgetful Bayes and myopic planning: Human learning and decision-making in a bandit setting, Advances in Neural Information Processing Systems 26 (2013) 2607–2615.

[46] A. Toyama, K. Katahira, H. Ohira, Reinforcement learning with parsimonious computation and a forgetting process, Frontiers in human neuroscience 13 (2019) 153.

[47] M. Carandini, D. J. Heeger, Normalization as a canonical neural computation, Nature Reviews Neuroscience 13 (1) (2012) 51–62. doi:10.1038/nrn3136.

[48] K. Louie, L. E. Grattan, P. W. Glimcher, Reward value-based gain control: divisive normalization in parietal cortex, Journal of Neuroscience 31 (29) (2011) 10627–10639. doi:10.1523/JNEUROSCI.1237-11.2011.

[49] K. Louie, P. W. Glimcher, Eficient coding and the neural representation of value, Annals of the New York Academy of Sciences 1251 (1) (2012) 13–32. doi:10.1111/j.1749-6632.2012.06496.x.

[50] J. Honda, A. Takemura, Non-asymptotic analysis of a new bandit algorithm for semi-bounded rewards, The Journal of Machine Learning Research 16 (1) (2015) 3721–3756.

## Appendix A. A Bayesian-like Interpretation of FQL as Sequential Updating

In Sec. 5, we compare the methods from the viewpoint of Bayesian inference models. Under this comparison, FQL must also be interpreted as a Bayes-like inference model. In this subsection, we show that although FQL is not strictly a Bayesian inference model, its update structure closely resembles that of sequential Bayesian updating. This interpretation provides a supplementary perspective for understanding FQL in relation to Bayesian inference models.

In FQL, a forgetting update that decays past value estimates is applied to unchosen actions a˜. The update rule is written as:

$$
Q _ { t + 1 } ( \tilde { a } )  Q _ { t } ( \tilde { a } ) + \alpha _ { F } ( \mu - Q _ { t } ( \tilde { a } ) )\tag{A.1}
$$

$$
= ( 1 - \alpha _ { F } ) Q _ { t } ( \tilde { a } ) + \alpha _ { F } \mu .\tag{A.2}
$$

Here, $\alpha _ { F } \in ( 0 , 1 )$ is the forgetting rate, and $\mu$ is the default value to which forgetting converges. $Q _ { t } ( \tilde { a } )$ is the action value of action $\tilde { a }$ at time step $t ,$ and in FQL, action selection is performed based on this $Q$ value.

For a formal comparison with Bayesian inference, we express $Q _ { t }$ in terms of the deviation from the initial value $Q _ { 0 }$ . In the experiment in Sec. 5, the initial value $Q _ { 0 }$ is set the default value $\mu ,$ and thus we assume $Q _ { 0 } = \mu$ in this subsection. Defining the deviation between $Q _ { t }$ and the initial value as

$$
\Delta _ { t } : = Q _ { t } - Q _ { 0 } ,\tag{A.3}
$$

we can write $\Delta _ { t + 1 }$ as follows:

$$
\begin{array} { r l } & { \Delta _ { t + 1 } = Q _ { t + 1 } - Q _ { 0 } } \\ & { \qquad = ( 1 - \alpha _ { F } ) ( Q _ { t } - Q _ { 0 } ) + \alpha _ { F } ( \mu - Q _ { 0 } ) } \\ & { \qquad = ( 1 - \alpha _ { F } ) \Delta _ { t } . } \end{array}\tag{A.4}
$$

From this recurrence relation, $\Delta _ { t }$ is given by

$$
\Delta _ { t } = ( 1 - \alpha _ { F } ) ^ { t - 1 } \Delta _ { 1 } .\tag{A.5}
$$

That is, the forgetting update in FQL exhibits exponential regression toward $Q _ { 0 }$ This exponential decay is structurally similar to the behavior of the corresponding update equation in sequential Bayesian inference, where the relative influence of the prior distribution weakens as observations accumulate.

Next, we consider the correspondence between FQL and Bayesian inference in the Beta-Bernoulli model. Let $q _ { t - 1 }$ be the mean of the prior distribution and $q _ { t }$ be the mean of the posterior distribution. When a successful reward (1) is observed, the update is given as follows:

$$
\begin{array} { c } { q _ { t } = \displaystyle \frac { \alpha _ { t } + 1 } { \alpha _ { t } + \beta _ { t } + 1 } } \\ { q _ { t } - q _ { t - 1 } = \displaystyle \frac { 1 } { \alpha _ { t } + \beta _ { t } + 1 } ( 1 - q _ { t - 1 } ) . } \end{array}\tag{A.6}
$$

In the context of RL, this expression can be generalized as

$$
q _ { t } - q _ { t - 1 } = \frac { 1 } { \alpha _ { t } + \beta _ { t } + 1 } ( r - q _ { t - 1 } ) .\tag{A.7}
$$

Comparing Eq. (A.1) and Eq.(A.7), the forgetting rate $\alpha _ { F }$ in FQL can be associated with the learning-rate term in Bayesian inference as follows:

$$
\alpha _ { F } \longleftrightarrow \frac { 1 } { \alpha _ { 0 } + \beta _ { 0 } + 1 } .\tag{A.8}
$$

Thus, the forgetting update in FQL can be interpreted as assigning unselected actions a pseudo-reward of $\mu$ with learning rate $\alpha _ { F }$ . When α and $\beta$ are suficiently large, the efect of the constant term 1 can be neglected, and thus we can approximate

$$
\alpha _ { F } \approx \frac { 1 } { \alpha _ { 0 } + \beta _ { 0 } } .\tag{A.9}
$$

Note that this correspondence is not a strict equivalence but an approximate analogy in the limit of a large number of trials.

From the above, we can see that the forgetting update in FQL has an update structure that is formally similar to the weighted updates in sequential Bayesian inference, in that it exhibits exponential regression toward the initial value. In this paper, based on this Bayesian-like interpretation, we position FQL as one of the comparison methods in the experiments in Sec. 5.

## Appendix B. Upper-Limit-Based Dynamic Adjustment of the Satisfaction Level

This appendix explains the functionality and the mechanism of the dynamic adjustment of the satisfaction level used by dynamically adjusted RS in Sec. 6.1. When the satisfaction level remains unattainable, RS continues exploring toward an equilibrium in which trial proportions compensate for diferences in how far sample means fall below the satisfaction level [15]. Such exploration is reasonable for satisficing, but it can be excessive when one would prefer to exploit higher-reward arms even though the satisfaction level is unattainable. This appendix presents the method used in Exp. 2 (the results in Fig 4a and 4b for RS with $S _ { 0 } )$ for using empirical information to lower the satisfaction level and adjust the balance between exploration and exploitation. Algorithm 1 in Sec. 6.1 shows the adjustment procedure. The following subsections explain Algorithm 1 line by line.

## Appendix B.1. Setting, Assumptions, and the Empirically Best Arm

Algorithm 1 takes the initial satisfaction level $S _ { 0 }$ , the trial counts $n _ { i } .$ and the sample means $\bar { r _ { i } }$ as inputs, and it outputs the adjusted satisfaction level $S ^ { \prime }$ . We consider Bernoulli rewards and an original satisfaction level $S _ { 0 } ~ \in ~ [ 0 , 1 ]$ We describe the adjustment at a fixed time step t and omit the superscript t for simplicity. We use the RS value $I _ { i } = \rho _ { i } ( \bar { r } _ { i } - S )$ and trial proportion $\rho _ { i } = n _ { i } / \sum _ { k } n _ { k }$ defined in the main text. In the numerical implementation, trial counts are initialized to $1 0 ^ { - 4 }$ (to avoid indeterminate $\rho _ { i }$ values from division by zero) and sample means to 0.5. Line 1 of Algorithm 1 selects the arm with the largest sample mean. Let $g = \mathrm { a r g }$ max $\bar { r _ { i } }$ denote the i empirically best arm, let $J = [ [ 1 , K ] \setminus \{ g \}$ be the set of all other arms, and define $\tau _ { j } = n _ { j } / n _ { g } = \rho _ { j } / \rho _ { g }$ . For the following derivation, we assume that the empirically best arm is unique and that $\bar { r } _ { g } ~ < ~ { \cal S } _ { 0 }$ Let $S ^ { \prime }$ denote the adjusted satisfaction level and $I _ { i } ^ { \prime } : = \rho _ { i } ( { \bar { r } } _ { i } - S ^ { \prime } )$ the changed RS value. We hold the observed data and trial proportions fixed to examine how changing the satisfaction level afects the current arm comparison. Setting $S ^ { \prime } = \bar { r } _ { g }$ gives g an RS value of zero and every other arm a negative value. Thus, g is the sole maximizer in the current comparison. Since $g$ is not always the optimal arm, we restrict the adjustment to $S ^ { \prime } \in ( \bar { r } _ { g } , S _ { 0 } ]$

Appendix B.2. Quantifying Evidence Against Each Arm Having a Higher True Mean

Line 2 iterates over all arms $j \in J$ other than $g .$ . Line $3$ computes the KL term $D _ { j }$ for each pair $( j , g )$ . We need a criterion for deciding whether to exploit the current empirically best arm $g$ or further explore another arm $j .$ This decision should account for both the diference in sample means and the amount of evidence supporting it. We therefore draw on the IMED algorithm, a bandit policy that computes an index for each arm from the observed data and selects an arm with the smallest index [50]. The index is a score for deciding which arm should be chosen, rather than a reward prediction. IMED constructs this index from the KL divergence between Bernoulli distributions. We abbreviate the KL divergence between Bernoulli distributions with means $p$ and $q$ as $d ( p , q ) : = D _ { \mathrm { K L } } ( \mathrm { B e r } ( p ) \parallel \mathrm { B e r } ( q ) )$ . It is given by

$$
d ( p , q ) = p \log \frac { p } { q } + ( 1 - p ) \log \frac { 1 - p } { 1 - q } ,\tag{B.1}
$$

with 0 · log $0 ~ = ~ 0$ . To assess whether arm $j$ might actually beat ${ \mathit { g } } ,$ we consider the hypothesis that the true reward mean of $j$ is some $q > \bar { r } _ { g } .$ The $n _ { j }$ Bernoulli observations of $j$ are best fit by the maximum-likelihood estimate $\bar { r } _ { j }$ , that is, by the empirical distribution $\mathrm { B e r } ( \bar { r } _ { j } )$ . Because $n _ { j } \bar { r } _ { j }$ of the $n _ { j }$ observations are successes, the log likelihood under mean $q$ is $\ell _ { j } ( q ) =$ $n _ { j } \{ \bar { r } _ { j } \log q + ( 1 - \bar { r } _ { j } ) \log ( 1 - q ) \}$ . Assuming the true mean to be $q$ rather than $\bar { r } _ { j }$ therefore lowers the log likelihood by

$$
\begin{array} { c l c r } { \ell _ { j } ( \bar { r } _ { j } ) - \ell _ { j } ( q ) = n _ { j } \Big \{ \bar { r } _ { j } \log \displaystyle \frac { \bar { r } _ { j } } { q } + ( 1 - \bar { r } _ { j } ) \log \displaystyle \frac { 1 - \bar { r } _ { j } } { 1 - q } \Big \} } \\ { = n _ { j } d ( \bar { r } _ { j } , q ) . } \end{array}\tag{B.2}
$$

The larger this decrease, the more strongly the observations contradict the hypothesis. Because $\bar { r } _ { j } < \bar { r } _ { g } < 1$ under our assumptions, $d ( \bar { r } _ { j } , q )$ is continuous and increasing in $q$ on $[ \bar { r } _ { g } , 1 )$ . By this monotonicity and continuity, the infimum of the KL divergence over all such $q$ is in $\mathrm { f } _ { q > \bar { r } _ { g } } d ( \bar { r } _ { j } , q ) = d ( \bar { r } _ { j } , \bar { r } _ { g } )$ Line 3 therefore sets $D _ { j } = d ( \bar { r } _ { j } , \bar { r } _ { g } ) = D _ { \mathrm { K L } } ( \mathrm { B e r } ( \bar { r } _ { j } ) \parallel \tilde { \mathrm { B e r } } ( \bar { r } _ { g } ) )$ . This infimum explains why the KL term takes the sample means as its arguments and why $\bar { r } _ { j }$ comes first.

## Appendix B.3. Deriving the Target Relative Exploration Ratio from the IMED Index

Line 4 converts the KL term into the target relative exploration ratio $\tilde { \tau } _ { j }$ For Bernoulli rewards with $\bar { r } _ { g } ~ < ~ 1$ IMED assigns arm $j$ the index $n _ { j } D _ { j } + \log n _ { j }$ . The first term, $n _ { j } D _ { j }$ , is the evidence accumulated from the observations of $j$ against alternatives in which $j$ beats $g .$ The second term, log $n _ { j } ,$ grows with the number of trials of $j$ . A small index therefore indicates that there is little evidence against $j$ being optimal or that $j$ has few trials. IMED selects such an arm for further exploration. We use this principle to design the RS satisfaction level through pairwise comparisons, rather than adopting the IMED selection rule itself. We define IMED-type indices for $j$ and $g$ as

$$
L _ { j } = n _ { j } d ( { \bar { r } } _ { j } , { \bar { r } } _ { g } ) + \log n _ { j } , \qquad L _ { g } = \log n _ { g } .\tag{B.3}
$$

(where the KL term for $g$ itself is zero, leaving only log $n _ { g }$ in its index.) Following IMED’s preference for smaller indices, $L _ { j } \leq L _ { g }$ means that sampling $j$ again has at least as high a priority as sampling g again. Rewriting this comparison in terms of the actual relative trial proportion $\tau _ { j } = n _ { j } / n _ { g }$ gives

$$
\begin{array} { r l } & { L _ { j } \leq L _ { g } \iff n _ { j } d ( \bar { r } _ { j } , \bar { r } _ { g } ) \leq \log \frac { n _ { g } } { n _ { j } } } \\ & { \iff \tau _ { j } \leq \exp \{ - n _ { j } d ( \bar { r } _ { j } , \bar { r } _ { g } ) \} . } \end{array}\tag{B.4}
$$

The right-hand side gives a threshold for assessing how extensively arm $j$ has been sampled relative to $g .$ We call this threshold the target relative exploration ratio and define it as

$$
\tilde { \tau } _ { j } : = \mathrm { e x p } \{ - n _ { j } d ( \bar { r } _ { j } , \bar { r } _ { g } ) \} .\tag{B.5}
$$

Because $D _ { j } = d ( \bar { r } _ { j } , \bar { r } _ { g } )$ , Line 4 computes exactly this quantity. If $\tau _ { j } < \tilde { \tau } _ { j }$ , we treat $j$ as a candidate for further exploration; if $\tau _ { j } > \tilde { \tau } _ { j }$ , we favor exploiting $g$ in this pairwise comparison. Here, “target” refers to the designed switching threshold for the pairwise priority; it does not mean that the actual trial ratio converges to this value. A unique empirical best arm in an under-achieved condition gives $0 \leq \bar { r } _ { j } < \bar { r } _ { g } < 1$ . Then $d ( \bar { r } _ { j } , \bar { r } _ { g } )$ is positive because $\bar { r } _ { j } \neq \bar { r } _ { g }$ and finite because $0 < \bar { r } _ { g } < 1$ . Since $n _ { i } > 0$ , we obtain $0 < \tilde { \tau } _ { j } < 1$

Constructing the KL term $d ( \bar { r } _ { j } , \bar { r } _ { g } )$ from the empirical distribution and the largest sample mean provides a plug-in estimate of the corresponding information quantity defined by the true reward distributions and the optimal mean. The target relative exploration ratio $\tilde { \tau } _ { j } = \exp \{ - n _ { j } d ( \bar { r } _ { j } , \bar { r } _ { g } ) \}$ takes values in (0, 1), but it is not a probability. It is not the posterior probability that arm $j$ is actually optimal. Nor is it the probability that further trials reverse the order of the sample means so that $\bar { r } _ { j }$ overtakes $\bar { r } _ { g }$ . By the Chernof bound, $\tilde { \tau } _ { j }$ is only an upper bound on the probability that $n _ { j }$ trials from Ber $\left( \hat { r } _ { g } \right)$ yield a sample mean of at most $\bar { r } _ { j }$ . The likelihood relation is an identity for the observed data, whereas recovery from estimation errors depends on the policy. IMED’s performance guarantees apply to its index-minimizing selection rule and do not automatically extend to our RS policy.

## Appendix B.4. Translating the Target Ratio into a Pairwise Satisfaction Level

Line 5 converts $\tilde { \tau } _ { j }$ into a pairwise candidate satisfaction level $S _ { j }$ . We translate the switching threshold defined by the target relative exploration ratio into the satisfaction-level scale used by RS. Specifically, we seek a pairwise satisfaction level that favors further exploration of $j$ when $\tau _ { j } < \tilde { \tau } _ { j }$ and exploitation of $g$ when $\tau _ { j } > \tilde { \tau } _ { j }$ . Since RS favors the arm with the larger value, we require the two RS values to tie at the switching threshold $\tau _ { j } = \tilde { \tau } _ { j }$ . We therefore derive the relation between the satisfaction level and the relative trial proportion from the tie condition $I _ { g } ^ { \prime } = I _ { j } ^ { \prime }$ . Substituting the RS-value definition and dividing by $\rho _ { g } > 0$ gives (note that $\tau _ { j } = n _ { j } / n _ { g } = \rho _ { j } / \rho _ { g } )$

$$
\begin{array} { r } { \rho _ { g } ( \bar { r } _ { g } - S ^ { \prime } ) = \rho _ { j } ( \bar { r } _ { j } - S ^ { \prime } ) , } \\ { \bar { r } _ { g } - S ^ { \prime } = \tau _ { j } ( \bar { r } _ { j } - S ^ { \prime } ) . } \end{array}\tag{B.6}
$$

For $\tau _ { j } \neq 1$ , solving for the new satisfaction level $S ^ { \prime }$ yields

$$
\begin{array} { c } { { S ^ { \prime } = \displaystyle \frac { \bar { r } _ { g } - \tau _ { j } \bar { r } _ { j } } { 1 - \tau _ { j } } } } \\ { { = \bar { r } _ { g } + \displaystyle \frac { \tau _ { j } } { 1 - \tau _ { j } } ( \bar { r } _ { g } - \bar { r } _ { j } ) . } } \end{array}\tag{B.7}
$$

For $\bar { r } _ { g } > \bar { r } _ { j }$ , this tie-inducing satisfaction level lies in the under-achieved range only when $0 < \tau _ { j } < 1$ . This relation maps each relative trial proportion to the satisfaction level at which the two RS values tie. We therefore substitute the target relative exploration ratio $\tilde { \tau } _ { j } \in ( 0 , 1 )$ for the current ratio $\tau _ { j }$ to define the pairwise candidate satisfaction level:

$$
S _ { j } = \bar { r } _ { g } + \frac { \tilde { \tau } _ { j } } { 1 - \tilde { \tau } _ { j } } ( \bar { r } _ { g } - \bar { r } _ { j } ) .\tag{B.8}
$$

Line 5 computes this candidate. A tie at the threshold alone does not determine the priority on either side, so we next examine the sign of the RS-value diference. Using $\rho _ { j } = \tau _ { j } \rho _ { g }$ , we can write the RS-value diference at $S ^ { \prime } = S _ { j }$ as

$$
I _ { g } ^ { \prime } - I _ { j } ^ { \prime } = \rho _ { g } \{ ( \bar { r } _ { g } - S _ { j } ) - \tau _ { j } ( \bar { r } _ { j } - S _ { j } ) \} .\tag{B.9}
$$

From Eq. (B.8), the diferences between each sample mean and $S _ { j }$ are

$$
\begin{array} { l } { { \displaystyle { \bar { r } } _ { g } - S _ { j } = - \frac { \tilde { \tau } _ { j } } { 1 - \tilde { \tau } _ { j } } ( \bar { r } _ { g } - \bar { r } _ { j } ) } , } \\ { { \displaystyle { \bar { r } } _ { j } - S _ { j } = - \frac { 1 } { 1 - \tilde { \tau } _ { j } } ( \bar { r } _ { g } - \bar { r } _ { j } ) } . } \end{array}\tag{B.10}
$$

Substituting these into this diference yields

$$
\begin{array} { r l r } & { } & { I _ { g } ^ { \prime } - I _ { j } ^ { \prime } = \frac { \rho _ { g } \left( \bar { r } _ { g } - \bar { r } _ { j } \right) } { 1 - \tilde { \tau } _ { j } } \{ - \tilde { \tau } _ { j } + \tau _ { j } \} } \\ & { } & { ~ = \frac { \rho _ { g } \left( \bar { r } _ { g } - \bar { r } _ { j } \right) } { 1 - \tilde { \tau } _ { j } } ( \tau _ { j } - \tilde { \tau } _ { j } ) . } \end{array}\tag{B.11}
$$

Because the coeficient is positive, this pairwise comparison favors $j$ simply when $\tau _ { j } < \tilde { \tau } _ { j }$ and $g$ when $\tau _ { j } > \tilde { \tau } _ { j } ;$ the two arms tie only when the ratios are equal. Thus, setting $S ^ { \prime } = S _ { j }$ aligns the pairwise RS priority with the comparison between the actual and target trial ratios.

Appendix B.5. Combining Pairwise Levels into a Common Satisfaction Level

Line 7 combines the pairwise candidates, and Line $8$ caps the result at the initial satisfaction level. To use one satisfaction level for all arms, we take the largest candidate and cap it at the initial satisfaction level:

$$
\begin{array} { l } { S _ { \mathrm { U L } } : = \underset { j \in J } { \operatorname* { m a x } } S _ { j } , } \\ { S ^ { \prime } : = \operatorname* { m i n } \{ S _ { 0 } , S _ { \mathrm { U L } } \} . } \end{array}\tag{B.12}
$$

Each candidate exceeds $\bar { r } _ { g } ,$ , so this preserves $\bar { r } _ { g } < S ^ { \prime } \le S _ { 0 }$ . When $\tau _ { j } < \tilde { \tau } _ { j }$ we have $\rho _ { g } > \rho _ { j }$ . In this case, the diference $I _ { g } ^ { \prime } \mathrm { ~ - ~ } I _ { j } ^ { \prime }$ decreases as the satisfaction level increases. Consequently, replacing $S _ { j }$ with $S _ { \mathrm { U L } }$ preserves the priority of candidate $j$ for further exploration over $g .$ . Taking the maximum is conservative in this sense, but it does not guarantee that $j$ will be selected because another arm may have a higher RS value. This choice also does not preserve every pairwise comparison that favors exploitation or make all pairs tie simultaneously. The final $S ^ { \prime }$ preserves the priority of exploration candidates when $S _ { 0 } \geq S _ { \mathrm { U L } }$ . When $S _ { 0 } < S _ { \mathrm { U L } }$ , we instead enforce the constraint that the initial satisfaction level must not be increased. Arms are selected by maximizing RS values at this adjusted satisfaction level, not by minimizing IMED indices; therefore, we make no claim of asymptotic optimality. Line 9 returns $S ^ { \prime }$ , and RS compares arms at the current step using this adjusted satisfaction level.

Note: Arms 1 and 2 are alternately chosen for 100 steps (i.e., 50 steps each). The reward probabilities for Arms 1 and 2 are 0.2 and 0.8, respectively.

![](images/d1175329fbac70ff7a77ed1ff199988f4da370565f7713e6aec9f3bdbfe7e54b.jpg)

![](images/cd4d4aff0f07f7ae491499b713037d4a58966fc1bdcd960a53db23b0ccc507d0.jpg)

(a) Memory persistence  
![](images/f1f2dbebf0bf878e54619d8e431314ebb077032355625f0e4849a04fe13e3762.jpg)

![](images/aeeaa21be4f31d25f0be7c04f49a80416c6144881d43eeb1ed9215a06c7dd53d.jpg)

(b) Belief relaxation  
![](images/e8c2fa5990d1d2cfa7ef25d2a8ccbf04ecb35b1190bdc9381f857d691cc31eeb.jpg)  
(c) Initial belief variability  
Figure 3: Comparison of internal dynamics among Bayesian-like methods.

![](images/f006ce34d58c74c93470a4f365382ffeab84dc0ca8987811450ffddb82e08f88.jpg)  
(a) Regret with $S = S _ { \mathrm { o p t } }$

![](images/065b71b437abe413d9af5e28188589bcd32e3b623e732634a1e42165d2c8c1d0.jpg)  
(b) Regret with $S = 1 . 0$  
Figure 4: Performance comparison among satisficing RL methods. The plot of TS is shown as a baseline. In both settings, dynamically adjusted RS achieves lower regret than $\mathrm { S A T \mathrm { - } U C B ^ { + } }$ . The legend RS $S = x$ denotes standard RS with the satisfaction level fixed at x, whereas RS $S _ { 0 } = x$ denotes RS whose satisfaction level is dynamically adjusted from the initial value x. (a) Regret over time with $S = S _ { \mathrm { o p t } }$ or $S _ { 0 } = S _ { \mathrm { o p t } }$ . (b) Regret over time with $S = 1 . 0$ or $S _ { 0 } = 1 . 0$