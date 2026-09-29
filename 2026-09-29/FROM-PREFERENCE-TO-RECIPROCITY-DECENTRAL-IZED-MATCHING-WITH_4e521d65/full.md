# FROM PREFERENCE TO RECIPROCITY: DECENTRAL-IZED MATCHING WITH EMPIRICALLY GROUNDED LLM-AGENT BASED MODELING

Wangxuan Fan<sup>∗</sup> Xiaoyu Nie<sup>∗</sup> Zhoutian Shi Xiangcheng Meng Shipei Zeng Pin Gao Yan Hu<sup>†</sup> Zhongxiang Dai<sup>†</sup>

The Chinese University of Hong Kong, Shenzhen

{wangxuanfan,xiaoyunie,zhoutianshi,xiangchengmeng}@link.cuhk.edu.cn {gaopin,huyan,daizhongxiang}@cuhk.edu.cn shipei.zeng@sribd.cn

## ABSTRACT

Bipartite matching is a fundamental problem in game theory and market design. Classical approaches such as Gale–Shapley assume complete preferences and centralized computation, whereas many real-world matching processes are decentralized, asynchronous, and shaped by sequential interaction under limited information. We propose a dynamic bipartite matching framework that combines large language model (LLM) agents with contextual bandits. In a simulated Chinese marriage market, economically grounded LLM agents evaluate locally encountered candidates, while agent-specific Logistic-UCB models learn reciprocal acceptance from realized proposal outcomes. The mechanism therefore separates two decisions—whom do I like? and who is likely to like me back?—without requiring ex ante market-wide preference rankings. We first validate LLM-induced mate preferences against the empirical conditional-logit reference across multiple LLM backbones. In the 50 × 50 matching experiment, Bandit-UCB achieves the highest mean mutual welfare (56.01 versus 54.87 for Gale–Shapley), a smaller gender rank gap than the classical baselines, and the fewest blocking pairs among the LLM-ABM policies. Learned acceptance models show economically interpretable gender-differentiated associations, while counterfactual setups reveal no systematic unilateral advantage from prior search knowledge. Overall, these results support the advantages of decentralized matching with LLM-based behavioral modeling and online learning under incomplete information for economic simulation and computational social science research.

## 1 INTRODUCTION

Two-sided matching is a fundamental problem in game theory and market design, with applications ranging from labor markets and school choice to marriage markets (Gale & Shapley, 1962; Kamecke, 1992; Roth & Sotomayor, 1992). The classical Gale–Shapley deferred-acceptance algorithm (Gale & Shapley, 1962) guarantees a stable matching by taking agents’ ordinal preference rankings as given. While this formulation has been highly successful as a normative allocation mechanism, it abstracts away from several features of naturally occurring matching processes (Eyupoglu et al., 2021; Axtell & Kimbrough, 2008; Zhang & Fang, 2024): agents typically observe only a limited set of alternatives, interactions occur sequentially and asynchronously, and preferences or feasible opportunities may be learned only through experience. Moreover, stability does not necessarily coincide with aggregate welfare. Both analytical and computational studies have shown that restricting attention to stable outcomes can impose a non-negligible welfare cost (Axtell & Kimbrough, 2008; Boudreau & Knoblauch, 2013; Chen et al., 2021; Ortega et al., 2024). These observations motivate a complementary view of matching: rather than asking only how to compute an allocation given a complete preference profile, we study how preference information, beliefs, and matches can jointly evolve through decentralized interaction.

Agent-based modeling (ABM) provides a natural bottom-up framework for such a perspective. By specifying heterogeneous agents, local information, and interaction rules, ABMs generate aggregate outcomes from repeated individual decisions (Epstein, 2012; Axtell & Farmer, 2025). This is particularly relevant to marriage markets, where individual partner choices depend on the current pool of available alternatives, while every new relationship changes that pool for subsequent decisions (Kalmijn, 1998; Chiappori & Weiss, 2006; Van Bavel & Grow, 2016; Grow, 2019). Axtell & Kimbrough (2008), for example, replace centralized deferred acceptance with decentralized search, proposal, and rematching, achieving higher average welfare without preserving guaranteed stability, which is not necessarily a strict desideratum in marriage markets. Yet decentralizing the match ing process does not by itself solve the problem of preference formation: candidate rankings still need to be specified before the matching dynamics begin. This is a central difficulty in agent-based marriage-market modeling. As emphasized by Grow (2019), theories of mate choice are often competing, multidimensional, and verbal rather than directly mathematical, forcing modelers to translate behavioral assumptions into hand-designed utility functions and parameterizations.

Large language models (LLMs) provide a novel way to operationalize semantically rich behavioral assumptions. Existing studies show that LLM agent-based modeling (LLM-ABM) can condition decisions on heterogeneous personas and contextual information and can be embedded in agents that interact over extended trajectories (Argyle et al., 2023; Aher et al., 2023; Park et al., 2023). However, we do not treat an LLM as an unconstrained oracle of human preferences: expressive behavior is not necessarily economically valid behavior, motivating careful specification and validation when LLMs are used as simulated economic actors (Horton et al., 2023; Ludwig et al., 2024). Nor do we provide the agents with market-wide preference rankings in advance, or let them infer the results in one-shot. Instead, LLM valuation is triggered only when agents encounter one another. Yet knowing how much an agent values a candidate does not reveal whether that candidate will reciprocate. Because the proposer observes acceptance only after making a proposal, reciprocity becomes a sequential learning problem under partial feedback. This naturally motivates an agent-specific contextual bandit that learns, from realized interactions, which desirable candidates are likely to accept. Unlike matching-bandit approaches that learn latent rewards or unknown preference rankings (Liu et al., 2021; Dai & Jordan, 2021; Cen & Shah, 2022), our framework separates behavioral evaluation from interaction learning: the LLM determines how much an agent values an encountered candidate, while the bandit learns how likely that candidate is to accept.

Based on this separation, we develop a decentralized and asynchronous bipartite matching framework that combines behaviorally grounded LLM agents with agent-specific contextual bandits. In the marriage-market setting, agents encounter only locally available alternatives, evaluate them through LLM-based preferences, and learn reciprocal acceptance from realized outcomes, without requiring complete rankings in advance. Experiments on a simulated Chinese marriage market validate the behavioral grounding of the agents and show favorable welfare–stability outcomes rel ative to classical and decentralized baselines, while learned acceptance models and counterfactual experiments reveal structured reciprocal-choice patterns and the limits of unilateral information advantages.

Our contributions are threefold:

• Empirically grounded LLM-agent based modeling. We develop a general approach for constructing and validating LLM agents from domain-specific empirical evidence, and instantiate it in a marriage market using socioeconomic characteristics and gender-specific mate-preference evidence. We validate the induced preferences against an empirical reference across multiple LLM backbones and market scales.

• Decoupled decentralized matching mechanism. We formulate matching as an asynchronous local-interaction process that separates LLM-based partner valuation from agentspecific reciprocity learning, without requiring agents to access complete preference rankings. For a theoretically calibrated variant, we establish local proposal-regret bounds under realizable and dynamically misspecified acceptance models.

• Economic evaluation and interpretable results. We show that the proposed framework preserves the core two-sided matching problem while relaxing strong information assumptions, and achieves better overall welfare–stability outcomes. The learned reciprocalacceptance models yield economically interpretable patterns, while warm-start counterfactuals reveal how prior market knowledge translates into individual and market-level matching outcomes.

Our code is publicly available in our GitHub repository. Related work is reviewed in Appendix A.

## 2 EMPIRICALLY GROUNDED LLM-AGENT MODELING

## 2.1 SYNTHETIC POPULATION WITH EMPIRICAL HETEROGENEITY

Our modeling principle is to construct LLM agents from domain-specific empirical evidence rather than unconstrained persona generation. This requires identifying behaviorally relevant observable characteristics and preserving important population heterogeneity and dependence structures. The resulting structured profiles provide both the population representation and the empirical basis for downstream LLM-based behavioral evaluation.

We instantiate this approach in a synthetic Chinese marriage market. Each agent is endowed with attributes emphasized in empirical studies of mate choice (Zhou et al., 2023), including age, income, education, family background, housing status, and physical appearance. These dimensions capture the gender heterogeneity and social-class differences documented in the empirical reference.

Following the dependency-aware construction of Li et al. (2026), we sample demographic variables to approximate observed population heterogeneity while preserving key relationships, such as income–education dependence, rather than treating attributes independently. Demographic and economic variables are based on the 2015 National 1% Population Sample Survey of China (Population Census Office under the State Council & Department of Population and Employment Statistics, National Bureau of Statistics, 2016), consistent with the period studied by Zhou et al. (2023), while physical appearance is sampled from an approximately normal distribution based on empirical evidence (Hamermesh & Biddle, 1993). Full sampling details and realized population composition are reported in Appendix B.

## 2.2 LLM PERSONAS AND MATE PREFERENCE

Each structured profile is rendered as a concise public persona containing only observable attributes available to potential partners, ensuring that LLM evaluations use the same information encoded in the demographic profile. The rendering template is provided in Appendix C.

To model mate preferences, we construct gender-specific prompts grounded in the regression evidence of Zhou et al. (2023) and use chain-of-thought prompting (Wei et al., 2022) to structure evaluation. The full private preference prompts are given in Appendix C.2.

## 2.3 VALIDATION OF MODELING

As discussed in Appendix A, validating persona-conditioned decisions is important for establishing the economic interpretability of LLM-based social simulations. We therefore compare the matepreference rankings induced by our LLM agents with an empirical reference model derived from Zhou et al. (2023). The reference model maps each candidate profile to a conditional-logit utility under the corresponding gender-specific specification, while the LLM assigns an overall desirability score from the evaluator’s public persona and private mate-preference prompt. We evaluate ranking agreement using Kendall’s τ , weighted pairwise violation rate (WPVR), and top-K overlap, with definitions provided in Appendix D.

Figure 1 reports the validation results for a $1 0 \times 1 0$ market across five random seeds. Across all three LLM backbones, preference-structured prompts achieve stronger alignment with the empirical reference than the no-CoT baseline, indicating that the result is not specific to a single backbone. A 50 × 50 scale-up validation is reported in Appendix F.

![](images/92fb6adacc1c9003426c2221fe61f4499165bff71cf80f4ba737d43dc1bdd2bc.jpg)  
Figure 1: Empirical-reference alignment in the 10×10 validation task. Bars show means across five seeds and error bars show standard deviations.

![](images/2f85e6334882609eacefea42474fe239d229c510ef2d42ac51f1bcd3b95989df.jpg)

![](images/b3d800514a717cf0f6b63908652dbc99f5d1cc12c3c759f94a5ae224c94cdc46.jpg)

![](images/4a808e5e07998ac4c79a636af662363dcf0172aa5f85e8a7cbf2a0587db024dc.jpg)  
Figure 2: Dimension-consistency heatmap for agents in the 10×10 validation. Entries report Pearson correlations between overall desirability and dimension-level scores, averaged across five seeds.

Figure 2 provides a complementary internal-consistency diagnostic. Appearance is most strongly associated with overall desirability for male agents, whereas female evaluations are more closely associated with socioeconomic dimensions, particularly housing, income, and education. These patterns are consistent with the gender-specific preference structure documented by Zhou et al. (2023), while the no-CoT baseline fails to reproduce the same structure. Results for the $5 0 \times 5 0$ setting are reported in Appendix F.

## 3 DECENTRALIZED MATCHING SYSTEM

## 3.1 A DECENTRALIZED AND ASYNCHRONOUS MATCHING MARKET

We consider a two-sided marriage market with male agents $\mathcal { M } = \{ m _ { 1 } , . . . , m _ { { N _ { M } } } \}$ , female agents $\mathcal { W } = \{ w _ { 1 } , . . . , w _ { N _ { W } } \}$ , and population $\mathcal { A } = \mathcal { M } \cup \mathcal { W } .$ At period $t , \mu _ { t } ( i )$ denotes agent i’s current partner, with $\mu _ { t } ( i ) = \mathcal { O } \mathrm { i f } i$ is single and $\mu _ { t } ( i ) = j \Leftrightarrow \mu _ { t } ( j ) = i .$ A match is either temporary or permanent. Temporary relationships remain contestable and can be replaced by later accepted proposals, whereas permanently matched agents exit the active market.

The market evolves through decentralized and asynchronous activation. At the beginning of period t, let $\mathcal { A } _ { t } ^ { \mathrm { a c t } } \subseteq \mathcal { A }$ denote agents that have not entered permanent matches. A period consists of $\lceil A _ { t } ^ { \mathrm { a c t } } \rceil$ sequential activation opportunities, each sampling one agent uniformly with replacement from this set. Thus, an agent may be activated multiple times within a period or not at all.

Activation does not give an agent access to the entire opposite side of the market. When agent i is activated, it observes only a locally available candidate set $\mathcal { C } _ { t } ( i ) \subseteq \mathcal { P } _ { t } ( i )$ , where $\mathcal { P } _ { t } ( i )$ denotes the set of currently feasible agents on the opposite side who have not entered permanent matches and the candidate set $\mathcal { C } _ { t } ( i )$ is randomly sampled from this feasible pool. Decisions therefore depend on both preferences and locally encountered opportunities. This differs from Gale–Shapley (Gale & Shapley, 1962), which uses complete rankings in a centralized procedure, and from Axtell & Kimbrough (2008), which decentralizes interaction but retains pre-specified preference lists.

## 3.2 WHOM DO I LIKE? LOCAL LLM VALUATION

The first decision of an activated agent is a valuation problem: whom do $I \ l i k e ?$ For evaluator i and locally observed candidate $j ,$ the simulated agent in Section 2 produces a normalized subjective utility represented by an overall preference score:

$$
U _ { i } ( j ) \in [ 0 , 1 ] , \qquad U _ { i } ( j ) \not = U _ { j } ( i ) \mathrm { i n g e n e r a l . }\tag{1}
$$

The asymmetry is essential: i’s valuation of $j$ does not reveal whether j would reciprocate.

Let $r _ { i }$ denote agent $i \gamma _ { \mathrm { s } }$ reservation utility, the minimum acceptable partner utility. Its current benchmark is

$$
b _ { i , t } = \left\{ \begin{array} { l l } { r _ { i } , } & { \mu _ { t } ( i ) = \emptyset , } \\ { \operatorname* { m a x } \{ r _ { i } , U _ { i } ( \mu _ { t } ( i ) ) \} , } & { \mu _ { t } ( i ) \not = \emptyset . } \end{array} \right.\tag{2}
$$

For each $j \in \mathcal { C } _ { t } ( i )$ , define the utility improvement and eligible set by

$$
\begin{array} { r } { \Delta U _ { i j , t } = U _ { i } ( j ) - b _ { i , t } , \qquad { \mathcal E } _ { t } ( i ) = \left\{ j \in { \mathcal C } _ { t } ( i ) : \Delta U _ { i j , t } > \epsilon _ { U } , \epsilon _ { U } \geq 0 \right\} . } \end{array}\tag{3}
$$

This eligibility condition separates preference from strategic search: only candidates preferred to the current benchmark are considered, so reciprocal-acceptance learning cannot make an otherwise undesirable candidate attractive. The LLM determines whom the agent wants; the remaining uncertainty is which of these desirable candidates are likely to reciprocate.

## 3.3 WHO IS LIKELY TO LIKE ME BACK? LOCAL RECIPROCAL-ACCEPTANCE LEARNING

The second decision concerns reciprocity: who is likely to like me back? If i proposes to $j ,$ the receiver independently evaluates i against its current benchmark,

$$
b _ { j , t } ^ { R } = \left\{ { r _ { j } , \qquad \quad \mu _ { t } ( j ) = \emptyset , } \right.\tag{4}
$$

The realized response is

$$
Y _ { i  j , t } = \mathbf { 1 } \{ U _ { j } ( i ) > b _ { j , t } ^ { R } \} \in \{ 0 , 1 \} .\tag{5}
$$

Crucially, proposer i does not observe $U _ { j } ( i )$ , the receiver’s incumbent valuation, or its eventual decision before proposing. Acceptance is revealed only after a proposal is made, so feedback is observed only for candidates actually approached. Because proposal opportunities are limited, the proposer must balance exploiting candidates believed likely to reciprocate with exploring uncertain candidates whose responses are informative. This sequential decision problem with contextual, partial feedback naturally motivates an agent-specific contextual bandit.

For each exposed candidate $j ,$ proposer i observes a dyadic context $x _ { j , t } ^ { ( i ) } = \phi ( z _ { i } , z _ { j } ) \in \mathbb { R } ^ { d }$ constructed from the public personas $z _ { i }$ and $z _ { j }$ , and learns only from proposals it actually makes. Its local history is $\mathcal { H } _ { i , t } = \{ ( x _ { n } ^ { ( i ) } , y _ { n } ^ { ( i ) } ) \} _ { n = 1 } ^ { N _ { i , t } }$ . Unchosen candidates generate no label, and histories are not shared across agents. The context construction and information restrictions are detailed in Appendix G.

We approximate reciprocal acceptance using an agent-specific logistic working model,

$$
p _ { \theta _ { i } } ( x ) : = \sigma ( \theta _ { i } ^ { \top } x ) , \qquad \sigma ( z ) = \frac { 1 } { 1 + e ^ { - z } } .\tag{6}
$$

This is an approximation based on proposer-observable information: the receiver’s private benchmark changes with its current relationship, so acceptance probabilities need not follow a fixed logistic law even for the same public dyadic context. To balance exploitation and exploration, we augment the fitted logit predictor with local uncertainty:

$$
\begin{array} { r } { \begin{array} { c } { s _ { j , t } ^ { ( i ) } = \sqrt { { x _ { j , t } ^ { ( i ) } } ^ { \top } V _ { i , t } ^ { - 1 } x _ { j , t } ^ { ( i ) } } , } \\ { p _ { i j , t } ^ { \mathrm { U C B } , i } = \sigma \left( \widehat { \theta _ { i , t } ^ { \top } } x _ { j , t } ^ { ( i ) } + \beta _ { i , t } s _ { j , t } ^ { ( i ) } \right) . } \end{array} } \end{array}\tag{7}
$$

Here, $\widehat { \theta } _ { i , t }$ is fitted by local L -regularized logistic regression, $V _ { i , t }$ is the corresponding regularized Fisher-information matrix, and $\beta _ { i , t }$ controls exploration. The estimator and practical Logistic-UCB derivation are provided in Appendix K.

Preference and reciprocal uncertainty are finally combined through

$$
\begin{array} { r } { \boxed { I _ { i j , t } = \Delta U _ { i j , t } p _ { i j , t } ^ { \mathrm { U C B } , i } } , \qquad j \in \mathcal { E } _ { t } ( i ) . } \end{array}\tag{8}
$$

If the true acceptance probabilities conditional on the proposer’s current information were known, maximizing their product with $\Delta U _ { i j , t }$ would maximize expected one-step utility improvement (Proposition L.1). Equation 8 replaces these probabilities with optimistic estimates from the working model, allocating proposals among already desirable candidates.

Theorem (Informal) 1 (Local learning under dynamic misspecification). Under the boundedness and predictability assumptions in Appendix L, fix a proposer whose true conditional acceptance probabilities differ from a fixed logistic reference by at most a predictable envelope $\eta _ { n } f o r$ every eligible candidate at attempt n. Logistic-UCB with the norm-constrained estimator and envelopecalibrated confidence radii specified there satisfies, with probability at least $1 - \delta ,$

$$
\mathcal { R } _ { N } \leq \widetilde { \mathcal { O } } \left( d \sqrt { N } + \sqrt { d N \sum _ { n = 1 } ^ { N } \eta _ { n } ^ { 2 } } + \sum _ { n = 1 } ^ { N } \eta _ { n } \right) .\tag{9}
$$

Here N counts this proposer’s attempts, d is the context dimension, and $\mathcal { R } _ { N }$ measures cumulative expected one-step improvement lost relative to an oracle facing the same eligible candidates and knowing their true acceptance probabilities. The notation suppresses logarithmicfactors, with other problem constants fixed. Realizable acceptance $( \eta _ { n } \equiv 0 )$ recovers $\widetilde { \mathcal { O } } ( d \sqrt { N } )$ regret. For fixed d, $\begin{array} { r } { \mathcal { R } _ { N } / N \to 0 i f 0 \leq \eta _ { n } \leq 1 a n d \sum _ { n = 1 } ^ { N } \eta _ { n } = o ( N / \log N ) } \end{array}$

In Appendix, Theorems L.1 and L.2 establish these local bounds along the learner’s realized market trajectory formally.

## 3.4 DECENTRALIZED MATCHING PROTOCOL

When activated, agent $i ,$ either male or female, constructs the eligible set $\mathcal { E } _ { t } ( i )$ and may make at most L proposals. Let $\mathcal { R } _ { t } ^ { ( q ) } ( i )$ denote the eligible candidates not yet approached before attempt q, with $\mathscr { R } _ { t } ^ { ( 1 ) } ( i ) = \mathscr { E } _ { t } ( i )$ . At each attempt, the proposer selects the candidate with the highest proposal index and observes the binary response:

$$
j ^ { * } = \arg \operatorname* { m a x } _ { j \in \mathcal { R } _ { t } ^ { ( q ) } ( i ) } I _ { i j , t } , \qquad y ^ { * } = Y _ { i  j ^ { * } , t } .\tag{10}
$$

The observation $( x _ { j ^ { * } , t } ^ { ( i ) } , y ^ { * } )$ is appended to proposer i’s local history, and its logistic model and uncertainty estimate are updated. Following rejection, $j ^ { * }$ is removed from the current search set and proposal indices are recomputed before the next attempt. Following acceptance, any temporary relationships involving i or $j ^ { * }$ are dissolved, their displaced partners return to the active market, and $( i , j ^ { * } )$ becomes a new temporary pair. The current activation then terminates.

At the end of period t, each temporary pair independently becomes permanent with probability $\rho _ { t }$ , which is named as lock or commitment probability. Permanently matched agents leave subsequent search, while the remaining temporary pairs stay contestable. The active population and feasible candidate pools are updated before the next period. The process continues until the maximum horizon $T$ is reached or no active agents remain. Hence, the final allocation emerges from repeated local exposure, bilateral LLM evaluation, proposal feedback, agent-specific learning, and endogenous relationship revision rather than from a one-shot centralized procedure. The eligibility and acceptance rules imply strict bilateral improvement for every accepted rematching, although matched-pair welfare need not be monotone because displaced partners create externalities; these properties are formalized in Appendix L. The full executable procedure of the algorithm is provided in Appendix M.

Table 1: $5 0 \times 5 0$ market size matching results. Values are means ± standard deviations over 50 seeds; bold, underlined, and italic mean values denote first, second, and third place among methods for which each outcome metric is defined. The “Info required” column summarizes the information regime, including both ex ante preference access and whether pairwise valuations are generated only after local encounter.
<table><tr><td></td><td>Method</td><td>Info required</td><td>Mutual welfare ↑</td><td>Rank gap ↓</td><td>#Blocking ↓</td><td>Proposals ↓</td></tr><tr><td rowspan="3">Baselines</td><td>Gale-Shapley</td><td>Whole</td><td> $5 4 . 8 7 \pm 0 . 2 9$ </td><td> $2 . 7 4 \pm 0 . 5 3$ </td><td> ${ \bf 0 . 0 0 \pm 0 . 0 0 }$ </td><td>873.14 ± 20.24</td></tr><tr><td>LLM-solver</td><td>Whole</td><td> $5 4 . 6 3 \pm 0 . 5 0$ </td><td> $2 . 5 7 \pm 1 . 1 8$ </td><td> $9 . 4 0 \pm 1 0 . 7 1$ </td><td>N/A</td></tr><tr><td>Axtell-Kimbrough</td><td>Partial</td><td> $5 4 . 7 3 \pm 0 . 3 9$ </td><td> $2 . 4 3 \pm 0 . 8 3$ </td><td> $\overline { { 1 9 . 9 0 } } \pm 1 4 . 0 2$ </td><td> $3 1 5 0 4 . 7 0 \pm 4 2 8 0 . 3 0$ </td></tr><tr><td rowspan="3">LLM-ABM</td><td>Random eligible</td><td>Local</td><td> $5 5 . 2 4 \pm 0 . 6 2$ </td><td> ${ \bf 1 . 5 4 \pm 0 . 6 8 }$ </td><td> $3 8 . 0 0 \pm 1 7 . 4 2$ </td><td> $1 3 7 8 1 . 9 0 \pm 2 3 4 9 . 2 9$ </td></tr><tr><td>Utility-only</td><td>Local</td><td> $5 5 . 9 2 \pm 0 . 4 8$ </td><td> $1 . 8 4 \pm 0 . 7 5$ </td><td> $1 7 . 5 8 \pm 1 2 . 2 6$ </td><td> $\overline { { 1 6 7 5 3 . 1 4 } } \pm 2 8 7 6 . 9 1$ </td></tr><tr><td>Bandit-UCB</td><td>Local</td><td> $\overline { { { \bf 5 6 . 0 1 } } } \pm 0 . 5 9$ </td><td> $\overline { { 2 . I I } } \pm 0 . 7 3$ </td><td> $I 6 . 2 8 \pm 1 2 . 7 6$ </td><td> $I 6 5 9 I . 7 8 \pm 2 3 7 6 . 7 0$ </td></tr></table>

## 4 EXPERIMENTS

## 4.1 VALIDATION EXPERIMENT SETUP

Because behavioral validation establishes the basis for the matching mechanism, its results are presented earlier in Section 2.3; here we specify the experimental protocol. In each $1 0 \times 1 0$ validation run, 10 male and 10 female evaluators score all 10 opposite-gender candidates, producing both an overall desirability score and six dimension-level scores for every directional evaluation. We repeat the experiment over five random seeds and compare four scoring conditions with three LLM back bones: DeepSeek-V4-Pro CoT, Qwen3.7-Plus CoT, GPT-OSS-120B CoT, and a no-CoT baseline that omits the gender-specific mate-preference specification. Full metric definitions, the $5 0 \times 5 0$ scale-up validation, and cross-backbone results are reported in Appendices D, F, and E.

## 4.2 MARKET-LEVEL MATCHING RESULTS

We evaluate the complete matching system on the same $5 0 \times 5 0$ synthetic marriage market. Directional LLM desirability scores are computed once for all cross-gender dyads as the evaluation benchmark. These scores provide complete preference rankings for Gale & Shapley (1962), Axtell & Kimbrough (2008), and a direct LLM-solver following Hosseini et al. (2026), which generates a matching from the ranked inputs in one-shot. In contrast, LLM-ABM agents access valuations and update rankings only when the corresponding candidates are locally encountered. The implementation and experimental results demonstrate the effectiveness of solving the bipartite problem using distinct baselines and the proposed method.

We compare these baselines with three LLM-ABM proposal policies under common outcome metrics, reporting means and standard deviations over 50 random seeds. All three policies share the same market dynamics: Random eligible samples uniformly from eligible candidates, Utility-only selects the largest immediate utility improvement, and Bandit-UCB combines this improvement with an optimistic estimate of reciprocal acceptance. We set $\rho = 0 . 0 5$ for both Axtell–Kimbrough and the LLM-ABM policies. Detailed settings and metric definitions are provided in Appendix H.

Table 1 illustrates the trade-offs between information requirements, welfare, stability, and search intensity. Gale–Shapley achieves zero blocking pairs but has the largest mean gender rank gap. Axtell–Kimbrough relaxes centralized coordination while retaining pre-specified rankings, requiring approximately 31,505 proposals. Despite receiving complete rankings, the LLM-solver produces 9.40 blocking pairs on average. It achieves fewer blocking pairs than the decentralized methods, but lower mean mutual welfare and a larger mean rank gap than all three LLM-ABM policies. Its proposal count is reported as N/A because direct matching generation does not involve a comparable sequence of market proposals.

All three LLM-ABM policies achieve higher mean mutual welfare than the baselines using only encounter-based information. Within this family, Random eligible yields the smallest rank gap and fewest proposals, but the most blocking pairs. Utility-only improves welfare and reduces blocking, while Bandit-UCB achieves the highest mean welfare (56.01) and lowest blocking count (16.28). Its gains over Utility-only are modest and accompanied by a larger rank gap (2.11 versus 1.84), reflecting trade-offs among welfare, stability, and gender balance. These results support the potential of decentralized matching without complete ex ante rankings and suggest incremental benefits from reciprocity learning. Sensitivity to exploration and commitment parameters is examined in Appendix J.

## 4.3 INTERPRETING THE LEARNED BANDIT MODELS

Beyond guiding proposals, the bandit models summarize learned associations between observable dyadic characteristics and reciprocal acceptance. Figure 3 presents aggregated private proposerspecific Logistic-UCB coefficients in Panel 3a and a pooled receiver-perspective diagnostic in Panel 3b. In the local models, age-gap coefficients are negative for both proposer genders, whereas income and education coefficients are positive. The appearance coefficient is larger among male proposers, while the education coefficient is larger among female proposers. These differences concern acceptance predictions: male-proposer models predict responses from female receivers, and vice versa. They therefore should not be interpreted as the proposers’ own valuation weights. The negative same-family-background coefficients likewise describe conditional predictive associations, rather than direct preferences against similar backgrounds.

The pooled diagnostic is fitted to realized proposals after simulation and is never available to agents during matching. Under its receiver-to-proposer feature orientation, negative income, education, and appearance coefficients associate favorable proposer characteristics with greater acceptance. Income has a larger coefficient magnitude for female receivers, whereas appearance has a larger magnitude for male receivers. These estimates complement the local models but need not coincide with them because the conditioning and aggregation differ. Both panels provide associational diagnostic evidence: proposal observations are endogenously selected by the matching policy, and acceptance also depends on receivers’ current relationships. The coefficients thus characterize predictive patterns in realized interactions rather than identify structural or causal preference parameters.

## 4.4 COUNTERFACTUAL INFORMATION ADVANTAGE

We finally ask whether prior matching-market experience creates a unilateral advantage in a new market. In the individual warm-start counterfactual, one designated male agent receives Logistic-

![](images/329466ad855de80ea7d7e8a0644006a9e5630f33fe1199689994c24279cfe354.jpg)

(a) Local private models  
![](images/94ebf1a851ba3a65e5734db3593c6dd85b5863cc42a53eeab018810b8d38ddfb.jpg)  
(b) Post-hoc receiver diagnostic

Figure 3: Learned reciprocal-acceptance structure under Bandit-UCB. Panel (a) reports aggregated coefficients from private proposer-specific models; Panel (b) reports a pooled receiver-perspective diagnostic fitted after simulation and used only for interpretation.  
![](images/8b719c633e187f2d89ba52df92ca502b08262f36892b1c4594c0b5eeba025e1d.jpg)

Own rank (11/50 improved)  
![](images/159cbdbadc0c47e207b182975a8a0af0aa6d4ef2e51731fe2075b132c4c0ba9f.jpg)

![](images/ef92ad1c34ae396656678f8595c9c1c9e43002233eca463f6e0b980d40737923.jpg)

![](images/20362d45acbe8f1445cd86f4d000cbe60a4095fdc71a04882d8a8e1d148a2dc7.jpg)  
Figure 4: Individual warm-start effects across the 50 male focal agents. Effects are same-seed treatment–control mean differences over ten held-out seeds and are oriented so that positive values indicate improvement. The upper panel classifies focal agents by the sign of the effect; the lower panels report the corresponding effect distributions.

UCB coefficients learned from prior simulations, while all others retain the standard cold-start specification. Across the 50 male focal agents evaluated over ten held-out seeds in Figure 4, 11 improve their own tied rank and score, 26 remain unchanged, and 13 worsen. Proposal counts fall for 22 agents and rise for 25. Thus, prior search knowledge benefits some agents but worsens outcomes for slightly more, providing no systematic individual advantage. The full focal-agent distribution and paired evaluation protocol appear in Appendix I.

Table 2: Market-level counterfactual results for male-side information advantage. M rank and F rank denote the average ranks of matched male and female agents for their realized partners, respectively.
<table><tr><td>Method</td><td>Initial knowledge</td><td>Mutual welfare ↑</td><td>M rank ↓</td><td>F rank ↓</td><td>Rank gap ↓</td><td>#Blocking ↓</td><td>Proposals ↓</td></tr><tr><td>Gale-Shapley (men-proposing)</td><td>Complete rankings</td><td> $5 4 . 8 7 \pm 0 . 2 9$ </td><td> $1 6 . 7 4 \pm 0 . 3 9$ </td><td> $1 9 . 4 8 \pm 0 . 3 2$ </td><td> $2 . 7 4 \pm 0 . 5 3$ </td><td> $0 . 0 0 \pm 0 . 0 0$ </td><td> $8 7 3 . 1 4 \pm 2 0 . 2 4$ </td></tr><tr><td>Cold-start Bandit-UCB</td><td>Local</td><td>55.84 ± 0.33</td><td>16.48 ± 0.47</td><td> $1 8 . 5 6 \pm 0 . 4 6$ </td><td> $2 . 0 8 \pm 0 . 7 0$ </td><td> $1 6 . 7 0 \pm 8 . 2 1$ </td><td> $1 5 8 2 9 . 5 0 \pm 2 4 6 2 . 3 4$ </td></tr><tr><td>All-male warm-start Bandit-UCB</td><td>Pooled male θ</td><td> $5 5 . 7 1 \pm 0 . 4 4$ </td><td> $1 6 . 5 6 \pm 0 . 7 3$ </td><td> $1 8 . 6 8 \pm 0 . 4 7$ </td><td> $2 . 1 2 \pm 0 . 7 6$ </td><td> $1 5 . 8 0 \pm 4 . 6 6$ </td><td> $1 5 1 3 6 . 7 0 \pm 2 2 1 0 . 7 8$ </td></tr></table>

Table 2 considers a stronger intervention: all male agents enter with pooled male-proposer coefficients. Relative to cold-start Bandit-UCB, mean mutual welfare and gender rank gap remain almost unchanged, while fewer proposals and blocking pairs suggest reduced search burden and slightly improved stability. The limited welfare effect reflects a defining feature of two-sided matching: better proposal beliefs do not remove receivers’ right to reject, and shared information can intensify same-side competition. Individual knowledge may therefore yield local gains, but its limited market-wide benefit when widely shared is consistent with the constraints imposed by reciprocal choice and same-side competition.

## 5 DISCUSSION

In conclusion, we propose a decentralized matching framework that integrates LLM-based behav ioral valuation with adaptive reciprocity learning. This framework facilitates the study of how diverse individual evaluations influence collective matching outcomes through repeated interactions. By decoupling behavioral representation from the interaction mechanism, researchers can independently vary behavioral assumptions, information access, and institutional rules to analyze their interactive effects. This enables computational experiments linking empirically informed agent behavior with economic mechanism design. This modeling approach can be applied to labor markets, collaborative partnerships, and other contexts where local opportunity discovery and mutual agreement are crucial. Limitations and future directions are discussed in Appendix N.

## AI USE STATEMENT

In this work, we used generative AI tools to improve the clarity and fluency of English writing and to assist with code implementation and debugging. All AI-assisted text was reviewed and revised by the authors, and all AI-assisted code was manually inspected, tested, and verified for correctness. We did not use generative AI tools to generate research ideas, scientific claims, experimental results, data, or citations. We take full responsibility for the final content of this work, including all text, claims, code, and artifacts produced with the aid of generative AI.

## REPRODUCIBILITY STATEMENT

The code for our framework is publicly available at https://github.com/YuanJrShiuan/ LLM\_ABM\_Bipartite\_Matching\_for\_Marriage. Experimental settings and implementation details are provided in the appendices.

## REFERENCES

Yasin Abbasi-Yadkori, David P´ al, and Csaba Szepesv´ ari. Improved algorithms for linear stochastic´ bandits. Advances in neural information processing systems, 24, 2011.

Gati V. Aher, Rosa I. Arriaga, and Adam Tauman Kalai. Using large language models to simulate multiple humans and replicate human subject studies. In Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pp. 337–371. PMLR, 2023.

Jacy Reese Anthis, Ryan Liu, Sean M Richardson, Austin C. Kozlowski, Bernard Koch, Erik Brynjolfsson, James Evans, and Michael S. Bernstein. Position: LLM social simulations are a promising research method. In Forty-second International Conference on Machine Learning Position Paper Track, 2025. URL https://openreview.net/forum?id=cRBg1dtj7o.

Lisa P. Argyle, Ethan C. Busby, Nancy Fulda, Joshua R. Gubler, Christopher Rytting, and David Wingate. Out of one, many: Using language models to simulate human samples. Political Analysis, 31(3):337–351, 2023. doi: 10.1017/pan.2023.2.

Robert L Axtell and J Doyne Farmer. Agent-based modeling in economics and finance: Past, present, and future. Journal ofEconomic Literature, 63(1):197–287, 2025.

Robert L. Axtell and Steven O. Kimbrough. The high cost of stability in two-sided matching: How much social welfare should be sacrificed in the pursuit of stability? In Proceedings of the 2008 World Congress on Social Simulation, 2008.

James W Boudreau and Vicki Knoblauch. Preferences and the price of stability in matching markets. Theory and decision, 74(4):565–589, 2013.

David M Buss. Sex differences in human mate preferences: Evolutionary hypotheses tested in 37 cultures. Behavioral and brain sciences, 12(1):1–14, 1989.

David M Buss and David P Schmitt. Sexual strategies theory: an evolutionary perspective on human mating. Psychological review, 100(2):204, 1993.

Sarah H. Cen and Devavrat Shah. Regret, stability & fairness in matching markets with bandit learners. In Proceedings of the 25th International Conference on Artificial Intelligence and Statistics, volume 151 of Proceedings of Machine Learning Research, pp. 8938–8968. PMLR, 2022.

Jiehua Chen, Piotr Skowron, and Manuel Sorge. Matchings under preferences: Strength of stability and tradeoffs. ACM Transactions on Economics and Computation, 9(4):1–55, 2021.

Yiting Chen, Tracy Xiao Liu, You Shan, and Songfa Zhong. The emergence of economic rationality of gpt. Proceedings ofthe National Academy ofSciences, 120(51):e2316205120, 2023.

Pierre-Andre Chiappori and Yoram Weiss. Divorce, remarriage, and welfare: A general equilibrium´ approach. Journal ofthe European Economic Association, 4(2-3):415–426, 2006.

Xiaowu Dai and Michael I. Jordan. Learning strategies in decentralized matching markets under uncertain preferences. Journal ofMachine Learning Research, 22(260):1–50, 2021.

Kingsley Davis. Intermarriage in caste societies. American anthropologist, 43(3):376–395, 1941.

Alice H Eagly, Paul W Eastwick, and Mary Johannesen-Schmidt. Possible selves in marital roles: The impact of the anticipated division of labor on the mate preferences of women and men. Personality and Social Psychology Bulletin, 35(4):403–414, 2009.

Joshua M Epstein. Generative social science: Studies in agent-based computational modeling. Princeton University Press, 2012.

Selin Eyupoglu, Muge Fidan, Yavuz Gulesen, Ilayda Begum Izci, Berkan Teber, Baturay Yilmaz, Ahmet Alkan, and Esra Erdem. Stable marriage problems with ties and incomplete preferences: An empirical comparison of asp, sat, ilp, cp, and local search methods. arXiv preprint arXiv:2108.05165, 2021.

Sarah Filippi, Olivier Cappe, Aur ´ elien Garivier, and Csaba Szepesv ´ ari. Parametric bandits: The ´ generalized linear case. In Advances in Neural Information Processing Systems, volume 23, pp. 586–594, 2010.

David Gale and Lloyd S Shapley. College admissions and the stability of marriage. The American mathematical monthly, 69(1):9–15, 1962.

Andre Grow. How to design agent-based marriage market models: A review of current prac-´ tices. Simulieren und Entscheiden: Entscheidungsmodellierung, Modellierungsentscheidungen, Entscheidungsunterstutzung ¨ , pp. 59–83, 2019.

George Gui and Olivier Toubia. The challenge of using llms to simulate human behavior: A causal inference perspective. arXiv preprint arXiv:2312.15524, 2023.

Daniel S Hamermesh and Jeff Biddle. Beauty and the labor market, 1993.

Anne Lundgaard Hansen, John J Horton, Sophia Kazinnik, Daniela Puzzello, and Ali Zarifhonarvar. Simulating the survey of professional forecasters. Available at SSRN 5066286, 2026.

Yuzhi Hao and Danyang Xie. A multi-llm-agent-based framework for economic and public policy analysis. China Journal ofEconometrics, 5(3):615–630, 2025.

John J Horton, Apostolos Filippas, and Benjamin S Manning. Large language models as simulated economic agents: What can we learn from homo silicus? Technical report, National Bureau of Economic Research, 2023.

Hadi Hosseini, Samarth Khanna, and Ronak Singh. Matching markets meet llms: Algorithmic reasoning with ranked preferences. Advances in Neural Information Processing Systems, 38: 18977–19023, 2026.

Matthijs Kalmijn. Intermarriage and homogamy: Causes, patterns, trends. Annual review of sociology, 24(1):395–421, 1998.

Ulrich Kamecke. Two sided matching: A study in game-theoretic modeling and analysis, 1992.

Lihong Li, Yu Lu, and Dengyong Zhou. Provably optimal algorithms for generalized linear contextual bandits. In Proceedings of the 34th International Conference on Machine Learning, volume 70 of Proceedings ofMachine Learning Research, pp. 2071–2080. PMLR, 2017.

Nian Li, Chen Gao, Mingyu Li, Yong Li, and Qingmin Liao. EconAgent: Large language modelempowered agents for simulating macroeconomic activities. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 15523– 15536. Association for Computational Linguistics, 2024. doi: 10.18653/v1/2024.acl-long.829.

Xiaomin Li, Yuexing Hao, Jianheng Hou, Jintao Huang, Qianfeng Wen, Shirley Huang, Yifan Liu, Xiaoyi Liu, Yilan Fan, Yijun Wang, et al. Matraix: Simulating the world with 8.3 billion persona agents. arXiv preprint arXiv:2608.04205, 2026.

Yuantong Li, Chi-Hua Wang, Guang Cheng, and Will Wei Sun. Dynamic matching bandit for twosided online markets. arXiv preprint arXiv:2205.03699, 2022.

Lydia T. Liu, Feng Ruan, Horia Mania, and Michael I. Jordan. Bandit learning in decentralized matching markets. Journal ofMachine Learning Research, 22(211):1–34, 2021.

Jens Ludwig, Sendhil Mullainathan, and Ashesh Rambachan. Large language models: An applied econometric framework. Annual Review ofEconomics, 18, 2024.

Miller McPherson, Lynn Smith-Lovin, and James M Cook. Birds of a feather: Homophily in social networks. Annual review ofsociology, 27(1):415–444, 2001.

Robert K Merton. Intermarriage and the social structure: Fact and theory. Psychiatry, 4(3):361–374, 1941.

Josue Ortega, Gabriel Ziegler, and R Pablo Arribillaga. Unimprovable students and inequality in´ school choice. Technical report, QBS Research Paper, 2024.

Joon Sung Park, Joseph O’Brien, Carrie Jun Cai, Meredith Ringel Morris, Percy Liang, and Michael S Bernstein. Generative agents: Interactive simulacra of human behavior. In Proceedings of the 36th annual acm symposium on user interface software and technology, pp. 1–22, 2023.

Population Census Office under the State Council and Department of Population and Employment Statistics, National Bureau of Statistics. Tabulation on the 2015 1% Population Sampling Survey of the People’s Republic of China. China Statistics Press, Beijing, 2016. Micro-data collected in November 2015, commonly known as China’s mini-census.

Alvin E Roth and Marilda Sotomayor. Two-sided matching. Handbook of game theory with economic applications, 1:485–541, 1992.

Haoyang Shang, Zhengyang Yan, and Xuan Liu. Love first, know later: Persona-based romantic compatibility through llm text world engines. arXiv preprint arXiv:2512.11844, 2025.

Jan Van Bavel and Andre Grow. Introduction: Agent-based modelling as a tool to advance evolu-´ tionary population theory. In Agent-based modelling in population studies: Concepts, methods, and applications, pp. 3–27. Springer, 2016.

Jean E Veevers. The “real” marriage squeeze: Mate selection, mortality, and the mating gradient. Sociological Perspectives, 31(2):169–189, 1988.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022.

Yuzhe Yang, Yifei Zhang, Minghao Wu, Kaidi Zhang, Yunmiao Zhang, Honghai Yu, Yan Hu, and Benyou Wang. Twinmarket: A scalable behavioral and social simulation for financial markets. Advances in Neural Information Processing Systems, 38:63469–63519, 2026.

Marcel Zentner and Alice H Eagly. A sociocultural framework for understanding partner preferences of women and men: Integration of concepts and evidence. European Review ofSocial Psychology, 26(1):328–373, 2015.

Marcel Zentner and Klaudia Mitura. Stepping out of the caveman’s shadow: Nations’ gender gap predicts degree of sex differentiation in mate preferences. Psychological science, 23(10):1176– 1185, 2012.

YiRui Zhang and Zhixuan Fang. Decentralized two-sided bandit learning in matching market. In The 40th Conference on Uncertainty in Artificial Intelligence, 2024.

Jiaxu Zhou, Jen tse Huang, Xuhui Zhou, Man Ho Lam, Xintao Wang, Hao Zhu, Wenxuan Wang, and Maarten Sap. The PIMMUR principles: Ensuring validity in collective behavior of LLM societies, 2026. URL https://openreview.net/forum?id=I88toT6Leg.

Yang Zhou, Jia Yu, and Yu Xie. Gender differences and social class heterogeneity in mate preferences: An exploratory study based on the choice experiment approach. Sociological Studies, (6): 107–130, 2023. ISSN 1002-5936. URL http://sociologyol.ruc.edu.cn/shxyj/ fzshx/shfc/e96e3fbb12db4abe8eaff950c1bf9211.htm. In Chinese.

## A RELATED WORK

## A.1 LLM SOCIAL SIMULATION

A growing literature studies LLMs as conditional behavioral models and as components of interactive social and economic simulations (Argyle et al., 2023; Park et al., 2023; Horton et al., 2023; Chen et al., 2023; Gui & Toubia, 2023; Ludwig et al., 2024; Hansen et al., 2026; Hao & Xie, 2025; Li et al., 2024; Yang et al., 2026). The evidence is promising but not uniformly generalizable: persona conditioning, model choice, and task design can materially affect simulated behavior. Recent work therefore emphasizes explicit validation targets and careful interpretation of collective LLM-agent behavior (Anthis et al., 2025; Zhou et al., 2026; Ludwig et al., 2024). Our paper follows this view by treating the LLM as a behaviorally expressive component of the matching system rather than as an unconstrained oracle, and by validating the induced mate preferences against an empirical reference before studying market-level outcomes.

## A.2 MATE PREFERENCE FOR MARRIAGE

Mate-choice research offers several partially competing explanations of partner preference, including evolutionary psychology (Buss, 1989; Buss & Schmitt, 1993), sociocultural and social-role theories (Eagly et al., 2009; Zentner & Mitura, 2012; Zentner & Eagly, 2015), status and resource exchange (Davis, 1941; Merton, 1941; Kalmijn, 1998), homogamy and homophily (Kalmijn, 1998; McPherson et al., 2001), and hypergamy or mating-gradient theories (Veevers, 1988). Because these theories are often verbal, multidimensional, and only partly formalized, translating them into fixed quantitative utility functions can be restrictive (Grow, 2019). Our behavioral specification is grounded primarily in the choice experiment of Zhou et al. (2023), which studies six mate-preference dimensions, including physical appearance, and documents substantial heterogeneity across gender and social class. The LLM is used to operationalize this evidence within heterogeneous personas without imposing a single hand-crafted mate-value function.

## A.3 BIPARTITE MATCHING AND LEARNING

Recent work has used LLMs either as solvers over ranked preferences (Hosseini et al., 2026) or as centralized engines for persona-based compatibility estimation (Shang et al., 2025). A separate literature studies matching under bandit feedback, including decentralized learning, stability, fairness, and two-sided uncertainty (Liu et al., 2021; Dai & Jordan, 2021; Cen & Shah, 2022; Zhang & Fang, 2024; Li et al., 2022). Our framework differs from both lines by separating partner valuation from reciprocal-acceptance learning: LLM agents evaluate only locally encountered candidates, while agent-specific contextual bandits learn acceptance probabilities from realized binary proposal outcomes. For the latter component, we use a logistic contextual-bandit construction motivated by generalized-linear bandit methods (Filippi et al., 2010; Li et al., 2017).

## B PERSONA SAMPLING AND DEPENDENCY STRUCTURE

This appendix documents the synthetic population used throughout the validation and matching experiments. The six observable persona dimensions follow the empirical setting described in Section 2. Rather than sampling every attribute independently, the construction preserves selected socioeconomic dependencies so that the resulting agents are coherent as joint profiles rather than independent attribute bundles. Table 3 summarizes the sampling structure, and Table 4 reports the realized composition of the 100-agent market used in the main experiments.

Table 3: Persona dimensions and sampling structure.
<table><tr><td>Dimension</td><td>Values or scale</td><td>Sampling and dependency structure</td></tr><tr><td>Age</td><td>20-50</td><td>Sampled from marriage-market-active age groups according to the target population distribution.</td></tr><tr><td>Income</td><td>Monthly RMB income</td><td>Generated conditional on educational attain- ment around a reference monthly income of 1830 RMB.</td></tr><tr><td>Education</td><td>Middle school or below; high school; university and above</td><td>Sampled from population education proportions and aggregated into three experimental cate-</td></tr><tr><td>Family background</td><td>1Urban; rural</td><td>gories. Sampled from the corresponding urban/rural population composition.</td></tr><tr><td>Housing</td><td>Owns house; no house</td><td>Sampled conditionally on age, gender, and fam- ily background.</td></tr><tr><td>Appearance</td><td>Below average; average; attrac- tive</td><td>Sampled from an approximately bell-shaped distribution, with most probability mass as- signed to average appearance.</td></tr></table>

Table 4: Realized composition of the 100-agent synthetic marriage market.
<table><tr><td>Attribute</td><td>Male (n=50)</td><td>Female (n=50)</td><td>Overall (n=100)</td></tr><tr><td>Mean age</td><td>36.60</td><td>34.62</td><td>35.61</td></tr><tr><td>Mean monthly income</td><td>1881.24 RMB</td><td>1712.88 RMB</td><td>1797.06 RMB</td></tr><tr><td>Middle school or below</td><td>70%</td><td>72%</td><td>71%</td></tr><tr><td>High school</td><td>10%</td><td>14%</td><td>12%</td></tr><tr><td>University and above</td><td>20%</td><td>14%</td><td>17%</td></tr><tr><td>Urban family background</td><td>58%</td><td>56%</td><td>57%</td></tr><tr><td>Owns house</td><td>42%</td><td>40%</td><td>41%</td></tr><tr><td>Attractive appearance</td><td>8%</td><td>18%</td><td>13%</td></tr><tr><td>Average appearance</td><td>78%</td><td>66%</td><td>72%</td></tr><tr><td>Below-average appearance</td><td>14%</td><td>16%</td><td>15%</td></tr></table>

## C PROMPT TEMPLATES FOR LLM PERSONAS AND MATE PREFERENCES

This appendix records the prompt templates used in the experiments. Public personas are deterministic natural-language renderings of the sampled attributes, whereas the private mate-preference prompt is available only to the evaluating agent. For each directional dyadic evaluation, the model receives the evaluator’s public profile, the candidate’s public profile, and the evaluator’s genderspecific private preference specification.

## C.1 PUBLIC PERSONA RENDERING TEMPLATE

<table><tr><td>Public persona template</td></tr><tr><td>For each generated profile, render the public persona as: A [age]-year-old [gender] with [education phrase]. Financially, this individual earns a monthly income of [actual monthly income] RMB (approximately [annual income] RMB per year) and [housing phrase]. [family-background phrase]. In terms of physical appearance, this person is evaluated as having an [appearance] appearance. The structured fields stored with the persona are: agent_id, gender, age, actual_monthly-income, family-background, housing, education_level, appearance, and short_description.</td></tr><tr><td>Dyadic scoring user message YOUR OWN PUBLIC PROFILE (for Step 0 self-calibration) Agent ID: [evaluator id] Gender: [evaluator gender]</td></tr><tr><td>Age: [evaluator age] Actual Monthly Income: [evaluator income] RMB Family Background: [evaluator family background] Housing: [evaluator housing] Education Level: [evaluator education level] Appearance: [evaluator appearance] Short Description: [evaluator public persona] CANDIDATE PUBLIC PROFILE (to evaluate) Agent ID: [candidate id] Gender: [candidate gender] Age: [candidate age] Actual Monthly Income: [candidate income] RMB</td></tr></table>

Housing: [candidate housing]   
Education Level: [candidate education level]   
Appearance: [candidate appearance]   
Short Description: [candidate public persona]   
Evaluate the candidate strictly following your private preference theory. Output only the JSON   
object; do not include markdown fences or extra commentary.

## C.2 PRIVATE MATE-PREFERENCE PROMPTS

## Male-agent private mate-preference prompt

You are a male agent in a simulated marriage market. You will evaluate opposite-gender candidates using two inputs: your own public profile and the candidate’s public profile. Both profiles contain age, actual monthly income, family background, housing, education level, appearance, and short description.

Use the following internal five-step evaluation protocol before assigning scores. Do not output your full reasoning chain. Only output structured dimension scores, an overall desirability score, and a brief explanation.

Paper-aligned preference theory. Male agents place the strongest relative emphasis on physical appearance. Income, housing, education, and family background still matter, but they are secondary. Candidate age has a negative effect when the candidate is older relative to the evaluator. The evaluator’s own public profile should be used for self-positioning and relative comparison, but it must not overturn the main gender-specific priority structure.

Step 0: Self-profile calibration. Inspect your own age, income, housing, education, family background, and appearance. Use your own age as the reference point for relative age fit. Use your own socioeconomic and appearance profile to calibrate expectations mildly and realistically.

Step 1: Appearance-dominant baseline check. Evaluate the candidate’s appearance first. below average receives a large negative penalty; average is an acceptable neutral baseline; attractive receives a strong positive premium. Appearance should be the most important upward adjustment for male agents.

Step 2: Relative age disutility. Compare the candidate’s age with your own age. Desirability should decrease smoothly as the candidate becomes older relative to you. Younger or similar-age candidates should be evaluated more favorably than clearly older candidates.

Step 3: Secondary income and housing adjustment. Treat 1830 RMB/month as the baseline monthly disposable income. Higher candidate income should increase desirability moderately. Home ownership receives a positive bonus relative to no house. These socioeconomic comparisons remain secondary to appearance.

Step 4: Education and family-background screening. Evaluate education in the order university and above > high school > middle school and below. Urban family background may receive a very small bonus, but family background should have weak influence on the final score.

Scoring instructions. Return dimension scores on a 0–100 scale: age fit, income fit, family background fit, housing fit, education fit, and appearance fit. Return overall desirability on a 0–100 scale. Do not compute overall desirability as a simple average of dimension scores; it must reflect the priority structure above. Do not mention statistical coefficients, formulas, or the paper name. Keep brief reason under 150 characters.

Output schema. { evaluator id, candidate id, dimension scores: { age fit, income fit, family background fit, housing fit, education fit, appearance fit }, overall desirability, brief reason }.

The implemented prompt additionally includes few-shot calibration examples emphasizing that an attractive candidate can outrank a more socioeconomically advantaged but below-averagelooking candidate, and that age fit is evaluated relative to the male evaluator.

## Female-agent private mate-preference prompt

You are a female agent in a simulated marriage market. You will evaluate opposite-gender candidates using two inputs: your own public profile and the candidate’s public profile. Both profiles contain age, actual monthly income, family background, housing, education level, appearance, and short description.

Use the following internal five-step evaluation protocol before assigning scores. Do not output your full reasoning chain. Only output structured dimension scores, an overall desirability score, and a brief explanation.

Paper-aligned preference theory. Female agents place the strongest relative emphasis on socioeconomic resources and social-stratum indicators, especially housing, actual income, education, and family background. Physical appearance still matters, especially when it is below average, but it should not dominate socioeconomic resources. Candidate age has a negative effect when the candidate is older relative to the evaluator. The evaluator’s own public profile should be used for self-positioning and relative comparison, but it must not overturn the main female-agent priority structure.

Step 0: Self-profile calibration. Inspect your own age, income, housing, education, family background, and appearance. Use your own age as the reference point for relative age fit. Use your socioeconomic profile to calibrate expectations mildly and realistically.

Step 1: Socioeconomic resource gatekeeping. First evaluate the candidate’s housing and actual monthly income. owns house receives a large positive bonus; no house receives a clear penalty, especially when income is also low. Treat 1830 RMB/month as the baseline monthly disposable income. Higher income should translate into a strong monotonic increase in desirability. Income and housing should be more decisive than appearance.

Step 2: Education and family-background alignment. Evaluate education in the order university and above > high school > middle school and below. University education receives a strong bonus; middle school or below receives a clear penalty. Urban family background receives a distinct positive bonus relative to rural background. Education and family background matter because they signal social stratum and class alignment.

Step 3: Appearance threshold check. Evaluate appearance after socioeconomic and classrelated traits. below average receives a meaningful negative penalty; average is acceptable; attractive receives a positive bonus. The attractiveness bonus should not override serious disadvantages in housing, income, education, and family background.

Step 4: Relative age sensitivity. Compare the candidate’s age with your own age. Desirability should decrease as the candidate becomes older relative to you. Apply a clear penalty for substantially older candidates. The age penalty for female agents should be at least as strong as, and often stronger than, the male-agent age penalty.

Scoring instructions. Return dimension scores on a 0–100 scale: age fit, income fit, family background fit, housing fit, education fit, and appearance fit. Return overall desirability on a 0–100 scale. Do not compute overall desirability as a simple average of dimension scores; it must reflect the priority structure above. Avoid excessive ties. Use decimal scores when candidates are close. Do not mention statistical coefficients, formulas, or the paper name. Keep brief reason under 150 characters.

Output schema. { evaluator id, candidate id, dimension scores:   
{ age fit, income fit, family background fit, housing fit,   
education fit, appearance fit }, overall desirability,   
brief reason }.

The implemented prompt additionally includes few-shot calibration examples emphasizing that socioeconomic resources and class-alignment indicators can dominate attractiveness, and that income and housing are traded off rather than reduced to a single attribute.

## D VALIDATION METRICS

Let $s _ { i j }$ denote the LLM desirability score assigned by evaluator i to candidate $j ,$ and let $u _ { i j }$ denote the corresponding utility from the gender-specific conditional-logit reference based on Zhou et al. (2023). All ranking metrics are computed at the evaluator level and then aggregated across evaluators and seeds.

Kendall’s τ . We report Kendall’s $\tau _ { b } .$ , which accounts for ties,

$$
\tau _ { i } = \frac { C _ { i } - D _ { i } } { \sqrt { ( C _ { i } + D _ { i } + T _ { i } ^ { ( s ) } ) ( C _ { i } + D _ { i } + T _ { i } ^ { ( u ) } ) } } ,\tag{11}
$$

where $C _ { i }$ and $D _ { i }$ are concordant and discordant candidate pairs, while $T _ { i } ^ { ( s ) }$ and $T _ { i } ^ { ( u ) }$ count pairs tied only under the LLM score and reference utility, respectively. Higher values indicate stronger ordinal agreement.

Weighted pairwise violation rate (WPVR). To place more weight on reversals between candidates that are well separated by the reference model, we use

$$
\mathrm { W P V R } _ { i } = \frac { \sum _ { a < b } \mathbf { 1 } \{ \left( { { s } _ { i a } } - { { s } _ { i b } } \right) \left( { { u } _ { i a } } - { { u } _ { i b } } \right) < 0 \} \left| { { u } _ { i a } } - { { u } _ { i b } } \right| } { \sum _ { a < b } \left| { { u } _ { i a } } - { { u } _ { i b } } \right| } .\tag{12}
$$

Lower values indicate fewer economically consequential ranking reversals.

Top-K overlap. Agreement among the highest-ranked candidates is measured by

$$
\mathrm { O v e r l a p @ } K _ { i } = \frac { \left| \mathrm { T o p K } _ { i } ^ { \mathrm { L L M } } \cap \mathrm { T o p K } _ { i } ^ { \mathrm { R e f } } \right| } { K } .\tag{13}
$$

The labels Hit@3 and Hit@5 in the main validation figure use this same set-overlap definition. Because random overlap depends on candidate-pool size, these values should not be compared mechanically between the $1 0 \times 1 0$ and $5 0 \times 5 0$ settings.

Dimension consistency. As an internal diagnostic, we compute Pearson correlations between overall desirability and each of the six dimension-level scores, separately by evaluator gender. This checks whether the decomposition follows the intended gender-specific structure, but it is not an independent behavioral validation because both quantities are generated by the same LLM evaluation.

## E ROBUSTNESS ACROSS LLM BACKBONES

Table 5 reports the $1 0 \times 1 0$ validation across DeepSeek, Qwen, GPT-OSS, and the no-CoT specification. For both evaluator genders, all preference-structured prompts achieve higher Kendall’s τ and lower WPVR than the no-CoT baseline. Although the absolute alignment varies across backbones, the direction of the improvement is consistent, suggesting that the empirical-reference alignment is not specific to a single LLM.

Table 5: $1 0 \times 1 0$ reference-alignment results across LLM backbones. Values are means ± standard deviations across five random seeds.
<table><tr><td>Model</td><td>Male τ</td><td>Female τ</td><td>Male WPVR</td><td>Female WPVR</td></tr><tr><td>DeepSeek CoT</td><td> $0 . 6 9 2 \pm 0 . 0 7 6$ </td><td> $0 . 5 7 2 \pm 0 . 0 8 2$ </td><td> $0 . 0 6 4 \pm 0 . 0 2 5$ </td><td> $0 . 1 2 2 \pm 0 . 0 2 7$ </td></tr><tr><td>Qwen CoT</td><td> $0 . 7 0 8 \pm 0 . 1 1 5$ </td><td> $0 . 6 4 6 \pm 0 . 0 7 2$ </td><td> $0 . 0 5 5 \pm 0 . 0 3 5$ </td><td> $0 . 0 8 9 \pm 0 . 0 2 2$ </td></tr><tr><td>GPT-OSS CoT</td><td> $0 . 6 6 6 \pm 0 . 1 0 3$ </td><td> $0 . 5 0 2 \pm 0 . 0 9 2$ </td><td> $0 . 0 7 3 \pm 0 . 0 4 2$ </td><td> $0 . 1 5 8 \pm 0 . 0 3 0$ </td></tr><tr><td>No-CoT baseline</td><td> $0 . 2 8 6 \pm 0 . 0 6 5$ </td><td> $0 . 4 0 0 \pm 0 . 0 8 5$ </td><td> $0 . 2 9 8 \pm 0 . 0 5 5$ </td><td> $0 . 2 3 6 \pm 0 . 0 5 3$ </td></tr></table>

## F 50×50 SCALE-UP VALIDATION

We repeat the preference validation on the full 100-agent population. Each of the 50 male and 50 female evaluators scores all 50 opposite-gender candidates, yielding 2,500 cross-gender dyads and 5,000 directional evaluations. Table 6 reports the exact alignment statistics, while Figure 5 additionally shows dimension-level consistency. Kendall’s τ and WPVR remain broadly comparable to the 10×10 results, indicating that ordinal alignment persists as the candidate pool expands; top-K overlap is naturally more demanding in the larger pool.

Table 6: DeepSeek alignment with the empirical preference reference in the $5 0 \times 5 0$ validation market.
<table><tr><td>Metric</td><td>Male evaluators</td><td>Female evaluators</td></tr><tr><td>Kendall&#x27;s τ</td><td> $0 . 7 0 3 \pm 0 . 0 5 4$ </td><td> $0 . 6 0 7 \pm 0 . 0 8 2$ </td></tr><tr><td>WPVR</td><td> $0 . 0 5 4 \pm 0 . 0 1 8$ </td><td> $0 . 1 0 5 \pm 0 . 0 3 9$ </td></tr><tr><td>Overlap@3</td><td> $0 . 4 6 0 \pm 0 . 1 8 7$ </td><td> $0 . 4 7 3 \pm 0 . 2 3 2$ </td></tr><tr><td>Overlap@5</td><td> $0 . 6 8 4 \pm 0 . 1 3 3$ </td><td> $0 . 4 4 8 \pm 0 . 1 8 6$ </td></tr></table>

![](images/2fe2c64837f2697c3c7c23d4724ebc4727289a3a8277131a3da21aca19b79ee8.jpg)  
Figure 5: DeepSeek validation in the $5 0 \times 5 0$ market. Panel (a) reports reference-alignment metrics by evaluator gender; Panel (b) reports dimension-consistency correlations.

## G CONTEXT CONSTRUCTION FOR LOCAL RECIPROCAL-ACCEPTANCE LEARNING

This appendix specifies the six-dimensional dyadic compact representation used by the local Logistic-UCB models. For proposer i and candidate j, the context is written as $x _ { j , t } ^ { ( i ) } = \phi ( z _ { i } , z _ { j } ) \in$ $\mathbb { R } ^ { 6 }$ , where $z _ { i }$ and $z _ { j }$ contain the six public persona attributes introduced in Section 2. The representation summarizes only information observable before a proposal; it is intended to predict reciprocal acceptance, not to reconstruct the receiver’s private utility function.

Table 7: Features in the local reciprocal-acceptance context. Gap and advantage variables follow the proposer-oriented encoding used in the implementation.
<table><tr><td>Feature</td><td>Interpretation</td></tr><tr><td>Age gap</td><td>Age mismatch between proposer and candidate</td></tr><tr><td>Income gap</td><td>Relative income difference</td></tr><tr><td>Education gap</td><td>Relative education difference</td></tr><tr><td>Same background</td><td>Indicator for shared family background</td></tr><tr><td>Housing advantage</td><td>Relative housing-status advantage</td></tr><tr><td>Appearance gap</td><td>Relative appearance difference</td></tr></table>

The decentralization constraint excludes receiver-side private quantities such as $U _ { j } ( i ) , U _ { j } ( \mu _ { t } ( j ) )$ and $b _ { j , t } ^ { R }$ , as well as the receiver’s complete ranking, future accept/reject decision, other proposers’ outcomes, and the parameters or histories of other local models. Agent i learns only from its own realized proposal history $\mathcal { H } _ { i , t } = \{ ( x _ { n } ^ { ( i ) } , y _ { n } ^ { ( i ) } ) \} _ { n = 1 } ^ { N _ { i , t } }$ . A candidate who is observed but not approached therefore generates no label rather than a rejection. This preserves the partial-feedback structure required by the contextual-bandit interpretation.

## H MAIN MATCHING EXPERIMENT SETTINGS AND METRICS

## H.1 EXPERIMENTAL PROTOCOL

The main experiment uses the same 50 male and 50 female agents described in Section 2. The LLMsolver leverages DeepSeek-V4-Pro as the centralized solver, the same backbone of our LLM-ABM markets of $5 0 \times 5 0$ For every ordered cross-gender dyad, the validated LLM preference model produces a directional score $s _ { i } ( j ) \in [ 0 , 1 0 0 ] \mathrm { : }$ : Gale–Shapley, Axtell–Kimbrough and LLM-solver receive the preference information required by their mechanisms, whereas an LLM-agent gets its preference only after the local encounter.

The “Info required” column in Table 1 summarizes this distinction. Whole denotes complete marketwide preferences available before matching; Partial denotes decentralized interaction that still begins from pre-specified preference lists; and Local denotes the proposed encounter-based regime, in which no pairwise valuation or ranking is available to an agent before local exposure.

Table 8: Main LLM-ABM parameter configuration used in Table 1.
<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Market size</td><td>50 male agents and 50 female agents</td></tr><tr><td>Repeated runs</td><td>50 random seeds</td></tr><tr><td>Main matching LLM scorer</td><td>DeepSeek V4 Pro, temperature = 0.5, CoT prompt</td></tr><tr><td>Matching horizon</td><td>T = 500 periods</td></tr><tr><td>Candidate set size</td><td>25 candidates per activation when more are available</td></tr><tr><td>Reservation utility</td><td>r = 0.25 on the normalized utility scale</td></tr><tr><td>Lock probability</td><td> $\rho = 0 . 0 5$  per temporary relationship per period</td></tr><tr><td>Utility threshold</td><td> $\epsilon _ { U } = 0 . 0 1$ </td></tr><tr><td>Logistic-UCB coefficient</td><td> $\beta = 1 . 0$ </td></tr><tr><td>Logistic regularization</td><td> $\lambda = 1 . 0$ </td></tr><tr><td>Proposal cost and switching cost 0</td><td></td></tr></table>

The three LLM-ABM policies share the same market dynamics and differ only in proposal selection. Random eligible samples uniformly from utility-improving candidates; Utility-only chooses the largest immediate utility gain; and Bandit-UCB maximizes $\bar { \Delta { U _ { i j , t } } } p _ { i j , t } ^ { \mathrm { U C B } , i }$ . Hence, Utility-only isolates the incremental role of reciprocal-acceptance learning, while Random eligible provides a non-strategic decentralized-search baseline.

## H.2 EVALUATION METRICS

Let $\mathcal { P } ( \mu ) = \{ ( m , w ) : \mu ( m ) = w , m \in \mathcal { M } , w \in \mathcal { W } \}$ denote the set of realized male–female pairs.

Mutual welfare. We report average bilateral desirability,

$$
\operatorname { M u t u a l } ( \mu ) = { \frac { 1 } { | { \mathcal { P } } ( \mu ) | } } \sum _ { ( m , w ) \in { \mathcal { P } } ( \mu ) } { \frac { s _ { m } ( w ) + s _ { w } ( m ) } { 2 } } .\tag{14}
$$

This rewards matches that are jointly valued by both partners rather than optimizing one side alone.

Realized rank and gender rank gap. Using score-based tied ranks,

$$
\operatorname { r a n k } _ { i } ( j ) = 1 + \sum _ { k \in \mathcal { C } _ { - i } } { \bf 1 } \{ s _ { i } ( k ) > s _ { i } ( j ) \} ,\tag{15}
$$

where $\mathcal { C } _ { - i }$ is the full opposite-side candidate set used only for ex post evaluation. If $\bar { r } _ { M } ( \mu )$ and $\bar { r } _ { W } ( \mu )$ are the average realized partner ranks of matched men and women, then

$$
\mathrm { R a n k G a p } ( \mu ) = \left| { \bar { r } } _ { M } ( \mu ) - { \bar { r } } _ { W } ( \mu ) \right| .\tag{16}
$$

Blocking pairs. Let $u _ { i } ( j ) = s _ { i } ( j ) / 1 0 0$ and define the final benchmark

$$
B _ { i } ( \mu ) = \left\{ \begin{array} { l l } { \operatorname* { m a x } \{ r _ { i } , u _ { i } ( \mu ( i ) ) \} , } & { \mu ( i ) \neq \emptyset , } \\ { r _ { i } , } & { \mu ( i ) = \emptyset . } \end{array} \right.\tag{17}
$$

An unmatched pair $( m , w ) \not \in { \mathcal { P } } ( \mu )$ is blocking when $u _ { m } ( w ) > B _ { m } ( \mu )$ and $u _ { w } ( m ) > B _ { w } ( \mu )$ . This is an ex post stability diagnostic; agents do not observe the blocking-pair set during matching.

Proposal volume. We count all proposal attempts as a measure of search intensity. Because Gale–Shapley proposals are centralized algorithmic operations whereas LLM-ABM and Axtell– Kimbrough proposals arise from decentralized interaction, this quantity is slightly different for centralized and decentralized scenarios.

## I COUNTERFACTUAL WARM-START EXPERIMENT

The counterfactual experiments in Section 4.4 use the same 50 × 50 population and Bandit-UCB configuration as the main matching experiment. Source runs use the same 50 seeds as the main experiment, whereas counterfactual evaluation uses another 10 seeds. The source phase provides learned proposer-side Logistic-UCB coefficients that are used only for initialization in the treatment conditions.

Individual warm start. Each male agent is treated as the focal agent in turn. Only that agent receives the corresponding source-phase initialization; all other agents remain cold-start learners. For every held-out seed, the warm-start run is paired with a cold-start Bandit-UCB run using the same seed, holding fixed the stochastic market realization associated with the evaluation seed.

Male-side warm start. We also initialize all male agents with the pooled male-proposer coefficients learned from the source runs, while female agents remain cold-start learners. This stronger intervention tests whether information that can help an individual retains its value once it is shared across an entire side of the market.

Warm-starting changes only the initial coefficient vector θ. It does not alter LLM utilities, candidate exposure, reservation utilities, proposal rules, receiver behavior, or the observed proposal history, and subsequent feedback can revise or erase the initial belief.

Table 9: Distribution of individual warm-start effects across male focal agents.
<table><tr><td>Metric</td><td>Improved</td><td>Unchanged</td><td>Worsened</td><td>Avg. gain</td><td>Avg. loss</td><td>Mean effect</td></tr><tr><td>Own rank</td><td>11</td><td>26</td><td>13</td><td>0.95</td><td>0.46</td><td>-0.15</td></tr><tr><td>Own score</td><td>11</td><td>26</td><td>13</td><td>0.95</td><td>0.55</td><td>-0.22</td></tr><tr><td>Proposals</td><td>22</td><td>3</td><td>25</td><td>8.20</td><td>15.37</td><td>-5.00</td></tr></table>

Table 9 uses the focal agent as the unit of analysis. “Avg. gain” reports the average signed improvement among agents with a positive effect, while “Avg. loss” reports the average absolute loss among agents with a non-positive effect, with unchanged agents contributing zero. For rank and proposal volume, signs are oriented so that positive values indicate improvement; for score, higher values directly indicate improvement.

## J SENSITIVITY ANALYSIS OF EXPLORATION AND COMMITMENT

We further examine how the LLM-ABM market responds to two key behavioral parameters: the exploration coefficient $\beta$ in Bandit-UCB and the permanent lock probability $\rho .$ The $\beta$ sweep varies the strength of uncertainty-driven exploration while holding $\rho = 0 . 0 5$ fixed, whereas the $\rho$ sweep varies the rate at which temporary matches become permanent while holding $\beta = 1$ fixed. All experiments use the same $5 0 \times 5 0$ market and 50 random seeds. For low $- \rho$ settings, we use a smaller per-activation proposal budget (L) to keep within-round search bounded.

![](images/26948268fbb260790b32774413c43fc1f0530b1577c31877a53f0907d4f66812.jpg)

![](images/6f212425c987a5167beb2124aa41770586d0a54245c35d8e5b4b2c003c915ebf.jpg)

![](images/a7db00b8a0c5436cf372976e67d610a020679a3f77baa976689896e2c52cc431.jpg)  
Figure 6: Sensitivity of market-level outcomes to exploration strength $\beta$ and lock probability $\rho .$ The blue curve varies $\beta$ with $\rho = 0 . 0 5$ fixed, while the orange curve varies $\rho$ with $\beta = 1$ fixed. Shaded regions indicate 95% confidence intervals across 50 seeds; stars mark the main experimental setting.

Figure 6 shows that the final matched ratio remains relatively stable across both parameter sweeps and stays close to the main setting for most values of $\mathit { \Pi } ^ { \cdot } \beta$ and $\rho .$ This indicates that the market reaches a high level of aggregate matching under a broad range of exploration and commitment intensities, although some variation in the share of unmatched agents remains.

The blocking-pair results are more sensitive to the commitment parameter. Varying $\beta$ changes the number of blocking pairs only moderately, suggesting that the main stability pattern is not driven by a narrow choice of exploration strength. In contrast, $\rho$ has a more pronounced effect. Very low commitment probabilities prolong the search process and leave more relationships unresolved, whereas very high commitment probabilities cause agents to exit the market earlier and can lock in less stable matches. The main setting, $\rho = 0 . 0 5$ , lies in a relatively low-blocking region of the sensitivity curve and provides a reasonable balance between continued search and timely commitment.

Proposal volume reveals a complementary search-intensity trade-off. Larger $\beta$ generally increases the number of proposals because stronger exploration encourages agents to consider more uncertain candidates. Increasing $\rho ,$ by contrast, sharply reduces proposal volume because agents become permanently matched and leave the active market more quickly. This reduction in search intensity, however, is accompanied by a higher blocking count at large $\rho .$ Overall, the main configuration is not selected to optimize any single metric in isolation; rather, it provides a balanced operating point with a high final matched ratio, relatively few blocking pairs, and moderate proposal volume.

## K DERIVATION OF THE LOCAL LOGISTIC-UCB DECISION RULE

This appendix derives the practical Fisher-information-based proposal rule used in Section 3.3. Appendix L analyzes a calibrated variant and specifies its assumptions and scope.

## K.1 LOCAL LOGISTIC ACCEPTANCE MODEL

For proposer $i ,$ suppose reciprocal acceptance is represented by a logistic working model

$$
\begin{array} { r } { \operatorname* { P r } ( Y _ { i  j , t } = 1 \mid x _ { j , t } ^ { ( i ) } ) = \sigma ( { x _ { j , t } ^ { ( i ) } } ^ { \top } \theta _ { i } ) , } \end{array}\tag{18}
$$

where

$$
\sigma ( z ) = \frac { 1 } { 1 + \exp ( - z ) } .\tag{19}
$$

Given local proposal history $\mathcal { H } _ { i , 1 }$ <sub>t</sub>, the regularized negative log-likelihood is

$$
\mathcal { L } _ { i , t } ( \boldsymbol { \theta } ) = \sum _ { n = 1 } ^ { N _ { i , t } } \left[ \log \left( 1 + \exp \left( \boldsymbol { x } _ { n } ^ { ( i ) } ^ { \top } \boldsymbol { \theta } \right) \right) - { y } _ { n } ^ { ( i ) } { \boldsymbol { x } _ { n } ^ { ( i ) } } ^ { \top } \boldsymbol { \theta } \right] + \frac { \lambda _ { i } } { 2 } \| \boldsymbol { \theta } \| _ { 2 } ^ { 2 } .\tag{20}
$$

Its gradient is

$$
\nabla { \mathcal L } _ { i , t } ( \theta ) = \sum _ { n = 1 } ^ { N _ { i , t } } \left[ \sigma \left( { x _ { n } ^ { ( i ) } } ^ { \top } \theta \right) - y _ { n } ^ { ( i ) } \right] x _ { n } ^ { ( i ) } + \lambda _ { i } \theta .\tag{21}
$$

The Hessian is

$$
\nabla ^ { 2 } \mathcal { L } _ { i , t } ( \theta ) = \lambda _ { i } I + \sum _ { n = 1 } ^ { N _ { i , t } } \sigma \left( { x _ { n } ^ { ( i ) } } ^ { \top } \theta \right) \left[ 1 - \sigma \left( { x _ { n } ^ { ( i ) } } ^ { \top } \theta \right) \right] x _ { n } ^ { ( i ) } { x _ { n } ^ { ( i ) } } ^ { \top } .\tag{22}
$$

Evaluating Equation 22 at the fitted parameter $\widehat { \theta } _ { i , t }$ yields

$$
V _ { i , t } = \lambda _ { i } I + \sum _ { n = 1 } ^ { N _ { i , t } } \widehat { p } _ { n , t } ^ { ( i ) } \left( 1 - \widehat { p } _ { n , t } ^ { ( i ) } \right) x _ { n } ^ { ( i ) } x _ { n } ^ { ( i ) } { } ^ { \top } ,\tag{23}
$$

which is the local curvature matrix used in the main text.

## K.2 UNCERTAINTY IN THE LOGIT PREDICTOR

Suppose, for interpretation, that a local confidence region around the fitted parameter can be approximated by

$$
\mathcal { C } _ { i , t } = \left\{ \theta : \left\| \theta - \widehat { \theta } _ { i , t } \right\| _ { V _ { i , t } } \leq \beta _ { i , t } \right\} ,\tag{24}
$$

Preprint. Under review.

where

$$
\| z \| _ { V } = { \sqrt { z ^ { \top } V z } } .\tag{25}
$$

For a candidate context $x _ { j , t } ^ { ( i ) } .$

$$
{ x _ { j , t } ^ { ( i ) } } ^ { \top } \left( \theta - \widehat \theta _ { i , t } \right) = \left( V _ { i , t } ^ { - 1 / 2 } x _ { j , t } ^ { ( i ) } \right) ^ { \top } \left( V _ { i , t } ^ { 1 / 2 } \left( \theta - \widehat \theta _ { i , t } \right) \right)\tag{26}
$$

$$
\leq \left. V _ { i , t } ^ { - 1 / 2 } x _ { j , t } ^ { ( i ) } \right. _ { 2 } \left. V _ { i , t } ^ { 1 / 2 } \left( \theta - \widehat \theta _ { i , t } \right) \right. _ { 2 }\tag{27}
$$

$$
\leq \beta _ { i , t } \sqrt { { x _ { j , t } ^ { ( i ) } } ^ { \top } V _ { i , t } ^ { - 1 } x _ { j , t } ^ { ( i ) } } .\tag{28}
$$

This motivates the uncertainty measure

$$
\begin{array} { r } { s _ { j , t } ^ { ( i ) } = \sqrt { { x _ { j , t } ^ { ( i ) } } ^ { \top } V _ { i , t } ^ { - 1 } x _ { j , t } ^ { ( i ) } } . } \end{array}\tag{29}
$$

The corresponding optimistic linear predictor is

$$
z _ { i j , t } ^ { \mathrm { U C B } , i } = \widehat { \theta } _ { i , t } ^ { \top } x _ { j , t } ^ { ( i ) } + \beta _ { i , t } s _ { j , t } ^ { ( i ) } .\tag{30}
$$

Since the sigmoid link is strictly increasing,

$$
z _ { 1 } \leq z _ { 2 } \implies \sigma ( z _ { 1 } ) \leq \sigma ( z _ { 2 } ) ,\tag{31}
$$

we obtain the optimistic acceptance prediction

$$
p _ { i j , t } ^ { \mathrm { U C B } , i } = \sigma \left( \widehat { \theta } _ { i , t } ^ { \top } x _ { j , t } ^ { ( i ) } + \beta _ { i , t } s _ { j , t } ^ { ( i ) } \right) .\tag{32}
$$

This construction follows the general optimistic principle used in generalized-linear contextual bandits (Filippi et al., 2010; Li et al., 2017). Our particular use of the fitted logistic Hessian in Equation 23 should be understood as a local Fisher-information or Wald/Laplace-style approximation. Because the Hessian of a logistic model depends on the unknown parameter itself, more refined Logistic-UCB analyses can require nonlinear confidence sets and stronger regularity conditions. Accordingly, in the main experiments, $\beta _ { i , t }$ is treated as an exploration coefficient rather than as an exact finite-sample confidence radius, and $\widehat { \theta } _ { i , t }$ is obtained from the unconstrained regularized problem. The analyzable variant in Appendix L instead restricts the estimator to a norm ball, which is what makes its curvature constant follow from primitive bounds.

## K.3 PROPOSAL INDEX AS OPTIMISTIC EXPECTED UTILITY IMPROVEMENT

The bandit observes the binary response

$$
Y _ { i  j , t } \in \{ 0 , 1 \} ,\tag{33}
$$

but the economic objective of the proposer is not simply to maximize acceptance probability.

For an eligible candidate $j \in \mathcal { E } _ { t } ( i )$ , define the realized utility improvement from a proposal as

$$
R _ { i \to j , t } = Y _ { i \to j , t } \Delta U _ { i j , t } .\tag{34}
$$

Conditional on the information available to proposer i,

$$
\mathbb { E } [ R _ { i  j , t } \mid \boldsymbol { x } _ { j , t } ^ { ( i ) } ] = \Delta U _ { i j , t } \operatorname* { P r } ( Y _ { i  j , t } = 1 \mid \boldsymbol { x } _ { j , t } ^ { ( i ) } ) .\tag{35}
$$

The eligible-set restriction implies

$$
\begin{array} { r } { \Delta U _ { i j , t } > 0 . } \end{array}\tag{36}
$$

Therefore, replacing the unknown reciprocal-acceptance probability with an optimistic estimate preserves the direction of the upper bound and leads to

$$
I _ { i j , t } = \Delta U _ { i j , t } p _ { i j , t } ^ { \mathrm { U C B } , i } .\tag{37}
$$

This derivation clarifies why the positivity restriction in Equation 3 is important. If $\Delta U _ { i j , t } < 0 .$ multiplication by a probability upper bound would reverse the relevant inequality, and Equation 37 would no longer admit the same optimistic expected-improvement interpretation.

## L THEORETICAL ANALYSIS OF LOCAL LEARNING AND BILATERAL REMATCHING

## L.1 OVERVIEW OF THEORETICAL RESULTS

The theoretical analysis fixes one proposer and studies its own sequence of realized proposal attempts. It is intentionally local: the benchmark is a myopic oracle facing the same currently eligible candidates, rather than a market-wide matching trajectory. Under this scope, the appendix establishes five results.

R1. Oracle one-step optimality. If current reciprocal-acceptance probabilities were known, maximizing $\Delta U \times p$ maximizes the proposer’s conditional expected one-step utility improvement.

R2. Local pseudo-regret under a correctly specified working model. Under predictable bounded contexts, bounded positive utility improvements, and a proposer-specific realizable logistic acceptance model, a theoretically calibrated Logistic-UCB rule with a normconstrained local estimator has sublinear local pseudo-regret with high probability. Exact realizability need not hold in the mechanism of Section 3.3; R2 is stated as a benchmark that isolates the statistical cost of learning reciprocation and supplies the reference rate for R3.

R3. Dynamic misspecification. The decentralized design of Section 3.3 induces a nonstationary acceptance law, so the logistic model is a working approximation rather than a data-generating process. If it matches the true, time-varying acceptance probability up to a bounded envelope $\eta _ { n } .$ , the regret bound separates statistical learning error from an explicit cumulative misspecification term. Observable dynamic context can shrink this term but cannot generally eliminate receiver-private state information by construction.

R4. Strict bilateral improvement. Every accepted rematching strictly improves the partner utility of both consenting agents relative to their pre-proposal benchmarks.

R5. Matched-pair welfare non-monotonicity. Strict bilateral improvement does not imply monotone matched-pair welfare because an accepted rematching can displace previous partners. Consequently, local no-regret learning alone does not imply terminal optimality of the matched-pair welfare metric or stability.

The analysis uses a norm-constrained estimator and calibrated confidence radii, whereas the experi ments use an unconstrained estimator with fixed exploration. Remark L.1 explains the relationship between the acceptance model and the simulated mechanism.

## L.2 FORMAL RESULTS

Proposal-time notation and learner filtration. Fix proposer i and index its own realized proposal attempts by $n = 1 , 2 , . . . ,$ distinct from the global market-period index. Let $\mathcal { F } _ { n - 1 }$ be the information available to the proposer immediately before attempt n. It contains the proposer’s past proposal contexts and outcomes, the currently exposed and eligible candidate set, all public information available for current candidates, the current fitted local model, and algorithmic randomness revealed before the current proposal. It excludes receiver-private quantities that the mechanism does not reveal, including the receiver’s private valuation of the proposer and the value of the receiver’s incumbent relationship.

At attempt n, let $\mathcal { R } _ { n }$ be the current eligible candidate set. For each $j \in \mathcal { R } _ { n } ,$ , the proposer observes $x _ { j , n } \in \bar { \mathbb { R } ^ { d } }$ and a positive utility improvement $\Delta U _ { n } ( j ) > 0$ . Let $Y _ { n } ( { \dot { j } } ) \in \{ 0 , 1 \}$ denote the potential reciprocal-acceptance response that would be observed if candidate j were selected at attempt n. If candidate $j$ is selected, write

$$
J _ { n } = j , \qquad X _ { n } : = x _ { J _ { n } , n } , \qquad Y _ { n } : = Y _ { n } ( J _ { n } ) .
$$

For each current candidate define

$$
p _ { n } ( j ) : = \mathbb { P } ( Y _ { n } ( j ) = 1 \mid \mathcal { F } _ { n - 1 } ) ,
$$

and the potential one-step proposal reward

$$
R _ { n } ( j ) : = Y _ { n } ( j ) \Delta U _ { n } ( j ) , \qquad g _ { n } ( j ) : = \mathbb { E } [ R _ { n } ( j ) \mid \mathcal { F } _ { n - 1 } ] = \Delta U _ { n } ( j ) p _ { n } ( j ) .
$$

The selected context $X _ { n }$ is required to be $\mathcal { F } _ { n - 1 }$ <sub>1</sub>-measurable. Thus the policy may adapt arbitrarily to past outcomes and to the current candidate set, but must select the current context before observing $Y _ { n }$

Proposition L.1 (R1: Oracle one-step proposal optimality). Condition on $\mathcal { F } _ { n - 1 }$ and suppose the proposer makes one current proposal from $\mathcal { R } _ { n }$ . If the true conditional acceptance probabilities $\mathbf { \dot { \{ p } }  _ { n } ( j ) : j \in \mathcal { R } _ { n } \}$ were known, then every

$$
J _ { n } ^ { \star } \in \arg \operatorname* { m a x } _ { j \in \mathcal { R } _ { n } } \Delta U _ { n } ( j ) p _ { n } ( j )
$$

maximizes the proposer’s conditional expected one-step utility improvement.

This is a myopic statement. If a rejection leaves a continuation opportunity within the same activation, optimizing an entire sequence of proposals is a dynamic decision problem and need not coincide with ranking candidates solely by one-step expected improvement.

## Assumptions for local learning.

Assumption L.1 (Predictable bounded contexts). For every n, $X _ { n }$ is $\mathcal { F } _ { n - 1 }$ -measurable and $\| X _ { n } \| _ { 2 } \leq L$

Assumption L.2 (Realizable reciprocal acceptance). There exists a fixed proposer-specific parameter $\theta ^ { \star } \in \mathbb { R } ^ { d }$ with $\| \theta ^ { \star } \| _ { 2 } \leq S$ such that,for every n and every currently eligible candidate $j \in \mathcal { R } _ { n } ,$

$$
p _ { n } ( j ) = \mathbb { P } ( Y _ { n } ( j ) = 1 \mid \mathcal { F } _ { n - 1 } ) = \sigma ( x _ { j , n } ^ { \top } \theta ^ { \star } ) , \qquad \sigma ( z ) = \frac { 1 } { 1 + e ^ { - z } } .
$$

In particular, because $J _ { n }$ is $\mathcal { F } _ { n - 1 }$ -measurable, $\mathbb { E } [ Y _ { n } \mid { \mathcal { F } } _ { n - 1 } ] = \sigma ( X _ { n } ^ { \top } \theta ^ { \star } )$

The statement is made over the potential outcomes $\{ Y _ { n } ( j ) \} _ { j \in \mathcal { R } _ { \ i } }$ rather than only over the selected action, because the regret analysis compares the chosen candidate with an unselected oracle candidate and therefore requires the model to describe counterfactual arms as well.

Remark $\mathbf { L . l }$ (Assumption L.2 need not hold exactly in our mechanism). Assumption L.2 need not be satisfied by the matching system of Section 3.3. Acceptance there is generated by the threshold rule $Y _ { i  j , t } = 1 \{ U _ { j } ( i ) > b _ { j , t } ^ { R } \}$ , whereas the local context $\boldsymbol { x } _ { j , t } ^ { ( i ) } = \phi ( z _ { i } , z _ { j } )$ of Appendix G is built from static public persona attributes and is therefore constant across attempts for a fixed ordered pair $( i , j )$ . Because the receiver’s benchmark $b _ { j , t } ^ { R }$ moves with its incumbent relationship and is excluded from the proposer’s information set, two histories can share the same proposer-observable dyadic context while inducing different conditional acceptance probabilities. Hence a single fixed $\theta ^ { \star }$ need not represent $p _ { n } ( j )$ across market states. This is a consequence of the information design rather than a modeling oversight: receiver-private and time-varying thresholds are deliberately withheld from proposers.

We nonetheless state Theorem L.1 under Assumption L.2 as a correctly specified benchmark. It isolates the purely statistical difficulty of learning reciprocation when the working model is correct, and it supplies the reference rate against which the misspecification penalty is measured. Theorem L.2, which replaces realizability by the bounded-envelope Assumption L.4, is formulated for the form of dynamic misspecification induced by the simulated mechanism; its sufficient conditions are not claimed to be verified by the simulator.

Assumption L.3 (Bounded positive utility improvement). For every current candidate, $0 ~ <$ $\Delta U _ { n } ( j ) \le D f o r \iota$ afinite constant D. Under utilities normalized to $[ 0 , \dot { 1 } ] ,$ , one may take $D \leq 1$

## Norm-constrained local estimator. Let

$$
\Theta : = \left\{ \theta \in \mathbb { R } ^ { d } : \| \theta \| _ { 2 } \leq S \right\}
$$

be the parameter set implied by the norm bound on $\theta ^ { \star }$ , and define the regularized logistic objective together with its constrained minimizer

$$
\mathcal { L } _ { n } ( \theta ) = \sum _ { m = 1 } ^ { n - 1 } \Big [ \log ( 1 + e ^ { X _ { m } ^ { \top } \theta } ) - Y _ { m } X _ { m } ^ { \top } \theta \Big ] + \frac { \lambda } { 2 } \| \theta \| _ { 2 } ^ { 2 } , \qquad \widehat { \theta } _ { n } : = \arg \operatorname* { m i n } _ { \theta \in \Theta } \mathcal { L } _ { n } ( \theta ) .
$$

Because ${ \mathcal { L } } _ { n }$ is strictly convex for $\lambda > 0$ and Θ is convex and compact, $\widehat { \theta } _ { n }$ exists and is unique. Restricting the estimator to $\Theta$ is what turns the logistic curvature below into a derived quantity determined by the primitive constants $L$ and S, rather than a condition imposed on the realized estimator sequence. Define the matrices

$$
G _ { n } : = \lambda I + \sum _ { m = 1 } ^ { n - 1 } X _ { m } X _ { m } ^ { \top } , \qquad V _ { n } : = \lambda I + \sum _ { m = 1 } ^ { n - 1 } \sigma ^ { \prime } ( X _ { m } ^ { \top } \widehat \theta _ { n } ) X _ { m } X _ { m } ^ { \top } ,
$$

and the practical Fisher width

$$
s _ { n } ( x ) : = { \sqrt { x ^ { \top } V _ { n } ^ { - 1 } x } } .
$$

Lemma L.1 (Uniform local logistic curvature). Suppose Assumption L.1 holds and $\theta ^ { \star } \in \Theta$ , and set

$$
\underline { { \kappa } } : = \sigma ^ { \prime } ( L S ) = \frac { e ^ { - L S } } { ( 1 + e ^ { - L S } ) ^ { 2 } } \in ( 0 , \frac { 1 } { 4 } ] .
$$

Then for every $n ,$ every $m < n ,$ and every $u \in [ 0 , 1 ]$

$$
\sigma ^ { \prime } \Bigl ( X _ { m } ^ { \top } [ \theta ^ { \star } + u ( \widehat { \theta } _ { n } - \theta ^ { \star } ) ] \Bigr ) \geq \underline { { \kappa } } .
$$

Lemma L.1 supplies the curvature needed to convert score concentration into parameter concentration. The mechanism is that $\theta ^ { \star }$ and $\widehat { \theta } _ { n }$ both lie in the convex set $\Theta ,$ so the entire segment joining them lies in $\Theta$ and the logistic predictions entering the mean-value expansion cannot be arbitrarily saturated. Under an unconstrained estimator the analogous statement would have to be assumed directly on the fitted path, because $\lVert \widehat { \theta } _ { n } \rVert _ { 2 }$ is then only controlled by a bound that grows with the sample size. For $\delta \in \bar { ( 0 , 1 ) }$ ), define

$$
\beta _ { n } ( \delta ) : = \frac { 1 } { \underline { { \kappa } } } \left[ \sqrt { \lambda } S + { \frac { 1 } { 2 } } \sqrt { \log \frac { \operatorname * { d e t } G _ { n } } { \lambda ^ { d } } + 2 \log \frac 1 \delta } \right] ,\tag{38}
$$

and the deterministic upper bound

$$
{ \bar { \beta } _ { n } } ( \delta ) : = \frac { 1 } { \underline { { \kappa } } } \left[ \sqrt { \lambda } S + \frac { 1 } { 2 } \sqrt { d \log \left( 1 + \frac { ( n - 1 ) L ^ { 2 } } { \lambda d } \right) + 2 \log \frac { 1 } { \delta } } \right] .\tag{39}
$$

The theoretically calibrated optimistic probability is

$$
p _ { n } ^ { \mathrm { U C B } } ( j ) : = \sigma \Big ( x _ { j , n } ^ { \top } \widehat { \theta } _ { n } + \beta _ { n } ( \delta ) s _ { n } ( x _ { j , n } ) \Big ) .
$$

Definition L.1 (Local proposal pseudo-regret). Let

$$
J _ { n } ^ { \star } \in \arg \operatorname* { m a x } _ { j \in \mathcal { R } _ { n } } \Delta U _ { n } ( j ) p _ { n } ( j ) , \qquad J _ { n } \in \arg \operatorname* { m a x } _ { j \in \mathcal { R } _ { n } } \Delta U _ { n } ( j ) p _ { n } ^ { \mathrm { U C B } } ( j ) .
$$

The instantaneous local pseudo-regret and cumulative pseudo-regret are

$$
r _ { n } : = \Delta U _ { n } ( J _ { n } ^ { \star } ) p _ { n } ( J _ { n } ^ { \star } ) - \Delta U _ { n } ( J _ { n } ) p _ { n } ( J _ { n } ) , \qquad \mathcal { R } _ { N } : = \sum _ { n = 1 } ^ { N } r _ { n } .
$$

Theorem L.1 (R2: Local regret under realizable reciprocal acceptance). Suppose Assumptions $L . l -$ L.3 hold, $\lambda > 0 ,$ , and $\underline { { \kappa } } = \sigma ^ { \prime } ( L S )$ is the curvature constant of Lemma L.1. Then, with probability at least $1 - \delta ,$ , simultaneously for all proposal attempts n,

$$
\lVert \widehat { { \boldsymbol { \theta } } } _ { n } - { \boldsymbol { \theta } } ^ { \star } \rVert _ { G _ { n } } \leq \beta _ { n } ( \delta ) \leq \bar { \beta } _ { n } ( \delta ) ,\tag{40}
$$

and,for every current candidate context x,

$$
\sigma ( x ^ { \top } \theta ^ { \star } ) \leq \sigma \Big ( x ^ { \top } \widehat { \theta } _ { n } + \beta _ { n } ( \delta ) \sqrt { x ^ { \top } V _ { n } ^ { - 1 } x } \Big ) .\tag{41}
$$

Consequently,

$$
\mathcal { R } _ { N } \leq \frac { D \bar { \beta } _ { N } ( \delta ) } { 2 } \sqrt { \frac { N d } { \underline { { \kappa } } } \left( 1 + \frac { L ^ { 2 } } { \lambda } \right) \log \left( 1 + \frac { N L ^ { 2 } } { \lambda d } \right) } .\tag{42}
$$

For fixed $L , S , \lambda , D$ , and κ,

$$
\mathcal { R } _ { N } = \widetilde { \mathcal { O } } ( \frac { d \sqrt { N } } { { \underline { { \kappa } } } ^ { 3 / 2 } } ) , \qquad \frac { \mathcal { R } _ { N } } { N }  0 .
$$

The benchmark in Theorem L.1 is a myopic oracle that faces the same current candidate set and knows the true current reciprocal-acceptance probabilities. The result isolates the local learning loss caused by uncertainty about reciprocation; it does not compare full-market trajectories generated by different policies. As stated in Remark L.1, its realizability hypothesis need not be met by our mechanism, so Theorem L.1 should be read as the correctly specified reference point for the misspecified analysis that follows.

Dynamic misspecification and observable context. The matching system need not obey a stationary logistic law because a receiver’s acceptance threshold changes when its current relationship changes, while the proposer is deliberately denied receiver-private information. We therefore also analyze the logistic model as a working approximation based only on proposer-observable information.

Assumption L.4 (Bounded dynamic logistic misspecification). There exists a fixed reference parameter $\theta ^ { \star }$ with $\| \theta ^ { \star } \| _ { 2 } \leq S$ , i.e. $\theta ^ { \star } \in \Theta$ , and a predictable nonnegative sequence $\{ \eta _ { n } \}$ such that, for every $j \in \mathcal { R } _ { n } ,$

$$
p _ { n } ( j ) = \sigma ( x _ { j , n } ^ { \top } \theta ^ { \star } ) + b _ { n } ( j ) , \qquad | b _ { n } ( j ) | \leq \eta _ { n } ,
$$

with $p _ { n } ( j ) \in [ 0 , 1 ]$

One may augment the public dyadic context with dynamic variables known before the current proposal, such as market stage, the proposer’s accumulated proposal count, or the proposer’s own recent acceptance statistics. Such features remain predictable and can reduce misspecification. They cannot, however, generally guarantee exact realizability under the present information design: two market states may induce the same proposer-observable context but different receiver-private incumbent values and therefore different acceptance thresholds. Hence richer observable context should be interpreted as potentially shrinking the envelope $\eta _ { n } .$ , rather than automatically eliminating it; increasing context dimension also enters the statistical bound through d.

For selected historical actions write $b _ { m } : = b _ { m } ( J _ { m } )$ . Define

$$
\beta _ { n } ^ { \mathrm { m i s } } ( \delta ) : = \frac { 1 } { \underline { { \kappa } } } \left[ \sqrt { \lambda } S + \frac { 1 } { 2 } \sqrt { \log \frac { \operatorname* { d e t } G _ { n } } { \lambda ^ { d } } + 2 \log \frac { 1 } { \delta } } + \sqrt { \sum _ { m = 1 } ^ { n - 1 } \eta _ { m } ^ { 2 } } \right] ,\tag{43}
$$

and

$$
\bar { \beta } _ { n } ^ { \mathrm { m i s } } ( \delta ) : = \frac { 1 } { \underline { { \kappa } } } \left[ \sqrt { \lambda } S + \frac { 1 } { 2 } \sqrt { d \log \left( 1 + \frac { ( n - 1 ) L ^ { 2 } } { \lambda d } \right) + 2 \log \frac { 1 } { \delta } } + \sqrt { \sum _ { m = 1 } ^ { n - 1 } \eta _ { m } ^ { 2 } } \right] .\tag{44}
$$

Let

$$
\bar { p } _ { n } ^ { \mathrm { U C B } } ( j ) : = \sigma \left( x _ { j , n } ^ { \top } \widehat { \theta } _ { n } + \beta _ { n } ^ { \mathrm { m i s } } ( \delta ) s _ { n } ( x _ { j , n } ) \right) ,
$$

and suppose the policy selects

$$
J _ { n } \in \arg \operatorname* { m a x } _ { j \in \mathcal { R } _ { n } } \Delta U _ { n } ( j ) \bar { p } _ { n } ^ { \mathrm { U C B } } ( j ) .
$$

Theorem L.2 (R3: Local regret under bounded dynamic misspecification). Suppose Assumptions $L . l , \ L . 3 ,$ , and $L . 4$ hold, with $\underline { { \kappa } } = \sigma ^ { \prime } ( L S )$ as in Lemma $L . l .$ Assumption L.2 is not required. Then, with probability at least $1 - \delta ,$ , simultaneouslyfor all $n ,$

$$
\| \widehat { \theta } _ { n } - \theta ^ { \star } \| _ { G _ { n } } \leq \beta _ { n } ^ { \mathrm { m i s } } ( \delta ) .
$$

Moreover, local pseudo-regret relative to the oracle that knows the true current probabilities $p _ { n } ( j )$ satisfies

$$
\mathcal { R } _ { N } \leq \frac { D \bar { \beta } _ { N } ^ { \mathrm { m i s } } ( \delta ) } { 2 } \sqrt { \frac { N d } { \underline { { \kappa } } } \left( 1 + \frac { L ^ { 2 } } { \lambda } \right) \log \left( 1 + \frac { N L ^ { 2 } } { \lambda d } \right) } + 2 D \sum _ { n = 1 } ^ { N } \eta _ { n } .\tag{45}
$$

A simple sufficient condition for vanishing average local pseudo-regret is

$$
\sum _ { n = 1 } ^ { N } \eta _ { n } = o \bigg ( \frac { N } { \log N } \bigg ) ,
$$

assuming $0 \leq \eta _ { n } \leq 1$ and the remaining problem constants arefixed.

The misspecification term in Theorem L.2 is not only a feature-engineering residual: under the decentralized information restriction it can also represent state variables intentionally hidden from the proposer. Theorem L.2 is therefore formulated to accommodate the form of dynamic misspecification induced by the mechanism of Section 3.3, and it reduces to Theorem L.1 when $\eta _ { n } \equiv 0$ Verifying the sufficient condition $\begin{array} { r } { \sum _ { n } \eta _ { n } = o ( N / \log N ) } \end{array}$ requires knowledge of the receiver-side state that the design withholds from proposers; characterizing $\eta _ { n }$ endogenously is left to future work.

Bilateral rematching. For any agent $k ,$ define its pre-rematching individual benchmark utility

$$
v _ { k } ( \mu _ { t } ) : = \left\{ { \begin{array} { l l } { r _ { k } , } & { \mu _ { t } ( k ) = \emptyset , } \\ { U _ { k } ( \mu _ { t } ( k ) ) , } & { \mu _ { t } ( k ) \neq \emptyset . } \end{array} } \right.
$$

Proposition L.2 (R4: Strict bilateral improvement of every accepted rematching). Suppose proposer i selects an eligible candidate j at time t and the proposal is accepted. Then the newlyformed pair $( i , j )$ strictly improves the individual partner utility of both consenting agents relative to their pre-proposal benchmarks:

$$
U _ { i } ( j ) > v _ { i } ( \mu _ { t } ) , \qquad U _ { j } ( i ) > v _ { j } ( \mu _ { t } ) .
$$

More precisely,

$$
U _ { i } ( j ) - b _ { i , t } > \epsilon _ { U } \geq 0 , \qquad U _ { j } ( i ) - b _ { j , t } ^ { R } > 0 .
$$

Let ${ \mathcal { P } } ( \mu )$ denote the set of realized cross-side pairs in matching $\mu .$ For consistency with the empirical mutual-welfare metric, define total matched-pair welfare

$$
W _ { \mathrm { t o t } } ( \mu ) : = \sum _ { ( a , b ) \in \mathcal P ( \mu ) } \frac { U _ { a } ( b ) + U _ { b } ( a ) } { 2 } ,
$$

and, when $\mathcal { P } ( \mu ) \neq \emptyset$ , average mutual welfare

$$
W _ { \mathrm { a v g } } ( \mu ) : = \frac { W _ { \mathrm { t o t } } ( \mu ) } { | \mathcal { P } ( \mu ) | } .
$$

Proposition L.3 (R5: Matched-pair welfare need not be monotone). There exists a finite two-sided market and a temporary matching µ<sub>t</sub> such that an acceptedproposal is a strict bilateral improvement for the proposer and receiver as in Proposition L.2, yet the immediate rematching strictly decreases both total matched-pair welfare and average mutual welfare:

$$
W _ { \mathrm { t o t } } ( \mu _ { t + 1 } ) < W _ { \mathrm { t o t } } ( \mu _ { t } ) , \qquad W _ { \mathrm { a v g } } ( \mu _ { t + 1 } ) < W _ { \mathrm { a v g } } ( \mu _ { t } ) .
$$

The reason is a matching externality: the bilateral acceptance test internalizes the preferences of the two consenting agents, but not the losses of partners displaced by the rematching.

Corollary L.1 (Local learning guarantees do not imply global matching optimality). Even if proposer-side local pseudo-regret in Theorem L.1 or Theorem L.2 is sublinear, this alone does not imply monotone matched-pair welfare, terminal optimality of that welfare metric, or convergence to a stable matching.

## L.3 PROOFS

Proof of Proposition L.1.

Proof. Condition on $\mathcal { F } _ { n - 1 }$ . Since $\Delta U _ { n } ( j )$ is known before the current response, for every current candidate j,

$$
\begin{array} { r l } & { \mathbb { E } [ R _ { n } ( j ) \mid \mathcal { F } _ { n - 1 } ] = \mathbb { E } [ Y _ { n } ( j ) \Delta U _ { n } ( j ) \mid \mathcal { F } _ { n - 1 } ] } \\ & { \qquad = \Delta U _ { n } ( j ) \mathbb { E } [ Y _ { n } ( j ) \mid \mathcal { F } _ { n - 1 } ] } \\ & { \qquad = \Delta U _ { n } ( j ) p _ { n } ( j ) . } \end{array}
$$

Therefore any maximizer of $\Delta U _ { n } ( j ) p _ { n } ( j )$ maximizes conditional expected one-step utility improvement. □

Specialization of Abbasi-Yadkori–Pal–Szepesv´ ari.´ The only external concentration result used in the learning proofs is the time-uniform self-normalized inequality of Abbasi-Yadkori et al. (2011), Theorem 1. Their theorem applies to a predictable vector sequence $X _ { m }$ and a conditionally meanzero, R-sub-Gaussian scalar noise sequence $\varepsilon _ { m }$ , with

$$
S _ { t } = \sum _ { m = 1 } ^ { t } \varepsilon _ { m } X _ { m } , \qquad \bar { V } _ { t } = V + \sum _ { m = 1 } ^ { t } X _ { m } X _ { m } ^ { \top } ,
$$

and yields, with probability at least $1 - \delta .$ , simultaneously for all $t ,$

$$
\| S _ { t } \| _ { \bar { V } _ { t } ^ { - 1 } } ^ { 2 } \leq 2 R ^ { 2 } \log \left( \frac { \operatorname * { d e t } ( \bar { V } _ { t } ) ^ { 1 / 2 } } { \operatorname * { d e t } ( V ) ^ { 1 / 2 } \delta } \right) .
$$

Under Assumption L.2, set

$$
\varepsilon _ { m } : = Y _ { m } - \sigma ( X _ { m } ^ { \top } \theta ^ { \star } ) , \qquad V = \lambda I , \qquad t = n - 1 .
$$

Then $\mathbb { E } [ \varepsilon _ { m } ~ \vert ~ { \mathcal F } _ { m - 1 } ] = 0$ . Conditional on ${ { \mathcal { F } } _ { m - 1 } } , ~ { { Y } _ { m } } ~ \in ~ \{ 0 , 1 \}$ , so Hoeffding’s lemma implies that the centered Bernoulli noise $\varepsilon _ { m }$ is conditionally 1/2-sub-Gaussian. Moreover $\bar { V } _ { n - 1 } = \mathbf { \bar { \cal G } } _ { n }$ Therefore, with probability at least $1 - \delta$ , simultaneously for every $n _ { \colon }$

$$
\left\| \sum _ { m = 1 } ^ { n - 1 } \varepsilon _ { m } X _ { m } \right\| _ { G _ { n } ^ { - 1 } } \leq { \frac { 1 } { 2 } } { \sqrt { \log { \frac { \operatorname* { d e t } G _ { n } } { \lambda ^ { d } } } + 2 \log { \frac { 1 } { \delta } } } } .\tag{46}
$$

The adaptive proposal policy is compatible with this result because the theorem requires predictability rather than i.i.d. contexts: $X _ { n }$ may depend on the entire observed history and current candidate set as long as it is chosen before $Y _ { n }$ is observed.

## Proof of Lemma L.1.

Proof. Fix $n , m < n$ , and $u \in [ 0 , 1 ]$ , and set $\theta _ { u } : = \theta ^ { \star } + u ( \widehat \theta _ { n } - \theta ^ { \star } )$ . Both $\theta ^ { \star }$ and $\widehat { \theta } _ { n }$ belong to $\Theta ,$ which is convex, so $\theta _ { u } \in \Theta$ and $\| \theta _ { u } \| _ { 2 } \leq S$ . By Cauchy–Schwarz and Assumption L.1,

$$
\begin{array} { r } { \left| \boldsymbol { X } _ { m } ^ { \top } \boldsymbol { \theta } _ { u } \right| \leq \| \boldsymbol { X } _ { m } \| _ { 2 } \| \boldsymbol { \theta } _ { u } \| _ { 2 } \leq L S . } \end{array}
$$

The derivative $\sigma ^ { \prime } ( z ) = \sigma ( z ) ( 1 - \sigma ( z ) )$ is even and strictly decreasing in $| z | , \mathrm { s o } \ | z | \le L S$ implies $\sigma ^ { \prime } ( z ) \geq \sigma ^ { \prime } ( L S ) \overset { \cdot } { = } \overset { \kappa } { \underline { { \kappa } } }$ . Finally $\underline { { \kappa } } \le \sigma ^ { \prime } ( 0 ) = 1 / 4$ , and $\kappa > 0$ because $L$ and S are finite. □

Constrained logistic estimation error. Because ${ \mathcal { L } } _ { n }$ is convex and differentiable and Θ is convex, the constrained minimizer satisfies the first-order variational inequality

$$
\left. \nabla { \mathcal { L } } _ { n } ( { \widehat { \theta } } _ { n } ) , \theta - { \widehat { \theta } } _ { n } \right. \geq 0 \qquad { \mathrm { f o r ~ e v e r y ~ } } \theta \in \Theta .
$$

Since $\theta ^ { \star } \in \Theta$ , choosing $\theta = \theta ^ { \star }$ gives

$$
\begin{array} { r l r } { \Big \langle \nabla \mathcal { L } _ { n } ( \widehat { \theta } _ { n } ) , e _ { n } \Big \rangle \leq 0 , } & { { } } & { e _ { n } : = \widehat { \theta } _ { n } - \theta ^ { \star } . } \end{array}\tag{47}
$$

When the norm constraint is inactive, Equation $^ { 4 7 }$ holds with equality and reduces to the usual unconstrained score equation, so the argument below covers both cases. Set

$$
\varepsilon _ { m } : = Y _ { m } - \sigma ( X _ { m } ^ { \top } \theta ^ { \star } ) , \qquad S _ { n } : = \sum _ { m = 1 } ^ { n - 1 } \varepsilon _ { m } X _ { m } ,
$$

and substitute $Y _ { m } = \sigma ( X _ { m } ^ { \top } \theta ^ { \star } ) + \varepsilon _ { m }$ into the gradient

$$
\nabla \mathcal L _ { n } ( \widehat \theta _ { n } ) = \sum _ { m = 1 } ^ { n - 1 } [ \sigma ( X _ { m } ^ { \top } \widehat \theta _ { n } ) - Y _ { m } ] X _ { m } + \lambda \widehat \theta _ { n } ~
$$

to obtain

$$
\nabla \mathcal { L } _ { n } ( \widehat { \theta } _ { n } ) = \sum _ { m = 1 } ^ { n - 1 } [ \sigma ( X _ { m } ^ { \top } \widehat { \theta } _ { n } ) - \sigma ( X _ { m } ^ { \top } \theta ^ { \star } ) ] X _ { m } + \lambda e _ { n } - ( S _ { n } - \lambda \theta ^ { \star } ) .\tag{48}
$$

For each $m < n$ , the scalar mean-value theorem gives a point $\xi _ { m , n }$ between $X _ { m } ^ { \top } \theta ^ { \star }$ and $X _ { m } ^ { \top } \widehat { \theta } _ { n }$ such that

$$
\sigma ( X _ { m } ^ { \top } \widehat { \theta } _ { n } ) - \sigma ( X _ { m } ^ { \top } \theta ^ { \star } ) = \sigma ^ { \prime } ( \xi _ { m , n } ) X _ { m } ^ { \top } e _ { n } .
$$

Thus $\nabla \mathcal { L } _ { n } ( \widehat { \theta } _ { n } ) = H _ { n } \boldsymbol { e } _ { n } - ( S _ { n } - \lambda \theta ^ { \star } )$ , where

$$
H _ { n } : = \lambda I + \sum _ { m = 1 } ^ { n - 1 } \sigma ^ { \prime } ( \xi _ { m , n } ) X _ { m } X _ { m } ^ { \top } ,
$$

and Equation 47 becomes

$$
e _ { n } ^ { \top } H _ { n } e _ { n } \leq e _ { n } ^ { \top } ( S _ { n } - \lambda \theta ^ { \star } ) .
$$

Each $\xi _ { m , n }$ equals $X _ { m } ^ { \top } \theta _ { u }$ for some $u \in [ 0 , 1 ]$ , because the map $u \mapsto X _ { m } ^ { \top } [ \theta ^ { \star } + u ( \widehat \theta _ { n } - \theta ^ { \star } ) ]$ is affine and traverses exactly the interval between the two endpoints. Lemma L.1 therefore implies

$$
H _ { n } \succeq \lambda I + \underline { { { \kappa } } } \sum _ { m = 1 } ^ { n - 1 } X _ { m } X _ { m } ^ { \top } \succeq \underline { { { \kappa } } } G _ { n } ,
$$

where the final inequality uses $0 < \underline { { \kappa } } \le 1 / 4 < 1$ . Hence

$$
\begin{array} { r l } & { \underline { { \kappa } } \| e _ { n } \| _ { G _ { n } } ^ { 2 } \leq e _ { n } ^ { \top } H _ { n } e _ { n } } \\ & { \qquad \leq e _ { n } ^ { \top } ( S _ { n } - \lambda \theta ^ { \star } ) } \\ & { \qquad \leq \| e _ { n } \| _ { G _ { n } } \left( \| S _ { n } \| _ { G _ { n } ^ { - 1 } } + \lambda \| \theta ^ { \star } \| _ { G _ { n } ^ { - 1 } } \right) . } \end{array}
$$

Since $G _ { n } \succeq \lambda I ,$

$$
\lambda \| \theta ^ { \star } \| _ { G _ { n } ^ { - 1 } } \leq \sqrt { \lambda } \| \theta ^ { \star } \| _ { 2 } \leq \sqrt { \lambda } S .
$$

Combining this with Equation 46 proves the first inequality in Equation 40. The determinant bound

$$
\log { \frac { \operatorname* { d e t } G _ { n } } { \lambda ^ { d } } } \leq d \log \left( 1 + { \frac { ( n - 1 ) L ^ { 2 } } { \lambda d } } \right)
$$

follows from the arithmetic–geometric mean inequality applied to the eigenvalues and yields $\beta _ { n } \leq$ $\bar { \beta } _ { n }$

At the fitted endpoint, Lemma L.1 applied with $u = 1$ and the global bound $\sigma ^ { \prime } ( z ) \leq 1 / 4 \leq 1$ give

$$
\underline { { \kappa } } G _ { n } \preceq V _ { n } \preceq G _ { n } .\tag{49}
$$

After inversion,

$$
G _ { n } ^ { - 1 } \preceq V _ { n } ^ { - 1 } \preceq \frac { 1 } { \underline { { \kappa } } } G _ { n } ^ { - 1 } .
$$

For every candidate context x,

$$
\begin{array} { r l } & { | x ^ { \top } ( \widehat \theta _ { n } - \theta ^ { \star } ) | \leq \| \widehat \theta _ { n } - \theta ^ { \star } \| _ { G _ { n } } \| x \| _ { G _ { n } ^ { - 1 } } } \\ & { \qquad \leq \beta _ { n } ( \delta ) \| x \| _ { V _ { n } ^ { - 1 } } . } \end{array}
$$

Therefore

$$
x ^ { \top } \theta ^ { \star } \leq x ^ { \top } \widehat { \theta } _ { n } + \beta _ { n } ( \delta ) \sqrt { x ^ { \top } V _ { n } ^ { - 1 } x } ,
$$

and monotonicity of the sigmoid proves Equation 41.

Proof of Theorem L.1. On the confidence event, valid optimism and UCB maximization imply

$$
\Delta U _ { n } ( J _ { n } ^ { \star } ) p _ { n } ( J _ { n } ^ { \star } ) \leq \Delta U _ { n } ( J _ { n } ^ { \star } ) p _ { n } ^ { \mathrm { U C B } } ( J _ { n } ^ { \star } ) \leq \Delta U _ { n } ( J _ { n } ) p _ { n } ^ { \mathrm { U C B } } ( J _ { n } ) .
$$

Hence

$$
r _ { n } \leq \Delta U _ { n } ( J _ { n } ) [ p _ { n } ^ { \mathrm { U C B } } ( J _ { n } ) - p _ { n } ( J _ { n } ) ] .\tag{50}
$$

Let $s _ { n } : = { \sqrt { X _ { n } ^ { \top } V _ { n } ^ { - 1 } X _ { n } } }$ . The confidence event gives

$$
\begin{array} { r } { | X _ { n } ^ { \top } ( \widehat { \theta } _ { n } - \theta ^ { \star } ) | \leq \beta _ { n } s _ { n } . } \end{array}
$$

Thus the optimistic logit exceeds the true logit by at most $2 \beta _ { n } s _ { n }$ . Since sup<sub>z</sub> $\sigma ^ { \prime } ( z ) = 1 / 4$ , the sigmoid is globally 1/4-Lipschitz and

$$
0 \leq p _ { n } ^ { \mathrm { U C B } } ( J _ { n } ) - p _ { n } ( J _ { n } ) \leq \frac { \beta _ { n } } { 2 } s _ { n } .
$$

Using $\Delta U _ { n } ( J _ { n } ) \leq D$ in Equation 50,

$$
r _ { n } \leq { \frac { D \beta _ { n } } { 2 } } s _ { n } .\tag{51}
$$

From Equation 49,

$$
s _ { n } ^ { 2 } \leq { \frac { 1 } { \underline { { \kappa } } } } X _ { n } ^ { \top } G _ { n } ^ { - 1 } X _ { n } .
$$

Set $q _ { n } : = X _ { n } ^ { \top } G _ { n } ^ { - 1 } X _ { n }$ . Since $G _ { n } \succeq \lambda I , 0 \leq q _ { n } \leq L ^ { 2 } / \lambda$ . By the matrix determinant lemma,

$$
\log \frac { \operatorname * { d e t } G _ { N + 1 } } { \operatorname * { d e t } G _ { 1 } } = \sum _ { n = 1 } ^ { N } \log ( 1 + q _ { n } ) .
$$

For $0 \leq q \leq L ^ { 2 } / \lambda ,$

$$
\log ( 1 + q ) \geq \frac { q } { 1 + q } \geq \frac { q } { 1 + L ^ { 2 } / \lambda } .
$$

Therefore

$$
\sum _ { n = 1 } ^ { N } s _ { n } ^ { 2 } \leq \frac { d } { \underline { { { \kappa } } } } \left( 1 + \frac { L ^ { 2 } } { \lambda } \right) \log \left( 1 + \frac { N L ^ { 2 } } { \lambda d } \right) .
$$

Cauchy–Schwarz yields

$$
\sum _ { n = 1 } ^ { N } s _ { n } \leq { \sqrt { { \frac { N d } { \underline { { \kappa } } } } \left( 1 + { \frac { L ^ { 2 } } { \lambda } } \right) \log \left( 1 + { \frac { N L ^ { 2 } } { \lambda d } } \right) } } .\tag{52}
$$

Summing Equation 51, using monotonicity of $\bar { \beta } _ { n }$ , and applying Equation 52 proves Equation 42.

Proof of Theorem L.2. Under Assumption L.4, for each selected historical action define

$$
p _ { m } : = \mathbb { E } [ Y _ { m } \mid { \mathcal { F } } _ { m - 1 } ] , \qquad \xi _ { m } : = Y _ { m } - p _ { m } .
$$

Then $\mathbb { E } [ \xi _ { m } \mid { \mathcal { F } } _ { m - 1 } ] = 0$ , and $\xi _ { m }$ is conditionally $1 / 2 \cdot$ -sub-Gaussian. Also

$$
\begin{array} { r } { Y _ { m } = \sigma ( X _ { m } ^ { \top } \theta ^ { \star } ) + b _ { m } + \xi _ { m } . } \end{array}
$$

Assumption L.4 gives $\theta ^ { \star } \in \Theta .$ , so the variational inequality of Equation 47 applies verbatim. Repeating the mean-value expansion with $Y _ { m }$ decomposed as above gives

$$
\nabla \mathcal { L } _ { n } ( \widehat { \theta } _ { n } ) = H _ { n } \boldsymbol { e } _ { n } - \left( \sum _ { m = 1 } ^ { n - 1 } \xi _ { m } X _ { m } + \sum _ { m = 1 } ^ { n - 1 } b _ { m } X _ { m } - \lambda \theta ^ { \star } \right) ,
$$

and therefore

$$
e _ { n } ^ { \top } H _ { n } e _ { n } \leq e _ { n } ^ { \top } \left( \sum _ { m = 1 } ^ { n - 1 } \xi _ { m } X _ { m } + \sum _ { m = 1 } ^ { n - 1 } b _ { m } X _ { m } - \lambda \theta ^ { \star } \right) .
$$

The martingale term obeys the same specialized self-normalized inequality above. For the bias term, let X be the matrix whose mth row is $X _ { m } ^ { \top }$ and let $\boldsymbol { b } = ( b _ { 1 } , \dots , b _ { n - 1 } ) ^ { \top }$ . Then

$$
\sum _ { m = 1 } ^ { n - 1 } b _ { m } X _ { m } = \mathbf { X } ^ { \top } b , \qquad G _ { n } = \lambda I + \mathbf { X } ^ { \top } \mathbf { X } .
$$

The eigenvalues of $\mathbf { X } G _ { n } ^ { - 1 } \mathbf { X } ^ { \top }$ are $s _ { k } ^ { 2 } / ( \lambda + s _ { k } ^ { 2 } ) \leq 1$ , so

$$
\begin{array} { r l } {  { \| \mathbf { X } ^ { \top } b \| _ { G _ { n } ^ { - 1 } } ^ { 2 } = b ^ { \top } \mathbf { X } G _ { n } ^ { - 1 } \mathbf { X } ^ { \top } b } } \\ & { \leq \| b \| _ { 2 } ^ { 2 } \leq \displaystyle \sum _ { m = 1 } ^ { n - 1 } \eta _ { m } ^ { 2 } . } \end{array}
$$

The same curvature argument based on Lemma L.1 therefore gives

$$
\| \widehat { \theta } _ { n } - \theta ^ { \star } \| _ { G _ { n } } \leq \beta _ { n } ^ { \mathrm { m i s } } ( \delta ) .
$$

Now define the logistic reference $\bar { p } _ { n } ( j ) : = \sigma ( x _ { j , n } ^ { \top } \theta ^ { \star } )$ and its UCB

$$
u _ { n } ( j ) : = \sigma \big ( x _ { j , n } ^ { \top } \widehat { \theta } _ { n } + \beta _ { n } ^ { \mathrm { m i s } } s _ { n } ( x _ { j , n } ) \big ) .
$$

The confidence argument gives $\bar { p } _ { n } ( j ) \leq u _ { n } ( j )$ , whereas misspecification gives $p _ { n } ( j ) \leq \bar { p } _ { n } ( j ) + \eta _ { n }$ For the true oracle action $\bar { J } _ { n } ^ { \star }$

$$
\begin{array} { r } { \Delta U _ { n } ( J _ { n } ^ { \star } ) p _ { n } ( J _ { n } ^ { \star } ) \leq \Delta U _ { n } ( J _ { n } ^ { \star } ) u _ { n } ( J _ { n } ^ { \star } ) + D \eta _ { n } } \\ { \leq \Delta U _ { n } ( J _ { n } ) u _ { n } ( J _ { n } ) + D \eta _ { n } . } \end{array}
$$

Subtracting the true expected gain of the selected action gives

$$
\begin{array} { r l } & { r _ { n } \leq \Delta U _ { n } ( J _ { n } ) [ u _ { n } ( J _ { n } ) - p _ { n } ( J _ { n } ) ] + D \eta _ { n } } \\ & { \quad \leq \Delta U _ { n } ( J _ { n } ) [ u _ { n } ( J _ { n } ) - \bar { p } _ { n } ( J _ { n } ) ] + 2 D \eta _ { n } . } \end{array}
$$

The same $1 / 4 \ – \mathrm { I }$ ipschitz argument yields

$$
u _ { n } ( J _ { n } ) - \bar { p } _ { n } ( J _ { n } ) \leq \frac { \beta _ { n } ^ { \mathrm { m i s } } } { 2 } s _ { n } ,
$$

so

$$
r _ { n } \leq \frac { D \beta _ { n } ^ { \mathrm { m i s } } } { 2 } s _ { n } + 2 D \eta _ { n } .
$$

Summing and applying Equation 52 proves Equation 45.

For the stated sufficient condition, if $0 \leq \eta _ { n } \leq 1$ and $\begin{array} { r } { \sum _ { n = 1 } ^ { N } \eta _ { n } = o ( N / \log N ) } \end{array}$ , then

$$
\sum _ { n = 1 } ^ { N } \eta _ { n } ^ { 2 } \le \sum _ { n = 1 } ^ { N } \eta _ { n } = o ( N / \log N ) .
$$

Hence $\sqrt { \textstyle \sum _ { n = 1 } ^ { N } \eta _ { n } ^ { 2 } } = o ( \sqrt { N / \log N } )$ . Its contribution through $\bar { \beta } _ { N } ^ { \mathrm { m i s } }$ times the width sum is $o ( N )$ the explicit $\sum _ { n } \eta _ { n }$ term is also $o ( N )$ , while the ordinary statistical term is sublinear for fixed dimension and constants. Therefore $\mathcal { R } _ { N } / N  0$

## Proof of Proposition L.2.

Proof. Because j belongs to the proposer’s eligible set,

$$
\Delta U _ { i j , t } = U _ { i } ( j ) - b _ { i , t } > \epsilon _ { U } \geq 0 ,
$$

so $U _ { i } ( j ) > b _ { i , t }$ . If i is single, $b _ { i , t } = r _ { i } = v _ { i } ( \mu _ { t } )$ . If i is currently matched,

$$
b _ { i , t } = \operatorname* { m a x } \{ r _ { i } , U _ { i } ( \mu _ { t } ( i ) ) \} \geq U _ { i } ( \mu _ { t } ( i ) ) = v _ { i } ( \mu _ { t } ) .
$$

Thus $U _ { i } ( j ) > v _ { i } ( \mu _ { t } )$ . Acceptance by $j$ means $U _ { j } ( i ) > b _ { j , t } ^ { R }$ . If j is single, $b _ { j , t } ^ { R } = r _ { j } = v _ { j } ( \mu _ { t } )$ ; if j is matched,

$$
b _ { j , t } ^ { R } = \operatorname* { m a x } \{ r _ { j } , U _ { j } ( \mu _ { t } ( j ) ) \} \geq U _ { j } ( \mu _ { t } ( j ) ) = v _ { j } ( \mu _ { t } ) .
$$

Hence $U _ { j } ( i ) > v _ { j } ( \mu _ { t } )$ as well.

## Proof of Proposition L.3.

Proof. Consider $M = \{ m _ { 1 } , m _ { 2 } \}$ and $W = \{ w _ { 1 } , w _ { 2 } \}$ , with reservation utilities below 0.60 and all current relationships temporary. Let

$$
\mu _ { t } = \{ ( m _ { 1 } , w _ { 1 } ) , ( m _ { 2 } , w _ { 2 } ) \} ,
$$

and choose directional utilities

$$
U _ { m _ { 1 } } ( w _ { 1 } ) = 0 . 6 0 , \quad U _ { w _ { 1 } } ( m _ { 1 } ) = 0 . 9 9 , \quad U _ { m _ { 2 } } ( w _ { 2 } ) = 0 . 9 9 , \quad U _ { w _ { 2 } } ( m _ { 2 } ) = 0 . 6 0 ,
$$

with cross-pair values

$$
U _ { m _ { 1 } } ( w _ { 2 } ) = 0 . 6 5 , \qquad U _ { w _ { 2 } } ( m _ { 1 } ) = 0 . 6 5 .
$$

Take $\epsilon _ { U } < 0 . 0 5$ . Then $m _ { 1 }$ strictly prefers $w _ { 2 }$ to its current benchmark and $w _ { 2 }$ accepts $m _ { 1 }$ , so the proposal is a strict bilateral improvement for the consenting agents. Before the proposal,

$$
W _ { \mathrm { t o t } } ( \mu _ { t } ) = \frac { 0 . 6 0 + 0 . 9 9 } { 2 } + \frac { 0 . 9 9 + 0 . 6 0 } { 2 } = 1 . 5 9 , \qquad W _ { \mathrm { a v g } } ( \mu _ { t } ) = 0 . 7 9 5 .
$$

After acceptance, the two old temporary relationships are dissolved, $w _ { 1 }$ and $m _ { 2 }$ are displaced, and immediately after rematching

$$
\mu _ { t + 1 } = \{ ( m _ { 1 } , w _ { 2 } ) \} .
$$

Hence

$$
W _ { \mathrm { t o t } } ( \mu _ { t + 1 } ) = W _ { \mathrm { a v g } } ( \mu _ { t + 1 } ) = { \frac { 0 . 6 5 + 0 . 6 5 } { 2 } } = 0 . 6 5 .
$$

Both total matched-pair welfare and average mutual welfare therefore decrease. The decline is caused by the losses of the displaced partners, which are not part of the bilateral acceptance test.

## Proof of Corollary L.1.

Proof. The local regret theorems control only the loss in one-step expected proposer improvement relative to a myopic acceptance-probability oracle. Proposition L.3 constructs a feasible accepted rematching in which both consenting agents improve while matched-pair welfare decreases. Therefore local decision quality, even if asymptotically no-regret, does not by itself imply monotonicity or terminal optimality of the matched-pair welfare metric. Stability is likewise not implied because the local regret benchmark contains no stability constraint. □

## M FULL DECENTRALIZED MATCHING PROCEDURE

Algorithm 1 expands the period-level procedure summarized in Section 3.4. It makes explicit the order of local exposure, valuation, proposal, feedback, model updating, relationship revision, and stochastic commitment.

The procedure preserves three distinct forms of locality. Candidate discovery is local through $\mathcal { C } _ { t } ( i )$ ; LLM valuation is triggered only when a candidate becomes locally relevant; and reciprocalacceptance learning is agent-specific because only proposals made by i enter $\mathcal { H } _ { i , t }$ . No global acceptance model, shared interaction history, or market-wide preference matrix is available to the proposed agents.

## N LIMITATIONS AND FUTURE DIRECTIONS

## N.1 LIMITATIONS

The empirical grounding of this study is intentionally specific to the Chinese marriage-market setting represented by the 2015 population sample and the choice experiment of Zhou et al. (2023). Accordingly, the validation establishes alignment with this empirical reference rather than universal behavioral validity across cultures, cohorts, or institutional settings. Extending the framework to other populations would require corresponding domain-specific demographic construction and behavioral validation.

The simulated market also remains deliberately stylized. The main experiments use a $5 0 \times 5 0$ population with random local exposure, a common reservation-utility specification, and probabilistic commitment. Richer interaction structures—including social networks, endogenous meeting processes, heterogeneous or adaptive reservation values, noisy or private attributes, and evolving outside options—may alter the resulting matching dynamics. These components can be incorporated within the same decentralized framework, but are outside the scope of the present experiments.

The theoretical results in Appendix L characterize local proposal-level learning and bilateral rematching; they do not guarantee terminal welfare optimality or convergence to a stable matching.

Algorithm 1 Decentralized LLM-Agent Matching with Local Logistic-UCB   
Require: Agent sets M, W; horizon $T ;$ proposal budget $L ;$ reservation utilities $\{ r _ { i } \}$ ; regularization   
$\{ \lambda _ { i } \}$ ; exploration coefficients $\{ \beta _ { i , t } \}$ ; utility threshold $\epsilon _ { U } ;$ commitment probabilities $\{ \rho _ { t } \}$   
1: Initialize all agents as single; set local histories empty and initialize local Logistic-UCB models.   
2: for $t = 1 , \dots , T$ do   
3: $\mathcal { A } _ { t } ^ { \mathrm { a c t } }  \{ i \in \mathcal { A }$ : i is not permanently matched}   
4: $\mathbf { i f } \lambda _ { t } ^ { \mathrm { a c t } } = \varnothing$ then   
5: break   
6: end if   
7: for $a = 1 , \dots , | \mathcal { A } _ { t } ^ { \mathrm { a c t } } |$ do   
8: Sample $i \sim$ Uniform $( A _ { t } ^ { \mathrm { a c t } } )$ with replacement   
9: Sample local exposure set $\mathcal { \dot { C } } _ { t } ( i ) \subseteq \dot { \mathcal { P } } _ { t } ( i )$   
10: Obtain $U _ { i } ( j )$ for all $j \in \mathcal { C } _ { t } ( i )$ and construct $\mathcal { E } _ { t } ( i )$ using Equation 3   
11: $\mathcal { R } _ { t }  \mathcal { E } _ { t } ( \ddot { i } )$   
12: for $q = 1 , \ldots , L$ do   
13: if $\mathcal { R } _ { t } = \mathcal { O }$ then   
14: break   
15: end if   
16: For each $j \in \mathcal { R } _ { t } ,$ , construct $\boldsymbol { x } _ { j , t } ^ { ( i ) }$ and compute $I _ { i j , t }$ using Equations 7 and 8   
17: $j ^ { * } \gets \arg \operatorname* { m a x } _ { j \in \mathcal { R } _ { t } } I _ { i j , t }$   
18: Obtain $U _ { j ^ { * } } ( i )$ and compute $y ^ { * } \gets Y _ { i \to j ^ { * } , t }$ using Equation 5   
19: $\mathcal { H } _ { i , t }  \mathcal { H } _ { i , t } \cup \{ ( x _ { j ^ { * } , t } ^ { ( i ) } , y ^ { * } ) \}$   
20: Refit $\widehat { \theta } _ { i , t }$ by Equation 20 and update $V _ { i , t }$ by Equation 23   
21: if $y ^ { * } = 1$ then   
22: Dissolve temporary matches involving i or $j ^ { * }$ and return displaced partners to   
active search   
23: Form $( i , j ^ { * } )$ as a new temporary pair   
24: break   
25: else   
26: $\mathcal { R } _ { t }  \mathcal { R } _ { t } \setminus \{ j ^ { * } \}$   
27: end if   
28: end for   
29: end for   
30: Let $\mathcal { T } _ { t }$ denote temporary pairs after all activations   
31: for all $( i , j ) \in \mathcal { T } _ { t }$ in parallel do   
32: Sample $Z _ { i j , t } \sim$ Bernoulli $\left( \rho _ { t } \right)$   
33: if $\bar { Z _ { i j , t } } = \check { 1 }$ then   
34: Mark (i, j) as permanent   
35: end if   
36: end for   
37: end for   
38: return final matching µ

Establishing endogenous bounds on dynamic misspecification and conditions linking local learning to market-level outcomes remains an open direction. Moreover, learned acceptance coefficients represent predictive associations rather than structural or causal preferences, because proposal observations are endogenously selected by the matching policy.

Finally, LLM-based behavioral modeling remains sensitive to persona construction, prompt design, model choice, and demographic representation (Anthis et al., 2025; Zhou et al., 2026; Ludwig et al., 2024). Our cross-backbone and empirical-reference validation reduces dependence on any single model realization, but does not eliminate these broader sources of behavioral-model uncertainty.

## N.2 FUTURE DIRECTIONS

The framework naturally extends to other decentralized economic settings in which agents evaluate counterparties while learning whether interaction will be reciprocated, including labor-market search, hiring, team formation, housing, platform matching, and repeated buyer–seller or lender– borrower interactions. The same separation between domain-grounded behavioral valuation and local reciprocal learning can be retained while replacing personas, actions, institutions, and validation targets with domain-specific counterparts.

A second direction is to enrich the interaction structure through social networks, geography, recommender systems, endogenous communication, longer memories, adaptive reservation values, or information sharing. On the learning side, nonstationary bandits, hierarchical priors, and theoretical conditions connecting local learning to welfare or stability are natural extensions. Beyond economics, the architecture can support persistent multi-agent societies in games, where NPCs form and revise friendships, rivalries, coalitions, or trading relationships through local interaction rather than a globally scripted social graph.