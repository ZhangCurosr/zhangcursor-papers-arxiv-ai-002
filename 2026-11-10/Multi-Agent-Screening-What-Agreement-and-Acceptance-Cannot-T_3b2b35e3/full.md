# Multi-Agent Screening: What Agreement and Acceptance Cannot Tell You

<sub>Benjamin</sub> <sub>Chang</sub>1[0009-0006-0332-0319] <sub>and</sub>

<sub>Dell</sub> <sub>Zhang</sub>2⋆[0000-0002-8774-3725]

<sup>1</sup> University of Cambridge, Cambridge, UK

btyc3@cam.ac.uk

<sup>2</sup> XSci, London, UK

dell.z@ieee.org

Abstract. When AI agents screen items for a costly human assessor, as in agentic search, e-discovery and AI reviewing of papers, only what they pass gets judged. We show what each kind of evidence can establish about their errors, even with unlimited data. Agreement shows that agents concur, not that they are right, however many there are. Acceptance by the assessor gives precision but only a lower bound on recall. Knowing which agents found which relevant items gives an upper bound on recall if agents’ finds are not negatively correlated, and caps what one more agent can add if agents are interchangeable. Without assumptions about the agents, recall can be pinned down only by judging items they did not pass: a random sample of these estimates how many relevant items were missed, to any chosen accuracy, and the sample needed does not grow with the number of agents.

Keywords: Multi-Agent Systems · Agentic Retrieval · Incomplete Judgments · Recall Estimation · LLM-as-a-Judge

## 1 Introduction

Retrieval is becoming a multi-agent activity: agents rewrite queries, retrieve and rerank candidates, judge relevance [10,19] and review papers alongside or ahead of human experts [6,5]. These systems share a business problem: agents are cheap, but the assessor at the end, whether a relevance assessor, an expert or a lawyer, is costly, and so sees only what the agents pass. This is the incomplete-judgment setting of pooled evaluation [7,23] and e-discovery [20,16].

Operators face four questions. Can we trust the agents when they agree? Are we missing relevant items? Is one more agent worth paying for? How much assessment budget must we keep for checking the system? We ask what the system’s own evidence can answer even with unlimited data, so the limits we find are not sampling error. Using classical tools adapted to multi-agent systems [21], we give a recall interval, a bound on what the next agent can add, and rules for spending on agents and audits, with AI reviewing of papers as a worked example.

![](images/21eecf9b8bf0d941a14df84df9eb349f200adbfea3da8b295325a6c803dc00e5.jpg)  
Fig. 1. A multi-agent screening system. Grey notes: what each kind of evidence shows.

## 2 Multi-Agent Screening

A multi-agent screening system (Figure 1) processes a fixed collection of items, such as documents, code changes or papers. Each of k agents returns an output $O _ { i } ,$ , and $O = ( O _ { 1 } , \ldots , O _ { k } )$ . A rule $A = \pi ( O )$ retrieves items with $A = 1$ to the pool, which is then submitted to the assessor. An item is relevant, $Y = 1$ if the assessor would accept it under a fixed standard (as in Cranfield-style evaluation), and $Y = 0$ otherwise; we observe $Y$ only in the pool. Note that only the assessor accepts: agents merely pass items on. Probabilities refer to an item drawn uniformly from the collection: $a = \mathbb { P } ( A = 1 )$ is the share of the collection in the pool, $p = \mathbb { P } ( Y = 1 \mid A = 1 )$ the acceptance rate in the pool (which is its precision), and $H = \mathbb { P } ( Y = 1 , A = 0 )$ the proportion of the collection that is relevant but unjudged. So recall is $r = a p / ( a p + H )$ , and H is the unknown.

We consider four kinds of evidence: agreement, the joint distribution of $O ;$ acceptance, the assessor’s accept-or-reject judgments on the pool; overlap, those judgments split by who nominated each item; and audit, judgments on a random sample from outside the pool (an elusion test in e-discovery). We give tight intervals: the values that fit the data and our assumptions, and no others.

## 3 What Each Kind of Evidence Can Tell Us

Agreement: consensus, not correctness. Agents built on similar models err together [14]. Model this as a shared representation: a latent $Z$ with Y and O conditionally independent given $Z ,$ so outputs reflect relevance only through Z. For a rule $A = \pi ( O )$ , let $g ( z ) = \mathbb { P } ( A = 1 \mid Z = z )$ and $\eta ( z ) = \mathbb { P } ( Y = 1 \mid Z = z )$

Theorem 1 (Agreement). For every k and every rule $\pi \colon \mathrm { ( a ) }$ the error of $\pi , \ \mathbb { P } ( A \neq Y )$ , is at least $\mathbb { E } \operatorname* { m i n } \{ \eta ( Z ) , 1 - \eta ( Z ) \} ; ~ ( \mathrm { b } )$ the distribution of O does not depend on η, and as η varies the accuracy of π takes every value in [E min $\{ g ( Z ) , 1 - g ( Z ) \}$ , E max $\{ g ( Z ) , 1 - g ( Z ) \} ]$ , which is [0, 1] if A is a function of Z.

Proof. Fix z and write g, η for $g ( z ) , \eta ( z )$ . Given $Z = z , A$ and Y are independent, so $A = Y$ with probability $g \eta + ( 1 - g ) ( 1 - \eta ) \leq \operatorname* { m a x } \{ \eta , 1 - \eta \}$ , and $A \neq Y$ with probability at least min $\{ \eta , 1 - \eta \}$ ; average over $Z .$ The distribution of $( Z , O )$ and so $^ { g , }$ is the same for every $\eta ( \cdot )$ , and the accuracy given $Z = z$ runs linearly from $1 - g$ at $\eta = 0$ to $g$ at $\eta = 1 \colon$ pick $\eta ( z ) \in \{ 0 , 1 \}$ for each end, and mix. ⊓⊔

For a representation that stays fixed as agents are added, (a) is an error floor at every $k ,$ positive unless $Z$ determines $\bar { Y } .$ , and (b) says no statistic of the outputs can tell a right consensus from a wrong one. Label-free methods such as Dawid–Skene identify error rates against a discrete $Z ,$ up to relabelling, under conditional independence and rank conditions [9,2]; reading $Z$ as $Y$ is an assumption outputs cannot test, and shared-confounder models need assumptions or heuristics to say which factor is quality [22]. With additive scores and centred errors, comparisons reveal agents’ relative biases but not their shared bias [4,3].

Acceptance: precision, not recall. Acceptance estimates $p ,$ not $H \colon$ : the selectivelabels problem [15]. Econometrics calls such bounds partial identification [17].

Proposition 1 (Acceptance). For any π and k with $a p > 0$ , the tight interval for recall is $[ a p / ( a p + 1 - a ) , 1 ]$ , since $r = a p / ( a p + H )$ decreases in H and the data allow any $H \in [ 0 , 1 - a ]$

Overlap: an upper bound on recall. Let each agent nominate, $A _ { i } \in \{ 0 , 1 \}$ , and the assessor judge the union $A = \operatorname* { m a x } _ { i } A _ { i } .$ , as in pooled evaluation. Judgments show who found each relevant item. Take two disjoint non-empty groups of agents, let $B$ and $B ^ { \prime }$ indicate a nomination by each and C that all other agents stay silent, and write $h _ { 1 0 } = \mathbb { P } ( Y = 1 , C , B = 1 , B ^ { \prime } = 0 )$ , and likewise $h _ { 0 1 }$ and $h _ { 1 1 } ;$ all are observed in the pool. Fix a family of such splits and assume on each positive dependence: Cov $( B , B ^ { \prime } \mid Y = 1 , C ) \ge 0$ , so that among relevant items all other agents miss, a nomination by one group makes one by the other no less likely, and $h _ { 1 1 } ~ > ~ 0$ . By Chebyshev’s association inequality the covariance condition holds on every split if agents are mutually independent given $( \theta , Y = 1 )$ for a scalar item findability $\Theta$ and each $\mathbb { P } ( A _ { i } = 1 \mid \theta , Y = 1 )$ is non-decreasing in Θ.

Theorem 2 (Overlap). Let L be the largest $h _ { 1 0 } h _ { 0 1 } / h _ { 1 1 }$ over the family. Then $H \geq L . \ I f L \leq 1 - a ;$ , the tight interval for recall under these assumptions is $\left[ a p / ( a p + 1 - a ) , \ a p / ( a p + L ) \right] _ { }$ ; if $L > 1 - a$ , the data refute them.

Proof. The joint shares of $( B , B ^ { \prime } )$ with $Y = 1$ and C form a $2 \times 2$ table whose unobserved cell, found by neither group, is H, and the conditional covariance has the sign of $H h _ { 1 1 } - h _ { 1 0 } h _ { 0 1 }$ . Any $H \in [ L , 1 - a ]$ satisfies these inequalities and fits the data, since items with $A = 0$ have no judgments. ⊓⊔

Little overlap means many misses. The ratio $h _ { 1 0 } h _ { 0 1 } / h _ { 1 1 }$ is the Lincoln– Petersen capture-recapture quantity [11]; without an assumption the unobserved cell cannot be determined, however many lists there are [1]. The theorem adapts the odds-ratio restrictions of Sun et al. [18] with lower bound one, adding the cap $H \leq 1 - a$ . Items invisible to every agent are consistent with positive dependence, so overlap cannot raise the lower end. Specialists covering diferent parts of the collection can violate the assumption; then apply the bound within a specialism where it holds. Sampled pools need weighting and add variance.

The next agent. Let $f _ { 1 } = \mathbb { P } ( Y = 1$ , exactly one of the first k agents nominates).

Proposition 2 (Next agent). $I f A _ { 1 } , \dotsc , A _ { k + 1 }$ are exchangeable given $Y = 1$ (relabelling agents changes nothing), agent $k + 1$ adds new relevant items worth at most a share $f _ { 1 } / k$ of the collection.

Proof. Among relevant items, “agent $k { + 1 }$ nominates and $1 , \ldots , k$ do not” implies “agent $k + 1$ nominates and $1 , \ldots , k - 1 { \mathrm { ~ d o ~ n o t } } ^ { \prime } $ , which by exchangeability is as likely as “agent k alone among the first k nominates”, a share $f _ { 1 } / k$ ⊓⊔

Here exchangeability replaces the independent sampling behind incidence-based Good–Turing estimates [12,8]. An agent of a new kind is not covered.

Audit: closing the gap. Only judgments outside the pool pin down H without assumptions on the agents.

$$
\hat { H }
$$

$$
1 - a
$$

$$
n _ { 0 }
$$

$$
| \hat { H } - H | \leq \epsilon
$$

$$
p r o b a b i l i t y \ge 1 - \delta
$$

$$
n _ { 0 } \ge ( 1 - a ) ^ { 2 } \ln ( 2 / \delta ) / ( 2 \epsilon ^ { 2 } )
$$

$$
\left( H o e f f d i n g \ : / 1 \mathcal { 3 } \right)
$$

This is suficient, not necessary: for 1000 items, $a = 0 . 2 , \delta = 0 . 0 5 \ \mathrm { a n d } \ \pm 0 . 0 5$ , it asks for 473 audits, where an exact hypergeometric calculation needs 192.

## 4 Worked Example: AI Reviewers and Human Experts

Here items are paper submissions, agents are AI reviewers, and a paper is relevant if a human expert would accept it. At a hypothetical venue, three reviewers nominate 200 of 1000 submissions $( a = 0 . 2 )$ , and experts accept 120 $( p = 0 . 6 )$ Nominations alone fit $p = 0$ or 1 as well as 0.6 unless reviewers err independently given acceptability (Theorem 1). Acceptance alone gives $r \in [ 1 3 \% , 1 0 0 \% ]$ (Proposition 1). Of the 120, each reviewer alone nominated 30, each pair alone $6 ,$ and all three 12. With reviewer 3 silent, reviewers 1 and 2 give counts 30, 30 and 6, so at least $3 0 { \times } 3 0 / 6 = 1 5 0$ acceptable papers were missed if reviewers are positively dependent, and $r \in [ 1 3 \% , 4 4 \% ]$ (Theorem 2). The venue can buy 200 more expert reviews. A 4th reviewer, if interchangeable, adds at most $9 0 / 3 = 3 0 $ acceptable papers (Proposition 2). An audit of 192 of the other 800 submissions estimates $H { \mathrm { ~ t o ~ } } { \pm } 0 . 0 5$ with 95% confidence (Proposition 3) and, under positive dependence, finds at least $1 9 2 \times 1 5 0 / 8 0 0 \ = \ 3 6$ such papers in expectation. It finds more and alone certifies the misses but uses nearly all the budget; the 4th reviewer needs reviews only for its new nominations, which could be far fewer.

## 5 Take-Home Messages

Rate agents on judgments, not agreement, which can reward shared bias. Report precision, and recall as an interval: a single number claims too much. Let agents nominate separately: judge their union and log who found each item, since a threshold on an aggregate score can leave nominations unjudged. Compare the next agent with an audit before buying either, and budget audits by the accuracy wanted, not the number of agents; an audit can also reject positive dependence.

## References

1. Aleshin-Guendel, S., Sadinle, M., Wakefield, J.: The central role of the identifying assumption in population size estimation. Biometrics 80(1), ujad028 (2024). https://doi.org/10.1093/biomtc/ujad028

2. Allman, E.S., Matias, C., Rhodes, J.A.: Identifiability of parameters in latent structure models with many observed variables. The Annals of Statistics 37(6A), 3099– 3132 (2009). https://doi.org/10.1214/09-AOS689

3. Balzano, L., Nowak, R.: Blind calibration of sensor networks. In: Proceedings of the 6th International Symposium on Information Processing in Sensor Networks. pp. 79–88. IEEE (2007). https://doi.org/10.1109/IPSN.2007.4379667

4. Barooah, P., Hespanha, J.P.: Estimation from relative measurements: Electrical analogy and large graphs. IEEE Transactions on Signal Processing 56(6), 2181– 2193 (2008). https://doi.org/10.1109/TSP.2007.912270

5. Bianchi, F., Queen, O., Thakkar, N., Sun, E., Zou, J.: Exploring the use of AI authors and reviewers at Agents4Science. Nature Biotechnology 44(1), 11–14 (2026). https://doi.org/10.1038/s41587-025-02963-8

6. Biswas, J., Schoepp, S., Vasan, G., Opipari, A., Zhang, A., Hu, Z., Joseph, S., Lease, M., Li, J.J., Stone, P., Wagstaf, K.L., Taylor, M.E., Jenkins, O.C.: AI-assisted peer review at scale: The AAAI-26 AI review pilot. arXiv:2604.13940 (2026)

7. Buckley, C., Voorhees, E.M.: Retrieval evaluation with incomplete information. In: Proceedings of the 27th Annual International ACM SIGIR Conference on Research and Development in Information Retrieval. pp. 25–32. ACM (2004). https://doi. org/10.1145/1008992.1009000

8. Chao, A., Colwell, R.K.: Thirty years of progeny from Chao’s inequality: Estimating and comparing richness with incidence data and incomplete sampling. SORT-Statistics and Operations Research Transactions 41(1), 3–54 (2017). https: //doi.org/10.2436/20.8080.02.49

9. Dawid, A.P., Skene, A.M.: Maximum likelihood estimation of observer error-rates using the EM algorithm. Applied Statistics 28(1), 20–28 (1979). https://doi. org/10.2307/2346806

10. Faggioli, G., Dietz, L., Clarke, C.L.A., Demartini, G., Hagen, M., Hauf, C., Kando, N., Kanoulas, E., Potthast, M., Stein, B., Wachsmuth, H.: Perspectives on large language models for relevance judgment. In: Proceedings of the 2023 ACM SIGIR International Conference on Theory of Information Retrieval. pp. 39–50. ACM (2023). https://doi.org/10.1145/3578337.3605136

11. Fienberg, S.E.: The multiple recapture census for closed populations and incomplete $2 ^ { k }$ contingency tables. Biometrika 59(3), 591–603 (1972). https://doi.org/ 10.1093/biomet/59.3.591

12. Good, I.J.: The population frequencies of species and the estimation of population parameters. Biometrika 40(3–4), 237–264 (1953). https://doi.org/10.1093/ biomet/40.3-4.237

13. Hoefding, W.: Probability inequalities for sums of bounded random variables. Journal of the American Statistical Association 58(301), 13–30 (1963). https: //doi.org/10.1080/01621459.1963.10500830

14. Kim, E.M., Garg, A., Peng, K., Garg, N.: Correlated errors in large language models. In: Proceedings of the 42nd International Conference on Machine Learning. Proceedings of Machine Learning Research, vol. 267, pp. 30038–30066. PMLR (2025)

15. Lakkaraju, H., Kleinberg, J., Leskovec, J., Ludwig, J., Mullainathan, S.: The selective labels problem: Evaluating algorithmic predictions in the presence of unobservables. In: Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. pp. 275–284. ACM (2017). https://doi.org/10.1145/3097983.3098066

16. Lewis, D.D., Yang, E., Frieder, O.: Certifying one-phase technology-assisted reviews. In: Proceedings of the 30th ACM International Conference on Information and Knowledge Management (CIKM). pp. 893–902. ACM (2021). https: //doi.org/10.1145/3459637.3482415

17. Manski, C.F.: Partial Identification of Probability Distributions. Springer Series in Statistics, Springer (2003). https://doi.org/10.1007/b97478

18. Sun, J., Van Baelen, L., Plettinckx, E., Crawford, F.W.: Dependence-robust confidence intervals for capture–recapture surveys. Journal of Survey Statistics and Methodology 11(5), 1133–1154 (2023). https://doi.org/10.1093/jssam/ smac031

19. Thomas, P., Spielman, S., Craswell, N., Mitra, B.: Large language models can accurately predict searcher preferences. In: Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval. pp. 1930–1940. ACM (2024). https://doi.org/10.1145/3626772.3657707

20. Webber, W.: Approximate recall confidence intervals. ACM Transactions on Information Systems 31(1), 2:1–2:33 (2013). https://doi.org/10.1145/2414782. 2414784

21. Zhang, D., Wang, J., Chang, B.: Multi-Agent Systems: A Contemporary Treatment. Bloo-Mind Press (2026), https://books.bloo-mind.ai/masact/

22. Zhao, J., Shin, C., Huang, T.H., Namburi GNVV, S.S.S., Sala, F.: CARE: Confounder-aware aggregation for reliable LLM evaluation. In: Proceedings of the 43rd International Conference on Machine Learning. Proceedings of Machine Learning Research, vol. 306, pp. 162114–162150. PMLR (2026)

23. Zobel, J.: How reliable are the results of large-scale information retrieval experiments? In: Proceedings of the 21st Annual International ACM SIGIR Conference on Research and Development in Information Retrieval. pp. 307–314. ACM (1998). https://doi.org/10.1145/290941.291014