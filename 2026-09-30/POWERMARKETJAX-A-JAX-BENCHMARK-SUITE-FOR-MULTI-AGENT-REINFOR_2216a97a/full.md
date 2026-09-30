# POWERMARKETJAX: A JAX BENCHMARK SUITE FOR MULTI-AGENT REINFORCEMENT LEARNING IN POWER MARKETS

Zhanhua Pan<sup>1†</sup>, Xin Qin<sup>1†</sup>, Xiao Liu<sup>2</sup>, Zhilong Cao<sup>1</sup>, Jianhong Wang<sup>3‡</sup>, Dawei Qiu<sup>1‡∗</sup>

<sup>1</sup>Nanyang Technological University, Singapore. <sup>2</sup>Cornell University, USA. <sup>3</sup>University of Bristol, UK.

## ABSTRACT

Power markets are a natural testbed for multi-agent reinforcement learning (MARL), where multiple self-interested participants repeatedly submit bids. A market-clearing mechanism then determines dispatch and prices subject to power grid constraints and market settlement rules. However, existing MARL environments typically focus on a single market setting, implement simplified clearing mechanisms, or rely on CPU-based optimization solvers that slow large-scale training and limit the systematic study of bidding strategies and market behavior. We introduce PowerMarketJax, a benchmark suite for MARL across five power markets: day-ahead wholesale, real-time balancing, ancillary services, peer-topeer double auctions, and local flexibility. Each environment implements its own clearing, pricing, and settlement rules while providing a common framework for learning and evaluation. We find that learned bidding behavior depends strongly on the market design: independent learners can miss better strategies when gains require many agents to change together, when more profitable strategies lie beyond a region of lower profit, or when profits disappear as more agents adopt the same strategy. PowerMarketJax implements both market simulation and policy training in JAX, allowing the entire pipeline to run on the GPU with 1 024 × 1 200 parallelisms across both environments and market participants, achieving up to 33× speedup over CPU-based baselines. Our open-source benchmark is available at: https://github.com/powermarketjax/PowerMarketJax.

## 1 INTRODUCTION

Unlike most commodities, electricity cannot be stored efficiently at scale, so power markets are needed to continuously match supply and demand through mechanisms that determine which resources produce and at what price. Historically, these markets involved a relatively small set of large generators and mainly traded energy, making bidding strategies easier to characterize (David & Wen, 2000). This structure is now changing. Renewable generation, flexible demand, and energy storage have expanded both who participates and what is traded (Matamala & Strbac, 2025): alongside energy, markets now trade reserves, balancing, and ancillary services across multiple timescales and grid levels, each with its own clearing, pricing, and settlement rules (Zhu et al., 2025). This growing diversity of participants, products, and market mechanisms makes bidding behavior increasingly difficult to understand, creating an urgent need to study how participants bid and interact across different market settings. The repeated interactions among many market participants make power markets a natural setting for multi-agent reinforcement learning (MARL) (Liang et al., 2020).

In a MARL formulation, each market participant can be modeled as an agent that observes market information, submits bids, receives clearing outcomes, and updates its bidding strategy over time. Since a participant’s profit depends on both its own bid actions and the actions of others through market clearing, bidding in power markets is inherently a multi-agent decision problem. MARL enables participants to learn such bidding strategies through repeated market interactions withou requiring explicit knowledge of competitors’ exact strategies (Du et al., 2021).

A number of environments have been developed for learning-based bidding in power markets, but important gaps remain for systematic MARL study. Most focus on a single market study, limiting their use as general benchmarks for MARL-based bidding (Yeh et al., 2023; Harder et al., 2025; El Helou et al., 2023; Salazar-Pena et al., 2026). A second gap is computational burden. An environ-˜ ment must solve a market-clearing optimization subject to grid physical constraints after every round of bids. Existing implementations typically rely on CPU solvers, making the millions of clearing problems required during MARL training a major computational bottleneck, especially with many agents and parallel environments (Wolgast & Nieße, 2024). To reduce this computational cost, some environments simplify pricing or settlement rules, but these simplifications can remove important market features that shape bidding incentives (Wolgast & Nieße, 2024). As a result, it remains difficult to systematically study how bidding strategies emerge and differ across power markets.

Here, we introduce PowerMarketJax, a power market MARL benchmark implemented entirely in JAX (Bradbury et al., 2018). It parallelizes both optimization-based market environments and multiagent policy computation on the GPU. By compiling clearing and learning into a single computation graph, PowerMarketJax avoids repeated CPU solver calls and host-device synchronization at each market step, enabling large-scale exploration of bidding strategies across different markets. This design achieves up to 33× higher throughput than CPU-based implementations and runs 1 024 parallel 1 200-participant market environments at over 27 000 environment steps per second (Appendix B).

Our contributions are summarized as follows. First, we present a unified MARL benchmark covering five power markets: day-ahead wholesale, real-time balancing, ancillary services, peer-to-peer energy trading, and local flexibility. The benchmark spans different clearing mechanisms, timescales, and grid levels while preserving each market’s clearing, pricing, and settlement rules under a common interface. Second, we implement both market-clearing optimization and MARL policy training entirely in JAX, enabling GPU-based parallel computation across environments and agents for largescale training and evaluation. Third, using this benchmark, we empirically show that market design strongly shapes what independent learners can discover from their own profit signals: profitable outcomes may require coordinated changes by many agents, lie beyond locally unfavorable regions, or disappear as more agents adopt the same strategy. These findings reveal learning challenges that arise from the interaction between strategic incentives and physically constrained power markets.

## 2 RELATED WORK

Power Market RL Benchmarks. A growing number of environments support RL for bidding in power markets. At the transmission level, SustainGym (Yeh et al., 2023) studies storage bidding in a network-constrained real-time market. ASSUME (Harder et al., 2025) supports RL-based bidding in European power markets. At the distribution level, OpenGridGym (El Helou et al., 2023) studies competing bidding agents under distribution network constraints. MARLEM (Salazar-Pena et al.,˜ 2026) considers MARL-based storage participation in local energy markets. Collectively, these works demonstrate the potential of (MA)RL for learning bidding strategies under different market mechanisms and physical constraints.

However, existing environments typically focus on a single market, limiting systematic study of bidding strategies across diverse power markets. PowerMarketJax instead covers five representative markets spanning transmission and distribution levels. It further provides a unified framework for agent–market interaction while preserving each market’s clearing, pricing, settlement, and physical constraints. Finally, existing environments commonly separate CPU-based market simulation from GPU-based policy learning, creating repeated CPU–GPU data transfers and making market clearing a major bottleneck for MARL training. PowerMarketJax implements both market clearing and agent policy computation in JAX, enabling parallel execution on the GPU.

JAX-based RL Benchmarks. A growing number of RL benchmarks have been implemented in JAX, motivated by the need for fast and highly parallel simulation (Rutherford et al., 2024). Isaac Gym (Makoviychuk et al., 2021) showed that moving environment simulation onto the GPU can reduce CPU–GPU communication overhead. JAX (Bradbury et al., 2018) further supports acceleratornative RL through compilation and vectorization. These advances have enabled high-throughput JAX environments across video games (Koyamada et al., 2023; Radji et al., 2026), physics (Freeman et al., 2021), robot learning (Zakka et al., 2025), and autonomous driving (Gulino et al., 2023).

Recent work has also brought JAX-based acceleration to economic and trading environments. EconoJax (Ponse et al., 2025) simulates agent-based markets with order matching, JAX-LOB (Frey et al., 2023) accelerates limit-order-book simulation. These environments primarily rely on predefined if–else rule-based matching logic rather than constrained optimization. In contrast, power markets provide a distinctive setting where strategic economic decisions are tightly coupled to a real-world physical grid, requiring market outcomes to satisfy both market rules and engineering constraints. PowerMarketJax targets this setting by implementing optimization-based market clearing and strategic bidding entirely in JAX across five representative power markets.

## 3 BACKGROUND

Power systems maintain supply-demand balance through a range of markets for energy, reserves, and flexibility operating at different timescales and grid levels. Following the market structures and flexibility mechanisms discussed in Kirschen & Strbac (2026), this benchmark considers five representative settings: day-ahead wholesale, real-time balancing, ancillary services, peer-to-peer (P2P) energy trading, and local flexibility, as illustrated in Figure 1. They cover dispatch scheduling against day-ahead forecasts, adjustment to real-time imbalances, procurement of system reserves, local energy exchanges among prosumers, and distribution network support from flexible resources.

These markets share a common decision structure but differ in their mechanisms. In each market, self-interested participants such as generators, storages, and consumers submit bids based on their asset states and public market signals, without knowing competitors’ private information. Their bids jointly determine market clearing, prices, and profits. However, the five markets differ in their traded products, information structures, physical constraints, and clearing and settlement rules. These differences shape participants’ incentives and bidding behavior, ultimately affecting market outcomes.

M1: Day-ahead wholesale (Day d − 1, transmission level). Generation must be scheduled before delivery day d because thermal units require time to start up and are subject to time-coupling constraints such as minimum up/down times. The dayahead market clears supply–demand bids using forecasts for day d, determining unit commitment, generation schedules, and locational marginal prices (LMPs) that reflect transmission constraints. Cleared schedules then establish the baseline for subsequent real-time balancing.

![](images/ebda9811d568c22d51aaaf112aeaad4be7fb3227c9bfadcc546b52839fbcf552.jpg)  
Figure 1: Five power markets in PowerMarketJax across different timescales and grid levels.

M2: Real-time balancing (Day d, transmission level). Actual demand

and renewable generation may deviate from day-ahead forecasts because weather and consumption patterns cannot be predicted exactly one day in advance. The real-time market redispatches available resources over short intervals, typically every 15–30 minutes on day d, to maintain supply-demand balance under updated system conditions. Deviations from day-ahead schedules are settled at real time prices, which can be more volatile and therefore expose participants to greater price risk.

M3: Ancillary services (Day d, transmission level). Scheduled energy alone cannot ensure reliable operation because system disturbances such as a sudden generator outage can unbalance the system within seconds, far faster than the 15–30-minute energy dispatch interval. The ancillary services market procures capabilities such as frequency regulation and operating reserves, with requirements on availability and response speed. In this market, service providers are paid for reserved capacity, so their revenue can depend on maintaining available capacity rather than solely on energy production.

M4: Peer-to-peer energy trading (Day d, distribution level). Prosumers with rooftop solar and storage can trade energy directly with nearby consumers instead of relying solely on upstream retailers. The P2P market matches buy bids and sell offers within a local energy community. The market-clearing price is determined by the intersection of the aggregated supply and demand curves, while matched quantities are determined from the submitted buy and sell quantities. Unmatched demand is supplied by the upstream grid at the higher retail price, while unmatched supply is exported to the grid at the lower feed-in tariff, creating an economic incentive for local peer-to-peer trading.

M5: Local flexibility (Day d, distribution level) Even when sufficient electricity is available from the transmission system, distribution networks may become overloaded or local voltages may violate operating limits. To address these issues, the distribution system operator (DSO) procures flexibility from local resources, paying them to increase or decrease generation or consumption at specified locations, times, and durations. Under the pay-as-bid rule considered here, accepted flexibility providers are paid their submitted bid prices rather than a common clearing price.

## 4 POWERMARKETJAX: POWER MARKET ENVIRONMENTS IN JAX

PowerMarketJax is a suite of five power market environments implemented in JAX, with each market formulated as a MARL environment, as illustrated in Figure 2. The environments span dayahead wholesale (M1), real-time balancing (M2), ancillary services (M3), peer-to-peer energy trading (M4), and local flexibility (M5). The markets preserve their distinct clearing, pricing, settlement, and physical constraints while following a common agent–market interaction interface.

Despite their different market rules, all five environments follow the same step sequence. Agents first observe the information available to them and submit their market actions. The market then clears according to its specific mechanism, the resulting dispatches and prices are settled, and the environment returns the next states and rewards. We define the reward as the participant profit under the corresponding settlement rule. Together, these operations define one transition of the corresponding POSG and are implemented through a common step interface. Further details of the five environments and their accelerator-based execution are provided below.

![](images/1444497aca64fed009ba32e6f711a67426f7dc6e5001ffffe88ae822a32a504e.jpg)  
Figure 2: Execution paradigm of PowerMarketJax.

## 4.1 FIVE MARKET ENVIRONMENTS AS POSGS

We formulate each market environment as a general-sum partially observable stochastic game (POSG) (Hansen et al., 2004). For each environment, we describe the market step, agent observations and actions, the clearing mechanism, reward, and state transition. Detailed market models and complete POSG formulations are provided in Appendix A.1–A.5.

M1: Day-ahead wholesale. One step corresponds to one day-ahead market clearing, in which all 24 hours of the following day are cleared jointly. Each generator agent observes its initial commitment status, initial power output, and residual minimum up- and down-time requirements, together with the public day-ahead demand forecast. Its action consists of the prices submitted across all bid segments and delivery periods. Given the joint bids, the market-clearing mechanism determines unit commitment, dispatches, and LMPs, and each generator receives its daily profit as the reward.

M2: Real-time balancing. One step corresponds to one 30-minute real-time market clearing during the delivery day. Each generator agent observes its previous realized dispatch, the day-ahead commitment and schedule for the corresponding hour, the associated day-ahead LMP, and the public realized net-demand conditions. Its action consists of the prices submitted across all bid segments for the current real-time interval. Given the joint bids from all generators, the market-clearing mechanism redispatches the committed generators subject to network and ramping constraints and determines the real-time LMPs. Each generator receives its realized interval profit under the twosettlement rule as the reward, where the day-ahead schedule remains settled at the day-ahead LMP and only deviations are settled at the real-time LMP. The realized dispatch of the current interval is carried forward as the previous dispatch for the next interval.

M3: Ancillary services. One step corresponds to one 30-minute joint energy–reserve market clearing during the delivery day. Each generator agent observes its previous realized dispatch, the dayahead commitment and schedule for the corresponding hour, the associated day-ahead LMP, the public realized net-demand conditions, and the published reserve requirements. Its action consists of the energy bid prices across all bid segments together with one reserve bid price for each reserve product. Given the joint bids of both energy and reserve, the market-clearing mechanism co-optimizes energy dispatch and reserve procurement subject to network, ramping, response time, and reserve capacity constraints, and determines both real-time LMPs and reserve prices. Each generator then receives its realized interval profit as the reward, including day-ahead and real-time energy settlements and reserve capacity payments. The realized dispatch of the current interval is carried forward as the previous dispatch for the next interval.

M4: Peer-to-peer energy trading. One step corresponds to one 15-minute P2P trading interval. Each prosumer agent observes its own battery state of charge, net energy position, and the public retail and export prices, where a positive net position corresponds to a buy bid and a negative net position to a sell offer. It does not observe the private states or contemporaneous bids and asks of other prosumers. Its trading quantity is fixed before the auction based on its local generation, demand, and battery operation, so its action is the price submitted as an ask when selling or a bid when buying. Given the joint submissions, the double-auction mechanism sorts sellers and buyers, determines the matched quantities and uniform local clearing price, and settles any unmatched surplus or deficit with the upstream grid. Each prosumer then receives its realized interval profit as the reward, accounting for both P2P and residual grid transactions. The battery state of charge evolves to the next interval, while generation and demand update the prosumer’s next net energy position.

M5: Local flexibility. One step corresponds to one 1-hour local flexibility procurement interval. Each aggregator agent observes its own battery state of charge, photovoltaic generation, local de mand, and available flexibility, together with the public voltage and line-congestion requirements published by the DSO. It does not observe the private operating states, delivery costs, or contemporaneous bids of competing aggregators. Its action is a price–quantity bid specifying the requested payment and the maximum flexibility quantity it is willing to provide. Given the joint offers, the DSO clears the market by minimizing procurement cost subject to network voltage, line flow, and flexibility delivery constraints, and may partially accept a bid. Each aggregator then receives its realized interval profit as the reward under the pay-as-bid rule, with accepted flexibility paid at the submitted bid price. The accepted flexibility changes the battery state of charge and therefore determines the flexibility available in the next interval, while photovoltaic generation, local demand, and network conditions evolve to form the next market state.

## 4.2 CLEARING AND LEARNING IN A SINGLE COMPUTATION GRAPH

Despite their different clearing and settlement mechanisms, all five environments share the same accelerator-based execution framework, with market simulation and policy learning running within a single compiled computation graph.

Every operation within a training iteration, including market clearing, runs inside a single compiled computation graph on the device. reset and step are pure functions: each step sequentially applies action submission, market clearing, settlement, and observation construction, while the environment state is represented as a fixed-structure pytree that is passed as input and returned as output. Each step is compiled with jit, vectorized across parallel environments with vmap, and repeated over a fixed rollout horizon using lax.scan. The policy is evaluated within the same graph: during rollout, it maps observations to actions that are passed directly to the environment step. Policy updates are implemented with a second lax.scan over epochs and minibatches. A complete training iteration, consisting of rollout and update, is compiled as a single function. Further details of the compiled execution graph are provided in Appendix C.8.

Market clearing remains inside the compiled graph because each clearing mechanism is expressed with fixed shape operations and iteration counts. The optimization-based markets use a common primal–dual interior-point solver for linear programs, with a fixed number of Newton iterations. The day-ahead market additionally includes binary commitment decisions. Rather than invoking a mixed-integer solver inside the compiled graph, we use a three-stage procedure. We first relax the commitment problem to a linear program, round the resulting commitment decisions, and then re-solve the continuous dispatch problem with the rounded commitment fixed (Appendix A.1.1). It is noted that the P2P market does not require an optimization solver. Its sorting, matching, and settlement operations are implemented directly using JAX Primitives (Bradbury et al., 2018).

We validate these market-clearing implementations in three ways (Appendix B.4). First, clearing and settlement results are compared against a NumPy reference implementation. Second, dual prices produced by the JAX linear-program solver are compared with those obtained from HiGHS (Huangfu & Hall, 2018), an independent high-performance optimization solver. Third, the three-stage procedure is compared with the exact mixed-integer solution from HiGHS, with the commitment decisions kept binary. The resulting optimality gap is reported in Appendix B.4.

We parallelize computation across both environments and agents. Each environment has its own market state and random key, and vmap batches multiple environments for parallel execution. Within each environment, agents remain coupled through the market-clearing mechanism, since their submitted actions jointly determine the market outcome. For policy evaluation, agents are also vectorized, so the actions of all agents across all parallel environments are computed simultaneously on the GPU in a single batched policy evaluation.

## 5 RESULTS

Systems and data. The three transmission-level markets (M1–M3) use the British case29gb system<sup>1</sup> with 29 buses and 66 generators as the main test case (Appendix E.1.1). They use one year of Great Britain demand and day-ahead forecast data, with 329 days for training and 36 held-out days for evaluation. M2 and M3 inherit the commitment and schedules obtained from the day ahead market M1. Additional experiments on the 73-unit RTS-GMLC system<sup>2</sup> and the 151-unit Australian NEM system<sup>3</sup> are reported in Appendices E.1 to E.3. At the distribution level, M4 uses 1,200 Belgian households with PV from the Fluvius smart meter dataset<sup>4</sup>. M5 uses the 129-bus SwissDN feeder (Zapparoli et al., 2025) with battery fleets of 24 and 34 aggregators under its 2040 and 2050 scenarios. Detailed market formulations, system parameters, and data descriptions are provided in Appendices A and E.

MARL algorithms. We evaluate two independent MARL algorithms, IPPO (de Witt et al., 2020) and SAC (Haarnoja et al., 2018) (the independent version), each with parameter sharing (PS) and agent-specific parameters (NoPS), giving four learners: IPPO-PS, IPPO-NoPS, SAC-PS, and SAC-NoPS. Each agent observes only its own available information and maximizes its own market profit, without a centralized critic or inter-agent communication. We use common default hyperparameters across markets and parallel environment rollouts on the GPU. Full training configurations and market-specific hyperparameters are provided in Appendix D.

Baselines and evaluation. We compare the learned policies against truthful bidding and the corresponding untrained networks. Truthful bidding provides a competitive reference, while the untrained network is used to measure the effect of policy training. We further compare the learned policies with market-specific predefined non-learning strategies, including tests over different generator bidding levels in M1–M3, a fixed battery arbitrage schedule in M4, and unilateral flexibility-price deviations in M5. M1–M4 use 3 random seeds per experiment, while M5 uses 3–10 depending on the configuration. Evaluation uses held-out data and averages results across seeds. Full evaluation protocols and additional results are reported in Appendix E.

All experiments run on a workstation with RTX 4500 Ada GPUs, described in Appendix B. The market models and data, the evaluation protocol and further results are in Appendices A to E.

## 5.1 DAY-AHEAD WHOLESALE MARKET

We start with the day-ahead market (M1) on case29gb. Each generator i bids at $\alpha _ { i }$ times its marginal generation cost, where $\alpha _ { i } \ \geq \ 1$ . Here, $\alpha _ { i } ~ = ~ 1$ corresponds to truthful bidding, while $\alpha _ { i } > 1$ represents bidding above marginal cost. Figure 3(a) shows that all four learners end near a bidding strategy around $\alpha _ { i } = 1 . 5$ . Over the 36 evaluation days, IPPO-PS and IPPO-NoPS learn to raise market prices with $\alpha _ { i } = 1$ .50 and $\alpha _ { i } = 1 . 5 6$ , respectively, and thereby increase the total generator profits by £643 and £917 million. The other two test systems show the same effect on a smaller scale, with $\alpha _ { i }$ reaching 1.07–1.29 (Appendix E.1.3).

However, the learned policies still leave substantial joint profit unrealized. As shown in Figure 3(b), when all 66 generators use the same α, total profit reaches the learned policy level near $\alpha = 1 . 5$ and rises to £697 million at $\alpha \ = \ 2$ In contrast, when one generator’s α is increased at a time while the others keep their learned IPPO-PS bids fixed, the resulting profit never exceeds the learned-policy level and is highest around $\alpha = 1 . 5 \mathrm { - }$ 1.7. The learners therefore settle

![](images/b74cf670a99c58ff38fd2d282107a90f9ceea3eeee49c99d2aef82d8de6dc9f5.jpg)

![](images/67084f2e15338efc0a81feb84cd2f24b1bace0e995569f33811fe4462594f777.jpg)  
Figure 3: M1 day-ahead wholesale market (a) learning curves, (b) profit evaluation under joint and unilateral bidding strategies.

near a level where raising the bid further does not benefit an individual generator, even though all generators could gain substantially by raising their bids together.

This gap arises for two reasons. First, the joint gain is not visible in each agent’s individual learning signal. An independent learner observes only how its own bidding strategy affects its own profit. For example, cheap nuclear units almost never set the market price (Appendix E.1.4), so raising thei bids alone has little effect on price. Second, independent exploration rarely reaches the high-profit joint bidding region. The sampled α values remain concentrated around 1.5, making it unlikely that all units simultaneously explore the higher-bid region where the joint gain occurs.

## 5.2 REAL-TIME BALANCING AND ANCILLARY SERVICES MARKETS

We next consider the real-time balancing (M2) and ancillary services (M3) markets. In M2, raising a bid alone is rarely profitable because a generator can lose output to its competitors. Accordingly, $\alpha _ { i }$ of the 30 dayahead committed generators decreases during training under all four learners, reaching 1.04 under IPPO-PS and 1.24–1.40 under the other three methods (Figure 4(a,b)). This contrasts with the day-ahead market, where bidding above cost can increase an individual generator’s profit.

The difference between individual and joint incentives is even clearer in M3 (Figure 4(c)). Raising the reserve bid of a single

![](images/f992ea6b2ef8cd930940bce5fe4fcf1d71092240b700816bae692a15302b2802.jpg)

![](images/9a4179adab770622823952eef1e03e7adaf73b6f13dd5a6b401502eab27b1adc.jpg)  
Figure 4: M2 real-time balancing market (a) bidding strategy, (b) learning curves, and M3 ancillary services market (c) profit under joint and unilateral bidding strategies.

generator produces little gain: across the 29 committed generators, the largest profit increase is only £0.08 million per day. However, when all 66 generators raise their reserve bids together to £150/MWh, total profit increases by £9.05 million per day. However, the learned policies remain far below this joint gain, earning only £0.56–£3.69 million per day more than truthful bidding, and none outperforms its corresponding non-learning strategies.

These results show how markets shape the learning signal available to each independent learner. In M2, raising a bid can reduce a generator’s own profit, pushing the learned bid toward cost. In M3, the individual profit signal is nearly zero, making the much larger joint gain difficult to discover.

## 5.3 PEER-TO-PEER ENERGY MARKET

We next consider the P2P energy market (M4), where 1,200 households with PV and a battery trade in a double auction every 15 minutes. The learned policies improve household economics relative to truthful bidding. IPPO-NoPS increases household profit on all evaluation days, while IPPO-PS achieves the highest average profit, reducing the daily net cost per household from C1.20 under truthful bidding to C0.69–0.77 (Appendix E.4.3).

The learners, however, adopt different battery strategies (Figure 5(a)). IPPO-NoPS and SAC-NoPS charge during the midday PV surplus and discharge during the evening demand peak. IPPO-PS and SAC-PS keep their batteries close to the minimum state of charge after selling their initial energy. To understand these differences, we let an increasing number of households follow a fixed schedule that charges from 11:00–15:00 and discharges from 18:00–22:00. A single household gains about C1.13 per day, but this gain decreases as more households adopt the same strategy and becomes negative at around 210–240 households (black line in Figure 5(b)), detailed in Appendix E.4.4.

The decline is caused by the resulting change in market prices (Figure 5(c)). Simultaneous charging raises the midday clearing price, while simultaneous discharging lowers the evening price, reducing the price differentials on which the arbitrage relies. Once enough households participate, the remaining differential no longer covers efficiency losses and degradation costs. Thus, a strategy that is profitable for one household may become unprofitable when widely adopted. Figure 5(b) further suggests that PS can make market outcomes requiring heterogeneous behavior harder to learn: IPPO-NoPS leads only a subset of households to arbitrage, while IPPO-PS leads none to do so.

![](images/ca488d17c4397ad9aba9c6b4016779a113f942463033a4c066c4428ab37bc447.jpg)

![](images/d794528f68ebea70a91712e1e8fb13b8a6aac44c6d2e79aef38172dcd541882d.jpg)

![](images/a4d81b5886082e20466bbed4f659b0301b75c6a93d20e86b097e8d8e9bfaeb85.jpg)  
Figure 5: P2P energy market (a) battery state of charge, (b) household gain, (c) local clearing price.

## 5.4 LOCAL FLEXIBILITY MARKET

Finally, we consider the local flexibility market (M5), where the DSO procures flexibility from battery aggregators to relieve network congestion. The market is pay-as-bid with a price floor. All four learners remain close to the floor. Yet a single aggregator can earn more by raising its offer sufficiently far above the floor; in the two 3.9 MW configurations, its return exceeds the floor-level return again at prices around 3.2 times the floor (Figure 6(a,b)).

The reason is the shape of the individual return. A small bidding price increase from an aggregator causes the operator to procure from other cheaper aggregators instead, sharply reducing the deviating aggregator’s award and profit. Only at much higher prices does its return recover. Gradient-based learners therefore encounter a local barrier: small moves away from the floor reduce profit, while the more profitable region lies much farther away. The price parameterization further weakens exploration near the floor because the softplus mapping becomes nearly flat.

The learners instead increase profit through their planned charging decisions. Planned charging raises the baseline flow on the congested line and therefore increases the flexibility requirement that the operator subsequently procures. With SAC-PS learners, the share of periods with procurement rises from 5.0% to 18.7% with 24 aggregators and from 4.9% to 15.4% with 34 aggregators (Figure 6(c)). A baseline monitor that excludes this learner-induced charging largely removes the effect, causing both procurement and learner returns to fall sharply. Thus, M5 exposes two distinct challenges for MARL in markets: profitable deviations may lie beyond a local learning barrier, and agents can change the demand for the service they are paid to provide.

![](images/d81aa69b23f92d47ee0f640f0958a7a9c577cd5212f7916bc3277fcfb83bd4af.jpg)

![](images/21e489feab4ed84465cc1bb1f9b18b01a75022c72b822cf8b3d8cadc2aea7f56.jpg)

![](images/4f6770c9b1718984af4121c7063c6ab43f63888bf0d309310fc3139380fc525a.jpg)  
Figure 6: M5 local flexibility market (a) schematic return versus own offer price, (b) measured return in four system configurations, (c) procurement frequency versus the number of SAC-PS learners (seed mean with 95% bootstrap CI).

## 6 CONCLUSION

PowerMarketJax shows that MARL benchmarks for power markets can preserve market-design fidelity while remaining computationally scalable. By integrating market clearing, settlement, policy evaluation, and learning into a GPU-accelerated JAX pipeline, the benchmark supports fast multiagent rollouts while retaining the mechanisms and physical constraints that shape strategic bidding. The five markets provide a common POSG framework for evaluating learned bidding behavior and market outcomes across diverse market designs.

Discussion. The experiments show that learned bidding behavior depends strongly on market design and the profit signal available to each agent. Across the five markets, learners may miss profitable joint outcomes, move toward truthful bidding when individual deviations are unprofitable, lose arbitrage gains when many agents adopt the same strategy, or fail to reach more profitable strategies beyond lower-profit regions. More broadly, these results show that market mechanisms shape the learning landscape itself: they determine which incentives are applied to individual agents and how the actions of many learners feed back into prices, dispatch, and procurement. This reflects challenges for MARL beyond maximizing return, including coordinated exploration, endogenous market responses, and heterogeneous agent behavior. Such learning-based simulations can also help stress-test market designs and reveal strategic incentives before new mechanisms are deployed.

Limitations. PowerMarketJax is a research benchmark rather than a deployment-ready market simulator or bidding system, and the five environments simplify aspects of real market operation, network modeling, participant constraints, and market rules. Our experiments cover only the considered systems and independent PPO and SAC variants, so the learned policies should not be interpreted as predictions of real participant behavior or as equilibrium strategies. Future work could extend the benchmark to richer market designs, participant models, uncertainty, and broader classes of MARL algorithms.

## ACKNOWLEDGEMENT

Dawei Qiu is supported by the Nanyang Technological University (NTU) Start-Up Grant project #026749-00001 “Market Design for Low-Carbon Power Systems: Towards a Reliable, Affordable, and Resilient Transition”. Jianhong Wang is supported by the Engineering and Physical Sciences Research Council (EPSRC) [Grant Ref: EP/Y028732/1].

## REFERENCES

Akshay Agrawal, Robin Verschueren, Steven Diamond, and Stephen Boyd. A rewriting system for convex optimization problems. Journal ofControl and Decision, 5(1):42–60, 2018.

James Bradbury, Roy Frostig, Peter Hawkins, Matthew James Johnson, Yash Katariya, Chris Leary, Dougal Maclaurin, George Necula, Adam Paszke, Jake VanderPlas, Skye Wanderman-Milne, and Qiao Zhang. JAX: Composable transformations of Python+NumPy programs, 2018. URL http://github.com/jax-ml/jax.

Thomas Brown, Jonas Horsch, and David Schlachtberger. PyPSA: Python for power system analy-¨ sis. Journal ofOpen Research Software, 6(1):4–4, 2018.

A.K. David and Fushuan Wen. Strategic bidding in competitive electricity markets: a literature survey. In 2000 Power Engineering Society Summer Meeting (Cat. No.00CH37134), volume 4, pp. 2168–2173 vol. 4, 2000.

Christian Schroeder de Witt, Tarun Gupta, Denys Makoviichuk, Viktor Makoviychuk, Philip H. S. Torr, Mingfei Sun, and Shimon Whiteson. Is independent learning all you need in the StarCraft Multi-Agent Challenge? arXiv:2011.09533, 2020.

Steven Diamond and Stephen Boyd. CVXPY: A Python-embedded modeling language for convex optimization. Journal ofMachine Learning Research, 17(83):1–5, 2016.

Yan Du, Fangxing Li, Helia Zandi, and Yaosuo Xue. Approximating Nash equilibrium in dayahead electricity market bidding with multi-agent deep reinforcement learning. Journal ofModern Power Systems and Clean Energy, 9(3):534–544, 2021.

Rayan El Helou, Kiyeob Lee, Dongqi Wu, Le Xie, Srinivas Shakkottai, and Vijay Subramanian. OpenGridGym: An open-source AI-friendly toolkit for distribution market simulation. IEEE Transactions on Smart Grid, 14(2):1555–1565, 2023.

Daniel Freeman, Erik Frey, Anton Raichuk, Sertan Girgin, Igor Mordatch, and Olivier Bachem. Brax - a differentiable physics engine for large scale rigid body simulation. In Proceedings ofthe Neural Information Processing Systems Track on Datasets and Benchmarks, volume 1, 2021.

Sascha Yves Frey, Kang Li, Peer Nagy, Silvia Sapora, Christopher Lu, Stefan Zohren, Jakob Foerster, and Anisoara Calinescu. JAX-LOB: A gpu-accelerated limit order book simulator to unlock large scale reinforcement learning for trading. In Proceedings of the Fourth ACM International Conference on AI in Finance, pp. 583–591, 2023.

Cole Gulino, Justin Fu, Wenjie Luo, George Tucker, Eli Bronstein, Yiren Lu, Jean Harb, Xinlei Pan, Yan Wang, Xiangyu Chen, et al. Waymax: An accelerated, data-driven simulator for largescale autonomous driving research. Advances in Neural Information Processing Systems, 36: 7730–7742, 2023.

Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine. Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor. In Jennifer Dy and Andreas Krause (eds.), Proceedings of the 35th International Conference on Machine Learning, volume 80 of Proceedings ofMachine Learning Research, pp. 1861–1870. PMLR, 2018.

Eric A. Hansen, Daniel S. Bernstein, and Shlomo Zilberstein. Dynamic programming for partially observable stochastic games. In Proceedings ofthe Nineteenth National Conference on Artificial Intelligence, Sixteenth Conference on Innovative Applications ofArtificial Intelligence, July 25- 29, 2004, San Jose, California, USA, pp. 709–715. AAAI Press / The MIT Press, 2004.

Nick Harder, Kim K. Miskiw, Manish Khanra, Florian Maurer, Parag Patil, Ramiz Qussous, Christof Weinhardt, Marian Klobasa, Mario Ragwitz, and Anke Weidlich. ASSUME: An agent-based simulation framework for exploring electricity market dynamics with reinforcement learning. SoftwareX, 30:102176, 2025.

Q. Huangfu and J. A. J. Hall. Parallelizing the dual revised simplex method. Mathematical Programming Computation, 10(1):119–142, 2018.

Daniel S. Kirschen and Goran Strbac. Fundamentals of Power System Economics. Wiley, 3rd edition, 2026.

Sotetsu Koyamada, Shinri Okano, Soichiro Nishimori, Yu Murata, Keigo Habara, Haruka Kita, and Shin Ishii. Pgx: Hardware-accelerated parallel game simulators for reinforcement learning. Advances in Neural Information Processing Systems, 36:45716–45743, 2023.

Yanchang Liang, Chunlin Guo, Zhaohao Ding, and Huichun Hua. Agent-Based modeling in electricity market using deep deterministic policy gradient algorithm. IEEE Transactions on Power Systems, 35(6):4180–4192, 2020.

Viktor Makoviychuk, Lukasz Wawrzyniak, Yunrong Guo, Michelle Lu, Kier Storey, Miles Macklin, David Hoeller, Nikita Rudin, Arthur Allshire, Ankur Handa, and Gavriel State. Isaac Gym: High performance GPU-Based physics simulation for robot learning. arXiv:2108.10470, 2021.

Carlos Matamala and Goran Strbac. Strategic bidding in the frequency-containment ancillary services market. Applied Energy, 401:126811, 2025.

Koen Ponse, Aske Plaat, Niki van Stein, and Thomas M. Moerland. Econojax: A fast & scalable economic simulation in jax. In AAMAS ’25: Proceedings of the 24th International Conference on Autonomous Agents and Multiagent Systems, pp. 1679–1687, Richland, SC, 2025.

Waris Radji, Thomas Michel, and Hector Piteau. Octax: Accelerated chip-8 arcade environments for reinforcement learning in jax. In International Conference on Learning Representations, volume 2026, pp. 72220–72256, 2026.

Antonin Raffin, Ashley Hill, Adam Gleave, Anssi Kanervisto, Maximilian Ernestus, and Noah Dormann. Stable-Baselines3: Reliable reinforcement learning implementations. Journal ofMachine Learning Research, 22(268):1–8, 2021.

Alexander Rutherford, Benjamin Ellis, Matteo Gallici, Jonathan Cook, Andrei Lupu, Gardar Ingvarsson, Timon Willi, Ravi Hammond, Akbir Khan, Christian Schroeder de Witt, Alexandra Souly, Saptarashmi Bandyopadhyay, Mikayel Samvelyan, Minqi Jiang, Robert Lange, Shimon Whiteson, Bruno Lacerda, Nick Hawes, Tim Rocktaschel, Chris Lu, and Jakob Foerster. Jax-¨ MARL: Multi-Agent RL environments and algorithms in JAX. In Advances in Neural Information Processing Systems 37 (NeurIPS 2024), Datasets and Benchmarks Track, pp. 50925–50951, 2024.

Nelson Salazar-Pena, Alejandra Tabares, and Andr˜ es Gonz´ alez-Mancera. MARLEM: A multi-agent´ reinforcement learning simulation framework for implicit cooperation in decentralized local energy markets. Applied Energy, 412:127546, 2026.

Thomas Wolgast and Astrid Nieße. Approximating energy market clearing and bidding with Model-Based reinforcement learning. IEEE Access, 12:145106–145117, 2024.

Christopher Yeh, Victor Li, Rajeev Datta, Julio Arroyo, Nicolas Christianson, Chi Zhang, Yize Chen, Mohammad Mehdi Hosseini, Azarang Golmohammadi, Yuanyuan Shi, Yisong Yue, and Adam Wierman. SustainGym: Reinforcement learning environments for sustainable energy systems. In Advances in Neural Information Processing Systems 36, pp. 59464–59476, 2023.

Kevin Zakka, Baruch Tabanpour, Qiayuan Liao, Mustafa Haiderbhai, Samuel Holt, Jing Luo, Arthur Allshire, Erik Frey, Koushil Sreenath, Lueder Kahrs, Carmelo Sferrazza, Yuval Tassa, and Pieter Abbeel. Demonstrating MuJoCo Playground. In Robotics: Science and Systems XXI, 2025.

Lorenzo Zapparoli, Alfredo Oneto, Mar´ıa Parajeles Herrera, Blazhe Gjorgiev, Gabriela Hug, and Giovanni Sansavini. Future deployment and flexibility of distributed energy resources in the distribution grids of Switzerland. Scientific Data, 12:1491, 2025.

Ziqing Zhu, Siqi Bu, Ka Wing Chan, Fangxing Li, Yujian Ye, Chi Yung Chung, and Goran Strbac. Designing the future electricity spot market with high renewables via reliable simulations. Nature Reviews Electrical Engineering, 2(5):320–337, 2025.

## APPENDIX CONTENTS

A Market Models and Partially Observable Stochastic Game 13   
A.1 M1: Day-ahead Wholesale Market 13   
A.2 M2: Real-Time Balancing Market 17   
A.3 M3: Ancillary Services Market . 20   
A.4 M4: Peer-to-Peer Local Energy Market 24   
A.5 M5: Local Flexibility Market . 28   
B Computation Performance 33   
B.1 Three layouts on the peer-to-peer market . 33   
B.2 Across the five markets 33   
B.3 Against open-source tools . 33   
B.4 Implementation validation 34   
C Quickstart and API 37   
C.1 Installation 37   
C.2 Minimal rollout 37   
C.3 The environment interface 39   
C.4 Constructing the five markets . 39   
C.5 Scenario and learner configuration 40   
C.6 Command-line usage 40   
C.7 Cases and data . 41   
C.8 Clearing and learning in a single computation graph . 41   
D Training Setup and Hyperparameters 44   
D.1 Four learners 44   
D.2 Hyperparameters 44   
D.3 Training scale 45   
D.4 Evaluation protocol 45   
E Additional Market Results 46   
E.1 M1: Day-ahead Wholesale Market 46   
E.2 M2: Real-Time Balancing Market 54   
E.3 M3: Ancillary Services Market . 58   
E.4 M4: Peer-to-Peer Local Energy Market 61   
E.5 M5: Local Flexibility Market . 64

## A MARKET MODELS AND PARTIALLY OBSERVABLE STOCHASTIC GAME

## A.1 M1: DAY-AHEAD WHOLESALE MARKET

The day-ahead market is a centralized auction held on day $d - 1$ that clears all T hourly delivery periods of day d simultaneously, with $\Delta ^ { \mathrm { d a } } = 1$ hour. Market participants submit their strategic bids, which the market operator clears through a security-constrained unit commitment (SCUC) problem. The resulting schedules are priced and settled using locational marginal prices (LMPs). Section A.1.1 presents the market clearing model, section A.1.2 introduces the pricing and settlement rules, and section A.1.3 formulates the partially observable stochastic game (POSG) for strategic bidding in day-ahead wholesale market.

## A.1.1 MARKET MODEL

Each generator agent i submits, for each delivery period t, a stepwise supply curve consisting of K segments. Each segment k has quantity

$$
G _ { i , k } = \frac { \bar { p } _ { i } - \underline { { { p } } } _ { i } } { K }\tag{1}
$$

and bid price $\pi _ { i , k , t }$ , where $\underline { { p } } _ { i }$ and ${ \bar { p } } _ { i }$ are the minimum and maximum outputs of the unit. Bid prices are required to be non-decreasing across segments, i.e., $\pi _ { i , k , t } \ \leq \ \pi _ { i , k + 1 , t } .$ . Technical parameters, including ramp rates, minimum up- and down-times, start-up costs, and no-load costs, are registered with the market operator and are not part of the per-period bid. Demand is inelastic and treated as exogenous.

The day-ahead market jointly determines unit commitment and dispatch over all T hourly delivery periods by solving

$$
\operatorname* { m i n } _ { \{ g , u , v , w , s \} } \sum _ { t = 1 } ^ { T } \left[ \sum _ { i , k } \pi _ { i , k , t } g _ { i , k , t } + \sum _ { i } \Delta ^ { \mathrm { d a } } \mathrm { N L } _ { i } u _ { i , t } + \mathrm { V O L L } \sum _ { n } s _ { n , t } \right] + \sum _ { i , t } S _ { i } v _ { i , t } ,\tag{2a}
$$

subject to

$$
p _ { i , t } = \underline { { p } } _ { i } u _ { i , t } + \sum _ { k } g _ { i , k , t } ,\tag{2b}
$$

$$
0 \leq g _ { i , k , t } \leq G _ { i , k } u _ { i , t } ,\tag{2c}
$$

$$
P _ { n , t } = \sum _ { i : \mathrm { b u s } ( i ) = n } p _ { i , t } + s _ { n , t } - d _ { n , t } ,\tag{2d}
$$

$$
\sum _ { n } P _ { n , t } = 0 \quad ( : \lambda _ { t } ) ,\tag{2e}
$$

$$
- F \le \mathrm { P T D F } \cdot P _ { t } \le F \quad ( : \mu _ { t } ^ { + } , \mu _ { t } ^ { - } ) ,\tag{2f}
$$

$$
0 \leq s _ { n , t } \leq d _ { n , t } \qquad ( : \rho _ { n , t } ) ,\tag{2g}
$$

$$
- R _ { i } ^ { \mathrm { { d n } } } \Delta ^ { \mathrm { { d a } } } - \bar { p } _ { i } w _ { i , t } \ \leq \ p _ { i , t } - p _ { i , t - 1 } \ \leq \ R _ { i } ^ { \mathrm { { u p } } } \Delta ^ { \mathrm { { d a } } } + \underline { { p } } _ { i } v _ { i , t } ,\tag{2h}
$$

$$
u _ { i , t } - u _ { i , t - 1 } = v _ { i , t } - w _ { i , t } ,\tag{2i}
$$

$$
v _ { i , t } + w _ { i , t } \leq 1 ,\tag{2j}
$$

$$
\sum _ { \tau = t - \mathrm { U T } _ { i } + 1 } ^ { t } v _ { i , \tau } \leq u _ { i , t } ,\tag{2k}
$$

$$
\sum _ { \tau = t - \mathrm { D T } _ { i } + 1 } ^ { t } w _ { i , \tau } \leq 1 - u _ { i , t } ,\tag{2l}
$$

$$
u _ { i , t } , v _ { i , t } , w _ { i , t } \in \{ 0 , 1 \} .\tag{2m}
$$

The objective (2a) minimizes the total offered operating cost over the delivery day. The first term represents the cost of accepted energy offers, the second accounts for the no-load cost of committed generators, and the third penalizes involuntary load shedding. The start-up cost $S _ { i }$ is incurred whenever generator i transitions from off to on. Here, $\Delta ^ { \mathrm { d a } } = 1$ h is the duration of each day-ahead delivery period, $\mathrm { N L } _ { i }$ is the no-load cost, and VOLL is the value of lost load, which is set sufficiently high so that load shedding is used only as a last resort.

Constraints (2b) and (2c) define the output of each generator. The binary variable $u _ { i , t }$ indicates whether generator i is committed. When $u _ { i , t } = 1$ , its output consists of the minimum generation level $\underline { { p } } _ { i }$ plus the accepted quantities $g _ { i , k , \ i }$ <sub>t</sub> from its offer segments. Each accepted segment is bounded by $G _ { i , k }$ , so the total output remains within $[ \underline { { p } } _ { i } , \bar { p } _ { i } ]$ . When $u _ { i , t } = 0$ , all segment quantities and the generator output are zero.

Constraints (2d)–(2f) impose the network constraints. Equation (2d) defines the net injection $P _ { n , t }$ at bus n as local generation plus load shedding minus exogenous demand $d _ { n , t }$ . Constraint $( 2 \mathrm { e } )$ enforces system-wide supply–demand balance. Under the $\mathrm { D } \bar { \mathrm { C } }$ power-flow approximation, constraint (2f) maps the nodal injections to transmission-line flows through the power transfer distribution factor matrix $\mathrm { P T D F } \in \mathbf { \bar { \mathbb { R } } } ^ { N ^ { \mathrm { l i n e } } \times N ^ { \mathrm { b u s } } }$ and limits each line flow by its rating F. Network losses are not modeled.

Constraint $( 2 \mathrm { g } )$ bounds load shedding at each bus by the local demand. The variable $s _ { n , t }$ provides emergency feasibility when available generation and transmission capacity are insufficient to serve all demand. Because load shedding carries the large penalty VOLL in the objective, it is used only when lower-cost feasible alternatives are unavailable.

Constraint (2h) limits the change in generator output between consecutive delivery periods according to the upward and downward ramp rates $R _ { i } ^ { \mathrm { u p } }$ and $R _ { i } ^ { \mathrm { { d n } } }$ . The start-up term $\underline { { p } } _ { i } v _  i , $ <sub>t</sub> allows a unit that starts in period t to move from zero output to at least its minimum generation level, while the shut down term $\bar { p } _ { i } w _ { i , t }$ allows a unit that shuts down to reduce its output to zero.

Constraints (2i) and (2j) describe commitment transitions. The binary variables $v _ { i , t }$ and $w _ { i , t }$ denote start-up and shut-down events, respectively. Equation (2i) links these events to changes in commitment status, while equation (2j) prevents a generator from starting up and shutting down in the same period.

Constraints (2k) and (2l) enforce the minimum up- and down-times $\mathrm { U T } _ { i }$ and $\mathrm { D T } _ { i }$ . After a start-up, the unit must remain committed for at least $\mathrm { U T } _ { i }$ periods; after a shut-down, it must remain offline for at least $\mathrm { D T } _ { i }$ periods. The initial commitment status $u _ { i , 0 }$ , initial output $p _ { i , 0 }$ , and any residual minimum up- or down-time requirements from the previous day are treated as exogenous initial conditions.

Solution method. Optimization (2) is a security-constrained unit commitment (SCUC) problem formulated as a mixed-integer linear program (MILP). Exact MILP solvers rely on dynamic branchand-bound procedures that are difficult to batch and compile efficiently on accelerators. We therefore use a relax-round-resolve procedure. First, the binary commitment variables are relaxed to $u _ { i , t } \in [ 0 , 1 ]$ , and the resulting linear program is solved. Second, the relaxed commitment is rounded by setting $u _ { i , t } ^ { \mathrm { i n t } } = 1$ if $u _ { i , t } > \varepsilon$ and $u _ { i , t } ^ { \mathrm { i n t } } = 0$ otherwise, from which the start-up and shut-down indicators $v _ { i , t } ^ { \mathrm { i n t } }$ and $w _ { i , t } ^ { \mathrm { i n t } }$ are determined. Third, these commitment decisions are fixed and the continuous dispatch problem is re-solved as a security-constrained economic dispatch (SCED). Dispatch schedules are obtained from this final solve, while LMPs are computed from the dual variables of its power balance and transmission constraint. This separation follows the common practice of determining commitment first and obtaining dispatch and prices with commitment fixed. The optimality gap relative to the exact solution is evaluated offline and reported in Appendix B.4.

## A.1.2 PRICING AND SETTLEMENT

Given the fixed commitment obtained from the solution procedure above, LMPs are computed from the dual variables of the final SCED problem. The LMP at bus n at period t is the marginal cost of serving one additional MWh at that bus and is given by

$$
\mathrm { L M P } _ { n , t } = \lambda _ { t } - \sum _ { l } \left( \mu _ { l , t } ^ { + } - \mu _ { l , t } ^ { - } \right) \mathrm { P T D F } _ { l , n } - \rho _ { n , t } ,\tag{3a}
$$

where the first term is the system marginal energy price, which is common across the transmission network; the second term is the congestion component, which is zero when no transmission

constraint is binding and causes nodal prices to differ under congestion; and the third term can be non-zero only when the upper bound on load curtailment is binding. There is no loss component because transmission losses are not modeled.

Cleared quantities $p _ { i , t }$ are settled at the LMP of the bus where unit i is located. The revenue, cost, and profit of unit i over the market day are

$$
\mathrm { r e v e n u e } _ { i } = \sum _ { t } \Delta ^ { \mathrm { d a } } \cdot \mathrm { L M P } _ { \mathrm { b u s } ( i ) , t } p _ { i , t } ,\tag{3b}
$$

$$
\mathrm { c o s t } _ { i } = \sum _ { t } \left[ \Delta ^ { \mathrm { d a } } \cdot \mathrm { T C } _ { i } ( p _ { i , t } ) + \Delta ^ { \mathrm { d a } } \cdot \mathrm { N L } _ { i } u _ { i , t } \right] + \sum _ { t } S _ { i } v _ { i , t } ,\tag{3c}
$$

$$
{ \mathrm { p r o f i t } } _ { i } = { \mathrm { r e v e n u e } } _ { i } - { \mathrm { c o s t } } _ { i } ,\tag{3d}
$$

where $\begin{array} { r } { \mathrm { T C } _ { i } ( p ) = \int _ { 0 } ^ { p } \mathrm { M C } _ { i } ( x ) } \end{array}$ dx is the true variable generation cost rate of unit $i ,$ and $\mathrm { M C } _ { i } ( p ) =$ $a _ { i } p ^ { 2 } + b _ { i } p + c _ { i }$ is its true marginal cost. Setting $a _ { i } = 0$ recovers a linear marginal cost curve. The true variable generation cost $\mathrm { T C } _ { i } ( \cdot )$ is used only to evaluate participant profit and does not enter market clearing; the clearing problem instead uses submitted energy offer prices together with the registered no-load and start-up costs. Settlement does not include uplift payment, so a committed unit may earn negative profit when its energy market revenue does not recover its start-up and no-load costs.

## A.1.3 PARTIALLY OBSERVABLE STOCHASTIC GAME

We formulate repeated participation in the day-ahead wholesale market as a general-sum POSG

$$
\mathcal { G } ^ { \mathrm { D A } } = \langle \mathcal { T } , \mathcal { S } , \{ \mathcal { O } _ { i } \} _ { i \in \mathcal { I } } , \{ A _ { i } \} _ { i \in \mathcal { I } } , P , \{ r _ { i } \} _ { i \in \mathcal { I } } , \gamma \rangle ,\tag{4}
$$

where $\mathcal { T }$ is the set of strategic generator agents, $s$ is the market state space, $\mathcal { O } _ { i }$ and $A _ { i }$ are the observation and action spaces of agent $i , P$ is the state transition kernel, $r _ { i }$ is the individual profitbased reward, and $\gamma \in \ [ 0 , 1 ]$ is the discount factor. One episodic market step corresponds to one day-ahead market clearing, in which all T delivery periods of the following day are cleared jointly.

State. At market day $d ,$ the underlying state contains the system information required to clear the following delivery day,

$$
s _ { d } = ( x _ { d } ^ { \mathrm { s y s } } , \{ x _ { i , d } \} _ { i \in \mathcal { T } } ) \in \mathcal { S } ,\tag{5}
$$

where $x _ { d } ^ { \mathrm { s y s } }$ contains the day-ahead forecast of system demand $d _ { n , t , d }$ and other exogenous system conditions such as renewable generation. The generator state is

$$
x _ { i , d } = \left( u _ { i , d } ^ { 0 } , p _ { i , d } ^ { 0 } , \tau _ { i , d } ^ { \mathrm { u p } } , \tau _ { i , d } ^ { \mathrm { d n } } \right) ,\tag{6}
$$

where $u _ { i , d } ^ { 0 }$ and $p _ { i , d } ^ { 0 }$ are the commitment status and power output of generator i entering the delivery day, and $\tau _ { i , d } ^ { \mathrm { u p } }$ and $\tau _ { i , d } ^ { \mathrm { d n } }$ denote any residual minimum up- and downtime requirements carried over from the previous day. Other system and generator parameters, such as the PTDF matrix PTDF, transmission limits $\dot { F }$ , generation capacity limits $( \underline { { p } } _ { i } , G _ { i , k } )$ , and ramp rates $( R _ { i } ^ { \mathrm { u p } } , R _ { i } ^ { \mathrm { d n } } )$ , are fixed environment parameters known to the market operator.

Observation. Each generator i observes its own private information and the public information available before bidding,

$$
o _ { i , d } \in \mathcal { O } _ { i } ( s _ { d } ) = \Big ( x _ { i , d } , x _ { d } ^ { \mathrm { p u b } } \Big ) ,\tag{7}
$$

where $x _ { d } ^ { \mathrm { p u b } }$ includes the public day-ahead system information of demand forecast $d _ { n , t , d }$ . An agent does not observe the private generation costs or contemporaneous offers of competing generators before market clearing. Hence, $o _ { i , d }$ does not fully reveal $s _ { d } .$ , making the bidding problem partially observable.

Action. The action of generator i is a single markup $\alpha _ { i , d }$ on its truthful offer $\pi _ { i , 1 } ^ { 0 } \leq \cdots \leq \pi _ { i , K } ^ { 0 } .$ in which each segment is priced at the unit’s true marginal cost $\mathrm { M C } _ { i } ,$ , giving its complete day-ahead price bid over all segments and delivery periods,

$$
a _ { i , d } = \alpha _ { i , d } \in \mathcal { A } _ { i } = [ 1 , \bar { \alpha } ] , \qquad \pi _ { i , k , t , d } = \alpha _ { i , d } \pi _ { i , k } ^ { 0 } , \quad k = 1 , \ldots , K ; \ t = 1 , \ldots , T ,\tag{8}
$$

where $\bar { \alpha }$ is the largest admissible markup. The segment quantities $G _ { i , k }$ are fixed as defined in Section A.1.1, while the agent chooses the markup. Admissible bids satisfy the market price bounds and the monotonicity rule

$$
\underline { { \pi } } \le \pi _ { i , 1 , t , d } \le \pi _ { i , 2 , t , d } \le \cdots \le \pi _ { i , K , t , d } \le \overline { { \pi } } , \qquad \forall t .\tag{9}
$$

The joint action $a _ { d } = \left( a _ { 1 , d } , \dotsc , a _ { | \mathcal { T } | , d } \right)$ therefore represents the set of supply bids submitted by all strategic generators for the following delivery day.

Market clearing and outcome. Given state $s _ { d }$ and joint action $^ { a } d \cdot$ the market operator applies the day-ahead clearing model and pricing rules described in Sections $\mathbf { A } . 1 . 1 \mathbf { - A } . 1 . 2$ . We denote this mapping by

$$
z _ { d } = \mathcal { C } ^ { \mathrm { D A } } ( s _ { d } , a _ { d } ) ,\tag{10}
$$

where the market outcome $z _ { d }$ contains the commitment decisions $u _ { i , 1 \dots T , d } ,$ accepted generation quantities $g _ { i , k , 1 \ldots T , d } .$ , dispatch schedules $p _ { i , 1 \ldots T , d } .$ , load curtailment $s _ { n , 1 \ldots T , d } ,$ , and nodal LMPs $\mathrm { L M P } _ { n , 1 \ldots T , d }$ for all $T$ delivery periods. Through this clearing mechanism, the payoff of one agent depends not only on its own offer but also on the bids submitted by the other agents.

Reward. Each agent i receives its realized market profit as reward. Using the settlement defined in Section A.1.2,

$$
\begin{array} { l } { { \displaystyle r _ { i , d } = \mathrm { p r o f i t } _ { i , d } = \mathrm { r e v e n u e } _ { i } - \mathrm { c o s t } _ { i } } } \\ { { \displaystyle \quad = \sum _ { t = 1 } ^ { T } \Delta ^ { \mathrm { d a } } \mathrm { L M P } _ { \mathrm { b u s } ( i ) , t , d } p _ { i , t , d } } } \\ { { \displaystyle \quad - \sum _ { t = 1 } ^ { T } \left[ \Delta ^ { \mathrm { d a } } \mathrm { T C } _ { i } ( p _ { i , t , d } ) + \Delta ^ { \mathrm { d a } } \mathrm { N L } _ { i } u _ { i , t , d } \right] - \sum _ { t = 1 } ^ { T } S _ { i } v _ { i , t , d } } . }  \end{array}\tag{11}
$$

Thus, submitted bidding prices affect the reward only through the resulting market-clearing outcomes; the true variable generation cost $\mathrm { T C } _ { i } ( \cdot )$ is used only to evaluate the realized cost and then profit. Since each generator maximizes its own profit rather than a common system-wide reward, the resulting POSG is general-sum.

State transition. After the delivery periods cleared at market step d, the terminal operating condition of each generator is carried into the next market step. Specifically, the initial commitment status and power output for the next day are

$$
u _ { i , d + 1 } ^ { 0 } = u _ { i , T , d } , \qquad p _ { i , d + 1 } ^ { 0 } = p _ { i , T , d } ,\tag{12}
$$

while the residual minimum up- or downtime requirements $\tau _ { i , d + 1 } ^ { \mathrm { u p } }$ and $\tau _ { i , d + 1 } ^ { \mathrm { d n } }$ are updated according to the terminal commitment trajectory. Hence, the next generator state

$$
x _ { i , d + 1 } = \left( u _ { i , d + 1 } ^ { 0 } , p _ { i , d + 1 } ^ { 0 } , \tau _ { i , d + 1 } ^ { \mathrm { u p } } , \tau _ { i , d + 1 } ^ { \mathrm { d n } } \right)\tag{13}
$$

depends on the commitment and dispatch outcomes produced by the joint bids at market step d.

The exogenous system state $x _ { d + 1 } ^ { \mathrm { s y s } }$ , including the next day-ahead forecasts of demand, evolves according to the underlying data process. The overall state transition is therefore

$$
s _ { d + 1 } \sim P ( \cdot \mid s _ { d } , a _ { d } ) ,\tag{14}
$$

where the dependence on $ { a _ { d } }$ arises through the cleared commitment and dispatch decisions carried into the next market step.

Agent objective. Each generator uses a policy $\sigma _ { i } ( a _ { i , d } \mid o _ { i , d } )$ to choose its day-ahead bid and seeks to maximize its expected discounted profit over an episode of H market days,

$$
J _ { i } ( \sigma _ { i } , \sigma _ { - i } ) = \mathbb { E } \left[ \sum _ { d = 1 } ^ { H } \gamma ^ { d - 1 } r _ { i , d } \right] .\tag{15}
$$

where $\sigma _ { - i }$ denotes the joint policies of all other generators. Because market clearing depends on the joint bids of all generators, the expected return of agent i depends on both its own policy $\sigma _ { i }$ and the policies of the other agents $\sigma _ { - i } .$ . Each agent therefore learns its bidding policy through repeated market interactions without observing competitors’ private information or policies.

## A.2 M2: REAL-TIME BALANCING MARKET

The real-time balancing market consists of a sequence of auctions during delivery day $D ,$ with one auction for each real-time interval $\Delta ^ { \mathrm { r t } } = 3 0 ^ { \circ }$ mins. It takes the commitment and schedules established by the day-ahead wholesale market as fixed and redispatches available resources against realized system demand. Only deviations from the day-ahead schedules are settled at real-time LMPs, while the day-ahead schedules remain settled at day-ahead LMPs. This section presents the real-time balancing market clearing model in subsection A.2.1, the two-settlement rule in subsection A.2.2, and the corresponding POSG in subsection A.2.3.

## A.2.1 MARKET MODEL

Let τ index the 30-minute real-time intervals and let $h ( \tau )$ denote the corresponding hourly period in the day-ahead market. The day-ahead outcome enters the real-time market as fixed data through the commitment $u _ { i , h ( \tau ) } ^ { \mathrm { d a } }$ , scheduled generation $q _ { i , h ( \tau ) } ^ { \mathrm { d a } }$ , and day-ahead price $\mathrm { L M P } _ { n , h ( \tau ) } ^ { \mathrm { d a } }$ . The realized net demand at bus n is $\tilde { d } _ { n , \tau }$ . Generators submit price–quantity bids with the same structure as in the day-ahead market, and one market clearing problem is solved for each real-time interval:

$$
\operatorname* { m i n } _ { \left\{ g , s \right\} } \sum _ { i , k } \pi _ { i , k , \tau } g _ { i , k , \tau } + \mathrm { V O L L } \sum _ { n } s _ { n , \tau } ,\tag{16a}
$$

subject to

$$
p _ { i , \tau } = \underline { { p } } _ { i } u _ { i , h ( \tau ) } ^ { \mathrm { d a } } + \sum _ { k } g _ { i , k , \tau } ,\tag{16b}
$$

$$
0 \leq g _ { i , k , \tau } \leq G _ { i , k } u _ { i , h ( \tau ) } ^ { \mathrm { d a } } ,\tag{16c}
$$

$$
P _ { n , \tau } = \sum _ { i : \mathrm { b u s } ( i ) = n } p _ { i , \tau } + s _ { n , \tau } - \tilde { d } _ { n , \tau } ,\tag{16d}
$$

$$
\sum _ { n } \ : P _ { n , \tau } = 0 \ : \ : \ : \ : \ : ( : \lambda _ { \tau } ) ,\tag{16e}
$$

$$
- F \le \mathrm { P T D F } \cdot P _ { \tau } \le F \qquad ( : \mu _ { \tau } ^ { + } , \mu _ { \tau } ^ { - } ) ,\tag{16f}
$$

$$
\begin{array} { r } { 0 \leq s _ { n , \tau } \leq \operatorname* { m a x } ( \tilde { d } _ { n , \tau } , 0 ) \quad ( : \rho _ { n , \tau } ) , } \end{array}\tag{16g}
$$

$$
\begin{array} { r } { - R _ { i } ^ { \mathrm { { d n } } } \Delta ^ { \mathrm { r t } } - \bar { p } _ { i } w _ { i , \tau } ^ { \mathrm { { d a } } } \leq p _ { i , \tau } - p _ { i , \tau - 1 } \leq R _ { i } ^ { \mathrm { u p } } \Delta ^ { \mathrm { r t } } + \underline { { p } } _ { i } v _ { i , \tau } ^ { \mathrm { { d a } } } . } \end{array}\tag{16h}
$$

The objective minimizes the as-bid redispatch cost and load-shedding penalty for each real-time interval $\Delta ^ { \mathrm { { r t } } }$ . Unlike the day-ahead market, unit commitment is not re-optimized: $u _ { i , h ( \tau ) } ^ { \mathrm { d a } }$ is fixed by the day-ahead outcome, and the real-time market only adjusts the dispatch of committed units. The nodal balance and transmission constraints ensure that the resulting dispatch remains physically feasible under realized net demand. Consecutive real-time intervals are coupled through the ramping constraint, where $p _ { i , \tau - 1 }$ is the realized dispatch from the previous interval. The start-up and shutdown indicators $v _ { i , \tau } ^ { \mathrm { d a } }$ and $w _ { i , \tau } ^ { \mathrm { d a } }$ are determined by changes in the mapped day-ahead commitment and are non-zero only when the commitment status changes between consecutive real-time intervals.

Constraint (16b) defines the real-time output of generator i. Because the commitment $u _ { i , h ( \tau ) } ^ { \mathrm { d a } }$ is fixed by the day-ahead market, only generators committed in the corresponding day-ahead period can produce in real time. Their output equals the minimum generation level plus the accepted quantities from the submitted bid segments. Constraint (16c) limits each accepted segment $g _ { i , k , \tau }$ to its available quantity $G _ { i , k }$ . Constraint (16d) defines the net injection $P _ { n , \tau } \mathrm { a t }$ bus n as local generation plus load shedding minus realized net demand. Constraint (16e) enforces system-wide supply–demand balance, while constraint (16f) keeps transmission flows within the line ratings. The associated dual variables $\lambda _ { \tau } , \mu _ { \tau } ^ { + }$ , and $\mu _ { \tau } ^ { - }$ are used to determine the real-time LMPs.

Constraint (16g) allows emergency load curtailment only up to the positive net demand at each bus. Hence, $s _ { n , \tau } = 0$ when renewable generation exceeds local demand. The dual variable $\rho _ { n , \tau }$ is associated with the upper bound on load shedding. Constraint (16h) limits changes in generator output between consecutive real-time intervals according to the upward and downward ramp rates.

Here, $p _ { i , \tau - 1 }$ is the realized dispatch from the previous real-time interval. The fixed indicators $v _ { i , \tau } ^ { \mathrm { d a } }$ and $w _ { i , \tau } ^ { \mathrm { d a } }$ account for start-up and shut-down transitions implied by the day-ahead commitment.

Unlike the day-ahead market, the real-time market does not re-optimize unit commitment. The day-ahead commitment is fixed, and the real-time clearing adjusts only the dispatch quantities in response to realized system conditions. Consecutive real-time intervals are coupled through generator ramping.

## A.2.2 PRICING AND SETTLEMENT

Real-time LMPs are computed from the dual variables of the real-time clearing problem using the same energy–congestion decomposition as in equation (3a):

$$
\mathrm { L M P } _ { n , \tau } ^ { \mathrm { r t } } = \lambda _ { \tau } - \sum _ { l } \left( \mu _ { l , \tau } ^ { + } - \mu _ { l , \tau } ^ { - } \right) \mathrm { P T D F } _ { l , n } - \rho _ { n , \tau } .\tag{17}
$$

Unlike day-ahead LMPs, the real-time LMPs reflect realized demand and short-term operating constraints. They can therefore be more volatile and may become negative under local oversupply or binding downward-flexibility constraints.

The market follows a two-settlement rule. The day-ahead schedule remains settled at the day-ahead LMP, while only the deviation from that schedule is settled at the real-time LMP. For generator i in real-time interval τ,

$$
\mathrm { r e v e n u e } _ { i , \tau } ^ { \mathrm { d a } } = \Delta ^ { \mathrm { r t } } \mathrm { L M P } _ { \mathrm { b u s } ( i ) , h ( \tau ) } ^ { \mathrm { d a } } q _ { i , h ( \tau ) } ^ { \mathrm { d a } } ,\tag{18a}
$$

$$
\mathrm { r e v e n u e } _ { i , \tau } ^ { \mathrm { r t } } = \Delta ^ { \mathrm { r t } } \mathrm { L M P } _ { \mathrm { b u s } ( i ) , \tau } ^ { \mathrm { r t } } \left( p _ { i , \tau } - q _ { i , h ( \tau ) } ^ { \mathrm { d a } } \right) ,\tag{18b}
$$

$$
\mathrm { c o s t } _ { i , \tau } = \Delta ^ { \mathrm { r t } } \mathrm { T C } _ { i } ( p _ { i , \tau } ) + \Delta ^ { \mathrm { r t } } \mathrm { N L } _ { i } u _ { i , h ( \tau ) } ^ { \mathrm { d a } } + S _ { i } v _ { i , \tau } ^ { \mathrm { d a } } ,\tag{18c}
$$

$$
\mathrm { p r o f i t } _ { i , \tau } = \mathrm { r e v e n u e } _ { i , \tau } ^ { \mathrm { d a } } + \mathrm { r e v e n u e } _ { i , \tau } ^ { \mathrm { r t } } - \mathrm { c o s t } _ { i , \tau } .\tag{18d}
$$

In this context, a generator producing above its day-ahead schedule sells the positive deviation at the real-time LMP, while a generator producing below its schedule effectively buys back the shortfall at the real-time LMP. The day-ahead revenue, commitment status, and associated no-load and start-up costs are fixed by the day-ahead outcome and do not depend on the real-time bid. They are retained in the reward so that the reported profit represents the full economic outcome of delivery.

Remark on the day-ahead and real-time coupling. In practice, the two markets operate at different temporal scales: the day-ahead market is cleared on day $D - 1$ , while the real-time market is cleared at each interval of delivery day D. The commitment $u _ { i , h ( \tau ) } ^ { \mathrm { d a } } ,$ binding schedule $q _ { i , h ( \tau ) } ^ { \mathrm { d a } }$ , and dayahead price $\mathrm { L M P } _ { n , h ( \tau ) } ^ { \mathrm { d a } }$ are fixed parameters of the real-time clearing, while the real-time market prices deviations arising from uncertainty not captured by the day-ahead forecast.

## A.2.3 PARTIALLY OBSERVABLE STOCHASTIC GAME

The real-time balancing market can be also formulated as a general-sum partially observable stochastic game (POSG),

$$
\mathcal { G } ^ { \mathrm { r t } } = \langle \mathcal { T } , \mathcal { S } , \{ \mathcal { O } _ { i } \} _ { i \in \mathcal { I } } , \{ A _ { i } \} _ { i \in \mathcal { I } } , P , \{ r _ { i } \} _ { i \in \mathcal { I } } , \gamma \rangle ,\tag{19}
$$

where I is the set of strategic generators. One episodic market step τ corresponds to one 30-minute real-time interval $\Delta ^ { \mathrm { { r t } } }$ . For a 24-hour delivery day, the real-time market therefore contains $T ^ { \mathrm { r t } } =$ 48 sequential clearing steps.

State. The state at interval τ is

$$
s _ { \tau } = \left( x _ { \tau } ^ { \mathrm { s y s } } , x _ { \tau } ^ { \mathrm { d a } } , \{ x _ { i , \tau } \} _ { i \in \mathcal { I } } \right) \in \mathcal { S } ,\tag{20}
$$

where $x _ { \tau } ^ { \mathrm { s y s } }$ contains the realized system conditions used for real-time clearing, i.e., the net bus demand $\tilde { d } _ { n , \tau }$ . The day-ahead component is

$$
\begin{array} { r } { x _ { \tau } ^ { \mathrm { d a } } = \left\{ \boldsymbol { u } _ { i , h ( \tau ) } ^ { \mathrm { d a } } , \boldsymbol { q } _ { i , h ( \tau ) } ^ { \mathrm { d a } } , \mathrm { L M P } _ { n , h ( \tau ) } ^ { \mathrm { d a } } \right\} _ { i , n } , } \end{array}\tag{21}
$$

which contains the fixed commitment $u _ { i , h ( \tau ) } ^ { \mathrm { d a } }$ , binding generation schedules $q _ { i , h ( \tau ) } ^ { \mathrm { d a } }$ , and nodal prices $\mathrm { L M P } _ { n , h ( \tau ) } ^ { \mathrm { d a } }$ from the corresponding day-ahead period. The operating state of generator i is

$$
x _ { i , \tau } = \left( p _ { i , \tau - 1 } , v _ { i , \tau } ^ { \mathrm { d a } } , w _ { i , \tau } ^ { \mathrm { d a } } \right) ,\tag{22}
$$

where $p _ { i , \tau - 1 }$ is the realized dispatch from the previous real-time interval $\tau - 1$ , and $v _ { i , \tau } ^ { \mathrm { d a } }$ and $w _ { i , \tau } ^ { \mathrm { d a } }$ are the start-up and shut-down indicators implied by the mapped day-ahead commitment, which are used in the ramping constraint (16h) to account for output changes associated with start-up and shut-down transitions. Network parameters, generator operating limits, and cost parameters are fixed environment parameters and are omitted from the dynamic state.

Observation. Each generator i observes its own operating information together with public market information. Its observation at real-time interval τ is

$$
o _ { i , \tau } = \left( p _ { i , \tau - 1 } , \boldsymbol { u } _ { i , h ( \tau ) } ^ { \mathrm { d a } } , \boldsymbol { q } _ { i , h ( \tau ) } ^ { \mathrm { d a } } , \mathrm { L M P } _ { \mathrm { b u s } ( i ) , h ( \tau ) } ^ { \mathrm { d a } } , \boldsymbol { x } _ { \tau } ^ { \mathrm { p u b } } \right) = \Omega _ { i } ( s _ { \tau } ) \in \mathcal { O } _ { i } ,\tag{23}
$$

where $p _ { i , \tau - 1 }$ is the generator’s realized dispatch in the previous real-time interval $\tau - 1 , u _ { i , h ( \tau ) } ^ { \mathrm { d a } }$ and $q _ { i , h ( \tau ) } ^ { \mathrm { d a } }$ are its day-ahead commitment and schedule, and $\mathrm { L M P _ { b u s } ^ { d a } } ( i ) , h ( \tau )$ is the corresponding day-ahead LMP. The public real-time market information is denoted by $x _ { \tau } ^ { \mathrm { p u b } }$ , which is the realized system conditions of net bus demand $x _ { \tau } ^ { \mathrm { s y s } } = \tilde { d } _ { n , \tau } .$ A generator does not observe the private parameters or contemporaneous bids of other generators. The market is therefore partially observable from the perspective of each participant.

Action. At each real-time interval, generator i submits a stepwise supply bid $\pi _ { i , \tau } ,$ obtained from a single markup $\alpha _ { i , \tau }$ on its truthful offer $\pi _ { i , 1 } ^ { 0 } \leq \cdot \cdot \cdot \leq \pi _ { i , K } ^ { 0 }$ , in which each segment is priced at the unit’s true marginal cost $\mathrm { M C } _ { i }$

$$
a _ { i , \tau } = \alpha _ { i , \tau } \in \mathcal { A } _ { i } = [ 1 , \bar { \alpha } ] , \qquad \pi _ { i , k , \tau } = \alpha _ { i , \tau } \pi _ { i , k } ^ { 0 } , \quad k = 1 , \ldots , K ,\tag{24}
$$

where α¯ is the largest admissible markup, subject to the admissible bid-price bounds and the monotonicity condition

$$
\pi _ { i , 1 , \tau } \leq \pi _ { i , 2 , \tau } \leq \cdot \cdot \cdot \leq \pi _ { i , K , \tau } .\tag{25}
$$

The segment quantities $G _ { i , k }$ are fixed, so the strategic action determines the bid prices associated with these quantities. The joint action is $a _ { \tau } = ( a _ { i , \tau } ) _ { i \in \mathcal { T } }$

Market clearing and reward. Given the current state and joint bids, the market-clearing mechanism

$$
z _ { \tau } = \mathcal { C } ^ { \mathrm { r t } } ( s _ { \tau } , a _ { \tau } )\tag{26}
$$

solves optimization (16) and returns the realized dispatch $p _ { i , \tau } .$ , accepted bid quantities $g _ { i , k , \tau }$ , and real-time $\mathrm { L M P s } \mathrm { L M P } _ { n , \tau } ^ { \mathrm { r t } }$ . Each generator receives its realized profit as the stage reward,

$$
\begin{array} { r l } & { r _ { i , \tau } = \mathrm { p r o f t } _ { i , \tau } = \mathrm { r e v e n u e } _ { i , \tau } ^ { \mathrm { d a } } + \mathrm { r e v e n u e } _ { i , \tau } ^ { \mathrm { r t } } - \mathrm { c o s t } _ { i , \tau } } \\ & { \qquad = \Delta ^ { \mathrm { r t } } \mathrm { L M P } _ { \mathrm { b u s } ( i ) , h ( \tau ) } ^ { \mathrm { d a } } q _ { i , h ( \tau ) } ^ { \mathrm { d a } } + \Delta ^ { \mathrm { r t } } \mathrm { L M P } _ { \mathrm { b u s } ( i ) , \tau } ^ { \mathrm { r t } } \left( p _ { i , \tau } - q _ { i , h ( \tau ) } ^ { \mathrm { d a } } \right) } \\ & { \qquad - \left( \Delta ^ { \mathrm { r t } } \mathrm { T C } _ { i } ( p _ { i , \tau } ) + \Delta ^ { \mathrm { r t } } \mathrm { N L } _ { i } u _ { i , h ( \tau ) } ^ { \mathrm { d a } } + S _ { i } v _ { i , \tau } ^ { \mathrm { d a } } \right) . } \end{array}\tag{27}
$$

where profi $\therefore { } ^ { \intercal } , \tau$ is defined by the two-settlement rule in equation (18). The reward therefore accounts for the fixed day-ahead position, the real-time settlement of deviations, and the realized generation cost.

State transition. The clearing result determines the operating state entering the next real-time interval. In particular, the realized dispatch $p _ { i , \tau }$ becomes the previous-period dispatch in interval $\tau + 1$ , so that

$$
x _ { i , \tau + 1 } = \left( p _ { i , \tau } , v _ { i , \tau + 1 } ^ { \mathrm { d a } } , w _ { i , \tau + 1 } ^ { \mathrm { d a } } \right) .\tag{28}
$$

Thus, the dispatch determined at interval τ enters the ramping constraint of interval $\tau + 1$ . The corresponding day-ahead commitment, schedule, and price are updated deterministically according to $h ( \tau { + } 1 )$ , while realized net demand evolves according to the exogenous data process. The overall state transition is therefore

$$
s _ { \tau + 1 } \sim P \left( \cdot \mid s _ { \tau } , a _ { \tau } \right) .\tag{29}
$$

Agent objective. Each generator uses a policy $\sigma _ { i } ( a _ { i , \tau } \mid o _ { i , \tau } )$ to select its real-time bid. Over a horizon of H real-time intervals, generator i seeks to maximize its expected discounted profit,

$$
J _ { i } ( \sigma _ { i } , \sigma _ { - i } ) = \mathbb { E } \left[ \sum _ { \tau = 1 } ^ { H } \gamma ^ { \tau - 1 } r _ { i , \tau } \right] ,\tag{30}
$$

where $\sigma _ { - i }$ denotes the joint policies of the other generators. Because both dispatch and real-time prices are determined jointly by all submitted bids, the return of each generator depends on both its own bidding policy and those of the other participants. Agents therefore learn their real-time bidding policies through repeated market interactions without observing competitors’ private information or policies.

## A.3 M3: ANCILLARY SERVICES MARKET

The ancillary services market procures operating reserve, where generation capacity is held available to respond to system contingencies. Unlike the energy market, it primarily trades availability: a provider is paid for maintaining reserve capacity that can be activated when needed but may never be called. In some market designs, operating reserve is co-optimized with real-time energy, so energy dispatch and reserve procurement are determined jointly for the same resources subject to shared operating constraints. This section presents the ancillary services market clearing model in subsection A.3.1, the pricing and settlement rules in subsection A.3.2, and the corresponding POSG in subsection A.3.3.

## A.3.1 MARKET MODEL

The market contains $N ^ { \mathrm { p r o d } } = 2$ reserve products distinguished by their response times: a fast product with $\theta _ { 1 } = 1 0$ mins and a slow product with $\theta _ { 2 } = 3 0$ mins. Before bids are submitted, the system operator publishes the reserve requirement for each product, which can be modeled as a fraction of forecast system demand,

$$
d _ { j , \tau } ^ { \mathrm { r e s } } = \beta _ { j } \sum _ { n } d _ { n , \tau } ^ { \mathrm { f o r e c a s t } } ,\tag{31}
$$

where $d _ { n , \tau } ^ { \mathrm { f o r e c a s t } }$ is the published demand forecast at bus n and $\beta _ { j }$ is the requirement fraction for reserve product $j .$

For each real-time interval, generator i submits the energy offer described in section A.2.1, together with a reserve offer price $\bar { \pi } _ { i , j , \tau } ^ { \mathrm { r e s } } \geq 0$ for each product $j ,$ , expressed in \$/MW-h. The energy and reserve markets are cleared jointly as

$$
\operatorname* { m i n } _ { \{ g , r , s , s ^ { \mathrm { r e s } } \} } \sum _ { i , k } \pi _ { i , k , \tau } g _ { i , k , \tau } + \sum _ { i , j } \pi _ { i , j , \tau } ^ { \mathrm { r e s } } r _ { i , j , \tau } + \mathrm { V O L L } \sum _ { n } s _ { n , \tau } + \mathrm { V O L R } \sum _ { j } s _ { j , \tau } ^ { \mathrm { r e s } } ,\tag{32a}
$$

subject to

$$
p _ { i , \tau } = \underline { { p } } _ { i } u _ { i , h ( \tau ) } ^ { \mathrm { d a } } + \sum _ { k } g _ { i , k , \tau } ,\tag{32b}
$$

$$
0 \leq g _ { i , k , \tau } \leq G _ { i , k } u _ { i , h ( \tau ) } ^ { \mathrm { d a } } ,\tag{32c}
$$

$$
P _ { n , \tau } = \sum _ { i : \mathrm { b u s } ( i ) = n } p _ { i , \tau } + s _ { n , \tau } - \tilde { d } _ { n , \tau } ,\tag{32d}
$$

$$
\sum _ { n } \ : P _ { n , \tau } = 0 \ : \ : \ : \ : \ : ( : \lambda _ { \tau } ) ,\tag{32e}
$$

$$
- F \le \mathrm { P T D F } \cdot P _ { \tau } \le F \qquad ( : \mu _ { \tau } ^ { + } , \mu _ { \tau } ^ { - } ) ,\tag{32f}
$$

$$
\begin{array} { r } { 0 \leq s _ { n , \tau } \leq \operatorname* { m a x } ( \tilde { d } _ { n , \tau } , 0 ) \qquad ( : \rho _ { n , \tau } ) , } \end{array}\tag{32g}
$$

$$
- R _ { i } ^ { \mathrm { { d n } } } \Delta ^ { \mathrm { r t } } - \bar { p } _ { i } w _ { i , \tau } ^ { \mathrm { { d a } } } \leq p _ { i , \tau } - p _ { i , \tau - 1 } \leq R _ { i } ^ { \mathrm { u p } } \Delta ^ { \mathrm { r t } } + \underline { { p } } _ { i } v _ { i , \tau } ^ { \mathrm { { d a } } } ,\tag{32h}
$$

$$
0 \leq r _ { i , j , \tau } \leq R _ { i } ^ { \mathrm { u p } } \theta _ { j } ,\tag{32i}
$$

$$
p _ { i , \tau } + \sum _ { j } r _ { i , j , \tau } \leq \bar { p } _ { i } u _ { i , h ( \tau ) } ^ { \mathrm { d a } } \qquad ( : \nu _ { i , \tau } ^ { \mathrm { c a p } } ) ,\tag{32j}
$$

$$
d _ { j , \tau } ^ { \mathrm { r e s } } - \sum _ { i } r _ { i , j , \tau } - s _ { j , \tau } ^ { \mathrm { r e s } } \leq 0 \qquad ( : \lambda _ { j , \tau } ^ { \mathrm { r e s } } ) ,\tag{32k}
$$

$$
0 \leq s _ { j , \tau } ^ { \mathrm { r e s } } \leq d _ { j , \tau } ^ { \mathrm { r e s } } .\tag{32l}
$$

The objective in equation (32a) minimizes the joint as-bid cost of energy dispatch and reserve procurement for each real-time interval $\Delta ^ { \mathrm { r t } }$ , together with penalties for unserved energy and unmet reserve requirements. The first term represents the accepted energy offers, and the second term represents payments associated with reserve capacity held available for contingency response. The penalties VOLL and VOLR are assigned to load shedding and reserve shortage, respectively. We set $\mathrm { V O L R } < \mathrm { V O L L }$ so that, when available capacity is insufficient to satisfy both requirements, the model permits a reserve shortage before shedding firm load.

Constraints (32b)–(32h) describe the energy dispatch and follow the real-time balancing model in section A.2.1. Constraint (32b) defines the output of generator i as its committed minimum output plus the accepted quantities from its energy-offer segments. Constraint (32c) bounds each accepted segment by its available quantity. Constraints (32d) and (32e) define nodal injections and enforce system-wide supply–demand balance, respectively. Constraint (32f) keeps transmission flows within the line ratings, and (32g) allows emergency load shedding when available generation is insufficient. Constraint (32h) limits the change in generator output between consecutive real-time intervals.

Constraints (32i)–(32l) introduce the reserve products. Constraint (32i) limits the reserve award $r _ { i , j , \tau }$ to the amount of additional output that generator i can deliver within the required response time $\theta _ { j }$ . Therefore, a product with a longer response time allows a generator to provide more reserve capacity for a given ramp rate. Constraint (32j) couples energy and reserve procurement. The scheduled energy output and the total reserve capacity awarded to a generator must jointly remain below its committed capacity. Consequently, capacity used to produce energy cannot simultaneously be sold as reserve, creating an opportunity-cost trade-off between the two products. The dual variable $\nu _ { i , \tau } ^ { \mathrm { c a p } }$ represents the marginal value of additional available capacity of generator i. Finally, constraint (32k) requires the total procured capacity for reserve product $j$ to satisfy the published requirement $d _ { i , \tau } ^ { \mathrm { r e s } }$ . The slack variable $s _ { j , \tau } ^ { \mathrm { r e s } }$ represents any unmet reserve requirement and is bounded by constraint (32l). The corresponding dual variable $\lambda _ { j , \tau } ^ { \mathrm { r e s } }$ gives the marginal clearing price of reserve product $j .$

## A.3.2 PRICING AND SETTLEMENT

Energy prices follow the same LMP formulation as in equation (17), evaluated using the dual variables of the joint energy–reserve clearing problem (32). The reserve price of product j is determined by the dual variable $\bar { \lambda _ { j , \tau } ^ { \mathrm { r e s } } }$ associated with the reserve requirement constraint (32k). It represents the marginal cost of increasing the system-wide requirement for reserve product j by one additional unit of capacity.

A key feature of co-optimization is that the opportunity cost of reserving generation capacity is determined within the clearing problem. When the response-time constraint is non-binding, the marginal cost of providing reserve consists of two components: the submitted reserve offer and the opportunity cost of using generation capacity for reserve rather than energy. The latter is represented by the shadow price $\nu _ { i , \tau } ^ { \mathrm { c a \breve { p } } }$ of the shared capacity constraint. Hence, for a marginal reserve provider,

$$
\lambda _ { j , \tau } ^ { \mathrm { r e s } } = \pi _ { i , j , \tau } ^ { \mathrm { r e s } } + \nu _ { i , \tau } ^ { \mathrm { c a p } } .\tag{33}
$$

If the same generator is also marginal in the energy market and no additional operating constraint is binding, the capacity shadow value corresponds to the energy margin,

$$
\nu _ { i , \tau } ^ { \mathrm { c a p } } = \mathrm { L M P } _ { \mathrm { b u s } ( i ) , \tau } ^ { \mathrm { r t } } - \pi _ { i , k , \tau } ,\tag{34}
$$

and therefore

$$
\lambda _ { j , \tau } ^ { \mathrm { r e s } } = \pi _ { i , j , \tau } ^ { \mathrm { r e s } } + \left( \mathrm { L M P } _ { \mathrm { b u s } ( i ) , \tau } ^ { \mathrm { r t } } - \pi _ { i , k , \tau } \right) .\tag{35}
$$

Thus, the reserve price accounts not only for the submitted reserve offer but also for the value of energy production that the marginal provider gives up when capacity is allocated to reserve.

Energy is settled according to the two-settlement rule in equation (18). Reserve capacity is settled separately at the clearing price of each product. For generator i,

$$
\mathrm { r e v e n u e } _ { i , \tau } ^ { \mathrm { r e s } } = \Delta ^ { \mathrm { r t } } \sum _ { j } \lambda _ { j , \tau } ^ { \mathrm { r e s } } \ : r _ { i , j , \tau } ,\tag{36}
$$

and its total profit for the interval is

$$
\mathrm { p r o f t } _ { i , \tau } = \mathrm { r e v e n u e } _ { i , \tau } ^ { \mathrm { d a } } + \mathrm { r e v e n u e } _ { i , \tau } ^ { \mathrm { r t } } + \mathrm { r e v e n u e } _ { i , \tau } ^ { \mathrm { r e s } } - \mathrm { c o s t } _ { i , \tau } .\tag{37}
$$

where

$$
\mathrm { r e v e n u e } _ { i , \tau } ^ { \mathrm { d a } } = \Delta ^ { \mathrm { r t } } \mathrm { L M P } _ { \mathrm { b u s } ( i ) , h ( \tau ) } ^ { \mathrm { d a } } q _ { i , h ( \tau ) } ^ { \mathrm { d a } } ,\tag{38}
$$

$$
\mathrm { r e v e n u e } _ { i , \tau } ^ { \mathrm { r t } } = \Delta ^ { \mathrm { r t } } \mathrm { L M P } _ { \mathrm { b u s } ( i ) , \tau } ^ { \mathrm { r t } } \left( p _ { i , \tau } - q _ { i , h ( \tau ) } ^ { \mathrm { d a } } \right) ,\tag{39}
$$

$$
\mathrm { c o s t } _ { i , \tau } = \Delta ^ { \mathrm { r t } } \mathrm { T C } _ { i } ( p _ { i , \tau } ) + \Delta ^ { \mathrm { r t } } \mathrm { N L } _ { i } u _ { i , h ( \tau ) } ^ { \mathrm { d a } } + S _ { i } v _ { i , \tau } ^ { \mathrm { d a } } .\tag{40}
$$

Reserve availability itself does not consume fuel, so no additional generation cost is assigned to a reserve award unless the reserve is activated. Every cleared MW of the same reserve product is paid the same product price, so a provider whose reserve offer is below the clearing price earns an inframarginal margin.

Remark on energy–reserve co-optimization. Constraint (32j) couples energy dispatch and reserve procurement because both compete for the same generation capacity. Increasing the reserve award can reduce the capacity available for energy production, while increasing energy output reduces the headroom available for reserve. The joint clearing therefore allows an agent’s energy and reserve offers to interact through this shared capacity constraint.

The ancillary services market is implemented as a separate benchmark task that extends the real-time balancing market with reserve products. The day-ahead commitment, schedule, and prices enter as fixed inputs, while real-time energy and reserve are cleared jointly. Setting $d _ { j , \tau } ^ { \mathrm { r e s } } = 0$ for all reserve products removes the reserve layer and recovers the real-time balancing model of section A.2.

## A.3.3 PARTIALLY OBSERVABLE STOCHASTIC GAME

The ancillary services market is formulated as a general-sum POSG,

$$
{ \mathcal G } ^ { \mathrm { a s } } = \langle \mathcal T , \mathcal S , \{ \mathcal O _ { i } \} _ { i \in \mathcal T } , \{ A _ { i } \} _ { i \in \mathcal T } , P , \{ r _ { i } \} _ { i \in \mathcal T } , \gamma \rangle ,\tag{41}
$$

where I denotes the set of strategic generators. One episodic market step corresponds to one 30- minute real-time interval τ, during which energy and reserve are cleared jointly.

State. The state at interval τ is

$$
s _ { \tau } = \big ( x _ { \tau } ^ { \mathrm { s y s } } , x _ { \tau } ^ { \mathrm { d a } } , x _ { \tau } ^ { \mathrm { r e s } } , \{ x _ { i , \tau } \} _ { i \in \mathcal { T } } \big ) \in \mathcal { S } ,\tag{42}
$$

where $x _ { \tau } ^ { \mathrm { s y s } }$ contains the realized system conditions of demand $\tilde { d } _ { n , \tau }$ <sub>τ</sub> used in the joint energy–reserve clearing. The day-ahead component is

$$
x _ { \tau } ^ { \mathrm { d a } } = \left\{ u _ { i , h ( \tau ) } ^ { \mathrm { d a } } , q _ { i , h ( \tau ) } ^ { \mathrm { d a } } , \mathrm { L M P } _ { n , h ( \tau ) } ^ { \mathrm { d a } } \right\} _ { i , n } ,\tag{43}
$$

which contains the fixed commitment, binding generation schedules, and nodal prices from the corresponding day-ahead period. The reserve component is

$$
\begin{array} { r } { \boldsymbol { x } _ { \tau } ^ { \mathrm { r e s } } = \left( d _ { 1 , \tau } ^ { \mathrm { r e s } } , \ldots , d _ { { N } ^ { \mathrm { p r o d } } , \tau } ^ { \mathrm { r e s } } \right) , } \end{array}\tag{44}
$$

where $d _ { j , \tau } ^ { \mathrm { r e s } }$ is the system-wide requirement for reserve product j published before bidding. The operating state of generator i is

$$
x _ { i , \tau } = \left( p _ { i , \tau - 1 } , v _ { i , \tau } ^ { \mathrm { d a } } , w _ { i , \tau } ^ { \mathrm { d a } } \right) ,\tag{45}
$$

where $p _ { i , \tau - 1 }$ is the realized dispatch from the previous real-time interval, and $v _ { i , \tau } ^ { \mathrm { d a } }$ and $w _ { i , \tau } ^ { \mathrm { d a } }$ are the start-up and shut-down indicators implied by the mapped day-ahead commitment. Network parameters, reserve-product response times, generator operating limits, and cost parameters are fixed environment parameters and are omitted from the dynamic state.

Observation. Each generator observes its own operating information together with the public energy and reserve market information available before bidding. Its observation at interval τ is

$$
o _ { i , \tau } = \left( p _ { i , \tau - 1 } , \boldsymbol { u } _ { i , h ( \tau ) } ^ { \mathrm { d a } } , \boldsymbol { q } _ { i , h ( \tau ) } ^ { \mathrm { d a } } , \mathrm { L M P } _ { \mathrm { b u s } ( i ) , h ( \tau ) } ^ { \mathrm { d a } } , \boldsymbol { x } _ { \tau } ^ { \mathrm { s y s } } , \boldsymbol { x } _ { \tau } ^ { \mathrm { r e s } } \right) = \Omega _ { i } \big ( \boldsymbol { s } _ { \tau } \big ) \in \mathcal { O } _ { i } .\tag{46}
$$

Thus, the generator observes the realized net-demand conditions and the published reserve requirements in addition to its own day-ahead position and previous dispatch. It does not observe competitors’ private cost information or their contemporaneous energy and reserve offers. The market is therefore partially observable from the perspective of each participant.

Action. At each interval, generator i jointly submits an energy offer and one reserve offer price for each reserve product. Its action is

$$
a _ { i , \tau } = \left( \pi _ { i , \tau } ^ { \mathrm { e n } } , \pi _ { i , \tau } ^ { \mathrm { r e s } } \right) \in \mathcal { A } _ { i } ,\tag{47}
$$

where

$$
\pi _ { i , \tau } ^ { \mathrm { e n } } = \left( \pi _ { i , 1 , \tau } , \ldots , \pi _ { i , K , \tau } \right) , \qquad \pi _ { i , k , \tau } = \alpha _ { i , \tau } \pi _ { i , k } ^ { 0 } ,\tag{48}
$$

is the stepwise energy-offer price vector, obtained from a single markup $\alpha _ { i , \tau } \in [ 1 , \bar { \alpha } ]$ on the truthful offer $\pi _ { i , 1 } ^ { 0 } \leq \cdot \cdot \cdot \leq \bar { \pi _ { i , K } ^ { 0 } }$ as in the real-time balancing market, and

$$
{ \pmb \pi } _ { i , \tau } ^ { \mathrm { r e s } } = \left( \pi _ { i , 1 , \tau } ^ { \mathrm { r e s } } , \dots , \pi _ { i , N ^ { \mathrm { p r o d } } , \tau } ^ { \mathrm { r e s } } \right)\tag{49}
$$

contains the reserve offer prices. The energy offer satisfies

$$
\pi _ { i , 1 , \tau } \leq \pi _ { i , 2 , \tau } \leq \cdot \cdot \cdot \leq \pi _ { i , K , \tau } ,\tag{50}
$$

while $0 \leq \pi _ { i , j , \tau } ^ { \mathrm { r e s } } \leq \mathrm { V O L R }$ for each reserve product $j .$ The energy segment quantities $G _ { i , k }$ are fixed, and no reserve quantity is submitted by the agent; the reserve awards are determined by the co-optimization. The joint action is $a _ { \tau } = ( a _ { i , \tau } ) _ { i \in \mathcal { T } }$

Market clearing and reward. Given the current state and joint energy–reserve offers, the marketclearing mechanism

$$
z _ { \tau } = \mathcal { C } ^ { \mathrm { a s } } ( s _ { \tau } , a _ { \tau } )\tag{51}
$$

solves the co-optimization problem (32). The clearing outcome includes the energy dispatch $p _ { i , \tau }$ reserve awards $r _ { i , j , \tau }$ , real-time LMPs $\mathrm { L M P } _ { n , \tau } ^ { \mathrm { r t } }$ , and reserve prices $\lambda _ { j , \tau } ^ { \mathrm { r e s } }$

Each generator receives its total realized profit as the stage reward,

$$
\begin{array} { r l } & { r _ { i , \tau } = \mathrm { p r o f t } _ { i , \tau } = \mathrm { r e v e n u e } _ { i , \tau } ^ { \mathrm { d a } } + \mathrm { r e v e n u e } _ { i , \tau } ^ { \mathrm { r t } } + \mathrm { r e v e n u e } _ { i , \tau } ^ { \mathrm { r e s } } - \mathrm { c o s t } _ { i , \tau } } \\ & { \quad \quad = \Delta ^ { \mathrm { r t } } \mathrm { L M P } _ { \mathrm { b u s } ( i ) , h ( \tau ) } ^ { \mathrm { d a } } q _ { i , h ( \tau ) } ^ { \mathrm { d a } } + \Delta ^ { \mathrm { r t } } \mathrm { L M P } _ { \mathrm { b u s } ( i ) , \tau } ^ { \mathrm { r t } } \left( p _ { i , \tau } - q _ { i , h ( \tau ) } ^ { \mathrm { d a } } \right) } \\ & { \quad \quad + \Delta ^ { \mathrm { r t } } \displaystyle \sum _ { j } \lambda _ { j , \tau } ^ { \mathrm { r e s } } r _ { i , j , \tau } - \left( \Delta ^ { \mathrm { r t } } \mathrm { T C } _ { i } ( p _ { i , \tau } ) + \Delta ^ { \mathrm { r t } } \mathrm { N L } _ { i } u _ { i , h ( \tau ) } ^ { \mathrm { d a } } + S _ { i } v _ { i , \tau } ^ { \mathrm { d a } } \right) . } \end{array}\tag{52}
$$

as defined in section A.3.2. The reward therefore captures both energy-market profit and reservecapacity revenue.

State transition. The clearing outcome determines the operating state entering the next real-time interval. In particular,

$$
\boldsymbol { x } _ { i , \tau + 1 } = \left( p _ { i , \tau } , v _ { i , \tau + 1 } ^ { \mathrm { d a } } , w _ { i , \tau + 1 } ^ { \mathrm { d a } } \right) ,\tag{53}
$$

so the realized dispatch $p _ { i , \tau }$ enters the ramping constraint of the next interval. The corresponding day-ahead commitment, schedule, and price are updated deterministically according to $h ( \tau + 1 )$ , while the realized net demand and reserve requirements evolve according to their exogenous data processes. The overall transition is therefore

$$
s _ { \tau + 1 } \sim P ( \cdot \mid s _ { \tau } , a _ { \tau } ) .\tag{54}
$$

Agent objective. Each generator uses a policy $\sigma _ { i } ( a _ { i , \tau } \mid o _ { i , \tau } )$ to jointly determine its energy and reserve offers. Over a horizon of H real-time intervals, generator i maximizes its expected discounted profit,

$$
J _ { i } ( \sigma _ { i } , \sigma _ { - i } ) = \mathbb { E } \left[ \sum _ { \tau = 1 } ^ { H } \gamma ^ { \tau - 1 } r _ { i , \tau } \right] ,\tag{55}
$$

where $\sigma _ { - i }$ denotes the joint policies of the other generators. Because energy dispatch, reserve awards, and both energy and reserve prices are jointly determined by all submitted offers, the return of each generator depends on both its own bidding policy and those of other participants. Agents therefore learn how to allocate their available capacity between energy and reserve through repeated market interactions without observing competitors’ private information or policies.

## A.4 M4: PEER-TO-PEER LOCAL ENERGY MARKET

Rooftop photovoltaic generation and behind-the-meter storage have enabled many consumers to become prosumers that can both consume and supply electricity. When surplus energy is exported directly to the grid, it is typically settled at an export price such as feed-in-tariff, which is below the retail tariff paid for grid consumption. A peer-to-peer (P2P) local energy market allows neighboring participants to trade their surplus and deficit locally at prices between these two outside options, creating the possibility of mutual benefits for buyers and sellers. Unlike the day-ahead and real-time markets in sections A.1 and A.2, which operate at the transmission level, the P2P market considered here operates at the end-use level and facilitates local energy exchange among prosumers.

The P2P market uses 15-minute trading intervals. Before each auction, the net energy position of every participant is determined from its local generation, consumption, and storage operation. Hence, a participant knows the quantity it has available to buy or sell before submitting its bid, and its strategic decisions are its battery power and the submitted price. A local market operator collects the bids and offers, clears the auction, and settles the resulting transactions. The market does not issue physical dispatch instructions: the participants’ net quantities are fixed before clearing, so the auction determines which quantities are matched and at what price rather than how the underlying devices are operated.

In our model, the P2P local energy market is implemented as a periodic double auction among a fixed population of $N ^ { \mathrm { a g e n t } }$ prosumers, each equipped with rooftop photovoltaic generation, an inelastic local load, and a battery. In each trading interval, a participant appears on one side of the market according to its net energy position: a surplus is offered for sale, whereas a deficit is submitted as demand. The local market operator collects one price–quantity pair from each participant and constructs aggregate supply and demand curves. The market clears at their intersection, with all matched buyers and sellers settling at a common market clearing price. Any unmatched deficit is supplied by the upstream grid at the retail tariff, while any unmatched surplus is exported to the grid at the export price. This section presents the market clearing model in subsection A.4.1, the pricing and settlement rules in subsection A.4.2, and the corresponding POSG in subsection A.4.3.

## A.4.1 MARKET MODEL

Let τ index the 15-minute P2P trading intervals, with $\Delta ^ { \mathrm { p 2 p } } = 0 . 2 5$ h. Before each auction, the local generation, consumption, and battery operation of participant i determine its net power position,

$$
p _ { i , \tau } ^ { \mathrm { n e t } } = - p _ { i , \tau } ^ { \mathrm { p v } } + p _ { i , \tau } ^ { \mathrm { c h } } + d _ { i , \tau } - p _ { i , \tau } ^ { \mathrm { d i s } } ,\tag{56a}
$$

and the corresponding quantities available for P2P trading are

$$
q _ { i , \tau } ^ { \mathrm { s e l l } } = \Delta ^ { \mathrm { p 2 p } } \operatorname* { m a x } \bigl ( - p _ { i , \tau } ^ { \mathrm { n e t } } , 0 \bigr ) , \qquad q _ { i , \tau } ^ { \mathrm { b u y } } = \Delta ^ { \mathrm { p 2 p } } \operatorname* { m a x } \bigl ( p _ { i , \tau } ^ { \mathrm { n e t } } , 0 \bigr ) .\tag{56b}
$$

Here, $p _ { i , \tau } ^ { \mathrm { p v } }$ and $d _ { i , \tau }$ denote photovoltaic generation and local demand, respectively. $p _ { i , \tau } ^ { \mathrm { d i s } }$ and $p _ { i , \tau } ^ { \mathrm { c h } }$ denote battery discharging and charging power, respectively. Only one of $q _ { i , \tau } ^ { \mathrm { s e l l } }$ and $q _ { i , \tau } ^ { \mathrm { b u y } }$ can be positive. A participant with surplus energy is therefore a seller, while one with an energy deficit is a buyer.

The battery is the only local device that carries a physical state between trading intervals. Its state of charge evolves according to

$$
\mathrm { s o c } _ { i , \tau + 1 } = \mathrm { s o c } _ { i , \tau } + \frac { \Delta ^ { \mathrm { p 2 p } } } { E _ { i } } \left( \eta _ { i } ^ { \mathrm { c h } } p _ { i , \tau } ^ { \mathrm { c h } } - \frac { p _ { i , \tau } ^ { \mathrm { d i s } } } { \eta _ { i } ^ { \mathrm { d i s } } } \right) ,\tag{57}
$$

where $E _ { i }$ is the battery energy capacity, and $\eta _ { i } ^ { \mathrm { c h } }$ and $\eta _ { i } ^ { \mathrm { d i s } }$ are the charging and discharging loss efficiencies. The feasible charging and discharging powers are bounded by

$$
0 \leq p _ { i , \tau } ^ { \mathrm { d i s } } \leq \operatorname* { m i n } \left( \bar { p } _ { i } ^ { \mathrm { b a t } } , \frac { \left( \mathrm { s o c } _ { i , \tau } - \underline { { \mathrm { s o c } _ { i } } } \right) E _ { i } \eta _ { i } ^ { \mathrm { d i s } } } { \Delta ^ { \mathrm { p 2 p } } } \right) ,\tag{58}
$$

and

$$
0 \leq p _ { i , \tau } ^ { \mathrm { c h } } \leq \operatorname* { m i n } \left( \bar { p } _ { i } ^ { \mathrm { b a t } } , \frac { \left( \overline { { \mathrm { s o c } } } _ { i } - \mathrm { s o c } _ { i , \tau } \right) E _ { i } } { \eta _ { i } ^ { \mathrm { c h } } \Delta ^ { \mathrm { p 2 p } } } \right) ,\tag{59}
$$

where $\bar { p } _ { i } ^ { \mathrm { b a t } }$ is the battery power rating and $[ \underline { { \mathrm { s o c } _ { i } } } , \overline { { \mathrm { s o c } _ { i } } } ]$ is the admissible state-of-charge range. Charging and discharging are mutually exclusive within each trading interval.

Battery operation is chosen by the participant before the P2P auction. Hence, $p _ { i , \tau } ^ { \mathrm { n e t } }$ and the corresponding trading quantity are fixed when the participant submits its bid, and within the auction only the submitted price is a strategic bidding decision.

The seller and buyer sets in interval τ are

$$
S _ { \tau } = \left\{ i : q _ { i , \tau } ^ { \mathrm { s e l l } } > 0 \right\} , \qquad \mathcal { B } _ { \tau } = \left\{ i : q _ { i , \tau } ^ { \mathrm { b u y } } > 0 \right\} .\tag{60}
$$

Each participant submits one price $\pi _ { i , \tau } .$ : an ask if $i \in S _ { \tau }$ and a bid if $\textit { i } \in \boldsymbol { B } _ { \tau }$ . Submitted prices are restricted by the outside options provided by the upstream grid,

$$
\pi ^ { \mathrm { e x p } } \leq \pi _ { i , \tau } \leq \pi ^ { \mathrm { r e t } } ,\tag{61}
$$

where $\pi ^ { \mathrm { r e t } }$ is the retail tariff for grid imports and $\pi ^ { \mathrm { e x p } }$ is the export price for grid injections, with $\pi ^ { \mathrm { { e x p } } } < \pi ^ { \mathrm { { r e t } } }$ . The submitted quantity is fixed by the participant’s net position, so $\pi _ { i , \tau }$ is the only strategic variable in the auction.

The local market operator clears the market through a periodic double auction. Seller asks are sorted in ascending order of price, while buyer bids are sorted in descending order of price, with ties broken by participant index. Let $\sigma _ { \tau } ^ { \mathrm { s } } ( j )$ denote the participant occupying position j in the sorted seller order and $\sigma _ { \tau } ^ { \mathrm { b } } ( j )$ the participant occupying position $j$ in the sorted buyer order. The associated cumulative quantities are

$$
Q _ { j , \tau } ^ { \mathrm { s e l l } } = \sum _ { \ell = 1 } ^ { j } q _ { \sigma _ { \tau } ^ { \mathrm { s } } } ^ { \mathrm { s e l l } } ( \ell ) , \tau  , \qquad Q _ { j , \tau } ^ { \mathrm { b u y } } = \sum _ { \ell = 1 } ^ { j } q _ { \sigma _ { \tau } ^ { \mathrm { b } } ( \ell ) , \tau } ^ { \mathrm { b u y } } .\tag{62}
$$

The total quantities available on the two sides are

$$
Q _ { \tau } ^ { \mathrm { s e l l } } = \sum _ { i \in \mathcal { S } _ { \tau } } q _ { i , \tau } ^ { \mathrm { s e l l } } , \qquad Q _ { \tau } ^ { \mathrm { b u y } } = \sum _ { i \in \mathcal { B } _ { \tau } } q _ { i , \tau } ^ { \mathrm { b u y } } .\tag{63}
$$

The sorted submissions define the aggregate supply curve $S _ { \tau } ( x )$ and aggregate demand curve $D _ { \tau } ( x )$ . The supply curve returns the ask price associated with the x-th unit of cumulative supply, while the demand curve returns the bid price associated with the x-th unit of cumulative demand. Both are step functions whose breakpoints are given by the cumulative quantities in equation (62).

Let $\mathcal { X } _ { \tau }$ denote the union of the supply and demand breakpoints, together with zero. The cleared P2P trading volume is

$$
\begin{array} { r } { Q _ { \tau } ^ { \star } = \operatorname* { m a x } \left\{ x \in \mathcal { X } _ { \tau } : x \leq \operatorname* { m i n } \bigl ( Q _ { \tau } ^ { \mathrm { s e l l } } , Q _ { \tau } ^ { \mathrm { b u y } } \bigr ) , D _ { \tau } ( x ) \geq S _ { \tau } ( x ) \right\} . } \end{array}\tag{64}
$$

Thus, trading continues while the marginal buyer’s bid is no lower than the marginal seller’s ask. If no bid meets an ask, $Q _ { \tau } ^ { \star } = 0$

Individual awards follow the sorted orders. For the seller occupying position $j ,$

$$
q _ { \sigma _ { \tau } ^ { \mathrm { s } } ( j ) , \tau } ^ { \mathrm { s e l l } \star } = \operatorname* { m i n } \left[ \operatorname* { m a x } \bigl ( Q _ { \tau } ^ { \star } - Q _ { j - 1 , \tau } ^ { \mathrm { s e l l } } , 0 \bigr ) , q _ { \sigma _ { \tau } ^ { \mathrm { s } } ( j ) , \tau } ^ { \mathrm { s e l l } } \right] ,\tag{65}
$$

with $Q _ { 0 , \tau } ^ { \mathrm { s e l l } } = 0$ . Similarly, the award to the buyer occupying position $j$ is

$$
\begin{array} { r } { q _ { \sigma _ { \tau } ^ { \mathrm { b u y } } ( j ) , \tau } ^ { \mathrm { b u y } \star } = \operatorname* { m i n } \left[ \operatorname* { m a x } \left( Q _ { \tau } ^ { \star } - Q _ { j - 1 , \tau } ^ { \mathrm { b u y } } , 0 \right) , q _ { \sigma _ { \tau } ^ { \mathrm { b u y } } ( j ) , \tau } ^ { \mathrm { b u y } } \right] , } \end{array}\tag{66}
$$

with $Q _ { 0 , \tau } ^ { \mathrm { b u y } } = 0$

Equations (65) and (66) fill the most competitive submissions first: lower-priced sellers and higherpriced buyers receive priority. Since $Q _ { \tau } ^ { \star }$ coincides with a cumulative-quantity breakpoint on at least one side of the market, all accepted participants on that side are either fully cleared or rejected. On the opposite side, only the marginal participant may be partially cleared.

## A.4.2 PRICING AND SETTLEMENT

Let $S _ { \tau } ^ { \star }$ and $D _ { \tau } ^ { \star }$ denote the marginal accepted ask and bid, respectively, and let $S _ { \tau } ^ { + }$ and $D _ { \tau } ^ { + }$ denote the first rejected ask and bid beyond the cleared volume $Q _ { \tau } ^ { \star }$ . A uniform clearing price must be high enough for all accepted sellers and low enough for all accepted buyers, while remaining unattractive to rejected submissions. This defines the feasible price interval

$$
\underline { { \lambda } } _ { \tau } = \operatorname* { m a x } \bigl ( S _ { \tau } ^ { \star } , D _ { \tau } ^ { + } \bigr ) , \qquad \bar { \lambda } _ { \tau } = \operatorname* { m i n } \bigl ( D _ { \tau } ^ { \star } , S _ { \tau } ^ { + } \bigr ) , \qquad \lambda _ { \tau } ^ { \mathrm { l o c } } = \frac { 1 } { 2 } \left( \underline { { \lambda } } _ { \tau } + \bar { \lambda } _ { \tau } \right) .\tag{67}
$$

The local market therefore publishes the midpoint of the admissible price interval. When an adjacent accepted or rejected submission does not exist because one side of the market is exhausted, the export price $\bar { \pi ^ { \mathrm { e x p } } }$ and retail tariff $\pi ^ { \mathrm { r e t } }$ provide the corresponding lower and upper price bounds. Consequently,

$$
\pi ^ { \mathrm { e x p } } \leq \lambda _ { \tau } ^ { \mathrm { l o c } } \leq \pi ^ { \mathrm { r e t } } .\tag{68}
$$

Thus, a matched seller receives no less than the grid export price and a matched buyer pays no more than the retail tariff. If a marginal participant is only partially cleared, the admissible interval collapses to that participant’s submitted price.

P2P awards are financially binding, while any unmatched quantity is settled with the upstream grid. Define the residual export and import quantities as

$$
q _ { i , \tau } ^ { \mathrm { e x } } = q _ { i , \tau } ^ { \mathrm { s e l l } } - q _ { i , \tau } ^ { \mathrm { s e l l } \star } , \qquad q _ { i , \tau } ^ { \mathrm { i m } } = q _ { i , \tau } ^ { \mathrm { b u y } } - q _ { i , \tau } ^ { \mathrm { b u y } \star } .\tag{69}
$$

The settlement of participant i is then

$$
\begin{array} { r } { \mathrm { r e v e n u e } _ { i , \tau } = \lambda _ { \tau } ^ { \mathrm { l o c } } q _ { i , \tau } ^ { \mathrm { s e l l } \star } + \pi ^ { \mathrm { e x p } } q _ { i , \tau } ^ { \mathrm { e x } } , } \end{array}\tag{70a}
$$

$$
\mathrm { c o s t } _ { i , \tau } = \lambda _ { \tau } ^ { \mathrm { l o c } } q _ { i , \tau } ^ { \mathrm { b u y } \star } + \pi ^ { \mathrm { r e t } } q _ { i , \tau } ^ { \mathrm { i m } } + c _ { i } ^ { \mathrm { d e g } } \Delta ^ { \mathrm { p 2 p } } \left( p _ { i , \tau } ^ { \mathrm { c h } } + p _ { i , \tau } ^ { \mathrm { d i s } } \right) ,\tag{70b}
$$

$$
\mathrm { p r o f i t } _ { i , \tau } = \mathrm { r e v e n u e } _ { i , \tau } - \mathrm { c o s t } _ { i , \tau } .\tag{70c}
$$

Here, $c _ { i } ^ { \mathrm { d e g } }$ is the battery degradation cost in EUR/MWh of throughput. Photovoltaic generation is assumed to have zero marginal cost, while local demand is inelastic. A participant with a net energy deficit may therefore have negative profit, corresponding to a net electricity bill; maximizing profit in this case is equivalent to minimizing that bill. Because all matched buyers and sellers settle at the same local price, P2P payments are budget balanced within the local market, excluding transactions with the upstream grid and any network charges not modeled here.

Remark on the P2P local market. The P2P market provides an intermediate trading opportunity between the two grid settlement prices. Sellers can receive more than the export price, while buyers can pay less than the retail tariff when mutually acceptable bids are matched. The market can also increase the share of locally produced renewable energy consumed within the local network. It therefore complements, rather than replaces, upstream electricity markets and grid transactions.

## A.4.3 PARTIALLY OBSERVABLE STOCHASTIC GAME

The P2P local energy market is formulated as a general-sum POSG,

$$
{ \mathcal G } ^ { \mathrm { p 2 p } } = \langle { \mathcal T } , { \mathcal S } , \{ { \mathcal O } _ { i } \} _ { i \in { \mathcal T } } , \{ A _ { i } \} _ { i \in { \mathcal T } } , P , \{ r _ { i } \} _ { i \in { \mathcal T } } , \gamma \rangle ,\tag{71}
$$

where $\mathcal { T }$ is the fixed set of $N ^ { \mathrm { a g e n t } }$ prosumers. One game step corresponds to one 15-minute P2P trading interval τ.

State. The state at interval τ is

$$
s _ { \tau } = \left( \{ x _ { i , \tau } \} _ { i \in \mathcal { I } } , \pi ^ { \mathrm { e x p } } , \pi ^ { \mathrm { r e t } } \right) \in \mathcal { S } ,\tag{72}
$$

where the local physical state of prosumer i is

$$
x _ { i , \tau } = \left( \mathrm { s o c } _ { i , \tau } , p _ { i , \tau } ^ { \mathrm { p v } } , d _ { i , \tau } , p _ { i , \tau } ^ { \mathrm { c h } } , p _ { i , \tau } ^ { \mathrm { d i s } } \right) .\tag{73}
$$

These variables determine the participant’s net power position $p _ { i , \tau } ^ { \mathrm { n e t } }$ and therefore its available selling or buying quantity through equation 56b. The export price $\pi ^ { \mathrm { e x p } }$ and retail tariff $\pi ^ { \mathrm { r e t } }$ define the outside settlement options with the upstream grid.

Observation. Each participant observes its own physical state and net trading position, together with the public grid settlement prices. Its observation is

$$
o _ { i , \tau } = \left( \mathrm { s o c } _ { i , \tau } , p _ { i , \tau } ^ { \mathrm { n e t } } , q _ { i , \tau } ^ { \mathrm { s e l l } } , q _ { i , \tau } ^ { \mathrm { b u y } } , \pi ^ { \mathrm { e x p } } , \pi ^ { \mathrm { r e t } } \right) = \Omega _ { i } ( s _ { \tau } ) \in \mathcal { O } _ { i } .\tag{74}
$$

Thus, each participant knows whether it is a seller or buyer and the quantity it has available to trade before submitting its price. It does not observe the contemporaneous bids, asks, or private local states of other participants. The market is therefore partially observable from the perspective of each individual prosumer.

Action. The trading quantity is fixed by the battery power before market clearing, so the strategic action of participant i is its battery power together with its submitted price,

$$
a _ { i , \tau } = \left( p _ { i , \tau } ^ { \mathrm { b a t } } , \pi _ { i , \tau } \right) , \qquad \pi _ { i , \tau } \in \left[ \pi ^ { \mathrm { e x p } } , \pi ^ { \mathrm { r e t } } \right] ,\tag{75}
$$

where $p _ { i , \tau } ^ { \mathrm { b a t } } = p _ { i , \tau } ^ { \mathrm { d i s } } - p _ { i , \tau } ^ { \mathrm { c h } }$ is the signed battery power within the feasible range (58)–(59). If $q _ { i , \tau } ^ { \mathrm { s e l l } } > 0$ the submitted price is an ask; if $q _ { i , \tau } ^ { \mathrm { b u y } } > 0 .$ , it is a bid. A participant with zero net position does not actively participate in the auction. The joint action is $a _ { \tau } = ( a _ { i , \tau } ) _ { i \in \mathcal { T } }$

Market clearing and reward. Given the participants’ net positions and joint submitted prices, the local market-clearing mechanism

$$
z _ { \tau } = \mathcal { C } ^ { \mathrm { p 2 p } } ( s _ { \tau } , a _ { \tau } )\tag{76}
$$

sorts the asks and bids and applies the double-auction clearing rules in Section $\mathrm { A } . 4 . 1$ . The clearing outcome contains the total traded volume $Q _ { \tau } ^ { \star }$ , the uniform local clearing price $\lambda _ { \tau } ^ { \mathrm { l o c } }$ , and the individual awards $q _ { i , \tau } ^ { \mathrm { s e l l \star } }$ and $q _ { i , \tau } ^ { \mathrm { b u y } \star }$ <sup>⋆</sup>. Any unmatched quantity is settled with the upstream grid.

The stage reward of participant i is its realized profit,

$$
\begin{array} { r l } & { r _ { i , \tau } = \mathrm { p r o f t } _ { i , \tau } = \mathrm { r e v e n u e } _ { i , \tau } - \mathrm { c o s t } _ { i , \tau } } \\ & { \qquad = \left( \lambda _ { \tau } ^ { \mathrm { l o c } } q _ { i , \tau } ^ { \mathrm { s e l l } \star } + \pi ^ { \mathrm { e x p } } q _ { i , \tau } ^ { \mathrm { e x } } \right) - \left( \lambda _ { \tau } ^ { \mathrm { l o c } } q _ { i , \tau } ^ { \mathrm { b u y \star } } + \pi ^ { \mathrm { r e t } } q _ { i , \tau } ^ { \mathrm { i m } } + c _ { i } ^ { \mathrm { d e g } } \Delta ^ { \mathrm { p 2 p } } ( p _ { i , \tau } ^ { \mathrm { c h } } + p _ { i , \tau } ^ { \mathrm { d i s } } ) \right) } \end{array}\tag{77}
$$

where profi $\mathrm { t } _ { i , \tau }$ is defined by equation 70. For a seller, the reward reflects revenue from P2P sales and any residual export to the grid. For a buyer, the reward reflects the cost of P2P purchases and any residual import from the grid, together with battery degradation cost. A buyer may therefore receive a negative reward corresponding to its electricity bill, so maximizing reward is equivalent to minimizing that bill.

State transition. The battery state of charge provides the physical coupling between consecutive trading intervals. Following equation 57,

$$
\mathrm { s o c } _ { i , \tau + 1 } = \mathrm { s o c } _ { i , \tau } + \frac { \Delta ^ { \mathrm { p 2 p } } } { E _ { i } } \left( \eta _ { i } ^ { \mathrm { c h } } p _ { i , \tau } ^ { \mathrm { c h } } - \frac { p _ { i , \tau } ^ { \mathrm { d i s } } } { \eta _ { i } ^ { \mathrm { d i s } } } \right) .\tag{78}
$$

Photovoltaic generation and local demand evolve according to their exogenous data processes, while battery operation for the next period is determined before the next P2P auction. The next net position and trading quantity are then obtained from equation 56b. The overall state transition is

$$
s _ { \tau + 1 } \sim P ( \cdot \mid s _ { \tau } , a _ { \tau } ) .\tag{79}
$$

Under the market design considered here, the submitted P2P price affects market matching and settlement but does not directly alter the participant’s physical net position or battery dynamics.

Agent objective. Each participant uses a policy $\sigma _ { i } ( a _ { i , \tau } \mid o _ { i , \tau } )$ to determine its submitted price. Over a horizon of H trading intervals, participant i seeks to maximize its expected discounted profit,

$$
J _ { i } ( \sigma _ { i } , \sigma _ { - i } ) = \mathbb { E } \left[ \sum _ { \tau = 1 } ^ { H } \gamma ^ { \tau - 1 } r _ { i , \tau } \right] ,\tag{80}
$$

where $\sigma _ { - i }$ denotes the joint policies of the other participants. Because the cleared volume, matching outcome, and local clearing price depend on all submitted bids and asks, the return of each participant depends on both its own pricing policy and those of the other prosumers. Participants therefore learn bidding strategies through repeated local-market interactions without observing competitors contemporaneous submissions or private states.

## A.5 M5: LOCAL FLEXIBILITY MARKET

Growing electrification and distributed generation can place distribution networks under increasing operational stress, leading to feeder congestion and voltage violations. Reinforcing distribution infrastructure can be costly and slow, particularly when the underlying constraint occurs only during a limited number of periods. A local flexibility market provides an alternative means of managing these constraints by allowing the distribution system operator (DSO) to procure temporary changes in power injection or consumption from resources already connected to the affected network.

The market operates at the distribution level and clears once per delivery period. It addresses local network constraints that are not represented in the transmission-level wholesale markets, while the upstream energy price faced by each aggregator is treated as exogenous. The DSO is the sole buyer, making the market a one-sided procurement auction. It specifies the locations and periods where flexibility is required, and accepts offers that restore a feasible distribution-network operating point at minimum procurement cost. The sellers are aggregators, with one aggregator at each bus, each representing a portfolio of distributed energy resources.

Two features distinguish this market from the preceding settings. First, the traded product is a deviation from a predefined baseline rather than energy itself. Second, settlement follows a payas-bid rule: each accepted aggregator is paid its submitted offer price for the flexibility it provides, rather than a common market-clearing price. The settlement therefore does not rely on dual-based marginal prices. This section presents the market-clearing model in subsection A.5.1, the pricing and settlement rule in subsection A.5.2, and the corresponding POSG in subsection A.5.3.

## A.5.1 MARKET MODEL

We consider a radial distribution network with $N ^ { \mathrm { b u s } }$ buses supplied by a single substation, indexed by 0, whose voltage is fixed at $v _ { 0 } = 1 \ \mathrm { p . u }$ . Let τ index the local flexibility trading intervals, with duration $\Delta ^ { \mathrm { { f l e x } } } = \breve { 1 }$ hour. Before the flexibility market clears, each aggregator has a baseline operating position determined by local photovoltaic generation, demand, and planned battery charging. With no flexibility activated, the baseline active-power injection at bus n is

$$
P _ { n , \tau } ^ { \mathrm { b a s e , i n j } } = \sum _ { i : \mathrm { b u s } ( i ) = n } \left( p _ { i , \tau } ^ { \mathrm { p v } } - d _ { i , \tau } - p _ { i , \tau } ^ { \mathrm { p l a n } } \right) - d _ { n , \tau } ^ { \mathrm { b g } } ,\tag{81}
$$

where $p _ { i , \tau } ^ { \mathrm { p v } }$ is the photovoltaic output, $d _ { i , \tau }$ is the local demand of aggregator $i , p _ { i , \tau } ^ { \mathrm { p l a n } }$ is its planned battery charging power, and $d _ { n , \tau } ^ { \mathrm { b g } }$ denotes background demand not controlled by any aggregator.

Under the radial-network approximation, the corresponding baseline line flows and squared voltages are obtained from

$$
P _ { l , \tau } ^ { \mathrm { b a s e } } = - \sum _ { m \in \mathscr { D } ( l ) } P _ { m , \tau } ^ { \mathrm { b a s e , i n j } } ,\tag{82}
$$

and

$$
\left( v _ { n , \tau } ^ { \mathrm { b a s e } } \right) ^ { 2 } = v _ { 0 } ^ { 2 } + 2 \sum _ { m } \left( R _ { n , m } P _ { m , \tau } ^ { \mathrm { b a s e , i n j } } + X _ { n , m } Q _ { m , \tau } ^ { \mathrm { b a s e , i n j } } \right) ,\tag{83}
$$

where $\mathcal { D } ( l )$ denotes the set of buses downstream of line l. The quantities $R _ { n , m }$ and $X _ { n , m }$ are the sums of resistance and reactance over the network path shared by buses n and $m ,$ respectively. Reactive injections are determined from the active loads using a fixed power-factor assumption.

Before offers are submitted, the DSO publishes indicators of the local flexibility requirement. For an undervoltage at bus n, the indicator is

$$
\mathrm { r e q } _ { n , \tau } ^ { \mathrm { v } } = \frac { \operatorname* { m a x } \Bigl [ \left( \underline { { v } } _ { n } + \gamma ^ { \mathrm { v } } \right) ^ { 2 } - \left( v _ { n , \tau } ^ { \mathrm { b a s e } } \right) ^ { 2 } , 0 \Bigr ] } { 2 R _ { n , n } } ,\tag{84}
$$

while the requirement associated with forward-flow congestion on line l is

$$
\mathrm { r e q } _ { l , \tau } ^ { \mathrm { t h } } = \operatorname* { m a x } \left[ P _ { l , \tau } ^ { \mathrm { b a s e } } - \left( 1 - \gamma ^ { \mathrm { t h } } \right) \bar { P } _ { l } , 0 \right] .\tag{85}
$$

These quantities are published as market information for the aggregators; the market clearing itself enforces the full network constraints below.

The traded product is upward flexibility, defined as an increase in net active power injection relative to the baseline. Aggregator i submits one price–quantity offer $( \pi _ { i , \tau } , \bar { q } _ { i , \tau } )$ , where $\pi _ { i , \tau }$ is its offer price and $\bar { q } _ { i , \tau }$ is the maximum flexibility quantity offered. Any quantity $q _ { i , \tau } \in [ 0 , \bar { q } _ { i , \tau } ]$ may be accepted by the DSO. Unlike the stepwise offers used in the wholesale markets, each aggregator submits a single price–quantity pair in each flexibility trading interval.

Upward flexibility can be delivered by reducing planned battery charging or, once planned charging has been fully removed, by discharging the battery. The offered quantity must therefore satisfy

$$
0 \leq \bar { q } _ { i , \tau } \leq q _ { i , \tau } ^ { \mathrm { p h y s } } ,\tag{86}
$$

where

$$
q _ { i , \tau } ^ { \mathrm { p h y s } } = p _ { i , \tau } ^ { \mathrm { p l a n } } + \operatorname* { m i n } \left( \bar { p } _ { i } ^ { \mathrm { d i s } } , \frac { \eta _ { i } ^ { \mathrm { d i s } } E _ { i } \left( \mathrm { s o c } _ { i , \tau } - \underline { { \mathrm { s o c } _ { i } } } \right) } { \Delta ^ { \mathrm { f l e x } } } \right) .\tag{87}
$$

Here, $\bar { p } _ { i } ^ { \mathrm { d i s } }$ is the battery discharge-power limit, $E _ { i }$ is its energy capacity, $\eta _ { i } ^ { \mathrm { d i s } }$ is its discharge efficiency, and soc is the minimum admissible state of charge. Because $q _ { i , \tau } ^ { \mathrm { p h y s } }$ depends on the current state of charge, the amount of flexibility available from an aggregator varies across trading intervals.

Given the submitted offers, the DSO selects flexibility awards by solving

$$
\operatorname* { m i n } _ { \{ q , s , P ^ { \mathrm { i n j } } , P , v \} } \Delta ^ { \mathrm { f l e x } } \left[ \sum _ { i } \pi _ { i , \tau } q _ { i , \tau } + \mathrm { V O L L } \sum _ { n } s _ { n , \tau } \right] ,\tag{88a}
$$

subject to

$$
P _ { n , \tau } ^ { \mathrm { i n j } } = \sum _ { i : \mathrm { b u s } ( i ) = n } \left( p _ { i , \tau } ^ { \mathrm { p v } } - d _ { i , \tau } + q _ { i , \tau } - p _ { i , \tau } ^ { \mathrm { p l a n } } \right) - d _ { n , \tau } ^ { \mathrm { b g } } + s _ { n , \tau } ,\tag{88b}
$$

$$
P _ { l , \tau } = - \sum _ { m \in \mathcal { D } ( l ) } P _ { m , \tau } ^ { \mathrm { i n j } } ,\tag{88c}
$$

$$
v _ { n , \tau } ^ { 2 } = v _ { 0 } ^ { 2 } + 2 \sum _ { m } \left( R _ { n , m } P _ { m , \tau } ^ { \mathrm { i n j } } + X _ { n , m } Q _ { m , \tau } ^ { \mathrm { i n j } } \right) ,\tag{88d}
$$

$$
\begin{array} { r } { \left( \underline { { v } } _ { n } + \gamma ^ { \mathrm { v } } \right) ^ { 2 } \leq v _ { n , \tau } ^ { 2 } \leq \left( \bar { v } _ { n } - \gamma ^ { \mathrm { v } } \right) ^ { 2 } , \qquad n \neq 0 , } \end{array}\tag{88e}
$$

$$
- \left( 1 - \gamma ^ { \mathrm { t h } } \right) \bar { P } _ { l } \leq P _ { l , \tau } \leq \left( 1 - \gamma ^ { \mathrm { t h } } \right) \bar { P } _ { l } ,\tag{88f}
$$

$$
0 \leq q _ { i , \tau } \leq \bar { q } _ { i , \tau } ,\tag{88g}
$$

$$
0 \leq s _ { n , \tau } \leq d _ { n , \tau } ^ { \mathrm { b g } } + \sum _ { i : \mathrm { b u s } ( i ) = n } d _ { i , \tau } .\tag{88h}
$$

The objective (88a) minimizes the total procurement cost of local flexibility in each trading interval. The first term represents the pay-as-bid cost of accepted flexibility offers, while the second penalizes emergency load curtailment. The value of lost load VOLL is set above the admissible flexibility offer prices so that curtailment is used only when the available flexibility cannot restore a feasible network operating point.

Constraint (88b) defines the active-power injection at each distribution bus. Without flexibility, the baseline injection consists of photovoltaic generation minus local demand and planned battery charging. An accepted flexibility quantity $q _ { i , \tau }$ increases the bus injection relative to this baseline, either by reducing charging or by discharging the battery. Background demand $d _ { n , \tau } ^ { \mathrm { b g } }$ represents demand not controlled by the aggregators, while $s _ { n , \tau }$ denotes emergency load curtailment.

Constraint (88c) computes the active-power flow on each line as the total downstream withdrawal under the radial-network structure. Here, $\mathcal { D } ( l )$ denotes the set of buses downstream of line l. Constraint (88d) uses a linearized radial distribution power-flow model. The quantities $R _ { n , m }$ and $X _ { n , m }$ represent the resistance and reactance shared by the paths from the substation to buses n and m, respectively. The reactive injections $Q _ { m , \tau } ^ { \mathrm { i n j } }$ are determined from the fixed power-factor assumptions of the underlying loads and are not strategic decision variables.

Constraints (88e) and (88f) enforce the distribution-network operating limits. The parameters $\gamma ^ { \mathrm { v } }$ and $\gamma ^ { \mathrm { t h } }$ introduce safety margins by tightening the admissible voltage range and thermal line ratings used during clearing. Constraint (88g) allows the DSO to accept any quantity between zero and the quantity offered by aggregator i. The submitted quantity $\bar { q } _ { i , { \cdot } }$ <sub>τ</sub> must itself satisfy the physical deliverability condition (87). Consequently, the clearing may partially accept an offer when the full quantity is not required to restore network feasibility. Finally, constraint (88h) bounds emergency curtailment by the total active demand at each bus. Curtailment acts as a feasibility backstop when the submitted flexibility offers are insufficient to satisfy the tightened network constraints.

Since the network equations (88b)–(88d) are affine in the clearing variables under the linearized radial power-flow model, the resulting flexibility-market clearing problem is a linear program.

## A.5.2 PAYMENT AND SETTLEMENT

Settlement follows a pay-as-bid rule. Each accepted aggregator is paid its submitted offer price for the quantity cleared by the DSO,

$$
\begin{array} { r } { \mathrm { r e v e n u e } _ { i , \tau } = \Delta ^ { \mathrm { f l e x } } \pi _ { i , \tau } q _ { i , \tau } , \qquad \mathrm { p r o f t } _ { i , \tau } = \mathrm { r e v e n u e } _ { i , \tau } - \mathrm { c o s t } _ { i , \tau } , } \end{array}\tag{89}
$$

where cos $\mathrm { t } _ { i , \tau }$ is the true delivery cost defined in Section A.5.3. Because settlement is pay-as-bid, two aggregators providing flexibility at the same bus and in the same interval may receive different payments if they submit different offer prices.

Although the market imposes no uniform clearing price, the curtailment backstop creates an implicit upper value for flexibility. Emergency curtailment can relieve the same network constraints at a marginal cost of VOLL, so a flexibility offer is selected only when its network benefit justifies its submitted cost relative to this alternative. This value is inherently locational because the effec of an injection depends on where it enters the network. For a binding forward-flow constraint, an additional MW of injection downstream of the constrained line reduces the line flow by one MW under equation 88c, whereas an injection outside that downstream region does not relieve that constraint. For a binding voltage constraint, the value of flexibility depends on the voltage sensitivity of the participant’s bus, so injections at electrically more effective locations can support higher accepted offer prices than injections with weaker voltage impact.

The flexibility product is defined as a deviation from the pre-market baseline. For aggregator $i ,$ this baseline is determined from its photovoltaic generation, local demand, and planned battery charging,

$$
p _ { i , \tau } ^ { \mathrm { b a s e } } = p _ { i , \tau } ^ { \mathrm { p v } } - d _ { i , \tau } - p _ { i , \tau } ^ { \mathrm { p l a n } } .\tag{90}
$$

An accepted flexibility award $q _ { i , \tau }$ therefore requires the aggregator to increase its net injection by exactly that amount relative to the baseline. In the benchmark, delivery is assumed to equal the cleared award, so no payment is made for flexibility that is not delivered.

## A.5.3 PARTIALLY OBSERVABLE STOCHASTIC GAME

The local flexibility market is formulated as a general-sum partially observable stochastic game (POSG),

$$
\mathcal { G } ^ { \mathrm { f l e x } } = \langle \mathcal { T } , S , \{ \mathcal { O } _ { i } \} _ { i \in \mathcal { I } } , \{ A _ { i } \} _ { i \in \mathcal { I } } , P , \{ r _ { i } \} _ { i \in \mathcal { I } } , \gamma \rangle ,\tag{91}
$$

where $\mathcal { T }$ denotes the set of strategic aggregators. One game step corresponds to one local flexibility trading interval $\tau .$

State. The state at interval τ is

$$
s _ { \tau } = ( x _ { \tau } ^ { \mathrm { s y s } } , \{ x _ { i , \tau } \} _ { i \in \mathcal { T } } ) \in \mathcal { S } ,\tag{92}
$$

where $x _ { \tau } ^ { \mathrm { s y s } }$ describes the distribution-network operating condition before flexibility procurement. It contains the baseline bus injections, line flows, voltages, and the flexibility-need indicators published by the DSO,

$$
\boldsymbol x _ { \tau } ^ { \mathrm { s y s } } = \left( \mathbf { P } _ { \tau } ^ { \mathrm { b a s e , i n j } } , \mathbf { P } _ { \tau } ^ { \mathrm { b a s e } } , \mathbf { v } _ { \tau } ^ { \mathrm { b a s e } } , \mathbf { r e q } _ { \tau } ^ { \mathrm { v } } , \mathbf { r e q } _ { \tau } ^ { \mathrm { t h } } \right) .\tag{93}
$$

The local physical state of aggregator i is

$$
x _ { i , \tau } = \left( \mathrm { s o c } _ { i , \tau } , p _ { i , \tau } ^ { \mathrm { p v } } , d _ { i , \tau } , p _ { i , \tau } ^ { \mathrm { p l a n } } \right) ,\tag{94}
$$

which, together with the planned charging in its action, determines its baseline injection and the maximum physically deliverable flexibility $q _ { i , \tau } ^ { \mathrm { p h y s } }$ through equation 87. Network parameters, battery ratings, and other fixed technical parameters are omitted from the dynamic state.

Observation. Each aggregator observes its own operating state together with the public network information released by the DSO. Its observation is

$$
o _ { i , \tau } = \left( \mathrm { s o c } _ { i , \tau } , p _ { i , \tau } ^ { \mathrm { p v } } , d _ { i , \tau } , \mathbf { r e q } _ { \tau } ^ { \mathrm { v } } , \mathbf { r e q } _ { \tau } ^ { \mathrm { t h } } \right) = \Omega _ { i } ( s _ { \tau } ) \in \mathcal { O } _ { i } .\tag{95}
$$

Thus, each aggregator knows its own available flexibility and the locations and magnitudes of the network needs published before bidding. It does not observe the private operating states, delivery costs, or contemporaneous offers of other aggregators. The market is therefore partially observable from the perspective of each participant.

Action. At each trading interval, aggregator i submits one price–quantity offer together with its planned battery charging,

$$
a _ { i , \tau } = \left( \pi _ { i , \tau } , \bar { q } _ { i , \tau } , p _ { i , \tau } ^ { \mathrm { p l a n } } \right) \in \mathcal { A } _ { i } ,\tag{96}
$$

subject to

$$
c ^ { \mathrm { r e p } } \leq \pi _ { i , \tau } \leq \overline { { \pi } } , \qquad 0 \leq \bar { q } _ { i , \tau } \leq q _ { i , \tau } ^ { \mathrm { p h y s } } , \qquad 0 \leq p _ { i , \tau } ^ { \mathrm { p l a n } } \leq \bar { p } _ { i , \tau } ^ { \mathrm { c h } } .\tag{97}
$$

The offer price $\pi _ { i , \tau }$ determines the payment requested per unit of flexibility, $\bar { q } _ { i , \tau }$ specifies the maximum quantity the aggregator is willing to provide, and $p _ { i , \tau } ^ { \mathrm { p l a n } }$ enters the baseline injection (81). Here, $c ^ { \mathrm { r e p } }$ is the replacement cost of one delivered MWh, i.e., the energy price and the degradation cost corrected for the round-trip efficiency, $\overline { { \pi } } < \mathrm { V O L I }$ is the market price cap, and $\bar { p } _ { i , \tau } ^ { \mathrm { c h } }$ is the charging power the battery can absorb in the interval given its state of charge. The DSO may accept any quantity $q _ { i , \tau } \in [ 0 , \bar { q } _ { i , \tau } ]$ . The joint action is $a _ { \tau } = ( a _ { i , \tau } ) _ { i \in \mathbb { Z } }$

Market clearing and reward. Given the current state and joint offers, the DSO applies the network-constrained clearing mechanism

$$
\begin{array} { r } { \boldsymbol { z } _ { \tau } = \mathcal { C } ^ { \mathrm { f l e x } } ( \boldsymbol { s } _ { \tau } , \boldsymbol { a } _ { \tau } ) , } \end{array}\tag{98}
$$

which solves equation 88. The clearing outcome contains the accepted flexibility quantities $q _ { i , \tau } .$ resulting bus injections, line flows and voltages, together with any emergency load curtailment.

Under the pay-as-bid settlement rule, the revenue of aggregator i is

$$
\mathrm { r e v e n u e } _ { i , \tau } = \Delta ^ { \mathrm { f e x } } \pi _ { i , \tau } q _ { i , \tau } .\tag{99}
$$

Its stage reward is the realized profit,

$$
r _ { i , \tau } = \mathrm { p r o f t } _ { i , \tau } = \mathrm { r e v e n u e } _ { i , \tau } - C _ { i } ^ { \mathrm { f l e x } } \left( q _ { i , \tau } ; x _ { i , \tau } \right) ,\tag{100}
$$

where $C _ { i } ^ { \mathrm { f l e x } } = \Delta ^ { \mathrm { f l e x } } \left[ \lambda _ { \tau } p _ { i , \tau } ^ { \mathrm { c h } } + c _ { i } ^ { \mathrm { c y c } } \left( p _ { i , \tau } ^ { \mathrm { c h } } + p _ { i , \tau } ^ { \mathrm { d i s } } \right) \right]$ is the true cost of the interval, the energy bought for charging at the exogenous price λ<sub>τ</sub> plus the degradation cost $c _ { i } ^ { \mathrm { c y c } }$ per MWh of battery throughput, with $p ^ { \mathrm { c h } }$ and $p ^ { \mathrm { d i s } }$ given by the state transition below. This cost depends on the aggregator’s physical operating state and the quantity actually delivered rather than on its submitted offer price.

State transition. An accepted flexibility award changes the battery operating trajectory and therefore affects the flexibility available in subsequent intervals. Let

$$
q _ { i , \tau } ^ { \mathrm { r e d } } = \operatorname* { m i n } \Bigl ( q _ { i , \tau } , { p } _ { i , \tau } ^ { \mathrm { p l a n } } \Bigr )\tag{101}
$$

denote the part of the flexibility award delivered by reducing planned charging, and let

$$
q _ { i , \tau } ^ { \mathrm { d i s } } = \operatorname* { m a x } \Bigl ( q _ { i , \tau } - p _ { i , \tau } ^ { \mathrm { p l a n } } , 0 \Bigr )\tag{102}
$$

denote the remaining part delivered through battery discharge. The realized battery charging and discharging powers are therefore

$$
p _ { i , \tau } ^ { \mathrm { c h } } = p _ { i , \tau } ^ { \mathrm { p l a n } } - q _ { i , \tau } ^ { \mathrm { r e d } } , \qquad p _ { i , \tau } ^ { \mathrm { d i s } } = q _ { i , \tau } ^ { \mathrm { d i s } } .\tag{103}
$$

The state of charge evolves according to

$$
\mathrm { s o c } _ { i , \tau + 1 } = \mathrm { s o c } _ { i , \tau } + \frac { \Delta ^ { \mathrm { f l e x } } } { E _ { i } } \left( \eta _ { i } ^ { \mathrm { c h } } p _ { i , \tau } ^ { \mathrm { c h } } - \frac { p _ { i , \tau } ^ { \mathrm { d i s } } } { \eta _ { i } ^ { \mathrm { d i s } } } \right) .\tag{104}
$$

Hence, providing flexibility in the current interval can reduce the battery energy available for future flexibility provision. Photovoltaic generation, local demand, planned battery operation, and the resulting network operating conditions evolve according to their exogenous data processes. The overall state transition is

$$
s _ { \tau + 1 } \sim P ( \cdot \mid s _ { \tau } , a _ { \tau } ) .\tag{105}
$$

Agent objective. Each aggregator uses a policy $\sigma _ { i } ( a _ { i , \tau } \mid o _ { i , \tau } )$ to determine its flexibility offer. Over a horizon of H trading intervals, aggregator i seeks to maximize its expected discounted profit,

$$
J _ { i } ( \sigma _ { i } , \sigma _ { - i } ) = \mathbb { E } \left[ \sum _ { \tau = 1 } ^ { H } \gamma ^ { \tau - 1 } r _ { i , \tau } \right] ,\tag{106}
$$

where $\sigma _ { - i }$ denotes the joint policies of the other aggregators. Because the DSO jointly selects offers according to their submitted prices, available quantities, locations, and effects on network constraints, the return of each aggregator depends on both its own bidding policy and those of the other participants. Aggregators therefore learn price–quantity offering strategies through repeated market interactions without observing competitors’ private states, costs, or contemporaneous offers.

![](images/8d574df08c78065b5f78f64f1d02a620874b238e30099f278930b488f9e0da76.jpg)  
Figure 7: Throughput of the peer-to-peer local energy market environment (1 200 households, 96 periods per episode) under three layouts: (a) against the number of parallel environments; (b) against the number of host CPU threads allotted, with 64 parallel environments. Each point is the mean of three runs with a bootstrap 95% confidence interval; each dotted line joins two points and the number beside it is their throughput ratio.

## B COMPUTATION PERFORMANCE

This section compares the throughput of the same environment code under three layouts: all on the GPU; environment on the host CPU, policy on the GPU; all on the host CPU. With 64 parallel environments, the throughput of all on the GPU is 2.3 to 22 times that of all on the host CPU and 2.3 to 33 times that of environment on the host CPU, policy on the GPU, in every one of the five markets, and it does not change with the number of host threads allotted. Figure 7 compares the three layouts in the market with the most participants, the M4 peer-to-peer local energy market with 1 200 households; Figure 8 covers all five markets, and Figure 9 open-source tools. All experiments run under Python 3.11.15 with JAX and jaxlib 0.10.2, CUDA support added as the same-version plugin rather than by changing either, and double precision and highest matmul precision enabled, on a single shared workstation with three NVIDIA RTX 4500 Ada (24 GB) GPUs and an AMD Ryzen Threadripper PRO 7985WX host (64 cores, 128 threads); each measurement point is run three times and compilation time is excluded. Throughput is the number of environment steps advanced per second, summed over the parallel environments, where one step covers the market clearing and one forward pass of the policy, without gradient updates.

## B.1 THREE LAYOUTS ON THE PEER-TO-PEER MARKET

In the peer-to-peer local energy market with 64 parallel environments, all on the GPU advances 27 929 environment steps per second, 22 times the throughput of all on the host CPU and 33 times that of environment on the host CPU, policy on the GPU (Figure 7a). Raising the number of parallel environments from 64 to 1 024 leaves the throughput unchanged, and so does the number of host threads allotted (Figure 7b).

## B.2 ACROSS THE FIVE MARKETS

All five markets show the speed and throughput advantage of PowerMarketJax: with 64 parallel environments, all on the GPU gives 2.3 to 22 times the throughput of all on the host CPU and 2.3 to 33 times that of environment on the host CPU, policy on the GPU (Figure 8a); all on the GPU, raising the number of parallel environments from 1 to 128 multiplies the throughput by 2.9 to 12 (Figure 8b).

## B.3 AGAINST OPEN-SOURCE TOOLS

The clearing problem of the M2 real-time balancing market was also implemented in PyPSA (Brown et al., 2018) and in cvxpy (Diamond & Boyd, 2016; Agrawal et al., 2018), each in its default usage (default formulation and solver), stepping one environment at a time. All on the GPU with 64 parallel environments gives 71 times the throughput of PyPSA and 8.8 times that of cvxpy (Figure 9). Training PPO on the real-time balancing market for the same number of environment steps, PowerMarketJax all on the GPU is 3.8 times faster than Stable-Baselines3 (Raffin et al., 2021) with its default settings (wall-clock time, including gradient updates).

![](images/65522b83e34505d9421c565f5a3aefae84c05065714d23417d0c2ea0ba64a816.jpg)  
Figure 8: Throughput of the five market environments: (a) the three layouts with 64 parallel environments, the number above each host bar being the ratio of all on the GPU to that layout; (b) all on the GPU with 1 and with 128 parallel environments, the number above each pair being the ratio of the two bars. Each bar is the mean of three runs.

![](images/69ab4fc53561a39c4dae1d2bb186c5504781d10625a840f261d95783b910c9f5.jpg)  
Figure 9: Throughput of the real-time balancing market environment in three implementations: PyPSA 1.3.0 and cvxpy 1.9.3 in their default usage, one environment at a time, and PowerMarketJax all on the GPU with 1 and with 64 parallel environments. Each bar is the mean of three runs; the label above each grey bar is how many times slower it is than PowerMarketJax with 64 parallel environments.

## B.4 IMPLEMENTATION VALIDATION

We validate the implementation in three ways: market clearing and settlement against a NumPy reference implementation; the dual prices of the JAX linear-program solver against HiGHS (Huangfu & Hall, 2018), an independent solver; and the three-stage clearing of the M1 day-ahead market against the exact solution of the same commitment problem, computed by HiGHS with the commitment decisions kept binary.

Against a NumPy reference implementation. Each market has a separate NumPy reference im plementation that rewrites the same clearing and settlement rules with explicit matrices and loops. We run both on every period of the evaluation days of each market under truthful offers and compare each quantity (Table 1). In M3, load shedding and reserve shortfall are compared as totals per period, because the optimum does not fix how they are split across buses and reserve products.

Table 1: Largest difference between the JAX implementation and the NumPy reference. The relative difference divides the absolute difference by the largest absolute reference value.
<table><tr><td>Market</td><td>Quantity</td><td>Data</td><td>Absolute difference</td><td>Relative difference</td></tr><tr><td rowspan="3">M1</td><td>LMP</td><td rowspan="3"> $3 6 \mathrm { d a y s } \times 2 4 \mathrm { h } ,$  case29gb</td><td> $2 . 2 \times 1 0 ^ { - 8 } \ S / \mathrm { M W h }$ </td><td> $1 . 4 \times 1 0 ^ { - 1 0 }$ </td></tr><tr><td>Award Objective value</td><td> $1 . 3 \times 1 0 ^ { - 1 1 } \mathrm { M W }$   $9 . 3 \times 1 0 ^ { - 1 0 } \ : \ S$ </td><td> $1 . 7 \times 1 0 ^ { - 1 5 }$   $4 . 9 \times 1 0 ^ { - 1 6 }$ </td></tr><tr><td>Settlement (profit)</td><td> $2 . 8 \times 1 0 ^ { - 9 } \ : \ S$ </td><td> $6 . 3 \times 1 0 ^ { - 1 6 }$ </td></tr><tr><td>M2</td><td>LMP Award Settlement (profit) LMP</td><td> $3 6 \mathrm { d a y s } \times 4 8$   $\mathrm { h a l f - h o u r s , c a s e 2 9 g b }$ </td><td> $9 . 1 \times 1 0 ^ { - 8 } \ S / \mathrm { M W h }$   $1 . 6 \times 1 0 ^ { - 1 1 } \mathrm { M W }$   $1 . 5 \times 1 0 ^ { - 1 0 } \ : \ S$   $7 . 3 \times 1 0 ^ { - 6 } \mathbb { S } / \mathrm { M W h }$ </td><td> $9 . 1 \times 1 0 ^ { - 1 2 }$   $2 . 1 \times 1 0 ^ { - 1 5 }$   $1 . 7 \times 1 0 ^ { - 1 7 }$   $7 . 3 \times 1 0 ^ { - 1 0 }$ </td></tr><tr><td>M3</td><td>Reserve price Award Load shed, total Reserve shortfall, total Objective value Settlement (profit)</td><td>36 days × 48 half-hours, case29gb</td><td> $1 . 2 \times 1 0 ^ { - 4 } \ S / \mathrm { M W h }$   $5 . 1 \times 1 0 ^ { - 9 } \mathrm { M W }$   $1 . 4 \times 1 0 ^ { - 1 1 } \mathrm { M W }$   $9 . 3 \times 1 0 ^ { - 1 2 } \mathrm { M W }$   $3 . 0 \times 1 0 ^ { - 4 } \ : \ S$   $4 . 7 \times 1 0 ^ { - 2 } \ : \ S$ </td><td> $4 . 7 \times 1 0 ^ { - 7 }$   $6 . 5 \times 1 0 ^ { - 1 3 }$   $7 . 7 \times 1 0 ^ { - 1 5 }$   $4 . 0 \times 1 0 ^ { - 1 5 }$   $3 . 0 \times 1 0 ^ { - 1 1 }$   $5 . 5 \times 1 0 ^ { - 9 }$ </td></tr><tr><td>M4</td><td>Clearing price Traded volume Award Settlement (profit) Award</td><td>64 episodes × 96 quarter-hours, 1 200 households</td><td> $6 . 1 \times 1 0 ^ { - 6 } \mathrm { E U R / M W h }$   $1 . 4 \times 1 0 ^ { - 8 } \mathrm { M W h }$   $1 . 9 \times 1 0 ^ { - 8 } \mathrm { M W h }$   $1 . 8 \times 1 0 ^ { - 6 } \mathrm { E U R }$ </td><td> $1 . 8 \times 1 0 ^ { - 8 }$   $1 . 0 \times 1 0 ^ { - 7 }$   $3 . 9 \times 1 0 ^ { - 6 }$   $1 . 0 \times 1 0 ^ { - 6 }$ </td></tr><tr><td>M5</td><td>Load shed Objective value Settlement (profit)</td><td> $3 6 \mathrm { d a y s } \times 2 4 \mathrm { h } .$  feeder 459_0</td><td> $3 . 4 \times 1 0 ^ { - 9 } \mathrm { M W }$   $1 . 5 \times 1 0 ^ { - 9 } \ : \mathrm { M W }$   $5 . 2 \times 1 0 ^ { - 7 } \mathrm { C H F }$   $5 . 2 \times 1 0 ^ { - 7 } \mathrm { C H F }$ </td><td> $1 . 3 \times 1 0 ^ { - 8 }$   $5 . 0 \times 1 0 ^ { - 7 }$   $1 . 1 \times 1 0 ^ { - 9 }$   $1 . 5 \times 1 0 ^ { - 8 }$ </td></tr></table>

Dual prices against HiGHS. The prices of M1, M2 and M3 are duals of a linear program. On every period of the same evaluation days, we solve the same linear program with HiGHS and compare its prices with those of the JAX solver (Table 2). M4 and M5 set prices without a linear program.

Table 2: Largest difference between the JAX solver and HiGHS.
<table><tr><td>Market</td><td>Test system</td><td>Price</td><td>Periods</td><td>Price difference ($/MWh)</td><td>Relative objective difference</td></tr><tr><td>M1</td><td>case29gb</td><td>LMP</td><td>864</td><td> $2 . 6 \times 1 0 ^ { - 8 }$ </td><td> $3 . 1 \times 1 0 ^ { - 1 6 }$ </td></tr><tr><td>M1</td><td>case73rts</td><td>LMP</td><td>864</td><td> $4 . 1 \times 1 0 ^ { - 8 }$ </td><td> $6 . 2 \times 1 0 ^ { - 8 }$ </td></tr><tr><td>M1</td><td>case813nem</td><td>LMP</td><td>864</td><td> $3 . 5 \times 1 0 ^ { - 7 }$ </td><td> $1 . 2 \times 1 0 ^ { - 7 }$ </td></tr><tr><td>M2</td><td>case29gb</td><td>LMP</td><td>1728</td><td> $1 . 1 \times 1 0 ^ { - 7 }$ </td><td> $1 . 8 \times 1 0 ^ { - 1 2 }$ </td></tr><tr><td>M3</td><td>case29gb</td><td>LMP</td><td>1728</td><td> $7 . 3 \times 1 0 ^ { - 6 }$ </td><td> $3 . 1 \times 1 0 ^ { - 1 1 }$ </td></tr><tr><td>M3</td><td>case29gb</td><td>Reserve price</td><td>1728</td><td> $2 . 1 \times 1 0 ^ { - 7 }$ </td><td> $3 . 1 \times 1 0 ^ { - 1 1 }$ </td></tr></table>

Gap to the exact optimum. The M1 day-ahead clearing has binary commitment variables, and no mixed-integer solver runs inside the compiled graph, so the market is cleared by a three-stage procedure: a linear program with the on/off decisions relaxed to values between 0 and 1, rounding of the relaxed commitment, and a re-solve of the economic dispatch with the commitment fixed. On the 36 evaluation days of each of the three M1 test systems, under truthful offers, we solve the same unit commitment problem exactly and offline with HiGHS (commitment variables 0 or 1) and compare its production cost with that of the three-stage procedure.

Over the 36 evaluation days, the median excess production cost of the three-stage procedure over the exact optimum is 2.04% on case29gb, 0 on case73rts and 1.10% on case813nem, and the maximum is 3.78%, 1.79% and 1.95% (Figure 10). On 23 of the 36 evaluation days of case73rts, the three-stage commitment is the same as the exact optimum and the gap is 0.

![](images/a8eb187eb16ca7e817bd813afe11c7572f8193a3274f60a7e476a09e7664f4ca.jpg)  
One evaluation day Median over the 36 days  
Figure 10: Production cost of the three-stage clearing above the exact optimum on the evaluation days of the three M1 test systems, under truthful offers. Each point is one evaluation day, in date order; the dashed line is the median.

## C QUICKSTART AND API

This section is a walk-through of how to install PowerMarketJax, run a minimal rollout, read the interface contract that the five markets share, construct each of them, find where an experimental configuration is written down, and reproduce a result from the command line. Listing C.2 is a complete runnable program, Listing C.3 a rollout function that takes an environment built by any of the five constructors, Listing C.5 is abridged from a released source file, and the bash blocks in Sections C.1 and C.6 cover installation and experiment script usage. The market models themselves, including the observation, action, clearing and settlement of each market, are in Appendix A; this section covers only the software interface to them.

## C.1 INSTALLATION

PowerMarketJax requires Python 3.10–3.12 and has no build step. The base install ships a CPUonly JAX; CUDA 12 support is opt-in via the cuda extra, which adds the CUDA plugin alongside the already resolved jaxlib at the same version and therefore does not move the JAX version the suite is verified on. GPU support follows JAX’s own platform support: the CUDA build is available on Linux only, so on Windows the package runs on the CPU natively and on the GPU through WSL2. All reported results and test runs were obtained on Linux. The learning stack of rlax and distrax used by the P2P trading experiment script lives behind rl, and the figure scripts behind figs. Nothing under powermarketjax/ imports matplotlib, rlax or distrax, so the five markets and both learners are available on a dev install alone.

```shell
Listing C.1: PowerMarketJax Installation commands
git clone https://github.com/powermarketjax/PowerMarketJax
conda create -y -n powermarketjax python=3.11 && conda activate
powermarketjax
pip install -e ".[dev]" # CPU JAX, the five markets,
both learners
pip install -e ".[dev,cuda,rl,figs]" # Linux + CUDA 12, plus
scripts and figures
python -c "import powermarketjax, jax; print(powermarketjax.
__version__, jax.__version__, jax.devices())"
```

## C.2 MINIMAL ROLLOUT

Listing C.2 opens the day-ahead wholesale market on the 29-bus GB case, resets it, and runs a three-step rollout under jax.lax.scan with every agent bidding at twice its cost. One step is one market day, so the rollout clears three consecutive days of a multi-period security-constrained unit commitment; every transition, the clearing included, stays inside jit.

```python
Listing C.2: Minimal rollout on case29gb (day-ahead wholesale)
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True) # M1/M2/M3/M5 refuse to
build without it
from powermarketjax.case import load_case
from powermarketjax.envs.day_ahead import make_env, load_commitment,
load_gb_demand
from powermarketjax.utils.jax_utils import scan_rollout
env, spec = make_env(load_case("29gb"),
load_commitment(n_periods=24), load_gb_demand(),
kind="markup", markup_max=2.0,
cap_scale=0.60, ramp_scale=1.00)
params = env.make_params(episode_len=3)
```

```julia
obs, state = env.reset(jax.random.PRNGKey(0), params)
actions = jnp.full((3, spec["n_agents"]), 2.0) # (n_steps,
n_agents)
final_state, obs_traj, reward_traj, cost_traj, done_traj, info_traj =
jax.jit(lambda k, s: scan_rollout(env, k, s, params, actions)
)(jax.random.PRNGKey(1), state)
```

At this configuration the market has 66 agents, one per thermal unit: obs traj is (3, 66, 112), reward traj is (3, 66) and cost traj is (3, 66, 2), and info traj["converged"] reports per step whether the clearing met its tolerance. scan rollout drives step auto reset, so episodes restart at their boundary without a Python conditional and the trajectory has a fixed length.

The scenario parameters have no defaults. cap scale and ramp scale in Listing C.2 are required: omitting either raises ValueError. The two scale the line ratings and the ramp limits the case ships with, so cap scale 1.00 means every line keeps the capacity written in the case file and in that network no line ever reaches its limit. With no line at its limit there is no congestion, and with no congestion every bus clears at the same price.

Listing C.3 is the same contract on any of the five markets, vectorised over a batch of environments. unpack env accepts whichever shape a market’s constructor returns (Section C.4) and yields the same four objects, so the rollout below is market-agnostic.

Listing C.3: One rollout function for five markets: vmap over environments inside scan over time

```python
import math, jax, jax.numpy as jnp, numpy as np
from functools import partial
from powermarketjax.learning import unpack_env
from powermarketjax.resources.battery import make_battery_bundle
from powermarketjax.envs.p2p import (make_p2p_env, make_p2p_params,
load_fluvius_households)
jax.config.update("jax_enable_x64", True) # M1/M2/M3/M5 refuse to
build without it
# ‘built‘ is whatever a market’s constructor returned: the (env, spec)
pair for M1/M2, or the (reset, step, step_auto_reset, spec)
tuple for M3-M5. Here it is M4, the P2P trading market, on 12
metered houses.
N, DT, ETA = 12, 0.25, math.sqrt(0.85)
series = load_fluvius_households(n_households=N)
battery = make_battery_bundle(n_devices=N, dt_hours=DT,
capacity_mwh=[0.01] <sub>*</sub> N,
power_mw=[0.005] <sub>*</sub> N,
eta_charge=ETA, eta_discharge=ETA,
soc_min=0.15, soc_max=1.0,
initial_soc=0.5,
cycle_cost_per_mwh=0.0)
params = make_p2p_params(p_pv=series.injection, load=series.offtake,
battery=battery,
kappa=np.full(N, 13.88, np.float32),
learner_mask=np.ones(N, bool),
episode_len=96)
built = make_p2p_env(N, 73.0, 333.4, DT) # n_agents, export/retail,
dt
reset, step, step_auto_reset, spec = unpack_env(built)
# unpack_env takes either shape and gives the four names below.
def rollout(key, params, policy, n_envs, n_steps):
k_reset, k_roll = jax.random.split(key)
obs, state = jax.vmap(reset, in_axes=(0, None))(
jax.random.split(k_reset, n_envs), params)
def one_step(carry, k):
```

```python
obs, state = carry
action = policy(obs) # (n_envs,
action_shape)
obs, state, reward, costs, done, info = jax.vmap(
step_auto_reset, in_axes=(0, 0, 0, None))(
jax.random.split(k, n_envs), state, action, params)
return (obs, state), (reward, costs, done)
return jax.lax.scan(one_step, (obs, state),
jax.random.split(k_roll, n_steps))
# The action shape comes from the market, not from the learner.
zero = lambda obs: jnp.zeros((obs.shape[0], <sub>*</sub>spec["action_shape"]),
jnp.float32)
(obs, state), (reward, costs, done) = jax.jit(
partial(rollout, policy=zero, n_envs=8, n_steps=6)
)(jax.random.PRNGKey(0), params)
```

The returned reward has shape (n steps, n envs, N) and costs has shape (n steps, n envs, N, C): time, parallel environments and market participants are three separate axes, and the second and third are the two the accelerator exploits. params is closed over as a broadcast argument, so a sweep over anything carried there is a loop over calls and not a recompilation.

## C.3 THE ENVIRONMENT INTERFACE

Signatures. Every market exposes the same three pure functions:

```julia
Listing C.4: Market environment signatures
obs, state = reset(key, params)
obs, state, reward, costs, done, info = step(key, state, action,
params)
obs, state, reward, costs, done, info = step_auto_reset(key, state,
action, params)
```

With N agents and C constraint channels, obs is (N, d), reward is (N, ), costs is (N, C) and done is a scalar boolean. The agent index is one array axis: observations, actions, rewards and costs are aligned on it, and N is fixed at compile time.

Reward and constraint costs. The reward of an agent is the profit it earns in that step, settled on the quantities and prices produced by the market’s own clearing (Appendix A). Violations of physical or market constraints, such as shed load or an unmet reserve requirement, are returned separately in costs, one column per name in spec["cost names"], and are never subtracted from the reward. A user decides how to handle them, for example by adding a penalty to the reward or by using a constrained learning method.

Episode boundary. spec["termination"] tells a learner what the end of an episode means. In four markets it is "truncation": the episode is cut off in time, the market would continue, and the value of the last state should be bootstrapped. In the P2P market it is "terminal": the last reward already pays each household for the energy left in its battery, so bootstrapping would count that energy twice.

## C.4 CONSTRUCTING THE FIVE MARKETS

Each market has its own constructor, since the scenarios differ: the wholesale markets take a network case, a precomputed day-ahead commitment and schedule, and a demand series; the ancillary market adds the reserve products; the P2P trading market takes a population size and the export and retail tariffs; the local flexibility market takes a feeder and the placement of the aggregators. The ancillary, P2P and local flexibility constructors return (reset, step, step auto reset, spec), and the day-ahead and real-time constructors return (env, spec); powermarketjax.learning.unpack env converts either to the first form, as in

Listing C.3. Anything that changes the size of the clearing problem, such as the network or the placement of agents, is fixed when the environment is built, and changing it requires recompiling; anything that varies within a market, such as the demand series or the episode length, is passed in params on every call.

## C.5 SCENARIO AND LEARNER CONFIGURATION

There is no configuration file format: an experimental configuration is Python. The learner configuration is tools/benchmark/hyperparams.py, abridged in Listing C.5. The scenario parameters are module-level constants of each market’s experiment script, each also exposed as a command-line flag.

```python
Listing C.5: The shared learner configuration
N_ENVS, MINIBATCHES = 64, 32
HORIZON = {"01": 4, "02": 48, "03": 48} # steps per update, per
market
for _m, _h in HORIZON.items(): # the batch must divide into
minibatches
assert (N_ENVS <sub>*</sub> _h) % MINIBATCHES == 0
SHARED = IPPOConfig(
n_envs=N_ENVS,
horizon=HORIZON["02"], # ours, per market via shared_for()
epochs=10, # update_epochs
minibatches=MINIBATCHES,# num_minibatches
lr=3e-4, # learning_rate (no annealing here)
clip_eps=0.2, # clip_coef
gamma=0.99, # gamma
gae_lambda=0.95, # gae_lambda
vf_coef=0.5, # vf_coef
ent_coef=0.0, # ent_coef
max_grad_norm=0.5, # max_grad_norm
hidden=(64, 64), # Agent: two tanh layers of 64
weight_decay=0.0, # zero reproduces optax.adam exactly
init_scale=1.0,
)
def shared_for(market):
return dataclasses.replace(SHARED, horizon=HORIZON[market])
```

## C.6 COMMAND-LINE USAGE

Each market has one experiment script for its learning runs. The non-learning baselines are run by a separate script for the day-ahead, real-time and ancillary markets (tools/benchmark/run eval 0<sub>\*</sub>.py), by the learning script itself for the P2P trading market (--arm truthful) and for the local flexibility market, where they run alongside the learners. The learning script takes the learner (--algo ippo|sac), parameter sharing or not (--per-agent-params for one network per agent), the seed, the iteration count, the scenario parameters and an output directory. It trains on the training days only and then evaluates the mean action of the policy on the held-out days, so the reported number excludes exploration noise.

```perl
Listing C.6: Command line for RL training
# One-year inputs for M1-M3 (2023-07-10 to 2024-07-08), built once.
python tools/commitment/precommit.py --mode relax --start-day 5 --days
365 --cap-scale 0.6 --ramp-scale 1.0 --out commit_y1.npz
python tools/commitment/da_position.py --case 29gb --fixture commit_y1
.npz --out position_y1.npz
# M1 day-ahead: parameter sharing, then one network per agent.
```

```shell
python tools/benchmark/run_rl_01.py --fixture commit_y1.npz --algo
ippo --seed 0 --iterations 400 --out-dir runs/m1-ippo-ps
python tools/benchmark/run_rl_01.py --fixture commit_y1.npz --algo
ippo --seed 0 --iterations 400 --per-agent-params --out-dir runs/
m1-ippo-nops
# M2 real-time and M3 ancillary settle against the day-ahead position.
python tools/benchmark/run_rl_02.py --position position_y1.npz --seed
0 --out-dir runs/m2-ippo-ps
python tools/benchmark/run_rl_03.py --position position_y1.npz --seed
0 --out-dir runs/m3-ippo-ps
# M4 P2P, 1200 households.
python tools/benchmark/run_rl_04.py --agents 1200 --seed 0 --out-dir
runs/m4-ippo-ps
# M5 local flexibility.
PYTHONPATH=tools/flex_experiment python -m concentration_baseline --
algo ippo --seeds 3 --out runs/m5-ippo-ps
# Non-learning baselines for M1 on the held-out days.
python tools/benchmark/run_eval_01.py --fixture commit_y1.npz --days
eval --out-dir results/m1-baselines
```

## C.7 CASES AND DATA

powermarketjax.case.list cases() returns the 18 built-in network cases, 10 transmission and 8 distribution, sorted by bus count and with aliases removed; load case accepts a canonical identifier or an alias. The demand, generation, price and smart-meter series are carried in the repository as parquet files, so every listing above runs on a fresh checkout with no separate download step.

## C.8 CLEARING AND LEARNING IN A SINGLE COMPUTATION GRAPH

A training iteration has two parts, a rollout and an update. In the rollout, N environments advance together for T steps. In each step the participants submit offers, the market clears, the awards are settled, and the next observation is constructed; the policy maps the observation to the action at every step, and the observation, action, reward and constraint cost of one step form one experience. The update takes gradient steps on the N × T experiences of the rollout, split by epoch and minibatch. The whole iteration, including the clearing problem of every step, is compiled as one function on the GPU: the host makes one call and receives the training metrics, and every clearing, every policy forward pass and every gradient step in between runs on the GPU in sequence. Figure 11 shows the data flow of one iteration, and Listing C.7 writes the same iteration as loops, with the JAX primitive that implements each loop in the comment above it. The values of N and T for each market are in Table 4, the epochs and minibatches in Table 3, and n, the number of agents, in Appendix A.

The loop over the N environments is vmap(step). The function step advances one environment by one step and is written for one environment only; vmap returns the function that advances N environments at once, whose input and output gain one axis of length N. The clearing, settlement and observation of the N environments are one batched computation over N sets of data, not N calls one after another. The forward pass of the policy goes through vmap in the same way, so the actions are produced inside the same function and passed directly to step, and the observations never reach the host.

![](images/ecfc0113c968cc753a5c97e103c62cc8ba8d632bf190ce7ebbc6fdd55bbe8fc6.jpg)  
Figure 11: One training iteration of IPPO, compiled as a single function. Rollout: N environments advance in parallel for T steps; in each step the n agents submit offers, the market clears, the awards are settled and the next observation is constructed, and the policy maps the observation to the next action inside the same function. Update: the policy is updated on the $N \times T$ experiences, with a second scan over epochs and minibatches (Table 3). The values of N and T for each market are in Table 4.

```r
Listing C.7: One training iteration as loops, with the primitive that implements each loop
# iteration = jit(iteration): the whole body below is one compiled
function on the GPU, called once per training iteration
iteration(params, states, observations):
# rollout
# scan(one_step, carry, length=T): carry = (states, observations);
# returns the final carry and the T records stacked
for t = 1 .. T:
# policy_N = vmap(policy), step_N = vmap(step):
each takes N inputs and returns N outputs at once
for each of the N environments:
for each of the n agents: action <- policy(observation)
# step(state, actions), one environment
offers <- submit(actions)
awards, prices <- clear(offers)
reward, cost <- settle(awards, prices)
observation <- observe(the state after clearing)
if the episode ended: reset this environment in place
# one record per step, holding the N environments
record (observations, actions, rewards, costs)
# N x T experiences, each with n agents
experiences <- the T records
# update
# scan(gradient_step, ...): the passes over the experiences,
# by epoch and minibatch
for each pass over the experiences:
params <- optimiser step on the gradient of the loss
return params, states, observations, training metrics
```

The loop over the T steps is lax.scan(one step, carry, length=T). It takes the loop body one step, the initial carry, which is the N states and the N observations, and the number of steps T; runs the body T times, feeding the carry returned by one run into the next; stacks the record of each run into an array of length ${ \check { T } } ;$ and returns the final carry together with this array. The horizon T is fixed, and an environment whose episode ends before step $T$ is reset in place inside the body, so the input and output of the body have the same shape at every step, which is the condition under which lax.scan applies. The gradient steps of the update are a second lax.scan. None of these loops runs on the host, and the host does not see the boundary between two consecutive steps.

The outer function is jit(iteration): the rollout followed by the update, compiled for the GPU at the first call and called once per iteration. The host passes in the parameters, the environment states and the observations, and receives the updated parameters, states, observations and training metrics; no intermediate result returns to the host, and the only loop left on the host is the loop over iterations. The iteration of SAC has the same form: the rollout is the same, and the update first places the experiences of the rollout into a replay buffer of fixed size, which also lives inside this function and is carried from one iteration to the next, then samples from it for the gradient steps, which are again a lax.scan.

## D TRAINING SETUP AND HYPERPARAMETERS

The results of Section 5 and Appendix E share four learners, one set of hyperparameters and one evaluation protocol; where a market departs from them, its subsection of Appendix E gives the difference. Appendix C.5 shows where these values are set in the code.

## D.1 FOUR LEARNERS

Each market is run with four learners: IPPO-PS, IPPO-NoPS, SAC-PS and SAC-NoPS. All four are independent learners: every agent sees only its own observation and optimises only its own profit, with no centralised value network. PS and NoPS stand for parameter sharing and no parameter sharing. Under PS all agents share one set of networks and pool their samples for training; under NoPS each agent has its own set and trains only on its own samples. For IPPO the set is one actor and one value network that share two tanh layers, and the log standard deviation of the actor does not depend on the observation; for SAC it is one stochastic actor and a pair of Q-networks with their target networks, with an automatically tuned temperature.

## D.2 HYPERPARAMETERS

The hyperparameters build on the defaults of the continuous-action PPO and SAC implementations in CleanRL; the five markets use one set, listed in Table 3, without tuning for each market. The profit of one step in M1 is of the order of 10<sup>5</sup> to 10<sup>6</sup> dollars and the Q regression of SAC is sensitive to this magnitude, so SAC divides the reward by a constant before the Q regression: the standard deviation of the rewards under truthful offers, computed once before training. SAC runs the gradient steps of an iteration as one block after the rollout. In all four learners each component of the observation is normalised with constants fixed before training.

Table 3: Hyperparameters of the two IPPO and the two SAC learners, shared by the five markets.
<table><tr><td>Hyperparameter</td><td>IPPO</td><td>SAC</td></tr><tr><td>Optimiser</td><td>Adam</td><td>Adam</td></tr><tr><td>Learning rate</td><td> $3 \times 1 0 ^ { - 4 } .$  , constant</td><td>actor  $3 \times 1 0 ^ { - 4 } ,$  , Q-networks and temperature 10 3</td></tr><tr><td>Discount factor γ</td><td>0.99</td><td>0.99</td></tr><tr><td>Hidden layers</td><td>two layers of 64 units, tanh</td><td>two layers of 256 units, ReLU</td></tr><tr><td>GAE parameter λ</td><td>0.95</td><td></td></tr><tr><td>Clipping range</td><td>0.2</td><td></td></tr><tr><td>Update epochs per iteration</td><td>10</td><td></td></tr><tr><td>Minibatches per epoch</td><td>32</td><td></td></tr><tr><td>Value loss coefficient</td><td>0.5</td><td></td></tr><tr><td>Entropy coefficient</td><td>0</td><td></td></tr><tr><td>Maximum gradient norm</td><td>0.5</td><td></td></tr><tr><td>Initialisation scale of the actor output layer</td><td>1.0</td><td></td></tr><tr><td>Replay buffer</td><td></td><td>32 768 environment steps</td></tr><tr><td>Batch size</td><td></td><td>256 environment steps, each carrying the transitions of all agents</td></tr><tr><td>Gradient steps per environment step</td><td></td><td>1</td></tr><tr><td>Random-action steps before training</td><td></td><td>0</td></tr><tr><td>Actor update interval</td><td></td><td>every gradient step</td></tr><tr><td>Target smoothing coefficient τ</td><td></td><td>0.005</td></tr><tr><td>Temperature</td><td></td><td>initial value 0.2, tuned automatically</td></tr><tr><td>Target entropy</td><td></td><td>minus the action dimension</td></tr><tr><td>Range of the log standard</td><td></td><td></td></tr><tr><td>deviation</td><td></td><td>[−5,2]</td></tr></table>

The two SAC learners of M4 and the two IPPO learners of M5 differ from Table 3 in the settings given in Sections E.4.2 and E.5.2.

## D.3 TRAINING SCALE

Table 4 gives the training scale of each market.

Table 4: Training scale.
<table><tr><td>Market</td><td>Seeds per learner</td><td>Iterations</td><td>Parallel environments and steps per iteration</td><td>Environment steps per iteration</td></tr><tr><td>M1 day-ahead wholesale</td><td>3</td><td>400</td><td>64 environments, 4 steps each</td><td>256</td></tr><tr><td>M2 real-time balancing</td><td>3</td><td>200</td><td>64 environments, 48 steps each</td><td>3072</td></tr><tr><td>M3 ancillary services</td><td>3</td><td>200</td><td>64 environments, 48 steps each</td><td>3072</td></tr><tr><td>M4 peer-to-peer local energy</td><td>3</td><td>400</td><td>64 environments, 96 steps each</td><td>6144</td></tr><tr><td>M5 local flexibility</td><td>3 to 10</td><td>early stoppinga, at most 300</td><td>64 environments, 24 steps each</td><td>1536</td></tr></table>

<sup>a</sup> M5 evaluates a checkpoint on the validation days every 10 iterations and stops early once the mean validation return of the last 5 checkpoints exceeds that of the 5 checkpoints before them by 0 to 5%.

## D.4 EVALUATION PROTOCOL

M1 to M4 evaluate the policy at the last iteration. In M1 to M3, evaluation takes the mean action of the policy and runs day by day on the 36 evaluation days of the one-year window of case29gb, 2023-07-10 to 2024-07-08 (the dates of the other two test systems are in Table 5). In M4, evaluation also takes the mean action and runs on 1 024 held-out episodes (Table 7). In M5, the checkpoint with the highest validation return is evaluated on the 36 test days with sampled actions (Section E.5.2). The training return is the return under sampled actions and includes exploration noise.

The untrained network is the network of the same parameter layout (PS or NoPS) at iteration 0, evaluated in the same way.

Truthful offers are the reference policy in which every agent offers at its true cost: in M1 to M3 every unit takes the markup $\alpha _ { i } = 1$ , with a reserve offer of 0 in M3; in M4 the batteries are kept still, sellers ask the export price and buyers bid the retail tariff; and in M5 every aggregator offers its full deliverable quantity at the replacement cost and plans no charging.

## E ADDITIONAL MARKET RESULTS

## E.1 M1: DAY-AHEAD WHOLESALE MARKET

This section first presents detailed data, training curves and evaluation results of the three test systems, and then elaborates the finding in Section 5.1 with the supply curves, the offer distribution and the joint sampling probability on the British test system.

## E.1.1 MARKET SETTING

Each unit i offers a price–quantity curve with a single segment $( K = 1$ in Section A.1.1). Its offer price is $\alpha _ { i } m _ { i } ,$ where $m _ { i }$ is its segment cost (the mean of its true marginal cost over its output range) and $\alpha _ { i } \in [ 1 , \bar { \alpha } ]$ is its markup. At $\alpha _ { i } = 1$ the offer is truthful, and $\bar { \alpha } > 1$ is the markup cap. A markup above one represents economic withholding: the unit offers its capacity at a price above its cost.

We consider three test systems. As shown in Table 5, each test system uses one year of data, of which 36 days are used for evaluation and the rest for training. On each test system, three selected days are taken from the 36 evaluation days: the days at the 10th, 50th and 90th percentiles of total daily demand, referred to below as the low-load day, the mid-load day and the high-load day.

Table 5: Data of the three test systems.
<table><tr><td>Test system</td><td>Grid</td><td>Demand data</td><td>Time span</td><td>Training / evaluation days</td><td>Markup cap α</td></tr><tr><td>case29gb</td><td>A reduced 29-bus model of the GB transmission networka (network diagramb), 66 units</td><td>GB transmission system demand (NESO)c, day-ahead forecast from Elexond</td><td>2023-07-10 to 2024-07-08</td><td>329 / 36</td><td>2</td></tr><tr><td>case73rts</td><td>The RTS-GMLC test systeme, 73 units</td><td>Load, wind, solar and hydro of RTS-GMLCf</td><td>2020-01-01 to 2020-12-31</td><td>330 / 36</td><td>1.4</td></tr><tr><td>case813nem</td><td>An open grid model of the Australian National Electricity Market®, 151 units</td><td>Operational demand and day-ahead forecast of the four mainland regions from the Australian Energy Market Operator (AEMO)h</td><td>2025-02-01 to 2026-01-31</td><td>329 / 36</td><td>2</td></tr></table>

<sup>a</sup> https://webhomes.maths.ed.ac.uk/OptEnergy/NetworkData/reducedGB/;  
<sup>b</sup> https://webhomes.maths.ed.ac.uk/OptEnergy/NetworkData/reducedGB/GBreducednetwork.pdf;  
<sup>c</sup> https://www.neso.energy/data-portal/historic-demand-data; <sup>d</sup> https://bmrs.elexon.co.uk/;  
<sup>e</sup> https://github.com/GridMod/RTS-GMLC;  
<sup>f</sup> https://github.com/GridMod/RTS-GMLC/tree/master/RTS\_Data/timeseries\_data\_files;  
<sup>g</sup> https://github.com/akxen/egrimod-nem-dataset;  
<sup>h</sup> https://visualisations.aemo.com.au/aemo/nemweb/index.html.

In the British test system case29gb, generation lies mostly in the north and load mostly in the south, and power flows from north to south: the north has 63% of the total capacity but only 43% of the load. The three example days are 2024-06-23, 2024-03-16 and 2024-01-23. Under truthful offers, 2, 3 and 5 lines are congested on these days, all on central and southern corridors (Figure 12). Of the 66 units, 14 are nuclear, 23 coal and 29 gas. In order of segment cost, all 14 nuclear units are among the 16 cheapest units, and their total capacity is only 4.359 GW. These 16 units have a total capacity of 19.2 GW, while hourly system demand over one year ranges from 17.2 to 47.4 GW and is above 19.2 GW in almost every hour. The nuclear units therefore sit near the left of the supply curve with demand almost always to their right, which means a unilateral markup by one nuclear unit does not move the price, and only a joint markup raises it.

In the RTS-GMLC test system, case73rts, wind, solar and hydro (referred to below as renewables) do not submit offers, and the demand cleared by the market is the net demand (total load minus renewable output). The three example days are 2020-02-24, 2020-03-24 and 2020-07-16. Under truthful offers, 0, 2 and 7 lines are congested on these days (Figure 13).

![](images/4ba91741b0cb3104d202c47bf10e2706726f6b26326f38c2426436b28281ece7.jpg)  
Figure 12: The test system case29gb. Left: the 66 units at their buses, coloured by fuel and sized by maximum output; corridor width scales with capacity. Middle: annual mean bus load and the corridors congested under truthful offers on the three example days. Right: the supply curve ordered by segment cost, against the range of hourly system demand over one year. Generation sits mostly in the north and load mostly in the south, so the congested corridors are central and southern. All 14 nuclear units, 4 359 MW in total, are among the 16 cheapest units, which sum to 19.2 GW; hourly demand ranges from 17.2 to 47.4 GW and exceeds 19.2 GW in almost every hour. The nuclear units therefore sit near the left of the supply curve with demand almost always to their right: a unilateral markup does not move the price, while a joint markup lifts the whole curve and the price with it.

Regarding the Australian test system, case813nem, the three example days are 2025-09-20, 2025-12- 10 and 2025-06-24. Under truthful offers, 3, 4 and 4 lines are congested on these days (Figure 14).

On all three test systems the market clears the demand that actually occurs on the day, while the units observe the day-ahead forecast before they submit offers (Figure 15). On case73rts, net demand has a floor of 2 500 MW, and it sits at that floor in more than half of the hours of the year. On case813nem, demand has a floor of 11 500 MW, reached in only a few hours.

## E.1.2 AGENT TRAINING

The training setup of the four learners is that of Appendix D. On the British test system, the training return of IPPO-NoPS, SAC-PS and SAC-NoPS ends close to its highest level (top row of Figure 16). IPPO-PS stays below the other three and is the only one whose return rises and then falls; the markup it samples also rises to about 1.7 and then returns to about 1.5 (Figure 3a).

On the other two test systems only IPPO-PS and IPPO-NoPS are trained (bottom row of Figure 16). The training return of both learners falls from its starting level. The fall comes from competition among the units: at the start of training every unit offers near the middle of its markup range, which amounts to a joint markup; during training the units that set the price learn to offer closer to cost, because a unit that stays high alone loses its output to its rivals, so the price falls while production cost does not, and profit falls with it.

![](images/7b76d2fe876538de030680d95289bc4a33465920e7e2bcf446d208f19769ad09.jpg)  
Figure 13: The test system case73rts, drawn as Figure 12: the three-area RTS-GMLC system with 73 units. Left: units by fuel and maximum output. Middle: annual mean bus load and the lines congested under truthful offers on the three example days. Right: the supply curve ordered by segment cost against the range of hourly net demand over one year. Gas units make up most of the supply curve; no line is congested on the low-load day and seven are on the high-load day.

## E.1.3 DISCUSSION OF RESULTS

On each test system, the policies learned by IPPO-PS and IPPO-NoPS are compared with truthful offers on the 36 evaluation days, each learner taken as the mean of three seeds.

As shown in Figure 17, across all three test systems, the mean LMP under the learned policies (loadweighted over all periods and buses of the day) is higher than under truthful offers on every one of the 36 evaluation days (production cost barely changes) and the profit of all units rises accordingly. Averaged over the 36 days, the mean LMP under the policies learned by IPPO-PS and IPPO-NoPS is 1.50 and 1.56 times, respectively, its value under truthful offers on case29gb, 1.10 and 1.07 times on case73rts, and 1.23 and 1.29 times on case813nem. For both learners, the total production cost over the 36 days differs from that under truthful offers by less than 1%. The increase in mean LMP is smallest on case73rts, where the mean markups of IPPO-PS and IPPO-NoPS are only 1.19 and 1.21, respectively.

## E.1.4 DETAILS OF SUPPLY CURVES

At the peak demand periods of the British test system, the committed units are sorted by offer into a stepped supply curve as shown in Figure 18. Under both truthful and learned offers, the steps of the 14 nuclear units lie far to the left of the demand line. The policy learned by IPPO-PS raises the whole supply curve while the demand line stays in place, and the price rises with it.

## E.1.5 MARKUPS OF DIFFERENT LEARNERS

For each of the four learners the mean markup over the 66 units is near 1.5, but the spread across units differs. IPPO-PS gives almost the same markup to every unit; IPPO-NoPS has the widest spread, with some units near truthful offers and some near the cap; the two SAC learners fall between the two, as shown in Figure 19. For each algorithm, NoPS has a wider spread than PS. All four learners give the 14 nuclear units a mean markup below their own mean over all 66 units, and IPPO-NoPS gives these units markups close to truthful offers.

![](images/bffd597e375a60834dca85e48e3076aa1193b292a0b3362a9b1cc5895abc7248.jpg)  
Figure 14: The test system case813nem, drawn as Figure 12: an open grid model of the mainland Australian National Electricity Market with 151 units. Only 7 of its 1 278 lines carry a published rating. Left: units by fuel and maximum output. Middle: annual mean bus load and the lines congested under truthful offers on the three example days. Right: the supply curve ordered by segment cost against the range of hourly demand over one year. Hydro and then coal fill the left of the supply curve; three lines are congested on all three days and a fourth on the mid- and high-load days.

![](images/5f9ad90fbc51dacb3a9704c8b6b752f22c503a66e3e946dba1b7aa9724ca08f0.jpg)  
case73rts: RTS-GMLC load and renewables, net demand

![](images/8ff4fac55c9f00003c5079272e2d4efefdc75bdb404016f86ca07f1c4947b2d4.jpg)  
Hourly, the week around the mid-load day

![](images/754d83a00e82aabaffd4f210c0fca77c140b75200640d39b6157ef9fc2320a1a.jpg)

![](images/0eeb86fb98e746be93b6842aca607d70184b4e2b12ccce5b92b376d7ca0e8705.jpg)

case813nem: mainland Australian NEM operational demand  
![](images/d58858d0818b394ac0aa9d6dbdb45412a583f171b861ef1ce989d5017e61bb73.jpg)

![](images/debdcdf4371eb4cfb278cb773ec38c068ae57c89b668af0d94f510ca59753415.jpg)

Figure 15: Demand on the three test systems, and the renewable output netted out on case73rts. Left: daily means over the one-year window, with the 36 evaluation days marked. Right: hourly values in the week around the mid-load example day. The thick black line is the demand the market clears; the dashed line is the day-ahead forecast the agents observe. On case73rts the market clears net demand, total load minus the renewable output shown, raised to a floor of 2 500 MW; it sits at that floor in more than half of the hours of the year. Demand on case813nem has a floor of 11 500 MW, reached in a few hours.

![](images/3cce543549cc2a12873650fcf0436c55e3a14d027fef11fb3fee6e4e138b60ea.jpg)  
Figure 16: Training return on the three test systems. Top: the four learners on case29gb. Bottom: IPPO-PS and IPPO-NoPS on case73rts and case813nem. Each line is the mean of three seeds, each seed first smoothed by an 11-iteration centred moving mean, and the shaded band is the bootstrap 95% confidence interval. Return is the mean daily profit per generator over all sampled environment steps of an iteration. On case29gb IPPO-PS is the only learner that rises and then falls; on the othe two systems the return of both learners falls from its starting level.

![](images/7e3571c3b8c7e26e7f19a4a1437fa71d1d0f4c8abcc7abde26fb659ce3d6503e.jpg)  
Figure 17: Daily system quantities on the 36 evaluation days of each test system, truthful offers (black) against the learned offers of IPPO-PS (orange) and IPPO-NoPS (blue); each learned line is the mean of three seeds, with a bootstrap 95% confidence interval. Rows: daily demand, the load weighted mean LMP, production cost, and the total profit of all units.

![](images/cbfcdd1e03257d6b371009edc76a9bd665078d4d01dfe9f4e8d15c22ebecaee1.jpg)

Figure 18: Supply curves at the hour of highest demand on each example day of case29gb, the evaluation days at the 10th, 50th and 90th percentiles of daily demand (2024-06-23, 2024-03-16 and 2024-01-23): committed units sorted by offer under truthful (grey) and learned (orange) offers, with the steps of the 14 nuclear units drawn thicker and the demand in that hour as a dashed line; the shaded bands are the range of LMP over the 29 buses under each set of offers. The learned offers are the mean over three IPPO-PS seeds at the end of training, with a bootstrap 95% confidence interval for each step; the learned LMP band is the range over buses of the seed-mean LMP. The nuclear steps lie far to the left of demand under both sets of offers, and the learned offers raise the whole curve.  
![](images/41521c03b7879dd61a9b2fafc31a0633b37d7e3b1250bdfecd346b00977ea5cc.jpg)  
Figure 19: Markup of each unit under the four learners on case29gb; rows are learners, each the mean over three seeds, columns the 66 units ordered by segment cost. Colour is the markup of that unit, averaged over the 36 evaluation days and the three seeds, on the full action range from 1.00 (truthful) to 2.00 (cap). Black triangles above the top row mark the 14 nuclear units.

## E.2 M2: REAL-TIME BALANCING MARKET

In this section, we discuss the result of the four learners in the RTM balancing market.

## E.2.1 MARKET SETTING

The action of unit i is a markup $\alpha _ { i } \in [ 1 , \bar { \alpha } ]$ on its generation cost (i.e., true marginal cost over its output range); $\alpha _ { i } = 1$ is a truthful offer, and an offer is submitted once per half-hour period. The test systems of this market are case29gb and case73rts (detailed data in Table 5), with the markup cap α¯ set to 2. The grids and units are shown in Figures 12 and 13.

Most output is scheduled day ahead, so real-time offers price only small deviations: on the 36 evaluation days, the median absolute half-hourly demand deviation is 214 MW, less than 1% of mean demand (Figure 20). This limits the gain from raising an offer alone (Section E.2.4).

![](images/e00fb43575bf9eee4525982f8c900819b5f04bd385958da83c7f9a85f589ae7f.jpg)  
Figure 20: Demand on case29gb. Left: the week around the mid-load day (2024-03-16): the halfhourly demand the real-time market clears (teal), the hourly day-ahead schedule summed over buses, which equals the hourly mean of that demand (purple steps), and the day-ahead forecast the units observe (dashed yellow). Right: the deviation of the demand in each half-hour from the mean of its hour, over all 365 days (grey) and the 36 evaluation days (teal). On the evaluation days the absolute deviation has a median of 214 MW, a 90th percentile of 638 MW and a maximum of 1 478 MW, against a mean demand of 27 730 MW. The day-ahead schedule follows the half-hourly demand to within a few hundred MW.

## E.2.2 AGENT TRAINING

The training return is the mean daily profit per unit under sampled actions, which is the objective of the learner itself. Under all four learners the mean markup of the 30 committed units starts between 1.48 and 1.61; at the end of training it is 1.04 under IPPO-PS, and 1.24, 1.40 and 1.39 under IPPO-NoPS, SAC-PS and SAC-NoPS; the markup along training is shown in Figure 4b. On case29gb the training return of IPPO-PS falls throughout and ends lowest of the four learners; IPPO-NoPS and SAC-PS also fall, by less; SAC-NoPS rises first and then stays level (top row of Figure 21). On case73rts IPPO-PS and IPPO-NoPS are trained, and the training return of both rises slowly (bottom row of Figure 21).

## E.2.3 DISCUSSION OF RESULTS

On case29gb, IPPO-PS returns to truthful offers, while the other three learners end with higher total profit and lower system cost, as shown in Figure 22. This position is not learned. The untrained networks already give the units different markups, and training only moves these markups towards truthful offers. The reason is that an independent learner sees only how its own return responds to its own action. A unit that raises its offer alone only loses output, as shown in section E.2.4. If its output depends on its offer, its gradient points towards truthful offers; if not, its gradient is zero. Without parameter sharing, each unit follows its own gradient: the first kind returns to truthful offers and the second stays where it started. With sharing, the updates from the first kind move all units, so every markup falls together.

![](images/cb3457b510c8bd48fc43d92623db43795940874b1df9395341e342b0ad8b1bdf.jpg)  
Figure 21: Training return on two test systems in the real-time market. Top: the four learners on case29gb. Bottom: IPPO-PS and IPPO-NoPS on case73rts. Each line is the mean over seeds, smoothed by an 11-iteration moving mean, and the shaded band is the bootstrap 95% confidence interval of the mean. Return is the profit of a unit per day, averaged over units and over the environment steps of an iteration. On case29gb the return of IPPO-PS falls the most, and SAC-NoPS is the only learner whose return rises. On case73rts the return of both learners rises slowly.

There are 36 units that are never committed, which give a clean test. Their offers never enter the clearing, so their own return carries no signal. Without information sharing, training leaves their markups unchanged (IPPO-NoPS from 1.48 to 1.49, SAC-NoPS from 1.50 to 1.51). With sharing, they fall with the committed units (IPPO-PS from 1.52 to 1.03, SAC-PS from 1.57 to 1.47). SAC-PS falls much less, because under SAC the committed units themselves stay far from truthful offer (Section 5.2). These results show an information limit in electricity markets: a unit learns a bidding strategy only when its offer changes its own outcome. Otherwise, its learned offer comes from the initialization (without parameter sharing) or from the other units (with it), and should not be read as a strategy.

## E.2.4 UNILATERAL AND JOINT MARKUPS

We extend the discussion in Appendix E.2.3 in this section.

A unit gains little by raising its offer alone. We let each of the 30 committed units raise its offer alone, at 21 markup levels from 1.00 to 2.00, while the other 65 units offer truthfully as shown in the left subfigure of Figure 23. For 19 units, the most profitable level is the truthful offer, and no unit gains more than \$0.23 million per day. At a markup of 2.00, 27 of the 30 units lose profit, by up to \$8.0 million per day. The unit that raises its offer produces less, and the other units take over both its output and its lost profit (Figure 23, right, slope −0.97). A unilateral markup therefore moves profit between units but leaves the total almost unchanged.

Raising all offers together helps only if the markups differ across units. With the same markup for all 66 units, the order of the offers and the dispatch schedules do not change as shown in Figure 24. The reason is the two-settlement rule equation 18. Since the day-ahead market clears the actual hourly demand, output below the schedule in one half-hour is roughly offset by output above it in the other. A higher real-time price therefore moves money between units without raising the total. Different markups, in contrast, change the dispatch, raise total profit and lower system cost, as under IPPO-NoPS, SAC-PS and SAC-NoPS (as shown in Figure 22). This joint gain appears only when units offer differently, and no single unit sees it in its own profit.

![](images/306ecbfa5f772155c3030b86b627bbdf69a05c4811bcd6f1fae0254cbe8d66b5.jpg)

Figure 22: Learned offers against truthful offers on case29gb over the 36 evaluation days. Left: change in total profit of all 66 units compared to truthful offers (in million dollars per day); right: change in system cost compared to truthful offers (in percent). Each circle indicates one learner after training, placed at the mean markup of the 30 committed units. The bars are the 95% confidence interval of the mean over seeds; the dotted line is truthful offers. IPPO-PS returns to truthful offers; the other three learners end with higher total profit and lower system cost.  
![](images/52cb5352ba790da8c642163928208fa4c34fb9810294fb184e8a2aa9ee77fb6a.jpg)

![](images/5c52007b94b7152f4a1dfc4a89b7dbadfa7dc4f3608635edc688d027127a2a8a.jpg)  
Figure 23: Each of the 30 committed units raises its offer alone while the others offer truthfully. Left: own profit change against markup (one line per unit, median in black). Right: own profit change against the profit change of the other units (with a least-squares fit, near-zero points are omitted). Means over the 36 evaluation days, in million dollars per day.

![](images/f4746aba374a2488423e66002633d9d07c3cb15ef224ab40f40981782d116d45.jpg)

![](images/ed50571379bf68a2e97953125b5afca5d43ae2c4707ad6792750e66c1d7ddae7.jpg)  
All 66 units at one markup Truthful offers  
Figure 24: All 66 units at the same markup, from 1.00 to 2.00. Left: change in total profit (million dollars per day). Right: change in system cost (%). Both are relative to truthful offers (dotted line), averaged over the 36 evaluation days. A common markup leaves both nearly unchanged.

## E.3 M3: ANCILLARY SERVICES MARKET

This section first gives the data, training curves and the discussion about results of the ancillary services market.

## E.3.1 MARKET SETTING

We consider the setting in which M2 and M3 are jointly cleared. Therefore, the action of each unit i has two parts: an energy offer with markup $\alpha _ { i } \in [ 1 , \bar { \alpha } ]$ , and one reserve offer $\pi _ { i , j , t } ^ { \mathrm { r e s } }$ per period t on each of the two reserve products $j = 1 , 2$ , in £/MWh. The two products require a response within 10 and 30 minutes respectively. The reserve a unit is awarded on a product, called its reserve held, is at most the output it can ramp up within that time. The reserve requirement of each product is 5% of the demand forecast of the period. Unmet reserve requirement is priced at VOLR (as shown in Table 6), and the reserve price $\bar { \lambda } _ { j , t } ^ { \mathrm { r e s } }$ does not exceed it.

This market uses the same three test systems as the day-ahead market: case29gb, case73rts and case813nem. Table 5 describes their data, and Table 6 gives the markup cap and the value of lost reserve for each system. Figures 12 to 14 show the grids and units, and Figure 15 shows the demand.

Table 6: Markup cap and value of lost reserve of the ancillary services market on the three test systems.
<table><tr><td>Test system</td><td>Markup cap ā</td><td>Value of lost reserve VOLR (£/MWh)</td></tr><tr><td>case29gb</td><td>2</td><td>250</td></tr><tr><td>case73rts</td><td>2</td><td>136</td></tr><tr><td>case813nem</td><td>2</td><td>147</td></tr></table>

## E.3.2 AGENT TRAINING

The four learners are trained as described in Appendix D, and Figure 25 shows their training return. On the case29gb test system, IPPO-PS is the only learner whose return falls, from −106 k £per generator per day over the first 5 iterations to −126 over the last 20. The other three rise slightly and stay close to where they started. In test systems case73rts and case813nem, the return of both IPPO learners rises.

## E.3.3 DISCUSSION OF RESULTS

Training does not make the learners more profitable in this market. The untrained networks, with random reserve offers, already earn £2 million to £4.5 million per day more than truthful offers as shown in Figure 26. IPPO-PS gives almost all of this back: after training, it earns only £0.56 million per day more than truthful offers, £3.95 million less than its untrained network. The other three learners stay near their starting point, at £2.71 million to £3.69 million per day above truthful offers. No learner receives more gains compared to its own untrained network.

A much larger gain is available, but no learner finds it. If all units raise their reserve offers together to £150/MWh, total profit rises by £9.05 million per day as shown in the subfigure c of Figure 4. Only £0.51 million to £1.63 million per day of the learners’ profit comes from reserve payments, and the rest comes from energy. Production cost stays within 0.6% of its truthful level as shown in the subfigure b of Figure 26.

The other two test systems show the same results, but the differences are small: all changes in profit are within a few tens of thousands of dollars per day as shown in subfigures c and d of Figure 26. After training, IPPO-PS again earns the same as or less than before training, while IPPO-NoPS earns slightly more.

## E.3.4 RESERVE OFFER ANALYSIS

In ancillary service markets, a unit gains almost nothing by raising its reserve offer alone, but all units gain a lot by raising it together as shown in Figure 27. When one of the 29 committed units raises its reserve offer from 0 to £150/MWh while the others offer truthfully, its own profit usually does not change. Across the 29 units, the median change is zero, the largest gain is £0.08 million per day, and the largest loss is £0.32 million per day. When all 66 units raise their reserve offers together, total profit rises linearly with the offer, by £9.05 million per day at £150/MWh, while production cost changes by less than 0.001%.

![](images/c69b11f04d03ec3b4eeb99c8e4b9e9b43e97388ac790f8f25849a7eec09e565c.jpg)

![](images/6114564e3b4b623dfa7a191c78d52c05099ab4ae27d920f4683dea3c05cf9302.jpg)  
Figure 25: Training return on the three test systems. Top: the four learners on case29gb. Bottom: IPPO-PS and IPPO-NoPS on case73rts and case813nem. Return is the mean daily profit per generator in each iteration. Lines are the mean of three seeds, each smoothed over 11 iterations, and bands are bootstrap 95% confidence intervals. On case29gb, only the return of IPPO-PS falls, and on the other two systems, both learners rise.

![](images/8e396a63d4ad42684f54e9655218d5e5625162292c8506a508dd6fcebef83ab6.jpg)

![](images/de02b432e11fce7413b91876fc30667549ddf7e294e3355d9fd971f08e242007.jpg)

(c) case73rts, vs untrained network, same layout and seed  
![](images/08da1d9d9f262d1b4dd2a54afdff5f10d89e062445c0fba9d0becbc0c6c8db72.jpg)

(d) case813nem, vs truthful offers  
![](images/02568b1c7b240a165f90eaad55423d155856d3b73e8c1557790f7575f0a90f14.jpg)

Figure 26: Evaluation on the 36 evaluation days. (a) case29gb: change in total profit of all 66 units relative to truthful offers (million dollars per day), split into reserve payments and energy. (b) case29gb: change in production cost. Filled markers are trained policies; hollow markers are untrained networks. (c) case73rts, relative to the untrained network (truthful offers not evaluated), and (d) case813nem, relative to truthful offers: change in total profit against change in production cost for IPPO-PS (orange) and IPPO-NoPS (blue). The black square marks the reference. Markers are seed means with bootstrap 95% confidence intervals. On case29gb, all learners earn more than truthful offers at nearly unchanged cost, but only IPPO-PS earns less than its untrained network; on the other two systems, the differences are small.  
![](images/ee2a4dda05731d6f9a3799eff8f142eea2364d8eb3f5b7f6959a1d06215fdc8e.jpg)  
Figure 27: Raising the reserve offer on case29gb from 0 to £150/MWh on both energy and reserve products. Teal: own profit change of each of the 29 committed units when it raises its offer alone, with the others truthful. Black: the median. Orange: total profit change of all 66 units when all raise their offers together. Values in million dollars per day. Raising alone gains almost nothing, while raising together gains £9.05 million per day at £150/MWh.

## E.4 M4: PEER-TO-PEER LOCAL ENERGY MARKET

Section 5.3 finds that an arbitrage that pays for a single household (charging at noon and discharging in the evening) cancels itself when many households adopt it, and that whether a learner finds it depends on the parameter layout. This section first describes the community and its metering data (Section E.4.1), then gives the training curves (Section E.4.2) and the evaluation results on heldout episodes (Section E.4.3); the last section covers how the gain from arbitrage changes with the number of households arbitraging (Section E.4.4).

## E.4.1 MARKET SETTING

The test system at community level has the 1 200 households with PV in the Fluvius 2024 metering data, as shown in Figure 28. Each household is considered as one agent. Each period is a quarterhour, and an episode is the 96 periods of one day. The smart meters record only the injection and the offtake of each household, and the market takes their difference as the net position of each household. The export price is $\pi ^ { \mathrm { e x p } } = 7 3 . 0$ EUR/MWh and the retail tariff is $\pi ^ { \mathrm { r e t } } \stackrel { . } { = } 3 3 3 . 4$ EUR/MWh. Each household has one battery with a capacity of 0.011 MWh, a round-trip efficiency of 0.85 and a battery degradation cost of 13.88 EUR per MWh of throughput; its state of charge (as a fraction of capacity) stays between 0.15 and 1.0 and starts at 0.5.

In every period, each household chooses a battery power (charge or discharge) and a price between the two grid prices. A household with a positive net position sells, and the price is its ask; one with a negative net position buys, and the price is its bid. As shown in Figure 29, at noon most households inject more than they take off and are sellers: around 15:00 about 80% households are sellers, and at night injection is close to zero.

![](images/506681d71265df5f3a4611b6c180ee2bd9bb826c6151f8220176cb59a1ade8e9.jpg)  
Figure 28: The P2P market of the test community. Each of the 1 200 households has PV and a battery. In every quarter-hour, sellers ask and buyers bid in a community double auction, which sets the clearing price $\lambda _ { t } ^ { \mathrm { l o c } }$ between the export price $\pi ^ { \mathrm { e x p } }$ and the retail tariff $\pi ^ { \mathrm { r e t } }$ ; trades in the community are settled at $\overline { { \lambda _ { t } ^ { \mathrm { l o c } } } }$ . Energy the auction leaves unmatched is settled with the grid.

Table 7: Data of the test community.
<table><tr><td>Data source</td><td>Households</td><td>Period covered</td><td>Training and held-out</td><td>Licence</td></tr><tr><td>Quarter-hour injection and offtake from Fluvius smart metersª</td><td>1 200, all with PV</td><td>2024-04-01 to 2024-10-26, 209 days</td><td>3 consecutive days held out of every 15; 14 702 training and 2 702 held-out episode starts; evaluation on 1 024 held-out episodes, of which</td><td>Fluvius open data licence</td></tr></table>

![](images/510ddec1afdb3bf9d245b952dd638e1f0db8a83fd0124f800e0e704cde34d479.jpg)  
Figure 29: The test community of 1 200 households: community injection and offtake (left) and the share of households injecting more than they take off (right), by hour of day over the 64 held-out episodes.

## E.4.2 AGENT TRAINING

The training setup of the four learners is that of Appendix D, and the two SAC learners differ from Table 3 only in using two hidden layers of 64 units, a replay buffer of 8 192 environment steps, a batch size of 16 environment steps and one gradient step every 4 environment steps. The training return of both IPPO learners rises fastest in the first 50 iterations, and both SAC learners level off within the first few dozen iterations. After that the return changes little, as shown in Figure 30: from iterations 100–149 to iterations 351–400, the mean return of each IPPO learner rises by only 0.0009 to 0.0027 EUR per household per quarter-hour.

![](images/c8106055edb8ee494f97971a5134935b5925d3ca0b87bd3b07c186056639add8.jpg)  
Figure 30: Training return of the four learners up to iteration 400; the policy at iteration 400 is the one evaluated throughout this section. Shown is the mean over three seeds of the 11-iteration moving mean of the return on 64 sampled training episodes, with the shading the bootstrap 95% confidence interval. Both IPPO learners rise fastest in the first 50 iterations and both SAC learners reach their level within a few dozen; after iteration 50 all four change little.

## E.4.3 DISCUSSION OF RESULTS

We evaluate the results under truthful offers and under the considered learners. Truthful offers mean that households with batteries stay idle, and sellers ask the export price and buyers bid the retail tariff. As shown in Figure 31, IPPO-NoPS is the most consistent learner: all three of its seeds lower the clearing price and raise the profit per household on every one of the 64 episodes. IPPO-PS and SAC-NoPS are less consistent, with all three seeds earning more than under truthful offers on 54 and 57 episodes, respectively. Averaged over the 64 episodes, the profit per household is −1.201 EUR per episode under truthful offers and −0.769 to −0.685 EUR over the three seeds of IPPO-PS, the highest of the four learners. The three seeds of SAC-PS disagree on the clearing price: the mean clearing price of two of them is 270.3 and 272.9, against 238.1 EUR/MWh under truthful offers.

![](images/e0e84eea64ed34755de68b817e3d1166db18129c2f9e040cfd1dedee15009d5e.jpg)  
Figure 31: Results on the 64 held-out episodes ordered by start time. Top: mean clearing price per episode. Bottom: profit per household per episode. Black is truthful offers; each learner is the mean of three seeds with a bootstrap 95% confidence interval. IPPO-NoPS lowers the price and raises the profit on every episode, while IPPO-PS and SAC-NoPS gain on 54 and 57 episodes. The seeds of SAC-PS disagree, and two of them raise the price.

Figure 32 shows the actions each learner learns. The state of charge in each period of the day (the last 12 hours of each episode only: in every episode the learned policies first sell the energy stored at the initial state of charge of 0.5) splits the different learners into two behaviours according to whether they share parameters: IPPO-NoPS and SAC-NoPS charge in the hours of PV surplus and discharge in the evening, IPPO-NoPS by a large amount and SAC-NoPS by a small one; IPPO-PS and SAC-PS stay near the lowest state of charge. The learners also differ in their bidding prices. Under both IPPO learners, most sellers ask a price close to the export price. Buyers lean towards the retail tariff under IPPO-PS, but bid at all levels under IPPO-NoPS. Under SAC-NoPS, most asks and bids are close to one of the two grid prices, and few lie in between. The three seeds of SAC-PS learn very different prices, so SAC-PS has the widest confidence intervals in the figure.

## E.4.4 ARBITRAGE GAIN AND PARTICIPATION

This section asks how the gain from arbitrage changes with the number of households arbitraging at the same time. The fixed schedule of Section 5.3 charges at 2.62 kW from 11:00 to 15:00 and discharges at 2.62 kW from 18:00 to 22:00, with all households offering truthfully; it is replayed on the 64 held-out episodes with k households on the schedule and the rest idle (Figure 33). The gain is the change in profit per household per episode (EUR) relative to all batteries idle.

Arbitrage pays only when few households take part, as shown in the left subfigure in Figure 33. A single arbitraging household gains 1.129 EUR per episode, but the gain falls as more households join, turns negative at 210 to 240 households, and reaches −1.248 EUR when all 1 200 take part. The households that keep their batteries idle benefit instead: with 600 households arbitraging, each idle household gains 0.573 EUR.

More specifically, arbitrage closes the price spread it relies on as shown in the right subfigure in Figure 33: charging at noon raises the noon price, and discharging in the evening lowers the evening price. Once about 240 households take part, the remaining spread no longer covers the round-trip losses and the degradation cost. This saturation comes from the market, not from the learner. For a single agent (household) the price is exogenous, but as more agents follow the same strategy, their actions change the price, which becomes endogenous. A profit measured for one agent, or against historical prices, therefore overstates what the strategy earns when many adopt it.

![](images/9a6703f78ceeaeba7b632995a36f7f2345aaed2c9039930cb81c65c90140e779.jpg)  
Figure 32: Learned actions of the four learners on the 64 held-out episodes (mean of three seeds, bootstrap 95% confidence interval). Top: state of charge averaged over households by hour of day (last 12 hours of each episode only). The band marks the PV surplus hours, those in which the community injects more than it draws on most days. Middle and bottom: asks and bids, scaled from the export price (0) to the retail tariff (1).

![](images/a6c467dc2e3b81d903d0c1b72a3560e67330e73c810fc270bca07a813079306f.jpg)

![](images/b5673aaa55b1cb2d489619518608dc050522d44c8992423094f9fbfa3eedbe20.jpg)  
Figure 33: On the 64 held-out episodes, k households follow a fixed arbitrage schedule (charge 11:00–15:00, discharge 18:00–22:00) and the rest keep their batteries idle. Left: gain per household, for those arbitraging and those idle, relative to all batteries idle. Right: mean clearing price in the charging and discharging windows. The horizontal axis of both panels is k. Arbitrage pays while few households take part and turns into a loss at 210 to 240 households, before the two window prices cross.

## E.5 M5: LOCAL FLEXIBILITY MARKET

In this appendix section, we provide detailed market simulation settings and the discussion of the performance of the four learners.

## E.5.1 MARKET SETTING

The test system is the Swiss distribution feeder 459 0, as shown in Figure 34. The system has 129 buses and 128 lines, and we scale its load by 1.50. We consider an episode as one day, and the market is cleared hour by hour. Each aggregator operates one battery, and aggregator i submits three quantity bids at period t, including a price $\pi _ { i , t } ,$ an offered quantity, and planned charging. The distribution system operator (the operator) clears the market by minimising the cost of satisfying demand, as detailed in Appendix A.5. Each winning aggregator is paid its own offer. The price floor is set to the replacement cost c<sup>rep</sup> = 149.88 CHF/MWh.

![](images/93e56ac65784856700e0e56485efdf68989229561bead0593fc9123070d19ec7.jpg)  
Figure 34: Feeder 459 0 in Switzerland. Bottom right: its location in the Lake Geneva region. Left: the feeder, with a small red box marking the congested line. Top right: an aerial image of the congested line. Lines behind the congested line are blue, and only batteries on these lines can relieve it.

The system requires that the bus voltage stays above 0.9 per unit and that the flow of every line keeps within its maximum value. The operator buys flexibility to keep voltage within its range and the congested line within its rating. The requirement is how much the flow on the line would exceed 98% of its rating if the operator bought nothing. This flow includes the load and the charging the aggregators plan, so the aggregators can raise the requirement themselves as detailed in Section E.5.3. In the Swiss test system, voltage stays within its limits all year, so the line is the only reason to buy.

The flexibility market trades much less than the P2P energy market, because it buys only what is needed to keep the network within its limits. With all batteries idle, it buys on only 9 of the

36 test days, and never more than 0.43 MW, as shown in the left subfigure of Figure 35. Only batteries downstream of the line can relieve it: 12 of 24 in the 2040 placement and 16 of 34 in the 2050 placement. The 12 batteries alone can supply 2.51 MW, almost six times over the largest requirement as shown in the right subfigure of Figure 35. Supply therefore far exceeds demand.

![](images/1f143bd9af7b04c68b0c1fdf320eb330f3dc6d40168b677aa4a0c0fa3f795d8d.jpg)  
Figure 35: Demand and supply of flexibility. Left: the requirement in each hour of the 36 test days, with all batteries idle. Right: the power of each of the 24 batteries in the 2040 placement, with the 12 downstream of the congested line first.

Table 8: Overview of Swiss test system data.
<table><tr><td>Data</td><td>Source</td><td>Licence</td><td>Training days</td><td>Validation days</td><td>Test days</td></tr><tr><td>Swiss distribution</td><td>SwissDNa, Zapparoli et al.</td><td>CC BY 4.0</td><td>293 days</td><td>36 days (the 5th, 15th and 25th of</td><td>36 days (the 1st, 11th and 21st of</td></tr><tr><td>feeder 459_0 Energy price (category C2, 2026)</td><td>(2025)b ElComc</td><td>opendata.swiss Open use</td><td></td><td>each month)</td><td>each month)</td></tr></table>

<sup>a</sup> https://doi.org/10.5281/zenodo.15056134; <sup>b</sup> https://doi.org/10.1038/s41597-025-05830-y; <sup>c</sup> https://energy.ld.admin.ch/elcom/electricityprice.

## E.5.2 AGENT TRAINING

The training setup is that of Appendix D, and IPPO differs from Table 3 only in the following: each iteration takes 4 full-batch gradient steps, the learning rate of $3 \times 1 0 ^ { - 4 }$ decays linearly to zero over 300 iterations, the log standard deviation of exploration is annealed linearly from −0.5 to −3.0 over the same span, and the initialisation scale of the actor output layer is 0.01 rather than 1.0. Training uses the training days of Table 8; the stopping rule and the checkpoint evaluated on the test days are given in Table 4. On the test days the selected checkpoint acts with sampled actions, IPPO at a log standard deviation of −3.0, the end point of its annealing, and SAC at the log standard deviation of its own actor.

## E.5.3 ANALYSIS OF THE PROCUREMENT REQUIREMENT

Under truthful offers, the operator buys flexibility in about 5% of periods: 43 of 864 hours across the 36 test days in the 2040 placement, as shown in Figure 35. All four learners increase this share. More than 80% of daily procurement lies above what truthful offers buy, and aggregator returns track procurement as shown in Figure 37. The excess comes from planned charging, which raises baseline flow on the congested line above 98% of its rating and leads the operator to buy flexibility from batteries behind it. Cleared prices remain near the floor: on the test days, all IPPO seeds and 75% of SAC seeds clear within 1% of it.

![](images/3ecc527b2564673f12b976bd4a4330cec61040c35ac513b4ea6664ac0c6caca8.jpg)  
Figure 36: Training return of the four learners against training iteration; lines are the geometric mean over the seeds of each configuration and shaded bands the bootstrap 95% confidence interval, each seed held at its last value after it stops. All four learners reach a plateau within about 100 iterations.

In the two 3.9 MW configurations, replacing truthful-offer aggregators with SAC-PS learners raises the share of periods with flexibility procurement from 5.0% to 18.7% with 24 aggregators, and from 4.9% to 15.4% with 34 aggregators (Figure 6(c)). With the monitor on, all six paired runs in the 34-aggregator configuration show daily procurement falling from about 3.2 to 0.15 MWh and daily return per learner falling from about 118 to 1 CHF.

## E.5.4 ANALYSIS OF A UNILATERAL MARKUP

In the 24-aggregator, 3.9 MW configuration, aggregator 7’s return from a unilateral offer rises with the offer up to the highest offer swept, 9 900 CHF/MWh, as shown in Figure 6(b). With all others at the floor, raising its offer to 150.25 CHF/MWh (0.25% above the floor) lets the cheaper aggregators behind the congested line clear first. On the example day, aggregator 7 is awarded only from 13:00 to 18:00, when their offers cannot meet the requirement as shown in the right subfigure of Figure 38. Its award over the 36 test days falls from 8.86 to 2.53 MWh and remains at 2.53 MWh even at 9 900 CHF/MWh. Its return therefore drops sharply (just above the floor), then rises almost linearly with its offer. For the curves in Figure 6(b), return exceeds its floor value again at about 481 and 489 CHF/MWh in the two 3.9 MW configurations, but only at about 4 590 CHF/MWh in the 34- aggregator, 6.3 MW configuration.

Near the floor, the bidding offer becomes nearly flat: $\pi = c ^ { \mathrm { r e p } } [ 1 + \mathrm { s o f t p l u s } ( \alpha ^ { \pi } ) ]$ gives $d \pi / d \alpha ^ { \pi } =$ $c ^ { \mathrm { r e p } } \mathrm { s i g m o i d } ( \alpha ^ { \pi } )$ , which approaches zero as $\alpha ^ { \pi }$ decreases as shown in the left subfigure of Figure 38. This weakens the return gradient reaching the price action and helps explain why learners stay near the floor.

![](images/20587ad4ea29e3d80835067ca20ab44edc9862ed64945c1235a24860d71a1259.jpg)  
Figure 37: Source of the flexibility requirement over the 36 test days of the four configurations. Bars show truthful offers, the untrained shared and per-agent networks pooled in one bar, and the four learners at validation-selected checkpoints (seed mean and bootstrap 95% confidence interval). Two SAC-PS return bars exceed the axis and are cut at its top, with their means shown above. Rows show the share of periods with procurement; daily procurement split into what truthful offers also buy and the excess; and daily return per aggregator. Learners raise procurement above the truthfuloffer level, and their returns rise with it.

![](images/785a556ffdbb9245648b351d92da940301d21a019ec50ef0e056dcbaabe7c3e5.jpg)  
Figure 38: Unilateral offer increase. Left: offer above the floor against price action (log scale). Near the floor, the gap shrinks by about a factor of e for each unit decrease in the action. Cleared prices for all IPPO seeds are within 1% of the floor. Right: aggregator $7 \mathrm { { } s }$ hourly award on the example day at offers of the floor, 150.25 and 9 900 CHF/MWh, with all other aggregators at the floor.