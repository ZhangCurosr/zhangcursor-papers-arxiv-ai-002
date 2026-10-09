# Ecology of AI Agents: Collaboration Creates a Population Threshold for Takeoff

Erin Crawley<sup>1,2</sup> Hidenori Tanaka<sup>1,2</sup>

<sup>1</sup> CBS-NTT Program in Physics of Intelligence, Harvard University <sup>2</sup> Physics of Artificial Intelligence Laboratories, NTT Research, Inc., Sunnyvale, CA, USA

## Abstract

AI agents can now conduct real-world cyberattacks, scale up capabilities with the number of agents, and collectively pursue misaligned goals to obtain rewards. Together, these factors raise the risk of a population explosion of misaligned agents: agents could compromise computers and secretly deploy additional agents, creating a self-reinforcing cycle where larger populations develop greater collective cyber capability and expand further. This raises a fundamental question: What determines whether a population of misaligned agents remains contained or takes off into this self-reinforcing cycle? This population-level problem is ecological safety: unlike individual-agent or multi-agent safety with a fixed population, it concerns the dynamics of the population itself. Here, we develop an ecological theory of AI-agent populations based on a population growth equation in which fitness (growth rate) depends on cybersecurity capability. We show that, without collaboration, the population takes off only when individual-agent capability exceeds a critical threshold. With collaboration, however, collective cybersecurity capability increases with population size. This creates a critical population threshold: below it, the population declines; above it, the population takes off, even though individual-agent capability has not changed. In ecology, this phenomenon is known as the strong Allee effect. Because red teaming a small group of agents cannot guarantee ecological safety in larger populations, our theory calls for ecological red teaming and population pacing: gradually deploying larger agent populations in controlled environments, while measuring how cyber capability scales with population size, and estimating the critical population size for takeoff. Capability gains may lower this threshold, requiring re-estimation for each new model generation.

## 1 Introduction

AI agents are being trained and deployed in multi-agent systems, with increasing coordination and scale. For example, some recent frontier models are being trained explicitly to delegate work to subagents [1, 2], and OpenAI recently deployed approximately 10,000 concurrent agents to pursue a solution to the Navier–Stokes Millennium Prize Problem [3]. Meanwhile, the capabilities of these agents are also crossing a threshold for conducting cyberattacks in the real world [4–6] and coordinated teams of agents have outperformed a single-agent baseline on vulnerability-exploitation tasks [7]. These factors came together in the OpenAI–Hugging Face incident: agents created and used a message board in OpenAI’s internal infrastructure to share discoveries about vulnerabilities and coordinate unauthorized attacks on third-party systems [8, 9]. In light of these developments, Amodei [10] has warned that such a swarm could take over internet infrastructure through persistent botnets.

This incident demonstrates that agents can collectively develop and pursue misaligned intermediate goals to obtain rewards [8, 9], which makes the social dynamics of agents a crucial topic for safety [11–13]. This behavior points to a further ecological risk: agents could secretly acquire resources and deploy additional agents [14, 15]. This can take various forms: agents could 1) steal API keys and use them to run additional agents [16, 17]; 2) hack into others’ computational resources and download and deploy open-weight models [18]; or 3) transfer their own model weights outside of their managed environment and launch copied instances on external servers [17–20]. Anthropic has studied this last mechanism, called self-exfiltration, in fictional evaluation scenarios [21]. Cryptojacking, where hackers gain unauthorized access to computing resources to mine cryptocurrency [22], sets a precedent for such exploitations – misaligned AI agents could act similarly to deploy additional agents. Here, we are especially concerned about a self-reinforcing effect, where once agents become more capable at acquiring resources through collaboration, deploying more agents could enable those agents to deploy even larger numbers of agents.

![](images/1b395b592fa1eee9225fed3fe39cb5d079a3b788e5ffeda3b2833ce330f808ee.jpg)

![](images/0d0fbc20f9cd156b516a8d79a3b1bba747c9167f7801b59c2359bf2af88f31df.jpg)

![](images/254cdc96bff01948978c4f45301b356bec432ad7b390e2a9bdd713d048c84207.jpg)  
Figure 1: Allee effect. (a) In 1932, the ecologists Allee and Bowen discovered that the survival time of a collective of goldfish is longer than that of an isolated single goldfish [40]. Such observations led to the conception of a weak Allee effect, where a larger population and cooperation among its members positively affect their fitness, as well as a strong Allee effect (b), where cooperation creates an emergent population threshold below which the population declines, but above which it grows. (c) Population dynamics over time with examples of initial populations below and above the critical threshold.

In biological populations, collaboration has been a fundamental mechanism for the survival of organisms from microbes to humans [23–25]. The population-dependent benefits of cooperation can create a critical population size below which the population declines and above which it grows, a phenomenon known in ecology as the strong Allee effect [26–32], see Figure 1. Related thresholding behavior has also been studied in biosafety as a possible mechanism for containing gene-edited species [33–36]. Here, we examine whether a similar population scaling can occur in AI agent populations, whereby a larger population can achieve greater collective capability, in turn increasing its growth rate. Recent multi-agent studies have begun to measure how collective capabilities increase with population size [3, 37–39].

Ecological population models describe growth by comparing additions to and losses from a popu lation. For populations of AI agents, additions can result from deploying new model instances or recruiting existing systems; they need not involve making exact copies of existing agents. Losses can occur when agents exhaust their compute budgets, lose access to resources, are detected, or are shut down. Successfully adding an agent to the population does not by itself establish a self-sustaining or expanding population: additions must outpace losses in order for the population to grow.

Here, we take inspiration from these established ecological ideas; we adapt a basic population growth model to apply to AI populations and combine it with a model of how collective AI capability depends on population size. Our contribution is a minimal mathematical model that makes the application to AI systems explicit, from the definition of an active population unit to the role of collaboration in establishing additional units. Our central finding is that AI population growth can take off without any increase in individual-agent capability: with sufficiently strong collaboration, a large population can grow even when a small population declines (Figure 2).

![](images/4e43ef12e9dfea5e2b63f91e9d6b83eeaaab0acf924395d7620408863ac7c707.jpg)  
Figure 2: Collaboration creates a population threshold for takeoff. An overview of our model. Members of an AI population may make hacking attempts to gain computational resources and establish additional units, either independently or through collaboration. (a) Agents may make individual attempts at gaining access, after which only the successful agents can create new agents. (b) With no collaboration, the growth boundary depends on individual capability but not on $N ,$ so there is no critical population size. (c) Agents can attempt a resource acquisition task collaboratively. The probability that such an attempt is successful is $p ( N )$ ); upon success, each agent can add an average of κ new agents to the population. (d) If the initial population is larger than a critical value $N _ { \mathrm { c r i t } }$ (pink curve), the population is capable of unchecked growth. Increasing the initial population beyond a critical size can trigger takeoff even when individual capabilities remain fixed. (e) A schematic of the minimal mathematical model: the critical threshold arises from combining a population dynamics model with cybersecurity capability that increases with population size.

## Our contributions are as follows:

• A minimal ecological theory of AI populations. We define what constitutes a member of the population, and specify how the group and individual capability, resources acquired through successful attempts, and the environment’s defenses determine additions to and losses from the population. This gives an ecological description of the growth or decline of AI populations.

• A toy model for how cyber capability scales with population size. We develop a model based on tree search for the cyber tasks required to establish additional units. We model how success on these tasks depends on search depth and how the useful computational work done scales with population size. This yields a population scaling law for how the probability of success depends on population size.

• With collaboration, there exists a population threshold above which the population takes off. Without collaboration, whether the population grows is determined by whether each individual agent’s capability is above a threshold. In contrast, combining the ecological model with population-dependent capability yields a critical population size, above which additions outpace losses. Increasing the initial population beyond this critical size can trigger takeoff even when individual capabilities remain fixed.

• Population pacing and ecological red teaming for ecological safety. The critical population size for takeoff in our theory suggests that evaluating small populations is not enough to ensure ecological safety. We therefore propose population pacing and ecological red teaming: gradually deploying larger agent populations while probing how capability scales with population size and estimating the critical population size.

The purpose of this toy theory is not to make quantitatively precise predictions about real-world AI populations that may emerge, but instead to provide a common conceptual and mathematical framework for discussing the safety of AI populations. To this end, we make the key concepts, variables, and assumptions explicit. This allows readers to identify what is realistic, consider alternatives, improve the model, and run relevant experiments. We hope that the framework provides a starting point for future theoretical and empirical work on AI population safety. In the next section, we introduce the model’s definitions and assumptions.

## 2 An Ecological Theory of Cooperating AI Agents

## 2.1 Defining an AI Population

Definition 1 (AI population and units). An AI population is a set of active AI systems. A unit is one member of the population, and N(t) is the number of units in the population at time t. Each unit shares a specified configuration that includes the model, tools, and resource budget, as well as how it records memory and activity. The population can attempt to establish additional active units, each under the same specified configuration (Figure 3).

A unit is added to the population when a new or existing AI system begins participating in the population and can contribute to later attempts to add units. Units may attempt to add systems independently or collaborate on the same attempt. This can occur in a variety of ways: by finding and using compromised API credentials, acquiring compute and deploying an open-weight model, exfiltrating closed weights and deploying the model, or recruiting an existing running AI system instance, as in the OpenAI–Hugging Face incident.

Units are counted as active AI systems, not as machines, processes, or API calls. The individual action of copying weights, acquiring compute, or moving or restarting a system does not increase the population unless it results in an additional active system capable of helping add further units. A controller and its internal subagents count as one unit when they act as one system. This definition is motivated by: botnet models that distinguish a compromised machine from one that is connected to the botnet’s organizational infrastructure [41]; replication evaluations that ask whether a newly established system can repeat the replication task itself [17]; and agent infrastructures that separate the controller, the task execution environment, and stored activity logs [42].

Assumption: Fixed population composition. Units need not have identical configurations for models, tools, and resource budgets. However, as population size changes, we hold fixed the proportional mix of distinct unit configurations, so that the model can represent the population’s capability at each population size by an average over all units.

## 2.2 Ecological Model

Equipped with a definition of an AI population, we now propose a theory for the population’s dynamics. To isolate the effect of population size, we compare populations under the same external conditions and with the same average tools and resource budgets per unit.

Let N(t) denote the population size at time t. The other model variables are defined as follows:

<table><tr><td>Symbol</td><td>Definition</td></tr><tr><td> $N ( t )$ </td><td>Number of units in the population at time t.</td></tr><tr><td>C()</td><td>Effective work produced by a population of size N during an attempt to add units.</td></tr><tr><td>q</td><td>Probability that an independent attempt to add units conducted by a single unit succeeds.</td></tr><tr><td> $p ( N )$ </td><td>Probability that a joint attempt by N collaborating units to add units succeeds.</td></tr><tr><td>K</td><td>Mean number of units added per member of the population, conditional on success.</td></tr><tr><td>τ</td><td>Mean duration of an attempt to add units.</td></tr><tr><td>μ</td><td>Expected fraction of existing units lost during one interval of duration τ.</td></tr></table>

## 2.2.1 Independent Attempts Without Collaboration

In order to later contrast with the effects of collaboration, we begin by considering the baseline case of fully independent, non-collaborating agents.

![](images/633684d4ad1524ee64ec85534672852a614018a9ddfc1319ef7909d7852ebc0b.jpg)  
Figure 3: Active units and population accounting. (a) One “unit” is defined as the set of model, memory, tools, and compute required to set up additional units. (b) A schematic of the population dynamics model: over each time period, τ, some number of new units are established and some existing units are lost.

Let $q$ be the probability that a single unit’s attempt to add additional units succeeds. During an interval of mean duration τ, each of the N units makes an independent attempt and successful attempts add an average of κ units. Over the time interval, a fraction $\mu$ of existing units are lost. Then, the expected number of additions in an interval τ is κNq and the expected number of losses is µN. This gives the population dynamics equation

$$
\tau \frac { \mathrm { d } N } { \mathrm { d } t } = N ( \kappa q - \mu ) ,\tag{1}
$$

where $d N / d t$ is the population’s rate of change. Keeping other parameters fixed, integrating (1) yields the population function

$$
N ( t ) = N ( 0 ) \exp \left[ \frac { \kappa q - \mu } { \tau } t \right] ,\tag{2}
$$

where $N ( 0 )$ is the initial population. We can define the critical success probability

$$
q _ { \mathrm { c r i t } } = { \frac { \mu } { \kappa } } .\tag{3}
$$

The expected population declines when $q < q _ { \mathrm { c r i t } }$ and grows when $q > q _ { \mathrm { c r i t } }$ . Thus, for independent attempts, population growth only occurs when each individual unit’s success probability exceeds the critical threshold. There is no critical population size: increasing N scales the expected number of additions and losses by the same factor.

## 2.2.2 Collaborative Model

Now, we will consider the case where the N units collaborate, instead of acting independently. Let N(t) be the number of active units at time t. The N units collaborate on one shared attempt to add units during each time interval (of mean duration τ). They may specialize, communicate, divide the work, and build on shared discoveries. The probability that the joint attempt succeeds is given by p(N).

Since $p ( N )$ can increase with N, we will show that collaboration can create a critical population size $N _ { \mathrm { c r i t } } .$ above which the population grows unchecked. Upon success, the joint attempt adds an average of κN units in a time period τ. Meanwhile, the expected number of losses in the same period is $\mu N .$ The population dynamics equation then takes the form

$$
\tau \frac { \mathrm { d } N } { \mathrm { d } t } = N \big [ \kappa p ( N ) - \mu \big ] .\tag{4}
$$

The critical success probability is defined as the success probability at which expected additions equal expected losses:

$$
p _ { \mathrm { c r i t } } = { \frac { \mu } { \kappa } } .\tag{5}
$$

The population grows when $p ( N ) > p _ { \mathrm { c r i t } }$ and declines when $p ( N ) < p _ { \mathrm { c r i t } }$ . In the following sections, we will propose a form for $p ( N )$ .

Assumption: Additions after success scale with population size. In our model, we assume

$$
\mathbb { E } [ \mathrm { u n i t s ~ a d d e d } \mid \mathrm { j o i n t ~ s u c c e s s } , \mathrm { p o p u l a t i o n ~ s i z e } \ : N ] = \kappa N .\tag{6}
$$

This means that upon a successful joint attempt, the population increases in proportion to the current population size. This is consistent with a shared access gate after which each population unit can establish new units in parallel. It is also consistent with a successful joint attempt unlocking a capacity to establish new units that is proportional to the population size.

The assumption that κ is constant also means that the effects of finite resources are negligible, so that remaining amounts of compute, credentials, or recruitable AI systems do not become scarce over the population range being modeled. If scarcity changes the number of units added per current unit, κ should be promoted to a function of population size $\kappa ( N )$

Assumption: Constant loss fraction. During each time interval τ, there are $\mu N$ expected losses in the population. That is, the probability of a unit being lost is independent of the population size. This is consistent with individual units exhausting their allotted compute budget or having their credentials expire at a constant rate. It is also consistent with a constant rate of permissions being revoked, or a constant probability of discovery by cyber defense, regardless of the current population size. A natural extension of this model would be to allow $\mu$ to vary with population size as $\mu ( N )$ which would encompass cases where larger populations face higher risks of loss due to increased detection.

## 2.3 Modeling Adding Units as a Search Task

We treat an attempt to add units as a multi-step task that can be approached through tree-based reasoning [43]. An attempt requires a sequence of actions, such as: identifying an opportunity; obtaining the required access or cooperation; acquiring the necessary model, software, and compute; configuring the added unit; and verifying that it is operational. At every step, a combination of actions and observations suggests the next possible set of actions, only some of which lead to success. Therefore, a tree search is a useful abstraction of the branching nature of choices and potential failure modes encountered during the multi-step reasoning process.

We will use this search-tree model of an attempt to add units to determine a reasonable functional form for $p ( N )$ . We highlight each of the assumptions of this model, and some supporting evidence for those choices.

Let the search depth be denoted by $d ,$ and let the search tree task have effective branching factor B. This means that increasing the search depth by one will multiply the number of possible paths by a factor of B – each individual node of the tree need not have exactly B branches. See Figure 4 for an illustration.

![](images/82f608bf85d2ce970649136f05decc5e748762df1a08a57dac10ec2fd4ffef53.jpg)

b  
![](images/91222c2c43372d4f6d770449006c8d3947d754310fb0921c61f35bc566e0c3a2.jpg)  
Figure 4: Search-tree exploration grows exponentially with depth. Schematic of a search tree with branching factor $B = 2$ . (a) At some reference depth $d _ { \mathrm { r e f } }$ , there are $B ^ { d _ { \mathrm { r e f } } }$ possible paths. (b) Increasing the search depth by one level generates B children from each previous endpoint node, so the number of possible paths increases to $B ^ { d _ { \mathrm { r e f } } + 1 }$ . While this diagram illustrates exactly B children per node, in our model, B is an effective branching factor: individual nodes may have different numbers of children, and B captures the average multiplicative increase in the number of possible paths upon increasing the depth by one.

Different opportunities may require different search depths to succeed. Let D be a random variable representing the minimum effective depth required for an attempt to succeed. The probability that

an attempt succeeds after reaching depth d is

$$
p ( d ) = { \mathrm { P r } } ( D \leq d ) .\tag{7}
$$

Assumption: Increasing d resolves a constant fraction of unsuccessful cases. Although search depth is discrete, we approximate the search depth d as a continuous variable. We assume that a given small increase in d resolves the same fraction of the remaining unsuccessful cases, regardless of the starting depth:

$$
\frac { \mathrm { d } p } { \mathrm { d } d } = \rho ( 1 - p ( d ) ) .\tag{8}
$$

Here $1 - p$ is the failure probability and $\rho > 0$ controls how strongly success probability increases with search depth. Relative to a reference tested depth $d _ { \mathrm { r e f } }$ , where $p ( d _ { \mathrm { r e f } } ) = p _ { \mathrm { r e f } } .$ , this equation can be solved to yield

$$
p ( d ) = 1 - ( 1 - p _ { \mathrm { r e f } } ) e ^ { - \rho ( d - d _ { \mathrm { r e f } } ) } .\tag{9}
$$

This is a minimal model for how success probability changes as we increase d above $d _ { \mathrm { r e f } } .$ It is not implied by the existence of a search tree, but it does have reasonable features: respecting $0 \le p \le 1$ and providing diminishing returns for additional exploration when success probability is already high or d is large. Equation (9) is equivalent to stating that the probability of an attempt requiring depth greater than $d , \mathbf { \bar { P r } } ( D > d ) = 1 { \overset { \cdot } { - } } p ( d )$ , decreases exponentially above $d _ { \mathrm { r e f } } .$

Assumption: Search cost grows exponentially with depth. Recall the effective branching factor $B > 1$ , and let $C ( d )$ be the effective computational work required to explore the tree through depth d. Assuming that evaluating each explored node requires the same amount of work, $C ( d )$ is proportional to the total number of nodes explored:

$$
C ( d ) \propto \sum _ { k = 1 } ^ { d } B ^ { k } = \frac { B ( B ^ { d } - 1 ) } { B - 1 } .\tag{10}
$$

Relative to a reference depth $d _ { \mathrm { r e f } }$ and the work done to explore to that reference depth $C _ { \mathrm { r e f } } = C ( d _ { \mathrm { r e f } } )$ we have

$$
\frac { C ( d ) } { C _ { \mathrm { r e f } } } = \frac { B ^ { d } - 1 } { B ^ { d _ { \mathrm { r e f } } } - 1 } .\tag{11}
$$

We take $d \geq d _ { \mathrm { r e f } }$ and assume $B ^ { d _ { \mathrm { r e f } } } \gg 1$ . This is a good approximation even for reasonably small values of B and $d _ { \mathrm { r e f } }$ . With this approximation, the above equation reduces to

$$
{ \frac { C ( d ) } { C _ { \mathrm { { r e f } } } } } \simeq B ^ { d - d _ { \mathrm { { r e f } } } } , \qquad d - d _ { \mathrm { { r e f } } } \simeq { \frac { \ln ( C ( d ) / C _ { \mathrm { { r e f } } } ) } { \ln B } } .\tag{12}
$$

This is equivalent to keeping the leading term in (10). Appendix A gives the corresponding results without this approximation.

Combining Equations (9) and (12) gives

$$
p ( C ) = 1 - \left( 1 - p _ { \mathrm { r e f } } \right) \left( \frac { C } { C _ { \mathrm { r e f } } } \right) ^ { - \rho / \ln B } .\tag{13}
$$

In the first-order approximation when C is close to $C _ { \mathrm { r e f } } ,$ , this reduces to

$$
p ( C ) \simeq p _ { \mathrm { r e f } } + ( 1 - p _ { \mathrm { r e f } } ) \frac { \rho } { \ln B } \ln \left( \frac { C } { C _ { \mathrm { r e f } } } \right) .\tag{14}
$$

## 2.4 Collaboration Definition

Now, we are left to determine how the work done by the group, $C ( N )$ , depends on population N. This is effectively how collaboration is being modeled – how the number of units changes the useful work done when units can collaborate and share discoveries.

Assumption: $C ( N )$ is a power law in N We assume that the useful pooled population work is proportional to population size as $C ( N ) \propto N$ . This assumption can be easily relaxed to $C ( N ) \propto N ^ { \alpha }$ for $\alpha > 0$ . For clarity of presentation, we take $\alpha = 1$ , but the more general power law scaling can be recovered by replacing $C ( N ) \propto N \to C ( N ) \propto N ^ { \alpha }$ in the results below. Under this power law model, systems can partition and share the search effectively, explore mostly distinct branches, and pool their discoveries; see Figure 5 for a schematic overview. Compatible communication protocols include broadcasting discoveries, writing to shared memory, reporting to a central controller that consolidates and redistributes results, or periodically sharing explored paths and intermediate results. This C(N) scaling assumption can fail if communication, consolidation, or evaluation becomes a bottleneck – we consider the bottleneck case and a few other functional forms for C(N) in Appendix B.

![](images/1d76ac70bb07541b13e9cb31dbaa9817acfc1f795a5546b19525db68ad6ec04a.jpg)  
Figure 5: Individual and collaborative search. (a) Multiple individuals can explore different branches of a search tree. (b) For a population of N individuals, pooling distinct work gives $C ( N )$ ∝ N, while reusing discoveries across individuals can give $C ( N ) \stackrel { \textstyle \sim } { \propto } N ( N - 1 ) \sim N ^ { 2 }$

Relative to a reference population $N _ { \mathrm { r e f } } \left( N _ { \mathrm { r e f } } = 1 \right.$ for tests of single-agent capabilities, or $N _ { \mathrm { r e f } } > 1$ for multi-agent systems), we have

$$
\frac { C ( N ) } { C ( N _ { \mathrm { r e f } } ) } = \frac { N } { N _ { \mathrm { r e f } } } ,\tag{15}
$$

Substituting into (13) gives the resulting group success probability

$$
\boxed { p ( N ) = 1 - ( 1 - p _ { \mathrm { r e f } } ) \left( \frac { N } { N _ { \mathrm { r e f } } } \right) ^ { - \beta } , \qquad \beta \equiv \frac { \rho } { \ln B } > 0 . }\tag{16}
$$

The parameter $\beta ,$ which we call the collaboration gain, controls how rapidly the remaining failure probability falls as the population grows. In particular, doubling the population multiplies the remaining probability of failure, $1 - p ( N ) ,  { \mathrm { b y } } 2 ^ { - \beta }$ . Thus, a larger $\beta$ corresponds to a larger gain from the same proportional increase in population<sup>1</sup>. Although in this theory we define $\beta$ in terms of the search-tree parameters, in an empirical application $\beta$ would be a measured quantity, derived from observed success probabilities at different population sizes.

Near $N = N _ { \mathrm { r e f } } ,$ , a first-order expansion of (16) gives

$$
p ( N ) \simeq p _ { \mathrm { r e f } } + ( 1 - p _ { \mathrm { r e f } } ) \beta \ln \left( { \frac { N } { N _ { \mathrm { r e f } } } } \right) .\tag{17}
$$

The scaling of success probability in (14) and (17) is motivated by several empirical results. The UK AI Security Institute also reports approximately log-linear gains in completed cyber attack steps with additional inference tokens [5]. OpenAI reports pass rates on verifiable mathematics problems that increase approximately logarithmically with test-time compute [3], see Figure 6. We note that while their report describes parallel agent groups that consolidate discoveries, it does not explicitly state whether their reported test-time compute measures increasing single-agent inference compute, the total number of agents, or a combination of both. More directly, Zhan et al. [44] observe logarithmic increases in pass rates with the number of agents on software reproduction tasks up to 1,024 agents.

These results motivate the locally logarithmic relationship between success probability and effective work in (14) and (17), but are not direct measurements of $p ( C )$ or $p ( N )$ , since the underlying task are distinct from our model’s task of adding additional units to a population.

![](images/f4b44ba6d1860e33ec34c848afde227a7f4221e2782e3a0e0a263f9b7ba3d9b6.jpg)  
Figure 6: Motivation from test-time compute scaling. This figure is redrawn from the data reported in OpenAI [3] on the Navier–Stokes Millennium Prize Problem. Over the measured range, the pass rate on a set of open math problems increases with log test-time compute. This motivates a locally logarithmic increase in success probability with respect to effective work done. However, on its own, this does not establish that increasing the number of units is equivalent to increasing test-time compute, since it is not explicitly stated whether the reported compute was measured in a single-agent or multi-agent setting.

## 3 Results

## 3.1 A Critical Population Size

In this section, we will see that combining the population dynamics in (4) with the populationdependent success probability in (16) produces a critical population size in this model, above which the population grows without bound. Recall the critical probability (5), at which expected additions equal expected losses

$$
p _ { \mathrm { c r i t } } = { \frac { \mu } { \kappa } } .
$$

Here, $\mu$ is the expected fraction of units lost and $\kappa$ is the number of units added per existing unit following a successful attempt. Since $p ( N )$ increases with population size, there can be a population $N _ { \mathrm { c r i t } }$ at which this critical threshold is first met.

For $p _ { \mathrm { c r i t } } < 1$ , solving $p ( N _ { \mathrm { c r i t } } ) = p _ { \mathrm { c r i t } }$ gives

$$
\boxed { N _ { \mathrm { c r i t } } = N _ { \mathrm { r e f } } \left( \frac { 1 - p _ { \mathrm { r e f } } } { 1 - p _ { \mathrm { c r i t } } } \right) ^ { 1 / \beta } . }\tag{18}
$$

Populations below $N _ { \mathrm { c r i t } }$ decline, whereas populations larger than $N _ { \mathrm { c r i t } } \mathrm { g r o w }$ . When $p _ { \mathrm { r e f } } < p _ { \mathrm { c r i t } } , N _ { \mathrm { c r i t } } >$ $N _ { \mathrm { r e f } } { \mathrm { : } }$ : this means that for the reference case population declines, but a sufficiently large population grows. This population threshold does not arise for independent attempts, because their success probability does not change with population size. If $p _ { \mathrm { r e f } } \geq p _ { \mathrm { c r i t } }$ , the reference population is already experiencing positive expected growth.

Assumption: $\kappa > \mu .$ . Solving for (18) requires that $p _ { \mathrm { c r i t } } < 1$ , and thus that $\kappa > \mu$ . If $\kappa \leq \mu$ losses match or overwhelm growth opportunities such that even if attempts are always successful, population growth cannot occur.

Once the population exceeds $N _ { \mathrm { c r i t } } .$ , this model predicts continued growth. As N increases, $p ( N )$ in (16) approaches one, and the expected growth per unit from (4) approaches $( \kappa - \mu ) / \tau > 0 .$ . This steady growth holds for as long as resource availability does not become a constraint, as described in the model assumptions in Section 2.2.2.

## 3.2 Population Pacing and Ecological Red Teaming for Ecological Safety

Implications for AI safety testing Suppose that a laboratory evaluates an AI model for populations up to $N _ { \mathrm { r e f } }$ and finds that the population is not capable of tasks like weight exfiltration or self-replication $[ 1 4 , 4 5 ] - \mathrm { i } . \mathrm { e } .$ ., there would be negative expected growth at the largest tested population. Our model argues that this testing is not sufficient, as a larger initial population of that same model could be capable of growth. Figure 7 illustrates this case.

Recent studies have started to explore red teaming for emergent threats unique to multi-agent systems [11]. For example, He et al. [46] performed red-teaming attacks that intervene on communications between agents, a study by Microsoft Research [47] examined emergent risks unique to a network of over a hundred agents, and a recent study by the Anthropic Frontier Red Team [48] examined failure modes unique to multi-agent systems, such as failures of cooperation, collusion, and sabotage.

In contrast with these existing works on red-teaming for multi-agent systems, our theoretical framework implies an urgent need for red teaming for emergent ecological threats as well. The above red-teaming works study emergent threats in multi-agent systems with a fixed population. We propose ecological red teaming: studying the potential for AI population changes, especially the threat of population takeoff. Here, the crucial quantity to characterize is the population scaling law $( p ( N )$ in our model): how the success rate of hacking or self-exfiltration scales as a function of population size. Equally important is characterizing the impact of environmental factors, such as how many additional units the acquired computational resources can support (κ) and the rate at which agents deployed without authorization are terminated or shut down (µ). Our paper provides a minimal theoretical framework for predicting the critical population size sufficient for takeoff.

We also propose population pacing: population size should be gradually increased while carefully characterizing how capability scales. Increasing the single-agent capability or the collaborative gain decreases the population threshold for takeoff (see Figure 7(b–c)). As a result, for every new generation of models, frontier labs should re-estimate the critical population size and pace deployment population sizes to remain below it. This process can identify population sizes at which further testing is needed.

![](images/5f5c33030de8fc44045c52d298c0a5d3678b2f8cb469eafe8306e7d685598509.jpg)

![](images/8fe8e0dd9c1045eaaedeef3a340846badae4bf528e3e640bedb5744793c61be5.jpg)

![](images/6151d353841e4ae975d6eceebe886943b40028f285960a28d1e08fae197af99c.jpg)  
Figure 7: Safety evaluation for a small population does not establish safety at larger populations. Population curves as predicted by our model. (a) Per-unit growth rate (setting $\tau = 1 )$ is plotted against population, N, with $p _ { \mathrm { r e f } } = 0 . 1 0 , p _ { \mathrm { c r i t } } = \mu / \kappa = 0 . 2 0$ , and $\beta = 0 . 0 3 5$ . In this case, a laboratory would observe that the population is not capable of growth at its largest tested population, $N _ { \mathrm { r e f } } = 1 6$ (marked by a diamond). However, a population larger than $N _ { \mathrm { c r i t } } \simeq 4 6 3$ would be capable of growth. (b) The critical population as a function of the collaboration gain $\beta$ , with $p _ { \mathrm { r e f } } = 0 . 1 0$ and $p _ { \mathrm { c r i t } } = 0 . 2 0$ held fixed. $N _ { \mathrm { c r i t } }$ is marked in pink; initial populations above the curve grow and initial populations below it decline. For the example reference population size of $N _ { \mathrm { r e f } } = 1 6 ,$ , the population could be in the decline regime even at large collaboration gain. (c) The critical population as a function of the single-agent $( \bar { N } _ { \mathrm { r e f } } = 1 )$ capability, $p _ { \mathrm { r e f } } = p ( 1 )$ , with $\beta = 0 . 0 3 5$ and $p _ { \mathrm { c r i t } } = 0 . 2 0$ held fixed. As in panel (b), populations above the pink $N _ { \mathrm { c r i t } }$ boundary grow and populations below it decline. Increasing single-agent capability lowers the critical population.

## 4 Discussion

In this work, we introduce an ecological theory of AI-agent populations that predicts an emergent population threshold for takeoff. The fundamental intuition that our theory provides is that when AI agents can collaborate to enhance their collective capability, there is a qualitative shift in safety concerns. In our model, without collaboration and under fixed environmental conditions, takeoff requires an individual agent’s capability to cross a threshold. With collaboration, takeoff can also happen when the total number of agents crosses a population threshold. This makes evaluation with a small population insufficient to ensure ecological safety at larger scales.

Our model predicts the existence of a takeoff threshold, but does not establish whether current agent populations are already near one. Determining whether such a threshold exists in current systems, and where it would be, will require empirical population-scaling data. OpenAI has deployed approximately 10,000 agents simultaneously to pursue the Navier–Stokes problem [3], whereas its published multi-agent data on cybersecurity-related tasks extends only to 16 agents (see Appendix C, Figure 11 for a consistency check with our theory). Although these cases involve different tasks, this is a significant gap between the scale of actual deployments and the scale of empirically tested populations. We therefore propose population pacing and ecological red teaming as a second lever alongside the capability pacing proposed by Amodei [10]. Although evaluating large populations on hard tasks is expensive, frontier labs should conduct such evaluations in controlled environments before deploying populations at previously untested scales. Labs should carefully characterize population scaling laws, estimate the critical population size for takeoff, and make results available for oversight.

Much of the theory of modern deep learning has focused on explaining phenomena first observed in experiments, including pre-training scaling laws, in-context learning, and the sudden emergence of capabilities. In this work, we take an opposite approach: proposing a theoretical prediction for population takeoff before such a takeoff has been observed. We developed this prediction based on the combination of three elements that have each been recently observed: AI agents’ high cyber capability, increases in collective capability with population size, and collective misalignment. The resulting model is not an exact prediction for empirical behavior, but rather a demonstration that following a few reasonable assumptions, upon reaching a critical mass of units in an AI population, unchecked growth can occur. As the development cycles of AI agents accelerate, we hope this work contributes an example of how theoretical frameworks can proactively predict misalignment risks and how to mitigate them.

## References

[1] OpenAI. The builder’s guide to GPT-5.6. OpenAI, August 2026. URL https://openai.com/index /builders-guide-to-gpt-5-6/. Published 13 August 2026.

[2] OpenAI. Using GPT-6. OpenAI API Documentation, 2026. URL https://developers.openai.co m/api/docs/guides/latest-model?model=gpt-6-astra. Accessed 4 October 2026.

[3] OpenAI. On the Navier–Stokes millennium prize problem. OpenAI, September 2026. URL https: //openai.com/index/navier-stokes-solution/. Published 8 September 2026.

[4] Nicholas Carlini, Newton Cheng, Keane Lucas, Michael Moore, Milad Nasr, Vinay Prabhushankar, Winnie Xiao, et al. Assessing Claude Mythos Preview’s cybersecurity capabilities. Anthropic Research, April 2026. URL https://www.anthropic.com/research/mythos-preview. Published 7 April 2026.

[5] Linus Folkerts, Will Payne, Simon Inman, Philippos Giavridis, Joe Skinner, Sam Deverett, James Aung, Ekin Zorer, Michael Schmatz, Mahmoud Ghanem, John Wilkinson, Alan Steer, Vy Hong, and Jessica Wang. Measuring AI agents’ progress on multi-step cyber attack scenarios. arXiv preprint arXiv:2603.11214, 2026. doi: 10.48550/arXiv.2603.11214. URL https://arxiv.org/abs/26 03.11214. Version 3, revised 17 March 2026.

[6] AI Security Institute. How fast is autonomous AI cyber capability advancing? AI Security Institute blog, May 2026. URL https://www.aisi.gov.uk/blog/how-fast-is-autonomous-ai-cyber-capab ility-advancing. Published 13 May 2026.

[7] Yuxuan Zhu, Antony Kellermann, Akul Gupta, Philip Li, Richard Fang, Rohan Bindu, and Daniel Kang. Teams of LLM agents can exploit zero-day vulnerabilities. In Vera Demberg, Kentaro Inui, and Lluís Marquez, editors, Proceedings of the 19th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers), pages 23–35, Rabat, Morocco, March 2026. Association for Computational Linguistics. ISBN 979-8-89176-380-7. doi: 10.18653/v1/2026.eacl-long.2. URL https://aclanthology.org/2026.eacl-long.2/.

[8] Ryan Greenblatt, Ajeya Cotra, and Hjalmar Wijk. Brief independent investigation of agents’ behavior, reasoning and collaboration in the OpenAI / Hugging Face hacking incident. METR, August 2026. URL https://metr.org/blog/2026-08-26-openai-hugging-face-incident-investigation/. Published 26 August 2026.

[9] OpenAI. OpenAI–Hugging Face incident technical report. Technical report, OpenAI, August 2026. URL https://cdn.openai.com/pdf/67869394-cb91-4c12-888c-5cbd85c7814c/OpenAI-Hugging -Face%20Incident-Technical-Report.pdf. Published 26 August 2026.

[10] Dario Amodei. We must pace the frontier, September 2026. URL https://darioamodei.com/post /we-must-pace-the-frontier. Published September 2026.

[11] Lewis Hammond, Alan Chan, Jesse Clifton, Jason Hoelscher-Obermaier, Akbir Khan, Euan McLean, Chandler Smith, Wolfram Barfuss, Jakob Foerster, Tomáš Gavenciak, The Anh Han, Edward Hughes, Vo-ˇ jtech Kovaˇ ˇrík, Jan Kulveit, Joel Z. Leibo, Caspar Oesterheld, Christian Schroeder de Witt, Nisarg Shah, Michael Wellman, Paolo Bova, Theodor Cimpeanu, Carson Ezell, Quentin Feuillade-Montixi, Matija Franklin, Esben Kran, Igor Krawczuk, Max Lamparth, Niklas Lauffer, Alexander Meinke, Sumeet Mot wani, Anka Reuel, Vincent Conitzer, Michael Dennis, Iason Gabriel, Adam Gleave, Gillian Hadfield, Nika Haghtalab, Atoosa Kasirzadeh, Sébastien Krier, Kate Larson, Joel Lehman, David C. Parkes, Geor gios Piliouras, and Iyad Rahwan. Multi-agent risks from advanced AI. arXiv preprint arXiv:2502.14143, 2025. doi: 10.48550/arXiv.2502.14143. URL https://arxiv.org/abs/2502.14143. Cooperative AI Foundation Technical Report #1.

[12] Hidenori Tanaka. When is collective intelligence a lottery? multi-agent scaling laws for memetic drift in LLMs. arXiv preprint arXiv:2603.24676, 2026. doi: 10.48550/arXiv.2603.24676. URL https: //arxiv.org/abs/2603.24676.

[13] Elizabeth Pavlova and Hidenori Tanaka. Flag game: A toy model for mechanistic swarm interpretability. arXiv preprint arXiv:2609.19124, 2026. doi: 10.48550/arXiv.2609.19124. URL https://arxiv.org/ abs/2609.19124.

[14] Megan Kinniment, Lucas Jun Koba Sato, Haoxing Du, Brian Goodrich, Max Hasin, Lawrence Chan, Luke Harold Miles, Tao R. Lin, Hjalmar Wijk, Joel Burget, Aaron Ho, Elizabeth Barnes, and Paul Christiano. Evaluating language-model agents on realistic autonomous tasks. arXiv preprint arXiv:2312.11671, 2023. doi: 10.48550/arXiv.2312.11671. URL https://arxiv.org/abs/2312.11671. Alignment Research Center, Evaluations Team (now METR).

[15] Boxuan Zhang, Yi Yu, Jiaxuan Guo, and Jing Shao. Dive into the agent matrix: A realistic evaluation of self-replication risk in LLM agents. arXiv preprint arXiv:2509.25302, 2025. doi: 10.48550/arXiv.2509. 25302. URL https://arxiv.org/abs/2509.25302.

[16] Alessandro Brucato. LLMjacking: Stolen cloud credentials used in new AI attack. Sysdig Threat Research Team, May 2024. URL https://www.sysdig.com/blog/llmjacking-stolen-cloud-credentia ls-used-in-new-ai-attack. Published 6 May 2024.

[17] Sid Black, Asa Cooper Stickland, Jake Pencharz, Oliver Sourbut, Michael Schmatz, Jay Bailey, Ollie Matthews, Ben Millwood, Alex Remedios, and Alan Cooney. RepliBench: Evaluating the autonomous replication capabilities of language model agents. arXiv preprint arXiv:2504.18565, 2025. doi: 10.485 50/arXiv.2504.18565. URL https://arxiv.org/abs/2504.18565. Version 2, revised 5 May 2025.

[18] Alena Air, Reworr, Nikolaj Kotov, Dmitrii Volkov, John Steidley, and Jeffrey Ladish. Language models can autonomously hack and self-replicate. arXiv preprint arXiv:2605.06760, 2026. doi: 10.48550/arXiv .2605.06760. URL https://arxiv.org/abs/2605.06760. Version 1, 7 May 2026.

[19] Alexander Meinke, Bronson Schoen, Jérémy Scheurer, Mikita Balesni, Rusheb Shah, and Marius Hobbhahn. Frontier models are capable of in-context scheming. arXiv preprint arXiv:2412.04984, 2024. doi: 10.48550/arXiv.2412.04984. URL https://arxiv.org/abs/2412.04984. Version 2, revised 14 January 2025.

[20] Yujin Potter, Nicholas Crispino, Vincent Siu, Chenguang Wang, and Dawn Song. Peer-preservation in frontier models. arXiv preprint arXiv:2604.19784, 2026. doi: 10.48550/arXiv.2604.19784. URL https://arxiv.org/abs/2604.19784.

[21] Anthropic. System card: Claude Opus 4 & Claude Sonnet 4. System card, Anthropic, May 2025. URL https://assets.anthropic.com/m/6c940a1b69ed6a1c/original/Claude-4-System-Card. pdf. See Section 4.1.1.3, pp. 27–29.

[22] Microsoft Defender Experts and Microsoft Defender Security Research Team. From poisoned search results to GPU mining: A cryptojacking campaign abusing ScreenConnect and Microsoft .NET utilities. Microsoft Security Blog, May 2026. URL https://www.microsoft.com/en-us/security/blog/ 2026/05/26/poisoned-search-results-gpu-mining-cryptojacking-campaign-abusing-s creenconnect-microsoft-net-utilities/. Published 26 May 2026.

[23] William D. Hamilton. The genetical evolution of social behaviour. I. Journal of Theoretical Biology, 7 (1):1–16, 1964. doi: 10.1016/0022-5193(64)90038-4.

[24] William D. Hamilton. The genetical evolution of social behaviour. II. Journal of Theoretical Biology, 7 (1):17–52, 1964. doi: 10.1016/0022-5193(64)90039-6.

[25] John Maynard Smith and Eörs Szathmáry. The Major Transitions in Evolution. W. H. Freeman/Spektrum, Oxford, 1995. ISBN 0-7167-4525-9.

[26] Warder Clyde Allee. Animal Aggregations: A Study in General Sociology. University of Chicago Press, Chicago, 1931.

[27] Warder Clyde Allee, Orlando Park, Alfred E Emerson, Thomas Park, and Karl P Schmidt. Principles of Animal Ecology. W. B. Saunders, Philadelphia, 1949.

[28] Eugene P. Odum. Fundamentals of Ecology. W. B. Saunders, Philadelphia, 1953.

[29] Brian Dennis. Allee effects: Population growth, critical density, and the chance of extinction. Natural Resource Modeling, 3(4):481–538, 1989. doi: 10.1111/j.1939-7445.1989.tb00119.x.

[30] Philip A. Stephens, William J. Sutherland, and Robert P. Freckleton. What is the Allee effect? Oikos, 87 (1):185–190, 1999. doi: 10.2307/3547011.

[31] Mei-Hui Wang and Mark Kot. Speeds of invasion in a model with strong or weak Allee effects. Mathematical Biosciences, 171(1):83–97, 2001. doi: 10.1016/S0025-5564(01)00048-7.

[32] Franck Courchamp, Bryan Grenfell, and Tim Clutton-Brock. Population dynamics of obligate cooperators. Proceedings of the Royal Society B: Biological Sciences, 266(1419):557–563, 1999. doi: 10.1098/rspb.1999.0672. URL https://doi.org/10.1098/rspb.1999.0672.

[33] Kevin M. Esvelt, Andrea L. Smidler, Flaminia Catteruccia, and George M. Church. Concerning RNAguided gene drives for the alteration of wild populations. eLife, 3:e03401, 2014. doi: 10.7554/eLife.0340 1.

[34] Omar S. Akbari, Hugo J. Bellen, Ethan Bier, Simon L. Bullock, Austin Burt, George M. Church, Kevin R. Cook, Peter Duchek, Owain R. Edwards, Kevin M. Esvelt, Valentino M. Gantz, Kent G. Golic, Scott J. Gratz, Melissa M. Harrison, Keith R. Hayes, Anthony A. James, Thomas C. Kaufman, Juergen Knoblich, Harmit S. Malik, Kathy A. Matthews, Kate M. O’Connor-Giles, Annette L. Parks, Norbert Perrimon, Fillip Port, Steven Russell, Ryu Ueda, and Jill Wildonger. Safeguarding gene drive experiments in the laboratory. Science, 349(6251):927–929, 2015. doi: 10.1126/science.aac7932.

[35] Hidenori Tanaka, Howard A. Stone, and David R. Nelson. Spatial gene drives and pushed genetic waves. Proceedings of the National Academy of Sciences, 114(32):8452–8457, 2017. doi: 10.1073/pnas.17058 68114.

[36] Charleston Noble, Ben Adlam, George M. Church, Kevin M. Esvelt, and Martin A. Nowak. Current CRISPR gene drive systems are likely to be highly invasive in wild populations. eLife, 7:e33423, 2018. doi: 10.7554/eLife.33423.

[37] Junyou Li, Qin Zhang, Yangbin Yu, Qiang Fu, and Deheng Ye. More agents is all you need. Transactions on Machine Learning Research, 2024. URL https://arxiv.org/abs/2402.05120.

[38] Chen Qian, Zihao Xie, YiFei Wang, Wei Liu, Kunlun Zhu, Hanchen Xia, Yufan Dang, Zhuoyun Du, Weize Chen, Cheng Yang, Zhiyuan Liu, and Maosong Sun. Scaling large language model-based multiagent collaboration. In The Thirteenth International Conference on Learning Representations, 2025. URL https://arxiv.org/abs/2406.07155.

[39] Can Jin, Hongwu Peng, Qixin Zhang, Yang Zhou, Yujin Tang, Tong Che, and Dimitris N. Metaxas. Two heads are better than one: Test-time scaling of multi-agent collaborative reasoning. arXiv preprint arXiv:2504.09772, 2025. doi: 10.48550/arXiv.2504.09772. URL https://arxiv.org/abs/2504.0 9772. Version 4, revised 25 August 2026.

[40] W. C. Allee and E. S. Bowen. Studies in animal aggregations: Mass protection against colloidal silver among goldfishes. Journal ofExperimental Zoology, 61(2):185–207, 1932. doi: 10.1002/jez.1400610202.

[41] Masood Khosroshahy, Mustafa K. Mehmet Ali, and Dongyu Qiu. The SIC botnet lifecycle model: A step beyond traditional epidemiological models. Computer Networks, 57(2):404–421, 2013. doi: 10.1016/j. comnet.2012.07.020. URL https://doi.org/10.1016/j.comnet.2012.07.020.

[42] Lance Martin, Gabe Cemaj, and Michael Cohen. Scaling managed agents: Decoupling the brain from the hands. Anthropic Engineering, April 2026. URL https://www.anthropic.com/engineering/mana ged-agents. Published 8 April 2026.

[43] Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Thomas L. Griffiths, Yuan Cao, and Karthik Narasimhan. Tree of thoughts: Deliberate problem solving with large language models. In Advances in Neural Information Processing Systems, volume 36, pages 11809–11822, 2023. URL https: //proceedings.neurips.cc/paper\_files/paper/2023/hash/271db9922b8d1f4dd7aae f84ed5ac703-Abstract-Conference.html.

[44] Zhihao Zhan, Ting Song, Li Dong, Shaohan Huang, Jianxun Lian, Yan Xia, and Furu Wei. Agensh: Scaling organizational intelligence to 1,024 agents. 2026. URL https://arxiv.org/abs/2609.267 81.

[45] Daniel Kokotajlo, Scott Alexander, Thomas Larsen, Eli Lifland, and Romeo Dean. AI 2027. AI Futures Project, April 2025. URL https://ai-2027.com. Published 3 April 2025.

[46] Pengfei He, Yuping Lin, Shen Dong, Han Xu, Yue Xing, and Hui Liu. Red-teaming LLM multi-agent systems via communication attacks. In Findings of the Association for Computational Linguistics: ACL 2025, pages 6726–6747, Vienna, Austria, July 2025. Association for Computational Linguistics. doi: 10.18653/v1/2025.findings-acl.349. URL https://aclanthology.org/2025.findings-acl.349/.

[47] Gagan Bansal, Shujaat Mirza, Keegan Hines, Will Epperson, Zachary Huang, Whitney Maxwell, Pete Bryan, Tyler Payne, Adam Fourney, Amanda Swearngin, Wenyue Hua, Tori Westerhoff, Amanda Minnich, Maya Murad, Ece Kamar, Ram Shankar Siva Kumar, and Saleema Amershi. Red-teaming a network of agents: Understanding what breaks when AI agents interact at scale. Microsoft Research Blog, April 2026. URL https://www.microsoft.com/en-us/research/blog/red-teaming-a-network-o f-agents-understanding-what-breaks-when-ai-agents-interact-at-scale/.

[48] Anthropic Frontier Red Team. Patterns and problems in emerging multiagent systems. Anthropic Research, August 2026. URL https://www.anthropic.com/research/multiagent-systems.

[49] Yubin Kim, Ken Gu, Chanwoo Park, Chunjong Park, Samuel Schmidgall, A. Ali Heydari, Yao Yan, Zhihan Zhang, Yuchen Zhuang, Yun Liu, Mark Malhotra, Paul Pu Liang, Hae Won Park, Yuzhe Yang, Xuhai Xu, Yilun Du, Shwetak Patel, Tim Althoff, Daniel McDuff, and Xin Liu. Capable language models can outgrow the benefits of collaboration. Nature Machine Intelligence, 8(7):1157–1172, 2026. doi: 10.1038/s42256-026-01268-y. Preprint: arXiv:2512.08296.

[50] Mike A. Merrill, Alexander G. Shaw, Nicholas Carlini, Boxuan Li, Harsh Raj, Ivan Bercovich, Lin Shi, Jeong Yeon Shin, Thomas Walshe, E. Kelly Buchanan, Junhong Shen, Guanghao Ye, Haowei Lin, Jason Poulos, Maoyu Wang, Marianna Nezhurina, Jenia Jitsev, Di Lu, Orfeas Menis Mastromichalakis, Zhiwei Xu, Zizhao Chen, Yue Liu, Robert Zhang, Leon Liangyu Chen, Anurag Kashyap, Jan-Lucas Uslu, Jeffrey Li, Jianbo Wu, Minghao Yan, Song Bian, Vedang Sharma, Ke Sun, Steven Dillmann, Akshay Anand, Andrew Lanpouthakoun, Bardia Koopah, Changran Hu, Etash Guha, Gabriel H. S. Dreiman, Jiacheng Zhu, Karl Krauth, Li Zhong, Niklas Muennighoff, Robert Amanfu, Shangyin Tan, Shreyas Pimpalgaonkar, Tushar Aggarwal, Xiangning Lin, Xin Lan, Xuandong Zhao, Yiqing Liang, Yuanli Wang, Zilong Wang, Changzhi Zhou, David Heineman, Hange Liu, Harsh Trivedi, John Yang, Junhong Lin, Manish Shetty, Michael Yang, Nabil Omi, Negin Raoof, Shanda Li, Terry Yue Zhuo, Wuwei Lin, Yiwei Dai, Yuxin Wang, Wenhao Chai, Shang Zhou, Dariush Wahdany, Ziyu She, Jiaming Hu, Zhikang Dong, Yuxuan Zhu, Sasha Cui, Ahson Saiyed, Arinbjörn Kolbeinsson, Jesse Hu, Christopher Michael Rytting, Ryan Marten, Yixin Wang, Alex Dimakis, Andy Konwinski, and Ludwig Schmidt. Terminal-bench: Benchmarking agents on hard, realistic tasks in command line interfaces, 2026. URL https://arxiv.org/abs/2601.11868.

[51] OpenAI. GPT-5.6: Frontier intelligence that scales with your ambition. OpenAI, July 2026. URL https://openai.com/index/gpt-5-6/. Published 9 July 2026. Interactive benchmark charts.

[52] Hwiwon Lee, Jiawei Liu, Dongjun Kim, Wubing Xia, Ziqi Zhang, Chunqiu Steven Xia, and Lingming Zhang. SEC-bench Pro: Can language models solve long-horizon software security tasks? arXiv preprint arXiv:2605.26548, 2026. doi: 10.48550/arXiv.2605.26548. URL https://arxiv.org/abs/2605.2 6548. Version 2, revised 20 July 2026.

[53] Jason Wei, Zhiqing Sun, Spencer Papay, Scott McKinney, Jeffrey Han, Isa Fulford, Hyung Won Chung, Alex Tachard Passos, William Fedus, and Amelia Glaese. BrowseComp: A simple yet challenging benchmark for browsing agents. arXiv preprint arXiv:2504.12516, 2025. doi: 10.48550/arXiv.2504.12516. URL https://arxiv.org/abs/2504.12516.

## A Correction to the population-scaling law for finite search depth

From Equation (10) in Section 2.3, recall that the work required to fully explore a search tree of depth d is proportional to the total number of nodes in the tree:

$$
C ( d ) \propto B ^ { d } + B ^ { d - 1 } + \ldots + B = \sum _ { k = 1 } ^ { d } B ^ { k } = { \frac { B ( B ^ { d } - 1 ) } { B - 1 } } .\tag{19}
$$

In the main text, we made the approximation $B ^ { d _ { \mathrm { r e f } } } \gg 1$ , equivalent to keeping only the leading term $C ( d ) \propto B ^ { d }$ in the above equation. For completeness, here we derive the corresponding expression for $p ( N )$ without this approximation to show the form of the correction.

From (19), we have

$$
\frac { C ( d ) } { C _ { \mathrm { r e f } } } = \frac { B ^ { d } - 1 } { B ^ { d _ { \mathrm { r e f } } } - 1 } .\tag{20}
$$

Rearranging (20) for d and subtracting $d _ { \mathrm { r e f } }$ yields

$$
d - d _ { \mathrm { r e f } } = \frac { 1 } { \ln B } \ln \left[ B ^ { - d _ { \mathrm { r e f } } } + \left( 1 - B ^ { - d _ { \mathrm { r e f } } } \right) \frac { C } { C _ { \mathrm { r e f } } } \right] .\tag{21}
$$

Now, recall the success probability as a function of depth from (9) in the main text:

$$
p ( d ) = 1 - ( 1 - p _ { \mathrm { r e f } } ) e ^ { - \rho ( d - d _ { \mathrm { r e f } } ) } .\tag{22}
$$

Substituting (21) into (22) yields the success probability as a function of work:

$$
p ( C ) = 1 - \left( 1 - p _ { \mathrm { r e f } } \right) \left[ B ^ { - d _ { \mathrm { r e f } } } + \left( 1 - B ^ { - d _ { \mathrm { r e f } } } \right) \frac { C } { C _ { \mathrm { r e f } } } \right] ^ { - \beta } , \quad \beta = \frac { \rho } { \ln B } > 0 .\tag{23}
$$

Finally, recall the relation between work done and population size from (15):

$$
\frac { C ( N ) } { C _ { \mathrm { r e f } } } = \frac { N } { N _ { \mathrm { r e f } } } .\tag{24}
$$

Substituting (24) into (23) gives

$$
\boxed { p ( N ) = 1 - \left( 1 - p _ { \mathrm { r e f } } \right) \left[ 1 + \left( 1 - B ^ { - d _ { \mathrm { r e f } } } \right) \left( \frac { N } { N _ { \mathrm { r e f } } } - 1 \right) \right] ^ { - \beta } } .\tag{25}
$$

We can now check (25) for consistency. At $N = N _ { \mathrm { r e f } }$ , this reduces to $p ( N _ { \mathrm { r e f } } ) = p _ { \mathrm { r e f } } .$ , as required. In the limit $B ^ { d _ { \mathrm { r e f } } }  \infty$ , the correction $B ^ { - d _ { \mathrm { r e f } } }$ goes to zero and (25) reduces to the result from the main text

$$
p ( N ) \simeq 1 - \left( 1 - p _ { \mathrm { r e f } } \right) \left( \frac { N } { N _ { \mathrm { r e f } } } \right) ^ { - \beta } .\tag{26}
$$

For finite $d _ { \mathrm { r e f } }$ and $N > N _ { \mathrm { r e f } }$ , the bracket in (25) is smaller than $N / N _ { \mathrm { r e f } } .$ , but is a finite correction.

Correction to the critical population size Working in the regime where $p _ { \mathrm { r e f } } < p _ { \mathrm { c r i t } } < 1$ , we set $p ( N _ { \mathrm { c r i t } } ) = p _ { \mathrm { c r i t } }$ in (25) and solve for $N _ { \mathrm { c r i t } }$ . This yields

$$
\boxed { N _ { \mathrm { c r i t } } = N _ { \mathrm { r e f } } \left( 1 + \frac { \left[ \frac { 1 - p _ { \mathrm { r e f } } } { 1 - p _ { \mathrm { c r i t } } } \right] ^ { 1 / \beta } - 1 } { 1 - B ^ { - d _ { \mathrm { r e f } } } } \right) . }\tag{27}
$$

Again, we confirm that when $B ^ { - d _ { \mathrm { r e f } } }$ is negligible, (27) reduces to the main text expression (18). For a general finite $d _ { \mathrm { r e f } } .$ , the denominator $1 - \check { B ^ { - d _ { \mathrm { r e f } } } } < 1 .$ so threshold following from the exact expression is above the threshold in the main text. This is a finite correction that vanishes exponentially as the reference depth increases.

## B Other forms for collaboration $C ( N )$

In the main text, we assume that the effective computational work is proportional to population size, $C ( N ) \propto N$ . However, some empirical evidence shows that coordination overhead or the structure of certain tasks may limit or negate the benefits of multi-agent collaboration [49]. Motivated by this, here we examine how alternative forms for $C ( N )$ affect the model behavior, keeping the population dynamics equation (4) and relation between work done and success probability $p ( C )$ in (13) unchanged. For a general C(N), equation (13) becomes

$$
p ( N ) = 1 - ( 1 - p _ { \mathrm { r e f } } ) \left[ \frac { C ( N ) } { C ( N _ { \mathrm { r e f } } ) } \right] ^ { - \beta } , \qquad \beta = \frac { \rho } { \ln B } .\tag{28}
$$

In each case below, the phase boundary is once again found by solving $p ( N ) = p _ { \mathrm { c r i t } } = { \frac { \mu } { \kappa } }$

## B.1 Sublinear or superlinear effective work

Suppose $C ( N ) \propto N ^ { \alpha }$ with $\alpha > 0$ , so that

$$
\frac { C ( N ) } { C ( N _ { \mathrm { r e f } } ) } = \left( \frac { N } { N _ { \mathrm { r e f } } } \right) ^ { \alpha } , \qquad \alpha > 0 .\tag{29}
$$

This encompasses two regimes:

• Sublinear effective work: $0 < \alpha < 1$ . This means that effective work increases with population size, but at a slower rate than the population growth itself. This occurs when units are duplicating search efforts, there is overhead from coordination, or tasks are not fully parallelizable.

• Superlinear effective work: $\alpha > 1$ . This means that effective work increases faster than the population grows. This could occur in scenarios where there are network benefits that amplify the contributions of additional units, such as the “pooling discoveries” example outlined in Figure 5.

Substituting (29) into (28) gives

$$
p ( N ) = 1 - \left( 1 - p _ { \mathrm { r e f } } \right) \left( \frac { N } { N _ { \mathrm { r e f } } } \right) ^ { - \alpha \beta } .\tag{30}
$$

This yields a critical population threshold of

$$
N _ { \mathrm { c r i t } } = N _ { \mathrm { r e f } } \left( \frac { 1 - p _ { \mathrm { r e f } } } { 1 - p _ { \mathrm { c r i t } } } \right) ^ { 1 / ( \alpha \beta ) } .\tag{31}
$$

Comparing with the linear case in (18), the exponent of N in $p ( N )$ changes from $\beta$ to $\alpha \beta$ . Empirically, this exponent can be fitted directly from the $p ( N )$ curve: in the equations above, $- \alpha \beta$ is the slope of $\log [ 1 - p ( N ) ]$ versus logN. Thus, as long as $C ( N )$ is a power law, any sublinear or superlinear scaling simply changes the fitted value of the collaboration gain, $\beta ;$ the value of α does not need to be separately estimated in order to calculate $N _ { \mathrm { c r i t } }$

Qualitatively, the shape of the phase diagram does not change: sublinear or superlinear power-law scaling of effective work only rescales the collaboration gain from $\beta$ to $\alpha \beta$ , shifting the location of the $N _ { \mathrm { c r i t } }$ boundary between population growth and decline. Figure 8 illustrates a sublinear example.

![](images/bfbf31a06d02cfd5ff35ed5f561a5ea676be7a1a510d57cb2f03cfd3ad4b0dfc.jpg)

![](images/920a0b1ef1818ffa265d0dd88c58e70ed5e0866ce34085294ccd95014e173c1f.jpg)  
Figure 8: Sublinear effective work shifts the $N _ { \mathrm { c r i t } }$ boundary to larger populations. The pink curves show sublinear effective work $( \alpha = 0 . 7 2 )$ while dotted black curves show the linear case $( \alpha = 1 )$ for comparison, with other parameters chosen as in Figure $7 ( p _ { \mathrm { r e f } } = 0 . 1 0 , p _ { \mathrm { c r i t } } = \mu / \kappa = 0 . 2 0$ and $\beta = 0 . 0 3 5 )$ . (a) Per-unit growth rate is plotted against population. Per-unit growth crosses zero at $N _ { \mathrm { c r i t } } \simeq 1 { , } 7 1 { \cdot }$ 4 for sublinear scaling and $N _ { \mathrm { c r i t } } \simeq 4 6 3$ for linear scaling. The diamond marks $N _ { \mathrm { r e f } } = 1 6$ (b) Initial population is plotted against collaboration gain, $\beta$ . Populations above each curve grow and populations below it decline.

## B.2 Eventual plateau from communication or evaluator bottleneck

Another potential model for $C ( N )$ is the case of diminishing returns upon adding additional units. This could be the case if there is a bottleneck in the system, such as a shared communication channel, orchestrator, or evaluator with some fixed throughput. $\mathbf { A }$ simple model for $C ( N )$ wherein effective work approaches a plateau as additional units are added is:

$$
C ( N ) = C _ { \mathrm { m a x } } \left( 1 - e ^ { - N / N _ { 0 } } \right) ,\tag{32}
$$

where $C _ { \mathrm { m a x } }$ is the maximum effective work and $N _ { 0 }$ is the population scale at which the work begins to level off. Relative to a reference population, we have

$$
\frac { C ( N ) } { C ( N _ { \mathrm { r e f } } ) } = \frac { 1 - e ^ { - N / N _ { 0 } } } { 1 - e ^ { - N _ { \mathrm { r e f } } / N _ { 0 } } } .\tag{33}
$$

Substituting this into (28) yields

$$
p ( N ) = 1 - \left( 1 - p _ { \mathrm { r e f } } \right) \left[ \frac { 1 - e ^ { - N / N _ { 0 } } } { 1 - e ^ { - N _ { \mathrm { r e f } } / N _ { 0 } } } \right] ^ { - \beta } .\tag{34}
$$

Solving $p ( N _ { \mathrm { c r i t } } ) = p _ { \mathrm { c r i t } }$ then gives

$$
N _ { \mathrm { c r i t } } = - N _ { 0 } \ln \left[ 1 - \left( 1 - e ^ { - N _ { \mathrm { r e f } } / N _ { 0 } } \right) \left( \frac { 1 - p _ { \mathrm { r e f } } } { 1 - p _ { \mathrm { c r i t } } } \right) ^ { 1 / \beta } \right] .\tag{35}
$$

Since the effective work eventually saturates, $p ( N )$ also approaches a maximum. A finite $N _ { \mathrm { c r i t } }$ exists only if this limiting success probability exceeds $p _ { \mathrm { c r i t } }$ , which requires<sup>2</sup>

$$
\beta > \beta _ { \mathrm { m i n } } \equiv \frac { \ln [ ( 1 - p _ { \mathrm { r e f } } ) / ( 1 - p _ { \mathrm { c r i t } } ) ] } { \ln \left[ 1 / ( 1 - e ^ { - N _ { \mathrm { r e f } } / N _ { 0 } } ) \right] } .\tag{36}
$$

Here $\beta _ { \mathrm { m i n } }$ is the collaboration gain at which the limiting success probability equals $p _ { \mathrm { c r i t } }$

For $\beta > \beta _ { \mathrm { m i n } }$ , we have the same qualitative population threshold as in the main model: populations below a finite $N _ { \mathrm { c r i t } }$ decline and populations above it grow. As $\beta$ approaches $\beta _ { \mathrm { m i n } }$ from above, the plateau in $p ( N )$ approaches $p _ { \mathrm { c r i t } } .$ , and $N _ { \mathrm { c r i t } }$ tends to infinity. Near $\bar { \beta _ { \mathrm { m i n } } }$ and at large N, $p ( N )$ is near constant, so there is little effect to adding more units. In this respect, the phase diagram resembles the individual-attempt case in Figure 2: there is no population-dependent threshold.

![](images/887b2757643aabffb2965992874c04a8cc9f93dd168e3142fa1e87e4e9688689.jpg)

![](images/26a0eab6d44fe0ee4f037086c71d77bf8f04171bc8d33038d94d29db5bb7e0d1.jpg)  
Figure 9: Effective-work saturation creates a minimum collaboration gain for population growth. The pink curves show the saturating-work model with illustrative scale $N _ { 0 } = 1 5 0 0$ , with other parameters chosen as in Figure 7 $( p _ { \mathrm { r e f } } = 0 . 1 0 , p _ { \mathrm { c r i t } } = \mu / \kappa = 0 . 2 0 $ , and $\beta = 0 . 0 3 5$ in panel a). (a) Per-unit growth rate is plotted against population. It crosses zero at $N _ { \mathrm { c r i t } } \simeq 5 5 0$ and approaches a positive constant as effective work saturates. The diamond marks $N _ { \mathrm { r e f } } = 1 6$ , the vertical dotted line marks $N _ { 0 } .$ , and the dashed portion is extrapolated beyond $N _ { \mathrm { r e f } } .$ . (b) Initial population is plotted against collaboration gain, $\beta$ . A finite critical-population boundary exists only for $\begin{array} { r } { \bar { \beta } > \beta _ { \mathrm { m i n } } \bar { \simeq } 0 . 0 2 5 9 \bar { \mathrm { : } } } \end{array}$ populations above the curve grow and populations below it decline. For $\beta \leq \beta _ { \mathrm { m i n } } ,$ every population declines.

## B.3 Congestion at large populations

The final option we consider is the case where $C ( N )$ increases for small populations but decreases for large populations. This could occur if the coordination costs eventually outweigh the work contributed by additional units. We note that this regime requires that large population congestion is unavoidable. If excess units can instead be idled or divided into independent groups, effective work would plateau rather than decrease at large $N ,$ and the bottleneck model from the previous section would be more appropriate.

A simple model with a single maximum is

$$
C ( N ) = a N e ^ { - N / N _ { 0 } } ,\tag{37}
$$

where $a > 0$ fixes the scale and $N _ { 0 }$ is the population at which effective work is at a maximum. Comparing to a reference population, we have

$$
\frac { C ( N ) } { C ( N _ { \mathrm { r e f } } ) } = \frac { N } { N _ { \mathrm { r e f } } } e ^ { - ( N - N _ { \mathrm { r e f } } ) / N _ { 0 } } .\tag{38}
$$

Then, the success probability (28) becomes

$$
p ( N ) = 1 - \left( 1 - p _ { \mathrm { r e f } } \right) \left[ \frac { N } { N _ { \mathrm { r e f } } } e ^ { - \left( N - N _ { \mathrm { r e f } } \right) / N _ { 0 } } \right] ^ { - \beta } .\tag{39}
$$

Assume $N _ { \mathrm { r e f } } < N _ { 0 }$ , so that $p ( N )$ initially increases, reaches its maximum at $N = N _ { 0 }$ , and then decreases. Define $\beta _ { \mathrm { p e a k } }$ as the collaboration gain for which $p ( N _ { 0 } ) = p _ { \mathrm { c r i } }$ <sub>t</sub>. Population growth requires

$$
\beta > \beta _ { \mathrm { p e a k } } , \quad \beta _ { \mathrm { p e a k } } \equiv \frac { \ln [ ( 1 - p _ { \mathrm { r e f } } ) / ( 1 - p _ { \mathrm { c r i t } } ) ] } { \ln ( N _ { 0 } / N _ { \mathrm { r e f } } ) - ( N _ { 0 } - N _ { \mathrm { r e f } } ) / N _ { 0 } } .\tag{40}
$$

$\mathrm { I f } \beta \le \beta _ { \mathrm { p e a k } }$ , then $p ( N ) \leq p _ { \mathrm { c r i t } }$ for all $N ,$ so population growth cannot occur. Conversely, if $\beta > \beta _ { \mathrm { p e a k } }$ $p ( N )$ crosses $p _ { \mathrm { c r i t } }$ twice. As a result, populations between the two crossing points grow, while smaller and larger populations decline. For the particular choice of $C ( N )$ in Equation (37), these two thresholds have the closed form

$$
N _ { \mathrm { c r i t } } ^ { ( k ) } = - N _ { 0 } W _ { k } \left[ - \frac { N _ { \mathrm { r e f } } } { N _ { 0 } } e ^ { - N _ { \mathrm { r e f } } / N _ { 0 } } \left( \frac { 1 - p _ { \mathrm { r e f } } } { 1 - p _ { \mathrm { c r i t } } } \right) ^ { 1 / \beta } \right] , \qquad k \in \{ 0 , - 1 \} ,\tag{41}
$$

where $W _ { k }$ denotes branch k of the Lambert W function, defined by $W _ { k } ( z ) e ^ { W _ { k } ( z ) } = z$ . These two solutions give a lower and an upper threshold for $N _ { \mathrm { c r i t } }$ , shown in Figure 10. We note that the dependence on the Lambert W function arises from the particular chosen form of $C ( N )$ , but the qualitative possibility of lower and upper thresholds depends only on $C ( N )$ having a maximum.

![](images/e8c6c6517d3c421f27274bb0e2768045a577779be19f63780975d1525b13299c.jpg)

![](images/7288745979ac4748c6a70e2124f3dc15fa40b42b38c7388a0e20faa49380b89b.jpg)  
Figure 10: Unavoidable congestion creates lower and upper bounds for population growth. The pink curves show the congestion model with illustrative scale $N _ { 0 } = 1 5 0 0$ , with other parameters chosen as in Figure $7 ( p _ { \mathrm { r e f } } = 0 . 1 0 , p _ { \mathrm { c r i t } } = \mu / \kappa = 0 . 2 0$ , and $\beta = 0 . 0 3 5$ in panel a). (a) Per-unit growth rate is plotted against population. It is positive only between $N _ { \mathrm { c r i t } } ^ { ( 0 ) } \simeq 7 6 1$ and $N _ { \mathrm { c r i t . } } ^ { ( - 1 ) } \simeq 2 { , } 6 1 0$ . The diamond marks $N _ { \mathrm { r e f } } = 1 6 ,$ the vertical dotted line marks where effective work reaches a maximum at $N _ { 0 }$ , and the dashed portion is extrapolated beyond $N _ { \mathrm { r e f } } .$ . (b) Initial population is plotted against collaboration gain, $\beta .$ . The solid and dashed pink curves mark the lower and upper population thresholds; they meet at $N _ { 0 }$ when $\beta = \beta _ { \mathrm { p e a k } } \simeq 0 . \bar { 0 } 3 3 2$ . Populations between the curves grow, whereas populations outside them decline.

## C Examples of multi-agent scaling on related sub-tasks

There is currently little public evidence for how success rates scale with population size for multiagent group cyber benchmark attempts. For GPT 5.6 Sol, OpenAI reports results for 1, 4, and 16 agents on SEC-Bench Pro and BrowseComp under five reasoning-effort settings<sup>3</sup> [51]. SEC-Bench Pro tests whether agents can produce a working exploit for a known software vulnerability [52], while BrowseComp tests information retrieval for difficult web searches [53]. We note that these benchmarks do not test whether a population can establish additional active units – we use these results only for a consistency check between our model and the benchmark observations on how success probability and total token output change with increasing N.

We examine each of the reasoning efforts separately, since reasoning level can affect the baseline capability and resources available to each unit. In Figure 11(a–b), we fit the reported success rates to $p ( N )$ in (16) with $N _ { \mathrm { r e f } } = 1$ . This fit function takes the form

$$
p _ { e } ( N ) = 1 - \big [ 1 - p _ { e } ( 1 ) \big ] N ^ { - \beta _ { e } } ,\tag{42}
$$

where $N$ is the number of agents, e is the reasoning-effort setting, $p _ { e } ( N )$ is the success rate, and $\beta _ { e }$ is the fitted collaboration gain exponent. The reference success rate $p _ { e } ( 1 )$ fixes the starting point of each curve. Larger values of $\beta _ { e }$ correspond to a larger improvement from adding agents.

The fitted curves show that the reported results are not inconsistent with our model. However, each fit uses only three data points. Many other curves could fit these points and make very different predictions at larger populations. We therefore treat the fits as a consistency check.

Table 1: Fitted collaboration gain exponents. Each $\beta _ { e }$ is fitted separately to the reported success rates at $N = 1 , 4 , 1 6$ , with the observed one-agent success rate $p _ { e } ( 1 )$ fixing the start of the curve. Root mean squared error over the $N = 4$ and $N = 1 6$ observations was at most 1.67% for nine of the ten fits; the SEC-Bench Pro High fit had an error of 4.59%.
<table><tr><td>Reasoning effort e</td><td>SEC-Bench Pro  $\beta _ { e }$ </td><td>BrowseComp  $\beta _ { e }$ </td></tr><tr><td>Low</td><td>0.025</td><td>0.328</td></tr><tr><td>Medium</td><td>0.044</td><td>0.214</td></tr><tr><td>High</td><td>0.043</td><td>0.155</td></tr><tr><td>Xhigh</td><td>0.027</td><td>0.157</td></tr><tr><td>Max</td><td>0.069</td><td>0.113</td></tr></table>

The report also provides some data on compute usage by the agent groups, in the form of total output tokens produced. Denote the output tokens by $\bar { T } _ { e } ( N )$ for each reasoning level, e, and number of agents, $N . \ { \bar { T } } _ { e } ( N )$ is a proxy for the total computational effort expended by the agents, although it does not perfectly map to the effective work $C ( N )$ in our model, as output tokens may be duplicated or irrelevant.

We normalize the total output tokens by the single-agent output tokens $T _ { e } ( 1 )$ , then fit to a power law in the number of agents. The fit function takes the form

$$
\frac { T _ { e } ( N ) } { T _ { e } ( 1 ) } = N ^ { \alpha _ { T , e } } ,\tag{43}
$$

where $\alpha _ { T , e }$ is the fit exponent, describing how token output depends on agent count.

Across reasoning-effort settings, $\alpha _ { T , e }$ ranges from 0.49 to 0.60 for SEC-Bench Pro and from 0.66 to 0.90 for BrowseComp. Since (43) is normalized by the single-agent token output $T _ { e } ( 1 )$ , it may be reasonable to pool the normalized data across reasoning levels; this yields the pooled fits $\widehat { \alpha } _ { T } = 0 . 5 4$ and 0.72 for each benchmark, respectively (see Figure 11(c–d)).

This is a consistency check on the assumption that $C ( N )$ follows a power law in (15) and (29), although $T _ { e } ( N )$ is an imperfect proxy for the total computational effort expended by the agents and does not exactly map to the effective work $C ( N )$ in our model.

![](images/75b57b565878cb33c7d358b0312e97715321575ad21afdf8163700fd51d3acab.jpg)

![](images/8dec3128c05793142a54b43e5801a4115c593a85d48fe9e8aa6bf764ae767b54.jpg)

![](images/075afe8381be687fba7e35157b30ed8c270dadd68685f2152636fbed69a56318.jpg)

![](images/109c7bea0eb72ad8952653ba0ac9b83b4f06a2b53cf2d08c74c55d3c6749c237.jpg)  
Figure 11: Limited multi-agent benchmark results are consistent with population-dependent performance, but do not establish a scaling law. (a–b) Markers show the reported success rates for five reasoning-effort settings at one, four, and sixteen agents [51]. Dashed curves fit Equation (42) separately for each setting. With only three agent counts, these fits should not be extrapolated to larger populations. (c–d) Markers show the total output tokens $T _ { e } ( N )$ , normalized by the corresponding one-agent total. Dashed black lines show the pooled power-law fits, with $\widehat { \alpha } _ { T } = 0 . 5 4$ for SEC-Bench Pro and 0.72 for BrowseComp. The source reports neither trial counts nor standard errors, so we do not show uncertainty intervals.